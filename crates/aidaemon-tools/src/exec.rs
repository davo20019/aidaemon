//! Sandboxes for the `shell` tool.
//!
//! A sandbox decides *where and how contained* a command runs. It never
//! decides *whether* it may run: approval is a policy concern and belongs in
//! an [`aidaemon_core::Hook`].

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use aidaemon_core::CancellationToken;

use crate::util::HeadTail;

/// Where `terminal` commands execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TerminalConfinement {
    /// Run commands directly on the host as the daemon's user, with the
    /// host's environment, network, and credentials. The default: what a
    /// person at the shell gets, so deploys, package managers, and
    /// authenticated CLIs just work.
    #[default]
    Host,
    /// Run every command inside the OS sandbox (seatbelt on macOS; on
    /// Linux the Codex sandbox adapter when installed, else bubblewrap)
    /// with deny-by-default filesystem and network policy scoped to the
    /// task. Opt-in for advanced users who want blast-radius
    /// containment; expect tools that rely on `$HOME` credentials or the
    /// network to need explicit grants.
    Sandbox,
}

impl std::fmt::Display for TerminalConfinement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host => write!(f, "host"),
            Self::Sandbox => write!(f, "sandbox"),
        }
    }
}

/// One command to run.
#[derive(Debug, Clone)]
pub struct SandboxCommand {
    /// Shell command line, run with `sh -c`.
    pub command: String,
    /// Working directory (absolute, inside `workspace_root`).
    pub cwd: PathBuf,
    /// Canonical workspace root; sandboxes make only this writable.
    pub workspace_root: PathBuf,
    /// The complete environment for the command (nothing else is inherited
    /// by sandboxed or local children).
    pub env: Vec<(String, String)>,
    pub timeout: Duration,
    /// Bytes of combined output kept (head + tail).
    pub max_output_bytes: usize,
    /// Whether the command may use the network (sandboxes only).
    pub allow_network: bool,
    /// Host paths the command must not read, on top of the backend's own
    /// defaults ([`HOME_SECRET_PATHS`] for bubblewrap and seatbelt). Only
    /// backends whose [`Sandbox::enforces_deny_read`] is true honour it;
    /// the shell tool refuses to run a command with a deny list on any
    /// other. Paths that do not exist are skipped. Under bubblewrap a denied directory appears empty and a
    /// denied file reads as empty; under seatbelt reading either fails.
    pub deny_read: Vec<PathBuf>,
    /// Cancelled when the run is cancelled; the command is killed.
    pub cancel: CancellationToken,
}

impl SandboxCommand {
    /// A spec with defaults: 60 s timeout, 20 000 output bytes, no network.
    pub fn new(command: impl Into<String>, workspace_root: impl Into<PathBuf>) -> Self {
        let root = workspace_root.into();
        Self {
            command: command.into(),
            cwd: root.clone(),
            workspace_root: root,
            env: Vec::new(),
            timeout: Duration::from_secs(60),
            max_output_bytes: 20_000,
            allow_network: false,
            deny_read: Vec::new(),
            cancel: CancellationToken::new(),
        }
    }
}

/// What a command did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    /// `None` when killed by a signal or timeout.
    pub exit_code: Option<i32>,
    /// Combined stdout and stderr, interleaved in arrival order, capped to
    /// head + tail.
    pub output: String,
    pub output_bytes: usize,
    pub truncated: bool,
    pub timed_out: bool,
    pub cancelled: bool,
    pub duration: Duration,
}

/// The command could not be started (as opposed to starting and failing).
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("execution backend `{0}` is not available on this machine")]
    Unavailable(&'static str),
    #[error("failed to start command: {0}")]
    Spawn(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ExecError>;

/// Where commands run.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// Short name recorded in receipts, for example `local` or `bubblewrap`.
    fn name(&self) -> &'static str;
    /// Whether this backend can run commands here.
    fn is_available(&self) -> bool;
    /// Whether the backend confines writes to the workspace.
    fn is_contained(&self) -> bool;
    /// Whether the backend keeps a command from reading the paths in
    /// [`SandboxCommand::deny_read`]. Default `false`: a backend claims it
    /// only when it enforces it, and callers refuse rather than pretend.
    fn enforces_deny_read(&self) -> bool {
        false
    }
    /// Whether the backend cuts the network when
    /// [`SandboxCommand::allow_network`] is false. Default `false`, as for
    /// [`Sandbox::enforces_deny_read`].
    fn enforces_network_off(&self) -> bool {
        false
    }
    async fn run(&self, cmd: &SandboxCommand) -> Result<CommandOutput>;
}

/// Spawn `command`, collect combined output with a bound, enforce the
/// timeout and cancellation, and kill the process group on expiry.
async fn run_process(
    mut command: Command,
    spec: &SandboxCommand,
    clear_env: bool,
) -> Result<CommandOutput> {
    if clear_env {
        command.env_clear();
        command.envs(spec.env.iter().map(|(k, v)| (k, v)));
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);

    let started = Instant::now();
    let mut child = command.spawn()?;
    let buffer = Arc::new(Mutex::new(HeadTail::new(spec.max_output_bytes)));

    let mut pumps = Vec::new();
    if let Some(stdout) = child.stdout.take() {
        pumps.push(tokio::spawn(pump(stdout, buffer.clone())));
    }
    if let Some(stderr) = child.stderr.take() {
        pumps.push(tokio::spawn(pump(stderr, buffer.clone())));
    }

    let mut timed_out = false;
    let mut cancelled = false;
    let status = tokio::select! {
        status = child.wait() => Some(status?),
        _ = tokio::time::sleep(spec.timeout) => { timed_out = true; None }
        _ = spec.cancel.cancelled() => { cancelled = true; None }
    };
    if status.is_none() {
        kill_tree(&mut child);
        let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
    }
    // Background grandchildren may hold the pipes open; give the readers a
    // short grace period, then stop waiting.
    let grace = Duration::from_millis(if status.is_some() { 1000 } else { 200 });
    for pump in pumps {
        let abort = pump.abort_handle();
        if tokio::time::timeout(grace, pump).await.is_err() {
            abort.abort();
        }
    }

    let buf = buffer.lock().expect("output buffer poisoned");
    Ok(CommandOutput {
        exit_code: status.and_then(|s| s.code()),
        output: buf.render(),
        output_bytes: buf.total(),
        truncated: buf.truncated(),
        timed_out,
        cancelled,
        duration: started.elapsed(),
    })
}

async fn pump(mut reader: impl tokio::io::AsyncRead + Unpin, buffer: Arc<Mutex<HeadTail>>) {
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buffer
                .lock()
                .expect("output buffer poisoned")
                .push(&chunk[..n]),
        }
    }
}

fn kill_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // The child leads its own process group; kill the whole group so
        // pipelines and background jobs die too.
        // SAFETY: kill(2) has no memory-safety preconditions.
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
    let _ = child.start_kill();
}

fn probe(program: &str, args: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

// ------------------------------------------------------------ secret paths

/// Credential locations under the home directory that the contained
/// backends ([`BubblewrapSandbox`], [`SeatbeltSandbox`]) keep commands from
/// reading by default: SSH keys, cloud and cluster credentials, GnuPG,
/// registry logins, `.netrc` and git's stored credentials. Replace the
/// list per backend with `default_deny_read(..)` (an empty list turns the
/// defaults off).
pub const HOME_SECRET_PATHS: &[&str] = &[
    ".ssh",
    ".aws",
    ".gnupg",
    ".config/gcloud",
    ".kube",
    ".docker/config.json",
    ".netrc",
    ".git-credentials",
];

/// [`HOME_SECRET_PATHS`] under `home`.
pub fn home_secret_paths(home: &Path) -> Vec<PathBuf> {
    HOME_SECRET_PATHS.iter().map(|rel| home.join(rel)).collect()
}

/// [`HOME_SECRET_PATHS`] under `$HOME` (empty when `HOME` is unset): the
/// contained backends' default deny-read list.
pub fn default_deny_read() -> Vec<PathBuf> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(|home| home_secret_paths(Path::new(&home)))
        .unwrap_or_default()
}

/// The backend's defaults plus the command's own list. A default that
/// contains the workspace is left out, so a workspace under, say, `~/.ssh`
/// is not hidden from itself; a path the caller names is always kept.
fn deny_list(defaults: &[PathBuf], cmd: &SandboxCommand) -> Vec<PathBuf> {
    let root =
        std::fs::canonicalize(&cmd.workspace_root).unwrap_or_else(|_| cmd.workspace_root.clone());
    let mut out: Vec<PathBuf> = existing_canonical(defaults)
        .into_iter()
        .filter(|path| !root.starts_with(path))
        .collect();
    for path in existing_canonical(&cmd.deny_read) {
        if !out.contains(&path) {
            out.push(path);
        }
    }
    out
}

// -------------------------------------------------------------------- local

/// Runs commands directly on the host with `sh -c`.
///
/// **No containment.** The command can read and write anything the current
/// user can, and use the network. The environment is still reduced to the
/// allowlist and the cwd is still confined to the workspace, but a command
/// can `cd` anywhere. Use it only with trusted models or behind an approval
/// hook.
#[derive(Debug, Clone, Default)]
pub struct LocalSandbox;

#[async_trait]
impl Sandbox for LocalSandbox {
    fn name(&self) -> &'static str {
        "local"
    }

    fn is_available(&self) -> bool {
        cfg!(unix) && Path::new("/bin/sh").exists()
    }

    fn is_contained(&self) -> bool {
        false
    }

    async fn run(&self, cmd: &SandboxCommand) -> Result<CommandOutput> {
        let mut command = Command::new("sh");
        command.arg("-c").arg(&cmd.command).current_dir(&cmd.cwd);
        run_process(command, cmd, true).await
    }
}

// --------------------------------------------------------------- bubblewrap

/// Linux sandbox using `bwrap`: the host filesystem is mounted read-only,
/// the workspace read-write, `/tmp` is a private tmpfs, PID and (unless
/// allowed) network namespaces are unshared, and the sandbox dies with its
/// parent. The credential locations in [`HOME_SECRET_PATHS`] are hidden by
/// default.
#[derive(Debug, Clone)]
pub struct BubblewrapSandbox {
    program: String,
    default_deny_read: Vec<PathBuf>,
}

impl Default for BubblewrapSandbox {
    fn default() -> Self {
        Self {
            program: "bwrap".into(),
            default_deny_read: default_deny_read(),
        }
    }
}

impl BubblewrapSandbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the paths hidden from every command (default:
    /// [`default_deny_read`]). An empty list hides nothing beyond each
    /// command's own [`SandboxCommand::deny_read`].
    pub fn default_deny_read(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        self.default_deny_read = paths.into_iter().collect();
        self
    }

    /// The `bwrap` argument list for a spec (before `sh -c <command>`).
    pub fn args(&self, cmd: &SandboxCommand) -> Vec<String> {
        let ws = cmd.workspace_root.display().to_string();
        let mut args: Vec<String> = [
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        // After the tmpfs so a workspace under /tmp stays visible.
        args.extend(["--bind".into(), ws.clone(), ws]);
        // After the workspace so a denied path inside it is hidden too.
        for path in deny_list(&self.default_deny_read, cmd) {
            let path = path.display().to_string();
            if Path::new(&path).is_dir() {
                args.extend(["--tmpfs".into(), path]);
            } else {
                args.extend(["--ro-bind".into(), "/dev/null".into(), path]);
            }
        }
        args.push("--unshare-pid".into());
        if !cmd.allow_network {
            args.push("--unshare-net".into());
        }
        args.extend([
            "--die-with-parent".into(),
            "--new-session".into(),
            "--chdir".into(),
            cmd.cwd.display().to_string(),
            "--".into(),
            "sh".into(),
            "-c".into(),
            cmd.command.clone(),
        ]);
        args
    }
}

#[async_trait]
impl Sandbox for BubblewrapSandbox {
    fn name(&self) -> &'static str {
        "bubblewrap"
    }

    fn is_available(&self) -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        cfg!(target_os = "linux")
            && *AVAILABLE.get_or_init(|| {
                // Probe a real sandbox with the same mounts and namespaces
                // `run` uses: bwrap may be installed while user namespaces
                // are disabled (or restricted by AppArmor).
                probe(
                    &self.program,
                    &[
                        "--ro-bind",
                        "/",
                        "/",
                        "--dev",
                        "/dev",
                        "--proc",
                        "/proc",
                        "--tmpfs",
                        "/tmp",
                        "--unshare-pid",
                        "--unshare-net",
                        "--die-with-parent",
                        "--new-session",
                        "true",
                    ],
                )
            })
    }

    fn is_contained(&self) -> bool {
        true
    }

    fn enforces_deny_read(&self) -> bool {
        true
    }

    fn enforces_network_off(&self) -> bool {
        true
    }

    async fn run(&self, cmd: &SandboxCommand) -> Result<CommandOutput> {
        if !self.is_available() {
            return Err(ExecError::Unavailable("bubblewrap"));
        }
        let mut command = Command::new(&self.program);
        command.args(self.args(cmd));
        run_process(command, cmd, true).await
    }
}

/// What a task-scoped bubblewrap sandbox may touch: deny by default, unlike
/// [`BubblewrapSandbox::args`] (which mounts the whole host read-only). The
/// command sees the base system directories (the dynamic loader, `/usr`,
/// `/bin`, `/lib`), a private `/dev`, `/proc` and `/tmp`, and exactly the
/// listed paths: `read_paths` read-only, `write_paths` read-write. With
/// `allow_network` it also sees the resolver and CA configuration it needs
/// to use the network. Listed paths that do not exist are skipped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BubblewrapScope {
    /// Working directory inside the sandbox (absolute).
    pub cwd: String,
    pub read_paths: Vec<String>,
    pub write_paths: Vec<String>,
    pub allow_network: bool,
}

/// Base system paths every scoped command can read.
const SCOPED_SYSTEM_READS: &[&str] = &[
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/alternatives",
];

/// Extra configuration a networked scoped command reads (name resolution,
/// TLS roots).
const SCOPED_NETWORK_READS: &[&str] = &[
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/gai.conf",
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/pki",
];

impl BubblewrapSandbox {
    /// The `bwrap` argument list for a task-scoped command (see
    /// [`BubblewrapScope`]), ending with `-- <command...>`. Listed paths are
    /// mounted shallowest first, so a narrower grant inside a wider one
    /// keeps its own access; a path listed both ways is writable.
    pub fn scoped_args(scope: &BubblewrapScope, command: &[String]) -> Vec<String> {
        let mut args: Vec<String> = Vec::new();
        let system = SCOPED_SYSTEM_READS.iter().chain(if scope.allow_network {
            SCOPED_NETWORK_READS.iter()
        } else {
            [].iter()
        });
        for path in system {
            args.extend(["--ro-bind-try".into(), path.to_string(), path.to_string()]);
        }
        args.extend(
            ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
                .iter()
                .map(|s| s.to_string()),
        );
        let mut mounts: Vec<(&str, bool)> = Vec::new();
        for path in &scope.write_paths {
            if !mounts.iter().any(|(p, _)| *p == path.as_str()) {
                mounts.push((path.as_str(), true));
            }
        }
        for path in &scope.read_paths {
            if !mounts.iter().any(|(p, _)| *p == path.as_str()) {
                mounts.push((path.as_str(), false));
            }
        }
        // Stable: equal depths keep writes before reads, and paths keep
        // their listed order.
        mounts.sort_by_key(|(path, _)| Path::new(path).components().count());
        for (path, writable) in mounts {
            let flag = if writable {
                "--bind-try"
            } else {
                "--ro-bind-try"
            };
            args.extend([flag.into(), path.to_string(), path.to_string()]);
        }
        args.push("--unshare-pid".into());
        if !scope.allow_network {
            args.push("--unshare-net".into());
        }
        args.extend([
            "--die-with-parent".into(),
            "--new-session".into(),
            "--chdir".into(),
            scope.cwd.clone(),
            "--".into(),
        ]);
        args.extend(command.iter().cloned());
        args
    }
}

// ----------------------------------------------------------------- seatbelt

/// macOS sandbox using `sandbox-exec` with a generated profile: everything
/// is readable except the credential locations in [`HOME_SECRET_PATHS`]
/// (by default) and each command's deny list, writes are allowed only under
/// the workspace (plus the `/dev` null/tty nodes), and the network is
/// denied unless allowed.
#[derive(Debug, Clone)]
pub struct SeatbeltSandbox {
    default_deny_read: Vec<PathBuf>,
}

impl Default for SeatbeltSandbox {
    fn default() -> Self {
        Self {
            default_deny_read: default_deny_read(),
        }
    }
}

/// The canonical form of each path that exists (sandboxes match real paths:
/// `/tmp` is `/private/tmp` on macOS, and bwrap cannot mount over a missing
/// path on a read-only root).
fn existing_canonical(paths: &[PathBuf]) -> Vec<PathBuf> {
    paths
        .iter()
        .filter_map(|p| std::fs::canonicalize(p).ok())
        .collect()
}

fn sbpl_string(path: &Path) -> String {
    let raw = path.display().to_string();
    format!("\"{}\"", raw.replace('\\', "\\\\").replace('"', "\\\""))
}

impl SeatbeltSandbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the paths no command may read (default:
    /// [`default_deny_read`]). An empty list denies nothing beyond each
    /// command's own [`SandboxCommand::deny_read`].
    pub fn default_deny_read(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        self.default_deny_read = paths.into_iter().collect();
        self
    }

    /// The generated sandbox profile (SBPL) for a spec.
    pub fn profile(&self, cmd: &SandboxCommand) -> String {
        let mut profile = String::from("(version 1)\n(allow default)\n(deny file-write*)\n");
        profile.push_str(&format!(
            "(allow file-write* (subpath {}) (literal \"/dev/null\") (literal \"/dev/tty\") \
             (literal \"/dev/zero\") (regex #\"^/dev/fd/\"))\n",
            sbpl_string(&cmd.workspace_root)
        ));
        if !cmd.allow_network {
            profile.push_str("(deny network*)\n");
        }
        for path in deny_list(&self.default_deny_read, cmd) {
            profile.push_str(&format!(
                "(deny file-read* file-write* (subpath {}))\n",
                sbpl_string(&path)
            ));
        }
        profile
    }

    pub fn args(&self, cmd: &SandboxCommand) -> Vec<String> {
        vec![
            "-p".into(),
            self.profile(cmd),
            "sh".into(),
            "-c".into(),
            cmd.command.clone(),
        ]
    }
}

#[async_trait]
impl Sandbox for SeatbeltSandbox {
    fn name(&self) -> &'static str {
        "seatbelt"
    }

    fn is_available(&self) -> bool {
        cfg!(target_os = "macos") && Path::new("/usr/bin/sandbox-exec").exists()
    }

    fn is_contained(&self) -> bool {
        true
    }

    fn enforces_deny_read(&self) -> bool {
        true
    }

    fn enforces_network_off(&self) -> bool {
        true
    }

    async fn run(&self, cmd: &SandboxCommand) -> Result<CommandOutput> {
        if !self.is_available() {
            return Err(ExecError::Unavailable("seatbelt"));
        }
        let mut command = Command::new("/usr/bin/sandbox-exec");
        command.args(self.args(cmd)).current_dir(&cmd.cwd);
        run_process(command, cmd, true).await
    }
}

// ------------------------------------------------------------------- docker

/// Runs each command in a fresh container: the workspace is mounted at
/// `/workspace`, `--network none` unless allowed, and the container is
/// removed afterwards (and killed on timeout). No other host path is
/// mounted, so host paths in [`SandboxCommand::deny_read`] are out of reach
/// already; denied paths inside the workspace are covered by an empty
/// mount. On Unix the container runs as
/// the calling user so files it creates stay editable.
///
/// The `docker` CLI keeps the host environment (it needs `DOCKER_HOST`,
/// config paths and so on); the command inside sees only the spec's env,
/// minus `PATH` and `HOME`, which come from the image.
#[derive(Debug, Clone)]
pub struct DockerSandbox {
    image: String,
    program: String,
}

static CONTAINER_SEQ: AtomicU64 = AtomicU64::new(0);

impl DockerSandbox {
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            program: "docker".into(),
        }
    }

    pub fn image(&self) -> &str {
        &self.image
    }

    /// The `docker` argument list for a spec and container name.
    pub fn args(&self, cmd: &SandboxCommand, name: &str) -> Vec<String> {
        let rel = cmd
            .cwd
            .strip_prefix(&cmd.workspace_root)
            .unwrap_or(Path::new(""));
        let workdir = Path::new("/workspace").join(rel);
        let mut args: Vec<String> = vec![
            "run".into(),
            "--rm".into(),
            "--init".into(),
            "--name".into(),
            name.into(),
        ];
        if !cmd.allow_network {
            args.extend(["--network".into(), "none".into()]);
        }
        #[cfg(unix)]
        {
            // SAFETY: getuid/getgid cannot fail and touch no memory.
            let (uid, gid) = unsafe { (libc::getuid(), libc::getgid()) };
            args.extend(["--user".into(), format!("{uid}:{gid}")]);
        }
        args.extend([
            "-v".into(),
            format!("{}:/workspace", cmd.workspace_root.display()),
            "-w".into(),
            workdir.display().to_string(),
        ]);
        let root = std::fs::canonicalize(&cmd.workspace_root)
            .unwrap_or_else(|_| cmd.workspace_root.clone());
        for path in existing_canonical(&cmd.deny_read) {
            let Ok(rel) = path.strip_prefix(&root) else {
                continue; // not mounted at all
            };
            let inside = Path::new("/workspace").join(rel).display().to_string();
            if path.is_dir() {
                args.extend(["--tmpfs".into(), inside]);
            } else {
                args.extend(["-v".into(), format!("/dev/null:{inside}:ro")]);
            }
        }
        for (key, value) in &cmd.env {
            if key != "PATH" && key != "HOME" {
                args.extend(["-e".into(), format!("{key}={value}")]);
            }
        }
        args.extend([
            self.image.clone(),
            "sh".into(),
            "-c".into(),
            cmd.command.clone(),
        ]);
        args
    }
}

#[async_trait]
impl Sandbox for DockerSandbox {
    fn name(&self) -> &'static str {
        "docker"
    }

    fn is_available(&self) -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| probe(&self.program, &["info", "--format", "{{.ServerVersion}}"]))
    }

    fn is_contained(&self) -> bool {
        true
    }

    fn enforces_deny_read(&self) -> bool {
        true
    }

    fn enforces_network_off(&self) -> bool {
        true
    }

    async fn run(&self, cmd: &SandboxCommand) -> Result<CommandOutput> {
        if !self.is_available() {
            return Err(ExecError::Unavailable("docker"));
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or_default();
        let name = format!(
            "aidaemon-{}-{}-{nanos}",
            std::process::id(),
            CONTAINER_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let mut command = Command::new(&self.program);
        command.args(self.args(cmd, &name));
        let out = run_process(command, cmd, false).await?;
        if out.timed_out || out.cancelled {
            let _ = Command::new(&self.program)
                .args(["kill", &name])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await;
        }
        Ok(out)
    }
}

/// The backends the default shell can choose between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxKind {
    /// [`BubblewrapSandbox`] (Linux).
    Bubblewrap,
    /// [`SeatbeltSandbox`] (macOS).
    Seatbelt,
    /// [`LocalSandbox`]: no containment.
    Local,
}

impl SandboxKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bubblewrap => "bubblewrap",
            Self::Seatbelt => "seatbelt",
            Self::Local => "local",
        }
    }

    /// A new backend of this kind.
    pub fn backend(self) -> Arc<dyn Sandbox> {
        match self {
            Self::Bubblewrap => Arc::new(BubblewrapSandbox::new()),
            Self::Seatbelt => Arc::new(SeatbeltSandbox::new()),
            Self::Local => Arc::new(LocalSandbox),
        }
    }
}

impl std::str::FromStr for SandboxKind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bubblewrap" | "bwrap" => Ok(Self::Bubblewrap),
            "seatbelt" | "sandbox-exec" => Ok(Self::Seatbelt),
            "local" | "host" | "none" => Ok(Self::Local),
            other => Err(format!(
                "unknown sandbox `{other}` (expected auto, bubblewrap, seatbelt or local)"
            )),
        }
    }
}

/// Environment variable that overrides the default shell's backend:
/// `auto` (or unset), `bubblewrap`, `seatbelt`, or `local` (no containment).
pub const SANDBOX_ENV: &str = "AIDAEMON_SHELL_SANDBOX";

/// The result of [`select_sandbox`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SandboxSelection {
    pub kind: SandboxKind,
    /// No contained backend was available, so commands run uncontained.
    /// Never set for an explicit choice.
    pub fell_back: bool,
}

/// Pick a backend. An explicit `preference` always wins (an unavailable one
/// then fails each command with [`ExecError::Unavailable`] rather than
/// running uncontained). Otherwise the first available of bubblewrap and
/// seatbelt, else [`SandboxKind::Local`] with `fell_back` set. `available`
/// is the probe, injectable for tests.
pub fn select_sandbox(
    preference: Option<SandboxKind>,
    available: impl Fn(SandboxKind) -> bool,
) -> SandboxSelection {
    if let Some(kind) = preference {
        return SandboxSelection {
            kind,
            fell_back: false,
        };
    }
    for kind in [SandboxKind::Bubblewrap, SandboxKind::Seatbelt] {
        if available(kind) {
            return SandboxSelection {
                kind,
                fell_back: false,
            };
        }
    }
    SandboxSelection {
        kind: SandboxKind::Local,
        fell_back: true,
    }
}

/// Read [`SANDBOX_ENV`]: `Ok(None)` for unset, empty or `auto`.
pub fn sandbox_preference_from_env() -> std::result::Result<Option<SandboxKind>, String> {
    match std::env::var(SANDBOX_ENV) {
        Ok(value) if value.trim().is_empty() || value.trim().eq_ignore_ascii_case("auto") => {
            Ok(None)
        }
        Ok(value) => value.parse().map(Some),
        Err(_) => Ok(None),
    }
}

fn is_kind_available(kind: SandboxKind) -> bool {
    match kind {
        SandboxKind::Bubblewrap => BubblewrapSandbox::new().is_available(),
        SandboxKind::Seatbelt => SeatbeltSandbox::new().is_available(),
        SandboxKind::Local => LocalSandbox.is_available(),
    }
}

/// The default shell's backend: the one named by [`SANDBOX_ENV`], else the
/// most contained one available here (bubblewrap on Linux, seatbelt on
/// macOS), else [`LocalSandbox`] (uncontained) with a one-time warning.
/// Docker is never picked automatically because it needs an image choice.
pub fn best_available_sandbox() -> Arc<dyn Sandbox> {
    static WARN_ENV: std::sync::Once = std::sync::Once::new();
    static WARN_FALLBACK: std::sync::Once = std::sync::Once::new();
    let preference = sandbox_preference_from_env().unwrap_or_else(|err| {
        WARN_ENV.call_once(|| tracing::warn!("{SANDBOX_ENV}: {err}; choosing automatically"));
        None
    });
    let selection = select_sandbox(preference, is_kind_available);
    if selection.fell_back {
        WARN_FALLBACK.call_once(|| {
            tracing::warn!(
                "no OS sandbox is available ({}); the shell tool runs commands \
                 uncontained on the host. Install bubblewrap on Linux, or set \
                 {SANDBOX_ENV}=local to choose this explicitly.",
                if cfg!(target_os = "linux") {
                    "bwrap is missing or user namespaces are disabled"
                } else {
                    "no supported sandbox on this platform"
                }
            )
        });
    }
    selection.kind.backend()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(root: &Path) -> SandboxCommand {
        let mut s = SandboxCommand::new("echo hi", root);
        s.cwd = root.join("sub");
        s.env = vec![
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("LANG".into(), "C".into()),
        ];
        s
    }

    #[test]
    fn bubblewrap_args_confine_and_drop_network() {
        let s = spec(Path::new("/tmp/ws"));
        let args = BubblewrapSandbox::new().args(&s);
        let joined = args.join(" ");
        assert!(joined.starts_with(
            "--ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind /tmp/ws /tmp/ws"
        ));
        assert!(args.contains(&"--unshare-net".to_string()));
        assert!(args.contains(&"--die-with-parent".to_string()));
        assert!(joined.ends_with("--chdir /tmp/ws/sub -- sh -c echo hi"));
        let mut open = s.clone();
        open.allow_network = true;
        assert!(
            !BubblewrapSandbox::new()
                .args(&open)
                .contains(&"--unshare-net".to_string())
        );
    }

    #[test]
    fn bubblewrap_scoped_args_mount_only_the_listed_paths() {
        let scope = BubblewrapScope {
            cwd: "/work/repo".into(),
            read_paths: vec!["/work/repo".into(), "/work/repo/src".into()],
            write_paths: vec!["/work/repo/target".into(), "/work/repo".into()],
            allow_network: false,
        };
        let command = vec!["/bin/sh".to_string(), "-c".into(), "make".into()];
        let args = BubblewrapSandbox::scoped_args(&scope, &command);
        let joined = args.join(" ");
        // Deny by default: never the whole host, and no network config.
        assert!(!joined.contains("--ro-bind / /"), "{joined}");
        assert!(!joined.contains("/etc/resolv.conf"), "{joined}");
        assert!(joined.contains("--ro-bind-try /usr /usr"));
        assert!(joined.contains("--dev /dev --proc /proc --tmpfs /tmp"));
        // Shallowest first; the repo is listed both ways and stays writable;
        // the narrower read-only `src` comes after it.
        let repo = joined.find("--bind-try /work/repo /work/repo").unwrap();
        let src = joined
            .find("--ro-bind-try /work/repo/src /work/repo/src")
            .unwrap();
        let target = joined
            .find("--bind-try /work/repo/target /work/repo/target")
            .unwrap();
        assert!(repo < src && repo < target, "{joined}");
        assert!(!joined.contains("--ro-bind-try /work/repo /work/repo"));
        // The private /tmp is mounted before any listed path under it.
        assert!(joined.find("--tmpfs /tmp").unwrap() < repo);
        assert!(args.contains(&"--unshare-net".to_string()));
        assert!(joined.ends_with("--chdir /work/repo -- /bin/sh -c make"));
    }

    #[test]
    fn bubblewrap_scoped_args_with_network_read_the_resolver() {
        let scope = BubblewrapScope {
            cwd: "/".into(),
            allow_network: true,
            ..BubblewrapScope::default()
        };
        let args = BubblewrapSandbox::scoped_args(&scope, &["true".to_string()]);
        let joined = args.join(" ");
        assert!(!args.contains(&"--unshare-net".to_string()));
        assert!(joined.contains("--ro-bind-try /etc/resolv.conf /etc/resolv.conf"));
        assert!(joined.contains("--ro-bind-try /etc/ssl /etc/ssl"));
    }

    #[test]
    fn bubblewrap_args_hide_denied_paths_after_the_workspace() {
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        std::fs::create_dir(root.join("secrets")).unwrap();
        std::fs::write(root.join("token"), "t").unwrap();
        let mut s = SandboxCommand::new("true", &root);
        s.deny_read = vec![
            root.join("secrets"),
            root.join("token"),
            root.join("missing"),
        ];
        let joined = BubblewrapSandbox::new().args(&s).join(" ");
        let ws_bind = format!("--bind {0} {0}", root.display());
        let dir = format!("--tmpfs {}", root.join("secrets").display());
        let file = format!("--ro-bind /dev/null {}", root.join("token").display());
        let at = |needle: &str| {
            joined
                .find(needle)
                .unwrap_or_else(|| panic!("{needle}: {joined}"))
        };
        assert!(at(&ws_bind) < at(&dir) && at(&ws_bind) < at(&file));
        assert!(!joined.contains("missing"), "{joined}");
    }

    #[test]
    fn seatbelt_profile_denies_reads_of_denied_paths() {
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        let mut s = SandboxCommand::new("true", &root);
        s.deny_read = vec![root.clone(), root.join("missing")];
        let profile = SeatbeltSandbox::new().profile(&s);
        assert!(
            profile.contains(&format!(
                "(deny file-read* file-write* (subpath {}))",
                sbpl_string(&root)
            )),
            "{profile}"
        );
        assert!(!profile.contains("missing"));
    }

    #[test]
    fn home_secrets_are_denied_by_default_after_the_workspace() {
        let home = tempfile::tempdir().unwrap();
        let home = std::fs::canonicalize(home.path()).unwrap();
        std::fs::create_dir_all(home.join(".aws")).unwrap();
        std::fs::write(home.join(".netrc"), "machine x").unwrap();
        let ws = home.join("project");
        std::fs::create_dir(&ws).unwrap();
        let s = SandboxCommand::new("true", &ws);

        let seatbelt = SeatbeltSandbox::new().default_deny_read(home_secret_paths(&home));
        let profile = seatbelt.profile(&s);
        for denied in [".aws", ".netrc"] {
            assert!(
                profile.contains(&format!(
                    "(deny file-read* file-write* (subpath {}))",
                    sbpl_string(&home.join(denied))
                )),
                "{profile}"
            );
        }
        assert!(!profile.contains(".ssh"), "missing paths are skipped");
        let none = SeatbeltSandbox::new().default_deny_read(Vec::new());
        assert!(!none.profile(&s).contains(".aws"));

        let bwrap = BubblewrapSandbox::new().default_deny_read(home_secret_paths(&home));
        let joined = bwrap.args(&s).join(" ");
        let ws_bind = format!("--bind {0} {0}", ws.display());
        let aws = format!("--tmpfs {}", home.join(".aws").display());
        let netrc = format!("--ro-bind /dev/null {}", home.join(".netrc").display());
        let at = |needle: &str| {
            joined
                .find(needle)
                .unwrap_or_else(|| panic!("{needle}: {joined}"))
        };
        assert!(at(&ws_bind) < at(&aws) && at(&ws_bind) < at(&netrc));

        // A workspace inside a default location is not hidden from itself.
        let inside = home.join(".aws/project");
        std::fs::create_dir(&inside).unwrap();
        let s = SandboxCommand::new("true", &inside);
        let joined = bwrap.args(&s).join(" ");
        assert!(!joined.contains(&aws), "{joined}");
        assert!(
            !seatbelt
                .profile(&s)
                .contains(&sbpl_string(&home.join(".aws")))
        );

        // The real defaults cover the usual locations under $HOME.
        let defaults = default_deny_read();
        if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
            assert!(defaults.contains(&Path::new(&home).join(".ssh")));
            assert!(defaults.contains(&Path::new(&home).join(".aws")));
        }
    }

    #[test]
    fn backends_say_what_they_enforce() {
        assert!(!LocalSandbox.enforces_deny_read());
        assert!(!LocalSandbox.enforces_network_off());
        for backend in [
            Arc::new(BubblewrapSandbox::new()) as Arc<dyn Sandbox>,
            Arc::new(SeatbeltSandbox::new()),
            Arc::new(DockerSandbox::new("alpine:3")),
        ] {
            assert!(backend.enforces_deny_read(), "{}", backend.name());
            assert!(backend.enforces_network_off(), "{}", backend.name());
        }
    }

    #[test]
    fn explicit_choice_wins_even_when_unavailable() {
        for kind in [
            SandboxKind::Bubblewrap,
            SandboxKind::Seatbelt,
            SandboxKind::Local,
        ] {
            let chosen = select_sandbox(Some(kind), |_| false);
            assert_eq!(chosen.kind, kind);
            assert!(!chosen.fell_back);
        }
    }

    #[test]
    fn automatic_choice_prefers_contained_backends() {
        let only = |k: SandboxKind| move |c: SandboxKind| c == k;
        let bwrap = select_sandbox(None, only(SandboxKind::Bubblewrap));
        assert_eq!(bwrap.kind, SandboxKind::Bubblewrap);
        assert!(!bwrap.fell_back);
        let seatbelt = select_sandbox(None, only(SandboxKind::Seatbelt));
        assert_eq!(seatbelt.kind, SandboxKind::Seatbelt);
        assert!(!seatbelt.fell_back);
        assert_eq!(select_sandbox(None, |_| true).kind, SandboxKind::Bubblewrap);
        // No OS sandbox: uncontained, and flagged so the caller warns.
        let none = select_sandbox(None, |_| false);
        assert_eq!(none.kind, SandboxKind::Local);
        assert!(none.fell_back);
        let local_only = select_sandbox(None, only(SandboxKind::Local));
        assert_eq!(local_only.kind, SandboxKind::Local);
        assert!(local_only.fell_back);
    }

    #[test]
    fn sandbox_kind_parses_names_and_builds_backends() {
        assert_eq!("bwrap".parse(), Ok(SandboxKind::Bubblewrap));
        assert_eq!(" Bubblewrap ".parse(), Ok(SandboxKind::Bubblewrap));
        assert_eq!("seatbelt".parse(), Ok(SandboxKind::Seatbelt));
        assert_eq!("none".parse(), Ok(SandboxKind::Local));
        assert_eq!("host".parse(), Ok(SandboxKind::Local));
        assert!("docker".parse::<SandboxKind>().is_err());
        for kind in [
            SandboxKind::Bubblewrap,
            SandboxKind::Seatbelt,
            SandboxKind::Local,
        ] {
            assert_eq!(kind.backend().name(), kind.as_str());
        }
    }

    #[test]
    fn the_platform_default_matches_the_probe() {
        if std::env::var_os(SANDBOX_ENV).is_some() {
            println!("skipping: {SANDBOX_ENV} is set");
            return;
        }
        let expected = select_sandbox(None, is_kind_available).kind;
        assert_eq!(best_available_sandbox().name(), expected.as_str());
        if cfg!(target_os = "macos") && SeatbeltSandbox::new().is_available() {
            assert_eq!(expected, SandboxKind::Seatbelt);
        }
    }

    #[tokio::test]
    async fn seatbelt_blocks_reads_of_home_secrets_by_default() {
        if !SeatbeltSandbox::new().is_available() {
            println!("skipping: sandbox-exec is not available");
            return;
        }
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir(home.path().join(".aws")).unwrap();
        std::fs::write(home.path().join(".aws/credentials"), "synthetic-secret").unwrap();
        let backend = SeatbeltSandbox::new().default_deny_read(home_secret_paths(home.path()));
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        let cat = format!("cat {}/.aws/credentials", home.path().display());
        let mut s = SandboxCommand::new(cat.clone(), &root);
        s.env = vec![("PATH".into(), "/usr/bin:/bin".into())];
        let out = backend.run(&s).await.unwrap();
        assert_ne!(out.exit_code, Some(0), "{}", out.output);
        assert!(!out.output.contains("synthetic-secret"), "{}", out.output);
        // Readable without the defaults, so the denial is the sandbox's.
        let open = SeatbeltSandbox::new().default_deny_read(Vec::new());
        let out = open.run(&s).await.unwrap();
        assert!(out.output.contains("synthetic-secret"), "{}", out.output);
    }

    #[tokio::test]
    async fn seatbelt_blocks_reads_of_denied_paths() {
        let backend = SeatbeltSandbox::new();
        if !backend.is_available() {
            println!("skipping: sandbox-exec is not available");
            return;
        }
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        let secrets = tempfile::tempdir().unwrap();
        std::fs::write(secrets.path().join("key"), "synthetic-secret").unwrap();
        let mut s = SandboxCommand::new(format!("cat {}/key", secrets.path().display()), &root);
        s.env = vec![("PATH".into(), "/usr/bin:/bin".into())];
        s.deny_read = vec![secrets.path().to_path_buf()];
        let out = backend.run(&s).await.unwrap();
        assert_ne!(out.exit_code, Some(0), "{}", out.output);
        assert!(!out.output.contains("synthetic-secret"), "{}", out.output);
    }

    #[test]
    fn seatbelt_profile_limits_writes_and_network() {
        let s = spec(Path::new("/Users/alice/ws \"q\""));
        let profile = SeatbeltSandbox::new().profile(&s);
        assert!(profile.contains("(deny file-write*)"));
        assert!(
            profile.contains("(subpath \"/Users/alice/ws \\\"q\\\"\")"),
            "{profile}"
        );
        assert!(profile.contains("(deny network*)"));
        let mut open = s.clone();
        open.allow_network = true;
        assert!(!SeatbeltSandbox::new().profile(&open).contains("network"));
    }

    #[test]
    fn docker_args_mount_workspace_without_network() {
        let s = spec(Path::new("/srv/ws"));
        let args = DockerSandbox::new("alpine:3").args(&s, "c1");
        let joined = args.join(" ");
        assert!(joined.contains("--network none"));
        assert!(joined.contains("-v /srv/ws:/workspace -w /workspace/sub"));
        assert!(joined.contains("-e LANG=C"));
        assert!(!joined.contains("PATH="));
        assert!(joined.ends_with("alpine:3 sh -c echo hi"));
    }

    #[test]
    fn docker_args_cover_denied_paths_inside_the_workspace() {
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        std::fs::create_dir(root.join("private")).unwrap();
        std::fs::write(root.join("token"), "t").unwrap();
        let mut s = SandboxCommand::new("true", &root);
        s.deny_read = vec![root.join("private"), root.join("token"), "/etc".into()];
        let joined = DockerSandbox::new("alpine:3").args(&s, "c1").join(" ");
        assert!(joined.contains("--tmpfs /workspace/private"), "{joined}");
        assert!(
            joined.contains("-v /dev/null:/workspace/token:ro"),
            "{joined}"
        );
        assert!(
            !joined.contains("/etc"),
            "not mounted, nothing to cover: {joined}"
        );
    }

    #[tokio::test]
    async fn bubblewrap_blocks_writes_outside_workspace() {
        let backend = BubblewrapSandbox::new();
        if !backend.is_available() {
            println!("skipping: bubblewrap is not available");
            return;
        }
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        let outside = tempfile::tempdir_in(std::env::var("HOME").unwrap_or("/".into()))
            .or_else(|_| tempfile::tempdir())
            .unwrap();
        let mut s = SandboxCommand::new(
            format!(
                "echo ok > inside.txt; echo bad > {}/x.txt; echo done",
                outside.path().display()
            ),
            &root,
        );
        s.env = vec![("PATH".into(), "/usr/bin:/bin".into())];
        let out = backend.run(&s).await.unwrap();
        assert!(root.join("inside.txt").exists(), "{}", out.output);
        assert!(!outside.path().join("x.txt").exists());
    }

    #[tokio::test]
    async fn seatbelt_blocks_writes_outside_workspace() {
        let backend = SeatbeltSandbox::new();
        if !backend.is_available() {
            println!("skipping: sandbox-exec is not available");
            return;
        }
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        let outside = tempfile::tempdir().unwrap();
        let mut s = SandboxCommand::new(
            format!(
                "echo ok > inside.txt; echo bad > {}/x.txt",
                outside.path().display()
            ),
            &root,
        );
        s.env = vec![("PATH".into(), "/usr/bin:/bin".into())];
        backend.run(&s).await.unwrap();
        assert!(root.join("inside.txt").exists());
        assert!(!outside.path().join("x.txt").exists());
    }
}
