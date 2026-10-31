//! Execution backends for the `shell` tool.
//!
//! A backend decides *where and how contained* a command runs. It never
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

/// One command to run.
#[derive(Debug, Clone)]
pub struct CommandSpec {
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
    /// Cancelled when the run is cancelled; the command is killed.
    pub cancel: CancellationToken,
}

impl CommandSpec {
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
pub trait ExecutionBackend: Send + Sync {
    /// Short name recorded in receipts, for example `local` or `bubblewrap`.
    fn name(&self) -> &'static str;
    /// Whether this backend can run commands here.
    fn is_available(&self) -> bool;
    /// Whether the backend confines writes to the workspace.
    fn is_contained(&self) -> bool;
    async fn run(&self, cmd: &CommandSpec) -> Result<CommandOutput>;
}

/// Spawn `command`, collect combined output with a bound, enforce the
/// timeout and cancellation, and kill the process group on expiry.
async fn run_process(
    mut command: Command,
    spec: &CommandSpec,
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

// -------------------------------------------------------------------- local

/// Runs commands directly on the host with `sh -c`.
///
/// **No containment.** The command can read and write anything the current
/// user can, and use the network. The environment is still reduced to the
/// allowlist and the cwd is still confined to the workspace, but a command
/// can `cd` anywhere. Use it only with trusted models or behind an approval
/// hook.
#[derive(Debug, Clone, Default)]
pub struct LocalBackend;

#[async_trait]
impl ExecutionBackend for LocalBackend {
    fn name(&self) -> &'static str {
        "local"
    }

    fn is_available(&self) -> bool {
        cfg!(unix) && Path::new("/bin/sh").exists()
    }

    fn is_contained(&self) -> bool {
        false
    }

    async fn run(&self, cmd: &CommandSpec) -> Result<CommandOutput> {
        let mut command = Command::new("sh");
        command.arg("-c").arg(&cmd.command).current_dir(&cmd.cwd);
        run_process(command, cmd, true).await
    }
}

// --------------------------------------------------------------- bubblewrap

/// Linux sandbox using `bwrap`: the host filesystem is mounted read-only,
/// the workspace read-write, `/tmp` is a private tmpfs, PID and (unless
/// allowed) network namespaces are unshared, and the sandbox dies with its
/// parent.
#[derive(Debug, Clone)]
pub struct BubblewrapBackend {
    program: String,
}

impl Default for BubblewrapBackend {
    fn default() -> Self {
        Self {
            program: "bwrap".into(),
        }
    }
}

impl BubblewrapBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// The `bwrap` argument list for a spec (before `sh -c <command>`).
    pub fn args(&self, cmd: &CommandSpec) -> Vec<String> {
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
impl ExecutionBackend for BubblewrapBackend {
    fn name(&self) -> &'static str {
        "bubblewrap"
    }

    fn is_available(&self) -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        cfg!(target_os = "linux")
            && *AVAILABLE.get_or_init(|| {
                // Probe a real sandbox: bwrap may be installed while user
                // namespaces are disabled.
                probe(
                    &self.program,
                    &[
                        "--ro-bind",
                        "/",
                        "/",
                        "--unshare-net",
                        "--die-with-parent",
                        "true",
                    ],
                )
            })
    }

    fn is_contained(&self) -> bool {
        true
    }

    async fn run(&self, cmd: &CommandSpec) -> Result<CommandOutput> {
        if !self.is_available() {
            return Err(ExecError::Unavailable("bubblewrap"));
        }
        let mut command = Command::new(&self.program);
        command.args(self.args(cmd));
        run_process(command, cmd, true).await
    }
}

// ----------------------------------------------------------------- seatbelt

/// macOS sandbox using `sandbox-exec` with a generated profile: everything
/// is readable, writes are allowed only under the workspace (plus the
/// `/dev` null/tty nodes), and the network is denied unless allowed.
#[derive(Debug, Clone, Default)]
pub struct SeatbeltBackend;

fn sbpl_string(path: &Path) -> String {
    let raw = path.display().to_string();
    format!("\"{}\"", raw.replace('\\', "\\\\").replace('"', "\\\""))
}

impl SeatbeltBackend {
    pub fn new() -> Self {
        Self
    }

    /// The generated sandbox profile (SBPL) for a spec.
    pub fn profile(&self, cmd: &CommandSpec) -> String {
        let mut profile = String::from("(version 1)\n(allow default)\n(deny file-write*)\n");
        profile.push_str(&format!(
            "(allow file-write* (subpath {}) (literal \"/dev/null\") (literal \"/dev/tty\") \
             (literal \"/dev/zero\") (regex #\"^/dev/fd/\"))\n",
            sbpl_string(&cmd.workspace_root)
        ));
        if !cmd.allow_network {
            profile.push_str("(deny network*)\n");
        }
        profile
    }

    pub fn args(&self, cmd: &CommandSpec) -> Vec<String> {
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
impl ExecutionBackend for SeatbeltBackend {
    fn name(&self) -> &'static str {
        "seatbelt"
    }

    fn is_available(&self) -> bool {
        cfg!(target_os = "macos") && Path::new("/usr/bin/sandbox-exec").exists()
    }

    fn is_contained(&self) -> bool {
        true
    }

    async fn run(&self, cmd: &CommandSpec) -> Result<CommandOutput> {
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
/// removed afterwards (and killed on timeout). On Unix the container runs as
/// the calling user so files it creates stay editable.
///
/// The `docker` CLI keeps the host environment (it needs `DOCKER_HOST`,
/// config paths and so on); the command inside sees only the spec's env,
/// minus `PATH` and `HOME`, which come from the image.
#[derive(Debug, Clone)]
pub struct DockerBackend {
    image: String,
    program: String,
}

static CONTAINER_SEQ: AtomicU64 = AtomicU64::new(0);

impl DockerBackend {
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
    pub fn args(&self, cmd: &CommandSpec, name: &str) -> Vec<String> {
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
impl ExecutionBackend for DockerBackend {
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

    async fn run(&self, cmd: &CommandSpec) -> Result<CommandOutput> {
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

/// The most contained backend available here: bubblewrap on Linux,
/// seatbelt on macOS, otherwise [`LocalBackend`] (uncontained). Docker is
/// never picked automatically because it needs an image choice.
pub fn best_available_backend() -> Arc<dyn ExecutionBackend> {
    let bwrap = BubblewrapBackend::new();
    if bwrap.is_available() {
        return Arc::new(bwrap);
    }
    let seatbelt = SeatbeltBackend::new();
    if seatbelt.is_available() {
        return Arc::new(seatbelt);
    }
    Arc::new(LocalBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(root: &Path) -> CommandSpec {
        let mut s = CommandSpec::new("echo hi", root);
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
        let args = BubblewrapBackend::new().args(&s);
        let joined = args.join(" ");
        assert!(joined.starts_with(
            "--ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --bind /tmp/ws /tmp/ws"
        ));
        assert!(args.contains(&"--unshare-net".to_string()));
        assert!(args.contains(&"--die-with-parent".to_string()));
        assert!(joined.ends_with("--chdir /tmp/ws/sub -- sh -c echo hi"));
        let mut open = s.clone();
        open.allow_network = true;
        assert!(!BubblewrapBackend::new()
            .args(&open)
            .contains(&"--unshare-net".to_string()));
    }

    #[test]
    fn seatbelt_profile_limits_writes_and_network() {
        let s = spec(Path::new("/Users/alice/ws \"q\""));
        let profile = SeatbeltBackend::new().profile(&s);
        assert!(profile.contains("(deny file-write*)"));
        assert!(
            profile.contains("(subpath \"/Users/alice/ws \\\"q\\\"\")"),
            "{profile}"
        );
        assert!(profile.contains("(deny network*)"));
        let mut open = s.clone();
        open.allow_network = true;
        assert!(!SeatbeltBackend::new().profile(&open).contains("network"));
    }

    #[test]
    fn docker_args_mount_workspace_without_network() {
        let s = spec(Path::new("/srv/ws"));
        let args = DockerBackend::new("alpine:3").args(&s, "c1");
        let joined = args.join(" ");
        assert!(joined.contains("--network none"));
        assert!(joined.contains("-v /srv/ws:/workspace -w /workspace/sub"));
        assert!(joined.contains("-e LANG=C"));
        assert!(!joined.contains("PATH="));
        assert!(joined.ends_with("alpine:3 sh -c echo hi"));
    }

    #[tokio::test]
    async fn bubblewrap_blocks_writes_outside_workspace() {
        let backend = BubblewrapBackend::new();
        if !backend.is_available() {
            println!("skipping: bubblewrap is not available");
            return;
        }
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        let outside = tempfile::tempdir_in(std::env::var("HOME").unwrap_or("/".into()))
            .or_else(|_| tempfile::tempdir())
            .unwrap();
        let mut s = CommandSpec::new(
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
        let backend = SeatbeltBackend::new();
        if !backend.is_available() {
            println!("skipping: sandbox-exec is not available");
            return;
        }
        let ws = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(ws.path()).unwrap();
        let outside = tempfile::tempdir().unwrap();
        let mut s = CommandSpec::new(
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
