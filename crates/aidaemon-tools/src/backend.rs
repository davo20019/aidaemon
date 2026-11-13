//! The workspace execution contract: one persistent agent-visible workspace
//! whose files and processes a tool reaches through an [`ExecutionBackend`].
//!
//! A backend *is* the workspace: it has a fixed root, an identity that keys
//! approvals, and file and process operations confined to it. Containment of
//! a single command is a different layer, the [`crate::Sandbox`].

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use tokio::process::{Child, ChildStderr, ChildStdout};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Local,
    Docker,
    Ssh,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Docker => "docker",
            Self::Ssh => "ssh",
        }
    }
}

/// An agent-visible path. Callers must not reinterpret this as a daemon-host
/// `PathBuf`; only an execution backend may do that.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendPath(String);

impl BackendPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    /// Whether `path` names its location on its own: absolute, `~` or
    /// `~/…` (the backend's home). Anything else, `~user` included, is
    /// relative to a working directory, as [`normalize_lexically`] reads it.
    pub fn is_anchored(path: &str) -> bool {
        path == "~" || path.starts_with("~/") || Path::new(path).is_absolute()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn display(&self) -> &str {
        &self.0
    }

    pub fn join(&self, child: impl AsRef<str>) -> Self {
        let child = child.as_ref();
        if child.is_empty() {
            return self.clone();
        }
        if self.0.starts_with('/') {
            if child.starts_with('/') {
                return Self(child.to_string());
            }
            return Self(format!(
                "{}/{}",
                self.0.trim_end_matches('/'),
                child.trim_start_matches('/')
            ));
        }
        let mut path = PathBuf::from(&self.0);
        path.push(child);
        Self(path.to_string_lossy().into_owned())
    }

    pub fn parent(&self) -> Option<Self> {
        if self.0.starts_with('/') {
            let trimmed = self.0.trim_end_matches('/');
            if trimmed.is_empty() {
                return None;
            }
            let (parent, _) = trimmed.rsplit_once('/')?;
            return Some(Self(if parent.is_empty() {
                "/".to_string()
            } else {
                parent.to_string()
            }));
        }
        Path::new(&self.0)
            .parent()
            .map(|path| Self(path.to_string_lossy().into_owned()))
    }

    pub fn file_name(&self) -> Option<&str> {
        if self.0.starts_with('/') {
            return self
                .0
                .trim_end_matches('/')
                .rsplit('/')
                .next()
                .filter(|name| !name.is_empty());
        }
        Path::new(&self.0).file_name().and_then(OsStr::to_str)
    }

    pub fn extension(&self) -> Option<&str> {
        if self.0.starts_with('/') {
            return self
                .file_name()?
                .rsplit_once('.')
                .and_then(|(stem, extension)| {
                    (!stem.is_empty() && !extension.is_empty()).then_some(extension)
                });
        }
        Path::new(&self.0).extension().and_then(OsStr::to_str)
    }

    pub fn with_extension(&self, extension: &str) -> Self {
        if self.0.starts_with('/') {
            let file_name = self.file_name().unwrap_or_default();
            let stem = file_name
                .rsplit_once('.')
                .map_or(file_name, |(stem, _)| stem);
            let replacement = if extension.is_empty() {
                stem.to_string()
            } else {
                format!("{stem}.{extension}")
            };
            return if let Some(parent) = self.parent() {
                parent.join(&replacement)
            } else {
                Self(replacement)
            };
        }
        let mut path = PathBuf::from(&self.0);
        path.set_extension(extension);
        Self(path.to_string_lossy().into_owned())
    }
}

impl std::fmt::Display for BackendPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFileType {
    File,
    Directory,
    Symlink,
    Other,
}

#[derive(Debug, Clone)]
pub struct BackendMetadata {
    pub file_type: BackendFileType,
    pub len: u64,
    pub modified: Option<SystemTime>,
}

impl BackendMetadata {
    pub fn is_file(&self) -> bool {
        self.file_type == BackendFileType::File
    }

    pub fn is_dir(&self) -> bool {
        self.file_type == BackendFileType::Directory
    }
}

#[derive(Debug, Clone)]
pub struct BackendDirEntry {
    pub path: BackendPath,
    pub metadata: BackendMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    Overwrite,
    Append,
}

#[derive(Debug, Clone)]
pub enum CommandSpec {
    Shell(String),
    Argv { program: String, args: Vec<String> },
}

impl CommandSpec {
    pub fn as_posix_shell(&self) -> String {
        match self {
            Self::Shell(command) => command.clone(),
            Self::Argv { program, args } => std::iter::once(program.as_str())
                .chain(args.iter().map(String::as_str))
                .map(shell_quote)
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    pub command: CommandSpec,
    pub cwd: Option<BackendPath>,
    pub env: BTreeMap<String, String>,
    pub env_remove: Vec<String>,
    pub stdin: Option<Vec<u8>>,
}

impl ExecutionRequest {
    pub fn shell(command: impl Into<String>) -> Self {
        Self {
            command: CommandSpec::Shell(command.into()),
            cwd: None,
            env: BTreeMap::new(),
            env_remove: Vec::new(),
            stdin: None,
        }
    }

    pub fn argv(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: CommandSpec::Argv {
                program: program.into(),
                args,
            },
            cwd: None,
            env: BTreeMap::new(),
            env_remove: Vec::new(),
            stdin: None,
        }
    }
}

#[derive(Debug)]
pub struct ExecutionOutput {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub duration_ms: u64,
    pub timed_out: bool,
}

impl ExecutionOutput {
    pub fn stdout_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    pub fn stderr_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

/// Opaque process identity. The numeric ID is only a user-facing handle; a
/// remote backend also carries a backend-internal token used for remote
/// termination.
#[derive(Debug, Clone)]
pub struct ProcessHandle {
    display_id: u32,
    local_pid: Option<u32>,
    remote_token: Option<String>,
}

impl ProcessHandle {
    /// Backend implementors only: tools receive handles from a backend and
    /// never build one.
    #[doc(hidden)]
    pub fn new(display_id: u32, local_pid: Option<u32>, remote_token: Option<String>) -> Self {
        Self {
            display_id,
            local_pid,
            remote_token,
        }
    }

    pub fn display_id(&self) -> u32 {
        self.display_id
    }

    pub fn local_pid(&self) -> Option<u32> {
        self.local_pid
    }

    /// Backend implementors only: the token a remote backend uses to
    /// terminate the process. It is internal to the backend that issued it,
    /// not a secret the type protects (it is readable here).
    #[doc(hidden)]
    pub fn remote_token(&self) -> Option<&str> {
        self.remote_token.as_deref()
    }
}

pub struct SpawnedProcess {
    handle: ProcessHandle,
    child: Child,
}

impl SpawnedProcess {
    pub fn new(handle: ProcessHandle, child: Child) -> Self {
        Self { handle, child }
    }

    /// Backend implementors only: attach the remote termination token once
    /// the backend has read it.
    #[doc(hidden)]
    pub fn set_remote_token(&mut self, token: String) {
        self.handle.remote_token = Some(token);
    }

    pub fn handle(&self) -> &ProcessHandle {
        &self.handle
    }

    pub fn take_stdout(&mut self) -> Option<ChildStdout> {
        self.child.stdout.take()
    }

    pub fn take_stderr(&mut self) -> Option<ChildStderr> {
        self.child.stderr.take()
    }

    pub fn into_child(self) -> Child {
        self.child
    }
}

#[async_trait]
pub trait ExecutionBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn id(&self) -> &str;
    fn approval_scope(&self) -> &str;
    fn workspace_root(&self) -> &BackendPath;
    fn home_hint(&self) -> &BackendPath;
    fn allows_outside_workspace(&self) -> bool;

    async fn home_dir(&self) -> anyhow::Result<BackendPath>;
    async fn resolve_path(&self, path: &str) -> anyhow::Result<BackendPath>;
    async fn canonicalize(&self, path: &BackendPath) -> anyhow::Result<BackendPath>;
    async fn metadata(&self, path: &BackendPath) -> anyhow::Result<BackendMetadata>;
    async fn read(&self, path: &BackendPath) -> anyhow::Result<Vec<u8>>;
    async fn write(
        &self,
        path: &BackendPath,
        content: &[u8],
        mode: WriteMode,
        create_parents: bool,
    ) -> anyhow::Result<()>;
    async fn create_dir_all(&self, path: &BackendPath) -> anyhow::Result<()>;
    async fn copy(&self, source: &BackendPath, destination: &BackendPath) -> anyhow::Result<()>;
    #[allow(dead_code)]
    async fn rename(&self, source: &BackendPath, destination: &BackendPath) -> anyhow::Result<()>;
    #[allow(dead_code)]
    async fn remove_file(&self, path: &BackendPath) -> anyhow::Result<()>;
    /// Remove a directory only if it is empty; never recursive.
    async fn remove_empty_dir(&self, path: &BackendPath) -> anyhow::Result<()>;
    async fn read_dir(&self, path: &BackendPath) -> anyhow::Result<Vec<BackendDirEntry>>;
    async fn spawn(&self, request: ExecutionRequest) -> anyhow::Result<SpawnedProcess>;
    async fn terminate(&self, handle: &ProcessHandle, grace: Duration) -> anyhow::Result<()>;

    async fn execute(
        &self,
        request: ExecutionRequest,
        timeout: Duration,
    ) -> anyhow::Result<ExecutionOutput> {
        execute_spawned(self, request, timeout).await
    }

    /// Copy a daemon-host file into the execution workspace. Local execution
    /// preserves the original path when it is permitted.
    async fn import_local_file(
        &self,
        local_path: &Path,
        destination: &BackendPath,
    ) -> anyhow::Result<BackendPath> {
        let bytes = tokio::fs::read(local_path).await?;
        self.write(destination, &bytes, WriteMode::Overwrite, true)
            .await?;
        Ok(destination.clone())
    }

    /// Copy an execution-side file to a daemon-host staging path.
    async fn export_local_file(
        &self,
        source: &BackendPath,
        local_path: &Path,
    ) -> anyhow::Result<()> {
        let bytes = self.read(source).await?;
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(local_path, bytes).await?;
        Ok(())
    }

    /// Resolve an executable in the same environment used for agent work.
    /// Bare names are checked against PATH first, then against conventional
    /// user-local and system prefixes. The returned path is concrete so callers
    /// do not have to rediscover it through a differently configured shell.
    async fn resolve_executable(&self, executable: &str) -> anyhow::Result<Option<BackendPath>> {
        let mut candidates = vec![executable.to_string()];
        if !executable.contains('/') && !executable.contains('\\') {
            let home = self.home_dir().await?;
            for relative in [
                ".local/bin",
                ".cargo/bin",
                ".npm-global/bin",
                ".bun/bin",
                ".deno/bin",
                "go/bin",
                "miniforge3/bin",
                "anaconda3/bin",
                "bin",
            ] {
                candidates.push(home.join(relative).join(executable).to_string());
            }
            for directory in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"] {
                candidates.push(format!("{directory}/{executable}"));
            }
        }
        candidates.dedup();

        for candidate in candidates {
            let request = ExecutionRequest::argv(
                "sh",
                vec![
                    "-c".to_string(),
                    "command -v -- \"$1\" 2>/dev/null".to_string(),
                    "sh".to_string(),
                    candidate,
                ],
            );
            let output = self.execute(request, Duration::from_secs(5)).await?;
            if output.exit_code == 0
                && let Some(resolved) = output
                    .stdout_lossy()
                    .lines()
                    .map(str::trim)
                    .find(|line| !line.is_empty())
                {
                    return Ok(Some(BackendPath::new(resolved.to_string())));
                }
        }
        Ok(None)
    }

    async fn executable_exists(&self, executable: &str) -> anyhow::Result<bool> {
        Ok(self.resolve_executable(executable).await?.is_some())
    }
}

async fn execute_spawned<B: ExecutionBackend + ?Sized>(
    backend: &B,
    request: ExecutionRequest,
    timeout: Duration,
) -> anyhow::Result<ExecutionOutput> {
    let started = Instant::now();
    let mut spawned = backend.spawn(request).await?;
    let handle = spawned.handle().clone();
    let stdout = spawned
        .take_stdout()
        .ok_or_else(|| anyhow::anyhow!("execution backend did not provide stdout"))?;
    let stderr = spawned
        .take_stderr()
        .ok_or_else(|| anyhow::anyhow!("execution backend did not provide stderr"))?;
    let mut child = spawned.into_child();

    let stdout_task = tokio::spawn(async move {
        let mut reader = stdout;
        let mut bytes = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    });
    let stderr_task = tokio::spawn(async move {
        let mut reader = stderr;
        let mut bytes = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    });

    let (status, timed_out) = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(status) => (Some(status?), false),
        Err(_) => {
            backend
                .terminate(&handle, Duration::from_secs(2))
                .await
                .ok();
            let status = tokio::time::timeout(Duration::from_secs(3), child.wait())
                .await
                .ok()
                .and_then(Result::ok);
            (status, true)
        }
    };

    let stdout = stdout_task
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default();
    let stderr = stderr_task
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default();

    Ok(ExecutionOutput {
        exit_code: status.and_then(|status| status.code()).unwrap_or(-1),
        stdout,
        stderr,
        duration_ms: started.elapsed().as_millis() as u64,
        timed_out,
    })
}

/// Quote a value for a POSIX shell command line.
pub fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

/// What lexical path normalization needs from a backend: its path flavor and
/// the two bases that give relative and `~` paths a meaning. A plain value, so
/// code that compares paths without touching the target (the run ledger) can
/// hold it without holding the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexicalPaths {
    kind: BackendKind,
    workspace_root: BackendPath,
    home: BackendPath,
}

impl LexicalPaths {
    pub fn of(backend: &dyn ExecutionBackend) -> Self {
        Self {
            kind: backend.kind(),
            workspace_root: backend.workspace_root().clone(),
            home: backend.home_hint().clone(),
        }
    }

    /// The daemon host's own paths, for a store opened outside the daemon's
    /// run (a one-shot CLI command) where no execution backend exists.
    pub fn daemon_host() -> Self {
        let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let canonical = std::fs::canonicalize(&workspace).unwrap_or_else(|_| workspace.clone());
        let home = dirs::home_dir().unwrap_or_else(|| canonical.clone());
        Self {
            kind: BackendKind::Local,
            workspace_root: BackendPath::new(canonical.to_string_lossy().into_owned()),
            home: BackendPath::new(home.to_string_lossy().into_owned()),
        }
    }

    pub fn normalize(&self, path: &str) -> anyhow::Result<BackendPath> {
        normalize_lexically(self.kind, &self.workspace_root, &self.home, path)
    }

    /// Paths on a POSIX target reached over a remote backend.
    pub fn posix(workspace_root: &str, home: &str) -> Self {
        Self {
            kind: BackendKind::Ssh,
            workspace_root: BackendPath::new(workspace_root),
            home: BackendPath::new(home),
        }
    }
}

pub fn normalize_lexically(
    kind: BackendKind,
    workspace_root: &BackendPath,
    home: &BackendPath,
    path: &str,
) -> anyhow::Result<BackendPath> {
    let expanded = if path == "~" {
        home.as_str().to_string()
    } else if let Some(rest) = path.strip_prefix("~/") {
        home.join(rest).to_string()
    } else if Path::new(path).is_absolute() {
        path.to_string()
    } else {
        workspace_root.join(path).to_string()
    };
    match kind {
        BackendKind::Local => Ok(BackendPath::new(
            normalize_local_lexically(Path::new(&expanded))?
                .to_string_lossy()
                .into_owned(),
        )),
        BackendKind::Docker | BackendKind::Ssh => {
            Ok(BackendPath::new(normalize_posix_absolute(&expanded)?))
        }
    }
}

pub fn normalize_posix_absolute(path: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        path.starts_with('/'),
        "Remote execution path must be absolute: {path}"
    );
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                anyhow::ensure!(
                    parts.pop().is_some(),
                    "Path traversal escapes filesystem root: {path}"
                );
            }
            other => parts.push(other),
        }
    }
    Ok(format!("/{}", parts.join("/")))
}

pub fn normalize_local_lexically(path: &Path) -> anyhow::Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                anyhow::ensure!(
                    normalized.pop(),
                    "Path traversal escapes filesystem root: {}",
                    path.display()
                );
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

/// A backend shared between tools.
pub type SharedExecutionBackend = std::sync::Arc<dyn ExecutionBackend>;

/// Synchronous lexical normalization for policy/scope code that cannot perform
/// backend I/O. Actual tool calls still run `resolve_path` and its confinement
/// checks before touching the target.
pub fn normalize_path_lexically(
    backend: &dyn ExecutionBackend,
    path: &str,
) -> anyhow::Result<BackendPath> {
    normalize_lexically(
        backend.kind(),
        backend.workspace_root(),
        backend.home_hint(),
        path,
    )
}

#[cfg(test)]
mod anchored_path_tests {
    use super::*;

    #[test]
    fn only_absolute_and_home_paths_are_anchored() {
        for anchored in ["/srv/app", "~", "~/notes"] {
            assert!(BackendPath::is_anchored(anchored), "{anchored}");
        }
        for relative in ["notes", "./notes", "~someone/notes", ""] {
            assert!(!BackendPath::is_anchored(relative), "{relative}");
        }
    }
}
