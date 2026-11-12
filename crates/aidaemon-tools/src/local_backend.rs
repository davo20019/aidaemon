//! The host's own filesystem and processes as an [`ExecutionBackend`].

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::backend::{
    normalize_local_lexically, BackendDirEntry, BackendFileType, BackendKind, BackendMetadata,
    BackendPath, CommandSpec, ExecutionBackend, ExecutionRequest, ProcessHandle, SpawnedProcess,
    WriteMode,
};
use crate::process_control::{configure_command_for_process_group, send_sigkill, send_sigterm};

/// Build the executable search path used by local agent processes.
///
/// Service managers commonly start daemons with only the system directories in
/// `PATH`, even when the owner installed development tools in user-local
/// prefixes.  Keep one backend-owned search contract instead of teaching each
/// tool about individual executables. Explicit per-request PATH values and
/// removals still win at the spawn boundary.
pub fn local_execution_search_path(home: &BackendPath) -> String {
    let mut directories = Vec::<String>::new();
    let mut push = |value: String| {
        if !value.trim().is_empty() && !directories.iter().any(|existing| existing == &value) {
            directories.push(value);
        }
    };

    if let Some(path) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&path) {
            if directory.is_absolute() {
                push(directory.to_string_lossy().into_owned());
            }
        }
    }
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
        push(home.join(relative).to_string());
    }
    for directory in [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/local/sbin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ] {
        push(directory.to_string());
    }
    std::env::join_paths(directories)
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// The local backend: the host filesystem, confined to a workspace unless
/// told otherwise.
pub struct LocalBackend {
    id: String,
    approval_scope: String,
    workspace_root: BackendPath,
    canonical_workspace_root: PathBuf,
    home: BackendPath,
    allow_outside_workspace: bool,
}

impl LocalBackend {
    /// The current directory as workspace, with no confinement: for tests
    /// that construct a tool directly.
    pub fn current_unrestricted() -> Self {
        let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let canonical_workspace_root =
            std::fs::canonicalize(&workspace).unwrap_or_else(|_| workspace.clone());
        let workspace_root =
            BackendPath::new(canonical_workspace_root.to_string_lossy().into_owned());
        let home = dirs::home_dir().unwrap_or_else(|| canonical_workspace_root.clone());
        Self {
            id: format!("local:{}", workspace_root.as_str()),
            approval_scope: "local".to_string(),
            workspace_root,
            canonical_workspace_root,
            home: BackendPath::new(home.to_string_lossy().into_owned()),
            allow_outside_workspace: true,
        }
    }

    /// A backend rooted at `workspace_root` (`~` expanded; default: the
    /// current directory, created if missing). Paths outside it are refused
    /// unless `allow_outside_workspace`.
    pub async fn new(
        workspace_root: Option<&str>,
        allow_outside_workspace: bool,
    ) -> anyhow::Result<Self> {
        let cwd = std::env::current_dir()?;
        let workspace = workspace_root
            .map(expand_local_path)
            .transpose()?
            .unwrap_or(cwd);
        tokio::fs::create_dir_all(&workspace).await?;
        let canonical_workspace_root = tokio::fs::canonicalize(&workspace).await?;
        let workspace_root =
            BackendPath::new(canonical_workspace_root.to_string_lossy().into_owned());
        let home = dirs::home_dir().unwrap_or_else(|| canonical_workspace_root.clone());
        Ok(Self {
            id: format!("local:{}", workspace_root.as_str()),
            approval_scope: "local".to_string(),
            workspace_root,
            canonical_workspace_root,
            home: BackendPath::new(home.to_string_lossy().into_owned()),
            allow_outside_workspace,
        })
    }

    fn host_path(&self, path: &BackendPath) -> PathBuf {
        PathBuf::from(path.as_str())
    }

    async fn ensure_allowed(&self, path: &Path) -> anyhow::Result<()> {
        if self.allow_outside_workspace {
            return Ok(());
        }
        let canonical = canonicalize_existing_or_parent(path).await?;
        anyhow::ensure!(
            canonical.starts_with(&self.canonical_workspace_root),
            "Path is outside execution workspace {}: {}",
            self.workspace_root,
            path.display()
        );
        Ok(())
    }
}

#[async_trait]
impl ExecutionBackend for LocalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Local
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn approval_scope(&self) -> &str {
        &self.approval_scope
    }

    fn workspace_root(&self) -> &BackendPath {
        &self.workspace_root
    }

    fn home_hint(&self) -> &BackendPath {
        &self.home
    }

    fn allows_outside_workspace(&self) -> bool {
        self.allow_outside_workspace
    }

    async fn home_dir(&self) -> anyhow::Result<BackendPath> {
        Ok(self.home.clone())
    }

    async fn resolve_path(&self, path: &str) -> anyhow::Result<BackendPath> {
        let expanded = if path == "~" {
            self.home.as_str().to_string()
        } else if let Some(rest) = path.strip_prefix("~/") {
            self.home.join(rest).as_str().to_string()
        } else {
            path.to_string()
        };
        let candidate = PathBuf::from(&expanded);
        let joined = if candidate.is_absolute() {
            candidate
        } else {
            self.host_path(&self.workspace_root).join(candidate)
        };
        let normalized = normalize_local_lexically(&joined)?;
        self.ensure_allowed(&normalized).await?;
        Ok(BackendPath::new(normalized.to_string_lossy().into_owned()))
    }

    async fn canonicalize(&self, path: &BackendPath) -> anyhow::Result<BackendPath> {
        let host = self.host_path(path);
        self.ensure_allowed(&host).await?;
        let canonical = tokio::fs::canonicalize(host).await?;
        self.ensure_allowed(&canonical).await?;
        Ok(BackendPath::new(canonical.to_string_lossy().into_owned()))
    }

    async fn metadata(&self, path: &BackendPath) -> anyhow::Result<BackendMetadata> {
        let host = self.host_path(path);
        self.ensure_allowed(&host).await?;
        let metadata = tokio::fs::symlink_metadata(host).await?;
        Ok(metadata_from_std(metadata))
    }

    async fn read(&self, path: &BackendPath) -> anyhow::Result<Vec<u8>> {
        let host = self.host_path(path);
        self.ensure_allowed(&host).await?;
        Ok(tokio::fs::read(host).await?)
    }

    async fn write(
        &self,
        path: &BackendPath,
        content: &[u8],
        mode: WriteMode,
        create_parents: bool,
    ) -> anyhow::Result<()> {
        let host = self.host_path(path);
        self.ensure_allowed(&host).await?;
        if create_parents {
            if let Some(parent) = host.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        match mode {
            WriteMode::Append => {
                let mut file = tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(host)
                    .await?;
                file.write_all(content).await?;
                file.flush().await?;
            }
            WriteMode::Overwrite => {
                let temp = sibling_temp_path(&host, "write");
                tokio::fs::write(&temp, content).await?;
                tokio::fs::rename(temp, host).await?;
            }
        }
        Ok(())
    }

    async fn create_dir_all(&self, path: &BackendPath) -> anyhow::Result<()> {
        let host = self.host_path(path);
        self.ensure_allowed(&host).await?;
        tokio::fs::create_dir_all(host).await?;
        Ok(())
    }

    async fn copy(&self, source: &BackendPath, destination: &BackendPath) -> anyhow::Result<()> {
        let source = self.host_path(source);
        let destination = self.host_path(destination);
        self.ensure_allowed(&source).await?;
        self.ensure_allowed(&destination).await?;
        tokio::fs::copy(source, destination).await?;
        Ok(())
    }

    #[allow(dead_code)]
    async fn rename(&self, source: &BackendPath, destination: &BackendPath) -> anyhow::Result<()> {
        let source = self.host_path(source);
        let destination = self.host_path(destination);
        self.ensure_allowed(&source).await?;
        self.ensure_allowed(&destination).await?;
        tokio::fs::rename(source, destination).await?;
        Ok(())
    }

    #[allow(dead_code)]
    async fn remove_file(&self, path: &BackendPath) -> anyhow::Result<()> {
        let host = self.host_path(path);
        self.ensure_allowed(&host).await?;
        tokio::fs::remove_file(host).await?;
        Ok(())
    }

    async fn remove_empty_dir(&self, path: &BackendPath) -> anyhow::Result<()> {
        let host = self.host_path(path);
        self.ensure_allowed(&host).await?;
        tokio::fs::remove_dir(host).await?;
        Ok(())
    }

    async fn read_dir(&self, path: &BackendPath) -> anyhow::Result<Vec<BackendDirEntry>> {
        let host = self.host_path(path);
        self.ensure_allowed(&host).await?;
        let mut entries = tokio::fs::read_dir(host).await?;
        let mut output = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            output.push(BackendDirEntry {
                path: BackendPath::new(entry.path().to_string_lossy().into_owned()),
                metadata: metadata_from_std(entry.metadata().await?),
            });
        }
        Ok(output)
    }

    async fn spawn(&self, request: ExecutionRequest) -> anyhow::Result<SpawnedProcess> {
        let cwd = request.cwd.as_ref().unwrap_or(&self.workspace_root).clone();
        let cwd_host = self.host_path(&cwd);
        self.ensure_allowed(&cwd_host).await?;

        let explicit_path = request.env.contains_key("PATH");
        let removes_path = request.env_remove.iter().any(|name| name == "PATH");

        let mut command = match request.command {
            CommandSpec::Shell(shell) => {
                let mut command = Command::new("sh");
                command.arg("-c").arg(shell);
                command
            }
            CommandSpec::Argv { program, args } => {
                let mut command = Command::new(program);
                command.args(args);
                command
            }
        };
        command
            .current_dir(cwd_host)
            .envs(request.env)
            .stdin(if request.stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if !explicit_path && !removes_path {
            command.env("PATH", local_execution_search_path(&self.home));
        }
        for name in request.env_remove {
            command.env_remove(name);
        }
        configure_command_for_process_group(&mut command);
        let mut child = command.spawn()?;
        if let (Some(stdin), Some(mut pipe)) = (request.stdin, child.stdin.take()) {
            tokio::spawn(async move {
                let _ = pipe.write_all(&stdin).await;
            });
        }
        let pid = child.id().unwrap_or(0);
        Ok(SpawnedProcess::new(
            ProcessHandle::new(pid, Some(pid), None),
            child,
        ))
    }

    async fn terminate(&self, handle: &ProcessHandle, grace: Duration) -> anyhow::Result<()> {
        let Some(pid) = handle.local_pid() else {
            return Ok(());
        };
        if send_sigterm(pid) {
            tokio::time::sleep(grace).await;
        }
        send_sigkill(pid);
        Ok(())
    }

    async fn import_local_file(
        &self,
        local_path: &Path,
        destination: &BackendPath,
    ) -> anyhow::Result<BackendPath> {
        if !self.allow_outside_workspace {
            let bytes = tokio::fs::read(local_path).await?;
            self.write(destination, &bytes, WriteMode::Overwrite, true)
                .await?;
            return Ok(destination.clone());
        }
        let path = BackendPath::new(local_path.to_string_lossy().into_owned());
        self.ensure_allowed(local_path).await?;
        Ok(path)
    }

    async fn export_local_file(
        &self,
        source: &BackendPath,
        local_path: &Path,
    ) -> anyhow::Result<()> {
        let source = self.host_path(source);
        self.ensure_allowed(&source).await?;
        if source == local_path {
            return Ok(());
        }
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::copy(source, local_path).await?;
        Ok(())
    }
}

fn metadata_from_std(metadata: std::fs::Metadata) -> BackendMetadata {
    let file_type = metadata.file_type();
    BackendMetadata {
        file_type: if file_type.is_symlink() {
            BackendFileType::Symlink
        } else if file_type.is_file() {
            BackendFileType::File
        } else if file_type.is_dir() {
            BackendFileType::Directory
        } else {
            BackendFileType::Other
        },
        len: metadata.len(),
        modified: metadata.modified().ok(),
    }
}

async fn canonicalize_existing_or_parent(path: &Path) -> anyhow::Result<PathBuf> {
    let mut candidate = path.to_path_buf();
    let mut suffix = Vec::new();
    while !candidate.exists() {
        let Some(name) = candidate.file_name().map(OsStr::to_owned) else {
            break;
        };
        suffix.push(name);
        anyhow::ensure!(
            candidate.pop(),
            "Cannot resolve path parent: {}",
            path.display()
        );
    }
    let mut canonical = tokio::fs::canonicalize(candidate).await?;
    for part in suffix.iter().rev() {
        canonical.push(part);
    }
    Ok(canonical)
}

fn sibling_temp_path(path: &Path, purpose: &str) -> PathBuf {
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("aidaemon-file");
    path.with_file_name(format!(".{name}.aidaemon-{purpose}-{suffix}"))
}

/// Expand `~` in a host path.
pub fn expand_local_path(path: &str) -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(shellexpand::tilde(path).to_string()))
}
