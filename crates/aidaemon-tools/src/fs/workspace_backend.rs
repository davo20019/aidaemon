use std::time::Duration;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;

use crate::backend::{
    BackendDirEntry, BackendFileType, BackendKind, BackendMetadata, BackendPath, ExecutionBackend,
    ExecutionRequest, ProcessHandle, SpawnedProcess, WriteMode,
};
use crate::workspace::Workspace;

/// File access confined to a [`Workspace`]. Every operation re-resolves its
/// path, so a symlink cannot lead outside the root. It runs no processes;
/// commands go through [`crate::Shell`] and its sandbox.
#[derive(Debug, Clone)]
pub struct WorkspaceBackend {
    workspace: Workspace,
    root: BackendPath,
    id: String,
}

impl WorkspaceBackend {
    pub fn new(workspace: Workspace) -> Self {
        let root = workspace.root().display().to_string();
        Self {
            id: format!("workspace:{root}"),
            root: BackendPath::new(root),
            workspace,
        }
    }

    fn confine(&self, path: &BackendPath) -> anyhow::Result<std::path::PathBuf> {
        Ok(self.workspace.resolve(path.as_str())?)
    }
}

fn metadata_of(metadata: &std::fs::Metadata) -> BackendMetadata {
    let kind = metadata.file_type();
    BackendMetadata {
        file_type: if kind.is_symlink() {
            BackendFileType::Symlink
        } else if kind.is_dir() {
            BackendFileType::Directory
        } else if kind.is_file() {
            BackendFileType::File
        } else {
            BackendFileType::Other
        },
        len: metadata.len(),
        modified: metadata.modified().ok(),
    }
}

#[async_trait]
impl ExecutionBackend for WorkspaceBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Local
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn approval_scope(&self) -> &str {
        &self.id
    }

    fn workspace_root(&self) -> &BackendPath {
        &self.root
    }

    fn home_hint(&self) -> &BackendPath {
        &self.root
    }

    fn allows_outside_workspace(&self) -> bool {
        false
    }

    async fn home_dir(&self) -> anyhow::Result<BackendPath> {
        Ok(self.root.clone())
    }

    async fn resolve_path(&self, path: &str) -> anyhow::Result<BackendPath> {
        let resolved = self.workspace.resolve(path)?;
        Ok(BackendPath::new(resolved.display().to_string()))
    }

    async fn canonicalize(&self, path: &BackendPath) -> anyhow::Result<BackendPath> {
        let resolved = tokio::fs::canonicalize(self.confine(path)?).await?;
        Ok(BackendPath::new(resolved.display().to_string()))
    }

    async fn metadata(&self, path: &BackendPath) -> anyhow::Result<BackendMetadata> {
        Ok(metadata_of(
            &tokio::fs::metadata(self.confine(path)?).await?,
        ))
    }

    async fn read(&self, path: &BackendPath) -> anyhow::Result<Vec<u8>> {
        Ok(tokio::fs::read(self.confine(path)?).await?)
    }

    async fn write(
        &self,
        path: &BackendPath,
        content: &[u8],
        mode: WriteMode,
        create_parents: bool,
    ) -> anyhow::Result<()> {
        let target = self.confine(path)?;
        if create_parents {
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        match mode {
            WriteMode::Overwrite => tokio::fs::write(&target, content).await?,
            WriteMode::Append => {
                let mut file = tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&target)
                    .await?;
                file.write_all(content).await?;
                file.flush().await?;
            }
        }
        Ok(())
    }

    async fn create_dir_all(&self, path: &BackendPath) -> anyhow::Result<()> {
        Ok(tokio::fs::create_dir_all(self.confine(path)?).await?)
    }

    async fn copy(&self, source: &BackendPath, destination: &BackendPath) -> anyhow::Result<()> {
        tokio::fs::copy(self.confine(source)?, self.confine(destination)?).await?;
        Ok(())
    }

    async fn rename(&self, source: &BackendPath, destination: &BackendPath) -> anyhow::Result<()> {
        Ok(tokio::fs::rename(self.confine(source)?, self.confine(destination)?).await?)
    }

    async fn remove_file(&self, path: &BackendPath) -> anyhow::Result<()> {
        Ok(tokio::fs::remove_file(self.confine(path)?).await?)
    }

    async fn remove_empty_dir(&self, path: &BackendPath) -> anyhow::Result<()> {
        Ok(tokio::fs::remove_dir(self.confine(path)?).await?)
    }

    async fn read_dir(&self, path: &BackendPath) -> anyhow::Result<Vec<BackendDirEntry>> {
        let directory = self.confine(path)?;
        let mut reader = tokio::fs::read_dir(&directory).await?;
        let mut entries = Vec::new();
        while let Some(entry) = reader.next_entry().await? {
            entries.push(BackendDirEntry {
                path: BackendPath::new(entry.path().display().to_string()),
                metadata: metadata_of(&tokio::fs::symlink_metadata(entry.path()).await?),
            });
        }
        Ok(entries)
    }

    async fn spawn(&self, _request: ExecutionRequest) -> anyhow::Result<SpawnedProcess> {
        anyhow::bail!("a workspace backend runs no processes; use the shell tool")
    }

    async fn terminate(&self, _handle: &ProcessHandle, _grace: Duration) -> anyhow::Result<()> {
        anyhow::bail!("a workspace backend runs no processes")
    }
}
