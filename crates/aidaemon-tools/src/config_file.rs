//! The host's config file, as the tools that edit it see it: where it is,
//! how a new text is checked before it is written, and the backups kept
//! beside it. The host owns what a valid config is ([`ConfigCheck`]); the
//! tools only read, change and save the TOML.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tracing::{info, warn};

/// The host's check of a whole config text before it is written: TOML
/// syntax, environment references and the host's own config structure.
pub trait ConfigCheck: Send + Sync {
    fn check(&self, content: &str) -> Result<(), String>;
}

/// Any `Fn(&str) -> Result<(), String>` is a check.
impl<F> ConfigCheck for F
where
    F: Fn(&str) -> Result<(), String> + Send + Sync,
{
    fn check(&self, content: &str) -> Result<(), String> {
        self(content)
    }
}

/// A check that accepts any valid TOML: for tests and hosts with no
/// config structure of their own.
pub fn toml_only(content: &str) -> Result<(), String> {
    content
        .parse::<toml::Table>()
        .map(|_| ())
        .map_err(|e| format!("Invalid TOML syntax: {}", e))
}

/// The config file a tool edits.
#[derive(Clone)]
pub struct ConfigFile {
    path: PathBuf,
    check: Arc<dyn ConfigCheck>,
}

impl std::fmt::Debug for ConfigFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigFile")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl ConfigFile {
    pub fn new(path: impl Into<PathBuf>, check: Arc<dyn ConfigCheck>) -> Self {
        Self {
            path: path.into(),
            check,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The host's check of a whole config text.
    pub fn validate(&self, content: &str) -> Result<(), String> {
        self.check.check(content)
    }

    pub async fn read(&self) -> std::io::Result<String> {
        tokio::fs::read_to_string(&self.path).await
    }

    pub async fn load_table(&self) -> anyhow::Result<toml::Table> {
        Ok(self.read().await?.parse()?)
    }

    /// Rotates the backups and copies the current file to `.bak`.
    pub async fn backup(&self) -> anyhow::Result<()> {
        rotate_backups(&self.path).await
    }

    /// Writes a text already checked, owner-only.
    pub async fn write_checked(&self, content: &str) -> anyhow::Result<()> {
        tokio::fs::write(&self.path, content).await?;
        set_owner_only_permissions(&self.path);
        Ok(())
    }

    /// Checks a text, backs the file up and writes it. A failed backup is
    /// logged, not fatal.
    pub async fn save(&self, content: &str) -> anyhow::Result<()> {
        if let Err(err) = self.validate(content) {
            anyhow::bail!("Refused to save invalid config: {}", err);
        }
        if let Err(err) = self.backup().await {
            warn!(error = %err, "Failed to create config backup before a config change");
        }
        self.write_checked(content).await
    }

    /// [`ConfigFile::save`] for a parsed table.
    pub async fn save_table(&self, doc: &toml::Table) -> anyhow::Result<()> {
        let rendered = toml::to_string_pretty(&toml::Value::Table(doc.clone()))?;
        self.save(&rendered).await
    }

    /// The backups, newest first.
    pub fn backups(&self) -> [PathBuf; 3] {
        backup_paths(&self.path)
    }
}

fn backup_paths(path: &Path) -> [PathBuf; 3] {
    [
        path.with_extension("toml.bak"),
        path.with_extension("toml.bak.1"),
        path.with_extension("toml.bak.2"),
    ]
}

/// Owner-only read/write (0600) on Unix, best effort.
#[cfg(unix)]
pub fn set_owner_only_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if !path.exists() {
        return;
    }
    if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
        warn!(path = %path.display(), "Failed to set 0600 permissions: {}", e);
    }
}

#[cfg(not(unix))]
pub fn set_owner_only_permissions(_path: &Path) {}

/// Rotates the 3-deep backup ring and copies the current file to `.bak`:
/// `.bak.1` -> `.bak.2`, `.bak` -> `.bak.1`, current -> `.bak`.
pub async fn rotate_backups(path: &Path) -> anyhow::Result<()> {
    let [bak, bak1, bak2] = backup_paths(path);
    if bak1.exists() {
        let _ = tokio::fs::rename(&bak1, &bak2).await;
    }
    if bak.exists() {
        let _ = tokio::fs::rename(&bak, &bak1).await;
    }
    tokio::fs::copy(path, &bak).await?;
    // Backups hold secrets too.
    set_owner_only_permissions(&bak);
    set_owner_only_permissions(&bak1);
    set_owner_only_permissions(&bak2);
    info!(backup = %bak.display(), "Config backup created (3-deep rotation)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn save_refuses_what_the_check_refuses_and_keeps_backups() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "a = 1\n").unwrap();
        let file = ConfigFile::new(
            &path,
            Arc::new(|content: &str| {
                toml_only(content)?;
                if content.contains("bad") {
                    return Err("bad key".to_string());
                }
                Ok(())
            }),
        );
        let err = file.save("bad = 1\n").await.unwrap_err();
        assert!(err.to_string().contains("bad key"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a = 1\n");

        file.save("a = 2\n").await.unwrap();
        file.save("a = 3\n").await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a = 3\n");
        let [bak, bak1, _] = file.backups();
        assert_eq!(std::fs::read_to_string(bak).unwrap(), "a = 2\n");
        assert_eq!(std::fs::read_to_string(bak1).unwrap(), "a = 1\n");
    }
}
