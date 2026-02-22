use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// A single resource file in a skill bundle. Category-agnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEntry {
    /// Relative path within the skill directory (e.g. "scripts/deploy.sh", "references/guide.md")
    pub path: String,
    /// Category inferred from parent directory (e.g. "script", "reference", "asset")
    /// or the dirname itself for non-standard directories
    pub category: String,
}

/// Trait for loading resource content from any backend.
/// Implementations handle the actual I/O — filesystem, archive, database, etc.
#[async_trait]
pub trait ResourceResolver: Send + Sync {
    /// Read the content of a resource file. Returns the file content as a string.
    async fn read_resource(&self, skill_name: &str, resource_path: &str) -> anyhow::Result<String>;

    /// Register (or refresh) a directory-based skill location for future reads.
    async fn register_skill_directory(&self, _skill_name: &str, _dir: &Path) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Filesystem-based resolver — reads resources from skill directories on disk.
pub struct FileSystemResolver {
    /// Map of skill_name → base directory path
    skill_dirs: RwLock<HashMap<String, PathBuf>>,
}

impl FileSystemResolver {
    pub fn new() -> Self {
        Self {
            skill_dirs: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register(&self, skill_name: &str, dir: PathBuf) {
        self.skill_dirs
            .write()
            .await
            .insert(skill_name.to_string(), dir);
    }
}

#[async_trait]
impl ResourceResolver for FileSystemResolver {
    async fn read_resource(&self, skill_name: &str, resource_path: &str) -> anyhow::Result<String> {
        let dirs = self.skill_dirs.read().await;
        let base = dirs
            .get(skill_name)
            .ok_or_else(|| anyhow::anyhow!("No directory registered for skill '{}'", skill_name))?;

        let full_path = base.join(resource_path);

        // Security: canonicalize and verify path stays within skill directory
        let canonical_base = base.canonicalize()?;
        let canonical_path = full_path
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("Cannot resolve resource '{}': {}", resource_path, e))?;
        if !canonical_path.starts_with(&canonical_base) {
            anyhow::bail!("Resource path traversal blocked");
        }

        let content = std::fs::read_to_string(&canonical_path)?;

        // Cap at 32K to protect context window
        if content.len() > 32_000 {
            Ok(format!(
                "{}\n\n[Truncated: {} total chars]",
                &content[..32_000],
                content.len()
            ))
        } else {
            Ok(content)
        }
    }

    async fn register_skill_directory(&self, skill_name: &str, dir: &Path) -> anyhow::Result<()> {
        self.register(skill_name, dir.to_path_buf()).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_filesystem_resolver_reads_file() {
        let dir = TempDir::new().unwrap();
        let refs_dir = dir.path().join("references");
        fs::create_dir(&refs_dir).unwrap();
        fs::write(refs_dir.join("guide.md"), "# Style Guide\nUse snake_case.").unwrap();

        let resolver = FileSystemResolver::new();
        resolver
            .register("test-skill", dir.path().to_path_buf())
            .await;

        let content = resolver
            .read_resource("test-skill", "references/guide.md")
            .await
            .unwrap();
        assert!(content.contains("snake_case"));
    }

    #[tokio::test]
    async fn test_filesystem_resolver_path_traversal() {
        let dir = TempDir::new().unwrap();
        let refs_dir = dir.path().join("references");
        fs::create_dir(&refs_dir).unwrap();
        fs::write(refs_dir.join("ok.md"), "safe").unwrap();

        let resolver = FileSystemResolver::new();
        resolver
            .register("test-skill", dir.path().to_path_buf())
            .await;

        let result = resolver
            .read_resource("test-skill", "../../../etc/passwd")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_filesystem_resolver_truncates_large_files() {
        let dir = TempDir::new().unwrap();
        let refs_dir = dir.path().join("data");
        fs::create_dir(&refs_dir).unwrap();
        // Create a file > 32K
        let big_content = "x".repeat(40_000);
        fs::write(refs_dir.join("big.txt"), &big_content).unwrap();

        let resolver = FileSystemResolver::new();
        resolver
            .register("test-skill", dir.path().to_path_buf())
            .await;

        let content = resolver
            .read_resource("test-skill", "data/big.txt")
            .await
            .unwrap();
        assert!(content.contains("[Truncated: 40000 total chars]"));
        assert!(content.len() < 40_000);
    }

    #[tokio::test]
    async fn test_filesystem_resolver_missing_skill() {
        let resolver = FileSystemResolver::new();
        let result = resolver
            .read_resource("nonexistent", "references/guide.md")
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No directory registered"));
    }
}
