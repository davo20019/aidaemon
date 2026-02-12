use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::fs_utils;

pub struct WriteFileTool;

const MAX_CONTENT_SIZE: usize = 1024 * 1024; // 1MB

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write or create a file with given content"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "write_file",
            "description": "Write content to a file, creating it if it doesn't exist. Use this instead of terminal echo/cat for file creation. Creates a backup of existing files before overwriting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write to (supports ~ expansion)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist (default: false)"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: content"))?;
        let create_dirs = args["create_dirs"].as_bool().unwrap_or(false);

        if content.len() > MAX_CONTENT_SIZE {
            anyhow::bail!(
                "Content too large: {} bytes (max {})",
                content.len(),
                MAX_CONTENT_SIZE
            );
        }

        let path = fs_utils::validate_path(path_str)?;

        // Block sensitive paths
        if fs_utils::is_sensitive_path(&path) {
            anyhow::bail!("Cannot write to sensitive path: {}", path_str);
        }

        // Create parent dirs if requested
        if create_dirs {
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
        } else if let Some(parent) = path.parent() {
            if !parent.exists() {
                anyhow::bail!(
                    "Parent directory does not exist: {}. Set create_dirs=true to create it.",
                    parent.display()
                );
            }
        }

        // Backup existing file
        let existed = path.exists();
        let old_size = if existed {
            let meta = tokio::fs::metadata(&path).await?;
            let size = meta.len();
            // Create backup
            let backup = path.with_extension(format!(
                "{}.bak",
                path.extension()
                    .map(|e| e.to_string_lossy().to_string())
                    .unwrap_or_default()
            ));
            // Only keep one backup
            let _ = tokio::fs::copy(&path, &backup).await;
            Some(size)
        } else {
            None
        };

        // Atomic write: write to temp file then rename
        let tmp_path = path.with_extension("tmp_write");
        tokio::fs::write(&tmp_path, content).await?;
        tokio::fs::rename(&tmp_path, &path).await?;

        let new_size = content.len();
        let line_count = content.lines().count();

        let action = if existed { "Updated" } else { "Created" };
        let size_info = if let Some(old) = old_size {
            format!(" (was {} bytes, backup saved)", old)
        } else {
            String::new()
        };

        Ok(format!(
            "{} {}\n{} bytes, {} lines{}",
            action, path_str, new_size, line_count, size_info
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = WriteFileTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "write_file");
        assert!(schema["description"].as_str().unwrap().len() > 0);
        assert!(schema["parameters"]["properties"]["path"].is_object());
        assert!(schema["parameters"]["properties"]["content"].is_object());
    }

    #[tokio::test]
    async fn test_write_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("new_file.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "Hello, world!"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await.unwrap();
        assert!(result.contains("Created"));
        assert!(result.contains("13 bytes"));

        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_write_overwrite_with_backup() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("existing.txt");
        tokio::fs::write(&file_path, "old content").await.unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "new content"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await.unwrap();
        assert!(result.contains("Updated"));
        assert!(result.contains("backup saved"));

        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "new content");

        // Verify backup exists
        let backup = file_path.with_extension("txt.bak");
        let backup_content = tokio::fs::read_to_string(&backup).await.unwrap();
        assert_eq!(backup_content, "old content");
    }

    #[tokio::test]
    async fn test_write_sensitive_path_blocked() {
        let args = json!({
            "path": "/tmp/.ssh/test_key",
            "content": "secret"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sensitive"));
    }

    #[tokio::test]
    async fn test_write_create_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("a").join("b").join("file.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "nested",
            "create_dirs": true
        })
        .to_string();

        let result = WriteFileTool.call(&args).await.unwrap();
        assert!(result.contains("Created"));
    }

    #[tokio::test]
    async fn test_write_no_parent_dir() {
        let args = json!({
            "path": "/tmp/nonexistent_dir_12345/file.txt",
            "content": "hello"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Parent directory"));
    }
}
