use async_trait::async_trait;
use serde_json::{json, Value};

use crate::execution::{active_execution_backend, WriteMode};
use crate::traits::{
    Tool, ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolRole, ToolTargetHintKind,
};

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
            "description": "Write content to a file, creating it if it doesn't exist. Use this instead of terminal echo/cat for file creation. Creates a backup of existing files before overwriting. ALWAYS prefer write_file over terminal heredocs (cat > file << EOF) — heredocs trigger the approval flow.",
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
                        "description": "Create parent directories if they don't exist (default: true)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode. \"overwrite\" (default) replaces the file (a backup is kept). \"append\" adds to the end of the file without a backup — use it to build a large file in chunks across multiple calls instead of emitting everything in one call (which can exceed the output token limit)."
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
            needs_approval: false,
            idempotent: false,
            // Not high-impact: creates backups before overwriting, user-scoped
            // file system only, and dedicated file tools are intentionally
            // available without terminal approval. Sensitive paths are blocked
            // before writing.
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let path = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| {
                for key in ["path", "file_path", "file", "filename"] {
                    if let Some(path) = args.get(key).and_then(|value| value.as_str()) {
                        return Some(path.to_string());
                    }
                }
                None
            })
            .unwrap_or_default();

        ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE)
            .with_target_hint(ToolTargetHintKind::Path, path)
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        // Parameter aliasing: models often use "file_path" or "file" instead of "path"
        let path_str = args["path"]
            .as_str()
            .or_else(|| args["file_path"].as_str())
            .or_else(|| args["file"].as_str())
            .or_else(|| args["filename"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
        let content = args["content"]
            .as_str()
            .or_else(|| args["data"].as_str())
            .or_else(|| args["text"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: content"))?;
        let create_dirs = args["create_dirs"].as_bool().unwrap_or(true);

        if content.len() > MAX_CONTENT_SIZE {
            anyhow::bail!(
                "Content too large: {} bytes (max {})",
                content.len(),
                MAX_CONTENT_SIZE
            );
        }

        let backend = active_execution_backend();
        let path = backend.resolve_path(path_str).await?;

        // Block sensitive paths
        if fs_utils::is_sensitive_path(std::path::Path::new(path.as_str())) {
            anyhow::bail!("Cannot write to sensitive path: {}", path_str);
        }

        // Create parent dirs if requested
        if create_dirs {
            if let Some(parent) = path.parent() {
                backend.create_dir_all(&parent).await?;
            }
        } else if let Some(parent) = path.parent() {
            if !backend
                .metadata(&parent)
                .await
                .is_ok_and(|metadata| metadata.is_dir())
            {
                anyhow::bail!(
                    "Parent directory does not exist: {}. Set create_dirs=true to create it.",
                    parent
                );
            }
        }

        // Write mode: "overwrite" (default, replaces with backup) or "append"
        // (additive, no backup — lets the model build a large file in chunks
        // instead of one oversized generation).
        let mode = args["mode"].as_str().unwrap_or("overwrite");
        if mode != "overwrite" && mode != "append" {
            anyhow::bail!(
                "Invalid mode: \"{}\" (expected \"overwrite\" or \"append\")",
                mode
            );
        }

        if mode == "append" {
            let existing_size = backend
                .metadata(&path)
                .await
                .map(|metadata| metadata.len)
                .unwrap_or(0);
            backend
                .write(&path, content.as_bytes(), WriteMode::Append, create_dirs)
                .await?;

            let appended = content.len();
            let appended_lines = content.lines().count();
            let total = existing_size + appended as u64;
            return Ok(format!(
                "Appended {} bytes ({} lines) to {} (now {} bytes total). \
                 To add more, call write_file again with mode=\"append\". \
                 When the file is complete, deliver it with send_file instead of pasting it inline.",
                appended, appended_lines, path_str, total
            ));
        }

        // Backup existing file
        let existing_metadata = backend.metadata(&path).await.ok();
        let existed = existing_metadata.is_some();
        let old_size = if existed {
            let size = existing_metadata
                .as_ref()
                .map_or(0, |metadata| metadata.len);
            // Create backup
            let backup_extension = format!(
                "{}.bak",
                path.extension().map(str::to_string).unwrap_or_default()
            );
            let backup = path.with_extension(&backup_extension);
            // Only keep one backup
            let _ = backend.copy(&path, &backup).await;
            Some(size)
        } else {
            None
        };

        backend
            .write(&path, content.as_bytes(), WriteMode::Overwrite, create_dirs)
            .await?;

        let new_size = content.len();
        let line_count = content.lines().count();

        let action = if existed { "Updated" } else { "Created" };
        let size_info = if let Some(old) = old_size {
            format!(" (was {} bytes, backup saved)", old)
        } else {
            String::new()
        };

        let diagnostics =
            fs_utils::post_write_diagnostics(std::path::Path::new(path.as_str())).await;

        Ok(format!(
            "{} {}\n{} bytes, {} lines{}{}",
            action, path_str, new_size, line_count, size_info, diagnostics
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
        assert!(!schema["description"].as_str().unwrap().is_empty());
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
    async fn test_write_auto_creates_parent_dirs_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("auto_created").join("file.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "hello"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await.unwrap();
        assert!(result.contains("Created"));
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "hello");
    }

    #[tokio::test]
    async fn test_write_no_parent_dir_when_create_dirs_false() {
        let args = json!({
            "path": "/tmp/nonexistent_dir_12345/file.txt",
            "content": "hello",
            "create_dirs": false
        })
        .to_string();

        let result = WriteFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Parent directory"));
    }

    #[test]
    fn test_write_file_capabilities_match_no_approval_tool_guidance() {
        let caps = WriteFileTool.capabilities();
        assert!(
            !caps.needs_approval,
            "write_file is documented as a dedicated file tool that does not require approval"
        );
        assert!(!caps.high_impact_write);
    }

    #[tokio::test]
    async fn test_append_creates_new_file_with_hint() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("appended.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "chunk one\n",
            "mode": "append"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await.unwrap();
        assert!(result.contains("Appended"), "result was: {result}");
        // Continuation hint must tell the model how to add more.
        assert!(result.contains("mode=\"append\""), "result was: {result}");

        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "chunk one\n");
    }

    #[tokio::test]
    async fn test_append_preserves_existing_and_no_backup() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("doc.txt");
        tokio::fs::write(&file_path, "first\n").await.unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "second\n",
            "mode": "append"
        })
        .to_string();
        WriteFileTool.call(&args).await.unwrap();

        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "first\nsecond\n");

        // Append must NOT create a backup (it is additive).
        let backup = file_path.with_extension("txt.bak");
        assert!(!backup.exists(), "append must not create a backup");
    }

    #[tokio::test]
    async fn test_invalid_mode_errors_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("untouched.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "x",
            "mode": "garbage"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid mode"));
        assert!(!file_path.exists(), "invalid mode must not write the file");
    }

    #[tokio::test]
    async fn test_append_sensitive_path_blocked() {
        let args = json!({
            "path": "/tmp/.ssh/test_key",
            "content": "secret",
            "mode": "append"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sensitive"));
    }

    #[tokio::test]
    async fn test_explicit_overwrite_mode_matches_default() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("ow.txt");
        tokio::fs::write(&file_path, "old").await.unwrap();
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "new",
            "mode": "overwrite"
        })
        .to_string();

        let result = WriteFileTool.call(&args).await.unwrap();
        assert!(result.contains("Updated"));
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert_eq!(content, "new");
    }
}
