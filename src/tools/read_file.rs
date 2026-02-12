use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::fs_utils;

pub struct ReadFileTool;

const MAX_FILE_SIZE: u64 = 100 * 1024; // 100KB

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read file contents with line numbers. Supports line range selection."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "read_file",
            "description": "Read file contents with line numbers. Use this instead of terminal cat/head/tail. Supports reading specific line ranges for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (supports ~ expansion)"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-based, inclusive). Omit to start from beginning."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (1-based, inclusive). Omit to read to end."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

        let path = fs_utils::validate_path(path_str)?;

        if !path.exists() {
            anyhow::bail!("File not found: {}", path_str);
        }

        let metadata = tokio::fs::metadata(&path).await?;

        if metadata.is_dir() {
            anyhow::bail!("Path is a directory, not a file: {}", path_str);
        }

        let file_size = metadata.len();

        // Check for binary
        if fs_utils::is_binary_file(&path).await? {
            return Ok(format!(
                "Binary file: {}\nSize: {} bytes\nType: binary (cannot display contents)",
                path_str, file_size
            ));
        }

        if file_size > MAX_FILE_SIZE {
            anyhow::bail!(
                "File too large: {} bytes (max {}). Use start_line/end_line to read a range.",
                file_size,
                MAX_FILE_SIZE
            );
        }

        let content = tokio::fs::read_to_string(&path).await?;
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = args["start_line"]
            .as_u64()
            .map(|n| (n as usize).saturating_sub(1))
            .unwrap_or(0);

        let end = args["end_line"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(total_lines);

        let end = end.min(total_lines);

        if total_lines == 0 {
            return Ok(format!("File: {} (0 lines, empty)", path_str));
        }

        if start >= total_lines {
            anyhow::bail!(
                "start_line {} exceeds total lines {} in file",
                start + 1,
                total_lines
            );
        }

        let selected: Vec<&str> = lines[start..end].to_vec();
        let selected_content = selected.join("\n");
        let formatted = fs_utils::format_with_line_numbers(&selected_content, start);

        let header = if start > 0 || end < total_lines {
            format!(
                "File: {} (lines {}-{} of {})\n",
                path_str,
                start + 1,
                end,
                total_lines
            )
        } else {
            format!("File: {} ({} lines)\n", path_str, total_lines)
        };

        Ok(format!("{}{}", header, formatted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = ReadFileTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "read_file");
        assert!(schema["description"].as_str().unwrap().len() > 0);
        assert!(schema["parameters"]["properties"]["path"].is_object());
    }

    #[tokio::test]
    async fn test_read_file_basic() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "line one\nline two\nline three\n").unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool.call(&args).await.unwrap();
        assert!(result.contains("line one"));
        assert!(result.contains("line two"));
        assert!(result.contains("line three"));
        assert!(result.contains("3 lines"));
    }

    #[tokio::test]
    async fn test_read_file_line_range() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "a\nb\nc\nd\ne\n").unwrap();
        let args =
            json!({"path": f.path().to_str().unwrap(), "start_line": 2, "end_line": 4}).to_string();
        let result = ReadFileTool.call(&args).await.unwrap();
        assert!(result.contains("b"));
        assert!(result.contains("c"));
        assert!(result.contains("d"));
        assert!(!result.contains("| a"));
        assert!(!result.contains("| e"));
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let args = json!({"path": "/tmp/nonexistent_read_file_test_12345.txt"}).to_string();
        let result = ReadFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_read_file_binary() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0xFF, 0xD8, 0xFF, 0x00, 0x10, 0x00]).unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool.call(&args).await.unwrap();
        assert!(result.contains("Binary file"));
    }

    #[tokio::test]
    async fn test_read_file_directory() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("directory"));
    }

    #[tokio::test]
    async fn test_read_file_empty() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let args = json!({"path": f.path().to_str().unwrap()}).to_string();
        let result = ReadFileTool.call(&args).await.unwrap();
        assert!(result.contains("0 lines"));
    }
}
