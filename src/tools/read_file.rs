use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::traits::{
    Tool, ToolCallSemantics, ToolCapabilities, ToolRole, ToolTargetHintKind, ToolVerificationMode,
};

use super::fs_utils;

pub struct ReadFileTool;

const MAX_FILE_SIZE: u64 = 100 * 1024; // 100KB

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read file contents with line numbers and basic metadata. Supports line ranges and tail selection."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "read_file",
            "description": "Read file contents with line numbers and basic metadata like size and modified time. Use this instead of terminal cat/head/tail. Supports reading specific line ranges or the last N lines for large files.",
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
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Read the last N lines of the file. Useful for large logs."
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

        ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent)
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

        let path = fs_utils::validate_path(path_str)?;

        if !path.exists() {
            anyhow::bail!("File not found: {}", path_str);
        }

        let metadata = tokio::fs::metadata(&path).await?;

        if metadata.is_dir() {
            anyhow::bail!("Path is a directory, not a file: {}", path_str);
        }

        let file_size = metadata.len();
        let modified = format_modified_rfc3339(&metadata);

        // Check for binary
        if fs_utils::is_binary_file(&path).await? {
            let mut out = format!("Binary file: {}\nSize: {} bytes\n", path_str, file_size);
            if let Some(modified) = &modified {
                out.push_str(&format!("Modified: {}\n", modified));
            }
            out.push_str("Type: binary (cannot display contents)");
            return Ok(out);
        }

        let start = args["start_line"]
            .as_u64()
            .map(|n| (n as usize).saturating_sub(1))
            .unwrap_or(0);

        let end = args["end_line"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(usize::MAX);

        let tail_lines = args["tail_lines"]
            .as_u64()
            .or_else(|| args["last_lines"].as_u64())
            .or_else(|| args["last_n_lines"].as_u64())
            .map(|n| n as usize);

        if matches!(tail_lines, Some(0)) {
            anyhow::bail!("tail_lines must be at least 1");
        }

        let uses_subset = start > 0 || end != usize::MAX || tail_lines.is_some();
        if file_size > MAX_FILE_SIZE && !uses_subset {
            anyhow::bail!(
                "File too large: {} bytes (max {}). Use start_line/end_line or tail_lines to read a subset.",
                file_size,
                MAX_FILE_SIZE
            );
        }

        let selection = if let Some(count) = tail_lines {
            ReadSelection::Tail { count }
        } else if uses_subset {
            ReadSelection::Range {
                start,
                end_exclusive: (end != usize::MAX).then_some(end),
            }
        } else {
            ReadSelection::Full
        };

        let selected = read_selected_lines(&path, selection).await?;
        let total_lines = selected.total_lines;

        if total_lines == 0 {
            let header =
                format_text_file_header(path_str, "0 lines, empty", file_size, modified.as_deref());
            return Ok(header.trim_end().to_string());
        }

        if selected.start_index >= total_lines {
            anyhow::bail!(
                "start_line {} exceeds total lines {} in file",
                selected.start_index + 1,
                total_lines
            );
        }

        let selected_content = selected.lines.join("\n");
        let formatted = fs_utils::format_with_line_numbers(&selected_content, selected.start_index);

        let header_summary = match selection {
            ReadSelection::Full => format!("{} lines", total_lines),
            ReadSelection::Range { .. } => format!(
                "lines {}-{} of {}",
                selected.start_index + 1,
                selected.end_display,
                total_lines
            ),
            ReadSelection::Tail { .. } => {
                format!("last {} lines of {}", selected.lines.len(), total_lines)
            }
        };
        let header =
            format_text_file_header(path_str, &header_summary, file_size, modified.as_deref());

        Ok(format!("{}{}", header, formatted))
    }
}

#[derive(Clone, Copy)]
enum ReadSelection {
    Full,
    Range {
        start: usize,
        end_exclusive: Option<usize>,
    },
    Tail {
        count: usize,
    },
}

struct SelectedLines {
    lines: Vec<String>,
    total_lines: usize,
    start_index: usize,
    end_display: usize,
}

async fn read_selected_lines(
    path: &Path,
    selection: ReadSelection,
) -> anyhow::Result<SelectedLines> {
    let file = tokio::fs::File::open(path).await?;
    let mut reader = BufReader::new(file).lines();

    match selection {
        ReadSelection::Full => {
            let mut lines = Vec::new();
            while let Some(line) = reader.next_line().await? {
                lines.push(line);
            }
            let total_lines = lines.len();
            Ok(SelectedLines {
                lines,
                total_lines,
                start_index: 0,
                end_display: total_lines,
            })
        }
        ReadSelection::Range {
            start,
            end_exclusive,
        } => {
            let mut lines = Vec::new();
            let mut total_lines: usize = 0;

            while let Some(line) = reader.next_line().await? {
                total_lines += 1;
                let zero_based_index = total_lines - 1;
                if zero_based_index >= start
                    && end_exclusive.is_none_or(|end| zero_based_index < end)
                {
                    lines.push(line);
                }
            }

            let end_display = end_exclusive.unwrap_or(total_lines).min(total_lines);
            Ok(SelectedLines {
                lines,
                total_lines,
                start_index: start,
                end_display,
            })
        }
        ReadSelection::Tail { count } => {
            let mut lines = VecDeque::with_capacity(count);
            let mut total_lines: usize = 0;

            while let Some(line) = reader.next_line().await? {
                total_lines += 1;
                if lines.len() == count {
                    lines.pop_front();
                }
                lines.push_back(line);
            }

            let lines: Vec<String> = lines.into_iter().collect();
            let start_index = total_lines.saturating_sub(lines.len());
            Ok(SelectedLines {
                lines,
                total_lines,
                start_index,
                end_display: total_lines,
            })
        }
    }
}

fn format_modified_rfc3339(metadata: &std::fs::Metadata) -> Option<String> {
    let modified = metadata.modified().ok()?;
    let modified_utc: chrono::DateTime<chrono::Utc> = modified.into();
    Some(modified_utc.to_rfc3339())
}

fn format_text_file_header(
    path: &str,
    line_summary: &str,
    file_size: u64,
    modified: Option<&str>,
) -> String {
    match modified {
        Some(modified) => format!(
            "File: {} ({}, {} bytes, modified {})\n",
            path, line_summary, file_size, modified
        ),
        None => format!("File: {} ({}, {} bytes)\n", path, line_summary, file_size),
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
        assert!(!schema["description"].as_str().unwrap().is_empty());
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
        assert!(result.contains("bytes"));
        assert!(result.contains("modified"));
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
        assert!(result.contains("lines 2-4 of 5"));
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
        assert!(result.contains("Size: 6 bytes"));
        assert!(result.contains("Modified:"));
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
        assert!(result.contains("0 bytes"));
    }

    #[tokio::test]
    async fn test_read_large_file_with_line_range() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for i in 1..=25_000 {
            writeln!(f, "line {}", i).unwrap();
        }
        assert!(f.as_file().metadata().unwrap().len() > MAX_FILE_SIZE);

        let args =
            json!({"path": f.path().to_str().unwrap(), "start_line": 24998, "end_line": 25000})
                .to_string();
        let result = ReadFileTool.call(&args).await.unwrap();

        assert!(result.contains("lines 24998-25000 of 25000"));
        assert!(result.contains("24998 | line 24998"));
        assert!(result.contains("25000 | line 25000"));
        assert!(!result.contains("24997 | line 24997"));
    }

    #[tokio::test]
    async fn test_read_large_file_with_tail_lines() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for i in 1..=25_000 {
            writeln!(f, "line {}", i).unwrap();
        }
        assert!(f.as_file().metadata().unwrap().len() > MAX_FILE_SIZE);

        let args = json!({"path": f.path().to_str().unwrap(), "tail_lines": 3}).to_string();
        let result = ReadFileTool.call(&args).await.unwrap();

        assert!(result.contains("last 3 lines of 25000"));
        assert!(result.contains("24998 | line 24998"));
        assert!(result.contains("25000 | line 25000"));
        assert!(!result.contains("24997 | line 24997"));
    }
}
