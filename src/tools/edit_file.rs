use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::fs_utils;

pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Find and replace text in a file"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "edit_file",
            "description": "Find and replace text in a file. Use this instead of terminal sed/awk. Shows context around the change. Fails safely if the text isn't found or is ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (supports ~ expansion)"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find and replace"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace with"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false, errors if multiple found)"
                    }
                },
                "required": ["path", "old_text", "new_text"],
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
        let old_text = args["old_text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: old_text"))?;
        let new_text = args["new_text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: new_text"))?;
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);

        let path = fs_utils::validate_path(path_str)?;

        if !path.exists() {
            anyhow::bail!("File not found: {}", path_str);
        }

        if fs_utils::is_binary_file(&path).await? {
            anyhow::bail!("Cannot edit binary file: {}", path_str);
        }

        let content = tokio::fs::read_to_string(&path).await?;

        // Count occurrences
        let count = content.matches(old_text).count();

        if count == 0 {
            anyhow::bail!(
                "Text not found in {}. The old_text must match exactly (including whitespace and indentation).",
                path_str
            );
        }

        if count > 1 && !replace_all {
            anyhow::bail!(
                "Found {} occurrences of the text in {}. Set replace_all=true to replace all, or provide more context to make old_text unique.",
                count,
                path_str
            );
        }

        // Perform replacement
        let new_content = if replace_all {
            content.replace(old_text, new_text)
        } else {
            content.replacen(old_text, new_text, 1)
        };

        // Backup + atomic write
        let backup = path.with_extension(format!(
            "{}.bak",
            path.extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default()
        ));
        let _ = tokio::fs::copy(&path, &backup).await;

        let tmp_path = path.with_extension("tmp_edit");
        tokio::fs::write(&tmp_path, &new_content).await?;
        tokio::fs::rename(&tmp_path, &path).await?;

        // Show context around the change
        let replaced_count = if replace_all { count } else { 1 };
        let context = get_change_context(&new_content, new_text);

        Ok(format!(
            "Edited {}: replaced {} occurrence{}\n\n{}",
            path_str,
            replaced_count,
            if replaced_count > 1 { "s" } else { "" },
            context
        ))
    }
}

/// Get a few lines of context around where the replacement was made.
fn get_change_context(content: &str, new_text: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let new_text_first_line = new_text.lines().next().unwrap_or(new_text);

    // Find the first line containing the new text
    if let Some(idx) = lines.iter().position(|l| l.contains(new_text_first_line)) {
        let start = idx.saturating_sub(2);
        let end = (idx + new_text.lines().count() + 2).min(lines.len());
        let context_lines: Vec<String> = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>4} | {}", start + i + 1, line))
            .collect();
        context_lines.join("\n")
    } else {
        String::from("(change context not available)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = EditFileTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "edit_file");
        assert!(schema["description"].as_str().unwrap().len() > 0);
        assert!(schema["parameters"]["properties"]["old_text"].is_object());
    }

    #[tokio::test]
    async fn test_edit_single_replace() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "fn main() {{\n    println!(\"hello\");\n}}\n").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "hello",
            "new_text": "world"
        })
        .to_string();

        let result = EditFileTool.call(&args).await.unwrap();
        assert!(result.contains("replaced 1 occurrence"));

        let content = tokio::fs::read_to_string(f.path()).await.unwrap();
        assert!(content.contains("world"));
        assert!(!content.contains("hello"));
    }

    #[tokio::test]
    async fn test_edit_multiple_without_flag() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "foo bar foo baz foo\n").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "foo",
            "new_text": "qux"
        })
        .to_string();

        let result = EditFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("3 occurrences"));
    }

    #[tokio::test]
    async fn test_edit_replace_all() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "foo bar foo baz foo\n").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "foo",
            "new_text": "qux",
            "replace_all": true
        })
        .to_string();

        let result = EditFileTool.call(&args).await.unwrap();
        assert!(result.contains("replaced 3 occurrences"));

        let content = tokio::fs::read_to_string(f.path()).await.unwrap();
        assert_eq!(content, "qux bar qux baz qux\n");
    }

    #[tokio::test]
    async fn test_edit_text_not_found() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "hello world\n").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "nonexistent",
            "new_text": "replacement"
        })
        .to_string();

        let result = EditFileTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_edit_file_not_found() {
        let args = json!({
            "path": "/tmp/nonexistent_edit_test_12345.txt",
            "old_text": "a",
            "new_text": "b"
        })
        .to_string();

        let result = EditFileTool.call(&args).await;
        assert!(result.is_err());
    }
}
