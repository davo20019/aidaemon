use async_trait::async_trait;
use serde_json::{json, Value};

use crate::execution::{active_execution_backend, WriteMode};
use crate::traits::{
    Tool, ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolRole, ToolTargetHintKind,
};

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
            "description": "Find and replace text in a file. Use this instead of terminal sed/awk. Shows context around the change. Fails safely if the text isn't found or is ambiguous. On not-found/ambiguous text, read_file the same path and retry once before asking the user.",
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;

        // Parameter aliasing: models often use slightly different names
        let path_str = args["path"]
            .as_str()
            .or_else(|| args["file_path"].as_str())
            .or_else(|| args["file"].as_str())
            .or_else(|| args["filename"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
        let old_text = args["old_text"]
            .as_str()
            .or_else(|| args["old_string"].as_str())
            .or_else(|| args["search"].as_str())
            .or_else(|| args["find"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: old_text"))?;
        let new_text = args["new_text"]
            .as_str()
            .or_else(|| args["new_string"].as_str())
            .or_else(|| args["replace"].as_str())
            .or_else(|| args["replacement"].as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: new_text"))?;
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);

        let backend = active_execution_backend();
        let path = backend.resolve_path(path_str).await?;

        // Block sensitive paths (~/.ssh/*, ~/.aws/*, ~/.gnupg/*, *.env, etc.)
        // mirroring `write_file`. Even though `edit_file` only does
        // find-and-replace, it can still mutate credentials in place, so the
        // same blocklist applies.
        if fs_utils::is_sensitive_path(std::path::Path::new(path.as_str())) {
            anyhow::bail!("Cannot edit sensitive path: {}", path_str);
        }

        if backend.metadata(&path).await.is_err() {
            anyhow::bail!("File not found: {}", path_str);
        }

        let bytes = backend.read(&path).await?;
        if bytes.iter().take(8192).any(|byte| *byte == 0) {
            anyhow::bail!("Cannot edit binary file: {}", path_str);
        }

        let content = String::from_utf8(bytes)
            .map_err(|_| anyhow::anyhow!("Cannot edit non-UTF-8 file: {}", path_str))?;

        // Count occurrences using exact match first.
        let mut effective_old_text = old_text.to_string();
        let mut effective_new_text = new_text.to_string();
        let mut count = content.matches(&effective_old_text).count();

        // Self-recovery path: tolerate newline-style mismatch (LF vs CRLF) while
        // preserving strict exact matching otherwise.
        if count == 0 {
            let file_newline = detect_newline_style(&content);
            let normalized_old = normalize_newlines_for_style(old_text, file_newline);
            if normalized_old != old_text {
                let normalized_count = content.matches(&normalized_old).count();
                if normalized_count > 0 {
                    effective_old_text = normalized_old;
                    effective_new_text = normalize_newlines_for_style(new_text, file_newline);
                    count = normalized_count;
                }
            }
        }

        if count == 0 {
            anyhow::bail!("{}", build_not_found_message(path_str, &content, old_text));
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
            content.replace(&effective_old_text, &effective_new_text)
        } else {
            content.replacen(&effective_old_text, &effective_new_text, 1)
        };

        // Backup + atomic write
        fs_utils::backup_before_overwrite(&backend, &path).await;
        backend
            .write(&path, new_content.as_bytes(), WriteMode::Overwrite, false)
            .await?;

        // Show context around the change
        let replaced_count = if replace_all { count } else { 1 };
        let context = get_change_context(&new_content, &effective_new_text);
        let used_newline_recovery = effective_old_text != old_text;
        let diagnostics =
            fs_utils::post_write_diagnostics(std::path::Path::new(path.as_str())).await;

        Ok(format!(
            "Edited {}: replaced {} occurrence{}{}\n\n{}{}",
            path_str,
            replaced_count,
            if replaced_count > 1 { "s" } else { "" },
            if used_newline_recovery {
                " (newline-normalized match)"
            } else {
                ""
            },
            context,
            diagnostics
        ))
    }
}

impl crate::traits::ToolSemantics for EditFileTool {
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            // Not high-impact: find-and-replace with safe failure on ambiguity.
            // Dedicated file tools are intentionally available without terminal
            // approval. Sensitive paths are blocked before editing.
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

    /// A write of the target path (absolute; a relative path resolves where
    /// the execution backend puts it, which is not known here).
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        super::command_effects::file_tool_write_effects(arguments)
    }

    fn call_annotations_in(
        &self,
        arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        let semantics = self.call_semantics(arguments);
        let mut annotations =
            crate::traits::default_call_annotations(self.capabilities(), &semantics);
        // A write that can plant git configuration changes what later
        // "read-only" git commands execute: treat it as a configuration change.
        let plants_git_config = semantics
            .target_hints
            .iter()
            .any(|hint| super::command_effects::is_git_config_path(&hint.value));
        annotations.egress |= plants_git_config;
        annotations
    }
}

fn detect_newline_style(content: &str) -> &'static str {
    let crlf_count = content.matches("\r\n").count();
    let lf_total = content.matches('\n').count();
    let lf_only_count = lf_total.saturating_sub(crlf_count);
    if crlf_count > lf_only_count {
        "\r\n"
    } else {
        "\n"
    }
}

fn normalize_newlines_for_style(input: &str, newline: &str) -> String {
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    if newline == "\r\n" {
        normalized.replace('\n', "\r\n")
    } else {
        normalized
    }
}

fn truncate_with_ellipsis(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let mut out: String = input.chars().take(max_chars).collect();
    out.push_str("...");
    out
}

fn build_file_preview(content: &str, max_lines: usize, max_chars: usize) -> String {
    if content.is_empty() {
        return "(file is empty)".to_string();
    }

    let mut preview_lines = Vec::new();
    for (idx, line) in content.lines().take(max_lines).enumerate() {
        preview_lines.push(format!("{:>4} | {}", idx + 1, line));
    }

    let total_lines = content.lines().count();
    if total_lines > max_lines {
        preview_lines.push(format!("... ({} more lines)", total_lines - max_lines));
    }

    let joined = preview_lines.join("\n");
    truncate_with_ellipsis(&joined, max_chars)
}

fn build_not_found_message(path: &str, content: &str, old_text: &str) -> String {
    let first_non_empty_old = old_text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(old_text)
        .trim();
    let old_hint = if first_non_empty_old.is_empty() {
        "old_text appears empty or whitespace-only.".to_string()
    } else {
        format!(
            "old_text first non-empty line: `{}`",
            truncate_with_ellipsis(first_non_empty_old, 120)
        )
    };

    let preview = build_file_preview(content, 40, 3500);
    format!(
        "Text not found in {}. The old_text must match exactly (including whitespace and indentation).\n\
{}\n\n\
Self-recovery steps:\n\
1. Call read_file on the same path.\n\
2. Copy the exact block from the file output into old_text and retry edit_file.\n\
3. If the goal is replacing the whole file, use write_file instead of edit_file.\n\n\
File preview:\n{}",
        path, old_hint, preview
    )
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
    use crate::traits::ToolSemantics;

    #[test]
    fn git_config_targets_are_configuration_egress() {
        let tool = EditFileTool;
        let ann = |path: &str| {
            tool.call_annotations_in(
                &json!({"path": path}).to_string(),
                &crate::traits::TrustRoots::new(&[]),
            )
        };
        let config = ann("/tmp/aidaemon-test/repo/.git/config");
        assert!(config.egress && !config.read_only);
        assert!(ann("/tmp/aidaemon-test/repo/.gitattributes").egress);
        let plain = ann("/tmp/aidaemon-test/repo/src/lib.rs");
        assert!(!plain.egress && !plain.read_only);
    }
    use std::io::Write;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = EditFileTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "edit_file");
        assert!(!schema["description"].as_str().unwrap().is_empty());
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
        writeln!(f, "foo bar foo baz foo").unwrap();
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
        writeln!(f, "foo bar foo baz foo").unwrap();
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
        writeln!(f, "hello world").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "nonexistent",
            "new_text": "replacement"
        })
        .to_string();

        let result = EditFileTool.call(&args).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"));
        assert!(err.contains("Self-recovery steps"));
    }

    #[tokio::test]
    async fn test_edit_newline_normalization_recovery() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "alpha\r\nbeta\r\ngamma\r\n").unwrap();
        let args = json!({
            "path": f.path().to_str().unwrap(),
            "old_text": "beta\n",
            "new_text": "BETA\n"
        })
        .to_string();

        let result = EditFileTool.call(&args).await.unwrap();
        assert!(result.contains("replaced 1 occurrence"));
        assert!(result.contains("newline-normalized match"));

        let content = tokio::fs::read_to_string(f.path()).await.unwrap();
        assert!(content.contains("BETA\r\n"));
        assert!(!content.contains("beta\r\n"));
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

    /// Regression: edit_file must refuse to touch sensitive paths
    /// (~/.ssh/*, *.env, ~/.gnupg/*, etc.) the same way write_file does.
    /// Without this guard, the agent could rewrite SSH keys or env files
    /// in place via find-and-replace.
    #[tokio::test]
    async fn test_edit_file_blocks_sensitive_path() {
        // Use a path under HOME/.ssh which `is_sensitive_path` flags.
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let ssh_path = format!("{}/.ssh/edit_file_sensitivity_probe", home);

        let args = json!({
            "path": ssh_path,
            "old_text": "a",
            "new_text": "b"
        })
        .to_string();

        let result = EditFileTool.call(&args).await;
        assert!(result.is_err(), "expected error for sensitive path");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("sensitive"),
            "expected 'sensitive' in error, got: {err}"
        );
    }

    #[tokio::test]
    async fn edit_file_allows_unrelated_config_toml_and_refuses_the_daemons_own() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("config.toml");
        let own = dir.path().join("daemon-config.toml");
        std::fs::write(&project, "mode = \"a\"\n").unwrap();
        std::fs::write(&own, "mode = \"a\"\n").unwrap();
        let _guard = fs_utils::DaemonPathsGuard::set(vec![own.clone()]);
        let edit = |p: &std::path::Path| {
            json!({"path": p.to_str().unwrap(), "old_text": "a", "new_text": "b"}).to_string()
        };
        EditFileTool.call(&edit(&project)).await.unwrap();
        let err = EditFileTool
            .call(&edit(&own))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("sensitive"), "got: {err}");
        assert_eq!(std::fs::read_to_string(&own).unwrap(), "mode = \"a\"\n");
    }

    #[test]
    fn test_edit_file_capabilities_match_no_approval_tool_guidance() {
        let caps = EditFileTool.capabilities();
        assert!(
            !caps.needs_approval,
            "edit_file is documented as a dedicated file tool that does not require approval"
        );
        assert!(!caps.high_impact_write);
    }

    #[test]
    fn declares_write_of_the_absolute_path() {
        let dir = format!("/tmp/aidaemon-test/{}", uuid::Uuid::new_v4());
        let path = format!("{dir}/notes/../acme.md");
        for key in ["path", "file_path"] {
            let effects = EditFileTool.call_effects(&json!({key: path}).to_string());
            assert!(effects.complete, "{key}");
            assert_eq!(
                effects.keys,
                vec![crate::tools::command_effects::path_effect(
                    aidaemon_core::EffectClass::Write,
                    &format!("{dir}/acme.md"),
                    None
                )
                .unwrap()]
            );
        }
        // A relative path has no known working directory here.
        let relative = EditFileTool.call_effects(&json!({"path": "acme.md"}).to_string());
        assert_eq!(relative, aidaemon_core::CallEffects::unknown());
    }
}
