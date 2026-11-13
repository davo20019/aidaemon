//! `edit_file`: exact find and replace that changes nothing when the text is
//! missing or ambiguous.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use aidaemon_core::{AccessManifest, ToolAnnotations};

use super::{FilePolicy, SharedBackend, ToolText, path_arg, workspace_parts};
use crate::backend::WriteMode;
use crate::outcome::{OutcomeTool, core_tool};
use crate::semantics::{
    ToolCallContext, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolMutationEffects,
    ToolRole, ToolTargetHintKind,
};
use crate::workspace::Workspace;

pub struct EditFile {
    backend: SharedBackend,
    policy: Arc<dyn FilePolicy>,
    text: ToolText,
}

impl EditFile {
    /// A tool confined to `workspace`.
    pub fn new(workspace: Workspace) -> Self {
        let (backend, policy) = workspace_parts(workspace);
        Self::on_backend(backend, policy)
    }

    /// A tool working through `backend` under an application's `policy`.
    pub fn on_backend(backend: SharedBackend, policy: Arc<dyn FilePolicy>) -> Self {
        Self {
            backend,
            policy,
            text: ToolText {
                short: "Find and replace text in a file".into(),
                schema: "Find and replace text in a file. Fails without changing anything when the text is not found or is ambiguous; read the file again and retry.".into(),
            },
        }
    }

    /// Replace the model-facing descriptions.
    pub fn with_text(mut self, text: ToolText) -> Self {
        self.text = text;
        self
    }
}

/// One edit applied to a file's bytes.
struct AppliedEdit {
    new_content: String,
    /// Occurrences of the text that were found.
    count: usize,
    effective_new_text: String,
    used_newline_recovery: bool,
}

/// Apply one edit to `bytes`, or say why it cannot be applied. The tool and
/// its `file_write_result` share this, so a predicted result is the one the
/// call writes.
fn apply_edit(
    bytes: &[u8],
    path_str: &str,
    old_text: &str,
    new_text: &str,
    replace_all: bool,
) -> anyhow::Result<AppliedEdit> {
    if bytes.iter().take(8192).any(|byte| *byte == 0) {
        anyhow::bail!("Cannot edit binary file: {}", path_str);
    }
    let content = std::str::from_utf8(bytes)
        .map_err(|_| anyhow::anyhow!("Cannot edit non-UTF-8 file: {}", path_str))?;

    // Count occurrences using exact match first.
    let mut effective_old_text = old_text.to_string();
    let mut effective_new_text = new_text.to_string();
    let mut count = content.matches(&effective_old_text).count();

    // Self-recovery path: tolerate newline-style mismatch (LF vs CRLF) while
    // preserving strict exact matching otherwise.
    if count == 0 {
        let file_newline = detect_newline_style(content);
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
        anyhow::bail!("{}", build_not_found_message(path_str, content, old_text));
    }

    if count > 1 && !replace_all {
        anyhow::bail!(
            "Found {} occurrences of the text in {}. Set replace_all=true to replace all, or provide more context to make old_text unique.",
            count,
            path_str
        );
    }

    let new_content = if replace_all {
        content.replace(&effective_old_text, &effective_new_text)
    } else {
        content.replacen(&effective_old_text, &effective_new_text, 1)
    };
    Ok(AppliedEdit {
        new_content,
        count,
        used_newline_recovery: effective_old_text != old_text,
        effective_new_text,
    })
}

#[async_trait]
impl OutcomeTool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        &self.text.short
    }

    fn schema(&self) -> Value {
        json!({
            "name": "edit_file",
            "description": self.text.schema,
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

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
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

        let backend = self.backend.clone();
        let path = backend.resolve_path(path_str).await?;

        if backend.metadata(&path).await.is_err() {
            anyhow::bail!("File not found: {}", path_str);
        }

        // Held from before the read until the write has landed.
        let write_guard = match crate::semantics::file_fence(facts) {
            Some(fence) => Some(
                fence
                    .begin_write(path_str)
                    .await
                    .map_err(|why| anyhow::anyhow!(why))?,
            ),
            None => None,
        };

        let bytes = backend.read(&path).await?;
        let AppliedEdit {
            new_content,
            count,
            effective_new_text,
            used_newline_recovery,
        } = apply_edit(&bytes, path_str, old_text, new_text, replace_all)?;

        // Backup + atomic write
        self.policy.backup_before_overwrite(&backend, &path).await;
        backend
            .write(&path, new_content.as_bytes(), WriteMode::Overwrite, false)
            .await?;
        if let Some(write_guard) = write_guard {
            write_guard.committed().await;
        }

        // Show context around the change
        let replaced_count = if replace_all { count } else { 1 };
        let context = get_change_context(&new_content, &effective_new_text);
        let diagnostics = self
            .policy
            .post_write_diagnostics(&*backend, std::path::Path::new(path.as_str()))
            .await;

        Ok(ToolCallOutcome::from_output(format!(
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
        )))
    }

    fn validate_call(&self, arguments: &Value) -> Result<(), String> {
        let text = |keys: [&str; 4]| {
            keys.iter()
                .find_map(|key| arguments.get(*key).and_then(Value::as_str))
        };
        let old = text(["old_text", "old_string", "search", "find"]);
        let new = text(["new_text", "new_string", "replace", "replacement"]);
        if old == Some("") {
            return Err("`old_text` must not be empty; use write_file to create a file".into());
        }
        if old.is_some() && old == new {
            return Err("`old_text` and `new_text` are identical; nothing to change".into());
        }
        Ok(())
    }
}

core_tool!(
    EditFile,
    annotations: ToolAnnotations::WRITE,
    access: |this, arguments| {
        let paths: Vec<String> = path_arg(arguments)
            .map(|path| vec![this.policy.declare_path(path)])
            .unwrap_or_default();
        AccessManifest {
            reads: paths.clone(),
            writes: paths,
            ..Default::default()
        }
    }
);

impl crate::semantics::ToolSemantics for EditFile {
    fn dispatch_facts(&self) -> crate::semantics::ToolDispatchFacts {
        crate::semantics::ToolDispatchFacts {
            always_offered: true,
            checkpoints_writes: true,
            file_guard: crate::semantics::FileGuard::Writes,
            ..Default::default()
        }
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

    fn call_file_write(&self, arguments: &str) -> Option<crate::semantics::FileWrite> {
        let args: Value = serde_json::from_str(arguments).ok()?;
        let old_text = ["old_text", "old_string", "search", "find"]
            .iter()
            .find_map(|key| args.get(*key).and_then(Value::as_str))?;
        Some(crate::semantics::FileWrite {
            path: path_arg(&args)?.to_string(),
            kind: crate::semantics::FileWriteKind::Edit {
                old_text: old_text.to_string(),
                replace_all: args["replace_all"].as_bool().unwrap_or(false),
            },
        })
    }

    fn file_write_result(
        &self,
        arguments: &str,
        current: Option<&[u8]>,
    ) -> Option<Result<Vec<u8>, String>> {
        let args: Value = serde_json::from_str(arguments).ok()?;
        let path = path_arg(&args)?;
        let old_text = ["old_text", "old_string", "search", "find"]
            .iter()
            .find_map(|key| args.get(*key).and_then(Value::as_str))?;
        let new_text = ["new_text", "new_string", "replace", "replacement"]
            .iter()
            .find_map(|key| args.get(*key).and_then(Value::as_str))?;
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);
        let Some(current) = current else {
            return Some(Err(format!("File not found: {path}")));
        };
        Some(
            apply_edit(current, path, old_text, new_text, replace_all)
                .map(|applied| applied.new_content.into_bytes())
                .map_err(|why| why.to_string()),
        )
    }

    /// A write of the target path (absolute; a relative path resolves where
    /// the execution backend puts it, which is not known here).
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        self.policy.write_effects(&*self.backend, arguments)
    }

    fn call_annotations_in(
        &self,
        arguments: &str,
        _roots: &crate::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        let semantics = self.call_semantics(arguments);
        let mut annotations =
            crate::semantics::default_call_annotations(self.capabilities(), &semantics);
        // A write that can plant git configuration changes what later
        // "read-only" git commands execute: treat it as a configuration change.
        let plants_git_config = semantics
            .target_hints
            .iter()
            .any(|hint| self.policy.plants_configuration(&hint.value));
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
