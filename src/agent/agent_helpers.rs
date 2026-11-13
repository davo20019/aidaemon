//! Free-function helpers extracted from `agent/mod.rs` (Phase 5 decoupling).
//!
//! Pure relocation — no logic changes. Groups status/heartbeat helpers, tool-arg
//! summarization, resume-checkpoint rendering and detection, project-scope and
//! filesystem-reference cue detection, untrusted-external-reference tool
//! filtering, and the intent-gate decision merge.

use serde_json::Value;

#[cfg(test)]
use super::contains_keyword_as_words;
use super::execution_state::StepExecutionOutcome;

pub use aidaemon_runtime::turn::helpers::touch_heartbeat;

pub(in crate::agent) use aidaemon_runtime::turn::tool_summary::summarize_tool_args;

/// Legacy intent-gate test fixture. Production routing is model-directed.
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub(in crate::agent) struct IntentGateDecision {
    /// Heuristics armed the tool requirement: the request references the
    /// filesystem, local execution, auth/integration management, or a
    /// connected API and cannot be satisfied by a text-only reply.
    pub(in crate::agent) needs_tools: Option<bool>,
}

#[derive(Debug, Clone)]
pub(in crate::agent) struct ResumeCheckpoint {
    pub(in crate::agent) task_id: String,
    pub(in crate::agent) description: String,
    pub(in crate::agent) original_user_message: Option<String>,
    pub(in crate::agent) elapsed_secs: u64,
    pub(in crate::agent) last_iteration: u32,
    pub(in crate::agent) tool_results_count: u32,
    pub(in crate::agent) pending_tool_call_ids: Vec<String>,
    pub(in crate::agent) last_assistant_summary: Option<String>,
    pub(in crate::agent) last_tool_summary: Option<String>,
    pub(in crate::agent) last_error: Option<String>,
    pub(in crate::agent) execution_snapshot: Option<ResumeExecutionSnapshot>,
    /// The interrupted task's ORIGINAL turn_id (from its TaskStart event).
    /// Recovery TaskEnd MUST use this, never the new resume turn. None for
    /// legacy tasks whose TaskStart predates turn_id persistence.
    pub(in crate::agent) turn_id: Option<String>,
}

#[derive(Debug, Clone)]
pub(in crate::agent) struct ResumeExecutionSnapshot {
    pub(in crate::agent) execution_id: String,
    pub(in crate::agent) current_step_id: Option<String>,
    pub(in crate::agent) current_tool: Option<String>,
    pub(in crate::agent) current_target: Option<String>,
    pub(in crate::agent) last_outcome: Option<StepExecutionOutcome>,
    pub(in crate::agent) background_handoff_active: bool,
    pub(in crate::agent) idempotency_key: Option<String>,
}

impl ResumeCheckpoint {
    pub(in crate::agent) fn render_prompt_section(&self) -> String {
        let mut lines = vec![
            "## Resume Checkpoint".to_string(),
            "The user explicitly asked to continue prior in-progress work. Resume from this checkpoint and avoid restarting completed steps."
                .to_string(),
            format!("- Previous task_id: {}", self.task_id),
            format!("- Original task: {}", self.description),
            format!("- Elapsed before interruption: {}s", self.elapsed_secs),
            format!("- Last completed iteration: {}", self.last_iteration),
            format!("- Completed tool results: {}", self.tool_results_count),
            format!(
                "- Pending unresolved tool calls: {}",
                self.pending_tool_call_ids.len()
            ),
        ];

        if !self.pending_tool_call_ids.is_empty() {
            let pending = self
                .pending_tool_call_ids
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("- Pending tool call IDs: {}", pending));
        }
        if let Some(msg) = &self.original_user_message {
            lines.push(format!(
                "- Original user request: {}",
                truncate_for_resume(msg, 180)
            ));
        }
        if let Some(summary) = &self.last_assistant_summary {
            lines.push(format!("- Last assistant output: {}", summary));
        }
        if let Some(summary) = &self.last_tool_summary {
            lines.push(format!("- Last tool result: {}", summary));
        }
        if let Some(err) = &self.last_error {
            lines.push(format!("- Last error: {}", err));
        }
        if let Some(snapshot) = &self.execution_snapshot {
            lines.push(format!("- Execution id: {}", snapshot.execution_id));
            if let Some(step_id) = &snapshot.current_step_id {
                lines.push(format!("- Last execution step: {}", step_id));
            }
            if let Some(tool) = &snapshot.current_tool {
                lines.push(format!("- Last execution tool: {}", tool));
            }
            if let Some(target) = &snapshot.current_target {
                lines.push(format!("- Last execution target: {}", target));
            }
            if let Some(outcome) = snapshot.last_outcome {
                lines.push(format!("- Last execution outcome: {:?}", outcome));
            }
            if snapshot.background_handoff_active {
                lines.push("- Background execution was active before interruption.".to_string());
            }
            if let Some(key) = &snapshot.idempotency_key {
                lines.push(format!(
                    "- Replay/idempotency key: {}",
                    truncate_for_resume(key, 120)
                ));
            }
        }
        lines.push(
            "Resume from the next concrete step immediately. Re-run tools only if needed to verify or recover."
                .to_string(),
        );
        lines.join("\n")
    }
}

pub(in crate::agent) use aidaemon_runtime::turn::helpers::truncate_for_resume;

fn normalize_for_resume_intent(text: &str) -> String {
    text.split_whitespace()
        .map(|part| part.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(in crate::agent) fn is_resume_request(text: &str) -> bool {
    let normalized = normalize_for_resume_intent(text);
    if normalized.is_empty() {
        return false;
    }

    const EXACT: &[&str] = &[
        "continue",
        "resume",
        "keep going",
        "go on",
        "carry on",
        "next phase",
        "next step",
    ];
    if EXACT.contains(&normalized.as_str()) {
        return true;
    }

    normalized.starts_with("continue ")
        || normalized.starts_with("resume ")
        || normalized.starts_with("keep going ")
        || normalized.starts_with("go on ")
        || normalized.starts_with("carry on ")
        || normalized.starts_with("next phase ")
        || normalized.starts_with("next step ")
}

#[cfg(test)]
pub(in crate::agent) fn user_text_references_filesystem_path(user_text: &str) -> bool {
    let user_text = crate::channels::attachments::user_authored_text(user_text);
    if user_text.trim().is_empty() {
        return false;
    }

    for raw in user_text.as_str().split_whitespace() {
        let token = raw
            .trim_matches(|c: char| {
                c.is_ascii_whitespace()
                    || matches!(
                        c,
                        '`' | '\'' | '"' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}'
                    )
            })
            .trim_end_matches(['.', '!', '?']);
        if token.is_empty() {
            continue;
        }
        if crate::tools::fs_utils::resolve_structural_filesystem_reference(
            &*crate::execution::active_execution_backend(),
            token,
            &[],
        )
        .is_some()
        {
            return true;
        }
    }

    false
}

#[cfg(test)]
pub(in crate::agent) fn user_explicitly_requests_local_file_inspection(user_text: &str) -> bool {
    if user_text_references_filesystem_path(user_text) {
        return true;
    }

    let lower = user_text.to_ascii_lowercase();
    let mentions_local_subject = [
        "file",
        "files",
        "repo",
        "repository",
        "codebase",
        "directory",
        "folder",
        "workspace",
        "local file",
        "local files",
        "current repo",
        "this repo",
        "the repo",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));
    let mentions_inspection_verb = [
        "read", "open", "inspect", "look in", "look at", "search", "scan", "check", "review",
        "show", "list", "find", "grep", "compare",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    mentions_local_subject && mentions_inspection_verb
}

pub(in crate::agent) fn is_untrusted_external_reference_blocked_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file"
            | "search_files"
            | "project_inspect"
            | "check_environment"
            | "web_fetch"
            | "web_search"
            | "browser"
            | "send_file"
            | "skill_resources"
    )
}

pub(in crate::agent) fn filter_tool_defs_for_untrusted_external_reference(
    defs: &[Value],
) -> Vec<Value> {
    defs.iter()
        .filter(|def| {
            let name = def
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str());
            !name.is_some_and(is_untrusted_external_reference_blocked_tool)
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_summary_wraps_command_in_backticks() {
        let args = serde_json::json!({"command": "cargo build --release"}).to_string();
        assert_eq!(
            summarize_tool_args("terminal", &args),
            "`cargo build --release`"
        );
    }

    #[test]
    fn terminal_summary_collapses_multiline_commands() {
        let args = serde_json::json!({"command": "echo one &&\n  echo   two"}).to_string();
        assert_eq!(
            summarize_tool_args("terminal", &args),
            "`echo one && echo two`"
        );
    }

    #[test]
    fn terminal_summary_redacts_secret_before_truncation() {
        // The key starts just before the 80-char truncation boundary. If
        // truncation ran first, the pattern would no longer match and a
        // prefix of the key would leak.
        let cmd = format!("echo {} sk-{}", "x".repeat(60), "a".repeat(30));
        let args = serde_json::json!({ "command": cmd }).to_string();
        let summary = summarize_tool_args("terminal", &args);
        assert!(
            !summary.contains("sk-a"),
            "no fragment of the key may survive: {summary}"
        );
    }

    #[test]
    fn terminal_summary_shortens_home_dir() {
        let home = dirs::home_dir().unwrap();
        let cmd = format!(
            "ls {}/projects",
            home.to_string_lossy().trim_end_matches('/')
        );
        let args = serde_json::json!({ "command": cmd }).to_string();
        assert_eq!(summarize_tool_args("terminal", &args), "`ls ~/projects`");
    }

    #[test]
    fn terminal_summary_truncation_preserves_closing_backtick() {
        // Long commands must stay within the downstream 80-char status cap
        // (sanitize::STATUS_SUMMARY_MAX_CHARS) so the second truncation never
        // strips the closing backtick.
        let cmd = "x".repeat(200);
        let args = serde_json::json!({ "command": cmd }).to_string();
        let summary = summarize_tool_args("terminal", &args);
        assert!(
            summary.chars().count() <= 80,
            "got {}",
            summary.chars().count()
        );
        assert!(
            summary.starts_with('`') && summary.ends_with('`'),
            "got: {summary}"
        );
        assert!(
            summary.contains("..."),
            "long command should show truncation"
        );
    }
}
