//! Free-function helpers extracted from `agent/mod.rs` (Phase 5 decoupling).
//!
//! Pure relocation — no logic changes. Groups status/heartbeat helpers, tool-arg
//! summarization, resume-checkpoint rendering and detection, project-scope and
//! filesystem-reference cue detection, untrusted-external-reference tool
//! filtering, and the intent-gate decision merge.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tokio::sync::mpsc;

#[cfg(test)]
use super::contains_keyword_as_words;
use super::execution_state::StepExecutionOutcome;
use super::StatusUpdate;

/// Best-effort send — never blocks the agent loop if the receiver is slow/full.
pub fn send_status(tx: &Option<mpsc::Sender<StatusUpdate>>, update: StatusUpdate) {
    if let Some(ref tx) = tx {
        let _ = tx.try_send(update);
    }
}

/// Update the heartbeat timestamp to signal the agent is alive.
/// No-op when heartbeat is None (sub-agents, triggers, tests).
pub fn touch_heartbeat(hb: &Option<Arc<AtomicU64>>) {
    if let Some(ref hb) = hb {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        hb.store(now, Ordering::Relaxed);
    }
}

/// Extract a brief human-readable summary from tool arguments JSON.
/// Helper to truncate a string and append "..." if it exceeds `max` chars.
fn truncate_summary(s: &str, max: usize) -> String {
    let truncated: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

/// Helper to extract the last path component (file/dir name) for compact display.
fn short_path(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

pub(in crate::agent) fn summarize_tool_args(name: &str, arguments: &str) -> String {
    let val: Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };

    // Helper closure to get a string field from the JSON args.
    let get_str = |key: &str| val.get(key).and_then(|v| v.as_str());

    match name {
        // --- Command execution ---
        "terminal" | "run_command" => get_str("command")
            .map(|cmd| {
                // Order matters: shorten the home dir first (the "File path"
                // secret pattern would otherwise swallow `/Users/...` whole),
                // redact before truncating (a secret cut at the boundary no
                // longer matches its pattern and its prefix would leak), then
                // collapse whitespace so multi-line commands render one-line.
                let cmd = crate::tools::sanitize::shorten_home_dir(cmd);
                let cmd = crate::tools::sanitize::redact_secrets(&cmd);
                let one_line = cmd.split_whitespace().collect::<Vec<_>>().join(" ");
                // 75 + "..." + 2 backticks = 80 — fits the downstream
                // STATUS_SUMMARY_MAX_CHARS cap exactly, so the closing
                // backtick survives the second truncation in sanitize.rs.
                format!("`{}`", truncate_summary(&one_line, 75))
            })
            .unwrap_or_default(),

        // --- File operations ---
        "read_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "write_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "edit_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "search_files" => {
            let pattern = get_str("pattern").or_else(|| get_str("glob")).unwrap_or("");
            if pattern.is_empty() {
                String::new()
            } else {
                truncate_summary(pattern, 40)
            }
        }
        "list_files" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),

        // --- Web & network ---
        "web_search" => get_str("query")
            .map(|q| truncate_summary(q, 50))
            .unwrap_or_default(),
        "web_fetch" => get_str("url")
            .map(|u| truncate_summary(u, 60))
            .unwrap_or_default(),
        "http_request" => {
            let method = get_str("method").unwrap_or("GET");
            let url = get_str("url").unwrap_or("");
            if url.is_empty() {
                method.to_string()
            } else {
                format!("{} {}", method, truncate_summary(url, 50))
            }
        }

        // --- Browser ---
        "browser" => {
            let action = get_str("action").unwrap_or("");
            let url = get_str("url").unwrap_or("");
            if !url.is_empty() {
                format!("{} {}", action, truncate_summary(url, 50))
            } else {
                action.to_string()
            }
        }

        // --- Desktop GUI ---
        "computer_use" => {
            let action = get_str("action").unwrap_or("");
            let app = get_str("app").unwrap_or("");
            let index = val
                .get("element_index")
                .and_then(|v| v.as_u64())
                .map(|i| i.to_string());
            match (action, app, index) {
                ("click", app, Some(index)) if !app.is_empty() => {
                    format!("click {app} #{index}")
                }
                (_, app, _) if !app.is_empty() => format!("{action} {app}"),
                _ => action.to_string(),
            }
        }

        // --- Git ---
        "git_info" => {
            let include = val.get("include").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            });
            include.unwrap_or_default()
        }
        "git_commit" => get_str("message")
            .map(|m| truncate_summary(m, 40))
            .unwrap_or_default(),

        // --- Memory ---
        "remember_fact" => {
            let fact = get_str("fact").or_else(|| get_str("value")).unwrap_or("");
            if fact.is_empty() {
                "saving to memory".to_string()
            } else {
                truncate_summary(fact, 40)
            }
        }
        "manage_memories" => get_str("action").unwrap_or("").to_string(),
        "search_history" => get_str("action").unwrap_or("").to_string(),

        // --- Skills ---
        "use_skill" => get_str("skill_name").unwrap_or("").to_string(),
        "manage_skills" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }

        // --- People ---
        "manage_people" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }

        // --- Agents ---
        "spawn_agent" => get_str("mission")
            .map(|m| truncate_summary(m, 50))
            .unwrap_or_default(),
        "cli_agent" => {
            let action = get_str("action").unwrap_or("run");
            if action != "run" {
                return format!("action={}", action);
            }
            let tool = get_str("tool").unwrap_or("auto");
            let prompt = get_str("prompt")
                .or_else(|| get_str("task"))
                .or_else(|| get_str("mission"))
                .or_else(|| get_str("description"))
                .or_else(|| get_str("command"))
                .unwrap_or("");
            let task_desc = truncate_summary(prompt, 50);
            if task_desc.is_empty() {
                format!("→ {}", tool)
            } else {
                format!("→ {}: {}", tool, task_desc)
            }
        }
        "manage_cli_agents" => get_str("action").unwrap_or("").to_string(),

        // --- Config / diagnostic ---
        "manage_config" => get_str("action").unwrap_or("").to_string(),
        "manage_mcp" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }
        "project_inspect" => {
            if let Some(path) = get_str("path") {
                short_path(path).to_string()
            } else if let Some(paths) = val.get("paths").and_then(|v| v.as_array()) {
                let mut summarized: Vec<String> = paths
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(short_path)
                    .map(str::to_string)
                    .take(3)
                    .collect();
                if summarized.is_empty() {
                    String::new()
                } else {
                    let total = paths.iter().filter_map(|v| v.as_str()).count();
                    if total > summarized.len() {
                        summarized.push(format!("+{} more", total - summarized.len()));
                    }
                    summarized.join(", ")
                }
            } else {
                String::new()
            }
        }

        // --- Channel operations ---
        "read_channel_history" => {
            let channel = get_str("channel_id").unwrap_or("");
            if channel.is_empty() {
                String::new()
            } else {
                truncate_summary(channel, 30)
            }
        }
        "send_file" => get_str("file_path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),

        // --- MCP tools: extract a human-readable name from the prefix ---
        _ if name.starts_with("mcp__") => {
            // mcp__chrome-devtools__take_screenshot → chrome-devtools: take_screenshot
            let without_prefix = &name[5..]; // skip "mcp__"
            if let Some(idx) = without_prefix.find("__") {
                let server = &without_prefix[..idx];
                let tool = &without_prefix[idx + 2..];
                // For common tools, add key arg info
                let arg_info = match tool {
                    "navigate_page" => get_str("url")
                        .map(|u| format!(" {}", truncate_summary(u, 40)))
                        .unwrap_or_default(),
                    "click" | "hover" | "fill" => get_str("uid")
                        .map(|u| format!(" #{}", u))
                        .unwrap_or_default(),
                    "evaluate_script" => get_str("function")
                        .map(|f| format!(" {}", truncate_summary(f, 30)))
                        .unwrap_or_default(),
                    _ => String::new(),
                };
                format!("{}: {}{}", server, tool.replace('_', " "), arg_info)
            } else {
                without_prefix.replace('_', " ")
            }
        }

        _ => String::new(),
    }
}

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

pub(in crate::agent) fn truncate_for_resume(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut out = String::new();
    for (count, ch) in text.chars().enumerate() {
        if count >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

pub(in crate::agent) fn build_empty_response_fallback(response_note: Option<&str>) -> String {
    let base = "I wasn't able to process that request because automatic model recovery returned no usable output.";
    let generic = base.to_string();
    let Some(note) = response_note.map(str::trim).filter(|s| !s.is_empty()) else {
        return generic;
    };

    let flattened = note.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = flattened.trim_matches(|c: char| c == '"' || c == '\'');
    let trimmed = trimmed.trim_end_matches(['.', '!', '?']);
    if trimmed.is_empty() {
        return generic;
    }

    let note_preview = truncate_for_resume(trimmed, 180);
    format!("{base} Provider detail: {note_preview}.")
}

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
        if crate::tools::fs_utils::resolve_structural_filesystem_reference(token, &[]).is_some() {
            return true;
        }
    }

    false
}

#[cfg(test)]
fn text_contains_any_phrase_as_words(text: &str, phrases: &[&str]) -> bool {
    phrases
        .iter()
        .any(|phrase| contains_keyword_as_words(text, phrase))
}

#[cfg(test)]
pub(in crate::agent) fn text_has_explicit_project_scope_cues(text: &str) -> bool {
    text_contains_any_phrase_as_words(
        text,
        &[
            "project",
            "repo",
            "repository",
            "workspace",
            "directory",
            "folder",
            "codebase",
            "code base",
            "work in",
            "inside",
            "under",
        ],
    )
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
