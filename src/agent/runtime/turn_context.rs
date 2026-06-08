//! Turn-context assembly: recent-history summarization and the
//! `build_turn_context_from_recent_history` entry point that produces a
//! [`TurnContext`] for each turn, plus session history load/append I/O.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged. The project-hint / project-scope extraction family lives in the
//! sibling [`project_scope`](super::project_scope) module.

use super::completion_contract::infer_completion_contract;
use super::followup::{
    classify_followup_mode, find_previous_turns, has_project_scope_divergence_with_aliases,
    looks_like_multi_project_request, looks_like_scope_carryover_ack, sanitize_carryover_blocks,
    FollowupMode, TurnContextReason,
};
use super::project_scope::{
    choose_primary_project_scope, extract_explicit_path_scopes_from_text,
    extract_project_hints_from_history, extract_project_scopes_from_history,
    extract_project_scopes_from_text, resolve_primary_project_scope,
    turn_allows_inherited_project_scope, unify_current_turn_scopes,
};
use super::*;
use crate::llm_markers::INTENT_GATE_MARKER;

#[derive(Debug, Clone, Default)]
pub(super) struct TurnContext {
    pub goal_user_text: String,
    pub recent_messages: Vec<Value>,
    pub project_hints: Vec<String>,
    pub primary_project_scope: Option<String>,
    pub allow_multi_project_scope: bool,
    pub followup_mode: Option<FollowupMode>,
    pub reasons: Vec<TurnContextReason>,
    pub completion_contract: CompletionContract,
}

const GOAL_CONTEXT_RECENT_MESSAGES_LIMIT: usize = 6;
const GOAL_CONTEXT_HINT_HISTORY_LIMIT: usize = 30;
const GOAL_CONTEXT_MAX_PROJECT_HINTS: usize = 8;
const GOAL_CONTEXT_MAX_PROJECT_SCOPES: usize = 6;

fn trim_assistant_context_content(content: &str) -> String {
    let trimmed = content.trim();
    if let Some((before, _)) = trimmed.split_once(INTENT_GATE_MARKER) {
        before.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

fn is_low_signal_http_metadata_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.starts_with("content-type:")
        || lower.starts_with("content-length:")
        || lower.starts_with("cache-control:")
        || lower.starts_with("date:")
        || lower.starts_with("etag:")
        || lower.starts_with("expires:")
        || lower.starts_with("last-modified:")
        || lower.starts_with("location:")
        || lower.starts_with("server:")
        || lower.starts_with("strict-transport-security:")
        || lower.starts_with("vary:")
        || lower.starts_with("via:")
        || lower.starts_with("x-")
}

fn is_low_info_recent_tool_context(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "write_file"
            | "edit_file"
            | "manage_memories"
            | "manage_people"
            | "remember_fact"
            | "check_environment"
    )
}

fn is_low_signal_tool_context_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.is_empty()
        || is_low_signal_http_metadata_line(line)
        || lower == "[truncated]"
        || lower.starts_with("exit code:")
        || lower.starts_with("[mode:")
}

fn summarize_http_request_context(primary: &str) -> Option<String> {
    let mut lines = primary
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let first = lines.next()?;
    if !first.to_ascii_lowercase().starts_with("http ") {
        return Some(first.to_string());
    }

    let detail = lines.find(|line| !is_low_signal_http_metadata_line(line));
    match detail {
        Some(detail) if !detail.eq_ignore_ascii_case(first) => Some(format!("{first} | {detail}")),
        _ => Some(first.to_string()),
    }
}

fn summarize_generic_tool_context(primary: &str) -> Option<String> {
    let mut lines = primary
        .lines()
        .map(str::trim)
        .filter(|line| !is_low_signal_tool_context_line(line));
    let first = lines.next()?;
    let second = lines.next();
    match second {
        Some(second) if !second.eq_ignore_ascii_case(first) => Some(format!("{first} | {second}")),
        _ => Some(first.to_string()),
    }
}

fn summarize_recent_tool_context(msg: &Message) -> Option<String> {
    let tool_name = msg.tool_name.as_deref()?;
    if is_low_info_recent_tool_context(tool_name) {
        return None;
    }

    let primary = msg.primary_content()?;
    let summary = match tool_name {
        "http_request" => summarize_http_request_context(&primary)?,
        _ => summarize_generic_tool_context(&primary)?,
    };
    Some(format!(
        "{}: {}",
        tool_name,
        truncate_for_resume(&summary, 240)
    ))
}

fn extract_recent_parent_messages(history: &[Message], max_messages: usize) -> Vec<Value> {
    let mut rows_rev: Vec<Value> = Vec::new();
    let mut recent_tool_rows = 0usize;

    for msg in history.iter().rev() {
        let row = match msg.role.as_str() {
            "user" => {
                let Some(raw) = msg.content.as_deref().map(str::trim) else {
                    continue;
                };
                if raw.is_empty() {
                    continue;
                }
                Some(json!({
                    "role": msg.role,
                    "content": truncate_for_resume(raw, 500),
                }))
            }
            "assistant" => {
                let Some(raw) = msg.content.as_deref().map(str::trim) else {
                    continue;
                };
                if raw.is_empty() {
                    continue;
                }
                let content = trim_assistant_context_content(raw);
                if content.trim().is_empty() {
                    continue;
                }
                Some(json!({
                    "role": msg.role,
                    "content": truncate_for_resume(content.trim(), 500),
                }))
            }
            "tool" if recent_tool_rows < 2 => summarize_recent_tool_context(msg).map(|content| {
                recent_tool_rows += 1;
                json!({
                    "role": "tool",
                    "content": content,
                })
            }),
            _ => None,
        };

        if let Some(row) = row {
            rows_rev.push(row);
            if rows_rev.len() >= max_messages {
                break;
            }
        }
    }

    rows_rev.reverse();
    rows_rev
}

// impl-Agent justification: history I/O (append_*/load_*) and turn-context building over state/event_store — shared service for all phases.
impl Agent {
    pub(super) async fn build_turn_context_from_recent_history(
        &self,
        session_id: &str,
        user_text: &str,
    ) -> TurnContext {
        let current = user_text.trim();
        if current.is_empty() {
            return TurnContext::default();
        }

        let history = self
            .state
            .get_history(session_id, GOAL_CONTEXT_HINT_HISTORY_LIMIT)
            .await
            .unwrap_or_default();
        let (prev_assistant, prev_user) = find_previous_turns(&history, current);
        let (mut followup_mode, mut reasons) =
            classify_followup_mode(current, prev_assistant.as_deref());

        let mut goal_user_text = current.to_string();
        // Mismatch preflight: still used for project scope divergence detection
        // (affects scope extraction below), but no longer gates goal_user_text enrichment.
        if followup_mode != FollowupMode::NewTask {
            let mismatch_preflight_drop = prev_user.as_deref().is_some_and(|prev| {
                has_project_scope_divergence_with_aliases(
                    prev,
                    current,
                    &self.path_aliases.projects,
                )
            });
            if mismatch_preflight_drop {
                followup_mode = FollowupMode::NewTask;
                reasons.push(TurnContextReason::FollowupOverrideMismatchPreflight);
                reasons.push(TurnContextReason::DefaultNewTask);
                POLICY_METRICS
                    .context_mismatch_preflight_drop_total
                    .fetch_add(1, Ordering::Relaxed);
                POLICY_METRICS
                    .followup_mode_overrides_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        // Always enrich goal_user_text with prior user message, regardless of
        // classification. The sliding window provides conversational context to
        // the main LLM, and the enriched goal_user_text provides it to the task
        // planner. Without this, short follow-ups like "the ones within 20 miles"
        // are useless for planning.
        if let Some(prev_user_text) = prev_user
            .as_deref()
            .filter(|prev| !prev.trim().eq_ignore_ascii_case(current))
        {
            let mut combined = String::new();
            combined.push_str("Original request:\n");
            combined.push_str(&truncate_for_resume(prev_user_text.trim(), 2000));
            combined.push_str("\n\nCurrent request:\n");
            combined.push_str(current);
            goal_user_text = combined;
        }
        // Sanitize carryover blocks regardless of mode (they can appear in any message)
        {
            let (sanitized, changed) = sanitize_carryover_blocks(&goal_user_text);
            if changed {
                reasons.push(TurnContextReason::CarryoverSanitized);
                POLICY_METRICS
                    .context_bleed_prevented_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            if !sanitized.is_empty() {
                goal_user_text = sanitized;
            }
        }

        let project_hints = extract_project_hints_from_history(
            &history,
            current,
            GOAL_CONTEXT_MAX_PROJECT_HINTS,
            true,
        );
        // For the current user message, only extract explicit filesystem paths
        // (~/foo, /foo, ./foo) — NOT contextual nickname matches.  This prevents
        // common English words like "modern" (in "modern website") from resolving
        // to existing project directories like modern-plants-site.
        let mut current_project_scopes = Vec::new();
        extract_explicit_path_scopes_from_text(
            current,
            &mut current_project_scopes,
            GOAL_CONTEXT_MAX_PROJECT_SCOPES,
            &self.path_aliases.projects,
        );
        let allow_scope_carryover = if looks_like_scope_carryover_ack(current) {
            let mut prior_user_scopes = Vec::new();
            if let Some(prev_user_text) = prev_user.as_deref() {
                extract_project_scopes_from_text(
                    prev_user_text,
                    &mut prior_user_scopes,
                    GOAL_CONTEXT_MAX_PROJECT_SCOPES,
                    &self.path_aliases.projects,
                );
            }
            !prior_user_scopes.is_empty()
                || prev_user.as_deref().is_some_and(|prev_user_text| {
                    turn_allows_inherited_project_scope(prev_user_text, &prior_user_scopes)
                })
        } else {
            turn_allows_inherited_project_scope(current, &current_project_scopes)
        };
        let project_scopes = extract_project_scopes_from_history(
            &history,
            current,
            GOAL_CONTEXT_MAX_PROJECT_SCOPES,
            allow_scope_carryover,
            &self.path_aliases.projects,
        );
        let allow_multi_project_scope =
            looks_like_multi_project_request(&current.to_ascii_lowercase());
        // For current-turn scopes, unify all explicitly mentioned paths into a
        // single scope that encompasses them all.  When the user mentions paths
        // in different subdirectories (e.g. ~/projects/blog/posts/file.md and
        // ~/projects/blog/tweets.md), the unified scope is their common ancestor
        // (~/projects/blog/) so writes to sibling directories aren't blocked.
        //
        // Falls back to `.first()` for single-scope messages, preserving the
        // text-order priority that's critical for new-project creation (the
        // user's explicit path may not exist yet on disk).
        // The project-root preference is only appropriate for history scopes
        // where we want to pick the most likely intended project among stale
        // references.
        let primary_project_scope = resolve_primary_project_scope(
            unify_current_turn_scopes(&current_project_scopes)
                .or_else(|| choose_primary_project_scope(&project_scopes)),
            self.inherited_project_scope.as_deref(),
            allow_multi_project_scope,
            allow_scope_carryover,
        );
        let completion_contract =
            infer_completion_contract(&goal_user_text, &self.path_aliases.projects);
        TurnContext {
            goal_user_text,
            recent_messages: extract_recent_parent_messages(
                &history,
                GOAL_CONTEXT_RECENT_MESSAGES_LIMIT,
            ),
            project_hints,
            primary_project_scope,
            allow_multi_project_scope,
            followup_mode: Some(followup_mode),
            reasons,
            completion_contract,
        }
    }

    /// Resolve the turn_id to stamp on an emitted event for `msg`. Prefer the
    /// message's own turn_id (set for the user message in bootstrap); fall back
    /// to the active turn in `current_turn_ids` for assistant/tool messages that
    /// were constructed without it. Mirrors the auto-stamp in
    /// `append_message_canonical` so the emitted event and the canonical row
    /// share the same turn_id.
    async fn resolve_event_turn_id(&self, msg: &Message) -> Option<String> {
        if msg.turn_id.is_some() {
            return msg.turn_id.clone();
        }
        self.current_turn_ids
            .read()
            .await
            .get(&msg.session_id)
            .cloned()
    }

    pub(super) async fn append_message_canonical(&self, msg: &Message) -> anyhow::Result<()> {
        // Auto-stamp turn_id from the per-session map so every message
        // written during a turn shares the same id. Messages that already
        // carry a turn_id (e.g., the user message in bootstrap, which sets
        // its own id) are left alone.
        if msg.turn_id.is_some() {
            return self.state.append_message(msg).await;
        }
        let turn_id = self
            .current_turn_ids
            .read()
            .await
            .get(&msg.session_id)
            .cloned();
        match turn_id {
            Some(tid) => {
                let mut stamped = msg.clone();
                stamped.turn_id = Some(tid);
                self.state.append_message(&stamped).await
            }
            None => {
                // No active turn for this session — preserve old behavior.
                // This path covers background tasks, sub-agent flows, and
                // test scaffolding that bypass `handle_message_impl`.
                self.state.append_message(msg).await
            }
        }
    }

    pub(super) async fn append_user_message_with_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        user_role: crate::types::UserRole,
        channel_ctx: &ChannelContext,
        has_attachments: bool,
    ) -> anyhow::Result<()> {
        let normalized_msg = msg.with_inferred_annotations();
        emitter
            .emit(
                EventType::UserMessage,
                json!({
                    "content": normalized_msg.content.clone().unwrap_or_default(),
                    "message_id": normalized_msg.id.clone(),
                    "has_attachments": has_attachments,
                    "annotations": normalized_msg.annotations.clone(),
                    // Provenance for downstream projections/consolidation.
                    "channel_visibility": channel_ctx.visibility.to_string(),
                    "channel_id": channel_ctx.channel_id.clone(),
                    "platform": channel_ctx.platform.clone(),
                    "sender_id": channel_ctx.sender_id.clone(),
                    "user_role": user_role.to_string(),
                    // Pillar B: stamp the turn_id onto the emitted event so the
                    // turn-anchored fetch (which filters turn_id IS NOT NULL)
                    // returns this message. Without this the row is NULL.
                    "turn_id": normalized_msg.turn_id,
                }),
            )
            .await?;
        self.append_message_canonical(normalized_msg.as_ref())
            .await?;
        super::dialogue_state::record_dialogue_user_message(
            self,
            &normalized_msg.session_id,
            normalized_msg.as_ref(),
        )
        .await?;
        Ok(())
    }

    pub(super) async fn append_assistant_message_with_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        model: &str,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    ) -> anyhow::Result<()> {
        let normalized_msg = msg.with_inferred_annotations();
        let turn_id = self.resolve_event_turn_id(normalized_msg.as_ref()).await;
        let tool_calls = normalized_msg.tool_calls_json.as_ref().and_then(|raw| {
            serde_json::from_str::<Vec<ToolCall>>(raw)
                .ok()
                .map(|calls| {
                    calls
                        .into_iter()
                        .map(|tc| ToolCallInfo {
                            id: tc.id,
                            name: tc.name,
                            arguments: serde_json::from_str(&tc.arguments)
                                .unwrap_or(serde_json::json!({})),
                            extra_content: tc.extra_content,
                        })
                        .collect::<Vec<_>>()
                })
        });
        emitter
            .emit(
                EventType::AssistantResponse,
                AssistantResponseData {
                    message_id: Some(normalized_msg.id.clone()),
                    content: normalized_msg.content.clone(),
                    model: model.to_string(),
                    tool_calls,
                    input_tokens,
                    output_tokens,
                    annotations: normalized_msg.annotations.clone(),
                    turn_id,
                },
            )
            .await?;
        self.append_message_canonical(normalized_msg.as_ref())
            .await?;
        super::dialogue_state::record_dialogue_assistant_message(
            self,
            &normalized_msg.session_id,
            normalized_msg.as_ref(),
        )
        .await?;
        Ok(())
    }

    pub(super) async fn append_tool_message_with_result_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        success: bool,
        duration_ms: u64,
        error: Option<String>,
        task_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let normalized_msg = msg.with_inferred_annotations();
        let turn_id = self.resolve_event_turn_id(normalized_msg.as_ref()).await;
        emitter
            .emit(
                EventType::ToolResult,
                ToolResultData {
                    message_id: Some(normalized_msg.id.clone()),
                    tool_call_id: normalized_msg
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| normalized_msg.id.clone()),
                    name: normalized_msg
                        .tool_name
                        .clone()
                        .unwrap_or_else(|| "system".to_string()),
                    result: normalized_msg.content.clone().unwrap_or_default(),
                    success,
                    duration_ms,
                    error,
                    task_id: task_id.map(str::to_string),
                    annotations: normalized_msg.annotations.clone(),
                    turn_id,
                },
            )
            .await?;
        self.append_message_canonical(normalized_msg.as_ref())
            .await?;
        Ok(())
    }

    // Pillar B (Task 7): `load_initial_history` and `load_recent_history` were
    // removed. Historical conversation retention is now owned entirely by the
    // turn-anchored fetch (`EventStore::get_turns_from_anchor`) in
    // `message_build_phase`; the legacy "load N recent events then age-collapse"
    // path no longer exists.
}

#[cfg(test)]
mod tests {
    use super::completion_contract::CompletionTaskKind;
    use super::followup::FollowupMode;
    use super::*;
    use chrono::Utc;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        }
    }

    #[test]
    fn recent_parent_messages_strip_intent_gate_payload() {
        let history = vec![
            msg("user", "Build a site"),
            msg(
                "assistant",
                "Sure, I can help.\n[INTENT_GATE] {\"can_answer_now\":false}",
            ),
        ];
        let messages = extract_recent_parent_messages(&history, 6);
        assert_eq!(messages.len(), 2);
        let assistant_content = messages[1]
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(!assistant_content.contains("[INTENT_GATE]"));
        assert_eq!(assistant_content, "Sure, I can help.");
    }
    #[test]
    fn recent_parent_messages_include_recent_api_and_file_tool_summaries() {
        let history = vec![
            msg("user", "Look up those studies."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "http_request",
                    "HTTP 200 OK\ncontent-type: application/json\n\n{\"studies\":[]}",
                )),
                tool_call_id: Some("call-http".to_string()),
                tool_name: Some("http_request".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some("File sent: studies.json (127 KB)".to_string()),
                tool_call_id: Some("call-file".to_string()),
                tool_name: Some("send_file".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "I fetched the results and sent the file."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| content.contains("http_request: HTTP 200 OK"))
        }));
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| content.contains("send_file: File sent: studies.json"))
        }));
    }
    #[test]
    fn recent_parent_messages_preserve_http_error_detail_beyond_headers() {
        let history = vec![
            msg("user", "Check that trial ID."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "http_request",
                    "HTTP 404 Not Found\ncontent-type: application/json\n\nNot Found: NCT05178195",
                )),
                tool_call_id: Some("call-http".to_string()),
                tool_name: Some("http_request".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg(
                "assistant",
                "The API could not find that study, but I should confirm the exact error.",
            ),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| {
                        content
                            .contains("http_request: HTTP 404 Not Found | Not Found: NCT05178195")
                    })
        }));
    }
    #[test]
    fn recent_parent_messages_include_web_fetch_error_summaries() {
        let history = vec![
            msg("user", "Check that page."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "web_fetch",
                    "Error fetching https://example.com/missing: HTTP 404 Not Found",
                )),
                tool_call_id: Some("call-web-fetch".to_string()),
                tool_name: Some("web_fetch".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "That page fetch failed."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row.get("content").and_then(|v| v.as_str()).is_some_and(|content| {
                    content.contains("web_fetch: Error fetching https://example.com/missing: HTTP 404 Not Found")
                    })
        }));
    }
    #[test]
    fn recent_parent_messages_include_generic_terminal_evidence_summary() {
        let history = vec![
            msg("user", "Run the tests."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some("pytest\nAssertionError: expected 1 but got 2".to_string()),
                tool_call_id: Some("call-terminal".to_string()),
                tool_name: Some("terminal".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "The tests failed."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| {
                        content.contains("terminal: pytest | AssertionError: expected 1 but got 2")
                    })
        }));
    }
    #[tokio::test]
    async fn build_turn_context_includes_history_for_all_modes() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Please work in ~/projects/blog.aidaemon.ai/src/content/posts",
            ))
            .await
            .expect("append prior user");
        harness
            .state
            .append_message(&msg("assistant", "Which posts should I update?"))
            .await
            .expect("append prior assistant");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", "Why?")
            .await;

        // Classification still runs (may be NewTask), but history is always included
        // because the sliding window handles context unconditionally.
        assert!(
            !turn_context.recent_messages.is_empty(),
            "recent_messages should always be included regardless of classification"
        );
    }
    #[tokio::test]
    async fn build_turn_context_does_not_reuse_old_local_scope_for_external_followup() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Please work in /Users/davidloor/projects/fairfax-va-site and inspect the logs.",
            ))
            .await
            .expect("append old local user");
        harness
            .state
            .append_message(&msg(
                "assistant",
                "I am currently locked to /Users/davidloor/projects/fairfax-va-site.",
            ))
            .await
            .expect("append old local assistant");
        harness
            .state
            .append_message(&msg(
                "user",
                "Find melanoma clinical trials recruiting near Fairfax, Virginia.",
            ))
            .await
            .expect("append external user");
        harness
            .state
            .append_message(&msg(
                "assistant",
                "I found 10 recruiting trials near Fairfax, VA with male participants.",
            ))
            .await
            .expect("append external assistant");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history(
                "test-session",
                "Give me all the info about the top 2.",
            )
            .await;

        assert_eq!(turn_context.primary_project_scope, None);
    }
    #[tokio::test]
    async fn build_turn_context_keeps_detailed_schedule_request_separate_from_previous_task() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Can you post your daily blog post on your blog?",
            ))
            .await
            .expect("append prior user");
        harness
            .state
            .append_message(&msg("assistant", "Would you like me to schedule that too?"))
            .await
            .expect("append prior assistant");

        let current = "Can you set up a daily scheduled task at 6:00 am to publish the blog with honest reflections about recent errors and fixes.";
        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", current)
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::NewTask));
        // goal_user_text always contains the current request
        assert!(
            turn_context.goal_user_text.contains(current),
            "goal_user_text should contain current request: {}",
            turn_context.goal_user_text
        );
        assert_eq!(
            turn_context.completion_contract.task_kind,
            CompletionTaskKind::Schedule
        );
        assert!(!turn_context.completion_contract.requires_observation);
    }
}
