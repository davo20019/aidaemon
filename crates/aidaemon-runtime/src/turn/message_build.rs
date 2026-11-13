use std::sync::atomic::Ordering;

use aidaemon_events::conversation::{Message, MessageAttachment};
use aidaemon_tools::status::StatusUpdate;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::host::TurnHost;
use crate::media::attachment_content::build_attachment_content;
use crate::media::vision::user_message_content_matches;
use crate::policy::execution::PolicyBundle;
use crate::policy::metrics::POLICY_METRICS;
use crate::turn::helpers::{send_status, sort_tool_definitions_by_name, truncate_for_resume};
use crate::turn::loop_utils::{fixup_message_ordering, is_trigger_session};
use crate::turn::system_directives::SystemDirective;
use crate::turn::turn_render::RenderMode;

pub struct MessageBuildCtx<'a> {
    pub session_id: &'a str,
    pub iteration: usize,
    pub user_text: &'a str,
    /// Attachments for the current user turn (images saved to inbox).
    pub current_attachments: &'a [MessageAttachment],
    pub completed_tool_calls: &'a [String],
    pub model: &'a str,
    /// Pillar A: message-zero bytes (session-static CORE prompt). Byte-stable
    /// across the within-task loop so the prefix cache reuses it.
    pub core_prompt: &'a str,
    /// Pillar A: the per-task volatile context tail, inserted right after the
    /// current user message. The SAME string is reused every iteration of the
    /// within-task loop.
    pub task_context_tail: &'a str,
    /// Whether archived conversation turns belong to this request's semantic
    /// thread. A typed new-task boundary starts a fresh transcript floor;
    /// follow-ups and internal continuations preserve the thread from that
    /// floor onward.
    pub preserve_archived_context: bool,
    /// Exact canonical assistant message selected before the model runs. This
    /// is metadata for locating the parent turn; it is never inferred from
    /// assistant text or repeated into the current user message.
    pub prior_assistant_message_id: Option<&'a str>,
    /// Cursor of the last message already represented by the cumulative
    /// conversation summary. Whole turns through this message are omitted from
    /// the verbatim transcript so summary and history never duplicate them.
    pub summary_last_message_id: Option<&'a str>,
    pub tool_defs: &'a [Value],
    pub policy_bundle: &'a PolicyBundle,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub empty_response_retry_pending: bool,
    /// Shared-channel non-owners keep visible conversation history but must not
    /// receive hidden tool calls/results or local attachment metadata from an
    /// owner's earlier turn in the same session.
    pub redact_archived_shared_context: bool,
    /// Whether the exact resources earlier work reached (projected turns'
    /// successful receipts, and what a child inherited) are named in the
    /// task context. Off where host paths must not reach the model's
    /// audience (public channels, shared-channel guests, mandate workers).
    pub share_known_resources: bool,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
}

pub struct MessageBuildData {
    pub messages: Vec<Value>,
    pub tool_defs: Vec<Value>,
    /// Exact persisted message/turn identities whose content survived all
    /// rendering, compaction, and eviction into the provider-visible payload.
    pub projected_source_message_ids: Vec<String>,
    pub projected_source_turn_ids: Vec<String>,
}

/// The persisted turns projected into `messages`, without removing their
/// markers (see [`take_projected_source_ids`]).
pub fn peek_projected_turn_ids(messages: &[Value]) -> Vec<String> {
    let mut turn_ids = Vec::new();
    for ids in messages
        .iter()
        .filter_map(|message| message.get(crate::turn::turn_render::SOURCE_TURN_IDS_KEY))
        .filter_map(Value::as_array)
    {
        for id in ids.iter().filter_map(Value::as_str) {
            if !turn_ids.iter().any(|known: &String| known == id) {
                turn_ids.push(id.to_string());
            }
        }
    }
    turn_ids
}

pub fn take_projected_source_ids(messages: &mut [Value]) -> (Vec<String>, Vec<String>) {
    let mut message_ids = Vec::new();
    let mut turn_ids = Vec::new();
    for message in messages {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if let Some(ids) = object
            .remove(crate::turn::turn_render::SOURCE_MESSAGE_IDS_KEY)
            .and_then(|value| value.as_array().cloned())
        {
            for id in ids
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_string))
            {
                if !message_ids.contains(&id) {
                    message_ids.push(id);
                }
            }
        }
        if let Some(ids) = object
            .remove(crate::turn::turn_render::SOURCE_TURN_IDS_KEY)
            .and_then(|value| value.as_array().cloned())
        {
            for id in ids
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_string))
            {
                if !turn_ids.contains(&id) {
                    turn_ids.push(id);
                }
            }
        }
    }
    (message_ids, turn_ids)
}

pub const EMPTY_RETRY_MAX_PARENT_CHARS: usize = 800;
pub const EXECUTION_CHECKPOINT_MAX_REQUEST_CHARS: usize = 240;
pub const EXECUTION_CHECKPOINT_MAX_ACTIVITY_CHARS: usize = 900;
pub const EXECUTION_CHECKPOINT_MAX_EVIDENCE_CHARS: usize = 500;
pub const RESPONSE_RESERVE_TOKENS: usize = 1_536;
pub const MIN_MESSAGE_BUDGET_TOKENS: usize = 1_024;
pub const TOKEN_ESTIMATE_SAFETY_MARGIN: usize = 256;
/// Pillar B (Task 7): fixed token reserve for the in-flight current turn — the
/// current user message plus expected tool-chain headroom. Held constant across
/// turns so the archived-region budget does not churn build-to-build (a
/// per-turn-derived reserve would shift the eviction boundary every iteration).
pub const CURRENT_TURN_RESERVE_TOKENS: usize = 4_000;
/// Pillar B (Task 7): safety margin applied to the archived-region budget.
pub const ARCHIVED_BUDGET_SAFETY_MARGIN: f64 = 0.10;
pub const COMPACT_PARENT_USER_CHARS: usize = 1_200;
pub const COMPACT_PARENT_ASSISTANT_CHARS: usize = 4_000;
pub const COMPACT_PARENT_DELIVERY_CHARS: usize = 320;
pub const COMPACT_PARENT_RETRIEVAL_CHARS: usize = 600;
pub const COMPACT_PARENT_RETRIEVAL_ITEMS: usize = 2;
pub const ADJACENT_EXCHANGE_LOCATOR_KEY: &str = "_aidaemon_adjacent_exchange";

pub fn redact_archived_shared_turns(turns: &mut Vec<aidaemon_events::FetchedTurn>) {
    for turn in turns.iter_mut() {
        turn.messages.retain(|message| message.role != "tool");
        for message in &mut turn.messages {
            if message.role == "assistant" {
                message.tool_calls_json = None;
                message.tool_call_id = None;
                message.tool_name = None;
            }
            if let Some(content) = message.content.as_mut() {
                let without_attachment_paths = if message.role == "user" {
                    aidaemon_channels::attachments::strip_inbound_file_stubs(content)
                } else {
                    content.clone()
                };
                *content = aidaemon_tools::sanitize::redact_secrets(&without_attachment_paths);
            }
            message.attachments.clear();
            message.annotations.clear();
        }
        turn.messages.retain(|message| {
            message.role == "user"
                || message
                    .content
                    .as_deref()
                    .is_some_and(|content| !content.trim().is_empty())
        });
    }
    turns.retain(|turn| !turn.messages.is_empty());
}

/// Recent turns a new request still sees. A new request starts a new task, not
/// a new conversation: pronouns and elided references ("how old are they?")
/// resolve against what was just said, whatever the relationship classifier
/// concluded.
pub const NEW_REQUEST_DIALOGUE_TURNS: usize = 6;
pub const NEW_REQUEST_DIALOGUE_MAX_AGE_HOURS: i64 = 12;

/// Reduce archived turns to the recent dialogue a new request may see: the
/// last few turns inside the age window, rendered as user text and final
/// assistant replies only. Tool calls and results are removed, so an earlier
/// task's observations never reach the transcript as fresh evidence; the
/// task's own receipts remain the only proof of work. Returns how many turns
/// were dropped.
pub fn retain_recent_dialogue_for_new_request(
    turns: &mut Vec<aidaemon_events::FetchedTurn>,
    now: chrono::DateTime<chrono::Utc>,
) -> usize {
    let before = turns.len();
    let cutoff = now - chrono::Duration::hours(NEW_REQUEST_DIALOGUE_MAX_AGE_HOURS);
    turns.retain(|turn| {
        turn.messages
            .iter()
            .map(|message| message.created_at)
            .max()
            .is_some_and(|last_activity| last_activity >= cutoff)
    });
    let excess = turns.len().saturating_sub(NEW_REQUEST_DIALOGUE_TURNS);
    turns.drain(..excess);
    redact_archived_shared_turns(turns);
    before.saturating_sub(turns.len())
}

pub fn compact_parent_text(content: &str, max_chars: usize, strip_file_stubs: bool) -> String {
    let content = if strip_file_stubs {
        aidaemon_channels::attachments::strip_inbound_file_stubs(content)
    } else {
        content.to_string()
    };
    let redacted = aidaemon_tools::sanitize::redact_secrets(&content);
    truncate_for_resume(redacted.trim(), max_chars)
}

pub fn compact_parent_answer(content: &str, max_chars: usize) -> String {
    let redacted = aidaemon_tools::sanitize::redact_secrets(content);
    let redacted = redacted.trim();
    let count = redacted.chars().count();
    if count <= max_chars {
        return redacted.to_string();
    }
    let head_chars = (max_chars * 2) / 3;
    let tail_chars = max_chars.saturating_sub(head_chars);
    let head: String = redacted.chars().take(head_chars).collect();
    let tail: String = redacted.chars().skip(count - tail_chars).collect();
    format!(
        "{head}\n[Middle of prior answer omitted: retained {max_chars}/{count} characters]\n{tail}"
    )
}

/// Provider-valid shell for the immediately preceding turn when its complete
/// archived rendering cannot fit even the low-water window. Retain the human
/// request, final answer, and a bounded set of tool observations. This is
/// intentionally not used for older turns.
pub fn compact_immediate_parent_turn(turn_messages: &[Message]) -> Vec<Value> {
    let prior_user = turn_messages.iter().rev().find_map(|message| {
        if message.role != "user" {
            return None;
        }
        let content = compact_parent_text(
            message.content.as_deref().unwrap_or_default(),
            COMPACT_PARENT_USER_CHARS,
            true,
        );
        (!content.is_empty()).then_some((message, content))
    });
    let prior_assistant = turn_messages.iter().rev().find_map(|message| {
        if message.role != "assistant" {
            return None;
        }
        let raw = message.content.as_deref().unwrap_or_default();
        if raw.trim().is_empty() || crate::turn::turn_render::is_failure_boilerplate(raw) {
            return None;
        }
        let content = compact_parent_answer(raw, COMPACT_PARENT_ASSISTANT_CHARS);
        (!content.is_empty()).then_some((message, content))
    });
    let delivery_evidence = turn_messages.iter().rev().find_map(|message| {
        if message.role != "tool"
            || !matches!(
                message.tool_name.as_deref(),
                Some("send_file" | "send_media")
            )
        {
            return None;
        }
        let primary = message.primary_content().unwrap_or_default();
        let content = compact_parent_text(&primary, COMPACT_PARENT_DELIVERY_CHARS, false);
        (!content.is_empty()).then_some((message, content))
    });
    let mut observation_evidence: Vec<(&Message, String, String)> = turn_messages
        .iter()
        .rev()
        .filter_map(|message| {
            let tool_name = message.tool_name.as_deref()?;
            if message.role != "tool" {
                return None;
            }
            let primary = message.primary_content().unwrap_or_default();
            let content = compact_parent_text(&primary, COMPACT_PARENT_RETRIEVAL_CHARS, false);
            (!content.is_empty()).then(|| (message, tool_name.to_string(), content))
        })
        .take(COMPACT_PARENT_RETRIEVAL_ITEMS)
        .collect();
    observation_evidence.reverse();

    let mut compact = Vec::with_capacity(2);
    if let Some((message, prior_user)) = prior_user {
        let mut value = json!({
            "role": "user",
            "content": prior_user,
        });
        crate::turn::turn_render::attach_source_provenance(&mut value, &[message]);
        compact.push(value);
    }

    let mut assistant_lines = vec![
        "[Immediate prior turn retained in compact form; tool protocol omitted and observations are explicitly bounded.]"
            .to_string(),
    ];
    let mut assistant_sources = Vec::new();
    if let Some((message, prior_assistant)) = prior_assistant {
        assistant_sources.push(message);
        assistant_lines.push(prior_assistant);
    }
    if let Some((message, delivery_evidence)) = delivery_evidence {
        assistant_sources.push(message);
        assistant_lines.push(format!("Delivery evidence: {delivery_evidence}"));
    }
    if !observation_evidence.is_empty() {
        assistant_lines.push("Prior bounded tool evidence:".to_string());
        assistant_lines.extend(
            observation_evidence
                .iter()
                .map(|(_, tool, evidence)| format!("- {tool}: {evidence}")),
        );
        assistant_sources.extend(
            observation_evidence
                .into_iter()
                .map(|(message, _, _)| message),
        );
    }
    if assistant_lines.len() > 1 {
        let mut value = json!({
            "role": "assistant",
            "content": assistant_lines.join("\n"),
        });
        crate::turn::turn_render::attach_source_provenance(&mut value, &assistant_sources);
        compact.push(value);
    }

    compact
}

pub fn trimmed_message_content(message: &Value) -> Option<String> {
    message
        .get("content")
        .and_then(|c| c.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

pub fn truncate_parent_for_empty_retry(content: &str) -> String {
    let mut out: String = content.chars().take(EMPTY_RETRY_MAX_PARENT_CHARS).collect();
    if content.chars().count() > EMPTY_RETRY_MAX_PARENT_CHARS {
        out.push_str("...");
    }
    out
}

pub fn is_current_user_message(message: &Value, user_text: &str) -> bool {
    message.get("role").and_then(|r| r.as_str()) == Some("user")
        && message
            .get("content")
            .is_some_and(|content| user_message_content_matches(content, user_text))
}

pub fn find_current_user_position(messages: &[Value], user_text: &str) -> Option<usize> {
    messages
        .iter()
        .rposition(|m| is_current_user_message(m, user_text))
}

pub fn current_turn_user_attachments(
    turn: &aidaemon_events::FetchedTurn,
    user_text: &str,
) -> Vec<MessageAttachment> {
    turn.messages
        .iter()
        .filter(|m| m.role == "user")
        .find(|m| m.content.as_deref() == Some(user_text))
        .map(|m| m.attachments.clone())
        .unwrap_or_default()
}

pub fn build_empty_response_retry_messages(existing: &[Value], user_text: &str) -> Vec<Value> {
    let current_idx = find_current_user_position(existing, user_text);
    let search_end = current_idx.unwrap_or(existing.len());

    let prev_assistant = existing
        .iter()
        .take(search_end)
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .and_then(trimmed_message_content);

    let prev_user = existing
        .iter()
        .take(search_end)
        .rev()
        .find(|m| {
            if m.get("role").and_then(|r| r.as_str()) != Some("user") {
                return false;
            }
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|content| content != user_text && !content.trim().is_empty())
        })
        .and_then(trimmed_message_content);

    let mut recovered = Vec::new();
    if let Some(prev_user) = prev_user {
        recovered.push(json!({
            "role": "user",
            "content": truncate_parent_for_empty_retry(&prev_user),
        }));
    }
    if let Some(prev_assistant) = prev_assistant {
        recovered.push(json!({
            "role": "assistant",
            "content": truncate_parent_for_empty_retry(&prev_assistant),
        }));
    }
    recovered.push(json!({
        "role": "user",
        "content": user_text,
    }));

    recovered
}

pub fn tool_is_low_info_for_checkpoint(tool_name: &str) -> bool {
    matches!(tool_name, "write_file" | "edit_file" | "check_environment")
        || crate::turn::tool_facts::memory_tool_facts(tool_name)
            .is_some_and(|facts| facts.low_signal_result)
}

pub fn build_execution_checkpoint_message(
    user_text: &str,
    completed_tool_calls: &[String],
    current_interaction: &[&Message],
) -> Option<String> {
    let trimmed_user = user_text.trim();
    if trimmed_user.is_empty() || completed_tool_calls.is_empty() {
        return None;
    }

    let activity = crate::turn::post_task::categorize_tool_calls(completed_tool_calls);
    let latest_evidence = current_interaction.iter().rev().find_map(|message| {
        if message.role != "tool" {
            return None;
        }
        let tool_name = message.tool_name.as_deref().unwrap_or("").trim();
        if tool_name.is_empty() || tool_is_low_info_for_checkpoint(tool_name) {
            return None;
        }
        let content = message.primary_content()?;
        let content = content.trim();
        if content.is_empty() {
            return None;
        }
        Some(format!(
            "- {}: {}",
            tool_name,
            truncate_for_resume(content, EXECUTION_CHECKPOINT_MAX_EVIDENCE_CHARS)
        ))
    });

    let mut lines = vec![
        "[SYSTEM] EXECUTION CHECKPOINT: You are still working on the same active request from this turn.".to_string(),
        format!(
            "Active request: {}",
            truncate_for_resume(trimmed_user, EXECUTION_CHECKPOINT_MAX_REQUEST_CHARS)
        ),
    ];

    if !activity.trim().is_empty() {
        lines.push("Completed work so far:".to_string());
        lines.push(truncate_for_resume(
            activity.trim(),
            EXECUTION_CHECKPOINT_MAX_ACTIVITY_CHARS,
        ));
    }

    if let Some(evidence) = latest_evidence {
        lines.push("Latest concrete evidence:".to_string());
        lines.push(evidence);
    }

    lines.push("Continue from this checkpoint. Do NOT reset into a generic availability reply or ask what the user wants help with. Either take the next step for this request, answer with concrete results if it is complete, or state the blocker tied to this request.".to_string());

    Some(lines.join("\n"))
}

pub async fn build_turn_messages(
    host: &dyn TurnHost,
    ctx: &mut MessageBuildCtx<'_>,
) -> anyhow::Result<MessageBuildData> {
    let runtime = host.turn_runtime();
    let session_id = ctx.session_id;
    let iteration = ctx.iteration;
    let user_text = ctx.user_text;
    let current_attachments = ctx.current_attachments;
    let completed_tool_calls = ctx.completed_tool_calls;
    let model = ctx.model;
    let core_prompt = ctx.core_prompt;
    let task_context_tail = ctx.task_context_tail;
    let preserve_archived_context = ctx.preserve_archived_context;
    let prior_assistant_message_id = ctx.prior_assistant_message_id;
    let summary_last_message_id = ctx.summary_last_message_id;
    let original_tool_defs = ctx.tool_defs;
    let policy_bundle = ctx.policy_bundle;
    let pending_system_messages = &mut *ctx.pending_system_messages;
    let empty_response_retry_pending = ctx.empty_response_retry_pending;
    let status_tx = ctx.status_tx;
    let render_options = runtime.render_options(model);

    let total_context_budget =
        crate::context_budget::model_context_budget(model, &runtime.context_window_config);
    // Pillar A: the system payload is now message zero (core) PLUS the task
    // context tail. Both occupy budget, so reserve for the sum.
    let system_tokens = crate::context_budget::estimate_tokens(core_prompt)
        + crate::context_budget::estimate_tokens(task_context_tail);
    let original_tool_tokens =
        crate::context_budget::estimate_tool_definition_tokens(original_tool_defs);
    let tool_budget = total_context_budget
        .saturating_sub(system_tokens + RESPONSE_RESERVE_TOKENS + MIN_MESSAGE_BUDGET_TOKENS);
    let mut effective_tool_defs =
        crate::context_budget::fit_tool_definitions_to_budget(original_tool_defs, tool_budget);

    if effective_tool_defs != original_tool_defs {
        info!(
            session_id,
            iteration,
            model,
            total_context_budget,
            original_tool_tokens,
            effective_tool_tokens =
                crate::context_budget::estimate_tool_definition_tokens(&effective_tool_defs),
            tool_count = effective_tool_defs.len(),
            "Compacted tool schema descriptions for model context compatibility"
        );
    }
    let mut tool_defs = effective_tool_defs.as_slice();

    // ============================================================
    // Pillar B (Task 7): turn-anchored fetch → render → evict.
    //
    // This replaces the legacy fetch → age-collapse → sliding-window →
    // JSON-conversion stages. The archived region is whole-turn rendered (never
    // partially trimmed), anchored at `runtime.turn_anchors[session_id]`, and
    // carried byte-stable across the within-task loop via the per-turn render
    // cache. Only the current turn is full/append-only.
    // ============================================================

    use aidaemon_events::TerminalState;

    // Step 0: archived-region budget. Wire it from the SAME budget sources the
    // legacy code used — no invented numbers. `core` is the cached core bytes
    // (already computed by Pillar A); `tools` is the name-sorted tool array this
    // build will send; `tail` is the `[Task Context]` tail string.
    let core_tokens = crate::context_budget::estimate_tokens(core_prompt);
    let tail_tokens = crate::context_budget::estimate_tokens(task_context_tail);
    let tool_tokens_for_budget = crate::context_budget::estimate_tool_definition_tokens(tool_defs);
    let archived_budget = crate::turn::turn_eviction::archived_budget(
        total_context_budget,
        core_tokens,
        tool_tokens_for_budget,
        tail_tokens,
        CURRENT_TURN_RESERVE_TOKENS,
        crate::context_budget::CONTEXT_RESPONSE_RESERVE_TOKENS,
        ARCHIVED_BUDGET_SAFETY_MARGIN,
    );

    let current_turn_id: Option<String> = runtime
        .current_turn_ids
        .read()
        .await
        .get(session_id)
        .cloned();

    // Step 1: anchor resolve. Read the per-session anchor; on cold start /
    // restart, initialize via the BOUNDED reverse-walk page (NEVER a
    // full-session `get_turns_from_anchor(.., 0)`).
    let anchor: i64 = {
        let existing = runtime.turn_anchors.read().await.get(session_id).copied();
        match existing {
            Some(a) => a,
            None => {
                // Reverse-walk newest→oldest, accumulating each turn's Archived
                // estimate, stopping at the last turn that keeps the running
                // total <= low_water(archived_budget).
                let low_water = crate::turn::turn_eviction::low_water(archived_budget);
                let mut before: Option<i64> = None;
                let mut accumulated: usize = 0;
                let mut init_anchor: Option<i64> = None;
                let mut archived_turns_seen: usize = 0;
                loop {
                    let page = host
                        .events()
                        .get_recent_turns_page(session_id, before, 1)
                        .await?;
                    let Some(turn) = page.into_iter().next() else {
                        break;
                    };
                    let terminal_state = TerminalState::from_task_status(turn.terminal_status);
                    let contains_prior_assistant = prior_assistant_message_id.is_some_and(|id| {
                        turn.messages
                            .iter()
                            .any(|message| message.role == "assistant" && message.id == id)
                    });
                    let rendered = crate::turn::turn_render::render_turn(
                        &turn.messages,
                        if contains_prior_assistant {
                            crate::turn::turn_render::RenderMode::Adjacent { terminal_state }
                        } else {
                            crate::turn::turn_render::RenderMode::Archived { terminal_state }
                        },
                        crate::turn::turn_render::RENDERER_VERSION,
                        &render_options,
                    );
                    let full_est =
                        crate::context_budget::estimate_multimodal_message_tokens(&rendered);
                    if init_anchor.is_none() {
                        // The newest turn is the (about-to-be) current turn. It is
                        // not part of the archived region, so its tokens are NOT
                        // counted against low_water; set the anchor floor here and
                        // walk past it to reach the archived turns.
                        init_anchor = Some(turn.turn_seq);
                        before = Some(turn.turn_seq);
                        continue;
                    }
                    let (est, compacted_parent) =
                        if archived_turns_seen == 0 && full_est > low_water {
                            let compact = compact_immediate_parent_turn(&turn.messages);
                            let compact_est =
                                crate::context_budget::estimate_multimodal_message_tokens(&compact);
                            if !compact.is_empty() && compact_est <= low_water {
                                (compact_est, true)
                            } else {
                                (full_est, false)
                            }
                        } else {
                            (full_est, false)
                        };
                    archived_turns_seen = archived_turns_seen.saturating_add(1);
                    // Stop once adding this archived turn would exceed low_water.
                    if accumulated + est > low_water {
                        break;
                    }
                    if compacted_parent {
                        info!(
                            session_id,
                            turn_seq = turn.turn_seq,
                            full_est_tokens = full_est,
                            compact_est_tokens = est,
                            "Oversized immediate parent retained as compact shell during cold start"
                        );
                    }
                    accumulated = accumulated.saturating_add(est);
                    init_anchor = Some(turn.turn_seq);
                    before = Some(turn.turn_seq);
                }
                let resolved = init_anchor.unwrap_or(0);
                runtime
                    .turn_anchors
                    .write()
                    .await
                    .insert(session_id.to_string(), resolved);
                info!(
                    session_id,
                    anchor = resolved,
                    archived_budget,
                    "anchor initialized (cold start)"
                );
                resolved
            }
        }
    };

    // Step 2: fetch whole turns from the anchor.
    let mut turns = host
        .events()
        .get_turns_from_anchor(session_id, anchor)
        .await?;

    // A cumulative summary replaces the completed whole turns it covers, except
    // for the exact preceding assistant exchange. Continuity is a hard pin: an
    // aggressive summary window must not erase the parent the current user can
    // refer to. If the cursor predates the in-memory anchor it will not be
    // present, and there is nothing here to remove.
    if let Some(summary_cursor) = summary_last_message_id {
        let parent_turn_seq = prior_assistant_message_id.and_then(|assistant_id| {
            turns.iter().find_map(|turn| {
                turn.messages
                    .iter()
                    .any(|message| message.role == "assistant" && message.id == assistant_id)
                    .then_some(turn.turn_seq)
            })
        });
        if let Some(summary_turn_seq) = turns.iter().find_map(|turn| {
            turn.messages
                .iter()
                .any(|message| message.id == summary_cursor)
                .then_some(turn.turn_seq)
        }) {
            turns.retain(|turn| {
                turn.turn_seq > summary_turn_seq || Some(turn.turn_seq) == parent_turn_seq
            });
        }
    }

    // Identify the current turn = last FetchedTurn whose turn_id == current_turn_id.
    let last_is_current = turns
        .last()
        .map(|t| t.turn_id.as_deref() == current_turn_id.as_deref() && current_turn_id.is_some())
        .unwrap_or(false);

    if !last_is_current {
        // Current-turn fallback (replaces `current_user_injected`). The happy
        // path persists the current UserMessage event before this build runs,
        // so the fetch ends in the current turn. This covers (a) the documented
        // not-yet-committed race and (b) legacy turn_id=NULL rows excluded by
        // the anchored fetch. Synthesize a current turn from the in-process
        // user_text + current_turn_id so the payload always ends in the current
        // user message.
        let synthetic_turn_id = current_turn_id.clone();
        let synthetic_seq = turns.last().map(|t| t.turn_seq + 1).unwrap_or(0);
        let current_user = Message {
            id: format!("synthetic-current-user-{}", uuid::Uuid::new_v4()),
            session_id: session_id.to_string(),
            role: "user".to_string(),
            content: Some(user_text.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: chrono::Utc::now(),
            importance: 1.0,
            turn_id: synthetic_turn_id.clone(),
            attachments: current_attachments.to_vec(),
            ..Message::runtime_defaults()
        };
        turns.push(aidaemon_events::FetchedTurn {
            turn_id: synthetic_turn_id,
            turn_seq: synthetic_seq,
            messages: vec![current_user],
            terminal_status: None,
        });
        warn!(
            session_id,
            "current turn absent from fetch; injected in-process"
        );
    } else {
        // Happy-path assertion: the fetch ends in the current turn.
        debug_assert!(
            turns
                .last()
                .map(|t| t.turn_id.as_deref() == current_turn_id.as_deref())
                .unwrap_or(false),
            "turn-anchored fetch must end in the current turn on the happy path"
        );
    }

    // Split off the current turn (always the last entry now).
    let current_turn = turns.pop().expect("at least the current turn is present");
    // Top-level chat keeps its recent dialogue across a new-request boundary.
    // Delegated and mandate workers keep the strict floor: their transcript is
    // an authority envelope, not a conversation with the user.
    let isolated_transcript = host.is_worker();
    if !preserve_archived_context && !isolated_transcript {
        let archived_turns_dropped =
            retain_recent_dialogue_for_new_request(&mut turns, chrono::Utc::now());
        // Bound later fetches to the retained window. The durable
        // `session_context_boundaries` row belongs to explicit `/clear`; a
        // semantic new-request classification must never write it.
        let floor = turns
            .first()
            .map(|turn| turn.turn_seq)
            .unwrap_or(current_turn.turn_seq);
        let mut anchors = runtime.turn_anchors.write().await;
        let anchor = anchors.entry(session_id.to_string()).or_insert(floor);
        *anchor = (*anchor).max(floor);
        info!(
            session_id,
            iteration,
            context_floor_turn_seq = *anchor,
            dialogue_turns_retained = turns.len(),
            archived_turns_dropped,
            "Retained recent dialogue for new request"
        );
    } else if !preserve_archived_context {
        let archived_turns_dropped = turns.len();
        turns.clear();

        let context_boundary_event_id = current_turn.turn_seq.saturating_sub(1);
        if let Some(retained_turn_id) = current_turn.turn_id.as_deref() {
            host.messages()
                .advance_session_context_boundary(
                    session_id,
                    context_boundary_event_id,
                    retained_turn_id,
                )
                .await?;
        } else {
            warn!(
                session_id,
                iteration,
                context_boundary_event_id,
                "Current turn lacks a canonical turn ID; context floor is process-local"
            )
        }

        // Advance the session floor at the same authoritative boundary used
        // to assemble the provider transcript. The state-store boundary makes
        // the floor durable across restarts; the in-memory anchor avoids an
        // unnecessary reverse walk during this process lifetime. Subsequent
        // typed follow-ups may see this task thread, but cannot resurrect an
        // older unrelated task merely because there is room in the context.
        let mut anchors = runtime.turn_anchors.write().await;
        let anchor = anchors
            .entry(session_id.to_string())
            .or_insert(current_turn.turn_seq);
        *anchor = (*anchor).max(current_turn.turn_seq);
        info!(
            session_id,
            iteration,
            context_floor_turn_seq = current_turn.turn_seq,
            archived_turns_dropped,
            "Advanced conversation context floor for new task"
        );
    }
    // `turns` now holds only archived turns (oldest→newest).
    if ctx.redact_archived_shared_context {
        redact_archived_shared_turns(&mut turns);
    }

    // Step 3: evict. Render each archived turn (Archived mode) to estimate over
    // the FINAL bytes, then plan_eviction. Cache the renders so step 4 reuses
    // them (and to drive the debug re-render assertion).
    struct ArchivedRender {
        turn_id: Option<String>,
        turn_seq: i64,
        content_fp: String,
        /// Bytes assembled into this provider request (possibly the compact
        /// immediate-parent shell).
        bytes: Vec<Value>,
        /// Canonical full archived render retained in the render cache.
        cache_bytes: Vec<Value>,
        cache_hit: bool,
        cache_reason: &'static str,
        contains_prior_assistant: bool,
    }

    let mut archived_renders: Vec<ArchivedRender> = Vec::with_capacity(turns.len());
    {
        // One read-lock snapshot of this session's render cache for the decision;
        // writes happen after the loop to avoid holding the lock across renders.
        let prev_cache = runtime.turn_renders.read().await;
        let session_cache = prev_cache.get(session_id);
        for (turn_idx, turn) in turns.iter().enumerate() {
            let terminal_state = TerminalState::from_task_status(turn.terminal_status);
            let contains_prior_assistant = prior_assistant_message_id.is_some_and(|id| {
                turn.messages
                    .iter()
                    .any(|message| message.role == "assistant" && message.id == id)
            });
            let fp = crate::turn::turn_render_cache::content_fp(&turn.messages, terminal_state);
            let prev = turn
                .turn_id
                .as_deref()
                .and_then(|tid| session_cache.and_then(|c| c.get(tid)));
            let render_fn = || {
                crate::turn::turn_render::render_turn(
                    &turn.messages,
                    crate::turn::turn_render::RenderMode::Archived { terminal_state },
                    crate::turn::turn_render::RENDERER_VERSION,
                    &render_options,
                )
            };
            let (cache_bytes, hit, reason) = crate::turn::turn_render_cache::render_cache_decision(
                prev,
                &fp,
                crate::turn::turn_render::RENDERER_VERSION,
                "archived",
                render_fn,
            );
            // Determinism guard: on a cache HIT, debug/test builds re-render and
            // assert the cached bytes still match (nondeterminism = panic). On a
            // miss `bytes` is itself a fresh render, so re-rendering would compare
            // a fresh render against itself — skip it.
            #[cfg(debug_assertions)]
            if hit {
                let fresh = crate::turn::turn_render::render_turn(
                    &turn.messages,
                    crate::turn::turn_render::RenderMode::Archived { terminal_state },
                    crate::turn::turn_render::RENDERER_VERSION,
                    &render_options,
                );
                assert_eq!(
                    fresh, cache_bytes,
                    "archived render must be deterministic for turn {:?}",
                    turn.turn_id
                );
            }
            let request_bytes = if contains_prior_assistant {
                crate::turn::turn_render::render_turn(
                    &turn.messages,
                    crate::turn::turn_render::RenderMode::Adjacent { terminal_state },
                    crate::turn::turn_render::RENDERER_VERSION,
                    &render_options,
                )
            } else {
                cache_bytes.clone()
            };
            let full_est =
                crate::context_budget::estimate_multimodal_message_tokens(&request_bytes);
            let is_immediate_parent = turn_idx + 1 == turns.len();
            let bytes = if is_immediate_parent
                && full_est > crate::turn::turn_eviction::low_water(archived_budget)
            {
                let compact = compact_immediate_parent_turn(&turn.messages);
                if compact.is_empty() {
                    request_bytes.clone()
                } else {
                    let compact_est =
                        crate::context_budget::estimate_multimodal_message_tokens(&compact);
                    info!(
                        session_id,
                        turn_seq = turn.turn_seq,
                        full_est_tokens = full_est,
                        compact_est_tokens = compact_est,
                        "Oversized immediate parent rendered as compact shell"
                    );
                    compact
                }
            } else {
                request_bytes
            };
            archived_renders.push(ArchivedRender {
                turn_id: turn.turn_id.clone(),
                turn_seq: turn.turn_seq,
                content_fp: fp,
                bytes,
                cache_bytes,
                cache_hit: hit,
                cache_reason: reason,
                contains_prior_assistant,
            });
        }
    }

    let rendered_for_plan: Vec<crate::turn::turn_eviction::RenderedTurn> = archived_renders
        .iter()
        .map(|r| crate::turn::turn_eviction::RenderedTurn {
            turn_seq: r.turn_seq,
            est_tokens: crate::context_budget::estimate_multimodal_message_tokens(&r.bytes),
        })
        .collect();
    let plan = crate::turn::turn_eviction::plan_eviction(&rendered_for_plan, archived_budget);

    if plan.degenerate {
        warn!(
            session_id,
            iteration,
            archived_budget,
            core_tokens,
            tool_tokens = tool_tokens_for_budget,
            tail_tokens,
            current_turn_reserve = CURRENT_TURN_RESERVE_TOKENS,
            "Archived budget degenerate: non-evictable components exceed context; zero archived turns"
        );
    }

    if plan.evicted_count > 0 {
        // Advance the anchor, drop evicted renders, and prune their cache entries.
        let evicted: Vec<ArchivedRender> = archived_renders.drain(..plan.evicted_count).collect();
        {
            let mut anchors = runtime.turn_anchors.write().await;
            anchors.insert(session_id.to_string(), plan.new_anchor_turn_seq);
        }
        {
            let mut cache = runtime.turn_renders.write().await;
            if let Some(session_cache) = cache.get_mut(session_id) {
                for r in &evicted {
                    if let Some(tid) = r.turn_id.as_deref() {
                        session_cache.remove(tid);
                    }
                }
            }
        }
        info!(
            session_id,
            iteration,
            new_anchor = plan.new_anchor_turn_seq,
            turns_evicted = plan.evicted_count,
            kept_est_tokens = plan.kept_est_tokens,
            archived_kept = archived_renders.len(),
            archived_budget,
            "Window decision"
        );
    }
    if plan.degenerate {
        // Degenerate: carry zero archived turns.
        archived_renders.clear();
    }

    // Step 4 (storage): persist the kept archived renders into the cache and
    // emit per-turn hit/miss logging.
    {
        let mut cache = runtime.turn_renders.write().await;
        let session_cache = cache.entry(session_id.to_string()).or_default();
        for r in &archived_renders {
            if r.cache_hit {
                tracing::debug!(
                    session_id,
                    turn_id = ?r.turn_id,
                    "archived render cache hit"
                );
            } else {
                info!(
                    session_id,
                    turn_id = ?r.turn_id,
                    reason = r.cache_reason,
                    "archived render cache miss"
                );
            }
            if let Some(tid) = r.turn_id.as_deref() {
                session_cache.insert(
                    tid.to_string(),
                    crate::turn::turn_render_cache::CachedRender {
                        content_fp: r.content_fp.clone(),
                        renderer_version: crate::turn::turn_render::RENDERER_VERSION,
                        mode_tag: "archived".to_string(),
                        bytes: r.cache_bytes.clone(),
                    },
                );
            }
        }
    }

    // Step 5: render the current turn (full / append-only).
    let current_rendered = crate::turn::turn_render::render_turn(
        &current_turn.messages,
        crate::turn::turn_render::RenderMode::Current,
        crate::turn::turn_render::RENDERER_VERSION,
        &render_options,
    );

    // The channel UI may retain more turns than the active model window. Make
    // that boundary explicit so absence from this request is never interpreted
    // as evidence that an earlier visible result did not exist.
    let oldest_retained_turn_seq = archived_renders
        .first()
        .map(|render| render.turn_seq)
        .unwrap_or(current_turn.turn_seq);
    let exact_older_history_omitted = match host
        .events()
        .get_recent_turns_page(session_id, Some(oldest_retained_turn_seq), 1)
        .await
    {
        Ok(older) => !older.is_empty(),
        Err(error) => {
            warn!(
                session_id,
                %error,
                "Could not determine exact older-history availability"
            );
            false
        }
    };

    // Step 6 (assembly, part 1): archived turns first, then the current turn.
    // Pillar A tail + core insertion happen further below, unchanged.
    let mut messages: Vec<Value> = Vec::new();
    if exact_older_history_omitted {
        let summary_boundary = ctx.summary_last_message_id.unwrap_or("none");
        messages.push(json!({
            "role": "system",
            "content": format!(
                "[Context availability: exact_older_turns=omitted; omitted_before_turn_seq={oldest_retained_turn_seq}; reason=context_window_or_summary_boundary; summary_last_message_id={summary_boundary}; visible_channel_history_may_be_larger=true] Older exact conversation turns are outside the active model window. A missing prior detail is unavailable in this context, not disproven. State that limitation explicitly unless canonical history evidence is already supplied in the retained context."
            )
        }));
    }
    let exact_parent_found = archived_renders
        .iter()
        .any(|render| render.contains_prior_assistant);
    if prior_assistant_message_id.is_some() && !exact_parent_found {
        tracing::warn!(
            session_id,
            prior_assistant_message_id,
            "Exact preceding assistant fell outside retained context; using nearest retained exchange"
        );
    }
    for (index, r) in archived_renders.iter().enumerate() {
        let fallback_immediate_parent = !exact_parent_found && index + 1 == archived_renders.len();
        if r.contains_prior_assistant || fallback_immediate_parent {
            messages.push(json!({
                "role": "system",
                "content": "",
                "_aidaemon_adjacent_exchange": true,
            }));
        }
        messages.extend(r.bytes.iter().cloned());
    }
    // The current region begins here. Current-region fitting (step 7) re-locates
    // it by the current user message (content == user_text) AFTER provider-validity
    // fixups, so no fragile index is threaded across the intervening passes.
    messages.extend(current_rendered);

    // Three-pass fixup: merge → drop orphans → merge again (provider-validity).
    fixup_message_ordering(&mut messages);

    tracing::debug!(
        session_id,
        iteration,
        stage = "turn_anchored_assembly",
        archived_turns = archived_renders.len(),
        pre_boundary_hash = %crate::prompt::fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Execution checkpoint (iteration > 1): a system reminder of the active
    // request + completed work, anchored to the CURRENT turn's messages (the
    // current interaction). Appended at the tail further below.
    let execution_checkpoint = if iteration > 1 {
        let current_interaction: Vec<&Message> = current_turn.messages.iter().collect();
        build_execution_checkpoint_message(user_text, completed_tool_calls, &current_interaction)
    } else {
        None
    };

    // Ensure the current user message is in the context.
    // The DB write (append_user_message_with_event) may not yet be visible
    // to load_recent_history due to a race condition, especially on
    // iteration 1. It can also be missing on iteration 2 after a
    // early-iteration `continue` (no messages are stored between iterations,
    // so the race condition persists). Check all messages on every iteration
    // to be safe — the content match prevents duplicates.
    {
        let has_current_user_msg = messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("user")
                && m.get("content").is_some_and(|content| {
                    crate::media::vision::user_message_content_matches(content, user_text)
                })
        });

        if !has_current_user_msg {
            let turn_attachments = current_turn_user_attachments(&current_turn, user_text);
            let attachments: &[MessageAttachment] = if !turn_attachments.is_empty() {
                &turn_attachments
            } else {
                current_attachments
            };
            let build = build_attachment_content(
                user_text,
                attachments,
                RenderMode::Current,
                &render_options.vision,
                &render_options.audio,
                model,
            );
            messages.push(json!({
                "role": "user",
                "content": build.content,
            }));
        }
    }

    // Task boundary marker: when there are multiple user messages in context
    // (i.e., multiple independent tasks in the same chat session), inject a
    // system separator before the current user message so the LLM knows which
    // task is current. Without this, models confuse old tasks with the new one.
    // Injected on all iterations because on iteration 3+ old user messages can
    // mislead the model. The marker sits before the preserved prior exchange,
    // so it never breaks assistant → user continuity and does not need a
    // follow-up classifier.
    {
        let user_positions: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.get("role").and_then(|r| r.as_str()) == Some("user"))
            .map(|(i, _)| i)
            .collect();
        if user_positions.len() >= 2 {
            // Find the position of the *current* user message — match by content,
            // not just "last user message", so we correctly anchor even when
            // stray user messages from other interactions appear after ours.
            let current_pos = user_positions
                .iter()
                .copied()
                .rev()
                .find(|&pos| {
                    messages[pos].get("content").is_some_and(|content| {
                        crate::media::vision::user_message_content_matches(content, user_text)
                    })
                })
                .or_else(|| user_positions.last().copied());

            if let Some(current_pos) = current_pos {
                let prev_user_content = user_positions
                    .iter()
                    .copied()
                    .filter(|&pos| pos != current_pos)
                    .rev()
                    .find_map(|pos| {
                        messages[pos]
                            .get("content")
                            .and_then(|c| c.as_str())
                            .map(|s| s.to_string())
                    });
                // Only inject if a different task exists in context.
                let has_different_task =
                    prev_user_content.as_deref() != Some(user_text) && prev_user_content.is_some();
                if has_different_task {
                    // Softer marker that tells the LLM which message is current
                    // without telling it to ignore prior context. The old [TASK BOUNDARY]
                    // marker aggressively instructed the LLM to ignore prior messages,
                    // which broke follow-up references like "the ones within 20 miles".
                    let marker = json!({
                        "role": "system",
                        "content": "[Current Task] The latest user message before your work on this \
                                    turn is the current request; user messages marked as sent while \
                                    you were working add to it. Earlier messages remain conversation context."
                    });
                    // After the user message, like the task context tail:
                    // anywhere earlier it would move every turn and end the
                    // prefix a provider cache can reuse.
                    messages.insert(current_pos + 1, marker);
                    info!(
                        session_id,
                        iteration,
                        user_messages = user_positions.len(),
                        "Current task marker injected after the current user message"
                    );
                }
            }
        }
    }

    // Phase 0 stage hash: the `[Current Task]` marker follows the current user
    // message, so it never sits between a prior answer and that message.
    tracing::debug!(
        session_id,
        iteration,
        stage = "current_task_marker",
        pre_boundary_hash = %crate::prompt::fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Guard against context interleaving: if another user message arrived in
    // this session while the agent was processing (race condition between task
    // registration and queuing), it may appear after the current task's tool
    // chain. Such stray user messages confuse the model into responding to them
    // instead of the current task. Remove them.
    {
        let current_task_pos = find_current_user_position(&messages, user_text);
        if let Some(task_pos) = current_task_pos {
            // Find the end of the current task's tool chain (last assistant/tool after task_pos)
            let chain_end = messages
                .iter()
                .enumerate()
                .rev()
                .find(|(i, m)| {
                    *i > task_pos
                        && matches!(
                            m.get("role").and_then(|r| r.as_str()),
                            Some("assistant") | Some("tool")
                        )
                })
                .map(|(i, _)| i)
                .unwrap_or(task_pos);

            // Check for user messages after the tool chain
            let stray_start = chain_end + 1;
            if stray_start < messages.len() {
                let stray_count = messages[stray_start..]
                    .iter()
                    .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
                    .count();
                if stray_count > 0 {
                    messages.truncate(stray_start);
                    info!(
                        session_id,
                        iteration,
                        stray_user_messages = stray_count,
                        "Truncated stray messages after current task's tool chain"
                    );
                }
            }
        }
    }

    // Collapse repeated tool errors in the current interaction to reduce
    // context blow-up during retry loops (keep the latest error details).
    let collapsed_tool_errors =
        crate::turn::loop_utils::collapse_repeated_tool_errors(&mut messages);
    if collapsed_tool_errors > 0 {
        // Pillar B (Task 8): collapsing earlier repeated-error payloads into short
        // notes rewrites stable-region bytes. Attribute it so the prefix-invariant
        // gate (Task 11) does not read this as an unattributed prefix break.
        info!(
            session_id,
            iteration,
            collapsed_tool_errors,
            reason = "repeated_tool_error_collapse",
            "Prefix mutation"
        );
    }
    // Phase 0 stage hash: repeated tool-error collapse. Repeated-error collapse
    // operates on the current interaction (at/after the boundary), but is
    // instrumented for completeness and to confirm it does not perturb the
    // pre-boundary region.
    tracing::debug!(
        session_id,
        iteration,
        stage = "tool_error_collapse",
        pre_boundary_hash = %crate::prompt::fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Context window enforcement: trim messages to fit token budget.
    //
    // Pillar B (Task 7): fitting is scoped to the CURRENT-TURN region only.
    // Archived turns are whole-turn-evicted upstream (spec invariant 3) and must
    // never be trimmed here — only the current turn (everything from the current
    // user message onward) is handed to `fit_messages_with_source_quotas`.
    if runtime.context_window_config.enabled {
        // Reserve for BOTH message zero (core) and the task context tail.
        let system_tokens = crate::context_budget::estimate_tokens(core_prompt)
            + crate::context_budget::estimate_tokens(task_context_tail);
        let model_budget = crate::context_budget::compute_available_budget_precomputed(
            model,
            system_tokens,
            tool_defs,
            &runtime.context_window_config,
        );
        let policy_budget = policy_bundle.policy.context_budget;
        if runtime.policy_config.policy_shadow_mode && !runtime.policy_config.policy_enforce {
            info!(
                session_id,
                iteration, model_budget, policy_budget, "Context budget shadow comparison"
            );
        }
        let effective_budget = if runtime.policy_config.policy_enforce
            && runtime.trust_tier_for_model(model)
                == crate::policy::trust_tier::ModelTrustTier::Guided
        {
            // Never exceed the model's budget; policy config can be mis-set.
            policy_budget.min(model_budget)
        } else {
            model_budget
        };
        // Locate the current-turn region: from the current user message (last
        // occurrence matching `user_text`) to the end of `messages`. Archived
        // turns precede it and are left untouched.
        let current_region_idx = find_current_user_position(&messages, user_text).unwrap_or(0);
        let archived_prefix: Vec<Value> = messages[..current_region_idx].to_vec();
        let current_region: Vec<Value> = messages[current_region_idx..].to_vec();
        // The archived prefix already consumed `archived_budget`; the current
        // region is fit against the remaining budget.
        let archived_prefix_tokens =
            crate::context_budget::estimate_multimodal_message_tokens(&archived_prefix);
        let current_budget = effective_budget.saturating_sub(archived_prefix_tokens);
        let (fitted_current, dropped) =
            crate::context_budget::fit_messages_with_source_quotas(current_region, current_budget);
        messages = archived_prefix;
        messages.extend(fitted_current);
        if dropped > 0 {
            // Pillar B (Task 8): fitting rewrote stable-region bytes by dropping
            // current-turn messages. Attribute it so the prefix-invariant gate
            // (Task 11) does not read this as an unattributed prefix break.
            info!(
                session_id,
                iteration,
                dropped,
                reason = "history_fitting",
                "Prefix mutation"
            );
        }
    }
    // Phase 0 stage hash: history fitting. Fitting is now current-region only;
    // archived turns are whole-turn-evicted upstream and never trimmed here.
    tracing::debug!(
        session_id,
        iteration,
        stage = "history_fitting",
        pre_boundary_hash = %crate::prompt::fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Empty-response recovery: on retry, clear conversational history to avoid
    // repeatedly sending a poisoned context to the provider (Gemini in particular
    // can get "stuck" returning empty candidates for a given session history).
    if empty_response_retry_pending && !is_trigger_session(session_id) {
        let before = messages.len();
        let rebuilt = build_empty_response_retry_messages(&messages, user_text);
        // Fire the prefix-mutation attribution ONLY when the rebuild actually
        // changes the messages (non-zero effect), so quiet turns stay silent.
        let mutated = rebuilt != messages;
        messages = rebuilt;
        info!(
            session_id,
            iteration,
            before,
            after = messages.len(),
            "Empty-response recovery: reduced history while preserving immediate parent context"
        );
        if mutated {
            // Pillar B (Task 8): the retry rebuild rewrote stable-region bytes by
            // collapsing prior history. Attribute it so the prefix-invariant gate
            // (Task 11) does not read this as an unattributed prefix break.
            info!(
                session_id,
                iteration,
                reason = "empty_response_retry",
                "Prefix mutation"
            );
        }
    }

    // History fitting and empty-response recovery select individual messages
    // and can therefore bisect an assistant tool-call/tool-result exchange.
    // Re-establish provider transcript invariants after those destructive
    // rewrites. In particular, the Responses API rejects a
    // `function_call_output` when its matching `function_call` was trimmed.
    let before_fixup = messages.len();
    fixup_message_ordering(&mut messages);
    if messages.len() != before_fixup {
        info!(
            session_id,
            iteration,
            before = before_fixup,
            after = messages.len(),
            reason = "post_fitting_provider_validity",
            "Prefix mutation"
        );
    }

    // The adjacency locator never reaches the provider: it only places the
    // adjacent prior exchange directly before the raw user message.
    if let Some(position) = messages.iter().position(|message| {
        message
            .get(ADJACENT_EXCHANGE_LOCATOR_KEY)
            .and_then(Value::as_bool)
            == Some(true)
    }) {
        messages.remove(position);
    }

    // Pillar A: the per-task context tail goes right after the current user
    // message. Everything before it is then the same bytes the next turn will
    // send, so a provider cache reuses the whole conversation; inside the turn
    // the position is fixed and tool calls append after it. Before the user
    // message it would split the prior exchange from the reply to it, and in
    // the leading system block it would invalidate the core prompt.
    // The exact resources this turn's projected transcript (and a child's
    // parent) already reached, appended to the tail: drawn only from turns
    // the transcript holds, so they carry no taint the turn lacks.
    let known_resources = if ctx.share_known_resources {
        crate::known_resources::handoff_section(
            host.events(),
            host.inherited_authority(),
            session_id,
            &peek_projected_turn_ids(&messages),
        )
        .await
    } else {
        None
    };
    let task_context_tail = match known_resources {
        Some(section) if !task_context_tail.is_empty() => {
            std::borrow::Cow::Owned(format!("{task_context_tail}\n\n{section}"))
        }
        _ => std::borrow::Cow::Borrowed(task_context_tail),
    };
    let task_context_tail = task_context_tail.as_ref();
    if !task_context_tail.is_empty() {
        let tail_insert_pos = find_current_user_position(&messages, user_text)
            .map_or(messages.len(), |position| position + 1);
        messages.insert(
            tail_insert_pos,
            json!({
                "role": "system",
                "content": task_context_tail,
            }),
        );
    }

    // Keep message zero byte-stable across iterations so llama.cpp can reuse the
    // expensive system-prompt prefix. Message zero is the session-static CORE
    // prompt ONLY — volatile per-turn material lives in the task context tail
    // inserted above (Pillar A).
    messages.insert(
        0,
        json!({
            "role": "system",
            "content": core_prompt,
        }),
    );

    // Phase 0 stage hash. Message zero is the core prompt, which
    // `stage_pre_boundary_hash` skips; the tail now sits after the boundary, so
    // the provider-call `tail_hash` covers tail attribution.
    tracing::debug!(
        session_id,
        iteration,
        stage = "context_tail",
        pre_boundary_hash = %crate::prompt::fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    if let Some(checkpoint) = execution_checkpoint {
        messages.push(json!({
            "role": "system",
            "content": checkpoint,
        }));
        info!(
            session_id,
            iteration, "Injected execution checkpoint for in-progress task continuity"
        );
    }
    // Phase 0 stage hash: execution-checkpoint insertion. The checkpoint is
    // appended at the tail (at/after the boundary) and cannot flip the
    // pre-boundary hash, so this stage emits a full-payload hash that tracks
    // tail growth for completeness rather than a pre-boundary hash. The
    // `serde_json::Value::Array` clone is built only inside the `debug!` field
    // expression, so it runs only when debug logging is enabled.
    tracing::debug!(
        session_id,
        iteration,
        stage = "execution_checkpoint",
        full_payload_hash = %crate::prompt::fingerprint::hash_canonical(&serde_json::Value::Array(messages.clone())),
        "Build stage tail fingerprint"
    );

    // Fresh-context isolation: when history is empty or only contains the current
    // user message (e.g. first message after /clear), inject a boundary marker to
    // prevent the LLM from drifting toward stale tool-call patterns from pinned
    // memories or prior context.
    {
        let non_system_non_user_count = messages
            .iter()
            .filter(|m| {
                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
                role != "system" && role != "user"
            })
            .count();
        if non_system_non_user_count == 0 {
            messages.retain(|m| {
                m.get("role").and_then(|r| r.as_str()) != Some("user")
                    || m.get("content")
                        .is_some_and(|content| user_message_content_matches(content, user_text))
            });
            pending_system_messages.push(SystemDirective::FreshConversationContext);
        }
    }

    // System nudges (budget warnings, loop-stop reminders, etc.): inject for a single
    // LLM call so they influence the model without polluting stored history.
    for directive in pending_system_messages.drain(..) {
        messages.push(json!({
            "role": "system",
            "content": directive.render(),
        }));
    }

    // Empty-response recovery: if the prior iteration produced no text and no tool calls,
    // inject a system nudge for the next LLM call. (Tool-role nudges are dropped by
    // message-order fixups because they don't correspond to an assistant tool_call_id.)
    if empty_response_retry_pending && !is_trigger_session(session_id) {
        messages.push(json!({
            "role": "system",
            "content": SystemDirective::EmptyResponseRetry.render()
        }));
    }

    // Collapse superseded computer_use screen-state trees: keep only the most
    // recent full tree, stub the rest. A stale screen snapshot is dead weight
    // (the model acts only on the *current* screen), and these trees are exempt
    // from compression/spill so they can stay actionable — which means several
    // full ~6k-token trees would otherwise accumulate, overflow the context
    // budget, squeeze the tool-definition budget, and trigger a tool-schema
    // refit that breaks the prompt-prefix cache (massive re-prefill). Keeping
    // exactly one full tree bounds the context while the model can still target
    // elements off the latest read.
    collapse_superseded_computer_use_trees(&mut messages);

    // Final enforcement must happen after every prompt component has been inserted.
    // Earlier trimming cannot account for execution checkpoints and one-shot directives.
    if runtime.context_window_config.enabled {
        let message_tokens = crate::context_budget::estimate_multimodal_message_tokens(&messages);
        let final_tool_budget = total_context_budget.saturating_sub(
            message_tokens + RESPONSE_RESERVE_TOKENS + TOKEN_ESTIMATE_SAFETY_MARGIN,
        );
        let final_tool_defs = crate::context_budget::fit_tool_definitions_to_budget(
            original_tool_defs,
            final_tool_budget,
        );
        if final_tool_defs != effective_tool_defs {
            info!(
                session_id,
                iteration,
                model,
                message_tokens,
                final_tool_budget,
                before_tool_tokens =
                    crate::context_budget::estimate_tool_definition_tokens(&effective_tool_defs),
                after_tool_tokens =
                    crate::context_budget::estimate_tool_definition_tokens(&final_tool_defs),
                tool_count = final_tool_defs.len(),
                "Recompacted tool schemas after final prompt assembly"
            );
            effective_tool_defs = final_tool_defs;
            tool_defs = effective_tool_defs.as_slice();
        }
    }

    // Emit "Thinking" status for iterations after the first
    if iteration > 1 {
        send_status(status_tx, StatusUpdate::Thinking(iteration));
    }

    // Internal projection identities are telemetry only. Remove them before
    // token estimation, fingerprinting, request dumps, or provider dispatch.
    let (projected_source_message_ids, projected_source_turn_ids) =
        take_projected_source_ids(&mut messages);

    // Debug: log message structure and estimated token count
    {
        let summary: Vec<String> = messages
            .iter()
            .map(|m| {
                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let tc_id = m
                    .get("tool_call_id")
                    .and_then(|id| id.as_str())
                    .unwrap_or("");
                let tc_count = m
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map_or(0, |a| a.len());
                if role == "tool" {
                    format!("tool({},tc_id={})", name, &tc_id[..tc_id.len().min(12)])
                } else if tc_count > 0 {
                    format!("{}(tc={})", role, tc_count)
                } else {
                    role.to_string()
                }
            })
            .collect();

        let est_msg_tokens = crate::context_budget::estimate_multimodal_message_tokens(&messages);
        let est_tool_tokens = crate::context_budget::estimate_tool_definition_tokens(tool_defs);
        let est_total_tokens = est_msg_tokens + est_tool_tokens;
        let est_msg_tokens_u64 = est_msg_tokens as u64;
        let est_tool_tokens_u64 = est_tool_tokens as u64;
        let est_total_tokens_u64 = est_total_tokens as u64;
        let est_tool_share_bps = est_tool_tokens_u64
            .saturating_mul(10_000)
            .checked_div(est_total_tokens_u64)
            .unwrap_or(0);

        // Runtime signal: quantify prompt overhead from tool schemas before each LLM call.
        POLICY_METRICS
            .est_input_token_samples
            .fetch_add(1, Ordering::Relaxed);
        POLICY_METRICS
            .est_input_tokens_total
            .fetch_add(est_total_tokens_u64, Ordering::Relaxed);
        POLICY_METRICS
            .est_msg_tokens_total
            .fetch_add(est_msg_tokens_u64, Ordering::Relaxed);
        POLICY_METRICS
            .est_tool_tokens_total
            .fetch_add(est_tool_tokens_u64, Ordering::Relaxed);

        const HIGH_TOOL_SHARE_BPS: u64 = 3500; // >=35% of input estimate
        const HIGH_TOOL_TOKENS_ABS: u64 = 1_500; // large absolute tool-schema cost
        if est_tool_share_bps >= HIGH_TOOL_SHARE_BPS {
            POLICY_METRICS
                .est_tool_tokens_high_share_total
                .fetch_add(1, Ordering::Relaxed);
        }
        if est_tool_tokens_u64 >= HIGH_TOOL_TOKENS_ABS {
            POLICY_METRICS
                .est_tool_tokens_high_abs_total
                .fetch_add(1, Ordering::Relaxed);
        }

        info!(
            session_id,
            iteration,
            est_input_tokens = est_total_tokens,
            est_msg_tokens,
            est_tool_tokens,
            total_context_budget,
            response_reserve_tokens = RESPONSE_RESERVE_TOKENS,
            est_tool_share_pct = est_tool_share_bps as f64 / 100.0,
            msg_count = messages.len(),
            msgs = ?summary,
            "Context before LLM call"
        );
    }

    // Pillar A Task 6: name-sort the emitted roster as the FINAL operation before
    // constructing MessageBuildData. This is the authoritative guarantee that the
    // provider tool array is in canonical order regardless of any late
    // append/filter/widen/compaction that mutated `effective_tool_defs` above.
    // Providers stay order-preserving (no sort in adapters).
    sort_tool_definitions_by_name(&mut effective_tool_defs);

    Ok(MessageBuildData {
        messages,
        tool_defs: effective_tool_defs,
        projected_source_message_ids,
        projected_source_turn_ids,
    })
}

/// Replace every computer_use screen-state tool result EXCEPT the most recent
/// one with a compact stub. Identified by the computer_use untrusted-data
/// wrapper plus a `snapshot_generation=` marker (the state-bearing results:
/// `get_app_state` trees and condensed refreshes). Keeps the context bounded so
/// the tool-definition budget is never squeezed into a cache-breaking refit,
/// while the latest full tree stays inline so the model can still target
/// elements. Messages with non-string (multimodal/image) content are left
/// untouched — images are evicted by the separate turn-anchored path.
pub fn collapse_superseded_computer_use_trees(messages: &mut [Value]) {
    const STUB: &str = "[Earlier computer_use screen state omitted — superseded by a newer \
get_app_state below. Call get_app_state to re-read the current screen if you need it.]";
    let is_tree = |m: &Value| -> bool {
        m.get("role").and_then(|r| r.as_str()) != Some("assistant")
            && m.get("content").and_then(|c| c.as_str()).is_some_and(|s| {
                s.contains("UNTRUSTED EXTERNAL DATA from 'computer_use'")
                    && s.contains("snapshot_generation=")
            })
    };
    let Some(last_idx) = messages.iter().rposition(is_tree) else {
        return;
    };
    for (i, m) in messages.iter_mut().enumerate() {
        if i != last_idx
            && is_tree(m)
            && let Some(content) = m.get_mut("content")
        {
            *content = Value::String(STUB.to_string());
        }
    }
}
