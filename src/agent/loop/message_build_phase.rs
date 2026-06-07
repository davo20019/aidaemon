use super::recall_guardrails::text_relates_to_critical_identity;
use super::*;
use crate::execution_policy::PolicyBundle;

pub(super) struct MessageBuildCtx<'a> {
    pub session_id: &'a str,
    pub iteration: usize,
    pub user_text: &'a str,
    pub completed_tool_calls: &'a [String],
    pub model: &'a str,
    /// Pillar A: message-zero bytes (session-static CORE prompt). Byte-stable
    /// across the within-task loop so the prefix cache reuses it.
    pub core_prompt: &'a str,
    /// Pillar A: the per-task volatile context tail. Inserted at boundary − 1
    /// (immediately before the current user message). The SAME string is reused
    /// every iteration of the within-task loop.
    pub task_context_tail: &'a str,
    pub pinned_memories: &'a [Message],
    pub tool_defs: &'a [Value],
    pub policy_bundle: &'a PolicyBundle,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub empty_response_retry_pending: bool,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
}

pub(super) struct MessageBuildData {
    pub messages: Vec<Value>,
    pub tool_defs: Vec<Value>,
    /// Estimated input tokens (messages + tool schemas) for this call, used for
    /// est-vs-actual drift telemetry in the `LlmCall` event.
    pub est_input_tokens: u32,
}

const EMPTY_RETRY_MAX_PARENT_CHARS: usize = 800;
const EXECUTION_CHECKPOINT_MAX_REQUEST_CHARS: usize = 240;
const EXECUTION_CHECKPOINT_MAX_ACTIVITY_CHARS: usize = 900;
const EXECUTION_CHECKPOINT_MAX_EVIDENCE_CHARS: usize = 500;
const RESPONSE_RESERVE_TOKENS: usize = 1_536;
const MIN_MESSAGE_BUDGET_TOKENS: usize = 1_024;
const TOKEN_ESTIMATE_SAFETY_MARGIN: usize = 256;

fn trimmed_message_content(message: &Value) -> Option<String> {
    message
        .get("content")
        .and_then(|c| c.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn truncate_parent_for_empty_retry(content: &str) -> String {
    let mut out: String = content.chars().take(EMPTY_RETRY_MAX_PARENT_CHARS).collect();
    if content.chars().count() > EMPTY_RETRY_MAX_PARENT_CHARS {
        out.push_str("...");
    }
    out
}

fn build_empty_response_retry_messages(existing: &[Value], user_text: &str) -> Vec<Value> {
    let current_idx = existing.iter().rposition(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user")
            && m.get("content").and_then(|c| c.as_str()) == Some(user_text)
    });
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

fn tool_is_low_info_for_checkpoint(tool_name: &str) -> bool {
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

fn build_execution_checkpoint_message(
    user_text: &str,
    completed_tool_calls: &[String],
    current_interaction: &[&Message],
) -> Option<String> {
    let trimmed_user = user_text.trim();
    if trimmed_user.is_empty() || completed_tool_calls.is_empty() {
        return None;
    }

    let activity = super::post_task::categorize_tool_calls(completed_tool_calls);
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

/// Phase 0 observability — project pre-JSON-conversion history `Message`s into
/// prompt-equivalent JSON for stage fingerprinting. Includes tool metadata
/// (`tool_calls`, `name`, `tool_call_id`) so a change to any of those is
/// observable, matching the provider-call fingerprint's complete-message
/// hashing. Called only inside `tracing::debug!` field expressions, so it runs
/// only when debug logging is enabled.
fn project_messages_for_stage_hash(msgs: &[&Message]) -> Vec<Value> {
    msgs.iter()
        .map(|m| {
            let mut obj = json!({ "role": m.role, "content": m.content });
            if let Some(tc_json) = &m.tool_calls_json {
                if let Ok(tcs) = serde_json::from_str::<Vec<ToolCall>>(tc_json) {
                    obj["tool_calls"] = json!(tcs
                        .iter()
                        .map(|tc| json!({
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }))
                        .collect::<Vec<_>>());
                }
            }
            if let Some(name) = &m.tool_name {
                if !name.is_empty() {
                    obj["name"] = json!(name);
                }
            }
            if let Some(tcid) = &m.tool_call_id {
                obj["tool_call_id"] = json!(tcid);
            }
            obj
        })
        .collect()
}

pub(super) async fn run_message_build_phase(
    services: &super::services::AgentServices<'_>,
    ctx: &mut MessageBuildCtx<'_>,
) -> anyhow::Result<MessageBuildData> {
    let agent = services.agent;
    let session_id = ctx.session_id;
    let iteration = ctx.iteration;
    let user_text = ctx.user_text;
    let completed_tool_calls = ctx.completed_tool_calls;
    let model = ctx.model;
    let core_prompt = ctx.core_prompt;
    let task_context_tail = ctx.task_context_tail;
    let pinned_memories = ctx.pinned_memories;
    let original_tool_defs = ctx.tool_defs;
    let policy_bundle = ctx.policy_bundle;
    let pending_system_messages = &mut *ctx.pending_system_messages;
    let empty_response_retry_pending = ctx.empty_response_retry_pending;
    let status_tx = ctx.status_tx;

    let total_context_budget =
        crate::memory::context_window::model_context_budget(model, &agent.context_window_config);
    // Pillar A: the system payload is now message zero (core) PLUS the task
    // context tail. Both occupy budget, so reserve for the sum.
    let system_tokens = crate::memory::context_window::estimate_tokens(core_prompt)
        + crate::memory::context_window::estimate_tokens(task_context_tail);
    let original_tool_tokens =
        crate::memory::context_window::estimate_tool_definition_tokens(original_tool_defs);
    let tool_budget = total_context_budget
        .saturating_sub(system_tokens + RESPONSE_RESERVE_TOKENS + MIN_MESSAGE_BUDGET_TOKENS);
    let mut effective_tool_defs = crate::memory::context_window::fit_tool_definitions_to_budget(
        original_tool_defs,
        tool_budget,
    );

    if effective_tool_defs != original_tool_defs {
        info!(
            session_id,
            iteration,
            model,
            total_context_budget,
            original_tool_tokens,
            effective_tool_tokens = crate::memory::context_window::estimate_tool_definition_tokens(
                &effective_tool_defs
            ),
            tool_count = effective_tool_defs.len(),
            "Compacted tool schema descriptions for model context compatibility"
        );
    }
    let mut tool_defs = effective_tool_defs.as_slice();

    // Fetch recent history from canonical event stream.
    // Base limit of 40 queries (120 events), scaled up for long-running tasks
    // so that early tool calls from the current task are not pushed out of the
    // window by their own later iterations.  Each iteration generates ~3
    // messages (assistant, tool result(s), sometimes parallel calls), so
    // iteration*3 covers the current task plus old-pair trimming removes the
    // rest.  Capped at 120 to avoid loading entire sessions.
    let history_limit = 40_usize.max(iteration.saturating_mul(3).min(120));
    let mut recent_history = agent.load_recent_history(session_id, history_limit).await?;

    // Phase 0 observability — capture fetch-window facts before any mutation so
    // the window-decision log can tie `keep_from` movement to fetch mechanics
    // (the prime suspect for prefix-cache breaks). `recent_history` is ordered
    // oldest→newest, so `first()` is the oldest fetched persisted message.
    let fetched_count = recent_history.len();
    let oldest_fetched_id = recent_history.first().map(|m| m.id.clone());
    let mut current_user_injected = false;

    // Guarantee the current user message is always present in history.
    // In sessions with heavy prior tool use, the 120-event window may not
    // include the current user message (it was just committed). Without it,
    // last_user_pos=None triggers the safe-collapse fallback which degrades
    // context quality. Appending it ensures the collapse boundary is always
    // correctly placed at the current task's user message.
    // Check if the current user message is ALREADY in history as the LAST user
    // message. We must check it's the last, not just any match: when the same
    // prompt is sent multiple times, an old instance with identical text would
    // falsely satisfy a content-only check. This causes rposition to find the
    // OLD instance as the collapse boundary, keeping the old attempt's entire
    // tool chain as "current interaction" — the model then thinks the task is
    // already done and produces confused responses like "Did you mean to send something?".
    let last_user_msg = recent_history.iter().rev().find(|m| m.role == "user");
    let user_msg_present = last_user_msg.is_some_and(|m| m.content.as_deref() == Some(user_text));
    if !user_msg_present && !user_text.is_empty() {
        let synthetic_turn_id = agent.current_turn_ids.read().await.get(session_id).cloned();
        recent_history.push(Message {
            id: format!("synthetic-user-{}", uuid::Uuid::new_v4()),
            session_id: session_id.to_string(),
            role: "user".to_string(),
            content: Some(user_text.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: chrono::Utc::now(),
            importance: 1.0,
            turn_id: synthetic_turn_id,
            ..Message::runtime_defaults()
        });
        current_user_injected = true;
        info!(
            session_id,
            iteration, "Injected current user message into history (was outside event window)"
        );
    }

    // Merge Pinned + Recent using iterators to avoid cloning the Message structs
    let mut seen_ids: std::collections::HashSet<&String> = std::collections::HashSet::new();

    // Deduplicated, ordered message list
    let deduped_msgs: Vec<&Message> = pinned_memories
        .iter()
        .chain(recent_history.iter())
        .filter(|m| seen_ids.insert(&m.id))
        .collect();

    // Collapse tool intermediates from previous interactions to prevent context bleeding.
    // Without this, old tool call chains (e.g., manage_people calls from a prior question)
    // overwhelm the current question's context and confuse the LLM.
    // Only the current interaction (after the last user message) keeps full tool chains.
    //
    // We drop tool-role messages (results) but keep assistant messages even if they
    // have tool_calls — the JSON conversion below strips orphaned tool_calls and drops
    // content-less assistant messages automatically. This preserves the assistant's
    // reasoning text and budget/timeout summaries as context for the next interaction.
    let identity_preserve_indices: std::collections::HashSet<usize> = deduped_msgs
        .iter()
        .enumerate()
        .filter_map(|(idx, msg)| {
            let content = msg.content.as_deref()?;
            if text_relates_to_critical_identity(content) {
                Some(idx)
            } else {
                None
            }
        })
        .flat_map(|idx| {
            let start = idx.saturating_sub(1);
            let end = (idx + 2).min(deduped_msgs.len().saturating_sub(1));
            start..=end
        })
        .collect();
    // Find the boundary between old and current interactions.
    //
    // Primary: match by `turn_id`. Every message written during this
    // turn was auto-stamped with the same id by `append_message_canonical`,
    // so the first user-role message with the current turn_id marks the
    // boundary — no content inference, no race-condition window where the
    // same text sent twice picks the wrong instance.
    //
    // Fallback: content match against `user_text`. Covers messages
    // persisted before this field existed and any code path that bypasses
    // the auto-stamping layer.
    let current_turn_id: Option<String> =
        agent.current_turn_ids.read().await.get(session_id).cloned();
    let last_user_pos: Option<usize> = current_turn_id
        .as_deref()
        .and_then(|tid| {
            deduped_msgs
                .iter()
                .position(|m| m.role == "user" && m.turn_id.as_deref() == Some(tid))
        })
        .or_else(|| {
            deduped_msgs
                .iter()
                .rposition(|m| m.role == "user" && m.content.as_deref() == Some(user_text))
        });
    if last_user_pos.is_none() {
        warn!(
            session_id,
            iteration,
            total = deduped_msgs.len(),
            "Collapse boundary: last_user_pos=None (should be rare after synthetic injection)"
        );
    }
    let pre_collapse_len = deduped_msgs.len();
    // Find "Prior 1" start: the user message immediately before the boundary.
    // Tool results in [prior_1_start, boundary) are summarized (not dropped).
    // Tool results before prior_1_start are dropped entirely (Prior 2+).
    let prior_1_start: Option<usize> = last_user_pos.and_then(|boundary| {
        deduped_msgs[..boundary]
            .iter()
            .rposition(|m| m.role == "user")
    });
    // Collect message IDs of tool results in the Prior 1 range for summary
    // replacement during JSON conversion. We collect IDs (not indices) because
    // the Vec is rebuilt by the filter below.
    let prior_1_tool_ids: std::collections::HashSet<String> =
        if let (Some(p1_start), Some(boundary)) = (prior_1_start, last_user_pos) {
            deduped_msgs[p1_start..boundary]
                .iter()
                .filter(|m| m.role == "tool")
                .map(|m| m.id.clone())
                .collect()
        } else {
            std::collections::HashSet::new()
        };
    let deduped_msgs: Vec<&Message> = if let Some(boundary) = last_user_pos {
        let p1 = prior_1_start.unwrap_or(boundary);
        deduped_msgs
            .into_iter()
            .enumerate()
            .filter(|(i, m)| {
                if *i >= boundary {
                    true // current interaction: keep everything
                } else if *i >= p1 {
                    // Prior 1 interaction: keep tool results (they will be
                    // summarized during JSON conversion), keep everything else
                    true
                } else {
                    // Prior 2+ interactions: drop tool results only; assistant
                    // messages survive (orphan stripping handles their
                    // tool_calls in JSON conversion)
                    m.role != "tool" || identity_preserve_indices.contains(i)
                }
            })
            .map(|(_, m)| m)
            .collect()
    } else {
        // Current user message not in history yet (race condition or history
        // window too small). Keep the most recent tool results intact — they
        // are very likely from the CURRENT task's previous iterations.
        // Collapse only older tool results to prevent context bloat.
        const KEEP_RECENT_TOOL_RESULTS: usize = 8;
        let tool_positions: Vec<usize> = deduped_msgs
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "tool")
            .map(|(i, _)| i)
            .collect();
        let protect_from = if tool_positions.len() > KEEP_RECENT_TOOL_RESULTS {
            tool_positions[tool_positions.len() - KEEP_RECENT_TOOL_RESULTS]
        } else {
            0
        };
        warn!(
                session_id,
                iteration,
                total_tool_results = tool_positions.len(),
                protect_from,
                "Current user message not in history — using safe collapse (keeping recent tool results)"
            );
        deduped_msgs
            .into_iter()
            .enumerate()
            .filter(|(i, m)| {
                // Keep non-tool messages, recent tool results, and identity-critical ones;
                // collapse old tool results.
                m.role != "tool" || *i >= protect_from || identity_preserve_indices.contains(i)
            })
            .map(|(_, m)| m)
            .collect()
    };
    let collapsed = pre_collapse_len.saturating_sub(deduped_msgs.len());
    if collapsed > 0 || !prior_1_tool_ids.is_empty() {
        info!(
            session_id,
            dropped = collapsed,
            summarized = prior_1_tool_ids.len(),
            "Age-based tool result clearing: dropped Prior 2+ results, summarizing Prior 1 results"
        );
    }
    tracing::debug!(
        session_id,
        iteration,
        stage = "age_collapse",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(
            &project_messages_for_stage_hash(&deduped_msgs),
            user_text,
        ),
        "Build stage pre-boundary fingerprint"
    );

    // Identify old-interaction assistant messages for content truncation.
    // After collapse, recompute the last-user boundary and collect IDs of
    // assistant messages before it — their full text is stale context.
    // Exception: the assistant message immediately before the boundary is exempt
    // from truncation — it typically contains the budget/timeout response with
    // handoff context (activity summary, files read, commands run) that the next
    // interaction needs to avoid re-exploring from scratch.
    // However, when the current message is a clearly NEW task (very different from
    // the prior user message), the old handoff context is harmful — truncate it too.
    // Anchor to the current user message by content (not just any last user message).
    // Without content matching, stray user messages from race conditions can shift
    // the boundary and cause wrong assistant messages to survive truncation.
    let collapse_boundary = deduped_msgs
        .iter()
        .rposition(|m| m.role == "user" && m.content.as_deref() == Some(user_text))
        .or_else(|| deduped_msgs.iter().rposition(|m| m.role == "user"));

    // Adaptive sliding window: keep `window_size` prior conversation pairs.
    // We compute how many old pairs fit within 30% of the available token
    // budget. This naturally adapts: large contexts keep more history, small
    // contexts keep less.
    //
    // Phase 0: capture the length entering the window trim so the trim counter
    // measures window-trim removals alone. The previous counter subtracted from
    // `pre_collapse_len` (captured before age-based collapse), conflating the
    // two and masking whether `keep_from` actually moved.
    let len_before_window_trim = deduped_msgs.len();
    // Phase 0: the window `keep_from` index chosen on the sliding-window path.
    // Defaults to 0 on the no-trim paths (`collapse_boundary == None` or no old
    // user pairs) so the boundary-movement event below fires on every build,
    // not just when a trim happens.
    let mut window_keep_from: usize = 0;
    let deduped_msgs: Vec<&Message> = if let Some(boundary) = collapse_boundary {
        use crate::memory::context_window::estimate_tokens;

        // Identify old user-assistant pair boundaries.
        let old_user_positions: Vec<usize> = deduped_msgs
            .iter()
            .enumerate()
            .filter(|(i, m)| *i < boundary && m.role == "user")
            .map(|(i, _)| i)
            .collect();

        if old_user_positions.is_empty() {
            deduped_msgs
        } else {
            // Build skeleton token estimates for each old pair.
            // A "pair" spans from one user message to the next (or to the boundary).
            let skeleton_pairs: Vec<(usize, usize)> = old_user_positions
                .iter()
                .enumerate()
                .map(|(pair_idx, &user_pos)| {
                    let pair_end = if pair_idx + 1 < old_user_positions.len() {
                        old_user_positions[pair_idx + 1]
                    } else {
                        boundary
                    };
                    let user_tokens =
                        estimate_tokens(deduped_msgs[user_pos].content.as_deref().unwrap_or(""));
                    let assistant_tokens: usize = deduped_msgs[user_pos + 1..pair_end]
                        .iter()
                        .filter(|m| m.role == "assistant")
                        .map(|m| estimate_tokens(m.content.as_deref().unwrap_or("")))
                        .sum();
                    (user_tokens, assistant_tokens)
                })
                .collect();

            // Compute available budget: model context - system prompt - tool defs - pinned memories.
            // System prompt = core (message zero) + task context tail.
            let system_tokens = estimate_tokens(core_prompt) + estimate_tokens(task_context_tail);
            let tools_json = serde_json::to_string(tool_defs).unwrap_or_default();
            let tools_tokens = estimate_tokens(&tools_json);
            let pinned_tokens: usize = pinned_memories
                .iter()
                .map(|m| estimate_tokens(m.content.as_deref().unwrap_or("")))
                .sum();
            let available_budget =
                total_context_budget.saturating_sub(system_tokens + tools_tokens + pinned_tokens);

            let computed_window_size =
                super::sliding_window::calculate_window_size(&skeleton_pairs, available_budget);

            // Idle gap reset: after 2+ hours of inactivity, don't inject stale
            // raw messages into the window — they'd appear as if typed seconds ago.
            // The compaction summary (if available) provides historical context instead.
            let idle_gap_detected = boundary > 0
                && deduped_msgs
                    .get(boundary.saturating_sub(1))
                    .is_some_and(|m| {
                        let now = chrono::Utc::now();
                        now.signed_duration_since(m.created_at).num_seconds() > 7200
                    });
            let window_size = if idle_gap_detected {
                info!(
                    session_id,
                    "Idle gap detected (>2h): resetting sliding window to 0"
                );
                0
            } else {
                computed_window_size
            };

            // Keep the last `window_size` old pairs + everything at/after boundary.
            let keep_from = if window_size == 0 {
                boundary
            } else if old_user_positions.len() > window_size {
                old_user_positions[old_user_positions.len() - window_size]
            } else {
                0
            };
            window_keep_from = keep_from;

            // Phase 0 window-decision log — emitted where persisted `Message`
            // metadata is still available. Ties `keep_from` (and the oldest
            // kept message id) to fetch mechanics: a `keep_from` that moves
            // while the fetch window slides is the prime cache-break suspect.
            let oldest_kept_msg_id: Option<String> =
                deduped_msgs.get(keep_from).map(|m| m.id.clone());
            let boundary_msg_id: Option<String> = deduped_msgs.get(boundary).map(|m| m.id.clone());
            let identity_preserve_bypass = identity_preserve_indices
                .iter()
                .filter(|&&i| i < keep_from)
                .count();
            info!(
                session_id,
                iteration,
                current_turn_id = ?current_turn_id,
                boundary_msg_id = ?boundary_msg_id,
                oldest_fetched_id = ?oldest_fetched_id,
                oldest_kept_msg_id = ?oldest_kept_msg_id,
                keep_from,
                window_size,
                identity_preserve_bypass,
                history_limit,
                fetched_count,
                current_user_injected,
                safe_collapse = last_user_pos.is_none(),
                "Window decision"
            );

            let trimmed: Vec<&Message> = deduped_msgs
                .into_iter()
                .enumerate()
                .filter(|(i, _)| *i >= keep_from || identity_preserve_indices.contains(i))
                .map(|(_, m)| m)
                .collect();
            // Report window-trim removals alone (not conflated with collapse),
            // and log `keep_from` movement explicitly via the oldest kept id.
            if trimmed.len() < len_before_window_trim {
                info!(
                    session_id,
                    window_trimmed = len_before_window_trim - trimmed.len(),
                    keep_from,
                    oldest_kept_msg_id = ?oldest_kept_msg_id,
                    window_size,
                    available_budget,
                    "Adaptive sliding window: trimmed old conversation pairs"
                );
            }
            trimmed
        }
    } else {
        deduped_msgs
    };

    // Phase 0 — explicit window-boundary movement event, emitted on every
    // build (trim or no-trim) so attribution sees a continuous signal. The
    // oldest *kept* message id (first element of the post-trim list) is the
    // robust cache-break signal: `keep_from` is an index into a per-build
    // vector whose composition shifts as the fetch window slides, so a changed
    // index with an unchanged id is benign re-indexing, whereas a changed id is
    // a genuine prefix-cache break. We log both, comparing against the previous
    // build for this session.
    {
        let oldest_kept_msg_id: Option<String> = deduped_msgs.first().map(|m| m.id.clone());
        let mut tracker = agent.window_keep_from_tracker.write().await;
        let previous = tracker.insert(
            session_id.to_string(),
            (window_keep_from, oldest_kept_msg_id.clone()),
        );
        if let Some((old_keep_from, old_oldest_kept_id)) = previous {
            let id_changed = old_oldest_kept_id != oldest_kept_msg_id;
            if id_changed || old_keep_from != window_keep_from {
                info!(
                    session_id,
                    iteration,
                    old_keep_from,
                    new_keep_from = window_keep_from,
                    old_oldest_kept_id = ?old_oldest_kept_id,
                    new_oldest_kept_id = ?oldest_kept_msg_id,
                    oldest_kept_id_changed = id_changed,
                    "Window trim boundary moved"
                );
            }
        }
    }

    tracing::debug!(
        session_id,
        iteration,
        stage = "window_trim",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(
            &project_messages_for_stage_hash(&deduped_msgs),
            user_text,
        ),
        "Build stage pre-boundary fingerprint"
    );

    // Remove duplicate old user messages that have identical content to the
    // current user message. When the same prompt is sent multiple times (e.g.,
    // retrying after a failed response), the old instances with truncated/failed
    // responses confuse the model into thinking the task was already handled.
    // Also remove the assistant response immediately following each duplicate.
    let deduped_msgs: Vec<&Message> = {
        let boundary = deduped_msgs
            .iter()
            .rposition(|m| m.role == "user" && m.content.as_deref() == Some(user_text))
            .or_else(|| deduped_msgs.iter().rposition(|m| m.role == "user"));
        if let Some(boundary) = boundary {
            let mut skip_indices = std::collections::HashSet::new();
            for (i, m) in deduped_msgs.iter().enumerate() {
                if i < boundary && m.role == "user" && m.content.as_deref() == Some(user_text) {
                    skip_indices.insert(i);
                    // Also remove the assistant response immediately after
                    if i + 1 < boundary && deduped_msgs[i + 1].role == "assistant" {
                        skip_indices.insert(i + 1);
                    }
                }
            }
            if !skip_indices.is_empty() {
                info!(
                    session_id,
                    duplicates_removed = skip_indices.len(),
                    "Removed duplicate old user messages matching current prompt"
                );
            }
            deduped_msgs
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !skip_indices.contains(i))
                .map(|(_, m)| m)
                .collect()
        } else {
            deduped_msgs
        }
    };
    tracing::debug!(
        session_id,
        iteration,
        stage = "duplicate_removal",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(
            &project_messages_for_stage_hash(&deduped_msgs),
            user_text,
        ),
        "Build stage pre-boundary fingerprint"
    );

    let execution_checkpoint = if iteration > 1 {
        let current_boundary = deduped_msgs
            .iter()
            .rposition(|m| m.role == "user" && m.content.as_deref() == Some(user_text))
            .or_else(|| deduped_msgs.iter().rposition(|m| m.role == "user"));
        let current_interaction: Vec<&Message> = current_boundary
            .map(|boundary| deduped_msgs.iter().skip(boundary).copied().collect())
            .unwrap_or_default();
        build_execution_checkpoint_message(user_text, completed_tool_calls, &current_interaction)
    } else {
        None
    };

    let old_interaction_assistant_ids: std::collections::HashSet<&str> = if let Some(boundary) =
        deduped_msgs
            .iter()
            .rposition(|m| m.role == "user" && m.content.as_deref() == Some(user_text))
            .or_else(|| deduped_msgs.iter().rposition(|m| m.role == "user"))
    {
        // Find the immediately-prior assistant message (right before boundary).
        // Always exempt it from truncation: it is the single highest-value
        // carryover message when the user sends a terse follow-up like "why?".
        let prior_assistant_id: Option<&str> = (0..boundary)
            .rev()
            .find(|&i| deduped_msgs[i].role == "assistant")
            .map(|i| deduped_msgs[i].id.as_str());

        deduped_msgs
            .iter()
            .enumerate()
            .filter(|(i, m)| {
                *i < boundary
                    && m.role == "assistant"
                    && Some(m.id.as_str()) != prior_assistant_id
                    && !m
                        .content
                        .as_deref()
                        .is_some_and(text_relates_to_critical_identity)
            })
            .map(|(_, m)| m.id.as_str())
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    // Collect tool result ids present in this context window (tool_call_id on tool-role
    // messages with a non-empty tool name). Used to drop assistant tool_calls that would
    // otherwise be orphaned.
    let tool_result_ids: std::collections::HashSet<&str> = deduped_msgs
        .iter()
        .filter(|m| m.role == "tool" && m.tool_name.as_ref().is_some_and(|n| !n.is_empty()))
        .filter_map(|m| m.tool_call_id.as_deref())
        .collect();

    // Build lookup: tool_call_id → (tool_name, arguments_json) from assistant
    // messages. Used to generate 1-line summaries for Prior 1 tool results.
    let tool_call_info: std::collections::HashMap<String, (String, String)> =
        if !prior_1_tool_ids.is_empty() {
            let mut map = std::collections::HashMap::new();
            for m in deduped_msgs.iter() {
                if m.role == "assistant" {
                    if let Some(tc_json) = &m.tool_calls_json {
                        if let Ok(tcs) = serde_json::from_str::<Vec<ToolCall>>(tc_json) {
                            for tc in &tcs {
                                map.insert(tc.id.clone(), (tc.name.clone(), tc.arguments.clone()));
                            }
                        }
                    }
                }
            }
            map
        } else {
            std::collections::HashMap::new()
        };

    let mut messages: Vec<Value> = deduped_msgs
        .iter()
        // Skip tool results with empty/missing tool_name
        .filter(|m| !(m.role == "tool" && m.tool_name.as_ref().is_none_or(|n| n.is_empty())))
        .filter_map(|m| {
            // Truncate stale assistant content from prior interactions.
            // We only shorten long messages to save tokens — we do NOT
            // append marker text (e.g. "[prior turn]") because LLMs tend
            // to echo such markers, producing empty or garbage replies.
            let is_old_assistant = old_interaction_assistant_ids.contains(m.id.as_str());

            // Age-based tool result summarization: Prior 1 tool results get
            // their verbose content replaced with a deterministic 1-line summary.
            // Exception: identity-critical tool results keep their full content.
            let is_identity_critical = m
                .content
                .as_deref()
                .is_some_and(text_relates_to_critical_identity);
            let content =
                if m.role == "tool" && prior_1_tool_ids.contains(&m.id) && !is_identity_critical {
                    let tc_id = m.tool_call_id.as_deref().unwrap_or("");
                    let (tool_name, args_json) = tool_call_info
                        .get(tc_id)
                        .map(|(n, a)| (n.as_str(), a.as_str()))
                        .unwrap_or_else(|| (m.tool_name.as_deref().unwrap_or("unknown"), ""));
                    let result_content = m.content.as_deref().unwrap_or("");
                    Some(super::sliding_window::summarize_tool_result(
                        tool_name,
                        args_json,
                        result_content,
                    ))
                } else if is_old_assistant {
                    m.content.as_ref().map(|c| {
                        if c.len() > MAX_OLD_ASSISTANT_CONTENT_CHARS {
                            let truncated: String =
                                c.chars().take(MAX_OLD_ASSISTANT_CONTENT_CHARS).collect();
                            format!("{}…", truncated)
                        } else {
                            c.clone()
                        }
                    })
                } else {
                    m.content.clone()
                };

            // Prevent stall/failure responses from accumulating as prompt context.
            // These messages are user-visible (stored in history) but poison
            // subsequent turns — the LLM reads its own prior "I failed" messages
            // and gives up without even trying ("learned helplessness").
            if m.role == "assistant"
                && m.tool_calls_json.is_none()
                && content.as_deref().is_some_and(|c| {
                    let t = c.trim_start();
                    t.starts_with("I wasn't able to process that request.")
                        || t.starts_with("I wasn't able to complete this task.")
                        || t.starts_with("I made some progress but wasn't able to fully complete")
                        || t.starts_with("I seem to be stuck on this task.")
                        || t.starts_with("I've reached my processing limit")
                        || t.starts_with("This goal hit its daily processing budget")
                        || t.starts_with("This scheduled goal hit its daily processing budget")
                        || t.starts_with("This scheduled run hit its per-run processing budget")
                        || t.starts_with("I sent the requested file(s), but ran into issues")
                        || t.starts_with(
                            "I completed the main deliverable but wasn't able to finish",
                        )
                })
            {
                return None;
            }

            let mut obj = json!({
                "role": m.role,
                "content": content,
            });
            // For assistant messages with tool_calls, convert from ToolCall struct format
            // to OpenAI wire format and strip any that lack a matching tool result
            if let Some(tc_json) = &m.tool_calls_json {
                if let Ok(tcs) = serde_json::from_str::<Vec<ToolCall>>(tc_json) {
                    let filtered: Vec<Value> = tcs
                        .iter()
                        .filter(|tc| tool_result_ids.contains(tc.id.as_str()))
                        .map(|tc| {
                            let mut val = json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments
                                }
                            });
                            if let Some(ref extra) = tc.extra_content {
                                val["extra_content"] = extra.clone();
                            }
                            val
                        })
                        .collect();
                    if !filtered.is_empty() {
                        obj["tool_calls"] = json!(filtered);
                        if m.content.is_none() {
                            obj["content"] = Value::Null;
                        }
                    } else if m.content.is_none()
                        || m.content.as_deref().is_some_and(|c| c.trim().is_empty())
                    {
                        // Assistant message had tool_calls but all were orphaned,
                        // and no text content — replace with [Action completed] to
                        // prevent dangling user messages (completion compulsion bug)
                        obj["content"] = json!("[Action completed]");
                    }
                }
            }
            if let Some(name) = &m.tool_name {
                if !name.is_empty() {
                    obj["name"] = json!(name);
                }
            }
            if let Some(tcid) = &m.tool_call_id {
                obj["tool_call_id"] = json!(tcid);
            }
            Some(obj)
        })
        .collect();

    // Collapse consecutive orphaned-turn placeholders into a single one. When
    // the agent runs several tool-only iterations in a row, each becomes an
    // identical "[Action completed]" assistant message. Feeding many identical
    // placeholders into the model's context invites repetition/degeneration
    // loops (the model starts regurgitating the placeholder verbatim), so keep
    // only the first of each consecutive run.
    {
        let mut collapsed: Vec<Value> = Vec::with_capacity(messages.len());
        let mut prev_was_placeholder = false;
        for m in messages {
            let is_placeholder = m.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && m.get("content").and_then(|c| c.as_str()) == Some("[Action completed]")
                && m.get("tool_calls").is_none();
            if is_placeholder && prev_was_placeholder {
                continue;
            }
            prev_was_placeholder = is_placeholder;
            collapsed.push(m);
        }
        messages = collapsed;
    }

    // Final safety: drop any tool-role messages that still lack a "name" field
    messages.retain(|m| {
        if m.get("role").and_then(|r| r.as_str()) == Some("tool") {
            let has_name = m
                .get("name")
                .and_then(|n| n.as_str())
                .is_some_and(|n| !n.is_empty());
            if !has_name {
                warn!(
                    "Dropping tool message with missing/empty name: tool_call_id={:?}",
                    m.get("tool_call_id")
                );
            }
            has_name
        } else {
            true
        }
    });

    // Three-pass fixup: merge → drop orphans → merge again.
    fixup_message_ordering(&mut messages);
    // Phase 0 stage hash: history is now converted to final JSON message
    // objects (Prior-1 tool summarization + old-assistant truncation applied
    // in the filter_map above). A flip here with a stable `keep_from` is
    // content mutation, not window-trim movement. `messages` is already
    // complete message objects, so it is hashed directly.
    tracing::debug!(
        session_id,
        iteration,
        stage = "json_conversion",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

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
                && m.get("content").and_then(|c| c.as_str()) == Some(user_text)
        });

        if !has_current_user_msg {
            messages.push(json!({
                "role": "user",
                "content": user_text,
            }));
        }
    }

    // Task boundary marker: when there are multiple user messages in context
    // (i.e., multiple independent tasks in the same chat session), inject a
    // system separator before the current user message so the LLM knows which
    // task is current. Without this, models confuse old tasks with the new one.
    // Injected on ALL iterations (not just early ones) because on iteration 3+
    // old user messages can mislead the model into responding to them instead of
    // the current task — especially after tool calls push the current user message
    // further up the context.
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
                    messages[pos].get("content").and_then(|c| c.as_str()) == Some(user_text)
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
                        "content": "[Current Task] The message below is the user's current request. \
                                    Prior messages are conversation history for context."
                    });
                    messages.insert(current_pos, marker);
                    info!(
                        session_id,
                        iteration,
                        user_messages = user_positions.len(),
                        "Current task marker injected before current user message"
                    );
                }
            }
        }
    }

    // Phase 0 stage hash: the `[Current Task]` marker is a system message
    // inserted just before the current user (boundary) message, so it falls in
    // the pre-boundary region. Marker movement between turns is a known cache
    // boundary; this stage makes it attributable.
    tracing::debug!(
        session_id,
        iteration,
        stage = "current_task_marker",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Guard against context interleaving: if another user message arrived in
    // this session while the agent was processing (race condition between task
    // registration and queuing), it may appear after the current task's tool
    // chain. Such stray user messages confuse the model into responding to them
    // instead of the current task. Remove them.
    {
        let current_task_pos = messages.iter().rposition(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("user")
                && m.get("content").and_then(|c| c.as_str()) == Some(user_text)
        });
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
    let collapsed_tool_errors = super::loop_utils::collapse_repeated_tool_errors(&mut messages);
    if collapsed_tool_errors > 0 {
        info!(
            session_id,
            iteration,
            collapsed_tool_errors,
            "Collapsed repeated tool errors in current interaction"
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
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Context window enforcement: trim messages to fit token budget
    if agent.context_window_config.enabled {
        // Reserve for BOTH message zero (core) and the task context tail.
        let combined_system = format!("{}\n{}", core_prompt, task_context_tail);
        let model_budget = crate::memory::context_window::compute_available_budget(
            model,
            &combined_system,
            tool_defs,
            &agent.context_window_config,
        );
        let policy_budget = policy_bundle.policy.context_budget;
        if agent.policy_config.policy_shadow_mode && !agent.policy_config.policy_enforce {
            info!(
                session_id,
                iteration, model_budget, policy_budget, "Context budget shadow comparison"
            );
        }
        let effective_budget = if agent.policy_config.policy_enforce {
            // Never exceed the model's budget; policy config can be mis-set.
            policy_budget.min(model_budget)
        } else {
            model_budget
        };
        messages = crate::memory::context_window::fit_messages_with_source_quotas(
            messages,
            effective_budget,
        );
    }
    // Phase 0 stage hash: history fitting. `fit_messages_with_source_quotas`
    // can drop history under budget pressure. Pillar A retired the summary
    // insertion from the fitter — the summary now lives only in the task
    // context tail — so this stage now captures the history-trim effect alone.
    tracing::debug!(
        session_id,
        iteration,
        stage = "history_fitting",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Empty-response recovery: on retry, clear conversational history to avoid
    // repeatedly sending a poisoned context to the provider (Gemini in particular
    // can get "stuck" returning empty candidates for a given session history).
    if empty_response_retry_pending && !is_trigger_session(session_id) {
        let before = messages.len();
        messages = build_empty_response_retry_messages(&messages, user_text);
        info!(
            session_id,
            iteration,
            before,
            after = messages.len(),
            "Empty-response recovery: reduced history while preserving immediate parent context"
        );
    }

    // Pillar A: insert the per-task context TAIL immediately BEFORE the current
    // user message (boundary − 1). The tail is a single `role:"system"` message
    // whose content starts with `TASK_CONTEXT_TAIL_MARKER`; the provider-call
    // fingerprint locates it by that marker. The session summary, current
    // date/time, session context, query-ranked memory, matched skill bodies, and
    // resume checkpoint all live INSIDE this string (compiled once per task in
    // bootstrap and reused byte-identically across the within-task loop).
    //
    // This insertion happens BEFORE message zero is inserted, so the boundary is
    // located against the current `messages` (no leading system prompt yet).
    if !task_context_tail.is_empty() {
        let tail_insert_pos = messages
            .iter()
            .rposition(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content").and_then(|c| c.as_str()) == Some(user_text)
            })
            .unwrap_or(messages.len());
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

    // Phase 0 stage hash: the task context tail sits at boundary − 1 (inside the
    // pre-boundary region). Message zero is the core prompt;
    // `stage_pre_boundary_hash` skips that leading system message, so the tail is
    // included in the pre-boundary hash and tail churn is attributable. (The
    // session-summary stage was retired with the index-1 summary insertion; the
    // provider-call `tail_hash` covers tail attribution at the call boundary.)
    tracing::debug!(
        session_id,
        iteration,
        stage = "context_tail",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
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
        full_payload_hash = %super::prefix_fingerprint::hash_canonical(&serde_json::Value::Array(messages.clone())),
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
                    || m.get("content").and_then(|c| c.as_str()) == Some(user_text)
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

    // Final enforcement must happen after every prompt component has been inserted.
    // Earlier trimming cannot account for execution checkpoints and one-shot directives.
    if agent.context_window_config.enabled {
        let message_tokens = crate::memory::context_window::estimate_tokens(
            &serde_json::to_string(&messages).unwrap_or_default(),
        );
        let final_tool_budget = total_context_budget.saturating_sub(
            message_tokens + RESPONSE_RESERVE_TOKENS + TOKEN_ESTIMATE_SAFETY_MARGIN,
        );
        let final_tool_defs = crate::memory::context_window::fit_tool_definitions_to_budget(
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
                before_tool_tokens = crate::memory::context_window::estimate_tool_definition_tokens(
                    &effective_tool_defs
                ),
                after_tool_tokens = crate::memory::context_window::estimate_tool_definition_tokens(
                    &final_tool_defs
                ),
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

        // Estimate tokens: ~4 chars per token for English text
        let messages_json = serde_json::to_string(&messages).unwrap_or_default();
        let est_msg_tokens = messages_json.len() / 4;
        let est_tool_tokens =
            crate::memory::context_window::estimate_tool_definition_tokens(tool_defs);
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
    Agent::sort_tool_definitions_by_name(&mut effective_tool_defs);

    let est_input_tokens = {
        let messages_json = serde_json::to_string(&messages).unwrap_or_default();
        let est_msg_tokens = messages_json.len() / 4;
        let est_tool_tokens =
            crate::memory::context_window::estimate_tool_definition_tokens(&effective_tool_defs);
        (est_msg_tokens + est_tool_tokens) as u32
    };

    Ok(MessageBuildData {
        messages,
        tool_defs: effective_tool_defs,
        est_input_tokens,
    })
}

#[cfg(test)]
mod tests {
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

    fn tool_msg(name: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            role: "tool".to_string(),
            content: Some(content.to_string()),
            tool_call_id: Some(format!("tool-call-{}", uuid::Uuid::new_v4())),
            tool_name: Some(name.to_string()),
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        }
    }

    #[test]
    fn empty_retry_preserves_parent_pair_and_current_user() {
        let messages = vec![
            json!({"role": "user", "content": "can you clear cache using drush?"}),
            json!({"role": "assistant", "content": "I can see updates available. Should I proceed with updating these?"}),
            json!({"role": "user", "content": "yes, update them"}),
        ];
        let recovered = build_empty_response_retry_messages(&messages, "yes, update them");
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0]["role"], "user");
        assert_eq!(recovered[1]["role"], "assistant");
        assert_eq!(recovered[2]["role"], "user");
        assert_eq!(recovered[2]["content"].as_str(), Some("yes, update them"));
    }

    #[test]
    fn empty_retry_falls_back_to_current_user_when_no_history() {
        let messages = vec![json!({"role": "user", "content": "help"})];
        let recovered = build_empty_response_retry_messages(&messages, "help");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0]["role"], "user");
        assert_eq!(recovered[0]["content"].as_str(), Some("help"));
    }

    #[tokio::test]
    async fn sliding_window_retains_pairs_that_fit_budget() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Older task"))
            .await
            .expect("append oldest user");
        harness
            .state
            .append_message(&msg("assistant", "Older answer"))
            .await
            .expect("append oldest assistant");
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
        harness
            .state
            .append_message(&msg("user", "Why?"))
            .await
            .expect("append current user");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Why?",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");
        let serialized = serde_json::to_string(&built.messages).expect("serialize messages");

        // Adaptive sliding window keeps all pairs that fit within 30% of the
        // token budget. Both small pairs easily fit, so all are retained.
        assert!(
            serialized.contains("blog.aidaemon.ai"),
            "immediately prior user turn should be retained: {}",
            serialized
        );
        assert!(
            serialized.contains("Which posts should I update?"),
            "immediately prior assistant turn should be retained: {}",
            serialized
        );
        assert!(
            serialized.contains("Older task"),
            "older pair within budget should be retained by sliding window: {}",
            serialized
        );
        assert!(
            serialized.contains("Older answer"),
            "older assistant within budget should be retained: {}",
            serialized
        );
        assert!(
            serialized.contains("Why?"),
            "current user message should remain present: {}",
            serialized
        );
    }

    /// Phase 0: the window-decision path records the session's `keep_from` and
    /// oldest-kept message id so a later build can emit an explicit
    /// `Window trim boundary moved` event. This asserts the tracker plumbing,
    /// not the log output itself.
    #[tokio::test]
    async fn window_decision_records_keep_from_tracker_for_session() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Older task"))
            .await
            .expect("append oldest user");
        harness
            .state
            .append_message(&msg("assistant", "Older answer"))
            .await
            .expect("append oldest assistant");
        harness
            .state
            .append_message(&msg("user", "Why?"))
            .await
            .expect("append current user");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "tracker-session",
            iteration: 1,
            user_text: "Why?",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        let tracker = harness.agent.window_keep_from_tracker.read().await;
        assert!(
            tracker.contains_key("tracker-session"),
            "window-decision path should record a keep_from entry for the session"
        );
        // A second build with identical inputs must not panic and must keep a
        // single entry per session (insert-overwrite, not accumulate).
        drop(tracker);
        let mut pending_system_messages2 = Vec::new();
        let mut ctx2 = MessageBuildCtx {
            session_id: "tracker-session",
            iteration: 2,
            user_text: "Why?",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages2,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx2,
        )
        .await
        .expect("second message build");
        let tracker = harness.agent.window_keep_from_tracker.read().await;
        assert_eq!(
            tracker.len(),
            1,
            "tracker should hold one entry per session, not accumulate"
        );
    }

    #[tokio::test]
    async fn later_iterations_include_execution_checkpoint_after_tool_progress() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Find the system details and summarize them."))
            .await
            .expect("append user");
        harness
            .state
            .append_message(&tool_msg(
                "system_info",
                "OS: macOS 15.0\nMemory: 16 GB\nHostname: dev-machine",
            ))
            .await
            .expect("append tool");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let completed_tool_calls = vec!["system_info({})".to_string()];

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Find the system details and summarize them.",
            completed_tool_calls: &completed_tool_calls,
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");
        let serialized = serde_json::to_string(&built.messages).expect("serialize messages");

        assert!(
            serialized.contains("EXECUTION CHECKPOINT"),
            "later iterations should carry a live execution checkpoint: {}",
            serialized
        );
        assert!(
            serialized.contains("Find the system details and summarize them."),
            "checkpoint should restate the active request: {}",
            serialized
        );
        assert!(
            serialized.contains("system_info"),
            "checkpoint should include completed tool/evidence context: {}",
            serialized
        );
        assert!(
            serialized.contains("Do NOT reset into a generic availability reply"),
            "checkpoint should explicitly block idle reset replies: {}",
            serialized
        );
    }

    #[tokio::test]
    async fn later_iterations_preserve_system_prompt_prefix_without_duplicate_guidance() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Inspect the repository."))
            .await
            .expect("append user");

        let system_prompt =
            "## Identity\nStable identity.\n\n## Tools\nVerbose tool guidance.\n\n## Behavior\nBe precise.";
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let tool_defs: Vec<Value> = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut first_pending_system_messages = Vec::new();
        let mut first_ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Inspect the repository.",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: system_prompt,
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut first_pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let first = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut first_ctx,
        )
        .await
        .expect("first message build");

        let mut second_pending_system_messages = Vec::new();
        let mut second_ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Inspect the repository.",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: system_prompt,
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut second_pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let second = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut second_ctx,
        )
        .await
        .expect("second message build");
        let first_system = first.messages[0]["content"]
            .as_str()
            .expect("first system content");
        let second_system = second.messages[0]["content"]
            .as_str()
            .expect("second system content");

        assert_eq!(first_system, system_prompt);
        assert_eq!(
            second_system, first_system,
            "message zero must remain byte-identical for prompt-cache reuse"
        );
        for (iteration, built) in [(1, &first), (2, &second)] {
            assert!(
                built.messages.iter().skip(1).all(|message| {
                    !message
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|content| content.contains("Stable identity."))
                }),
                "iteration {iteration} must not duplicate the system prompt later in the request"
            );
        }
    }

    /// After 2+ hours idle, the sliding window should NOT include old pairs —
    /// only the compaction summary provides historical context.
    #[tokio::test]
    async fn idle_gap_resets_sliding_window_to_zero() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;
        use chrono::Duration as ChronoDuration;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");

        // Insert old messages with timestamps > 2 hours ago.
        let old_time = Utc::now() - ChronoDuration::hours(3);
        let old_user = Message {
            created_at: old_time,
            ..msg("user", "Old stale question from 3 hours ago")
        };
        let old_assistant = Message {
            created_at: old_time,
            ..msg("assistant", "Old stale answer from 3 hours ago")
        };
        harness
            .state
            .append_message(&old_user)
            .await
            .expect("append old user");
        harness
            .state
            .append_message(&old_assistant)
            .await
            .expect("append old assistant");

        // Insert current user message (now).
        harness
            .state
            .append_message(&msg("user", "Fresh question now"))
            .await
            .expect("append current user");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Fresh question now",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");
        let serialized = serde_json::to_string(&built.messages).expect("serialize messages");

        // Old stale messages should NOT be present after idle gap reset.
        assert!(
            !serialized.contains("Old stale question from 3 hours ago"),
            "idle gap should reset window to 0, removing old stale pairs: {}",
            serialized
        );
        assert!(
            !serialized.contains("Old stale answer from 3 hours ago"),
            "idle gap should reset window to 0, removing old stale assistant: {}",
            serialized
        );

        // Current user message must still be present.
        assert!(
            serialized.contains("Fresh question now"),
            "current user message should always be present: {}",
            serialized
        );
    }

    /// Pillar A: the session summary now travels INSIDE the per-task context
    /// tail (compiled in bootstrap). Message-build no longer takes a summary
    /// argument; instead it inserts the tail (containing `[Session Summary]`) at
    /// boundary − 1 as a single system message. This test passes the summary via
    /// `task_context_tail` and asserts it lands in the tail message and NOT at a
    /// separate index-1 message.
    #[tokio::test]
    async fn session_summary_travels_inside_task_context_tail() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");

        harness
            .state
            .append_message(&msg("user", "Current question"))
            .await
            .expect("append user");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let tool_defs: Vec<Value> = Vec::new();
        let tail = format!(
            "{TASK_CONTEXT_TAIL_MARKER}\n\n[Session Summary]\nUser previously asked about deploying a blog. Config was created."
        );
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Current question",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: &tail,
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        // The summary lives inside the tail message (starts with the marker).
        let tail_msg = built.messages.iter().find(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("system")
                && m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
        });
        let tail_content = tail_msg
            .and_then(|m| m["content"].as_str())
            .expect("tail message must be present");
        assert!(
            tail_content.contains("[Session Summary]"),
            "summary must live inside the task context tail: {tail_content}"
        );
        assert!(
            tail_content.contains("deploying a blog"),
            "summary content must be present in the tail: {tail_content}"
        );

        // No separate index-1 [Session Summary] message exists anymore.
        let summary_only_messages = built
            .messages
            .iter()
            .filter(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("[Session Summary]"))
            })
            .count();
        assert_eq!(
            summary_only_messages, 1,
            "summary must appear exactly once (inside the tail), not as a separate message"
        );
    }

    #[tokio::test]
    async fn small_context_model_compacts_tool_schemas_without_dropping_tools() {
        use crate::execution_policy::PolicyBundle;
        use crate::memory::context_window::{estimate_tokens, estimate_tool_definition_tokens};
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .agent
            .context_window_config
            .model_budgets
            .insert("gemma-4-26b".to_string(), 16_384);
        harness
            .state
            .append_message(&msg("user", "List all available tools."))
            .await
            .expect("append user");

        let verbose =
            "Detailed operational guidance for selecting and safely using this tool. ".repeat(300);
        let tool_defs: Vec<Value> = (0..20)
            .map(|idx| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("tool_{idx}"),
                        "description": verbose,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": verbose
                                },
                                "mode": {
                                    "type": "string",
                                    "description": verbose,
                                    "enum": ["read", "write"]
                                }
                            },
                            "required": ["path"],
                            "additionalProperties": false
                        }
                    }
                })
            })
            .collect();
        assert!(estimate_tool_definition_tokens(&tool_defs) > 16_384);

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "List all available tools.",
            completed_tool_calls: &[],
            model: "gemma-4-26b",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        assert_eq!(built.tool_defs.len(), tool_defs.len());
        // Pillar A: the emitted roster is now name-sorted (lexicographic), so
        // assert the SET of tool names is preserved and parameter contracts are
        // intact — not a positional numeric order.
        let got_names: std::collections::HashSet<String> = built
            .tool_defs
            .iter()
            .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
            .collect();
        let expected_names: std::collections::HashSet<String> = (0..tool_defs.len())
            .map(|idx| format!("tool_{idx}"))
            .collect();
        assert_eq!(got_names, expected_names, "all tools must be preserved");
        for tool in &built.tool_defs {
            assert_eq!(
                tool["function"]["parameters"]["properties"]["mode"]["enum"],
                json!(["read", "write"])
            );
        }
        // Confirm the final order is name-sorted (the authoritative final sort).
        let ordered_names: Vec<&str> = built
            .tool_defs
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        let mut sorted = ordered_names.clone();
        sorted.sort();
        assert_eq!(ordered_names, sorted, "tool order must be name-sorted");

        let message_tokens =
            estimate_tokens(&serde_json::to_string(&built.messages).expect("serialize messages"));
        let tool_tokens = estimate_tool_definition_tokens(&built.tool_defs);
        assert!(
            message_tokens + tool_tokens + 1_536 <= 16_384,
            "request estimate should fit Gemma context: messages={message_tokens}, tools={tool_tokens}"
        );
    }

    #[tokio::test]
    async fn small_context_model_rechecks_budget_after_final_prompt_assembly() {
        use crate::execution_policy::PolicyBundle;
        use crate::memory::context_window::{estimate_tokens, estimate_tool_definition_tokens};
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .agent
            .context_window_config
            .model_budgets
            .insert("gemma-4-26b".to_string(), 16_384);
        harness
            .state
            .append_message(&msg("user", "Can you test your tools?"))
            .await
            .expect("append user");
        harness
            .state
            .append_message(&tool_msg(
                "system_info",
                "OS: macOS\nMemory: 64 GB\nHostname: workstation",
            ))
            .await
            .expect("append tool");

        let verbose = "Detailed parameter guidance for local agent tool execution. ".repeat(40);
        let tool_defs: Vec<Value> = (0..38)
            .map(|idx| {
                let properties: serde_json::Map<String, Value> = (0..8)
                    .map(|prop_idx| {
                        (
                            format!("parameter_{prop_idx}"),
                            json!({
                                "type": "string",
                                "description": verbose,
                                "enum": ["one", "two", "three"]
                            }),
                        )
                    })
                    .collect();
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("tool_{idx}"),
                        "description": verbose,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": ["parameter_0"],
                            "additionalProperties": false
                        }
                    }
                })
            })
            .collect();

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let mut pending_system_messages = vec![SystemDirective::FreshConversationContext];
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let completed_tool_calls = vec!["system_info({})".to_string()];
        let system_prompt = "Root agent operating guidance and tool policy. ".repeat(650);

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Can you test your tools?",
            completed_tool_calls: &completed_tool_calls,
            model: "gemma-4-26b",
            core_prompt: &system_prompt,
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        assert_eq!(built.tool_defs.len(), tool_defs.len());
        let message_tokens =
            estimate_tokens(&serde_json::to_string(&built.messages).expect("serialize messages"));
        let tool_tokens = estimate_tool_definition_tokens(&built.tool_defs);
        assert!(
            message_tokens + tool_tokens + RESPONSE_RESERVE_TOKENS <= 16_384,
            "final assembled request should fit: messages={message_tokens}, tools={tool_tokens}"
        );
    }

    // ---- Pillar A Task 6: payload assembly tests ----

    /// Test 1: exactly one system message starts with TASK_CONTEXT_TAIL_MARKER,
    /// positioned immediately BEFORE the current user message (boundary − 1).
    /// Test 2: no standalone `[Session Summary]` message; the summary appears
    /// ONLY inside the tail.
    /// Test 3: message zero equals the core bytes exactly (no volatile suffix).
    #[tokio::test]
    async fn tail_precedes_current_turn_and_summary_lives_only_in_tail() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Old question"))
            .await
            .expect("append old user");
        harness
            .state
            .append_message(&msg("assistant", "Old answer"))
            .await
            .expect("append old assistant");
        harness
            .state
            .append_message(&msg("user", "Current question"))
            .await
            .expect("append current user");

        let core = "You are aidaemon. CORE PROMPT BODY.";
        let tail = format!(
            "{TASK_CONTEXT_TAIL_MARKER}\n\n[Session Summary]\nUser deploying a blog.\n\n[Current Date & Time]\nMonday"
        );

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Current question",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        // Test 3: message zero equals the core bytes exactly.
        assert_eq!(
            built.messages[0]["content"].as_str(),
            Some(core),
            "message zero must be the core prompt bytes with no volatile suffix"
        );

        // Test 1: exactly one tail message; it precedes the current user message.
        let tail_positions: Vec<usize> = built
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.get("role").and_then(|r| r.as_str()) == Some("system")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(tail_positions.len(), 1, "exactly one tail message expected");
        let tail_pos = tail_positions[0];
        let current_user_pos = built
            .messages
            .iter()
            .rposition(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content").and_then(|c| c.as_str()) == Some("Current question")
            })
            .expect("current user message present");
        assert_eq!(
            tail_pos + 1,
            current_user_pos,
            "tail must sit immediately before the current user message (boundary − 1)"
        );

        // Test 2: no standalone `[Session Summary]` message; summary only in tail.
        assert!(
            !built.messages[1]["content"]
                .as_str()
                .unwrap_or("")
                .starts_with("[Session Summary]"),
            "index 1 must not be a standalone session-summary message"
        );
        let summary_msgs = built
            .messages
            .iter()
            .filter(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("[Session Summary]"))
            })
            .count();
        assert_eq!(
            summary_msgs, 1,
            "summary appears exactly once, inside the tail"
        );
    }

    /// Test 4: within-task tail reuse — two consecutive build iterations of the
    /// same task produce a byte-identical tail message.
    #[tokio::test]
    async fn within_task_tail_reuse_is_byte_identical() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Same task"))
            .await
            .expect("append user");

        let core = "CORE";
        let tail = format!("{TASK_CONTEXT_TAIL_MARKER}\n\n[Current Date & Time]\nFixed timestamp");
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let tool_defs: Vec<Value> = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let extract_tail = |built: &MessageBuildData| -> String {
            built
                .messages
                .iter()
                .find(|m| {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
                })
                .and_then(|m| m["content"].as_str())
                .expect("tail present")
                .to_string()
        };

        let mut p1 = Vec::new();
        let mut ctx1 = MessageBuildCtx {
            session_id: "reuse-session",
            iteration: 1,
            user_text: "Same task",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut p1,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let built1 = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx1,
        )
        .await
        .expect("build 1");

        let mut p2 = Vec::new();
        let mut ctx2 = MessageBuildCtx {
            session_id: "reuse-session",
            iteration: 2,
            user_text: "Same task",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut p2,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let built2 = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx2,
        )
        .await
        .expect("build 2");

        assert_eq!(
            extract_tail(&built1),
            extract_tail(&built2),
            "tail must be byte-identical across within-task iterations"
        );
    }

    /// Test 5: the final emitted tool order is name-sorted even when the input
    /// roster is unsorted — proving the sort happens as the final op before
    /// MessageBuildData, after any mutation.
    #[tokio::test]
    async fn final_tool_order_is_name_sorted_after_mutations() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Do work"))
            .await
            .expect("append user");

        // Deliberately unsorted roster (zebra, alpha, mango).
        let tool_defs: Vec<Value> = ["zebra_tool", "alpha_tool", "mango_tool"]
            .iter()
            .map(|name| {
                json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "x",
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
                    }
                })
            })
            .collect();

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let pinned_memories: Vec<Message> = Vec::new();
        let mut pending = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "sort-session",
            iteration: 1,
            user_text: "Do work",
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "CORE",
            task_context_tail: "",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("build");

        let names: Vec<&str> = built
            .tool_defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str())
            .collect();
        assert_eq!(
            names,
            vec!["alpha_tool", "mango_tool", "zebra_tool"],
            "final tool_defs must be name-sorted"
        );
    }
}
