use super::recall_guardrails::text_relates_to_critical_identity;
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ConversationSummary;

pub(super) struct MessageBuildCtx<'a> {
    pub session_id: &'a str,
    pub iteration: usize,
    pub user_text: &'a str,
    pub completed_tool_calls: &'a [String],
    pub model: &'a str,
    pub system_prompt: &'a str,
    pub pinned_memories: &'a [Message],
    pub tool_defs: &'a [Value],
    pub policy_bundle: &'a PolicyBundle,
    pub session_summary: &'a Option<ConversationSummary>,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub empty_response_retry_pending: bool,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    /// Sticky flag computed once at the start of the agent loop from
    /// `TurnContext::followup_mode`. Prevents mid-loop reclassification
    /// that causes context collapse when history shifts between iterations.
    pub is_new_task: bool,
}

pub(super) struct MessageBuildData {
    pub messages: Vec<Value>,
}

const EMPTY_RETRY_MAX_PARENT_CHARS: usize = 800;
const EXECUTION_CHECKPOINT_MAX_REQUEST_CHARS: usize = 240;
const EXECUTION_CHECKPOINT_MAX_ACTIVITY_CHARS: usize = 900;
const EXECUTION_CHECKPOINT_MAX_EVIDENCE_CHARS: usize = 500;

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

impl Agent {
    pub(super) async fn run_message_build_phase(
        &self,
        ctx: &mut MessageBuildCtx<'_>,
    ) -> anyhow::Result<MessageBuildData> {
        let session_id = ctx.session_id;
        let iteration = ctx.iteration;
        let user_text = ctx.user_text;
        let completed_tool_calls = ctx.completed_tool_calls;
        let model = ctx.model;
        let system_prompt = ctx.system_prompt;
        let pinned_memories = ctx.pinned_memories;
        let tool_defs = ctx.tool_defs;
        let policy_bundle = ctx.policy_bundle;
        let session_summary = ctx.session_summary;
        let pending_system_messages = &mut *ctx.pending_system_messages;
        let empty_response_retry_pending = ctx.empty_response_retry_pending;
        let status_tx = ctx.status_tx;

        // Fetch recent history from canonical event stream.
        // Base limit of 40 queries (120 events), scaled up for long-running tasks
        // so that early tool calls from the current task are not pushed out of the
        // window by their own later iterations.  Each iteration generates ~3
        // messages (assistant, tool result(s), sometimes parallel calls), so
        // iteration*3 covers the current task plus old-pair trimming removes the
        // rest.  Capped at 120 to avoid loading entire sessions.
        let history_limit = 40_usize.max(iteration.saturating_mul(3).min(120));
        let mut recent_history = self.load_recent_history(session_id, history_limit).await?;

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
        let user_msg_present =
            last_user_msg.is_some_and(|m| m.content.as_deref() == Some(user_text));
        if !user_msg_present && !user_text.is_empty() {
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
                ..Message::runtime_defaults()
            });
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
        // If the current user_text is already in the history, use its position.
        // Otherwise, the DB write hasn't committed yet (race condition) — treat
        // ALL loaded messages as "old" so we collapse their tool results.
        let last_user_pos: Option<usize> = deduped_msgs
            .iter()
            .rposition(|m| m.role == "user" && m.content.as_deref() == Some(user_text));
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
        // Instead of branching on is_new_task (brittle classifier), we compute how
        // many old pairs fit within 30% of the available token budget. This naturally
        // adapts: large contexts keep more history, small contexts keep less.
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
                        let user_tokens = estimate_tokens(
                            deduped_msgs[user_pos].content.as_deref().unwrap_or(""),
                        );
                        let assistant_tokens: usize = deduped_msgs[user_pos + 1..pair_end]
                            .iter()
                            .filter(|m| m.role == "assistant")
                            .map(|m| estimate_tokens(m.content.as_deref().unwrap_or("")))
                            .sum();
                        (user_tokens, assistant_tokens)
                    })
                    .collect();

                // Compute available budget: model context - system prompt - tool defs - pinned memories.
                let system_tokens = estimate_tokens(system_prompt);
                let tools_json = serde_json::to_string(tool_defs).unwrap_or_default();
                let tools_tokens = estimate_tokens(&tools_json);
                let pinned_tokens: usize = pinned_memories
                    .iter()
                    .map(|m| estimate_tokens(m.content.as_deref().unwrap_or("")))
                    .sum();
                // Use a reasonable default context window (128k tokens).
                // The exact model budget doesn't matter much — we only use 30% of
                // the remainder, so over-estimating is safe.
                let model_budget = 128_000usize;
                let available_budget =
                    model_budget.saturating_sub(system_tokens + tools_tokens + pinned_tokens);

                let window_size =
                    super::sliding_window::calculate_window_size(&skeleton_pairs, available_budget);

                // Keep the last `window_size` old pairs + everything at/after boundary.
                let keep_from = if window_size == 0 {
                    boundary
                } else if old_user_positions.len() > window_size {
                    old_user_positions[old_user_positions.len() - window_size]
                } else {
                    0
                };

                let trimmed: Vec<&Message> = deduped_msgs
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| *i >= keep_from || identity_preserve_indices.contains(i))
                    .map(|(_, m)| m)
                    .collect();
                if trimmed.len() < pre_collapse_len {
                    info!(
                        session_id,
                        old_pairs_trimmed = pre_collapse_len - trimmed.len(),
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

        let execution_checkpoint = if iteration > 1 {
            let current_boundary = deduped_msgs
                .iter()
                .rposition(|m| m.role == "user" && m.content.as_deref() == Some(user_text))
                .or_else(|| deduped_msgs.iter().rposition(|m| m.role == "user"));
            let current_interaction: Vec<&Message> = current_boundary
                .map(|boundary| deduped_msgs.iter().skip(boundary).copied().collect())
                .unwrap_or_default();
            build_execution_checkpoint_message(
                user_text,
                completed_tool_calls,
                &current_interaction,
            )
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
                                    map.insert(
                                        tc.id.clone(),
                                        (tc.name.clone(), tc.arguments.clone()),
                                    );
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
                let content = if m.role == "tool"
                    && prior_1_tool_ids.contains(&m.id)
                    && !is_identity_critical
                {
                    let tc_id = m.tool_call_id.as_deref().unwrap_or("");
                    let (tool_name, args_json) = tool_call_info
                        .get(tc_id)
                        .map(|(n, a)| (n.as_str(), a.as_str()))
                        .unwrap_or_else(|| {
                            (
                                m.tool_name.as_deref().unwrap_or("unknown"),
                                "",
                            )
                        });
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
                            || t.starts_with(
                                "I made some progress but wasn't able to fully complete",
                            )
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
                            || m.content
                                .as_deref()
                                .is_some_and(|c| c.trim().is_empty())
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

        // Uploaded-file tasks should not inherit unrelated topical turns. The
        // artifact marker is the anchor for the new task, so keep system
        // guidance plus the current user request and drop earlier user/assistant
        // exchanges when the turn is clearly a fresh uploaded-file request.
        if ctx.is_new_task && user_text.contains("[File received:") {
            if let Some(current_user_pos) = messages.iter().rposition(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content").and_then(|c| c.as_str()) == Some(user_text)
            }) {
                messages = messages
                    .into_iter()
                    .enumerate()
                    .filter_map(|(idx, message)| {
                        let role = message.get("role").and_then(|r| r.as_str());
                        (idx >= current_user_pos || role == Some("system")).then_some(message)
                    })
                    .collect();
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
                    let has_different_task = prev_user_content.as_deref() != Some(user_text)
                        && prev_user_content.is_some();
                    if has_different_task {
                        let marker = json!({
                            "role": "system",
                            "content": "[TASK BOUNDARY] The user has started a NEW, UNRELATED task below. \
                                        Previous tasks in this session are COMPLETED — do NOT \
                                        reference, revisit, or repeat them. Focus EXCLUSIVELY \
                                        on the new request. If the new task asks to create files, \
                                        search the web, write code, or perform any action, you MUST \
                                        use the appropriate tools — do NOT answer with information \
                                        from previous tasks instead."
                        });
                        messages.insert(current_pos, marker);
                        info!(
                            session_id,
                            iteration,
                            user_messages = user_positions.len(),
                            "Task boundary marker injected before current user message"
                        );
                    }
                }
            }
        }

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

        // Context window enforcement: trim messages to fit token budget
        if self.context_window_config.enabled {
            let model_budget = crate::memory::context_window::compute_available_budget(
                model,
                system_prompt,
                tool_defs,
                &self.context_window_config,
            );
            let policy_budget = policy_bundle.policy.context_budget;
            if self.policy_config.policy_shadow_mode && !self.policy_config.policy_enforce {
                info!(
                    session_id,
                    iteration, model_budget, policy_budget, "Context budget shadow comparison"
                );
            }
            let effective_budget = if self.policy_config.policy_enforce {
                // Never exceed the model's budget; policy config can be mis-set.
                policy_budget.min(model_budget)
            } else {
                model_budget
            };
            messages = crate::memory::context_window::fit_messages_with_source_quotas(
                messages,
                effective_budget,
                session_summary.as_ref().map(|s| s.summary.as_str()),
            );
        }

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

        // Prompt shaping:
        // - Iterations >1: use compact tool-loop prompt to reduce repeated token overhead.
        let effective_system_prompt = if iteration > 1 {
            let style = match policy_bundle.policy.model_profile {
                ModelProfile::Cheap => ToolLoopPromptStyle::Lite,
                ModelProfile::Balanced | ModelProfile::Strong => ToolLoopPromptStyle::Standard,
            };
            build_tool_loop_system_prompt(system_prompt, style)
        } else {
            system_prompt.to_string()
        };

        messages.insert(
            0,
            json!({
                "role": "system",
                "content": effective_system_prompt,
            }),
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
            let tools_json = serde_json::to_string(tool_defs).unwrap_or_default();
            let est_msg_tokens = messages_json.len() / 4;
            let est_tool_tokens = tools_json.len() / 4;
            let est_total_tokens = est_msg_tokens + est_tool_tokens;
            let est_msg_tokens_u64 = est_msg_tokens as u64;
            let est_tool_tokens_u64 = est_tool_tokens as u64;
            let est_total_tokens_u64 = est_total_tokens as u64;
            let est_tool_share_bps = if est_total_tokens_u64 > 0 {
                est_tool_tokens_u64.saturating_mul(10_000) / est_total_tokens_u64
            } else {
                0
            };

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
                est_tool_share_pct = est_tool_share_bps as f64 / 100.0,
                msg_count = messages.len(),
                msgs = ?summary,
                "Context before LLM call"
            );
        }

        Ok(MessageBuildData { messages })
    }
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
        use crate::traits::{ConversationSummary, MessageStore};

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
        let session_summary: Option<ConversationSummary> = None;
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Why?",
            completed_tool_calls: &[],
            model: "mock-model",
            system_prompt: "You are a helpful test assistant.",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            session_summary: &session_summary,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
            is_new_task: true,
        };

        let built = harness
            .agent
            .run_message_build_phase(&mut ctx)
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

    #[tokio::test]
    async fn later_iterations_include_execution_checkpoint_after_tool_progress() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::{ConversationSummary, MessageStore};

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
        let session_summary: Option<ConversationSummary> = None;
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let completed_tool_calls = vec!["system_info({})".to_string()];

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Find the system details and summarize them.",
            completed_tool_calls: &completed_tool_calls,
            model: "mock-model",
            system_prompt: "You are a helpful test assistant.",
            pinned_memories: &pinned_memories,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            session_summary: &session_summary,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
            is_new_task: false,
        };

        let built = harness
            .agent
            .run_message_build_phase(&mut ctx)
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
}
