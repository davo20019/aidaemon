use super::recall_guardrails::text_relates_to_critical_identity;
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ConversationSummary;

pub(super) struct MessageBuildCtx<'a> {
    pub session_id: &'a str,
    pub iteration: usize,
    pub user_text: &'a str,
    pub model: &'a str,
    pub system_prompt: &'a str,
    pub consultant_pass_active: bool,
    pub pinned_memories: &'a [Message],
    pub tool_defs: &'a [Value],
    pub policy_bundle: &'a PolicyBundle,
    pub session_summary: &'a Option<ConversationSummary>,
    pub pending_system_messages: &'a mut Vec<String>,
    pub empty_response_retry_pending: bool,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
}

pub(super) struct MessageBuildData {
    pub messages: Vec<Value>,
}

const EMPTY_RETRY_MAX_PARENT_CHARS: usize = 800;

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

impl Agent {
    pub(super) async fn run_message_build_phase(
        &self,
        ctx: &mut MessageBuildCtx<'_>,
    ) -> anyhow::Result<MessageBuildData> {
        let session_id = ctx.session_id;
        let iteration = ctx.iteration;
        let user_text = ctx.user_text;
        let model = ctx.model;
        let system_prompt = ctx.system_prompt;
        let _consultant_pass_active = ctx.consultant_pass_active;
        let pinned_memories = ctx.pinned_memories;
        let tool_defs = ctx.tool_defs;
        let policy_bundle = ctx.policy_bundle;
        let session_summary = ctx.session_summary;
        let pending_system_messages = &mut *ctx.pending_system_messages;
        let empty_response_retry_pending = ctx.empty_response_retry_pending;
        let status_tx = ctx.status_tx;

        // Fetch recent history from canonical event stream.
        let recent_history = self.load_recent_history(session_id, 20).await?;

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
        let last_user_pos = deduped_msgs.iter().rposition(|m| m.role == "user");
        let pre_collapse_len = deduped_msgs.len();
        let deduped_msgs: Vec<&Message> = if let Some(boundary) = last_user_pos {
            deduped_msgs
                .into_iter()
                .enumerate()
                .filter(|(i, m)| {
                    if *i >= boundary {
                        true // current interaction: keep everything
                    } else {
                        // old interactions: drop tool results only; assistant messages
                        // survive (orphan stripping handles their tool_calls in JSON conversion)
                        m.role != "tool" || identity_preserve_indices.contains(i)
                    }
                })
                .map(|(_, m)| m)
                .collect()
        } else {
            deduped_msgs
        };
        let collapsed = pre_collapse_len.saturating_sub(deduped_msgs.len());
        if collapsed > 0 {
            info!(
                session_id,
                collapsed, "Collapsed tool results from previous interactions"
            );
        }

        // Identify old-interaction assistant messages for content truncation.
        // After collapse, recompute the last-user boundary and collect IDs of
        // assistant messages before it — their full text is stale context.
        // Exception: the assistant message immediately before the boundary is exempt
        // from truncation — it typically contains the budget/timeout response with
        // handoff context (activity summary, files read, commands run) that the next
        // interaction needs to avoid re-exploring from scratch.
        let collapse_boundary = deduped_msgs.iter().rposition(|m| m.role == "user");
        let old_interaction_assistant_ids: std::collections::HashSet<&str> =
            if let Some(boundary) = collapse_boundary {
                // Find the immediately-prior assistant message (right before boundary).
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
                let content = if is_old_assistant {
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

                // Prevent "empty response" fallbacks from accumulating as prompt context.
                // These messages are user-visible (stored in history) but not useful for
                // subsequent turns and can contribute to degraded model behavior.
                if m.role == "assistant"
                    && m.tool_calls_json.is_none()
                    && content.as_deref().is_some_and(|c| {
                        c.trim_start()
                            .starts_with("I wasn't able to process that request.")
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
                        } else if m.content.is_none() {
                            // Assistant message had tool_calls but all were orphaned,
                            // and no text content — drop it entirely
                            return None;
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
        // consultant-pass `continue` (no messages are stored between iterations,
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

        // System nudges (budget warnings, loop-stop reminders, etc.): inject for a single
        // LLM call so they influence the model without polluting stored history.
        for content in pending_system_messages.drain(..) {
            messages.push(json!({
                "role": "system",
                "content": content,
            }));
        }

        // Empty-response recovery: if the prior iteration produced no text and no tool calls,
        // inject a system nudge for the next LLM call. (Tool-role nudges are dropped by
        // message-order fixups because they don't correspond to an assistant tool_call_id.)
        if empty_response_retry_pending && !is_trigger_session(session_id) {
            messages.push(json!({
                "role": "system",
                "content": "[SYSTEM] Your previous reply was empty (no text and no tool calls). This retry is running with reduced conversation history to recover. You MUST either (1) call the required tools, or (2) reply with a concrete blocker and the missing info. Do NOT return an empty response."
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
}
