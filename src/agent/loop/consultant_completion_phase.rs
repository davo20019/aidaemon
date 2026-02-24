use super::consultant_phase::ConsultantPhaseOutcome;
use super::recall_guardrails::filter_tool_defs_for_personal_memory;
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;

fn build_tool_output_completion_reply(tool_output: &str) -> String {
    format!("Here is the latest tool output:\n\n{}", tool_output.trim())
}

fn should_recover_completion_from_tool_output(
    reply: &str,
    depth: usize,
    total_successful_tool_calls: usize,
) -> bool {
    if depth != 0 || total_successful_tool_calls == 0 {
        return false;
    }
    reply.trim().is_empty() || is_low_signal_task_lead_reply(reply)
}

fn should_enforce_no_tool_text_when_tools_required(
    reply: &str,
    needs_tools_for_turn: bool,
    attempted_tool_calls: usize,
    depth: usize,
) -> bool {
    if depth != 0 || !needs_tools_for_turn || attempted_tool_calls > 0 {
        return false;
    }
    !reply.trim().is_empty()
}

pub(super) struct ConsultantCompletionCtx<'a> {
    pub resp: &'a mut ProviderResponse,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub learning_ctx: &'a mut LearningContext,
    pub pending_system_messages: &'a mut Vec<String>,
    pub tool_defs: &'a mut Vec<Value>,
    pub base_tool_defs: &'a mut Vec<Value>,
    pub available_capabilities: &'a mut HashMap<String, ToolCapabilities>,
    pub policy_bundle: &'a mut PolicyBundle,
    pub restrict_to_personal_memory_tools: bool,
    pub llm_provider: Arc<dyn ModelProvider>,
    pub llm_router: Option<Router>,
    pub model: &'a mut String,
    pub channel_ctx: ChannelContext,
    pub total_successful_tool_calls: usize,
    pub stall_count: &'a mut usize,
    pub consecutive_clean_iterations: &'a mut usize,
    pub deferred_no_tool_streak: &'a mut usize,
    pub deferred_no_tool_model_switches: &'a mut usize,
    pub fallback_expanded_once: &'a mut bool,
    pub empty_response_retry_used: &'a mut bool,
    pub empty_response_retry_pending: &'a mut bool,
    pub empty_response_retry_note: &'a mut Option<String>,
    pub identity_prefill_text: &'a mut Option<String>,
    pub pending_background_ack: &'a mut Option<String>,
    pub require_file_recheck_before_answer: &'a mut bool,
    pub needs_tools_for_turn: bool,
}

impl Agent {
    pub(super) async fn run_consultant_completion_phase(
        &self,
        ctx: &mut ConsultantCompletionCtx<'_>,
    ) -> anyhow::Result<Option<ConsultantPhaseOutcome>> {
        let resp = &mut *ctx.resp;
        let emitter = ctx.emitter;
        let task_id = ctx.task_id;
        let session_id = ctx.session_id;
        let user_text = ctx.user_text;
        let iteration = ctx.iteration;
        let task_start = ctx.task_start;
        let learning_ctx = &mut *ctx.learning_ctx;
        let pending_system_messages = &mut *ctx.pending_system_messages;
        let mut tool_defs = std::mem::take(ctx.tool_defs);
        let base_tool_defs = &*ctx.base_tool_defs;
        let available_capabilities = &*ctx.available_capabilities;
        let policy_bundle = &mut *ctx.policy_bundle;
        let restrict_to_personal_memory_tools = ctx.restrict_to_personal_memory_tools;
        let llm_provider = ctx.llm_provider.clone();
        let llm_router = ctx.llm_router.clone();
        let mut model = ctx.model.clone();
        let channel_ctx = ctx.channel_ctx.clone();
        let total_successful_tool_calls = ctx.total_successful_tool_calls;
        let mut stall_count = *ctx.stall_count;
        let mut consecutive_clean_iterations = *ctx.consecutive_clean_iterations;
        let mut deferred_no_tool_streak = *ctx.deferred_no_tool_streak;
        let mut deferred_no_tool_model_switches = *ctx.deferred_no_tool_model_switches;
        let mut fallback_expanded_once = *ctx.fallback_expanded_once;
        let mut empty_response_retry_used = *ctx.empty_response_retry_used;
        let mut empty_response_retry_pending = *ctx.empty_response_retry_pending;
        let mut empty_response_retry_note = ctx.empty_response_retry_note.clone();
        let mut identity_prefill_text = ctx.identity_prefill_text.clone();
        let mut pending_background_ack = std::mem::take(ctx.pending_background_ack);
        let mut require_file_recheck_before_answer = *ctx.require_file_recheck_before_answer;
        let needs_tools_for_turn = ctx.needs_tools_for_turn;

        macro_rules! commit_state {
            () => {
                *ctx.tool_defs = tool_defs;
                *ctx.model = model.clone();
                *ctx.stall_count = stall_count;
                *ctx.consecutive_clean_iterations = consecutive_clean_iterations;
                *ctx.deferred_no_tool_streak = deferred_no_tool_streak;
                *ctx.deferred_no_tool_model_switches = deferred_no_tool_model_switches;
                *ctx.fallback_expanded_once = fallback_expanded_once;
                *ctx.empty_response_retry_used = empty_response_retry_used;
                *ctx.empty_response_retry_pending = empty_response_retry_pending;
                *ctx.empty_response_retry_note = empty_response_retry_note.clone();
                *ctx.identity_prefill_text = identity_prefill_text.clone();
                *ctx.pending_background_ack = pending_background_ack.clone();
                *ctx.require_file_recheck_before_answer = require_file_recheck_before_answer;
            };
        }
        // === NATURAL COMPLETION: No tool calls ===
        if resp.tool_calls.is_empty() {
            let mut reply = resp
                .content
                .clone()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_default();

            // If we used an identity-attack prefill, prepend it so the user
            // sees the full decline (the API only returns continuation tokens).
            let used_identity_prefill = identity_prefill_text.is_some();
            if let Some(ref prefill) = identity_prefill_text {
                if reply.is_empty() {
                    reply = prefill.clone();
                } else {
                    reply = format!("{} {}", prefill, reply.trim_start());
                }
                identity_prefill_text = None;
            }

            // Deterministic cross-model behavior: once a long-running tool detaches
            // to background, do not rely on model compliance for the handoff text.
            if self.depth == 0 {
                if let Some(background_ack) = pending_background_ack.take() {
                    info!(
                        session_id,
                        iteration, "Background detach acknowledgement enforced"
                    );
                    reply = background_ack;
                }
            }

            if should_enforce_no_tool_text_when_tools_required(
                &reply,
                needs_tools_for_turn,
                learning_ctx.tool_calls.len(),
                self.depth,
            ) {
                if tool_defs.is_empty() {
                    warn!(
                        session_id,
                        iteration, "Tool-required response blocked, but no tools are available"
                    );
                    reply = "I can't complete that request in this context because it requires running tools, but no tools are currently available. Please retry in a tool-enabled context."
                        .to_string();
                } else {
                    deferred_no_tool_streak = deferred_no_tool_streak.saturating_add(1);
                    stall_count = 0;
                    consecutive_clean_iterations = 0;

                    // Early acceptance: after enough retries, if the model's text is
                    // substantive (not just "I'll do X"), accept it instead of looping
                    // forever.  This prevents stalls on queries the intent gate
                    // classified as needing tools but the model can answer directly
                    // (e.g., "Tell me a joke in Spanish", "List your capabilities").
                    if deferred_no_tool_streak >= DEFERRED_NO_TOOL_ACCEPT_THRESHOLD
                        && is_substantive_text_response(&reply, 15)
                    {
                        info!(
                            session_id,
                            iteration,
                            deferred_no_tool_streak,
                            reply_len = reply.len(),
                            "Accepting substantive text-only response after repeated tool-required retries"
                        );
                        deferred_no_tool_streak = 0;
                        // Fall through to normal completion path
                    } else {
                        pending_system_messages.push(
                            "[SYSTEM] ROUTING CONTRACT ENFORCEMENT: This turn requires tool execution. \
Ignore prior-turn outputs, run the required tool call(s) for the current user message, and then answer with concrete results."
                                .to_string(),
                        );
                        self.emit_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::IntentGate,
                            "Intent gate contract enforced: blocked text-only answer while tools required"
                                .to_string(),
                            json!({
                                "condition":"tools_required_no_tool_response",
                                "reply_len": reply.len(),
                                "deferred_no_tool_streak": deferred_no_tool_streak
                            }),
                        )
                        .await;
                        warn!(
                            session_id,
                            iteration,
                            deferred_no_tool_streak,
                            "Blocked no-tool completion because current turn requires tools"
                        );
                        commit_state!();
                        return Ok(Some(ConsultantPhaseOutcome::ContinueLoop));
                    }
                }
            }

            let low_signal_completion = is_low_signal_task_lead_reply(&reply);
            if should_recover_completion_from_tool_output(
                &reply,
                self.depth,
                total_successful_tool_calls,
            ) {
                if let Some(tool_output) = self
                    .latest_non_system_tool_output_excerpt(session_id, 2500)
                    .await
                {
                    reply = build_tool_output_completion_reply(&tool_output);
                    info!(
                        session_id,
                        iteration,
                        low_signal_completion,
                        "Recovered completion reply from latest tool output"
                    );
                }
            }

            if reply.is_empty() && total_successful_tool_calls > 0 && self.depth == 0 {
                reply = "I executed the requested tools, but I couldn't recover a usable output snapshot. Please ask me to rerun the command and I'll return the exact result.".to_string();
                info!(
                    session_id,
                    iteration, "Tool execution completed but no output snapshot was available"
                );
            }

            if reply.is_empty() {
                // User-facing empty response: never return silence.
                // Retry once; if the model remains empty, return an explicit fallback.
                if !is_trigger_session(session_id) {
                    if !empty_response_retry_used {
                        empty_response_retry_used = true;
                        empty_response_retry_pending = true;
                        empty_response_retry_note = resp
                            .response_note
                            .as_deref()
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string);

                        stall_count += 1;
                        consecutive_clean_iterations = 0;

                        // Retry once with a stronger model profile to avoid repeated empties,
                        // unless the user explicitly pinned a model override.
                        let is_override = *self.model_override.read().await;
                        if !is_override {
                            let reason =
                                format!("empty_response(iter={},model={})", iteration, model);
                            if policy_bundle.policy.escalate(reason.clone()) {
                                POLICY_METRICS
                                    .escalation_total
                                    .fetch_add(1, Ordering::Relaxed);
                                if let Some(ref router) = llm_router {
                                    let next_model = router
                                        .select_for_profile(policy_bundle.policy.model_profile)
                                        .to_string();
                                    if next_model != model {
                                        info!(
                                            session_id,
                                            iteration,
                                            reason = %reason,
                                            from_model = %model,
                                            to_model = %next_model,
                                            "Empty-response recovery: escalated model for retry"
                                        );
                                        model = next_model;
                                    }
                                }
                            }
                        }

                        info!(
                            session_id,
                            iteration,
                            response_note = ?resp.response_note,
                            "Empty-response recovery: issuing one retry before fallback"
                        );

                        commit_state!();
                        return Ok(Some(ConsultantPhaseOutcome::ContinueLoop));
                    }

                    let response_note = if empty_response_retry_pending {
                        resp.response_note
                            .as_deref()
                            .or(empty_response_retry_note.as_deref())
                    } else {
                        resp.response_note.as_deref()
                    };
                    let fallback = build_empty_response_fallback(response_note);
                    info!(
                        session_id,
                        iteration,
                        response_note = ?resp.response_note,
                        retry_response_note = ?empty_response_retry_note,
                        "Agent completed with no work done — LLM returned empty with tools available"
                    );
                    let assistant_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "assistant".to_string(),
                        content: Some(fallback.clone()),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.5,
                        embedding: None,
                    };
                    self.append_assistant_message_with_event(
                        emitter,
                        &assistant_msg,
                        &model,
                        resp.usage.as_ref().map(|u| u.input_tokens),
                        resp.usage.as_ref().map(|u| u.output_tokens),
                    )
                    .await?;

                    self.emit_task_end(
                        emitter,
                        task_id,
                        TaskStatus::Completed,
                        task_start,
                        iteration,
                        learning_ctx.tool_calls.len(),
                        None,
                        Some(fallback.chars().take(200).collect()),
                    )
                    .await;

                    commit_state!();
                    return Ok(Some(ConsultantPhaseOutcome::Return(Ok(fallback))));
                }
                // First iteration or sub-agent — stay silent
                info!(session_id, iteration, "Agent completed with empty response");
                commit_state!();
                return Ok(Some(ConsultantPhaseOutcome::Return(Ok(String::new()))));
            }

            if require_file_recheck_before_answer {
                if tool_defs.is_empty() {
                    warn!(
                        session_id,
                        iteration,
                        "File re-check required but no tools available; returning explicit blocker"
                    );
                    reply = "I found conflicting file evidence from prior tool results, and I can't re-check now because no file tools are available in this context. Please retry in a tool-enabled context."
                        .to_string();
                    require_file_recheck_before_answer = false;
                } else {
                    stall_count = stall_count.saturating_add(1);
                    consecutive_clean_iterations = 0;
                    pending_system_messages.push(
                        "[SYSTEM] Contradictory file evidence was detected (one tool found files while another reported no matches). \
                         Before answering, you MUST run at least one file re-check tool with an explicit path (e.g. search_files or project_inspect with path)."
                            .to_string(),
                    );
                    warn!(
                        session_id,
                        iteration,
                        stall_count,
                        "Blocking completion until required file re-check is performed"
                    );
                    commit_state!();
                    return Ok(Some(ConsultantPhaseOutcome::ContinueLoop));
                }
            }

            // Guardrail: don't accept "I'll do X" / workflow narration as
            // completion text. Either keep the loop alive (if tools exist)
            // or return an explicit blocker (if no tools are available).
            // When tools have already succeeded: allow ONE retry (the agent may
            // produce a better response), but if the guard fires a second time,
            // accept the reply to avoid "Stuck" loops (e.g., after remember_fact
            // the LLM says "I'll remember that" — a confirmation, not a real deferral).
            if self.depth == 0
                && !used_identity_prefill
                && looks_like_deferred_action_response(&reply)
            {
                // Post-tool-success: if we've already caught one deferral after tools
                // succeeded, accept this reply instead of stalling further.
                if total_successful_tool_calls > 0 && stall_count >= 1 {
                    info!(
                        session_id,
                        iteration,
                        total_successful_tool_calls,
                        stall_count,
                        "Accepting deferred-looking reply as completion after tool progress"
                    );
                    // Fall through to the normal completion path below
                } else if tool_defs.is_empty() {
                    warn!(
                        session_id,
                        iteration,
                        "Deferred-action reply with no available tools; returning explicit blocker"
                    );
                    reply = "I wasn't able to complete that request because no execution tools are available in this context. Please try again in a context with tool access."
                    .to_string();
                } else if total_successful_tool_calls == 0
                    && deferred_no_tool_streak >= DEFERRED_NO_TOOL_ACCEPT_THRESHOLD
                    && is_substantive_text_response(&reply, 50)
                {
                    // Early acceptance: the model keeps producing deferred-action text
                    // but the underlying content is substantive (e.g., a greeting,
                    // explanation, joke, or capability listing).  Queries that genuinely
                    // don't need tools should not stall for 6 retries.
                    info!(
                        session_id,
                        iteration,
                        deferred_no_tool_streak,
                        reply_len = reply.len(),
                        "Accepting substantive text-only response after repeated deferred-no-tool retries"
                    );
                    deferred_no_tool_streak = 0;
                    // Fall through to the normal completion path below
                } else {
                    // Pre-execution deferrals ("I'll do X") should not consume the
                    // main stall budget. Reserve stall_count for post-tool loops so
                    // we don't fail as "stuck" before any tool ever executes.
                    if total_successful_tool_calls == 0 {
                        deferred_no_tool_streak = deferred_no_tool_streak.saturating_add(1);
                        POLICY_METRICS
                            .deferred_no_tool_deferral_detected_total
                            .fetch_add(1, Ordering::Relaxed);
                    } else {
                        stall_count = stall_count.saturating_add(1);
                        deferred_no_tool_streak = 0;
                    }
                    consecutive_clean_iterations = 0;
                    warn!(
                        session_id,
                        iteration,
                        stall_count,
                        deferred_no_tool_streak,
                        total_successful_tool_calls,
                        "Deferred-action reply without concrete results; continuing loop"
                    );

                    let deferred_nudge = if total_successful_tool_calls == 0 {
                        "[SYSTEM] HARD REQUIREMENT: your next reply MUST include at least one tool call. \
                     Do NOT return planning text like \"I'll do X\". Text-only replies are invalid for this request."
                            .to_string()
                    } else {
                        "[SYSTEM] You narrated future work instead of providing results. \
                     Execute any remaining required tools, or return concrete outcomes and blockers now."
                        .to_string()
                    };

                    pending_system_messages.push(deferred_nudge);

                    // Fallback expansion: widen tool set once after exactly two
                    // no-progress iterations, even in no-tool-call paths.
                    let fallback_trigger = if total_successful_tool_calls == 0 {
                        deferred_no_tool_streak == 2
                    } else {
                        stall_count == 2
                    };
                    if fallback_trigger && !fallback_expanded_once {
                        fallback_expanded_once = true;
                        let previous_count = tool_defs.len();
                        let widened = self.filter_tool_definitions_for_policy(
                            base_tool_defs,
                            available_capabilities,
                            &policy_bundle.policy,
                            policy_bundle.risk_score,
                            true,
                        );
                        let widened = if restrict_to_personal_memory_tools {
                            filter_tool_defs_for_personal_memory(&widened)
                        } else {
                            widened
                        };
                        if !widened.is_empty() {
                            POLICY_METRICS
                                .fallback_expansion_total
                                .fetch_add(1, Ordering::Relaxed);
                            tool_defs = widened;
                            info!(
                                session_id,
                                iteration,
                                previous_count,
                                widened_count = tool_defs.len(),
                                "No-progress fallback expansion applied (deferred-action path)"
                            );
                        }
                    }

                    if total_successful_tool_calls == 0
                        && deferred_no_tool_streak >= DEFERRED_NO_TOOL_SWITCH_THRESHOLD
                        && deferred_no_tool_model_switches < MAX_DEFERRED_NO_TOOL_MODEL_SWITCHES
                    {
                        if let Some(next_model) = self
                            .pick_fallback_excluding(&model, &[], llm_router.as_ref())
                            .await
                        {
                            info!(
                                session_id,
                                iteration,
                                from_model = %model,
                                to_model = %next_model,
                                "Deferred/no-tool recovery: switching model for one retry window"
                            );
                            model = next_model;
                            deferred_no_tool_model_switches += 1;
                            POLICY_METRICS
                                .deferred_no_tool_model_switch_total
                                .fetch_add(1, Ordering::Relaxed);
                            // Strategy changed, give the new model a fresh stall budget.
                            stall_count = 0;
                            pending_system_messages.push(
                            "[SYSTEM] Recovery mode: a model switch was applied because prior replies kept promising actions without tool calls. Call the required tools now and return concrete results."
                                .to_string(),
                        );
                        }
                    }

                    if total_successful_tool_calls == 0
                        && deferred_no_tool_streak >= MAX_STALL_ITERATIONS
                        && !learning_ctx
                            .errors
                            .iter()
                            .any(|(e, _)| e == DEFERRED_NO_TOOL_ERROR_MARKER)
                    {
                        learning_ctx
                            .errors
                            .push((DEFERRED_NO_TOOL_ERROR_MARKER.to_string(), false));
                        POLICY_METRICS
                            .deferred_no_tool_error_marker_total
                            .fetch_add(1, Ordering::Relaxed);
                        warn!(
                            session_id,
                            iteration,
                            deferred_no_tool_streak,
                            "Deferred/no-tool recovery exhausted: recording terminal marker"
                        );
                    }

                    commit_state!();
                    return Ok(Some(ConsultantPhaseOutcome::ContinueLoop));
                }
            }

            let assistant_msg = Message {
                id: Uuid::new_v4().to_string(),
                session_id: session_id.to_string(),
                role: "assistant".to_string(),
                content: Some(reply.clone()),
                tool_call_id: None,
                tool_name: None,
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                embedding: None,
            };
            self.append_assistant_message_with_event(
                emitter,
                &assistant_msg,
                &model,
                resp.usage.as_ref().map(|u| u.input_tokens),
                resp.usage.as_ref().map(|u| u.output_tokens),
            )
            .await?;

            // Emit TaskEnd event
            self.emit_task_end(
                emitter,
                task_id,
                TaskStatus::Completed,
                task_start,
                iteration,
                learning_ctx.tool_calls.len(),
                None,
                Some(reply.chars().take(200).collect()),
            )
            .await;

            // Process learning in background
            learning_ctx.completed_naturally = true;
            let learning_ctx_for_task = learning_ctx.clone();
            let state = self.state.clone();
            tokio::spawn(async move {
                if let Err(e) = post_task::process_learning(&state, learning_ctx_for_task).await {
                    warn!("Learning failed: {}", e);
                }
            });

            // Progressive fact extraction: extract durable facts immediately
            if self.context_window_config.progressive_facts
                && crate::memory::context_window::should_extract_facts(user_text)
            {
                let fast_model = llm_router
                    .as_ref()
                    .map(|r| r.select(crate::router::Tier::Fast).to_string())
                    .unwrap_or_else(|| model.clone());
                crate::memory::context_window::spawn_progressive_extraction(
                    llm_provider.clone(),
                    fast_model.clone(),
                    self.state.clone(),
                    user_text.to_string(),
                    reply.clone(),
                    channel_ctx.channel_id.clone(),
                    channel_ctx.visibility,
                );

                // Incremental summarization: update summary if threshold reached
                if self.context_window_config.enabled {
                    crate::memory::context_window::spawn_incremental_summarization(
                        llm_provider.clone(),
                        fast_model,
                        self.state.clone(),
                        session_id.to_string(),
                        self.context_window_config.summarize_threshold,
                        self.context_window_config.summary_window,
                    );
                }
            }

            // Sanitize output for public channels
            let reply = match channel_ctx.visibility {
                ChannelVisibility::Public | ChannelVisibility::PublicExternal => {
                    let (sanitized, had_redactions) =
                        crate::tools::sanitize::sanitize_output(&reply);
                    if had_redactions && channel_ctx.visibility == ChannelVisibility::PublicExternal
                    {
                        format!("{}\n\n(Some content was filtered for security)", sanitized)
                    } else {
                        sanitized
                    }
                }
                _ => reply,
            };

            // Diagnostic: warn when completing with zero tool calls and deferred-action
            // text. This catches cases where the agent promises future work ("I'll search
            // for TODOs...") but never actually executes any tools (G2 stall pattern).
            if total_successful_tool_calls == 0
                && !reply.trim().is_empty()
                && looks_like_deferred_action_response(&reply)
            {
                warn!(
                    session_id,
                    iteration,
                    reply_preview = &reply.chars().take(200).collect::<String>() as &str,
                    "Zero-tool completion with deferred-action text detected — possible stall pattern"
                );
            }

            info!(
                session_id,
                iteration,
                reply_len = reply.len(),
                reply_empty = reply.trim().is_empty(),
                reply_preview = &reply.chars().take(120).collect::<String>() as &str,
                "Agent completed naturally"
            );
            commit_state!();
            return Ok(Some(ConsultantPhaseOutcome::Return(Ok(reply))));
        }

        commit_state!();
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_tool_output_completion_reply, should_enforce_no_tool_text_when_tools_required,
        should_recover_completion_from_tool_output,
    };

    #[test]
    fn tool_output_reply_is_result_focused() {
        let reply = build_tool_output_completion_reply("cat: /nonexistent/file.txt: No such file");
        assert!(reply.contains("latest tool output"));
        assert!(reply.contains("/nonexistent/file.txt"));
        assert!(!reply.starts_with("Done —"));
    }

    #[test]
    fn recover_completion_when_reply_is_empty_after_tools() {
        assert!(should_recover_completion_from_tool_output("", 0, 1));
    }

    #[test]
    fn recover_completion_when_reply_is_low_signal_after_tools() {
        assert!(should_recover_completion_from_tool_output(
            "Done — Run the command \"cat /nonexistent/file.txt\" and tell me what happens",
            0,
            2
        ));
    }

    #[test]
    fn do_not_recover_completion_for_substantive_reply() {
        assert!(!should_recover_completion_from_tool_output(
            "The command returned: file not found.",
            0,
            1
        ));
    }

    #[test]
    fn do_not_recover_completion_without_tool_progress() {
        assert!(!should_recover_completion_from_tool_output("", 0, 0));
    }

    #[test]
    fn do_not_recover_completion_for_sub_agent_depth() {
        assert!(!should_recover_completion_from_tool_output("Done.", 1, 1));
    }

    #[test]
    fn enforce_tools_contract_for_text_reply_without_any_tool_attempt() {
        assert!(should_enforce_no_tool_text_when_tools_required(
            "The file was not found.",
            true,
            0,
            0
        ));
    }

    #[test]
    fn do_not_enforce_tools_contract_after_tool_attempts_exist() {
        assert!(!should_enforce_no_tool_text_when_tools_required(
            "The command failed.",
            true,
            1,
            0
        ));
    }

    #[test]
    fn do_not_enforce_tools_contract_when_turn_does_not_require_tools() {
        assert!(!should_enforce_no_tool_text_when_tools_required(
            "Paris.", false, 0, 0
        ));
    }

    #[test]
    fn do_not_enforce_tools_contract_for_empty_reply() {
        assert!(!should_enforce_no_tool_text_when_tools_required(
            "", true, 0, 0
        ));
    }

    #[test]
    fn do_not_enforce_tools_contract_for_sub_agent_depth() {
        assert!(!should_enforce_no_tool_text_when_tools_required(
            "Need to run tools.",
            true,
            0,
            1
        ));
    }
}
