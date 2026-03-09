use super::recall_guardrails::filter_tool_defs_for_personal_memory;
use super::response_phase::ResponsePhaseOutcome;
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::llm_markers::INTENT_GATE_MARKER;
use crate::traits::ProviderResponse;

fn build_tool_output_completion_reply(tool_output: &str) -> Option<String> {
    let trimmed = tool_output.trim();
    // Don't use trivially uninformative tool outputs as completion replies.
    // These produce confusing messages like "Here is the latest tool output: (no output)".
    if is_trivial_tool_output(trimmed) {
        return None;
    }
    Some(format!("Here is the latest tool output:\n\n{}", trimmed))
}

fn is_trivial_tool_output(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.is_empty()
        || lower == "(no output)"
        || lower == "no output"
        || lower == "ok"
        || lower == "done"
        || lower == "success"
        || lower.starts_with("exit code:")
        || lower.starts_with("[exit code:")
        || lower.starts_with("blocked:") // terminal safety rejection, not a user-facing answer
        || lower.starts_with("error:")
        || lower.starts_with("duplicate send_file suppressed:")
        || (lower.starts_with("file written") && lower.len() < 100)
        || (lower.starts_with("wrote ") && lower.len() < 100)
        || looks_like_directory_listing(&lower)
}

/// Detect `ls -la` style output: starts with "total N" and contains
/// permission-style lines (e.g. "drwxr-xr-x", "-rw-r--r--").
fn looks_like_directory_listing(lower: &str) -> bool {
    if !lower.starts_with("total ") {
        return false;
    }
    let mut perm_lines = 0;
    for line in lower.lines().skip(1) {
        let trimmed = line.trim();
        if trimmed.starts_with("drwx") || trimmed.starts_with("-rw") || trimmed.starts_with("lrwx")
        {
            perm_lines += 1;
        }
    }
    perm_lines >= 2
}

fn build_activity_summary_reply(tool_calls: &[&str]) -> String {
    let calls: Vec<String> = tool_calls.iter().map(|call| (*call).to_string()).collect();
    let summary = post_task::categorize_tool_calls(&calls);
    if !summary.trim().is_empty() {
        return summary.trim().to_string();
    }

    let external_only = tool_calls
        .iter()
        .any(|call| call.starts_with("http_request(") || call.starts_with("web_fetch("));
    if external_only {
        "I checked the requested external sources, but I still need a final confirmation before I can claim success."
            .to_string()
    } else {
        format!(
            "I completed {} action{}.",
            tool_calls.len(),
            if tool_calls.len() == 1 { "" } else { "s" }
        )
    }
}

fn build_verification_pending_reply(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
) -> String {
    let target = turn_context
        .completion_contract
        .primary_target_hint()
        .map(|value| format!(" against {}", value))
        .unwrap_or_default();
    let mut reply = format!(
        "I completed part of the request, but I haven't verified the final outcome{} yet.",
        target
    );
    if !learning_ctx.tool_calls.is_empty() {
        let actions: Vec<&str> = learning_ctx
            .tool_calls
            .iter()
            .map(|call| call.as_str())
            .collect();
        reply.push_str("\n\n");
        reply.push_str(&build_activity_summary_reply(&actions));
        reply.push_str("\n\nI need a final read-only check before I can claim success.");
    }
    reply
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

fn completion_verification_still_required(
    turn_context: &TurnContext,
    completion_progress: &CompletionProgress,
) -> bool {
    let contract = &turn_context.completion_contract;
    let has_concrete_verification_reason = contract.explicit_verification_requested
        || !contract.verification_targets.is_empty()
        || matches!(
            contract.task_kind,
            CompletionTaskKind::Diagnose | CompletionTaskKind::Monitor
        );

    contract.requires_observation
        && completion_progress.verification_pending
        && has_concrete_verification_reason
}

pub(super) struct CompletionCtx<'a> {
    pub resp: &'a mut ProviderResponse,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub learning_ctx: &'a mut LearningContext,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
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
    pub pending_external_action_ack: &'a mut Option<String>,
    pub require_file_recheck_before_answer: &'a mut bool,
    pub completion_progress: &'a mut CompletionProgress,
    pub turn_context: &'a TurnContext,
    pub needs_tools_for_turn: bool,
    pub force_text_response: bool,
}

impl Agent {
    pub(super) async fn run_completion_phase(
        &self,
        ctx: &mut CompletionCtx<'_>,
    ) -> anyhow::Result<Option<ResponsePhaseOutcome>> {
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
        let mut pending_external_action_ack = std::mem::take(ctx.pending_external_action_ack);
        let mut require_file_recheck_before_answer = *ctx.require_file_recheck_before_answer;
        let mut completion_progress = ctx.completion_progress.clone();
        let turn_context = ctx.turn_context;
        let needs_tools_for_turn = ctx.needs_tools_for_turn;
        let force_text_response = ctx.force_text_response;

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
                *ctx.pending_external_action_ack = pending_external_action_ack.clone();
                *ctx.require_file_recheck_before_answer = require_file_recheck_before_answer;
                *ctx.completion_progress = completion_progress.clone();
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

            if self.depth == 0
                && !completion_verification_still_required(turn_context, &completion_progress)
                && should_recover_completion_from_tool_output(
                    &reply,
                    self.depth,
                    total_successful_tool_calls,
                )
            {
                if let Some(external_action_ack) = pending_external_action_ack.take() {
                    info!(
                        session_id,
                        iteration, "Successful external-action acknowledgement enforced"
                    );
                    reply = external_action_ack;
                }
            }

            if self.depth == 0
                && force_text_response
                && learning_ctx
                    .tool_calls
                    .iter()
                    .any(|call| call.starts_with("send_file("))
                && (reply.trim().is_empty() || is_low_signal_task_lead_reply(&reply))
            {
                reply = Self::send_file_completion_reply().to_string();
                info!(
                    session_id,
                    iteration, "Force-text send_file completion upgraded to shared closeout"
                );
            }

            // Force-text fast-path: when the model can't use tools, all guards
            // that require tool execution (file-recheck, tool-required, deferred-
            // action) are pointless — they would block the reply and return
            // ContinueLoop, but the next iteration strips tools again, creating
            // a deadlock.  Skip directly to completion.  If the reply is empty or
            // low-signal, upgrade it to an activity summary.
            if force_text_response
                && self.depth == 0
                && total_successful_tool_calls >= 3
                && !completion_verification_still_required(turn_context, &completion_progress)
            {
                if reply.trim().is_empty()
                    || is_low_signal_task_lead_reply(&reply)
                    || looks_like_deferred_action_response(&reply)
                {
                    let actions: Vec<&str> =
                        learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                    if !actions.is_empty() {
                        reply = build_activity_summary_reply(&actions);
                    }
                }
                require_file_recheck_before_answer = false;
                info!(
                    session_id,
                    iteration,
                    total_successful_tool_calls,
                    reply_len = reply.len(),
                    "Force-text fast-path: bypassing all tool-requiring guards"
                );
                // Fall through to the normal completion path (sanitize + return)
            } else if should_enforce_no_tool_text_when_tools_required(
                &reply,
                needs_tools_for_turn,
                learning_ctx.tool_calls.len(),
                self.depth,
            ) {
                if tool_defs.is_empty() || force_text_response {
                    if !force_text_response {
                        // Only show the "no tools available" message when tools are genuinely
                        // absent. In force-text mode the model already has a reply — let it through.
                        reply = "I can't complete that request in this context because it requires running tools, but no tools are currently available. Please retry in a tool-enabled context."
                            .to_string();
                    }
                    warn!(
                        session_id,
                        iteration,
                        force_text_response,
                        "Tool-required response bypassed: tools unavailable or force-text active"
                    );
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
                        pending_system_messages.push(SystemDirective::RoutingContractEnforcement);
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
                        return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                    }
                }
            }

            if self.depth == 0
                && total_successful_tool_calls == 0
                && !used_identity_prefill
                && looks_like_deferred_action_response(&reply)
                && !is_substantive_text_response(&reply, 200)
            {
                if tool_defs.is_empty() {
                    warn!(
                        session_id,
                        iteration,
                        "Deferred-action reply with no available tools; returning explicit blocker"
                    );
                    reply = "I wasn't able to complete that request because no execution tools are available in this context. Please try again in a context with tool access."
                        .to_string();
                } else if deferred_no_tool_streak >= DEFERRED_NO_TOOL_ACCEPT_THRESHOLD
                    && is_substantive_text_response(&reply, 50)
                {
                    info!(
                        session_id,
                        iteration,
                        deferred_no_tool_streak,
                        reply_len = reply.len(),
                        "Accepting substantive text-only response after repeated deferred-no-tool retries"
                    );
                    deferred_no_tool_streak = 0;
                } else {
                    deferred_no_tool_streak = deferred_no_tool_streak.saturating_add(1);
                    consecutive_clean_iterations = 0;
                    pending_system_messages.push(SystemDirective::DeferredToolCallRequired);
                    warn!(
                        session_id,
                        iteration,
                        deferred_no_tool_streak,
                        "Deferred-action reply before first tool call; continuing loop"
                    );

                    if deferred_no_tool_streak >= DEFERRED_NO_TOOL_SWITCH_THRESHOLD
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
                        }
                    }

                    commit_state!();
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }
            }

            let has_tool_attempts = !learning_ctx.tool_calls.is_empty();
            let false_capability_denial =
                looks_like_false_capability_denial_after_tool_success(&reply);

            if false_capability_denial {
                if !force_text_response && !tool_defs.is_empty() && stall_count == 0 {
                    stall_count = stall_count.saturating_add(1);
                    consecutive_clean_iterations = 0;
                    pending_system_messages.push(SystemDirective::SuccessfulToolEvidenceMustBeUsed);
                    warn!(
                        session_id,
                        iteration,
                        reply_preview = %reply.chars().take(180).collect::<String>(),
                        "Rejected completion that denied live capabilities after successful tool use"
                    );
                    commit_state!();
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }

                let mut recovered = false;
                if let Some((tool_name, tool_output)) =
                    self.latest_non_system_tool_result(session_id, 2500).await
                {
                    if tool_name == "send_file" {
                        reply = Self::send_file_completion_reply().to_string();
                        recovered = true;
                    } else if let Some(tool_reply) =
                        build_tool_output_completion_reply(&tool_output)
                    {
                        reply = tool_reply;
                        recovered = true;
                    }
                }
                if !recovered && !learning_ctx.tool_calls.is_empty() {
                    let actions: Vec<&str> = learning_ctx
                        .tool_calls
                        .iter()
                        .map(|call| call.as_str())
                        .collect();
                    reply = build_activity_summary_reply(&actions);
                }
                info!(
                    session_id,
                    iteration,
                    recovered,
                    "Recovered false capability-denial completion after successful tools"
                );
            }

            let low_signal_completion = is_low_signal_task_lead_reply(&reply);
            let was_truly_empty = reply.trim().is_empty();
            if should_recover_completion_from_tool_output(
                &reply,
                self.depth,
                total_successful_tool_calls,
            ) {
                let mut recovered = false;
                if let Some((tool_name, tool_output)) =
                    self.latest_non_system_tool_result(session_id, 2500).await
                {
                    if tool_name == "send_file" {
                        reply = Self::send_file_completion_reply().to_string();
                        recovered = true;
                        info!(
                            session_id,
                            iteration,
                            "Recovered completion reply after send_file with shared closeout"
                        );
                    } else if tool_name == "read_file" && learning_ctx.tool_calls.len() > 1 {
                        // When the latest tool is read_file and there were multiple tool
                        // calls, the activity summary is more useful than a raw file dump.
                        // Skip tool-output recovery so the activity summary branch fires.
                        info!(
                            session_id,
                            iteration,
                            tool_call_count = learning_ctx.tool_calls.len(),
                            "Skipping read_file output recovery in favor of activity summary"
                        );
                    } else if let Some(tool_reply) =
                        build_tool_output_completion_reply(&tool_output)
                    {
                        reply = tool_reply;
                        recovered = true;
                        info!(
                            session_id,
                            iteration,
                            low_signal_completion,
                            "Recovered completion reply from latest tool output"
                        );
                    }
                }
                // If tool output was trivial/empty and the LLM returned a truly empty
                // response (not just low-signal), don't build an activity summary —
                // leave reply empty so the empty-response retry mechanism kicks in
                // and gives the model another chance to complete the task properly.
                if !recovered && !was_truly_empty && !learning_ctx.tool_calls.is_empty() {
                    let actions: Vec<&str> =
                        learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                    reply = build_activity_summary_reply(&actions);
                    info!(
                        session_id,
                        iteration,
                        tool_call_count = learning_ctx.tool_calls.len(),
                        "Built activity summary as completion (low-signal reply, trivial tool output)"
                    );
                } else if !recovered && was_truly_empty {
                    info!(
                        session_id,
                        iteration,
                        "Empty LLM response with no recoverable tool output — deferring to empty-response retry"
                    );
                }
            }

            if reply.is_empty()
                && self.depth == 0
                && force_text_response
                && learning_ctx
                    .tool_calls
                    .iter()
                    .any(|call| call.starts_with("send_file("))
            {
                reply = Self::send_file_completion_reply().to_string();
                info!(
                    session_id,
                    iteration,
                    "Recovered empty force-text completion with shared send_file closeout"
                );
            } else if reply.is_empty() && total_successful_tool_calls > 0 && self.depth == 0 {
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
                        let is_override = match tokio::time::timeout(
                            Duration::from_secs(2),
                            self.model_override.read(),
                        )
                        .await
                        {
                            Ok(guard) => *guard,
                            Err(_) => {
                                warn!(
                                        session_id,
                                        iteration,
                                        "Timed out acquiring model_override lock during empty-response recovery"
                                    );
                                false
                            }
                        };
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
                        return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
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
                        ..Message::runtime_defaults()
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
                    return Ok(Some(ResponsePhaseOutcome::Return(Ok(fallback))));
                }
                // First iteration or sub-agent — stay silent
                info!(session_id, iteration, "Agent completed with empty response");
                commit_state!();
                return Ok(Some(ResponsePhaseOutcome::Return(Ok(String::new()))));
            }

            if require_file_recheck_before_answer {
                if tool_defs.is_empty() || force_text_response {
                    warn!(
                        session_id,
                        iteration,
                        force_text_response,
                        "File re-check required but tools unavailable (empty or force-text); clearing guard"
                    );
                    // In force-text mode the model can't use tools, so blocking
                    // on file re-check is a deadlock. Clear the guard and let
                    // the response through.
                    require_file_recheck_before_answer = false;
                } else {
                    stall_count = stall_count.saturating_add(1);
                    consecutive_clean_iterations = 0;
                    pending_system_messages
                        .push(SystemDirective::ContradictoryFileEvidenceRecheckRequired);
                    warn!(
                        session_id,
                        iteration,
                        stall_count,
                        "Blocking completion until required file re-check is performed"
                    );
                    commit_state!();
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }
            }

            if completion_verification_still_required(turn_context, &completion_progress) {
                if tool_defs.is_empty() || force_text_response {
                    warn!(
                        session_id,
                        iteration,
                        force_text_response,
                        "Completion verification required but tools unavailable (empty or force-text); clearing guard"
                    );
                    reply = build_verification_pending_reply(turn_context, learning_ctx);
                    pending_external_action_ack = None;
                    // Avoid deadlocks when tools cannot run in this phase, but
                    // preserve the fact that verification did not happen in the reply itself.
                    completion_progress.verification_pending = false;
                } else {
                    stall_count = stall_count.saturating_add(1);
                    consecutive_clean_iterations = 0;
                    pending_system_messages.push(SystemDirective::CompletionVerificationRequired {
                        target_hint: turn_context.completion_contract.primary_target_hint(),
                    });
                    warn!(
                        session_id,
                        iteration,
                        stall_count,
                        "Blocking completion until request outcome verification is performed"
                    );
                    commit_state!();
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }
            }

            // Guardrail: don't accept "I'll do X" / workflow narration as
            // completion text. Either keep the loop alive (if tools exist)
            // or return an explicit blocker (if no tools are available).
            // When tools have already succeeded: allow ONE retry (the agent may
            // produce a better response), but if the guard fires a second time,
            // accept the reply to avoid "Stuck" loops (e.g., after remember_fact
            // the LLM says "I'll remember that" — a confirmation, not a real deferral).
            // Substantive-response fast path: if the model produced a long,
            // content-rich answer (≥200 chars after stripping deferred-action
            // lines) AND it doesn't contain leaked structural markers
            // ([tool_use:], [INTENT_GATE], etc.), accept it immediately even
            // if it opens with an action-promise phrase like "I'll recall…".
            // This prevents recall/informational queries from being rejected
            // and forced through unnecessary tool-call loops.
            let has_structural_markers = {
                let lower = reply.trim().to_ascii_lowercase();
                lower.contains("[consultation]")
                    || lower.contains(&INTENT_GATE_MARKER.to_ascii_lowercase())
                    || lower.contains("[tool_use:")
                    || lower.contains("[tool_call:")
            };
            let reply_is_substantive =
                !has_structural_markers && is_substantive_text_response(&reply, 200);
            let incomplete_live_work_summary = looks_like_incomplete_live_work_summary(&reply);
            if !used_identity_prefill
                && (looks_like_deferred_action_response(&reply) || incomplete_live_work_summary)
                && (!reply_is_substantive || incomplete_live_work_summary)
            {
                // Post-tool-success: if we've already caught one deferral after tools
                // succeeded, accept this reply instead of stalling further.
                // Exception: when force_text is active (tools stripped), a deferred
                // reply like "Let me examine..." is useless — the model can't act.
                // Replace it with an activity summary of what was actually done.
                if has_tool_attempts && stall_count >= 1 {
                    if force_text_response && !learning_ctx.tool_calls.is_empty() {
                        let actions: Vec<&str> =
                            learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                        reply = build_activity_summary_reply(&actions);
                        info!(
                            session_id,
                            iteration,
                            total_successful_tool_calls,
                            stall_count,
                            "Force-text active: replaced deferred reply with activity summary"
                        );
                    } else {
                        info!(
                            session_id,
                            iteration,
                            total_successful_tool_calls,
                            stall_count,
                            "Accepting deferred-looking reply as completion after tool progress"
                        );
                    }
                    // Fall through to the normal completion path below
                } else if tool_defs.is_empty() {
                    warn!(
                        session_id,
                        iteration,
                        "Deferred-action reply with no available tools; returning explicit blocker"
                    );
                    reply = "I wasn't able to complete that request because no execution tools are available in this context. Please try again in a context with tool access."
                    .to_string();
                } else if !has_tool_attempts
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
                    if !has_tool_attempts {
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
                        has_tool_attempts,
                        "Deferred-action reply without concrete results; continuing loop"
                    );

                    let deferred_nudge = if !has_tool_attempts {
                        SystemDirective::DeferredToolCallRequired
                    } else if incomplete_live_work_summary {
                        SystemDirective::LiveWorkPivotRequired
                    } else {
                        SystemDirective::DeferredProvideConcreteResults
                    };

                    pending_system_messages.push(deferred_nudge);

                    // Fallback expansion: widen tool set once after exactly two
                    // no-progress iterations, even in no-tool-call paths.
                    let fallback_trigger = if !has_tool_attempts {
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

                    if !has_tool_attempts
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
                            pending_system_messages.push(SystemDirective::RecoveryModeModelSwitch);
                        }
                    }

                    if !has_tool_attempts
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
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
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
                ..Message::runtime_defaults()
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

            // Sanitize user-facing output before any channel-specific redaction.
            let reply = crate::tools::sanitize::sanitize_user_facing_reply(&reply);
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
            return Ok(Some(ResponsePhaseOutcome::Return(Ok(reply))));
        }

        commit_state!();
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_activity_summary_reply, build_tool_output_completion_reply,
        build_verification_pending_reply, should_enforce_no_tool_text_when_tools_required,
        should_recover_completion_from_tool_output,
    };
    use crate::agent::post_task::LearningContext;
    use crate::agent::{
        history::CompletionTaskKind, CompletionContract, TurnContext, VerificationTarget,
        VerificationTargetKind,
    };
    use chrono::Utc;

    #[test]
    fn tool_output_reply_is_result_focused() {
        let reply =
            build_tool_output_completion_reply("cat: /nonexistent/file.txt: No such file").unwrap();
        assert!(reply.contains("latest tool output"));
        assert!(reply.contains("/nonexistent/file.txt"));
        assert!(!reply.starts_with("Done —"));
    }

    #[test]
    fn trivial_tool_output_returns_none() {
        assert!(build_tool_output_completion_reply("(no output)").is_none());
        assert!(build_tool_output_completion_reply("").is_none());
        assert!(build_tool_output_completion_reply("exit code: 0").is_none());
        assert!(build_tool_output_completion_reply(
            "Duplicate send_file suppressed: this exact file+caption was already sent in this task."
        )
        .is_none());
        assert!(
            build_tool_output_completion_reply("File written to /tmp/foo.py, 200 bytes").is_none()
        );
        // Directory listing is trivial
        assert!(build_tool_output_completion_reply(
            "total 24\ndrwxr-xr-x  3 user  wheel  96 Mar  4 21:08 __pycache__\n-rw-r--r--  1 user  wheel  1041 Mar  4 21:09 regex_engine.py\n-rw-r--r--  1 user  wheel  4972 Mar  4 21:03 test_regex.py"
        ).is_none());
        // Substantive output should still work
        assert!(
            build_tool_output_completion_reply("test_foo PASSED\ntest_bar PASSED\n2 passed")
                .is_some()
        );
    }

    #[test]
    fn activity_summary_lists_tool_calls() {
        let calls = vec!["terminal(mkdir -p /tmp/foo)", "write_file(/tmp/foo/bar.py)"];
        let reply = build_activity_summary_reply(&calls);
        assert!(reply.contains("Commands run:"));
        assert!(reply.contains("Files written:"));
        assert!(!reply.contains("terminal("));
        assert!(!reply.contains("write_file("));
    }

    #[test]
    fn verification_pending_reply_mentions_target_and_actions() {
        let turn_context = TurnContext {
            completion_contract: CompletionContract {
                task_kind: CompletionTaskKind::Diagnose,
                requires_observation: true,
                verification_targets: vec![VerificationTarget {
                    kind: VerificationTargetKind::Url,
                    value: "https://blog.aidaemon.ai".to_string(),
                }],
                ..CompletionContract::default()
            },
            ..TurnContext::default()
        };
        let learning_ctx = LearningContext {
            user_text: "I still don't see the posts.".to_string(),
            intent_domains: Vec::new(),
            tool_calls: vec!["terminal(vite build)".to_string()],
            errors: Vec::new(),
            first_error: None,
            recovery_actions: Vec::new(),
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
        };

        let reply = build_verification_pending_reply(&turn_context, &learning_ctx);
        assert!(reply.contains("haven't verified"));
        assert!(reply.contains("https://blog.aidaemon.ai"));
        assert!(reply.contains("Commands run:"));
        assert!(!reply.contains("terminal(vite build)"));
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
