use super::recall_guardrails::filter_tool_defs_for_personal_memory;
use super::response_phase::ResponsePhaseOutcome;
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::llm_markers::INTENT_GATE_MARKER;
use crate::traits::ProviderResponse;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompletionRecoveryCandidate {
    tool_name: String,
    tool_output: String,
    artifact_delivered: bool,
}

fn build_tool_output_completion_reply(
    tool_name: &str,
    tool_output: &str,
    artifact_delivered: bool,
) -> Option<String> {
    let trimmed = tool_output.trim();
    // Don't use trivially uninformative tool outputs as completion replies.
    // These produce confusing messages like "Here is the latest tool output: (no output)".
    if is_trivial_tool_output(trimmed) || tool_output_requires_final_synthesis(tool_name, trimmed) {
        return None;
    }
    if artifact_delivered {
        Some(format!(
            "I sent the requested file. Here is the latest result snapshot:\n\n{}",
            trimmed
        ))
    } else {
        Some(format!("Here is the latest tool output:\n\n{}", trimmed))
    }
}

fn build_force_text_deferred_completion_reply(
    candidate: &CompletionRecoveryCandidate,
    tool_call_count: usize,
) -> Option<String> {
    if candidate.tool_name == "send_file" {
        return Some(Agent::send_file_completion_reply().to_string());
    }

    if candidate.tool_name == "read_file" && tool_call_count > 1 && !candidate.artifact_delivered {
        return None;
    }

    build_tool_output_completion_reply(
        &candidate.tool_name,
        &candidate.tool_output,
        candidate.artifact_delivered,
    )
}

fn is_low_signal_http_metadata_line_for_completion(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.starts_with("content-type:")
        || lower.starts_with("content-length:")
        || lower.starts_with("server:")
        || lower.starts_with("date:")
        || lower.starts_with("cache-control:")
        || lower.starts_with("etag:")
        || lower.starts_with("last-modified:")
        || lower.starts_with("strict-transport-security:")
        || lower.starts_with("x-")
}

fn extract_structured_tool_output_excerpt(tool_output: &str, max_chars: usize) -> Option<String> {
    let trimmed = tool_output.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut lines = trimmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let status_line = lines
        .next()
        .filter(|line| line.to_ascii_lowercase().starts_with("http "))
        .map(str::to_string);

    let body = trimmed
        .split_once("\n\n")
        .map(|(_, rest)| rest.trim())
        .filter(|rest| !rest.is_empty())
        .unwrap_or(trimmed);

    let sanitized = crate::tools::sanitize::sanitize_external_content(body);
    let sanitized = sanitized.trim();
    if sanitized.is_empty() {
        return status_line.map(|status| crate::utils::truncate_with_note(&status, max_chars));
    }

    let compact = if sanitized.starts_with('{') || sanitized.starts_with('[') {
        match serde_json::from_str::<serde_json::Value>(sanitized) {
            Ok(value) => value.to_string(),
            Err(_) => sanitized.split_whitespace().collect::<Vec<_>>().join(" "),
        }
    } else {
        let lines: Vec<&str> = sanitized
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .filter(|line| !is_low_signal_http_metadata_line_for_completion(line))
            .take(8)
            .collect();
        if lines.is_empty() {
            sanitized.to_string()
        } else {
            lines.join("\n")
        }
    };

    let mut excerpt = crate::utils::truncate_with_note(compact.trim(), max_chars);
    if excerpt.is_empty() {
        return status_line.map(|status| crate::utils::truncate_with_note(&status, max_chars));
    }

    if let Some(status) = status_line {
        if !excerpt.eq_ignore_ascii_case(&status)
            && !excerpt.to_ascii_lowercase().starts_with("http ")
        {
            excerpt = crate::utils::truncate_with_note(&format!("{status}\n{excerpt}"), max_chars);
        }
    }

    if is_trivial_tool_output(&excerpt) {
        None
    } else {
        Some(excerpt)
    }
}

fn build_structured_tool_output_completion_reply(
    tool_name: &str,
    tool_output: &str,
    artifact_delivered: bool,
) -> Option<String> {
    if !tool_output_requires_final_synthesis(tool_name, tool_output) {
        return None;
    }

    let excerpt = extract_structured_tool_output_excerpt(tool_output, 1600)?;
    if artifact_delivered {
        Some(format!(
            "I sent the requested file. Here is the latest result excerpt:\n\n{}",
            excerpt
        ))
    } else {
        Some(format!("Here is the latest result excerpt:\n\n{}", excerpt))
    }
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

fn tool_output_requires_final_synthesis(tool_name: &str, tool_output: &str) -> bool {
    if tool_output.trim().is_empty() {
        return false;
    }

    if matches!(tool_name, "http_request" | "web_fetch") {
        return true;
    }

    let trimmed = tool_output.trim_start();
    trimmed.starts_with('{')
        || trimmed.starts_with('[')
        || trimmed
            .to_ascii_lowercase()
            .starts_with("http 200 ok\ncontent-type: application/json")
}

fn structured_result_synthesis_directive(
    candidate: &CompletionRecoveryCandidate,
) -> SystemDirective {
    SystemDirective::StructuredToolResultSynthesis {
        tool_name: candidate.tool_name.clone(),
        excerpt: crate::utils::truncate_with_note(&candidate.tool_output, 1200),
    }
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

fn candidate_allowed_for_completion_fallback(
    candidate: Option<&CompletionRecoveryCandidate>,
    tool_call_count: usize,
) -> Option<&CompletionRecoveryCandidate> {
    match candidate {
        Some(candidate)
            if candidate.tool_name == "read_file"
                && tool_call_count > 1
                && !candidate.artifact_delivered =>
        {
            None
        }
        other => other,
    }
}

fn build_completion_fallback_reply(
    candidate: Option<&CompletionRecoveryCandidate>,
    tool_calls: &[&str],
    tool_call_count: usize,
) -> String {
    if let Some(candidate) = candidate_allowed_for_completion_fallback(candidate, tool_call_count) {
        if candidate.tool_name == "send_file" {
            return Agent::send_file_completion_reply().to_string();
        }
        if let Some(reply) = build_tool_output_completion_reply(
            &candidate.tool_name,
            &candidate.tool_output,
            candidate.artifact_delivered,
        ) {
            return reply;
        }
        if let Some(reply) = build_structured_tool_output_completion_reply(
            &candidate.tool_name,
            &candidate.tool_output,
            candidate.artifact_delivered,
        ) {
            return reply;
        }
    }

    build_activity_summary_reply(tool_calls)
}

fn is_low_info_completion_tool(tool_name: &str) -> bool {
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

fn is_delivery_completion_tool(tool_name: &str) -> bool {
    matches!(tool_name, "send_file" | "send_media")
}

fn choose_completion_recovery_candidate(
    candidates: &[(String, String)],
    max_chars: usize,
) -> Option<CompletionRecoveryCandidate> {
    let mut latest_delivery: Option<(String, String)> = None;
    let mut latest_observational: Option<(String, String)> = None;

    for (tool_name, detail) in candidates {
        let tool_name = tool_name.trim();
        let detail = detail.trim();
        if tool_name.is_empty() || detail.is_empty() || is_low_info_completion_tool(tool_name) {
            continue;
        }

        if is_delivery_completion_tool(tool_name) {
            if latest_delivery.is_none() {
                latest_delivery = Some((
                    tool_name.to_string(),
                    crate::utils::truncate_with_note(detail, max_chars),
                ));
            }
            continue;
        }

        if is_trivial_tool_output(detail) {
            continue;
        }

        if latest_observational.is_none() {
            latest_observational = Some((
                tool_name.to_string(),
                crate::utils::truncate_with_note(detail, max_chars),
            ));
        }
    }

    if let Some((tool_name, tool_output)) = latest_observational {
        return Some(CompletionRecoveryCandidate {
            tool_name,
            tool_output,
            artifact_delivered: latest_delivery.is_some(),
        });
    }

    latest_delivery.map(|(tool_name, tool_output)| CompletionRecoveryCandidate {
        tool_name,
        tool_output,
        artifact_delivered: false,
    })
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

fn looks_like_idle_reengagement_reply(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let generic_help_prompt = lower.contains("what would you like me to help you with")
        || lower.contains("what can i help you with")
        || lower.contains("how can i help")
        || lower.contains("what would you like to continue with")
        || lower.contains("what would you like to do next");

    let reset_intro = lower.starts_with("i'm here")
        || lower.starts_with("im here")
        || lower.starts_with("i am here")
        || lower.starts_with("ready when you are")
        || lower.starts_with("ready to help");

    generic_help_prompt || (reset_intro && lower.len() <= 180)
}

async fn latest_task_tool_result_for_completion(
    agent: &Agent,
    session_id: &str,
    task_id: &str,
    max_chars: usize,
) -> Option<CompletionRecoveryCandidate> {
    let mut task_results: Vec<(String, String)> = Vec::new();

    let events = match tokio::time::timeout(
        Duration::from_secs(5),
        agent
            .event_store
            .query_task_events_for_session(session_id, task_id),
    )
    .await
    {
        Ok(Ok(events)) => events,
        Ok(Err(_)) | Err(_) => Vec::new(),
    };

    for event in events.iter().rev() {
        if event.event_type != EventType::ToolResult {
            continue;
        }
        let Ok(data) = event.parse_data::<ToolResultData>() else {
            continue;
        };
        let tool_name = data.name.trim();
        if tool_name.is_empty() {
            continue;
        }
        let detail = if data.success {
            data.result.trim()
        } else {
            data.error.as_deref().unwrap_or(&data.result).trim()
        };
        if detail.is_empty() {
            continue;
        }
        task_results.push((tool_name.to_string(), detail.to_string()));
    }

    if let Some(candidate) = choose_completion_recovery_candidate(&task_results, max_chars) {
        return Some(candidate);
    }

    let history = match tokio::time::timeout(
        Duration::from_secs(5),
        agent.state.get_history(session_id, 80),
    )
    .await
    {
        Ok(Ok(history)) => history,
        Ok(Err(_)) | Err(_) => return None,
    };

    let mut interaction_results: Vec<(String, String)> = Vec::new();
    let mut hit_user_boundary = false;
    for msg in history.iter().rev() {
        if msg.role == "user" {
            hit_user_boundary = true;
        }
        if hit_user_boundary && msg.role == "tool" {
            break;
        }
        if msg.role != "tool" {
            continue;
        }
        let Some(tool_name) = msg.tool_name.as_deref().map(str::trim) else {
            continue;
        };
        let Some(detail) = msg.primary_content() else {
            continue;
        };
        let detail = detail.trim();
        if tool_name.is_empty() || detail.is_empty() {
            continue;
        }
        interaction_results.push((tool_name.to_string(), detail.to_string()));
    }

    choose_completion_recovery_candidate(&interaction_results, max_chars)
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
    pub user_role: UserRole,
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
    pub force_text_response: &'a mut bool,
    pub execution_state: &'a mut ExecutionState,
    pub validation_state: &'a mut ValidationState,
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
        let user_role = ctx.user_role;
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
        let mut validation_state = ctx.validation_state.clone();
        let turn_context = ctx.turn_context;
        let needs_tools_for_turn = ctx.needs_tools_for_turn;
        let mut force_text_response = *ctx.force_text_response;
        let execution_state = &mut *ctx.execution_state;

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
                *ctx.force_text_response = force_text_response;
                *ctx.validation_state = validation_state.clone();
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
                        let candidate =
                            latest_task_tool_result_for_completion(self, session_id, task_id, 2500)
                                .await;
                        reply = build_completion_fallback_reply(
                            candidate.as_ref(),
                            &actions,
                            learning_ctx.tool_calls.len(),
                        );
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
                && needs_tools_for_turn
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
                let candidate =
                    latest_task_tool_result_for_completion(self, session_id, task_id, 2500).await;
                if let Some(candidate) = candidate.as_ref() {
                    if candidate.tool_name == "send_file" {
                        reply = Self::send_file_completion_reply().to_string();
                        recovered = true;
                    } else if let Some(tool_reply) = build_tool_output_completion_reply(
                        &candidate.tool_name,
                        &candidate.tool_output,
                        candidate.artifact_delivered,
                    ) {
                        reply = tool_reply;
                        recovered = true;
                    } else if let Some(tool_reply) = build_structured_tool_output_completion_reply(
                        &candidate.tool_name,
                        &candidate.tool_output,
                        candidate.artifact_delivered,
                    ) {
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
                    reply = build_completion_fallback_reply(
                        candidate.as_ref(),
                        &actions,
                        learning_ctx.tool_calls.len(),
                    );
                }
                info!(
                    session_id,
                    iteration,
                    recovered,
                    "Recovered false capability-denial completion after successful tools"
                );
            }

            let low_signal_completion = is_low_signal_task_lead_reply(&reply);
            let idle_reengagement_completion = looks_like_idle_reengagement_reply(&reply);
            let was_truly_empty = reply.trim().is_empty();
            if should_recover_completion_from_tool_output(
                &reply,
                self.depth,
                total_successful_tool_calls,
            ) || idle_reengagement_completion
            {
                let mut recovered = false;
                let mut candidate_requires_synthesis = false;
                let mut synthesis_retry_scheduled = false;
                let candidate =
                    latest_task_tool_result_for_completion(self, session_id, task_id, 2500).await;
                if let Some(candidate) = candidate.as_ref() {
                    candidate_requires_synthesis = tool_output_requires_final_synthesis(
                        &candidate.tool_name,
                        &candidate.tool_output,
                    );
                    if candidate.tool_name == "send_file" {
                        reply = Self::send_file_completion_reply().to_string();
                        recovered = true;
                        info!(
                            session_id,
                            iteration,
                            "Recovered completion reply after send_file with shared closeout"
                        );
                    } else if candidate.tool_name == "read_file"
                        && learning_ctx.tool_calls.len() > 1
                        && !candidate.artifact_delivered
                    {
                        // When the latest tool is read_file and there were multiple tool
                        // calls, the activity summary is more useful than a raw file dump.
                        // Skip tool-output recovery so the activity summary branch fires.
                        info!(
                            session_id,
                            iteration,
                            tool_call_count = learning_ctx.tool_calls.len(),
                            "Skipping read_file output recovery in favor of activity summary"
                        );
                    } else if let Some(tool_reply) = build_tool_output_completion_reply(
                        &candidate.tool_name,
                        &candidate.tool_output,
                        candidate.artifact_delivered,
                    ) {
                        reply = tool_reply;
                        recovered = true;
                        info!(
                            session_id,
                            iteration,
                            low_signal_completion,
                            idle_reengagement_completion,
                            "Recovered completion reply from latest tool output"
                        );
                    } else if candidate_requires_synthesis {
                        if !empty_response_retry_used {
                            empty_response_retry_used = true;
                            empty_response_retry_pending = true;
                            empty_response_retry_note =
                                Some("structured_tool_output_requires_synthesis".to_string());
                            pending_system_messages
                                .push(structured_result_synthesis_directive(candidate));
                            synthesis_retry_scheduled = true;
                        } else if let Some(tool_reply) =
                            build_structured_tool_output_completion_reply(
                                &candidate.tool_name,
                                &candidate.tool_output,
                                candidate.artifact_delivered,
                            )
                        {
                            reply = tool_reply;
                            recovered = true;
                        }
                        if !recovered {
                            reply.clear();
                            info!(
                                session_id,
                                iteration,
                                tool = %candidate.tool_name,
                                retry_scheduled = synthesis_retry_scheduled,
                                "Deferring structured tool output to synthesis recovery or deterministic fallback"
                            );
                        }
                    }
                }
                // If tool output was trivial/empty and the LLM returned a truly empty
                // response (not just low-signal), don't build an activity summary —
                // leave reply empty so the empty-response retry mechanism kicks in
                // and gives the model another chance to complete the task properly.
                if !recovered
                    && !was_truly_empty
                    && !learning_ctx.tool_calls.is_empty()
                    && !synthesis_retry_scheduled
                {
                    let actions: Vec<&str> =
                        learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                    reply = build_completion_fallback_reply(
                        candidate.as_ref(),
                        &actions,
                        learning_ctx.tool_calls.len(),
                    );
                    info!(
                        session_id,
                        iteration,
                        tool_call_count = learning_ctx.tool_calls.len(),
                        candidate_requires_synthesis,
                        "Built deterministic completion fallback from latest tool result or activity summary"
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
                    execution_state.record_validation_round();
                    validation_state.record_failure(ValidationFailure::ContradictoryEvidence);
                    validation_state.note_retry(LoopRepetitionReason::ContradictoryEvidence);
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::ValidationFailure,
                        "contradictory_file_evidence",
                        "Blocked completion because current file evidence contradicted an earlier read."
                            .to_string(),
                        true,
                    );
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::RetryReason,
                        "contradictory_evidence",
                        "Retried because contradictory file evidence required a fresh re-check."
                            .to_string(),
                        true,
                    );
                    execution_state.mark_persisted_now();
                    self.emit_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Blocked completion until contradictory file evidence is rechecked"
                            .to_string(),
                        json!({
                            "outcome": ValidationOutcome::VerifyAgain,
                            "reason": "contradictory_file_evidence",
                            "loop_repetition_reason": validation_state.loop_repetition_reason,
                            "target_hint": turn_context.completion_contract.primary_target_hint(),
                            "completed_tool_calls": learning_ctx.tool_calls.len(),
                        }),
                    )
                    .await;
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
                execution_state.record_validation_round();
                validation_state.record_failure(ValidationFailure::VerificationPending);
                execution_state.mark_persisted_now();
                if matches!(
                    execution_state.exhausted_limit(0, task_start.elapsed()),
                    Some(ExecutionBudgetLimit::ValidationRounds)
                ) {
                    validation_state.record_failure(ValidationFailure::BudgetExhausted);
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::ValidationFailure,
                        "validation_budget_exhausted",
                        "Stopped final verification because the current validation budget was exhausted."
                            .to_string(),
                        true,
                    );
                    let made_progress = !learning_ctx.tool_calls.is_empty()
                        || completion_progress.mutation_count > 0
                        || completion_progress.observation_count > 0;
                    let request = if made_progress {
                        build_reduce_scope_request(
                            turn_context,
                            learning_ctx,
                            "I used the current validation budget and still do not have a confirmed final result.",
                            "Confirm the narrower scope or exact verification target I should spend the next pass on.",
                            "I will spend the next validation pass on the reduced scope and then report the confirmed outcome.",
                        )
                    } else {
                        build_partial_done_blocked_request(
                            turn_context,
                            learning_ctx,
                            "I used the current validation budget and still do not have a confirmed final result.",
                            "A narrower scope, explicit permission to keep validating, or the exact verification target I should confirm.",
                            "I will spend the next validation pass on a concrete re-check and then report the confirmed outcome.",
                        )
                    };
                    self.emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Surfacing partial result because validation budget is exhausted"
                            .to_string(),
                        json!({
                            "condition": "validation_budget_exhausted",
                            "outcome": request.outcome.clone(),
                            "approval_state": request.approval_state.clone(),
                            "validation_state": validation_state.clone(),
                            "request": request.clone(),
                            "validation_rounds_used": execution_state.validation_rounds_used,
                            "validation_round_budget": execution_state.budget.max_validation_rounds,
                            "execution_id": execution_state.execution_id,
                        }),
                    )
                    .await;
                    reply = request.render_user_message();
                    pending_external_action_ack = None;
                    completion_progress.verification_pending = false;
                } else if tool_defs.is_empty() || force_text_response {
                    validation_state.note_retry(LoopRepetitionReason::VerificationPending);
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::ValidationFailure,
                        "verification_unavailable_in_phase",
                        "Verification was still required, but this phase could not run the needed read-only checks."
                            .to_string(),
                        true,
                    );
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::RetryReason,
                        "verification_pending",
                        "Retried because verification was still pending at completion time."
                            .to_string(),
                        true,
                    );
                    let request = build_partial_done_blocked_request(
                        turn_context,
                        learning_ctx,
                        "I completed part of the request, but the final outcome still needs a read-only verification step.",
                        "A final read-only verification against the current target/output.",
                        "Once verification is available, I will run that check and then report the confirmed result.",
                    );
                    self.emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Surfacing partial result because post-execution verification cannot run in this phase"
                            .to_string(),
                        json!({
                            "outcome": request.outcome.clone(),
                            "approval_state": request.approval_state.clone(),
                            "validation_state": validation_state.clone(),
                            "request": request.clone(),
                            "force_text_response": force_text_response,
                            "tools_available": !tool_defs.is_empty(),
                        }),
                    )
                    .await;
                    warn!(
                        session_id,
                        iteration,
                        force_text_response,
                        "Completion verification required but tools unavailable (empty or force-text); clearing guard"
                    );
                    reply = request.render_user_message();
                    pending_external_action_ack = None;
                    // Avoid deadlocks when tools cannot run in this phase, but
                    // preserve the fact that verification did not happen in the reply itself.
                    completion_progress.verification_pending = false;
                } else {
                    validation_state.note_retry(LoopRepetitionReason::VerificationPending);
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::ValidationFailure,
                        "verification_pending",
                        "Blocked completion until the final verification step could run."
                            .to_string(),
                        true,
                    );
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::RetryReason,
                        "verification_pending",
                        "Retried because verification was still pending at completion time."
                            .to_string(),
                        true,
                    );
                    self.emit_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Post-execution verification required before completion".to_string(),
                        json!({
                            "outcome": ValidationOutcome::VerifyAgain,
                            "reason": "verification_pending",
                            "loop_repetition_reason": validation_state.loop_repetition_reason,
                            "target_hint": turn_context.completion_contract.primary_target_hint(),
                            "completed_tool_calls": learning_ctx.tool_calls.len(),
                            "verification_pending": completion_progress.verification_pending,
                        }),
                    )
                    .await;
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
                        let mut recovered_tool_output = false;
                        let mut needs_synthesis_retry = false;
                        let candidate =
                            latest_task_tool_result_for_completion(self, session_id, task_id, 2500)
                                .await;
                        if let Some(candidate) = candidate.as_ref() {
                            if let Some(tool_reply) = build_force_text_deferred_completion_reply(
                                candidate,
                                learning_ctx.tool_calls.len(),
                            ) {
                                reply = tool_reply;
                                recovered_tool_output = true;
                            } else if tool_output_requires_final_synthesis(
                                &candidate.tool_name,
                                &candidate.tool_output,
                            ) && !empty_response_retry_used
                            {
                                empty_response_retry_used = true;
                                empty_response_retry_pending = true;
                                empty_response_retry_note =
                                    Some("structured_tool_output_requires_synthesis".to_string());
                                pending_system_messages
                                    .push(structured_result_synthesis_directive(candidate));
                                consecutive_clean_iterations = 0;
                                info!(
                                    session_id,
                                    iteration,
                                    tool = %candidate.tool_name,
                                    "Force-text active: retrying once so the model synthesizes the structured tool result"
                                );
                                commit_state!();
                                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                            } else if let Some(tool_reply) =
                                build_structured_tool_output_completion_reply(
                                    &candidate.tool_name,
                                    &candidate.tool_output,
                                    candidate.artifact_delivered,
                                )
                            {
                                reply = tool_reply;
                                recovered_tool_output = true;
                            } else {
                                needs_synthesis_retry = true;
                            }
                        }
                        if !recovered_tool_output {
                            let actions: Vec<&str> =
                                learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                            reply = build_completion_fallback_reply(
                                candidate.as_ref(),
                                &actions,
                                learning_ctx.tool_calls.len(),
                            );
                        }
                        info!(
                            session_id,
                            iteration,
                            total_successful_tool_calls,
                            stall_count,
                            recovered = recovered_tool_output,
                            needs_synthesis_retry,
                            "Force-text active: replaced deferred reply with recovered tool result or activity summary"
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

                    // Check if the deferred-action reply itself contains an
                    // INTENT_GATE marker claiming needs_tools:true — i.e. the model
                    // explicitly told us it needs tool access to fulfil this request.
                    // This is more reliable than `expects_mutation` which also matches
                    // pure text-generation tasks ("write a tweet").
                    let response_claims_needs_tools = {
                        let lower_reply = reply.to_ascii_lowercase();
                        lower_reply.contains(&INTENT_GATE_MARKER.to_ascii_lowercase())
                            && lower_reply.contains("\"needs_tools\":true")
                    };
                    let deferred_nudge = if !has_tool_attempts {
                        if needs_tools_for_turn || response_claims_needs_tools {
                            SystemDirective::DeferredToolCallRequired
                        } else {
                            force_text_response = true;
                            SystemDirective::ToolModeDisabledPlainText
                        }
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
                        let widened = self
                            .restrict_connected_api_setup_tools_for_request(user_text, &widened);
                        let widened = self.ensure_connected_api_tools_exposed(
                            user_text,
                            &widened,
                            base_tool_defs,
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
            validation_state.refresh_success_criteria_matches(&reply);
            if !validation_state.active_success_criteria.is_empty()
                && validation_state.matched_success_criteria.is_empty()
            {
                validation_state.record_failure(ValidationFailure::SuccessCriteriaUnmatched);
            }
            validation_state.clear_loop_repetition_reason();
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
                    user_role,
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
                        user_role,
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
        build_activity_summary_reply, build_completion_fallback_reply,
        build_force_text_deferred_completion_reply, build_structured_tool_output_completion_reply,
        build_tool_output_completion_reply, choose_completion_recovery_candidate,
        extract_structured_tool_output_excerpt, looks_like_idle_reengagement_reply,
        should_enforce_no_tool_text_when_tools_required,
        should_recover_completion_from_tool_output, CompletionRecoveryCandidate,
    };
    use crate::agent::post_task::LearningContext;
    use crate::agent::{
        build_partial_done_blocked_request, history::CompletionTaskKind, CompletionContract,
        TurnContext, VerificationTarget, VerificationTargetKind,
    };
    use chrono::Utc;

    #[test]
    fn tool_output_reply_is_result_focused() {
        let reply = build_tool_output_completion_reply(
            "terminal",
            "cat: /nonexistent/file.txt: No such file",
            false,
        )
        .unwrap();
        assert!(reply.contains("latest tool output"));
        assert!(reply.contains("/nonexistent/file.txt"));
        assert!(!reply.starts_with("Done —"));
    }

    #[test]
    fn tool_output_reply_notes_when_artifact_was_also_delivered() {
        let reply = build_tool_output_completion_reply(
            "terminal",
            "test_foo PASSED\ntest_bar PASSED\n2 passed",
            true,
        )
        .unwrap();
        assert!(reply.contains("sent the requested file"));
        assert!(reply.contains("test_foo PASSED"));
    }

    #[test]
    fn structured_http_tool_output_requires_synthesis() {
        assert!(build_tool_output_completion_reply(
            "http_request",
            "HTTP 200 OK\n{\"items\":[]}",
            true
        )
        .is_none());
    }

    #[test]
    fn structured_tool_output_excerpt_uses_http_body_not_headers() {
        let excerpt = extract_structured_tool_output_excerpt(
            "HTTP 200 OK\ncontent-type: application/json\nserver: nginx\n\n{\"nct_id\":\"NCT05746897\",\"status\":\"Recruiting\"}",
            400,
        )
        .unwrap();

        assert!(excerpt.contains("\"nct_id\":\"NCT05746897\""));
        assert!(excerpt.contains("\"status\":\"Recruiting\""));
        assert!(!excerpt.contains("server: nginx"));
    }

    #[test]
    fn structured_completion_reply_uses_excerpt_for_generic_json() {
        let reply = build_structured_tool_output_completion_reply(
            "project_inspect",
            "{\"status\":\"ok\",\"count\":2}",
            false,
        )
        .unwrap();

        assert!(reply.contains("latest result excerpt"));
        assert!(reply.contains("\"status\":\"ok\""));
        assert!(reply.contains("\"count\":2"));
    }

    #[test]
    fn trivial_tool_output_returns_none() {
        assert!(build_tool_output_completion_reply("terminal", "(no output)", false).is_none());
        assert!(build_tool_output_completion_reply("terminal", "", false).is_none());
        assert!(build_tool_output_completion_reply("terminal", "exit code: 0", false).is_none());
        assert!(build_tool_output_completion_reply(
            "send_file",
            "Duplicate send_file suppressed: this exact file+caption was already sent in this task.",
            false,
        )
        .is_none());
        assert!(build_tool_output_completion_reply(
            "write_file",
            "File written to /tmp/foo.py, 200 bytes",
            false
        )
        .is_none());
        // Directory listing is trivial
        assert!(build_tool_output_completion_reply(
            "terminal",
            "total 24\ndrwxr-xr-x  3 user  wheel  96 Mar  4 21:08 __pycache__\n-rw-r--r--  1 user  wheel  1041 Mar  4 21:09 regex_engine.py\n-rw-r--r--  1 user  wheel  4972 Mar  4 21:03 test_regex.py",
            false,
        ).is_none());
        // Substantive output should still work
        assert!(build_tool_output_completion_reply(
            "terminal",
            "test_foo PASSED\ntest_bar PASSED\n2 passed",
            false,
        )
        .is_some());
    }

    #[test]
    fn completion_recovery_prefers_observational_result_over_delivery_ack() {
        let candidates = vec![
            (
                "send_file".to_string(),
                "File sent: studies.json (127 KB)".to_string(),
            ),
            (
                "http_request".to_string(),
                "HTTP 200 OK\ncontent-type: application/json\n\n{\"studies\":[]}".to_string(),
            ),
        ];

        let selected = choose_completion_recovery_candidate(&candidates, 2500).unwrap();
        assert_eq!(selected.tool_name, "http_request");
        assert!(selected.artifact_delivered);
    }

    #[test]
    fn completion_recovery_returns_delivery_ack_when_no_better_result_exists() {
        let candidates = vec![(
            "send_file".to_string(),
            "File sent: studies.json (127 KB)".to_string(),
        )];

        let selected = choose_completion_recovery_candidate(&candidates, 2500).unwrap();
        assert_eq!(selected.tool_name, "send_file");
        assert!(!selected.artifact_delivered);
    }

    #[test]
    fn force_text_deferred_completion_skips_structured_observational_tool_output() {
        let candidate = choose_completion_recovery_candidate(
            &[(
                "http_request".to_string(),
                "HTTP 200 OK\ncontent-type: application/json\n\n{\"studies\":[]}".to_string(),
            )],
            2500,
        )
        .unwrap();

        assert!(build_force_text_deferred_completion_reply(&candidate, 2).is_none());
    }

    #[test]
    fn force_text_deferred_completion_uses_send_file_closeout() {
        let candidate = choose_completion_recovery_candidate(
            &[(
                "send_file".to_string(),
                "File sent: studies.json (127 KB)".to_string(),
            )],
            2500,
        )
        .unwrap();

        let reply = build_force_text_deferred_completion_reply(&candidate, 1).unwrap();
        assert!(reply.contains("I've sent the requested file"));
    }

    #[test]
    fn force_text_deferred_completion_skips_multi_read_file_dump() {
        let candidate = CompletionRecoveryCandidate {
            tool_name: "read_file".to_string(),
            tool_output: "src/main.rs\nfn main() {}".to_string(),
            artifact_delivered: false,
        };

        assert!(build_force_text_deferred_completion_reply(&candidate, 2).is_none());
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
    fn completion_fallback_prefers_structured_result_excerpt_over_activity_summary() {
        let candidate = CompletionRecoveryCandidate {
            tool_name: "web_fetch".to_string(),
            tool_output: "Title: Trial A\nStatus: Recruiting\nLocation: Fairfax, VA".to_string(),
            artifact_delivered: false,
        };
        let calls = vec![
            "web_search(trial results)",
            "web_fetch(https://example.com/trial-a)",
        ];

        let reply = build_completion_fallback_reply(Some(&candidate), &calls, calls.len());
        assert!(reply.contains("latest result excerpt"));
        assert!(reply.contains("Trial A"));
        assert!(!reply.contains("Activity summary:"));
    }

    #[test]
    fn completion_fallback_keeps_multi_read_file_activity_summary() {
        let candidate = CompletionRecoveryCandidate {
            tool_name: "read_file".to_string(),
            tool_output: "src/main.rs\nfn main() {}".to_string(),
            artifact_delivered: false,
        };
        let calls = vec!["read_file(src/main.rs)", "read_file(src/lib.rs)"];

        let reply = build_completion_fallback_reply(Some(&candidate), &calls, calls.len());
        assert!(reply.contains("Activity summary:"));
        assert!(reply.contains("Files read:"));
        assert!(!reply.contains("latest tool output"));
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
            replay_notes: Vec::new(),
        };

        let request = build_partial_done_blocked_request(
            &turn_context,
            &learning_ctx,
            "I still need a live verification check.",
            "A fresh read-only verification against the deployed URL.",
            "I will run the final verification check and then confirm the deployment state.",
        );
        let reply = request.render_user_message();
        assert!(reply.contains("Current blocker:"));
        assert!(reply.contains("https://blog.aidaemon.ai"));
        assert!(reply.contains("What I need from you:"));
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
    fn idle_reengagement_reply_detected() {
        assert!(looks_like_idle_reengagement_reply(
            "I'm here. What would you like me to help you with?"
        ));
        assert!(looks_like_idle_reengagement_reply(
            "Ready when you are. How can I help?"
        ));
        assert!(!looks_like_idle_reengagement_reply(
            "I found the requested result and included it below."
        ));
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
