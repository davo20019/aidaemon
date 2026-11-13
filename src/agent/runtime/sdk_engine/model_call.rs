//! Model calls: the kernel's provider requests, served through the daemon's
//! router, fallbacks, telemetry and the mandate token lease.

use super::*;
#[cfg(test)]
pub(super) use aidaemon_runtime::engine::hooks::{
    raised_output_ceiling, thinking_cutoff_effort, MAX_RAISED_OUTPUT_TOKENS,
};

/// Per-call ceilings: the default, a foreground turn's, a mandate run's.
pub(super) const DEFAULT_MODEL_CALL_TIMEOUT: Duration = Duration::from_secs(360);
pub(super) const FOREGROUND_MAX_MODEL_CALL: Duration = Duration::from_secs(90);
pub(super) const MANDATE_MAX_MODEL_CALL: Duration = Duration::from_secs(840);
/// Total time a call may take while its stream keeps delivering chunks; the
/// per-call ceiling above bounds silence, not progress.
pub(super) const STREAMING_MAX_MODEL_CALL: Duration = Duration::from_secs(600);
/// A timed-out call is retried: the ceiling bounds a call so the turn
/// regains control, not so it ends.
pub(super) const MAX_MODEL_CALL_TIMEOUT_RETRIES: u32 = 2;

/// Retries after a context-overflow rejection, each with a smaller budget.
pub(super) const MAX_CONTEXT_OVERFLOW_RETRIES: u32 = 2;

/// Heads a user message taken into a running turn. It joins the request
/// the turn serves rather than replacing it: several questions sent back to
/// back all arrive this way, and each still needs its answer.
pub(super) const STEERING_MARKER: &str =
    "[The user sent this while you were working. It adds to the current request: \
     still answer or finish everything they asked before it, unless this message \
     changes or cancels that.]";

/// Follows [`STEERING_MARKER`] once the run has declared a checklist. Only
/// the model can tell whether a message changes the work, and the checklist
/// is its own statement of that work, so this asks and nothing enforces it.
pub(super) const STEERING_CHECKLIST_NOTE: &str =
    "[You declared a checklist for this request. If this message adds, changes or cancels \
     work, call track_requirements with the full revised list before continuing. If it only \
     clarifies or grants permission, leave the list as it is.]";

/// Run `call`, failing once it has been silent for `idle` or has run for
/// `hard` in all. Silence is measured from the latest streamed chunk, so a
/// streaming call that keeps producing output is not cut off at `idle`; a
/// provider that does not stream reports nothing and gets `idle` in all.
/// The error carries the limit that fired.
/// The first provider attempt's own deadline under an idle watchdog of
/// `idle`: a little shorter, so a hung primary hands over to the fallback
/// chain (whose switch event counts as progress) instead of the watchdog
/// cancelling the whole call.
pub(super) fn first_attempt_deadline(idle: Duration) -> Duration {
    idle.saturating_sub(Duration::from_secs(2))
        .max(Duration::from_secs(10))
}

pub(super) async fn with_idle_timeout<F: std::future::Future>(
    idle: Duration,
    hard: Duration,
    call: F,
) -> Result<F::Output, Duration> {
    let progress = crate::providers::streaming::StreamProgress::new();
    let started = Instant::now();
    let call = crate::providers::streaming::with_stream_progress(progress.clone(), call);
    tokio::pin!(call);
    loop {
        let idle_deadline = progress.last_activity() + idle;
        let hard_deadline = started + hard;
        tokio::select! {
            output = &mut call => return Ok(output),
            _ = tokio::time::sleep_until(idle_deadline.min(hard_deadline).into()) => {
                let now = Instant::now();
                if now >= hard_deadline {
                    return Err(hard);
                }
                if now >= progress.last_activity() + idle {
                    return Err(idle);
                }
            }
        }
    }
}

/// `run` with each placed message (a system directive, or a user message
/// taken mid-turn) inserted before the run message at its position (after
/// the last one when it is at the end).
pub(super) fn interleave_directives(run: Vec<Value>, directives: &[(usize, Value)]) -> Vec<Value> {
    if directives.is_empty() {
        return run;
    }
    let len = run.len();
    let at = |index: usize| {
        directives
            .iter()
            .filter(move |(position, _)| (*position).min(len) == index)
            .map(|(_, message)| message.clone())
    };
    let mut out = Vec::with_capacity(len + directives.len());
    for (index, message) in run.into_iter().enumerate() {
        out.extend(at(index));
        out.push(message);
    }
    out.extend(at(len));
    out
}

/// A model call that ran past its per-call ceiling.
#[derive(Debug)]
pub(super) struct ModelCallTimedOut(Duration);

impl std::fmt::Display for ModelCallTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "model call timed out after {}s", self.0.as_secs())
    }
}

impl std::error::Error for ModelCallTimedOut {}

impl Agent {
    /// Make one model call for the SDK run through the daemon's provider
    /// stack: billing-aware retries, the fallback cascade and failover
    /// providers all apply.
    pub(super) async fn serve_sdk_model_call(&self, scope: &TurnScope<'_>, call: ModelCall) {
        let reply = self.sdk_model_call(scope, &call.request).await;
        if reply.is_ok() {
            // Every message taken so far rode in this call (placed messages
            // stay pinned in each later call): the model has answered it with
            // its tools in reach.
            if let Ok(mut absorbed) = scope.absorbed_steering.lock() {
                for entry in absorbed.iter_mut() {
                    entry.answered = true;
                }
            }
        }
        let _ = call.reply.send(reply);
    }

    /// Whether to try a failed model call again. The kernel runs the retry
    /// as a new turn, so `before_model` hooks shape it afresh.
    pub(super) async fn serve_sdk_model_error(
        &self,
        scope: &TurnScope<'_>,
        check: ModelErrorCheck,
    ) {
        let _ = check
            .reply
            .send(self.model_error_decision(scope, check.timed_out));
    }

    fn model_error_decision(
        &self,
        scope: &TurnScope<'_>,
        timed_out: bool,
    ) -> sdk::ModelErrorDecision {
        let retry = sdk::ModelErrorDecision::Retry {
            after: Duration::ZERO,
        };
        // The provider rejected the request as too long: its window is now
        // recorded, so the retry compacts the run to fit.
        if scope.context_overflowed.swap(false, Ordering::Relaxed)
            && scope
                .context_overflow_retries
                .fetch_add(1, Ordering::Relaxed)
                < MAX_CONTEXT_OVERFLOW_RETRIES
        {
            return retry;
        }
        if !timed_out {
            return sdk::ModelErrorDecision::Fail;
        }
        let retries = scope.model_call_timeouts.fetch_add(1, Ordering::Relaxed);
        if retries >= MAX_MODEL_CALL_TIMEOUT_RETRIES {
            return sdk::ModelErrorDecision::Fail;
        }
        warn!(
            session_id = scope.session_id,
            task_id = scope.task_id,
            retry = retries + 1,
            "SDK engine: retrying a timed-out model call"
        );
        if let Ok(mut extra) = scope.extra_system.lock() {
            extra.push(
                crate::agent::system_directives::SystemDirective::ModelCallTimedOut {
                    timeout_secs: self.sdk_model_call_timeout().as_secs(),
                }
                .render(),
            );
        }
        retry
    }

    /// The per-call ceiling: a hung provider must not hold the turn.
    pub(super) fn sdk_model_call_timeout(&self) -> Duration {
        let ceiling = if self.mandate_execution.is_some() {
            MANDATE_MAX_MODEL_CALL
        } else {
            FOREGROUND_MAX_MODEL_CALL
        };
        self.limits
            .llm_call_timeout
            .unwrap_or(DEFAULT_MODEL_CALL_TIMEOUT)
            .min(ceiling)
    }

    pub(super) async fn sdk_model_call(
        &self,
        scope: &TurnScope<'_>,
        request: &sdk::ModelRequest,
    ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
        let iteration = scope.model_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let snapshot = self.llm_runtime.snapshot();
        let tools = to_daemon_tools(&request.tools);
        let budget = crate::context_budget::compute_available_budget_precomputed(
            &request.model,
            0,
            &tools,
            &self.runtime.context_window_config,
        );
        // Hook notes (recalled memory) ride in the wire prefix; the rest of
        // the kernel transcript after the starting prefix is this run's.
        let (notes, kernel) = split_injected_notes(&request.messages);
        let run = &kernel[scope.prefix_len.min(kernel.len())..];
        let prefix = prefix_with_notes(&scope.prefix, &notes);
        // Directives raised since the last call are pinned where the run
        // stands now, so the sent transcript only ever grows at its end.
        let run = to_daemon_messages_from(run);
        // A parent hears about its children here: pinned at the run's end
        // like any other directive, so the sent prefix never changes.
        if let Some(note) = scope
            .spawn_family
            .active_coordination()
            .and_then(|coordination| coordination.parent_note(scope.session_id))
        {
            if let Ok(mut extra) = scope.extra_system.lock() {
                extra.push(note);
            }
        }
        self.sdk_note_delivered_notices(scope).await;
        let run = {
            let raised: Vec<String> = scope
                .extra_system
                .lock()
                .map(|mut queue| std::mem::take(&mut *queue))
                .unwrap_or_default();
            // A call that offers no tools (a text-only reply retry) cannot
            // act on a new message: it stays queued for its own turn.
            let offers_tools =
                !request.tools.is_empty() && request.tool_choice != sdk::ToolChoice::None;
            let steering = if offers_tools {
                self.sdk_take_steering(scope).await
            } else {
                Vec::new()
            };
            let mut placed = scope
                .placed_directives
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            placed.extend(raised.into_iter().map(|text| {
                (
                    run.len(),
                    serde_json::json!({"role": "system", "content": text}),
                )
            }));
            placed.extend(steering.into_iter().map(|message| (run.len(), message)));
            interleave_directives(run, &placed)
        };
        // An oversized run is compacted (older rounds summarized) before
        // quota fitting, which stays as the fallback.
        let prefix_tokens = crate::context_budget::estimate_multimodal_message_tokens(&prefix);
        let run = self
            .sdk_compacted_run(scope, &request.model, prefix_tokens, &run, budget)
            .await;
        let messages = fit_run_values(&prefix, run, budget);
        crate::otel_export::observe_sent(scope.run_id, &messages);
        let mut options = to_chat_options(request);
        options.cache_key = Some(scope.session_id.to_string());
        // llama.cpp slot pinning (opt-in): the root interactive agent carries
        // `Some(interactive_slot)` to keep its KV cache warm; sub-agents carry
        // `None` and use the provider's background slot.
        options.id_slot = self.interactive_slot;
        // Before any tool work a provider blip gets the long retry ladder: a
        // failure there loses the whole turn.
        options.uncommitted_work = scope
            .learning
            .lock()
            .map(|learning| learning.tool_calls.is_empty())
            .unwrap_or(true);
        let est_input_tokens = crate::context_budget::estimate_multimodal_message_tokens(&messages);
        let mut telemetry = crate::agent::llm::LlmCallTelemetry::default();
        // Mandate runs share one durable token balance across the task lead
        // and its executors: reserve before the call, settle after it.
        let lease = match self
            .sdk_mandate_token_lease(&mut options, est_input_tokens as u32)
            .await
        {
            Ok(lease) => lease,
            Err(error) => {
                // The balance is used up. A WAIT/ASK/STOP the deliberator
                // already committed completed the review, which closes from
                // that decision. Without one the call fails, and the review
                // closes as failed: retried with backoff, owner told.
                if error.to_string().contains(MANDATE_BUDGET_EXHAUSTED) {
                    if let Some(summary) =
                        crate::agent::mandate_run::committed_non_action_mandate_summary(self).await
                    {
                        info!(
                            task_id = scope.task_id,
                            "SDK engine: mandate review closed from a committed non-action decision"
                        );
                        scope
                            .verdict
                            .record_budget_stop(summary, TaskOutcome::Succeeded);
                    }
                }
                return Err(error);
            }
        };
        let started = Instant::now();
        let idle_timeout = self.sdk_model_call_timeout();
        let result = match with_idle_timeout(
            idle_timeout,
            idle_timeout.max(STREAMING_MAX_MODEL_CALL),
            crate::agent::llm::call_llm_with_recovery(
                self,
                snapshot.provider(),
                snapshot.router(),
                &request.model,
                &messages,
                &tools,
                &options,
                &mut telemetry,
                None,
                first_attempt_deadline(idle_timeout),
            ),
        )
        .await
        {
            Ok(result) => result,
            Err(limit) => Err(anyhow::Error::new(ModelCallTimedOut(limit))),
        };
        let latency_ms = started.elapsed().as_millis() as u64;
        if let Some(lease) = lease.as_ref() {
            match &result {
                Ok(response) => {
                    let delta = response
                        .usage
                        .as_ref()
                        .map(|usage| i64::try_from(usage.budget_tokens()).unwrap_or(i64::MAX))
                        .unwrap_or_else(|| {
                            lease
                                .budget_per_cycle
                                .saturating_sub(lease.tokens_used_before)
                        });
                    if let Err(error) = self
                        .state
                        .settle_mandate_run_token_lease(
                            &lease.goal_run_id,
                            &lease.lease_token,
                            delta,
                        )
                        .await
                    {
                        warn!(task_id = scope.task_id, %error, "SDK engine: could not settle mandate spend");
                    }
                }
                // Every model refused before doing any work: nothing was
                // spent, so nothing is charged.
                Err(error) if spent_nothing(error) => {
                    if let Err(error) = self
                        .state
                        .settle_mandate_run_token_lease(&lease.goal_run_id, &lease.lease_token, 0)
                        .await
                    {
                        warn!(task_id = scope.task_id, %error, "SDK engine: could not settle mandate spend");
                    }
                }
                Err(_) => {
                    crate::agent::mandate_run::exhaust_mandate_run_token_lease_after_ambiguous_call(
                        self,
                        Some(lease),
                        scope.session_id,
                    )
                    .await;
                }
            }
        }
        // Every call is recorded, failed ones too, so failure telemetry
        // (`db_probe --gate-outcomes`, provider error kinds) sees SDK turns.
        let usage = result
            .as_ref()
            .ok()
            .and_then(|response| response.usage.clone());
        let (input_tokens, output_tokens) = usage
            .as_ref()
            .map(|usage| (usage.input_tokens, usage.output_tokens))
            .unwrap_or_default();
        let failure = result.as_ref().err();
        let provider_error =
            failure.and_then(|error| error.downcast_ref::<crate::providers::ProviderError>());
        let chosen_tools: Vec<String> = result
            .as_ref()
            .map(|response| {
                response
                    .tool_calls
                    .iter()
                    .map(|tc| tc.name.clone())
                    .collect()
            })
            .unwrap_or_default();
        crate::events::record_model_call_telemetry(
            scope.emitter,
            self.state.as_ref(),
            crate::events::ModelCallTelemetryInput {
                session_id: scope.session_id.to_string(),
                task_id: scope.task_id.to_string(),
                call_purpose: None,
                iteration: Some(iteration as u32),
                llm_call: crate::events::LlmCallData {
                    call_id: None,
                    call_purpose: None,
                    task_id: scope.task_id.to_string(),
                    iteration: Some(iteration as u32),
                    model: request.model.clone(),
                    final_model: Some(if telemetry.final_model.is_empty() {
                        request.model.clone()
                    } else {
                        telemetry.final_model.clone()
                    }),
                    fell_back: telemetry.fell_back,
                    attempts: if failure.is_some() {
                        telemetry.attempts.max(1)
                    } else {
                        telemetry.attempts
                    },
                    latency_ms,
                    prompt_ms: usage.as_ref().and_then(|usage| usage.prompt_ms),
                    decode_ms: usage.as_ref().and_then(|usage| usage.decode_ms),
                    input_tokens,
                    output_tokens,
                    cached_input_tokens: usage.as_ref().and_then(|u| u.cached_input_tokens),
                    cache_creation_input_tokens: usage
                        .as_ref()
                        .and_then(|u| u.cache_creation_input_tokens),
                    fresh_input_tokens: usage.as_ref().and_then(|u| u.fresh_input_tokens()),
                    est_input_tokens: Some(est_input_tokens as u32),
                    tool_calls_count: chosen_tools.len() as u32,
                    build_ms: None,
                    prefix_hash_system: None,
                    prefix_hash_pre_boundary: None,
                    tool_defs_hash: None,
                    offered_tools: request.tools.iter().map(|tool| tool.name.clone()).collect(),
                    chosen_tools,
                    session_summary_hash: None,
                    tail_hash: None,
                    prefix_hash_archived: None,
                    boundary_pos: None,
                    message_count: Some(messages.len()),
                    projected_source_message_ids: scope.projected_source_message_ids.clone(),
                    projected_source_turn_ids: scope.projected_source_turn_ids.clone(),
                    force_text: false,
                    token_usage_present: usage.is_some(),
                    token_usage_evidence: crate::events::TokenUsageEvidence::Unavailable,
                    failed: failure.is_some(),
                    error: failure.map(|error| error.to_string()),
                    provider_error_kind: provider_error.map(|error| error.kind),
                    provider_status: provider_error.and_then(|error| error.status),
                },
                token_usage: usage.clone(),
            },
        )
        .await;
        if let Some(error) = failure {
            let _ = scope
                .emitter
                .emit(
                    EventType::Error,
                    crate::events::ErrorData::llm_error(
                        error.to_string(),
                        Some(scope.task_id.to_string()),
                    )
                    .with_context("llm_call_failed"),
                )
                .await;
        } else if usage.is_some() {
            // Calibrates the chars/4 estimate the summarization threshold uses.
            crate::context_budget::record_token_estimate_calibration(
                &request.model,
                est_input_tokens,
                input_tokens as usize,
            );
        }
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                warn!(
                    session_id = scope.session_id,
                    task_id = scope.task_id,
                    model = %request.model,
                    %error,
                    "SDK engine: model call failed after recovery"
                );
                if error.is::<ModelCallTimedOut>() {
                    return Err(sdk::ProviderError::Transient(error.to_string()));
                }
                if let Some(stated) = crate::providers::context_overflow(&error.to_string()) {
                    let budget = crate::config::record_context_overflow(
                        &request.model,
                        stated,
                        est_input_tokens,
                    );
                    warn!(
                        task_id = scope.task_id,
                        model = %request.model,
                        est_input_tokens,
                        ?stated,
                        budget,
                        "SDK engine: the model's context window is smaller than assumed; \
                         recorded it and retrying with the run compacted"
                    );
                    scope.context_overflowed.store(true, Ordering::Relaxed);
                }
                if let (Some(typed), Ok(mut slot)) = (
                    error.downcast_ref::<crate::providers::ProviderError>(),
                    scope.provider_failure.lock(),
                ) {
                    *slot = Some(typed.clone());
                }
                // A used-up quota stays typed: the reply names it and its
                // reset instead of a generic provider error.
                if let Some(quota) = error
                    .downcast_ref::<crate::providers::ProviderError>()
                    .filter(|e| e.kind == crate::providers::ProviderErrorKind::QuotaExhausted)
                {
                    return Err(sdk::ProviderError::QuotaExhausted {
                        message: error.to_string(),
                        resets_at: quota.resets_at,
                        per_model: false,
                        in_stream: !quota.spent_nothing,
                    });
                }
                return Err(sdk::ProviderError::Other(error.to_string()));
            }
        };
        *scope
            .pending_charge
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = usage;
        if let Some(note) = response.response_note.clone() {
            if let Ok(mut latest) = scope.verdict.response_note.lock() {
                *latest = Some(note);
            }
        }
        scope
            .last_batch_size
            .store(response.tool_calls.len(), Ordering::Relaxed);
        Ok(from_provider_response(response))
    }

    /// Notices delivered to this chat since the last model call of the turn
    /// (`notices_turn_context`), raised as context so the reply never
    /// contradicts what the owner was just told. Only the turn serving the
    /// chat reads them: children and mandate runs have sessions of their own.
    async fn sdk_note_delivered_notices(&self, scope: &TurnScope<'_>) {
        if self.depth > 0 || self.mandate_execution.is_some() {
            return;
        }
        let Ok(since) = scope.notices_seen.lock().map(|seen| seen.0) else {
            return;
        };
        let notices = match self
            .state
            .notices_delivered_since(scope.session_id, since)
            .await
        {
            Ok(notices) => notices,
            Err(error) => {
                warn!(task_id = scope.task_id, %error, "SDK engine: could not read notices sent during the turn");
                return;
            }
        };
        let fresh: Vec<_> = match scope.notices_seen.lock() {
            Ok(mut seen) => notices
                .into_iter()
                .filter(|notice| seen.1.insert(notice.id.clone()))
                .collect(),
            Err(_) => return,
        };
        let Some(context) = aidaemon_work::goals::notices_turn_context(&fresh) else {
            return;
        };
        if let Ok(mut extra) = scope.extra_system.lock() {
            extra.push(context);
        }
    }

    /// Messages the same sender sent while this root turn runs, taken from
    /// the channel queue (see `TaskRegistry::take_steering_messages`),
    /// recorded in the conversation history, and returned as user messages
    /// for the run. Mandate runs and agents without a channel queue take
    /// none.
    pub(super) async fn sdk_take_steering(&self, scope: &TurnScope<'_>) -> Vec<Value> {
        // The queue is keyed by the chat's session, which only the turn
        // serving that chat runs in: children and goal runs have sessions of
        // their own. Mandate runs never take user input.
        if self.mandate_execution.is_some() {
            return Vec::new();
        }
        let Some(registry) = self.task_registry.get() else {
            return Vec::new();
        };
        let taken = registry
            .take_steering_messages(scope.session_id, scope.user_role, scope.channel_ctx)
            .await;
        let header = if scope.checklist_declared.load(Ordering::Relaxed) {
            format!("{STEERING_MARKER}\n{STEERING_CHECKLIST_NOTE}")
        } else {
            STEERING_MARKER.to_string()
        };
        let mut messages = Vec::with_capacity(taken.len());
        for queued in taken {
            let mut message =
                Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "user");
            message.content = Some(queued.text.clone());
            message.turn_id = Some(scope.turn_id.to_string());
            if let Err(error) = self
                .append_user_message_with_event(
                    scope.emitter,
                    &message,
                    scope.user_role,
                    scope.channel_ctx,
                    false,
                )
                .await
            {
                warn!(task_id = scope.task_id, %error, "SDK engine: could not record a mid-turn message");
            }
            info!(
                task_id = scope.task_id,
                chars = queued.text.len(),
                "SDK engine: took a message sent mid-turn into the run"
            );
            messages.push(serde_json::json!({
                "role": "user",
                "content": format!("{header}\n{}", queued.text),
            }));
            if let Ok(mut absorbed) = scope.absorbed_steering.lock() {
                absorbed.push(AbsorbedSteering {
                    message: queued,
                    answered: false,
                });
            }
        }
        messages
    }

    /// At run end, hand every message taken mid-turn that no model call
    /// answered (the call that carried it failed and the run ended) back to
    /// the channel queue, so it runs as its own turn right after this one
    /// instead of being dropped.
    pub(super) async fn sdk_return_unanswered_steering(&self, scope: &TurnScope<'_>) {
        let unanswered: Vec<crate::tasks::QueuedMessage> = scope
            .absorbed_steering
            .lock()
            .map(|mut absorbed| {
                std::mem::take(&mut *absorbed)
                    .into_iter()
                    .filter(|entry| !entry.answered)
                    .map(|entry| entry.message)
                    .collect()
            })
            .unwrap_or_default();
        if unanswered.is_empty() {
            return;
        }
        let Some(registry) = self.task_registry.get() else {
            return;
        };
        warn!(
            task_id = scope.task_id,
            count = unanswered.len(),
            "SDK engine: a mid-turn message was never answered; queued for its own turn"
        );
        registry
            .return_steering_messages(scope.session_id, unanswered)
            .await;
    }

    /// Admit a mandate run's model call against its durable balance. Returns
    /// the lease to settle, or an error that ends the run when the balance
    /// cannot admit the call.
    pub(super) async fn sdk_mandate_token_lease(
        &self,
        options: &mut crate::traits::ChatOptions,
        est_input_tokens: u32,
    ) -> Result<Option<crate::agent::mandate_run::MandateRunTokenLease>, sdk::ProviderError> {
        use crate::agent::mandate_run::{
            acquire_mandate_run_token_lease, mandate_output_token_ceiling,
            release_mandate_run_token_lease_without_dispatch, MandateRunTokenAdmission,
        };
        let lease_secs = i64::try_from(MANDATE_MAX_MODEL_CALL.as_secs())
            .unwrap_or(900)
            .saturating_add(30);
        let admission = acquire_mandate_run_token_lease(self, lease_secs, &None)
            .await
            .map_err(|error| sdk::ProviderError::Other(error.to_string()))?;
        match admission {
            MandateRunTokenAdmission::NotApplicable => Ok(None),
            MandateRunTokenAdmission::Exhausted => {
                Err(sdk::ProviderError::Other(MANDATE_BUDGET_EXHAUSTED.into()))
            }
            MandateRunTokenAdmission::Acquired(lease) => {
                let remaining = lease
                    .budget_per_cycle
                    .saturating_sub(lease.tokens_used_before);
                let Some(ceiling) = mandate_output_token_ceiling(remaining, est_input_tokens)
                else {
                    release_mandate_run_token_lease_without_dispatch(self, Some(&lease), "").await;
                    return Err(sdk::ProviderError::Other(MANDATE_BUDGET_EXHAUSTED.into()));
                };
                options.max_tokens_override = Some(
                    options
                        .max_tokens_override
                        .map_or(ceiling, |configured| configured.min(ceiling)),
                );
                options.single_attempt_fail_closed = true;
                // A model that refused before doing any work spent nothing,
                // so the next model may answer; anything else stays one shot.
                options.fallback_when_unspent = true;
                // Durably mark the reservation dispatched before the provider
                // sees the request, so an interrupted call settles fail-closed.
                let fence = self
                    .mandate_execution
                    .as_ref()
                    .ok_or_else(|| sdk::ProviderError::Other("mandate fence missing".into()))?;
                let dispatched = self
                    .state
                    .mark_mandate_run_token_lease_dispatched(
                        &lease.goal_run_id,
                        &fence.mandate_id,
                        fence.mandate_version,
                        &lease.lease_token,
                    )
                    .await
                    .unwrap_or(false);
                if !dispatched {
                    release_mandate_run_token_lease_without_dispatch(self, Some(&lease), "").await;
                    return Err(sdk::ProviderError::Other(
                        "mandate model call was not durably reserved before provider dispatch"
                            .into(),
                    ));
                }
                Ok(Some(lease))
            }
        }
    }
}

/// The failed call provably used no tokens. A failover chain only moves
/// past such errors on a fenced call and stops at any other, so the error
/// it ends with speaks for every attempt.
fn spent_nothing(error: &anyhow::Error) -> bool {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<crate::providers::ProviderError>())
        .is_some_and(|error| error.spent_nothing)
}
