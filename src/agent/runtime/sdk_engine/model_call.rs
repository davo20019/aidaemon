//! Model calls: the kernel's provider requests, served through the daemon's
//! router, fallbacks, telemetry and the mandate token lease.

use super::*;

/// Per-call ceilings: the default, a foreground turn's, a mandate run's.
pub(super) const DEFAULT_MODEL_CALL_TIMEOUT: Duration = Duration::from_secs(360);
pub(super) const FOREGROUND_MAX_MODEL_CALL: Duration = Duration::from_secs(90);
pub(super) const MANDATE_MAX_MODEL_CALL: Duration = Duration::from_secs(840);
/// A timed-out call is retried: the ceiling bounds a call so the turn
/// regains control, not so it ends.
pub(super) const MAX_MODEL_CALL_TIMEOUT_RETRIES: u32 = 2;

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
        let reply = loop {
            let reply = self.sdk_model_call(scope, &call.request).await;
            // Only a timed-out call comes back Transient here: the provider
            // stack has already retried everything else it can.
            let Err(sdk::ProviderError::Transient(_)) = &reply else {
                break reply;
            };
            let retries = scope.model_call_timeouts.fetch_add(1, Ordering::Relaxed);
            if retries >= MAX_MODEL_CALL_TIMEOUT_RETRIES {
                break reply;
            }
            let timeout_secs = self.sdk_model_call_timeout().as_secs();
            warn!(
                session_id = scope.session_id,
                task_id = scope.task_id,
                retry = retries + 1,
                "SDK engine: retrying a timed-out model call"
            );
            if let Ok(mut extra) = scope.extra_system.lock() {
                extra.push(
                    crate::agent::system_directives::SystemDirective::ModelCallTimedOut {
                        timeout_secs,
                    }
                    .render(),
                );
            }
        };
        let _ = call.reply.send(reply);
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
        if scope.budget_exhausted.load(Ordering::Relaxed) {
            return Err(sdk::ProviderError::Other(GOAL_BUDGET_EXHAUSTED.into()));
        }
        // A computer_use GUI loop pins a vision-capable model for its task;
        // every later call of the task uses it.
        #[cfg(feature = "computer_use")]
        let pinned_request = {
            let mut pinned = request.clone();
            pinned.model =
                crate::agent::computer_use::resolve_model_for_task(scope.task_id, &pinned.model)
                    .await;
            pinned
        };
        #[cfg(feature = "computer_use")]
        let request = &pinned_request;
        let iteration = scope.model_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let snapshot = self.llm_runtime.snapshot();
        let tools = to_daemon_tools(&request.tools);
        let budget = crate::memory::context_window::compute_available_budget_precomputed(
            &request.model,
            0,
            &tools,
            &self.context_window_config,
        );
        // Hook notes (recalled memory) ride in the wire prefix; the rest of
        // the kernel transcript after the starting prefix is this run's.
        let (notes, kernel) = split_injected_notes(&request.messages);
        let run = &kernel[scope.prefix_len.min(kernel.len())..];
        let extra: Vec<Value> = scope
            .extra_system
            .lock()
            .map(|extra| {
                extra
                    .iter()
                    .map(|text| serde_json::json!({"role": "system", "content": text}))
                    .collect()
            })
            .unwrap_or_default();
        let mut prefix = prefix_with_notes(&scope.prefix, &notes);
        prefix.extend(extra);
        let messages = fit_run(&prefix, run, budget);
        let mut options = to_chat_options(&request.tool_choice, request.max_output_tokens);
        // A reasoning model that spent its whole output budget thinking gets
        // less room to think next time: low effort, then none.
        match scope.thinking_truncations.load(Ordering::Relaxed) {
            0 => {}
            1 => options.reasoning_effort_override = Some("low".to_string()),
            _ => options.reasoning_effort_override = Some("off".to_string()),
        }
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
        if scope.text_only_next_call.swap(false, Ordering::Relaxed) {
            options.tool_choice = crate::traits::ToolChoiceMode::None;
        }
        let est_input_tokens =
            crate::memory::context_window::estimate_multimodal_message_tokens(&messages);
        let mut telemetry = crate::agent::llm::LlmCallTelemetry::default();
        // Mandate runs share one durable token balance across the task lead
        // and its executors: reserve before the call, settle after it.
        let lease = match self
            .sdk_mandate_token_lease(&mut options, est_input_tokens as u32)
            .await
        {
            Ok(lease) => lease,
            Err(error) => {
                // The balance is used up. A committed WAIT/ASK/STOP already
                // completed the review; otherwise a bounded WAIT is recorded
                // so the controller retries. The
                // review closes from that durable decision, not as a
                // provider failure.
                if error.to_string().contains(MANDATE_BUDGET_EXHAUSTED) {
                    if let Some(summary) =
                        crate::agent::mandate_run::recover_mandate_capacity_as_wait(self).await
                    {
                        info!(
                            task_id = scope.task_id,
                            "SDK engine: mandate review closed from a committed or recovered \
                             non-action decision"
                        );
                        scope.end_with(summary, TaskOutcome::Succeeded);
                    }
                }
                return Err(error);
            }
        };
        let started = Instant::now();
        let timeout = self.sdk_model_call_timeout();
        let result = match tokio::time::timeout(
            timeout,
            self.call_llm_with_recovery(
                snapshot.provider(),
                snapshot.router(),
                &request.model,
                &messages,
                &tools,
                &options,
                &mut telemetry,
                None,
            ),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(anyhow::Error::new(ModelCallTimedOut(timeout))),
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
            crate::memory::context_window::record_token_estimate_calibration(
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
                return Err(sdk::ProviderError::Other(error.to_string()));
            }
        };
        let produced_output = !response.tool_calls.is_empty()
            || response
                .content
                .as_deref()
                .is_some_and(|content| !content.trim().is_empty());
        if produced_output {
            scope.thinking_truncations.store(0, Ordering::Relaxed);
        } else if response
            .response_note
            .as_deref()
            .is_some_and(|note| note.contains("truncated"))
        {
            let count = scope.thinking_truncations.fetch_add(1, Ordering::Relaxed) + 1;
            warn!(
                task_id = scope.task_id,
                consecutive_truncations = count,
                "SDK engine: response truncated at the token limit with no output; lowering \
                 reasoning effort for the next call"
            );
        }
        if let Some(usage) = usage.as_ref() {
            if let Some(stop) = self
                .sdk_charge_budgets(scope, usage, iteration as usize)
                .await
            {
                // Closeout grace: a final answer on the first call,
                // or after real progress, still ships; otherwise the budget
                // reply replaces it and no further work runs.
                let grace = response.tool_calls.is_empty()
                    && (iteration == 1 || scope.succeeded_calls.load(Ordering::Relaxed) > 0);
                if !grace {
                    scope.end_with(stop, TaskOutcome::Failed);
                    return Err(sdk::ProviderError::Other(GOAL_BUDGET_EXHAUSTED.into()));
                }
            }
        }
        if let Some(note) = response.response_note.clone() {
            if let Ok(mut latest) = scope.verdict.response_note.lock() {
                *latest = Some(note);
            }
        }
        scope
            .last_batch_size
            .store(response.tool_calls.len(), Ordering::Relaxed);
        let response = from_provider_response(response);
        if !response.tool_calls.is_empty() {
            self.persist_sdk_tool_call_message(scope, &response).await;
        }
        Ok(response)
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
