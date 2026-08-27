use std::collections::HashSet;

use super::turn_transition::{TurnRestartReason, TurnTransition};
use super::*;
use crate::events::TaskOutcome;
use crate::traits::ProviderResponse;

/// A JoinHandle wrapper that aborts the spawned task when dropped.
/// Mirrors the identical type in `tool_execution/types.rs`; defined here so
/// `llm_phase` (which lives at the `crate::agent` level) can use it without
/// widening the visibility of the tool_execution-private copy.
struct AbortOnDrop(tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Touch the heartbeat every 30 s while held, so the channel stale-watchdog
/// does not cancel a slow-but-progressing LLM generation.  Auto-aborted on
/// drop (bounded in practice by the LLM call's own timeout).  No-op when
/// `heartbeat` is None.
fn spawn_heartbeat_keeper(heartbeat: &Option<Arc<AtomicU64>>) -> Option<AbortOnDrop> {
    heartbeat.as_ref().map(|hb| {
        let hb = Arc::clone(hb);
        AbortOnDrop(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                hb.store(now, Ordering::Relaxed);
            }
        }))
    })
}

#[derive(Debug)]
struct MandateRunTokenLease {
    goal_run_id: String,
    lease_token: String,
    tokens_used_before: i64,
    budget_per_cycle: i64,
}

enum MandateRunTokenAdmission {
    NotApplicable,
    Acquired(MandateRunTokenLease),
    Exhausted {
        tokens_used: i64,
        budget_per_cycle: i64,
    },
}

/// Serialize every task-lead/executor model call through the exact mandate
/// goal run and its immutable durable token balance. Waiting workers poll the
/// short lease. If a process dies after dispatch, expiry consumes the remaining
/// balance fail-closed instead of admitting a retry with unknowable prior spend.
async fn acquire_mandate_run_token_lease(
    agent: &Agent,
    lease_secs: i64,
    heartbeat: &Option<Arc<AtomicU64>>,
) -> anyhow::Result<MandateRunTokenAdmission> {
    let Some(fence) = agent.mandate_execution.as_ref() else {
        return Ok(MandateRunTokenAdmission::NotApplicable);
    };
    let goal = agent
        .state
        .get_goal(&fence.goal_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("mandate controller goal is missing"))?;
    let configured_budget = goal
        .budget_per_check
        .filter(|budget| *budget > 0)
        .ok_or_else(|| anyhow::anyhow!("mandate controller has no positive cycle token budget"))?;
    let (budget_per_cycle, tokens_used) = agent
        .state
        .ensure_mandate_run_token_budget(
            &fence.goal_run_id,
            &fence.mandate_id,
            fence.mandate_version,
            configured_budget,
        )
        .await?;
    if tokens_used >= budget_per_cycle {
        return Ok(MandateRunTokenAdmission::Exhausted {
            tokens_used,
            budget_per_cycle,
        });
    }

    let lease_token = Uuid::new_v4().to_string();
    loop {
        let (acquired, current_tokens, current_budget) = agent
            .state
            .try_acquire_mandate_run_token_lease(
                &fence.goal_run_id,
                &fence.mandate_id,
                fence.mandate_version,
                &lease_token,
                lease_secs,
            )
            .await?;
        if acquired {
            return Ok(MandateRunTokenAdmission::Acquired(MandateRunTokenLease {
                goal_run_id: fence.goal_run_id.clone(),
                lease_token,
                tokens_used_before: current_tokens,
                budget_per_cycle: current_budget,
            }));
        }
        if current_tokens >= current_budget {
            return Ok(MandateRunTokenAdmission::Exhausted {
                tokens_used: current_tokens,
                budget_per_cycle: current_budget,
            });
        }
        if agent
            .cancel_token
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            anyhow::bail!("mandate model-call admission cancelled while waiting for cycle lease");
        }
        touch_heartbeat(heartbeat);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn release_mandate_run_token_lease_without_dispatch(
    agent: &Agent,
    lease: Option<&MandateRunTokenLease>,
    session_id: &str,
) {
    let Some(lease) = lease else {
        return;
    };
    match agent
        .state
        .release_mandate_run_token_lease(&lease.goal_run_id, &lease.lease_token)
        .await
    {
        Ok(true) => {}
        Ok(false) => warn!(
            session_id,
            goal_run_id = %lease.goal_run_id,
            "Mandate model-call token lease was not released before provider dispatch"
        ),
        Err(error) => warn!(
            session_id,
            goal_run_id = %lease.goal_run_id,
            error = %error,
            "Failed to release mandate model-call token lease before provider dispatch"
        ),
    }
}

/// Once dispatch begins, a provider error or timeout does not prove that no
/// tokens were consumed. Charge the entire remaining balance and release the
/// lease atomically; retrying unknown spend would fail open.
async fn exhaust_mandate_run_token_lease_after_ambiguous_call(
    agent: &Agent,
    lease: Option<&MandateRunTokenLease>,
    session_id: &str,
) {
    let Some(lease) = lease else {
        return;
    };
    let remaining = lease
        .budget_per_cycle
        .saturating_sub(lease.tokens_used_before);
    match agent
        .state
        .settle_mandate_run_token_lease(&lease.goal_run_id, &lease.lease_token, remaining)
        .await
    {
        Ok((tokens_used, budget_per_cycle)) => warn!(
            session_id,
            goal_run_id = %lease.goal_run_id,
            tokens_used,
            budget_per_cycle,
            "Provider call failed without trustworthy usage; exhausted mandate cycle budget"
        ),
        Err(error) => warn!(
            session_id,
            goal_run_id = %lease.goal_run_id,
            error = %error,
            "Failed to settle ambiguous mandate provider spend; lease remains fail-closed"
        ),
    }
}

const MANDATE_MAX_LLM_CALL_TIMEOUT: Duration = Duration::from_secs(840);
/// A foreground turn must regain control soon enough to select another model,
/// render verified partial progress, or let its supervisor reconcile the task.
/// Provider HTTP timeouts may be longer, but they are not the lifecycle budget.
const FOREGROUND_MAX_LLM_CALL_TIMEOUT: Duration = Duration::from_secs(90);
const MANDATE_INPUT_RESERVATION_MARGIN_NUMERATOR: i64 = 5;
const MANDATE_INPUT_RESERVATION_MARGIN_DENOMINATOR: i64 = 4;
const MANDATE_INPUT_RESERVATION_FIXED_TOKENS: i64 = 512;

/// Prompt estimates are calibrated but can undercount. Reserve 125% plus a
/// fixed 512-token cushion before deriving the provider-side output ceiling.
fn mandate_input_token_reservation(estimated_input_tokens: u32) -> i64 {
    i64::from(estimated_input_tokens)
        .saturating_mul(MANDATE_INPUT_RESERVATION_MARGIN_NUMERATOR)
        .saturating_add(MANDATE_INPUT_RESERVATION_MARGIN_DENOMINATOR - 1)
        / MANDATE_INPUT_RESERVATION_MARGIN_DENOMINATOR
        + MANDATE_INPUT_RESERVATION_FIXED_TOKENS
}

fn mandate_output_token_ceiling(remaining: i64, estimated_input_tokens: u32) -> Option<u32> {
    let available =
        remaining.saturating_sub(mandate_input_token_reservation(estimated_input_tokens));
    (available > 0).then(|| u32::try_from(available).unwrap_or(u32::MAX))
}

async fn committed_non_action_mandate_summary(agent: &Agent) -> Option<String> {
    let fence = agent.mandate_execution.as_ref()?;
    let decision = agent
        .state
        .get_mandate_decision_for_run(&fence.goal_run_id)
        .await
        .ok()
        .flatten()?;
    if decision.mandate_id != fence.mandate_id || decision.mandate_version != fence.mandate_version
    {
        return None;
    }
    match decision.outcome {
        crate::traits::MandateDecisionOutcome::Act => None,
        crate::traits::MandateDecisionOutcome::Wait => Some(format!(
            "WAIT recorded: {} Review again at {}.",
            decision.rationale,
            decision
                .reconsider_at
                .as_deref()
                .unwrap_or("the bounded default")
        )),
        crate::traits::MandateDecisionOutcome::Ask => Some(format!(
            "ASK recorded: {}",
            decision.question.as_deref().unwrap_or(&decision.rationale)
        )),
        crate::traits::MandateDecisionOutcome::Stop => {
            Some(format!("STOP recorded: {}", decision.rationale))
        }
    }
}

const CAPACITY_RECOVERY_WAIT_RATIONALE: &str = "This review reached its internal runaway-protection ceiling before another model turn could be admitted. No new action was authorized; the controller will retry automatically at the earliest bounded review time.";

async fn recover_mandate_capacity_as_wait(agent: &Agent) -> Option<String> {
    if let Some(summary) = committed_non_action_mandate_summary(agent).await {
        return Some(summary);
    }
    let fence = agent.mandate_execution.as_ref()?;
    if fence.worker_task_id != fence.root_task_id {
        return None;
    }
    if agent
        .state
        .get_mandate_decision_for_run(&fence.goal_run_id)
        .await
        .ok()
        .flatten()
        .is_some()
    {
        return None;
    }
    let mandate = agent
        .state
        .get_mandate_for_goal(&fence.goal_id)
        .await
        .ok()
        .flatten()?;
    if !mandate.is_active()
        || mandate.id != fence.mandate_id
        || mandate.version != fence.mandate_version
    {
        return None;
    }
    let mut decision = crate::traits::MandateDecisionCycle::new(
        &mandate.id,
        &fence.goal_run_id,
        crate::traits::MandateDecisionOutcome::Wait,
        CAPACITY_RECOVERY_WAIT_RATIONALE,
        mandate.version,
    );
    decision.reconsider_at =
        Some(mandate.bounded_next_review_at(Some(mandate.min_review_secs), chrono::Utc::now()));
    if let Err(error) = agent
        .state
        .record_mandate_decision(&decision, None, Some(fence.root_task_attempt_id.as_str()))
        .await
    {
        warn!(
            mandate_id = %fence.mandate_id,
            goal_run_id = %fence.goal_run_id,
            %error,
            "Could not persist capacity-recovery WAIT"
        );
    }
    committed_non_action_mandate_summary(agent).await
}

#[allow(clippy::too_many_arguments)]
async fn mandate_cycle_budget_stop(
    agent: &Agent,
    emitter: &crate::events::EventEmitter,
    task_id: &str,
    session_id: &str,
    iteration: usize,
    task_start: Instant,
    learning_ctx: &LearningContext,
    task_tokens_used: u64,
    condition: &str,
    tokens_used: i64,
    budget_per_cycle: i64,
    detail: &str,
) -> LlmPhaseOutcome {
    // A committed WAIT/ASK/STOP already completed the review. The durable
    // decision outranks a later model-call admission failure, which can occur
    // during crash recovery or when an older worker attempts an unnecessary
    // narration turn. Close successfully from typed state instead of
    // overwriting the review with a token-budget error.
    if let Some(summary) = recover_mandate_capacity_as_wait(agent).await {
        agent
            .emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::StoppingCondition,
                "Closed mandate review from a committed or recovered non-action decision"
                    .to_string(),
                json!({
                    "condition": "mandate_non_action_decision_recovered",
                    "goal_run_id": agent
                        .mandate_execution
                        .as_ref()
                        .map(|fence| fence.goal_run_id.as_str()),
                    "tokens_used": tokens_used,
                    "budget_per_cycle": budget_per_cycle,
                }),
            )
            .await;
        agent
            .emit_task_end(
                emitter,
                task_id,
                TaskStatus::Completed,
                TaskOutcome::Succeeded,
                task_start,
                iteration,
                learning_ctx.tool_calls.len(),
                None,
                Some(summary.chars().take(200).collect()),
            )
            .await;
        return LlmPhaseOutcome::Return(Ok(summary));
    }

    let error_message = format!(
        "Mandate cycle token budget stopped execution: {detail} (used {tokens_used} / limit {budget_per_cycle})."
    );
    warn!(
        session_id,
        task_id,
        iteration,
        tokens_used,
        budget_per_cycle,
        condition,
        "Mandate cycle token budget stopped execution"
    );
    agent
        .emit_warning_decision_point(
            emitter,
            task_id,
            iteration,
            DecisionType::StoppingCondition,
            "Stopping condition fired: immutable mandate cycle token budget".to_string(),
            json!({
                "condition": condition,
                "goal_run_id": agent
                    .mandate_execution
                    .as_ref()
                    .map(|fence| fence.goal_run_id.as_str()),
                "tokens_used": tokens_used,
                "budget_per_cycle": budget_per_cycle,
                "detail": detail,
            }),
        )
        .await;
    record_failed_task_tokens(task_tokens_used);
    agent
        .emit_task_end(
            emitter,
            task_id,
            TaskStatus::Failed,
            TaskOutcome::Failed,
            task_start,
            iteration,
            learning_ctx.tool_calls.len(),
            Some(error_message.clone()),
            None,
        )
        .await;
    LlmPhaseOutcome::Return(Err(anyhow::anyhow!(error_message)))
}

/// Observation-only predicate: the model hit the output token limit
/// (`finish_reason=length`) while answering inline (no tool call). Used to
/// emit `inline_dump` telemetry — does NOT change behavior.
fn should_observe_inline_dump(is_truncated: bool, has_tool_calls: bool) -> bool {
    is_truncated && !has_tool_calls
}

/// Whether to disable chain-of-thought for this turn. Computer-use is a
/// mechanical GUI loop (click / type / observe) where the model burns ~400-900
/// reasoning tokens per action with no measured accuracy benefit — and the ~45%
/// of turns that are pure observations (screenshot / get_app_state) need no
/// reasoning at all. Decode is the latency bottleneck, so suppressing thinking
/// here roughly cuts per-call decode ~85% (≈30s → ≈5s). Keyed on the *last*
/// tool: computer-use tasks are long runs of back-to-back computer_use calls,
/// so "last tool was computer_use" reliably means "still in the GUI loop".
fn should_suppress_thinking(last_tool: &str) -> bool {
    last_tool == "computer_use"
}

/// Check if the continuation text has significant word overlap with the prefix,
/// indicating the LLM re-started from scratch instead of continuing.
fn has_significant_overlap(prefix: &str, continuation: &str) -> bool {
    let normalize = |s: &str| -> Vec<String> {
        s.split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_lowercase()
            })
            .filter(|w| w.len() > 2)
            .collect()
    };

    let prefix_words = normalize(prefix);
    let cont_words = normalize(continuation);

    // A short continuation cannot be a complete rewrite of a long response.
    // Treating phrase overlap in a brief tail as a rewrite can discard the
    // entire saved prefix and expose a response that starts mid-word.
    if prefix_words.is_empty() || cont_words.len() < 20 {
        return false;
    }

    // Build a set of distinctive phrases (3-word windows) from the prefix
    let prefix_trigrams: HashSet<String> = prefix_words.windows(3).map(|w| w.join(" ")).collect();

    // Check how many of the first 30 trigrams from the continuation
    // appear in the prefix
    let cont_trigrams: Vec<String> = cont_words
        .windows(3)
        .take(30)
        .map(|w| w.join(" "))
        .collect();

    if cont_trigrams.is_empty() {
        return false;
    }

    let overlap_count = cont_trigrams
        .iter()
        .filter(|t| prefix_trigrams.contains(*t))
        .count();

    // If ≥25% of the first 30 continuation trigrams already appeared in
    // the prefix, the model re-started instead of continuing.
    let ratio = overlap_count as f64 / cont_trigrams.len() as f64;
    ratio >= 0.25
}

pub(super) enum LlmPhaseOutcome {
    ContinueLoop,
    Return(anyhow::Result<String>),
    Proceed(ProviderResponse),
}

impl LlmPhaseOutcome {
    pub(super) fn into_turn_transition(self) -> TurnTransition<ProviderResponse> {
        match self {
            Self::ContinueLoop => TurnTransition::Restart(TurnRestartReason::LlmPhaseRecovery),
            Self::Return(result) => TurnTransition::Finish(result),
            Self::Proceed(response) => TurnTransition::Advance(response),
        }
    }
}

pub(super) struct LlmPhaseCtx<'a> {
    pub messages: &'a mut Vec<Value>,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub iteration: usize,
    pub force_text_response: bool,
    pub task_start: Instant,
    pub task_tokens_used: &'a mut u64,
    pub learning_ctx: &'a mut LearningContext,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub llm_provider: Arc<dyn ModelProvider>,
    pub llm_router: Option<Router>,
    pub model: &'a str,
    pub user_role: UserRole,
    pub tool_defs: &'a [Value],
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    pub resolved_goal_id: &'a Option<String>,
    pub is_scheduled_goal: bool,
    pub effective_goal_daily_budget: &'a mut Option<i64>,
    pub budget_extensions_count: &'a mut usize,
    pub evidence_gain_count: usize,
    pub stall_count: &'a mut usize,
    pub consecutive_same_tool: &'a (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a HashSet<u64>,
    pub total_successful_tool_calls: usize,
    pub pending_external_action_ack: &'a mut Option<String>,
    pub heartbeat: &'a Option<Arc<AtomicU64>>,
    pub empty_response_retry_pending: &'a mut bool,
    pub empty_response_retry_note: &'a mut Option<String>,
    pub deferred_no_tool_streak: usize,
    pub execution_requirement: &'a ExecutionRequirement,
    pub completion_contract: &'a CompletionContract,
    pub completion_progress: &'a CompletionProgress,
    pub force_text_allowed: bool,
    pub max_budget_extensions: usize,
    pub hard_token_cap: i64,
    /// Accumulated text from a previous truncated text response.  When set,
    /// the current iteration's text content is prepended with this prefix
    /// so the user sees the full answer.
    pub truncated_text_prefix: &'a mut Option<String>,
    /// Accumulates milliseconds lost to LLM provider timeouts for diagnostic
    /// attribution. Owner-visible wall-clock budgets still include this time.
    pub provider_timeout_ms: &'a mut u64,
    /// Remaining time in the authoritative execution envelope. The physical
    /// provider call is capped to this duration so one call cannot overshoot
    /// the task-level deadline by a full per-call timeout.
    pub remaining_execution_wall_clock: Option<Duration>,
    /// Counts consecutive iterations where the response was truncated with all
    /// tokens spent on thinking and no usable output.  Escalating recovery:
    /// 1 → reasoning_effort = "low", 2 → disable reasoning entirely,
    /// 3+ → force text with no tools and no reasoning.
    pub thinking_truncation_count: &'a mut u8,
    /// Estimated input tokens computed by the message-build phase, for
    /// est-vs-actual drift telemetry in the emitted `LlmCall` event.
    pub est_input_tokens: u32,
    /// Wall-clock duration of the message-build phase for this iteration, in ms.
    pub build_ms: u64,
    pub projected_source_message_ids: &'a [String],
    pub projected_source_turn_ids: &'a [String],
}

#[allow(clippy::too_many_arguments)]
async fn finalize_verified_external_action_ack(
    agent: &Agent,
    emitter: &crate::events::EventEmitter,
    task_id: &str,
    session_id: &str,
    iteration: usize,
    task_start: Instant,
    learning_ctx: &mut LearningContext,
    model: &str,
    reply: String,
) -> anyhow::Result<String> {
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
    agent
        .append_assistant_message_with_event(emitter, &assistant_msg, model, None, None)
        .await?;
    agent
        .emit_task_end(
            emitter,
            task_id,
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            task_start,
            iteration,
            learning_ctx.tool_calls.len(),
            None,
            Some(reply.chars().take(200).collect()),
        )
        .await;

    learning_ctx.completed_naturally = true;
    learning_ctx.task_outcome = Some(TaskOutcome::Succeeded);
    if agent.mandate_execution.is_none() {
        let learning_ctx_for_task = learning_ctx.clone();
        let state = agent.state.clone();
        tokio::spawn(async move {
            if let Err(e) = post_task::process_learning(&state, learning_ctx_for_task).await {
                warn!("Learning failed: {}", e);
            }
        });
    }

    Ok(reply)
}

/// Close from a durable typed receipt when the provider fails after tool I/O.
/// A response model is useful for synthesis, but it is not authoritative for
/// whether an already-dispatched operation has a terminal observation.  This
/// boundary prevents provider outages from reopening a completed receipt into
/// repeated 90-second calls or a silent running task.
#[allow(clippy::too_many_arguments)]
async fn finalize_durable_receipt_after_provider_failure(
    agent: &Agent,
    emitter: &crate::events::EventEmitter,
    task_id: &str,
    session_id: &str,
    iteration: usize,
    task_start: Instant,
    learning_ctx: &mut LearningContext,
    model: &str,
    provider_error: &str,
    completion_contract: &CompletionContract,
    completion_progress: &CompletionProgress,
) -> anyhow::Result<Option<String>> {
    if agent
        .event_store
        .task_event_count(session_id, task_id, crate::events::EventType::TaskEnd)
        .await
        .unwrap_or_default()
        > 0
    {
        return Ok(None);
    }
    let aggregate = agent
        .event_store
        .task_run_aggregate(session_id, task_id)
        .await?;
    let result = if let Some(operation_id) = aggregate.primary_causal_operation_id.as_deref() {
        agent
            .event_store
            .task_tool_result_by_call_id(session_id, task_id, operation_id)
            .await?
    } else {
        agent
            .event_store
            .latest_task_tool_result(session_id, task_id)
            .await?
    };
    let Some(result) = result else {
        return Ok(None);
    };
    let receipt_closed = if aggregate.contract_present {
        // A response-presentation obligation is closed by the assistant event
        // emitted below. Do not spend a provider call merely to narrate work
        // that the aggregate has already proved. Work proved by the receipt
        // set alone (an exhausted contract that could not credit it) closes
        // the same way: the receipts are the proof.
        aggregate.work_is_fulfilled()
            || aggregate.terminal_decision()
                == crate::events::RunTerminalDecision::SucceededByEvidence
    } else {
        result.receipt.as_ref().is_some_and(|receipt| {
            !receipt.completion_obligation_ids.is_empty()
                || (result.completed_observation()
                    && receipt.semantics.observes_state()
                    && !receipt.semantics.mutates_state())
                || (result.completed_observation()
                    && completion_contract.expects_mutation
                    && crate::agent::mutation_contract_fulfilled(
                        completion_contract,
                        completion_progress,
                    ))
                || (result.succeeded()
                    && receipt.semantics.mutates_state()
                    && receipt.semantics.mutation_effects.has_specific_effects())
        })
    };
    let reply = aggregate
        .projected_success_response()
        .map(str::to_string)
        .unwrap_or_else(|| {
            crate::agent::completion_checks::build_receipt_closeout_reply(&result, receipt_closed)
        });
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
    agent
        .append_assistant_message_with_event(emitter, &assistant_msg, model, None, None)
        .await?;
    let aggregate_failed =
        aggregate.terminal_decision() == crate::events::RunTerminalDecision::Failed;
    let (status, outcome, error) = if receipt_closed {
        (TaskStatus::Completed, TaskOutcome::Succeeded, None)
    } else if aggregate_failed {
        (
            TaskStatus::Completed,
            TaskOutcome::Failed,
            Some(provider_error.to_string()),
        )
    } else {
        (
            TaskStatus::Completed,
            TaskOutcome::Partial,
            Some(provider_error.to_string()),
        )
    };
    agent
        .emit_task_end(
            emitter,
            task_id,
            status,
            outcome,
            task_start,
            iteration,
            learning_ctx.tool_calls.len(),
            error,
            Some(reply.chars().take(200).collect()),
        )
        .await;
    learning_ctx.completed_naturally = receipt_closed || aggregate_failed;
    learning_ctx.task_outcome = Some(outcome);
    Ok(Some(reply))
}

fn aggregate_should_close_before_provider(aggregate: &crate::events::RunAggregate) -> bool {
    aggregate.projected_success_response().is_some() && !aggregate.operations.is_empty()
}

pub(super) async fn run_llm_phase(
    services: &super::services::AgentServices<'_>,
    ctx: &mut LlmPhaseCtx<'_>,
) -> anyhow::Result<LlmPhaseOutcome> {
    let messages = &mut *ctx.messages;
    let emitter = ctx.emitter;
    let task_id = ctx.task_id;
    let session_id = ctx.session_id;
    let user_text = ctx.user_text;
    let iteration = ctx.iteration;
    let force_text_response = ctx.force_text_response && ctx.force_text_allowed;
    let task_start = ctx.task_start;
    let task_tokens_used = &mut *ctx.task_tokens_used;
    let learning_ctx = &mut *ctx.learning_ctx;
    let pending_system_messages = &mut *ctx.pending_system_messages;
    let llm_provider = ctx.llm_provider.clone();
    let llm_router = ctx.llm_router.clone();
    let model = ctx.model;
    let user_role = ctx.user_role;
    let tool_defs = ctx.tool_defs;
    let status_tx = ctx.status_tx;
    let resolved_goal_id = ctx.resolved_goal_id;
    let is_scheduled_goal = ctx.is_scheduled_goal;
    let effective_goal_daily_budget = &mut *ctx.effective_goal_daily_budget;
    let budget_extensions_count = &mut *ctx.budget_extensions_count;
    let evidence_gain_count = ctx.evidence_gain_count;
    let stall_count = &mut *ctx.stall_count;
    let consecutive_same_tool = ctx.consecutive_same_tool;
    let consecutive_same_tool_arg_hashes = ctx.consecutive_same_tool_arg_hashes;
    let total_successful_tool_calls = ctx.total_successful_tool_calls;
    let pending_external_action_ack = &mut *ctx.pending_external_action_ack;
    let heartbeat = ctx.heartbeat;
    let empty_response_retry_pending = &mut *ctx.empty_response_retry_pending;
    let empty_response_retry_note = &mut *ctx.empty_response_retry_note;
    let deferred_no_tool_streak = ctx.deferred_no_tool_streak;
    let execution_requirement = ctx.execution_requirement;
    let completion_contract = ctx.completion_contract;
    let completion_progress = ctx.completion_progress;
    let force_text_allowed = ctx.force_text_allowed;
    let max_budget_extensions = ctx.max_budget_extensions;
    let hard_token_cap = ctx.hard_token_cap;
    let truncated_text_prefix = &mut *ctx.truncated_text_prefix;
    let provider_timeout_ms = &mut *ctx.provider_timeout_ms;
    let thinking_truncation_count = &mut *ctx.thinking_truncation_count;
    let est_input_tokens = ctx.est_input_tokens;
    let build_ms = ctx.build_ms;
    let timeout_after_external_action = Duration::from_secs(90);

    // An exhausted typed obligation is already a terminal lifecycle fact.
    // Do not spend another provider timeout asking an LLM to rediscover it or
    // to authorize closure. The durable causal receipt supplies the closeout;
    // successful runs still reach normal synthesis for a useful user answer.
    let aggregate_before_provider = services
        .agent
        .event_store
        .task_run_aggregate(session_id, task_id)
        .await
        .ok();
    if aggregate_before_provider
        .as_ref()
        .is_some_and(aggregate_should_close_before_provider)
    {
        if let Some(reply) = finalize_durable_receipt_after_provider_failure(
            services.agent,
            emitter,
            task_id,
            session_id,
            iteration,
            task_start,
            learning_ctx,
            model,
            "typed response artifact ready",
            completion_contract,
            completion_progress,
        )
        .await?
        {
            return Ok(LlmPhaseOutcome::Return(Ok(reply)));
        }
    }
    if aggregate_before_provider.as_ref().is_some_and(|aggregate| {
        aggregate.terminal_decision() == crate::events::RunTerminalDecision::Failed
    }) {
        if let Some(reply) = finalize_durable_receipt_after_provider_failure(
            services.agent,
            emitter,
            task_id,
            session_id,
            iteration,
            task_start,
            learning_ctx,
            model,
            "typed obligation exhausted",
            completion_contract,
            completion_progress,
        )
        .await?
        {
            return Ok(LlmPhaseOutcome::Return(Ok(reply)));
        }
    }

    // Force-text: after too many tool calls, force a plain-text response.
    // The tool DEFINITIONS stay in the payload (they are rendered into the
    // prompt prefix; removing them breaks server-side prefix-cache reuse) —
    // calling is disabled via tool_choice=none below, and any tool calls the
    // model still emits are dropped by the hard force-text guard further down.
    let effective_tools: &[Value] = effective_tools_for_call(force_text_response, tool_defs);
    // Prompt-composition telemetry: where the fixed prefix goes (tool defs vs
    // system prompt+memory vs history). Lets us evaluate prompt-size levers
    // (tool subsetting, leaner system prompt) with real numbers per call.
    {
        let comp =
            crate::memory::context_window::prompt_composition(messages.as_slice(), effective_tools);
        info!(
            session_id,
            iteration,
            system_tokens = comp.system_tokens,
            tools_tokens = comp.tools_tokens,
            history_tokens = comp.history_tokens,
            "prompt composition (est)"
        );
    }
    if force_text_response {
        info!(
            session_id,
            iteration,
            total_successful_tool_calls,
            "Force-text mode: requiring plain text via tool_choice=none (tool defs retained for prefix stability)"
        );
    }
    // llama.cpp slot pinning (opt-in). The root interactive agent carries
    // `Some(interactive_slot)`; sub-agents carry `None` and fall through to the
    // provider's background slot. When routing is disabled this is `None` and
    // the provider omits `id_slot` entirely.
    let mut llm_options = ChatOptions {
        id_slot: services.agent.interactive_slot,
        ..ChatOptions::default()
    };
    // Escalating recovery for thinking-model truncation.
    // Count tracks how many consecutive iterations were truncated with all
    // tokens spent on thinking and no usable output.
    if *thinking_truncation_count > 0 {
        match *thinking_truncation_count {
            1 => {
                // First retry: reduce reasoning effort to "low"
                llm_options.reasoning_effort_override = Some("low".to_string());
                info!(
                    session_id,
                    iteration,
                    count = *thinking_truncation_count,
                    "Thinking truncation retry: reducing reasoning_effort to low"
                );
            }
            2 => {
                // Second retry: disable reasoning entirely
                llm_options.reasoning_effort_override = Some("off".to_string());
                info!(
                    session_id,
                    iteration,
                    count = *thinking_truncation_count,
                    "Thinking truncation retry: disabling reasoning entirely"
                );
            }
            _ => {
                // Third+ retry: disable reasoning. Text-only mode is only
                // permitted once a Change/Deliver mutation is fulfilled.
                llm_options.reasoning_effort_override = Some("off".to_string());
                if force_text_allowed {
                    llm_options.tool_choice = ToolChoiceMode::None;
                }
                warn!(
                    session_id,
                    iteration,
                    count = *thinking_truncation_count,
                    force_text_allowed,
                    "Thinking truncation retry: disabling reasoning"
                );
            }
        }
        // Don't reset the count here — it gets reset when a successful
        // response is received (below).
    }
    // Disable thinking while in a computer-use flow (decode-dominated, no
    // accuracy benefit). Placed after truncation recovery so it wins: a CU turn
    // never needs reasoning, even mid-recovery. "off" makes the provider omit
    // enable_thinking, which disables Gemma's chat-template thinking entirely.
    if should_suppress_thinking(&consecutive_same_tool.0) {
        llm_options.reasoning_effort_override = Some("off".to_string());
        info!(
            session_id,
            iteration, "Computer-use flow: disabling reasoning for this turn (decode bottleneck)"
        );
    }
    if force_text_response {
        llm_options.tool_choice = ToolChoiceMode::None;
    } else if execution_requirement.requires_execution()
        && deferred_no_tool_streak > 0
        && deferred_no_tool_streak < DEFERRED_NO_TOOL_ACCEPT_THRESHOLD
        && total_successful_tool_calls == 0
        && !effective_tools.is_empty()
    {
        // Deterministic escalation: once any model has returned text for a
        // still-open execution obligation without evidence, require one tool
        // call on the bounded recovery pass. The model retains full control of
        // which action or read-only capability probe is appropriate.
        // BUT: after DEFERRED_NO_TOOL_ACCEPT_THRESHOLD retries, stop forcing —
        // the contract may be imperfect and forcing tool_choice=required beyond
        // the bounded evidence pass would cause stalls.
        // AND: skip models that previously ignored a forced `required` — the
        // forcing only burns tokens there, and the substantive-text acceptance
        // path in the completion phase converges without it.
        if services.agent.required_tool_choice_ignored(model).await {
            info!(
                session_id,
                iteration,
                deferred_no_tool_streak,
                model,
                "Deferred/no-tool recovery: skipping tool_choice=required — model previously ignored it"
            );
        } else {
            llm_options.tool_choice = ToolChoiceMode::Required;
            POLICY_METRICS
                .deferred_no_tool_forced_required_total
                .fetch_add(1, Ordering::Relaxed);
            info!(
                session_id,
                iteration,
                deferred_no_tool_streak,
                "Deferred/no-tool recovery: forcing tool_choice=required"
            );
        }
    }

    // Always enforce a timeout — never allow unbounded LLM calls.
    // The configured timeout is used if set; otherwise a generous default
    // prevents hung provider calls from blocking the agent loop forever.
    const DEFAULT_LLM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(360);
    let effective_llm_timeout = if pending_external_action_ack.is_some() {
        services
            .agent
            .limits
            .llm_call_timeout
            .map(|timeout| timeout.min(timeout_after_external_action))
            .unwrap_or(timeout_after_external_action)
    } else {
        services
            .agent
            .limits
            .llm_call_timeout
            .unwrap_or(DEFAULT_LLM_TIMEOUT)
    };
    let timeout_dur = if services.agent.mandate_execution.is_some() {
        effective_llm_timeout.min(MANDATE_MAX_LLM_CALL_TIMEOUT)
    } else {
        effective_llm_timeout.min(FOREGROUND_MAX_LLM_CALL_TIMEOUT)
    }
    .min(
        ctx.remaining_execution_wall_clock
            .unwrap_or(effective_llm_timeout)
            .max(Duration::from_millis(1)),
    );

    // Mandate runs use one durable immutable balance across the task lead and
    // every executor. The lease also prevents parallel model calls from each
    // spending the same remaining balance. Reserve enough room for the
    // conservatively estimated input, then cap decoded output to what remains.
    let mandate_budget_lease = match acquire_mandate_run_token_lease(
        services.agent,
        i64::try_from(timeout_dur.as_secs())
            .unwrap_or(900)
            .saturating_add(30),
        heartbeat,
    )
    .await?
    {
        MandateRunTokenAdmission::NotApplicable => None,
        MandateRunTokenAdmission::Exhausted {
            tokens_used,
            budget_per_cycle,
        } => {
            return Ok(mandate_cycle_budget_stop(
                services.agent,
                emitter,
                task_id,
                session_id,
                iteration,
                task_start,
                learning_ctx,
                *task_tokens_used,
                "mandate_cycle_token_budget",
                tokens_used,
                budget_per_cycle,
                "the aggregate lead/executor balance is exhausted",
            )
            .await);
        }
        MandateRunTokenAdmission::Acquired(lease) => {
            let remaining = lease
                .budget_per_cycle
                .saturating_sub(lease.tokens_used_before);
            let Some(output_ceiling) = mandate_output_token_ceiling(remaining, est_input_tokens)
            else {
                release_mandate_run_token_lease_without_dispatch(
                    services.agent,
                    Some(&lease),
                    session_id,
                )
                .await;
                return Ok(mandate_cycle_budget_stop(
                    services.agent,
                    emitter,
                    task_id,
                    session_id,
                    iteration,
                    task_start,
                    learning_ctx,
                    *task_tokens_used,
                    "mandate_cycle_token_admission",
                    lease.tokens_used_before,
                    lease.budget_per_cycle,
                    "the remaining balance cannot admit the next prompt conservatively",
                )
                .await);
            };
            llm_options.max_tokens_override = Some(
                llm_options
                    .max_tokens_override
                    .map_or(output_ceiling, |configured| configured.min(output_ceiling)),
            );
            llm_options.single_attempt_fail_closed = true;
            Some(lease)
        }
    };

    // Phase 0 observability — prefix fingerprint of the final provider payload.
    // Emitted once per LLM phase, here (after security-message injection and
    // force-text tool selection), so it reflects exactly the bytes sent on the
    // normal successful primary attempt. Region sub-hashes let attribution
    // pinpoint which part of the prompt churned and broke llama.cpp prefix
    // reuse. Hashes never carry raw content. See `prefix_fingerprint.rs`.
    let prefix_fp = super::prefix_fingerprint::provider_call_fingerprint(
        messages,
        user_text,
        effective_tools,
        force_text_response,
    );
    {
        info!(
            session_id,
            iteration,
            prefix_hash_system = %prefix_fp.hash_system,
            prefix_hash_pre_boundary = %prefix_fp.hash_pre_boundary,
            prefix_hash_archived = %prefix_fp.prefix_hash_archived,
            tail_hash = %prefix_fp.tail_hash,
            boundary_pos = prefix_fp.boundary_pos,
            message_count = prefix_fp.message_count,
            tool_defs_hash = %prefix_fp.tool_defs_hash,
            session_summary_hash = %prefix_fp.session_summary_hash,
            force_text = prefix_fp.force_text,
            "Provider-call prefix fingerprint"
        );
    }

    // Opt-in debug dump of the exact provider payload (AIDAEMON_DUMP_LLM_REQUESTS).
    // Placed here, alongside the prefix fingerprint, so the dumped bytes are the
    // same finalized payload the fingerprint hashed. See `request_dump.rs`.
    if let Some(dump_dir) = super::request_dump::dump_dir_from_env(
        std::env::var("AIDAEMON_DUMP_LLM_REQUESTS").ok().as_deref(),
    ) {
        match super::request_dump::write_request_dump(
            &dump_dir,
            session_id,
            iteration,
            model,
            messages,
            effective_tools,
            force_text_response,
        ) {
            Ok(path) => info!(
                session_id,
                iteration,
                path = %path.display(),
                "Dumped LLM request payload"
            ),
            Err(e) => warn!(
                session_id,
                iteration,
                error = %e,
                "Failed to dump LLM request payload"
            ),
        }
    }

    let mut llm_telemetry = LlmCallTelemetry::default();
    let llm_call_start = Instant::now();
    #[cfg(feature = "computer_use")]
    let pin_model = crate::agent::computer_use::pinned_model_for_task(task_id).await;
    #[cfg(not(feature = "computer_use"))]
    let pin_model: Option<String> = None;
    #[cfg(feature = "computer_use")]
    let effective_model = pin_model.as_deref().unwrap_or(model);
    #[cfg(not(feature = "computer_use"))]
    let effective_model = model;
    let offered_tools: Vec<String> = effective_tools
        .iter()
        .filter_map(|d| {
            d.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .map(str::to_string)
        })
        .collect();
    let base_llm_call = LlmCallData {
        call_id: None,
        call_purpose: None,
        task_id: task_id.to_string(),
        iteration: Some(iteration as u32),
        model: model.to_string(),
        final_model: None,
        fell_back: false,
        attempts: 0,
        latency_ms: 0,
        prompt_ms: None,
        decode_ms: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: None,
        cache_creation_input_tokens: None,
        fresh_input_tokens: None,
        est_input_tokens: Some(est_input_tokens),
        tool_calls_count: 0,
        offered_tools,
        chosen_tools: Vec::new(),
        build_ms: Some(build_ms),
        prefix_hash_system: Some(prefix_fp.hash_system.clone()),
        prefix_hash_pre_boundary: Some(prefix_fp.hash_pre_boundary.clone()),
        tool_defs_hash: Some(prefix_fp.tool_defs_hash.clone()),
        session_summary_hash: Some(prefix_fp.session_summary_hash.clone()),
        tail_hash: Some(prefix_fp.tail_hash.clone()),
        prefix_hash_archived: Some(prefix_fp.prefix_hash_archived.clone()),
        boundary_pos: Some(prefix_fp.boundary_pos),
        message_count: Some(prefix_fp.message_count),
        projected_source_message_ids: ctx.projected_source_message_ids.to_vec(),
        projected_source_turn_ids: ctx.projected_source_turn_ids.to_vec(),
        force_text: prefix_fp.force_text,
        token_usage_present: false,
        token_usage_evidence: crate::events::TokenUsageEvidence::Unavailable,
        failed: false,
        error: None,
        provider_error_kind: None,
        provider_status: None,
    };
    // Keep the heartbeat alive for the duration of the LLM call so the
    // channel stale-watchdog does not cancel a slow-but-progressing
    // generation.  Auto-aborted on drop; bounded by the LLM timeout.
    let _llm_heartbeat_keeper = spawn_heartbeat_keeper(heartbeat);
    if let Some(lease) = mandate_budget_lease.as_ref() {
        let fence = services
            .agent
            .mandate_execution
            .as_ref()
            .expect("mandate token lease requires execution fence");
        let dispatched = services
            .agent
            .state
            .mark_mandate_run_token_lease_dispatched(
                &lease.goal_run_id,
                &fence.mandate_id,
                fence.mandate_version,
                &lease.lease_token,
            )
            .await?;
        anyhow::ensure!(
            dispatched,
            "mandate model call was not durably reserved before provider dispatch"
        );
    }
    let mut resp = match tokio::time::timeout(
        timeout_dur,
        services.agent.call_llm_with_recovery(
            llm_provider,
            llm_router,
            effective_model,
            messages,
            effective_tools,
            &llm_options,
            &mut llm_telemetry,
            pin_model.as_deref(),
        ),
    )
    .await
    {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            drop(_llm_heartbeat_keeper);
            touch_heartbeat(heartbeat);
            exhaust_mandate_run_token_lease_after_ambiguous_call(
                services.agent,
                mandate_budget_lease.as_ref(),
                session_id,
            )
            .await;
            let error_message = error.to_string();
            let provider_error = error.downcast_ref::<crate::providers::ProviderError>();
            let mut failed_call = base_llm_call.clone();
            failed_call.final_model = Some(if llm_telemetry.final_model.is_empty() {
                effective_model.to_string()
            } else {
                llm_telemetry.final_model.clone()
            });
            failed_call.fell_back = llm_telemetry.fell_back;
            failed_call.attempts = llm_telemetry.attempts.max(1);
            failed_call.latency_ms = llm_call_start.elapsed().as_millis() as u64;
            failed_call.failed = true;
            failed_call.error = Some(error_message.clone());
            failed_call.provider_error_kind = provider_error.map(|error| error.kind);
            failed_call.provider_status = provider_error.and_then(|error| error.status);
            crate::events::record_model_call_telemetry(
                emitter,
                services.agent.state.as_ref(),
                crate::events::ModelCallTelemetryInput {
                    session_id: session_id.to_string(),
                    task_id: task_id.to_string(),
                    call_purpose: None,
                    iteration: Some(iteration as u32),
                    llm_call: failed_call,
                    token_usage: None,
                },
            )
            .await;
            let _ = emitter
                .emit(
                    EventType::Error,
                    ErrorData::llm_error(error_message.clone(), Some(task_id.to_string()))
                        .with_context("llm_call_failed"),
                )
                .await;
            learning_ctx.errors.push((
                format!("LLM call failed after verified work: {error_message}"),
                false,
            ));
            if let Some(reply) = pending_external_action_ack.take() {
                if let Some(last_error) = learning_ctx.errors.last_mut() {
                    last_error.1 = true;
                }
                info!(
                    session_id,
                    iteration,
                    error = %error_message,
                    "Returning deterministic completion after post-action provider failure"
                );
                let reply =
                    crate::agent::tool_execution_phase::user_facing_external_action_ack(&reply);
                let result = finalize_verified_external_action_ack(
                    services.agent,
                    emitter,
                    task_id,
                    session_id,
                    iteration,
                    task_start,
                    learning_ctx,
                    model,
                    reply,
                )
                .await;
                return Ok(LlmPhaseOutcome::Return(result));
            }
            if let Some(reply) = finalize_durable_receipt_after_provider_failure(
                services.agent,
                emitter,
                task_id,
                session_id,
                iteration,
                task_start,
                learning_ctx,
                model,
                &error_message,
                completion_contract,
                completion_progress,
            )
            .await?
            {
                if let Some(last_error) = learning_ctx.errors.last_mut() {
                    last_error.1 = true;
                }
                return Ok(LlmPhaseOutcome::Return(Ok(reply)));
            }
            return Err(error);
        }
        Err(_elapsed) => {
            drop(_llm_heartbeat_keeper);
            touch_heartbeat(heartbeat);
            exhaust_mandate_run_token_lease_after_ambiguous_call(
                services.agent,
                mandate_budget_lease.as_ref(),
                session_id,
            )
            .await;
            // Retain provider-attributed delay for telemetry. It remains part
            // of the end-to-end wall clock seen by the owner.
            *provider_timeout_ms += timeout_dur.as_millis() as u64;
            warn!(
                session_id,
                iteration,
                timeout_secs = timeout_dur.as_secs(),
                "LLM call timed out"
            );
            let timeout_message = format!("LLM call timed out after {}s", timeout_dur.as_secs());
            let mut failed_call = base_llm_call.clone();
            failed_call.final_model = Some(if llm_telemetry.final_model.is_empty() {
                effective_model.to_string()
            } else {
                llm_telemetry.final_model.clone()
            });
            failed_call.fell_back = llm_telemetry.fell_back;
            failed_call.attempts = llm_telemetry.attempts.max(1);
            failed_call.latency_ms = llm_call_start.elapsed().as_millis() as u64;
            failed_call.failed = true;
            failed_call.error = Some(timeout_message.clone());
            failed_call.provider_error_kind = Some(crate::providers::ProviderErrorKind::Timeout);
            crate::events::record_model_call_telemetry(
                emitter,
                services.agent.state.as_ref(),
                crate::events::ModelCallTelemetryInput {
                    session_id: session_id.to_string(),
                    task_id: task_id.to_string(),
                    call_purpose: None,
                    iteration: Some(iteration as u32),
                    llm_call: failed_call,
                    token_usage: None,
                },
            )
            .await;
            let _ = emitter
                .emit(
                    EventType::Error,
                    ErrorData::llm_error(timeout_message, Some(task_id.to_string()))
                        .with_context("llm_call_timeout"),
                )
                .await;
            learning_ctx.errors.push((
                format!("LLM call timed out after {}s", timeout_dur.as_secs()),
                false,
            ));
            if let Some(reply) = pending_external_action_ack.take() {
                if let Some(last_error) = learning_ctx.errors.last_mut() {
                    last_error.1 = true;
                }
                info!(
                    session_id,
                    iteration,
                    timeout_secs = timeout_dur.as_secs(),
                    "Returning deterministic completion after post-action LLM timeout"
                );
                // The stashed ack is model-facing (carries a "Latest result:"
                // excerpt). Shipping it verbatim dumped raw JSON at the user;
                // derive the user-facing form (drops structured-data
                // excerpts, keeps short prose results).
                let reply =
                    crate::agent::tool_execution_phase::user_facing_external_action_ack(&reply);
                let result = finalize_verified_external_action_ack(
                    services.agent,
                    emitter,
                    task_id,
                    session_id,
                    iteration,
                    task_start,
                    learning_ctx,
                    model,
                    reply,
                )
                .await;
                return Ok(LlmPhaseOutcome::Return(result));
            }
            if let Some(reply) = finalize_durable_receipt_after_provider_failure(
                services.agent,
                emitter,
                task_id,
                session_id,
                iteration,
                task_start,
                learning_ctx,
                model,
                &format!("LLM call timed out after {}s", timeout_dur.as_secs()),
                completion_contract,
                completion_progress,
            )
            .await?
            {
                if let Some(last_error) = learning_ctx.errors.last_mut() {
                    last_error.1 = true;
                }
                return Ok(LlmPhaseOutcome::Return(Ok(reply)));
            }
            *stall_count += 1;
            return Ok(LlmPhaseOutcome::ContinueLoop);
        }
    };
    drop(_llm_heartbeat_keeper);
    touch_heartbeat(heartbeat);

    // Settle the serialized mandate call before doing any fallible telemetry
    // work or exposing tool calls. Missing provider usage is unknowable spend,
    // so consume the remaining balance and fail closed for subsequent calls.
    let mandate_run_budget_after_call = if let Some(lease) = mandate_budget_lease.as_ref() {
        let usage_reported = resp.usage.is_some();
        let delta_tokens = resp
            .usage
            .as_ref()
            .map(|usage| i64::try_from(usage.budget_tokens()).unwrap_or(i64::MAX))
            .unwrap_or_else(|| {
                lease
                    .budget_per_cycle
                    .saturating_sub(lease.tokens_used_before)
            });
        let (tokens_used, budget_per_cycle) = services
            .agent
            .state
            .settle_mandate_run_token_lease(&lease.goal_run_id, &lease.lease_token, delta_tokens)
            .await?;
        info!(
            session_id,
            iteration,
            goal_run_id = %lease.goal_run_id,
            delta_tokens,
            tokens_used,
            budget_per_cycle,
            usage_reported,
            "Settled aggregate mandate cycle token usage"
        );
        Some((tokens_used, budget_per_cycle, usage_reported))
    } else {
        None
    };

    // Per-call observability: latency, actual-vs-estimated tokens, and fallback
    // metadata. Persisted as an `LlmCall` event so the request can be fully
    // reconstructed (with timing) via db_probe / the dashboard.
    let llm_latency_ms = llm_call_start.elapsed().as_millis() as u64;
    {
        let (in_tok, out_tok, cached_input_tokens, cache_creation_input_tokens, fresh_input_tokens) =
            resp.usage
                .as_ref()
                .map(|u| {
                    (
                        u.input_tokens,
                        u.output_tokens,
                        u.cached_input_tokens,
                        u.cache_creation_input_tokens,
                        u.fresh_input_tokens(),
                    )
                })
                .unwrap_or((0, 0, None, None, None));
        // Server-side prefill/decode split (llama.cpp `timings`). The remainder
        // `latency_ms - prompt_ms - decode_ms` is queue/transport overhead — the
        // contention signal when a warm call is unexpectedly slow.
        let (prompt_ms, decode_ms) = resp
            .usage
            .as_ref()
            .map(|u| (u.prompt_ms, u.decode_ms))
            .unwrap_or((None, None));
        let final_model = if llm_telemetry.final_model.is_empty() {
            model.to_string()
        } else {
            llm_telemetry.final_model.clone()
        };
        crate::memory::context_window::record_token_estimate_calibration(
            &final_model,
            est_input_tokens as usize,
            in_tok as usize,
        );
        info!(
            session_id,
            iteration,
            latency_ms = llm_latency_ms,
            prefill_ms = prompt_ms,
            decode_ms,
            build_ms,
            model,
            final_model = %final_model,
            fell_back = llm_telemetry.fell_back,
            attempts = llm_telemetry.attempts,
            "LLM call completed"
        );
        crate::events::record_model_call_telemetry(
            emitter,
            services.agent.state.as_ref(),
            crate::events::ModelCallTelemetryInput {
                session_id: session_id.to_string(),
                task_id: task_id.to_string(),
                call_purpose: None,
                iteration: Some(iteration as u32),
                llm_call: LlmCallData {
                    final_model: Some(final_model),
                    fell_back: llm_telemetry.fell_back,
                    attempts: llm_telemetry.attempts,
                    latency_ms: llm_latency_ms,
                    prompt_ms,
                    decode_ms,
                    input_tokens: in_tok,
                    output_tokens: out_tok,
                    cached_input_tokens,
                    cache_creation_input_tokens,
                    fresh_input_tokens,
                    tool_calls_count: resp.tool_calls.len() as u32,
                    // The exact tools offered to the model on this call (post
                    // policy/force-text/budget) and what it chose — so a single
                    // db_probe query can show whether a tool was available when
                    // the model fell back to another one.
                    chosen_tools: resp.tool_calls.iter().map(|tc| tc.name.clone()).collect(),
                    token_usage_present: resp.usage.is_some(),
                    token_usage_evidence: crate::events::TokenUsageEvidence::Unavailable,
                    ..base_llm_call.clone()
                },
                token_usage: resp.usage.clone(),
            },
        )
        .await;
    }

    let llm_text_closeout_candidate = resp.tool_calls.is_empty()
        && resp
            .content
            .as_ref()
            .is_some_and(|content| !content.trim().is_empty());
    let has_unrecovered_errors = learning_ctx.errors.iter().any(|(_, recovered)| !*recovered);
    let llm_budget_closeout_candidate = llm_text_closeout_candidate
        && !has_unrecovered_errors
        && !force_text_response
        && (iteration == 1 || total_successful_tool_calls > 0);

    // Record token usage (both for task budget and daily budget)
    if let Some(ref usage) = resp.usage {
        // Charge budgets on real incremental work (fresh input + output), not
        // the re-read cached prefix — see TokenUsage::budget_tokens.
        *task_tokens_used += usage.budget_tokens();
        info!(
            session_id,
            iteration,
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            cached_input_tokens = usage.cached_input_tokens.unwrap_or(0),
            billed_tokens = usage.budget_tokens(),
            task_tokens_used = *task_tokens_used,
            "LLM token usage"
        );
        // Goal budget accounting: increment tokens_used_today for daily
        // admission control. Scheduled runs use a separate per-run budget
        // once they have started.
        if let Some(goal_id) = resolved_goal_id.as_ref() {
            let delta_tokens = usage.budget_tokens() as i64;
            match services
                .agent
                .state
                .add_goal_tokens_and_get_budget_status(goal_id, delta_tokens)
                .await
            {
                Ok(Some(status)) => {
                    if is_scheduled_goal {
                        let run_budget_status = if let Some(registry) =
                            &services.agent.goal_token_registry
                        {
                            let _ = registry.add_run_tokens(goal_id, delta_tokens).await;
                            registry
                                .update_run_health(
                                    goal_id,
                                    Agent::scheduled_run_health_snapshot(
                                        learning_ctx,
                                        graceful::ScheduledRunActivityMetrics {
                                            evidence_gain_count,
                                            stall_count: *stall_count,
                                            consecutive_same_tool_count: consecutive_same_tool.1,
                                            consecutive_same_tool_unique_args:
                                                consecutive_same_tool_arg_hashes.len(),
                                            total_successful_tool_calls,
                                        },
                                        completion_contract,
                                        completion_progress,
                                    ),
                                )
                                .await
                        } else {
                            None
                        };
                        if let Some(run_budget_status) = run_budget_status {
                            persist_scheduled_run_state(
                                &services.agent.state,
                                goal_id,
                                None,
                                &run_budget_status,
                            )
                            .await;
                            let mut run_budget_ctx = graceful::ScheduledRunBudgetControlCtx {
                                emitter,
                                task_id,
                                session_id,
                                iteration,
                                goal_id,
                                status: &run_budget_status,
                                pending_system_messages,
                                max_budget_extensions,
                                hard_token_cap,
                            };
                            if let graceful::ScheduledRunBudgetControlOutcome::Exhausted {
                                tokens_used,
                                budget_per_check,
                            } = services
                                .agent
                                .enforce_scheduled_run_budget_control(&mut run_budget_ctx)
                                .await
                            {
                                if llm_budget_closeout_candidate {
                                    services.agent.emit_decision_point(
                                            emitter,
                                            task_id,
                                            iteration,
                                            DecisionType::StoppingCondition,
                                            "Allowing scheduled-run final text closeout after budget exhaustion"
                                                .to_string(),
                                            json!({
                                                "condition": "scheduled_run_budget_closeout_grace",
                                                "goal_id": goal_id,
                                                "budget_per_check": budget_per_check,
                                                "tokens_used": tokens_used,
                                                "delta_tokens": delta_tokens,
                                            }),
                                        )
                                        .await;
                                } else {
                                    warn!(
                                        session_id,
                                        iteration,
                                        goal_id = %goal_id,
                                        delta_tokens,
                                        tokens_used,
                                        budget_per_check,
                                        "Scheduled run budget exhausted after LLM call"
                                    );
                                    services.agent.emit_decision_point(
                                        emitter,
                                        task_id,
                                        iteration,
                                        DecisionType::StoppingCondition,
                                        "Stopping condition fired: scheduled run budget exhausted"
                                            .to_string(),
                                        json!({
                                            "condition":"scheduled_run_budget",
                                            "goal_id": goal_id,
                                            "budget_per_check": budget_per_check,
                                            "tokens_used": tokens_used,
                                            "delta_tokens": delta_tokens
                                        }),
                                    )
                                    .await;
                                    let alert_msg = format!(
                                        "Token alert: scheduled run for goal '{}' hit its per-run budget (used {} / limit {}). The run stopped safely before completion.",
                                        goal_id, tokens_used, budget_per_check
                                    );
                                    services
                                        .agent
                                        .fanout_token_alert(
                                            Some(goal_id.as_str()),
                                            session_id,
                                            &alert_msg,
                                            Some(session_id),
                                        )
                                        .await;
                                    let result = services
                                        .agent
                                        .graceful_scheduled_run_budget_response(
                                            emitter,
                                            session_id,
                                            learning_ctx,
                                            tokens_used,
                                            budget_per_check,
                                        )
                                        .await;
                                    let (status, error, summary) = match &result {
                                        Ok(reply) => (
                                            TaskStatus::Completed,
                                            None,
                                            Some(reply.chars().take(200).collect()),
                                        ),
                                        Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                                    };
                                    if status == TaskStatus::Failed {
                                        record_failed_task_tokens(*task_tokens_used);
                                    }
                                    let outcome = TaskOutcome::Failed;
                                    services
                                        .agent
                                        .emit_task_end(
                                            emitter,
                                            task_id,
                                            status,
                                            outcome,
                                            task_start,
                                            iteration,
                                            learning_ctx.tool_calls.len(),
                                            error,
                                            summary,
                                        )
                                        .await;
                                    return Ok(LlmPhaseOutcome::Return(result));
                                }
                            }
                        }
                    } else {
                        let mut goal_budget_ctx = graceful::GoalBudgetControlCtx {
                            emitter,
                            task_id,
                            session_id,
                            iteration,
                            goal_id,
                            status: &status,
                            user_role,
                            learning_ctx,
                            evidence_gain_count,
                            stall_count: *stall_count,
                            consecutive_same_tool_count: consecutive_same_tool.1,
                            consecutive_same_tool_unique_args: consecutive_same_tool_arg_hashes
                                .len(),
                            total_successful_tool_calls,
                            pending_system_messages,
                            status_tx,
                            is_scheduled_goal,
                            effective_goal_daily_budget,
                            budget_extensions_count,
                            max_budget_extensions,
                            hard_token_cap,
                            source: graceful::GoalBudgetCheckSource::PostLlm,
                        };
                        if let graceful::GoalBudgetControlOutcome::Exhausted {
                            tokens_used_today,
                            budget_daily,
                        } = services
                            .agent
                            .enforce_goal_daily_budget_control(&mut goal_budget_ctx)
                            .await
                        {
                            if llm_budget_closeout_candidate {
                                services.agent.emit_decision_point(
                                        emitter,
                                        task_id,
                                        iteration,
                                        DecisionType::StoppingCondition,
                                        "Allowing final text closeout after goal daily budget exhaustion"
                                            .to_string(),
                                        json!({
                                            "condition": "goal_daily_budget_closeout_grace",
                                            "goal_id": goal_id,
                                            "budget_daily": budget_daily,
                                            "tokens_used_today": tokens_used_today,
                                            "delta_tokens": delta_tokens,
                                        }),
                                    )
                                    .await;
                            } else {
                                warn!(
                                    session_id,
                                    iteration,
                                    goal_id = %goal_id,
                                    delta_tokens,
                                    tokens_used_today,
                                    budget_daily,
                                    "Goal daily token budget exhausted after LLM call"
                                );
                                services.agent.emit_decision_point(
                                    emitter,
                                    task_id,
                                    iteration,
                                    DecisionType::StoppingCondition,
                                    "Stopping condition fired: goal daily token budget exhausted"
                                        .to_string(),
                                    json!({
                                        "condition":"goal_daily_token_budget",
                                        "goal_id": goal_id,
                                        "budget_daily": budget_daily,
                                        "tokens_used_today": tokens_used_today,
                                        "delta_tokens": delta_tokens
                                    }),
                                )
                                .await;
                                let alert_msg = format!(
                                    "Token alert: goal '{}' hit daily token budget (used {} / limit {}). Execution was stopped to prevent overspending.",
                                    goal_id, tokens_used_today, budget_daily
                                );
                                services
                                    .agent
                                    .fanout_token_alert(
                                        Some(goal_id.as_str()),
                                        session_id,
                                        &alert_msg,
                                        Some(session_id),
                                    )
                                    .await;
                                let result = services
                                    .agent
                                    .graceful_goal_daily_budget_response(
                                        emitter,
                                        session_id,
                                        learning_ctx,
                                        tokens_used_today,
                                        budget_daily,
                                        is_scheduled_goal,
                                    )
                                    .await;
                                let (status, error, summary) = match &result {
                                    Ok(reply) => (
                                        TaskStatus::Completed,
                                        None,
                                        Some(reply.chars().take(200).collect()),
                                    ),
                                    Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                                };
                                if status == TaskStatus::Failed {
                                    record_failed_task_tokens(*task_tokens_used);
                                }
                                let outcome = TaskOutcome::Failed;
                                services
                                    .agent
                                    .emit_task_end(
                                        emitter,
                                        task_id,
                                        status,
                                        outcome,
                                        task_start,
                                        iteration,
                                        learning_ctx.tool_calls.len(),
                                        error,
                                        summary,
                                    )
                                    .await;
                                return Ok(LlmPhaseOutcome::Return(result));
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(
                        session_id,
                        iteration,
                        goal_id = %goal_id,
                        error = %e,
                        "Failed to update goal token usage"
                    );
                }
            }
        }
    }

    if let Some((tokens_used, budget_per_cycle, usage_reported)) = mandate_run_budget_after_call {
        if tokens_used >= budget_per_cycle {
            if llm_budget_closeout_candidate && usage_reported {
                services
                    .agent
                    .emit_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::StoppingCondition,
                        "Allowing final text closeout after immutable mandate cycle budget exhaustion"
                            .to_string(),
                        json!({
                            "condition": "mandate_cycle_token_budget_closeout_grace",
                            "goal_run_id": services
                                .agent
                                .mandate_execution
                                .as_ref()
                                .map(|fence| fence.goal_run_id.as_str()),
                            "tokens_used": tokens_used,
                            "budget_per_cycle": budget_per_cycle,
                            "provider_usage_reported": usage_reported,
                        }),
                    )
                    .await;
            } else {
                let detail = if usage_reported {
                    "the aggregate balance was exhausted by the completed model call"
                } else {
                    "the provider omitted usage, so the remaining balance was consumed fail-closed"
                };
                return Ok(mandate_cycle_budget_stop(
                    services.agent,
                    emitter,
                    task_id,
                    session_id,
                    iteration,
                    task_start,
                    learning_ctx,
                    *task_tokens_used,
                    "mandate_cycle_token_budget_post_llm",
                    tokens_used,
                    budget_per_cycle,
                    detail,
                )
                .await);
            }
        }
    }

    // Log LLM call activity for executor agents
    if let Some(tid) = services.agent.task_id.as_ref() {
        let tokens = resp
            .usage
            .as_ref()
            .map(|u| (u.input_tokens + u.output_tokens) as i64);
        let activity = TaskActivity {
            id: 0,
            task_id: tid.clone(),
            activity_type: "llm_call".to_string(),
            tool_name: None,
            tool_args: None,
            result: resp.content.as_ref().map(|c| c.chars().take(500).collect()),
            success: Some(true),
            tokens_used: tokens,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        if let Err(e) = services.agent.state.log_task_activity(&activity).await {
            warn!(task_id = %tid, error = %e, "Failed to log LLM activity");
        }
    }

    // Log tool call names for debugging
    let tc_names: Vec<&str> = resp.tool_calls.iter().map(|tc| tc.name.as_str()).collect();
    info!(
        session_id,
        has_content = resp.content.is_some(),
        tool_calls = resp.tool_calls.len(),
        tool_names = ?tc_names,
        "LLM response received"
    );

    // Response-composition telemetry: where the decoded output goes (narration
    // text vs the structured tool call vs thinking). Decode time scales with
    // output tokens, so this shows whether the cost is trimmable verbosity or
    // the essential tool call — the output-side counterpart to prompt composition.
    {
        let tool_calls_serialized: String = resp
            .tool_calls
            .iter()
            .map(|tc| format!("{} {}", tc.name, tc.arguments))
            .collect::<Vec<_>>()
            .join("\n");
        let out_comp = crate::memory::context_window::response_composition(
            resp.content.as_deref(),
            &tool_calls_serialized,
            resp.thinking.as_deref(),
        );
        info!(
            session_id,
            iteration,
            text_tokens = out_comp.text_tokens,
            tool_call_tokens = out_comp.tool_call_tokens,
            thinking_tokens = out_comp.thinking_tokens,
            "response composition (est)"
        );
    }

    // Clear pending empty-response retry context once the model produces
    // any actionable output (text or tool calls).
    let has_non_empty_content = resp.content.as_ref().is_some_and(|s| !s.is_empty());
    if !resp.tool_calls.is_empty() || has_non_empty_content {
        *empty_response_retry_pending = false;
        *empty_response_retry_note = None;
        // Reset thinking-truncation counter on any successful response.
        *thinking_truncation_count = 0;
    }

    // Contract check: a forced `tool_choice=required` call that comes back
    // with text and zero tool calls means the serving stack ignored the
    // constraint (llama.cpp + Gemma does this, and generation can degenerate
    // into a repetition loop until the token limit). Flag the model so the
    // deferred/no-tool recovery never forces `required` on it again.
    if matches!(llm_options.tool_choice, ToolChoiceMode::Required)
        && resp.tool_calls.is_empty()
        && has_non_empty_content
        && services
            .agent
            .record_required_tool_choice_ignored(&llm_telemetry.final_model)
            .await
    {
        warn!(
            session_id,
            iteration,
            model = %llm_telemetry.final_model,
            "Forced tool_choice=required returned no tool calls — model flagged, future recovery will not force it"
        );
    }

    // Token-limit truncation recovery: if the response was cut off at the
    // model's max_tokens and produced no usable output, nudge the model to
    // use tools (write_file) for long content instead of generating inline.
    let is_truncated = resp
        .response_note
        .as_ref()
        .is_some_and(|n| n.contains("truncated"));
    if should_observe_inline_dump(is_truncated, !resp.tool_calls.is_empty()) {
        let reply_chars = resp.content.as_deref().unwrap_or("").chars().count();
        let output_tokens = resp.usage.as_ref().map(|u| u.output_tokens).unwrap_or(0);
        tracing::info!(
            target: "inline_dump",
            session_id,
            iteration,
            model = %llm_telemetry.final_model,
            depth = services.agent.depth,
            output_tokens,
            reply_chars,
            "Model hit the output token limit answering inline (no tool call)"
        );
    }
    if is_truncated && resp.tool_calls.is_empty() && !has_non_empty_content {
        *thinking_truncation_count = thinking_truncation_count.saturating_add(1);
        warn!(
            session_id,
            iteration,
            consecutive_truncations = *thinking_truncation_count,
            "Response truncated at token limit with no usable output — injecting retry nudge"
        );
        pending_system_messages.push(SystemDirective::TruncationRecoveryUseWriteFile);
        *stall_count += 1;
        return Ok(LlmPhaseOutcome::ContinueLoop);
    }

    // Text response truncation continuation: if the response was cut off
    // mid-sentence but has partial text content, save the partial text and
    // ask the model to continue from where it left off.  This prevents
    // sending half-finished sentences to the user.
    //
    // Detection: explicit `is_truncated` from finish_reason=length, OR
    // heuristic: text-only response that ends mid-sentence (no terminal
    // punctuation).  Some free-tier models report finish_reason=stop even
    // when they hit an internal output cap.
    let probable_text_truncation = if has_non_empty_content && resp.tool_calls.is_empty() {
        let partial = resp.content.as_deref().unwrap_or("");
        let trimmed_end = partial.trim_end();
        let ends_mid_sentence = !trimmed_end.is_empty()
            && !trimmed_end.ends_with('.')
            && !trimmed_end.ends_with('!')
            && !trimmed_end.ends_with('?')
            && !trimmed_end.ends_with("```")
            && !trimmed_end.ends_with('"')
            && !trimmed_end.ends_with(')')
            && !trimmed_end.ends_with(':')
            && !trimmed_end.ends_with('}')
            && !trimmed_end.ends_with(']')
            && !trimmed_end.ends_with(';');
        // Require the explicit flag OR the heuristic (ends mid-sentence
        // AND the response is very long — short/medium responses that just
        // omit final punctuation are almost always complete).
        // Previous threshold of 20 words caused false positives on recall
        // responses, haikus, and other short-form text without terminal
        // punctuation.
        ends_mid_sentence && (is_truncated || trimmed_end.split_whitespace().count() > 200)
    } else {
        false
    };

    if probable_text_truncation && truncated_text_prefix.is_none() {
        let partial = resp.content.as_deref().unwrap_or("");
        let tail_chars: String = partial
            .chars()
            .rev()
            .take(80)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        warn!(
            session_id,
            iteration,
            partial_len = partial.len(),
            is_truncated,
            tail = %tail_chars,
            "Text response truncated mid-sentence — requesting continuation"
        );
        *truncated_text_prefix = Some(partial.to_string());
        pending_system_messages.push(SystemDirective::TruncationRecoveryTextContinuation {
            truncated_tail: tail_chars,
        });
        return Ok(LlmPhaseOutcome::ContinueLoop);
    }

    // If there is a saved truncated text prefix from a previous iteration,
    // merge it with the continuation.  If the LLM re-started from scratch
    // instead of continuing, detect the overlap and use only the new
    // (complete) response to avoid sending duplicated text.
    if let Some(prefix) = truncated_text_prefix.take() {
        let continuation = resp.content.as_deref().unwrap_or("").trim_start();
        if continuation.is_empty() {
            resp.content = Some(prefix);
        } else if has_significant_overlap(&prefix, continuation) {
            // LLM generated a new complete response — use it instead of
            // concatenating (which would duplicate content).
            info!(
                session_id,
                iteration,
                "Truncation continuation has significant overlap with prefix — using continuation only"
            );
            // continuation is already in resp.content
        } else {
            resp.content = Some(format!("{}{}", prefix, continuation));
        }
    }

    // Hard force-text mode: if the model still emits tool calls despite
    // tool_choice=none, ignore those calls and require plain text.
    if force_text_response && !resp.tool_calls.is_empty() {
        let dropped = resp.tool_calls.len();
        warn!(
            session_id,
            iteration,
            dropped_tool_calls = dropped,
            "Force-text mode: dropping hallucinated tool calls"
        );
        if has_non_empty_content {
            resp.tool_calls.clear();
        } else {
            pending_system_messages.push(SystemDirective::ToolModeDisabledPlainText);
            *stall_count += 1;
            return Ok(LlmPhaseOutcome::ContinueLoop);
        }
    }

    Ok(LlmPhaseOutcome::Proceed(resp))
}

/// Tool definitions to include in a provider call.
///
/// Force-text mode intentionally returns the SAME definitions instead of an
/// empty slice: tool defs are rendered into the prompt prefix by chat
/// templates, so stripping them changes the prompt bytes and breaks
/// server-side prefix-cache reuse (full ~23k-token re-prefills were measured
/// and attributed to `tool_defs_refit` in the 2026-06-06 Phase 0 run).
/// Calling is disabled via `tool_choice=none`; stray tool calls are dropped
/// by the hard force-text guard after the response arrives.
fn effective_tools_for_call(force_text_response: bool, tool_defs: &[Value]) -> &[Value] {
    // Deliberately ignored — and load-bearing. The cross-turn prefix
    // stability spec's force-text invariant ("tool_defs_hash and the
    // rendered prefix stay stable across a force-text turn") depends on
    // this function returning the full roster in BOTH modes. Do NOT "wire
    // up" this flag to strip definitions in force-text: that silently
    // reintroduces the tool_defs_refit cache break and fails exit
    // criterion 2 of 2026-06-06-cross-turn-prefix-stability-design.md.
    // The flag stays in the signature so every call site names the mode
    // decision, and `force_text_keeps_tool_defs_for_prefix_stability`
    // pins the behavior.
    let _ = force_text_response;
    tool_defs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::store_prelude::*;

    async fn install_due_mandate_root(
        harness: &mut crate::testing::TestHarness,
    ) -> (crate::traits::Mandate, crate::traits::GoalRun) {
        use crate::traits::{Goal, Mandate, MandateAuthority, Task};

        let goal = Goal::new_continuous(
            "Review a synthetic bounded source",
            "owner-session",
            Some(250_000),
            Some(2_000_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Review a synthetic bounded source",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        harness
            .state
            .create_mandate_controller(&goal, &mandate)
            .await
            .expect("persist mandate controller");

        let leased = harness
            .state
            .claim_due_mandates(1, "capacity-recovery-test", 300)
            .await
            .expect("claim due mandate")
            .pop()
            .expect("one due mandate");
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let root_task = Task {
            id: root_task_id.clone(),
            goal_id: goal.id.clone(),
            description: "Review the synthetic bounded source".to_string(),
            status: "pending".to_string(),
            priority: "high".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        let goal_run_id = uuid::Uuid::new_v4().to_string();
        let run = harness
            .state
            .create_mandate_review_run(
                &mandate.id,
                leased
                    .review_lease_token
                    .as_deref()
                    .expect("review lease token"),
                &goal_run_id,
                &root_task,
            )
            .await
            .expect("create mandate review run");
        let root_attempt = harness
            .state
            .claim_task_with_lease(&root_task_id, "capacity-recovery-lead", None, 300)
            .await
            .expect("claim root task")
            .expect("root task attempt");
        harness.agent.set_test_mandate_execution(
            &mandate.id,
            mandate.version,
            mandate.authority.clone(),
            &goal.id,
            &root_task_id,
            &root_attempt.id,
            &root_attempt,
        );
        (mandate, run)
    }

    #[tokio::test]
    async fn exhausted_mandate_review_recovers_with_one_typed_wait_and_bounded_retry() {
        let mut harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .expect("setup test agent");
        let (mandate, run) = install_due_mandate_root(&mut harness).await;
        let before = chrono::Utc::now();

        let summary = recover_mandate_capacity_as_wait(&harness.agent)
            .await
            .expect("capacity exhaustion should recover as WAIT");
        assert!(summary.contains("WAIT recorded"));
        assert!(summary.contains("retry automatically"));

        let decision = harness
            .state
            .get_mandate_decision_for_run(&run.id)
            .await
            .expect("read recovered decision")
            .expect("recovered decision");
        assert_eq!(
            decision.outcome,
            crate::traits::MandateDecisionOutcome::Wait
        );
        assert_eq!(decision.rationale, CAPACITY_RECOVERY_WAIT_RATIONALE);
        let retry_at = chrono::DateTime::parse_from_rfc3339(
            decision
                .reconsider_at
                .as_deref()
                .expect("bounded retry time"),
        )
        .expect("valid retry timestamp")
        .with_timezone(&chrono::Utc);
        assert!(retry_at >= before + chrono::Duration::seconds(mandate.min_review_secs - 1));
        assert!(retry_at <= before + chrono::Duration::seconds(mandate.min_review_secs + 2));

        let second_summary = recover_mandate_capacity_as_wait(&harness.agent)
            .await
            .expect("recovery is idempotent");
        assert_eq!(second_summary, summary);
        let decisions = harness
            .state
            .list_mandate_decisions(&mandate.id, 10)
            .await
            .expect("list decisions");
        assert_eq!(decisions.len(), 1);
    }

    #[test]
    fn mandate_output_ceiling_reserves_margin_and_never_exceeds_remaining_balance() {
        assert_eq!(mandate_input_token_reservation(1_000), 1_762);
        assert_eq!(mandate_output_token_ceiling(2_000, 1_000), Some(238));
        assert_eq!(mandate_output_token_ceiling(1_762, 1_000), None);
        assert_eq!(mandate_output_token_ceiling(1_700, 1_000), None);
    }

    #[test]
    fn mandate_provider_timeout_cannot_outlive_token_lease() {
        const SQLITE_MANDATE_TOKEN_LEASE_MAX_SECS: u64 = 900;
        const LEASE_GRACE_SECS: u64 = 30;
        assert!(
            MANDATE_MAX_LLM_CALL_TIMEOUT.as_secs() + LEASE_GRACE_SECS
                <= SQLITE_MANDATE_TOKEN_LEASE_MAX_SECS
        );
    }

    #[test]
    fn foreground_provider_timeout_is_a_bounded_recovery_interval() {
        assert_eq!(
            Duration::from_secs(300).min(FOREGROUND_MAX_LLM_CALL_TIMEOUT),
            Duration::from_secs(90)
        );
        assert_eq!(
            Duration::from_secs(30).min(FOREGROUND_MAX_LLM_CALL_TIMEOUT),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn typed_response_with_closed_work_skips_optional_narrator() {
        let mut aggregate = crate::events::RunAggregate::new("task-synthetic");
        aggregate.contract_present = true;
        aggregate.response_contract = Some(Box::new(
            crate::traits::RequestResponseContract::ExactText {
                success_text: "phase=synthetic; outcome=complete".to_string(),
                source_message_hash: "synthetic-hash".to_string(),
            },
        ));
        aggregate.obligations.insert(
            "obligation-1".to_string(),
            crate::events::RunObligation {
                id: "obligation-1".to_string(),
                class: crate::events::RunObligationClass::Perform,
                state: crate::events::RunObligationState::Satisfied,
                receipt: None,
                evidence_requirement: None,
                required_effect: crate::traits::ToolMutationEffects::NONE,
                satisfied_at_revision: Some(0),
                satisfying_receipt_ids: vec!["result-1".to_string()],
            },
        );
        aggregate.operations.insert(
            "operation-1".to_string(),
            crate::events::RunOperation {
                operation_id: "operation-1".to_string(),
                tool_name: "terminal".to_string(),
                stable_operation_key: Some("operation-1".to_string()),
                obligation_ids: vec!["obligation-1".to_string()],
                max_attempts: Some(1),
                max_invocations: Some(1),
                idempotency_key: None,
                outcome: Some(crate::traits::ToolOutcomeStatus::Succeeded),
                dispatched: true,
                result_id: Some("result-1".to_string()),
                operation_lineage: None,
            },
        );

        assert!(aggregate_should_close_before_provider(&aggregate));
    }

    #[test]
    fn suppresses_thinking_only_in_computer_use_flow() {
        // Once the last tool was computer_use, the agent is in a mechanical GUI
        // loop — disable thinking. Other tools keep reasoning.
        assert!(should_suppress_thinking("computer_use"));
        assert!(!should_suppress_thinking("terminal"));
        assert!(!should_suppress_thinking("spawn_agent"));
        assert!(!should_suppress_thinking(""));
    }

    #[tokio::test(start_paused = true)]
    async fn heartbeat_keeper_advances_during_long_await() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        let hb = Arc::new(AtomicU64::new(0));
        let keeper = spawn_heartbeat_keeper(&Some(hb.clone()));
        // Simulate a long LLM call.
        tokio::time::sleep(std::time::Duration::from_secs(31)).await;
        assert!(
            hb.load(Ordering::Relaxed) > 0,
            "heartbeat should advance during the await"
        );
        drop(keeper);
    }

    #[test]
    fn force_text_keeps_tool_defs_for_prefix_stability() {
        // Tool definitions are rendered into the llama prompt prefix.
        // Force-text must disable calling via tool_choice=none, NOT by
        // stripping the defs — stripping changes the rendered prompt and
        // breaks server-side prefix-cache reuse (measured 2026-06-06:
        // full ~23k-token re-prefills attributed to tool_defs_refit).
        let tools = vec![serde_json::json!({"name": "t1"})];
        assert_eq!(effective_tools_for_call(true, &tools), tools.as_slice());
        assert_eq!(effective_tools_for_call(false, &tools), tools.as_slice());
    }

    #[test]
    fn overlap_detects_duplicate_response() {
        let prefix = "Based on my memory:\n\nYour dog's name: Luna 🐕\n\
                       What you like to eat: Sushi 🍣\n\n---\n\n\
                       Haiku about your weekend hobby (hiking):\n\n\
                       Boots on rocky trails\nMountains call, the summit waits\n\
                       Weekend peace is found";
        let continuation = "Based on what I have stored in memory:\n\n\
                            Your dog's name: Luna 🐕\nYour favorite food: Sushi 🍣\n\n\
                            And here's a haiku about your weekend hobby (hiking):\n\n\
                            Boots on rocky trails\nMountains call, the summit waits\n\
                            Weekend peace is found";
        assert!(
            has_significant_overlap(prefix, continuation),
            "Should detect duplicate response with overlapping content"
        );
    }

    #[test]
    fn overlap_allows_genuine_continuation() {
        let prefix = "Let me explain the three main architectural patterns used in \
                       modern web development. First, the Model-View-Controller (MVC) \
                       pattern separates concerns into three distinct components that";
        let continuation = "interact through well-defined interfaces. The Model handles \
                            data and business logic, the View renders the user interface, \
                            and the Controller processes user input and coordinates between them.";
        assert!(
            !has_significant_overlap(prefix, continuation),
            "Should allow genuine continuation with no overlap"
        );
    }

    #[test]
    fn overlap_does_not_replace_prefix_with_short_continuation_tail() {
        let prefix = "Which company or role are you targeting? After comparing the resumes, \
                      the AI Expert version is the strongest choice because it emphasizes the \
                      architecture experience that makes the chosen one";
        let continuation = "even stronger. Which company or role are you looking at right now?";

        assert!(
            !has_significant_overlap(prefix, continuation),
            "a short continuation tail must not replace the saved response prefix"
        );
    }

    #[test]
    fn overlap_empty_inputs() {
        assert!(!has_significant_overlap("", "hello world"));
        assert!(!has_significant_overlap("hello world", ""));
        assert!(!has_significant_overlap("", ""));
    }

    #[test]
    fn overlap_short_inputs() {
        assert!(!has_significant_overlap("hi", "hi"));
        assert!(!has_significant_overlap("a b", "a b"));
    }

    #[test]
    fn observes_inline_dump_only_when_truncated_and_no_tool_call() {
        // truncated, no tool call → observe
        assert!(should_observe_inline_dump(true, false));
        // truncated but the model DID call a tool → not an inline dump
        assert!(!should_observe_inline_dump(true, true));
        // not truncated → nothing to observe
        assert!(!should_observe_inline_dump(false, false));
        assert!(!should_observe_inline_dump(false, true));
    }
}
