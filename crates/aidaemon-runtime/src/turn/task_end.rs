//! How a task ends: the run's terminal decision and outcome from its
//! replayed aggregate, the TaskEnd record with its efficiency, proof and
//! harness evaluation, the task-end hooks, and decision points.

use std::time::Instant;

use aidaemon_events::{DecisionPointData, DecisionType, EventType, TaskEndData, TaskStatus};
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::host::TurnHost;

/// One decision point to record.
struct DecisionPointEmission {
    decision_type: DecisionType,
    severity: aidaemon_events::DiagnosticSeverity,
    summary: String,
    metadata: Value,
}

pub async fn run_task_end_tool_hooks(host: &dyn TurnHost, task_id: &str, session_id: &str) {
    for tool in host.tools() {
        if let Err(e) = tool.on_task_end(task_id, session_id).await {
            warn!(
                task_id,
                session_id,
                tool = tool.name(),
                error = %e,
                "Task-end cleanup hook failed"
            );
        }
    }
    if let Some(manager) = aidaemon_tools::checkpoints::active_manager()
        && let Err(error) = manager.finalize_task(task_id, session_id).await
    {
        warn!(
            task_id,
            session_id,
            error = %error,
            "Failed to finalize filesystem checkpoint"
        );
    }
}

/// Emit TaskEnd after recording an exact bootstrap protocol direct-return.
#[allow(clippy::too_many_arguments)]
pub async fn emit_direct_return_task_end(
    host: &dyn TurnHost,
    emitter: &aidaemon_events::EventEmitter,
    task_id: &str,
    status: TaskStatus,
    outcome: aidaemon_events::TaskOutcome,
    task_start: Instant,
    iteration: usize,
    tool_calls_count: usize,
    error: Option<String>,
    summary: Option<String>,
    direct_return_succeeded: bool,
) {
    if host.turn_runtime().harness_eval_enabled() {
        host.turn_runtime()
            .with_harness_eval(|eval| eval.record_direct_return(true, direct_return_succeeded))
            .await;
    }
    emit_task_end(
        host,
        emitter,
        task_id,
        status,
        outcome,
        task_start,
        iteration,
        tool_calls_count,
        error,
        summary,
    )
    .await;
}

/// Emit a TaskEnd event. Called from every exit path in the agent loop.
#[allow(clippy::too_many_arguments)]
pub async fn emit_task_end(
    host: &dyn TurnHost,
    emitter: &aidaemon_events::EventEmitter,
    task_id: &str,
    status: TaskStatus,
    outcome: aidaemon_events::TaskOutcome,
    task_start: Instant,
    iteration: usize,
    tool_calls_count: usize,
    error: Option<String>,
    summary: Option<String>,
) {
    // One replayed task aggregate is the final authority. Individual exit
    // paths propose a transport outcome, but cannot override durable
    // obligations in either direction. This deliberately replaces the
    // former asymmetric receipt merge that could downgrade success yet
    // refused to promote a fully proven task after a finalizer failure.
    let run_aggregate = host
        .events()
        .task_run_aggregate(emitter.session_id(), task_id)
        .await
        .unwrap_or_else(|_| host.events().empty_run_aggregate(task_id));
    // A background launch receipt is a nonterminal transition owned by a
    // durable notifier. A local finalizer/budget path may have proposed
    // `Failed` after the inline loop stopped, but it cannot turn that
    // in-flight invocation into a terminal failure. Persist the parent as
    // a completed partial handoff; the later terminal receipt/continuation
    // owns the eventual success or failure.
    let background_in_flight = run_aggregate.has_background_in_flight();
    let effective_status = if background_in_flight && status != TaskStatus::Cancelled {
        TaskStatus::Completed
    } else {
        status
    };
    let (terminal_decision, effective_outcome) = task_end_decision_and_outcome(
        &run_aggregate,
        effective_status,
        outcome,
        background_in_flight,
    );
    let durable_tool_calls_count = host
        .events()
        .task_event_count(
            emitter.session_id(),
            task_id,
            aidaemon_events::EventType::ToolCall,
        )
        .await
        .unwrap_or(tool_calls_count as u32);
    let durable_duration_secs = host
        .events()
        .task_elapsed_secs(emitter.session_id(), task_id)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| task_start.elapsed().as_secs());
    // Every terminalization path carries a task-level cost aggregate,
    // including cancellation/watchdog exits with zero provider calls.
    // A missing row is a real zero, not an unknown lifecycle state.
    let efficiency = Some(
        task_efficiency_data(host, task_id)
            .await
            .unwrap_or_default(),
    );
    // Task outcome is the authoritative failure signal.  Some bounded
    // closeout paths intentionally return a user-facing fallback with a
    // `Completed` transport status while the semantic outcome is
    // `Failed`; recording only from the transport status silently drops
    // their token cost from policy telemetry.  Existing `Failed` paths
    // retain their legacy pre-boundary increment, so this branch covers
    // only the previously uncounted status projection.
    if !effective_outcome.task_success() && effective_status != TaskStatus::Failed {
        let tokens = efficiency
            .as_ref()
            .map(|data| data.input_tokens.saturating_add(data.output_tokens))
            .unwrap_or_default();
        crate::policy::metrics::record_failed_task_tokens(tokens);
    }
    let harness_eval_snapshot = if host.turn_runtime().harness_eval_config.enabled {
        let acc = host.turn_runtime().harness_eval.write().await.take();
        acc.map(|accumulator| {
            accumulator.finalize(
                effective_outcome,
                iteration as u32,
                durable_tool_calls_count,
                efficiency.as_ref(),
            )
        })
    } else {
        None
    };
    // Stamp the active turn so the normal TaskEnd carries turn identity
    // (recovery TaskEnd uses the checkpoint's original turn_id instead).
    let turn_id = host
        .turn_runtime()
        .current_turn_ids
        .read()
        .await
        .get(emitter.session_id())
        .cloned();
    if let Some(ref snapshot) = harness_eval_snapshot {
        crate::policy::metrics::record_harness_eval_task(snapshot);
    }
    let proof_basis = match terminal_decision {
        aidaemon_events::RunTerminalDecision::Succeeded => {
            Some(aidaemon_events::TaskProofBasis::Contract)
        }
        aidaemon_events::RunTerminalDecision::SucceededByEvidence => {
            Some(aidaemon_events::TaskProofBasis::Evidence)
        }
        aidaemon_events::RunTerminalDecision::ClosedByPolicyDenial => {
            Some(aidaemon_events::TaskProofBasis::PolicyDenial)
        }
        _ => None,
    };
    if proof_basis == Some(aidaemon_events::TaskProofBasis::Evidence) {
        tracing::info!(
            task_id,
            operation_count = run_aggregate.operations.len(),
            obligations_required = run_aggregate.required_count(),
            obligations_satisfied = run_aggregate.satisfied_count(),
            "Task closed by receipt evidence; compiled contract could not credit completed work"
        );
    }
    emit_decision_point(
        host,
        emitter,
        task_id,
        iteration,
        aidaemon_events::DecisionType::PostExecutionValidation,
        "Task finalization reconciled against durable receipt state",
        serde_json::json!({
            "condition": "task_finalization_reconciled",
            "requested_outcome": outcome,
            "effective_outcome": effective_outcome,
            "requested_status": status,
            "status": effective_status,
            "background_in_flight": background_in_flight,
            "run_aggregate_schema_version": run_aggregate.schema_version,
            "run_contract_present": run_aggregate.contract_present,
            "run_terminal_decision": format!("{:?}", terminal_decision).to_lowercase(),
            "proof_basis": proof_basis,
            "evidence_closed": run_aggregate.evidence_closed(),
            "executor_claimed_outcome": run_aggregate.executor_claimed_outcome.map(|claim| claim.as_str()),
            "task_kernel_phase": format!("{:?}", run_aggregate.lifecycle_phase()).to_lowercase(),
            "obligations_required": run_aggregate.required_count(),
            "obligations_satisfied": run_aggregate.satisfied_count(),
            "effect_revision": run_aggregate.effect_revision,
            "operation_count": run_aggregate.operations.len(),
            "receipt_cardinality_violations": run_aggregate.cardinality_violations,
            "primary_causal_operation_id": run_aggregate.primary_causal_operation_id,
            "invariant_violations": run_aggregate.invariant_violations,
            "tool_calls_count": durable_tool_calls_count,
        }),
    )
    .await;
    let completion_proof = aidaemon_events::TaskCompletionProofData {
        schema_version: 1,
        task_id: task_id.to_string(),
        request_turn_id: turn_id.clone(),
        response_message_ids: host
            .events()
            .task_response_message_ids(task_id)
            .await
            .unwrap_or_default(),
        receipt_refs: host
            .events()
            .task_completion_proof_references(task_id)
            .await
            .unwrap_or_default(),
        closed_at: chrono::Utc::now().to_rfc3339(),
        proof_basis,
    };
    let _ = emitter
        .emit(
            EventType::TaskEnd,
            TaskEndData {
                task_id: task_id.to_string(),
                status: effective_status,
                outcome: Some(effective_outcome),
                duration_secs: durable_duration_secs,
                iterations: iteration as u32,
                tool_calls_count: durable_tool_calls_count,
                error,
                summary,
                efficiency,
                turn_id,
                completion_proof: Some(completion_proof),
                harness_eval: harness_eval_snapshot,
            },
        )
        .await;
    // Counters remain lock-free on hot paths. Checkpoint their cumulative
    // value at the authoritative task boundary so restarts do not erase
    // longitudinal policy telemetry. Readers select the latest row per
    // boot_id rather than summing cumulative rows.
    let _ = emitter
        .emit(
            EventType::PolicyMetricsSnapshot,
            crate::policy::metrics::durable_policy_metrics_snapshot(),
        )
        .await;
    if let Err(err) = crate::turn::dialogue_state::record_dialogue_task_end(
        host,
        emitter.session_id(),
        task_id,
        effective_status,
        effective_outcome,
    )
    .await
    {
        warn!(
            session_id = emitter.session_id(),
            task_id,
            error = %err,
            "Failed to record dialogue task end"
        );
    }
    run_task_end_tool_hooks(host, task_id, emitter.session_id()).await;
    emit_turn_efficiency_signal(host, emitter, task_id, iteration).await;
}

/// Tier 2 reflection signal: roll up this turn's `LlmCall` telemetry and
/// log it. When the turn looks inefficient (fallbacks, retries, heavy
/// iteration loops, or large token-estimate drift), also emit a warning
/// `DecisionPoint` so the agent's own `self_diagnose` surfaces it.
pub async fn task_efficiency_data(
    host: &dyn TurnHost,
    task_id: &str,
) -> Option<aidaemon_events::TaskEfficiencyData> {
    let summary = host.events().get_task_llm_stats(task_id).await.ok()?;
    if summary.total_calls == 0 {
        return None;
    }
    let drift = summary.est_input_drift();
    let reasons = task_efficiency_reasons(&summary, drift);
    Some(aidaemon_events::TaskEfficiencyData {
        llm_calls: summary.total_calls,
        attempts: summary.total_attempts,
        fell_back_count: summary.fell_back_count,
        p95_latency_ms: summary.p95_latency_ms,
        max_latency_ms: summary.max_latency_ms,
        max_latency_iteration: summary.max_latency_iteration,
        input_tokens: summary.total_input_tokens,
        output_tokens: summary.total_output_tokens,
        failed_est_input_tokens: (summary.failed_est_input_tokens > 0)
            .then_some(summary.failed_est_input_tokens),
        cached_input_tokens: if summary.cached_input_token_samples > 0 {
            Some(summary.total_cached_input_tokens)
        } else {
            None
        },
        cache_creation_input_tokens: if summary.cache_creation_input_token_samples > 0 {
            Some(summary.total_cache_creation_input_tokens)
        } else {
            None
        },
        fresh_input_tokens: if summary.cached_input_token_samples > 0 {
            Some(
                summary
                    .total_input_tokens
                    .saturating_sub(summary.total_cached_input_tokens),
            )
        } else {
            None
        },
        est_input_drift: drift,
        final_model: summary.final_model,
        reasons,
    })
}

pub fn task_efficiency_reasons(
    summary: &aidaemon_events::TaskLlmSummary,
    drift: i64,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if summary.fell_back_count > 0 {
        reasons.push(format!("{} fallback(s)", summary.fell_back_count));
    }
    if summary.total_attempts > summary.total_calls {
        reasons.push(format!(
            "{} retry attempt(s) over {} call(s)",
            summary.total_attempts - summary.total_calls,
            summary.total_calls
        ));
    }
    if summary.total_calls >= 8 {
        reasons.push(format!("{} LLM calls (heavy loop)", summary.total_calls));
    }
    if summary.est_samples > 0 && summary.actual_input_tokens_with_est > 0 {
        let pct = (drift.abs() as f64 / summary.actual_input_tokens_with_est as f64) * 100.0;
        if pct >= 30.0 {
            reasons.push(format!("token estimate off by {pct:.0}%"));
        }
    }
    reasons
}

pub async fn emit_turn_efficiency_signal(
    host: &dyn TurnHost,
    emitter: &aidaemon_events::EventEmitter,
    task_id: &str,
    iteration: usize,
) {
    let summary = match host.events().get_task_llm_stats(task_id).await {
        Ok(s) if s.total_calls > 0 => s,
        _ => return,
    };
    let drift = summary.est_input_drift();
    info!(
        task_id,
        session_id = emitter.session_id(),
        llm_calls = summary.total_calls,
        attempts = summary.total_attempts,
        fell_back = summary.fell_back_count,
        p95_latency_ms = summary.p95_latency_ms,
        max_latency_ms = summary.max_latency_ms,
        input_tokens = summary.total_input_tokens,
        output_tokens = summary.total_output_tokens,
        est_input_drift = drift,
        final_model = summary.final_model.as_deref().unwrap_or("?"),
        "Turn efficiency summary"
    );

    if !summary.is_inefficient() {
        return;
    }
    let reasons = task_efficiency_reasons(&summary, drift);
    let summary_text = format!("Inefficient turn: {}", reasons.join(", "));
    emit_warning_decision_point(
        host,
        emitter,
        task_id,
        iteration,
        DecisionType::LlmEfficiencyAlert,
        summary_text,
        json!({
            "reason": reasons.join(", "),
            "llm_calls": summary.total_calls,
            "attempts": summary.total_attempts,
            "fell_back_count": summary.fell_back_count,
            "p95_latency_ms": summary.p95_latency_ms,
            "max_latency_ms": summary.max_latency_ms,
            "max_latency_iteration": summary.max_latency_iteration,
            "input_tokens": summary.total_input_tokens,
            "output_tokens": summary.total_output_tokens,
            "est_input_drift": drift,
            "final_model": summary.final_model,
        }),
    )
    .await;
}

pub async fn emit_decision_point(
    host: &dyn TurnHost,
    emitter: &aidaemon_events::EventEmitter,
    task_id: &str,
    iteration: usize,
    decision_type: DecisionType,
    summary: impl Into<String>,
    metadata: Value,
) {
    emit_decision_point_with_severity(
        host,
        emitter,
        task_id,
        iteration,
        DecisionPointEmission {
            decision_type,
            severity: aidaemon_events::DiagnosticSeverity::Info,
            summary: summary.into(),
            metadata,
        },
    )
    .await;
}

pub async fn emit_warning_decision_point(
    host: &dyn TurnHost,
    emitter: &aidaemon_events::EventEmitter,
    task_id: &str,
    iteration: usize,
    decision_type: DecisionType,
    summary: impl Into<String>,
    metadata: Value,
) {
    emit_decision_point_with_severity(
        host,
        emitter,
        task_id,
        iteration,
        DecisionPointEmission {
            decision_type,
            severity: aidaemon_events::DiagnosticSeverity::Warning,
            summary: summary.into(),
            metadata,
        },
    )
    .await;
}

async fn emit_decision_point_with_severity(
    host: &dyn TurnHost,
    emitter: &aidaemon_events::EventEmitter,
    task_id: &str,
    iteration: usize,
    emission: DecisionPointEmission,
) {
    if !host.turn_runtime().record_decision_points {
        return;
    }
    emit_decision_point_event(
        emitter,
        task_id,
        iteration,
        emission.decision_type,
        emission.severity,
        emission.summary,
        emission.metadata,
    )
    .await;
}

/// Emit one decision point event. Callers check whether decision points are
/// recorded.
pub async fn emit_decision_point_event(
    emitter: &aidaemon_events::EventEmitter,
    task_id: &str,
    iteration: usize,
    decision_type: DecisionType,
    severity: aidaemon_events::DiagnosticSeverity,
    summary: String,
    metadata: Value,
) {
    let code = metadata
        .as_object()
        .and_then(|obj| {
            ["condition", "route_reason", "reason"]
                .iter()
                .find_map(|key| obj.get(*key).and_then(Value::as_str))
        })
        .map(|value| value.to_string())
        .or_else(|| Some(decision_type.as_str().to_string()));
    let _ = emitter
        .emit(
            EventType::DecisionPoint,
            DecisionPointData {
                decision_type,
                task_id: task_id.to_string(),
                iteration: iteration.min(u32::MAX as usize) as u32,
                severity,
                code,
                metadata,
                summary,
            },
        )
        .await;
}

/// The run's terminal decision and semantic outcome at task end, from the
/// replayed aggregate, the effective transport status, and the outcome the
/// exit path proposed.
pub fn task_end_decision_and_outcome(
    run_aggregate: &aidaemon_events::RunAggregate,
    effective_status: TaskStatus,
    outcome: aidaemon_events::TaskOutcome,
    background_in_flight: bool,
) -> (
    aidaemon_events::RunTerminalDecision,
    aidaemon_events::TaskOutcome,
) {
    // At task end there is no loop left to make progress: a `Pending`
    // ledger whose terminal receipts all succeeded or were credited is
    // closed by evidence (the still-open obligations are descriptions the
    // contract could not bind to the work that demonstrably happened).
    let terminal_decision = match run_aggregate.terminal_decision() {
        aidaemon_events::RunTerminalDecision::Pending
            if effective_status != TaskStatus::Cancelled && run_aggregate.evidence_closed() =>
        {
            aidaemon_events::RunTerminalDecision::SucceededByEvidence
        }
        // `Unspecified` means no expectations were declared (no assessed
        // contract and no executor checklist) or none bound to an obligation.
        // Such a run whose only receipts are typed policy denials, and which
        // would otherwise close as uncredited plain success, closed at the
        // authority boundary: record that proof basis. Any other run (failed,
        // interrupted, cancelled, or proposed failed/partial) keeps its own
        // outcome; a denial never turns it into success.
        aidaemon_events::RunTerminalDecision::Unspecified
            if effective_status == TaskStatus::Completed
                && outcome == aidaemon_events::TaskOutcome::Succeeded
                && run_aggregate.terminal_policy_denial()
                && !run_aggregate.owner_declined() =>
        {
            aidaemon_events::RunTerminalDecision::ClosedByPolicyDenial
        }
        decision => decision,
    };
    let effective_outcome = match (effective_status, terminal_decision) {
        (TaskStatus::Cancelled, _) => aidaemon_events::TaskOutcome::Failed,
        (_, aidaemon_events::RunTerminalDecision::Pending) if background_in_flight => {
            aidaemon_events::TaskOutcome::Partial
        }
        // A refused operation is work the owner asked for that did not happen,
        // whether or not the contract listed it. Obligations only cover what the
        // contract expected, so a fulfilled contract beside a refused call would
        // otherwise close the request as answered and drop the refused work.
        // Stay partial so the request remains open for the owner's follow-up.
        // `ClosedByPolicyDenial` (nothing else dispatched) keeps its boundary
        // rule below.
        (
            _,
            aidaemon_events::RunTerminalDecision::Succeeded
            | aidaemon_events::RunTerminalDecision::SucceededByEvidence,
        ) if outcome == aidaemon_events::TaskOutcome::Succeeded
            && run_aggregate.owed_policy_denials() > 0 =>
        {
            aidaemon_events::TaskOutcome::Partial
        }
        (
            _,
            aidaemon_events::RunTerminalDecision::Succeeded
            | aidaemon_events::RunTerminalDecision::SucceededByEvidence,
        ) => aidaemon_events::TaskOutcome::Succeeded,
        // Policy refused everything the request asked for: the work did not
        // happen. The run is final at that boundary (partial, never failed,
        // so nothing retries a denial), and the request stays open for the
        // owner's follow-up.
        (_, aidaemon_events::RunTerminalDecision::ClosedByPolicyDenial) => {
            aidaemon_events::TaskOutcome::Partial
        }
        (_, aidaemon_events::RunTerminalDecision::Failed) => aidaemon_events::TaskOutcome::Failed,
        (TaskStatus::Failed, _) => aidaemon_events::TaskOutcome::Failed,
        // The contract's work is still owed and nothing ran at all: there
        // is no progress to call partial.
        (_, aidaemon_events::RunTerminalDecision::Pending)
            if outcome == aidaemon_events::TaskOutcome::Succeeded
                && run_aggregate.operations.is_empty() =>
        {
            aidaemon_events::TaskOutcome::Failed
        }
        (_, aidaemon_events::RunTerminalDecision::Pending)
            if outcome == aidaemon_events::TaskOutcome::Succeeded =>
        {
            aidaemon_events::TaskOutcome::Partial
        }
        (_, aidaemon_events::RunTerminalDecision::Pending) => outcome,
        // No contract to judge by: receipts are the only proof, and a run
        // whose operations all failed accomplished nothing, whether it
        // ended on a reply or was stopped.
        (_, aidaemon_events::RunTerminalDecision::Unspecified)
            if run_aggregate.only_failed_operations() =>
        {
            aidaemon_events::TaskOutcome::Failed
        }
        // A refused operation is owed work that did not happen. When the run
        // also dispatched something else, the refusal is not "no proof of
        // work" (`ClosedByPolicyDenial` handles that case); without this arm
        // the reply's success would close the owner's request as answered and
        // the refused work would be lost. Stay partial so the request remains
        // open for the owner's follow-up.
        // Same refusal rule for runs whose contract expected nothing.
        (_, aidaemon_events::RunTerminalDecision::Unspecified)
            if outcome == aidaemon_events::TaskOutcome::Succeeded
                && run_aggregate.owed_policy_denials() > 0 =>
        {
            aidaemon_events::TaskOutcome::Partial
        }
        (_, aidaemon_events::RunTerminalDecision::Unspecified) => outcome,
    };
    // The owner declined part of what the run set out to do, or the working
    // agent itself said the request is partial or blocked: whatever the
    // receipts or the closing reply say, that is not a success.
    let effective_outcome = if effective_outcome == aidaemon_events::TaskOutcome::Succeeded
        && (run_aggregate.owner_declined() || run_aggregate.executor_claimed_unfinished())
    {
        aidaemon_events::TaskOutcome::Partial
    } else {
        effective_outcome
    };
    (terminal_decision, effective_outcome)
}
