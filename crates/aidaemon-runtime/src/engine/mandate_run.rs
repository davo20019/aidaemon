//! A mandate worker's model calls: each one is admitted through the exact
//! goal run's durable token balance (a short lease), and a worker's run
//! commits its decision before it may end.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use tracing::warn;
use uuid::Uuid;

use crate::host::TurnHost;
use crate::turn::helpers::touch_heartbeat;

#[derive(Debug)]
pub struct MandateRunTokenLease {
    pub goal_run_id: String,
    pub lease_token: String,
    pub tokens_used_before: i64,
    pub budget_per_cycle: i64,
}

pub enum MandateRunTokenAdmission {
    NotApplicable,
    Acquired(MandateRunTokenLease),
    Exhausted,
}

/// Serialize every task-lead/executor model call through the exact mandate
/// goal run and its immutable durable token balance. Waiting workers poll the
/// short lease. If a process dies after dispatch, expiry consumes the remaining
/// balance fail-closed instead of admitting a retry with unknowable prior spend.
pub async fn acquire_mandate_run_token_lease<H: TurnHost + ?Sized>(
    agent: &H,
    lease_secs: i64,
    heartbeat: &Option<Arc<AtomicU64>>,
) -> anyhow::Result<MandateRunTokenAdmission> {
    let Some(fence) = agent.mandate_execution() else {
        return Ok(MandateRunTokenAdmission::NotApplicable);
    };
    let goal = agent
        .goals()
        .get_goal(&fence.goal_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("mandate controller goal is missing"))?;
    let configured_budget = goal
        .budget_per_check
        .filter(|budget| *budget > 0)
        .ok_or_else(|| anyhow::anyhow!("mandate controller has no positive cycle token budget"))?;
    let (budget_per_cycle, tokens_used) = agent
        .token_usage()
        .ensure_mandate_run_token_budget(
            &fence.goal_run_id,
            &fence.mandate_id,
            fence.mandate_version,
            configured_budget,
        )
        .await?;
    if tokens_used >= budget_per_cycle {
        return Ok(MandateRunTokenAdmission::Exhausted);
    }

    let lease_token = Uuid::new_v4().to_string();
    loop {
        let (acquired, current_tokens, current_budget) = agent
            .token_usage()
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
            return Ok(MandateRunTokenAdmission::Exhausted);
        }
        if agent
            .cancel_token()
            .is_some_and(|token| token.is_cancelled())
        {
            anyhow::bail!("mandate model-call admission cancelled while waiting for cycle lease");
        }
        touch_heartbeat(heartbeat);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

pub async fn release_mandate_run_token_lease_without_dispatch<H: TurnHost + ?Sized>(
    agent: &H,
    lease: Option<&MandateRunTokenLease>,
    session_id: &str,
) {
    let Some(lease) = lease else {
        return;
    };
    match agent
        .token_usage()
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
pub async fn exhaust_mandate_run_token_lease_after_ambiguous_call<H: TurnHost + ?Sized>(
    agent: &H,
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
        .token_usage()
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

const MANDATE_INPUT_RESERVATION_MARGIN_NUMERATOR: i64 = 5;
const MANDATE_INPUT_RESERVATION_MARGIN_DENOMINATOR: i64 = 4;
const MANDATE_INPUT_RESERVATION_FIXED_TOKENS: i64 = 512;

/// Prompt estimates are calibrated but can undercount. Reserve 125% plus a
/// fixed 512-token cushion before deriving the provider-side output ceiling.
pub fn mandate_input_token_reservation(estimated_input_tokens: u32) -> i64 {
    i64::from(estimated_input_tokens)
        .saturating_mul(MANDATE_INPUT_RESERVATION_MARGIN_NUMERATOR)
        .saturating_add(MANDATE_INPUT_RESERVATION_MARGIN_DENOMINATOR - 1)
        / MANDATE_INPUT_RESERVATION_MARGIN_DENOMINATOR
        + MANDATE_INPUT_RESERVATION_FIXED_TOKENS
}

/// The most output one mandate model call may ask for, however much of the
/// cycle's balance is left. The balance bounds what a call may spend; it is
/// not a request size. Sent as the call's output cap, a whole cycle's balance
/// (hundreds of thousands of tokens) exceeds every model's output limit, and
/// the provider rejects the request before it runs. Every model the daemon
/// talks to accepts this many.
pub const MANDATE_MAX_OUTPUT_TOKENS_PER_CALL: u32 = 8_192;

pub fn mandate_output_token_ceiling(remaining: i64, estimated_input_tokens: u32) -> Option<u32> {
    let available =
        remaining.saturating_sub(mandate_input_token_reservation(estimated_input_tokens));
    (available > 0).then(|| {
        u32::try_from(available)
            .unwrap_or(u32::MAX)
            .min(MANDATE_MAX_OUTPUT_TOKENS_PER_CALL)
    })
}

pub async fn committed_non_action_mandate_summary<H: TurnHost + ?Sized>(
    agent: &H,
) -> Option<String> {
    let fence = agent.mandate_execution()?;
    let decision = agent
        .mandates()
        .get_mandate_decision_for_run(&fence.goal_run_id)
        .await
        .ok()
        .flatten()?;
    if decision.mandate_id != fence.mandate_id
        || decision.mandate_version != fence.mandate_version
        || decision.is_runtime_fallback()
    {
        return None;
    }
    decision.non_action_summary()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MandateDecisionCommitState {
    NotApplicable,
    Missing,
    Committed,
    Invalid,
}

pub async fn mandate_decision_commit_state<H: TurnHost + ?Sized>(
    agent: &H,
) -> anyhow::Result<MandateDecisionCommitState> {
    let Some(fence) = agent.mandate_execution() else {
        return Ok(MandateDecisionCommitState::NotApplicable);
    };
    if fence.worker_task_id != fence.root_task_id {
        return Ok(MandateDecisionCommitState::NotApplicable);
    }
    let Some(decision) = agent
        .mandates()
        .get_mandate_decision_for_run(&fence.goal_run_id)
        .await?
    else {
        return Ok(MandateDecisionCommitState::Missing);
    };
    if decision.goal_run_id == fence.goal_run_id
        && decision.mandate_id == fence.mandate_id
        && decision.mandate_version == fence.mandate_version
    {
        Ok(MandateDecisionCommitState::Committed)
    } else {
        Ok(MandateDecisionCommitState::Invalid)
    }
}
