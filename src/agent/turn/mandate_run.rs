use super::*;

#[derive(Debug)]
pub(in crate::agent) struct MandateRunTokenLease {
    pub(in crate::agent) goal_run_id: String,
    pub(in crate::agent) lease_token: String,
    pub(in crate::agent) tokens_used_before: i64,
    pub(in crate::agent) budget_per_cycle: i64,
}

pub(in crate::agent) enum MandateRunTokenAdmission {
    NotApplicable,
    Acquired(MandateRunTokenLease),
    Exhausted,
}

/// Serialize every task-lead/executor model call through the exact mandate
/// goal run and its immutable durable token balance. Waiting workers poll the
/// short lease. If a process dies after dispatch, expiry consumes the remaining
/// balance fail-closed instead of admitting a retry with unknowable prior spend.
pub(in crate::agent) async fn acquire_mandate_run_token_lease(
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
        return Ok(MandateRunTokenAdmission::Exhausted);
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
            return Ok(MandateRunTokenAdmission::Exhausted);
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

pub(in crate::agent) async fn release_mandate_run_token_lease_without_dispatch(
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
pub(in crate::agent) async fn exhaust_mandate_run_token_lease_after_ambiguous_call(
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

pub(in crate::agent) fn mandate_output_token_ceiling(
    remaining: i64,
    estimated_input_tokens: u32,
) -> Option<u32> {
    let available =
        remaining.saturating_sub(mandate_input_token_reservation(estimated_input_tokens));
    (available > 0).then(|| u32::try_from(available).unwrap_or(u32::MAX))
}

pub(in crate::agent) async fn committed_non_action_mandate_summary(
    agent: &Agent,
) -> Option<String> {
    let fence = agent.mandate_execution.as_ref()?;
    let decision = agent
        .state
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
pub(in crate::agent) enum MandateDecisionCommitState {
    NotApplicable,
    Missing,
    Committed,
    Invalid,
}

pub(in crate::agent) async fn mandate_decision_commit_state(
    agent: &Agent,
) -> anyhow::Result<MandateDecisionCommitState> {
    let Some(fence) = agent.mandate_execution.as_ref() else {
        return Ok(MandateDecisionCommitState::NotApplicable);
    };
    if fence.worker_task_id != fence.root_task_id {
        return Ok(MandateDecisionCommitState::NotApplicable);
    }
    let Some(decision) = agent
        .state
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

    /// A review that runs out of budget before deciding has no committed
    /// decision to close from; a runtime fallback WAIT never counts as one,
    /// so finalization closes the review as failed (retried with backoff).
    #[tokio::test]
    async fn only_a_deliberator_decision_closes_an_exhausted_review() {
        let mut harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .expect("setup test agent");
        let (mandate, run) = install_due_mandate_root(&mut harness).await;
        assert!(committed_non_action_mandate_summary(&harness.agent)
            .await
            .is_none());

        let fallback = crate::traits::MandateDecisionCycle::runtime_fallback_wait(
            &mandate,
            &run.id,
            chrono::Utc::now(),
        );
        assert!(fallback.is_runtime_fallback());
        let attempt = harness
            .agent
            .mandate_execution
            .as_ref()
            .expect("mandate fence")
            .root_task_attempt_id
            .clone();
        harness
            .state
            .record_mandate_decision(&fallback, None, Some(attempt.as_str()))
            .await
            .expect("record runtime fallback");
        let stored = harness
            .state
            .get_mandate_decision_for_run(&run.id)
            .await
            .expect("read decision")
            .expect("stored decision");
        assert_eq!(
            stored.origin,
            crate::traits::MandateDecisionOrigin::RuntimeFallback
        );
        assert!(committed_non_action_mandate_summary(&harness.agent)
            .await
            .is_none());
    }

    #[test]
    fn mandate_output_ceiling_reserves_margin_and_never_exceeds_remaining_balance() {
        assert_eq!(mandate_input_token_reservation(1_000), 1_762);
        assert_eq!(mandate_output_token_ceiling(2_000, 1_000), Some(238));
        assert_eq!(mandate_output_token_ceiling(1_762, 1_000), None);
        assert_eq!(mandate_output_token_ceiling(1_700, 1_000), None);
    }
}
