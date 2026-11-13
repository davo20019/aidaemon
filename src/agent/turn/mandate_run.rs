#[cfg(test)]
pub(in crate::agent) use aidaemon_runtime::engine::mandate_run::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::store_prelude::*;
    use aidaemon_runtime::host::TurnHost as _;

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
            .mandate_execution()
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

    /// A fresh cycle's balance bounds what a call may spend; it is never sent
    /// as the call's output cap (a 250k cap was rejected by a model whose
    /// output limit is 128k, and the review never ran).
    #[test]
    fn a_large_balance_never_becomes_an_oversized_output_cap() {
        assert_eq!(
            mandate_output_token_ceiling(250_000, 2_000),
            Some(MANDATE_MAX_OUTPUT_TOKENS_PER_CALL)
        );
        assert_eq!(
            mandate_output_token_ceiling(i64::MAX, 0),
            Some(MANDATE_MAX_OUTPUT_TOKENS_PER_CALL)
        );
    }
}
