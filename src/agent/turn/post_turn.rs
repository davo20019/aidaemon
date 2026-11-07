#![allow(clippy::items_after_test_module)]

use super::*;

/// Learn from a finished task in the background.
pub(in crate::agent) fn spawn_learning(
    state: Arc<dyn crate::traits::StateStore>,
    learning_ctx: LearningContext,
) {
    tokio::spawn(async move {
        if let Err(e) = post_task::process_learning(&state, learning_ctx).await {
            warn!("Learning failed: {}", e);
        }
    });
}

/// Decide whether ordinary conversational memory maintenance may run after a
/// completed turn. Mandate workers have their own bounded, typed continuity;
/// allowing either background path here would both leak mandate content into
/// global owner memory and make unmetered provider calls outside the durable
/// per-run token lease.
///
/// Worker sessions (specialists, sub-agents, background jobs) keep their
/// session-local summaries but never extract owner facts: their "user" turn
/// is a delegated brief, so every scheduled run would re-learn its own
/// instructions as durable memory. Consolidation and retention already treat
/// these sessions as memory-isolated.
pub(in crate::agent) fn post_completion_memory_plan(
    mandate_execution: bool,
    memory_isolated_session: bool,
    memory_pipeline_allowed: bool,
    progressive_facts_enabled: bool,
    fact_extraction_eligible: bool,
    summarization_enabled: bool,
) -> (bool, bool) {
    if mandate_execution || !memory_pipeline_allowed {
        return (false, false);
    }
    (
        !memory_isolated_session && progressive_facts_enabled && fact_extraction_eligible,
        summarization_enabled,
    )
}

#[cfg(test)]
mod gate_telemetry_tests {
    use super::*;

    #[test]
    fn mandate_completion_disables_fact_extraction_and_summarization() {
        assert_eq!(
            post_completion_memory_plan(true, false, true, true, true, true),
            (false, false)
        );
        assert_eq!(
            post_completion_memory_plan(false, false, true, true, true, true),
            (true, true)
        );
        assert_eq!(
            post_completion_memory_plan(false, false, false, true, true, true),
            (false, false),
            "a task-local no-memory policy suppresses every post-turn pipeline"
        );
    }

    #[test]
    fn worker_sessions_keep_summaries_but_never_extract_owner_facts() {
        assert_eq!(
            post_completion_memory_plan(false, true, true, true, true, true),
            (false, true)
        );
    }

    #[tokio::test]
    async fn mandate_prose_closeout_retries_and_corrected_typed_wait_commits() {
        use crate::testing::{setup_test_agent_with_mandates, MockProvider};
        use crate::traits::store_prelude::*;
        use crate::traits::{Goal, Mandate, MandateAuthority, MandateDecisionOutcome, Task};

        let provider = MockProvider::with_responses(vec![
            MockProvider::text_response("Decision: WAIT. No worthwhile action is available."),
            MockProvider::tool_call_response(
                "manage_mandates",
                &json!({
                    "action": "record_decision",
                    "outcome": "wait",
                    "activity_level": "quiet",
                    "rationale": "No worthwhile action is available in this review.",
                    "reconsider_minutes": 180,
                    "learning_note": "",
                    "learning_evidence_receipt_ids": [],
                    "strategy_key": "",
                    "strategy_kind": "explore",
                    "strategy_confidence_bps": 0
                })
                .to_string(),
            ),
        ]);
        let mut harness = setup_test_agent_with_mandates(provider).await.unwrap();
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
            3_600,
            21_600,
            10_800,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        harness
            .state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let leased = harness
            .state
            .claim_due_mandates(1, "prose-closeout-test", 300)
            .await
            .unwrap()
            .pop()
            .expect("one due mandate");
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let root_task = Task {
            id: root_task_id.clone(),
            goal_id: goal.id.clone(),
            description: "Record one bounded decision".to_string(),
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
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        let run_id = uuid::Uuid::new_v4().to_string();
        let run = harness
            .state
            .create_mandate_review_run(
                &mandate.id,
                leased.review_lease_token.as_deref().unwrap(),
                &run_id,
                &root_task,
            )
            .await
            .unwrap();
        let attempt = harness
            .state
            .claim_task_with_lease(&root_task_id, "prose-closeout-task-lead", None, 300)
            .await
            .unwrap()
            .unwrap();
        harness.agent.set_test_mandate_execution(
            &mandate.id,
            mandate.version,
            mandate.authority.clone(),
            &goal.id,
            &root_task_id,
            &attempt.id,
            &attempt,
        );

        let response = harness
            .agent
            .handle_message(
                "synthetic-mandate-session",
                "Run one bounded autonomous review and record exactly one decision.",
                None,
                crate::types::UserRole::Owner,
                crate::types::ChannelContext::internal(),
                None,
            )
            .await
            .unwrap();

        assert!(!response.contains("Decision: WAIT"));
        assert_eq!(harness.provider.call_count().await, 2);
        let decision = harness
            .state
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .expect("corrected durable decision");
        assert_eq!(decision.outcome, MandateDecisionOutcome::Wait);
        let calls = harness.provider.call_log.lock().await;
        let retry_messages = &calls[1].messages;
        assert!(retry_messages.iter().any(|message| {
            message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("no durable decision"))
        }));
    }
}
