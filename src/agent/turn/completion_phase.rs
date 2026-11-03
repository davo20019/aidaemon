#![allow(clippy::items_after_test_module)]

use super::*;

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
/// After a completed turn: progressive fact extraction and incremental
/// summary maintenance on the fast model, unless the turn's memory policy,
/// a mandate run or a memory-isolated session rules them out.
// Only the legacy completion phase calls these; they go when it does.
#[allow(clippy::too_many_arguments, dead_code)]
pub(in crate::agent) fn spawn_post_turn_memory(
    agent: &Agent,
    llm_provider: Arc<dyn ModelProvider>,
    llm_router: Option<&Router>,
    model: &str,
    task_id: &str,
    session_id: &str,
    user_text: &str,
    reply: &str,
    channel_ctx: &ChannelContext,
    user_role: UserRole,
    memory_persistence_allowed: bool,
) {
    let fast_model = llm_router
        .map(|router| router.select(crate::router::Tier::Fast).to_string())
        .unwrap_or_else(|| model.to_string());
    let (spawn_progressive_facts, spawn_summary_maintenance) = post_completion_memory_plan(
        agent.mandate_execution.is_some(),
        agent.depth > 0 || crate::events::is_memory_isolated_worker_session(session_id),
        memory_persistence_allowed,
        agent.context_window_config.progressive_facts,
        crate::memory::context_window::should_extract_facts(user_text),
        agent.context_window_config.enabled,
    );
    if spawn_progressive_facts {
        crate::memory::context_window::spawn_progressive_extraction(
            llm_provider.clone(),
            fast_model.clone(),
            agent.state.clone(),
            agent.event_store.clone(),
            task_id.to_string(),
            user_text.to_string(),
            reply.to_string(),
            channel_ctx.channel_id.clone(),
            channel_ctx.visibility,
            user_role,
        );
    }
    if spawn_summary_maintenance {
        let summary_token_threshold = agent
            .context_window_config
            .summarize_token_threshold_for(&fast_model);
        let summary_recent_tokens = agent
            .context_window_config
            .summary_recent_tokens_for(&fast_model);
        crate::memory::context_window::spawn_incremental_summarization(
            llm_provider,
            fast_model,
            agent.state.clone(),
            agent.event_store.clone(),
            session_id.to_string(),
            summary_token_threshold,
            summary_recent_tokens,
            user_role,
        );
    }
}

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
