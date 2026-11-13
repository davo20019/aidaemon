//! A goal's background task lead lives in
//! `aidaemon_runtime::background_task_lead`.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aidaemon_runtime::background_task_lead::*;

    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{
        Goal, Intention, Mandate, MandateAuthority, MandateDecisionCycle, MandateDecisionOutcome,
        MandateFinalizationRejectReason, MandateRunNotificationKind, MandateStatus, StateStore,
        Task,
    };
    use aidaemon_work::scheduler::SchedulerStore;

    async fn mandate_test_state() -> (Arc<dyn SchedulerStore>, tempfile::NamedTempFile) {
        let database = tempfile::NamedTempFile::new().unwrap();
        let store: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::for_tests(
                database.path().to_str().unwrap(),
                100,
                None,
                Arc::new(EmbeddingService::deterministic()),
            )
            .await
            .unwrap(),
        );
        (store, database)
    }

    fn due_mandate_controller(session_id: &str) -> (Goal, Mandate) {
        let goal = Goal::new_continuous(
            "Steward an account",
            session_id,
            Some(10_000),
            Some(100_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Maintain a useful account presence",
            session_id,
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        (goal, mandate)
    }

    async fn claimed_mandate_run(
        state: &Arc<dyn SchedulerStore>,
        goal: &Goal,
        mandate: &Mandate,
    ) -> crate::traits::GoalRun {
        state
            .create_mandate_controller(goal, mandate)
            .await
            .unwrap();
        assert_eq!(
            state
                .claim_due_mandates(1, "background-finalizer-test", 300)
                .await
                .unwrap()
                .len(),
            1
        );
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let run = state
            .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
            .await
            .unwrap();
        state
            .create_task(&Task {
                id: root_task_id.clone(),
                goal_id: goal.id.clone(),
                description: "Run one bounded mandate review".to_string(),
                status: "pending".to_string(),
                priority: goal.priority.clone(),
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
            })
            .await
            .unwrap();
        let attempt = state
            .claim_task_with_lease(
                &root_task_id,
                "background-finalizer-test-root",
                Some("profile-task-lead"),
                300,
            )
            .await
            .unwrap()
            .unwrap();
        assert!(
            state
                .patch_task_from_attempt(
                    &attempt.id,
                    &attempt.lease_token,
                    &crate::traits::TaskAttemptPatch {
                        status: "completed".to_string(),
                        result: Some("bounded review completed".to_string()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
        );
        run
    }

    #[tokio::test]
    async fn dispatch_trigger_promotion_moves_the_claimed_attempt_to_running() {
        let (state, _database) = mandate_test_state().await;
        let goal = Goal::new_finite("Coordinate synthetic work", "owner-session");
        state.create_goal(&goal).await.unwrap();
        state
            .start_goal_run(&goal.id, "manual", None, None)
            .await
            .unwrap();
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Run a synthetic task lead".to_string(),
            status: "pending".to_string(),
            priority: goal.priority.clone(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&task).await.unwrap();
        let attempt = state
            .claim_task_with_lease(
                &task.id,
                "heartbeat-dispatch-synthetic",
                Some("profile-task-lead"),
                180,
            )
            .await
            .unwrap()
            .expect("trigger claim");

        assert!(
            promote_dispatch_trigger_attempt(&state, &attempt, "task-lead-synthetic")
                .await
                .unwrap()
        );

        let promoted_task = state.get_task(&task.id).await.unwrap().unwrap();
        assert_eq!(promoted_task.status, "running");
        assert_eq!(
            promoted_task.agent_id.as_deref(),
            Some("task-lead-synthetic")
        );
        let promoted_attempt = state
            .get_current_task_attempt(&task.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(promoted_attempt.status, "running");
        assert_eq!(promoted_attempt.worker_instance_id, "task-lead-synthetic");
    }

    #[tokio::test]
    async fn wait_finalization_keeps_controller_active_and_does_not_require_a_live_lease() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mandate) = due_mandate_controller("owner-session");
        let run = claimed_mandate_run(&state, &goal, &mandate).await;
        let mut decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Wait,
            "No useful action is available yet",
            mandate.version,
        );
        decision.reconsider_at =
            Some((chrono::Utc::now() + chrono::Duration::minutes(10)).to_rfc3339());
        state
            .record_mandate_decision(&decision, None, None)
            .await
            .unwrap();
        let after_decision = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert!(after_decision.review_lease_token.is_none());

        let notification =
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false)
                .await;

        assert!(notification.is_none());
        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
        assert_eq!(
            state
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::Active
        );
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "completed"
        );
    }

    #[tokio::test]
    async fn completed_review_without_explicit_decision_is_a_retriable_failure() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mut mandate) = due_mandate_controller("owner-session");
        let expiry = chrono::Utc::now() + chrono::Duration::minutes(2);
        mandate.expires_at = Some(expiry.to_rfc3339());
        let run = claimed_mandate_run(&state, &goal, &mandate).await;

        assert!(persist_safe_wait_if_decision_missing(&state, &goal.id, &run.id).await);
        let decision = state
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(decision.outcome, MandateDecisionOutcome::Wait);
        let reconsider_at = chrono::DateTime::parse_from_rfc3339(
            decision
                .reconsider_at
                .as_deref()
                .expect("safe WAIT reconsideration"),
        )
        .unwrap()
        .with_timezone(&chrono::Utc);
        assert_eq!(reconsider_at, expiry);

        let notification =
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false)
                .await
                .expect("safe fallback emits a durable review-failure notice");
        assert!(matches!(
            notification.kind,
            MandateRunNotificationKind::ReviewFailed {
                reason: MandateFinalizationRejectReason::DeliberatorFailed
            }
        ));
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "failed"
        );
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0]
                .outcome_summary
                .as_deref(),
            Some("mandate_review_failed:deliberator_failed")
        );
        let pending = state.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].notification_type, "mandate_review_failed");
    }

    #[tokio::test]
    async fn mandate_ask_notice_quotes_the_question_but_never_promotes_rationale_or_task_prose() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mandate) = due_mandate_controller("owner-session");
        let run = claimed_mandate_run(&state, &goal, &mandate).await;
        let sentinel = "UNTRUSTED_RATIONALE_DO_NOT_PROMOTE";
        let mut decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Ask,
            &format!("generated rationale {sentinel}"),
            mandate.version,
        );
        // The owner must see what is being asked, but only as quoted,
        // attributed, single-line generated text.
        decision.question = Some("May I read\n\"reply\" counts?\u{7}".to_string());
        state
            .record_mandate_decision(&decision, None, None)
            .await
            .unwrap();

        let notification =
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false)
                .await
                .expect("ASK emits a static control notification");
        let entry = notification.to_notification_entry();

        assert_eq!(entry.notification_type, "mandate_ask");
        assert_eq!(entry.priority, "critical");
        assert!(entry.expires_at.is_none());
        assert!(!entry.message.contains(sentinel));
        assert!(
            entry
                .message
                .contains("It asked: \"May I read 'reply' counts?\""),
            "{}",
            entry.message
        );
        assert!(!entry.message.contains('\u{7}'));
        assert!(entry.message.contains("never gives it new permissions"));
        assert!(
            entry
                .message
                .starts_with("Your automation \"Maintain a useful account presence\"")
        );
        assert!(!entry.message.contains("manage_mandates("));
        let pending = state.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, entry.id);
        assert_eq!(pending[0].message, entry.message);
        assert!(
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false,)
                .await
                .is_none()
        );
        assert_eq!(state.get_pending_notifications(10).await.unwrap().len(), 1);
        let finalized = state.get_goal_runs(&goal.id).await.unwrap();
        let summary = finalized[0].outcome_summary.as_deref().unwrap_or_default();
        assert!(!summary.contains(sentinel));
        assert_eq!(summary, "mandate_non_action_satisfied");
    }

    #[tokio::test]
    async fn failed_review_without_a_decision_releases_lease_for_bounded_retry() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mandate) = due_mandate_controller("owner-session");
        let run = claimed_mandate_run(&state, &goal, &mandate).await;
        let before = chrono::Utc::now();

        let notification = finalize_mandate_review(
            &state,
            &goal,
            &run.id,
            Some("failed"),
            &[],
            Some("deliberator exited"),
            false,
        )
        .await;

        let entry = notification.unwrap().to_notification_entry();
        assert_eq!(entry.notification_type, "mandate_review_failed");
        let pending = state.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, entry.id);
        assert_eq!(pending[0].message, entry.message);
        let retriable = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert!(retriable.review_lease_token.is_none());
        let retry_at = chrono::DateTime::parse_from_rfc3339(&retriable.next_review_at)
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(retry_at >= before + chrono::Duration::seconds(mandate.min_review_secs));
        assert!(retry_at <= chrono::Utc::now() + chrono::Duration::seconds(65));
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "failed"
        );
    }

    #[tokio::test]
    async fn policy_revision_suspends_an_act_intention_instead_of_satisfying_it() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mut mandate) = due_mandate_controller("owner-session");
        mandate.authority = MandateAuthority {
            allowed_tools: vec!["http_request".to_string()],
            allowed_mutation_effects: vec!["external_delivery".to_string()],
            allowed_target_prefixes: vec!["https://api.example.test/v1/".to_string()],
            max_mutating_actions_per_cycle: 1,
            max_mutating_actions_per_rolling_24h: 1,
            min_seconds_between_mutations: 900,
            ..MandateAuthority::default()
        };
        let run = claimed_mandate_run(&state, &goal, &mandate).await;
        let decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "A useful action is available",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &decision.id,
            &run.id,
            "Publish one relevant update",
            "It advances the mandate",
        );
        state
            .record_mandate_decision(&decision, Some(&intention), None)
            .await
            .unwrap();
        let mut revised = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        revised.version += 1;
        revised.objective = "Revised owner policy".to_string();
        state.update_mandate(&revised).await.unwrap();

        let notification =
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false)
                .await;

        assert!(notification.is_none());
        let intentions = state.list_intentions(&mandate.id, 10).await.unwrap();
        assert_eq!(intentions.len(), 1);
        assert_eq!(
            intentions[0].status,
            crate::traits::IntentionStatus::Suspended
        );
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "cancelled"
        );
    }

    #[tokio::test]
    async fn scheduled_recovery_rebinds_persisted_run_state_to_its_real_root() {
        let (state, _database) = mandate_test_state().await;
        let goal = Goal::new_continuous(
            "Publish a synthetic site update",
            "synthetic-session",
            Some(400_000),
            Some(1_000_000),
        );
        state.create_goal(&goal).await.unwrap();
        let make_task = |id: &str, description: &str| Task {
            id: id.to_string(),
            goal_id: goal.id.clone(),
            description: description.to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
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
            max_retries: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };

        let old_task = make_task("old-root", "Initial scheduled attempt");
        state.create_task(&old_task).await.unwrap();
        let old_run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        state
            .finish_goal_run(&old_run.id, "failed", Some("synthetic failure"))
            .await
            .unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
                goal_id: goal.id.clone(),
                root_task_id: old_task.id,
                effective_budget_per_check: 800_000,
                tokens_used: 525_063,
                budget_extensions_count: 1,
                health: crate::traits::ScheduledRunHealth::default(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let recovery_task = make_task("recovery-root", "Recover the failed scheduled attempt");
        state
            .start_goal_run(&goal.id, "recovery", None, Some(&recovery_task.id))
            .await
            .unwrap();
        state.create_task(&recovery_task).await.unwrap();
        let recovery_run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        assert_eq!(
            recovery_run.root_task_id.as_deref(),
            Some(recovery_task.id.as_str())
        );

        rebind_scheduled_recovery_state(&state, None, &goal.id, &recovery_task.id).await;

        let rebound = state
            .get_scheduled_run_state(&goal.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(rebound.root_task_id, recovery_task.id);
        assert_eq!(rebound.tokens_used, 525_063);
        assert_eq!(rebound.effective_budget_per_check, 800_000);
        assert_eq!(rebound.budget_extensions_count, 1);
    }
}
