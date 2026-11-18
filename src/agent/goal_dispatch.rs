//! Goal and task dispatch helpers live in `aidaemon_runtime::goal_dispatch`.

#[cfg(test)]
pub use aidaemon_runtime::goal_dispatch::*;

#[cfg(test)]
mod provenance_tests {
    use std::sync::Arc;

    use aidaemon_runtime::goal_dispatch::{
        active_scheduled_root_task_id, task_has_scheduled_provenance,
    };

    use crate::testing::{MockProvider, setup_test_agent};
    use crate::traits::{Goal, StateStore, Task};

    fn root_task(goal_id: &str) -> Task {
        Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            description: "Scheduled check".to_string(),
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
            max_retries: 3,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        }
    }

    #[tokio::test]
    async fn recovery_run_is_its_own_root_scheduled_cycle() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state: Arc<dyn StateStore> = harness.state.clone();
        let goal = Goal::new_continuous("Synthetic daily publication", "session-a", None, None);
        state.create_goal(&goal).await.unwrap();

        let failed_root = root_task(&goal.id);
        let failed = state
            .start_goal_run(&goal.id, "scheduled", None, Some(&failed_root.id))
            .await
            .unwrap();
        state.create_task(&failed_root).await.unwrap();
        state
            .finish_goal_run(&failed.id, "failed", Some("budget exhausted"))
            .await
            .unwrap();

        let recovery_root = root_task(&goal.id);
        state
            .start_goal_run(&goal.id, "recovery", None, Some(&recovery_root.id))
            .await
            .unwrap();
        state.create_task(&recovery_root).await.unwrap();

        // The recovery lead must own a fresh run budget, which requires it to
        // be recognized as the active root of a scheduled cycle rather than a
        // continuation of the failed run's exhausted token accounting.
        assert!(task_has_scheduled_provenance(&state, Some(&recovery_root.id)).await);
        assert_eq!(
            active_scheduled_root_task_id(&state, &goal.id)
                .await
                .as_deref(),
            Some(recovery_root.id.as_str())
        );
    }
}
