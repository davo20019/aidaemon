
mod scheduler_flaw_test {
    use std::sync::Arc;
    use chrono::{Utc, Duration};

    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{Goal, GoalSchedule, StateStore};

    #[tokio::test]
    async fn test_scheduler_ignores_utc_timezone() {
        // 1. Setup isolated state store
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        // 2. Create a continuous goal
        let goal = Goal::new_continuous("UTC Test Goal", "test-session", None, None);
        state.create_goal(&goal).await.unwrap();

        // 3. Attempt to create a schedule with explicit "UTC" timezone.
        // The current code rejects non-"local" timezones at creation time,
        // so this should fail with an appropriate error.
        let now = Utc::now();
        let now_ts = now.to_rfc3339();
        let due_ts = (now - Duration::minutes(5)).to_rfc3339();

        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "UTC".to_string(), // <--- This is the key. Current code expects "local".
            original_schedule: Some("every minute".to_string()),
            fire_policy: "always_fire".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: due_ts,
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };

        // UTC timezone is rejected at creation time, not at tick time.
        // The create_goal_schedule call should return an error.
        let result = state.create_goal_schedule(&schedule).await;
        assert!(
            result.is_err(),
            "UTC timezone should be rejected at schedule creation time"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("tz='local'"),
            "Error message should mention tz='local' requirement, got: {}",
            err_msg
        );
    }
}

