mod scheduler_flaw_test {
    use chrono::{Duration, Utc};
    use std::sync::Arc;

    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{Goal, GoalSchedule, StateStore};

    #[tokio::test]
    async fn a_schedule_takes_a_named_or_fixed_zone_and_refuses_an_unknown_one() {
        // 1. Setup isolated state store
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::for_tests(
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

        // 3. A schedule may be stored in UTC or a named IANA zone; a zone
        // this cannot read is refused at creation, never run on the owner's
        // clock.
        let now = Utc::now();
        let now_ts = now.to_rfc3339();
        let due_ts = (now - Duration::minutes(5)).to_rfc3339();

        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "UTC".to_string(),
            original_schedule: Some("every minute".to_string()),
            fire_policy: "always_fire".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: due_ts,
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };

        state.create_goal_schedule(&schedule).await.unwrap();
        let madrid = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            tz: "Europe/Madrid".to_string(),
            ..schedule.clone()
        };
        state.create_goal_schedule(&madrid).await.unwrap();

        let unknown = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            tz: "Mars/Olympus".to_string(),
            ..schedule
        };
        let err_msg = state
            .create_goal_schedule(&unknown)
            .await
            .unwrap_err()
            .to_string();
        assert!(err_msg.contains("Mars/Olympus"), "{err_msg}");
    }
}
