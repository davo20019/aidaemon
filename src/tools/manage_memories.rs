//! `manage_memories` lives in `aidaemon_runtime::tools::manage_memories`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_runtime::tools::manage_memories::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::testing::CallWithFacts;
    use crate::traits::store_prelude::*;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolOutcomeStatus;
    use crate::traits::ToolSemantics;
    use crate::traits::{Goal, GoalSchedule, Task, TaskActivity};
    use crate::types::ChannelVisibility;
    use crate::types::FactPrivacy;
    use serde_json::json;
    use serde_json::Value;
    use std::sync::Arc;

    /// Memory-store actions declare no effects; privacy, goal and schedule
    /// actions stay undeclared.
    #[tokio::test]
    async fn memory_store_actions_declare_no_effects() {
        let tool = test_tool(setup_state().await);
        let effects = |args: Value| tool.call_effects(&args.to_string());
        for action in ["forget", "repair", "list", "search"] {
            assert!(effects(json!({"action": action})).complete, "{action}");
        }
        for action in [
            "set_privacy",
            "create_personal_goal",
            "create_scheduled_goal",
            "cancel_scheduled",
            "retry_scheduled",
        ] {
            assert!(!effects(json!({"action": action})).complete, "{action}");
        }
    }

    #[tokio::test]
    async fn scheduled_actions_declare_goal_identity_contract() {
        let state = setup_state().await;
        let tool = test_tool(state);
        assert!(tool
            .validate_arguments(r#"{"action":"diagnose_scheduled"}"#)
            .is_err());
        assert!(tool
            .validate_arguments(r#"{"action":"diagnose_scheduled","goal_id":"goal-synthetic"}"#)
            .is_ok());
        assert!(tool
            .validate_arguments(r#"{"action":"list_scheduled"}"#)
            .is_ok());
    }

    /// Search action appends the relational neighborhood when results contain a
    /// relationship-typed key. No LLM/MockProvider needed — entities are derived
    /// purely from matched result values.
    ///
    /// Seed: mother_name=Carol, father_name=Frank Mendez.
    /// Query "carol's mother" with limit=1 → only mother_name in initial_ids.
    /// Expansion derives "Carol" from mother_name value, assembles neighborhood,
    /// and surfaces father_name under "Related context".
    #[tokio::test]
    async fn test_search_relational_expansion() {
        let state = setup_state().await;

        state
            .upsert_fact(
                "user",
                "mother_name",
                "Carol",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "user",
                "father_name",
                "Frank Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // No MockProvider needed — expansion is driven purely by result entities.
        let tool = test_tool(state.clone());

        // limit=1 keeps only the top-ranked initial hit (mother_name) so
        // initial_ids does NOT contain father_name. The neighborhood expansion
        // is then the only path that can surface "Frank".
        let out = tool
            .call(&json!({"action": "search", "query": "carol's mother", "limit": 1}).to_string())
            .await
            .unwrap();

        // Guard: "Frank" must NOT appear in the primary (pre-neighborhood) results.
        let split = out.find("Related context").unwrap_or(out.len());
        assert!(
            !out[..split].contains("Frank"),
            "Frank must NOT be in primary results for the expansion-isolation test to be meaningful; output:\n{out}"
        );

        assert!(
            out.contains("Frank"),
            "neighborhood expansion should surface father_name fact; got:\n{out}"
        );
        assert!(
            out.contains("Related context"),
            "output should have a neighborhood header; got:\n{out}"
        );
    }

    /// Regression: keyword query (no possessive/recall phrasing) still triggers
    /// neighborhood expansion via the result-signal path.
    ///
    /// Seed: mother_name=Carol, father_name=Frank.
    /// Query "carol" (bare keyword) matches mother_name. Expansion fires
    /// because mother_name is a relationship key in the matched results.
    /// No MockProvider needed — classifier is no longer invoked.
    #[tokio::test]
    async fn test_search_keyword_query_triggers_neighborhood_via_result_signal() {
        let state = setup_state().await;

        state
            .upsert_fact(
                "user",
                "mother_name",
                "Carol",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "user",
                "father_name",
                "Frank",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // No MockProvider — result-signal is sufficient.
        let tool = test_tool(state.clone());

        // limit=1 so only the top-ranked initial hit is in the base results;
        // neighborhood expansion is the only path that can surface "Frank".
        let out = tool
            .call(&json!({"action": "search", "query": "carol", "limit": 1}).to_string())
            .await
            .unwrap();

        // Guard: Frank must NOT be in the primary block.
        let split = out.find("Related context").unwrap_or(out.len());
        assert!(
            !out[..split].contains("Frank"),
            "Frank must NOT be in primary results; output:\n{out}"
        );

        assert!(
            out.contains("Frank"),
            "keyword-query neighborhood expansion should surface father_name; got:\n{out}"
        );
        assert!(
            out.contains("Related context"),
            "output should have a neighborhood header; got:\n{out}"
        );
    }

    /// Expansion works with multi-word values (the common production case where
    /// mother_name="Carol Mendez" and father="Frank Mendez").
    ///
    /// Previously, the classifier was needed because a bare keyword query yielded
    /// no entities from the LLM. Now the value of the matched relationship-key
    /// fact IS the candidate entity — no classifier call at all.
    #[tokio::test]
    async fn test_search_neighborhood_from_result_entities_when_classifier_empty() {
        let state = setup_state().await;

        state
            .upsert_fact(
                "user",
                "mother_name",
                "Carol Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "user",
                "father",
                "Frank Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // No MockProvider — the result-derived entity path is the only path.
        let tool = test_tool(state.clone());

        // limit=1 ensures only the top-ranked initial hit (mother_name) is in the
        // primary block; neighborhood expansion is the ONLY path that can surface Frank.
        let out = tool
            .call(&json!({"action": "search", "query": "carol", "limit": 1}).to_string())
            .await
            .unwrap();

        // Guard: Frank must NOT appear before the "Related context" separator.
        let split = out.find("Related context").unwrap_or(out.len());
        assert!(
            !out[..split].contains("Frank"),
            "Frank must NOT be in primary results — expansion-isolation; output:\n{out}"
        );

        assert!(
            out.contains("Frank"),
            "result-derived entity path must surface father=Frank even without a classifier; got:\n{out}"
        );
        assert!(
            out.contains("Related context"),
            "output should have a neighborhood header; got:\n{out}"
        );
    }

    #[tokio::test]
    async fn search_header_reports_honest_counts() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        let topics = [
            "gardening",
            "bicycle",
            "jazz",
            "sourdough",
            "chess",
            "hiking",
            "kettle",
            "piano",
            "camera",
            "pottery",
        ];
        for i in 0..27 {
            state
                .upsert_fact(
                    "general",
                    &format!("hobby_{i}"),
                    &format!("{} session {i}", topics[i % topics.len()]),
                    "test",
                    None,
                    FactPrivacy::Global,
                )
                .await
                .unwrap();
        }
        for i in 0..3 {
            state
                .upsert_fact(
                    "project",
                    &format!("zephyr_item_{i}"),
                    &format!("zephyr build {i}"),
                    "test",
                    None,
                    FactPrivacy::Global,
                )
                .await
                .unwrap();
        }
        let out = tool
            .call(&json!({"action": "search", "query": "zephyr"}).to_string())
            .await
            .unwrap();
        assert!(out.contains("(3 shown of"), "{out}");
        assert!(
            !out.contains("relevant of"),
            "no floor, no relevance claim: {out}"
        );
        assert!(!out.contains("of 30 matches"), "{out}");
        assert!(!out.contains("more relevant"), "{out}");
    }

    #[tokio::test]
    async fn test_search_tool_finds_port_among_substring_noise() {
        // End-to-end regression for the live recall bug: "database port" used to
        // bury `local_dev_db_port` under substring noise (report/portfolio/…) and
        // truncate it away. The hybrid search must surface it.
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        state
            .upsert_fact_with_provenance(
                "preference",
                "local_dev_db_port",
                "54329",
                "progressive",
                None,
                FactPrivacy::Global,
                Some(chrono::Utc::now()),
                Some("You told me:\nThe local database port is 54329."),
            )
            .await
            .unwrap();
        // Substring-noise facts that previously outranked/truncated the real one.
        for (i, (k, v)) in [
            ("report_output_location", "~/reports"),
            ("portfolio_site", "example.com"),
            ("important_note", "ship it"),
            ("support_email", "help@example.com"),
        ]
        .iter()
        .enumerate()
        {
            state
                .upsert_fact(
                    "project",
                    k,
                    v,
                    &format!("test{i}"),
                    None,
                    FactPrivacy::Global,
                )
                .await
                .unwrap();
        }

        let result = tool
            .call(&json!({"action": "search", "query": "database port"}).to_string())
            .await
            .unwrap();

        assert!(
            result.contains("local_dev_db_port") && result.contains("54329"),
            "search for 'database port' must surface the port fact; got:\n{result}"
        );
        assert!(
            !result.contains("report_output_location"),
            "substring noise must not appear; got:\n{result}"
        );
        // Temporal affordance: each result reports when the fact was learned so
        // the model can answer "when did I tell you X?" without thrashing.
        assert!(
            result.contains("learned:"),
            "search results should include a 'learned:' recency; got:\n{result}"
        );
        assert!(result.contains("fact_id:"), "missing fact id:\n{result}");
        assert!(
            result.contains("source: progressive"),
            "missing extraction source:\n{result}"
        );
        assert!(
            result.contains("evidence: \"You told me: The local database port is 54329.\""),
            "missing normalized source evidence:\n{result}"
        );
    }

    #[tokio::test]
    async fn test_search_episodes_recalls_past_conversation() {
        // Concrete store so we can use the inherent insert_episode helper.
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let concrete = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        let tool = test_tool(concrete.clone());
        let now = chrono::Utc::now();
        let ep = crate::traits::Episode {
            id: 0,
            session_id: "test".to_string(),
            summary: "Wrote and ran a Python primes script and a stats module with pytest"
                .to_string(),
            topics: Some(vec!["python".to_string(), "testing".to_string()]),
            emotional_tone: None,
            outcome: Some("completed".to_string()),
            importance: 0.7,
            recall_count: 0,
            last_recalled_at: None,
            message_count: 10,
            start_time: now,
            end_time: now,
            created_at: now,
            channel_id: None,
        };
        concrete.insert_episode(&ep).await.unwrap();

        // Empty query → recency browse (no embedding dependency).
        let result = tool
            .call_with(
                &json!({
                    "action": "search_episodes"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    session_id: "test".into(),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(
            result.contains("primes script") || result.contains("stats module"),
            "should recall the episode summary; got:\n{result}"
        );
        assert!(
            result.contains("topics:"),
            "should surface topics; got:\n{result}"
        );
        assert!(
            result.contains("just now") || result.contains("ago"),
            "should include a recency label; got:\n{result}"
        );
    }

    /// Trust comes only from the dispatcher's typed context: a
    /// `_trusted_session` argument does not widen internal recall to every
    /// session's episodes.
    #[tokio::test]
    async fn trusted_session_argument_bypasses_nothing() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let concrete = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        let tool = test_tool(concrete.clone());
        let now = chrono::Utc::now();
        for (session, summary) in [
            (
                "session-alice",
                "Planned the Acme Corp offsite with Alice Rivera",
            ),
            (
                "session-jordan",
                "Reviewed the synthetic budget with Jordan Lee",
            ),
        ] {
            let ep = crate::traits::Episode {
                id: 0,
                session_id: session.to_string(),
                summary: summary.to_string(),
                topics: None,
                emotional_tone: None,
                outcome: None,
                importance: 0.7,
                recall_count: 0,
                last_recalled_at: None,
                message_count: 4,
                start_time: now,
                end_time: now,
                created_at: now,
                channel_id: None,
            };
            concrete.insert_episode(&ep).await.unwrap();
        }
        let args = json!({
            "action": "search_episodes",
            "_trusted_session": true
        })
        .to_string();
        let facts = crate::traits::ToolCallContext {
            session_id: "session-alice".into(),
            channel_visibility: Some(ChannelVisibility::Internal),
            ..Default::default()
        };
        let result = tool
            .call_with_execution_context(&args, None, facts.clone())
            .await
            .unwrap()
            .output;
        assert!(result.contains("Acme Corp offsite"), "{result}");
        assert!(
            !result.contains("Jordan Lee"),
            "an argument does not widen recall to other sessions: {result}"
        );
        // The dispatcher's typed trust does.
        let ctx = crate::traits::ToolCallContext {
            trusted_session: true,
            ..facts
        };
        let trusted = tool
            .call_with_execution_context(&args, None, ctx)
            .await
            .unwrap()
            .output;
        assert!(trusted.contains("Jordan Lee"), "{trusted}");
    }

    fn test_tool(state: Arc<SqliteStateStore>) -> ManageMemoriesTool {
        let store = state.memory_stores();
        ManageMemoriesTool::new(state, store)
    }

    async fn setup_state() -> Arc<SqliteStateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        state
    }

    async fn add_schedule(state: &Arc<SqliteStateStore>, goal_id: &str, cron_expr: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run(cron_expr)
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            cron_expr: cron_expr.to_string(),
            tz: "local".to_string(),
            original_schedule: Some(cron_expr.to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();
    }

    #[tokio::test]
    async fn add_schedule_creates_schedule_row() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let goal = Goal::new_continuous(
            "English research for pronunciation",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "add_schedule",
                    "goal_id": goal_id,
                    "schedule": "daily at 9am"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Added schedule"));

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 1);
        assert_eq!(schedules[0].cron_expr, "0 9 * * *");
        assert!(!schedules[0].is_paused);
    }

    #[tokio::test]
    async fn trigger_now_uses_run_aware_path_and_preserves_goal_context() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let mut goal = Goal::new_continuous(
            "Publish one researched daily insight",
            "user-session",
            Some(1000),
            Some(5000),
        );
        goal.context = Some(
            json!({
                "instructions": "Search a current authoritative source before drafting."
            })
            .to_string(),
        );
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 9 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "trigger_now",
                    "goal_id": goal.id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Queued manual run"));
        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].context, goal.context);

        let run = state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("compatibility trigger creates an open goal run");
        assert_eq!(run.trigger_type, "manual");
        assert_eq!(run.status, "pending");
        assert_eq!(run.root_task_id.as_deref(), Some(tasks[0].id.as_str()));
    }

    #[tokio::test]
    async fn add_schedule_rejects_personal_goal() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let mut goal = Goal::new_finite("Personal goal", "user-session");
        goal.domain = "personal".to_string();
        state.create_goal(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "add_schedule",
                    "goal_id": goal.id,
                    "schedule": "daily at 9am"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Personal goals cannot be scheduled"));
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(schedules.is_empty());
    }

    #[tokio::test]
    async fn cancel_scheduled_blocks_protected_system_maintenance_goal() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let goal = Goal::new_continuous(
            "Maintain memory health: prune old events, clean up retention, remove stale data",
            "system",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "30 3 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Cannot cancel protected system maintenance goal"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
    }

    #[tokio::test]
    async fn cancel_scheduled_allows_non_system_goal() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let goal = Goal::new_continuous(
            "English research for pronunciation",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 5,12,19 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Cancelled scheduled goal"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "cancelled");
    }

    #[tokio::test]
    async fn cancel_scheduled_all_keyword_cancels_cancellable_goals() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let g1 = Goal::new_continuous("Recurring A", "user-session", Some(1000), Some(5000));
        let g2 = Goal::new_continuous("Recurring B", "user-session", Some(1000), Some(5000));
        let protected = Goal::new_continuous(
            "Maintain memory health: prune old events, clean up retention, remove stale data",
            "system",
            Some(1000),
            Some(5000),
        );
        state.create_goal(&g1).await.unwrap();
        state.create_goal(&g2).await.unwrap();
        state.create_goal(&protected).await.unwrap();
        add_schedule(&state, &g1.id, "0 6 * * *").await;
        add_schedule(&state, &g2.id, "0 12 * * *").await;
        add_schedule(&state, &protected.id, "0 3 * * *").await;

        let result = tool
            .call_with(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id": "all"
                })
                .to_string(),
                crate::testing::call_facts("user-session"),
            )
            .await
            .unwrap();

        assert!(result.contains("Cancelled 2 scheduled goals."));
        let g1_after = state.get_goal(&g1.id).await.unwrap().unwrap();
        let g2_after = state.get_goal(&g2.id).await.unwrap().unwrap();
        let protected_after = state.get_goal(&protected.id).await.unwrap().unwrap();
        assert_eq!(g1_after.status, "cancelled");
        assert_eq!(g2_after.status, "cancelled");
        assert_eq!(protected_after.status, "active");
        assert_eq!(state.get_schedules_for_goal(&g1.id).await.unwrap().len(), 0);
        assert_eq!(state.get_schedules_for_goal(&g2.id).await.unwrap().len(), 0);
        assert_eq!(
            state
                .get_schedules_for_goal(&protected.id)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn cancel_scheduled_all_keyword_scoped_to_session() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let same_session = Goal::new_continuous(
            "Recurring Same Session",
            "session-a",
            Some(1000),
            Some(5000),
        );
        let other_session = Goal::new_continuous(
            "Recurring Other Session",
            "session-b",
            Some(1000),
            Some(5000),
        );
        state.create_goal(&same_session).await.unwrap();
        state.create_goal(&other_session).await.unwrap();
        add_schedule(&state, &same_session.id, "0 6 * * *").await;
        add_schedule(&state, &other_session.id, "0 12 * * *").await;

        let result = tool
            .call_with(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id": "all"
                })
                .to_string(),
                crate::testing::call_facts("session-a"),
            )
            .await
            .unwrap();

        assert!(result.contains("Cancelled 1 scheduled goals."));
        let same_after = state.get_goal(&same_session.id).await.unwrap().unwrap();
        let other_after = state.get_goal(&other_session.id).await.unwrap().unwrap();
        assert_eq!(same_after.status, "cancelled");
        assert_eq!(other_after.status, "active");
        assert_eq!(
            state
                .get_schedules_for_goal(&same_session.id)
                .await
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            state
                .get_schedules_for_goal(&other_session.id)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn retry_scheduled_reactivates_failed_goal() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let mut goal = Goal::new_continuous(
            "Retryable scheduled goal",
            "user-session",
            Some(2000),
            Some(20000),
        );
        goal.status = "failed".to_string();
        goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 */6 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "retry_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Retried scheduled goal"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
        assert!(fetched.completed_at.is_none());
    }

    #[tokio::test]
    async fn retry_scheduled_reactivates_stalled_goal_and_resets_breaker() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let mut goal = Goal::new_continuous(
            "Stalled synthetic blog goal",
            "user-session",
            Some(2000),
            Some(20000),
        );
        goal.status = "stalled".to_string();
        goal.dispatch_failures = 3;
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 12 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "retry_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Retried scheduled goal"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
        assert_eq!(fetched.dispatch_failures, 0);
    }

    #[tokio::test]
    async fn retry_scheduled_rejects_non_failed_goal() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let goal = Goal::new_continuous(
            "Already active scheduled goal",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 */6 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "retry_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Only failed or stalled scheduled goals can be retried"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
    }

    #[tokio::test]
    async fn retry_failed_scheduled_reactivates_all_failed() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let mut failed_one = Goal::new_continuous(
            "Maintain knowledge base: process embeddings, consolidate memories, decay old facts",
            "system",
            Some(1000),
            Some(5000),
        );
        failed_one.status = "failed".to_string();
        let failed_one_id = failed_one.id.clone();
        state.create_goal(&failed_one).await.unwrap();
        add_schedule(&state, &failed_one.id, "0 */6 * * *").await;

        let mut failed_two = Goal::new_continuous(
            "English pronunciation slot A",
            "user-session",
            Some(1000),
            Some(5000),
        );
        failed_two.status = "failed".to_string();
        let failed_two_id = failed_two.id.clone();
        state.create_goal(&failed_two).await.unwrap();
        add_schedule(&state, &failed_two.id, "0 5 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "retry_failed_scheduled"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("retried 2"));
        let fetched_one = state.get_goal(&failed_one_id).await.unwrap().unwrap();
        let fetched_two = state.get_goal(&failed_two_id).await.unwrap().unwrap();
        assert_eq!(fetched_one.status, "active");
        assert_eq!(fetched_two.status, "active");
    }

    #[tokio::test]
    async fn cancel_scheduled_matching_cancels_matching_goals() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let g1 = Goal::new_continuous(
            "English pronunciation slot A",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let g2 = Goal::new_continuous(
            "English pronunciation slot B",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let g3 = Goal::new_continuous(
            "Unrelated recurring task",
            "user-session",
            Some(2000),
            Some(20000),
        );
        state.create_goal(&g1).await.unwrap();
        state.create_goal(&g2).await.unwrap();
        state.create_goal(&g3).await.unwrap();
        add_schedule(&state, &g1.id, "0 5 * * *").await;
        add_schedule(&state, &g2.id, "0 12 * * *").await;
        add_schedule(&state, &g3.id, "0 19 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "cancel_scheduled_matching",
                    "query": "english pronunciation"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("cancelled 2"));
        let g1_after = state.get_goal(&g1.id).await.unwrap().unwrap();
        let g2_after = state.get_goal(&g2.id).await.unwrap().unwrap();
        let g3_after = state.get_goal(&g3.id).await.unwrap().unwrap();
        assert_eq!(g1_after.status, "cancelled");
        assert_eq!(g2_after.status, "cancelled");
        assert_eq!(g3_after.status, "active");

        // Schedules for cancelled goals should be purged.
        assert_eq!(state.get_schedules_for_goal(&g1.id).await.unwrap().len(), 0);
        assert_eq!(state.get_schedules_for_goal(&g2.id).await.unwrap().len(), 0);
        assert_eq!(state.get_schedules_for_goal(&g3.id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn diagnose_scheduled_reports_latest_failed_task() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let mut goal = Goal::new_continuous("Diagnose me", "user-session", Some(2000), Some(20000));
        goal.status = "failed".to_string();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 */6 * * *").await;

        let task = Task {
            id: "diag-task-1".to_string(),
            goal_id: goal.id.clone(),
            description: "Run maintenance".to_string(),
            status: "failed".to_string(),
            priority: "medium".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: Some("Connection timeout to embedding backend".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 1,
            max_retries: 3,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
        };
        state.create_task(&task).await.unwrap();

        let activity = TaskActivity {
            id: 0,
            task_id: task.id.clone(),
            activity_type: "tool_result".to_string(),
            tool_name: Some("http_request".to_string()),
            tool_args: Some("{\"url\":\"https://example.com\"}".to_string()),
            result: Some("timeout while contacting backend".to_string()),
            success: Some(false),
            tokens_used: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        state.log_task_activity(&activity).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "diagnose_scheduled",
                    "goal_id": goal.id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Scheduled Goal Diagnosis"));
        assert!(result.contains("Latest Failed Task"));
        assert!(result.contains("Connection timeout to embedding backend"));
        assert!(result.contains("Recent Activity"));
    }

    // ==================== create_personal_goal tests ====================

    #[tokio::test]
    async fn create_personal_goal_creates_goal_in_registry() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Learn conversational Spanish"
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        assert!(result.contains("Created personal goal"));
        assert!(result.contains("Learn conversational Spanish"));

        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].description, "Learn conversational Spanish");
        assert_eq!(goals[0].domain, "personal");
        assert_eq!(goals[0].priority, "medium");
        assert_eq!(goals[0].status, "active");
        assert_eq!(goals[0].session_id, "owner-session");
    }

    #[tokio::test]
    async fn created_goal_with_rate_limit_language_reports_typed_success() {
        let state = setup_state().await;
        let tool = test_tool(state);

        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Steward a synthetic account under owner-confirmed rate limits"
                })
                .to_string(),
                None,
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        assert!(outcome.output.starts_with("Created personal goal "));
    }

    #[tokio::test]
    async fn list_goals_marks_only_same_session_sources_as_mandate_compatible() {
        let state = setup_state().await;
        let compatible = crate::traits::Goal::new_personal("Compatible goal", "owner-session");
        let foreign = crate::traits::Goal::new_personal("Foreign goal", "other-session");
        state.create_goal(&foreign).await.unwrap();
        state.create_goal(&compatible).await.unwrap();
        let tool = test_tool(state);

        let output = tool
            .call_with(
                &json!({
                    "action": "list_goals",
                    "limit": 10
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        let compatible_line = output
            .lines()
            .find(|line| line.contains("Compatible goal"))
            .unwrap();
        let foreign_line = output
            .lines()
            .find(|line| line.contains("Foreign goal"))
            .unwrap();
        assert!(compatible_line.contains("mandate-source compatible"));
        assert!(foreign_line.contains("not compatible"));
        assert!(output.find("Compatible goal") < output.find("Foreign goal"));
    }

    #[tokio::test]
    async fn create_personal_goal_with_priority() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Exercise daily",
                    "priority": "high"
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        assert!(result.contains("Created personal goal"));
        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert_eq!(goals[0].priority, "high");
    }

    #[tokio::test]
    async fn create_personal_goal_rejects_invalid_priority() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Some goal",
                    "priority": "ultra"
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        assert!(result.contains("Invalid priority"));
        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_personal_goal_rejects_empty_description() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "   "
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await;

        assert!(result.is_err() || result.unwrap().contains("required"));
    }

    #[tokio::test]
    async fn create_personal_goal_rejects_missing_internal_session() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Cannot become unowned"
                })
                .to_string(),
            )
            .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("create_personal_goal: the session fact is missing from the call context"));
        assert!(state
            .get_active_personal_goals(50)
            .await
            .unwrap()
            .is_empty());
    }

    // ==================== create_scheduled_goal tests ====================

    #[tokio::test]
    async fn create_scheduled_goal_success_recurring() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Check API health",
                    "schedule": "every 6h"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("Confirmed and activated scheduled goal"));
        assert!(result.contains("1 schedule(s)"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Check API health")
            .unwrap();
        assert_eq!(goal.status, "active");
        assert_eq!(goal.domain, "orchestration");
        assert_eq!(goal.goal_type, "continuous");
        assert_eq!(goal.budget_per_check, Some(400_000));
        assert_eq!(goal.budget_daily, Some(1_000_000));

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 1);
        assert!(!schedules[0].is_one_shot);
        assert_eq!(schedules[0].cron_expr, "0 */6 * * *");
    }

    #[tokio::test]
    async fn activate_scheduled_is_typed_owner_and_session_scoped() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        // A goal still pending confirmation (one created before the kernel
        // confirmed goals up front).
        let mut goal =
            Goal::new_continuous_pending("Review operational health", "owner-session", None, None);
        goal.status = "pending_confirmation".to_string();
        state.create_goal(&goal).await.unwrap();

        let wrong_session = tool
            .call_with(
                &json!({
                    "action": "activate_scheduled",
                    "goal_id": goal.id
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "different-session".into(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(wrong_session.contains("different owner session"));
        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().status,
            "pending_confirmation"
        );

        let activated = tool
            .call_with(
                &json!({
                    "action": "activate_scheduled",
                    "goal_id": goal.id
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "owner-session".into(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(activated.contains("Activated scheduled goal"));
        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
    }

    #[tokio::test]
    async fn activate_scheduled_rejects_non_owner() {
        let state = setup_state().await;
        let tool = test_tool(state);
        let result = tool
            .call_with(
                &json!({
                    "action": "activate_scheduled",
                    "goal_id": "goal-id"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Guest),
                    session_id: "owner-session".into(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(result.contains("Only the owner"));
    }

    /// Outside the kernel (no `kernel_approved`) a scheduled goal that runs
    /// the agent is refused and nothing is stored; a reminder still works.
    #[tokio::test]
    async fn create_scheduled_goal_refuses_without_kernel_approval() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        let error = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Check confirmation transport",
                    "schedule": "every 6h"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    ..Default::default()
                },
            )
            .await
            .expect_err("refused outside the kernel");
        assert!(
            error
                .to_string()
                .contains("requires approval through the agent kernel"),
            "{error}"
        );
        assert!(state
            .get_scheduled_goals()
            .await
            .unwrap()
            .iter()
            .all(|goal| goal.description != "Check confirmation transport"));
    }

    /// The owner is asked exactly where the call would create a goal that
    /// runs the agent: not for a reminder, a non-owner, an internal run, a
    /// bad schedule or a duplicate. The card binds the exact goal.
    #[tokio::test]
    async fn create_scheduled_goal_judges_its_own_approval() {
        use crate::traits::OwnerApprovalNeed;
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        let owner = crate::traits::ToolCallContext {
            user_role: Some(crate::types::UserRole::Owner),
            session_id: "test-session".into(),
            channel_visibility: Some(ChannelVisibility::Private),
            ..Default::default()
        };
        let args = json!({
            "action": "create_scheduled_goal",
            "goal": "Check API health",
            "schedule": "every 6h"
        });
        let OwnerApprovalNeed::Ask(request) = tool.owner_approval(&args, &owner).await else {
            panic!("a scheduled goal asks");
        };
        assert_eq!(request.summary, "Check API health");
        assert_eq!(request.risk, aidaemon_core::ApprovalRisk::Medium);
        // The same five-line card as a watching automation, then the next run.
        let card =
            aidaemon_core::mandate_card::MandateConfirmationCard::from_lines(&request.warnings)
                .expect("all five lines");
        assert_eq!(card.what, "Check API health");
        assert_eq!(card.how_often, "Every 6 hours");
        assert_eq!(
            card.can_do,
            "Anything you can do – it runs with your full access"
        );
        assert_eq!(card.ends, "Only when you stop it");
        assert!(card.details.iter().any(|w| w.starts_with("Next: ")));
        let mut injected = args.clone();
        injected["_session_id"] = json!("synthetic-session");
        let OwnerApprovalNeed::Ask(same) = tool.owner_approval(&injected, &owner).await else {
            panic!("a scheduled goal asks");
        };
        assert_eq!(request.subject, same.subject);
        let OwnerApprovalNeed::Ask(other) = tool
            .owner_approval(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Check API health",
                    "schedule": "every 2h"
                }),
                &owner,
            )
            .await
        else {
            panic!("a scheduled goal asks");
        };
        assert_ne!(request.subject, other.subject);

        async fn judged(
            tool: &ManageMemoriesTool,
            args: Value,
            facts: crate::traits::ToolCallContext,
        ) -> OwnerApprovalNeed {
            tool.owner_approval(&args, &facts).await
        }
        let mut reminder = args.clone();
        reminder["reminder"] = json!("Check the oven.");
        assert_eq!(
            judged(&tool, reminder, owner.clone()).await,
            OwnerApprovalNeed::NotNeeded
        );
        assert_eq!(
            judged(
                &tool,
                args.clone(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Guest),
                    ..owner.clone()
                }
            )
            .await,
            OwnerApprovalNeed::NotNeeded
        );
        assert_eq!(
            judged(
                &tool,
                args.clone(),
                crate::traits::ToolCallContext {
                    channel_visibility: Some(ChannelVisibility::Internal),
                    ..owner.clone()
                }
            )
            .await,
            OwnerApprovalNeed::NotNeeded
        );
        let mut bad = args.clone();
        bad["schedule"] = json!("whenever the moon is full");
        assert_eq!(
            judged(&tool, bad, owner.clone()).await,
            OwnerApprovalNeed::NotNeeded
        );
        assert_eq!(
            judged(&tool, json!({"action": "list"}), owner.clone()).await,
            OwnerApprovalNeed::NotNeeded
        );

        // Once the approved goal exists, the identical request is a duplicate
        // the call answers without creating anything: no card.
        let created = tool
            .call_with(
                &args.to_string(),
                crate::traits::ToolCallContext {
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..owner.clone()
                },
            )
            .await
            .unwrap();
        assert!(created.contains("Confirmed and activated"), "{created}");
        assert_eq!(
            judged(&tool, args, owner).await,
            OwnerApprovalNeed::NotNeeded
        );
    }

    #[tokio::test]
    async fn create_scheduled_goal_success_one_shot() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Deploy release",
                    "schedule": "in 2h"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("activated scheduled goal"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Deploy release")
            .unwrap();
        assert_eq!(goal.goal_type, "finite");

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 1);
        assert!(schedules[0].is_one_shot);
    }

    /// Strict-schema providers send every field, so an unused `schedules`
    /// arrives as `[]` next to the real `schedule`.
    #[tokio::test]
    async fn create_scheduled_goal_uses_schedule_when_schedules_is_empty() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Check the oven",
                    "schedule": "in 2h",
                    "schedules": []
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(result.contains("activated scheduled goal"), "{result}");

        let missing = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Check the oven",
                    "schedule": "",
                    "schedules": []
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await;
        assert!(missing.is_err(), "a missing schedule is a failed call");
    }

    /// A typed reminder runs no agent and only messages the owner, so it is
    /// active at once (no confirmation prompt) and carries its message.
    #[tokio::test]
    async fn create_scheduled_goal_with_reminder_message_is_active_at_once() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Remind Alice Rivera to check the oven",
                    "schedule": "in 2h",
                    "reminder": "Check the oven."
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(result.contains("Reminder set"), "{result}");

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description.contains("oven"))
            .unwrap();
        assert_eq!(goal.status, "active");
        assert_eq!(
            crate::reminders::message_from_context(goal.context.as_deref()).as_deref(),
            Some("Check the oven.")
        );
    }

    /// Cancelling a goal's only schedule cancels the goal: an active goal
    /// with no schedule would linger in listings and never run.
    #[tokio::test]
    async fn cancelling_the_last_schedule_cancels_the_goal() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        tool.call_with(
            &json!({
                "action": "create_scheduled_goal",
                "goal": "Review the inbox",
                "schedule": "every weekday at 9am",
                "reminder": "Review your inbox."
            })
            .to_string(),
            crate::traits::ToolCallContext {
                user_role: Some(crate::types::UserRole::Owner),
                session_id: "test-session".into(),
                kernel_approved: Some(crate::traits::ApprovalScope::Call),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let goal = state
            .get_scheduled_goals()
            .await
            .unwrap()
            .into_iter()
            .find(|g| g.description.contains("inbox"))
            .unwrap();
        let schedule = state.get_schedules_for_goal(&goal.id).await.unwrap()[0].clone();

        let result = tool
            .call_with(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id": goal.id,
                    "schedule_id": schedule.id
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(result.contains("cancelled"), "{result}");
        let goal = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(goal.status, "cancelled");
    }

    /// An active continuous goal with no schedule never runs; the listing
    /// shows it so it can be cancelled, and cancelling by goal_id works.
    #[tokio::test]
    async fn list_scheduled_shows_and_cancels_unscheduled_active_goals() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        let mut orphan = crate::traits::Goal::new_continuous(
            "Post one synthetic update for Acme Corp",
            "telegram:synthetic-user-1",
            None,
            None,
        );
        orphan.status = "active".to_string();
        state.create_goal(&orphan).await.unwrap();

        let listing = tool
            .call_with(
                &json!({"action": "list_scheduled"}).to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(listing.contains("Active without a schedule"), "{listing}");
        assert!(listing.contains(&orphan.id), "{listing}");

        tool.call_with(
            &json!({
                "action": "cancel_scheduled",
                "goal_id": orphan.id
            })
            .to_string(),
            crate::traits::ToolCallContext {
                user_role: Some(crate::types::UserRole::Owner),
                session_id: "test-session".into(),
                kernel_approved: Some(crate::traits::ApprovalScope::Call),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            state.get_goal(&orphan.id).await.unwrap().unwrap().status,
            "cancelled"
        );
        let listing = tool
            .call_with(
                &json!({"action": "list_scheduled"}).to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(!listing.contains(&orphan.id), "{listing}");
    }

    #[tokio::test]
    async fn create_scheduled_goal_cleans_filler_and_schedule_from_goal_text() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "in 2 hours remind me to send release notes",
                    "schedule": "in 2 hours"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("activated scheduled goal"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Send release notes")
            .unwrap();
        assert_eq!(goal.goal_type, "finite");
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_non_owner() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Steal data",
                    "schedule": "every 1h"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Guest),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("Only owners"));
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_missing_session_id() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Do something",
                    "schedule": "daily at 9am"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("the session fact is missing from the call context"));
        assert!(!result.contains("_session_id"), "{result}");
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_bad_schedule() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Do something",
                    "schedule": "whenever the moon is full"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("Couldn't parse schedule"));
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_missing_schedule() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Do something"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await;

        assert!(result.is_err() || result.unwrap().contains("required"));
    }

    #[tokio::test]
    async fn create_scheduled_goal_multiple_schedules() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Multi-slot monitoring",
                    "schedules": ["daily at 8am", "daily at 2pm", "daily at 9pm"]
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("activated scheduled goal"));
        assert!(result.contains("3 schedule(s)"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Multi-slot monitoring")
            .unwrap();
        assert_eq!(goal.goal_type, "continuous");

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 3);
        let crons: Vec<&str> = schedules.iter().map(|s| s.cron_expr.as_str()).collect();
        assert!(crons.contains(&"0 8 * * *"));
        assert!(crons.contains(&"0 14 * * *"));
        assert!(crons.contains(&"0 21 * * *"));
    }

    #[tokio::test]
    async fn create_scheduled_goal_starts_paused() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Paused from start",
                    "schedule": "daily at 9am",
                    "is_paused": true
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("activated scheduled goal"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Paused from start")
            .unwrap();
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(schedules[0].is_paused);
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_internal_execution_context() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Nested schedule",
                    "schedule": "in 1 minute"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "sub-1-abc123".into(),
                    channel_visibility: Some(crate::types::ChannelVisibility::Internal),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains(
            "Cannot create scheduled goals from within internal scheduled-task execution"
        ));
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_scheduled_goal_deduplicates_same_description_and_schedule() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let first = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Check API health",
                    "schedule": "every 6h"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(first.contains("activated scheduled goal"));

        let second = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "  check   api HEALTH ",
                    "schedule": "every 6h"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(second.contains("A similar scheduled goal already exists"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let matching = goals
            .iter()
            .filter(|g| g.description == "Check API health")
            .count();
        assert_eq!(matching, 1);
    }

    #[tokio::test]
    async fn create_scheduled_goal_bad_schedule_in_multi_does_not_create_goal() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Partial failure",
                    "schedules": ["daily at 8am", "whenever I feel like it"]
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    session_id: "test-session".into(),
                    kernel_approved: Some(crate::traits::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("Couldn't parse schedule"));
        // No goal should be created — fail-fast before DB writes
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(
            goals.iter().all(|g| g.description != "Partial failure"),
            "No goal should be created when any schedule fails to parse"
        );
    }

    /// Regression: an empty / whitespace `key` to `forget` previously matched
    /// every fact via `fk.contains("")` and wiped all active facts across all
    /// categories. The action must refuse to run instead.
    #[tokio::test]
    async fn forget_rejects_empty_or_whitespace_key() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        for (cat, key, val) in [
            ("preferences", "favorite_color", "blue"),
            ("preferences", "favorite_food", "tacos"),
            ("personal", "birthday", "1990-01-01"),
        ] {
            state
                .upsert_fact(cat, key, val, "test", None, FactPrivacy::Global)
                .await
                .unwrap();
        }

        for blank in ["", "   ", "\t", "\n"] {
            let err = tool
                .call(
                    &json!({
                        "action": "forget",
                        "category": "preferences",
                        "key": blank,
                    })
                    .to_string(),
                )
                .await
                .expect_err("blank key must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains("empty") || msg.contains("whitespace"),
                "expected refusal message, got {msg:?}"
            );
        }

        // All facts should still be present.
        let remaining = state.get_facts(None).await.unwrap();
        let active = remaining
            .iter()
            .filter(|f| f.superseded_at.is_none())
            .count();
        assert_eq!(active, 3, "no facts should have been deleted");
    }

    /// Regression: single-character keys are also too broad to use for substring
    /// matching across the fact store.
    #[tokio::test]
    async fn forget_rejects_single_character_key() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        state
            .upsert_fact(
                "preferences",
                "favorite_color",
                "blue",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let err = tool
            .call(
                &json!({
                    "action": "forget",
                    "category": "preferences",
                    "key": "a",
                })
                .to_string(),
            )
            .await
            .expect_err("single-character key must be rejected");
        assert!(err.to_string().contains("at least 2 characters"));

        let remaining = state.get_facts(None).await.unwrap();
        assert_eq!(
            remaining
                .iter()
                .filter(|f| f.superseded_at.is_none())
                .count(),
            1
        );
    }

    /// Substring matching is still allowed within the explicitly-requested
    /// category — the caller knows what they are targeting there.
    #[tokio::test]
    async fn forget_substring_matches_within_requested_category() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        state
            .upsert_fact(
                "preferences",
                "favorite_color",
                "blue",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "personal",
                "color_of_car",
                "red",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "forget",
                    "category": "preferences",
                    "key": "color",
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Forgotten"), "got: {result}");

        // Only the `preferences` fact (substring match) should be gone.
        // The `personal.color_of_car` substring match must NOT be deleted
        // because cross-category sweep only allows exact/canonical matches.
        let remaining = state.get_facts(None).await.unwrap();
        let active: Vec<_> = remaining
            .iter()
            .filter(|f| f.superseded_at.is_none())
            .collect();
        assert_eq!(active.len(), 1, "got: {active:?}");
        assert_eq!(active[0].category, "personal");
        assert_eq!(active[0].key, "color_of_car");
    }

    /// Regression: forgetting `qa_pet_name` also deleted the owner's `name`
    /// fact because a fact key contained in the requested key matched, and
    /// partial matching ran even when an exact key existed.
    #[tokio::test]
    async fn forget_never_deletes_keys_contained_in_the_requested_key() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        for (key, value) in [
            ("qa_pet_name", "SYNTH-FINCH"),
            ("name", "Alice Rivera"),
            ("pet_name_history", "Rex"),
        ] {
            state
                .upsert_fact("user", key, value, "test", None, FactPrivacy::Global)
                .await
                .unwrap();
        }

        let result = tool
            .call(
                &json!({"action": "forget", "category": "user", "key": "qa_pet_name"}).to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Forgotten 1 fact"), "got: {result}");

        let active: Vec<_> = state
            .get_facts(None)
            .await
            .unwrap()
            .into_iter()
            .filter(|f| f.superseded_at.is_none())
            .map(|f| f.key)
            .collect();
        assert!(active.contains(&"name".to_string()), "got: {active:?}");
        assert!(active.contains(&"pet_name_history".to_string()));
        assert!(!active.contains(&"qa_pet_name".to_string()));
    }

    #[tokio::test]
    async fn forget_partial_key_matches_whole_segments_only() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        for key in ["favorite_color", "colorado_trip"] {
            state
                .upsert_fact("preferences", key, "x", "test", None, FactPrivacy::Global)
                .await
                .unwrap();
        }
        let result = tool
            .call(
                &json!({"action": "forget", "category": "preferences", "key": "color"}).to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Forgotten 1 fact"), "got: {result}");
        assert!(result.contains("favorite_color"), "got: {result}");
    }

    /// Cross-category sweep should still catch exact-key duplicates (the
    /// original "in case duplicates exist elsewhere" intent), but only via
    /// exact or canonical match — never substring.
    #[tokio::test]
    async fn forget_cross_category_only_exact_or_canonical() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        // Exact duplicate across categories should be cleaned up.
        state
            .upsert_fact(
                "preferences",
                "favorite_color",
                "blue",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "personal",
                "favorite_color",
                "blue",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        // An unrelated fact whose key contains "color" — must NOT be deleted.
        state
            .upsert_fact(
                "personal",
                "color_of_car",
                "red",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "forget",
                    "category": "preferences",
                    "key": "favorite_color",
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Forgotten 2 fact"), "got: {result}");

        let remaining = state.get_facts(None).await.unwrap();
        let active: Vec<_> = remaining
            .iter()
            .filter(|f| f.superseded_at.is_none())
            .collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].key, "color_of_car");
    }
}
