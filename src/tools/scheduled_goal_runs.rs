//! `scheduled_goal_runs` lives in `aidaemon_work::tools::scheduled_goal_runs`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_work::tools::scheduled_goal_runs::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::testing::CallWithFacts;
    use crate::traits::StateStore;
    use crate::traits::Tool;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolResultPresentation;
    use crate::traits::ToolSemantics;
    use crate::traits::{Goal, GoalSchedule, Task};
    use serde_json::Value;
    use serde_json::json;
    use std::sync::Arc;

    async fn setup_state() -> Arc<dyn StateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        state as Arc<dyn StateStore>
    }

    #[tokio::test]
    async fn schema_supports_id_free_overview_but_targeted_actions_require_goal_id() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state);
        let schema = tool.schema();
        let required = schema
            .get("parameters")
            .and_then(|p| p.get("required"))
            .and_then(|r| r.as_array())
            .expect("required array exists");
        let required_values: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required_values.contains(&"action"));
        assert!(!required_values.contains(&"goal_id"));
        assert!(schema["parameters"]["properties"]["include_diagnostics"].is_object());
        assert!(tool.validate_arguments(r#"{"action":"overview"}"#).is_ok());
        assert!(
            tool.validate_arguments(r#"{"action":"run_history"}"#)
                .is_err()
        );
    }

    #[tokio::test]
    async fn overview_returns_schedule_and_latest_run_without_internal_ids() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous(
            "Publish a synthetic weekly report",
            "synthetic-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: Some(now.clone()),
            next_run_at: now.clone(),
            created_at: now.clone(),
            updated_at: now,
        };
        let schedule_id = schedule.id.clone();
        state.create_goal_schedule(&schedule).await.unwrap();
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.clone(),
            description: "Synthetic scheduled run".to_string(),
            status: "blocked".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: Some("Synthetic blocker".to_string()),
            idempotent: true,
            retry_count: 0,
            max_retries: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        let task_id = task.id.clone();
        state.create_task(&task).await.unwrap();

        let outcome = tool
            .call_with_status_outcome(r#"{"action":"overview"}"#, None)
            .await
            .unwrap();
        let output = outcome.output;
        let snapshot: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(snapshot["snapshot"], "scheduled_objectives");
        assert_eq!(snapshot["complete"], true);
        assert_eq!(snapshot["objectives"][0]["schedule_state"], "active");
        assert_eq!(snapshot["objectives"][0]["latest_run"]["status"], "blocked");
        // A goal without a mandate answers the control question with the
        // literal "absent" — a populated object here reads as control being
        // present (observed in R47), so absence must not have a shape.
        assert_eq!(snapshot["objectives"][0]["objective_control"], "absent");
        assert_eq!(snapshot["objectives"][0]["objective_measurement_count"], 0);
        let objective_id = crate::tools::objective_status::objective_resource_id(&goal_id);
        assert_eq!(snapshot["objectives"][0]["objective_id"], objective_id);
        let observation = outcome
            .metadata
            .observations
            .iter()
            .find(|observation| observation.subject.value == objective_id)
            .expect("exact objective receipt assertion");
        assert!(
            observation
                .facets
                .contains(&crate::traits::ToolSemanticFacet::Schedule)
        );
        assert!(
            observation
                .facets
                .contains(&crate::traits::ToolSemanticFacet::Control)
        );
        let collection = outcome
            .metadata
            .collection_observations
            .iter()
            .find(|collection| {
                collection.collection.value
                    == crate::tools::objective_status::ObjectiveCollection::ScheduledGoals
                        .resource_id()
            })
            .expect("scheduled collection coverage");
        assert_eq!(
            collection.completeness,
            crate::traits::ToolCollectionCompleteness::Complete
        );
        assert!(
            collection
                .members
                .iter()
                .any(|member| member.value == objective_id)
        );
        assert!(!output.contains(&goal_id));
        assert!(!output.contains(&schedule_id));
        assert!(!output.contains(&task_id));
    }

    #[tokio::test]
    async fn overview_projects_typed_objective_control_with_measurements() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous(
            "Publish a synthetic weekly report",
            "synthetic-session",
            None,
            None,
        );
        let authority = crate::traits::MandateAuthority::from_operation_scopes(
            true,
            serde_json::from_value(serde_json::json!([
                {
                    "tool": "http_request",
                    "operation": "POST",
                    "kind": "mutation",
                    "target_prefixes": [
                        "https://api.x.com/2/tweets",
                        "auth_profile:twitter",
                        "account:12345"
                    ],
                    "mutation_effects": ["remote_mutation", "external_delivery"]
                }
            ]))
            .unwrap(),
            1,
            4,
            900,
        );
        let mut mandate = crate::traits::Mandate::new(
            &goal.id,
            None,
            "Publish a synthetic weekly report",
            "synthetic-session",
            authority,
            3_600,
            21_600,
            10_800,
        );
        mandate.objective_control = Some(crate::traits::MandateObjectiveControl {
            schema_version: crate::traits::MandateObjectiveControl::SCHEMA_VERSION,
            metric_name: "weekly_visits".to_string(),
            unit: "visits".to_string(),
            baseline_micros: 0,
            target_micros: 1_000_000,
            direction: crate::traits::ObjectiveMetricDirection::AtLeast,
            measurement_source: "metric_source:synthetic-analytics".to_string(),
            measurement_cadence_secs: 7_200,
            baseline_observed_at: chrono::Utc::now().to_rfc3339(),
        });
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: now.clone(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();
        let output = tool.call(r#"{"action":"overview"}"#).await.unwrap();
        let snapshot: Value = serde_json::from_str(&output).unwrap();
        let control = &snapshot["objectives"][0]["objective_control"];
        assert_eq!(control["mandate_status"], "active", "{output}");
        assert_eq!(control["autonomy_mode"], "bounded");
        assert_eq!(control["metric"], "weekly_visits");
        assert_eq!(control["measurement_count"], 0);
        assert_eq!(
            control["delegated_identities"],
            serde_json::json!(["auth_profile:twitter", "account:12345"])
        );
        assert_eq!(
            snapshot["objectives"][0]["objective_measurement_count"],
            control["measurement_count"]
        );
    }

    #[tokio::test]
    async fn goal_id_accepts_a_distinctive_objective_phrase() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous(
            "Each day, independently manage the synthetic blog at https://blog.example.test/",
            "user-session",
            None,
            None,
        );
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("0 6 * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: now.clone(),
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();
        assert_eq!(
            tool.resolve_goal_id("manage the synthetic blog")
                .await
                .unwrap(),
            goal.id
        );
        assert_eq!(tool.resolve_goal_id(&goal.id[..8]).await.unwrap(), goal.id);
        assert!(tool.resolve_goal_id("blog").await.is_err());
        assert!(
            tool.resolve_goal_id("no such objective anywhere")
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn run_now_creates_pending_task() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let mut goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        goal.context = Some(
            serde_json::json!({
                "instructions": "Search a current authoritative source before drafting."
            })
            .to_string(),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "run_now",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Queued manual run"));

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status, "pending");
        assert!(!tasks[0].idempotent);
        assert_eq!(tasks[0].max_retries, 0);
        assert!(tasks[0].description.starts_with("Manual scheduled run:"));
        assert_eq!(tasks[0].context, goal.context);

        let run = state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("manual trigger creates an open goal run");
        assert_eq!(run.trigger_type, "manual");
        assert_eq!(run.status, "pending");
        assert_eq!(run.root_task_id.as_deref(), Some(tasks[0].id.as_str()));
        assert_eq!(
            state.get_tasks_for_goal_run(&run.id).await.unwrap()[0].id,
            tasks[0].id
        );
    }

    #[tokio::test]
    async fn run_now_durably_extends_an_exhausted_daily_budget_once() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let mut goal = Goal::new_continuous(
            "Publish a synthetic daily entry",
            "synthetic-session",
            Some(400_000),
            Some(1_000_000),
        );
        goal.tokens_used_today = 1_253_197;
        goal.tokens_used_day = chrono::Utc::now().date_naive().to_string();
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: crate::cron_utils::compute_next_run("0 6 * * *")
                    .unwrap()
                    .to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(&json!({"action": "run_now", "goal_id": goal.id}).to_string())
            .await
            .unwrap();

        assert!(result.contains("Queued manual run"), "result: {result}");
        assert!(result.contains("Same-day adaptive capacity"));
        let durable = crate::goal_tokens::load_goal_daily_budget_override(
            state.as_ref(),
            &goal.id,
            1_000_000,
            crate::agent::SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
        )
        .await
        .expect("manual run should persist the adaptive budget");
        assert_eq!(durable.budget_daily, 2_000_000);
        assert_eq!(durable.extensions_count, 1);
    }

    #[tokio::test]
    async fn run_now_still_stops_at_the_bounded_daily_hard_cap() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let mut goal = Goal::new_continuous(
            "Run bounded synthetic work",
            "synthetic-session",
            Some(400_000),
            Some(1_000_000),
        );
        goal.tokens_used_today = 2_100_000;
        goal.tokens_used_day = chrono::Utc::now().date_naive().to_string();
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: crate::cron_utils::compute_next_run("0 6 * * *")
                    .unwrap()
                    .to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(&json!({"action": "run_now", "goal_id": goal.id}).to_string())
            .await
            .unwrap();

        assert!(result.contains("hard cap is exhausted"), "result: {result}");
        assert!(state.get_tasks_for_goal(&goal.id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn run_now_marks_internal_ids_for_natural_owner_presentation() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous("Run synthetic diagnostics", "session", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: crate::cron_utils::compute_next_run("0 */6 * * *")
                .unwrap()
                .to_rfc3339(),
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let outcome = tool
            .call_with_status_outcome(
                &json!({"action": "run_now", "goal_id": goal.id}).to_string(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.presentation,
            Some(ToolResultPresentation::NaturalSummary)
        );
        assert!(outcome.metadata.internal_identifiers.contains(&goal.id));
        let run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        assert!(outcome.metadata.internal_identifiers.contains(&run.id));
        assert!(!outcome.metadata.internal_identifiers.is_empty());
    }

    #[tokio::test]
    async fn explicit_run_diagnostics_select_diagnostic_presentation() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous("Run synthetic diagnostics", "session", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 */6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("every 6h".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: crate::cron_utils::compute_next_run("0 */6 * * *")
                    .unwrap()
                    .to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "action": "run_now",
                    "goal_id": goal.id,
                    "include_diagnostics": true
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.presentation,
            Some(ToolResultPresentation::DiagnosticDetail)
        );
    }

    #[tokio::test]
    async fn run_now_reactivates_stalled_goal_without_duplicate_run() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let mut goal = Goal::new_continuous(
            "Publish one diary entry",
            "user-session",
            Some(1000),
            Some(5000),
        );
        goal.status = "stalled".to_string();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: crate::cron_utils::compute_next_run("0 */6 * * *")
                .unwrap()
                .to_rfc3339(),
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: format!("Manual scheduled run: {}", goal.description),
            status: "pending".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: goal.context.clone(),
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
        let original_run = state
            .start_goal_run(&goal.id, "manual", Some(&schedule.id), Some(&task.id))
            .await
            .unwrap();
        state.create_task(&task).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "run_now",
                    "goal_id": goal.id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Resumed the existing manual run"));
        assert!(result.contains("no duplicate run was created"));
        assert!(result.contains(&original_run.id));
        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
        assert_eq!(state.get_tasks_for_goal(&goal.id).await.unwrap().len(), 1);
        assert_eq!(state.get_goal_runs(&goal.id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn update_instructions_persists_without_triggering_and_replaces_prior_update() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let mut goal = Goal::new_continuous(
            "Publish one daily insight",
            "user-session",
            Some(1000),
            Some(5000),
        );
        goal.context = Some(json!({"existing": "preserved"}).to_string());
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily at 9am".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: crate::cron_utils::compute_next_run("0 9 * * *")
                .unwrap()
                .to_rfc3339(),
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        for instructions in [
            "Search a current authoritative source before drafting.",
            "Use one current primary source and write a human-useful insight.",
        ] {
            let result = tool
                .call_with(
                    &json!({
                        "action": "update_instructions",
                        "goal_id": goal.id,
                        "instructions": instructions
                    })
                    .to_string(),
                    crate::traits::ToolCallContext {
                        user_role: Some(crate::types::UserRole::Owner),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            assert!(result.contains("No run was triggered"));
        }

        let updated = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert!(
            updated
                .description
                .contains("Use one current primary source and write a human-useful insight.")
        );
        assert!(
            !updated
                .description
                .contains("Search a current authoritative source before drafting.")
        );
        assert_eq!(
            updated
                .description
                .matches(ScheduledGoalRunsTool::RUN_INSTRUCTIONS_MARKER)
                .count(),
            1
        );
        let context: Value = serde_json::from_str(updated.context.as_deref().unwrap()).unwrap();
        assert_eq!(context["existing"], "preserved");
        assert_eq!(
            context["run_instructions"],
            "Use one current primary source and write a human-useful insight."
        );
        assert_eq!(context["run_instruction_spec"]["version"], 1);
        assert_eq!(context["run_instruction_spec"]["revision"], 2);
        assert_eq!(
            context["run_instruction_spec"]["conflict_rule"],
            "latest_revision_wins"
        );
        assert!(state.get_tasks_for_goal(&goal.id).await.unwrap().is_empty());
        assert!(
            state
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            state.get_schedules_for_goal(&goal.id).await.unwrap()[0].id,
            schedule.id
        );
    }

    #[tokio::test]
    async fn update_instructions_rejects_structurally_conflicting_policy() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous("Publish one verified note", "session", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 9 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: crate::cron_utils::compute_next_run("0 9 * * *")
                    .unwrap()
                    .to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call_with(
                &json!({
                    "action": "update_instructions",
                    "goal_id": goal.id,
                    "instructions": "Always finish the run.",
                    "run_requirement": "must_complete",
                    "insufficient_evidence": "skip_run"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.contains("Conflicting run policy"));
        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().description,
            goal.description
        );
    }

    #[tokio::test]
    async fn run_history_reports_open_tasks_as_schedule_blockers() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        assert!(
            tool.validate_arguments(r#"{"action":"run_history"}"#)
                .is_err()
        );
        assert!(
            tool.validate_arguments(r#"{"action":"run_history","goal_id":"goal-synthetic"}"#)
                .is_ok()
        );

        let goal = Goal::new_continuous(
            "Post daily optimized tweets",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 9 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily at 9am".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let open_task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Analyze fetched engagement metrics".to_string(),
            status: "pending".to_string(),
            priority: "low".to_string(),
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
            max_retries: 3,
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        state.create_task(&open_task).await.unwrap();
        let result = tool
            .call(
                &json!({
                    "action": "run_history",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains(&format!("next_run={}", schedule.next_run_at)));
        assert!(result.contains("Open task(s) blocking coalesced schedule fires"));
        assert!(result.contains("Analyze fetched engagement metrics"));
    }

    #[tokio::test]
    async fn run_history_projects_terminal_completion_timestamp() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous("Inspect a synthetic ledger", "session", None, None);
        state.create_goal(&goal).await.unwrap();
        let run = state
            .start_goal_run(&goal.id, "manual", None, None)
            .await
            .unwrap();
        state
            .finish_goal_run(&run.id, "completed", Some("Synthetic check complete"))
            .await
            .unwrap();
        let completed_at = state
            .get_goal_runs(&goal.id)
            .await
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.id == run.id)
            .and_then(|candidate| candidate.completed_at)
            .expect("terminal run timestamp");

        let result = tool
            .call(&json!({"action": "run_history", "goal_id": goal.id}).to_string())
            .await
            .unwrap();

        assert!(result.contains(&format!("completed={completed_at}")));
    }

    #[tokio::test]
    async fn unblock_hints_reports_timeout_guidance() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Knowledge base maintenance",
            "system",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Run scheduled check".to_string(),
            status: "failed".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: Some("Timeout while contacting embeddings service".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 1,
            max_retries: 3,
            created_at: now.clone(),
            started_at: Some(now.clone()),
            completed_at: Some(now),
        };
        state.create_task(&task).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "unblock_hints",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Unblock Hints"));
        assert!(result.contains("transient service/network"));
    }

    #[tokio::test]
    async fn set_budget_updates_scheduled_goal() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let result = tool
            .call_with(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_per_check": 1234,
                    "budget_daily": 5678
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(result.contains("Updated budget"));

        let updated = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(updated.budget_per_check, Some(1234));
        assert_eq!(updated.budget_daily, Some(5678));
    }

    #[tokio::test]
    async fn set_budget_rejects_non_owner() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let result = tool
            .call_with(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_daily": 9999
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Guest),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(result.contains("Only owners"));
    }

    #[tokio::test]
    async fn set_budget_validates_values() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let negative = tool
            .call_with(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_daily": -1
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(negative.contains("budget_daily must be >="));

        let too_large = tool
            .call_with(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_daily": 2000001
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(too_large.contains("max"));

        let allowed_relation = tool
            .call_with(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_per_check": 200,
                    "budget_daily": 100
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(allowed_relation.contains("Updated budget"));

        let updated = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(updated.budget_per_check, Some(200));
        assert_eq!(updated.budget_daily, Some(100));
    }
}
