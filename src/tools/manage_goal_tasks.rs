//! `manage_goal_tasks` lives in `aidaemon_work::tools::manage_goal_tasks`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_work::tools::manage_goal_tasks::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::store_prelude::*;
    use crate::traits::Goal;
    use crate::traits::StateStore;
    use crate::traits::Task;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolSemantics;
    use serde_json::json;
    use serde_json::Value;
    use std::sync::Arc;

    async fn setup_test_state() -> (Arc<dyn StateStore>, String) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );

        // Create a goal
        let goal = Goal::new_finite("Test goal", "test-session");
        state.create_goal(&goal).await.unwrap();

        // We need to keep db_file alive, but for tests we'll leak it
        std::mem::forget(db_file);
        (state as Arc<dyn StateStore>, goal.id)
    }

    /// Bookkeeping on existing tasks declares no effects; actions that queue
    /// work or end a goal stay undeclared.
    #[tokio::test]
    async fn only_bookkeeping_actions_declare_no_effects() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id, state);
        let effects = |args: Value| tool.call_effects(&args.to_string());
        let none = aidaemon_core::CallEffects::none();
        assert_eq!(effects(json!({"action": "list_tasks"})), none);
        assert_eq!(
            effects(json!({"action": "claim_task", "task_id": "t1"})),
            none
        );
        assert_eq!(
            effects(json!({"action": "update_task", "task_id": "t1", "status": "completed"})),
            none
        );
        for queues_work in [
            json!({"action": "update_task", "task_id": "t1", "status": "pending"}),
            json!({"action": "create_task", "description": "synthetic"}),
            json!({"action": "retry_task", "task_id": "t1"}),
            json!({"action": "resolve_blocker", "task_id": "t1"}),
            json!({"action": "complete_goal"}),
            json!({"action": "fail_goal"}),
        ] {
            assert!(!effects(queues_work.clone()).complete, "{queues_work}");
        }
    }

    fn test_task(goal_id: &str, id: &str, description: &str, created_at: &str) -> Task {
        Task {
            id: id.to_string(),
            goal_id: goal_id.to_string(),
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
            max_retries: 0,
            created_at: created_at.to_string(),
            started_at: None,
            completed_at: None,
        }
    }

    #[tokio::test]
    async fn scheduled_task_tool_isolates_current_run_and_hides_root() {
        let (state, goal_id) = setup_test_state().await;
        let prior = test_task(
            &goal_id,
            "11111111-1111-1111-1111-111111111111",
            "Prior run task",
            "2026-07-27T12:00:00Z",
        );
        let root = test_task(
            &goal_id,
            "22222222-2222-2222-2222-222222222222",
            "Manual scheduled run: publish the blog",
            "2026-07-28T12:00:00Z",
        );
        let current = test_task(
            &goal_id,
            "33333333-3333-3333-3333-333333333333",
            "Current run task",
            "2026-07-28T12:00:01Z",
        );
        state.create_task(&prior).await.unwrap();
        let prior_run = state.get_current_goal_run(&goal_id).await.unwrap().unwrap();
        state
            .finish_goal_run(&prior_run.id, "completed", Some("prior run"))
            .await
            .unwrap();
        let current_run = state
            .start_goal_run(&goal_id, "manual", None, Some(&root.id))
            .await
            .unwrap();
        state.create_task(&root).await.unwrap();
        state.create_task(&current).await.unwrap();

        let tool =
            ManageGoalTasksTool::new(goal_id, state.clone()).with_goal_run_id(Some(current_run.id));
        let listed = tool
            .call(&json!({ "action": "list_tasks" }).to_string())
            .await
            .unwrap();
        assert!(listed.contains("Current run task"));
        assert!(!listed.contains("Prior run task"));
        assert!(!listed.contains("Manual scheduled run"));

        let update = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": prior.id,
                    "status": "completed"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(update.contains("earlier run"));
        assert_eq!(
            state.get_task(&prior.id).await.unwrap().unwrap().status,
            "pending"
        );
    }

    #[tokio::test]
    async fn fail_goal_keeps_continuous_goal_active() {
        let (state, _finite) = setup_test_state().await;
        let cg = Goal::new_continuous("Daily recurring goal", "test-session", None, None);
        assert_eq!(cg.status, "active");
        state.create_goal(&cg).await.unwrap();
        let tool = ManageGoalTasksTool::new(cg.id.clone(), state.clone());

        tool.call(&json!({ "action": "fail_goal", "summary": "deploy failed today" }).to_string())
            .await
            .unwrap();

        let g = state.get_goal(&cg.id).await.unwrap().unwrap();
        assert_eq!(
            g.status, "active",
            "a single run failure must NOT terminally fail a continuous goal"
        );
        assert!(
            g.dispatch_failures >= 1,
            "the run failure should be recorded for repeated-failure detection"
        );
    }

    #[tokio::test]
    async fn fail_goal_alerts_after_repeated_continuous_failures() {
        let (state, _finite) = setup_test_state().await;
        let cg = Goal::new_continuous("Daily recurring goal", "test-session", None, None);
        state.create_goal(&cg).await.unwrap();
        let tool = ManageGoalTasksTool::new(cg.id.clone(), state.clone());
        let body = json!({ "action": "fail_goal", "summary": "run failed" }).to_string();

        // First two failures: no alert yet.
        tool.call(&body).await.unwrap();
        tool.call(&body).await.unwrap();
        let count = |ns: Vec<crate::traits::NotificationEntry>, id: &str| {
            ns.into_iter()
                .filter(|n| n.notification_type == "evergreen_alert" && n.goal_id == id)
                .count()
        };
        assert_eq!(
            count(state.get_pending_notifications(20).await.unwrap(), &cg.id),
            0,
            "should not alert before the failure threshold"
        );

        // Third consecutive failure crosses the threshold: one alert.
        tool.call(&body).await.unwrap();
        assert_eq!(
            count(state.get_pending_notifications(20).await.unwrap(), &cg.id),
            1,
            "should alert once a continuous goal has failed 3 runs in a row"
        );
    }

    #[tokio::test]
    async fn complete_goal_keeps_continuous_active_and_clears_failures() {
        let (state, _finite) = setup_test_state().await;
        let cg = Goal::new_continuous("Daily recurring goal", "test-session", None, None);
        state.create_goal(&cg).await.unwrap();
        let tool = ManageGoalTasksTool::new(cg.id.clone(), state.clone());

        // Record a couple of failures.
        let fail = json!({ "action": "fail_goal", "summary": "x" }).to_string();
        tool.call(&fail).await.unwrap();
        tool.call(&fail).await.unwrap();

        // A successful run: one completed task, then complete_goal.
        tool.call(
            &json!({ "action": "create_task", "description": "daily check", "task_order": 1 })
                .to_string(),
        )
        .await
        .unwrap();
        let tasks = state.get_tasks_for_goal(&cg.id).await.unwrap();
        let mut t = tasks[0].clone();
        t.status = "completed".to_string();
        t.completed_at = Some(chrono::Utc::now().to_rfc3339());
        state.update_task(&t).await.unwrap();
        tool.call(&json!({ "action": "complete_goal", "summary": "daily check done" }).to_string())
            .await
            .unwrap();

        let g = state.get_goal(&cg.id).await.unwrap().unwrap();
        assert_eq!(
            g.status, "active",
            "a continuous goal stays active after a successful run"
        );
        assert_eq!(
            g.dispatch_failures, 0,
            "a successful run clears the failure streak"
        );
    }

    #[tokio::test]
    async fn complete_goal_not_blocked_by_dead_interrupted_task() {
        // Regression: the daily-tweet goal 9a744834 accumulated watchdog-
        // interrupted tasks from prior fires; because complete_goal treated any
        // non-completed/skipped status as "still incomplete", a dead interrupted
        // task permanently blocked every new run, which then looped and got
        // interrupted itself. An interrupted (dead terminal) task must NOT block.
        let (state, _finite) = setup_test_state().await;
        let cg = Goal::new_continuous("Daily recurring goal", "test-session", None, None);
        state.create_goal(&cg).await.unwrap();
        let tool = ManageGoalTasksTool::new(cg.id.clone(), state.clone());

        // This run's real work: post the tweet (completed).
        tool.call(&json!({ "action": "create_task", "description": "Post the tweet" }).to_string())
            .await
            .unwrap();
        // A dead task left over from a prior fire.
        tool.call(
            &json!({ "action": "create_task", "description": "Prior scheduled fire" }).to_string(),
        )
        .await
        .unwrap();
        let tasks = state.get_tasks_for_goal(&cg.id).await.unwrap();
        let mut done = tasks[0].clone();
        done.status = "completed".to_string();
        done.completed_at = Some(chrono::Utc::now().to_rfc3339());
        state.update_task(&done).await.unwrap();
        let mut dead = tasks[1].clone();
        dead.status = "interrupted".to_string();
        state.update_task(&dead).await.unwrap();

        let result = tool
            .call(&json!({ "action": "complete_goal", "summary": "tweet posted" }).to_string())
            .await
            .unwrap();

        assert!(
            !result.contains("Blocked"),
            "a dead interrupted task must not block completion: {result}"
        );
        assert!(result.contains("run completed"), "got: {result}");
        let g = state.get_goal(&cg.id).await.unwrap().unwrap();
        assert_eq!(
            g.status, "active",
            "continuous goal stays active after the run"
        );
    }

    #[tokio::test]
    async fn complete_goal_still_blocked_by_active_running_task() {
        // The guard's remaining purpose: genuinely in-flight work still blocks.
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());
        tool.call(&json!({ "action": "create_task", "description": "In-flight work" }).to_string())
            .await
            .unwrap();
        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let mut running = tasks[0].clone();
        running.status = "running".to_string();
        state.update_task(&running).await.unwrap();

        let result = tool
            .call(&json!({ "action": "complete_goal", "summary": "done" }).to_string())
            .await
            .unwrap();
        assert!(
            result.contains("Blocked") && result.contains("still running"),
            "an active running task must still block completion: {result}"
        );
    }

    #[tokio::test]
    async fn finite_goal_cannot_complete_with_failed_terminal_task() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Build and publish the site"
            })
            .to_string(),
        )
        .await
        .unwrap();
        let mut failed = state.get_tasks_for_goal(&goal_id).await.unwrap()[0].clone();
        failed.status = "failed".to_string();
        failed.error = Some("build never ran".to_string());
        failed.completed_at = Some(chrono::Utc::now().to_rfc3339());
        state.update_task(&failed).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "The source files exist"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(
            result.contains("requires successful task state"),
            "{result}"
        );
        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "active");
        assert!(goal.completed_at.is_none());
    }

    #[tokio::test]
    async fn finite_goal_cannot_complete_with_stale_blocker_on_completed_task() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Verify the deployment"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Reproduce a legacy inconsistent row without first moving through the
        // blocked state. The transition fence protects new blocked tasks; this
        // success predicate also protects already-corrupt persisted data.
        let mut inconsistent = state.get_tasks_for_goal(&goal_id).await.unwrap()[0].clone();
        inconsistent.status = "completed".to_string();
        inconsistent.blocker = Some("verification never ran".to_string());
        inconsistent.completed_at = Some(chrono::Utc::now().to_rfc3339());
        state.update_task(&inconsistent).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "The deployment is ready"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(
            result.contains("requires successful task state"),
            "{result}"
        );
        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "active");
        assert!(goal.completed_at.is_none());
    }

    #[tokio::test]
    async fn lead_completion_after_failed_attempt_clears_stale_error() {
        let (state, _finite) = setup_test_state().await;
        let goal = Goal::new_continuous("Synthetic daily publication", "test-session", None, None);
        state.create_goal(&goal).await.unwrap();
        let tool = ManageGoalTasksTool::new(goal.id.clone(), state.clone());
        tool.call(&json!({"action": "create_task", "description": "Publish one post"}).to_string())
            .await
            .unwrap();
        let run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        // The specialist attempt failed and left its error on the task.
        let mut failed = state.get_tasks_for_goal_run(&run.id).await.unwrap()[0].clone();
        failed.status = "failed".to_string();
        failed.error = Some("Specialist handoff failed permanently".to_string());
        state.update_task(&failed).await.unwrap();

        // The lead then did the work directly and records success, once
        // without restating the error and once describing it as history.
        let scoped =
            ManageGoalTasksTool::new(goal.id.clone(), state.clone()).with_goal_run_id(Some(run.id));
        for error in [None, Some("specialist failed; lead published directly")] {
            let mut update = json!({
                "action": "update_task",
                "task_id": failed.id,
                "status": "completed",
                "result": "Published and verified"
            });
            if let Some(error) = error {
                update["error"] = json!(error);
            }
            scoped.call(&update.to_string()).await.unwrap();
            let task = state.get_task(&failed.id).await.unwrap().unwrap();
            assert_eq!(task.status, "completed");
            assert!(task.error.is_none(), "{error:?}: {:?}", task.error);
        }

        let result = scoped
            .call(&json!({"action": "complete_goal", "summary": "published"}).to_string())
            .await
            .unwrap();
        assert!(result.contains("run completed"), "{result}");
    }

    #[tokio::test]
    async fn current_recurring_run_cannot_complete_with_failed_task() {
        let (state, _finite) = setup_test_state().await;
        let goal = Goal::new_continuous("Recurring publication", "test-session", None, None);
        state.create_goal(&goal).await.unwrap();
        let legacy_tool = ManageGoalTasksTool::new(goal.id.clone(), state.clone());
        legacy_tool
            .call(
                &json!({
                    "action": "create_task",
                    "description": "Publish this run"
                })
                .to_string(),
            )
            .await
            .unwrap();
        let run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        let mut failed = state.get_tasks_for_goal_run(&run.id).await.unwrap()[0].clone();
        failed.status = "failed".to_string();
        failed.error = Some("publication failed".to_string());
        state.update_task(&failed).await.unwrap();
        let scoped_tool =
            ManageGoalTasksTool::new(goal.id.clone(), state.clone()).with_goal_run_id(Some(run.id));

        let result = scoped_tool
            .call(&json!({"action": "complete_goal", "summary": "run done"}).to_string())
            .await
            .unwrap();
        assert!(
            result.contains("requires successful task state"),
            "{result}"
        );
    }

    #[tokio::test]
    async fn test_create_task_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_task",
                    "description": "Write the code",
                    "priority": "high",
                    "task_order": 1
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Created task"));
        assert!(result.contains("Write the code"));

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "Write the code");
        assert_eq!(tasks[0].priority, "high");
        assert_eq!(tasks[0].task_order, 1);
    }

    #[tokio::test]
    async fn test_list_tasks_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create two tasks
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task one",
                "task_order": 1
            })
            .to_string(),
        )
        .await
        .unwrap();

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task two",
                "task_order": 2
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(&json!({"action": "list_tasks"}).to_string())
            .await
            .unwrap();

        assert!(result.contains("2 total"));
        assert!(result.contains("Task one"));
        assert!(result.contains("Task two"));
    }

    #[tokio::test]
    async fn test_update_task_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Do something"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = &tasks[0].id;

        // Update it
        let result = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": task_id,
                    "status": "completed",
                    "result": "Done successfully"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("completed"));

        let updated = state.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.status, "completed");
        assert_eq!(updated.result.as_deref(), Some("Done successfully"));
        assert!(updated.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_complete_goal_requires_completed_tasks() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Do the work"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let task_id = state.get_tasks_for_goal(&goal_id).await.unwrap()[0]
            .id
            .clone();
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": task_id,
                "status": "completed",
                "result": "All tasks done"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "All tasks done"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("completed"));
        assert!(result.contains("All tasks done"));

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "completed");
        assert!(goal.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_complete_goal_includes_final_task_result_excerpt() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Find largest directories"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = &tasks[0].id;

        tool.call(
            &json!({
                "action": "update_task",
                "task_id": task_id,
                "status": "completed",
                "result": "1.2G /Users/me/projects/aidaemon/target"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "Finished disk usage audit"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Goal"));
        assert!(result.contains("Finished disk usage audit"));
        assert!(result.contains("Task result:"));
        assert!(result.contains("1.2G /Users/me/projects/aidaemon/target"));
    }

    #[tokio::test]
    async fn test_complete_goal_includes_multiple_recent_task_results() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        for i in 1..=4 {
            tool.call(
                &json!({
                    "action": "create_task",
                    "description": format!("Research step {}", i)
                })
                .to_string(),
            )
            .await
            .unwrap();
        }

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        for task in &tasks {
            tool.call(
                &json!({
                    "action": "update_task",
                    "task_id": task.id,
                    "status": "completed",
                    "result": format!("Result payload for {}", task.description)
                })
                .to_string(),
            )
            .await
            .unwrap();
        }

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "Compiled multi-step research outputs"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("4/4 tasks completed."));
        assert!(result.contains("Recent task results:"));
        assert!(result.contains("Research step 4"));
        assert!(result.contains("Research step 3"));
        assert!(result.contains("Research step 2"));
        assert!(result.contains("(+1 earlier completed task result omitted)"));
    }

    #[tokio::test]
    async fn test_complete_goal_uses_task_state_not_summary_wording() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Verify the synthetic target"
            })
            .to_string(),
        )
        .await
        .unwrap();
        let task = state.get_tasks_for_goal(&goal_id).await.unwrap().remove(0);
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": task.id,
                "status": "completed",
                "result": "Structured verification receipt recorded"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "The documentation explains that partially completed runs say verification pending."
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(
            result.contains(" completed:"),
            "unexpected result: {result}"
        );

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "completed");
        assert!(goal.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_complete_goal_blocks_when_tasks_are_not_done() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Run final verification"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "Everything is finished"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Blocked:"));
        assert!(result.contains("Run final verification"));

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "active");
        assert!(goal.completed_at.is_none());
    }

    #[tokio::test]
    async fn test_fail_goal_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "fail_goal",
                    "summary": "Could not complete"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("failed"));
        assert!(result.contains("Could not complete"));

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "failed");
        assert_eq!(
            goal.context
                .as_deref()
                .and_then(|ctx| serde_json::from_str::<Value>(ctx).ok())
                .and_then(|ctx| {
                    ctx.get("failure_summary")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned)
                }),
            Some("Could not complete".to_string())
        );
    }

    #[tokio::test]
    async fn test_claim_task_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Claimable task"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // First claim should succeed
        let result = tool
            .call(
                &json!({
                    "action": "claim_task",
                    "task_id": &task_id,
                    "agent_id": "executor-1"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Claimed task"));
        assert!(result.contains("executor-1"));

        // Second claim should fail
        let result2 = tool
            .call(
                &json!({
                    "action": "claim_task",
                    "task_id": &task_id,
                    "agent_id": "executor-2"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result2.contains("Failed to claim"));
    }

    #[tokio::test]
    async fn test_claim_task_dependency_check() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create task A
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task A",
                "task_order": 1
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_a_id = tasks[0].id.clone();

        // Create task B that depends on A
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task B",
                "task_order": 2,
                "depends_on": [&task_a_id]
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_b_id = tasks
            .iter()
            .find(|t| t.description == "Task B")
            .unwrap()
            .id
            .clone();

        // Claim B should fail — A not completed
        let result = tool
            .call(
                &json!({
                    "action": "claim_task",
                    "task_id": &task_b_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Cannot claim"));
        assert!(result.contains("not completed"));

        // Complete task A
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_a_id,
                "status": "completed",
                "result": "Done"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Now claim B should succeed
        let result2 = tool
            .call(
                &json!({
                    "action": "claim_task",
                    "task_id": &task_b_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result2.contains("Claimed task"));
    }

    #[tokio::test]
    async fn test_retry_task_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create an idempotent task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Retryable task",
                "idempotent": true
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // Fail the task
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "failed",
                "error": "Something went wrong"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Retry should succeed
        let result = tool
            .call(
                &json!({
                    "action": "retry_task",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("reset to pending"));
        assert!(result.contains("1/3"));

        // Verify the task was reset
        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "pending");
        assert_eq!(task.retry_count, 1);
        assert!(task.error.is_none());
        assert!(task.agent_id.is_none());
    }

    #[tokio::test]
    async fn test_retry_task_non_idempotent() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a non-idempotent task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Non-retryable task"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // Fail it
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "failed",
                "error": "Oops"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Retry should fail — not idempotent
        let result = tool
            .call(
                &json!({
                    "action": "retry_task",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("not marked as idempotent"));
    }

    #[tokio::test]
    async fn test_retry_task_max_retries_exceeded() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create an idempotent task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Exhaustible task",
                "idempotent": true
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // Exhaust all retries (max_retries = 3)
        for _ in 0..3 {
            // Fail the task
            tool.call(
                &json!({
                    "action": "update_task",
                    "task_id": &task_id,
                    "status": "failed",
                    "error": "Failed again"
                })
                .to_string(),
            )
            .await
            .unwrap();

            // Retry
            tool.call(
                &json!({
                    "action": "retry_task",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        }

        // Fail it one more time
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "failed",
                "error": "Failed again"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Next retry should fail — max retries reached
        let result = tool
            .call(
                &json!({
                    "action": "retry_task",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("max retries reached"));
    }

    #[tokio::test]
    async fn test_update_task_dependency_enforcement() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create task A
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task A"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_a_id = tasks[0].id.clone();

        // Create task B that depends on A
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task B",
                "depends_on": [&task_a_id]
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_b_id = tasks
            .iter()
            .find(|t| t.description == "Task B")
            .unwrap()
            .id
            .clone();

        // Try to set B to "running" — should fail
        let result = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": &task_b_id,
                    "status": "running"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Cannot set task"));
        assert!(result.contains("not completed"));

        // Complete A
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_a_id,
                "status": "completed"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Now setting B to "running" should succeed
        let result2 = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": &task_b_id,
                    "status": "running"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result2.contains("running"));
    }

    #[tokio::test]
    async fn test_list_tasks_shows_deps_and_groups() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create task A with parallel_group
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task A",
                "task_order": 1,
                "parallel_group": "phase-1"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_a_id = tasks[0].id.clone();

        // Create task B with depends_on and idempotent
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task B",
                "task_order": 2,
                "depends_on": [&task_a_id],
                "idempotent": true
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(&json!({"action": "list_tasks"}).to_string())
            .await
            .unwrap();

        assert!(result.contains("group: phase-1"));
        assert!(result.contains("deps: ["));
        assert!(result.contains("retries: 0/3"));
    }

    #[tokio::test]
    async fn test_context_accumulation_on_completion() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Build the frontend"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = &tasks[0].id;

        // Complete with a result
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": task_id,
                "status": "completed",
                "result": "Built React frontend with login page and dashboard"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Check goal context has the task result
        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert!(
            goal.context.is_some(),
            "Goal should have context after task completion"
        );
        let ctx: serde_json::Value =
            serde_json::from_str(goal.context.as_deref().unwrap()).unwrap();
        let results = ctx["task_results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["description"], "Build the frontend");
        assert!(results[0]["result_summary"]
            .as_str()
            .unwrap()
            .contains("React frontend"));
    }

    #[tokio::test]
    async fn test_context_accumulation_multiple_tasks() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create two tasks
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task Alpha"
            })
            .to_string(),
        )
        .await
        .unwrap();
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task Beta"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();

        // Complete both
        for task in &tasks {
            tool.call(
                &json!({
                    "action": "update_task",
                    "task_id": task.id,
                    "status": "completed",
                    "result": format!("Completed {}", task.description)
                })
                .to_string(),
            )
            .await
            .unwrap();
        }

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        let ctx: serde_json::Value =
            serde_json::from_str(goal.context.as_deref().unwrap()).unwrap();
        let results = ctx["task_results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_resolve_blocker_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Blocked task"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // Set it to blocked
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "blocked"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Resolve the blocker
        let result = tool
            .call(
                &json!({
                    "action": "resolve_blocker",
                    "task_id": &task_id,
                    "result": "Found alternative approach"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Blocker resolved"));
        assert!(result.contains("pending"));

        // Verify task is pending with resolution context
        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "pending");
        assert!(task.blocker.is_none());
        let journal = state.get_task_journal(&task_id, 10).await.unwrap();
        assert!(journal.iter().any(|entry| {
            entry.entry_type == "unblocked" && entry.body.contains("Found alternative approach")
        }));
    }

    #[tokio::test]
    async fn blocked_task_cannot_be_overwritten_as_completed() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Inspect deployment tooling"
            })
            .to_string(),
        )
        .await
        .unwrap();
        let task_id = state.get_tasks_for_goal(&goal_id).await.unwrap()[0]
            .id
            .clone();
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "blocked",
                "result": "A preflight check did not run"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": &task_id,
                    "status": "completed",
                    "result": "Done"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Use resolve_blocker"));
        assert_eq!(
            state.get_task(&task_id).await.unwrap().unwrap().status,
            "blocked"
        );
    }

    #[tokio::test]
    async fn test_resolve_blocker_not_blocked() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Normal task"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        let result = tool
            .call(
                &json!({
                    "action": "resolve_blocker",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("not blocked"));
    }

    #[tokio::test]
    async fn mandate_create_task_ignores_runtime_owned_placement_hints() {
        use crate::traits::{
            Intention, Mandate, MandateAuthority, MandateDecisionCycle, MandateDecisionOutcome,
        };
        let (state, _) = setup_test_state().await;
        let goal = Goal::new_continuous("Synthetic mandate", "owner-session", None, None);
        let authority = MandateAuthority {
            allow_observations: true,
            allowed_tools: vec!["http_request".to_string()],
            allowed_mutation_effects: vec!["remote_mutation".to_string()],
            allowed_target_prefixes: vec!["https://api.x.com/2/".to_string()],
            operation_scopes: Vec::new(),
            max_mutating_actions_per_cycle: 1,
            max_mutating_actions_per_rolling_24h: 8,
            min_seconds_between_mutations: 900,
            owner_alerts: false,
        };
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Publish one synthetic post",
            "owner-session",
            authority,
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        assert_eq!(
            state
                .claim_due_mandates(1, "test-heartbeat", 300)
                .await
                .unwrap()
                .len(),
            1
        );
        let root = test_task(
            &goal.id,
            &uuid::Uuid::new_v4().to_string(),
            "Run one bounded mandate cycle",
            &chrono::Utc::now().to_rfc3339(),
        );
        let run = state
            .start_goal_run(&goal.id, "mandate", None, Some(&root.id))
            .await
            .unwrap();
        state.create_task(&root).await.unwrap();
        let root_attempt = state
            .claim_task_with_lease(&root.id, "task-lead", Some("profile-task-lead"), 180)
            .await
            .unwrap()
            .expect("mandate root should be claimable");
        let decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "Publish the synthetic post",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &decision.id,
            &run.id,
            "Publish one synthetic post",
            "It keeps the requested cadence",
        );
        state
            .record_mandate_decision(&decision, Some(&intention), None)
            .await
            .unwrap();

        let tool = ManageGoalTasksTool::new(goal.id.clone(), state.clone())
            .with_goal_run_id(Some(run.id.clone()));
        // The schema offers placement hints; a mandate task lead that fills
        // them in must still be able to create its committed work task.
        let fence = crate::traits::ToolCallContext {
            mandate_id: Some(mandate.id.clone()),
            mandate_version: Some(mandate.version),
            goal_run_id: Some(run.id.clone()),
            task_attempt_id: Some(root_attempt.id.clone()),
            ..Default::default()
        };
        let result = tool
            .call_with_execution_context(
                &json!({
                    "action": "create_task",
                    "description": "Publish exactly one synthetic post",
                    "worker_profile": "executor",
                    "workspace_policy": "isolated",
                })
                .to_string(),
                None,
                fence,
            )
            .await
            .expect("placement hints must not block a committed ACT")
            .output;
        assert!(!result.contains("Cannot create task"), "{result}");
        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert!(tasks
            .iter()
            .any(|task| task.description == "Publish exactly one synthetic post"));
    }
}
