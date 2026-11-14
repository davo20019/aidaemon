//! `report_blocker` lives in `aidaemon_runtime::tools::report_blocker`.

#[cfg(test)]
pub use aidaemon_runtime::tools::report_blocker::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::StateStore;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolSemantics;
    use crate::traits::store_prelude::*;
    use crate::traits::{Goal, Task};
    use serde_json::json;
    use std::sync::Arc;

    async fn setup_test_state() -> (Arc<dyn StateStore>, String, String) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );

        let goal = Goal::new_finite("Test goal", "test-session");
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Test task".to_string(),
            status: "running".to_string(),
            priority: "medium".to_string(),
            task_order: 1,
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
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        state.create_task(&task).await.unwrap();

        std::mem::forget(db_file);
        (state as Arc<dyn StateStore>, goal.id, task.id)
    }

    #[tokio::test]
    async fn test_report_blocker_updates_task() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "reason": "Missing API credentials",
                    "blocker_class": "external_dependency",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "exact_need": "Provide credentials for the requested API."
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Executor outcome: blocked"));
        assert!(result.contains("Summary: Missing API credentials"));

        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        assert!(
            task.blocker
                .as_deref()
                .unwrap()
                .contains("Missing API credentials")
        );
        assert!(
            task.context
                .as_deref()
                .unwrap()
                .contains("\"executor_result\"")
        );
    }

    /// A recorded blocker ends the executor's run as partial work, from the
    /// tool's own outcome; the dispatcher ships the summary as the reply. A
    /// rejected report ends nothing.
    #[tokio::test]
    async fn report_blocker_finishes_partial() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());
        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "reason": "Missing API credentials",
                    "blocker_class": "external_dependency",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "exact_need": "Provide credentials for the requested API."
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();
        assert!(outcome.output.contains("Executor outcome: blocked"));
        let finish = outcome.finish.expect("a recorded blocker finishes the run");
        assert_eq!(finish.outcome, aidaemon_core::FinishOutcome::Partial);
        assert_eq!(finish.reply, None, "the reply is the summary the model saw");

        let rejected = tool
            .call_with_status_outcome(
                &json!({
                    "reason": "The upload may have gone through",
                    "blocker_class": "ambiguous_external_effect",
                    "external_effect_state": "none",
                    "recovery_attempts": []
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();
        assert!(
            rejected
                .output
                .starts_with("Error: blocker report rejected")
        );
        assert!(rejected.finish.is_none());
    }

    #[tokio::test]
    async fn test_report_blocker_enqueues_user_notification() {
        let (state, goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        tool.call(
            &json!({
                "reason": "Docker daemon is not reachable",
                "blocker_class": "external_dependency",
                "external_effect_state": "none",
                "recovery_attempts": [],
                "exact_need": "Start Docker, then ask me to retry."
            })
            .to_string(),
        )
        .await
        .unwrap();

        let pending = state.get_pending_notifications(10).await.unwrap();
        let entry = pending
            .iter()
            .find(|n| n.goal_id == goal_id)
            .expect("blocker should queue an immediate user notification");
        assert_eq!(entry.session_id, "test-session");
        assert_eq!(entry.notification_type, "escalation");
        assert!(entry.message.contains("Docker daemon is not reachable"));
        assert!(
            entry
                .message
                .contains("Start Docker, then ask me to retry.")
        );
        assert!(entry.message.starts_with("⚠️ **Action needed**"));
        assert!(entry.message.contains("**Blocked:**"));
        assert!(entry.message.contains("**Needed:**"));
        assert!(entry.message.contains("**Resume:** `/work unblock"));
    }

    #[tokio::test]
    async fn fenced_executor_blocker_is_handoff_not_owner_escalation() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let mut task = state.get_task(&task_id).await.unwrap().unwrap();
        task.status = "pending".to_string();
        task.started_at = None;
        task.completed_at = None;
        state.update_task(&task).await.unwrap();
        let attempt = state
            .claim_task_with_lease(&task_id, "synthetic-executor", None, 180)
            .await
            .unwrap()
            .expect("executor should claim the task");
        let tool = ReportBlockerTool::for_attempt(task_id.clone(), state.clone(), attempt, false);

        tool.call(
            &json!({
                "reason": "Current source state contradicts the earlier deployment receipt.",
                "blocker_class": "ambiguous_external_effect",
                "external_effect_state": "ambiguous",
                "recovery_attempts": [],
                "outcome": "partial_done_blocked",
                "partial_work": "The deployment completed and returned a public URL.",
                "exact_need": "Reconcile the deployment receipt with current source state.",
                "next_step": "Parent task lead should inspect current state and choose the safe recovery path."
            })
            .to_string(),
        )
        .await
        .unwrap();

        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        assert!(task.context.as_deref().unwrap().contains("executor_result"));
        assert!(
            state
                .get_pending_notifications(10)
                .await
                .unwrap()
                .is_empty(),
            "the parent task lead owns reconciliation and terminal notification"
        );
    }

    #[tokio::test]
    async fn operational_failure_without_recovery_evidence_is_not_escalated_to_the_user() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());
        let original_status = state.get_task(&task_id).await.unwrap().unwrap().status;

        let result = tool
            .call(
                &json!({
                    "reason": "The build reports invalid frontmatter while a direct parser check reports a valid integer.",
                    "blocker_class": "recovery_exhausted",
                    "external_effect_state": "none",
                    "recovery_attempts": [{
                        "action": "Parsed the current file directly",
                        "outcome": "The id is a valid integer",
                        "evidence": "Number.isInteger(data.id) returned true"
                    }]
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.starts_with("Error: blocker report rejected"));
        assert!(result.contains("at least two bounded in-scope recovery attempts"));
        assert_eq!(
            state.get_task(&task_id).await.unwrap().unwrap().status,
            original_status
        );
        assert!(
            state
                .get_pending_notifications(10)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn terminal_recovery_blocker_uses_parent_terminal_notification_only() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let mut task = state.get_task(&task_id).await.unwrap().unwrap();
        task.context = Some(json!({ "terminal_recovery": true }).to_string());
        state.update_task(&task).await.unwrap();
        let tool = ReportBlockerTool::new(task_id, state.clone());

        tool.call(
            &json!({
                "reason": "Browser verification is unavailable",
                "blocker_class": "recovery_exhausted",
                "external_effect_state": "none",
                "recovery_attempts": [
                    {
                        "action": "Tried the browser verifier",
                        "outcome": "Browser session was unavailable",
                        "evidence": "Browser tool returned an unavailable-session error"
                    },
                    {
                        "action": "Tried an authorized HTTP verification",
                        "outcome": "No HTTP-capable tool was present",
                        "evidence": "Executor tool inventory contained no HTTP client"
                    }
                ],
                "exact_need": "Use another verification path."
            })
            .to_string(),
        )
        .await
        .unwrap();

        assert!(
            state
                .get_pending_notifications(10)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn report_blocker_has_administrative_semantics() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id, state);

        let semantics = tool.call_semantics(
            r#"{"reason":"OAuth authorization required","blocker_class":"external_dependency","external_effect_state":"none","recovery_attempts":[]}"#,
        );

        assert_eq!(
            semantics.effect,
            crate::traits::ToolCallEffect::Administrative
        );
    }

    #[tokio::test]
    async fn test_report_blocker_with_partial_work() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "reason": "Need clarification on API version",
                    "blocker_class": "owner_input",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "outcome": "partial_done_blocked",
                    "partial_work": "Set up project structure and dependencies",
                    "exact_need": "Choose between the v1 and v2 API contract.",
                    "next_step": "Resume the client implementation once the API version is confirmed.",
                    "artifacts": ["/tmp/demo/Cargo.toml"],
                    "options": ["Use v1 API", "Use v2 API"]
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Executor outcome: partial_done_blocked"));
        assert!(
            result.contains("Completed work so far: Set up project structure and dependencies")
        );

        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        assert!(
            task.blocker
                .as_deref()
                .unwrap()
                .contains("Need clarification")
        );
        assert!(
            task.blocker
                .as_deref()
                .unwrap()
                .contains("Possible resolutions")
        );
        assert!(
            task.context
                .as_deref()
                .unwrap()
                .contains("\"partial_done_blocked\"")
        );
    }

    #[tokio::test]
    async fn test_report_blocker_supports_needs_approval() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "reason": "Need approval to rotate the production credentials",
                    "blocker_class": "missing_authority",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "outcome": "needs_approval",
                    "partial_work": "Validated the pending rotation script and staged the change plan",
                    "exact_need": "Owner approval to rotate the credentials in production.",
                    "next_step": "Run the approved credential rotation and verify the service health.",
                    "target": "production credentials",
                    "dependency_repair": false
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Executor outcome: needs_approval"));
        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert!(
            task.context
                .as_deref()
                .unwrap()
                .contains("\"needs_approval\"")
        );
    }

    #[tokio::test]
    async fn proven_causal_local_repair_is_not_accepted_as_missing_authority() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let project = tempfile::tempdir().unwrap();
        let target = project.path().join("content/posts/synthetic-post.md");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(
            &target,
            "---\ntitle: Synthetic\n---\n\nBody stays intact.\n",
        )
        .unwrap();

        let mut task = state.get_task(&task_id).await.unwrap().unwrap();
        let handoff = crate::agent::ExecutorHandoff {
            task_id: task_id.clone(),
            mission: "Build and publish the synthetic site.".to_string(),
            task_description: "Repair the failed build and continue the workflow.".to_string(),
            target_scope: crate::agent::TargetScope {
                allowed_targets: vec![
                    crate::traits::ToolTargetHint::new(
                        crate::traits::ToolTargetHintKind::ProjectScope,
                        project.path().to_string_lossy(),
                    )
                    .unwrap(),
                ],
                hard_fail_outside_scope: true,
            },
            expected_targets: Vec::new(),
            allowed_tools: None,
        };
        task.context = Some(
            crate::agent::persist_executor_handoff_context(task.context.as_deref(), &handoff)
                .unwrap(),
        );
        state.update_task(&task).await.unwrap();
        state
            .log_task_activity(&crate::traits::TaskActivity {
                id: 0,
                task_id: task_id.clone(),
                activity_type: "tool_call".to_string(),
                tool_name: Some("terminal".to_string()),
                tool_args: Some("{\"command\":\"npm run build\"}".to_string()),
                result: Some(format!(
                    "Build failed: {} lacks the required numeric frontmatter id.",
                    target.display()
                )),
                success: Some(false),
                tokens_used: None,
                created_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());
        let result = tool
            .call(
                &json!({
                    "reason": "The pre-existing post needs a mechanical frontmatter repair.",
                    "blocker_class": "missing_authority",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "outcome": "needs_approval",
                    "exact_need": "Authority to add the required numeric id.",
                    "next_step": "Add only the missing id, rerun the build, and continue.",
                    "target": target.to_string_lossy(),
                    "dependency_repair": true
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("already in scope"), "result: {result}");
        assert!(result.contains("durable tool evidence"), "result: {result}");
        assert_eq!(
            state.get_task(&task_id).await.unwrap().unwrap().status,
            "running"
        );
    }
}
