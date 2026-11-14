//! `goal_trace` lives in `aidaemon_work::tools::goal_trace`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_work::tools::goal_trace::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::StateStore;
    use crate::traits::ToolCalls as _;
    use crate::traits::{Goal, Task, TaskActivity};
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
    async fn goal_trace_includes_task_and_tool_sequence() {
        let state = setup_state().await;
        let tool = GoalTraceTool::new(state.clone());

        let goal = Goal::new_finite("Trace this goal", "user-session");
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.clone(),
            description: "Inspect state".to_string(),
            status: "failed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: Some("agent-1".to_string()),
            context: None,
            result: None,
            error: Some("command timed out".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 1,
            max_retries: 3,
            created_at: now.clone(),
            started_at: Some(now.clone()),
            completed_at: Some(now.clone()),
        };
        state.create_task(&task).await.unwrap();

        state
            .log_task_activity(&TaskActivity {
                id: 0,
                task_id: task.id.clone(),
                activity_type: "tool_call".to_string(),
                tool_name: Some("terminal".to_string()),
                tool_args: Some("{\"command\":\"ls\"}".to_string()),
                result: None,
                success: None,
                tokens_used: Some(0),
                created_at: now.clone(),
            })
            .await
            .unwrap();
        state
            .log_task_activity(&TaskActivity {
                id: 0,
                task_id: task.id.clone(),
                activity_type: "tool_result".to_string(),
                tool_name: Some("terminal".to_string()),
                tool_args: None,
                result: Some("timed out".to_string()),
                success: Some(false),
                tokens_used: Some(12),
                created_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "goal_trace",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Goal Trace"));
        assert!(result.contains("terminal(err)"));
        assert!(result.contains("retry=1/3"));
    }

    #[tokio::test]
    async fn tool_trace_filters_by_tool_name() {
        let state = setup_state().await;
        let tool = GoalTraceTool::new(state.clone());

        let goal = Goal::new_finite("Trace tool filter", "user-session");
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Do work".to_string(),
            status: "completed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some("done".to_string()),
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: now.clone(),
            started_at: Some(now.clone()),
            completed_at: Some(now.clone()),
        };
        state.create_task(&task).await.unwrap();

        state
            .log_task_activity(&TaskActivity {
                id: 0,
                task_id: task.id.clone(),
                activity_type: "tool_result".to_string(),
                tool_name: Some("web_fetch".to_string()),
                tool_args: None,
                result: Some("ok".to_string()),
                success: Some(true),
                tokens_used: Some(8),
                created_at: now.clone(),
            })
            .await
            .unwrap();
        state
            .log_task_activity(&TaskActivity {
                id: 0,
                task_id: task.id.clone(),
                activity_type: "tool_result".to_string(),
                tool_name: Some("terminal".to_string()),
                tool_args: None,
                result: Some("ok".to_string()),
                success: Some(true),
                tokens_used: Some(5),
                created_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "tool_trace",
                    "goal_id": goal.id,
                    "tool_name": "web_fetch"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Tool Trace"));
        assert!(result.contains("Filter tool: web_fetch"));
        assert!(result.contains("web_fetch: calls 1"));
    }

    #[tokio::test]
    async fn goal_trace_without_goal_id_returns_recent_goal_summary() {
        let state = setup_state().await;
        let tool = GoalTraceTool::new(state.clone());

        let goal = Goal::new_finite("Investigate scheduler failures", "user-session");
        state.create_goal(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "goal_trace"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Goal Trace (Recent Goals)"));
        assert!(result.contains("Investigate scheduler failures"));
        assert!(result.contains("Tip: call `goal_trace` again with `goal_id`"));
    }

    #[tokio::test]
    async fn goal_trace_resolves_a_mandate_id_to_its_controller_goal() {
        let state = setup_state().await;
        let tool = GoalTraceTool::new(state.clone());
        let goal = Goal::new_continuous("Mandate controller", "owner-session", None, None);
        let mandate = crate::traits::Mandate::new(
            &goal.id,
            None,
            "Observe a bounded domain",
            "owner-session",
            crate::traits::MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "goal_trace",
                    "goal_id": mandate.id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Goal Trace"));
        assert!(result.contains(&format!("Goal ID: {}", goal.id)));
    }

    #[tokio::test]
    async fn tool_trace_without_scope_returns_recent_goal_summary() {
        let state = setup_state().await;
        let tool = GoalTraceTool::new(state.clone());

        let goal = Goal::new_finite("Collect deployment diagnostics", "user-session");
        state.create_goal(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "tool_trace"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Goal Trace (Recent Goals)"));
        assert!(result.contains("Collect deployment diagnostics"));
    }
}
