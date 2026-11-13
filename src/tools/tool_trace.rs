//! `tool_trace` lives in `aidaemon_work::tools::tool_trace`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_work::tools::tool_trace::*;

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
    async fn alias_returns_tool_trace_output() {
        let state = setup_state().await;
        let tool = ToolTraceTool::new(state.clone());

        let goal = Goal::new_finite("Alias trace goal", "user-session");
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Run task".to_string(),
            status: "completed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some("ok".to_string()),
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
                tokens_used: Some(7),
                created_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "goal_id": goal.id,
                    "tool_name": "web_fetch"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Tool Trace"));
        assert!(result.contains("web_fetch: calls 1"));
    }
}
