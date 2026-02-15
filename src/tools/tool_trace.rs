use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{StateStore, Tool, ToolCapabilities};

use super::goal_trace::GoalTraceTool;

pub struct ToolTraceTool {
    inner: GoalTraceTool,
}

impl ToolTraceTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self {
            inner: GoalTraceTool::new(state),
        }
    }
}

#[derive(Deserialize)]
struct ToolTraceArgs {
    #[serde(default, alias = "goal_id_v3")]
    goal_id: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ToolTraceTool {
    fn name(&self) -> &str {
        "tool_trace"
    }

    fn description(&self) -> &str {
        "Quick alias for tool-level execution traces (same backend as goal_trace action=tool_trace)"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "tool_trace",
            "description": "Quick alias for tool-level execution traces. Equivalent to goal_trace(action='tool_trace').",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "Goal ID (full or unique prefix). Required when task_id is not provided."
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID for task-scoped trace"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Optional tool name filter"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max events to return (default 30, max 200)"
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ToolTraceArgs = serde_json::from_str(arguments)?;

        let delegated = json!({
            "action": "tool_trace",
            "goal_id": args.goal_id,
            "task_id": args.task_id,
            "tool_name": args.tool_name,
            "limit": args.limit
        });
        self.inner.call(&delegated.to_string()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{Goal, Task, TaskActivity};

    async fn setup_state() -> Arc<dyn StateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
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
