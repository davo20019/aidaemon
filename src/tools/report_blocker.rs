use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::traits::{StateStore, Tool, ToolRole};

/// Tool for executors to report they are blocked and cannot proceed.
///
/// Phase 2 simplified behavior: updates the task to "blocked" status with
/// blocker details, then tells the executor to stop. Full blocker-resolution
/// channel deferred to Phase 3.
pub struct ReportBlockerTool {
    task_id: String,
    state: Arc<dyn StateStore>,
}

impl ReportBlockerTool {
    pub fn new(task_id: String, state: Arc<dyn StateStore>) -> Self {
        Self { task_id, state }
    }
}

#[derive(Deserialize)]
struct ReportBlockerArgs {
    reason: String,
    #[serde(default)]
    partial_work: Option<String>,
    #[serde(default)]
    options: Option<Vec<String>>,
}

#[async_trait]
impl Tool for ReportBlockerTool {
    fn name(&self) -> &str {
        "report_blocker"
    }

    fn description(&self) -> &str {
        "Report that you are blocked and cannot proceed. Use this instead of guessing \
         when you encounter ambiguity, missing information, or an obstacle you cannot resolve."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "report_blocker",
            "description": "Report that you are blocked and cannot proceed. Use this instead of guessing when you encounter ambiguity, missing information, or an obstacle you cannot resolve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you are blocked"
                    },
                    "partial_work": {
                        "type": "string",
                        "description": "What you completed so far"
                    },
                    "options": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Possible resolutions (if any)"
                    }
                },
                "required": ["reason"],
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ReportBlockerArgs = serde_json::from_str(arguments)?;

        // Build blocker details
        let mut blocker = format!("BLOCKED: {}", args.reason);
        if let Some(partial) = &args.partial_work {
            blocker.push_str(&format!("\nPartial work: {}", partial));
        }
        if let Some(options) = &args.options {
            blocker.push_str(&format!("\nPossible resolutions: {}", options.join(", ")));
        }

        // Update the task in the database
        if let Ok(Some(mut task)) = self.state.get_task_v3(&self.task_id).await {
            task.status = "blocked".to_string();
            task.blocker = Some(blocker.clone());
            if let Some(partial) = &args.partial_work {
                // Append partial work to existing result
                let existing = task.result.unwrap_or_default();
                task.result = if existing.is_empty() {
                    Some(partial.clone())
                } else {
                    Some(format!("{}\n{}", existing, partial))
                };
            }
            let _ = self.state.update_task_v3(&task).await;
            info!(task_id = %self.task_id, reason = %args.reason, "V3: executor reported blocker");
        }

        Ok(format!(
            "Blocker reported for task {}. Stop working and return a summary of what you completed. \
             The task lead will handle the blocker.",
            self.task_id
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{GoalV3, TaskV3};

    async fn setup_test_state() -> (Arc<dyn StateStore>, String, String) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );

        let goal = GoalV3::new_finite("Test goal", "test-session");
        state.create_goal_v3(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = TaskV3 {
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
        state.create_task_v3(&task).await.unwrap();

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
                    "reason": "Missing API credentials"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Blocker reported"));
        assert!(result.contains("Stop working"));

        let task = state.get_task_v3(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        assert!(task
            .blocker
            .as_deref()
            .unwrap()
            .contains("Missing API credentials"));
    }

    #[tokio::test]
    async fn test_report_blocker_with_partial_work() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "reason": "Need clarification on API version",
                    "partial_work": "Set up project structure and dependencies",
                    "options": ["Use v1 API", "Use v2 API"]
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Blocker reported"));

        let task = state.get_task_v3(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        assert!(task
            .blocker
            .as_deref()
            .unwrap()
            .contains("Need clarification"));
        assert!(task
            .blocker
            .as_deref()
            .unwrap()
            .contains("Possible resolutions"));
        assert_eq!(
            task.result.as_deref(),
            Some("Set up project structure and dependencies")
        );
    }
}
