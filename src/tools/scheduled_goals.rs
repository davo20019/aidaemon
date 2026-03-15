use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::tools::manage_memories::ManageMemoriesTool;
use crate::tools::terminal::ApprovalRequest;
use crate::traits::{
    StateStore, Tool, ToolCallSemantics, ToolCapabilities, ToolSemanticAffordances,
    ToolSemanticFacet, ToolSemanticScope, ToolVerificationMode,
};

const SCHEDULED_GOAL_ACTIONS: &[&str] = &[
    "create_scheduled_goal",
    "list_scheduled",
    "list_scheduled_matching",
    "add_schedule",
    "cancel_scheduled",
    "pause_scheduled",
    "resume_scheduled",
    "retry_scheduled",
    "retry_failed_scheduled",
    "cancel_scheduled_matching",
    "retry_scheduled_matching",
    "diagnose_scheduled",
    "trigger_now",
];

#[derive(Deserialize)]
struct ScheduledGoalArgs {
    action: String,
}

pub struct ScheduledGoalsTool {
    inner: ManageMemoriesTool,
}

impl ScheduledGoalsTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self {
            inner: ManageMemoriesTool::new(state),
        }
    }

    pub fn with_approval_tx(mut self, tx: mpsc::Sender<ApprovalRequest>) -> Self {
        self.inner = self.inner.with_approval_tx(tx);
        self
    }

    fn action_from_args(arguments: &str) -> Option<String> {
        serde_json::from_str::<ScheduledGoalArgs>(arguments)
            .ok()
            .map(|args| args.action)
    }
}

#[async_trait]
impl Tool for ScheduledGoalsTool {
    fn name(&self) -> &str {
        "scheduled_goals"
    }

    fn description(&self) -> &str {
        "Create, list, diagnose, and manage scheduled goals and reminders. Use this for new reminders/recurring tasks and for questions about existing scheduled tasks."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "scheduled_goals",
            "description": "Manage scheduled goals and reminders. \
        Use `create_scheduled_goal` for a new reminder or recurring task. \
        Use `list_scheduled`, `list_scheduled_matching`, or `diagnose_scheduled` for existing scheduled-task questions like \"what are my scheduled tasks\" or \"what happened with my Twitter reminder\". \
        `list_scheduled_matching` returns matching goals or explicitly says \"no match\" — if no match is found, that means the goal does not exist; do not retry with different search terms. \
        Use `cancel_scheduled`, `pause_scheduled`, `resume_scheduled`, `retry_scheduled`, `retry_failed_scheduled`, `add_schedule`, or `trigger_now` to modify an existing scheduled goal. \
        If you do not know a goal ID yet, list scheduled goals first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": SCHEDULED_GOAL_ACTIONS,
                        "description": "Scheduled goal action"
                    },
                    "goal": {
                        "type": "string",
                        "description": "Goal description for a new scheduled goal"
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Goal ID or unique prefix for an existing scheduled goal"
                    },
                    "schedule_id": {
                        "type": "string",
                        "description": "Specific schedule ID when modifying one schedule on a goal"
                    },
                    "schedule": {
                        "type": "string",
                        "description": "Natural-language schedule like 'in 2 hours' or 'every Monday at 9am'"
                    },
                    "schedules": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Multiple natural-language schedules for batch creation"
                    },
                    "query": {
                        "type": "string",
                        "description": "Text query for matching scheduled goals"
                    },
                    "fire_policy": {
                        "type": "string",
                        "enum": ["coalesce", "always_fire"],
                        "description": "How recurring schedules behave when runs overlap"
                    },
                    "is_one_shot": {
                        "type": "boolean",
                        "description": "Whether the schedule should auto-complete after the first run"
                    },
                    "is_paused": {
                        "type": "boolean",
                        "description": "Whether a new scheduled goal should start paused"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max items to return for list actions"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: true,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        match Self::action_from_args(arguments).as_deref() {
            Some("list_scheduled" | "list_scheduled_matching" | "diagnose_scheduled") => {
                ToolCallSemantics::observation()
                    .with_verification_mode(ToolVerificationMode::ResultContent)
            }
            Some(_) | None => ToolCallSemantics::mutation(),
        }
    }

    fn semantic_affordances(&self) -> ToolSemanticAffordances {
        ToolSemanticAffordances::new(
            ToolSemanticScope::GoalState,
            &[ToolSemanticFacet::GoalState],
        )
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ScheduledGoalArgs = serde_json::from_str(arguments)?;
        if !SCHEDULED_GOAL_ACTIONS.contains(&args.action.as_str()) {
            return Ok(format!(
                "Unknown scheduled_goals action: '{}'. Use one of: {}.",
                args.action,
                SCHEDULED_GOAL_ACTIONS.join(", ")
            ));
        }

        <ManageMemoriesTool as Tool>::call(&self.inner, arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::Tool;

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
        state
    }

    #[tokio::test]
    async fn list_actions_are_observation_semantics() {
        let tool = ScheduledGoalsTool::new(setup_state().await);
        let semantics = tool.call_semantics(r#"{"action":"list_scheduled"}"#);
        assert!(semantics.observes_state());
        assert!(!semantics.mutates_state());
    }

    #[tokio::test]
    async fn create_actions_are_mutation_semantics() {
        let tool = ScheduledGoalsTool::new(setup_state().await);
        let semantics = tool.call_semantics(
            r#"{"action":"create_scheduled_goal","goal":"check logs","schedule":"in 2 hours"}"#,
        );
        assert!(semantics.mutates_state());
    }

    #[tokio::test]
    async fn affordances_are_goal_state() {
        let tool = ScheduledGoalsTool::new(setup_state().await);
        let affordances = tool.semantic_affordances();
        assert_eq!(affordances.scope, ToolSemanticScope::GoalState);
        assert!(affordances.supports(ToolSemanticFacet::GoalState));
    }
}
