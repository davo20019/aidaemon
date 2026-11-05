//! `set_reminder`: a plain reminder in two typed fields.
//!
//! The same scheduling path as `manage_memories(create_scheduled_goal)` with a
//! typed `reminder`, behind a schema that cannot pick the wrong action or drop
//! the schedule: `message` and `when`, both required.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Map, Value};

use crate::tools::ManageMemoriesTool;
use crate::traits::{
    StateStore, Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolOutcomeStatus,
};

pub struct SetReminderTool {
    scheduler: ManageMemoriesTool,
}

impl SetReminderTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self {
            scheduler: ManageMemoriesTool::new(state),
        }
    }
}

/// The equivalent `manage_memories` call, keeping the runtime's `_` fields
/// (session, role, channel) so ownership and scoping checks apply unchanged.
fn scheduler_arguments(arguments: &str) -> anyhow::Result<String> {
    let args: Map<String, Value> = serde_json::from_str(arguments)?;
    let field = |name: &str| {
        args.get(name)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let message = field("message").ok_or_else(|| anyhow::anyhow!("'message' is required"))?;
    let when = field("when").ok_or_else(|| anyhow::anyhow!("'when' is required"))?;
    let mut call = Map::new();
    call.insert("action".into(), json!("create_scheduled_goal"));
    call.insert("goal".into(), json!(message));
    call.insert("schedule".into(), json!(when));
    call.insert("reminder".into(), json!(message));
    for (key, value) in args.iter().filter(|(key, _)| key.starts_with('_')) {
        call.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(call).to_string())
}

#[async_trait]
impl Tool for SetReminderTool {
    fn name(&self) -> &str {
        "set_reminder"
    }

    fn description(&self) -> &str {
        "Remind the user later"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "set_reminder",
            "description": "Send the user a reminder message at a time or on a cadence. Nothing else runs when it fires. For scheduled work, use manage_memories create_scheduled_goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "Exact text to send, e.g. 'Check the oven.'" },
                    "when": { "type": "string", "description": "e.g. 'in 2 minutes', 'tomorrow at 9am', 'every weekday at 8am'" }
                },
                "required": ["message", "when"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.scheduler.call(&scheduler_arguments(arguments)?).await
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<crate::types::StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let output = self.call(arguments).await?;
        let outcome_status = if output.starts_with("Reminder set") {
            ToolOutcomeStatus::Succeeded
        } else {
            // Duplicate, owner-only refusal or an unparseable time.
            ToolOutcomeStatus::FailedPermanent
        };
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(outcome_status),
                ..ToolCallMetadata::default()
            },
        })
    }
}

impl crate::traits::ToolSemantics for SetReminderTool {
    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::mutation()
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn state() -> Arc<dyn StateStore> {
        let db = tempfile::NamedTempFile::new().unwrap();
        let path = db.path().to_str().unwrap().to_string();
        std::mem::forget(db);
        let embeddings = Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
        Arc::new(
            crate::state::SqliteStateStore::new(&path, 100, None, embeddings)
                .await
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn sets_an_active_typed_reminder() {
        let state = state().await;
        let tool = SetReminderTool::new(state.clone());
        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "message": "Check the oven.",
                    "when": "in 2h",
                    "_user_role": "owner",
                    "_session_id": "telegram:synthetic-user-1"
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();
        assert!(
            outcome.output.starts_with("Reminder set"),
            "{}",
            outcome.output
        );
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        let goal = state
            .get_scheduled_goals()
            .await
            .unwrap()
            .into_iter()
            .find(|goal| goal.session_id == "telegram:synthetic-user-1")
            .unwrap();
        assert_eq!(goal.status, "active");
        assert_eq!(
            crate::reminders::message_from_context(goal.context.as_deref()).as_deref(),
            Some("Check the oven.")
        );
    }

    #[tokio::test]
    async fn non_owner_cannot_set_a_reminder() {
        let tool = SetReminderTool::new(state().await);
        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "message": "Check the oven.",
                    "when": "in 2h",
                    "_user_role": "guest",
                    "_session_id": "telegram:synthetic-user-2"
                })
                .to_string(),
                None,
            )
            .await;
        let failed = match outcome {
            Err(_) => true,
            Ok(outcome) => outcome.metadata.outcome_status != Some(ToolOutcomeStatus::Succeeded),
        };
        assert!(failed);
    }

    #[test]
    fn missing_fields_are_errors() {
        assert!(scheduler_arguments(r#"{"message":"x","when":""}"#).is_err());
        assert!(scheduler_arguments(r#"{"when":"in 2h"}"#).is_err());
    }
}
