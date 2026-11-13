//! `set_reminder`: a plain reminder in two typed fields.
//!
//! The same scheduling path as `manage_memories(create_scheduled_goal)` with a
//! typed `reminder`, behind a schema that cannot pick the wrong action or drop
//! the schedule: `message` and `when`, both required.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Map, Value};

use crate::tools::manage_memories::ManageMemoriesTool;
use crate::tools::traits::{
    AutomationStore, Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolOutcomeStatus,
};

pub struct SetReminderTool {
    scheduler: ManageMemoriesTool,
}

impl SetReminderTool {
    pub fn new(
        state: Arc<dyn AutomationStore>,
        store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    ) -> Self {
        Self {
            scheduler: ManageMemoriesTool::new(state, store),
        }
    }
}

/// The equivalent `manage_memories` call. It runs with this call's
/// dispatcher facts (session, role, channel), so ownership and scoping
/// checks apply unchanged.
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
    if let Some(name) = field("name") {
        call.insert("name".into(), json!(name));
    }
    Ok(Value::Object(call).to_string())
}

#[async_trait]
impl Tool for SetReminderTool {
    fn name(&self) -> &str {
        "set_reminder"
    }

    fn description(&self) -> &str {
        "Set up a scheduled automation that only sends the user a reminder message at a time or on a cadence. Nothing else runs when it fires. For scheduled work, use manage_memories create_scheduled_goal."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "set_reminder",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "Exact text to send, e.g. 'Check the oven.'" },
                    "when": { "type": "string", "description": "e.g. 'in 2 minutes', 'tomorrow at 9am', 'every weekday at 8am'" },
                    "name": { "type": "string", "description": "The short name the user gave it, exactly as given; omit when none." }
                },
                "required": ["message", "when"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::tools::traits::ToolCallContext,
    ) -> anyhow::Result<crate::tools::traits::ToolCallOutcome> {
        let facts = facts.clone();

        let output = self
            .scheduler
            .run(&scheduler_arguments(arguments)?, &facts)
            .await?;
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
            finish: None,
        })
    }
}

impl crate::tools::traits::ToolSemantics for SetReminderTool {
    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            always_offered: true,
            ..Default::default()
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::mutation_with(crate::tools::traits::ToolMutationEffects::SCHEDULED_WORK)
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

    #[test]
    fn missing_fields_are_errors() {
        assert!(scheduler_arguments(r#"{"message":"x","when":""}"#).is_err());
        assert!(scheduler_arguments(r#"{"when":"in 2h"}"#).is_err());
    }
}
