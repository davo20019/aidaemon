//! `set_reminder`: a plain reminder in two typed fields.
//!
//! The same creation path as `schedule_task` with a typed `reminder`, behind
//! a schema that cannot drop the schedule: `message` and `when`, both
//! required. The path answers with a typed outcome, so a refusal is never
//! read as a reminder set.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tools::schedule_task::{ScheduleRequest, ScheduledWork};
use crate::tools::traits::{AutomationStore, Tool, ToolCallSemantics, ToolCapabilities};

pub struct SetReminderTool {
    work: ScheduledWork,
}

impl SetReminderTool {
    pub fn new(state: Arc<dyn AutomationStore>) -> Self {
        Self {
            work: ScheduledWork::new(state),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReminderArgs {
    message: String,
    when: String,
    name: Option<String>,
}

/// The reminder as a scheduling request. It runs with this call's
/// dispatcher facts (session, role, channel), so ownership and scoping
/// checks apply unchanged.
fn reminder_request(arguments: &str) -> anyhow::Result<ScheduleRequest> {
    let args: ReminderArgs = serde_json::from_str(arguments)?;
    let message = args.message.trim();
    let when = args.when.trim();
    anyhow::ensure!(!message.is_empty(), "'message' is required");
    anyhow::ensure!(!when.is_empty(), "'when' is required");
    Ok(ScheduleRequest {
        task: message.to_string(),
        when: when.to_string(),
        name: args.name,
        reminder: Some(message.to_string()),
    })
}

#[async_trait]
impl Tool for SetReminderTool {
    fn name(&self) -> &str {
        "set_reminder"
    }

    fn description(&self) -> &str {
        "Set up a scheduled automation that only sends the user a reminder message at a time or on a cadence. Nothing else runs when it fires. For scheduled work, use schedule_task."
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
        Ok(self
            .work
            .create(&reminder_request(arguments)?, facts)
            .await?
            .into_tool_outcome())
    }
}

impl crate::tools::traits::ToolSemantics for SetReminderTool {
    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            offering: aidaemon_tools::semantics::Offering::Core,
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
        assert!(reminder_request(r#"{"message":"x","when":""}"#).is_err());
        assert!(reminder_request(r#"{"when":"in 2h"}"#).is_err());
    }
}
