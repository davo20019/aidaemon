use async_trait::async_trait;
use serde_json::{json, Value};

use crate::semantics::{ToolCallSemantics, ToolCapabilities, ToolRole};
use crate::OutcomeTool as Tool;

/// Read-only owner tool for inspecting external workspace checkpoints.
pub struct ListCheckpointsTool;

#[async_trait]
impl Tool for ListCheckpointsTool {
    fn name(&self) -> &str {
        "list_checkpoints"
    }

    fn description(&self) -> &str {
        "List recent local workspace checkpoints. This is read-only. Checkpoint content is held in an external shadow-Git store, not in the user's repository."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "list_checkpoints",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum checkpoints to return (default: 20)"
                    },
                    "root": {
                        "type": "string",
                        "description": "Optional substring filter for the project root"
                    }
                },
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::semantics::ToolCallContext,
    ) -> anyhow::Result<crate::semantics::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args = serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({}));
            let limit = args
                .get("limit")
                .and_then(Value::as_u64)
                .unwrap_or(20)
                .clamp(1, 100) as usize;
            let root = args.get("root").and_then(Value::as_str);
            Ok(match super::active_manager() {
                Some(manager) => manager.list_text(limit, root).await,
                None => "Filesystem checkpoints are disabled.".to_string(),
            })
        }
        .await;
        output.map(crate::semantics::ToolCallOutcome::from_output)
    }
}

impl crate::semantics::ToolSemantics for ListCheckpointsTool {
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
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

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::observation()
    }
}
