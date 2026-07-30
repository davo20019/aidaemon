use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCallSemantics, ToolCapabilities, ToolRole};

/// Read-only owner tool for inspecting external workspace checkpoints.
pub struct ListCheckpointsTool;

#[async_trait]
impl Tool for ListCheckpointsTool {
    fn name(&self) -> &str {
        "list_checkpoints"
    }

    fn description(&self) -> &str {
        "List local filesystem checkpoints and their rollback readiness"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "list_checkpoints",
            "description": "List recent local workspace checkpoints. This is read-only. Checkpoint content is held in an external shadow-Git store, not in the user's repository.",
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args = serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({}));
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(20)
            .clamp(1, 100) as usize;
        let root = args.get("root").and_then(Value::as_str);
        Ok(match crate::checkpoints::active_manager() {
            Some(manager) => manager.list_text(limit, root).await,
            None => "Filesystem checkpoints are disabled.".to_string(),
        })
    }
}
