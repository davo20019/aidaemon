use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{GoalTraceStore, ToolCapabilities};
use aidaemon_tools::OutcomeTool as Tool;

use super::goal_trace::GoalTraceTool;

pub struct ToolTraceTool {
    inner: GoalTraceTool,
}

impl ToolTraceTool {
    pub fn new(state: Arc<dyn GoalTraceStore>) -> Self {
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
        "Quick alias for tool-level execution traces. Equivalent to goal_trace(action='tool_trace')."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "tool_trace",
            "description": self.description(),
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

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: ToolTraceArgs = serde_json::from_str(arguments)?;

            let delegated = json!({
                "action": "tool_trace",
                "goal_id": args.goal_id,
                "task_id": args.task_id,
                "tool_name": args.tool_name,
                "limit": args.limit
            });
            self.inner
                .execute(
                    &delegated.to_string(),
                    &crate::traits::ToolCallContext::default(),
                )
                .await
                .map(|outcome| outcome.output)
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for ToolTraceTool {
    /// Past tool results replay content the tools read, whoever wrote it.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Untrusted
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
}
