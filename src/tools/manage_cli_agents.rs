use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::tools::cli_agent::CliAgentTool;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::tools::ApprovalBroker;
use crate::traits::{DynamicCliAgentStore, Tool, ToolCapabilities};
use crate::types::ApprovalResponse;

pub struct ManageCliAgentsTool {
    cli_tool: Arc<CliAgentTool>,
    state: Arc<dyn DynamicCliAgentStore>,
    approval_tx: ApprovalBroker,
}

impl ManageCliAgentsTool {
    pub fn new(
        cli_tool: Arc<CliAgentTool>,
        state: Arc<dyn DynamicCliAgentStore>,
        approval_tx: ApprovalBroker,
    ) -> Self {
        Self {
            cli_tool,
            state,
            approval_tx,
        }
    }

    async fn request_approval(
        &self,
        session_id: &str,
        description: &str,
    ) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: description.to_string(),
                session_id: session_id.to_string(),
                risk_level: RiskLevel::Medium,
                warnings: vec!["This will register a new CLI agent process".to_string()],
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: Default::default(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;
        match tokio::time::timeout(std::time::Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(anyhow::anyhow!("Approval response channel closed")),
            Err(_) => Err(anyhow::anyhow!("Approval request timed out after 300s")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn handle_add(
        &self,
        session_id: &str,
        name: &str,
        command: &str,
        args: Vec<String>,
        description: &str,
        timeout_secs: Option<u64>,
        max_output_chars: Option<usize>,
    ) -> anyhow::Result<String> {
        // Request user approval
        let args_display = args.join(" ");
        let approval_desc = format!(
            "Add CLI agent '{}' ({} {})\n\
             This will register a new CLI-based AI agent that can execute commands.",
            name, command, args_display
        );
        let response = self.request_approval(session_id, &approval_desc).await?;
        match response {
            ApprovalResponse::Deny => {
                return Ok("CLI agent addition denied by user.".to_string());
            }
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {}
        }

        let result = self
            .cli_tool
            .add_agent(
                name,
                command,
                args,
                description,
                timeout_secs,
                max_output_chars,
            )
            .await?;
        info!(name, command, "CLI agent added via manage_cli_agents");
        Ok(result)
    }

    async fn handle_remove(&self, name: &str) -> anyhow::Result<String> {
        self.cli_tool.remove_agent(name).await
    }

    async fn handle_list(&self) -> anyhow::Result<String> {
        let agents = self.cli_tool.list_agents();
        if agents.is_empty() {
            return Ok(
                "No CLI agents registered. Use action='add' to register one, \
                 or install a CLI agent (claude, gemini, codex, copilot, aider) \
                 and restart to auto-discover it."
                    .to_string(),
            );
        }

        let mut output = String::from("Registered CLI agents:\n\n");
        for (name, description, source, enabled) in &agents {
            let status = if *enabled { "enabled" } else { "disabled" };
            output.push_str(&format!("**{}** [{}] ({})\n", name, source, status));
            if !description.is_empty() {
                output.push_str(&format!("  {}\n", description));
            }
            output.push('\n');
        }
        Ok(output)
    }

    async fn handle_enable(&self, name: &str) -> anyhow::Result<String> {
        self.cli_tool.enable_agent(name, true).await
    }

    async fn handle_disable(&self, name: &str) -> anyhow::Result<String> {
        self.cli_tool.enable_agent(name, false).await
    }

    async fn handle_history(&self, limit: usize) -> anyhow::Result<String> {
        let invocations = self.state.get_cli_agent_invocations(limit).await?;
        if invocations.is_empty() {
            return Ok("No CLI agent invocations recorded yet.".to_string());
        }

        let mut output = String::from("Recent CLI agent invocations:\n\n");
        for inv in &invocations {
            let status = match inv.success {
                Some(true) => "success",
                Some(false) => "failed",
                None => "running",
            };
            let duration = inv
                .duration_secs
                .map(|d| format!("{:.1}s", d))
                .unwrap_or_else(|| "—".to_string());
            output.push_str(&format!(
                "- **{}** ({}) [{}] {}\n  {}\n",
                inv.agent_name, inv.started_at, status, duration, inv.prompt_summary
            ));
        }
        Ok(output)
    }
}

#[derive(Deserialize)]
struct ManageCliAgentsArgs {
    action: String,
    name: Option<String>,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    description: Option<String>,
    timeout_secs: Option<u64>,
    max_output_chars: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    _session_id: Option<String>,
}

fn manage_cli_agents_schema() -> Value {
    json!({
        "name": "manage_cli_agents",
        "description": "Manage CLI AI agents (claude/gemini/etc.). Actions: add (requires approval), remove, list, enable, disable, history. Discovery is automatic — use only to add custom agents or inspect invocation history.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "list", "enable", "disable", "history"],
                    "description": "Action"
                },
                "name": {
                    "type": "string",
                    "description": "Name (add/remove/enable/disable)"
                },
                "command": {
                    "type": "string",
                    "description": "Executable; must be installed (add)"
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Extra CLI args (add)"
                },
                "description": {
                    "type": "string",
                    "description": "Description (add)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout secs (add)"
                },
                "max_output_chars": {
                    "type": "integer",
                    "description": "Max output chars (add)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Results to show (history, default 10)"
                }
            },
            "required": ["action"],
            "additionalProperties": false
        }
    })
}

#[async_trait]
impl Tool for ManageCliAgentsTool {
    fn name(&self) -> &str {
        "manage_cli_agents"
    }

    fn description(&self) -> &str {
        "Manage CLI AI agents (claude/gemini/etc.). Actions: add (requires approval), remove, list, enable, disable, history. Discovery is automatic — use only to add custom agents or inspect invocation history."
    }

    fn schema(&self) -> Value {
        manage_cli_agents_schema()
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageCliAgentsArgs = serde_json::from_str(arguments)
            .map_err(|e| anyhow::anyhow!("Invalid arguments: {}", e))?;

        let session_id = args._session_id.as_deref().unwrap_or("");

        match args.action.as_str() {
            "add" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for add action"))?;
                let command = args
                    .command
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'command' is required for add action"))?;
                let description = args.description.as_deref().unwrap_or("");
                self.handle_add(
                    session_id,
                    name,
                    command,
                    args.args,
                    description,
                    args.timeout_secs,
                    args.max_output_chars,
                )
                .await
            }
            "remove" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for remove action"))?;
                self.handle_remove(name).await
            }
            "list" => self.handle_list().await,
            "enable" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for enable action"))?;
                self.handle_enable(name).await
            }
            "disable" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for disable action"))?;
                self.handle_disable(name).await
            }
            "history" => {
                let limit = args.limit.unwrap_or(10);
                self.handle_history(limit).await
            }
            other => Ok(format!(
                "Unknown action '{}'. Valid actions: add, remove, list, enable, disable, history",
                other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_fits_payload_budget() {
        // Pillar C of 2026-06-06-cross-turn-prefix-stability-design.md:
        // admin-tool schemas ride in EVERY provider call; this ceiling is the
        // per-tool payload budget. If you trip this assert by adding features,
        // compress the description text — do not raise the ceiling without
        // updating the Pillar C implementation plan.
        let bytes = serde_json::to_string(&manage_cli_agents_schema())
            .unwrap()
            .len();
        assert!(
            bytes <= 1000,
            "manage_cli_agents schema is {bytes} bytes, budget is 1000"
        );
    }
}
