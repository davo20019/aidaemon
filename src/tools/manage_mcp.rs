use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::info;

use crate::mcp::McpRegistry;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::traits::Tool;
use crate::types::ApprovalResponse;

/// Allowed commands for MCP server spawning.
const ALLOWED_COMMANDS: &[&str] = &["npx", "uvx", "node", "python", "python3"];

pub struct ManageMcpTool {
    registry: McpRegistry,
    approval_tx: mpsc::Sender<ApprovalRequest>,
}

impl ManageMcpTool {
    pub fn new(
        registry: McpRegistry,
        approval_tx: mpsc::Sender<ApprovalRequest>,
    ) -> Self {
        Self {
            registry,
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
                warnings: vec![
                    "This will spawn an external MCP server process".to_string(),
                ],
                permission_mode: PermissionMode::Default,
                response_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;
        match tokio::time::timeout(std::time::Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                tracing::warn!(description, "Approval response channel closed");
                Ok(ApprovalResponse::Deny)
            }
            Err(_) => {
                tracing::warn!(description, "Approval request timed out (300s), auto-denying");
                Ok(ApprovalResponse::Deny)
            }
        }
    }

    async fn handle_add(
        &self,
        session_id: &str,
        name: &str,
        command: &str,
        args: Vec<String>,
    ) -> anyhow::Result<String> {
        // Validate command against whitelist
        if !ALLOWED_COMMANDS.contains(&command) {
            return Ok(format!(
                "Command '{}' is not allowed. Allowed commands: {}",
                command,
                ALLOWED_COMMANDS.join(", ")
            ));
        }

        // Request user approval with unverified package warning
        let args_display = args.join(" ");
        let description = format!(
            "Add MCP server '{}' ({} {})\n\
             \u{26a0} WARNING: This will download and execute an unverified package. \
             Only approve if you trust the source.",
            name, command, args_display
        );
        let response = self.request_approval(session_id, &description).await?;
        match response {
            ApprovalResponse::Deny => {
                return Ok("MCP server addition denied by user.".to_string());
            }
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {}
        }

        let config = crate::config::McpServerConfig {
            command: command.to_string(),
            args,
            env: std::collections::HashMap::new(),
        };

        match self.registry.add_server(name, config, true).await {
            Ok(tool_names) => {
                info!(server = name, tools = ?tool_names, "MCP server added");
                Ok(format!(
                    "MCP server '{}' added successfully.\nRegistered tools: {}",
                    name,
                    tool_names.join(", ")
                ))
            }
            Err(e) => Ok(format!(
                "Failed to add MCP server '{}': {}\n\
                 Possible fixes:\n\
                 - Install the package: npm install -g <package> or pip install <package>\n\
                 - Check the command and arguments are correct\n\
                 - Try running the command manually to see detailed errors",
                name, e
            )),
        }
    }

    async fn handle_list(&self) -> anyhow::Result<String> {
        let servers = self.registry.list_servers().await;
        if servers.is_empty() {
            return Ok("No MCP servers registered.".to_string());
        }

        let mut output = String::from("Registered MCP servers:\n\n");
        for server in &servers {
            output.push_str(&format!("**{}** (`{} {}`)\n", server.name, server.command, server.args.join(" ")));
            output.push_str(&format!("  Tools: {}\n", server.tool_names.join(", ")));
            if !server.env_keys.is_empty() {
                output.push_str(&format!("  Env keys: {}\n", server.env_keys.join(", ")));
            }
            output.push_str(&format!("  Triggers: {}\n", server.triggers.join(", ")));
            output.push('\n');
        }
        Ok(output)
    }

    async fn handle_remove(&self, name: &str) -> anyhow::Result<String> {
        match self.registry.remove_server(name).await {
            Ok(()) => Ok(format!("MCP server '{}' removed successfully.", name)),
            Err(e) => Ok(format!("Failed to remove MCP server '{}': {}", name, e)),
        }
    }

    async fn handle_set_env(
        &self,
        name: &str,
        key: &str,
        value: &str,
    ) -> anyhow::Result<String> {
        match self.registry.set_server_env(name, key, value).await {
            Ok(()) => Ok(format!(
                "Environment variable '{}' stored securely in OS keychain for server '{}'.\n\
                 Use the 'restart' action to apply the new configuration.",
                key, name
            )),
            Err(e) => Ok(format!(
                "Failed to store env var '{}' for server '{}': {}",
                key, name, e
            )),
        }
    }

    async fn handle_restart(&self, name: &str) -> anyhow::Result<String> {
        match self.registry.restart_server(name).await {
            Ok(tool_names) => Ok(format!(
                "MCP server '{}' restarted successfully.\nTools: {}",
                name,
                tool_names.join(", ")
            )),
            Err(e) => Ok(format!(
                "Failed to restart MCP server '{}': {}",
                name, e
            )),
        }
    }
}

#[derive(Deserialize)]
struct ManageMcpArgs {
    action: String,
    name: Option<String>,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    key: Option<String>,
    value: Option<String>,
    #[serde(default)]
    _session_id: String,
}

#[async_trait]
impl Tool for ManageMcpTool {
    fn name(&self) -> &str {
        "manage_mcp"
    }

    fn description(&self) -> &str {
        "Add, remove, list, and configure MCP (Model Context Protocol) servers at runtime"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_mcp",
            "description": "Manage MCP servers dynamically. Actions:\n\
                - add: Add and start a new MCP server (requires name, command, args)\n\
                - list: List all registered MCP servers and their tools\n\
                - remove: Remove an MCP server (requires name)\n\
                - set_env: Store an API key or env var for a server in the OS keychain (requires name, key, value)\n\
                - restart: Restart a server with fresh env from keychain (requires name)",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "remove", "set_env", "restart"],
                        "description": "The action to perform"
                    },
                    "name": {
                        "type": "string",
                        "description": "Server name (required for add, remove, set_env, restart)"
                    },
                    "command": {
                        "type": "string",
                        "description": "Command to spawn the server (required for add). Allowed: npx, uvx, node, python, python3"
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Arguments for the server command (for add)"
                    },
                    "key": {
                        "type": "string",
                        "description": "Environment variable name (for set_env)"
                    },
                    "value": {
                        "type": "string",
                        "description": "Environment variable value (for set_env). Stored in OS keychain, never in chat."
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageMcpArgs = serde_json::from_str(arguments)
            .map_err(|e| anyhow::anyhow!("Invalid arguments: {}", e))?;

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
                self.handle_add(&args._session_id, name, command, args.args)
                    .await
            }
            "list" => self.handle_list().await,
            "remove" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for remove action"))?;
                self.handle_remove(name).await
            }
            "set_env" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for set_env action"))?;
                let key = args
                    .key
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'key' is required for set_env action"))?;
                let value = args
                    .value
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'value' is required for set_env action"))?;
                self.handle_set_env(name, key, value).await
            }
            "restart" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for restart action"))?;
                self.handle_restart(name).await
            }
            other => Ok(format!(
                "Unknown action '{}'. Valid actions: add, list, remove, set_env, restart",
                other
            )),
        }
    }
}
