//! The `manage_mcp` tool: add, list, remove and restart MCP servers at
//! runtime through the [`McpRegistry`].

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use aidaemon_tools::semantics::{
    semantics_for_exact_read_actions, ApprovalCall, OwnerApprovalNeed, ToolCallContext,
    ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolSemantics,
};
use aidaemon_tools::OutcomeTool as Tool;

use super::{McpRegistry, McpServerConfig};

/// The process effect key naming an MCP server, whatever its command.
fn server_process_key(name: &str) -> String {
    format!("mcp-server:{name}")
}

/// Allowed commands for MCP server spawning.
const ALLOWED_COMMANDS: &[&str] = &["npx", "uvx", "node", "python", "python3"];

pub struct ManageMcpTool {
    registry: McpRegistry,
}

impl ManageMcpTool {
    pub fn new(registry: McpRegistry) -> Self {
        Self { registry }
    }

    /// The owner approval for one `add` call. Starting an external process
    /// always asks, even in a trusted session. A call the add path would
    /// refuse (unknown command, missing fields) needs no card: it never runs.
    pub fn owner_approval(arguments: &Value) -> OwnerApprovalNeed {
        let Ok(args) = serde_json::from_value::<ManageMcpArgs>(arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        if args.action != "add" {
            return OwnerApprovalNeed::NotNeeded;
        }
        let (Some(name), Some(command)) = (
            args.name.as_deref().filter(|s| !s.trim().is_empty()),
            args.command.as_deref(),
        ) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        if !ALLOWED_COMMANDS.contains(&command) {
            return OwnerApprovalNeed::NotNeeded;
        }
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!("Add MCP server '{name}' ({command} {})", args.args.join(" ")),
            // The exact arguments the owner approves; a changed argument is a new card.
            subject: json!({"action": "add", "name": name, "command": command, "args": args.args})
                .to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec!["This will download and execute an unverified package and register an external MCP server. Only approve if you trust the source.".to_string()],
        })
    }

    /// The typed effects of one call: starting a server runs its command as a
    /// process. The process may also write files or contact hosts, so the
    /// effects are declared incomplete.
    pub fn effects_for(arguments: &str) -> aidaemon_core::CallEffects {
        use aidaemon_core::{CallEffects, EffectClass, EffectKey};
        let Ok(args) = serde_json::from_str::<ManageMcpArgs>(arguments) else {
            return CallEffects::unknown();
        };
        if args.action == "list" {
            return CallEffects::none();
        }
        let mut effects = CallEffects::unknown();
        if args.action == "add"
            && let Some(command) = args.command.filter(|command| !command.is_empty()) {
                effects
                    .keys
                    .push(EffectKey::named(EffectClass::Process, &command));
            }
        // Every call that can start a server's process names that server, so
        // a declined add, restart or enable binds the others for it.
        if matches!(args.action.as_str(), "add" | "restart" | "enable")
            && let Some(name) = args.name.filter(|name| !name.trim().is_empty()) {
                effects.keys.push(EffectKey::named(
                    EffectClass::Process,
                    &server_process_key(&name),
                ));
            }
        effects
    }

    /// The owner approval for a `restart` or `enable` that would start a
    /// process: the stored command runs again, so the owner sees it and
    /// approves those exact arguments.
    pub async fn respawn_approval(&self, action: &str, name: &str) -> OwnerApprovalNeed {
        let Some((command, args)) = self.registry.pending_launch(action, name).await else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let verb = if action == "restart" {
            "Restart"
        } else {
            "Start"
        };
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!("{verb} MCP server '{name}' ({command} {})", args.join(" ")),
            subject: json!({"action": action, "name": name, "command": command, "args": args})
                .to_string(),
            risk: aidaemon_core::ApprovalRisk::Medium,
            warnings: vec!["This runs the server's stored command as a local process.".to_string()],
        })
    }

    /// Starts a server. The owner was asked by the kernel before this runs.
    async fn handle_add(
        &self,
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

        let config = McpServerConfig {
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
        let servers = self.registry.list_servers_with_status().await?;
        if servers.is_empty() {
            return Ok("No MCP servers registered.".to_string());
        }

        let mut output = String::from("Registered MCP servers:\n\n");
        for server in &servers {
            let status = if server.enabled {
                "enabled"
            } else {
                "disabled"
            };
            let source = if server.db_id.is_some() {
                "dynamic"
            } else {
                "static"
            };
            output.push_str(&format!(
                "**{}** (`{} {}`)\n",
                server.name,
                server.command,
                server.args.join(" ")
            ));
            output.push_str(&format!("  Status: {} ({})\n", status, source));
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

    async fn handle_set_env(&self, name: &str, key: &str, value: &str) -> anyhow::Result<String> {
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
            Err(e) => Ok(format!("Failed to restart MCP server '{}': {}", name, e)),
        }
    }

    async fn handle_enable(&self, name: &str) -> anyhow::Result<String> {
        match self.registry.enable_server(name).await {
            Ok(tool_names) => Ok(format!(
                "MCP server '{}' enabled successfully.\nTools: {}",
                name,
                tool_names.join(", ")
            )),
            Err(e) => Ok(format!("Failed to enable MCP server '{}': {}", name, e)),
        }
    }

    async fn handle_disable(&self, name: &str) -> anyhow::Result<String> {
        match self.registry.disable_server(name).await {
            Ok(()) => Ok(format!("MCP server '{}' disabled successfully.", name)),
            Err(e) => Ok(format!("Failed to disable MCP server '{}': {}", name, e)),
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
}

/// The tool's one description: the schema the model reads and
/// `description()` both use it.
const TOOL_DESCRIPTION: &str =
    "Manage MCP servers at runtime. set_env stores env vars in OS keychain, never in chat.";

fn manage_mcp_schema() -> Value {
    json!({
        "name": "manage_mcp",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove", "set_env", "restart", "enable", "disable"]
                },
                "name": { "type": "string", "description": "Server name" },
                "command": {
                    "type": "string",
                    "description": "Spawn command (add). Allowed: npx, uvx, node, python, python3"
                },
                "args": { "type": "array", "items": { "type": "string" }, "description": "Command args (add)" },
                "key": { "type": "string", "description": "Env var name (set_env)" },
                "value": { "type": "string", "description": "Env var value (set_env); stored in keychain" }
            },
            "required": ["action"],
            "additionalProperties": false
        }
    })
}

#[async_trait]
impl Tool for ManageMcpTool {
    fn name(&self) -> &str {
        "manage_mcp"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        manage_mcp_schema()
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.run(arguments).await.map(ToolCallOutcome::from_output)
    }
}

impl ManageMcpTool {
    /// Run one call. Approval is judged by the kernel before this runs.
    pub async fn run(&self, arguments: &str) -> anyhow::Result<String> {
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
                self.handle_add(name, command, args.args)
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
            other => Ok(format!(
                "Unknown action '{}'. Valid actions: add, list, remove, set_env, restart, enable, disable",
                other
            )),
        }
    }
}

#[async_trait]
impl ToolSemantics for ManageMcpTool {
    /// Starting a server asks the owner, whatever the session. The dispatcher
    /// runs the call only with that answer; a decline is recorded by the kernel.
    async fn owner_approval_request(
        &self,
        call: &ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        if let Ok(args) = serde_json::from_value::<ManageMcpArgs>(call.arguments.clone())
            && matches!(args.action.as_str(), "restart" | "enable") {
                let Some(name) = args.name.as_deref().filter(|name| !name.trim().is_empty()) else {
                    return OwnerApprovalNeed::NotNeeded;
                };
                return self.respawn_approval(&args.action, name).await;
            }
        Self::owner_approval(call.arguments)
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        Self::effects_for(arguments)
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        semantics_for_exact_read_actions(arguments, &["list"], ToolMutationEffects::CONFIGURATION)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition_declares_process_but_does_not_claim_exhaustive_effects() {
        let effects = ManageMcpTool::effects_for(
            r#"{"action":"add","name":"synthetic","command":"node","args":["server.js"]}"#,
        );
        assert_eq!(
            effects.keys,
            vec![
                aidaemon_core::EffectKey::named(aidaemon_core::EffectClass::Process, "node"),
                aidaemon_core::EffectKey::named(
                    aidaemon_core::EffectClass::Process,
                    "mcp-server:synthetic"
                ),
            ]
        );
        for action in ["restart", "enable"] {
            let effects = ManageMcpTool::effects_for(&format!(
                r#"{{"action":"{action}","name":"synthetic"}}"#
            ));
            assert_eq!(
                effects.keys,
                vec![aidaemon_core::EffectKey::named(
                    aidaemon_core::EffectClass::Process,
                    "mcp-server:synthetic"
                )],
                "{action}"
            );
            assert!(!effects.complete);
        }
        assert!(!effects.complete);
        assert_eq!(
            ManageMcpTool::effects_for(r#"{"action":"list"}"#),
            aidaemon_core::CallEffects::none()
        );
    }

    #[test]
    fn approval_subject_binds_exact_arguments_and_ignores_injected_context() {
        let args = json!({"action":"add","name":"synthetic","command":"node","args":["server.js"]});
        let OwnerApprovalNeed::Ask(first) = ManageMcpTool::owner_approval(&args) else {
            panic!("an add must ask");
        };
        let mut injected = args.clone();
        injected["_session_id"] = json!("synthetic-session");
        let OwnerApprovalNeed::Ask(same) = ManageMcpTool::owner_approval(&injected) else {
            panic!("an add must ask");
        };
        assert_eq!(first, same);
        let mut changed = args.clone();
        changed["args"] = json!(["other.js"]);
        let OwnerApprovalNeed::Ask(other) = ManageMcpTool::owner_approval(&changed) else {
            panic!("an add must ask");
        };
        assert_ne!(first.subject, other.subject);
        assert!(matches!(
            ManageMcpTool::owner_approval(
                &json!({"action":"add","name":"synthetic","command":"sh"})
            ),
            OwnerApprovalNeed::NotNeeded
        ));
    }

    #[test]
    fn schema_fits_payload_budget() {
        // Pillar C of 2026-06-06-cross-turn-prefix-stability-design.md:
        // admin-tool schemas ride in EVERY provider call; this ceiling is the
        // per-tool payload budget. If you trip this assert by adding features,
        // compress the description text — do not raise the ceiling without
        // updating the Pillar C implementation plan.
        let bytes = serde_json::to_string(&manage_mcp_schema()).unwrap().len();
        assert!(
            bytes <= 850,
            "manage_mcp schema is {bytes} bytes, budget is 850"
        );
    }
}
