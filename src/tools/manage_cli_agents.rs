use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::tools::cli_agent::CliAgentTool;
use crate::traits::{
    semantics_for_exact_read_actions, DynamicCliAgentStore, OwnerApprovalNeed, Tool,
    ToolCallContext, ToolCallSemantics, ToolCapabilities, ToolMutationEffects,
};

pub struct ManageCliAgentsTool {
    cli_tool: Arc<CliAgentTool>,
    state: Arc<dyn DynamicCliAgentStore>,
}

impl ManageCliAgentsTool {
    pub fn new(cli_tool: Arc<CliAgentTool>, state: Arc<dyn DynamicCliAgentStore>) -> Self {
        Self { cli_tool, state }
    }

    /// The owner approval for one call. Registering a CLI agent always asks,
    /// even in a trusted session; a call the add path refuses (no name or
    /// command, unreadable arguments) never runs and needs no card.
    pub(crate) fn owner_approval(arguments: &Value) -> OwnerApprovalNeed {
        let Ok(args) = serde_json::from_value::<ManageCliAgentsArgs>(arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        if args.action != "add" {
            return OwnerApprovalNeed::NotNeeded;
        }
        let (Some(name), Some(command)) = (args.name.as_deref(), args.command.as_deref()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!(
                "Add CLI agent '{}' ({} {})\n\
                 This will register a new CLI-based AI agent that can execute commands.",
                name,
                command,
                args.args.join(" ")
            ),
            // The exact arguments the owner approves; a changed argument is a new card.
            subject: json!({
                "action": "add",
                "name": name,
                "command": command,
                "args": args.args,
                "description": args.description.as_deref().unwrap_or(""),
                "timeout_secs": args.timeout_secs,
                "max_output_chars": args.max_output_chars,
            })
            .to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec!["This will register a new CLI agent process".to_string()],
        })
    }

    /// The typed effects of one call: registering an agent names the program
    /// it will run. That program may also write files or contact hosts, so
    /// the effects are declared incomplete. Listing and history are reads.
    pub(crate) fn effects_for(arguments: &str) -> aidaemon_core::CallEffects {
        use aidaemon_core::{CallEffects, EffectClass, EffectKey};
        let Ok(args) = serde_json::from_str::<ManageCliAgentsArgs>(arguments) else {
            return CallEffects::unknown();
        };
        if matches!(args.action.as_str(), "list" | "history") {
            return CallEffects::none();
        }
        let mut effects = CallEffects::unknown();
        if args.action == "add" {
            if let Some(command) = args.command.filter(|command| !command.is_empty()) {
                effects
                    .keys
                    .push(EffectKey::named(EffectClass::Process, &command));
            }
        }
        effects
    }

    #[allow(clippy::too_many_arguments)]
    async fn handle_add(
        &self,
        name: &str,
        command: &str,
        args: Vec<String>,
        description: &str,
        timeout_secs: Option<u64>,
        max_output_chars: Option<usize>,
    ) -> anyhow::Result<String> {
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
}

/// The tool's one description: the schema the model reads and
/// `description()` both use it.
const TOOL_DESCRIPTION: &str = "Manage CLI AI agents (claude/gemini/etc.). Actions: add (requires approval), remove, list, enable, disable, history. Discovery is automatic — use only to add custom agents or inspect invocation history.";

fn manage_cli_agents_schema() -> Value {
    json!({
        "name": "manage_cli_agents",
        "description": TOOL_DESCRIPTION,
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
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        manage_cli_agents_schema()
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let facts = facts.clone();

        self.run(arguments, &facts)
            .await
            .map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl ManageCliAgentsTool {
    /// Run one call with the dispatcher's `facts`.
    pub(crate) async fn run(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<String> {
        let args: ManageCliAgentsArgs = serde_json::from_str(arguments)
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
                let description = args.description.as_deref().unwrap_or("");
                if facts.kernel_approved.is_none() {
                    anyhow::bail!(
                        "manage_cli_agents add requires approval through the agent kernel; it \
                         was called outside an approved agent call"
                    );
                }
                self.handle_add(
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

#[async_trait]
impl crate::traits::ToolSemantics for ManageCliAgentsTool {
    /// Registering an agent asks the owner, whatever the session. The
    /// dispatcher runs the call only with that answer; a decline is recorded
    /// by the kernel.
    async fn owner_approval_request(
        &self,
        call: &crate::traits::ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        Self::owner_approval(call.arguments)
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        Self::effects_for(arguments)
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        semantics_for_exact_read_actions(
            arguments,
            &["list", "history"],
            ToolMutationEffects::CONFIGURATION,
        )
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
    fn addition_asks_with_exact_arguments_and_declares_its_process() {
        let args =
            json!({"action":"add","name":"synthetic","command":"synthetic-cli","args":["--quiet"]});
        let OwnerApprovalNeed::Ask(first) = ManageCliAgentsTool::owner_approval(&args) else {
            panic!("an add must ask");
        };
        assert!(first
            .summary
            .starts_with("Add CLI agent 'synthetic' (synthetic-cli --quiet)"));
        assert_eq!(first.risk, aidaemon_core::ApprovalRisk::High);
        let mut changed = args.clone();
        changed["args"] = json!(["--loud"]);
        let OwnerApprovalNeed::Ask(other) = ManageCliAgentsTool::owner_approval(&changed) else {
            panic!("an add must ask");
        };
        assert_ne!(first.subject, other.subject);
        for no_card in [
            json!({"action":"add","name":"synthetic"}),
            json!({"action":"list"}),
            json!({"action":"history"}),
            json!({"action":"remove","name":"synthetic"}),
        ] {
            assert!(
                matches!(
                    ManageCliAgentsTool::owner_approval(&no_card),
                    OwnerApprovalNeed::NotNeeded
                ),
                "{no_card}"
            );
        }

        let effects = ManageCliAgentsTool::effects_for(&args.to_string());
        assert_eq!(
            effects.keys,
            vec![aidaemon_core::EffectKey::named(
                aidaemon_core::EffectClass::Process,
                "synthetic-cli"
            )]
        );
        assert!(!effects.complete);
        assert_eq!(
            ManageCliAgentsTool::effects_for(r#"{"action":"history"}"#),
            aidaemon_core::CallEffects::none()
        );
        assert!(!ManageCliAgentsTool::effects_for(r#"{"action":"remove","name":"x"}"#).complete);
    }

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
