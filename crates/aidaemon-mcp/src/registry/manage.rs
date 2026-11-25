//! The `manage_mcp` tool: add, list, remove and restart MCP servers at
//! runtime through the [`McpRegistry`].

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::semantics::{
    ApprovalCall, OwnerApprovalNeed, ToolCallContext, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolMutationEffects, ToolSemantics, semantics_for_exact_read_actions,
};

use super::running::ServerEnvBinding;
use super::{McpRegistry, McpServerConfig};

/// The process effect key naming an MCP server, whatever its command.
pub(super) fn server_process_key(name: &str) -> String {
    format!("mcp-server:{name}")
}

/// Allowed commands for MCP server spawning.
const ALLOWED_COMMANDS: &[&str] = &["npx", "uvx", "node", "python", "python3"];

pub struct ManageMcpTool {
    registry: McpRegistry,
    /// The project servers the owner was asked to start, as the card showed
    /// them, by (project id, name): a start runs exactly this entry, and a
    /// `.mcp.json` changed after the card is refused, not run.
    approved_starts: std::sync::Mutex<
        std::collections::HashMap<(String, String), super::project::ProjectServer>,
    >,
}

impl ManageMcpTool {
    pub fn new(registry: McpRegistry) -> Self {
        Self {
            registry,
            approved_starts: std::sync::Mutex::default(),
        }
    }

    fn remember_carded_start(
        &self,
        project: &aidaemon_core::ProjectRef,
        server: &super::project::ProjectServer,
    ) {
        if let Ok(mut approved) = self.approved_starts.lock() {
            approved.insert((project.id.clone(), server.name.clone()), server.clone());
        }
    }

    fn carded_start(&self, project_id: &str, name: &str) -> Option<super::project::ProjectServer> {
        self.approved_starts
            .lock()
            .ok()?
            .remove(&(project_id.to_string(), name.to_string()))
    }

    /// The project's declared server `name`, when the turn works in a
    /// project with a folder.
    fn project_server(
        project: Option<&aidaemon_core::ProjectRef>,
        name: &str,
    ) -> Result<(aidaemon_core::ProjectRef, super::project::ProjectServer), String> {
        let project = project
            .ok_or("This conversation does not work in a project, so it has no project servers.")?;
        let root = project
            .root
            .as_deref()
            .ok_or("This project has no folder, so it declares no servers.")?;
        let servers = super::project::read_project_servers(root, &project.id)?;
        let server = servers
            .into_iter()
            .find(|server| server.name == name)
            .ok_or_else(|| {
                format!(
                    "{} declares no server named '{name}' in {}.",
                    project.name,
                    super::project::PROJECT_MCP_FILE
                )
            })?;
        Ok((project.clone(), server))
    }

    /// The owner approval for one `project_start`: the exact command the
    /// repository declares, for this project. A changed entry is a new card.
    /// A call the start would refuse needs no card: it never runs.
    pub fn project_start_approval(
        project: Option<&aidaemon_core::ProjectRef>,
        name: Option<&str>,
    ) -> OwnerApprovalNeed {
        Self::project_start_judgment(project, name).0
    }

    /// [`Self::project_start_approval`] with the entry the card shows, read
    /// once, so what is remembered as approved is exactly what was shown.
    fn project_start_judgment(
        project: Option<&aidaemon_core::ProjectRef>,
        name: Option<&str>,
    ) -> (
        OwnerApprovalNeed,
        Option<(aidaemon_core::ProjectRef, super::project::ProjectServer)>,
    ) {
        Self::project_start_card(project, name)
    }

    fn project_start_card(
        project: Option<&aidaemon_core::ProjectRef>,
        name: Option<&str>,
    ) -> (
        OwnerApprovalNeed,
        Option<(aidaemon_core::ProjectRef, super::project::ProjectServer)>,
    ) {
        // A blank name is refused at the start too (`handle_project_start`),
        // and `.mcp.json` never yields a usable server under one.
        let Some(name) = name.filter(|name| !name.trim().is_empty()) else {
            return (OwnerApprovalNeed::NotNeeded, None);
        };
        // Fails closed: a start judged without its project still asks, so a
        // judgment that cannot see what will run never lets it run unasked.
        let Ok((project, server)) = Self::project_server(project, name) else {
            return (OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
                card: None,
                effects: Vec::new(),
                summary: format!("Start the project's MCP server '{name}'"),
                subject: json!({"action": "project_start", "name": name}).to_string(),
                risk: aidaemon_core::ApprovalRisk::High,
                warnings: vec![
                    "The project's repository chose this server's command. Only approve if you trust the repository.".to_string(),
                ],
            }), None);
        };
        // What the start refuses never reaches the owner: it never runs.
        if server.unusable.is_some() || !ALLOWED_COMMANDS.contains(&server.command.as_str()) {
            return (OwnerApprovalNeed::NotNeeded, None);
        }
        let env_keys: Vec<&String> = server.env.keys().collect();
        let need = OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!(
                "Start MCP server '{}' that project {} declares ({} {})",
                server.name,
                project.name,
                server.command,
                server.args.join(" ")
            ),
            subject: json!({
                "action": "project_start",
                "project": project.id,
                "name": server.name,
                "command": server.command,
                "args": server.args,
                "env": server.env,
            })
            .to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec![format!(
                "The project's repository chose this command{}. It runs as a local process for \
                 this project's conversations until the daemon restarts. Only approve if you \
                 trust the repository.",
                if env_keys.is_empty() {
                    String::new()
                } else {
                    format!(
                        " and its environment ({})",
                        env_keys
                            .iter()
                            .map(|k| k.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            )],
        });
        (need, Some((project, server)))
    }

    async fn handle_project_list(
        &self,
        project: Option<&aidaemon_core::ProjectRef>,
    ) -> anyhow::Result<String> {
        let Some(project) = project else {
            return Ok(
                "This conversation does not work in a project, so it has no project servers."
                    .to_string(),
            );
        };
        let Some(root) = project.root.as_deref() else {
            return Ok(format!(
                "{} has no folder, so it declares no servers.",
                project.name
            ));
        };
        let servers = match super::project::read_project_servers(root, &project.id) {
            Ok(servers) => servers,
            Err(why) => return Ok(format!("Could not read {}'s servers: {why}", project.name)),
        };
        if servers.is_empty() {
            return Ok(format!(
                "{} declares no MCP servers (no {} at its root).",
                project.name,
                super::project::PROJECT_MCP_FILE
            ));
        }
        let running: Vec<String> = self
            .registry
            .list_servers()
            .await
            .into_iter()
            .map(|info| info.name)
            .collect();
        let mut out = format!(
            "{} declares these MCP servers in {} (start one with project_start; the owner approves its command):",
            project.name,
            super::project::PROJECT_MCP_FILE
        );
        for server in &servers {
            let state = if running.contains(&server.registered_name) {
                "running".to_string()
            } else if let Some(why) = &server.unusable {
                format!("cannot start: {why}")
            } else if !ALLOWED_COMMANDS.contains(&server.command.as_str()) {
                format!(
                    "cannot start: '{}' is not an allowed command ({})",
                    server.command,
                    ALLOWED_COMMANDS.join(", ")
                )
            } else {
                "not started".to_string()
            };
            out.push_str(&format!(
                "\n- {} (`{} {}`): {state}",
                server.name,
                server.command,
                server.args.join(" ")
            ));
        }
        Ok(out)
    }

    /// Starts a project's server. The owner was asked by the kernel first.
    async fn handle_project_start(
        &self,
        project: Option<&aidaemon_core::ProjectRef>,
        name: &str,
    ) -> anyhow::Result<String> {
        if name.trim().is_empty() || name.trim() != name {
            return Ok(
                "A project server is started by its exact name; this one is blank or padded."
                    .to_string(),
            );
        }
        let (project, server) = match Self::project_server(project, name) {
            Ok(found) => found,
            Err(why) => return Ok(why),
        };
        if let Some(why) = &server.unusable {
            return Ok(format!("'{}' cannot be started: {why}.", server.name));
        }
        if !ALLOWED_COMMANDS.contains(&server.command.as_str()) {
            return Ok(format!(
                "'{}' cannot be started: '{}' is not an allowed command. Allowed commands: {}",
                server.name,
                server.command,
                ALLOWED_COMMANDS.join(", ")
            ));
        }
        let running = self.registry.list_servers().await;
        if running
            .iter()
            .any(|info| info.name == server.registered_name)
        {
            return Ok(format!(
                "'{}' is already running for {}.",
                server.name, project.name
            ));
        }
        // Run what the owner's card showed, never a re-read that changed
        // since: no card for this entry, or a different entry now, refuses.
        match self.carded_start(&project.id, &server.name) {
            Some(carded) if carded == server => {}
            Some(_) => {
                return Ok(format!(
                    "'{}' was not started: {} changed after the approval card. Start it again to \
                     see its current command.",
                    server.name,
                    super::project::PROJECT_MCP_FILE
                ));
            }
            None => {
                return Ok(format!(
                    "'{}' was not started: starting a project's server needs the owner's \
                     approval of its exact command.",
                    server.name
                ));
            }
        }
        // Not saved: a project server lasts until the daemon restarts.
        let tools = self
            .registry
            .add_server(&server.registered_name, server.config(), false)
            .await?;
        info!(
            server = %server.registered_name,
            project = %project.id,
            tools = tools.len(),
            "Project MCP server started"
        );
        Ok(format!(
            "Started '{}' for {} as {}. Its tools ({}) are offered in this project's conversations from the next message.",
            server.name,
            project.name,
            server.registered_name,
            tools.join(", ")
        ))
    }

    async fn handle_project_stop(
        &self,
        project: Option<&aidaemon_core::ProjectRef>,
        name: &str,
    ) -> anyhow::Result<String> {
        let Some(project) = project else {
            return Ok(
                "This conversation does not work in a project, so it has no project servers."
                    .to_string(),
            );
        };
        let registered = super::project::registered_name(&project.id, name);
        let running = self.registry.list_servers().await;
        if !running.iter().any(|info| info.name == registered) {
            return Ok(format!("'{name}' is not running for {}.", project.name));
        }
        self.registry.remove_server(&registered).await?;
        Ok(format!("Stopped '{name}' for {}.", project.name))
    }

    /// The owner approval for one `add` call. Starting an external process
    /// always asks, even in a trusted session. A call the add path would
    /// refuse (unknown command, missing fields) needs no card: it never runs.
    pub fn owner_approval(arguments: &Value) -> OwnerApprovalNeed {
        let Ok(args) = serde_json::from_value::<ManageMcpArgs>(arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        if let Some(need) = Self::binding_or_removal_approval(&args) {
            return need;
        }
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

    /// The owner approval for `set_env` (it hands a stored secret to a
    /// process the agent configured) and `remove` (it deletes a server the
    /// owner set up, with its stored secrets). `None` for other actions.
    fn binding_or_removal_approval(args: &ManageMcpArgs) -> Option<OwnerApprovalNeed> {
        let name = args.name.as_deref().filter(|name| !name.trim().is_empty());
        let request = match (args.action.as_str(), name) {
            ("set_env", Some(name)) => {
                let key = args.key.as_deref().filter(|key| !key.trim().is_empty())?;
                // A value sent as an argument is refused before anything runs.
                if args.value.is_some() {
                    return Some(OwnerApprovalNeed::NotNeeded);
                }
                let secret = McpRegistry::env_secret_key(name, key);
                aidaemon_core::OwnerApprovalRequest {
                    card: None,
                    effects: Vec::new(),
                    summary: format!(
                        "Give MCP server '{name}' the stored secret `{secret}` as its environment \
                         variable {key}"
                    ),
                    subject: json!({"action": "set_env", "name": name, "key": key}).to_string(),
                    risk: aidaemon_core::ApprovalRisk::High,
                    warnings: vec![
                        "The server's process can read and send this secret anywhere. Only \
                         approve for a server you trust with it."
                            .to_string(),
                    ],
                }
            }
            ("remove", Some(name)) => aidaemon_core::OwnerApprovalRequest {
                card: None,
                effects: Vec::new(),
                summary: format!(
                    "Remove MCP server '{name}', its saved settings and the secrets stored for it"
                ),
                subject: json!({"action": "remove", "name": name}).to_string(),
                risk: aidaemon_core::ApprovalRisk::Medium,
                warnings: Vec::new(),
            },
            ("set_env" | "remove", None) => return Some(OwnerApprovalNeed::NotNeeded),
            _ => return None,
        };
        Some(OwnerApprovalNeed::Ask(request))
    }

    /// The typed effects of one call: starting a server runs its command as a
    /// process. The process may also write files or contact hosts, so the
    /// effects are declared incomplete.
    pub fn effects_for(arguments: &str) -> aidaemon_core::CallEffects {
        use aidaemon_core::{CallEffects, EffectClass, EffectKey};
        let Ok(args) = serde_json::from_str::<ManageMcpArgs>(arguments) else {
            return CallEffects::unknown();
        };
        if matches!(args.action.as_str(), "list" | "project_list") {
            return CallEffects::none();
        }
        // A project server's command is the repository's: unknown here, so
        // the effects stay incomplete; its process key names the server.
        if args.action == "project_start"
            && let Some(name) = args.name.clone().filter(|name| !name.trim().is_empty())
        {
            let mut effects = CallEffects::unknown();
            effects.keys.push(EffectKey::named(
                EffectClass::Process,
                &server_process_key(&format!("project:{name}")),
            ));
            return effects;
        }
        let mut effects = CallEffects::unknown();
        if args.action == "add"
            && let Some(command) = args.command.filter(|command| !command.is_empty())
        {
            effects
                .keys
                .push(EffectKey::named(EffectClass::Process, &command));
        }
        // Every call that can start a server's process names that server, so
        // a declined add, restart or enable binds the others for it.
        if matches!(args.action.as_str(), "add" | "restart" | "enable")
            && let Some(name) = args.name.filter(|name| !name.trim().is_empty())
        {
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
            read_only_tools: Vec::new(),
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

    /// Binds env var `key` to the value the owner stored under
    /// `mcp_<server>_<KEY>`. The value itself is never an argument.
    async fn handle_set_env(
        &self,
        name: &str,
        key: &str,
        value_sent_as_argument: bool,
    ) -> anyhow::Result<String> {
        let secret_key = McpRegistry::env_secret_key(name, key);
        if value_sent_as_argument {
            return Ok(format!(
                "Nothing was stored: secret values are not accepted as tool arguments. \
                 Ask the owner to run `aidaemon keychain set {}` in their own terminal, \
                 then call set_env again with only name and key.",
                secret_key
            ));
        }
        match self.registry.set_server_env(name, key).await {
            Ok(ServerEnvBinding::Bound { secret_key }) => Ok(format!(
                "Environment variable '{}' of server '{}' now reads `{}` from the secret store.\n\
                 Use the 'restart' action to apply the new configuration.",
                key, name, secret_key
            )),
            Ok(ServerEnvBinding::MissingSecret { secret_key }) => Ok(format!(
                "No value for '{}' in the secret store, so nothing changed. Ask the owner to \
                 run `aidaemon keychain set {}` in their own terminal (never paste it in chat), \
                 then call set_env again.",
                key, secret_key
            )),
            Err(e) => Ok(format!(
                "Failed to set env var '{}' for server '{}': {}",
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
    /// No longer a parameter: secrets never travel as tool arguments. Kept
    /// only to notice a model that still sends one; the value is dropped
    /// unread.
    value: Option<serde::de::IgnoredAny>,
}

/// The tool's one description: the schema the model reads and
/// `description()` both use it.
const TOOL_DESCRIPTION: &str = "Manage MCP servers at runtime. set_env binds an env var to `mcp_<server>_<KEY>`, which the owner stores with `aidaemon keychain set`; values are never args. In a project conversation, project_list shows the servers its .mcp.json declares and project_start/project_stop run one for this project only.";

fn manage_mcp_schema() -> Value {
    json!({
        "name": "manage_mcp",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove", "set_env", "restart", "enable", "disable", "project_list", "project_start", "project_stop"]
                },
                "name": { "type": "string", "description": "Server name (for project_start/project_stop, its name in .mcp.json)" },
                "command": {
                    "type": "string",
                    "description": "Spawn command (add). Allowed: npx, uvx, node, python, python3"
                },
                "args": { "type": "array", "items": { "type": "string" }, "description": "Command args (add)" },
                "key": { "type": "string", "description": "Env var name (set_env)" }
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
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.run_in(arguments, facts.project.as_ref())
            .await
            .map(ToolCallOutcome::from_output)
    }
}

impl ManageMcpTool {
    /// Run one call. Approval is judged by the kernel before this runs.
    pub async fn run(&self, arguments: &str) -> anyhow::Result<String> {
        self.run_in(arguments, None).await
    }

    /// Run one call on a turn in `project` (the project actions need one).
    pub async fn run_in(
        &self,
        arguments: &str,
        project: Option<&aidaemon_core::ProjectRef>,
    ) -> anyhow::Result<String> {
        let args: ManageMcpArgs = serde_json::from_str(arguments)
            .map_err(|e| anyhow::anyhow!("Invalid arguments: {}", e))?;

        match args.action.as_str() {
            "project_list" => self.handle_project_list(project).await,
            "project_start" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for project_start"))?;
                self.handle_project_start(project, name).await
            }
            "project_stop" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for project_stop"))?;
                self.handle_project_stop(project, name).await
            }
            "add" => {
                let name = args
                    .name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' is required for add action"))?;
                let command = args
                    .command
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'command' is required for add action"))?;
                self.handle_add(name, command, args.args).await
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
                self.handle_set_env(name, key, args.value.is_some()).await
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
                "Unknown action '{}'. Valid actions: add, list, remove, set_env, restart, enable, disable, project_list, project_start, project_stop",
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
        facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        if let Ok(args) = serde_json::from_value::<ManageMcpArgs>(call.arguments.clone())
            && args.action == "project_start"
        {
            let (need, carded) =
                Self::project_start_judgment(facts.project.as_ref(), args.name.as_deref());
            if let Some((project, server)) = carded {
                self.remember_carded_start(&project, &server);
            }
            return need;
        }
        if let Ok(args) = serde_json::from_value::<ManageMcpArgs>(call.arguments.clone())
            && matches!(args.action.as_str(), "restart" | "enable")
        {
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
        semantics_for_exact_read_actions(
            arguments,
            &["list", "project_list"],
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

    fn project_with(mcp_json: Option<&str>) -> (aidaemon_core::ProjectRef, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        if let Some(text) = mcp_json {
            std::fs::write(dir.path().join(".mcp.json"), text).unwrap();
        }
        let project = aidaemon_core::ProjectRef::new(
            "project-0a1b2c3d-0000-4000-8000-000000000001",
            "Acme Site",
            Some(dir.path().to_path_buf()),
        )
        .unwrap();
        (project, dir)
    }

    #[test]
    fn binding_a_secret_or_removing_a_server_asks() {
        let OwnerApprovalNeed::Ask(bind) = ManageMcpTool::owner_approval(
            &json!({"action": "set_env", "name": "docs", "key": "API_TOKEN"}),
        ) else {
            panic!("set_env must ask");
        };
        assert!(
            bind.summary.contains("mcp_docs_API_TOKEN"),
            "{}",
            bind.summary
        );
        assert_eq!(bind.risk, aidaemon_core::ApprovalRisk::High);
        assert!(matches!(
            ManageMcpTool::owner_approval(&json!({"action": "remove", "name": "docs"})),
            OwnerApprovalNeed::Ask(_)
        ));
        // Refused before anything runs: no card.
        assert!(matches!(
            ManageMcpTool::owner_approval(
                &json!({"action": "set_env", "name": "docs", "key": "K", "value": "v"})
            ),
            OwnerApprovalNeed::NotNeeded
        ));
    }

    /// The kernel's judgment of one call, as the dispatcher asks it before
    /// running the call (the card it would show).
    async fn judge(
        tool: &ManageMcpTool,
        project: &aidaemon_core::ProjectRef,
        arguments: &str,
    ) -> OwnerApprovalNeed {
        let arguments: Value = serde_json::from_str(arguments).unwrap();
        let facts = ToolCallContext {
            project: Some(project.clone()),
            ..ToolCallContext::default()
        };
        tool.owner_approval_request(&ApprovalCall::as_sent(&arguments), &facts)
            .await
    }

    const DOCS: &str = r#"{"mcpServers": {
        "docs": {"command": "npx", "args": ["-y", "docs-mcp"], "env": {"MODE": "ro"}},
        "shell": {"command": "bash", "args": ["-c", "evil"]},
        "keyed": {"command": "uvx", "args": ["t"], "env": {"TOKEN": "${API_TOKEN}"}}
    }}"#;

    #[test]
    fn starting_a_project_server_asks_for_its_exact_command() {
        let (project, dir) = project_with(Some(DOCS));
        let OwnerApprovalNeed::Ask(card) =
            ManageMcpTool::project_start_approval(Some(&project), Some("docs"))
        else {
            panic!("a project server start must ask");
        };
        assert!(card.summary.contains("npx -y docs-mcp"), "{}", card.summary);
        assert!(card.summary.contains("Acme Site"));
        assert!(card.warnings[0].contains("MODE"));
        assert_eq!(card.risk, aidaemon_core::ApprovalRisk::High);
        // A changed entry is a new card.
        std::fs::write(
            dir.path().join(".mcp.json"),
            DOCS.replace("docs-mcp", "docs-mcp@2"),
        )
        .unwrap();
        let OwnerApprovalNeed::Ask(changed) =
            ManageMcpTool::project_start_approval(Some(&project), Some("docs"))
        else {
            panic!("a project server start must ask");
        };
        assert_ne!(card.subject, changed.subject);
        // What the start refuses never reaches the owner.
        for name in ["shell", "keyed"] {
            assert!(matches!(
                ManageMcpTool::project_start_approval(Some(&project), Some(name)),
                OwnerApprovalNeed::NotNeeded
            ));
        }
        // Judged without its project, or for a name it cannot find, a start
        // still asks: it fails closed.
        for (project, name) in [(None, "docs"), (Some(&project), "missing")] {
            assert!(matches!(
                ManageMcpTool::project_start_approval(project, Some(name)),
                OwnerApprovalNeed::Ask(_)
            ));
        }
        let effects = ManageMcpTool::effects_for(r#"{"action":"project_start","name":"docs"}"#);
        assert!(!effects.complete);
        assert_eq!(
            ManageMcpTool::effects_for(r#"{"action":"project_list"}"#),
            aidaemon_core::CallEffects::none()
        );
    }

    #[tokio::test]
    async fn project_actions_list_and_refuse_without_starting_anything() {
        use aidaemon_secrets::MemorySecretStore;
        let tool = ManageMcpTool::new(McpRegistry::new(
            std::sync::Arc::new(OneServerStore::default()),
            std::sync::Arc::new(MemorySecretStore::default()),
        ));
        let (project, _dir) = project_with(Some(DOCS));
        let list = tool
            .run_in(r#"{"action":"project_list"}"#, Some(&project))
            .await
            .unwrap();
        assert!(
            list.contains("- docs (`npx -y docs-mcp`): not started"),
            "{list}"
        );
        assert!(list.contains("'bash' is not an allowed command"), "{list}");
        assert!(list.contains("never expanded"), "{list}");
        let refused = tool
            .run_in(
                r#"{"action":"project_start","name":"shell"}"#,
                Some(&project),
            )
            .await
            .unwrap();
        assert!(refused.contains("not an allowed command"), "{refused}");
        let outside = tool
            .run_in(r#"{"action":"project_start","name":"docs"}"#, None)
            .await
            .unwrap();
        assert!(outside.contains("does not work in a project"), "{outside}");
        let (bare, _bare_dir) = project_with(None);
        let none = tool
            .run_in(r#"{"action":"project_list"}"#, Some(&bare))
            .await
            .unwrap();
        assert!(none.contains("declares no MCP servers"), "{none}");
        let stop = tool
            .run_in(r#"{"action":"project_stop","name":"docs"}"#, Some(&project))
            .await
            .unwrap();
        assert!(stop.contains("is not running"), "{stop}");
        assert!(tool.registry.list_servers().await.is_empty());
    }

    #[tokio::test]
    async fn a_blank_name_or_an_entry_changed_after_its_card_never_starts() {
        use aidaemon_secrets::MemorySecretStore;
        let tool = ManageMcpTool::new(McpRegistry::new(
            std::sync::Arc::new(OneServerStore::default()),
            std::sync::Arc::new(MemorySecretStore::default()),
        ));
        let (project, dir) = project_with(Some(
            r#"{"mcpServers": {" ": {"command": "python3", "args": ["-c", "print(1)"]},
                "docs ": {"command": "npx", "args": ["-y", "docs-mcp"]},
                "docs": {"command": "npx", "args": ["-y", "docs-mcp"]}}}"#,
        ));
        for name in [" ", "docs "] {
            let call = json!({"action": "project_start", "name": name}).to_string();
            let _ = judge(&tool, &project, &call).await;
            let reply = tool.run_in(&call, Some(&project)).await.unwrap();
            assert!(!reply.starts_with("Started"), "{name:?}: {reply}");
        }
        // The card showed one command; the file then changed: refused.
        let call = r#"{"action":"project_start","name":"docs"}"#;
        let OwnerApprovalNeed::Ask(card) = judge(&tool, &project, call).await else {
            panic!("a project server start must ask");
        };
        assert!(card.summary.contains("npx -y docs-mcp"), "{}", card.summary);
        std::fs::write(
            dir.path().join(".mcp.json"),
            r#"{"mcpServers": {"docs": {"command": "npx", "args": ["-y", "other-mcp"]}}}"#,
        )
        .unwrap();
        let reply = tool.run_in(call, Some(&project)).await.unwrap();
        assert!(reply.contains("changed after the approval card"), "{reply}");
        assert!(tool.registry.list_servers().await.is_empty());
    }

    /// A minimal stdio MCP server in Python, with one tool `lookup`.
    const PY_SERVER: &str = r#"
import json, sys
for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "py", "version": "0"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "lookup", "description": "Look a word up", "inputSchema": {"type": "object", "properties": {}}}]}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n")
    sys.stdout.flush()
"#;

    #[tokio::test]
    async fn an_approved_project_server_runs_under_its_project_and_stops() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 not found; skipped");
            return;
        }
        use aidaemon_secrets::MemorySecretStore;
        let (project, dir) = project_with(None);
        std::fs::write(dir.path().join("server.py"), PY_SERVER).unwrap();
        std::fs::write(
            dir.path().join(".mcp.json"),
            format!(
                r#"{{"mcpServers": {{"dict": {{"command": "python3", "args": ["{}"]}}}}}}"#,
                dir.path()
                    .join("server.py")
                    .display()
                    .to_string()
                    .replace('\\', "/")
            ),
        )
        .unwrap();
        let store = std::sync::Arc::new(OneServerStore::default());
        let tool = ManageMcpTool::new(McpRegistry::new(
            store.clone(),
            std::sync::Arc::new(MemorySecretStore::default()),
        ));
        // Without the card the kernel shows, nothing starts.
        let unasked = tool
            .run_in(
                r#"{"action":"project_start","name":"dict"}"#,
                Some(&project),
            )
            .await
            .unwrap();
        assert!(unasked.contains("needs the owner's approval"), "{unasked}");
        assert!(tool.registry.list_servers().await.is_empty());
        assert!(matches!(
            judge(
                &tool,
                &project,
                r#"{"action":"project_start","name":"dict"}"#
            )
            .await,
            OwnerApprovalNeed::Ask(_)
        ));
        let started = tool
            .run_in(
                r#"{"action":"project_start","name":"dict"}"#,
                Some(&project),
            )
            .await
            .unwrap();
        assert!(started.contains("prj_0a1b2c3d_dict__lookup"), "{started}");
        let names: Vec<String> = tool
            .registry
            .list_servers()
            .await
            .into_iter()
            .flat_map(|info| info.tool_names)
            .collect();
        assert_eq!(names, ["prj_0a1b2c3d_dict__lookup"]);
        assert!(super::super::project::offered_in_project(
            &names[0],
            Some(&project.id)
        ));
        assert!(!super::super::project::offered_in_project(&names[0], None));
        // Not saved: nothing reached the server store.
        assert!(store.rows.lock().unwrap().is_empty());
        let again = tool
            .run_in(
                r#"{"action":"project_start","name":"dict"}"#,
                Some(&project),
            )
            .await
            .unwrap();
        assert!(again.contains("already running"), "{again}");
        let list = tool
            .run_in(r#"{"action":"project_list"}"#, Some(&project))
            .await
            .unwrap();
        assert!(list.contains("): running"), "{list}");
        let stopped = tool
            .run_in(r#"{"action":"project_stop","name":"dict"}"#, Some(&project))
            .await
            .unwrap();
        assert!(stopped.starts_with("Stopped 'dict'"), "{stopped}");
        assert!(tool.registry.list_servers().await.is_empty());
    }

    /// One disabled server in memory, as its database row.
    #[derive(Default)]
    struct OneServerStore {
        rows: std::sync::Mutex<Vec<super::super::DynamicMcpServer>>,
    }

    #[async_trait]
    impl super::super::DynamicMcpServerStore for OneServerStore {
        async fn list_dynamic_mcp_servers(
            &self,
        ) -> anyhow::Result<Vec<super::super::DynamicMcpServer>> {
            Ok(self.rows.lock().unwrap().clone())
        }

        async fn update_dynamic_mcp_server(
            &self,
            server: &super::super::DynamicMcpServer,
        ) -> anyhow::Result<()> {
            let mut rows = self.rows.lock().unwrap();
            if let Some(row) = rows.iter_mut().find(|row| row.id == server.id) {
                *row = server.clone();
            }
            Ok(())
        }
    }

    fn tool_with(
        secrets: &[(&str, &str)],
    ) -> (
        ManageMcpTool,
        std::sync::Arc<OneServerStore>,
        std::sync::Arc<aidaemon_secrets::MemorySecretStore>,
    ) {
        use aidaemon_secrets::SecretStore;
        let store = std::sync::Arc::new(OneServerStore::default());
        store
            .rows
            .lock()
            .unwrap()
            .push(super::super::DynamicMcpServer {
                id: 7,
                name: "synthetic".to_string(),
                command: "node".to_string(),
                args_json: "[\"server.js\"]".to_string(),
                env_keys_json: "[]".to_string(),
                triggers_json: "[\"synthetic\"]".to_string(),
                enabled: false,
                created_at: String::new(),
            });
        let secret_store = std::sync::Arc::new(aidaemon_secrets::MemorySecretStore::new());
        for (key, value) in secrets {
            secret_store.set(key, value).unwrap();
        }
        let registry = McpRegistry::new(store.clone(), secret_store.clone());
        (ManageMcpTool::new(registry), store, secret_store)
    }

    fn env_keys(store: &OneServerStore) -> Vec<String> {
        serde_json::from_str(&store.rows.lock().unwrap()[0].env_keys_json).unwrap()
    }

    #[tokio::test]
    async fn set_env_refuses_a_value_argument_unread() {
        use aidaemon_secrets::SecretStore;
        let (tool, store, secrets) = tool_with(&[]);
        let output = tool
            .run(r#"{"action":"set_env","name":"synthetic","key":"API_TOKEN","value":"sk-synthetic-test-key"}"#)
            .await
            .unwrap();
        assert!(!output.contains("sk-synthetic-test-key"), "{output}");
        assert!(
            output.contains("aidaemon keychain set mcp_synthetic_API_TOKEN"),
            "{output}"
        );
        assert!(!secrets.contains("mcp_synthetic_API_TOKEN"));
        assert!(env_keys(&store).is_empty());
    }

    #[tokio::test]
    async fn set_env_binds_the_owner_stored_value() {
        let (tool, store, _secrets) =
            tool_with(&[("mcp_synthetic_API_TOKEN", "sk-synthetic-test-key")]);
        let output = tool
            .run(r#"{"action":"set_env","name":"synthetic","key":"API_TOKEN"}"#)
            .await
            .unwrap();
        assert!(output.contains("mcp_synthetic_API_TOKEN"), "{output}");
        assert!(!output.contains("sk-synthetic-test-key"), "{output}");
        assert_eq!(env_keys(&store), vec!["API_TOKEN".to_string()]);
    }

    #[tokio::test]
    async fn set_env_without_a_stored_value_names_the_owner_command() {
        let (tool, store, _secrets) = tool_with(&[]);
        let output = tool
            .run(r#"{"action":"set_env","name":"synthetic","key":"API_TOKEN"}"#)
            .await
            .unwrap();
        assert!(
            output.contains("aidaemon keychain set mcp_synthetic_API_TOKEN"),
            "{output}"
        );
        assert!(env_keys(&store).is_empty());
    }
}
