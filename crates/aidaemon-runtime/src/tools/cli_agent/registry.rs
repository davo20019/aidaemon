//! The configured agents: choosing one, discovery, wiring, and adding,
//! removing or enabling agents.

use super::*;

impl CliAgentTool {
    pub(super) fn tool_priority(name: &str) -> usize {
        let lower = name.to_ascii_lowercase();
        DEFAULT_TOOL_PRIORITY
            .iter()
            .position(|known| *known == lower)
            .unwrap_or(DEFAULT_TOOL_PRIORITY.len())
    }

    /// Pick a default tool if caller omits `tool`.
    /// Preference: known coding CLIs first, then lexicographic order.
    pub(super) fn default_tool_name(&self) -> Option<String> {
        let tools = self.tools.read().unwrap();
        if tools.is_empty() {
            return None;
        }

        let mut names: Vec<String> = tools.keys().cloned().collect();
        names.sort_by(|a, b| {
            Self::tool_priority(a)
                .cmp(&Self::tool_priority(b))
                .then_with(|| a.cmp(b))
        });
        names.into_iter().next()
    }

    /// The agent to retry a delegated run with after the one picked was
    /// unavailable (and quarantined out of `tools`). It obeys the first
    /// pick's adapter rule: a read/write run under an enforced access
    /// contract falls back only to an agent with a native split adapter,
    /// since any other is refused the moment it starts.
    pub(super) fn fallback_tool_name(
        &self,
        enforce_native_access: bool,
        read_only_run: bool,
    ) -> Option<String> {
        if enforce_native_access && !read_only_run {
            self.scoped_read_write_tool_name(None)
        } else {
            self.default_tool_name()
        }
    }

    /// Whether a coding agent's approval requests can be answered in this
    /// call: the run can judge them (the kernel's nested-call handle is
    /// here) and the process is local (its stdin stays open). A turn that
    /// read untrusted content may host one too: each request is judged by
    /// that tainted run, its network hosts as its egress (logged per call
    /// with the turn's taint).
    pub(super) fn can_host(&self, facts: &ToolCallContext) -> bool {
        let host = facts
            .extensions
            .get::<aidaemon_core::NestedCalls>()
            .is_some()
            || facts
                .extensions
                .get::<Arc<crate::approvals::DetachedNestedJudge>>()
                .is_some();
        let local = self.backend.kind() == aidaemon_tools::backend::BackendKind::Local;
        let authority = crate::authority::parent_authority(facts);
        let clean = authority.is_some_and(|authority| !authority.untrusted_context);
        info!(
            can_host = host && local,
            host,
            local,
            has_authority = authority.is_some(),
            clean,
            "Whether a coding agent's requests can be answered for this call"
        );
        host && local
    }

    /// The coding agent an entry launches, by its program.
    pub(super) fn entry_agent(
        &self,
        entry: &CliToolEntry,
    ) -> Arc<dyn aidaemon_coding_agents::CodingAgent> {
        self.coding_agents.for_program(&entry_program(entry))
    }

    /// The coding agent `name` launches, when it is configured.
    pub(super) fn agent_named(
        &self,
        name: &str,
    ) -> Option<Arc<dyn aidaemon_coding_agents::CodingAgent>> {
        let tools = self.tools.read().ok()?;
        Some(self.entry_agent(tools.get(name)?))
    }

    /// Whether `name` may run this run in its owner-set mode here.
    pub(super) async fn runnable(&self, name: &str, read_only: bool, can_host: bool) -> bool {
        let Some(agent) = self.agent_named(name) else {
            return false;
        };
        if !crate::tools::agent_mode::readiness_of(name).is_usable() {
            return false;
        }
        self.permissions()
            .agent_mode(name, &*agent)
            .is_some_and(|mode| agent.can_run(mode, read_only, can_host))
    }

    pub(super) fn configured_tool(&self, name: &str) -> Option<String> {
        self.tools
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(name)
            .then(|| name.to_string())
    }

    /// Select a configured agent that confines a read-write run on its own.
    /// Adapter capability—not prompt wording or preference order—controls the
    /// choice for least-privilege delegated mutation.
    pub(super) fn scoped_read_write_tool_name(&self, preferred: Option<&str>) -> Option<String> {
        let tools = self.tools.read().ok()?;
        // An agent that confines a write run on its own, with no host.
        let supports_split_access = |entry: &CliToolEntry| {
            self.entry_agent(entry).can_run(
                aidaemon_coding_agents::PermissionMode::Checked,
                false,
                false,
            )
        };
        if let Some(preferred) = preferred
            && tools.get(preferred).is_some_and(supports_split_access)
        {
            return Some(preferred.to_string());
        }
        let mut compatible = tools
            .iter()
            .filter(|(_, entry)| supports_split_access(entry))
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        compatible.sort();
        compatible.into_iter().next()
    }

    /// Normalize a working directory for lock/key comparisons.
    /// Uses canonical paths when possible so aliases like `/repo` and `/repo/`
    /// map to the same lock key.
    pub(super) async fn normalize_working_dir(
        backend: &SharedExecutionBackend,
        dir: &str,
    ) -> anyhow::Result<String> {
        let resolved = backend.resolve_path(dir).await?;
        Ok(backend
            .canonicalize(&resolved)
            .await
            .unwrap_or(resolved)
            .to_string())
    }

    /// Normalize an access target against the call's execution cwd. Unlike a
    /// working directory, a write target may not exist yet; canonicalization
    /// is therefore best-effort after backend policy resolution.
    pub(super) async fn normalize_access_path(
        backend: &SharedExecutionBackend,
        path: &str,
        working_dir: Option<&str>,
    ) -> anyhow::Result<String> {
        let path = path.trim();
        anyhow::ensure!(!path.is_empty(), "CLI access path cannot be empty");
        let prepared = if BackendPath::is_anchored(path) {
            path.to_string()
        } else if let Some(working_dir) = working_dir {
            BackendPath::new(working_dir).join(path).to_string()
        } else {
            path.to_string()
        };
        let resolved = backend.resolve_path(&prepared).await?;
        Ok(backend
            .canonicalize(&resolved)
            .await
            .unwrap_or(resolved)
            .to_string())
    }

    /// Release a working-directory claim if it belongs to the given task.
    pub(super) fn release_working_dir_claim(&self, dir: &str, task_id: &str) {
        release_claim(&mut lock_claims(&self.working_dir_claims), dir, task_id);
    }

    pub async fn discover(
        backend: SharedExecutionBackend,
        config: CliAgentsConfig,
        state: Arc<dyn CliAgentStore>,
    ) -> Self {
        let default_timeout = Duration::from_secs(config.timeout_secs);
        let default_max_output = config.max_output_chars;

        type ToolCandidate = (
            String,
            String,
            Vec<String>,
            String,
            Option<u64>,
            Option<usize>,
        );
        let mut candidates: Vec<ToolCandidate> = Vec::new();

        if config.tools.is_empty() {
            for (name, cmd, args, desc) in default_tool_definitions() {
                candidates.push((
                    name.to_string(),
                    cmd.to_string(),
                    args.into_iter().map(|s| s.to_string()).collect(),
                    desc.to_string(),
                    None,
                    None,
                ));
            }
        } else {
            for (name, tool_cfg) in &config.tools {
                candidates.push((
                    name.clone(),
                    tool_cfg.command.clone(),
                    tool_cfg.args.clone(),
                    tool_cfg.description.clone(),
                    tool_cfg.timeout_secs,
                    tool_cfg.max_output_chars,
                ));
            }
        }

        // Resolve launchers in parallel. This validates a nested executable in
        // `env ... agent` configurations instead of merely proving that
        // `/usr/bin/env` exists.
        let resolution_futures: Vec<_> = candidates
            .iter()
            .map(|(_, command, args, _, _, _)| resolve_agent_launch(&backend, command, args))
            .collect();
        let resolution_results = futures::future::join_all(resolution_futures).await;

        let mut tools = HashMap::new();

        for (i, (name, command, _args, description, timeout_override, max_output_override)) in
            candidates.into_iter().enumerate()
        {
            if let Ok((resolved_command, resolved_args)) = &resolution_results[i] {
                info!(name = %name, command = %resolved_command, "CLI agent tool discovered");
                tools.insert(
                    name,
                    CliToolEntry {
                        command: resolved_command.clone(),
                        args: resolved_args.clone(),
                        description,
                        timeout: timeout_override
                            .map(Duration::from_secs)
                            .unwrap_or(default_timeout),
                        max_output_chars: max_output_override.unwrap_or(default_max_output),
                        is_dynamic: false,
                    },
                );
            } else {
                info!(
                    name = %name,
                    command = %command,
                    error = %resolution_results[i].as_ref().unwrap_err(),
                    "CLI agent tool not found, skipping"
                );
            }
        }

        let mut tool_names: Vec<String> = tools.keys().cloned().collect();
        tool_names.sort();

        // An agent the owner says speaks the Agent Client Protocol is
        // hosted over it, ahead of any built-in agent for its program.
        let mut coding_agents = aidaemon_coding_agents::CodingAgents::builtin();
        for (name, tool_config) in &config.tools {
            if tool_config.protocol.as_deref() == Some("acp") {
                let program = std::path::Path::new(&tool_config.command)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(&tool_config.command)
                    .to_string();
                coding_agents = coding_agents.preferring(
                    aidaemon_coding_agents::acp::AcpAgent::custom(name, &program, &[]),
                );
            }
        }
        let tool = CliAgentTool {
            context_facts: None,
            project_notes: None,
            backend,
            tools: Arc::new(std::sync::RwLock::new(tools)),
            tool_names: Arc::new(std::sync::RwLock::new(tool_names)),
            running: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
            state,
            default_timeout,
            default_max_output,
            max_concurrent: DEFAULT_MAX_CONCURRENT,
            concurrency_limiter: Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT)),
            hub: OnceLock::new(),
            runs: OnceLock::new(),
            plan_store: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
            coding_agents,
            permissions: Arc::new(OnceLock::new()),
        };

        // Load dynamic agents from DB
        tool.load_dynamic_agents().await;
        // The Settings page lists these agents with their modes.
        let listed = Arc::downgrade(&tool.tools);
        let registry = tool.coding_agents.clone();
        let local = tool.backend.kind() == aidaemon_tools::backend::BackendKind::Local;
        crate::tools::agent_mode::publish(Arc::new(move || {
            listed
                .upgrade()
                .and_then(|tools| {
                    tools.read().ok().map(|tools| {
                        tools
                            .iter()
                            .map(|(name, entry)| {
                                // A launcher's own arguments, through the
                                // program it runs.
                                let prefix = is_env_launcher(&entry.command)
                                    .then(|| env_wrapped_program_index(&entry.args))
                                    .flatten()
                                    .map(|index| entry.args[..=index].to_vec())
                                    .unwrap_or_default();
                                crate::tools::agent_mode::AgentInfo {
                                    name: name.clone(),
                                    agent: registry.for_program(&entry_program(entry)),
                                    command: entry.command.clone(),
                                    prefix,
                                    local,
                                }
                            })
                            .collect()
                    })
                })
                .unwrap_or_default()
        }));
        // Each agent says whether it can work, in the background (never
        // under test: it would ask the agents installed on the machine).
        if cfg!(not(test)) {
            tokio::spawn(crate::tools::agent_mode::check_all());
        }

        tool
    }

    /// Set channel hub reference for immediate background completion delivery.
    /// Follow the daemon's permission choices (shared with Settings). Set
    /// once, at start; until then the defaults (Standard, no choices).
    pub fn set_permissions(&self, permissions: crate::permissions::Permissions) {
        let _ = self.permissions.set(permissions);
    }

    pub(super) fn permissions(&self) -> crate::permissions::Permissions {
        self.permissions.get().cloned().unwrap_or_default()
    }

    pub fn with_context_facts(mut self, facts: Arc<dyn crate::tools::traits::FactStore>) -> Self {
        self.context_facts = Some(facts);
        self
    }

    /// Hand delegated agents of a turn in a project its memory index.
    pub fn with_project_notes(
        mut self,
        notes: Option<Arc<aidaemon_memory::project::ProjectNotes>>,
    ) -> Self {
        self.project_notes = notes;
        self
    }

    pub fn set_hub(&self, hub: Weak<dyn OutboundRouter>) {
        self.hub
            .set(hub)
            .expect("CliAgentTool::set_hub called more than once");
    }

    pub fn set_agent(&self, agent: Weak<dyn ConversationRuntime>) {
        self.agent
            .set(agent)
            .expect("CliAgentTool::set_agent called more than once");
    }

    /// Wire the runtime's run records (see `runs`).
    pub fn set_runs(&self, runs: Arc<crate::run_state::RunStates>) {
        let _ = self.runs.set(runs);
    }

    /// Where a background run of this tool reports.
    pub(super) fn run_home(&self) -> crate::background_run::RunHome {
        crate::background_run::RunHome {
            runs: self.runs.get_or_init(Default::default).clone(),
            hub: self.hub.get().cloned(),
            plans: self.plan_store.get().cloned(),
        }
    }

    /// Wire the session checklists (see `plan_store`).
    pub fn set_plan_store(&self, plan_store: Arc<aidaemon_work::plans::PlanStore>) {
        let _ = self.plan_store.set(plan_store);
    }

    pub(super) fn get_hub(&self) -> Option<Arc<dyn OutboundRouter>> {
        self.hub.get().and_then(|w| w.upgrade())
    }

    pub(super) fn get_agent(&self) -> Option<Arc<dyn ConversationRuntime>> {
        self.agent.get().and_then(|w| w.upgrade())
    }

    pub fn wiring_ready(&self) -> bool {
        self.hub.get().and_then(Weak::upgrade).is_some()
            && self.agent.get().and_then(Weak::upgrade).is_some()
    }

    /// Load dynamically registered agents from the database.
    pub(super) async fn load_dynamic_agents(&self) {
        match self.state.list_dynamic_cli_agents().await {
            Ok(agents) => {
                for agent in agents {
                    if !agent.enabled {
                        continue;
                    }
                    let args: Vec<String> =
                        serde_json::from_str(&agent.args_json).unwrap_or_default();
                    let (command, args) =
                        match resolve_agent_launch(&self.backend, &agent.command, &args).await {
                            Ok(launch) => launch,
                            Err(error) => {
                                info!(
                                    name = %agent.name,
                                    command = %agent.command,
                                    %error,
                                    "Dynamic CLI agent command not found, skipping"
                                );
                                continue;
                            }
                        };
                    let entry = CliToolEntry {
                        command,
                        args,
                        description: agent.description.clone(),
                        timeout: agent
                            .timeout_secs
                            .map(Duration::from_secs)
                            .unwrap_or(self.default_timeout),
                        max_output_chars: agent.max_output_chars.unwrap_or(self.default_max_output),
                        is_dynamic: true,
                    };
                    let mut tools = self.tools.write().unwrap();
                    tools.insert(agent.name.clone(), entry);
                    let mut names = self.tool_names.write().unwrap();
                    if !names.contains(&agent.name) {
                        names.push(agent.name.clone());
                        names.sort();
                    }
                    info!(name = %agent.name, "Loaded dynamic CLI agent from DB");
                }
            }
            Err(e) => {
                warn!("Failed to load dynamic CLI agents: {}", e);
            }
        }
    }

    pub fn has_tools(&self) -> bool {
        !self.tools.read().unwrap().is_empty()
    }

    /// Add a new CLI agent at runtime. Returns error message string on validation failure.
    pub async fn add_agent(
        &self,
        name: &str,
        command: &str,
        args: Vec<String>,
        description: &str,
        timeout_secs: Option<u64>,
        max_output_chars: Option<usize>,
    ) -> anyhow::Result<String> {
        let (resolved_command, resolved_args) =
            match resolve_agent_launch(&self.backend, command, &args).await {
                Ok(launch) => launch,
                Err(error) => return Ok(format!("CLI agent configuration is invalid: {error}.")),
            };

        // Save to database
        let dynamic = DynamicCliAgent {
            id: 0,
            name: name.to_string(),
            command: command.to_string(),
            args_json: serde_json::to_string(&args)?,
            description: description.to_string(),
            timeout_secs,
            max_output_chars,
            enabled: true,
            created_at: String::new(),
        };
        self.state.save_dynamic_cli_agent(&dynamic).await?;

        // Add to runtime map
        let entry = CliToolEntry {
            command: resolved_command,
            args: resolved_args,
            description: description.to_string(),
            timeout: timeout_secs
                .map(Duration::from_secs)
                .unwrap_or(self.default_timeout),
            max_output_chars: max_output_chars.unwrap_or(self.default_max_output),
            is_dynamic: true,
        };
        let mut tools = self.tools.write().unwrap();
        tools.insert(name.to_string(), entry);
        let mut names = self.tool_names.write().unwrap();
        if !names.contains(&name.to_string()) {
            names.push(name.to_string());
            names.sort();
        }

        Ok(format!("CLI agent '{}' added successfully.", name))
    }

    /// Remove a CLI agent by name.
    pub async fn remove_agent(&self, name: &str) -> anyhow::Result<String> {
        // Find and remove from DB
        let agents = self.state.list_dynamic_cli_agents().await?;
        if let Some(agent) = agents.iter().find(|a| a.name == name) {
            self.state.delete_dynamic_cli_agent(agent.id).await?;
        }

        // Remove from runtime map
        let mut tools = self.tools.write().unwrap();
        if tools.remove(name).is_some() {
            let mut names = self.tool_names.write().unwrap();
            names.retain(|n| n != name);
            Ok(format!("CLI agent '{}' removed.", name))
        } else {
            Ok(format!("CLI agent '{}' not found.", name))
        }
    }

    /// Enable or disable a CLI agent.
    pub async fn enable_agent(&self, name: &str, enabled: bool) -> anyhow::Result<String> {
        let agents = self.state.list_dynamic_cli_agents().await?;
        if let Some(mut agent) = agents.into_iter().find(|a| a.name == name) {
            agent.enabled = enabled;
            self.state.update_dynamic_cli_agent(&agent).await?;

            if enabled {
                let args: Vec<String> = serde_json::from_str(&agent.args_json).unwrap_or_default();
                // Re-add to runtime map only after validating both a wrapper
                // command and its nested agent executable.
                if let Ok((command, args)) =
                    resolve_agent_launch(&self.backend, &agent.command, &args).await
                {
                    let entry = CliToolEntry {
                        command,
                        args,
                        description: agent.description.clone(),
                        timeout: agent
                            .timeout_secs
                            .map(Duration::from_secs)
                            .unwrap_or(self.default_timeout),
                        max_output_chars: agent.max_output_chars.unwrap_or(self.default_max_output),
                        is_dynamic: true,
                    };
                    let mut tools = self.tools.write().unwrap();
                    tools.insert(name.to_string(), entry);
                    let mut names = self.tool_names.write().unwrap();
                    if !names.contains(&name.to_string()) {
                        names.push(name.to_string());
                        names.sort();
                    }
                }
            } else {
                // Remove from runtime map
                let mut tools = self.tools.write().unwrap();
                tools.remove(name);
                let mut names = self.tool_names.write().unwrap();
                names.retain(|n| n != name);
            }

            let action = if enabled { "enabled" } else { "disabled" };
            Ok(format!("CLI agent '{}' {}.", name, action))
        } else {
            // Check if it's a discovered (non-dynamic) agent
            let tools = self.tools.read().unwrap();
            if tools.contains_key(name) {
                Ok(format!(
                    "CLI agent '{}' is a discovered agent (not dynamic). Cannot toggle — it's always available while installed.",
                    name
                ))
            } else {
                Ok(format!("CLI agent '{}' not found.", name))
            }
        }
    }

    /// List all registered agents with their status.
    pub fn list_agents(&self) -> Vec<(String, String, String, bool)> {
        let tools = self.tools.read().unwrap();
        let mut result: Vec<(String, String, String, bool)> = tools
            .iter()
            .map(|(name, entry)| {
                let source = if entry.is_dynamic {
                    "dynamic".to_string()
                } else {
                    "discovered".to_string()
                };
                (
                    name.clone(),
                    entry.description.clone(),
                    source,
                    true, // enabled if in map
                )
            })
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}
