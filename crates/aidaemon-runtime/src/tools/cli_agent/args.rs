//! The tool's arguments and the prompt a run is given.

use super::*;

#[derive(Deserialize)]
pub(super) struct CliAgentArgs {
    /// The indices of the caller's checklist items this run carries out
    /// (the same typed link `terminal` takes): a change the run makes
    /// settles them.
    #[serde(default)]
    pub(super) checklist_items: Vec<usize>,
    #[serde(default)]
    pub(super) action: Option<String>,
    pub(super) tool: Option<String>,
    pub(super) prompt: Option<String>,
    /// Legacy/incorrect alias sometimes emitted by the model for delegated work.
    pub(super) mission: Option<String>,
    /// Legacy/incorrect alias sometimes emitted by the model for delegated work.
    pub(super) task: Option<String>,
    /// Terminal-like alias sometimes emitted instead of `prompt`.
    pub(super) command: Option<String>,
    /// Optional description paired with `command`.
    pub(super) description: Option<String>,
    /// What the work is, in a few plain words: the name the owner reads
    /// for the run on every card ([`crate::helper_run`]).
    #[serde(default)]
    pub(super) title: Option<String>,
    /// Folders made fresh for this run (a new install or project), exactly
    /// as `spawn_agent` takes them: each must not exist yet, and the call
    /// creates it empty for the agent to work in.
    #[serde(default)]
    pub(super) new_folders: Vec<String>,
    /// Files or directories the run will change.
    #[serde(default)]
    pub(super) scope: Vec<String>,
    pub(super) working_dir: Option<String>,
    /// Additional readable paths for a split access manifest. The working
    /// directory is always readable context.
    #[serde(default)]
    pub(super) read_paths: Vec<String>,
    /// Exact writable files/directories for a read-write run. When omitted,
    /// legacy single-root runs grant writes only to `working_dir`.
    #[serde(default)]
    pub(super) write_paths: Vec<String>,
    /// Typed delegation authority. Prompt prose never narrows the sandbox.
    #[serde(default)]
    pub(super) workspace_mode: CliWorkspaceMode,
    pub(super) task_id: Option<String>,
    /// Optional system instruction to shape the CLI agent into a specialist
    pub(super) system_instruction: Option<String>,
    /// If true, start the task in background and return immediately with task_id
    #[serde(default)]
    pub(super) async_mode: Option<bool>,
    /// The dispatcher's facts ([`Self::bind_facts`]), never arguments: the
    /// session for cancel_all filtering, the goal for routing async/timeout
    /// notifications, the task for delegated task recovery, and the role
    /// for role-aware safeguards.
    #[serde(skip)]
    pub(super) call_session: Option<String>,
    #[serde(skip)]
    pub(super) call_goal_id: Option<String>,
    #[serde(skip)]
    pub(super) call_task_id: Option<String>,
    #[serde(skip)]
    pub(super) call_role: Option<UserRole>,
}

pub(super) fn non_empty_prompt_field(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub(super) fn compose_delegated_prompt(
    mission: Option<String>,
    task: Option<String>,
) -> Option<String> {
    if mission.is_none() && task.is_none() {
        return None;
    }

    let mut parts = Vec::new();
    if let Some(mission) = mission {
        parts.push(format!("Mission: {}", mission));
    }
    if let Some(task) = task {
        parts.push(format!("Task: {}", task));
    }
    Some(parts.join("\n"))
}

/// Whether an entry launches Claude Code: its program (or the program an
/// `env` launcher wraps) is the `claude` executable.
/// The file name of the program an entry launches (the one `env` wraps,
/// else its command).
pub(super) fn entry_program(entry: &CliToolEntry) -> String {
    let program_index = is_env_launcher(&entry.command)
        .then(|| env_wrapped_program_index(&entry.args))
        .flatten();
    let program = program_index
        .and_then(|index| entry.args.get(index).map(String::as_str))
        .unwrap_or(&entry.command);
    std::path::Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(program)
        .to_string()
}

/// The line a run's result starts with when another agent than the one
/// asked for did the work.
pub(super) fn substituted_agent_note(asked: &str, ran: &str) -> String {
    format!(
        "[This run was done by `{ran}`, not `{asked}`: `{asked}` could not run it under this \
         run's access rules. Attribute the work to `{ran}`.]"
    )
}

/// Whether a run's call output says it never started (a refusal before
/// launch, typed by this tool's own prefixes).
pub(super) fn run_did_not_start(output: &str) -> bool {
    let output = output.trim_start();
    output.starts_with("Blocked:")
        || output.starts_with("BLOCKED:")
        || output.starts_with("ERROR: CLI agent recovery exhausted")
        || output.starts_with("Maximum ")
}

/// Where a CLI agent's run may write: what its call named, plus its
/// declared scope and new folders, all within the parent turn's authority
/// narrowed by that declaration ([`crate::helper_run::helper_authority`],
/// the same narrowing a sub-agent gets). Without an inherited authority (a
/// call outside a turn) the named paths stand as given.
pub(super) fn helper_write_paths(
    args: &CliAgentArgs,
    spec: &crate::helper_run::HelperSpec,
    session_id: &str,
    facts: &ToolCallContext,
) -> Vec<String> {
    let declared = !spec.new_folders.is_empty() || !spec.scope.is_empty();
    let mut paths = args.write_paths.clone();
    for path in spec.new_folders.iter().chain(&spec.scope) {
        if !paths.contains(path) {
            paths.push(path.clone());
        }
    }
    let Some(parent) = crate::authority::parent_authority(facts) else {
        return paths;
    };
    let narrowed = crate::helper_run::helper_authority(parent, spec.declaration(session_id), None);
    let Some(roots) = crate::helper_run::write_roots(&narrowed) else {
        return paths;
    };
    let within = |path: &String| {
        !std::path::Path::new(path).is_absolute()
            || roots
                .iter()
                .any(|root| std::path::Path::new(path).starts_with(root))
    };
    paths.retain(within);
    if declared {
        for root in roots {
            if !paths.contains(&root) {
                paths.push(root);
            }
        }
    }
    paths
}

impl CliAgentArgs {
    /// Take the caller's session, goal, task and role from the dispatcher.
    pub(super) fn bind_facts(&mut self, facts: &ToolCallContext) {
        self.call_session = Some(facts.session_id.clone());
        self.call_goal_id = facts.goal_id.clone();
        self.call_task_id = facts.task_id.clone();
        self.call_role = facts.user_role;
    }

    pub(super) fn run_prompt(&self) -> Option<String> {
        if let Some(prompt) = non_empty_prompt_field(self.prompt.as_deref()) {
            return Some(prompt);
        }

        if let Some(prompt) = compose_delegated_prompt(
            non_empty_prompt_field(self.mission.as_deref()),
            non_empty_prompt_field(self.task.as_deref()),
        ) {
            return Some(prompt);
        }

        let description = non_empty_prompt_field(self.description.as_deref());
        let command = non_empty_prompt_field(self.command.as_deref());
        if description.is_some() || command.is_some() {
            let mut parts = Vec::new();
            if let Some(description) = description {
                parts.push(description);
            }
            if let Some(command) = command {
                parts.push(format!(
                    "Run this exact shell command in the working directory and report the result:\n{}",
                    command
                ));
            }
            return Some(parts.join("\n\n"));
        }

        None
    }

    pub(super) async fn contextual_run_prompt(
        &self,
        state: &Arc<dyn CliAgentStore>,
    ) -> Option<String> {
        let delegated_task = match self.call_task_id.as_deref() {
            Some(task_id) if !task_id.trim().is_empty() => {
                state.get_task(task_id).await.ok().flatten()
            }
            _ => None,
        };
        let delegated_goal_id = self
            .call_goal_id
            .clone()
            .or_else(|| delegated_task.as_ref().map(|task| task.goal_id.clone()));
        let delegated_goal = match delegated_goal_id.as_deref() {
            Some(goal_id) if !goal_id.trim().is_empty() => {
                state.get_goal(goal_id).await.ok().flatten()
            }
            _ => None,
        };

        compose_delegated_prompt(
            delegated_goal
                .as_ref()
                .and_then(|goal| non_empty_prompt_field(Some(goal.description.as_str()))),
            delegated_task
                .as_ref()
                .and_then(|task| non_empty_prompt_field(Some(task.description.as_str()))),
        )
    }
}
