use super::*;
use crate::agent::specialists::{
    validation as specialist_validation, SpecialistRegistry, SpecialistRenderContext,
};
use crate::traits::SpecialistKind;

struct TaskLeadSpec {
    tools: Vec<Arc<dyn Tool>>,
    system_prompt: String,
    root_tools: Vec<Arc<dyn Tool>>,
    input_text: String,
}

impl Agent {
    pub(crate) fn select_specialist_kind(
        role: AgentRole,
        mission: &str,
        task: &str,
    ) -> SpecialistKind {
        match role {
            AgentRole::TaskLead => return SpecialistKind::TaskLead,
            AgentRole::Orchestrator => return SpecialistKind::Generic,
            AgentRole::Executor => {}
        }

        let text = format!("{} {}", mission, task).to_ascii_lowercase();
        let has_marker = |needles: &[&str]| needles.iter().any(|needle| text.contains(needle));
        let has_word = |needles: &[&str]| {
            needles
                .iter()
                .any(|needle| contains_keyword_as_words(&text, needle))
        };

        if has_marker(&[".md", "write-up"])
            || has_word(&[
                "markdown",
                "report",
                "document",
                "writeup",
                "save it as",
                "create a file",
                "write a file",
            ])
        {
            return SpecialistKind::ArtifactWriter;
        }

        // Check browser-verifier BEFORE Code so a task like "open the homepage
        // in the browser and run the smoke test" isn't captured by Code's
        // `test` keyword.
        if has_word(&[
            "browser",
            "web page",
            "website",
            "screenshot",
            "playwright",
            "verify ui",
            "localhost",
        ]) {
            return SpecialistKind::BrowserVerifier;
        }

        // `test` is intentionally specific (`cargo test`, `unit test`, `pytest`)
        // to avoid matching incidental uses like "smoke test in the browser".
        if has_marker(&[".rs", ".ts", ".tsx", ".js", ".py"])
            || has_word(&[
                "cargo",
                "cargo test",
                "unit test",
                "pytest",
                "bug",
                "code",
                "compile",
                "refactor",
                "implement",
            ])
        {
            return SpecialistKind::Code;
        }

        if has_word(&["review", "audit", "inspect", "risk", "regression"]) {
            return SpecialistKind::Review;
        }

        if has_word(&[
            "research",
            "look up",
            "web search",
            "current",
            "latest",
            "source",
            "sources",
            "investigate",
        ]) {
            return SpecialistKind::Research;
        }

        if has_word(&["draft", "email", "message", "reply", "comms"]) {
            return SpecialistKind::CommsDraft;
        }

        SpecialistKind::Executor
    }

    pub(crate) fn build_specialist_session_id(kind: SpecialistKind, id: Uuid) -> String {
        format!("specialist:{}:{}", kind.as_str(), id)
    }

    /// Resolve the effective `SpecialistKind` for a spawn given:
    /// - explicit role (role-typed spawns ignore the arg),
    /// - optional caller-supplied `arg_specialist` (from `spawn_agent`'s schema),
    /// - mission + task text (heuristic fallback).
    ///
    /// "task_lead" is role-typed and rejected when passed as an arg (the
    /// `AgentRole::TaskLead` spawn path produces it). Invalid arg values fall
    /// through to the executor heuristic and produce a warn log.
    pub(crate) fn resolve_specialist_kind(
        role: Option<AgentRole>,
        arg_specialist: Option<&str>,
        mission: &str,
        task: &str,
    ) -> SpecialistKind {
        if let Some(role) = role {
            return Self::select_specialist_kind(role, mission, task);
        }
        if let Some(s) = arg_specialist {
            match SpecialistKind::from_str(s) {
                Some(kind) if kind != SpecialistKind::TaskLead => return kind,
                Some(_) => {
                    // Caller passed "task_lead" but that's role-typed only.
                    warn!(arg = %s, "ignoring 'task_lead' specialist arg; role-typed only");
                }
                None => {
                    warn!(
                        arg = %s,
                        "ignoring invalid specialist arg; falling back to heuristic"
                    );
                }
            }
        }
        Self::select_specialist_kind(AgentRole::Executor, mission, task)
    }

    /// Compute the set of tool names a given child role is permitted to use.
    /// Used to intersect a specialist's declared tool allowlist with the
    /// hard role boundary so a specialist cannot escape its role scope.
    fn role_tool_scope_names(role: Option<AgentRole>, all_tools: &[Arc<dyn Tool>]) -> Vec<String> {
        match role {
            Some(AgentRole::Executor) => all_tools
                .iter()
                .filter(|t| matches!(t.tool_role(), ToolRole::Action | ToolRole::Universal))
                .map(|t| t.name().to_string())
                .collect(),
            Some(AgentRole::TaskLead) => all_tools
                .iter()
                .filter(|t| {
                    matches!(t.tool_role(), ToolRole::Management | ToolRole::Universal)
                        || t.name() == "spawn_agent"
                })
                .map(|t| t.name().to_string())
                .collect(),
            Some(AgentRole::Orchestrator) | None => {
                all_tools.iter().map(|t| t.name().to_string()).collect()
            }
        }
    }

    /// Apply a specialist's declared tool allowlist to the in-flight tool set,
    /// intersected with the role scope. Tools outside the intersection are
    /// dropped. Returns the (possibly unchanged) tool set.
    fn apply_specialist_tool_allowlist(
        tools: Vec<Arc<dyn Tool>>,
        declared: &[String],
        kind: SpecialistKind,
        role: Option<AgentRole>,
        full_tools: &[Arc<dyn Tool>],
    ) -> Vec<Arc<dyn Tool>> {
        let scope = Self::role_tool_scope_names(role, full_tools);
        let scope_refs: Vec<&str> = scope.iter().map(|s| s.as_str()).collect();
        let known_owned: Vec<String> = full_tools.iter().map(|t| t.name().to_string()).collect();
        let known: Vec<&str> = known_owned.iter().map(|s| s.as_str()).collect();
        let permitted = specialist_validation::intersect_tools(kind, declared, &scope_refs, &known);
        tools
            .into_iter()
            .filter(|t| permitted.iter().any(|p| p == t.name()))
            .collect()
    }

    /// Render the task-lead system prompt using the `SpecialistRegistry` as the
    /// source of truth for the base template, then append the same dynamic
    /// sections (Prior Knowledge, CLI Agent Delegation) the legacy builder
    /// produced. This keeps byte-equivalence with `build_task_lead_prompt`
    /// when `goal_context=None` and `has_cli_agent=false`.
    #[allow(clippy::too_many_arguments)]
    fn compose_task_lead_prompt_from_registry(
        registry: &SpecialistRegistry,
        goal_id: &str,
        goal_description: &str,
        goal_context: Option<&str>,
        depth: usize,
        max_depth: usize,
        has_cli_agent: bool,
        is_scheduled: bool,
    ) -> String {
        let execution_mode = if is_scheduled {
            "You have full tool access including `terminal`. For simple steps (single shell commands, \
             file writes), execute them directly. For complex multi-step work, you may still delegate \
             to executors via the workflow below.".to_string()
        } else {
            "Your primary job is to plan and delegate work via executors or cli_agent. \
             However, you also have direct access to essential tools (read_file, write_file, \
             edit_file, terminal, search_files). Use delegation first, but if delegation fails \
             (cli_agent errors, spawn_agent blocked, executor failures), switch to direct \
             execution with your own tools rather than retrying broken delegation paths."
                .to_string()
        };
        let ctx = SpecialistRenderContext {
            mission: goal_description.to_string(),
            task: String::new(),
            depth,
            max_depth,
            max_iterations: 0,
            goal_id: goal_id.to_string(),
            working_dir: String::new(),
            is_scheduled,
            parent_session_id: String::new(),
            execution_mode,
        };
        let mut prompt = registry.render(SpecialistKind::TaskLead, &ctx);

        if let Some(ctx_text) = goal_context {
            prompt.push_str(&format!(
                "\n\n## Prior Knowledge\n\
                 The following knowledge was gathered from previous tasks and may be relevant:\n{}",
                format_goal_context(ctx_text)
            ));
        }

        if has_cli_agent {
            prompt.push_str(
                "\n\n## CLI Agent Delegation\n\
                 You have direct access to `cli_agent` (a specialized coding/research agent running on this machine).\n\
                 Treat `cli_agent` as a delegation surface, not as a reason to skip task structure.\n\
                 If the work should stay tied to a claimed task with executor results or blocker handling, claim the task and use `spawn_agent`.\n\
                 Prefer direct `cli_agent` calls for focused execution-heavy work when you do not need aidaemon-only tools in the child.\n\
                 When calling `cli_agent`, use `action=\"run\"` and include a non-empty `prompt` describing the work.\n\
                 Pass `working_dir` whenever the task targets a specific repo or directory.\n\
                 Example: `cli_agent(action=\"run\", prompt=\"Inspect the latest service logs, patch the root cause, run cargo fmt, and run the narrowest relevant tests\", working_dir=\"/absolute/project/path\")`.\n\
                 Note: If cli_agent fails repeatedly (auth errors, timeouts, environment issues), do NOT keep retrying. Switch to using your direct tools (read_file, write_file, edit_file, terminal) to complete the work yourself.",
            );
        }

        prompt
    }

    /// Render the executor system prompt using the `SpecialistRegistry` as the
    /// source of truth for the base template, then splice in the same dynamic
    /// sections (working directory, task contract, cli-agent suffix) the
    /// legacy builder produced. Keeps byte-equivalence with
    /// `build_executor_prompt` when all dynamic inputs are empty.
    #[allow(clippy::too_many_arguments)]
    fn compose_executor_prompt_from_registry(
        registry: &SpecialistRegistry,
        task_description: &str,
        parent_mission: &str,
        depth: usize,
        max_depth: usize,
        has_cli_agent: bool,
        task_id: Option<&str>,
        project_scope: Option<&str>,
    ) -> String {
        let mut all_dirs = Self::extract_directory_paths(parent_mission);
        for dir in Self::extract_directory_paths(task_description) {
            if !all_dirs.contains(&dir) {
                all_dirs.push(dir);
            }
        }

        // Render the base template (header + body) verbatim from the registry.
        let ctx = SpecialistRenderContext {
            mission: parent_mission.to_string(),
            task: task_description.to_string(),
            depth,
            max_depth,
            max_iterations: 0,
            goal_id: String::new(),
            working_dir: String::new(),
            is_scheduled: false,
            parent_session_id: String::new(),
            execution_mode: String::new(),
        };
        let base = registry.render(SpecialistKind::Executor, &ctx);

        // Build the dynamic mid-section (working directory + task contract)
        // that the legacy builder inserts between the sub-agent header and
        // "## Original User Request".
        let mut middle = String::new();
        if !all_dirs.is_empty() {
            middle.push_str("## WORKING DIRECTORY (CRITICAL)\n");
            middle.push_str("All files for this task are in: ");
            middle.push_str(&all_dirs.join(", "));
            middle.push_str("\n\nYou MUST use absolute paths when calling read_file, edit_file, write_file, search_files.\n");
            middle.push_str("Examples:\n");
            for dir in &all_dirs {
                middle.push_str(&format!(
                    "- read_file: path=\"{dir}/filename.py\"\n\
                     - edit_file: path=\"{dir}/filename.py\"\n\
                     - search_files: path=\"{dir}\"\n"
                ));
            }
            middle.push_str(
                "Do NOT use relative paths. Do NOT search in the default project directory.\n\n",
            );
        }

        if let Some(task_id) = task_id {
            let handoff = Self::build_executor_handoff(
                task_id,
                parent_mission,
                task_description,
                &[],
                project_scope,
            );
            middle.push_str(&handoff.render_prompt_section());
            middle.push_str("\n\n");
        }

        // Splice `middle` immediately before the "## Original User Request"
        // section so the result matches the legacy builder layout exactly.
        let marker = "## Original User Request";
        let mut prompt = if middle.is_empty() {
            base.clone()
        } else if let Some(idx) = base.find(marker) {
            let (head, tail) = base.split_at(idx);
            let mut out = String::with_capacity(base.len() + middle.len());
            out.push_str(head);
            out.push_str(&middle);
            out.push_str(tail);
            out
        } else {
            // Defensive: marker should always be present in the bundled
            // executor template; if it isn't, fall back to base + middle.
            warn!(
                "executor template missing '## Original User Request' marker; appending dynamic content"
            );
            let mut out = base.clone();
            out.push_str(&middle);
            out
        };

        if has_cli_agent {
            prompt.push_str(
                "\n- Delegation mode is active: `terminal`, `browser`, and `run_command` are not available here.\n\
                 Use direct file tools (`read_file`, `edit_file`, `write_file`, `search_files`) for narrow file work.\n\
                 Use `cli_agent` for shell/test flows or multi-step coding and research work.\n\
                 When you use `cli_agent`, always provide `action=\"run\"`, a concrete `prompt`, and `working_dir` when you know the repo path.",
            );
        }

        prompt
    }

    fn collect_full_child_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.root_tools
            .as_ref()
            .unwrap_or(&self.tools)
            .iter()
            .filter(|t| t.name() != "spawn_agent")
            .cloned()
            .collect()
    }

    async fn build_task_lead_spec(
        &self,
        full_tools: &[Arc<dyn Tool>],
        goal_id: &str,
        goal_description: &str,
        child_depth: usize,
        wrap_input: bool,
    ) -> TaskLeadSpec {
        // Scheduled goals are pre-authorized by the user, so the TaskLead needs
        // full tool access (including Action tools like terminal, write_file, etc.)
        // to complete work autonomously without human intervention.
        let is_scheduled = goal_has_scheduled_provenance(&self.state, goal_id, None).await;

        let mut tools: Vec<Arc<dyn Tool>> = if is_scheduled {
            // Scheduled goals: include Action tools so TaskLead can execute directly
            full_tools.to_vec()
        } else {
            // Start with Management + Universal tools.
            let mut base: Vec<Arc<dyn Tool>> = full_tools
                .iter()
                .filter(|t| matches!(t.tool_role(), ToolRole::Management | ToolRole::Universal))
                .cloned()
                .collect();
            // Always include essential Action tools as a direct-execution fallback.
            // When cli_agent or spawn_agent fail (auth errors, budget blocks, depth
            // limits), the TaskLead needs basic file and terminal access to avoid
            // wasting iterations retrying broken delegation paths.
            const ESSENTIAL_ACTION_TOOLS: &[&str] = &[
                "read_file",
                "write_file",
                "edit_file",
                "terminal",
                "search_files",
                "web_search",
                "web_fetch",
                "project_inspect",
            ];
            for tool in full_tools {
                if tool.tool_role() == ToolRole::Action
                    && ESSENTIAL_ACTION_TOOLS.contains(&tool.name())
                    && !base.iter().any(|t| t.name() == tool.name())
                {
                    base.push(tool.clone());
                }
            }
            base
        };

        let has_cli_agent = if let Some(cli_tool) = full_tools
            .iter()
            .find(|t| t.name() == "cli_agent" && t.is_available())
        {
            if !tools.iter().any(|t| t.name() == "cli_agent") {
                tools.push(cli_tool.clone());
            }
            true
        } else {
            false
        };

        tools.push(Arc::new(crate::tools::ManageGoalTasksTool::new(
            goal_id.to_string(),
            self.state.clone(),
        )));

        let goal_context = self
            .state
            .get_goal(goal_id)
            .await
            .ok()
            .flatten()
            .and_then(|g| g.context);

        // TODO(Task 13): move SpecialistRegistry to Agent::new and store as
        // `Arc<SpecialistRegistry>` field; for now we load per-spawn.
        let registry = SpecialistRegistry::load(None);
        let system_prompt = Self::compose_task_lead_prompt_from_registry(
            &registry,
            goal_id,
            goal_description,
            goal_context.as_deref(),
            child_depth,
            self.max_depth,
            has_cli_agent,
            is_scheduled,
        );

        let input_text = if wrap_input {
            format!(
                "Plan and execute this goal by creating tasks and delegating to executors:\n\n{}",
                goal_description
            )
        } else {
            goal_description.to_string()
        };

        TaskLeadSpec {
            tools,
            system_prompt,
            root_tools: full_tools.to_vec(),
            input_text,
        }
    }

    async fn resolve_task_lead_cancel_token(
        &self,
        goal_id: &str,
    ) -> Option<tokio_util::sync::CancellationToken> {
        if let Some(ref registry) = self.goal_token_registry {
            if let Some(token) = registry.child_token(goal_id).await {
                return Some(token);
            }
        }

        self.cancel_token.as_ref().map(|t| t.child_token())
    }

    fn collect_executor_expected_targets(
        mission: &str,
        task_description: &str,
        project_scope: Option<&str>,
    ) -> Vec<crate::traits::ToolTargetHint> {
        let mut targets = Vec::new();

        if let Some(scope) = project_scope {
            if let Some(target) = crate::traits::ToolTargetHint::new(
                crate::traits::ToolTargetHintKind::ProjectScope,
                scope,
            ) {
                targets.push(target);
            }
        }

        let mut add_dir = |dir: String| {
            if let Some(target) =
                crate::traits::ToolTargetHint::new(crate::traits::ToolTargetHintKind::Path, dir)
            {
                if !targets.iter().any(|existing| existing == &target) {
                    targets.push(target);
                }
            }
        };

        for dir in Self::extract_directory_paths(mission) {
            add_dir(dir);
        }
        for dir in Self::extract_directory_paths(task_description) {
            add_dir(dir);
        }

        targets
    }

    fn build_executor_handoff(
        task_id: &str,
        mission: &str,
        task_description: &str,
        tools: &[Arc<dyn Tool>],
        project_scope: Option<&str>,
    ) -> ExecutorHandoff {
        let expected_targets =
            Self::collect_executor_expected_targets(mission, task_description, project_scope);
        let allowed_targets = if let Some(scope) = project_scope {
            crate::traits::ToolTargetHint::new(
                crate::traits::ToolTargetHintKind::ProjectScope,
                scope,
            )
            .into_iter()
            .collect()
        } else {
            expected_targets.clone()
        };

        ExecutorHandoff {
            task_id: task_id.to_string(),
            mission: mission.to_string(),
            task_description: task_description.to_string(),
            target_scope: crate::agent::execution_state::TargetScope {
                allowed_targets,
                hard_fail_outside_scope: project_scope.is_some(),
            },
            expected_targets,
            allowed_tools: Some(
                tools
                    .iter()
                    .map(|tool| tool.name().to_string())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    async fn prepare_executor_task_handoff(
        &self,
        task_id: &str,
        handoff: &ExecutorHandoff,
        child_session: &str,
    ) {
        if let Ok(Some(mut task)) = self.state.get_task(task_id).await {
            task.status = "running".to_string();
            if task.started_at.is_none() {
                task.started_at = Some(chrono::Utc::now().to_rfc3339());
            }
            if let Ok(context) = persist_executor_handoff_context(task.context.as_deref(), handoff)
            {
                task.context = Some(context);
            }
            let _ = self.state.update_task(&task).await;
        }

        let activity = crate::traits::TaskActivity {
            id: 0,
            task_id: task_id.to_string(),
            activity_type: "executor_handoff".to_string(),
            tool_name: Some("spawn_agent".to_string()),
            tool_args: serde_json::to_string(handoff).ok(),
            result: None,
            success: Some(true),
            tokens_used: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        let _ = self.state.log_task_activity(&activity).await;

        if self.record_decision_points {
            let emitter = crate::events::EventEmitter::new(
                self.event_store.clone(),
                child_session.to_string(),
            );
            let _ = emitter
                .emit(
                    EventType::DecisionPoint,
                    DecisionPointData {
                        decision_type: DecisionType::ExecutionPlanningGate,
                        task_id: task_id.to_string(),
                        iteration: 0,
                        severity: crate::events::DiagnosticSeverity::Info,
                        code: Some("executor_handoff".to_string()),
                        metadata: json!({
                            "condition": "executor_handoff",
                            "executor_handoff": handoff,
                        }),
                        summary: "Persisted executor handoff contract before delegated execution."
                            .to_string(),
                    },
                )
                .await;
        }
    }

    async fn finalize_executor_task_outcome(
        &self,
        task_id: &str,
        response: Option<&str>,
        error: Option<&str>,
        child_session: &str,
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        let latest_task = self.state.get_task(task_id).await.ok().flatten();
        let structured =
            derive_executor_step_result(task_id, latest_task.as_ref(), response, error);
        let task_lead_summary = structured.render_task_lead_summary();

        if let Some(mut task) = latest_task {
            if let Ok(context) =
                persist_executor_result_context(task.context.as_deref(), &structured)
            {
                task.context = Some(context);
            }

            match error {
                Some(error) => {
                    task.status = "failed".to_string();
                    task.error = Some(error.to_string());
                    task.completed_at = Some(now.clone());
                    if task
                        .result
                        .as_deref()
                        .is_none_or(|result| result.trim().is_empty())
                    {
                        task.result = Some(structured.summary.clone());
                    }
                }
                None => {
                    match structured.task_outcome {
                        TaskValidationOutcome::TaskDone
                        | TaskValidationOutcome::ContinueWithNextStep => {
                            if task
                                .result
                                .as_deref()
                                .is_none_or(|result| result.trim().is_empty())
                            {
                                if let Some(response) = response {
                                    if !response.trim().is_empty() {
                                        task.result = Some(response.to_string());
                                    } else {
                                        task.result = Some(structured.summary.clone());
                                    }
                                } else {
                                    task.result = Some(structured.summary.clone());
                                }
                            }
                            task.status = "completed".to_string();
                            task.blocker = None;
                            task.error = None;
                        }
                        _ => {
                            task.result = Some(task_lead_summary.clone());
                            task.status = "blocked".to_string();
                            task.blocker = structured
                                .blocker
                                .clone()
                                .or_else(|| structured.exact_need.clone())
                                .or_else(|| Some(structured.summary.clone()));
                        }
                    }
                    task.completed_at = Some(now.clone());
                }
            }

            let _ = self.state.update_task(&task).await;
        }

        let activity = crate::traits::TaskActivity {
            id: 0,
            task_id: task_id.to_string(),
            activity_type: "step_validation".to_string(),
            tool_name: None,
            tool_args: None,
            result: serde_json::to_string(&structured).ok(),
            success: Some(error.is_none()),
            tokens_used: None,
            created_at: now.clone(),
        };
        let _ = self.state.log_task_activity(&activity).await;

        if self.record_decision_points {
            let emitter = crate::events::EventEmitter::new(
                self.event_store.clone(),
                child_session.to_string(),
            );
            let _ = emitter
                .emit(
                    EventType::DecisionPoint,
                    DecisionPointData {
                        decision_type: DecisionType::PostExecutionValidation,
                        task_id: task_id.to_string(),
                        iteration: 0,
                        severity: if error.is_some() {
                            crate::events::DiagnosticSeverity::Error
                        } else if matches!(structured.task_outcome, TaskValidationOutcome::TaskDone)
                        {
                            crate::events::DiagnosticSeverity::Info
                        } else {
                            crate::events::DiagnosticSeverity::Warning
                        },
                        code: Some("executor_task_validation".to_string()),
                        metadata: json!({
                            "condition": "executor_task_validation",
                            "step_validation_outcome": structured.step_outcome,
                            "task_validation_outcome": structured.task_outcome,
                            "executor_result": structured,
                        }),
                        summary: "Recorded delegated executor step/task validation outcome."
                            .to_string(),
                    },
                )
                .await;
        }
    }

    pub(crate) async fn mark_executor_task_timeout(&self, task_id: &str, timeout_secs: u64) {
        let session_id = format!("executor-timeout-{task_id}");
        let error = format!("Executor timed out after {timeout_secs} seconds");
        self.finalize_executor_task_outcome(task_id, None, Some(&error), &session_id)
            .await;
    }

    #[allow(clippy::too_many_arguments)]
    async fn create_child_agent(
        &self,
        mut tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        child_depth: usize,
        role: AgentRole,
        task_id: Option<String>,
        goal_id: Option<String>,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
        root_tools: Option<Vec<Arc<dyn Tool>>>,
        add_spawn_tool: bool,
        inherited_project_scope: Option<String>,
        max_iterations_override: Option<usize>,
        timeout_secs_override: Option<u64>,
    ) -> Arc<Agent> {
        let spawn_tool = if add_spawn_tool {
            Some(Arc::new(
                crate::tools::spawn::SpawnAgentTool::new_deferred(
                    self.max_response_chars,
                    self.timeout_secs,
                )
                .with_state(self.state.clone()),
            ))
        } else {
            None
        };

        if let Some(ref spawn_tool) = spawn_tool {
            tools.push(spawn_tool.clone());
        }

        let hub = match tokio::time::timeout(Duration::from_secs(2), self.hub.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring hub lock while spawning child agent");
                None
            }
        };

        let effective_max_iterations = max_iterations_override.unwrap_or(self.max_iterations);
        let effective_timeout_secs = timeout_secs_override.unwrap_or(self.timeout_secs);
        let child = Arc::new(Agent::with_depth(
            self.llm_runtime.clone(),
            self.state.clone(),
            self.event_store.clone(),
            tools,
            model,
            system_prompt,
            self.config_path.clone(),
            self.skills_dir.clone(),
            child_depth,
            self.max_depth,
            self.iteration_config.clone(),
            effective_max_iterations,
            self.max_iterations_cap,
            self.max_response_chars,
            effective_timeout_secs,
            self.max_facts,
            self.task_timeout,
            self.task_token_budget,
            self.llm_call_timeout,
            self.mcp_registry.clone(),
            self.verification_tracker.clone(),
            role,
            task_id,
            goal_id,
            cancel_token,
            self.goal_token_registry.clone(),
            hub,
            self.schedule_approved_sessions.clone(),
            self.billing_failed_models.clone(),
            self.record_decision_points,
            self.context_window_config.clone(),
            self.policy_config.clone(),
            self.path_aliases.clone(),
            inherited_project_scope,
            root_tools,
        ));

        if let Some(spawn_tool) = spawn_tool {
            spawn_tool.set_agent(Arc::downgrade(&child));
        }

        child
    }

    /// Spawn a child agent with an incremented depth and a focused mission.
    ///
    /// The child runs its own agentic loop in a fresh session and returns the
    /// final text response. It inherits the parent's provider, state, model,
    /// and non-spawn tools. If the child hasn't reached max_depth it also gets
    /// its own `spawn_agent` tool so it can recurse further.
    ///
    /// When `child_role` is `Some`, tools are scoped by role:
    /// - TaskLead: Management + Universal + cli_agent (if available) +
    ///   ManageGoalTasksTool + SpawnAgentTool
    /// - Executor: Action + Universal + ReportBlockerTool, NO SpawnAgentTool
    #[allow(clippy::too_many_arguments)]
    pub async fn spawn_child(
        self: &Arc<Self>,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        child_role: Option<AgentRole>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
        inherited_project_scope: Option<&str>,
        arg_specialist: Option<&str>,
    ) -> anyhow::Result<String> {
        if self.depth >= self.max_depth {
            anyhow::bail!(
                "Cannot spawn sub-agent: max recursion depth ({}) reached",
                self.max_depth
            );
        }

        let child_depth = self.depth + 1;
        let model = match tokio::time::timeout(Duration::from_secs(2), self.model.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring model lock while spawning child agent");
                self.llm_runtime.snapshot().primary_model()
            }
        };

        // Collect parent's non-spawn tools for the child.
        // Use root_tools if available (TaskLead spawning Executor needs the full
        // unfiltered set so Action tools aren't lost through double-filtering).
        let full_tools = self.collect_full_child_tools();

        // Apply role-based tool scoping when child_role is specified.
        let (scoped_tools, child_system_prompt, child_root_tools) = if let Some(role) = child_role {
            match role {
                AgentRole::TaskLead => {
                    let Some(goal_id) = goal_id else {
                        anyhow::bail!("Cannot spawn task lead without goal_id");
                    };
                    let TaskLeadSpec {
                        tools,
                        system_prompt,
                        root_tools,
                        input_text,
                    } = self
                        .build_task_lead_spec(&full_tools, goal_id, task, child_depth, false)
                        .await;
                    let cancel_token = self.resolve_task_lead_cancel_token(goal_id).await;
                    return self
                        .spawn_child_inner(
                            &tools,
                            model,
                            system_prompt,
                            child_depth,
                            mission,
                            &input_text,
                            status_tx,
                            channel_ctx,
                            user_role,
                            AgentRole::TaskLead,
                            Some(AgentRole::TaskLead),
                            arg_specialist,
                            true,
                            None,
                            Some(goal_id.to_string()),
                            Some(root_tools),
                            cancel_token,
                            inherited_project_scope,
                        )
                        .await;
                }
                AgentRole::Executor => {
                    let has_cli_agent = full_tools
                        .iter()
                        .any(|t| t.name() == "cli_agent" && t.is_available());
                    // Executors get Action + Universal tools.
                    let mut tools: Vec<Arc<dyn Tool>> = full_tools
                        .iter()
                        .filter(|t| matches!(t.tool_role(), ToolRole::Action | ToolRole::Universal))
                        .cloned()
                        .collect();
                    // Scheduled goals keep full tool access (terminal, browser, etc.)
                    // since they run unattended and need reliability over delegation elegance.
                    let is_scheduled_goal = if let Some(gid) = goal_id {
                        goal_has_scheduled_provenance(&self.state, gid, task_id).await
                    } else {
                        false
                    };
                    let effective_delegation_mode = has_cli_agent && !is_scheduled_goal;
                    if effective_delegation_mode {
                        // Delegation mode: avoid competing execution surfaces when
                        // cli_agent is available for the same task.
                        tools.retain(|t| !recall_guardrails::is_delegation_blocked_tool(t.name()));
                    }
                    // Add ReportBlockerTool
                    if let Some(tid) = task_id {
                        tools.push(Arc::new(crate::tools::ReportBlockerTool::new(
                            tid.to_string(),
                            self.state.clone(),
                        )));
                    }
                    // TODO(Task 13): move SpecialistRegistry to Agent::new
                    // and store as `Arc<SpecialistRegistry>` field; for now we
                    // load per-spawn.
                    let registry = SpecialistRegistry::load(None);
                    let prompt = Self::compose_executor_prompt_from_registry(
                        &registry,
                        task,
                        mission,
                        child_depth,
                        self.max_depth,
                        effective_delegation_mode,
                        task_id,
                        inherited_project_scope,
                    );
                    // Executors never get SpawnAgentTool
                    return self
                        .spawn_child_inner(
                            &tools,
                            model,
                            prompt,
                            child_depth,
                            mission,
                            task,
                            status_tx,
                            channel_ctx,
                            user_role,
                            role,
                            Some(role),
                            arg_specialist,
                            false, // no spawn tool
                            task_id.map(|s| s.to_string()),
                            goal_id.map(|s| s.to_string()),
                            None, // root_tools (executors don't spawn children)
                            None, // cancel token override
                            inherited_project_scope,
                        )
                        .await;
                }
                AgentRole::Orchestrator => {
                    // Orchestrator: full loop with spawn available (unless at max depth)
                    let at_max_depth = child_depth >= self.max_depth;
                    let depth_note = if at_max_depth {
                        "\nYou are at the maximum sub-agent depth. You CANNOT spawn further sub-agents; \
                        the `spawn_agent` tool is not available to you. Complete the task directly."
                    } else {
                        ""
                    };
                    let prompt = format!(
                        "{}\n\n## Sub-Agent Context\n\
                        You are a sub-agent (depth {}/{}) spawned to accomplish a specific mission.\n\
                        **Mission:** {}\n\n\
                        Focus exclusively on this mission. Be concise. Return your findings/results \
                        directly — they will be consumed by the parent agent.{}",
                        self.system_prompt, child_depth, self.max_depth, mission, depth_note
                    );
                    (full_tools, prompt, None)
                }
            }
        } else {
            // Legacy behavior: no role scoping
            let at_max_depth = child_depth >= self.max_depth;
            let depth_note = if at_max_depth {
                "\nYou are at the maximum sub-agent depth. You CANNOT spawn further sub-agents; \
                the `spawn_agent` tool is not available to you. Complete the task directly."
            } else {
                ""
            };
            let prompt = format!(
                "{}\n\n## Sub-Agent Context\n\
                You are a sub-agent (depth {}/{}) spawned to accomplish a specific mission.\n\
                **Mission:** {}\n\n\
                Focus exclusively on this mission. Be concise. Return your findings/results \
                directly — they will be consumed by the parent agent.{}",
                self.system_prompt, child_depth, self.max_depth, mission, depth_note
            );
            (full_tools, prompt, None)
        };

        let effective_role = child_role.unwrap_or(AgentRole::Orchestrator);
        let can_spawn = child_depth < self.max_depth && effective_role != AgentRole::Executor;

        // For TaskLead, pass goal_id; other roles get no goal context injection.
        let goal_for_child = if effective_role == AgentRole::TaskLead {
            goal_id.map(|s| s.to_string())
        } else {
            None
        };

        self.spawn_child_inner(
            &scoped_tools,
            model,
            child_system_prompt,
            child_depth,
            mission,
            task,
            status_tx,
            channel_ctx,
            user_role,
            effective_role,
            child_role,
            arg_specialist,
            can_spawn,
            None,             // task_id (executor activity tracking)
            goal_for_child,   // goal_id (task lead context injection)
            child_root_tools, // root_tools for TaskLead → Executor inheritance
            None,             // cancel token override
            inherited_project_scope,
        )
        .await
    }

    /// Internal helper to create and run a child agent.
    #[allow(clippy::too_many_arguments)]
    async fn spawn_child_inner(
        self: &Arc<Self>,
        tools: &[Arc<dyn Tool>],
        model: String,
        system_prompt: String,
        child_depth: usize,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        role: AgentRole,
        original_child_role: Option<AgentRole>,
        arg_specialist: Option<&str>,
        add_spawn_tool: bool,
        task_id: Option<String>,
        goal_id: Option<String>,
        root_tools: Option<Vec<Arc<dyn Tool>>>,
        cancel_token_override: Option<tokio_util::sync::CancellationToken>,
        inherited_project_scope: Option<&str>,
    ) -> anyhow::Result<String> {
        let specialist_kind =
            Self::resolve_specialist_kind(original_child_role, arg_specialist, mission, task);
        let child_session = Self::build_specialist_session_id(specialist_kind, Uuid::new_v4());

        // Apply specialist overrides (tool allowlist + budgets) from the
        // registry. Tools that fall outside the role scope or aren't known
        // are dropped with a warn (see `intersect_tools`). Budget overrides
        // are clamped via `clamp_max_iterations` / `clamp_timeout`.
        //
        // TODO(Task 13): move the registry to an Agent field so we don't
        // reload bundled markdown on every spawn.
        let registry = SpecialistRegistry::load(None);
        let def = registry.get(specialist_kind);
        let scoped_tools: Vec<Arc<dyn Tool>> = if let Some(declared) = def.tools.as_deref() {
            Self::apply_specialist_tool_allowlist(
                tools.to_vec(),
                declared,
                specialist_kind,
                original_child_role,
                tools,
            )
        } else {
            tools.to_vec()
        };
        let max_iterations_override = def.max_iterations.map(|raw| {
            specialist_validation::clamp_max_iterations(
                specialist_kind,
                raw,
                self.max_iterations_cap,
            )
        });
        let timeout_secs_override = def.timeout_secs.map(|raw| {
            specialist_validation::clamp_timeout(specialist_kind, raw, self.timeout_secs.max(1))
        });
        if let Some(declared_model) = def.model.as_deref() {
            // Provider-side model availability checks are deferred (Task 13+).
            // For now we warn that the override was observed; the spawn keeps
            // the parent-selected model.
            warn!(
                kind = specialist_kind.as_str(),
                model = declared_model,
                "specialist model override declared; provider availability check deferred — using parent model"
            );
        }

        info!(
            parent_depth = self.depth,
            child_depth,
            child_session = %child_session,
            specialist_kind = specialist_kind.as_str(),
            mission,
            ?role,
            "Spawning sub-agent"
        );

        // Emit SubAgentSpawn event
        {
            let emitter =
                crate::events::EventEmitter::new(self.event_store.clone(), child_session.clone());
            let _ = emitter
                .emit(
                    EventType::SubAgentSpawn,
                    SubAgentSpawnData {
                        child_session_id: child_session.clone(),
                        specialist_kind: Some(specialist_kind.as_str().to_string()),
                        mission: mission.to_string(),
                        task: task.chars().take(500).collect(),
                        depth: child_depth as u32,
                        parent_task_id: None,
                    },
                )
                .await;
        }

        let start = std::time::Instant::now();
        // Save task_id for post-completion knowledge extraction (Phase 4)
        let saved_task_id = task_id.clone();
        if role == AgentRole::Executor {
            if let Some(task_id) = saved_task_id.as_deref() {
                let handoff = Self::build_executor_handoff(
                    task_id,
                    mission,
                    task,
                    &scoped_tools,
                    inherited_project_scope,
                );
                self.prepare_executor_task_handoff(task_id, &handoff, &child_session)
                    .await;
            }
        }
        let cancel_token =
            cancel_token_override.or_else(|| self.cancel_token.as_ref().map(|t| t.child_token()));
        let child = self
            .create_child_agent(
                scoped_tools,
                model,
                system_prompt,
                child_depth,
                role,
                task_id,
                goal_id,
                cancel_token,
                root_tools,
                add_spawn_tool,
                inherited_project_scope.map(ToOwned::to_owned),
                max_iterations_override,
                timeout_secs_override,
            )
            .await;
        let result = child
            .handle_message(
                &child_session,
                task,
                status_tx,
                user_role,
                channel_ctx,
                None,
            )
            .await;

        if role == AgentRole::Executor {
            if let Some(task_id) = saved_task_id.as_deref() {
                let error_text = result.as_ref().err().map(|error| error.to_string());
                self.finalize_executor_task_outcome(
                    task_id,
                    result.as_ref().ok().map(String::as_str),
                    error_text.as_deref(),
                    &child_session,
                )
                .await;
            }
        }

        let duration = start.elapsed();

        // Emit SubAgentComplete event
        {
            let emitter =
                crate::events::EventEmitter::new(self.event_store.clone(), child_session.clone());
            let (success, summary) = match &result {
                Ok(response) => (true, response.chars().take(200).collect()),
                Err(e) => (false, format!("{}", e)),
            };
            let _ = emitter
                .emit(
                    EventType::SubAgentComplete,
                    SubAgentCompleteData {
                        child_session_id: child_session,
                        specialist_kind: Some(specialist_kind.as_str().to_string()),
                        success,
                        result_summary: summary,
                        duration_secs: duration.as_secs(),
                        parent_task_id: None,
                    },
                )
                .await;
        }

        // Spawn background knowledge extraction for completed executor tasks.
        if let Some(ref task_id) = saved_task_id {
            if result.is_ok() {
                if let Ok(Some(completed_task)) = self.state.get_task(task_id).await {
                    if completed_task.status == "completed" {
                        let state = self.state.clone();
                        let provider = self.llm_runtime.provider();
                        let tid = task_id.clone();
                        let model = match tokio::time::timeout(
                            Duration::from_secs(2),
                            self.fallback_model.read(),
                        )
                        .await
                        {
                            Ok(guard) => guard.clone(),
                            Err(_) => {
                                warn!(
                                    task_id = %tid,
                                    "Timed out acquiring fallback_model lock for task knowledge extraction"
                                );
                                self.llm_runtime.snapshot().primary_model()
                            }
                        };
                        tokio::spawn(async move {
                            if let Err(e) = crate::memory::task_learning::extract_task_knowledge(
                                state,
                                provider,
                                model,
                                completed_task,
                            )
                            .await
                            {
                                warn!(
                                    task_id = %tid,
                                    error = %e,
                                    "Task knowledge extraction failed"
                                );
                            }
                        });
                    }
                }
            }
        }

        result
    }

    /// Spawn a task lead for a goal. Called from handle_message (&self context).
    ///
    /// This is a simplified version of spawn_child that doesn't require &Arc<Self>,
    /// since handle_message takes &self. The task lead gets management + universal tools
    /// plus ManageGoalTasksTool and SpawnAgentTool (for spawning executors).
    pub(super) fn spawn_task_lead(
        &self,
        goal_id: &str,
        goal_description: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<String>> + Send + '_>>
    {
        // Box::pin to break async recursion (handle_message -> spawn_task_lead -> handle_message)
        let goal_id = goal_id.to_string();
        let goal_description = goal_description.to_string();
        let user_text = user_text.to_string();
        Box::pin(async move {
            let goal_id = &goal_id;
            let goal_description = &goal_description;
            let user_text = &user_text;
            if self.depth >= self.max_depth {
                anyhow::bail!(
                    "Cannot spawn task lead: max recursion depth ({}) reached",
                    self.max_depth
                );
            }

            let child_depth = self.depth + 1;
            let model = match tokio::time::timeout(Duration::from_secs(2), self.model.read()).await
            {
                Ok(guard) => guard.clone(),
                Err(_) => {
                    warn!("Timed out acquiring model lock while spawning task lead");
                    self.llm_runtime.snapshot().primary_model()
                }
            };

            let full_tools = self.collect_full_child_tools();
            let TaskLeadSpec {
                tools,
                system_prompt,
                root_tools,
                input_text,
            } = self
                .build_task_lead_spec(&full_tools, goal_id, user_text, child_depth, true)
                .await;
            let mission = format!(
                "Task Lead for goal: {}",
                &goal_description[..goal_description.len().min(100)]
            );
            let specialist_kind = SpecialistKind::TaskLead;
            let child_session = Self::build_specialist_session_id(specialist_kind, Uuid::new_v4());

            info!(
                parent_depth = self.depth,
                child_depth,
                child_session = %child_session,
                specialist_kind = specialist_kind.as_str(),
                goal_id,
                "Spawning task lead"
            );

            // Emit SubAgentSpawn event
            {
                let emitter = crate::events::EventEmitter::new(
                    self.event_store.clone(),
                    child_session.clone(),
                );
                let _ = emitter
                    .emit(
                        EventType::SubAgentSpawn,
                        SubAgentSpawnData {
                            child_session_id: child_session.clone(),
                            specialist_kind: Some(specialist_kind.as_str().to_string()),
                            mission: mission.clone(),
                            task: input_text.chars().take(500).collect(),
                            depth: child_depth as u32,
                            parent_task_id: None,
                        },
                    )
                    .await;
            }

            let start = std::time::Instant::now();
            let child_cancel_token = self.resolve_task_lead_cancel_token(goal_id).await;
            let child = self
                .create_child_agent(
                    tools,
                    model,
                    system_prompt,
                    child_depth,
                    AgentRole::TaskLead,
                    None,                      // task_id (task leads aren't executors)
                    Some(goal_id.to_string()), // goal_id (context injection for child)
                    child_cancel_token,
                    Some(root_tools), // root_tools for Executor inheritance
                    true,
                    None,
                    None, // max_iterations override (task leads use parent default)
                    None, // timeout_secs override
                )
                .await;

            let result = child
                .handle_message(
                    &child_session,
                    &input_text,
                    status_tx,
                    user_role,
                    channel_ctx,
                    None,
                )
                .await;

            let duration = start.elapsed();

            // Emit SubAgentComplete event
            {
                let emitter = crate::events::EventEmitter::new(
                    self.event_store.clone(),
                    child_session.clone(),
                );
                let (success, summary) = match &result {
                    Ok(response) => (true, response.chars().take(200).collect()),
                    Err(e) => (false, format!("{}", e)),
                };
                let _ = emitter
                    .emit(
                        EventType::SubAgentComplete,
                        SubAgentCompleteData {
                            child_session_id: child_session,
                            specialist_kind: Some(specialist_kind.as_str().to_string()),
                            success,
                            result_summary: summary,
                            duration_secs: duration.as_secs(),
                            parent_task_id: None,
                        },
                    )
                    .await;
            }

            result
        }) // end Box::pin(async move { ... })
    }

    /// Build system prompt for a Task Lead agent.
    ///
    /// Retained as the oracle for `specialists::equivalence_tests` and as the
    /// reference legacy implementation. Production callers now go through
    /// `compose_task_lead_prompt_from_registry`, which renders the same base
    /// text from `task_lead.md` and appends the same dynamic sections.
    #[allow(dead_code)] // test-only oracle; production uses the registry path
    pub(in crate::agent) fn build_task_lead_prompt(
        goal_id: &str,
        goal_description: &str,
        goal_context: Option<&str>,
        depth: usize,
        max_depth: usize,
        has_cli_agent: bool,
        is_scheduled: bool,
    ) -> String {
        let execution_mode = if is_scheduled {
            "You have full tool access including `terminal`. For simple steps (single shell commands, \
             file writes), execute them directly. For complex multi-step work, you may still delegate \
             to executors via the workflow below."
        } else {
            "Your primary job is to plan and delegate work via executors or cli_agent. \
             However, you also have direct access to essential tools (read_file, write_file, \
             edit_file, terminal, search_files). Use delegation first, but if delegation fails \
             (cli_agent errors, spawn_agent blocked, executor failures), switch to direct \
             execution with your own tools rather than retrying broken delegation paths."
        };

        let mut prompt = format!(
            "You are a Task Lead managing goal: {goal_id}\n\
             Goal: {goal_description}\n\n\
             You are a sub-agent (depth {depth}/{max_depth}).\n\
             {execution_mode}\n\n\
             ## Workflow\n\
             1. Analyze the goal and break it into concrete tasks using manage_goal_tasks(create_task)\n\
                - Start with 2-5 tasks for the NEXT PHASE (not the entire project)\n\
                - After those tasks complete, reassess and create more tasks if the goal isn't done\n\
                - Set `depends_on` (array of task IDs) for tasks that require prior tasks to complete\n\
                - Set `parallel_group` for tasks that belong to the same logical phase\n\
                - Set `idempotent: true` for tasks safe to retry on failure\n\
                - Set `task_order` for display ordering\n\
             2. Before spawning an executor, claim the task: manage_goal_tasks(claim_task, task_id=...)\n\
                - This verifies dependencies are met and atomically reserves the task\n\
                - If claiming fails due to unmet dependencies, work on other available tasks first\n\
             3. Spawn an executor: spawn_agent(mission=..., task=..., task_id=<the task ID>)\n\
                - Always pass the task_id so executor activity is tracked\n\
             4. After each executor returns, update: manage_goal_tasks(update_task, task_id, status, result)\n\
             5. If a task fails and is idempotent: manage_goal_tasks(retry_task, task_id) then re-spawn\n\
                - If not idempotent or max retries exceeded: create alternative task or fail the goal\n\
             6. When all tasks complete: manage_goal_tasks(complete_goal, summary)\n\n\
             ## Rules\n\
             - Keep each planning step small: 2-5 tasks at a time, then iterate\n\
             - Spawn executors one at a time (sequential execution)\n\
             - Each executor gets a single, focused task\n\
             - Always check list_tasks before spawning the next executor\n\
             - If an executor reports a blocker, inspect the recorded task status/result and resolve it or adjust the plan\n\
             - Executors persist a structured handoff/result contract onto the claimed task record; do not treat vague prose alone as proof of completion\n\
             - When finishing the goal, your final reply MUST include concrete executor results (outputs, paths, data), not just \"goal completed\"\n\n\
             ## Pre-flight and Verification\n\
             - Before any task that modifies external state (deploy, publish, push, send, upload, migrate), \
             create a prerequisite-check task that verifies readiness (e.g., all changes committed, \
             dependencies installed, credentials valid, build passing)\n\
             - After any task that modifies external state, ALWAYS create a verification task that \
             confirms the change was applied correctly (e.g., fetch the live URL, query the database, \
             check the published package version)\n\
             - Never mark the goal as complete until the verification task passes\n\
             - If verification fails, create a remediation task to fix the issue and re-verify"
        );

        if let Some(ctx) = goal_context {
            prompt.push_str(&format!(
                "\n\n## Prior Knowledge\n\
                 The following knowledge was gathered from previous tasks and may be relevant:\n{}",
                format_goal_context(ctx)
            ));
        }

        if has_cli_agent {
            prompt.push_str(
                "\n\n## CLI Agent Delegation\n\
                 You have direct access to `cli_agent` (a specialized coding/research agent running on this machine).\n\
                 Treat `cli_agent` as a delegation surface, not as a reason to skip task structure.\n\
                 If the work should stay tied to a claimed task with executor results or blocker handling, claim the task and use `spawn_agent`.\n\
                 Prefer direct `cli_agent` calls for focused execution-heavy work when you do not need aidaemon-only tools in the child.\n\
                 When calling `cli_agent`, use `action=\"run\"` and include a non-empty `prompt` describing the work.\n\
                 Pass `working_dir` whenever the task targets a specific repo or directory.\n\
                 Example: `cli_agent(action=\"run\", prompt=\"Inspect the latest service logs, patch the root cause, run cargo fmt, and run the narrowest relevant tests\", working_dir=\"/absolute/project/path\")`.\n\
                 Note: If cli_agent fails repeatedly (auth errors, timeouts, environment issues), do NOT keep retrying. Switch to using your direct tools (read_file, write_file, edit_file, terminal) to complete the work yourself.",
            );
        }

        prompt
    }

    /// Extract absolute directory paths from text (e.g. /tmp/debugme3/, /home/user/project/).
    /// Returns deduplicated list of directory paths found.
    fn extract_directory_paths(text: &str) -> Vec<String> {
        let mut dirs = Vec::new();
        // Match absolute paths: /word/word... optionally ending with /
        for word in text.split_whitespace() {
            // Strip trailing punctuation
            let clean = word.trim_end_matches(|c: char| {
                c == '.' || c == ',' || c == ':' || c == ';' || c == ')' || c == '\''
            });
            if clean.starts_with('/')
                && clean.len() > 2
                && !clean.starts_with("//")
                // Must have at least 2 path components
                && clean.matches('/').count() >= 2
                // Skip common non-directory paths
                && !clean.ends_with(".rs")
                && !clean.ends_with(".toml")
            {
                // Normalize to directory (remove trailing filename if it has an extension)
                let path = std::path::Path::new(clean);
                let dir = if path.extension().is_some() {
                    // Looks like a file path — take parent directory
                    path.parent()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| clean.to_string())
                } else {
                    clean.trim_end_matches('/').to_string()
                };
                if !dirs.contains(&dir) {
                    dirs.push(dir);
                }
            }
        }
        dirs
    }

    /// Build system prompt for an Executor agent.
    ///
    /// Retained as the oracle for `specialists::equivalence_tests` and as the
    /// reference legacy implementation. Production callers now go through
    /// `compose_executor_prompt_from_registry`, which renders the same base
    /// text from `executor.md` and splices in the same dynamic sections.
    #[allow(dead_code)] // test-only oracle; production uses the registry path
    pub(in crate::agent) fn build_executor_prompt(
        task_description: &str,
        parent_mission: &str,
        depth: usize,
        max_depth: usize,
        has_cli_agent: bool,
        task_id: Option<&str>,
        project_scope: Option<&str>,
    ) -> String {
        // Extract directory paths from both parent mission and task description
        let mut all_dirs = Self::extract_directory_paths(parent_mission);
        for dir in Self::extract_directory_paths(task_description) {
            if !all_dirs.contains(&dir) {
                all_dirs.push(dir);
            }
        }

        let mut prompt = format!(
            "You are an Executor. Complete this single task and return your results.\n\n\
             You are a sub-agent (depth {depth}/{max_depth}).\n\n"
        );

        // Inject extracted directory paths at the very top — before anything else
        if !all_dirs.is_empty() {
            prompt.push_str("## WORKING DIRECTORY (CRITICAL)\n");
            prompt.push_str("All files for this task are in: ");
            prompt.push_str(&all_dirs.join(", "));
            prompt.push_str("\n\nYou MUST use absolute paths when calling read_file, edit_file, write_file, search_files.\n");
            prompt.push_str("Examples:\n");
            for dir in &all_dirs {
                prompt.push_str(&format!(
                    "- read_file: path=\"{dir}/filename.py\"\n\
                     - edit_file: path=\"{dir}/filename.py\"\n\
                     - search_files: path=\"{dir}\"\n"
                ));
            }
            prompt.push_str(
                "Do NOT use relative paths. Do NOT search in the default project directory.\n\n",
            );
        }

        if let Some(task_id) = task_id {
            let handoff = Self::build_executor_handoff(
                task_id,
                parent_mission,
                task_description,
                &[],
                project_scope,
            );
            prompt.push_str(&handoff.render_prompt_section());
            prompt.push_str("\n\n");
        }

        prompt.push_str(&format!(
            "## Original User Request\n\
             {parent_mission}\n\n\
             ## Your Specific Task\n\
             {task_description}\n\n\
             Rules:\n\
             - Focus ONLY on your specific task. Do not expand scope.\n\
             - EXECUTE the task immediately. Do NOT ask for permission or confirmation.\n\
             - Do NOT ask \"Shall I proceed?\" or \"Would you like me to...?\". Just do the work.\n\
             - There is no human in this loop — you are an autonomous executor.\n\
             - For modifying code: use `edit_file` (preferred) or `write_file`. NEVER use `python3 -c` to rewrite files — it is blocked.\n\
             - For reading code: use `read_file` with ABSOLUTE paths. For searching: use `search_files` with ABSOLUTE directory path.\n\
             - For running commands, use the execution surface actually available in your tool set.\n\
             - If `terminal` is available, keep commands simple and single-line.\n\
             - If `terminal` is available, scope commands to explicit directories and avoid scanning `target`, `node_modules`, and `.git` trees.\n\
             - If you encounter ambiguity or a blocker you cannot resolve, use report_blocker immediately.\n\
             - When using report_blocker, include outcome, reason, partial_work when applicable, exact_need, next_step, and target.\n\
             - Return the FULL content you produced — not a meta-description of what you did.\n\
             - NEVER return just \"I researched X\" or \"Generated a report about Y\". Return the actual content.\n\
             - Include specific outputs (file paths, data retrieved, commands run).\n\
             - If you create or write a file, include its FULL ABSOLUTE PATH in your result text.\n\
             - Do NOT claim the overall goal is complete. You may only finish this single task.\n\
             - Do NOT spawn sub-agents."
        ));

        if has_cli_agent {
            prompt.push_str(
                "\n- Delegation mode is active: `terminal`, `browser`, and `run_command` are not available here.\n\
                 Use direct file tools (`read_file`, `edit_file`, `write_file`, `search_files`) for narrow file work.\n\
                 Use `cli_agent` for shell/test flows or multi-step coding and research work.\n\
                 When you use `cli_agent`, always provide `action=\"run\"`, a concrete `prompt`, and `working_dir` when you know the repo path.",
            );
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::Agent;
    use crate::traits::{AgentRole, SpecialistKind};
    use uuid::Uuid;

    #[test]
    fn specialist_arg_wins_over_heuristic() {
        let kind = Agent::resolve_specialist_kind(
            None,
            Some("research"),
            "Implement the sorting algorithm in src/sort.rs",
            "Add a test for the edge case",
        );
        assert_eq!(kind, SpecialistKind::Research);
    }

    #[test]
    fn invalid_specialist_arg_falls_back_to_heuristic() {
        let kind = Agent::resolve_specialist_kind(
            None,
            Some("not_a_real_kind"),
            "Implement the sorting algorithm in src/sort.rs",
            "Add a unit test",
        );
        assert_eq!(kind, SpecialistKind::Code);
    }

    #[test]
    fn role_typed_spawn_ignores_specialist_arg() {
        let kind = Agent::resolve_specialist_kind(
            Some(AgentRole::TaskLead),
            Some("code"),
            "any mission",
            "any task",
        );
        assert_eq!(kind, SpecialistKind::TaskLead);
    }

    #[test]
    fn task_lead_arg_is_rejected_falling_back_to_heuristic() {
        // "task_lead" is role-typed only — when passed as an arg (without
        // explicit role) it must be ignored and the heuristic should run.
        let kind = Agent::resolve_specialist_kind(
            None,
            Some("task_lead"),
            "Implement the sorting algorithm in src/sort.rs",
            "Add a unit test",
        );
        assert_eq!(kind, SpecialistKind::Code);
    }

    #[test]
    fn specialist_kind_prefers_artifact_writer_for_report_files() {
        let kind = Agent::select_specialist_kind(
            AgentRole::Executor,
            "Compile and format morning AI job preparation tips report",
            "Create a markdown report and save it as ~/morning_ai_job_preparation_tips_report.md",
        );
        assert_eq!(kind, SpecialistKind::ArtifactWriter);
    }

    #[test]
    fn specialist_session_id_uses_kind_prefix() {
        let id = Uuid::parse_str("344ee9c6-a93f-48ef-84bf-ae3f4d68fc5b").unwrap();
        let session_id = Agent::build_specialist_session_id(SpecialistKind::Research, id);
        assert_eq!(
            session_id,
            "specialist:research:344ee9c6-a93f-48ef-84bf-ae3f4d68fc5b"
        );
    }

    #[test]
    fn specialist_session_id_format_holds_for_every_kind() {
        let id = Uuid::parse_str("344ee9c6-a93f-48ef-84bf-ae3f4d68fc5b").unwrap();
        let kinds = [
            SpecialistKind::TaskLead,
            SpecialistKind::Executor,
            SpecialistKind::Research,
            SpecialistKind::ArtifactWriter,
            SpecialistKind::Code,
            SpecialistKind::BrowserVerifier,
            SpecialistKind::Review,
            SpecialistKind::CommsDraft,
            SpecialistKind::Generic,
        ];
        for kind in kinds {
            let session_id = Agent::build_specialist_session_id(kind, id);
            assert!(
                session_id.starts_with("specialist:"),
                "{:?}: missing prefix in {}",
                kind,
                session_id
            );
            let expected_segment = format!(":{}:", kind.as_str());
            assert!(
                session_id.contains(&expected_segment),
                "{:?}: missing kind segment in {}",
                kind,
                session_id
            );
            assert!(
                session_id.ends_with(&id.to_string()),
                "{:?}: missing uuid suffix in {}",
                kind,
                session_id
            );
        }
    }

    #[test]
    fn specialist_session_ids_are_unique_per_invocation() {
        let a = Agent::build_specialist_session_id(SpecialistKind::Code, Uuid::new_v4());
        let b = Agent::build_specialist_session_id(SpecialistKind::Code, Uuid::new_v4());
        assert_ne!(a, b, "fresh uuids must produce unique session ids");
    }

    #[test]
    fn specialist_kind_browser_check_wins_over_code_for_smoke_tests() {
        let kind = Agent::select_specialist_kind(
            AgentRole::Executor,
            "Smoke-check the landing page",
            "Open the homepage in a browser and run the smoke test",
        );
        assert_eq!(kind, SpecialistKind::BrowserVerifier);
    }

    #[test]
    fn specialist_kind_code_still_wins_for_cargo_test() {
        let kind = Agent::select_specialist_kind(
            AgentRole::Executor,
            "Fix the broken assertion in math::add",
            "Run cargo test until the failing case in src/math.rs passes",
        );
        assert_eq!(kind, SpecialistKind::Code);
    }

    #[test]
    fn executor_prompt_includes_search_files_preference() {
        let prompt =
            Agent::build_executor_prompt("find async fns", "user request", 2, 4, false, None, None);
        assert!(prompt.contains("search_files"));
        assert!(prompt.contains("edit_file"));
        assert!(prompt.contains("avoid scanning `target`, `node_modules`, and `.git`"));
    }

    #[test]
    fn executor_prompt_extracts_directory_paths_from_mission() {
        let prompt = Agent::build_executor_prompt(
            "Fix the bug in task_scheduler.py",
            "There are 5 bugs in /tmp/debugme3/. Fix them all.",
            2,
            4,
            false,
            None,
            Some("/tmp/debugme3"),
        );
        assert!(
            prompt.contains("WORKING DIRECTORY"),
            "Should have WORKING DIRECTORY section"
        );
        assert!(
            prompt.contains("/tmp/debugme3"),
            "Should extract /tmp/debugme3 path"
        );
        assert!(
            prompt.contains("read_file: path=\"/tmp/debugme3/filename.py\""),
            "Should show read_file example"
        );
    }

    #[test]
    fn extract_directory_paths_basic() {
        let dirs = Agent::extract_directory_paths("Fix bugs in /tmp/debugme3/ and run tests");
        assert_eq!(dirs, vec!["/tmp/debugme3"]);

        let dirs = Agent::extract_directory_paths("Edit /home/user/project/foo.py");
        assert_eq!(dirs, vec!["/home/user/project"]);

        let dirs = Agent::extract_directory_paths("No paths here");
        assert!(dirs.is_empty());
    }

    #[test]
    fn task_lead_prompt_requires_concrete_final_results() {
        let prompt =
            Agent::build_task_lead_prompt("goal_1", "audit disk usage", None, 1, 3, false, false);
        assert!(prompt.contains("final reply MUST include concrete executor results"));
        assert!(prompt.contains("not just \"goal completed\""));
    }

    #[test]
    fn executor_prompt_mentions_cli_delegate_mode_when_cli_present() {
        let prompt =
            Agent::build_executor_prompt("refactor auth", "user request", 2, 4, true, None, None);
        assert!(prompt.contains("Delegation mode is active"));
        assert!(prompt.contains("`terminal`, `browser`, and `run_command` are not available"));
        assert!(!prompt.contains("prefer `terminal` directly"));
        assert!(prompt.contains("action=\"run\""));
        assert!(prompt.contains("working_dir"));
    }

    #[test]
    fn executor_prompt_includes_task_contract_when_task_id_present() {
        let prompt = Agent::build_executor_prompt(
            "patch /tmp/demo/src/main.rs",
            "fix the scoped regression in /tmp/demo",
            2,
            4,
            false,
            Some("task-123"),
            Some("/tmp/demo"),
        );
        assert!(prompt.contains("## Task Contract"));
        assert!(prompt.contains("task_id: task-123"));
        assert!(prompt.contains("allowed targets (hard boundary): /tmp/demo"));
        assert!(prompt.contains("report_blocker"));
    }

    #[test]
    fn task_lead_prompt_mentions_cli_agent_when_available() {
        let prompt =
            Agent::build_task_lead_prompt("goal_2", "build release", None, 1, 3, true, false);
        assert!(prompt.contains("## CLI Agent Delegation"));
        assert!(prompt.contains("Treat `cli_agent` as a delegation surface"));
        assert!(prompt.contains("claim the task and use `spawn_agent`"));
        assert!(prompt.contains("action=\"run\""));
        assert!(prompt.contains("working_dir"));
        assert!(prompt.contains("do NOT keep retrying"));
    }

    #[test]
    fn scheduled_task_lead_prompt_allows_direct_execution() {
        let prompt =
            Agent::build_task_lead_prompt("goal_3", "deploy blog", None, 1, 3, false, true);
        assert!(
            prompt.contains("full tool access including `terminal`"),
            "Scheduled task lead should mention terminal access"
        );
        assert!(
            !prompt.contains("MUST NOT execute tasks yourself"),
            "Scheduled task lead should NOT prohibit direct execution"
        );
    }

    #[test]
    fn non_scheduled_task_lead_prompt_allows_fallback_direct_execution() {
        let prompt =
            Agent::build_task_lead_prompt("goal_4", "deploy blog", None, 1, 3, false, false);
        assert!(
            prompt.contains("plan and delegate work"),
            "Non-scheduled task lead should prefer delegation"
        );
        assert!(
            prompt.contains("switch to direct execution"),
            "Non-scheduled task lead should allow fallback to direct execution"
        );
        assert!(
            !prompt.contains("full tool access including `terminal`"),
            "Non-scheduled task lead should NOT mention full tool access"
        );
    }
}
