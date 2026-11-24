//! The `Tool` and `ToolSemantics` impls and the call's typed outcome.

use super::*;

#[async_trait]
impl Tool for CliAgentTool {
    fn name(&self) -> &str {
        "cli_agent"
    }

    fn description(&self) -> &str {
        "Delegate a task to a CLI-based AI coding agent running on this machine"
    }

    fn schema(&self) -> Value {
        let tool_names = self.tool_names.read().unwrap();

        let tools_help = tool_names.join(", ");
        let names_vec: Vec<Value> = tool_names.iter().map(|n| json!(n)).collect();

        json!({
            "name": "cli_agent",
            "description": format!(
                "Delegate coding or research to an installed CLI agent. Available agents: {}. \
                 Hand it the task directly: it reads the folder itself, so don't look around \
                 first unless what you ask depends on something you must find out before. \
                 Runs can be checked or cancelled. The owner sees a run's reply as the agent's \
                 own message, its report checked against the run's record (the files it \
                 changed, the commands it ran); the result says whether the record bears out \
                 every claim. That check is the verification: don't plan your own step to \
                 verify what the agent changes, and when the record bears it out, don't \
                 re-read or restate its work. If nothing else in the request remains, end your \
                 turn with no reply. {}",
                tools_help,
                CLI_SANDBOX.describe()
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run", "check", "cancel", "list"],
                        "description": "run, check, cancel, or list"
                    },
                    "tool": {
                        "type": "string",
                        "enum": names_vec,
                        "description": "Agent name"
                    },
                    "title": {
                        "type": "string",
                        "description": "For run: what the work is in a few plain words, e.g. 'Refactor the parser'. The owner sees it on the run's card."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Task prompt"
                    },
                    "continue_run": {
                        "type": "string",
                        "description": "For run: an earlier run's id in this conversation to follow up on. Its agent goes on in its own session, in its folder, knowing what it did; give it the access the follow-up needs as for any run."
                    },
                    "new_folders": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Absolute folders that don't exist yet, created empty for this run to work in (a new project)."
                    },
                    "scope": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Files or directories this run will change."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Absolute execution directory; readable but not implicitly writable."
                    },
                    "read_paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Additional exact readable files/directories for run."
                    },
                    "write_paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Exact writable files/directories for read_write run. Required and non-empty for read_write; working_dir never implies write authority."
                    },
                    "workspace_mode": {
                        "type": "string",
                        "enum": ["read_only", "read_write"],
                        "description": "Run authority: enforced read_only or mutation-capable read_write."
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID"
                    },
                    "system_instruction": {
                        "type": "string",
                        "description": "Optional system instruction shaping the agent into a specialist (e.g. 'You are a security auditor')"
                    },
                    "async_mode": {
                        "type": "boolean",
                        "description": "Run in background; use true to dispatch independent sub-tasks in parallel"
                    },
                    "checklist_items": {
                        "type": "array",
                        "items": { "type": "integer", "minimum": 0 },
                        "uniqueItems": true,
                        "description": "The indices of your checklist items this run carries out, so a change it makes settles them."
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::tools::traits::ToolCallContext,
    ) -> anyhow::Result<crate::tools::traits::ToolCallOutcome> {
        let status_tx = aidaemon_tools::status::status_sender(facts);
        let exec_ctx = facts.clone();

        self.outcome_with_facts(arguments, status_tx, true, &exec_ctx)
            .await
    }
}

/// An earlier run a follow-up goes on from ([`CliAgentArgs::continue_run`]).
struct EarlierRun {
    /// Its agent's configured name.
    agent: String,
    working_dir: Option<String>,
    continues: crate::helper_run::Continues,
    /// For a run that ended before it finished (a restart stopped it, it
    /// failed part way): what it had done, by the events kept as it ran,
    /// for the follow-up's agent ([`super::RunProgress::note`]).
    progress: Option<String>,
}

impl CliAgentTool {
    /// Run `run` of conversation `session_id`, when a follow-up can go on
    /// from it: it ended, and its agent kept its session. `Err`: why not,
    /// in plain words.
    async fn earlier_run(&self, run: &str, session_id: &str) -> Result<EarlierRun, String> {
        let found = self
            .state
            .cli_agent_run(run)
            .await
            .map_err(|error| format!("run `{run}` could not be read: {error}"))?
            .filter(|found| found.session_id == session_id)
            .ok_or_else(|| format!("this conversation has no coding agent run `{run}`"))?;
        if self
            .run_handles()
            .runs
            .get(run)
            .is_some_and(|state| state.ended_at.is_none())
        {
            return Err(format!(
                "run `{run}` is still working: wait for it to end, or stop it, before following up"
            ));
        }
        let Some(agent_session) = found.agent_session.filter(|_| found.keeps_session) else {
            return Err(format!(
                "{} didn't keep run `{run}`'s session, so nothing can go on from it; start a new \
                 run instead",
                found.agent_name
            ));
        };
        let progress = match found.success {
            Some(false) => {
                let events = self
                    .state
                    .get_cli_agent_events(found.invocation)
                    .await
                    .unwrap_or_default();
                Some(
                    super::RunProgress::of(&events)
                        .note(run, super::why_unfinished(found.output_summary.as_deref())),
                )
            }
            _ => None,
        };
        Ok(EarlierRun {
            progress,
            agent: found.agent_name,
            working_dir: found.working_dir,
            continues: crate::helper_run::Continues {
                run: run.to_string(),
                agent_session,
                thread: found.thread.unwrap_or_else(|| run.to_string()),
            },
        })
    }
}

/// Direct calls from tests, without the dispatcher's native-access checks.
#[cfg(test)]
impl CliAgentTool {
    pub(super) async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.call_with_status(arguments, None).await
    }

    pub(super) async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        self.call_with_status_under_contract(
            arguments,
            status_tx,
            false,
            false,
            &ToolCallContext::default(),
        )
        .await
    }

    pub(super) async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.outcome_with_facts(arguments, status_tx, false, &ToolCallContext::default())
            .await
    }
}

#[async_trait]
impl crate::tools::traits::ToolSemantics for CliAgentTool {
    async fn background_jobs(
        &self,
        session_id: &str,
    ) -> Vec<aidaemon_tools::semantics::BackgroundJob> {
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return Vec::new();
        }
        let now = chrono::Utc::now();
        let mut jobs: Vec<aidaemon_tools::semantics::BackgroundJob> = self
            .running
            .lock()
            .await
            .iter()
            .filter(|(_, agent)| {
                agent.session_id == session_id && !agent.finished.load(Ordering::Acquire)
            })
            .map(
                |(task_id, agent)| aidaemon_tools::semantics::BackgroundJob {
                    key: task_id.clone(),
                    title: agent.label.clone(),
                    started_at: now
                        - chrono::Duration::from_std(agent.accepted_at.elapsed())
                            .unwrap_or_default(),
                    task_id: agent.delegated_task_id.clone(),
                    run: Some(cli_run_id(task_id)),
                },
            )
            .collect();
        jobs.sort_by_key(|job| job.started_at);
        jobs
    }

    async fn stop_background_job(&self, key: &str) -> bool {
        let (finished, done_notify) = match self.running.lock().await.get(key) {
            Some(agent) => (agent.finished.clone(), agent.done_notify.clone()),
            None => return true,
        };
        if let Err(error) = self.handle_cancel(key).await {
            warn!(task_id = key, %error, "Could not stop a background CLI agent for the owner");
            return false;
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let notified = done_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if finished.load(Ordering::Acquire) {
                return true;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return finished.load(Ordering::Acquire);
            }
        }
    }

    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            task_lead_delegation: true,
            manages_own_timeout: true,
            goal_context: crate::tools::traits::GoalContext::BackgroundDelivery,
            checkpoints_writes: true,
            // A CLI agent's run is a helper run like a sub-agent's: it
            // inherits the turn's authority, narrowed by its declaration
            // ([`crate::helper_run::helper_authority`]).
            spawns_child_agents: true,
            activity: aidaemon_tools::semantics::ToolActivity::Fixed(
                aidaemon_core::card::ActivityKind::Delegate,
            ),
            project_dir_argument: aidaemon_tools::semantics::ProjectDirArgument::WorkingDir,
            status_label: Some("delegating to a CLI agent"),
            result_fit: aidaemon_tools::semantics::ResultFit {
                cap_large_errors: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn validate_arguments(&self, arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        let Ok(parsed) = serde_json::from_str::<Value>(arguments) else {
            return Ok(());
        };
        if parsed.get("action").and_then(Value::as_str) != Some("run") {
            return Ok(());
        }
        let workspace_mode = parsed.get("workspace_mode").and_then(Value::as_str);
        let explicit_read_write = workspace_mode == Some("read_write");
        let has_working_dir = parsed
            .get("working_dir")
            .and_then(Value::as_str)
            .is_some_and(|dir| !dir.trim().is_empty());
        // `CliWorkspaceMode` defaults to `ReadWrite`, so omitting
        // `workspace_mode` when targeting a `working_dir` is a read-write run
        // and must also declare explicit write targets.
        let requires_write_manifest =
            explicit_read_write || (workspace_mode.is_none() && has_working_dir);
        let non_empty_list = |key: &str| {
            parsed
                .get(key)
                .and_then(Value::as_array)
                .is_some_and(|items| {
                    items
                        .iter()
                        .any(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
                })
        };
        let has_declared_writes = non_empty_list("write_paths")
            || non_empty_list("scope")
            || non_empty_list("new_folders");
        if requires_write_manifest && !has_declared_writes {
            return Err(ToolArgumentContractViolation::new(
                "read-write CLI delegation requires an explicit non-empty write_paths access manifest",
            )
            .with_recovery_hint(
                "Retry with readable context separate from exact mutation paths.",
            ));
        }
        Ok(())
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

    /// A delegated agent may do anything its own tools allow.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects::unknown()
    }

    fn receipt_kind(&self, arguments: &str) -> crate::tools::traits::ToolReceiptKind {
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| {
                value
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "run".to_string());
        if action == "run" {
            crate::tools::traits::ToolReceiptKind::Process
        } else {
            crate::tools::traits::ToolReceiptKind::Generic
        }
    }

    fn is_available(&self) -> bool {
        self.has_tools()
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let action = parsed
            .as_ref()
            .and_then(|value| value.get("action"))
            .and_then(Value::as_str)
            .unwrap_or("run");
        let mut semantics = match action {
            "list" | "check" => ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
            "run"
                if parsed
                    .as_ref()
                    .and_then(|value| value.get("workspace_mode"))
                    .and_then(Value::as_str)
                    == Some("read_only") =>
            {
                ToolCallSemantics::observation()
                    .with_verification_mode(ToolVerificationMode::ResultContent)
            }
            // A read-write delegated run may both inspect and change state.
            // Keep the effect conservative for preflight policy, but omit a
            // synthetic mutation-effect bit so a structured pre-launch
            // failure can authoritatively downgrade to administrative.
            "run" => ToolCallSemantics::observation_and_mutation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
            "cancel" | "cancel_all" => ToolCallSemantics::mutation(),
            _ => ToolCallSemantics::mutation(),
        };
        if let Some(working_dir) = parsed
            .as_ref()
            .and_then(|value| value.get("working_dir"))
            .and_then(Value::as_str)
        {
            semantics = semantics.with_target_hint(ToolTargetHintKind::ProjectScope, working_dir);
        }
        semantics
    }

    fn call_access_manifest(&self, arguments: &str) -> ToolCallAccessManifest {
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let execution_cwd = parsed
            .as_ref()
            .and_then(|value| value.get("working_dir"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let workspace_mode = parsed
            .as_ref()
            .and_then(|value| value.get("workspace_mode"))
            .and_then(Value::as_str)
            .unwrap_or("read_write");
        let mut read_targets = Vec::new();
        if let Some(cwd) = execution_cwd.as_deref()
            && let Some(target) = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, cwd)
        {
            read_targets.push(target);
        }
        for path in parsed
            .as_ref()
            .and_then(|value| value.get("read_paths"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let path = execution_cwd
                .as_deref()
                .filter(|_| !aidaemon_tools::BackendPath::is_anchored(path))
                .map(|cwd| aidaemon_tools::BackendPath::new(cwd).join(path).to_string())
                .unwrap_or_else(|| path.to_string());
            if let Some(target) = ToolTargetHint::new(ToolTargetHintKind::Path, path)
                && !read_targets.contains(&target)
            {
                read_targets.push(target);
            }
        }
        let mut write_targets = parsed
            .as_ref()
            .and_then(|value| value.get("write_paths"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|path| {
                execution_cwd
                    .as_deref()
                    .filter(|_| !aidaemon_tools::BackendPath::is_anchored(path))
                    .map(|cwd| aidaemon_tools::BackendPath::new(cwd).join(path).to_string())
                    .unwrap_or_else(|| path.to_string())
            })
            .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::Path, path))
            .collect::<Vec<_>>();
        if workspace_mode == "read_only" {
            write_targets.clear();
        }
        ToolCallAccessManifest {
            execution_cwd,
            read_targets,
            write_targets,
            adapter_read_targets: Vec::new(),
        }
    }
}

#[cfg(test)]
pub(super) fn cli_agent_outcome(
    action: &str,
    output: String,
    read_only_run: bool,
    left_in_background: bool,
    never_started: bool,
) -> ToolCallOutcome {
    cli_agent_outcome_with_refusal(
        action,
        output,
        read_only_run,
        left_in_background,
        never_started,
        None,
    )
}

pub(super) fn cli_agent_outcome_with_refusal(
    action: &str,
    output: String,
    read_only_run: bool,
    left_in_background: bool,
    never_started: bool,
    refusal: Option<PrelaunchRefusal>,
) -> ToolCallOutcome {
    let exit_code = cli_agent_exit_code(&output);
    let prelaunch_failure = never_started;
    let trimmed = output.trim_start();
    let contention_refusal = refusal == Some(PrelaunchRefusal::Contention)
        || (action == "run"
            && (trimmed.starts_with("BLOCKED:") || trimmed.starts_with("Maximum ")));
    let policy_block = refusal == Some(PrelaunchRefusal::Policy) || trimmed.starts_with("Blocked:");
    let reported_failure = trimmed.starts_with("ERROR: CLI agent");
    let metadata = if contention_refusal {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::FailedRetryable),
            semantics: ToolCallSemantics::administrative(),
            ..ToolCallMetadata::default()
        }
    } else if exit_code.is_some() || prelaunch_failure || policy_block || reported_failure {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::FailedPermanent),
            exit_code,
            transport_error: prelaunch_failure.then(|| first_cli_failure_detail(&output)),
            semantics: if prelaunch_failure || policy_block || reported_failure {
                ToolCallSemantics::administrative()
            } else {
                ToolCallSemantics::default()
            },
            ..ToolCallMetadata::default()
        }
    } else if left_in_background {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::Backgrounded),
            background_started: true,
            completion_notifications_enabled: action == "run",
            ..ToolCallMetadata::default()
        }
    } else if action == "check"
        && (output.starts_with("No CLI agent task matched")
            || output.starts_with("No running CLI agent"))
    {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::CompletedWithNegativeResult),
            ..ToolCallMetadata::default()
        }
    } else {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::Succeeded),
            semantics: if read_only_run {
                ToolCallSemantics::observation()
                    .with_verification_mode(ToolVerificationMode::ResultContent)
            } else {
                ToolCallSemantics::default()
            },
            ..ToolCallMetadata::default()
        }
    };
    // A run the agent moved to background tracking hands the turn off to it.
    let handed_off = metadata.background_started;
    let outcome = ToolCallOutcome {
        output,
        metadata,
        finish: None,
    };
    if handed_off {
        outcome.handed_to_background()
    } else {
        outcome
    }
}

impl CliAgentTool {
    /// One call's typed outcome with the dispatcher's `facts`. Only the
    /// dispatcher's path enforces the native access contract.
    pub(super) async fn outcome_with_facts(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        enforce_native_access: bool,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let parsed = serde_json::from_str::<CliAgentArgs>(arguments).ok();
        let action = parsed
            .as_ref()
            .and_then(|args| args.action.clone())
            .unwrap_or_else(|| "run".to_string());
        let read_only_run = action == "run"
            && (facts.mutation_forbidden
                || parsed
                    .as_ref()
                    .is_some_and(|args| args.workspace_mode == CliWorkspaceMode::ReadOnly));
        let (output, left_in_background, settled, changed, run_label, never_started, refusal) =
            REFUSED_BEFORE_LAUNCH
                .scope(
                    std::cell::Cell::new(None),
                    NEVER_STARTED.scope(
                        std::cell::Cell::new(false),
                        BACKGROUND_RUN_LABEL.scope(
                            std::cell::RefCell::new(None),
                            RUN_CHANGED_LOCALLY.scope(
                                std::cell::Cell::new(false),
                                BACKGROUND_SETTLED.scope(
                                    std::cell::Cell::new(false),
                                    LEFT_IN_BACKGROUND.scope(std::cell::Cell::new(false), async {
                                        let output = self
                                            .call_with_status_under_contract(
                                                arguments,
                                                status_tx,
                                                read_only_run,
                                                enforce_native_access,
                                                facts,
                                            )
                                            .await?;
                                        anyhow::Ok((
                                            output,
                                            LEFT_IN_BACKGROUND.with(std::cell::Cell::get),
                                            BACKGROUND_SETTLED.with(std::cell::Cell::get),
                                            RUN_CHANGED_LOCALLY.with(std::cell::Cell::get),
                                            BACKGROUND_RUN_LABEL
                                                .with(|slot| slot.borrow_mut().take()),
                                            NEVER_STARTED.with(std::cell::Cell::get),
                                            REFUSED_BEFORE_LAUNCH.with(std::cell::Cell::get),
                                        ))
                                    }),
                                ),
                            ),
                        ),
                    ),
                )
                .await?;
        let mut outcome = cli_agent_outcome_with_refusal(
            &action,
            output,
            read_only_run,
            left_in_background,
            never_started,
            refusal,
        );
        if changed && outcome.metadata.outcome_status == Some(ToolOutcomeStatus::Succeeded) {
            outcome.metadata.semantics = changed_locally_semantics(true)
                .with_verification_mode(ToolVerificationMode::ResultContent);
            // The change receipt says the run's write roots changed, not
            // which files or what kind of write: a delegated agent's edits
            // are untyped. The items the run says it carries out are settled
            // by it (`UNSPECIFIED`); without a claim, the one change item in
            // progress may be (live 2026-11-20: an agy run that built the
            // site was read as not proving its "write the source" item, and
            // the turn spent four more model calls ticking it).
            let claimed = parsed
                .as_ref()
                .map(|args| args.checklist_items.clone())
                .unwrap_or_default();
            if claimed.is_empty() {
                outcome.metadata.changes_untyped = true;
            } else {
                outcome.metadata.semantics = outcome.metadata.semantics.with_untyped_change();
                if outcome.metadata.checklist_items.is_empty() {
                    outcome.metadata.checklist_items = claimed;
                }
            }
        }
        outcome.metadata.background_settled = settled && !left_in_background;
        if outcome.metadata.background_started {
            outcome.metadata.background_run_labels = run_label.into_iter().collect();
        }
        Ok(outcome)
    }

    pub(super) async fn call_with_status_under_contract(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        read_only_run: bool,
        enforce_native_access: bool,
        // The dispatcher's typed facts; never the arguments.
        facts: &ToolCallContext,
    ) -> anyhow::Result<String> {
        let mut args: CliAgentArgs = serde_json::from_str(arguments)?;
        args.bind_facts(facts);
        let trusted_session = facts.trusted_session;

        // Reap any finished tasks
        self.reap_finished().await;

        let action = args.action.as_deref().unwrap_or("run");

        // Default to the caller session (often an internal child-agent session).
        // If this CLI agent is running in the context of an orchestration goal,
        // route background notifications and cancel_all scoping to the *origin*
        // session_id (goal.session_id), which is routable via the ChannelHub.
        let mut session_id = args.call_session.clone().unwrap_or_default();
        if let Some(ref goal_id) = args.call_goal_id
            && let Ok(Some(goal)) = self.state.get_goal(goal_id).await
            && !goal.session_id.trim().is_empty()
        {
            session_id = goal.session_id;
        }

        match action {
            "run" => {
                // In a conversation set to this agent, each message goes on
                // from the agent's latest run here that kept its session,
                // so the agent has the conversation so far.
                let direct_agent_conversation = args.tool.as_deref().is_some()
                    && crate::session_model::get(self.state.as_ref(), &session_id)
                        .await
                        .agent
                        .as_deref()
                        == args.tool.as_deref();
                if args.continue_run.is_none()
                    && direct_agent_conversation
                    && let Some(agent) = args.tool.clone()
                {
                    match self
                        .state
                        .latest_kept_cli_agent_run(&session_id, &agent)
                        .await
                    {
                        Ok(Some(run)) => match self.earlier_run(&run, &session_id).await {
                            Ok(_) => {
                                info!(%session_id, %agent, %run, "Coding agent conversation: going on from the agent's last session");
                                args.continue_run = Some(run);
                            }
                            Err(why) => {
                                info!(%session_id, %agent, %run, %why, "Coding agent conversation: a new session (the last one can't be continued)");
                            }
                        },
                        Ok(None) => {
                            info!(%session_id, %agent, "Coding agent conversation: a new session (no earlier run here kept its session)");
                        }
                        Err(error) => {
                            warn!(%session_id, %agent, %error, "Coding agent conversation: could not look up the agent's last session");
                        }
                    }
                }
                // A follow-up to an earlier run: the same agent, in its
                // folder, going on in its own session.
                let mut unfinished = None;
                let continues = match args
                    .continue_run
                    .as_deref()
                    .map(str::trim)
                    .filter(|run| !run.is_empty())
                {
                    None => None,
                    Some(run) => match self.earlier_run(run, &session_id).await {
                        Ok(earlier) => {
                            args.tool = Some(earlier.agent);
                            if args.working_dir.is_none() {
                                args.working_dir = earlier.working_dir;
                            }
                            unfinished = earlier.progress;
                            Some(earlier.continues)
                        }
                        Err(why) => {
                            mark_refused_before_launch(PrelaunchRefusal::Policy);
                            return Ok(format!("Blocked: {why}"));
                        }
                    },
                };
                // An agent asked for by name runs when its owner-set mode
                // lets it run this run here (`agent_mode`).
                let can_host = self.can_host(facts);
                let by_mode = match args
                    .tool
                    .as_deref()
                    .and_then(|name| self.configured_tool(name))
                {
                    Some(name)
                        if enforce_native_access
                            && self.runnable(&name, read_only_run, can_host).await =>
                    {
                        Some(name)
                    }
                    _ => None,
                };
                let mut tool = if by_mode.is_some() {
                    by_mode
                } else if !enforce_native_access {
                    args.tool
                        .as_deref()
                        .and_then(|name| self.configured_tool(name))
                        .or_else(|| self.default_tool_name())
                } else if read_only_run {
                    args.tool
                        .as_deref()
                        .and_then(|name| self.configured_tool(name))
                        .or_else(|| self.default_tool_name())
                } else {
                    self.scoped_read_write_tool_name(args.tool.as_deref())
                }
                .ok_or_else(|| {
                    if enforce_native_access {
                        anyhow::anyhow!(
                            "No configured CLI agent has a native adapter for this access contract"
                        )
                    } else {
                        anyhow::anyhow!("No CLI agents available")
                    }
                })?;
                // Only the agent that kept the session can go on in it.
                if continues.is_some()
                    && args
                        .tool
                        .as_deref()
                        .and_then(|name| self.configured_tool(name))
                        != Some(tool.clone())
                {
                    mark_refused_before_launch(PrelaunchRefusal::Policy);
                    return Ok(format!(
                        "Blocked: {} can't take this follow-up here (its mode doesn't let it \
                         run this run), and no other agent can go on in its session.",
                        args.tool.as_deref().unwrap_or("its agent")
                    ));
                }
                let prompt = if let Some(prompt) = args.run_prompt() {
                    prompt
                } else if let Some(prompt) = args.contextual_run_prompt(&self.state).await {
                    warn!(
                        task_id = ?args.call_task_id,
                        goal_id = ?args.call_goal_id,
                        working_dir = ?args.working_dir,
                        "cli_agent run omitted prompt; synthesized delegated prompt from stored task/goal context"
                    );
                    prompt
                } else {
                    warn!(
                        task_id = ?args.call_task_id,
                        goal_id = ?args.call_goal_id,
                        has_prompt = args.prompt.as_ref().is_some_and(|s| !s.trim().is_empty()),
                        has_mission = args.mission.as_ref().is_some_and(|s| !s.trim().is_empty()),
                        has_task = args.task.as_ref().is_some_and(|s| !s.trim().is_empty()),
                        has_description = args
                            .description
                            .as_ref()
                            .is_some_and(|s| !s.trim().is_empty()),
                        has_command = args.command.as_ref().is_some_and(|s| !s.trim().is_empty()),
                        "cli_agent run missing prompt and had no recoverable delegated context"
                    );
                    anyhow::bail!("Missing 'prompt' parameter for action=run");
                };

                // Trusted sessions bypass the interactive approval boundary.
                // A CLI delegate is an opaque, mutation-capable executor, so
                // prompt wording cannot narrow its effects. Require a concrete
                // clean repository for every unattended run rather than trying
                // to recognize deployment verbs or negations in prose.
                if trusted_session {
                    let Some(working_dir) = args.working_dir.as_deref() else {
                        mark_refused_before_launch(PrelaunchRefusal::Policy);
                        return Ok(
                            "Blocked: unattended CLI delegation requires an explicit working_dir so repository cleanliness can be verified."
                                .to_string(),
                        );
                    };
                    if let Some(entry_count) =
                        Self::dirty_worktree_entry_count(&self.backend, working_dir)
                            .await
                            .filter(|count| *count > 0)
                    {
                        mark_refused_before_launch(PrelaunchRefusal::Policy);
                        return Ok(format!(
                            "Blocked: unattended CLI delegation found {entry_count} pre-existing changed or untracked worktree entries. Use a clean isolated worktree containing only the intended change, or request explicit owner review; unrelated workspace state was not exposed to opaque delegated mutation."
                        ));
                    }
                }

                // Goal-scoped cli_agent runs should not detach immediately:
                // the goal/task lead is already running in the background, and
                // returning early makes results easy to drop.
                let async_mode = if args.call_goal_id.is_some() {
                    false
                } else {
                    args.async_mode.unwrap_or(false)
                };

                // The run is a helper run: declared, named and given its
                // authority exactly as a sub-agent is.
                let made_folders = if args.new_folders.is_empty() {
                    Vec::new()
                } else {
                    match crate::helper_run::folders::claim_request_folders(
                        &args.new_folders,
                        facts,
                    ) {
                        Ok(made) => made,
                        Err(error) => {
                            mark_refused_before_launch(PrelaunchRefusal::Policy);
                            return Ok(format!("Blocked: {error}"));
                        }
                    }
                };
                let spec = crate::helper_run::HelperSpec {
                    title: args
                        .title
                        .clone()
                        .or_else(|| args.description.clone())
                        .filter(|title| !title.trim().is_empty()),
                    mission: String::new(),
                    new_folders: made_folders.clone(),
                    scope: args.scope.clone(),
                    owner_follow_up: continues.is_some()
                        && facts
                            .tool_call_id
                            .as_deref()
                            .is_some_and(crate::helper_run::is_owner_follow_up),
                    continues,
                };
                let write_paths = helper_write_paths(&args, &spec, &session_id, facts);
                let working_dir = args
                    .working_dir
                    .clone()
                    .or_else(|| made_folders.first().cloned());
                let grant_session = if facts.grant_session.trim().is_empty() {
                    session_id.clone()
                } else {
                    facts.grant_session.clone()
                };

                let prompt = if direct_agent_conversation && spec.continues.is_none() {
                    self.prepend_direct_conversation_context(&session_id, &prompt)
                        .await
                } else {
                    prompt
                };
                // Going on from a run that never finished: the agent is told
                // where it stopped, from the daemon's record, whatever its
                // own session kept.
                let prompt = match unfinished {
                    Some(note) => format!("{note}\n\n{prompt}"),
                    None => prompt,
                };
                let detached_judge = facts
                    .extensions
                    .get::<Arc<crate::approvals::DetachedNestedJudge>>()
                    .cloned();
                let spawn_family = crate::authority::parent_authority(facts)
                    .and_then(|parent| parent.spawn_family.clone());
                let mut unavailable = Vec::new();
                let output = loop {
                    match self
                        .handle_run(
                            &tool,
                            &prompt,
                            working_dir.as_deref(),
                            &session_id,
                            args.call_goal_id.as_deref(),
                            args.call_task_id.as_deref(),
                            args.system_instruction.as_deref(),
                            if read_only_run {
                                CliWorkspaceMode::ReadOnly
                            } else {
                                CliWorkspaceMode::ReadWrite
                            },
                            &args.read_paths,
                            &write_paths,
                            enforce_native_access,
                            async_mode,
                            status_tx.clone(),
                            &spec,
                            &grant_session,
                            facts.project.as_ref(),
                            facts
                                .extensions
                                .get::<aidaemon_core::NestedCalls>()
                                .cloned(),
                            detached_judge.clone(),
                            spawn_family.clone(),
                        )
                        .await
                    {
                        Ok(output) => break Ok(output),
                        Err(error) => {
                            let Some(failure) = error.downcast_ref::<CliAgentUnavailableError>()
                            else {
                                break Err(error);
                            };
                            unavailable.push((failure.tool_name.clone(), failure.message.clone()));

                            // A follow-up goes on in its own agent's
                            // session; another agent can't take it.
                            let Some(next_tool) = spec
                                .continues
                                .is_none()
                                .then(|| {
                                    self.fallback_tool_name(enforce_native_access, read_only_run)
                                })
                                .flatten()
                            else {
                                let mut message = String::from(
                                    "ERROR: CLI agent recovery exhausted before task execution. \
                                     No configured CLI agent was able to start the delegated task.",
                                );
                                for (name, detail) in &unavailable {
                                    message.push_str(&format!("\n- {name}: {detail}"));
                                }
                                if let Some(task_id) = args.call_task_id.as_deref() {
                                    self.persist_delegated_cli_result(
                                        task_id,
                                        None,
                                        Some(&message),
                                    )
                                    .await;
                                }
                                mark_never_started();
                                break Ok(message);
                            };

                            info!(
                                unavailable_tool = %tool,
                                fallback_tool = %next_tool,
                                "Retrying delegated task with another configured CLI agent"
                            );
                            tool = next_tool;
                        }
                    }
                };
                // Folders made for a run that never started are given back
                // (only while still empty), as for a sub-agent.
                let started = matches!(&output, Ok(text) if !run_did_not_start(text));
                if !started {
                    crate::helper_run::folders::release_request_folders(&made_folders, facts);
                }
                // Which agent did the work is the daemon's fact to state: a
                // run that went to another agent than the one asked for says
                // so first (live 2026-11-18: codex ran a run asked of claude
                // and the reply said "Claude Code reported").
                match (output, args.tool.as_deref()) {
                    (Ok(text), Some(asked)) if started && !asked.eq_ignore_ascii_case(&tool) => {
                        Ok(format!("{}\n{text}", substituted_agent_note(asked, &tool)))
                    }
                    (output, _) => output,
                }
            }
            "check" => {
                let task_id = args.task_id.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Missing 'task_id' parameter for action=check")
                })?;
                self.handle_check(task_id, &session_id).await
            }
            "cancel" => {
                let task_id = args.task_id.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Missing 'task_id' parameter for action=cancel")
                })?;
                self.handle_cancel(task_id).await
            }
            "cancel_all" => self.handle_cancel_all(&session_id).await,
            "list" => self.handle_list().await,
            _ => Ok(format!(
                "Unknown action '{}'. Use run, check, cancel, cancel_all, or list.",
                action
            )),
        }
    }
}
