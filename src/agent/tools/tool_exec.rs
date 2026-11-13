use super::*;
use crate::types::WorkspaceGrant;

fn argument_contract_rejection_outcome(
    violation: crate::traits::ToolArgumentContractViolation,
) -> crate::traits::ToolCallOutcome {
    let output = match violation.recovery_hint {
        Some(hint) => format!(
            "Invocation rejected before I/O: {}\nRecovery: {hint}",
            violation.reason
        ),
        None => format!("Invocation rejected before I/O: {}", violation.reason),
    };
    crate::traits::ToolCallOutcome::contract_rejection(output)
}

fn prepare_tool_arguments(
    tool: &dyn crate::traits::Tool,
    arguments: &str,
) -> Result<crate::traits::PreparedToolInvocation, Box<crate::traits::ToolCallOutcome>> {
    match crate::traits::prepare_invocation(tool, arguments) {
        Ok(prepared) => Ok(prepared),
        Err(violation) => Err(Box::new(argument_contract_rejection_outcome(violation))),
    }
}

fn finalize_dispatched_outcome(
    prepared: &crate::traits::PreparedToolInvocation,
    mut outcome: crate::traits::ToolCallOutcome,
) -> crate::traits::ToolCallOutcome {
    if outcome.metadata.invocation_stage == crate::traits::ToolInvocationStage::Unknown {
        outcome.metadata.invocation_stage = if outcome.metadata.contract_rejected {
            crate::traits::ToolInvocationStage::RejectedBeforeIo
        } else {
            crate::traits::ToolInvocationStage::Dispatched
        };
    }
    // A pre-I/O rejection owns validation semantics only. It must never
    // inherit the proposed operation's effects, evidence, or targets.
    if outcome.metadata.invocation_stage.reached_dispatch() {
        outcome
            .metadata
            .semantics
            .merge_missing_from(prepared.semantics.clone());
        if outcome.metadata.access_manifest.is_none() {
            outcome.metadata.access_manifest = Some(prepared.access_manifest.clone());
        }
        outcome.metadata.receipt_kind = prepared.receipt_kind;
    } else {
        outcome.metadata.access_manifest = None;
    }
    outcome
}

fn sanitize_workspace_tool_text(text: &str, grant: &WorkspaceGrant) -> String {
    let relative = text.replace(&grant.project_root, ".");
    crate::tools::sanitize::redact_secrets(&relative)
}

fn sanitize_workspace_tool_outcome(
    mut outcome: crate::traits::ToolCallOutcome,
    grant: &WorkspaceGrant,
) -> crate::traits::ToolCallOutcome {
    outcome.output = sanitize_workspace_tool_text(&outcome.output, grant);
    if let Some(reply) = outcome
        .finish
        .as_mut()
        .and_then(|finish| finish.reply.as_mut())
    {
        *reply = sanitize_workspace_tool_text(reply, grant);
    }
    if let Some(persistent) = outcome.metadata.persistent_output.as_mut() {
        *persistent = sanitize_workspace_tool_text(persistent, grant);
    }
    if let Some(error) = outcome.metadata.transport_error.as_mut() {
        *error = sanitize_workspace_tool_text(error, grant);
    }
    if let Some(truncation) = outcome.metadata.truncation.as_mut() {
        if let Some(hint) = truncation.remediation_hint.as_mut() {
            *hint = sanitize_workspace_tool_text(hint, grant);
        }
    }
    if let Some(read_file) = outcome.metadata.read_file.as_mut() {
        read_file.display_path = sanitize_workspace_tool_text(&read_file.display_path, grant);
        for line in &mut read_file.selected_lines {
            *line = sanitize_workspace_tool_text(line, grant);
        }
    }
    outcome
}

async fn call_scoped_builtin_file_tool(
    backend: crate::execution::SharedExecutionBackend,
    name: &str,
    arguments: &str,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    exec_ctx: crate::traits::ToolCallContext,
    paths: &crate::tools::DaemonPaths,
) -> anyhow::Result<crate::traits::ToolCallOutcome> {
    // Instantiate the daemon's exact built-in implementation. Looking up by
    // name in the general registry would let an injected/custom duplicate tool
    // inherit the collaborator allowlist merely by choosing the same name.
    let tool: Box<dyn crate::traits::Tool> = match name {
        "read_file" => Box::new(crate::tools::ReadFileTool::on_backend(
            backend.clone(),
            paths.clone(),
        )),
        "search_files" => Box::new(crate::tools::SearchFilesTool::on_backend(
            backend.clone(),
            paths.clone(),
        )),
        "write_file" => Box::new(crate::tools::WriteFileTool::on_backend(
            backend.clone(),
            paths.clone(),
        )),
        "edit_file" => Box::new(crate::tools::EditFileTool::on_backend(
            backend.clone(),
            paths.clone(),
        )),
        _ => anyhow::bail!("The workspace grant does not allow this tool."),
    };
    let prepared = match prepare_tool_arguments(tool.as_ref(), arguments) {
        Ok(prepared) => prepared,
        Err(rejection) => return Ok(*rejection),
    };
    let exec_ctx = crate::traits::with_status(exec_ctx, status_tx);
    let mut outcome = tool
        .execute(&prepared.canonical_arguments, &exec_ctx)
        .await?;
    outcome = finalize_dispatched_outcome(&prepared, outcome);
    Ok(outcome)
}

pub(super) use aidaemon_runtime::engine::tool_exec::ToolExecCtx;

async fn scoped_workspace_arguments(
    backend: &dyn crate::execution::ExecutionBackend,
    name: &str,
    arguments: &str,
    grant: &WorkspaceGrant,
    paths: &crate::tools::DaemonPaths,
) -> anyhow::Result<Value> {
    anyhow::ensure!(
        grant.allows_tool(name),
        "Tool access denied: this workspace grant does not allow '{}'.",
        name
    );

    anyhow::ensure!(
        backend.kind() == crate::execution::BackendKind::Local,
        "Tool access denied: collaborator grants require the local execution backend."
    );
    let mut args = serde_json::from_str::<Value>(arguments)?;
    let map = args
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Tool arguments must be a JSON object"))?;

    let raw_path = if name == "search_files" {
        map.get("path").and_then(Value::as_str).unwrap_or(".")
    } else {
        ["path", "file_path", "file", "filename"]
            .iter()
            .find_map(|key| map.get(*key).and_then(Value::as_str))
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?
    };
    anyhow::ensure!(!raw_path.trim().is_empty(), "Path cannot be empty");

    let root_resolved = backend.resolve_path(&grant.project_root).await?;
    let root_canonical = backend
        .canonicalize(&root_resolved)
        .await
        .map_err(|_| anyhow::anyhow!("The delegated project is no longer available"))?;
    anyhow::ensure!(
        root_canonical.as_str() == grant.project_root,
        "The delegated project path changed after authorization; ask the owner to grant it again."
    );
    let root_path = std::path::PathBuf::from(root_canonical.as_str());
    anyhow::ensure!(
        root_path.is_absolute() && root_path.parent().is_some(),
        "The workspace grant has an invalid project root."
    );
    if let Ok(home) = backend.canonicalize(backend.home_hint()).await {
        anyhow::ensure!(
            root_canonical != home,
            "The home directory cannot be used as a delegated project."
        );
    }
    anyhow::ensure!(
        !paths.is_sensitive(&root_path)
            && crate::tools::fs_utils::find_nearest_project_root(&root_path)
                .is_some_and(|project| project == root_path),
        "The workspace grant no longer identifies a safe project root."
    );

    // Relative paths are intentionally rooted at the delegated project, not at
    // the daemon's broader execution workspace. This makes safe relative calls
    // usable without exposing the host's absolute path in conversation.
    let requested = if raw_path == "." {
        root_path.clone()
    } else if crate::execution::BackendPath::is_anchored(raw_path) {
        std::path::PathBuf::from(raw_path)
    } else {
        root_path.join(raw_path)
    };
    anyhow::ensure!(
        !paths.is_sensitive(&requested),
        "Access denied: sensitive files are excluded from workspace grants."
    );

    let requested_string = requested.to_string_lossy().into_owned();
    let resolved = backend
        .resolve_path(&requested_string)
        .await
        .map_err(|_| anyhow::anyhow!("Access denied: path is outside the delegated project"))?;

    let target = match backend.metadata(&resolved).await {
        Ok(_) => backend
            .canonicalize(&resolved)
            .await
            .map_err(|_| anyhow::anyhow!("Access denied: path could not be safely resolved"))?,
        Err(_) if name == "write_file" => {
            // Collaborators may create a file only in an existing directory.
            // Refusing implicit directory creation keeps symlink/ancestor checks
            // complete and avoids a path changing meaning between check and write.
            let parent = resolved
                .parent()
                .ok_or_else(|| anyhow::anyhow!("Access denied: invalid destination path"))?;
            let parent_metadata = backend
                .metadata(&parent)
                .await
                .map_err(|_| anyhow::anyhow!("Parent directory must already exist"))?;
            anyhow::ensure!(parent_metadata.is_dir(), "Parent path is not a directory");
            let canonical_parent = backend.canonicalize(&parent).await.map_err(|_| {
                anyhow::anyhow!("Access denied: parent could not be safely resolved")
            })?;
            let filename = resolved
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("Access denied: invalid destination filename"))?;
            canonical_parent.join(filename)
        }
        Err(_) => anyhow::bail!("Requested path does not exist in the delegated project"),
    };

    let target_path = std::path::Path::new(target.as_str());
    anyhow::ensure!(
        target_path.starts_with(&root_path),
        "Access denied: path is outside the delegated project"
    );
    anyhow::ensure!(
        !paths.is_sensitive(target_path),
        "Access denied: sensitive files are excluded from workspace grants."
    );

    // Normalize every accepted alias to the one canonical field consumed by
    // the built-in file tools. Keeping an untrusted secondary alias around
    // would be unsafe if a future tool changed parameter precedence.
    for alias in ["file_path", "file", "filename"] {
        map.remove(alias);
    }
    map.insert("path".to_string(), json!(target.as_str()));
    Ok(args)
}

// impl-Agent justification: tool dispatch with watchdog over tools/state/event_store.
impl Agent {
    pub(super) async fn semantics_for_exact_tool_call(
        &self,
        name: &str,
        arguments: &str,
    ) -> crate::traits::ToolCallSemantics {
        if let Some(tool) = self
            .tools
            .iter()
            .find(|tool| tool.name() == name && tool.is_available())
        {
            return tool.call_semantics(arguments);
        }
        if let Some(registry) = self.mcp_registry.as_ref() {
            if let Some(tool) = registry.find_tool(name).await {
                return tool.call_semantics(arguments);
            }
        }
        crate::traits::ToolCallSemantics::default()
    }

    /// Final complete-mediation check at the last common dispatcher. The
    /// execution loop issues grants, but this boundary reloads mandate/run
    /// state and compares the exact digest immediately before adapter I/O.
    async fn validate_mandate_dispatch(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
        claim_dispatch: bool,
    ) -> anyhow::Result<bool> {
        let Some(goal_id) =
            aidaemon_runtime::engine::work_scope::WorkScope::mandate_goal_id_for_tool_exec(
                self,
                ctx.task_id,
            )
            .await?
        else {
            anyhow::ensure!(
                ctx.mandate_authority.is_none(),
                "Mandate authority is not valid outside its controller goal."
            );
            return Ok(false);
        };
        let mandate = self
            .state
            .get_mandate_for_goal(&goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate authority record disappeared."))?;
        anyhow::ensure!(mandate.is_active(), "Mandate is no longer active.");
        self.require_live_mandate_execution(&goal_id).await?;
        let goal_run = self
            .state
            .get_current_goal_run(&goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate has no active decision cycle."))?;
        anyhow::ensure!(
            goal_run.trigger_type == "mandate" && goal_run.status == "running",
            "Mandate action is outside a mandate run."
        );
        // The MCP tool judged at turn start, when this is one.
        let semantics = match ctx.mcp_tool.filter(|tool| tool.name() == name) {
            Some(tool) if !self.tools.iter().any(|daemon| daemon.name() == name) => {
                tool.call_semantics(arguments)
            }
            _ => self.semantics_for_exact_tool_call(name, arguments).await,
        };
        let class = crate::mandates::classify_mandate_call(name, arguments, &semantics);
        let fence = self.mandate_execution.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Mandate dispatch is missing its immutable execution role fence.")
        })?;
        let is_task_lead = fence.worker_task_id == fence.root_task_id;
        anyhow::ensure!(
            crate::mandates::role_allows_mandate_call(class, name, is_task_lead),
            if is_task_lead && class == crate::mandates::MandateCallClass::GovernedMutation {
                "The mandate task lead cannot perform governed mutations."
            } else {
                "This call is not permitted for this mandate execution role."
            }
        );

        match class {
            crate::mandates::MandateCallClass::ProtocolObservation => {
                anyhow::ensure!(
                    ctx.mandate_authority.is_none(),
                    "Protocol observations cannot carry mutation authority."
                );
                Ok(true)
            }
            crate::mandates::MandateCallClass::RecordDecision
            | crate::mandates::MandateCallClass::RecordMeasurement => {
                anyhow::ensure!(
                    ctx.mandate_authority.is_none(),
                    "Decision and measurement recording cannot carry mutation authority."
                );
                Ok(true)
            }
            crate::mandates::MandateCallClass::Observation => {
                anyhow::ensure!(
                    ctx.mandate_authority.is_none(),
                    "Observation calls cannot carry mutation authority."
                );
                if let Err(reason) = crate::mandates::authority::authorize_mandate_observation(
                    &mandate,
                    name,
                    arguments,
                    &semantics,
                    &chrono::Utc::now(),
                ) {
                    anyhow::bail!(
                        "Mandate observation authority was revoked before dispatch ({}).",
                        reason.as_str()
                    );
                }
                Ok(true)
            }
            crate::mandates::MandateCallClass::ActControl
            | crate::mandates::MandateCallClass::GovernedMutation => {
                let decision = self
                    .state
                    .get_mandate_decision_for_run(&goal_run.id)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("A current ACT decision is required."))?;
                anyhow::ensure!(
                    decision.mandate_id == mandate.id
                        && decision.goal_run_id == goal_run.id
                        && decision.mandate_version == mandate.version
                        && decision.outcome == crate::traits::MandateDecisionOutcome::Act,
                    "The mandate no longer has a current ACT decision."
                );
                if class == crate::mandates::MandateCallClass::ActControl {
                    anyhow::ensure!(
                        ctx.mandate_authority.is_none(),
                        "Control-plane calls cannot carry mutation authority."
                    );
                    return Ok(true);
                }

                anyhow::ensure!(
                    fence.worker_task_id != fence.root_task_id,
                    "The mandate task lead may deliberate and orchestrate, but only a fenced non-root executor may perform mutations."
                );

                let grant = ctx.mandate_authority.ok_or_else(|| {
                    anyhow::anyhow!("This mutation has no action-bound mandate grant.")
                })?;
                anyhow::ensure!(
                    grant.counts_toward_cycle_budget
                        && decision.action_attempts >= 0
                        && decision.action_attempts
                            < i64::from(mandate.authority.max_mutating_actions_per_cycle)
                        && grant.reserved_action_attempt == decision.action_attempts + 1,
                    "The mandate action candidate is invalid."
                );

                let mut expected = match crate::mandates::authority::authorize_mandate_action(
                    &mandate,
                    &decision,
                    name,
                    arguments,
                    &semantics,
                    &chrono::Utc::now(),
                ) {
                    crate::mandates::authority::MandateAuthorityDecision::Allow(expected) => {
                        expected
                    }
                    crate::mandates::authority::MandateAuthorityDecision::Deny(reason) => {
                        anyhow::bail!(
                            "Mandate authority was revoked before dispatch ({}).",
                            reason.as_str()
                        );
                    }
                };
                expected.tool_call_id.clone_from(&grant.tool_call_id);
                anyhow::ensure!(
                    &expected == grant,
                    "Mandate authority was revoked before dispatch (grant_mismatch)."
                );
                if claim_dispatch {
                    let bound_tool_call_id = grant.tool_call_id.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("The mandate grant is not bound to this tool call.")
                    })?;
                    let actual_tool_call_id = ctx.tool_call_id.ok_or_else(|| {
                        anyhow::anyhow!(
                            "The mandate dispatch is missing its actual tool call identity."
                        )
                    })?;
                    anyhow::ensure!(
                        actual_tool_call_id == bound_tool_call_id,
                        "The mandate grant belongs to a different tool call."
                    );
                    let (mutation_effects, targets, account_identifiers) =
                        crate::mandates::authority::mutation_audit_scope(&semantics)
                            .map_err(|reason| anyhow::anyhow!(reason.as_str()))?;
                    let now = chrono::Utc::now().to_rfc3339();
                    let reservation = crate::traits::MandateMutationReservation {
                        grant: grant.clone(),
                        goal_run_id: fence.goal_run_id.clone(),
                        root_task_id: fence.root_task_id.clone(),
                        root_task_attempt_id: fence.root_task_attempt_id.clone(),
                        task_id: fence.worker_task_id.clone(),
                        task_attempt_id: fence.attempt_id.clone(),
                        tool_call_id: actual_tool_call_id.to_string(),
                        tool_name: name.to_string(),
                        mutation_effects,
                        targets,
                        account_identifiers,
                        reserved_at: now.clone(),
                    };
                    let reserved = self
                        .state
                        .reserve_mandate_action_attempt(&reservation)
                        .await?;
                    anyhow::ensure!(
                        reserved.is_some(),
                        "The mandate action budget, policy, or execution fence changed before dispatch."
                    );
                    let mut reserved_decision = decision.clone();
                    reserved_decision.action_attempts = grant.reserved_action_attempt;
                    if let Err(reason) = crate::mandates::authority::validate_mandate_grant(
                        &mandate,
                        &reserved_decision,
                        name,
                        arguments,
                        &semantics,
                        &chrono::Utc::now(),
                        grant,
                    ) {
                        anyhow::bail!(
                            "The exact reserved mandate grant failed final validation ({}).",
                            reason.as_str()
                        );
                    }
                    let claimed = self
                        .state
                        .claim_mandate_mutation_dispatch(
                            &crate::traits::MandateMutationDispatchClaim {
                                grant: grant.clone(),
                                goal_run_id: fence.goal_run_id.clone(),
                                root_task_id: fence.root_task_id.clone(),
                                root_task_attempt_id: fence.root_task_attempt_id.clone(),
                                task_id: fence.worker_task_id.clone(),
                                task_attempt_id: fence.attempt_id.clone(),
                                tool_call_id: actual_tool_call_id.to_string(),
                                tool_name: name.to_string(),
                                claimed_at: chrono::Utc::now().to_rfc3339(),
                            },
                        )
                        .await?;
                    anyhow::ensure!(
                        claimed,
                        "The exact mandate mutation reservation is stale, mismatched, or already dispatched."
                    );
                }
                Ok(true)
            }
            crate::mandates::MandateCallClass::Deny => {
                anyhow::bail!("This call is not supported inside a mandate run.")
            }
        }
    }

    #[cfg(test)]
    pub(super) async fn execute_tool_with_watchdog(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<String> {
        self.execute_tool_with_watchdog_outcome(name, arguments, ctx)
            .await
            .map(|outcome| outcome.output)
    }

    pub(super) async fn execute_tool_with_watchdog_outcome(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let name = name.trim();
        let session_id = ctx.session_id;
        // A tool that manages its own timeout and backgrounding (`cli_agent`)
        // is never wrapped: the watchdog would cancel it early and could
        // orphan the process it started.
        if self.exec_dispatch_facts(name, ctx).manages_own_timeout {
            return self.execute_tool_outcome(name, arguments, ctx).await;
        }
        if let Some(timeout_dur) = self.limits.llm_call_timeout {
            match tokio::time::timeout(timeout_dur, self.execute_tool_outcome(name, arguments, ctx))
                .await
            {
                Ok(result) => result,
                Err(_) => {
                    warn!(
                        session_id,
                        tool = name,
                        timeout_secs = timeout_dur.as_secs(),
                        "Tool call timed out"
                    );
                    Ok(crate::traits::ToolCallOutcome {
                        output: format!(
                            "Tool '{}' timed out after {}s",
                            name,
                            timeout_dur.as_secs()
                        ),
                        metadata: crate::traits::ToolCallMetadata {
                            outcome_status: Some(crate::traits::ToolOutcomeStatus::FailedRetryable),
                            timed_out: true,
                            ..crate::traits::ToolCallMetadata::default()
                        },
                        finish: None,
                    })
                }
            }
        } else {
            self.execute_tool_outcome(name, arguments, ctx).await
        }
    }

    /// The dispatcher's typed facts about one call of `name`, taken from
    /// the turn (`ctx`), this agent, and the store, never from the call's
    /// arguments. `mandate_preapproved` is left false: the caller sets it
    /// after revalidating the exact grant.
    /// The dispatch facts of the tool a call runs: the daemon tool, else
    /// the MCP tool judged at turn start.
    pub(super) fn exec_dispatch_facts(
        &self,
        name: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> crate::traits::ToolDispatchFacts {
        self.tools
            .iter()
            .find(|tool| tool.name() == name)
            .or_else(|| ctx.mcp_tool.filter(|tool| tool.name() == name))
            .map(|tool| tool.dispatch_facts())
            .unwrap_or_default()
    }

    pub(super) async fn tool_call_context(
        &self,
        name: &str,
        ctx: &ToolExecCtx<'_>,
        under_mandate: bool,
        trusted: bool,
        workspace_grant: Option<&WorkspaceGrant>,
    ) -> crate::traits::ToolCallContext {
        let mut facts = crate::traits::ToolCallContext {
            mandate_preapproved: false,
            mandate_execution: under_mandate,
            mutation_forbidden: ctx.mutation_forbidden,
            kernel_approved: ctx
                .owner_approval_by_kernel
                .then_some(crate::traits::ApprovalScope::Call),
            trusted_session: trusted,
            user_role: Some(ctx.user_role),
            untrusted_source: is_trigger_session(ctx.session_id),
            scope_escalation: ctx.scope_escalation.filter(|e| !e.is_empty()).cloned(),
            session_id: ctx.session_id.to_string(),
            // A sub-agent's own grants are its parent chat's.
            grant_session: crate::agent::approvals::grant_session(
                self.approval_session_id.as_deref(),
                ctx.session_id,
            ),
            channel_visibility: Some(ctx.channel_visibility),
            channel_id: ctx.channel_id.map(str::to_string),
            task_id: ctx.task_id.map(str::to_string),
            tool_call_id: ctx.tool_call_id.map(str::to_string),
            lone_root_call: ctx.lone_root_call,
            turn_id: self
                .runtime
                .current_turn_ids
                .read()
                .await
                .get(ctx.session_id)
                .cloned(),
            project_scope: ctx.project_scope.map(str::to_string),
            workspace_collaboration_root: workspace_grant.map(|grant| grant.project_root.clone()),
            ..Default::default()
        };
        if let Some(fence) = ctx.file_fence.clone() {
            facts
                .extensions
                .insert(aidaemon_tools::semantics::CallFileFence(fence.clone()));
            facts.extensions.insert(fence);
        }
        let dispatch = self.exec_dispatch_facts(name, ctx);
        // Only a tool that records it gets the trust its launch was judged
        // with (the terminal).
        if dispatch.records_launch_trust {
            if let Some((roots, cwd)) = ctx.trust_roots {
                facts.trust_roots = Some(roots.to_vec());
                facts.trust_cwd = Some(cwd.to_string());
            }
        }
        // A child inherits this turn's authority, not a re-resolution.
        if dispatch.spawns_child_agents {
            if let Some(turn_authority) = ctx.turn_authority {
                facts.extensions.insert(turn_authority.clone());
            }
        }
        // Goal context for the tools that need it. Background delivery
        // routes async/timeout notifications to the *origin* session
        // (goal.session_id), since internal child-agent sessions are not
        // routable.
        if dispatch.goal_context != crate::traits::GoalContext::None {
            if let Some(ref gid) = self.goal_id {
                facts.goal_id = Some(gid.clone());
            } else if dispatch.goal_context == crate::traits::GoalContext::BackgroundDelivery {
                // Executors typically don't carry goal_id, but do carry
                // task_id: resolve it via the task so background
                // notifications stay deliverable.
                if let Some(ref executor_task_id) = self.task_id {
                    if let Ok(Some(task)) = self.state.get_task(executor_task_id).await {
                        facts.goal_id = Some(task.goal_id);
                    }
                }
            }
            if dispatch.goal_context == crate::traits::GoalContext::MandateRun {
                if let Some(fence) = self.mandate_execution.as_ref() {
                    facts.goal_run_id = Some(fence.goal_run_id.clone());
                    facts.task_attempt_id = Some(fence.attempt_id.clone());
                    facts.mandate_id = Some(fence.mandate_id.clone());
                    facts.mandate_version = Some(fence.mandate_version);
                }
            }
        }
        #[cfg(feature = "computer_use")]
        if dispatch.needs_live_model {
            let runtime = self.llm_runtime.snapshot();
            let current_model =
                match tokio::time::timeout(Duration::from_secs(2), self.model.read()).await {
                    Ok(guard) => guard.clone(),
                    Err(_) => runtime.primary_model(),
                };
            facts.model = Some(current_model);
            facts.provider_kind = Some(format!("{:?}", runtime.provider_kind()));
            facts.model_chain = runtime.router().map(|router| router.all_models_ordered());
        }
        facts
    }

    pub(super) async fn execute_tool_outcome(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let name = name.trim();
        let task_id = ctx.task_id;
        let channel_visibility = ctx.channel_visibility;
        let under_mandate = self
            .validate_mandate_dispatch(name, arguments, ctx, false)
            .await?;
        // A mandate-authorized call is never a trusted session, even if
        // ChannelContext.trusted is true or scheduled provenance would ordinarily
        // authorize it; the entire expression short-circuits to false.
        let trusted = !under_mandate
            && aidaemon_runtime::engine::work_scope::WorkScope::session_trusted(
                self,
                ctx.trusted,
                task_id,
            )
            .await;
        let user_role = ctx.user_role;

        let active_workspace_grant = match user_role {
            UserRole::Owner => None,
            UserRole::Guest => {
                anyhow::ensure!(
                    channel_visibility == ChannelVisibility::PrivateGroup,
                    "Tool access denied: workspace grants apply only in their private group."
                );
                let grant = ctx.workspace_grant.filter(|grant| grant.is_active()).ok_or_else(|| {
                    anyhow::anyhow!("Tool access denied: no active workspace grant for this user and channel.")
                })?;
                Some(grant)
            }
            UserRole::Public => anyhow::bail!("Tool access denied: public users cannot use tools."),
        };

        let call_facts = self
            .tool_call_context(name, ctx, under_mandate, trusted, active_workspace_grant)
            .await;

        let scoped_args = if let Some(grant) = active_workspace_grant {
            Some(
                scoped_workspace_arguments(
                    &*self.execution_backend,
                    name,
                    arguments,
                    grant,
                    &self.daemon_paths,
                )
                .await?,
            )
        } else {
            None
        };

        let parsed_args = match scoped_args {
            Some(args) => Ok(args),
            None => serde_json::from_str::<Value>(arguments),
        };
        let enriched_args = match parsed_args {
            Ok(Value::Object(mut map)) => {
                strip_model_arguments(
                    &mut map,
                    self.tools
                        .iter()
                        .find(|tool| tool.name() == name)
                        .map(|tool| tool.as_ref()),
                );
                // Every fact about the call reaches the tool as a typed
                // `ToolCallContext` field (`call_facts`); the arguments stay
                // the model's own, with no `_` key in them.
                serde_json::to_string(&map)?
            }
            _ => arguments.to_string(),
        };

        let checkpoints_writes = !under_mandate
            && !ctx.mutation_forbidden
            && self.exec_dispatch_facts(name, ctx).checkpoints_writes;
        if let Some(grant) = active_workspace_grant {
            if checkpoints_writes {
                if let Some(manager) = aidaemon_tools::checkpoints::active_manager() {
                    manager
                        .begin_for_tool(name, &enriched_args, &call_facts)
                        .await?;
                }
            }
            let mandate_preapproved = if under_mandate {
                self.validate_mandate_dispatch(name, arguments, ctx, true)
                    .await?;
                true
            } else {
                false
            };
            let mut exec_ctx = call_facts.clone();
            exec_ctx.mandate_preapproved = mandate_preapproved;
            let result = call_scoped_builtin_file_tool(
                self.execution_backend.clone(),
                name,
                &enriched_args,
                ctx.status_tx.clone(),
                exec_ctx,
                &self.daemon_paths,
            )
            .await
            .map(|outcome| sanitize_workspace_tool_outcome(outcome, grant))
            .map_err(|error| {
                anyhow::anyhow!(sanitize_workspace_tool_text(&error.to_string(), grant))
            });

            return result;
        }

        for tool in &self.tools {
            if tool.name() == name {
                let prepared = match prepare_tool_arguments(tool.as_ref(), &enriched_args) {
                    Ok(prepared) => prepared,
                    Err(rejection) => return Ok(*rejection),
                };
                let prepared_args = &prepared.canonical_arguments;
                if checkpoints_writes {
                    if let Some(manager) = aidaemon_tools::checkpoints::active_manager() {
                        manager
                            .begin_for_access_manifest(name, &prepared.access_manifest, &call_facts)
                            .await?;
                    }
                }
                let mandate_preapproved = if under_mandate {
                    self.validate_mandate_dispatch(name, arguments, ctx, true)
                        .await?;
                    true
                } else {
                    false
                };
                let mut exec_ctx = call_facts.clone();
                exec_ctx.mandate_preapproved = mandate_preapproved;
                let exec_ctx = crate::traits::with_status(exec_ctx, ctx.status_tx.clone());
                let result = tool
                    .execute(prepared_args, &exec_ctx)
                    .await
                    .map(|outcome| finalize_dispatched_outcome(&prepared, outcome));
                if result.as_ref().is_ok_and(|outcome| {
                    outcome.metadata.background_started || outcome.metadata.detached
                }) {
                    if let (Some(manager), Some(task_id)) =
                        (aidaemon_tools::checkpoints::active_manager(), task_id)
                    {
                        manager
                            .mark_task_unsafe(
                                task_id,
                                "mutation continued in a background or detached process",
                            )
                            .await;
                    }
                }

                return result;
            }
        }

        // Workspace grants apply only to the daemon's dedicated built-in file
        // tools. Never fall through to an MCP tool that happens to reuse an
        // allowlisted name, because its implementation/provenance is external.
        if active_workspace_grant.is_some() {
            anyhow::bail!("The granted built-in file tool is unavailable.");
        }

        // Search MCP registry for dynamically registered tools.
        // A mandate bit can reach an MCP adapter only after the same exact-grant revalidation used
        // for builtins (MCP tools are non-delegable under the v1 policy).
        let mcp_tool = match (ctx.mcp_tool, &self.mcp_registry) {
            (Some(judged), _) => Some(judged.clone()),
            // The kernel judged no MCP tool under this name for the turn.
            _ if ctx.owner_approval_by_kernel => None,
            (None, Some(registry)) => registry.find_tool(name).await,
            (None, None) => None,
        };
        {
            if let Some(tool) = mcp_tool {
                let prepared = match prepare_tool_arguments(tool.as_ref(), &enriched_args) {
                    Ok(prepared) => prepared,
                    Err(rejection) => return Ok(*rejection),
                };
                let mandate_preapproved = if under_mandate {
                    self.validate_mandate_dispatch(name, arguments, ctx, true)
                        .await?;
                    true
                } else {
                    false
                };
                let mut exec_ctx = call_facts.clone();
                exec_ctx.mandate_preapproved = mandate_preapproved;
                let exec_ctx = crate::traits::with_status(exec_ctx, ctx.status_tx.clone());
                return tool
                    .execute(&prepared.canonical_arguments, &exec_ctx)
                    .await
                    .map(|outcome| finalize_dispatched_outcome(&prepared, outcome));
            }
        }

        let mut available: Vec<String> = self.tools.iter().map(|t| t.name().to_string()).collect();
        if let Some(ref reg) = self.mcp_registry {
            for info in reg.list_servers().await {
                available.extend(info.tool_names);
            }
        }
        Err(UnknownToolError {
            name: name.to_string(),
            available,
        }
        .into())
    }
}

pub(crate) use aidaemon_runtime::engine::tool_exec::UnknownToolError;

pub(crate) use aidaemon_runtime::engine::tool_calls::strip_model_arguments;

#[cfg(test)]
#[path = "tool_watchdog_tests.rs"]
mod tool_watchdog_tests;

#[cfg(test)]
#[path = "tool_call_context_tests.rs"]
mod tool_call_context_tests;

#[cfg(test)]
mod scoped_workspace_tests {
    use super::*;
    use crate::types::WorkspaceAccessLevel;

    async fn scoped_workspace_arguments(
        name: &str,
        arguments: &str,
        grant: &WorkspaceGrant,
        paths: &crate::tools::DaemonPaths,
    ) -> anyhow::Result<Value> {
        super::scoped_workspace_arguments(
            &*crate::execution::active_execution_backend(),
            name,
            arguments,
            grant,
            paths,
        )
        .await
    }

    fn grant(root: &std::path::Path) -> WorkspaceGrant {
        std::fs::write(root.join("package.json"), r#"{"name":"test-project"}"#).unwrap();
        WorkspaceGrant {
            platform: "slack".to_string(),
            workspace_id: "T_TEST".to_string(),
            channel_id: "C_PRIVATE".to_string(),
            user_id: "U_GUEST".to_string(),
            project_root: std::fs::canonicalize(root)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            access: WorkspaceAccessLevel::Edit,
            expires_at: Utc::now() + chrono::Duration::hours(1),
        }
    }

    #[test]
    fn scoped_outcomes_hide_absolute_roots_and_secret_values() {
        let root = tempfile::tempdir().unwrap();
        let grant = grant(root.path());
        let secret = "sk-abcdefghijklmnopqrstuvwxyz123456";
        let outcome = crate::traits::ToolCallOutcome::from_output(format!(
            "Read {}/src/main.rs\nAPI_KEY={secret}",
            grant.project_root
        ));

        let sanitized = sanitize_workspace_tool_outcome(outcome, &grant);
        assert!(sanitized.output.contains("./src/main.rs"));
        assert!(!sanitized.output.contains(&grant.project_root));
        assert!(!sanitized.output.contains(secret));
        assert!(sanitized.output.contains("[REDACTED:API key]"));
    }

    #[tokio::test]
    async fn scoped_arguments_root_relative_paths_and_block_escape() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("index.html"), "hello").unwrap();
        std::fs::write(root.path().join(".env.local"), "TOKEN=nope").unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let grant = grant(root.path());

        let inside = scoped_workspace_arguments(
            "read_file",
            r#"{"path":"index.html"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::canonicalize(inside["path"].as_str().unwrap()).unwrap(),
            std::fs::canonicalize(root.path().join("index.html")).unwrap()
        );

        let traversal = scoped_workspace_arguments(
            "read_file",
            r#"{"path":"../outside.txt"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(traversal.is_err());

        let absolute_outside = scoped_workspace_arguments(
            "read_file",
            &json!({"path": outside.path()}).to_string(),
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(absolute_outside.is_err());

        let sensitive = scoped_workspace_arguments(
            "read_file",
            r#"{"path":".env.local"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(sensitive.is_err());

        let dangerous = scoped_workspace_arguments(
            "terminal",
            r#"{"command":"pwd"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(dangerous.is_err());

        let aliased = scoped_workspace_arguments(
            "read_file",
            r#"{"file_path":"index.html","filename":"../../outside"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await
        .unwrap();
        assert!(aliased.get("file_path").is_none());
        assert!(aliased.get("filename").is_none());
        assert_eq!(
            std::fs::canonicalize(aliased["path"].as_str().unwrap()).unwrap(),
            std::fs::canonicalize(root.path().join("index.html")).unwrap()
        );
    }

    #[tokio::test]
    async fn scoped_write_requires_existing_parent() {
        let root = tempfile::tempdir().unwrap();
        let grant = grant(root.path());

        let allowed = scoped_workspace_arguments(
            "write_file",
            r#"{"path":"new.txt","content":"ok"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await
        .unwrap();
        assert!(std::path::Path::new(allowed["path"].as_str().unwrap())
            .starts_with(std::path::Path::new(&grant.project_root)));

        let missing_parent = scoped_workspace_arguments(
            "write_file",
            r#"{"path":"missing/new.txt","content":"no"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(missing_parent.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn scoped_read_blocks_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        symlink(outside.path(), root.path().join("link.txt")).unwrap();
        let grant = grant(root.path());

        let result = scoped_workspace_arguments(
            "read_file",
            r#"{"path":"link.txt"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(result.is_err());
    }

    /// The dispatcher's per-tool treatment comes from facts the tools
    /// declare, never their names: each fact is pinned to the tools that
    /// hold it.
    #[tokio::test]
    async fn dispatch_treatment_is_declared_by_the_tools() {
        use crate::traits::{GoalContext, Tool};
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let state: std::sync::Arc<dyn crate::traits::StateStore> = harness.state.clone();
        let backend = crate::execution::active_execution_backend;
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(crate::tools::WriteFileTool::new(backend())),
            Box::new(crate::tools::EditFileTool::new(backend())),
            Box::new(crate::tools::RunCommandTool::new(backend())),
            Box::new(crate::tools::spawn::SpawnAgentTool::new_deferred(4000, 60)),
            Box::new(crate::tools::ManagePeopleTool::new(
                state.clone(),
                harness.state.memory_stores(),
            )),
            Box::new(crate::tools::ManageMemoriesTool::new(
                state.clone(),
                harness.state.memory_stores(),
            )),
            Box::new(crate::tools::ManageGoalTasksTool::new(
                "goal-synthetic-1".to_string(),
                state.clone(),
            )),
        ];
        for tool in &tools {
            let facts = tool.dispatch_facts();
            let name = tool.name();
            assert_eq!(
                facts.checkpoints_writes,
                matches!(name, "write_file" | "edit_file" | "run_command"),
                "{name}"
            );
            assert_eq!(
                facts.status_from_arguments,
                matches!(name, "manage_people" | "manage_memories"),
                "{name}"
            );
            let goal = match name {
                "spawn_agent" => GoalContext::Goal,
                "manage_goal_tasks" => GoalContext::MandateRun,
                _ => GoalContext::None,
            };
            assert_eq!(facts.goal_context, goal, "{name}");
            assert_eq!(facts.manages_own_timeout, name == "spawn_agent", "{name}");
            assert!(!facts.records_launch_trust, "{name}");
            assert!(!facts.needs_live_model, "{name}");
        }
    }
}
