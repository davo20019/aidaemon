//! Running one tool call on a host ([`ToolDispatch`], on any [`TurnHost`]):
//! the watchdog, the final mandate check at the last common dispatcher, the
//! typed call context a tool sees, a guest's workspace grant (the scoped
//! built-in file tools) and the lookup of the daemon or MCP tool the call
//! names.

use std::time::Duration;

use aidaemon_channels::context::WorkspaceGrant;
use aidaemon_core::ChannelVisibility;
use aidaemon_tools::semantics::{
    prepare_invocation, ApprovalScope, GoalContext, PreparedToolInvocation,
    ToolArgumentContractViolation, ToolCallContext, ToolCallMetadata, ToolCallOutcome,
    ToolCallSemantics, ToolDispatchFacts, ToolInvocationStage, ToolOutcomeStatus, UserRole,
};
use aidaemon_tools::status::{with_status, StatusUpdate};
use aidaemon_tools::{ExecutionBackend, OutcomeTool};
use aidaemon_work::authority::{
    authorize_mandate_action, authorize_mandate_observation, mutation_audit_scope,
    validate_mandate_grant, MandateAuthorityDecision,
};
use aidaemon_work::mandates::policy::{
    classify_mandate_call, role_allows_mandate_call, MandateCallClass,
};
use aidaemon_work::mandates::{
    MandateDecisionOutcome, MandateMutationDispatchClaim, MandateMutationReservation,
};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::engine::tool_calls::strip_model_arguments;
use crate::engine::tool_exec::{ToolExecCtx, UnknownToolError};
use crate::engine::work_scope::WorkScope;
use crate::host::TurnHost;

fn argument_contract_rejection_outcome(
    violation: ToolArgumentContractViolation,
) -> ToolCallOutcome {
    let output = match violation.recovery_hint {
        Some(hint) => format!(
            "Invocation rejected before I/O: {}\nRecovery: {hint}",
            violation.reason
        ),
        None => format!("Invocation rejected before I/O: {}", violation.reason),
    };
    ToolCallOutcome::contract_rejection(output)
}

fn prepare_tool_arguments(
    tool: &dyn OutcomeTool,
    arguments: &str,
) -> Result<PreparedToolInvocation, Box<ToolCallOutcome>> {
    prepare_invocation(tool, arguments)
        .map_err(|violation| Box::new(argument_contract_rejection_outcome(violation)))
}

fn finalize_dispatched_outcome(
    prepared: &PreparedToolInvocation,
    mut outcome: ToolCallOutcome,
) -> ToolCallOutcome {
    if outcome.metadata.invocation_stage == ToolInvocationStage::Unknown {
        outcome.metadata.invocation_stage = if outcome.metadata.contract_rejected {
            ToolInvocationStage::RejectedBeforeIo
        } else {
            ToolInvocationStage::Dispatched
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
    aidaemon_tools::sanitize::redact_secrets(&relative)
}

/// `outcome` with the grant's absolute project root made relative and
/// secret values redacted, everywhere a guest could read it.
pub fn sanitize_workspace_tool_outcome(
    mut outcome: ToolCallOutcome,
    grant: &WorkspaceGrant,
) -> ToolCallOutcome {
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
    if let Some(truncation) = outcome.metadata.truncation.as_mut()
        && let Some(hint) = truncation.remediation_hint.as_mut() {
            *hint = sanitize_workspace_tool_text(hint, grant);
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
    host: &(impl TurnHost + ?Sized),
    name: &str,
    arguments: &str,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    exec_ctx: ToolCallContext,
) -> anyhow::Result<ToolCallOutcome> {
    // Instantiate the exact built-in implementation. Looking up by name in
    // the general registry would let an injected/custom duplicate tool
    // inherit the collaborator allowlist merely by choosing the same name.
    let backend = host.execution_backend_handle();
    let policy = host.file_policy();
    let tool: Box<dyn OutcomeTool> = match name {
        "read_file" => Box::new(aidaemon_tools::ReadFile::on_backend(backend, policy)),
        "search_files" => Box::new(aidaemon_tools::SearchFiles::on_backend(backend, policy)),
        "write_file" => Box::new(aidaemon_tools::WriteFile::on_backend(backend, policy)),
        "edit_file" => Box::new(aidaemon_tools::EditFile::on_backend(backend, policy)),
        _ => anyhow::bail!("The workspace grant does not allow this tool."),
    };
    let prepared = match prepare_tool_arguments(tool.as_ref(), arguments) {
        Ok(prepared) => prepared,
        Err(rejection) => return Ok(*rejection),
    };
    let exec_ctx = with_status(exec_ctx, status_tx);
    let outcome = tool
        .execute(&prepared.canonical_arguments, &exec_ctx)
        .await?;
    Ok(finalize_dispatched_outcome(&prepared, outcome))
}

/// A guest's call of a granted file tool, its path resolved inside the
/// grant's project: the arguments with the one canonical `path` the built-in
/// file tools read, or why the call is refused.
pub async fn scoped_workspace_arguments(
    backend: &dyn ExecutionBackend,
    name: &str,
    arguments: &str,
    grant: &WorkspaceGrant,
    paths: &aidaemon_tools::path_class::DaemonPaths,
) -> anyhow::Result<Value> {
    anyhow::ensure!(
        grant.allows_tool(name),
        "Tool access denied: this workspace grant does not allow '{}'.",
        name
    );

    anyhow::ensure!(
        backend.kind() == aidaemon_tools::backend::BackendKind::Local,
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
            && aidaemon_tools::fs_utils::find_nearest_project_root(&root_path)
                .is_some_and(|project| project == root_path),
        "The workspace grant no longer identifies a safe project root."
    );

    // Relative paths are rooted at the delegated project, not at the
    // daemon's broader execution workspace, so safe relative calls work
    // without exposing the host's absolute path in conversation.
    let requested = if raw_path == "." {
        root_path.clone()
    } else if aidaemon_tools::backend::BackendPath::is_anchored(raw_path) {
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
            // Refusing implicit directory creation keeps symlink/ancestor
            // checks complete and avoids a path changing meaning between
            // check and write.
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

/// How a host runs the tool calls of its turns.
#[async_trait::async_trait]
pub trait ToolDispatch: TurnHost {
    /// The semantics of an exact call of `name`: the daemon tool's, else the
    /// MCP tool's, else none.
    async fn semantics_for_exact_tool_call(
        &self,
        name: &str,
        arguments: &str,
    ) -> ToolCallSemantics {
        if let Some(tool) = self
            .tools()
            .iter()
            .find(|tool| tool.name() == name && tool.is_available())
        {
            return tool.call_semantics(arguments);
        }
        if let Some(tool) = self.mcp_tool(name).await {
            return tool.call_semantics(arguments);
        }
        ToolCallSemantics::default()
    }

    /// Final complete-mediation check at the last common dispatcher. The
    /// execution loop issues grants, but this boundary reloads mandate/run
    /// state and compares the exact digest immediately before adapter I/O.
    /// `Ok(true)`: the call runs under a mandate.
    async fn validate_mandate_dispatch(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
        claim_dispatch: bool,
    ) -> anyhow::Result<bool> {
        let Some(goal_id) = self.mandate_goal_id_for_tool_exec(ctx.task_id).await? else {
            anyhow::ensure!(
                ctx.mandate_authority.is_none(),
                "Mandate authority is not valid outside its controller goal."
            );
            return Ok(false);
        };
        let mandate = self
            .mandates()
            .get_mandate_for_goal(&goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate authority record disappeared."))?;
        anyhow::ensure!(mandate.is_active(), "Mandate is no longer active.");
        self.require_live_mandate_execution(&goal_id).await?;
        let goal_run = self
            .work()
            .get_current_goal_run(&goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate has no active decision cycle."))?;
        anyhow::ensure!(
            goal_run.trigger_type == "mandate" && goal_run.status == "running",
            "Mandate action is outside a mandate run."
        );
        // The MCP tool judged at turn start, when this is one.
        let semantics = match ctx.mcp_tool.filter(|tool| tool.name() == name) {
            Some(tool) if !self.tools().iter().any(|daemon| daemon.name() == name) => {
                tool.call_semantics(arguments)
            }
            _ => self.semantics_for_exact_tool_call(name, arguments).await,
        };
        let class = classify_mandate_call(name, arguments, &semantics);
        let fence = self.mandate_execution().ok_or_else(|| {
            anyhow::anyhow!("Mandate dispatch is missing its immutable execution role fence.")
        })?;
        let is_task_lead = fence.worker_task_id == fence.root_task_id;
        anyhow::ensure!(
            role_allows_mandate_call(class, name, is_task_lead),
            if is_task_lead && class == MandateCallClass::GovernedMutation {
                "The mandate task lead cannot perform governed mutations."
            } else {
                "This call is not permitted for this mandate execution role."
            }
        );

        match class {
            MandateCallClass::ProtocolObservation => {
                anyhow::ensure!(
                    ctx.mandate_authority.is_none(),
                    "Protocol observations cannot carry mutation authority."
                );
                Ok(true)
            }
            MandateCallClass::RecordDecision | MandateCallClass::RecordMeasurement => {
                anyhow::ensure!(
                    ctx.mandate_authority.is_none(),
                    "Decision and measurement recording cannot carry mutation authority."
                );
                Ok(true)
            }
            MandateCallClass::Observation => {
                anyhow::ensure!(
                    ctx.mandate_authority.is_none(),
                    "Observation calls cannot carry mutation authority."
                );
                if let Err(reason) = authorize_mandate_observation(
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
            MandateCallClass::ActControl | MandateCallClass::GovernedMutation => {
                let decision = self
                    .mandates()
                    .get_mandate_decision_for_run(&goal_run.id)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("A current ACT decision is required."))?;
                anyhow::ensure!(
                    decision.mandate_id == mandate.id
                        && decision.goal_run_id == goal_run.id
                        && decision.mandate_version == mandate.version
                        && decision.outcome == MandateDecisionOutcome::Act,
                    "The mandate no longer has a current ACT decision."
                );
                if class == MandateCallClass::ActControl {
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

                let mut expected = match authorize_mandate_action(
                    &mandate,
                    &decision,
                    name,
                    arguments,
                    &semantics,
                    &chrono::Utc::now(),
                ) {
                    MandateAuthorityDecision::Allow(expected) => expected,
                    MandateAuthorityDecision::Deny(reason) => {
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
                        mutation_audit_scope(&semantics)
                            .map_err(|reason| anyhow::anyhow!(reason.as_str()))?;
                    let now = chrono::Utc::now().to_rfc3339();
                    let reservation = MandateMutationReservation {
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
                        .mandates()
                        .reserve_mandate_action_attempt(&reservation)
                        .await?;
                    anyhow::ensure!(
                        reserved.is_some(),
                        "The mandate action budget, policy, or execution fence changed before dispatch."
                    );
                    let mut reserved_decision = decision.clone();
                    reserved_decision.action_attempts = grant.reserved_action_attempt;
                    if let Err(reason) = validate_mandate_grant(
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
                        .mandates()
                        .claim_mandate_mutation_dispatch(&MandateMutationDispatchClaim {
                            grant: grant.clone(),
                            goal_run_id: fence.goal_run_id.clone(),
                            root_task_id: fence.root_task_id.clone(),
                            root_task_attempt_id: fence.root_task_attempt_id.clone(),
                            task_id: fence.worker_task_id.clone(),
                            task_attempt_id: fence.attempt_id.clone(),
                            tool_call_id: actual_tool_call_id.to_string(),
                            tool_name: name.to_string(),
                            claimed_at: chrono::Utc::now().to_rfc3339(),
                        })
                        .await?;
                    anyhow::ensure!(
                        claimed,
                        "The exact mandate mutation reservation is stale, mismatched, or already dispatched."
                    );
                }
                Ok(true)
            }
            MandateCallClass::Deny => {
                anyhow::bail!("This call is not supported inside a mandate run.")
            }
        }
    }

    /// Run one tool call under the watchdog: the configured per-call timeout,
    /// except for a tool that manages its own timeout and backgrounding
    /// (`cli_agent`), which the watchdog would cancel early and orphan.
    async fn execute_tool_with_watchdog_outcome(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let name = name.trim();
        if self.exec_dispatch_facts(name, ctx).manages_own_timeout {
            return self.execute_tool_outcome(name, arguments, ctx).await;
        }
        let Some(timeout_dur) = self.turn_limits().llm_call_timeout else {
            return self.execute_tool_outcome(name, arguments, ctx).await;
        };
        match tokio::time::timeout(timeout_dur, self.execute_tool_outcome(name, arguments, ctx))
            .await
        {
            Ok(result) => result,
            Err(_) => {
                tracing::warn!(
                    session_id = ctx.session_id,
                    tool = name,
                    timeout_secs = timeout_dur.as_secs(),
                    "Tool call timed out"
                );
                Ok(ToolCallOutcome {
                    output: format!("Tool '{}' timed out after {}s", name, timeout_dur.as_secs()),
                    metadata: ToolCallMetadata {
                        outcome_status: Some(ToolOutcomeStatus::FailedRetryable),
                        timed_out: true,
                        ..ToolCallMetadata::default()
                    },
                    finish: None,
                })
            }
        }
    }

    /// The dispatch facts of the tool a call runs: the daemon tool, else the
    /// MCP tool judged at turn start. Taken from the tools, never from the
    /// call's arguments.
    fn exec_dispatch_facts(&self, name: &str, ctx: &ToolExecCtx<'_>) -> ToolDispatchFacts {
        self.tools()
            .iter()
            .find(|tool| tool.name() == name)
            .or_else(|| ctx.mcp_tool.filter(|tool| tool.name() == name))
            .map(|tool| tool.dispatch_facts())
            .unwrap_or_default()
    }

    /// The typed facts a tool sees about one call of `name`, from the turn
    /// (`ctx`), this agent and the stores. `mandate_preapproved` is left
    /// false: the caller sets it after revalidating the exact grant.
    async fn tool_call_context(
        &self,
        name: &str,
        ctx: &ToolExecCtx<'_>,
        under_mandate: bool,
        trusted: bool,
        workspace_grant: Option<&WorkspaceGrant>,
    ) -> ToolCallContext {
        let mut facts = ToolCallContext {
            mandate_preapproved: false,
            mandate_execution: under_mandate,
            mutation_forbidden: ctx.mutation_forbidden,
            kernel_approved: ctx.owner_approval_by_kernel.then_some(ApprovalScope::Call),
            trusted_session: trusted,
            user_role: Some(ctx.user_role),
            untrusted_source: crate::turn::loop_utils::is_trigger_session(ctx.session_id),
            scope_escalation: ctx.scope_escalation.filter(|e| !e.is_empty()).cloned(),
            session_id: ctx.session_id.to_string(),
            // A sub-agent's own grants are its parent chat's.
            grant_session: crate::approvals::grant_session(
                self.approval_session_id(),
                ctx.session_id,
            ),
            channel_visibility: Some(ctx.channel_visibility),
            channel_id: ctx.channel_id.map(str::to_string),
            task_id: ctx.task_id.map(str::to_string),
            tool_call_id: ctx.tool_call_id.map(str::to_string),
            lone_root_call: ctx.lone_root_call,
            turn_id: self
                .turn_runtime()
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
        if dispatch.records_launch_trust
            && let Some((roots, cwd)) = ctx.trust_roots {
                facts.trust_roots = Some(roots.to_vec());
                facts.trust_cwd = Some(cwd.to_string());
            }
        // A child inherits this turn's authority, not a re-resolution.
        if dispatch.spawns_child_agents
            && let Some(turn_authority) = ctx.turn_authority {
                facts.extensions.insert(turn_authority.clone());
            }
        // Goal context for the tools that need it. Background delivery routes
        // async/timeout notifications to the *origin* session
        // (goal.session_id), since internal child-agent sessions are not
        // routable.
        if dispatch.goal_context != GoalContext::None {
            if let Some(gid) = self.goal_id() {
                facts.goal_id = Some(gid.to_string());
            } else if dispatch.goal_context == GoalContext::BackgroundDelivery {
                // Executors typically carry a task, not a goal: resolve it
                // via the task so background notifications stay deliverable.
                if let Some(executor_task_id) = self.task_id()
                    && let Ok(Some(task)) = self.tasks().get_task(executor_task_id).await {
                        facts.goal_id = Some(task.goal_id);
                    }
            }
            if dispatch.goal_context == GoalContext::MandateRun
                && let Some(fence) = self.mandate_execution() {
                    facts.goal_run_id = Some(fence.goal_run_id.clone());
                    facts.task_attempt_id = Some(fence.attempt_id.clone());
                    facts.mandate_id = Some(fence.mandate_id.clone());
                    facts.mandate_version = Some(fence.mandate_version);
                }
        }
        if dispatch.needs_live_model {
            let runtime = self.llm_runtime().snapshot();
            let current_model = match tokio::time::timeout(
                Duration::from_secs(2),
                self.model_lock().read(),
            )
            .await
            {
                Ok(guard) => guard.clone(),
                Err(_) => runtime.primary_model(),
            };
            facts.model = Some(current_model);
            facts.provider_kind = Some(format!("{:?}", runtime.provider_kind()));
            facts.model_chain = runtime.router().map(|router| router.all_models_ordered());
        }
        facts
    }

    /// Run one tool call: mandate validation, the call's trust and role, a
    /// guest's workspace scoping, checkpoints, then the daemon tool or the
    /// MCP tool it names.
    async fn execute_tool_outcome(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let name = name.trim();
        let task_id = ctx.task_id;
        let under_mandate = self
            .validate_mandate_dispatch(name, arguments, ctx, false)
            .await?;
        // A mandate-authorized call is never a trusted session, even if the
        // channel is trusted or scheduled provenance would ordinarily
        // authorize it.
        let trusted = !under_mandate && self.session_trusted(ctx.trusted, task_id).await;

        let active_workspace_grant = match ctx.user_role {
            UserRole::Owner => None,
            UserRole::Guest => {
                anyhow::ensure!(
                    ctx.channel_visibility == ChannelVisibility::PrivateGroup,
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

        let parsed_args = match active_workspace_grant {
            Some(grant) => Ok(scoped_workspace_arguments(
                self.execution_backend(),
                name,
                arguments,
                grant,
                self.daemon_paths(),
            )
            .await?),
            None => serde_json::from_str::<Value>(arguments),
        };
        let enriched_args = match parsed_args {
            Ok(Value::Object(mut map)) => {
                strip_model_arguments(
                    &mut map,
                    self.tools()
                        .iter()
                        .find(|tool| tool.name() == name)
                        .map(|tool| tool.as_ref()),
                );
                // Every fact about the call reaches the tool as a typed
                // `ToolCallContext` field; the arguments stay the model's own.
                serde_json::to_string(&map)?
            }
            _ => arguments.to_string(),
        };

        let checkpoints_writes = !under_mandate
            && !ctx.mutation_forbidden
            && self.exec_dispatch_facts(name, ctx).checkpoints_writes;
        if let Some(grant) = active_workspace_grant {
            if checkpoints_writes
                && let Some(manager) = aidaemon_tools::checkpoints::active_manager() {
                    manager
                        .begin_for_tool(name, &enriched_args, &call_facts)
                        .await?;
                }
            let mut exec_ctx = call_facts.clone();
            exec_ctx.mandate_preapproved = self
                .claim_mandate_dispatch(under_mandate, name, arguments, ctx)
                .await?;
            return call_scoped_builtin_file_tool(
                self,
                name,
                &enriched_args,
                ctx.status_tx.clone(),
                exec_ctx,
            )
            .await
            .map(|outcome| sanitize_workspace_tool_outcome(outcome, grant))
            .map_err(|error| {
                anyhow::anyhow!(sanitize_workspace_tool_text(&error.to_string(), grant))
            });
        }

        if let Some(tool) = self.tools().iter().find(|tool| tool.name() == name) {
            let prepared = match prepare_tool_arguments(tool.as_ref(), &enriched_args) {
                Ok(prepared) => prepared,
                Err(rejection) => return Ok(*rejection),
            };
            if checkpoints_writes
                && let Some(manager) = aidaemon_tools::checkpoints::active_manager() {
                    manager
                        .begin_for_access_manifest(name, &prepared.access_manifest, &call_facts)
                        .await?;
                }
            let mut exec_ctx = call_facts.clone();
            exec_ctx.mandate_preapproved = self
                .claim_mandate_dispatch(under_mandate, name, arguments, ctx)
                .await?;
            let exec_ctx = with_status(exec_ctx, ctx.status_tx.clone());
            let result = tool
                .execute(&prepared.canonical_arguments, &exec_ctx)
                .await
                .map(|outcome| finalize_dispatched_outcome(&prepared, outcome));
            if result.as_ref().is_ok_and(|outcome| {
                outcome.metadata.background_started || outcome.metadata.detached
            })
                && let (Some(manager), Some(task_id)) =
                    (aidaemon_tools::checkpoints::active_manager(), task_id)
                {
                    manager
                        .mark_task_unsafe(
                            task_id,
                            "mutation continued in a background or detached process",
                        )
                        .await;
                }
            return result;
        }

        // Workspace grants apply only to the dedicated built-in file tools,
        // handled above; never an MCP tool reusing an allowlisted name.

        // The MCP tool the kernel judged for this call; else (outside the
        // kernel) the registry's. A mandate bit reaches an MCP adapter only
        // after the same exact-grant revalidation as builtins.
        let mcp_tool = match ctx.mcp_tool {
            Some(judged) => Some(judged.clone()),
            // The kernel judged no MCP tool under this name for the turn.
            None if ctx.owner_approval_by_kernel => None,
            None => self.mcp_tool(name).await,
        };
        if let Some(tool) = mcp_tool {
            let prepared = match prepare_tool_arguments(tool.as_ref(), &enriched_args) {
                Ok(prepared) => prepared,
                Err(rejection) => return Ok(*rejection),
            };
            let mut exec_ctx = call_facts.clone();
            exec_ctx.mandate_preapproved = self
                .claim_mandate_dispatch(under_mandate, name, arguments, ctx)
                .await?;
            let exec_ctx = with_status(exec_ctx, ctx.status_tx.clone());
            return tool
                .execute(&prepared.canonical_arguments, &exec_ctx)
                .await
                .map(|outcome| finalize_dispatched_outcome(&prepared, outcome));
        }

        let mut available: Vec<String> =
            self.tools().iter().map(|t| t.name().to_string()).collect();
        available.extend(self.mcp_tool_names().await);
        Err(UnknownToolError {
            name: name.to_string(),
            available,
        }
        .into())
    }

    /// Under a mandate, revalidate the exact grant and claim the dispatch
    /// right before adapter I/O: whether the call is mandate-preapproved.
    async fn claim_mandate_dispatch(
        &self,
        under_mandate: bool,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<bool> {
        if !under_mandate {
            return Ok(false);
        }
        self.validate_mandate_dispatch(name, arguments, ctx, true)
            .await?;
        Ok(true)
    }
}

impl<H: TurnHost + ?Sized> ToolDispatch for H {}
