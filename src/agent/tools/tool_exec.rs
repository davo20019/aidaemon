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

fn validate_tool_arguments(
    tool: &dyn crate::traits::Tool,
    arguments: &str,
) -> Option<crate::traits::ToolCallOutcome> {
    match tool.validate_arguments(arguments) {
        Ok(()) => None,
        Err(violation) => Some(argument_contract_rejection_outcome(violation)),
    }
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
    if let Some(response) = outcome.metadata.direct_response.as_mut() {
        *response = sanitize_workspace_tool_text(response, grant);
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
    name: &str,
    arguments: &str,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    exec_ctx: crate::traits::ToolExecutionContext,
) -> anyhow::Result<crate::traits::ToolCallOutcome> {
    // Instantiate the daemon's exact built-in implementation. Looking up by
    // name in the general registry would let an injected/custom duplicate tool
    // inherit the collaborator allowlist merely by choosing the same name.
    let tool: Box<dyn crate::traits::Tool> = match name {
        "read_file" => Box::new(crate::tools::ReadFileTool),
        "search_files" => Box::new(crate::tools::SearchFilesTool),
        "write_file" => Box::new(crate::tools::WriteFileTool),
        "edit_file" => Box::new(crate::tools::EditFileTool),
        _ => anyhow::bail!("The workspace grant does not allow this tool."),
    };
    if let Some(rejection) = validate_tool_arguments(tool.as_ref(), arguments) {
        return Ok(rejection);
    }
    let mut outcome = tool
        .call_with_execution_context(arguments, status_tx, exec_ctx)
        .await?;
    let fallback = tool.call_semantics(arguments);
    outcome.metadata.semantics.merge_missing_from(fallback);
    Ok(outcome)
}

pub(super) struct ToolExecCtx<'a> {
    pub session_id: &'a str,
    pub task_id: Option<&'a str>,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub channel_visibility: ChannelVisibility,
    pub channel_id: Option<&'a str>,
    pub project_scope: Option<&'a str>,
    pub trusted: bool,
    pub user_role: UserRole,
    /// Present only after `ChannelContext::active_workspace_grant` validated
    /// the exact guest/workspace/channel/sender binding.
    pub workspace_grant: Option<&'a WorkspaceGrant>,
    /// Set by the correction gate when this specific tool call has already
    /// been classified as allowed for unattended execution.
    /// False on all normal (non-correction) paths.
    pub correction_preapproved: bool,
    /// When true, the `_trusted_session` enrichment flag must NOT be injected
    /// into tool args (the correction sandbox overrides trusted-session semantics).
    /// False on all normal paths.
    pub suppress_trusted_session: bool,
    /// Exact action-bound authority for a call made under an autonomous
    /// mandate. It is issued by the execution loop and never model-visible.
    pub mandate_authority: Option<&'a crate::traits::MandateAuthorityGrant>,
    /// Dispatcher-owned identity of the originating model tool call.
    ///
    /// This is general causal lineage, not mandate-specific state: transparent
    /// adapters and detached work must retain it so their result can be joined
    /// back to the request. Mandate validation also binds its grant to this
    /// independently supplied identity.
    pub tool_call_id: Option<&'a str>,
    /// Rust-side hard boundary inherited from the request contract.
    pub mutation_forbidden: bool,
}

async fn scoped_workspace_arguments(
    name: &str,
    arguments: &str,
    grant: &WorkspaceGrant,
) -> anyhow::Result<Value> {
    anyhow::ensure!(
        grant.allows_tool(name),
        "Tool access denied: this workspace grant does not allow '{}'.",
        name
    );

    let backend = crate::execution::active_execution_backend();
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
        !crate::tools::fs_utils::is_sensitive_path(&root_path)
            && crate::tools::fs_utils::find_nearest_project_root(&root_path)
                .is_some_and(|project| project == root_path),
        "The workspace grant no longer identifies a safe project root."
    );

    // Relative paths are intentionally rooted at the delegated project, not at
    // the daemon's broader execution workspace. This makes safe relative calls
    // usable without exposing the host's absolute path in conversation.
    let requested = if raw_path == "." {
        root_path.clone()
    } else if std::path::Path::new(raw_path).is_absolute() || raw_path.starts_with('~') {
        std::path::PathBuf::from(raw_path)
    } else {
        root_path.join(raw_path)
    };
    anyhow::ensure!(
        !crate::tools::fs_utils::is_sensitive_path(&requested),
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
        !crate::tools::fs_utils::is_sensitive_path(target_path),
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
    async fn mandate_goal_id_for_tool_exec(
        &self,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<Option<String>> {
        // A run-bound child remains a mandate execution even if its controller
        // record is concurrently removed or corrupted. Never downgrade that
        // child into the ordinary owner/tool path.
        if let Some(fence) = self.mandate_execution.as_ref() {
            anyhow::ensure!(
                self.goal_id.as_deref() == Some(fence.goal_id.as_str()),
                "Mandate child goal identity is inconsistent."
            );
            return Ok(Some(fence.goal_id.clone()));
        }
        if let Some(goal_id) = self.goal_id.as_deref() {
            if self.state.get_mandate_for_goal(goal_id).await?.is_some() {
                return Ok(Some(goal_id.to_string()));
            }
        }
        for task_id in [self.task_id.as_deref(), ctx.task_id].into_iter().flatten() {
            if let Some(task) = self.state.get_task(task_id).await? {
                if self
                    .state
                    .get_mandate_for_goal(&task.goal_id)
                    .await?
                    .is_some()
                {
                    return Ok(Some(task.goal_id));
                }
            }
        }
        Ok(None)
    }

    async fn semantics_for_exact_tool_call(
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
        let Some(goal_id) = self.mandate_goal_id_for_tool_exec(ctx).await? else {
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
        let semantics = self.semantics_for_exact_tool_call(name, arguments).await;
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
            crate::mandates::MandateCallClass::RecordDecision => {
                anyhow::ensure!(
                    ctx.mandate_authority.is_none(),
                    "Decision recording cannot carry mutation authority."
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
        // `cli_agent` can legitimately run longer than the generic watchdog
        // because it manages its own timeout/backgrounding behavior.
        // Wrapping it here causes premature cancellation (and can orphan the
        // underlying child process).
        if name == "cli_agent" {
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
                    })
                }
            }
        } else {
            self.execute_tool_outcome(name, arguments, ctx).await
        }
    }

    pub(super) async fn execute_tool_outcome(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let name = name.trim();
        let session_id = ctx.session_id;
        let task_id = ctx.task_id;
        let channel_visibility = ctx.channel_visibility;
        let channel_id = ctx.channel_id;
        let under_mandate = self
            .validate_mandate_dispatch(name, arguments, ctx, false)
            .await?;
        // Trust suppression: when the correction gate sets suppress_trusted_session,
        // we must NOT inject `_trusted_session` even if ChannelContext.trusted is true
        // or scheduled provenance would ordinarily authorize it.  Setting ctx.trusted=false
        // alone is insufficient because the OR below would still pick up scheduled
        // provenance; the entire expression must short-circuit to false.
        let trusted = if ctx.suppress_trusted_session || under_mandate {
            false
        } else {
            ctx.trusted
                || if let Some(goal_id) = self.goal_id.as_deref() {
                    // Scheduled goal runs are user-confirmed automation, so treat
                    // their tool calls as trusted even when the execution context
                    // was recreated later by heartbeat/orphan dispatch.
                    goal_has_scheduled_provenance(&self.state, goal_id, self.task_id.as_deref())
                        .await
                } else if let Some(executor_task_id) = self.task_id.as_deref() {
                    if let Ok(Some(task)) = self.state.get_task(executor_task_id).await {
                        goal_has_scheduled_provenance(
                            &self.state,
                            &task.goal_id,
                            Some(executor_task_id),
                        )
                        .await
                    } else {
                        task_has_scheduled_provenance(&self.state, Some(executor_task_id)).await
                    }
                } else if let Some(task_id) = task_id {
                    task_has_scheduled_provenance(&self.state, Some(task_id)).await
                } else {
                    false
                }
        };
        let user_role = ctx.user_role;
        let turn_id = self.current_turn_ids.read().await.get(session_id).cloned();

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

        let scoped_args = if let Some(grant) = active_workspace_grant {
            Some(scoped_workspace_arguments(name, arguments, grant).await?)
        } else {
            None
        };

        let parsed_args = match scoped_args {
            Some(args) => Ok(args),
            None => serde_json::from_str::<Value>(arguments),
        };
        let enriched_args = match parsed_args {
            Ok(Value::Object(mut map)) => {
                // Strip any underscore-prefixed fields the LLM might have injected
                // to prevent spoofing of internal enrichment fields.
                map.retain(|k, _| !k.starts_with('_'));
                map.insert("_session_id".to_string(), json!(session_id));
                map.insert(
                    "_channel_visibility".to_string(),
                    json!(channel_visibility.to_string()),
                );
                if let Some(ch_id) = channel_id {
                    map.insert("_channel_id".to_string(), json!(ch_id));
                }
                if let Some(tid) = task_id {
                    map.insert("_task_id".to_string(), json!(tid));
                }
                if let Some(tool_call_id) = ctx.tool_call_id {
                    map.insert("_tool_call_id".to_string(), json!(tool_call_id));
                }
                if let Some(ref turn_id) = turn_id {
                    map.insert("_turn_id".to_string(), json!(turn_id));
                }
                // Mark as untrusted if this session originated from an automated
                // trigger (e.g., email) rather than direct user interaction.
                // This forces tools like terminal to require explicit approval.
                if is_trigger_session(session_id) {
                    map.insert("_untrusted_source".to_string(), json!(true));
                }
                // Inject explicit trust flag from ChannelContext — only trusted
                // scheduled tasks set this. Never derived from session ID strings.
                if trusted {
                    map.insert("_trusted_session".to_string(), json!(true));
                }
                // Inject user role so tools can enforce role-based access control
                map.insert("_user_role".to_string(), json!(format!("{:?}", user_role)));
                if let Some(grant) = active_workspace_grant {
                    map.insert(
                        "_workspace_collaboration_root".to_string(),
                        json!(grant.project_root),
                    );
                }
                // Inject goal context for tools that need it (e.g. spawn_agent, cli_agent, terminal).
                //
                // `cli_agent` uses this to route async/timeout notifications to the *origin* session
                // (goal.session_id), since internal child-agent sessions are not routable.
                //
                // `terminal` uses this for the same reason when commands move to background.
                if matches!(
                    name,
                    "spawn_agent"
                        | "cli_agent"
                        | "terminal"
                        | "manage_mandates"
                        | "manage_goal_tasks"
                ) {
                    if let Some(ref gid) = self.goal_id {
                        map.insert("_goal_id".to_string(), json!(gid));
                    } else if matches!(name, "cli_agent" | "terminal") {
                        // Executors typically don't carry goal_id, but do carry task_id.
                        // Resolve goal_id via task so background notifications stay deliverable.
                        if let Some(ref executor_task_id) = self.task_id {
                            if let Ok(Some(task)) = self.state.get_task(executor_task_id).await {
                                map.insert("_goal_id".to_string(), json!(task.goal_id));
                            }
                        }
                    }
                    if matches!(name, "manage_mandates" | "manage_goal_tasks") {
                        if let Some(fence) = self.mandate_execution.as_ref() {
                            map.insert(
                                "_goal_run_id".to_string(),
                                json!(fence.goal_run_id.as_str()),
                            );
                            map.insert(
                                "_task_attempt_id".to_string(),
                                json!(fence.attempt_id.as_str()),
                            );
                            if name == "manage_goal_tasks" {
                                map.insert(
                                    "_mandate_id".to_string(),
                                    json!(fence.mandate_id.as_str()),
                                );
                                map.insert(
                                    "_mandate_version".to_string(),
                                    json!(fence.mandate_version),
                                );
                            }
                        }
                    }
                }
                if let Some(project_scope) = ctx.project_scope {
                    map.insert("_project_scope".to_string(), json!(project_scope));
                }
                #[cfg(feature = "computer_use")]
                if name == "computer_use" {
                    let runtime = self.llm_runtime.snapshot();
                    let current_model =
                        match tokio::time::timeout(Duration::from_secs(2), self.model.read()).await
                        {
                            Ok(guard) => guard.clone(),
                            Err(_) => runtime.primary_model(),
                        };
                    map.insert("_model".to_string(), json!(current_model));
                    map.insert(
                        "_provider_kind".to_string(),
                        json!(format!("{:?}", runtime.provider_kind())),
                    );
                    if let Some(router) = runtime.router() {
                        map.insert(
                            "_model_chain".to_string(),
                            json!(router.all_models_ordered()),
                        );
                    }
                }
                serde_json::to_string(&map)?
            }
            _ => arguments.to_string(),
        };

        if let Some(grant) = active_workspace_grant {
            if !under_mandate && matches!(name, "write_file" | "edit_file") {
                if let Some(manager) = crate::checkpoints::active_manager() {
                    manager.begin_for_tool(name, &enriched_args).await?;
                }
            }
            let mandate_preapproved = if under_mandate {
                self.validate_mandate_dispatch(name, arguments, ctx, true)
                    .await?;
                true
            } else {
                false
            };
            let exec_ctx = crate::traits::ToolExecutionContext {
                // Workspace-grant builtins are outside the correction approval
                // bypass; preserve the existing boundary between authority paths.
                correction_preapproved: false,
                mandate_preapproved,
                mandate_execution: under_mandate,
                mutation_forbidden: ctx.mutation_forbidden,
            };
            let result = call_scoped_builtin_file_tool(
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

            return result;
        }

        for tool in &self.tools {
            if tool.name() == name {
                if let Some(rejection) = validate_tool_arguments(tool.as_ref(), &enriched_args) {
                    return Ok(rejection);
                }
                if !under_mandate
                    && name != "terminal"
                    && matches!(
                        name,
                        "write_file" | "edit_file" | "run_command" | "cli_agent"
                    )
                    && !(name == "cli_agent" && ctx.mutation_forbidden)
                {
                    if let Some(manager) = crate::checkpoints::active_manager() {
                        manager.begin_for_tool(name, &enriched_args).await?;
                    }
                }
                let mandate_preapproved = if under_mandate {
                    self.validate_mandate_dispatch(name, arguments, ctx, true)
                        .await?;
                    true
                } else {
                    false
                };
                let exec_ctx = crate::traits::ToolExecutionContext {
                    correction_preapproved: ctx.correction_preapproved,
                    mandate_preapproved,
                    mandate_execution: under_mandate,
                    mutation_forbidden: ctx.mutation_forbidden,
                };
                let result = tool
                    .call_with_execution_context(&enriched_args, ctx.status_tx.clone(), exec_ctx)
                    .await
                    .map(|mut outcome| {
                        let fallback = tool.call_semantics(&enriched_args);
                        outcome.metadata.semantics.merge_missing_from(fallback);
                        outcome
                    });
                if result.as_ref().is_ok_and(|outcome| {
                    outcome.metadata.background_started || outcome.metadata.detached
                }) {
                    if let (Some(manager), Some(task_id)) =
                        (crate::checkpoints::active_manager(), task_id)
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
        // MCP adapters remain outside the correction approval bypass. A mandate
        // bit can reach one only after the same exact-grant revalidation used
        // for builtins (MCP tools are non-delegable under the v1 policy).
        if let Some(ref registry) = self.mcp_registry {
            if let Some(tool) = registry.find_tool(name).await {
                if let Some(rejection) = validate_tool_arguments(tool.as_ref(), &enriched_args) {
                    return Ok(rejection);
                }
                let mandate_preapproved = if under_mandate {
                    self.validate_mandate_dispatch(name, arguments, ctx, true)
                        .await?;
                    true
                } else {
                    false
                };
                let exec_ctx = crate::traits::ToolExecutionContext {
                    correction_preapproved: false,
                    mandate_preapproved,
                    mandate_execution: under_mandate,
                    mutation_forbidden: ctx.mutation_forbidden,
                };
                return tool
                    .call_with_execution_context(&enriched_args, ctx.status_tx.clone(), exec_ctx)
                    .await
                    .map(|mut outcome| {
                        let fallback = tool.call_semantics(&enriched_args);
                        outcome.metadata.semantics.merge_missing_from(fallback);
                        outcome
                    });
            }
        }

        let mut available: Vec<String> = self.tools.iter().map(|t| t.name().to_string()).collect();
        if let Some(ref reg) = self.mcp_registry {
            for info in reg.list_servers().await {
                available.extend(info.tool_names);
            }
        }
        anyhow::bail!(
            "Unknown tool '{}'. Available tools: [{}]. Use one of these or respond with text only.",
            name,
            available.join(", ")
        )
    }
}

#[cfg(test)]
#[path = "tool_watchdog_tests.rs"]
mod tool_watchdog_tests;

#[cfg(test)]
#[path = "correction_preapproval_tests.rs"]
mod correction_preapproval_tests;

#[cfg(test)]
mod scoped_workspace_tests {
    use super::*;
    use crate::types::WorkspaceAccessLevel;

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

        let inside = scoped_workspace_arguments("read_file", r#"{"path":"index.html"}"#, &grant)
            .await
            .unwrap();
        assert_eq!(
            std::fs::canonicalize(inside["path"].as_str().unwrap()).unwrap(),
            std::fs::canonicalize(root.path().join("index.html")).unwrap()
        );

        let traversal =
            scoped_workspace_arguments("read_file", r#"{"path":"../outside.txt"}"#, &grant).await;
        assert!(traversal.is_err());

        let absolute_outside = scoped_workspace_arguments(
            "read_file",
            &json!({"path": outside.path()}).to_string(),
            &grant,
        )
        .await;
        assert!(absolute_outside.is_err());

        let sensitive =
            scoped_workspace_arguments("read_file", r#"{"path":".env.local"}"#, &grant).await;
        assert!(sensitive.is_err());

        let dangerous =
            scoped_workspace_arguments("terminal", r#"{"command":"pwd"}"#, &grant).await;
        assert!(dangerous.is_err());

        let aliased = scoped_workspace_arguments(
            "read_file",
            r#"{"file_path":"index.html","filename":"../../outside"}"#,
            &grant,
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
        )
        .await
        .unwrap();
        assert!(std::path::Path::new(allowed["path"].as_str().unwrap())
            .starts_with(std::path::Path::new(&grant.project_root)));

        let missing_parent = scoped_workspace_arguments(
            "write_file",
            r#"{"path":"missing/new.txt","content":"no"}"#,
            &grant,
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

        let result =
            scoped_workspace_arguments("read_file", r#"{"path":"link.txt"}"#, &grant).await;
        assert!(result.is_err());
    }
}
