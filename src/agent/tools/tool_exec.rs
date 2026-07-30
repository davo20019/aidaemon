use super::*;

pub(super) struct ToolExecCtx<'a> {
    pub session_id: &'a str,
    pub task_id: Option<&'a str>,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub channel_visibility: ChannelVisibility,
    pub channel_id: Option<&'a str>,
    pub project_scope: Option<&'a str>,
    pub trusted: bool,
    pub user_role: UserRole,
    /// Set by the correction gate when this specific tool call has already
    /// been classified as allowed for unattended execution.
    /// False on all normal (non-correction) paths.
    pub correction_preapproved: bool,
    /// When true, the `_trusted_session` enrichment flag must NOT be injected
    /// into tool args (the correction sandbox overrides trusted-session semantics).
    /// False on all normal paths.
    pub suppress_trusted_session: bool,
}

// impl-Agent justification: tool dispatch with watchdog over tools/state/event_store/verification_tracker.
impl Agent {
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
                    anyhow::bail!("Tool '{}' timed out after {}s", name, timeout_dur.as_secs());
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
        // Trust suppression: when the correction gate sets suppress_trusted_session,
        // we must NOT inject `_trusted_session` even if ChannelContext.trusted is true
        // or scheduled provenance would ordinarily authorize it.  Setting ctx.trusted=false
        // alone is insufficient because the OR below would still pick up scheduled
        // provenance; the entire expression must short-circuit to false.
        let trusted = if ctx.suppress_trusted_session {
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

        if user_role != UserRole::Owner {
            anyhow::bail!("Tool access denied: only owners can use tools.");
        }

        let enriched_args = match serde_json::from_str::<Value>(arguments) {
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
                // Inject goal context for tools that need it (e.g. spawn_agent, cli_agent, terminal).
                //
                // `cli_agent` uses this to route async/timeout notifications to the *origin* session
                // (goal.session_id), since internal child-agent sessions are not routable.
                //
                // `terminal` uses this for the same reason when commands move to background.
                if matches!(name, "spawn_agent" | "cli_agent" | "terminal") {
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

        // Path verification pre-check: gate file-modifying terminal commands
        if name == "terminal" {
            if let Some(ref tracker) = self.verification_tracker {
                if let Some(cmd) = extract_command_from_args(&enriched_args) {
                    if let Some(warning) = tracker.check_modifying_command(session_id, &cmd).await {
                        return Ok(crate::traits::ToolCallOutcome::from_output(format!(
                            "[VERIFICATION WARNING] {}\nUnverified paths: {}\n\
                                 Verify targets exist using 'ls' or 'stat' first, then retry.",
                            warning.message,
                            warning.unverified_paths.join(", ")
                        )));
                    }
                }
            }
        }

        for tool in &self.tools {
            if tool.name() == name {
                if name != "terminal"
                    && matches!(
                        name,
                        "write_file" | "edit_file" | "run_command" | "cli_agent"
                    )
                {
                    if let Some(manager) = crate::checkpoints::active_manager() {
                        manager.begin_for_tool(name, &enriched_args).await?;
                    }
                }
                let exec_ctx = crate::traits::ToolExecutionContext {
                    correction_preapproved: ctx.correction_preapproved,
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

                // Post-execution: record seen paths from successful commands
                if result.is_ok() {
                    if let Some(ref tracker) = self.verification_tracker {
                        match name {
                            "terminal" | "run_command" => {
                                if let Some(cmd) = extract_command_from_args(&enriched_args) {
                                    tracker.record_from_command(session_id, &cmd).await;
                                }
                            }
                            "send_file" | "read_file" | "write_file" | "edit_file" => {
                                if let Some(path) = extract_file_path_from_args(&enriched_args) {
                                    tracker.record_seen_path(session_id, &path).await;
                                }
                            }
                            _ => {}
                        }
                    }
                }

                return result;
            }
        }

        // Search MCP registry for dynamically registered tools.
        // MCP tools are never correction-preapproved in 3b; they receive the
        // default exec_ctx (correction_preapproved=false).
        if let Some(ref registry) = self.mcp_registry {
            if let Some(tool) = registry.find_tool(name).await {
                let exec_ctx = crate::traits::ToolExecutionContext {
                    correction_preapproved: false,
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
