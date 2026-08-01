use super::*;
use crate::types::WorkspaceGrant;

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

        if let Some(grant) = active_workspace_grant {
            if matches!(name, "write_file" | "edit_file") {
                if let Some(manager) = crate::checkpoints::active_manager() {
                    manager.begin_for_tool(name, &enriched_args).await?;
                }
            }
            let exec_ctx = crate::traits::ToolExecutionContext {
                correction_preapproved: false,
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

            if result.is_ok() {
                if let Some(ref tracker) = self.verification_tracker {
                    if matches!(name, "read_file" | "write_file" | "edit_file") {
                        if let Some(path) = extract_file_path_from_args(&enriched_args) {
                            tracker.record_seen_path(session_id, &path).await;
                        }
                    }
                }
            }
            return result;
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

        // Workspace grants apply only to the daemon's dedicated built-in file
        // tools. Never fall through to an MCP tool that happens to reuse an
        // allowlisted name, because its implementation/provenance is external.
        if active_workspace_grant.is_some() {
            anyhow::bail!("The granted built-in file tool is unavailable.");
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
