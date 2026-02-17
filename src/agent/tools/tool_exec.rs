use super::*;

pub(super) struct ToolExecCtx<'a> {
    pub session_id: &'a str,
    pub task_id: Option<&'a str>,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub channel_visibility: ChannelVisibility,
    pub channel_id: Option<&'a str>,
    pub trusted: bool,
    pub user_role: UserRole,
}

impl Agent {
    pub(super) async fn execute_tool_with_watchdog(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<String> {
        let session_id = ctx.session_id;
        // `cli_agent` can legitimately run longer than the generic watchdog
        // because it manages its own timeout/backgrounding behavior.
        // Wrapping it here causes premature cancellation (and can orphan the
        // underlying child process).
        if name == "cli_agent" {
            return self.execute_tool(name, arguments, ctx).await;
        }
        if let Some(timeout_dur) = self.llm_call_timeout {
            match tokio::time::timeout(timeout_dur, self.execute_tool(name, arguments, ctx)).await {
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
            self.execute_tool(name, arguments, ctx).await
        }
    }

    pub(super) async fn execute_tool(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<String> {
        let session_id = ctx.session_id;
        let task_id = ctx.task_id;
        let channel_visibility = ctx.channel_visibility;
        let channel_id = ctx.channel_id;
        let trusted = ctx.trusted;
        let user_role = ctx.user_role;

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
                // (goal.session_id), since sub-agent sessions ("sub-...") are not routable.
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
                serde_json::to_string(&map)?
            }
            _ => arguments.to_string(),
        };

        // Path verification pre-check: gate file-modifying terminal commands
        if name == "terminal" {
            if let Some(ref tracker) = self.verification_tracker {
                if let Some(cmd) = extract_command_from_args(&enriched_args) {
                    if let Some(warning) = tracker.check_modifying_command(session_id, &cmd).await {
                        return Ok(format!(
                            "[VERIFICATION WARNING] {}\nUnverified paths: {}\n\
                             Verify targets exist using 'ls' or 'stat' first, then retry.",
                            warning.message,
                            warning.unverified_paths.join(", ")
                        ));
                    }
                }
            }
        }

        for tool in &self.tools {
            if tool.name() == name {
                let result = tool
                    .call_with_status(&enriched_args, ctx.status_tx.clone())
                    .await;

                // Post-execution: record seen paths from successful commands
                if result.is_ok() {
                    if let Some(ref tracker) = self.verification_tracker {
                        match name {
                            "terminal" => {
                                if let Some(cmd) = extract_command_from_args(&enriched_args) {
                                    tracker.record_from_command(session_id, &cmd).await;
                                }
                            }
                            "send_file" => {
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

        // Search MCP registry for dynamically registered tools
        if let Some(ref registry) = self.mcp_registry {
            if let Some(tool) = registry.find_tool(name).await {
                return tool
                    .call_with_status(&enriched_args, ctx.status_tx.clone())
                    .await;
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
