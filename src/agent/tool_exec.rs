use super::*;

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tool_with_watchdog(
        &self,
        name: &str,
        arguments: &str,
        session_id: &str,
        task_id: Option<&str>,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_visibility: ChannelVisibility,
        channel_id: Option<&str>,
        trusted: bool,
        user_role: UserRole,
    ) -> anyhow::Result<String> {
        // `cli_agent` can legitimately run longer than the generic watchdog
        // because it manages its own timeout/backgrounding behavior.
        // Wrapping it here causes premature cancellation (and can orphan the
        // underlying child process).
        if name == "cli_agent" {
            return self
                .execute_tool(
                    name,
                    arguments,
                    session_id,
                    task_id,
                    status_tx,
                    channel_visibility,
                    channel_id,
                    trusted,
                    user_role,
                )
                .await;
        }
        if let Some(timeout_dur) = self.llm_call_timeout {
            match tokio::time::timeout(
                timeout_dur,
                self.execute_tool(
                    name,
                    arguments,
                    session_id,
                    task_id,
                    status_tx,
                    channel_visibility,
                    channel_id,
                    trusted,
                    user_role,
                ),
            )
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
            self.execute_tool(
                name,
                arguments,
                session_id,
                task_id,
                status_tx,
                channel_visibility,
                channel_id,
                trusted,
                user_role,
            )
            .await
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tool(
        &self,
        name: &str,
        arguments: &str,
        session_id: &str,
        task_id: Option<&str>,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_visibility: ChannelVisibility,
        channel_id: Option<&str>,
        trusted: bool,
        user_role: UserRole,
    ) -> anyhow::Result<String> {
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
                // Inject V3 context for task lead → executor spawning
                if name == "spawn_agent" {
                    if let Some(ref gid) = self.v3_goal_id {
                        map.insert("_goal_id".to_string(), json!(gid));
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
                let result = tool.call_with_status(&enriched_args, status_tx).await;

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
                return tool.call_with_status(&enriched_args, status_tx).await;
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
