//! Preparing a CLI agent launch: mode, readiness, executable resolution,
//! argument checks, sandbox shaping, working-directory claims, and prompt
//! enrichment.

use super::*;

impl CliAgentTool {
    /// Make a launch ready up to its process start: its agent in the
    /// owner's mode, its arguments, its prompt, and its claim on its
    /// working directory (taken now, so two runs that would change the
    /// same files are refused up front, queued or not, unless it
    /// `waits_for_folder`: then it waits for the folder instead
    /// ([`PreparedRun::folder_wait`])). `Ok(Err(..))` is why it was not
    /// started.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn prepare_run(
        &self,
        tool_name: &str,
        task_id: &str,
        prompt: &str,
        task_prompt: &str,
        working_dir: Option<&str>,
        session_id: &str,
        delegated_task_id: Option<&str>,
        system_instruction: Option<&str>,
        workspace_mode: CliWorkspaceMode,
        read_paths: &[String],
        write_paths: &[String],
        enforce_native_access: bool,
        spec: &crate::helper_run::HelperSpec,
        project: Option<&aidaemon_core::ProjectRef>,
        nested: Option<aidaemon_core::NestedCalls>,
        detached_judge: Option<Arc<crate::approvals::DetachedNestedJudge>>,
        spawn_family: Option<crate::authority::SpawnFamily>,
        waits_for_folder: bool,
        defer_claim: bool,
        direct_agent: bool,
        mcp_servers: &[aidaemon_coding_agents::McpEndpoint],
        nests_in: &[String],
        // A peer run: the most its lead may do. A peer never runs in a mode
        // above the agent that started it.
        mode_cap: Option<aidaemon_coding_agents::PermissionMode>,
    ) -> anyhow::Result<Result<PreparedRun, String>> {
        // Get entry from the tools map (clone what we need, release lock)
        let (mut command, mut args, timeout, max_output, is_dynamic, coding_agent) = {
            let tools = self.tools.read().unwrap();
            let entry = tools
                .get(tool_name)
                .ok_or_else(|| anyhow::anyhow!("Unknown CLI agent tool: {}", tool_name))?;
            (
                entry.command.clone(),
                entry.args.clone(),
                entry.timeout,
                entry.max_output_chars,
                entry.is_dynamic,
                self.entry_agent(entry),
            )
        };
        // The owner's mode for this agent decides how it runs
        // (`agent_mode`). Only the dispatcher's path enforces it.
        use aidaemon_coding_agents::PermissionMode;
        let mode = self.permissions().agent_mode(tool_name, &*coding_agent);
        // The model the owner chose for it, if any (Settings, Coding agents).
        let model = coding_agent
            .takes_model()
            .then(|| self.permissions().model_of(tool_name))
            .flatten();
        let mode = mode.or((!enforce_native_access).then_some(PermissionMode::Checked));
        let mode = capped_mode(&*coding_agent, mode, mode_cap);
        let Some(mode) = mode else {
            let level = self.permissions().level();
            let program = coding_agent.name();
            mark_refused_before_launch(PrelaunchRefusal::Policy);
            return Ok(Err(
                if aidaemon_coding_agents::modes_within(
                    &*coding_agent,
                    crate::permissions::agent_cap(level),
                )
                .is_empty()
                {
                    format!(
                        "Blocked: {program} (`{tool_name}`) may not run delegated work: none of \
                         its permission modes fits under AIdaemon's own level ({}), and no \
                         coding agent may do more than AIdaemon. The owner can raise the level \
                         in Settings (AIdaemon permissions).",
                        level.title()
                    )
                } else {
                    format!(
                        "Blocked: {program} (`{tool_name}`) may not run delegated work: AIDaemon \
                         can't check its actions, and the owner has not given it full access \
                         (Settings, Coding agent permissions)."
                    )
                },
            ));
        };
        // Its plan ran out (its own answer after a run of it failed): not
        // started until the usage resets, so the work goes elsewhere.
        if let Some(until) = crate::tools::agent_mode::out_of_usage(tool_name) {
            mark_refused_before_launch(PrelaunchRefusal::Policy);
            return Ok(Err(format!(
                "Blocked: {} (`{tool_name}`) is out of usage on its plan until {}. Hand this \
                 to another agent, or wait until then.",
                coding_agent.name(),
                aidaemon_work::schedule::owner_time(&until)
            )));
        }
        // One that last said it cannot work is asked again first (it may
        // have been signed in since); still not usable, it is not started.
        if !crate::tools::agent_mode::readiness_of(tool_name).is_usable()
            && let aidaemon_coding_agents::Readiness::NotUsable { reason, fix } =
                crate::tools::agent_mode::recheck(tool_name).await
        {
            mark_refused_before_launch(PrelaunchRefusal::Policy);
            return Ok(Err(format!(
                "Blocked: {} (`{tool_name}`) is unavailable: {reason}{}",
                coding_agent.name(),
                fix.map(|fix| format!(" {fix}")).unwrap_or_default()
            )));
        }
        // How it signs in (Settings, Coding agents): its own sign-in adds
        // nothing; a key the owner gave it is handed to it at launch.
        let signed = match self
            .permissions()
            .sign_in_of(tool_name)
            .and_then(|sign_in| coding_agent.signed_in(&sign_in))
        {
            Ok(signed) => signed,
            Err(reason) => {
                mark_refused_before_launch(PrelaunchRefusal::Policy);
                return Ok(Err(format!(
                    "Blocked: {} (`{tool_name}`) can't sign in: {reason}",
                    coding_agent.name()
                )));
            }
        };
        info!(
            agent = %tool_name,
            mode = mode.as_str(),
            model = model.as_deref().unwrap_or("its own"),
            "A coding agent runs in its mode"
        );
        // Full access: nothing is asked; the owner chose it.
        let nested = nested.filter(|_| mode != PermissionMode::FullAccess);
        let detached_judge = detached_judge.filter(|_| mode != PermissionMode::FullAccess);
        let can_host = (nested.is_some() || detached_judge.is_some())
            && self.backend.kind() == aidaemon_tools::backend::BackendKind::Local;

        // Re-check both the launcher and a wrapped nested executable. Agent
        // binaries may move during self-update, and GUI daemon PATH values can
        // differ from an interactive shell.
        match resolve_agent_launch(&self.backend, &command, &args).await {
            Ok((resolved_command, resolved_args)) => {
                command = resolved_command;
                args = resolved_args;
            }
            Err(error) => {
                // Auto-disable dynamic agents that disappeared
                if let Ok(agents) = self.state.list_dynamic_cli_agents().await
                    && let Some(mut agent) = agents.into_iter().find(|a| a.name == tool_name)
                {
                    agent.enabled = false;
                    let _ = self.state.update_dynamic_cli_agent(&agent).await;
                }
                let message = format!(
                    "ERROR: CLI agent '{tool_name}' could not start: {error}. \
                 Re-register it with an absolute executable path or reinstall it."
                );
                Self::quarantine_unavailable_tool(
                    &self.tools,
                    &self.tool_names,
                    tool_name,
                    "launch resolution failed",
                );
                return Err(CliAgentUnavailableError {
                    tool_name: tool_name.to_string(),
                    message,
                }
                .into());
            }
        }

        let agent_kind = std::path::Path::new(&command)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !is_env_launcher(name))
            .unwrap_or(tool_name);
        let agent_kind = if tool_name.eq_ignore_ascii_case("gemini") {
            "gemini"
        } else {
            agent_kind
        };
        if strip_permission_bypass_flags(&mut args, agent_kind).is_err() {
            mark_refused_before_launch(PrelaunchRefusal::Policy);
            return Ok(Err(
                "Blocked: CLI delegation was not started because its permission-bypass flags could not be removed."
                    .to_string(),
            ));
        }
        // The agent's own arguments start after an `env` launcher's.
        let launch_start = if is_env_launcher(&command) {
            env_wrapped_program_index(&args).map_or(0, |index| index + 1)
        } else {
            0
        };
        if is_dynamic
            && let Some(allowed) = coding_agent.launch_arguments()
            && let Some(unlisted) = allowed.unlisted(&args[launch_start..])
        {
            mark_refused_before_launch(PrelaunchRefusal::Policy);
            return Ok(Err(format!(
                "Blocked: CLI delegation was not started because the stored agent \
                 '{tool_name}' launches with `{unlisted}`, which is not on the list of \
                 arguments allowed for {}. Remove it with manage_cli_agents, or the owner can \
                 define this agent under [cli_agents.tools] in config.toml.",
                coding_agent.name()
            )));
        }

        let canonical_working_dir = match working_dir {
            Some(dir) => Some(Self::normalize_working_dir(&self.backend, dir).await?),
            None => None,
        };
        let mut canonical_read_paths = Vec::new();
        if let Some(working_dir) = canonical_working_dir.as_ref() {
            canonical_read_paths.push(working_dir.clone());
        }
        for path in read_paths {
            let path =
                Self::normalize_access_path(&self.backend, path, canonical_working_dir.as_deref())
                    .await?;
            if !canonical_read_paths.contains(&path) {
                canonical_read_paths.push(path);
            }
        }
        let mut canonical_write_paths = Vec::new();
        for path in write_paths {
            let path =
                Self::normalize_access_path(&self.backend, path, canonical_working_dir.as_deref())
                    .await?;
            if !canonical_write_paths.contains(&path) {
                canonical_write_paths.push(path);
            }
        }
        // The agent shapes its own launch for its mode
        // (`aidaemon_coding_agents`): its sandbox, its flags, and whether a
        // host answers its requests. Direct-agent conversations skip the
        // 3-sentence report schema so the agent can answer the owner naturally.
        let base_launch = aidaemon_coding_agents::Launch {
            args,
            start: launch_start,
        };
        let mut launch = base_launch.clone();
        let plan = match coding_agent.prepare(
            &aidaemon_coding_agents::RunRequest {
                mode,
                enforce: enforce_native_access,
                read_only: workspace_mode == CliWorkspaceMode::ReadOnly,
                working_dir: canonical_working_dir.as_deref(),
                read_paths: &canonical_read_paths,
                write_paths: &canonical_write_paths,
                can_host,
                stdin_open: self.backend.kind() == aidaemon_tools::backend::BackendKind::Local,
                model: model.as_deref(),
                resume: spec
                    .continues
                    .as_ref()
                    .map(|earlier| earlier.agent_session.as_str()),
                report: !direct_agent,
                mcp_servers,
            },
            &mut launch,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                mark_refused_before_launch(PrelaunchRefusal::Policy);
                if direct_agent {
                    return Ok(Err(format!(
                        "{} could not start: {error}.",
                        coding_agent.name()
                    )));
                }
                return Ok(Err(format!(
                    "Blocked: CLI delegation was not started because {error}. Select an agent with a registered native sandbox adapter or provide a narrower access manifest."
                )));
            }
        };
        let mut args = launch.args;
        let aidaemon_coding_agents::Plan {
            prompt: prompt_placement,
            env: mut plan_env,
            env_remove: mut plan_env_remove,
            host: host_starter,
            keep_alive,
            keeps_session,
        } = plan;
        args.extend(signed.args.iter().cloned());
        plan_env.extend(signed.env.iter().cloned());
        plan_env_remove.extend(signed.env_remove.iter().cloned());
        let writes = WriteScope::of(workspace_mode, &canonical_write_paths);
        let short_summary: String = task_prompt.chars().take(50).collect();

        // Claim the working directory (or the first explicit write path when
        // working_dir was omitted) and perform conflict checks.
        // Lock ordering when both locks are needed: running -> working_dir_claims.
        let claim_dir = canonical_working_dir.clone().or_else(|| {
            (workspace_mode == CliWorkspaceMode::ReadWrite)
                .then(|| canonical_write_paths.first().cloned())
                .flatten()
        });
        let mut claim_guard: Option<WorkingDirClaimGuard> = None;
        let mut folder_wait: Option<FolderWait> = None;
        if !defer_claim && let Some(ref dir) = claim_dir {
            let blocked_message: Option<String> = {
                let running = self.running.lock().await;
                let mut claims = lock_claims(&self.working_dir_claims);
                let gone = |id: &str| {
                    id == task_id
                        || nests_in.iter().any(|ancestor| ancestor == id)
                        || running
                            .get(id)
                            .is_some_and(|agent| agent.finished.load(Ordering::Acquire))
                };
                let conflict = conflicting(&claims, dir, &writes, gone).map(|(held_dir, claim)| {
                    (conflict_message(held_dir, claim), claim.tool_name.clone())
                });
                if let Some((message, behind)) = conflict {
                    if waits_for_folder {
                        folder_wait = Some(FolderWait {
                            dir: dir.clone(),
                            writes: writes.clone(),
                            prompt_summary: short_summary.clone(),
                            behind,
                        });
                        None
                    } else {
                        Some(message)
                    }
                } else {
                    claims
                        .entry(dir.clone())
                        .or_default()
                        .push(WorkingDirClaim {
                            task_id: task_id.to_string(),
                            tool_name: tool_name.to_string(),
                            prompt_summary: short_summary.clone(),
                            writes: writes.clone(),
                            state: ClaimState::Reserved,
                        });
                    None
                }
            };

            if let Some(message) = blocked_message {
                if let Some(task_id) = delegated_task_id {
                    Self::persist_delegated_cli_result_with_state(
                        self.state.clone(),
                        task_id,
                        Some(&message),
                        None,
                    )
                    .await;
                }
                mark_refused_before_launch(PrelaunchRefusal::Contention);
                return Ok(Err(message));
            }

            if folder_wait.is_none() {
                claim_guard = Some(WorkingDirClaimGuard::new(
                    self.working_dir_claims.clone(),
                    self.claims_changed.clone(),
                    dir.clone(),
                    task_id.to_string(),
                ));
            }
        }

        let task = task_prompt.trim().to_string();
        let prompt_with_rules = if coding_agent.launch_arguments().is_some() {
            format!(
                "{}\n\n{}",
                prompt.trim_end(),
                crate::helper_run::mission_rules(&spec.new_folders).trim()
            )
        } else {
            prompt.to_string()
        };
        let prompt = prompt_with_rules.as_str();
        let final_prompt = if let Some(instruction) = system_instruction {
            self.build_enriched_prompt(
                session_id,
                instruction,
                prompt,
                canonical_working_dir.as_deref(),
                project,
            )
            .await
        } else {
            prompt.to_string()
        };
        Ok(Ok(PreparedRun {
            command,
            args,
            base_launch,
            enforce_native_access,
            read_paths: canonical_read_paths,
            model,
            can_host,
            mode,
            coding_agent,
            nested,
            detached_judge,
            spawn_family,
            timeout,
            max_output,
            workspace_mode,
            working_dir: canonical_working_dir,
            claim_dir,
            write_scope: writes,
            write_paths: canonical_write_paths,
            prompt_placement,
            env: plan_env,
            env_remove: plan_env_remove,
            signed,
            host: host_starter,
            keep_alive,
            mcp_guard: None,
            keeps_session,
            continues: spec.continues.clone(),
            mcp_servers: mcp_servers.to_vec(),
            prompt: prompt.to_string(),
            task,
            final_prompt,
            prior_context: spec.prior_context.clone(),
            claim: claim_guard,
            folder_wait,
            thread_wait: None,
        }))
    }
}

impl CliAgentTool {
    /// Whether a run of `tool_name` may read any file on the machine: its
    /// mode, held to `cap`, is full access and enforced (as
    /// [`Self::prepare_run`] decides it). Otherwise it reads only its
    /// working folder, as far as a file shared with it goes.
    pub(super) fn reads_anywhere(
        &self,
        tool_name: &str,
        enforce_native_access: bool,
        cap: Option<aidaemon_coding_agents::PermissionMode>,
    ) -> bool {
        if !enforce_native_access {
            return false;
        }
        let Some(agent) = self
            .tools
            .read()
            .ok()
            .and_then(|tools| tools.get(tool_name).map(|entry| self.entry_agent(entry)))
        else {
            return false;
        };
        let mode = self.permissions().agent_mode(tool_name, &*agent);
        capped_mode(&*agent, mode, cap) == Some(aidaemon_coding_agents::PermissionMode::FullAccess)
    }
}

/// `mode` held to `cap`: above it, the most `agent` offers within it
/// (`None` when it offers nothing that low, so it does not run).
pub(super) fn capped_mode(
    agent: &dyn aidaemon_coding_agents::CodingAgent,
    mode: Option<aidaemon_coding_agents::PermissionMode>,
    cap: Option<aidaemon_coding_agents::PermissionMode>,
) -> Option<aidaemon_coding_agents::PermissionMode> {
    match (mode, cap) {
        (Some(mode), Some(cap)) if mode > cap => {
            aidaemon_coding_agents::mode_within(agent, Some(mode), false, cap)
        }
        (mode, _) => mode,
    }
}

#[cfg(test)]
mod mode_cap_tests {
    use super::capped_mode;
    use aidaemon_coding_agents::PermissionMode::{Checked, EveryCommand, FullAccess};

    #[test]
    fn a_peer_never_runs_above_its_lead() {
        let claude = aidaemon_coding_agents::claude::ClaudeCode;
        assert_eq!(
            capped_mode(&claude, Some(FullAccess), Some(Checked)),
            Some(Checked)
        );
        assert_eq!(
            capped_mode(&claude, Some(FullAccess), None),
            Some(FullAccess)
        );
        assert_eq!(
            capped_mode(&claude, Some(EveryCommand), Some(FullAccess)),
            Some(EveryCommand)
        );
        let lowest = capped_mode(&claude, Some(Checked), Some(EveryCommand));
        assert!(lowest.is_none_or(|mode| mode <= EveryCommand));
    }
}
