//! `action="check"` and `action="kill"`, and the tool's entry point.

use super::*;

impl TerminalTool {
    /// Check on a background process: return partial output or final result.
    pub(super) async fn handle_check(&self, pid: u32) -> anyhow::Result<ToolCallOutcome> {
        let mut running = self.running.lock().await;

        let Some(proc) = running.get(&pid) else {
            drop(running);
            let mut completed = self.completed.lock().await;
            if let Some(done) = completed.remove(&pid) {
                return Ok(ToolCallOutcome {
                    output: done.output,
                    metadata: done.metadata,
                    finish: None,
                });
            }
            return Ok(ToolCallOutcome::completed_negative_result(format!(
                "No tracked process with pid={}. It may have already finished and been reaped.",
                pid
            )));
        };

        if proc.reader_handle.is_finished() {
            // Process done — collect final output and remove from map.
            let proc = running.remove(&pid).unwrap();
            self.remove_indexes_for_process(pid, &proc).await;
            proc.notify_on_completion.store(false, Ordering::Relaxed);
            let semantics = proc.semantics.clone();
            let exit_code = proc.reader_handle.await.ok().flatten();
            let stdout = String::from_utf8_lossy(&proc.stdout_buf.lock().await).to_string();
            let stderr = String::from_utf8_lossy(&proc.stderr_buf.lock().await).to_string();
            let mut output = format!(
                "[Process pid={} finished after {:.0}s]\n",
                pid,
                proc.started_at.elapsed().as_secs_f64()
            );
            let (formatted, truncation) = format_output(
                &stdout,
                &stderr,
                self.max_output_chars,
                self.spill_session(&proc.notify_session_id),
            );
            output.push_str(&formatted);
            push_exit_status(&mut output, exit_code);
            let mut metadata = background_completion_metadata(exit_code);
            metadata.truncation = truncation;
            metadata.semantics = semantics;
            Ok(ToolCallOutcome {
                output,
                metadata,
                finish: None,
            })
        } else {
            // Still running — return tail of buffer.
            let elapsed = proc.started_at.elapsed().as_secs();
            let stdout_tail = {
                let b = proc.stdout_buf.lock().await;
                let tail_start = b.len().saturating_sub(2000);
                String::from_utf8_lossy(&b[tail_start..]).to_string()
            };
            let stderr_tail = {
                let b = proc.stderr_buf.lock().await;
                let tail_start = b.len().saturating_sub(500);
                String::from_utf8_lossy(&b[tail_start..]).to_string()
            };
            let mut output = format!(
                "[Process pid={} still running ({} seconds elapsed, command: `{}`)]",
                pid, elapsed, proc.command
            );
            if proc.detached {
                output.push_str("\n[mode: detached]");
            } else if let Some(task_id) = proc.owner_task_id.as_deref() {
                output.push_str(&format!("\n[mode: task-owned, task_id={}]", task_id));
            } else if proc.notifier_active {
                output.push_str("\n[mode: background, notifications active]");
            }
            if !stdout_tail.is_empty() {
                output.push_str(&format!("\n\nRecent stdout:\n{}", stdout_tail));
            }
            if !stderr_tail.is_empty() {
                output.push_str(&format!("\n\nRecent stderr:\n{}", stderr_tail));
            }
            output.push_str(&format!(
                "\n\nUse action=\"check\" pid={} to check again, or action=\"kill\" pid={} to stop.",
                pid, pid
            ));
            // Looking at a running process starts nothing: the check is a
            // finished observation of it ("still running"). The invocation
            // still in flight is the launch, whose own receipt is the
            // provisional one and whose completion ends it. Typed as a
            // `Backgrounded` handoff carrying the launch's effects, each
            // check read as one more launch that never ended, and the run
            // that made it as never finished (live 2026-11-13: a Drupal
            // helper that checked its `ddev start` while it ran was judged
            // "didn't finish" though its site was up).
            let mut metadata = background_observation_metadata(
                ToolOutcomeStatus::Succeeded,
                proc.detached,
                proc.notifier_active && !proc.detached,
            );
            metadata.semantics = terminal_call_semantics(r#"{"action":"check"}"#);
            Ok(ToolCallOutcome {
                output,
                metadata,
                finish: None,
            })
        }
    }

    /// Kill a background process: SIGTERM, wait 2s, SIGKILL if needed.
    pub(super) async fn handle_kill(&self, pid: u32) -> anyhow::Result<ToolCallOutcome> {
        let mut running = self.running.lock().await;

        let Some(proc) = running.remove(&pid) else {
            return Ok(ToolCallOutcome::completed_negative_result(format!(
                "No tracked process with pid={}. It may have already finished.",
                pid
            )));
        };
        drop(running);
        self.remove_indexes_for_process(pid, &proc).await;
        self.completed.lock().await.remove(&pid);

        let detached = proc.detached;
        let (output, truncation) = self
            .terminate_running_process(pid, proc, "manual kill")
            .await?;
        let mut metadata =
            background_observation_metadata(ToolOutcomeStatus::Succeeded, detached, false);
        metadata.truncation = truncation;
        Ok(ToolCallOutcome {
            output,
            metadata,
            finish: None,
        })
    }

    /// Shared implementation for `call_with_status_outcome` and `call_with_execution_context`.
    ///
    /// For action="run": the hard blocks (`pre_approval_block`) always run
    /// here; the owner's approval ladder (`approval_ladder`) ran in the
    /// agent kernel before a call that carries
    /// `ToolCallContext::kernel_approved`, and refuses any other call
    /// it would ask about. Trust facts come only from `exec_ctx`.
    pub(super) async fn execute_terminal(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        exec_ctx: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let mutation_forbidden = exec_ctx.mutation_forbidden;
        let args: TerminalArgs = serde_json::from_str(arguments)?;
        // Keep exact paths and directory capabilities separate until the
        // sandbox request is built. The native adapter receives both as
        // authorized paths, while the typed call manifest preserves whether
        // a descendant grant was intentional.
        let mut declared_read_paths = args.read_paths.clone();
        declared_read_paths.extend(args.read_roots.iter().cloned());
        let mut declared_write_paths = args.write_paths.clone();
        declared_write_paths.extend(args.write_roots.iter().cloned());
        let declared_write_roots = args.write_roots.clone();

        // Reap any finished background processes on each call.
        self.reap_finished().await;

        // Route background completion notifications to the origin session
        // when this terminal run is tied to a goal/task lead.
        let mut notify_session_id = exec_ctx.session_id.clone();
        if let (Some(state), Some(goal_id)) = (self.goals.as_ref(), exec_ctx.goal_id.as_deref())
            && let Ok(Some(goal)) = state.get_goal(goal_id).await
            && !goal.session_id.trim().is_empty()
        {
            notify_session_id = goal.session_id;
        }

        let mut outcome = match TerminalAction::parse(&args.action) {
            TerminalAction::Check => {
                let pid = args
                    .pid
                    .ok_or_else(|| anyhow::anyhow!("pid is required for action=\"check\""))?;
                self.handle_check(pid).await?
            }
            TerminalAction::Kill => {
                let pid = args
                    .pid
                    .ok_or_else(|| anyhow::anyhow!("pid is required for action=\"kill\""))?;
                self.handle_kill(pid).await?
            }
            TerminalAction::TrustAll => {
                ToolCallOutcome::from_output(Self::TRUST_ALL_RETIRED.to_string())
            }
            TerminalAction::Run | TerminalAction::Unrecognized => {
                // "run", the default, or any other spelling
                let command = args
                    .command
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty());
                let script = args
                    .script
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty());
                let (command, script_via_stdin) = match (command, script) {
                    (Some(_), Some(_)) => anyhow::bail!(
                        "command and script are mutually exclusive for action=\"run\""
                    ),
                    (Some(command), None) => (command, command_requires_stdin_shell(command)),
                    (None, Some(script)) => (script, true),
                    (None, None) => {
                        anyhow::bail!("command or script is required for action=\"run\"")
                    }
                };

                let command_semantics =
                    terminal_run_semantics_from_access(true, !declared_write_paths.is_empty());
                if mutation_forbidden
                    && !terminal_run_annotations(
                        &args,
                        self.confinement,
                        &crate::tools::traits::TrustRoots::new(&[]),
                    )
                    .read_only
                {
                    return Ok(ToolCallOutcome::blocked("Blocked by the explicit read-only contract: the command is not proven observation-only. Use an observational command or ask the user to change the constraint.").with_semantics(command_semantics));
                }

                let trust = TerminalTrust::from_execution_context(&exec_ctx.session_id, exec_ctx);
                if let Some(block) = self.pre_approval_block(&args, command, &trust) {
                    return Ok(ToolCallOutcome::blocked(block));
                }

                // The owner's approval comes from the agent kernel: its
                // `ApprovalPolicy` ran this ladder (`owner_approval_request`)
                // before the call and asked the owner where it requires it.
                // The tool never asks. Outside the kernel, a command the
                // ladder would ask about is refused.
                if exec_ctx.kernel_approved != Some(crate::tools::traits::ApprovalScope::Call)
                    && self.approval_ladder(&args, command, &trust).await.is_some()
                {
                    anyhow::bail!(
                        "terminal command `{}` requires approval through the agent kernel; it \
                         was called outside an approved agent call",
                        truncate_str(command, 200)
                    );
                }

                // The checkpoint belongs after every command-safety and
                // user-approval gate, but immediately before process spawn.
                if let Some(manager) = aidaemon_tools::checkpoints::active_manager() {
                    let access_manifest = terminal_access_manifest(arguments);
                    manager
                        .begin_for_access_manifest("terminal", &access_manifest, exec_ctx)
                        .await?;
                }

                // The confined tier runs confined, as the ladder judged it.
                if let Some(access) = self.confined_tier(&args, command, &trust).await {
                    return self
                        .run_confined_tier(&args, command, access, &exec_ctx.session_id)
                        .await;
                }

                // A typed root grant is allowed to name a future directory.
                // The write grant itself authorizes the confined process to
                // create that root (the sandbox checks the created path
                // against the granted literal/subpath), so the command owns
                // its own "create the leaf" step. Pre-creating the leaf here
                // made the canonical first step of a create-a-workspace
                // request — a plain `mkdir <root>` — fail with EEXIST against
                // the runtime's own directory, and the post-failure undo then
                // contradicted the reported state. The runtime therefore
                // prepares only what the leaf's creation depends on: the
                // root's PARENT CHAIN (validated typed, so a dispatcher can
                // repair an unpreparable projection it authored without
                // involving the model), and the full root only when it is the
                // command's working directory, which must exist before exec.
                let mut created_future_roots: Vec<aidaemon_tools::BackendPath> = Vec::new();
                if !args.write_roots.is_empty() {
                    let cwd = if let Some(path) = args.working_dir.as_deref() {
                        Some(self.backend.resolve_path(path).await?)
                    } else {
                        None
                    };
                    let materialized_roots = materialized_write_roots(
                        &self.backend,
                        &args.write_roots,
                        &args.write_paths,
                        cwd.as_ref(),
                    )
                    .await;
                    for root in &materialized_roots {
                        // Preparation failures are the runtime's, not the
                        // command's: report them typed, attributed to the
                        // exact declaration, so the dispatcher can repair a
                        // projection it authored without involving the model.
                        let resolved =
                            match resolve_access_path(&self.backend, root, cwd.as_ref()).await {
                                Ok(resolved) => resolved,
                                Err(error) => {
                                    return Ok(ToolCallOutcome::runtime_preparation_failure(
                                        "write_roots",
                                        root,
                                        error,
                                    ));
                                }
                            };
                        let prepare_target = if cwd.as_ref() == Some(&resolved) {
                            Some(resolved.clone())
                        } else {
                            std::path::Path::new(resolved.as_str())
                                .parent()
                                .map(|parent| {
                                    aidaemon_tools::BackendPath::new(
                                        parent.to_string_lossy().to_string(),
                                    )
                                })
                        };
                        let Some(prepare_target) = prepare_target else {
                            continue;
                        };
                        let existed = self.backend.metadata(&prepare_target).await.is_ok();
                        if let Err(error) = self.backend.create_dir_all(&prepare_target).await {
                            return Ok(ToolCallOutcome::runtime_preparation_failure(
                                "write_roots",
                                root,
                                error,
                            ));
                        }
                        if !existed {
                            created_future_roots.push(prepare_target);
                        }
                    }
                }

                let launch_trust = match exec_ctx.trust_roots.as_deref() {
                    Some(roots) => self.output_trust(
                        arguments,
                        &crate::tools::traits::TrustRoots::new(roots)
                            .with_cwd(exec_ctx.trust_cwd.as_deref()),
                    ),
                    None => aidaemon_core::SourceTrust::Untrusted,
                };
                let run_outcome = self
                    .handle_run(TerminalRunRequest {
                        command,
                        title: call_title(&args).as_deref(),
                        script_via_stdin,
                        working_dir: args.working_dir.as_deref(),
                        read_paths: &declared_read_paths,
                        write_paths: &declared_write_paths,
                        write_roots: &declared_write_roots,
                        notify_session_id: &notify_session_id,
                        notify_goal_id: exec_ctx.goal_id.as_deref(),
                        notifications_reach: notifications_reach(exec_ctx, &notify_session_id),
                        task_id: exec_ctx.task_id.as_deref(),
                        tool_call_id: exec_ctx.tool_call_id.as_deref(),
                        detach: args.detach,
                        background: args.background,
                        network: args.network,
                        confined: run_is_confined(self.confinement),
                        status_tx,
                        launch_trust,
                    })
                    .await;
                // A future root the runtime created is the runtime's own
                // side effect. If the command did not succeed and that leaf is
                // still an empty directory, undo it so a mis-typed grant (a
                // file declared as a root) cannot leave a directory squatting
                // on the command's target path for the next attempt.
                let failed = match run_outcome.as_ref() {
                    Ok(outcome) => outcome
                        .metadata
                        .exit_code
                        .or_else(|| extract_terminal_exit_code(&outcome.output))
                        .is_some_and(|code| code != 0),
                    Err(_) => true,
                };
                if failed {
                    for path in created_future_roots.iter().rev() {
                        let still_empty = self
                            .backend
                            .read_dir(path)
                            .await
                            .is_ok_and(|entries| entries.is_empty());
                        if still_empty && let Err(error) = self.backend.remove_empty_dir(path).await
                        {
                            tracing::debug!(path = %path, %error, "Could not undo runtime-created future write root");
                        }
                    }
                }
                run_outcome?
            }
        };

        if outcome.metadata.exit_code.is_none() {
            outcome.metadata.exit_code = extract_terminal_exit_code(&outcome.output);
        }
        if TerminalAction::parse(&args.action).runs_command()
            && outcome.metadata.invocation_stage.reached_dispatch()
        {
            outcome.metadata.access_enforcement = if run_is_confined(self.confinement) {
                confined_process_access_enforcement()
            } else {
                crate::tools::traits::ToolAccessEnforcement::NotApplicable
            };
        }

        Ok(outcome)
    }
}
