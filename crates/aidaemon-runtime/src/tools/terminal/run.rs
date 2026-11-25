//! `action="run"`: launching a command in the foreground or background.

use super::*;

impl TerminalTool {
    /// Run a command: spawn, wait up to initial_timeout, return output or move to background.
    pub(super) async fn handle_run(
        &self,
        request: TerminalRunRequest<'_>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let TerminalRunRequest {
            command,
            title,
            script_via_stdin,
            working_dir,
            read_paths,
            write_paths,
            write_roots,
            notify_session_id,
            notify_goal_id,
            notifications_reach,
            task_id,
            tool_call_id,
            detach,
            background,
            network,
            confined,
            status_tx,
            launch_trust,
        } = request;
        let has_write_targets = !write_paths.is_empty() || !write_roots.is_empty();
        let command_semantics = terminal_run_semantics_from_access(true, has_write_targets);
        // Snapshot which deletion-relevant paths exist before the process
        // runs, so a successful synchronous run can type observed deletions
        // (present → absent) on its receipt.
        let deletion_candidates =
            deletion_observation_candidates(command, working_dir, write_paths, write_roots);
        let mut deletion_preexisting: Vec<String> = Vec::new();
        for path in &deletion_candidates {
            let backend_path = aidaemon_tools::BackendPath::new(path.clone());
            if self.backend.metadata(&backend_path).await.is_ok() {
                deletion_preexisting.push(path.clone());
            }
        }
        let execution_mode = if script_via_stdin {
            "script"
        } else {
            "command"
        };
        let dedupe_identity = format!(
            "mode={execution_mode}\0cwd={}\0read={}\0write={}\0roots={}\0{command}",
            working_dir.unwrap_or_default(),
            serde_json::to_string(read_paths).unwrap_or_default(),
            serde_json::to_string(write_paths).unwrap_or_default(),
            serde_json::to_string(write_roots).unwrap_or_default(),
        );
        let dedupe_key =
            Self::dedupe_key_for_run(&dedupe_identity, notify_session_id, notify_goal_id, task_id);
        if let Some(existing_pid) = self.resolve_duplicate_running_pid(&dedupe_key).await {
            // The work is already running in the background: the run hands
            // off to it only when its notifier will report the result (a
            // task-owned job without one dies at task end).
            let reports_back = self
                .running
                .lock()
                .await
                .get(&existing_pid)
                .is_some_and(|proc| proc.notifier_active);
            let outcome = ToolCallOutcome {
                output: format!(
                    "Equivalent command is already running in this scope (pid={}). \
                     Use action=\"check\" pid={} to inspect progress or action=\"kill\" pid={} to stop it.",
                    existing_pid, existing_pid, existing_pid
                ),
                metadata: ToolCallMetadata {
                    outcome_status: Some(ToolOutcomeStatus::Backgrounded),
                    background_started: true,
                    semantics: command_semantics,
                    ..ToolCallMetadata::default()
                },
                finish: None,
            };
            return Ok(if reports_back {
                outcome.handed_to_background()
            } else {
                outcome
            });
        }

        let capabilities = ConfinedCapabilities {
            network,
            ..ConfinedCapabilities::default()
        };
        let execution_request = if !confined {
            host_terminal_execution_request(&self.backend, command, script_via_stdin, working_dir)
                .await?
        } else if script_via_stdin {
            confined_terminal_script_execution_request(
                &self.backend,
                command,
                working_dir,
                read_paths,
                write_paths,
                write_roots,
                capabilities,
            )
            .await?
        } else {
            confined_terminal_execution_request_inner(
                &self.backend,
                command,
                false,
                working_dir,
                read_paths,
                write_paths,
                write_roots,
                capabilities,
            )
            .await?
        };
        let mut spawned = self.backend.spawn(execution_request).await?;
        let process_handle = spawned.handle().clone();
        // The process moves into the reader task below, so dropping this call
        // (its run stopped) would leave it running: the guard ends its group
        // unless the call sees it finish or hands it to background tracking
        // (live 2026-11-21: a stopped sub-agent's `for … sleep 2` loop kept
        // writing to the file the owner asked to delete).
        let mut stop_with_call = StopWithCall(process_handle.local_pid());
        let pid = process_handle.display_id();
        let stdout_pipe = spawned.take_stdout().expect("stdout piped");
        let stderr_pipe = spawned.take_stderr().expect("stderr piped");
        let mut child = spawned.into_child();

        let stdout_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));

        let stdout_buf_c = stdout_buf.clone();
        let stderr_buf_c = stderr_buf.clone();
        let (completion_tx, completion_rx) = tokio::sync::oneshot::channel::<Option<i32>>();
        // Side channel for the child's terminating signal (if any). `code()`
        // is None for a signal-killed child, so the exit-code path alone cannot
        // distinguish "the program exited N" from "the kernel/sandbox killed it
        // with signal M" — the two demand different remedies. Capturing the
        // signal here lets a confined opaque failure be diagnosed truthfully.
        let signal_slot = Arc::new(Mutex::new(None::<i32>));
        let signal_slot_c = signal_slot.clone();

        // Spawn a task that drains both streams and then waits for the child to exit.
        let reader_handle = tokio::spawn(async move {
            let stdout_drain = drain_to_buffer(stdout_pipe, stdout_buf_c);
            let stderr_drain = drain_to_buffer(stderr_pipe, stderr_buf_c);
            tokio::join!(stdout_drain, stderr_drain);
            let status = child.wait().await.ok();
            let exit_code = status.as_ref().and_then(|status| status.code());
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                *signal_slot_c.lock().await = status.as_ref().and_then(|status| status.signal());
            }
            let _ = completion_tx.send(exit_code);
            exit_code
        });

        // Wait up to initial_timeout for the reader (and thus the process) to
        // finish. A detached or background launch waits only for the startup
        // grace.
        let foreground_wait = foreground_wait(
            self.initial_timeout,
            detach || background,
            notifications_reach,
        );
        // While it runs, the latest line it printed is its progress, at
        // most every few seconds: a helper's card line shows it (live
        // 2026-11-13: a 60 s count showed nothing until it ended).
        let output_progress = status_tx.clone();
        let (progress_stdout, progress_stderr) = (stdout_buf.clone(), stderr_buf.clone());
        let poll_finished = async {
            let mut shown = 0;
            let mut last_check = tokio::time::Instant::now();
            loop {
                if reader_handle.is_finished() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                let Some(tx) = output_progress.as_ref() else {
                    continue;
                };
                if last_check.elapsed() < COMMAND_OUTPUT_PROGRESS_EVERY {
                    continue;
                }
                last_check = tokio::time::Instant::now();
                let lines =
                    printed_lines(&progress_stdout.lock().await, &progress_stderr.lock().await);
                if lines != shown {
                    let _ = tx.try_send(StatusUpdate::CommandOutput {
                        name: "terminal".to_string(),
                        lines,
                        tail: None,
                    });
                    shown = lines;
                }
            }
        };

        let waited = tokio::time::timeout(foreground_wait, poll_finished).await;
        // Finished, or kept running on purpose: no longer this call's to end.
        stop_with_call.0 = None;
        match waited {
            Ok(()) => {
                // A command done before its first progress tick still shows
                // what it printed in the owner's app (live 2026-11-15: a quick
                // command showed no output under its step).
                if let Some(tx) = status_tx.as_ref() {
                    let (out, err) = (stdout_buf.lock().await, stderr_buf.lock().await);
                    let _ = tx.try_send(StatusUpdate::CommandOutput {
                        name: "terminal".to_string(),
                        lines: printed_lines(&out, &err),
                        tail: aidaemon_tools::status::command_output_tail(&out, &err),
                    });
                }
                // Process finished within timeout — collect output.
                let exit_code = reader_handle.await.ok().flatten();
                let stdout_data = stdout_buf.lock().await;
                let stderr_data = stderr_buf.lock().await;
                let mut stdout = String::from_utf8_lossy(&stdout_data).into_owned();
                let mut stderr = String::from_utf8_lossy(&stderr_data).into_owned();
                let mut exit_code = exit_code;
                let terminating_signal = *signal_slot.lock().await;
                drop(stdout_data);
                drop(stderr_data);
                // Ground-truth capture: when a confined command fails, record
                // the child's raw termination in the daemon log. A failure that
                // reaches the user as a bare "exit 255" with empty stderr is
                // otherwise undiagnosable from outside the daemon; this line ties
                // the exit code, the terminating signal, and the captured stream
                // sizes together so the true cause is visible in the daemon's own
                // log even when the child wrote nothing the agent can see.
                if exit_code != Some(0) {
                    warn!(
                        exit_code = ?exit_code,
                        terminating_signal = ?terminating_signal,
                        stdout_bytes = stdout.len(),
                        stderr_bytes = stderr.len(),
                        command = %LoggedCommand(command),
                        "Confined command finished with a non-zero result"
                    );
                }
                let mut self_heal_note = String::new();
                // Self-heal: if a confined command failed because the sandbox
                // denied a config file the tool needs at startup, grant that
                // exact read-only path and re-run ONCE. This is the general
                // form of the manual "grant the denied path and retry" fix —
                // no per-tool table, one bounded attempt, secret stores excluded.
                if confined && exit_code != Some(0) && !detach {
                    let home = self.backend.home_hint().to_string();
                    let combined = format!("{stdout}\n{stderr}");
                    let heal_paths = self_healable_denied_reads(&combined, &home, read_paths);
                    if !heal_paths.is_empty() {
                        let mut augmented = read_paths.to_vec();
                        augmented.extend(heal_paths.iter().cloned());
                        match run_confined_once(
                            &self.backend,
                            command,
                            script_via_stdin,
                            working_dir,
                            &augmented,
                            write_paths,
                            write_roots,
                            capabilities,
                            self.initial_timeout,
                        )
                        .await
                        {
                            Ok((retry_exit, retry_out, retry_err)) => {
                                info!(
                                    granted = ?heal_paths,
                                    prior_exit = ?exit_code,
                                    retry_exit = ?retry_exit,
                                    "Self-healed a confined sandbox denial and re-ran the command"
                                );
                                self_heal_note = format!(
                                    "\n[SYSTEM] The first attempt was denied read access to {}, which the tool needs; the daemon granted it read-only and re-ran the command automatically.",
                                    heal_paths.join(", ")
                                );
                                stdout = retry_out;
                                stderr = retry_err;
                                exit_code = retry_exit;
                            }
                            Err(error) => {
                                warn!(%error, "Self-heal retry failed to execute");
                            }
                        }
                    }
                }
                let (mut output, truncation) = format_output(
                    &stdout,
                    &stderr,
                    self.max_output_chars,
                    self.spill_session(request.notify_session_id),
                );
                push_exit_status(&mut output, exit_code);
                if !self_heal_note.is_empty() {
                    output.push_str(&self_heal_note);
                }
                if confined
                    && let Some(hint) = confined_opaque_failure_hint(exit_code, &stdout, &stderr)
                {
                    output.push_str(&hint);
                }
                // A signal-killed child reports no exit code, so the hint above
                // (which keys on a non-zero code) stays silent for what is often
                // the most confusing failure. Name the signal explicitly: it is
                // the difference between "the program chose to exit" and "the OS
                // killed it" (a sandbox denial the kernel enforces by SIGKILL, an
                // out-of-memory kill, or a crash), which point at different fixes.
                if exit_code != Some(0)
                    && let Some(signal) = terminating_signal
                {
                    output.push_str(&format!(
                            "\n[SYSTEM diagnostic] The command was terminated by signal {signal} ({}), not by exiting on its own. This is the OS killing the process, not the task failing: {}",
                            signal_name(signal),
                            signal_remedy_hint(signal),
                        ));
                }
                let mut metadata = foreground_terminal_metadata(exit_code);
                metadata.truncation = truncation;
                metadata.semantics = command_semantics.clone();
                // Type observed deletions on the receipt: a nominated path
                // that existed before the run and is absent after a
                // successful exit was destroyed by this process. This is the
                // only receipt shape that can close an executor checklist
                // obligation declared `destructive`.
                if exit_code == Some(0) {
                    let mut semantics = metadata.semantics.clone();
                    let mut observed_deletion = false;
                    for path in &deletion_preexisting {
                        let backend_path = aidaemon_tools::BackendPath::new(path.clone());
                        if self.backend.metadata(&backend_path).await.is_err() {
                            observed_deletion = true;
                            semantics = semantics.with_target_hint(
                                crate::tools::traits::ToolTargetHintKind::Path,
                                path,
                            );
                        }
                    }
                    if observed_deletion {
                        semantics.mutation_effects = semantics
                            .mutation_effects
                            .union(ToolMutationEffects::DESTRUCTIVE);
                        if !semantics.mutates_state() {
                            semantics.effect =
                                crate::tools::traits::ToolCallEffect::ObservationAndMutation;
                        }
                    }
                    // Type the URLs a fetch-only command proved it reached:
                    // its exit status vouches for every fetch, so the receipt
                    // is an observation of each URL, as written.
                    let fetched = aidaemon_tools::command_effects::proven_fetch_urls(command)
                        .unwrap_or_default();
                    for url in &fetched {
                        semantics = semantics
                            .with_target_hint(crate::tools::traits::ToolTargetHintKind::Url, url);
                    }
                    if observed_deletion || !fetched.is_empty() {
                        metadata.semantics = semantics;
                    }
                }
                metadata.access_denial =
                    sandbox_access_denial(stderr.as_ref(), exit_code, confined);
                Ok(ToolCallOutcome {
                    metadata,
                    output,
                    finish: None,
                })
            }
            Err(_) => {
                // Timeout — check if this is a daemon/background command where the
                // parent shell exited but pipes are held open by the detached child.
                // In that case the reader task will never finish naturally, so capture
                // partial output and return immediately instead of entering the
                // infinite background tracking loop.
                let daemon_hits = detect_daemonization_primitives(command);
                if !daemon_hits.is_empty() {
                    let partial_stdout = {
                        let b = stdout_buf.lock().await;
                        String::from_utf8_lossy(&b).to_string()
                    };
                    let partial_stderr = {
                        let b = stderr_buf.lock().await;
                        String::from_utf8_lossy(&b).to_string()
                    };
                    let (formatted, truncation) = format_output(
                        &partial_stdout,
                        &partial_stderr,
                        self.max_output_chars,
                        self.spill_session(request.notify_session_id),
                    );
                    reader_handle.abort();
                    let output = format!(
                        "Detached background command launched (pid={}).\n\
                         The process is running independently and is not task-owned.\n\
                         This detached daemonized process is not tracked by action=\"check\"/\"kill\".\n\n\
                         Initial output:\n{}",
                        pid, formatted
                    );
                    return Ok(ToolCallOutcome {
                        metadata: ToolCallMetadata {
                            outcome_status: Some(ToolOutcomeStatus::Backgrounded),
                            invocation_stage: crate::tools::traits::ToolInvocationStage::Dispatched,
                            access_enforcement: confined_process_access_enforcement(),
                            background_started: true,
                            detached: true,
                            timed_out: false,
                            completion_notifications_enabled: false,
                            truncation,
                            semantics: command_semantics.clone(),
                            ..ToolCallMetadata::default()
                        },
                        output,
                        finish: None,
                    });
                }

                // Non-daemon command: move process to background tracking.
                let elapsed = foreground_wait.as_secs();
                // Both streams: many services (cloudflared, ngrok, dev
                // servers) log their URL or ready line on stderr only.
                let partial_output = {
                    let tail_of = |b: &[u8]| {
                        let start = b.len().saturating_sub(PARTIAL_OUTPUT_TAIL_BYTES);
                        String::from_utf8_lossy(&b[start..]).trim().to_string()
                    };
                    let stdout = tail_of(&stdout_buf.lock().await);
                    let stderr = tail_of(&stderr_buf.lock().await);
                    match (stdout.is_empty(), stderr.is_empty()) {
                        (true, true) => String::new(),
                        (false, true) => stdout,
                        (true, false) => stderr,
                        (false, false) => format!("{stdout}\n{stderr}"),
                    }
                };
                let notify_on_completion = Arc::new(AtomicBool::new(true));
                let owner_task_id = task_id
                    .map(str::to_string)
                    .filter(|id| !id.trim().is_empty());
                let progress_contract = progress_contract_for_command(command);

                info!(
                    pid,
                    workload = progress_contract.workload.as_str(),
                    stall_multiplier = progress_contract.stall_multiplier,
                    idle_confirmations_required = progress_contract.idle_confirmations_required,
                    hard_max_runtime = progress_contract.hard_max_runtime,
                    "Attached background progress contract"
                );

                let proc = RunningProcess {
                    title: title.map(str::to_string),
                    command: command.to_string(),
                    declared_write_paths: write_paths.to_vec(),
                    semantics: command_semantics.clone(),
                    dedupe_key: Some(dedupe_key.clone()),
                    owner_task_id: owner_task_id.clone(),
                    origin_task_id: owner_task_id.clone(),
                    detached: detach,
                    started_at: Instant::now() - foreground_wait,
                    stdout_buf,
                    stderr_buf,
                    reader_handle,
                    process_handle,
                    notify_on_completion: notify_on_completion.clone(),
                    notifier_active: false,
                    notify_session_id: notify_session_id.trim().to_string(),
                    notify_goal_id: notify_goal_id.unwrap_or("").to_string(),
                    run_id: None,
                    last_progress_len: 0,
                    last_progress_at: Instant::now(),
                    last_cpu_ms: 0,
                    last_io_bytes: 0,
                    last_tree_fingerprint: 0,
                    idle_confirmations: 0,
                    progress_contract,
                };

                self.launch_trust
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(pid, launch_trust);
                self.running.lock().await.insert(pid, proc);
                self.insert_indexes_for_process(
                    pid,
                    Some(&dedupe_key),
                    owner_task_id.as_deref(),
                    detach,
                )
                .await;

                // Deterministic completion delivery: notify user when background command finishes
                // even if the agent loop ends before an explicit `action="check"` call.
                // Also re-engages the agent loop so it can process the output and continue
                // working on the original task.
                let mut notifier_started = false;
                // The name of the run this command became, when it is one.
                let mut run_label: Option<String> = None;
                let state_for_notify = self.notifications.clone();
                let hub_for_notify = self.get_hub();
                let agent_for_notify = self.agent.get().and_then(|w| w.upgrade());
                let reengagements_for_notify = self.reengagements.clone();
                let recent_background_deliveries_for_notify =
                    self.recent_background_deliveries.clone();
                // Deliver-once ledger + delivery dirs + durable plan store for
                // harness-side deliverable attribution and direct file delivery.
                let delivered_deliverables_for_notify = self.delivered_deliverables.clone();
                let declared_write_paths_for_notify = write_paths.to_vec();
                let plan_store_for_notify = self.plan_store.get().cloned();
                let inbox_dir_for_notify = self.inbox_dir.clone();
                let outbox_dirs_for_notify = self.outbox_dirs.clone();
                // Cloned for the notifier task so it can remove the finished process
                // from `running` (and its indexes) after delivery, preventing the
                // idle-reaper from sending a contradictory "stopped, no results" message
                // for a process that already delivered its output.
                let running_for_notify = self.running.clone();
                let running_by_dedupe_key_for_notify = self.running_by_dedupe_key.clone();
                let task_processes_for_notify = self.task_processes.clone();
                let dedupe_key_for_notify = dedupe_key.clone();
                let owner_task_id_for_notify = owner_task_id.clone();
                let tool_call_id_for_notify = tool_call_id.map(str::to_string);
                let event_store_for_notify = self.event_store.clone();
                let backend_for_notify = self.backend.clone();
                // The completion receipt terminates the same invocation whose
                // launch receipt declared the process-state change. The
                // launch receipt is provisional and proves nothing on its
                // own, so the terminal receipt must carry that declared
                // effect or the mutation is never credited anywhere.
                let completion_semantics_for_notify =
                    background_process_semantics(&command_semantics);
                if !notifications_reach {
                    info!(
                        pid,
                        session_id = notify_session_id,
                        "Terminal background notifier not started: a sub-agent's session has no reader"
                    );
                    notify_on_completion.store(false, Ordering::Relaxed);
                    // Nothing reports this job, but the run that started it
                    // holds its answer until it ends: its latest output line
                    // stays its progress (live 2026-11-13: a helper's card
                    // sat still for its whole command), and its end is
                    // recorded on the launch's own call, so the run's ledger
                    // knows it ended.
                    let (stdout_buf, stderr_buf) = {
                        let running = self.running.lock().await;
                        match running.get(&pid) {
                            Some(proc) => (proc.stdout_buf.clone(), proc.stderr_buf.clone()),
                            None => (Arc::default(), Arc::default()),
                        }
                    };
                    let progress = status_tx.clone();
                    let event_store = event_store_for_notify.clone();
                    let task_id = owner_task_id_for_notify.clone();
                    let call_id = tool_call_id_for_notify.clone();
                    let session = notify_session_id.trim().to_string();
                    let spill_session = self.spill_session(&session).map(str::to_string);
                    let semantics = completion_semantics_for_notify.clone();
                    let max_output_chars = self.max_output_chars;
                    let started = Instant::now() - foreground_wait;
                    tokio::spawn(async move {
                        let mut completion_rx = completion_rx;
                        let mut tick = tokio::time::interval(COMMAND_OUTPUT_PROGRESS_EVERY);
                        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                        tick.tick().await;
                        let mut shown = 0;
                        let exit_code = loop {
                            tokio::select! {
                                exit = &mut completion_rx => break exit.ok().flatten(),
                                _ = tick.tick() => {
                                    let Some(tx) = progress.as_ref() else { continue };
                                    let lines = printed_lines(
                                        &stdout_buf.lock().await,
                                        &stderr_buf.lock().await,
                                    );
                                    if lines != shown {
                                        let _ = tx.try_send(StatusUpdate::CommandOutput {
                                            name: "terminal".to_string(),
                                            lines,
                                            tail: None,
                                        });
                                        shown = lines;
                                    }
                                }
                            }
                        };
                        let (Some(event_store), Some(task_id), Some(call_id)) =
                            (event_store, task_id, call_id)
                        else {
                            return;
                        };
                        let stdout = String::from_utf8_lossy(&stdout_buf.lock().await).to_string();
                        let stderr = String::from_utf8_lossy(&stderr_buf.lock().await).to_string();
                        let (mut output, truncation) = format_output(
                            &stdout,
                            &stderr,
                            max_output_chars,
                            spill_session.as_deref(),
                        );
                        if let Some(info) = truncation.as_ref() {
                            output.push('\n');
                            output.push_str(&crate::turn::helpers::render_truncation_notice(info));
                        }
                        persist_background_completion_receipt(
                            &event_store,
                            &BackgroundCompletion {
                                session_id: &session,
                                task_id: &task_id,
                                tool_call_id: &call_id,
                                exit_code,
                                output: &output,
                                truncation,
                                stderr: &stderr,
                                semantics,
                                elapsed_secs: started.elapsed().as_secs(),
                                confined,
                            },
                        )
                        .await;
                    });
                } else if state_for_notify.is_some() || hub_for_notify.is_some() {
                    let goal_id_for_notify = notify_goal_id.unwrap_or("").to_string();
                    let session_for_notify = notify_session_id.trim().to_string();
                    let notify_spill_session =
                        self.spill_session(&session_for_notify).map(str::to_string);
                    let command_for_notify = command.to_string();
                    let stdout_for_notify = {
                        let running = self.running.lock().await;
                        running.get(&pid).map(|p| p.stdout_buf.clone())
                    };
                    let stderr_for_notify = {
                        let running = self.running.lock().await;
                        running.get(&pid).map(|p| p.stderr_buf.clone())
                    };
                    let started_at_for_notify = Instant::now() - foreground_wait;
                    let max_output_chars = self.max_output_chars;
                    // The command is a run in the runtime's records, on its
                    // turn's card like a background helper: its latest output
                    // line is its step, and its end settles the card before
                    // the result is sent. A detached service never "finishes"
                    // the work it was asked for, so it is not one.
                    let run = if detach || session_for_notify.is_empty() {
                        None
                    } else {
                        let label = crate::background_run::plain_label(
                            title,
                            &aidaemon_tools::command_effects::program_label(command),
                        );
                        let work_began = chrono::Utc::now()
                            - chrono::Duration::from_std(foreground_wait).unwrap_or_default();
                        Some(
                            crate::background_run::BackgroundRun::start(
                                self.run_home(),
                                crate::background_run::RunStart {
                                    id: &format!("command-{pid}-{}", uuid::Uuid::new_v4()),
                                    chat: &session_for_notify,
                                    label: &label,
                                    task_id: owner_task_id.as_deref(),
                                    status_tx: status_tx.clone(),
                                    work_began,
                                },
                            )
                            .await,
                        )
                    };
                    run_label = run.as_ref().map(|run| run.label().to_string());
                    if let Some(run) = run.as_ref()
                        && let Some(proc) = self.running.lock().await.get_mut(&pid)
                    {
                        proc.run_id = Some(run.id().to_string());
                    }
                    if let (Some(stdout_buf), Some(stderr_buf)) =
                        (stdout_for_notify, stderr_for_notify)
                    {
                        tokio::spawn(async move {
                            if session_for_notify.is_empty() {
                                warn!(
                                    pid,
                                    command = %LoggedCommand(&command_for_notify),
                                    "Terminal background notifier skipped enqueue due to empty session id"
                                );
                                notify_on_completion.store(false, Ordering::Relaxed);
                                return;
                            }
                            let command_summary = truncate_str(
                                &command_for_notify
                                    .split_whitespace()
                                    .collect::<Vec<_>>()
                                    .join(" "),
                                160,
                            );
                            let run = run.as_deref();
                            // Its result in the chat, after its card settled:
                            // the body, how it ended only when no card shows
                            // it, and the measured time.
                            let conclude = |phase| async move {
                                match run {
                                    Some(run) => {
                                        let (state, on_card) =
                                            run.finish(phase, Default::default()).await;
                                        Some((state, on_card))
                                    }
                                    None => None,
                                }
                            };
                            let present =
                                |ended: &Option<(Option<crate::run_state::RunState>, bool)>,
                                 body: &str| {
                                    match (run, ended) {
                                        (Some(run), Some((state, on_card))) => {
                                            run.result(state.as_ref(), body, *on_card)
                                        }
                                        _ => body.to_string(),
                                    }
                                };

                            let mut completion_rx = completion_rx;
                            let mut step_tick =
                                tokio::time::interval(COMMAND_OUTPUT_PROGRESS_EVERY);
                            step_tick
                                .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                            step_tick.tick().await;
                            let exit_code = loop {
                                tokio::select! {
                                    exit = &mut completion_rx => break match exit {
                                        Ok(code) => code,
                                        Err(e) => {
                                            warn!(
                                                pid,
                                                error = %e,
                                                command = %LoggedCommand(&command_for_notify),
                                                "Terminal background notifier lost completion signal"
                                            );
                                            None
                                        }
                                    },
                                    _ = step_tick.tick() => {
                                        // How much it has printed is its step.
                                        let Some(run) = run else { continue };
                                        if !notify_on_completion.load(Ordering::Relaxed) {
                                            continue;
                                        }
                                        let lines = printed_lines(
                                            &stdout_buf.lock().await,
                                            &stderr_buf.lock().await,
                                        );
                                        if lines > 0 {
                                            run.step(Step::Output { lines }).await;
                                        }
                                    }
                                }
                            };
                            // How it ended, with why when it did not
                            // succeed: its exit status (live 2026-11-13: a
                            // `du` that exited 1 read "Failed in the
                            // background" with no reason on its card or in
                            // its result).
                            let end = crate::run_state::RunEnd::of_exit(exit_code);
                            if !notify_on_completion.load(Ordering::Relaxed) {
                                // A check or a stop already reported its end:
                                // the run ends on its card, with no message.
                                warn!(
                                    pid,
                                    command = %LoggedCommand(&command_for_notify),
                                    "Terminal background notifier suppressed (check/kill already handled notification)"
                                );
                                if let Some(run) = run {
                                    let (state, _) =
                                        run.finish(end.clone(), Default::default()).await;
                                    // Its end still counts for its turn's
                                    // group: when it was the last, the others'
                                    // results go out now.
                                    if let Some(state) = state
                                        && let crate::background_run::RunsTurn::Answer(answer) = run
                                            .end_in_group(
                                                &state,
                                                crate::run_state::RunReport {
                                                    text: String::new(),
                                                    untrusted: false,
                                                },
                                                String::new(),
                                                max_output_chars,
                                            )
                                    {
                                        for message in answer
                                            .messages
                                            .iter()
                                            .filter(|message| !message.trim().is_empty())
                                        {
                                            deliver_background_text(
                                                hub_for_notify.as_ref(),
                                                state_for_notify.as_ref(),
                                                &session_for_notify,
                                                &goal_id_for_notify,
                                                message,
                                                pid,
                                            )
                                            .await;
                                        }
                                    }
                                }
                                return;
                            }

                            // The run ends now, the moment its process did:
                            // its card settles to its final header at once,
                            // before any result is worked out or delivered.
                            // A continuation turn waits for its slot behind
                            // other continuations and runs a whole turn, so an
                            // end recorded after it left the card on the last
                            // output line for minutes (live 2026-11-13: "last
                            // output: 8" after the loop had ended).
                            let ended = conclude(end.clone()).await;

                            let stdout =
                                String::from_utf8_lossy(&stdout_buf.lock().await).to_string();
                            let stderr =
                                String::from_utf8_lossy(&stderr_buf.lock().await).to_string();
                            // Background delivery bypasses the agent loop
                            // entirely, so the truncation notice is rendered
                            // inline here.
                            let (formatted, truncation) = format_output(
                                &stdout,
                                &stderr,
                                max_output_chars,
                                notify_spill_session.as_deref(),
                            );
                            let mut with_notice = formatted;
                            if let Some(ref info) = truncation {
                                with_notice.push('\n');
                                with_notice.push_str(
                                    &crate::turn::helpers::render_truncation_notice(info),
                                );
                            }
                            // Its end stays: a build's or a test run's verdict
                            // is its last lines (live 2026-11-15: a background
                            // `cargo test` came back cut after "Checking …",
                            // and the agent could not say whether it passed).
                            let output = keep_head_and_tail(&with_notice, 2500);
                            let elapsed_secs = started_at_for_notify.elapsed().as_secs();

                            // The background transition receipt is nonterminal.
                            // Persist a second, terminal receipt on actual process
                            // completion so recovery and continuation never have
                            // to infer success from a notification sentence.
                            let mut completion_parent_result_id: Option<String> = None;
                            if let (Some(event_store), Some(parent_task_id), Some(tool_call_id)) = (
                                event_store_for_notify.as_ref(),
                                owner_task_id_for_notify.as_ref(),
                                tool_call_id_for_notify.as_ref(),
                            ) {
                                completion_parent_result_id =
                                    persist_background_completion_receipt(
                                        event_store,
                                        &BackgroundCompletion {
                                            session_id: &session_for_notify,
                                            task_id: parent_task_id,
                                            tool_call_id,
                                            exit_code,
                                            output: &with_notice,
                                            truncation: truncation.clone(),
                                            stderr: &stderr,
                                            semantics: completion_semantics_for_notify.clone(),
                                            elapsed_secs,
                                            confined,
                                        },
                                    )
                                    .await;
                            }

                            // Deliverable attribution: a command that produced
                            // exactly one safe explicit output file has THAT file
                            // as its answer, delivered directly (no re-engagement).
                            let command_end = std::time::SystemTime::now();
                            let command_start = command_end
                                .checked_sub(started_at_for_notify.elapsed())
                                .unwrap_or(command_end);
                            let deliverable_attribution = attribute_background_deliverable(
                                &backend_for_notify,
                                &declared_write_paths_for_notify,
                                command_start,
                                command_end,
                            )
                            .await;
                            let deliverable_note = match &deliverable_attribution {
                                DeliverableAttribution::One(_) if exit_code == Some(0) => None,
                                DeliverableAttribution::One(path) => Some(format!(
                                    "⚠️ The background command finished with errors ({}) and produced `{}`. I'm not sending it automatically because the command did not complete successfully.",
                                    exit_code
                                        .map(|c| format!("exit code {c}"))
                                        .unwrap_or_else(|| "an unknown exit status".to_string()),
                                    path.file_name()
                                        .map(|f| f.to_string_lossy().to_string())
                                        .unwrap_or_else(|| path.to_string_lossy().to_string())
                                )),
                                DeliverableAttribution::Ambiguous(paths) => Some(format!(
                                    "⚠️ The background command finished, but multiple output files matched: {}. I did not auto-send a file because the deliverable is ambiguous.",
                                    format_deliverable_paths(paths)
                                )),
                                DeliverableAttribution::ExpectedMissing(paths) => Some(format!(
                                    "⚠️ The background command finished before the expected output file appeared: {}. There's nothing to send.",
                                    format_deliverable_paths(paths)
                                )),
                                DeliverableAttribution::None => None,
                            };

                            if let DeliverableAttribution::One(path) = &deliverable_attribution
                                && exit_code == Some(0)
                            {
                                info!(
                                    pid,
                                    session_id = %session_for_notify,
                                    "Attributed a produced-output file; delivering it directly (no re-engagement)"
                                );
                                deliver_attributed_background_file(
                                    &backend_for_notify,
                                    path,
                                    &session_for_notify,
                                    &command_summary,
                                    &inbox_dir_for_notify,
                                    &outbox_dirs_for_notify,
                                    &delivered_deliverables_for_notify,
                                    plan_store_for_notify.as_ref(),
                                    hub_for_notify.as_ref(),
                                    state_for_notify.as_ref(),
                                    &goal_id_for_notify,
                                    pid,
                                )
                                .await;
                            }
                            // The runs its turn left running are answered
                            // together: until the last of them ends this
                            // one's end is on the card only, and the last
                            // one's end answers the turn once from all.
                            let raw_result = format!(
                                "Output from `{}`:\n\n```\n{}\n```",
                                command_summary, output
                            );
                            let group = match (run, ended.as_ref()) {
                                (Some(run), Some((Some(state), _))) => run.end_in_group(
                                    state,
                                    crate::run_state::RunReport {
                                        text: output.clone(),
                                        untrusted: true,
                                    },
                                    present(&ended, &raw_result),
                                    max_output_chars,
                                ),
                                _ => crate::background_run::RunsTurn::Alone,
                            };
                            // Waiting for the rest of its turn's runs, or
                            // stopped by the owner (alone, or with every run of
                            // its turn): its card says so, and nothing answers
                            // the chat or picks the work up again.
                            let owner_stopped = ended
                                .as_ref()
                                .and_then(|(state, _)| state.as_ref())
                                .is_some_and(crate::run_state::RunState::stopped_by_owner);
                            let group_waiting = match &group {
                                crate::background_run::RunsTurn::Waiting
                                | crate::background_run::RunsTurn::Stopped => true,
                                crate::background_run::RunsTurn::Alone => owner_stopped,
                                crate::background_run::RunsTurn::Answer(_) => false,
                            };
                            let group_answer = match group {
                                crate::background_run::RunsTurn::Answer(answer) => Some(answer),
                                _ => None,
                            };
                            if group_waiting {
                                info!(
                                    pid,
                                    session_id = %session_for_notify,
                                    "A command's result waits for the rest of its turn's runs"
                                );
                            }
                            if let Some(note) = deliverable_note.filter(|_| !group_waiting) {
                                deliver_background_text(
                                    hub_for_notify.as_ref(),
                                    state_for_notify.as_ref(),
                                    &session_for_notify,
                                    &goal_id_for_notify,
                                    &present(&ended, &note),
                                    pid,
                                )
                                .await;
                            }
                            if matches!(deliverable_attribution, DeliverableAttribution::None)
                                && !group_waiting
                            {
                                let output_trimmed = output.trim();
                                // Only genuinely empty output is trivial. A short
                                // result is often the whole answer (a `wc -l`
                                // count, a one-word status), so it is never
                                // dropped for its length.
                                let is_trivial =
                                    output_trimmed.is_empty() || output_trimmed == "(no output)";
                                if is_trivial && agent_for_notify.is_none() {
                                    info!(pid, "No agent for empty background output");
                                } else if !is_trivial
                                    && group_answer.is_none()
                                    && is_short_complete_output(output_trimmed)
                                    && owner_task_id_for_notify.is_none()
                                {
                                    // SHORT, unowned utility result: a tool-less
                                    // interpretation (it cannot re-run anything),
                                    // else the raw result, so the answer is never
                                    // lost. Task-owned completions use the typed
                                    // continuation below.
                                    if let Some(run) = run {
                                        run.step(Step::Reporting).await;
                                    }
                                    let interpreted = match agent_for_notify {
                                        Some(ref agent) => {
                                            agent
                                                .interpret_background_result(
                                                    &command_for_notify,
                                                    output_trimmed,
                                                )
                                                .await
                                        }
                                        None => None,
                                    };
                                    let message = interpreted.unwrap_or_else(|| {
                                        format_short_background_result(output_trimmed)
                                    });
                                    // Background deliveries bypass the agent loop's
                                    // completion sanitizer.
                                    let message =
                                        aidaemon_tools::sanitize::sanitize_user_facing_reply(
                                            &message,
                                        );
                                    // A result is a duplicate by its body: the
                                    // measured footer differs per run.
                                    let delivery_allowed = {
                                        let mut log =
                                            recent_background_deliveries_for_notify.lock().await;
                                        background_delivery_allowed(
                                            &mut log,
                                            &session_for_notify,
                                            &message,
                                            Instant::now(),
                                        )
                                    };
                                    let message = present(&ended, &message);
                                    if delivery_allowed {
                                        deliver_background_text(
                                            hub_for_notify.as_ref(),
                                            state_for_notify.as_ref(),
                                            &session_for_notify,
                                            &goal_id_for_notify,
                                            &message,
                                            pid,
                                        )
                                        .await;
                                    } else {
                                        info!(
                                            pid,
                                            session_id = %session_for_notify,
                                            "Suppressed duplicate short background command output"
                                        );
                                    }
                                } else {
                                    // Feed the output back through the agent so the
                                    // owner gets a written answer; only if that path
                                    // is unavailable or yields nothing does the raw
                                    // output go out, so content is never lost.
                                    //
                                    // Re-engagement budget: past the per-session cap
                                    // the raw output goes out instead.
                                    let reengage_budget_ok = {
                                        let mut log = reengagements_for_notify.lock().await;
                                        reengagement_allowed(
                                            &mut log,
                                            &session_for_notify,
                                            &command_for_notify,
                                            Instant::now(),
                                        )
                                    };
                                    if !reengage_budget_ok {
                                        warn!(
                                            pid,
                                            session_id = %session_for_notify,
                                            command = %LoggedCommand(&command_for_notify),
                                            "Background re-engagement budget exhausted; delivering raw output instead of re-entering agent loop"
                                        );
                                    }
                                    if let Some(run) = run {
                                        run.step(Step::Reporting).await;
                                    }
                                    let mut formatted_delivered = false;
                                    if let (true, Some(agent)) =
                                        (reengage_budget_ok, agent_for_notify.as_ref())
                                    {
                                        let (followup, parent_task_id) = match group_answer.as_ref()
                                        {
                                            Some(answer) => (
                                                answer.followup.clone(),
                                                answer.parent_task_id.clone(),
                                            ),
                                            None => (
                                                build_background_reengagement_followup(
                                                    &command_summary,
                                                    &output,
                                                ),
                                                owner_task_id_for_notify.clone(),
                                            ),
                                        };
                                        info!(
                                            pid,
                                            session_id = %session_for_notify,
                                            command = %LoggedCommand(&command_for_notify),
                                            "Re-engaging agent loop to process background command output"
                                        );
                                        let (continued, _followup_end) = run_background_continuation(
                                            agent.as_ref(),
                                            ConversationRequest {
                                                session_id: session_for_notify.clone(),
                                                user_text: followup,
                                                status_tx: None,
                                                user_role: aidaemon_tools::semantics::UserRole::Owner,
                                                channel_ctx: aidaemon_channels::context::ChannelContext::internal(
                                                ),
                                                heartbeat: None,
                                                parent_task_id,
                                                parent_tool_call_id: tool_call_id_for_notify
                                                    .clone(),
                                                parent_result_id: completion_parent_result_id
                                                    .clone(),
                                                clock: match group_answer.as_ref() {
                                                    Some(answer) => answer.clock,
                                                    None => ended
                                                        .as_ref()
                                                        .and_then(|(state, _)| state.as_ref())
                                                        .and_then(crate::run_state::RequestClock::of_run),
                                                },
                                            },
                                        )
                                        .await;
                                        match continued {
                                            Ok(envelope) => {
                                                // Defense-in-depth: never leak scaffolding
                                                // the re-engaged loop may echo.
                                                let reply =
                                                    aidaemon_tools::sanitize::sanitize_user_facing_reply(
                                                        &envelope.text,
                                                    );
                                                if !reply.trim().is_empty() {
                                                    let delivery_allowed = {
                                                        let mut log =
                                                            recent_background_deliveries_for_notify
                                                                .lock()
                                                                .await;
                                                        background_delivery_allowed(
                                                            &mut log,
                                                            &session_for_notify,
                                                            &reply,
                                                            Instant::now(),
                                                        )
                                                    };
                                                    // A group's answer stands alone:
                                                    // the card shows each run's end.
                                                    let reply = if group_answer.is_some() {
                                                        reply
                                                    } else {
                                                        present(&ended, &reply)
                                                    };
                                                    if !delivery_allowed {
                                                        formatted_delivered = true;
                                                        let _ = agent
                                                            .record_continuation_delivery(
                                                                &session_for_notify,
                                                                envelope.delivery(
                                                                    "background_router",
                                                                    aidaemon_events::ResponseDeliveryState::Failed,
                                                                    Vec::new(),
                                                                    Some("duplicate_suppressed".to_string()),
                                                                ),
                                                            )
                                                            .await;
                                                        info!(
                                                            pid,
                                                            session_id = %session_for_notify,
                                                            "Suppressed duplicate agent follow-up for background command"
                                                        );
                                                    } else if let Some(ref hub) = hub_for_notify {
                                                        let _ = agent
                                                            .record_continuation_delivery(
                                                                &session_for_notify,
                                                                envelope.delivery(
                                                                    "background_router",
                                                                    aidaemon_events::ResponseDeliveryState::Queued,
                                                                    Vec::new(),
                                                                    None,
                                                                ),
                                                            )
                                                            .await;
                                                        match hub
                                                            .send_text_tracked(
                                                                &session_for_notify,
                                                                &reply,
                                                            )
                                                            .await
                                                        {
                                                            Ok(platform_id) => {
                                                                formatted_delivered = true;
                                                                let _ = agent
                                                                    .record_continuation_delivery(
                                                                        &session_for_notify,
                                                                        envelope.delivery(
                                                                            "background_router",
                                                                            aidaemon_events::ResponseDeliveryState::PlatformAcknowledged,
                                                                            platform_id.into_iter().collect(),
                                                                            None,
                                                                        ),
                                                                    )
                                                                    .await;
                                                            }
                                                            Err(e) => {
                                                                let _ = agent
                                                                    .record_continuation_delivery(
                                                                        &session_for_notify,
                                                                        envelope.delivery(
                                                                            "background_router",
                                                                            aidaemon_events::ResponseDeliveryState::Failed,
                                                                            Vec::new(),
                                                                            Some("transport_error".to_string()),
                                                                        ),
                                                                    )
                                                                    .await;
                                                                warn!(
                                                                    pid,
                                                                    error = %e,
                                                                    "Failed to deliver agent follow-up for background command"
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    "Agent re-engagement failed for background command"
                                                );
                                            }
                                        }
                                    }

                                    // Fallback: no written answer reached the chat,
                                    // so the raw output goes out rather than nothing.
                                    if !formatted_delivered {
                                        let fallback = if let Some(answer) = group_answer.as_ref() {
                                            answer.messages.join("\n\n")
                                        } else if is_trivial {
                                            format_background_continuation_failure(&[])
                                        } else {
                                            format!(
                                                "Output from `{}`:\n\n```\n{}\n```",
                                                command_summary, output
                                            )
                                        };
                                        let delivery_allowed = {
                                            let mut log = recent_background_deliveries_for_notify
                                                .lock()
                                                .await;
                                            background_delivery_allowed(
                                                &mut log,
                                                &session_for_notify,
                                                &fallback,
                                                Instant::now(),
                                            )
                                        };
                                        let fallback = if group_answer.is_some() {
                                            fallback
                                        } else {
                                            present(&ended, &fallback)
                                        };
                                        if delivery_allowed {
                                            deliver_background_text(
                                                hub_for_notify.as_ref(),
                                                state_for_notify.as_ref(),
                                                &session_for_notify,
                                                &goal_id_for_notify,
                                                &fallback,
                                                pid,
                                            )
                                            .await;
                                        } else {
                                            info!(
                                                pid,
                                                session_id = %session_for_notify,
                                                "Suppressed duplicate fallback background command output"
                                            );
                                        }
                                    }
                                }
                            }
                            // However the result went, the run has ended.
                            if let Some(run) = run {
                                run.finish(end, Default::default()).await;
                            }
                            // Remove the finished process from `running` (and
                            // its dedupe / task-process indexes), so the idle
                            // reaper never reports a process whose output was
                            // already delivered. check/kill return early above
                            // and own their cleanup.
                            if let Some(reaped) = running_for_notify.lock().await.remove(&pid) {
                                if let Some(ref key) = reaped.dedupe_key
                                    && key == &dedupe_key_for_notify
                                {
                                    let mut dedupe = running_by_dedupe_key_for_notify.lock().await;
                                    if dedupe.get(key).copied() == Some(pid) {
                                        dedupe.remove(key);
                                    }
                                }
                                if !reaped.detached
                                    && let Some(ref task_id) = owner_task_id_for_notify
                                {
                                    let mut task_map = task_processes_for_notify.lock().await;
                                    let mut remove_task_key = false;
                                    if let Some(pids) = task_map.get_mut(task_id) {
                                        pids.remove(&pid);
                                        remove_task_key = pids.is_empty();
                                    }
                                    if remove_task_key {
                                        task_map.remove(task_id);
                                    }
                                }
                                info!(
                                    pid,
                                    "Notifier removed finished process from running map after delivery"
                                );
                            }
                        });
                        notifier_started = true;
                        // Mark the process so cleanup_task_processes knows the notifier
                        // is actively monitoring it and will deliver the result.
                        if let Some(proc) = self.running.lock().await.get_mut(&pid) {
                            proc.notifier_active = true;
                        }
                    } else {
                        warn!(
                            pid,
                            command = %LoggedCommand(command),
                            "Terminal background notifier not started because process buffers were unavailable"
                        );
                        notify_on_completion.store(false, Ordering::Relaxed);
                    }
                } else {
                    warn!(
                        pid,
                        command = %LoggedCommand(command),
                        "Terminal background notifier disabled: neither state queue nor channel hub is configured"
                    );
                    notify_on_completion.store(false, Ordering::Relaxed);
                }

                let mut msg = if detach {
                    format!(
                        "Detached launch: the process passed its {:.1}s startup check and is running in the background (pid={}).\n\
                         It keeps running until killed; do not launch it again. If its startup output below does not yet show what you need (a URL, port, or ready line), wait briefly and use action=\"check\" with pid={} to read it. Use action=\"kill\" with pid={} to stop it.",
                        foreground_wait.as_secs_f32(),
                        pid,
                        pid,
                        pid
                    )
                } else if notifier_started {
                    // The first line is the turn's reply when it hands off.
                    let lead = run_label
                        .as_deref()
                        .map(|label| {
                            format!("{}\n", crate::background_run::started_in_background(label))
                        })
                        .unwrap_or_default();
                    format!(
                        "{lead}Command still running after {}s. Moved to background (pid={}).\n\
                         IMPORTANT: Continue with your next steps immediately — do NOT wait or repeatedly check this process.\n\
                         You can run other commands (like curl) while this runs in the background.\n\
                         Use action=\"check\" with pid={} to see output later, or action=\"kill\" with pid={} to stop it.",
                        elapsed, pid, pid, pid
                    )
                } else {
                    // Nothing will report this job's result: the run holds
                    // its answer until the job ends, then wakes it to read
                    // the result (`sdk_wait_for_unreported_jobs`).
                    format!(
                        "Command still running after {}s. Moved to background (pid={}).\n\
                         You can run other commands (like curl) while this runs in the background.\n\
                         Its result comes back to this run, not as a separate message. Do not wait for it yourself: \
                         no sleep, no loop polling its pid, no repeated checks. When you have nothing else to do, give your answer; \
                         the run holds it until this command ends, then wakes you to read its output and result with action=\"check\" with pid={}. \
                         To stop it instead, use action=\"kill\" with pid={}.",
                        elapsed, pid, pid, pid
                    )
                };
                if detach {
                    msg.push_str(
                        "\n\nDetached mode is enabled: this process will not be auto-killed at task end.",
                    );
                    // The notifier also watches detached processes and
                    // delivers their result; say so, or the model tells the
                    // user it cannot notify them.
                    if notifier_started {
                        msg.push_str(
                            " Completion notifications are enabled. The user will be notified when this process finishes.",
                        );
                    }
                } else if notifier_started {
                    msg.push_str(
                        "\n\nCompletion notifications are enabled. The user will be notified when this process finishes.",
                    );
                } else {
                    msg.push_str(
                        "\n\nThis process belongs to this task: if the task is stopped, the process is stopped with it.",
                    );
                }
                if !partial_output.is_empty() {
                    msg.push_str(&format!("\n\nPartial output so far:\n{}", partial_output));
                }
                let background_semantics = background_process_semantics(&command_semantics);
                let outcome = ToolCallOutcome {
                    metadata: ToolCallMetadata {
                        outcome_status: Some(ToolOutcomeStatus::Backgrounded),
                        background_started: true,
                        timed_out: !detach,
                        detached: detach,
                        completion_notifications_enabled: notifier_started,
                        semantics: background_semantics,
                        background_run_labels: run_label
                            .clone()
                            .filter(|_| notifier_started)
                            .into_iter()
                            .collect(),
                        ..ToolCallMetadata::default()
                    },
                    output: msg,
                    finish: None,
                };
                // Work the daemon moved to background tracking hands the run
                // off only when the notifier will report its result: a
                // task-owned job without one dies at task end, so the run
                // keeps going and reads the result itself. A launch the model
                // detached on purpose is a running service whose startup
                // output the model reports.
                Ok(if notifier_started && !detach {
                    outcome.handed_to_background()
                } else {
                    outcome
                })
            }
        }
    }
}
