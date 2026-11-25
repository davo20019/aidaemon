//! Watching a started CLI agent process to completion: output streaming,
//! guardrails, loop & timeout enforcement, single-pass change receipts,
//! coordination commit, durable completion persistence, and settlement.

use super::*;

/// Watch a started run to its end: stream its output (emitting progress
/// and its background step), judge how it ended by the rule every helper
/// run is judged by, record it, deliver a background run's end to its chat
/// ([`deliver_end`]) and signal `completion_tx`.
#[allow(clippy::too_many_arguments)]
pub(super) async fn watch_run(
    handles: RunHandles,
    call: RunCall,
    shared: RunShared,
    started: StartedRun,
    coding_agent: Arc<dyn aidaemon_coding_agents::CodingAgent>,
    max_output: usize,
    background_timeout: Duration,
    idle_timeout: Duration,
    keep_alive: Option<Box<dyn std::any::Any + Send>>,
    permit_slot: peers::PermitSlot,
    completion_tx: tokio::sync::oneshot::Sender<(Option<i32>, bool, Option<usize>)>,
) {
    if let Some(binding) = call.peer_binding.as_ref() {
        binding.register_permit(permit_slot.clone());
    }
    // What the agent's launch needs until the run is over.
    let _keep_alive = keep_alive;
    let StartedRun {
        mut child,
        stdout,
        stderr,
        handle: process_handle_for_task,
        host: agent_host_for_task,
        events: mut agent_events,
        run_tag: run_tag_for_task,
        invocation_id,
        started_at: invocation_started_at,
        watched: watched_for_task,
        sent_to_write,
        snapshot_before,
        diff_dir: notify_working_dir,
        coordination,
    } = started;
    let pid_for_kill = process_handle_for_task.display_id();
    let backend_for_task = handles.backend.clone();
    let backend_for_receipt = handles.backend.clone();
    let stdout_buf_writer = shared.stdout_buf.clone();
    let display_buf_writer = shared.display_buf.clone();
    let status_tx_clone = call.status_tx.clone();
    let tool_name_owned = call.tool_name.clone();
    let run_for_task = shared.run_slot.clone();
    let task_id_for_notify = call.task_id.clone();
    let left_running_for_task = shared.left_running.clone();
    // A run given paths to write was sent to change them, so ending with
    // nothing changed is ending short. The owner's own follow-up and
    // direct-agent conversations are given their folder as leave to write,
    // not as a requirement to change files: the owner may only ask a question.
    let expects_change = sent_to_write && !call.owner_follow_up && !call.direct_agent;
    let auth_failed_for_task = shared.auth_failed.clone();
    let task_context = call.label.clone();
    let coding_agent_for_task = coding_agent;
    let end_record_for_task = shared.end_record.clone();
    let grant_session_for_task = call.grant_session.clone();
    let folders_for_task = call.new_folders.clone();
    let state_for_completion = handles.state.clone();
    let max_output_for_log = max_output;
    let delegated_task_id_for_completion = call.delegated_task_id.clone();
    let finished_for_task = shared.finished.clone();

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();
    let mut last_progress = Instant::now();
    let mut last_output_at = Instant::now();
    let mut last_tick_at = Instant::now();
    let mut active_elapsed = Duration::ZERO;
    let mut status_tick = tokio::time::interval(PROGRESS_INTERVAL);
    status_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    status_tick.tick().await; // consume the immediate first tick
    let started_at = Instant::now();
    let mut pending_lines: Vec<String> = Vec::new();
    let mut pending_typed_progress: Vec<String> = Vec::new();
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut last_parsed_action: Option<String> = None;
    let mut printed_lines: u64 = 0;
    let mut loop_detector = LoopDetector::new();
    let mut loop_detected = false;
    let mut loop_pattern_count: Option<usize> = None;
    let mut timed_out_reason: Option<String> = None;

    loop {
        if stdout_done && stderr_done {
            break;
        }

        if loop_detected {
            info!(
                pid = pid_for_kill,
                pattern_count = ?loop_pattern_count,
                "Infinite loop detected in CLI agent output, killing process"
            );
            kill_process(&backend_for_task, &process_handle_for_task).await;
            break;
        }

        tokio::select! {
            line = stdout_reader.next_line(), if !stdout_done => {
                match line {
                    Ok(Some(text)) => {
                        last_output_at = Instant::now();
                        // Guardrail: kill external coding agents if they attempt to
                        // manipulate aidaemon's state DB (covers Claude, Codex, ACP, generic).
                        if looks_like_json(&text) {
                            let mut blocked: Option<(String, &'static str)> = None;
                            for cmd in extract_terminal_commands_from_json(&text) {
                                if let Some(reason) = prohibited_cli_agent_command_reason(&cmd) {
                                    blocked = Some((cmd, reason));
                                    break;
                                }
                            }

                            if let Some((cmd, reason)) = blocked {
                                info!(
                                    pid = pid_for_kill,
                                    cmd = %cmd,
                                    "Prohibited CLI agent command detected; killing process"
                                );
                                {
                                    let mut buf = display_buf_writer.lock().await;
                                    append_bounded_line(
                                        &mut buf,
                                        "[killed] Prohibited CLI agent command: ",
                                        &format!("{cmd}\nReason: {reason}"),
                                        BUFFER_CAP,
                                    );
                                }
                                kill_process(
                                    &backend_for_task,
                                    &process_handle_for_task,
                                )
                                .await;
                                break;
                            }
                        }

                        if loop_detector.add_line(&text) && !loop_detected {
                            loop_detected = true;
                            loop_pattern_count = loop_detector.get_loop_pattern();
                        }

                        let typed_progress = agent_events.read(&text).await;
                        let had_typed = !typed_progress.is_empty();
                        pending_typed_progress.extend(typed_progress);
                        {
                            let mut buf = stdout_buf_writer.lock().await;
                            append_bounded_line(&mut buf, "", &text, BUFFER_CAP);
                        }
                        {
                            let mut buf = display_buf_writer.lock().await;
                            append_bounded_line(&mut buf, "", &text, BUFFER_CAP);
                        }
                        if !had_typed {
                            pending_lines.push(text);
                        } else {
                            printed_lines += 1;
                        }
                    }
                    _ => stdout_done = true,
                }
            }
            line = stderr_reader.next_line(), if !stderr_done => {
                match line {
                    Ok(Some(text)) => {
                        last_output_at = Instant::now();
                        if loop_detector.add_line(&text) && !loop_detected {
                            loop_detected = true;
                            loop_pattern_count = loop_detector.get_loop_pattern();
                        }

                        let mut buf = display_buf_writer.lock().await;
                        append_bounded_line(&mut buf, "[stderr] ", &text, BUFFER_CAP);
                        pending_lines.push(format!("[stderr] {}", text));
                    }
                    _ => stderr_done = true,
                }
            }
            _ = status_tick.tick() => {
                let now = Instant::now();
                let tick_delta = now.saturating_duration_since(last_tick_at);
                last_tick_at = now;
                let awaiting_approval = shared.in_flight_approvals.load(Ordering::Acquire) > 0;
                if awaiting_approval {
                    // Do not penalize a run while the owner is deciding on an
                    // approval card.
                    last_output_at = now;
                } else {
                    active_elapsed = active_elapsed.saturating_add(tick_delta);
                    let reason = if active_elapsed >= background_timeout {
                        Some(format!(
                            "CLI agent '{}' exceeded its {}s background execution limit and was stopped.",
                            tool_name_owned,
                            background_timeout.as_secs()
                        ))
                    } else if now.saturating_duration_since(last_output_at) >= idle_timeout {
                        Some(format!(
                            "CLI agent '{}' produced no output for {}s and was stopped.",
                            tool_name_owned,
                            idle_timeout.as_secs()
                        ))
                    } else {
                        None
                    };
                    if let Some(reason) = reason {
                        warn!(
                            task_id = %task_id_for_notify,
                            pid = pid_for_kill,
                            tool = %tool_name_owned,
                            %reason,
                            "Background CLI agent timed out; stopping process"
                        );
                        {
                            let mut buf = display_buf_writer.lock().await;
                            append_bounded_line(&mut buf, "[killed] ", &reason, BUFFER_CAP);
                        }
                        if let Some(host) = agent_host_for_task.as_ref() {
                            host.interrupt().await;
                            tokio::time::sleep(Duration::from_millis(250)).await;
                        }
                        kill_process(&backend_for_task, &process_handle_for_task).await;
                        timed_out_reason = Some(reason);
                        break;
                    }
                }
            }
        }

        if last_progress.elapsed() >= PROGRESS_INTERVAL {
            let mut progress_items: Vec<String> = std::mem::take(&mut pending_typed_progress);
            if let Some(last) = progress_items.last() {
                last_parsed_action = Some(last.clone());
            }
            for line in &pending_lines {
                if looks_like_json(line) {
                    if let Some(progress) = extract_progress_from_json(line) {
                        progress_items.push(progress.clone());
                        last_parsed_action = Some(progress);
                    }
                } else {
                    progress_items.push(line.clone());
                }
            }

            let elapsed_secs = started_at.elapsed().as_secs();
            printed_lines += pending_lines.len() as u64;
            let chunk = if !progress_items.is_empty() {
                progress_items.dedup();
                progress_items.join("\n")
            } else if let Some(ref action) = last_parsed_action {
                format!("⏳ {} ({}s)", action, elapsed_secs)
            } else {
                format!("⏳ Working... ({}s)", elapsed_secs)
            };

            if let Some(ref tx) = status_tx_clone {
                let _ = tx.try_send(StatusUpdate::ToolProgress {
                    name: tool_name_owned.clone(),
                    chunk: truncate_with_note(&chunk, 500),
                });
            }

            if let Some(run) = run_for_task
                .get()
                .cloned()
                .filter(|_| printed_lines > 0 && !agent_events.structured())
            {
                let lines = printed_lines;
                tokio::spawn(async move { run.step(Step::Output { lines }).await });
            }

            pending_lines.clear();
            last_progress = Instant::now();
        }
    }

    // Send any remaining progress lines
    if (!pending_lines.is_empty() || !pending_typed_progress.is_empty())
        && let Some(ref tx) = status_tx_clone
    {
        let mut progress_items: Vec<String> = std::mem::take(&mut pending_typed_progress);
        for line in &pending_lines {
            if looks_like_json(line) {
                if let Some(progress) = extract_progress_from_json(line) {
                    progress_items.push(progress);
                }
            } else {
                progress_items.push(line.clone());
            }
        }
        if !progress_items.is_empty() {
            progress_items.dedup();
            let chunk = progress_items.join("\n");
            let _ = tx.try_send(StatusUpdate::ToolProgress {
                name: tool_name_owned.clone(),
                chunk: truncate_with_note(&chunk, 500),
            });
        }
    }

    let timed_out = timed_out_reason.is_some();
    let exit_code = if loop_detected || timed_out {
        None
    } else {
        match child.wait().await {
            Ok(status) => status.code(),
            Err(_) => None,
        }
    };
    let exit_code = match (exit_code, agent_host_for_task.as_ref()) {
        (Some(0), Some(host)) if host.unfinished() => Some(1),
        (code, _) => code,
    };
    let left = match process_handle_for_task.local_pid() {
        Some(agent_pid) => {
            left_running::find_left_running(run_tag_for_task.clone(), Some(agent_pid)).await
        }
        None => left_running::LeftRunning::default(),
    };
    if !left.is_empty() {
        info!(
            task_id = %task_id_for_notify,
            run_tag = %run_tag_for_task,
            count = left.total,
            pids = ?left.processes.iter().map(|p| p.pid).collect::<Vec<_>>(),
            "CLI agent left processes running after its run ended"
        );
    }
    let _ = left_running_for_task.set(left.clone());

    // Single-pass post-run snapshot for both aggregate change receipt and
    // per-file changed paths.
    let snapshot_after = snapshot_roots(&backend_for_receipt, &watched_for_task).await;
    let changed_something = snapshot_after.receipt != snapshot_before.receipt;
    let changed_files = changed_files_from(&snapshot_before.marks, &snapshot_after.marks);
    for path in &changed_files {
        handles.runs.edited(&cli_run_id(&call.task_id), path);
    }
    if let Some(coordination) = coordination {
        coordination
            .finish(&backend_for_receipt, &changed_files)
            .await;
    }

    {
        let run_id = cli_run_id(&call.task_id);
        let edited = handles
            .runs
            .get(&run_id)
            .map(|state| state.edited)
            .unwrap_or_default();
        let fallback = extract_meaningful_output(&stdout_buf_writer.lock().await, 4_000);
        handles.runs.result(
            &run_id,
            agent_events.result(&edited, &watched_for_task, &fallback),
        );
    }
    let auth_failed = cli_auth_failed(
        &stdout_buf_writer.lock().await,
        &display_buf_writer.lock().await,
        exit_code,
    );
    auth_failed_for_task.store(auth_failed, Ordering::Release);
    if auth_failed {
        crate::tools::agent_mode::mark_not_usable(
            &tool_name_owned,
            "It couldn't sign in on its last run.",
            None,
        );
    }
    let (run_phase, run_evidence) =
        crate::helper_run::program_phase(&crate::helper_run::ProgramEnd {
            exit_code,
            stopped: loop_detected || auth_failed,
            timed_out,
            expects_change,
            changed_something,
            step: task_context.clone(),
            refused: coding_agent_for_task
                .refused_actions(&stdout_buf_writer.lock().await)
                .into_iter()
                .chain(
                    agent_host_for_task
                        .as_ref()
                        .map(|host| host.answers().refused)
                        .unwrap_or_default(),
                )
                .collect(),
            declined: agent_host_for_task
                .as_ref()
                .map(|host| host.answers().declined)
                .unwrap_or_default(),
        });
    let _ = end_record_for_task.set(CliRunEnd {
        end: run_phase.clone(),
        evidence: run_evidence.clone(),
        changed_something,
    });
    {
        let status = if loop_detected || timed_out || auth_failed || exit_code.is_none() {
            aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent
        } else if exit_code == Some(0) {
            aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded
        } else {
            aidaemon_tools::semantics::ToolOutcomeStatus::CompletedWithNegativeResult
        };
        if aidaemon_tools::local_sites::set_up_locally(
            status,
            &changed_locally_semantics(changed_something),
        ) {
            let said = format!(
                "{}\n{}",
                stdout_buf_writer.lock().await,
                display_buf_writer.lock().await
            );
            aidaemon_tools::local_sites::note_sites_in(&grant_session_for_task, &said).await;
        }
    }
    if run_phase.phase() != crate::run_state::RunPhase::Succeeded {
        crate::helper_run::folders::release_new_folders(&folders_for_task);
    }
    forget_background(&handles.state, &call).await;
    {
        let duration = invocation_started_at.elapsed().as_secs_f64();
        let completion = if loop_detected {
            CliCompletionResult {
                success: false,
                authentication_failed: false,
                persisted_output: "Killed - infinite loop detected".to_string(),
                response: None,
                error: Some("Killed - infinite loop detected".to_string()),
            }
        } else if let Some(ref reason) = timed_out_reason {
            CliCompletionResult {
                success: false,
                authentication_failed: false,
                persisted_output: reason.clone(),
                response: None,
                error: Some(reason.clone()),
            }
        } else {
            let stdout_text = stdout_buf_writer.lock().await.clone();
            let display_text = display_buf_writer.lock().await.clone();
            let mut completion = CliAgentTool::classify_completion_result(
                &tool_name_owned,
                exit_code,
                &stdout_text,
                &display_text,
                max_output_for_log,
            );
            if completion.success
                && let Some(reported) = agent_events.reported_text()
            {
                let reported = truncate_with_note(&reported, max_output_for_log);
                completion.persisted_output = reported.clone();
                completion.response = Some(reported);
            }
            completion
        };
        let success =
            completion.success && run_phase.phase() == crate::run_state::RunPhase::Succeeded;
        if completion.authentication_failed {
            CliAgentTool::quarantine_unavailable_tool(
                &handles.tools,
                &handles.tool_names,
                &tool_name_owned,
                "authentication failed",
            );
        }
        let agent_report = completion.persisted_output;
        let mut persisted_output = match left.result_section() {
            Some(section) => format!("{agent_report}\n\n{section}"),
            None => agent_report.clone(),
        };
        // A lead CLI's own report summarizes its peers; the tree's record
        // (every peer and board post) is what the caller can quote.
        if let Some(record) = call
            .peer_binding
            .as_ref()
            .and_then(peers::ActivePeerBinding::lead_record)
        {
            persisted_output.push_str("\n\n");
            persisted_output.push_str(&record);
        }
        let (structured_response, structured_error) = if success {
            (completion.response, completion.error)
        } else {
            let why = run_phase.line(&run_evidence).unwrap_or_default();
            (
                None,
                Some(
                    completion
                        .error
                        .unwrap_or_else(|| format!("{why}\n\n{persisted_output}")),
                ),
            )
        };

        if invocation_id != 0 {
            let _ = state_for_completion
                .log_cli_agent_complete(
                    invocation_id,
                    exit_code,
                    &persisted_output,
                    success,
                    duration,
                )
                .await;
        }

        if let Some(ref delegated_task_id) = delegated_task_id_for_completion {
            CliAgentTool::persist_delegated_cli_result_with_state(
                state_for_completion.clone(),
                delegated_task_id,
                structured_response.as_deref(),
                structured_error.as_deref(),
            )
            .await;
        }

        let _ = shared.verdict.set(RunVerdict {
            success,
            text: match &structured_error {
                Some(error) => error.clone(),
                None => persisted_output.clone(),
            },
        });

        // Durable state is authoritative for completion. Publish the
        // finished flag and wake waiters before optional chat delivery.
        finished_for_task.store(true, Ordering::Release);
        if let Some(binding) = call.peer_binding.as_ref() {
            binding.unregister_permit();
            let peer_state = if success {
                aidaemon_coding_agents::peer::PeerState::Completed
            } else {
                aidaemon_coding_agents::peer::PeerState::Failed
            };
            let resumable = state_for_completion
                .cli_agent_run(&cli_run_id(&call.task_id))
                .await
                .ok()
                .flatten()
                .is_some_and(|r| r.keeps_session && r.agent_session.is_some());
            binding.tree.update_peer(
                &binding.peer_id,
                aidaemon_coding_agents::peer::PeerUpdate {
                    state: peer_state,
                    run_id: Some(cli_run_id(&call.task_id)),
                    working_dir: None,
                    branch: binding.branch.clone(),
                    output: Some(match &structured_error {
                        Some(error) => error.clone(),
                        None => persisted_output.clone(),
                    }),
                    resumable: Some(resumable),
                },
            );
            binding.cleanup_children().await;
        }
        permit_slot.release();
        shared.done_notify.notify_waiters();
        handles.claims_changed.notify_waiters();

        let ends_by = shared.ends_by();
        if ends_by == EndsBy::Moved {
            shared.moved.notified().await;
        }
        if ends_by != EndsBy::Caller {
            deliver_end(
                &handles,
                &call,
                &shared,
                EndDelivery {
                    phase: &run_phase,
                    evidence: &run_evidence,
                    success,
                    report: &agent_report,
                    persisted_output: &persisted_output,
                    left: &left,
                    diff_dir: notify_working_dir.as_ref(),
                    max_output: max_output_for_log,
                    answer_alone: !call.is_peer_run,
                },
            )
            .await;
        }
    }
    settle_run(&handles, &call, &shared, run_phase, run_evidence);
    let _ = completion_tx.send((exit_code, loop_detected, loop_pattern_count));
}

/// However a run ended, its record ends with it (a second end is a no-op),
/// and it reads as finished to `check`, the reaper and a stop waiting on it.
pub(super) fn settle_run(
    handles: &RunHandles,
    call: &RunCall,
    shared: &RunShared,
    phase: crate::run_state::RunEnd,
    evidence: crate::run_state::RunEvidence,
) {
    if let Some(run) = shared.run_slot.get().cloned() {
        let claims_changed = handles.claims_changed.clone();
        tokio::spawn(async move {
            run.finish(phase, evidence).await;
            claims_changed.notify_waiters();
        });
    } else {
        handles.runs.finish(
            &cli_run_id(&call.task_id),
            phase,
            crate::run_state::RunMeasure {
                ended_at: chrono::Utc::now(),
                work_began: None,
                waited: std::time::Duration::ZERO,
            },
            evidence,
            crate::run_state::RunReport {
                text: String::new(),
                untrusted: false,
            },
        );
    }
    shared.finished.store(true, Ordering::Release);
    shared.done_notify.notify_waiters();
    handles.claims_changed.notify_waiters();
}
