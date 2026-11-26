//! Queuing background CLI agent launches for a working-directory claim or a
//! concurrency slot, refreshing direct-agent sessions after waiting, and
//! settling unstarted runs.

use super::*;

/// Safety fallback wake interval for a folder waiter if a notification is ever
/// missed; normal wakeups are immediate via [`RunHandles::claims_changed`].
const FOLDER_POLL_FALLBACK: Duration = Duration::from_secs(5);

/// What a run waiting for its folder reads on its row.
fn waiting_for_folder(agent: &str) -> String {
    format!("Waiting for {agent} to finish in this folder")
}

fn lock_folder_wait(on: &FolderWaitOn) -> std::sync::MutexGuard<'_, Option<String>> {
    on.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl CliAgentTool {
    /// Accept a background launch that cannot start yet: every slot is
    /// taken (`slot` is its place in line), or another run holds its folder
    /// ([`PreparedRun::folder_wait`]; `slot` is `None`). It is a run at
    /// once: its task id and its record, reading what it waits for. A
    /// folder waiter takes its claim once nothing it would collide with
    /// holds the folder (first come, first served among colliding waiters),
    /// then waits for a slot as any queued launch does; a slot waiter already
    /// holds its claim. It starts by itself, first come first served, when a
    /// slot frees, and runs from then on as an immediate launch does.
    pub(super) async fn queue_run(
        &self,
        mut call: RunCall,
        mut prep: PreparedRun,
        slot: Option<SlotWait>,
    ) -> String {
        let task_id = call.task_id.clone();
        let run_id = cli_run_id(&task_id);
        let shared = RunShared::default();
        shared.starts_in_background();
        self.start_background_run(
            &shared.run_slot,
            &task_id,
            &call.session_id,
            &call.label,
            call.delegated_task_id.as_deref(),
            call.status_tx.clone(),
            call.began,
        )
        .await;
        record_background(&self.state, &call).await;
        let handles = self.run_handles();
        handles.runs.set_agent(&run_id, prep.coding_agent.name());
        handles.runs.asked(&run_id, run_ask(&prep, &call));
        if let Some(earlier) = &prep.continues {
            handles
                .runs
                .continues(&run_id, &earlier.run, &earlier.thread);
        }
        handles.runs.keeps_session(&run_id, prep.keeps_session);
        let folder_wait = prep.folder_wait.take();
        let thread_wait = prep.thread_wait.take();
        if let Some(tw) = &thread_wait {
            handles.runs.set_thread(&run_id, &tw.thread);
        }
        let names: HashMap<String, String> = self
            .tools
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(tool, entry)| (tool.clone(), self.entry_agent(entry).name().to_string()))
            .collect();
        let display =
            move |tool: &str| names.get(tool).cloned().unwrap_or_else(|| tool.to_string());
        let behind = folder_wait.as_ref().map(|wait| display(&wait.behind));
        *lock_folder_wait(&shared.folder_wait) = behind.clone();
        let initial_activity = if let Some(tw) = &thread_wait {
            format!("Waiting for {} to finish", tw.behind_agent)
        } else {
            behind
                .as_deref()
                .map_or_else(|| WAITING_FOR_SLOT.to_string(), waiting_for_folder)
        };
        handles.runs.activity(&run_id, initial_activity);
        if let Some(ref tx) = call.status_tx {
            let _ = tx.try_send(StatusUpdate::ToolCancellable {
                name: call.tool_name.clone(),
                task_id: task_id.clone(),
            });
        }
        let ahead = {
            let mut running = self.running.lock().await;
            let ahead = running
                .values()
                .filter(|agent| agent.process.is_queued() && agent.waits_for_folder().is_none())
                .count();
            running.insert(
                task_id.clone(),
                tracked(
                    &call,
                    &shared,
                    Instant::now(),
                    prep.claim_dir.clone(),
                    prep.write_scope.clone(),
                    prep.max_output,
                ),
            );
            ahead
        };
        // The reaper / cancel paths own the slot waiter's claim now.
        if let Some(guard) = prep.claim.as_mut() {
            guard.disarm();
        }
        info!(
            task_id = %task_id,
            tool = %call.tool_name,
            ahead,
            waits_for_folder = behind.is_some(),
            "A CLI agent waits to start"
        );

        let label = call.label.clone();
        let tool_name = call.tool_name.clone();
        let dir = folder_wait.as_ref().map(|wait| wait.dir.clone());
        let thread_behind = thread_wait.as_ref().map(|tw| tw.behind_agent.clone());
        let waited_for_folder = folder_wait.is_some() || thread_wait.is_some();
        let mut shown = behind.clone();
        let limiter = self.concurrency_limiter.clone();
        let running = self.running.clone();
        let claims = self.working_dir_claims.clone();
        let claims_changed = self.claims_changed.clone();
        let claim_dir_for_cleanup = prep.claim_dir.clone();
        tokio::spawn(async move {
            let mut folder_wait = folder_wait;
            if let Some(tw) = thread_wait {
                // Wait until the targeted thread's active run ends, then
                // refresh to resume that exact thread's session.
                let behind_task_id = cli_agent_task_id(&tw.behind_run).map(str::to_string);
                loop {
                    if shared.process.was_stopped() {
                        end_unstarted(
                            &handles,
                            &call,
                            &shared,
                            claim_dir_for_cleanup.as_deref(),
                            crate::run_state::RunEnd::Stopped(None),
                            &format!(
                                "CLI agent '{}' was stopped while waiting for {} to finish; it \
                                 never started.",
                                call.tool_name, tw.behind_agent
                            ),
                            false,
                            prep.max_output,
                        )
                        .await;
                        return;
                    }
                    let done_notify = if let Some(ref tid) = behind_task_id {
                        let table = running.lock().await;
                        match table.get(tid) {
                            Some(agent) if !agent.finished.load(Ordering::Acquire) => {
                                Some(agent.done_notify.clone())
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let still_running_in_runs = handles
                        .runs
                        .get(&tw.behind_run)
                        .is_some_and(|state| state.ended_at.is_none());
                    let durable_done = done_notify.is_none()
                        && handles
                            .state
                            .cli_agent_run(&tw.behind_run)
                            .await
                            .ok()
                            .flatten()
                            .is_some_and(|rec| rec.output_summary.is_some());
                    if done_notify.is_none() && (!still_running_in_runs || durable_done) {
                        break;
                    }
                    if let Some(notify) = done_notify {
                        let notified = notify.notified();
                        tokio::pin!(notified);
                        notified.as_mut().enable();
                        tokio::select! {
                            biased;
                            () = shared.process.stopped() => {}
                            () = notified => {}
                            () = tokio::time::sleep(FOLDER_POLL_FALLBACK) => {}
                        }
                    } else {
                        tokio::select! {
                            biased;
                            () = shared.process.stopped() => {}
                            () = tokio::time::sleep(Duration::from_millis(100)) => {}
                        }
                    }
                }
                if let Err(why) =
                    refresh_thread_session(&handles, &running, &call, &mut prep, &tw.thread).await
                {
                    end_unstarted(
                        &handles,
                        &call,
                        &shared,
                        claim_dir_for_cleanup.as_deref(),
                        crate::run_state::RunEnd::Failed(crate::run_state::RunFailure::Errored {
                            error: why.clone(),
                        }),
                        &why,
                        true,
                        prep.max_output,
                    )
                    .await;
                    return;
                }
                // Now acquire this run's folder claim (waiting if another run
                // took the folder in the meantime).
                if let Some(dir) = prep.claim_dir.clone()
                    && prep.claim.is_none()
                {
                    folder_wait = Some(FolderWait {
                        dir,
                        writes: prep.write_scope.clone(),
                        prompt_summary: call.short_summary.clone(),
                        behind: tw.behind_agent.clone(),
                    });
                }
            }
            if let Some(wait) = folder_wait {
                loop {
                    // Register notification BEFORE checking claims so no
                    // release between the check and the select can be missed.
                    let notified = claims_changed.notified();
                    tokio::pin!(notified);
                    notified.as_mut().enable();
                    if shared.process.was_stopped() {
                        end_unstarted(
                            &handles,
                            &call,
                            &shared,
                            claim_dir_for_cleanup.as_deref(),
                            crate::run_state::RunEnd::Stopped(None),
                            &format!(
                                "CLI agent '{}' was stopped while it waited for its folder; it \
                                 never started.",
                                call.tool_name
                            ),
                            false,
                            prep.max_output,
                        )
                        .await;
                        return;
                    }
                    match try_take_folder(
                        &running,
                        &claims,
                        &claims_changed,
                        &call.task_id,
                        &call.tool_name,
                        &wait,
                    )
                    .await
                    {
                        FolderTurn::Taken(guard) => {
                            prep.claim = Some(guard);
                            *lock_folder_wait(&shared.folder_wait) = None;
                            break;
                        }
                        FolderTurn::Waits(Some(holder)) => {
                            let holder = display(&holder);
                            if shown.as_deref() != Some(holder.as_str()) {
                                handles.runs.activity(
                                    &cli_run_id(&call.task_id),
                                    waiting_for_folder(&holder),
                                );
                                *lock_folder_wait(&shared.folder_wait) = Some(holder.clone());
                                shown = Some(holder);
                            }
                        }
                        FolderTurn::Waits(None) => {}
                    }
                    tokio::select! {
                        biased;
                        () = shared.process.stopped() => {}
                        () = notified => {}
                        () = tokio::time::sleep(FOLDER_POLL_FALLBACK) => {}
                    }
                }
                info!(task_id = %call.task_id, tool = %call.tool_name, "A CLI agent's folder is free");
                if call.direct_agent && !call.fresh_session && prep.continues.is_none() {
                    refresh_direct_session(&handles, &running, &call, &mut prep).await;
                }
            }
            let mut slot = slot.unwrap_or_else(|| Box::pin(limiter.clone().acquire_owned()));
            let waited = match futures::poll!(slot.as_mut()) {
                std::task::Poll::Ready(permit) => Some(permit),
                std::task::Poll::Pending => {
                    if waited_for_folder {
                        handles
                            .runs
                            .activity(&cli_run_id(&call.task_id), WAITING_FOR_SLOT.to_string());
                    }
                    tokio::select! {
                        biased;
                        () = shared.process.stopped() => None,
                        permit = slot => Some(permit),
                    }
                }
            };
            let permit = match waited {
                Some(Ok(permit)) if !shared.process.was_stopped() => permit,
                Some(Err(error)) => {
                    end_unstarted(
                        &handles,
                        &call,
                        &shared,
                        claim_dir_for_cleanup.as_deref(),
                        crate::run_state::RunEnd::Failed(crate::run_state::RunFailure::Errored {
                            error: error.to_string(),
                        }),
                        &format!(
                            "CLI agent '{}' could not get a slot: {error}",
                            call.tool_name
                        ),
                        true,
                        prep.max_output,
                    )
                    .await;
                    return;
                }
                Some(Ok(_)) | None => {
                    end_unstarted(
                        &handles,
                        &call,
                        &shared,
                        claim_dir_for_cleanup.as_deref(),
                        crate::run_state::RunEnd::Stopped(None),
                        &format!(
                            "CLI agent '{}' was stopped while it waited for a free slot; it \
                             never started.",
                            call.tool_name
                        ),
                        false,
                        prep.max_output,
                    )
                    .await;
                    return;
                }
            };
            info!(task_id = %call.task_id, tool = %call.tool_name, "A queued CLI agent got a free slot");
            handles.runs.clear_activity(&cli_run_id(&call.task_id));
            super::run::adopt_conversation_tree(&mut call, &prep);
            let started = match handles.start_process(&call, &mut prep, &shared).await {
                Ok(started) => started,
                Err(error) => {
                    end_unstarted(
                        &handles,
                        &call,
                        &shared,
                        claim_dir_for_cleanup.as_deref(),
                        crate::run_state::RunEnd::Failed(crate::run_state::RunFailure::Errored {
                            error: error.to_string(),
                        }),
                        &format!("Failed to spawn CLI agent '{}': {error}", call.tool_name),
                        true,
                        prep.max_output,
                    )
                    .await;
                    return;
                }
            };
            if !shared
                .process
                .start(started.handle.clone(), started.host.clone())
            {
                kill_process(&handles.backend, &started.handle).await;
            } else if started
                .host
                .as_ref()
                .is_some_and(|host| host.accepts_mid_run())
            {
                handles
                    .runs
                    .set_accepts_mid_run(&cli_run_id(&call.task_id), true);
            }
            // A folder waiter's claim: the reaper / cancel paths own it now.
            if let Some(guard) = prep.claim.as_mut() {
                guard.disarm();
            }
            let (completion_tx, _completion_rx) = tokio::sync::oneshot::channel();
            let coding_agent = prep.coding_agent.clone();
            let keep_alive = prep.take_keep_alive();
            let max_output = prep.max_output;
            let background_timeout = background_cap(handles.background_timeout, prep.timeout);
            let idle_timeout = handles.idle_timeout;
            let permit_slot = peers::PermitSlot::new(
                permit,
                limiter.clone(),
                shared.finished.clone(),
                shared.in_flight_approvals.clone(),
            );
            drop(prep);
            watch_run(
                handles,
                call,
                shared,
                started,
                coding_agent,
                max_output,
                background_timeout,
                idle_timeout,
                keep_alive,
                permit_slot,
                completion_tx,
            )
            .await;
        });

        mark_background_run(&label);
        if let Some(behind_agent) = thread_behind {
            return format!(
                "{}\nCLI agent '{tool_name}' is queued (task_id={task_id}): waiting for \
                 {behind_agent} to finish its current turn before continuing that session. It \
                 starts by itself then, so do not start it again.",
                crate::background_run::started_in_background(&label),
            );
        }
        if let (Some(behind), Some(dir)) = (behind, dir) {
            return format!(
                "{}\nCLI agent '{tool_name}' is queued (task_id={task_id}): {behind} is already \
                 working in {dir}, so this run waits for it to finish there. It starts by itself \
                 then, so do not start it again. Use action=\"check\" with task_id=\"{task_id}\" \
                 to see output when done, or action=\"cancel\" to drop it.",
                crate::background_run::started_in_background(&label),
            );
        }
        let place = if ahead == 0 {
            "it is next in line".to_string()
        } else {
            format!("{ahead} queued ahead of it")
        };
        format!(
            "{}\nCLI agent '{tool_name}' is queued (task_id={task_id}): all {} agent slots are taken \
             and {place}. It starts by itself when a slot frees, so do not start it again. \
             Use action=\"check\" with task_id=\"{task_id}\" to see output when done, or \
             action=\"cancel\" to drop it.",
            crate::background_run::started_in_background(&label),
            self.max_concurrent,
        )
    }
}

/// If a run that waited for its folder in a direct coding-agent conversation
/// finds a newer finished resumable session once the folder frees, re-prepare
/// its launch so it continues that session instead of starting cold or forking.
pub(super) async fn refresh_direct_session(
    handles: &RunHandles,
    running: &Mutex<HashMap<String, RunningCliAgent>>,
    call: &RunCall,
    prep: &mut PreparedRun,
) {
    let Ok(Some(run)) = handles
        .state
        .latest_kept_cli_agent_run(&call.session_id, &call.tool_name)
        .await
    else {
        return;
    };
    if prep
        .continues
        .as_ref()
        .is_some_and(|earlier| earlier.run == run)
    {
        return;
    }
    let Ok(Some(found)) = handles.state.cli_agent_run(&run).await else {
        return;
    };
    if found.session_id != call.session_id {
        return;
    }
    // Check whether the earlier run has actually finished: its process may
    // have exited and persisted durable completion (`output_summary.is_some()`
    // / `finished == true`) while `deliver_end` is still waiting to stamp
    // `RunState::ended_at`. Only bail if it is genuinely still working.
    let still_working_in_runs = handles
        .runs
        .get(&run)
        .is_some_and(|state| state.ended_at.is_none());
    if still_working_in_runs && found.output_summary.is_none() {
        let finished_in_table = if let Some(id) = cli_agent_task_id(&run) {
            running
                .lock()
                .await
                .get(id)
                .is_some_and(|agent| agent.finished.load(Ordering::Acquire))
        } else {
            false
        };
        if !finished_in_table {
            return;
        }
    }
    let Some(agent_session) = found.agent_session.filter(|_| found.keeps_session) else {
        return;
    };
    let mut launch = prep.base_launch.clone();
    let plan = match prep.coding_agent.prepare(
        &aidaemon_coding_agents::RunRequest {
            mode: prep.mode,
            enforce: prep.enforce_native_access,
            read_only: prep.workspace_mode == CliWorkspaceMode::ReadOnly,
            working_dir: prep.working_dir.as_deref(),
            read_paths: &prep.read_paths,
            write_paths: &prep.write_paths,
            can_host: prep.can_host,
            stdin_open: handles.backend.kind() == aidaemon_tools::backend::BackendKind::Local,
            model: prep.model.as_deref(),
            resume: Some(agent_session.as_str()),
            report: !call.direct_agent,
            mcp_servers: &prep.mcp_servers,
        },
        &mut launch,
    ) {
        Ok(plan) => plan,
        Err(why) => {
            info!(
                task_id = %call.task_id,
                session_id = %call.session_id,
                earlier_run = %run,
                %why,
                "Queued coding agent run could not continue the newly finished session; keeping prepared launch"
            );
            return;
        }
    };
    let thread = found.thread.unwrap_or_else(|| run.clone());
    let continues = crate::helper_run::Continues {
        run: run.clone(),
        agent_session,
        thread: thread.clone(),
    };
    info!(
        task_id = %call.task_id,
        session_id = %call.session_id,
        earlier_run = %run,
        "Queued coding agent run continuing the session that finished in its folder"
    );
    prep.args = launch.args;
    prep.prompt_placement = plan.prompt;
    prep.env = plan.env;
    prep.env_remove = plan.env_remove;
    // Its sign-in, as its first launch had it.
    prep.args.extend(prep.signed.args.iter().cloned());
    prep.env.extend(prep.signed.env.iter().cloned());
    prep.env_remove
        .extend(prep.signed.env_remove.iter().cloned());
    prep.host = plan.host;
    prep.keep_alive = plan.keep_alive;
    prep.keeps_session = plan.keeps_session;
    prep.continues = Some(continues);
    let run_id = cli_run_id(&call.task_id);
    handles.runs.continues(&run_id, &run, &thread);
    handles.runs.keeps_session(&run_id, prep.keeps_session);
}

/// After waiting for a specific thread's active run to finish, resolve the
/// latest run in `thread`, verify it kept its session, and re-prepare `prep`
/// to resume that exact session. Returns `Err` if the session cannot be
/// continued so the queued run ends cleanly instead of forking a cold session.
async fn refresh_thread_session(
    handles: &RunHandles,
    running: &Mutex<HashMap<String, RunningCliAgent>>,
    call: &RunCall,
    prep: &mut PreparedRun,
    thread: &str,
) -> Result<(), String> {
    let run = match handles
        .state
        .latest_cli_agent_run_in_thread(&call.session_id, thread)
        .await
    {
        Ok(Some(run)) => run,
        Ok(None) => thread.to_string(),
        Err(error) => {
            return Err(format!(
                "Could not read thread `{thread}` after its run ended: {error}"
            ));
        }
    };
    let found = handles
        .state
        .cli_agent_run(&run)
        .await
        .map_err(|error| format!("Could not read run `{run}`: {error}"))?
        .filter(|found| found.session_id == call.session_id)
        .ok_or_else(|| format!("This conversation has no coding agent run `{run}`"))?;
    let still_working_in_runs = handles
        .runs
        .get(&run)
        .is_some_and(|state| state.ended_at.is_none());
    if still_working_in_runs && found.output_summary.is_none() {
        let finished_in_table = if let Some(id) = cli_agent_task_id(&run) {
            running
                .lock()
                .await
                .get(id)
                .is_some_and(|agent| agent.finished.load(Ordering::Acquire))
        } else {
            false
        };
        if !finished_in_table {
            return Err(format!(
                "Run `{run}` is still working; cannot continue its session yet"
            ));
        }
    }
    let Some(agent_session) = found.agent_session.filter(|_| found.keeps_session) else {
        return Err(format!(
            "{} didn't keep run `{run}`'s session, so the queued follow-up couldn't continue it.",
            found.agent_name
        ));
    };
    let mut launch = prep.base_launch.clone();
    let plan = prep
        .coding_agent
        .prepare(
            &aidaemon_coding_agents::RunRequest {
                mode: prep.mode,
                enforce: prep.enforce_native_access,
                read_only: prep.workspace_mode == CliWorkspaceMode::ReadOnly,
                working_dir: prep.working_dir.as_deref(),
                read_paths: &prep.read_paths,
                write_paths: &prep.write_paths,
                can_host: prep.can_host,
                stdin_open: handles.backend.kind() == aidaemon_tools::backend::BackendKind::Local,
                model: prep.model.as_deref(),
                resume: Some(agent_session.as_str()),
                report: !call.direct_agent,
                mcp_servers: &prep.mcp_servers,
            },
            &mut launch,
        )
        .map_err(|why| format!("{} could not continue its session: {why}", found.agent_name))?;
    let canonical_thread = found.thread.unwrap_or_else(|| thread.to_string());
    let continues = crate::helper_run::Continues {
        run: run.clone(),
        agent_session,
        thread: canonical_thread.clone(),
    };
    prep.args = launch.args;
    prep.prompt_placement = plan.prompt;
    prep.env = plan.env;
    prep.env_remove = plan.env_remove;
    prep.args.extend(prep.signed.args.iter().cloned());
    prep.env.extend(prep.signed.env.iter().cloned());
    prep.env_remove
        .extend(prep.signed.env_remove.iter().cloned());
    prep.host = plan.host;
    prep.keep_alive = plan.keep_alive;
    prep.keeps_session = plan.keeps_session;
    prep.continues = Some(continues);
    let run_id = cli_run_id(&call.task_id);
    handles.runs.continues(&run_id, &run, &canonical_thread);
    handles.runs.keeps_session(&run_id, prep.keeps_session);
    Ok(())
}

/// End a queued run that never started: stopped while it waited, or its
/// process could not be started once a slot freed. `report` is what its
/// result reads.
#[allow(clippy::too_many_arguments)]
async fn end_unstarted(
    handles: &RunHandles,
    call: &RunCall,
    shared: &RunShared,
    claim_dir: Option<&str>,
    phase: crate::run_state::RunEnd,
    report: &str,
    answer_alone: bool,
    max_output: usize,
) {
    let evidence = crate::run_state::RunEvidence::default();
    let _ = shared.end_record.set(CliRunEnd {
        end: phase.clone(),
        evidence: evidence.clone(),
        changed_something: false,
    });
    if let Some(dir) = claim_dir {
        release_claim(&mut lock_claims(&handles.claims), dir, &call.task_id);
        handles.claims_changed.notify_waiters();
    }
    crate::helper_run::folders::release_new_folders(&call.new_folders);
    append_bounded_line(&mut *shared.stdout_buf.lock().await, "", report, BUFFER_CAP);
    append_bounded_line(
        &mut *shared.display_buf.lock().await,
        "",
        report,
        BUFFER_CAP,
    );
    forget_background(&handles.state, call).await;
    let _ = shared.verdict.set(RunVerdict {
        success: false,
        text: report.to_string(),
    });
    // Finished before its chat hears of it, as a run that ran is.
    shared.finished.store(true, Ordering::Release);
    shared.done_notify.notify_waiters();
    handles.claims_changed.notify_waiters();
    deliver_end(
        handles,
        call,
        shared,
        EndDelivery {
            phase: &phase,
            evidence: &evidence,
            success: false,
            report,
            persisted_output: report,
            left: &left_running::LeftRunning::default(),
            diff_dir: None,
            max_output,
            answer_alone,
        },
    )
    .await;
    settle_run(handles, call, shared, phase, evidence);
}
