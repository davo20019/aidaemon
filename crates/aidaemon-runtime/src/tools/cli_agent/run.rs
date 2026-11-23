//! `action="run"`: launching an agent and waiting on or backgrounding it.
//!
//! A launch is made ready first ([`CliAgentTool::prepare_run`]: its agent,
//! arguments, prompt and working-dir claim), its process is started once it
//! holds a slot ([`RunHandles::start_process`]) and watched to its end
//! ([`watch_run`]). A background launch that finds every slot taken waits in
//! line for one ([`CliAgentTool::queue_run`]) and starts by itself.

use super::*;

pub(super) async fn kill_process(backend: &SharedExecutionBackend, handle: &ProcessHandle) {
    let _ = backend.terminate(handle, Duration::from_secs(2)).await;
}

impl CliAgentTool {
    /// The run's failure message once [`cli_auth_failed`] judged the run
    /// an authentication failure.
    pub(super) fn auth_error_message(tool_name: &str) -> String {
        format!(
            "ERROR: CLI agent '{}' authentication failed before it produced a task result. It has been disabled for this daemon session; retry with another configured CLI agent, or authenticate '{}' before restarting it.",
            tool_name, tool_name,
        )
    }

    pub(super) fn quarantine_unavailable_tool(
        tools: &Arc<std::sync::RwLock<HashMap<String, CliToolEntry>>>,
        tool_names: &Arc<std::sync::RwLock<Vec<String>>>,
        tool_name: &str,
        reason: &'static str,
    ) {
        tools
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(tool_name);
        tool_names
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|name| name != tool_name);
        warn!(
            tool = tool_name,
            reason, "Quarantined unavailable CLI agent"
        );
    }

    pub(super) fn classify_completion_result(
        tool_name: &str,
        exit_code: Option<i32>,
        stdout: &str,
        display_output: &str,
        max_output: usize,
    ) -> CliCompletionResult {
        if cli_auth_failed(stdout, display_output, exit_code) {
            let error = Self::auth_error_message(tool_name);
            return CliCompletionResult {
                success: false,
                authentication_failed: true,
                persisted_output: error.clone(),
                response: None,
                error: Some(error),
            };
        }

        if exit_code == Some(0) {
            let result = extract_meaningful_output(stdout, max_output);
            return CliCompletionResult {
                success: true,
                authentication_failed: false,
                persisted_output: result.clone(),
                response: Some(result),
                error: None,
            };
        }

        CliCompletionResult {
            success: false,
            authentication_failed: false,
            persisted_output: display_output.to_string(),
            response: None,
            error: Some(display_output.to_string()),
        }
    }
}

/// What a queued run's row reads until a slot frees.
pub(super) const WAITING_FOR_SLOT: &str = "Waiting for a free slot";

/// A launch made ready up to its process start ([`CliAgentTool::prepare_run`]):
/// its agent and arguments in their mode, its prompt and its working-dir
/// claim.
pub(super) struct PreparedRun {
    command: String,
    args: Vec<String>,
    mode: aidaemon_coding_agents::PermissionMode,
    coding_agent: Arc<dyn aidaemon_coding_agents::CodingAgent>,
    nested: Option<aidaemon_core::NestedCalls>,
    timeout: Duration,
    max_output: usize,
    workspace_mode: CliWorkspaceMode,
    /// Its working directory, normalized: the one it claimed.
    working_dir: Option<String>,
    write_paths: Vec<String>,
    prompt_placement: aidaemon_coding_agents::PromptPlacement,
    env: Vec<(String, String)>,
    env_remove: Vec<String>,
    host: Option<Box<dyn aidaemon_coding_agents::HostStarter>>,
    keep_alive: Option<Box<dyn std::any::Any + Send>>,
    /// The agent keeps its session where a follow-up can continue it.
    keeps_session: bool,
    /// The earlier run it continues.
    continues: Option<crate::helper_run::Continues>,
    /// The prompt with its mission rules, before any enrichment.
    prompt: String,
    /// What it was asked, without the daemon's mission rules.
    task: String,
    final_prompt: String,
    /// Released on drop until the run is tracked in the task table, whose
    /// reaper and cancel paths then own it.
    claim: Option<WorkingDirClaimGuard>,
}

/// Who launched a run and where it reports.
#[derive(Clone)]
pub(super) struct RunCall {
    tool_name: String,
    task_id: String,
    session_id: String,
    goal_id: Option<String>,
    delegated_task_id: Option<String>,
    /// What it was sent to do, as the owner and the parent read it: its
    /// label, never a cut prompt (live 2026-11-13: "Delegated step: …
    /// /Users/alice/projects/claude-wordpre" made the verifier judge the
    /// next call against a folder that does not exist).
    label: String,
    short_summary: String,
    async_mode: bool,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    grant_session: String,
    new_folders: Vec<String>,
    /// When the call accepted it: its run record's start.
    began: chrono::DateTime<chrono::Utc>,
}

/// What a run shares with its entry in the tool's task table.
#[derive(Clone, Default)]
pub(super) struct RunShared {
    /// Combined output for display (with `[stderr]` prefixes).
    display_buf: Arc<Mutex<String>>,
    /// Pure stdout, for JSON extraction.
    stdout_buf: Arc<Mutex<String>>,
    finished: Arc<AtomicBool>,
    auth_failed: Arc<AtomicBool>,
    left_running: Arc<OnceLock<left_running::LeftRunning>>,
    end_record: Arc<OnceLock<CliRunEnd>>,
    /// Its background run, once it is one (see `start_background_run`).
    run_slot: Arc<OnceLock<Arc<BackgroundRun>>>,
    /// Whether its end is delivered to its chat: a background run's is,
    /// and a foreground one's once it moved to the background.
    should_notify: Arc<AtomicBool>,
    diff_dir: Arc<OnceLock<String>>,
    process: Arc<RunProcess>,
}

/// The tool's handles a run keeps after its call returned.
#[derive(Clone)]
pub(super) struct RunHandles {
    backend: SharedExecutionBackend,
    state: Arc<dyn CliAgentStore>,
    tools: Arc<std::sync::RwLock<HashMap<String, CliToolEntry>>>,
    tool_names: Arc<std::sync::RwLock<Vec<String>>>,
    pub(super) runs: Arc<crate::run_state::RunStates>,
    hub: Option<Arc<dyn OutboundRouter>>,
    agent: Option<Arc<dyn ConversationRuntime>>,
    reengagements: Arc<Mutex<HashMap<String, VecDeque<Instant>>>>,
}

/// A run's process once it started ([`RunHandles::start_process`]).
pub(super) struct StartedRun {
    child: tokio::process::Child,
    stdout: Box<dyn tokio::io::AsyncRead + Send + Unpin>,
    stderr: tokio::process::ChildStderr,
    handle: ProcessHandle,
    host: Option<Arc<dyn aidaemon_coding_agents::AgentHost>>,
    events: crate::tools::cli_agent_events::AgentEventRecorder,
    run_tag: String,
    invocation_id: i64,
    started_at: Instant,
    /// Its write roots and what they held when it started.
    watched: Vec<String>,
    receipt_before: TreeReceipt,
    /// Its file write roots' marks when it started ([`file_marks`]).
    files_before: Vec<FileMark>,
    diff_dir: Option<String>,
}

/// The task table's entry for a run.
fn tracked(
    call: &RunCall,
    shared: &RunShared,
    accepted_at: Instant,
    claim_dir: Option<String>,
) -> RunningCliAgent {
    RunningCliAgent {
        tool_name: call.tool_name.clone(),
        label: call.label.clone(),
        prompt_summary: call.short_summary.clone(),
        accepted_at,
        display_buf: shared.display_buf.clone(),
        stdout_buf: shared.stdout_buf.clone(),
        process: shared.process.clone(),
        finished: shared.finished.clone(),
        auth_failed: shared.auth_failed.clone(),
        session_id: call.session_id.clone(),
        delegated_task_id: call.delegated_task_id.clone(),
        diff_dir: shared.diff_dir.clone(),
        claim_dir,
        left_running: shared.left_running.clone(),
    }
}

/// Keep a background run on record until its process ends, so a restart
/// that cuts it off is reported (or, if it never started, started again)
/// at the next boot: `resume_turns` in the daemon.
async fn record_background(state: &Arc<dyn CliAgentStore>, call: &RunCall) {
    if call.session_id.trim().is_empty() {
        return;
    }
    let record = aidaemon_work::goals::BackgroundSpawn {
        id: cli_run_id(&call.task_id),
        session_id: call.session_id.clone(),
        mission: call.label.chars().take(BACKGROUND_MISSION_CHARS).collect(),
        started_at: chrono::Utc::now().to_rfc3339(),
    };
    if let Err(error) = state.record_background_spawn(&record).await {
        warn!(%error, task_id = %call.task_id, "Could not record a background CLI agent");
    }
}

/// Most characters of a background run's label kept on its record.
const BACKGROUND_MISSION_CHARS: usize = 200;

/// The run's process ended (or it never will): a restart from now on has
/// nothing of it to report.
async fn forget_background(state: &Arc<dyn CliAgentStore>, call: &RunCall) {
    if let Err(error) = state
        .finish_background_spawn(&cli_run_id(&call.task_id))
        .await
    {
        warn!(%error, task_id = %call.task_id, "Could not clear a background CLI agent's record");
    }
}

impl CliAgentTool {
    pub(super) fn run_handles(&self) -> RunHandles {
        RunHandles {
            backend: self.backend.clone(),
            state: self.state.clone(),
            tools: self.tools.clone(),
            tool_names: self.tool_names.clone(),
            runs: self.runs.get_or_init(Default::default).clone(),
            hub: self.get_hub(),
            agent: self.get_agent(),
            reengagements: self.reengagements.clone(),
        }
    }

    /// Run a CLI agent with streaming output. A background launch that
    /// finds every slot taken waits in line for one ([`Self::queue_run`]);
    /// a foreground one is refused.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_run(
        &self,
        tool_name: &str,
        prompt: &str,
        working_dir: Option<&str>,
        session_id: &str,
        goal_id: Option<&str>,
        delegated_task_id: Option<&str>,
        system_instruction: Option<&str>,
        workspace_mode: CliWorkspaceMode,
        read_paths: &[String],
        write_paths: &[String],
        enforce_native_access: bool,
        async_mode: bool,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        spec: &crate::helper_run::HelperSpec,
        grant_session: &str,
        project: Option<&aidaemon_core::ProjectRef>,
        nested: Option<aidaemon_core::NestedCalls>,
    ) -> anyhow::Result<String> {
        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        let mut prep = match self
            .prepare_run(
                tool_name,
                &task_id,
                prompt,
                working_dir,
                session_id,
                delegated_task_id,
                system_instruction,
                workspace_mode,
                read_paths,
                write_paths,
                enforce_native_access,
                spec,
                project,
                nested,
            )
            .await?
        {
            Ok(prep) => prep,
            Err(message) => return Ok(message),
        };
        let call = RunCall {
            tool_name: tool_name.to_string(),
            task_id: task_id.clone(),
            session_id: session_id.to_string(),
            goal_id: goal_id.map(str::to_string),
            delegated_task_id: delegated_task_id.map(str::to_string),
            label: spec.label(&format!("{tool_name} coding agent")),
            short_summary: prompt.chars().take(50).collect(),
            async_mode,
            status_tx: status_tx.clone(),
            grant_session: grant_session.to_string(),
            new_folders: spec.new_folders.clone(),
            began: chrono::Utc::now(),
        };

        // Slots go first come, first served: polled once here, a launch
        // that finds every slot taken is in line for the next one from now
        // on (a background one keeps its place; a foreground one is refused
        // and leaves the line).
        let mut slot = Box::pin(self.concurrency_limiter.clone().acquire_owned());
        let slot_permit = match futures::poll!(slot.as_mut()) {
            std::task::Poll::Ready(Ok(permit)) => permit,
            std::task::Poll::Pending if async_mode => {
                return Ok(self.queue_run(call, prep, slot).await);
            }
            _ => {
                // prep's claim guard drops on return and releases the claim.
                let running: Vec<String> = self
                    .running
                    .lock()
                    .await
                    .iter()
                    .map(|(id, agent)| {
                        let queued = if agent.process.is_queued() {
                            ", queued"
                        } else {
                            ""
                        };
                        format!(
                            "{} ({}{queued}, task_id={id})",
                            agent.label, agent.tool_name
                        )
                    })
                    .collect();
                let message = format!(
                    "Maximum {} CLI agents already running: {}. This one was NOT started and will not start by itself. \
                     Start it again once one of them ends (you are told when your own background runs end), \
                     or cancel one with action='cancel'.",
                    self.max_concurrent,
                    running.join("; ")
                );
                if let Some(task_id) = delegated_task_id {
                    Self::persist_delegated_cli_result_with_state(
                        self.state.clone(),
                        task_id,
                        None,
                        Some(&message),
                    )
                    .await;
                }
                return Ok(message);
            }
        };
        drop(slot);

        let handles = self.run_handles();
        let shared = RunShared::default();
        shared.should_notify.store(async_mode, Ordering::Relaxed);
        // prep's claim guard drops on an error return and releases the
        // claim.
        let started = handles.start_process(&call, &mut prep, &shared).await?;
        shared.process.start(started.handle.clone());
        let started_at_instant = started.started_at;
        let run_label = call.label.clone();
        let run_began = call.began;
        let timeout = prep.timeout;
        let max_output = prep.max_output;

        // Create completion channel - includes loop detection info
        // Result: (exit_code, was_killed_for_loop, loop_repetition_count)
        let (completion_tx, completion_rx) =
            tokio::sync::oneshot::channel::<(Option<i32>, bool, Option<usize>)>();

        // A run its turn waits on has the same record as a background one,
        // as a waited-on sub-agent has, so every surface (the app's run
        // rows, a question about running work) reads it the same way. If it
        // is moved to the background later, its background run takes the
        // record over under the same id.
        if async_mode {
            self.start_background_run(
                &shared.run_slot,
                &task_id,
                session_id,
                &run_label,
                delegated_task_id,
                status_tx.clone(),
                run_began,
            )
            .await;
            record_background(&self.state, &call).await;
        } else if !session_id.trim().is_empty() {
            handles.runs.begin_waited(
                &cli_run_id(&task_id),
                session_id,
                &run_label,
                run_began,
                delegated_task_id,
            );
        }
        // Which agent it is, for every surface that shows the run.
        handles
            .runs
            .set_agent(&cli_run_id(&task_id), prep.coding_agent.name());
        if let Some(earlier) = &prep.continues {
            handles
                .runs
                .continues(&cli_run_id(&task_id), &earlier.run, &earlier.thread);
        }
        handles
            .runs
            .keeps_session(&cli_run_id(&task_id), prep.keeps_session);

        // Read its output, emit progress, judge and deliver its end, and
        // signal completion. `should_notify` starts as async_mode: a
        // foreground run that times out and moves to the background flips
        // it so its end is still delivered.
        tokio::spawn(watch_run(
            handles,
            call.clone(),
            shared.clone(),
            started,
            prep.coding_agent.clone(),
            max_output,
            prep.keep_alive.take(),
            slot_permit,
            completion_tx,
        ));

        // For async_mode, return immediately with task_id
        if async_mode {
            let agent = tracked(&call, &shared, started_at_instant, prep.working_dir.clone());
            self.running.lock().await.insert(task_id.clone(), agent);
            // The reaper / cancel paths own the claim now (no await between
            // the insert above and this disarm, so no cancellation window).
            if let Some(guard) = prep.claim.as_mut() {
                guard.disarm();
            }

            mark_background_run(&run_label);
            return Ok(format!(
                "{}\nCLI agent '{}' started in background (task_id={}). \
                 Use action=\"check\" with task_id=\"{}\" to see output when done.",
                crate::background_run::started_in_background(&run_label),
                tool_name,
                task_id,
                task_id
            ));
        }

        // Wait for completion with timeout
        let display_buf = shared.display_buf.clone();
        let stdout_buf = shared.stdout_buf.clone();
        let working_dir_owned = shared.diff_dir.get().cloned();
        let result = tokio::time::timeout(timeout, completion_rx).await;

        match result {
            Ok(Ok((exit_code, was_loop_killed, loop_count))) => {
                // Sync completion path: release the claim eagerly (the rest
                // of this arm still awaits diff capture and persistence).
                drop(prep.claim.take());

                // What the agent left running: a line for the owner now, a
                // section in the result for the model.
                let left = shared.left_running.get().cloned().unwrap_or_default();
                if let (Some(line), Some(tx)) = (left.owner_line(), status_tx.as_ref()) {
                    let _ = tx.try_send(StatusUpdate::ToolProgress {
                        name: tool_name.to_string(),
                        chunk: line,
                    });
                }
                let with_left = |text: String| match left.result_section() {
                    Some(section) => format!("{text}\n\n{section}"),
                    None => text,
                };

                // Check if killed due to infinite loop
                if was_loop_killed {
                    let display_output = display_buf.lock().await.clone();
                    let last_lines: String = display_output
                        .lines()
                        .rev()
                        .take(10)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join("\n");

                    // Emit error status
                    if let Some(ref tx) = status_tx {
                        let (label, summary) = aidaemon_tools::sanitize::user_facing_tool_activity(
                            tool_name,
                            "killed - infinite loop detected",
                            &aidaemon_tools::semantics::ToolDispatchFacts::default(),
                            false,
                        );
                        let _ = tx.try_send(StatusUpdate::tool_complete(label, summary));
                    }

                    return Ok(with_left(format!(
                        "ERROR: CLI agent '{}' was automatically killed - INFINITE LOOP DETECTED.\n\n\
                         The same output line repeated {} times in the last 100 lines.\n\
                         This is a known bug in some CLI agent versions where they get stuck.\n\n\
                         Last 10 lines before kill:\n{}\n\n\
                         Do NOT retry with the same agent. Try a different approach or use a different tool.",
                        tool_name,
                        loop_count.unwrap_or(0),
                        last_lines
                    )));
                }

                // Authentication/setup failures are failures even when a CLI
                // exits zero (Gemini CLI has emitted its auth-help URL and a
                // successful process status without executing the prompt).
                // Classify this before emitting ToolComplete so the live
                // status cannot briefly report a false success.
                let display_output = display_buf.lock().await.clone();
                let stdout_output = stdout_buf.lock().await.clone();
                if cli_auth_failed(&stdout_output, &display_output, exit_code) {
                    let auth_msg = Self::auth_error_message(tool_name);
                    if let Some(ref tx) = status_tx {
                        let (label, summary) = aidaemon_tools::sanitize::user_facing_tool_activity(
                            tool_name,
                            "authentication failed",
                            &aidaemon_tools::semantics::ToolDispatchFacts::default(),
                            false,
                        );
                        let _ = tx.try_send(StatusUpdate::tool_complete(label, summary));
                    }
                    Self::quarantine_unavailable_tool(
                        &self.tools,
                        &self.tool_names,
                        tool_name,
                        "authentication failed",
                    );
                    return Err(CliAgentUnavailableError {
                        tool_name: tool_name.to_string(),
                        message: auth_msg,
                    }
                    .into());
                }

                // Completed within timeout normally
                // Use stdout_buf for JSON extraction (clean, no stderr prefixes)
                let stdout_output = stdout_buf.lock().await.clone();
                info!(
                    tool = %tool_name,
                    stdout_len = stdout_output.len(),
                    stdout_preview = %truncate_str(&stdout_output, 200),
                    "CLI agent stdout captured"
                );
                let result_text = extract_meaningful_output(&stdout_output, max_output);
                info!(
                    tool = %tool_name,
                    result_len = result_text.len(),
                    result_preview = %truncate_str(&result_text, 200),
                    "CLI agent result extracted"
                );

                // Capture git diff
                let diff_section = if let Some(ref dir) = working_dir_owned {
                    Self::capture_git_diff(&self.backend, dir)
                        .await
                        .map(|diff| format!("\n\n## File Changes\n```diff\n{}\n```", diff))
                } else {
                    None
                };

                // Emit completion status
                if let Some(ref tx) = status_tx {
                    let raw_summary = if exit_code == Some(0) {
                        "completed successfully".to_string()
                    } else {
                        format!("exited with code {:?}", exit_code)
                    };
                    let (label, summary) = aidaemon_tools::sanitize::user_facing_tool_activity(
                        tool_name,
                        &raw_summary,
                        &aidaemon_tools::semantics::ToolDispatchFacts::default(),
                        false,
                    );
                    let _ = tx.try_send(StatusUpdate::tool_complete(label, summary));
                }

                if exit_code != Some(0) {
                    if exited_before_agent_output(exit_code, &stdout_output) {
                        mark_never_started();
                    }
                    // On error, show the display buffer which includes stderr
                    let error_msg = with_left(format_cli_agent_failure(
                        tool_name,
                        exit_code,
                        &display_output,
                        max_output,
                        diff_section.as_deref(),
                    ));

                    if let Some(task_id) = delegated_task_id {
                        self.persist_delegated_cli_result(task_id, None, Some(&error_msg))
                            .await;
                    }

                    return Ok(error_msg);
                }

                // A run that did not do its work failed, whatever it said,
                // by the rule every helper run is judged by.
                if let Some(end) = shared.end_record.get() {
                    if end.changed_something {
                        mark_run_changed_locally();
                    }
                    if end.end.phase() != crate::run_state::RunPhase::Succeeded {
                        let why = end.end.line(&end.evidence).unwrap_or_default();
                        let error_msg = with_left(format!(
                            "ERROR: CLI agent '{tool_name}' did not do its work. {why}\n\n\
                             Its report:\n{result_text}"
                        ));
                        if let Some(task_id) = delegated_task_id {
                            self.persist_delegated_cli_result(task_id, None, Some(&error_msg))
                                .await;
                        }
                        return Ok(error_msg);
                    }
                }

                // Success path
                let mut final_result = result_text;
                if let Some(diff) = diff_section {
                    final_result.push_str(&diff);
                }
                let final_result = with_left(final_result);
                if let Some(task_id) = delegated_task_id {
                    self.persist_delegated_cli_result(task_id, Some(final_result.as_str()), None)
                        .await;
                }
                Ok(final_result)
            }
            Ok(Err(_)) => {
                drop(prep.claim.take());
                // Channel closed unexpectedly
                let error_msg =
                    format!("ERROR: CLI agent '{}' task failed unexpectedly", tool_name);
                if let Some(task_id) = delegated_task_id {
                    self.persist_delegated_cli_result(task_id, None, Some(&error_msg))
                        .await;
                }
                Ok(error_msg)
            }
            Err(_) => {
                // Timeout - move to background
                // Note: the spawned task continues running and will update buffers
                let elapsed = timeout.as_secs();
                let partial_output = {
                    let buf = display_buf.lock().await;
                    truncate_with_note(&buf, 1000)
                };

                // The caller got an early return: the agent is a background
                // run now, and its completion is delivered.
                self.start_background_run(
                    &shared.run_slot,
                    &task_id,
                    session_id,
                    &run_label,
                    delegated_task_id,
                    status_tx.clone(),
                    run_began,
                )
                .await;
                record_background(&self.state, &call).await;
                // Its process may have ended (and its record been cleared)
                // just before the line above: its end is set before that
                // clearing, so a record written after it is cleared here.
                if shared.end_record.get().is_some() {
                    forget_background(&self.state, &call).await;
                }
                shared.should_notify.store(true, Ordering::Relaxed);

                // Store the running agent for later checking/cancellation
                let agent = tracked(&call, &shared, started_at_instant, prep.working_dir.clone());
                self.running.lock().await.insert(task_id.clone(), agent);
                // The reaper / cancel paths own the claim now (no await
                // between the insert above and this disarm).
                if let Some(guard) = prep.claim.as_mut() {
                    guard.disarm();
                }

                mark_background_run(&run_label);
                Ok(format!(
                    "{}\nCLI agent '{}' still running after {}s. Moved to background (task_id={}).\n\
                     Use action=\"check\" with task_id=\"{}\" to see output, or action=\"cancel\" to stop it.\n\n\
                     Partial output:\n{}",
                    crate::background_run::started_in_background(&run_label),
                    tool_name,
                    elapsed,
                    task_id,
                    task_id,
                    partial_output
                ))
            }
        }
    }

    /// Make a launch ready up to its process start: its agent in the
    /// owner's mode, its arguments, its prompt, and its claim on its
    /// working directory (taken now, so two runs that would change the
    /// same files are refused up front, queued or not). `Ok(Err(..))` is
    /// why it was not started.
    #[allow(clippy::too_many_arguments)]
    async fn prepare_run(
        &self,
        tool_name: &str,
        task_id: &str,
        prompt: &str,
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
        let Some(mode) = mode.or((!enforce_native_access).then_some(PermissionMode::Checked))
        else {
            let level = self.permissions().level();
            let program = coding_agent.name();
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
        // One that last said it cannot work is asked again first (it may
        // have been signed in since); still not usable, it is not started.
        if !crate::tools::agent_mode::readiness_of(tool_name).is_usable()
            && let aidaemon_coding_agents::Readiness::NotUsable { reason, fix } =
                crate::tools::agent_mode::recheck(tool_name).await
        {
            return Ok(Err(format!(
                "Blocked: {} (`{tool_name}`) is unavailable: {reason}{}",
                coding_agent.name(),
                fix.map(|fix| format!(" {fix}")).unwrap_or_default()
            )));
        }
        info!(
            agent = %tool_name,
            mode = mode.as_str(),
            model = model.as_deref().unwrap_or("its own"),
            "A coding agent runs in its mode"
        );
        // Full access: nothing is asked; the owner chose it.
        let nested = nested.filter(|_| mode != PermissionMode::FullAccess);
        let can_host =
            nested.is_some() && self.backend.kind() == aidaemon_tools::backend::BackendKind::Local;

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
            return Ok(Err("Blocked: CLI delegation was not started because its permission-bypass flags could not be removed.".to_string()));
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
        // host answers its requests.
        let mut launch = aidaemon_coding_agents::Launch {
            args,
            start: launch_start,
        };
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
            },
            &mut launch,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                return Ok(Err(format!(
                    "Blocked: CLI delegation was not started because {error}. Select an agent with a registered native sandbox adapter or provide a narrower access manifest."
                )));
            }
        };
        let args = launch.args;
        let aidaemon_coding_agents::Plan {
            prompt: prompt_placement,
            env: plan_env,
            env_remove: plan_env_remove,
            host: host_starter,
            keep_alive,
            keeps_session,
        } = plan;
        let writes = WriteScope::of(workspace_mode, &canonical_write_paths);
        let short_summary: String = prompt.chars().take(50).collect();

        // Claim the working directory and perform conflict checks.
        // Lock ordering when both locks are needed: running -> working_dir_claims.
        // The claims (std) lock must not be held across an await, so the
        // conflict message is computed inside the block and persisted after.
        let mut claim_guard: Option<WorkingDirClaimGuard> = None;
        if let Some(ref dir) = canonical_working_dir {
            let blocked_message: Option<String> = {
                let _running_guard = self.running.lock().await;
                let mut claims = lock_claims(&self.working_dir_claims);

                let held = claims.entry(dir.clone()).or_default();
                if let Some(claim) = conflicting(held, &writes) {
                    Some(conflict_message(dir, claim))
                } else {
                    held.push(WorkingDirClaim {
                        task_id: task_id.to_string(),
                        tool_name: tool_name.to_string(),
                        prompt_summary: short_summary.clone(),
                        writes: writes.clone(),
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
                return Ok(Err(message));
            }

            // From here on the claim is released by the guard's Drop on any
            // early return or future cancellation, unless it is disarmed when
            // the task hands off to the background `running` map.
            claim_guard = Some(WorkingDirClaimGuard::new(
                self.working_dir_claims.clone(),
                dir.clone(),
                task_id.to_string(),
            ));
        }

        // Every helper's mission carries the same rules, whichever program
        // runs it: the folders made for it and that its own duration is the
        // daemon's to state.
        // Only a coding agent reads a mission: any other program is given
        // its prompt as is.
        // What it was asked, as the parent wrote it: what its run view
        // shows (the rules below are the daemon's, not the task).
        let task = prompt.trim().to_string();
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
        // Build the enriched prompt if system_instruction is provided
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
            mode,
            coding_agent,
            nested,
            timeout,
            max_output,
            workspace_mode,
            working_dir: canonical_working_dir,
            write_paths: canonical_write_paths,
            prompt_placement,
            env: plan_env,
            env_remove: plan_env_remove,
            host: host_starter,
            keep_alive,
            keeps_session,
            continues: spec.continues.clone(),
            prompt: prompt.to_string(),
            task,
            final_prompt,
            claim: claim_guard,
        }))
    }

    /// Accept a background launch that found every slot taken. It is a run
    /// at once: its task id, its record (reading [`WAITING_FOR_SLOT`]) and
    /// its working-dir claim; it starts by itself, first come first served,
    /// when a slot frees, and runs from then on as an immediate launch
    /// does. Stopped before then, it ends without ever starting. What
    /// `prepare_run` decided (its mode, model, readiness and sandbox
    /// paths) holds from when it was accepted: a change the owner makes
    /// while it waits applies to the next launch.
    async fn queue_run<F>(
        &self,
        call: RunCall,
        mut prep: PreparedRun,
        slot: std::pin::Pin<Box<F>>,
    ) -> String
    where
        F: std::future::Future<
                Output = Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::AcquireError>,
            > + Send
            + 'static,
    {
        let task_id = call.task_id.clone();
        let run_id = cli_run_id(&task_id);
        let shared = RunShared::default();
        shared.should_notify.store(true, Ordering::Relaxed);
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
        if let Some(earlier) = &prep.continues {
            handles
                .runs
                .continues(&run_id, &earlier.run, &earlier.thread);
        }
        handles.runs.keeps_session(&run_id, prep.keeps_session);
        handles.runs.activity(&run_id, WAITING_FOR_SLOT.to_string());
        if let Some(ref tx) = call.status_tx {
            let _ = tx.try_send(StatusUpdate::ToolCancellable {
                name: call.tool_name.clone(),
                task_id: task_id.clone(),
            });
        }
        // Every run still waiting was accepted before this one, so is ahead
        // of it.
        let ahead = {
            let mut running = self.running.lock().await;
            let ahead = running
                .values()
                .filter(|agent| agent.process.is_queued())
                .count();
            running.insert(
                task_id.clone(),
                tracked(&call, &shared, Instant::now(), prep.working_dir.clone()),
            );
            ahead
        };
        // The reaper / cancel paths own the claim now.
        if let Some(guard) = prep.claim.as_mut() {
            guard.disarm();
        }
        info!(
            task_id = %task_id,
            tool = %call.tool_name,
            ahead,
            "A CLI agent waits for a free slot"
        );

        let label = call.label.clone();
        let tool_name = call.tool_name.clone();
        tokio::spawn(async move {
            let waited = tokio::select! {
                biased;
                () = shared.process.stopped() => None,
                permit = slot => Some(permit),
            };
            let permit = match waited {
                Some(Ok(permit)) if !shared.process.was_stopped() => permit,
                Some(Err(error)) => {
                    end_unstarted(
                        &handles,
                        &call,
                        &shared,
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
                // Stopped while it waited: it never starts.
                Some(Ok(_)) | None => {
                    end_unstarted(
                        &handles,
                        &call,
                        &shared,
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
            let started = match handles.start_process(&call, &mut prep, &shared).await {
                Ok(started) => started,
                Err(error) => {
                    end_unstarted(
                        &handles,
                        &call,
                        &shared,
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
            if !shared.process.start(started.handle.clone()) {
                // Stopped while it was being started: its process stops now
                // and its run ends as any stopped run does.
                kill_process(&handles.backend, &started.handle).await;
            }
            let (completion_tx, _completion_rx) = tokio::sync::oneshot::channel();
            let coding_agent = prep.coding_agent.clone();
            let keep_alive = prep.keep_alive.take();
            let max_output = prep.max_output;
            // The claim guard is disarmed: dropping it releases nothing.
            drop(prep);
            watch_run(
                handles,
                call,
                shared,
                started,
                coding_agent,
                max_output,
                keep_alive,
                permit,
                completion_tx,
            )
            .await;
        });

        mark_background_run(&label);
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

impl RunHandles {
    /// Start a prepared run's process, its slot held: note its handoff and
    /// invocation, take what its write roots hold, spawn it and start its
    /// host.
    async fn start_process(
        &self,
        call: &RunCall,
        prep: &mut PreparedRun,
        shared: &RunShared,
    ) -> anyhow::Result<StartedRun> {
        let tool_name = call.tool_name.as_str();
        let task_id = call.task_id.as_str();
        // A repository-wide diff is useful only when this invocation starts
        // from a clean baseline. Otherwise it attributes and exposes unrelated
        // pre-existing work to the child. Read-only runs never need a diff.
        let diff_dir = if prep.workspace_mode == CliWorkspaceMode::ReadWrite {
            if let Some(dir) = prep.working_dir.as_deref() {
                (CliAgentTool::dirty_worktree_entry_count(&self.backend, dir).await == Some(0))
                    .then(|| dir.to_string())
            } else {
                None
            }
        } else {
            None
        };
        if let Some(dir) = diff_dir.clone() {
            let _ = shared.diff_dir.set(dir);
        }
        // What its write roots hold now: its end is judged by whether they
        // changed. A read-only run changes nothing by contract.
        let watched = if prep.workspace_mode == CliWorkspaceMode::ReadWrite {
            prep.write_paths.clone()
        } else {
            Vec::new()
        };
        let receipt_before = tree_receipt(&self.backend, &watched).await;
        let files_before = file_marks(&self.backend, &watched).await;

        if let Some(delegated_task_id) = call.delegated_task_id.as_deref() {
            CliAgentTool::persist_delegated_cli_handoff(
                &self.state,
                delegated_task_id,
                tool_name,
                &prep.final_prompt,
                prep.working_dir.as_deref(),
            )
            .await;
        }

        info!(
            tool = tool_name,
            session = call.session_id.as_str(),
            prompt_len = prep.final_prompt.len(),
            working_dir = prep.working_dir.as_deref().unwrap_or("(default)"),
            async_mode = call.async_mode,
            workspace_mode = ?prep.workspace_mode,
            "CLI agent invocation"
        );

        // Log invocation start
        let prompt_summary: String = prep.prompt.chars().take(100).collect();
        let invocation_id = self
            .state
            .log_cli_agent_start(
                Some(task_id),
                &call.session_id,
                tool_name,
                &prompt_summary,
                prep.working_dir.as_deref(),
            )
            .await
            .unwrap_or(0);
        if invocation_id != 0
            && let Err(error) = self
                .state
                .note_cli_agent_run(
                    invocation_id,
                    &cli_run_id(task_id),
                    Some(prep.mode.as_str()),
                )
                .await
        {
            warn!(%error, "Could not note a coding agent's run on its invocation");
        }
        // What it was asked, whole, for its run view (the summary above is
        // the log's).
        if invocation_id != 0
            && let Err(error) = self
                .state
                .note_cli_agent_task(invocation_id, &prep.task)
                .await
        {
            warn!(%error, "Could not keep a coding agent's task on its invocation");
        }
        if invocation_id != 0
            && prep.keeps_session
            && let Err(error) = self
                .state
                .set_cli_agent_keeps_session(invocation_id, true)
                .await
        {
            warn!(%error, "Could not note that a coding agent kept its session");
        }
        if invocation_id != 0
            && let Some(earlier) = &prep.continues
            && let Err(error) = self
                .state
                .set_cli_agent_thread(invocation_id, &earlier.thread)
                .await
        {
            warn!(%error, "Could not note which thread a coding agent's run continues");
        }
        // The agent's typed events: kept on its invocation, and its latest
        // step on its run record. Its host's judge keeps each decision too.
        let events = crate::tools::cli_agent_events::AgentEventRecorder::new(
            prep.coding_agent.clone(),
            self.state.clone(),
            invocation_id,
            self.runs.clone(),
            cli_run_id(task_id),
        );

        let mut command_args = prep.args.clone();
        if prep.prompt_placement == aidaemon_coding_agents::PromptPlacement::LastArgument {
            command_args.push(prep.final_prompt.clone());
        }
        let mut request = ExecutionRequest::argv(prep.command.clone(), command_args);
        // A host sends the task and answers on stdin, which stays open.
        request.keep_stdin = prep.host.is_some();
        request.env.extend(std::mem::take(&mut prep.env));
        request
            .env_remove
            .extend(std::mem::take(&mut prep.env_remove));
        // Every process the agent starts inherits the run's tag, so the
        // ones still running after it ends can be found and reported.
        let run_tag = Uuid::new_v4().to_string();
        request
            .env
            .insert(left_running::RUN_TAG_VAR.to_string(), run_tag.clone());
        if let Some(ref dir) = prep.working_dir {
            request.cwd = Some(BackendPath::new(dir));
        }

        info!(
            task_id,
            tool = %tool_name,
            command = %prep.command,
            working_dir = ?prep.working_dir,
            "Starting CLI agent"
        );

        // Notify user this task can be cancelled
        if let Some(ref tx) = call.status_tx {
            let _ = tx.try_send(StatusUpdate::ToolCancellable {
                name: tool_name.to_string(),
                task_id: task_id.to_string(),
            });
        }

        let started_at = Instant::now();
        let mut spawned = match self.backend.spawn(request).await {
            Ok(child) => child,
            Err(e) => {
                // Ensure invocations don't stay "running" forever when spawn fails.
                if invocation_id != 0 {
                    let duration = started_at.elapsed().as_secs_f64();
                    let msg = format!("Failed to spawn CLI agent '{}': {}", tool_name, e);
                    let summary: String = msg.chars().take(200).collect();
                    let _ = self
                        .state
                        .log_cli_agent_complete(invocation_id, None, &summary, false, duration)
                        .await;
                }
                if let Some(ref delegated_task_id) = call.delegated_task_id {
                    CliAgentTool::persist_delegated_cli_result_with_state(
                        self.state.clone(),
                        delegated_task_id,
                        None,
                        Some(&format!("Failed to spawn CLI agent '{}': {}", tool_name, e)),
                    )
                    .await;
                }
                return Err(e);
            }
        };
        let handle = spawned.handle().clone();
        // Its host answers its approval requests (judged by this run) and
        // sends it its task.
        let host: Option<Arc<dyn aidaemon_coding_agents::AgentHost>> =
            match (prep.host.take(), spawned.take_stdin()) {
                (Some(starter), Some(stdin)) => {
                    // Full access: an agent whose protocol still asks is
                    // allowed as it asks, unjudged.
                    let judge: Arc<dyn aidaemon_coding_agents::ActionJudge> =
                        match prep.nested.take() {
                            Some(nested) => {
                                Arc::new(crate::tools::agent_mode::DaemonJudge::new(nested))
                            }
                            None => Arc::new(aidaemon_coding_agents::AllowAll),
                        };
                    let judge = events.judging(judge);
                    Some(starter.start(stdin, judge, &prep.final_prompt).await)
                }
                _ => None,
            };
        let Some(stdout) = spawned.take_stdout() else {
            let error = "Failed to capture stdout".to_string();
            if let Some(ref delegated_task_id) = call.delegated_task_id {
                CliAgentTool::persist_delegated_cli_result_with_state(
                    self.state.clone(),
                    delegated_task_id,
                    None,
                    Some(&error),
                )
                .await;
            }
            return Err(anyhow::anyhow!(error));
        };
        // A hosted agent's output is read by its host first (which may
        // translate it, as Codex's protocol is into `codex exec` events).
        let stdout: Box<dyn tokio::io::AsyncRead + Send + Unpin> = match host.clone() {
            Some(host) => Box::new(aidaemon_coding_agents::read_through(stdout, host)),
            None => Box::new(stdout),
        };
        let Some(stderr) = spawned.take_stderr() else {
            let error = "Failed to capture stderr".to_string();
            if let Some(ref delegated_task_id) = call.delegated_task_id {
                CliAgentTool::persist_delegated_cli_result_with_state(
                    self.state.clone(),
                    delegated_task_id,
                    None,
                    Some(&error),
                )
                .await;
            }
            return Err(anyhow::anyhow!(error));
        };
        Ok(StartedRun {
            child: spawned.into_child(),
            stdout,
            stderr,
            handle,
            host,
            events,
            run_tag,
            invocation_id,
            started_at,
            watched,
            receipt_before,
            files_before,
            diff_dir,
        })
    }
}

/// Watch a started run to its end: stream its output (emitting progress
/// and its background step), judge how it ended by the rule every helper
/// run is judged by, record it, deliver a background run's end to its chat
/// ([`deliver_end`]) and signal `completion_tx`.
#[allow(clippy::too_many_arguments)]
async fn watch_run(
    handles: RunHandles,
    call: RunCall,
    shared: RunShared,
    started: StartedRun,
    coding_agent: Arc<dyn aidaemon_coding_agents::CodingAgent>,
    max_output: usize,
    keep_alive: Option<Box<dyn std::any::Any + Send>>,
    slot_permit: tokio::sync::OwnedSemaphorePermit,
    completion_tx: tokio::sync::oneshot::Sender<(Option<i32>, bool, Option<usize>)>,
) {
    let _slot_permit = slot_permit;
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
        receipt_before: receipt_before_for_task,
        files_before: files_before_for_task,
        diff_dir: notify_working_dir,
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
    let expects_change = !watched_for_task.is_empty();
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
    let mut status_tick = tokio::time::interval(PROGRESS_INTERVAL);
    status_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    status_tick.tick().await; // consume the immediate first tick
    let started_at = Instant::now();
    let mut pending_lines: Vec<String> = Vec::new();
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut last_parsed_action: Option<String> = None;
    // How many lines the agent has printed: its background step.
    let mut printed_lines: u64 = 0;
    let mut loop_detector = LoopDetector::new();
    let mut loop_detected = false;
    let mut loop_pattern_count: Option<usize> = None;

    loop {
        if stdout_done && stderr_done {
            break;
        }

        // Check for loop detection and kill if needed
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
                        // Guardrail: kill external coding agents if they attempt to
                        // manipulate aidaemon's state DB (E2E rabbit-hole).
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

                        // Check for infinite loop pattern
                        if loop_detector.add_line(&text) && !loop_detected {
                            loop_detected = true;
                            loop_pattern_count = loop_detector.get_loop_pattern();
                        }

                        agent_events.read(&text).await;
                        // Write to stdout buffer (for JSON extraction)
                        {
                            let mut buf = stdout_buf_writer.lock().await;
                            append_bounded_line(&mut buf, "", &text, BUFFER_CAP);
                        }
                        // Write to display buffer
                        {
                            let mut buf = display_buf_writer.lock().await;
                            append_bounded_line(&mut buf, "", &text, BUFFER_CAP);
                        }
                        pending_lines.push(text);
                    }
                    _ => stdout_done = true,
                }
            }
            line = stderr_reader.next_line(), if !stderr_done => {
                match line {
                    Ok(Some(text)) => {
                        // Check for infinite loop pattern in stderr too
                        if loop_detector.add_line(&text) && !loop_detected {
                            loop_detected = true;
                            loop_pattern_count = loop_detector.get_loop_pattern();
                        }

                        // Only write to display buffer with [stderr] prefix
                        let mut buf = display_buf_writer.lock().await;
                        append_bounded_line(&mut buf, "[stderr] ", &text, BUFFER_CAP);
                        pending_lines.push(format!("[stderr] {}", text));
                    }
                    _ => stderr_done = true,
                }
            }
            // No wording of a silent agent's last line ends the run:
            // its stdin is null, so a prompt reads end-of-file at
            // once, and a run that stays silent is bounded by the
            // run's own limits, never by what its last line says.
            _ = status_tick.tick() => {}
        }

        // Emit progress updates at intervals
        // Parse JSON lines to extract meaningful progress, filter raw JSON
        if last_progress.elapsed() >= PROGRESS_INTERVAL {
            let mut progress_items: Vec<String> = Vec::new();
            for line in &pending_lines {
                if looks_like_json(line) {
                    // Try to extract meaningful progress from JSON
                    if let Some(progress) = extract_progress_from_json(line) {
                        progress_items.push(progress.clone());
                        last_parsed_action = Some(progress);
                    }
                } else {
                    // Non-JSON line, include as-is
                    progress_items.push(line.clone());
                }
            }

            let elapsed_secs = started_at.elapsed().as_secs();
            printed_lines += pending_lines.len() as u64;
            let chunk = if !progress_items.is_empty() {
                // Deduplicate consecutive items
                progress_items.dedup();
                progress_items.join("\n")
            } else if let Some(ref action) = last_parsed_action {
                // No new progress, but we have a last action - show heartbeat
                format!("⏳ {} ({}s)", action, elapsed_secs)
            } else {
                // No parsed progress at all - show generic heartbeat
                format!("⏳ Working... ({}s)", elapsed_secs)
            };

            if let Some(ref tx) = status_tx_clone {
                let _ = tx.try_send(StatusUpdate::ToolProgress {
                    name: tool_name_owned.clone(),
                    chunk: truncate_with_note(&chunk, 500),
                });
            }

            // A background run's step is how much the agent printed,
            // never a heartbeat or a timer, and never what it printed:
            // its output is its own data and can carry a credential. An
            // agent whose output reads as typed events has its steps from
            // them instead (until its first, its row shows its task).
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

    // Send any remaining lines (with JSON parsing)
    if !pending_lines.is_empty()
        && let Some(ref tx) = status_tx_clone
    {
        let mut progress_items: Vec<String> = Vec::new();
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

    // Wait for process to complete and signal via channel
    let exit_code = if loop_detected {
        // Process was killed due to loop detection
        None
    } else {
        match child.wait().await {
            Ok(status) => status.code(),
            Err(_) => None,
        }
    };
    // A hosted agent may exit 0 once its stdin closes, whatever its
    // turn did: work it left unfinished is a failed run.
    let exit_code = match (exit_code, agent_host_for_task.as_ref()) {
        (Some(0), Some(host)) if host.unfinished() => Some(1),
        (code, _) => code,
    };
    // What the agent left running, read once it has exited (and,
    // when the runtime stopped it, after its group was signalled).
    // Reported, never stopped. Only a local process can be read.
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
    // How the run ended, from what the runtime observed of it (its
    // exit status, whether the runtime stopped it, whether its write
    // roots changed), by the rule every helper run is judged by. Its
    // reply never decides it.
    let changed_something =
        tree_receipt(&backend_for_receipt, &watched_for_task).await != receipt_before_for_task;
    // The files it wrote among its write paths, however it wrote
    // them: what its run says it changed.
    for path in changed_files(&backend_for_receipt, &files_before_for_task).await {
        handles.runs.edited(&cli_run_id(&call.task_id), &path);
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
            expects_change,
            changed_something,
            step: task_context.clone(),
            // What the agent's own check refused (from its output)
            // and what its host was refused or declined.
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
    // The local sites a run set up within its own write roots are
    // part of its work, by the same rule as any call's
    // (`local_sites::set_up_locally`).
    {
        let status = if loop_detected || auth_failed || exit_code.is_none() {
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
    // Its process ended: a restart from now on cuts off nothing of it.
    forget_background(&handles.state, &call).await;
    // Persist completion even if the caller timed out and moved the task to background.
    if invocation_id != 0 {
        let duration = invocation_started_at.elapsed().as_secs_f64();
        let completion = if loop_detected {
            CliCompletionResult {
                success: false,
                authentication_failed: false,
                persisted_output: "Killed - infinite loop detected".to_string(),
                response: None,
                error: Some("Killed - infinite loop detected".to_string()),
            }
        } else {
            // Prefer stdout for success summaries; fall back to display output for failures.
            let stdout_text = stdout_buf_writer.lock().await.clone();
            let display_text = display_buf_writer.lock().await.clone();
            CliAgentTool::classify_completion_result(
                &tool_name_owned,
                exit_code,
                &stdout_text,
                &display_text,
                max_output_for_log,
            )
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
        // The agent's own report, then what it left running.
        let agent_report = completion.persisted_output;
        let persisted_output = match left.result_section() {
            Some(section) => format!("{agent_report}\n\n{section}"),
            None => agent_report.clone(),
        };
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

        let _ = state_for_completion
            .log_cli_agent_complete(
                invocation_id,
                exit_code,
                &persisted_output,
                success,
                duration,
            )
            .await;

        if let Some(ref delegated_task_id) = delegated_task_id_for_completion {
            CliAgentTool::persist_delegated_cli_result_with_state(
                state_for_completion.clone(),
                delegated_task_id,
                structured_response.as_deref(),
                structured_error.as_deref(),
            )
            .await;
        }

        // Durable state is authoritative for completion. Publish the
        // finished flag before optional chat delivery, which may be
        // slow, but never before the delegated task update above.
        finished_for_task.store(true, Ordering::Release);

        // Send proactive notification when the caller won't necessarily be waiting
        // for completion (async_mode or sync->timeout moved to background).
        if call.async_mode || shared.should_notify.load(Ordering::Relaxed) {
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
                    diff_dir: notify_working_dir.as_deref(),
                    max_output: max_output_for_log,
                    answer_alone: true,
                },
            )
            .await;
        }
    }
    // Publish completion only after durable invocation/task state has
    // been flushed. Reapers and `check` callers must never observe a
    // finished process while its delegated board task is still running.
    settle_run(&handles, &call, &shared, run_phase, run_evidence);
    let _ = completion_tx.send((exit_code, loop_detected, loop_pattern_count));
}

/// How a background run ended, for its chat ([`deliver_end`]).
struct EndDelivery<'a> {
    phase: &'a crate::run_state::RunEnd,
    evidence: &'a crate::run_state::RunEvidence,
    success: bool,
    /// The agent's own report.
    report: &'a str,
    /// Its report with what it left running: what the parent reads.
    persisted_output: &'a str,
    left: &'a left_running::LeftRunning,
    diff_dir: Option<&'a str>,
    max_output: usize,
    /// Whether a run that ends alone (in no group of its turn's runs)
    /// answers its chat. A queued run stopped before it started does not:
    /// nothing ran, and whoever stopped it knows.
    answer_alone: bool,
}

/// Deliver a background run's end: settle its card, enter its result in
/// the chat's history, and answer the chat (re-engaging the coordinator)
/// once the last run of its turn's group ended.
async fn deliver_end(
    handles: &RunHandles,
    call: &RunCall,
    shared: &RunShared,
    end: EndDelivery<'_>,
) {
    let run = shared.run_slot.get().cloned();
    let phase = end.phase.clone();
    let success = end.success;
    let status_word = if success { "completed" } else { "failed" };
    let task_id_for_notify = &call.task_id;
    let notify_session_id = call.session_id.clone();
    let task_context = &call.label;
    let persisted_output = end.persisted_output;
    // The run ends now, the moment its program did: its card
    // settles to its final header before anything else (a
    // continuation turn can wait minutes for its slot; live
    // 2026-11-13: a finished run read "Done 9m 40s" because
    // its end was recorded after the parent's next turn).
    let ended = match run.as_ref() {
        Some(run) => Some(run.finish(phase, end.evidence.clone()).await),
        None => None,
    };
    let notification_summary: String = end.report.chars().take(500).collect();
    let diff_section = if let Some(dir) = end.diff_dir {
        CliAgentTool::capture_git_diff(&handles.backend, dir)
            .await
            .map(|diff| format!("\n\n{}", diff))
    } else {
        None
    };
    // The result alone: the run's card says what it was and
    // how it ended, with its measured time.
    let body = format!(
        "{}{}{}",
        notification_summary.trim(),
        diff_section.unwrap_or_default(),
        end.left
            .owner_line()
            .map(|line| format!("\n\n{line}"))
            .unwrap_or_default()
    );
    let notification_type = if success { "completed" } else { "failed" };

    // Direct channel delivery is intentionally not a second
    // agent turn, but the result must still enter the session's
    // hot history. Otherwise the very next user follow-up sees
    // the handoff acknowledgement but not the completed work.
    let continuity_output = format!(
        "Background CLI task {status_word}.\nTask ID: {task_id_for_notify}\nTask: {task_context}\nResult:\n{persisted_output}"
    );
    let continuity_message = Message {
        content: Some(aidaemon_tools::sanitize::wrap_untrusted_output(
            "cli_agent",
            &continuity_output,
        )),
        importance: 0.8,
        ..Message::new_runtime(
            Uuid::new_v4().to_string(),
            notify_session_id.clone(),
            "assistant",
        )
    };
    if let Err(e) = handles.state.append_message(&continuity_message).await {
        warn!(
            task_id = %task_id_for_notify,
            session_id = %notify_session_id,
            error = %e,
            "cli_agent background completion could not enter session history"
        );
    }

    // The runs its turn left running are answered together:
    // until the last of them ends this one's end is on the
    // card only, and the last one's end answers the turn once
    // from every result.
    let group = match (run.as_ref(), ended.as_ref()) {
        (Some(run), Some((Some(state), on_card))) => run.end_in_group(
            state,
            crate::run_state::RunReport {
                text: persisted_output.to_string(),
                untrusted: true,
            },
            run.result(Some(state), &body, *on_card),
            end.max_output,
        ),
        _ => crate::background_run::RunsTurn::Alone,
    };
    let group_answer = match group {
        crate::background_run::RunsTurn::Waiting => {
            info!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                "A CLI agent's result waits for the rest of its turn's runs"
            );
            return;
        }
        // The owner stopped it (or every run of its turn): its card says
        // so, and nothing picks the work up again.
        crate::background_run::RunsTurn::Stopped => return,
        crate::background_run::RunsTurn::Answer(answer) => Some(answer),
        // Its record says who stopped it: the owner's Stop marks it there,
        // and the agent's own exit does not know.
        crate::background_run::RunsTurn::Alone
            if !end.answer_alone
                || end.phase.stopped_by_owner()
                || ended
                    .as_ref()
                    .and_then(|(state, _)| state.as_ref())
                    .is_some_and(crate::run_state::RunState::stopped_by_owner) =>
        {
            return;
        }
        crate::background_run::RunsTurn::Alone => None,
    };

    // The written answer, when the agent loop gives one.
    let mut reply: Option<(
        String,
        Option<aidaemon_channels::ports::AgentResponseEnvelope>,
    )> = None;
    let goal_scoped = call
        .goal_id
        .as_deref()
        .is_some_and(|goal_id| !goal_id.is_empty());
    if !goal_scoped && !notify_session_id.trim().is_empty() {
        let reengagement_allowed = {
            let mut log = handles.reengagements.lock().await;
            crate::tools::terminal::reengagement_allowed(
                &mut log,
                &notify_session_id,
                &format!("cli_agent {}", call.tool_name),
                Instant::now(),
            )
        };
        if !reengagement_allowed {
            warn!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                "cli_agent completion re-engagement budget exhausted; delivering raw result"
            );
        } else if let Some(ref agent) = handles.agent {
            let ending = match end.phase.line(end.evidence) {
                Some(why) => format!("{status_word}. {why}"),
                None => status_word.to_string(),
            };
            let (followup, parent_task_id) = match group_answer.as_ref() {
                Some(answer) => (answer.followup.clone(), answer.parent_task_id.clone()),
                None => (
                    CliAgentTool::build_background_reengagement_followup(
                        &call.tool_name,
                        task_id_for_notify,
                        task_context,
                        &ending,
                        persisted_output,
                    ),
                    None,
                ),
            };
            // Where the chat shows the run's own row, the answer need not
            // say again how it ended.
            let followup = match &handles.hub {
                Some(hub) if hub.shows_runs(&notify_session_id).await => format!(
                    "{followup}\n\n{}",
                    crate::helper_run::card::ROWS_SHOW_THE_END
                ),
                _ => followup,
            };
            info!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                "Re-engaging agent loop after CLI agent completion"
            );
            if let Some(run) = run.as_ref() {
                run.step(Step::Reporting).await;
            }
            // Owed until given: a stop before the answer leaves it to the
            // next boot instead of to no one.
            crate::owed_reports::owe(
                &*handles.state,
                crate::owed_reports::OwedReport {
                    run: task_id_for_notify.to_string(),
                    chat: notify_session_id.clone(),
                    parent_task_id: parent_task_id.clone(),
                    followup: followup.clone(),
                },
            )
            .await;
            let (continued, _followup_end) = crate::tools::terminal::run_background_continuation(
                agent.as_ref(),
                ConversationRequest {
                    session_id: notify_session_id.clone(),
                    user_text: followup,
                    status_tx: None,
                    user_role: aidaemon_tools::semantics::UserRole::Owner,
                    channel_ctx: aidaemon_channels::context::ChannelContext::internal(),
                    heartbeat: None,
                    parent_task_id,
                    parent_tool_call_id: None,
                    parent_result_id: None,
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
                Ok(envelope) if !envelope.text.trim().is_empty() => {
                    let text = aidaemon_tools::sanitize::sanitize_user_facing_reply(&envelope.text);
                    reply = Some((text, Some(envelope)));
                }
                Ok(_) => warn!(
                    task_id = %task_id_for_notify,
                    session_id = %notify_session_id,
                    "cli_agent completion re-engagement returned an empty reply"
                ),
                Err(e) => warn!(
                    task_id = %task_id_for_notify,
                    session_id = %notify_session_id,
                    error = %e,
                    "cli_agent completion re-engagement failed"
                ),
            }
        }
    }

    // The run ended above: its card shows how; the result
    // follows without repeating it.
    let present = |body: &str| match (run.as_ref(), ended.as_ref()) {
        (Some(run), Some((state, on_card))) => run.result(state.as_ref(), body, *on_card),
        _ => aidaemon_channels::presentation::present_notification(notification_type, body),
    };
    let (text, envelope) = match (reply, group_answer) {
        // A group's answer stands alone: the card shows
        // each run's end.
        (Some((text, envelope)), Some(_)) => (text, envelope),
        (Some((text, envelope)), None) => (present(&text), envelope),
        // No answer was written: every run's own result.
        (None, Some(answer)) => (answer.messages.join("\n\n"), None),
        (None, None) => (present(&body), None),
    };

    let mut delivered = false;
    if let Some(ref hub) = handles.hub {
        if let (Some(envelope), Some(agent)) = (envelope.as_ref(), handles.agent.as_ref()) {
            let _ = agent
                .record_continuation_delivery(
                    &notify_session_id,
                    envelope.delivery(
                        "background_router",
                        aidaemon_events::ResponseDeliveryState::Queued,
                        Vec::new(),
                        None,
                    ),
                )
                .await;
        }
        match hub.send_text_tracked(&notify_session_id, &text).await {
            Ok(platform_id) => {
                delivered = true;
                if let (Some(envelope), Some(agent)) = (envelope.as_ref(), handles.agent.as_ref()) {
                    let _ = agent
                        .record_continuation_delivery(
                            &notify_session_id,
                            envelope.delivery(
                                "background_router",
                                aidaemon_events::ResponseDeliveryState::PlatformAcknowledged,
                                platform_id.into_iter().collect(),
                                None,
                            ),
                        )
                        .await;
                }
            }
            Err(e) => warn!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                error = %e,
                "cli_agent background completion direct hub delivery failed"
            ),
        }
    }

    if !delivered {
        let queue_goal_id = match call.goal_id.as_deref() {
            Some(goal_id) if !goal_id.trim().is_empty() => goal_id,
            _ => "global",
        };
        let entry = crate::tools::traits::NotificationEntry::new(
            queue_goal_id,
            &notify_session_id,
            notification_type,
            &text,
        );
        if let Err(e) = handles.state.enqueue_notification(&entry).await {
            warn!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                error = %e,
                "cli_agent background completion enqueue failed"
            );
        }
    }
    crate::owed_reports::paid(&*handles.state, task_id_for_notify).await;
}

/// However a run ended, its record ends with it (a second end is a no-op),
/// and it reads as finished to `check`, the reaper and a stop waiting on it.
fn settle_run(
    handles: &RunHandles,
    call: &RunCall,
    shared: &RunShared,
    phase: crate::run_state::RunEnd,
    evidence: crate::run_state::RunEvidence,
) {
    if let Some(run) = shared.run_slot.get().cloned() {
        tokio::spawn(async move {
            run.finish(phase, evidence).await;
        });
    } else {
        // The waited-on record (none for a run with no chat): ended
        // here, with the end the runtime observed.
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
}

/// End a queued run that never started: stopped while it waited, or its
/// process could not be started once a slot freed. `report` is what its
/// result reads.
#[allow(clippy::too_many_arguments)]
async fn end_unstarted(
    handles: &RunHandles,
    call: &RunCall,
    shared: &RunShared,
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
    crate::helper_run::folders::release_new_folders(&call.new_folders);
    append_bounded_line(&mut *shared.stdout_buf.lock().await, "", report, BUFFER_CAP);
    append_bounded_line(
        &mut *shared.display_buf.lock().await,
        "",
        report,
        BUFFER_CAP,
    );
    forget_background(&handles.state, call).await;
    // Finished before its chat hears of it, as a run that ran is.
    shared.finished.store(true, Ordering::Release);
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
