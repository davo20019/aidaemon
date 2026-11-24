//! `action="run"`: launching an agent, tracking its coordination membership,
//! and waiting on or backgrounding it.
//!
//! Preparation lives in [`super::prepare`], concurrency/folder queuing in
//! [`super::queue`], process watching in [`super::watch`], and background
//! completion delivery in [`super::delivery`].

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
    pub(super) command: String,
    pub(super) args: Vec<String>,
    /// The launch arguments prior to `coding_agent.prepare`, kept so a queued
    /// folder waiter can re-prepare with `resume` if an earlier run finishes
    /// and stores a resumable session while it waits.
    pub(super) base_launch: aidaemon_coding_agents::Launch,
    pub(super) enforce_native_access: bool,
    pub(super) read_paths: Vec<String>,
    pub(super) model: Option<String>,
    pub(super) can_host: bool,
    pub(super) mode: aidaemon_coding_agents::PermissionMode,
    pub(super) coding_agent: Arc<dyn aidaemon_coding_agents::CodingAgent>,
    pub(super) nested: Option<aidaemon_core::NestedCalls>,
    pub(super) detached_judge: Option<Arc<crate::approvals::DetachedNestedJudge>>,
    pub(super) spawn_family: Option<crate::authority::SpawnFamily>,
    pub(super) timeout: Duration,
    pub(super) max_output: usize,
    pub(super) workspace_mode: CliWorkspaceMode,
    /// Its working directory, normalized.
    pub(super) working_dir: Option<String>,
    /// The key under which its [`WorkingDirClaim`] is stored (usually
    /// `working_dir`, or the first `write_paths` entry when `working_dir` was
    /// omitted).
    pub(super) claim_dir: Option<String>,
    pub(super) write_scope: WriteScope,
    pub(super) write_paths: Vec<String>,
    pub(super) prompt_placement: aidaemon_coding_agents::PromptPlacement,
    pub(super) env: Vec<(String, String)>,
    pub(super) env_remove: Vec<String>,
    pub(super) host: Option<Box<dyn aidaemon_coding_agents::HostStarter>>,
    pub(super) keep_alive: Option<Box<dyn std::any::Any + Send>>,
    /// The agent keeps its session where a follow-up can continue it.
    pub(super) keeps_session: bool,
    /// The earlier run it continues.
    pub(super) continues: Option<crate::helper_run::Continues>,
    /// The prompt with its mission rules, before any enrichment.
    pub(super) prompt: String,
    /// What it was asked, without the daemon's mission rules.
    pub(super) task: String,
    pub(super) final_prompt: String,
    /// Released on drop until the run is tracked in the task table, whose
    /// reaper and cancel paths then own it.
    pub(super) claim: Option<WorkingDirClaimGuard>,
    /// Another run holds its folder and it waits for it instead of being
    /// refused (a launch the owner sent to the agent directly): it has no
    /// claim until then.
    pub(super) folder_wait: Option<FolderWait>,
}

/// A concurrency slot being waited for, in line.
pub(super) type SlotWait = std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::AcquireError>,
            > + Send,
    >,
>;

/// The longest task a run's record keeps of what it was asked.
const ASKED_CHARS: usize = 2000;

/// What a run was asked, as its record keeps it: the task without the
/// daemon's rules, and whether the owner sent it to the agent directly.
pub(super) fn run_ask(prep: &PreparedRun, call: &RunCall) -> aidaemon_channels::turns::RunAsk {
    let mut text: String = prep.task.chars().take(ASKED_CHARS).collect();
    if text.len() < prep.task.len() {
        text.push('…');
    }
    aidaemon_channels::turns::RunAsk {
        text,
        by_owner: call.owner_follow_up,
    }
}

/// Who launched a run and where it reports.
#[derive(Clone)]
pub(super) struct RunCall {
    pub(super) tool_name: String,
    pub(super) task_id: String,
    pub(super) session_id: String,
    pub(super) goal_id: Option<String>,
    pub(super) delegated_task_id: Option<String>,
    /// What it was sent to do, as the owner and the parent read it: its
    /// label, never a cut prompt.
    pub(super) label: String,
    pub(super) short_summary: String,
    pub(super) async_mode: bool,
    pub(super) status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub(super) grant_session: String,
    pub(super) new_folders: Vec<String>,
    /// When the call accepted it: its run record's start.
    pub(super) began: chrono::DateTime<chrono::Utc>,
    /// The owner's own follow-up ([`crate::helper_run::HelperSpec::owner_follow_up`]).
    pub(super) owner_follow_up: bool,
    /// Conversation pinned directly to a coding agent (no model reads the tool output).
    pub(super) direct_agent: bool,
}

/// What a run shares with its entry in the tool's task table.
#[derive(Clone, Default)]
pub(super) struct RunShared {
    /// Combined output for display (with `[stderr]` prefixes).
    pub(super) display_buf: Arc<Mutex<String>>,
    /// Pure stdout, for JSON extraction.
    pub(super) stdout_buf: Arc<Mutex<String>>,
    pub(super) finished: Arc<AtomicBool>,
    /// Woken whenever `finished` is published so `check` / `stop_background_job`
    /// wait without polling.
    pub(super) done_notify: Arc<tokio::sync::Notify>,
    pub(super) auth_failed: Arc<AtomicBool>,
    pub(super) left_running: Arc<OnceLock<left_running::LeftRunning>>,
    pub(super) end_record: Arc<OnceLock<CliRunEnd>>,
    /// How it ended, as its end judged it.
    pub(super) verdict: Arc<OnceLock<RunVerdict>>,
    /// Its background run, once it is one (see `start_background_run`).
    pub(super) run_slot: Arc<OnceLock<Arc<BackgroundRun>>>,
    /// Who answers for its end ([`handoff`]): its waiting caller, or the
    /// run itself once it is a background run.
    handoff: Arc<std::sync::atomic::AtomicU8>,
    /// Woken once a caller that moved it to the background is done.
    pub(super) moved: Arc<tokio::sync::Notify>,
    pub(super) diff_dir: Arc<OnceLock<super::prompt::DiffBase>>,
    pub(super) process: Arc<RunProcess>,
    /// The agent it waits on to leave its folder, while it does.
    pub(super) folder_wait: FolderWaitOn,
    /// Number of owner approval prompts currently in flight for this hosted
    /// run (pauses background idle/wall-clock timeouts while the owner decides).
    pub(super) in_flight_approvals: Arc<std::sync::atomic::AtomicUsize>,
}

/// Who answers for a run's end. A foreground run's caller waits for it; if
/// the wait runs out the caller moves it to the background, and then the
/// run delivers its own end to its chat. The caller and the run's end race,
/// so one compare-and-swap decides it: exactly one of them answers.
mod handoff {
    /// Its caller waits for it (the start of a foreground run).
    pub(super) const WAITED: u8 = 0;
    /// A background run: its end is delivered to its chat.
    pub(super) const BACKGROUND: u8 = 1;
    /// It ended while its caller still waited: the caller answers.
    pub(super) const ANSWERED: u8 = 2;
    /// A foreground run its caller moved to the background: its end is
    /// delivered to its chat once its background run is set up
    /// ([`super::RunShared::moved`]).
    pub(super) const MOVED: u8 = 3;
}

/// Who delivers a run's end, as its end found it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EndsBy {
    /// Its caller, still waiting: the end is its answer.
    Caller,
    /// The run itself, a background run from its start.
    Background,
    /// The run itself, moved to the background by its caller: it waits for
    /// that move to finish first.
    Moved,
}

/// Wakes a moved run's end once its caller set up its background run, or
/// gave up doing so (its call was dropped): on drop, either way.
pub(super) struct MoveDone(Arc<tokio::sync::Notify>);

impl Drop for MoveDone {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

impl RunShared {
    /// It is a background run from its start.
    pub(super) fn starts_in_background(&self) {
        self.handoff.store(handoff::BACKGROUND, Ordering::SeqCst);
    }

    /// Its caller's wait ran out: it becomes a background run, unless it
    /// already ended (`None`), when its caller answers for it after all.
    /// Its end waits until the returned guard drops, so the caller sets up
    /// its background run (its record, its turn's group) before the end
    /// reads it, and only for a run that did move.
    pub(super) fn move_to_background(&self) -> Option<MoveDone> {
        self.handoff
            .compare_exchange(
                handoff::WAITED,
                handoff::MOVED,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .ok()
            .map(|_| MoveDone(self.moved.clone()))
    }

    /// At its end: who delivers it, decided with its caller in one step.
    pub(super) fn ends_by(&self) -> EndsBy {
        match self.handoff.compare_exchange(
            handoff::WAITED,
            handoff::ANSWERED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => EndsBy::Caller,
            Err(handoff::MOVED) => EndsBy::Moved,
            Err(_) => EndsBy::Background,
        }
    }
}

/// The tool's handles a run keeps after its call returned.
#[derive(Clone)]
pub(super) struct RunHandles {
    pub(super) backend: SharedExecutionBackend,
    pub(super) state: Arc<dyn CliAgentStore>,
    pub(super) tools: Arc<std::sync::RwLock<HashMap<String, CliToolEntry>>>,
    pub(super) tool_names: Arc<std::sync::RwLock<Vec<String>>>,
    pub(super) runs: Arc<crate::run_state::RunStates>,
    pub(super) claims: Arc<WorkingDirClaims>,
    pub(super) claims_changed: Arc<tokio::sync::Notify>,
    pub(super) background_timeout: Duration,
    pub(super) idle_timeout: Duration,
    pub(super) hub: Option<Arc<dyn OutboundRouter>>,
    pub(super) agent: Option<Arc<dyn ConversationRuntime>>,
    pub(super) reengagements: Arc<Mutex<HashMap<String, VecDeque<Instant>>>>,
}

/// Registers a read-write CLI agent run in its request's [`SpawnFamily`]
/// [`Coordination`](crate::coordination::Coordination) ledger so concurrent
/// sub-agents and direct file/terminal tools share file-ownership tracking
/// and combined verification. Dropping before [`Self::finish`] closes the
/// open shell window and marks the member finished.
pub(super) struct CliCoordinationGuard {
    coordination: crate::coordination::Coordination,
    member_id: String,
    active: bool,
}

impl CliCoordinationGuard {
    fn register(
        family: &crate::authority::SpawnFamily,
        parent_session: &str,
        member_id: String,
        tool_name: &str,
        mission: &str,
        scope: Vec<String>,
    ) -> Option<Self> {
        if !family.coordinates() {
            return None;
        }
        family.note_cli_member();
        let coordination = family.coordination();
        coordination.add_declared_member(
            crate::coordination::Member {
                session: member_id.clone(),
                parent: parent_session.to_string(),
                mission: mission.to_string(),
                specialist: Some(tool_name.to_string()),
                scope,
                running: true,
            },
            None,
            None,
        );
        coordination.shell_began(&member_id, &member_id);
        Some(Self {
            coordination,
            member_id,
            active: true,
        })
    }

    /// Read all files changed by the CLI run (plus any files already tracked in
    /// the coordination ledger) and commit them to the shared ledger.
    pub(super) async fn finish(
        mut self,
        backend: &SharedExecutionBackend,
        changed_files: &[String],
    ) {
        self.active = false;
        let mut paths = self.coordination.tracked_paths();
        for path in changed_files {
            if !paths.contains(path) {
                paths.push(path.clone());
            }
        }
        let mut observed = Vec::with_capacity(paths.len());
        for path in paths {
            let bytes = backend.read(&BackendPath::new(&path)).await.ok();
            observed.push((path, bytes));
        }
        self.coordination
            .finish_cli_run(&self.member_id, &self.member_id, observed);
    }
}

impl Drop for CliCoordinationGuard {
    fn drop(&mut self) {
        if self.active {
            self.coordination.finish_member(&self.member_id);
        }
    }
}

/// A run's process once it started ([`RunHandles::start_process`]).
pub(super) struct StartedRun {
    pub(super) child: tokio::process::Child,
    pub(super) stdout: Box<dyn tokio::io::AsyncRead + Send + Unpin>,
    pub(super) stderr: tokio::process::ChildStderr,
    pub(super) handle: ProcessHandle,
    pub(super) host: Option<Arc<dyn aidaemon_coding_agents::AgentHost>>,
    pub(super) events: crate::tools::cli_agent_events::AgentEventRecorder,
    pub(super) run_tag: String,
    pub(super) invocation_id: i64,
    pub(super) started_at: Instant,
    /// Its write roots and what they held when it started: the paths it
    /// was given to write, or its whole folder when it may write anywhere
    /// in it.
    pub(super) watched: Vec<String>,
    /// It was given paths to write: sent to change them.
    pub(super) sent_to_write: bool,
    pub(super) snapshot_before: RootSnapshot,
    pub(super) diff_dir: Option<super::prompt::DiffBase>,
    pub(super) coordination: Option<CliCoordinationGuard>,
}

/// The task table's entry for a run.
pub(super) fn tracked(
    call: &RunCall,
    shared: &RunShared,
    accepted_at: Instant,
    claim_dir: Option<String>,
    write_scope: WriteScope,
    max_output: usize,
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
        done_notify: shared.done_notify.clone(),
        auth_failed: shared.auth_failed.clone(),
        session_id: call.session_id.clone(),
        delegated_task_id: call.delegated_task_id.clone(),
        diff_dir: shared.diff_dir.clone(),
        claim_dir,
        write_scope,
        left_running: shared.left_running.clone(),
        verdict: shared.verdict.clone(),
        max_output,
        folder_wait: shared.folder_wait.clone(),
    }
}

/// Keep a background run on record until its process ends, so a restart
/// that cuts it off is reported (or, if it never started, started again)
/// at the next boot: `resume_turns` in the daemon.
pub(super) async fn record_background(state: &Arc<dyn CliAgentStore>, call: &RunCall) {
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
pub(super) async fn forget_background(state: &Arc<dyn CliAgentStore>, call: &RunCall) {
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
            claims: self.working_dir_claims.clone(),
            claims_changed: self.claims_changed.clone(),
            background_timeout: self.background_timeout,
            idle_timeout: self.idle_timeout,
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
        detached_judge: Option<Arc<crate::approvals::DetachedNestedJudge>>,
        spawn_family: Option<crate::authority::SpawnFamily>,
    ) -> anyhow::Result<String> {
        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        // A background launch in a conversation set to a coding agent is
        // the owner's own message to it: one that finds its folder taken
        // waits for it rather than being refused.
        let direct_agent = !session_id.trim().is_empty()
            && crate::session_model::get(self.state.as_ref(), session_id)
                .await
                .agent
                .is_some();
        let waits_for_folder = async_mode && direct_agent;
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
                detached_judge,
                spawn_family,
                waits_for_folder,
                direct_agent,
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
            owner_follow_up: spec.owner_follow_up,
            direct_agent,
        };

        if prep.folder_wait.is_some() {
            return Ok(self.queue_run(call, prep, None).await);
        }

        // Slots go first come, first served: polled once here, a launch
        // that finds every slot taken is in line for the next one from now
        // on (a background one keeps its place; a foreground one is refused
        // and leaves the line).
        let mut slot = Box::pin(self.concurrency_limiter.clone().acquire_owned());
        let slot_permit = match futures::poll!(slot.as_mut()) {
            std::task::Poll::Ready(Ok(permit)) => permit,
            std::task::Poll::Pending if async_mode => {
                return Ok(self.queue_run(call, prep, Some(slot)).await);
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
                mark_refused_before_launch(PrelaunchRefusal::Contention);
                return Ok(message);
            }
        };
        drop(slot);

        let handles = self.run_handles();
        let shared = RunShared::default();
        if async_mode {
            shared.starts_in_background();
        }
        // prep's claim guard drops on an error return and releases the
        // claim.
        let started = handles.start_process(&call, &mut prep, &shared).await?;
        shared
            .process
            .start(started.handle.clone(), started.host.clone());
        let started_at_instant = started.started_at;
        let run_label = call.label.clone();
        let run_began = call.began;
        let timeout = prep.timeout;
        let max_output = prep.max_output;

        // Create completion channel - includes loop detection info
        // Result: (exit_code, was_killed_for_loop, loop_repetition_count)
        let (completion_tx, mut completion_rx) =
            tokio::sync::oneshot::channel::<(Option<i32>, bool, Option<usize>)>();

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
        handles
            .runs
            .asked(&cli_run_id(&task_id), run_ask(&prep, &call));
        if let Some(earlier) = &prep.continues {
            handles
                .runs
                .continues(&cli_run_id(&task_id), &earlier.run, &earlier.thread);
        }
        handles
            .runs
            .keeps_session(&cli_run_id(&task_id), prep.keeps_session);

        let background_timeout = handles.background_timeout.max(timeout);
        let idle_timeout = handles.idle_timeout;
        tokio::spawn(watch_run(
            handles,
            call.clone(),
            shared.clone(),
            started,
            prep.coding_agent.clone(),
            max_output,
            background_timeout,
            idle_timeout,
            prep.keep_alive.take(),
            slot_permit,
            completion_tx,
        ));

        // For async_mode, return immediately with task_id
        if async_mode {
            let agent = tracked(
                &call,
                &shared,
                started_at_instant,
                prep.claim_dir.clone(),
                prep.write_scope.clone(),
                prep.max_output,
            );
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
        let mut result = tokio::time::timeout(timeout, &mut completion_rx).await;
        if result.is_err() {
            match shared.move_to_background() {
                Some(_done) => {
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
                }
                None => result = Ok((&mut completion_rx).await),
            }
        }

        match result {
            Ok(Ok((exit_code, was_loop_killed, loop_count))) => {
                // Sync completion path: release the claim eagerly.
                drop(prep.claim.take());

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

                let diff_section = if let Some(ref base) = working_dir_owned {
                    Self::capture_git_diff(&self.backend, base)
                        .await
                        .map(|diff| format!("\n\n## File Changes\n```diff\n{}\n```", diff))
                } else {
                    None
                };

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

                let mut final_result = match self.run_handles().runs.get(&cli_run_id(&task_id)) {
                    Some(run) => format!("{}\n\n{result_text}", shown_note(&run)),
                    None => result_text,
                };
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
                let error_msg =
                    format!("ERROR: CLI agent '{}' task failed unexpectedly", tool_name);
                if let Some(task_id) = delegated_task_id {
                    self.persist_delegated_cli_result(task_id, None, Some(&error_msg))
                        .await;
                }
                Ok(error_msg)
            }
            Err(_) => {
                let elapsed = timeout.as_secs();
                let partial_output = {
                    let buf = display_buf.lock().await;
                    truncate_with_note(&buf, 1000)
                };

                if shared.end_record.get().is_some() {
                    forget_background(&self.state, &call).await;
                }

                let agent = tracked(
                    &call,
                    &shared,
                    started_at_instant,
                    prep.claim_dir.clone(),
                    prep.write_scope.clone(),
                    prep.max_output,
                );
                self.running.lock().await.insert(task_id.clone(), agent);
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
}

impl RunHandles {
    /// Start a prepared run's process, its slot held: note its handoff and
    /// invocation, take what its write roots hold, spawn it and start its
    /// host.
    pub(super) async fn start_process(
        &self,
        call: &RunCall,
        prep: &mut PreparedRun,
        shared: &RunShared,
    ) -> anyhow::Result<StartedRun> {
        let tool_name = call.tool_name.as_str();
        let task_id = call.task_id.as_str();
        if let Some(ref claim_dir) = prep.claim_dir {
            activate_claim(&mut lock_claims(&self.claims), claim_dir, task_id);
        }

        let diff_dir = if prep.workspace_mode == CliWorkspaceMode::ReadWrite {
            if let Some(dir) = prep.working_dir.as_deref()
                && CliAgentTool::dirty_worktree_entry_count(&self.backend, dir).await == Some(0)
            {
                Some(super::prompt::DiffBase {
                    dir: dir.to_string(),
                    head: CliAgentTool::git_head(&self.backend, dir).await,
                })
            } else {
                None
            }
        } else {
            None
        };
        if let Some(dir) = diff_dir.clone() {
            let _ = shared.diff_dir.set(dir);
        }

        let sent_to_write =
            prep.workspace_mode == CliWorkspaceMode::ReadWrite && !prep.write_paths.is_empty();
        let watched = match prep.workspace_mode {
            CliWorkspaceMode::ReadWrite if sent_to_write => prep.write_paths.clone(),
            CliWorkspaceMode::ReadWrite => prep.working_dir.iter().cloned().collect(),
            CliWorkspaceMode::ReadOnly => Vec::new(),
        };
        let snapshot_before = snapshot_roots(&self.backend, &watched).await;

        let coordination = if prep.workspace_mode == CliWorkspaceMode::ReadWrite {
            prep.spawn_family.as_ref().and_then(|family| {
                CliCoordinationGuard::register(
                    family,
                    &call.session_id,
                    cli_run_id(task_id),
                    tool_name,
                    &call.label,
                    watched.clone(),
                )
            })
        } else {
            None
        };

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
        request.keep_stdin = prep.host.is_some();
        request.env.extend(std::mem::take(&mut prep.env));
        request
            .env_remove
            .extend(std::mem::take(&mut prep.env_remove));
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
        let host: Option<Arc<dyn aidaemon_coding_agents::AgentHost>> =
            match (prep.host.take(), spawned.take_stdin()) {
                (Some(starter), Some(stdin)) => {
                    let nested = prep.nested.take();
                    let detached = prep.detached_judge.take();
                    let judge: Arc<dyn aidaemon_coding_agents::ActionJudge> =
                        if nested.is_some() || detached.is_some() {
                            Arc::new(
                                crate::tools::agent_mode::DaemonJudge::for_run_with_detached(
                                    nested,
                                    detached,
                                    crate::tools::agent_mode::Asker {
                                        run: cli_run_id(task_id),
                                        agent: prep.coding_agent.name().to_string(),
                                    },
                                )
                                .with_controls(
                                    shared.process.cancel_token(),
                                    shared.in_flight_approvals.clone(),
                                ),
                            )
                        } else {
                            Arc::new(aidaemon_coding_agents::AllowAll)
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
            sent_to_write,
            snapshot_before,
            diff_dir,
            coordination,
        })
    }
}
