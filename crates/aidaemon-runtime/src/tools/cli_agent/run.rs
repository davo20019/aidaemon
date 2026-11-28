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
    /// Its maker's word on being handed this work, for the owner watching
    /// ([`aidaemon_coding_agents::Delegation::WhileWatched`]).
    pub(super) delegation_note: Option<aidaemon_coding_agents::SignInNote>,
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
    /// The owner gave the agent full access and the run's device ceiling
    /// held it to `mode`: its asks at or below the ceiling get the owner's
    /// full-access answer (`ceiling`).
    pub(super) held_from_full_access: bool,
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
    /// What its sign-in added to `args` and its environment, for a
    /// re-prepared launch to add again.
    pub(super) signed: aidaemon_coding_agents::SignedIn,
    pub(super) host: Option<Box<dyn aidaemon_coding_agents::HostStarter>>,
    pub(super) keep_alive: Option<Box<dyn std::any::Any + Send>>,
    /// The peer coordination endpoint's lifetime guard: dropping it revokes
    /// the run's `aidaemon_peers` token. Kept apart from `keep_alive`, which a
    /// re-prepared launch replaces with its new plan's.
    pub(super) mcp_guard: Option<Box<dyn std::any::Any + Send>>,
    /// The agent keeps its session where a follow-up can continue it.
    pub(super) keeps_session: bool,
    /// The earlier run it continues.
    pub(super) continues: Option<crate::helper_run::Continues>,
    /// Host-injected MCP servers for this run (kept for queued re-prepares).
    pub(super) mcp_servers: Vec<aidaemon_coding_agents::McpEndpoint>,
    /// The prompt with its mission rules, before any enrichment.
    pub(super) prompt: String,
    /// What it was asked, without the daemon's mission rules.
    pub(super) task: String,
    pub(super) final_prompt: String,
    /// The conversation history put before the prompt for a new session
    /// ([`crate::helper_run::HelperSpec::prior_context`]).
    pub(super) prior_context: Option<String>,
    /// Released on drop until the run is tracked in the task table, whose
    /// reaper and cancel paths then own it.
    pub(super) claim: Option<WorkingDirClaimGuard>,
    /// Another run holds its folder and it waits for it instead of being
    /// refused (a launch the owner sent to the agent directly): it has no
    /// claim until then.
    pub(super) folder_wait: Option<FolderWait>,
    /// This queued follow-up targets an active session thread and waits for
    /// that thread's running run to finish before resuming its session.
    pub(super) thread_wait: Option<ThreadWait>,
}

/// Waiting for an active run in a specific coding-agent session thread to
/// finish before resuming that thread's session.
#[derive(Debug, Clone)]
pub(super) struct ThreadWait {
    pub(super) thread: String,
    pub(super) behind_run: String,
    pub(super) behind_agent: String,
}

/// A concurrency slot being waited for, in line.
pub(super) type SlotWait = std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::AcquireError>,
            > + Send,
    >,
>;

impl PreparedRun {
    /// Everything the running process must outlive: the plan's own guard and
    /// the peer endpoint's, handed to the run's watcher together.
    pub(super) fn take_keep_alive(&mut self) -> Option<Box<dyn std::any::Any + Send>> {
        match (self.keep_alive.take(), self.mcp_guard.take()) {
            (Some(kept), Some(mcp)) => Some(Box::new((kept, mcp))),
            (kept, mcp) => kept.or(mcp),
        }
    }
}

/// A lead run about to start: onto its conversation's tree when it was
/// queued behind that tree's lead ([`peers::ActivePeerBinding::adopt_conversation_tree`]).
pub(super) fn adopt_conversation_tree(call: &mut RunCall, prep: &PreparedRun) {
    let Some(binding) = call.peer_binding.as_mut() else {
        return;
    };
    let Some(scoped) = prep
        .mcp_guard
        .as_ref()
        .and_then(|guard| guard.downcast_ref::<aidaemon_mcp::ScopedEndpoint>())
    else {
        return;
    };
    binding.adopt_conversation_tree(scoped);
}

impl PreparedRun {
    /// It continues its agent's own session after all: the history put
    /// before its prompt for a new session is left out (the session has it).
    pub(super) fn drop_prior_context(&mut self) {
        if let Some(prefix) = self.prior_context.take() {
            // The daemon's own text, removed exactly as it was put there.
            self.final_prompt = self.final_prompt.replacen(&prefix, "", 1);
            self.prompt = self.prompt.replacen(&prefix, "", 1);
        }
    }
}

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
    /// Explicitly started as a brand-new coding-agent session (`"New <Agent>"`):
    /// never folded into an earlier session after waiting for its folder.
    pub(super) fresh_session: bool,
    /// Run spawned by another CLI via the peer-coordination MCP server.
    pub(super) is_peer_run: bool,
    /// It continues an earlier session of its agent (a peer answering a
    /// message, a run continued): leave to write, not a fresh assignment
    /// to change files, since what it was asked may be a question.
    pub(super) continues_session: bool,
    /// Tree binding for permit parking, completion updates, and cascading cancel.
    pub(super) peer_binding: Option<peers::ActivePeerBinding>,
    /// The device ceiling of the request that launched it
    /// ([`crate::device_cap`]): the turn that answers its chat when it ends
    /// stays capped if the request was.
    pub(super) device_cap: Option<crate::device_cap::DeviceCap>,
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
    /// It runs with full access: nothing it does is judged, so a note
    /// from a run capped by its device never goes into it.
    pub(super) unchecked: Arc<AtomicBool>,
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
        unchecked: shared.unchecked.clone(),
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
        task_prompt: &str,
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
        may_send_files: bool,
        fresh_session: bool,
        thread_wait: Option<ThreadWait>,
    ) -> anyhow::Result<String> {
        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        // A background launch the owner sent directly (a direct-agent
        // conversation, an explicit session follow-up, or "New <Agent>") waits
        // for its folder rather than being refused.
        let direct_agent = !session_id.trim().is_empty()
            && crate::session_model::get(self.state.as_ref(), session_id)
                .await
                .agent
                .is_some();
        let waits_for_folder =
            async_mode && (direct_agent || spec.owner_follow_up || fresh_session);

        // When peer coordination is enabled on a local backend, mint a tree and
        // scoped MCP server endpoint for this root CLI run so it can delegate to
        // and coordinate with sibling CLIs (including in direct-agent mode).
        let (root_mcp, root_keep_alive, effective_prompt, root_peer_binding) = if let Some(coord) =
            self.peer_coord.get()
            && self.backend.kind() == aidaemon_tools::backend::BackendKind::Local
        {
            let available: Vec<String> = self
                .tool_names
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            // The lead's own mode: no peer it starts runs above it.
            let lead_mode = self
                .tools
                .read()
                .ok()
                .and_then(|tools| tools.get(tool_name).map(|entry| self.entry_agent(entry)))
                .and_then(|agent| self.permissions().agent_mode(tool_name, &*agent))
                .or((!enforce_native_access)
                    .then_some(aidaemon_coding_agents::PermissionMode::Checked));
            let root_ctx = peers::PeerRootContext {
                lead_mode,
                session_id: session_id.to_string(),
                grant_session: grant_session.to_string(),
                working_dir: working_dir.map(str::to_string),
                read_paths: read_paths.to_vec(),
                write_paths: write_paths.to_vec(),
                enforce_native_access,
                project: project.cloned(),
                detached_judge: detached_judge.clone(),
                spawn_family: spawn_family.clone(),
                status_tx: status_tx.clone(),
                may_send_files,
                // What the run was asked, as recorded (the owner's message
                // for a direct agent; a helper's mission otherwise).
                owner_watches: direct_agent,
                lead_task: [task_prompt, spec.mission.as_str()]
                    .into_iter()
                    .find(|task| !task.trim().is_empty())
                    .map(str::to_string),
            };
            // An owner's conversation with its agent keeps one tree from
            // exchange to exchange: a follow-up reads the board and resumes
            // the peers of the runs before it.
            let continuity =
                (direct_agent || spec.owner_follow_up).then(|| peers::TreeContinuity {
                    lead: peers::ConversationLead {
                        session_id: session_id.to_string(),
                        agent: tool_name.to_string(),
                    },
                    fresh: fresh_session,
                    run_id: cli_run_id(&task_id),
                });
            if let Some(continuity) = &continuity {
                coord.load_conversation_tree(&continuity.lead).await;
            }
            let root_session = coord.start_root_tree(
                tool_name,
                &spec.label(&format!("{tool_name} coding agent")),
                root_ctx,
                &available,
                continuity,
            );
            root_session
                .tree
                .set_peer_run_id("peer-root", cli_run_id(&task_id));
            let binding = peers::ActivePeerBinding {
                tree: root_session.tree,
                peer_id: "peer-root".to_string(),
                permit_slots: root_session.permit_slots,
                is_root: true,
                launcher: Some(root_session.launcher),
                branch: None,
                rebind: root_session.rebind,
                conversation: root_session.conversation,
            };
            let has_known_agent = self
                .tools
                .read()
                .ok()
                .and_then(|tools| tools.get(tool_name).map(|entry| self.entry_agent(entry)))
                .is_some_and(|agent| agent.launch_arguments().is_some());
            let augmented = if has_known_agent {
                format!("{}\n\n{}", prompt.trim_end(), root_session.prompt_hint)
            } else {
                prompt.to_string()
            };
            (
                vec![root_session.endpoint],
                Some(Box::new(root_session.scoped) as Box<dyn std::any::Any + Send>),
                augmented,
                Some(binding),
            )
        } else {
            (Vec::new(), None, prompt.to_string(), None)
        };

        let mut prep = match self
            .prepare_run(
                tool_name,
                &task_id,
                &effective_prompt,
                task_prompt,
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
                thread_wait.is_some(),
                direct_agent,
                false,
                &root_mcp,
                &[],
                None,
            )
            .await?
        {
            Ok(prep) => prep,
            Err(message) => return Ok(message),
        };
        prep.thread_wait = thread_wait;
        prep.mcp_guard = root_keep_alive;
        let mut call = RunCall {
            tool_name: tool_name.to_string(),
            task_id: task_id.clone(),
            session_id: session_id.to_string(),
            goal_id: goal_id.map(str::to_string),
            delegated_task_id: delegated_task_id.map(str::to_string),
            label: spec.label(&format!("{tool_name} coding agent")),
            short_summary: task_prompt.chars().take(50).collect(),
            async_mode,
            status_tx: status_tx.clone(),
            grant_session: grant_session.to_string(),
            new_folders: spec.new_folders.clone(),
            began: chrono::Utc::now(),
            owner_follow_up: spec.owner_follow_up,
            direct_agent,
            fresh_session,
            is_peer_run: false,
            continues_session: spec.continues.is_some(),
            peer_binding: root_peer_binding,
            device_cap: prep
                .spawn_family
                .as_ref()
                .map(|family| family.device_cap().clone()),
        };

        if prep.thread_wait.is_some() || prep.folder_wait.is_some() {
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
        adopt_conversation_tree(&mut call, &prep);
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
        if started
            .host
            .as_ref()
            .is_some_and(|host| host.accepts_mid_run())
        {
            handles
                .runs
                .set_accepts_mid_run(&cli_run_id(&task_id), true);
        }

        let background_timeout = background_cap(handles.background_timeout, timeout);
        let idle_timeout = handles.idle_timeout;
        let permit_slot = peers::PermitSlot::new(
            slot_permit,
            self.concurrency_limiter.clone(),
            shared.finished.clone(),
            shared.in_flight_approvals.clone(),
        );
        tokio::spawn(watch_run(
            handles,
            call.clone(),
            shared.clone(),
            started,
            prep.coding_agent.clone(),
            max_output,
            background_timeout,
            idle_timeout,
            prep.take_keep_alive(),
            permit_slot,
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
                let mut result_text = extract_meaningful_output(&stdout_output, max_output);
                // A lead run's tree record: what the caller can quote of its
                // peers and board, which its own report only summarizes.
                if let Some(record) = call
                    .peer_binding
                    .as_ref()
                    .and_then(peers::ActivePeerBinding::lead_record)
                {
                    result_text.push_str("\n\n");
                    result_text.push_str(&record);
                }
                if call
                    .peer_binding
                    .as_ref()
                    .is_some_and(peers::ActivePeerBinding::acted)
                {
                    super::mark_run_acted_through_peers();
                }
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

    /// Launch a child CLI peer requested through the peer-coordination MCP
    /// server. Child peers inherit narrowed authority and owner approval cards,
    /// show up in `RunStates`, and report completion back to their [`PeerTree`]
    /// rather than posting standalone replies to the owner's chat.
    pub(super) async fn launch_peer_run(
        &self,
        spec: peers::PeerLaunchSpec,
    ) -> Result<peers::PeerLaunched, String> {
        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        let run_id = cli_run_id(&task_id);
        let mut prep = self
            .prepare_run(
                &spec.tool_name,
                &task_id,
                &spec.prompt,
                &spec.task_prompt,
                spec.working_dir.as_deref(),
                &spec.session_id,
                None,
                None,
                spec.workspace_mode,
                &spec.read_paths,
                &spec.write_paths,
                spec.enforce_native_access,
                &spec.spec,
                spec.project.as_ref(),
                None,
                spec.detached_judge,
                spec.spawn_family,
                false,
                false,
                false,
                spec.owner_watches,
                &spec.mcp_endpoints,
                &spec.nests_in,
                spec.mode_cap,
            )
            .await
            .map_err(|error| error.to_string())??;

        // Handed work its maker's terms leave unsettled: the owner watching
        // hears it each time.
        if let (Some(note), Some(tx)) = (prep.delegation_note, spec.status_tx.as_ref()) {
            let _ = tx.try_send(StatusUpdate::ToolProgress {
                name: spec.tool_name.clone(),
                chunk: format!(
                    "{} was handed work on its own sign-in. {} ({})",
                    prep.coding_agent.name(),
                    note.text,
                    note.link
                ),
            });
        }
        if let Some(mcp_guard) = spec.mcp_keep_alive {
            prep.mcp_guard = Some(mcp_guard as Box<dyn std::any::Any + Send>);
        }

        let slot_permit = tokio::select! {
            _ = spec.cancel.cancelled() => {
                return Err("peer launch cancelled while waiting for a concurrency slot".to_string());
            }
            acquired = self.concurrency_limiter.clone().acquire_owned() => {
                acquired.map_err(|_| "concurrency pool closed".to_string())?
            }
        };

        let call = RunCall {
            tool_name: spec.tool_name.clone(),
            task_id: task_id.clone(),
            session_id: spec.session_id.clone(),
            goal_id: None,
            delegated_task_id: None,
            label: spec.spec.label(&format!("{} peer agent", spec.tool_name)),
            short_summary: spec.task_prompt.chars().take(50).collect(),
            async_mode: true,
            status_tx: spec.status_tx.clone(),
            grant_session: spec.grant_session,
            new_folders: Vec::new(),
            began: chrono::Utc::now(),
            owner_follow_up: false,
            direct_agent: false,
            fresh_session: false,
            is_peer_run: true,
            continues_session: spec.spec.continues.is_some(),
            peer_binding: Some(spec.peer_binding),
            device_cap: prep
                .spawn_family
                .as_ref()
                .map(|family| family.device_cap().clone()),
        };

        let handles = self.run_handles();
        let shared = RunShared::default();
        shared.starts_in_background();

        let started = handles
            .start_process(&call, &mut prep, &shared)
            .await
            .map_err(|error| error.to_string())?;
        shared
            .process
            .start(started.handle.clone(), started.host.clone());
        let started_at_instant = started.started_at;
        let max_output = prep.max_output;
        let keeps_session = prep.keeps_session;
        let background_timeout = background_cap(handles.background_timeout, prep.timeout);
        // How long a caller that asked to wait waits before hearing the
        // peer is still running (a bound on the wait, not on the work).
        let wait_for = background_timeout.unwrap_or(prep.timeout);
        let idle_timeout = handles.idle_timeout;

        self.start_background_run(
            &shared.run_slot,
            &task_id,
            &call.session_id,
            &call.label,
            None,
            call.status_tx.clone(),
            call.began,
        )
        .await;
        handles.runs.set_agent(&run_id, prep.coding_agent.name());
        handles.runs.asked(&run_id, run_ask(&prep, &call));
        if let Some(earlier) = &prep.continues {
            handles
                .runs
                .continues(&run_id, &earlier.run, &earlier.thread);
        }
        handles.runs.keeps_session(&run_id, keeps_session);

        // Wire cascading tree cancellation to stop this peer process cleanly.
        let cancel_process = shared.process.clone();
        let cancel_backend = self.backend.clone();
        let cancel_finished = shared.finished.clone();
        let cancel_token = spec.cancel.clone();
        tokio::spawn(async move {
            cancel_token.cancelled().await;
            if !cancel_finished.load(Ordering::Acquire) {
                cancel_process.stop(&cancel_backend).await;
            }
        });

        let (completion_tx, _completion_rx) =
            tokio::sync::oneshot::channel::<(Option<i32>, bool, Option<usize>)>();
        let permit_slot = peers::PermitSlot::new(
            slot_permit,
            self.concurrency_limiter.clone(),
            shared.finished.clone(),
            shared.in_flight_approvals.clone(),
        );
        tokio::spawn(watch_run(
            handles,
            call.clone(),
            shared.clone(),
            started,
            prep.coding_agent.clone(),
            max_output,
            background_timeout,
            idle_timeout,
            prep.take_keep_alive(),
            permit_slot,
            completion_tx,
        ));

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

        if !spec.wait {
            return Ok(peers::PeerLaunched {
                run_id,
                finished: false,
                success: None,
                output: None,
                resumable: keeps_session,
            });
        }

        let wait_deadline = tokio::time::Instant::now() + wait_for;
        loop {
            let notified = shared.done_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if shared.finished.load(Ordering::Acquire) {
                break;
            }
            tokio::select! {
                _ = spec.cancel.cancelled() => {
                    let _ = self.handle_cancel(&task_id).await;
                    return Err("peer run cancelled while waiting".to_string());
                }
                res = tokio::time::timeout_at(wait_deadline, notified) => {
                    if res.is_err() {
                        return Ok(peers::PeerLaunched {
                            run_id,
                            finished: false,
                            success: None,
                            output: Some("Peer is still running; use check_peer to inspect later.".to_string()),
                            resumable: keeps_session,
                        });
                    }
                }
            }
        }

        let verdict = shared.verdict.get().cloned();
        let success = verdict.as_ref().map(|v| v.success).unwrap_or(false);
        let output = verdict.map(|v| v.text);
        let resumable = self
            .state
            .cli_agent_run(&run_id)
            .await
            .ok()
            .flatten()
            .is_some_and(|r| r.keeps_session && r.agent_session.is_some());

        Ok(peers::PeerLaunched {
            run_id,
            finished: true,
            success: Some(success),
            output,
            resumable,
        })
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
        shared.unchecked.store(
            prep.mode == aidaemon_coding_agents::PermissionMode::FullAccess,
            Ordering::SeqCst,
        );
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
                                )
                                .held_from_full_access(prep.held_from_full_access),
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
