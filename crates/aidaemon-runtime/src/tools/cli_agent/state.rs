//! The records of configured agents and of runs in flight or finished.

use super::*;

pub(super) struct CliToolEntry {
    pub(super) command: String,
    pub(super) args: Vec<String>,
    pub(super) description: String,
    pub(super) timeout: Duration,
    pub(super) max_output_chars: usize,
    /// Whether this was dynamically added (vs discovered at startup)
    pub(super) is_dynamic: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CliWorkspaceMode {
    ReadOnly,
    #[default]
    ReadWrite,
}

/// How a CLI agent's run ended, as the runtime decided it
/// ([`crate::helper_run::program_phase`]).
#[derive(Debug, Clone)]
pub(super) struct CliRunEnd {
    pub(super) end: crate::run_state::RunEnd,
    pub(super) evidence: crate::run_state::RunEvidence,
    pub(super) changed_something: bool,
}

/// A run's typed effects from its change receipt: local writes and the
/// processes it started when its write roots changed, nothing otherwise.
pub(super) fn changed_locally_semantics(changed: bool) -> ToolCallSemantics {
    if changed {
        ToolCallSemantics::observation_and_mutation_with(
            aidaemon_tools::semantics::ToolMutationEffects::LOCAL_WORKSPACE_WRITE
                .union(aidaemon_tools::semantics::ToolMutationEffects::PROCESS_STATE),
        )
    } else {
        ToolCallSemantics::observation()
    }
}

/// How a run ended, as its end judged it: whether it did its work, and its
/// answer (its report, or why it failed with what it said). What a check or
/// the reaper reports of a finished background run.
#[derive(Debug, Clone)]
pub(super) struct RunVerdict {
    pub(super) success: bool,
    pub(super) text: String,
}

/// A running CLI agent being tracked.
pub(super) struct RunningCliAgent {
    pub(super) tool_name: String,
    /// The name the owner reads for it: its title in plain words, else
    /// what it is ("bash-agent coding agent"), never its prompt.
    pub(super) label: String,
    pub(super) prompt_summary: String,
    /// When its call accepted it: a queued run has waited since then.
    pub(super) accepted_at: Instant,
    /// Combined output for display (includes [stderr] prefixes)
    pub(super) display_buf: Arc<Mutex<String>>,
    /// Pure stdout for JSON extraction
    pub(super) stdout_buf: Arc<Mutex<String>>,
    /// Its process, for status display and cancellation: none while it
    /// waits in line for a free slot.
    pub(super) process: Arc<RunProcess>,
    /// Set by the stream/wait task after the backend process exits.
    pub(super) finished: Arc<AtomicBool>,
    /// Woken when `finished` becomes true so `check` and `stop_background_job`
    /// wait without polling.
    pub(super) done_notify: Arc<tokio::sync::Notify>,
    /// Set by the stream/wait task when the run failed to authenticate
    /// ([`cli_auth_failed`]), before `finished`.
    pub(super) auth_failed: Arc<AtomicBool>,
    /// Session ID for filtering cancel_all by session
    pub(super) session_id: String,
    /// Delegated task ID when this cli_agent run is acting as an executor.
    pub(super) delegated_task_id: Option<String>,
    /// The working directory whose diff its result shows: set when it
    /// starts from a clean read-write checkout.
    pub(super) diff_dir: Arc<OnceLock<super::prompt::DiffBase>>,
    /// The working directory (or claim key) it claimed ([`WorkingDirClaim`]),
    /// released when it ends or is cancelled.
    pub(super) claim_dir: Option<String>,
    /// The typed write scope for this run (used for collision-aware FIFO
    /// folder waiting).
    pub(super) write_scope: WriteScope,
    /// What the agent left running, set once it has exited.
    pub(super) left_running: Arc<OnceLock<left_running::LeftRunning>>,
    /// How it ended, set before `finished` once its end was judged.
    pub(super) verdict: Arc<OnceLock<RunVerdict>>,
    /// How much of its output a result shows (its tool's limit).
    pub(super) max_output: usize,
    /// While it waits for another run to leave its folder: that run's
    /// agent ([`FolderWait`]).
    pub(super) folder_wait: FolderWaitOn,
    /// It runs with full access (set as its process starts): nothing it
    /// does is judged.
    pub(super) unchecked: Arc<AtomicBool>,
}

/// The agent a run waits on to leave its folder, while it does.
pub(super) type FolderWaitOn = Arc<std::sync::Mutex<Option<String>>>;

/// Where a run's process stands: a background launch that found every slot
/// taken waits in line ([`ProcessState::Queued`]) until one frees.
#[derive(Default)]
pub(super) struct RunProcess {
    state: std::sync::Mutex<ProcessState>,
    /// Wakes a queued run that was stopped before it started.
    stop: tokio::sync::Notify,
    /// Per-run cancellation token wired into `DaemonJudge` / `DetachedNestedJudge`
    /// so stopping the run immediately cancels any pending approval card.
    cancel: tokio_util::sync::CancellationToken,
}

#[derive(Default)]
enum ProcessState {
    /// Waiting for a free slot: no process yet.
    #[default]
    Queued,
    /// Its process, its protocol host (when hosted), and when it started.
    Started(
        ProcessHandle,
        Option<Arc<dyn aidaemon_coding_agents::AgentHost>>,
        Instant,
    ),
    /// Stopped while it waited: it never starts.
    Stopped,
}

impl RunProcess {
    fn lock(&self) -> std::sync::MutexGuard<'_, ProcessState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Its process, once started.
    pub(super) fn handle(&self) -> Option<ProcessHandle> {
        match &*self.lock() {
            ProcessState::Started(handle, _, _) => Some(handle.clone()),
            _ => None,
        }
    }

    /// Its protocol host, once started (when hosted).
    pub(super) fn host(&self) -> Option<Arc<dyn aidaemon_coding_agents::AgentHost>> {
        match &*self.lock() {
            ProcessState::Started(_, host, _) => host.clone(),
            _ => None,
        }
    }

    /// When its process started.
    pub(super) fn started_at(&self) -> Option<Instant> {
        match &*self.lock() {
            ProcessState::Started(_, _, at) => Some(*at),
            _ => None,
        }
    }

    /// Whether it still waits for a free slot.
    pub(super) fn is_queued(&self) -> bool {
        matches!(*self.lock(), ProcessState::Queued)
    }

    /// Whether it was stopped while it waited.
    pub(super) fn was_stopped(&self) -> bool {
        matches!(*self.lock(), ProcessState::Stopped)
    }

    /// Record its started process. False when it was stopped while it
    /// waited: its caller stops the process it just started.
    pub(super) fn start(
        &self,
        handle: ProcessHandle,
        host: Option<Arc<dyn aidaemon_coding_agents::AgentHost>>,
    ) -> bool {
        let mut state = self.lock();
        if matches!(*state, ProcessState::Stopped) {
            return false;
        }
        *state = ProcessState::Started(handle, host, Instant::now());
        true
    }

    /// Per-run cancellation token cancelled by [`Self::stop`].
    pub(super) fn cancel_token(&self) -> tokio_util::sync::CancellationToken {
        self.cancel.clone()
    }

    /// Stop it: cancels any pending owner approval card, asks a started hosted
    /// process to interrupt cooperatively first before falling back to process
    /// termination; a queued run never starts.
    pub(super) async fn stop(&self, backend: &SharedExecutionBackend) {
        self.cancel.cancel();
        let started = {
            let mut state = self.lock();
            match &*state {
                ProcessState::Started(handle, host, _) => Some((handle.clone(), host.clone())),
                ProcessState::Queued | ProcessState::Stopped => {
                    *state = ProcessState::Stopped;
                    None
                }
            }
        };
        match started {
            Some((handle, host)) => {
                if let Some(host) = host {
                    host.interrupt().await;
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                kill_process(backend, &handle).await;
            }
            None => self.stop.notify_one(),
        }
    }

    /// Resolves once it is stopped while it waits.
    pub(super) async fn stopped(&self) {
        self.stop.notified().await;
    }
}

impl RunningCliAgent {
    /// When its process started; while it waits, when it was accepted.
    pub(super) fn running_since(&self) -> Instant {
        self.process.started_at().unwrap_or(self.accepted_at)
    }

    /// The agent it waits on to leave its folder, while it does.
    pub(super) fn waits_for_folder(&self) -> Option<String> {
        self.folder_wait
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

/// Finalized background CLI task output retained briefly so `action="check"`
/// can still return results after automatic reaping.
#[derive(Clone)]
pub(super) struct CompletedCliAgent {
    pub(super) result: String,
    pub(super) completed_at: Instant,
    pub(super) session_id: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CliCompletionResult {
    pub(super) success: bool,
    pub(super) authentication_failed: bool,
    pub(super) persisted_output: String,
    pub(super) response: Option<String>,
    pub(super) error: Option<String>,
}

/// A configured CLI could not execute any part of the delegated task, so it is
/// safe to retry the same request with another configured CLI. This remains a
/// typed internal signal: the caller must not infer retryability from the
/// wording of a tool error.
#[derive(Debug)]
pub(super) struct CliAgentUnavailableError {
    pub(super) tool_name: String,
    pub(super) message: String,
}

impl std::fmt::Display for CliAgentUnavailableError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CliAgentUnavailableError {}

/// What every launched agent's run id starts with: a marker the daemon
/// writes itself ([`cli_run_id`]), read back by [`is_cli_agent_run`].
const CLI_RUN_PREFIX: &str = "cli-agent-";

/// The id of a launched agent's background run in the runtime's records.
pub(super) fn cli_run_id(task_id: &str) -> String {
    format!("{CLI_RUN_PREFIX}{task_id}")
}

/// Whether `id` is a launched agent's run id (`cli-agent-<task id>`): how a boot
/// tells the coding agents a restart cut off from the other background
/// jobs on record.
pub fn is_cli_agent_run(id: &str) -> bool {
    cli_agent_task_id(id).is_some()
}

/// The task id a launched agent's run id (`cli-agent-<task id>`) carries:
/// the `task_id=` its launch's result gave.
pub fn cli_agent_task_id(run_id: &str) -> Option<&str> {
    // A marker the daemon itself writes (`cli_run_id`).
    run_id.strip_prefix(CLI_RUN_PREFIX)
}
