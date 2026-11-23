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

/// A running CLI agent being tracked.
pub(super) struct RunningCliAgent {
    pub(super) tool_name: String,
    /// The name the owner reads for it: its title in plain words, else
    /// what it is ("bash-agent coding agent"), never its prompt.
    pub(super) label: String,
    pub(super) prompt_summary: String,
    pub(super) started_at: Instant,
    /// Combined output for display (includes [stderr] prefixes)
    pub(super) display_buf: Arc<Mutex<String>>,
    /// Pure stdout for JSON extraction
    pub(super) stdout_buf: Arc<Mutex<String>>,
    /// Backend-scoped handle for status display and cancellation.
    pub(super) process_handle: ProcessHandle,
    /// Set by the stream/wait task after the backend process exits.
    pub(super) finished: Arc<AtomicBool>,
    /// Set by the stream/wait task when the run failed to authenticate
    /// ([`cli_auth_failed`]), before `finished`.
    pub(super) auth_failed: Arc<AtomicBool>,
    /// Session ID for filtering cancel_all by session
    pub(super) session_id: String,
    /// Delegated task ID when this cli_agent run is acting as an executor.
    pub(super) delegated_task_id: Option<String>,
    /// Working directory for git diff capture
    pub(super) working_dir: Option<String>,
    /// What the agent left running, set once it has exited.
    pub(super) left_running: Arc<OnceLock<left_running::LeftRunning>>,
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

/// The id of a launched agent's background run in the runtime's records.
pub(super) fn cli_run_id(task_id: &str) -> String {
    format!("cli-agent-{task_id}")
}
