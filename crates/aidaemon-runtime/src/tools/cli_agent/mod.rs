//! The `cli_agent` tool: runs coding agents (Claude Code, Codex, ...) on
//! a prompt, in the foreground or background, and reports what they did.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{Mutex, Semaphore, mpsc};
use tracing::{info, warn};
use uuid::Uuid;

use crate::background_run::BackgroundRun;
use crate::spawn::{ConversationRequest, ConversationRuntime};
use crate::tools::cli_agent_left_running as left_running;
use crate::tools::traits::{
    DynamicCliAgent, Message, Tool, ToolArgumentContractViolation, ToolCallAccessManifest,
    ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolOutcomeStatus, ToolTargetHint, ToolTargetHintKind, ToolVerificationMode,
};
use crate::turn::execution_state::TargetScope;
use crate::turn::validation_state::{
    ExecutorHandoff, TaskValidationOutcome, derive_executor_step_result,
    persist_executor_handoff_context, persist_executor_result_context,
};
use aidaemon_channels::ports::OutboundRouter;
use aidaemon_core::card::Step;
use aidaemon_events::text::{truncate_str, truncate_with_note};
use aidaemon_tools::backend::SharedExecutionBackend;
use aidaemon_tools::permission_bypass::strip_permission_bypass_flags;
#[cfg(test)]
use aidaemon_tools::permission_bypass::without_permission_bypass;
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::status::StatusUpdate;
use aidaemon_tools::text::floor_char_boundary;
use aidaemon_tools::{BackendFileType, BackendPath, ExecutionRequest, ProcessHandle};

mod args;
mod background;
mod claims;
mod launch;
mod output;
mod prompt;
mod receipts;
mod registry;
mod run;
mod state;
#[cfg(test)]
mod tests;
mod tool_impl;

use args::*;
use claims::*;
use launch::*;
use output::*;
use receipts::*;
use run::*;
use state::*;

/// What the tool keeps in its host's store: the agents it may run and their
/// invocation log, the chat's recent history for a run's context, the goal
/// and task a delegated run reports to, and notices for a chat no one reads.
pub trait CliAgentStore:
    aidaemon_tools::services::DynamicCliAgentStore
    + aidaemon_tools::services::SettingsStore
    + aidaemon_events::stores::MessageStore
    + aidaemon_work::stores::GoalStore
    + aidaemon_work::stores::TaskStore
    + aidaemon_work::stores::NotificationStore
{
}

impl<T> CliAgentStore for T where
    T: aidaemon_tools::services::DynamicCliAgentStore
        + aidaemon_tools::services::SettingsStore
        + aidaemon_events::stores::MessageStore
        + aidaemon_work::stores::GoalStore
        + aidaemon_work::stores::TaskStore
        + aidaemon_work::stores::NotificationStore
        + ?Sized
{
}

/// `[cli_agents]`: the coding agents the daemon may run, and their limits.
#[derive(Debug, Deserialize, Clone)]
pub struct CliAgentsConfig {
    #[serde(default = "default_cli_agents_enabled")]
    pub enabled: bool,
    #[serde(default = "default_cli_agents_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_cli_agents_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default)]
    pub tools: HashMap<String, CliToolConfig>,
    /// `[cli_agents.permissions]`: the owner's mode per agent, by name
    /// (`same`, `checked`, `every_command`, `full_access`), read through
    /// [`crate::permissions`]; never more than AIdaemon's level allows.
    #[serde(default)]
    pub permissions: HashMap<String, String>,
}

impl Default for CliAgentsConfig {
    fn default() -> Self {
        Self {
            enabled: default_cli_agents_enabled(),
            timeout_secs: default_cli_agents_timeout_secs(),
            max_output_chars: default_cli_agents_max_output_chars(),
            tools: HashMap::new(),
            permissions: HashMap::new(),
        }
    }
}

fn default_cli_agents_enabled() -> bool {
    false
}

fn default_cli_agents_timeout_secs() -> u64 {
    600
}
fn default_cli_agents_max_output_chars() -> usize {
    16000
}

/// One `[cli_agents.tools.<name>]` entry.
#[derive(Debug, Deserialize, Clone)]
pub struct CliToolConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub max_output_chars: Option<usize>,
    /// The protocol it speaks when it is not one the daemon already knows:
    /// `"acp"` (the Agent Client Protocol) hosts it, every action it takes
    /// asked first.
    #[serde(default)]
    pub protocol: Option<String>,
}

/// Interval for emitting progress updates (avoid spamming the channel).
#[cfg(test)]
const PROGRESS_INTERVAL: Duration = Duration::from_millis(25);
#[cfg(not(test))]
const PROGRESS_INTERVAL: Duration = Duration::from_secs(2);

/// How long `action="check"` waits for a running task to finish before it
/// reports "still running". A check that answers at once invites a model to
/// poll in a tight loop, each poll re-billing the whole context (live
/// 2026-11-11: ten checks in 36 s, ~685k input tokens, while Codex worked for
/// six minutes). Waiting turns each check into a long poll that returns the
/// moment the task ends.
#[cfg(test)]
const CHECK_WAIT: Duration = Duration::from_millis(300);
#[cfg(not(test))]
const CHECK_WAIT: Duration = Duration::from_secs(60);

/// How often a waiting check looks at the task's finished flag.
const CHECK_POLL: Duration = Duration::from_millis(250);

/// Loop detection: window size for tracking recent lines
const LOOP_DETECTION_WINDOW: usize = 100;

/// Loop detection: threshold - if same line appears this many times in window, it's a loop
const LOOP_DETECTION_THRESHOLD: usize = 50;

/// Max concurrent CLI agent processes
const DEFAULT_MAX_CONCURRENT: usize = 3;

/// Max enriched prompt size (16 KB)
const MAX_PROMPT_SIZE: usize = 16384;

/// Max git diff size to append to results (4 KB)
const MAX_DIFF_SIZE: usize = 4096;

tokio::task_local! {
    /// Set by the call when it leaves its work running in the background
    /// (started async, moved to background, or a check of a task still
    /// running), so the outcome is typed from what the call did, never
    /// guessed from its text (a finished agent's output can say anything).
    static LEFT_IN_BACKGROUND: std::cell::Cell<bool>;
    /// Set by a call that observed a background task's end (a check that
    /// returned its finished result, or a cancel), so the run knows the job
    /// it launched is no longer running.
    static BACKGROUND_SETTLED: std::cell::Cell<bool>;
    /// Set by a run that finished in the call and whose change receipt
    /// shows it changed its write roots: its outcome carries those typed
    /// effects, so the dispatcher treats what it set up like any call's.
    static RUN_CHANGED_LOCALLY: std::cell::Cell<bool>;
    /// The name of the run this call left in the background, when it
    /// started or moved one there.
    static BACKGROUND_RUN_LABEL: std::cell::RefCell<Option<String>>;
    /// Set when no agent ever started: every configured agent failed to
    /// launch (a typed [`CliAgentUnavailableError`]), or the launcher exited
    /// 127 before the agent wrote anything ([`exited_before_agent_output`]).
    static NEVER_STARTED: std::cell::Cell<bool>;
}

/// Record that no agent started for this call.
fn mark_never_started() {
    let _ = NEVER_STARTED.try_with(|flag| flag.set(true));
}

/// Record the run this call left in the background, by its name.
fn mark_background_run(label: &str) {
    mark_left_in_background();
    let _ = BACKGROUND_RUN_LABEL.try_with(|slot| *slot.borrow_mut() = Some(label.to_string()));
}

/// Record that this call's finished run changed its write roots.
fn mark_run_changed_locally() {
    let _ = RUN_CHANGED_LOCALLY.try_with(|flag| flag.set(true));
}

/// Record that this call left its work running in the background.
fn mark_left_in_background() {
    let _ = LEFT_IN_BACKGROUND.try_with(|flag| flag.set(true));
}

/// Record that this call observed a background task's end.
fn mark_background_settled() {
    let _ = BACKGROUND_SETTLED.try_with(|flag| flag.set(true));
}

/// What a delegated CLI agent's sandbox lets it do, as the adapters below
/// enforce it (`apply_read_only_cli_adapter`, `apply_scoped_cli_adapter`:
/// the network off, only `:minimal` plus the run's own paths reachable).
pub const CLI_SANDBOX: crate::helper_run::CliSandbox = crate::helper_run::DELEGATED_CLI_SANDBOX;

pub struct CliAgentTool {
    /// Stored facts handed to delegated agents, when the daemon's own memory
    /// keeps them.
    context_facts: Option<Arc<dyn crate::tools::traits::FactStore>>,
    /// Project memory, handed to delegated agents of a turn in a project.
    project_notes: Option<Arc<aidaemon_memory::project::ProjectNotes>>,
    backend: SharedExecutionBackend,
    // MUST be std::sync::RwLock because schema() is sync
    tools: Arc<std::sync::RwLock<HashMap<String, CliToolEntry>>>,
    tool_names: Arc<std::sync::RwLock<Vec<String>>>,
    running: Arc<Mutex<HashMap<String, RunningCliAgent>>>, // task_id -> RunningCliAgent
    completed: Arc<Mutex<HashMap<String, CompletedCliAgent>>>, // task_id -> finalized output
    working_dir_claims: Arc<WorkingDirClaims>, // normalized dir -> claim (std Mutex: see WorkingDirClaimGuard)
    state: Arc<dyn CliAgentStore>,
    default_timeout: Duration,
    default_max_output: usize,
    max_concurrent: usize,
    concurrency_limiter: Arc<Semaphore>,
    hub: OnceLock<Weak<dyn OutboundRouter>>,
    /// The runtime's run records: an agent left running in the background is
    /// a run there, shown on its turn's card. Wired via [`Self::set_runs`].
    runs: OnceLock<Arc<crate::run_state::RunStates>>,
    /// The session checklists a background run's turn waits on.
    plan_store: OnceLock<Arc<aidaemon_work::plans::PlanStore>>,
    /// Re-enter the root agent when background delegation completes so the
    /// delegated step advances the original user outcome instead of ending it.
    agent: OnceLock<Weak<dyn ConversationRuntime>>,
    reengagements: Arc<Mutex<HashMap<String, VecDeque<Instant>>>>,
    /// The coding agents it knows: how each is launched in each mode.
    coding_agents: aidaemon_coding_agents::CodingAgents,
    /// The owner's choice per agent and AIdaemon's level, which caps it:
    /// the daemon's shared value once [`Self::set_permissions`] wires it.
    permissions: Arc<OnceLock<crate::permissions::Permissions>>,
}
