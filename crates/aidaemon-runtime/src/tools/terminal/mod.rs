//! The `terminal` tool: runs shell commands on the host or confined,
//! in the foreground or background, and reports what they did.

use crate::tools::traits::ToolSemantics;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use aidaemon_policies::contains_shell_operator;
use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::SqlitePool;
use std::sync::LazyLock;
use tokio::io::AsyncReadExt;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::spawn::{ConversationRequest, ConversationRuntime};
use crate::tools::traits::{
    GoalStore, NotificationStore, Tool, ToolArgumentContractViolation, ToolCallContext,
    ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolMutationEffects,
    ToolOutcomeStatus, ToolTargetHint, ToolTargetHintKind, ToolVerificationMode,
};
use aidaemon_channels::hub::{MediaKind, MediaMessage};
use aidaemon_channels::ports::OutboundRouter;
use aidaemon_core::card::Step;
use aidaemon_events::text::truncate_str;
use aidaemon_events::{
    ApprovalDeniedData, ApprovalGrantedData, ApprovalRequestedData, DecisionPointData,
    DecisionType, DiagnosticSeverity, EventStore, EventType,
};
use aidaemon_tools::semantics::ApprovalResponse;
use aidaemon_tools::status::StatusUpdate;
use aidaemon_tools::{
    BackendKind, ExecutionRequest, ProcessHandle, backend::SharedExecutionBackend,
};

use super::command_patterns::{find_matching_pattern, record_approval, record_denial};
use super::semantic_command_risk::assess_command;
use super::terminal_confined::{self, ConfinedAccess};
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel, approval_floor_reason};
use aidaemon_tools::daemon_guard::detect_daemonization_primitives;
use aidaemon_tools::process_control::{send_sigkill, send_sigterm};

mod actions;
mod approvals;
mod background;
mod command_checks;
mod construct;
mod continuation;
mod output;
mod processes;
mod reaping;
mod receipts;
mod request;
mod run;
mod sandbox;
mod tool_impl;

pub use background::*;
use command_checks::*;
pub use continuation::*;
pub(crate) use output::*;
pub use reaping::*;
pub(crate) use receipts::*;
pub(crate) use request::*;
pub(crate) use sandbox::*;

pub use aidaemon_channels::hub::ApprovalRequest;

pub struct TerminalTool {
    /// Immutable target shared by terminal, file, Git, and CLI-agent tools.
    backend: SharedExecutionBackend,
    /// Permanently allowed prefixes (from config + DB)
    allowed_prefixes: Arc<RwLock<Vec<String>>>,
    /// Session-only allowed prefixes (cleared on restart), per chat
    /// session (`TerminalTrust::grant_session`): an "allow for session" in
    /// one chat covers that chat and its child agents, never another chat.
    session_approved: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    /// Elevated capabilities granted per command key (`network`,
    /// `read:<path>`, `write:<path>`), persisted by Allow-Always.
    capability_grants: Arc<RwLock<HashSet<(String, String)>>>,
    /// Elevated capabilities granted for one chat session only:
    /// (grant session, command key, capability).
    session_capability_grants: Arc<RwLock<HashSet<(String, String, String)>>>,
    /// AIdaemon's permission level: at Full access every command is
    /// allowed (the owner's `/trustall`). The daemon's shared value once
    /// [`Self::set_permissions`] wires it; until then this terminal's own.
    permissions: crate::permissions::Permissions,
    daemon_permissions: OnceLock<crate::permissions::Permissions>,
    /// Permission persistence mode: picks the buttons on the owner's cards.
    permission_mode: PermissionMode,
    /// Host vs sandbox execution.
    confinement: aidaemon_tools::TerminalConfinement,
    running: Arc<Mutex<HashMap<u32, RunningProcess>>>,
    running_by_dedupe_key: Arc<Mutex<HashMap<String, u32>>>,
    task_processes: Arc<Mutex<HashMap<String, HashSet<u32>>>>,
    completed: Arc<Mutex<HashMap<u32, CompletedProcess>>>,
    /// The trust judged for each background process when it was launched,
    /// by pid: `check` and `kill` output carries it; an unknown pid is
    /// untrusted. Daemon-held; the model can only name a pid.
    launch_trust: Arc<std::sync::Mutex<HashMap<u32, aidaemon_core::SourceTrust>>>,
    initial_timeout: Duration,
    max_output_chars: usize,
    pool: Option<SqlitePool>,
    event_store: Option<Arc<EventStore>>,
    /// Where a goal's run is looked up (its owner's session).
    goals: Option<Arc<dyn GoalStore>>,
    /// Progress notes for a session no chat reaches now.
    notifications: Option<Arc<dyn NotificationStore>>,
    hub: OnceLock<Weak<dyn OutboundRouter>>,
    /// Weak reference to the agent, used to re-engage the agent loop when
    /// a background terminal command completes so the agent can process the
    /// output and continue working on the original task.
    agent: OnceLock<Weak<dyn ConversationRuntime>>,
    /// Per-session timestamps of recent background-completion re-engagements,
    /// used by [`reengagement_allowed`] to cap runaway re-engagement loops.
    reengagements: Arc<Mutex<HashMap<String, std::collections::VecDeque<Instant>>>>,
    recent_background_deliveries: Arc<Mutex<HashMap<String, HashMap<String, Instant>>>>,
    /// Process-scoped deliver-once ledger keyed by `(session_id, canonical path)`.
    /// `Delivered` is recorded only after a real document/media send succeeds;
    /// failed sends remain retryable.
    delivered_deliverables: Arc<Mutex<HashMap<(String, String), DeliverableDeliveryState>>>,
    /// Durable plan store, used at background completion to read the session's
    /// incomplete checklist (for deliverable attribution + the conservative
    /// delivery-step write-back). Wired via [`set_plan_store`].
    plan_store: OnceLock<Arc<aidaemon_work::plans::PlanStore>>,
    /// The runtime's run records: a command moved to the background is a run
    /// there, shown on its turn's card. Wired via [`set_runs`].
    runs: OnceLock<Arc<crate::run_state::RunStates>>,
    /// Inbox directory for harness-side file delivery recovery-copy. Defaults to
    /// the system temp dir until wired via [`with_delivery_dirs`].
    inbox_dir: PathBuf,
    /// Allowed outbox directories for harness-side file delivery validation.
    outbox_dirs: Vec<PathBuf>,
    /// The gateway novel shell commands are assessed through, by their
    /// complete effects. Deterministic checks remain a non-downgradable safety
    /// floor; invalid/unavailable model assessments fail closed to approval.
    side_calls: OnceLock<crate::llm::gateway::SideCallGateway>,
}

#[cfg(test)]
mod tests;
