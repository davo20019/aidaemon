//! Narrow runtime interfaces used at subsystem boundaries.
//!
//! These ports keep tools and outbound channel routing independent from the
//! concrete `Agent` and `ChannelHub` implementations. Startup owns the only
//! place where those implementations are connected.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::events::TaskOutcome;
use crate::traits::AgentRole;
use crate::types::{ChannelContext, StatusUpdate, UserRole};

/// Result returned by a delegated child-agent run.
#[derive(Debug)]
pub(crate) struct ChildAgentRun {
    /// The child's own session.
    pub session: String,
    pub response: String,
    pub outcome: TaskOutcome,
    /// `Untrusted` when the child's run read (or started with) content
    /// someone other than the owner wrote.
    pub trust: aidaemon_core::SourceTrust,
    /// The child's own typed verdict was `done`, and its outcome is partial
    /// only because no receipt confirmed some of its listed steps: the owner
    /// is told it finished, and that its tools did not confirm every step.
    pub reported_done: bool,
}

/// A terminal task result recovered after a child run timed out.
#[derive(Debug)]
pub(crate) struct SalvagedTaskOutcome {
    pub status: String,
    pub details: String,
}

/// How long a child may go without progress (no model or tool call running
/// and no event of its own) before it is stopped. Never a limit on how long
/// it runs: a child that keeps working is bounded by its budgets.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChildTimeLimit {
    pub limit: std::time::Duration,
}

/// What a stopped child had already done, from its own tool records.
#[derive(Debug, Default)]
pub(crate) struct ChildWorkReport {
    /// Calls that finished and may have changed something.
    pub changed: Vec<String>,
    /// Calls still running when the child was stopped.
    pub unfinished: Vec<String>,
    /// The records could not be read in full (unreadable, or more than the
    /// report reads), so the lists may be missing actions.
    pub unknown: bool,
}

impl ChildWorkReport {
    pub(crate) fn may_have_changed_something(&self) -> bool {
        self.unknown || !self.changed.is_empty() || !self.unfinished.is_empty()
    }
}

/// The child was stopped by the owner from the chat (`/cancel`, or a stop
/// request read as one).
#[derive(Debug)]
pub(crate) struct ChildStoppedByOwner;

impl std::fmt::Display for ChildStoppedByOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("stopped by the owner")
    }
}

impl std::error::Error for ChildStoppedByOwner {}

/// A child made no progress for its idle limit and was stopped.
#[derive(Debug)]
pub(crate) struct ChildTimedOut {
    pub limit_secs: u64,
    pub report: ChildWorkReport,
}

impl std::fmt::Display for ChildTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Specialist timed out: no progress for {} seconds",
            self.limit_secs
        )
    }
}

impl std::error::Error for ChildTimedOut {}

impl ChildTimedOut {
    /// What the parent needs to decide whether running it again is safe.
    pub(crate) fn render(&self) -> String {
        if !self.report.may_have_changed_something() {
            return format!(
                "The sub-agent made no progress for {} seconds and was stopped. It only \
                 used read-only tools, so running it again is safe.",
                self.limit_secs
            );
        }
        let mut text = format!(
            "The sub-agent made no progress for {} seconds and was stopped, but some of its \
             work may already have taken effect. Do NOT run it again until you have checked \
             the state of the items below: repeating them could duplicate side effects.",
            self.limit_secs
        );
        if self.report.unknown {
            text.push_str("\n\nIts action records could not be read in full.");
        }
        if !self.report.changed.is_empty() {
            text.push_str("\n\nActions it completed:");
            for item in &self.report.changed {
                text.push_str("\n- ");
                text.push_str(item);
            }
        }
        if !self.report.unfinished.is_empty() {
            text.push_str("\n\nStill running when it was stopped (outcome unknown):");
            for item in &self.report.unfinished {
                text.push_str("\n- ");
                text.push_str(item);
            }
        }
        text
    }
}

/// Owned input for a child-agent run. Using an owned request keeps the port
/// object-safe and safe to move into background tasks.
pub(crate) struct ChildAgentRequest {
    pub mission: String,
    pub task: String,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub channel_ctx: ChannelContext,
    pub user_role: UserRole,
    pub child_role: Option<AgentRole>,
    pub goal_id: Option<String>,
    pub task_id: Option<String>,
    pub project_scope: Option<String>,
    pub specialist: Option<String>,
    pub approval_session_id: Option<String>,
    /// The spawning turn's actual authority, taint and root request
    /// (dispatcher-injected; `None` falls back to re-resolution).
    pub parent_authority: Option<crate::agent::authority::InheritedAuthority>,
    /// `None` leaves the child unlimited.
    pub time_limit: Option<ChildTimeLimit>,
    pub model_tier: Option<crate::router::Tier>,
}

/// Child-delegation capability required by `spawn_agent`.
#[async_trait]
pub(crate) trait ChildAgentRuntime: Send + Sync {
    fn depth(&self) -> usize;
    fn max_depth(&self) -> usize;
    fn role(&self) -> AgentRole;
    fn specialist_descriptions(&self) -> Vec<(String, String)>;

    /// The names of the tools this agent has, which bound what it can grant
    /// a child. `None`: not known (the grant is then checked only against
    /// the agent's own authority).
    fn tool_names(&self) -> Option<Vec<String>> {
        None
    }

    async fn validate_executor_task_for_spawn(
        &self,
        task_id: &str,
        expected_goal_id: Option<&str>,
    ) -> anyhow::Result<()>;

    async fn run_child(&self, request: ChildAgentRequest) -> anyhow::Result<ChildAgentRun>;

    /// The paths a child's declared scope names, as the file tools resolve
    /// them.
    async fn resolve_child_scope(&self, scope: Vec<String>) -> Vec<String> {
        scope
    }

    /// A token that stops the background child `id` when the owner cancels.
    async fn background_cancel_token(
        &self,
        _id: &str,
    ) -> Option<tokio_util::sync::CancellationToken> {
        None
    }

    /// Forget the token of a background child that has ended.
    async fn release_background_cancel_token(&self, _id: &str) {}

    /// The session a background child's result belongs to, given the session
    /// of the call that spawned it.
    fn background_delivery_session(&self, call_session: &str) -> String {
        call_session.to_string()
    }

    async fn salvage_executor_task_outcome(
        &self,
        task_id: &str,
        timeout_secs: u64,
    ) -> Option<SalvagedTaskOutcome>;

    async fn mark_executor_task_timeout(&self, task_id: &str, timeout_secs: u64);

    async fn deliver_background_child_result(
        &self,
        router: Option<&Weak<dyn OutboundRouter>>,
        parent_session_id: &str,
        text: &str,
        trust: aidaemon_core::SourceTrust,
    ) -> anyhow::Result<bool>;
}

/// Owned input for re-entering an existing conversation.
pub(crate) struct ConversationRequest {
    pub session_id: String,
    pub user_text: String,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub heartbeat: Option<Arc<AtomicU64>>,
    /// Typed lifecycle edge back to the task whose background receipt caused
    /// this continuation.
    pub parent_task_id: Option<String>,
    pub parent_tool_call_id: Option<String>,
    pub parent_result_id: Option<String>,
}

/// Minimal agent surface needed by background command/CLI completion paths.
#[async_trait]
pub(crate) trait ConversationRuntime: Send + Sync {
    async fn continue_conversation(
        &self,
        request: ConversationRequest,
    ) -> anyhow::Result<AgentResponseEnvelope>;

    async fn record_continuation_delivery(
        &self,
        session_id: &str,
        delivery: crate::events::ResponseDeliveryData,
    ) -> anyhow::Result<()>;
}

pub(crate) use aidaemon_channels::ports::{
    AgentIngress, AgentResponseEnvelope, AssistantNoteSink, ChannelAgentRuntime,
    InboundMessageRequest, InboundMessageTiming, OutboundRouter, SessionRouteLease,
};

/// The daemon's additions to [`ChannelAgentRuntime`]: what its own chat
/// glue (Slack project grants, `/reload`) needs from the agent.
#[async_trait]
pub(crate) trait DaemonAgentRuntime: ChannelAgentRuntime {
    /// Whether the file tools must not write `path` on this daemon.
    fn is_sensitive_path(&self, path: &std::path::Path) -> bool;
    /// The daemon's execution environment.
    fn execution_backend(&self) -> crate::execution::SharedExecutionBackend;
    async fn reload_provider(&self, config: &crate::config::AppConfig) -> anyhow::Result<String>;
}
