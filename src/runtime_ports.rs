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
use crate::traits::MessageAttachment;
use crate::types::{
    ApprovalResponse, ChannelContext, MediaMessage, PermissionMode, RiskLevel, StatusUpdate,
    UserRole,
};

/// Result returned by a delegated child-agent run.
#[derive(Debug)]
pub(crate) struct ChildAgentRun {
    pub response: String,
    pub outcome: TaskOutcome,
    /// `Untrusted` when the child's run read (or started with) content
    /// someone other than the owner wrote.
    pub trust: aidaemon_core::SourceTrust,
}

/// A terminal task result recovered after a child run timed out.
#[derive(Debug)]
pub(crate) struct SalvagedTaskOutcome {
    pub status: String,
    pub details: String,
}

/// How long a child may work.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChildTimeLimit {
    pub limit: std::time::Duration,
    /// The child outlives the call that spawned it (background), so the
    /// spawning agent's own limit does not bound it.
    pub detached: bool,
}

/// What a stopped child had already done, from its own tool records.
#[derive(Debug, Default)]
pub(crate) struct ChildWorkReport {
    /// Calls that finished and may have changed something.
    pub changed: Vec<String>,
    /// Calls still running when the child was stopped.
    pub unfinished: Vec<String>,
    /// The records could not be read, so nothing is known.
    pub unknown: bool,
}

impl ChildWorkReport {
    pub(crate) fn may_have_changed_something(&self) -> bool {
        self.unknown || !self.changed.is_empty() || !self.unfinished.is_empty()
    }
}

/// A child ran out of working time and was stopped.
#[derive(Debug)]
pub(crate) struct ChildTimedOut {
    pub limit_secs: u64,
    pub report: ChildWorkReport,
}

impl std::fmt::Display for ChildTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Specialist timed out after {} seconds", self.limit_secs)
    }
}

impl std::error::Error for ChildTimedOut {}

impl ChildTimedOut {
    /// What the parent needs to decide whether running it again is safe.
    pub(crate) fn render(&self) -> String {
        if !self.report.may_have_changed_something() {
            return format!(
                "The sub-agent ran out of time after {} seconds. It only used read-only \
                 tools, so running it again is safe.",
                self.limit_secs
            );
        }
        let mut text = format!(
            "The sub-agent ran out of time after {} seconds and was stopped, but some of its \
             work may already have taken effect. Do NOT run it again until you have checked \
             the state of the items below: repeating them could duplicate side effects.",
            self.limit_secs
        );
        if self.report.unknown {
            text.push_str("\n\nIts action records could not be read.");
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

    async fn validate_executor_task_for_spawn(
        &self,
        task_id: &str,
        expected_goal_id: Option<&str>,
    ) -> anyhow::Result<()>;

    async fn run_child(&self, request: ChildAgentRequest) -> anyhow::Result<ChildAgentRun>;

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

/// Owned inbound turn delivered by a chat transport.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct InboundMessageTiming {
    /// Timestamp assigned by the source platform, when available.
    pub platform_message_at: Option<chrono::DateTime<chrono::Utc>>,
    /// First instant the daemon transport handler observed the message.
    pub transport_received_at: chrono::DateTime<chrono::Utc>,
    /// Instant the message entered a per-session queue, if it waited.
    pub queue_entered_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Instant the transport handed the turn to the agent runtime.
    pub agent_dispatched_at: chrono::DateTime<chrono::Utc>,
}

impl InboundMessageTiming {
    pub(crate) fn new(
        platform_message_at: Option<chrono::DateTime<chrono::Utc>>,
        transport_received_at: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        Self {
            platform_message_at,
            transport_received_at,
            queue_entered_at: None,
            agent_dispatched_at: transport_received_at,
        }
    }

    pub(crate) fn mark_queued(&mut self, at: chrono::DateTime<chrono::Utc>) {
        self.queue_entered_at = Some(at);
    }

    pub(crate) fn mark_dispatched_now(&mut self) {
        self.agent_dispatched_at = chrono::Utc::now();
    }
}

pub(crate) struct InboundMessageRequest {
    pub session_id: String,
    pub user_text: String,
    pub attachments: Vec<MessageAttachment>,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub heartbeat: Option<Arc<AtomicU64>>,
    pub ingress_timing: Option<InboundMessageTiming>,
}

#[derive(Debug, Clone)]
pub(crate) struct AgentResponseEnvelope {
    pub response_id: String,
    pub task_id: String,
    pub turn_id: Option<String>,
    pub text: String,
    pub referenced_receipts: Vec<crate::events::CompletionProofReference>,
    pub disposition: crate::events::AssistantResponseDisposition,
}

impl AgentResponseEnvelope {
    pub(crate) fn delivery(
        &self,
        platform: &str,
        state: crate::events::ResponseDeliveryState,
        platform_message_ids: Vec<String>,
        error_code: Option<String>,
    ) -> crate::events::ResponseDeliveryData {
        debug_assert!(self
            .referenced_receipts
            .iter()
            .all(|reference| !reference.receipt_id.trim().is_empty()));
        crate::events::ResponseDeliveryData {
            response_id: self.response_id.clone(),
            task_id: self.task_id.clone(),
            turn_id: self.turn_id.clone(),
            platform: platform.to_string(),
            state,
            platform_message_ids,
            error_code,
            occurred_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// The only agent capability chat transports need.
#[async_trait]
pub(crate) trait AgentIngress: Send + Sync {
    async fn handle_inbound_message(
        &self,
        request: InboundMessageRequest,
    ) -> anyhow::Result<AgentResponseEnvelope>;

    async fn record_response_delivery(
        &self,
        _session_id: &str,
        _delivery: crate::events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Agent administration surface used by shared channel commands.
#[async_trait]
pub(crate) trait ChannelAgentRuntime: AgentIngress {
    async fn terminalize_supervised_task(
        &self,
        session_id: &str,
        status: crate::events::TaskStatus,
        error: String,
    ) -> anyhow::Result<Option<String>>;
    async fn cancel_active_finite_work_for_session(&self, session_id: &str) -> Vec<String>;
    async fn current_model(&self) -> String;
    async fn context_debug_settings(
        &self,
        session_id: &str,
        model: &str,
    ) -> (bool, usize, usize, usize, Option<i64>);
    async fn set_model(&self, model: String);
    async fn list_models(&self) -> anyhow::Result<Vec<String>>;
    async fn clear_model_override(&self);
    async fn reload_provider(&self, config: &crate::config::AppConfig) -> anyhow::Result<String>;
    async fn clear_session_context(&self, session_id: &str) -> anyhow::Result<()>;
    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()>;
}

/// Persistence hook used by outbound delivery without depending on `Agent`.
#[async_trait]
pub(crate) trait AssistantNoteSink: Send + Sync {
    async fn record_assistant_note(&self, session_id: &str, note: &str) -> anyhow::Result<()>;
}

/// Outbound operations shared by the agent runtime, tools, and heartbeat.
#[async_trait]
pub(crate) trait OutboundRouter: Send + Sync {
    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()>;
    async fn send_text_tracked(
        &self,
        session_id: &str,
        text: &str,
    ) -> anyhow::Result<Option<String>>;
    async fn edit_text(
        &self,
        session_id: &str,
        message_id: &str,
        text: &str,
    ) -> anyhow::Result<bool>;
    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()>;
    async fn send_media_strict(&self, session_id: &str, media: &MediaMessage)
        -> anyhow::Result<()>;
    async fn background_status_surface(&self, session_id: &str) -> Option<String>;
    async fn take_background_status_surface(&self, session_id: &str) -> Option<String>;
    async fn request_inline_approval(
        &self,
        session_id: &str,
        description: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
    ) -> anyhow::Result<ApprovalResponse>;
    async fn register_session_route(&self, child_session: &str, parent_session: &str) -> bool;
    async fn unregister_session_route(&self, child_session: &str, parent_session: &str);
}

/// Cancellation-safe route lifetime that is independent of the concrete hub.
pub(crate) struct SessionRouteLease {
    router: Weak<dyn OutboundRouter>,
    child_session: String,
    parent_session: String,
}

impl SessionRouteLease {
    pub async fn register(
        router: &Arc<dyn OutboundRouter>,
        child_session: &str,
        parent_session: &str,
    ) -> Option<Self> {
        if !router
            .register_session_route(child_session, parent_session)
            .await
        {
            return None;
        }
        Some(Self {
            router: Arc::downgrade(router),
            child_session: child_session.to_string(),
            parent_session: parent_session.to_string(),
        })
    }
}

impl Drop for SessionRouteLease {
    fn drop(&mut self) {
        let Some(router) = self.router.upgrade() else {
            return;
        };
        let child_session = self.child_session.clone();
        let parent_session = self.parent_session.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                router
                    .unregister_session_route(&child_session, &parent_session)
                    .await;
            });
        }
    }
}
