//! The ports between chat channels and the agent runtime: what a turn
//! hands the agent ([`InboundMessageRequest`]), what comes back
//! ([`AgentResponseEnvelope`]), the agent administration channel commands use
//! ([`ChannelAgentRuntime`]), and the outbound router the agent and tools
//! reach the chat through ([`OutboundRouter`]). Internal to aidaemon, no
//! stability promise; the public contract is [`crate::channel`].

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};

use aidaemon_events::conversation::MessageAttachment;
use aidaemon_tools::semantics::{ApprovalResponse, PermissionMode, RiskLevel, UserRole};
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::context::ChannelContext;
use crate::hub::MediaMessage;

/// Owned inbound turn delivered by a chat transport.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InboundMessageTiming {
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
    pub fn new(
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

    pub fn mark_queued(&mut self, at: chrono::DateTime<chrono::Utc>) {
        self.queue_entered_at = Some(at);
    }

    pub fn mark_dispatched_now(&mut self) {
        self.agent_dispatched_at = chrono::Utc::now();
    }
}

pub struct InboundMessageRequest {
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
pub struct AgentResponseEnvelope {
    pub response_id: String,
    pub task_id: String,
    pub turn_id: Option<String>,
    pub text: String,
    pub referenced_receipts: Vec<aidaemon_events::CompletionProofReference>,
    pub disposition: aidaemon_events::AssistantResponseDisposition,
}

impl AgentResponseEnvelope {
    pub fn delivery(
        &self,
        platform: &str,
        state: aidaemon_events::ResponseDeliveryState,
        platform_message_ids: Vec<String>,
        error_code: Option<String>,
    ) -> aidaemon_events::ResponseDeliveryData {
        debug_assert!(self
            .referenced_receipts
            .iter()
            .all(|reference| !reference.receipt_id.trim().is_empty()));
        aidaemon_events::ResponseDeliveryData {
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
pub trait AgentIngress: Send + Sync {
    async fn handle_inbound_message(
        &self,
        request: InboundMessageRequest,
    ) -> anyhow::Result<AgentResponseEnvelope>;

    async fn record_response_delivery(
        &self,
        _session_id: &str,
        _delivery: aidaemon_events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Agent administration surface used by shared channel commands.
#[async_trait]
pub trait ChannelAgentRuntime: AgentIngress {
    async fn terminalize_supervised_task(
        &self,
        session_id: &str,
        status: aidaemon_events::TaskStatus,
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
    async fn clear_session_context(&self, session_id: &str) -> anyhow::Result<()>;
    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()>;
}

/// Persistence hook used by outbound delivery without depending on `Agent`.
#[async_trait]
pub trait AssistantNoteSink: Send + Sync {
    async fn record_assistant_note(&self, session_id: &str, note: &str) -> anyhow::Result<()>;
}

/// Outbound operations shared by the agent runtime, tools, and heartbeat.
#[async_trait]
pub trait OutboundRouter: Send + Sync {
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
    /// Delete a message this daemon sent. `Ok(false)` when the channel
    /// cannot delete it; the default suits routers that never can.
    async fn delete_message(&self, _session_id: &str, _message_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }
    /// The plan card a turn left in the session's chat, if the channel keeps
    /// one; the default suits routers without plan cards.
    async fn checklist_surface(&self, _session_id: &str) -> Option<String> {
        None
    }
    /// Record a plan card sent outside a turn (see [`Self::checklist_surface`]).
    async fn register_checklist_surface(&self, _session_id: &str, _message_id: &str) {}
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
pub struct SessionRouteLease {
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
