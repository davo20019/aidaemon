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
}

/// A terminal task result recovered after a child run timed out.
#[derive(Debug)]
pub(crate) struct SalvagedTaskOutcome {
    pub status: String,
    pub details: String,
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
}

/// Minimal agent surface needed by background command/CLI completion paths.
#[async_trait]
pub(crate) trait ConversationRuntime: Send + Sync {
    async fn continue_conversation(&self, request: ConversationRequest) -> anyhow::Result<String>;
}

/// Owned inbound turn delivered by a chat transport.
pub(crate) struct InboundMessageRequest {
    pub session_id: String,
    pub user_text: String,
    pub attachments: Vec<MessageAttachment>,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub heartbeat: Option<Arc<AtomicU64>>,
}

#[derive(Debug, Clone)]
pub(crate) struct AgentResponseEnvelope {
    pub response_id: String,
    pub task_id: String,
    pub turn_id: Option<String>,
    pub text: String,
    pub referenced_receipts: Vec<crate::events::CompletionProofReference>,
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
    async fn cancel_active_goals_for_session(&self, session_id: &str) -> Vec<String>;
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
