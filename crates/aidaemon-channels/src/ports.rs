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

/// What a message sent while a run is going is about, as the model read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidRunIntent {
    /// A question about the running work: how it is going, what it found,
    /// whether a result is ready. Answered now from the work's typed live
    /// state (never in the model's words); the run goes on undisturbed.
    AboutRunningWork,
    /// It changes or adds to the running work: the run takes it in.
    ChangesWork,
    /// Something else: it runs as its own turn once the current one ends.
    Unrelated,
    /// It asks to stop running work: the items [`MidRunReading::stop`]
    /// names, or, when it names none, what bare `/cancel` would stop.
    StopsWork,
}

/// One helper (sub-agent) of the running work, as its progress last said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperProgress {
    /// The helper's job, in the words it was given.
    pub label: String,
    /// Its own progress line ("step 2 of 4: ...").
    pub progress: String,
    /// How long ago that progress came.
    pub since: std::time::Duration,
    /// How long it has been running (since its first progress). The reply
    /// says the running list's time instead when the helper is on it, the
    /// clock `/tasks` reads.
    pub running_for: std::time::Duration,
    /// The running turn started it (an earlier turn's helper runs on in the
    /// background).
    pub this_turn: bool,
}

/// The running work as its typed live state shows it, for answering a
/// question about it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunningWork {
    /// The request the run is working on.
    pub request: String,
    /// How long it has been running.
    pub elapsed: std::time::Duration,
    /// Its latest step and how long ago it started.
    pub last_step: Option<(String, std::time::Duration)>,
    /// Its plan card (checklist), as shown in chat.
    pub plan: Option<String>,
    /// Its helpers still reporting progress.
    pub helpers: Vec<HelperProgress>,
    /// Everything running for the chat, numbered as `/tasks` shows it.
    pub running: Vec<RunningItem>,
}

/// One thing running for a chat, as `/tasks` numbers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunningItem {
    pub number: u64,
    pub title: String,
    pub elapsed: std::time::Duration,
    /// What it is: `None` for a chat turn.
    pub work: Option<ChatWorkKind>,
}

/// How a message sent mid-run is handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MidRunReading {
    pub intent: MidRunIntent,
    /// For [`MidRunIntent::StopsWork`]: the numbers of the
    /// [`RunningWork::running`] items to stop; empty for what bare `/cancel`
    /// stops.
    pub stop: Vec<u64>,
}

/// A sub-agent running for a chat, as `/status` and `/tasks` list it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunningChild {
    /// The title its parent gave it (the spawn's mission, first line).
    pub label: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

/// What kind of work runs for a chat besides its turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatWorkKind {
    SubAgent,
    BackgroundJob,
}

/// One piece of work running for a chat outside its turn: a sub-agent or a
/// background command, read from the agent's typed state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatWork {
    /// The agent's handle for it, stable while it runs; never shown.
    pub key: String,
    pub kind: ChatWorkKind,
    /// What it is, in plain words.
    pub title: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// The request (turn) that started it, when known; opaque.
    pub request: Option<String>,
}

/// Everything running for a chat besides its turns.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatWorkView {
    /// The request the chat's running turn is working on, if one runs.
    pub current_request: Option<String>,
    pub items: Vec<ChatWork>,
}

/// Agent administration surface used by shared channel commands.
#[async_trait]
pub trait ChannelAgentRuntime: AgentIngress {
    /// Read a message the owner sent while `work` runs in `session_id`.
    /// `None` (the default, or when the reading failed) hands it to the
    /// running work, as a steering message.
    async fn read_mid_run_message(
        &self,
        _session_id: &str,
        _text: &str,
        _work: &RunningWork,
    ) -> Option<MidRunReading> {
        None
    }

    /// The sub-agents running now for `session_id` (waited on by a turn or
    /// in the background), oldest first.
    async fn running_children(&self, _session_id: &str) -> Vec<RunningChild> {
        Vec::new()
    }

    /// Everything running for `session_id` besides its turns: sub-agents
    /// and background commands.
    async fn chat_work(&self, _session_id: &str) -> ChatWorkView {
        ChatWorkView::default()
    }

    /// Stop the chat's work `key` (from [`Self::chat_work`]) and read its
    /// state back: `true` only once it is verified gone.
    async fn stop_chat_work(&self, _session_id: &str, _key: &str) -> bool {
        false
    }

    /// Cancel the chat's finite goals (task leads and their executors);
    /// returns what was cancelled. Its sub-agents and background commands are
    /// stopped through [`Self::stop_chat_work`].
    async fn cancel_finite_goals_for_session(&self, session_id: &str) -> Vec<String> {
        self.cancel_active_finite_work_for_session(session_id).await
    }
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
