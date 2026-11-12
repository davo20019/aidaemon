//! The daemon's outbound surface to a bot: where a session's messages,
//! media, approvals and goal confirmations go, plus the runtime-managed bot
//! and session-routing stores. Internal to aidaemon (moves with the turn
//! runner in Phase 9.5), no stability promise; the public contract is
//! [`crate::channel`].

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use aidaemon_tools::semantics::{
    ApprovalResponse, GoalConfirmationStyle, PermissionMode, RiskLevel,
};

/// The kind of media being sent.
#[allow(dead_code)]
pub enum MediaKind {
    /// An in-memory photo (e.g. screenshot).
    Photo { data: Vec<u8> },
    /// A file on disk to send as a document.
    Document { file_path: String, filename: String },
}

/// A media message to be sent through a channel.
pub struct MediaMessage {
    pub session_id: String,
    pub caption: String,
    pub kind: MediaKind,
    /// Optional one-shot used to report the ACTUAL delivery outcome back to the
    /// enqueuing tool. `Some(tx)` means the sender wants honest confirmation:
    /// `media_listener` replies `Ok(())` only when the channel's `send_media`
    /// succeeded, and `Err(reason)` when delivery failed / no channel was found /
    /// the message was shed under overload. `None` keeps the legacy
    /// fire-and-forget behavior (the sender does not care about the outcome).
    ///
    /// `oneshot::Sender` is neither `Clone` nor `Default`; `MediaMessage` is moved
    /// (never cloned) through the media channel, so no derive is affected.
    pub result_tx: Option<tokio::sync::oneshot::Sender<Result<(), String>>>,
}

/// Capabilities that vary by channel (Telegram, WhatsApp, SMS, Web, etc.).
///
/// Used by the agent and hub to adapt output format for each channel.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ChannelCapabilities {
    /// Whether the channel supports markdown/rich text formatting.
    pub markdown: bool,
    /// Whether the channel supports inline buttons (e.g., Telegram inline keyboards).
    pub inline_buttons: bool,
    /// Whether the channel supports sending media (photos, files).
    pub media: bool,
    /// Maximum message length in characters. Messages longer than this will be split.
    pub max_message_len: usize,
}

/// A bot as the channel hub sees it: where
/// a session's messages, media, approvals and goal confirmations go.
///
/// Chat platforms implement it on top of an SDK transport
/// (`aidaemon_channels::Channel`), which carries the traffic; this trait is
/// the daemon's outbound surface, not a transport.
#[async_trait]
pub trait HubChannel: Send + Sync {
    /// Unique name for this channel (e.g., "telegram", "telegram:my_bot", "discord").
    /// For multi-bot setups, includes the bot username (e.g., "telegram:coding_bot").
    fn name(&self) -> String;

    /// Channel capabilities — used to adapt output format.
    fn capabilities(&self) -> ChannelCapabilities;

    /// Send a text message to a session.
    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()>;

    /// Send a text message and return a channel-specific message id that can be
    /// passed to [`HubChannel::edit_text`] later. Channels that support in-place
    /// editing (e.g. Telegram) override this; the default sends via
    /// [`HubChannel::send_text`] and returns `None` (no tracking), so callers fall
    /// back to non-editing UX.
    async fn send_text_tracked(
        &self,
        session_id: &str,
        text: &str,
    ) -> anyhow::Result<Option<String>> {
        self.send_text(session_id, text).await?;
        Ok(None)
    }

    /// Edit a previously-sent message in place. Returns `Ok(true)` if the edit
    /// was applied (or the content was already current), `Ok(false)` if the
    /// channel does not support editing. Default: unsupported.
    async fn edit_text(
        &self,
        _session_id: &str,
        _message_id: &str,
        _text: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Delete a previously-sent message. Channels that support deletion
    /// override this; the default is unsupported.
    async fn delete_message(&self, _session_id: &str, _message_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Send media (photo/file) to a session.
    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()>;

    /// Request user approval for a command. Blocks until the user responds.
    /// Channels without inline buttons should fall back to text-based approval
    /// (e.g., "Reply YES, ALWAYS, or NO").
    ///
    /// The `risk_level`, `warnings`, and `permission_mode` parameters provide
    /// context about why approval is being requested and which buttons to show.
    /// `one_time_only` suppresses session/persistent choices for point-of-action
    /// approvals whose grant is deliberately not reusable.
    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
        one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse>;

    /// Request user confirmation for a scheduled goal. Blocks until the user
    /// responds. Returns `true` for confirm, `false` for cancel.
    ///
    /// Channels with inline buttons should show [Confirm ✅ | Cancel ❌].
    /// Default implementation auto-confirms (for channels without button support).
    async fn request_goal_confirmation(
        &self,
        _session_id: &str,
        _goal_description: &str,
        _details: &[String],
        style: GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        // Preserve the historical fallback for ordinary scheduled goals, but
        // never auto-confirm an Autopilot posture on a channel that has not
        // implemented an explicit confirmation surface.
        Ok(matches!(style, GoalConfirmationStyle::Standard))
    }
}

/// A dynamically added bot configuration (stored in database).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBot {
    pub id: i64,
    pub channel_type: String, // "telegram", "discord", "slack"
    pub bot_token: String,
    pub app_token: Option<String>,     // Only for Slack
    pub allowed_user_ids: Vec<String>, // Stored as JSON
    pub extra_config: String,          // JSON for channel-specific settings
    pub created_at: String,
}

/// Dynamic bots (runtime-managed) persistence.
#[async_trait]
pub trait DynamicBotStore: Send + Sync {
    /// Store a dynamically added bot configuration.
    async fn add_dynamic_bot(&self, _bot: &DynamicBot) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamically added bots.
    async fn get_dynamic_bots(&self) -> anyhow::Result<Vec<DynamicBot>> {
        Ok(vec![])
    }

    /// Update the allowed_user_ids for a dynamic bot identified by its token.
    #[allow(dead_code)]
    async fn update_dynamic_bot_allowed_users(
        &self,
        _bot_token: &str,
        _allowed_user_ids: &[String],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete a dynamic bot by ID.
    #[allow(dead_code)]
    async fn delete_dynamic_bot(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Session → channel mapping persistence.
#[async_trait]
pub trait SessionChannelStore: Send + Sync {
    /// Persist a session_id → channel_name mapping so it survives restarts.
    async fn save_session_channel(
        &self,
        _session_id: &str,
        _channel_name: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Load all persisted session → channel mappings (for populating session_map on startup).
    async fn load_session_channels(&self) -> anyhow::Result<Vec<(String, String)>> {
        Ok(vec![])
    }
}
