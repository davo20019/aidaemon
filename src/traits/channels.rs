use async_trait::async_trait;

use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::types::{ApprovalResponse, MediaMessage};

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

/// A communication channel (Telegram, WhatsApp, Web, SMS, etc.).
///
/// Each implementation handles transport-specific details for sending messages,
/// media, and approval requests. New channels (e.g., Discord, Slack, SMS) only
/// need to implement this trait to integrate with aidaemon.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Unique name for this channel (e.g., "telegram", "telegram:my_bot", "discord").
    /// For multi-bot setups, includes the bot username (e.g., "telegram:coding_bot").
    fn name(&self) -> String;

    /// Channel capabilities — used to adapt output format.
    fn capabilities(&self) -> ChannelCapabilities;

    /// Send a text message to a session.
    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()>;

    /// Send media (photo/file) to a session.
    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()>;

    /// Request user approval for a command. Blocks until the user responds.
    /// Channels without inline buttons should fall back to text-based approval
    /// (e.g., "Reply YES, ALWAYS, or NO").
    ///
    /// The `risk_level`, `warnings`, and `permission_mode` parameters provide
    /// context about why approval is being requested and which buttons to show.
    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
    ) -> anyhow::Result<ApprovalResponse>;
}

