//! The contract every chat channel implements.
//!
//! A channel is a transport: it delivers inbound events and sends, edits and
//! asks. It holds no agent logic. Authorization, sessions, queueing and
//! approvals live in the runner, so every channel behaves the same way.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Where a conversation happens. `id` is the channel's own identifier
/// (Telegram chat id, Slack `channel` or `channel:thread_ts`, Discord
/// channel id). Together with the channel name it forms the session id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub kind: ConversationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationKind {
    /// One-to-one with the bot.
    Direct,
    /// A group, channel or server channel.
    Group,
    /// A thread inside a group conversation.
    Thread,
}

/// Who sent a message, as the platform identifies them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sender {
    /// Stable platform user id (never a display name).
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// True if the platform marks the sender as a bot.
    #[serde(default)]
    pub is_bot: bool,
}

/// A file or image attached to an inbound message. Adapters fill what the
/// platform provides; the bytes are fetched lazily by the app if needed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Platform URL or file id to download from.
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InboundMessage {
    pub conversation: Conversation,
    pub sender: Sender,
    /// Platform message id, used for replies and edits.
    pub message_id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,
    /// The bot was addressed explicitly (mention, reply to the bot, or a
    /// slash command). Always true in direct conversations.
    pub addressed: bool,
}

/// A message to send. `text` is Markdown; adapters convert it to the
/// platform's dialect and split it to fit `max_message_len`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutboundMessage {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
}

impl OutboundMessage {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            reply_to: None,
        }
    }
}

/// Identifies a message the bot sent, for later edits.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageRef {
    pub conversation_id: String,
    pub message_id: String,
}

/// One button on an approval prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalChoice {
    /// Returned in [`ApprovalAnswer::choice_id`].
    pub id: String,
    pub label: String,
}

/// A question with buttons, used for tool approvals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalPrompt {
    /// Opaque id chosen by the runner; echoed back in the answer.
    pub id: String,
    pub conversation: Conversation,
    /// Markdown body describing what is being approved.
    pub text: String,
    pub choices: Vec<ApprovalChoice>,
}

/// A button press (or the text-command fallback) answering a prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalAnswer {
    pub prompt_id: String,
    pub choice_id: String,
    pub sender: Sender,
    pub conversation: Conversation,
}

/// Everything a channel reports to the runner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ChannelEvent {
    Message(InboundMessage),
    Approval(ApprovalAnswer),
}

/// What a channel supports; the runner adapts to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelCapabilities {
    pub edit_messages: bool,
    pub buttons: bool,
    pub typing_indicator: bool,
    pub threads: bool,
    /// Longest message the platform accepts, in characters.
    pub max_message_len: usize,
}

impl Default for ChannelCapabilities {
    fn default() -> Self {
        Self {
            edit_messages: false,
            buttons: false,
            typing_indicator: false,
            threads: false,
            max_message_len: 4000,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error("not supported by this channel: {0}")]
    Unsupported(&'static str),
    #[error("rate limited; retry after {retry_after_secs:?}s")]
    RateLimited { retry_after_secs: Option<u64> },
    #[error("authentication failed: {0}")]
    Auth(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("{0}")]
    Other(String),
}

/// A chat transport. Implementations translate platform updates into
/// [`ChannelEvent`]s and runner requests into platform API calls.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Short stable name, used as the session-id prefix (`telegram`,
    /// `slack`, `discord`, or a bot-specific name for multi-bot setups).
    fn name(&self) -> &str;

    fn capabilities(&self) -> ChannelCapabilities;

    /// Receive updates and forward them until `shutdown` is cancelled or the
    /// receiver is dropped. Reconnects on transient failures; returns an
    /// error only for unrecoverable ones (for example a revoked token).
    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError>;

    /// Send a message, splitting it if needed. Returns the last part's ref.
    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError>;

    /// Replace the text of a message the bot sent.
    async fn edit(&self, _message: &MessageRef, _text: &str) -> Result<(), ChannelError> {
        Err(ChannelError::Unsupported("edit"))
    }

    /// Show an approval prompt with buttons. Channels without buttons may
    /// leave the default, which the runner replaces with a text prompt and
    /// `/approve <id>` / `/deny <id>` commands.
    async fn ask(&self, _prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        Err(ChannelError::Unsupported("buttons"))
    }

    /// Show a "typing" indicator. Best effort.
    async fn typing(&self, _conversation: &Conversation) -> Result<(), ChannelError> {
        Ok(())
    }
}
