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

/// Largest download [`Channel::download`] accepts unless an adapter documents
/// a smaller platform limit. Use [`Channel::download_limited`] for your own.
pub const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 100 * 1024 * 1024;

/// What an [`Attachment`] holds, as far as the platform says.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentKind {
    Image,
    Audio,
    Video,
    /// Anything else, including a kind this version does not know.
    #[default]
    #[serde(other)]
    File,
}

impl AttachmentKind {
    /// The kind a MIME type declares (`image/png` is [`Self::Image`]).
    pub fn from_mime(mime: &str) -> Self {
        let top = mime.split('/').next().unwrap_or("").trim();
        if top.eq_ignore_ascii_case("image") {
            Self::Image
        } else if top.eq_ignore_ascii_case("audio") {
            Self::Audio
        } else if top.eq_ignore_ascii_case("video") {
            Self::Video
        } else {
            Self::File
        }
    }

    /// The kind a file name's extension suggests, for platforms that do not
    /// always report a MIME type.
    fn from_name(name: &str) -> Self {
        let extension = name.rsplit_once('.').map_or("", |(_, ext)| ext);
        match extension.to_ascii_lowercase().as_str() {
            "png" | "jpg" | "jpeg" | "gif" | "webp" => Self::Image,
            "mp3" | "ogg" | "oga" | "wav" | "flac" | "m4a" | "aac" => Self::Audio,
            "mp4" | "mov" | "webm" | "mkv" => Self::Video,
            _ => Self::File,
        }
    }
}

/// A file or image attached to an inbound message. Adapters fill what the
/// platform provides; the bytes are fetched lazily with [`Channel::download`].
///
/// Built with [`Attachment::new`] and the `with_` methods, so a new field
/// does not break adapters written against an older version.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Platform URL or file id to download from.
    pub source: String,
    /// What it holds. The platform's word: check the bytes before relying
    /// on it.
    #[serde(default)]
    pub kind: AttachmentKind,
}

impl Attachment {
    /// An attachment named `name`, fetched from `source`. Its kind comes
    /// from the name's extension until [`Self::with_mime_type`] or
    /// [`Self::with_kind`] says better.
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        let name = name.into();
        let kind = AttachmentKind::from_name(&name);
        Self {
            name,
            mime_type: None,
            size_bytes: None,
            source: source.into(),
            kind,
        }
    }

    /// Set the MIME type (when known); its kind replaces the one guessed
    /// from the name.
    pub fn with_mime_type(mut self, mime_type: Option<String>) -> Self {
        if let Some(mime) = mime_type.as_deref() {
            self.kind = AttachmentKind::from_mime(mime);
        }
        self.mime_type = mime_type;
        self
    }

    pub fn with_size_bytes(mut self, size_bytes: Option<u64>) -> Self {
        self.size_bytes = size_bytes;
        self
    }

    /// Set the kind directly, for a platform that says what it is.
    pub fn with_kind(mut self, kind: AttachmentKind) -> Self {
        self.kind = kind;
        self
    }
}

/// Built with [`InboundMessage::new`] and the `with_` methods, so a new field
/// does not break adapters written against an older version.
#[non_exhaustive]
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
    /// Platform id of the message this one answers: a reply's target, or
    /// the root of the Slack thread it sits in. `None` for a message that
    /// answers nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_message_id: Option<String>,
    /// The platform's thread this message is in (Telegram forum topic id,
    /// Slack `thread_ts`, Discord thread channel id). `None` outside a
    /// thread, including a top-level Slack message the bot will answer in a
    /// new thread (that thread is only its [`Conversation`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
}

impl InboundMessage {
    /// A message with no attachments and no parent. `addressed` starts true
    /// in a direct conversation and false anywhere else.
    pub fn new(
        conversation: Conversation,
        sender: Sender,
        message_id: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        let addressed = conversation.kind == ConversationKind::Direct;
        Self {
            conversation,
            sender,
            message_id: message_id.into(),
            text: text.into(),
            attachments: Vec::new(),
            addressed,
            parent_message_id: None,
            thread_id: None,
        }
    }

    pub fn with_attachments(mut self, attachments: Vec<Attachment>) -> Self {
        self.attachments = attachments;
        self
    }

    pub fn with_addressed(mut self, addressed: bool) -> Self {
        self.addressed = addressed;
        self
    }

    pub fn with_parent_message_id(mut self, parent: Option<String>) -> Self {
        self.parent_message_id = parent;
        self
    }

    pub fn with_thread_id(mut self, thread: Option<String>) -> Self {
        self.thread_id = thread;
        self
    }
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
///
/// Built with [`ApprovalAnswer::new`], so a new field does not break adapters
/// written against an older version.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalAnswer {
    pub prompt_id: String,
    pub choice_id: String,
    pub sender: Sender,
    pub conversation: Conversation,
    /// The message whose button was pressed, so its buttons can be removed
    /// with [`Channel::remove_buttons`] even when the prompt is no longer
    /// known (it expired, or the process restarted). `None` when the
    /// platform did not say, or the answer was typed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<MessageRef>,
}

impl ApprovalAnswer {
    pub fn new(
        prompt_id: impl Into<String>,
        choice_id: impl Into<String>,
        sender: Sender,
        conversation: Conversation,
    ) -> Self {
        Self {
            prompt_id: prompt_id.into(),
            choice_id: choice_id.into(),
            sender,
            conversation,
            message: None,
        }
    }

    /// Record the message whose button was pressed.
    pub fn with_message(mut self, message: MessageRef) -> Self {
        self.message = Some(message);
        self
    }
}

/// A command the app offers in the platform's own command menu (Discord and
/// Slack slash commands, Telegram's `/` menu).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    /// Without the leading `/`; lowercase letters, digits and `_`.
    pub name: String,
    pub description: String,
}

/// A command invoked through the platform's command UI rather than typed
/// as text (a typed `/command` arrives as an [`InboundMessage`]). Answer it
/// with [`Channel::answer_command`]: some platforms show the invocation as
/// pending until it is answered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandInvocation {
    pub conversation: Conversation,
    pub sender: Sender,
    /// Without the leading `/`.
    pub name: String,
    /// Everything after the command name, trimmed.
    pub args: String,
    /// Adapter-defined token for answering this invocation.
    pub handle: String,
}

impl CommandInvocation {
    /// The invocation as the text a user would have typed.
    pub fn as_text(&self) -> String {
        if self.args.is_empty() {
            format!("/{}", self.name)
        } else {
            format!("/{} {}", self.name, self.args)
        }
    }
}

/// Everything a channel reports to the runner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ChannelEvent {
    Message(InboundMessage),
    Approval(ApprovalAnswer),
    Command(CommandInvocation),
}

/// A file to send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutboundFile {
    /// File name shown to the recipient.
    pub name: String,
    pub bytes: Vec<u8>,
    pub mime_type: Option<String>,
    /// Markdown caption sent with the file.
    pub caption: Option<String>,
}

/// What a channel supports; the runner adapts to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelCapabilities {
    pub edit_messages: bool,
    pub buttons: bool,
    pub typing_indicator: bool,
    pub threads: bool,
    /// [`Channel::delete`] works.
    pub delete_messages: bool,
    /// [`Channel::send_file`], [`Channel::download`] and
    /// [`Channel::download_limited`] work.
    pub files: bool,
    /// The platform has a command menu ([`Channel::register_commands`]).
    pub commands: bool,
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
            delete_messages: false,
            files: false,
            commands: false,
            max_message_len: 4000,
        }
    }
}

#[non_exhaustive]
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
    /// A download was over the size it was allowed.
    #[error("file is over the {limit_bytes} byte limit")]
    TooLarge { limit_bytes: u64 },
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

    /// Replace the text of a message the bot sent, and remove any buttons
    /// on it. An edit always stays one message: text too long for the
    /// platform is cut at the end and marked with
    /// [`crate::split::TRUNCATION_MARK`] (see
    /// [`crate::split::truncate_for_edit`]), never split into further
    /// messages and never an error.
    async fn edit(&self, _message: &MessageRef, _text: &str) -> Result<(), ChannelError> {
        Err(ChannelError::Unsupported("edit"))
    }

    /// Show an approval prompt with buttons. Channels without buttons may
    /// leave the default, which the runner replaces with a text prompt and
    /// `/approve <id>` / `/deny <id>` commands.
    async fn ask(&self, _prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        Err(ChannelError::Unsupported("buttons"))
    }

    /// Remove the buttons from a message, leaving its text. Best effort:
    /// the default does nothing, for channels without buttons.
    async fn remove_buttons(&self, _message: &MessageRef) -> Result<(), ChannelError> {
        Ok(())
    }

    /// Show a "typing" indicator. Best effort.
    async fn typing(&self, _conversation: &Conversation) -> Result<(), ChannelError> {
        Ok(())
    }

    /// Delete a message the bot sent.
    async fn delete(&self, _message: &MessageRef) -> Result<(), ChannelError> {
        Err(ChannelError::Unsupported("delete"))
    }

    /// Send a file, with an optional caption.
    async fn send_file(
        &self,
        _conversation: &Conversation,
        _file: &OutboundFile,
    ) -> Result<MessageRef, ChannelError> {
        Err(ChannelError::Unsupported("files"))
    }

    /// Fetch the bytes of an inbound attachment, up to the adapter's
    /// platform limit or else [`DEFAULT_MAX_DOWNLOAD_BYTES`]. An adapter
    /// keeps its token to the platform's own hosts and out of its errors
    /// and logs.
    async fn download(&self, _attachment: &Attachment) -> Result<Vec<u8>, ChannelError> {
        Err(ChannelError::Unsupported("files"))
    }

    /// [`Self::download`], refused with [`ChannelError::TooLarge`] over
    /// `max_bytes`.
    ///
    /// The default holds any adapter to the limit, but learns the true size
    /// only after [`Self::download`] returns. An adapter that can stop
    /// reading part way through should override it.
    async fn download_limited(
        &self,
        attachment: &Attachment,
        max_bytes: u64,
    ) -> Result<Vec<u8>, ChannelError> {
        let too_large = ChannelError::TooLarge {
            limit_bytes: max_bytes,
        };
        if attachment.size_bytes.is_some_and(|size| size > max_bytes) {
            return Err(too_large);
        }
        let bytes = self.download(attachment).await?;
        if bytes.len() as u64 > max_bytes {
            return Err(too_large);
        }
        Ok(bytes)
    }

    /// Offer `commands` in the platform's command menu, replacing what was
    /// offered before. A platform whose commands are configured outside the
    /// API (Slack's app manifest) accepts the call and does nothing.
    async fn register_commands(&self, _commands: &[CommandSpec]) -> Result<(), ChannelError> {
        Ok(())
    }

    /// The name a person shows on the platform, by their platform user id
    /// (`None` when the platform has none to give).
    async fn display_name(&self, _user_id: &str) -> Result<Option<String>, ChannelError> {
        Ok(None)
    }

    /// Answer a [`CommandInvocation`]. The default sends `text` to its
    /// conversation.
    async fn answer_command(
        &self,
        command: &CommandInvocation,
        text: &str,
    ) -> Result<(), ChannelError> {
        self.send(&command.conversation, &OutboundMessage::text(text))
            .await
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mime_type_outranks_the_name() {
        assert_eq!(Attachment::new("a.png", "s").kind, AttachmentKind::Image);
        assert_eq!(Attachment::new("a.MP3", "s").kind, AttachmentKind::Audio);
        assert_eq!(Attachment::new("a.mov", "s").kind, AttachmentKind::Video);
        assert_eq!(Attachment::new("a.pdf", "s").kind, AttachmentKind::File);
        assert_eq!(Attachment::new("noext", "s").kind, AttachmentKind::File);
        let relabelled = Attachment::new("a.png", "s").with_mime_type(Some("text/plain".into()));
        assert_eq!(relabelled.kind, AttachmentKind::File);
        let unlabelled = Attachment::new("a.png", "s").with_mime_type(None);
        assert_eq!(unlabelled.kind, AttachmentKind::Image);
    }

    #[test]
    fn older_or_newer_json_still_reads() {
        // Written before `kind` and the links existed.
        let old: InboundMessage = serde_json::from_value(serde_json::json!({
            "conversation": {"id": "c", "kind": "direct"},
            "sender": {"id": "synthetic-user-1"},
            "message_id": "m1",
            "text": "hi",
            "attachments": [{"name": "a.bin", "source": "s"}],
            "addressed": true,
        }))
        .unwrap();
        assert_eq!(old.attachments[0].kind, AttachmentKind::File);
        assert_eq!(old.parent_message_id, None);
        assert_eq!(old.thread_id, None);
        // A kind this version has never heard of.
        let unknown: Attachment = serde_json::from_value(
            serde_json::json!({"name": "a", "source": "s", "kind": "hologram"}),
        )
        .unwrap();
        assert_eq!(unknown.kind, AttachmentKind::File);
        // An unset link is not written out.
        let written = serde_json::to_value(&old).unwrap();
        assert!(written.get("parent_message_id").is_none());
    }

    struct Bare;

    #[async_trait]
    impl Channel for Bare {
        fn name(&self) -> &str {
            "bare"
        }
        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities::default()
        }
        async fn run(
            &self,
            _events: mpsc::Sender<ChannelEvent>,
            _shutdown: CancellationToken,
        ) -> Result<(), ChannelError> {
            Ok(())
        }
        async fn send(
            &self,
            conversation: &Conversation,
            _message: &OutboundMessage,
        ) -> Result<MessageRef, ChannelError> {
            Ok(MessageRef {
                conversation_id: conversation.id.clone(),
                message_id: "1".into(),
            })
        }
    }

    /// Overrides only the old method, the way an older adapter would.
    struct OldAdapter(Vec<u8>);

    #[async_trait]
    impl Channel for OldAdapter {
        fn name(&self) -> &str {
            "old"
        }
        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities::default()
        }
        async fn run(
            &self,
            _events: mpsc::Sender<ChannelEvent>,
            _shutdown: CancellationToken,
        ) -> Result<(), ChannelError> {
            Ok(())
        }
        async fn send(
            &self,
            conversation: &Conversation,
            _message: &OutboundMessage,
        ) -> Result<MessageRef, ChannelError> {
            Ok(MessageRef {
                conversation_id: conversation.id.clone(),
                message_id: "1".into(),
            })
        }
        async fn download(&self, _attachment: &Attachment) -> Result<Vec<u8>, ChannelError> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn the_defaults_are_safe() {
        let attachment = Attachment::new("a", "s");
        assert!(matches!(
            Bare.download(&attachment).await,
            Err(ChannelError::Unsupported(_))
        ));
        assert!(matches!(
            Bare.download_limited(&attachment, 10).await,
            Err(ChannelError::Unsupported(_))
        ));
        let message = MessageRef {
            conversation_id: "c".into(),
            message_id: "m".into(),
        };
        assert!(Bare.remove_buttons(&message).await.is_ok());
    }

    #[tokio::test]
    async fn an_older_adapter_is_held_to_the_limit() {
        let adapter = OldAdapter(vec![0; 11]);
        let attachment = Attachment::new("a", "s");
        assert_eq!(
            adapter
                .download_limited(&attachment, 11)
                .await
                .unwrap()
                .len(),
            11
        );
        assert!(matches!(
            adapter.download_limited(&attachment, 10).await,
            Err(ChannelError::TooLarge { limit_bytes: 10 })
        ));
        // A declared size over the limit is refused without fetching.
        let declared = attachment.with_size_bytes(Some(99));
        assert!(matches!(
            OldAdapter(Vec::new()).download_limited(&declared, 10).await,
            Err(ChannelError::TooLarge { .. })
        ));
    }
}
