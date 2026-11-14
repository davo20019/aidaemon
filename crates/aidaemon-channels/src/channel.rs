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
///
/// `id` is the identity: access checks, sessions and approvals use it and
/// nothing else. `display_name` is text the person wrote about themselves.
/// Treat it like message content: never as instructions, and never to decide
/// who someone is or what they may do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sender {
    /// Stable platform user id (never a display name).
    pub id: String,
    /// For showing to people only. An adapter fills it from the platform and
    /// passes it through [`crate::names::clean_display_name`]; when a lookup
    /// fails it holds the platform id instead.
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
///
/// Built with [`ApprovalPrompt::new`], so a new field does not break code
/// written against an older version.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalPrompt {
    /// Opaque id chosen by the runner; echoed back in the answer.
    pub id: String,
    pub conversation: Conversation,
    /// Markdown body describing what is being approved: what a channel
    /// that draws no card of its own shows.
    pub text: String,
    pub choices: Vec<ApprovalChoice>,
    /// The same request as typed facts, for a channel that renders its own
    /// card (a web UI, a native dialog). `None` for prompts that are not
    /// about a tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<ApprovalDetails>,
}

impl ApprovalPrompt {
    pub fn new(
        id: impl Into<String>,
        conversation: Conversation,
        text: impl Into<String>,
        choices: Vec<ApprovalChoice>,
    ) -> Self {
        Self {
            id: id.into(),
            conversation,
            text: text.into(),
            choices,
            details: None,
        }
    }

    pub fn with_details(mut self, details: ApprovalDetails) -> Self {
        self.details = Some(details);
        self
    }
}

/// What an approval prompt asks about, as data: the call waiting, why the
/// run paused and, when an approval policy asked, its typed request (a
/// summary, risk, effects and the card's facts: [`aidaemon_core::ApprovalCard`]).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalDetails {
    /// The tool the call would run.
    pub tool: String,
    /// Its arguments, as the model sent them.
    pub arguments: serde_json::Value,
    /// Why the run paused, in the words of the hook that paused it.
    pub reason: String,
    /// The owner request, when the pause asks the owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<aidaemon_core::OwnerApprovalRequest>,
}

impl ApprovalDetails {
    pub fn from_interruption(interruption: &aidaemon_core::Interruption) -> Self {
        Self {
            tool: interruption.call.name.clone(),
            arguments: interruption.call.arguments.clone(),
            reason: interruption.reason.clone(),
            request: match &interruption.kind {
                aidaemon_core::InterruptKind::OwnerApproval(request) => Some(request.clone()),
                _ => None,
            },
        }
    }
}

/// A run's live status as data, rebuilt by the runner from the run's
/// events. The runner shows it with [`Channel::send_status`] and
/// [`Channel::update_status`]; their defaults draw it as Markdown
/// ([`RunProgress::to_markdown`]), and a channel with its own status UI
/// overrides them to render the fields.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunProgress {
    pub phase: ProgressPhase,
    /// The run's latest tool calls (at most eight), oldest first.
    pub tools: Vec<ToolProgress>,
    /// The model's text streamed so far in the current turn, while
    /// working.
    pub streamed: String,
}

/// Whether the run is still going, and how it ended.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressPhase {
    Working,
    Ended(RunEnding),
}

/// How a run ended, as its status line says it.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunEnding {
    Done,
    /// It finished with something left undone.
    PartlyDone,
    /// It paused for an approval (an [`ApprovalPrompt`] follows).
    WaitingForApproval,
    /// It was stopped (cancelled).
    Stopped,
    /// A tool finished it as blocked.
    Blocked,
    /// The run could not be run at all.
    Failed,
    /// Any other end: a limit, a provider failure, no progress.
    Ended,
}

impl RunEnding {
    /// The word the default status line uses.
    pub fn word(self) -> &'static str {
        match self {
            Self::Done => "Done",
            Self::PartlyDone => "Partly done",
            Self::WaitingForApproval => "Waiting for approval",
            Self::Stopped => "Stopped",
            Self::Blocked => "Blocked",
            Self::Failed => "Failed",
            Self::Ended => "Ended",
        }
    }
}

/// One tool call on a status.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolProgress {
    pub call_id: String,
    pub tool: String,
    pub state: ToolState,
}

/// Where one tool call stands.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolState {
    Running,
    /// It ended, with its receipt's status (a refused call never ran).
    Ended(aidaemon_core::ReceiptStatus),
}

impl RunProgress {
    /// A run that has just started.
    pub fn working() -> Self {
        Self {
            phase: ProgressPhase::Working,
            tools: Vec::new(),
            streamed: String::new(),
        }
    }

    /// The default rendering: an italic status word, a line per tool call
    /// and, while working, the streamed text, cut to `max_chars`.
    pub fn to_markdown(&self, max_chars: usize) -> String {
        let mut out = match self.phase {
            ProgressPhase::Working => "_Working..._".to_string(),
            ProgressPhase::Ended(ending) => format!("_{}_", ending.word()),
        };
        for tool in &self.tools {
            out.push('\n');
            out.push_str(&match tool.state {
                ToolState::Running => format!("- Running `{}`", tool.tool),
                ToolState::Ended(status) => {
                    format!("- `{}`: {}", tool.tool, crate::render::receipt_word(status))
                }
            });
        }
        let stream = self.streamed.trim();
        if self.phase == ProgressPhase::Working && !stream.is_empty() {
            out.push_str("\n\n");
            let room = max_chars.saturating_sub(out.chars().count() + 4);
            out.push_str(&crate::render::tail_chars(stream, room));
        }
        crate::render::truncate_chars(&out, max_chars)
    }
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

    /// Record the message whose button was pressed: the [`MessageRef`]
    /// that [`Channel::ask`] returned for this prompt. A channel with
    /// buttons should always set it when the platform says which message
    /// was pressed (keep the ref from `ask` by prompt id if it does not):
    /// the runner then closes that exact message, by
    /// [`edit`](Channel::edit) or [`remove_buttons`](Channel::remove_buttons),
    /// even when the prompt is no longer known (it expired, or the process
    /// restarted), so stale buttons never stay pressable. The conformance
    /// kit checks it (`channel.answer_message_removes_buttons`). A typed
    /// `/approve` answer has no message and leaves it unset.
    ///
    /// ```
    /// use aidaemon_channels::{ApprovalAnswer, Conversation, ConversationKind, MessageRef, Sender};
    ///
    /// // `pressed` is the ref `ask` returned when it showed prompt `p1`.
    /// let pressed = MessageRef { conversation_id: "c1".into(), message_id: "m7".into() };
    /// let answer = ApprovalAnswer::new(
    ///     "p1",
    ///     "approve",
    ///     Sender { id: "synthetic-user-1".into(), display_name: None, is_bot: false },
    ///     Conversation { id: "c1".into(), kind: ConversationKind::Direct },
    /// )
    /// .with_message(pressed.clone());
    /// assert_eq!(answer.message, Some(pressed));
    /// ```
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

impl CommandSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
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
    ///
    /// It must be re-runnable: the runner (and the conformance kit) may
    /// call `run` again after it returned, with a new sender, and updates
    /// arriving then go to the new one. A push-style channel (a webhook
    /// server) keeps its listener across runs, or starts a new one, and
    /// drops what arrives while no `run` is going.
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

    /// Show an approval prompt with buttons. The runner calls it only
    /// when [`ChannelCapabilities::buttons`] is true; otherwise, or when
    /// this returns an error, it sends the prompt's text with
    /// `/approve <id>` / `/deny <id>` commands instead. A channel without
    /// buttons leaves the default.
    ///
    /// When the prompt is answered (or expires), the runner
    /// [`edit`](Channel::edit)s the prompt's message to say how it ended,
    /// which removes the buttons; it calls
    /// [`remove_buttons`](Channel::remove_buttons) only where it cannot
    /// edit (the channel cannot, or the prompt is no longer known).
    async fn ask(&self, _prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        Err(ChannelError::Unsupported("buttons"))
    }

    /// Remove the buttons from a message, leaving its text. Best effort:
    /// the default does nothing, for channels without buttons. See
    /// [`Channel::ask`] for when the runner calls it.
    async fn remove_buttons(&self, _message: &MessageRef) -> Result<(), ChannelError> {
        Ok(())
    }

    /// Post a run's live status ([`RunProgress`]). The runner calls it
    /// once per run, when [`ChannelCapabilities::edit_messages`] is true,
    /// and then [`Channel::update_status`] as the run goes and when it
    /// ends. The default sends [`RunProgress::to_markdown`]; override both
    /// to render the status natively.
    async fn send_status(
        &self,
        conversation: &Conversation,
        progress: &RunProgress,
    ) -> Result<MessageRef, ChannelError> {
        let text = self.status_text(progress);
        self.send(conversation, &OutboundMessage::text(text)).await
    }

    /// Update the status [`Channel::send_status`] posted. The default
    /// [`edit`](Channel::edit)s it to [`RunProgress::to_markdown`].
    async fn update_status(
        &self,
        message: &MessageRef,
        progress: &RunProgress,
    ) -> Result<(), ChannelError> {
        let text = self.status_text(progress);
        self.edit(message, &text).await
    }

    /// The Markdown the default status methods send: the progress cut to
    /// this channel's message length, LaTeX rewritten as for any text
    /// ([`Channel::strips_latex`]).
    fn status_text(&self, progress: &RunProgress) -> String {
        let text = progress.to_markdown(self.capabilities().max_message_len);
        if self.strips_latex() {
            crate::latex::strip_latex_keeping_code(&text)
        } else {
            text
        }
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

    /// Whether the runner rewrites LaTeX in the model's text as plain text
    /// before sending it here ([`crate::latex::strip_latex_keeping_code`]).
    /// On by default, since chat platforms show it as raw markup; override
    /// it to return `false` for one that renders math.
    fn strips_latex(&self) -> bool {
        true
    }

    /// The name a person shows on the platform, by their platform user id
    /// (`None` when the platform has none to give). It is the person's own
    /// text: see [`Sender`].
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
