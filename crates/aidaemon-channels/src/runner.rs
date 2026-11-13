//! The runner: channel events in, agent runs out.
//!
//! [`ChannelRunner`] owns everything that must behave the same on every
//! channel: who may talk to the agent ([`AccessPolicy`]), one session per
//! conversation, one run at a time per session with a bounded queue,
//! streamed progress, typing indicators, approvals that pause and resume
//! interrupted runs, and the `/stop`, `/new`, `/status`, `/approve <id>` and
//! `/deny <id>` commands, next to any a host adds ([`crate::commands`]).
//!
//! The model's text has its LaTeX rewritten as plain text on the way out,
//! code excepted, unless the channel opts out with
//! [`Channel::strips_latex`].
//!
//! # Attachments
//!
//! On a channel with the `files` capability, an inbound image reaches the
//! agent as an image, checked by its bytes. Anything else from the owner is
//! saved under [`AttachmentConfig::save_dir`], when one is set, and the agent
//! is told the path. Whatever cannot be loaded is still named in the message
//! with the reason. A file's author is unknown even when the owner sent it,
//! so a run that received loaded content starts as having read untrusted
//! content ([`RunState::with_untrusted_context`]), and that stays true for
//! the session's later runs until `/new`.
//!
//! # Wiring
//!
//! ```ignore
//! let router = EventRouter::new();
//! let agent = Agent::builder("assistant").allow_unguarded()
//!     .provider(provider)
//!     .model("model-name")
//!     .hook(my_approval_hook)          // returns ToolDecision::Interrupt for risky tools
//!     .event_sink(router.clone())      // streamed progress reaches the runner
//!     .build()?;
//! let runner = ChannelRunner::builder(agent)
//!     .channel(Arc::new(telegram))
//!     .access(AccessPolicy::new().owner("telegram", "123456"))
//!     .event_router(router)
//!     .build()?;
//! runner.run(shutdown_token).await;
//! ```
//!
//! # Sessions and continuation
//!
//! The session id is `"{channel_name}:{conversation_id}"`. Each finished run
//! is the starting point of the next one in the same session
//! (`state.with_user_message(text)` + [`Agent::run_from`]). With a
//! [`RunStore`] the runner saves every outcome's state and records its run
//! id in the [`SessionIndex`], so continuation survives restarts; without one
//! it continues from the last state kept in memory.
//!
//! # Approvals across restarts
//!
//! With an [`ApprovalStore`] (which needs a [`RunStore`], where the interrupted
//! run lives) open approvals come back on the next start: their buttons and
//! `/approve <id>` keep working, and the time spent stopped counts toward
//! `approval_timeout`.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aidaemon_core::{
    Agent, CoreError, Event, Image, Message, Resolution, Role, RunId, RunOutcome, RunState,
    RunStatus, RunStore,
};
use base64::Engine as _;
use tokio::sync::{mpsc, Semaphore};
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::approvals::{ApprovalStore, StoredApproval};
use crate::channel::{
    ApprovalChoice, ApprovalDetails, ApprovalPrompt, Attachment, AttachmentKind, Channel,
    ChannelError, ChannelEvent, CommandInvocation, CommandSpec, Conversation, ConversationKind,
    InboundMessage, MessageRef, OutboundMessage, RunEnding,
};
use crate::commands::{
    builtin_commands, invalid_spec, is_builtin, CommandContext, CommandHandler, CommandReply,
};
use crate::inbound_files::{list_line, save_attachment, sniff_image, ATTACHMENTS_HEADING};
use crate::latex::strip_latex_keeping_code;
use crate::policy::{AccessPolicy, Admission, SenderRole};
use crate::render::{self, Progress};
use crate::router::EventRouter;
use crate::session::{session_id, MemorySessionIndex, SessionIndex};
use crate::split::split_message;

/// Choice id of the "Approve" button.
pub const APPROVE_CHOICE: &str = "approve";
/// Choice id of the "Deny" button.
pub const DENY_CHOICE: &str = "deny";

/// What the runner loads from inbound attachments. `Default` gives the
/// documented defaults.
#[derive(Debug, Clone)]
pub struct AttachmentConfig {
    /// Attachments loaded per message; the rest are only named. 0 loads
    /// none. Default 8.
    pub max_per_message: usize,
    /// Largest image passed to the model. Default 5 MiB, which every
    /// provider accepts.
    pub max_image_bytes: u64,
    /// Largest file saved under [`Self::save_dir`]. Default 20 MiB.
    pub max_file_bytes: u64,
    /// How long one download may take. Default 60 s.
    pub download_timeout: Duration,
    /// Where the owner's non-image files are saved (one private folder per
    /// session, files readable only by this user), and the agent told the
    /// path. Keep it outside the workspaces the agent trusts, so a tool that
    /// reads a saved file reports it as untrusted. Without one, and for
    /// senders other than the owner, such files are only named. Default
    /// none.
    pub save_dir: Option<PathBuf>,
}

impl Default for AttachmentConfig {
    fn default() -> Self {
        Self {
            max_per_message: 8,
            max_image_bytes: 5 * 1024 * 1024,
            max_file_bytes: 20 * 1024 * 1024,
            download_timeout: Duration::from_secs(60),
            save_dir: None,
        }
    }
}

/// Tunables. `Default` gives the documented defaults.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Messages kept per session while a run is active or an approval is
    /// pending. When full, the oldest is dropped with a notice. Minimum 1.
    pub max_queued_per_session: usize,
    /// Runs executing at once across all sessions. Minimum 1.
    pub max_concurrent_runs: usize,
    /// Minimum time between status-message edits.
    pub status_edit_interval: Duration,
    /// How often the typing indicator is refreshed during a run.
    pub typing_interval: Duration,
    /// Pending approvals older than this are answered with Deny.
    pub approval_timeout: Duration,
    /// Pretty-printed tool arguments in approval prompts are cut here.
    pub max_argument_chars: usize,
    /// First delay before restarting a failed channel; doubles per failure.
    pub restart_backoff: Duration,
    pub max_restart_backoff: Duration,
    /// Buffer between each channel task and the dispatcher.
    pub event_buffer: usize,
    /// How long [`ChannelRunner::run`] waits for in-flight work at shutdown.
    pub shutdown_grace: Duration,
    /// What is loaded from inbound attachments.
    pub attachments: AttachmentConfig,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_queued_per_session: 8,
            max_concurrent_runs: 4,
            status_edit_interval: Duration::from_millis(1500),
            typing_interval: Duration::from_secs(4),
            approval_timeout: Duration::from_secs(3600),
            max_argument_chars: 1500,
            restart_backoff: Duration::from_secs(1),
            max_restart_backoff: Duration::from_secs(60),
            event_buffer: 256,
            shutdown_grace: Duration::from_secs(5),
            attachments: AttachmentConfig::default(),
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RunnerBuildError {
    #[error("a runner needs at least one channel")]
    NoChannels,
    #[error("two channels are named `{0}`; channel names must be unique")]
    DuplicateChannel(String),
    #[error("invalid config: {0}")]
    InvalidConfig(&'static str),
}

/// Builder for [`ChannelRunner`].
pub struct ChannelRunnerBuilder {
    agent: Agent,
    channels: Vec<Arc<dyn Channel>>,
    policy: AccessPolicy,
    router: Option<EventRouter>,
    store: Option<Arc<dyn RunStore>>,
    index: Option<Arc<dyn SessionIndex>>,
    approval_store: Option<Arc<dyn ApprovalStore>>,
    commands: Vec<HostCommand>,
    config: RunnerConfig,
}

impl ChannelRunnerBuilder {
    /// Add a channel. Names must be unique; they prefix session ids.
    pub fn channel(mut self, channel: Arc<dyn Channel>) -> Self {
        self.channels.push(channel);
        self
    }

    pub fn access(mut self, policy: AccessPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The router registered on the agent with `.event_sink(router.clone())`.
    /// Without it, no streamed progress is shown.
    pub fn event_router(mut self, router: EventRouter) -> Self {
        self.router = Some(router);
        self
    }

    /// Persist each outcome's state and continue sessions from it.
    pub fn store(mut self, store: Arc<dyn RunStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Where the latest run of each session is recorded. Defaults to
    /// [`MemorySessionIndex`]. Only consulted when a store is set.
    pub fn session_index(mut self, index: Arc<dyn SessionIndex>) -> Self {
        self.index = Some(index);
        self
    }

    /// Keep open approvals here so they survive a restart. Needs a
    /// [`Self::store`]: an approval resumes the run saved there.
    pub fn approval_store(mut self, approvals: Arc<dyn ApprovalStore>) -> Self {
        self.approval_store = Some(approvals);
        self
    }

    /// Add a slash command, offered in each platform's command menu.
    /// `name` is without the `/`: lowercase letters, digits and `_`, at most
    /// 32, and not a built-in. See [`crate::commands`].
    pub fn command(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl CommandHandler + 'static,
    ) -> Self {
        self.commands.push(HostCommand {
            spec: CommandSpec::new(name, description),
            handler: Arc::new(handler),
        });
        self
    }

    pub fn config(mut self, config: RunnerConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> Result<ChannelRunner, RunnerBuildError> {
        if self.channels.is_empty() {
            return Err(RunnerBuildError::NoChannels);
        }
        let mut seen = std::collections::HashSet::new();
        for channel in &self.channels {
            if !seen.insert(channel.name().to_string()) {
                return Err(RunnerBuildError::DuplicateChannel(
                    channel.name().to_string(),
                ));
            }
        }
        if self.config.max_queued_per_session == 0 {
            return Err(RunnerBuildError::InvalidConfig(
                "max_queued_per_session must be at least 1",
            ));
        }
        if self.config.max_concurrent_runs == 0 {
            return Err(RunnerBuildError::InvalidConfig(
                "max_concurrent_runs must be at least 1",
            ));
        }
        if self.config.event_buffer == 0 {
            return Err(RunnerBuildError::InvalidConfig(
                "event_buffer must be at least 1",
            ));
        }
        if self.approval_store.is_some() && self.store.is_none() {
            return Err(RunnerBuildError::InvalidConfig(
                "an approval store needs a run store",
            ));
        }
        let mut commands = HashMap::new();
        for command in &self.commands {
            if let Some(problem) = invalid_spec(&command.spec) {
                return Err(RunnerBuildError::InvalidConfig(problem));
            }
            if is_builtin(&command.spec.name) {
                return Err(RunnerBuildError::InvalidConfig(
                    "a command cannot replace a built-in",
                ));
            }
            if commands
                .insert(command.spec.name.clone(), command.handler.clone())
                .is_some()
            {
                return Err(RunnerBuildError::InvalidConfig(
                    "two commands have the same name",
                ));
            }
        }
        let command_specs = self.commands.into_iter().map(|c| c.spec).collect();
        let permits = Arc::new(Semaphore::new(self.config.max_concurrent_runs));
        Ok(ChannelRunner {
            inner: Arc::new(Inner {
                agent: self.agent,
                channels: self.channels,
                policy: self.policy,
                router: self.router,
                store: self.store,
                index: self
                    .index
                    .unwrap_or_else(|| Arc::new(MemorySessionIndex::new())),
                approval_store: self.approval_store,
                commands,
                command_specs,
                config: self.config,
                sessions: Mutex::new(HashMap::new()),
                approvals: Mutex::new(HashMap::new()),
                permits,
                root: CancellationToken::new(),
                tasks: TaskTracker::new(),
            }),
        })
    }
}

/// Connects channels to one agent. See the module docs for the wiring.
#[derive(Clone)]
pub struct ChannelRunner {
    inner: Arc<Inner>,
}

impl ChannelRunner {
    pub fn builder(agent: Agent) -> ChannelRunnerBuilder {
        ChannelRunnerBuilder {
            agent,
            channels: Vec::new(),
            policy: AccessPolicy::default(),
            router: None,
            store: None,
            index: None,
            approval_store: None,
            commands: Vec::new(),
            config: RunnerConfig::default(),
        }
    }

    /// Drive every channel until `shutdown` is cancelled. A channel whose
    /// task fails is restarted with exponential backoff, except on
    /// [`ChannelError::Auth`], which stops that channel. At shutdown, active
    /// runs are cancelled and in-flight work gets `shutdown_grace` to finish.
    ///
    /// Call it once per runner.
    pub async fn run(&self, shutdown: CancellationToken) {
        let inner = self.inner.clone();
        let root = inner.root.clone();
        // Before any channel starts, so a press that arrives first finds it.
        let restored = inner.restore_approvals().await;
        inner.register_commands();
        let mut handles = Vec::new();
        for (index, channel) in inner.channels.iter().enumerate() {
            let (tx, rx) = mpsc::channel(inner.config.event_buffer);
            handles.push(tokio::spawn(supervise(
                channel.clone(),
                tx,
                root.clone(),
                inner.config.clone(),
            )));
            handles.push(tokio::spawn(dispatch(inner.clone(), index, rx)));
        }
        for (prompt_id, remaining) in restored {
            inner.arm_expiry(prompt_id, remaining);
        }
        tokio::select! {
            _ = shutdown.cancelled() => {}
            _ = root.cancelled() => {}
        }
        root.cancel();
        for handle in handles {
            let _ = handle.await;
        }
        inner.tasks.close();
        if tokio::time::timeout(inner.config.shutdown_grace, inner.tasks.wait())
            .await
            .is_err()
        {
            tracing::warn!("channel runner: in-flight work did not finish before shutdown");
        }
    }
}

/// Where replies for a job go.
#[derive(Debug, Clone)]
struct Origin {
    channel: usize,
    conversation: Conversation,
    reply_to: Option<String>,
}

enum Job {
    Message {
        origin: Origin,
        text: String,
        /// Loaded when the job runs, so a download never holds up the
        /// channel's other events (`/stop` above all).
        attachments: Vec<Attachment>,
        role: SenderRole,
    },
    Resume {
        origin: Origin,
        state: Box<RunState>,
        resolution: Resolution,
        /// `/stop` while waiting: settle the call, then end cancelled.
        stopped: bool,
    },
}

impl Job {
    fn origin(&self) -> &Origin {
        match self {
            Job::Message { origin, .. } | Job::Resume { origin, .. } => origin,
        }
    }
}

enum SlotState {
    Idle,
    Running { cancel: CancellationToken },
    Awaiting { prompt_id: String },
}

struct Slot {
    state: SlotState,
    queue: VecDeque<Job>,
    /// Continuation when no store is configured (or it has nothing).
    last_state: Option<RunState>,
    /// Bumped by `/new`, so a run that started before it does not become
    /// the session's continuation.
    generation: u64,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            state: SlotState::Idle,
            queue: VecDeque::new(),
            last_state: None,
            generation: 0,
        }
    }
}

/// What became of one inbound attachment.
enum Loaded {
    Image(Image),
    Saved(PathBuf),
    /// Named in the message only, with this status.
    Listed(&'static str),
}

struct PendingApproval {
    prompt_id: String,
    session_id: String,
    origin: Origin,
    state: RunState,
    prompt_text: String,
    prompt_ref: Option<MessageRef>,
    expires_at_ms: u64,
}

enum Command {
    Stop,
    New,
    Status,
    Approve(Option<String>),
    Deny(Option<String>),
}

/// The name (lowercase) and arguments of text in slash-command syntax.
fn command_parts(text: &str) -> Option<(String, &str)> {
    let rest = text.trim().strip_prefix('/')?;
    let head = rest.split_whitespace().next()?;
    // Telegram appends the bot name in groups: `/stop@my_bot`.
    let name = head.split('@').next().unwrap_or(head).to_ascii_lowercase();
    Some((name, rest[head.len()..].trim()))
}

/// Explicit slash-command syntax. Unknown commands go to the agent.
fn parse_command(text: &str) -> Option<Command> {
    let (name, args) = command_parts(text)?;
    let arg = args.split_whitespace().next().map(str::to_string);
    match name.as_str() {
        "stop" => Some(Command::Stop),
        "new" => Some(Command::New),
        "status" => Some(Command::Status),
        "approve" => Some(Command::Approve(arg)),
        "deny" => Some(Command::Deny(arg)),
        _ => None,
    }
}

struct HostCommand {
    spec: CommandSpec,
    handler: Arc<dyn CommandHandler>,
}

/// One accepted use of a host command, and where its answer goes.
struct CommandCall {
    context: CommandContext,
    origin: Origin,
    /// Set when it came from the platform's menu, which wants an answer.
    invocation: Option<CommandInvocation>,
}

/// Answers a menu invocation when there is nothing else to say, so the
/// platform does not leave it pending.
const COMMAND_DONE: &str = "Done.";
const COMMAND_FAILED: &str = "That command failed.";
const COMMAND_REFUSED: &str = "Not available.";

struct Inner {
    agent: Agent,
    channels: Vec<Arc<dyn Channel>>,
    policy: AccessPolicy,
    router: Option<EventRouter>,
    store: Option<Arc<dyn RunStore>>,
    index: Arc<dyn SessionIndex>,
    approval_store: Option<Arc<dyn ApprovalStore>>,
    commands: HashMap<String, Arc<dyn CommandHandler>>,
    command_specs: Vec<CommandSpec>,
    config: RunnerConfig,
    sessions: Mutex<HashMap<String, Slot>>,
    approvals: Mutex<HashMap<String, PendingApproval>>,
    permits: Arc<Semaphore>,
    /// Cancelled at shutdown; every run's token is a child of it.
    root: CancellationToken,
    tasks: TaskTracker,
}

async fn supervise(
    channel: Arc<dyn Channel>,
    tx: mpsc::Sender<ChannelEvent>,
    root: CancellationToken,
    config: RunnerConfig,
) {
    let mut backoff = config.restart_backoff;
    loop {
        let started = Instant::now();
        let result = channel.run(tx.clone(), root.child_token()).await;
        if root.is_cancelled() {
            return;
        }
        match result {
            Err(ChannelError::Auth(reason)) => {
                tracing::error!(
                    channel = channel.name(),
                    %reason,
                    "channel authentication failed; not restarting it"
                );
                return;
            }
            Err(err) => {
                tracing::warn!(channel = channel.name(), error = %err, "channel task failed; restarting")
            }
            Ok(()) => {
                tracing::warn!(
                    channel = channel.name(),
                    "channel task returned; restarting"
                )
            }
        }
        if started.elapsed() >= config.max_restart_backoff {
            backoff = config.restart_backoff;
        }
        tokio::select! {
            _ = tokio::time::sleep(backoff) => {}
            _ = root.cancelled() => return,
        }
        backoff = (backoff * 2).min(config.max_restart_backoff);
    }
}

async fn dispatch(inner: Arc<Inner>, channel: usize, mut rx: mpsc::Receiver<ChannelEvent>) {
    loop {
        tokio::select! {
            _ = inner.root.cancelled() => return,
            event = rx.recv() => match event {
                Some(event) => inner.handle_event(channel, event).await,
                None => return,
            },
        }
    }
}

/// Waits forever when there is no receiver.
async fn recv_opt(rx: &mut Option<mpsc::UnboundedReceiver<Event>>) -> Option<Event> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

impl Inner {
    fn channel_name(&self, channel: usize) -> &str {
        self.channels[channel].name()
    }

    /// Offer the commands in each platform's menu. Best effort and in the
    /// background, so a slow or failing platform delays nothing.
    fn register_commands(self: &Arc<Self>) {
        for channel in &self.channels {
            let caps = channel.capabilities();
            if !caps.commands {
                continue;
            }
            let mut menu = builtin_commands(caps.buttons);
            menu.extend(self.command_specs.iter().cloned());
            let (channel, root) = (channel.clone(), self.root.clone());
            self.tasks.spawn(async move {
                tokio::select! {
                    _ = root.cancelled() => {}
                    result = channel.register_commands(&menu) => {
                        if let Err(error) = result {
                            tracing::warn!(channel = channel.name(), %error, "could not register commands");
                        }
                    }
                }
            });
        }
    }

    /// The model's text as the channel should receive it.
    fn outbound_text(&self, channel: &dyn Channel, text: &str) -> String {
        if channel.strips_latex() {
            strip_latex_keeping_code(text)
        } else {
            text.to_string()
        }
    }

    async fn say(&self, origin: &Origin, text: &str) {
        let channel = &self.channels[origin.channel];
        let max = channel.capabilities().max_message_len;
        for part in split_message(text, max) {
            let message = OutboundMessage::text(part);
            if let Err(err) = channel.send(&origin.conversation, &message).await {
                tracing::warn!(channel = channel.name(), error = %err, "send failed");
            }
        }
    }

    /// Send the final answer, replying to the triggering message in groups.
    async fn reply(&self, origin: &Origin, text: &str) {
        let channel = &self.channels[origin.channel];
        let max = channel.capabilities().max_message_len;
        let text = self.outbound_text(channel.as_ref(), text);
        for (i, part) in split_message(&text, max).into_iter().enumerate() {
            let message = OutboundMessage {
                text: part,
                reply_to: if i == 0 {
                    origin.reply_to.clone()
                } else {
                    None
                },
            };
            if let Err(err) = channel.send(&origin.conversation, &message).await {
                tracing::warn!(channel = channel.name(), error = %err, "send failed");
            }
        }
    }

    async fn handle_event(self: &Arc<Self>, channel: usize, event: ChannelEvent) {
        match event {
            ChannelEvent::Message(message) => self.handle_message(channel, message).await,
            ChannelEvent::Command(command) => self.handle_invocation(channel, command).await,
            ChannelEvent::Approval(answer) => {
                let approve = match answer.choice_id.as_str() {
                    APPROVE_CHOICE => true,
                    DENY_CHOICE => false,
                    other => {
                        tracing::debug!(choice = other, "ignoring unknown approval choice");
                        return;
                    }
                };
                let origin = Origin {
                    channel,
                    conversation: answer.conversation,
                    reply_to: None,
                };
                self.answer(
                    origin,
                    &answer.sender.id,
                    Some(answer.prompt_id),
                    approve,
                    answer.message,
                )
                .await;
            }
        }
    }

    /// A command picked from the platform's menu. Held to the same access
    /// check as a typed message; a built-in or unknown one is then handled
    /// as if typed. The invocation is always answered, so the platform does
    /// not leave it pending.
    async fn handle_invocation(self: &Arc<Self>, channel: usize, command: CommandInvocation) {
        let name = self.channel_name(channel).to_string();
        let adapter = self.channels[channel].clone();
        let mut message = InboundMessage::new(
            command.conversation.clone(),
            command.sender.clone(),
            String::new(),
            command.as_text(),
        );
        message.addressed = true;
        let role = match self.policy.admit(&name, &message) {
            Admission::Ignore => return,
            Admission::Refuse => {
                tracing::info!(channel = %name, sender = %command.sender.id, "refused a command from an unknown sender");
                let text = self.policy.refusal().unwrap_or(COMMAND_REFUSED);
                if let Err(error) = adapter.answer_command(&command, text).await {
                    tracing::debug!(%error, "could not answer a command invocation");
                }
                return;
            }
            Admission::Accept(role) => role,
        };
        let command_name = command.name.to_ascii_lowercase();
        if let Some(handler) = self.commands.get(&command_name).cloned() {
            let origin = Origin {
                channel,
                reply_to: None,
                conversation: command.conversation.clone(),
            };
            let context = CommandContext {
                session_id: session_id(&name, &command.conversation.id),
                channel: name,
                conversation: command.conversation.clone(),
                sender: command.sender.clone(),
                role,
                name: command_name,
                args: command.args.trim().to_string(),
            };
            let invocation = Some(command);
            self.spawn_command(
                handler,
                CommandCall {
                    context,
                    origin,
                    invocation,
                },
            );
            return;
        }
        if let Err(error) = adapter.answer_command(&command, "Got it.").await {
            tracing::debug!(%error, "could not answer a command invocation");
        }
        self.handle_message(channel, message).await;
    }

    /// Run a host command in its own task. A handler that panics answers
    /// with a failure instead of taking anything else down.
    fn spawn_command(self: &Arc<Self>, handler: Arc<dyn CommandHandler>, call: CommandCall) {
        let inner = self.clone();
        self.tasks.spawn(async move {
            let CommandCall {
                context,
                origin,
                invocation,
            } = call;
            let command = context.name.clone();
            let mut run = tokio::spawn(async move { handler.handle(context).await });
            let result = tokio::select! {
                _ = inner.root.cancelled() => {
                    run.abort();
                    return;
                }
                result = &mut run => result,
            };
            let text = match result {
                Ok(CommandReply::Text(text)) => Some(text),
                Ok(CommandReply::Silent) => None,
                Err(error) => {
                    tracing::warn!(%command, %error, "command handler failed");
                    Some(COMMAND_FAILED.to_string())
                }
            };
            match (invocation, text) {
                (Some(invocation), text) => {
                    let channel = &inner.channels[origin.channel];
                    let text = text.as_deref().unwrap_or(COMMAND_DONE);
                    if let Err(error) = channel.answer_command(&invocation, text).await {
                        tracing::debug!(%error, "could not answer a command invocation");
                    }
                }
                (None, Some(text)) => inner.say(&origin, &text).await,
                (None, None) => {}
            }
        });
    }

    async fn handle_message(self: &Arc<Self>, channel: usize, message: InboundMessage) {
        let name = self.channel_name(channel).to_string();
        let origin = Origin {
            channel,
            // A menu command has no message to reply to.
            reply_to: (message.conversation.kind != ConversationKind::Direct
                && !message.message_id.is_empty())
            .then(|| message.message_id.clone()),
            conversation: message.conversation.clone(),
        };
        let role = match self.policy.admit(&name, &message) {
            Admission::Ignore => return,
            Admission::Refuse => {
                tracing::info!(channel = %name, sender = %message.sender.id, "refused unknown sender");
                if let Some(refusal) = self.policy.refusal() {
                    self.say(&origin, refusal).await;
                }
                return;
            }
            Admission::Accept(role) => role,
        };
        let session = session_id(&name, &message.conversation.id);
        match parse_command(&message.text) {
            Some(Command::Stop) => self.stop(&session, &origin).await,
            Some(Command::New) => self.reset(&session, &origin).await,
            Some(Command::Status) => self.status(&session, &origin).await,
            Some(Command::Approve(id)) => {
                self.answer(origin, &message.sender.id, id, true, None)
                    .await;
            }
            Some(Command::Deny(id)) => {
                self.answer(origin, &message.sender.id, id, false, None)
                    .await;
            }
            None => {
                if let Some((command, args)) = command_parts(&message.text)
                    && let Some(handler) = self.commands.get(&command).cloned() {
                        let context = CommandContext {
                            channel: name,
                            conversation: message.conversation.clone(),
                            sender: message.sender.clone(),
                            role,
                            session_id: session,
                            args: args.to_string(),
                            name: command,
                        };
                        let call = CommandCall {
                            context,
                            origin,
                            invocation: None,
                        };
                        self.spawn_command(handler, call);
                        return;
                    }
                self.submit(
                    session,
                    Job::Message {
                        origin,
                        text: message.text,
                        attachments: message.attachments,
                        role,
                    },
                )
                .await;
            }
        }
    }

    /// Start the job now if the session is idle, otherwise queue it.
    async fn submit(self: &Arc<Self>, session: String, job: Job) {
        let mut notices: Vec<(Origin, String)> = Vec::new();
        let start = {
            let mut sessions = self.sessions.lock().expect("sessions poisoned");
            let slot = sessions.entry(session.clone()).or_default();
            match &slot.state {
                SlotState::Idle => {
                    slot.state = SlotState::Running {
                        cancel: self.root.child_token(),
                    };
                    Some(job)
                }
                busy => {
                    if let SlotState::Awaiting { prompt_id } = busy {
                        notices.push((
                            job.origin().clone(),
                            format!(
                                "Waiting for an answer to approval `{prompt_id}`; I will handle \
                                 this message afterwards."
                            ),
                        ));
                    }
                    if slot.queue.len() >= self.config.max_queued_per_session
                        && let Some(dropped) = slot.queue.pop_front() {
                            notices.push((
                                dropped.origin().clone(),
                                "Too many messages are waiting, so I dropped the oldest one."
                                    .to_string(),
                            ));
                        }
                    slot.queue.push_back(job);
                    None
                }
            }
        };
        for (origin, text) in notices {
            self.say(&origin, &text).await;
        }
        if let Some(job) = start {
            self.spawn_worker(session, job);
        }
    }

    fn spawn_worker(self: &Arc<Self>, session: String, job: Job) {
        let inner = self.clone();
        self.tasks
            .spawn(async move { inner.worker(session, job).await });
    }

    /// Runs jobs of one session in order until its queue is empty or a run
    /// is waiting for approval. Only one worker per session exists at a time
    /// (the slot is `Running` while it does).
    async fn worker(self: Arc<Self>, session: String, mut job: Job) {
        loop {
            let cancel = {
                let mut sessions = self.sessions.lock().expect("sessions poisoned");
                let slot = sessions.entry(session.clone()).or_default();
                match &slot.state {
                    SlotState::Running { cancel } => cancel.clone(),
                    _ => {
                        let cancel = self.root.child_token();
                        slot.state = SlotState::Running {
                            cancel: cancel.clone(),
                        };
                        cancel
                    }
                }
            };
            let permit = self.permits.clone().acquire_owned().await;
            let approval = self.process(&session, job, cancel).await;
            drop(permit);

            let next = {
                let mut sessions = self.sessions.lock().expect("sessions poisoned");
                let slot = sessions.entry(session.clone()).or_default();
                if let Some(pending) = &approval {
                    slot.state = SlotState::Awaiting {
                        prompt_id: pending.prompt_id.clone(),
                    };
                    None
                } else if let Some(next) = slot.queue.pop_front() {
                    slot.state = SlotState::Running {
                        cancel: self.root.child_token(),
                    };
                    Some(next)
                } else {
                    slot.state = SlotState::Idle;
                    None
                }
            };
            if let Some(pending) = approval {
                self.open_approval(pending).await;
                return;
            }
            match next {
                Some(next) => job = next,
                None => return,
            }
        }
    }

    /// The state a new message continues from, if any.
    async fn continuation(&self, session: &str) -> Option<RunState> {
        let usable =
            |state: &RunState| state.pending.is_none() && state.agent_id == self.agent.id();
        if let Some(store) = &self.store
            && let Some(run_id) = self.index.latest_run(session).await {
                match store.load_state(run_id).await {
                    Ok(Some(state)) if usable(&state) => return Some(state),
                    Ok(Some(_)) => {
                        tracing::warn!(session, %run_id, "latest run cannot be continued; starting fresh")
                    }
                    Ok(None) => tracing::warn!(session, %run_id, "latest run not found in store"),
                    Err(err) => {
                        tracing::warn!(session, error = %err, "loading session state failed")
                    }
                }
            }
        let sessions = self.sessions.lock().expect("sessions poisoned");
        sessions
            .get(session)
            .and_then(|slot| slot.last_state.clone())
            .filter(|state| usable(state))
    }

    fn generation(&self, session: &str) -> u64 {
        let sessions = self.sessions.lock().expect("sessions poisoned");
        sessions.get(session).map_or(0, |slot| slot.generation)
    }

    /// Run one job to its end and deliver the output. Returns an approval
    /// to open when the run was interrupted.
    async fn process(
        self: &Arc<Self>,
        session: &str,
        job: Job,
        cancel: CancellationToken,
    ) -> Option<PendingApproval> {
        let generation = self.generation(session);
        let agent = self.agent.clone();
        let (origin, outcome, base) = match job {
            Job::Message {
                origin,
                text,
                attachments,
                role,
            } => {
                let (message, untrusted) = self
                    .input_message(session, &origin, text, &attachments, role, &cancel)
                    .await;
                let mut state = match self.continuation(session).await {
                    Some(state) => state.with_user_message(message),
                    None => RunState::new(self.agent.id(), session, message),
                };
                if untrusted {
                    state = state.with_untrusted_context();
                }
                let base = state.messages.len();
                let run_id = state.run_id;
                let run = {
                    let cancel = cancel.clone();
                    async move { agent.run_from(state, cancel).await }
                };
                let outcome = self.drive(session, &origin, Some(run_id), run).await;
                (origin, outcome, base)
            }
            Job::Resume {
                origin,
                state,
                resolution,
                stopped,
            } => {
                if stopped {
                    cancel.cancel();
                }
                let state = *state;
                let base = state.messages.len();
                let run_id = state.run_id;
                let run = {
                    let cancel = cancel.clone();
                    async move { agent.resume_with_cancel(state, resolution, cancel).await }
                };
                let outcome = self.drive(session, &origin, Some(run_id), run).await;
                (origin, outcome, base)
            }
        };

        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                tracing::error!(session, error = %err, "agent run failed");
                self.say(
                    &origin,
                    &format!(
                        "Something went wrong while running the agent: {}",
                        render::truncate_chars(&err, 200)
                    ),
                )
                .await;
                return None;
            }
        };

        self.remember(session, generation, &outcome).await;

        match &outcome.status {
            // The model's final answer, or the reply a tool finished the
            // run with.
            RunStatus::Completed | RunStatus::Finished { .. } => {
                let text = outcome.final_text.as_deref().unwrap_or("").trim();
                if text.is_empty() {
                    self.reply(&origin, &render::outcome_ending_text(&outcome))
                        .await;
                } else {
                    self.reply(&origin, text).await;
                }
                None
            }
            RunStatus::Interrupted { interruption } => {
                let prompt_id = self.new_prompt_id();
                let prompt_text =
                    render::approval_text(interruption, self.config.max_argument_chars);
                Some(PendingApproval {
                    prompt_id,
                    session_id: session.to_string(),
                    origin,
                    state: outcome.state,
                    prompt_text,
                    prompt_ref: None,
                    expires_at_ms: now_ms()
                        .saturating_add(self.config.approval_timeout.as_millis() as u64),
                })
            }
            status => {
                let mut text = render::ending_text(status);
                if let Some(partial) = partial_text(&outcome.state, base) {
                    text.push_str("\n\nWhat I had so far:\n\n");
                    text.push_str(&partial);
                }
                self.reply(&origin, &text).await;
                None
            }
        }
    }

    /// The user message for a job, and whether it carries content someone
    /// else may have written.
    async fn input_message(
        &self,
        session: &str,
        origin: &Origin,
        text: String,
        attachments: &[Attachment],
        role: SenderRole,
        cancel: &CancellationToken,
    ) -> (Message, bool) {
        if attachments.is_empty() {
            return (Message::user(text), false);
        }
        let channel = self.channels[origin.channel].clone();
        let caps = channel.capabilities();
        if caps.typing_indicator {
            let _ = channel.typing(&origin.conversation).await;
        }
        let mut images = Vec::new();
        let mut untrusted = false;
        let mut body = text;
        if !body.is_empty() {
            body.push_str("\n\n");
        }
        body.push_str(ATTACHMENTS_HEADING);
        for (index, attachment) in attachments.iter().enumerate() {
            let loaded = if caps.files && index < self.config.attachments.max_per_message {
                self.load_attachment(session, channel.as_ref(), attachment, role, cancel)
                    .await
            } else {
                Loaded::Listed("not loaded")
            };
            let saved_note;
            let status = match loaded {
                Loaded::Image(image) => {
                    images.push(image);
                    untrusted = true;
                    "included as an image"
                }
                Loaded::Saved(path) => {
                    untrusted = true;
                    saved_note = format!("saved at {}", path.display());
                    saved_note.as_str()
                }
                Loaded::Listed(reason) => reason,
            };
            body.push('\n');
            body.push_str(&list_line(attachment, status));
        }
        (Message::user(body).with_images(images), untrusted)
    }

    /// Fetch one attachment and turn it into something the agent can use.
    async fn load_attachment(
        &self,
        session: &str,
        channel: &dyn Channel,
        attachment: &Attachment,
        role: SenderRole,
        cancel: &CancellationToken,
    ) -> Loaded {
        let config = &self.config.attachments;
        let save_dir = config
            .save_dir
            .as_ref()
            .filter(|_| role == SenderRole::Owner);
        let wants_image = attachment.kind == AttachmentKind::Image;
        if !wants_image && save_dir.is_none() {
            return Loaded::Listed("not loaded");
        }
        let limit = if save_dir.is_some() {
            config.max_file_bytes
        } else {
            config.max_image_bytes
        };
        let download = tokio::time::timeout(
            config.download_timeout,
            channel.download_limited(attachment, limit),
        );
        let bytes = tokio::select! {
            _ = cancel.cancelled() => return Loaded::Listed("not loaded (the run was stopped)"),
            result = download => match result {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(ChannelError::TooLarge { .. })) => {
                    return Loaded::Listed("not loaded (over the size limit)")
                }
                Ok(Err(err)) => {
                    // The reason stays in the log: it is the platform's text.
                    tracing::warn!(channel = channel.name(), error = %err, "attachment download failed");
                    return Loaded::Listed("not loaded (the download failed)");
                }
                Err(_) => return Loaded::Listed("not loaded (the download timed out)"),
            },
        };
        if wants_image && bytes.len() as u64 <= config.max_image_bytes
            && let Some(media_type) = sniff_image(&bytes) {
                let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
                return Loaded::Image(Image::base64(media_type, data));
            }
        let Some(dir) = save_dir.cloned() else {
            return Loaded::Listed("not loaded (not a supported image)");
        };
        let (session, name) = (session.to_string(), attachment.name.clone());
        let saved =
            tokio::task::spawn_blocking(move || save_attachment(&dir, &session, &name, &bytes))
                .await;
        match saved {
            Ok(Ok(path)) => Loaded::Saved(path),
            Ok(Err(err)) => {
                tracing::warn!(error = %err, "could not save an attachment");
                Loaded::Listed("not loaded (it could not be saved)")
            }
            Err(err) => {
                tracing::warn!(error = %err, "saving an attachment failed");
                Loaded::Listed("not loaded (it could not be saved)")
            }
        }
    }

    /// Persist the outcome and make it the session's continuation.
    async fn remember(&self, session: &str, generation: u64, outcome: &RunOutcome) {
        if let Some(store) = &self.store
            && let Err(err) = store.save_state(&outcome.state).await {
                tracing::warn!(session, error = %err, "saving run state failed");
            }
        if outcome.state.pending.is_some() || self.generation(session) != generation {
            return;
        }
        if self.store.is_some() {
            self.index.set_latest(session, outcome.state.run_id).await;
        }
        let mut sessions = self.sessions.lock().expect("sessions poisoned");
        if let Some(slot) = sessions.get_mut(session)
            && slot.generation == generation {
                slot.last_state = Some(outcome.state.clone());
            }
    }

    /// Execute a run while showing progress and typing.
    async fn drive<F>(
        &self,
        session: &str,
        origin: &Origin,
        run_id: Option<RunId>,
        run: F,
    ) -> Result<RunOutcome, String>
    where
        F: Future<Output = Result<RunOutcome, CoreError>> + Send + 'static,
    {
        let channel = self.channels[origin.channel].clone();
        let caps = channel.capabilities();
        let mut events = self.router.as_ref().map(|r| r.attach(session, run_id));
        let mut handle = tokio::spawn(run);

        let mut progress = Progress::default();
        let mut status_ref = None;
        if caps.edit_messages && events.is_some() {
            match channel
                .send_status(&origin.conversation, &progress.working())
                .await
            {
                Ok(r) => status_ref = Some(r),
                Err(err) => tracing::warn!(error = %err, "could not post status message"),
            }
        }
        let interval = self.config.status_edit_interval;
        let mut last_edit = Instant::now();
        let mut dirty = false;
        let mut typing = tokio::time::interval(self.config.typing_interval);
        typing.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let result = loop {
            tokio::select! {
                joined = &mut handle => {
                    break match joined {
                        Ok(Ok(outcome)) => Ok(outcome),
                        Ok(Err(err)) => Err(err.to_string()),
                        Err(err) => Err(format!("run task failed: {err}")),
                    };
                }
                event = recv_opt(&mut events) => match event {
                    Some(event) => {
                        if progress.apply(&event) {
                            dirty = true;
                        }
                    }
                    None => events = None,
                },
                _ = typing.tick(), if caps.typing_indicator => {
                    let _ = channel.typing(&origin.conversation).await;
                }
                _ = tokio::time::sleep_until(last_edit + interval), if dirty && status_ref.is_some() => {
                    if let Some(r) = &status_ref
                        && let Err(err) = channel.update_status(r, &progress.working()).await {
                            tracing::debug!(error = %err, "status edit failed");
                        }
                    last_edit = Instant::now();
                    dirty = false;
                }
            }
        };

        if let Some(rx) = events.as_mut() {
            while let Ok(event) = rx.try_recv() {
                progress.apply(&event);
            }
        }
        if let Some(router) = &self.router {
            router.detach(session);
        }
        if let Some(r) = &status_ref {
            let ending = match &result {
                Ok(outcome) => render::outcome_ending(outcome),
                Err(_) => RunEnding::Failed,
            };
            if let Err(err) = channel.update_status(r, &progress.ended(ending)).await {
                tracing::debug!(error = %err, "final status edit failed");
            }
        }
        result
    }

    fn new_prompt_id(&self) -> String {
        let approvals = self.approvals.lock().expect("approvals poisoned");
        loop {
            let id: String = RunId::new().to_string().chars().take(8).collect();
            if !approvals.contains_key(&id) {
                return id;
            }
        }
    }

    /// Register the approval, show the prompt, and arm its expiry.
    async fn open_approval(self: &Arc<Self>, pending: PendingApproval) {
        let prompt_id = pending.prompt_id.clone();
        let origin = pending.origin.clone();
        let mut prompt = ApprovalPrompt::new(
            prompt_id.clone(),
            origin.conversation.clone(),
            pending.prompt_text.clone(),
            vec![
                ApprovalChoice {
                    id: APPROVE_CHOICE.to_string(),
                    label: "Approve".to_string(),
                },
                ApprovalChoice {
                    id: DENY_CHOICE.to_string(),
                    label: "Deny".to_string(),
                },
            ],
        );
        if let Some(waiting) = &pending.state.pending {
            prompt = prompt.with_details(ApprovalDetails::from_interruption(&waiting.interruption));
        }
        // Saved before it is shown, so a prompt on screen is always one a
        // restart can still settle.
        self.persist_approval(&pending).await;
        self.approvals
            .lock()
            .expect("approvals poisoned")
            .insert(prompt_id.clone(), pending);

        let channel = self.channels[origin.channel].clone();
        // A channel without buttons is never asked for them: it gets the
        // text prompt and its commands, whatever its `ask` would do.
        let asked = if channel.capabilities().buttons {
            channel.ask(&prompt).await
        } else {
            Err(ChannelError::Unsupported("buttons"))
        };
        let prompt_ref = match asked {
            Ok(r) => Some(r),
            Err(err) => {
                if !matches!(err, ChannelError::Unsupported(_)) {
                    tracing::warn!(error = %err, "approval buttons failed; using text commands");
                }
                let text = format!(
                    "{}\n\n{}",
                    prompt.text,
                    render::approval_fallback(&prompt_id)
                );
                let message = OutboundMessage::text(render::truncate_chars(
                    &text,
                    channel.capabilities().max_message_len,
                ));
                match channel.send(&origin.conversation, &message).await {
                    Ok(r) => Some(r),
                    Err(err) => {
                        tracing::error!(error = %err, "could not show approval prompt");
                        None
                    }
                }
            }
        };
        let updated = {
            let mut approvals = self.approvals.lock().expect("approvals poisoned");
            approvals.get_mut(&prompt_id).and_then(|entry| {
                entry.prompt_ref = prompt_ref;
                self.stored_approval(entry)
            })
        };
        // Only while still open: an answer may have settled it during `ask`.
        if let (Some(store), Some(record)) = (&self.approval_store, updated)
            && let Err(err) = store.save(&record).await {
                tracing::warn!(prompt = %prompt_id, error = %err, "saving approval failed");
            }

        self.arm_expiry(prompt_id, self.config.approval_timeout);
    }

    /// Deny the approval when `after` has passed, unless it is settled first.
    fn arm_expiry(self: &Arc<Self>, prompt_id: String, after: Duration) {
        let inner = self.clone();
        self.tasks.spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(after) => {}
                _ = inner.root.cancelled() => return,
            }
            let expired = inner
                .approvals
                .lock()
                .expect("approvals poisoned")
                .remove(&prompt_id);
            if let Some(pending) = expired {
                inner.forget_approval(&prompt_id).await;
                inner.mark_prompt(&pending, "Expired (denied)", None).await;
                inner
                    .say(
                        &pending.origin,
                        &format!("Approval `{prompt_id}` expired, so the action was denied."),
                    )
                    .await;
                inner.resume(
                    pending,
                    Resolution::Deny {
                        reason: "The approval request expired without an answer.".to_string(),
                    },
                    false,
                );
            }
        });
    }

    /// The durable form of an open approval, if approvals are persisted.
    fn stored_approval(&self, pending: &PendingApproval) -> Option<StoredApproval> {
        self.approval_store.as_ref()?;
        Some(StoredApproval {
            prompt_id: pending.prompt_id.clone(),
            session_id: pending.session_id.clone(),
            channel: self.channel_name(pending.origin.channel).to_string(),
            conversation: pending.origin.conversation.clone(),
            reply_to: pending.origin.reply_to.clone(),
            run_id: pending.state.run_id,
            prompt_text: pending.prompt_text.clone(),
            prompt_ref: pending.prompt_ref.clone(),
            expires_at_ms: pending.expires_at_ms,
        })
    }

    async fn persist_approval(&self, pending: &PendingApproval) {
        if let (Some(store), Some(record)) = (&self.approval_store, self.stored_approval(pending))
            && let Err(err) = store.save(&record).await {
                tracing::warn!(prompt = %pending.prompt_id, error = %err, "saving approval failed");
            }
    }

    /// Drop the durable record of an approval that has been settled.
    async fn forget_approval(&self, prompt_id: &str) {
        if let Some(store) = &self.approval_store
            && let Err(err) = store.remove(prompt_id).await {
                tracing::warn!(prompt = prompt_id, error = %err, "removing approval failed");
            }
    }

    /// Bring back the approvals open at the last stop, each with its session
    /// waiting on it. Returns how long each has left, for [`Self::arm_expiry`].
    async fn restore_approvals(&self) -> Vec<(String, Duration)> {
        let (Some(approvals), Some(store)) = (&self.approval_store, &self.store) else {
            return Vec::new();
        };
        let records = match approvals.load_all().await {
            Ok(records) => records,
            Err(err) => {
                tracing::error!(error = %err, "could not load pending approvals");
                return Vec::new();
            }
        };
        let now = now_ms();
        let mut restored = Vec::new();
        for record in records {
            let Some(channel) = self
                .channels
                .iter()
                .position(|c| c.name() == record.channel)
            else {
                // Not configured on this start; it may be back on the next.
                if record.expires_at_ms <= now {
                    self.forget_approval(&record.prompt_id).await;
                }
                continue;
            };
            let state = match store.load_state(record.run_id).await {
                Ok(Some(state)) if state.pending.is_some() && state.agent_id == self.agent.id() => {
                    state
                }
                Ok(_) => {
                    tracing::warn!(prompt = %record.prompt_id, run = %record.run_id, "approval has no interrupted run to resume; dropping it");
                    self.forget_approval(&record.prompt_id).await;
                    continue;
                }
                Err(err) => {
                    tracing::warn!(prompt = %record.prompt_id, error = %err, "could not load the run of a pending approval");
                    continue;
                }
            };
            let remaining = Duration::from_millis(record.expires_at_ms.saturating_sub(now));
            self.sessions
                .lock()
                .expect("sessions poisoned")
                .entry(record.session_id.clone())
                .or_default()
                .state = SlotState::Awaiting {
                prompt_id: record.prompt_id.clone(),
            };
            restored.push((record.prompt_id.clone(), remaining));
            self.approvals.lock().expect("approvals poisoned").insert(
                record.prompt_id.clone(),
                PendingApproval {
                    prompt_id: record.prompt_id,
                    session_id: record.session_id,
                    origin: Origin {
                        channel,
                        conversation: record.conversation,
                        reply_to: record.reply_to,
                    },
                    state,
                    prompt_text: record.prompt_text,
                    prompt_ref: record.prompt_ref,
                    expires_at_ms: record.expires_at_ms,
                },
            );
        }
        restored
    }

    /// Show on the prompt how it was settled, which also removes its
    /// buttons. `pressed` is the message a button press came from; it is the
    /// way to the prompt when its own ref was never recorded (an answer
    /// that raced `ask`, or a restart before it was saved).
    async fn mark_prompt(
        &self,
        pending: &PendingApproval,
        verdict: &str,
        pressed: Option<&MessageRef>,
    ) {
        let channel = &self.channels[pending.origin.channel];
        let caps = channel.capabilities();
        let Some(target) = pending.prompt_ref.as_ref().or(pressed) else {
            return;
        };
        if caps.edit_messages {
            // Room for the verdict first, since an edit keeps only what fits.
            let room = caps
                .max_message_len
                .saturating_sub(verdict.chars().count() + 6);
            let text = format!(
                "{}\n\n**{verdict}**",
                render::truncate_chars(&pending.prompt_text, room)
            );
            match channel.edit(target, &text).await {
                Ok(()) => return,
                Err(err) => tracing::debug!(error = %err, "could not mark the prompt"),
            }
        }
        if caps.buttons {
            self.remove_buttons(pending.origin.channel, target).await;
        }
    }

    async fn remove_buttons(&self, channel: usize, message: &MessageRef) {
        if let Err(err) = self.channels[channel].remove_buttons(message).await {
            tracing::debug!(error = %err, "could not remove the buttons");
        }
    }

    /// Hand a settled approval back to its session's worker.
    fn resume(self: &Arc<Self>, pending: PendingApproval, resolution: Resolution, stopped: bool) {
        let job = Job::Resume {
            origin: pending.origin,
            state: Box::new(pending.state),
            resolution,
            stopped,
        };
        let start = {
            let mut sessions = self.sessions.lock().expect("sessions poisoned");
            let slot = sessions.entry(pending.session_id.clone()).or_default();
            match slot.state {
                SlotState::Running { .. } => {
                    // Should not happen: a session waiting for approval has
                    // no worker. Run the resume next rather than lose it.
                    slot.queue.push_front(job);
                    None
                }
                _ => {
                    slot.state = SlotState::Running {
                        cancel: self.root.child_token(),
                    };
                    Some(job)
                }
            }
        };
        if let Some(job) = start {
            self.spawn_worker(pending.session_id, job);
        }
    }

    /// An approval answer (button or command) from `sender_id`.
    async fn answer(
        self: &Arc<Self>,
        origin: Origin,
        sender_id: &str,
        prompt_id: Option<String>,
        approve: bool,
        pressed: Option<MessageRef>,
    ) {
        let name = self.channel_name(origin.channel).to_string();
        if !self.policy.may_approve(&name, sender_id) {
            tracing::info!(channel = %name, sender = sender_id, "ignoring approval from a user who may not approve");
            return;
        }
        let session = session_id(&name, &origin.conversation.id);
        let settled = {
            let mut approvals = self.approvals.lock().expect("approvals poisoned");
            let id = prompt_id.clone().or_else(|| {
                let mut ids = approvals
                    .iter()
                    .filter(|(_, p)| p.session_id == session)
                    .map(|(id, _)| id.clone());
                match (ids.next(), ids.next()) {
                    (Some(only), None) => Some(only),
                    _ => None,
                }
            });
            match id {
                Some(id) if approvals.get(&id).is_some_and(|p| p.session_id == session) => {
                    approvals.remove(&id).map(|p| (id, p))
                }
                _ => None,
            }
        };
        let Some((id, pending)) = settled else {
            let notice = match prompt_id {
                Some(id) => {
                    format!("Approval `{id}` is not pending here (already answered or expired).")
                }
                None => {
                    "There is no single pending approval here; use `/approve <id>` or `/deny <id>`."
                        .to_string()
                }
            };
            // A button that can no longer do anything should not stay.
            if let Some(message) = &pressed {
                self.remove_buttons(origin.channel, message).await;
            }
            self.say(&origin, &notice).await;
            return;
        };
        tracing::info!(session = %session, prompt = %id, approve, "approval settled");
        self.forget_approval(&id).await;
        self.mark_prompt(
            &pending,
            if approve { "Approved" } else { "Denied" },
            pressed.as_ref(),
        )
        .await;
        let resolution = if approve {
            Resolution::Approve
        } else {
            Resolution::Deny {
                reason: "The user denied this action.".to_string(),
            }
        };
        self.resume(pending, resolution, false);
    }

    async fn stop(self: &Arc<Self>, session: &str, origin: &Origin) {
        enum Action {
            Cancelled(usize),
            Settle(Box<PendingApproval>, usize),
            Nothing,
        }
        let action = {
            let mut sessions = self.sessions.lock().expect("sessions poisoned");
            match sessions.get_mut(session) {
                Some(slot) => {
                    let dropped = slot.queue.len();
                    slot.queue.clear();
                    match &slot.state {
                        SlotState::Running { cancel } => {
                            cancel.cancel();
                            Action::Cancelled(dropped)
                        }
                        SlotState::Awaiting { prompt_id } => {
                            let pending = self
                                .approvals
                                .lock()
                                .expect("approvals poisoned")
                                .remove(prompt_id);
                            match pending {
                                Some(pending) => Action::Settle(Box::new(pending), dropped),
                                // Being settled right now by an answer.
                                None => Action::Cancelled(dropped),
                            }
                        }
                        SlotState::Idle => Action::Nothing,
                    }
                }
                None => Action::Nothing,
            }
        };
        match action {
            Action::Nothing => self.say(origin, "Nothing is running.").await,
            Action::Cancelled(dropped) => {
                if dropped > 0 {
                    self.say(origin, &format!("Dropped {dropped} queued message(s)."))
                        .await;
                }
            }
            Action::Settle(pending, dropped) => {
                if dropped > 0 {
                    self.say(origin, &format!("Dropped {dropped} queued message(s)."))
                        .await;
                }
                self.forget_approval(&pending.prompt_id).await;
                self.mark_prompt(&pending, "Stopped", None).await;
                self.resume(
                    *pending,
                    Resolution::Deny {
                        reason: "The user stopped the run.".to_string(),
                    },
                    true,
                );
            }
        }
    }

    async fn reset(self: &Arc<Self>, session: &str, origin: &Origin) {
        let dropped_approval = {
            let mut sessions = self.sessions.lock().expect("sessions poisoned");
            let slot = sessions.entry(session.to_string()).or_default();
            slot.generation += 1;
            slot.last_state = None;
            slot.queue.clear();
            match &slot.state {
                SlotState::Running { cancel } => {
                    cancel.cancel();
                    None
                }
                SlotState::Awaiting { prompt_id } => {
                    let pending = self
                        .approvals
                        .lock()
                        .expect("approvals poisoned")
                        .remove(prompt_id);
                    slot.state = SlotState::Idle;
                    pending
                }
                SlotState::Idle => None,
            }
        };
        if let Some(pending) = dropped_approval {
            self.forget_approval(&pending.prompt_id).await;
            self.mark_prompt(&pending, "Cancelled", None).await;
        }
        self.index.clear(session).await;
        self.say(origin, "Started a new conversation.").await;
    }

    async fn status(&self, session: &str, origin: &Origin) {
        let text = {
            let sessions = self.sessions.lock().expect("sessions poisoned");
            match sessions.get(session) {
                Some(slot) => {
                    let queued = slot.queue.len();
                    match &slot.state {
                        SlotState::Idle => format!("Idle. Queued messages: {queued}."),
                        SlotState::Running { .. } => {
                            format!("A run is active. Queued messages: {queued}.")
                        }
                        SlotState::Awaiting { prompt_id } => format!(
                            "Waiting for approval `{prompt_id}`. Queued messages: {queued}."
                        ),
                    }
                }
                None => "Idle. Queued messages: 0.".to_string(),
            }
        };
        self.say(origin, &text).await;
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis() as u64)
}

/// The last assistant text produced by this run (messages from `base` on).
fn partial_text(state: &RunState, base: usize) -> Option<String> {
    state
        .messages
        .get(base.min(state.messages.len())..)?
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant && !m.content.trim().is_empty())
        .map(|m| m.content.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_are_parsed_from_explicit_syntax_only() {
        assert!(matches!(parse_command("/stop"), Some(Command::Stop)));
        assert!(matches!(
            parse_command(" /stop@my_bot "),
            Some(Command::Stop)
        ));
        assert!(matches!(parse_command("/NEW"), Some(Command::New)));
        assert!(matches!(
            parse_command("/approve ab12cd34"),
            Some(Command::Approve(Some(id))) if id == "ab12cd34"
        ));
        assert!(matches!(parse_command("/deny"), Some(Command::Deny(None))));
        assert!(parse_command("please stop").is_none());
        assert!(parse_command("/unknown thing").is_none());
    }
}
