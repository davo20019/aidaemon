//! The whole client, with no rendering: the reducer's [`State`], the
//! connection as the owner should see it, the commands waiting for the
//! link, and everything the client keeps between frames that is not a
//! screen's own business: the reads it asked the daemon for and their
//! answers (matched by request id), each turn's activity, which
//! conversations wrote code, what is unread, what is stopping, files
//! uploading with a draft, settings changes in flight.
//!
//! A UI holds one [`Client`], calls its methods for what the owner does,
//! hands it the link's [`Update`]s with a [`Context`] (the few facts only
//! the screen knows: what is selected, whether the owner is looking), sends
//! what [`Client::take_outgoing`] returns, and acts on the [`Effect`]s it
//! gets back. Nothing here knows how anything is drawn.

use std::collections::{BTreeMap, BTreeSet};

use aidaemon_client_protocol::{
    AgentMode, Approval, ApprovalDecision, AttachmentRef, AutomationChange, AutomationNotify,
    AutomationSummary, Command, CommandResult, Effort, ErrorCode, Event, LiveTurnState, MemoryItem,
    MidRun, ModelChoice, ModelOptions, ProjectDir, ProjectFile, ProjectSummary, QueuedMessage,
    SearchHit, ServerFrame, SessionChanges, SessionModel, Settings, SettingsChange,
    SettingsOutcome, SetupAction, SetupStep, SlashCommand, StreamAccept, SurfaceController,
    SurfaceInfo, SurfaceMode, SurfaceTransport, SystemStatus, TaskSummary, TurnState, TurnStatus,
};

use crate::surfaces::LiveView;

use crate::machine::{DropKind, Update, request_id};
use crate::state::{Line, State, ViewRequest};

/// The connection to the daemon, as the owner should see it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Connection {
    #[default]
    Connecting,
    Online,
    /// This device has no key the daemon knows: pair it first.
    NeedsPairing,
    /// No daemon to sign in to yet (no local token, no paired one): run one
    /// here, or connect to one elsewhere.
    NoDaemon,
    Offline {
        kind: DropKind,
        reason: String,
        /// When the link tries again, in ms since the epoch.
        retry_at_ms: u64,
        /// When a working connection was lost, through every failed try
        /// since; `None` when it never came up.
        since_ms: Option<u64>,
    },
}

/// A drop this short is not news: a daemon restart or a phone waking a
/// page is back by then.
pub const QUIET_DROP_MS: u64 = 5_000;
/// After this long, a drop is an outage worth alarm.
pub const LONG_DROP_MS: u64 = 30_000;

/// The connection as the owner should feel it at a moment: a short drop
/// reads as still connected, a longer one as reconnecting, and only a
/// long one (or a refusal) as down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Felt {
    Online,
    Connecting,
    NeedsPairing,
    NoDaemon,
    Reconnecting,
    Down { kind: DropKind, reason: String },
}

impl Connection {
    /// How this reads at `now_ms`.
    pub fn felt(&self, now_ms: u64) -> Felt {
        match self {
            Connection::Online => Felt::Online,
            Connection::Connecting => Felt::Connecting,
            Connection::NeedsPairing => Felt::NeedsPairing,
            Connection::NoDaemon => Felt::NoDaemon,
            Connection::Offline {
                kind,
                reason,
                since_ms,
                ..
            } => {
                let down_for = since_ms.map_or(u64::MAX, |since| now_ms.saturating_sub(since));
                let refused = matches!(kind, DropKind::Unauthorized | DropKind::VersionUnsupported);
                if refused || down_for >= LONG_DROP_MS {
                    Felt::Down {
                        kind: *kind,
                        reason: reason.clone(),
                    }
                } else if down_for >= QUIET_DROP_MS {
                    Felt::Reconnecting
                } else {
                    Felt::Online
                }
            }
        }
    }

    /// Whether the connection is down, however briefly.
    pub fn is_offline(&self) -> bool {
        matches!(self, Connection::Offline { .. })
    }
}

/// The protocol minor that added `get_settings`.
const GET_SETTINGS_MINOR: u16 = 4;

/// Something read from the daemon, as last asked.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Read<T> {
    #[default]
    NotYet,
    Loading,
    Ready(T),
    Failed(String),
}

impl<T> Read<T> {
    pub fn is_ready(&self) -> bool {
        matches!(self, Read::Ready(_))
    }

    pub fn ready(&self) -> Option<&T> {
        match self {
            Read::Ready(value) => Some(value),
            _ => None,
        }
    }
}

/// The System page's data, as last read, and when.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SystemRead {
    #[default]
    NotYet,
    Loading,
    Ready {
        status: Box<SystemStatus>,
        at_ms: u64,
    },
    Failed(String),
}

/// A search of every conversation, as last asked.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SearchRead {
    #[default]
    Idle,
    Loading(String),
    Ready {
        query: String,
        hits: Vec<SearchHit>,
    },
    Failed(String),
}

/// One thing a turn did, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Activity {
    pub id: u64,
    pub kind: ActivityKind,
    pub title: String,
    pub detail: String,
    pub at_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityKind {
    Thinking,
    Tool,
    ToolDone,
    Output,
    Plan,
    Note,
}

/// A turn's activity, kept after it ends so it can still be read.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TurnLog {
    pub session_id: String,
    /// When this client saw it start.
    pub started_ms: u64,
    pub items: Vec<Activity>,
    pub done: bool,
}

/// Where a conversation came from: the app or a chat platform, and
/// whether others are in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    All,
    App,
    Telegram,
    Slack,
    Discord,
    Chat,
    /// Conversations with other people in them, on any platform.
    Groups,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::All => "All",
            Source::App => "App",
            Source::Telegram => "Telegram",
            Source::Slack => "Slack",
            Source::Discord => "Discord",
            Source::Chat => "Other chats",
            Source::Groups => "Groups",
        }
    }
}

/// A chat bot the daemon has conversations through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bot {
    /// Its sessions' `channel`: `telegram:coding_bot`.
    pub channel: String,
    /// "Telegram · coding_bot".
    pub name: String,
    /// How many conversations it has.
    pub chats: usize,
    /// The project its direct-message chats go to.
    pub project: Option<String>,
}

/// A file attached to the message being written.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftFile {
    pub number: u64,
    pub name: String,
    pub state: DraftFileState,
    /// A copy of a sent picture the owner drew marks over: that picture's
    /// attachment id.
    pub marked_from: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DraftFileState {
    Uploading,
    Ready(AttachmentRef),
    Failed(String),
}

/// The largest file the gateway takes.
pub const MAX_FILE_BYTES: usize = 10 * 1024 * 1024;

/// Tools whose work can change files: the changes read again after one.
const CHANGING_TOOLS: [&str; 4] = ["edit_file", "write_file", "terminal", "cli_agent"];
/// Tools that write files themselves: the first in a turn shows the changes.
const WRITING_TOOLS: [&str; 2] = ["edit_file", "write_file"];

/// What only the screen knows, handed in with each update.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Context {
    /// The wall clock, ms since the epoch.
    pub now_ms: u64,
    /// The conversation on screen.
    pub selected: Option<String>,
    /// The owner is looking at the selected conversation now (its page is
    /// up and the app is in front): its replies are not unread.
    pub looking: bool,
    /// The changes of the selected conversation are on screen: a tool
    /// that can change files reads them again at once, else on next
    /// showing.
    pub changes_visible: bool,
    /// The Settings page is up: a settings change reads them again.
    pub settings_open: bool,
}

/// What the UI should do after an update or a command.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    /// Tell the owner, briefly.
    Say(String),
    /// A message sent from a new conversation started it: show it (the
    /// new conversation's draft is spent).
    Started(String),
    /// The greeting's conversation the setup asked to open.
    OpenGreeting(String),
    /// `session` was just found to work on code: its details belong
    /// beside it.
    NowCoding(String),
    /// Show the changes of `session` (if it is on screen).
    OpenChanges(String),
    /// A project was made or saved.
    ProjectSaved(ProjectSummary),
    /// The agent asked for a view move.
    View(ViewRequest),
    /// A live view opened: connect to its frame socket (protocol 1.30).
    OpenLive {
        live_id: String,
        transport: SurfaceTransport,
    },
    /// The live view ended (closed here, or its surface went away): close
    /// its frame socket.
    CloseLive,
}

/// The client, as [`crate::client`] describes it.
#[derive(Debug, Clone, Default)]
pub struct Client {
    pub state: State,
    pub connection: Connection,
    /// Commands waiting to go to the link, in order.
    outgoing: Vec<(String, Command)>,
    /// Sessions whose history was asked for.
    fetched: BTreeSet<String>,
    /// The request of a message that starts a new session.
    starting: Option<String>,
    turns: BTreeMap<String, TurnLog>,
    next_activity: u64,

    pub system: SystemRead,
    system_request: Option<String>,
    pub settings: Read<Box<Settings>>,
    settings_request: Option<String>,
    /// The settings were asked for again while a read was out: read once
    /// more when it lands, so a change made meanwhile is shown.
    settings_again: bool,
    /// Settings changes sent and not yet answered.
    settings_changes: BTreeSet<String>,
    /// Why the last settings change was refused, until dismissed or the
    /// next change.
    pub settings_error: Option<String>,
    /// The protocol version the daemon said it speaks when this socket
    /// opened.
    pub daemon_version: Option<aidaemon_client_protocol::ProtocolVersion>,
    /// The providers and models a conversation can answer with.
    pub models: Read<ModelOptions>,
    models_request: Option<String>,
    /// What each app conversation answers with, as last read or set.
    pub session_models: BTreeMap<String, Read<SessionModel>>,
    /// Reads and changes of a conversation's model: request id to session.
    session_model_requests: BTreeMap<String, String>,
    /// `/model` and `/auto` sent in a conversation: request id to session,
    /// so its model is read again once the command has run.
    model_commands: BTreeMap<String, String>,
    /// The model and effort the next new conversation starts with.
    pub new_model: (ModelChoice, Effort),
    /// The model a new conversation's first message carried, until the
    /// daemon names the conversation.
    starting_model: Option<(ModelChoice, Effort)>,
    /// Settings changes waiting for an approval, and whether another
    /// device must give it.
    pub settings_waiting: BTreeMap<String, bool>,
    pub memories: Read<Vec<MemoryItem>>,
    memories_request: Option<String>,
    pub automations: Read<Vec<AutomationSummary>>,
    automations_request: Option<String>,
    /// The automation a change was sent for and not answered yet.
    pub automation_busy: Option<String>,
    pub search: SearchRead,
    search_request: Option<String>,
    /// The selected session's changes, and the session they are of.
    pub changes: Read<SessionChanges>,
    changes_of: Option<String>,
    changes_request: Option<String>,
    /// Something changed while a read was in flight: read again after it.
    changes_again: bool,
    /// A project's Files view (protocol 1.32): the folder open, and the
    /// file open in it. Only the latest read of each is kept.
    pub files_dir: Read<ProjectDir>,
    files_dir_request: Option<String>,
    /// What the unanswered folder read asked: the same read again is not
    /// sent (a view drawn again while offline asked once per draw).
    files_dir_asked: Option<Command>,
    pub files_file: Read<ProjectFile>,
    files_file_request: Option<String>,
    files_file_asked: Option<Command>,
    /// The daemon's slash commands.
    pub commands: Vec<SlashCommand>,
    project_request: Option<String>,
    pub project_error: Option<String>,
    /// The folder the last save named, as typed.
    project_path: Option<String>,
    /// The daemon refused the last save only because this folder (as
    /// typed) is missing: saving again with `create_folder` makes it.
    pub project_missing_folder: Option<String>,
    setup_request: Option<String>,
    pub setup_error: Option<String>,
    /// Open the next conversation the daemon starts (the greeting).
    open_greeting: bool,
    /// Files attached to the message being written.
    pub draft_files: Vec<DraftFile>,
    /// Uploads not yet answered: request id to the draft file's number.
    uploads: BTreeMap<String, u64>,
    next_upload: u64,
    /// Conversations that got a reply the owner has not looked at.
    pub unread: BTreeSet<String>,
    /// Conversations asked to stop, until their turn ends.
    stopping: BTreeSet<String>,
    /// The latest turn that finished: its session and when.
    pub finished: Option<(String, u64)>,
    /// Conversations that wrote files or showed code changes.
    coded: BTreeSet<String>,
    /// Each conversation's messages waiting for its running work, as last
    /// read.
    pub queued: BTreeMap<String, Vec<QueuedMessage>>,
    /// Messages sent while the work runs (to be read now or after), until
    /// accepted: the waiting list reads again then.
    queue_sends: BTreeSet<String>,
    /// Stop requests in flight, by request id: the conversation each was
    /// for, so its report lands there and not on every page.
    stop_sends: BTreeMap<String, String>,
    /// The file undo in flight: its request and the file.
    revert_request: Option<(String, String)>,
    /// A project delete in flight, and the project's name.
    delete_project_request: Option<(String, String)>,
    /// Turns that already showed the changes, so the owner's own choice
    /// sticks for the rest of a turn.
    changes_shown: BTreeSet<String>,
    /// The live view open on this client (protocol 1.30), and the request
    /// that opens it.
    pub live: Option<LiveView>,
    live_request: Option<String>,
    /// Why the last take or release of control was refused.
    pub control_error: Option<String>,
    control_requests: BTreeSet<String>,
    /// Closing a live view: the view going away says it, so the daemon's
    /// answer ("Closed", "Already closed") is no notice.
    close_requests: BTreeSet<String>,
    /// This device asked for control and has not given it back: the
    /// surface's owner controller is this device.
    live_controller_is_mine: bool,
    /// The paired devices (protocol 1.31), and whether this client may
    /// change what they may do.
    pub devices: Read<(Vec<aidaemon_client_protocol::DeviceInfo>, bool)>,
    devices_request: Option<String>,
    /// Why the last device change was refused.
    pub devices_error: Option<String>,
    /// The reads of a refresh the owner asked for (or the app's return),
    /// until each is answered: the refresh indicator spins meanwhile.
    refreshing: BTreeSet<String>,
    /// When the app went out of sight (ms since the epoch), until it is
    /// back.
    away_since: Option<u64>,
}

/// Away at least this long, the app reads everything again on its return:
/// a phone may have frozen the page, its socket's stream with it.
pub const AWAY_REFRESH_MS: u64 = 30_000;

impl Client {
    pub fn new() -> Self {
        Self::default()
    }

    // ----- The link ---------------------------------------------------------

    /// Queue `command` for the link; its request id.
    pub fn send(&mut self, command: Command) -> String {
        let id = request_id();
        self.outgoing.push((id.clone(), command));
        id
    }

    /// The commands to send now, in order.
    pub fn take_outgoing(&mut self) -> Vec<(String, Command)> {
        std::mem::take(&mut self.outgoing)
    }

    /// Commands waiting for the link, oldest first.
    pub fn outgoing(&self) -> &[(String, Command)] {
        &self.outgoing
    }

    pub fn needs_pairing(&mut self) {
        self.connection = Connection::NeedsPairing;
    }

    pub fn needs_daemon(&mut self) {
        self.connection = Connection::NoDaemon;
    }

    pub fn is_online(&self) -> bool {
        self.connection == Connection::Online
    }

    /// What the link reported.
    pub fn update(&mut self, update: Update, cx: &Context) -> Vec<Effect> {
        match update {
            Update::Connecting => {
                if !matches!(self.connection, Connection::Offline { .. }) {
                    self.connection = Connection::Connecting;
                }
                Vec::new()
            }
            Update::Connected => {
                self.connection = Connection::Online;
                self.send(Command::ListSessions {
                    cursor: None,
                    limit: 200,
                });
                self.send(Command::ListCommands);
                self.send(Command::ListProjects);
                // Today's token use, for the sidebar; what automations did,
                // for their news.
                self.refresh_system();
                self.refresh_automations();
                // A reconnect may have missed history: ask again on view.
                self.fetched.clear();
                // A read sent before a daemon restart may never be answered:
                // read the settings afresh, or the page keeps what it showed
                // before every change since.
                self.settings_request = None;
                self.settings_again = false;
                // So are a project's files: shown as last read until asked
                // again.
                if self.files_dir_request.take().is_some() && !self.files_dir.is_ready() {
                    self.files_dir = Read::NotYet;
                }
                if self.files_file_request.take().is_some() {
                    self.files_file = Read::NotYet;
                }
                if self.settings.is_ready() || cx.settings_open {
                    self.refresh_settings();
                }
                if let Some(id) = &cx.selected {
                    // Held lines may have missed what was said meanwhile:
                    // read it again (history merges, never doubles).
                    self.read_history(id);
                    // Its changes too: whether it works on code, and what.
                    self.changes_of = None;
                    self.refresh_changes(cx.selected.as_deref());
                }
                Vec::new()
            }
            Update::Disconnected(disconnect) => {
                // Changes in flight get no answer now.
                self.settings_changes.clear();
                self.refreshing.clear();
                let retry = u64::try_from(disconnect.retry_in.as_millis()).unwrap_or(0);
                // Only a connection that worked drops quietly: one that
                // never came up is news at once.
                let since_ms = match self.connection {
                    Connection::Offline { since_ms, .. } => since_ms,
                    Connection::Online => Some(cx.now_ms),
                    _ => None,
                };
                self.connection = Connection::Offline {
                    kind: disconnect.kind,
                    reason: disconnect.reason,
                    retry_at_ms: cx.now_ms.saturating_add(retry),
                    since_ms,
                };
                Vec::new()
            }
            Update::Frame(frame) => self.frame(frame, cx),
        }
    }

    fn frame(&mut self, frame: ServerFrame, cx: &Context) -> Vec<Effect> {
        let mut effects = Vec::new();
        if let ServerFrame::Ack { request_id, .. }
        | ServerFrame::Error {
            request_id: Some(request_id),
            ..
        } = &frame
        {
            self.refreshing.remove(request_id);
        }
        if let ServerFrame::Welcome { version, .. } = &frame {
            self.daemon_version = Some(*version);
        }
        if self.live_answer(&frame, &mut effects) {
            return effects;
        }
        if self.answer(&frame, cx, &mut effects) {
            return effects;
        }
        if let ServerFrame::Event {
            event: Event::SurfaceClosed { surface_id },
            ..
        } = &frame
            && self
                .live
                .as_ref()
                .is_some_and(|l| &l.surface_id == surface_id)
        {
            self.live = None;
            self.live_request = None;
            effects.push(Effect::CloseLive);
        }
        self.state.apply(frame);
        if let Some(request) = self.state.take_view_request() {
            effects.push(Effect::View(request));
        }
        effects
    }

    /// What the client does with `frame` besides the reducer: answers to
    /// its own requests, and the notes it keeps from events. `true` when
    /// the frame was the answer to a request and the reducer has nothing
    /// to learn from it.
    fn answer(&mut self, frame: &ServerFrame, cx: &Context, out: &mut Vec<Effect>) -> bool {
        let mine = |slot: &Option<String>, id: &str| slot.as_deref() == Some(id);
        match frame {
            // A stop's report belongs to the conversation stopped: a note
            // there, not the app-wide notice that followed the owner onto
            // every other page until dismissed (live 2026-11-18).
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Stopped { report },
            } if self.stop_sends.contains_key(request_id) => {
                if let Some(session) = self.stop_sends.remove(request_id) {
                    self.state.note(&session, report.clone());
                }
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Uploaded { attachment },
            } => {
                if let Some(number) = self.uploads.remove(request_id)
                    && let Some(file) = self.draft_files.iter_mut().find(|f| f.number == number)
                {
                    file.state = DraftFileState::Ready(attachment.clone());
                }
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self.uploads.contains_key(request_id) => {
                let number = self.uploads.remove(request_id).unwrap_or(0);
                if let Some(file) = self.draft_files.iter_mut().find(|f| f.number == number) {
                    file.state = DraftFileState::Failed(error.message.clone());
                }
                out.push(Effect::Say(format!(
                    "Could not attach the file: {}",
                    error.message
                )));
                return true;
            }
            _ => {}
        }
        if let ServerFrame::Ack {
            request_id,
            result: CommandResult::MessageAccepted { session_id, .. },
        } = frame
            && mine(&self.starting, request_id)
        {
            self.starting = None;
            self.fetched.insert(session_id.clone());
            if let Some((model, effort)) = self.starting_model.take() {
                self.session_models.insert(
                    session_id.clone(),
                    Read::Ready(SessionModel {
                        session_id: session_id.clone(),
                        model,
                        effort,
                    }),
                );
            }
            out.push(Effect::Started(session_id.clone()));
        }
        match frame {
            // `/model` or `/auto` changed the conversation's model: read it
            // again, and let the reply show as usual.
            ServerFrame::Ack { request_id, .. } if self.model_commands.contains_key(request_id) => {
                let session = self.model_commands.remove(request_id).unwrap_or_default();
                self.session_models.remove(&session);
                self.load_session_model(&session);
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Setup { .. },
            } if mine(&self.setup_request, request_id) => {
                self.setup_request = None;
                self.setup_error = None;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.setup_request, request_id) => {
                self.setup_request = None;
                self.setup_error = Some(error.message.clone());
                return true;
            }
            // In setup mode the usual reads on connecting are refused.
            ServerFrame::Error { error, .. }
                if error.code == ErrorCode::Forbidden && self.in_setup_mode() =>
            {
                return true;
            }
            ServerFrame::Event {
                event: Event::SessionCreated { session },
                ..
            } if self.open_greeting => {
                self.open_greeting = false;
                out.push(Effect::OpenGreeting(session.id.clone()));
            }
            // The chat step heard from the owner: read where it stands.
            ServerFrame::Event {
                event: Event::UserMessage { origin, .. },
                ..
            } if origin != "app"
                && self
                    .state
                    .setup
                    .as_ref()
                    .is_some_and(|s| s.step == SetupStep::Channel) =>
            {
                self.send(Command::GetSetup);
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::System { system },
            } if mine(&self.system_request, request_id) => {
                self.system_request = None;
                self.system = SystemRead::Ready {
                    status: Box::new(system.clone()),
                    at_ms: cx.now_ms,
                };
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.system_request, request_id) => {
                self.system_request = None;
                self.system = SystemRead::Failed(error.message.clone());
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Models { options },
            } if mine(&self.models_request, request_id) => {
                self.models_request = None;
                self.models = Read::Ready(options.clone());
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.models_request, request_id) => {
                self.models_request = None;
                self.models = Read::Failed(error.message.clone());
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::SessionModel { session_model },
            } if self.session_model_requests.contains_key(request_id) => {
                self.session_model_requests.remove(request_id);
                self.session_models.insert(
                    session_model.session_id.clone(),
                    Read::Ready(session_model.clone()),
                );
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self.session_model_requests.contains_key(request_id) => {
                let session = self
                    .session_model_requests
                    .remove(request_id)
                    .unwrap_or_default();
                // A refused change leaves the conversation as the daemon
                // has it: read that again.
                out.push(Effect::Say(error.message.clone()));
                self.session_models.remove(&session);
                self.load_session_model(&session);
                return true;
            }
            // Only this client's own list or change is answered with it.
            ServerFrame::Ack {
                result:
                    CommandResult::Automations {
                        automations, note, ..
                    },
                ..
            } => {
                self.automations_request = None;
                self.automation_busy = None;
                self.automations = Read::Ready(automations.clone());
                if let Some(note) = note {
                    out.push(Effect::Say(note.clone()));
                }
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.automations_request, request_id) => {
                self.automations_request = None;
                if self.automation_busy.take().is_some() {
                    out.push(Effect::Say(error.message.clone()));
                } else {
                    self.automations = Read::Failed(error.message.clone());
                }
                return true;
            }
            ServerFrame::Ack {
                result: CommandResult::Commands { commands },
                ..
            } => {
                self.commands = commands.clone();
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::SearchResults { hits },
            } if mine(&self.search_request, request_id) => {
                self.search_request = None;
                if let SearchRead::Loading(query) = &self.search {
                    self.search = SearchRead::Ready {
                        query: query.clone(),
                        hits: hits.clone(),
                    };
                }
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.search_request, request_id) => {
                self.search_request = None;
                self.search = SearchRead::Failed(error.message.clone());
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Project { project },
            } if mine(&self.project_request, request_id) => {
                self.project_request = None;
                self.clear_project_error();
                out.push(Effect::ProjectSaved(project.clone()));
                // The reducer keeps the project too.
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.project_request, request_id) => {
                self.project_request = None;
                self.project_error = Some(error.message.clone());
                self.project_missing_folder = (error.code == ErrorCode::FolderMissing)
                    .then(|| self.project_path.clone())
                    .flatten();
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Projects { .. },
            } if self
                .delete_project_request
                .as_ref()
                .is_some_and(|(id, _)| id == request_id) =>
            {
                let (_, name) = self.delete_project_request.take().unwrap_or_default();
                out.push(Effect::Say(format!("Deleted {name}")));
                // The reducer keeps the list as it is now.
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .delete_project_request
                .as_ref()
                .is_some_and(|(id, _)| id == request_id) =>
            {
                self.delete_project_request = None;
                out.push(Effect::Say(error.message.clone()));
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Changes { changes },
            } if self
                .revert_request
                .as_ref()
                .is_some_and(|(id, _)| id == request_id) =>
            {
                let (_, path) = self.revert_request.take().unwrap_or_default();
                self.changes = Read::Ready(changes.clone());
                out.push(Effect::Say(format!("Undid the changes to {path}")));
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .revert_request
                .as_ref()
                .is_some_and(|(id, _)| id == request_id) =>
            {
                self.revert_request = None;
                out.push(Effect::Say(error.message.clone()));
                return true;
            }
            ServerFrame::Ack {
                result:
                    CommandResult::Queued {
                        session_id,
                        messages,
                    },
                ..
            } => {
                self.queued.insert(session_id.clone(), messages.clone());
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::MessageAccepted { session_id, .. },
            } if self.queue_sends.remove(request_id) => {
                self.refresh_queued(session_id);
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::ProjectDir { dir },
            } if mine(&self.files_dir_request, request_id) => {
                self.files_dir_request = None;
                self.files_dir = Read::Ready(dir.clone());
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::ProjectFile { file },
            } if mine(&self.files_file_request, request_id) => {
                self.files_file_request = None;
                self.files_file = Read::Ready(file.clone());
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.files_dir_request, request_id) => {
                self.files_dir_request = None;
                self.files_dir = Read::Failed(error.message.clone());
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.files_file_request, request_id) => {
                self.files_file_request = None;
                self.files_file = Read::Failed(error.message.clone());
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Changes { changes },
            } if mine(&self.changes_request, request_id) => {
                self.changes_request = None;
                // Changes found: the conversation is a coding one, and one
                // just found to be shows them.
                if changes.repos.iter().any(|r| !r.files.is_empty())
                    && let Some(session) = self.changes_of.clone()
                    && !self.is_coding(&session)
                {
                    self.mark_coded(&session, out);
                    out.push(Effect::OpenChanges(session));
                }
                self.changes = Read::Ready(changes.clone());
                if std::mem::take(&mut self.changes_again) {
                    self.refresh_changes(cx.selected.as_deref());
                }
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.changes_request, request_id) => {
                self.changes_request = None;
                self.changes_again = false;
                self.changes = Read::Failed(error.message.clone());
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Memories { memories },
            } if mine(&self.memories_request, request_id) => {
                self.memories_request = None;
                self.memories = Read::Ready(memories.clone());
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.memories_request, request_id) => {
                self.memories_request = None;
                self.memories = Read::Failed(error.message.clone());
                return true;
            }
            ServerFrame::Ack {
                result: CommandResult::Forgotten { memory_id },
                ..
            } => {
                if let Read::Ready(items) = &mut self.memories {
                    items.retain(|item| item.id != *memory_id);
                }
                out.push(Effect::Say("Forgotten".into()));
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Settings { settings },
            } if mine(&self.settings_request, request_id) => {
                self.settings_request = None;
                self.settings = Read::Ready(Box::new(settings.clone()));
                self.settings_read_again();
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if mine(&self.settings_request, request_id) => {
                self.settings_request = None;
                self.settings = Read::Failed(error.message.clone());
                self.settings_read_again();
                return true;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::SettingsChanged { outcome },
            } => {
                self.settings_changes.remove(request_id);
                // `Done`: the daemon announces what it did to every
                // client (`settings_updated`), which says it here too.
                if let SettingsOutcome::AwaitingApproval {
                    approval_id,
                    other_device,
                } = outcome
                {
                    self.settings_waiting
                        .insert(approval_id.clone(), *other_device);
                    out.push(Effect::Say(
                        if *other_device {
                            "Approve it on another device, or on the daemon's computer"
                        } else {
                            "Waiting for your approval"
                        }
                        .into(),
                    ));
                }
                return true;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self.settings_changes.contains(request_id) => {
                self.settings_changes.remove(request_id);
                self.settings_error = Some(error.message.clone());
                // A refusal may be for a page out of date: read it again.
                if self.settings.is_ready() || cx.settings_open {
                    self.refresh_settings();
                }
                return true;
            }
            ServerFrame::Event {
                event: Event::SettingsUpdated { message, .. },
                ..
            } => {
                out.push(Effect::Say(message.clone()));
                if self.settings.is_ready() || cx.settings_open {
                    self.refresh_settings();
                }
                return true;
            }
            ServerFrame::Event {
                event: Event::ApprovalResolved { approval_id, .. },
                ..
            } if self.settings_waiting.contains_key(approval_id) => {
                self.settings_waiting.remove(approval_id);
            }
            ServerFrame::Event {
                event, session_id, ..
            } => {
                // A turn starting or ending takes waiting messages in, and so
                // does a running turn that reads one (it arrives as the
                // owner's message, mid-turn).
                if let (
                    Some(session),
                    Event::TurnStarted { .. }
                    | Event::TurnCompleted { .. }
                    | Event::UserMessage { .. },
                ) = (session_id.as_deref(), event)
                    && self.queued.get(session).is_some_and(|q| !q.is_empty())
                {
                    self.refresh_queued(session);
                }
                self.note_activity(event, cx.now_ms);
                self.note_code(event, cx, out);
                self.note_reply(session_id.as_deref(), event, cx);
            }
            // After a reload or a dropped connection: what running turns
            // have got to, so the turn's activity is not empty.
            ServerFrame::Snapshot { state, .. }
            | ServerFrame::Ack {
                result: CommandResult::Snapshot { state, .. },
                ..
            } => self.note_live_turns(&state.live_turns, cx.now_ms),
            _ => {}
        }
        false
    }

    /// A reply landed: unread unless the owner is looking at it; a stop
    /// asked for is over once its turn ends.
    fn note_reply(&mut self, session: Option<&str>, event: &Event, cx: &Context) {
        let (session, done) = match event {
            Event::AssistantMessage { session_id, .. } => (session_id.as_str(), false),
            Event::TurnCompleted { .. } => match session {
                Some(session) => (session, true),
                None => return,
            },
            _ => return,
        };
        let looking = cx.looking && cx.selected.as_deref() == Some(session);
        if !looking {
            self.unread.insert(session.to_owned());
        }
        if done {
            self.stopping.remove(session);
            self.finished = Some((session.to_owned(), cx.now_ms));
        }
    }

    // ----- Setup ------------------------------------------------------------

    /// The daemon is in setup mode: only the model step is open.
    pub fn in_setup_mode(&self) -> bool {
        self.state
            .setup
            .as_ref()
            .is_some_and(|s| matches!(s.step, SetupStep::Model | SetupStep::Starting))
    }

    pub fn setup_step(&mut self, action: SetupAction) {
        // First run ends on the greeting.
        if matches!(action, SetupAction::SkipChannel | SetupAction::Finish) {
            self.open_greeting = true;
        }
        self.setup_step_later(action);
    }

    /// A setup step taken after first run (the chat step, from Settings):
    /// no greeting follows, so the next conversation is not opened.
    pub fn setup_step_later(&mut self, action: SetupAction) {
        self.setup_error = None;
        let id = self.send(Command::Setup { action });
        self.setup_request = Some(id);
    }

    pub fn setup_busy(&self) -> bool {
        self.setup_request.is_some()
    }

    // ----- Reads the pages ask for -----------------------------------------

    pub fn refresh_system(&mut self) {
        if self.system_request.is_some() {
            return;
        }
        if !matches!(self.system, SystemRead::Ready { .. }) {
            self.system = SystemRead::Loading;
        }
        self.system_request = Some(self.send(Command::GetSystem));
    }

    /// Tokens used on `today` (a UTC date, as usage days are), when the
    /// System read has them.
    pub fn tokens_on(&self, today: &str) -> Option<u64> {
        let SystemRead::Ready { status, .. } = &self.system else {
            return None;
        };
        Some(
            status
                .usage
                .iter()
                .filter(|d| d.day == today)
                .map(|d| d.input_tokens + d.output_tokens)
                .sum(),
        )
    }

    pub fn refresh_automations(&mut self) {
        if self.automations_request.is_some() {
            return;
        }
        if !self.automations.is_ready() {
            self.automations = Read::Loading;
        }
        self.automations_request = Some(self.send(Command::ListAutomations));
    }

    /// Pause, resume, cancel or run one automation now; the answer brings
    /// the list as it is after, and a note of what happened.
    pub fn change_automation(&mut self, id: &str, change: AutomationChange) {
        if self.automations_request.is_some() {
            return;
        }
        self.automation_busy = Some(id.to_owned());
        self.automations_request = Some(self.send(Command::ChangeAutomation {
            automation_id: id.to_owned(),
            change,
        }));
    }

    /// Rename, reschedule or re-instruct one automation; a `None` field
    /// stays as it is. `false` when another change is still in flight.
    pub fn edit_automation(
        &mut self,
        id: &str,
        name: Option<String>,
        when: Option<String>,
        instructions: Option<String>,
        notify: Option<AutomationNotify>,
    ) -> bool {
        if self.automations_request.is_some() {
            return false;
        }
        self.automation_busy = Some(id.to_owned());
        self.automations_request = Some(self.send(Command::EditAutomation {
            automation_id: id.to_owned(),
            name,
            when,
            instructions,
            notify,
        }));
        true
    }

    /// Runs that finished after `seen_ms` (when the owner last looked at
    /// the automations) with something to read: a report, or a failure.
    /// A quiet run with nothing to say is not news.
    pub fn automation_news(&self, seen_ms: u64) -> usize {
        self.automations.ready().map_or(0, |all| {
            all.iter()
                .flat_map(|a| a.recent_runs.iter())
                .filter(|run| run_is_news(run, seen_ms))
                .count()
        })
    }

    pub fn refresh_memories(&mut self) {
        if self.memories_request.is_some() {
            return;
        }
        if !self.memories.is_ready() {
            self.memories = Read::Loading;
        }
        self.memories_request = Some(self.send(Command::ListMemories));
    }

    pub fn forget_memory(&mut self, id: &str) {
        self.send(Command::ForgetMemory {
            memory_id: id.to_owned(),
        });
    }

    pub fn refresh_settings(&mut self) {
        if self.settings_request.is_some() {
            self.settings_again = true;
            return;
        }
        if !self.settings.is_ready() {
            self.settings = Read::Loading;
        }
        self.settings_request = Some(self.send(Command::GetSettings));
    }

    fn settings_read_again(&mut self) {
        if std::mem::take(&mut self.settings_again) {
            self.refresh_settings();
        }
    }

    /// Ask the daemon to make `change`; the answer arrives as a note, and
    /// the settings read again when it lands.
    pub fn change_settings(&mut self, change: SettingsChange) {
        self.settings_error = None;
        let id = self.send(Command::ChangeSettings { change });
        self.settings_changes.insert(id);
    }

    /// Set coding agent `agent`'s permission mode. Picking the mode it
    /// already has, or one it does not offer, sends nothing; full access
    /// is asked first by the daemon (`SettingsOutcome::AwaitingApproval`).
    pub fn set_agent_mode(&mut self, agent: &str, mode: AgentMode) {
        if let Read::Ready(settings) = &self.settings
            && let Some(known) = settings.coding_agents.iter().find(|a| a.name == agent)
            && (known.mode == Some(mode) || !known.modes.contains(&mode))
        {
            return;
        }
        self.change_settings(SettingsChange::SetAgentMode {
            agent: agent.to_owned(),
            mode,
        });
    }

    /// Ask every coding agent again whether it can do work now. The
    /// daemon answers like any settings change, and the settings it
    /// announces carry each agent's new readiness.
    pub fn check_agents(&mut self) {
        self.change_settings(SettingsChange::CheckAgents);
    }

    /// A settings change was sent and has no answer yet.
    pub fn settings_busy(&self) -> bool {
        !self.settings_changes.is_empty()
    }

    /// Ask for messages matching `query`, in every conversation.
    pub fn search_messages(&mut self, query: &str) {
        let query = query.trim().to_owned();
        if query.chars().count() < 2 {
            self.search_request = None;
            self.search = SearchRead::Idle;
            return;
        }
        self.search = SearchRead::Loading(query.clone());
        self.search_request = Some(self.send(Command::SearchMessages { query, limit: 30 }));
    }

    // ----- A project's files ----------------------------------------------------

    /// Read folder `path` (relative, `""` for the folder itself) of project
    /// `project_id`. What was shown stays until the answer, unless it was
    /// another project's.
    pub fn list_project_files(&mut self, project_id: &str, path: &str, ignored: bool) {
        let command = Command::ListProjectFiles {
            project_id: project_id.to_owned(),
            path: (!path.is_empty()).then(|| path.to_owned()),
            ignored,
        };
        if self.files_dir_request.is_some() && self.files_dir_asked.as_ref() == Some(&command) {
            return;
        }
        if self
            .files_dir
            .ready()
            .is_none_or(|dir| dir.project_id != project_id)
        {
            self.files_dir = Read::Loading;
        }
        self.files_dir_asked = Some(command.clone());
        self.files_dir_request = Some(self.send(command));
    }

    /// Read file `path` of project `project_id`.
    pub fn read_project_file(&mut self, project_id: &str, path: &str) {
        let command = Command::ReadProjectFile {
            project_id: project_id.to_owned(),
            path: path.to_owned(),
        };
        if self.files_file_request.is_some() && self.files_file_asked.as_ref() == Some(&command) {
            return;
        }
        self.files_file = Read::Loading;
        self.files_file_asked = Some(command.clone());
        self.files_file_request = Some(self.send(command));
    }

    /// Close the open file; an answer still on its way is dropped.
    pub fn close_project_file(&mut self) {
        self.files_file_request = None;
        self.files_file = Read::NotYet;
    }

    // ----- Changes ------------------------------------------------------------

    /// Read `selected`'s changes again; one read at a time, and one more
    /// after it when asked meanwhile.
    pub fn refresh_changes(&mut self, selected: Option<&str>) {
        let Some(session_id) = selected else {
            self.changes_of = None;
            self.changes = Read::NotYet;
            return;
        };
        if self.changes_request.is_some() {
            self.changes_again = true;
            return;
        }
        if self.changes_of.as_deref() != Some(session_id) || !self.changes.is_ready() {
            self.changes = Read::Loading;
        }
        self.changes_of = Some(session_id.to_owned());
        self.changes_request = Some(self.send(Command::GetChanges {
            session_id: session_id.to_owned(),
        }));
    }

    /// Read `selected`'s changes unless those held are its own and current.
    pub fn refresh_changes_if_stale(&mut self, selected: Option<&str>) {
        if self.changes_of.as_deref() != selected {
            self.refresh_changes(selected);
        }
    }

    /// How many files `selected` has changed, once read.
    pub fn changed_files(&self, selected: &str) -> Option<usize> {
        if self.changes_of.as_deref() != Some(selected) {
            return None;
        }
        self.changes
            .ready()
            .map(|changes| changes.repos.iter().map(|r| r.files.len()).sum())
    }

    /// Whether conversation `id` works on code: it is in a project with a
    /// folder, or it wrote files or showed changes.
    pub fn is_coding(&self, id: &str) -> bool {
        self.coded.contains(id)
            || self
                .session_project(id)
                .and_then(|p| self.state.projects.get(&p))
                .is_some_and(|p| p.path.is_some())
    }

    fn mark_coded(&mut self, session: &str, out: &mut Vec<Effect>) {
        if self.coded.insert(session.to_owned()) {
            out.push(Effect::NowCoding(session.to_owned()));
        }
    }

    /// A tool that can change files finished in the selected session: the
    /// changes read again, and the turn's first file write shows them.
    fn note_code(&mut self, event: &Event, cx: &Context, out: &mut Vec<Effect>) {
        let (turn_id, name) = match event {
            Event::TurnStatus {
                turn_id,
                status: TurnStatus::ToolCompleted { name, .. },
            } => (turn_id, Some(name.as_str())),
            Event::TurnCompleted { turn_id, .. } => (turn_id, None),
            _ => return,
        };
        let session = self.turns.get(turn_id).map(|log| log.session_id.clone());
        if session.is_none() || session != cx.selected {
            return;
        }
        if let Some(name) = name {
            if !CHANGING_TOOLS.contains(&name) {
                return;
            }
            if WRITING_TOOLS.contains(&name) && self.changes_shown.insert(turn_id.clone()) {
                if let Some(session) = &session {
                    self.mark_coded(session, out);
                    out.push(Effect::OpenChanges(session.clone()));
                }
                // Showing them reads them.
                self.refresh_changes(cx.selected.as_deref());
                return;
            }
        }
        if cx.changes_visible {
            self.refresh_changes(cx.selected.as_deref());
        } else {
            // Read on next showing.
            self.changes_of = None;
        }
    }

    // ----- Turns' activity ----------------------------------------------------

    /// The latest turn of `session` this client saw.
    pub fn latest_turn(&self, session: &str) -> Option<&TurnLog> {
        self.turns
            .values()
            .filter(|log| log.session_id == session)
            .max_by_key(|log| log.started_ms)
    }

    /// Activity for running turns this client did not see start: their
    /// tools in order, then what they are doing now. Turns it already
    /// follows keep their own record.
    fn note_live_turns(&mut self, live: &[LiveTurnState], now: u64) {
        for turn in live {
            if self.turns.contains_key(&turn.turn_id) {
                continue;
            }
            self.note_activity(
                &Event::TurnStarted {
                    turn_id: turn.turn_id.clone(),
                    session_id: turn.session_id.clone(),
                    origin: String::new(),
                },
                now,
            );
            let last = turn.tools.len().saturating_sub(1);
            for (n, name) in turn.tools.iter().enumerate() {
                // The current status names the last tool with its summary.
                let (summary, activity, input) = match &turn.status {
                    Some(TurnStatus::ToolStarted {
                        name: now,
                        summary,
                        activity,
                        input,
                    }) if n == last && now == name => (summary.clone(), *activity, input.clone()),
                    _ => (String::new(), None, String::new()),
                };
                self.note_activity(
                    &Event::TurnStatus {
                        turn_id: turn.turn_id.clone(),
                        status: TurnStatus::ToolStarted {
                            name: name.clone(),
                            summary,
                            activity,
                            input,
                        },
                    },
                    now,
                );
            }
            if let Some(status) = &turn.status
                && !matches!(status, TurnStatus::ToolStarted { .. })
            {
                self.note_activity(
                    &Event::TurnStatus {
                        turn_id: turn.turn_id.clone(),
                        status: status.clone(),
                    },
                    now,
                );
            }
        }
    }

    fn note_activity(&mut self, event: &Event, now: u64) {
        let (turn_id, kind, title, detail) = match event {
            Event::TurnStarted {
                turn_id,
                session_id,
                ..
            } => {
                self.turns.insert(
                    turn_id.clone(),
                    TurnLog {
                        session_id: session_id.clone(),
                        started_ms: now,
                        ..TurnLog::default()
                    },
                );
                return;
            }
            Event::TurnCompleted { turn_id, .. } => {
                if let Some(log) = self.turns.get_mut(turn_id) {
                    log.done = true;
                }
                return;
            }
            Event::TurnStatus { turn_id, status } => {
                let (kind, title, detail) = match status {
                    TurnStatus::Thinking => {
                        (ActivityKind::Thinking, "Thinking".into(), String::new())
                    }
                    TurnStatus::ToolStarted { name, summary, .. } => {
                        (ActivityKind::Tool, name.clone(), summary.clone())
                    }
                    TurnStatus::ToolProgress { name, chunk } => {
                        (ActivityKind::Output, name.clone(), chunk.clone())
                    }
                    TurnStatus::ToolCompleted { name, summary, .. } => {
                        (ActivityKind::ToolDone, name.clone(), summary.clone())
                    }
                    TurnStatus::PlanStep {
                        step,
                        total,
                        description,
                        ..
                    } => (
                        ActivityKind::Plan,
                        format!("Step {} of {total}", step + 1),
                        description.clone(),
                    ),
                    TurnStatus::Checklist { text } => {
                        (ActivityKind::Plan, "Plan".into(), text.clone())
                    }
                    TurnStatus::Note { text } => (ActivityKind::Note, "Note".into(), text.clone()),
                    TurnStatus::Unknown => return,
                };
                (turn_id, kind, title, detail)
            }
            _ => return,
        };
        self.next_activity += 1;
        let id = self.next_activity;
        let Some(log) = self.turns.get_mut(turn_id) else {
            return;
        };
        // Progress of one tool folds into its last output row.
        if kind == ActivityKind::Output
            && let Some(last) = log.items.last_mut()
            && last.kind == ActivityKind::Output
            && last.title == title
        {
            last.detail.push_str(&detail);
            return;
        }
        log.items.push(Activity {
            id,
            kind,
            title,
            detail,
            at_ms: now,
        });
    }

    // ----- Projects and bots ------------------------------------------------

    // ----- A conversation's model ------------------------------------------

    /// Read the providers and models a conversation can answer with.
    pub fn refresh_models(&mut self) {
        if self.models_request.is_some() {
            return;
        }
        if !self.models.is_ready() {
            self.models = Read::Loading;
        }
        self.models_request = Some(self.send(Command::ListModels));
    }

    /// Whether `session` (`None`: a new conversation) can be set to a
    /// model: only the app's own conversations can.
    pub fn can_pick_model(&self, session: Option<&str>) -> bool {
        session.is_none_or(|id| id.starts_with("app:"))
    }

    /// Read what `session` answers with, once.
    pub fn load_session_model(&mut self, session: &str) {
        if !self.can_pick_model(Some(session)) || self.session_models.contains_key(session) {
            return;
        }
        self.session_models
            .insert(session.to_owned(), Read::Loading);
        let id = self.send(Command::GetSessionModel {
            session_id: session.to_owned(),
        });
        self.session_model_requests.insert(id, session.to_owned());
    }

    /// The model and effort `session` (`None`: a new conversation) answers
    /// with, as far as this client knows.
    pub fn model_of(&self, session: Option<&str>) -> (ModelChoice, Effort) {
        match session {
            None => self.new_model.clone(),
            Some(id) => self
                .session_models
                .get(id)
                .and_then(Read::ready)
                .map(|m| (m.model.clone(), m.effort))
                .unwrap_or_default(),
        }
    }

    /// The efforts `model` takes besides automatic: a pinned model's own;
    /// automatic, those of the main provider's first model.
    pub fn efforts_of(&self, model: &ModelChoice) -> Vec<Effort> {
        let Some(options) = self.models.ready() else {
            return Vec::new();
        };
        match model {
            ModelChoice::Pinned { provider, model } => options
                .providers
                .iter()
                .find(|p| p.id == *provider)
                .and_then(|p| p.models.iter().find(|m| m.id == *model))
                .map(|m| m.efforts.clone())
                .unwrap_or_default(),
            _ => options
                .providers
                .first()
                .and_then(|p| p.models.first())
                .map(|m| m.efforts.clone())
                .unwrap_or_default(),
        }
    }

    /// Set what `session` (`None`: the next new conversation) answers
    /// with; a `None` stays as it is. An effort the new model does not take
    /// goes back to automatic.
    pub fn set_session_model(
        &mut self,
        session: Option<&str>,
        model: Option<ModelChoice>,
        effort: Option<Effort>,
    ) {
        if !self.can_pick_model(session) {
            return;
        }
        let (mut next_model, mut next_effort) = self.model_of(session);
        if let Some(model) = model.clone() {
            next_model = model;
        }
        if let Some(effort) = effort {
            next_effort = effort;
        }
        if next_effort != Effort::Auto && !self.efforts_of(&next_model).contains(&next_effort) {
            next_effort = Effort::Auto;
        }
        let Some(session) = session else {
            self.new_model = (next_model, next_effort);
            return;
        };
        self.session_models.insert(
            session.to_owned(),
            Read::Ready(SessionModel {
                session_id: session.to_owned(),
                model: next_model,
                effort: next_effort,
            }),
        );
        let id = self.send(Command::SetSessionModel {
            session_id: session.to_owned(),
            model,
            effort: Some(next_effort),
        });
        self.session_model_requests.insert(id, session.to_owned());
    }

    /// The owner's projects, by name.
    pub fn projects(&self) -> Vec<ProjectSummary> {
        let mut projects: Vec<ProjectSummary> = self.state.projects.values().cloned().collect();
        projects.sort_by_key(|p| p.name.to_lowercase());
        projects
    }

    /// The project a conversation is in.
    pub fn session_project(&self, session: &str) -> Option<String> {
        self.state
            .sessions
            .get(session)
            .and_then(|v| v.summary.as_ref())
            .and_then(|summary| summary.project_id.clone())
    }

    /// The project whose folder holds `folder` (the innermost).
    pub fn project_of_folder(&self, folder: &str) -> Option<String> {
        let here = folder.trim_end_matches('/');
        self.state
            .projects
            .values()
            .filter_map(|p| {
                let path = p.path.as_deref()?.trim_end_matches('/');
                (here == path || here.starts_with(&format!("{path}/")))
                    .then_some((path.len(), p.id.clone()))
            })
            .max_by_key(|(len, _)| *len)
            .map(|(_, id)| id)
    }

    /// Make a project (`project_id` `None`) or save one's name and folder
    /// (an empty folder removes it); with `create_folder`, the daemon makes
    /// a missing folder. `false` when it has no name.
    pub fn save_project(
        &mut self,
        project_id: Option<String>,
        name: &str,
        path: &str,
        create_folder: bool,
    ) -> bool {
        let name = name.trim();
        if name.is_empty() {
            self.clear_project_error();
            self.project_error = Some("Give the project a name".to_owned());
            return false;
        }
        let path = path.trim();
        self.project_path = Some(path.to_owned());
        let command = match project_id {
            Some(project_id) => Command::UpdateProject {
                project_id,
                name: Some(name.to_owned()),
                path: Some(path.to_owned()),
                create_folder,
            },
            None => Command::CreateProject {
                name: name.to_owned(),
                path: (!path.is_empty()).then(|| path.to_owned()),
                create_folder,
            },
        };
        self.project_request = Some(self.send(command));
        true
    }

    /// Forget the project form's refusal.
    pub fn clear_project_error(&mut self) {
        self.project_error = None;
        self.project_missing_folder = None;
    }

    pub fn saving_project(&self) -> bool {
        self.project_request.is_some()
    }

    /// Delete a project: its conversations go to no project and its folder
    /// stays. The daemon's answer says it, or why not.
    pub fn delete_project(&mut self, project_id: &str) {
        let name = self
            .state
            .projects
            .get(project_id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "the project".to_owned());
        let id = self.send(Command::DeleteProject {
            project_id: project_id.to_owned(),
        });
        self.delete_project_request = Some((id, name));
    }

    /// Put a conversation in a project, or in none.
    pub fn move_to_project(&mut self, session: &str, project: Option<String>) {
        self.send(Command::SetSessionProject {
            session_id: session.to_owned(),
            project_id: project,
        });
    }

    /// The chat bots with conversations here, by channel.
    pub fn bots(&self) -> Vec<Bot> {
        let projects = self.projects();
        let mut bots: BTreeMap<String, usize> = BTreeMap::new();
        for view in self.state.sessions.values() {
            if let Some(summary) = &view.summary
                && !matches!(
                    platform_of(&summary.channel, &summary.id),
                    Source::App | Source::Chat
                )
            {
                *bots.entry(summary.channel.clone()).or_default() += 1;
            }
        }
        bots.into_iter()
            .map(|(channel, chats)| Bot {
                name: channel_name(&channel, ""),
                project: projects
                    .iter()
                    .find(|p| p.channels.contains(&channel))
                    .map(|p| p.id.clone()),
                channel,
                chats,
            })
            .collect()
    }

    /// Send a bot's direct-message chats to `project` (`None`: to none).
    pub fn set_bot_project(&mut self, channel: &str, project: Option<String>) {
        self.send(Command::SetChannelProject {
            channel: channel.to_owned(),
            project_id: project,
        });
    }

    // ----- Conversations ------------------------------------------------------

    /// Conversations, the most recently active first; ones never active
    /// keep the gateway's order after them.
    pub fn session_ids(&self) -> Vec<String> {
        let mut ids = self.state.order.clone();
        ids.sort_by_key(|id| std::cmp::Reverse(self.last_activity(id)));
        ids
    }

    /// When a conversation last had a message.
    pub fn last_activity(&self, id: &str) -> Option<u64> {
        let view = self.state.sessions.get(id)?;
        let line = view.lines.iter().rev().find_map(|line| match line {
            Line::User { at_ms, .. } | Line::Assistant { at_ms, .. } if *at_ms > 0 => Some(*at_ms),
            _ => None,
        });
        let summary = view.summary.as_ref().and_then(|m| m.last_activity_ms);
        line.max(summary)
    }

    /// A conversation's name: the daemon's title, else its first message,
    /// else where it lives ("Slack conversation").
    pub fn title(&self, id: &str) -> String {
        let view = self.state.sessions.get(id);
        if let Some(title) = view
            .and_then(|v| v.summary.as_ref())
            .and_then(|m| m.title.clone())
        {
            return title;
        }
        let first = view.and_then(|v| {
            v.lines.iter().find_map(|line| match line {
                Line::User { text, .. } => {
                    Some(text.lines().next().unwrap_or_default().trim().to_owned())
                }
                _ => None,
            })
        });
        match first.filter(|t| !t.is_empty()) {
            Some(text) if text.chars().count() > 60 => {
                format!("{}…", text.chars().take(60).collect::<String>().trim_end())
            }
            Some(text) => text,
            None => format!("{} conversation", platform_name(id)),
        }
    }

    pub fn channel(&self, id: &str) -> Option<String> {
        self.state
            .sessions
            .get(id)
            .and_then(|v| v.summary.as_ref())
            .map(|summary| summary.channel.clone())
    }

    /// Other people are in conversation `id`.
    pub fn is_group(&self, id: &str) -> bool {
        self.state
            .sessions
            .get(id)
            .and_then(|v| v.summary.as_ref())
            .is_some_and(|summary| summary.group)
    }

    /// Where a conversation lives: platform, bot, and whether it is a group.
    pub fn place(&self, id: &str) -> String {
        let place = channel_name(&self.channel(id).unwrap_or_default(), id);
        if self.is_group(id) {
            format!("{place} · group")
        } else {
            place
        }
    }

    /// Where conversation `id` came from.
    pub fn source(&self, id: &str) -> Source {
        platform_of(&self.channel(id).unwrap_or_default(), id)
    }

    /// The source filters worth offering: every source with a
    /// conversation, and groups when there are any; none when all come
    /// from one place.
    pub fn sources(&self) -> Vec<Source> {
        let ids = &self.state.order;
        let mut found: BTreeSet<Source> = ids.iter().map(|id| self.source(id)).collect();
        if found.len() < 2 {
            return Vec::new();
        }
        if ids.iter().any(|id| self.is_group(id)) {
            found.insert(Source::Groups);
        }
        std::iter::once(Source::All).chain(found).collect()
    }

    /// Whether conversation `id` passes the source filter `source`.
    pub fn in_source(&self, id: &str, source: Source) -> bool {
        match source {
            Source::All => true,
            Source::Groups => self.is_group(id),
            source => self.source(id) == source,
        }
    }

    /// Whether conversation `id` matches every word of `query` (its title,
    /// place, id or any loaded message).
    pub fn matches(&self, id: &str, query: &str) -> bool {
        let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
        if words.is_empty() {
            return true;
        }
        let mut hay = format!("{} {} {id}", self.title(id), self.place(id)).to_lowercase();
        if let Some(view) = self.state.sessions.get(id) {
            for line in &view.lines {
                if let Line::User { text, .. } | Line::Assistant { text, .. } = line {
                    hay.push(' ');
                    hay.push_str(&text.to_lowercase());
                }
            }
        }
        words.iter().all(|w| hay.contains(w.as_str()))
    }

    pub fn lines(&self, id: &str) -> &[Line] {
        self.state
            .sessions
            .get(id)
            .map(|v| v.lines.as_slice())
            .unwrap_or_default()
    }

    /// The last thing the owner said in `session`.
    pub fn last_user_text(&self, session: &str) -> Option<String> {
        self.lines(session)
            .iter()
            .rev()
            .find_map(|line| match line {
                Line::User { text, .. } => Some(text.clone()),
                _ => None,
            })
    }

    /// The last message in `session`, and whether the owner sent it.
    pub fn last_message(&self, session: &str) -> Option<(bool, String)> {
        self.lines(session)
            .iter()
            .rev()
            .find_map(|line| match line {
                Line::User { text, .. } => Some((true, text.clone())),
                Line::Assistant { text, .. } => Some((false, text.clone())),
                _ => None,
            })
    }

    /// The last reply in `session`.
    pub fn last_reply(&self, session: &str) -> Option<String> {
        self.lines(session)
            .iter()
            .rev()
            .find_map(|line| match line {
                Line::Assistant { text, .. } => Some(text.clone()),
                _ => None,
            })
    }

    pub fn is_running(&self, id: &str) -> bool {
        self.state
            .sessions
            .get(id)
            .is_some_and(|v| v.live.is_some())
            || self.state.tasks.values().any(|t| t.session_id == id)
    }

    /// Its last turn handed work to the background and the report is
    /// still to come.
    pub fn in_background(&self, id: &str) -> bool {
        self.state.sessions.get(id).is_some_and(|v| v.background)
    }

    pub fn is_stopping(&self, id: &str) -> bool {
        self.stopping.contains(id) && self.is_running(id)
    }

    /// Whether the settings read failed because the daemon predates
    /// `get_settings` (protocol 1.4): read from the version it speaks,
    /// never from the error's words.
    pub fn settings_unsupported(&self) -> bool {
        matches!(self.settings, Read::Failed(_))
            && self
                .daemon_version
                .is_some_and(|v| v.major == 1 && v.minor < GET_SETTINGS_MINOR)
    }

    /// Whether the last turn of `session` failed or was stopped.
    pub fn can_retry(&self, session: &str) -> bool {
        !self.is_running(session)
            && matches!(
                self.lines(session).last(),
                Some(Line::Note {
                    ended: Some(TurnState::Cancelled | TurnState::Failed),
                    ..
                })
            )
    }

    /// Conversations with an approval waiting.
    pub fn needs_you(&self, id: &str) -> bool {
        self.state.approvals.values().any(|a| a.session_id == id)
    }

    /// The approvals waiting, oldest first.
    pub fn approvals(&self) -> Vec<Approval> {
        let mut all: Vec<Approval> = self.state.approvals.values().cloned().collect();
        all.sort_by_key(|a| a.requested_at_ms);
        all
    }

    pub fn running(&self) -> Vec<TaskSummary> {
        self.state.running().into_iter().cloned().collect()
    }

    /// Slash commands completing `draft`: it starts with `/` and has no
    /// space yet. At most eight, in the daemon's order.
    pub fn slash_matches(&self, draft: &str) -> Vec<SlashCommand> {
        let Some(typed) = draft.strip_prefix('/') else {
            return Vec::new();
        };
        if typed.contains(char::is_whitespace) {
            return Vec::new();
        }
        let typed = typed.to_lowercase();
        self.commands
            .iter()
            .filter(|c| c.name.starts_with(&typed) && c.name != typed)
            .take(8)
            .cloned()
            .collect()
    }

    /// Ask for `id`'s history the first time it is shown (what arrived
    /// live before then merges with it).
    pub fn load_history(&mut self, id: &str) {
        if !self.fetched.contains(id) {
            self.read_history(id);
        }
    }

    /// Read `id`'s latest history now; it merges with what is shown.
    fn read_history(&mut self, id: &str) -> String {
        self.fetched.insert(id.to_owned());
        self.send(Command::GetHistory {
            session_id: id.to_owned(),
            before: None,
            limit: 50,
        })
    }

    // ----- Refresh ----------------------------------------------------------

    /// Read everything on screen again: the conversations, what is running
    /// and waiting for approval, the open conversation, and each page read
    /// before. For a pull to refresh, and the app's return after a while
    /// away. `false` when offline (nothing to read from).
    pub fn refresh_all(&mut self, cx: &Context) -> bool {
        if !self.is_online() {
            return false;
        }
        let mut reads = vec![
            self.send(Command::GetSnapshot),
            self.send(Command::ListSessions {
                cursor: None,
                limit: 200,
            }),
            self.send(Command::ListProjects),
        ];
        if let Some(id) = &cx.selected {
            reads.push(self.read_history(id));
            if self.queued.contains_key(id.as_str()) {
                self.refresh_queued(id);
            }
            self.refresh_changes(cx.selected.as_deref());
        }
        self.refreshing.extend(reads);
        self.refresh_system();
        self.refresh_automations();
        if self.memories.is_ready() {
            self.refresh_memories();
        }
        if self.models.is_ready() {
            self.refresh_models();
        }
        if self.devices.is_ready() {
            self.refresh_devices();
        }
        if self.settings.is_ready() || cx.settings_open {
            self.refresh_settings();
        }
        true
    }

    /// A refresh's reads are still out.
    pub fn is_refreshing(&self) -> bool {
        !self.refreshing.is_empty()
    }

    /// The app went out of sight (backgrounded, hidden, minimized).
    pub fn went_away(&mut self, now_ms: u64) {
        self.away_since.get_or_insert(now_ms);
    }

    /// The app is in sight again: after [`AWAY_REFRESH_MS`] or more away,
    /// read everything again. `true` when it did.
    pub fn came_back(&mut self, cx: &Context) -> bool {
        let Some(since) = self.away_since.take() else {
            return false;
        };
        // A drop while out of sight (a phone suspends a backgrounded app's
        // sockets) is not news: it gets the quiet grace from now, while the
        // link reconnects, instead of reading as a long outage at once.
        if let Connection::Offline {
            since_ms: Some(lost),
            ..
        } = &mut self.connection
        {
            *lost = (*lost).max(cx.now_ms);
        }
        cx.now_ms.saturating_sub(since) >= AWAY_REFRESH_MS && self.refresh_all(cx)
    }

    /// The owner opened `id`: its replies are read.
    pub fn seen(&mut self, id: &str) {
        self.unread.remove(id);
    }

    // ----- Sending --------------------------------------------------------

    /// Send `text` (with the draft's ready files) to `session`, or start a
    /// conversation in `project`. A leading `/` runs a slash command in an
    /// existing session. `Ok(true)` when it went (the draft is spent),
    /// `Ok(false)` when there was nothing to send, and `Err` with what to
    /// tell the owner when it cannot go yet.
    pub fn submit(
        &mut self,
        session: Option<&str>,
        text: &str,
        mid_run: Option<MidRun>,
        project: Option<String>,
    ) -> Result<bool, String> {
        let text = text.trim().to_owned();
        if self
            .draft_files
            .iter()
            .any(|f| matches!(f.state, DraftFileState::Uploading))
        {
            return Err("Wait for the files to finish attaching".into());
        }
        let attachments: Vec<AttachmentRef> = self
            .draft_files
            .iter()
            .filter_map(|f| match &f.state {
                DraftFileState::Ready(attachment) => Some(attachment.clone()),
                _ => None,
            })
            .collect();
        if text.is_empty() && attachments.is_empty() {
            return Ok(false);
        }
        // A command runs in an open conversation. In a new one, sending it
        // would hand the agent a request to act on instead.
        if session.is_none()
            && attachments.is_empty()
            && let Some(name) = text
                .strip_prefix('/')
                .and_then(|rest| rest.split_whitespace().next())
            && self.commands.iter().any(|c| c.name == name.to_lowercase())
        {
            return Err(format!(
                "/{name} runs inside a conversation: open one, or start this one with a message first"
            ));
        }
        if !attachments.is_empty() {
            self.draft_files.clear();
        }
        match (
            session,
            text.strip_prefix('/').filter(|_| attachments.is_empty()),
        ) {
            (Some(session), Some(slash)) => self.run_slash(session, slash),
            _ => {
                // A new conversation starts on the model picked for it; an
                // existing one already answers with its own.
                let (model, effort) = match session {
                    None => {
                        let pick = std::mem::take(&mut self.new_model);
                        self.starting_model = Some(pick.clone());
                        (
                            Some(pick.0).filter(|m| *m != ModelChoice::Auto),
                            Some(pick.1).filter(|e| *e != Effort::Auto),
                        )
                    }
                    Some(_) => (None, None),
                };
                // Sent while the work runs, either way: it waits (to be read
                // or for its own turn) and the list shows it.
                let queueing = mid_run.is_some();
                let id = self.send(Command::SendMessage {
                    session_id: session.map(str::to_owned),
                    text,
                    attachments,
                    mid_run,
                    project_id: session.is_none().then_some(project).flatten(),
                    model,
                    effort,
                });
                if queueing {
                    self.queue_sends.insert(id.clone());
                }
                if session.is_none() {
                    // The draft is spent once the conversation exists.
                    self.starting = Some(id);
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Say `text` in `session`, as if typed (a suggested prompt, Retry).
    pub fn ask(&mut self, session: &str, text: String) {
        self.send(Command::SendMessage {
            session_id: Some(session.to_owned()),
            text,
            attachments: Vec::new(),
            mid_run: None,
            project_id: None,
            model: None,
            effort: None,
        });
    }

    /// Run the slash command `slash` ("model fast", no leading `/`) in
    /// `session`.
    fn run_slash(&mut self, session: &str, slash: &str) {
        let (command, args) = slash.split_once(' ').unwrap_or((slash, ""));
        let id = self.send(Command::RunSlash {
            session_id: session.to_owned(),
            command: command.to_owned(),
            args: args.trim().to_owned(),
        });
        if matches!(command.to_lowercase().as_str(), "model" | "auto") {
            self.model_commands.insert(id, session.to_owned());
        }
    }

    /// Send the last message of `session` again, with its files.
    pub fn retry(&mut self, session: &str) {
        let last = self
            .lines(session)
            .iter()
            .rev()
            .find_map(|line| match line {
                Line::User {
                    text, attachments, ..
                } => Some((text.clone(), attachments.clone())),
                _ => None,
            });
        if let Some((text, attachments)) = last {
            // A turn that failed or was stopped is not running.
            let _ = self.resend(session, &text, attachments);
        }
    }

    /// Say a message of `session` again, with the files it had: a new
    /// message at the end, so nothing said or done since is lost (the
    /// agent's work after the first one already happened). A slash
    /// command runs again. `Err` with what to tell the owner while the
    /// conversation works: the composer places a message then.
    pub fn resend(
        &mut self,
        session: &str,
        text: &str,
        attachments: Vec<AttachmentRef>,
    ) -> Result<(), String> {
        let text = text.trim();
        if text.is_empty() && attachments.is_empty() {
            return Ok(());
        }
        if self.is_running(session) {
            return Err("Wait for this to finish, or stop it, then send it again".into());
        }
        match text.strip_prefix('/').filter(|_| attachments.is_empty()) {
            Some(slash) => self.run_slash(session, slash),
            None => {
                self.send(Command::SendMessage {
                    session_id: Some(session.to_owned()),
                    text: text.to_owned(),
                    attachments,
                    mid_run: None,
                    project_id: None,
                    model: None,
                    effort: None,
                });
            }
        }
        Ok(())
    }

    /// Put the files of a sent message back on the message being written
    /// (the owner edits it before sending it again). They are already
    /// uploaded; one already on the draft is not added twice.
    pub fn attach_sent(&mut self, attachments: Vec<AttachmentRef>) {
        for attachment in attachments {
            let there = self
                .draft_files
                .iter()
                .any(|f| matches!(&f.state, DraftFileState::Ready(a) if a.id == attachment.id));
            if there {
                continue;
            }
            self.next_upload += 1;
            self.draft_files.push(DraftFile {
                number: self.next_upload,
                name: attachment.name.clone(),
                state: DraftFileState::Ready(attachment),
                marked_from: None,
            });
        }
    }

    /// Attach a file to the message being written: it uploads now. `Err`
    /// with what to tell the owner when it is too large.
    pub fn attach_file(&mut self, name: String, mime: String, bytes: &[u8]) -> Result<(), String> {
        self.attach(name, mime, bytes, None)
    }

    /// Attach a picture the owner drew marks over: a copy of sent
    /// attachment `marked_from`. The daemon tells the agent the marks are
    /// the owner's.
    pub fn attach_marked_up(
        &mut self,
        name: String,
        mime: String,
        bytes: &[u8],
        marked_from: String,
    ) -> Result<(), String> {
        self.attach(name, mime, bytes, Some(marked_from))
    }

    fn attach(
        &mut self,
        name: String,
        mime: String,
        bytes: &[u8],
        marked_from: Option<String>,
    ) -> Result<(), String> {
        use base64::Engine as _;
        if bytes.len() > MAX_FILE_BYTES {
            return Err(format!(
                "{name} is too large to attach (up to {} MB)",
                MAX_FILE_BYTES / (1024 * 1024)
            ));
        }
        self.next_upload += 1;
        let number = self.next_upload;
        self.draft_files.push(DraftFile {
            number,
            name: name.clone(),
            state: DraftFileState::Uploading,
            marked_from: marked_from.clone(),
        });
        let id = self.send(Command::UploadFile {
            name,
            mime: (!mime.is_empty()).then_some(mime),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
            marked_from,
        });
        self.uploads.insert(id, number);
        Ok(())
    }

    pub fn remove_draft_file(&mut self, number: u64) {
        self.uploads.retain(|_, n| *n != number);
        self.draft_files.retain(|f| f.number != number);
    }

    pub fn stop(&mut self, session: &str) {
        self.stopping.insert(session.to_owned());
        let id = self.send(Command::StopTurn {
            session_id: session.to_owned(),
        });
        self.stop_sends.insert(id, session.to_owned());
    }

    pub fn cancel(&mut self, task: &str) {
        self.send(Command::CancelTask {
            task_id: task.to_owned(),
        });
    }

    /// Name conversation `id` (`None` or blank: back to the name taken
    /// from what was said).
    pub fn rename_session(&mut self, id: &str, title: Option<String>) {
        self.send(Command::RenameSession {
            session_id: id.to_owned(),
            title: title.map(|t| t.trim().to_owned()).filter(|t| !t.is_empty()),
        });
    }

    /// Archive conversation `id`, or bring it back.
    pub fn archive_session(&mut self, id: &str, archived: bool) {
        self.send(Command::ArchiveSession {
            session_id: id.to_owned(),
            archived,
        });
    }

    /// Whether the owner archived conversation `id`.
    pub fn is_archived(&self, id: &str) -> bool {
        self.state
            .sessions
            .get(id)
            .and_then(|v| v.summary.as_ref())
            .is_some_and(|s| s.archived)
    }

    /// Undo the uncommitted changes to `path` in repository `root` of
    /// `session`'s changes; the changes read back after.
    pub fn revert_file(&mut self, session: &str, root: &str, path: &str) {
        let id = self.send(Command::RevertFile {
            session_id: session.to_owned(),
            root: root.to_owned(),
            path: path.to_owned(),
        });
        self.revert_request = Some((id, path.to_owned()));
    }

    /// A file undo is in flight.
    pub fn reverting(&self) -> bool {
        self.revert_request.is_some()
    }

    /// Read `session`'s waiting messages again.
    pub fn refresh_queued(&mut self, session: &str) {
        self.send(Command::ListQueued {
            session_id: session.to_owned(),
        });
    }

    /// Change a waiting message's text (`Some`), or take it out (`None`).
    pub fn edit_queued(&mut self, session: &str, message_id: u64, text: Option<String>) {
        self.send(Command::EditQueued {
            session_id: session.to_owned(),
            message_id,
            text,
        });
    }

    /// `session`'s waiting messages, oldest first.
    pub fn queued_for(&self, session: &str) -> &[QueuedMessage] {
        self.queued
            .get(session)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    // ----- Live surfaces (protocol 1.30) -------------------------------------

    /// The screens the owner can watch from `session`: its own (the one the
    /// agent waits on first), then the computer's displays, which belong to
    /// no conversation.
    pub fn surfaces_of(&self, session: &str) -> Vec<SurfaceInfo> {
        let mut found: Vec<SurfaceInfo> = self
            .state
            .surfaces
            .values()
            .filter(|s| s.session_id.is_none() || s.session_id.as_deref() == Some(session))
            .cloned()
            .collect();
        found.sort_by_key(|s| (s.session_id.is_none(), !s.handoff_pending, s.id.clone()));
        found
    }

    /// The tab of `session` where a run waits for the owner (a handoff).
    pub fn waiting_surface(&self, session: &str) -> Option<SurfaceInfo> {
        self.state
            .surfaces
            .values()
            .find(|s| s.handoff_pending && s.session_id.as_deref() == Some(session))
            .cloned()
    }

    pub fn surface(&self, id: &str) -> Option<&SurfaceInfo> {
        self.state.surfaces.get(id)
    }

    /// The waiting take-over card for `surface_id`, if any.
    pub fn handoff_for(&self, surface_id: &str) -> Option<Approval> {
        self.approvals().into_iter().find(|a| {
            matches!(
                a.card.as_ref().map(|c| &c.action),
                Some(aidaemon_client_protocol::CardAction::TakeOver { surface_id: s, .. }) if s == surface_id
            )
        })
    }

    /// Whether this device controls `surface_id` now.
    pub fn controls(&self, surface_id: &str) -> bool {
        self.live
            .as_ref()
            .is_some_and(|l| l.surface_id == surface_id)
            && matches!(
                self.surface(surface_id).map(|s| &s.controller),
                Some(SurfaceController::Owner { .. })
            )
            && self.live_controller_is_mine
    }

    /// Open a live view of `surface_id` (closing any other first); with
    /// `control`, take control too.
    pub fn open_surface(&mut self, surface_id: &str, control: bool) -> Vec<Effect> {
        self.open_surface_at(surface_id, control, None, None)
    }

    /// [`Self::open_surface`] asking for up to `max_fps` frames a second
    /// (a mouse wants a quicker picture than a finger; the daemon caps it),
    /// and with `fit`, the room this screen has for the picture: while it
    /// holds control, a page lays itself out for that room.
    pub fn open_surface_at(
        &mut self,
        surface_id: &str,
        control: bool,
        max_fps: Option<u32>,
        fit: Option<(u32, u32)>,
    ) -> Vec<Effect> {
        let mut effects = self.close_surface();
        self.control_error = None;
        self.live = Some(LiveView {
            surface_id: surface_id.to_owned(),
            control,
            live_id: None,
            transport: None,
            error: None,
        });
        self.live_request = Some(self.send(Command::OpenSurface {
            surface_id: surface_id.to_owned(),
            mode: if control {
                SurfaceMode::Control
            } else {
                SurfaceMode::View
            },
            accept: StreamAccept {
                codecs: vec!["jpeg".into()],
                max_fps,
                fit_w: fit.map(|(w, _)| w),
                fit_h: fit.map(|(_, h)| h),
                ..StreamAccept::default()
            },
        }));
        self.live_controller_is_mine = control;
        effects.retain(|e| *e == Effect::CloseLive);
        effects
    }

    /// Close the live view, if one is open.
    pub fn close_surface(&mut self) -> Vec<Effect> {
        let Some(live) = self.live.take() else {
            return Vec::new();
        };
        self.live_request = None;
        self.live_controller_is_mine = false;
        if let Some(live_id) = live.live_id {
            let id = self.send(Command::CloseSurface { live_id });
            self.close_requests.insert(id);
        }
        vec![Effect::CloseLive]
    }

    pub fn take_control(&mut self, surface_id: &str) {
        self.control_error = None;
        let id = self.send(Command::TakeControl {
            surface_id: surface_id.to_owned(),
        });
        self.control_requests.insert(id);
        self.live_controller_is_mine = true;
    }

    pub fn release_control(&mut self, surface_id: &str) {
        let id = self.send(Command::ReleaseControl {
            surface_id: surface_id.to_owned(),
        });
        self.control_requests.insert(id);
        self.live_controller_is_mine = false;
    }

    /// Answer the take-over card of `surface_id`: done (the step is made)
    /// or not now. Control goes back to the agent either way.
    pub fn finish_handoff(&mut self, surface_id: &str, done: bool) {
        if let Some(approval) = self.handoff_for(surface_id) {
            self.resolve(
                &approval.id,
                if done {
                    ApprovalDecision::ApproveOnce
                } else {
                    ApprovalDecision::Deny
                },
            );
        }
        if self.controls(surface_id) {
            self.release_control(surface_id);
        }
    }

    /// Point the agent at `(x, y)` (fractions of the frame) on `surface_id`,
    /// with an optional note: it arrives as the owner's message.
    pub fn point_at(&mut self, surface_id: &str, x: f32, y: f32, note: Option<String>) {
        self.send(Command::PointAt {
            surface_id: surface_id.to_owned(),
            x: f64::from(x.clamp(0.0, 1.0)),
            y: f64::from(y.clamp(0.0, 1.0)),
            note,
        });
    }

    // ----- Paired devices (protocol 1.31) ----------------------------------

    pub fn refresh_devices(&mut self) {
        if !self.devices.is_ready() {
            self.devices = Read::Loading;
        }
        self.devices_request = Some(self.send(Command::ListDevices));
    }

    /// Let `device_id` watch and control screens, or not.
    pub fn set_device_control(&mut self, device_id: &str, on: bool) {
        let Some((devices, _)) = self.devices.ready() else {
            return;
        };
        let Some(device) = devices.iter().find(|d| d.device_id == device_id) else {
            return;
        };
        let mut scopes: Vec<aidaemon_client_protocol::Scope> = device
            .scopes
            .iter()
            .copied()
            .filter(|s| *s != aidaemon_client_protocol::Scope::Control)
            .collect();
        if on {
            scopes.push(aidaemon_client_protocol::Scope::Control);
        }
        self.devices_error = None;
        self.devices_request = Some(self.send(Command::SetDeviceScopes {
            device_id: device_id.to_owned(),
            scopes,
        }));
    }

    /// Answers to the live view's own commands.
    fn live_answer(&mut self, frame: &ServerFrame, out: &mut Vec<Effect>) -> bool {
        match frame {
            ServerFrame::Ack {
                request_id,
                result:
                    CommandResult::Devices {
                        devices,
                        can_change,
                    },
            } if self.devices_request.as_deref() == Some(request_id) => {
                self.devices_request = None;
                self.devices = Read::Ready((devices.clone(), *can_change));
                true
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self.devices_request.as_deref() == Some(request_id) => {
                self.devices_request = None;
                self.devices_error = Some(error.message.clone());
                if !self.devices.is_ready() {
                    self.devices = Read::Failed(error.message.clone());
                }
                true
            }
            ServerFrame::Ack {
                request_id,
                result:
                    CommandResult::SurfaceOpened {
                        live_id,
                        surface,
                        transports,
                    },
            } if self.live_request.as_deref() == Some(request_id) => {
                self.live_request = None;
                self.state
                    .surfaces
                    .insert(surface.id.clone(), surface.clone());
                let Some(live) = self.live.as_mut() else {
                    // Closed meanwhile: close it on the daemon too.
                    self.send(Command::CloseSurface {
                        live_id: live_id.clone(),
                    });
                    return true;
                };
                let transport = transports
                    .iter()
                    .find(|t| t.kind == aidaemon_client_protocol::TransportKind::WsFrames)
                    .cloned();
                match transport {
                    Some(transport) => {
                        live.live_id = Some(live_id.clone());
                        live.transport = Some(transport.clone());
                        out.push(Effect::OpenLive {
                            live_id: live_id.clone(),
                            transport,
                        });
                    }
                    None => {
                        live.error =
                            Some("This daemon offers no way to show the screen here.".into());
                    }
                }
                true
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self.live_request.as_deref() == Some(request_id) => {
                self.live_request = None;
                self.live_controller_is_mine = false;
                if let Some(live) = self.live.as_mut() {
                    // An old link, or one answered elsewhere: the view is gone.
                    live.error = Some(
                        if error.code == aidaemon_client_protocol::ErrorCode::NotFound {
                            "This live view has ended: the request was answered or the page \
                             closed. Nothing is waiting here."
                                .to_owned()
                        } else {
                            error.message.clone()
                        },
                    );
                }
                true
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::SurfaceControl { surface },
            } if self.control_requests.contains(request_id) => {
                self.control_requests.remove(request_id);
                self.state
                    .surfaces
                    .insert(surface.id.clone(), surface.clone());
                true
            }
            ServerFrame::Ack { request_id, .. }
            | ServerFrame::Error {
                request_id: Some(request_id),
                ..
            } if self.close_requests.contains(request_id) => {
                self.close_requests.remove(request_id);
                true
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self.control_requests.contains(request_id) => {
                self.control_requests.remove(request_id);
                self.live_controller_is_mine = false;
                self.control_error = Some(error.message.clone());
                true
            }
            _ => false,
        }
    }

    pub fn resolve(&mut self, approval: &str, decision: ApprovalDecision) {
        self.send(Command::ResolveApproval {
            approval_id: approval.to_owned(),
            decision,
        });
    }

    /// The approval of `session` a single key answers: the oldest waiting
    /// one, while the composer's `draft` is empty. A key it does not offer
    /// answers nothing.
    pub fn key_answer(
        &self,
        session: &str,
        draft: &str,
        key: &str,
    ) -> Option<(String, ApprovalDecision)> {
        let decision = approval_for_key(key)?;
        self.key_approval(session, draft)
            .filter(|a| a.options.contains(&decision))
            .map(|a| (a.id, decision))
    }

    /// The approval single keys answer in `session` now: its oldest
    /// waiting one, while the composer's `draft` is empty.
    pub fn key_approval(&self, session: &str, draft: &str) -> Option<Approval> {
        if !draft.is_empty() {
            return None;
        }
        self.approvals()
            .into_iter()
            .find(|a| a.session_id == session)
    }

    /// Answer `session`'s waiting approval with `key` (see
    /// [`Self::key_answer`]); whether it did.
    pub fn answer_by_key(&mut self, session: &str, draft: &str, key: &str) -> bool {
        match self.key_answer(session, draft, key) {
            Some((id, decision)) => {
                self.resolve(&id, decision);
                true
            }
            None => false,
        }
    }

    /// Whether single keys answer an approval in `session` now: one waits
    /// and the composer's `draft` is empty.
    pub fn answers_by_key(&self, session: &str, draft: &str) -> bool {
        self.key_approval(session, draft).is_some()
    }
}

/// The keys that answer a waiting approval while the composer is empty,
/// with what each answers. Digits, as other agents' approval prompts use,
/// not letters: a message the owner starts typing ("no wait…", "add…")
/// would otherwise answer the card with its first letter. "Always allow"
/// has no key: it reaches too far for one stray keystroke.
pub const APPROVAL_KEYS: [(&str, ApprovalDecision); 3] = [
    ("1", ApprovalDecision::ApproveOnce),
    ("2", ApprovalDecision::ApproveSession),
    ("3", ApprovalDecision::Deny),
];

/// The answer `key` gives a waiting approval.
pub fn approval_for_key(key: &str) -> Option<ApprovalDecision> {
    APPROVAL_KEYS
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, d)| *d)
}

/// The key that gives `decision`, if one does.
pub fn key_for_approval(decision: ApprovalDecision) -> Option<&'static str> {
    APPROVAL_KEYS
        .iter()
        .find(|(_, d)| *d == decision)
        .map(|(k, _)| *k)
}

/// Whether `run` finished after `seen_ms` with something to read.
pub fn run_is_news(run: &aidaemon_client_protocol::AutomationRun, seen_ms: u64) -> bool {
    use aidaemon_client_protocol::RunOutcome;
    let ended = run.finished_ms.unwrap_or(run.started_ms);
    ended > seen_ms
        && match run.outcome {
            RunOutcome::Failed | RunOutcome::Blocked | RunOutcome::Missed => true,
            RunOutcome::Done => run.summary.as_deref().is_some_and(|s| !s.trim().is_empty()),
            RunOutcome::Running | RunOutcome::Cancelled | RunOutcome::Unknown => false,
        }
}

/// Where a session came from: its `channel` (`telegram:coding_bot`,
/// `app`), else what its id names.
pub fn platform_of(channel: &str, id: &str) -> Source {
    let platform = channel.split(':').next().unwrap_or_default();
    match platform {
        "app" => Source::App,
        "telegram" => Source::Telegram,
        "slack" => Source::Slack,
        "discord" => Source::Discord,
        _ => match platform_name(id) {
            "App" => Source::App,
            "Telegram" => Source::Telegram,
            "Slack" => Source::Slack,
            "Discord" => Source::Discord,
            _ => Source::Chat,
        },
    }
}

/// A channel as people name it: "Telegram · coding_bot", "App".
pub fn channel_name(channel: &str, id: &str) -> String {
    let platform = match platform_of(channel, id) {
        Source::App => "App",
        Source::Telegram => "Telegram",
        Source::Slack => "Slack",
        Source::Discord => "Discord",
        _ => platform_name(id),
    };
    match channel.split_once(':') {
        Some((_, bot)) if !bot.is_empty() => format!("{platform} · {bot}"),
        _ => platform.to_owned(),
    }
}

/// The platform a session id names (`aidaemon:slack:…`, `app:…`).
pub fn platform_name(id: &str) -> &'static str {
    let lower = id.to_ascii_lowercase();
    let has = |word: &str| {
        lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|part| part == word)
    };
    if lower.starts_with("app:") {
        "App"
    } else if has("slack") {
        "Slack"
    } else if has("discord") {
        "Discord"
    } else if has("telegram") || lower.contains("_bot:") {
        "Telegram"
    } else {
        "Chat"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(json: &str) -> Update {
        Update::Frame(serde_json::from_str::<ServerFrame>(json).expect("a frame"))
    }

    fn cx(selected: Option<&str>) -> Context {
        Context {
            now_ms: 1_000,
            selected: selected.map(str::to_owned),
            ..Context::default()
        }
    }

    fn surface_json(id: &str, waiting: bool) -> String {
        format!(
            r#"{{"id":"{id}","kind":"browser_tab","session_id":"app:s1","title":"Sign in","site":"login.acme.example","width":1280,"height":720,"controller":{{"kind":"agent"}},"handoff_pending":{waiting},"interactive":true}}"#
        )
    }

    fn surface_command(client: &mut Client) -> (String, Command) {
        client
            .take_outgoing()
            .into_iter()
            .rfind(|(_, c)| {
                matches!(
                    c,
                    Command::OpenSurface { .. }
                        | Command::CloseSurface { .. }
                        | Command::TakeControl { .. }
                        | Command::ReleaseControl { .. }
                        | Command::ResolveApproval { .. }
                )
            })
            .expect("a surface command")
    }

    #[test]
    fn a_settings_read_is_unsupported_only_on_a_daemon_older_than_1_4() {
        for (minor, unsupported) in [(3, true), (34, false)] {
            let mut client = Client::new();
            client.update(Update::Connected, &cx(None));
            client.update(
                frame(&format!(
                    r#"{{"type":"welcome","version":{{"major":1,"minor":{minor}}},"owner":{{"name":"Alice Rivera"}},"latest_seq":0}}"#
                )),
                &cx(None),
            );
            client.refresh_settings();
            let (id, _) = client
                .take_outgoing()
                .into_iter()
                .rfind(|(_, c)| matches!(c, Command::GetSettings))
                .expect("a settings read");
            client.update(
                frame(&format!(
                    r#"{{"type":"error","request_id":"{id}","error":{{"code":"invalid","message":"Unknown command for this gateway version","retryable":false}}}}"#
                )),
                &cx(None),
            );
            assert_eq!(client.settings_unsupported(), unsupported, "1.{minor}");
        }
    }

    #[test]
    fn retry_follows_the_turn_end_not_the_note_text() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        // A daemon note that happens to read like a stop is not a stop.
        client.update(
            frame(
                r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:s1","event":{"type":"user_action_required","turn_id":"t0","message":"Stopped."}}"#,
            ),
            &cx(None),
        );
        assert!(!client.can_retry("app:s1"));
        for (seq, state) in [(2, "failed"), (3, "cancelled")] {
            client.update(
                frame(&format!(
                    r#"{{"type":"event","at_ms":2,"session_id":"app:s1","event":{{"type":"turn_started","turn_id":"t{seq}","session_id":"app:s1","origin":"app"}}}}"#
                )),
                &cx(None),
            );
            client.update(
                frame(&format!(
                    r#"{{"type":"event","seq":{seq},"at_ms":2,"session_id":"app:s1","event":{{"type":"turn_completed","turn_id":"t{seq}","text":"","state":"{state}","error":"synthetic outage"}}}}"#
                )),
                &cx(None),
            );
            assert!(client.can_retry("app:s1"), "{state}");
        }
        client.update(
            frame(
                r#"{"type":"event","at_ms":3,"session_id":"app:s1","event":{"type":"turn_started","turn_id":"t4","session_id":"app:s1","origin":"app"}}"#,
            ),
            &cx(None),
        );
        client.update(
            frame(
                r#"{"type":"event","seq":4,"at_ms":3,"session_id":"app:s1","event":{"type":"turn_completed","turn_id":"t4","text":"Done with the synthetic notes.","state":"completed"}}"#,
            ),
            &cx(None),
        );
        assert!(!client.can_retry("app:s1"));
    }

    #[test]
    fn the_same_files_read_unanswered_is_not_sent_again() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.take_outgoing();
        // A view drawn again and again while offline asks each time.
        for _ in 0..8 {
            client.list_project_files("p-acme", "", false);
            client.read_project_file("p-acme", "README.md");
        }
        assert_eq!(client.take_outgoing().len(), 2);
        // Another read is sent, and once answered the same one may be again.
        client.list_project_files("p-acme", "", true);
        let (id, _) = client.take_outgoing().pop().unwrap();
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"project_dir","dir":{{"project_id":"p-acme"}}}}}}"#,
            )),
            &cx(None),
        );
        client.list_project_files("p-acme", "", true);
        assert_eq!(client.take_outgoing().len(), 1);
    }

    #[test]
    fn a_projects_files_read_keep_only_the_latest_answer() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.take_outgoing();
        client.list_project_files("p-acme", "", false);
        client.list_project_files("p-acme", "src", false);
        let sent = client.take_outgoing();
        assert!(matches!(
            &sent[1].1,
            Command::ListProjectFiles { path: Some(p), .. } if p == "src"
        ));
        assert!(matches!(client.files_dir, Read::Loading));
        // The first folder's answer comes late: it is not the one asked for.
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{}","result":{{"type":"project_dir","dir":{{"project_id":"p-acme"}}}}}}"#,
                sent[0].0
            )),
            &cx(None),
        );
        assert!(matches!(client.files_dir, Read::Loading));
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{}","result":{{"type":"project_dir","dir":{{"project_id":"p-acme","path":"src","entries":[{{"name":"contact.rs","kind":"file"}}]}}}}}}"#,
                sent[1].0
            )),
            &cx(None),
        );
        assert_eq!(client.files_dir.ready().unwrap().entries.len(), 1);

        client.read_project_file("p-acme", "src/contact.rs");
        let (request, _) = client.take_outgoing().pop().unwrap();
        client.close_project_file();
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{request}","result":{{"type":"project_file","file":{{"project_id":"p-acme","path":"src/contact.rs","size_bytes":3,"version":"sha256:00","text":"abc"}}}}}}"#
            )),
            &cx(None),
        );
        assert!(matches!(client.files_file, Read::NotYet));
        client.read_project_file("p-acme", "nope.rs");
        let (request, _) = client.take_outgoing().pop().unwrap();
        client.update(
            frame(&format!(
                r#"{{"type":"error","request_id":"{request}","error":{{"code":"not_found","message":"No such file in this project","retryable":false}}}}"#
            )),
            &cx(None),
        );
        assert!(matches!(&client.files_file, Read::Failed(m) if m.contains("No such file")));
    }

    #[test]
    fn a_live_view_opens_with_its_transport_and_closes_with_its_surface() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.update(
            frame(&format!(
                r#"{{"type":"event","at_ms":1,"event":{{"type":"surface_updated","surface":{}}}}}"#,
                surface_json("srf_1", true)
            )),
            &cx(None),
        );
        assert_eq!(client.surfaces_of("app:s1").len(), 1);
        client.open_surface("srf_1", true);
        let (request, command) = surface_command(&mut client);
        assert!(matches!(
            command,
            Command::OpenSurface {
                mode: SurfaceMode::Control,
                ..
            }
        ));
        let effects = client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{request}","result":{{"type":"surface_opened","live_id":"live_1","surface":{},"transports":[{{"kind":"ws_frames","url":"/client/v1/surfaces/live_1/ws","ticket":"tkt_1","codecs":["jpeg"]}}]}}}}"#,
                surface_json("srf_1", true)
            )),
            &cx(None),
        );
        assert!(
            matches!(&effects[..], [Effect::OpenLive { live_id, transport }] if live_id == "live_1" && transport.ticket == "tkt_1")
        );
        assert_eq!(
            client.live.as_ref().unwrap().live_id.as_deref(),
            Some("live_1")
        );

        let effects = client.update(
            frame(r#"{"type":"event","at_ms":2,"event":{"type":"surface_closed","surface_id":"srf_1"}}"#),
            &cx(None),
        );
        assert_eq!(effects, vec![Effect::CloseLive]);
        assert!(client.live.is_none());
        assert!(client.surfaces_of("app:s1").is_empty());
    }

    /// Live 2026-11-17: Done ended the view, the app closed it too, and
    /// "Already closed" stayed on screen with a Dismiss button.
    #[test]
    fn closing_a_view_says_nothing() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.open_surface("srf_1", true);
        let (request, _) = surface_command(&mut client);
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{request}","result":{{"type":"surface_opened","live_id":"live_1","surface":{},"transports":[{{"kind":"ws_frames","url":"/client/v1/surfaces/live_1/ws","ticket":"tkt_1","codecs":["jpeg"]}}]}}}}"#,
                surface_json("srf_1", true)
            )),
            &cx(None),
        );
        client.close_surface();
        let close = client
            .take_outgoing()
            .into_iter()
            .find_map(|(request_id, command)| {
                matches!(command, Command::CloseSurface { .. }).then_some(request_id)
            })
            .expect("a close");
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{close}","result":{{"type":"reply","text":"Already closed"}}}}"#
            )),
            &cx(None),
        );
        assert_eq!(client.state.notice, None);
    }

    #[test]
    fn a_view_asks_for_the_frame_rate_and_room_its_device_has() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.open_surface_at("srf_1", true, Some(20), Some((390, 560)));
        let (_, command) = surface_command(&mut client);
        assert!(matches!(
            command,
            Command::OpenSurface { accept, .. }
                if accept.max_fps == Some(20)
                    && accept.fit_w == Some(390)
                    && accept.fit_h == Some(560)
        ));
        client.open_surface("srf_1", false);
        let (_, command) = surface_command(&mut client);
        assert!(matches!(
            command,
            Command::OpenSurface { accept, .. }
                if accept.max_fps.is_none() && accept.fit_w.is_none()
        ));
    }

    #[test]
    fn a_link_to_an_ended_view_says_it_ended() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.open_surface("srf_gone", true);
        let (request, _) = surface_command(&mut client);
        client.update(
            frame(&format!(
                r#"{{"type":"error","request_id":"{request}","error":{{"code":"not_found","message":"no such surface","retryable":false}}}}"#
            )),
            &cx(None),
        );
        let error = client.live.as_ref().unwrap().error.clone().unwrap();
        assert!(error.starts_with("This live view has ended"), "{error}");
    }

    #[test]
    fn a_refused_view_says_why_and_closing_tells_the_daemon() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.open_surface("srf_9", false);
        let (request, _) = surface_command(&mut client);
        client.update(
            frame(&format!(
                r#"{{"type":"error","request_id":"{request}","error":{{"code":"forbidden","message":"This device may not Control","retryable":false}}}}"#
            )),
            &cx(None),
        );
        assert_eq!(
            client.live.as_ref().unwrap().error.as_deref(),
            Some("This device may not Control")
        );
        assert_eq!(client.close_surface(), vec![Effect::CloseLive]);
        assert!(client.live.is_none());
    }

    #[test]
    fn the_control_switch_keeps_a_devices_other_scopes() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.refresh_devices();
        let (request, _) = client
            .take_outgoing()
            .into_iter()
            .rfind(|(_, c)| matches!(c, Command::ListDevices))
            .unwrap();
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{request}","result":{{"type":"devices","devices":[{{"device_id":"d1","name":"Synthetic phone","kind":"mobile","scopes":["observe","chat","approve","control"],"created_at_ms":1}}],"can_change":true}}}}"#
            )),
            &cx(None),
        );
        assert!(matches!(&client.devices, Read::Ready((d, true)) if d.len() == 1));
        client.set_device_control("d1", false);
        let (_, command) = client.take_outgoing().pop().unwrap();
        assert_eq!(
            command,
            Command::SetDeviceScopes {
                device_id: "d1".into(),
                scopes: vec![
                    aidaemon_client_protocol::Scope::Observe,
                    aidaemon_client_protocol::Scope::Chat,
                    aidaemon_client_protocol::Scope::Approve
                ]
            }
        );
    }

    #[test]
    fn done_answers_the_take_over_card() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.update(
            frame(r#"{"type":"event","seq":3,"at_ms":1,"session_id":"app:s1","event":{"type":"approval_requested","approval":{"id":"a-1","session_id":"app:s1","kind":"handoff","risk":"low","summary":"Sign in, then press Done.","card":{"action":{"kind":"take_over","surface_id":"srf_1","what":"Sign in"}},"options":["approve_once","deny"],"requested_at_ms":1}}}"#),
            &cx(None),
        );
        assert!(client.handoff_for("srf_1").is_some());
        client.finish_handoff("srf_1", true);
        let (_, command) = surface_command(&mut client);
        assert_eq!(
            command,
            Command::ResolveApproval {
                approval_id: "a-1".into(),
                decision: ApprovalDecision::ApproveOnce
            }
        );
    }

    fn settings_reads(client: &mut Client) -> Vec<String> {
        client
            .take_outgoing()
            .into_iter()
            .filter(|(_, c)| matches!(c, Command::GetSettings))
            .map(|(id, _)| id)
            .collect()
    }

    fn settings_ack(id: &str) -> Update {
        frame(&format!(
            r#"{{"type":"ack","request_id":"{id}","result":{{"type":"settings","settings":{{"config_path":"/home/synthetic/config.toml","database_path":"","groups":[]}}}}}}"#
        ))
    }

    #[test]
    fn leaving_the_chat_step_opens_the_greeting_on_first_run_only() {
        let created = |id: &str| {
            frame(&format!(
                r#"{{"type":"event","seq":1,"at_ms":5,"session_id":"{id}","event":{{"type":"session_created","session":{{"id":"{id}","channel":"app","title":"Welcome","writable":true}}}}}}"#
            ))
        };
        // First run: the next conversation is the greeting.
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.setup_step(SetupAction::SkipChannel);
        let effects = client.update(created("app:welcome"), &cx(None));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::OpenGreeting(id) if id == "app:welcome"))
        );
        // Later, from Settings: none is opened.
        client.setup_step_later(SetupAction::SkipChannel);
        let effects = client.update(created("app:synthetic-9"), &cx(None));
        assert!(!effects.iter().any(|e| matches!(e, Effect::OpenGreeting(_))));
    }

    #[test]
    fn checking_agents_again_is_one_settings_change() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.take_outgoing();
        client.check_agents();
        let mut sent = client.take_outgoing();
        assert_eq!(sent.len(), 1);
        let (id, command) = sent.pop().unwrap();
        assert_eq!(
            command,
            Command::ChangeSettings {
                change: SettingsChange::CheckAgents
            }
        );
        assert!(client.settings_busy());
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"settings_changed","outcome":{{"kind":"done","message":"Checked the coding agents"}}}}}}"#
            )),
            &cx(None),
        );
        assert!(!client.settings_busy());
    }

    #[test]
    fn an_agent_mode_change_is_sent_once_and_full_access_waits_for_approval() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.take_outgoing();
        client.refresh_settings();
        let read = settings_reads(&mut client);
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{}","result":{{"type":"settings","settings":{{"config_path":"/home/synthetic/config.toml","database_path":"","groups":[],"coding_agents":[{{"name":"claude","program":"Claude Code","mode":"checked","modes":["checked","every_command","full_access"]}},{{"name":"synthetic-agent","program":"Acme Coder","modes":["full_access"],"note":"Acme Coder cannot be checked."}}]}}}}}}"#,
                read[0]
            )),
            &cx(None),
        );
        let Read::Ready(settings) = &client.settings else {
            panic!("settings not ready");
        };
        assert_eq!(settings.coding_agents.len(), 2);
        assert_eq!(settings.coding_agents[1].mode, None);
        // The mode it has, or one it does not offer: nothing sent.
        client.set_agent_mode("claude", AgentMode::Checked);
        client.set_agent_mode("synthetic-agent", AgentMode::Checked);
        assert!(client.take_outgoing().is_empty());
        // Full access: sent, and the daemon asks for approval first.
        client.set_agent_mode("claude", AgentMode::FullAccess);
        let mut sent = client.take_outgoing();
        assert_eq!(sent.len(), 1);
        let (id, command) = sent.pop().unwrap();
        assert_eq!(
            command,
            Command::ChangeSettings {
                change: SettingsChange::SetAgentMode {
                    agent: "claude".into(),
                    mode: AgentMode::FullAccess
                }
            }
        );
        assert!(client.settings_busy());
        let effects = client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"settings_changed","outcome":{{"kind":"awaiting_approval","approval_id":"ap-1"}}}}}}"#
            )),
            &cx(None),
        );
        assert!(!client.settings_busy());
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::Say(s) if s == "Waiting for your approval"))
        );
    }

    #[test]
    fn settings_asked_for_meanwhile_or_lost_to_a_restart_are_read_again() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.take_outgoing();
        client.refresh_settings();
        let first = settings_reads(&mut client);
        assert_eq!(first.len(), 1);
        // A change lands while the read is out: one more read after it.
        client.refresh_settings();
        client.refresh_settings();
        assert!(settings_reads(&mut client).is_empty());
        client.update(settings_ack(&first[0]), &cx(None));
        assert!(client.settings.is_ready());
        let again = settings_reads(&mut client);
        assert_eq!(again.len(), 1);
        // The daemon restarts with that read unanswered: the reconnect reads
        // afresh, and later changes are not stuck behind the lost read.
        client.update(
            Update::Disconnected(crate::machine::Disconnect {
                kind: crate::machine::DropKind::DaemonNotRunning,
                reason: "restarting".into(),
                retry_in: std::time::Duration::from_millis(10),
            }),
            &cx(None),
        );
        client.update(Update::Connected, &cx(None));
        let fresh = settings_reads(&mut client);
        assert_eq!(fresh.len(), 1);
        assert_ne!(fresh[0], again[0]);
        client.update(settings_ack(&fresh[0]), &cx(None));
        client.refresh_settings();
        assert_eq!(settings_reads(&mut client).len(), 1);
    }

    #[test]
    fn a_marked_up_picture_names_the_picture_it_was_drawn_over() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.take_outgoing();
        client
            .attach_marked_up(
                "shot-marked.jpg".into(),
                "image/jpeg".into(),
                b"jpeg",
                "0123456789abcdef01234567".into(),
            )
            .unwrap();
        let (_, sent) = client.take_outgoing().pop().unwrap();
        assert!(matches!(
            sent,
            Command::UploadFile { marked_from: Some(ref id), .. } if id == "0123456789abcdef01234567"
        ));
        assert_eq!(
            client.draft_files[0].marked_from.as_deref(),
            Some("0123456789abcdef01234567")
        );
        // A plain attachment names none.
        client
            .attach_file("notes.txt".into(), "text/plain".into(), b"hi")
            .unwrap();
        let (_, sent) = client.take_outgoing().pop().unwrap();
        assert!(matches!(
            sent,
            Command::UploadFile {
                marked_from: None,
                ..
            }
        ));
    }

    #[test]
    fn a_deleted_project_leaves_its_conversations_in_none() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.take_outgoing();
        client.update(
            frame(r#"{"type":"event","seq":1,"at_ms":1,"event":{"type":"project_updated","project":{"id":"project-acme","name":"Acme site"}}}"#),
            &cx(None),
        );
        client.update(
            frame(r#"{"type":"event","seq":2,"at_ms":1,"session_id":"app:6c1f","event":{"type":"session_created","session":{"id":"app:6c1f","channel":"app","title":"Fix the footer","last_activity_ms":1,"writable":true,"project_id":"project-acme"}}}"#),
            &cx(None),
        );
        assert_eq!(
            client.session_project("app:6c1f").as_deref(),
            Some("project-acme")
        );
        client.delete_project("project-acme");
        let (id, sent) = client.take_outgoing().pop().unwrap();
        assert!(matches!(sent, Command::DeleteProject { .. }));
        client.update(
            frame(r#"{"type":"event","seq":3,"at_ms":1,"event":{"type":"project_deleted","project_id":"project-acme"}}"#),
            &cx(None),
        );
        let said = client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"projects","projects":[]}}}}"#
            )),
            &cx(None),
        );
        assert!(client.projects().is_empty());
        assert_eq!(client.session_project("app:6c1f"), None);
        assert!(
            said.contains(&Effect::Say("Deleted Acme site".to_owned())),
            "{said:?}"
        );
    }

    #[test]
    fn a_missing_project_folder_is_offered_to_be_made() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.take_outgoing();
        assert!(client.save_project(None, "Acme site", "~/projects/acme-site", false));
        let (id, sent) = client.take_outgoing().pop().unwrap();
        assert!(matches!(
            sent,
            Command::CreateProject {
                create_folder: false,
                ..
            }
        ));
        client.update(
            frame(&format!(
                r#"{{"type":"error","request_id":"{id}","error":{{"code":"folder_missing","message":"There is no folder at /home/synthetic/projects/acme-site on this computer","retryable":false}}}}"#
            )),
            &cx(None),
        );
        assert_eq!(
            client.project_missing_folder.as_deref(),
            Some("~/projects/acme-site")
        );
        // Confirmed: the same save asks the daemon to make it.
        assert!(client.save_project(None, "Acme site", "~/projects/acme-site", true));
        let (id, sent) = client.take_outgoing().pop().unwrap();
        assert!(matches!(
            sent,
            Command::CreateProject {
                create_folder: true,
                ..
            }
        ));
        // Any other refusal offers nothing.
        client.update(
            frame(&format!(
                r#"{{"type":"error","request_id":"{id}","error":{{"code":"invalid","message":"There is already a project called Acme site","retryable":false}}}}"#
            )),
            &cx(None),
        );
        assert!(client.project_missing_folder.is_none());
        assert!(client.project_error.is_some());
    }

    #[test]
    fn connecting_asks_for_the_lists_and_reads_answer_by_request_id() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        let sent: Vec<String> = client
            .take_outgoing()
            .into_iter()
            .map(|(_, c)| {
                serde_json::to_value(c).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(
            sent,
            [
                "list_sessions",
                "list_commands",
                "list_projects",
                "get_system",
                "list_automations"
            ]
        );
        client.refresh_memories();
        let (id, _) = client.take_outgoing().pop().unwrap();
        assert_eq!(client.memories, Read::Loading);
        // Someone else's answer is not this read's.
        client.update(
            frame(
                r#"{"type":"ack","request_id":"other","result":{"type":"memories","memories":[]}}"#,
            ),
            &cx(None),
        );
        assert_eq!(client.memories, Read::Loading);
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"memories","memories":[]}}}}"#
            )),
            &cx(None),
        );
        assert_eq!(client.memories, Read::Ready(Vec::new()));
    }

    #[test]
    fn a_reply_elsewhere_is_unread_and_a_first_file_write_shows_the_changes() {
        let mut client = Client::new();
        let here = cx(Some("app:a"));
        client.update(
            frame(r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:b","event":{"type":"assistant_message","session_id":"app:b","text":"hi"}}"#),
            &here,
        );
        assert!(client.unread.contains("app:b"));
        client.update(
            frame(r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:a","event":{"type":"turn_started","turn_id":"t1","session_id":"app:a","origin":"app"}}"#),
            &here,
        );
        let effects = client.update(
            frame(r#"{"type":"event","at_ms":3,"session_id":"app:a","event":{"type":"turn_status","turn_id":"t1","status":{"kind":"tool_completed","name":"write_file","summary":"a.rs"}}}"#),
            &here,
        );
        assert_eq!(
            effects,
            [
                Effect::NowCoding("app:a".into()),
                Effect::OpenChanges("app:a".into())
            ]
        );
        assert!(client.is_coding("app:a"));
        assert_eq!(client.changes, Read::Loading);
        let log = client.latest_turn("app:a").unwrap();
        assert_eq!(log.items.len(), 1);
        assert_eq!(log.items[0].kind, ActivityKind::ToolDone);
    }

    #[test]
    fn only_runs_with_something_to_read_after_the_last_look_are_news() {
        use aidaemon_client_protocol::{AutomationRun, RunOutcome};
        let run = |finished: u64, outcome, summary: Option<&str>| AutomationRun {
            started_ms: finished - 10,
            finished_ms: Some(finished),
            outcome,
            summary: summary.map(str::to_owned),
        };
        assert!(run_is_news(
            &run(200, RunOutcome::Done, Some("2 open PRs.")),
            100
        ));
        assert!(run_is_news(&run(200, RunOutcome::Failed, None), 100));
        assert!(!run_is_news(&run(200, RunOutcome::Done, None), 100));
        assert!(!run_is_news(&run(90, RunOutcome::Failed, None), 100));
        // A run that was due and never ran is news, though not a failure.
        assert!(run_is_news(&run(200, RunOutcome::Missed, None), 100));
        assert!(!run_is_news(&run(90, RunOutcome::Missed, None), 100));
    }

    /// A missed run arrives in an automation's runs as its own outcome,
    /// newest first, and counts as news since the last look.
    #[test]
    fn a_missed_run_is_kept_in_the_automations_and_is_news() {
        use aidaemon_client_protocol::RunOutcome;
        let mut client = Client::new();
        client.refresh_automations();
        let (id, _) = client.take_outgoing().pop().unwrap();
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"automations","automations":[{{"id":"goal-1","kind":"scheduled","name":"Inbox digest","what":"Summarize the synthetic inbox","how_often":"Every day at 9:00 AM ET","status":"active","recent_runs":[{{"started_ms":300,"finished_ms":400,"outcome":"missed","summary":"The daemon wasn't running when this was due."}},{{"started_ms":100,"finished_ms":120,"outcome":"done"}}]}}]}}}}"#
            )),
            &cx(None),
        );
        let all = client.automations.ready().unwrap();
        assert_eq!(all[0].recent_runs[0].outcome, RunOutcome::Missed);
        assert_eq!(all[0].recent_runs[1].outcome, RunOutcome::Done);
        assert_eq!(client.automation_news(200), 1);
    }

    #[test]
    fn a_queued_message_reads_the_queue_and_a_revert_reads_the_changes() {
        let mut client = Client::new();
        let here = cx(Some("app:a"));
        client
            .submit(Some("app:a"), "after this", Some(MidRun::Queue), None)
            .unwrap();
        let (id, _) = client.take_outgoing().pop().unwrap();
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"message_accepted","session_id":"app:a"}}}}"#
            )),
            &here,
        );
        let (_, list) = client.take_outgoing().pop().unwrap();
        assert!(matches!(list, Command::ListQueued { .. }));
        client.update(
            frame(r#"{"type":"ack","request_id":"q","result":{"type":"queued","session_id":"app:a","messages":[{"id":7,"text":"after this","queued_at_ms":5}]}}"#),
            &here,
        );
        assert_eq!(client.queued_for("app:a")[0].id, 7);

        client.revert_file("app:a", "/repo", "a.rs");
        let (id, _) = client.take_outgoing().pop().unwrap();
        let effects = client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"changes","changes":{{"repos":[]}}}}}}"#
            )),
            &here,
        );
        assert_eq!(effects, [Effect::Say("Undid the changes to a.rs".into())]);
        assert!(client.changes.is_ready());
        assert!(!client.reverting());
    }

    #[test]
    fn a_message_for_the_running_work_reads_the_waiting_list_too() {
        let mut client = Client::new();
        let here = cx(Some("app:a"));
        client
            .submit(Some("app:a"), "read this now", Some(MidRun::Steer), None)
            .unwrap();
        let (id, _) = client.take_outgoing().pop().unwrap();
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"message_accepted","session_id":"app:a"}}}}"#
            )),
            &here,
        );
        let (_, list) = client.take_outgoing().pop().unwrap();
        assert!(matches!(list, Command::ListQueued { .. }));
    }

    /// Live 2026-11-17: the running turn read "AID inspired." and showed it
    /// as the owner's message, while the waiting list kept it until the turn
    /// ended. The owner's message arriving reads the list afresh.
    #[test]
    fn a_message_the_running_turn_reads_leaves_the_waiting_list() {
        let mut client = Client::new();
        let here = cx(Some("app:a"));
        client.update(
            frame(r#"{"type":"ack","request_id":"q","result":{"type":"queued","session_id":"app:a","messages":[{"id":7,"text":"AID inspired.","queued_at_ms":5}]}}"#),
            &here,
        );
        client.take_outgoing();
        client.update(
            frame(r#"{"type":"event","seq":4,"at_ms":6,"session_id":"app:a","event":{"type":"user_message","session_id":"app:a","text":"AID inspired.","origin":"app"}}"#),
            &here,
        );
        let sent = client.take_outgoing();
        assert!(
            sent.iter().any(
                |(_, c)| matches!(c, Command::ListQueued { session_id } if session_id == "app:a")
            ),
            "{sent:?}"
        );
    }

    /// Live 2026-11-18: a stop's report became the app-wide notice and
    /// showed on every other conversation until dismissed. It is a note in
    /// the conversation that was stopped.
    #[test]
    fn a_stop_report_is_a_note_in_the_stopped_conversation() {
        let mut client = Client::new();
        let here = cx(Some("app:a"));
        client.stop("app:a");
        let (id, _) = client.take_outgoing().pop().unwrap();
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"stopped","report":"Stopped: working on your message"}}}}"#
            )),
            &here,
        );
        assert_eq!(client.state.notice, None);
        let lines = &client.state.sessions["app:a"].lines;
        assert!(
            matches!(lines.last(), Some(Line::Note { text, ended: None }) if text.contains("Stopped")),
            "{lines:?}"
        );
    }

    #[test]
    fn places_name_the_platform_and_bot() {
        assert_eq!(
            channel_name("telegram:coding_bot", ""),
            "Telegram · coding_bot"
        );
        assert_eq!(channel_name("", "app:x"), "App");
        assert_eq!(platform_of("", "aidaemon:slack:T1"), Source::Slack);
        assert_eq!(platform_name("synthetic_bot:7"), "Telegram");
    }

    #[test]
    fn a_new_conversation_is_started_once_its_message_is_accepted() {
        let mut client = Client::new();
        assert_eq!(client.submit(None, "hello", None, None), Ok(false));
        let (id, _) = client.take_outgoing().pop().unwrap();
        let effects = client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"message_accepted","session_id":"app:new"}}}}"#
            )),
            &cx(None),
        );
        assert_eq!(effects, [Effect::Started("app:new".into())]);
        assert_eq!(client.submit(None, "  ", None, None), Ok(false));
    }

    #[test]
    fn a_command_typed_in_a_new_conversation_is_not_sent_as_a_message() {
        let mut client = Client::new();
        client.commands = vec![SlashCommand {
            name: "model".into(),
            description: "Show or switch AI model".into(),
        }];
        let refused = client.submit(None, "/model synthetic-claude", None, None);
        assert!(
            refused
                .unwrap_err()
                .starts_with("/model runs inside a conversation")
        );
        assert!(
            client.take_outgoing().is_empty(),
            "nothing went to the agent"
        );
        // A path is not a command: it is sent as the message it is.
        assert_eq!(
            client.submit(None, "/etc/hosts looks wrong", None, None),
            Ok(false)
        );
        assert!(matches!(
            client.take_outgoing().pop(),
            Some((_, Command::SendMessage { .. }))
        ));
    }

    #[test]
    fn a_conversations_model_is_read_again_after_model_or_auto() {
        let mut client = Client::new();
        client
            .submit(Some("app:a"), "/model synthetic-gemini", None, None)
            .unwrap();
        let (id, command) = client.take_outgoing().pop().unwrap();
        assert!(matches!(command, Command::RunSlash { .. }), "{command:?}");
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"reply","text":"This conversation now uses synthetic-gemini."}}}}"#
            )),
            &cx(None),
        );
        let (_, reread) = client.take_outgoing().pop().unwrap();
        assert_eq!(
            reread,
            Command::GetSessionModel {
                session_id: "app:a".into()
            }
        );
    }

    fn models_ready(client: &mut Client) {
        client.refresh_models();
        let (id, command) = client.take_outgoing().pop().unwrap();
        assert_eq!(command, Command::ListModels);
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"models","options":{{"providers":[
                    {{"id":"main","name":"Anthropic","billing":"per_token","models":[{{"id":"synthetic-claude","efforts":["low","medium","high"]}}]}},
                    {{"id":"failover-0","name":"Google Gemini","billing":"per_token","models":[{{"id":"synthetic-gemini"}}]}}
                ]}}}}}}"#
            )),
            &cx(None),
        );
        assert!(client.models.is_ready());
    }

    fn gemini() -> ModelChoice {
        ModelChoice::Pinned {
            provider: "failover-0".into(),
            model: "synthetic-gemini".into(),
        }
    }

    #[test]
    fn a_new_conversation_starts_on_its_pick_and_keeps_it() {
        let mut client = Client::new();
        models_ready(&mut client);
        client.set_session_model(None, None, Some(Effort::High));
        assert_eq!(client.model_of(None), (ModelChoice::Auto, Effort::High));
        // Gemini takes no effort: picking it drops the effort.
        client.set_session_model(None, Some(gemini()), None);
        assert_eq!(client.model_of(None), (gemini(), Effort::Auto));
        assert!(client.take_outgoing().is_empty(), "nothing sent yet");

        client.submit(None, "hello", None, None).unwrap();
        let (id, command) = client.take_outgoing().pop().unwrap();
        let Command::SendMessage { model, effort, .. } = command else {
            panic!("{command:?}");
        };
        assert_eq!(model, Some(gemini()));
        assert_eq!(effort, None);
        assert_eq!(client.model_of(None), (ModelChoice::Auto, Effort::Auto));
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"message_accepted","session_id":"app:new","new_session":true}}}}"#
            )),
            &cx(None),
        );
        assert_eq!(client.model_of(Some("app:new")), (gemini(), Effort::Auto));
    }

    #[test]
    fn a_conversations_pick_is_sent_and_a_refusal_reads_it_again() {
        let mut client = Client::new();
        models_ready(&mut client);
        assert!(!client.can_pick_model(Some("telegram:synthetic-user-1")));
        client.set_session_model(Some("telegram:synthetic-user-1"), Some(gemini()), None);
        assert!(client.take_outgoing().is_empty());

        client.set_session_model(Some("app:a"), Some(gemini()), None);
        assert_eq!(client.model_of(Some("app:a")).0, gemini());
        let (id, command) = client.take_outgoing().pop().unwrap();
        assert!(
            matches!(command, Command::SetSessionModel { .. }),
            "{command:?}"
        );
        let effects = client.update(
            frame(&format!(
                r#"{{"type":"error","request_id":"{id}","error":{{"code":"invalid","message":"synthetic-gemini is not a model this daemon is set up with","retryable":false}}}}"#
            )),
            &cx(None),
        );
        assert!(matches!(&effects[..], [Effect::Say(_)]), "{effects:?}");
        let (id, command) = client.take_outgoing().pop().unwrap();
        assert_eq!(
            command,
            Command::GetSessionModel {
                session_id: "app:a".into()
            }
        );
        client.update(
            frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"session_model","session_model":{{"session_id":"app:a","model":{{"mode":"auto"}},"effort":"auto"}}}}}}"#
            )),
            &cx(None),
        );
        assert_eq!(
            client.model_of(Some("app:a")),
            (ModelChoice::Auto, Effort::Auto)
        );
    }

    #[test]
    fn a_short_drop_is_felt_as_connected_and_a_long_one_as_down() {
        let down = |since: u64| Connection::Offline {
            kind: DropKind::Dropped,
            reason: "lost".into(),
            retry_at_ms: 0,
            since_ms: Some(since),
        };
        assert_eq!(down(1_000).felt(1_000 + QUIET_DROP_MS - 1), Felt::Online);
        assert_eq!(down(1_000).felt(1_000 + QUIET_DROP_MS), Felt::Reconnecting);
        assert!(matches!(
            down(1_000).felt(1_000 + LONG_DROP_MS),
            Felt::Down { .. }
        ));
        let refused = Connection::Offline {
            kind: DropKind::Unauthorized,
            reason: "revoked".into(),
            retry_at_ms: 0,
            since_ms: None,
        };
        assert!(
            matches!(refused.felt(1), Felt::Down { .. }),
            "a refusal is news at once"
        );

        let lost = || {
            Update::Disconnected(crate::machine::Disconnect {
                kind: DropKind::DaemonNotRunning,
                reason: "refused".into(),
                retry_in: std::time::Duration::from_millis(10),
            })
        };
        let mut client = Client::new();
        client.update(lost(), &cx(None));
        assert!(
            matches!(client.connection.felt(1_000), Felt::Down { .. }),
            "a daemon that never answered is news at once"
        );
        client.update(Update::Connected, &cx(None));
        client.update(lost(), &cx(None));
        client.update(Update::Connecting, &cx(None));
        client.update(lost(), &cx(None));
        assert_eq!(
            client.connection.felt(1_000),
            Felt::Online,
            "a fresh drop is quiet"
        );
        assert_eq!(
            client.connection.felt(1_000 + QUIET_DROP_MS),
            Felt::Reconnecting,
            "counted from the first failure, through every retry"
        );
    }

    fn waiting(client: &mut Client, id: &str, session: &str, at: u64, options: &str) {
        client.update(
            frame(&format!(
                r#"{{"type":"event","at_ms":{at},"session_id":"{session}","event":{{"type":"approval_requested","approval":{{"id":"{id}","session_id":"{session}","kind":"command","risk":"medium","summary":"Run a command","options":[{options}],"requested_at_ms":{at}}}}}}}"#
            )),
            &cx(Some(session)),
        );
    }

    fn resolutions(client: &mut Client) -> Vec<(String, ApprovalDecision)> {
        client
            .take_outgoing()
            .into_iter()
            .filter_map(|(_, c)| match c {
                Command::ResolveApproval {
                    approval_id,
                    decision,
                } => Some((approval_id, decision)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn single_keys_answer_the_oldest_waiting_approval_while_the_draft_is_empty() {
        let mut client = Client::new();
        let all = r#""approve_once","approve_session","approve_always","deny""#;
        waiting(&mut client, "ap2", "app:synthetic-1", 9, all);
        waiting(&mut client, "ap1", "app:synthetic-1", 5, all);
        waiting(&mut client, "other", "app:synthetic-2", 1, all);
        client.take_outgoing();

        assert!(client.answers_by_key("app:synthetic-1", ""));
        assert!(!client.answers_by_key("app:synthetic-1", "1"));
        assert!(!client.answers_by_key("app:synthetic-3", ""));
        assert_eq!(
            client.key_approval("app:synthetic-1", "").map(|a| a.id),
            Some("ap1".to_owned())
        );

        // Typing goes on as usual once something is typed.
        assert!(!client.answer_by_key("app:synthetic-1", "3", "1"));
        // A letter is typing, never an answer: "no wait…" stays a message.
        assert!(!client.answer_by_key("app:synthetic-1", "", "n"));
        // A key that answers nothing.
        assert!(!client.answer_by_key("app:synthetic-1", "", "x"));
        assert!(resolutions(&mut client).is_empty());

        for (key, decision) in [
            ("1", ApprovalDecision::ApproveOnce),
            ("2", ApprovalDecision::ApproveSession),
            ("3", ApprovalDecision::Deny),
        ] {
            assert!(client.answer_by_key("app:synthetic-1", "", key));
            assert_eq!(
                resolutions(&mut client),
                vec![("ap1".to_owned(), decision)],
                "{key} answers the oldest approval of this conversation"
            );
        }
    }

    #[test]
    fn a_key_answers_only_with_what_the_approval_offers() {
        let mut client = Client::new();
        waiting(
            &mut client,
            "ap1",
            "app:synthetic-1",
            5,
            r#""approve_once","deny""#,
        );
        client.take_outgoing();
        assert_eq!(client.key_answer("app:synthetic-1", "", "2"), None);
        assert_eq!(
            client.key_answer("app:synthetic-1", "", "3"),
            Some(("ap1".to_owned(), ApprovalDecision::Deny))
        );
    }

    #[test]
    fn the_keys_name_their_answers_both_ways() {
        for (key, decision) in APPROVAL_KEYS {
            assert_eq!(approval_for_key(key), Some(decision));
            assert_eq!(key_for_approval(decision), Some(key));
        }
        assert_eq!(key_for_approval(ApprovalDecision::ApproveAlways), None);
    }

    #[test]
    fn a_sent_message_goes_again_with_its_files_and_waits_while_the_work_runs() {
        let file: AttachmentRef =
            serde_json::from_str(r#"{"id":"f1","name":"plan.pdf"}"#).expect("a file");
        let mut client = Client::new();
        client
            .resend("app:a", " Read the plan ", vec![file.clone()])
            .expect("it goes");
        client
            .resend("app:a", "/model fast", Vec::new())
            .expect("it runs");
        let sent = client.take_outgoing();
        assert!(matches!(
            &sent[0].1,
            Command::SendMessage { session_id: Some(s), text, attachments, mid_run: None, .. }
                if s == "app:a" && text == "Read the plan" && attachments.as_slice() == std::slice::from_ref(&file)
        ));
        assert!(matches!(
            &sent[1].1,
            Command::RunSlash { command, args, .. } if command == "model" && args == "fast"
        ));

        // Edited first: its files are on the draft once, already uploaded.
        client.attach_sent(vec![file.clone()]);
        client.attach_sent(vec![file.clone()]);
        assert_eq!(client.draft_files.len(), 1);
        assert_eq!(client.draft_files[0].state, DraftFileState::Ready(file));

        client.update(
            frame(r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:a","event":{"type":"turn_started","turn_id":"t1","session_id":"app:a","origin":"app"}}"#),
            &cx(Some("app:a")),
        );
        assert!(client.resend("app:a", "Read the plan", Vec::new()).is_err());
        assert!(
            client
                .take_outgoing()
                .iter()
                .all(|(_, c)| !matches!(c, Command::SendMessage { .. }))
        );
    }

    fn reads(client: &mut Client) -> Vec<(String, Command)> {
        client.take_outgoing()
    }

    /// A pull to refresh reads the conversations, what runs and waits, and
    /// the open conversation's history, and spins until those answer.
    #[test]
    fn a_refresh_reads_the_screen_again_and_spins_until_answered() {
        let mut client = Client::new();
        assert!(!client.refresh_all(&cx(Some("app:s1"))), "offline");
        client.update(Update::Connected, &cx(None));
        reads(&mut client);
        assert!(client.refresh_all(&cx(Some("app:s1"))));
        assert!(client.is_refreshing());
        let sent = reads(&mut client);
        for want in [
            "get_snapshot",
            "list_sessions",
            "list_projects",
            "get_history",
        ] {
            assert!(
                sent.iter()
                    .any(|(_, c)| serde_json::to_value(c).unwrap()["type"] == want),
                "{want} in {sent:?}"
            );
        }
        for (id, command) in sent {
            let result = match command {
                Command::GetSnapshot => r#"{"type":"snapshot","seq":1,"state":{"sessions":[],"running_tasks":[],"pending_approvals":[]}}"#.to_owned(),
                Command::GetHistory { .. } => {
                    r#"{"type":"history","session_id":"app:s1","messages":[]}"#.to_owned()
                }
                Command::ListSessions { .. } => r#"{"type":"sessions","sessions":[]}"#.to_owned(),
                Command::ListProjects => r#"{"type":"projects","projects":[]}"#.to_owned(),
                _ => continue,
            };
            client.update(
                frame(&format!(
                    r#"{{"type":"ack","request_id":"{id}","result":{result}}}"#
                )),
                &cx(Some("app:s1")),
            );
        }
        assert!(!client.is_refreshing());
    }

    /// Back after a while away, the app reads everything again; a glance
    /// away (another window for a moment) reads nothing.
    #[test]
    fn coming_back_after_a_while_refreshes_a_moment_does_not() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        reads(&mut client);
        let at = |now_ms| Context {
            now_ms,
            ..Context::default()
        };
        client.went_away(1_000);
        assert!(!client.came_back(&at(5_000)));
        assert!(reads(&mut client).is_empty());
        client.went_away(10_000);
        client.went_away(20_000);
        assert!(client.came_back(&at(10_000 + AWAY_REFRESH_MS)));
        assert!(!reads(&mut client).is_empty());
        assert!(!client.came_back(&at(100_000)), "once per return");
    }

    /// A connection lost while the app was in the background reads as
    /// connected for a moment on return, while the link reconnects.
    #[test]
    fn a_drop_while_away_is_quiet_on_return() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(None));
        client.went_away(1_000);
        client.connection = Connection::Offline {
            kind: DropKind::Dropped,
            reason: "lost".into(),
            retry_at_ms: 0,
            since_ms: Some(2_000),
        };
        let back = 2_000 + LONG_DROP_MS * 4;
        client.came_back(&Context {
            now_ms: back,
            ..Context::default()
        });
        assert_eq!(client.connection.felt(back + 1_000), Felt::Online);
        assert!(matches!(
            client.connection.felt(back + LONG_DROP_MS),
            Felt::Down { .. }
        ));
    }

    /// A reconnect reads the open conversation again even when it shows
    /// lines: they may have missed what was said meanwhile.
    #[test]
    fn a_reconnect_reads_the_open_conversation_again() {
        let mut client = Client::new();
        client.update(Update::Connected, &cx(Some("app:s1")));
        let history = |client: &mut Client| {
            reads(client)
                .into_iter()
                .filter(|(_, c)| matches!(c, Command::GetHistory { .. }))
                .count()
        };
        assert_eq!(history(&mut client), 1);
        client.update(
            frame(
                r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:s1","event":{"type":"user_message","session_id":"app:s1","text":"hi","origin":"app"}}"#,
            ),
            &cx(Some("app:s1")),
        );
        client.update(Update::Connected, &cx(Some("app:s1")));
        assert_eq!(history(&mut client), 1);
    }
}
