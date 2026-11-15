//! The app's state: the shared reducer's [`State`] (built only from the
//! gateway's frames), the connection, what the owner has selected and
//! typed, and each turn's activity as it happened. Pages read it through
//! these methods and change it only through them.

use std::collections::{BTreeMap, BTreeSet};

use aidaemon_client_core::machine::{DropKind, Update, request_id};
use aidaemon_client_core::state::{Line, LiveTurn, SessionView, State, ViewRequest};
use aidaemon_client_protocol::{
    Approval, ApprovalDecision, Command, CommandResult, ErrorCode, Event, Feature, MemoryItem,
    MidRun, ProjectSummary, SearchHit, ServerFrame, SessionChanges, Settings, SettingsChange,
    SettingsOutcome, SetupAction, SetupState, SetupStep, SlashCommand, SystemStatus, TaskSummary,
    TurnStatus,
};
use aidaemon_client_protocol::{AttachmentRef, Pane, ViewAction};
use nsh_core::prelude::*;

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
        /// When the link tries again, on [`nsh_core::time::unix_ms`]'s clock.
        retry_at_ms: u64,
    },
}

/// One thing a turn did, in order: the inspector's timeline.
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
    pub items: Vec<Activity>,
    pub done: bool,
}

/// What the main area shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Page {
    #[default]
    Conversation,
    /// How the daemon itself is doing.
    System,
    /// What the daemon remembers about the owner.
    Memory,
    /// The daemon's settings.
    Settings,
    /// What runs on its own: scheduled work, watching, health checks.
    Automations,
}

/// The Automations page's list, as last read.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AutomationsRead {
    #[default]
    NotYet,
    Loading,
    Ready(Vec<aidaemon_client_protocol::AutomationSummary>),
    Failed(String),
}

/// How much of each turn's tool use the transcript shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivityMode {
    /// None: replies only.
    Hidden,
    /// One line per turn ("Used 4 tools"), opened on demand.
    #[default]
    Collapsed,
    /// Every tool, in order.
    Expanded,
}

impl ActivityMode {
    fn key(self) -> &'static str {
        match self {
            Self::Hidden => "hidden",
            Self::Collapsed => "collapsed",
            Self::Expanded => "expanded",
        }
    }

    fn from_key(key: &str) -> Self {
        match key {
            "hidden" => Self::Hidden,
            "expanded" => Self::Expanded,
            _ => Self::Collapsed,
        }
    }

    /// The next mode, as the palette cycles them.
    pub fn next(self) -> Self {
        match self {
            Self::Hidden => Self::Collapsed,
            Self::Collapsed => Self::Expanded,
            Self::Expanded => Self::Hidden,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Hidden => "hidden",
            Self::Collapsed => "one line per reply",
            Self::Expanded => "every tool",
        }
    }
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

/// The inspector's Code tab: a session's uncommitted changes, as last
/// read.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ChangesRead {
    #[default]
    NotYet,
    Loading,
    Ready(SessionChanges),
    Failed(String),
}

/// The details pane's tabs for a coding conversation, in order; any other
/// conversation shows only its steps.
pub const DETAIL_TABS: [&str; 2] = ["Changes", "Steps"];
/// The Changes tab's index in [`DETAIL_TABS`].
pub const CHANGES_TAB: usize = 0;
/// The Steps tab's index in [`DETAIL_TABS`].
pub const STEPS_TAB: usize = 1;

/// Where a conversation came from, for the sidebar's filter: the app or a
/// chat platform, and whether others are in it.
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

/// Tools whose work can change files: the Code tab reads again after one.
const CHANGING_TOOLS: [&str; 4] = ["edit_file", "write_file", "terminal", "cli_agent"];
/// Tools that write files themselves: the first in a turn opens the Code tab.
const WRITING_TOOLS: [&str; 2] = ["edit_file", "write_file"];

/// The Memory page's data, as last read.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MemoryRead {
    #[default]
    NotYet,
    Loading,
    Ready(Vec<MemoryItem>),
    Failed(String),
}

/// The largest file the gateway takes.
const MAX_FILE_BYTES: usize = 10 * 1024 * 1024;

/// A file attached to the message being written.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftFile {
    pub number: u64,
    pub name: String,
    pub state: DraftFileState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DraftFileState {
    Uploading,
    Ready(AttachmentRef),
    Failed(String),
}

/// The Settings page's data, as last read.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SettingsRead {
    #[default]
    NotYet,
    Loading,
    Ready(Box<Settings>),
    Failed(String),
}

/// The System page's data, as last read.
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

/// Where drafts are kept between runs.
const DRAFT_KEY: &str = "aidaemon.draft.";
const PINS_KEY: &str = "aidaemon.pins";
const OPEN_KEY: &str = "aidaemon.open";
const ACTIVITY_KEY: &str = "aidaemon.activity";

/// Where commands go: the link of this platform.
pub type Outbox = Box<dyn Fn(String, Command)>;

#[derive(Clone, Copy)]
pub struct Store {
    state: Signal<State>,
    connection: Signal<Connection>,
    selected: Signal<Option<String>>,
    drafts: Signal<BTreeMap<String, String>>,
    turns: Signal<BTreeMap<String, TurnLog>>,
    /// The turn the inspector follows: the latest of the selected session.
    inspected: Signal<Option<String>>,
    /// The side panes, which the owner and the agent's view moves open.
    sidebar: Signal<bool>,
    inspector: Signal<bool>,
    /// Whether the inspector shows on a wide window: what the platform
    /// starts with, restored when a narrow window widens again.
    inspector_default: Stored<bool>,
    /// A narrow window (a phone): the side panes open full-screen, one at a
    /// time, and picking a conversation closes them.
    compact: Signal<bool>,
    /// A view move the agent asked for while the owner was typing: offered,
    /// not taken.
    suggestion: Signal<Option<ViewRequest>>,
    /// The approval a view move brought into view, shown highlighted.
    spotlight: Signal<Option<String>>,
    /// What the app last told the owner about a move (or anything else).
    toast: Signal<Option<String>>,
    /// Why the last settings change was refused: stays until dismissed or
    /// the next change, as an error is worth reading.
    settings_error: Signal<Option<String>>,
    outbox: Stored<Option<Outbox>>,
    /// Commands queued before the link was up.
    queued: Stored<Vec<(String, Command)>>,
    /// Sessions whose history was asked for.
    fetched: Stored<BTreeSet<String>>,
    /// The request of a message that starts a new session.
    starting: Stored<Option<String>>,
    next_activity: Stored<u64>,
    page: Signal<Page>,
    system: Signal<SystemRead>,
    /// The request of the System read in flight.
    system_request: Stored<Option<String>>,
    /// The sidebar's search.
    filter: Signal<String>,
    /// Conversations that got a reply the owner has not looked at.
    unread: Signal<BTreeSet<String>>,
    /// Conversations asked to stop, until their turn ends.
    stopping: Signal<BTreeSet<String>>,
    /// The latest turn that finished, for "done" notifications.
    finished: Signal<Option<(String, u64)>>,
    /// Conversations kept at the top of the list (this device's storage).
    pins: Signal<BTreeSet<String>>,
    /// Seconds, for elapsed times (advanced by the shell while work runs).
    tick: Signal<u64>,
    activity_mode: Signal<ActivityMode>,
    /// The daemon's slash commands, for the composer's completions.
    commands: Signal<Vec<SlashCommand>>,
    search: Signal<SearchRead>,
    search_request: Stored<Option<String>>,
    memories: Signal<MemoryRead>,
    memories_request: Stored<Option<String>>,
    /// Files attached to the message being written.
    draft_files: Signal<Vec<DraftFile>>,
    /// Uploads not yet answered: request id to the draft file's number.
    uploads: Stored<BTreeMap<String, u64>>,
    next_upload: Stored<u64>,
    /// The gateway's HTTP origin, before a file's link; empty in the
    /// browser, where links are on the page's own origin.
    gateway_base: Stored<String>,
    /// The Code tab: the selected session's changes, and the session they
    /// are of.
    changes: Signal<ChangesRead>,
    changes_of: Stored<Option<String>>,
    changes_request: Stored<Option<String>>,
    /// Something changed while a read was in flight: read again after it.
    changes_again: Stored<bool>,
    /// The project a new conversation starts in (`None`: no project).
    new_project: Signal<Option<String>>,
    /// The folder the app was started in (the terminal's): new
    /// conversations start in the project of that folder.
    here: Stored<Option<String>>,
    /// Projects shown open in the sidebar.
    open_projects: Signal<BTreeSet<String>>,
    /// The New project form: shown, the request in flight, why it failed.
    project_form: Signal<bool>,
    project_request: Stored<Option<String>>,
    project_error: Signal<Option<String>>,
    /// The project the form edits; `None`: it makes a new one.
    project_editing: Signal<Option<String>>,
    /// The details pane's open tab (an index of [`DETAIL_TABS`]).
    inspector_tab: Signal<usize>,
    /// Turns that already opened the Changes tab, so the owner's own choice
    /// of tab sticks for the rest of a turn.
    code_opened: Stored<BTreeSet<String>>,
    /// Conversations that wrote files or showed code changes: they get
    /// the Changes tab.
    coded: Signal<BTreeSet<String>>,
    /// The sidebar's source filter.
    source: Signal<Source>,
    /// The Automations page: the list, the request in flight, and the
    /// automation a change is pending for.
    automations: Signal<AutomationsRead>,
    automations_request: Stored<Option<String>>,
    automation_busy: Signal<Option<String>>,
    /// Advanced when the keyboard should go back to the composer (a
    /// conversation picked, a pane closed).
    composer_focus: Signal<u64>,
    /// Group conversations the owner chose to post in from here.
    posting: Signal<BTreeSet<String>>,
    settings: Signal<SettingsRead>,
    settings_request: Stored<Option<String>>,
    /// Settings changes sent and not yet answered.
    settings_changes: Stored<BTreeSet<String>>,
    /// Settings changes sent and not yet answered.
    settings_sending: Signal<usize>,
    /// Changes waiting for an approval, and whether another device must
    /// give it.
    settings_waiting: Signal<BTreeMap<String, bool>>,
    /// The setup step sent and not yet answered.
    setup_request: Signal<Option<String>>,
    /// Why the last setup step failed.
    setup_error: Signal<Option<String>>,
    /// Starting the daemon on this computer or pairing with one elsewhere,
    /// before any link: what is happening, or why it stopped.
    starting_daemon: Signal<Option<(String, bool)>>,
    /// Open the next conversation the daemon starts (the greeting).
    open_greeting: Stored<bool>,
}

/// The first approval card says how approvals work, once per device.
const APPROVALS_TAUGHT_KEY: &str = "aidaemon.approvals-taught";

/// The draft key of a conversation not started yet.
const NEW: &str = "";

impl Store {
    pub fn new() -> Self {
        Self {
            state: signal_with(State::default(), SignalEquals::Never),
            connection: signal(Connection::default()),
            selected: signal(None),
            drafts: signal(BTreeMap::new()),
            turns: signal_with(BTreeMap::new(), SignalEquals::Never),
            inspected: signal(None),
            sidebar: signal(true),
            inspector: signal(true),
            inspector_default: stored(true),
            compact: signal(false),
            suggestion: signal(None),
            spotlight: signal(None),
            toast: signal(None),
            settings_error: signal(None),
            outbox: stored(None),
            queued: stored(Vec::new()),
            fetched: stored(BTreeSet::new()),
            starting: stored(None),
            next_activity: stored(1),
            page: signal(Page::Conversation),
            system: signal(SystemRead::NotYet),
            system_request: stored(None),
            filter: signal(String::new()),
            unread: signal(BTreeSet::new()),
            stopping: signal(BTreeSet::new()),
            finished: signal(None),
            pins: signal(BTreeSet::new()),
            tick: signal(0),
            activity_mode: signal(ActivityMode::default()),
            commands: signal(Vec::new()),
            search: signal(SearchRead::Idle),
            search_request: stored(None),
            memories: signal(MemoryRead::NotYet),
            memories_request: stored(None),
            draft_files: signal(Vec::new()),
            uploads: stored(BTreeMap::new()),
            next_upload: stored(1),
            gateway_base: stored(String::new()),
            changes: signal(ChangesRead::NotYet),
            changes_of: stored(None),
            changes_request: stored(None),
            changes_again: stored(false),
            new_project: signal(None),
            here: stored(None),
            open_projects: signal(BTreeSet::new()),
            project_form: signal(false),
            project_request: stored(None),
            project_error: signal(None),
            project_editing: signal(None),
            inspector_tab: signal(STEPS_TAB),
            code_opened: stored(BTreeSet::new()),
            coded: signal(BTreeSet::new()),
            source: signal(Source::All),
            composer_focus: signal(0),
            automations: signal(AutomationsRead::NotYet),
            automations_request: stored(None),
            automation_busy: signal(None),
            posting: signal(BTreeSet::new()),
            settings: signal(SettingsRead::NotYet),
            settings_request: stored(None),
            settings_changes: stored(BTreeSet::new()),
            settings_sending: signal(0),
            settings_waiting: signal(BTreeMap::new()),
            setup_request: signal(None),
            setup_error: signal(None),
            starting_daemon: signal(None),
            open_greeting: stored(false),
        }
    }

    // ----- First-run setup -------------------------------------------------

    /// Setup still to finish, as the daemon last said.
    pub fn setup(&self) -> Option<SetupState> {
        self.state.with(|s| s.setup.clone())
    }

    /// The daemon is in setup mode: only the model step is open.
    fn in_setup_mode(&self) -> bool {
        self.state.with_untracked(|s| {
            s.setup
                .as_ref()
                .is_some_and(|s| matches!(s.step, SetupStep::Model | SetupStep::Starting))
        })
    }

    /// Take a setup step.
    pub fn setup_step(&self, action: SetupAction) {
        if matches!(action, SetupAction::SkipChannel | SetupAction::Finish) {
            self.open_greeting.replace(true);
        }
        self.setup_error.set(None);
        let id = self.send(Command::Setup { action });
        self.setup_request.set(Some(id));
    }

    /// A setup step is waiting for its answer.
    pub fn setup_busy(&self) -> bool {
        self.setup_request.with(Option::is_some)
    }

    pub fn setup_error(&self) -> Option<String> {
        self.setup_error.get()
    }

    /// No daemon to sign in to: show where the agent should run.
    pub fn needs_daemon(&self) {
        self.connection.set(Connection::NoDaemon);
    }

    /// Starting a daemon here or pairing with one: `(what, failed)`.
    pub fn starting_daemon(&self) -> Option<(String, bool)> {
        self.starting_daemon.get()
    }

    pub fn set_starting_daemon(&self, status: Option<(String, bool)>) {
        self.starting_daemon.set(status);
    }

    /// Prompts the daemon offered under the last reply of `session`.
    pub fn suggestions(&self, session: &str) -> Vec<String> {
        self.state.with(|s| {
            s.sessions
                .get(session)
                .map(|v| v.suggestions.clone())
                .unwrap_or_default()
        })
    }

    /// Features the agent asked for in `session`.
    pub fn feature_requests(&self, session: &str) -> Vec<(Feature, String)> {
        self.state.with(|s| {
            s.feature_requests
                .iter()
                .filter(|r| r.session_id == session)
                .map(|r| (r.feature, r.reason.clone()))
                .collect()
        })
    }

    /// Turn `feature` on (saved, then a restart puts it in effect).
    pub fn enable_feature(&self, feature: Feature) {
        let key = match feature {
            Feature::Browser => "browser.enabled",
            Feature::ComputerUse => "computer_use.enabled",
            // Voice needs a local model set up: the agent does that.
            Feature::Voice | Feature::Unknown => {
                self.dismiss_feature(feature);
                return;
            }
        };
        self.change_settings(SettingsChange::Set {
            key: key.into(),
            value: Some("true".into()),
        });
        self.change_settings(SettingsChange::Apply {
            how: aidaemon_client_protocol::ApplyHow::Restart,
        });
        self.dismiss_feature(feature);
    }

    pub fn dismiss_feature(&self, feature: Feature) {
        self.state.update(|s| s.settle_feature(feature));
    }

    /// Whether this device has yet to be told how approvals work; telling
    /// it once is remembered.
    pub fn teach_approvals(&self) -> bool {
        nsh_core::storage::Storage::get_item(APPROVALS_TAUGHT_KEY).is_none()
    }

    pub fn approvals_taught(&self) {
        let _ = nsh_core::storage::Storage::set_item(APPROVALS_TAUGHT_KEY, "1");
    }

    /// Put back what this device had: drafts, pins and the open conversation.
    pub fn restore(&self) {
        self.restore_drafts();
        let pins: BTreeSet<String> = nsh_core::storage::Storage::get_item(PINS_KEY)
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        self.pins.set(pins);
        if let Some(mode) = nsh_core::storage::Storage::get_item(ACTIVITY_KEY) {
            self.activity_mode.set(ActivityMode::from_key(&mode));
        }
        // Started in a folder (a terminal), the app starts on a new
        // conversation there, as a command-line tool would.
        if self.here.with(|h| h.is_some()) {
            return;
        }
        if let Some(open) =
            nsh_core::storage::Storage::get_item(OPEN_KEY).filter(|id| !id.is_empty())
        {
            // Its history loads once the link is up (Connected asks for it).
            self.selected.set(Some(open));
        }
    }

    pub fn is_pinned(&self, id: &str) -> bool {
        self.pins.with(|p| p.contains(id))
    }

    pub fn toggle_pin(&self, id: &str) {
        self.pins.update(|p| {
            if !p.remove(id) {
                p.insert(id.to_owned());
            }
            let text = p.iter().cloned().collect::<Vec<_>>().join("\n");
            let _ = nsh_core::storage::Storage::set_item(PINS_KEY, text);
        });
    }

    /// The selected conversation as Markdown, for Copy conversation.
    pub fn transcript_markdown(&self) -> Option<String> {
        let id = self.selected.get_untracked()?;
        let title = self.title(&id);
        let mut out = format!("# {title}\n");
        for line in self.lines(&id) {
            let (who, text, at) = match line {
                Line::User { text, at_ms, .. } => ("You", text, at_ms),
                Line::Assistant { text, at_ms, .. } => ("AIdaemon", text, at_ms),
                Line::Note { text } => ("Note", text, 0),
                Line::Activity { tools, .. } => {
                    out.push_str(&format!("\n_Used: {}_\n", tools.join(", ")));
                    continue;
                }
            };
            let when = if at > 0 {
                format!(
                    " · {} {}",
                    crate::ui::time::day_label(at, now_ms()),
                    crate::ui::time::clock(at)
                )
            } else {
                String::new()
            };
            out.push_str(&format!("\n**{who}**{when}\n\n{}\n", text.trim()));
        }
        Some(out)
    }

    /// Drafts left from the last run (this device's storage).
    fn restore_drafts(&self) {
        let mut drafts = BTreeMap::new();
        for key in nsh_core::storage::Storage::get_all_keys() {
            if let Some(session) = key.strip_prefix(DRAFT_KEY)
                && let Some(text) = nsh_core::storage::Storage::get_item(&key)
                && !text.trim().is_empty()
            {
                drafts.insert(session.to_owned(), text);
            }
        }
        if !drafts.is_empty() {
            self.drafts.update(|d| d.extend(drafts));
        }
    }

    fn keep_draft(key: &str, text: &str) {
        let key = format!("{DRAFT_KEY}{key}");
        if text.trim().is_empty() {
            nsh_core::storage::Storage::remove_item(&key);
        } else {
            let _ = nsh_core::storage::Storage::set_item(key, text);
        }
    }

    /// The link is up: commands go to `outbox` from now on.
    pub fn attach(&self, outbox: Outbox) {
        for (id, command) in self.queued.replace(Vec::new()) {
            outbox(id, command);
        }
        self.outbox.replace(Some(outbox));
    }

    fn send(&self, command: Command) -> String {
        let id = request_id();
        let sent = self.outbox.with(|outbox| match outbox {
            Some(outbox) => {
                outbox(id.clone(), command.clone());
                true
            }
            None => false,
        });
        if !sent {
            self.queued
                .with(|queued| queued.push((id.clone(), command)));
        }
        id
    }

    // ----- What the link reports -----------------------------------------

    pub fn update(&self, update: Update) {
        match update {
            Update::Connecting => {
                if self
                    .connection
                    .with_untracked(|c| !matches!(c, Connection::Offline { .. }))
                {
                    self.connection.set(Connection::Connecting);
                }
            }
            Update::Connected => {
                self.connection.set(Connection::Online);
                self.send(Command::ListSessions {
                    cursor: None,
                    limit: 200,
                });
                self.send(Command::ListCommands);
                self.send(Command::ListProjects);
                // Today's token use, for the sidebar.
                self.refresh_system();
                // A reconnect may have missed history: ask again on view.
                self.fetched.with(BTreeSet::clear);
                if let Some(id) = self.selected.get_untracked() {
                    self.load_history(&id);
                    // Its changes too: whether it works on code, and what.
                    self.changes_of.replace(None);
                    self.refresh_changes();
                }
            }
            Update::Disconnected(disconnect) => {
                // Changes in flight get no answer now.
                self.settings_changes.with(BTreeSet::clear);
                self.settings_sending.set(0);
                let retry = u64::try_from(disconnect.retry_in.as_millis()).unwrap_or(0);
                self.connection.set(Connection::Offline {
                    kind: disconnect.kind,
                    reason: disconnect.reason,
                    retry_at_ms: now_ms().saturating_add(retry),
                });
            }
            Update::Frame(frame) => {
                self.frame(frame);
                self.settle_here();
            }
        }
    }

    fn frame(&self, frame: ServerFrame) {
        match &frame {
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Uploaded { attachment },
            } => {
                if let Some(number) = self.uploads.with(|u| u.remove(request_id)) {
                    let attachment = attachment.clone();
                    self.draft_files.update(|files| {
                        if let Some(file) = files.iter_mut().find(|f| f.number == number) {
                            file.state = DraftFileState::Ready(attachment);
                        }
                    });
                }
                return;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self.uploads.with(|u| u.contains_key(request_id)) => {
                let number = self.uploads.with(|u| u.remove(request_id)).unwrap_or(0);
                let reason = error.message.clone();
                self.draft_files.update(|files| {
                    if let Some(file) = files.iter_mut().find(|f| f.number == number) {
                        file.state = DraftFileState::Failed(reason.clone());
                    }
                });
                self.say(format!("Could not attach the file: {}", error.message));
                return;
            }
            _ => {}
        }
        if let ServerFrame::Ack {
            request_id,
            result: CommandResult::MessageAccepted { session_id, .. },
        } = &frame
            && self
                .starting
                .with(|s| s.as_deref() == Some(request_id.as_str()))
        {
            self.starting.replace(None);
            // The draft was sent: the new conversation starts empty.
            Self::keep_draft(NEW, "");
            self.drafts.update(|d| {
                d.remove(NEW);
            });
            self.fetched.with(|f| f.insert(session_id.clone()));
            self.selected.set(Some(session_id.clone()));
        }
        match &frame {
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Setup { .. },
            } if self
                .setup_request
                .with_untracked(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.setup_request.set(None);
                self.setup_error.set(None);
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .setup_request
                .with_untracked(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.setup_request.set(None);
                self.setup_error.set(Some(error.message.clone()));
                return;
            }
            // In setup mode the usual reads on connecting are refused.
            ServerFrame::Error { error, .. }
                if error.code == ErrorCode::Forbidden && self.in_setup_mode() =>
            {
                return;
            }
            ServerFrame::Event {
                event: Event::SessionCreated { session },
                ..
            } if self.open_greeting.with(|o| *o) => {
                self.open_greeting.replace(false);
                self.selected.set(Some(session.id.clone()));
                self.page.set(Page::Conversation);
            }
            // The chat step heard from the owner: read where it stands.
            ServerFrame::Event {
                event: Event::UserMessage { origin, .. },
                ..
            } if origin != "app"
                && self.state.with_untracked(|s| {
                    s.setup
                        .as_ref()
                        .is_some_and(|s| s.step == SetupStep::Channel)
                }) =>
            {
                self.send(Command::GetSetup);
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::System { system },
            } if self
                .system_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.system_request.replace(None);
                self.system.set(SystemRead::Ready {
                    status: Box::new(system.clone()),
                    at_ms: now_ms(),
                });
                return;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .system_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.system_request.replace(None);
                self.system.set(SystemRead::Failed(error.message.clone()));
                return;
            }
            // Only this client's own list or change is answered with it.
            ServerFrame::Ack {
                result:
                    CommandResult::Automations {
                        automations, note, ..
                    },
                ..
            } => {
                self.automations_request.replace(None);
                self.automation_busy.set(None);
                self.automations
                    .set(AutomationsRead::Ready(automations.clone()));
                if let Some(note) = note {
                    self.say(note.clone());
                }
                return;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .automations_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.automations_request.replace(None);
                if self.automation_busy.get_untracked().is_some() {
                    self.automation_busy.set(None);
                    self.say(error.message.clone());
                } else {
                    self.automations
                        .set(AutomationsRead::Failed(error.message.clone()));
                }
                return;
            }
            ServerFrame::Ack {
                result: CommandResult::Commands { commands },
                ..
            } => {
                self.commands.set(commands.clone());
                return;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::SearchResults { hits },
            } if self
                .search_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.search_request.replace(None);
                if let SearchRead::Loading(query) = self.search.get_untracked() {
                    self.search.set(SearchRead::Ready {
                        query,
                        hits: hits.clone(),
                    });
                }
                return;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Project { project },
            } if self
                .project_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.project_request.replace(None);
                self.project_form.set(false);
                self.project_error.set(None);
                if self.project_editing.get_untracked().is_some() {
                    self.project_editing.set(None);
                    self.say(format!("Project {} saved", project.name));
                    return;
                }
                let id = project.id.clone();
                self.open_projects.update(|open| {
                    open.insert(id.clone());
                });
                self.say(format!("Project {} created", project.name));
                // A new project opens on a new conversation in it.
                self.new_in_project(Some(id));
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .project_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.project_request.replace(None);
                self.project_error.set(Some(error.message.clone()));
                return;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Changes { changes },
            } if self
                .changes_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.changes_request.replace(None);
                // Changes found: the conversation is a coding one, and one
                // just found to be opens on them.
                if changes.repos.iter().any(|r| !r.files.is_empty())
                    && let Some(session) = self.changes_of.with(|of| of.clone())
                    && !self.is_coding(&session)
                {
                    self.mark_coded(&session);
                    if self.selected.get_untracked().as_deref() == Some(session.as_str()) {
                        self.inspector_tab.set(CHANGES_TAB);
                    }
                }
                self.changes.set(ChangesRead::Ready(changes.clone()));
                if self.changes_again.replace(false) {
                    self.refresh_changes();
                }
                return;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .changes_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.changes_request.replace(None);
                self.changes_again.replace(false);
                self.changes.set(ChangesRead::Failed(error.message.clone()));
                return;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Memories { memories },
            } if self
                .memories_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.memories_request.replace(None);
                self.memories.set(MemoryRead::Ready(memories.clone()));
                return;
            }
            ServerFrame::Ack {
                result: CommandResult::Forgotten { memory_id },
                ..
            } => {
                let id = memory_id.clone();
                self.memories.update(|m| {
                    if let MemoryRead::Ready(items) = m {
                        items.retain(|item| item.id != id);
                    }
                });
                self.say("Forgotten");
                return;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .search_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.search_request.replace(None);
                self.search.set(SearchRead::Failed(error.message.clone()));
                return;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .memories_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.memories_request.replace(None);
                self.memories.set(MemoryRead::Failed(error.message.clone()));
                return;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::Settings { settings },
            } if self
                .settings_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.settings_request.replace(None);
                self.settings
                    .set(SettingsRead::Ready(Box::new(settings.clone())));
                return;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self
                .settings_request
                .with(|r| r.as_deref() == Some(request_id.as_str())) =>
            {
                self.settings_request.replace(None);
                self.settings
                    .set(SettingsRead::Failed(error.message.clone()));
                return;
            }
            ServerFrame::Ack {
                request_id,
                result: CommandResult::SettingsChanged { outcome },
            } => {
                if self.settings_changes.with(|c| c.remove(request_id)) {
                    self.settings_sending.update(|n| *n = n.saturating_sub(1));
                }
                match outcome {
                    // The daemon announces what it did to every client
                    // (`settings_updated`): that note says it here too.
                    SettingsOutcome::Done { .. } => {}
                    SettingsOutcome::AwaitingApproval {
                        approval_id,
                        other_device,
                    } => {
                        let other = *other_device;
                        let id = approval_id.clone();
                        self.settings_waiting.update(|w| {
                            w.insert(id, other);
                        });
                        self.say(if other {
                            "Approve it on another device, or on the daemon's computer"
                        } else {
                            "Waiting for your approval"
                        });
                    }
                    SettingsOutcome::Unknown => {}
                }
                return;
            }
            ServerFrame::Error {
                request_id: Some(request_id),
                error,
            } if self.settings_changes.with(|c| c.remove(request_id)) => {
                self.settings_sending.update(|n| *n = n.saturating_sub(1));
                self.settings_error.set(Some(error.message.clone()));
                return;
            }
            ServerFrame::Event {
                event: Event::SettingsUpdated { message, .. },
                ..
            } => {
                self.say(message.clone());
                if matches!(self.settings.get_untracked(), SettingsRead::Ready(_))
                    || self.page.get_untracked() == Page::Settings
                {
                    self.refresh_settings();
                }
                return;
            }
            ServerFrame::Event {
                event: Event::ApprovalResolved { approval_id, .. },
                ..
            } if self
                .settings_waiting
                .with_untracked(|w| w.contains_key(approval_id)) =>
            {
                let id = approval_id.clone();
                self.settings_waiting.update(|w| {
                    w.remove(&id);
                });
            }
            ServerFrame::Event {
                event, session_id, ..
            } => {
                self.note_activity(event);
                self.note_code(event);
                self.note_reply(session_id.as_deref(), event);
            }
            // After a reload or a dropped connection: what running turns
            // have got to, so the inspector's "This turn" is not empty.
            ServerFrame::Snapshot { state, .. }
            | ServerFrame::Ack {
                result: CommandResult::Snapshot { state, .. },
                ..
            } => self.note_live_turns(&state.live_turns),
            _ => {}
        }
        let mut request = None;
        self.state.update(|state| {
            state.apply(frame);
            request = state.take_view_request();
        });
        if let Some(request) = request {
            self.view_request(request);
        }
    }

    /// A reply landed: unread unless the owner is looking at it; a stop
    /// asked for is over once its turn ends.
    fn note_reply(&self, session: Option<&str>, event: &Event) {
        let (session, done) = match event {
            Event::AssistantMessage { session_id, .. } => (session_id.as_str(), false),
            Event::TurnCompleted { .. } => match session {
                Some(session) => (session, true),
                None => return,
            },
            _ => return,
        };
        let looking = self.page.get_untracked() == Page::Conversation
            && self.selected.get_untracked().as_deref() == Some(session)
            && nsh_core::element::lifecycle() == nsh_core::platform::Lifecycle::Active;
        if !looking {
            self.unread.update(|u| {
                u.insert(session.to_owned());
            });
        }
        if done {
            self.stopping.update(|s| {
                s.remove(session);
            });
            self.finished.set(Some((session.to_owned(), now_ms())));
        }
    }

    // ----- The System page ---------------------------------------------------

    pub fn page(&self) -> Page {
        self.page.get()
    }

    pub fn open_system(&self) {
        crate::ui::motion::next_layout();
        self.close_panes();
        self.page.set(Page::System);
        self.refresh_system();
    }

    // ----- The Automations page -------------------------------------------

    pub fn open_automations(&self) {
        crate::ui::motion::next_layout();
        self.close_panes();
        self.page.set(Page::Automations);
        self.refresh_automations();
    }

    /// Read the automations again (on open, Refresh, and the page's timer).
    pub fn refresh_automations(&self) {
        if self.automations_request.with(|r| r.is_some()) {
            return;
        }
        if !matches!(self.automations.get_untracked(), AutomationsRead::Ready(_)) {
            self.automations.set(AutomationsRead::Loading);
        }
        let id = self.send(Command::ListAutomations);
        self.automations_request.replace(Some(id));
    }

    pub fn automations(&self) -> AutomationsRead {
        self.automations.get()
    }

    /// The automation a change was sent for and not answered yet.
    pub fn automation_busy(&self) -> Option<String> {
        self.automation_busy.get()
    }

    /// Pause, resume, cancel or run one automation now; the answer brings
    /// the list as it is after, and a note of what happened.
    pub fn change_automation(&self, id: &str, change: aidaemon_client_protocol::AutomationChange) {
        if self.automations_request.with(|r| r.is_some()) {
            return;
        }
        self.automation_busy.set(Some(id.to_owned()));
        let request = self.send(Command::ChangeAutomation {
            automation_id: id.to_owned(),
            change,
        });
        self.automations_request.replace(Some(request));
    }

    /// Read the daemon's status again (the page's Refresh, and its timer).
    pub fn refresh_system(&self) {
        if self.system_request.with(|r| r.is_some()) {
            return;
        }
        if !matches!(self.system.get_untracked(), SystemRead::Ready { .. }) {
            self.system.set(SystemRead::Loading);
        }
        let id = self.send(Command::GetSystem);
        self.system_request.replace(Some(id));
    }

    pub fn system(&self) -> SystemRead {
        self.system.get()
    }

    /// Tokens used today (the owner's day), when the System read has them.
    pub fn today_tokens(&self) -> Option<u64> {
        let SystemRead::Ready { status, .. } = self.system.get() else {
            return None;
        };
        // Usage days are UTC dates; the latest one is today's or nothing.
        let today = crate::ui::time::utc_date(now_ms());
        Some(
            status
                .usage
                .iter()
                .filter(|d| d.day == today)
                .map(|d| d.input_tokens + d.output_tokens)
                .sum(),
        )
    }

    // ----- The Memory page ---------------------------------------------------

    pub fn open_memory(&self) {
        crate::ui::motion::next_layout();
        self.close_panes();
        self.page.set(Page::Memory);
        self.refresh_memories();
    }

    pub fn refresh_memories(&self) {
        if self.memories_request.with(|r| r.is_some()) {
            return;
        }
        if !matches!(self.memories.get_untracked(), MemoryRead::Ready(_)) {
            self.memories.set(MemoryRead::Loading);
        }
        let id = self.send(Command::ListMemories);
        self.memories_request.replace(Some(id));
    }

    pub fn memories(&self) -> MemoryRead {
        self.memories.get()
    }

    pub fn forget_memory(&self, id: &str) {
        self.send(Command::ForgetMemory {
            memory_id: id.to_owned(),
        });
    }

    // ----- Projects -----------------------------------------------------------

    /// The owner's projects, by name.
    pub fn projects(&self) -> Vec<ProjectSummary> {
        let mut projects: Vec<ProjectSummary> =
            self.state.with(|s| s.projects.values().cloned().collect());
        projects.sort_by_key(|p| p.name.to_lowercase());
        projects
    }

    pub fn project(&self, id: &str) -> Option<ProjectSummary> {
        self.state.with(|s| s.projects.get(id).cloned())
    }

    /// The project a conversation is in.
    pub fn session_project(&self, session: &str) -> Option<String> {
        self.state.with(|s| {
            s.sessions
                .get(session)
                .and_then(|v| v.summary.as_ref())
                .and_then(|summary| summary.project_id.clone())
        })
    }

    /// The project of what is on screen: the selected conversation's, or
    /// the one a new conversation will start in.
    pub fn current_project(&self) -> Option<String> {
        match self.selected.get() {
            Some(id) => self.session_project(&id),
            None => self.new_project.get(),
        }
    }

    /// The app was started in `folder`: new conversations start in its
    /// project.
    pub fn set_here(&self, folder: String) {
        self.here.replace(Some(folder));
    }

    /// The project whose folder holds the folder the app was started in
    /// (the innermost).
    pub fn here_project(&self) -> Option<String> {
        let here = self.here.with(|h| h.clone())?;
        let here = here.trim_end_matches('/');
        untracked(|| self.projects())
            .into_iter()
            .filter_map(|p| {
                let path = p.path.as_deref()?.trim_end_matches('/').to_owned();
                (here == path || here.starts_with(&format!("{path}/")))
                    .then_some((path.len(), p.id))
            })
            .max_by_key(|(len, _)| *len)
            .map(|(_, id)| id)
    }

    /// Once the projects are known, a new conversation on screen starts in
    /// the project of the folder the app was started in.
    fn settle_here(&self) {
        if self.selected.get_untracked().is_none()
            && self.new_project.get_untracked().is_none()
            && let Some(project) = self.here_project()
        {
            self.new_project.set(Some(project));
        }
    }

    /// Start a new conversation in `project` (`None`: in no project).
    pub fn new_in_project(&self, project: Option<String>) {
        self.select(None);
        self.new_project.set(project);
    }

    /// Choose the project the new conversation on screen starts in.
    /// Where a new conversation works; then typing goes on in the
    /// composer, not the picker that just closed (live 2026-11-15: in the
    /// terminal, a project picked left no way to type the message).
    pub fn set_new_project(&self, project: Option<String>) {
        self.new_project.set(project);
        self.focus_composer();
    }

    pub fn project_open(&self, id: &str) -> bool {
        self.open_projects.with(|open| open.contains(id))
    }

    pub fn toggle_project(&self, id: &str) {
        crate::ui::motion::next_unfold();
        let id = id.to_owned();
        self.open_projects.update(|open| {
            if !open.remove(&id) {
                open.insert(id);
            }
        });
    }

    pub fn project_form_open(&self) -> bool {
        self.project_form.get()
    }

    pub fn show_project_form(&self, show: bool) {
        crate::ui::motion::next_unfold();
        self.project_form.set(show);
        self.project_error.set(None);
        self.project_editing.set(None);
    }

    /// Open the project form on project `id`, to rename it or change its
    /// folder.
    pub fn edit_project(&self, id: &str) {
        crate::ui::motion::next_unfold();
        self.project_error.set(None);
        self.project_editing.set(Some(id.to_owned()));
        self.project_form.set(true);
    }

    /// The project the form edits; `None` when it makes a new one.
    pub fn project_editing(&self) -> Option<String> {
        self.project_editing.get()
    }

    /// Save the form: a new project, or the edited one's name and folder
    /// (an empty folder removes it).
    pub fn save_project(&self, name: &str, path: &str) {
        let Some(id) = self.project_editing.get_untracked() else {
            self.create_project(name, path);
            return;
        };
        let name = name.trim();
        if name.is_empty() {
            self.project_error
                .set(Some("Give the project a name".to_owned()));
            return;
        }
        let request = self.send(Command::UpdateProject {
            project_id: id,
            name: Some(name.to_owned()),
            path: Some(path.trim().to_owned()),
        });
        self.project_request.replace(Some(request));
    }

    pub fn project_error(&self) -> Option<String> {
        self.project_error.get()
    }

    pub fn creating_project(&self) -> bool {
        self.project_request.with(|r| r.is_some())
    }

    /// Make a project; it opens on a new conversation once made.
    pub fn create_project(&self, name: &str, path: &str) {
        let name = name.trim();
        if name.is_empty() {
            self.project_error
                .set(Some("Give the project a name".to_owned()));
            return;
        }
        let path = path.trim();
        let id = self.send(Command::CreateProject {
            name: name.to_owned(),
            path: (!path.is_empty()).then(|| path.to_owned()),
        });
        self.project_request.replace(Some(id));
    }

    /// Put a conversation in a project, or in none.
    pub fn move_to_project(&self, session: &str, project: Option<String>) {
        if let Some(id) = &project {
            self.open_projects.update(|open| {
                open.insert(id.clone());
            });
        }
        self.send(Command::SetSessionProject {
            session_id: session.to_owned(),
            project_id: project,
        });
    }

    // ----- Chat bots ----------------------------------------------------------

    /// The chat bots with conversations here, by name.
    pub fn bots(&self) -> Vec<Bot> {
        let projects = self.projects();
        let mut bots: BTreeMap<String, usize> = BTreeMap::new();
        self.state.with(|s| {
            for view in s.sessions.values() {
                if let Some(summary) = &view.summary
                    && platform_of(&summary.channel, &summary.id) != Source::App
                    && platform_of(&summary.channel, &summary.id) != Source::Chat
                {
                    *bots.entry(summary.channel.clone()).or_default() += 1;
                }
            }
        });
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
    pub fn set_bot_project(&self, channel: &str, project: Option<String>) {
        self.send(Command::SetChannelProject {
            channel: channel.to_owned(),
            project_id: project,
        });
    }

    // ----- The details pane: Changes and Steps ---------------------------------

    /// The details pane's open tab, for its tab bar.
    pub fn inspector_tab(&self) -> Signal<usize> {
        self.inspector_tab
    }

    /// Whether conversation `id` works on code: it is in a project with a
    /// folder, or it wrote files or showed changes.
    pub fn is_coding(&self, id: &str) -> bool {
        self.coded.with(|c| c.contains(id))
            || self
                .session_project(id)
                .and_then(|p| self.project(&p))
                .is_some_and(|p| p.path.is_some())
    }

    /// Whether what is on screen (the selected conversation, or a new one
    /// in a project with a folder) works on code.
    pub fn coding_on_screen(&self) -> bool {
        match self.selected.get() {
            Some(id) => self.is_coding(&id),
            None => false,
        }
    }

    /// The Changes tab is the open one (and the conversation has it).
    pub fn changes_shown(&self) -> bool {
        self.coding_on_screen() && self.inspector_tab.get() == CHANGES_TAB
    }

    /// Read the selected session's changes again; one read at a time, and
    /// one more after it when asked meanwhile.
    pub fn refresh_changes(&self) {
        let Some(session_id) = self.selected.get_untracked() else {
            self.changes_of.replace(None);
            self.changes.set(ChangesRead::NotYet);
            return;
        };
        if self.changes_request.with(|r| r.is_some()) {
            self.changes_again.replace(true);
            return;
        }
        let same = self
            .changes_of
            .with(|of| of.as_deref() == Some(session_id.as_str()));
        if !same || !matches!(self.changes.get_untracked(), ChangesRead::Ready(_)) {
            self.changes.set(ChangesRead::Loading);
        }
        self.changes_of.replace(Some(session_id.clone()));
        let id = self.send(Command::GetChanges { session_id });
        self.changes_request.replace(Some(id));
    }

    /// Read the selected session's changes unless those shown are its own
    /// and current.
    pub fn refresh_changes_if_stale(&self) {
        let selected = self.selected.get_untracked();
        let stale = self.changes_of.with(|of| of.as_ref() != selected.as_ref());
        if stale {
            self.refresh_changes();
        }
    }

    /// The selected session's changes, as last read.
    pub fn changes(&self) -> ChangesRead {
        self.changes.get()
    }

    /// How many files the conversation on screen has changed, once read.
    pub fn changed_files(&self) -> Option<usize> {
        let selected = self.selected.get()?;
        if self
            .changes_of
            .with(|of| of.as_deref() != Some(selected.as_str()))
        {
            return None;
        }
        match self.changes.get() {
            ChangesRead::Ready(changes) => Some(changes.repos.iter().map(|r| r.files.len()).sum()),
            _ => None,
        }
    }

    fn mark_coded(&self, session: &str) {
        if self.coded.with_untracked(|c| !c.contains(session)) {
            self.coded.update(|c| {
                c.insert(session.to_owned());
            });
            if self.selected.get_untracked().as_deref() == Some(session) {
                self.show_details(true);
            }
        }
    }

    /// Show or hide the details beside the conversation, on a wide window
    /// of a platform that shows them by default (not the terminal).
    fn show_details(&self, show: bool) {
        if !self.compact.get_untracked() && self.inspector_default.get() {
            self.inspector.set(show);
        }
    }

    /// The Changes tab is on screen.
    fn code_visible(&self) -> bool {
        self.inspector.get_untracked()
            && self.inspector_tab.get_untracked() == CHANGES_TAB
            && self.page.get_untracked() == Page::Conversation
    }

    /// A tool that can change files finished in the selected session: the
    /// Code tab reads again, and the turn's first file write opens it.
    fn note_code(&self, event: &Event) {
        let (turn_id, name) = match event {
            Event::TurnStatus {
                turn_id,
                status: TurnStatus::ToolCompleted { name, .. },
            } => (turn_id, Some(name.as_str())),
            Event::TurnCompleted { turn_id, .. } => (turn_id, None),
            _ => return,
        };
        let session = self
            .turns
            .with_untracked(|t| t.get(turn_id).map(|log| log.session_id.clone()));
        if session.is_none() || session != self.selected.get_untracked() {
            return;
        }
        if let Some(name) = name {
            if !CHANGING_TOOLS.contains(&name) {
                return;
            }
            if WRITING_TOOLS.contains(&name) && self.code_opened.with(|o| !o.contains(turn_id)) {
                self.code_opened.with(|o| {
                    o.insert(turn_id.clone());
                });
                if let Some(session) = &session {
                    self.mark_coded(session);
                }
                self.inspector_tab.set(CHANGES_TAB);
            }
        }
        if self.code_visible() {
            self.refresh_changes();
        } else {
            // Read on next showing.
            self.changes_of.replace(None);
        }
    }

    // ----- The Settings page -------------------------------------------------

    pub fn open_settings(&self) {
        crate::ui::motion::next_layout();
        self.close_panes();
        self.page.set(Page::Settings);
        self.refresh_settings();
    }

    pub fn refresh_settings(&self) {
        if self.settings_request.with(|r| r.is_some()) {
            return;
        }
        if !matches!(self.settings.get_untracked(), SettingsRead::Ready(_)) {
            self.settings.set(SettingsRead::Loading);
        }
        let id = self.send(Command::GetSettings);
        self.settings_request.replace(Some(id));
    }

    pub fn settings(&self) -> SettingsRead {
        self.settings.get()
    }

    /// Ask the daemon to make `change`; the answer arrives as a note, and
    /// the page reads the settings again when it lands.
    pub fn change_settings(&self, change: SettingsChange) {
        self.settings_error.set(None);
        let id = self.send(Command::ChangeSettings { change });
        self.settings_changes.with(|c| c.insert(id));
        self.settings_sending.update(|n| *n += 1);
    }

    /// A settings change was sent and has no answer yet.
    pub fn settings_busy(&self) -> bool {
        self.settings_sending.get() > 0
    }

    /// Why the last settings change was refused, until dismissed.
    pub fn settings_error(&self) -> Option<String> {
        self.settings_error.get()
    }

    pub fn dismiss_settings_error(&self) {
        self.settings_error.set(None);
    }

    /// Settings changes waiting for an approval.
    pub fn settings_waiting(&self) -> usize {
        self.settings_waiting.with(BTreeMap::len)
    }

    // ----- Search of every conversation ------------------------------------

    /// Ask the daemon for messages matching `query`, in every conversation.
    pub fn search_messages(&self, query: &str) {
        let query = query.trim().to_owned();
        if query.chars().count() < 2 {
            self.search_request.replace(None);
            self.search.set(SearchRead::Idle);
            return;
        }
        self.search.set(SearchRead::Loading(query.clone()));
        let id = self.send(Command::SearchMessages { query, limit: 30 });
        self.search_request.replace(Some(id));
    }

    pub fn search(&self) -> SearchRead {
        self.search.get()
    }

    // ----- How replies show their work --------------------------------------

    pub fn activity_mode(&self) -> ActivityMode {
        self.activity_mode.get()
    }

    pub fn cycle_activity_mode(&self) {
        let next = self.activity_mode.get_untracked().next();
        self.activity_mode.set(next);
        let _ = nsh_core::storage::Storage::set_item(ACTIVITY_KEY, next.key());
        self.say(format!("Tool activity: {}", next.label()));
    }

    // ----- Composer helpers -------------------------------------------------

    /// Slash commands completing the draft: it starts with `/` and has no
    /// space yet. At most eight, in the daemon's order.
    pub fn slash_matches(&self) -> Vec<SlashCommand> {
        let draft = self.draft();
        let Some(typed) = draft.strip_prefix('/') else {
            return Vec::new();
        };
        if typed.contains(char::is_whitespace) {
            return Vec::new();
        }
        let typed = typed.to_lowercase();
        self.commands.with(|commands| {
            commands
                .iter()
                .filter(|c| c.name.starts_with(&typed) && c.name != typed)
                .take(8)
                .cloned()
                .collect()
        })
    }

    /// Put `/name ` in the composer.
    pub fn complete_slash(&self, name: &str) {
        self.set_draft(format!("/{name} "));
    }

    /// The last thing the owner said in `session`.
    pub fn last_user_text(&self, session: &str) -> Option<String> {
        self.lines(session)
            .into_iter()
            .rev()
            .find_map(|line| match line {
                Line::User { text, .. } => Some(text),
                _ => None,
            })
    }

    /// With an empty composer, bring back the last message sent here.
    pub fn recall_last(&self) -> bool {
        if !self.draft().is_empty() {
            return false;
        }
        let Some(session) = self.selected.get_untracked() else {
            return false;
        };
        match self.last_user_text(&session) {
            Some(text) => {
                self.set_draft(text);
                true
            }
            None => false,
        }
    }

    /// Whether the last turn of `session` failed or was stopped: Retry
    /// is offered under it.
    pub fn can_retry(&self, session: &str) -> bool {
        !self.is_running(session)
            && self.state.with(|s| {
                s.sessions.get(session).is_some_and(|v| {
                    matches!(v.lines.last(), Some(Line::Note { text }) if text == "Stopped." || text.starts_with("Failed:"))
                })
            })
    }

    /// Send the last message of `session` again.
    pub fn retry(&self, session: &str) {
        if let Some(text) = self.last_user_text(session) {
            self.send(Command::SendMessage {
                session_id: Some(session.to_owned()),
                text,
                attachments: Vec::new(),
                mid_run: None,
                project_id: None,
            });
        }
    }

    /// Say `text` in `session`, as if typed (a suggested prompt pressed).
    pub fn ask(&self, session: &str, text: String) {
        self.send(Command::SendMessage {
            session_id: Some(session.to_owned()),
            text,
            attachments: Vec::new(),
            mid_run: None,
            project_id: None,
        });
    }

    /// Conversations with an approval waiting.
    pub fn needs_you(&self, id: &str) -> bool {
        self.state
            .with(|s| s.approvals.values().any(|a| a.session_id == id))
    }

    // ----- Where conversations come from ----------------------------------------

    /// Where conversation `id` came from.
    pub fn source(&self, id: &str) -> Source {
        platform_of(&self.channel(id).unwrap_or_default(), id)
    }

    /// Other people are in conversation `id`: what is sent there from here,
    /// they see.
    pub fn is_group(&self, id: &str) -> bool {
        self.state.with(|s| {
            s.sessions
                .get(id)
                .and_then(|v| v.summary.as_ref())
                .is_some_and(|summary| summary.group)
        })
    }

    /// Whether the app may send to conversation `id` now: anywhere but a
    /// group the owner has not chosen to post in.
    pub fn may_post(&self, id: &str) -> bool {
        !self.is_group(id) || self.posting.with(|p| p.contains(id))
    }

    /// The owner chose to post in group conversation `id` from here.
    pub fn allow_posting(&self, id: &str) {
        self.posting.update(|p| {
            p.insert(id.to_owned());
        });
    }

    /// The sidebar's source filter.
    pub fn source_filter(&self) -> Source {
        self.source.get()
    }

    pub fn set_source_filter(&self, source: Source) {
        self.source.set(source);
    }

    /// The filters worth offering: every source with a conversation, and
    /// groups when there are any; nothing when all come from one place.
    pub fn sources(&self) -> Vec<Source> {
        let ids = self.session_ids();
        let mut found: BTreeSet<Source> = ids.iter().map(|id| self.source(id)).collect();
        if found.len() < 2 {
            return Vec::new();
        }
        if ids.iter().any(|id| self.is_group(id)) {
            found.insert(Source::Groups);
        }
        std::iter::once(Source::All).chain(found).collect()
    }

    /// Whether conversation `id` passes the source filter.
    pub fn in_source(&self, id: &str) -> bool {
        match self.source.get() {
            Source::All => true,
            Source::Groups => self.is_group(id),
            source => self.source(id) == source,
        }
    }

    // ----- Search, unread, stopping ------------------------------------------

    pub fn filter(&self) -> String {
        self.filter.get()
    }

    pub fn set_filter(&self, text: String) {
        self.filter.set(text);
    }

    /// Session ids matching the sidebar's search (title, place or any loaded
    /// message), newest first.
    pub fn visible_ids(&self) -> Vec<String> {
        let query = self.filter.get().trim().to_lowercase();
        let mut ids = self.session_ids();
        // Pinned first, each part keeping newest first.
        ids.sort_by_key(|id| !self.is_pinned(id));
        if query.is_empty() {
            return ids;
        }
        let words: Vec<&str> = query.split_whitespace().collect();
        ids.into_iter()
            .filter(|id| {
                let mut hay = format!("{} {} {id}", self.title(id), self.place(id)).to_lowercase();
                for line in self.lines(id) {
                    if let Line::User { text, .. } | Line::Assistant { text, .. } = line {
                        hay.push(' ');
                        hay.push_str(&text.to_lowercase());
                    }
                }
                words.iter().all(|w| hay.contains(w))
            })
            .collect()
    }

    pub fn is_unread(&self, id: &str) -> bool {
        self.unread.with(|u| u.contains(id))
    }

    pub fn unread_count(&self) -> usize {
        self.unread.with(BTreeSet::len)
    }

    pub fn is_stopping(&self, id: &str) -> bool {
        self.stopping.with(|s| s.contains(id)) && self.is_running(id)
    }

    /// The latest finished turn: its session and when.
    pub fn finished(&self) -> Option<(String, u64)> {
        self.finished.get()
    }

    /// When a conversation last had a message.
    pub fn last_activity(&self, id: &str) -> Option<u64> {
        self.state.with(|s| {
            let view = s.sessions.get(id)?;
            let line = view.lines.iter().rev().find_map(|line| match line {
                Line::User { at_ms, .. } | Line::Assistant { at_ms, .. } if *at_ms > 0 => {
                    Some(*at_ms)
                }
                _ => None,
            });
            let summary = view.summary.as_ref().and_then(|m| m.last_activity_ms);
            line.max(summary)
        })
    }

    /// The selected conversation's last reply (Copy reply).
    pub fn last_reply(&self) -> Option<String> {
        let id = self.selected.get_untracked()?;
        self.state.with_untracked(|s| {
            s.sessions
                .get(&id)?
                .lines
                .iter()
                .rev()
                .find_map(|line| match line {
                    Line::Assistant { text, .. } => Some(text.clone()),
                    _ => None,
                })
        })
    }

    // ----- The agent's view moves ------------------------------------------

    /// A move the agent asked for: taken now, or offered while the owner
    /// has a draft (a move never sends anything but the reads it needs).
    fn view_request(&self, request: ViewRequest) {
        if !self.draft_untracked().trim().is_empty() {
            self.suggestion.set(Some(request));
        } else {
            self.suggestion.set(None);
            self.move_view(request);
        }
    }

    /// The move offered, in words.
    pub fn suggestion(&self) -> Option<String> {
        let request = self.suggestion.get()?;
        let reason = if request.reason.trim().is_empty() {
            String::new()
        } else {
            format!(" ({})", request.reason.trim())
        };
        Some(format!(
            "The assistant suggests {}{reason}.",
            self.describe(&request.action)
        ))
    }

    pub fn take_suggestion(&self) {
        if let Some(request) = self.suggestion.get_untracked() {
            self.suggestion.set(None);
            self.move_view(request);
        }
    }

    pub fn dismiss_suggestion(&self) {
        self.suggestion.set(None);
    }

    fn describe(&self, action: &ViewAction) -> String {
        self.state.with_untracked(|s| match action {
            ViewAction::OpenSession { session_id } => format!("opening {}", s.title(session_id)),
            ViewAction::ShowApproval { approval_id } => match s.approvals.get(approval_id) {
                Some(a) => format!("looking at the approval \"{}\"", a.summary),
                None => "an approval that is no longer waiting".into(),
            },
            ViewAction::ShowTask { task_id } => match s.tasks.get(task_id) {
                Some(t) => format!("looking at work #{} {}", t.number, t.description),
                None => "work that is no longer running".into(),
            },
            ViewAction::FocusPane { pane } => format!("the {} pane", pane_name(*pane)),
            ViewAction::NewSession => "a new conversation".into(),
            ViewAction::ShowDevices => "your paired devices".into(),
            ViewAction::Unknown => "something this version cannot show".into(),
        })
    }

    fn move_view(&self, request: ViewRequest) {
        let what = self.describe(&request.action);
        let note = match request.action {
            ViewAction::OpenSession { session_id } => {
                self.select(Some(session_id));
                format!("Moved to {}", what.trim_start_matches("opening "))
            }
            ViewAction::ShowApproval { approval_id } => {
                let session = self.state.with_untracked(|s| {
                    s.approvals.get(&approval_id).map(|a| a.session_id.clone())
                });
                match session {
                    Some(session) => {
                        self.select(Some(session));
                        self.spotlight.set(Some(approval_id));
                        "Showing the approval: read it, then answer".into()
                    }
                    None => format!("Could not show {what}"),
                }
            }
            ViewAction::ShowTask { task_id } => {
                let session = self
                    .state
                    .with_untracked(|s| s.tasks.get(&task_id).map(|t| t.session_id.clone()));
                match session {
                    Some(session) => {
                        self.select(Some(session));
                        self.inspector.set(true);
                        format!("Showing {}", what.trim_start_matches("looking at "))
                    }
                    None => format!("Could not show {what}"),
                }
            }
            ViewAction::FocusPane { pane } => {
                match pane {
                    Pane::Sessions => self.sidebar.set(true),
                    Pane::Approvals | Pane::Tasks => self.inspector.set(true),
                    Pane::Chat | Pane::Unknown => {}
                }
                format!("Moved to {what}")
            }
            ViewAction::NewSession => {
                self.select(None);
                "Ready for a new conversation".into()
            }
            ViewAction::ShowDevices => {
                "Paired devices: ask your agent for /devices (not shown in this app yet)".into()
            }
            ViewAction::Unknown => return,
        };
        let reason = request.reason.trim();
        self.toast.set(Some(if reason.is_empty() {
            note
        } else {
            format!("{note} · {reason}")
        }));
    }

    /// Bring a waiting approval into view (a notification was pressed).
    pub fn show_approval(&self, approval_id: &str) {
        self.move_view(ViewRequest {
            action: ViewAction::ShowApproval {
                approval_id: approval_id.to_owned(),
            },
            reason: String::new(),
        });
    }

    pub fn toast(&self) -> Option<String> {
        self.toast.get()
    }

    /// Read to rerun once a second while work runs.
    pub fn tick(&self) -> u64 {
        self.tick.get()
    }

    /// The shell's second hand.
    pub fn advance_tick(&self) {
        if !self.state.with_untracked(|s| s.tasks.is_empty()) {
            self.tick.update(|t| *t += 1);
        }
    }

    /// A short note to the owner ("Copied the reply").
    pub fn say(&self, text: impl Into<String>) {
        self.toast.set(Some(text.into()));
    }

    pub fn dismiss_toast(&self) {
        self.toast.set(None);
    }

    pub fn spotlight(&self) -> Option<String> {
        self.spotlight.get()
    }

    // ----- Panes -----------------------------------------------------------

    pub fn sidebar_open(&self) -> bool {
        self.sidebar.get()
    }

    pub fn inspector_open(&self) -> bool {
        self.inspector.get()
    }

    /// A wide window's panes change at once: the resizable group keeps the
    /// widths it measures, and an animated close would leave it at none.
    pub fn toggle_sidebar(&self) {
        self.sidebar.update(|s| *s = !*s);
        if !self.sidebar.get_untracked() {
            self.focus_composer();
        }
    }

    pub fn toggle_inspector(&self) {
        self.inspector.update(|s| *s = !*s);
    }

    pub fn set_sidebar(&self, open: bool) {
        self.sidebar.set(open);
    }

    /// Changes when the keyboard should go back to the composer.
    pub fn composer_focus(&self) -> u64 {
        self.composer_focus.get()
    }

    fn focus_composer(&self) {
        self.composer_focus.update(|n| *n += 1);
    }

    /// Open the details on the Changes of the conversation on screen (its
    /// Steps when it has no changes).
    pub fn show_changes(&self) {
        let coding = self
            .selected
            .get_untracked()
            .is_some_and(|id| self.is_coding(&id));
        self.inspector_tab
            .set(if coding { CHANGES_TAB } else { STEPS_TAB });
        if self.compact.get_untracked() {
            self.sidebar.set(false);
        }
        self.inspector.set(true);
        if coding {
            self.refresh_changes();
        }
    }

    pub fn set_inspector(&self, open: bool) {
        self.inspector_default.replace(open);
        self.inspector
            .set(open && untracked(|| self.coding_on_screen()));
    }

    pub fn is_compact(&self) -> bool {
        self.compact.get()
    }

    /// The window became narrow (or wide again). Narrow starts on the
    /// conversation, with both panes closed.
    pub fn set_compact(&self, compact: bool) {
        if self.compact.get_untracked() != compact {
            self.compact.set(compact);
            if compact {
                self.sidebar.set(false);
                self.inspector.set(false);
            } else {
                self.sidebar.set(true);
                self.inspector
                    .set(self.inspector_default.get() && untracked(|| self.coding_on_screen()));
            }
        }
    }

    /// Open one pane on a narrow window (the other closes); toggle it on a
    /// wide one.
    pub fn open_sidebar(&self) {
        if self.compact.get_untracked() {
            self.inspector.set(false);
            self.sidebar.set(true);
        } else {
            self.toggle_sidebar();
        }
    }

    /// Show the sidebar (closing the details on a narrow window).
    pub fn reveal_sidebar(&self) {
        if self.compact.get_untracked() {
            self.inspector.set(false);
        }
        self.sidebar.set(true);
    }

    pub fn open_inspector(&self) {
        if self.compact.get_untracked() {
            self.sidebar.set(false);
            self.inspector.set(true);
        } else {
            self.toggle_inspector();
        }
    }

    /// Back to the conversation (a narrow window's panes).
    pub fn close_panes(&self) {
        if self.compact.get_untracked() {
            self.sidebar.set(false);
            self.inspector.set(false);
        }
        self.focus_composer();
    }

    /// Activity for running turns this client did not see start: their
    /// tools in order, then what they are doing now. Turns it already
    /// follows keep their own record.
    fn note_live_turns(&self, live: &[aidaemon_client_protocol::LiveTurnState]) {
        for turn in live {
            if self.turns.with_untracked(|t| t.contains_key(&turn.turn_id)) {
                continue;
            }
            self.note_activity(&Event::TurnStarted {
                turn_id: turn.turn_id.clone(),
                session_id: turn.session_id.clone(),
                origin: String::new(),
            });
            let last = turn.tools.len().saturating_sub(1);
            for (n, name) in turn.tools.iter().enumerate() {
                // The current status names the last tool with its summary.
                let summary = match &turn.status {
                    Some(TurnStatus::ToolStarted { name: now, summary })
                        if n == last && now == name =>
                    {
                        summary.clone()
                    }
                    _ => String::new(),
                };
                self.note_activity(&Event::TurnStatus {
                    turn_id: turn.turn_id.clone(),
                    status: TurnStatus::ToolStarted {
                        name: name.clone(),
                        summary,
                    },
                });
            }
            if let Some(status) = &turn.status
                && !matches!(status, TurnStatus::ToolStarted { .. })
            {
                self.note_activity(&Event::TurnStatus {
                    turn_id: turn.turn_id.clone(),
                    status: status.clone(),
                });
            }
        }
    }

    fn note_activity(&self, event: &Event) {
        let now = now_ms();
        let push = |turn_id: &str, kind: ActivityKind, title: String, detail: String| {
            let id = self.next_activity.with(|n| {
                *n += 1;
                *n
            });
            self.turns.update(|turns| {
                if let Some(log) = turns.get_mut(turn_id) {
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
            });
        };
        match event {
            Event::TurnStarted {
                turn_id,
                session_id,
                ..
            } => {
                self.turns.update(|turns| {
                    turns.insert(
                        turn_id.clone(),
                        TurnLog {
                            session_id: session_id.clone(),
                            ..TurnLog::default()
                        },
                    );
                });
                if self.selected.get_untracked().as_deref() == Some(session_id.as_str()) {
                    self.inspected.set(Some(turn_id.clone()));
                }
            }
            Event::TurnStatus { turn_id, status } => match status {
                TurnStatus::Thinking => push(
                    turn_id,
                    ActivityKind::Thinking,
                    "Thinking".into(),
                    String::new(),
                ),
                TurnStatus::ToolStarted { name, summary } => {
                    push(turn_id, ActivityKind::Tool, name.clone(), summary.clone())
                }
                TurnStatus::ToolProgress { name, chunk } => {
                    push(turn_id, ActivityKind::Output, name.clone(), chunk.clone())
                }
                TurnStatus::ToolCompleted { name, summary } => push(
                    turn_id,
                    ActivityKind::ToolDone,
                    name.clone(),
                    summary.clone(),
                ),
                TurnStatus::PlanStep {
                    step,
                    total,
                    description,
                    ..
                } => push(
                    turn_id,
                    ActivityKind::Plan,
                    format!("Step {} of {total}", step + 1),
                    description.clone(),
                ),
                TurnStatus::Checklist { text } => {
                    push(turn_id, ActivityKind::Plan, "Plan".into(), text.clone())
                }
                TurnStatus::Note { text } => {
                    push(turn_id, ActivityKind::Note, "Note".into(), text.clone())
                }
                TurnStatus::Unknown => {}
            },
            Event::TurnCompleted { turn_id, .. } => self.turns.update(|turns| {
                if let Some(log) = turns.get_mut(turn_id) {
                    log.done = true;
                }
            }),
            _ => {}
        }
    }

    // ----- Reads ----------------------------------------------------------

    pub fn connection(&self) -> Connection {
        self.connection.get()
    }

    /// The connection, read without subscribing (for the link's own loop).
    pub fn connection_untracked(&self) -> Connection {
        self.connection.get_untracked()
    }

    pub fn is_online(&self) -> bool {
        self.connection.with(|c| *c == Connection::Online)
    }

    /// Session ids, newest activity first.
    /// Conversations, the most recently active first; ones never active
    /// keep the gateway's order after them.
    pub fn session_ids(&self) -> Vec<String> {
        let mut ids = self.state.with(|s| s.order.clone());
        ids.sort_by_key(|id| std::cmp::Reverse(self.last_activity(id)));
        ids
    }

    /// A conversation's name: the daemon's title, else its first message,
    /// else where it lives ("Slack conversation").
    pub fn title(&self, id: &str) -> String {
        self.state.with(|s| {
            let view = s.sessions.get(id);
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
        })
    }

    /// Where a conversation lives, for its second line: platform, bot,
    /// and whether it is a group.
    pub fn place(&self, id: &str) -> String {
        let place = channel_name(&self.channel(id).unwrap_or_default(), id);
        if self.is_group(id) {
            format!("{place} · group")
        } else {
            place
        }
    }

    pub fn channel(&self, id: &str) -> Option<String> {
        self.state.with(|s| {
            s.sessions
                .get(id)
                .and_then(|v| v.summary.as_ref())
                .map(|summary| summary.channel.clone())
        })
    }

    pub fn session(&self, id: &str) -> Option<SessionView> {
        self.state.with(|s| s.sessions.get(id).cloned())
    }

    pub fn lines(&self, id: &str) -> Vec<Line> {
        self.state.with(|s| {
            s.sessions
                .get(id)
                .map(|v| v.lines.clone())
                .unwrap_or_default()
        })
    }

    pub fn live(&self, id: &str) -> Option<LiveTurn> {
        self.state
            .with(|s| s.sessions.get(id).and_then(|v| v.live.clone()))
    }

    /// A turn runs, or work it handed to the background does.
    pub fn is_working(&self, id: &str) -> bool {
        self.is_running(id) || self.in_background(id)
    }

    /// Its last turn handed work to the background and the report is
    /// still to come.
    pub fn in_background(&self, id: &str) -> bool {
        self.state
            .with(|s| s.sessions.get(id).is_some_and(|v| v.background))
    }

    pub fn is_running(&self, id: &str) -> bool {
        self.state.with(|s| {
            s.sessions.get(id).is_some_and(|v| v.live.is_some())
                || s.tasks.values().any(|t| t.session_id == id)
        })
    }

    pub fn approvals(&self) -> Vec<Approval> {
        self.state.with(|s| {
            let mut all: Vec<Approval> = s.approvals.values().cloned().collect();
            all.sort_by_key(|a| a.requested_at_ms);
            all
        })
    }

    pub fn approvals_for(&self, session: &str) -> Vec<Approval> {
        self.approvals()
            .into_iter()
            .filter(|a| a.session_id == session)
            .collect()
    }

    pub fn running(&self) -> Vec<TaskSummary> {
        self.state
            .with(|s| s.running().into_iter().cloned().collect())
    }

    pub fn notice(&self) -> Option<String> {
        self.state.with(|s| s.notice.clone())
    }

    pub fn selected(&self) -> Option<String> {
        self.selected.get()
    }

    pub fn selected_untracked(&self) -> Option<String> {
        self.selected.get_untracked()
    }

    fn draft_untracked(&self) -> String {
        let key = self.selected.get_untracked().unwrap_or_default();
        self.drafts
            .with_untracked(|d| d.get(&key).cloned().unwrap_or_default())
    }

    pub fn draft(&self) -> String {
        let key = self.selected.get().unwrap_or_default();
        self.drafts
            .with(|d| d.get(&key).cloned().unwrap_or_default())
    }

    /// The activity of the turn the inspector shows.
    pub fn inspected_turn(&self) -> Option<TurnLog> {
        let turn = self.inspected.get()?;
        self.turns.with(|t| t.get(&turn).cloned())
    }

    /// The latest turn of `session` this client saw.
    fn latest_turn(&self, session: &str) -> Option<String> {
        self.turns.with_untracked(|turns| {
            turns
                .iter()
                .filter(|(_, log)| log.session_id == session)
                .max_by_key(|(_, log)| log.items.first().map_or(0, |a| a.at_ms))
                .map(|(id, _)| id.clone())
        })
    }

    // ----- What the owner does --------------------------------------------

    pub fn select(&self, id: Option<String>) {
        crate::ui::motion::next_layout();
        self.close_panes();
        self.page.set(Page::Conversation);
        if let Some(id) = &id {
            self.unread.update(|u| {
                u.remove(id);
            });
        }
        if let Some(id) = &id {
            self.load_history(id);
            self.inspected.set(self.latest_turn(id));
            // A coding conversation opens on its changes; a newly picked one
            // reads them to know whether it is one.
            self.inspector_tab.set(if self.is_coding(id) {
                CHANGES_TAB
            } else {
                STEPS_TAB
            });
        } else {
            self.inspected.set(None);
        }
        // On a wide window the details show beside a coding conversation;
        // any other gets the room (they are a key away).
        let coding = id.as_deref().is_some_and(|id| self.is_coding(id));
        self.show_details(coding);
        let _ = nsh_core::storage::Storage::set_item(OPEN_KEY, id.clone().unwrap_or_default());
        // A plain new conversation starts in no project (or the project of
        // the folder the app runs in); `new_in_project` places it after.
        self.new_project.set(if id.is_none() {
            self.here_project()
        } else {
            None
        });
        let picked = id.is_some();
        self.selected.set(id);
        // Its changes say whether it works on code (and are ready to show).
        if picked {
            self.refresh_changes();
        }
    }

    fn load_history(&self, id: &str) {
        let first = self.fetched.with(|f| f.insert(id.to_owned()));
        let empty = self
            .state
            .with_untracked(|s| s.sessions.get(id).is_none_or(|v| v.lines.is_empty()));
        if first && empty {
            self.send(Command::GetHistory {
                session_id: id.to_owned(),
                before: None,
                limit: 50,
            });
        }
    }

    pub fn set_draft(&self, text: String) {
        let key = self.selected.get_untracked().unwrap_or_default();
        Self::keep_draft(&key, &text);
        self.drafts.update(|d| {
            d.insert(key, text);
        });
    }

    /// Send the draft of the selected conversation (or start one). A
    /// leading `/` runs a slash command in an existing session.
    pub fn submit(&self) {
        self.submit_placed(None);
    }

    /// [`Self::submit`], putting a message sent while the conversation works
    /// where the owner chose: into the running work, or after it.
    pub fn submit_placed(&self, mid_run: Option<MidRun>) {
        let key = self.selected.get_untracked();
        let text = self
            .drafts
            .with_untracked(|d| d.get(key.as_deref().unwrap_or(NEW)).cloned())
            .unwrap_or_default();
        let text = text.trim().to_owned();
        let files = self.draft_files.get_untracked();
        if files
            .iter()
            .any(|f| matches!(f.state, DraftFileState::Uploading))
        {
            self.say("Wait for the files to finish attaching");
            return;
        }
        let attachments: Vec<AttachmentRef> = files
            .into_iter()
            .filter_map(|f| match f.state {
                DraftFileState::Ready(attachment) => Some(attachment),
                _ => None,
            })
            .collect();
        if text.is_empty() && attachments.is_empty() {
            return;
        }
        if !attachments.is_empty() {
            self.draft_files.set(Vec::new());
        }
        match (
            &key,
            text.strip_prefix('/').filter(|_| attachments.is_empty()),
        ) {
            (Some(session), Some(slash)) => {
                let (command, args) = slash.split_once(' ').unwrap_or((slash, ""));
                self.send(Command::RunSlash {
                    session_id: session.clone(),
                    command: command.to_owned(),
                    args: args.trim().to_owned(),
                });
            }
            _ => {
                let id = self.send(Command::SendMessage {
                    session_id: key.clone(),
                    text,
                    attachments,
                    mid_run,
                    // A new conversation starts in the project it was opened in.
                    project_id: key
                        .is_none()
                        .then(|| self.new_project.get_untracked())
                        .flatten(),
                });
                if key.is_none() {
                    self.starting.replace(Some(id));
                    return;
                }
            }
        }
        Self::keep_draft(key.as_deref().unwrap_or(NEW), "");
        self.drafts.update(|d| {
            d.remove(key.as_deref().unwrap_or(NEW));
        });
    }

    // ----- Files with messages -------------------------------------------

    /// Where the gateway is, for files' links (native apps).
    pub fn set_gateway_base(&self, base: String) {
        self.gateway_base
            .replace(base.trim_end_matches('/').to_owned());
    }

    /// Where to fetch `file`.
    pub fn file_url(&self, file: &AttachmentRef) -> Option<String> {
        let url = file.url.as_deref()?;
        Some(format!("{}{url}", self.gateway_base.with(|b| b.clone())))
    }

    /// Files attached to the message being written.
    pub fn draft_files(&self) -> Vec<DraftFile> {
        self.draft_files.get()
    }

    /// Whether files can go with a message here: the app's own
    /// conversations (or a new one).
    pub fn takes_files(&self) -> bool {
        self.selected.get().is_none_or(|id| id.starts_with("app:"))
    }

    /// Attach a file to the message being written: it uploads now.
    pub fn attach_file(&self, name: String, mime: String, bytes: &[u8]) {
        use base64::Engine as _;
        if bytes.len() > MAX_FILE_BYTES {
            self.say(format!(
                "{name} is too large to attach (up to {} MB)",
                MAX_FILE_BYTES / (1024 * 1024)
            ));
            return;
        }
        let number = self.next_upload.with(|n| {
            *n += 1;
            *n
        });
        self.draft_files.update(|files| {
            files.push(DraftFile {
                number,
                name: name.clone(),
                state: DraftFileState::Uploading,
            })
        });
        let id = self.send(Command::UploadFile {
            name,
            mime: (!mime.is_empty()).then_some(mime),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
        });
        self.uploads.with(|u| u.insert(id, number));
    }

    /// Pick pictures to attach (scaled to fit 2048 px).
    pub fn pick_pictures(&self) {
        let store = *self;
        spawn(async move {
            let mut request = nsh_core::picker::PickRequest::new(true, 4);
            request.max_dimension = Some(DevicePixels(2048));
            for file in pick_images(request).await {
                store.attach_file(file.name, file.mime, file.data.bytes());
            }
        });
    }

    pub fn remove_draft_file(&self, number: u64) {
        self.uploads.with(|u| u.retain(|_, n| *n != number));
        self.draft_files
            .update(|files| files.retain(|f| f.number != number));
    }

    pub fn stop(&self, session: &str) {
        self.stopping.update(|s| {
            s.insert(session.to_owned());
        });
        self.send(Command::StopTurn {
            session_id: session.to_owned(),
        });
    }

    pub fn cancel(&self, task: &str) {
        self.send(Command::CancelTask {
            task_id: task.to_owned(),
        });
    }

    pub fn resolve(&self, approval: &str, decision: ApprovalDecision) {
        self.send(Command::ResolveApproval {
            approval_id: approval.to_owned(),
            decision,
        });
    }

    /// The link found no paired key on this device.
    pub fn needs_pairing(&self) {
        self.connection.set(Connection::NeedsPairing);
    }

    pub fn dismiss_notice(&self) {
        self.state.update(|s| s.notice = None);
    }
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}

fn pane_name(pane: Pane) -> &'static str {
    match pane {
        Pane::Sessions => "conversations",
        Pane::Chat => "chat",
        Pane::Approvals => "approvals",
        Pane::Tasks => "running work",
        Pane::Unknown => "unknown",
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

/// The wall clock in milliseconds.
pub fn now_ms() -> u64 {
    nsh_core::time::unix_ms().max(0.0) as u64
}

/// The store the root provides.
pub fn store() -> Store {
    use_context::<Store>().expect("the root provides the store")
}
