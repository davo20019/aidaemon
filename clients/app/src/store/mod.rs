//! The app's state: the shared [`Client`] (the reducer, the connection,
//! the reads, turns' activity, uploads, everything that is not drawing),
//! held in one signal, and the screen's own state beside it: the page,
//! the panes, what is selected and typed, toasts, the sidebar's filters,
//! this device's pins and drafts. Pages read it through these methods and
//! change it only through them. Protocol and connection logic lives in
//! `aidaemon_client_core::client`, never here.

use std::collections::{BTreeMap, BTreeSet};

use aidaemon_client_core::client::{Client, Context, Effect};
use aidaemon_client_core::machine::Update;
use aidaemon_client_core::state::{Line, LiveTurn, SessionView, ViewRequest};
use aidaemon_client_protocol::{
    Approval, ApprovalDecision, AttachmentRef, Command, Feature, MemoryItem, MidRun,
    ProjectSummary, SessionChanges, Settings, SettingsChange, SetupAction, SetupState,
    SlashCommand, TaskSummary,
};
use aidaemon_client_protocol::{Pane, ViewAction};
use nsh_core::prelude::*;

pub use aidaemon_client_core::client::{
    Activity, ActivityKind, Bot, Connection, DraftFile, DraftFileState, Read, SearchRead, Source,
    SystemRead, TurnLog, channel_name, platform_name, platform_of,
};

/// The Automations page's list, as last read.
pub type AutomationsRead = Read<Vec<aidaemon_client_protocol::AutomationSummary>>;
/// The details pane's Changes: a session's uncommitted changes.
pub type ChangesRead = Read<SessionChanges>;
/// The Memory page's data.
pub type MemoryRead = Read<Vec<MemoryItem>>;
/// The Settings page's data.
pub type SettingsRead = Read<Box<Settings>>;

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

/// The details pane's tabs for a coding conversation, in order; any other
/// conversation shows only its steps.
pub const DETAIL_TABS: [&str; 2] = ["Changes", "Steps"];
/// The Changes tab's index in [`DETAIL_TABS`].
pub const CHANGES_TAB: usize = 0;
/// The Steps tab's index in [`DETAIL_TABS`].
pub const STEPS_TAB: usize = 1;

/// Where drafts are kept between runs.
const DRAFT_KEY: &str = "aidaemon.draft.";
const PINS_KEY: &str = "aidaemon.pins";
const OPEN_KEY: &str = "aidaemon.open";
const ACTIVITY_KEY: &str = "aidaemon.activity";
const AUTOMATIONS_SEEN_KEY: &str = "aidaemon.automations-seen";
const ENTER_QUEUES_KEY: &str = "aidaemon.enter-queues";
/// The first approval card says how approvals work, once per device.
const APPROVALS_TAUGHT_KEY: &str = "aidaemon.approvals-taught";

/// The draft key of a conversation not started yet.
const NEW: &str = "";

/// Where commands go: the link of this platform.
pub type Outbox = Box<dyn Fn(String, Command)>;

/// The client's reads that pages build from, each a memo: a page rebuilds
/// when its own read changes, not on every frame.
#[derive(Clone, Copy)]
struct Reads {
    connection: Memo<Connection>,
    setup: Memo<Option<SetupState>>,
    setup_busy: Memo<bool>,
    setup_error: Memo<Option<String>>,
    system: Memo<SystemRead>,
    automations: Memo<AutomationsRead>,
    automation_busy: Memo<Option<String>>,
    memories: Memo<MemoryRead>,
    settings: Memo<SettingsRead>,
    settings_busy: Memo<bool>,
    settings_error: Memo<Option<String>>,
    settings_waiting: Memo<usize>,
    search: Memo<SearchRead>,
    changes: Memo<ChangesRead>,
    draft_files: Memo<Vec<DraftFile>>,
    projects: Memo<Vec<ProjectSummary>>,
    project_error: Memo<Option<String>>,
    saving_project: Memo<bool>,
    sources: Memo<Vec<Source>>,
    bots: Memo<Vec<Bot>>,
    approvals: Memo<Vec<Approval>>,
    running: Memo<Vec<TaskSummary>>,
    notice: Memo<Option<String>>,
    session_ids: Memo<Vec<String>>,
    unread: Memo<BTreeSet<String>>,
    finished: Memo<Option<(String, u64)>>,
    commands: Memo<Vec<SlashCommand>>,
    models: Memo<Read<aidaemon_client_protocol::ModelOptions>>,
}

impl Reads {
    fn new(client: Signal<Client>) -> Self {
        Self {
            connection: computed(move || client.with(|c| c.connection.clone())),
            setup: computed(move || client.with(|c| c.state.setup.clone())),
            setup_busy: computed(move || client.with(Client::setup_busy)),
            setup_error: computed(move || client.with(|c| c.setup_error.clone())),
            system: computed(move || client.with(|c| c.system.clone())),
            automations: computed(move || client.with(|c| c.automations.clone())),
            automation_busy: computed(move || client.with(|c| c.automation_busy.clone())),
            memories: computed(move || client.with(|c| c.memories.clone())),
            settings: computed(move || client.with(|c| c.settings.clone())),
            settings_busy: computed(move || client.with(Client::settings_busy)),
            settings_error: computed(move || client.with(|c| c.settings_error.clone())),
            settings_waiting: computed(move || client.with(|c| c.settings_waiting.len())),
            search: computed(move || client.with(|c| c.search.clone())),
            changes: computed(move || client.with(|c| c.changes.clone())),
            draft_files: computed(move || client.with(|c| c.draft_files.clone())),
            projects: computed(move || client.with(Client::projects)),
            project_error: computed(move || client.with(|c| c.project_error.clone())),
            saving_project: computed(move || client.with(Client::saving_project)),
            sources: computed(move || client.with(Client::sources)),
            bots: computed(move || client.with(Client::bots)),
            approvals: computed(move || client.with(Client::approvals)),
            running: computed(move || client.with(Client::running)),
            notice: computed(move || client.with(|c| c.state.notice.clone())),
            session_ids: computed(move || client.with(Client::session_ids)),
            unread: computed(move || client.with(|c| c.unread.clone())),
            finished: computed(move || client.with(|c| c.finished.clone())),
            commands: computed(move || client.with(|c| c.commands.clone())),
            models: computed(move || client.with(|c| c.models.clone())),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Store {
    client: Signal<Client>,
    reads: Reads,
    outbox: Stored<Option<Outbox>>,
    selected: Signal<Option<String>>,
    drafts: Signal<BTreeMap<String, String>>,
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
    /// What the app last told the owner.
    toast: Signal<Option<String>>,
    page: Signal<Page>,
    /// The sidebar's search.
    filter: Signal<String>,
    /// Conversations kept at the top of the list (this device's storage).
    pins: Signal<BTreeSet<String>>,
    /// Seconds, for elapsed times (advanced by the shell while work runs).
    tick: Signal<u64>,
    activity_mode: Signal<ActivityMode>,
    /// The gateway's HTTP origin, before a file's link; empty in the
    /// browser, where links are on the page's own origin.
    gateway_base: Stored<String>,
    /// The project a new conversation starts in (`None`: no project).
    new_project: Signal<Option<String>>,
    /// The folder the app was started in (the terminal's): new
    /// conversations start in the project of that folder.
    here: Stored<Option<String>>,
    /// Projects shown open in the sidebar.
    open_projects: Signal<BTreeSet<String>>,
    /// The New project form, and the project it edits (`None`: a new one).
    project_form: Signal<bool>,
    project_editing: Signal<Option<String>>,
    /// The details pane's open tab (an index of [`DETAIL_TABS`]).
    inspector_tab: Signal<usize>,
    /// The sidebar's source filter.
    source: Signal<Source>,
    /// The automation whose edit form is open: the page's timed refresh
    /// waits, so what the owner is typing stays.
    automation_editing: Signal<Option<String>>,
    /// Advanced when the keyboard should go back to the composer.
    composer_focus: Signal<u64>,
    /// The sidebar lists archived conversations instead of the others.
    show_archived: Signal<bool>,
    /// While the agent works, Enter queues (sends after) and ⌘↩ tells it
    /// now, instead of the other way round (this device).
    enter_queues: Signal<bool>,
    /// The conversation whose name is being edited in the sidebar.
    renaming: Signal<Option<String>>,
    /// When the owner last looked at the automations (this device): runs
    /// after it are news.
    automations_seen: Signal<u64>,
    /// The look before this one, while the page is open: what it marks new.
    automations_seen_before: Signal<u64>,
    /// Group conversations the owner chose to post in from here.
    posting: Signal<BTreeSet<String>>,
    /// Starting the daemon on this computer or pairing with one elsewhere,
    /// before any link: what is happening, or why it stopped.
    starting_daemon: Signal<Option<(String, bool)>>,
}

impl Store {
    pub fn new() -> Self {
        let client = signal_with(Client::new(), SignalEquals::Never);
        Self {
            client,
            reads: Reads::new(client),
            outbox: stored(None),
            selected: signal(None),
            drafts: signal(BTreeMap::new()),
            sidebar: signal(true),
            inspector: signal(true),
            inspector_default: stored(true),
            compact: signal(false),
            suggestion: signal(None),
            spotlight: signal(None),
            toast: signal(None),
            page: signal(Page::Conversation),
            filter: signal(String::new()),
            pins: signal(BTreeSet::new()),
            tick: signal(0),
            activity_mode: signal(ActivityMode::default()),
            gateway_base: stored(String::new()),
            new_project: signal(None),
            here: stored(None),
            open_projects: signal(BTreeSet::new()),
            project_form: signal(false),
            project_editing: signal(None),
            inspector_tab: signal(STEPS_TAB),
            source: signal(Source::All),
            automation_editing: signal(None),
            composer_focus: signal(0),
            show_archived: signal(false),
            enter_queues: signal(
                nsh_core::storage::Storage::get_item(ENTER_QUEUES_KEY).as_deref() == Some("1"),
            ),
            renaming: signal(None),
            automations_seen: signal(
                nsh_core::storage::Storage::get_item(AUTOMATIONS_SEEN_KEY)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
            ),
            automations_seen_before: signal(0),
            posting: signal(BTreeSet::new()),
            starting_daemon: signal(None),
        }
    }

    // ----- The client --------------------------------------------------------

    /// Read the client (tracked).
    fn read<R>(&self, f: impl FnOnce(&Client) -> R) -> R {
        self.client.with(f)
    }

    /// Change the client, then send what it queued once there is a link.
    fn act<R>(&self, f: impl FnOnce(&mut Client) -> R) -> R {
        let mut result = None;
        let linked = self.outbox.with(|o| o.is_some());
        let mut outgoing = Vec::new();
        self.client.update(|client| {
            result = Some(f(client));
            if linked {
                outgoing = client.take_outgoing();
            }
        });
        self.flush(outgoing);
        result.expect("ran")
    }

    fn flush(&self, outgoing: Vec<(String, Command)>) {
        if outgoing.is_empty() {
            return;
        }
        self.outbox.with(|outbox| {
            if let Some(outbox) = outbox {
                for (id, command) in outgoing {
                    outbox(id, command);
                }
            }
        });
    }

    /// The link is up: commands go to `outbox` from now on.
    pub fn attach(&self, outbox: Outbox) {
        self.outbox.replace(Some(outbox));
        self.act(|_| ());
    }

    /// What only the screen knows, for the client.
    fn context(&self) -> Context {
        let selected = self.selected.get_untracked();
        let page = self.page.get_untracked();
        Context {
            now_ms: now_ms(),
            looking: page == Page::Conversation
                && nsh_core::element::lifecycle() == nsh_core::platform::Lifecycle::Active,
            changes_visible: self.inspector.get_untracked()
                && self.inspector_tab.get_untracked() == CHANGES_TAB
                && page == Page::Conversation,
            settings_open: page == Page::Settings,
            selected,
        }
    }

    // ----- What the link reports -----------------------------------------

    pub fn update(&self, update: Update) {
        let cx = self.context();
        let effects = self.act(|client| client.update(update, &cx));
        for effect in effects {
            self.effect(effect);
        }
        self.settle_here();
    }

    fn effect(&self, effect: Effect) {
        match effect {
            Effect::Say(text) => self.say(text),
            Effect::Started(session) => {
                // The draft was sent: the new conversation starts empty.
                Self::keep_draft(NEW, "");
                self.drafts.update(|d| {
                    d.remove(NEW);
                });
                self.selected.set(Some(session));
            }
            Effect::OpenGreeting(session) => {
                self.selected.set(Some(session));
                self.page.set(Page::Conversation);
            }
            Effect::NowCoding(session) => {
                if self.selected.get_untracked().as_deref() == Some(session.as_str()) {
                    self.show_details(true);
                }
            }
            Effect::OpenChanges(session) => {
                if self.selected.get_untracked().as_deref() == Some(session.as_str()) {
                    self.inspector_tab.set(CHANGES_TAB);
                }
            }
            Effect::ProjectSaved(project) => {
                self.project_form.set(false);
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
            Effect::View(request) => self.view_request(request),
        }
    }

    /// The id of the last queued command of `kind` (its `type`): what a
    /// test answers with an ack, as the daemon would.
    pub fn queued_id(&self, kind: &str) -> Option<String> {
        self.client.with_untracked(|client| {
            client.outgoing().iter().rev().find_map(|(id, command)| {
                let json = serde_json::to_value(command).ok()?;
                (json.get("type")?.as_str()? == kind).then(|| id.clone())
            })
        })
    }

    // ----- First-run setup -------------------------------------------------

    /// Setup still to finish, as the daemon last said.
    pub fn setup(&self) -> Option<SetupState> {
        self.reads.setup.get()
    }

    pub fn setup_step(&self, action: SetupAction) {
        self.act(|c| c.setup_step(action));
    }

    pub fn setup_busy(&self) -> bool {
        self.reads.setup_busy.get()
    }

    pub fn setup_error(&self) -> Option<String> {
        self.reads.setup_error.get()
    }

    /// No daemon to sign in to: show where the agent should run.
    pub fn needs_daemon(&self) {
        self.act(Client::needs_daemon);
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
        self.read(|c| {
            c.state
                .sessions
                .get(session)
                .map(|v| v.suggestions.clone())
                .unwrap_or_default()
        })
    }

    /// Features the agent asked for in `session`.
    pub fn feature_requests(&self, session: &str) -> Vec<(Feature, String)> {
        self.read(|c| {
            c.state
                .feature_requests
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
        self.act(|c| {
            c.change_settings(SettingsChange::Set {
                key: key.into(),
                value: Some("true".into()),
            });
            c.change_settings(SettingsChange::Apply {
                how: aidaemon_client_protocol::ApplyHow::Restart,
            });
            c.state.settle_feature(feature);
        });
    }

    pub fn dismiss_feature(&self, feature: Feature) {
        self.act(|c| c.state.settle_feature(feature));
    }

    /// Whether this device has yet to be told how approvals work.
    pub fn teach_approvals(&self) -> bool {
        nsh_core::storage::Storage::get_item(APPROVALS_TAUGHT_KEY).is_none()
    }

    pub fn approvals_taught(&self) {
        let _ = nsh_core::storage::Storage::set_item(APPROVALS_TAUGHT_KEY, "1");
    }

    // ----- This device's storage ------------------------------------------

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

    // ----- Pages -----------------------------------------------------------

    pub fn page(&self) -> Page {
        self.page.get()
    }

    fn open_page(&self, page: Page) {
        crate::ui::motion::next_layout();
        self.close_panes();
        self.page.set(page);
    }

    pub fn open_system(&self) {
        self.open_page(Page::System);
        self.refresh_system();
    }

    pub fn refresh_system(&self) {
        self.act(Client::refresh_system);
    }

    pub fn system(&self) -> SystemRead {
        self.reads.system.get()
    }

    /// Tokens used today, when the System read has them.
    pub fn today_tokens(&self) -> Option<u64> {
        // Usage days are UTC dates.
        let today = crate::ui::time::utc_date(now_ms());
        self.read(|c| c.tokens_on(&today))
    }

    pub fn open_automations(&self) {
        self.open_page(Page::Automations);
        // What came since the last look shows as new while the page is
        // open; from now on it is seen.
        let now = now_ms();
        self.automations_seen_before
            .set(self.automations_seen.get_untracked());
        self.automations_seen.set(now);
        let _ = nsh_core::storage::Storage::set_item(AUTOMATIONS_SEEN_KEY, now.to_string());
        self.refresh_automations();
    }

    /// Runs with something to read since the owner last looked.
    pub fn automation_news(&self) -> usize {
        let seen = self.automations_seen.get();
        self.reads.automations.with(|_| ());
        self.client.with_untracked(|c| c.automation_news(seen))
    }

    /// Whether `run` is new to the owner on the page open now.
    pub fn run_is_new(&self, run: &aidaemon_client_protocol::AutomationRun) -> bool {
        aidaemon_client_core::client::run_is_news(run, self.automations_seen_before.get())
    }

    pub fn refresh_automations(&self) {
        self.act(Client::refresh_automations);
    }

    // ----- A conversation's model -------------------------------------------

    /// The providers and models a conversation can answer with.
    pub fn model_options(&self) -> Read<aidaemon_client_protocol::ModelOptions> {
        self.reads.models.get()
    }

    pub fn refresh_models(&self) {
        self.act(Client::refresh_models);
    }

    /// Read what `session` answers with, once.
    pub fn load_session_model(&self, session: &str) {
        self.act(|c| c.load_session_model(session));
    }

    pub fn can_pick_model(&self, session: Option<&str>) -> bool {
        self.read(|c| c.can_pick_model(session))
    }

    /// The model and effort `session` (`None`: a new conversation) answers
    /// with.
    pub fn model_of(
        &self,
        session: Option<&str>,
    ) -> (
        aidaemon_client_protocol::ModelChoice,
        aidaemon_client_protocol::Effort,
    ) {
        self.read(|c| c.model_of(session))
    }

    /// The efforts `model` takes besides automatic.
    pub fn efforts_of(
        &self,
        model: &aidaemon_client_protocol::ModelChoice,
    ) -> Vec<aidaemon_client_protocol::Effort> {
        self.read(|c| c.efforts_of(model))
    }

    pub fn set_session_model(
        &self,
        session: Option<&str>,
        model: Option<aidaemon_client_protocol::ModelChoice>,
        effort: Option<aidaemon_client_protocol::Effort>,
    ) {
        self.act(|c| c.set_session_model(session, model, effort));
    }

    pub fn automations(&self) -> AutomationsRead {
        self.reads.automations.get()
    }

    /// The automation a change was sent for and not answered yet.
    pub fn automation_busy(&self) -> Option<String> {
        self.reads.automation_busy.get()
    }

    pub fn change_automation(&self, id: &str, change: aidaemon_client_protocol::AutomationChange) {
        self.act(|c| c.change_automation(id, change));
    }

    pub fn edit_automation(
        &self,
        id: &str,
        name: Option<String>,
        when: Option<String>,
        instructions: Option<String>,
        notify: Option<aidaemon_client_protocol::AutomationNotify>,
    ) {
        if self.act(|c| c.edit_automation(id, name, when, instructions, notify)) {
            self.automation_editing.set(None);
        }
    }

    pub fn automation_editing(&self) -> Option<String> {
        self.automation_editing.get()
    }

    /// Open (`Some`) or close the edit form of one automation.
    pub fn set_automation_editing(&self, id: Option<String>) {
        self.automation_editing.set(id);
    }

    /// Change an automation by asking: a new conversation, started.
    pub fn change_automation_in_chat(&self, title: &str) {
        self.select(None);
        self.set_draft(format!("Change my automation \"{title}\": "));
    }

    pub fn open_memory(&self) {
        self.open_page(Page::Memory);
        self.refresh_memories();
    }

    pub fn refresh_memories(&self) {
        self.act(Client::refresh_memories);
    }

    pub fn memories(&self) -> MemoryRead {
        self.reads.memories.get()
    }

    pub fn forget_memory(&self, id: &str) {
        self.act(|c| c.forget_memory(id));
    }

    pub fn open_settings(&self) {
        self.open_page(Page::Settings);
        self.refresh_settings();
    }

    pub fn refresh_settings(&self) {
        self.act(Client::refresh_settings);
    }

    pub fn settings(&self) -> SettingsRead {
        self.reads.settings.get()
    }

    pub fn change_settings(&self, change: SettingsChange) {
        self.act(|c| c.change_settings(change));
    }

    pub fn settings_busy(&self) -> bool {
        self.reads.settings_busy.get()
    }

    pub fn settings_error(&self) -> Option<String> {
        self.reads.settings_error.get()
    }

    pub fn dismiss_settings_error(&self) {
        self.act(|c| c.settings_error = None);
    }

    /// Settings changes waiting for an approval.
    pub fn settings_waiting(&self) -> usize {
        self.reads.settings_waiting.get()
    }

    // ----- Projects ----------------------------------------------------------

    pub fn projects(&self) -> Vec<ProjectSummary> {
        self.reads.projects.get()
    }

    pub fn project(&self, id: &str) -> Option<ProjectSummary> {
        self.read(|c| c.state.projects.get(id).cloned())
    }

    pub fn session_project(&self, session: &str) -> Option<String> {
        self.read(|c| c.session_project(session))
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

    /// The project whose folder holds the folder the app was started in.
    pub fn here_project(&self) -> Option<String> {
        let here = self.here.with(|h| h.clone())?;
        self.client.with_untracked(|c| c.project_of_folder(&here))
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

    /// Choose the project the new conversation on screen starts in; then
    /// typing goes on in the composer.
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
        self.act(|c| c.project_error = None);
        self.project_editing.set(None);
    }

    /// Open the project form on project `id`, to rename it or change its
    /// folder.
    pub fn edit_project(&self, id: &str) {
        crate::ui::motion::next_unfold();
        self.act(|c| c.project_error = None);
        self.project_editing.set(Some(id.to_owned()));
        self.project_form.set(true);
    }

    pub fn project_editing(&self) -> Option<String> {
        self.project_editing.get()
    }

    /// Save the form: a new project, or the edited one's name and folder.
    pub fn save_project(&self, name: &str, path: &str) {
        let id = self.project_editing.get_untracked();
        self.act(|c| c.save_project(id, name, path));
    }

    pub fn project_error(&self) -> Option<String> {
        self.reads.project_error.get()
    }

    pub fn creating_project(&self) -> bool {
        self.reads.saving_project.get()
    }

    pub fn create_project(&self, name: &str, path: &str) {
        self.act(|c| c.save_project(None, name, path));
    }

    /// Put a conversation in a project, or in none.
    pub fn move_to_project(&self, session: &str, project: Option<String>) {
        if let Some(id) = &project {
            self.open_projects.update(|open| {
                open.insert(id.clone());
            });
        }
        self.act(|c| c.move_to_project(session, project));
    }

    pub fn bots(&self) -> Vec<Bot> {
        self.reads.bots.get()
    }

    pub fn set_bot_project(&self, channel: &str, project: Option<String>) {
        self.act(|c| c.set_bot_project(channel, project));
    }

    // ----- The details pane: Changes and Steps ---------------------------------

    pub fn inspector_tab(&self) -> Signal<usize> {
        self.inspector_tab
    }

    pub fn is_coding(&self, id: &str) -> bool {
        self.read(|c| c.is_coding(id))
    }

    /// Whether the selected conversation works on code.
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

    pub fn refresh_changes(&self) {
        let selected = self.selected.get_untracked();
        self.act(|c| c.refresh_changes(selected.as_deref()));
    }

    pub fn refresh_changes_if_stale(&self) {
        let selected = self.selected.get_untracked();
        self.act(|c| c.refresh_changes_if_stale(selected.as_deref()));
    }

    pub fn changes(&self) -> ChangesRead {
        self.reads.changes.get()
    }

    /// How many files the conversation on screen has changed, once read.
    pub fn changed_files(&self) -> Option<usize> {
        let selected = self.selected.get()?;
        self.read(|c| c.changed_files(&selected))
    }

    /// Show or hide the details beside the conversation, on a wide window
    /// of a platform that shows them by default (not the terminal).
    fn show_details(&self, show: bool) {
        if !self.compact.get_untracked() && self.inspector_default.get() {
            self.inspector.set(show);
        }
    }

    // ----- Search of every conversation ------------------------------------

    pub fn search_messages(&self, query: &str) {
        self.act(|c| c.search_messages(query));
    }

    pub fn search(&self) -> SearchRead {
        self.reads.search.get()
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

    pub fn slash_matches(&self) -> Vec<SlashCommand> {
        let draft = self.draft();
        if !draft.starts_with('/') {
            return Vec::new();
        }
        self.reads.commands.with(|_| ());
        self.client.with_untracked(|c| c.slash_matches(&draft))
    }

    /// Put `/name ` in the composer.
    pub fn complete_slash(&self, name: &str) {
        self.set_draft(format!("/{name} "));
    }

    pub fn last_user_text(&self, session: &str) -> Option<String> {
        self.read(|c| c.last_user_text(session))
    }

    pub fn last_message(&self, session: &str) -> Option<(bool, String)> {
        self.read(|c| c.last_message(session))
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

    pub fn can_retry(&self, session: &str) -> bool {
        self.read(|c| c.can_retry(session))
    }

    pub fn retry(&self, session: &str) {
        self.act(|c| c.retry(session));
    }

    /// Say `text` in `session`, as if typed (a suggested prompt pressed).
    pub fn ask(&self, session: &str, text: String) {
        self.act(|c| c.ask(session, text));
    }

    pub fn needs_you(&self, id: &str) -> bool {
        self.read(|c| c.needs_you(id))
    }

    // ----- Where conversations come from ----------------------------------------

    pub fn source(&self, id: &str) -> Source {
        self.read(|c| c.source(id))
    }

    pub fn is_group(&self, id: &str) -> bool {
        self.read(|c| c.is_group(id))
    }

    /// Whether the app may send to conversation `id` now: anywhere but a
    /// group the owner has not chosen to post in.
    pub fn may_post(&self, id: &str) -> bool {
        !self.is_group(id) || self.posting.with(|p| p.contains(id))
    }

    pub fn allow_posting(&self, id: &str) {
        self.posting.update(|p| {
            p.insert(id.to_owned());
        });
    }

    pub fn source_filter(&self) -> Source {
        self.source.get()
    }

    pub fn set_source_filter(&self, source: Source) {
        self.source.set(source);
    }

    pub fn sources(&self) -> Vec<Source> {
        self.reads.sources.get()
    }

    pub fn in_source(&self, id: &str) -> bool {
        let source = self.source.get();
        self.read(|c| c.in_source(id, source))
    }

    // ----- Search, unread, stopping ------------------------------------------

    pub fn filter(&self) -> String {
        self.filter.get()
    }

    pub fn set_filter(&self, text: String) {
        self.filter.set(text);
    }

    /// Session ids matching the sidebar's search, pinned first, each part
    /// newest first.
    pub fn visible_ids(&self) -> Vec<String> {
        let query = self.filter.get();
        let archived = self.show_archived.get();
        let mut ids = self.read(|c| {
            c.session_ids()
                .into_iter()
                .filter(|id| c.is_archived(id) == archived && c.matches(id, &query))
                .collect::<Vec<_>>()
        });
        ids.sort_by_key(|id| !self.is_pinned(id));
        ids
    }

    pub fn is_unread(&self, id: &str) -> bool {
        self.reads.unread.with(|u| u.contains(id))
    }

    pub fn unread_count(&self) -> usize {
        self.reads.unread.with(BTreeSet::len)
    }

    pub fn is_stopping(&self, id: &str) -> bool {
        self.read(|c| c.is_stopping(id))
    }

    /// The latest finished turn: its session and when.
    pub fn finished(&self) -> Option<(String, u64)> {
        self.reads.finished.get()
    }

    pub fn last_activity(&self, id: &str) -> Option<u64> {
        self.read(|c| c.last_activity(id))
    }

    /// The selected conversation's last reply (Copy reply).
    pub fn last_reply(&self) -> Option<String> {
        let id = self.selected.get_untracked()?;
        self.client.with_untracked(|c| c.last_reply(&id))
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
        self.client.with_untracked(|c| match action {
            ViewAction::OpenSession { session_id } => format!("opening {}", c.title(session_id)),
            ViewAction::ShowApproval { approval_id } => match c.state.approvals.get(approval_id) {
                Some(a) => format!("looking at the approval \"{}\"", a.summary),
                None => "an approval that is no longer waiting".into(),
            },
            ViewAction::ShowTask { task_id } => match c.state.tasks.get(task_id) {
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
                let session = self.client.with_untracked(|c| {
                    c.state
                        .approvals
                        .get(&approval_id)
                        .map(|a| a.session_id.clone())
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
                    .client
                    .with_untracked(|c| c.state.tasks.get(&task_id).map(|t| t.session_id.clone()));
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
        if !self.client.with_untracked(|c| c.state.tasks.is_empty()) {
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

    /// Give the keyboard back to the composer (a palette command ran).
    pub fn refocus_composer(&self) {
        self.focus_composer();
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
            .is_some_and(|id| untracked(|| self.is_coding(&id)));
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

    // ----- Reads ----------------------------------------------------------

    pub fn connection(&self) -> Connection {
        self.reads.connection.get()
    }

    /// The connection, read without subscribing (for the link's own loop).
    pub fn connection_untracked(&self) -> Connection {
        self.client.with_untracked(|c| c.connection.clone())
    }

    pub fn is_online(&self) -> bool {
        self.reads.connection.with(|c| *c == Connection::Online)
    }

    /// Conversations, the most recently active first.
    pub fn session_ids(&self) -> Vec<String> {
        self.reads.session_ids.get()
    }

    pub fn title(&self, id: &str) -> String {
        self.read(|c| c.title(id))
    }

    pub fn place(&self, id: &str) -> String {
        self.read(|c| c.place(id))
    }

    pub fn channel(&self, id: &str) -> Option<String> {
        self.read(|c| c.channel(id))
    }

    pub fn session(&self, id: &str) -> Option<SessionView> {
        self.read(|c| c.state.sessions.get(id).cloned())
    }

    pub fn lines(&self, id: &str) -> Vec<Line> {
        self.read(|c| c.lines(id).to_vec())
    }

    pub fn live(&self, id: &str) -> Option<LiveTurn> {
        self.read(|c| c.state.sessions.get(id).and_then(|v| v.live.clone()))
    }

    /// A turn runs, or work it handed to the background does.
    pub fn is_working(&self, id: &str) -> bool {
        self.read(|c| c.is_running(id) || c.in_background(id))
    }

    pub fn in_background(&self, id: &str) -> bool {
        self.read(|c| c.in_background(id))
    }

    pub fn is_running(&self, id: &str) -> bool {
        self.read(|c| c.is_running(id))
    }

    pub fn approvals(&self) -> Vec<Approval> {
        self.reads.approvals.get()
    }

    pub fn approvals_for(&self, session: &str) -> Vec<Approval> {
        self.approvals()
            .into_iter()
            .filter(|a| a.session_id == session)
            .collect()
    }

    pub fn running(&self) -> Vec<TaskSummary> {
        self.reads.running.get()
    }

    pub fn notice(&self) -> Option<String> {
        self.reads.notice.get()
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

    /// The activity of the selected conversation's latest turn.
    pub fn inspected_turn(&self) -> Option<TurnLog> {
        let selected = self.selected.get()?;
        self.read(|c| c.latest_turn(&selected).cloned())
    }

    // ----- What the owner does --------------------------------------------

    pub fn select(&self, id: Option<String>) {
        crate::ui::motion::next_layout();
        self.close_panes();
        self.page.set(Page::Conversation);
        if let Some(id) = &id {
            self.act(|c| {
                c.seen(id);
                c.load_history(id);
            });
        }
        // A coding conversation opens on its changes, and on a wide window
        // shows its details beside it; any other gets the room (they are a
        // key away).
        let coding = id
            .as_deref()
            .is_some_and(|id| self.client.with_untracked(|c| c.is_coding(id)));
        if id.is_some() {
            self.inspector_tab
                .set(if coding { CHANGES_TAB } else { STEPS_TAB });
        }
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

    pub fn set_draft(&self, text: String) {
        let key = self.selected.get_untracked().unwrap_or_default();
        Self::keep_draft(&key, &text);
        self.drafts.update(|d| {
            d.insert(key, text);
        });
    }

    /// Send the draft of the selected conversation (or start one).
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
        let project = self.new_project.get_untracked();
        match self.act(|c| c.submit(key.as_deref(), &text, mid_run, project)) {
            Ok(true) => {
                Self::keep_draft(key.as_deref().unwrap_or(NEW), "");
                self.drafts.update(|d| {
                    d.remove(key.as_deref().unwrap_or(NEW));
                });
            }
            Ok(false) => {}
            Err(note) => self.say(note),
        }
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

    pub fn draft_files(&self) -> Vec<DraftFile> {
        self.reads.draft_files.get()
    }

    /// Whether files can go with a message here: the app's own
    /// conversations (or a new one).
    pub fn takes_files(&self) -> bool {
        self.selected.get().is_none_or(|id| id.starts_with("app:"))
    }

    /// Attach a file to the message being written: it uploads now.
    pub fn attach_file(&self, name: String, mime: String, bytes: &[u8]) {
        if let Err(note) = self.act(|c| c.attach_file(name, mime, bytes)) {
            self.say(note);
        }
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
        self.act(|c| c.remove_draft_file(number));
    }

    pub fn stop(&self, session: &str) {
        self.act(|c| c.stop(session));
    }

    pub fn cancel(&self, task: &str) {
        self.act(|c| c.cancel(task));
    }

    // ----- Naming, archiving, undoing, waiting messages ------------------

    pub fn renaming(&self) -> Option<String> {
        self.renaming.get()
    }

    /// Start (`Some`) or stop editing a conversation's name.
    pub fn set_renaming(&self, id: Option<String>) {
        self.renaming.set(id);
    }

    /// Name conversation `id` (blank: back to the name from what was said).
    pub fn rename_session(&self, id: &str, title: String) {
        self.renaming.set(None);
        self.act(|c| c.rename_session(id, Some(title)));
    }

    pub fn is_archived(&self, id: &str) -> bool {
        self.read(|c| c.is_archived(id))
    }

    /// Archive conversation `id` or bring it back; archiving the one on
    /// screen moves to a new conversation.
    pub fn archive_session(&self, id: &str, archived: bool) {
        self.act(|c| c.archive_session(id, archived));
        if archived && self.selected.get_untracked().as_deref() == Some(id) {
            self.select(None);
        }
        self.say(if archived {
            "Archived: it is under Archived at the bottom of the list"
        } else {
            "Back in your conversations"
        });
    }

    /// How many conversations are archived.
    pub fn archived_count(&self) -> usize {
        self.read(|c| c.state.order.iter().filter(|id| c.is_archived(id)).count())
    }

    /// While the agent works, whether Enter queues the message (sends it
    /// after) rather than telling the running work now.
    pub fn enter_queues(&self) -> bool {
        self.enter_queues.get()
    }

    pub fn set_enter_queues(&self, queues: bool) {
        self.enter_queues.set(queues);
        let _ =
            nsh_core::storage::Storage::set_item(ENTER_QUEUES_KEY, if queues { "1" } else { "0" });
    }

    pub fn showing_archived(&self) -> bool {
        self.show_archived.get()
    }

    pub fn show_archived(&self, show: bool) {
        crate::ui::motion::next_unfold();
        self.show_archived.set(show);
    }

    /// Undo the uncommitted changes to `path` in repository `root` of the
    /// conversation on screen.
    pub fn revert_file(&self, root: &str, path: &str) {
        if let Some(session) = self.selected.get_untracked() {
            self.act(|c| c.revert_file(&session, root, path));
        }
    }

    pub fn reverting(&self) -> bool {
        self.read(Client::reverting)
    }

    /// The messages waiting for `session`'s running work, oldest first.
    pub fn queued(&self, session: &str) -> Vec<aidaemon_client_protocol::QueuedMessage> {
        self.read(|c| c.queued_for(session).to_vec())
    }

    /// Change a waiting message's text, or take it out (`None`).
    pub fn edit_queued(&self, session: &str, message_id: u64, text: Option<String>) {
        self.act(|c| c.edit_queued(session, message_id, text));
    }

    pub fn resolve(&self, approval: &str, decision: ApprovalDecision) {
        self.act(|c| c.resolve(approval, decision));
    }

    /// The link found no paired key on this device.
    pub fn needs_pairing(&self) {
        self.act(Client::needs_pairing);
    }

    pub fn dismiss_notice(&self) {
        self.act(|c| c.state.notice = None);
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

/// The wall clock in milliseconds.
pub fn now_ms() -> u64 {
    nsh_core::time::unix_ms().max(0.0) as u64
}

/// The store the root provides.
pub fn store() -> Store {
    use_context::<Store>().expect("the root provides the store")
}
