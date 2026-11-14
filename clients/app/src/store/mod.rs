//! The app's state: the shared reducer's [`State`] (built only from the
//! gateway's frames), the connection, what the owner has selected and
//! typed, and each turn's activity as it happened. Pages read it through
//! these methods and change it only through them.

use std::collections::{BTreeMap, BTreeSet};

use aidaemon_client_core::machine::{DropKind, Update, request_id};
use aidaemon_client_core::state::{Line, LiveTurn, SessionView, State, ViewRequest};
use aidaemon_client_protocol::{
    Approval, ApprovalDecision, Command, CommandResult, Event, ServerFrame, SystemStatus,
    TaskSummary, TurnStatus,
};
use aidaemon_client_protocol::{Pane, ViewAction};
use nsh_core::prelude::*;

/// The connection to the daemon, as the owner should see it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Connection {
    #[default]
    Connecting,
    Online,
    /// This device has no key the daemon knows: pair it first.
    NeedsPairing,
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
}

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
            compact: signal(false),
            suggestion: signal(None),
            spotlight: signal(None),
            toast: signal(None),
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
        }
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
        if let Some(open) =
            nsh_core::storage::Storage::get_item(OPEN_KEY).filter(|id| !id.is_empty())
        {
            self.select(Some(open));
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
                Line::Assistant { text, at_ms } => ("AIdaemon", text, at_ms),
                Line::Note { text } => ("Note", text, 0),
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
                // A reconnect may have missed history: ask again on view.
                self.fetched.with(BTreeSet::clear);
                if let Some(id) = self.selected.get_untracked() {
                    self.load_history(&id);
                }
            }
            Update::Disconnected(disconnect) => {
                let retry = u64::try_from(disconnect.retry_in.as_millis()).unwrap_or(0);
                self.connection.set(Connection::Offline {
                    kind: disconnect.kind,
                    reason: disconnect.reason,
                    retry_at_ms: now_ms().saturating_add(retry),
                });
            }
            Update::Frame(frame) => self.frame(frame),
        }
    }

    fn frame(&self, frame: ServerFrame) {
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
            ServerFrame::Event {
                event, session_id, ..
            } => {
                self.note_activity(event);
                self.note_reply(session_id.as_deref(), event);
            }
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
        self.close_panes();
        self.page.set(Page::System);
        self.refresh_system();
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

    pub fn toggle_sidebar(&self) {
        self.sidebar.update(|s| *s = !*s);
    }

    pub fn toggle_inspector(&self) {
        self.inspector.update(|s| *s = !*s);
    }

    pub fn set_inspector(&self, open: bool) {
        self.inspector.set(open);
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
    pub fn session_ids(&self) -> Vec<String> {
        self.state.with(|s| s.order.clone())
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

    /// Where a conversation lives, for its second line: platform and place.
    pub fn place(&self, id: &str) -> String {
        let channel = self.channel(id).unwrap_or_default();
        let platform = platform_name(id);
        match channel.split_once(':') {
            Some((_, bot)) if !bot.is_empty() => format!("{platform} · {bot}"),
            _ => platform.to_owned(),
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
        } else {
            self.inspected.set(None);
        }
        let _ = nsh_core::storage::Storage::set_item(OPEN_KEY, id.clone().unwrap_or_default());
        self.selected.set(id);
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
        let key = self.selected.get_untracked();
        let text = self
            .drafts
            .with_untracked(|d| d.get(key.as_deref().unwrap_or(NEW)).cloned())
            .unwrap_or_default();
        let text = text.trim().to_owned();
        if text.is_empty() {
            return;
        }
        match (&key, text.strip_prefix('/')) {
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
                    attachments: Vec::new(),
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
