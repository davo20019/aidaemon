//! The screen: sessions on the left, the chat in the middle, running work on
//! the right, the oldest waiting approval as a card above the message box,
//! and a status row that always says what state the connection is in. Keys
//! turn into protocol commands; nothing here talks to the network.
//!
//! The message box owns the keyboard. The common actions are one key away:
//! Enter sends, Esc stops the session's work, `y` / `s` / `a` / `n` answer
//! the card (only with the box empty, and only once the card has been on
//! screen for [`ARM_AFTER`] unchanged, so a card that appears never takes a
//! key meant for something else), Ctrl-P opens the palette of sessions and
//! commands, Ctrl-N starts a session, Alt-Up/Down switch sessions.
//!
//! The agent may move the view (`view_action` events): open a session,
//! bring a card into view, show a piece of work, focus a pane. Each move
//! shows a one-line note of what moved and why (Ctrl-B goes back, Esc or
//! any key dismisses it); while the owner has text in the box the move is
//! only offered (Ctrl-G takes it), so a message never lands in a session
//! the agent picked. A move never answers anything: a card it brings up is
//! armed afresh, and it can only ever ask for history.

use std::time::{Duration, Instant};

use aidaemon_client_core::link::{DropKind, Update};
use aidaemon_client_core::state::{Line, State, ViewRequest};
use aidaemon_client_protocol::{
    Approval, ApprovalDecision, AskReason, Capability, CardAction, Command, CommandResult, Impact,
    Pane, Risk, ServerFrame, ViewAction,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

/// What this client asks for in `hello`.
pub const CAPABILITIES: &[Capability] = &[Capability::ViewActions];

/// How long a note of a view move stays up.
const NOTE_FOR: Duration = Duration::from_secs(15);

/// How long a card must be on screen, unchanged, before a key answers it.
pub const ARM_AFTER: Duration = Duration::from_millis(600);

/// Slash commands the client handles itself; the rest run in the chat.
const LOCAL_SLASH: &[(&str, &str)] = &[
    ("new", "start a new session"),
    ("stop", "stop this session's work"),
    ("sessions", "switch session"),
    ("devices", "list paired devices"),
    ("help", "show the keys"),
];

const KEYS_HELP: &str = "Enter send · Esc stop · y/s/a/n answer the card · Ctrl-P sessions and \
                         commands · Ctrl-N new · Alt-Up/Down switch · Tab panes · Ctrl-C quit";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Sessions,
    Chat,
    Tasks,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Send(Command),
    Quit,
}

/// The connection as the status row tells it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkView {
    Connecting,
    Connected,
    Down {
        kind: DropKind,
        reason: String,
        retry_at: Instant,
    },
}

/// One entry of the palette.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaletteItem {
    NewSession,
    Stop,
    GoToApproval,
    Devices,
    Quit,
    Session(String),
}

/// Where the view was before a move, for Ctrl-B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Back {
    pub selected: Option<String>,
    pub focus: Focus,
    pub shown_approval: Option<String>,
}

/// The one-line note of a view move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub text: String,
    pub back: Option<Back>,
    pub since: Instant,
}

#[derive(Debug, Default)]
pub struct Palette {
    pub query: String,
    pub index: usize,
}

#[derive(Debug)]
pub struct App {
    pub state: State,
    pub focus: Focus,
    /// The session the chat shows; `None`: the next message starts one.
    pub selected: Option<String>,
    pub task_index: usize,
    pub input: String,
    pub connected: bool,
    pub link: LinkView,
    pub palette: Option<Palette>,
    /// The card on screen and since when, for arming its keys.
    card: Option<(String, Instant)>,
    /// A line for the status row that a key clears.
    pub flash: Option<String>,
    /// The note of the last view move.
    pub note: Option<Note>,
    /// A move offered while the owner was typing (Ctrl-G takes it).
    pub held: Option<ViewRequest>,
    /// The approval a view move brought up, shown instead of the oldest
    /// while it waits.
    pub shown_approval: Option<String>,
    /// Sessions whose history was asked for.
    fetched: std::collections::HashSet<String>,
    /// Waiting for the session the next message starts.
    pub awaiting_new: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            state: State::default(),
            focus: Focus::Chat,
            selected: None,
            task_index: 0,
            input: String::new(),
            connected: false,
            link: LinkView::Connecting,
            palette: None,
            card: None,
            flash: None,
            note: None,
            held: None,
            shown_approval: None,
            fetched: Default::default(),
            awaiting_new: false,
        }
    }
}

fn key_of(decision: ApprovalDecision) -> (char, &'static str) {
    match decision {
        ApprovalDecision::ApproveOnce => ('y', "allow once"),
        ApprovalDecision::ApproveSession => ('s', "allow this session"),
        ApprovalDecision::ApproveAlways => ('a', "always allow"),
        ApprovalDecision::Deny => ('n', "deny"),
    }
}

impl App {
    /// Apply what the link reports; the commands to send in reply.
    pub fn update(&mut self, update: Update) -> Vec<Command> {
        let now = Instant::now();
        let mut commands = Vec::new();
        match update {
            Update::Connecting => {
                if !matches!(self.link, LinkView::Down { .. }) {
                    self.link = LinkView::Connecting;
                }
            }
            Update::Connected => {
                self.connected = true;
                self.link = LinkView::Connected;
                commands.push(Command::ListSessions {
                    cursor: None,
                    limit: 200,
                });
            }
            Update::Disconnected(disconnect) => {
                self.connected = false;
                self.link = LinkView::Down {
                    kind: disconnect.kind,
                    reason: disconnect.reason,
                    retry_at: now + disconnect.retry_in,
                };
            }
            Update::Frame(frame) => {
                let notice = self.state.notice.clone();
                self.state.apply(frame.clone());
                if self.state.notice != notice {
                    self.flash = self.state.notice.clone();
                }
                self.after_frame(&frame);
                if let Some(request) = self.state.take_view_request() {
                    commands.extend(self.view_request(request, now));
                }
            }
        }
        self.observe_card(now);
        commands
    }

    /// Time passes (redraw tick).
    pub fn tick(&mut self) {
        let now = Instant::now();
        if self
            .note
            .as_ref()
            .is_some_and(|note| now.duration_since(note.since) > NOTE_FOR)
        {
            self.note = None;
        }
        self.observe_card(now);
    }

    /// The card on screen: the one a view move brought up while it waits,
    /// else the oldest.
    pub fn shown_approval(&self) -> Option<&Approval> {
        self.shown_approval
            .as_ref()
            .and_then(|id| self.state.approvals.get(id))
            .or_else(|| self.state.next_approval())
    }

    /// Note which card is on screen, so its keys arm after [`ARM_AFTER`].
    pub fn observe_card(&mut self, now: Instant) {
        let shown = self.shown_approval().map(|a| a.id.clone());
        match (&self.card, shown) {
            (Some((id, _)), Some(shown)) if *id == shown => {}
            (_, Some(shown)) => self.card = Some((shown, now)),
            (_, None) => self.card = None,
        }
    }

    /// The card on screen, when a key may answer it at `now`.
    pub fn armed_card(&self, now: Instant) -> Option<&Approval> {
        let approval = self.shown_approval()?;
        let (id, since) = self.card.as_ref()?;
        (*id == approval.id && now.duration_since(*since) >= ARM_AFTER).then_some(approval)
    }

    /// A view move the agent asked for: taken now, or only offered while
    /// the owner has text in the box. Returns the reads it needs (history);
    /// a view move never sends anything else.
    pub fn view_request(&mut self, request: ViewRequest, now: Instant) -> Vec<Command> {
        if !self.input.is_empty() || self.palette.is_some() {
            self.note = Some(Note {
                text: format!(
                    "The assistant suggests: {}{}. Ctrl-G to go there.",
                    self.describe(&request.action),
                    because(&request.reason)
                ),
                back: None,
                since: now,
            });
            self.held = Some(request);
            return Vec::new();
        }
        self.held = None;
        self.move_view(request, now)
    }

    fn describe(&self, action: &ViewAction) -> String {
        match action {
            ViewAction::OpenSession { session_id } => {
                format!("open {}", self.state.title(session_id))
            }
            ViewAction::ShowApproval { approval_id } => match self.state.approvals.get(approval_id)
            {
                Some(approval) => format!("the approval \"{}\"", approval.summary),
                None => "an approval that is no longer waiting".into(),
            },
            ViewAction::ShowTask { task_id } => match self.state.tasks.get(task_id) {
                Some(task) => format!("work #{} {}", task.number, task.description),
                None => "work that is no longer running".into(),
            },
            ViewAction::FocusPane { pane } => format!("the {pane:?} pane").to_lowercase(),
            ViewAction::NewSession => "a new session".into(),
            ViewAction::ShowDevices => "your paired devices".into(),
            ViewAction::Unknown => "something this version cannot show".into(),
        }
    }

    fn move_view(&mut self, request: ViewRequest, now: Instant) -> Vec<Command> {
        let back = Back {
            selected: self.selected.clone(),
            focus: self.focus,
            shown_approval: self.shown_approval.clone(),
        };
        let what = self.describe(&request.action);
        let mut commands = Vec::new();
        let mut text = format!("Moved to {what}");
        match request.action {
            ViewAction::OpenSession { session_id } => {
                if let Some(Action::Send(command)) = self.open_session(Some(session_id)) {
                    commands.push(command);
                }
            }
            ViewAction::ShowApproval { approval_id } => {
                if self.state.approvals.contains_key(&approval_id) {
                    self.shown_approval = Some(approval_id.clone());
                    // A card a move brings up is armed afresh.
                    self.card = Some((approval_id, now));
                    text = format!("Showing {what}: read it, then y / n");
                } else {
                    text = format!("Could not show {what}");
                }
            }
            ViewAction::ShowTask { task_id } => {
                if let Some(at) = self.state.running().iter().position(|t| t.id == task_id) {
                    self.task_index = at;
                    self.focus = Focus::Tasks;
                } else {
                    text = format!("Could not show {what}");
                }
            }
            ViewAction::FocusPane { pane } => match pane {
                Pane::Sessions => self.focus = Focus::Sessions,
                Pane::Chat => self.focus = Focus::Chat,
                Pane::Tasks => self.focus = Focus::Tasks,
                // Cards are answered from the message box; it keeps the keys.
                Pane::Approvals => {
                    self.focus = Focus::Chat;
                    text = "The approval card is above the message box".into();
                }
                Pane::Unknown => return Vec::new(),
            },
            ViewAction::NewSession => {
                self.selected = None;
                self.focus = Focus::Chat;
            }
            ViewAction::ShowDevices => {
                // Listing them is the owner's own command: one Enter away.
                self.input = "/devices".into();
                self.focus = Focus::Chat;
                text = "Press Enter to list your paired devices".into();
            }
            ViewAction::Unknown => return Vec::new(),
        }
        self.note = Some(Note {
            text: format!(
                "{text}{}. Ctrl-B back · Esc dismiss",
                because(&request.reason)
            ),
            back: Some(back),
            since: now,
        });
        commands
    }

    /// After a frame: follow a session this client just started.
    pub fn after_frame(&mut self, frame: &ServerFrame) {
        if let ServerFrame::Ack {
            result:
                CommandResult::MessageAccepted {
                    session_id,
                    new_session: true,
                },
            ..
        } = frame
            && self.awaiting_new {
                self.awaiting_new = false;
                self.selected = Some(session_id.clone());
                self.fetched.insert(session_id.clone());
            }
    }

    /// Show session `id`; the history request when it was never fetched.
    pub fn open_session(&mut self, id: Option<String>) -> Option<Action> {
        self.selected = id.clone();
        self.focus = Focus::Chat;
        let id = id?;
        // History is fetched once per session.
        if !self.fetched.insert(id.clone()) {
            return None;
        }
        Some(Action::Send(Command::GetHistory {
            session_id: id,
            before: None,
            limit: 100,
        }))
    }

    fn step_session(&mut self, delta: isize) -> Option<Action> {
        let order = &self.state.order;
        if order.is_empty() {
            return None;
        }
        let at = self
            .selected
            .as_ref()
            .and_then(|id| order.iter().position(|o| o == id));
        let next = match at {
            None => 0,
            Some(at) => (at as isize + delta).clamp(0, order.len() as isize - 1) as usize,
        };
        let id = order[next].clone();
        self.open_session(Some(id))
    }

    fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Sessions => Focus::Chat,
            Focus::Chat => Focus::Tasks,
            Focus::Tasks => Focus::Sessions,
        };
    }

    /// Whether the selected session has work running.
    fn busy(&self) -> bool {
        self.selected.as_ref().is_some_and(|id| {
            self.state
                .sessions
                .get(id)
                .is_some_and(|v| v.live.is_some())
                || self.state.tasks.values().any(|t| &t.session_id == id)
        })
    }

    fn stop(&mut self) -> Option<Action> {
        if !self.busy() {
            self.flash = Some("Nothing is running in this session.".into());
            return None;
        }
        let session_id = self.selected.clone()?;
        Some(Action::Send(Command::StopTurn { session_id }))
    }

    /// What a key does, now.
    pub fn key(&mut self, key: KeyEvent) -> Option<Action> {
        self.key_at(key, Instant::now())
    }

    /// What a key does at `now` (tests pass the time).
    pub fn key_at(&mut self, key: KeyEvent, now: Instant) -> Option<Action> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        self.flash = None;
        if self.palette.is_some() {
            return self.palette_key(key);
        }
        match key.code {
            KeyCode::Char('b') if ctrl && self.note.as_ref().is_some_and(|n| n.back.is_some()) => {
                if let Some(back) = self.note.take().and_then(|note| note.back) {
                    self.selected = back.selected;
                    self.focus = back.focus;
                    self.shown_approval = back.shown_approval;
                    self.observe_card(now);
                }
                return None;
            }
            KeyCode::Char('g') if ctrl && self.held.is_some() => {
                let request = self.held.take()?;
                self.note = None;
                return self
                    .move_view(request, now)
                    .into_iter()
                    .next()
                    .map(Action::Send);
            }
            KeyCode::Esc if self.note.is_some() => {
                self.note = None;
                self.held = None;
                return None;
            }
            _ => {
                // Any other key dismisses the note and does what it does.
                self.note = None;
            }
        }
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.input.is_empty() {
                    return Some(Action::Quit);
                }
                self.input.clear();
                return None;
            }
            KeyCode::Char('n') if ctrl => return self.open_session(None),
            KeyCode::Char('p') if ctrl => {
                self.palette = Some(Palette::default());
                return None;
            }
            KeyCode::Up if alt => return self.step_session(-1),
            KeyCode::Down if alt => return self.step_session(1),
            KeyCode::Esc => {
                if self.focus != Focus::Chat {
                    self.focus = Focus::Chat;
                    return None;
                }
                if !self.input.is_empty() {
                    return None;
                }
                return self.stop();
            }
            KeyCode::Tab if self.focus == Focus::Chat && self.completing() => {
                if let Some((name, _)) = self.completions().first() {
                    self.input = format!("/{name} ");
                }
                return None;
            }
            KeyCode::Tab => {
                self.cycle_focus();
                return None;
            }
            _ => {}
        }
        match self.focus {
            Focus::Sessions => match key.code {
                KeyCode::Up => self
                    .step_session(-1)
                    .inspect(|_| self.focus = Focus::Sessions),
                KeyCode::Down => self
                    .step_session(1)
                    .inspect(|_| self.focus = Focus::Sessions),
                KeyCode::Enter => {
                    self.focus = Focus::Chat;
                    None
                }
                _ => None,
            },
            Focus::Tasks => {
                let running = self.state.running();
                match key.code {
                    KeyCode::Up => {
                        self.task_index = self.task_index.saturating_sub(1);
                        None
                    }
                    KeyCode::Down => {
                        self.task_index =
                            (self.task_index + 1).min(running.len().saturating_sub(1));
                        None
                    }
                    KeyCode::Char('c') | KeyCode::Delete => {
                        running.get(self.task_index).map(|task| {
                            Action::Send(Command::CancelTask {
                                task_id: task.id.clone(),
                            })
                        })
                    }
                    _ => None,
                }
            }
            Focus::Chat => match key.code {
                KeyCode::Enter => self.submit(),
                KeyCode::Backspace => {
                    self.input.pop();
                    None
                }
                KeyCode::Char(pressed) if self.input.is_empty() && !ctrl && !alt => {
                    if let Some(approval) = self.armed_card(now)
                        && let Some(decision) = approval
                            .options
                            .iter()
                            .find(|decision| key_of(**decision).0 == pressed)
                        {
                            return Some(Action::Send(Command::ResolveApproval {
                                approval_id: approval.id.clone(),
                                decision: *decision,
                            }));
                        }
                    self.input.push(pressed);
                    None
                }
                KeyCode::Char(c) if !ctrl => {
                    self.input.push(c);
                    None
                }
                _ => None,
            },
        }
    }

    /// Whether a move is waiting for Ctrl-G.
    pub fn holds_move(&self) -> bool {
        self.held.is_some()
    }

    fn completing(&self) -> bool {
        self.input.starts_with('/') && !self.input.contains(' ')
    }

    fn completions(&self) -> Vec<(&'static str, &'static str)> {
        let typed = self.input.trim_start_matches('/').to_lowercase();
        LOCAL_SLASH
            .iter()
            .copied()
            .filter(|(name, _)| name.starts_with(&typed))
            .collect()
    }

    /// The palette's entries for its query, in order.
    pub fn palette_items(&self) -> Vec<(PaletteItem, String)> {
        let mut items = vec![
            (PaletteItem::NewSession, "New session  (Ctrl-N)".to_string()),
            (
                PaletteItem::Stop,
                "Stop this session's work  (Esc)".to_string(),
            ),
        ];
        if let Some(approval) = self.state.next_approval() {
            items.push((
                PaletteItem::GoToApproval,
                format!("Go to the waiting approval: {}", approval.summary),
            ));
        }
        items.push((PaletteItem::Devices, "Show paired devices".to_string()));
        items.push((PaletteItem::Quit, "Quit  (Ctrl-C)".to_string()));
        for id in &self.state.order {
            items.push((PaletteItem::Session(id.clone()), self.state.title(id)));
        }
        let query = self
            .palette
            .as_ref()
            .map(|p| p.query.to_lowercase())
            .unwrap_or_default();
        items
            .into_iter()
            .filter(|(_, label)| {
                let label = label.to_lowercase();
                query.split_whitespace().all(|word| label.contains(word))
            })
            .collect()
    }

    fn palette_key(&mut self, key: KeyEvent) -> Option<Action> {
        let items = self.palette_items();
        let palette = self.palette.as_mut()?;
        match key.code {
            KeyCode::Esc => self.palette = None,
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.palette = None
            }
            KeyCode::Up => palette.index = palette.index.saturating_sub(1),
            KeyCode::Down => palette.index = (palette.index + 1).min(items.len().saturating_sub(1)),
            KeyCode::Backspace => {
                palette.query.pop();
                palette.index = 0;
            }
            KeyCode::Char(c) => {
                palette.query.push(c);
                palette.index = 0;
            }
            KeyCode::Enter => {
                let chosen = items.get(palette.index).map(|(item, _)| item.clone());
                self.palette = None;
                return chosen.and_then(|item| self.run(item));
            }
            _ => {}
        }
        None
    }

    fn run(&mut self, item: PaletteItem) -> Option<Action> {
        match item {
            PaletteItem::NewSession => self.open_session(None),
            PaletteItem::Stop => self.stop(),
            PaletteItem::GoToApproval => {
                let session = self.state.next_approval()?.session_id.clone();
                self.open_session(Some(session))
            }
            PaletteItem::Devices => self.slash("devices", ""),
            PaletteItem::Quit => Some(Action::Quit),
            PaletteItem::Session(id) => self.open_session(Some(id)),
        }
    }

    fn slash(&mut self, name: &str, args: &str) -> Option<Action> {
        match name {
            "new" => return self.open_session(None),
            "stop" => return self.stop(),
            "sessions" => {
                self.palette = Some(Palette::default());
                return None;
            }
            "help" => {
                self.flash = Some(KEYS_HELP.into());
                return None;
            }
            _ => {}
        }
        let Some(session_id) = self
            .selected
            .clone()
            .or_else(|| self.state.order.first().cloned())
        else {
            self.flash = Some("Open a session first: commands run in a session.".into());
            return None;
        };
        Some(Action::Send(Command::RunSlash {
            session_id,
            command: name.to_string(),
            args: args.to_string(),
        }))
    }

    fn submit(&mut self) -> Option<Action> {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return None;
        }
        if let Some(command) = text.strip_prefix('/') {
            self.input.clear();
            let (name, args) = command.split_once(' ').unwrap_or((command, ""));
            return self.slash(name, args);
        }
        let writable = self.selected.as_ref().is_none_or(|id| {
            self.state
                .sessions
                .get(id)
                .and_then(|v| v.summary.as_ref())
                .is_none_or(|s| s.writable)
        });
        if !writable {
            self.flash = Some(
                "That session belongs to another chat: reply there, or Ctrl-N for a new one."
                    .into(),
            );
            return None;
        }
        self.input.clear();
        self.awaiting_new = self.selected.is_none();
        Some(Action::Send(Command::SendMessage {
            session_id: self.selected.clone(),
            text,
            attachments: Vec::new(),
        }))
    }

    /// The connection state in words.
    pub fn link_text(&self) -> String {
        match &self.link {
            LinkView::Connecting => "Connecting...".into(),
            LinkView::Connected => "Connected".into(),
            LinkView::Down {
                kind,
                reason,
                retry_at,
            } => {
                let seconds = retry_at
                    .saturating_duration_since(Instant::now())
                    .as_secs()
                    .max(1);
                match kind {
                    DropKind::DaemonNotRunning => {
                        format!("Daemon not running · retrying in {seconds} s")
                    }
                    DropKind::Unauthorized => format!("Signed out: {reason}"),
                    DropKind::VersionUnsupported => format!("Version mismatch: {reason}"),
                    DropKind::Dropped => format!("Reconnecting in {seconds} s · {reason}"),
                }
            }
        }
    }

    fn waiting_text(&self) -> String {
        let waiting = self.state.approvals.len();
        let running = self.state.tasks.len();
        let mut text = match waiting {
            0 => "nothing waiting".to_string(),
            1 => "1 approval waiting".to_string(),
            n => format!("{n} approvals waiting"),
        };
        if running > 0 {
            text.push_str(&format!(" · {running} running"));
        }
        text
    }

    fn hint_text(&self) -> String {
        if self.completing() {
            let names = self.completions();
            if names.is_empty() {
                return "Runs in the chat as a slash command".into();
            }
            let list: Vec<String> = names
                .iter()
                .map(|(name, what)| format!("/{name}: {what}"))
                .collect();
            return format!("{} · Tab completes", list.join(" · "));
        }
        if let Some(flash) = &self.flash {
            return flash.clone();
        }
        if self.input.is_empty() && self.armed_card(Instant::now()).is_some() {
            return "Press y / s / a / n to answer the card".into();
        }
        "Enter send · Esc stop · Ctrl-P sessions and commands · Ctrl-N new · /help".into()
    }

    pub fn draw(&self, frame: &mut Frame) {
        let card = self.shown_approval();
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),
                Constraint::Length(if self.note.is_some() { 1 } else { 0 }),
                Constraint::Length(if card.is_some() { 10 } else { 0 }),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(frame.area());
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(24),
                Constraint::Min(30),
                Constraint::Percentage(26),
            ])
            .split(rows[0]);
        self.draw_sessions(frame, columns[0]);
        self.draw_chat(frame, columns[1]);
        self.draw_tasks(frame, columns[2]);
        if let Some(note) = &self.note {
            frame.render_widget(
                Paragraph::new(TextLine::from(Span::styled(
                    format!(" ↪ {}", note.text),
                    Style::default().fg(Color::Black).bg(Color::Cyan),
                ))),
                rows[1],
            );
        }
        if let Some(approval) = card {
            self.draw_card(frame, rows[2], approval);
        }
        let input = Paragraph::new(self.input.as_str()).block(self.block(
            Focus::Chat,
            if self.selected.is_some() {
                "Message"
            } else {
                "Message (new session)"
            },
        ));
        frame.render_widget(input, rows[3]);
        self.draw_status(frame, rows[4]);
        if self.palette.is_some() {
            self.draw_palette(frame);
        }
    }

    fn draw_status(&self, frame: &mut Frame, area: Rect) {
        let link_style = match self.link {
            LinkView::Connected => Style::default().fg(Color::Green),
            LinkView::Connecting => Style::default().fg(Color::DarkGray),
            LinkView::Down { .. } => Style::default().fg(Color::Yellow),
        };
        let waiting_style = if self.state.approvals.is_empty() {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::Yellow)
        };
        let line = TextLine::from(vec![
            Span::styled(format!(" {} ", self.link_text()), link_style),
            Span::raw("│ "),
            Span::styled(self.waiting_text(), waiting_style),
            Span::raw(" │ "),
            Span::styled(self.hint_text(), Style::default().fg(Color::DarkGray)),
        ]);
        frame.render_widget(Paragraph::new(line), area);
    }

    fn block(&self, focus: Focus, title: &str) -> Block<'static> {
        let style = if self.focus == focus {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default()
        };
        Block::default()
            .borders(Borders::ALL)
            .border_style(style)
            .title(title.to_string())
    }

    fn draw_sessions(&self, frame: &mut Frame, area: Rect) {
        let mut items: Vec<ListItem> = self
            .state
            .order
            .iter()
            .map(|id| {
                let busy = self
                    .state
                    .sessions
                    .get(id)
                    .is_some_and(|v| v.live.is_some());
                ListItem::new(format!(
                    "{}{}",
                    if busy { "● " } else { "  " },
                    self.state.title(id)
                ))
            })
            .collect();
        if items.is_empty() {
            items.push(ListItem::new(Span::styled(
                "No sessions yet",
                Style::default().fg(Color::DarkGray),
            )));
        }
        let mut list_state = ListState::default();
        list_state.select(
            self.selected
                .as_ref()
                .and_then(|id| self.state.order.iter().position(|o| o == id)),
        );
        let list = List::new(items)
            .block(self.block(Focus::Sessions, "Sessions (Alt-Up/Down)"))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
        frame.render_stateful_widget(list, area, &mut list_state);
    }

    fn draw_chat(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<TextLine> = Vec::new();
        let view = self
            .selected
            .as_ref()
            .and_then(|id| self.state.sessions.get(id));
        if let Some(view) = view {
            for line in &view.lines {
                match line {
                    Line::User { text, origin } => {
                        let who = if origin.is_empty() || origin == "app" {
                            "You".to_string()
                        } else {
                            format!("You ({origin})")
                        };
                        lines.push(TextLine::from(Span::styled(
                            format!("{who}: "),
                            Style::default().fg(Color::Green),
                        )));
                        lines.extend(text.lines().map(|l| TextLine::from(l.to_string())));
                    }
                    Line::Assistant { text } => {
                        lines.push(TextLine::from(Span::styled(
                            "Assistant:",
                            Style::default().fg(Color::Magenta),
                        )));
                        lines.extend(text.lines().map(|l| TextLine::from(l.to_string())));
                    }
                    Line::Note { text } => lines.push(TextLine::from(Span::styled(
                        text.clone(),
                        Style::default().fg(Color::Yellow),
                    ))),
                }
                lines.push(TextLine::from(""));
            }
            if let Some(live) = &view.live {
                if !live.streamed.is_empty() {
                    lines.extend(live.streamed.lines().map(|l| TextLine::from(l.to_string())));
                }
                lines.push(TextLine::from(Span::styled(
                    format!(
                        "{} · Esc to stop",
                        live.status.clone().unwrap_or_else(|| "Working...".into())
                    ),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            if view.lines.is_empty() && view.live.is_none() {
                lines.push(TextLine::from(Span::styled(
                    "Nothing in this session yet.",
                    Style::default().fg(Color::DarkGray),
                )));
            }
        } else if self.state.order.is_empty() {
            lines.push(TextLine::from(
                "Nothing here yet. Type a message to start your first session.",
            ));
        } else {
            lines.push(TextLine::from(
                "Type a message to start a new session, or Ctrl-P to open one.",
            ));
        }
        // Keep the newest lines in view.
        let height = area.height.saturating_sub(2) as usize;
        let skip = lines.len().saturating_sub(height);
        let title = self
            .selected
            .as_ref()
            .map(|id| self.state.title(id))
            .unwrap_or_else(|| "New session".into());
        let chat = Paragraph::new(lines.split_off(skip))
            .block(self.block(Focus::Chat, &title))
            .wrap(Wrap { trim: false });
        frame.render_widget(chat, area);
    }

    fn draw_card(&self, frame: &mut Frame, area: Rect, approval: &Approval) {
        let risk_style = match approval.risk {
            Risk::Low => Style::default().fg(Color::Blue),
            Risk::Medium => Style::default().fg(Color::Yellow),
            Risk::High | Risk::Critical => Style::default().fg(Color::Red),
        };
        let mut lines = vec![TextLine::from(Span::styled(
            approval.summary.clone(),
            risk_style.add_modifier(Modifier::BOLD),
        ))];
        if let Some(card) = &approval.card {
            if let Some(impact) = &card.impact {
                let warning = match impact {
                    Impact::Deletes { targets } if !targets.is_empty() => {
                        format!("⚠ Deletes {}", targets.join(", "))
                    }
                    Impact::Deletes { .. } => "⚠ Deletes files".into(),
                    Impact::Sends => "⚠ Sends something to another service".into(),
                    Impact::Spends => "⚠ Spends money".into(),
                    Impact::WritesOutsideWorkArea => "⚠ Writes outside the work area".into(),
                    Impact::Unknown => String::new(),
                };
                if !warning.is_empty() {
                    lines.push(TextLine::from(Span::styled(warning, risk_style)));
                }
            }
            match &card.reason {
                Some(AskReason::ReadOutsideContent { check_says }) => {
                    lines.push(TextLine::from(check_says.clone().unwrap_or_else(|| {
                        "It read outside content, so this step needs your OK.".into()
                    })))
                }
                Some(AskReason::AfterADecline) => lines.push(TextLine::from(
                    "You said no to an earlier step, so this one asks first.",
                )),
                _ => {}
            }
            if let CardAction::Command { looks_inside, .. } = &card.action
                && !looks_inside.is_empty() {
                    lines.push(TextLine::from(format!(
                        "Only looks inside: {}",
                        looks_inside.join(", ")
                    )));
                }
        }
        for warning in &approval.warnings {
            lines.push(TextLine::from(format!("• {warning}")));
        }
        if let Some(detail) = &approval.detail {
            lines.push(TextLine::from(Span::styled(
                detail.lines().next().unwrap_or_default().to_string(),
                Style::default().fg(Color::DarkGray),
            )));
        }
        let armed = self.armed_card(Instant::now()).is_some() && self.input.is_empty();
        let keys: Vec<String> = approval
            .options
            .iter()
            .map(|decision| {
                let (key, label) = key_of(*decision);
                format!("[{key}] {label}")
            })
            .collect();
        lines.push(TextLine::from(Span::styled(
            keys.join("   "),
            if armed {
                Style::default().add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        )));
        let mut title = format!(
            "Approval needed ({:?} risk) in {}",
            approval.risk,
            self.state.title(&approval.session_id)
        );
        if let Some(channel) = &approval.channel {
            title.push_str(&format!(", also in {channel}"));
        }
        if self.state.approvals.len() > 1 {
            title.push_str(&format!(", {} waiting", self.state.approvals.len()));
        }
        let card = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(risk_style)
                    .title(title),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(card, area);
    }

    fn draw_tasks(&self, frame: &mut Frame, area: Rect) {
        let running = self.state.running();
        let mut items: Vec<ListItem> = running
            .iter()
            .map(|task| {
                ListItem::new(format!(
                    "#{} {} ({})",
                    task.number,
                    task.description,
                    self.state.title(&task.session_id)
                ))
            })
            .collect();
        if items.is_empty() {
            items.push(ListItem::new(Span::styled(
                "Nothing running",
                Style::default().fg(Color::DarkGray),
            )));
        }
        let mut list_state = ListState::default();
        if !running.is_empty() {
            list_state.select(Some(self.task_index.min(running.len() - 1)));
        }
        let list = List::new(items)
            .block(self.block(Focus::Tasks, "Running (c: cancel)"))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
        frame.render_stateful_widget(list, area, &mut list_state);
    }

    fn draw_palette(&self, frame: &mut Frame) {
        let area = frame.area();
        let width = area.width.saturating_sub(8).min(72);
        let height = area.height.saturating_sub(6).min(16);
        let rect = Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + 2,
            width,
            height,
        };
        frame.render_widget(Clear, rect);
        let query = self
            .palette
            .as_ref()
            .map(|p| p.query.clone())
            .unwrap_or_default();
        let items: Vec<ListItem> = self
            .palette_items()
            .into_iter()
            .map(|(_, label)| ListItem::new(label))
            .collect();
        let mut list_state = ListState::default();
        if !items.is_empty() {
            list_state.select(Some(
                self.palette
                    .as_ref()
                    .map_or(0, |p| p.index.min(items.len() - 1)),
            ));
        }
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan))
                    .title(format!("Sessions and commands: {query}_")),
            )
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
        frame.render_stateful_widget(list, rect, &mut list_state);
    }
}

/// `reason` as a clause of a note.
fn because(reason: &str) -> String {
    let reason = reason.trim().trim_end_matches('.');
    if reason.is_empty() {
        String::new()
    } else {
        format!(" ({reason})")
    }
}
