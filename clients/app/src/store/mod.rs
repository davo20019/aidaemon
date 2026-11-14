//! The app's state: the shared reducer's [`State`] (built only from the
//! gateway's frames), the connection, what the owner has selected and
//! typed, and each turn's activity as it happened. Pages read it through
//! these methods and change it only through them.

use std::collections::{BTreeMap, BTreeSet};

use aidaemon_client_core::machine::{DropKind, Update, request_id};
use aidaemon_client_core::state::{Line, LiveTurn, SessionView, State};
use aidaemon_client_protocol::{
    Approval, ApprovalDecision, CommandResult, Command, Event, ServerFrame, TaskSummary,
    TurnStatus,
};
use nsh_core::prelude::*;

/// The connection to the daemon, as the owner should see it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Connection {
    #[default]
    Connecting,
    Online,
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
    outbox: Stored<Option<Outbox>>,
    /// Commands queued before the link was up.
    queued: Stored<Vec<(String, Command)>>,
    /// Sessions whose history was asked for.
    fetched: Stored<BTreeSet<String>>,
    /// The request of a message that starts a new session.
    starting: Stored<Option<String>>,
    next_activity: Stored<u64>,
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
            outbox: stored(None),
            queued: stored(Vec::new()),
            fetched: stored(BTreeSet::new()),
            starting: stored(None),
            next_activity: stored(1),
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
            self.queued.with(|queued| queued.push((id.clone(), command)));
        }
        id
    }

    // ----- What the link reports -----------------------------------------

    pub fn update(&self, update: Update) {
        match update {
            Update::Connecting => {
                if self.connection.with_untracked(|c| !matches!(c, Connection::Offline { .. })) {
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
            && self.starting.with(|s| s.as_deref() == Some(request_id.as_str()))
        {
            self.starting.replace(None);
            let draft = self.drafts.with_untracked(|d| d.get(NEW).cloned());
            self.drafts.update(|d| {
                d.remove(NEW);
                if let Some(draft) = draft.filter(|d| !d.is_empty()) {
                    d.insert(session_id.clone(), draft);
                }
            });
            self.fetched.with(|f| f.insert(session_id.clone()));
            self.selected.set(Some(session_id.clone()));
        }
        if let ServerFrame::Event { event, .. } = &frame {
            self.note_activity(event);
        }
        self.state.update(|state| state.apply(frame));
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
                TurnStatus::Thinking => {
                    push(turn_id, ActivityKind::Thinking, "Thinking".into(), String::new())
                }
                TurnStatus::ToolStarted { name, summary } => {
                    push(turn_id, ActivityKind::Tool, name.clone(), summary.clone())
                }
                TurnStatus::ToolProgress { name, chunk } => {
                    push(turn_id, ActivityKind::Output, name.clone(), chunk.clone())
                }
                TurnStatus::ToolCompleted { name, summary } => {
                    push(turn_id, ActivityKind::ToolDone, name.clone(), summary.clone())
                }
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

    pub fn is_online(&self) -> bool {
        self.connection.with(|c| *c == Connection::Online)
    }

    /// Session ids, newest activity first.
    pub fn session_ids(&self) -> Vec<String> {
        self.state.with(|s| s.order.clone())
    }

    pub fn title(&self, id: &str) -> String {
        self.state.with(|s| s.title(id))
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
        self.state
            .with(|s| s.sessions.get(id).map(|v| v.lines.clone()).unwrap_or_default())
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

    pub fn draft(&self) -> String {
        let key = self.selected.get().unwrap_or_default();
        self.drafts.with(|d| d.get(&key).cloned().unwrap_or_default())
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
        if let Some(id) = &id {
            self.load_history(id);
            self.inspected.set(self.latest_turn(id));
        } else {
            self.inspected.set(None);
        }
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
        self.drafts.update(|d| {
            d.remove(key.as_deref().unwrap_or(NEW));
        });
    }

    pub fn stop(&self, session: &str) {
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

    pub fn dismiss_notice(&self) {
        self.state.update(|s| s.notice = None);
    }
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
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
