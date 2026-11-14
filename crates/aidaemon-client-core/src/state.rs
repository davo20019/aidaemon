//! What the client knows, built only from server frames: sessions, each
//! session's transcript, the approvals waiting, the work running, and the
//! last `seq` applied (the `resume_from` of the next connection).

use std::collections::BTreeMap;

use serde::Serialize;

use aidaemon_client_protocol::{
    Approval, CommandResult, Event, HistoryMessage, ServerFrame, SessionSummary, Snapshot,
    TaskState, TaskSummary, TurnState, TurnStatus, ViewAction,
};

/// One line of a transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Line {
    User {
        text: String,
        origin: String,
        /// When it was said (ms since the epoch); 0 when unknown.
        #[serde(default, skip_serializing_if = "is_zero")]
        at_ms: u64,
    },
    Assistant {
        text: String,
        #[serde(default, skip_serializing_if = "is_zero")]
        at_ms: u64,
    },
    /// A note from the client itself (a stopped turn, an error).
    Note { text: String },
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// A turn running now: its streamed text and its latest status.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct LiveTurn {
    pub turn_id: String,
    pub streamed: String,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SessionView {
    pub summary: Option<SessionSummary>,
    pub lines: Vec<Line>,
    pub live: Option<LiveTurn>,
}

/// The client's view. Its JSON form (`serde_json::to_value`) is what the
/// shared reducer fixtures compare, in Rust and in the web client.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct State {
    /// The last durable `seq` applied.
    pub last_seq: Option<u64>,
    pub sessions: BTreeMap<String, SessionView>,
    /// Session ids, newest activity first, as the list shows them.
    pub order: Vec<String>,
    pub approvals: BTreeMap<String, Approval>,
    pub tasks: BTreeMap<String, TaskSummary>,
    /// The latest error or notice from the gateway, shown in the status bar.
    pub notice: Option<String>,
    /// The latest view move the agent asked this client for, until the
    /// client takes it ([`State::take_view_request`]). Never touches the
    /// approvals, the work or the transcripts.
    pub view_request: Option<ViewRequest>,
}

/// A view move the agent asked for, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ViewRequest {
    pub action: ViewAction,
    pub reason: String,
}

/// A turn status in a few words.
pub fn status_text(status: &TurnStatus) -> String {
    match status {
        TurnStatus::Thinking => "Thinking...".into(),
        TurnStatus::ToolStarted { name, summary } if summary.is_empty() => format!("{name}..."),
        TurnStatus::ToolStarted { name, summary } => format!("{name}: {summary}..."),
        TurnStatus::ToolProgress { name, chunk } => {
            format!("{name}: {}", chunk.chars().take(80).collect::<String>())
        }
        TurnStatus::ToolCompleted { name, summary } => format!("✓ {name} {summary}"),
        TurnStatus::PlanStep {
            step,
            total,
            description,
            ..
        } => format!("Step {} of {total}: {description}", step + 1),
        TurnStatus::Checklist { .. } => "Updated the plan".into(),
        TurnStatus::Note { text } => text.clone(),
        TurnStatus::Unknown => "Working...".into(),
    }
}

impl State {
    fn session(&mut self, id: &str) -> &mut SessionView {
        if !self.sessions.contains_key(id) {
            self.order.insert(0, id.to_string());
        }
        self.sessions.entry(id.to_string()).or_default()
    }

    /// Move `id` to the top of the list.
    fn touch(&mut self, id: &str) {
        self.session(id);
        self.order.retain(|other| other != id);
        self.order.insert(0, id.to_string());
    }

    fn reset(&mut self, seq: u64, state: Snapshot) {
        let transcripts = std::mem::take(&mut self.sessions);
        self.order.clear();
        self.approvals.clear();
        self.tasks.clear();
        for summary in state.sessions {
            let id = summary.id.clone();
            let mut view = transcripts.get(&id).cloned().unwrap_or_default();
            view.summary = Some(summary);
            view.live = None;
            self.sessions.insert(id.clone(), view);
            self.order.push(id);
        }
        for approval in state.pending_approvals {
            self.approvals.insert(approval.id.clone(), approval);
        }
        for task in state.running_tasks {
            self.tasks.insert(task.id.clone(), task);
        }
        self.last_seq = Some(seq);
    }

    /// Apply one frame from the gateway. Durable events at or below the
    /// last `seq` applied are ignored, so a replay never doubles a line.
    pub fn apply(&mut self, frame: ServerFrame) {
        match frame {
            ServerFrame::Snapshot { seq, state }
            | ServerFrame::Ack {
                result: CommandResult::Snapshot { seq, state },
                ..
            } => self.reset(seq, state),
            ServerFrame::Event {
                seq,
                session_id,
                event,
                at_ms,
            } => {
                if let Some(seq) = seq {
                    if self.last_seq.is_some_and(|last| seq <= last) {
                        return;
                    }
                    self.last_seq = Some(seq);
                }
                self.apply_event(session_id, event, at_ms);
            }
            ServerFrame::Ack { result, .. } => self.apply_result(result),
            ServerFrame::Error { error, .. } => self.notice = Some(error.message),
            ServerFrame::Welcome { .. } | ServerFrame::Pong => {}
        }
    }

    fn apply_result(&mut self, result: CommandResult) {
        match result {
            CommandResult::Sessions { sessions, .. } => {
                for summary in sessions {
                    let id = summary.id.clone();
                    self.session(&id).summary = Some(summary);
                }
            }
            CommandResult::History {
                session_id,
                messages,
                ..
            } => self.set_history(&session_id, messages),
            CommandResult::Stopped { report } | CommandResult::Reply { text: report } => {
                self.notice = Some(report)
            }
            _ => {}
        }
    }

    fn set_history(&mut self, session_id: &str, messages: Vec<HistoryMessage>) {
        let view = self.session(session_id);
        // History is older than anything live: it goes first.
        let mut lines: Vec<Line> = messages
            .into_iter()
            .map(|m| match m.role.as_str() {
                "user" => Line::User {
                    text: m.text,
                    origin: String::new(),
                    at_ms: m.at_ms,
                },
                _ => Line::Assistant {
                    text: m.text,
                    at_ms: m.at_ms,
                },
            })
            .collect();
        lines.append(&mut view.lines);
        view.lines = lines;
    }

    fn live_turn<'a>(&'a mut self, turn_id: &str) -> Option<&'a mut LiveTurn> {
        self.sessions
            .values_mut()
            .filter_map(|view| view.live.as_mut())
            .find(|live| live.turn_id == turn_id)
    }

    fn apply_event(&mut self, session_id: Option<String>, event: Event, at_ms: u64) {
        match event {
            Event::SessionCreated { session } | Event::SessionUpdated { session } => {
                let id = session.id.clone();
                self.touch(&id);
                self.session(&id).summary = Some(session);
            }
            Event::UserMessage {
                session_id,
                text,
                origin,
                ..
            } => {
                self.touch(&session_id);
                self.session(&session_id).lines.push(Line::User {
                    text,
                    origin,
                    at_ms,
                });
            }
            Event::AssistantMessage { session_id, text } => {
                self.session(&session_id)
                    .lines
                    .push(Line::Assistant { text, at_ms });
            }
            Event::TurnStarted {
                turn_id,
                session_id,
                ..
            } => {
                self.session(&session_id).live = Some(LiveTurn {
                    turn_id,
                    ..LiveTurn::default()
                });
            }
            Event::TurnDelta { turn_id, text } => {
                if let Some(live) = self.live_turn(&turn_id) {
                    live.streamed.push_str(&text);
                }
            }
            Event::TurnStatus { turn_id, status } => {
                if let Some(live) = self.live_turn(&turn_id) {
                    live.status = Some(status_text(&status));
                }
            }
            Event::UserActionRequired { message, .. } => {
                if let Some(session_id) = session_id {
                    self.session(&session_id)
                        .lines
                        .push(Line::Note { text: message });
                }
            }
            Event::TurnCompleted {
                turn_id,
                text,
                state,
                error,
                ..
            } => {
                let Some(view) = self.sessions.values_mut().find(|view| {
                    view.live
                        .as_ref()
                        .is_some_and(|live| live.turn_id == turn_id)
                }) else {
                    // Its start was before this client's window.
                    if let Some(session_id) = session_id
                        && !text.is_empty()
                    {
                        self.session(&session_id)
                            .lines
                            .push(Line::Assistant { text, at_ms });
                    }
                    return;
                };
                view.live = None;
                if !text.is_empty() {
                    view.lines.push(Line::Assistant { text, at_ms });
                }
                match state {
                    TurnState::Cancelled => view.lines.push(Line::Note {
                        text: "Stopped.".into(),
                    }),
                    TurnState::Failed => view.lines.push(Line::Note {
                        text: format!("Failed: {}", error.unwrap_or_default()),
                    }),
                    _ => {}
                }
            }
            Event::ApprovalRequested { approval } => {
                self.approvals.insert(approval.id.clone(), approval);
            }
            Event::ApprovalResolved { approval_id, .. } => {
                self.approvals.remove(&approval_id);
            }
            Event::TaskUpdated { task } => {
                if matches!(task.state, TaskState::Running | TaskState::Cancelling) {
                    self.tasks.insert(task.id.clone(), task);
                } else {
                    self.tasks.remove(&task.id);
                }
            }
            // Kinds this client does not know are ignored.
            Event::ViewAction {
                action: ViewAction::Unknown,
                ..
            } => {}
            Event::ViewAction { action, reason } => {
                self.view_request = Some(ViewRequest { action, reason });
            }
            Event::GoalUpdated { .. } | Event::Unknown => {}
        }
    }

    /// The view move asked for, once.
    pub fn take_view_request(&mut self) -> Option<ViewRequest> {
        self.view_request.take()
    }

    /// The oldest approval waiting, which the card shows.
    pub fn next_approval(&self) -> Option<&Approval> {
        self.approvals.values().min_by_key(|a| a.requested_at_ms)
    }

    /// Running tasks, oldest first.
    pub fn running(&self) -> Vec<&TaskSummary> {
        let mut tasks: Vec<&TaskSummary> = self.tasks.values().collect();
        tasks.sort_by_key(|t| t.started_at_ms);
        tasks
    }

    pub fn title(&self, id: &str) -> String {
        let summary = self.sessions.get(id).and_then(|v| v.summary.as_ref());
        match summary {
            Some(SessionSummary {
                title: Some(title), ..
            }) => title.clone(),
            Some(summary) => format!("{} ({})", short(id), summary.channel),
            None => short(id),
        }
    }
}

fn short(id: &str) -> String {
    if id.chars().count() > 24 {
        format!("{}…", id.chars().take(23).collect::<String>())
    } else {
        id.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(seq: Option<u64>, event: Event) -> ServerFrame {
        ServerFrame::Event {
            seq,
            at_ms: 1,
            session_id: Some("app:s1".into()),
            event,
        }
    }

    #[test]
    fn a_turn_streams_then_completes_once_even_when_replayed() {
        let mut state = State::default();
        let frames = vec![
            event(
                Some(1),
                Event::UserMessage {
                    session_id: "app:s1".into(),
                    text: "Draft the synthetic report".into(),
                    attachments: Vec::new(),
                    origin: "app".into(),
                },
            ),
            event(
                Some(2),
                Event::TurnStarted {
                    turn_id: "t1".into(),
                    session_id: "app:s1".into(),
                    origin: "app".into(),
                },
            ),
            event(
                None,
                Event::TurnDelta {
                    turn_id: "t1".into(),
                    text: "Draft".into(),
                },
            ),
            event(
                None,
                Event::TurnStatus {
                    turn_id: "t1".into(),
                    status: TurnStatus::Thinking,
                },
            ),
        ];
        for frame in frames.clone() {
            state.apply(frame);
        }
        let live = state.sessions["app:s1"].live.clone().unwrap();
        assert_eq!(live.streamed, "Draft");
        assert_eq!(live.status.as_deref(), Some("Thinking..."));
        let done = event(
            Some(3),
            Event::TurnCompleted {
                turn_id: "t1".into(),
                text: "Drafted.".into(),
                attachments: Vec::new(),
                state: TurnState::Completed,
                error: None,
            },
        );
        state.apply(done.clone());
        // A replay of what was already applied changes nothing.
        for frame in frames {
            state.apply(frame);
        }
        state.apply(done);
        let view = &state.sessions["app:s1"];
        assert_eq!(view.lines.len(), 2, "{:?}", view.lines);
        assert!(view.live.is_none());
        assert_eq!(state.last_seq, Some(3));
    }

    #[test]
    fn a_snapshot_replaces_approvals_and_tasks() {
        let mut state = State::default();
        state.apply(event(
            Some(5),
            Event::ApprovalResolved {
                approval_id: "a0".into(),
                outcome: aidaemon_client_protocol::ApprovalOutcome::Expired,
                resolved_by: None,
            },
        ));
        state.apply(ServerFrame::Snapshot {
            seq: 9,
            state: Snapshot::default(),
        });
        assert_eq!(state.last_seq, Some(9));
        assert!(state.approvals.is_empty() && state.tasks.is_empty());
    }
}
