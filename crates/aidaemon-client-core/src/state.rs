//! What the client knows, built only from server frames: sessions, each
//! session's transcript, the approvals waiting, the work running, and the
//! last `seq` applied (the `resume_from` of the next connection).

use std::collections::BTreeMap;

use serde::Serialize;

use aidaemon_client_protocol::{
    Approval, AttachmentRef, CommandResult, ErrorCode, Event, Feature, HistoryMessage,
    ProjectSummary, RunSummary, ServedBy, ServerFrame, SessionSummary, SetupState, ShellInfo,
    Snapshot, SurfaceInfo, TaskState, TaskSummary, ToolCallInfo, TurnState, TurnStatus, ViewAction,
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
        /// Files sent with it.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<AttachmentRef>,
    },
    Assistant {
        text: String,
        #[serde(default, skip_serializing_if = "is_zero")]
        at_ms: u64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<AttachmentRef>,
        /// It only says the turn's work went on in the background; the
        /// report comes later as its own reply.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        handed_off: bool,
        /// The provider and model that wrote it, when the daemon said.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        served_by: Option<ServedBy>,
    },
    /// A note from the client itself (a stopped turn, an error).
    Note {
        text: String,
        /// The turn end this note reports (`cancelled`, `failed`); none
        /// for any other note. Retry reads this, never the text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ended: Option<TurnState>,
    },
    /// What the turn behind the next reply did: the tools it used, in
    /// order, and how long it took when known.
    Activity {
        tools: Vec<String>,
        #[serde(default, skip_serializing_if = "is_zero")]
        secs: u64,
        /// What each tool was given and gave back, in the same order, as
        /// far as known: a daemon's full record from history, the live
        /// status summaries for a turn seen running. May be shorter than
        /// `tools`.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        calls: Vec<ToolCallInfo>,
    },
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
    /// The tools the turn has started, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    /// What each started tool was doing (its start summary) and how it
    /// ended (its completion summary), in the same order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<ToolCallInfo>,
    /// When the turn started (ms since the epoch); 0 when unknown.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub started_ms: u64,
    /// It started while the conversation waited on work handed to the
    /// background: it is that work's report.
    #[serde(skip)]
    pub after_background: bool,
}

/// How long a turn runs before its end is worth a notification to an owner
/// who left the app.
pub const NOTABLE_TURN_MS: u64 = 60_000;

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SessionView {
    pub summary: Option<SessionSummary>,
    pub lines: Vec<Line>,
    pub live: Option<LiveTurn>,
    /// What the owner might ask next, offered under the last reply until
    /// they say something.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
    /// Its last turn handed work off to run in the background, whose
    /// report has not come yet: the conversation is working, not idle.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub background: bool,
    /// Its history has been read at least once: before that, an empty
    /// conversation is one still loading, not one with nothing in it.
    #[serde(skip)]
    pub history_read: bool,
}

/// The client's view. Its JSON form (`serde_json::to_value`) is what the
/// shared reducer fixtures (`tests/fixtures/reducer/`) compare.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct State {
    /// The last durable `seq` applied.
    pub last_seq: Option<u64>,
    pub sessions: BTreeMap<String, SessionView>,
    /// Session ids, newest activity first, as the list shows them.
    pub order: Vec<String>,
    pub approvals: BTreeMap<String, Approval>,
    pub tasks: BTreeMap<String, TaskSummary>,
    /// The owner's projects, by id (protocol 1.10).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub projects: BTreeMap<String, ProjectSummary>,
    /// The latest error or notice from the gateway, shown in the status bar.
    pub notice: Option<String>,
    /// The latest view move the agent asked this client for, until the
    /// client takes it ([`State::take_view_request`]). Never touches the
    /// approvals, the work or the transcripts.
    pub view_request: Option<ViewRequest>,
    /// First-run setup still to finish, as the daemon last said; `None`
    /// once it is set up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup: Option<SetupState>,
    /// Features the agent asked for and the owner has not answered.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feature_requests: Vec<FeatureRequest>,
    /// Live surfaces, by id (protocol 1.30, capability `surfaces`): the
    /// screens the agent works on that the owner can watch or control.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub surfaces: BTreeMap<String, SurfaceInfo>,
    /// Conversation shells, by conversation (protocol 1.42, capability
    /// `shells`): one per conversation at most.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub shells: BTreeMap<String, ShellInfo>,
    /// Helper runs, by id (protocol 1.46): sub-agents, launched coding
    /// agents and background commands, running or recently ended
    /// ([`MAX_ENDED_RUNS`] ended ones per conversation).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub runs: BTreeMap<String, RunSummary>,
}

/// How many ended runs a conversation keeps on show.
pub const MAX_ENDED_RUNS: usize = 5;

/// The agent needs a feature that is off.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FeatureRequest {
    pub session_id: String,
    pub feature: Feature,
    pub reason: String,
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
        TurnStatus::ToolStarted { name, summary, .. } if summary.is_empty() => {
            format!("{name}...")
        }
        TurnStatus::ToolStarted { name, summary, .. } => format!("{name}: {summary}..."),
        // The output's latest line as a reader sees it, never its escapes.
        TurnStatus::ToolProgress { name, chunk } => {
            let plain = crate::text::plain_output(chunk);
            let last = plain
                .lines()
                .rev()
                .find(|line| !line.trim().is_empty())
                .unwrap_or_default();
            format!(
                "{name}: {}",
                last.trim().chars().take(80).collect::<String>()
            )
        }
        TurnStatus::ToolCompleted { name, summary, .. } => format!("✓ {name} {summary}"),
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
    fn set_setup(&mut self, setup: SetupState) {
        self.setup = (setup.step != aidaemon_client_protocol::SetupStep::Done).then_some(setup);
    }

    /// The owner answered the request for `feature` (turned it on or not).
    pub fn settle_feature(&mut self, feature: Feature) {
        self.feature_requests.retain(|r| r.feature != feature);
    }

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
        self.surfaces = state
            .surfaces
            .into_iter()
            .map(|surface| (surface.id.clone(), surface))
            .collect();
        self.shells = state
            .shells
            .into_iter()
            .map(|shell| (shell.session_id.clone(), shell))
            .collect();
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
        self.runs.clear();
        for run in state.runs {
            self.put_run(run);
        }
        // What running turns have got to: the working line comes back after
        // a reload or a dropped connection, and later status events find
        // their turn.
        for mut turn in state.live_turns {
            // The app pairs a tool with its call by position: an older
            // daemon sends no calls, and the steps after the snapshot must
            // still land at their own index.
            if turn.calls.len() != turn.tools.len() {
                turn.calls = turn
                    .tools
                    .iter()
                    .map(|name| ToolCallInfo {
                        name: name.clone(),
                        ..ToolCallInfo::default()
                    })
                    .collect();
            }
            self.session(&turn.session_id).live = Some(LiveTurn {
                turn_id: turn.turn_id,
                streamed: turn.streamed,
                status: turn.status.as_ref().map(status_text),
                tools: turn.tools,
                calls: turn.calls,
                started_ms: turn.started_at_ms,
                // A turn joined from a snapshot: whether it reports
                // background work is not known here.
                after_background: false,
            });
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
            // A refused sign-in is the connection's to say (the link signs
            // in again, or goes offline as refused): not a note that stays
            // on the page after the link is back.
            ServerFrame::Error {
                request_id: None,
                error,
            } if matches!(
                error.code,
                ErrorCode::Unauthorized | ErrorCode::VersionUnsupported
            ) => {}
            ServerFrame::Error { error, .. } => self.notice = Some(error.message),
            ServerFrame::Welcome { setup, .. } => {
                self.setup = setup.filter(|s| s.step != aidaemon_client_protocol::SetupStep::Done)
            }
            ServerFrame::Pong => {}
        }
    }

    /// A note from the client in conversation `session` (not a turn's end).
    pub(crate) fn note(&mut self, session: &str, text: String) {
        self.session(session)
            .lines
            .push(Line::Note { text, ended: None });
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
            CommandResult::Setup { setup } => self.set_setup(setup),
            CommandResult::Projects { projects } => {
                self.projects = projects.into_iter().map(|p| (p.id.clone(), p)).collect();
            }
            CommandResult::Project { project } => {
                self.projects.insert(project.id.clone(), project);
            }
            _ => {}
        }
    }

    fn set_history(&mut self, session_id: &str, messages: Vec<HistoryMessage>) {
        let view = self.session(session_id);
        view.history_read = true;
        let mut lines: Vec<Line> = Vec::with_capacity(messages.len());
        for m in messages {
            match m.role.as_str() {
                "user" => lines.push(Line::User {
                    text: m.text,
                    origin: String::new(),
                    at_ms: m.at_ms,
                    attachments: m.attachments,
                }),
                _ => {
                    if !m.tools.is_empty() {
                        lines.push(Line::Activity {
                            tools: m.tools,
                            secs: 0,
                            calls: m.calls,
                        });
                    }
                    lines.push(Line::Assistant {
                        text: m.text,
                        at_ms: m.at_ms,
                        attachments: m.attachments,
                        handed_off: m.handed_off,
                        served_by: m.served_by,
                    });
                }
            }
        }
        view.lines = merge_history(lines, std::mem::take(&mut view.lines));
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
                attachments,
            } => {
                self.touch(&session_id);
                let view = self.session(&session_id);
                view.suggestions.clear();
                view.lines.push(Line::User {
                    text,
                    origin,
                    at_ms,
                    attachments,
                });
            }
            Event::AssistantMessage {
                session_id,
                text,
                suggestions,
                attachments,
            } => {
                let view = self.session(&session_id);
                view.lines.push(Line::Assistant {
                    text,
                    at_ms,
                    attachments,
                    handed_off: false,
                    served_by: None,
                });
                view.suggestions = suggestions;
                // A message after a hand-off is its report.
                view.background = false;
            }
            Event::SetupUpdated { setup } => self.set_setup(setup),
            Event::FeatureRequested {
                session_id,
                feature,
                reason,
            } => {
                self.feature_requests
                    .retain(|r| !(r.session_id == session_id && r.feature == feature));
                self.feature_requests.push(FeatureRequest {
                    session_id,
                    feature,
                    reason,
                });
            }
            Event::SurfaceUpdated { surface } => {
                self.surfaces.insert(surface.id.clone(), surface);
            }
            Event::SurfaceClosed { surface_id } => {
                self.surfaces.remove(&surface_id);
            }
            Event::ShellUpdated { shell } => {
                self.shells.insert(shell.session_id.clone(), shell);
            }
            Event::ShellClosed {
                shell_id,
                session_id,
            } => {
                if self
                    .shells
                    .get(&session_id)
                    .is_some_and(|shell| shell.id == shell_id)
                {
                    self.shells.remove(&session_id);
                }
            }
            Event::TurnStarted {
                turn_id,
                session_id,
                ..
            } => {
                let view = self.session(&session_id);
                let after_background = view.background;
                view.background = false;
                view.live = Some(LiveTurn {
                    turn_id,
                    started_ms: at_ms,
                    after_background,
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
                    match &status {
                        TurnStatus::ToolStarted { name, activity, .. } => {
                            live.tools.push(name.clone());
                            live.calls.push(ToolCallInfo {
                                name: name.clone(),
                                input: status.call_detail().unwrap_or_default().to_string(),
                                output: String::new(),
                                kind: *activity,
                            });
                        }
                        // The latest call of that name still running.
                        TurnStatus::ToolCompleted { name, .. } => {
                            if let Some(call) = live
                                .calls
                                .iter_mut()
                                .rev()
                                .find(|c| &c.name == name && c.output.is_empty())
                            {
                                call.output = status.call_detail().unwrap_or_default().to_string();
                            }
                        }
                        _ => {}
                    }
                    live.status = Some(status_text(&status));
                }
            }
            Event::UserActionRequired { message, .. } => {
                if let Some(session_id) = session_id {
                    self.session(&session_id).lines.push(Line::Note {
                        text: message,
                        ended: None,
                    });
                }
            }
            Event::TurnCompleted {
                turn_id,
                text,
                state,
                error,
                attachments,
                served_by,
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
                        self.session(&session_id).lines.push(Line::Assistant {
                            text,
                            at_ms,
                            attachments,
                            handed_off: state == TurnState::HandedOff,
                            served_by,
                        });
                    }
                    return;
                };
                if let Some(live) = view.live.take()
                    && !live.tools.is_empty()
                {
                    let secs = if live.started_ms > 0 && at_ms > live.started_ms {
                        (at_ms - live.started_ms) / 1000
                    } else {
                        0
                    };
                    view.lines.push(Line::Activity {
                        tools: live.tools,
                        calls: live.calls,
                        secs,
                    });
                }
                if !text.is_empty() || !attachments.is_empty() {
                    view.lines.push(Line::Assistant {
                        text,
                        at_ms,
                        attachments,
                        handed_off: state == TurnState::HandedOff,
                        served_by,
                    });
                }
                view.background = state == TurnState::HandedOff;
                match state {
                    TurnState::Cancelled => view.lines.push(Line::Note {
                        text: "Stopped.".into(),
                        ended: Some(TurnState::Cancelled),
                    }),
                    TurnState::Failed => view.lines.push(Line::Note {
                        text: format!("Failed: {}", error.unwrap_or_default()),
                        ended: Some(TurnState::Failed),
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
            Event::ProjectUpdated { project } => {
                self.projects.insert(project.id.clone(), project);
            }
            Event::ProjectDeleted { project_id } => {
                self.projects.remove(&project_id);
                for view in self.sessions.values_mut() {
                    if let Some(summary) = view.summary.as_mut()
                        && summary.project_id.as_deref() == Some(project_id.as_str())
                    {
                        summary.project_id = None;
                    }
                }
            }
            Event::TaskUpdated { task } => {
                if matches!(task.state, TaskState::Running | TaskState::Cancelling) {
                    self.tasks.insert(task.id.clone(), task);
                } else {
                    self.tasks.remove(&task.id);
                }
            }
            Event::RunUpdated { run } => self.put_run(run),
            // A step for a run this client never heard start (it joined
            // late) waits for the run's own record.
            Event::RunStep { run_id, step, .. } => {
                if let Some(run) = self.runs.get_mut(&run_id)
                    && run.phase.is_running()
                {
                    run.step = step;
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
            // The app reads the settings again (its store, not this state).
            Event::GoalUpdated { .. } | Event::SettingsUpdated { .. } | Event::Unknown => {}
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

    /// Keep `run`'s record, and only the newest [`MAX_ENDED_RUNS`] ended
    /// runs of its conversation.
    fn put_run(&mut self, run: RunSummary) {
        let session = run.session_id.clone();
        self.runs.insert(run.id.clone(), run);
        let mut ended: Vec<(u64, String)> = self
            .runs
            .values()
            .filter(|run| run.session_id == session && !run.phase.is_running())
            .map(|run| (run.ended_at_ms.unwrap_or(run.started_at_ms), run.id.clone()))
            .collect();
        ended.sort();
        let excess = ended.len().saturating_sub(MAX_ENDED_RUNS);
        for (_, id) in ended.into_iter().take(excess) {
            self.runs.remove(&id);
        }
    }

    /// A conversation's runs, oldest first.
    pub fn runs_for(&self, session_id: &str) -> Vec<&RunSummary> {
        let mut runs: Vec<&RunSummary> = self
            .runs
            .values()
            .filter(|run| run.session_id == session_id)
            .collect();
        runs.sort_by_key(|run| run.started_at_ms);
        runs
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
                archived: false,
                title: Some(title),
                ..
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

/// A said message's identity across history and the live stream: who said
/// it and its words (history has no event ids to match on).
fn said(line: &Line) -> Option<(bool, &str)> {
    match line {
        Line::User { text, .. } => Some((true, text)),
        Line::Assistant { text, .. } => Some((false, text)),
        _ => None,
    }
}

fn said_at(line: &Line) -> u64 {
    match line {
        Line::User { at_ms, .. } | Line::Assistant { at_ms, .. } => *at_ms,
        _ => 0,
    }
}

/// The transcript after a history read (`history`, the latest page, oldest
/// first) meets what the client already shows (`shown`).
///
/// - Nothing said shown yet: history goes first, before anything live.
/// - The newest shown message is in history: what history has after it is
///   what this client missed (a frozen page, a dropped stream); it goes
///   after, and the client's own notes and the live tool lines stay.
///   Shown lines after that point that history also has are not doubled.
/// - None shown is in history: the page is older than all of history's
///   window (it was away long), so history replaces it; else (the read
///   is older than what arrived live) history goes first.
fn merge_history(history: Vec<Line>, shown: Vec<Line>) -> Vec<Line> {
    if !shown.iter().any(|l| said(l).is_some()) {
        let mut lines = history;
        lines.extend(shown);
        return lines;
    }
    // The newest shown message history also has, and where.
    let found = shown.iter().enumerate().rev().find_map(|(i, line)| {
        let key = said(line)?;
        let at = history.iter().rposition(|h| said(h) == Some(key))?;
        Some((i, at))
    });
    let Some((upto, at)) = found else {
        let newest_shown = shown.iter().map(said_at).max().unwrap_or(0);
        let oldest_read = history
            .iter()
            .map(said_at)
            .filter(|&t| t > 0)
            .min()
            .unwrap_or(0);
        if newest_shown > 0 && oldest_read > 0 && newest_shown < oldest_read {
            return history;
        }
        let mut lines = history;
        lines.extend(shown);
        return lines;
    };
    let mut lines = shown;
    let mut tail = lines.split_off(upto + 1);
    // Notes already shown right after it were there before what was missed.
    let notes = tail
        .iter()
        .take_while(|l| matches!(l, Line::Note { .. }))
        .count();
    lines.extend(tail.drain(..notes));
    let missed: Vec<Line> = history.into_iter().skip(at + 1).collect();
    let known: Vec<(bool, String)> = missed
        .iter()
        .filter_map(|l| said(l).map(|(u, t)| (u, t.to_owned())))
        .collect();
    lines.extend(missed);
    // What was shown after that point: notes stay, messages history just
    // brought are not doubled, and a tool line goes with its message.
    let mut activity = None;
    for line in tail {
        match said(&line) {
            Some((user, text)) => {
                let doubled = known.iter().any(|(u, t)| *u == user && t == text);
                if doubled {
                    activity = None;
                } else {
                    lines.extend(activity.take());
                    lines.push(line);
                }
            }
            None if matches!(line, Line::Activity { .. }) => {
                lines.extend(activity.replace(line));
            }
            None => lines.push(line),
        }
    }
    lines.extend(activity);
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_client_protocol::ProtocolError;

    fn event(seq: Option<u64>, event: Event) -> ServerFrame {
        ServerFrame::Event {
            seq,
            at_ms: 1,
            session_id: Some("app:s1".into()),
            event,
        }
    }

    fn run(id: &str, phase: aidaemon_client_protocol::RunPhase, at: u64) -> RunSummary {
        RunSummary {
            id: id.into(),
            session_id: "app:s1".into(),
            label: "Claude Code: synthetic release".into(),
            phase,
            background: true,
            step: String::new(),
            started_at_ms: at,
            ended_at_ms: (!phase.is_running()).then_some(at + 1),
            took: None,
            reason: None,
            stopped_by_owner: false,
        }
    }

    /// A run is shown from its start, its step follows it while it runs,
    /// and it stays on show once it ended: the work after a turn is over is
    /// never silent.
    #[test]
    fn a_run_is_followed_from_start_to_end() {
        use aidaemon_client_protocol::RunPhase;
        let mut state = State::default();
        state.apply(event(
            Some(1),
            Event::RunUpdated {
                run: run("r1", RunPhase::Running, 10),
            },
        ));
        state.apply(event(
            None,
            Event::RunStep {
                run_id: "r1".into(),
                session_id: "app:s1".into(),
                step: "Running a command".into(),
                at_ms: 11,
            },
        ));
        assert_eq!(state.runs_for("app:s1")[0].step, "Running a command");
        // A step for a run never heard of is not invented into one.
        state.apply(event(
            None,
            Event::RunStep {
                run_id: "unknown".into(),
                session_id: "app:s1".into(),
                step: "Running a command".into(),
                at_ms: 12,
            },
        ));
        assert_eq!(state.runs.len(), 1);
        state.apply(event(
            Some(2),
            Event::RunUpdated {
                run: run("r1", RunPhase::Succeeded, 10),
            },
        ));
        let runs = state.runs_for("app:s1");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].phase, RunPhase::Succeeded);
        assert!(runs[0].step.is_empty());
        // A late step does not bring an ended run back to life.
        state.apply(event(
            None,
            Event::RunStep {
                run_id: "r1".into(),
                session_id: "app:s1".into(),
                step: "Running a command".into(),
                at_ms: 13,
            },
        ));
        assert!(state.runs_for("app:s1")[0].step.is_empty());
    }

    /// A conversation keeps only its newest ended runs; running ones always
    /// stay.
    #[test]
    fn only_the_newest_ended_runs_stay() {
        use aidaemon_client_protocol::RunPhase;
        let mut state = State::default();
        state.apply(event(
            Some(1),
            Event::RunUpdated {
                run: run("live", RunPhase::Running, 1),
            },
        ));
        for n in 0..(MAX_ENDED_RUNS as u64 + 3) {
            state.apply(event(
                Some(n + 2),
                Event::RunUpdated {
                    run: run(&format!("done-{n}"), RunPhase::Failed, 100 + n),
                },
            ));
        }
        let runs = state.runs_for("app:s1");
        assert_eq!(runs.len(), MAX_ENDED_RUNS + 1);
        assert!(runs.iter().any(|run| run.id == "live"));
        assert!(!runs.iter().any(|run| run.id == "done-0"));
    }

    /// A turn that hands its work to the background leaves the
    /// conversation working until the follow-up starts or the report comes.
    #[test]
    fn a_handed_off_turn_stays_working_until_its_report() {
        let mut state = State::default();
        let started = |turn: &str| {
            event(
                None,
                Event::TurnStarted {
                    turn_id: turn.into(),
                    session_id: "app:s1".into(),
                    origin: "app".into(),
                },
            )
        };
        let completed = |turn: &str, state: TurnState| {
            event(
                None,
                Event::TurnCompleted {
                    turn_id: turn.into(),
                    text: String::new(),
                    attachments: Vec::new(),
                    state,
                    error: None,
                    served_by: None,
                },
            )
        };
        state.apply(started("t1"));
        state.apply(completed("t1", TurnState::HandedOff));
        assert!(state.sessions["app:s1"].background);
        // The follow-up is a live turn of its own.
        state.apply(started("t2"));
        assert!(!state.sessions["app:s1"].background);
        state.apply(completed("t2", TurnState::Completed));
        assert!(!state.sessions["app:s1"].background);
        // Or the report comes as a message.
        state.apply(started("t3"));
        state.apply(completed("t3", TurnState::HandedOff));
        state.apply(event(
            Some(9),
            Event::AssistantMessage {
                session_id: "app:s1".into(),
                text: "The synthetic build passed.".into(),
                attachments: Vec::new(),
                suggestions: Vec::new(),
            },
        ));
        assert!(!state.sessions["app:s1"].background);
    }

    /// The reply of a handed-off turn is marked as only a launch line; a
    /// finished turn's reply is not.
    #[test]
    fn a_handed_off_reply_is_marked() {
        let mut state = State::default();
        for (turn, end) in [("t1", TurnState::HandedOff), ("t2", TurnState::Completed)] {
            state.apply(event(
                None,
                Event::TurnStarted {
                    turn_id: turn.into(),
                    session_id: "app:s1".into(),
                    origin: "app".into(),
                },
            ));
            state.apply(event(
                None,
                Event::TurnCompleted {
                    turn_id: turn.into(),
                    text: format!("Reply of {turn}"),
                    attachments: Vec::new(),
                    state: end,
                    error: None,
                    served_by: None,
                },
            ));
        }
        let marks: Vec<bool> = state.sessions["app:s1"]
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Assistant { handed_off, .. } => Some(*handed_off),
                _ => None,
            })
            .collect();
        assert_eq!(marks, [true, false]);
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
                served_by: None,
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

    /// A reply keeps the provider and model the daemon says wrote it.
    #[test]
    fn a_reply_keeps_who_wrote_it() {
        let mut state = State::default();
        let served = ServedBy {
            preset: "anthropic".into(),
            provider: "Anthropic".into(),
            model: "synthetic-model-1".into(),
        };
        state.apply(event(
            Some(1),
            Event::TurnStarted {
                turn_id: "t1".into(),
                session_id: "app:s1".into(),
                origin: "app".into(),
            },
        ));
        state.apply(event(
            Some(2),
            Event::TurnCompleted {
                turn_id: "t1".into(),
                text: "Done.".into(),
                attachments: Vec::new(),
                state: TurnState::Completed,
                error: None,
                served_by: Some(served.clone()),
            },
        ));
        let Some(Line::Assistant { served_by, .. }) = state.sessions["app:s1"].lines.last() else {
            panic!("{:?}", state.sessions["app:s1"].lines);
        };
        assert_eq!(served_by.as_ref(), Some(&served));
    }

    /// A turn seen running keeps what each tool was doing and how it ended,
    /// for its activity line to open to.
    #[test]
    fn a_finished_turn_keeps_each_tools_start_and_end() {
        let mut state = State::default();
        let status = |status: TurnStatus| {
            event(
                None,
                Event::TurnStatus {
                    turn_id: "t1".into(),
                    status,
                },
            )
        };
        state.apply(event(
            None,
            Event::TurnStarted {
                turn_id: "t1".into(),
                session_id: "app:s1".into(),
                origin: "app".into(),
            },
        ));
        state.apply(status(
            TurnStatus::tool_started(
                "running a command",
                "typechecking",
                Some(aidaemon_client_protocol::ToolKind::Run),
            )
            .with_detail("{\n  \"command\": \"npm run typecheck\"\n}"),
        ));
        state.apply(status(
            TurnStatus::tool_completed("running a command", "exit 0")
                .with_detail("Found 0 errors."),
        ));
        // An older daemon sends no arguments: the summary stands in.
        state.apply(status(TurnStatus::tool_started(
            "reading a file",
            "src/synthetic.ts",
            None,
        )));
        state.apply(event(
            Some(1),
            Event::TurnCompleted {
                turn_id: "t1".into(),
                text: "Typechecked.".into(),
                attachments: Vec::new(),
                state: TurnState::Completed,
                error: None,
                served_by: None,
            },
        ));
        let Line::Activity { tools, calls, .. } = &state.sessions["app:s1"].lines[0] else {
            panic!("{:?}", state.sessions["app:s1"].lines);
        };
        assert_eq!(tools.len(), 2);
        // A live call shows what history would: its arguments and result.
        assert_eq!(calls[0].input, "{\n  \"command\": \"npm run typecheck\"\n}");
        assert_eq!(calls[0].kind, Some(aidaemon_client_protocol::ToolKind::Run));
        assert_eq!(calls[0].output, "Found 0 errors.");
        assert_eq!(calls[1].input, "src/synthetic.ts");
        assert!(calls[1].output.is_empty());
    }

    /// History's full record of each call reaches the reply's activity line.
    #[test]
    fn history_brings_each_calls_input_and_output() {
        let mut state = State::default();
        state.set_history(
            "app:s1",
            vec![HistoryMessage {
                id: "m1".into(),
                role: "assistant".into(),
                text: "Read it.".into(),
                at_ms: 1,
                tools: vec!["read_file".into()],
                calls: vec![ToolCallInfo {
                    name: "read_file".into(),
                    input: "{\"path\": \"notes.md\"}".into(),
                    output: "Synthetic notes".into(),
                    kind: None,
                }],
                attachments: Vec::new(),
                handed_off: false,
                served_by: None,
            }],
        );
        let Line::Activity { calls, .. } = &state.sessions["app:s1"].lines[0] else {
            panic!("{:?}", state.sessions["app:s1"].lines);
        };
        assert_eq!(calls[0].output, "Synthetic notes");
    }

    #[test]
    fn a_snapshot_brings_back_what_a_running_turn_has_got_to() {
        let mut state = State::default();
        state.apply(ServerFrame::Snapshot {
            seq: 4,
            state: Snapshot {
                live_turns: vec![aidaemon_client_protocol::LiveTurnState {
                    session_id: "app:synthetic-1".into(),
                    turn_id: "t1".into(),
                    started_at_ms: 10,
                    streamed: "Half a reply".into(),
                    status: Some(TurnStatus::Thinking),
                    tools: vec!["terminal".into()],
                    // An older daemon's snapshot: names only.
                    calls: Vec::new(),
                }],
                ..Snapshot::default()
            },
        });
        let live = state.sessions["app:synthetic-1"]
            .live
            .clone()
            .expect("live");
        assert_eq!(live.streamed, "Half a reply");
        assert_eq!(live.tools, vec!["terminal".to_string()]);
        assert!(live.status.is_some());
        // A later status event finds its turn.
        state.apply(event(
            None,
            Event::TurnStatus {
                turn_id: "t1".into(),
                status: TurnStatus::tool_started("read_file", "notes.md", None),
            },
        ));
        let live = state.sessions["app:synthetic-1"]
            .live
            .clone()
            .expect("live");
        assert_eq!(live.tools.len(), 2);
        // The new step's call sits at its own index, after the snapshot's.
        assert_eq!(live.calls.len(), 2);
        assert_eq!(live.calls[1].name, "read_file");
        assert_eq!(live.calls[1].input, "notes.md");
    }

    /// A client that joins mid-turn shows each step as one that watched it.
    #[test]
    fn a_snapshot_brings_back_what_each_step_worked_on() {
        let mut state = State::default();
        state.apply(ServerFrame::Snapshot {
            seq: 4,
            state: Snapshot {
                live_turns: vec![aidaemon_client_protocol::LiveTurnState {
                    session_id: "app:synthetic-1".into(),
                    turn_id: "t1".into(),
                    started_at_ms: 10,
                    tools: vec!["reading a file".into()],
                    calls: vec![ToolCallInfo {
                        name: "reading a file".into(),
                        input: "src/synthetic.rs".into(),
                        output: "120 lines".into(),
                        kind: None,
                    }],
                    ..Default::default()
                }],
                ..Snapshot::default()
            },
        });
        let live = state.sessions["app:synthetic-1"]
            .live
            .clone()
            .expect("live");
        assert_eq!(live.calls[0].input, "src/synthetic.rs");
        assert_eq!(live.calls[0].output, "120 lines");
    }

    #[test]
    fn a_refused_sign_in_leaves_no_note_but_a_refused_message_does() {
        let mut state = State::default();
        state.apply(ServerFrame::Error {
            request_id: None,
            error: ProtocolError::new(ErrorCode::Unauthorized, "A valid client token is required"),
        });
        assert_eq!(state.notice, None);
        state.apply(ServerFrame::Error {
            request_id: Some("r-1".into()),
            error: ProtocolError::new(ErrorCode::Invalid, "That conversation is gone"),
        });
        assert_eq!(state.notice.as_deref(), Some("That conversation is gone"));
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

    fn user(text: &str, at_ms: u64) -> Line {
        Line::User {
            text: text.into(),
            origin: String::new(),
            at_ms,
            attachments: Vec::new(),
        }
    }

    fn reply(text: &str, at_ms: u64) -> Line {
        Line::Assistant {
            text: text.into(),
            at_ms,
            attachments: Vec::new(),
            handed_off: false,
            served_by: None,
        }
    }

    fn note(text: &str) -> Line {
        Line::Note {
            text: text.into(),
            ended: None,
        }
    }

    fn words(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| match l {
                Line::User { text, .. } => format!("u:{text}"),
                Line::Assistant { text, .. } => format!("a:{text}"),
                Line::Note { text, .. } => format!("n:{text}"),
                Line::Activity { .. } => "tools".into(),
            })
            .collect()
    }

    /// The first read of a conversation goes before what arrived live.
    #[test]
    fn a_first_history_read_goes_before_live_lines() {
        let merged = merge_history(vec![user("hi", 1), reply("hello", 2)], vec![note("x")]);
        assert_eq!(words(&merged), ["u:hi", "a:hello", "n:x"]);
    }

    /// Reading again after a frozen page brings what was missed, after what
    /// is shown, without doubling it or losing the client's own notes.
    #[test]
    fn a_history_read_again_adds_only_what_was_missed() {
        let shown = vec![user("hi", 1), reply("hello", 2), note("Stopped.")];
        let history = vec![
            user("hi", 1),
            reply("hello", 2),
            user("and now?", 3),
            reply("done", 4),
        ];
        let merged = merge_history(history, shown);
        assert_eq!(
            words(&merged),
            ["u:hi", "a:hello", "n:Stopped.", "u:and now?", "a:done"]
        );
    }

    /// A line that arrived live after the read stays; one the read also has
    /// is not doubled, nor is its tool line.
    #[test]
    fn a_history_read_keeps_newer_live_lines_once() {
        let tools = Line::Activity {
            tools: vec!["search".into()],
            secs: 2,
            calls: Vec::new(),
        };
        let shown = vec![
            user("hi", 1),
            reply("hello", 2),
            user("again", 3),
            tools.clone(),
            reply("found it", 4),
            user("thanks", 5),
        ];
        let history = vec![
            user("hi", 1),
            reply("hello", 2),
            user("again", 3),
            tools,
            reply("found it", 4),
        ];
        let merged = merge_history(history, shown);
        assert_eq!(
            words(&merged),
            [
                "u:hi",
                "a:hello",
                "u:again",
                "tools",
                "a:found it",
                "u:thanks"
            ]
        );
    }

    /// Away longer than history's window: history replaces the stale page.
    #[test]
    fn a_page_older_than_the_whole_read_is_replaced() {
        let shown = vec![user("old", 1), reply("older", 2)];
        let history = vec![user("new", 10), reply("newer", 11)];
        assert_eq!(words(&merge_history(history, shown)), ["u:new", "a:newer"]);
    }
}
