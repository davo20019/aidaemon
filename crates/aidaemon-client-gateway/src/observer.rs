//! How the gateway sees the daemon's work: every turn from every chat
//! ([`TurnObserver`]), the app's own turns ([`AppSurface`]), and what the
//! daemon sends to an app session outside a turn ([`AppChannel`]).

use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use aidaemon_channels::hub::{ChannelCapabilities, HubChannel, MediaKind, MediaMessage};
use aidaemon_channels::ports::AgentResponseEnvelope;
use aidaemon_channels::turns::{
    Delivered, RunChange, TurnEnd, TurnFinish, TurnObserver, TurnStart, TurnSurface,
};
use aidaemon_client_protocol::{
    Event, LiveTurnState, RunPhase, RunSummary, StepState, TaskState, TaskSummary, ToolCallInfo,
    TurnState, TurnStatus,
};
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::ApprovalResponse;
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::bus::now_ms;
use crate::{APP_CHANNEL, Publisher};

/// Ids clients see for the daemon's numbered things, unique across
/// restarts (the registry's numbers start again at each boot).
#[derive(Clone)]
pub struct Ids {
    pub boot: String,
}

impl Ids {
    pub fn turn(&self, turn: u64) -> String {
        format!("{}-t{turn}", self.boot)
    }

    /// A running-work entry: `cancel_task` takes it back.
    pub fn task(&self, session_id: &str, number: u64) -> String {
        format!("{}:{session_id}#{number}", self.boot)
    }

    /// The session and number of task `id`, when it is from this boot.
    pub fn parse_task(&self, id: &str) -> Option<(String, u64)> {
        let rest = id.strip_prefix(&self.boot)?.strip_prefix(':')?;
        let (session, number) = rest.rsplit_once('#')?;
        Some((session.to_string(), number.parse().ok()?))
    }
}

/// Reports every chat's turns as client events.
pub struct GatewayObserver {
    pub(crate) publisher: Publisher,
    pub(crate) ids: Ids,
    /// Sessions that ran a turn this boot: where running work can be.
    pub(crate) active: Arc<Mutex<HashSet<String>>>,
    /// Turns running now, for their start time and description.
    pub(crate) running: Mutex<std::collections::HashMap<u64, (String, u64)>>,
    /// Turns running now, as far as each has got: what a snapshot gives a
    /// client that connects mid-turn.
    pub(crate) live: LiveTurns,
    /// Each chat conversation as its own channel last saw it: what a reply
    /// from the app into that conversation runs under.
    pub(crate) contexts: Contexts,
    /// Where a message's files are shown from; `None`: not shown.
    pub(crate) files: Option<Arc<crate::files::FileStore>>,
    /// Names the provider of the model that wrote a reply.
    pub(crate) host: Arc<dyn crate::ClientHost>,
}

/// Files as clients reach them.
pub(crate) fn shown_files(
    files: Option<&crate::files::FileStore>,
    attachments: &[aidaemon_tools::semantics::MessageAttachment],
) -> Vec<aidaemon_client_protocol::AttachmentRef> {
    files.map(|f| f.show(attachments)).unwrap_or_default()
}

/// A picture or document the agent sent, as clients reach it.
fn media_file(
    files: &crate::files::FileStore,
    media: &MediaMessage,
) -> Option<aidaemon_client_protocol::AttachmentRef> {
    let saved = match &media.kind {
        MediaKind::Photo { data } => files.save("picture.png", None, data),
        MediaKind::Document {
            file_path,
            filename,
        } => {
            let path = std::path::Path::new(file_path);
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let mime = std::fs::read(path)
                .ok()
                .and_then(|bytes| aidaemon_tools::sniff_image_mime(&bytes).map(str::to_string))
                .unwrap_or_else(|| "application/octet-stream".to_string());
            files.register(path, filename, &mime, size)
        }
    };
    saved
        .map_err(|e| tracing::warn!(error = %e, "Could not keep a file the agent sent"))
        .ok()
}

/// Shows clients the pictures and documents the agent sends to a chat
/// (Telegram, Slack...): the app's own sessions get theirs through
/// [`AppChannel`].
pub(crate) struct MediaToClients {
    pub(crate) publisher: Publisher,
    pub(crate) files: Arc<crate::files::FileStore>,
    /// The copy [`AppChannel`] kept of what it just sent, by session.
    pub(crate) kept: Kept,
}

/// Files the app channel kept as it sent them, by session, until the media
/// tap hands them to the transcript.
pub(crate) type Kept =
    Arc<Mutex<std::collections::HashMap<String, aidaemon_tools::semantics::MessageAttachment>>>;

/// A kept file as a message's attachment, so the transcript shows it again.
fn kept_attachment(
    files: &crate::files::FileStore,
    file: &aidaemon_client_protocol::AttachmentRef,
) -> Option<aidaemon_tools::semantics::MessageAttachment> {
    let record = files.get(&file.id)?;
    Some(aidaemon_tools::semantics::MessageAttachment {
        local_path: record.path.to_string_lossy().into_owned(),
        filename: record.name,
        mime_type: record.mime,
        size_bytes: record.size_bytes,
        provenance: aidaemon_tools::semantics::AttachmentProvenance::ToolArtifact,
        ..Default::default()
    })
}

impl aidaemon_channels::hub::MediaTap for MediaToClients {
    fn delivered(
        &self,
        channel: &str,
        session_id: &str,
        media: &MediaMessage,
    ) -> Option<aidaemon_tools::semantics::MessageAttachment> {
        if channel == APP_CHANNEL {
            // The app channel published it already.
            return self.kept.lock().ok()?.remove(session_id);
        }
        let file = media_file(&self.files, media)?;
        let kept = kept_attachment(&self.files, &file);
        self.publisher.publish(
            Some(session_id.to_string()),
            Event::AssistantMessage {
                session_id: session_id.to_string(),
                text: media.caption.trim().to_string(),
                attachments: vec![file],
                suggestions: Vec::new(),
            },
        );
        kept
    }
}

/// Turns running now, by turn number.
pub(crate) type LiveTurns = Arc<Mutex<std::collections::HashMap<u64, LiveTurnState>>>;

/// Chat conversations' channel contexts, by session.
pub(crate) type Contexts =
    Arc<Mutex<std::collections::HashMap<String, aidaemon_channels::context::ChannelContext>>>;

fn status_of(update: &StatusUpdate) -> Option<TurnStatus> {
    let n = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
    Some(match update {
        StatusUpdate::Thinking(_) => TurnStatus::Thinking,
        StatusUpdate::ToolStart {
            name,
            summary,
            kind,
            input,
            ..
        } => TurnStatus::tool_started(name.clone(), summary.clone(), Some(crate::tool_kind(*kind)))
            .with_detail(input.clone()),
        StatusUpdate::ToolProgress { name, chunk } => TurnStatus::ToolProgress {
            name: name.clone(),
            chunk: chunk.clone(),
        },
        StatusUpdate::ToolComplete {
            name,
            summary,
            output,
            ..
        } => TurnStatus::tool_completed(name.clone(), summary.clone()).with_detail(output.clone()),
        StatusUpdate::PlanStepStart {
            step_index,
            total_steps,
            description,
            ..
        } => TurnStatus::PlanStep {
            step: n(*step_index),
            total: n(*total_steps),
            description: description.clone(),
            state: StepState::Started,
        },
        StatusUpdate::PlanStepComplete {
            step_index,
            total_steps,
            description,
            ..
        } => TurnStatus::PlanStep {
            step: n(*step_index),
            total: n(*total_steps),
            description: description.clone(),
            state: StepState::Completed,
        },
        StatusUpdate::PlanStepFailed {
            step_index,
            description,
            ..
        } => TurnStatus::PlanStep {
            step: n(*step_index),
            total: 0,
            description: description.clone(),
            state: StepState::Failed,
        },
        StatusUpdate::Checklist(plan) => TurnStatus::Checklist {
            text: plan.text.clone(),
        },
        StatusUpdate::Helper(helper) => TurnStatus::Note {
            text: format!(
                "{}: {}",
                helper.label,
                aidaemon_channels::live_status::helper_status_words(helper)
            ),
        },
        StatusUpdate::ProgressSummary { summary, .. } => TurnStatus::Note {
            text: summary.clone(),
        },
        StatusUpdate::PlanCreated { description, .. } => TurnStatus::Note {
            text: format!("Plan: {description}"),
        },
        StatusUpdate::PlanRevised { reason, .. } => TurnStatus::Note {
            text: format!("Plan revised: {reason}"),
        },
        StatusUpdate::PlanComplete { description, .. } => TurnStatus::Note {
            text: format!("Plan done: {description}"),
        },
        StatusUpdate::PlanAbandoned { description, .. } => TurnStatus::Note {
            text: format!("Plan abandoned: {description}"),
        },
        // What a command printed, redacted, for the owner's own app: the
        // chats' cards read only its line count.
        StatusUpdate::CommandOutput {
            name,
            tail: Some(tail),
            ..
        } => TurnStatus::ToolProgress {
            name: name.clone(),
            chunk: tail.clone(),
        },
        StatusUpdate::StepHandedOff { kind } => TurnStatus::Note {
            text: aidaemon_core::card::Step::HandedOff(*kind).words(),
        },
        _ => return None,
    })
}

impl GatewayObserver {
    /// A turn begins in `session_id`: it is live, running, and shown.
    fn begin(&self, session_id: &str, turn: u64, description: &str, origin: &str) {
        let started = now_ms();
        if let Ok(mut active) = self.active.lock() {
            active.insert(session_id.to_string());
        }
        if let Ok(mut running) = self.running.lock() {
            running.insert(turn, (description.to_string(), started));
        }
        if let Ok(mut live) = self.live.lock() {
            live.insert(
                turn,
                LiveTurnState {
                    session_id: session_id.to_string(),
                    turn_id: self.ids.turn(turn),
                    started_at_ms: started,
                    ..LiveTurnState::default()
                },
            );
        }
        let session = Some(session_id.to_string());
        self.publisher.publish(
            session.clone(),
            Event::TurnStarted {
                turn_id: self.ids.turn(turn),
                session_id: session_id.to_string(),
                origin: origin.to_string(),
            },
        );
        self.publisher.publish(
            session,
            Event::TaskUpdated {
                task: TaskSummary {
                    id: self.ids.task(session_id, turn),
                    session_id: session_id.to_string(),
                    number: turn,
                    description: description.to_string(),
                    state: TaskState::Running,
                    started_at_ms: started,
                    finished_at_ms: None,
                    error: None,
                },
            },
        );
    }
}

impl TurnObserver for GatewayObserver {
    fn turn_started(&self, turn: &TurnStart<'_>) {
        if turn.platform != APP_CHANNEL
            && let Ok(mut contexts) = self.contexts.lock()
        {
            contexts.insert(turn.session_id.to_string(), turn.channel_ctx.clone());
        }
        // What the sender said comes first, then the work on it.
        self.publisher.publish(
            Some(turn.session_id.to_string()),
            Event::UserMessage {
                session_id: turn.session_id.to_string(),
                // What the sender typed: the files' notes for the agent
                // come as the files themselves.
                text: if turn.attachments.is_empty() {
                    turn.text.to_string()
                } else {
                    aidaemon_channels::attachments::user_authored_text(turn.text)
                },
                attachments: shown_files(self.files.as_deref(), turn.attachments),
                origin: turn.platform.to_string(),
            },
        );
        self.begin(turn.session_id, turn.turn, turn.description, turn.platform);
    }

    fn followup_started(&self, session_id: &str, turn: u64, description: &str) {
        self.begin(session_id, turn, description, "background");
    }

    fn turn_status(&self, session_id: &str, turn: u64, update: &StatusUpdate) {
        let turn_id = self.ids.turn(turn);
        let session = Some(session_id.to_string());
        if let StatusUpdate::ReplyText(text) = update {
            if let Ok(mut live) = self.live.lock()
                && let Some(state) = live.get_mut(&turn)
            {
                state.streamed.push_str(text);
            }
            self.publisher.publish(
                session,
                Event::TurnDelta {
                    turn_id,
                    text: text.clone(),
                },
            );
            return;
        }
        // A message the running turn took in: shown where it was read, as
        // what the sender said (the turn's opening message comes from
        // `turn_started`; one sent mid-run had nothing to show it, and left
        // the waiting list when the turn ended).
        if let StatusUpdate::MessageTaken { text, origin } = update {
            self.publisher.publish(
                session,
                Event::UserMessage {
                    session_id: session_id.to_string(),
                    text: text.clone(),
                    attachments: Vec::new(),
                    origin: origin.clone(),
                },
            );
            return;
        }
        if let StatusUpdate::UserActionRequired { message } = update {
            self.publisher.publish(
                session,
                Event::UserActionRequired {
                    turn_id,
                    message: message.clone(),
                },
            );
            return;
        }
        if let Some(status) = status_of(update) {
            if let Ok(mut live) = self.live.lock()
                && let Some(state) = live.get_mut(&turn)
            {
                match &status {
                    TurnStatus::ToolStarted { name, activity, .. } => {
                        state.tools.push(name.clone());
                        state.calls.push(ToolCallInfo {
                            name: name.clone(),
                            input: status.call_detail().unwrap_or_default().to_string(),
                            output: String::new(),
                            kind: *activity,
                        });
                    }
                    // The latest call of that name still running, as the
                    // client marks it.
                    TurnStatus::ToolCompleted { name, .. } => {
                        if let Some(call) = state
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
                state.status = Some(status.clone());
            }
            self.publisher
                .publish(session, Event::TurnStatus { turn_id, status });
        }
    }

    fn run_changed(&self, change: &RunChange) {
        let session = Some(change.session_id.clone());
        if change.step_only {
            self.publisher.publish(
                session,
                Event::RunStep {
                    run_id: change.id.clone(),
                    session_id: change.session_id.clone(),
                    step: step_words(&change.view, change.activity.as_deref()),
                    at_ms: now_ms(),
                },
            );
            return;
        }
        let run = run_summary(
            &change.session_id,
            &change.id,
            &change.view,
            RunRecord {
                activity: change.activity.as_deref(),
                agent: change.agent.as_deref(),
                group: change.group.as_deref(),
                edited: &change.edited,
                continues: change.continues.as_deref(),
                thread: change.thread.as_deref(),
                keeps_session: change.keeps_session,
                result: change.result.as_ref(),
                started_at: change.started_at,
                ended_at: change.ended_at,
            },
        );
        // An ended run stays in its conversation past a restart.
        if run.ended_at_ms.is_some() {
            self.host.keep_run(&run);
        }
        self.publisher.publish(session, Event::RunUpdated { run });
    }

    fn turn_finished(&self, turn: &TurnFinish<'_>) {
        let (description, started) = self
            .running
            .lock()
            .ok()
            .and_then(|mut running| running.remove(&turn.turn))
            .unwrap_or_default();
        if let Ok(mut live) = self.live.lock() {
            live.remove(&turn.turn);
        }
        let (state, task_state) = match turn.end {
            TurnEnd::Delivered if turn.handed_off => (TurnState::HandedOff, TaskState::Completed),
            TurnEnd::Delivered => (TurnState::Completed, TaskState::Completed),
            TurnEnd::Empty => (TurnState::Empty, TaskState::Completed),
            TurnEnd::Failed => (TurnState::Failed, TaskState::Failed),
            TurnEnd::Cancelled => (TurnState::Cancelled, TaskState::Cancelled),
        };
        let session = Some(turn.session_id.to_string());
        self.publisher.publish(
            session.clone(),
            Event::TurnCompleted {
                turn_id: self.ids.turn(turn.turn),
                text: turn.reply.to_string(),
                attachments: Vec::new(),
                state,
                error: turn.error.map(str::to_string),
                served_by: turn
                    .model
                    .filter(|_| !turn.reply.is_empty())
                    .and_then(|model| self.host.served_by(model)),
            },
        );
        self.publisher.publish(
            session,
            Event::TaskUpdated {
                task: TaskSummary {
                    id: self.ids.task(turn.session_id, turn.turn),
                    session_id: turn.session_id.to_string(),
                    number: turn.turn,
                    description,
                    state: task_state,
                    started_at_ms: started,
                    finished_at_ms: Some(now_ms()),
                    error: turn.error.map(str::to_string),
                },
            },
        );
    }
}

/// A run's record as the owner's app reads it. Every field is the
/// runtime's: phase, step, reason and time taken come from the run's typed
/// record ([`aidaemon_core::card::HelperView`]), never from its reply.
/// What a run's record adds to its view, for its summary.
pub(crate) struct RunRecord<'a> {
    pub activity: Option<&'a str>,
    pub agent: Option<&'a str>,
    pub group: Option<&'a str>,
    pub edited: &'a [String],
    pub continues: Option<&'a str>,
    pub thread: Option<&'a str>,
    pub keeps_session: bool,
    pub result: Option<&'a aidaemon_channels::turns::RunResult>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub ended_at: Option<chrono::DateTime<chrono::Utc>>,
}

pub(crate) fn run_summary(
    session_id: &str,
    id: &str,
    view: &aidaemon_core::card::HelperView,
    record: RunRecord<'_>,
) -> RunSummary {
    let RunRecord {
        activity,
        agent,
        group,
        edited,
        continues,
        thread,
        keeps_session,
        result,
        started_at,
        ended_at,
    } = record;
    use aidaemon_core::card::RunPhase as Phase;
    let phase = match view.phase {
        Phase::Running => RunPhase::Running,
        Phase::Succeeded => RunPhase::Succeeded,
        Phase::Failed => RunPhase::Failed,
        Phase::Stopped => RunPhase::Stopped,
    };
    let ms = |at: chrono::DateTime<chrono::Utc>| u64::try_from(at.timestamp_millis()).unwrap_or(0);
    RunSummary {
        id: id.to_string(),
        session_id: session_id.to_string(),
        label: view.label.clone(),
        agent: agent.map(str::to_string),
        phase,
        background: view.background,
        step: step_words(view, activity),
        started_at_ms: ms(started_at),
        ended_at_ms: ended_at.map(ms),
        took: view.took.clone(),
        reason: view.reason.clone(),
        stopped_by_owner: view.stopped_by_owner,
        group: group.map(str::to_string),
        edited: edited.to_vec(),
        continues: continues.map(str::to_string),
        thread: thread.map(str::to_string),
        // A follow-up goes on from an ended run whose agent kept its session.
        can_continue: keeps_session && ended_at.is_some(),
        result: result.map(run_result),
    }
}

/// A run's result as the app reads it: the runtime's typed record.
fn run_result(result: &aidaemon_channels::turns::RunResult) -> aidaemon_client_protocol::RunResult {
    use aidaemon_channels::turns::RunMismatch as Seen;
    use aidaemon_client_protocol::{RunCheck, RunMismatch, RunReport};
    aidaemon_client_protocol::RunResult {
        text: result.text.clone(),
        report: result.claims.as_ref().map(|claims| RunReport {
            summary: claims.summary.clone(),
            changed_files: claims.changed_files.clone(),
            checks: claims
                .checks
                .iter()
                .map(|(command, passed)| RunCheck {
                    command: command.clone(),
                    passed: *passed,
                })
                .collect(),
            open_questions: claims.open_questions.clone(),
            needs_owner: claims.needs_owner,
        }),
        mismatches: result
            .mismatches
            .iter()
            .map(|mismatch| match mismatch {
                Seen::ChangeNotSeen { path } => RunMismatch::ChangeNotSeen { path: path.clone() },
                Seen::ChangeNotClaimed { path } => {
                    RunMismatch::ChangeNotClaimed { path: path.clone() }
                }
                Seen::CheckNotRun { command } => RunMismatch::CheckNotRun {
                    command: command.clone(),
                },
                Seen::CheckFailed { command } => RunMismatch::CheckFailed {
                    command: command.clone(),
                },
            })
            .collect(),
    }
}

/// A running run's step in plain words; empty once it ended or when the
/// step says nothing ([`aidaemon_core::card::Step::is_shown`]).
fn step_words(view: &aidaemon_core::card::HelperView, activity: Option<&str>) -> String {
    if view.phase.is_terminal() {
        return String::new();
    }
    if let Some(activity) = activity.filter(|words| !words.is_empty()) {
        return activity.to_string();
    }
    view.step
        .as_ref()
        .filter(|step| step.is_shown())
        .map(aidaemon_core::card::Step::words)
        .unwrap_or_default()
}

/// Where an app session's turns are shown: their events. The reply itself
/// is reported by the observer, as every chat's is, so `deliver` sends
/// nothing more.
pub struct AppSurface {
    pub(crate) publisher: Publisher,
    pub(crate) session_id: String,
}

#[async_trait]
impl TurnSurface for AppSurface {
    fn platform(&self) -> &'static str {
        APP_CHANNEL
    }

    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
        self.publisher.publish(
            Some(self.session_id.clone()),
            Event::AssistantMessage {
                attachments: Vec::new(),
                session_id: self.session_id.clone(),
                text: text.to_string(),
                suggestions: Vec::new(),
            },
        );
        Ok(Vec::new())
    }

    fn show_working(&self, _stop: CancellationToken, _heartbeat: Arc<AtomicU64>, _stale: u64) {}

    fn show_status(
        &self,
        mut updates: mpsc::Receiver<StatusUpdate>,
        _private: bool,
    ) -> Result<JoinHandle<()>, mpsc::Receiver<StatusUpdate>> {
        // The observer already publishes every update.
        Ok(tokio::spawn(async move {
            while updates.recv().await.is_some() {}
        }))
    }

    async fn deliver(&self, _envelope: &AgentResponseEnvelope, _reply: &str) -> Delivered {
        Delivered {
            state: aidaemon_events::ResponseDeliveryState::PlatformAcknowledged,
            message_ids: Vec::new(),
            error: None,
        }
    }
}

/// A reply from the app into a chat's own conversation (Slack, Telegram,
/// Discord): the agent's answer goes to that chat through the hub, and the
/// app sees it as it sees its own turns.
pub struct ChatReplySurface {
    pub(crate) publisher: Publisher,
    pub(crate) hub: Arc<aidaemon_channels::hub::ChannelHub>,
    pub(crate) session_id: String,
}

#[async_trait]
impl TurnSurface for ChatReplySurface {
    fn platform(&self) -> &'static str {
        APP_CHANNEL
    }

    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
        self.hub.send_text(&self.session_id, text).await?;
        self.publisher.publish(
            Some(self.session_id.clone()),
            Event::AssistantMessage {
                attachments: Vec::new(),
                session_id: self.session_id.clone(),
                text: text.to_string(),
                suggestions: Vec::new(),
            },
        );
        Ok(Vec::new())
    }

    fn show_working(&self, _stop: CancellationToken, _heartbeat: Arc<AtomicU64>, _stale: u64) {}

    fn show_status(
        &self,
        mut updates: mpsc::Receiver<StatusUpdate>,
        _private: bool,
    ) -> Result<JoinHandle<()>, mpsc::Receiver<StatusUpdate>> {
        // The observer publishes every update to the app.
        Ok(tokio::spawn(async move {
            while updates.recv().await.is_some() {}
        }))
    }

    /// The answer goes to the chat; the app gets it with the turn's end.
    async fn deliver(&self, _envelope: &AgentResponseEnvelope, reply: &str) -> Delivered {
        match self.hub.send_text(&self.session_id, reply).await {
            Ok(()) => Delivered {
                state: aidaemon_events::ResponseDeliveryState::PlatformAcknowledged,
                message_ids: Vec::new(),
                error: None,
            },
            Err(error) => Delivered {
                state: aidaemon_events::ResponseDeliveryState::Failed,
                message_ids: Vec::new(),
                error: Some(error.to_string()),
            },
        }
    }
}

/// The app as the hub sees it: where an app session's notifications, files
/// and approval cards go. Cards are answered through the approval tap, so
/// asking here only waits.
pub struct AppChannel {
    pub(crate) publisher: Publisher,
    pub(crate) files: Option<Arc<crate::files::FileStore>>,
    pub(crate) kept: Kept,
}

impl AppChannel {
    fn say(&self, session_id: &str, text: String) {
        self.publisher.publish(
            Some(session_id.to_string()),
            Event::AssistantMessage {
                attachments: Vec::new(),
                session_id: session_id.to_string(),
                text,
                suggestions: Vec::new(),
            },
        );
    }
}

#[async_trait]
impl HubChannel for AppChannel {
    fn name(&self) -> String {
        APP_CHANNEL.to_string()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            markdown: true,
            inline_buttons: true,
            media: self.files.is_some(),
            max_message_len: 64 * 1024,
        }
    }

    /// The app shows each run as its own row (protocol 1.46 run records).
    fn shows_runs(&self) -> bool {
        true
    }

    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        self.say(session_id, text.to_string());
        Ok(())
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        if let Some(files) = &self.files {
            let file = media_file(files, media)
                .ok_or_else(|| anyhow::anyhow!("could not keep the file for the app"))?;
            if let (Some(attachment), Ok(mut kept)) =
                (kept_attachment(files, &file), self.kept.lock())
            {
                kept.insert(session_id.to_string(), attachment);
            }
            self.publisher.publish(
                Some(session_id.to_string()),
                Event::AssistantMessage {
                    session_id: session_id.to_string(),
                    text: media.caption.trim().to_string(),
                    attachments: vec![file],
                    suggestions: Vec::new(),
                },
            );
            return Ok(());
        }
        let name = match &media.kind {
            MediaKind::Photo { .. } => "a picture".to_string(),
            MediaKind::Document { filename, .. } => format!("`{filename}`"),
        };
        // Files reach clients with the attachments capability (later).
        let caption = media.caption.trim();
        let text = if caption.is_empty() {
            format!("(Sent {name}; open it from a chat that shows files.)")
        } else {
            format!("{caption}\n\n(Sent {name}; open it from a chat that shows files.)")
        };
        self.say(session_id, text);
        Ok(())
    }

    async fn request_approval(
        &self,
        _session_id: &str,
        _command: &str,
        _risk_level: RiskLevel,
        _warnings: &[String],
        _permission_mode: PermissionMode,
        _one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        // Clients answer through the approval tap; whoever asked stops
        // waiting when its time is up.
        std::future::pending().await
    }

    async fn request_goal_confirmation(
        &self,
        _session_id: &str,
        _goal_description: &str,
        _details: &[String],
        _style: aidaemon_tools::semantics::GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        // Never confirmed unasked: clients answer through the approval tap.
        std::future::pending().await
    }
}
