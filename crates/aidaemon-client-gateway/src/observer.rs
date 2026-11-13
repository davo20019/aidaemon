//! How the gateway sees the daemon's work: every turn from every chat
//! ([`TurnObserver`]), the app's own turns ([`AppSurface`]), and what the
//! daemon sends to an app session outside a turn ([`AppChannel`]).

use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use aidaemon_channels::hub::{ChannelCapabilities, HubChannel, MediaKind, MediaMessage};
use aidaemon_channels::ports::AgentResponseEnvelope;
use aidaemon_channels::turns::{
    Delivered, TurnEnd, TurnFinish, TurnObserver, TurnStart, TurnSurface,
};
use aidaemon_client_protocol::{Event, StepState, TaskState, TaskSummary, TurnState, TurnStatus};
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::ApprovalResponse;
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::bus::now_ms;
use crate::{Publisher, APP_CHANNEL};

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
}

fn status_of(update: &StatusUpdate) -> Option<TurnStatus> {
    let n = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
    Some(match update {
        StatusUpdate::Thinking(_) => TurnStatus::Thinking,
        StatusUpdate::ToolStart { name, summary, .. } => TurnStatus::ToolStarted {
            name: name.clone(),
            summary: summary.clone(),
        },
        StatusUpdate::ToolProgress { name, chunk } => TurnStatus::ToolProgress {
            name: name.clone(),
            chunk: chunk.clone(),
        },
        StatusUpdate::ToolComplete { name, summary } => TurnStatus::ToolCompleted {
            name: name.clone(),
            summary: summary.clone(),
        },
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
        StatusUpdate::StepHandedOff { kind } => TurnStatus::Note {
            text: aidaemon_core::card::Step::HandedOff(*kind).words(),
        },
        _ => return None,
    })
}

impl TurnObserver for GatewayObserver {
    fn turn_started(&self, turn: &TurnStart<'_>) {
        let started = now_ms();
        if let Ok(mut active) = self.active.lock() {
            active.insert(turn.session_id.to_string());
        }
        if let Ok(mut running) = self.running.lock() {
            running.insert(turn.turn, (turn.description.to_string(), started));
        }
        let session = Some(turn.session_id.to_string());
        self.publisher.publish(
            session.clone(),
            Event::UserMessage {
                session_id: turn.session_id.to_string(),
                text: turn.text.to_string(),
                attachments: Vec::new(),
                origin: turn.platform.to_string(),
            },
        );
        self.publisher.publish(
            session.clone(),
            Event::TurnStarted {
                turn_id: self.ids.turn(turn.turn),
                session_id: turn.session_id.to_string(),
                origin: turn.platform.to_string(),
            },
        );
        self.publisher.publish(
            session,
            Event::TaskUpdated {
                task: TaskSummary {
                    id: self.ids.task(turn.session_id, turn.turn),
                    session_id: turn.session_id.to_string(),
                    number: turn.turn,
                    description: turn.description.to_string(),
                    state: TaskState::Running,
                    started_at_ms: started,
                    finished_at_ms: None,
                    error: None,
                },
            },
        );
    }

    fn turn_status(&self, session_id: &str, turn: u64, update: &StatusUpdate) {
        let turn_id = self.ids.turn(turn);
        let session = Some(session_id.to_string());
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
            self.publisher
                .publish(session, Event::TurnStatus { turn_id, status });
        }
    }

    fn turn_finished(&self, turn: &TurnFinish<'_>) {
        let (description, started) = self
            .running
            .lock()
            .ok()
            .and_then(|mut running| running.remove(&turn.turn))
            .unwrap_or_default();
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
                session_id: self.session_id.clone(),
                text: text.to_string(),
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

/// The app as the hub sees it: where an app session's notifications, files
/// and approval cards go. Cards are answered through the approval tap, so
/// asking here only waits.
pub struct AppChannel {
    pub(crate) publisher: Publisher,
}

impl AppChannel {
    fn say(&self, session_id: &str, text: String) {
        self.publisher.publish(
            Some(session_id.to_string()),
            Event::AssistantMessage {
                session_id: session_id.to_string(),
                text,
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
            media: false,
            max_message_len: 64 * 1024,
        }
    }

    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        self.say(session_id, text.to_string());
        Ok(())
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
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
