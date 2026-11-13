//! A session's turns, the same on every chat platform.
//!
//! A platform hands the runner an authorized [`Turn`] and a [`TurnSurface`]
//! (where the turn's output goes). The runner owns everything that follows:
//! session registration, cancel words, queueing and deduplication, task
//! tracking, the inactivity watchdog and wall-clock limit, the working
//! indicator and status updates, reply delivery and its receipts, and
//! draining the session's queue in order. Platforms differ only in how the
//! surface shows things.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use aidaemon_core::ChannelVisibility;
use aidaemon_events::conversation::MessageAttachment;
use aidaemon_events::ResponseDeliveryState;
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::status::StatusUpdate;

use crate::context::ChannelContext;
use crate::hub::{SessionChannelStore, SessionMap};
use crate::owner_wait::OwnerWaitClock;
use crate::ports::{
    AgentResponseEnvelope, ChannelAgentRuntime, InboundMessageRequest, InboundMessageTiming,
    MidRunIntent, MidRunReading, RunningWork,
};
use crate::stop::StopRequest;
use crate::tasks::{QueueOutcome, QueuedMessage, TaskRegistry};

/// Least time between two ordinary status messages.
const STATUS_INTERVAL: Duration = Duration::from_secs(3);
/// Most time a finished turn waits for its status display to catch up
/// before the reply is sent.
const STATUS_DRAIN: Duration = Duration::from_secs(3);
const CANCELLED: &str = "Task cancelled";
const INACTIVITY_PREFIX: &str = "Task auto-cancelled due to inactivity";

/// One authorized message, ready to run.
pub struct Turn {
    pub session_id: String,
    pub text: String,
    pub attachments: Vec<MessageAttachment>,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    /// The platform's message id. A redelivery of the same message is
    /// dropped; the same text typed again is not.
    pub dedup_key: Option<String>,
    pub ingress_timing: Option<InboundMessageTiming>,
}

/// What delivering a reply achieved.
pub struct Delivered {
    pub state: ResponseDeliveryState,
    pub message_ids: Vec<String>,
    pub error: Option<String>,
}

impl Delivered {
    fn from_send(result: anyhow::Result<Vec<String>>, platform: &str) -> Self {
        match result {
            Ok(message_ids) => Self {
                state: ResponseDeliveryState::PlatformAcknowledged,
                message_ids,
                error: None,
            },
            Err(error) => {
                warn!(platform, %error, "Failed to send a reply");
                Self {
                    state: ResponseDeliveryState::Failed,
                    message_ids: Vec::new(),
                    error: Some(format!("{platform}_send_failed")),
                }
            }
        }
    }
}

/// How a turn ended, for [`TurnSurface::turn_ended`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnEnd {
    Delivered,
    /// The agent had nothing to say.
    Empty,
    Failed,
    Cancelled,
}

/// Where one conversation's turns are shown.
#[async_trait]
pub trait TurnSurface: Send + Sync + 'static {
    /// The platform recorded on delivery receipts (`telegram`, `slack`...).
    fn platform(&self) -> &'static str;

    /// Send Markdown `text`, split to fit. Returns the ids of the messages
    /// sent, or an error if any part failed.
    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>>;

    /// Replace the text of message `message_id` that [`Self::say`] sent.
    /// `Ok(false)` when the platform cannot edit it; the default suits
    /// surfaces that never can.
    async fn edit(&self, _message_id: &str, _text: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Show that work is in progress until `stop` is cancelled. When the
    /// heartbeat has been quiet for `stale_secs` (0 = never), stop showing it.
    fn show_working(&self, stop: CancellationToken, heartbeat: Arc<AtomicU64>, stale_secs: u64);

    /// Show one turn's status updates. Handing them back (the default)
    /// uses throttled status messages, sent only in private conversations.
    fn show_status(
        &self,
        updates: mpsc::Receiver<StatusUpdate>,
        _private: bool,
    ) -> Result<JoinHandle<()>, mpsc::Receiver<StatusUpdate>> {
        Err(updates)
    }

    /// Whether this surface's status is one card that the reply settles
    /// (its closing state decides whether the card stays). Such a surface
    /// gets a finished turn's last updates before the reply; plain status
    /// messages do not, since a late one would only repeat the reply.
    fn status_settles_before_reply(&self) -> bool {
        false
    }

    /// Deliver the agent's reply (already prepared for chat).
    async fn deliver(&self, _envelope: &AgentResponseEnvelope, reply: &str) -> Delivered {
        Delivered::from_send(self.say(reply).await, self.platform())
    }

    /// A turn is over: its reply was delivered, or it had none, or it
    /// failed or was cancelled (after any error message was shown).
    async fn turn_ended(&self, _ended: TurnEnd) {}

    /// Re-check a queued message just before it runs; its authority may
    /// have narrowed while it waited.
    async fn before_queued(&self, queued: QueuedMessage) -> QueuedMessage {
        queued
    }
}

/// What a [`TurnRunner`] runs turns with.
#[derive(Clone)]
pub struct TurnDeps {
    pub agent: Arc<dyn ChannelAgentRuntime>,
    /// Which bot each session belongs to, for outbound routing.
    pub session_map: SessionMap,
    /// Where that mapping is persisted.
    pub sessions: Arc<dyn SessionChannelStore>,
    pub task_registry: Arc<TaskRegistry>,
    /// Heartbeat silence (seconds) after which a turn is cancelled as hung;
    /// 0 never cancels.
    pub watchdog_stale_threshold_secs: u64,
}

/// Per session: each queued message's text and its notice's message id.
type QueuedNotices = std::collections::HashMap<String, Vec<(String, String)>>;

/// Runs turns for one channel (bot).
#[derive(Clone)]
pub struct TurnRunner {
    deps: TurnDeps,
    /// Per session, the "got it" notice sent for each queued message (its
    /// text and the notice's message id), so the notice can be edited when
    /// that message's turn starts instead of a second bubble being sent.
    queued_notices: Arc<std::sync::Mutex<QueuedNotices>>,
    /// Each session's running work as its status updates show it, for
    /// answering a question about it mid-run.
    live: crate::mid_run::LiveWork,
}

/// Longest the owner waits for a mid-run message to be read before it
/// joins the running work unread, as before.
const MID_RUN_READ_LIMIT: Duration = Duration::from_secs(15);

impl TurnRunner {
    pub fn new(deps: TurnDeps) -> Self {
        Self {
            deps,
            queued_notices: Arc::default(),
            live: crate::mid_run::LiveWork::default(),
        }
    }

    fn remember_queued_notice(&self, session_id: &str, text: &str, message_id: String) {
        if let Ok(mut notices) = self.queued_notices.lock() {
            notices
                .entry(session_id.to_string())
                .or_default()
                .push((text.to_string(), message_id));
        }
    }

    /// The notice sent for the queued `text`, taken out of the map.
    fn take_queued_notice(&self, session_id: &str, text: &str) -> Option<String> {
        let mut notices = self.queued_notices.lock().ok()?;
        let session = notices.get_mut(session_id)?;
        let at = session.iter().position(|(queued, _)| queued == text)?;
        let (_, message_id) = session.remove(at);
        if session.is_empty() {
            notices.remove(session_id);
        }
        Some(message_id)
    }

    fn forget_queued_notices(&self, session_id: &str) {
        if let Ok(mut notices) = self.queued_notices.lock() {
            notices.remove(session_id);
        }
    }

    /// Remember that `session_id` belongs to `channel_name`, so outbound
    /// messages (approvals, media, notifications) route back to it.
    pub async fn register_session(&self, session_id: &str, channel_name: &str) {
        self.deps
            .session_map
            .write()
            .await
            .insert(session_id.to_string(), channel_name.to_string());
        let _ = self
            .deps
            .sessions
            .save_session_channel(session_id, channel_name)
            .await;
    }

    /// Handle an authorized message: a cancel word, a message queued behind
    /// a running turn, or a new turn (and then every turn queued behind it).
    pub async fn submit(&self, surface: Arc<dyn TurnSurface>, turn: Turn) {
        if is_cancel_word(&turn.text) {
            let reply = self
                .cancel(&turn.session_id, turn.user_role, StopRequest::Bare)
                .await;
            let _ = surface.say(&reply).await;
            return;
        }

        let registry = &self.deps.task_registry;
        let session_id = turn.session_id.clone();
        let dedup_key = turn.dedup_key.as_deref();
        if !registry.has_running_task(&session_id).await {
            if let Some(request) = self.read_between_runs(&turn).await {
                if registry.mark_seen(&session_id, &turn.text, dedup_key).await {
                    let reply = self.cancel(&session_id, turn.user_role, request).await;
                    let _ = surface.say(&reply).await;
                }
                return;
            }
        }
        if registry.has_running_task(&session_id).await {
            let mut steerable = true;
            // The agent was asked to read it but gave no reading (its model
            // call failed): the owner still sees where the work stands.
            let mut unread = false;
            match self.read_mid_run(&turn).await {
                Some(
                    reading @ MidRunReading {
                        intent: MidRunIntent::StopsWork,
                        ..
                    },
                ) => {
                    if registry.mark_seen(&session_id, &turn.text, dedup_key).await {
                        let reply = self
                            .cancel(&session_id, turn.user_role, stop_request(&reading))
                            .await;
                        let _ = surface.say(&reply).await;
                    }
                    return;
                }
                Some(MidRunReading {
                    intent: MidRunIntent::AboutRunningWork,
                    ..
                }) => {
                    // Answered now from the typed state, read afresh with
                    // the clock `/tasks` uses; the run goes on undisturbed.
                    if registry.mark_seen(&session_id, &turn.text, dedup_key).await {
                        let answer = crate::mid_run::running_work_reply(
                            &self.running_work(&session_id).await,
                        );
                        let _ = surface
                            .say(&crate::presentation::prepare_chat_message(&answer))
                            .await;
                    }
                    return;
                }
                Some(MidRunReading {
                    intent: MidRunIntent::Unrelated,
                    ..
                }) => steerable = false,
                Some(MidRunReading {
                    intent: MidRunIntent::ChangesWork,
                    ..
                }) => {}
                None => unread = Self::mid_run_readable(&turn),
            }
            let mut queued = QueuedMessage::new(
                &turn.text,
                turn.user_role,
                turn.channel_ctx.clone(),
                turn.attachments.clone(),
            );
            if !steerable {
                queued = queued.not_steerable();
            }
            if let Some(timing) = turn.ingress_timing.clone() {
                queued = queued.with_ingress_timing(timing);
            }
            // Atomic check-and-queue: if the running turn finished since the
            // check above, run this one directly instead of stranding it.
            match registry
                .queue_message_if_running(&session_id, queued, dedup_key)
                .await
            {
                QueueOutcome::Queued(position) => {
                    // Notify for the first few only: a long message pasted
                    // in parts can arrive as many.
                    if position <= 3 {
                        let notice = if unread {
                            crate::mid_run::unread_message_notice(
                                &self.running_work(&session_id).await,
                            )
                        } else {
                            crate::presentation::queued_message_notice(
                                !turn.attachments.is_empty() || !steerable,
                            )
                        };
                        if let Ok(ids) = surface.say(&notice).await {
                            if let [message_id] = ids.as_slice() {
                                self.remember_queued_notice(
                                    &session_id,
                                    &turn.text,
                                    message_id.clone(),
                                );
                            }
                        }
                    }
                    return;
                }
                QueueOutcome::Duplicate => {
                    debug!(session_id, "Dropped duplicate queued message");
                    return;
                }
                QueueOutcome::NoRunningTask => {
                    info!(
                        session_id,
                        "Task finished during queue attempt — processing message directly"
                    );
                }
            }
        }

        // Mark the message seen so concurrent deliveries of it don't all run.
        if !registry.mark_seen(&session_id, &turn.text, dedup_key).await {
            debug!(
                session_id,
                "Dropped duplicate message (direct processing race)"
            );
            return;
        }

        let runner = self.clone();
        tokio::spawn(async move { runner.run(surface, turn).await });
    }

    /// The session's running work as its typed state shows it: the running
    /// list `/tasks` reads and the turn's live progress, timed at one
    /// instant.
    async fn running_work(&self, session_id: &str) -> RunningWork {
        let listed = crate::stop::running_work(
            &self.deps.task_registry,
            self.deps.agent.as_ref(),
            session_id,
        )
        .await;
        let now = chrono::Utc::now();
        let mut work = self.running_turn(session_id, now).await;
        work.running = listed.items(now);
        work
    }

    /// The session's running turn as its live state shows it at `now`.
    async fn running_turn(
        &self,
        session_id: &str,
        now: chrono::DateTime<chrono::Utc>,
    ) -> RunningWork {
        let running = self
            .deps
            .task_registry
            .list_for_session(session_id)
            .await
            .into_iter()
            .find(|entry| matches!(entry.status, crate::tasks::TaskStatus::Running));
        let (request, elapsed) = running.map_or_else(
            || (String::new(), Duration::ZERO),
            |entry| {
                let elapsed = (now - entry.started_at).to_std().unwrap_or_default();
                (entry.description, elapsed)
            },
        );
        self.live.snapshot(session_id, &request, elapsed)
    }

    /// Whether the agent reads `turn` sent mid-run: the owner's own text,
    /// without files.
    fn mid_run_readable(turn: &Turn) -> bool {
        turn.user_role == UserRole::Owner
            && turn.attachments.is_empty()
            && !turn.text.trim().is_empty()
    }

    /// Have the agent read a message the owner sent while a turn runs:
    /// only the owner's own text is read (another sender's message or one
    /// with files joins the queue as before), within a time limit.
    async fn read_mid_run(&self, turn: &Turn) -> Option<MidRunReading> {
        if !Self::mid_run_readable(turn) {
            return None;
        }
        let work = self.running_work(&turn.session_id).await;
        let reading = tokio::time::timeout(
            MID_RUN_READ_LIMIT,
            self.deps
                .agent
                .read_mid_run_message(&turn.session_id, &turn.text, &work),
        )
        .await
        .ok()
        .flatten();
        info!(
            session_id = %turn.session_id,
            intent = ?reading.as_ref().map(|r| r.intent),
            "Read a message sent mid-run"
        );
        reading
    }

    /// The reply to a cancel word, or to a message read as a stop request:
    /// the same stop as `/cancel`, read back before it is reported.
    async fn cancel(&self, session_id: &str, role: UserRole, request: StopRequest) -> String {
        if role != UserRole::Owner {
            return "Only the owner can cancel running work in this session.".to_string();
        }
        crate::stop::stop(
            &self.deps.task_registry,
            self.deps.agent.as_ref(),
            session_id,
            request,
        )
        .await
    }

    /// Read a message sent while the chat has sub-agents or background
    /// commands running but no turn: a stop request stops them (the same
    /// stop as `/cancel`); anything else runs as a turn. `None` when it is
    /// not a stop request or nothing runs.
    async fn read_between_runs(&self, turn: &Turn) -> Option<StopRequest> {
        if turn.user_role != UserRole::Owner
            || !turn.attachments.is_empty()
            || turn.text.trim().is_empty()
        {
            return None;
        }
        let list = crate::stop::running_work(
            &self.deps.task_registry,
            self.deps.agent.as_ref(),
            &turn.session_id,
        )
        .await;
        if list.entries.is_empty() {
            return None;
        }
        let work = RunningWork {
            running: list.items(chrono::Utc::now()),
            ..RunningWork::default()
        };
        let reading = tokio::time::timeout(
            MID_RUN_READ_LIMIT,
            self.deps
                .agent
                .read_mid_run_message(&turn.session_id, &turn.text, &work),
        )
        .await
        .ok()
        .flatten()?;
        info!(
            session_id = %turn.session_id,
            intent = ?reading.intent,
            "Read a message sent while background work runs"
        );
        (reading.intent == MidRunIntent::StopsWork).then(|| stop_request(&reading))
    }

    /// Run `turn` and then each turn queued behind it.
    /// Run `first`, then every turn queued behind it, in order.
    pub async fn run(self, surface: Arc<dyn TurnSurface>, first: Turn) {
        let registry = Arc::clone(&self.deps.task_registry);
        let agent = Arc::clone(&self.deps.agent);
        let stale_secs = self.deps.watchdog_stale_threshold_secs;
        let session_id = first.session_id;
        let mut text = first.text;
        let mut attachments = first.attachments;
        let mut role = first.user_role;
        let mut channel_ctx = first.channel_ctx;
        let mut timing = first.ingress_timing;
        // Stops the working indicator even if this task panics.
        let indicator = Arc::new(std::sync::Mutex::new(CancellationToken::new()));
        let _indicator_guard = CancelOnDrop(indicator.clone());

        loop {
            // Each turn gets the whole limit; a message queued behind a long
            // turn does not inherit what that turn used.
            let owner_wait = OwnerWaitClock::default();
            // Shown in "current task (…)" notices and cancel replies.
            let description = crate::presentation::word_preview(&text, 80);
            let (task_id, cancel) = registry.register(&session_id, &description).await;
            self.live.begin(&session_id, task_id);
            let heartbeat = Arc::new(AtomicU64::new(now_secs()));
            registry.set_heartbeat(task_id, heartbeat.clone()).await;
            let working = CancellationToken::new();
            if let Ok(mut slot) = indicator.lock() {
                *slot = working.clone();
            }
            surface.show_working(working.clone(), heartbeat.clone(), stale_secs);
            registry.set_typing_cancel(task_id, working.clone()).await;
            let (status_tx, mut run_updates) = mpsc::channel::<StatusUpdate>(16);
            let (shown_tx, status_rx) = mpsc::channel::<StatusUpdate>(16);
            // Each update is noted as the work's live state, then shown.
            {
                let live = self.live.clone();
                let session_id = session_id.clone();
                tokio::spawn(async move {
                    while let Some(update) = run_updates.recv().await {
                        live.record(&session_id, task_id, &update);
                        let _ = shown_tx.send(update).await;
                    }
                });
            }
            let private = channel_ctx.visibility == ChannelVisibility::Private;
            let status_task = surface
                .show_status(status_rx, private)
                .unwrap_or_else(|updates| throttled_status(surface.clone(), updates, private));
            if let Some(timing) = timing.as_mut() {
                timing.mark_dispatched_now();
            }

            let result = tokio::select! {
                result = OwnerWaitClock::within(
                    Some(owner_wait.clone()),
                    agent.handle_inbound_message(InboundMessageRequest {
                        session_id: session_id.clone(),
                        user_text: text.clone(),
                        attachments: attachments.clone(),
                        status_tx: Some(status_tx),
                        user_role: role,
                        channel_ctx: channel_ctx.clone(),
                        heartbeat: Some(heartbeat.clone()),
                        ingress_timing: timing.clone(),
                    }),
                ) => result,
                _ = cancel.cancelled() => Err(anyhow::anyhow!(CANCELLED)),
                stale_mins = wait_for_stale_heartbeat(heartbeat.clone(), stale_secs, 4), if stale_secs > 0 => {
                    Err(anyhow::anyhow!(
                        "{INACTIVITY_PREFIX} ({} minute{} without progress).",
                        stale_mins,
                        if stale_mins == 1 { "" } else { "s" }
                    ))
                },
            };
            working.cancel();
            self.live.end(&session_id, task_id);
            // The run is over, so its status sender is gone and the surface
            // finishes showing what it sent: the closing checklist must reach
            // the card before the reply, or the card stays on "Working" above
            // a reply. A background job may still hold a sender, so the wait
            // is bounded and a handoff, whose job keeps running, skips it.
            let mut status_task = status_task;
            let run_finished = matches!(
                &result,
                Ok(envelope) if envelope.disposition
                    != aidaemon_events::AssistantResponseDisposition::BackgroundHandoff
            );
            if run_finished && surface.status_settles_before_reply() {
                let _ = tokio::time::timeout(STATUS_DRAIN, &mut status_task).await;
            }
            status_task.abort();

            // The task stays running while the reply is sent, so a message
            // arriving meanwhile queues instead of starting a second turn.
            let error = match result {
                Ok(envelope) => {
                    let platform = surface.platform();
                    let _ = agent
                        .record_response_delivery(
                            &session_id,
                            envelope.delivery(
                                platform,
                                ResponseDeliveryState::Queued,
                                Vec::new(),
                                None,
                            ),
                        )
                        .await;
                    let reply = crate::presentation::prepare_chat_message(&envelope.text);
                    let tasks: Vec<(u64, String)> = registry
                        .list_for_session(&session_id)
                        .await
                        .into_iter()
                        .map(|entry| (entry.id, entry.description))
                        .collect();
                    let reply = crate::presentation::render_task_numbers(&reply, &tasks);
                    let empty = reply.trim().is_empty();
                    let delivered = if empty {
                        Delivered {
                            state: ResponseDeliveryState::PlatformAcknowledged,
                            message_ids: Vec::new(),
                            error: None,
                        }
                    } else {
                        surface.deliver(&envelope, &reply).await
                    };
                    let _ = agent
                        .record_response_delivery(
                            &session_id,
                            envelope.delivery(
                                platform,
                                delivered.state,
                                delivered.message_ids,
                                delivered.error,
                            ),
                        )
                        .await;
                    surface
                        .turn_ended(if empty {
                            TurnEnd::Empty
                        } else {
                            TurnEnd::Delivered
                        })
                        .await;
                    None
                }
                Err(error) => {
                    let message = error.to_string();
                    if message == CANCELLED {
                        info!("Task #{} cancelled", task_id);
                        surface.turn_ended(TurnEnd::Cancelled).await;
                    } else {
                        if message.starts_with(INACTIVITY_PREFIX) {
                            info!("Task #{} auto-cancelled by stale watchdog", task_id);
                            let _ = surface.say(&format!("⚠️ {message}")).await;
                        } else {
                            warn!("Agent error: {}", error);
                            let _ = surface.say(&format!("Error: {message}")).await;
                        }
                        surface.turn_ended(TurnEnd::Failed).await;
                    }
                    Some(message)
                }
            };

            if let Some(error) = error.as_ref() {
                let status = if error == CANCELLED {
                    aidaemon_events::TaskStatus::Cancelled
                } else {
                    aidaemon_events::TaskStatus::Failed
                };
                if let Err(terminalize_error) = agent
                    .terminalize_supervised_task(&session_id, status, error.clone())
                    .await
                {
                    warn!(%terminalize_error, "Failed to persist supervised task termination");
                }
            }

            // Finalize and drain together, so a message arriving meanwhile is
            // either drained here or runs directly, never stranded.
            let Some(queued) = registry
                .finalize_and_drain(task_id, &session_id, error.as_deref())
                .await
            else {
                self.forget_queued_notices(&session_id);
                break;
            };
            // Let the finished turn's messages commit first.
            tokio::time::sleep(Duration::from_millis(100)).await;
            let notice = self.take_queued_notice(&session_id, &queued.text);
            let queued = surface.before_queued(queued).await;
            let preview = crate::presentation::word_preview(&queued.text, 50);
            info!(session_id, "Processing queued message: {}", preview);
            let starting = crate::presentation::queued_message_starting(&queued.text);
            let edited = match notice {
                Some(message_id) => matches!(surface.edit(&message_id, &starting).await, Ok(true)),
                None => false,
            };
            if !edited {
                let _ = surface.say(&starting).await;
            }
            text = queued.text;
            role = queued.user_role;
            channel_ctx = queued.channel_ctx;
            attachments = queued.attachments;
            timing = queued.ingress_timing;
        }
    }
}

/// Cancels the current token when dropped.
struct CancelOnDrop(Arc<std::sync::Mutex<CancellationToken>>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Ok(token) = self.0.lock() {
            token.cancel();
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// `cancel`, `stop` or `abort`, said on its own.
pub fn is_cancel_word(text: &str) -> bool {
    matches!(
        text.trim().to_lowercase().as_str(),
        "cancel" | "stop" | "abort"
    )
}

/// The stop a reading names: its items, or what bare `/cancel` stops.
fn stop_request(reading: &MidRunReading) -> StopRequest {
    if reading.stop.is_empty() {
        StopRequest::Bare
    } else {
        StopRequest::Numbers(reading.stop.clone())
    }
}

/// Status messages, at most one per [`STATUS_INTERVAL`] unless the update
/// must be seen at once. Outside private conversations only updates that
/// must be seen are sent; the working indicator says the rest.
fn throttled_status(
    surface: Arc<dyn TurnSurface>,
    mut updates: mpsc::Receiver<StatusUpdate>,
    private: bool,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut last_sent: Option<tokio::time::Instant> = None;
        while let Some(update) = updates.recv().await {
            let immediate = update.requires_immediate_delivery();
            if !private && !immediate {
                continue;
            }
            let now = tokio::time::Instant::now();
            if !immediate && last_sent.is_some_and(|at| now.duration_since(at) < STATUS_INTERVAL) {
                continue;
            }
            let _ = surface.say(&status_line(&update)).await;
            last_sent = Some(tokio::time::Instant::now());
        }
    })
}

/// The status line for one update, as Markdown.
pub fn status_line(update: &StatusUpdate) -> String {
    let progress = |text: String| format!("_{text}_");
    match update {
        StatusUpdate::Thinking(_) => progress("Thinking...".to_string()),
        StatusUpdate::ToolStart { name, summary } => {
            let mut chars = name.chars();
            let name: String = match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            };
            if summary.is_empty() {
                progress(format!("{name}..."))
            } else {
                progress(format!("{name}: {summary}..."))
            }
        }
        StatusUpdate::ToolProgress { name, chunk } => {
            let preview: String = chunk.chars().take(100).collect();
            let more = if chunk.chars().count() > 100 {
                "..."
            } else {
                ""
            };
            progress(format!("📤 {name}: {preview}{more}"))
        }
        StatusUpdate::UserActionRequired { message } => message.clone(),
        StatusUpdate::ToolComplete { name, summary } => progress(format!("✓ {name}: {summary}")),
        StatusUpdate::ToolCancellable { name, task_id } => {
            progress(format!("⏳ {name} started (task_id: {task_id})"))
        }
        StatusUpdate::ProgressSummary {
            elapsed_mins,
            summary,
        } => progress(format!("📊 Progress ({elapsed_mins} min): {summary}")),
        StatusUpdate::IterationWarning { current, threshold } => progress(format!(
            "⚠️ Approaching soft limit: {current} of {threshold} iterations"
        )),
        StatusUpdate::PlanCreated {
            description,
            total_steps,
            ..
        } => progress(format!(
            "📋 Plan created: {description} ({total_steps} steps)"
        )),
        StatusUpdate::PlanStepStart {
            step_index,
            total_steps,
            description,
            ..
        } => progress(format!(
            "▶️ Step {}/{total_steps}: {description}",
            step_index + 1
        )),
        StatusUpdate::PlanStepComplete {
            step_index,
            total_steps,
            description,
            summary,
            ..
        } => {
            let base = format!("✅ Step {}/{total_steps} done: {description}", step_index + 1);
            match summary {
                Some(summary) => progress(format!("{base} - {summary}")),
                None => progress(base),
            }
        }
        StatusUpdate::PlanStepFailed {
            step_index,
            description,
            error,
            ..
        } => progress(format!(
            "❌ Step {} failed: {description} - {error}",
            step_index + 1
        )),
        StatusUpdate::PlanComplete {
            description,
            total_steps,
            duration_secs,
            ..
        } => progress(format!(
            "🎉 Plan complete: {description} ({total_steps} steps in {}m {}s)",
            duration_secs / 60,
            duration_secs % 60
        )),
        StatusUpdate::PlanAbandoned { description, .. } => {
            progress(format!("🚫 Plan abandoned: {description}"))
        }
        StatusUpdate::PlanRevised {
            description,
            reason,
            new_total_steps,
            ..
        } => progress(format!(
            "🔄 Plan revised: {description} ({new_total_steps} steps) - {reason}"
        )),
        StatusUpdate::BudgetExtended {
            old_budget,
            new_budget,
            extension,
            max_extensions,
        } => progress(format!(
            "💰 Auto-extended token budget {old_budget} → {new_budget} ({extension}/{max_extensions}) — continuing."
        )),
        StatusUpdate::Checklist { text } => text.clone(),
        StatusUpdate::SubAgentProgress {
            label,
            progress: step,
        } => progress(format!("🤖 {label}: {step}")),
        StatusUpdate::HelperStarted { label } => {
            progress(crate::live_status::helper_started_line(label))
        }
        StatusUpdate::StepHandedOff { name, to_helper } => progress(
            crate::live_status::step_handed_off_line(name, *to_helper),
        ),
    }
}

/// Repeat `send` every `every` until `stop` is cancelled, the heartbeat has
/// been quiet for `stale_secs` (0 = never), or 30 minutes pass.
pub fn repeat_while_working<F, Fut>(
    every: Duration,
    stop: CancellationToken,
    heartbeat: Arc<AtomicU64>,
    stale_secs: u64,
    mut send: F,
) where
    F: FnMut() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    tokio::spawn(async move {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30 * 60);
        loop {
            send().await;
            tokio::select! {
                _ = tokio::time::sleep(every) => {
                    if stale_secs > 0
                        && now_secs().saturating_sub(heartbeat.load(Ordering::Relaxed)) > stale_secs
                    {
                        break;
                    }
                    if tokio::time::Instant::now() >= deadline {
                        warn!("Working indicator reached its 30-minute limit");
                        break;
                    }
                }
                _ = stop.cancelled() => break,
            }
        }
    });
}

/// Wait until the heartbeat becomes stale and return stale minutes.
///
/// `stale_threshold_secs` must be > 0.
pub async fn wait_for_stale_heartbeat(
    heartbeat: Arc<AtomicU64>,
    stale_threshold_secs: u64,
    check_interval_secs: u64,
) -> u64 {
    debug_assert!(stale_threshold_secs > 0);

    loop {
        tokio::time::sleep(Duration::from_secs(check_interval_secs.max(1))).await;
        let last_hb = heartbeat.load(Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if now.saturating_sub(last_hb) > stale_threshold_secs {
            return now.saturating_sub(last_hb) / 60;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stale_heartbeat_returns_minutes() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let heartbeat = Arc::new(AtomicU64::new(now.saturating_sub(125)));

        let mins = tokio::time::timeout(
            Duration::from_secs(3),
            wait_for_stale_heartbeat(heartbeat, 60, 1),
        )
        .await
        .expect("stale heartbeat should resolve quickly");

        assert!(mins >= 2);
    }

    /// An agent whose turn runs until released and whose reading of a
    /// mid-run message is scripted.
    struct ScriptedAgent {
        release: tokio::sync::Notify,
        reading: std::sync::Mutex<Option<MidRunReading>>,
        /// The chat's sub-agents and background commands; a stop removes one.
        work: std::sync::Mutex<Vec<crate::ports::ChatWork>>,
    }

    #[async_trait]
    impl crate::ports::AgentIngress for ScriptedAgent {
        async fn handle_inbound_message(
            &self,
            request: InboundMessageRequest,
        ) -> anyhow::Result<AgentResponseEnvelope> {
            if let Some(status) = request.status_tx.as_ref() {
                let _ = status
                    .send(StatusUpdate::ToolStart {
                        name: "terminal".into(),
                        summary: "installing the blog".into(),
                    })
                    .await;
            }
            self.release.notified().await;
            Ok(AgentResponseEnvelope {
                response_id: "r".into(),
                task_id: "t".into(),
                turn_id: None,
                text: "Done.".into(),
                referenced_receipts: Vec::new(),
                disposition: Default::default(),
            })
        }
    }

    #[async_trait]
    impl ChannelAgentRuntime for ScriptedAgent {
        async fn read_mid_run_message(
            &self,
            _session_id: &str,
            _text: &str,
            _work: &RunningWork,
        ) -> Option<MidRunReading> {
            self.reading.lock().unwrap().clone()
        }
        async fn chat_work(&self, _session_id: &str) -> crate::ports::ChatWorkView {
            crate::ports::ChatWorkView {
                current_request: None,
                items: self.work.lock().unwrap().clone(),
            }
        }
        async fn stop_chat_work(&self, _session_id: &str, key: &str) -> bool {
            let mut work = self.work.lock().unwrap();
            work.retain(|item| item.key != key);
            true
        }
        async fn terminalize_supervised_task(
            &self,
            _: &str,
            _: aidaemon_events::TaskStatus,
            _: String,
        ) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
        async fn cancel_active_finite_work_for_session(&self, _: &str) -> Vec<String> {
            Vec::new()
        }
        async fn current_model(&self) -> String {
            String::new()
        }
        async fn context_debug_settings(
            &self,
            _: &str,
            _: &str,
        ) -> (bool, usize, usize, usize, Option<i64>) {
            (false, 0, 0, 0, None)
        }
        async fn set_model(&self, _: String) {}
        async fn list_models(&self) -> anyhow::Result<Vec<String>> {
            Ok(Vec::new())
        }
        async fn clear_model_override(&self) {}
        async fn clear_session_context(&self, _: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn clear_session(&self, _: &str) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct NoSessions;
    impl SessionChannelStore for NoSessions {}

    #[derive(Default)]
    struct Said(std::sync::Mutex<Vec<String>>);

    #[async_trait]
    impl TurnSurface for Said {
        fn platform(&self) -> &'static str {
            "test"
        }
        async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
            self.0.lock().unwrap().push(text.to_string());
            Ok(Vec::new())
        }
        fn show_working(&self, _: CancellationToken, _: Arc<AtomicU64>, _: u64) {}
    }

    fn owner_turn(text: &str, key: &str) -> Turn {
        Turn {
            session_id: "test:mid-run".into(),
            text: text.into(),
            attachments: Vec::new(),
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("test"),
            dedup_key: Some(key.into()),
            ingress_timing: None,
        }
    }

    /// A question about the running work is answered at once from its live
    /// state and never reaches the run; a change joins the run; anything
    /// else waits for its own turn.
    #[tokio::test]
    async fn a_mid_run_message_is_answered_joined_or_kept_by_its_reading() {
        let agent = Arc::new(ScriptedAgent {
            release: tokio::sync::Notify::new(),
            reading: std::sync::Mutex::new(None),
            work: std::sync::Mutex::new(Vec::new()),
        });
        let registry = Arc::new(TaskRegistry::new(10));
        let runner = TurnRunner::new(TurnDeps {
            agent: agent.clone(),
            session_map: Arc::default(),
            sessions: Arc::new(NoSessions),
            task_registry: registry.clone(),
            watchdog_stale_threshold_secs: 0,
        });
        let surface = Arc::new(Said::default());
        let session = "test:mid-run";
        runner
            .submit(surface.clone(), owner_turn("Set up two demo sites", "m1"))
            .await;
        for _ in 0..200 {
            if registry.has_running_task(session).await
                && runner.running_work(session).await.last_step.is_some()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let set = |reading: Option<MidRunReading>| *agent.reading.lock().unwrap() = reading;
        let last = |surface: &Said| {
            surface
                .0
                .lock()
                .unwrap()
                .last()
                .cloned()
                .unwrap_or_default()
        };

        // A question: the typed state itself is the reply.
        set(Some(MidRunReading {
            intent: MidRunIntent::AboutRunningWork,
            stop: Vec::new(),
        }));
        runner
            .submit(surface.clone(), owner_turn("How is it going so far?", "m2"))
            .await;
        let reply = last(&surface);
        assert!(
            reply.starts_with("Still working on \"Set up two demo sites\""),
            "{reply}"
        );
        assert!(
            reply.contains("Latest step: installing the blog"),
            "{reply}"
        );
        // Worded differently, it gets the same typed state.
        runner
            .submit(
                surface.clone(),
                owner_turn("Send me the screenshots now.", "m3"),
            )
            .await;
        let reply = last(&surface);
        assert!(
            reply.starts_with("Still working on \"Set up two demo sites\""),
            "{reply}"
        );
        assert_eq!(registry.queue_len(session).await, 0);

        // Something else: queued for its own turn, not taken into the run.
        set(Some(MidRunReading {
            intent: MidRunIntent::Unrelated,
            stop: Vec::new(),
        }));
        runner
            .submit(
                surface.clone(),
                owner_turn("What is the capital of Peru?", "m4"),
            )
            .await;
        assert_eq!(
            last(&surface),
            crate::presentation::queued_message_notice(true)
        );
        let ctx = ChannelContext::private("test");
        assert!(registry
            .take_steering_messages(session, UserRole::Owner, &ctx)
            .await
            .is_empty());
        registry.clear_queue(session).await;

        // A change to the work (or no reading at all): it joins the run.
        // With no reading (the model call failed) the owner also sees where
        // the work stands, so a question is answered either way.
        for (reading, key) in [
            (
                Some(MidRunReading {
                    intent: MidRunIntent::ChangesWork,
                    stop: Vec::new(),
                }),
                "m5",
            ),
            (None, "m6"),
        ] {
            let unread = reading.is_none();
            set(reading);
            runner
                .submit(surface.clone(), owner_turn("how's it going?", key))
                .await;
            let notice = last(&surface);
            assert!(
                notice.starts_with(&crate::presentation::queued_message_notice(false)),
                "{notice}"
            );
            assert_eq!(
                notice.contains("Still working on \"Set up two demo sites\""),
                unread,
                "{notice}"
            );
            assert_eq!(
                notice.contains("Latest step: installing the blog"),
                unread,
                "{notice}"
            );
            let taken = registry
                .take_steering_messages(session, UserRole::Owner, &ctx)
                .await;
            assert_eq!(taken.len(), 1);
        }

        agent.release.notify_one();
    }

    #[test]
    fn cancel_words_are_whole_messages() {
        assert!(is_cancel_word("Stop"));
        assert!(is_cancel_word(" cancel "));
        assert!(!is_cancel_word("stop the server"));
        assert!(!is_cancel_word("don't stop"));
    }

    /// Two background commands started by one request: a bare stop word
    /// stops both and names each, read back as gone; with nothing left it
    /// says so. A stop request read in plain words between runs takes the
    /// same path.
    #[tokio::test]
    async fn a_bare_stop_stops_every_job_of_one_request_and_names_them() {
        let agent = Arc::new(ScriptedAgent {
            release: tokio::sync::Notify::new(),
            reading: std::sync::Mutex::new(None),
            work: std::sync::Mutex::new(two_counting_jobs()),
        });
        let registry = Arc::new(TaskRegistry::new(10));
        let runner = TurnRunner::new(TurnDeps {
            agent: agent.clone(),
            session_map: Arc::default(),
            sessions: Arc::new(NoSessions),
            task_registry: registry.clone(),
            watchdog_stale_threshold_secs: 0,
        });
        let surface = Arc::new(Said::default());
        runner
            .submit(surface.clone(), owner_turn("stop", "s1"))
            .await;
        let reply = surface.0.lock().unwrap().last().cloned().unwrap();
        assert_eq!(
            reply,
            "⏹️ Stopped:\n• Sub-agent: Counting agent A\n• Background command: count to 300"
        );
        assert!(agent.work.lock().unwrap().is_empty());
        runner
            .submit(surface.clone(), owner_turn("stop", "s2"))
            .await;
        assert_eq!(
            surface.0.lock().unwrap().last().cloned().unwrap(),
            "Nothing is running in this chat right now."
        );

        // "Stop both counting agents now", read as a stop request: no turn
        // starts, and the same read-back stop runs.
        *agent.work.lock().unwrap() = two_counting_jobs();
        *agent.reading.lock().unwrap() = Some(MidRunReading {
            intent: MidRunIntent::StopsWork,
            stop: Vec::new(),
        });
        runner
            .submit(
                surface.clone(),
                owner_turn("Stop both counting agents now.", "s3"),
            )
            .await;
        assert!(!registry.has_running_task("test:mid-run").await);
        let reply = surface.0.lock().unwrap().last().cloned().unwrap();
        assert!(reply.contains("• Sub-agent: Counting agent A"), "{reply}");
        assert!(
            reply.contains("• Background command: count to 300"),
            "{reply}"
        );
        assert!(agent.work.lock().unwrap().is_empty());
    }

    /// Work from two requests: a bare stop lists it to pick from and stops
    /// nothing; a number stops just that one.
    #[tokio::test]
    async fn a_bare_stop_over_two_requests_asks_which() {
        let mut work = two_counting_jobs();
        work[1].request = Some("req-2".into());
        let agent = Arc::new(ScriptedAgent {
            release: tokio::sync::Notify::new(),
            reading: std::sync::Mutex::new(None),
            work: std::sync::Mutex::new(work),
        });
        let registry = TaskRegistry::new(10);
        let reply =
            crate::stop::stop(&registry, agent.as_ref(), "test:mid-run", StopRequest::Bare).await;
        assert!(
            reply.starts_with("Work from more than one request is running:"),
            "{reply}"
        );
        assert!(reply.contains("/cancel all"), "{reply}");
        assert_eq!(agent.work.lock().unwrap().len(), 2);
        let list = crate::stop::running_work(&registry, agent.as_ref(), "test:mid-run").await;
        let job = list.entries[1].number;
        let reply = crate::stop::stop(
            &registry,
            agent.as_ref(),
            "test:mid-run",
            StopRequest::Numbers(vec![job]),
        )
        .await;
        assert_eq!(reply, "⏹️ Stopped:\n• Background command: count to 300");
        assert_eq!(agent.work.lock().unwrap().len(), 1);
    }

    /// `/clear` and `/wipe` stop the chat's sub-agents and background
    /// commands through the same read-back stop.
    #[tokio::test]
    async fn clearing_stops_every_sub_agent_and_command() {
        let agent = Arc::new(ScriptedAgent {
            release: tokio::sync::Notify::new(),
            reading: std::sync::Mutex::new(None),
            work: std::sync::Mutex::new(two_counting_jobs()),
        });
        let registry = TaskRegistry::new(10);
        let stopped =
            crate::stop::stop_background_work(&registry, agent.as_ref(), "test:mid-run").await;
        assert_eq!(stopped.len(), 2);
        // `/clear` names what stopped, as `/cancel` does.
        let named = crate::stop::stopped_list(&stopped);
        assert!(named.starts_with("⏹️ Stopped:\n• "), "{named}");
        for title in &stopped {
            assert!(named.contains(title.as_str()), "{named}");
        }
        assert!(agent.work.lock().unwrap().is_empty());
    }

    fn two_counting_jobs() -> Vec<crate::ports::ChatWork> {
        let now = chrono::Utc::now();
        vec![
            crate::ports::ChatWork {
                key: "agent:1".into(),
                kind: crate::ports::ChatWorkKind::SubAgent,
                title: "Sub-agent: Counting agent A".into(),
                started_at: now - chrono::Duration::seconds(60),
                request: Some("req-1".into()),
            },
            crate::ports::ChatWork {
                key: "job:0:4242".into(),
                kind: crate::ports::ChatWorkKind::BackgroundJob,
                title: "Background command: count to 300".into(),
                started_at: now - chrono::Duration::seconds(50),
                request: Some("req-1".into()),
            },
        ]
    }
}
