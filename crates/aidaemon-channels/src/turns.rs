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

    /// Where this surface keeps a turn's one status card, when it keeps
    /// one (a message it can create, edit and delete). The runner then owns
    /// the card: what it shows and how it ends with the turn, by the one
    /// rule ([`aidaemon_core::card::CardEnd`]). `None` (the default): the
    /// turn's status goes to [`Self::show_status`].
    fn card_sink(&self) -> Option<Arc<dyn crate::live_status::SurfaceSink>> {
        None
    }

    /// The turn kept its card: message `message_id` stays as the record of
    /// the work, for what outlives the turn (a helper ending) to settle.
    async fn card_kept(&self, _message_id: &str) {}

    /// Show one turn's status updates. Handing them back (the default)
    /// uses throttled status messages, sent only in private conversations.
    /// Not called for a surface with a [`Self::card_sink`].
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

tokio::task_local! {
    static TURN_OVER: TurnOver;
}

/// Marks the end of the turn running this task: everything it shows is out
/// (its reply and the card it kept). Work the turn started that outlives it
/// (a background helper) waits on this before it touches the chat, so the
/// turn's reply always comes before the work's own end.
#[derive(Clone)]
pub struct TurnOver {
    over: CancellationToken,
    /// When the owner's request reached the runner: what a time "since your
    /// request" is measured from.
    requested_at: chrono::DateTime<chrono::Utc>,
}

impl Default for TurnOver {
    fn default() -> Self {
        Self {
            over: CancellationToken::new(),
            requested_at: chrono::Utc::now(),
        }
    }
}

impl TurnOver {
    /// When the owner's request that started this turn reached the runner.
    pub fn requested_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.requested_at
    }

    /// The turn is over: everything it shows is out.
    pub fn end(&self) {
        self.over.cancel();
    }

    /// The turn running this task, if any (scheduled work has none).
    pub fn current() -> Option<Self> {
        TURN_OVER.try_with(Clone::clone).ok()
    }

    /// Wait until that turn is over.
    pub async fn wait(&self) {
        self.over.cancelled().await;
    }

    /// Whether that turn is over.
    pub fn is_over(&self) -> bool {
        self.over.is_cancelled()
    }

    /// Run `future` as part of the turn this marks.
    pub async fn within<F: std::future::Future>(&self, future: F) -> F::Output {
        TURN_OVER.scope(self.clone(), future).await
    }
}

/// A turn's card on a surface that keeps one: its state and where it is
/// drawn. The runner owns it for the turn.
#[derive(Clone)]
struct TurnCard {
    live: Arc<tokio::sync::Mutex<crate::live_status::LiveStatus>>,
    sink: Arc<dyn crate::live_status::SurfaceSink>,
}

impl TurnCard {
    fn new(sink: Arc<dyn crate::live_status::SurfaceSink>) -> Self {
        Self {
            live: Arc::default(),
            sink,
        }
    }

    /// Deliver the reply, always on its own: the card is the state, and it
    /// ends after the reply ([`Self::close`]).
    async fn deliver(
        &self,
        surface: &dyn TurnSurface,
        envelope: &AgentResponseEnvelope,
        reply: &str,
    ) -> Delivered {
        surface.deliver(envelope, reply).await
    }

    /// End the card with the turn, once, by the one rule: kept and
    /// settled (handed to the surface for what outlives the turn), or
    /// removed.
    async fn close(&self, surface: &dyn TurnSurface) {
        let mut live = self.live.lock().await;
        live.close(self.sink.as_ref()).await;
        if let Some(id) = live.kept_card_message_id() {
            surface.card_kept(&id).await;
        }
    }
}

/// A turn as it starts, for a [`TurnObserver`].
pub struct TurnStart<'a> {
    pub session_id: &'a str,
    /// The task registry's id for the turn.
    pub turn: u64,
    /// The surface's platform (`telegram`, `app`...).
    pub platform: &'static str,
    /// What the sender said.
    pub text: &'a str,
    pub attachments: &'a [MessageAttachment],
    pub user_role: UserRole,
    /// How the running-work list names it.
    pub description: &'a str,
}

/// A turn as it ends, for a [`TurnObserver`].
pub struct TurnFinish<'a> {
    pub session_id: &'a str,
    pub turn: u64,
    pub platform: &'static str,
    pub end: TurnEnd,
    /// The reply as delivered (Markdown); empty when there was none.
    pub reply: &'a str,
    /// The work goes on in the background; its report comes later.
    pub handed_off: bool,
    /// Why it failed or was cancelled.
    pub error: Option<&'a str>,
}

/// Sees every turn a runner runs, whatever its platform, so one place can
/// show all the work (the Client Gateway). Called inline: return quickly.
pub trait TurnObserver: Send + Sync {
    fn turn_started(&self, turn: &TurnStart<'_>);
    /// One of the turn's status updates, as the surface gets it.
    fn turn_status(&self, session_id: &str, turn: u64, update: &StatusUpdate);
    fn turn_finished(&self, turn: &TurnFinish<'_>);
}

/// Where a runner reports its turns: empty until an observer is set, which
/// may happen after the runners are built. Clones share the slot.
#[derive(Clone, Default)]
pub struct TurnWatch(Arc<std::sync::RwLock<Option<Arc<dyn TurnObserver>>>>);

impl TurnWatch {
    /// Report turns to `observer` from now on.
    pub fn set(&self, observer: Arc<dyn TurnObserver>) {
        *self.0.write().unwrap_or_else(|p| p.into_inner()) = Some(observer);
    }

    /// The observer, if one is set.
    pub fn observer(&self) -> Option<Arc<dyn TurnObserver>> {
        self.0.read().unwrap_or_else(|p| p.into_inner()).clone()
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
    /// Who sees every turn (empty by default).
    pub watch: TurnWatch,
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
            match self.read_between_runs(&turn).await {
                Some(BetweenRuns::Stop(request)) => {
                    if registry.mark_seen(&session_id, &turn.text, dedup_key).await {
                        let reply = self.cancel(&session_id, turn.user_role, request).await;
                        let _ = surface.say(&reply).await;
                    }
                    return;
                }
                // A question about the helpers and commands running for
                // the chat: answered from their typed state, the same reply
                // as mid-run, never a turn of its own that would time
                // itself (live 2026-11-13: "Still working on \"how's it
                // going?\" (1s so far)").
                Some(BetweenRuns::Status) => {
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
                None => {}
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
    /// stop as `/cancel`), a question about them is answered from their
    /// typed state; anything else runs as a turn. `None` when it is neither
    /// or nothing runs.
    async fn read_between_runs(&self, turn: &Turn) -> Option<BetweenRuns> {
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
        match reading.intent {
            MidRunIntent::StopsWork => Some(BetweenRuns::Stop(stop_request(&reading))),
            MidRunIntent::AboutRunningWork => Some(BetweenRuns::Status),
            MidRunIntent::ChangesWork | MidRunIntent::Unrelated => None,
        }
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
            let observer = self.deps.watch.observer();
            if let Some(observer) = observer.as_ref() {
                observer.turn_started(&TurnStart {
                    session_id: &session_id,
                    turn: task_id,
                    platform: surface.platform(),
                    text: &text,
                    attachments: &attachments,
                    user_role: role,
                    description: &description,
                });
            }
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
                let observer = observer.clone();
                tokio::spawn(async move {
                    while let Some(update) = run_updates.recv().await {
                        // Reply text is for app clients: no chat surface,
                        // card or live state shows it.
                        if matches!(update, StatusUpdate::ReplyText(_)) {
                            if let Some(observer) = observer.as_ref() {
                                observer.turn_status(&session_id, task_id, &update);
                            }
                            continue;
                        }
                        live.record(&session_id, task_id, &update);
                        if let Some(observer) = observer.as_ref() {
                            observer.turn_status(&session_id, task_id, &update);
                        }
                        let _ = shown_tx.send(update).await;
                    }
                });
            }
            let private = channel_ctx.visibility == ChannelVisibility::Private;
            // A surface that keeps a card gets it from the runner, which
            // owns how it ends; any other shows the updates its own way.
            let card = surface.card_sink().map(TurnCard::new);
            let settle = CancellationToken::new();
            let status_task = match &card {
                Some(card) => tokio::spawn(show_on_card(
                    surface.clone(),
                    card.clone(),
                    status_rx,
                    private,
                    settle.clone(),
                )),
                None => surface
                    .show_status(status_rx, private)
                    .unwrap_or_else(|updates| throttled_status(surface.clone(), updates, private)),
            };
            // Marked once everything the turn shows is out, however this
            // iteration ends.
            let turn_over = TurnOver::default();
            let over_when_dropped = turn_over.over.clone().drop_guard();
            if let Some(timing) = timing.as_mut() {
                timing.mark_dispatched_now();
            }

            let result = tokio::select! {
                result = turn_over.within(OwnerWaitClock::within(
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
                )) => result,
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
            // The run is over: what it sent must reach the card before the
            // reply, or the card stays on "Working" above a reply (and a
            // helper it started is missing when the card's end is decided).
            // The card shows what is queued and stops, whatever still holds
            // a sender. Another surface finishes showing what it sent; a
            // background job may still hold a sender, so that wait is
            // bounded and a handoff, whose job keeps running, skips it.
            let mut status_task = status_task;
            let run_finished = matches!(
                &result,
                Ok(envelope) if envelope.disposition
                    != aidaemon_events::AssistantResponseDisposition::BackgroundHandoff
            );
            if card.is_some() {
                settle.cancel();
                let _ = tokio::time::timeout(STATUS_DRAIN, &mut status_task).await;
            } else if run_finished && surface.status_settles_before_reply() {
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
                        match &card {
                            Some(card) => card.deliver(surface.as_ref(), &envelope, &reply).await,
                            None => surface.deliver(&envelope, &reply).await,
                        }
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
                    let end = if empty {
                        TurnEnd::Empty
                    } else {
                        TurnEnd::Delivered
                    };
                    if let Some(observer) = observer.as_ref() {
                        observer.turn_finished(&TurnFinish {
                            session_id: &session_id,
                            turn: task_id,
                            platform,
                            end,
                            reply: if empty { "" } else { &reply },
                            handed_off: envelope.disposition
                                == aidaemon_events::AssistantResponseDisposition::BackgroundHandoff,
                            error: None,
                        });
                    }
                    if let Some(card) = &card {
                        card.close(surface.as_ref()).await;
                    }
                    surface.turn_ended(end).await;
                    None
                }
                Err(error) => {
                    let message = error.to_string();
                    if message == CANCELLED {
                        info!("Task #{} cancelled", task_id);
                    } else if message.starts_with(INACTIVITY_PREFIX) {
                        info!("Task #{} auto-cancelled by stale watchdog", task_id);
                        let _ = surface.say(&format!("⚠️ {message}")).await;
                    } else {
                        warn!("Agent error: {}", error);
                        let _ = surface.say(&format!("Error: {message}")).await;
                    }
                    if let Some(card) = &card {
                        card.close(surface.as_ref()).await;
                    }
                    surface
                        .turn_ended(if message == CANCELLED {
                            TurnEnd::Cancelled
                        } else {
                            TurnEnd::Failed
                        })
                        .await;
                    if let Some(observer) = observer.as_ref() {
                        observer.turn_finished(&TurnFinish {
                            session_id: &session_id,
                            turn: task_id,
                            platform: surface.platform(),
                            end: if message == CANCELLED {
                                TurnEnd::Cancelled
                            } else {
                                TurnEnd::Failed
                            },
                            reply: "",
                            handed_off: false,
                            error: Some(&message),
                        });
                    }
                    Some(message)
                }
            };

            drop(over_when_dropped);

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

/// What a message sent while only background work runs asks for.
enum BetweenRuns {
    /// Stop that work.
    Stop(StopRequest),
    /// Where that work stands.
    Status,
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
            let Some(line) = status_line(&update) else {
                continue;
            };
            let _ = surface.say(&line).await;
            last_sent = Some(tokio::time::Instant::now());
        }
    })
}

/// A turn's status updates on its card. Every update notes the card's
/// state ([`crate::live_status::LiveStatus::note`]); the card is sent at
/// most once per [`STATUS_INTERVAL`], and an update that arrives inside the
/// interval is shown when it ends instead of being dropped (live
/// 2026-11-12: a throttled step left the card stale through six minutes of
/// helpers working). A long step's line says how long it has run, refreshed
/// on a tick. Updates that must be seen at once are their own messages.
/// Outside private conversations only those are shown. When `settle` fires
/// (the turn's run is over), whatever is queued is noted and shown, then
/// this ends: the card is settled before the reply, without waiting on
/// work (a background helper) that still holds a sender.
async fn show_on_card(
    surface: Arc<dyn TurnSurface>,
    card: TurnCard,
    mut updates: mpsc::Receiver<StatusUpdate>,
    private: bool,
    settle: CancellationToken,
) {
    let mut last_sent: Option<tokio::time::Instant> = None;
    let mut tick = tokio::time::interval(Duration::from_secs(5));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        let mut last = false;
        let backlog = tokio::select! {
            biased;
            _ = settle.cancelled() => {
                last = true;
                let mut backlog = Vec::new();
                while let Ok(next) = updates.try_recv() {
                    backlog.push(next);
                }
                backlog
            }
            first = updates.recv() => {
                let Some(first) = first else { break };
                // Whatever piled up while the last edit was on its way is
                // applied in order and shown in one edit.
                let mut backlog = vec![first];
                while let Ok(next) = updates.try_recv() {
                    backlog.push(next);
                }
                backlog
            }
            // Nothing new: show what an interval held back, or a long
            // step's new elapsed time.
            _ = tick.tick() => Vec::new(),
        };
        let mut show_now = last;
        for update in backlog {
            if update.requires_immediate_delivery() {
                if let Some(line) = status_line(&update) {
                    let _ = surface.say(&line).await;
                }
                continue;
            }
            if private {
                show_now |= card.live.lock().await.note(&update);
            }
        }
        let due = last_sent.is_none_or(|at| at.elapsed() >= STATUS_INTERVAL);
        if private && (show_now || due) {
            let sent = card
                .live
                .lock()
                .await
                .flush_if_changed(card.sink.as_ref())
                .await;
            if sent {
                last_sent = Some(tokio::time::Instant::now());
            }
        }
        if last {
            break;
        }
    }
}

/// The status line for one update, as Markdown, when it has one: progress
/// in italics, a plan card or a needed action as they are.
pub fn status_line(update: &StatusUpdate) -> Option<String> {
    let text = status_text(update)?;
    Some(match update {
        StatusUpdate::UserActionRequired { .. } | StatusUpdate::Checklist(_) => text,
        _ => format!("_{text}_"),
    })
}

/// The plain status text for one update, when it has one. A step reads as
/// its typed kind in plain words; a tool ending is not a step.
pub fn status_text(update: &StatusUpdate) -> Option<String> {
    use aidaemon_core::card::Step;
    let progress = |text: String| text;
    Some(match update {
        // The working indicator says it; no line of its own.
        StatusUpdate::Thinking(_) | StatusUpdate::ReplyText(_) => return None,
        // A generic kind of work ("Checking things") says nothing.
        StatusUpdate::ToolStart { kind, .. } if !kind.says_what() => return None,
        StatusUpdate::ToolStart { kind, .. } => progress(format!("{}…", kind.words())),
        StatusUpdate::ToolProgress { chunk, .. } => {
            let preview: String = chunk.chars().take(100).collect();
            let more = if chunk.chars().count() > 100 {
                "…"
            } else {
                ""
            };
            progress(format!("📤 {preview}{more}"))
        }
        StatusUpdate::UserActionRequired { message } => message.clone(),
        StatusUpdate::ToolComplete { .. } => return None,
        // Bookkeeping for `/stop`: no line of its own.
        StatusUpdate::ToolCancellable { .. } => return None,
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
            let base = format!(
                "✅ Step {}/{total_steps} done: {description}",
                step_index + 1
            );
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
        StatusUpdate::Checklist(plan) => plan.text.clone(),
        StatusUpdate::Helper(helper) => progress(crate::live_status::render_card(
            &crate::live_status::CardView {
                helpers: vec![(helper.clone(), None)],
                ..Default::default()
            },
        )?),
        StatusUpdate::StepHandedOff { kind } => progress(Step::HandedOff(*kind).words()),
        StatusUpdate::CommandOutput { line, .. } => progress(Step::Output(line.clone()).words()),
        StatusUpdate::Finished { by_me, total } => progress(crate::live_status::finished_block(
            by_me.as_deref(),
            total.as_deref(),
        )?),
    })
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
                        kind: aidaemon_core::card::ActivityKind::Run,
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
            watch: TurnWatch::default(),
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
        assert!(reply.contains("Latest step: Running a command"), "{reply}");
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
                notice.contains("Latest step: Running a command"),
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

    /// A turn that reports `updates`, starts work that outlives it (it
    /// waits on [`TurnOver`], then notes its end in the chat's log) and
    /// replies `reply`.
    struct CardAgent {
        updates: Vec<StatusUpdate>,
        reply: &'static str,
        disposition: aidaemon_events::AssistantResponseDisposition,
        log: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl crate::ports::AgentIngress for CardAgent {
        async fn handle_inbound_message(
            &self,
            request: InboundMessageRequest,
        ) -> anyhow::Result<AgentResponseEnvelope> {
            if let Some(turn) = TurnOver::current() {
                let log = self.log.clone();
                tokio::spawn(async move {
                    turn.wait().await;
                    log.lock().unwrap().push("background work ended".into());
                });
            }
            if let Some(status) = request.status_tx.as_ref() {
                for update in &self.updates {
                    let _ = status.send(update.clone()).await;
                }
            }
            Ok(AgentResponseEnvelope {
                response_id: "r".into(),
                task_id: "t".into(),
                turn_id: None,
                text: self.reply.into(),
                referenced_receipts: Vec::new(),
                disposition: self.disposition,
            })
        }
    }

    #[async_trait]
    impl ChannelAgentRuntime for CardAgent {
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

    /// A chat that keeps a card (as Telegram does): every message, card
    /// change and hand-off lands in one log, in order.
    struct CardChat {
        log: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl crate::live_status::SurfaceSink for CardChat {
        async fn create(&self, text: &str) -> anyhow::Result<Option<String>> {
            self.log.lock().unwrap().push(format!("card:{text}"));
            Ok(Some("card-1".into()))
        }
        async fn edit(&self, _: &str, text: &str) -> anyhow::Result<bool> {
            self.log.lock().unwrap().push(format!("card:{text}"));
            Ok(true)
        }
        async fn delete(&self, _: &str) -> anyhow::Result<bool> {
            self.log.lock().unwrap().push("card deleted".into());
            Ok(true)
        }
    }

    struct CardSurface(Arc<CardChat>);

    #[async_trait]
    impl TurnSurface for CardSurface {
        fn platform(&self) -> &'static str {
            "test"
        }
        async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
            self.0.log.lock().unwrap().push(format!("say:{text}"));
            Ok(vec!["reply-1".into()])
        }
        fn show_working(&self, _: CancellationToken, _: Arc<AtomicU64>, _: u64) {}
        fn card_sink(&self) -> Option<Arc<dyn crate::live_status::SurfaceSink>> {
            Some(self.0.clone())
        }
        async fn card_kept(&self, id: &str) {
            self.0.log.lock().unwrap().push(format!("card kept {id}"));
        }
    }

    /// Run one turn of `agent` on a chat that keeps a card; the chat's log.
    async fn run_card_turn(
        updates: Vec<StatusUpdate>,
        reply: &'static str,
        disposition: aidaemon_events::AssistantResponseDisposition,
    ) -> Vec<String> {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let agent = Arc::new(CardAgent {
            updates,
            reply,
            disposition,
            log: log.clone(),
        });
        let runner = TurnRunner::new(TurnDeps {
            agent,
            session_map: Arc::default(),
            sessions: Arc::new(NoSessions),
            task_registry: Arc::new(TaskRegistry::new(10)),
            watchdog_stale_threshold_secs: 0,
            watch: TurnWatch::default(),
        });
        let surface = Arc::new(CardSurface(Arc::new(CardChat { log: log.clone() })));
        runner
            .run(surface, owner_turn("Count both files", "m1"))
            .await;
        for _ in 0..200 {
            if log
                .lock()
                .unwrap()
                .iter()
                .any(|e| e == "background work ended")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let log = log.lock().unwrap().clone();
        log
    }

    fn helper(label: &str, phase: aidaemon_core::card::RunPhase) -> StatusUpdate {
        StatusUpdate::Helper(aidaemon_core::card::HelperView {
            phase,
            ..aidaemon_core::card::HelperView::running(label, false, None)
        })
    }

    /// The runner's path: the same card state ends the same way as the
    /// card's own rule says. Two helpers ran side by side, so the card is
    /// kept and settled after the reply (live 2026-11-13: deleted at turn
    /// end with no final header); a single step collapses.
    #[tokio::test]
    async fn the_runner_ends_each_card_by_its_typed_state() {
        use aidaemon_core::card::{ActivityKind, RunPhase};
        let log = run_card_turn(
            vec![
                StatusUpdate::ToolStart {
                    name: "spawn".into(),
                    summary: String::new(),
                    kind: ActivityKind::Delegate,
                },
                helper("Count lines", RunPhase::Running),
                helper("Count words", RunPhase::Running),
                helper("Count lines", RunPhase::Succeeded),
                helper("Count words", RunPhase::Succeeded),
            ],
            "Lines: 12. Words: 80.",
            Default::default(),
        )
        .await;
        // The card already reads settled once both helpers ended (the turn
        // waiting on them is no step of its own), and it is kept after the
        // reply, never deleted.
        let reply = log.iter().position(|e| e == "say:Lines: 12. Words: 80.");
        let settled = log.iter().rposition(|e| {
            e == "card:✅ A helper finished: Count lines\n\n✅ A helper finished: Count words"
        });
        let kept = log.iter().position(|e| e == "card kept card-1");
        assert!(
            settled.is_some() && reply.is_some() && kept > reply,
            "{log:#?}"
        );
        assert!(!log.contains(&"card deleted".to_string()), "{log:#?}");

        let log = run_card_turn(
            vec![StatusUpdate::ToolStart {
                name: "read".into(),
                summary: String::new(),
                kind: ActivityKind::Read,
            }],
            "It says hello.",
            Default::default(),
        )
        .await;
        let reply = log.iter().position(|e| e == "say:It says hello.");
        let deleted = log.iter().position(|e| e == "card deleted");
        assert!(reply.is_some() && deleted > reply, "{log:#?}");
        assert!(!log.iter().any(|e| e.starts_with("card kept")), "{log:#?}");
    }

    /// A background handoff with a run on its card keeps the card: the ack
    /// goes out as its own message, before anything the background work does
    /// when it ends. One with nothing kept is removed after the ack: the
    /// card never becomes a message of its own.
    #[tokio::test]
    async fn a_handoffs_ack_comes_before_the_work_ends() {
        use aidaemon_core::card::HelperView;
        use aidaemon_events::AssistantResponseDisposition::BackgroundHandoff;
        let log = run_card_turn(
            vec![StatusUpdate::Helper(HelperView::running(
                "Slow count",
                true,
                None,
            ))],
            "Started in the background: Slow count.",
            BackgroundHandoff,
        )
        .await;
        let ack = log
            .iter()
            .position(|e| e == "say:Started in the background: Slow count.");
        let kept = log.iter().position(|e| e == "card kept card-1");
        let ended = log.iter().position(|e| e == "background work ended");
        assert!(ack.is_some() && kept > ack && ended > kept, "{log:#?}");
        assert!(
            !log.iter().any(|e| e.starts_with("card became")),
            "{log:#?}"
        );

        let log = run_card_turn(
            vec![StatusUpdate::StepHandedOff {
                kind: aidaemon_core::card::ActivityKind::Run,
            }],
            "Started the install in the background.",
            BackgroundHandoff,
        )
        .await;
        let ack = log
            .iter()
            .position(|e| e == "say:Started the install in the background.");
        let deleted = log.iter().position(|e| e == "card deleted");
        let ended = log.iter().position(|e| e == "background work ended");
        assert!(
            ack.is_some() && deleted > ack && ended > deleted,
            "{log:#?}"
        );
        assert!(
            !log.contains(&"card:Started the install in the background.".to_string()),
            "{log:#?}"
        );
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
            watch: TurnWatch::default(),
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

    /// Only helpers and background commands run, no turn (live 2026-11-13:
    /// "how's it going?" became a turn that said "Still working on
    /// \"how's it going?\" (1s so far)"). A question about them is answered
    /// from their typed state, each helper with the step it last reported
    /// (a step from after the turn that started it ended) and the time
    /// `/tasks` shows; no turn starts. Anything else still runs as a turn.
    #[tokio::test]
    async fn a_status_question_between_runs_gets_the_typed_status() {
        let agent = Arc::new(ScriptedAgent {
            release: tokio::sync::Notify::new(),
            reading: std::sync::Mutex::new(Some(MidRunReading {
                intent: MidRunIntent::AboutRunningWork,
                stop: Vec::new(),
            })),
            work: std::sync::Mutex::new(two_counting_jobs()),
        });
        let registry = Arc::new(TaskRegistry::new(10));
        let runner = TurnRunner::new(TurnDeps {
            agent: agent.clone(),
            session_map: Arc::default(),
            sessions: Arc::new(NoSessions),
            task_registry: registry.clone(),
            watchdog_stale_threshold_secs: 0,
            watch: TurnWatch::default(),
        });
        let session = "test:mid-run";
        // The turn that started the helper has ended; the helper reports on.
        runner.live.begin(session, 7);
        runner.live.record(
            session,
            7,
            &StatusUpdate::Helper(aidaemon_core::card::HelperView::running(
                "Counting agent A",
                true,
                None,
            )),
        );
        runner.live.end(session, 7);
        runner.live.record(
            session,
            7,
            &StatusUpdate::Helper(aidaemon_core::card::HelperView::running(
                "Counting agent A",
                true,
                Some(aidaemon_core::card::Step::Doing(
                    aidaemon_core::card::ActivityKind::Run,
                )),
            )),
        );

        let surface = Arc::new(Said::default());
        runner
            .submit(surface.clone(), owner_turn("how's it going?", "b1"))
            .await;
        assert!(!registry.has_running_task(session).await);
        let reply = surface.0.lock().unwrap().last().cloned().unwrap();
        assert!(reply.starts_with("Working in the background:"), "{reply}");
        assert!(
            reply.contains("\n• Counting agent A: Running a command (1m so far)"),
            "{reply}"
        );
        assert!(
            reply.contains("\n• Background command: count to 300 (50s so far)"),
            "{reply}"
        );
        assert!(!reply.contains("how's it going"), "{reply}");

        // A new request is not a status question: it runs as a turn.
        *agent.reading.lock().unwrap() = Some(MidRunReading {
            intent: MidRunIntent::Unrelated,
            stop: Vec::new(),
        });
        runner
            .submit(
                surface.clone(),
                owner_turn("What is the capital of Peru?", "b2"),
            )
            .await;
        for _ in 0..200 {
            if registry.has_running_task(session).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(registry.has_running_task(session).await);
        agent.release.notify_one();
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

    /// `/tasks` gives each helper's latest step beside its title, as the
    /// status reply does (live 2026-11-13: "#2 Sub-agent: WordPress Install
    /// (29s so far)" with nothing about what it was doing).
    #[tokio::test]
    async fn tasks_lists_a_helpers_latest_step() {
        let mut work = two_counting_jobs();
        work[0].step = Some(aidaemon_core::card::Step::Output("14".into()));
        let agent = ScriptedAgent {
            release: tokio::sync::Notify::new(),
            reading: std::sync::Mutex::new(None),
            work: std::sync::Mutex::new(work),
        };
        let registry = TaskRegistry::new(10);
        let running = crate::stop::running_work(&registry, &agent, "s1").await;
        let tasks = crate::slash::render_tasks(&running, &[], chrono::Utc::now(), None);
        assert!(
            tasks.contains("Sub-agent: Counting agent A – Running a command · last output: 14 ("),
            "{tasks}"
        );
        assert!(
            tasks.contains("Background command: count to 300 ("),
            "{tasks}"
        );
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
                step: None,
            },
            crate::ports::ChatWork {
                key: "job:0:4242".into(),
                kind: crate::ports::ChatWorkKind::BackgroundJob,
                title: "Background command: count to 300".into(),
                started_at: now - chrono::Duration::seconds(50),
                request: Some("req-1".into()),
                step: None,
            },
        ]
    }
}
