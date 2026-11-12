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

use super::ChannelRuntimeDeps;
use crate::events::ResponseDeliveryState;
use crate::runtime_ports::{AgentResponseEnvelope, InboundMessageRequest, InboundMessageTiming};
use crate::tasks::{QueueOutcome, QueuedMessage};
use crate::traits::MessageAttachment;
use crate::types::{ChannelContext, ChannelVisibility, StatusUpdate, UserRole};

/// Least time between two ordinary status messages.
const STATUS_INTERVAL: Duration = Duration::from_secs(3);
/// Most time a finished turn waits for its status display to catch up
/// before the reply is sent.
const STATUS_DRAIN: Duration = Duration::from_secs(3);
const CANCELLED: &str = "Task cancelled";
const INACTIVITY_PREFIX: &str = "Task auto-cancelled due to inactivity";

/// One authorized message, ready to run.
pub(crate) struct Turn {
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
pub(crate) struct Delivered {
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
pub(crate) enum TurnEnd {
    Delivered,
    /// The agent had nothing to say.
    Empty,
    Failed,
    Cancelled,
}

/// Where one conversation's turns are shown.
#[async_trait]
pub(crate) trait TurnSurface: Send + Sync + 'static {
    /// The platform recorded on delivery receipts (`telegram`, `slack`...).
    fn platform(&self) -> &'static str;

    /// Send Markdown `text`, split to fit. Returns the ids of the messages
    /// sent, or an error if any part failed.
    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>>;

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

/// Runs turns for one channel (bot).
#[derive(Clone)]
pub(crate) struct TurnRunner {
    deps: ChannelRuntimeDeps,
}

impl TurnRunner {
    pub(crate) fn new(deps: ChannelRuntimeDeps) -> Self {
        Self { deps }
    }

    /// Remember that `session_id` belongs to `channel_name`, so outbound
    /// messages (approvals, media, notifications) route back to it.
    pub(crate) async fn register_session(&self, session_id: &str, channel_name: &str) {
        self.deps
            .session_map
            .write()
            .await
            .insert(session_id.to_string(), channel_name.to_string());
        let _ = self
            .deps
            .state
            .save_session_channel(session_id, channel_name)
            .await;
    }

    /// Handle an authorized message: a cancel word, a message queued behind
    /// a running turn, or a new turn (and then every turn queued behind it).
    pub(crate) async fn submit(&self, surface: Arc<dyn TurnSurface>, turn: Turn) {
        if is_cancel_word(&turn.text) {
            let reply = self.cancel(&turn.session_id, turn.user_role).await;
            let _ = surface.say(&reply).await;
            return;
        }

        let registry = &self.deps.task_registry;
        let session_id = turn.session_id.clone();
        let dedup_key = turn.dedup_key.as_deref();
        if registry.has_running_task(&session_id).await {
            let mut queued = QueuedMessage::new(
                &turn.text,
                turn.user_role,
                turn.channel_ctx.clone(),
                turn.attachments.clone(),
            );
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
                        let current = registry
                            .get_running_task_description(&session_id)
                            .await
                            .unwrap_or_else(|| "processing".to_string());
                        let notice = super::presentation::queued_message_notice(
                            &turn.text,
                            &current,
                            !turn.attachments.is_empty(),
                        );
                        let _ = surface.say(&notice).await;
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

    /// The reply to a cancel word.
    async fn cancel(&self, session_id: &str, role: UserRole) -> String {
        if role != UserRole::Owner {
            return "Only the owner can cancel running work in this session.".to_string();
        }
        let registry = &self.deps.task_registry;
        let cancelled = registry.cancel_running_for_session(session_id).await;
        let queued_work_preserved = registry.queue_len(session_id).await;
        let cancelled_goals = self
            .deps
            .agent
            .cancel_active_finite_work_for_session(session_id)
            .await;
        cancel_reply(&cancelled, queued_work_preserved, &cancelled_goals)
    }

    /// Run `turn` and then each turn queued behind it.
    async fn run(self, surface: Arc<dyn TurnSurface>, first: Turn) {
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
            let owner_wait = crate::tools::OwnerWaitClock::default();
            // Shown in "current task (…)" notices and cancel replies.
            let description = super::presentation::word_preview(&text, 80);
            let (task_id, cancel) = registry.register(&session_id, &description).await;
            let heartbeat = Arc::new(AtomicU64::new(now_secs()));
            registry.set_heartbeat(task_id, heartbeat.clone()).await;
            let working = CancellationToken::new();
            if let Ok(mut slot) = indicator.lock() {
                *slot = working.clone();
            }
            surface.show_working(working.clone(), heartbeat.clone(), stale_secs);
            registry.set_typing_cancel(task_id, working.clone()).await;
            let (status_tx, status_rx) = mpsc::channel::<StatusUpdate>(16);
            let private = channel_ctx.visibility == ChannelVisibility::Private;
            let status_task = surface
                .show_status(status_rx, private)
                .unwrap_or_else(|updates| throttled_status(surface.clone(), updates, private));
            if let Some(timing) = timing.as_mut() {
                timing.mark_dispatched_now();
            }

            let result = tokio::select! {
                result = crate::tools::OwnerWaitClock::within(
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
                stale_mins = super::wait_for_stale_heartbeat(heartbeat.clone(), stale_secs, 4), if stale_secs > 0 => {
                    Err(anyhow::anyhow!(
                        "{INACTIVITY_PREFIX} ({} minute{} without progress).",
                        stale_mins,
                        if stale_mins == 1 { "" } else { "s" }
                    ))
                },
            };
            working.cancel();
            // The run is over, so its status sender is gone and the surface
            // finishes showing what it sent: the closing checklist must reach
            // the card before the reply, or the card stays on "Working" above
            // a reply. A background job may still hold a sender, so the wait
            // is bounded and a handoff, whose job keeps running, skips it.
            let mut status_task = status_task;
            let run_finished = matches!(
                &result,
                Ok(envelope) if envelope.disposition
                    != crate::events::AssistantResponseDisposition::BackgroundHandoff
            );
            if run_finished {
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
                    let reply = super::prepare_chat_message(&envelope.text);
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
                    crate::events::TaskStatus::Cancelled
                } else {
                    crate::events::TaskStatus::Failed
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
                break;
            };
            // Let the finished turn's messages commit first.
            tokio::time::sleep(Duration::from_millis(100)).await;
            let queued = surface.before_queued(queued).await;
            let preview = super::presentation::word_preview(&queued.text, 50);
            info!(session_id, "Processing queued message: {}", preview);
            let _ = surface
                .say(&format!("▶️ Processing queued: \"{preview}\""))
                .await;
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
pub(crate) fn is_cancel_word(text: &str) -> bool {
    matches!(
        text.trim().to_lowercase().as_str(),
        "cancel" | "stop" | "abort"
    )
}

fn cancel_reply(
    cancelled: &[(u64, String)],
    queued_work_preserved: usize,
    cancelled_goals: &[String],
) -> String {
    if let Some((_, description)) = cancelled.first() {
        let mut reply = format!("⏹️ Cancelled: {description}");
        if queued_work_preserved > 0 {
            reply.push_str(&format!(
                " ({queued_work_preserved} queued request(s) will continue in order)"
            ));
        }
        if !cancelled_goals.is_empty() {
            reply.push_str(&format!(" (+{} more cancelled)", cancelled_goals.len()));
        }
        return reply;
    }
    match cancelled_goals {
        [] => "No running task to cancel.".to_string(),
        [goal] => format!("⏹️ Cancelled: {goal}"),
        goals => format!(
            "⏹️ Cancelled {} items:\n{}",
            goals.len(),
            goals
                .iter()
                .map(|goal| format!("- {goal}"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
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
pub(crate) fn status_line(update: &StatusUpdate) -> String {
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
    }
}

/// Repeat `send` every `every` until `stop` is cancelled, the heartbeat has
/// been quiet for `stale_secs` (0 = never), or 30 minutes pass.
pub(crate) fn repeat_while_working<F, Fut>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{setup_test_agent, MockProvider};

    /// Records what the runner says.
    #[derive(Default)]
    struct Recording(std::sync::Mutex<Vec<String>>);

    #[async_trait]
    impl TurnSurface for Recording {
        fn platform(&self) -> &'static str {
            "test"
        }
        async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
            self.0.lock().unwrap().push(text.to_string());
            Ok(Vec::new())
        }
        fn show_working(&self, _: CancellationToken, _: Arc<AtomicU64>, _: u64) {}
    }

    fn turn(text: &str, key: &str) -> Turn {
        Turn {
            session_id: "test:wall-clock".to_string(),
            text: text.to_string(),
            attachments: Vec::new(),
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("test"),
            dedup_key: Some(key.to_string()),
            ingress_timing: None,
        }
    }

    #[tokio::test]
    async fn stop_ends_finite_work_but_keeps_a_recurring_goal() {
        use crate::traits::store_prelude::*;
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let session = "test:stop";
        let recurring =
            crate::traits::Goal::new_continuous("Publish a daily blog post", session, None, None);
        let finite = crate::traits::Goal::new_finite("Answer the current question", session);
        harness.state.create_goal(&recurring).await.unwrap();
        harness.state.create_goal(&finite).await.unwrap();
        let inbox = tempfile::tempdir().unwrap().keep();
        let state = harness.state.clone();
        let runner = TurnRunner::new(ChannelRuntimeDeps {
            agent: Arc::new(harness.agent),
            config_path: inbox.join("config.toml"),
            session_map: Arc::default(),
            task_registry: Arc::new(crate::tasks::TaskRegistry::new(50)),
            files_enabled: false,
            inbox_dir: inbox,
            max_file_size_mb: 10,
            state: state.clone(),
            watchdog_stale_threshold_secs: 0,
        });
        let surface = Arc::new(Recording::default());
        let mut stop = turn("stop", "m1");
        stop.session_id = session.to_string();
        runner.submit(surface.clone(), stop).await;

        let said = surface.0.lock().unwrap().clone();
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].contains("Answer the current question"), "{said:?}");
        assert!(!said[0].contains("daily blog"), "{said:?}");
        let status = |goal: &crate::traits::Goal| {
            let state = state.clone();
            let id = goal.id.clone();
            async move { state.get_goal(&id).await.unwrap().unwrap().status }
        };
        assert_eq!(status(&finite).await, "cancelled");
        assert_ne!(status(&recurring).await, "cancelled");
    }

    /// A surface whose status display is slow, as a loaded Telegram edit is:
    /// each update shows a while after it arrives.
    #[derive(Default)]
    struct SlowStatus {
        log: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl TurnSurface for SlowStatus {
        fn platform(&self) -> &'static str {
            "test"
        }
        async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
            self.log.lock().unwrap().push(format!("say:{text}"));
            Ok(vec!["1".to_string()])
        }
        fn show_working(&self, _: CancellationToken, _: Arc<AtomicU64>, _: u64) {}
        fn show_status(
            &self,
            mut updates: mpsc::Receiver<StatusUpdate>,
            _private: bool,
        ) -> Result<JoinHandle<()>, mpsc::Receiver<StatusUpdate>> {
            let log = self.log.clone();
            Ok(tokio::spawn(async move {
                while let Some(update) = updates.recv().await {
                    log.lock().unwrap().push("status-received".to_string());
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    log.lock()
                        .unwrap()
                        .push(format!("status-shown:{}", status_line(&update)));
                }
            }))
        }
    }

    #[tokio::test]
    async fn status_sent_before_the_reply_is_shown_before_the_reply() {
        let tool = Arc::new(crate::testing::MockTool::new(
            "lookup_widget",
            "Look up a synthetic widget",
            "widget-7 is blue",
        ));
        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response("lookup_widget", "{}"),
            MockProvider::text_response("Widget-7 is blue."),
        ]);
        let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
            provider,
            vec![tool as Arc<dyn crate::traits::Tool>],
            None,
        )
        .await
        .unwrap();
        let inbox = tempfile::tempdir().unwrap().keep();
        let runner = TurnRunner::new(ChannelRuntimeDeps {
            agent: Arc::new(harness.agent),
            config_path: inbox.join("config.toml"),
            session_map: Arc::default(),
            task_registry: Arc::new(crate::tasks::TaskRegistry::new(50)),
            files_enabled: false,
            inbox_dir: inbox,
            max_file_size_mb: 10,
            state: harness.state.clone(),
            watchdog_stale_threshold_secs: 0,
        });
        let surface = Arc::new(SlowStatus::default());
        runner
            .clone()
            .run(surface.clone(), turn("What colour is widget-7?", "m1"))
            .await;

        let log = surface.log.lock().unwrap().clone();
        let reply = log
            .iter()
            .position(|entry| entry.starts_with("say:") && entry.contains("blue"))
            .unwrap_or_else(|| panic!("no reply in {log:#?}"));
        let received = log[..reply]
            .iter()
            .filter(|entry| *entry == "status-received")
            .count();
        let shown = log[..reply]
            .iter()
            .filter(|entry| entry.starts_with("status-shown:"))
            .count();
        assert!(received > 0, "the turn sent no status: {log:#?}");
        assert_eq!(
            shown, received,
            "every status sent during the turn must show before the reply: {log:#?}"
        );
    }

    #[test]
    fn cancel_words_are_whole_messages() {
        assert!(is_cancel_word("Stop"));
        assert!(is_cancel_word(" cancel "));
        assert!(!is_cancel_word("stop the server"));
        assert!(!is_cancel_word("don't stop"));
    }

    #[test]
    fn cancel_reply_names_what_stopped() {
        assert_eq!(cancel_reply(&[], 0, &[]), "No running task to cancel.");
        assert_eq!(
            cancel_reply(&[(1, "deploy".into())], 2, &["blog".into()]),
            "⏹️ Cancelled: deploy (2 queued request(s) will continue in order) (+1 more cancelled)"
        );
        assert_eq!(
            cancel_reply(&[], 0, &["a".into(), "b".into()]),
            "⏹️ Cancelled 2 items:\n- a\n- b"
        );
    }
}
