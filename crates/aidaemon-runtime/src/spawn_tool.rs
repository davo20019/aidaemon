//! `spawn_agent`: the tool a model delegates with, over the
//! [`crate::spawn::ChildAgentRuntime`] port.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt as _;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

use crate::run_state::RunPhase;
use crate::spawn::{
    ChildAgentRequest, ChildAgentRun, ChildAgentRuntime, ChildTimeLimit, ChildTimedOut,
};
use crate::turn::sliding_window::MEASURED_TIMING_MARK;
use aidaemon_channels::context::ChannelContext;
use aidaemon_channels::ports::OutboundRouter;
use aidaemon_core::ChannelVisibility;
use aidaemon_core::SourceTrust;
use aidaemon_events::traits::AgentRole;
use aidaemon_events::TaskOutcome;
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::semantics::{ToolCallContext, ToolCallOutcome, ToolCapabilities};
use aidaemon_tools::status::StatusUpdate;
use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_work::stores::NotificationStore as StateStore;

/// A tool that allows the LLM to spawn a sub-agent for a focused task.
///
/// The sub-agent runs its own agentic loop with a dedicated session and a
/// system prompt that includes the given mission. Its final text response is
/// returned as the tool result.
///
/// The circular dependency (runtime → tools → SpawnAgentTool → runtime) is
/// broken by storing a weak [`ChildAgentRuntime`] port inside a `OnceLock`.
/// The reference is installed after the concrete root runtime is constructed.
pub struct SpawnAgentTool {
    pub agent: OnceLock<Weak<dyn ChildAgentRuntime>>,
    pub hub: OnceLock<Weak<dyn OutboundRouter>>,
    pub state: Option<Arc<dyn StateStore>>,
    pub max_response_chars: usize,
    pub timeout_secs: u64,
    pub executor_task_runs: Arc<Mutex<HashSet<String>>>,
    /// The session checklists, so a background child can carry the steps it
    /// was given and settle them when it ends.
    pub plan_store: OnceLock<Arc<aidaemon_work::plans::PlanStore>>,
}

/// Fallback description used when the parent runtime isn't wired yet
/// (early bootstrap) or when the registry is empty.
pub const STATIC_SPECIALIST_ARG_DESCRIPTION: &str =
    "Optional. Pick the specialist profile matching this task. \
     Omit to use the broad executor profile.";

/// Format the per-kind `(name, description)` pairs from
/// `SpecialistRegistry::llm_visible_kinds()` into the multi-line description
/// surfaced via the `specialist` parameter of `spawn_agent`. Kept as a free
/// function so tests can exercise the formatter without wiring a full Agent.
pub fn format_specialist_arg_description<S: AsRef<str>>(entries: &[(S, String)]) -> String {
    if entries.is_empty() {
        return STATIC_SPECIALIST_ARG_DESCRIPTION.to_string();
    }
    let mut s =
        String::from("Optional. Pick the specialist profile that best matches this task:\n");
    for (name, description) in entries {
        s.push_str("- ");
        s.push_str(name.as_ref());
        s.push_str(": ");
        s.push_str(description);
        if !description.ends_with('.') {
            s.push('.');
        }
        s.push('\n');
    }
    s.push_str("Omit `specialist` to use the broad executor profile.");
    s
}

impl SpawnAgentTool {
    /// Create a SpawnAgentTool with a known child-runtime reference.
    #[allow(dead_code)]
    pub fn new(
        agent: Weak<dyn ChildAgentRuntime>,
        max_response_chars: usize,
        timeout_secs: u64,
    ) -> Self {
        let lock = OnceLock::new();
        let _ = lock.set(agent);
        Self {
            agent: lock,
            hub: OnceLock::new(),
            state: None,
            max_response_chars,
            timeout_secs,
            executor_task_runs: Arc::new(Mutex::new(HashSet::new())),
            plan_store: OnceLock::new(),
        }
    }

    /// Create a SpawnAgentTool with a deferred agent reference.
    /// Call [`set_agent`] after constructing the root runtime.
    pub fn new_deferred(max_response_chars: usize, timeout_secs: u64) -> Self {
        Self {
            agent: OnceLock::new(),
            hub: OnceLock::new(),
            state: None,
            max_response_chars,
            timeout_secs,
            executor_task_runs: Arc::new(Mutex::new(HashSet::new())),
            plan_store: OnceLock::new(),
        }
    }

    /// The tool a child agent uses to delegate further: fully wired, because
    /// a child's own results reach the chat through the same hub.
    pub fn for_child(
        max_response_chars: usize,
        timeout_secs: u64,
        state: Arc<dyn StateStore>,
        hub: Option<Weak<dyn OutboundRouter>>,
    ) -> Self {
        let tool = Self::new_deferred(max_response_chars, timeout_secs).with_state(state);
        if let Some(hub) = hub {
            tool.set_hub(hub);
        }
        tool
    }

    /// Set state store reference for queued background notifications when direct
    /// channel delivery is unavailable.
    pub fn with_state(mut self, state: Arc<dyn StateStore>) -> Self {
        self.state = Some(state);
        self
    }

    /// Set the child-runtime reference exactly once after root construction.
    /// Panics if called more than once.
    pub fn set_agent(&self, agent: Weak<dyn ChildAgentRuntime>) {
        self.agent
            .set(agent)
            .expect("SpawnAgentTool::set_agent called more than once");
    }

    pub fn get_agent(&self) -> anyhow::Result<Arc<dyn ChildAgentRuntime>> {
        let weak = self
            .agent
            .get()
            .ok_or_else(|| anyhow::anyhow!("SpawnAgentTool: agent reference not set"))?;
        weak.upgrade()
            .ok_or_else(|| anyhow::anyhow!("SpawnAgentTool: parent agent has been dropped"))
    }

    /// Set the channel hub reference for background mode notifications.
    pub fn set_hub(&self, hub: Weak<dyn OutboundRouter>) {
        self.hub
            .set(hub)
            .expect("SpawnAgentTool::set_hub called more than once");
    }

    /// Wire the session checklists (see `plan_store`).
    pub fn set_plan_store(&self, plan_store: Arc<aidaemon_work::plans::PlanStore>) {
        let _ = self.plan_store.set(plan_store);
    }

    pub fn get_hub(&self) -> Option<Arc<dyn OutboundRouter>> {
        self.hub.get().and_then(|w| w.upgrade())
    }

    pub fn wiring_ready(&self) -> bool {
        self.agent_wired() && self.hub.get().and_then(Weak::upgrade).is_some()
    }

    pub fn agent_wired(&self) -> bool {
        self.agent.get().and_then(Weak::upgrade).is_some()
    }

    /// Render the `specialist` parameter's description text, pulling each
    /// kind's frontmatter description from the live `SpecialistRegistry`
    /// (so user overrides at `~/.aidaemon/specialists/<kind>.md` flow
    /// through to the LLM-facing schema on next start).
    ///
    /// During early bootstrap the parent `Agent` may not yet be wired up
    /// (the weak ref hasn't been set). In that case we fall back to a
    /// static description with no per-kind text — the `enum` list still
    /// constrains the value, so the LLM can still pick a valid kind even
    /// without descriptions.
    pub fn build_specialist_arg_description(&self) -> String {
        let agent = match self.get_agent() {
            Ok(a) => a,
            Err(_) => {
                warn!(
                    "spawn_agent schema: parent agent not yet available; falling back to static specialist description"
                );
                return STATIC_SPECIALIST_ARG_DESCRIPTION.to_string();
            }
        };

        format_specialist_arg_description(&agent.specialist_descriptions())
    }

    /// How long a child spawned by this tool may go without progress
    /// (`subagents.timeout_secs`); never a limit on how long it works.
    pub fn time_limit(&self) -> Option<ChildTimeLimit> {
        Some(ChildTimeLimit {
            limit: Duration::from_secs(self.timeout_secs),
        })
    }

    /// Acquire a per-task in-flight lock for executor spawns.
    pub async fn try_begin_executor_task(&self, task_id: &str) -> bool {
        let mut runs = self.executor_task_runs.lock().await;
        if runs.contains(task_id) {
            return false;
        }
        runs.insert(task_id.to_string());
        true
    }

    /// Release a per-task in-flight lock for executor spawns.
    pub async fn finish_executor_task(&self, task_id: &str) {
        self.executor_task_runs.lock().await.remove(task_id);
    }
}

/// Truncate a string to at most `max_chars` bytes without splitting a
/// multi-byte UTF-8 character. Returns the original string when it fits.
pub fn truncate_utf8(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        return s;
    }
    // Find the last char boundary at or before `max_chars`.
    let boundary = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= max_chars)
        .last()
        .unwrap_or(0);
    &s[..boundary]
}

/// A child's reply cut to `max_len`, and what was left out if anything.
pub fn bounded_child_response(
    response: &str,
    max_len: usize,
) -> (&str, Option<aidaemon_tools::semantics::TruncationInfo>) {
    if response.len() <= max_len {
        return (response, None);
    }
    let shown = truncate_utf8(response, max_len);
    let info = aidaemon_tools::semantics::TruncationInfo {
        shown_chars: shown.chars().count(),
        total_chars: response.chars().count(),
        remediation_hint: Some(
            "To see the rest, delegate again and ask the sub-agent to write the full \
             result to a file and return its path, or to answer a narrower question."
                .to_string(),
        ),
    };
    (shown, Some(info))
}

/// A background helper's result for the chat, rendered from its run record
/// alone: the one evidence line names the typed action steps nothing
/// verified, and the footer is the time the runtime measured. The helper's
/// report is the body, never what decides the rest. How it ended is said
/// once: when its card shows it (`on_card`), the result has no header of
/// its own; else the header is its phase.
pub fn render_run_result(
    state: &crate::run_state::RunState,
    name: &str,
    report: &str,
    max_response_chars: usize,
    on_card: bool,
) -> String {
    // This goes straight to the chat, past the loop that renders truncation
    // metadata for tool results.
    let body = match bounded_child_response(report, max_response_chars) {
        (shown, Some(info)) => {
            format!(
                "{shown}\n\n{}",
                crate::turn::helpers::render_truncation_notice(&info)
            )
        }
        (shown, None) => shown.to_string(),
    };
    let mut parts: Vec<String> = Vec::new();
    let mut lead = if on_card {
        String::new()
    } else {
        format!("{}: {name}", state.phase.headline())
    };
    if let Some(line) = state.evidence.line() {
        if !lead.is_empty() {
            lead.push('\n');
        }
        lead.push_str(&line);
    }
    if !lead.is_empty() {
        parts.push(lead);
    }
    if !body.trim().is_empty() {
        parts.push(body.trim_end().to_string());
    }
    if let Some(footer) = background_timing(state) {
        parts.push(footer);
    }
    parts.join("\n\n")
}

/// The notification kind a run's phase is queued under.
pub fn run_notification_type(phase: RunPhase) -> &'static str {
    match phase {
        RunPhase::Succeeded => "completed",
        RunPhase::Stopped => "partial",
        RunPhase::Failed | RunPhase::Running => "failed",
    }
}

/// The plan steps a run's phase settles.
pub fn job_end(phase: RunPhase) -> aidaemon_work::plans::JobEnd {
    match phase {
        RunPhase::Succeeded => aidaemon_work::plans::JobEnd::Succeeded,
        RunPhase::Stopped => aidaemon_work::plans::JobEnd::Stopped,
        RunPhase::Failed | RunPhase::Running => aidaemon_work::plans::JobEnd::Failed,
    }
}

/// A kept card, drawn from the records: the plan (as it renders itself) and
/// each run drawn on that card, running or ended, by the same renderer as
/// the live card.
pub fn kept_card(
    plan: Option<&aidaemon_work::plans::TaskPlan>,
    runs: &[crate::run_state::RunState],
) -> Option<String> {
    let plan = plan.map(aidaemon_work::plans::TaskPlan::card);
    aidaemon_channels::live_status::render_card(&aidaemon_channels::live_status::CardView {
        plan: plan.as_ref(),
        helpers: runs.iter().map(|run| (run.view(), None)).collect(),
        ..Default::default()
    })
}

/// The runs drawn on card `card` in `chat`: each run whose own card it is
/// (a run still waiting to learn its card is resolved first).
pub async fn runs_on_card(
    hub: &dyn OutboundRouter,
    runs: &crate::run_state::RunStates,
    chat: &str,
    card: &str,
) -> Vec<crate::run_state::RunState> {
    for run in runs.for_chat(chat) {
        if matches!(run.card, crate::run_state::RunCard::Pending { .. }) {
            run_card(hub, runs, &run.id, chat).await;
        }
    }
    runs.on_card(chat, card)
}

/// One writer at a time for a chat's kept card. A helper's step redraw and
/// the redraw its end makes take turns, and a redraw that starts after the
/// helper ended draws nothing: a step edit still in flight (a slow or
/// rate-limited Telegram edit) can never land after the finished card and
/// leave a stale step on it (live 2026-11-13: the card stayed on "running a
/// command: 13" after the count ended).
pub fn card_writer(chat: &str) -> Arc<tokio::sync::Mutex<()>> {
    static WRITERS: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    > = std::sync::LazyLock::new(Default::default);
    let mut writers = WRITERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    writers.retain(|_, writer| Arc::strong_count(writer) > 1);
    writers.entry(chat.to_string()).or_default().clone()
}

/// The card a background helper's turn leaves in the chat, for showing
/// the helper's steps after that turn ended.
pub struct HelperCard {
    pub runs: Arc<crate::run_state::RunStates>,
    pub hub: Option<Weak<dyn OutboundRouter>>,
    pub plans: Option<Arc<aidaemon_work::plans::PlanStore>>,
    pub plan_session: Option<String>,
    pub chat: String,
    pub job: String,
    pub label: String,
    /// The chat's kept card when the helper started: one that differs
    /// later is the card its own turn left.
    pub at_start: Option<String>,
    /// What this helper last drew, so an unchanged card is not sent again.
    pub drawn: std::sync::Mutex<Option<String>>,
}

/// How often a helper's steps may edit its card after its turn.
pub const HELPER_CARD_EVERY: Duration = Duration::from_secs(3);

impl HelperCard {
    /// Redraw the card the helper's turn left (its plan, when it has one,
    /// and each helper's record), when that turn has ended and left one.
    /// While the turn runs, its own live card shows the steps.
    pub async fn refresh(&self) {
        let writer = card_writer(&self.chat);
        let _turn = writer.lock().await;
        // Ended meanwhile: its end drew the finished card.
        if self
            .runs
            .get(&self.job)
            .is_none_or(|run| run.phase != RunPhase::Running)
        {
            return;
        }
        let Some(hub) = self.hub.as_ref().and_then(Weak::upgrade) else {
            return;
        };
        if self
            .plan_session
            .as_deref()
            .is_some_and(|plan_session| plan_session != self.chat)
        {
            return;
        }
        let now = hub.checklist_surface(&self.chat).await;
        if now == self.at_start {
            return;
        }
        let plan = job_plan(self.plans.as_ref(), self.plan_session.as_deref(), &self.job).await;
        // Without a plan the card is the helpers' own: the one the turn that
        // started this helper kept, never a later turn's.
        let surface = match run_card(hub.as_ref(), &self.runs, &self.job, &self.chat).await {
            Some(card) => card,
            None => match (&plan, now) {
                (Some(_), Some(card)) => card,
                _ => return,
            },
        };
        let on_card = runs_on_card(hub.as_ref(), &self.runs, &self.chat, &surface).await;
        let Some(card) = kept_card(plan.as_ref(), &on_card) else {
            return;
        };
        let unchanged = self
            .drawn
            .lock()
            .is_ok_and(|drawn| drawn.as_deref() == Some(card.as_str()));
        if unchanged {
            return;
        }
        match hub.edit_text(&self.chat, &surface, &card).await {
            Ok(true) => {
                if let Ok(mut drawn) = self.drawn.lock() {
                    *drawn = Some(card);
                }
            }
            Ok(false) => {}
            Err(error) => warn!(
                session_id = %self.chat,
                job_id = %self.job,
                %error,
                "Could not show a background helper's step on its card"
            ),
        }
    }
}

/// The recent plan of `plan_session` that background job `job` works on.
async fn job_plan(
    plans: Option<&Arc<aidaemon_work::plans::PlanStore>>,
    plan_session: Option<&str>,
    job: &str,
) -> Option<aidaemon_work::plans::TaskPlan> {
    plans?
        .get_recent_for_session(plan_session?, 5)
        .await
        .ok()?
        .into_iter()
        .find(|plan| {
            plan.steps
                .iter()
                .any(|step| step.background_jobs.iter().any(|known| known == job))
        })
}

/// A background helper's status channel: every update goes on to its
/// parent's, and its own steps are also noted for its line on the card and
/// shown there once its turn has ended (live 2026-11-13: the turn's live
/// card stopped with the turn, and a 60 s count showed nothing until it
/// ended).
pub fn show_helper_steps_on_card(
    parent: Option<mpsc::Sender<StatusUpdate>>,
    card: HelperCard,
) -> Option<mpsc::Sender<StatusUpdate>> {
    let parent = parent?;
    let card = Arc::new(card);
    let (tx, mut rx) = mpsc::channel::<StatusUpdate>(64);
    tokio::spawn(async move {
        let mut redraw = tokio::time::interval(HELPER_CARD_EVERY);
        redraw.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut changed = false;
        // The redraw runs beside the updates: a slow edit never holds up
        // the helper's steps, and the next redraw shows the latest one.
        let mut drawing: Option<tokio::task::JoinHandle<()>> = None;
        loop {
            tokio::select! {
                update = rx.recv() => {
                    let Some(update) = update else { break };
                    // The helper's own view goes on as its record: the live
                    // card shows what the kept card will.
                    let update = match update {
                        StatusUpdate::Helper(view)
                            if view.label == card.label && !view.phase.is_terminal() =>
                        {
                            if let Some(step) = view.step.clone() {
                                card.runs.step(&card.job, step);
                                changed = true;
                            }
                            match card.runs.get(&card.job) {
                                Some(run) if !run.phase.is_terminal() => {
                                    StatusUpdate::Helper(run.view())
                                }
                                _ => continue,
                            }
                        }
                        other => other,
                    };
                    let _ = parent.send(update).await;
                }
                _ = redraw.tick(), if changed => {
                    if drawing.as_ref().is_some_and(|draw| !draw.is_finished()) {
                        continue;
                    }
                    changed = false;
                    let card = card.clone();
                    drawing = Some(tokio::spawn(async move { card.refresh().await }));
                }
            }
        }
    });
    Some(tx)
}

/// How much of a mission a background record keeps.
pub const BACKGROUND_MISSION_CHARS: usize = 200;

/// Tell each chat which of its background sub-agents the last shutdown lost.
pub async fn report_interrupted_background_children(state: &Arc<dyn StateStore>) {
    let spawns = match state.take_background_spawns().await {
        Ok(spawns) => spawns,
        Err(error) => {
            warn!(%error, "Could not read interrupted background sub-agents");
            return;
        }
    };
    for spawn in spawns {
        let message = format!(
            "\u{26a0}\u{fe0f} A restart stopped this background job before it finished: {}\n\n\
             It may have done part of the work. Check before running it again.",
            spawn.mission
        );
        enqueue_background_notification(
            Some(state),
            "global",
            &spawn.session_id,
            "failed",
            &message,
            "interrupted background sub-agent notifier",
            true,
        )
        .await;
    }
}

/// Queue a background job's notification for later delivery.
#[allow(clippy::too_many_arguments)]
pub async fn enqueue_background_notification(
    state: Option<&Arc<dyn StateStore>>,
    goal_id: &str,
    session_id: &str,
    notification_type: &str,
    message: &str,
    context: &str,
    untrusted: bool,
) {
    let message = aidaemon_channels::presentation::present_notification(notification_type, message);
    let Some(state_store) = state else {
        warn!(
            session_id = %session_id,
            goal_id = %goal_id,
            notification_type = %notification_type,
            "{context}: no queue configured; update dropped"
        );
        return;
    };
    let entry = aidaemon_work::goals::NotificationEntry::new(
        goal_id,
        session_id,
        notification_type,
        &message,
    )
    .with_untrusted(untrusted);
    if let Err(e) = state_store.enqueue_notification(&entry).await {
        warn!(
            session_id = %session_id,
            goal_id = %goal_id,
            notification_type = %notification_type,
            error = %e,
            "{context}: enqueue failed"
        );
    }
}

/// Where a background child's result goes: straight to the parent session
/// through the hub, else the queue.
pub struct BackgroundDelivery<'a> {
    pub agent: &'a dyn ChildAgentRuntime,
    pub hub: Option<&'a Weak<dyn OutboundRouter>>,
    pub state: Option<&'a Arc<dyn StateStore>>,
    pub session_id: &'a str,
    pub goal_id: &'a str,
}

impl BackgroundDelivery<'_> {
    pub async fn deliver(&self, notification_type: &str, message: &str, trust: SourceTrust) {
        // Without a hub the queue delivers it (and the heartbeat records
        // it in the parent session then), so it is recorded once.
        let direct = match self.hub {
            Some(hub) => {
                self.agent
                    .deliver_background_child_result(Some(hub), self.session_id, message, trust)
                    .await
            }
            None => Ok(false),
        };
        let delivered = match direct {
            Ok(sent) => {
                // The chat's record of the push: the daemon's log showed
                // nothing for a finished helper whose result had in fact
                // been sent (live 2026-11-13).
                if sent {
                    info!(
                        session_id = %self.session_id,
                        notification_type = %notification_type,
                        "Delivered a background helper's result to its chat"
                    );
                }
                sent
            }
            Err(e) => {
                warn!(
                    session_id = %self.session_id,
                    goal_id = %self.goal_id,
                    notification_type = %notification_type,
                    error = %e,
                    "spawn_agent background completion notifier: parent delivery failed"
                );
                false
            }
        };

        if !delivered {
            // The queued copy carries the child's taint into whichever
            // parent turn records it.
            enqueue_background_notification(
                self.state,
                self.goal_id,
                self.session_id,
                notification_type,
                message,
                "spawn_agent background completion notifier",
                trust == SourceTrust::Untrusted,
            )
            .await;
        }
    }
}

/// The most sub-agents one `spawn_agent` call may start; at most
/// [`MAX_CONCURRENT_CHILDREN`] of them run at once.
pub const MAX_BATCH_TASKS: usize = 6;
use aidaemon_tools::outcome::MAX_CONCURRENT_CHILDREN;

/// The children a batch call (`tasks`) asks for, each as a full single-spawn
/// argument object: the call's other fields are defaults every task may
/// override. `None` for a single spawn.
pub fn batch_tasks(arguments: &str) -> anyhow::Result<Option<Vec<Value>>> {
    let Ok(Value::Object(mut call)) = serde_json::from_str::<Value>(arguments) else {
        return Ok(None);
    };
    let Some(tasks) = call.remove("tasks") else {
        return Ok(None);
    };
    let Value::Array(tasks) = tasks else {
        anyhow::bail!("`tasks` must be a list of sub-agent tasks");
    };
    anyhow::ensure!(
        !tasks.is_empty() && tasks.len() <= MAX_BATCH_TASKS,
        "`tasks` must list 1 to {MAX_BATCH_TASKS} sub-agents"
    );
    anyhow::ensure!(
        !call.contains_key("mission") && !call.contains_key("task"),
        "With `tasks`, give each sub-agent its own mission and task inside `tasks`"
    );
    let merged = tasks
        .into_iter()
        .map(|task| {
            let Value::Object(task) = task else {
                anyhow::bail!("each entry of `tasks` must be an object with mission and task");
            };
            anyhow::ensure!(
                !task.contains_key("tasks"),
                "an entry of `tasks` cannot hold its own `tasks`"
            );
            let mut item = call.clone();
            item.extend(task);
            Ok(Value::Object(item))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let background = |item: &Value| item.get("background").and_then(Value::as_bool) == Some(true);
    anyhow::ensure!(
        merged.iter().all(background) || !merged.iter().any(background),
        "Every sub-agent in one `tasks` call must share `background`"
    );
    Ok(Some(merged))
}

/// A child's run as the daemon measured it: when it was started and when it
/// finished, when its own steps (its tool calls) ran, apart from its
/// start-up and its report, and how long it waited for the owner's answer
/// on a card, from its own clock.
pub struct MeasuredRun {
    pub started: chrono::DateTime<chrono::Utc>,
    pub finished: chrono::DateTime<chrono::Utc>,
    pub steps: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
    pub waited: Duration,
}

impl MeasuredRun {
    /// The run of a child whose clock is `clock`, started at `started` and
    /// finished now.
    pub fn ended_now(
        started: chrono::DateTime<chrono::Utc>,
        clock: &aidaemon_channels::owner_wait::OwnerWaitClock,
    ) -> Self {
        Self {
            started,
            finished: chrono::Utc::now(),
            steps: clock.steps(),
            waited: clock.waited(),
        }
    }

    /// When its work began: its first step, else when it was started.
    pub fn began(&self) -> chrono::DateTime<chrono::Utc> {
        self.steps.map_or(self.started, |(first, _)| first)
    }

    /// From its first step to its report, its start-up left out (live
    /// 2026-11-13: "Ran 1m 19s" from its first to its last step, beside a
    /// footer time from its start), to the whole second
    /// ([`crate::run_state::measured_span`]).
    pub fn span(&self) -> crate::run_state::MeasuredSpan {
        crate::run_state::measured_span(self.began(), self.finished)
    }

    /// How long it took ([`Self::span`]).
    pub fn took(&self) -> chrono::Duration {
        self.span().took()
    }

    /// Its first step to its report, on the owner's clock. A range under
    /// ten minutes gives seconds, so the stated times agree with the
    /// duration ("8:34:43 PM to 8:37:15 PM ET").
    pub fn range(&self) -> String {
        let span = self.span();
        owner_clock_range(span.from, span.to)
    }
}

/// `from` to `to` in the owner's zone and 12-hour clock, with seconds when
/// the range is under ten minutes.
pub fn owner_clock_range(
    from: chrono::DateTime<chrono::Utc>,
    to: chrono::DateTime<chrono::Utc>,
) -> String {
    if (to - from) >= chrono::Duration::minutes(10) {
        return format!(
            "{} to {}",
            aidaemon_work::schedule::owner_time(&from),
            aidaemon_work::schedule::owner_time(&to)
        );
    }
    let local = |at: chrono::DateTime<chrono::Utc>| at.with_timezone(&chrono::Local);
    let zone = aidaemon_work::schedule::owner_zone_label(local(to).offset().local_minus_utc());
    format!(
        "{} to {} {zone}",
        local(from).format("%-I:%M:%S %p"),
        local(to).format("%-I:%M:%S %p")
    )
}

pub fn compact(duration: chrono::Duration) -> String {
    aidaemon_tools::duration_format::compact_chrono_duration(
        duration,
        aidaemon_tools::duration_format::ZeroUnitStyle::Trim,
    )
}

/// When a child ran and how long it took, measured by the daemon, for the
/// parent: one timing line naming the helper (how long it took, from its
/// first step to its report, and when, in the owner's zone and 12-hour
/// clock, and its wait for the owner), then that it is quoted, never
/// recomputed. The owner reads the same measurement on the card
/// ([`took_line`]). The line
/// keeps [`MEASURED_TIMING_MARK`], so a later turn still sees it when this
/// result is cut to a summary. A parent sees which of its children
/// overlapped, and the times, durations and waits it reports are these
/// measurements, never arithmetic on times the child printed (live
/// 2026-11-12: "about 16 minutes (17:44 to 18:00)", and a "5-minute wait for
/// approval" the model guessed).
pub fn run_window(name: &str, run: &MeasuredRun) -> String {
    let took = run.took();
    format!(
        "{MEASURED_TIMING_MARK}{name} took {} ({}s), {}.{}\n\
         These are the daemon's measurements, already on the owner's card: quote them as \
         given if you mention them (the owner's time zone, 12-hour clock), never work one \
         out; times in the sub-agent's own reply are its estimates.",
        compact(took),
        took.num_seconds(),
        run.range(),
        owner_wait_note(run.waited),
    )
}

/// A failed child's measured run, for the parent: how long it ran before it
/// failed, which is never a time to finish its work. Its work's time to
/// finish exists only once something finishes it: quote a later helper's
/// measured time, or the run's own measured end when it does the work itself.
pub fn failed_run_window(name: &str, run: &MeasuredRun) -> String {
    let took = run.took();
    format!(
        "{MEASURED_TIMING_MARK}{name} ran {} ({}s), {}, and failed.{}\n\
         These are the daemon's measurements: it failed after this long, so this is not how \
         long its work takes. If you start another helper for this work, its own measured time \
         is the time to finish; if you do the work yourself, the daemon measures it from this \
         failure to the end of your turn and puts that time under your reply. Never estimate \
         one.",
        compact(took),
        took.num_seconds(),
        run.range(),
        owner_wait_note(run.waited),
    )
}

/// How long a child waited for the owner's answer on its cards, from its
/// own wait clock; empty when it never waited.
pub fn owner_wait_note(waited: Duration) -> String {
    if waited.as_secs() == 0 {
        return String::new();
    }
    format!(
        " Of that, it waited {} for the owner's answer on an approval card.",
        aidaemon_tools::duration_format::compact_seconds(
            i64::try_from(waited.as_secs()).unwrap_or(i64::MAX),
            aidaemon_tools::duration_format::ZeroUnitStyle::Trim,
        )
    )
}

/// A background helper's measured run for the chat, from its run record:
/// one duration, from its first step to its end, and the times it spans
/// (live 2026-11-13: "Ran 2m 44s (8:34 PM ET to 8:37 PM ET)" read as the
/// whole run, but was its first to last tool call). `None` while it runs.
/// When the owner's request that started it is on record, the line also
/// says how long after that request it was done, so "how long did it take"
/// is answered from the record, request to finish.
pub fn background_timing(run: &crate::run_state::RunState) -> Option<String> {
    let span = run.span()?;
    Some(format!(
        "{MEASURED_TIMING_MARK}{}{}",
        took_line(span, run.waited),
        run.requested_at
            .map(|requested| since_request(requested, span.to))
            .unwrap_or_default()
    ))
}

/// " Done 4m 10s after your request (4:25:46 AM ET)." from the measured
/// request time and end.
pub fn since_request(
    requested: chrono::DateTime<chrono::Utc>,
    done: chrono::DateTime<chrono::Utc>,
) -> String {
    let span = crate::run_state::measured_span(requested, done);
    format!(
        " Done {} after your request ({}).",
        compact(span.took()),
        owner_clock_range(span.from, span.to)
    )
}

/// How long a helper took and when, for the owner, from what the runtime
/// measured: "Took 21s (2:21:16 PM to 2:21:37 PM ET)." and its wait for the
/// owner's answer. The one wording every surface shows a helper's time in.
pub fn took_line(span: crate::run_state::MeasuredSpan, waited: Duration) -> String {
    let wait = if waited.as_secs() == 0 {
        String::new()
    } else {
        format!(
            " It waited {} of that for your answer.",
            aidaemon_tools::duration_format::compact_seconds(
                i64::try_from(waited.as_secs()).unwrap_or(i64::MAX),
                aidaemon_tools::duration_format::ZeroUnitStyle::Trim,
            )
        )
    };
    format!(
        "Took {} ({}).{wait}",
        compact(span.took()),
        owner_clock_range(span.from, span.to)
    )
}

/// One result for a batch call: each child's result under its mission, in
/// the order given. It succeeded only if every child did, and it fails only
/// if every child failed.
pub fn combine_batch(
    runs: Vec<(String, anyhow::Result<ToolCallOutcome>)>,
    background: bool,
) -> anyhow::Result<ToolCallOutcome> {
    let count = runs.len();
    let mut sections = Vec::with_capacity(count);
    let mut failures = 0;
    let mut all_succeeded = true;
    let mut trust = SourceTrust::Trusted;
    let mut truncation: Option<aidaemon_tools::semantics::TruncationInfo> = None;
    for (mission, run) in runs {
        // Its own name, in words the parent can repeat to the owner. A
        // background start already names its helper in one line, which the
        // run may ship as its reply: no heading over it.
        let heading = if background {
            String::new()
        } else {
            format!("### {mission}\n")
        };
        match run {
            Ok(outcome) => {
                if outcome.metadata.output_trust != Some(SourceTrust::Trusted) {
                    trust = SourceTrust::Untrusted;
                }
                if outcome.metadata.outcome_status
                    != Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded)
                {
                    all_succeeded = false;
                }
                if let Some(cut) = outcome.metadata.truncation {
                    let sum = truncation.get_or_insert(aidaemon_tools::semantics::TruncationInfo {
                        shown_chars: 0,
                        total_chars: 0,
                        remediation_hint: cut.remediation_hint.clone(),
                    });
                    sum.shown_chars += cut.shown_chars;
                    sum.total_chars += cut.total_chars;
                }
                sections.push(format!("{heading}{}", outcome.output));
            }
            Err(error) => {
                failures += 1;
                all_succeeded = false;
                // An error may quote what the child wrote.
                trust = SourceTrust::Untrusted;
                sections.push(if background {
                    format!("Could not start {mission}: {error}")
                } else {
                    format!("{heading}Failed: {error}")
                });
            }
        }
    }
    let output = sections.join(if background { "\n" } else { "\n\n" });
    if failures == count {
        anyhow::bail!("{output}");
    }
    let mut outcome = reported(output, trust);
    outcome.metadata.truncation = truncation;
    if background {
        return Ok(outcome.handed_to_background());
    }
    outcome.metadata.outcome_status = Some(if all_succeeded {
        aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded
    } else {
        aidaemon_tools::semantics::ToolOutcomeStatus::CompletedWithNegativeResult
    });
    Ok(outcome)
}

#[derive(Deserialize)]
pub struct SpawnArgs {
    /// High-level mission / role description for the sub-agent. Optional:
    /// a call that gives a title and a task has said what the child is for
    /// (live 2026-11-13: two models gave `title` and `task` without
    /// `mission`, and the spawn failed on the missing field), so it
    /// defaults to the title, else the task.
    #[serde(default)]
    pub mission: String,
    /// A short name the owner reads for this sub-agent on its cards and
    /// status lines ("Drupal install").
    #[serde(default)]
    pub title: Option<String>,
    /// The concrete task or question the sub-agent should work on.
    pub task: String,
    /// When true, spawn the sub-agent in the background and return immediately.
    #[serde(default)]
    pub background: bool,
    /// Zero-based indices of the `track_requirements` items a background
    /// child does: they show as running in the background, and its end
    /// marks them done or failed.
    #[serde(default)]
    pub checklist_items: Vec<usize>,
    /// Task ID — when provided by a task lead, the executor tracks activity against this task.
    #[serde(default)]
    pub task_id: Option<String>,
    /// Optional specialist profile hint. The parent LLM can pick one of the supported
    /// kinds (code, browser_verifier, artifact_writer, research, review, comms_draft,
    /// executor, generic) so the child agent uses the matching profile. Threaded
    /// through `spawn_child`; Task 12 wires it into kind resolution.
    #[serde(default)]
    pub specialist: Option<String>,
    /// Which model tier the child runs on.
    #[serde(default)]
    pub model_tier: Option<ChildModelTier>,
    /// Only these tools. The child is refused any other.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// `read_only` keeps the child from changing anything.
    #[serde(default)]
    pub access: Option<ChildAccess>,
    /// Files or directories the child will change.
    #[serde(default)]
    pub scope: Vec<String>,
    /// Who wins where that scope overlaps a running sibling's.
    #[serde(default)]
    pub precedence: Option<crate::coordination::Precedence>,
    /// Folders this child creates fresh (a new install or project): each
    /// must not exist yet; the spawn creates them ([`claim_new_folders`]).
    #[serde(default)]
    pub new_folders: Vec<String>,
}

/// Make each of `folders` fresh and empty for a child's work, and return
/// them as absolute paths: the one place a spawn's `new_folders` become real.
/// Creating the folder is what claims its name, so two siblings can never
/// both start in it, and the child finds it there to work in (live
/// 2026-11-13: a helper sent to install into a folder that did not exist yet
/// found it missing and gave up). A folder that already exists, or a path
/// that is not absolute, is refused with the next free name, and nothing is
/// created (live 2026-11-13: "fresh" installs went into claude-drupal-3,
/// which held an earlier site, and preserved it).
pub fn claim_new_folders(folders: &[String]) -> anyhow::Result<Vec<String>> {
    let mut wanted = Vec::with_capacity(folders.len());
    let mut refused = Vec::new();
    for folder in folders {
        let path = std::path::PathBuf::from(shellexpand::tilde(folder.trim()).as_ref());
        if !path.is_absolute() {
            refused.push(format!("{} is not an absolute path", path.display()));
        } else if path.exists() {
            refused.push(format!(
                "{} already exists (the next free name is {})",
                path.display(),
                next_free_folder(&path).display()
            ));
        } else if !wanted.contains(&path) {
            wanted.push(path);
        }
    }
    let mut made: Vec<std::path::PathBuf> = Vec::new();
    if refused.is_empty() {
        for path in &wanted {
            let parent_ready = path.parent().map_or(Ok(()), std::fs::create_dir_all);
            match parent_ready.and_then(|()| std::fs::create_dir(path)) {
                Ok(()) => made.push(path.clone()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    refused.push(format!(
                        "{} already exists (the next free name is {})",
                        path.display(),
                        next_free_folder(path).display()
                    ));
                    break;
                }
                Err(error) => {
                    refused.push(format!("{} could not be created: {error}", path.display()));
                    break;
                }
            }
        }
    }
    if !refused.is_empty() {
        release_new_folders(&made);
        anyhow::bail!(
            "No helper started: {}. New work needs an absolute folder that doesn't exist yet. \
             Call again with the free folder in title, task, scope and new_folders, or ask \
             the owner whether to use the existing one.",
            refused.join("; ")
        );
    }
    Ok(made
        .into_iter()
        .map(|path| path.display().to_string())
        .collect())
}

/// Give back folders [`claim_new_folders`] made for a child that did not
/// start or that failed: each is removed only while it is still empty, so
/// the name is free for the next try.
pub fn release_new_folders<P: AsRef<std::path::Path>>(folders: &[P]) {
    for folder in folders {
        let _ = std::fs::remove_dir(folder.as_ref());
    }
}

/// The line a child's task gets for the folders made for it: they exist,
/// empty, and are where its work goes.
pub fn new_folders_note(folders: &[String]) -> String {
    match folders {
        [] => String::new(),
        [one] => format!(
            "\n\nThe folder for this work, {one}, was just created for you and is empty: \
             work in it. It is part of what you may change."
        ),
        many => format!(
            "\n\nThe folders for this work were just created for you and are empty: {}. \
             Work in them. They are part of what you may change.",
            many.join(", ")
        ),
    }
}

/// The first sibling of `path` named like it with a higher number
/// (`site-3` -> `site-4`, `site` -> `site-2`) that does not exist.
pub fn next_free_folder(path: &std::path::Path) -> std::path::PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    let (stem, mut number) = match name.rsplit_once('-') {
        Some((stem, digits))
            if !stem.is_empty()
                && !digits.is_empty()
                && digits.bytes().all(|b| b.is_ascii_digit()) =>
        {
            (stem.to_string(), digits.parse::<u64>().unwrap_or(1))
        }
        _ => (name.clone(), 1),
    };
    loop {
        number += 1;
        let candidate = path.with_file_name(format!("{stem}-{number}"));
        if !candidate.exists() {
            return candidate;
        }
    }
}

impl SpawnArgs {
    /// Parse a single spawn's arguments, filling `mission` from the title
    /// (else the task) when the call left it out.
    pub fn parse(arguments: &str) -> anyhow::Result<Self> {
        let mut args: SpawnArgs = serde_json::from_str(arguments).map_err(|error| {
            anyhow::anyhow!("{error}. Give a title and a task, or several sub-agents as `tasks`.")
        })?;
        anyhow::ensure!(
            !args.task.trim().is_empty(),
            "`task` is empty. Say what the sub-agent should do in `task`."
        );
        if args.mission.trim().is_empty() {
            args.mission = args
                .title
                .as_deref()
                .map(str::trim)
                .filter(|title| !title.is_empty())
                .unwrap_or_else(|| args.task.trim())
                .to_string();
        }
        Ok(args)
    }

    /// The one name the owner reads for this child, from its start line to
    /// its result: its title, else its mission.
    pub fn name(&self) -> &str {
        self.title
            .as_deref()
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .unwrap_or(self.mission.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildAccess {
    ReadOnly,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildModelTier {
    Fast,
    Primary,
    Smart,
}

impl From<ChildModelTier> for aidaemon_providers::routing::Tier {
    fn from(tier: ChildModelTier) -> Self {
        match tier {
            ChildModelTier::Fast => Self::Fast,
            ChildModelTier::Primary => Self::Primary,
            ChildModelTier::Smart => Self::Smart,
        }
    }
}

/// A tool name as a caller wrote it in a grant. Models sometimes copy the
/// provider's function-call namespace (`functions.read_file`) into the
/// argument; the tool is `read_file`.
pub fn tool_name_from_grant(raw: &str) -> String {
    let name = raw.trim();
    name.strip_prefix("functions.")
        .unwrap_or(name)
        .trim()
        .to_string()
}

/// The granted tool names the caller cannot hand on: not among its own
/// tools, or outside its own authority. A grant naming none of the caller's
/// tools would start a child that can do nothing, so any unknown name is an
/// error the caller can correct.
pub fn ungrantable_tools(
    requested: &aidaemon_core::Authority,
    own_authority: &aidaemon_core::Authority,
    own_tools: Option<&[String]>,
) -> Vec<String> {
    let Some(names) = requested.tools.as_ref() else {
        return Vec::new();
    };
    names
        .iter()
        .filter(|name| {
            let outside_authority = own_authority
                .tools
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(*name));
            let not_owned = own_tools.is_some_and(|tools| !tools.iter().any(|tool| tool == *name));
            outside_authority || not_owned
        })
        .cloned()
        .collect()
}

/// The authority the caller asked for its child, if it asked for one.
pub fn requested_child_authority(
    args: &SpawnArgs,
) -> anyhow::Result<Option<aidaemon_core::Authority>> {
    if args.tools.is_none() && args.access.is_none() {
        return Ok(None);
    }
    let tools = args.tools.as_ref().map(|names| {
        names
            .iter()
            .map(|name| tool_name_from_grant(name))
            .filter(|name| !name.is_empty())
            .collect::<std::collections::BTreeSet<_>>()
    });
    anyhow::ensure!(
        tools.as_ref().is_none_or(|names| !names.is_empty()),
        "`tools` must name at least one tool"
    );
    Ok(Some(aidaemon_core::Authority {
        level: match args.access {
            Some(ChildAccess::ReadOnly) => aidaemon_core::AccessLevel::ReadOnly,
            _ => aidaemon_core::AccessLevel::Write,
        },
        tools,
    }))
}

/// The child's channel context: the parent call's visibility and trust,
/// from the dispatcher's facts.
pub fn build_child_channel_context(facts: &ToolCallContext) -> ChannelContext {
    let visibility = facts
        .channel_visibility
        .unwrap_or(ChannelVisibility::Internal);
    ChannelContext {
        visibility,
        platform: "internal".to_string(),
        channel_name: None,
        channel_id: None,
        workspace_id: None,
        sender_name: None,
        sender_id: None,
        channel_member_names: vec![],
        user_id_map: std::collections::HashMap::new(),
        workspace_grant: None,
        trusted: facts.trusted_session,
    }
}

#[async_trait]
impl Tool for SpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Delegate a focused task to a fresh sub-agent. PREFER delegation when the work is multi-step \
         (research → synthesize, code → test, fetch → analyze), matches a named specialist (see the \
         `specialist` enum: code, browser_verifier, research, review, artifact_writer, comms_draft, ...), \
         benefits from a fresh context window, or should keep running while you do other work (background=true: its result goes to the user's chat, not back to you, so never use it for work you need to combine). To run several sub-agents at the same time, give them all in one call as `tasks` (up to 6, 3 running at once); you receive every result, each with when it ran. Do NOT delegate \
         trivial one-shot reads (a single read_file, a single web_search) — handle those directly. \
         The sub-agent starts with a fresh context and the same tools unless its specialist narrows them (a named specialist delegates further only when its `tools` include spawn_agent), so reference files by path and \
         do not paste prior tool output."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "spawn_agent",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "mission": {
                        "type": "string",
                        "description": "Optional: the wider goal this work serves, for the \
                            sub-agent's own context. Left out, it is the title."
                    },
                    "title": {
                        "type": "string",
                        "description": "What this sub-agent does, in a few plain words the owner \
                            reads on its card and approval cards: the task itself, as the owner \
                            would say it, and, when the work creates or works in a folder, that \
                            folder's name, so the owner sees where it goes from the start (e.g. \
                            'Count the lines in numbers.txt', 'Install Drupal in claude-drupal-6'). \
                            It names the work, not the sub-agent doing it, and is never a sentence."
                    },
                    "task": {
                        "type": "string",
                        "description": "The specific task or question the sub-agent should accomplish. \
                            Keep it self-contained but concise: state the goal, success criteria, and any \
                            non-obvious constraints. Do NOT paste file contents, prior tool output, or \
                            conversation excerpts — give file paths and let the sub-agent read them itself. \
                            Ask only for the work and what it found: the daemon measures every \
                            helper's time (start, finish, and since the owner's request) on its own \
                            record, so a helper never times itself or reports a duration."
                    },
                    "background": {
                        "type": "boolean",
                        "description": "When true, spawn the sub-agent in the background and return immediately. \
                            The result will be delivered through the parent session when the sub-agent finishes. \
                            Use this for long-running tasks where the user doesn't need to wait.",
                        "default": false
                    },
                    "checklist_items": {
                        "type": "array",
                        "items": { "type": "integer", "minimum": 0 },
                        "uniqueItems": true,
                        "description": "With background=true: the zero-based indices of your track_requirements items this sub-agent does. They show as running in the background until it ends, then as done or failed by its result."
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to associate with this executor (used by task leads to connect executor work to task tracking)"
                    },
                    "specialist": {
                        "type": "string",
                        "enum": [
                            "code",
                            "browser_verifier",
                            "artifact_writer",
                            "research",
                            "review",
                            "comms_draft",
                            "executor",
                            "generic"
                        ],
                        "description": self.build_specialist_arg_description()
                    },
                    "model_tier": {
                        "type": "string",
                        "enum": ["fast", "primary", "smart"],
                        "description": "Optional: fast for simple lookups, smart for hard reasoning. Omit to use the same model as this agent."
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "description": "Optional: the only tools this child may use. Any other call is refused, and it can never gain a tool you lack."
                    },
                    "access": {
                        "type": "string",
                        "enum": ["read_only", "write"],
                        "description": "Optional: read_only lets the child only observe: every call that could change something is refused, including any command that runs a program or script (python, node, a build). Leave it out when the task runs code or changes files. It can never exceed your own access."
                    },
                    "scope": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional: the files or directories this child will change. Given (here or as new_folders), the child may write only there; anything else it would change is put to the owner. When several children will write, give them disjoint scopes so they do not collide. If a scope must overlap a running child's, say who wins there with precedence."
                    },
                    "new_folders": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "When the work makes something new in a folder of its own (a fresh install, a new project), that folder's absolute path. It is created empty before the child starts, and the child may write in it. The call fails if it already exists and names the next free folder, so pick that one (in title, task, scope and here) or ask the owner; never send a fresh install into a folder that holds something."
                    },
                    "precedence": {
                        "type": "string",
                        "enum": ["this_child", "running_siblings", "shared"],
                        "description": "Give this whenever scope overlaps a running child's: whose change stands if both change the same lines there. this_child: this child's. running_siblings: the children already running. shared: neither, and each such change is put to you to decide. Omitted, it is shared. No child ever overwrites another's lines without one of these deciding it."
                    },
                    "tasks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_BATCH_TASKS,
                        "items": { "type": "object" },
                        "description": "Several independent sub-agents to run in parallel, instead of mission and task. Each entry is an object with its own mission and task and any of the other fields above; fields given beside tasks apply to every entry unless it sets its own. All entries share background."
                    }
                },
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
        if let Some(tasks) = batch_tasks(arguments)? {
            return self.execute_batch(tasks, facts).await;
        }
        match self.execute_one(arguments, facts).await {
            Ok(outcome) => Ok(outcome),
            Err(error) => Err(self.steps_not_started(arguments, facts, error).await),
        }
    }
}

impl SpawnAgentTool {
    /// A `spawn_agent` call that failed started no helper: the checklist
    /// steps it named (`checklist_items`) are marked as not running, the
    /// card shows it, and the failure tells the model which of its steps
    /// nothing works on, to retry or to tell the owner (live 2026-11-13: a
    /// spawn that failed on its arguments left its step reading as handed
    /// off).
    pub async fn steps_not_started(
        &self,
        arguments: &str,
        facts: &aidaemon_tools::semantics::ToolCallContext,
        error: anyhow::Error,
    ) -> anyhow::Error {
        let items: Vec<usize> = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|call| call.get("checklist_items").cloned())
            .and_then(|items| serde_json::from_value(items).ok())
            .unwrap_or_default();
        if items.is_empty() {
            return error;
        }
        let reason = format!("its helper didn't start: {error}");
        if let (Some(plans), Some(session)) = (self.plan_store.get(), facts.session()) {
            match plans
                .helper_did_not_start(session, facts.task_id.as_deref(), &items, &reason)
                .await
            {
                Ok(Some(plan)) => {
                    if let Some(tx) = aidaemon_tools::status::status_sender(facts) {
                        let _ = tx.try_send(StatusUpdate::Checklist(plan.card()));
                    }
                }
                Ok(None) => {}
                Err(store_error) => {
                    warn!(%store_error, "Could not mark checklist steps whose helper didn't start")
                }
            }
        }
        let steps = items
            .iter()
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::anyhow!(
            "{error}\n\nNo helper started, so nothing is working on checklist item(s) {steps}. \
             Call spawn_agent again with corrected arguments, or tell the owner that this part \
             isn't running."
        )
    }

    pub async fn execute_one(
        &self,
        arguments: &str,
        facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
        let mut args = SpawnArgs::parse(arguments)?;
        let new_folders = claim_new_folders(&args.new_folders)?;
        args.task.push_str(&new_folders_note(&new_folders));
        let result = self.start_child(args, new_folders.clone(), facts).await;
        if result.is_err() {
            release_new_folders(&new_folders);
        }
        result
    }

    /// Start one child for parsed `args`, whose `new_folders` were made.
    async fn start_child(
        &self,
        args: SpawnArgs,
        new_folders: Vec<String>,
        facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
        let status_tx = aidaemon_tools::status::status_sender(facts);
        let facts = facts.clone();
        let project_scope = facts.project_scope.clone();
        let mut parent_authority = crate::authority::parent_authority(&facts).cloned();
        // A child can be made narrower than its parent and never wider.
        let agent = self.get_agent()?;
        if let Some(requested) = requested_child_authority(&args)? {
            let parent = parent_authority.as_mut().ok_or_else(|| {
                anyhow::anyhow!("A child's tools and access can only be set inside a turn")
            })?;
            let own_tools = agent.tool_names();
            let unknown = ungrantable_tools(&requested, &parent.authority, own_tools.as_deref());
            if !unknown.is_empty() {
                let mut grantable = own_tools.unwrap_or_default();
                if let Some(allowed) = parent.authority.tools.as_ref() {
                    grantable.retain(|tool| allowed.contains(tool));
                }
                grantable.sort();
                anyhow::bail!(
                    "`tools` names tools you cannot give this child: {}. Use exact tool names \
                     from: {}.",
                    unknown.join(", "),
                    grantable.join(", ")
                );
            }
            parent.authority = parent.authority.narrow(&requested);
        }

        info!(
            depth = agent.depth(),
            max_depth = agent.max_depth(),
            mission = %args.mission,
            background = args.background,
            "spawn_agent tool invoked"
        );

        // Preserve the parent call's execution context.
        let channel_ctx = build_child_channel_context(&facts);

        // Propagate parent's user role to prevent privilege escalation.
        // Default to Guest (least privilege) if not provided.
        let user_role = match facts.user_role {
            Some(UserRole::Owner) => UserRole::Owner,
            _ => UserRole::Guest,
        };

        // Determine child role based on parent's role.
        let child_role = if agent.role() == AgentRole::TaskLead {
            Some(AgentRole::Executor)
        } else {
            None
        };

        // LLM-provided task_id takes priority; fall back to the call's task
        let task_id_ref = args.task_id.clone().or(facts.task_id.clone());
        // The durable task this child works on, if any. A dispatcher call
        // always carries its turn's own id, which names no durable task, so
        // a plain child has one only when the caller named it.
        let durable_task_id = if child_role == Some(AgentRole::Executor) {
            task_id_ref.clone()
        } else {
            args.task_id.clone()
        };
        let goal_id_ref = facts.goal_id.clone();

        // Plain children count against the request's tree. A task lead's
        // executors are bounded per task instead, and keep budgets of their own.
        let family_permit = if child_role.is_some() {
            if let Some(parent) = parent_authority.as_mut() {
                parent.spawn_family = None;
            }
            None
        } else {
            match parent_authority
                .as_ref()
                .and_then(|parent| parent.spawn_family.as_ref())
            {
                Some(family) => Some(family.try_admit().map_err(|why| anyhow::anyhow!(why))?),
                None => None,
            }
        };

        // A plain child in a tree joins its roster, saying what it will change.
        let mut scope_overlap_note = String::new();
        let tree = family_permit
            .is_some()
            .then(|| {
                parent_authority
                    .as_ref()
                    .and_then(|parent| parent.spawn_family.clone())
            })
            .flatten();
        // What the child will change, resolved once: its write authority
        // ([`crate::authority::narrow_for_child`]) and, in a tree, its place
        // on the roster.
        let mut scope = agent.resolve_child_scope(args.scope.clone()).await;
        for folder in &new_folders {
            if !scope.contains(folder) {
                scope.push(folder.clone());
            }
        }
        if let Some(parent) = parent_authority.as_mut() {
            parent.child_declaration = Some(crate::authority::ChildDeclaration {
                parent_session: facts.session_id.clone(),
                scope: scope.clone(),
                new_folders: new_folders.clone(),
                precedence: args.precedence,
                claim: None,
            });
        }
        if let Some(family) = tree.as_ref() {
            use crate::coordination::Precedence;
            let (claim, overlaps) = family.coordination().declare_scope(&args.mission, &scope);
            if !overlaps.is_empty() {
                let listed = overlaps
                    .iter()
                    .map(|(sibling, path)| format!("{path} (also {sibling})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let rule = match args.precedence {
                    Some(Precedence::ThisChild) => {
                        "This child takes precedence there: its change to lines the other \
                         changed is made, and the other's change to its lines is refused."
                    }
                    Some(Precedence::RunningSiblings) => {
                        "The running sub-agents take precedence there: this child's change to \
                         lines they changed is refused."
                    }
                    Some(Precedence::Shared) => {
                        "The area is shared: a change either makes to the other's lines there \
                         is put to you to decide."
                    }
                    None => {
                        "You gave no precedence, so the area is shared: a change either makes \
                         to the other's lines there is put to you to decide, one at a time. \
                         When you start children whose scopes overlap, pass precedence \
                         (this_child, running_siblings or shared) to settle it up front."
                    }
                };
                scope_overlap_note = format!(
                    "\n\nThis child's scope overlaps running sub-agents at: {listed}. {rule}"
                );
            }
            if let Some(declaration) = parent_authority
                .as_mut()
                .and_then(|parent| parent.child_declaration.as_mut())
            {
                declaration.claim = Some(Arc::new(claim));
            }
        }

        // The child learns the chain of work it belongs to: its parent's
        // lineage plus the mission this call delegates.
        if let Some(parent) = parent_authority.as_mut() {
            parent.lineage.push(args.mission.clone());
            parent.card_title = args
                .title
                .as_deref()
                .map(str::trim)
                .filter(|title| !title.is_empty())
                .map(str::to_string);
        }

        // TaskLead -> Executor spawns must target a concrete, pre-claimed task and
        // are deduplicated per task_id so the same work is not launched twice.
        let executor_task_id = if child_role == Some(AgentRole::Executor) {
            let Some(task_id) = task_id_ref.clone() else {
                return Err(anyhow::anyhow!(
                    "TaskLead must pass task_id when spawning an executor. Claim a task first with manage_goal_tasks(action='claim_task')."
                ));
            };
            if let Err(e) = agent
                .validate_executor_task_for_spawn(&task_id, goal_id_ref.as_deref())
                .await
            {
                return Err(anyhow::anyhow!(
                    "Blocked executor spawn for task {}: {}",
                    task_id,
                    e
                ));
            }
            if !self.try_begin_executor_task(&task_id).await {
                return Err(anyhow::anyhow!(
                    "Task {} already has an executor running. Wait for it to finish before spawning another.",
                    task_id
                ));
            }
            Some(task_id)
        } else {
            None
        };

        let approval_session_id = facts.session().map(str::to_string);

        // Background needs a way to deliver the result and a session to
        // deliver it to; without either the child runs in the foreground.
        let hub = self.get_hub();
        let state = self.state.clone();
        let hub_for_parent_delivery = hub.as_ref().map(Arc::downgrade);
        let background_session = if !args.background {
            None
        } else if hub.is_none() && state.is_none() {
            info!(
                "Background mode requested but no hub/state notification path is available, falling back to sync"
            );
            None
        } else if let Some(id) = approval_session_id.as_deref().filter(|id| !id.is_empty()) {
            Some(agent.background_delivery_session(id))
        } else {
            info!("Background mode requested but no session_id, falling back to sync");
            None
        };
        let Some(session_id) = background_session else {
            let result = crate::authority::within_permit(
                family_permit.as_ref(),
                self.run_sync(
                    agent,
                    &args.mission,
                    &args.task,
                    status_tx,
                    channel_ctx,
                    user_role,
                    child_role,
                    goal_id_ref.as_deref(),
                    task_id_ref.as_deref(),
                    durable_task_id.as_deref(),
                    project_scope.as_deref(),
                    args.specialist.as_deref(),
                    approval_session_id.as_deref(),
                    parent_authority.clone(),
                    args.model_tier.map(Into::into),
                    tree.as_ref(),
                ),
            )
            .await;
            if let Some(ref task_id) = executor_task_id {
                self.finish_executor_task(task_id).await;
            }
            return result.map(|mut outcome| {
                outcome.output.push_str(&scope_overlap_note);
                outcome
            });
        };

        let task = args.task.clone();
        let mission = args.mission.clone();
        // The one name the owner reads for it, on its start line and its
        // result alike (live 2026-11-13: "Count slowly to 30" started and
        // "Time a slow count and notify the user when finished" finished).
        let name = args.name().to_string();
        let started_as = background_spawn_outcome(&name);
        let timeout_secs = self.timeout_secs;
        let time_limit = self.time_limit();
        let max_response_chars = self.max_response_chars;
        let executor_task_runs = Arc::clone(&self.executor_task_runs);
        let executor_task_id_for_bg = executor_task_id;
        let notify_goal_id = goal_id_ref.clone().unwrap_or_else(|| "global".to_string());
        let tree_for_bg = tree.clone();

        let background_id = format!("background_spawn:{}", uuid::Uuid::new_v4());
        if let Some(state) = state.as_ref() {
            let record = aidaemon_work::goals::BackgroundSpawn {
                id: background_id.clone(),
                session_id: session_id.clone(),
                mission: name.chars().take(BACKGROUND_MISSION_CHARS).collect(),
                started_at: chrono::Utc::now().to_rfc3339(),
            };
            if let Err(error) = state.record_background_spawn(&record).await {
                warn!(%error, "Could not record a background sub-agent");
            }
        }
        let cancel_token = agent.background_cancel_token(&background_id).await;
        let plan_store = self.plan_store.get().cloned();
        // The steps this child does now wait on it (typed: the launch names
        // them). The card shows them running in the background. A launch
        // that names none still leaves the turn's open steps waiting on it
        // (without crediting them to it): the card never says "stopped
        // short" for work that is still running.
        if let (Some(plans), Some(plan_session)) =
            (plan_store.as_ref(), approval_session_id.as_deref())
        {
            let linked = if args.checklist_items.is_empty() {
                plans
                    .wait_on_background_jobs(
                        plan_session,
                        facts.task_id.as_deref(),
                        std::slice::from_ref(&background_id),
                    )
                    .await
            } else {
                plans
                    .link_background_job(
                        plan_session,
                        facts.task_id.as_deref(),
                        &args.checklist_items,
                        &background_id,
                    )
                    .await
            };
            match linked {
                Ok(Some(plan)) => {
                    if let Some(tx) = status_tx.as_ref() {
                        let _ = tx.try_send(StatusUpdate::Checklist(plan.card()));
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    warn!(%error, "Could not hand checklist steps to a background sub-agent")
                }
            }
        }
        let plan_session = approval_session_id.clone();
        let helper_label = crate::spawn::child_status_label(&name);
        // The run's one typed record. Its end pushes the result to the chat
        // through the subscriber below, the moment it ends.
        let runs = agent.helper_runs();
        let (ended_tx, ended_rx) = mpsc::unbounded_channel();
        runs.begin(&background_id, &session_id, &helper_label, Some(ended_tx));
        tokio::spawn(deliver_when_run_ends(
            ended_rx,
            ResultRoute {
                agent: agent.clone(),
                runs: runs.clone(),
                hub: hub_for_parent_delivery.clone(),
                state: state.clone(),
                plans: plan_store.clone(),
                plan_session: plan_session.clone(),
                goal_id: notify_goal_id.clone(),
                name: name.clone(),
                max_response_chars,
                turn: aidaemon_channels::turns::TurnOver::current(),
            },
        ));
        let at_start = match hub.as_ref() {
            Some(hub) => hub.checklist_surface(&session_id).await,
            None => None,
        };
        runs.card_before(&background_id, at_start.clone());
        let status_tx = show_helper_steps_on_card(
            status_tx,
            HelperCard {
                runs: runs.clone(),
                hub: hub_for_parent_delivery.clone(),
                plans: plan_store.clone(),
                plan_session: plan_session.clone(),
                chat: session_id.clone(),
                job: background_id.clone(),
                label: helper_label.clone(),
                at_start,
                drawn: Default::default(),
            },
        );

        tokio::spawn(async move {
            // This task owns `family_permit`, so the slot stays taken until
            // the child has finished.
            let waits = aidaemon_channels::owner_wait::OwnerWaitClock::current()
                .unwrap_or_default()
                .nested();
            // A panic in the child must not skip the cleanup below, and the
            // run still ends (failed), which tells the chat.
            let finished = std::panic::AssertUnwindSafe(async {
                let mut cancelled_by_owner = false;
                // Scoped so that leaving it drops the child's future, which stops
                // the child.
                let result: anyhow::Result<ChildAgentRun> = {
                    let mut result_fut = std::pin::pin!(crate::authority::within_permit(
                        family_permit.as_ref(),
                        aidaemon_channels::owner_wait::OwnerWaitClock::within(
                            Some(waits.clone()),
                            crate::run_state::within_run(
                                &background_id,
                                agent.run_child(ChildAgentRequest {
                                    mission: mission.clone(),
                                    task: task.clone(),
                                    status_tx: status_tx.clone(),
                                    channel_ctx,
                                    user_role,
                                    child_role,
                                    goal_id: goal_id_ref.clone(),
                                    task_id: task_id_ref.clone(),
                                    project_scope: project_scope.clone(),
                                    specialist: args.specialist.clone(),
                                    approval_session_id: Some(session_id.clone()),
                                    parent_authority: parent_authority.clone(),
                                    time_limit,
                                    model_tier: args.model_tier.map(Into::into),
                                })
                            )
                        )
                    ),);
                    let owner_cancelled = async {
                        match cancel_token.as_ref() {
                            Some(token) => token.cancelled().await,
                            None => std::future::pending().await,
                        }
                    };
                    tokio::pin!(owner_cancelled);
                    tokio::select! {
                        res = &mut result_fut => res,
                        _ = &mut owner_cancelled => {
                            cancelled_by_owner = true;
                            Err(anyhow::anyhow!("cancelled by the owner"))
                        }
                    }
                };

                // Only a finished child's own record decides; any other result
                // may carry child-written text and counts as untrusted.
                let trust = match &result {
                    Ok(run) => run.trust,
                    Err(error)
                        if durable_task_id.is_none()
                            && error
                                .downcast_ref::<ChildTimedOut>()
                                .is_some_and(|t| !t.report.may_have_changed_something()) =>
                    {
                        SourceTrust::Trusted
                    }
                    Err(_) => SourceTrust::Untrusted,
                };
                // Stopped from the chat (`/cancel`, a stop request): the
                // same as the owner cancelling it here.
                let cancelled_by_owner = cancelled_by_owner
                    || matches!(&result, Err(error) if error
                        .downcast_ref::<crate::spawn::ChildStoppedByOwner>()
                        .is_some());
                // How it ended (typed), what it found to say about it, and
                // what its receipts left unconfirmed.
                let (phase, evidence, report) = match result {
                    _ if cancelled_by_owner => (
                        RunPhase::Stopped,
                        Default::default(),
                        "You cancelled it. It may have done part of the work.".to_string(),
                    ),
                    Ok(run) => {
                        let mut report = run.response.clone();
                        report.push_str(&coordination_trailer(tree_for_bg.as_ref(), &run.session));
                        if let Some(note) = tree_for_bg
                            .as_ref()
                            .and_then(|family| family.coordination().shared_files_note())
                        {
                            report.push_str(&note);
                        }
                        (run.phase, run.evidence, report)
                    }
                    Err(error) => match error.downcast_ref::<ChildTimedOut>() {
                        None => (RunPhase::Failed, Default::default(), error.to_string()),
                        Some(timed_out) => {
                            // The child may have persisted a terminal outcome
                            // before it was stopped — deliver that instead of a
                            // generic timeout failure.
                            let salvaged = match durable_task_id.as_deref() {
                                Some(task_id) => {
                                    agent
                                        .salvage_executor_task_outcome(task_id, timeout_secs)
                                        .await
                                }
                                None => None,
                            };
                            // Nothing to salvage: release the executor's claimed
                            // task, as a waited-on executor does.
                            if salvaged.is_none() && child_role == Some(AgentRole::Executor) {
                                if let Some(task_id) = durable_task_id.as_deref() {
                                    agent
                                        .mark_executor_task_timeout(task_id, timeout_secs)
                                        .await;
                                }
                            }
                            match salvaged {
                                Some(outcome) => (
                                    match outcome.status.as_str() {
                                        "completed" => RunPhase::Succeeded,
                                        "blocked" => RunPhase::Stopped,
                                        _ => RunPhase::Failed,
                                    },
                                    Default::default(),
                                    outcome.details,
                                ),
                                None => (
                                    RunPhase::Stopped,
                                    Default::default(),
                                    format!(
                                        "It made no progress for its limit and was stopped.\n\n{}",
                                        timed_out.render()
                                    ),
                                ),
                            }
                        }
                    },
                };
                runs.finish(
                    &background_id,
                    phase,
                    measured_end(&waits),
                    evidence,
                    crate::run_state::RunReport {
                        text: report,
                        untrusted: trust == SourceTrust::Untrusted,
                    },
                );
            })
            .catch_unwind()
            .await;
            if finished.is_err() {
                warn!(
                    session_id = %session_id,
                    %background_id,
                    "Background sub-agent panicked; reporting it as failed"
                );
                // The note is the daemon's own text, so it is trusted. A run
                // that already ended keeps how it ended.
                runs.finish(
                    &background_id,
                    RunPhase::Failed,
                    measured_end(&waits),
                    Default::default(),
                    crate::run_state::RunReport {
                        text: "It stopped unexpectedly.".to_string(),
                        untrusted: false,
                    },
                );
            }

            if let Some(state) = state.as_ref() {
                if let Err(error) = state.finish_background_spawn(&background_id).await {
                    warn!(%error, "Could not clear a background sub-agent's record");
                }
            }
            agent.release_background_cancel_token(&background_id).await;

            if let Some(task_id) = executor_task_id_for_bg {
                executor_task_runs.lock().await.remove(&task_id);
            }
        });

        let mut outcome = started_as;
        outcome.output.push_str(&scope_overlap_note);
        Ok(outcome)
    }
}

/// Where a background run's result goes when the run ends, and what it
/// settles there.
pub struct ResultRoute {
    pub agent: Arc<dyn ChildAgentRuntime>,
    pub runs: Arc<crate::run_state::RunStates>,
    pub hub: Option<Weak<dyn OutboundRouter>>,
    pub state: Option<Arc<dyn StateStore>>,
    pub plans: Option<Arc<aidaemon_work::plans::PlanStore>>,
    pub plan_session: Option<String>,
    pub goal_id: String,
    pub name: String,
    pub max_response_chars: usize,
    /// The turn that started the run, when one did: the run's end waits
    /// until it is over.
    pub turn: Option<aidaemon_channels::turns::TurnOver>,
}

/// The subscriber on one run's terminal transition: when the run ends it
/// pushes the result, rendered from the run record, to the chat (or the
/// queue), then settles the plan steps the run was handed and redraws the
/// card. It waits on nothing but the transition: no turn and no inbound
/// message is involved.
pub async fn deliver_when_run_ends(
    mut ended: mpsc::UnboundedReceiver<crate::run_state::RunEnded>,
    route: ResultRoute,
) {
    let Some(ended) = ended.recv().await else {
        return;
    };
    // The turn that started it shows everything first (its reply, the
    // "Started in the background" ack, and the card it kept): the run's end
    // never lands above the turn's own reply.
    if let Some(turn) = route.turn.as_ref() {
        turn.wait().await;
    }
    let state = ended.state;
    let trust = if ended.report.untrusted {
        SourceTrust::Untrusted
    } else {
        SourceTrust::Trusted
    };
    // The card settles to its final header first; the result follows,
    // without saying again how it ended.
    let on_card = settle_background_steps(
        &route.runs,
        route.plans.as_ref(),
        route.hub.as_ref(),
        route.plan_session.as_deref(),
        &state.chat,
        &state.id,
        job_end(state.phase),
    )
    .await;
    let message = render_run_result(
        &state,
        &route.name,
        &ended.report.text,
        route.max_response_chars,
        on_card,
    );
    let notification_type = run_notification_type(state.phase);
    let delivery = BackgroundDelivery {
        agent: route.agent.as_ref(),
        hub: route.hub.as_ref(),
        state: route.state.as_ref(),
        session_id: &state.chat,
        goal_id: &route.goal_id,
    };
    let delivered =
        std::panic::AssertUnwindSafe(delivery.deliver(notification_type, &message, trust))
            .catch_unwind()
            .await;
    if delivered.is_err() {
        warn!(session_id = %state.chat, job_id = %state.id, "Delivering a helper's result panicked; queuing it");
        enqueue_background_notification(
            route.state.as_ref(),
            &route.goal_id,
            &state.chat,
            notification_type,
            &message,
            "spawn_agent background completion notifier",
            ended.report.untrusted,
        )
        .await;
    }
}

/// What the runtime measured of a background run as it ends: now, its
/// first tool step and its wait for the owner.
pub fn measured_end(
    clock: &aidaemon_channels::owner_wait::OwnerWaitClock,
) -> crate::run_state::RunMeasure {
    crate::run_state::RunMeasure {
        ended_at: chrono::Utc::now(),
        work_began: clock.steps().map(|(first, _)| first),
        waited: clock.waited(),
    }
}

/// The coordination log's entries about one child, for the end of its result.
pub fn coordination_trailer(tree: Option<&crate::authority::SpawnFamily>, session: &str) -> String {
    tree.and_then(|family| family.coordination().result_trailer(session))
        .unwrap_or_default()
}

/// A background run ended: settle the checklist steps it was given, then
/// redraw its own card (the one its starting turn kept) from the records:
/// the plan and every run drawn on that card. Once nothing on the card is
/// still running, the plan's steps still open settle as not done (nothing
/// is left to do them) and the card settles to its final state, by the same
/// rule as a turn's card ([`aidaemon_core::card::CardEnd::of`]); the
/// records of its runs are then let go. A run whose card is unknown edits
/// the chat's plan card (or sends it fresh) when it had a plan. `true` when
/// a card now shows how the run ended; the result message follows on its
/// own and never says it again.
pub async fn settle_background_steps(
    runs: &crate::run_state::RunStates,
    plan_store: Option<&Arc<aidaemon_work::plans::PlanStore>>,
    hub: Option<&Weak<dyn OutboundRouter>>,
    plan_session: Option<&str>,
    chat_session: &str,
    job_id: &str,
    end: aidaemon_work::plans::JobEnd,
) -> bool {
    // Its run record already says it ended, so a step redraw that starts
    // now draws nothing; this waits for any still in flight.
    let writer = card_writer(chat_session);
    let _turn = writer.lock().await;
    let plan_id = job_plan(plan_store, plan_session, job_id)
        .await
        .map(|plan| plan.id);
    let settled = match (plan_store, plan_session) {
        (Some(plans), Some(plan_session)) => {
            match plans.settle_background_job(plan_session, job_id, end).await {
                Ok(plan) => plan,
                Err(error) => {
                    warn!(%error, job_id, "Could not settle a background job's checklist steps");
                    None
                }
            }
        }
        _ => None,
    };
    let plan_id = plan_id.or_else(|| settled.as_ref().map(|plan| plan.id.clone()));
    // A sub-agent's own checklist is not the card in the chat.
    if plan_session.is_some_and(|plan_session| plan_session != chat_session) {
        runs.forget(job_id);
        return false;
    }
    let Some(hub) = hub.and_then(Weak::upgrade) else {
        runs.forget(job_id);
        return false;
    };
    let Some(surface) = run_card(hub.as_ref(), runs, job_id, chat_session).await else {
        // Its turn kept no card it can be told by: the plan's card shows
        // the plan and this run.
        let this: Vec<_> = runs.get(job_id).into_iter().collect();
        runs.forget(job_id);
        return match kept_card(settled.as_ref(), &this) {
            Some(card) if settled.is_some() => {
                update_checklist_card(hub.as_ref(), chat_session, &card).await
            }
            _ => false,
        };
    };
    let on_card = runs_on_card(hub.as_ref(), runs, chat_session, &surface).await;
    let all_ended = on_card.iter().all(|run| run.phase.is_terminal());
    let mut plan = settled;
    if let (Some(plans), Some(plan_id)) = (plan_store, plan_id.as_deref()) {
        let latest = if all_ended {
            plans.settle_open_steps(plan_id).await
        } else {
            plans.get(plan_id).await
        };
        match latest {
            Ok(Some(latest)) => plan = Some(latest),
            Ok(None) => {}
            Err(error) => warn!(%error, job_id, "Could not read a background run's plan"),
        }
    }
    let shown = match kept_card(plan.as_ref(), &on_card) {
        Some(card) => match hub.edit_text(chat_session, &surface, &card).await {
            Ok(edited) => edited,
            Err(error) => {
                warn!(session_id = %chat_session, %error, "Could not settle a run's card");
                false
            }
        },
        None => false,
    };
    if all_ended {
        for run in &on_card {
            runs.forget(&run.id);
        }
    }
    shown
}

/// The card run `job` is drawn on: the one its run record names, else the
/// first card the chat kept after the run began (the card its starting turn
/// kept), which is then recorded on the run. `None` while that turn has not
/// kept one, or once the chat's card history no longer tells which it was.
pub async fn run_card(
    hub: &dyn OutboundRouter,
    runs: &crate::run_state::RunStates,
    job: &str,
    chat: &str,
) -> Option<String> {
    match runs.get(job)?.card {
        crate::run_state::RunCard::Kept(card) => Some(card),
        crate::run_state::RunCard::Pending { before } => {
            let card = hub.checklist_surface_after(chat, before.as_deref()).await?;
            runs.keep_card(job, &card);
            Some(card)
        }
    }
}

/// Show `card` in the session's chat: edit the plan card the turn left in
/// place, else send it as a new card. `true` when the chat shows it.
pub async fn update_checklist_card(hub: &dyn OutboundRouter, session_id: &str, card: &str) -> bool {
    if let Some(id) = hub.checklist_surface(session_id).await {
        if matches!(hub.edit_text(session_id, &id, card).await, Ok(true)) {
            return true;
        }
    }
    match hub.send_text_tracked(session_id, card).await {
        Ok(Some(id)) => {
            hub.register_checklist_surface(session_id, &id).await;
            true
        }
        Ok(None) => true,
        Err(error) => {
            warn!(session_id, %error, "Could not show the updated checklist");
            false
        }
    }
}

/// A sub-agent now running in the background: the turn hands off to it, and
/// its result reaches the parent session when it completes.
/// What a background start returns. A run that ends on the start ships this
/// as its reply, so it is one plain line naming the helper.
pub fn background_spawn_outcome(name: &str) -> ToolCallOutcome {
    reported(
        crate::background_run::started_in_background(name),
        SourceTrust::Trusted,
    )
    .handed_to_background()
}

impl aidaemon_tools::semantics::ToolSemantics for SpawnAgentTool {
    fn dispatch_facts(&self) -> aidaemon_tools::semantics::ToolDispatchFacts {
        aidaemon_tools::semantics::ToolDispatchFacts {
            task_lead_delegation: true,
            spawns_child_agents: true,
            // Each child runs under its own deadline, which pauses while the
            // owner is asked and ends in a salvaged work report. The generic
            // watchdog would drop that report, and the results of children
            // that already finished, the moment it fired.
            manages_own_timeout: true,
            goal_context: aidaemon_tools::semantics::GoalContext::Goal,
            ..Default::default()
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: true,
        }
    }

    /// A spawn whose child is held to read-only access changes nothing: the
    /// child's authority is narrowed to it (never wider than requested), so
    /// the call is read-only, and a read-only agent may delegate read-only
    /// work. Any other spawn keeps the tool's worst case.
    fn call_annotations_in(
        &self,
        arguments: &str,
        _roots: &aidaemon_tools::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        let read_only = |arguments: &str| {
            serde_json::from_str::<SpawnArgs>(arguments)
                .is_ok_and(|args| matches!(args.access, Some(ChildAccess::ReadOnly)))
        };
        let read_only_child = match batch_tasks(arguments) {
            Ok(Some(tasks)) => tasks.iter().all(|task| read_only(&task.to_string())),
            Ok(None) => read_only(arguments),
            Err(_) => false,
        };
        if read_only_child {
            return aidaemon_core::ToolAnnotations::READ_ONLY;
        }
        aidaemon_tools::semantics::default_call_annotations(
            self.capabilities(),
            &self.call_semantics(arguments),
        )
    }

    /// A child's result carries the child's taint; anything the call does
    /// not report (errors) counts as untrusted.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &aidaemon_tools::semantics::TrustRoots<'_>,
    ) -> SourceTrust {
        SourceTrust::Untrusted
    }
}

/// A child's run phase as its spawn receipt's outcome: only a child that
/// succeeded makes the spawn a success, never its reply's wording.
pub fn child_receipt_status(phase: RunPhase) -> aidaemon_tools::semantics::ToolOutcomeStatus {
    match phase {
        RunPhase::Succeeded => aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded,
        RunPhase::Failed => aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent,
        RunPhase::Stopped | RunPhase::Running => {
            aidaemon_tools::semantics::ToolOutcomeStatus::CompletedWithNegativeResult
        }
    }
}

/// A spawn result with the trust of its content stamped for the dispatcher.
pub fn reported(output: String, trust: SourceTrust) -> ToolCallOutcome {
    let mut outcome = ToolCallOutcome::from_output(output);
    outcome.metadata.output_trust = Some(trust);
    outcome
}

impl SpawnAgentTool {
    /// Start every child of a batch call, at most [`MAX_CONCURRENT_CHILDREN`]
    /// at a time, each exactly as its own single spawn would run.
    pub async fn execute_batch(
        &self,
        tasks: Vec<Value>,
        facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        use futures::StreamExt as _;
        let background = tasks
            .first()
            .and_then(|task| task.get("background"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let runs = futures::stream::iter(tasks.into_iter().map(|task| async move {
            let mission = ["title", "mission"]
                .iter()
                .filter_map(|key| task.get(key).and_then(Value::as_str))
                .map(str::trim)
                .find(|name| !name.is_empty())
                .unwrap_or_default()
                .to_string();
            (mission, self.execute(&task.to_string(), facts).await)
        }))
        .buffered(MAX_CONCURRENT_CHILDREN)
        .collect::<Vec<_>>()
        .await;
        combine_batch(runs, background)
    }

    /// Run the sub-agent synchronously (blocking until completion or timeout).
    #[allow(clippy::too_many_arguments)]
    pub async fn run_sync(
        &self,
        agent: Arc<dyn ChildAgentRuntime>,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        child_role: Option<AgentRole>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
        durable_task_id: Option<&str>,
        project_scope: Option<&str>,
        arg_specialist: Option<&str>,
        approval_session_id: Option<&str>,
        parent_authority: Option<crate::authority::InheritedAuthority>,
        model_tier: Option<aidaemon_providers::routing::Tier>,
        tree: Option<&crate::authority::SpawnFamily>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let started = chrono::Utc::now();
        // The name its timing line gives it: the title the owner sees on
        // the card, else its mission.
        let helper_name = crate::spawn::child_status_label(
            parent_authority
                .as_ref()
                .and_then(|parent| parent.card_title.as_deref())
                .unwrap_or(mission),
        );
        // The child's own wait clock: its card waits, apart from its
        // siblings', still counting for the turn.
        let waits = aidaemon_channels::owner_wait::OwnerWaitClock::current()
            .unwrap_or_default()
            .nested();
        let card_tx = status_tx.clone();
        let result = aidaemon_channels::owner_wait::OwnerWaitClock::within(
            Some(waits.clone()),
            agent.run_child(ChildAgentRequest {
                mission: mission.to_string(),
                task: task.to_string(),
                status_tx,
                channel_ctx,
                user_role,
                child_role,
                goal_id: goal_id.map(str::to_string),
                task_id: task_id.map(str::to_string),
                project_scope: project_scope.map(str::to_string),
                specialist: arg_specialist.map(str::to_string),
                approval_session_id: approval_session_id.map(str::to_string),
                parent_authority,
                time_limit: self.time_limit(),
                model_tier,
            }),
        )
        .await;
        let measured = MeasuredRun::ended_now(started, &waits);
        let failed = match &result {
            Ok(run) => run.phase == RunPhase::Failed,
            Err(error) => error.downcast_ref::<ChildTimedOut>().is_none(),
        };
        let window = if failed {
            failed_run_window(&helper_name, &measured)
        } else {
            run_window(&helper_name, &measured)
        };
        // Its header on the turn's card ends with how it ended and the time
        // the runtime measured: the owner reads the time there, never as
        // the parent's arithmetic.
        if let Some(tx) = card_tx {
            let phase = match &result {
                Ok(run) => run.phase,
                Err(error) if error.downcast_ref::<ChildTimedOut>().is_some() => RunPhase::Stopped,
                Err(_) => RunPhase::Failed,
            };
            let _ = tx.try_send(StatusUpdate::Helper(aidaemon_core::card::HelperView {
                label: helper_name.clone(),
                phase,
                background: false,
                step: None,
                took: Some(took_line(measured.span(), measured.waited)),
            }));
        }

        match result {
            Ok(run) => {
                // How it ended is its run record's phase, decided from typed
                // facts; its reply is only the body.
                let response = match run.phase {
                    RunPhase::Succeeded => run.response,
                    RunPhase::Stopped | RunPhase::Running => {
                        format!("\u{26a0}\u{fe0f} Sub-agent incomplete\n\n{}", run.response)
                    }
                    RunPhase::Failed => {
                        let reason = run
                            .evidence
                            .failure
                            .as_ref()
                            .map(|failure| format!("{}\n\n", failure.reason()))
                            .unwrap_or_default();
                        return Err(anyhow::anyhow!(
                            "Sub-agent failed. {reason}Its own report: {}\n{window}",
                            run.response
                        ));
                    }
                };
                let (shown, truncation) =
                    bounded_child_response(&response, self.max_response_chars);
                // The measured run leads, so a cut of a long result never
                // loses it: whatever the parent says of the helper's time
                // quotes it.
                let mut outcome = reported(
                    format!(
                        "{window}\n\n{shown}{}",
                        coordination_trailer(tree, &run.session)
                    ),
                    run.trust,
                );
                outcome.metadata.truncation = truncation;
                // The child's run already closed with a typed outcome: that,
                // not a scan of its reply's wording (a review listing "Bugs"
                // read as a failure), is the receipt's outcome.
                outcome.metadata.outcome_status = Some(child_receipt_status(run.phase));
                Ok(outcome)
            }
            Err(error) => match error.downcast_ref::<ChildTimedOut>() {
                None => Err(anyhow::anyhow!("Specialist failed: {error}\n{window}")),
                Some(timed_out) => {
                    if let Some(task_id) = durable_task_id {
                        // The child may have persisted a terminal outcome (e.g.
                        // report_blocker) before it was stopped — use that
                        // instead of discarding the work.
                        if let Some(salvaged) = agent
                            .salvage_executor_task_outcome(task_id, self.timeout_secs)
                            .await
                        {
                            let (prefix, outcome) = match salvaged.status.as_str() {
                                "completed" => {
                                    ("\u{2705} Sub-agent complete", TaskOutcome::Succeeded)
                                }
                                "blocked" => {
                                    ("\u{26a0}\u{fe0f} Sub-agent blocked", TaskOutcome::Partial)
                                }
                                _ => ("\u{274c} Sub-agent failed", TaskOutcome::Failed),
                            };
                            // The child wrote these details; its taint is
                            // unknown after a timeout.
                            let mut result = reported(
                                format!("{prefix}\n\n{}\n\n{window}", salvaged.details),
                                SourceTrust::Untrusted,
                            );
                            result.metadata.outcome_status =
                                Some(child_receipt_status(crate::run_state::phase_of(outcome)));
                            return Ok(result);
                        }
                        if child_role == Some(AgentRole::Executor) {
                            agent
                                .mark_executor_task_timeout(task_id, self.timeout_secs)
                                .await;
                        }
                    }
                    if timed_out.report.may_have_changed_something() {
                        // Not a failure the loop may retry on its own: the
                        // parent has to look at what already happened. The
                        // action details come from the child's run.
                        return Ok(reported(
                            format!(
                                "\u{26a0}\u{fe0f} Sub-agent incomplete\n{window}\n\n{}",
                                timed_out.render()
                            ),
                            SourceTrust::Untrusted,
                        ));
                    }
                    Err(anyhow::anyhow!("{timed_out}\n{window}"))
                }
            },
        }
    }
}
