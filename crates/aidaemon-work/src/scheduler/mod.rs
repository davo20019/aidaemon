//! The scheduler's goal cycle: due mandate reviews, scheduled goals and
//! reminders, stuck and orphaned tasks, recovery runs and goal
//! notifications. One pass is [`GoalCycle::tick`]; [`GoalCycleService`] runs
//! it on an [`aidaemon_core::ServiceHost`].
//!
//! The host gives what is its own: the agent runtime that runs a task lead
//! ([`TaskLeadHost`]), the channels that carry a notice to the owner
//! ([`NoticeDelivery`]) and how text is shown to the owner ([`WorkPresenter`]).

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use aidaemon_core::{Schedule, Service, ServiceContext, ServiceError};
use async_trait::async_trait;
use chrono::Utc;
use tracing::{error, info, warn};

use crate::goal_tokens::GoalTokenRegistry;
use crate::traits::{GoalSchedule, Mandate, MandateStatus};

mod mandates;
mod notifications;
mod schedules;
mod tasks;

/// The stores the goal cycle reads and writes.
pub trait SchedulerStore:
    crate::traits::GoalStore
    + crate::traits::TaskStore
    + crate::traits::GoalScheduleStore
    + crate::traits::GoalBudgetStore
    + crate::traits::ScheduledRunStore
    + crate::traits::TaskDispatchStore
    + crate::traits::GoalNotificationStore
    + crate::traits::NotificationStore
    + crate::traits::MandateStore
    + crate::traits::WorkCoordinationStore
    + aidaemon_tools::services::SettingsStore
{
}

impl<T> SchedulerStore for T where
    T: crate::traits::GoalStore
        + crate::traits::TaskStore
        + crate::traits::GoalScheduleStore
        + crate::traits::GoalBudgetStore
        + crate::traits::ScheduledRunStore
        + crate::traits::TaskDispatchStore
        + crate::traits::GoalNotificationStore
        + crate::traits::NotificationStore
        + crate::traits::MandateStore
        + crate::traits::WorkCoordinationStore
        + aidaemon_tools::services::SettingsStore
{
}

/// The agent runtime, as the goal cycle hands it work.
#[async_trait]
pub trait TaskLeadHost: Send + Sync {
    /// Whether the runtime is still there to take work.
    fn available(&self) -> bool;

    /// Start a background task lead for `goal` on its root task `task_id`,
    /// already claimed for it, as the owner on an internal channel.
    /// `running_surface_id` is a message the lead may edit with progress.
    fn spawn_task_lead(
        &self,
        goal: crate::traits::Goal,
        task_text: String,
        task_id: String,
        running_surface_id: Option<String>,
    );

    /// When some configured model can answer again, if none can now.
    fn next_model_answer_at(&self) -> Option<SystemTime>;

    /// Re-run requests deferred by a provider outage; how many ran.
    async fn retry_deferred_provider_requests(&self) -> usize;

    /// Record a tainted turn in `session_id` for an untrusted notice; its id.
    async fn tainted_note_turn(&self, session_id: &str) -> anyhow::Result<String>;

    /// Keep a delivered notice in the owner's history, in `turn_id` if given.
    async fn record_parent_visible_result_note_in_turn(
        &self,
        session_id: &str,
        prefix: &str,
        text: &str,
        turn_id: Option<&str>,
    ) -> anyhow::Result<()>;

    /// Bind the owner's next reply in `session_id` to a mandate question.
    async fn record_mandate_owner_input_context(
        &self,
        session_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        action_token: &str,
    ) -> anyhow::Result<()>;
}

/// The channels a notice reaches the owner through.
#[async_trait]
pub trait NoticeDelivery: Send + Sync {
    /// Send `text` to `session_id`.
    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()>;

    /// Send `text` and return an id the sender can later edit, if the
    /// channel has one.
    async fn send_text_tracked(
        &self,
        session_id: &str,
        text: &str,
    ) -> anyhow::Result<Option<String>>;
}

/// How the host shows work to the owner.
pub trait WorkPresenter: Send + Sync {
    /// A short label for a goal's description.
    fn goal_label(&self, description: &str) -> String;

    /// `text` made safe to show the owner.
    fn user_facing(&self, text: &str) -> String;

    /// A notification of kind `notification_type` as the channel shows it.
    fn notification(&self, notification_type: &str, text: &str) -> String;

    /// The summary of a goal's task results sent when it ends.
    fn goal_task_results(
        &self,
        tasks: &[crate::traits::Task],
        root_task_id: Option<&str>,
        fallback: &str,
    ) -> String;
}

pub const TASK_ESCALATION_SETTLE_SECS: i64 = 8;
/// History prefix for notices delivered from the notification queue; matches
/// the direct goal-notification path so both read the same to the agent.
pub const QUEUED_NOTIFICATION_NOTE_PREFIX: &str = "Parent-visible result (goal notification)";

/// Notices after which the owner's next reply may answer a mandate question:
/// the ASK itself, and the timeout notice that says it can still be answered.
pub fn notice_accepts_mandate_answer(notification_type: &str) -> bool {
    matches!(notification_type, "mandate_ask" | "mandate_ask_timeout")
}
pub const MANDATE_REVIEW_LEASE_SECS: i64 = 30 * 60;
pub const MANDATE_REVIEW_BATCH_SIZE: i64 = 20;
pub const MANDATE_DISPATCH_BUSY_RETRIES: u32 = 5;
pub const MANDATE_DISPATCH_BUSY_RETRY_SECS: i64 = 30;

pub fn mandate_retry_at(mandate: &Mandate, requested_secs: Option<i64>) -> String {
    let delay = mandate.clamp_review_secs(requested_secs);
    (chrono::Utc::now() + chrono::Duration::seconds(delay)).to_rfc3339()
}

pub fn mandate_budget_retry_at(mandate: &Mandate, now: chrono::DateTime<Utc>) -> String {
    let next_day = now
        .date_naive()
        .succ_opt()
        .expect("the next UTC date is representable")
        .and_hms_opt(0, 0, 0)
        .expect("UTC midnight is representable");
    let mut retry_at = chrono::DateTime::<Utc>::from_naive_utc_and_offset(next_day, Utc);
    if let Some(expiry) = mandate
        .expires_at
        .as_deref()
        .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
    {
        retry_at = retry_at.min(expiry);
    }
    retry_at.to_rfc3339()
}

pub fn mandate_transient_retry_at(mandate: &Mandate, now: chrono::DateTime<Utc>) -> String {
    let mut retry_at = now + chrono::Duration::seconds(MANDATE_DISPATCH_BUSY_RETRY_SECS);
    if let Some(expiry) = mandate
        .expires_at
        .as_deref()
        .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
    {
        retry_at = retry_at.min(expiry);
    }
    retry_at.to_rfc3339()
}

pub fn sqlite_busy_code(value: &str) -> bool {
    value
        .parse::<i32>()
        .ok()
        .is_some_and(|code| code & 0xff == 5)
}

pub fn is_sqlite_busy_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if let Some(sqlx::Error::Database(database)) = cause.downcast_ref::<sqlx::Error>()
            && database.code().is_some_and(|code| sqlite_busy_code(&code)) {
                return true;
            }
        let message = cause.to_string().to_ascii_lowercase();
        message.contains("database is locked")
            || message.contains("database table is locked")
            || message.contains("sqlite_busy")
            || message.split("code:").skip(1).any(|suffix| {
                let code = suffix
                    .trim_start()
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>();
                sqlite_busy_code(&code)
            })
    })
}

pub fn mandate_review_task_description(mandate: &Mandate, goal_run_id: &str) -> String {
    // Owner text and authority targets intentionally do not enter ordinary
    // task prose. The isolated mandate system prompt loads the exact current
    // policy from durable state and labels its JSON fields as policy or data.
    format!(
        "Run one bounded autonomous review for mandate {} version {} in goal run {}. \
         Use only the isolated built-in mandate protocol and immutable policy supplied \
         by the runtime.",
        mandate.id, mandate.version, goal_run_id
    )
}

pub fn mandate_review_task_context(mandate: &Mandate) -> String {
    serde_json::json!({
        "mandate_id": mandate.id,
        "mandate_version": mandate.version,
        "provenance": "runtime_mandate_fence_only",
    })
    .to_string()
}

pub fn timestamp_is_at_or_after(value: &str, lower_bound: &str) -> bool {
    match (
        chrono::DateTime::parse_from_rfc3339(value),
        chrono::DateTime::parse_from_rfc3339(lower_bound),
    ) {
        (Ok(value), Ok(lower_bound)) => value >= lower_bound,
        _ => value >= lower_bound,
    }
}

/// Whether a goal's daily token budget is exhausted *for today*. Only usage
/// recorded on the current UTC day counts: stale `tokens_used_today` from a
/// previous day (before the daily reset has run) must NOT block a goal, or an
/// expensive run can deadlock the goal across the day boundary (the budget gate
/// keeps deferring it, so it never fires, so it never resets).
/// Header for the goal-completion notification, derived from the ACTUAL task
/// rows rather than metadata the task lead may never have written (a
/// watchdog-killed lead leaves `partial_success` unset). "Goal completed:"
/// must never accompany incomplete tasks — name what's missing instead.
pub fn goal_completion_header(tasks: &[crate::traits::Task]) -> String {
    if tasks.is_empty() {
        return "Goal completed:".to_string();
    }
    let total = tasks.len();
    let done = tasks.iter().filter(|t| t.status == "completed").count();
    if done == total {
        return "Goal completed:".to_string();
    }
    let missing: Vec<String> = tasks
        .iter()
        .filter(|t| t.status != "completed")
        .map(|t| {
            format!(
                "• {} ({})",
                t.description.chars().take(80).collect::<String>(),
                t.status
            )
        })
        .collect();
    format!(
        "Goal partially completed ({done}/{total} tasks). Not completed:\n{}\n",
        missing.join("\n")
    )
}

pub fn daily_budget_exhausted(
    budget_daily: Option<i64>,
    tokens_used_today: i64,
    tokens_used_day: &str,
    today: &str,
) -> bool {
    match budget_daily {
        Some(budget) => tokens_used_day == today && tokens_used_today >= budget,
        None => false,
    }
}

pub fn daily_budget_has_run_capacity(
    budget_daily: Option<i64>,
    budget_per_check: Option<i64>,
    tokens_used_today: i64,
    tokens_used_day: &str,
    today: &str,
) -> bool {
    let (Some(daily), Some(per_run)) = (budget_daily, budget_per_check) else {
        return true;
    };
    if tokens_used_day != today {
        return true;
    }
    daily.saturating_sub(tokens_used_today) >= per_run.max(0)
}

/// Seconds since a task's most recent activity (or its `started_at` when it has
/// no activity rows). Used to tag stuck-task interrupts so the inactivity
/// threshold can be tuned from data. Returns 0 if no timestamp parses (never
/// panics). Both inputs are RFC3339 or SQLite-datetime strings.
/// Parses a timestamp that may be either RFC3339 (`2026-07-04T13:10:45+00:00`,
/// how application code writes it) or SQLite's own `datetime()`-coerced form
/// (`2026-07-04 13:10:45`, space-separated, no offset, second precision —
/// what columns wrapped in `datetime(?)` at INSERT time read back as, e.g.
/// `task_activity.created_at`). Returns `None` for anything else rather than
/// panicking or guessing.
pub fn parse_datetime_flexible(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&chrono::Utc))
        .ok()
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                .ok()
                .map(|nd| nd.and_utc())
        })
}

pub fn task_escalation_has_settled(created_at: &str, now: chrono::DateTime<chrono::Utc>) -> bool {
    parse_datetime_flexible(created_at)
        .is_none_or(|created_at| (now - created_at).num_seconds() >= TASK_ESCALATION_SETTLE_SECS)
}

pub fn task_inactivity_secs(
    last_activity: Option<&str>,
    started_at: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> i64 {
    let reference = last_activity
        .and_then(parse_datetime_flexible)
        .or_else(|| parse_datetime_flexible(started_at));
    match reference {
        Some(ts) => (now - ts).num_seconds().max(0),
        None => 0,
    }
}

pub fn task_is_blocked_by_terminal_dependency(task: &crate::traits::Task) -> bool {
    task.status == "blocked"
        && task.blocker.as_deref().is_some_and(|blocker| {
            blocker.starts_with("Dependency ") && blocker.contains(" ended with status ")
        })
}

pub fn task_blocks_schedule_fire(task: &crate::traits::Task) -> bool {
    matches!(task.status.as_str(), "pending" | "claimed" | "running")
        || (task.status == "blocked" && !task_is_blocked_by_terminal_dependency(task))
}

pub fn task_blocks_later_schedule_fire(
    run: &crate::traits::GoalRun,
    task: &crate::traits::Task,
) -> bool {
    // Backpressure belongs to one active occurrence. Historical task rows are
    // audit records after their parent run reaches a terminal state and can no
    // longer suppress later schedule occurrences.
    if !matches!(run.status.as_str(), "pending" | "running" | "blocked") {
        return false;
    }
    // A blocked run has no active worker and terminates that occurrence,
    // whatever triggered it (scheduled, manual, or recovery). Keeping it
    // resumable is useful for explicit recovery, but it must not suppress
    // every later occurrence of a recurring schedule. Blockers inside a
    // still-running occurrence remain open.
    run.status != "blocked" && task_blocks_schedule_fire(task)
}

pub fn task_is_terminal_schedule_failure(task: &crate::traits::Task) -> bool {
    matches!(
        task.status.as_str(),
        "failed" | "interrupted" | "cancelled" | "blocked"
    )
}

/// Reconcile an open scheduled occurrence from its authoritative task graph.
/// A blocked child is a terminal failure of this occurrence, not a second
/// meaning for the run's `blocked` waiting state. Autonomous recovery is a
/// later linked run rather than an indefinitely open scheduled occurrence.
pub fn scheduled_run_reconciliation_status(
    run: &crate::traits::GoalRun,
    tasks: &[crate::traits::Task],
) -> Option<&'static str> {
    if !matches!(
        run.trigger_type.as_str(),
        "scheduled" | "manual" | "recovery"
    ) || !matches!(run.status.as_str(), "pending" | "running")
        || tasks.is_empty()
    {
        return None;
    }
    // A blocked child terminates a scheduled occurrence. In a manual or
    // recovery occurrence the lead may still be waiting on an owner decision,
    // so only hard task failures reconcile it; the run-level `blocked` state
    // (the lead already ended the occurrence) is handled at fire time.
    let terminal_failure = tasks.iter().any(|task| {
        if run.trigger_type == "scheduled" {
            task_is_terminal_schedule_failure(task)
        } else {
            matches!(task.status.as_str(), "failed" | "interrupted" | "cancelled")
        }
    });
    if terminal_failure {
        Some("failed")
    } else if tasks
        .iter()
        .all(crate::traits::Task::satisfies_run_completion)
    {
        Some("completed")
    } else {
        None
    }
}

pub fn stranded_manual_run_matches_pending_tasks(
    run: &crate::traits::GoalRun,
    run_tasks: &[crate::traits::Task],
    pending_tasks: &[&crate::traits::Task],
) -> bool {
    run.trigger_type == "manual"
        && matches!(run.status.as_str(), "pending" | "running")
        && pending_tasks
            .iter()
            .any(|pending| run_tasks.iter().any(|task| task.id == pending.id))
}

/// The goal cycle and what it needs between passes.
pub struct GoalCycle {
    pub(crate) state: Arc<dyn SchedulerStore>,
    pub(crate) agent: Option<Arc<dyn TaskLeadHost>>,
    pub(crate) hub: Option<Arc<dyn NoticeDelivery>>,
    pub(crate) presenter: Arc<dyn WorkPresenter>,
    pub(crate) goal_token_registry: Option<GoalTokenRegistry>,
    deferred_retry_running: Arc<AtomicBool>,
    last_stale_goal_cleanup: Option<Instant>,
    /// Seconds of no task activity before `detect_stuck_tasks` interrupts a
    /// running/claimed task. Defaults to 300.
    pub task_inactivity_timeout_secs: i64,
}

impl GoalCycle {
    /// Wait between an escalation (or the previous automatic attempt) and the
    /// next automatic recovery run. Long enough for a transient external
    /// cause to clear; short enough that a daily objective is not lost for a
    /// week.
    const ESCALATED_RECOVERY_COOLDOWN_SECS: i64 = 6 * 3600;
    /// Automatic recovery attempts per escalation before the objective waits
    /// for the owner.
    const ESCALATED_RECOVERY_MAX_ATTEMPTS: u16 = 3;

    pub fn new(
        state: Arc<dyn SchedulerStore>,
        presenter: Arc<dyn WorkPresenter>,
        goal_token_registry: Option<GoalTokenRegistry>,
    ) -> Self {
        Self {
            state,
            agent: None,
            hub: None,
            presenter,
            goal_token_registry,
            deferred_retry_running: Arc::new(AtomicBool::new(false)),
            last_stale_goal_cleanup: None,
            task_inactivity_timeout_secs: 300,
        }
    }

    /// Override the stuck-task inactivity timeout.
    pub fn set_task_inactivity_timeout(&mut self, secs: u64) {
        self.task_inactivity_timeout_secs = secs as i64;
    }

    /// The channels notices go out through.
    pub fn set_delivery(&mut self, delivery: Option<Arc<dyn NoticeDelivery>>) {
        self.hub = delivery;
    }

    /// The agent runtime that runs task leads.
    pub fn set_agent(&mut self, agent: Option<Arc<dyn TaskLeadHost>>) {
        self.agent = agent;
    }

    /// The runtime, while it can take work.
    fn live_agent(&self) -> Option<Arc<dyn TaskLeadHost>> {
        self.agent.clone().filter(|agent| agent.available())
    }

    /// One-time recovery after startup: mark interrupted tasks, rebuild token registry.
    pub async fn startup_recovery(&self) {
        info!("Running startup recovery");

        match self.state.recover_expired_task_attempts().await {
            Ok(tasks) if !tasks.is_empty() => {
                info!(
                    count = tasks.len(),
                    "Startup recovery processed expired execution leases"
                );
            }
            Ok(_) => {}
            Err(error) => {
                error!(%error, "Failed to recover expired execution leases");
            }
        }

        // Mark any tasks stuck in running/claimed as interrupted,
        // then auto-retry idempotent ones ONLY if their parent goal is still active.
        // We do NOT aggressively fail goals here — the progress-based circuit breaker
        // in spawn_background_task_lead handles stale goals on next dispatch.
        match self.state.get_stuck_tasks(0).await {
            Ok(tasks) => {
                let mut interrupted = 0;
                let mut auto_retried = 0;
                for task in &tasks {
                    if let Err(e) = self.state.mark_task_interrupted(&task.id).await {
                        error!(task_id = %task.id, error = %e, "Failed to mark task interrupted");
                        continue;
                    }
                    interrupted += 1;

                    // Check if the parent goal is still active/non-stalled before retrying
                    let goal_active = match self.state.get_goal(&task.goal_id).await {
                        Ok(Some(g)) => g.status == "active",
                        _ => false,
                    };

                    if !goal_active {
                        // Goal is failed/completed/stalled — don't retry, just leave interrupted.
                        continue;
                    }

                    // A recurring goal remains active between fires, so goal
                    // status alone is not enough to authorize recovery. Only
                    // retry work that belongs to its explicitly persisted
                    // active run. Otherwise a restart can resurrect a child
                    // from a closed/cancelled cycle and repeat side effects.
                    let schedules = self
                        .state
                        .get_schedules_for_goal(&task.goal_id)
                        .await
                        .unwrap_or_default();
                    if !schedules.is_empty() {
                        let active_run = self
                            .state
                            .get_scheduled_run_state(&task.goal_id)
                            .await
                            .ok()
                            .flatten();
                        if active_run.as_ref().is_none_or(|run| {
                            !timestamp_is_at_or_after(&task.created_at, &run.created_at)
                        }) {
                            continue;
                        }
                    }

                    // Auto-retry idempotent tasks that haven't exceeded max retries
                    if task.idempotent && task.retry_count < task.max_retries {
                        let mut retry_task = task.clone();
                        retry_task.status = "pending".to_string();
                        retry_task.retry_count += 1;
                        retry_task.result = None;
                        retry_task.error = None;
                        retry_task.blocker = None;
                        retry_task.agent_id = None;
                        retry_task.started_at = None;
                        retry_task.completed_at = None;
                        if let Err(e) = self.state.update_task(&retry_task).await {
                            error!(task_id = %task.id, error = %e, "Failed to auto-retry task");
                        } else {
                            auto_retried += 1;
                            info!(task_id = %task.id, retry = retry_task.retry_count, "Auto-retried idempotent task");
                        }
                    }
                    // Non-retryable tasks stay interrupted — the circuit breaker will
                    // detect no progress on next dispatch and eventually stall the goal.
                }

                if interrupted > 0 {
                    info!(
                        count = interrupted,
                        auto_retried, "Startup recovery: marked interrupted tasks"
                    );
                }
            }
            Err(e) => {
                error!(error = %e, "Failed to get stuck tasks during recovery");
            }
        }

        // Mark stale active goals as abandoned/failed.
        // Finite goals stuck active for >2 hours are clearly orphaned.
        match self.state.cleanup_stale_goals(2).await {
            Ok(count) if count > 0 => {
                info!(count, "Startup recovery: cleaned up stale goals");
            }
            Err(e) => {
                error!(error = %e, "Failed to cleanup stale goals during recovery");
            }
            _ => {}
        }

        // Rebuild goal token registry from active goals
        if let Some(ref registry) = self.goal_token_registry {
            match self.state.get_active_goals().await {
                Ok(goals) => {
                    registry.rebuild_from_goals(&goals).await;
                    info!(
                        count = goals.len(),
                        "Rebuilt goal token registry from active goals"
                    );
                }
                Err(e) => {
                    error!(error = %e, "Failed to rebuild goal token registry");
                }
            }
        }

        info!("Startup recovery complete");
    }

    /// One pass of the goal cycle. Periodic jobs and the database check are
    /// the host's.
    pub async fn tick(&mut self) -> anyhow::Result<()> {
        let now = Instant::now();

        // Phase 1d: Pending tasks whose dependencies reached terminal failure
        // can never dispatch. Mark them blocked before schedule coalescing so
        // stale dependent work does not suppress future scheduled runs forever.
        self.block_pending_tasks_with_terminal_failed_dependencies()
            .await;

        // Phase 2a: Review due owner mandates. Mandates have their own durable
        // wake clock and lease; they must never flow through goal_schedules or
        // acquire scheduled-run provenance.
        self.check_due_mandates().await;

        // Phase 2b: Reconcile scheduled lifecycle state independently of the
        // next fire time. A completed/failed task graph must not remain
        // observably "running" until tomorrow's recurrence happens to wake it.
        self.reconcile_open_scheduled_runs().await;

        // Phase 2b': Escalated scheduled objectives get a bounded, cooled-down
        // automatic recovery run. Escalation pauses the cron so it cannot
        // keep failing; without this phase nothing would ever run again.
        self.launch_escalated_recoveries().await;

        // Phase 2c: Fire due schedules (recurring + one-shot)
        self.check_due_goal_schedules().await;

        // Phase 3: Detect stuck tasks
        self.detect_stuck_tasks().await;

        // Phase 3b: Cleanup stale pending confirmations (1 hour timeout)
        match self
            .state
            .cancel_stale_pending_confirmation_goals(3600)
            .await
        {
            Ok(count) if count > 0 => {
                info!(count, "Cancelled stale pending_confirmation goals");
            }
            Err(e) => {
                error!(error = %e, "Failed to cancel stale pending_confirmation goals");
            }
            _ => {}
        }

        // Phase 3c: Periodically cleanup stale goals (every 30 minutes)
        let should_cleanup_goals = match self.last_stale_goal_cleanup {
            None => true,
            Some(last) => now.duration_since(last) >= Duration::from_secs(1800),
        };
        if should_cleanup_goals {
            self.last_stale_goal_cleanup = Some(now);
            match self.state.cleanup_stale_goals(2).await {
                Ok(count) if count > 0 => {
                    info!(count, "Periodic cleanup: marked stale goals");
                }
                Err(e) => {
                    error!(error = %e, "Failed to cleanup stale goals");
                }
                _ => {}
            }
        }

        // Phase 3d: Auto-retry failed idempotent tasks
        self.auto_retry_failed_tasks().await;

        // Phase 4: Dispatch orphaned pending tasks
        self.dispatch_pending_tasks().await;

        // Phase 5: Deliver notifications for completed/failed goals
        self.deliver_notifications().await;

        // Phase 6: Re-run user requests that failed only because the model
        // provider was unavailable before any tool work committed.
        // Each is a whole agent turn, so they run beside the cycle: schedules,
        // mandates and stuck-task checks must not wait for them.
        if let Some(agent) = self.live_agent()
            && !self.deferred_retry_running.swap(true, Ordering::AcqRel) {
                let running = self.deferred_retry_running.clone();
                tokio::spawn(async move {
                    let dispatched = agent.retry_deferred_provider_requests().await;
                    running.store(false, Ordering::Release);
                    if dispatched > 0 {
                        info!(dispatched, "Re-ran requests deferred by a provider outage");
                    }
                });
            }

        Ok(())
    }
}

/// The goal cycle as one service: its phases depend on their order.
pub struct GoalCycleService {
    cycle: tokio::sync::Mutex<GoalCycle>,
}

impl GoalCycleService {
    pub fn new(cycle: GoalCycle) -> Self {
        Self {
            cycle: tokio::sync::Mutex::new(cycle),
        }
    }
}

#[async_trait]
impl Service for GoalCycleService {
    fn name(&self) -> &str {
        "goal_cycle"
    }

    fn schedule(&self) -> Schedule {
        Schedule::EveryTick
    }

    async fn start(&self, _ctx: &ServiceContext) -> Result<(), ServiceError> {
        self.cycle.lock().await.startup_recovery().await;
        Ok(())
    }

    async fn run(&self, _ctx: &ServiceContext) -> Result<(), ServiceError> {
        self.cycle
            .lock()
            .await
            .tick()
            .await
            .map_err(|e| ServiceError::new(e.to_string()))
    }
}
