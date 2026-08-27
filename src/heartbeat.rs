use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use chrono::Utc;
use futures::FutureExt;
use serde::Serialize;
use tokio::sync::{mpsc, Semaphore};
use tracing::{error, info, warn};

use crate::agent::{build_goal_task_results_summary, is_group_session, Agent};
use crate::goal_tokens::GoalTokenRegistry;
use crate::runtime_ports::OutboundRouter;
use crate::traits::{GoalSchedule, Mandate, MandateStatus, StateStore};
use crate::types::{ChannelContext, UserRole};

const TASK_ESCALATION_SETTLE_SECS: i64 = 8;
const MANDATE_REVIEW_LEASE_SECS: i64 = 30 * 60;
const MANDATE_REVIEW_BATCH_SIZE: i64 = 20;
const MANDATE_DISPATCH_BUSY_RETRIES: u32 = 5;
const MANDATE_DISPATCH_BUSY_RETRY_SECS: i64 = 30;

fn mandate_retry_at(mandate: &Mandate, requested_secs: Option<i64>) -> String {
    let delay = mandate.clamp_review_secs(requested_secs);
    (chrono::Utc::now() + chrono::Duration::seconds(delay)).to_rfc3339()
}

fn mandate_budget_retry_at(mandate: &Mandate, now: chrono::DateTime<Utc>) -> String {
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

fn mandate_transient_retry_at(mandate: &Mandate, now: chrono::DateTime<Utc>) -> String {
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

fn sqlite_busy_code(value: &str) -> bool {
    value
        .parse::<i32>()
        .ok()
        .is_some_and(|code| code & 0xff == 5)
}

fn is_sqlite_busy_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if let Some(sqlx::Error::Database(database)) = cause.downcast_ref::<sqlx::Error>() {
            if database.code().is_some_and(|code| sqlite_busy_code(&code)) {
                return true;
            }
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

fn mandate_review_task_description(mandate: &Mandate, goal_run_id: &str) -> String {
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

fn mandate_review_task_context(mandate: &Mandate) -> String {
    serde_json::json!({
        "mandate_id": mandate.id,
        "mandate_version": mandate.version,
        "provenance": "runtime_mandate_fence_only",
    })
    .to_string()
}

fn timestamp_is_at_or_after(value: &str, lower_bound: &str) -> bool {
    match (
        chrono::DateTime::parse_from_rfc3339(value),
        chrono::DateTime::parse_from_rfc3339(lower_bound),
    ) {
        (Ok(value), Ok(lower_bound)) => value >= lower_bound,
        _ => value >= lower_bound,
    }
}

/// Best-effort check that a logged `http_request` tool-call's args used a
/// mutating HTTP method (anything but GET/HEAD/OPTIONS). `tool_args` is a
/// (possibly truncated) JSON string; returns `false` — i.e. "not mutating" —
/// if it can't be parsed or has no `method` field, so a truncated/odd
/// payload never blocks dispatch on a parse miss.
fn is_mutating_http_method(tool_args_json: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(tool_args_json) else {
        return false;
    };
    let Some(method) = value.get("method").and_then(|m| m.as_str()) else {
        return false;
    };
    !matches!(
        method.to_ascii_uppercase().as_str(),
        "GET" | "HEAD" | "OPTIONS"
    )
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
fn goal_completion_header(tasks: &[crate::traits::Task]) -> String {
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

fn daily_budget_exhausted(
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

fn daily_budget_has_run_capacity(
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
fn parse_datetime_flexible(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&chrono::Utc))
        .ok()
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                .ok()
                .map(|nd| nd.and_utc())
        })
}

fn task_escalation_has_settled(created_at: &str, now: chrono::DateTime<chrono::Utc>) -> bool {
    parse_datetime_flexible(created_at)
        .is_none_or(|created_at| (now - created_at).num_seconds() >= TASK_ESCALATION_SETTLE_SECS)
}

fn task_inactivity_secs(
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

fn task_is_blocked_by_terminal_dependency(task: &crate::traits::Task) -> bool {
    task.status == "blocked"
        && task.blocker.as_deref().is_some_and(|blocker| {
            blocker.starts_with("Dependency ") && blocker.contains(" ended with status ")
        })
}

fn task_blocks_schedule_fire(task: &crate::traits::Task) -> bool {
    matches!(task.status.as_str(), "pending" | "claimed" | "running")
        || (task.status == "blocked" && !task_is_blocked_by_terminal_dependency(task))
}

pub(crate) fn task_blocks_later_schedule_fire(
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

fn task_is_terminal_schedule_failure(task: &crate::traits::Task) -> bool {
    matches!(
        task.status.as_str(),
        "failed" | "interrupted" | "cancelled" | "blocked"
    )
}

/// Reconcile an open scheduled occurrence from its authoritative task graph.
/// A blocked child is a terminal failure of this occurrence, not a second
/// meaning for the run's `blocked` waiting state. Autonomous recovery is a
/// later linked run rather than an indefinitely open scheduled occurrence.
fn scheduled_run_reconciliation_status(
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

fn stranded_manual_run_matches_pending_tasks(
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

/// Runtime snapshot of a heartbeat background job.
#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatJobSnapshot {
    pub name: String,
    pub interval_secs: u64,
    pub last_run_at: Option<String>,
    pub last_success_at: Option<String>,
    pub last_error_at: Option<String>,
    pub last_error: Option<String>,
    pub consecutive_failures: u32,
    pub is_running: bool,
}

impl HeartbeatJobSnapshot {
    fn new(name: &str, interval: Duration) -> Self {
        Self {
            name: name.to_string(),
            interval_secs: interval.as_secs(),
            last_run_at: None,
            last_success_at: None,
            last_error_at: None,
            last_error: None,
            consecutive_failures: 0,
            is_running: false,
        }
    }
}

/// Shared telemetry for heartbeat jobs (dashboard/API consumption).
#[derive(Default)]
pub struct HeartbeatTelemetry {
    jobs: Mutex<HashMap<String, HeartbeatJobSnapshot>>,
}

impl HeartbeatTelemetry {
    pub fn new() -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
        }
    }

    pub fn register_job(&self, name: &str, interval: Duration) {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        jobs.entry(name.to_string())
            .or_insert_with(|| HeartbeatJobSnapshot::new(name, interval));
    }

    pub fn mark_started(&self, name: &str) {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(job) = jobs.get_mut(name) {
            job.last_run_at = Some(Utc::now().to_rfc3339());
            job.is_running = true;
        }
    }

    pub fn mark_success(&self, name: &str) {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(job) = jobs.get_mut(name) {
            job.last_success_at = Some(Utc::now().to_rfc3339());
            job.last_error = None;
            job.last_error_at = None;
            job.consecutive_failures = 0;
            job.is_running = false;
        }
    }

    pub fn mark_failure(&self, name: &str, consecutive_failures: u32, message: String) {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(job) = jobs.get_mut(name) {
            job.last_error_at = Some(Utc::now().to_rfc3339());
            job.last_error = Some(message);
            job.consecutive_failures = consecutive_failures;
            job.is_running = false;
        }
    }

    pub fn mark_not_running(&self, name: &str) {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(job) = jobs.get_mut(name) {
            job.is_running = false;
        }
    }

    pub fn snapshots(&self) -> Vec<HeartbeatJobSnapshot> {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<HeartbeatJobSnapshot> = jobs.values().cloned().collect();
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        rows
    }
}

/// Type alias for the async closure that heartbeat jobs execute.
type HeartbeatRunFn =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>> + Send + Sync>;

/// A registered periodic job.
pub struct HeartbeatJob {
    pub name: String,
    pub interval: Duration,
    pub last_run: Option<Instant>,
    /// Guards against overlapping invocations of the same job.
    pub is_running: Arc<AtomicBool>,
    /// Consecutive failure count — drives exponential backoff.
    pub consecutive_failures: Arc<AtomicU32>,
    /// The async function to call. Runs in a spawned tokio task.
    pub run: HeartbeatRunFn,
    /// Deferrable LLM work (memory pipeline): skip ticks while an agent task
    /// is in flight so it can't evict the task's KV prefix or steal compute.
    /// NEVER set for correctness-critical jobs (watchdogs, goal dispatch,
    /// orphan reclaim) — those must run while tasks are active.
    pub defer_while_agent_busy: bool,
}

/// Coordinates all background periodic tasks in a single tick loop.
///
/// Each tick takes milliseconds (SQLite reads + task spawns). Actual work
/// runs in parallel tokio tasks gated by an `Arc<Semaphore>`.
pub struct HeartbeatCoordinator {
    jobs: Vec<HeartbeatJob>,
    state: Arc<dyn StateStore>,
    semaphore: Arc<Semaphore>,
    tick_interval: Duration,
    wake_rx: mpsc::Receiver<()>,
    hub: Option<Weak<dyn OutboundRouter>>,
    goal_token_registry: Option<GoalTokenRegistry>,
    telemetry: Option<Arc<HeartbeatTelemetry>>,
    agent: Option<Weak<Agent>>,
    db_healthy: bool,
    last_stale_goal_cleanup: Option<Instant>,
    /// Seconds of no task activity before `detect_stuck_tasks` interrupts a
    /// running/claimed task. Defaults to 300; overridden from config via
    /// `set_task_inactivity_timeout`.
    task_inactivity_timeout_secs: i64,
}

impl HeartbeatCoordinator {
    /// Wait between an escalation (or the previous automatic attempt) and the
    /// next automatic recovery run. Long enough for a transient external
    /// cause to clear; short enough that a daily objective is not lost for a
    /// week.
    const ESCALATED_RECOVERY_COOLDOWN_SECS: i64 = 6 * 3600;
    /// Automatic recovery attempts per escalation before the objective waits
    /// for the owner.
    const ESCALATED_RECOVERY_MAX_ATTEMPTS: u16 = 3;

    pub fn new(
        state: Arc<dyn StateStore>,
        tick_interval_secs: u64,
        max_concurrent: usize,
        wake_rx: mpsc::Receiver<()>,
        hub: Option<Weak<dyn OutboundRouter>>,
        goal_token_registry: Option<GoalTokenRegistry>,
        telemetry: Option<Arc<HeartbeatTelemetry>>,
    ) -> Self {
        Self {
            jobs: Vec::new(),
            state,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            tick_interval: Duration::from_secs(tick_interval_secs),
            wake_rx,
            hub,
            goal_token_registry,
            telemetry,
            agent: None,
            db_healthy: true,
            last_stale_goal_cleanup: None,
            task_inactivity_timeout_secs: 300,
        }
    }

    /// Override the stuck-task inactivity timeout (deferred, since the value
    /// comes from config resolved after construction). Mirrors `set_hub`/`set_agent`.
    pub fn set_task_inactivity_timeout(&mut self, secs: u64) {
        self.task_inactivity_timeout_secs = secs as i64;
    }

    /// Set the hub reference (deferred, since hub is created after heartbeat).
    pub fn set_hub(&mut self, hub: Weak<dyn OutboundRouter>) {
        self.hub = Some(hub);
    }

    /// Set the agent reference (deferred, since agent is created before heartbeat starts).
    /// Used for dispatching orphaned pending tasks.
    pub fn set_agent(&mut self, agent: Weak<Agent>) {
        self.agent = Some(agent);
    }

    /// Register a periodic job with the heartbeat coordinator.
    pub fn register_job<F, Fut>(&mut self, name: &str, interval: Duration, f: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        self.jobs.push(HeartbeatJob {
            name: name.to_string(),
            interval,
            last_run: None,
            is_running: Arc::new(AtomicBool::new(false)),
            consecutive_failures: Arc::new(AtomicU32::new(0)),
            run: Box::new(move || Box::pin(f())),
            defer_while_agent_busy: false,
        });
        if let Some(telemetry) = &self.telemetry {
            telemetry.register_job(name, interval);
        }
    }

    /// Like `register_job`, but the job yields to in-flight agent tasks
    /// (interactive turns, goal runs, specialists). Use for the deferrable
    /// memory pipeline only — see `HeartbeatJob::defer_while_agent_busy`.
    pub fn register_deferrable_job<F, Fut>(&mut self, name: &str, interval: Duration, f: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        self.register_job(name, interval, f);
        if let Some(job) = self.jobs.last_mut() {
            job.defer_while_agent_busy = true;
        }
    }

    /// Consume self and start the tick loop in a spawned tokio task.
    pub fn start(mut self) {
        tokio::spawn(async move {
            // One-time startup recovery before entering the tick loop
            self.startup_recovery().await;

            loop {
                if let Err(e) = self.tick().await {
                    error!("Heartbeat tick failed: {}", e);
                }
                tokio::select! {
                    _ = tokio::time::sleep(self.tick_interval) => {},
                    _ = self.wake_rx.recv() => {},  // user message wakes immediately
                }
            }
        });
    }

    /// One-time recovery after startup: mark interrupted tasks, rebuild token registry.
    async fn startup_recovery(&self) {
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

    /// Run one tick: 5-phase cycle.
    pub(crate) async fn tick(&mut self) -> anyhow::Result<()> {
        // Phase 0: Health check
        if let Err(e) = self.state.health_check().await {
            if self.db_healthy {
                error!(error = %e, "DB health check failed — skipping tick");
                self.db_healthy = false;
            }
            return Ok(());
        }
        if !self.db_healthy {
            info!("DB health restored");
            self.db_healthy = true;
        }

        // Phase 1: Fire registered periodic jobs
        let now = Instant::now();
        for job in &mut self.jobs {
            let should_run = match job.last_run {
                None => true,
                Some(last) => now.duration_since(last) >= job.interval,
            };

            if should_run {
                // Skip if previous invocation is still running
                if job.is_running.load(Ordering::Relaxed) {
                    tracing::debug!(job = %job.name, "Skipping — previous invocation still running");
                    continue;
                }

                // Deferrable memory-pipeline work yields to in-flight agent
                // tasks (KV-slot and compute contention — measured 5x budget
                // inflation on a goal run, 2026-07-03). `last_run` stays
                // untouched, so the job remains due and fires on the first
                // idle tick; the 3x-interval starvation cap inside the policy
                // lets a long-starved job run regardless.
                if crate::agent::activity_gate::should_defer_heartbeat_job(
                    job.defer_while_agent_busy,
                    crate::agent::activity_gate::agent_busy(),
                    job.last_run.map(|l| now.duration_since(l)),
                    job.interval,
                ) {
                    tracing::debug!(job = %job.name, "Deferring — agent task in flight");
                    continue;
                }

                // Exponential backoff: if the job has been failing, delay its next run.
                // effective_interval = interval * 2^min(failures, 5)
                let failures = job.consecutive_failures.load(Ordering::Relaxed);
                if failures > 0 {
                    let backoff_multiplier = 2u32.pow(failures.min(5));
                    let effective_interval = job.interval * backoff_multiplier;
                    let actual_elapsed = match job.last_run {
                        Some(last) => now.duration_since(last),
                        None => effective_interval, // first run, allow it
                    };
                    if actual_elapsed < effective_interval {
                        tracing::debug!(
                            job = %job.name,
                            failures,
                            backoff_secs = effective_interval.as_secs(),
                            "Skipping — backoff not elapsed"
                        );
                        continue;
                    }
                }

                job.last_run = Some(now);
                let sem = self.semaphore.clone();
                let run_fn = &job.run;
                let fut = (run_fn)();
                let job_name = job.name.clone();
                let is_running = job.is_running.clone();
                let consecutive_failures = job.consecutive_failures.clone();
                let telemetry = self.telemetry.clone();
                is_running.store(true, Ordering::Relaxed);
                if let Some(ref t) = telemetry {
                    t.mark_started(&job_name);
                }
                tokio::spawn(async move {
                    let _permit = sem.acquire().await;
                    tracing::debug!(job = %job_name, "Heartbeat job starting");
                    // Catch panics as failures for backoff purposes
                    let result = AssertUnwindSafe(fut).catch_unwind().await;
                    is_running.store(false, Ordering::Relaxed);
                    match result {
                        Ok(Ok(())) => {
                            let prev = consecutive_failures.swap(0, Ordering::Relaxed);
                            if prev > 0 {
                                info!(job = %job_name, prev_failures = prev, "Heartbeat job recovered");
                            }
                            if let Some(ref t) = telemetry {
                                t.mark_success(&job_name);
                            }
                            tracing::debug!(job = %job_name, "Heartbeat job completed");
                        }
                        Ok(Err(e)) => {
                            let count = consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                            error!(
                                job = %job_name,
                                error = %e,
                                consecutive_failures = count,
                                "Heartbeat job failed — backing off"
                            );
                            if let Some(ref t) = telemetry {
                                t.mark_failure(&job_name, count, e.to_string());
                            }
                        }
                        Err(_) => {
                            let count = consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                            error!(
                                job = %job_name,
                                consecutive_failures = count,
                                "Heartbeat job panicked — backing off"
                            );
                            if let Some(ref t) = telemetry {
                                t.mark_failure(
                                    &job_name,
                                    count,
                                    "Heartbeat job panicked".to_string(),
                                );
                            }
                        }
                    }
                    if let Some(ref t) = telemetry {
                        t.mark_not_running(&job_name);
                    }
                });
            }
        }

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
        if let Some(agent) = self.agent.as_ref().and_then(Weak::upgrade) {
            let dispatched = agent.retry_deferred_provider_requests().await;
            if dispatched > 0 {
                info!(dispatched, "Re-ran requests deferred by a provider outage");
            }
        }

        Ok(())
    }

    /// Detect tasks that have been running/claimed longer than the timeout and mark them interrupted.
    async fn detect_stuck_tasks(&self) {
        match self.state.recover_expired_task_attempts().await {
            Ok(tasks) if !tasks.is_empty() => {
                warn!(
                    count = tasks.len(),
                    "Recovered tasks whose execution leases expired"
                );
            }
            Ok(_) => {}
            Err(error) => {
                error!(%error, "Failed to recover expired execution leases");
                return;
            }
        }
        let stuck = match self
            .state
            .get_stuck_tasks(self.task_inactivity_timeout_secs)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                error!(error = %e, "Failed to get stuck tasks");
                return;
            }
        };
        for task in &stuck {
            let last_activity = self
                .state
                .get_task_activities(&task.id)
                .await
                .ok()
                .and_then(|acts| acts.last().map(|a| a.created_at.clone()));
            let inactivity_secs = task_inactivity_secs(
                last_activity.as_deref(),
                task.started_at.as_deref().unwrap_or(&task.created_at),
                chrono::Utc::now(),
            );
            warn!(
                task_id = %task.id,
                goal_id = %task.goal_id,
                inactivity_secs,
                "Marking stuck task as interrupted"
            );
            if let Err(e) = self.state.mark_task_interrupted(&task.id).await {
                error!(task_id = %task.id, error = %e, "Failed to mark stuck task");
            }
        }
    }

    async fn block_pending_tasks_with_terminal_failed_dependencies(&self) {
        let goals = match self.state.get_active_goals().await {
            Ok(g) => g,
            Err(e) => {
                error!(error = %e, "Failed to get active goals for pending dependency cleanup");
                return;
            }
        };

        let now = chrono::Utc::now().to_rfc3339();
        let terminal_failed = |status: &str| {
            matches!(
                status,
                "failed" | "blocked" | "interrupted" | "cancelled" | "abandoned"
            )
        };

        for goal in &goals {
            let tasks = match self.state.get_tasks_for_goal(&goal.id).await {
                Ok(t) => t,
                Err(e) => {
                    error!(goal_id = %goal.id, error = %e, "Failed to get tasks for pending dependency cleanup");
                    continue;
                }
            };
            let graph = match crate::traits::task_execution_graph(&tasks) {
                Ok(graph) => graph,
                Err(error) => {
                    error!(goal_id = %goal.id, %error, "Invalid task dependency graph; leaving tasks unclaimed");
                    continue;
                }
            };
            let by_id: std::collections::HashMap<&str, &crate::traits::Task> =
                tasks.iter().map(|t| (t.id.as_str(), t)).collect();

            for task in tasks.iter().filter(|t| t.status == "pending") {
                let Some(failed_dep) = graph
                    .unresolved_dependencies(&task.id)
                    .iter()
                    .filter_map(|dependency| by_id.get(dependency.id.as_str()))
                    .find(|dep| terminal_failed(&dep.status))
                else {
                    continue;
                };

                let mut updated = task.clone();
                updated.status = "blocked".to_string();
                updated.blocker = Some(format!(
                    "Dependency {} ended with status {} before this task could run.",
                    failed_dep.id, failed_dep.status
                ));
                updated.completed_at = Some(now.clone());
                if let Err(e) = self.state.update_task(&updated).await {
                    error!(
                        task_id = %task.id,
                        goal_id = %goal.id,
                        dependency_id = %failed_dep.id,
                        error = %e,
                        "Failed to block pending task with terminal failed dependency"
                    );
                } else {
                    warn!(
                        task_id = %task.id,
                        goal_id = %goal.id,
                        dependency_id = %failed_dep.id,
                        dependency_status = %failed_dep.status,
                        "Blocked pending task with terminal failed dependency"
                    );
                }
            }
        }
    }

    /// Auto-retry failed idempotent tasks that haven't exceeded their max retries.
    /// Resets them to "pending" so they get picked up by dispatch_pending_tasks.
    async fn auto_retry_failed_tasks(&self) {
        let goals = match self.state.get_active_goals().await {
            Ok(g) => g,
            Err(e) => {
                error!(error = %e, "Failed to get active goals for auto-retry");
                return;
            }
        };

        let mut retried = 0;
        for goal in &goals {
            // Only retry within active orchestration goals.
            if goal.status != "active" {
                continue;
            }

            // The active task lead owns retries for its run. Promoting one of
            // its freshly-failed tasks back to pending here can race the lead's
            // own completion accounting and manufacture an orphaned run.
            if self
                .goal_token_registry
                .as_ref()
                .is_some_and(|registry| registry.is_run_active(&goal.id))
            {
                continue;
            }

            let tasks = match self.state.get_tasks_for_goal(&goal.id).await {
                Ok(t) => t,
                Err(_) => continue,
            };
            let schedules = self
                .state
                .get_schedules_for_goal(&goal.id)
                .await
                .unwrap_or_default();
            let scheduled_run = if schedules.is_empty() {
                None
            } else {
                self.state
                    .get_scheduled_run_state(&goal.id)
                    .await
                    .ok()
                    .flatten()
            };

            // Scheduled goals intentionally stay active between runs. Without
            // an active run record there is no retryable cycle, regardless of
            // whether old child tasks were individually marked idempotent.
            if !schedules.is_empty() && scheduled_run.is_none() {
                continue;
            }

            for task in &tasks {
                if scheduled_run
                    .as_ref()
                    .is_some_and(|run| !timestamp_is_at_or_after(&task.created_at, &run.created_at))
                {
                    continue;
                }
                if task.status == "failed" && task.idempotent && task.retry_count < task.max_retries
                {
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
                        error!(task_id = %task.id, error = %e, "Failed to auto-retry failed task");
                    } else {
                        retried += 1;
                        info!(
                            task_id = %task.id,
                            goal_id = %goal.id,
                            retry = retry_task.retry_count,
                            max_retries = retry_task.max_retries,
                            "Auto-retried failed idempotent task"
                        );

                        // Notify user that a retry is in progress
                        let task_desc: String = task.description.chars().take(160).collect();
                        let goal_desc: String = goal.description.chars().take(120).collect();
                        let msg = format!(
                            "Retrying task (attempt {}/{}): {} (goal: {})",
                            retry_task.retry_count, retry_task.max_retries, task_desc, goal_desc
                        );
                        let entry = crate::traits::NotificationEntry::new(
                            &goal.id,
                            &goal.session_id,
                            "status_update",
                            &msg,
                        );
                        if let Err(error) = self.state.enqueue_notification(&entry).await {
                            warn!(
                                task_id = %task.id,
                                %error,
                                "Failed to enqueue task retry notification"
                            );
                        }
                    }
                }
            }
        }

        if retried > 0 {
            info!(count = retried, "Auto-retried failed idempotent tasks");
        }
    }

    /// Dispatch orphaned pending tasks by atomically claiming them and spawning task leads.
    ///
    /// For each active goal with pending tasks but no running agent:
    /// 1. Atomically claim a task via `claim_task` (prevents duplicate dispatch)
    /// 2. Spawn a background task lead gated by the semaphore
    /// 3. The task lead picks up all pending tasks for the goal
    ///
    /// Falls back to notification if no agent reference is available.
    async fn dispatch_pending_tasks(&self) {
        let pending = match self.state.get_pending_tasks_by_priority(20).await {
            Ok(t) => t,
            Err(e) => {
                error!(error = %e, "Failed to get pending tasks");
                return;
            }
        };
        if pending.is_empty() {
            return;
        }

        // Group pending tasks by goal_id
        let mut goals_with_pending: std::collections::HashMap<String, Vec<&crate::traits::Task>> =
            std::collections::HashMap::new();
        for task in &pending {
            goals_with_pending
                .entry(task.goal_id.clone())
                .or_default()
                .push(task);
        }

        for (goal_id, tasks) in &goals_with_pending {
            let mut goal = match self.state.get_goal(goal_id).await {
                Ok(Some(g)) => g,
                _ => continue,
            };

            // Self-heal the impossible state produced by older run_now code:
            // a manual run with a pending root task attached to a stalled goal.
            // The task dispatcher normally filters non-active goals, so without
            // this invariant repair the run can say "running" forever despite
            // never having acquired a worker.
            if goal.status == "stalled" {
                let stranded_run = match self.state.get_current_goal_run(goal_id).await {
                    Ok(Some(run)) => {
                        let run_tasks = self
                            .state
                            .get_tasks_for_goal_run(&run.id)
                            .await
                            .unwrap_or_default();
                        stranded_manual_run_matches_pending_tasks(&run, &run_tasks, tasks)
                            .then_some(run)
                    }
                    _ => None,
                };
                if let Some(run) = stranded_run {
                    goal.status = "active".to_string();
                    goal.completed_at = None;
                    goal.updated_at = chrono::Utc::now().to_rfc3339();
                    if let Err(error) = self.state.update_goal(&goal).await {
                        error!(goal_id, %error, "Failed to reactivate stranded manual run");
                        continue;
                    }
                    warn!(
                        goal_id,
                        run_id = %run.id,
                        pending_count = tasks.len(),
                        "Reactivated stranded manual run before dispatch"
                    );
                    let message = format!(
                        "Recovered the existing manual run `{}`; it had been queued behind a stalled goal and never started. No duplicate was created. Starting it now, with progress updates enabled.",
                        run.id
                    );
                    let entry = crate::traits::NotificationEntry::new(
                        goal_id,
                        &goal.session_id,
                        "status_update",
                        &message,
                    );
                    if let Err(error) = self.state.enqueue_notification(&entry).await {
                        warn!(goal_id, %error, "Failed to enqueue manual-run recovery update");
                    }
                }
            }

            // Only active goals are dispatchable. Stalled non-manual work is
            // intentionally left alone for explicit recovery.
            if goal.status != "active" {
                continue;
            }

            // A live task lead is authoritative for this goal even during a
            // brief interval where none of its task rows are marked running.
            // Do not let orphan recovery claim a pending task during that gap.
            if self
                .goal_token_registry
                .as_ref()
                .is_some_and(|registry| registry.is_run_active(goal_id))
            {
                continue;
            }

            // Mandate controllers are not ordinary unscheduled goals. Generic
            // orphan recovery may provide a fallback for the exact pending root
            // of the current mandate run, but it must never promote a model-
            // authored non-root action task into a task lead. Such tasks remain
            // owned by mandate finalization/orphan reconciliation.
            let mandate_controller = match self.state.get_mandate_for_goal(goal_id).await {
                Ok(mandate) => mandate.is_some(),
                Err(error) => {
                    warn!(
                        goal_id,
                        %error,
                        "Skipping orphan dispatch because mandate ownership could not be resolved"
                    );
                    continue;
                }
            };
            let current_mandate_root_task_id = if mandate_controller {
                self.state
                    .get_current_goal_run(goal_id)
                    .await
                    .ok()
                    .flatten()
                    .filter(|run| run.trigger_type == "mandate" && run.status == "running")
                    .and_then(|run| run.root_task_id)
            } else {
                None
            };

            let schedules_for_goal = self.state.get_schedules_for_goal(goal_id).await.ok();
            let goal_runs = self.state.get_goal_runs(goal_id).await.unwrap_or_default();
            let is_scheduled_goal = schedules_for_goal
                .as_ref()
                .is_some_and(|schedules| !schedules.is_empty())
                || goal_runs.iter().any(|run| run.trigger_type == "scheduled");
            let active_scheduled_run = if is_scheduled_goal {
                self.state
                    .get_scheduled_run_state(goal_id)
                    .await
                    .ok()
                    .flatten()
            } else {
                None
            };
            // A goal run is the durable isolation boundary for one scheduled or
            // explicit manual firing.  In particular, `trigger_now` starts a new
            // goal run without advancing the cron schedule's `last_run_at`, so
            // using only the schedule timestamp here would let a mutation from
            // the previous run suppress the newly requested run. Prefer the
            // scheduled-run projection's root-task relationship: legacy runs
            // may have been created before trigger provenance was typed, but
            // the root task still identifies their exact owning run.
            let projected_goal_run = if let Some(scheduled_run) = active_scheduled_run.as_ref() {
                self.state
                    .get_goal_run_for_task(&scheduled_run.root_task_id)
                    .await
                    .ok()
                    .flatten()
                    .filter(|run| {
                        run.goal_id == *goal_id
                            && matches!(run.status.as_str(), "pending" | "running" | "blocked")
                    })
            } else {
                None
            };
            let current_goal_run = if is_scheduled_goal {
                match projected_goal_run {
                    Some(run) => Some(run),
                    None => self
                        .state
                        .get_current_goal_run(goal_id)
                        .await
                        .ok()
                        .flatten()
                        // An automatic recovery occurrence is a valid current
                        // cycle of a scheduled objective: its root task must be
                        // dispatched, never retired as a stale child.
                        .filter(|run| matches!(run.trigger_type.as_str(), "scheduled" | "recovery"))
                        // Legacy/repair-created open runs can contain old child
                        // rows without an execution root. They are not a valid
                        // current scheduled cycle and must keep using the stale
                        // child retirement fallback below.
                        .filter(|run| run.root_task_id.is_some()),
                }
            } else {
                None
            };
            let current_goal_run_task_ids = if let Some(run) = current_goal_run.as_ref() {
                self.state
                    .get_tasks_for_goal_run(&run.id)
                    .await
                    .ok()
                    .map(|tasks| {
                        tasks
                            .into_iter()
                            .map(|task| task.id)
                            .collect::<HashSet<_>>()
                    })
            } else {
                None
            };
            let mut eligible_tasks: Vec<&crate::traits::Task> = Vec::with_capacity(tasks.len());
            for task in tasks.iter().copied() {
                let eligible = if mandate_controller {
                    current_mandate_root_task_id.as_deref() == Some(task.id.as_str())
                } else if !is_scheduled_goal {
                    true
                } else if let Some(run_task_ids) = current_goal_run_task_ids.as_ref() {
                    // Durable run membership is authoritative and avoids
                    // timestamp races between task construction and the run
                    // row created milliseconds later.
                    run_task_ids.contains(&task.id)
                } else if let Some(run) = active_scheduled_run.as_ref() {
                    timestamp_is_at_or_after(&task.created_at, &run.created_at)
                } else {
                    // No typed current-run identity means ownership is
                    // indeterminate. Do not infer it from task prose.
                    false
                };
                if eligible {
                    eligible_tasks.push(task);
                    continue;
                }

                if mandate_controller {
                    // Non-root mandate tasks are intentionally invisible to
                    // generic orphan recovery. Leave their state untouched for
                    // the mandate finalizer/reconciler that owns this run.
                    continue;
                }

                let mut retired = task.clone();
                retired.status = "cancelled".to_string();
                retired.idempotent = false;
                retired.max_retries = 0;
                retired.error = Some(
                    "Cancelled stale task from a closed scheduled run; a future schedule fire \
                     will create a fresh root task."
                        .to_string(),
                );
                retired.completed_at = Some(chrono::Utc::now().to_rfc3339());
                if let Err(error) = self.state.update_task(&retired).await {
                    warn!(
                        task_id = %task.id,
                        goal_id = %goal_id,
                        %error,
                        "Failed to retire stale scheduled-run task"
                    );
                } else {
                    info!(
                        task_id = %task.id,
                        goal_id = %goal_id,
                        "Retired stale task from closed scheduled run"
                    );
                }
            }
            if eligible_tasks.is_empty() {
                continue;
            }

            // Daily budget is admission control for scheduled runs, not a reason
            // to abandon already-pending scheduled work.
            if !is_scheduled_goal {
                let budget_today = chrono::Utc::now().date_naive().to_string();
                if daily_budget_exhausted(
                    goal.budget_daily,
                    goal.tokens_used_today,
                    &goal.tokens_used_day,
                    &budget_today,
                ) {
                    tracing::info!(
                        goal_id = %goal.id,
                        tokens_used = goal.tokens_used_today,
                        budget = ?goal.budget_daily,
                        "Skipping pending-task dispatch — today's goal daily budget exhausted"
                    );
                    continue;
                }
            }

            // Check if any tasks are still running (task lead is alive)
            let all_tasks = match self.state.get_tasks_for_goal(goal_id).await {
                Ok(t) => t,
                Err(_) => continue,
            };
            // Only consider a task as actively running if it was started/claimed
            // within the last 10 minutes. Stale claimed tasks (executor crashed)
            // should not block dispatch forever.
            let stale_threshold_secs: i64 = 600; // 10 minutes
            let has_active_nonstale = all_tasks.iter().any(|t| {
                let belongs_to_current_run = if mandate_controller {
                    current_mandate_root_task_id.as_deref() == Some(t.id.as_str())
                } else if !is_scheduled_goal {
                    true
                } else if let Some(run_task_ids) = current_goal_run_task_ids.as_ref() {
                    run_task_ids.contains(&t.id)
                } else if let Some(run) = active_scheduled_run.as_ref() {
                    timestamp_is_at_or_after(&t.created_at, &run.created_at)
                } else {
                    false
                };
                if !belongs_to_current_run {
                    return false;
                }
                if t.status != "running" && t.status != "claimed" {
                    return false;
                }
                let timestamp = t.started_at.as_deref().unwrap_or(&t.created_at);
                chrono::DateTime::parse_from_rfc3339(timestamp)
                    .map(|dt| {
                        let age = chrono::Utc::now() - dt.with_timezone(&chrono::Utc);
                        age.num_seconds() < stale_threshold_secs
                    })
                    .unwrap_or(false)
            });

            if has_active_nonstale {
                // Task lead is still working (recently active) — it'll pick up the pending tasks
                continue;
            }

            // Orphaned: active goal with pending tasks but nothing running.
            // Only consider tasks orphaned if they've been pending for > 60 seconds.
            // This prevents racing with a task lead that just created the tasks
            // but hasn't started dispatching them yet.
            let min_age_secs = 60;
            let all_too_new = eligible_tasks.iter().all(|t| {
                chrono::DateTime::parse_from_rfc3339(&t.created_at)
                    .map(|dt| {
                        let age = chrono::Utc::now() - dt.with_timezone(&chrono::Utc);
                        age.num_seconds() < min_age_secs
                    })
                    .unwrap_or(false)
            });
            if all_too_new {
                continue;
            }

            // Idempotency backstop: these pending tasks are orphaned (nothing's
            // actively running, and they've sat >60s), which is exactly the
            // situation that let a scheduled goal's side effect get repeated —
            // e.g. a task lead posts a tweet, a downstream step (verification,
            // task tracking) hiccups, the run ends up interrupted, and a fresh
            // task lead gets dispatched here for what looks like unfinished
            // work but whose real-world action already succeeded. Before
            // reclaiming, check whether this exact goal run already has a
            // successful mutating (non-GET) http_request logged; if so, the
            // run's job is done — close out the leftover pending tasks instead
            // of spawning another attempt. Never use a prior run's receipt:
            // an explicit manual run is a new authorized execution even when
            // the recurring schedule's `last_run_at` has not advanced.
            // See docs/2026-06-30-telegram-edge-case-findings.md (2026-07-04
            // escalation: 5 duplicate tweets from repeated goal dispatch).
            if is_scheduled_goal {
                if let Some(current_run) = current_goal_run.as_ref() {
                    let mutating_success_task_ids =
                        self.goal_run_mutating_success_tasks(&current_run.id).await;
                    if !mutating_success_task_ids.is_empty() {
                        info!(
                            goal_id = %goal_id,
                            run_id = %current_run.id,
                            pending_count = eligible_tasks.len(),
                            "Goal run already has a successful mutating action; \
                             closing out orphaned pending tasks instead of re-dispatching"
                        );
                        // The receipt is authoritative evidence that the task's
                        // external mutation succeeded, even if a downstream
                        // verification/tracking hiccup left its task status as
                        // interrupted or failed.
                        for task_id in &mutating_success_task_ids {
                            let Ok(Some(mut done)) = self.state.get_task(task_id).await else {
                                continue;
                            };
                            if !done.satisfies_run_completion() {
                                done.status = "completed".to_string();
                                done.error = None;
                                done.blocker = None;
                                done.result = Some(
                                    "Auto-reconciled: this task has a successful mutating \
                                     tool receipt in the current goal run."
                                        .to_string(),
                                );
                                done.completed_at = Some(chrono::Utc::now().to_rfc3339());
                                let _ = self.state.update_task(&done).await;
                            }
                        }
                        for task in &eligible_tasks {
                            let mut done = (*task).clone();
                            done.status = "completed".to_string();
                            done.result = Some(
                                "Auto-closed: a mutating action for this goal's current cycle \
                                 already succeeded (see task_activity log); skipped to avoid a \
                                 duplicate side effect."
                                    .to_string(),
                            );
                            done.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            let _ = self.state.update_task(&done).await;
                        }
                        if self
                            .state
                            .get_tasks_for_goal_run(&current_run.id)
                            .await
                            .is_ok_and(|run_tasks| {
                                !run_tasks.is_empty()
                                    && run_tasks.iter().all(|task| task.satisfies_run_completion())
                            })
                        {
                            let _ = self
                                .state
                                .finish_goal_run(
                                    &current_run.id,
                                    "completed",
                                    Some(
                                        "Auto-reconciled from a successful mutating receipt in \
                                         this goal run; duplicate dispatch was suppressed.",
                                    ),
                                )
                                .await;
                        }
                        continue;
                    }
                }
            }

            // Try to atomically claim the first pending task and spawn a task lead.
            let first_task = eligible_tasks[0];
            let agent_id = format!("heartbeat-dispatch-{}", uuid::Uuid::new_v4());

            let attempt = match self
                .state
                .claim_task_with_lease(&first_task.id, &agent_id, Some("profile-task-lead"), 180)
                .await
            {
                Ok(Some(attempt)) => attempt,
                Ok(None) => continue,
                Err(e) => {
                    error!(task_id = %first_task.id, error = %e, "Failed to claim task for dispatch");
                    continue;
                }
            };

            info!(
                goal_id = %goal_id,
                task_id = %first_task.id,
                pending_count = eligible_tasks.len(),
                "Claimed orphaned task, dispatching task lead"
            );

            // Try to spawn a task lead via agent reference
            if let Some(agent_weak) = &self.agent {
                if let Some(agent_arc) = agent_weak.upgrade() {
                    let state = self.state.clone();
                    let hub = self.hub.clone();
                    let goal_token_registry = self.goal_token_registry.clone();
                    let goal_clone = goal.clone();
                    let session_id = goal.session_id.clone();

                    // Register cancellation token for this goal if not already present
                    if let Some(ref registry) = goal_token_registry {
                        registry.register(&goal.id).await;
                    }

                    // spawn_background_task_lead internally calls tokio::spawn.
                    // Semaphore gating happens at the Agent level during LLM calls.
                    // Use the actual pending task description as TaskLead input.
                    // Passing a generic "resume orphaned tasks" string causes the
                    // TaskLead to re-scope work away from the user's goal.
                    let dispatch_task_text = if first_task.description.trim().is_empty() {
                        goal.description.clone()
                    } else {
                        first_task.description.clone()
                    };

                    crate::agent::spawn_background_task_lead(
                        agent_arc,
                        goal_clone,
                        dispatch_task_text,
                        session_id,
                        ChannelContext::internal(),
                        UserRole::Owner,
                        state,
                        hub,
                        goal_token_registry,
                        Some(first_task.id.clone()),
                        None,
                    );
                    continue;
                }
            }

            // No agent available — revert claimed task back to pending so it's
            // not stranded, then enqueue a stalled notification.
            warn!(
                goal_id = %goal_id,
                task_id = %first_task.id,
                pending_count = eligible_tasks.len(),
                "No agent available for dispatch — reverting claim and notifying user"
            );
            let patch = crate::traits::TaskAttemptPatch {
                status: "cancelled".to_string(),
                error: Some("No agent was available for dispatch.".to_string()),
                ..Default::default()
            };
            let released = self
                .state
                .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
                .await
                .unwrap_or(false);
            if released {
                let _ = self
                    .state
                    .retry_work_task(&first_task.id, "heartbeat", None)
                    .await;
            }

            let msg = format!(
                "Goal stalled: \"{}\" has {} pending task(s) but no active agent. \
                 You can re-trigger this by asking me about it again.",
                goal.description.chars().take(200).collect::<String>(),
                eligible_tasks.len(),
            );
            let entry =
                crate::traits::NotificationEntry::new(goal_id, &goal.session_id, "stalled", &msg);
            if let Err(error) = self.state.enqueue_notification(&entry).await {
                error!(
                    goal_id,
                    %error,
                    "Failed to enqueue stalled-goal notification"
                );
            }
        }
    }

    /// Idempotency backstop for scheduled goals: returns the tasks in one
    /// durable goal run that have a logged, successful, mutating (non-GET)
    /// `http_request` activity.
    ///
    /// This is a defense-in-depth check, not the primary fix — it exists so
    /// that *whatever* caused a scheduled goal's task to look unfinished
    /// (timeout, tool cooldown, orchestrator misjudgment, a bug nobody's hit
    /// yet) can never turn into a repeated real-world side effect. It reads
    /// task_activity rows that are already logged for every tool call
    /// (`src/agent/loop/tool_execution/run.rs`), so it needs no schema change.
    /// Fails closed toward "don't block" on any parse/lookup miss — a false
    /// negative here just falls back to prior behavior, never a false positive
    /// that would wrongly skip real unfinished work. Scoping by `goal_run_id`
    /// is what prevents a successful prior run from suppressing a new manual
    /// trigger.
    async fn goal_run_mutating_success_tasks(&self, run_id: &str) -> Vec<String> {
        let tasks = match self.state.get_tasks_for_goal_run(run_id).await {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let mut successful_task_ids = Vec::new();
        for task in &tasks {
            let activities = match self.state.get_task_activities(&task.id).await {
                Ok(a) => a,
                Err(_) => continue,
            };
            for activity in activities {
                if activity.tool_name.as_deref() != Some("http_request") {
                    continue;
                }
                if activity.success != Some(true) {
                    continue;
                }
                if activity
                    .tool_args
                    .as_deref()
                    .is_some_and(is_mutating_http_method)
                {
                    successful_task_ids.push(task.id.clone());
                    break;
                }
            }
        }
        successful_task_ids
    }

    /// Phase 5a: Scan goals that completed/failed and enqueue notifications.
    /// Phase 5b: Process notification queue — attempt delivery.
    /// Phase 5c: Cleanup expired status_update notifications.
    async fn deliver_notifications(&self) {
        // Phase 5a: Enqueue notifications for goals that need them
        self.enqueue_goal_notifications().await;

        // Phase 5b: Process notification queue
        self.process_notification_queue().await;

        // Phase 5c: Cleanup expired status_update notifications (24h TTL)
        match self.state.cleanup_expired_notifications().await {
            Ok(count) if count > 0 => {
                info!(count, "Cleaned up expired status_update notifications");
            }
            Err(e) => {
                error!(error = %e, "Failed to cleanup expired notifications");
            }
            _ => {}
        }
    }

    /// Scan for goals needing notification and enqueue them.
    async fn enqueue_goal_notifications(&self) {
        let goals = match self.state.get_goals_needing_notification().await {
            Ok(g) => g,
            Err(e) => {
                error!(error = %e, "Failed to get goals needing notification");
                return;
            }
        };
        for goal in &goals {
            let (notification_type, msg) = match goal.status.as_str() {
                "completed" => {
                    // Prefer the latest run's task state. Historical failed runs
                    // remain valuable audit data but cannot invalidate a later,
                    // successfully tracked recovery run.
                    let latest_run = self
                        .state
                        .get_goal_runs(&goal.id)
                        .await
                        .unwrap_or_default()
                        .into_iter()
                        .next();
                    let latest_root_task_id = latest_run
                        .as_ref()
                        .and_then(|run| run.root_task_id.as_deref())
                        .map(str::to_string);
                    let completed_tasks = match latest_run {
                        Some(run) => self
                            .state
                            .get_tasks_for_goal_run(&run.id)
                            .await
                            .unwrap_or_default(),
                        None => self
                            .state
                            .get_tasks_for_goal(&goal.id)
                            .await
                            .unwrap_or_default(),
                    };
                    let fallback_summary: String = goal.description.chars().take(300).collect();
                    let task_results_summary = build_goal_task_results_summary(
                        &completed_tasks,
                        latest_root_task_id.as_deref(),
                        &fallback_summary,
                    );

                    if !completed_tasks.is_empty()
                        && !crate::tools::manage_goal_tasks::tasks_satisfy_goal_completion(
                            &completed_tasks,
                        )
                    {
                        let mut corrected = goal.clone();
                        corrected.status = "failed".to_string();
                        corrected.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        corrected.updated_at = chrono::Utc::now().to_rfc3339();
                        if let Err(error) = self.state.update_goal(&corrected).await {
                            error!(goal_id = %goal.id, %error, "Failed to repair inconsistent completed goal state");
                        }
                        (
                            "failed",
                            format!(
                                "Goal incomplete: required tasks did not finish successfully.\n\n{}",
                                task_results_summary.chars().take(4000).collect::<String>()
                            ),
                        )
                    } else {
                        // Check for partial success metadata in context
                        let partial_info = goal
                            .context
                            .as_deref()
                            .and_then(|ctx| serde_json::from_str::<serde_json::Value>(ctx).ok())
                            .filter(|v| {
                                v.get("partial_success")
                                    .and_then(|p| p.as_bool())
                                    .unwrap_or(false)
                            });

                        if let Some(summary) = partial_info {
                            let completed = summary
                                .get("completed")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let failed =
                                summary.get("failed").and_then(|v| v.as_u64()).unwrap_or(0);
                            let blocked = summary
                                .get("blocked")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let total =
                                summary.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
                            (
                                "failed",
                                format!(
                                    "Goal partially completed ({}/{} tasks succeeded, {} failed, {} blocked):\n\n{}",
                                    completed,
                                    total,
                                    failed,
                                    blocked,
                                    task_results_summary.chars().take(4000).collect::<String>()
                                ),
                            )
                        } else if crate::tools::manage_goal_tasks::goal_completion_summary_indicates_not_finished(&task_results_summary) {
                            // The summary contains verification-blocked or incomplete-work
                            // language. Suppress a duplicate terminal notice; structural task
                            // state above remains the authoritative completion gate.
                            continue;
                        } else {
                            (
                                "completed",
                                format!(
                                    "{}\n\n{}",
                                    goal_completion_header(&completed_tasks),
                                    task_results_summary.chars().take(4000).collect::<String>()
                                ),
                            )
                        }
                    }
                }
                "failed" => (
                    "failed",
                    format!(
                        "Goal failed: {}",
                        goal.description.chars().take(300).collect::<String>()
                    ),
                ),
                _ => continue,
            };

            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                notification_type,
                &msg,
            );

            if let Err(e) = self.state.enqueue_goal_notification(&entry).await {
                error!(goal_id = %goal.id, error = %e, "Failed to atomically enqueue notification");
            }
        }
    }

    /// Process the notification queue: attempt delivery, track attempts.
    async fn process_notification_queue(&self) {
        // Read past the normal delivery batch so quiet-hour routine entries at
        // the front of the queue cannot hide a later owner-action request.
        let pending = match self.state.get_pending_notifications(100).await {
            Ok(n) => n,
            Err(e) => {
                error!(error = %e, "Failed to get pending notifications");
                return;
            }
        };

        let mut delivery_attempts = 0usize;
        for entry in &pending {
            let local_hour = chrono::Timelike::hour(&chrono::Local::now());
            if !entry.should_deliver_at_local_hour(local_hour) {
                tracing::debug!(
                    notification_id = %entry.id,
                    notification_type = %entry.notification_type,
                    local_hour,
                    "Deferring recoverable notification during quiet hours"
                );
                continue;
            }
            if delivery_attempts >= 20 {
                break;
            }
            delivery_attempts = delivery_attempts.saturating_add(1);
            if entry.notification_type == "escalation" {
                if let Some(task_id) = entry.task_id.as_deref() {
                    // Give the coordinator a brief chance to reconcile the
                    // executor handoff, then verify the linked task is still
                    // blocked. This prevents a stale critical notification
                    // from racing a same-turn unblock or completion.
                    if !task_escalation_has_settled(&entry.created_at, Utc::now()) {
                        continue;
                    }
                    match self.state.get_task(task_id).await {
                        Ok(Some(task)) if task.status == "blocked" => {}
                        Ok(_) => {
                            if let Err(error) =
                                self.state.mark_notification_delivered(&entry.id).await
                            {
                                warn!(
                                    notification_id = %entry.id,
                                    task_id,
                                    error = %error,
                                    "Failed to close stale task escalation"
                                );
                            }
                            continue;
                        }
                        Err(error) => {
                            warn!(
                                notification_id = %entry.id,
                                task_id,
                                error = %error,
                                "Deferred task escalation because task state could not be verified"
                            );
                            continue;
                        }
                    }
                }
            }
            let delivered = if let Some(hub) = self.hub.as_ref().and_then(|w| w.upgrade()) {
                let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(&entry.message);
                let message =
                    crate::channels::present_notification(&entry.notification_type, &sanitized);
                // A channel adapter is external I/O. It must never be allowed to
                // pin the heartbeat forever because this same loop admits due
                // schedules and dispatches pending tasks on the next tick.
                match tokio::time::timeout(
                    Duration::from_secs(15),
                    hub.send_text(&entry.session_id, &message),
                )
                .await
                {
                    Ok(result) => result.is_ok(),
                    Err(_) => {
                        warn!(
                            notification_id = %entry.id,
                            session_id = %entry.session_id,
                            "Notification delivery timed out"
                        );
                        false
                    }
                }
            } else {
                false
            };

            if delivered {
                if entry.notification_type == "mandate_ask" {
                    if let Some(agent) = self.agent.as_ref().and_then(Weak::upgrade) {
                        // Queue delivery is the crash/retry path and bypasses
                        // `deliver_parent_text_result`, so persist the visible
                        // notice before installing its typed dialogue binding.
                        if let Err(error) = agent
                            .record_auxiliary_assistant_note(&entry.session_id, &entry.message)
                            .await
                        {
                            warn!(
                                notification_id = %entry.id,
                                %error,
                                "Failed to record delivered mandate ASK in owner history"
                            );
                        }
                        match self.state.get_mandate_for_goal(&entry.goal_id).await {
                            Ok(Some(mandate)) if mandate.status == MandateStatus::AwaitingInput => {
                                if let Err(error) = agent
                                    .record_mandate_owner_input_context(
                                        &entry.session_id,
                                        &mandate.id,
                                        mandate.version,
                                        entry.action_token.as_deref().unwrap_or(&entry.id),
                                    )
                                    .await
                                {
                                    warn!(
                                        mandate_id = %mandate.id,
                                        notification_id = %entry.id,
                                        %error,
                                        "Failed to bind queued mandate ASK to owner dialogue state"
                                    );
                                }
                            }
                            Ok(_) => {}
                            Err(error) => warn!(
                                goal_id = %entry.goal_id,
                                notification_id = %entry.id,
                                %error,
                                "Failed to load mandate after delivering queued ASK"
                            ),
                        }
                    }
                }
                if let Err(e) = self.state.mark_notification_delivered(&entry.id).await {
                    error!(notification_id = %entry.id, error = %e, "Failed to mark notification delivered");
                }
            } else {
                // Increment attempt counter — critical notifications will keep retrying
                // (no expiry), status_update notifications will eventually expire via TTL
                if let Err(e) = self.state.increment_notification_attempt(&entry.id).await {
                    error!(notification_id = %entry.id, error = %e, "Failed to increment notification attempt");
                }
            }
        }
    }

    /// Claim and dispatch due mandate reviews from the mandate-owned wake clock.
    ///
    /// This path is deliberately independent from `goal_schedules`: a mandate
    /// review gets `trigger_type = "mandate"`, no schedule id, and a distinct
    /// root-task description. The mandate lease is the admission fence; the
    /// task-attempt lease and per-goal run guard remain execution fences.
    async fn check_due_mandates(&self) {
        let lease_owner = format!("heartbeat-mandate-{}", uuid::Uuid::new_v4());
        let due = match self
            .state
            .claim_due_mandates(
                MANDATE_REVIEW_BATCH_SIZE,
                &lease_owner,
                MANDATE_REVIEW_LEASE_SECS,
            )
            .await
        {
            Ok(mandates) => mandates,
            Err(error) => {
                error!(%error, "Failed to claim due mandate reviews");
                return;
            }
        };

        if due.is_empty() {
            return;
        }

        info!(count = due.len(), "Found due mandate reviews");
        for mandate in due {
            let mandate_id = mandate.id.clone();
            let lease_token = mandate.review_lease_token.clone();
            if let Err(error) = self
                .dispatch_due_mandate_with_busy_retry(mandate.clone())
                .await
            {
                error!(
                    mandate_id,
                    goal_id = %mandate.goal_id,
                    %error,
                    "Failed to dispatch due mandate review"
                );
                if let Some(token) = lease_token.as_deref() {
                    let retry_at = if is_sqlite_busy_error(&error) {
                        mandate_transient_retry_at(&mandate, chrono::Utc::now())
                    } else {
                        mandate_retry_at(&mandate, Some(mandate.min_review_secs))
                    };
                    if let Err(release_error) = self
                        .state
                        .release_mandate_review_lease(&mandate.id, token, &retry_at)
                        .await
                    {
                        warn!(
                            mandate_id = %mandate.id,
                            %release_error,
                            "Failed to release mandate review lease after dispatch error"
                        );
                    }
                }
            }
        }
    }

    async fn dispatch_due_mandate_with_busy_retry(&self, mandate: Mandate) -> anyhow::Result<()> {
        let mut busy_retries = 0_u32;
        loop {
            match self.dispatch_due_mandate(mandate.clone()).await {
                Ok(()) => return Ok(()),
                Err(error)
                    if busy_retries < MANDATE_DISPATCH_BUSY_RETRIES
                        && is_sqlite_busy_error(&error) =>
                {
                    busy_retries += 1;
                    let delay_ms = 100_u64 << (busy_retries - 1);
                    warn!(
                        mandate_id = %mandate.id,
                        retry = busy_retries,
                        delay_ms,
                        "Retrying mandate dispatch after transient SQLite contention"
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn dispatch_due_mandate(&self, mandate: Mandate) -> anyhow::Result<()> {
        let lease_token = mandate
            .review_lease_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("claimed mandate did not carry a review lease token"))?;

        let Some(goal) = self.state.get_goal(&mandate.goal_id).await? else {
            let _ = self
                .state
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Active,
                    MandateStatus::Paused,
                )
                .await;
            let retry_at = mandate_retry_at(&mandate, None);
            let _ = self
                .state
                .release_mandate_review_lease(&mandate.id, lease_token, &retry_at)
                .await;
            anyhow::bail!("mandate backing goal is missing");
        };

        if goal.domain != "orchestration"
            || goal.goal_type != "continuous"
            || goal.status != "active"
        {
            let _ = self
                .state
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Active,
                    MandateStatus::Paused,
                )
                .await;
            let retry_at = mandate_retry_at(&mandate, None);
            let _ = self
                .state
                .release_mandate_review_lease(&mandate.id, lease_token, &retry_at)
                .await;
            let message = format!(
                "Mandate paused because its backing controller goal is not active: {}",
                crate::tools::sanitize::short_goal_label(&mandate.objective)
            );
            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                "mandate_paused",
                &message,
            );
            let _ = self.state.enqueue_notification(&entry).await;
            return Ok(());
        }

        // The mandate lease prevents concurrent claims. The durable open-run
        // check is the crash/lease-expiry backstop: never create a second review
        // while an earlier mandate run still owns work for this controller.
        if let Some(open_run) = self.state.get_current_goal_run(&goal.id).await? {
            let retry_at = mandate_retry_at(&mandate, Some(mandate.min_review_secs));
            let released = self
                .state
                .release_mandate_review_lease(&mandate.id, lease_token, &retry_at)
                .await?;
            info!(
                mandate_id = %mandate.id,
                goal_id = %goal.id,
                run_id = %open_run.id,
                trigger_type = %open_run.trigger_type,
                lease_released = released,
                "Deferred mandate review because the controller already has an open run"
            );
            return Ok(());
        }

        let budget_today = chrono::Utc::now().date_naive().to_string();
        if daily_budget_exhausted(
            goal.budget_daily,
            goal.tokens_used_today,
            &goal.tokens_used_day,
            &budget_today,
        ) || !daily_budget_has_run_capacity(
            goal.budget_daily,
            goal.budget_per_check,
            goal.tokens_used_today,
            &goal.tokens_used_day,
            &budget_today,
        ) {
            let retry_at = mandate_budget_retry_at(&mandate, chrono::Utc::now());
            self.state
                .release_mandate_review_lease(&mandate.id, lease_token, &retry_at)
                .await?;
            info!(
                mandate_id = %mandate.id,
                goal_id = %goal.id,
                retry_at = %retry_at,
                "Deferred mandate review because its daily goal budget cannot fund a full cycle"
            );
            return Ok(());
        }

        let goal_run_id = uuid::Uuid::new_v4().to_string();
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let task = crate::traits::Task {
            id: root_task_id,
            goal_id: goal.id.clone(),
            description: mandate_review_task_description(&mandate, &goal_run_id),
            status: "pending".to_string(),
            priority: goal.priority.clone(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: Some(mandate_review_task_context(&mandate)),
            result: None,
            error: None,
            blocker: None,
            // A mandate can authorize external side effects. Never retry the
            // entire deliberation root after an ambiguous outcome.
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        let goal_run = self
            .state
            .create_mandate_review_run(&mandate.id, lease_token, &goal_run_id, &task)
            .await?;

        info!(
            mandate_id = %mandate.id,
            goal_id = %goal.id,
            run_id = %goal_run.id,
            task_id = %task.id,
            "Enqueued mandate review root task"
        );

        // Immediate dispatch avoids the orphan-recovery grace period. If the
        // agent or worker slot is unavailable, the pending task remains durable
        // and the ordinary orphan dispatcher may claim this same root later.
        if let Some(agent) = self.agent.as_ref().and_then(Weak::upgrade) {
            let worker = format!("heartbeat-dispatch-mandate-{}", uuid::Uuid::new_v4());
            match self
                .state
                .claim_task_with_lease(&task.id, &worker, Some("profile-task-lead"), 180)
                .await
            {
                Ok(Some(_)) => {
                    if let Some(registry) = self.goal_token_registry.as_ref() {
                        registry.register(&goal.id).await;
                    }
                    crate::agent::spawn_background_task_lead(
                        agent,
                        goal.clone(),
                        task.description.clone(),
                        goal.session_id.clone(),
                        ChannelContext::internal(),
                        UserRole::Owner,
                        self.state.clone(),
                        self.hub.clone(),
                        self.goal_token_registry.clone(),
                        Some(task.id.clone()),
                        None,
                    );
                }
                Ok(None) => {
                    tracing::debug!(
                        mandate_id = %mandate.id,
                        task_id = %task.id,
                        "Mandate root remains pending because no task-lead slot was available"
                    );
                }
                Err(error) => {
                    warn!(
                        mandate_id = %mandate.id,
                        task_id = %task.id,
                        %error,
                        "Failed to claim mandate root for immediate dispatch"
                    );
                }
            }
        }

        Ok(())
    }

    /// Check for due schedules across active orchestration goals and enqueue tasks.
    ///
    /// Scheduling is per-schedule (`goal_schedules`) rather than a goal column.
    async fn check_due_goal_schedules(&self) {
        let due = match self.state.get_due_goal_schedules(50).await {
            Ok(s) => s,
            Err(e) => {
                error!(error = %e, "Failed to get due goal schedules");
                return;
            }
        };

        if due.is_empty() {
            return;
        }

        info!(count = due.len(), "Found due goal schedules");
        for schedule in due {
            let schedule_id = schedule.id.clone();
            let goal_id = schedule.goal_id.clone();
            if let Err(e) = self.fire_due_schedule(schedule).await {
                error!(
                    schedule_id = %schedule_id,
                    goal_id = %goal_id,
                    error = %e,
                    "Failed to fire due schedule"
                );
            }
        }
    }

    async fn reconcile_open_scheduled_runs(&self) {
        let goals = match self.state.get_scheduled_goals().await {
            Ok(goals) => goals,
            Err(error) => {
                error!(%error, "Failed to enumerate scheduled goals for lifecycle reconciliation");
                return;
            }
        };
        for goal in goals {
            if let Err(error) = self.reconcile_open_scheduled_runs_for_goal(&goal.id).await {
                error!(goal_id = %goal.id, %error, "Failed to reconcile scheduled goal lifecycle");
            }
        }
    }

    async fn reconcile_open_scheduled_runs_for_goal(
        &self,
        goal_id: &str,
    ) -> anyhow::Result<Vec<crate::traits::GoalRun>> {
        let candidate_open_runs = self
            .state
            .get_goal_runs(goal_id)
            .await?
            .into_iter()
            .filter(|run| matches!(run.status.as_str(), "pending" | "running" | "blocked"))
            .collect::<Vec<_>>();
        let mut open_runs = Vec::new();
        for run in candidate_open_runs {
            let run_tasks = self.state.get_tasks_for_goal_run(&run.id).await?;
            if let Some(status) = scheduled_run_reconciliation_status(&run, &run_tasks) {
                self.state
                    .finish_goal_run(
                        &run.id,
                        status,
                        Some("Reconciled from the scheduled run's authoritative task graph."),
                    )
                    .await?;
                info!(
                    goal_id,
                    run_id = %run.id,
                    prior_status = %run.status,
                    reconciled_status = status,
                    "Reconciled split-brain scheduled run from task lifecycle"
                );
                continue;
            }
            open_runs.push(run);
        }
        Ok(open_runs)
    }

    /// Launch one automatic recovery run per escalated objective once its
    /// cool-down has elapsed, up to a fixed attempt cap. The recovery run is
    /// the same typed recovery lineage the task lead creates inline
    /// (`terminal_recovery` + `recovery_for_run`), so a verified success
    /// resets the failure budget and resumes the paused schedules through
    /// `finish_goal_run`, and a failure lands in the same budget accounting.
    /// The paused schedules stay paused during recovery so the cron cannot
    /// double-fire.
    async fn launch_escalated_recoveries(&self) {
        let candidates = match self
            .state
            .list_escalated_recovery_candidates(
                Self::ESCALATED_RECOVERY_COOLDOWN_SECS,
                Self::ESCALATED_RECOVERY_MAX_ATTEMPTS,
            )
            .await
        {
            Ok(candidates) => candidates,
            Err(error) => {
                error!(%error, "Failed to list escalated recovery candidates");
                return;
            }
        };
        for recovery in candidates {
            let goal_id = recovery.goal_id.clone();
            let Ok(Some(goal)) = self.state.get_goal(&goal_id).await else {
                continue;
            };
            if goal.status != "active" {
                continue;
            }
            // Automatic recovery is admission-controlled by the owner's daily
            // budget exactly like a scheduled fire; it never spends past it.
            let budget_today = chrono::Utc::now().date_naive().to_string();
            if daily_budget_exhausted(
                goal.budget_daily,
                goal.tokens_used_today,
                &goal.tokens_used_day,
                &budget_today,
            ) || !daily_budget_has_run_capacity(
                goal.budget_daily,
                goal.budget_per_check,
                goal.tokens_used_today,
                &goal.tokens_used_day,
                &budget_today,
            ) {
                info!(
                    goal_id = %goal_id,
                    tokens_used = goal.tokens_used_today,
                    budget = ?goal.budget_daily,
                    "Deferring automatic recovery — today's goal daily budget cannot admit a run"
                );
                continue;
            }
            // An open occurrence (of any trigger type) means work is still in
            // flight or wedged; reconcile first and try again next tick.
            match self.reconcile_open_scheduled_runs_for_goal(&goal_id).await {
                Ok(open_runs) if !open_runs.is_empty() => {
                    let stale = open_runs.iter().all(|run| run.status == "blocked");
                    if !stale {
                        continue;
                    }
                    for run in &open_runs {
                        let _ = self
                            .state
                            .finish_goal_run(
                                &run.id,
                                "failed",
                                Some("Closed before automatic recovery."),
                            )
                            .await;
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    warn!(goal_id = %goal_id, %error, "Could not reconcile before recovery");
                    continue;
                }
            }
            // Bump the attempt counter before creating any work so a crash
            // between the two cannot hot-loop recovery launches.
            if let Err(error) = self.state.record_scheduled_recovery_attempt(&goal_id).await {
                warn!(goal_id = %goal_id, %error, "Failed to record recovery attempt");
                continue;
            }
            let attempt = recovery.recovery_attempts.saturating_add(1);
            let now = chrono::Utc::now().to_rfc3339();
            let recovery_task_id = uuid::Uuid::new_v4().to_string();
            let mut context = goal
                .context
                .as_deref()
                .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
                .filter(serde_json::Value::is_object)
                .unwrap_or_else(|| serde_json::json!({}));
            context["recovery_for_run"] = serde_json::json!(recovery.last_failed_run_id);
            context["terminal_recovery"] = serde_json::json!(true);
            context["recovery_attempt"] = serde_json::json!(attempt);
            context["recovery_cause"] = serde_json::json!(recovery
                .latest_failure_kind
                .map(crate::traits::ScheduledFailureKind::as_str));
            let task = crate::traits::Task {
                id: recovery_task_id.clone(),
                goal_id: goal_id.clone(),
                description: format!(
                    "Directly recover and finish: {} [SYSTEM: automatic recovery attempt {attempt} of {} after {} consecutive failed runs (typed cause {}). The schedule is paused until this run completes with verified receipts. Earlier failures recorded in the ledger are historical evidence from a previous environment, not proof that their cause persists: re-run the previously failing step once in this attempt and judge from its fresh receipt before concluding that recovery is exhausted. If the objective's repository or workspace is not present in the attempt workspace, locate it with project_inspect and bind it with scheduled_goal_runs bind_workspace before reporting a blocker.]",
                    goal.description,
                    Self::ESCALATED_RECOVERY_MAX_ATTEMPTS,
                    recovery.consecutive_failures,
                    recovery
                        .latest_failure_kind
                        .map_or("unknown", crate::traits::ScheduledFailureKind::as_str),
                ),
                status: "pending".to_string(),
                priority: "high".to_string(),
                task_order: 0,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: Some(context.to_string()),
                result: None,
                error: None,
                blocker: None,
                idempotent: false,
                retry_count: 0,
                max_retries: 0,
                created_at: now,
                started_at: None,
                completed_at: None,
            };
            if let Err(error) = self
                .state
                .start_goal_run(&goal_id, "recovery", None, Some(&recovery_task_id))
                .await
            {
                warn!(goal_id = %goal_id, %error, "Failed to start automatic recovery run");
                continue;
            }
            match self.state.create_task(&task).await {
                Ok(()) => info!(
                    goal_id = %goal_id,
                    task_id = %recovery_task_id,
                    attempt,
                    max_attempts = Self::ESCALATED_RECOVERY_MAX_ATTEMPTS,
                    "Launched automatic recovery run for escalated scheduled objective"
                ),
                Err(error) => {
                    warn!(goal_id = %goal_id, %error, "Failed to create automatic recovery task")
                }
            }
        }
    }

    async fn fire_due_schedule(&self, mut schedule: GoalSchedule) -> anyhow::Result<()> {
        // Guardrails (unknown policy/tz -> treat as coalesce/local-only).
        if schedule.tz != "local" {
            tracing::warn!(
                schedule_id = %schedule.id,
                goal_id = %schedule.goal_id,
                tz = %schedule.tz,
                "Skipping schedule with unsupported tz"
            );
            return Ok(());
        }

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();

        let Some(mut goal) = self.state.get_goal(&schedule.goal_id).await? else {
            return Ok(());
        };

        // Safety: only active orchestration goals should fire.
        if goal.domain != "orchestration" || goal.status != "active" {
            return Ok(());
        }

        // A continuous goal idle >30 days has likely stopped doing useful
        // work. We KEEP firing it (so it can recover on its own once it starts
        // succeeding again), but surface a one-time alert so a silently-dead
        // goal becomes visible instead of being skipped forever. The alert is
        // gated on `notified_at` so we don't re-alert on every tick while the
        // goal stays idle (e.g. blocked by an open task via coalescing).
        if goal.goal_type == "continuous" && goal.notified_at.is_none() {
            if let Some(ref last_action) = goal.last_useful_action {
                if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(last_action) {
                    let days_idle =
                        (chrono::Utc::now() - ts.with_timezone(&chrono::Utc)).num_days();
                    if days_idle > 30 {
                        warn!(
                            goal_id = %goal.id,
                            description = %goal.description,
                            days_idle,
                            "Continuous goal idle >30 days; alerting user and continuing to fire"
                        );
                        let msg = format!(
                            "Heads up: your recurring goal \"{}\" hasn't made progress in {} days. \
                             It's still scheduled and will keep trying — reply if you'd like to pause or cancel it.",
                            crate::tools::sanitize::short_goal_label(&goal.description),
                            days_idle
                        );
                        let entry = crate::traits::NotificationEntry::new(
                            &goal.id,
                            &goal.session_id,
                            "evergreen_alert",
                            &msg,
                        );
                        match self.state.enqueue_goal_notification(&entry).await {
                            Ok(_) => {
                                // Keep the in-memory copy consistent so the later
                                // update_goal (after task creation) doesn't clobber
                                // notified_at back to NULL and re-alert next tick.
                                goal.notified_at = Some(now_ts.clone());
                            }
                            Err(error) => {
                                warn!(
                                    goal_id = %goal.id,
                                    %error,
                                    "Failed to atomically enqueue idle-goal alert"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Only work belonging to an open goal run can apply backpressure to a
        // new scheduled firing. Historical blocked tasks are intentionally
        // retained as audit records after their run is closed; counting all
        // tasks for the goal would make those records suppress every future
        // coalesced firing forever.
        let open_runs = self
            .reconcile_open_scheduled_runs_for_goal(&goal.id)
            .await?;
        let mut open_count = 0;
        for run in &open_runs {
            open_count += self
                .state
                .get_tasks_for_goal_run(&run.id)
                .await?
                .iter()
                .filter(|task| task_blocks_later_schedule_fire(run, task))
                .count();
        }

        let fire_policy = schedule.fire_policy.as_str();
        let coalesce = fire_policy != "always_fire";
        const ALWAYS_FIRE_OPEN_TASK_CAP: usize = 3;

        // Backpressure: coalesce by default; always_fire only up to a cap.
        if (coalesce && open_count > 0) || (!coalesce && open_count >= ALWAYS_FIRE_OPEN_TASK_CAP) {
            if schedule.is_one_shot {
                // Keep the one-shot due, but avoid hot-looping while open work exists.
                schedule.next_run_at = (now + chrono::Duration::minutes(5)).to_rfc3339();
            } else if let Ok(next) = crate::cron_utils::compute_next_run(&schedule.cron_expr) {
                schedule.next_run_at = next.to_rfc3339();
            }
            schedule.updated_at = now_ts.clone();
            let _ = self.state.update_goal_schedule(&schedule).await;
            info!(
                goal_id = %goal.id,
                schedule_id = %schedule.id,
                fire_policy,
                open_run_count = open_runs.len(),
                open_task_count = open_count,
                "Deferred schedule fire because open goal-run work is still active"
            );
            return Ok(());
        }

        // Budget check: skip if *today's* daily budget is exhausted, but back off
        // schedule to avoid hot-loop. Stale prior-day usage must not count, or the
        // goal deadlocks across the day boundary (it would defer forever and never
        // fire, so its counter would never reset).
        let budget_today = now.date_naive().to_string();
        let effective_daily_budget = if let Some(configured_budget) = goal.budget_daily {
            crate::goal_tokens::load_goal_daily_budget_override(
                self.state.as_ref(),
                &goal.id,
                configured_budget,
                crate::agent::SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
            )
            .await
            .map(|value| value.budget_daily)
            .or(Some(configured_budget))
        } else {
            None
        };
        if daily_budget_exhausted(
            effective_daily_budget,
            goal.tokens_used_today,
            &goal.tokens_used_day,
            &budget_today,
        ) {
            schedule.last_run_at = Some(now_ts.clone());
            schedule.next_run_at = if schedule.is_one_shot {
                (now + chrono::Duration::hours(24)).to_rfc3339()
            } else if let Ok(next) = crate::cron_utils::compute_next_run(&schedule.cron_expr) {
                next.to_rfc3339()
            } else {
                (now + chrono::Duration::hours(24)).to_rfc3339()
            };
            schedule.updated_at = now_ts.clone();
            let _ = self.state.update_goal_schedule(&schedule).await;
            let msg = format!(
                "Skipped the scheduled run for \"{}\" because today's daily token budget \
                 ({}) is exhausted by cumulative usage across this goal's runs today, not by \
                 this unstarted run alone. No work was started. The schedule remains active and \
                 will try again at its next normal fire after the counter resets.",
                crate::tools::sanitize::short_goal_label(&goal.description),
                effective_daily_budget.unwrap_or_default(),
            );
            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                "token_alert",
                &msg,
            );
            if let Err(error) = self.state.enqueue_notification(&entry).await {
                error!(
                    goal_id = %goal.id,
                    %error,
                    "Failed to enqueue daily-budget notification"
                );
            }
            return Ok(());
        }
        if !daily_budget_has_run_capacity(
            effective_daily_budget,
            goal.budget_per_check,
            goal.tokens_used_today,
            &goal.tokens_used_day,
            &budget_today,
        ) {
            let remaining = effective_daily_budget
                .unwrap_or_default()
                .saturating_sub(goal.tokens_used_today)
                .max(0);
            let required = goal.budget_per_check.unwrap_or_default().max(0);
            schedule.last_run_at = Some(now_ts.clone());
            schedule.next_run_at = if schedule.is_one_shot {
                (now + chrono::Duration::minutes(15)).to_rfc3339()
            } else if let Ok(next) = crate::cron_utils::compute_next_run(&schedule.cron_expr) {
                next.to_rfc3339()
            } else {
                (now + chrono::Duration::hours(24)).to_rfc3339()
            };
            schedule.updated_at = now_ts.clone();
            let _ = self.state.update_goal_schedule(&schedule).await;

            let msg = format!(
                "Skipped the scheduled run for \"{}\" because only {} daily-budget tokens \
                 remain and a full run is allowed up to {}. No work was started. The schedule \
                 remains active and will try again at its next normal fire after the daily \
                 counter resets.",
                crate::tools::sanitize::short_goal_label(&goal.description),
                remaining,
                required,
            );
            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                "token_alert",
                &msg,
            );
            if let Err(error) = self.state.enqueue_notification(&entry).await {
                error!(
                    goal_id = %goal.id,
                    %error,
                    "Failed to enqueue remaining-budget notification"
                );
            }
            info!(
                goal_id = %goal.id,
                remaining,
                required,
                "Skipped scheduled run before admission because remaining daily budget cannot fund one full run"
            );
            return Ok(());
        }

        // ── Advance schedule BEFORE creating task to prevent race-condition
        //    double-fires. If task creation fails afterwards, we skip one firing
        //    (harmless) rather than risk duplicate fires from concurrent ticks.
        let advanced_next_run = if schedule.is_one_shot {
            // Park one-shot far in the future; we delete it after successful task creation.
            (now + chrono::Duration::hours(24)).to_rfc3339()
        } else if let Ok(next) = crate::cron_utils::compute_next_run(&schedule.cron_expr) {
            next.to_rfc3339()
        } else {
            warn!(schedule_id = %schedule.id, cron = %schedule.cron_expr, "Failed to compute next run for recurring schedule");
            return Ok(());
        };

        schedule.last_run_at = Some(now_ts.clone());
        schedule.next_run_at = advanced_next_run;
        schedule.updated_at = now_ts.clone();
        if let Err(e) = self.state.update_goal_schedule(&schedule).await {
            warn!(schedule_id = %schedule.id, error = %e, "Failed to advance schedule before task creation — skipping to avoid potential duplicate");
            return Ok(());
        }

        let run_root_task_id = uuid::Uuid::new_v4().to_string();
        if coalesce {
            if let Some(open_run) = self.state.get_current_goal_run(&goal.id).await? {
                let run_tasks = self.state.get_tasks_for_goal_run(&open_run.id).await?;
                if !run_tasks.is_empty() {
                    let active_or_human_blocked = run_tasks
                        .iter()
                        .any(|task| task_blocks_later_schedule_fire(&open_run, task));
                    if active_or_human_blocked {
                        warn!(
                            goal_id = %goal.id,
                            run_id = %open_run.id,
                            "Skipped schedule fire because the previous goal run is still open"
                        );
                        return Ok(());
                    }
                    let status = if run_tasks.iter().any(task_is_terminal_schedule_failure) {
                        "failed"
                    } else {
                        "completed"
                    };
                    // A blocked child of the superseded occurrence is an
                    // audit record now; mark it so it never reads as open
                    // work waiting on an owner.
                    for task in run_tasks.iter().filter(|task| task.status == "blocked") {
                        let mut superseded = task.clone();
                        superseded.status = "superseded".to_string();
                        superseded.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        superseded.result = Some(format!(
                            "Superseded by the next scheduled occurrence. Prior blocker: {}",
                            task.blocker.as_deref().unwrap_or("none")
                        ));
                        if let Err(error) = self.state.update_task(&superseded).await {
                            warn!(task_id = %task.id, %error, "Failed to supersede blocked task");
                        }
                    }
                    let _ = self
                        .state
                        .finish_goal_run(
                            &open_run.id,
                            status,
                            Some("Closed before the next scheduled firing."),
                        )
                        .await?;
                }
            }
        }
        let goal_run = self
            .state
            .start_goal_run(
                &goal.id,
                "scheduled",
                Some(&schedule.id),
                Some(&run_root_task_id),
            )
            .await?;

        // ── Plain-reminder fast path: the only deliverable is a message back
        // to the user, so send it now instead of going through task creation
        // and the task-lead pipeline (which adds a minute-plus of dispatch and
        // LLM latency). Falls through to the normal pipeline if the message
        // can't be delivered.
        if let Some(reminder) = crate::reminders::parse_reminder(&goal.description) {
            let fire_text = crate::reminders::fire_message(&reminder);
            let delivered = if let Some(hub_arc) = self.hub.as_ref().and_then(|w| w.upgrade()) {
                hub_arc
                    .send_text(&goal.session_id, &fire_text)
                    .await
                    .is_ok()
            } else {
                false
            };
            if delivered {
                // Record a completed task so history/diagnostics show the run.
                let task = crate::traits::Task {
                    id: run_root_task_id.clone(),
                    goal_id: goal.id.clone(),
                    description: format!("Deliver reminder: {}", goal.description),
                    status: "completed".to_string(),
                    priority: "medium".to_string(),
                    task_order: 0,
                    parallel_group: None,
                    depends_on: None,
                    agent_id: Some("reminder-fast-path".to_string()),
                    context: None,
                    result: Some(fire_text),
                    error: None,
                    blocker: None,
                    idempotent: false,
                    retry_count: 0,
                    max_retries: 1,
                    created_at: now_ts.clone(),
                    started_at: Some(now_ts.clone()),
                    completed_at: Some(now_ts.clone()),
                };
                self.state.create_task(&task).await?;
                let _ = self
                    .state
                    .finish_goal_run(
                        &goal_run.id,
                        "completed",
                        Some("Reminder delivered successfully."),
                    )
                    .await?;

                let mut updated = goal.clone();
                if schedule.is_one_shot {
                    updated.status = "completed".to_string();
                    updated.completed_at = Some(now_ts.clone());
                    // The reminder itself is the notification — suppress the
                    // generic "Goal completed" follow-up.
                    updated.notified_at = Some(now_ts.clone());
                }
                updated.last_useful_action = Some(now_ts.clone());
                updated.updated_at = now_ts.clone();
                let _ = self.state.update_goal(&updated).await;

                if schedule.is_one_shot {
                    let _ = self.state.delete_goal_schedule(&schedule.id).await;
                }
                info!(
                    goal_id = %goal.id,
                    one_shot = schedule.is_one_shot,
                    "Delivered plain reminder via fast path"
                );
                return Ok(());
            }
            warn!(
                goal_id = %goal.id,
                "Reminder fast path could not deliver; falling back to task pipeline"
            );
        }

        // Create a pending task for this scheduled run.
        let task = crate::traits::Task {
            id: run_root_task_id,
            goal_id: goal.id.clone(),
            description: if schedule.is_one_shot || goal.goal_type == "finite" {
                format!(
                    "Execute scheduled goal: {} [SYSTEM: already scheduled and firing now; do not reschedule.]",
                    goal.description
                )
            } else {
                format!(
                    "Scheduled check: {} [SYSTEM: already scheduled and firing now; do not reschedule.]",
                    goal.description
                )
            },
            status: "pending".to_string(),
            priority: if goal.goal_type == "continuous" {
                "low".to_string()
            } else {
                "medium".to_string()
            },
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: goal.context.clone(),
            result: None,
            error: None,
            blocker: None,
            // A scheduled root can perform externally visible writes. Retrying
            // the whole orchestration after an ambiguous failure can duplicate
            // those writes, so only individual, explicitly idempotent child
            // tasks may be retried.
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: now_ts.clone(),
            started_at: None,
            completed_at: None,
        };

        self.state.create_task(&task).await?;

        // Update goal timestamp.
        let mut updated_goal = goal.clone();
        updated_goal.last_useful_action = Some(now_ts.clone());
        updated_goal.updated_at = now_ts.clone();
        if let Err(e) = self.state.update_goal(&updated_goal).await {
            warn!(goal_id = %goal.id, error = %e, "Failed to update goal timestamp after schedule fire");
        }

        // Clean up one-shot schedules after successful task creation.
        // (Recurring schedules were already advanced before task creation.)
        if schedule.is_one_shot {
            if let Err(e) = self.state.delete_goal_schedule(&schedule.id).await {
                warn!(schedule_id = %schedule.id, error = %e, "Failed to delete one-shot schedule after fire");
            }
        }

        info!(
            goal_id = %goal.id,
            schedule_id = %schedule.id,
            task_id = %task.id,
            "Enqueued scheduled task"
        );

        // Notify user that the scheduled goal is executing (DMs only —
        // group channels just get the results without progress noise).
        // Post the "Running scheduled task" announcement as a TRACKED surface so
        // the task lead's progress heartbeat can edit it in place — one self-
        // updating message instead of a separate announcement + progress stream.
        let mut running_surface_id: Option<String> = None;
        let local_hour = chrono::Timelike::hour(&chrono::Local::now());
        if !is_group_session(&goal.session_id)
            && crate::traits::NotificationEntry::routine_delivery_allowed_at_local_hour(local_hour)
        {
            if let Some(hub_weak) = &self.hub {
                if let Some(hub_arc) = hub_weak.upgrade() {
                    let short_desc = crate::tools::sanitize::short_goal_label(&goal.description);
                    running_surface_id = hub_arc
                        .send_text_tracked(
                            &goal.session_id,
                            &format!("⏳ **Scheduled run in progress**\n\n{}", short_desc),
                        )
                        .await
                        .ok()
                        .flatten();
                }
            }
        }

        // Dispatch immediately instead of waiting for the orphan-recovery pass
        // (which only picks up tasks older than 60s, so every scheduled run
        // used to start at least a minute late). If no agent is available the
        // orphan dispatcher still picks the task up on a later tick.
        if let Some(agent_arc) = self.agent.as_ref().and_then(|w| w.upgrade()) {
            let agent_id = format!("heartbeat-schedule-fire-{}", uuid::Uuid::new_v4());
            match self
                .state
                .claim_task_with_lease(&task.id, &agent_id, Some("profile-task-lead"), 180)
                .await
            {
                Ok(Some(_)) => {
                    if let Some(ref registry) = self.goal_token_registry {
                        registry.register(&goal.id).await;
                    }
                    crate::agent::spawn_background_task_lead(
                        agent_arc,
                        goal.clone(),
                        task.description.clone(),
                        goal.session_id.clone(),
                        ChannelContext::internal(),
                        UserRole::Owner,
                        self.state.clone(),
                        self.hub.clone(),
                        self.goal_token_registry.clone(),
                        Some(task.id.clone()),
                        running_surface_id,
                    );
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(
                        task_id = %task.id,
                        error = %e,
                        "Failed to claim scheduled task for immediate dispatch"
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{Goal, Mandate, MandateAuthority, Task, TaskActivity};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn synthetic_task(desc: &str, status: &str) -> Task {
        Task {
            id: format!("task-{desc}"),
            goal_id: "goal-1".to_string(),
            description: desc.to_string(),
            status: status.to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: String::new(),
            started_at: None,
            completed_at: None,
        }
    }

    #[test]
    fn scheduled_run_reconciliation_terminalizes_split_brain_task_graphs() {
        let mut run = crate::traits::GoalRun::new("goal-1", "default", "scheduled");
        run.status = "running".to_string();

        assert_eq!(
            scheduled_run_reconciliation_status(
                &run,
                &[
                    synthetic_task("root", "completed"),
                    synthetic_task("child", "blocked")
                ],
            ),
            Some("failed")
        );
        assert_eq!(
            scheduled_run_reconciliation_status(&run, &[synthetic_task("root", "completed")],),
            Some("completed")
        );
        assert_eq!(
            scheduled_run_reconciliation_status(
                &run,
                &[
                    synthetic_task("root", "completed"),
                    synthetic_task("child", "running")
                ],
            ),
            None
        );
    }

    #[test]
    fn terminal_run_tasks_never_apply_schedule_backpressure() {
        let mut run = crate::traits::GoalRun::new("goal-1", "default", "scheduled");
        run.status = "completed".to_string();
        let mut task = synthetic_task("child", "blocked");
        task.blocker = Some("Historical unresolved child".to_string());

        assert!(!task_blocks_later_schedule_fire(&run, &task));
    }

    #[test]
    fn only_open_manual_run_can_reactivate_stalled_pending_work() {
        let pending = synthetic_task("manual root", "pending");
        let candidates = vec![&pending];
        let mut run = crate::traits::GoalRun::new("goal-1", "default", "manual");

        assert!(stranded_manual_run_matches_pending_tasks(
            &run,
            std::slice::from_ref(&pending),
            &candidates,
        ));

        run.trigger_type = "scheduled".to_string();
        assert!(!stranded_manual_run_matches_pending_tasks(
            &run,
            std::slice::from_ref(&pending),
            &candidates,
        ));

        run.trigger_type = "manual".to_string();
        run.status = "completed".to_string();
        assert!(!stranded_manual_run_matches_pending_tasks(
            &run,
            std::slice::from_ref(&pending),
            &candidates,
        ));

        let other = synthetic_task("different task", "pending");
        run.status = "pending".to_string();
        assert!(!stranded_manual_run_matches_pending_tasks(
            &run,
            std::slice::from_ref(&other),
            &candidates,
        ));
    }

    #[test]
    fn scheduled_blocker_only_retires_the_prior_scheduled_occurrence() {
        let blocked = synthetic_task("scheduled blocker", "blocked");
        let mut provider_blocked = blocked.clone();
        provider_blocked.blocker = Some(
            "LLM error: Codex stream failed: Our servers are currently overloaded. Please try again later."
                .to_string(),
        );
        let pending = synthetic_task("still in flight", "pending");
        let mut scheduled_run = crate::traits::GoalRun::new("goal-1", "default", "scheduled");
        scheduled_run.status = "blocked".to_string();
        let mut manual_run = crate::traits::GoalRun::new("goal-1", "default", "manual");
        manual_run.status = "blocked".to_string();

        assert!(!task_blocks_later_schedule_fire(&scheduled_run, &blocked,));
        // A blocked manual or recovery occurrence is equally terminal for
        // backpressure: it must never suppress the recurring schedule forever.
        assert!(!task_blocks_later_schedule_fire(&manual_run, &blocked));
        let mut recovery_run = crate::traits::GoalRun::new("goal-1", "default", "recovery");
        recovery_run.status = "blocked".to_string();
        assert!(!task_blocks_later_schedule_fire(&recovery_run, &blocked));
        assert!(!task_blocks_later_schedule_fire(
            &scheduled_run,
            &provider_blocked,
        ));
        assert!(!task_blocks_later_schedule_fire(
            &manual_run,
            &provider_blocked,
        ));
        // A still-running manual occurrence keeps its backpressure.
        manual_run.status = "running".to_string();
        assert!(task_blocks_later_schedule_fire(&manual_run, &blocked));
        assert_eq!(
            scheduled_run_reconciliation_status(
                &manual_run,
                &[synthetic_task("manual root", "completed")]
            ),
            Some("completed")
        );
        assert!(!task_blocks_later_schedule_fire(&scheduled_run, &pending,));

        scheduled_run.status = "running".to_string();
        assert!(task_blocks_later_schedule_fire(&scheduled_run, &blocked));
        assert!(task_blocks_later_schedule_fire(&scheduled_run, &pending));
        assert!(task_is_terminal_schedule_failure(&blocked));
    }

    fn due_mandate_controller(session_id: &str) -> (Goal, Mandate) {
        let goal = Goal::new_continuous(
            "Steward an account autonomously",
            session_id,
            Some(10_000),
            Some(100_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Maintain a useful and authentic account presence",
            session_id,
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        (goal, mandate)
    }

    #[tokio::test]
    async fn due_mandate_creates_a_distinct_run_without_a_goal_schedule() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                Arc::new(EmbeddingService::new().unwrap()),
            )
            .await
            .unwrap(),
        );
        let (goal, mandate) = due_mandate_controller("owner-session");
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.check_due_mandates().await;

        let run = state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("mandate review should create an open run");
        assert_eq!(run.trigger_type, "mandate");
        assert!(run.schedule_id.is_none());
        let tasks = state.get_tasks_for_goal_run(&run.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(run.root_task_id.as_deref(), Some(tasks[0].id.as_str()));
        assert!(tasks[0]
            .description
            .starts_with("Run one bounded autonomous review"));
        assert!(tasks[0]
            .description
            .contains("isolated built-in mandate protocol"));
        assert!(!tasks[0].description.contains(&mandate.objective));
        assert!(!tasks[0].description.contains("allowed tools"));
        let context: serde_json::Value =
            serde_json::from_str(tasks[0].context.as_deref().expect("minimal fence context"))
                .unwrap();
        assert_eq!(context["mandate_id"], mandate.id);
        assert_eq!(context["mandate_version"], mandate.version);
        assert_eq!(context["provenance"], "runtime_mandate_fence_only");
        assert_eq!(context.as_object().unwrap().len(), 3);
        assert!(state
            .get_schedules_for_goal(&goal.id)
            .await
            .unwrap()
            .is_empty());
        assert!(state
            .get_scheduled_run_state(&goal.id)
            .await
            .unwrap()
            .is_none());
        assert!(state
            .get_mandate(&mandate.id)
            .await
            .unwrap()
            .unwrap()
            .review_lease_token
            .is_some());
    }

    #[tokio::test]
    async fn mandate_budget_deferral_waits_for_the_next_utc_reset() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                Arc::new(EmbeddingService::new().unwrap()),
            )
            .await
            .unwrap(),
        );
        let (mut goal, mandate) = due_mandate_controller("owner-session");
        goal.tokens_used_today = 95_000;
        goal.tokens_used_day = chrono::Utc::now().date_naive().to_string();
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let before = chrono::Utc::now();
        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.check_due_mandates().await;

        assert!(state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .is_none());
        let deferred = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert!(deferred.review_lease_token.is_none());
        let retry_at = chrono::DateTime::parse_from_rfc3339(&deferred.next_review_at)
            .unwrap()
            .with_timezone(&chrono::Utc);
        let expected_midnight = chrono::DateTime::<Utc>::from_naive_utc_and_offset(
            before
                .date_naive()
                .succ_opt()
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            Utc,
        );
        assert_eq!(retry_at, expected_midnight);
    }

    #[tokio::test]
    async fn open_mandate_run_prevents_a_duplicate_review_after_lease_release() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                Arc::new(EmbeddingService::new().unwrap()),
            )
            .await
            .unwrap(),
        );
        let (goal, mandate) = due_mandate_controller("owner-session");
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.check_due_mandates().await;
        let first_run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        let claimed = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        let lease_token = claimed.review_lease_token.as_deref().unwrap();
        assert!(state
            .release_mandate_review_lease(
                &mandate.id,
                lease_token,
                &(chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339(),
            )
            .await
            .unwrap());

        coordinator.check_due_mandates().await;

        let still_open = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        assert_eq!(still_open.id, first_run.id);
        assert_eq!(state.get_goal_runs(&goal.id).await.unwrap().len(), 1);
        assert_eq!(
            state
                .get_tasks_for_goal_run(&first_run.id)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn generic_orphan_dispatch_leaves_non_root_mandate_tasks_untouched() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                Arc::new(EmbeddingService::new().unwrap()),
            )
            .await
            .unwrap(),
        );
        let (goal, mandate) = due_mandate_controller("owner-session");
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.check_due_mandates().await;
        let run = state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("mandate run");
        let root_id = run.root_task_id.clone().expect("mandate root");
        let mut root = state.get_task(&root_id).await.unwrap().expect("root task");
        root.status = "running".to_string();
        root.started_at = Some((chrono::Utc::now() - chrono::Duration::minutes(20)).to_rfc3339());
        state.update_task(&root).await.unwrap();

        let non_root = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Model-authored exact action task".to_string(),
            status: "pending".to_string(),
            priority: "high".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: (chrono::Utc::now() - chrono::Duration::minutes(5)).to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&non_root).await.unwrap();
        assert!(state
            .get_tasks_for_goal_run(&run.id)
            .await
            .unwrap()
            .iter()
            .any(|task| task.id == non_root.id));

        coordinator.dispatch_pending_tasks().await;

        let after = state
            .get_task(&non_root.id)
            .await
            .unwrap()
            .expect("non-root task remains");
        assert_eq!(after.status, "pending");
        assert!(after.agent_id.is_none());
        assert!(after.started_at.is_none());
        assert!(state
            .get_current_task_attempt(&non_root.id)
            .await
            .unwrap()
            .is_none());
        assert!(
            state
                .get_pending_notifications(10)
                .await
                .unwrap()
                .is_empty(),
            "generic orphan recovery must not announce or claim mandate child work"
        );
    }

    #[test]
    fn goal_completion_header_is_honest_about_incomplete_tasks() {
        // Live repro (goal df0925b4, 2026-07-02): the watchdog killed the task
        // lead mid-run, no partial_success metadata was written, and the user
        // received "Goal completed:" over a summary that itself said "2/4
        // tasks completed" — two steps silently missing. The header must be
        // derived from the actual task rows, not from metadata a dead lead
        // never wrote.
        let tasks = vec![
            synthetic_task("check disk space", "completed"),
            synthetic_task("count rs files", "completed"),
            synthetic_task("find 3 largest files", "pending"),
            synthetic_task("write summary report", "pending"),
        ];
        let header = goal_completion_header(&tasks);
        assert!(
            header.starts_with("Goal partially completed (2/4 tasks)"),
            "got: {header}"
        );
        assert!(header.contains("find 3 largest files"));
        assert!(header.contains("write summary report"));

        // All tasks done → plain completion header.
        let done = vec![
            synthetic_task("a", "completed"),
            synthetic_task("b", "completed"),
        ];
        assert_eq!(goal_completion_header(&done), "Goal completed:");

        // No task rows at all (goal completed without decomposition) → plain.
        assert_eq!(goal_completion_header(&[]), "Goal completed:");
    }

    #[test]
    fn daily_budget_only_blocks_on_current_day_usage() {
        let today = "2026-06-28";
        // Over budget, usage recorded today -> exhausted.
        assert!(daily_budget_exhausted(Some(500_000), 964_666, today, today));
        // Over budget, but usage is from a PRIOR day (reset hasn't run) -> NOT
        // exhausted, so the goal can fire and reset instead of deadlocking.
        assert!(!daily_budget_exhausted(
            Some(500_000),
            964_666,
            "2026-06-27",
            today
        ));
        // Under budget today -> not exhausted.
        assert!(!daily_budget_exhausted(
            Some(500_000),
            100_000,
            today,
            today
        ));
        // No daily budget configured -> never exhausted.
        assert!(!daily_budget_exhausted(None, 999_999, today, today));

        assert!(!daily_budget_has_run_capacity(
            Some(1_000_000),
            Some(400_000),
            798_965,
            today,
            today,
        ));
        assert!(daily_budget_has_run_capacity(
            Some(1_000_000),
            Some(400_000),
            598_965,
            today,
            today,
        ));
        assert!(daily_budget_has_run_capacity(
            Some(1_000_000),
            Some(400_000),
            999_999,
            "2026-06-27",
            today,
        ));
    }

    #[test]
    fn mandate_sqlite_contention_uses_a_short_expiry_bounded_retry() {
        assert!(is_sqlite_busy_error(&anyhow::anyhow!(
            "error returned from database: (code: 5) database is locked"
        )));
        assert!(!is_sqlite_busy_error(&anyhow::anyhow!(
            "controller goal is missing"
        )));

        let now = chrono::DateTime::parse_from_rfc3339("2026-08-04T05:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let goal = crate::traits::Goal::new_continuous("retry", "owner", None, None);
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "retry transient contention",
            "owner",
            crate::traits::MandateAuthority::default(),
            1_800,
            86_400,
            1_800,
        );
        mandate.expires_at = Some("2026-08-04T05:00:20Z".to_string());
        assert_eq!(
            mandate_transient_retry_at(&mandate, now),
            "2026-08-04T05:00:20+00:00"
        );
    }

    async fn test_state_store() -> Arc<dyn StateStore> {
        // Persist the temp DB file for the test process's lifetime. If the
        // NamedTempFile were dropped here, the path would be unlinked and a
        // later (lazily-opened) pool connection would re-create an empty
        // database — surfacing as "no such table" under parallel test load on
        // CI. keep() leaks the file, which is fine for short-lived test runs.
        let (_db_file, db_path) = tempfile::NamedTempFile::new().unwrap().keep().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        Arc::new(
            SqliteStateStore::new(db_path.to_str().unwrap(), 100, None, embedding_service)
                .await
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn coordinator_defaults_task_inactivity_timeout_to_300() {
        let (_tx, rx) = mpsc::channel::<()>(1);
        let state = test_state_store().await;
        let hb = HeartbeatCoordinator::new(state, 5, 2, rx, None, None, None);
        assert_eq!(hb.task_inactivity_timeout_secs, 300);
    }

    #[tokio::test]
    async fn coordinator_set_task_inactivity_timeout_updates_field() {
        let (_tx, rx) = mpsc::channel::<()>(1);
        let state = test_state_store().await;
        let mut hb = HeartbeatCoordinator::new(state, 5, 2, rx, None, None, None);
        hb.set_task_inactivity_timeout(777);
        assert_eq!(hb.task_inactivity_timeout_secs, 777);
    }

    #[tokio::test]
    async fn test_new_continuous_goal() {
        let goal = Goal::new_continuous("Test continuous goal", "system", Some(5000), Some(20000));
        assert_eq!(goal.domain, "orchestration");
        assert_eq!(goal.goal_type, "continuous");
        assert_eq!(goal.status, "active");
        assert_eq!(goal.priority, "low");
        assert_eq!(goal.budget_per_check, Some(5000));
        assert_eq!(goal.budget_daily, Some(20000));
        assert_eq!(goal.tokens_used_today, 0);
        assert!(goal.last_useful_action.is_none());
        assert_eq!(goal.session_id, "system");
    }

    #[tokio::test]
    async fn test_heartbeat_job_fires() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let (wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let mut coordinator = HeartbeatCoordinator::new(state, 1, 3, wake_rx, None, None, None);

        coordinator.register_job("test_job", Duration::from_secs(0), move || {
            let c = counter_clone.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        });

        // Tick once
        coordinator.tick().await.unwrap();

        // Give the spawned task time to execute
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(
            counter.load(Ordering::SeqCst) >= 1,
            "Job should have fired at least once"
        );

        drop(wake_tx); // Keep sender alive until here
    }

    async fn reminder_test_setup() -> (
        Arc<dyn StateStore>,
        Arc<crate::channels::ChannelHub>,
        Arc<crate::testing::TestChannel>,
        HeartbeatCoordinator,
        tempfile::NamedTempFile,
    ) {
        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let channel = Arc::new(crate::testing::TestChannel::new());
        let session_map: crate::channels::SessionMap = Arc::new(tokio::sync::RwLock::new(
            HashMap::from([("test_session".to_string(), "test".to_string())]),
        ));
        let hub = Arc::new(crate::channels::ChannelHub::new(
            vec![channel.clone() as Arc<dyn crate::traits::Channel>],
            session_map,
        ));
        let outbound: Arc<dyn OutboundRouter> = hub.clone();
        let coordinator = HeartbeatCoordinator::new(
            state.clone(),
            1,
            3,
            wake_rx,
            Some(Arc::downgrade(&outbound)),
            None,
            None,
        );
        (state, hub, channel, coordinator, db_file)
    }

    fn due_schedule(goal_id: &str, cron_expr: &str, is_one_shot: bool) -> GoalSchedule {
        let now = chrono::Utc::now().to_rfc3339();
        GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            cron_expr: cron_expr.to_string(),
            tz: "local".to_string(),
            original_schedule: None,
            fire_policy: "coalesce".to_string(),
            is_one_shot,
            is_paused: false,
            last_run_at: None,
            next_run_at: now.clone(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    #[tokio::test]
    async fn test_reminder_fast_path_one_shot_delivers_and_completes() {
        let (state, _hub, channel, coordinator, _db_file) = reminder_test_setup().await;

        let mut goal = Goal::new_deferred_finite("Remind me to call my daughter", "test_session");
        goal.status = "active".to_string();
        state.create_goal(&goal).await.unwrap();
        let schedule = due_schedule(&goal.id, "46 13 11 6 *", true);
        state.create_goal_schedule(&schedule).await.unwrap();

        coordinator
            .fire_due_schedule(schedule.clone())
            .await
            .unwrap();

        // The reminder itself is the only message — no "Running scheduled
        // task", no progress updates.
        let msgs = channel.messages_for("test_session").await;
        assert_eq!(msgs, vec!["⏰ Reminder: call your daughter".to_string()]);

        // Goal completed with notified_at set (suppresses the generic
        // "Goal completed" notification) and the one-shot schedule removed.
        let g = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(g.status, "completed");
        assert!(g.notified_at.is_some());
        assert!(state
            .get_schedules_for_goal(&goal.id)
            .await
            .unwrap()
            .is_empty());

        // A completed task records the run; nothing is left pending for the
        // task-lead pipeline.
        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status, "completed");
    }

    #[tokio::test]
    async fn test_reminder_fast_path_recurring_keeps_goal_active() {
        let (state, _hub, channel, coordinator, _db_file) = reminder_test_setup().await;

        let mut goal =
            Goal::new_continuous("Remind me to take my meds", "test_session", None, None);
        goal.status = "active".to_string();
        state.create_goal(&goal).await.unwrap();
        let schedule = due_schedule(&goal.id, "0 9 * * *", false);
        state.create_goal_schedule(&schedule).await.unwrap();

        coordinator
            .fire_due_schedule(schedule.clone())
            .await
            .unwrap();

        let msgs = channel.messages_for("test_session").await;
        assert_eq!(msgs, vec!["⏰ Reminder: take your meds".to_string()]);

        // Recurring reminders stay active with their schedule intact.
        let g = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(g.status, "active");
        assert_eq!(
            state.get_schedules_for_goal(&goal.id).await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn test_non_reminder_schedule_creates_pending_task() {
        let (state, _hub, channel, coordinator, _db_file) = reminder_test_setup().await;

        let mut goal = Goal::new_deferred_finite("Check the deploy status", "test_session");
        goal.status = "active".to_string();
        state.create_goal(&goal).await.unwrap();
        let schedule = due_schedule(&goal.id, "46 13 11 6 *", true);
        state.create_goal_schedule(&schedule).await.unwrap();

        coordinator
            .fire_due_schedule(schedule.clone())
            .await
            .unwrap();

        // Non-reminder goals keep the normal pipeline: a task is created and
        // the user is told the scheduled task is running.
        let msgs = channel.messages_for("test_session").await;
        let local_hour = chrono::Timelike::hour(&chrono::Local::now());
        if crate::traits::NotificationEntry::routine_delivery_allowed_at_local_hour(local_hour) {
            assert_eq!(msgs.len(), 1);
            assert!(msgs[0].starts_with("⏳ **Scheduled run in progress**"));
            assert!(msgs[0].contains("Check the deploy status"));
        } else {
            assert!(msgs.is_empty(), "routine progress must respect quiet hours");
        }
        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        // No agent wired in this test, so the task stays pending for the
        // orphan dispatcher.
        assert_eq!(tasks[0].status, "pending");
    }

    #[tokio::test]
    async fn test_heartbeat_job_respects_interval() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let mut coordinator = HeartbeatCoordinator::new(state, 1, 3, wake_rx, None, None, None);

        coordinator.register_job("test_job", Duration::from_secs(3600), move || {
            let c = counter_clone.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        });

        // Tick twice rapidly
        coordinator.tick().await.unwrap();
        coordinator.tick().await.unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Should only have fired once (interval is 1 hour)
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "Job should have fired exactly once due to 1h interval"
        );
    }

    #[tokio::test]
    async fn test_heartbeat_telemetry_tracks_failures_and_recovery() {
        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let telemetry = Arc::new(HeartbeatTelemetry::new());
        let mut coordinator =
            HeartbeatCoordinator::new(state, 1, 3, wake_rx, None, None, Some(telemetry.clone()));

        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_clone = attempts.clone();
        coordinator.register_job("test_job", Duration::from_secs(0), move || {
            let a = attempts_clone.clone();
            async move {
                let n = a.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    anyhow::bail!("first run fails");
                }
                Ok(())
            }
        });

        coordinator.tick().await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let first = telemetry.snapshots();
        let first_job = first
            .iter()
            .find(|j| j.name == "test_job")
            .expect("telemetry row should exist");
        assert_eq!(first_job.consecutive_failures, 1);
        assert!(first_job.last_error.is_some());
        assert!(first_job.last_run_at.is_some());

        coordinator.tick().await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let second = telemetry.snapshots();
        let second_job = second
            .iter()
            .find(|j| j.name == "test_job")
            .expect("telemetry row should exist");
        assert_eq!(second_job.consecutive_failures, 0);
        assert!(second_job.last_error.is_none());
        assert!(second_job.last_success_at.is_some());
    }

    #[tokio::test]
    async fn test_due_one_shot_schedule_creates_task_and_deletes_schedule() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let goal = Goal::new_finite("Send deployment reminder", "session-1");
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: true,
            is_paused: false,
            last_run_at: None,
            next_run_at: due_ts,
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1, "One execution task should be created");
        assert_eq!(tasks[0].status, "pending");
        assert!(
            tasks[0].description.starts_with("Execute scheduled goal:"),
            "Task description should indicate scheduled execution"
        );
        assert!(
            !tasks[0].idempotent,
            "scheduled orchestration roots must never auto-retry after ambiguous writes"
        );
        assert_eq!(tasks[0].max_retries, 0);

        let sched = state.get_goal_schedule(&schedule.id).await.unwrap();
        assert!(
            sched.is_none(),
            "One-shot schedules should be deleted after firing"
        );
    }

    /// A continuous goal idle >30 days must KEEP firing (not be silently
    /// skipped forever) and must surface a one-time alert to the user.
    #[tokio::test]
    async fn test_idle_continuous_goal_keeps_firing_and_alerts_user() {
        let state = test_state_store().await;

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();

        let mut goal =
            Goal::new_continuous("Post daily tweets", "session-1", Some(5000), Some(500_000));
        // Idle for 40 days — well past the 30-day auto-retire threshold.
        goal.last_useful_action = Some((now - chrono::Duration::days(40)).to_rfc3339());
        state.create_goal(&goal).await.unwrap();

        let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("0 9 * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: due_ts,
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        // Keeps firing: a task is created instead of being silently skipped.
        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(
            tasks.len(),
            1,
            "Idle continuous goal should still fire (create a task), not be skipped"
        );
        assert_eq!(tasks[0].status, "pending");

        // Surfaces a one-time alert so the idle goal is no longer silent.
        let notifs = state.get_pending_notifications(10).await.unwrap();
        let alerts: Vec<_> = notifs
            .iter()
            .filter(|n| n.notification_type == "evergreen_alert" && n.goal_id == goal.id)
            .collect();
        assert_eq!(
            alerts.len(),
            1,
            "Idle continuous goal should enqueue exactly one evergreen_alert notification"
        );
    }

    /// The idle alert must fire at most once per idle episode, even if the
    /// goal stays idle across many ticks (e.g. an open task keeps coalescing
    /// the fire so `last_useful_action` never resets).
    #[tokio::test]
    async fn test_idle_continuous_goal_alerts_only_once() {
        let state = test_state_store().await;

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();

        let mut goal =
            Goal::new_continuous("Post daily tweets", "session-1", Some(5000), Some(500_000));
        goal.last_useful_action = Some((now - chrono::Duration::days(40)).to_rfc3339());
        state.create_goal(&goal).await.unwrap();

        // A pre-existing open task makes the fire coalesce (back off) every
        // time, so `last_useful_action` never resets and the goal stays idle.
        let open_task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "in-flight work".to_string(),
            status: "pending".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: now_ts.clone(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&open_task).await.unwrap();

        let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: due_ts,
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);

        // Fire repeatedly while the goal stays idle.
        coordinator
            .fire_due_schedule(schedule.clone())
            .await
            .unwrap();
        coordinator
            .fire_due_schedule(schedule.clone())
            .await
            .unwrap();
        coordinator
            .fire_due_schedule(schedule.clone())
            .await
            .unwrap();

        let alerts = state
            .get_pending_notifications(20)
            .await
            .unwrap()
            .into_iter()
            .filter(|n| n.notification_type == "evergreen_alert" && n.goal_id == goal.id)
            .count();
        assert_eq!(
            alerts, 1,
            "Idle alert must be sent at most once, not every tick"
        );

        // Coalescing is still respected: no new execution tasks were created
        // while the open task is in flight.
        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(
            tasks.len(),
            1,
            "No new task should be created while coalescing"
        );
    }

    #[tokio::test]
    async fn test_coalesce_policy_one_shot_backs_off_when_open_task_exists() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let goal = Goal::new_finite("Coalesce test", "session-1");
        state.create_goal(&goal).await.unwrap();

        // Existing open task should block coalesced firing.
        let existing_task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Existing work".to_string(),
            status: "running".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: Some("agent-1".to_string()),
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: Some(chrono::Utc::now().to_rfc3339()),
            completed_at: None,
        };
        state.create_task(&existing_task).await.unwrap();

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: true,
            is_paused: false,
            last_run_at: None,
            next_run_at: due_ts,
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        // No new tasks should be created when coalescing and open work exists.
        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, existing_task.id);

        // One-shot schedules should back off instead of hot-looping.
        let updated_sched = state
            .get_goal_schedule(&schedule.id)
            .await
            .unwrap()
            .expect("schedule should still exist");
        let next = chrono::DateTime::parse_from_rfc3339(&updated_sched.next_run_at).unwrap();
        assert!(next.with_timezone(&chrono::Utc) > now);
    }

    #[tokio::test]
    async fn test_closed_run_blocker_does_not_block_next_schedule_fire() {
        let state = test_state_store().await;
        let goal = Goal::new_continuous("Daily blog", "session-1", None, None);
        state.create_goal(&goal).await.unwrap();

        let historical_run = state
            .start_goal_run(&goal.id, "manual", None, None)
            .await
            .unwrap();
        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        let historical_blocker = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Old deployment attempt".to_string(),
            status: "blocked".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: Some("Deployment credentials were unavailable".to_string()),
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: now_ts.clone(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&historical_blocker).await.unwrap();
        state
            .finish_goal_run(
                &historical_run.id,
                "failed",
                Some("Archived failed deployment run"),
            )
            .await
            .unwrap();

        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: (now - chrono::Duration::minutes(2)).to_rfc3339(),
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().any(|task| task.id == historical_blocker.id));
        assert!(
            tasks
                .iter()
                .any(|task| task.description.starts_with("Scheduled check:")),
            "a blocker retained in a closed run must not suppress future scheduled work"
        );
    }

    #[tokio::test]
    async fn test_blocked_scheduled_run_does_not_coalesce_later_occurrence() {
        let state = test_state_store().await;
        let goal = Goal::new_continuous("Daily synthetic journal", "session-1", None, None);
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: Some((now - chrono::Duration::minutes(2)).to_rfc3339()),
            next_run_at: (now - chrono::Duration::minutes(1)).to_rfc3339(),
            created_at: now_ts.clone(),
            updated_at: now_ts.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let blocked_run = state
            .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
            .await
            .unwrap();
        let blocked_task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Review the latest synthetic repository work".to_string(),
            status: "blocked".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: Some("No verifiable new material was available for this run".to_string()),
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: now_ts.clone(),
            started_at: Some(now_ts.clone()),
            completed_at: Some(now_ts),
        };
        state.create_task(&blocked_task).await.unwrap();
        state
            .finish_goal_run(
                &blocked_run.id,
                "blocked",
                Some("This scheduled occurrence could not proceed."),
            )
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.fire_due_schedule(schedule).await.unwrap();

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().any(|task| task.id == blocked_task.id));
        assert!(tasks
            .iter()
            .any(|task| task.description.starts_with("Scheduled check:")));

        let runs = state.get_goal_runs(&goal.id).await.unwrap();
        let retired = runs
            .iter()
            .find(|run| run.id == blocked_run.id)
            .expect("the prior run remains available as history");
        assert_eq!(retired.status, "failed");
        assert!(retired.completed_at.is_some());
        assert_ne!(
            state
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .expect("the later occurrence has an open run")
                .id,
            blocked_run.id
        );
    }

    #[tokio::test]
    async fn scheduled_lifecycle_reconciles_before_the_next_fire_is_due() {
        let state = test_state_store().await;
        let goal = Goal::new_continuous("Synthetic future schedule", "session-1", None, None);
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("0 6 * * *".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: Some(now_ts.clone()),
                next_run_at: (now + chrono::Duration::days(1)).to_rfc3339(),
                created_at: now_ts.clone(),
                updated_at: now_ts.clone(),
            })
            .await
            .unwrap();
        let run = state
            .start_goal_run(&goal.id, "scheduled", None, None)
            .await
            .unwrap();
        let completed_root = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Synthetic scheduled root".to_string(),
            status: "completed".to_string(),
            priority: "normal".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some("root finished".to_string()),
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: now_ts.clone(),
            started_at: Some(now_ts.clone()),
            completed_at: Some(now_ts.clone()),
        };
        state.create_task(&completed_root).await.unwrap();
        let blocked_child = Task {
            id: uuid::Uuid::new_v4().to_string(),
            description: "Synthetic blocked child".to_string(),
            status: "blocked".to_string(),
            task_order: 1,
            result: None,
            blocker: Some("Typed terminal blocker".to_string()),
            ..completed_root
        };
        state.create_task(&blocked_child).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.reconcile_open_scheduled_runs().await;

        let reconciled = state
            .get_goal_runs(&goal.id)
            .await
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.id == run.id)
            .unwrap();
        assert_eq!(reconciled.status, "failed");
        assert!(reconciled.completed_at.is_some());
    }

    #[tokio::test]
    async fn escalated_goal_gets_one_automatic_recovery_run_whose_root_survives_dispatch() {
        let (_db_file, db_path) = tempfile::NamedTempFile::new().unwrap().keep().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let sqlite = Arc::new(
            SqliteStateStore::new(db_path.to_str().unwrap(), 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = sqlite.pool();
        let state: Arc<dyn StateStore> = sqlite.clone();
        let goal = Goal::new_continuous(
            "Publish the synthetic daily digest",
            "session-1",
            None,
            None,
        );
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now();
        let escalated_at = (now - chrono::Duration::hours(12)).to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("0 6 * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: now.to_rfc3339(),
            created_at: escalated_at.clone(),
            updated_at: escalated_at.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();
        let failed_run = state
            .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
            .await
            .unwrap();
        state
            .finish_goal_run(&failed_run.id, "failed", Some("synthetic"))
            .await
            .unwrap();
        // Escalate through the store's own failure accounting so the paused
        // schedule is owned by the recovery machine.
        for _ in 0..3 {
            let run = state
                .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
                .await
                .unwrap();
            state
                .finish_goal_run(&run.id, "failed", Some("synthetic"))
                .await
                .unwrap();
        }
        let recovery = state
            .get_scheduled_recovery_state(&goal.id)
            .await
            .unwrap()
            .expect("recovery state");
        assert_eq!(
            recovery.disposition,
            crate::traits::ScheduledRecoveryDisposition::Escalated
        );
        // Age the escalation past the cool-down.
        sqlx::query("UPDATE scheduled_recovery_state SET updated_at = ? WHERE goal_id = ?")
            .bind(&escalated_at)
            .bind(&goal.id)
            .execute(&pool)
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        let runs = state.get_goal_runs(&goal.id).await.unwrap();
        let recovery_run = runs
            .iter()
            .find(|run| run.trigger_type == "recovery")
            .expect("automatic recovery run");
        assert!(matches!(
            recovery_run.status.as_str(),
            "pending" | "running"
        ));
        let root = state
            .get_task(recovery_run.root_task_id.as_deref().expect("root task"))
            .await
            .unwrap()
            .expect("root task row");
        assert_ne!(
            root.status, "cancelled",
            "recovery root must not be retired as stale"
        );
        assert!(root
            .description
            .contains("automatic recovery attempt 1 of 3"));
        let state_after = state
            .get_scheduled_recovery_state(&goal.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(state_after.recovery_attempts, 1);

        // A second tick inside the cool-down launches nothing more.
        coordinator.tick().await.unwrap();
        let runs = state.get_goal_runs(&goal.id).await.unwrap();
        assert_eq!(
            runs.iter()
                .filter(|run| run.trigger_type == "recovery")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn test_open_run_blocker_coalesces_next_schedule_fire() {
        let state = test_state_store().await;
        let goal = Goal::new_continuous("Daily tweet", "session-1", None, None);
        state.create_goal(&goal).await.unwrap();

        let open_run = state
            .start_goal_run(&goal.id, "manual", None, None)
            .await
            .unwrap();
        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        let active_blocker = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Await editorial approval".to_string(),
            status: "blocked".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: Some("Waiting for a user decision".to_string()),
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: now_ts.clone(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&active_blocker).await.unwrap();

        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: (now - chrono::Duration::minutes(2)).to_rfc3339(),
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, active_blocker.id);
        assert_eq!(
            state
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .expect("blocked run should remain open")
                .id,
            open_run.id
        );

        let updated_schedule = state
            .get_goal_schedule(&schedule.id)
            .await
            .unwrap()
            .expect("recurring schedule should remain active");
        assert!(updated_schedule.last_run_at.is_none());
        assert!(
            chrono::DateTime::parse_from_rfc3339(&updated_schedule.next_run_at)
                .unwrap()
                .with_timezone(&chrono::Utc)
                > now
        );
    }

    #[tokio::test]
    async fn test_provider_overload_blocker_does_not_coalesce_next_schedule_fire() {
        let state = test_state_store().await;
        let goal = Goal::new_continuous("Daily synthetic report", "session-1", None, None);
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: (now - chrono::Duration::minutes(2)).to_rfc3339(),
            created_at: now_ts.clone(),
            updated_at: now_ts.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let overloaded_run = state
            .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
            .await
            .unwrap();
        let overloaded_task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Generate the scheduled report".to_string(),
            status: "blocked".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: Some("synthetic-worker".to_string()),
            context: None,
            result: None,
            error: None,
            blocker: Some(
                "LLM error: Codex stream failed: Our servers are currently overloaded. Please try again later."
                    .to_string(),
            ),
            idempotent: true,
            retry_count: 1,
            max_retries: 1,
            created_at: now_ts.clone(),
            started_at: Some(now_ts.clone()),
            completed_at: Some(now_ts),
        };
        state.create_task(&overloaded_task).await.unwrap();
        state
            .finish_goal_run(
                &overloaded_run.id,
                "blocked",
                Some("The model service was overloaded."),
            )
            .await
            .unwrap();
        let blocked_before_fire = state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("the overloaded scheduled run should still be open");
        assert_eq!(blocked_before_fire.id, overloaded_run.id);
        assert_eq!(blocked_before_fire.status, "blocked");
        assert!(blocked_before_fire.completed_at.is_none());

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        let runs = state.get_goal_runs(&goal.id).await.unwrap();
        let prior_run = runs
            .iter()
            .find(|run| run.id == overloaded_run.id)
            .expect("the overloaded run remains in history");
        assert_eq!(prior_run.status, "failed");
        assert!(prior_run.completed_at.is_some());

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().any(|task| task.id == overloaded_task.id));
        assert!(
            tasks
                .iter()
                .any(|task| task.description.starts_with("Scheduled check:")),
            "a provider-overload blocker from the prior run must not suppress the next scheduled run"
        );
    }

    #[tokio::test]
    async fn test_pending_task_with_interrupted_dependency_does_not_block_next_schedule_fire() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let goal = Goal::new_continuous("Daily dependency cleanup test", "session-1", None, None);
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        let dependency = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Fetch prerequisite data".to_string(),
            status: "interrupted".to_string(),
            priority: "low".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: Some("agent-1".to_string()),
            context: None,
            result: None,
            error: Some("interrupted".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 1,
            max_retries: 1,
            created_at: (now - chrono::Duration::minutes(10)).to_rfc3339(),
            started_at: Some((now - chrono::Duration::minutes(9)).to_rfc3339()),
            completed_at: Some((now - chrono::Duration::minutes(5)).to_rfc3339()),
        };
        state.create_task(&dependency).await.unwrap();

        let dependent = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Analyze prerequisite data".to_string(),
            status: "pending".to_string(),
            priority: "low".to_string(),
            task_order: 2,
            parallel_group: None,
            depends_on: Some(serde_json::json!([dependency.id.clone()]).to_string()),
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: (now - chrono::Duration::minutes(9)).to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&dependent).await.unwrap();

        let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: due_ts,
            created_at: now_ts.clone(),
            updated_at: now_ts,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        let dependent_after = tasks
            .iter()
            .find(|t| t.id == dependent.id)
            .expect("dependent task should still exist");
        assert_eq!(dependent_after.status, "blocked");

        assert!(
            tasks
                .iter()
                .any(|t| t.description.starts_with("Scheduled check:")),
            "new scheduled run should be enqueued after unfulfillable pending dependency is blocked"
        );
    }

    #[tokio::test]
    async fn test_multiple_due_schedules_always_fire_enqueues_multiple_tasks() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let goal = Goal::new_continuous("Take medicine", "session-1", Some(5000), Some(20000));
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();
        let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();

        for _ in 0..3 {
            let schedule = GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "* * * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("* * * * *".to_string()),
                fire_policy: "always_fire".to_string(),
                is_one_shot: true,
                is_paused: false,
                last_run_at: None,
                next_run_at: due_ts.clone(),
                created_at: now_ts.clone(),
                updated_at: now_ts.clone(),
            };
            state.create_goal_schedule(&schedule).await.unwrap();
        }

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(
            tasks.len(),
            3,
            "always_fire should enqueue multiple due runs"
        );

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(
            schedules.is_empty(),
            "one-shot schedules should be deleted after firing"
        );
    }

    #[tokio::test]
    async fn test_daily_budget_reset() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let sqlite_state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let state: Arc<dyn StateStore> = sqlite_state.clone();

        // Create a continuous goal with some tokens used
        let goal = Goal::new_continuous("Test budget goal", "system", Some(5000), Some(20000));
        state.create_goal(&goal).await.unwrap();

        // Manually set prior-day usage via the concrete pool.
        sqlx::query(
            "UPDATE goals
             SET tokens_used_today = 1500, tokens_used_day = date('now', '-1 day')
             WHERE id = ?",
        )
        .bind(&goal.id)
        .execute(&sqlite_state.pool())
        .await
        .unwrap();

        // Reset
        let count = state.reset_daily_token_budgets().await.unwrap();
        assert!(count >= 1, "Should have reset at least one goal");

        // Verify it's 0 now
        let updated = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(
            updated.tokens_used_today, 0,
            "tokens_used_today should be reset to 0"
        );
    }

    #[tokio::test]
    async fn test_daily_budget_reset_preserves_same_day_usage() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let sqlite_state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let state: Arc<dyn StateStore> = sqlite_state.clone();
        let goal = Goal::new_continuous("Same-day budget", "system", Some(5000), Some(20000));
        state.create_goal(&goal).await.unwrap();
        sqlx::query(
            "UPDATE goals
             SET tokens_used_today = 1500, tokens_used_day = date('now')
             WHERE id = ?",
        )
        .bind(&goal.id)
        .execute(&sqlite_state.pool())
        .await
        .unwrap();

        let count = state.reset_daily_token_budgets().await.unwrap();
        assert_eq!(count, 0);
        let updated = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(updated.tokens_used_today, 1500);
    }

    #[tokio::test]
    async fn test_dispatch_no_agent_reverts_claim() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        // Create an active goal with a pending task (no running tasks = orphaned)
        let goal = Goal::new_finite("Build website", "session-1");
        state.create_goal(&goal).await.unwrap();

        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Deploy to production".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: (chrono::Utc::now() - chrono::Duration::seconds(120)).to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&task).await.unwrap();

        // Create coordinator with NO agent reference
        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator = HeartbeatCoordinator::new(
            state.clone(),
            60,
            3,
            wake_rx,
            None, // no hub
            None, // no goal_token_registry
            None, // no telemetry
        );
        // agent is None by default — dispatch will fail to spawn

        coordinator.dispatch_pending_tasks().await;

        // Task must be back to "pending", NOT stranded in "claimed"
        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(
            tasks[0].status, "pending",
            "Task should be reverted to pending when no agent is available"
        );
        assert!(
            tasks[0].agent_id.is_none(),
            "agent_id should be cleared on revert"
        );
        assert!(
            tasks[0].started_at.is_none(),
            "started_at should be cleared on revert"
        );

        // A stalled notification should have been enqueued
        let notifications = state.get_pending_notifications(10).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].notification_type, "stalled");
        assert_eq!(notifications[0].goal_id, goal.id);
    }

    #[tokio::test]
    async fn heartbeat_recovers_stranded_manual_run_before_dispatch() {
        let state = test_state_store().await;
        let mut goal = Goal::new_continuous(
            "Publish one diary entry",
            "session-1",
            Some(5_000),
            Some(20_000),
        );
        goal.status = "stalled".to_string();
        state.create_goal(&goal).await.unwrap();

        let root = synthetic_task("manual root", "pending");
        let mut root = Task {
            goal_id: goal.id.clone(),
            created_at: (chrono::Utc::now() - chrono::Duration::minutes(15)).to_rfc3339(),
            ..root
        };
        root.id = uuid::Uuid::new_v4().to_string();
        let run = state
            .start_goal_run(&goal.id, "manual", None, Some(&root.id))
            .await
            .unwrap();
        state.create_task(&root).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.dispatch_pending_tasks().await;

        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
        assert_eq!(
            state.get_tasks_for_goal(&goal.id).await.unwrap()[0].status,
            "pending"
        );
        assert_eq!(state.get_goal_runs(&goal.id).await.unwrap().len(), 1);
        assert_eq!(state.get_goal_runs(&goal.id).await.unwrap()[0].id, run.id);

        let notifications = state.get_pending_notifications(10).await.unwrap();
        assert!(notifications
            .iter()
            .all(|entry| entry.notification_type != "status_update"));
        assert!(notifications
            .iter()
            .any(|entry| entry.notification_type == "stalled"));
    }

    #[tokio::test]
    async fn active_task_lead_owns_retries_and_pending_dispatch() {
        let state = test_state_store().await;
        let goal = Goal::new_finite("Build and deploy a site", "session-1");
        state.create_goal(&goal).await.unwrap();
        let old = (chrono::Utc::now() - chrono::Duration::seconds(120)).to_rfc3339();
        let failed = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Research visual direction".to_string(),
            status: "failed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some("Research was incorporated directly.".to_string()),
            error: Some("Late persistence race".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: old.clone(),
            started_at: None,
            completed_at: Some(old.clone()),
        };
        let pending = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Return the deployment URL".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: old,
            started_at: None,
            completed_at: None,
        };
        state.create_task(&failed).await.unwrap();
        state.create_task(&pending).await.unwrap();

        let registry = GoalTokenRegistry::new();
        let _active_run = registry
            .try_acquire_run(&goal.id)
            .expect("test task lead should own run");
        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, Some(registry), None);

        coordinator.auto_retry_failed_tasks().await;
        coordinator.dispatch_pending_tasks().await;

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(
            tasks
                .iter()
                .find(|task| task.id == failed.id)
                .unwrap()
                .status,
            "failed"
        );
        assert_eq!(
            tasks
                .iter()
                .find(|task| task.id == pending.id)
                .unwrap()
                .status,
            "pending"
        );
        assert!(state
            .get_pending_notifications(10)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn closed_scheduled_run_children_are_neither_retried_nor_dispatched() {
        let state = test_state_store().await;
        let goal = Goal::new_continuous("Publish a daily post", "session-1", None, None);
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now();
        let cycle_start = now - chrono::Duration::hours(2);
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 8 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: None,
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: Some(cycle_start.to_rfc3339()),
                next_run_at: (now + chrono::Duration::hours(22)).to_rfc3339(),
                created_at: cycle_start.to_rfc3339(),
                updated_at: cycle_start.to_rfc3339(),
            })
            .await
            .unwrap();

        let failed = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Deploy the post".to_string(),
            status: "failed".to_string(),
            priority: "medium".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: Some("ambiguous deploy result".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: (cycle_start + chrono::Duration::minutes(5)).to_rfc3339(),
            started_at: None,
            completed_at: Some((cycle_start + chrono::Duration::minutes(6)).to_rfc3339()),
        };
        state.create_task(&failed).await.unwrap();
        let pending = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Verify the old deployment".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 2,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: (cycle_start + chrono::Duration::minutes(7)).to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&pending).await.unwrap();
        let run = state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("task creation should bind both children to one implicit run");
        state
            .finish_goal_run(&run.id, "failed", Some("synthetic closed run"))
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.auto_retry_failed_tasks().await;
        coordinator.dispatch_pending_tasks().await;

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(
            tasks
                .iter()
                .find(|task| task.id == failed.id)
                .unwrap()
                .status,
            "failed",
            "closed-run failures must not be promoted back to pending"
        );
        let pending_after = tasks.iter().find(|task| task.id == pending.id).unwrap();
        assert_eq!(pending_after.status, "cancelled");
        assert!(!pending_after.idempotent);
        assert_eq!(pending_after.max_retries, 0);
    }

    #[test]
    fn mutating_http_method_detection() {
        assert!(is_mutating_http_method(r#"{"method":"POST","url":"x"}"#));
        assert!(is_mutating_http_method(r#"{"method":"post","url":"x"}"#));
        assert!(is_mutating_http_method(r#"{"method":"DELETE","url":"x"}"#));
        assert!(!is_mutating_http_method(r#"{"method":"GET","url":"x"}"#));
        assert!(!is_mutating_http_method(r#"{"method":"HEAD","url":"x"}"#));
        // Missing method / malformed JSON must fail closed (not mutating),
        // never block dispatch on a parse miss.
        assert!(!is_mutating_http_method(r#"{"url":"x"}"#));
        assert!(!is_mutating_http_method("not json"));
        assert!(!is_mutating_http_method(""));
    }

    /// Live repro (goal 9a744834, 2026-07-04): a scheduled "post one tweet"
    /// goal posted 5 real duplicate tweets in one cycle because orphaned
    /// pending tasks kept getting redispatched after the tweet had already
    /// posted successfully (downstream verification/task-tracking hiccups
    /// made it look unfinished). This is the regression test for the fix:
    /// dispatch_pending_tasks must close out orphaned tasks instead of
    /// redispatching once a successful mutating http_request is logged for
    /// the goal's current cycle.
    #[tokio::test]
    async fn test_dispatch_skips_redispatch_after_mutating_success_this_cycle() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let goal = Goal::new_finite("Post one tweet", "session-1");
        state.create_goal(&goal).await.unwrap();

        // Schedule fired 10 minutes ago — that's the current cycle's start.
        let cycle_start = chrono::Utc::now() - chrono::Duration::minutes(10);
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: None,
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: Some(cycle_start.to_rfc3339()),
            next_run_at: (chrono::Utc::now() + chrono::Duration::hours(23)).to_rfc3339(),
            created_at: cycle_start.to_rfc3339(),
            updated_at: cycle_start.to_rfc3339(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        // The task lead's original task already posted the tweet successfully.
        let posted_task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Scheduled check: Post one short tweet".to_string(),
            status: "interrupted".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: (cycle_start + chrono::Duration::minutes(1)).to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&posted_task).await.unwrap();
        state
            .log_task_activity(&TaskActivity {
                id: 0,
                task_id: posted_task.id.clone(),
                activity_type: "tool_call".to_string(),
                tool_name: Some("http_request".to_string()),
                tool_args: Some(
                    r#"{"method":"POST","url":"https://api.x.com/2/tweets"}"#.to_string(),
                ),
                result: Some("HTTP 201 Created".to_string()),
                success: Some(true),
                tokens_used: None,
                created_at: (cycle_start + chrono::Duration::minutes(2)).to_rfc3339(),
            })
            .await
            .unwrap();

        // A leftover subtask the orchestrator created before things got
        // confused, now orphaned (pending, >60s old, nothing running).
        let orphaned = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Post the tweet".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: (chrono::Utc::now() - chrono::Duration::seconds(120)).to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&orphaned).await.unwrap();
        state
            .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
                goal_id: goal.id.clone(),
                root_task_id: posted_task.id.clone(),
                effective_budget_per_check: 5000,
                tokens_used: 0,
                budget_extensions_count: 0,
                health: Default::default(),
                created_at: cycle_start.to_rfc3339(),
                updated_at: cycle_start.to_rfc3339(),
            })
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        // agent is None — if the gate didn't fire, this would fall through to
        // the "no agent available" revert-to-pending + stalled-notification path.
        coordinator.dispatch_pending_tasks().await;

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        let posted_after = tasks.iter().find(|t| t.id == posted_task.id).unwrap();
        assert_eq!(
            posted_after.status, "completed",
            "a successful mutation receipt should reconcile its interrupted task"
        );
        let orphaned_after = tasks.iter().find(|t| t.id == orphaned.id).unwrap();
        assert_eq!(
            orphaned_after.status, "completed",
            "orphaned task should be closed out, not redispatched, once a \
             mutating success is already logged for this cycle"
        );
        assert!(
            state
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .is_none(),
            "a fully reconciled run must not remain open after orphan closeout"
        );
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "completed"
        );

        let notifications = state.get_pending_notifications(10).await.unwrap();
        assert!(
            notifications
                .iter()
                .all(|n| n.notification_type != "stalled"),
            "should not fall through to the stalled-notification path once the \
             idempotency gate has closed out the orphaned task"
        );
    }

    /// A successful mutation in an earlier run is not a receipt for a new
    /// explicit manual run. This reproduces task 30e0c819 (2026-08-01):
    /// `trigger_now` created a new run without changing the cron schedule's
    /// `last_run_at`, and goal-wide timestamp matching auto-closed the new task
    /// from the previous run's tweet receipt.
    #[tokio::test]
    async fn test_manual_run_is_not_suppressed_by_prior_run_mutating_success() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let goal = Goal::new_continuous("Post one useful tweet", "session-1", None, None);
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now();
        let prior_cycle = now - chrono::Duration::minutes(15);
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: None,
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: Some(prior_cycle.to_rfc3339()),
            next_run_at: (now + chrono::Duration::hours(20)).to_rfc3339(),
            created_at: prior_cycle.to_rfc3339(),
            updated_at: prior_cycle.to_rfc3339(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let first_run = state
            .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
            .await
            .unwrap();
        let first_task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Scheduled check: Post one useful tweet".to_string(),
            status: "completed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some("Posted and verified.".to_string()),
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: prior_cycle.to_rfc3339(),
            started_at: Some(prior_cycle.to_rfc3339()),
            completed_at: Some((prior_cycle + chrono::Duration::minutes(1)).to_rfc3339()),
        };
        state.create_task(&first_task).await.unwrap();
        state
            .log_task_activity(&TaskActivity {
                id: 0,
                task_id: first_task.id.clone(),
                activity_type: "tool_call".to_string(),
                tool_name: Some("http_request".to_string()),
                tool_args: Some(
                    r#"{"method":"POST","url":"https://api.x.com/2/tweets"}"#.to_string(),
                ),
                result: Some("HTTP 201 Created".to_string()),
                success: Some(true),
                tokens_used: None,
                created_at: (prior_cycle + chrono::Duration::minutes(1)).to_rfc3339(),
            })
            .await
            .unwrap();
        state
            .finish_goal_run(&first_run.id, "completed", Some("First tweet posted."))
            .await
            .unwrap();

        let second_run = state
            .start_goal_run(&goal.id, "manual", None, None)
            .await
            .unwrap();
        let pending = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Manual scheduled run: Post one useful tweet".to_string(),
            status: "pending".to_string(),
            priority: "high".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: (now - chrono::Duration::minutes(2)).to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&pending).await.unwrap();
        state
            .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
                goal_id: goal.id.clone(),
                root_task_id: pending.id.clone(),
                effective_budget_per_check: 5000,
                tokens_used: 0,
                budget_extensions_count: 0,
                health: Default::default(),
                created_at: (now - chrono::Duration::minutes(3)).to_rfc3339(),
                updated_at: (now - chrono::Duration::minutes(3)).to_rfc3339(),
            })
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.dispatch_pending_tasks().await;

        let pending_after = state.get_task(&pending.id).await.unwrap().unwrap();
        assert_ne!(
            pending_after.status, "completed",
            "the previous run's mutation receipt must not auto-close a new manual run"
        );
        assert!(!pending_after
            .result
            .as_deref()
            .is_some_and(|result| result.contains("mutating action")));
        assert_eq!(
            state
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .unwrap()
                .id,
            second_run.id,
            "the new manual run must remain eligible for dispatch"
        );
    }

    /// Control for the test above: without any logged mutating success, the
    /// existing orphan-dispatch behavior (revert to pending + stalled
    /// notification, since there's no agent) must still happen — the new
    /// gate must not fire on vibes.
    #[tokio::test]
    async fn test_dispatch_still_redispatches_without_prior_mutating_success() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let goal = Goal::new_finite("Post one tweet", "session-1");
        state.create_goal(&goal).await.unwrap();

        let cycle_start = chrono::Utc::now() - chrono::Duration::minutes(10);
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: None,
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: Some(cycle_start.to_rfc3339()),
            next_run_at: (chrono::Utc::now() + chrono::Duration::hours(23)).to_rfc3339(),
            created_at: cycle_start.to_rfc3339(),
            updated_at: cycle_start.to_rfc3339(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let orphaned = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Post the tweet".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: (chrono::Utc::now() - chrono::Duration::seconds(120)).to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&orphaned).await.unwrap();
        state
            .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
                goal_id: goal.id.clone(),
                root_task_id: orphaned.id.clone(),
                effective_budget_per_check: 5000,
                tokens_used: 0,
                budget_extensions_count: 0,
                health: Default::default(),
                created_at: cycle_start.to_rfc3339(),
                updated_at: cycle_start.to_rfc3339(),
            })
            .await
            .unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.dispatch_pending_tasks().await;

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        let orphaned_after = tasks.iter().find(|t| t.id == orphaned.id).unwrap();
        assert_eq!(
            orphaned_after.status, "pending",
            "without a logged mutating success, normal orphan-dispatch behavior \
             (revert to pending since there's no agent) must be unchanged"
        );

        let notifications = state.get_pending_notifications(10).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].notification_type, "stalled");
    }

    #[tokio::test]
    async fn test_deferred_finite_goal_fires() {
        // Deprecated: deferred finite goals are now represented as one-shot goal_schedules.
        // This behavior is covered by test_due_one_shot_schedule_creates_task_and_deletes_schedule().
    }

    #[test]
    fn task_inactivity_secs_uses_last_activity_then_started_at() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-06-23T00:10:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        // Last activity 2 min ago → 120s.
        assert_eq!(
            task_inactivity_secs(Some("2026-06-23T00:08:00Z"), "2026-06-23T00:00:00Z", now),
            120
        );
        // No activity → falls back to started_at (10 min ago → 600s).
        assert_eq!(task_inactivity_secs(None, "2026-06-23T00:00:00Z", now), 600);
        // Unparseable inputs → 0 (never panics).
        assert_eq!(task_inactivity_secs(Some("garbage"), "garbage", now), 0);
        // SQLite-datetime format ('YYYY-MM-DD HH:MM:SS', the stored task_activity format)
        // must parse identically — last activity 5 min ago → 300s.
        assert_eq!(
            task_inactivity_secs(Some("2026-06-23 00:05:00"), "2026-06-23 00:00:00", now),
            300
        );
    }

    #[test]
    fn task_escalations_wait_for_same_turn_state_reconciliation() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-07-31T16:00:10Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(!task_escalation_has_settled("2026-07-31T16:00:05Z", now));
        assert!(task_escalation_has_settled("2026-07-31T16:00:01Z", now));
        assert!(task_escalation_has_settled("legacy timestamp", now));
    }

    #[tokio::test]
    async fn test_stale_pending_confirmation_cleanup() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );

        let mut goal = Goal::new_deferred_finite("Remind me tomorrow", "session-1");
        goal.created_at = (chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
        goal.updated_at = goal.created_at.clone();
        state.create_goal(&goal).await.unwrap();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 12 2 *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("0 9 12 2 *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: true,
            is_paused: false,
            last_run_at: None,
            next_run_at: (chrono::Utc::now() + chrono::Duration::hours(2)).to_rfc3339(),
            created_at: goal.created_at.clone(),
            updated_at: goal.updated_at.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
        let mut coordinator =
            HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
        coordinator.tick().await.unwrap();

        let updated = state
            .get_goal(&goal.id)
            .await
            .unwrap()
            .expect("goal should exist");
        assert_eq!(
            updated.status, "cancelled",
            "Stale pending_confirmation goal should be auto-cancelled"
        );

        // Stale pending-confirmation cleanup should also remove schedules.
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(schedules.is_empty());
    }
}
