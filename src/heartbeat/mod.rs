use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use aidaemon_core::{
    CancellationToken, FnService, Schedule, Service, ServiceContext, ServiceError, ServiceEvent,
    ServiceGate, ServiceHost, ServiceObserver,
};
use async_trait::async_trait;

use chrono::Utc;
use serde::Serialize;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::agent::{build_goal_task_results_summary, is_group_session, Agent};
use crate::goal_tokens::GoalTokenRegistry;
use crate::runtime_ports::OutboundRouter;
use crate::traits::{GoalSchedule, Mandate, MandateStatus, StateStore};
use crate::types::{ChannelContext, UserRole};

const TASK_ESCALATION_SETTLE_SECS: i64 = 8;
/// History prefix for notices delivered from the notification queue; matches
/// the direct goal-notification path so both read the same to the agent.
const QUEUED_NOTIFICATION_NOTE_PREFIX: &str = "Parent-visible result (goal notification)";

/// Notices after which the owner's next reply may answer a mandate question:
/// the ASK itself, and the timeout notice that says it can still be answered.
fn notice_accepts_mandate_answer(notification_type: &str) -> bool {
    matches!(notification_type, "mandate_ask" | "mandate_ask_timeout")
}
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
    /// The async function to call.
    pub run: HeartbeatRunFn,
}

/// The daemon's background work: its periodic jobs, and the goal cycle
/// (schedules, mandates, stuck tasks, notifications), run on a
/// [`ServiceHost`].
pub struct HeartbeatCoordinator {
    jobs: Vec<HeartbeatJob>,
    services: Vec<Arc<dyn Service>>,
    state: Arc<dyn StateStore>,
    max_concurrent: usize,
    tick_interval: Duration,
    wake_rx: Option<mpsc::Receiver<()>>,
    hub: Option<Weak<dyn OutboundRouter>>,
    goal_token_registry: Option<GoalTokenRegistry>,
    telemetry: Option<Arc<HeartbeatTelemetry>>,
    agent: Option<Weak<Agent>>,
    last_stale_goal_cleanup: Option<Instant>,
    /// Seconds of no task activity before `detect_stuck_tasks` interrupts a
    /// running/claimed task. Defaults to 300; overridden from config via
    /// `set_task_inactivity_timeout`.
    task_inactivity_timeout_secs: i64,
}

/// Skips a whole tick while the database cannot answer.
struct DbHealthGate {
    state: Arc<dyn StateStore>,
    healthy: AtomicBool,
}

#[async_trait]
impl ServiceGate for DbHealthGate {
    async fn ready(&self) -> bool {
        match self.state.health_check().await {
            Ok(_) => {
                if !self.healthy.swap(true, Ordering::Relaxed) {
                    info!("DB health restored");
                }
                true
            }
            Err(e) => {
                if self.healthy.swap(false, Ordering::Relaxed) {
                    error!(error = %e, "DB health check failed — skipping tick");
                }
                false
            }
        }
    }
}

/// Logs service outcomes and feeds the dashboard's job table.
struct JobObserver {
    telemetry: Option<Arc<HeartbeatTelemetry>>,
}

impl ServiceObserver for JobObserver {
    fn event(&self, event: ServiceEvent) {
        let telemetry = self.telemetry.as_ref();
        match event {
            ServiceEvent::Launched { service } => {
                tracing::debug!(job = %service, "Heartbeat job starting");
                if let Some(t) = telemetry {
                    t.mark_started(&service);
                }
            }
            ServiceEvent::Succeeded {
                service,
                previous_failures,
            } => {
                if previous_failures > 0 {
                    info!(job = %service, prev_failures = previous_failures, "Heartbeat job recovered");
                }
                if let Some(t) = telemetry {
                    t.mark_success(&service);
                }
            }
            ServiceEvent::Failed {
                service,
                failures,
                error,
            } => {
                error!(
                    job = %service,
                    error = %error,
                    consecutive_failures = failures,
                    "Heartbeat job failed — backing off"
                );
                if let Some(t) = telemetry {
                    t.mark_failure(&service, failures, error);
                }
            }
            ServiceEvent::StartFailed { service, error } => {
                error!(job = %service, error = %error, "Heartbeat service failed to start");
            }
            ServiceEvent::Finished { service } => {
                if let Some(t) = telemetry {
                    t.mark_not_running(&service);
                }
            }
        }
    }
}

/// The goal cycle as one service: its phases depend on their order.
struct GoalCycle {
    coordinator: tokio::sync::Mutex<HeartbeatCoordinator>,
}

#[async_trait]
impl Service for GoalCycle {
    fn name(&self) -> &str {
        "goal_cycle"
    }

    fn schedule(&self) -> Schedule {
        Schedule::EveryTick
    }

    async fn start(&self, _ctx: &ServiceContext) -> Result<(), ServiceError> {
        self.coordinator.lock().await.startup_recovery().await;
        Ok(())
    }

    async fn run(&self, _ctx: &ServiceContext) -> Result<(), ServiceError> {
        self.coordinator
            .lock()
            .await
            .tick()
            .await
            .map_err(|e| ServiceError::new(e.to_string()))
    }
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
            services: Vec::new(),
            state,
            max_concurrent,
            tick_interval: Duration::from_secs(tick_interval_secs),
            wake_rx: Some(wake_rx),
            hub,
            goal_token_registry,
            telemetry,
            agent: None,
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
            run: Box::new(move || Box::pin(f())),
        });
        if let Some(telemetry) = &self.telemetry {
            telemetry.register_job(name, interval);
        }
    }

    #[cfg(test)]
    pub(crate) fn scheduled_names(&self) -> Vec<String> {
        self.jobs
            .iter()
            .map(|job| job.name.clone())
            .chain(self.services.iter().map(|s| s.name().to_string()))
            .collect()
    }

    /// Run `service` alongside the jobs, under the same gate and telemetry.
    pub fn add_service(&mut self, service: Arc<dyn Service>) {
        if let (Some(telemetry), Schedule::Every(interval)) = (&self.telemetry, service.schedule())
        {
            telemetry.register_job(service.name(), interval);
        }
        self.services.push(service);
    }

    /// The host that runs this coordinator's jobs and goal cycle.
    pub(crate) fn into_host(mut self) -> ServiceHost {
        let mut host = ServiceHost::new(self.tick_interval)
            .max_concurrent(self.max_concurrent)
            .gate(Arc::new(DbHealthGate {
                state: self.state.clone(),
                healthy: AtomicBool::new(true),
            }))
            .observer(Arc::new(JobObserver {
                telemetry: self.telemetry.clone(),
            }));
        for job in std::mem::take(&mut self.jobs) {
            let HeartbeatJob {
                name,
                interval,
                run,
            } = job;
            let service = FnService::new(name, Schedule::Every(interval), move || {
                let run = run();
                async move { run.await.map_err(|e| ServiceError::new(e.to_string())) }
            });
            host = host.service(Arc::new(service));
        }
        for service in std::mem::take(&mut self.services) {
            host = host.service(service);
        }
        // A user message wakes the cycle at once.
        if let Some(mut wake_rx) = self.wake_rx.take() {
            let waker = host.waker();
            tokio::spawn(async move {
                while wake_rx.recv().await.is_some() {
                    waker.wake();
                }
            });
        }
        host.service(Arc::new(GoalCycle {
            coordinator: tokio::sync::Mutex::new(self),
        }))
    }

    /// Run in the background until `shutdown` is cancelled.
    pub fn start(self, shutdown: CancellationToken) -> tokio::task::JoinHandle<()> {
        tokio::spawn(self.into_host().run(shutdown))
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

    /// One pass of the goal cycle. Periodic jobs and the database check are
    /// the host's.
    pub(crate) async fn tick(&mut self) -> anyhow::Result<()> {
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
        if let Some(agent) = self.agent.as_ref().and_then(Weak::upgrade) {
            let dispatched = agent.retry_deferred_provider_requests().await;
            if dispatched > 0 {
                info!(dispatched, "Re-ran requests deferred by a provider outage");
            }
        }

        Ok(())
    }
}

mod mandates;
mod notifications;
mod schedules;
mod tasks;
#[cfg(test)]
mod tests;
