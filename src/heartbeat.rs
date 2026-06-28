use std::collections::HashMap;
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
use crate::channels::ChannelHub;
use crate::goal_tokens::GoalTokenRegistry;
use crate::traits::{GoalSchedule, StateStore};
use crate::types::{ChannelContext, UserRole};

fn is_scheduled_task_description(text: &str) -> bool {
    let trimmed = text.trim_start().to_ascii_lowercase();
    trimmed.starts_with("execute scheduled goal:")
        || trimmed.starts_with("scheduled check:")
        || trimmed.starts_with("manual scheduled run:")
}

/// Whether a goal's daily token budget is exhausted *for today*. Only usage
/// recorded on the current UTC day counts: stale `tokens_used_today` from a
/// previous day (before the daily reset has run) must NOT block a goal, or an
/// expensive run can deadlock the goal across the day boundary (the budget gate
/// keeps deferring it, so it never fires, so it never resets).
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

/// Seconds since a task's most recent activity (or its `started_at` when it has
/// no activity rows). Used to tag stuck-task interrupts so the inactivity
/// threshold can be tuned from data. Returns 0 if no timestamp parses (never
/// panics). Both inputs are RFC3339 or SQLite-datetime strings.
fn task_inactivity_secs(
    last_activity: Option<&str>,
    started_at: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> i64 {
    let parse = |s: &str| {
        chrono::DateTime::parse_from_rfc3339(s)
            .map(|d| d.with_timezone(&chrono::Utc))
            .ok()
            .or_else(|| {
                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                    .ok()
                    .map(|nd| nd.and_utc())
            })
    };
    let reference = last_activity.and_then(parse).or_else(|| parse(started_at));
    match reference {
        Some(ts) => (now - ts).num_seconds().max(0),
        None => 0,
    }
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
    hub: Option<Weak<ChannelHub>>,
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
    pub fn new(
        state: Arc<dyn StateStore>,
        tick_interval_secs: u64,
        max_concurrent: usize,
        wake_rx: mpsc::Receiver<()>,
        hub: Option<Weak<ChannelHub>>,
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
    pub fn set_hub(&mut self, hub: Weak<ChannelHub>) {
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
        });
        if let Some(telemetry) = &self.telemetry {
            telemetry.register_job(name, interval);
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

                    // Auto-retry idempotent tasks that haven't exceeded max retries
                    if task.idempotent && task.retry_count < 3 {
                        let mut retry_task = task.clone();
                        retry_task.status = "pending".to_string();
                        retry_task.retry_count += 1;
                        retry_task.started_at = None;
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

        // Phase 2: Fire due schedules (recurring + one-shot)
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

        Ok(())
    }

    /// Detect tasks that have been running/claimed longer than the timeout and mark them interrupted.
    async fn detect_stuck_tasks(&self) {
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
            let by_id: std::collections::HashMap<&str, &crate::traits::Task> =
                tasks.iter().map(|t| (t.id.as_str(), t)).collect();

            for task in tasks.iter().filter(|t| t.status == "pending") {
                let Some(deps_json) = task.depends_on.as_deref() else {
                    continue;
                };
                let deps: Vec<String> = serde_json::from_str(deps_json).unwrap_or_default();
                let Some(failed_dep) = deps
                    .iter()
                    .filter_map(|dep_id| by_id.get(dep_id.as_str()))
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

            let tasks = match self.state.get_tasks_for_goal(&goal.id).await {
                Ok(t) => t,
                Err(_) => continue,
            };

            for task in &tasks {
                if task.status == "failed" && task.idempotent && task.retry_count < task.max_retries
                {
                    let mut retry_task = task.clone();
                    retry_task.status = "pending".to_string();
                    retry_task.retry_count += 1;
                    retry_task.error = None;
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
                        let _ = self.state.enqueue_notification(&entry).await;
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
            let goal = match self.state.get_goal(goal_id).await {
                Ok(Some(g)) => g,
                _ => continue,
            };

            // Only care about active goals
            if goal.status != "active" {
                continue;
            }

            let is_scheduled_goal =
                if let Ok(schedules) = self.state.get_schedules_for_goal(goal_id).await {
                    !schedules.is_empty()
                } else {
                    tasks
                        .iter()
                        .any(|task| is_scheduled_task_description(&task.description))
                };

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
            let all_too_new = tasks.iter().all(|t| {
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

            // Try to atomically claim the first pending task and spawn a task lead.
            let first_task = tasks[0];
            let agent_id = format!("heartbeat-dispatch-{}", uuid::Uuid::new_v4());

            let claimed = match self.state.claim_task(&first_task.id, &agent_id).await {
                Ok(c) => c,
                Err(e) => {
                    error!(task_id = %first_task.id, error = %e, "Failed to claim task for dispatch");
                    continue;
                }
            };

            if !claimed {
                // Another tick or agent already claimed it — skip
                continue;
            }

            info!(
                goal_id = %goal_id,
                task_id = %first_task.id,
                pending_count = tasks.len(),
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
                    );
                    continue;
                }
            }

            // No agent available — revert claimed task back to pending so it's
            // not stranded, then enqueue a stalled notification.
            warn!(
                goal_id = %goal_id,
                task_id = %first_task.id,
                pending_count = tasks.len(),
                "No agent available for dispatch — reverting claim and notifying user"
            );
            let mut reverted = first_task.clone();
            reverted.status = "pending".to_string();
            reverted.agent_id = None;
            reverted.started_at = None;
            let _ = self.state.update_task(&reverted).await;

            let msg = format!(
                "Goal stalled: \"{}\" has {} pending task(s) but no active agent. \
                 You can re-trigger this by asking me about it again.",
                goal.description.chars().take(200).collect::<String>(),
                tasks.len(),
            );
            let entry =
                crate::traits::NotificationEntry::new(goal_id, &goal.session_id, "stalled", &msg);
            let _ = self.state.enqueue_notification(&entry).await;
        }
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
                    // Build notification from actual task results, not goal description
                    let completed_tasks = self
                        .state
                        .get_tasks_for_goal(&goal.id)
                        .await
                        .unwrap_or_default();
                    let fallback_summary: String = goal.description.chars().take(300).collect();
                    let task_results_summary =
                        build_goal_task_results_summary(&completed_tasks, &fallback_summary);

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
                        let failed = summary.get("failed").and_then(|v| v.as_u64()).unwrap_or(0);
                        let blocked = summary.get("blocked").and_then(|v| v.as_u64()).unwrap_or(0);
                        let total = summary.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
                        (
                            "completed",
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
                        // language (e.g., "I'm blocked from safely finishing").
                        // Don't present this as "Goal completed:" — the user already
                        // received the actual result via the agent's direct reply.
                        // Suppress the misleading notification entirely.
                        continue;
                    } else {
                        (
                            "completed",
                            format!(
                                "Goal completed:\n\n{}",
                                task_results_summary.chars().take(4000).collect::<String>()
                            ),
                        )
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

            if let Err(e) = self.state.enqueue_notification(&entry).await {
                error!(goal_id = %goal.id, error = %e, "Failed to enqueue notification");
                continue;
            }

            // Mark goal as notified so we don't enqueue again
            if let Err(e) = self.state.mark_goal_notified(&goal.id).await {
                error!(goal_id = %goal.id, error = %e, "Failed to mark goal notified after enqueue");
            }
        }
    }

    /// Process the notification queue: attempt delivery, track attempts.
    async fn process_notification_queue(&self) {
        let pending = match self.state.get_pending_notifications(20).await {
            Ok(n) => n,
            Err(e) => {
                error!(error = %e, "Failed to get pending notifications");
                return;
            }
        };

        for entry in &pending {
            let delivered = if let Some(hub) = self.hub.as_ref().and_then(|w| w.upgrade()) {
                let message = crate::tools::sanitize::sanitize_user_facing_reply(&entry.message);
                hub.send_text(&entry.session_id, &message).await.is_ok()
            } else {
                false
            };

            if delivered {
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
                            goal.description, days_idle
                        );
                        let entry = crate::traits::NotificationEntry::new(
                            &goal.id,
                            &goal.session_id,
                            "evergreen_alert",
                            &msg,
                        );
                        let _ = self.state.enqueue_notification(&entry).await;
                        let _ = self.state.mark_goal_notified(&goal.id).await;
                        // Keep the in-memory copy consistent so the later
                        // update_goal (after task creation) doesn't clobber
                        // notified_at back to NULL and re-alert next tick.
                        goal.notified_at = Some(now_ts.clone());
                    }
                }
            }
        }

        let tasks = self
            .state
            .get_tasks_for_goal(&goal.id)
            .await
            .unwrap_or_default();
        let open_count = tasks
            .iter()
            .filter(|t| matches!(t.status.as_str(), "pending" | "claimed" | "running"))
            .count();

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
            return Ok(());
        }

        // Budget check: skip if *today's* daily budget is exhausted, but back off
        // schedule to avoid hot-loop. Stale prior-day usage must not count, or the
        // goal deadlocks across the day boundary (it would defer forever and never
        // fire, so its counter would never reset).
        let budget_today = now.date_naive().to_string();
        if daily_budget_exhausted(
            goal.budget_daily,
            goal.tokens_used_today,
            &goal.tokens_used_day,
            &budget_today,
        ) {
            schedule.next_run_at = (now + chrono::Duration::minutes(15)).to_rfc3339();
            schedule.updated_at = now_ts.clone();
            let _ = self.state.update_goal_schedule(&schedule).await;
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
                    id: uuid::Uuid::new_v4().to_string(),
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
                let _ = self.state.create_task(&task).await;

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
            id: uuid::Uuid::new_v4().to_string(),
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
            idempotent: goal.goal_type == "continuous",
            retry_count: 0,
            max_retries: 1,
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
        if !is_group_session(&goal.session_id) {
            if let Some(hub_weak) = &self.hub {
                if let Some(hub_arc) = hub_weak.upgrade() {
                    let short_desc: String = goal.description.chars().take(200).collect();
                    let _ = hub_arc
                        .send_text(
                            &goal.session_id,
                            &format!("🔄 Running scheduled task: {}", short_desc),
                        )
                        .await;
                }
            }
        }

        // Dispatch immediately instead of waiting for the orphan-recovery pass
        // (which only picks up tasks older than 60s, so every scheduled run
        // used to start at least a minute late). If no agent is available the
        // orphan dispatcher still picks the task up on a later tick.
        if let Some(agent_arc) = self.agent.as_ref().and_then(|w| w.upgrade()) {
            let agent_id = format!("heartbeat-schedule-fire-{}", uuid::Uuid::new_v4());
            match self.state.claim_task(&task.id, &agent_id).await {
                Ok(true) => {
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
                    );
                }
                Ok(false) => {}
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
    use crate::traits::{Goal, Task};
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        Arc<ChannelHub>,
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
        let hub = Arc::new(ChannelHub::new(
            vec![channel.clone() as Arc<dyn crate::traits::Channel>],
            session_map,
        ));
        let coordinator = HeartbeatCoordinator::new(
            state.clone(),
            1,
            3,
            wake_rx,
            Some(Arc::downgrade(&hub)),
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
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].starts_with("🔄 Running scheduled task:"));
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

        // Manually set tokens_used_today > 0 via the concrete pool
        sqlx::query("UPDATE goals SET tokens_used_today = 1500 WHERE id = ?")
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
