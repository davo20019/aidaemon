//! Background work an application runs beside its agents.
//!
//! A [`Service`] states when it runs; a [`ServiceHost`] runs it. The host
//! gives every service the same treatment: one run at a time, backoff after
//! failures, a panic counted as a failure, and a bounded drain on shutdown.
//!
//! An agent run started from a service is built with [`crate::AgentBuilder`]
//! like any other, so it carries the same safeguards.
//!
//! How the host schedules, exactly:
//!
//! - [`Schedule::Every`] is measured from one launch to the next, not from
//!   the end of a run. A service never overlaps itself: a run still going
//!   when the next is due delays it.
//! - After a failure (an `Err` or a panic) the wait is the interval doubled
//!   per failure in a row, up to 2^5 times; a success resets it.
//! - [`Schedule::DailyUtc`] runs once a day at a wall-clock time, in UTC,
//!   counted on the runtime's clock from the host's start (so paused Tokio
//!   time in a test moves it too). A failed run waits for the next day.
//! - The host looks for due work every `tick` and on [`ServiceWaker::wake`],
//!   so a run starts up to one tick late.
//! - [`ServiceHost::run`] consumes the host: add every service before it
//!   starts.
//! - [`ServiceHost::gate`] is asked before each tick's launches; while it
//!   says no, nothing new starts (runs already going continue).
//!
//! Every launch gets one [`ServiceEvent::Launched`] and exactly one
//! [`ServiceEvent::Ended`], with the run's number, how it ended
//! ([`ServiceRunEnd`]: succeeded, failed, aborted at shutdown, or skipped
//! because the host stopped before it began) and how long it ran, on
//! Tokio's clock. The run's [`ServiceContext`] carries the same number, to
//! tie the agent runs it starts to it.

use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::FutureExt;
use tokio::sync::{Notify, Semaphore};
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Most doublings of a failing service's interval.
const MAX_BACKOFF_DOUBLINGS: u32 = 5;

/// When a service runs (see the [module](self) for the exact rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Schedule {
    /// At most this often, measured from launch to launch; the first run is
    /// at once. Runs share the host's concurrency limit.
    Every(Duration),
    /// On every tick and wake-up, for cycle work that must not wait behind
    /// [`Schedule::Every`] services.
    EveryTick,
    /// Long-lived work, started once and restarted if it ends.
    Continuous,
    /// Once a day at `hour:minute` UTC, the first time at the next such
    /// moment after the host starts. Runs share the host's concurrency
    /// limit.
    DailyUtc { hour: u8, minute: u8 },
}

impl Schedule {
    /// [`Schedule::DailyUtc`], or `None` for a time that does not exist.
    pub fn daily_utc(hour: u8, minute: u8) -> Option<Self> {
        (hour < 24 && minute < 60).then_some(Self::DailyUtc { hour, minute })
    }

    fn takes_a_permit(self) -> bool {
        matches!(self, Self::Every(_) | Self::DailyUtc { .. })
    }
}

/// Why a service's run failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ServiceError(pub String);

impl ServiceError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// What a service can see of its host and of the run it is in.
#[derive(Debug, Clone)]
pub struct ServiceContext {
    shutdown: CancellationToken,
    run: u64,
    failures: u32,
}

impl ServiceContext {
    /// Cancelled when the host stops. Long runs should watch it.
    pub fn shutdown(&self) -> &CancellationToken {
        &self.shutdown
    }

    /// This run's number: 1 for the service's first run, then one more per
    /// launch, the same number its [`ServiceEvent`]s carry. 0 in
    /// [`Service::start`]. To tie an agent run to it, start the agent from
    /// a state you made, whose id you know:
    ///
    /// ```no_run
    /// # async fn demo(agent: aidaemon_core::Agent, ctx: &aidaemon_core::ServiceContext) {
    /// let state = aidaemon_core::RunState::new(
    ///     agent.id(),
    ///     &format!("service:nightly:{}", ctx.run()),
    ///     "Write the nightly report",
    /// );
    /// let run_id = state.run_id; // every Event of the run carries it
    /// let outcome = agent.run_from(state, ctx.shutdown().child_token()).await;
    /// # let _ = (run_id, outcome);
    /// # }
    /// ```
    pub fn run(&self) -> u64 {
        self.run
    }

    /// How many runs in a row failed just before this one.
    pub fn failures(&self) -> u32 {
        self.failures
    }
}

/// Work a [`ServiceHost`] runs in the background.
#[async_trait]
pub trait Service: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn schedule(&self) -> Schedule;

    /// Once, before any service's first run. A failure is reported and the
    /// service still runs.
    async fn start(&self, _ctx: &ServiceContext) -> Result<(), ServiceError> {
        Ok(())
    }

    async fn run(&self, ctx: &ServiceContext) -> Result<(), ServiceError>;

    /// Once, after the host has drained.
    async fn stop(&self) {}
}

/// A [`Service`] from a closure.
pub struct FnService<F> {
    name: String,
    schedule: Schedule,
    run: F,
}

impl<F, Fut> FnService<F>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), ServiceError>> + Send + 'static,
{
    pub fn new(name: impl Into<String>, schedule: Schedule, run: F) -> Self {
        Self {
            name: name.into(),
            schedule,
            run,
        }
    }
}

#[async_trait]
impl<F, Fut> Service for FnService<F>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), ServiceError>> + Send + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn schedule(&self) -> Schedule {
        self.schedule
    }

    async fn run(&self, _ctx: &ServiceContext) -> Result<(), ServiceError> {
        (self.run)().await
    }
}

/// What happened to a service. Every [`ServiceEvent::Launched`] is
/// followed by exactly one [`ServiceEvent::Ended`] with the same `run`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ServiceEvent {
    /// A run was launched.
    Launched { service: String, run: u64 },
    /// A run ended, however it ended. `elapsed` is how long it ran, on
    /// Tokio's clock (zero for a run that never began).
    Ended {
        service: String,
        run: u64,
        outcome: ServiceRunEnd,
        elapsed: Duration,
    },
    /// `start` failed.
    StartFailed { service: String, error: String },
}

/// How one run of a service ended.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ServiceRunEnd {
    /// `previous_failures` is the streak it ended.
    Succeeded { previous_failures: u32 },
    /// It returned an error or panicked. `failures` counts the streak.
    Failed { failures: u32, error: String },
    /// Still going when the host's grace period ran out, so it was
    /// aborted. Neither a success nor a failure.
    Aborted,
    /// The host stopped while the run waited for a concurrency permit, so
    /// it never began.
    Skipped,
}

/// Receives every [`ServiceEvent`], for logs and dashboards.
pub trait ServiceObserver: Send + Sync {
    fn event(&self, event: ServiceEvent);
}

/// Decides whether a tick launches anything.
#[async_trait]
pub trait ServiceGate: Send + Sync {
    async fn ready(&self) -> bool;
}

/// Ends the host's wait for the next tick.
#[derive(Debug, Clone)]
pub struct ServiceWaker(Arc<Notify>);

impl ServiceWaker {
    pub fn wake(&self) {
        self.0.notify_one();
    }
}

struct Slot {
    service: Arc<dyn Service>,
    schedule: Schedule,
    last_run: Option<Instant>,
    runs: u64,
    running: Arc<AtomicBool>,
    failures: Arc<AtomicU32>,
}

/// Runs launched and not yet ended: their service, number and when they
/// began, so a host that aborts them can still end each one.
type InFlight = Arc<Mutex<HashMap<(usize, u64), (String, Option<Instant>)>>>;

/// The wall clock as the runtime's clock moves: UTC at the host's start
/// plus Tokio time since.
#[derive(Clone, Copy)]
struct WallClock {
    started: Instant,
    wall_at_start: Duration,
}

impl WallClock {
    fn now() -> Self {
        Self {
            started: Instant::now(),
            wall_at_start: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default(),
        }
    }

    /// Seconds since the Unix epoch at `at`.
    fn wall(&self, at: Instant) -> u64 {
        (self.wall_at_start + at.saturating_duration_since(self.started)).as_secs()
    }
}

/// The latest `hour:minute` UTC at or before `wall` (seconds since the
/// epoch).
fn last_daily(wall: u64, hour: u8, minute: u8) -> u64 {
    const DAY: u64 = 86_400;
    let at = u64::from(hour) * 3600 + u64::from(minute) * 60;
    let today = wall - wall % DAY + at;
    if today <= wall {
        today
    } else {
        today.saturating_sub(DAY)
    }
}

/// Runs [`Service`]s until told to stop.
pub struct ServiceHost {
    slots: Vec<Slot>,
    tick: Duration,
    permits: Arc<Semaphore>,
    gate: Option<Arc<dyn ServiceGate>>,
    observer: Option<Arc<dyn ServiceObserver>>,
    wake: Arc<Notify>,
    grace: Duration,
}

impl ServiceHost {
    /// A host that looks for due work every `tick`.
    pub fn new(tick: Duration) -> Self {
        Self {
            slots: Vec::new(),
            tick,
            permits: Arc::new(Semaphore::new(Semaphore::MAX_PERMITS)),
            gate: None,
            observer: None,
            wake: Arc::new(Notify::new()),
            grace: Duration::from_secs(10),
        }
    }

    /// At most `limit` [`Schedule::Every`] runs at once.
    pub fn max_concurrent(mut self, limit: usize) -> Self {
        self.permits = Arc::new(Semaphore::new(limit.max(1)));
        self
    }

    /// How long a stopping host waits for runs to end before aborting them.
    pub fn grace(mut self, grace: Duration) -> Self {
        self.grace = grace;
        self
    }

    pub fn gate(mut self, gate: Arc<dyn ServiceGate>) -> Self {
        self.gate = Some(gate);
        self
    }

    pub fn observer(mut self, observer: Arc<dyn ServiceObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    pub fn service(mut self, service: Arc<dyn Service>) -> Self {
        self.slots.push(Slot {
            schedule: service.schedule(),
            service,
            last_run: None,
            runs: 0,
            running: Arc::new(AtomicBool::new(false)),
            failures: Arc::new(AtomicU32::new(0)),
        });
        self
    }

    pub fn waker(&self) -> ServiceWaker {
        ServiceWaker(self.wake.clone())
    }

    /// Run until `shutdown` is cancelled, then stop launching, drain for the
    /// grace period, abort what is left and call each service's `stop`.
    pub async fn run(mut self, shutdown: CancellationToken) {
        let ctx = ServiceContext {
            shutdown,
            run: 0,
            failures: 0,
        };
        for slot in &self.slots {
            if let Err(error) = slot.service.start(&ctx).await {
                self.emit(ServiceEvent::StartFailed {
                    service: slot.service.name().to_string(),
                    error: error.0,
                });
            }
        }

        let clock = WallClock::now();
        let in_flight: InFlight = Arc::default();
        let mut runs = JoinSet::new();
        while !ctx.shutdown.is_cancelled() {
            let ready = match &self.gate {
                Some(gate) => gate.ready().await,
                None => true,
            };
            if ready {
                self.launch_due(&ctx, clock, &in_flight, &mut runs);
            }
            while runs.try_join_next().is_some() {}
            tokio::select! {
                _ = tokio::time::sleep(self.tick) => {}
                _ = self.wake.notified() => {}
                _ = ctx.shutdown.cancelled() => {}
            }
        }

        let drained = tokio::time::timeout(self.grace, async {
            while runs.join_next().await.is_some() {}
        })
        .await;
        if drained.is_err() {
            runs.shutdown().await;
            // Every task has stopped: what is left in flight was aborted
            // (or never began) and gets its one end here.
            let now = Instant::now();
            let mut left: Vec<_> = in_flight
                .lock()
                .expect("in-flight runs poisoned")
                .drain()
                .collect();
            left.sort_by_key(|((slot, run), _)| (*slot, *run));
            for ((_, run), (service, began)) in left {
                let (outcome, elapsed) = match began {
                    Some(began) => (ServiceRunEnd::Aborted, now - began),
                    None => (ServiceRunEnd::Skipped, Duration::ZERO),
                };
                self.emit(ServiceEvent::Ended {
                    service,
                    run,
                    outcome,
                    elapsed,
                });
            }
        }
        for slot in &self.slots {
            slot.service.stop().await;
        }
    }

    fn emit(&self, event: ServiceEvent) {
        if let Some(observer) = &self.observer {
            observer.event(event);
        }
    }

    fn launch_due(
        &mut self,
        ctx: &ServiceContext,
        clock: WallClock,
        in_flight: &InFlight,
        runs: &mut JoinSet<()>,
    ) {
        let now = Instant::now();
        let tick = self.tick;
        for index in 0..self.slots.len() {
            let slot = &mut self.slots[index];
            if slot.running.load(Ordering::Relaxed) || !is_due(slot, tick, now, clock) {
                continue;
            }
            slot.last_run = Some(now);
            slot.runs += 1;
            slot.running.store(true, Ordering::Relaxed);
            let run = Run {
                service: slot.service.clone(),
                ctx: ServiceContext {
                    shutdown: ctx.shutdown.clone(),
                    run: slot.runs,
                    failures: slot.failures.load(Ordering::Relaxed),
                },
                permits: slot.schedule.takes_a_permit().then(|| self.permits.clone()),
                running: slot.running.clone(),
                failures: slot.failures.clone(),
                observer: self.observer.clone(),
                in_flight: in_flight.clone(),
                key: (index, slot.runs),
                name: slot.service.name().to_string(),
            };
            in_flight
                .lock()
                .expect("in-flight runs poisoned")
                .insert(run.key, (run.name.clone(), None));
            let launched = ServiceEvent::Launched {
                service: run.name.clone(),
                run: run.key.1,
            };
            self.emit(launched);
            runs.spawn(run_once(run));
        }
    }
}

fn is_due(slot: &Slot, tick: Duration, now: Instant, clock: WallClock) -> bool {
    if let Schedule::DailyUtc { hour, minute } = slot.schedule {
        // Due once the day's moment has passed since the last run (or the
        // host's start).
        let since = clock.wall(slot.last_run.unwrap_or(clock.started));
        let moment = last_daily(clock.wall(now), hour, minute);
        return moment > since;
    }
    let interval = match slot.schedule {
        Schedule::Every(interval) => interval,
        _ => tick,
    };
    let Some(last) = slot.last_run else {
        return true;
    };
    let failures = slot.failures.load(Ordering::Relaxed);
    let wait = match (slot.schedule, failures) {
        (Schedule::EveryTick, 0) | (Schedule::Continuous, 0) => Duration::ZERO,
        (_, 0) => interval,
        (_, failures) => interval * 2u32.pow(failures.min(MAX_BACKOFF_DOUBLINGS)),
    };
    now.duration_since(last) >= wait
}

/// One launched run and what it reports to.
struct Run {
    service: Arc<dyn Service>,
    ctx: ServiceContext,
    permits: Option<Arc<Semaphore>>,
    running: Arc<AtomicBool>,
    failures: Arc<AtomicU32>,
    observer: Option<Arc<dyn ServiceObserver>>,
    in_flight: InFlight,
    key: (usize, u64),
    name: String,
}

impl Run {
    /// End the run: leave the in-flight set and report, with no await in
    /// between, so an abort can never end it a second time.
    fn end(&self, outcome: ServiceRunEnd, elapsed: Duration) {
        self.in_flight
            .lock()
            .expect("in-flight runs poisoned")
            .remove(&self.key);
        if let Some(observer) = &self.observer {
            observer.event(ServiceEvent::Ended {
                service: self.name.clone(),
                run: self.key.1,
                outcome,
                elapsed,
            });
        }
    }
}

async fn run_once(run: Run) {
    let _permit = match &run.permits {
        Some(permits) => tokio::select! {
            permit = permits.clone().acquire_owned() => permit.ok(),
            _ = run.ctx.shutdown.cancelled() => {
                run.running.store(false, Ordering::Relaxed);
                run.end(ServiceRunEnd::Skipped, Duration::ZERO);
                return;
            }
        },
        None => None,
    };
    let began = Instant::now();
    if let Some(entry) = run
        .in_flight
        .lock()
        .expect("in-flight runs poisoned")
        .get_mut(&run.key)
    {
        entry.1 = Some(began);
    }
    let result = AssertUnwindSafe(run.service.run(&run.ctx))
        .catch_unwind()
        .await;
    let elapsed = began.elapsed();
    run.running.store(false, Ordering::Relaxed);
    let failed = |error: String| ServiceRunEnd::Failed {
        failures: run.failures.fetch_add(1, Ordering::Relaxed) + 1,
        error,
    };
    let outcome = match result {
        Ok(Ok(())) => ServiceRunEnd::Succeeded {
            previous_failures: run.failures.swap(0, Ordering::Relaxed),
        },
        Ok(Err(error)) => failed(error.0),
        Err(_) => failed("the service panicked".to_string()),
    };
    run.end(outcome, elapsed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    const TICK: Duration = Duration::from_secs(1);

    fn counted(
        schedule: Schedule,
        result: impl Fn(usize) -> Result<(), ServiceError> + Send + Sync + 'static,
    ) -> (Arc<dyn Service>, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let runs = count.clone();
        let result = Arc::new(result);
        let service = FnService::new("synthetic", schedule, move || {
            let n = runs.fetch_add(1, Ordering::SeqCst);
            let result = result.clone();
            async move { result(n) }
        });
        (Arc::new(service), count)
    }

    /// Run `host` for `length` of paused time, then stop it.
    async fn run_for(host: ServiceHost, length: Duration) {
        let shutdown = CancellationToken::new();
        let stopping = shutdown.clone();
        let handle = tokio::spawn(host.run(shutdown));
        tokio::time::sleep(length).await;
        stopping.cancel();
        handle.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn runs_at_once_then_at_its_interval() {
        let (service, count) = counted(Schedule::Every(Duration::from_secs(10)), |_| Ok(()));
        run_for(
            ServiceHost::new(TICK).service(service),
            Duration::from_millis(9_500),
        )
        .await;
        assert_eq!(count.load(Ordering::SeqCst), 1);

        let (service, count) = counted(Schedule::Every(Duration::from_secs(10)), |_| Ok(()));
        run_for(
            ServiceHost::new(TICK).service(service),
            Duration::from_millis(10_500),
        )
        .await;
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn every_tick_runs_each_tick_and_on_wake() {
        let (service, count) = counted(Schedule::EveryTick, |_| Ok(()));
        run_for(
            ServiceHost::new(TICK).service(service),
            Duration::from_millis(3_500),
        )
        .await;
        assert_eq!(count.load(Ordering::SeqCst), 4);

        let (service, count) = counted(Schedule::EveryTick, |_| Ok(()));
        let host = ServiceHost::new(Duration::from_secs(1000)).service(service);
        let waker = host.waker();
        let shutdown = CancellationToken::new();
        let handle = tokio::spawn(host.run(shutdown.clone()));
        tokio::time::sleep(Duration::from_millis(10)).await;
        waker.wake();
        tokio::time::sleep(Duration::from_millis(10)).await;
        shutdown.cancel();
        handle.await.unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn a_slow_run_never_overlaps_itself() {
        let active = Arc::new(AtomicUsize::new(0));
        let worst = Arc::new(AtomicUsize::new(0));
        let (a, w) = (active.clone(), worst.clone());
        let service = FnService::new("slow", Schedule::EveryTick, move || {
            let (a, w) = (a.clone(), w.clone());
            async move {
                w.fetch_max(a.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_secs(5)).await;
                a.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
        });
        run_for(
            ServiceHost::new(TICK).service(Arc::new(service)),
            Duration::from_secs(20),
        )
        .await;
        assert_eq!(worst.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn failures_back_off_and_a_success_resets() {
        // Fails twice: runs at 0, then +20s (2x), then +40s (4x), then ok.
        let (service, count) = counted(Schedule::Every(Duration::from_secs(10)), |n| {
            if n < 2 {
                Err(ServiceError::new("synthetic"))
            } else {
                Ok(())
            }
        });
        run_for(
            ServiceHost::new(TICK).service(service),
            Duration::from_millis(19_500),
        )
        .await;
        assert_eq!(count.load(Ordering::SeqCst), 1, "still backing off");

        let (service, count) = counted(Schedule::Every(Duration::from_secs(10)), |n| {
            if n < 2 {
                Err(ServiceError::new("synthetic"))
            } else {
                Ok(())
            }
        });
        run_for(
            ServiceHost::new(TICK).service(service),
            Duration::from_millis(70_500),
        )
        .await;
        // 0, 20, 60, then the interval again: 70.
        assert_eq!(count.load(Ordering::SeqCst), 4);
    }

    #[derive(Default)]
    struct Recorded(Mutex<Vec<ServiceEvent>>);

    impl ServiceObserver for Recorded {
        fn event(&self, event: ServiceEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_panic_is_a_failure_and_the_host_survives() {
        let (service, count) = counted(Schedule::EveryTick, |n| {
            assert!(n > 0, "synthetic panic");
            Ok(())
        });
        let seen = Arc::new(Recorded::default());
        let host = ServiceHost::new(TICK)
            .observer(seen.clone())
            .service(service);
        run_for(host, Duration::from_millis(2_500)).await;
        assert_eq!(count.load(Ordering::SeqCst), 2);
        let events = seen.0.lock().unwrap();
        assert!(events.iter().any(|e| matches!(
            e,
            ServiceEvent::Ended {
                outcome: ServiceRunEnd::Failed { failures: 1, error },
                ..
            } if error.contains("panicked")
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            ServiceEvent::Ended {
                outcome: ServiceRunEnd::Succeeded {
                    previous_failures: 1
                },
                ..
            }
        )));
    }

    struct Closed;

    #[async_trait]
    impl ServiceGate for Closed {
        async fn ready(&self) -> bool {
            false
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_closed_gate_launches_nothing() {
        let (service, count) = counted(Schedule::EveryTick, |_| Ok(()));
        let host = ServiceHost::new(TICK)
            .gate(Arc::new(Closed))
            .service(service);
        run_for(host, Duration::from_secs(5)).await;
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn the_limit_binds_every_services_but_not_cycle_work() {
        let held = |name: &'static str, schedule, seen: Arc<AtomicUsize>| {
            Arc::new(FnService::new(name, schedule, move || {
                let seen = seen.clone();
                async move {
                    seen.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_secs(100)).await;
                    Ok(())
                }
            }))
        };
        let periodic = Arc::new(AtomicUsize::new(0));
        let cycle = Arc::new(AtomicUsize::new(0));
        let every = Schedule::Every(Duration::from_secs(1000));
        let host = ServiceHost::new(TICK)
            .max_concurrent(1)
            .grace(Duration::ZERO)
            .service(held("a", every, periodic.clone()))
            .service(held("b", every, periodic.clone()))
            .service(held("cycle", Schedule::EveryTick, cycle.clone()));
        run_for(host, Duration::from_secs(5)).await;
        assert_eq!(periodic.load(Ordering::SeqCst), 1);
        assert_eq!(cycle.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn continuous_work_restarts_when_it_ends() {
        let (service, count) = counted(Schedule::Continuous, |_| Ok(()));
        run_for(
            ServiceHost::new(TICK).service(service),
            Duration::from_millis(2_500),
        )
        .await;
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    struct Lifecycle(Arc<Mutex<Vec<&'static str>>>);

    #[async_trait]
    impl Service for Lifecycle {
        fn name(&self) -> &str {
            "lifecycle"
        }
        fn schedule(&self) -> Schedule {
            Schedule::Continuous
        }
        async fn start(&self, _ctx: &ServiceContext) -> Result<(), ServiceError> {
            self.0.lock().unwrap().push("start");
            Ok(())
        }
        async fn run(&self, ctx: &ServiceContext) -> Result<(), ServiceError> {
            self.0.lock().unwrap().push("run");
            ctx.shutdown().cancelled().await;
            self.0.lock().unwrap().push("drained");
            Ok(())
        }
        async fn stop(&self) {
            self.0.lock().unwrap().push("stop");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_drains_then_stops() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let host = ServiceHost::new(TICK).service(Arc::new(Lifecycle(order.clone())));
        run_for(host, Duration::from_secs(1)).await;
        assert_eq!(*order.lock().unwrap(), ["start", "run", "drained", "stop"]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_run_that_ignores_shutdown_is_aborted_after_the_grace() {
        let service = FnService::new("stubborn", Schedule::Continuous, || async {
            std::future::pending::<()>().await;
            Ok(())
        });
        let host = ServiceHost::new(TICK)
            .grace(Duration::from_secs(2))
            .service(Arc::new(service));
        let shutdown = CancellationToken::new();
        let handle = tokio::spawn(host.run(shutdown.clone()));
        tokio::time::sleep(Duration::from_millis(100)).await;
        shutdown.cancel();
        let started = Instant::now();
        handle.await.unwrap();
        assert_eq!(started.elapsed(), Duration::from_secs(2));
    }

    /// Each run's events, in order, as (run, what).
    fn by_run(events: &[ServiceEvent]) -> Vec<(u64, String)> {
        events
            .iter()
            .filter_map(|event| match event {
                ServiceEvent::Launched { run, .. } => Some((*run, "launched".to_string())),
                ServiceEvent::Ended {
                    run,
                    outcome,
                    elapsed,
                    ..
                } => Some((*run, format!("{outcome:?} after {}ms", elapsed.as_millis()))),
                _ => None,
            })
            .collect()
    }

    #[tokio::test(start_paused = true)]
    async fn every_launch_ends_once_with_its_number_and_time() {
        let numbers = Arc::new(Mutex::new(Vec::new()));
        struct Timed(Arc<Mutex<Vec<(u64, u32)>>>);
        #[async_trait]
        impl Service for Timed {
            fn name(&self) -> &str {
                "timed"
            }
            fn schedule(&self) -> Schedule {
                Schedule::Every(Duration::from_secs(10))
            }
            async fn run(&self, ctx: &ServiceContext) -> Result<(), ServiceError> {
                self.0.lock().unwrap().push((ctx.run(), ctx.failures()));
                tokio::time::sleep(Duration::from_millis(50)).await;
                if ctx.run() == 1 {
                    Err(ServiceError::new("synthetic"))
                } else {
                    Ok(())
                }
            }
        }
        let seen = Arc::new(Recorded::default());
        let host = ServiceHost::new(TICK)
            .observer(seen.clone())
            .service(Arc::new(Timed(numbers.clone())));
        run_for(host, Duration::from_millis(25_000)).await;
        assert_eq!(*numbers.lock().unwrap(), [(1, 0), (2, 1)]);
        assert_eq!(
            by_run(&seen.0.lock().unwrap()),
            [
                (1, "launched".to_string()),
                (
                    1,
                    "Failed { failures: 1, error: \"synthetic\" } after 50ms".to_string()
                ),
                (2, "launched".to_string()),
                (
                    2,
                    "Succeeded { previous_failures: 1 } after 50ms".to_string()
                ),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_run_aborted_at_shutdown_still_ends() {
        let stubborn = FnService::new("stubborn", Schedule::Continuous, || async {
            std::future::pending::<()>().await;
            Ok(())
        });
        let waiting = FnService::new("waiting", Schedule::Every(TICK), || async { Ok(()) });
        let holder = FnService::new("holder", Schedule::Every(TICK), || async {
            std::future::pending::<()>().await;
            Ok(())
        });
        let seen = Arc::new(Recorded::default());
        let host = ServiceHost::new(TICK)
            .max_concurrent(1)
            .grace(Duration::from_secs(2))
            .observer(seen.clone())
            .service(Arc::new(stubborn))
            .service(Arc::new(holder))
            .service(Arc::new(waiting));
        run_for(host, Duration::from_millis(500)).await;
        let events = seen.0.lock().unwrap();
        let ends: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ServiceEvent::Ended {
                    service, outcome, ..
                } => Some((service.as_str(), outcome.clone())),
                _ => None,
            })
            .collect();
        let launched = events
            .iter()
            .filter(|event| matches!(event, ServiceEvent::Launched { .. }))
            .count();
        assert_eq!(launched, 3);
        assert_eq!(
            ends,
            [
                ("waiting", ServiceRunEnd::Skipped),
                ("stubborn", ServiceRunEnd::Aborted),
                ("holder", ServiceRunEnd::Aborted),
            ]
        );
    }

    #[test]
    fn the_last_daily_moment_is_today_or_yesterday() {
        let day = 86_400;
        let noon = 10 * day + 12 * 3600;
        assert_eq!(last_daily(noon, 2, 0), 10 * day + 2 * 3600);
        assert_eq!(last_daily(noon, 23, 30), 9 * day + 23 * 3600 + 1800);
        assert_eq!(last_daily(noon, 12, 0), noon);
        assert_eq!(Schedule::daily_utc(24, 0), None);
        assert_eq!(
            Schedule::daily_utc(2, 59),
            Some(Schedule::DailyUtc {
                hour: 2,
                minute: 59
            })
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_daily_service_runs_once_a_day_at_its_time() {
        // Due in one to two minutes (UTC), then once a day.
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 120;
        let (hour, minute) = (((wall % 86_400) / 3600) as u8, ((wall % 3600) / 60) as u8);
        let (service, count) = counted(Schedule::DailyUtc { hour, minute }, |_| Ok(()));
        let host = ServiceHost::new(Duration::from_secs(30)).service(service);
        let shutdown = CancellationToken::new();
        let handle = tokio::spawn(host.run(shutdown.clone()));
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(count.load(Ordering::SeqCst), 0, "not at start");
        tokio::time::sleep(Duration::from_secs(125)).await;
        assert_eq!(count.load(Ordering::SeqCst), 1, "at its time");
        tokio::time::sleep(Duration::from_secs(3600)).await;
        assert_eq!(count.load(Ordering::SeqCst), 1, "once");
        tokio::time::sleep(Duration::from_secs(86_400)).await;
        assert_eq!(count.load(Ordering::SeqCst), 2, "next day");
        shutdown.cancel();
        handle.await.unwrap();
    }
}
