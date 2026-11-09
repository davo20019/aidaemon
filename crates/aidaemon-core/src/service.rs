//! Background work an application runs beside its agents.
//!
//! A [`Service`] states when it runs; a [`ServiceHost`] runs it. The host
//! gives every service the same treatment: one run at a time, backoff after
//! failures, a panic counted as a failure, and a bounded drain on shutdown.
//!
//! An agent run started from a service is built with [`crate::AgentBuilder`]
//! like any other, so it carries the same safeguards.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt;
use tokio::sync::{Notify, Semaphore};
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Most doublings of a failing service's interval.
const MAX_BACKOFF_DOUBLINGS: u32 = 5;

/// When a service runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// At most this often. Runs share the host's concurrency limit.
    Every(Duration),
    /// On every tick and wake-up, for cycle work that must not wait behind
    /// [`Schedule::Every`] services.
    EveryTick,
    /// Long-lived work, started once and restarted if it ends.
    Continuous,
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

/// What a service can see of its host.
#[derive(Debug, Clone)]
pub struct ServiceContext {
    shutdown: CancellationToken,
}

impl ServiceContext {
    /// Cancelled when the host stops. Long runs should watch it.
    pub fn shutdown(&self) -> &CancellationToken {
        &self.shutdown
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

/// What happened to a service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceEvent {
    /// A run was launched.
    Launched { service: String },
    /// A run succeeded. `previous_failures` is the streak it ended.
    Succeeded {
        service: String,
        previous_failures: u32,
    },
    /// A run failed or panicked. `failures` counts the streak.
    Failed {
        service: String,
        failures: u32,
        error: String,
    },
    /// `start` failed.
    StartFailed { service: String, error: String },
    /// A run ended, however it ended.
    Finished { service: String },
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
    running: Arc<AtomicBool>,
    failures: Arc<AtomicU32>,
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
        let ctx = ServiceContext { shutdown };
        for slot in &self.slots {
            if let Err(error) = slot.service.start(&ctx).await {
                self.emit(ServiceEvent::StartFailed {
                    service: slot.service.name().to_string(),
                    error: error.0,
                });
            }
        }

        let mut runs = JoinSet::new();
        while !ctx.shutdown.is_cancelled() {
            let ready = match &self.gate {
                Some(gate) => gate.ready().await,
                None => true,
            };
            if ready {
                self.launch_due(&ctx, &mut runs);
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

    fn launch_due(&mut self, ctx: &ServiceContext, runs: &mut JoinSet<()>) {
        let now = Instant::now();
        let tick = self.tick;
        for index in 0..self.slots.len() {
            let slot = &mut self.slots[index];
            if slot.running.load(Ordering::Relaxed) || !is_due(slot, tick, now) {
                continue;
            }
            slot.last_run = Some(now);
            slot.running.store(true, Ordering::Relaxed);
            let service = slot.service.clone();
            let running = slot.running.clone();
            let failures = slot.failures.clone();
            let permits = matches!(slot.schedule, Schedule::Every(_)).then(|| self.permits.clone());
            let observer = self.observer.clone();
            let ctx = ctx.clone();
            let name = service.name().to_string();
            self.emit(ServiceEvent::Launched {
                service: name.clone(),
            });
            runs.spawn(run_once(
                service, ctx, permits, running, failures, observer, name,
            ));
        }
    }
}

fn is_due(slot: &Slot, tick: Duration, now: Instant) -> bool {
    let interval = match slot.schedule {
        Schedule::Every(interval) => interval,
        Schedule::EveryTick | Schedule::Continuous => tick,
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

async fn run_once(
    service: Arc<dyn Service>,
    ctx: ServiceContext,
    permits: Option<Arc<Semaphore>>,
    running: Arc<AtomicBool>,
    failures: Arc<AtomicU32>,
    observer: Option<Arc<dyn ServiceObserver>>,
    name: String,
) {
    let emit = |event| {
        if let Some(observer) = &observer {
            observer.event(event);
        }
    };
    let _permit = match permits {
        Some(permits) => tokio::select! {
            permit = permits.acquire_owned() => permit.ok(),
            _ = ctx.shutdown.cancelled() => {
                running.store(false, Ordering::Relaxed);
                emit(ServiceEvent::Finished { service: name });
                return;
            }
        },
        None => None,
    };
    let result = AssertUnwindSafe(service.run(&ctx)).catch_unwind().await;
    running.store(false, Ordering::Relaxed);
    match result {
        Ok(Ok(())) => emit(ServiceEvent::Succeeded {
            service: name.clone(),
            previous_failures: failures.swap(0, Ordering::Relaxed),
        }),
        Ok(Err(error)) => emit(ServiceEvent::Failed {
            service: name.clone(),
            failures: failures.fetch_add(1, Ordering::Relaxed) + 1,
            error: error.0,
        }),
        Err(_) => emit(ServiceEvent::Failed {
            service: name.clone(),
            failures: failures.fetch_add(1, Ordering::Relaxed) + 1,
            error: "the service panicked".to_string(),
        }),
    }
    emit(ServiceEvent::Finished { service: name });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Mutex;

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
            ServiceEvent::Failed { failures: 1, error, .. } if error.contains("panicked")
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            ServiceEvent::Succeeded {
                previous_failures: 1,
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
}
