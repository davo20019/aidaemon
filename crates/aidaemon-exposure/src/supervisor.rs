//! Runs one exposure per configured service, restarts it with capped
//! backoff, and publishes its validated origin into the service's
//! [`EndpointSource`](aidaemon_core::EndpointSource).

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use aidaemon_core::{CancellationToken, EndpointPublisher};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::{
    validate_https_origin, ExposedService, ExposureAdapter, ExposureHealth, OriginStability,
    ServiceListeners,
};

/// Restart delays: `initial`, doubling up to `max`, back to `initial` once an
/// exposure stayed up for `reset_after`.
#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    pub initial: Duration,
    pub max: Duration,
    pub reset_after: Duration,
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            initial: Duration::from_secs(1),
            max: Duration::from_secs(60),
            reset_after: Duration::from_secs(300),
        }
    }
}

/// One service to publish.
pub struct ExposurePlan {
    pub service: ExposedService,
    pub adapter: Arc<dyn ExposureAdapter>,
    /// Where the live origin goes; the only writer is the supervisor.
    pub publisher: EndpointPublisher,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceState {
    Starting,
    Up,
    Degraded,
    /// Down, retrying after a backoff.
    Down,
    /// Never started: the plan breaks an invariant.
    Refused,
    Stopped,
}

impl ServiceState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Up => "up",
            Self::Degraded => "degraded",
            Self::Down => "down",
            Self::Refused => "refused",
            Self::Stopped => "stopped",
        }
    }
}

/// What `/exposure` shows for one service.
#[derive(Debug, Clone)]
pub struct ServiceStatus {
    pub service: ExposedService,
    pub adapter: String,
    pub state: ServiceState,
    /// The origin published now, if any.
    pub origin: Option<String>,
    /// When `state` or `origin` last changed.
    pub since: SystemTime,
    pub last_failure: Option<String>,
    pub restarts: u32,
}

type Statuses = Arc<Mutex<Vec<ServiceStatus>>>;

/// The running exposures. Owns no global state: it takes its plans as values.
pub struct ExposureSupervisor {
    statuses: Statuses,
    tasks: Vec<JoinHandle<()>>,
}

/// The longest failure reason kept (no full child stderr in status).
const MAX_REASON: usize = 300;

fn bounded(reason: &str) -> String {
    let reason = reason.trim();
    if reason.len() <= MAX_REASON {
        return reason.to_string();
    }
    let mut end = MAX_REASON;
    while !reason.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &reason[..end])
}

impl ExposureSupervisor {
    /// Start every plan whose service is a registered listener. Plans that
    /// break an invariant are recorded as refused and never started. Runs
    /// until `shutdown`; then every exposure is stopped ([`Self::join`]
    /// waits for that).
    pub fn spawn(
        plans: Vec<ExposurePlan>,
        listeners: &ServiceListeners,
        shutdown: CancellationToken,
        backoff: Backoff,
    ) -> Self {
        let statuses: Statuses = Arc::default();
        let mut tasks = Vec::new();
        for plan in plans {
            let index = {
                let mut all = statuses.lock().unwrap();
                all.push(ServiceStatus {
                    service: plan.service.clone(),
                    adapter: plan.adapter.id().to_string(),
                    state: ServiceState::Starting,
                    origin: None,
                    since: SystemTime::now(),
                    last_failure: None,
                    restarts: 0,
                });
                all.len() - 1
            };
            let refusal = match listeners.get(&plan.service) {
                None => Some(format!(
                    "{} is not a running service listener",
                    plan.service
                )),
                Some(_)
                    if plan.service.needs_stable_origin()
                        && plan.adapter.stability() == OriginStability::Ephemeral =>
                {
                    Some(format!(
                        "{} needs a stable origin and {} gives an ephemeral one",
                        plan.service,
                        plan.adapter.id()
                    ))
                }
                Some(_) => None,
            };
            if let Some(reason) = refusal {
                warn!(service = %plan.service, adapter = plan.adapter.id(), %reason, "Exposure refused");
                let mut all = statuses.lock().unwrap();
                all[index].state = ServiceState::Refused;
                all[index].last_failure = Some(reason);
                continue;
            }
            let local = listeners.get(&plan.service).expect("checked above");
            let runner = Runner {
                index,
                statuses: statuses.clone(),
                plan,
                local,
                shutdown: shutdown.clone(),
                backoff,
            };
            tasks.push(tokio::spawn(runner.run()));
        }
        Self { statuses, tasks }
    }

    /// Every service's status, in plan order.
    pub fn status(&self) -> Vec<ServiceStatus> {
        self.statuses.lock().unwrap().clone()
    }

    /// A handle that reads status after the supervisor moved into `join`.
    pub fn status_handle(&self) -> StatusHandle {
        StatusHandle(self.statuses.clone())
    }

    /// Wait until every exposure stopped (after the shutdown token fired).
    pub async fn join(self) {
        for task in self.tasks {
            let _ = task.await;
        }
    }
}

/// Reads the supervisor's status from elsewhere (a slash command).
#[derive(Clone)]
pub struct StatusHandle(Statuses);

impl StatusHandle {
    pub fn status(&self) -> Vec<ServiceStatus> {
        self.0.lock().unwrap().clone()
    }
}

struct Runner {
    index: usize,
    statuses: Statuses,
    plan: ExposurePlan,
    local: std::net::SocketAddr,
    shutdown: CancellationToken,
    backoff: Backoff,
}

enum Ended {
    Shutdown,
    Failed { reason: String, up_for: Duration },
}

impl Runner {
    fn update(&self, f: impl FnOnce(&mut ServiceStatus)) {
        let mut all = self.statuses.lock().unwrap();
        let status = &mut all[self.index];
        let before = (status.state, status.origin.clone());
        f(status);
        if (status.state, status.origin.clone()) != before {
            status.since = SystemTime::now();
            info!(
                service = %status.service,
                adapter = %status.adapter,
                state = status.state.as_str(),
                origin = status.origin.as_deref().unwrap_or("-"),
                "Exposure state changed"
            );
        }
    }

    fn publish(&self, origin: Option<String>) {
        self.plan.publisher.set(origin);
    }

    /// Validate and publish what the adapter reported; `Err` is why it is
    /// a failure instead.
    fn on_published(
        &self,
        endpoint: crate::PublicEndpoint,
        degraded: Option<String>,
    ) -> Result<(), String> {
        if self.plan.service.needs_stable_origin()
            && endpoint.stability == OriginStability::Ephemeral
        {
            return Err(format!(
                "{} reported an ephemeral origin, which {} cannot use",
                self.plan.adapter.id(),
                self.plan.service
            ));
        }
        let origin = validate_https_origin(&endpoint.origin).map_err(|error| {
            format!(
                "{} reported an invalid origin: it {error}",
                self.plan.adapter.id()
            )
        })?;
        self.publish(Some(origin.clone()));
        self.update(|s| {
            s.origin = Some(origin);
            match degraded {
                Some(reason) => {
                    s.state = ServiceState::Degraded;
                    s.last_failure = Some(bounded(&reason));
                }
                None => s.state = ServiceState::Up,
            }
        });
        Ok(())
    }

    async fn run(self) {
        let mut delay = self.backoff.initial;
        loop {
            if self.shutdown.is_cancelled() {
                break;
            }
            match self.once().await {
                Ended::Shutdown => break,
                Ended::Failed { reason, up_for } => {
                    self.publish(None);
                    let reason = bounded(&reason);
                    warn!(service = %self.plan.service, adapter = self.plan.adapter.id(), %reason, "Exposure down");
                    self.update(|s| {
                        s.state = ServiceState::Down;
                        s.origin = None;
                        s.last_failure = Some(reason);
                        s.restarts = s.restarts.saturating_add(1);
                    });
                    if up_for >= self.backoff.reset_after {
                        delay = self.backoff.initial;
                    }
                    tokio::select! {
                        _ = self.shutdown.cancelled() => break,
                        _ = tokio::time::sleep(delay) => {}
                    }
                    delay = (delay * 2).min(self.backoff.max);
                }
            }
        }
        self.publish(None);
        self.update(|s| {
            s.state = ServiceState::Stopped;
            s.origin = None;
        });
    }

    /// One exposure, from `expose` until it fails or the daemon stops.
    async fn once(&self) -> Ended {
        self.update(|s| s.state = ServiceState::Starting);
        let exposed = tokio::select! {
            _ = self.shutdown.cancelled() => return Ended::Shutdown,
            exposed = self.plan.adapter.expose(&self.plan.service, self.local) => exposed,
        };
        let handle = match exposed {
            Ok(handle) => handle,
            Err(error) => {
                return Ended::Failed {
                    reason: error.to_string(),
                    up_for: Duration::ZERO,
                }
            }
        };
        let mut rx = handle.watch();
        let mut up_since: Option<tokio::time::Instant> = None;
        let up_for = |since: Option<tokio::time::Instant>| {
            since.map(|t| t.elapsed()).unwrap_or(Duration::ZERO)
        };
        let failure = loop {
            let health = rx.borrow_and_update().clone();
            match health {
                ExposureHealth::Starting => {
                    self.publish(None);
                    self.update(|s| {
                        s.state = ServiceState::Starting;
                        s.origin = None;
                    });
                }
                ExposureHealth::Up(endpoint) => {
                    if let Err(reason) = self.on_published(endpoint, None) {
                        break reason;
                    }
                    up_since.get_or_insert_with(tokio::time::Instant::now);
                }
                ExposureHealth::Degraded { endpoint, reason } => {
                    if let Err(reason) = self.on_published(endpoint, Some(reason)) {
                        break reason;
                    }
                    up_since.get_or_insert_with(tokio::time::Instant::now);
                }
                ExposureHealth::Down { reason } => break reason,
            }
            tokio::select! {
                _ = self.shutdown.cancelled() => {
                    if let Err(error) = handle.stop().await {
                        warn!(service = %self.plan.service, %error, "Exposure did not stop cleanly");
                    }
                    return Ended::Shutdown;
                }
                changed = rx.changed() => {
                    if changed.is_err() {
                        break "the exposure ended".to_string();
                    }
                }
            }
        };
        let up_for = up_for(up_since);
        if let Err(error) = handle.stop().await {
            warn!(service = %self.plan.service, %error, "Exposure did not stop cleanly");
        }
        Ended::Failed {
            reason: failure,
            up_for,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{FakeAdapter, FakeStep};
    use aidaemon_core::EndpointSource;

    fn listeners() -> ServiceListeners {
        let mut listeners = ServiceListeners::new();
        listeners
            .register(
                ExposedService::ClientGateway,
                "127.0.0.1:8788".parse().unwrap(),
            )
            .unwrap();
        listeners
            .register(
                ExposedService::NodeGateway,
                "127.0.0.1:8787".parse().unwrap(),
            )
            .unwrap();
        listeners
    }

    fn fast() -> Backoff {
        Backoff {
            initial: Duration::from_millis(10),
            max: Duration::from_millis(40),
            reset_after: Duration::from_secs(300),
        }
    }

    async fn until(mut check: impl FnMut() -> bool) {
        for _ in 0..500 {
            if check() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("condition never held");
    }

    #[tokio::test]
    async fn publishes_origin_and_falls_back_while_down() {
        let adapter = FakeAdapter::new(OriginStability::Stable);
        let (source, publisher) = EndpointSource::live(vec!["https://static.example.com".into()]);
        let shutdown = CancellationToken::new();
        let supervisor = ExposureSupervisor::spawn(
            vec![ExposurePlan {
                service: ExposedService::ClientGateway,
                adapter: adapter.clone(),
                publisher,
            }],
            &listeners(),
            shutdown.clone(),
            fast(),
        );
        until(|| adapter.exposures() == 1).await;
        adapter.step(FakeStep::Up("https://live.example.com/"));
        until(|| source.live_origin().is_some()).await;
        assert_eq!(
            source.current(),
            vec!["https://live.example.com", "https://static.example.com"]
        );
        assert_eq!(supervisor.status()[0].state, ServiceState::Up);

        adapter.step(FakeStep::Down("connection lost"));
        until(|| source.live_origin().is_none()).await;
        assert_eq!(source.current(), vec!["https://static.example.com"]);
        // Restarted after the backoff.
        until(|| adapter.exposures() == 2).await;
        let status = supervisor.status();
        assert_eq!(status[0].restarts, 1);
        assert_eq!(status[0].last_failure.as_deref(), Some("connection lost"));

        shutdown.cancel();
        supervisor.join().await;
        assert_eq!(adapter.stopped(), 2);
        assert_eq!(source.live_origin(), None);
    }

    #[tokio::test]
    async fn backoff_doubles_up_to_the_cap() {
        let adapter = FakeAdapter::new(OriginStability::Stable);
        adapter.fail_expose("cloudflared is not installed");
        let (_source, publisher) = EndpointSource::live(vec![]);
        let shutdown = CancellationToken::new();
        let supervisor = ExposureSupervisor::spawn(
            vec![ExposurePlan {
                service: ExposedService::ClientGateway,
                adapter: adapter.clone(),
                publisher,
            }],
            &listeners(),
            shutdown.clone(),
            fast(),
        );
        until(|| adapter.attempts() >= 5).await;
        let gaps = adapter.attempt_gaps();
        // 10, 20, 40, 40 ms: never shorter than the doubled delay.
        assert!(gaps[0] >= Duration::from_millis(10));
        assert!(gaps[1] >= Duration::from_millis(20));
        assert!(gaps[2] >= Duration::from_millis(40));
        assert!(gaps[3] < Duration::from_millis(200), "capped: {gaps:?}");
        assert_eq!(supervisor.status()[0].state, ServiceState::Down);
        shutdown.cancel();
        supervisor.join().await;
    }

    #[tokio::test]
    async fn ephemeral_adapter_is_refused_for_the_node_gateway() {
        let adapter = FakeAdapter::new(OriginStability::Ephemeral);
        let (source, publisher) = EndpointSource::live(vec![]);
        let shutdown = CancellationToken::new();
        let supervisor = ExposureSupervisor::spawn(
            vec![ExposurePlan {
                service: ExposedService::NodeGateway,
                adapter: adapter.clone(),
                publisher,
            }],
            &listeners(),
            shutdown.clone(),
            fast(),
        );
        assert_eq!(supervisor.status()[0].state, ServiceState::Refused);
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(adapter.attempts(), 0);
        assert!(source.current().is_empty());
        shutdown.cancel();
        supervisor.join().await;
    }

    #[tokio::test]
    async fn ephemeral_origin_reported_to_the_node_gateway_is_a_failure() {
        let adapter = FakeAdapter::new(OriginStability::Stable);
        let (source, publisher) = EndpointSource::live(vec![]);
        let shutdown = CancellationToken::new();
        let supervisor = ExposureSupervisor::spawn(
            vec![ExposurePlan {
                service: ExposedService::NodeGateway,
                adapter: adapter.clone(),
                publisher,
            }],
            &listeners(),
            shutdown.clone(),
            fast(),
        );
        until(|| adapter.exposures() == 1).await;
        adapter.step(FakeStep::UpEphemeral("https://random.example.com"));
        until(|| supervisor.status()[0].restarts == 1).await;
        assert!(source.current().is_empty());
        shutdown.cancel();
        supervisor.join().await;
    }

    #[tokio::test]
    async fn unregistered_service_is_refused() {
        let adapter = FakeAdapter::new(OriginStability::Stable);
        let (_source, publisher) = EndpointSource::live(vec![]);
        let shutdown = CancellationToken::new();
        let supervisor = ExposureSupervisor::spawn(
            vec![ExposurePlan {
                service: ExposedService::Webhook {
                    slug: "telegram".into(),
                },
                adapter: adapter.clone(),
                publisher,
            }],
            &listeners(),
            shutdown.clone(),
            fast(),
        );
        assert_eq!(supervisor.status()[0].state, ServiceState::Refused);
        shutdown.cancel();
        supervisor.join().await;
        assert_eq!(adapter.attempts(), 0);
    }

    #[tokio::test]
    async fn invalid_origin_is_a_failure_not_an_endpoint() {
        let adapter = FakeAdapter::new(OriginStability::Stable);
        let (source, publisher) = EndpointSource::live(vec![]);
        let shutdown = CancellationToken::new();
        let supervisor = ExposureSupervisor::spawn(
            vec![ExposurePlan {
                service: ExposedService::ClientGateway,
                adapter: adapter.clone(),
                publisher,
            }],
            &listeners(),
            shutdown.clone(),
            fast(),
        );
        until(|| adapter.exposures() == 1).await;
        adapter.step(FakeStep::Up("http://plain.example.com/admin"));
        until(|| supervisor.status()[0].restarts == 1).await;
        assert!(source.current().is_empty());
        let reason = supervisor.status()[0].last_failure.clone().unwrap();
        assert!(reason.contains("invalid origin"), "{reason}");
        shutdown.cancel();
        supervisor.join().await;
    }

    #[tokio::test]
    async fn degraded_keeps_publishing_the_origin() {
        let adapter = FakeAdapter::new(OriginStability::Stable);
        let (source, publisher) = EndpointSource::live(vec![]);
        let shutdown = CancellationToken::new();
        let supervisor = ExposureSupervisor::spawn(
            vec![ExposurePlan {
                service: ExposedService::ClientGateway,
                adapter: adapter.clone(),
                publisher,
            }],
            &listeners(),
            shutdown.clone(),
            fast(),
        );
        until(|| adapter.exposures() == 1).await;
        adapter.step(FakeStep::Degraded(
            "https://live.example.com",
            "HEAD returned 502",
        ));
        until(|| supervisor.status()[0].state == ServiceState::Degraded).await;
        assert_eq!(source.current(), vec!["https://live.example.com"]);
        shutdown.cancel();
        supervisor.join().await;
        assert_eq!(adapter.stopped(), 1);
    }

    #[test]
    fn reasons_are_bounded() {
        let long = "x".repeat(1000);
        assert!(bounded(&long).len() <= MAX_REASON + '…'.len_utf8());
    }
}
