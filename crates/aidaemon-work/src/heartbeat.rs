//! What a host's background jobs report: the per-job table a dashboard
//! shows ([`HeartbeatTelemetry`]), the observer that logs each outcome and
//! fills it ([`JobObserver`]), and the gate that skips a tick while the
//! database cannot answer ([`DbHealthGate`]). The host schedules its jobs on
//! an `aidaemon_core::ServiceHost` and plugs these in.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::{ServiceEvent, ServiceGate, ServiceObserver, ServiceRunEnd};
use async_trait::async_trait;
use chrono::Utc;
use serde::Serialize;
use tracing::{error, info};

use crate::stores::HealthCheckStore;

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

/// Skips a whole tick while the database cannot answer.
pub struct DbHealthGate {
    state: Arc<dyn HealthCheckStore>,
    healthy: AtomicBool,
}

impl DbHealthGate {
    /// A gate over `state`'s liveness probe, assumed healthy until it fails.
    pub fn new(state: Arc<dyn HealthCheckStore>) -> Self {
        Self {
            state,
            healthy: AtomicBool::new(true),
        }
    }
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
pub struct JobObserver {
    telemetry: Option<Arc<HeartbeatTelemetry>>,
}

impl JobObserver {
    /// Log every outcome, and record it in `telemetry` when given.
    pub fn new(telemetry: Option<Arc<HeartbeatTelemetry>>) -> Self {
        Self { telemetry }
    }
}

impl ServiceObserver for JobObserver {
    fn event(&self, event: ServiceEvent) {
        let telemetry = self.telemetry.as_ref();
        match event {
            ServiceEvent::Launched { service, .. } => {
                tracing::debug!(job = %service, "Heartbeat job starting");
                if let Some(t) = telemetry {
                    t.mark_started(&service);
                }
            }
            ServiceEvent::Ended {
                service, outcome, ..
            } => match outcome {
                ServiceRunEnd::Succeeded { previous_failures } => {
                    if previous_failures > 0 {
                        info!(job = %service, prev_failures = previous_failures, "Heartbeat job recovered");
                    }
                    if let Some(t) = telemetry {
                        t.mark_success(&service);
                    }
                }
                ServiceRunEnd::Failed { failures, error } => {
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
                _ => {
                    if let Some(t) = telemetry {
                        t.mark_not_running(&service);
                    }
                }
            },
            ServiceEvent::StartFailed { service, error } => {
                error!(job = %service, error = %error, "Heartbeat service failed to start");
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_records_failure_then_recovery() {
        let telemetry = Arc::new(HeartbeatTelemetry::new());
        telemetry.register_job("job", Duration::from_secs(60));
        let observer = JobObserver::new(Some(telemetry.clone()));

        observer.event(ServiceEvent::Launched {
            service: "job".into(),
            run: 1,
        });
        assert!(telemetry.snapshots()[0].is_running);
        observer.event(ServiceEvent::Ended {
            service: "job".into(),
            run: 1,
            outcome: ServiceRunEnd::Failed {
                failures: 2,
                error: "boom".into(),
            },
            elapsed: Duration::ZERO,
        });
        let failed = &telemetry.snapshots()[0];
        assert_eq!(failed.consecutive_failures, 2);
        assert_eq!(failed.last_error.as_deref(), Some("boom"));
        assert!(!failed.is_running);

        observer.event(ServiceEvent::Ended {
            service: "job".into(),
            run: 2,
            outcome: ServiceRunEnd::Succeeded {
                previous_failures: 2,
            },
            elapsed: Duration::ZERO,
        });
        let recovered = &telemetry.snapshots()[0];
        assert_eq!(recovered.consecutive_failures, 0);
        assert!(recovered.last_error.is_none());
        assert!(recovered.last_success_at.is_some());
        assert_eq!(recovered.interval_secs, 60);
    }

    struct Flaky(std::sync::atomic::AtomicBool);

    #[async_trait]
    impl HealthCheckStore for Flaky {
        async fn health_check(&self) -> anyhow::Result<()> {
            if self.0.load(Ordering::Relaxed) {
                Ok(())
            } else {
                anyhow::bail!("database is locked")
            }
        }
    }

    #[tokio::test]
    async fn gate_follows_the_database_probe() {
        let store = Arc::new(Flaky(AtomicBool::new(true)));
        let gate = DbHealthGate::new(store.clone());
        assert!(gate.ready().await);
        store.0.store(false, Ordering::Relaxed);
        assert!(!gate.ready().await);
        store.0.store(true, Ordering::Relaxed);
        assert!(gate.ready().await);
    }
}
