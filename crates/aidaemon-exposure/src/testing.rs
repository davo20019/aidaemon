//! A scripted adapter for supervisor and wiring tests: no network, no child
//! processes.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::watch;

use crate::{
    ExposedService, Exposure, ExposureAdapter, ExposureError, ExposureHealth, OriginStability,
    PublicEndpoint,
};

/// What the current exposure reports next.
pub enum FakeStep {
    Up(&'static str),
    UpEphemeral(&'static str),
    Degraded(&'static str, &'static str),
    Down(&'static str),
}

#[derive(Default)]
struct Shared {
    attempts: Vec<tokio::time::Instant>,
    current: Option<watch::Sender<ExposureHealth>>,
    exposures: usize,
    stopped: usize,
    fail_expose: Option<String>,
}

/// Counts `expose` calls and stops; [`Self::step`] drives the latest
/// exposure.
#[derive(Clone)]
pub struct FakeAdapter {
    stability: OriginStability,
    shared: Arc<Mutex<Shared>>,
}

impl FakeAdapter {
    pub fn new(stability: OriginStability) -> Arc<Self> {
        Arc::new(Self {
            stability,
            shared: Arc::default(),
        })
    }

    /// Every `expose` fails with `reason` from now on.
    pub fn fail_expose(&self, reason: &str) {
        self.shared.lock().unwrap().fail_expose = Some(reason.to_string());
    }

    pub fn step(&self, step: FakeStep) {
        let stable = |origin: &str| PublicEndpoint {
            origin: origin.to_string(),
            stability: OriginStability::Stable,
        };
        let health = match step {
            FakeStep::Up(origin) => ExposureHealth::Up(stable(origin)),
            FakeStep::UpEphemeral(origin) => ExposureHealth::Up(PublicEndpoint {
                origin: origin.to_string(),
                stability: OriginStability::Ephemeral,
            }),
            FakeStep::Degraded(origin, reason) => ExposureHealth::Degraded {
                endpoint: stable(origin),
                reason: reason.to_string(),
            },
            FakeStep::Down(reason) => ExposureHealth::Down {
                reason: reason.to_string(),
            },
        };
        if let Some(tx) = &self.shared.lock().unwrap().current {
            tx.send_replace(health);
        }
    }

    /// `expose` calls, successful or not.
    pub fn attempts(&self) -> usize {
        self.shared.lock().unwrap().attempts.len()
    }

    /// Time between consecutive `expose` calls.
    pub fn attempt_gaps(&self) -> Vec<Duration> {
        let shared = self.shared.lock().unwrap();
        shared
            .attempts
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect()
    }

    /// Exposures handed out.
    pub fn exposures(&self) -> usize {
        self.shared.lock().unwrap().exposures
    }

    /// Exposures stopped.
    pub fn stopped(&self) -> usize {
        self.shared.lock().unwrap().stopped
    }
}

#[async_trait]
impl ExposureAdapter for FakeAdapter {
    fn id(&self) -> &str {
        "fake"
    }

    fn stability(&self) -> OriginStability {
        self.stability
    }

    async fn expose(
        &self,
        _service: &ExposedService,
        _local: SocketAddr,
    ) -> Result<Box<dyn Exposure>, ExposureError> {
        let mut shared = self.shared.lock().unwrap();
        shared.attempts.push(tokio::time::Instant::now());
        if let Some(reason) = &shared.fail_expose {
            return Err(ExposureError::Failed(reason.clone()));
        }
        let (tx, rx) = watch::channel(ExposureHealth::Starting);
        shared.current = Some(tx);
        shared.exposures += 1;
        Ok(Box::new(FakeExposure {
            rx,
            shared: self.shared.clone(),
        }))
    }
}

struct FakeExposure {
    rx: watch::Receiver<ExposureHealth>,
    shared: Arc<Mutex<Shared>>,
}

#[async_trait]
impl Exposure for FakeExposure {
    fn watch(&self) -> watch::Receiver<ExposureHealth> {
        self.rx.clone()
    }

    async fn stop(self: Box<Self>) -> Result<(), ExposureError> {
        self.shared.lock().unwrap().stopped += 1;
        Ok(())
    }
}
