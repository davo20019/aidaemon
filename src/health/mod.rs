//! First-class health probe system with structured metrics, trend analysis, and alerting.

pub mod alerts;
pub mod probes;
pub mod store;
pub mod trends;

use chrono::{DateTime, Utc};
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

use crate::channels::ChannelHub;
use crate::config::HealthProbeConfig;
use crate::cron_utils::{compute_next_run, parse_schedule};

pub use alerts::AlertManager;
pub use probes::{HealthProbe, ProbeConfig, ProbeExecutor, ProbeResult, ProbeType};
pub use store::HealthProbeStore;
pub use trends::TrendAnalyzer;

/// Manager for health probes with background tick loop.
pub struct HealthProbeManager {
    store: Arc<HealthProbeStore>,
    alerts: Arc<AlertManager>,
    tick_interval: Duration,
}

impl HealthProbeManager {
    /// Create a new health probe manager.
    pub fn new(
        store: Arc<HealthProbeStore>,
        hub: Arc<ChannelHub>,
        tick_interval_secs: u64,
    ) -> Self {
        let alerts = Arc::new(AlertManager::new(store.clone(), hub));

        Self {
            store,
            alerts,
            tick_interval: Duration::from_secs(tick_interval_secs),
        }
    }

    /// Get a reference to the store for external use.
    pub fn store(&self) -> &Arc<HealthProbeStore> {
        &self.store
    }

    /// Seed probes from configuration.
    ///
    /// Uses upsert by name for config-sourced probes and removes stale entries.
    pub async fn seed_from_config(
        &self,
        probes: &[HealthProbeConfig],
        default_alert_sessions: &[String],
    ) {
        let now = Utc::now();

        for probe_config in probes {
            // Parse schedule to cron
            let cron_expr = match parse_schedule(&probe_config.schedule) {
                Ok(expr) => expr,
                Err(e) => {
                    error!(
                        name = %probe_config.name,
                        schedule = %probe_config.schedule,
                        "Failed to parse health probe schedule: {}",
                        e
                    );
                    continue;
                }
            };

            // Compute next run time
            let next_run = match compute_next_run(&cron_expr) {
                Ok(dt) => dt,
                Err(e) => {
                    error!(
                        name = %probe_config.name,
                        "Failed to compute next run for health probe: {}",
                        e
                    );
                    continue;
                }
            };

            // Build probe configuration
            let config = ProbeConfig {
                timeout_secs: probe_config.config.timeout_secs.unwrap_or(10),
                expected_status: probe_config.config.expected_status,
                expected_body: probe_config.config.expected_body.clone(),
                method: probe_config
                    .config
                    .method
                    .clone()
                    .unwrap_or_else(|| "GET".to_string()),
                headers: probe_config.config.headers.clone().unwrap_or_default(),
                max_age_secs: probe_config.config.max_age_secs,
                expected_exit_code: probe_config.config.expected_exit_code,
            };

            // Determine alert sessions - use probe-specific or fall back to defaults
            let alert_sessions = if probe_config.alert_session_ids.is_empty() {
                default_alert_sessions.to_vec()
            } else {
                probe_config.alert_session_ids.clone()
            };

            let probe = HealthProbe {
                id: uuid::Uuid::new_v4().to_string(),
                name: probe_config.name.clone(),
                description: probe_config.description.clone(),
                probe_type: ProbeType::from_str(&probe_config.probe_type),
                target: probe_config.target.clone(),
                schedule: cron_expr,
                source: "config".to_string(),
                config,
                consecutive_failures_alert: probe_config.consecutive_failures_alert.unwrap_or(3),
                latency_threshold_ms: probe_config.latency_threshold_ms,
                alert_session_ids: alert_sessions,
                is_paused: false,
                last_run_at: None,
                next_run_at: next_run,
                created_at: now,
                updated_at: now,
            };

            match self.store.upsert_probe(&probe).await {
                Ok(_) => info!(
                    name = %probe_config.name,
                    probe_type = %probe_config.probe_type,
                    schedule = %probe_config.schedule,
                    "Seeded health probe from config"
                ),
                Err(e) => error!(
                    name = %probe_config.name,
                    "Failed to seed health probe: {}",
                    e
                ),
            }
        }

        // Remove config-sourced probes that are no longer in config
        let config_names: Vec<&str> = probes.iter().map(|p| p.name.as_str()).collect();
        match self.store.remove_stale_config_probes(&config_names).await {
            Ok(removed) if removed > 0 => {
                info!(count = removed, "Removed stale config health probes");
            }
            Err(e) => {
                warn!("Failed to remove stale config probes: {}", e);
            }
            _ => {}
        }
    }

    /// Spawn the background tick loop.
    pub fn spawn(self: Arc<Self>) {
        tokio::spawn(async move {
            info!(
                interval_secs = self.tick_interval.as_secs(),
                "Health probe manager started"
            );

            loop {
                tokio::time::sleep(self.tick_interval).await;

                if let Err(e) = self.tick().await {
                    error!("Health probe tick error: {}", e);
                }
            }
        });
    }

    /// Check for due probes and execute them.
    async fn tick(&self) -> anyhow::Result<()> {
        let now = Utc::now();
        let due_probes = self.store.get_due_probes(now).await?;

        if due_probes.is_empty() {
            return Ok(());
        }

        info!(count = due_probes.len(), "Running due health probes");

        for probe in due_probes {
            // Execute the probe
            let result = ProbeExecutor::execute(&probe).await;

            info!(
                probe = %probe.name,
                status = %result.status.as_str(),
                latency_ms = ?result.latency_ms,
                "Probe check completed"
            );

            // Store the result
            if let Err(e) = self.store.insert_result(&result).await {
                warn!(probe = %probe.name, "Failed to store probe result: {}", e);
            }

            // Process alerts
            self.alerts.process_result(&probe, &result).await;

            // Update next run time
            let next_run = match compute_next_run(&probe.schedule) {
                Ok(dt) => dt,
                Err(e) => {
                    warn!(
                        probe = %probe.name,
                        "Failed to compute next run, using +1 minute: {}",
                        e
                    );
                    now + chrono::Duration::minutes(1)
                }
            };

            if let Err(e) = self.store.update_probe_run(&probe.id, now, next_run).await {
                warn!(probe = %probe.name, "Failed to update probe run time: {}", e);
            }
        }

        Ok(())
    }

    /// Run a single probe immediately (on-demand execution).
    pub async fn run_probe_now(&self, probe_id: &str) -> anyhow::Result<ProbeResult> {
        let probe = self
            .store
            .get_probe(probe_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Probe not found: {}", probe_id))?;

        let result = ProbeExecutor::execute(&probe).await;

        // Store the result
        self.store.insert_result(&result).await?;

        // Process alerts
        self.alerts.process_result(&probe, &result).await;

        // Update last run time (but don't change next_run_at for on-demand runs)
        let now = Utc::now();
        if let Err(e) = self
            .store
            .update_probe_run(&probe.id, now, probe.next_run_at)
            .await
        {
            warn!(probe = %probe.name, "Failed to update last run time: {}", e);
        }

        Ok(result)
    }

    /// Get current status for all probes.
    pub async fn get_all_status(&self) -> anyhow::Result<Vec<ProbeStatusSummary>> {
        let probes = self.store.list_probes().await?;
        let mut summaries = Vec::with_capacity(probes.len());

        for probe in probes {
            let latest = self.store.get_latest_result(&probe.id).await?;
            let consecutive_failures = self.store.count_consecutive_failures(&probe.id).await?;
            let alert_state = self.alerts.get_probe_state(&probe.id).await;

            summaries.push(ProbeStatusSummary {
                id: probe.id,
                name: probe.name,
                probe_type: probe.probe_type.as_str().to_string(),
                target: probe.target,
                is_paused: probe.is_paused,
                last_status: latest.as_ref().map(|r| r.status.as_str().to_string()),
                last_latency_ms: latest.as_ref().and_then(|r| r.latency_ms),
                last_checked: latest.as_ref().map(|r| r.checked_at),
                next_run: probe.next_run_at,
                consecutive_failures,
                has_active_alert: alert_state.map(|(_, sent)| sent).unwrap_or(false),
            });
        }

        Ok(summaries)
    }

    /// Cleanup old probe results beyond retention period.
    pub async fn cleanup_old_results(&self, retention_days: u32) -> anyhow::Result<u64> {
        let cutoff = Utc::now() - chrono::Duration::days(retention_days as i64);
        let deleted = self.store.delete_old_results(cutoff).await?;

        if deleted > 0 {
            info!(deleted, retention_days, "Cleaned up old probe results");
        }

        Ok(deleted)
    }
}

/// Summary of a probe's current status for API responses.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProbeStatusSummary {
    pub id: String,
    pub name: String,
    pub probe_type: String,
    pub target: String,
    pub is_paused: bool,
    pub last_status: Option<String>,
    pub last_latency_ms: Option<u32>,
    pub last_checked: Option<DateTime<Utc>>,
    pub next_run: DateTime<Utc>,
    pub consecutive_failures: u32,
    pub has_active_alert: bool,
}

/// Spawn a cleanup task that runs daily.
pub fn spawn_cleanup_task(manager: Arc<HealthProbeManager>, retention_days: u32) {
    tokio::spawn(async move {
        loop {
            // Run at 3:40 AM (after other cleanup tasks)
            let now = chrono::Utc::now();
            let next_340am = {
                let today_340am = now.date_naive().and_hms_opt(3, 40, 0).unwrap();
                let today_340am_utc = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
                    today_340am,
                    chrono::Utc,
                );
                if now < today_340am_utc {
                    today_340am_utc
                } else {
                    today_340am_utc + chrono::Duration::days(1)
                }
            };

            let sleep_duration = (next_340am - now)
                .to_std()
                .unwrap_or(Duration::from_secs(3600));
            tokio::time::sleep(sleep_duration).await;

            match manager.cleanup_old_results(retention_days).await {
                Ok(deleted) => {
                    if deleted > 0 {
                        info!(deleted, "Health probe result cleanup complete");
                    }
                }
                Err(e) => {
                    error!("Health probe result cleanup failed: {}", e);
                }
            }
        }
    });
}
