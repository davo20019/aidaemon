//! Alert management for health probes including threshold-based alerting and recovery notifications.

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::channels::ChannelHub;
use super::probes::{HealthProbe, ProbeResult, ProbeStatus};
use super::store::HealthProbeStore;

/// Alert types for health probes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertType {
    /// Probe is down (consecutive failures exceeded threshold)
    Down,
    /// Probe has recovered after being down
    Recovered,
    /// Probe is degraded (high latency or intermittent failures)
    Degraded,
    /// Latency threshold exceeded
    LatencyWarning,
}

impl AlertType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AlertType::Down => "down",
            AlertType::Recovered => "recovered",
            AlertType::Degraded => "degraded",
            AlertType::LatencyWarning => "latency_warning",
        }
    }
}

/// State tracking for a single probe's alert status.
#[derive(Debug, Clone)]
#[derive(Default)]
struct ProbeAlertState {
    /// Current consecutive failures
    consecutive_failures: u32,
    /// When the first failure in this streak occurred
    first_failure_at: Option<DateTime<Utc>>,
    /// Whether a "down" alert has been sent for the current failure streak
    down_alert_sent: bool,
    /// Last time any alert was sent (for deduplication)
    last_alert_at: Option<DateTime<Utc>>,
    /// Last probe status
    last_status: Option<ProbeStatus>,
}


/// Manager for health probe alerts.
///
/// Tracks consecutive failures, sends alerts via ChannelHub, and handles recovery notifications.
pub struct AlertManager {
    store: Arc<HealthProbeStore>,
    hub: Arc<ChannelHub>,
    /// In-memory state for each probe (keyed by probe ID)
    states: RwLock<HashMap<String, ProbeAlertState>>,
    /// Minimum time between repeated alerts for the same probe (default: 5 minutes)
    alert_cooldown: Duration,
}

impl AlertManager {
    /// Create a new alert manager.
    pub fn new(store: Arc<HealthProbeStore>, hub: Arc<ChannelHub>) -> Self {
        Self {
            store,
            hub,
            states: RwLock::new(HashMap::new()),
            alert_cooldown: Duration::minutes(5),
        }
    }

    /// Process a probe result and send alerts if necessary.
    pub async fn process_result(&self, probe: &HealthProbe, result: &ProbeResult) {
        let mut states = self.states.write().await;
        let state = states.entry(probe.id.clone()).or_default();

        let now = Utc::now();
        let is_healthy = result.status.is_healthy();

        // Update consecutive failures
        if is_healthy {
            // Check if we need to send a recovery alert
            if state.down_alert_sent {
                self.send_recovery_alert(probe, result, state.first_failure_at).await;
            }

            // Reset failure tracking
            state.consecutive_failures = 0;
            state.first_failure_at = None;
            state.down_alert_sent = false;
        } else {
            // Increment failures
            state.consecutive_failures += 1;
            if state.first_failure_at.is_none() {
                state.first_failure_at = Some(now);
            }

            // Check if we should send a down alert
            if state.consecutive_failures >= probe.consecutive_failures_alert && !state.down_alert_sent {
                self.send_down_alert(probe, result, state.first_failure_at.unwrap_or(now)).await;
                state.down_alert_sent = true;
                state.last_alert_at = Some(now);
            }
        }

        // Check latency threshold
        if is_healthy {
            if let (Some(threshold), Some(latency)) = (probe.latency_threshold_ms, result.latency_ms) {
                if latency > threshold {
                    // Check cooldown
                    let should_alert = state.last_alert_at
                        .map(|last| now - last >= self.alert_cooldown)
                        .unwrap_or(true);

                    if should_alert {
                        self.send_latency_alert(probe, result, threshold).await;
                        state.last_alert_at = Some(now);
                    }
                }
            }
        }

        state.last_status = Some(result.status.clone());
    }

    /// Send a "down" alert for a probe.
    async fn send_down_alert(
        &self,
        probe: &HealthProbe,
        result: &ProbeResult,
        first_failure_at: DateTime<Utc>,
    ) {
        let duration = Utc::now() - first_failure_at;
        let duration_str = format_duration(duration);

        let message = format!(
            "🔴 **Health Alert: {} is DOWN**\n\
             Target: `{}`\n\
             Status: {} for {}\n\
             Error: {}\n\
             Consecutive failures: {}",
            probe.name,
            probe.target,
            result.status.as_str(),
            duration_str,
            result.error_message.as_deref().unwrap_or("Unknown error"),
            probe.consecutive_failures_alert
        );

        info!(
            probe = %probe.name,
            target = %probe.target,
            failures = probe.consecutive_failures_alert,
            "Sending down alert"
        );

        // Send to configured alert sessions
        let sessions = &probe.alert_session_ids;
        if !sessions.is_empty() {
            self.hub.broadcast_text(sessions, &message).await;
        }

        // Record alert in database
        if let Err(e) = self.store.insert_alert(
            &probe.id,
            AlertType::Down.as_str(),
            &message,
            first_failure_at,
        ).await {
            warn!(probe = %probe.name, "Failed to record alert: {}", e);
        }
    }

    /// Send a "recovered" alert for a probe.
    async fn send_recovery_alert(
        &self,
        probe: &HealthProbe,
        result: &ProbeResult,
        first_failure_at: Option<DateTime<Utc>>,
    ) {
        let downtime = first_failure_at
            .map(|start| Utc::now() - start)
            .unwrap_or_else(Duration::zero);
        let downtime_str = format_duration(downtime);

        let latency_str = result.latency_ms
            .map(|ms| format!(" (latency: {}ms)", ms))
            .unwrap_or_default();

        let message = format!(
            "🟢 **Health Alert: {} has RECOVERED**\n\
             Target: `{}`\n\
             Downtime: {}\n\
             Status: Healthy{}",
            probe.name,
            probe.target,
            downtime_str,
            latency_str
        );

        info!(
            probe = %probe.name,
            target = %probe.target,
            downtime = %downtime_str,
            "Sending recovery alert"
        );

        let sessions = &probe.alert_session_ids;
        if !sessions.is_empty() {
            self.hub.broadcast_text(sessions, &message).await;
        }

        // Record alert in database
        if let Err(e) = self.store.insert_alert(
            &probe.id,
            AlertType::Recovered.as_str(),
            &message,
            first_failure_at.unwrap_or_else(Utc::now),
        ).await {
            warn!(probe = %probe.name, "Failed to record recovery alert: {}", e);
        }
    }

    /// Send a latency warning alert.
    async fn send_latency_alert(
        &self,
        probe: &HealthProbe,
        result: &ProbeResult,
        threshold: u32,
    ) {
        let latency = result.latency_ms.unwrap_or(0);

        let message = format!(
            "⚠️ **Health Warning: {} - High Latency**\n\
             Target: `{}`\n\
             Latency: {}ms (threshold: {}ms)\n\
             Status: {}",
            probe.name,
            probe.target,
            latency,
            threshold,
            result.status.as_str()
        );

        info!(
            probe = %probe.name,
            latency_ms = latency,
            threshold_ms = threshold,
            "Sending latency warning"
        );

        let sessions = &probe.alert_session_ids;
        if !sessions.is_empty() {
            self.hub.broadcast_text(sessions, &message).await;
        }

        // Record alert in database
        if let Err(e) = self.store.insert_alert(
            &probe.id,
            AlertType::LatencyWarning.as_str(),
            &message,
            Utc::now(),
        ).await {
            warn!(probe = %probe.name, "Failed to record latency alert: {}", e);
        }
    }

    /// Send a degradation alert.
    #[allow(dead_code)]
    pub async fn send_degradation_alert(&self, probe: &HealthProbe, details: &str) {
        let message = format!(
            "⚠️ **Health Warning: {} is DEGRADED**\n\
             Target: `{}`\n\
             Details: {}",
            probe.name,
            probe.target,
            details
        );

        info!(
            probe = %probe.name,
            target = %probe.target,
            "Sending degradation alert"
        );

        let sessions = &probe.alert_session_ids;
        if !sessions.is_empty() {
            self.hub.broadcast_text(sessions, &message).await;
        }

        // Record alert in database
        if let Err(e) = self.store.insert_alert(
            &probe.id,
            AlertType::Degraded.as_str(),
            &message,
            Utc::now(),
        ).await {
            warn!(probe = %probe.name, "Failed to record degradation alert: {}", e);
        }
    }

    /// Get the current alert state for a probe (for debugging/status).
    pub async fn get_probe_state(&self, probe_id: &str) -> Option<(u32, bool)> {
        let states = self.states.read().await;
        states.get(probe_id).map(|s| (s.consecutive_failures, s.down_alert_sent))
    }

    /// Clear all tracked states (useful for testing or reset).
    #[allow(dead_code)]
    pub async fn clear_states(&self) {
        let mut states = self.states.write().await;
        states.clear();
    }
}

/// Format a duration in a human-readable way.
fn format_duration(duration: Duration) -> String {
    let total_secs = duration.num_seconds();

    if total_secs < 60 {
        format!("{}s", total_secs)
    } else if total_secs < 3600 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        if secs > 0 {
            format!("{}m {}s", mins, secs)
        } else {
            format!("{}m", mins)
        }
    } else {
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        if mins > 0 {
            format!("{}h {}m", hours, mins)
        } else {
            format!("{}h", hours)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::seconds(30)), "30s");
        assert_eq!(format_duration(Duration::seconds(90)), "1m 30s");
        assert_eq!(format_duration(Duration::seconds(3600)), "1h");
        assert_eq!(format_duration(Duration::seconds(3660)), "1h 1m");
        assert_eq!(format_duration(Duration::seconds(7200)), "2h");
    }
}
