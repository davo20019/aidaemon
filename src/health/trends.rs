//! Trend analysis for health probes including stats calculation and degradation detection.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Statistics for a probe over a time period.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeStats {
    pub probe_id: String,
    pub check_count: u32,
    pub healthy_count: u32,
    pub uptime_percent: f64,
    pub avg_latency_ms: Option<u32>,
    pub p95_latency_ms: Option<u32>,
    pub is_degraded: bool,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
}

impl ProbeStats {
    /// Create new stats for a probe.
    pub fn new(probe_id: String, period_start: DateTime<Utc>, period_end: DateTime<Utc>) -> Self {
        Self {
            probe_id,
            check_count: 0,
            healthy_count: 0,
            uptime_percent: 100.0,
            avg_latency_ms: None,
            p95_latency_ms: None,
            is_degraded: false,
            period_start,
            period_end,
        }
    }

    /// Calculate degradation based on thresholds.
    ///
    /// A probe is considered degraded if:
    /// - Uptime is below 99%
    /// - P95 latency is more than 2x the average latency
    /// - Average latency exceeds the provided threshold (if any)
    pub fn check_degradation(&mut self, latency_threshold_ms: Option<u32>) {
        // Uptime-based degradation
        if self.uptime_percent < 99.0 {
            self.is_degraded = true;
            return;
        }

        // Latency-based degradation
        if let (Some(avg), Some(p95)) = (self.avg_latency_ms, self.p95_latency_ms) {
            // P95 > 2x average indicates high variance/jitter
            if p95 > avg * 2 {
                self.is_degraded = true;
                return;
            }

            // Check against explicit threshold
            if let Some(threshold) = latency_threshold_ms {
                if p95 > threshold {
                    self.is_degraded = true;
                    return;
                }
            }
        }

        self.is_degraded = false;
    }

    /// Get a human-readable status summary.
    pub fn status_summary(&self) -> String {
        if self.check_count == 0 {
            return "No data".to_string();
        }

        let status = if self.is_degraded {
            "Degraded"
        } else if self.uptime_percent >= 99.9 {
            "Excellent"
        } else if self.uptime_percent >= 99.0 {
            "Good"
        } else if self.uptime_percent >= 95.0 {
            "Fair"
        } else {
            "Poor"
        };

        let latency_str = match self.avg_latency_ms {
            Some(ms) => format!(", avg {}ms", ms),
            None => String::new(),
        };

        format!(
            "{} ({:.2}% uptime{})",
            status, self.uptime_percent, latency_str
        )
    }
}

/// Trend analyzer for detecting patterns across multiple time windows.
pub struct TrendAnalyzer;

impl TrendAnalyzer {
    /// Compare stats across two time periods to detect trend direction.
    ///
    /// Returns a tuple of (is_improving, change_description)
    pub fn compare_periods(current: &ProbeStats, previous: &ProbeStats) -> (bool, String) {
        if current.check_count == 0 || previous.check_count == 0 {
            return (true, "Insufficient data for comparison".to_string());
        }

        let uptime_diff = current.uptime_percent - previous.uptime_percent;
        let latency_diff = match (current.avg_latency_ms, previous.avg_latency_ms) {
            (Some(curr), Some(prev)) => Some(curr as i64 - prev as i64),
            _ => None,
        };

        // Determine if improving
        let uptime_improving = uptime_diff >= 0.0;
        let latency_improving = latency_diff.map(|d| d <= 0).unwrap_or(true);
        let is_improving = uptime_improving && latency_improving;

        // Build description
        let mut parts = Vec::new();

        if uptime_diff.abs() >= 0.1 {
            let direction = if uptime_diff > 0.0 { "up" } else { "down" };
            parts.push(format!("uptime {} {:.2}%", direction, uptime_diff.abs()));
        }

        if let Some(diff) = latency_diff {
            if diff.abs() >= 10 {
                let direction = if diff < 0 { "improved" } else { "increased" };
                parts.push(format!("latency {} by {}ms", direction, diff.abs()));
            }
        }

        let description = if parts.is_empty() {
            "No significant change".to_string()
        } else {
            parts.join(", ")
        };

        (is_improving, description)
    }

    /// Detect if a probe is in a failure streak.
    pub fn is_failure_streak(consecutive_failures: u32, threshold: u32) -> bool {
        consecutive_failures >= threshold
    }

    /// Calculate a health score (0-100) based on stats.
    pub fn health_score(stats: &ProbeStats) -> u32 {
        if stats.check_count == 0 {
            return 100; // No data = assume healthy
        }

        let mut score = 100u32;

        // Uptime penalty (up to -50 points)
        if stats.uptime_percent < 100.0 {
            let uptime_penalty = ((100.0 - stats.uptime_percent) * 0.5) as u32;
            score = score.saturating_sub(uptime_penalty.min(50));
        }

        // Latency penalty (up to -30 points)
        if let Some(p95) = stats.p95_latency_ms {
            if p95 > 1000 {
                let latency_penalty = ((p95 - 1000) / 100).min(30);
                score = score.saturating_sub(latency_penalty);
            }
        }

        // Degradation penalty
        if stats.is_degraded {
            score = score.saturating_sub(20);
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_stats_degradation_uptime() {
        let mut stats = ProbeStats {
            probe_id: "test".to_string(),
            check_count: 100,
            healthy_count: 98,
            uptime_percent: 98.0,
            avg_latency_ms: Some(100),
            p95_latency_ms: Some(150),
            is_degraded: false,
            period_start: Utc::now(),
            period_end: Utc::now(),
        };

        stats.check_degradation(None);
        assert!(stats.is_degraded);
    }

    #[test]
    fn test_probe_stats_degradation_latency() {
        let mut stats = ProbeStats {
            probe_id: "test".to_string(),
            check_count: 100,
            healthy_count: 100,
            uptime_percent: 100.0,
            avg_latency_ms: Some(100),
            p95_latency_ms: Some(250), // > 2x avg
            is_degraded: false,
            period_start: Utc::now(),
            period_end: Utc::now(),
        };

        stats.check_degradation(None);
        assert!(stats.is_degraded);
    }

    #[test]
    fn test_probe_stats_healthy() {
        let mut stats = ProbeStats {
            probe_id: "test".to_string(),
            check_count: 100,
            healthy_count: 100,
            uptime_percent: 100.0,
            avg_latency_ms: Some(100),
            p95_latency_ms: Some(150),
            is_degraded: false,
            period_start: Utc::now(),
            period_end: Utc::now(),
        };

        stats.check_degradation(None);
        assert!(!stats.is_degraded);
    }

    #[test]
    fn test_health_score() {
        let perfect = ProbeStats {
            uptime_percent: 100.0,
            check_count: 100,
            p95_latency_ms: Some(500),
            is_degraded: false,
            ..Default::default()
        };
        assert_eq!(TrendAnalyzer::health_score(&perfect), 100);

        let degraded = ProbeStats {
            uptime_percent: 95.0,
            check_count: 100,
            p95_latency_ms: Some(2000),
            is_degraded: true,
            ..Default::default()
        };
        // -2.5 for uptime (5% * 0.5), -10 for latency ((2000-1000)/100), -20 for degraded
        // 100 - 2 - 10 - 20 = 68 (approximately)
        let score = TrendAnalyzer::health_score(&degraded);
        assert!(score < 80);
        assert!(score > 50);
    }

    #[test]
    fn test_status_summary() {
        let stats = ProbeStats {
            uptime_percent: 99.95,
            check_count: 100,
            avg_latency_ms: Some(150),
            is_degraded: false,
            ..Default::default()
        };
        let summary = stats.status_summary();
        assert!(summary.contains("Excellent"));
        assert!(summary.contains("99.95%"));
        assert!(summary.contains("150ms"));
    }
}
