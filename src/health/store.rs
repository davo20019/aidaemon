//! Health probe storage with database operations for probes, results, and alerts.

use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;

use super::probes::{HealthProbe, ProbeResult, ProbeStatus, ProbeType};
use super::trends::ProbeStats;

/// Database operations for health probes.
pub struct HealthProbeStore {
    pool: SqlitePool,
}

impl HealthProbeStore {
    /// Create a new store and initialize database tables.
    pub async fn new(pool: SqlitePool) -> anyhow::Result<Self> {
        let store = Self { pool };
        store.create_tables().await?;
        Ok(store)
    }

    async fn create_tables(&self) -> anyhow::Result<()> {
        crate::db::migrations::migrate_health_probes(&self.pool).await
    }

    // ==================== Probe CRUD ====================

    /// Insert or update a probe (upsert by name for config source).
    pub async fn upsert_probe(&self, probe: &HealthProbe) -> anyhow::Result<()> {
        let now = Utc::now();
        let config_json = serde_json::to_string(&probe.config)?;
        let alert_sessions_json = serde_json::to_string(&probe.alert_session_ids)?;
        let probe_type_str = probe.probe_type.as_str();

        sqlx::query(
            "INSERT INTO health_probes (
                id, name, description, probe_type, target, schedule, source,
                config, consecutive_failures_alert, latency_threshold_ms,
                alert_session_ids, is_paused, next_run_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                description = excluded.description,
                probe_type = excluded.probe_type,
                target = excluded.target,
                schedule = excluded.schedule,
                config = excluded.config,
                consecutive_failures_alert = excluded.consecutive_failures_alert,
                latency_threshold_ms = excluded.latency_threshold_ms,
                alert_session_ids = excluded.alert_session_ids,
                updated_at = excluded.updated_at",
        )
        .bind(&probe.id)
        .bind(&probe.name)
        .bind(&probe.description)
        .bind(probe_type_str)
        .bind(&probe.target)
        .bind(&probe.schedule)
        .bind(&probe.source)
        .bind(&config_json)
        .bind(probe.consecutive_failures_alert as i32)
        .bind(probe.latency_threshold_ms.map(|v| v as i64))
        .bind(&alert_sessions_json)
        .bind(probe.is_paused as i32)
        .bind(probe.next_run_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get a probe by ID.
    pub async fn get_probe(&self, id: &str) -> anyhow::Result<Option<HealthProbe>> {
        let row = sqlx::query("SELECT * FROM health_probes WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;

        match row {
            Some(row) => Ok(Some(self.row_to_probe(&row)?)),
            None => Ok(None),
        }
    }

    /// Get a probe by name.
    pub async fn get_probe_by_name(&self, name: &str) -> anyhow::Result<Option<HealthProbe>> {
        let row = sqlx::query("SELECT * FROM health_probes WHERE name = ?")
            .bind(name)
            .fetch_optional(&self.pool)
            .await?;

        match row {
            Some(row) => Ok(Some(self.row_to_probe(&row)?)),
            None => Ok(None),
        }
    }

    /// List all probes.
    pub async fn list_probes(&self) -> anyhow::Result<Vec<HealthProbe>> {
        let rows = sqlx::query("SELECT * FROM health_probes ORDER BY name ASC")
            .fetch_all(&self.pool)
            .await?;

        let mut probes = Vec::with_capacity(rows.len());
        for row in rows {
            probes.push(self.row_to_probe(&row)?);
        }
        Ok(probes)
    }

    /// Get probes that are due to run.
    pub async fn get_due_probes(&self, now: DateTime<Utc>) -> anyhow::Result<Vec<HealthProbe>> {
        let rows =
            sqlx::query("SELECT * FROM health_probes WHERE next_run_at <= ? AND is_paused = 0")
                .bind(now.to_rfc3339())
                .fetch_all(&self.pool)
                .await?;

        let mut probes = Vec::with_capacity(rows.len());
        for row in rows {
            probes.push(self.row_to_probe(&row)?);
        }
        Ok(probes)
    }

    /// Update probe run times after execution.
    pub async fn update_probe_run(
        &self,
        id: &str,
        last_run: DateTime<Utc>,
        next_run: DateTime<Utc>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE health_probes SET last_run_at = ?, next_run_at = ?, updated_at = ? WHERE id = ?"
        )
        .bind(last_run.to_rfc3339())
        .bind(next_run.to_rfc3339())
        .bind(Utc::now().to_rfc3339())
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Pause a probe.
    pub async fn pause_probe(&self, id: &str) -> anyhow::Result<bool> {
        let result =
            sqlx::query("UPDATE health_probes SET is_paused = 1, updated_at = ? WHERE id = ?")
                .bind(Utc::now().to_rfc3339())
                .bind(id)
                .execute(&self.pool)
                .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Resume a paused probe.
    pub async fn resume_probe(&self, id: &str, next_run: DateTime<Utc>) -> anyhow::Result<bool> {
        let result = sqlx::query(
            "UPDATE health_probes SET is_paused = 0, next_run_at = ?, updated_at = ? WHERE id = ?",
        )
        .bind(next_run.to_rfc3339())
        .bind(Utc::now().to_rfc3339())
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Delete a probe by ID.
    pub async fn delete_probe(&self, id: &str) -> anyhow::Result<bool> {
        let result = sqlx::query("DELETE FROM health_probes WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Remove config-sourced probes not in the provided list.
    pub async fn remove_stale_config_probes(&self, current_names: &[&str]) -> anyhow::Result<u64> {
        if current_names.is_empty() {
            let result = sqlx::query("DELETE FROM health_probes WHERE source = 'config'")
                .execute(&self.pool)
                .await?;
            return Ok(result.rows_affected());
        }

        let placeholders: Vec<String> = current_names.iter().map(|_| "?".to_string()).collect();
        let query_str = format!(
            "DELETE FROM health_probes WHERE source = 'config' AND name NOT IN ({})",
            placeholders.join(", ")
        );
        let mut query = sqlx::query(&query_str);
        for name in current_names {
            query = query.bind(name);
        }
        let result = query.execute(&self.pool).await?;
        Ok(result.rows_affected())
    }

    fn row_to_probe(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<HealthProbe> {
        let probe_type_str: String = row.get("probe_type");
        let config_json: String = row.get("config");
        let alert_sessions_json: Option<String> = row.get("alert_session_ids");
        let last_run_str: Option<String> = row.get("last_run_at");
        let next_run_str: String = row.get("next_run_at");
        let created_str: String = row.get("created_at");
        let updated_str: String = row.get("updated_at");

        Ok(HealthProbe {
            id: row.get("id"),
            name: row.get("name"),
            description: row.get("description"),
            probe_type: ProbeType::from_str(&probe_type_str),
            target: row.get("target"),
            schedule: row.get("schedule"),
            source: row.get("source"),
            config: serde_json::from_str(&config_json).unwrap_or_default(),
            consecutive_failures_alert: row.get::<i32, _>("consecutive_failures_alert") as u32,
            latency_threshold_ms: row
                .get::<Option<i64>, _>("latency_threshold_ms")
                .map(|v| v as u32),
            alert_session_ids: alert_sessions_json
                .and_then(|j| serde_json::from_str(&j).ok())
                .unwrap_or_default(),
            is_paused: row.get::<i32, _>("is_paused") != 0,
            last_run_at: last_run_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            next_run_at: DateTime::parse_from_rfc3339(&next_run_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            created_at: DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&updated_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }

    // ==================== Results ====================

    /// Insert a probe result.
    pub async fn insert_result(&self, result: &ProbeResult) -> anyhow::Result<i64> {
        let status_str = result.status.as_str();
        let row = sqlx::query(
            "INSERT INTO probe_results (probe_id, status, latency_ms, error_message, response_body, checked_at)
             VALUES (?, ?, ?, ?, ?, ?)"
        )
        .bind(&result.probe_id)
        .bind(status_str)
        .bind(result.latency_ms.map(|v| v as i64))
        .bind(&result.error_message)
        .bind(&result.response_body)
        .bind(result.checked_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(row.last_insert_rowid())
    }

    /// Get recent results for a probe.
    pub async fn get_results(
        &self,
        probe_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ProbeResult>> {
        let rows = sqlx::query(
            "SELECT * FROM probe_results WHERE probe_id = ? ORDER BY checked_at DESC LIMIT ?",
        )
        .bind(probe_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            results.push(self.row_to_result(&row)?);
        }
        Ok(results)
    }

    /// Get results for a probe within a time range.
    pub async fn get_results_in_range(
        &self,
        probe_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<Vec<ProbeResult>> {
        let rows = sqlx::query(
            "SELECT * FROM probe_results
             WHERE probe_id = ? AND checked_at >= ? AND checked_at <= ?
             ORDER BY checked_at DESC",
        )
        .bind(probe_id)
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            results.push(self.row_to_result(&row)?);
        }
        Ok(results)
    }

    /// Get the most recent result for a probe.
    pub async fn get_latest_result(&self, probe_id: &str) -> anyhow::Result<Option<ProbeResult>> {
        let row = sqlx::query(
            "SELECT * FROM probe_results WHERE probe_id = ? ORDER BY checked_at DESC LIMIT 1",
        )
        .bind(probe_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(self.row_to_result(&row)?)),
            None => Ok(None),
        }
    }

    /// Count consecutive failures for a probe.
    pub async fn count_consecutive_failures(&self, probe_id: &str) -> anyhow::Result<u32> {
        let rows = sqlx::query(
            "SELECT status FROM probe_results WHERE probe_id = ? ORDER BY checked_at DESC LIMIT 100"
        )
        .bind(probe_id)
        .fetch_all(&self.pool)
        .await?;

        let mut count = 0u32;
        for row in rows {
            let status: String = row.get("status");
            if status == "healthy" {
                break;
            }
            count += 1;
        }
        Ok(count)
    }

    /// Delete results older than a cutoff date.
    pub async fn delete_old_results(&self, cutoff: DateTime<Utc>) -> anyhow::Result<u64> {
        let result = sqlx::query("DELETE FROM probe_results WHERE checked_at < ?")
            .bind(cutoff.to_rfc3339())
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    fn row_to_result(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<ProbeResult> {
        let status_str: String = row.get("status");
        let checked_str: String = row.get("checked_at");

        Ok(ProbeResult {
            id: row.get("id"),
            probe_id: row.get("probe_id"),
            status: ProbeStatus::from_str(&status_str),
            latency_ms: row.get::<Option<i64>, _>("latency_ms").map(|v| v as u32),
            error_message: row.get("error_message"),
            response_body: row.get("response_body"),
            checked_at: DateTime::parse_from_rfc3339(&checked_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }

    // ==================== Alerts ====================

    /// Insert an alert record.
    pub async fn insert_alert(
        &self,
        probe_id: &str,
        alert_type: &str,
        message: &str,
        first_failure_at: DateTime<Utc>,
    ) -> anyhow::Result<i64> {
        let row = sqlx::query(
            "INSERT INTO probe_alerts (probe_id, alert_type, message, sent_at, first_failure_at)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(probe_id)
        .bind(alert_type)
        .bind(message)
        .bind(Utc::now().to_rfc3339())
        .bind(first_failure_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(row.last_insert_rowid())
    }

    /// Get the last alert of a specific type for a probe.
    pub async fn get_last_alert(
        &self,
        probe_id: &str,
        alert_type: &str,
    ) -> anyhow::Result<Option<DateTime<Utc>>> {
        let row = sqlx::query(
            "SELECT sent_at FROM probe_alerts
             WHERE probe_id = ? AND alert_type = ?
             ORDER BY sent_at DESC LIMIT 1",
        )
        .bind(probe_id)
        .bind(alert_type)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let sent_str: String = row.get("sent_at");
                Ok(DateTime::parse_from_rfc3339(&sent_str)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc)))
            }
            None => Ok(None),
        }
    }

    // ==================== Trend Queries ====================

    /// Calculate stats for a probe over a time period.
    pub async fn calculate_stats(
        &self,
        probe_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<ProbeStats> {
        let results = self.get_results_in_range(probe_id, start, end).await?;

        if results.is_empty() {
            return Ok(ProbeStats::default());
        }

        let total = results.len() as f64;
        let healthy_count = results
            .iter()
            .filter(|r| r.status == ProbeStatus::Healthy)
            .count() as f64;
        let uptime_percent = (healthy_count / total) * 100.0;

        // Collect latencies
        let mut latencies: Vec<u32> = results.iter().filter_map(|r| r.latency_ms).collect();

        let (avg_latency_ms, p95_latency_ms) = if latencies.is_empty() {
            (None, None)
        } else {
            let sum: u64 = latencies.iter().map(|&v| v as u64).sum();
            let avg = (sum / latencies.len() as u64) as u32;

            latencies.sort_unstable();
            let p95_idx = (latencies.len() as f64 * 0.95).ceil() as usize - 1;
            let p95 = latencies.get(p95_idx.min(latencies.len() - 1)).copied();

            (Some(avg), p95)
        };

        // Detect degradation: p95 > 2x avg or uptime < 99%
        let is_degraded = match (avg_latency_ms, p95_latency_ms) {
            (Some(avg), Some(p95)) => p95 > avg * 2 || uptime_percent < 99.0,
            _ => uptime_percent < 99.0,
        };

        Ok(ProbeStats {
            probe_id: probe_id.to_string(),
            check_count: results.len() as u32,
            healthy_count: healthy_count as u32,
            uptime_percent,
            avg_latency_ms,
            p95_latency_ms,
            is_degraded,
            period_start: start,
            period_end: end,
        })
    }

    /// Get summary stats for all probes.
    pub async fn get_all_probe_stats(
        &self,
        hours: u32,
    ) -> anyhow::Result<HashMap<String, ProbeStats>> {
        let end = Utc::now();
        let start = end - chrono::Duration::hours(hours as i64);

        let probes = self.list_probes().await?;
        let mut stats_map = HashMap::new();

        for probe in probes {
            let stats = self.calculate_stats(&probe.id, start, end).await?;
            stats_map.insert(probe.id, stats);
        }

        Ok(stats_map)
    }
}
