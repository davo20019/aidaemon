use anyhow::Context;
use chrono::{DateTime, Duration, Utc};
use serde::Serialize;
use serde_json::{json, Value};
use sqlx::{Row, Sqlite, SqlitePool, Transaction};
use uuid::Uuid;

use crate::config::NodeMonitoringConfig;
use crate::traits::NotificationEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorComparison {
    Above,
    Below,
}

impl MonitorComparison {
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "above" => Ok(Self::Above),
            "below" => Ok(Self::Below),
            _ => anyhow::bail!("comparison must be 'above' or 'below'"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Above => "above",
            Self::Below => "below",
        }
    }

    fn condition_matches(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::Above => value > threshold,
            Self::Below => value < threshold,
        }
    }

    fn recovery_matches(self, value: f64, clear_threshold: f64) -> bool {
        match self {
            Self::Above => value <= clear_threshold,
            Self::Below => value >= clear_threshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CreateNodeMonitor {
    pub name: String,
    pub owner_session_id: String,
    pub node: Option<String>,
    pub capability_id: String,
    pub comparison: MonitorComparison,
    pub threshold: f64,
    pub clear_threshold: f64,
    pub duration_seconds: u64,
    pub stale_after_seconds: Option<u64>,
    pub offline_after_seconds: Option<u64>,
    pub repeat_seconds: u64,
    pub send_recovery: bool,
    pub duration_minutes: u64,
    pub mandate_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeMonitorView {
    pub monitor_id: String,
    pub name: String,
    pub node: String,
    pub capability_id: String,
    pub unit: String,
    pub comparison: String,
    pub threshold: f64,
    pub clear_threshold: f64,
    pub duration_seconds: u64,
    pub stale_after_seconds: u64,
    pub offline_after_seconds: u64,
    pub repeat_seconds: u64,
    pub send_recovery: bool,
    pub status: String,
    pub threshold_state: String,
    pub availability_state: String,
    pub last_value: Option<f64>,
    pub last_received_at: Option<String>,
    pub mandate_id: Option<String>,
    pub expires_at: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeMonitorEventView {
    pub event_kind: String,
    pub evidence: Value,
    pub created_at: String,
    pub delivered: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeSensorHistoryPoint {
    pub value: f64,
    pub unit: String,
    pub server_received_at: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct NodeMonitorMaintenanceStats {
    pub evaluated: u64,
    pub alerts: u64,
    pub recoveries: u64,
    pub expired: u64,
    pub suspended: u64,
    pub history_rows_pruned: u64,
    pub event_rows_pruned: u64,
}

#[derive(Clone)]
pub struct NodeMonitoringService {
    pool: SqlitePool,
    config: NodeMonitoringConfig,
}

#[derive(Debug)]
struct MonitorRow {
    monitor_id: String,
    name: String,
    owner_session_id: String,
    node_id: String,
    node_name: String,
    capability_id: String,
    capability_version: i64,
    unit: String,
    comparison: MonitorComparison,
    threshold: f64,
    clear_threshold: f64,
    duration_seconds: i64,
    stale_after_seconds: i64,
    offline_after_seconds: i64,
    repeat_seconds: i64,
    send_recovery: bool,
    status: String,
    condition_since: Option<DateTime<Utc>>,
    threshold_triggered_at: Option<DateTime<Utc>>,
    last_threshold_alert_at: Option<DateTime<Utc>>,
    availability_state: String,
    last_availability_alert_at: Option<DateTime<Utc>>,
    last_value: Option<f64>,
    last_received_at: Option<DateTime<Utc>>,
    mandate_id: Option<String>,
    mandate_goal_id: Option<String>,
    mandate_version: Option<i64>,
    expires_at: DateTime<Utc>,
    created_at: DateTime<Utc>,
    node_last_seen_at: Option<DateTime<Utc>>,
}

fn parse_time(value: &str) -> anyhow::Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc))
}

fn parse_optional_time(value: Option<String>) -> anyhow::Result<Option<DateTime<Utc>>> {
    value.as_deref().map(parse_time).transpose()
}

fn safe_name(value: &str, label: &str, max_chars: usize) -> anyhow::Result<String> {
    let trimmed = value.trim();
    anyhow::ensure!(!trimmed.is_empty(), "{label} is required");
    anyhow::ensure!(trimmed.chars().count() <= max_chars, "{label} is too long");
    anyhow::ensure!(
        !trimmed.chars().any(char::is_control),
        "{label} contains control characters"
    );
    Ok(trimmed.to_string())
}

impl NodeMonitoringService {
    pub fn new(pool: SqlitePool, config: NodeMonitoringConfig) -> Self {
        Self { pool, config }
    }

    fn ensure_enabled(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.config.enabled,
            "Node environmental monitoring is disabled"
        );
        Ok(())
    }

    pub async fn create(&self, request: CreateNodeMonitor) -> anyhow::Result<NodeMonitorView> {
        self.ensure_enabled()?;
        let name = safe_name(&request.name, "monitor name", 100)?;
        let owner_session_id = safe_name(&request.owner_session_id, "owner session", 200)?;
        anyhow::ensure!(
            request.capability_id == "sensor.environment.temperature"
                || request.capability_id == "sensor.environment.humidity",
            "v1 monitoring supports only temperature and humidity"
        );
        anyhow::ensure!(request.threshold.is_finite(), "threshold must be finite");
        anyhow::ensure!(
            request.clear_threshold.is_finite(),
            "clear threshold must be finite"
        );
        match request.comparison {
            MonitorComparison::Above => anyhow::ensure!(
                request.clear_threshold <= request.threshold,
                "an above monitor clear threshold must be at or below its alert threshold"
            ),
            MonitorComparison::Below => anyhow::ensure!(
                request.clear_threshold >= request.threshold,
                "a below monitor clear threshold must be at or above its alert threshold"
            ),
        }
        anyhow::ensure!(
            request.duration_seconds <= 86_400,
            "duration_seconds exceeds 86400"
        );
        anyhow::ensure!(
            request.repeat_seconds <= 604_800,
            "repeat_seconds exceeds 604800"
        );
        anyhow::ensure!(
            (1..=self.config.max_duration_days * 24 * 60).contains(&request.duration_minutes),
            "monitor duration must be between 1 minute and the configured maximum"
        );
        let stale_after = request
            .stale_after_seconds
            .unwrap_or(self.config.default_stale_after_seconds);
        // A Node cannot be meaningfully considered offline before its sensor
        // stream is stale. Treat an underspecified/shorter offline interval as
        // the stale interval instead of making conversational callers retry.
        let offline_after = request
            .offline_after_seconds
            .unwrap_or(self.config.default_offline_after_seconds)
            .max(stale_after);
        anyhow::ensure!(
            (30..=86_400).contains(&stale_after),
            "stale_after_seconds must be between 30 and 86400"
        );
        anyhow::ensure!(
            (30..=86_400).contains(&offline_after),
            "offline_after_seconds must be between 30 and 86400"
        );
        let active_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM node_monitors WHERE status IN ('active','paused','suspended')",
        )
        .fetch_one(&self.pool)
        .await?;
        anyhow::ensure!(
            active_count < self.config.max_monitors as i64,
            "maximum number of Node monitors has been reached"
        );

        let node_row = if let Some(node) = request.node.as_deref() {
            sqlx::query(
                "SELECT n.node_id, n.display_name, c.capability_version, l.unit, l.value, l.received_at
                 FROM nodes n
                 JOIN node_capabilities c ON c.node_id = n.node_id AND c.capability_id = ?
                 JOIN node_sensor_readings_latest l ON l.node_id = n.node_id AND l.capability_id = c.capability_id
                 JOIN node_authorizations a ON a.node_id = n.node_id AND a.action = 'report_sensor' AND a.revoked_at IS NULL
                 WHERE n.revoked_at IS NULL AND lower(n.display_name) = lower(?)",
            )
            .bind(&request.capability_id)
            .bind(node.trim())
            .fetch_optional(&self.pool)
            .await?
            .context("No active, authorized Node with that name and sensor capability was found")?
        } else {
            let rows = sqlx::query(
                "SELECT n.node_id, n.display_name, c.capability_version, l.unit, l.value, l.received_at
                 FROM nodes n
                 JOIN node_capabilities c ON c.node_id = n.node_id AND c.capability_id = ?
                 JOIN node_sensor_readings_latest l ON l.node_id = n.node_id AND l.capability_id = c.capability_id
                 JOIN node_authorizations a ON a.node_id = n.node_id AND a.action = 'report_sensor' AND a.revoked_at IS NULL
                 WHERE n.revoked_at IS NULL ORDER BY n.created_at",
            )
            .bind(&request.capability_id)
            .fetch_all(&self.pool)
            .await?;
            anyhow::ensure!(
                !rows.is_empty(),
                "No active, authorized Node has reported that sensor capability"
            );
            anyhow::ensure!(
                rows.len() == 1,
                "More than one eligible Node exists; specify its exact display name"
            );
            rows.into_iter().next().expect("checked nonempty")
        };
        let node_id: String = node_row.get("node_id");
        let capability_version: i64 = node_row.get("capability_version");
        let unit: String = node_row.get("unit");
        let initial_value: f64 = node_row.get("value");
        let initial_received_at: String = node_row.get("received_at");
        validate_unit_and_range(
            &request.capability_id,
            &unit,
            request.threshold,
            request.clear_threshold,
        )?;

        let (mandate_id, mandate_goal_id, mandate_version) = self
            .validate_mandate(request.mandate_id.as_deref(), &owner_session_id)
            .await?;
        let now = Utc::now();
        let expires_at = now + Duration::minutes(request.duration_minutes as i64);
        let monitor_id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO node_monitors
             (monitor_id, name, owner_session_id, node_id, capability_id, capability_version,
              unit, comparison, threshold, clear_threshold, duration_seconds,
              stale_after_seconds, offline_after_seconds, repeat_seconds, send_recovery,
              status, last_value, last_received_at, mandate_id, mandate_goal_id, mandate_version,
              expires_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&monitor_id)
        .bind(&name)
        .bind(&owner_session_id)
        .bind(&node_id)
        .bind(&request.capability_id)
        .bind(capability_version)
        .bind(&unit)
        .bind(request.comparison.as_str())
        .bind(request.threshold)
        .bind(request.clear_threshold)
        .bind(request.duration_seconds as i64)
        .bind(stale_after as i64)
        .bind(offline_after as i64)
        .bind(request.repeat_seconds as i64)
        .bind(request.send_recovery)
        .bind(initial_value)
        .bind(initial_received_at)
        .bind(mandate_id)
        .bind(mandate_goal_id)
        .bind(mandate_version)
        .bind(expires_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await?;
        self.get(&monitor_id, &owner_session_id).await
    }

    async fn validate_mandate(
        &self,
        mandate_id: Option<&str>,
        owner_session_id: &str,
    ) -> anyhow::Result<(Option<String>, Option<String>, Option<i64>)> {
        let Some(mandate_id) = mandate_id else {
            return Ok((None, None, None));
        };
        let row = sqlx::query(
            "SELECT id, goal_id, status, confirmed_at, version, created_by_session, expires_at
             FROM mandates WHERE id = ?",
        )
        .bind(mandate_id)
        .fetch_optional(&self.pool)
        .await?
        .context(
            "mandate not found; omit mandate_id for a direct owner monitor—no separate monitoring authorization is required",
        )?;
        anyhow::ensure!(
            row.get::<String, _>("created_by_session") == owner_session_id,
            "mandate belongs to a different owner session"
        );
        anyhow::ensure!(
            row.get::<String, _>("status") == "active"
                && row.get::<Option<String>, _>("confirmed_at").is_some(),
            "mandate must be owner-confirmed and active"
        );
        if let Some(expires_at) = row.get::<Option<String>, _>("expires_at") {
            anyhow::ensure!(parse_time(&expires_at)? > Utc::now(), "mandate has expired");
        }
        Ok((
            Some(row.get("id")),
            Some(row.get("goal_id")),
            Some(row.get("version")),
        ))
    }
}

fn validate_unit_and_range(
    capability_id: &str,
    unit: &str,
    threshold: f64,
    clear_threshold: f64,
) -> anyhow::Result<()> {
    let (expected_unit, low, high) = match capability_id {
        "sensor.environment.temperature" => ("celsius", -50.0, 100.0),
        "sensor.environment.humidity" => ("percent_rh", 0.0, 100.0),
        _ => anyhow::bail!("unsupported sensor capability"),
    };
    anyhow::ensure!(
        unit == expected_unit,
        "Node reported an unsupported unit for this capability"
    );
    anyhow::ensure!(
        (low..=high).contains(&threshold),
        "threshold is outside the supported sensor range"
    );
    anyhow::ensure!(
        (low..=high).contains(&clear_threshold),
        "clear threshold is outside the supported sensor range"
    );
    Ok(())
}

const MONITOR_SELECT: &str =
    "SELECT m.*, n.display_name AS node_name, n.last_seen_at AS node_last_seen_at
     FROM node_monitors m JOIN nodes n ON n.node_id = m.node_id";

fn monitor_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<MonitorRow> {
    Ok(MonitorRow {
        monitor_id: row.get("monitor_id"),
        name: row.get("name"),
        owner_session_id: row.get("owner_session_id"),
        node_id: row.get("node_id"),
        node_name: row.get("node_name"),
        capability_id: row.get("capability_id"),
        capability_version: row.get("capability_version"),
        unit: row.get("unit"),
        comparison: MonitorComparison::parse(row.get::<String, _>("comparison").as_str())?,
        threshold: row.get("threshold"),
        clear_threshold: row.get("clear_threshold"),
        duration_seconds: row.get("duration_seconds"),
        stale_after_seconds: row.get("stale_after_seconds"),
        offline_after_seconds: row.get("offline_after_seconds"),
        repeat_seconds: row.get("repeat_seconds"),
        send_recovery: row.get("send_recovery"),
        status: row.get("status"),
        condition_since: parse_optional_time(row.get("condition_since"))?,
        threshold_triggered_at: parse_optional_time(row.get("threshold_triggered_at"))?,
        last_threshold_alert_at: parse_optional_time(row.get("last_threshold_alert_at"))?,
        availability_state: row.get("availability_state"),
        last_availability_alert_at: parse_optional_time(row.get("last_availability_alert_at"))?,
        last_value: row.get("last_value"),
        last_received_at: parse_optional_time(row.get("last_received_at"))?,
        mandate_id: row.get("mandate_id"),
        mandate_goal_id: row.get("mandate_goal_id"),
        mandate_version: row.get("mandate_version"),
        expires_at: parse_time(&row.get::<String, _>("expires_at"))?,
        created_at: parse_time(&row.get::<String, _>("created_at"))?,
        node_last_seen_at: parse_optional_time(row.get("node_last_seen_at"))?,
    })
}

impl MonitorRow {
    fn view(&self) -> NodeMonitorView {
        NodeMonitorView {
            monitor_id: self.monitor_id.clone(),
            name: self.name.clone(),
            node: self.node_name.clone(),
            capability_id: self.capability_id.clone(),
            unit: self.unit.clone(),
            comparison: self.comparison.as_str().to_string(),
            threshold: self.threshold,
            clear_threshold: self.clear_threshold,
            duration_seconds: self.duration_seconds as u64,
            stale_after_seconds: self.stale_after_seconds as u64,
            offline_after_seconds: self.offline_after_seconds as u64,
            repeat_seconds: self.repeat_seconds as u64,
            send_recovery: self.send_recovery,
            status: self.status.clone(),
            threshold_state: if self.threshold_triggered_at.is_some() {
                "triggered".to_string()
            } else if self.condition_since.is_some() {
                "pending_duration".to_string()
            } else {
                "normal".to_string()
            },
            availability_state: self.availability_state.clone(),
            last_value: self.last_value,
            last_received_at: self.last_received_at.map(|value| value.to_rfc3339()),
            mandate_id: self.mandate_id.clone(),
            expires_at: self.expires_at.to_rfc3339(),
            created_at: self.created_at.to_rfc3339(),
        }
    }
}

impl NodeMonitoringService {
    async fn load_monitor(&self, monitor_id: &str) -> anyhow::Result<Option<MonitorRow>> {
        let query = format!("{MONITOR_SELECT} WHERE m.monitor_id = ?");
        let row = sqlx::query(&query)
            .bind(monitor_id)
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(monitor_from_row).transpose()
    }

    pub async fn get(
        &self,
        monitor_id: &str,
        owner_session_id: &str,
    ) -> anyhow::Result<NodeMonitorView> {
        self.ensure_enabled()?;
        let monitor = self
            .load_monitor(monitor_id)
            .await?
            .context("monitor not found")?;
        anyhow::ensure!(
            monitor.owner_session_id == owner_session_id,
            "monitor belongs to a different owner session"
        );
        Ok(monitor.view())
    }

    pub async fn list(&self, owner_session_id: &str) -> anyhow::Result<Vec<NodeMonitorView>> {
        self.ensure_enabled()?;
        let query = format!(
            "{MONITOR_SELECT} WHERE m.owner_session_id = ? ORDER BY m.created_at DESC LIMIT 100"
        );
        let rows = sqlx::query(&query)
            .bind(owner_session_id)
            .fetch_all(&self.pool)
            .await?;
        rows.iter()
            .map(monitor_from_row)
            .map(|result| result.map(|monitor| monitor.view()))
            .collect()
    }

    pub async fn change_status(
        &self,
        monitor_id: &str,
        owner_session_id: &str,
        action: &str,
    ) -> anyhow::Result<NodeMonitorView> {
        self.ensure_enabled()?;
        let monitor = self
            .load_monitor(monitor_id)
            .await?
            .context("monitor not found")?;
        anyhow::ensure!(
            monitor.owner_session_id == owner_session_id,
            "monitor belongs to a different owner session"
        );
        let new_status = match action {
            "pause" => {
                anyhow::ensure!(
                    monitor.status == "active",
                    "only an active monitor can be paused"
                );
                "paused"
            }
            "resume" => {
                anyhow::ensure!(
                    monitor.status == "paused" || monitor.status == "suspended",
                    "only a paused or suspended monitor can be resumed"
                );
                anyhow::ensure!(monitor.expires_at > Utc::now(), "monitor has expired");
                self.validate_mandate(monitor.mandate_id.as_deref(), owner_session_id)
                    .await?;
                "active"
            }
            "cancel" => {
                anyhow::ensure!(
                    !matches!(monitor.status.as_str(), "cancelled" | "expired"),
                    "monitor has already ended"
                );
                "cancelled"
            }
            _ => anyhow::bail!("action must be pause, resume, or cancel"),
        };
        sqlx::query(
            "UPDATE node_monitors SET status = ?, condition_since = NULL,
             threshold_triggered_at = CASE WHEN ? = 'active' THEN NULL ELSE threshold_triggered_at END,
             updated_at = ? WHERE monitor_id = ? AND owner_session_id = ?",
        )
        .bind(new_status)
        .bind(new_status)
        .bind(Utc::now().to_rfc3339())
        .bind(monitor_id)
        .bind(owner_session_id)
        .execute(&self.pool)
        .await?;
        self.get(monitor_id, owner_session_id).await
    }

    pub async fn history(
        &self,
        monitor_id: &str,
        owner_session_id: &str,
        since_hours: u64,
        limit: u32,
    ) -> anyhow::Result<Vec<NodeMonitorEventView>> {
        self.ensure_enabled()?;
        let _ = self.get(monitor_id, owner_session_id).await?;
        anyhow::ensure!(
            (1..=720).contains(&since_hours),
            "since_hours must be between 1 and 720"
        );
        anyhow::ensure!(
            (1..=500).contains(&limit),
            "limit must be between 1 and 500"
        );
        let since = (Utc::now() - Duration::hours(since_hours as i64)).to_rfc3339();
        let rows = sqlx::query(
            "SELECT e.event_kind, e.evidence_json, e.created_at,
                    q.delivered_at IS NOT NULL AS delivered
             FROM node_monitor_events e
             LEFT JOIN notification_queue q ON q.id = e.notification_id
             WHERE e.monitor_id = ? AND e.created_at >= ?
             ORDER BY e.created_at DESC LIMIT ?",
        )
        .bind(monitor_id)
        .bind(since)
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                Ok(NodeMonitorEventView {
                    event_kind: row.get("event_kind"),
                    evidence: serde_json::from_str(&row.get::<String, _>("evidence_json"))?,
                    created_at: row.get("created_at"),
                    delivered: row.get("delivered"),
                })
            })
            .collect()
    }

    pub async fn sensor_history(
        &self,
        monitor_id: &str,
        owner_session_id: &str,
        since_hours: u64,
        limit: u32,
    ) -> anyhow::Result<Vec<NodeSensorHistoryPoint>> {
        self.ensure_enabled()?;
        let monitor = self
            .load_monitor(monitor_id)
            .await?
            .context("monitor not found")?;
        anyhow::ensure!(
            monitor.owner_session_id == owner_session_id,
            "monitor belongs to a different owner session"
        );
        anyhow::ensure!(
            (1..=self.config.history_hours).contains(&since_hours),
            "since_hours must be within the configured history window"
        );
        anyhow::ensure!(
            (1..=500).contains(&limit),
            "limit must be between 1 and 500"
        );
        let since = (Utc::now() - Duration::hours(since_hours as i64)).to_rfc3339();
        let rows = sqlx::query(
            "SELECT value, unit, received_at FROM node_sensor_readings_history
             WHERE node_id = ? AND capability_id = ? AND received_at >= ?
             ORDER BY received_at DESC, reading_id DESC LIMIT ?",
        )
        .bind(&monitor.node_id)
        .bind(&monitor.capability_id)
        .bind(since)
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|row| NodeSensorHistoryPoint {
                value: row.get("value"),
                unit: row.get("unit"),
                server_received_at: row.get("received_at"),
            })
            .collect())
    }

    pub async fn evaluate_node_readings(&self, node_id: &str) -> anyhow::Result<u64> {
        if !self.config.enabled {
            return Ok(0);
        }
        let query = format!(
            "{MONITOR_SELECT} WHERE m.node_id = ? AND m.status = 'active' AND m.expires_at > ?"
        );
        let rows = sqlx::query(&query)
            .bind(node_id)
            .bind(Utc::now().to_rfc3339())
            .fetch_all(&self.pool)
            .await?;
        let mut events = 0;
        for row in &rows {
            let monitor = monitor_from_row(row)?;
            let latest = sqlx::query(
                "SELECT value, unit, capability_version, received_at
                 FROM node_sensor_readings_latest
                 WHERE node_id = ? AND capability_id = ?",
            )
            .bind(&monitor.node_id)
            .bind(&monitor.capability_id)
            .fetch_optional(&self.pool)
            .await?;
            let Some(latest) = latest else { continue };
            let value: f64 = latest.get("value");
            let unit: String = latest.get("unit");
            let version: i64 = latest.get("capability_version");
            let received_at = parse_time(&latest.get::<String, _>("received_at"))?;
            if unit != monitor.unit || version != monitor.capability_version {
                self.suspend_monitor(
                    &monitor,
                    "sensor_contract_changed",
                    json!({
                        "node": monitor.node_name,
                        "capability_id": monitor.capability_id,
                        "expected_unit": monitor.unit,
                        "reported_unit": unit,
                        "expected_version": monitor.capability_version,
                        "reported_version": version,
                    }),
                )
                .await?;
                events += 1;
                continue;
            }
            events += self
                .evaluate_threshold(&monitor, value, received_at)
                .await?;
        }
        Ok(events)
    }

    async fn evaluate_threshold(
        &self,
        monitor: &MonitorRow,
        value: f64,
        received_at: DateTime<Utc>,
    ) -> anyhow::Result<u64> {
        if monitor
            .last_received_at
            .is_some_and(|last| received_at <= last)
        {
            return Ok(0);
        }
        let now = Utc::now();
        let matches = monitor
            .comparison
            .condition_matches(value, monitor.threshold);
        let recovered = monitor
            .comparison
            .recovery_matches(value, monitor.clear_threshold);
        let mut transaction = self.pool.begin().await?;

        if monitor.threshold_triggered_at.is_some() {
            if recovered {
                sqlx::query(
                    "UPDATE node_monitors SET condition_since = NULL, threshold_triggered_at = NULL,
                     last_threshold_alert_at = NULL, last_value = ?, last_received_at = ?, updated_at = ?
                     WHERE monitor_id = ? AND status = 'active'",
                )
                .bind(value)
                .bind(received_at.to_rfc3339())
                .bind(now.to_rfc3339())
                .bind(&monitor.monitor_id)
                .execute(&mut *transaction)
                .await?;
                if monitor.send_recovery {
                    let evidence = threshold_evidence(monitor, value, received_at, "recovered");
                    let message = format!(
                        "{} on {} recovered: {} is now {:.1} {} (clear point {:.1} {}).",
                        monitor.name,
                        monitor.node_name,
                        capability_label(&monitor.capability_id),
                        value,
                        unit_label(&monitor.unit),
                        monitor.clear_threshold,
                        unit_label(&monitor.unit),
                    );
                    insert_event_and_notification(
                        &mut transaction,
                        monitor,
                        "threshold_recovery",
                        evidence,
                        "node_monitor_recovery",
                        &message,
                    )
                    .await?;
                }
                transaction.commit().await?;
                return Ok(u64::from(monitor.send_recovery));
            }

            let should_repeat = matches
                && monitor.repeat_seconds > 0
                && monitor
                    .last_threshold_alert_at
                    .is_some_and(|last| now - last >= Duration::seconds(monitor.repeat_seconds));
            sqlx::query(
                "UPDATE node_monitors SET last_value = ?, last_received_at = ?, updated_at = ?
                 WHERE monitor_id = ? AND status = 'active'",
            )
            .bind(value)
            .bind(received_at.to_rfc3339())
            .bind(now.to_rfc3339())
            .bind(&monitor.monitor_id)
            .execute(&mut *transaction)
            .await?;
            if should_repeat {
                sqlx::query(
                    "UPDATE node_monitors SET last_threshold_alert_at = ? WHERE monitor_id = ?",
                )
                .bind(now.to_rfc3339())
                .bind(&monitor.monitor_id)
                .execute(&mut *transaction)
                .await?;
                let evidence = threshold_evidence(monitor, value, received_at, "still_triggered");
                let message = threshold_alert_message(monitor, value, true);
                insert_event_and_notification(
                    &mut transaction,
                    monitor,
                    "threshold_repeat",
                    evidence,
                    "node_monitor_alert",
                    &message,
                )
                .await?;
                transaction.commit().await?;
                return Ok(1);
            }
            transaction.commit().await?;
            return Ok(0);
        }

        if !matches {
            sqlx::query(
                "UPDATE node_monitors SET condition_since = NULL, last_value = ?,
                 last_received_at = ?, updated_at = ? WHERE monitor_id = ? AND status = 'active'",
            )
            .bind(value)
            .bind(received_at.to_rfc3339())
            .bind(now.to_rfc3339())
            .bind(&monitor.monitor_id)
            .execute(&mut *transaction)
            .await?;
            transaction.commit().await?;
            return Ok(0);
        }

        let continuity_broken = monitor.last_received_at.is_some_and(|last| {
            received_at - last > Duration::seconds(monitor.stale_after_seconds)
        });
        let condition_since = if continuity_broken {
            received_at
        } else {
            monitor.condition_since.unwrap_or(received_at)
        };
        let duration_met =
            received_at - condition_since >= Duration::seconds(monitor.duration_seconds.max(0));
        sqlx::query(
            "UPDATE node_monitors SET condition_since = ?, last_value = ?,
             last_received_at = ?, updated_at = ? WHERE monitor_id = ? AND status = 'active'",
        )
        .bind(condition_since.to_rfc3339())
        .bind(value)
        .bind(received_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(&monitor.monitor_id)
        .execute(&mut *transaction)
        .await?;
        if !duration_met {
            transaction.commit().await?;
            return Ok(0);
        }

        sqlx::query(
            "UPDATE node_monitors SET threshold_triggered_at = ?, last_threshold_alert_at = ?
             WHERE monitor_id = ? AND status = 'active'",
        )
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(&monitor.monitor_id)
        .execute(&mut *transaction)
        .await?;
        let evidence = threshold_evidence(monitor, value, received_at, "triggered");
        let message = threshold_alert_message(monitor, value, false);
        insert_event_and_notification(
            &mut transaction,
            monitor,
            "threshold_alert",
            evidence,
            "node_monitor_alert",
            &message,
        )
        .await?;
        transaction.commit().await?;
        Ok(1)
    }
}

fn capability_label(capability_id: &str) -> &'static str {
    match capability_id {
        "sensor.environment.temperature" => "temperature",
        "sensor.environment.humidity" => "humidity",
        _ => "sensor value",
    }
}

fn unit_label(unit: &str) -> &'static str {
    match unit {
        "celsius" => "°C",
        "percent_rh" => "% RH",
        _ => "units",
    }
}

fn threshold_evidence(
    monitor: &MonitorRow,
    value: f64,
    received_at: DateTime<Utc>,
    state: &str,
) -> Value {
    json!({
        "state": state,
        "node": monitor.node_name,
        "capability_id": monitor.capability_id,
        "value": value,
        "unit": monitor.unit,
        "comparison": monitor.comparison.as_str(),
        "threshold": monitor.threshold,
        "clear_threshold": monitor.clear_threshold,
        "duration_seconds": monitor.duration_seconds,
        "server_received_at": received_at.to_rfc3339(),
    })
}

fn threshold_alert_message(monitor: &MonitorRow, value: f64, repeated: bool) -> String {
    format!(
        "{} on {}{}: {} is {:.1} {}, {} the {:.1} {} threshold{}.",
        monitor.name,
        monitor.node_name,
        if repeated { " is still active" } else { "" },
        capability_label(&monitor.capability_id),
        value,
        unit_label(&monitor.unit),
        monitor.comparison.as_str(),
        monitor.threshold,
        unit_label(&monitor.unit),
        if monitor.duration_seconds > 0 {
            " for the configured duration"
        } else {
            ""
        },
    )
}

async fn insert_event_and_notification(
    transaction: &mut Transaction<'_, Sqlite>,
    monitor: &MonitorRow,
    event_kind: &str,
    evidence: Value,
    notification_type: &str,
    message: &str,
) -> anyhow::Result<()> {
    let entry = NotificationEntry::new(
        monitor
            .mandate_goal_id
            .as_deref()
            .unwrap_or(&monitor.monitor_id),
        &monitor.owner_session_id,
        notification_type,
        message,
    );
    let event_id = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO notification_queue
         (id, goal_id, session_id, notification_type, priority, message,
          created_at, delivered_at, attempts, expires_at, task_id, action_token)
         VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, NULL, NULL)",
    )
    .bind(&entry.id)
    .bind(&entry.goal_id)
    .bind(&entry.session_id)
    .bind(&entry.notification_type)
    .bind(&entry.priority)
    .bind(&entry.message)
    .bind(&entry.created_at)
    .bind(&entry.expires_at)
    .execute(&mut **transaction)
    .await?;
    sqlx::query(
        "INSERT INTO node_monitor_events
         (event_id, monitor_id, event_kind, evidence_json, notification_id, created_at)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(event_id)
    .bind(&monitor.monitor_id)
    .bind(event_kind)
    .bind(serde_json::to_string(&evidence)?)
    .bind(&entry.id)
    .bind(&entry.created_at)
    .execute(&mut **transaction)
    .await?;
    Ok(())
}

impl NodeMonitoringService {
    pub async fn run_maintenance(&self) -> anyhow::Result<NodeMonitorMaintenanceStats> {
        if !self.config.enabled {
            return Ok(NodeMonitorMaintenanceStats::default());
        }
        let mut stats = NodeMonitorMaintenanceStats::default();
        let now = Utc::now();

        let expired_query = format!(
            "{MONITOR_SELECT} WHERE m.status IN ('active','paused','suspended') AND m.expires_at <= ?"
        );
        let expired = sqlx::query(&expired_query)
            .bind(now.to_rfc3339())
            .fetch_all(&self.pool)
            .await?;
        for row in &expired {
            let monitor = monitor_from_row(row)?;
            let mut transaction = self.pool.begin().await?;
            let changed = sqlx::query(
                "UPDATE node_monitors SET status = 'expired', condition_since = NULL, updated_at = ?
                 WHERE monitor_id = ? AND status IN ('active','paused','suspended')",
            )
            .bind(now.to_rfc3339())
            .bind(&monitor.monitor_id)
            .execute(&mut *transaction)
            .await?
            .rows_affected();
            if changed == 1 {
                let message = format!(
                    "{} on {} ended at its configured expiration time.",
                    monitor.name, monitor.node_name
                );
                insert_event_and_notification(
                    &mut transaction,
                    &monitor,
                    "monitor_expired",
                    json!({"node": monitor.node_name, "expired_at": monitor.expires_at.to_rfc3339()}),
                    "node_monitor_ended",
                    &message,
                )
                .await?;
                stats.expired += 1;
            }
            transaction.commit().await?;
        }

        let active_query = format!("{MONITOR_SELECT} WHERE m.status = 'active'");
        let rows = sqlx::query(&active_query).fetch_all(&self.pool).await?;
        for row in &rows {
            let monitor = monitor_from_row(row)?;
            stats.evaluated += 1;
            if !self.mandate_still_valid(&monitor).await? {
                self.suspend_monitor(
                    &monitor,
                    "mandate_authority_changed",
                    json!({"node": monitor.node_name, "reason": "linked mandate is no longer active at the recorded version"}),
                )
                .await?;
                stats.suspended += 1;
                continue;
            }
            let state = availability_state(&monitor, now);
            if state == monitor.availability_state {
                let should_repeat = state != "normal"
                    && monitor.repeat_seconds > 0
                    && monitor.last_availability_alert_at.is_some_and(|last| {
                        now - last >= Duration::seconds(monitor.repeat_seconds)
                    });
                if should_repeat {
                    self.emit_availability(&monitor, state, true).await?;
                    stats.alerts += 1;
                }
                continue;
            }
            if state == "normal" {
                self.update_availability(&monitor, state, monitor.send_recovery)
                    .await?;
                if monitor.send_recovery {
                    stats.recoveries += 1;
                }
            } else {
                self.update_availability(&monitor, state, true).await?;
                stats.alerts += 1;
            }
        }

        let history_cutoff = (now - Duration::hours(self.config.history_hours as i64)).to_rfc3339();
        stats.history_rows_pruned =
            sqlx::query("DELETE FROM node_sensor_readings_history WHERE received_at < ?")
                .bind(history_cutoff)
                .execute(&self.pool)
                .await?
                .rows_affected();
        let event_cutoff =
            (now - Duration::days(self.config.event_retention_days as i64)).to_rfc3339();
        stats.event_rows_pruned =
            sqlx::query("DELETE FROM node_monitor_events WHERE created_at < ?")
                .bind(event_cutoff)
                .execute(&self.pool)
                .await?
                .rows_affected();
        stats.event_rows_pruned += sqlx::query(
            "DELETE FROM node_monitor_events
             WHERE event_id IN (
               SELECT event_id FROM (
                 SELECT event_id,
                        ROW_NUMBER() OVER (
                          PARTITION BY monitor_id ORDER BY created_at DESC, event_id DESC
                        ) AS row_number
                 FROM node_monitor_events
               ) WHERE row_number > ?
             )",
        )
        .bind(self.config.max_events_per_monitor as i64)
        .execute(&self.pool)
        .await?
        .rows_affected();
        Ok(stats)
    }

    async fn mandate_still_valid(&self, monitor: &MonitorRow) -> anyhow::Result<bool> {
        let Some(mandate_id) = monitor.mandate_id.as_deref() else {
            return Ok(true);
        };
        let row = sqlx::query(
            "SELECT status, confirmed_at, version, created_by_session, expires_at
             FROM mandates WHERE id = ?",
        )
        .bind(mandate_id)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else { return Ok(false) };
        let expires_valid = row
            .get::<Option<String>, _>("expires_at")
            .as_deref()
            .map(parse_time)
            .transpose()?
            .is_none_or(|expires| expires > Utc::now());
        Ok(row.get::<String, _>("status") == "active"
            && row.get::<Option<String>, _>("confirmed_at").is_some()
            && row.get::<String, _>("created_by_session") == monitor.owner_session_id
            && Some(row.get::<i64, _>("version")) == monitor.mandate_version
            && expires_valid)
    }

    async fn suspend_monitor(
        &self,
        monitor: &MonitorRow,
        reason: &str,
        evidence: Value,
    ) -> anyhow::Result<()> {
        let mut transaction = self.pool.begin().await?;
        let changed = sqlx::query(
            "UPDATE node_monitors SET status = 'suspended', condition_since = NULL, updated_at = ?
             WHERE monitor_id = ? AND status = 'active'",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(&monitor.monitor_id)
        .execute(&mut *transaction)
        .await?
        .rows_affected();
        if changed == 1 {
            let message = format!(
                "{} on {} was suspended because {}. Review it before resuming.",
                monitor.name,
                monitor.node_name,
                reason.replace('_', " ")
            );
            insert_event_and_notification(
                &mut transaction,
                monitor,
                "monitor_suspended",
                evidence,
                "node_monitor_suspended",
                &message,
            )
            .await?;
        }
        transaction.commit().await?;
        Ok(())
    }

    async fn update_availability(
        &self,
        monitor: &MonitorRow,
        state: &str,
        notify: bool,
    ) -> anyhow::Result<()> {
        let mut transaction = self.pool.begin().await?;
        sqlx::query(
            "UPDATE node_monitors SET availability_state = ?, last_availability_alert_at = ?, updated_at = ?
             WHERE monitor_id = ? AND status = 'active'",
        )
        .bind(state)
        .bind(if notify { Some(Utc::now().to_rfc3339()) } else { None })
        .bind(Utc::now().to_rfc3339())
        .bind(&monitor.monitor_id)
        .execute(&mut *transaction)
        .await?;
        if notify {
            insert_availability_event(&mut transaction, monitor, state, false).await?;
        }
        transaction.commit().await?;
        Ok(())
    }

    async fn emit_availability(
        &self,
        monitor: &MonitorRow,
        state: &str,
        repeated: bool,
    ) -> anyhow::Result<()> {
        let mut transaction = self.pool.begin().await?;
        sqlx::query(
            "UPDATE node_monitors SET last_availability_alert_at = ?, updated_at = ?
             WHERE monitor_id = ? AND status = 'active'",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(Utc::now().to_rfc3339())
        .bind(&monitor.monitor_id)
        .execute(&mut *transaction)
        .await?;
        insert_availability_event(&mut transaction, monitor, state, repeated).await?;
        transaction.commit().await?;
        Ok(())
    }
}

fn availability_state(monitor: &MonitorRow, now: DateTime<Utc>) -> &'static str {
    if monitor
        .node_last_seen_at
        .is_none_or(|seen| now - seen >= Duration::seconds(monitor.offline_after_seconds))
    {
        "offline"
    } else if monitor
        .last_received_at
        .is_none_or(|received| now - received >= Duration::seconds(monitor.stale_after_seconds))
    {
        "stale"
    } else {
        "normal"
    }
}

async fn insert_availability_event(
    transaction: &mut Transaction<'_, Sqlite>,
    monitor: &MonitorRow,
    state: &str,
    repeated: bool,
) -> anyhow::Result<()> {
    let recovered = state == "normal";
    let event_kind = if recovered {
        "availability_recovery"
    } else if repeated {
        "availability_repeat"
    } else if state == "offline" {
        "node_offline"
    } else {
        "sensor_stale"
    };
    let message = if recovered {
        format!(
            "{} on {} recovered: the Node is connected and {} readings are current again.",
            monitor.name,
            monitor.node_name,
            capability_label(&monitor.capability_id)
        )
    } else if state == "offline" {
        format!(
            "{} on {}{}: the Node has not checked in within {} seconds.",
            monitor.name,
            monitor.node_name,
            if repeated {
                " is still offline"
            } else {
                " is offline"
            },
            monitor.offline_after_seconds
        )
    } else {
        format!(
            "{} on {}{}: its {} reading has not updated within {} seconds.",
            monitor.name,
            monitor.node_name,
            if repeated {
                " is still stale"
            } else {
                " has stale data"
            },
            capability_label(&monitor.capability_id),
            monitor.stale_after_seconds
        )
    };
    insert_event_and_notification(
        transaction,
        monitor,
        event_kind,
        json!({
            "state": state,
            "node": monitor.node_name,
            "capability_id": monitor.capability_id,
            "last_sensor_received_at": monitor.last_received_at.map(|value| value.to_rfc3339()),
            "node_last_seen_at": monitor.node_last_seen_at.map(|value| value.to_rfc3339()),
            "stale_after_seconds": monitor.stale_after_seconds,
            "offline_after_seconds": monitor.offline_after_seconds,
        }),
        if recovered {
            "node_monitor_recovery"
        } else {
            "node_monitor_alert"
        },
        &message,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn test_service() -> NodeMonitoringService {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE notification_queue (
                id TEXT PRIMARY KEY, goal_id TEXT NOT NULL, session_id TEXT NOT NULL,
                notification_type TEXT NOT NULL, priority TEXT NOT NULL, message TEXT NOT NULL,
                created_at TEXT NOT NULL, delivered_at TEXT, attempts INTEGER NOT NULL,
                expires_at TEXT, task_id TEXT, action_token TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE mandates (
                id TEXT PRIMARY KEY, goal_id TEXT NOT NULL, status TEXT NOT NULL,
                confirmed_at TEXT, version INTEGER NOT NULL, created_by_session TEXT NOT NULL,
                expires_at TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        super::super::store::migrate(&pool).await.unwrap();
        let now = Utc::now() - Duration::seconds(2);
        sqlx::query(
            "INSERT INTO nodes
             (node_id, owner_id, kind, display_name, policy_profile, created_at, last_seen_at)
             VALUES ('node-test', 'owner-test', 'companion', 'Kitchen K10', 'child_companion', ?, ?)",
        )
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO node_capabilities
             (node_id, capability_id, capability_version, limits_json, observed_at)
             VALUES ('node-test', 'sensor.environment.temperature', 1, '{}', ?),
                    ('node-test', 'sensor.environment.humidity', 1, '{}', ?)",
        )
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO node_authorizations
             (node_id, action, constraints_json, revision, granted_at)
             VALUES ('node-test', 'report_sensor', '{}', 1, ?)",
        )
        .bind(now.to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO node_sensor_readings_latest
             (node_id, capability_id, capability_version, value, unit,
              sample_uptime_ms, request_id, observed_session_id, received_at)
             VALUES ('node-test', 'sensor.environment.temperature', 1, 25.0, 'celsius',
                     1000, 'request-test-temperature', 'session-test', ?),
                    ('node-test', 'sensor.environment.humidity', 1, 45.0, 'percent_rh',
                     1000, 'request-test-humidity', 'session-test', ?)",
        )
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();
        let config = NodeMonitoringConfig {
            enabled: true,
            ..NodeMonitoringConfig::default()
        };
        NodeMonitoringService::new(pool, config)
    }

    fn temperature_request() -> CreateNodeMonitor {
        CreateNodeMonitor {
            name: "Warm kitchen".to_string(),
            owner_session_id: "owner-session".to_string(),
            node: Some("Kitchen K10".to_string()),
            capability_id: "sensor.environment.temperature".to_string(),
            comparison: MonitorComparison::Above,
            threshold: 24.0,
            clear_threshold: 23.0,
            duration_seconds: 0,
            stale_after_seconds: Some(30),
            offline_after_seconds: Some(60),
            repeat_seconds: 0,
            send_recovery: true,
            duration_minutes: 60,
            mandate_id: None,
        }
    }

    #[tokio::test]
    async fn threshold_alert_and_hysteresis_recovery_are_deterministic() {
        let service = test_service().await;
        let monitor = service.create(temperature_request()).await.unwrap();

        sqlx::query(
            "UPDATE node_sensor_readings_latest
             SET request_id = 'request-test-trigger', received_at = ?
             WHERE node_id = 'node-test' AND capability_id = 'sensor.environment.temperature'",
        )
        .bind(Utc::now().to_rfc3339())
        .execute(&service.pool)
        .await
        .unwrap();
        assert_eq!(
            service.evaluate_node_readings("node-test").await.unwrap(),
            1
        );
        let triggered = service
            .get(&monitor.monitor_id, "owner-session")
            .await
            .unwrap();
        assert_eq!(triggered.threshold_state, "triggered");

        let received_at = Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE node_sensor_readings_latest
             SET value = 22.5, request_id = 'request-test-recovery', received_at = ?
             WHERE node_id = 'node-test' AND capability_id = 'sensor.environment.temperature'",
        )
        .bind(received_at)
        .execute(&service.pool)
        .await
        .unwrap();
        assert_eq!(
            service.evaluate_node_readings("node-test").await.unwrap(),
            1
        );

        let recovered = service
            .get(&monitor.monitor_id, "owner-session")
            .await
            .unwrap();
        assert_eq!(recovered.threshold_state, "normal");
        let events = service
            .history(&monitor.monitor_id, "owner-session", 1, 10)
            .await
            .unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_kind, "threshold_recovery");
        assert_eq!(events[1].event_kind, "threshold_alert");
        let priorities: Vec<String> =
            sqlx::query_scalar("SELECT priority FROM notification_queue ORDER BY created_at, id")
                .fetch_all(&service.pool)
                .await
                .unwrap();
        assert!(priorities.contains(&"critical".to_string()));
        assert!(priorities.contains(&"status_update".to_string()));
    }

    #[tokio::test]
    async fn stale_offline_and_recovery_are_separate_states() {
        let service = test_service().await;
        let monitor = service.create(temperature_request()).await.unwrap();
        service.evaluate_node_readings("node-test").await.unwrap();

        let old_sensor = (Utc::now() - Duration::seconds(40)).to_rfc3339();
        sqlx::query("UPDATE node_monitors SET last_received_at = ? WHERE monitor_id = ?")
            .bind(old_sensor)
            .bind(&monitor.monitor_id)
            .execute(&service.pool)
            .await
            .unwrap();
        let stale = service.run_maintenance().await.unwrap();
        assert_eq!(stale.alerts, 1);
        assert_eq!(
            service
                .get(&monitor.monitor_id, "owner-session")
                .await
                .unwrap()
                .availability_state,
            "stale"
        );

        let old_node = (Utc::now() - Duration::seconds(70)).to_rfc3339();
        sqlx::query("UPDATE nodes SET last_seen_at = ? WHERE node_id = 'node-test'")
            .bind(old_node)
            .execute(&service.pool)
            .await
            .unwrap();
        let offline = service.run_maintenance().await.unwrap();
        assert_eq!(offline.alerts, 1);
        assert_eq!(
            service
                .get(&monitor.monitor_id, "owner-session")
                .await
                .unwrap()
                .availability_state,
            "offline"
        );

        let current = Utc::now().to_rfc3339();
        sqlx::query("UPDATE nodes SET last_seen_at = ? WHERE node_id = 'node-test'")
            .bind(&current)
            .execute(&service.pool)
            .await
            .unwrap();
        sqlx::query("UPDATE node_monitors SET last_received_at = ? WHERE monitor_id = ?")
            .bind(&current)
            .bind(&monitor.monitor_id)
            .execute(&service.pool)
            .await
            .unwrap();
        let recovery = service.run_maintenance().await.unwrap();
        assert_eq!(recovery.recoveries, 1);
        assert_eq!(
            service
                .get(&monitor.monitor_id, "owner-session")
                .await
                .unwrap()
                .availability_state,
            "normal"
        );
    }

    #[tokio::test]
    async fn duration_requires_fresh_continuous_reports() {
        let service = test_service().await;
        let mut request = temperature_request();
        request.duration_seconds = 60;
        let monitor = service.create(request).await.unwrap();
        let base = Utc::now();

        for (index, offset) in [0_i64, 20, 40, 61].into_iter().enumerate() {
            sqlx::query(
                "UPDATE node_sensor_readings_latest SET request_id = ?, received_at = ?
                 WHERE node_id = 'node-test' AND capability_id = 'sensor.environment.temperature'",
            )
            .bind(format!("duration-request-{index}"))
            .bind((base + Duration::seconds(offset)).to_rfc3339())
            .execute(&service.pool)
            .await
            .unwrap();
            let events = service.evaluate_node_readings("node-test").await.unwrap();
            assert_eq!(events, u64::from(offset == 61));
        }
        assert_eq!(
            service
                .get(&monitor.monitor_id, "owner-session")
                .await
                .unwrap()
                .threshold_state,
            "triggered"
        );

        sqlx::query(
            "UPDATE node_monitors SET condition_since = ?, threshold_triggered_at = NULL,
             last_threshold_alert_at = NULL, last_received_at = ? WHERE monitor_id = ?",
        )
        .bind(base.to_rfc3339())
        .bind(base.to_rfc3339())
        .bind(&monitor.monitor_id)
        .execute(&service.pool)
        .await
        .unwrap();
        sqlx::query(
            "UPDATE node_sensor_readings_latest SET request_id = 'duration-after-gap', received_at = ?
             WHERE node_id = 'node-test' AND capability_id = 'sensor.environment.temperature'",
        )
        .bind((base + Duration::seconds(100)).to_rfc3339())
        .execute(&service.pool)
        .await
        .unwrap();
        assert_eq!(
            service.evaluate_node_readings("node-test").await.unwrap(),
            0
        );
        assert_eq!(
            service
                .get(&monitor.monitor_id, "owner-session")
                .await
                .unwrap()
                .threshold_state,
            "pending_duration"
        );
    }

    #[tokio::test]
    async fn linked_mandate_version_change_suspends_monitor() {
        let service = test_service().await;
        sqlx::query(
            "INSERT INTO mandates
             (id, goal_id, status, confirmed_at, version, created_by_session, expires_at)
             VALUES ('mandate-test', 'goal-test', 'active', ?, 1, 'owner-session', NULL)",
        )
        .bind(Utc::now().to_rfc3339())
        .execute(&service.pool)
        .await
        .unwrap();
        let mut request = temperature_request();
        request.mandate_id = Some("mandate-test".to_string());
        let monitor = service.create(request).await.unwrap();

        sqlx::query("UPDATE mandates SET version = 2 WHERE id = 'mandate-test'")
            .execute(&service.pool)
            .await
            .unwrap();
        let stats = service.run_maintenance().await.unwrap();
        assert_eq!(stats.suspended, 1);
        assert_eq!(
            service
                .get(&monitor.monitor_id, "owner-session")
                .await
                .unwrap()
                .status,
            "suspended"
        );
    }

    #[tokio::test]
    async fn shorter_offline_interval_is_normalized_to_stale_interval() {
        let service = test_service().await;
        let mut request = temperature_request();
        request.stale_after_seconds = Some(120);
        request.offline_after_seconds = Some(60);

        let monitor = service.create(request).await.unwrap();

        assert_eq!(monitor.stale_after_seconds, 120);
        assert_eq!(monitor.offline_after_seconds, 120);
    }

    #[tokio::test]
    async fn missing_mandate_error_explains_that_direct_owner_requests_need_none() {
        let service = test_service().await;
        let mut request = temperature_request();
        request.mandate_id = Some("not-a-mandate".to_string());

        let error = service.create(request).await.unwrap_err();

        assert!(error.to_string().contains("omit mandate_id"));
        assert!(error
            .to_string()
            .contains("no separate monitoring authorization is required"));
    }

    #[tokio::test]
    async fn expiration_and_owner_isolation_fail_closed() {
        let service = test_service().await;
        let monitor = service.create(temperature_request()).await.unwrap();
        let wrong_owner = service
            .get(&monitor.monitor_id, "other-session")
            .await
            .unwrap_err();
        assert!(wrong_owner.to_string().contains("different owner session"));

        sqlx::query("UPDATE node_monitors SET expires_at = ? WHERE monitor_id = ?")
            .bind((Utc::now() - Duration::seconds(1)).to_rfc3339())
            .bind(&monitor.monitor_id)
            .execute(&service.pool)
            .await
            .unwrap();
        let stats = service.run_maintenance().await.unwrap();
        assert_eq!(stats.expired, 1);
        assert_eq!(
            service
                .get(&monitor.monitor_id, "owner-session")
                .await
                .unwrap()
                .status,
            "expired"
        );
    }

    #[tokio::test]
    async fn disabled_service_rejects_creation_before_database_access() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let service = NodeMonitoringService::new(pool, NodeMonitoringConfig::default());
        let error = service.create(temperature_request()).await.unwrap_err();
        assert!(error.to_string().contains("monitoring is disabled"));
    }
}
