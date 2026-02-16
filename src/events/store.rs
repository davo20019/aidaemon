//! Event store implementation using SQLite.
//!
//! The EventStore provides CRUD operations for events, with support for:
//! - Efficient querying by session, time window, and event type
//! - Conversation history retrieval from canonical events
//! - Consolidation tracking and pruning

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use sqlx::{Row, SqlitePool};
use tracing::{info, warn};

use super::{
    DecisionPointData, DecisionType, Event, EventType, PolicyDecisionData, TaskEndData, TaskStatus,
    ToolResultData,
};
use crate::traits::Message;

/// The event store backed by SQLite.
pub struct EventStore {
    pool: SqlitePool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskWindowStats {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub stalled: u64,
    pub error_events: u64,
    pub completion_rate: f64,
    pub error_rate: f64,
    pub stall_rate: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolStats {
    pub total_calls: u64,
    pub successful: u64,
    pub failed: u64,
    pub avg_duration_ms: u64,
    /// (error pattern, count), top 3.
    pub common_errors: Vec<(String, u64)>,
}

impl Default for TaskWindowStats {
    fn default() -> Self {
        Self {
            total: 0,
            completed: 0,
            failed: 0,
            cancelled: 0,
            stalled: 0,
            error_events: 0,
            completion_rate: 1.0,
            error_rate: 0.0,
            stall_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyGraduationReport {
    pub window_days: u32,
    pub observed_days: f64,
    pub total_decisions: u64,
    pub diverged_decisions: u64,
    pub divergence_rate: f64,
    pub current: TaskWindowStats,
    pub previous: TaskWindowStats,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionWriteDrift {
    pub session_id: String,
    pub message_rows: u64,
    pub event_rows: u64,
    pub delta: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WriteConsistencyReport {
    pub generated_at: String,
    pub conversation_event_rows: u64,
    pub missing_message_id_events: u64,
    pub global_delta: i64,
    pub session_mismatch_count: u64,
    pub stale_task_starts: u64,
    pub top_session_drifts: Vec<SessionWriteDrift>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct WriteConsistencyThresholds {
    pub max_abs_global_delta: u64,
    pub max_session_mismatch_count: u64,
    pub max_stale_task_starts: u64,
    pub max_missing_message_id_events: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WriteConsistencyGateStatus {
    pub passed: bool,
    pub reasons: Vec<String>,
    pub thresholds: WriteConsistencyThresholds,
}

impl PolicyGraduationReport {
    pub fn gate_passes(&self, max_divergence: f64) -> bool {
        if self.observed_days < self.window_days as f64 {
            return false;
        }
        if self.total_decisions == 0 {
            return false;
        }
        if self.divergence_rate >= max_divergence {
            return false;
        }
        // No-regression gate:
        // completion must not decrease; error/stall must not increase.
        let completion_ok = self.current.completion_rate >= self.previous.completion_rate;
        let error_ok = self.current.error_rate <= self.previous.error_rate;
        let stall_ok = self.current.stall_rate <= self.previous.stall_rate;
        completion_ok && error_ok && stall_ok
    }
}

impl Default for WriteConsistencyThresholds {
    fn default() -> Self {
        Self {
            // Canonical event-path defaults.
            max_abs_global_delta: 3,
            max_session_mismatch_count: 0,
            max_stale_task_starts: 0,
            max_missing_message_id_events: 0,
        }
    }
}

impl WriteConsistencyReport {
    pub fn evaluate_gate(&self) -> WriteConsistencyGateStatus {
        self.evaluate_gate_with(WriteConsistencyThresholds::default())
    }

    pub fn evaluate_gate_with(
        &self,
        thresholds: WriteConsistencyThresholds,
    ) -> WriteConsistencyGateStatus {
        let mut reasons = Vec::new();

        let abs_global_delta = self.global_delta.unsigned_abs();
        if abs_global_delta > thresholds.max_abs_global_delta {
            reasons.push(format!(
                "global delta {} exceeds threshold {}",
                abs_global_delta, thresholds.max_abs_global_delta
            ));
        }
        if self.session_mismatch_count > thresholds.max_session_mismatch_count {
            reasons.push(format!(
                "session mismatch count {} exceeds threshold {}",
                self.session_mismatch_count, thresholds.max_session_mismatch_count
            ));
        }

        if self.stale_task_starts > thresholds.max_stale_task_starts {
            reasons.push(format!(
                "stale task starts {} exceeds threshold {}",
                self.stale_task_starts, thresholds.max_stale_task_starts
            ));
        }

        if self.missing_message_id_events > thresholds.max_missing_message_id_events {
            reasons.push(format!(
                "events missing message_id {} exceeds threshold {}",
                self.missing_message_id_events, thresholds.max_missing_message_id_events
            ));
        }

        WriteConsistencyGateStatus {
            passed: reasons.is_empty(),
            reasons,
            thresholds,
        }
    }
}

impl EventStore {
    /// Create a new EventStore with the given database pool.
    /// This also runs migrations to create/update the events table.
    pub async fn new(pool: SqlitePool) -> anyhow::Result<Self> {
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    /// Get the underlying database pool (for sharing with other components)
    pub fn pool(&self) -> SqlitePool {
        self.pool.clone()
    }

    /// Run database migrations for the events table
    async fn migrate(&self) -> anyhow::Result<()> {
        crate::db::migrations::migrate_events(&self.pool).await
    }

    // =========================================================================
    // Write Operations
    // =========================================================================

    /// Append a new event to the store. Returns the assigned event ID.
    pub async fn append(&self, event: Event) -> anyhow::Result<i64> {
        let data_json = serde_json::to_string(&event.data)?;
        let event_type_str = event.event_type.as_str();
        let created_at_str = event.created_at.to_rfc3339();

        let result = sqlx::query(
            r#"
            INSERT INTO events (session_id, event_type, data, created_at, task_id, tool_name)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&event.session_id)
        .bind(event_type_str)
        .bind(&data_json)
        .bind(&created_at_str)
        .bind(&event.task_id)
        .bind(&event.tool_name)
        .execute(&self.pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Mark events as consolidated
    pub async fn mark_consolidated(&self, event_ids: &[i64]) -> anyhow::Result<()> {
        if event_ids.is_empty() {
            return Ok(());
        }

        let now = Utc::now().to_rfc3339();
        let placeholders: Vec<String> = event_ids.iter().map(|_| "?".to_string()).collect();
        let query = format!(
            "UPDATE events SET consolidated_at = ? WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut q = sqlx::query(&query).bind(&now);
        for id in event_ids {
            q = q.bind(id);
        }
        q.execute(&self.pool).await?;

        Ok(())
    }

    /// Delete old consolidated events (for pruning)
    pub async fn delete_old_consolidated(&self, before: DateTime<Utc>) -> anyhow::Result<u64> {
        let before_str = before.to_rfc3339();

        let result =
            sqlx::query("DELETE FROM events WHERE consolidated_at IS NOT NULL AND created_at < ?")
                .bind(&before_str)
                .execute(&self.pool)
                .await?;

        Ok(result.rows_affected())
    }

    // =========================================================================
    // Read Operations - General Queries
    // =========================================================================

    /// Query events for a session within a time window
    pub async fn query_events(
        &self,
        session_id: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<Vec<Event>> {
        let since_str = since.to_rfc3339();

        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND created_at >= ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .bind(&since_str)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query events by type for a session
    pub async fn query_events_by_types(
        &self,
        session_id: &str,
        types: &[EventType],
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        if types.is_empty() {
            return Ok(vec![]);
        }

        let type_strs: Vec<&str> = types.iter().map(|t| t.as_str()).collect();
        let placeholders: Vec<String> = types.iter().map(|_| "?".to_string()).collect();

        let query = format!(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND event_type IN ({})
            ORDER BY created_at DESC
            LIMIT ?
            "#,
            placeholders.join(",")
        );

        let mut q = sqlx::query(&query).bind(session_id);
        for type_str in type_strs {
            q = q.bind(type_str);
        }
        q = q.bind(limit as i64);

        let rows = q.fetch_all(&self.pool).await?;
        self.rows_to_events(rows)
    }

    /// Query recent events for a session (all types)
    pub async fn query_recent_events(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        // Reverse to get chronological order
        let mut events = self.rows_to_events(rows)?;
        events.reverse();
        Ok(events)
    }

    /// Query events for a specific task
    pub async fn query_task_events(&self, task_id: &str) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query events for a specific task scoped to a session.
    pub async fn query_task_events_for_session(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query recent task_end events for a session.
    /// When failures_only is true, only failed task_end events are returned.
    pub async fn query_recent_task_ends(
        &self,
        session_id: &str,
        failures_only: bool,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        let fetch_limit = if failures_only {
            limit.saturating_mul(8)
        } else {
            limit
        }
        .max(1);
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ?
              AND event_type = 'task_end'
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(fetch_limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut events = self.rows_to_events(rows)?;
        if failures_only {
            events.retain(|e| {
                e.parse_data::<TaskEndData>()
                    .ok()
                    .is_some_and(|d| matches!(d.status, TaskStatus::Failed))
            });
        }
        events.truncate(limit.max(1));
        Ok(events)
    }

    /// Query decision_point events for a specific task scoped to a session.
    pub async fn query_decision_points(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND task_id = ? AND event_type = 'decision_point'
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query recent intent-gate decision_point events scoped to a session.
    /// Returned in reverse-chronological order.
    pub async fn query_recent_intent_gate_decision_points(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        let fetch_limit = limit.max(1).saturating_mul(5).max(20);
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND event_type = 'decision_point'
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(fetch_limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut events = self.rows_to_events(rows)?;
        events.retain(|e| {
            e.parse_data::<DecisionPointData>()
                .ok()
                .is_some_and(|d| d.decision_type == DecisionType::IntentGate)
        });
        events.truncate(limit.max(1));
        Ok(events)
    }

    /// Get unconsolidated events for a session (for consolidation)
    pub async fn query_unconsolidated(&self, session_id: &str) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND consolidated_at IS NULL
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Get sessions with unconsolidated events older than a cutoff
    pub async fn get_sessions_needing_consolidation(&self) -> anyhow::Result<Vec<String>> {
        let rows = sqlx::query(
            r#"
            SELECT DISTINCT session_id
            FROM events
            WHERE consolidated_at IS NULL
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(|r| r.get("session_id")).collect())
    }

    /// Get sessions with old unconsolidated events (before cutoff)
    pub async fn get_sessions_with_old_unconsolidated_events(
        &self,
        before: DateTime<Utc>,
    ) -> anyhow::Result<Vec<String>> {
        let before_str = before.to_rfc3339();

        let rows = sqlx::query(
            r#"
            SELECT DISTINCT session_id
            FROM events
            WHERE consolidated_at IS NULL AND created_at < ?
            "#,
        )
        .bind(&before_str)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(|r| r.get("session_id")).collect())
    }

    // =========================================================================
    // Read Operations - Conversation History
    // =========================================================================

    /// Get conversation history for a session (for LLM context)
    /// Returns runtime messages projected from canonical events.
    pub async fn get_conversation_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let events = self
            .query_events_by_types(
                session_id,
                &[
                    EventType::UserMessage,
                    EventType::AssistantResponse,
                    EventType::ToolResult,
                ],
                limit * 3, // Fetch more to account for tool results
            )
            .await?;

        // Convert events to runtime Message format.
        // Reverse to chronological (query returns newest-first).
        let mut messages = Vec::new();
        for event in events.into_iter().rev() {
            if let Some(msg) = crate::events::turn_from_event(
                event.id,
                &event.session_id,
                event.event_type.as_str(),
                &event.data,
                event.created_at,
            )
            .map(|turn| turn.into_message())
            {
                messages.push(msg);
            }
        }

        Ok(crate::conversation::truncate_with_anchor(messages, limit))
    }

    // =========================================================================
    // Read Operations - Specific Queries for Context
    // =========================================================================

    /// Get the most recent error for a session
    pub async fn get_last_error(&self, session_id: &str) -> anyhow::Result<Option<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND event_type = 'error'
            ORDER BY created_at DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let events = self.rows_to_events(rows)?;
        Ok(events.into_iter().next())
    }

    /// Get the current active task (TaskStart without matching TaskEnd)
    pub async fn get_active_task(&self, session_id: &str) -> anyhow::Result<Option<Event>> {
        // Get all task events in the last hour
        let since = Utc::now() - Duration::hours(1);
        let since_str = since.to_rfc3339();

        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND event_type IN ('task_start', 'task_end') AND created_at >= ?
            ORDER BY created_at DESC
            "#,
        )
        .bind(session_id)
        .bind(&since_str)
        .fetch_all(&self.pool)
        .await?;

        let events = self.rows_to_events(rows)?;

        // Find TaskStart without matching TaskEnd
        let mut ended_tasks: std::collections::HashSet<String> = std::collections::HashSet::new();

        for event in &events {
            if event.event_type == EventType::TaskEnd {
                if let Some(task_id) = &event.task_id {
                    ended_tasks.insert(task_id.clone());
                }
            }
        }

        for event in events {
            if event.event_type == EventType::TaskStart {
                if let Some(task_id) = &event.task_id {
                    if !ended_tasks.contains(task_id) {
                        return Ok(Some(event));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Reconcile stale TaskStart events that never received a matching TaskEnd.
    ///
    /// This emits synthetic failed TaskEnd events so UI/DB task state self-heals
    /// even when an agent process died outside channel watchdog loops.
    pub async fn reconcile_stale_task_starts(
        &self,
        stale_after_secs: i64,
        batch_size: usize,
    ) -> anyhow::Result<u64> {
        let stale_after_secs = stale_after_secs.max(1);
        let cutoff = Utc::now() - Duration::seconds(stale_after_secs);
        let cutoff_str = cutoff.to_rfc3339();

        let rows = sqlx::query(
            r#"
            SELECT s.session_id AS session_id,
                   s.task_id AS task_id,
                   MIN(s.created_at) AS started_at
            FROM events s
            WHERE s.event_type = 'task_start'
              AND s.task_id IS NOT NULL
              AND s.created_at < ?
              AND NOT EXISTS (
                SELECT 1
                FROM events e
                WHERE e.session_id = s.session_id
                  AND e.task_id = s.task_id
                  AND e.event_type = 'task_end'
              )
            GROUP BY s.session_id, s.task_id
            ORDER BY MIN(s.created_at) ASC
            LIMIT ?
            "#,
        )
        .bind(&cutoff_str)
        .bind(batch_size.max(1) as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut reconciled = 0u64;
        for row in rows {
            let session_id: String = row.get("session_id");
            let task_id: String = row.get("task_id");
            let started_at_raw: String = row.get("started_at");

            // Re-check to avoid duplicate synthetic task_end if a real one was
            // written between candidate query and append.
            let has_end = sqlx::query(
                r#"
                SELECT 1
                FROM events
                WHERE session_id = ? AND task_id = ? AND event_type = 'task_end'
                LIMIT 1
                "#,
            )
            .bind(&session_id)
            .bind(&task_id)
            .fetch_optional(&self.pool)
            .await?
            .is_some();
            if has_end {
                continue;
            }

            let started_at = DateTime::parse_from_rfc3339(&started_at_raw)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(cutoff);
            let duration_secs = (Utc::now() - started_at).num_seconds().max(0) as u64;
            let stale_after_mins = (stale_after_secs / 60).max(1);

            let event = Event::new(
                session_id.clone(),
                EventType::TaskEnd,
                serde_json::to_value(TaskEndData {
                    task_id: task_id.clone(),
                    status: TaskStatus::Failed,
                    duration_secs,
                    iterations: 0,
                    tool_calls_count: 0,
                    error: Some(format!(
                        "Auto-failed by watchdog after {} minute(s) without task_end",
                        stale_after_mins
                    )),
                    summary: Some("Recovered stale in-flight task".to_string()),
                })?,
            );
            self.append(event).await?;
            reconciled += 1;
            info!(
                session_id = %session_id,
                task_id = %task_id,
                duration_secs,
                "Reconciled stale task_start with synthetic task_end"
            );
        }

        Ok(reconciled)
    }

    /// Get recent tool calls for a session
    pub async fn get_recent_tool_calls(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        self.query_events_by_types(
            session_id,
            &[EventType::ToolCall, EventType::ToolResult],
            limit,
        )
        .await
    }

    pub async fn get_tool_stats(
        &self,
        tool_name: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<ToolStats> {
        let since_str = since.to_rfc3339();
        let rows = sqlx::query(
            r#"
            SELECT data
            FROM events
            WHERE event_type = 'tool_result'
              AND tool_name = ?
              AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 500
            "#,
        )
        .bind(tool_name)
        .bind(&since_str)
        .fetch_all(&self.pool)
        .await?;

        let mut total_calls = 0u64;
        let mut successful = 0u64;
        let mut failed = 0u64;
        let mut duration_sum_ms: u128 = 0;
        let mut error_counts: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();

        for row in rows {
            let data_str: String = row.get("data");
            let tr: ToolResultData = match serde_json::from_str(&data_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if is_synthetic_tool_result(&tr) {
                continue;
            }

            total_calls += 1;
            duration_sum_ms += tr.duration_ms as u128;
            if tr.success {
                successful += 1;
                continue;
            }
            failed += 1;

            let raw_error = tr.error.as_deref().unwrap_or(&tr.result);
            let normalized = normalize_tool_error_text(raw_error);
            let pattern = crate::memory::procedures::extract_error_pattern(normalized);
            if !pattern.trim().is_empty() {
                *error_counts.entry(pattern).or_insert(0) += 1;
            }
        }

        let avg_duration_ms = if total_calls == 0 {
            0
        } else {
            (duration_sum_ms / total_calls as u128) as u64
        };

        let mut common_errors: Vec<(String, u64)> = error_counts.into_iter().collect();
        common_errors.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        common_errors.truncate(3);

        Ok(ToolStats {
            total_calls,
            successful,
            failed,
            avg_duration_ms,
            common_errors,
        })
    }

    /// Get the last completed task for a session
    pub async fn get_last_completed_task(&self, session_id: &str) -> anyhow::Result<Option<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE session_id = ? AND event_type = 'task_end'
            ORDER BY created_at DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let events = self.rows_to_events(rows)?;
        Ok(events.into_iter().next())
    }

    /// Query all events of a single type in [start, end).
    pub async fn query_events_by_type_between(
        &self,
        event_type: EventType,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name
            FROM events
            WHERE event_type = ? AND created_at >= ? AND created_at < ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(event_type.as_str())
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(&self.pool)
        .await?;
        self.rows_to_events(rows)
    }

    /// Return the earliest created_at for an event type.
    pub async fn earliest_event_time_by_type(
        &self,
        event_type: EventType,
    ) -> anyhow::Result<Option<DateTime<Utc>>> {
        let row = sqlx::query(
            r#"
            SELECT created_at
            FROM events
            WHERE event_type = ?
            ORDER BY created_at ASC
            LIMIT 1
            "#,
        )
        .bind(event_type.as_str())
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let raw: String = row.get("created_at");
        let parsed = DateTime::parse_from_rfc3339(&raw)
            .map(|dt| dt.with_timezone(&Utc))
            .ok();
        Ok(parsed)
    }

    /// Build a graduation report for policy routing gate checks.
    pub async fn policy_graduation_report(
        &self,
        window_days: u32,
    ) -> anyhow::Result<PolicyGraduationReport> {
        let now = Utc::now();
        let window = Duration::days(window_days as i64);
        let start_current = now - window;
        let start_previous = start_current - window;

        let decisions = self
            .query_events_by_type_between(EventType::PolicyDecision, start_current, now)
            .await?;
        let mut total_decisions = 0u64;
        let mut diverged_decisions = 0u64;
        for event in decisions {
            if let Ok(data) = event.parse_data::<PolicyDecisionData>() {
                total_decisions += 1;
                if data.diverged {
                    diverged_decisions += 1;
                }
            }
        }
        let divergence_rate = if total_decisions > 0 {
            diverged_decisions as f64 / total_decisions as f64
        } else {
            0.0
        };

        let current = self
            .task_window_stats(start_current, now)
            .await
            .unwrap_or_default();
        let previous = self
            .task_window_stats(start_previous, start_current)
            .await
            .unwrap_or_default();

        let observed_days = match self
            .earliest_event_time_by_type(EventType::PolicyDecision)
            .await?
        {
            Some(first) => (now - first).num_seconds().max(0) as f64 / 86_400.0,
            None => 0.0,
        };

        Ok(PolicyGraduationReport {
            window_days,
            observed_days,
            total_decisions,
            diverged_decisions,
            divergence_rate,
            current,
            previous,
        })
    }

    /// Return canonical write-path consistency metrics from the event stream.
    pub async fn write_consistency_report(
        &self,
        _top_n_sessions: usize,
    ) -> anyhow::Result<WriteConsistencyReport> {
        let conversation_event_rows: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM events
            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let missing_message_id_events: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM events
            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
              AND (
                json_extract(data, '$.message_id') IS NULL
                OR TRIM(CAST(json_extract(data, '$.message_id') AS TEXT)) = ''
              )
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let stale_task_starts: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM (
                SELECT s.session_id, s.task_id
                FROM events s
                WHERE s.event_type = 'task_start'
                  AND s.task_id IS NOT NULL
                GROUP BY s.session_id, s.task_id
                HAVING NOT EXISTS (
                    SELECT 1
                    FROM events e
                    WHERE e.session_id = s.session_id
                      AND e.task_id = s.task_id
                      AND e.event_type = 'task_end'
                )
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        Ok(WriteConsistencyReport {
            generated_at: Utc::now().to_rfc3339(),
            conversation_event_rows: to_u64(conversation_event_rows),
            missing_message_id_events: to_u64(missing_message_id_events),
            global_delta: 0,
            session_mismatch_count: 0,
            stale_task_starts: to_u64(stale_task_starts),
            top_session_drifts: Vec::new(),
        })
    }

    async fn task_window_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<TaskWindowStats> {
        let task_ends = self
            .query_events_by_type_between(EventType::TaskEnd, start, end)
            .await?;
        let errors = self
            .query_events_by_type_between(EventType::Error, start, end)
            .await?;

        let mut stats = TaskWindowStats {
            total: task_ends.len() as u64,
            ..TaskWindowStats::default()
        };
        for event in task_ends {
            if let Ok(data) = event.parse_data::<TaskEndData>() {
                match data.status {
                    TaskStatus::Completed => stats.completed += 1,
                    TaskStatus::Failed => stats.failed += 1,
                    TaskStatus::Cancelled => stats.cancelled += 1,
                }
                let stalled = data
                    .error
                    .as_deref()
                    .map(|e| e.to_ascii_lowercase().contains("stalled"))
                    .unwrap_or(false)
                    || data
                        .summary
                        .as_deref()
                        .map(|s| s.to_ascii_lowercase().contains("stalled"))
                        .unwrap_or(false);
                if stalled {
                    stats.stalled += 1;
                }
            }
        }
        stats.error_events = errors.len() as u64;

        if stats.total > 0 {
            stats.completion_rate = stats.completed as f64 / stats.total as f64;
            stats.error_rate = stats.error_events as f64 / stats.total as f64;
            stats.stall_rate = stats.stalled as f64 / stats.total as f64;
        }

        Ok(stats)
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn rows_to_events(&self, rows: Vec<sqlx::sqlite::SqliteRow>) -> anyhow::Result<Vec<Event>> {
        let mut events = Vec::new();
        for row in rows {
            let id: i64 = row.get("id");
            let session_id: String = row.get("session_id");
            let event_type_str: String = row.get("event_type");
            let data_str: String = row.get("data");
            let created_at_str: String = row.get("created_at");
            let consolidated_at_str: Option<String> = row.get("consolidated_at");
            let task_id: Option<String> = row.get("task_id");
            let tool_name: Option<String> = row.get("tool_name");

            let event_type = match EventType::from_str(&event_type_str) {
                Some(et) => et,
                None => {
                    warn!("Unknown event type: {}", event_type_str);
                    continue;
                }
            };

            let data: serde_json::Value = serde_json::from_str(&data_str)?;

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let consolidated_at = consolidated_at_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            });

            events.push(Event {
                id,
                session_id,
                event_type,
                data,
                created_at,
                consolidated_at,
                task_id,
                tool_name,
            });
        }
        Ok(events)
    }
}

fn to_u64(value: i64) -> u64 {
    if value <= 0 {
        0
    } else {
        value as u64
    }
}

fn normalize_tool_error_text(raw: &str) -> &str {
    let diag = raw.find("\n\n[DIAGNOSTIC]");
    let stats = raw.find("\n\n[TOOL STATS]");
    let sys = raw.find("\n\n[SYSTEM]");
    let cut_at = [diag, stats, sys].into_iter().flatten().min();
    let trimmed = match cut_at {
        Some(idx) => &raw[..idx],
        None => raw,
    };
    trimmed.trim()
}

fn is_synthetic_tool_result(tr: &ToolResultData) -> bool {
    tr.success
        && tr.duration_ms == 0
        && tr.error.is_none()
        && tr.result.trim_start().starts_with("[SYSTEM]")
}

/// Builder for emitting events with a consistent session context
pub struct EventEmitter {
    store: Arc<EventStore>,
    session_id: String,
    current_task_id: Option<String>,
}

impl EventEmitter {
    pub fn new(store: Arc<EventStore>, session_id: impl Into<String>) -> Self {
        Self {
            store,
            session_id: session_id.into(),
            current_task_id: None,
        }
    }

    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.current_task_id = Some(task_id.into());
        self
    }

    pub fn set_task_id(&mut self, task_id: Option<String>) {
        self.current_task_id = task_id;
    }

    /// Emit an event with the current context
    pub async fn emit<T: serde::Serialize>(
        &self,
        event_type: EventType,
        data: T,
    ) -> anyhow::Result<i64> {
        let mut json_data = serde_json::to_value(data)?;

        // Inject task_id if present and not already in data
        if let Some(task_id) = &self.current_task_id {
            if let Some(obj) = json_data.as_object_mut() {
                if !obj.contains_key("task_id") {
                    obj.insert("task_id".to_string(), serde_json::json!(task_id));
                }
            }
        }

        let event = Event::new(&self.session_id, event_type, json_data);
        self.store.append(event).await
    }

    /// Get the underlying store
    pub fn store(&self) -> Arc<EventStore> {
        self.store.clone()
    }

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use serde_json::json;

    async fn setup_store() -> (EventStore, tempfile::NamedTempFile) {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.expect("connect sqlite");
        let store = EventStore::new(pool).await.expect("init event store");
        (store, db_file)
    }

    async fn append_event_at(
        store: &EventStore,
        session_id: &str,
        event_type: EventType,
        data: serde_json::Value,
        created_at: DateTime<Utc>,
    ) {
        let mut event = Event::new(session_id, event_type, data);
        event.created_at = created_at;
        store.append(event).await.expect("append event");
    }

    async fn append_policy_decision(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        diverged: bool,
        created_at: DateTime<Utc>,
    ) {
        let payload = PolicyDecisionData {
            task_id: task_id.to_string(),
            old_model: "old-model".to_string(),
            new_model: "new-model".to_string(),
            old_tier: "primary".to_string(),
            new_profile: "balanced".to_string(),
            diverged,
            policy_enforce: false,
            risk_score: 0.3,
            uncertainty_score: 0.2,
        };
        append_event_at(
            store,
            session_id,
            EventType::PolicyDecision,
            serde_json::to_value(payload).expect("serialize policy decision"),
            created_at,
        )
        .await;
    }

    async fn append_task_end(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        status: TaskStatus,
        created_at: DateTime<Utc>,
        error: Option<&str>,
        summary: Option<&str>,
    ) {
        let payload = TaskEndData {
            task_id: task_id.to_string(),
            status,
            duration_secs: 1,
            iterations: 1,
            tool_calls_count: 0,
            error: error.map(str::to_string),
            summary: summary.map(str::to_string),
        };
        append_event_at(
            store,
            session_id,
            EventType::TaskEnd,
            serde_json::to_value(payload).expect("serialize task end"),
            created_at,
        )
        .await;
    }

    async fn append_task_start(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        created_at: DateTime<Utc>,
    ) {
        append_event_at(
            store,
            session_id,
            EventType::TaskStart,
            json!({
                "task_id": task_id,
                "description": format!("task {}", task_id)
            }),
            created_at,
        )
        .await;
    }

    async fn append_decision_point(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        created_at: DateTime<Utc>,
    ) {
        append_event_at(
            store,
            session_id,
            EventType::DecisionPoint,
            json!({
                "decision_type":"intent_gate",
                "task_id": task_id,
                "iteration": 1,
                "metadata":{"needs_tools":true},
                "summary":"intent gate forced tool mode"
            }),
            created_at,
        )
        .await;
    }

    struct ToolResultFixture<'a> {
        tool: &'a str,
        success: bool,
        duration_ms: u64,
        result: &'a str,
        error: Option<&'a str>,
        created_at: DateTime<Utc>,
    }

    async fn append_tool_result(
        store: &EventStore,
        session_id: &str,
        fixture: ToolResultFixture<'_>,
    ) {
        let mut payload = json!({
            "tool_call_id": format!(
                "tc-{}-{}",
                fixture.tool,
                fixture.created_at.timestamp_nanos_opt().unwrap_or(0)
            ),
            "name": fixture.tool,
            "result": fixture.result,
            "success": fixture.success,
            "duration_ms": fixture.duration_ms,
        });
        if let Some(err) = fixture.error {
            payload["error"] = json!(err);
        }
        append_event_at(
            store,
            session_id,
            EventType::ToolResult,
            payload,
            fixture.created_at,
        )
        .await;
    }

    #[tokio::test]
    async fn graduation_report_passes_with_low_divergence_and_no_regression() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-pass";

        // Ensure observed_days >= 7
        append_policy_decision(&store, session, "old-task", false, now - Duration::days(8)).await;
        for i in 0..20 {
            append_policy_decision(
                &store,
                session,
                &format!("cur-{i}"),
                false,
                now - Duration::hours(6) + Duration::minutes(i as i64),
            )
            .await;
        }

        // Previous window: weaker quality
        append_task_end(
            &store,
            session,
            "prev-1",
            TaskStatus::Completed,
            now - Duration::days(10),
            None,
            Some("completed"),
        )
        .await;
        append_task_end(
            &store,
            session,
            "prev-2",
            TaskStatus::Failed,
            now - Duration::days(9),
            Some("stalled waiting for output"),
            Some("stalled"),
        )
        .await;
        append_event_at(
            &store,
            session,
            EventType::Error,
            json!({"message":"previous error"}),
            now - Duration::days(9),
        )
        .await;

        // Current window: improved quality
        append_task_end(
            &store,
            session,
            "cur-1",
            TaskStatus::Completed,
            now - Duration::days(2),
            None,
            Some("done"),
        )
        .await;
        append_task_end(
            &store,
            session,
            "cur-2",
            TaskStatus::Completed,
            now - Duration::days(1),
            None,
            Some("done"),
        )
        .await;

        let report = store.policy_graduation_report(7).await.expect("report");
        assert!(report.observed_days >= 7.0);
        assert_eq!(report.total_decisions, 20);
        assert_eq!(report.diverged_decisions, 0);
        assert!(report.gate_passes(0.05));
        assert!(report.current.completion_rate >= report.previous.completion_rate);
        assert!(report.current.error_rate <= report.previous.error_rate);
        assert!(report.current.stall_rate <= report.previous.stall_rate);
    }

    #[tokio::test]
    async fn graduation_report_fails_when_divergence_exceeds_threshold() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-diverge";

        append_policy_decision(&store, session, "old-task", false, now - Duration::days(8)).await;
        for i in 0..20 {
            append_policy_decision(
                &store,
                session,
                &format!("cur-{i}"),
                i < 2,
                now - Duration::hours(3) + Duration::minutes(i as i64),
            )
            .await;
        }

        // Keep quality metrics equal so divergence is the failing reason.
        append_task_end(
            &store,
            session,
            "prev-1",
            TaskStatus::Completed,
            now - Duration::days(9),
            None,
            Some("done"),
        )
        .await;
        append_task_end(
            &store,
            session,
            "cur-1",
            TaskStatus::Completed,
            now - Duration::days(1),
            None,
            Some("done"),
        )
        .await;

        let report = store.policy_graduation_report(7).await.expect("report");
        assert!(report.observed_days >= 7.0);
        assert!(report.divergence_rate > 0.05);
        assert!(!report.gate_passes(0.05));
    }

    #[tokio::test]
    async fn graduation_report_fails_when_observation_window_is_too_short() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-short-window";

        // Earliest policy decision is only 2 days old.
        for i in 0..8 {
            append_policy_decision(
                &store,
                session,
                &format!("cur-{i}"),
                false,
                now - Duration::days(2) + Duration::hours(i as i64),
            )
            .await;
        }

        append_task_end(
            &store,
            session,
            "cur-1",
            TaskStatus::Completed,
            now - Duration::hours(12),
            None,
            Some("done"),
        )
        .await;

        let report = store.policy_graduation_report(7).await.expect("report");
        assert!(report.observed_days < 7.0);
        assert!(!report.gate_passes(0.05));
    }

    #[tokio::test]
    async fn query_recent_task_ends_and_decision_points_are_session_scoped() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        append_task_end(
            &store,
            "s1",
            "task-failed",
            TaskStatus::Failed,
            now - Duration::minutes(2),
            Some("boom"),
            None,
        )
        .await;
        append_task_end(
            &store,
            "s1",
            "task-ok",
            TaskStatus::Completed,
            now - Duration::minutes(1),
            None,
            Some("ok"),
        )
        .await;
        append_task_end(
            &store,
            "s2",
            "task-s2",
            TaskStatus::Failed,
            now - Duration::minutes(1),
            Some("other"),
            None,
        )
        .await;
        append_decision_point(&store, "s1", "task-failed", now - Duration::minutes(2)).await;
        append_decision_point(&store, "s2", "task-failed", now - Duration::minutes(2)).await;

        let s1_failed = store
            .query_recent_task_ends("s1", true, 10)
            .await
            .expect("query failed");
        assert_eq!(s1_failed.len(), 1);
        assert_eq!(s1_failed[0].session_id, "s1");

        let s1_decisions = store
            .query_decision_points("s1", "task-failed")
            .await
            .expect("query decision points");
        assert_eq!(s1_decisions.len(), 1);
        assert_eq!(s1_decisions[0].session_id, "s1");
    }

    #[tokio::test]
    async fn query_recent_intent_gate_decision_points_filters_and_scopes() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        append_decision_point(&store, "s1", "task-1", now - Duration::minutes(3)).await;
        append_event_at(
            &store,
            "s1",
            EventType::DecisionPoint,
            json!({
                "decision_type":"stopping_condition",
                "task_id":"task-1",
                "iteration":2,
                "metadata":{"reason":"stall"},
                "summary":"stopping condition fired"
            }),
            now - Duration::minutes(2),
        )
        .await;
        append_decision_point(&store, "s2", "task-2", now - Duration::minutes(1)).await;

        let s1_recent = store
            .query_recent_intent_gate_decision_points("s1", 10)
            .await
            .expect("query recent intent gate decision points");
        assert_eq!(s1_recent.len(), 1);
        assert_eq!(s1_recent[0].session_id, "s1");
        let parsed = s1_recent[0]
            .parse_data::<DecisionPointData>()
            .expect("parse decision point");
        assert_eq!(parsed.decision_type, DecisionType::IntentGate);
    }

    #[tokio::test]
    async fn reconcile_stale_task_starts_appends_failed_task_end() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        // Stale task with no task_end -> should be reconciled.
        append_task_start(
            &store,
            "s-reconcile",
            "task-stale",
            now - Duration::minutes(10),
        )
        .await;

        // Stale task with task_end already present -> should be ignored.
        append_task_start(
            &store,
            "s-reconcile",
            "task-complete",
            now - Duration::minutes(10),
        )
        .await;
        append_task_end(
            &store,
            "s-reconcile",
            "task-complete",
            TaskStatus::Completed,
            now - Duration::minutes(9),
            None,
            Some("ok"),
        )
        .await;

        // Recent task start -> should remain active.
        append_task_start(
            &store,
            "s-reconcile",
            "task-recent",
            now - Duration::minutes(1),
        )
        .await;

        let reconciled = store
            .reconcile_stale_task_starts(300, 10)
            .await
            .expect("reconcile stale starts");
        assert_eq!(reconciled, 1);

        let stale_events = store
            .query_task_events_for_session("s-reconcile", "task-stale")
            .await
            .expect("query stale task events");
        assert_eq!(stale_events.len(), 2, "task-stale should have start+end");
        assert_eq!(stale_events[1].event_type, EventType::TaskEnd);
        let stale_end = stale_events[1]
            .parse_data::<TaskEndData>()
            .expect("parse stale task_end");
        assert_eq!(stale_end.status, TaskStatus::Failed);
        assert!(
            stale_end
                .error
                .as_deref()
                .is_some_and(|e| e.contains("Auto-failed by watchdog")),
            "synthetic task_end should include watchdog reason"
        );

        let recent_events = store
            .query_task_events_for_session("s-reconcile", "task-recent")
            .await
            .expect("query recent task events");
        assert_eq!(recent_events.len(), 1, "recent task should stay open");

        // Running again should be idempotent.
        let reconciled_again = store
            .reconcile_stale_task_starts(300, 10)
            .await
            .expect("second reconcile");
        assert_eq!(reconciled_again, 0);
    }

    #[tokio::test]
    async fn conversation_history_preserves_tool_call_extra_content() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        append_event_at(
            &store,
            "s-extra",
            EventType::AssistantResponse,
            json!({
                "message_id": "assistant-msg-1",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "name": "run_command",
                    "arguments": { "command": "ls -la" },
                    "extra_content": { "thought_signature": "sig-123" }
                }],
                "model": "gemini-2.5-pro",
                "input_tokens": 12,
                "output_tokens": 3
            }),
            now,
        )
        .await;

        let history = store
            .get_conversation_history("s-extra", 10)
            .await
            .expect("conversation history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "assistant");

        let tool_calls_json = history[0]
            .tool_calls_json
            .as_deref()
            .expect("assistant tool calls should exist");
        let tool_calls: Vec<crate::traits::ToolCall> =
            serde_json::from_str(tool_calls_json).expect("parse tool calls");
        assert_eq!(tool_calls.len(), 1);
        let extra = tool_calls[0]
            .extra_content
            .as_ref()
            .expect("extra_content should be preserved");
        assert_eq!(extra["thought_signature"], "sig-123");
    }

    #[tokio::test]
    async fn write_consistency_report_uses_event_stream_only() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        append_event_at(
            &store,
            "s-no-messages",
            EventType::UserMessage,
            json!({
                "content": "hello from event stream",
                "message_id": "event-msg-1",
                "has_attachments": false
            }),
            now,
        )
        .await;

        let report = store
            .write_consistency_report(5)
            .await
            .expect("write consistency");
        assert_eq!(report.conversation_event_rows, 1);
        assert_eq!(report.missing_message_id_events, 0);
        assert_eq!(report.global_delta, 0);
        assert_eq!(report.session_mismatch_count, 0);
        assert!(report.top_session_drifts.is_empty());
        assert!(
            report.evaluate_gate().passed,
            "event-only mode should pass with complete message IDs"
        );
    }

    #[tokio::test]
    async fn write_consistency_report_counts_missing_message_ids() {
        let (store, _db_file) = setup_store().await;
        append_event_at(
            &store,
            "s-drift",
            EventType::UserMessage,
            json!({
                "content": "hello from event stream",
                "message_id": null,
                "has_attachments": false
            }),
            Utc::now(),
        )
        .await;

        let report = store
            .write_consistency_report(5)
            .await
            .expect("write consistency");

        assert_eq!(report.conversation_event_rows, 1);
        assert_eq!(report.missing_message_id_events, 1);
        assert_eq!(report.global_delta, 0);
        assert_eq!(report.session_mismatch_count, 0);
        assert!(report.top_session_drifts.is_empty());
        assert!(
            !report.evaluate_gate().passed,
            "default gate should fail when event payloads are missing message_id"
        );
    }

    #[tokio::test]
    async fn get_tool_stats_aggregates_and_groups_errors() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-tool-stats-1";

        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "terminal",
                success: true,
                duration_ms: 100,
                result: "ok",
                error: None,
                created_at: now - Duration::minutes(50),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "terminal",
                success: true,
                duration_ms: 300,
                result: "ok",
                error: None,
                created_at: now - Duration::minutes(40),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "terminal",
                success: false,
                duration_ms: 200,
                result: "Error: Connection timed out at /tmp/foo.rs:12:3",
                error: Some("Error: Connection timed out at /tmp/foo.rs:12:3"),
                created_at: now - Duration::minutes(30),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "terminal",
                success: false,
                duration_ms: 400,
                result: "Error: Connection timed out at /tmp/bar.rs:99:1",
                error: Some("Error: Connection timed out at /tmp/bar.rs:99:1"),
                created_at: now - Duration::minutes(20),
            },
        )
        .await;

        let stats = store
            .get_tool_stats("terminal", now - Duration::hours(24))
            .await
            .expect("tool stats");

        assert_eq!(stats.total_calls, 4);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.failed, 2);
        assert_eq!(stats.avg_duration_ms, 250);
        assert_eq!(stats.common_errors.len(), 1);
        assert_eq!(stats.common_errors[0].1, 2);
    }

    #[tokio::test]
    async fn get_tool_stats_excludes_synthetic_system_results() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-tool-stats-2";

        // Synthetic: success + duration 0 + no error + [SYSTEM] prefix.
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "web_search",
                success: true,
                duration_ms: 0,
                result: "[SYSTEM] You have already called web_search 3 times.",
                error: None,
                created_at: now - Duration::minutes(10),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "web_search",
                success: true,
                duration_ms: 0,
                result: "[SYSTEM] BLOCKED: repetitive tool call",
                error: None,
                created_at: now - Duration::minutes(9),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "web_search",
                success: true,
                duration_ms: 0,
                result: "[SYSTEM] Before executing tools, briefly state what you understand...",
                error: None,
                created_at: now - Duration::minutes(8),
            },
        )
        .await;

        // Real execution result
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "web_search",
                success: true,
                duration_ms: 120,
                result: "some results",
                error: None,
                created_at: now - Duration::minutes(7),
            },
        )
        .await;

        let stats = store
            .get_tool_stats("web_search", now - Duration::hours(24))
            .await
            .expect("tool stats");

        assert_eq!(stats.total_calls, 1);
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.avg_duration_ms, 120);
    }

    #[test]
    fn write_consistency_gate_can_be_tuned_with_custom_thresholds() {
        let report = WriteConsistencyReport {
            generated_at: Utc::now().to_rfc3339(),
            conversation_event_rows: 10,
            missing_message_id_events: 1,
            global_delta: 2,
            session_mismatch_count: 1,
            stale_task_starts: 0,
            top_session_drifts: Vec::new(),
        };

        let strict = report.evaluate_gate_with(WriteConsistencyThresholds {
            max_abs_global_delta: 0,
            max_session_mismatch_count: 0,
            max_stale_task_starts: 0,
            max_missing_message_id_events: 0,
        });
        assert!(!strict.passed);
        assert!(!strict.reasons.is_empty());

        let relaxed = report.evaluate_gate_with(WriteConsistencyThresholds {
            max_abs_global_delta: 2,
            max_session_mismatch_count: 1,
            max_stale_task_starts: 0,
            max_missing_message_id_events: 1,
        });
        assert!(relaxed.passed);
    }
}
