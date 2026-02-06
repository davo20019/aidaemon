//! Event store implementation using SQLite.
//!
//! The EventStore provides CRUD operations for events, with support for:
//! - Efficient querying by session, time window, and event type
//! - Conversation history retrieval (replacing messages table)
//! - Consolidation tracking and pruning

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use sqlx::{Row, SqlitePool};
use tracing::{info, warn};

use super::{Event, EventType};
use crate::traits::{Message, ToolCall};

/// The event store backed by SQLite.
pub struct EventStore {
    pool: SqlitePool,
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
        // Create events table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT,
                task_id TEXT,
                tool_name TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create indexes for efficient queries
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_events_session_time
             ON events(session_id, created_at DESC)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_events_task
             ON events(task_id) WHERE task_id IS NOT NULL",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_events_consolidation
             ON events(consolidated_at) WHERE consolidated_at IS NULL",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_events_prune
             ON events(created_at) WHERE consolidated_at IS NOT NULL",
        )
        .execute(&self.pool)
        .await?;

        info!("Events table migration complete");
        Ok(())
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

        let result = sqlx::query(
            "DELETE FROM events WHERE consolidated_at IS NOT NULL AND created_at < ?",
        )
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
    // Read Operations - Conversation History (replaces messages table)
    // =========================================================================

    /// Get conversation history for a session (for LLM context)
    /// Returns events in the format needed for provider messages
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

        // Convert events to Message format (for backwards compatibility)
        let mut messages = Vec::new();
        for event in events.into_iter().rev() {
            // Reverse to chronological
            match event.event_type {
                EventType::UserMessage => {
                    if let Ok(data) = event.parse_data::<super::UserMessageData>() {
                        messages.push(Message {
                            id: event.id.to_string(),
                            session_id: event.session_id.clone(),
                            role: "user".to_string(),
                            content: Some(data.content),
                            tool_call_id: None,
                            tool_name: None,
                            tool_calls_json: None,
                            created_at: event.created_at,
                            importance: 0.5,
                            embedding: None,
                        });
                    }
                }
                EventType::AssistantResponse => {
                    if let Ok(data) = event.parse_data::<super::AssistantResponseData>() {
                        let tool_calls_json = data.tool_calls.as_ref().map(|calls| {
                            let tool_calls: Vec<ToolCall> = calls
                                .iter()
                                .map(|tc| ToolCall {
                                    id: tc.id.clone(),
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.to_string(),
                                    extra_content: None,
                                })
                                .collect();
                            serde_json::to_string(&tool_calls).unwrap_or_default()
                        });

                        messages.push(Message {
                            id: event.id.to_string(),
                            session_id: event.session_id.clone(),
                            role: "assistant".to_string(),
                            content: data.content,
                            tool_call_id: None,
                            tool_name: None,
                            tool_calls_json,
                            created_at: event.created_at,
                            importance: 0.5,
                            embedding: None,
                        });
                    }
                }
                EventType::ToolResult => {
                    if let Ok(data) = event.parse_data::<super::ToolResultData>() {
                        messages.push(Message {
                            id: event.id.to_string(),
                            session_id: event.session_id.clone(),
                            role: "tool".to_string(),
                            content: Some(data.result),
                            tool_call_id: Some(data.tool_call_id),
                            tool_name: Some(data.name),
                            tool_calls_json: None,
                            created_at: event.created_at,
                            importance: 0.5,
                            embedding: None,
                        });
                    }
                }
                _ => {}
            }
        }

        // Apply limit - keep only the last `limit` messages
        // IMPORTANT: Always preserve the first user message (anchor) to satisfy
        // provider ordering requirements (Gemini requires assistant+tool calls
        // to follow a user message)
        if messages.len() > limit {
            // Find the first user message (anchor)
            let anchor_idx = messages.iter().position(|m| m.role == "user");

            if let Some(anchor) = anchor_idx {
                // Keep the anchor + last (limit - 1) messages
                let skip_count = messages.len() - limit;

                if skip_count > anchor {
                    // We would skip past the anchor - preserve it
                    let anchor_msg = messages[anchor].clone();
                    let remaining: Vec<_> = messages.into_iter().skip(skip_count).collect();

                    // Only prepend anchor if it's not already in remaining
                    if remaining.first().map(|m| m.role.as_str()) != Some("user") {
                        let mut result = vec![anchor_msg];
                        // Take limit - 1 from remaining to stay within limit
                        result.extend(remaining.into_iter().take(limit - 1));
                        messages = result;
                    } else {
                        messages = remaining;
                    }
                } else {
                    // Normal case - anchor is within the kept range
                    messages = messages.into_iter().skip(skip_count).collect();
                }
            } else {
                // No user message found - just truncate normally
                let skip_count = messages.len() - limit;
                messages = messages.into_iter().skip(skip_count).collect();
            }
        }

        Ok(messages)
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

    /// Get recent tool calls for a session
    pub async fn get_recent_tool_calls(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        self.query_events_by_types(session_id, &[EventType::ToolCall, EventType::ToolResult], limit)
            .await
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
