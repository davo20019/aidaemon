//! The daemon's conversation history (its event log) as memory reads it:
//! sessions, turns, who spoke, audience, memory isolation and the
//! consolidation cursors.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use tracing::warn;

use crate::traits::StateStore;
use crate::types::{ChannelVisibility, UserRole};

/// Reads the event log for memory consolidation.
#[derive(Clone)]
pub(crate) struct EventHistory {
    pool: SqlitePool,
    /// Where consolidation cursors are kept (settings).
    state: Option<Arc<dyn StateStore>>,
}

impl EventHistory {
    pub(crate) fn new(pool: SqlitePool, state: Option<Arc<dyn StateStore>>) -> Self {
        Self { pool, state }
    }

    pub(crate) fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub(crate) async fn session_visibility_from_events(
        &self,
        session_id: &str,
    ) -> Option<ChannelVisibility> {
        let row = sqlx::query(
            r#"
            SELECT data
            FROM events
            WHERE session_id = ?
              AND event_type = 'user_message'
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await;

        let row = match row {
            Ok(r) => r,
            Err(e) => {
                if e.to_string().contains("no such table: events") {
                    return None;
                }
                warn!(session_id, error = %e, "Failed to fetch session visibility from events");
                return None;
            }
        }?;

        let raw: String = row.get("data");
        let val: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let vis = val.get("channel_visibility").and_then(|v| v.as_str())?;
        Some(ChannelVisibility::from_str_lossy(vis))
    }

    pub(crate) fn parse_event_user_role(data: &serde_json::Value) -> Option<UserRole> {
        match data.get("user_role").and_then(|v| v.as_str()) {
            Some("owner") => Some(UserRole::Owner),
            Some("guest") => Some(UserRole::Guest),
            Some("public") => Some(UserRole::Public),
            _ => None,
        }
    }

    pub(crate) async fn session_is_memory_isolated(
        &self,
        session_id: &str,
    ) -> anyhow::Result<bool> {
        if crate::events::is_memory_isolated_worker_session(session_id) {
            return Ok(true);
        }
        crate::events::session_has_mandate_execution_provenance(&self.pool, session_id).await
    }

    pub(crate) async fn fact_consolidation_batch_is_owner_only(
        &self,
        session_id: &str,
        cutoff: &str,
        after_event_id: i64,
    ) -> anyhow::Result<bool> {
        if self.session_is_memory_isolated(session_id).await? {
            return Ok(false);
        }
        let rows = sqlx::query(
            r#"
            SELECT data
            FROM events
            WHERE session_id = ?
              AND event_type = 'user_message'
              AND id > ?
              AND created_at < ?
            ORDER BY id ASC
            "#,
        )
        .bind(session_id)
        .bind(after_event_id)
        .bind(cutoff)
        .fetch_all(&self.pool)
        .await?;

        let mut saw_owner_message = false;
        for row in rows {
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(val) => val,
                Err(_) => return Ok(false),
            };

            match Self::parse_event_user_role(&data) {
                Some(role) if role.can_persist_owner_memory() => {
                    saw_owner_message = true;
                }
                _ => return Ok(false),
            }
        }

        Ok(saw_owner_message)
    }

    /// Whether every user message after `last_episode` (the session's
    /// latest episode; `None`: the whole session) came from the owner, and
    /// there was at least one.
    pub(crate) async fn session_since_last_episode_is_owner_only(
        &self,
        session_id: &str,
        last_episode: Option<&aidaemon_memory::records::EpisodeBound>,
    ) -> anyhow::Result<bool> {
        if self.session_is_memory_isolated(session_id).await? {
            return Ok(false);
        }
        let rows = if let Some(end_event_id) = last_episode.and_then(|bound| bound.end_event_id) {
            sqlx::query(
                "SELECT data FROM events
                 WHERE session_id = ?
                   AND event_type = 'user_message'
                   AND id > ?
                 ORDER BY id ASC",
            )
            .bind(session_id)
            .bind(end_event_id)
            .fetch_all(&self.pool)
            .await?
        } else if let Some(since) = last_episode.and_then(|bound| bound.end_time.as_ref()) {
            sqlx::query(
                "SELECT data FROM events
                 WHERE session_id = ?
                   AND event_type = 'user_message'
                   AND created_at > ?
                 ORDER BY created_at ASC",
            )
            .bind(session_id)
            .bind(since)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT data FROM events
                 WHERE session_id = ?
                   AND event_type = 'user_message'
                 ORDER BY created_at ASC",
            )
            .bind(session_id)
            .fetch_all(&self.pool)
            .await?
        };

        let mut saw_owner_message = false;
        for row in rows {
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(val) => val,
                Err(_) => return Ok(false),
            };

            match Self::parse_event_user_role(&data) {
                Some(role) if role.can_persist_owner_memory() => {
                    saw_owner_message = true;
                }
                _ => return Ok(false),
            }
        }

        Ok(saw_owner_message)
    }

    pub(crate) fn fact_consolidation_cursor_key(session_id: &str) -> String {
        format!("memory_fact_consolidation_cursor:{}", session_id)
    }

    pub(crate) async fn get_fact_consolidation_cursor(&self, session_id: &str) -> i64 {
        let Some(state) = &self.state else {
            return 0;
        };
        let key = Self::fact_consolidation_cursor_key(session_id);
        match state.get_setting(&key).await {
            Ok(Some(raw)) => raw.parse::<i64>().unwrap_or(0),
            _ => 0,
        }
    }

    pub(crate) async fn set_fact_consolidation_cursor(
        &self,
        session_id: &str,
        event_id: i64,
    ) -> anyhow::Result<()> {
        let Some(state) = &self.state else {
            return Ok(());
        };
        let key = Self::fact_consolidation_cursor_key(session_id);
        state.set_setting(&key, &event_id.to_string()).await
    }

    pub(crate) async fn list_candidate_sessions_for_fact_consolidation(
        &self,
        cutoff: &str,
    ) -> anyhow::Result<Vec<String>> {
        let rows: Vec<String> = sqlx::query_scalar(
            r#"
            SELECT DISTINCT session_id
            FROM events
            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
              AND created_at < ?
            "#,
        )
        .bind(cutoff)
        .fetch_all(&self.pool)
        .await?;
        self.exclude_memory_isolated_sessions(rows).await
    }

    pub(crate) async fn exclude_memory_isolated_sessions(
        &self,
        candidates: Vec<String>,
    ) -> anyhow::Result<Vec<String>> {
        let mut eligible = Vec::with_capacity(candidates.len());
        for session_id in candidates {
            if !self.session_is_memory_isolated(&session_id).await? {
                eligible.push(session_id);
            }
        }
        Ok(eligible)
    }

    pub(crate) async fn memory_suppressed_tasks_for_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<std::collections::HashSet<String>> {
        Ok(sqlx::query_scalar(
            "SELECT DISTINCT task_id
             FROM events
             WHERE session_id = ? AND event_type = 'memory_policy_compiled'
               AND task_id IS NOT NULL
               AND json_extract(data, '$.access') = 'suppressed'",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .collect())
    }

    pub(crate) async fn all_memory_suppressed_tasks(
        &self,
    ) -> anyhow::Result<std::collections::HashSet<String>> {
        Ok(sqlx::query_scalar(
            "SELECT DISTINCT task_id
             FROM events
             WHERE event_type = 'memory_policy_compiled' AND task_id IS NOT NULL
               AND json_extract(data, '$.access') = 'suppressed'",
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .collect())
    }

    pub(crate) async fn fetch_fact_consolidation_messages(
        &self,
        session_id: &str,
        cutoff: &str,
        after_event_id: i64,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::events::ConversationTurn>> {
        let suppressed_tasks = self.memory_suppressed_tasks_for_session(session_id).await?;
        let rows = sqlx::query(
            r#"
            SELECT id, event_type, data, created_at, task_id
            FROM events
            WHERE session_id = ?
              AND event_type IN ('user_message', 'assistant_response', 'tool_result')
              AND id > ?
              AND created_at < ?
            ORDER BY id ASC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(after_event_id)
        .bind(cutoff)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            if row
                .try_get::<Option<String>, _>("task_id")?
                .is_some_and(|task_id| suppressed_tasks.contains(&task_id))
            {
                continue;
            }
            let event_id: i64 = row.get("id");
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        session_id,
                        event_id,
                        event_type = %event_type,
                        error = %e,
                        "Skipping malformed event payload during memory consolidation"
                    );
                    continue;
                }
            };

            if let Some(turn) =
                crate::events::turn_from_event(event_id, session_id, &event_type, &data, created_at)
            {
                out.push(turn);
            }
        }
        Ok(out)
    }
}

#[async_trait::async_trait]
impl aidaemon_memory::history::ConversationSource for EventHistory {
    async fn sessions_with_turns_before(
        &self,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<Vec<String>> {
        self.list_candidate_sessions_for_fact_consolidation(&cutoff.to_rfc3339())
            .await
    }

    async fn is_memory_isolated(&self, session_id: &str) -> anyhow::Result<bool> {
        self.session_is_memory_isolated(session_id).await
    }

    async fn owner_only_since(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<bool> {
        self.fact_consolidation_batch_is_owner_only(session_id, &cutoff.to_rfc3339(), after_id)
            .await
    }

    async fn last_turn_id(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<Option<i64>> {
        Ok(sqlx::query_scalar(
            r#"
            SELECT MAX(id)
            FROM events
            WHERE session_id = ?
              AND event_type IN ('user_message', 'assistant_response', 'tool_result')
              AND id > ?
              AND created_at < ?
            "#,
        )
        .bind(session_id)
        .bind(after_id)
        .bind(cutoff.to_rfc3339())
        .fetch_optional(&self.pool)
        .await?
        .flatten())
    }

    async fn turns(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<aidaemon_memory::history::HistoryTurn>> {
        use aidaemon_memory::history::{HistoryTurn, TurnRole};
        let turns = self
            .fetch_fact_consolidation_messages(session_id, &cutoff.to_rfc3339(), after_id, limit)
            .await?;
        Ok(turns
            .into_iter()
            .map(|turn| HistoryTurn {
                id: turn.event_id,
                role: match turn.role {
                    crate::events::ConversationTurnRole::User => TurnRole::User,
                    crate::events::ConversationTurnRole::Assistant => TurnRole::Assistant,
                    crate::events::ConversationTurnRole::Tool => TurnRole::Tool,
                },
                content: turn.content,
                created_at: turn.created_at,
            })
            .collect())
    }

    async fn visibility(&self, session_id: &str) -> Option<ChannelVisibility> {
        self.session_visibility_from_events(session_id).await
    }

    fn channel_id(&self, session_id: &str) -> Option<String> {
        crate::memory::derive_channel_id_from_session(session_id)
    }
}

#[async_trait::async_trait]
impl aidaemon_memory::history::ConsolidationCursors for EventHistory {
    async fn cursor(&self, session_id: &str) -> i64 {
        self.get_fact_consolidation_cursor(session_id).await
    }

    async fn set_cursor(&self, session_id: &str, turn_id: i64) -> anyhow::Result<()> {
        self.set_fact_consolidation_cursor(session_id, turn_id)
            .await
    }
}

impl EventHistory {
    /// The user message event `event_id`, if memory may learn from it: its
    /// task's memory policy allowed access and never suppressed it (a message
    /// outside any task is always learnable).
    pub(crate) async fn learnable_user_message(
        &self,
        event_id: i64,
    ) -> anyhow::Result<Option<aidaemon_memory::history::LoggedMessage>> {
        let Some(row) = sqlx::query(
            "SELECT e.id, e.session_id, e.data, e.created_at FROM events e
             WHERE e.id = ? AND e.event_type = 'user_message'
               AND (e.task_id IS NULL OR EXISTS (
                   SELECT 1 FROM events policy
                   WHERE policy.task_id = e.task_id
                     AND policy.event_type = 'memory_policy_compiled'
                     AND json_extract(policy.data, '$.access') = 'allowed'
               ) AND NOT EXISTS (
                   SELECT 1 FROM events policy
                   WHERE policy.task_id = e.task_id
                     AND policy.event_type = 'memory_policy_compiled'
                     AND json_extract(policy.data, '$.access') = 'suppressed'
               ))",
        )
        .bind(event_id)
        .fetch_optional(&self.pool)
        .await?
        else {
            return Ok(None);
        };
        let data: String = row.get("data");
        let content = serde_json::from_str::<serde_json::Value>(&data)
            .ok()
            .and_then(|value| {
                value
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or_default();
        Ok(Some(aidaemon_memory::history::LoggedMessage {
            id: event_id,
            session_id: row.get("session_id"),
            content,
            created_at: row.get("created_at"),
        }))
    }
}

/// The event log as the memory store links into it.
#[async_trait::async_trait]
impl aidaemon_memory::history::MessageLog for EventHistory {
    async fn message_bounds(
        &self,
        session_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<(Option<i64>, Option<i64>)> {
        let row = sqlx::query(
            "SELECT MIN(id) AS start_event_id, MAX(id) AS end_event_id
             FROM events WHERE session_id = ?
               AND event_type IN ('user_message', 'assistant_response')
               AND created_at >= ? AND created_at <= ?",
        )
        .bind(session_id)
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_one(&self.pool)
        .await;
        match row {
            Ok(row) => Ok((
                row.try_get("start_event_id").unwrap_or(None),
                row.try_get("end_event_id").unwrap_or(None),
            )),
            Err(error) if error.to_string().contains("no such table") => Ok((None, None)),
            Err(error) => Err(error.into()),
        }
    }

    async fn retained(&self, ids: &[i64]) -> anyhow::Result<std::collections::HashSet<i64>> {
        if ids.is_empty() {
            return Ok(Default::default());
        }
        let placeholders = vec!["?"; ids.len()].join(", ");
        let sql = format!("SELECT id FROM events WHERE id IN ({placeholders})");
        let mut query = sqlx::query_scalar::<_, i64>(sqlx::AssertSqlSafe(&*sql));
        for id in ids {
            query = query.bind(id);
        }
        Ok(query.fetch_all(&self.pool).await?.into_iter().collect())
    }
}
