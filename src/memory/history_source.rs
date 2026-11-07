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

    pub(crate) async fn session_since_last_episode_is_owner_only(
        &self,
        session_id: &str,
    ) -> anyhow::Result<bool> {
        if self.session_is_memory_isolated(session_id).await? {
            return Ok(false);
        }
        let last_episode = sqlx::query(
            "SELECT end_event_id, end_time FROM episodes
             WHERE session_id = ?
             ORDER BY COALESCE(end_event_id, 0) DESC, end_time DESC LIMIT 1",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        let rows = if let Some(end_event_id) = last_episode
            .as_ref()
            .and_then(|row| row.try_get::<Option<i64>, _>("end_event_id").ok())
            .flatten()
        {
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
        } else if let Some(since) = last_episode
            .as_ref()
            .and_then(|row| row.try_get::<String, _>("end_time").ok())
        {
            sqlx::query(
                "SELECT data FROM events
                 WHERE session_id = ?
                   AND event_type = 'user_message'
                   AND created_at > ?
                 ORDER BY created_at ASC",
            )
            .bind(session_id)
            .bind(&since)
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
