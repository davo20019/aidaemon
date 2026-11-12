//! The daemon's event log as the full memory system's background jobs read
//! it ([`ActivityLog`]): sessions with turns no episode covers, those turns,
//! and the recent activity patterns are learned from.

use std::collections::HashMap;

use aidaemon_memory::history::{ActivityLog, LoggedTurn, TurnRole, UncapturedSessions};
use aidaemon_memory::records::EpisodeBound;
use chrono::{DateTime, Utc};
use sqlx::Row;

use super::history_source::EventHistory;
use crate::traits::{Message, MessageAnnotation};
use crate::types::UserRole;

impl EventHistory {
    /// Fetch messages after the last stable canonical event bound. Legacy
    /// episodes without bounds fall back to end_time.
    /// If no episode exists, returns all session messages up to `limit`.
    pub(crate) async fn session_messages_after(
        &self,
        session_id: &str,
        last_episode: Option<&EpisodeBound>,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let suppressed_tasks = self.memory_suppressed_tasks_for_session(session_id).await?;

        let rows = if let Some(end_event_id) = last_episode.and_then(|bound| bound.end_event_id) {
            sqlx::query(
                "SELECT id, session_id, event_type, data, created_at, task_id
                 FROM events
                 WHERE session_id = ?
                   AND event_type IN ('user_message', 'assistant_response', 'tool_result')
                   AND id > ?
                 ORDER BY id ASC
                 LIMIT ?",
            )
            .bind(session_id)
            .bind(end_event_id)
            .bind(limit as i64)
            .fetch_all(self.pool())
            .await?
        } else if let Some(since) = last_episode.and_then(|bound| bound.end_time.as_ref()) {
            sqlx::query(
                "SELECT id, session_id, event_type, data, created_at, task_id
                 FROM events
                 WHERE session_id = ?
                   AND event_type IN ('user_message', 'assistant_response', 'tool_result')
                   AND created_at > ?
                 ORDER BY created_at ASC
                 LIMIT ?",
            )
            .bind(session_id)
            .bind(since)
            .bind(limit as i64)
            .fetch_all(self.pool())
            .await?
        } else {
            sqlx::query(
                "SELECT id, session_id, event_type, data, created_at, task_id
                 FROM events
                 WHERE session_id = ?
                   AND event_type IN ('user_message', 'assistant_response', 'tool_result')
                 ORDER BY created_at ASC
                 LIMIT ?",
            )
            .bind(session_id)
            .bind(limit as i64)
            .fetch_all(self.pool())
            .await?
        };

        let mut messages = Vec::with_capacity(rows.len());
        for row in rows {
            if row
                .try_get::<Option<String>, _>("task_id")?
                .is_some_and(|task_id| suppressed_tasks.contains(&task_id))
            {
                continue;
            }
            let event_id: i64 = row.get("id");
            let row_session_id: String = row.get("session_id");
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(turn) = crate::events::turn_from_event(
                event_id,
                &row_session_id,
                &event_type,
                &data,
                created_at,
            ) {
                let mut msg = turn.clone().into_message();
                if msg
                    .effective_annotations()
                    .contains(&MessageAnnotation::InternalContinuation)
                {
                    continue;
                }
                msg.importance =
                    aidaemon_memory::turn_importance(turn.role.as_str(), turn.content.as_deref());
                messages.push(msg);
            }
        }

        Ok(messages)
    }

    /// Fetch messages from all sessions since a given time.
    pub(crate) async fn fetch_messages_since(
        &self,
        since: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let suppressed_tasks = self.all_memory_suppressed_tasks().await?;
        let rows = sqlx::query(
            "SELECT id, session_id, event_type, data, created_at, task_id
             FROM events
             WHERE created_at >= ?
               AND event_type IN ('user_message', 'assistant_response', 'tool_result')
             ORDER BY created_at DESC
             LIMIT ?",
        )
        .bind(since)
        .bind(limit as i64)
        .fetch_all(self.pool())
        .await?;

        let mut messages = Vec::with_capacity(rows.len());
        let mut isolation_by_session: HashMap<String, bool> = HashMap::new();
        for row in rows {
            if row
                .try_get::<Option<String>, _>("task_id")?
                .is_some_and(|task_id| suppressed_tasks.contains(&task_id))
            {
                continue;
            }
            let event_id: i64 = row.get("id");
            let session_id: String = row.get("session_id");
            let memory_isolated = match isolation_by_session.get(&session_id) {
                Some(isolated) => *isolated,
                None => {
                    let isolated = self.session_is_memory_isolated(&session_id).await?;
                    isolation_by_session.insert(session_id.clone(), isolated);
                    isolated
                }
            };
            if memory_isolated {
                continue;
            }
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(turn) = crate::events::turn_from_event(
                event_id,
                &session_id,
                &event_type,
                &data,
                created_at,
            ) {
                let mut msg = turn.clone().into_message();
                if msg
                    .effective_annotations()
                    .contains(&MessageAnnotation::InternalContinuation)
                {
                    continue;
                }
                msg.importance =
                    aidaemon_memory::turn_importance(turn.role.as_str(), turn.content.as_deref());
                messages.push(msg);
            }
        }

        Ok(messages)
    }

    pub(crate) async fn fetch_owner_user_messages_since(
        &self,
        since: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let suppressed_tasks = self.all_memory_suppressed_tasks().await?;
        let rows = sqlx::query(
            "SELECT id, session_id, event_type, data, created_at, task_id
             FROM events
             WHERE created_at >= ?
               AND event_type = 'user_message'
             ORDER BY created_at DESC
             LIMIT ?",
        )
        .bind(since)
        .bind(limit as i64)
        .fetch_all(self.pool())
        .await?;

        let mut messages = Vec::with_capacity(rows.len());
        let mut isolation_by_session: HashMap<String, bool> = HashMap::new();
        for row in rows {
            if row
                .try_get::<Option<String>, _>("task_id")?
                .is_some_and(|task_id| suppressed_tasks.contains(&task_id))
            {
                continue;
            }
            let event_id: i64 = row.get("id");
            let session_id: String = row.get("session_id");
            let memory_isolated = match isolation_by_session.get(&session_id) {
                Some(isolated) => *isolated,
                None => {
                    let isolated = self.session_is_memory_isolated(&session_id).await?;
                    isolation_by_session.insert(session_id.clone(), isolated);
                    isolated
                }
            };
            if memory_isolated {
                continue;
            }
            let event_type: String = row.get("event_type");
            let created_str: String = row.get("created_at");
            let created_at = DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if !Self::parse_event_user_role(&data).is_some_and(UserRole::can_persist_owner_memory) {
                continue;
            }

            if let Some(turn) = crate::events::turn_from_event(
                event_id,
                &session_id,
                &event_type,
                &data,
                created_at,
            ) {
                let mut msg = turn.clone().into_message();
                if msg
                    .effective_annotations()
                    .contains(&MessageAnnotation::InternalContinuation)
                {
                    continue;
                }
                msg.importance =
                    aidaemon_memory::turn_importance(turn.role.as_str(), turn.content.as_deref());
                messages.push(msg);
            }
        }

        Ok(messages)
    }
}

/// Sessions whose conversation events since their latest episode end pass
/// `having`. `?1` is the JSON list of `[session_id, latest_end]` pairs from
/// `latest_episode_ends_json`; `filter` narrows the events further.
///
/// The list has no index, so it is never the inner side of a join over
/// events: sessions without an episode are found with `NOT IN` (SQLite builds
/// a lookup table for it), and sessions with one are driven from the list into
/// `idx_events_session_time`. A `LEFT JOIN` against `json_each` re-read the
/// whole list for every event row, which grew with every session and episode.
pub(crate) fn uncaptured_sessions_sql(filter: &str, having: &str) -> String {
    format!(
        "SELECT ev.session_id
         FROM events ev
         WHERE ev.event_type IN ('user_message', 'assistant_response', 'tool_result')
           AND ev.session_id NOT IN (
               SELECT json_extract(value, '$[0]') FROM json_each(?1)
           )
           {filter}
         GROUP BY ev.session_id
         {having}
         UNION ALL
         SELECT ep.session_id
         FROM (
             SELECT json_extract(value, '$[0]') AS session_id,
                    json_extract(value, '$[1]') AS latest_end
             FROM json_each(?1)
         ) ep
         JOIN events ev
           ON ev.session_id = ep.session_id AND ev.created_at > ep.latest_end
         WHERE ev.event_type IN ('user_message', 'assistant_response', 'tool_result')
           {filter}
         GROUP BY ep.session_id
         {having}"
    )
}

/// A message as the jobs read it.
fn logged_turn(message: Message) -> Option<LoggedTurn> {
    let role = match message.role.as_str() {
        "user" => TurnRole::User,
        "assistant" => TurnRole::Assistant,
        "tool" => TurnRole::Tool,
        _ => return None,
    };
    let tool_calls = message
        .tool_calls_json
        .as_deref()
        .and_then(|json| serde_json::from_str::<Vec<serde_json::Value>>(json).ok())
        .into_iter()
        .flatten()
        .filter_map(|call| {
            call.get("name")
                .and_then(|n| n.as_str())
                .map(str::to_string)
        })
        .collect();
    Some(LoggedTurn {
        role,
        content: message.content,
        tool_calls,
        tool_name: message.tool_name,
        created_at: message.created_at,
    })
}

fn logged_turns(messages: Vec<Message>) -> Vec<LoggedTurn> {
    messages.into_iter().filter_map(logged_turn).collect()
}

#[async_trait::async_trait]
impl ActivityLog for EventHistory {
    async fn uncaptured_sessions(
        &self,
        episode_ends: &[(String, String)],
        which: UncapturedSessions,
    ) -> anyhow::Result<Vec<String>> {
        let episode_ends = serde_json::to_string(episode_ends)?;
        let min = which.min_turns;
        Ok(match which.idle_before {
            Some(idle_before) => {
                sqlx::query_scalar(sqlx::AssertSqlSafe(&*uncaptured_sessions_sql(
                    "AND ev.created_at < ?2",
                    &format!("HAVING COUNT(ev.id) >= {min} AND MAX(ev.created_at) < ?2"),
                )))
                .bind(&episode_ends)
                .bind(idle_before.to_rfc3339())
                .fetch_all(self.pool())
                .await?
            }
            None => {
                sqlx::query_scalar(sqlx::AssertSqlSafe(&*uncaptured_sessions_sql(
                    "",
                    &format!("HAVING COUNT(ev.id) >= {min}"),
                )))
                .bind(&episode_ends)
                .fetch_all(self.pool())
                .await?
            }
        })
    }

    async fn owner_only_after(
        &self,
        session_id: &str,
        since: Option<&EpisodeBound>,
    ) -> anyhow::Result<bool> {
        self.session_since_last_episode_is_owner_only(session_id, since)
            .await
    }

    async fn turns_after(
        &self,
        session_id: &str,
        since: Option<&EpisodeBound>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>> {
        Ok(logged_turns(
            self.session_messages_after(session_id, since, limit)
                .await?,
        ))
    }

    async fn recent_turns(
        &self,
        since: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>> {
        Ok(logged_turns(
            self.fetch_messages_since(&since.to_rfc3339(), limit)
                .await?,
        ))
    }

    async fn recent_owner_messages(
        &self,
        since: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>> {
        Ok(logged_turns(
            self.fetch_owner_user_messages_since(&since.to_rfc3339(), limit)
                .await?,
        ))
    }
}
