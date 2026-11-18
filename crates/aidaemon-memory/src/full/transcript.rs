//! A conversation log for hosts that keep none: the owner's turns, recorded
//! at the end of each run, in the memory database. It is what the
//! background jobs ([`MemoryJobs`](super::jobs::MemoryJobs)) read when
//! [`FullMemory::with_background`](super::FullMemory::with_background) runs
//! them for an SDK agent. A host with its own log (the daemon's events)
//! implements [`ActivityLog`] over it instead.

use std::collections::HashSet;

use aidaemon_core::{
    ChannelVisibility, Hook, HookContext, Message, MessageOrigin, Role, RunOutcome,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};

use crate::history::{
    ActivityLog, ConsolidationCursors, ConversationSource, HistoryTurn, LoggedTurn, MessageLog,
    TurnRole, UncapturedSessions,
};
use crate::records::EpisodeBound;

/// The owner's conversation turns, in two tables of their own
/// (`memory_transcript`, `memory_transcript_cursors`) created on open.
#[derive(Clone)]
pub struct TranscriptLog {
    pool: SqlitePool,
}

impl TranscriptLog {
    pub async fn open(pool: SqlitePool) -> anyhow::Result<Self> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_transcript (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT NOT NULL DEFAULT '[]',
                tool_name TEXT,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_memory_transcript_session
             ON memory_transcript(session_id, created_at)",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_transcript_cursors (
                session_id TEXT PRIMARY KEY,
                turn_id INTEGER NOT NULL
            )",
        )
        .execute(&pool)
        .await?;
        Ok(Self { pool })
    }

    /// Append one turn.
    pub async fn record(&self, session_id: &str, turn: &LoggedTurn) -> anyhow::Result<i64> {
        Ok(sqlx::query(
            "INSERT INTO memory_transcript
                (session_id, role, content, tool_calls, tool_name, created_at)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind(turn.role.as_str())
        .bind(&turn.content)
        .bind(serde_json::to_string(&turn.tool_calls)?)
        .bind(&turn.tool_name)
        .bind(turn.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?
        .last_insert_rowid())
    }

    /// A run-end hook that records the run's turns: its last user message
    /// and everything after it.
    pub fn recorder(&self) -> TranscriptRecorder {
        TranscriptRecorder { log: self.clone() }
    }

    async fn rows(&self, sql: &str, binds: &[Bind<'_>]) -> anyhow::Result<Vec<(i64, LoggedTurn)>> {
        let mut query = sqlx::query(sqlx::AssertSqlSafe(sql));
        for bind in binds {
            query = match bind {
                Bind::Text(text) => query.bind(*text),
                Bind::Int(int) => query.bind(*int),
            };
        }
        let mut turns = Vec::new();
        for row in query.fetch_all(&self.pool).await? {
            let role = match row.get::<String, _>("role").as_str() {
                "user" => TurnRole::User,
                "assistant" => TurnRole::Assistant,
                "tool" => TurnRole::Tool,
                _ => continue,
            };
            let created: String = row.get("created_at");
            turns.push((
                row.get("id"),
                LoggedTurn {
                    role,
                    content: row.get("content"),
                    tool_calls: serde_json::from_str(&row.get::<String, _>("tool_calls"))
                        .unwrap_or_default(),
                    tool_name: row.get("tool_name"),
                    created_at: parse_time(&created),
                },
            ));
        }
        Ok(turns)
    }
}

enum Bind<'a> {
    Text(&'a str),
    Int(i64),
}

const COLUMNS: &str = "id, role, content, tool_calls, tool_name, created_at";

fn parse_time(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .map(|at| at.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

/// The filter after a session's latest episode, and its bind: past the last
/// turn it covers, else past its end.
fn after_bound(bound: Option<&EpisodeBound>) -> (&'static str, Option<Bind<'_>>) {
    match bound {
        Some(EpisodeBound {
            end_event_id: Some(id),
            ..
        }) => (" AND id > ?", Some(Bind::Int(*id))),
        Some(EpisodeBound {
            end_time: Some(end),
            ..
        }) => (" AND created_at > ?", Some(Bind::Text(end))),
        _ => ("", None),
    }
}

/// Records each run's turns into a [`TranscriptLog`].
pub struct TranscriptRecorder {
    log: TranscriptLog,
}

#[async_trait]
impl Hook for TranscriptRecorder {
    fn name(&self) -> &'static str {
        "memory_transcript"
    }

    async fn on_run_end(&self, ctx: &HookContext<'_>, _outcome: &RunOutcome) {
        let Some(start) = ctx
            .messages
            .iter()
            .rposition(|m| m.role == Role::User && m.origin != Some(MessageOrigin::Harness))
        else {
            return;
        };
        let now = Utc::now();
        let names: std::collections::HashMap<&str, &str> = ctx
            .messages
            .iter()
            .flat_map(|m| &m.tool_calls)
            .map(|call| (call.id.as_str(), call.name.as_str()))
            .collect();
        for message in &ctx.messages[start..] {
            let Some(turn) = logged(message, &names, now) else {
                continue;
            };
            if let Err(error) = self.log.record(ctx.session_id, &turn).await {
                tracing::warn!(%error, "Failed to record a transcript turn");
                return;
            }
        }
    }
}

fn logged(
    message: &Message,
    names: &std::collections::HashMap<&str, &str>,
    at: DateTime<Utc>,
) -> Option<LoggedTurn> {
    let role = match message.role {
        Role::User if message.origin.is_none() || message.origin == Some(MessageOrigin::User) => {
            TurnRole::User
        }
        Role::Assistant => TurnRole::Assistant,
        Role::Tool => TurnRole::Tool,
        _ => return None,
    };
    let tool_name = (role == TurnRole::Tool)
        .then(|| {
            message
                .tool_call_id
                .as_deref()
                .and_then(|id| names.get(id))
                .map(|name| name.to_string())
        })
        .flatten();
    Some(LoggedTurn {
        role,
        content: Some(message.content.clone()),
        tool_calls: message
            .tool_calls
            .iter()
            .map(|call| call.name.clone())
            .collect(),
        tool_name,
        created_at: at,
    })
}

#[async_trait]
impl ConversationSource for TranscriptLog {
    async fn sessions_with_turns_before(
        &self,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<Vec<String>> {
        Ok(sqlx::query_scalar(
            "SELECT DISTINCT session_id FROM memory_transcript WHERE created_at < ?",
        )
        .bind(cutoff.to_rfc3339())
        .fetch_all(&self.pool)
        .await?)
    }

    /// Every recorded turn is the owner's.
    async fn is_memory_isolated(&self, _session_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn owner_only_since(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<bool> {
        let users: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_transcript
             WHERE session_id = ? AND role = 'user' AND id > ? AND created_at < ?",
        )
        .bind(session_id)
        .bind(after_id)
        .bind(cutoff.to_rfc3339())
        .fetch_one(&self.pool)
        .await?;
        Ok(users > 0)
    }

    async fn last_turn_id(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
    ) -> anyhow::Result<Option<i64>> {
        Ok(sqlx::query_scalar(
            "SELECT MAX(id) FROM memory_transcript
             WHERE session_id = ? AND id > ? AND created_at < ?",
        )
        .bind(session_id)
        .bind(after_id)
        .bind(cutoff.to_rfc3339())
        .fetch_one(&self.pool)
        .await?)
    }

    async fn turns(
        &self,
        session_id: &str,
        after_id: i64,
        cutoff: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<HistoryTurn>> {
        let cutoff = cutoff.to_rfc3339();
        let sql = format!(
            "SELECT {COLUMNS} FROM memory_transcript
             WHERE session_id = ? AND id > ? AND created_at < ?
             ORDER BY id ASC LIMIT ?"
        );
        let rows = self
            .rows(
                &sql,
                &[
                    Bind::Text(session_id),
                    Bind::Int(after_id),
                    Bind::Text(&cutoff),
                    Bind::Int(limit as i64),
                ],
            )
            .await?;
        Ok(rows
            .into_iter()
            .map(|(id, turn)| HistoryTurn {
                id,
                role: turn.role,
                content: turn.content,
                created_at: turn.created_at,
            })
            .collect())
    }

    async fn visibility(&self, _session_id: &str) -> Option<ChannelVisibility> {
        Some(ChannelVisibility::Private)
    }

    fn channel_id(&self, _session_id: &str) -> Option<String> {
        None
    }
}

#[async_trait]
impl ConsolidationCursors for TranscriptLog {
    async fn cursor(&self, session_id: &str) -> i64 {
        sqlx::query_scalar("SELECT turn_id FROM memory_transcript_cursors WHERE session_id = ?")
            .bind(session_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten()
            .unwrap_or(0)
    }

    async fn set_cursor(&self, session_id: &str, turn_id: i64) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO memory_transcript_cursors (session_id, turn_id) VALUES (?, ?)
             ON CONFLICT(session_id) DO UPDATE SET turn_id = excluded.turn_id",
        )
        .bind(session_id)
        .bind(turn_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

#[async_trait]
impl ActivityLog for TranscriptLog {
    async fn uncaptured_sessions(
        &self,
        episode_ends: &[(String, String)],
        which: UncapturedSessions,
    ) -> anyhow::Result<Vec<String>> {
        let idle_before = which.idle_before.map(|at| at.to_rfc3339());
        let sessions: Vec<String> =
            sqlx::query_scalar("SELECT DISTINCT session_id FROM memory_transcript")
                .fetch_all(&self.pool)
                .await?;
        let mut found = Vec::new();
        for session_id in sessions {
            let end = episode_ends
                .iter()
                .find(|(session, _)| *session == session_id)
                .map(|(_, end)| end.clone())
                .unwrap_or_default();
            let mut sql = String::from(
                "SELECT COUNT(*), MAX(created_at) FROM memory_transcript
                 WHERE session_id = ? AND created_at > ?",
            );
            if idle_before.is_some() {
                sql.push_str(" AND created_at < ?");
            }
            let mut query = sqlx::query_as::<_, (i64, Option<String>)>(sqlx::AssertSqlSafe(&*sql))
                .bind(&session_id)
                .bind(&end);
            if let Some(idle_before) = &idle_before {
                query = query.bind(idle_before);
            }
            let (count, last) = query.fetch_one(&self.pool).await?;
            let idle = match (&idle_before, &last) {
                (Some(before), Some(last)) => last < before,
                _ => true,
            };
            if count as usize >= which.min_turns && idle {
                found.push(session_id);
            }
        }
        Ok(found)
    }

    async fn owner_only_after(
        &self,
        session_id: &str,
        since: Option<&EpisodeBound>,
    ) -> anyhow::Result<bool> {
        let (filter, end) = after_bound(since);
        let sql = format!(
            "SELECT COUNT(*) FROM memory_transcript
             WHERE session_id = ? AND role = 'user'{filter}"
        );
        let mut query = sqlx::query_scalar::<_, i64>(sqlx::AssertSqlSafe(&*sql)).bind(session_id);
        query = match end {
            Some(Bind::Text(end)) => query.bind(end),
            Some(Bind::Int(id)) => query.bind(id),
            None => query,
        };
        Ok(query.fetch_one(&self.pool).await? > 0)
    }

    async fn turns_after(
        &self,
        session_id: &str,
        since: Option<&EpisodeBound>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>> {
        let (filter, end) = after_bound(since);
        let sql = format!(
            "SELECT {COLUMNS} FROM memory_transcript
             WHERE session_id = ?{filter} ORDER BY id ASC LIMIT ?"
        );
        let mut binds = vec![Bind::Text(session_id)];
        binds.extend(end);
        binds.push(Bind::Int(limit as i64));
        Ok(self
            .rows(&sql, &binds)
            .await?
            .into_iter()
            .map(|(_, turn)| turn)
            .collect())
    }

    async fn recent_turns(
        &self,
        since: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>> {
        let since = since.to_rfc3339();
        let sql = format!(
            "SELECT {COLUMNS} FROM memory_transcript
             WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?"
        );
        Ok(self
            .rows(&sql, &[Bind::Text(&since), Bind::Int(limit as i64)])
            .await?
            .into_iter()
            .map(|(_, turn)| turn)
            .collect())
    }

    async fn recent_owner_messages(
        &self,
        since: DateTime<Utc>,
        limit: usize,
    ) -> anyhow::Result<Vec<LoggedTurn>> {
        let since = since.to_rfc3339();
        let sql = format!(
            "SELECT {COLUMNS} FROM memory_transcript
             WHERE created_at >= ? AND role = 'user' ORDER BY created_at DESC LIMIT ?"
        );
        Ok(self
            .rows(&sql, &[Bind::Text(&since), Bind::Int(limit as i64)])
            .await?
            .into_iter()
            .map(|(_, turn)| turn)
            .collect())
    }
}

/// Episodes record the turns they cover by id.
#[async_trait]
impl MessageLog for TranscriptLog {
    async fn message_bounds(
        &self,
        session_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<(Option<i64>, Option<i64>)> {
        Ok(sqlx::query_as(
            "SELECT MIN(id), MAX(id) FROM memory_transcript
             WHERE session_id = ? AND role IN ('user', 'assistant')
               AND created_at >= ? AND created_at <= ?",
        )
        .bind(session_id)
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_one(&self.pool)
        .await?)
    }

    async fn retained(&self, ids: &[i64]) -> anyhow::Result<HashSet<i64>> {
        let mut kept = HashSet::new();
        for chunk in ids.chunks(500) {
            let placeholders = vec!["?"; chunk.len()].join(", ");
            let sql = format!("SELECT id FROM memory_transcript WHERE id IN ({placeholders})");
            let mut query = sqlx::query_scalar::<_, i64>(sqlx::AssertSqlSafe(&*sql));
            for id in chunk {
                query = query.bind(id);
            }
            kept.extend(query.fetch_all(&self.pool).await?);
        }
        Ok(kept)
    }
}
