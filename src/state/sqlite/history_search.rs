//! Repairable full-text projection over canonical user/assistant events.
//!
//! `events` is authoritative. There are deliberately no triggers on it: a
//! missing or corrupt FTS table must never roll back a canonical message.

use std::collections::HashSet;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use sqlx::{QueryBuilder, Row, Sqlite, SqlitePool, Transaction};

use crate::types::{ChannelVisibility, UserRole};

const BACKFILL_BATCH: i64 = 500;

#[derive(Debug, Clone)]
pub struct HistoryScope {
    pub session_id: String,
    pub channel_id: Option<String>,
    pub visibility: ChannelVisibility,
    pub user_role: UserRole,
    pub trusted: bool,
    pub include_subagents: bool,
    pub session_filter: Option<String>,
    pub task_filter: Option<String>,
    pub snapshot_max_event_id: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryMessage {
    pub event_id: i64,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    pub role: String,
    pub content: String,
    pub created_at: String,
    pub source_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskBookends {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective: Option<HistoryMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_objective: Option<String>,
    pub objective_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<HistoryMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_resolution: Option<String>,
    pub resolution_source: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ProjectionStats {
    pub projected: u64,
    pub orphans_removed: u64,
    pub fts_rebuilt: bool,
    pub episodes_repaired: u64,
    pub pending: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryCoverage {
    pub canonical_messages: i64,
    pub indexed_messages: i64,
    pub pending_messages: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oldest_indexed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest_indexed_at: Option<String>,
}

/// Exact-history projection operations needed by the search tool. The tool
/// depends on this port rather than SQLite or a raw connection pool.
#[async_trait]
pub(crate) trait HistorySearchStore: crate::traits::EpisodeStore {
    async fn history_snapshot_max_event_id(&self) -> anyhow::Result<i64>;
    async fn history_coverage(&self) -> anyhow::Result<HistoryCoverage>;
    async fn repair_history_projection(
        &self,
        max_batches: usize,
    ) -> anyhow::Result<ProjectionStats>;
    async fn search_history(
        &self,
        query: &str,
        scope: &HistoryScope,
        limit: usize,
        semantic_sessions: &HashSet<String>,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_context(
        &self,
        event_id: i64,
        radius: usize,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_page(
        &self,
        anchor: i64,
        older: bool,
        scope: &HistoryScope,
        limit: usize,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_task_bookends(
        &self,
        task_id: Option<&str>,
        session_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<TaskBookends>;
}

#[async_trait]
impl HistorySearchStore for crate::state::SqliteStateStore {
    async fn history_snapshot_max_event_id(&self) -> anyhow::Result<i64> {
        snapshot_max_event_id(&self.pool()).await
    }

    async fn history_coverage(&self) -> anyhow::Result<HistoryCoverage> {
        coverage(&self.pool()).await
    }

    async fn repair_history_projection(
        &self,
        max_batches: usize,
    ) -> anyhow::Result<ProjectionStats> {
        repair_and_backfill(&self.pool(), max_batches).await
    }

    async fn search_history(
        &self,
        query: &str,
        scope: &HistoryScope,
        limit: usize,
        semantic_sessions: &HashSet<String>,
    ) -> anyhow::Result<Vec<HistoryMessage>> {
        search(&self.pool(), query, scope, limit, semantic_sessions).await
    }

    async fn history_context(
        &self,
        event_id: i64,
        radius: usize,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryMessage>> {
        context(&self.pool(), event_id, radius, scope).await
    }

    async fn history_page(
        &self,
        anchor: i64,
        older: bool,
        scope: &HistoryScope,
        limit: usize,
    ) -> anyhow::Result<Vec<HistoryMessage>> {
        page(&self.pool(), anchor, older, scope, limit).await
    }

    async fn history_task_bookends(
        &self,
        task_id: Option<&str>,
        session_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<TaskBookends> {
        task_bookends(&self.pool(), task_id, session_id, scope).await
    }
}

pub(crate) async fn migrate_history_search(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS history_message_index (
            event_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            task_id TEXT,
            turn_id TEXT,
            message_id TEXT,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            created_at TEXT NOT NULL,
            channel_id TEXT,
            channel_visibility TEXT NOT NULL DEFAULT 'internal',
            user_role TEXT,
            source_kind TEXT NOT NULL DEFAULT 'root'
        )",
    )
    .execute(pool)
    .await?;
    for statement in [
        "CREATE INDEX IF NOT EXISTS idx_history_session_event ON history_message_index(session_id, event_id)",
        "CREATE INDEX IF NOT EXISTS idx_history_task_event ON history_message_index(task_id, event_id)",
        "CREATE INDEX IF NOT EXISTS idx_history_channel_event ON history_message_index(channel_id, event_id)",
        "CREATE INDEX IF NOT EXISTS idx_history_source_event ON history_message_index(source_kind, event_id)",
    ] {
        sqlx::query(statement).execute(pool).await?;
    }
    create_fts(pool).await
}

async fn create_fts(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query(
        r#"CREATE VIRTUAL TABLE IF NOT EXISTS history_message_fts USING fts5(
            content,
            content='',
            contentless_delete=1,
            tokenize="unicode61 tokenchars '_-'"
        )"#,
    )
    .execute(pool)
    .await?;
    // FTS secure-delete (SQLite >= 3.42) prevents deleted terms from remaining
    // recoverable in free index blocks. Keep compatibility if unavailable.
    let _ = sqlx::query(
        "INSERT INTO history_message_fts(history_message_fts, rank)
         VALUES('secure-delete', 1)",
    )
    .execute(pool)
    .await;
    Ok(())
}

/// Drop and recreate the replaceable history search projection.
///
/// Canonical events and projection metadata remain untouched. Startup
/// backfilling repopulates the contentless index after database verification.
pub(crate) async fn reset_fts_projection(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query("DROP TABLE IF EXISTS history_message_fts")
        .execute(pool)
        .await?;
    create_fts(pool).await
}

fn source_kind(session_id: &str) -> &'static str {
    if session_id.starts_with("specialist:") || session_id.starts_with("sub-") {
        "specialist"
    } else {
        "root"
    }
}

pub(crate) async fn project_event(pool: &SqlitePool, event_id: i64) -> anyhow::Result<bool> {
    let Some(row) = sqlx::query(
        "SELECT id, session_id, event_type, data, created_at, task_id, turn_id
         FROM events WHERE id = ?",
    )
    .bind(event_id)
    .fetch_optional(pool)
    .await?
    else {
        return Ok(false);
    };

    let event_type: String = row.get("event_type");
    let role = match event_type.as_str() {
        "user_message" => "user",
        "assistant_response" => "assistant",
        _ => return Ok(false),
    };
    let raw: String = row.get("data");
    let data: Value = serde_json::from_str(&raw)?;
    let content = data
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if content.is_empty() {
        remove_projection_rows(pool, &[event_id]).await?;
        return Ok(false);
    }

    let session_id: String = row.get("session_id");
    let mut channel_id = data
        .get("channel_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| crate::session::derive_channel_id_from_session(&session_id));
    let mut visibility = data
        .get("channel_visibility")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mut user_role = data
        .get("user_role")
        .and_then(Value::as_str)
        .map(str::to_owned);

    // Assistant payloads predate explicit channel provenance. Resolve it from
    // the closest preceding canonical user event in the same session.
    if role == "assistant" && (visibility.is_none() || channel_id.is_none()) {
        if let Some(origin) = sqlx::query(
            "SELECT data FROM events
             WHERE session_id = ? AND event_type = 'user_message' AND id <= ?
             ORDER BY id DESC LIMIT 1",
        )
        .bind(&session_id)
        .bind(event_id)
        .fetch_optional(pool)
        .await?
        {
            let origin_raw: String = origin.get("data");
            if let Ok(origin_data) = serde_json::from_str::<Value>(&origin_raw) {
                channel_id = channel_id.or_else(|| {
                    origin_data
                        .get("channel_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .or_else(|| crate::session::derive_channel_id_from_session(&session_id))
                });
                visibility = visibility.or_else(|| {
                    origin_data
                        .get("channel_visibility")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                });
                user_role = user_role.or_else(|| {
                    origin_data
                        .get("user_role")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                });
            }
        }
    }

    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM history_message_fts WHERE rowid = ?")
        .bind(event_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query(
        "INSERT INTO history_message_index
            (event_id, session_id, task_id, turn_id, message_id, role, created_at,
             channel_id, channel_visibility, user_role, source_kind)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(event_id) DO UPDATE SET
            session_id=excluded.session_id, task_id=excluded.task_id,
            turn_id=excluded.turn_id, message_id=excluded.message_id,
            role=excluded.role, created_at=excluded.created_at,
            channel_id=excluded.channel_id,
            channel_visibility=excluded.channel_visibility,
            user_role=excluded.user_role, source_kind=excluded.source_kind",
    )
    .bind(event_id)
    .bind(&session_id)
    .bind(row.try_get::<Option<String>, _>("task_id").unwrap_or(None))
    .bind(row.try_get::<Option<String>, _>("turn_id").unwrap_or(None))
    .bind(
        data.get("message_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
    )
    .bind(role)
    .bind(row.get::<String, _>("created_at"))
    .bind(channel_id)
    .bind(visibility.unwrap_or_else(|| "internal".to_string()))
    .bind(user_role)
    .bind(source_kind(&session_id))
    .execute(&mut *tx)
    .await?;
    sqlx::query("INSERT INTO history_message_fts(rowid, content) VALUES (?, ?)")
        .bind(event_id)
        .bind(content)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(true)
}

async fn missing_event_ids(pool: &SqlitePool, limit: i64) -> anyhow::Result<Vec<i64>> {
    Ok(sqlx::query_scalar(
        "SELECT e.id
         FROM events e
         LEFT JOIN history_message_index h ON h.event_id = e.id
         WHERE h.event_id IS NULL
           AND e.event_type IN ('user_message', 'assistant_response')
           AND json_type(e.data, '$.content') = 'text'
           AND trim(json_extract(e.data, '$.content')) <> ''
         ORDER BY e.id LIMIT ?",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?)
}

pub(crate) async fn backfill(pool: &SqlitePool, max_batches: usize) -> anyhow::Result<(u64, i64)> {
    let mut projected = 0_u64;
    for _ in 0..max_batches.max(1) {
        let ids = missing_event_ids(pool, BACKFILL_BATCH).await?;
        if ids.is_empty() {
            break;
        }
        for id in ids {
            if project_event(pool, id).await? {
                projected += 1;
            }
        }
        tokio::task::yield_now().await;
    }
    let pending: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)
         FROM events e LEFT JOIN history_message_index h ON h.event_id=e.id
         WHERE h.event_id IS NULL
           AND e.event_type IN ('user_message','assistant_response')
           AND json_type(e.data, '$.content')='text'
           AND trim(json_extract(e.data, '$.content'))<>''",
    )
    .fetch_one(pool)
    .await?;
    Ok((projected, pending))
}

pub(crate) async fn remove_projection_rows(
    pool: &SqlitePool,
    event_ids: &[i64],
) -> anyhow::Result<()> {
    if event_ids.is_empty() {
        return Ok(());
    }
    let mut tx = pool.begin().await?;
    for id in event_ids {
        // If FTS is damaged, metadata deletion still proceeds on a later repair.
        let _ = sqlx::query("DELETE FROM history_message_fts WHERE rowid = ?")
            .bind(id)
            .execute(&mut *tx)
            .await;
        sqlx::query("DELETE FROM history_message_index WHERE event_id = ?")
            .bind(id)
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;
    Ok(())
}

pub(crate) async fn remove_orphans(pool: &SqlitePool, limit: i64) -> anyhow::Result<u64> {
    let ids: Vec<i64> = sqlx::query_scalar(
        "SELECT h.event_id FROM history_message_index h
         LEFT JOIN events e ON e.id=h.event_id
         WHERE e.id IS NULL LIMIT ?",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    let count = ids.len() as u64;
    remove_projection_rows(pool, &ids).await?;
    Ok(count)
}

async fn rebuild_fts(pool: &SqlitePool) -> anyhow::Result<()> {
    reset_fts_projection(pool).await?;
    let ids: Vec<i64> =
        sqlx::query_scalar("SELECT event_id FROM history_message_index ORDER BY event_id")
            .fetch_all(pool)
            .await?;
    for id in ids {
        project_event(pool, id).await?;
    }
    Ok(())
}

pub(crate) async fn repair_and_backfill(
    pool: &SqlitePool,
    max_batches: usize,
) -> anyhow::Result<ProjectionStats> {
    migrate_history_search(pool).await?;
    let mut stats = ProjectionStats {
        orphans_removed: remove_orphans(pool, BACKFILL_BATCH).await?,
        ..ProjectionStats::default()
    };
    let metadata: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM history_message_index")
        .fetch_one(pool)
        .await?;
    let fts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM history_message_fts")
        .fetch_one(pool)
        .await?;
    let fts_integrity_ok = sqlx::query(
        "INSERT INTO history_message_fts(history_message_fts) VALUES('integrity-check')",
    )
    .execute(pool)
    .await
    .is_ok();
    if metadata != fts || !fts_integrity_ok {
        rebuild_fts(pool).await?;
        stats.fts_rebuilt = true;
    }
    stats.episodes_repaired = repair_episode_provenance(pool, 2_000).await.unwrap_or(0);
    let (projected, pending) = backfill(pool, max_batches).await?;
    stats.projected = projected;
    stats.pending = pending;
    Ok(stats)
}

async fn repair_episode_provenance(pool: &SqlitePool, limit: i64) -> anyhow::Result<u64> {
    let rows = sqlx::query(
        "SELECT id, session_id, start_time, end_time, channel_id,
                start_event_id, end_event_id
         FROM episodes ep
         WHERE
           (ep.channel_id IS NULL AND (
              instr(ep.session_id, 'slack:') > 0
              OR instr(ep.session_id, 'discord:') > 0
              OR instr(ep.session_id, 'telegram:') > 0
              OR EXISTS (
                 SELECT 1 FROM events e
                 WHERE e.session_id = ep.session_id
                   AND e.event_type IN ('user_message', 'assistant_response')
                   AND julianday(e.created_at) >= julianday(ep.start_time)
                   AND julianday(e.created_at) <= julianday(ep.end_time)
                   AND json_type(e.data, '$.channel_id') = 'text'
              )
           ))
           OR ((ep.start_event_id IS NULL OR ep.end_event_id IS NULL) AND EXISTS (
              SELECT 1 FROM events e
              WHERE e.session_id = ep.session_id
                AND e.event_type IN ('user_message', 'assistant_response')
                AND julianday(e.created_at) >= julianday(ep.start_time)
                AND julianday(e.created_at) <= julianday(ep.end_time)
           ))
         ORDER BY id LIMIT ?",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    let mut repaired = 0;
    for row in rows {
        let episode_id: i64 = row.get("id");
        let session_id: String = row.get("session_id");
        let bounds = sqlx::query(
            "SELECT MIN(id) AS start_event_id, MAX(id) AS end_event_id,
                    MAX(CASE
                        WHEN json_type(data, '$.channel_id') = 'text'
                        THEN json_extract(data, '$.channel_id')
                    END) AS event_channel_id
             FROM events WHERE session_id=?
               AND event_type IN ('user_message','assistant_response')
               AND julianday(created_at)>=julianday(?)
               AND julianday(created_at)<=julianday(?)",
        )
        .bind(&session_id)
        .bind(row.get::<String, _>("start_time"))
        .bind(row.get::<String, _>("end_time"))
        .fetch_one(pool)
        .await?;
        let channel_id = row
            .try_get::<Option<String>, _>("channel_id")
            .unwrap_or(None)
            .or_else(|| crate::session::derive_channel_id_from_session(&session_id))
            .or_else(|| bounds.try_get("event_channel_id").ok().flatten());
        let start_event_id = row
            .try_get::<Option<i64>, _>("start_event_id")
            .unwrap_or(None)
            .or_else(|| bounds.try_get("start_event_id").ok().flatten());
        let end_event_id = row
            .try_get::<Option<i64>, _>("end_event_id")
            .unwrap_or(None)
            .or_else(|| bounds.try_get("end_event_id").ok().flatten());
        let changed = row
            .try_get::<Option<String>, _>("channel_id")
            .unwrap_or(None)
            .is_none()
            && channel_id.is_some()
            || row
                .try_get::<Option<i64>, _>("start_event_id")
                .unwrap_or(None)
                .is_none()
                && start_event_id.is_some()
            || row
                .try_get::<Option<i64>, _>("end_event_id")
                .unwrap_or(None)
                .is_none()
                && end_event_id.is_some();
        if !changed {
            continue;
        }
        repaired += sqlx::query(
            "UPDATE episodes
             SET channel_id=COALESCE(channel_id, ?),
                 start_event_id=COALESCE(start_event_id, ?),
                 end_event_id=COALESCE(end_event_id, ?)
             WHERE id=?
               AND (channel_id IS NULL OR start_event_id IS NULL OR end_event_id IS NULL)",
        )
        .bind(channel_id)
        .bind(start_event_id)
        .bind(end_event_id)
        .bind(episode_id)
        .execute(pool)
        .await?
        .rows_affected();
    }
    Ok(repaired)
}

pub(crate) async fn remove_session_projection_in_tx(
    tx: &mut Transaction<'_, Sqlite>,
    session_id: &str,
) -> anyhow::Result<u64> {
    let metadata_exists: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master
         WHERE type='table' AND name='history_message_index'",
    )
    .fetch_one(&mut **tx)
    .await?;
    if metadata_exists == 0 {
        return Ok(0);
    }
    let ids: Vec<i64> =
        sqlx::query_scalar("SELECT event_id FROM history_message_index WHERE session_id = ?")
            .bind(session_id)
            .fetch_all(&mut **tx)
            .await?;
    let count = ids.len() as u64;
    if ids.is_empty() {
        return Ok(0);
    }
    let fts_exists: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master
         WHERE type='table' AND name='history_message_fts'",
    )
    .fetch_one(&mut **tx)
    .await?;
    if fts_exists > 0 {
        // Unlike background orphan cleanup, an explicit wipe is strict: a
        // damaged FTS index must roll the wipe transaction back rather than
        // leave recoverable terms behind while reporting success.
        for id in &ids {
            sqlx::query("DELETE FROM history_message_fts WHERE rowid = ?")
                .bind(id)
                .execute(&mut **tx)
                .await?;
        }
    }
    sqlx::query("DELETE FROM history_message_index WHERE session_id = ?")
        .bind(session_id)
        .execute(&mut **tx)
        .await?;
    Ok(count)
}

pub(crate) async fn checkpoint_after_wipe(pool: &SqlitePool) -> anyhow::Result<()> {
    for attempt in 0..5 {
        let (busy, _, _): (i64, i64, i64) = sqlx::query_as("PRAGMA wal_checkpoint(TRUNCATE)")
            .fetch_one(pool)
            .await?;
        if busy == 0 {
            return Ok(());
        }
        if attempt < 4 {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    }
    anyhow::bail!("could not truncate SQLite WAL after wiping session data")
}

fn match_tokens(query: &str) -> anyhow::Result<Vec<String>> {
    let tokens: Vec<String> = query
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '-' | '+' | '#')))
        .filter(|token| !token.is_empty())
        .take(24)
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect();
    if tokens.is_empty() {
        anyhow::bail!("query must contain at least one searchable character");
    }
    Ok(tokens)
}

fn match_expression(tokens: &[String], operator: &str) -> String {
    tokens.join(operator)
}

fn push_scope(qb: &mut QueryBuilder<'_, Sqlite>, scope: &HistoryScope, alias: &str) {
    let current_is_specialist =
        scope.session_id.starts_with("specialist:") || scope.session_id.starts_with("sub-");
    if !(scope.include_subagents
        || scope.visibility == ChannelVisibility::Internal && current_is_specialist)
    {
        qb.push(format!(" AND {alias}.source_kind = 'root'"));
    }
    if let Some(session) = scope.session_filter.as_deref() {
        qb.push(format!(" AND {alias}.session_id = "));
        qb.push_bind(session.to_string());
    }
    if let Some(task) = scope.task_filter.as_deref() {
        qb.push(format!(" AND {alias}.task_id = "));
        qb.push_bind(task.to_string());
    }
    match scope.visibility {
        ChannelVisibility::Private => {}
        ChannelVisibility::PrivateGroup | ChannelVisibility::Public => {
            qb.push(format!(" AND {alias}.channel_id = "));
            qb.push_bind(scope.channel_id.clone().unwrap_or_default());
        }
        ChannelVisibility::Internal if scope.trusted => {}
        ChannelVisibility::Internal => {
            qb.push(format!(" AND {alias}.session_id = "));
            qb.push_bind(scope.session_id.clone());
        }
        ChannelVisibility::PublicExternal => {
            qb.push(" AND 0 = 1");
        }
    };
}

fn ensure_scope(scope: &HistoryScope) -> anyhow::Result<()> {
    if scope.user_role != UserRole::Owner {
        anyhow::bail!("exact history is restricted to the owner");
    }
    if scope.visibility == ChannelVisibility::PublicExternal {
        anyhow::bail!("exact history is disabled in public-external channels");
    }
    if matches!(
        scope.visibility,
        ChannelVisibility::PrivateGroup | ChannelVisibility::Public
    ) && scope.channel_id.is_none()
    {
        anyhow::bail!("channel-scoped history requires a trusted channel identifier");
    }
    Ok(())
}

fn row_to_message(row: &sqlx::sqlite::SqliteRow) -> HistoryMessage {
    HistoryMessage {
        event_id: row.get("event_id"),
        session_id: row.get("session_id"),
        task_id: row.try_get("task_id").unwrap_or(None),
        turn_id: row.try_get("turn_id").unwrap_or(None),
        message_id: row.try_get("message_id").unwrap_or(None),
        role: row.get("role"),
        content: row.get("content"),
        created_at: row.get("created_at"),
        source_kind: row.get("source_kind"),
        lexical_rank: row.try_get("lexical_rank").unwrap_or(None),
    }
}

/// Collapse copies of the same canonical message that survived under multiple
/// compressed/continued session IDs. Message IDs are generated as stable UUIDs
/// on the canonical write path; rows without one remain distinct rather than
/// falling back to unsafe content-hash deduplication.
fn dedup_stable_message_ids(messages: &mut Vec<HistoryMessage>) {
    let mut seen = HashSet::new();
    messages.retain(|message| {
        let Some(message_id) = message
            .message_id
            .as_deref()
            .filter(|message_id| !message_id.trim().is_empty())
        else {
            return true;
        };
        seen.insert((message.role.clone(), message_id.to_string()))
    });
}

pub(crate) async fn snapshot_max_event_id(pool: &SqlitePool) -> anyhow::Result<i64> {
    Ok(
        sqlx::query_scalar("SELECT COALESCE(MAX(id), 0) FROM events")
            .fetch_one(pool)
            .await?,
    )
}

pub(crate) async fn search(
    pool: &SqlitePool,
    query: &str,
    scope: &HistoryScope,
    limit: usize,
    semantic_sessions: &HashSet<String>,
) -> anyhow::Result<Vec<HistoryMessage>> {
    ensure_scope(scope)?;
    let tokens = match_tokens(query)?;
    let strict_expression = match_expression(&tokens, " AND ");
    let candidate_limit = limit.clamp(1, 50).saturating_mul(8).min(240) as i64;
    let mut messages = search_expression(pool, &strict_expression, scope, candidate_limit).await?;

    // Exact all-term matching is precise but brittle under paraphrase. Broaden
    // only when it cannot fill the requested set, then merge/deduplicate below.
    if tokens.len() > 1 && messages.len() < limit.clamp(1, 50) {
        let broad_expression = match_expression(&tokens, " OR ");
        let mut broad = search_expression(pool, &broad_expression, scope, candidate_limit).await?;
        for message in &mut broad {
            // FTS5 bm25 ranks lower values first. Pull broad-only candidates
            // toward zero so exact all-term hits remain preferred.
            message.lexical_rank = message.lexical_rank.map(|rank| rank * 0.85);
        }
        messages.extend(broad);
    }

    // Semantic episodes are a soft prior only. Global lexical discovery is
    // never restricted to the episode set.
    messages.sort_by(|a, b| {
        let adjusted = |m: &HistoryMessage| {
            let rank = m.lexical_rank.unwrap_or(0.0);
            if semantic_sessions.contains(&m.session_id) {
                rank * 1.08
            } else {
                rank
            }
        };
        adjusted(a)
            .partial_cmp(&adjusted(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    dedup_stable_message_ids(&mut messages);
    messages.truncate(limit.clamp(1, 50));
    Ok(messages)
}

async fn search_expression(
    pool: &SqlitePool,
    expression: &str,
    scope: &HistoryScope,
    candidate_limit: i64,
) -> anyhow::Result<Vec<HistoryMessage>> {
    let mut qb = QueryBuilder::<Sqlite>::new(
        "SELECT h.event_id, h.session_id, h.task_id, h.turn_id, h.message_id,
                h.role, h.created_at, h.source_kind,
                json_extract(e.data, '$.content') AS content,
                bm25(history_message_fts) AS lexical_rank
         FROM history_message_fts
         JOIN history_message_index h ON h.event_id=history_message_fts.rowid
         JOIN events e ON e.id=h.event_id
         WHERE history_message_fts MATCH ",
    );
    qb.push_bind(expression.to_string());
    qb.push(" AND h.event_id <= ");
    qb.push_bind(scope.snapshot_max_event_id);
    push_scope(&mut qb, scope, "h");
    qb.push(" ORDER BY lexical_rank LIMIT ");
    qb.push_bind(candidate_limit);
    let rows = qb.build().fetch_all(pool).await?;
    Ok(rows.iter().map(row_to_message).collect())
}

pub(crate) async fn open(
    pool: &SqlitePool,
    event_id: i64,
    scope: &HistoryScope,
) -> anyhow::Result<Option<HistoryMessage>> {
    ensure_scope(scope)?;
    let mut qb = QueryBuilder::<Sqlite>::new(
        "SELECT h.event_id, h.session_id, h.task_id, h.turn_id, h.message_id,
                h.role, h.created_at, h.source_kind,
                json_extract(e.data, '$.content') AS content,
                NULL AS lexical_rank
         FROM history_message_index h JOIN events e ON e.id=h.event_id
         WHERE h.event_id = ",
    );
    qb.push_bind(event_id);
    qb.push(" AND h.event_id <= ");
    qb.push_bind(scope.snapshot_max_event_id);
    push_scope(&mut qb, scope, "h");
    Ok(qb
        .build()
        .fetch_optional(pool)
        .await?
        .as_ref()
        .map(row_to_message))
}

pub(crate) async fn page(
    pool: &SqlitePool,
    anchor_event_id: i64,
    older: bool,
    scope: &HistoryScope,
    limit: usize,
) -> anyhow::Result<Vec<HistoryMessage>> {
    ensure_scope(scope)?;
    let mut qb = QueryBuilder::<Sqlite>::new(
        "SELECT h.event_id, h.session_id, h.task_id, h.turn_id, h.message_id,
                h.role, h.created_at, h.source_kind,
                json_extract(e.data, '$.content') AS content,
                NULL AS lexical_rank
         FROM history_message_index h JOIN events e ON e.id=h.event_id WHERE h.event_id ",
    );
    qb.push(if older { " < " } else { " > " });
    qb.push_bind(anchor_event_id);
    qb.push(" AND h.event_id <= ");
    qb.push_bind(scope.snapshot_max_event_id);
    push_scope(&mut qb, scope, "h");
    qb.push(if older {
        " ORDER BY h.event_id DESC LIMIT "
    } else {
        " ORDER BY h.event_id ASC LIMIT "
    });
    qb.push_bind(limit.clamp(1, 50) as i64);
    let rows = qb.build().fetch_all(pool).await?;
    let mut messages: Vec<_> = rows.iter().map(row_to_message).collect();
    if older {
        messages.reverse();
    }
    Ok(messages)
}

pub(crate) async fn context(
    pool: &SqlitePool,
    event_id: i64,
    radius: usize,
    scope: &HistoryScope,
) -> anyhow::Result<Vec<HistoryMessage>> {
    let Some(anchor) = open(pool, event_id, scope).await? else {
        return Ok(Vec::new());
    };
    let mut narrowed = scope.clone();
    narrowed.session_filter = Some(anchor.session_id.clone());
    let mut messages = page(pool, event_id, true, &narrowed, radius).await?;
    messages.push(anchor);
    messages.extend(page(pool, event_id, false, &narrowed, radius).await?);
    dedup_stable_message_ids(&mut messages);
    Ok(messages)
}

pub(crate) async fn task_bookends(
    pool: &SqlitePool,
    task_id: Option<&str>,
    session_id: &str,
    scope: &HistoryScope,
) -> anyhow::Result<TaskBookends> {
    let Some(task_id) = task_id else {
        return Ok(TaskBookends {
            objective: None,
            generated_objective: None,
            objective_source: "none".to_string(),
            resolution: None,
            generated_resolution: None,
            resolution_source: "none".to_string(),
        });
    };
    let mut task_scope = scope.clone();
    task_scope.task_filter = Some(task_id.to_string());
    task_scope.session_filter = Some(session_id.to_string());
    let objective = page(pool, 0, false, &task_scope, 50)
        .await?
        .into_iter()
        .find(|m| m.role == "user");
    let generated_objective = if objective.is_none() {
        sqlx::query_scalar::<_, Option<String>>(
            "SELECT COALESCE(
                        json_extract(data, '$.user_message'),
                        json_extract(data, '$.description')
                    )
             FROM events
             WHERE task_id=? AND session_id=? AND event_type='task_start' AND id<=?
             ORDER BY id ASC LIMIT 1",
        )
        .bind(task_id)
        .bind(session_id)
        .bind(scope.snapshot_max_event_id)
        .fetch_optional(pool)
        .await?
        .flatten()
    } else {
        None
    };
    let resolution = page(
        pool,
        scope.snapshot_max_event_id.saturating_add(1),
        true,
        &task_scope,
        50,
    )
    .await?
    .into_iter()
    .rev()
    .find(|m| m.role == "assistant");
    let generated_resolution = if resolution.is_none() {
        sqlx::query_scalar::<_, Option<String>>(
            "SELECT json_extract(data, '$.summary') FROM events
             WHERE task_id=? AND session_id=? AND event_type='task_end' AND id<=?
             ORDER BY id DESC LIMIT 1",
        )
        .bind(task_id)
        .bind(session_id)
        .bind(scope.snapshot_max_event_id)
        .fetch_optional(pool)
        .await?
        .flatten()
    } else {
        None
    };
    let resolution_source = if resolution.is_some() {
        "exact_assistant_message"
    } else if generated_resolution.is_some() {
        "generated_task_summary"
    } else {
        "none"
    };
    let objective_source = if objective.is_some() {
        "exact_user_message"
    } else if generated_objective.is_some() {
        "generated_task_envelope"
    } else {
        "none"
    };
    Ok(TaskBookends {
        objective,
        objective_source: objective_source.to_string(),
        generated_objective,
        resolution,
        generated_resolution,
        resolution_source: resolution_source.to_string(),
    })
}

pub(crate) async fn coverage(pool: &SqlitePool) -> anyhow::Result<HistoryCoverage> {
    let canonical_messages: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM events
         WHERE event_type IN ('user_message','assistant_response')
           AND json_type(data, '$.content')='text'
           AND trim(json_extract(data, '$.content'))<>''",
    )
    .fetch_one(pool)
    .await?;
    let row = sqlx::query(
        "SELECT COUNT(*) AS count, MIN(created_at) AS oldest, MAX(created_at) AS newest
         FROM history_message_index",
    )
    .fetch_one(pool)
    .await?;
    let indexed_messages: i64 = row.get("count");
    Ok(HistoryCoverage {
        canonical_messages,
        indexed_messages,
        pending_messages: canonical_messages.saturating_sub(indexed_messages),
        oldest_indexed_at: row.try_get("oldest").unwrap_or(None),
        newest_indexed_at: row.try_get("newest").unwrap_or(None),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::db::migrations::migrate_events(&pool).await.unwrap();
        migrate_history_search(&pool).await.unwrap();
        pool
    }

    async fn insert(pool: &SqlitePool, session: &str, role: &str, content: &str) -> i64 {
        let event_type = if role == "user" {
            "user_message"
        } else {
            "assistant_response"
        };
        let data = serde_json::json!({
            "content": content,
            "channel_visibility": "private",
            "user_role": "owner",
            "message_id": format!("{session}-{role}")
        });
        let result = sqlx::query(
            "INSERT INTO events(session_id,event_type,data,created_at,task_id,turn_id)
             VALUES(?,?,?,?,?,?)",
        )
        .bind(session)
        .bind(event_type)
        .bind(data.to_string())
        .bind(Utc::now().to_rfc3339())
        .bind("task-1")
        .bind("turn-1")
        .execute(pool)
        .await
        .unwrap();
        result.last_insert_rowid()
    }

    fn private_scope(snapshot: i64) -> HistoryScope {
        HistoryScope {
            session_id: "root".into(),
            channel_id: None,
            visibility: ChannelVisibility::Private,
            user_role: UserRole::Owner,
            trusted: false,
            include_subagents: false,
            session_filter: None,
            task_filter: None,
            snapshot_max_event_id: snapshot,
        }
    }

    #[tokio::test]
    async fn short_terms_are_searchable_and_canonical_survives_missing_fts() {
        let pool = pool().await;
        let ai = insert(&pool, "root", "user", "AI with Go and C").await;
        project_event(&pool, ai).await.unwrap();
        let scope = private_scope(ai);
        for term in ["AI", "Go", "C"] {
            assert_eq!(
                search(&pool, term, &scope, 5, &HashSet::new())
                    .await
                    .unwrap()
                    .len(),
                1
            );
        }

        sqlx::query("DROP TABLE history_message_fts")
            .execute(&pool)
            .await
            .unwrap();
        let canonical = insert(&pool, "root", "assistant", "still canonical").await;
        assert!(project_event(&pool, canonical).await.is_err());
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM events WHERE id=?")
            .bind(canonical)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn group_scope_cannot_cross_channels_and_specialists_are_opt_in() {
        let pool = pool().await;
        let a = insert(&pool, "slack:C1", "user", "shared needle").await;
        let b = insert(&pool, "slack:C2", "user", "shared needle").await;
        let child = insert(&pool, "specialist:research:1", "user", "shared needle").await;
        for id in [a, b, child] {
            project_event(&pool, id).await.unwrap();
        }
        let mut scope = private_scope(child);
        scope.visibility = ChannelVisibility::PrivateGroup;
        scope.channel_id = Some("slack:C1".into());
        let hits = search(&pool, "needle", &scope, 10, &HashSet::new())
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].event_id, a);
    }

    #[tokio::test]
    async fn stable_message_ids_deduplicate_compressed_lineages() {
        let pool = pool().await;
        let a = insert(&pool, "root-before-compression", "user", "lineage needle").await;
        let b = insert(&pool, "root-after-compression", "user", "lineage needle").await;
        for id in [a, b] {
            sqlx::query(
                "UPDATE events
                 SET data=json_set(data, '$.message_id', 'stable-message-id')
                 WHERE id=?",
            )
            .bind(id)
            .execute(&pool)
            .await
            .unwrap();
            project_event(&pool, id).await.unwrap();
        }

        let hits = search(
            &pool,
            "lineage needle",
            &private_scope(b),
            10,
            &HashSet::new(),
        )
        .await
        .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn search_broadens_without_phrase_specific_rules() {
        let pool = pool().await;
        let event_id = insert(
            &pool,
            "root",
            "assistant",
            "The service was deployed to the Frankfurt region.",
        )
        .await;
        project_event(&pool, event_id).await.unwrap();

        // `source` is absent, so strict AND has no hit. The generic OR fallback
        // still recovers the semantically adjacent canonical message.
        let hits = search(
            &pool,
            "service deployment source",
            &private_scope(event_id),
            10,
            &HashSet::new(),
        )
        .await
        .unwrap();
        assert!(hits.iter().any(|hit| hit.event_id == event_id));
    }

    #[tokio::test]
    async fn generated_bookends_are_bound_to_the_anchor_session() {
        let pool = pool().await;
        for (session, description) in [
            ("slack:C1", "objective from channel one"),
            ("slack:C2", "objective from channel two"),
        ] {
            sqlx::query(
                "INSERT INTO events(session_id,event_type,data,created_at,task_id,turn_id)
                 VALUES(?, 'task_start', ?, ?, 'task-1', 'turn-1')",
            )
            .bind(session)
            .bind(
                serde_json::json!({
                    "description": description,
                    "task_id": "task-1"
                })
                .to_string(),
            )
            .bind(Utc::now().to_rfc3339())
            .execute(&pool)
            .await
            .unwrap();
        }
        let anchor = insert(&pool, "slack:C1", "assistant", "channel one answer").await;
        project_event(&pool, anchor).await.unwrap();

        let mut scope = private_scope(anchor);
        scope.visibility = ChannelVisibility::PrivateGroup;
        scope.channel_id = Some("slack:C1".into());
        let bookends = task_bookends(&pool, Some("task-1"), "slack:C1", &scope)
            .await
            .unwrap();
        assert_eq!(
            bookends.generated_objective.as_deref(),
            Some("objective from channel one")
        );
    }

    #[tokio::test]
    async fn lagging_projection_backfills_and_orphans_repair() {
        let pool = pool().await;
        let id = insert(&pool, "root", "user", "repairable lag").await;
        let before = coverage(&pool).await.unwrap();
        assert_eq!(before.pending_messages, 1);
        let (projected, pending) = backfill(&pool, 1).await.unwrap();
        assert_eq!(projected, 1);
        assert_eq!(pending, 0);
        let healthy = repair_and_backfill(&pool, 1).await.unwrap();
        assert!(!healthy.fts_rebuilt);
        assert_eq!(healthy.pending, 0);

        sqlx::query("DELETE FROM events WHERE id=?")
            .bind(id)
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(remove_orphans(&pool, 500).await.unwrap(), 1);
        let indexed: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM history_message_index")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(indexed, 0);
    }

    #[tokio::test]
    async fn episode_provenance_repair_skips_irreparable_rows_and_counts_real_changes_once() {
        let pool = pool().await;
        sqlx::query(
            "CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                channel_id TEXT,
                start_event_id INTEGER,
                end_event_id INTEGER
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO episodes(session_id,start_time,end_time)
             VALUES ('specialist:research:no-events',
                     '2026-01-01T00:00:00Z', '2026-01-01T00:01:00Z'),
                    ('slack:C1',
                     '2026-01-01T11:59:00Z', '2026-01-01T12:01:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();
        let event_id = sqlx::query(
            "INSERT INTO events(session_id,event_type,data,created_at)
             VALUES ('slack:C1','user_message',?, '2026-01-01 12:00:00')",
        )
        .bind(
            serde_json::json!({
                "content": "repair me",
                "channel_id": "slack:C1"
            })
            .to_string(),
        )
        .execute(&pool)
        .await
        .unwrap()
        .last_insert_rowid();

        assert_eq!(repair_episode_provenance(&pool, 1).await.unwrap(), 1);
        assert_eq!(repair_episode_provenance(&pool, 1).await.unwrap(), 0);

        let repaired = sqlx::query(
            "SELECT channel_id,start_event_id,end_event_id
             FROM episodes WHERE session_id='slack:C1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(repaired.get::<String, _>("channel_id"), "slack:C1");
        assert_eq!(repaired.get::<i64, _>("start_event_id"), event_id);
        assert_eq!(repaired.get::<i64, _>("end_event_id"), event_id);

        let untouched: (Option<String>, Option<i64>, Option<i64>) = sqlx::query_as(
            "SELECT channel_id,start_event_id,end_event_id
             FROM episodes WHERE session_id='specialist:research:no-events'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(untouched, (None, None, None));
    }
}
