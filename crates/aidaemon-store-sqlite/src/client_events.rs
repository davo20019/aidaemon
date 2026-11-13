//! The Client Gateway's durable event log and its remembered replies
//! (feature `client-events`).
//!
//! `client_events` holds every durable event a client may need on resume,
//! numbered per owner by a gap-free `seq` assigned inside the insert (which
//! SQLite runs under its write lock, so concurrent appends never collide).
//! Retention removes old rows but never an owner's newest one, so numbering
//! continues where it stopped. `client_requests` remembers each command's
//! reply by its `request_id`, so a retried command gets the first answer.
//!
//! The payloads are opaque JSON text: this crate knows nothing of the
//! protocol. The tables are created with `CREATE TABLE IF NOT EXISTS` when
//! the log is opened, not by a versioned migration, so a daemon that never
//! enables the gateway leaves its schema (and the version older builds
//! check) untouched.

use sqlx::{Row, SqlitePool};

use crate::Error;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS client_events (
    owner      TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    at_ms      INTEGER NOT NULL,
    session_id TEXT,
    kind       TEXT    NOT NULL,
    ref_id     TEXT,
    payload    TEXT    NOT NULL CHECK (json_valid(payload)),
    PRIMARY KEY (owner, seq)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_client_events_kind ON client_events (owner, kind, seq);
CREATE INDEX IF NOT EXISTS idx_client_events_ref ON client_events (owner, ref_id);
CREATE TABLE IF NOT EXISTS client_requests (
    owner      TEXT    NOT NULL,
    request_id TEXT    NOT NULL,
    at_ms      INTEGER NOT NULL,
    reply      TEXT    NOT NULL CHECK (json_valid(reply)),
    PRIMARY KEY (owner, request_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_client_requests_at ON client_requests (at_ms);
"#;

/// One logged event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientEventRow {
    pub seq: u64,
    pub at_ms: u64,
    pub session_id: Option<String>,
    /// The event's type tag.
    pub kind: String,
    /// What the event is about (an approval id), for [`SqliteClientEventLog::open_refs`].
    pub ref_id: Option<String>,
    /// The JSON text appended.
    pub payload: String,
}

/// The log on a pool. Cheap to clone.
#[derive(Debug, Clone)]
pub struct SqliteClientEventLog {
    pool: SqlitePool,
}

fn to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

impl SqliteClientEventLog {
    /// Use `pool`, creating the tables if they are not there.
    pub async fn open(pool: SqlitePool) -> Result<Self, Error> {
        sqlx::raw_sql(SCHEMA).execute(&pool).await?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Append one event; returns its `seq`.
    pub async fn append(
        &self,
        owner: &str,
        at_ms: u64,
        session_id: Option<&str>,
        kind: &str,
        ref_id: Option<&str>,
        payload: &str,
    ) -> Result<u64, Error> {
        let seq: i64 = sqlx::query_scalar(
            "INSERT INTO client_events (owner, seq, at_ms, session_id, kind, ref_id, payload)
             VALUES (?1, (SELECT COALESCE(MAX(seq), 0) + 1 FROM client_events WHERE owner = ?1),
                     ?2, ?3, ?4, ?5, ?6)
             RETURNING seq",
        )
        .bind(owner)
        .bind(to_i64(at_ms))
        .bind(session_id)
        .bind(kind)
        .bind(ref_id)
        .bind(payload)
        .fetch_one(&self.pool)
        .await?;
        Ok(to_u64(seq))
    }

    /// Up to `limit` events after `seq`, oldest first.
    pub async fn after(
        &self,
        owner: &str,
        seq: u64,
        limit: u32,
    ) -> Result<Vec<ClientEventRow>, Error> {
        let rows = sqlx::query(
            "SELECT seq, at_ms, session_id, kind, ref_id, payload FROM client_events
             WHERE owner = ?1 AND seq > ?2 ORDER BY seq LIMIT ?3",
        )
        .bind(owner)
        .bind(to_i64(seq))
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row).collect())
    }

    /// The oldest and newest `seq` kept; `None` for an empty log.
    pub async fn bounds(&self, owner: &str) -> Result<Option<(u64, u64)>, Error> {
        let row = sqlx::query("SELECT MIN(seq), MAX(seq) FROM client_events WHERE owner = ?1")
            .bind(owner)
            .fetch_one(&self.pool)
            .await?;
        let min: Option<i64> = row.get(0);
        let max: Option<i64> = row.get(1);
        Ok(min.zip(max).map(|(a, b)| (to_u64(a), to_u64(b))))
    }

    /// The newest event of `kind`.
    pub async fn latest_of_kind(
        &self,
        owner: &str,
        kind: &str,
    ) -> Result<Option<ClientEventRow>, Error> {
        let found = sqlx::query(
            "SELECT seq, at_ms, session_id, kind, ref_id, payload FROM client_events
             WHERE owner = ?1 AND kind = ?2 ORDER BY seq DESC LIMIT 1",
        )
        .bind(owner)
        .bind(kind)
        .fetch_optional(&self.pool)
        .await?;
        Ok(found.as_ref().map(row))
    }

    /// Events of `open_kind` whose `ref_id` no event of `close_kind` has
    /// closed, oldest first: approvals still pending in the log.
    pub async fn open_refs(
        &self,
        owner: &str,
        open_kind: &str,
        close_kind: &str,
    ) -> Result<Vec<ClientEventRow>, Error> {
        let rows = sqlx::query(
            "SELECT o.seq, o.at_ms, o.session_id, o.kind, o.ref_id, o.payload
             FROM client_events o
             WHERE o.owner = ?1 AND o.kind = ?2 AND o.ref_id IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM client_events c
                   WHERE c.owner = ?1 AND c.kind = ?3 AND c.ref_id = o.ref_id)
             ORDER BY o.seq",
        )
        .bind(owner)
        .bind(open_kind)
        .bind(close_kind)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row).collect())
    }

    /// Remove events older than `before_ms`, and beyond the newest
    /// `keep_newest`, but never the owner's newest event. Returns how many
    /// were removed.
    pub async fn prune(&self, owner: &str, before_ms: u64, keep_newest: u64) -> Result<u64, Error> {
        let keep = keep_newest.max(1);
        let done = sqlx::query(
            "DELETE FROM client_events
             WHERE owner = ?1
               AND seq < (SELECT MAX(seq) FROM client_events WHERE owner = ?1)
               AND (at_ms < ?2
                    OR seq <= (SELECT MAX(seq) FROM client_events WHERE owner = ?1) - ?3)",
        )
        .bind(owner)
        .bind(to_i64(before_ms))
        .bind(to_i64(keep))
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected())
    }

    /// The reply remembered for `request_id`.
    pub async fn remembered_reply(
        &self,
        owner: &str,
        request_id: &str,
    ) -> Result<Option<String>, Error> {
        Ok(sqlx::query_scalar(
            "SELECT reply FROM client_requests WHERE owner = ?1 AND request_id = ?2",
        )
        .bind(owner)
        .bind(request_id)
        .fetch_optional(&self.pool)
        .await?)
    }

    /// Remember `reply` for `request_id`; the first reply stays. Returns the
    /// reply now remembered.
    pub async fn remember_reply(
        &self,
        owner: &str,
        request_id: &str,
        at_ms: u64,
        reply: &str,
    ) -> Result<String, Error> {
        sqlx::query(
            "INSERT INTO client_requests (owner, request_id, at_ms, reply) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT (owner, request_id) DO NOTHING",
        )
        .bind(owner)
        .bind(request_id)
        .bind(to_i64(at_ms))
        .bind(reply)
        .execute(&self.pool)
        .await?;
        Ok(self
            .remembered_reply(owner, request_id)
            .await?
            .unwrap_or_else(|| reply.to_string()))
    }

    /// Forget replies remembered before `before_ms`; returns how many.
    pub async fn forget_replies_before(&self, before_ms: u64) -> Result<u64, Error> {
        let done = sqlx::query("DELETE FROM client_requests WHERE at_ms < ?1")
            .bind(to_i64(before_ms))
            .execute(&self.pool)
            .await?;
        Ok(done.rows_affected())
    }
}

fn row(row: &sqlx::sqlite::SqliteRow) -> ClientEventRow {
    ClientEventRow {
        seq: to_u64(row.get("seq")),
        at_ms: to_u64(row.get("at_ms")),
        session_id: row.get("session_id"),
        kind: row.get("kind"),
        ref_id: row.get("ref_id"),
        payload: row.get("payload"),
    }
}
