//! The Client Gateway's durable event log, its remembered replies, and the
//! devices paired with it (feature `client-events`).
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
CREATE TABLE IF NOT EXISTS client_pairing (
    offer_id       TEXT PRIMARY KEY NOT NULL,
    code_hash      TEXT    NOT NULL,
    origin_session TEXT,
    created_at_ms  INTEGER NOT NULL,
    expires_at_ms  INTEGER NOT NULL,
    attempts       INTEGER NOT NULL DEFAULT 0,
    status         TEXT    NOT NULL,
    device_name    TEXT,
    device_kind    TEXT,
    public_key     TEXT,
    scopes         TEXT,
    device_id      TEXT
);
CREATE TABLE IF NOT EXISTS client_devices (
    device_id     TEXT PRIMARY KEY NOT NULL,
    name          TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    public_key    TEXT    NOT NULL,
    scopes        TEXT    NOT NULL,
    risk_ceiling  TEXT,
    created_at_ms INTEGER NOT NULL,
    last_seen_ms  INTEGER,
    revoked_at_ms INTEGER
);
"#;

/// One pairing offer and, once redeemed, the device asking to pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairingRow {
    pub offer_id: String,
    pub code_hash: String,
    /// The chat the offer was made in (where the owner confirms); `None`
    /// for one made locally.
    pub origin_session: Option<String>,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
    pub attempts: u32,
    /// `open`, `pending` (redeemed, waiting for the owner), `approved`,
    /// `denied` or `expired`.
    pub status: String,
    pub device_name: Option<String>,
    pub device_kind: Option<String>,
    pub public_key: Option<String>,
    pub scopes: Option<String>,
    pub device_id: Option<String>,
}

/// One paired device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRow {
    pub device_id: String,
    pub name: String,
    pub kind: String,
    pub public_key: String,
    /// Comma-separated.
    pub scopes: String,
    pub risk_ceiling: Option<String>,
    pub created_at_ms: u64,
    pub last_seen_ms: Option<u64>,
    pub revoked_at_ms: Option<u64>,
}

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

impl SqliteClientEventLog {
    pub async fn create_offer(&self, offer: &PairingRow) -> Result<(), Error> {
        sqlx::query(
            "INSERT INTO client_pairing (offer_id, code_hash, origin_session, created_at_ms,
                 expires_at_ms, attempts, status)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, 'open')",
        )
        .bind(&offer.offer_id)
        .bind(&offer.code_hash)
        .bind(&offer.origin_session)
        .bind(to_i64(offer.created_at_ms))
        .bind(to_i64(offer.expires_at_ms))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn offer(&self, offer_id: &str) -> Result<Option<PairingRow>, Error> {
        let found = sqlx::query("SELECT * FROM client_pairing WHERE offer_id = ?1")
            .bind(offer_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(found.as_ref().map(pairing_row))
    }

    /// Redeem an open, unexpired offer whose code hash matches, at most
    /// `max_attempts` tries: one atomic update, so an offer pairs one
    /// device. Returns whether it was redeemed; a wrong code counts an
    /// attempt and closes the offer at the limit.
    #[allow(clippy::too_many_arguments)]
    pub async fn redeem_offer(
        &self,
        offer_id: &str,
        code_hash: &str,
        now_ms: u64,
        max_attempts: u32,
        device_name: &str,
        device_kind: &str,
        public_key: &str,
        scopes: &str,
    ) -> Result<bool, Error> {
        let done = sqlx::query(
            "UPDATE client_pairing SET status = 'pending', device_name = ?4, device_kind = ?5,
                 public_key = ?6, scopes = ?7
             WHERE offer_id = ?1 AND code_hash = ?2 AND status = 'open' AND expires_at_ms > ?3
               AND attempts < ?8",
        )
        .bind(offer_id)
        .bind(code_hash)
        .bind(to_i64(now_ms))
        .bind(device_name)
        .bind(device_kind)
        .bind(public_key)
        .bind(scopes)
        .bind(i64::from(max_attempts))
        .execute(&self.pool)
        .await?;
        if done.rows_affected() == 1 {
            return Ok(true);
        }
        sqlx::query(
            "UPDATE client_pairing SET attempts = attempts + 1,
                 status = CASE WHEN attempts + 1 >= ?2 THEN 'expired' ELSE status END
             WHERE offer_id = ?1 AND status = 'open'",
        )
        .bind(offer_id)
        .bind(i64::from(max_attempts))
        .execute(&self.pool)
        .await?;
        Ok(false)
    }

    /// Settle a pending offer: `approved` (with the new device's id) or
    /// `denied`. Returns whether it was pending.
    pub async fn decide_offer(
        &self,
        offer_id: &str,
        status: &str,
        device_id: Option<&str>,
    ) -> Result<bool, Error> {
        let done = sqlx::query(
            "UPDATE client_pairing SET status = ?2, device_id = ?3
             WHERE offer_id = ?1 AND status = 'pending'",
        )
        .bind(offer_id)
        .bind(status)
        .bind(device_id)
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected() == 1)
    }

    pub async fn add_device(&self, device: &DeviceRow) -> Result<(), Error> {
        sqlx::query(
            "INSERT INTO client_devices (device_id, name, kind, public_key, scopes, risk_ceiling,
                 created_at_ms, last_seen_ms, revoked_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, NULL)",
        )
        .bind(&device.device_id)
        .bind(&device.name)
        .bind(&device.kind)
        .bind(&device.public_key)
        .bind(&device.scopes)
        .bind(&device.risk_ceiling)
        .bind(to_i64(device.created_at_ms))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn device(&self, device_id: &str) -> Result<Option<DeviceRow>, Error> {
        let found = sqlx::query("SELECT * FROM client_devices WHERE device_id = ?1")
            .bind(device_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(found.as_ref().map(device_row))
    }

    /// Every device, revoked ones included, oldest first.
    pub async fn devices(&self) -> Result<Vec<DeviceRow>, Error> {
        let rows = sqlx::query("SELECT * FROM client_devices ORDER BY created_at_ms, device_id")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.iter().map(device_row).collect())
    }

    /// Revoke a device; returns whether it was active.
    pub async fn revoke_device(&self, device_id: &str, now_ms: u64) -> Result<bool, Error> {
        let done = sqlx::query(
            "UPDATE client_devices SET revoked_at_ms = ?2
             WHERE device_id = ?1 AND revoked_at_ms IS NULL",
        )
        .bind(device_id)
        .bind(to_i64(now_ms))
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected() == 1)
    }

    pub async fn touch_device(&self, device_id: &str, now_ms: u64) -> Result<(), Error> {
        sqlx::query("UPDATE client_devices SET last_seen_ms = ?2 WHERE device_id = ?1")
            .bind(device_id)
            .bind(to_i64(now_ms))
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

fn opt_u64(row: &sqlx::sqlite::SqliteRow, column: &str) -> Option<u64> {
    row.get::<Option<i64>, _>(column).map(to_u64)
}

fn pairing_row(row: &sqlx::sqlite::SqliteRow) -> PairingRow {
    PairingRow {
        offer_id: row.get("offer_id"),
        code_hash: row.get("code_hash"),
        origin_session: row.get("origin_session"),
        created_at_ms: to_u64(row.get("created_at_ms")),
        expires_at_ms: to_u64(row.get("expires_at_ms")),
        attempts: u32::try_from(row.get::<i64, _>("attempts")).unwrap_or(u32::MAX),
        status: row.get("status"),
        device_name: row.get("device_name"),
        device_kind: row.get("device_kind"),
        public_key: row.get("public_key"),
        scopes: row.get("scopes"),
        device_id: row.get("device_id"),
    }
}

fn device_row(row: &sqlx::sqlite::SqliteRow) -> DeviceRow {
    DeviceRow {
        device_id: row.get("device_id"),
        name: row.get("name"),
        kind: row.get("kind"),
        public_key: row.get("public_key"),
        scopes: row.get("scopes"),
        risk_ceiling: row.get("risk_ceiling"),
        created_at_ms: to_u64(row.get("created_at_ms")),
        last_seen_ms: opt_u64(row, "last_seen_ms"),
        revoked_at_ms: opt_u64(row, "revoked_at_ms"),
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
