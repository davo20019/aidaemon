//! # aidaemon-store-sqlite
//!
//! A SQLite implementation of [`aidaemon_core::RunStore`]: an append-only
//! event journal (`run_events`) plus the latest [`RunState`] snapshot per run
//! (`run_states`), with versioned migrations.
//!
//! Beyond the trait, [`SqliteStore`] offers query helpers an application
//! needs after a restart: [`SqliteStore::interrupted_runs`] (waiting on a
//! human decision, settle with `Agent::resume`) and
//! [`SqliteStore::in_flight_runs`] (stopped mid tool batch, continue with
//! `Agent::run_from`).
//!
//! ```no_run
//! # async fn demo() -> Result<(), aidaemon_store_sqlite::Error> {
//! use std::sync::Arc;
//! use aidaemon_store_sqlite::SqliteStore;
//!
//! let store = Arc::new(SqliteStore::open("runs.db").await?);
//! for state in store.in_flight_runs().await? {
//!     // agent.run_from(state, CancellationToken::new()).await
//!     # let _ = state;
//! }
//! # Ok(()) }
//! ```

mod migrations;

use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use aidaemon_core::{Event, RunId, RunState, RunStore, StoreError};
use async_trait::async_trait;
use serde_json::Value;
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};
use sqlx::Row;

pub use migrations::{latest_version, Migration, MIGRATIONS};

const NOW: &str = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')";

/// Errors from this crate. Converted to [`StoreError`] at the trait boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("sqlite: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "database schema version {found} is newer than this build supports ({supported}); \
         upgrade aidaemon-store-sqlite or use a different database file"
    )]
    SchemaTooNew { found: i64, supported: i64 },
    #[error("migration {version} ({name}) failed: {source}")]
    Migration {
        version: i64,
        name: &'static str,
        source: sqlx::Error,
    },
}

impl From<Error> for StoreError {
    fn from(err: Error) -> Self {
        StoreError(err.to_string())
    }
}

/// One journal row as stored.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredEvent {
    pub run_id: String,
    /// 1-based, gap-free, assigned atomically per run.
    pub seq: i64,
    /// The event's serde tag, e.g. `"tool_finished"`.
    pub event_type: String,
    pub payload: Value,
    /// RFC 3339 UTC timestamp with milliseconds.
    pub created_at: String,
}

impl StoredEvent {
    /// Decode the payload. Fails for event types this build does not know.
    pub fn event(&self) -> Result<Event, serde_json::Error> {
        serde_json::from_value(self.payload.clone())
    }
}

/// SQLite-backed [`RunStore`]. Cheap to clone (wraps a pool).
#[derive(Debug, Clone)]
pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    /// Open (creating if needed) a database file. Uses WAL,
    /// `synchronous=NORMAL`, foreign keys and a busy timeout, then migrates.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true)
            .busy_timeout(Duration::from_secs(10));
        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(options)
            .await?;
        Self::from_pool(pool).await
    }

    /// A private in-memory database (one connection, lives as long as the
    /// store). Handy for tests.
    pub async fn open_in_memory() -> Result<Self, Error> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")?.foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .min_connections(1)
            .idle_timeout(None)
            .max_lifetime(None)
            .connect_with(options)
            .await?;
        Self::from_pool(pool).await
    }

    /// Use an existing pool (its connection options are kept as given) and
    /// apply migrations.
    pub async fn from_pool(pool: SqlitePool) -> Result<Self, Error> {
        migrations::run(&pool).await?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// The schema version recorded in the database.
    pub async fn schema_version(&self) -> Result<i64, Error> {
        migrations::current_version(&self.pool).await
    }

    /// Append one event and return its sequence number. The number is
    /// computed inside the insert statement, which SQLite runs under the
    /// write lock, so concurrent appends to one run never collide or skip.
    pub async fn append(&self, run_id: RunId, event: &Event) -> Result<i64, Error> {
        let payload = serde_json::to_value(event)?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let run_id = run_id.to_string();
        let seq: i64 = sqlx::query_scalar(&format!(
            "INSERT INTO run_events (run_id, seq, event_type, payload, created_at)
             SELECT ?1, COALESCE(MAX(seq), 0) + 1, ?2, ?3, {NOW}
             FROM run_events WHERE run_id = ?1
             RETURNING seq"
        ))
        .bind(&run_id)
        .bind(event_type)
        .bind(payload.to_string())
        .fetch_one(&self.pool)
        .await?;
        Ok(seq)
    }

    /// The run's journal in order, decoded.
    pub async fn events(&self, run_id: RunId) -> Result<Vec<Event>, Error> {
        self.raw_events(run_id)
            .await?
            .iter()
            .map(|e| e.event().map_err(Error::from))
            .collect()
    }

    /// The run's journal in order, as stored.
    pub async fn raw_events(&self, run_id: RunId) -> Result<Vec<StoredEvent>, Error> {
        let rows = sqlx::query(
            "SELECT run_id, seq, event_type, payload, created_at
             FROM run_events WHERE run_id = ? ORDER BY seq",
        )
        .bind(run_id.to_string())
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                let payload: String = row.try_get("payload")?;
                Ok(StoredEvent {
                    run_id: row.try_get("run_id")?,
                    seq: row.try_get("seq")?,
                    event_type: row.try_get("event_type")?,
                    payload: serde_json::from_str(&payload)?,
                    created_at: row.try_get("created_at")?,
                })
            })
            .collect()
    }

    /// Insert or replace the snapshot for `state.run_id`.
    pub async fn upsert_state(&self, state: &RunState) -> Result<(), Error> {
        let json = serde_json::to_string(state)?;
        sqlx::query(&format!(
            "INSERT INTO run_states (run_id, agent_id, session_id, state, updated_at)
             VALUES (?, ?, ?, ?, {NOW})
             ON CONFLICT (run_id) DO UPDATE SET
                agent_id = excluded.agent_id,
                session_id = excluded.session_id,
                state = excluded.state,
                updated_at = excluded.updated_at"
        ))
        .bind(state.run_id.to_string())
        .bind(&state.agent_id)
        .bind(&state.session_id)
        .bind(json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// The latest snapshot of one run.
    pub async fn state(&self, run_id: RunId) -> Result<Option<RunState>, Error> {
        let row: Option<String> =
            sqlx::query_scalar("SELECT state FROM run_states WHERE run_id = ?")
                .bind(run_id.to_string())
                .fetch_optional(&self.pool)
                .await?;
        row.map(|s| serde_json::from_str(&s).map_err(Error::from))
            .transpose()
    }

    /// Every run of a session, least recently updated first.
    pub async fn runs_for_session(&self, session_id: &str) -> Result<Vec<RunState>, Error> {
        self.states_where("session_id = ?", Some(session_id), false, None)
            .await
    }

    /// The most recently updated run of a session.
    pub async fn latest_state_for_session(
        &self,
        session_id: &str,
    ) -> Result<Option<RunState>, Error> {
        let mut states = self
            .states_where("session_id = ?", Some(session_id), true, Some(1))
            .await?;
        Ok(states.pop())
    }

    /// Runs paused on an interruption (`pending` set). Settle each with
    /// `Agent::resume`.
    pub async fn interrupted_runs(&self) -> Result<Vec<RunState>, Error> {
        self.states_where(
            "json_extract(state, '$.pending') IS NOT NULL",
            None,
            false,
            None,
        )
        .await
    }

    /// Runs whose last checkpoint was taken mid tool batch (`in_flight`
    /// set): the process stopped while tools ran. Continue each with
    /// `Agent::run_from`, which re-runs idempotent calls and marks the rest
    /// indeterminate.
    pub async fn in_flight_runs(&self) -> Result<Vec<RunState>, Error> {
        self.states_where(
            "json_extract(state, '$.in_flight') IS NOT NULL",
            None,
            false,
            None,
        )
        .await
    }

    async fn states_where(
        &self,
        predicate: &str,
        bind: Option<&str>,
        newest_first: bool,
        limit: Option<u32>,
    ) -> Result<Vec<RunState>, Error> {
        // `predicate` is a crate-internal constant, never caller input.
        let dir = if newest_first { "DESC" } else { "ASC" };
        let limit = limit.map(|n| format!(" LIMIT {n}")).unwrap_or_default();
        let sql = format!(
            "SELECT state FROM run_states WHERE {predicate}
             ORDER BY updated_at {dir}, rowid {dir}{limit}"
        );
        let mut query = sqlx::query_scalar::<_, String>(&sql);
        if let Some(value) = bind {
            query = query.bind(value);
        }
        query
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(|s| serde_json::from_str(s).map_err(Error::from))
            .collect()
    }

    /// Close the pool, checkpointing the WAL.
    pub async fn close(&self) {
        self.pool.close().await;
    }
}

#[async_trait]
impl RunStore for SqliteStore {
    async fn append_event(&self, run_id: RunId, event: &Event) -> Result<(), StoreError> {
        self.append(run_id, event).await?;
        Ok(())
    }

    async fn save_state(&self, state: &RunState) -> Result<(), StoreError> {
        Ok(self.upsert_state(state).await?)
    }

    async fn load_state(&self, run_id: RunId) -> Result<Option<RunState>, StoreError> {
        Ok(self.state(run_id).await?)
    }
}

/// Erase every run of one session from the journal and the checkpoints, in
/// the caller's transaction (a `/wipe` deletes conversation text atomically).
///
/// `run_events` is append-only: a trigger aborts every `DELETE`. This is the
/// one sanctioned exception. It drops the guard, deletes the session's rows
/// and restores the guard inside the same transaction, so no other delete can
/// slip through. The trigger body must match migration 1.
pub async fn erase_session_in_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    session_id: &str,
) -> Result<(), Error> {
    sqlx::query("DROP TRIGGER IF EXISTS run_events_no_delete")
        .execute(&mut **tx)
        .await?;
    sqlx::query(
        "DELETE FROM run_events
         WHERE run_id IN (SELECT run_id FROM run_states WHERE session_id = ?)",
    )
    .bind(session_id)
    .execute(&mut **tx)
    .await?;
    sqlx::query(
        "CREATE TRIGGER run_events_no_delete BEFORE DELETE ON run_events
         BEGIN SELECT RAISE(ABORT, 'run_events is append-only'); END",
    )
    .execute(&mut **tx)
    .await?;
    sqlx::query("DELETE FROM run_states WHERE session_id = ?")
        .bind(session_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}
