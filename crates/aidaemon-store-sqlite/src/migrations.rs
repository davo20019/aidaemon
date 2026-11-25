//! Versioned, embedded schema migrations.
//!
//! Each migration runs in its own transaction together with the row that
//! records it in `schema_migrations`, so a crash mid-migration leaves the
//! database at the previous version. Migrations are append-only: never edit
//! a released entry, add a new one with the next version instead.

use sqlx::SqlitePool;

use crate::Error;

/// One schema step.
#[derive(Debug, Clone, Copy)]
pub struct Migration {
    /// Strictly increasing, starting at 1.
    pub version: i64,
    pub name: &'static str,
    pub sql: &'static str,
}

/// Every migration this build knows, in order.
pub const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "run_events_and_run_states",
        sql: r#"
CREATE TABLE run_events (
    run_id     TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    event_type TEXT    NOT NULL,
    payload    TEXT    NOT NULL CHECK (json_valid(payload)),
    created_at TEXT    NOT NULL,
    PRIMARY KEY (run_id, seq)
) WITHOUT ROWID;

CREATE TABLE run_states (
    run_id     TEXT PRIMARY KEY NOT NULL,
    agent_id   TEXT NOT NULL,
    session_id TEXT NOT NULL,
    state      TEXT NOT NULL CHECK (json_valid(state)),
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_run_states_session ON run_states (session_id, updated_at);

-- The journal is append-only.
CREATE TRIGGER run_events_no_update BEFORE UPDATE ON run_events
BEGIN SELECT RAISE(ABORT, 'run_events is append-only'); END;
CREATE TRIGGER run_events_no_delete BEFORE DELETE ON run_events
BEGIN SELECT RAISE(ABORT, 'run_events is append-only'); END;
"#,
    },
    Migration {
        version: 2,
        name: "channel_approvals",
        sql: r#"
-- Open `ChannelRunner` approvals (`SqliteApprovalStore`). The scope columns
-- mirror the JSON so they can be queried and indexed; `record` is the whole
-- `StoredApproval` and is what is loaded back.
CREATE TABLE channel_approvals (
    prompt_id       TEXT    PRIMARY KEY NOT NULL,
    session_id      TEXT    NOT NULL,
    channel         TEXT    NOT NULL,
    conversation_id TEXT    NOT NULL,
    expires_at_ms   INTEGER NOT NULL,
    record          TEXT    NOT NULL CHECK (json_valid(record)),
    saved_at        TEXT    NOT NULL
);

CREATE INDEX idx_channel_approvals_session ON channel_approvals (session_id);
"#,
    },
];

/// The newest schema version this build understands.
pub fn latest_version() -> i64 {
    MIGRATIONS.last().map(|m| m.version).unwrap_or(0)
}

/// The version recorded in the database (0 for a fresh one).
pub async fn current_version(pool: &SqlitePool) -> Result<i64, Error> {
    Ok(recorded_version(pool, VERSION_TABLE).await?)
}

/// Bring the schema up to [`latest_version`]. Idempotent, safe to call from
/// several processes at once, and refuses databases written by newer code.
pub(crate) async fn run(pool: &SqlitePool) -> Result<(), Error> {
    apply(pool, VERSION_TABLE, MIGRATIONS)
        .await
        .map_err(Into::into)
}

/// The table this store records its versions in.
const VERSION_TABLE: &str = "schema_migrations";

/// Why [`apply`] stopped.
#[derive(Debug, thiserror::Error)]
pub enum MigrateError {
    #[error("sqlite: {0}")]
    Sqlx(#[from] sqlx::Error),
    /// The database was written by a newer build.
    #[error("schema version {found} is newer than this build supports ({supported})")]
    SchemaTooNew { found: i64, supported: i64 },
    /// One step failed; the database stays at the version before it.
    #[error("migration {version} ({name}) failed: {source}")]
    Step {
        version: i64,
        name: &'static str,
        source: sqlx::Error,
    },
}

impl From<MigrateError> for Error {
    fn from(error: MigrateError) -> Self {
        match error {
            MigrateError::Sqlx(error) => Error::Sqlx(error),
            MigrateError::SchemaTooNew { found, supported } => {
                Error::SchemaTooNew { found, supported }
            }
            MigrateError::Step {
                version,
                name,
                source,
            } => Error::Migration {
                version,
                name,
                source,
            },
        }
    }
}

/// The highest version recorded in `table` (0 for none).
pub async fn recorded_version(pool: &SqlitePool, table: &str) -> sqlx::Result<i64> {
    let version: Option<i64> = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
        "SELECT MAX(version) FROM {table}"
    )))
    .fetch_one(pool)
    .await?;
    Ok(version.unwrap_or(0))
}

/// Apply `migrations` (strictly increasing versions from 1) that `table`
/// has not recorded, each in its own `BEGIN IMMEDIATE` transaction with the
/// row that records it, and refuse a database a newer build wrote. A store
/// sharing the database file keeps its own `table`, so the version
/// sequences never collide. Idempotent and safe from several processes.
pub async fn apply(
    pool: &SqlitePool,
    table: &'static str,
    migrations: &[Migration],
) -> Result<(), MigrateError> {
    debug_assert!(
        migrations
            .windows(2)
            .all(|w| w[1].version == w[0].version + 1)
    );
    sqlx::query(sqlx::AssertSqlSafe(format!(
        "CREATE TABLE IF NOT EXISTS {table} (
            version    INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )"
    )))
    .execute(pool)
    .await?;

    let found = recorded_version(pool, table).await?;
    let supported = migrations.last().map(|m| m.version).unwrap_or(0);
    if found > supported {
        return Err(MigrateError::SchemaTooNew { found, supported });
    }
    let claim = format!(
        "INSERT OR IGNORE INTO {table} (version, applied_at)
         VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
    );
    for migration in migrations.iter().filter(|m| m.version > found) {
        let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
        // Claim the version first: this takes the write lock, and if another
        // process applied it meanwhile the insert is ignored and we skip.
        let claimed = sqlx::query(sqlx::AssertSqlSafe(claim.as_str()))
            .bind(migration.version)
            .execute(&mut *tx)
            .await?
            .rows_affected();
        if claimed == 0 {
            tx.rollback().await?;
            continue;
        }
        sqlx::Executor::execute(&mut *tx, migration.sql)
            .await
            .map_err(|source| MigrateError::Step {
                version: migration.version,
                name: migration.name,
                source,
            })?;
        tx.commit().await?;
    }
    Ok(())
}

/// Compile-time check: opening must stay spawnable. `sqlx::raw_sql` on a
/// transaction trips a higher-ranked lifetime bug that makes the future
/// `!Send`, which is why the migration body uses `Executor::execute`.
#[allow(dead_code)]
fn assert_send(pool: SqlitePool) {
    fn is_send<T: Send>(_: T) {}
    is_send(run(&pool));
    is_send(crate::SqliteStore::open("x"));
}
