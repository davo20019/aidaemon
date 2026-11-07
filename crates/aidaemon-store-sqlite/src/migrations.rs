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
pub const MIGRATIONS: &[Migration] = &[Migration {
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
}];

/// The newest schema version this build understands.
pub fn latest_version() -> i64 {
    MIGRATIONS.last().map(|m| m.version).unwrap_or(0)
}

/// The version recorded in the database (0 for a fresh one).
pub async fn current_version(pool: &SqlitePool) -> Result<i64, Error> {
    let version: Option<i64> = sqlx::query_scalar("SELECT MAX(version) FROM schema_migrations")
        .fetch_one(pool)
        .await?;
    Ok(version.unwrap_or(0))
}

/// Bring the schema up to [`latest_version`]. Idempotent, safe to call from
/// several processes at once, and refuses databases written by newer code.
pub(crate) async fn run(pool: &SqlitePool) -> Result<(), Error> {
    debug_assert!(MIGRATIONS
        .windows(2)
        .all(|w| w[1].version == w[0].version + 1));

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    let found = current_version(pool).await?;
    let supported = latest_version();
    if found > supported {
        return Err(Error::SchemaTooNew { found, supported });
    }

    for migration in MIGRATIONS.iter().filter(|m| m.version > found) {
        let mut tx = pool.begin().await?;
        // Claim the version first: this takes the write lock, and if another
        // process applied it meanwhile the insert is ignored and we skip.
        let claimed = sqlx::query(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at)
             VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
        )
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
            .map_err(|source| Error::Migration {
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
