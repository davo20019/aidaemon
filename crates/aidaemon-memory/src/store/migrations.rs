//! Versioned, embedded schema migrations for [`super::SqliteMemoryStore`].
//!
//! Versions are recorded in `memory_schema_migrations` (not the generic
//! `schema_migrations`) so this store can share a database file with
//! `aidaemon-store-sqlite` without the two version sequences colliding.
//! Each migration runs in one transaction with its version row. Migrations
//! are append-only.

use sqlx::SqlitePool;

use crate::error::{MemoryError, Result};

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
    name: "facts_episodes_fts",
    sql: r#"
CREATE TABLE memory_facts (
    id                TEXT PRIMARY KEY NOT NULL,
    scope             TEXT NOT NULL,
    category          TEXT NOT NULL,
    key               TEXT NOT NULL,
    value             TEXT NOT NULL,
    confidence        REAL NOT NULL,
    source_run_id     TEXT,
    source_session_id TEXT,
    source_excerpt    TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    superseded_at     TEXT,
    recall_count      INTEGER NOT NULL DEFAULT 0,
    last_recalled_at  TEXT,
    embedding         BLOB,
    embedding_model   TEXT
);

-- At most one active row per key; superseded rows are history.
CREATE UNIQUE INDEX idx_memory_facts_active
    ON memory_facts (scope, category, key) WHERE superseded_at IS NULL;
CREATE INDEX idx_memory_facts_history ON memory_facts (scope, category, key, created_at);
CREATE INDEX idx_memory_facts_superseded ON memory_facts (superseded_at)
    WHERE superseded_at IS NOT NULL;

-- Lexical index over ACTIVE facts only, maintained in the same transaction
-- as every write.
CREATE VIRTUAL TABLE memory_facts_fts USING fts5(
    fact_id UNINDEXED,
    scope UNINDEXED,
    content,
    tokenize = 'unicode61'
);

CREATE TABLE memory_episodes (
    id               TEXT PRIMARY KEY NOT NULL,
    scope            TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    summary          TEXT NOT NULL,
    topics           TEXT NOT NULL CHECK (json_valid(topics)),
    importance       REAL NOT NULL,
    created_at       TEXT NOT NULL,
    recall_count     INTEGER NOT NULL DEFAULT 0,
    last_recalled_at TEXT,
    embedding        BLOB,
    embedding_model  TEXT
);

CREATE INDEX idx_memory_episodes_scope ON memory_episodes (scope, created_at);

CREATE VIRTUAL TABLE memory_episodes_fts USING fts5(
    episode_id UNINDEXED,
    scope UNINDEXED,
    content,
    tokenize = 'unicode61'
);
"#,
}];

/// The newest schema version this build understands.
pub fn latest_version() -> i64 {
    MIGRATIONS.last().map(|m| m.version).unwrap_or(0)
}

pub(crate) async fn current_version(pool: &SqlitePool) -> Result<i64> {
    let version: Option<i64> =
        sqlx::query_scalar("SELECT MAX(version) FROM memory_schema_migrations")
            .fetch_one(pool)
            .await?;
    Ok(version.unwrap_or(0))
}

/// Bring the schema up to date. Idempotent and refuses newer databases.
pub(crate) async fn run(pool: &SqlitePool) -> Result<()> {
    debug_assert!(MIGRATIONS
        .windows(2)
        .all(|w| w[1].version == w[0].version + 1));

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_schema_migrations (
            version    INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    let found = current_version(pool).await?;
    let supported = latest_version();
    if found > supported {
        return Err(MemoryError::SchemaTooNew { found, supported });
    }

    for migration in MIGRATIONS.iter().filter(|m| m.version > found) {
        let mut tx = pool.begin().await?;
        let claimed = sqlx::query(
            "INSERT OR IGNORE INTO memory_schema_migrations (version, applied_at)
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
            .map_err(|source| MemoryError::Migration {
                version: migration.version,
                name: migration.name,
                source,
            })?;
        tx.commit().await?;
    }
    Ok(())
}
