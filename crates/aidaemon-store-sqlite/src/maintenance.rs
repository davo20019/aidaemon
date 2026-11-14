//! Upkeep for a long-lived SQLite database: query-planner statistics and
//! consistent snapshots.
//!
//! Both work on any pool, so an application that keeps its own tables in the
//! same file as the run store (as the aidaemon daemon does) maintains the
//! whole file with one call.
//!
//! ```no_run
//! # async fn demo(pool: sqlx::SqlitePool) -> Result<(), aidaemon_store_sqlite::Error> {
//! use aidaemon_store_sqlite::maintenance::{optimize, snapshot, Optimize};
//!
//! optimize(&pool, Optimize::AtOpen).await?;       // once, after migrations
//! optimize(&pool, Optimize::Periodic).await?;     // every few hours
//! snapshot(&pool, "backups/app.db".as_ref()).await?;
//! # Ok(()) }
//! ```

use std::path::Path;

use sqlx::SqlitePool;

use crate::Error;

/// Rows sampled per index when statistics are gathered. Bounds the cost of
/// an analysis on a large table to milliseconds, as SQLite recommends for
/// `PRAGMA optimize`; the estimates stay good enough to pick indexes.
pub const ANALYSIS_LIMIT: u32 = 1000;

/// When [`optimize`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimize {
    /// Right after opening and migrating: examines every table, including
    /// ones never queried yet on this connection (`PRAGMA optimize=0x10002`).
    AtOpen,
    /// On a timer while the database is in use: refreshes statistics only
    /// for tables whose size changed a lot since they were last analyzed.
    Periodic,
}

/// Gather the query planner's statistics where they are missing or stale.
///
/// Without them SQLite picks indexes by rule of thumb, which goes wrong on
/// tables that have grown unevenly. The pass is bounded by
/// [`ANALYSIS_LIMIT`], and when nothing changed it does nothing.
pub async fn optimize(pool: &SqlitePool, when: Optimize) -> Result<(), Error> {
    // Both pragmas must run on the same connection: analysis_limit is
    // per connection.
    let mut connection = pool.acquire().await?;
    sqlx::query(sqlx::AssertSqlSafe(format!(
        "PRAGMA analysis_limit = {ANALYSIS_LIMIT}"
    )))
    .execute(&mut *connection)
    .await?;
    let statement = match when {
        Optimize::AtOpen => "PRAGMA optimize = 0x10002",
        Optimize::Periodic => "PRAGMA optimize",
    };
    sqlx::query(statement).execute(&mut *connection).await?;
    Ok(())
}

/// Write a consistent, compacted copy of the database to `destination` with
/// `VACUUM INTO`, while other connections keep reading and writing.
///
/// Unlike copying the file, the copy includes what is still in the WAL and
/// never catches a half-written page. With SQLCipher the copy is encrypted
/// with the source's key. `destination` must not exist.
pub async fn snapshot(pool: &SqlitePool, destination: &Path) -> Result<(), Error> {
    let destination = destination
        .to_str()
        .ok_or_else(|| sqlx::Error::Configuration("snapshot path is not valid UTF-8".into()))?;
    sqlx::query("VACUUM INTO ?")
        .bind(destination)
        .execute(pool)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};

    async fn file_pool(path: &Path) -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(
                SqliteConnectOptions::new()
                    .filename(path)
                    .create_if_missing(true)
                    .journal_mode(SqliteJournalMode::Wal),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn optimize_gathers_statistics_for_an_indexed_table() {
        let dir = tempfile::tempdir().unwrap();
        let pool = file_pool(&dir.path().join("a.db")).await;
        sqlx::query("CREATE TABLE items (id INTEGER PRIMARY KEY, kind TEXT)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE INDEX items_kind ON items(kind)")
            .execute(&pool)
            .await
            .unwrap();
        for i in 0..2000 {
            sqlx::query("INSERT INTO items (kind) VALUES (?)")
                .bind(format!("kind-{}", i % 7))
                .execute(&pool)
                .await
                .unwrap();
        }
        optimize(&pool, Optimize::AtOpen).await.unwrap();
        let analyzed: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM sqlite_stat1 WHERE tbl = 'items'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(analyzed > 0, "optimize should have analyzed items");
        optimize(&pool, Optimize::Periodic).await.unwrap();
    }

    #[tokio::test]
    async fn snapshot_includes_uncheckpointed_writes() {
        let dir = tempfile::tempdir().unwrap();
        let pool = file_pool(&dir.path().join("live.db")).await;
        sqlx::query("CREATE TABLE notes (body TEXT)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO notes VALUES ('synthetic note')")
            .execute(&pool)
            .await
            .unwrap();
        let copy = dir.path().join("copy.db");
        snapshot(&pool, &copy).await.unwrap();

        let copy_pool = file_pool(&copy).await;
        let body: String = sqlx::query_scalar("SELECT body FROM notes")
            .fetch_one(&copy_pool)
            .await
            .unwrap();
        assert_eq!(body, "synthetic note");
        assert!(snapshot(&pool, &copy).await.is_err(), "never overwrites");
    }
}
