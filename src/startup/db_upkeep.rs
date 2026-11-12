//! Database upkeep around startup: a snapshot before an upgrade migrates the
//! database, and planner statistics once the schema is current.
//!
//! Nothing here is required for an upgrade: every schema change is applied
//! by the migrations at startup, so users never run a script. The snapshot
//! is the way back if a new version's migration goes wrong.

use std::path::{Path, PathBuf};
use std::time::Instant;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::config::StateConfig;

/// The `settings` key holding the version that last started on this
/// database.
const LAST_STARTED_VERSION: &str = "aidaemon_last_started_version";

fn current_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Before migrations run: if this database was last started by another
/// aidaemon version (or by one from before versions were recorded), write a
/// consistent copy to `backups/` beside it and prune older ones down to
/// `state.upgrade_backups`. A new database, the same version, or
/// `upgrade_backups = 0` writes nothing.
///
/// A failed snapshot is logged and startup continues: it is a safety net,
/// not a precondition.
pub async fn snapshot_before_upgrade(state: &StateConfig) -> Option<PathBuf> {
    if state.upgrade_backups == 0 {
        return None;
    }
    let db_path = Path::new(&state.db_path);
    if !db_path.exists() {
        return None;
    }
    match try_snapshot_before_upgrade(state, db_path).await {
        Ok(path) => path,
        Err(error) => {
            warn!(%error, "Could not snapshot the database before upgrading; continuing");
            None
        }
    }
}

async fn try_snapshot_before_upgrade(
    state: &StateConfig,
    db_path: &Path,
) -> anyhow::Result<Option<PathBuf>> {
    let pool = open(db_path, state.encryption_key.as_deref()).await?;
    let previous = started_version(&pool).await?;
    let Some(previous) = previous else {
        // No settings table: a database no aidaemon has migrated yet.
        pool.close().await;
        return Ok(None);
    };
    if previous.as_deref() == Some(current_version()) {
        pool.close().await;
        return Ok(None);
    }

    let directory = db_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .join("backups");
    create_private_dir(&directory)?;
    let stem = db_file_stem(db_path);
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let destination = directory.join(format!("{stem}.pre-v{}-{stamp}.db", current_version()));

    let started = Instant::now();
    aidaemon_store_sqlite::maintenance::snapshot(&pool, &destination).await?;
    pool.close().await;
    set_private(&destination)?;
    let bytes = std::fs::metadata(&destination)
        .map(|meta| meta.len())
        .unwrap_or(0);
    info!(
        from = previous.as_deref().unwrap_or("unrecorded"),
        to = current_version(),
        path = %destination.display(),
        bytes,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "Snapshotted the database before upgrading"
    );
    prune(&directory, &stem, state.upgrade_backups);
    Ok(Some(destination))
}

/// After migrations: remember that this version has started here.
pub async fn record_started_version(pool: &SqlitePool) {
    let result = sqlx::query(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
    )
    .bind(LAST_STARTED_VERSION)
    .bind(current_version())
    .execute(pool)
    .await;
    if let Err(error) = result {
        warn!(%error, "Could not record the started version");
    }
}

/// After migrations: refresh the query planner's statistics. The first run
/// on a database that was never analyzed examines every table, bounded per
/// index by `ANALYSIS_LIMIT`; later runs do nothing unless tables changed.
pub async fn optimize_at_open(pool: &SqlitePool) {
    let started = Instant::now();
    match aidaemon_store_sqlite::maintenance::optimize(
        pool,
        aidaemon_store_sqlite::maintenance::Optimize::AtOpen,
    )
    .await
    {
        Ok(()) => info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "SQLite planner statistics refreshed"
        ),
        Err(error) => warn!(%error, "SQLite optimize failed"),
    }
}

/// `Ok(None)`: no settings table. `Ok(Some(None))`: an aidaemon database
/// from before versions were recorded.
async fn started_version(pool: &SqlitePool) -> anyhow::Result<Option<Option<String>>> {
    let has_settings: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'settings'",
    )
    .fetch_optional(pool)
    .await?;
    if has_settings.is_none() {
        return Ok(None);
    }
    let version: Option<String> = sqlx::query_scalar("SELECT value FROM settings WHERE key = ?")
        .bind(LAST_STARTED_VERSION)
        .fetch_optional(pool)
        .await?;
    Ok(Some(version))
}

async fn open(db_path: &Path, encryption_key: Option<&str>) -> anyhow::Result<SqlitePool> {
    let key = encryption_key
        .map(str::to_string)
        .or_else(|| std::env::var("AIDAEMON_ENCRYPTION_KEY").ok())
        .filter(|key| !key.trim().is_empty());
    let mut options = SqliteConnectOptions::new()
        .filename(db_path)
        .busy_timeout(std::time::Duration::from_secs(30));
    #[cfg(feature = "encryption")]
    if let Some(key) = &key {
        options = options.pragma("key", format!("'{}'", key.trim().replace('\'', "''")));
    }
    #[cfg(not(feature = "encryption"))]
    let _ = key;
    Ok(SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await?)
}

fn db_file_stem(db_path: &Path) -> String {
    db_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("aidaemon.db")
        .trim_end_matches(".db")
        .to_string()
}

/// Keep the `keep` newest pre-upgrade snapshots of this database.
fn prune(directory: &Path, stem: &str, keep: usize) {
    let prefix = format!("{stem}.pre-v");
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    let mut snapshots: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".db"))
        })
        .filter_map(|entry| {
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((modified, entry.path()))
        })
        .collect();
    snapshots.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    for (_, path) in snapshots.into_iter().skip(keep) {
        match std::fs::remove_file(&path) {
            Ok(()) => info!(path = %path.display(), "Removed an older pre-upgrade snapshot"),
            Err(error) => warn!(%error, path = %path.display(), "Could not remove an old snapshot"),
        }
    }
}

fn create_private_dir(directory: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn set_private(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(db_path: &Path, keep: usize) -> StateConfig {
        StateConfig {
            db_path: db_path.to_str().unwrap().to_string(),
            upgrade_backups: keep,
            ..StateConfig::default()
        }
    }

    async fn database(path: &Path, version: Option<&str>) {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(
                SqliteConnectOptions::new()
                    .filename(path)
                    .create_if_missing(true),
            )
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,
             updated_at TEXT NOT NULL DEFAULT (datetime('now')))",
        )
        .execute(&pool)
        .await
        .unwrap();
        if let Some(version) = version {
            sqlx::query("INSERT INTO settings (key, value) VALUES (?, ?)")
                .bind(LAST_STARTED_VERSION)
                .bind(version)
                .execute(&pool)
                .await
                .unwrap();
        }
        pool.close().await;
    }

    fn snapshots(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir.join("backups"))
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    #[tokio::test]
    async fn an_upgrade_snapshots_once_and_the_same_version_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("state.db");
        database(&db, Some("0.0.1")).await;

        let copy = snapshot_before_upgrade(&config(&db, 2)).await.unwrap();
        assert!(copy.exists());
        let name = copy.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with(&format!("state.pre-v{}-", current_version())));

        let pool = open(&db, None).await.unwrap();
        record_started_version(&pool).await;
        pool.close().await;
        assert!(snapshot_before_upgrade(&config(&db, 2)).await.is_none());
        assert_eq!(snapshots(dir.path()).len(), 1);
    }

    #[tokio::test]
    async fn a_database_from_before_versions_were_recorded_counts_as_an_upgrade() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("state.db");
        database(&db, None).await;
        assert!(snapshot_before_upgrade(&config(&db, 2)).await.is_some());
    }

    #[tokio::test]
    async fn new_databases_and_a_zero_setting_write_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("state.db");
        assert!(snapshot_before_upgrade(&config(&db, 2)).await.is_none());

        // A file with no aidaemon tables yet.
        SqlitePoolOptions::new()
            .connect_with(
                SqliteConnectOptions::new()
                    .filename(&db)
                    .create_if_missing(true),
            )
            .await
            .unwrap()
            .close()
            .await;
        assert!(snapshot_before_upgrade(&config(&db, 2)).await.is_none());

        database(&dir.path().join("old.db"), Some("0.0.1")).await;
        assert!(
            snapshot_before_upgrade(&config(&dir.path().join("old.db"), 0))
                .await
                .is_none()
        );
        assert!(snapshots(dir.path()).is_empty());
    }

    #[test]
    fn pruning_keeps_the_newest_of_this_database_only() {
        let dir = tempfile::tempdir().unwrap();
        let backups = dir.path().join("backups");
        std::fs::create_dir_all(&backups).unwrap();
        for (i, name) in [
            "state.pre-v0.1.0-a.db",
            "state.pre-v0.2.0-b.db",
            "state.pre-v0.3.0-c.db",
            "other.pre-v0.1.0-a.db",
            "state.db.unrelated",
        ]
        .iter()
        .enumerate()
        {
            let path = backups.join(name);
            std::fs::write(&path, b"x").unwrap();
            let time =
                std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(i as u64 * 60);
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(time)
                .unwrap();
        }
        prune(&backups, "state", 2);
        assert_eq!(
            snapshots(dir.path()),
            [
                "other.pre-v0.1.0-a.db",
                "state.db.unrelated",
                "state.pre-v0.2.0-b.db",
                "state.pre-v0.3.0-c.db"
            ]
        );
    }
}

/// Measures upkeep on a real database. Ignored; point it at the daemon's
/// database (the file is only read) and an empty output directory:
///
/// ```text
/// AIDAEMON_BENCH_DB=aidaemon.db AIDAEMON_BENCH_OUT=/tmp/upkeep \
///   cargo test --release --features encryption --lib upkeep_bench -- --ignored --nocapture
/// ```
///
/// It snapshots the live file (timing `VACUUM INTO`), then on the copy times
/// hot daemon queries, runs `optimize`, and times them again.
#[cfg(test)]
mod upkeep_bench {
    use super::*;
    use std::time::Duration;

    const QUERIES: &[(&str, &str)] = &[
        (
            "procedures_missing_embedding",
            "SELECT id, trigger_pattern FROM procedures p
             WHERE trigger_pattern IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM memory_embeddings me
                   WHERE me.owner_type = 'procedure' AND me.owner_id = CAST(p.id AS TEXT)
                     AND me.embedding_purpose = 'trigger' AND me.embedding_model = ?1
                     AND me.stale_at IS NULL)
             LIMIT 10",
        ),
        (
            "orphan_embeddings",
            "SELECT COUNT(*) FROM memory_embeddings
             WHERE (owner_type = 'claim' AND NOT EXISTS (
                       SELECT 1 FROM memory_claims c
                       WHERE c.id = CAST(owner_id AS INTEGER) AND CAST(c.id AS TEXT) = owner_id))
                OR (owner_type = 'span' AND NOT EXISTS (
                       SELECT 1 FROM memory_spans s
                       WHERE s.id = CAST(owner_id AS INTEGER) AND CAST(s.id AS TEXT) = owner_id))
                OR (stale_at IS NOT NULL AND stale_at < datetime('now', '-30 days'))",
        ),
        (
            "session_history",
            "SELECT id, session_id, event_type, data, created_at FROM events
             WHERE session_id = ?2
               AND event_type IN ('user_message', 'assistant_response', 'tool_result')
             ORDER BY created_at DESC LIMIT 200",
        ),
        (
            "recent_task_lifecycle",
            "SELECT session_id, task_id, state, status FROM task_lifecycle
             WHERE session_id = ?2 ORDER BY started_at DESC LIMIT 20",
        ),
        (
            "active_claims_by_privacy",
            "SELECT COUNT(*) FROM memory_claims
             WHERE privacy = 'global' AND deleted_at IS NULL AND valid_to IS NULL",
        ),
        (
            "pending_notifications",
            "SELECT id FROM notification_queue WHERE session_id = ?2
             ORDER BY created_at DESC LIMIT 20",
        ),
    ];

    async fn time_queries(
        pool: &SqlitePool,
        model: &str,
        session: &str,
    ) -> Vec<(String, Duration, String)> {
        let mut out = Vec::new();
        for (name, sql) in QUERIES {
            let mut samples = Vec::new();
            for _ in 0..7 {
                let started = Instant::now();
                sqlx::query(*sql)
                    .bind(model)
                    .bind(session)
                    .fetch_all(pool)
                    .await
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                samples.push(started.elapsed());
            }
            samples.sort();
            let rows = sqlx::query(sqlx::AssertSqlSafe(format!("EXPLAIN QUERY PLAN {sql}")))
                .bind(model)
                .bind(session)
                .fetch_all(pool)
                .await
                .unwrap();
            let plan = rows
                .iter()
                .map(|row| sqlx::Row::get::<String, _>(row, "detail"))
                .collect::<Vec<_>>()
                .join(" | ");
            out.push((name.to_string(), samples[samples.len() / 2], plan));
        }
        out
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "benchmark on a real database: set AIDAEMON_BENCH_DB and AIDAEMON_BENCH_OUT"]
    async fn upkeep_bench() {
        let source = PathBuf::from(std::env::var("AIDAEMON_BENCH_DB").expect("AIDAEMON_BENCH_DB"));
        let out = PathBuf::from(std::env::var("AIDAEMON_BENCH_OUT").expect("AIDAEMON_BENCH_OUT"));
        std::fs::create_dir_all(&out).unwrap();
        let key = std::env::var("AIDAEMON_ENCRYPTION_KEY").ok();

        let live = {
            let key = key.clone().filter(|k| !k.trim().is_empty());
            let mut options = SqliteConnectOptions::new()
                .filename(&source)
                .read_only(true)
                .busy_timeout(Duration::from_secs(30));
            #[cfg(feature = "encryption")]
            if let Some(key) = &key {
                options = options.pragma("key", format!("'{}'", key.trim().replace('\'', "''")));
            }
            #[cfg(not(feature = "encryption"))]
            let _ = key;
            SqlitePoolOptions::new()
                .max_connections(1)
                .connect_with(options)
                .await
                .unwrap()
        };
        let copy = out.join("snapshot.db");
        let _ = std::fs::remove_file(&copy);
        let started = Instant::now();
        aidaemon_store_sqlite::maintenance::snapshot(&live, &copy)
            .await
            .unwrap();
        let snapshot_time = started.elapsed();
        live.close().await;
        let wal = source.with_extension("db-wal");
        println!(
            "snapshot source_bytes={} source_wal_bytes={} copy_bytes={} elapsed={snapshot_time:.2?}",
            std::fs::metadata(&source).unwrap().len(),
            std::fs::metadata(&wal).map(|m| m.len()).unwrap_or(0),
            std::fs::metadata(&copy).unwrap().len()
        );

        // Start from no statistics, as a database that was never analyzed
        // (the live file may already have them), then reopen so the planner
        // forgets any it loaded.
        let pool = open(&copy, key.as_deref()).await.unwrap();
        let had_stats: Option<i64> =
            sqlx::query_scalar("SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'")
                .fetch_optional(&pool)
                .await
                .unwrap();
        if had_stats.is_some() {
            sqlx::query("DELETE FROM sqlite_stat1")
                .execute(&pool)
                .await
                .unwrap();
        }
        pool.close().await;
        let pool = open(&copy, key.as_deref()).await.unwrap();
        let model: String = sqlx::query_scalar(
            "SELECT embedding_model FROM memory_embeddings GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let session: String = sqlx::query_scalar(
            "SELECT session_id FROM events GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let before = time_queries(&pool, &model, &session).await;
        let started = Instant::now();
        aidaemon_store_sqlite::maintenance::optimize(
            &pool,
            aidaemon_store_sqlite::maintenance::Optimize::AtOpen,
        )
        .await
        .unwrap();
        println!("optimize_at_open elapsed={:.2?}", started.elapsed());
        let started = Instant::now();
        aidaemon_store_sqlite::maintenance::optimize(
            &pool,
            aidaemon_store_sqlite::maintenance::Optimize::Periodic,
        )
        .await
        .unwrap();
        println!("optimize_periodic_after elapsed={:.2?}", started.elapsed());
        let after = time_queries(&pool, &model, &session).await;
        for ((name, b, plan_b), (_, a, plan_a)) in before.iter().zip(&after) {
            println!("query {name}: before={b:.2?} after={a:.2?}");
            if plan_b != plan_a {
                println!("  plan before: {plan_b}\n  plan after:  {plan_a}");
            }
        }
        pool.close().await;
    }
}
