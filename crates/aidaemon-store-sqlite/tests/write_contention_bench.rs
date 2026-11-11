//! What a deferred read-then-write transaction costs under WAL contention.
//!
//! Ignored by default; run it to compare `BEGIN` with `BEGIN IMMEDIATE`:
//!
//! ```text
//! cargo test -p aidaemon-store-sqlite --release --test write_contention_bench -- --ignored --nocapture
//! ```
//!
//! A deferred transaction that reads first takes a snapshot; if another
//! connection commits before it writes, SQLite fails the upgrade with
//! `SQLITE_BUSY_SNAPSHOT` at once (the busy timeout cannot help a stale
//! snapshot). `BEGIN IMMEDIATE` takes the write lock up front and waits.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::SqlitePool;

const WORKERS: usize = 8;
const OPS_PER_WORKER: usize = 200;

async fn pool(path: &std::path::Path) -> SqlitePool {
    let options = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal)
        .busy_timeout(Duration::from_secs(30));
    let pool = SqlitePoolOptions::new()
        .max_connections(10)
        .connect_with(options)
        .await
        .unwrap();
    sqlx::query("CREATE TABLE counters (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO counters (id, value) VALUES (1, 0)")
        .execute(&pool)
        .await
        .unwrap();
    pool
}

/// Each operation reads the counter, then writes it back incremented: the
/// shape of a claim, a lease or an idempotency check.
async fn run(pool: SqlitePool, begin: &'static str) -> (usize, usize, Duration, i64) {
    let failures = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let mut workers = Vec::new();
    for _ in 0..WORKERS {
        let pool = pool.clone();
        let failures = failures.clone();
        workers.push(tokio::spawn(async move {
            for _ in 0..OPS_PER_WORKER {
                let attempt = async {
                    let mut tx = pool.begin_with(begin).await?;
                    let value: i64 = sqlx::query_scalar("SELECT value FROM counters WHERE id = 1")
                        .fetch_one(&mut *tx)
                        .await?;
                    tokio::task::yield_now().await;
                    sqlx::query("UPDATE counters SET value = ? WHERE id = 1")
                        .bind(value + 1)
                        .execute(&mut *tx)
                        .await?;
                    tx.commit().await?;
                    Ok::<_, sqlx::Error>(())
                };
                if attempt.await.is_err() {
                    failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    for worker in workers {
        worker.await.unwrap();
    }
    let elapsed = started.elapsed();
    let value: i64 = sqlx::query_scalar("SELECT value FROM counters WHERE id = 1")
        .fetch_one(&pool)
        .await
        .unwrap();
    (
        WORKERS * OPS_PER_WORKER,
        failures.load(Ordering::Relaxed),
        elapsed,
        value,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "benchmark: run explicitly in release mode"]
async fn deferred_versus_immediate_read_then_write() {
    for (label, begin) in [("deferred", "BEGIN"), ("immediate", "BEGIN IMMEDIATE")] {
        let dir = tempfile::tempdir().unwrap();
        let pool = pool(&dir.path().join("contention.db")).await;
        let (ops, failed, elapsed, value) = run(pool, begin).await;
        println!(
            "write_contention mode={label} ops={ops} failed={failed} committed={value} \
             elapsed={elapsed:.2?} ops_per_sec={:.0}",
            (ops - failed) as f64 / elapsed.as_secs_f64()
        );
        if begin == "BEGIN IMMEDIATE" {
            assert_eq!(failed, 0, "immediate transactions should wait, not fail");
            assert_eq!(value as usize, ops);
        }
    }
}
