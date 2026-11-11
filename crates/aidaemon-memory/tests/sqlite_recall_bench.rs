//! Recall latency on a file-backed SQLite memory of realistic size.
//!
//! Ignored by default; run it in release mode and compare before/after a
//! change to recall or the embedding index:
//!
//! ```text
//! cargo test -p aidaemon-memory --release --test sqlite_recall_bench -- --ignored --nocapture
//! ```
//!
//! `AIDAEMON_BENCH_FACTS` sets the number of facts (default 4000, about the
//! size of a long-lived personal daemon).

use std::sync::Arc;
use std::time::{Duration, Instant};

use aidaemon_memory::sqlite::{migrate, SqliteMemory};
use aidaemon_memory::stores::FactStore;
use aidaemon_memory::HashEmbedder;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};

const TOPICS: &[&str] = &[
    "coffee", "hiking", "rust", "piano", "garden", "travel", "budget", "fitness", "cooking",
    "reading", "music", "school", "work", "health", "pets", "movies", "chess", "running",
];
const QUERIES: &[&str] = &[
    "what coffee do I like",
    "plan a hiking trip",
    "help me with my rust project",
    "when is the piano lesson",
    "what should I plant in the garden",
    "remind me about the travel budget",
    "what is my fitness routine",
    "suggest a recipe for dinner",
];

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[index]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "benchmark: run explicitly in release mode"]
async fn fact_recall_latency() {
    let facts: usize = std::env::var("AIDAEMON_BENCH_FACTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4000);
    let dir = tempfile::tempdir().unwrap();
    let options = SqliteConnectOptions::new()
        .filename(dir.path().join("bench.db"))
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal)
        .busy_timeout(Duration::from_secs(30));
    let pool = SqlitePoolOptions::new()
        .max_connections(10)
        .connect_with(options)
        .await
        .unwrap();
    migrate(&pool).await.unwrap();
    let memory = SqliteMemory::new(pool.clone(), Arc::new(HashEmbedder::new(384)));

    let seed_started = Instant::now();
    // Rows go in directly: seeding through `upsert_fact` compares each new
    // fact with every stored one, which is fine one write at a time but
    // quadratic for thousands. The store then builds its own claims and
    // embeddings, as it does for a migrated database.
    let mut tx = pool.begin_with("BEGIN IMMEDIATE").await.unwrap();
    for i in 0..facts {
        let topic = TOPICS[i % TOPICS.len()];
        sqlx::query(
            "INSERT INTO facts (category, key, value, source, created_at, updated_at, privacy)
             VALUES ('preference', ?, ?, 'agent', datetime('now'), datetime('now'), 'global')",
        )
        .bind(format!("{topic}_{i}"))
        .bind(format!(
            "Synthetic note {i}: Alice Rivera likes {topic} variant {}",
            i * 7
        ))
        .execute(&mut *tx)
        .await
        .unwrap();
    }
    tx.commit().await.unwrap();
    while memory.backfill_fact_embeddings().await.unwrap() > 0 {}
    for _ in 0..50 {
        let done = memory.backfill_missing_memory_projections().await.unwrap();
        if done == (0, 0, 0, 0, 0) {
            break;
        }
    }
    let seeded = seed_started.elapsed();
    let indexed: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_embeddings WHERE stale_at IS NULL AND owner_type = 'claim'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    // Warm up once so the first-query cost (page cache, any in-memory index)
    // is reported separately from steady state.
    let first_started = Instant::now();
    memory.get_relevant_facts(QUERIES[0], 12).await.unwrap();
    let first = first_started.elapsed();

    let mut samples = Vec::new();
    for round in 0..5 {
        for query in QUERIES {
            let started = Instant::now();
            let found = memory.get_relevant_facts(query, 12).await.unwrap();
            samples.push(started.elapsed());
            if round == 0 && query == &QUERIES[0] {
                assert!(!found.is_empty(), "recall should find seeded facts");
            }
        }
    }
    samples.sort();
    let mean = samples.iter().sum::<Duration>() / samples.len() as u32;
    println!(
        "fact_recall facts={facts} claim_vectors={indexed} seed={seeded:.2?} first={first:.2?} \
         mean={mean:.2?} p50={:.2?} p95={:.2?} n={}",
        percentile(&samples, 0.5),
        percentile(&samples, 0.95),
        samples.len()
    );
}
