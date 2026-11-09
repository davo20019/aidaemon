use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteRow,
    SqliteSynchronous,
};
use sqlx::{Row, Sqlite, Transaction};

use super::{migrations, related_fact_scores, MemoryStore};
use crate::embed::{decode_embedding, encode_embedding};
use crate::error::{MemoryError, Result};
use crate::scoring::{term_overlap, tokens};
use crate::types::{
    episode_search_text, fact_search_text, new_id, normalize_category, normalize_key, parse_ts,
    same_value, ts, DecayReport, Episode, EpisodeRecord, Fact, FactAccess, FactRecord, MemoryStats,
    NewEpisode, NewFact, Provenance, RecordKind, RetentionConfig, StaleEmbedding, StoredEmbedding,
    UpsertOutcome,
};

const FACT_COLUMNS: &str = "id, scope, category, key, value, confidence, source_run_id, \
     source_session_id, source_excerpt, created_at, updated_at, superseded_at, recall_count, \
     last_recalled_at, embedding, embedding_model";

const EPISODE_COLUMNS: &str = "id, scope, session_id, summary, topics, importance, created_at, \
     recall_count, last_recalled_at, embedding, embedding_model";

/// SQLite-backed [`MemoryStore`]: versioned migrations, embeddings as
/// little-endian f32 BLOBs, and FTS5 indexes (BM25) over active facts and
/// episodes, kept in sync inside each write transaction. Cheap to clone.
#[derive(Debug, Clone)]
pub struct SqliteMemoryStore {
    pool: SqlitePool,
}

impl SqliteMemoryStore {
    /// Open (creating if needed) a database file with WAL and a busy
    /// timeout, then migrate.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(Duration::from_secs(10));
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(options)
            .await?;
        Self::from_pool(pool).await
    }

    /// A private in-memory database, for tests.
    pub async fn open_in_memory() -> Result<Self> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")?;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .min_connections(1)
            .idle_timeout(None)
            .max_lifetime(None)
            .connect_with(options)
            .await?;
        Self::from_pool(pool).await
    }

    /// Use an existing pool (for example one shared with
    /// `aidaemon-store-sqlite`) and apply this crate's migrations.
    pub async fn from_pool(pool: SqlitePool) -> Result<Self> {
        migrations::run(&pool).await?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// The schema version recorded in the database.
    pub async fn schema_version(&self) -> Result<i64> {
        migrations::current_version(&self.pool).await
    }

    /// Rows in the fact FTS index for `scope` (diagnostics and tests).
    pub async fn fts_fact_ids(&self, scope: &str) -> Result<Vec<String>> {
        let ids = sqlx::query_scalar("SELECT fact_id FROM memory_facts_fts WHERE scope = ?")
            .bind(scope)
            .fetch_all(&self.pool)
            .await?;
        Ok(ids)
    }

    async fn try_upsert(
        &self,
        fact: &NewFact,
        embedding: &Option<StoredEmbedding>,
        access: &FactAccess,
        now: DateTime<Utc>,
    ) -> std::result::Result<UpsertOutcome, sqlx::Error> {
        let now_s = ts(now);
        let mut tx = self.pool.begin().await?;
        let existing = sqlx::query(
            "SELECT id, value, source_excerpt, privacy, channel_id FROM memory_facts
             WHERE scope = ? AND category = ? AND key = ? AND superseded_at IS NULL",
        )
        .bind(&fact.scope)
        .bind(&fact.category)
        .bind(&fact.key)
        .fetch_optional(&mut *tx)
        .await?;

        let mut previous = None;
        if let Some(row) = existing {
            let id: String = row.get("id");
            let value: String = row.get("value");
            if same_value(&value, &fact.value) {
                let (blob, model) = embedding_columns(embedding);
                let narrowed = row_access(&row).narrowest(access);
                sqlx::query(
                    "UPDATE memory_facts SET updated_at = ?, confidence = MAX(confidence, ?),
                        source_excerpt = COALESCE(source_excerpt, ?),
                        embedding = COALESCE(?, embedding),
                        embedding_model = COALESCE(?, embedding_model),
                        privacy = ?, channel_id = ?
                     WHERE id = ?",
                )
                .bind(&now_s)
                .bind(fact.confidence)
                .bind(&fact.source.excerpt)
                .bind(blob)
                .bind(model)
                .bind(narrowed.privacy.to_string())
                .bind(&narrowed.channel_id)
                .bind(&id)
                .execute(&mut *tx)
                .await?;
                tx.commit().await?;
                return Ok(UpsertOutcome::Reinforced { id });
            }
            sqlx::query("UPDATE memory_facts SET superseded_at = ? WHERE id = ?")
                .bind(&now_s)
                .bind(&id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM memory_facts_fts WHERE fact_id = ?")
                .bind(&id)
                .execute(&mut *tx)
                .await?;
            previous = Some((id, value));
        }

        let id = new_id();
        let (blob, model) = embedding_columns(embedding);
        sqlx::query(
            "INSERT INTO memory_facts (id, scope, category, key, value, confidence,
                source_run_id, source_session_id, source_excerpt, created_at, updated_at,
                embedding, embedding_model, privacy, channel_id)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(&fact.scope)
        .bind(&fact.category)
        .bind(&fact.key)
        .bind(&fact.value)
        .bind(fact.confidence)
        .bind(&fact.source.run_id)
        .bind(&fact.source.session_id)
        .bind(&fact.source.excerpt)
        .bind(&now_s)
        .bind(&now_s)
        .bind(blob)
        .bind(model)
        .bind(access.privacy.to_string())
        .bind(&access.channel_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query("INSERT INTO memory_facts_fts (fact_id, scope, content) VALUES (?, ?, ?)")
            .bind(&id)
            .bind(&fact.scope)
            .bind(fact_search_text(&fact.category, &fact.key, &fact.value))
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(match previous {
            None => UpsertOutcome::Inserted { id },
            Some((previous_id, previous_value)) => UpsertOutcome::Superseded {
                id,
                previous_id,
                previous_value,
            },
        })
    }

    /// Stored access of `scope`'s rows in `table`, for the ids asked about.
    async fn access_of(
        &self,
        table: &str,
        scope: &str,
        ids: &[String],
    ) -> Result<HashMap<String, FactAccess>> {
        let wanted: HashSet<&String> = ids.iter().collect();
        let sql = format!("SELECT id, privacy, channel_id FROM {table} WHERE scope = ?");
        let rows = sqlx::query(&sql).bind(scope).fetch_all(&self.pool).await?;
        Ok(rows
            .iter()
            .map(|row| (row.get::<String, _>("id"), row_access(row)))
            .filter(|(id, _)| wanted.contains(id))
            .collect())
    }

    /// BM25 over an FTS table, normalized to 0..=1: word coverage of the
    /// query, weighted by BM25 relative to the best hit.
    async fn fts_scores(
        &self,
        table: &str,
        id_column: &str,
        scope: &str,
        query: &str,
    ) -> Result<HashMap<String, f32>> {
        let unique: HashSet<String> = tokens(query).into_iter().collect();
        if unique.is_empty() {
            return Ok(HashMap::new());
        }
        // Tokens are alphanumeric only, so quoting makes every term a literal.
        let mut terms: Vec<String> = unique.into_iter().map(|t| format!("\"{t}\"")).collect();
        terms.sort();
        let sql = format!(
            "SELECT {id_column} AS id, bm25({table}) AS rank, content FROM {table}
             WHERE {table} MATCH ? AND scope = ?"
        );
        let rows = sqlx::query(&sql)
            .bind(terms.join(" OR "))
            .bind(scope)
            .fetch_all(&self.pool)
            .await?;
        let hits: Vec<(String, f64, String)> = rows
            .iter()
            .map(|r| (r.get("id"), r.get::<f64, _>("rank").abs(), r.get("content")))
            .collect();
        let best = hits.iter().map(|h| h.1).fold(0.0f64, f64::max);
        Ok(hits
            .into_iter()
            .map(|(id, rank, content)| {
                let relative = if best > 0.0 {
                    (rank / best) as f32
                } else {
                    1.0
                };
                let score = term_overlap(query, &content) * (0.5 + 0.5 * relative);
                (id, score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect())
    }
}

fn embedding_columns(embedding: &Option<StoredEmbedding>) -> (Option<Vec<u8>>, Option<String>) {
    match embedding {
        Some(e) => (Some(encode_embedding(&e.vector)), Some(e.model.clone())),
        None => (None, None),
    }
}

fn row_embedding(row: &SqliteRow) -> Option<StoredEmbedding> {
    let blob: Option<Vec<u8>> = row.get("embedding");
    let model: Option<String> = row.get("embedding_model");
    Some(StoredEmbedding {
        vector: decode_embedding(&blob?)?,
        model: model?,
    })
}

fn row_access(row: &SqliteRow) -> FactAccess {
    FactAccess::from_stored(row.get("privacy"), row.get("channel_id"))
}

fn row_time(row: &SqliteRow, column: &str) -> DateTime<Utc> {
    row.get::<Option<String>, _>(column)
        .and_then(|s| parse_ts(&s))
        .unwrap_or_default()
}

fn row_opt_time(row: &SqliteRow, column: &str) -> Option<DateTime<Utc>> {
    row.get::<Option<String>, _>(column)
        .and_then(|s| parse_ts(&s))
}

fn row_to_fact(row: &SqliteRow) -> FactRecord {
    FactRecord {
        fact: Fact {
            id: row.get("id"),
            scope: row.get("scope"),
            category: row.get("category"),
            key: row.get("key"),
            value: row.get("value"),
            confidence: row.get::<f64, _>("confidence") as f32,
            source: Provenance {
                run_id: row.get("source_run_id"),
                session_id: row.get("source_session_id"),
                excerpt: row.get("source_excerpt"),
            },
            created_at: row_time(row, "created_at"),
            updated_at: row_time(row, "updated_at"),
            superseded_at: row_opt_time(row, "superseded_at"),
            recall_count: row.get::<i64, _>("recall_count").max(0) as u32,
            last_recalled_at: row_opt_time(row, "last_recalled_at"),
        },
        embedding: row_embedding(row),
    }
}

fn row_to_episode(row: &SqliteRow) -> EpisodeRecord {
    let topics: String = row.get("topics");
    EpisodeRecord {
        episode: Episode {
            id: row.get("id"),
            scope: row.get("scope"),
            session_id: row.get("session_id"),
            summary: row.get("summary"),
            topics: serde_json::from_str(&topics).unwrap_or_default(),
            importance: row.get::<f64, _>("importance") as f32,
            created_at: row_time(row, "created_at"),
            recall_count: row.get::<i64, _>("recall_count").max(0) as u32,
            last_recalled_at: row_opt_time(row, "last_recalled_at"),
        },
        embedding: row_embedding(row),
    }
}

fn is_unique_violation(err: &sqlx::Error) -> bool {
    matches!(err, sqlx::Error::Database(db) if db.is_unique_violation())
}

async fn bump_recalls(
    tx: &mut Transaction<'_, Sqlite>,
    table: &str,
    ids: &[String],
    now: &str,
) -> std::result::Result<(), sqlx::Error> {
    let sql = format!(
        "UPDATE {table} SET recall_count = recall_count + 1, last_recalled_at = ? WHERE id = ?"
    );
    for id in ids {
        sqlx::query(&sql)
            .bind(now)
            .bind(id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    async fn upsert_fact(
        &self,
        fact: NewFact,
        embedding: Option<StoredEmbedding>,
        now: DateTime<Utc>,
    ) -> Result<UpsertOutcome> {
        self.upsert_fact_with_access(fact, embedding, &FactAccess::default(), now)
            .await
    }

    async fn upsert_fact_with_access(
        &self,
        fact: NewFact,
        embedding: Option<StoredEmbedding>,
        access: &FactAccess,
        now: DateTime<Utc>,
    ) -> Result<UpsertOutcome> {
        match self.try_upsert(&fact, &embedding, access, now).await {
            // A concurrent writer inserted the same key between our read and
            // write; the retry sees its row and reinforces or supersedes it.
            Err(err) if is_unique_violation(&err) => {
                Ok(self.try_upsert(&fact, &embedding, access, now).await?)
            }
            other => Ok(other?),
        }
    }

    async fn fact_access(
        &self,
        scope: &str,
        ids: &[String],
    ) -> Result<HashMap<String, FactAccess>> {
        self.access_of("memory_facts", scope, ids).await
    }

    async fn supersede_fact(
        &self,
        scope: &str,
        category: &str,
        key: &str,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let ids: Vec<String> = sqlx::query_scalar(
            "SELECT id FROM memory_facts
             WHERE scope = ? AND category = ? AND key = ? AND superseded_at IS NULL",
        )
        .bind(scope)
        .bind(normalize_category(category))
        .bind(normalize_key(key))
        .fetch_all(&mut *tx)
        .await?;
        for id in &ids {
            sqlx::query("UPDATE memory_facts SET superseded_at = ? WHERE id = ?")
                .bind(ts(now))
                .bind(id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM memory_facts_fts WHERE fact_id = ?")
                .bind(id)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        Ok(!ids.is_empty())
    }

    async fn active_facts(&self, scope: &str) -> Result<Vec<FactRecord>> {
        let sql = format!(
            "SELECT {FACT_COLUMNS} FROM memory_facts
             WHERE scope = ? AND superseded_at IS NULL ORDER BY updated_at DESC"
        );
        let rows = sqlx::query(&sql).bind(scope).fetch_all(&self.pool).await?;
        Ok(rows.iter().map(row_to_fact).collect())
    }

    async fn fact_graph_scores(
        &self,
        scope: &str,
        seed_ids: &[String],
    ) -> Result<HashMap<String, f32>> {
        let active: Vec<Fact> = self
            .active_facts(scope)
            .await?
            .into_iter()
            .map(|r| r.fact)
            .collect();
        Ok(related_fact_scores(&active, seed_ids))
    }

    async fn fact_history(&self, scope: &str, category: &str, key: &str) -> Result<Vec<Fact>> {
        let sql = format!(
            "SELECT {FACT_COLUMNS} FROM memory_facts
             WHERE scope = ? AND category = ? AND key = ? ORDER BY created_at, rowid"
        );
        let rows = sqlx::query(&sql)
            .bind(scope)
            .bind(normalize_category(category))
            .bind(normalize_key(key))
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.iter().map(|r| row_to_fact(r).fact).collect())
    }

    async fn fact_lexical_scores(&self, scope: &str, query: &str) -> Result<HashMap<String, f32>> {
        self.fts_scores("memory_facts_fts", "fact_id", scope, query)
            .await
    }

    async fn record_fact_recalls(&self, ids: &[String], now: DateTime<Utc>) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        bump_recalls(&mut tx, "memory_facts", ids, &ts(now)).await?;
        tx.commit().await?;
        Ok(())
    }

    async fn insert_episode(
        &self,
        episode: NewEpisode,
        embedding: Option<StoredEmbedding>,
        now: DateTime<Utc>,
    ) -> Result<Episode> {
        self.insert_episode_with_access(episode, embedding, &FactAccess::default(), now)
            .await
    }

    async fn insert_episode_with_access(
        &self,
        episode: NewEpisode,
        embedding: Option<StoredEmbedding>,
        access: &FactAccess,
        now: DateTime<Utc>,
    ) -> Result<Episode> {
        let episode = Episode {
            id: new_id(),
            scope: episode.scope,
            session_id: episode.session_id,
            summary: episode.summary,
            topics: episode.topics,
            importance: episode.importance,
            created_at: now,
            recall_count: 0,
            last_recalled_at: None,
        };
        let topics = serde_json::to_string(&episode.topics)
            .map_err(|e| MemoryError::Store(e.to_string()))?;
        let (blob, model) = embedding_columns(&embedding);
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "INSERT INTO memory_episodes (id, scope, session_id, summary, topics, importance,
                created_at, embedding, embedding_model, privacy, channel_id)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&episode.id)
        .bind(&episode.scope)
        .bind(&episode.session_id)
        .bind(&episode.summary)
        .bind(topics)
        .bind(episode.importance)
        .bind(ts(now))
        .bind(blob)
        .bind(model)
        .bind(access.privacy.to_string())
        .bind(&access.channel_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "INSERT INTO memory_episodes_fts (episode_id, scope, content) VALUES (?, ?, ?)",
        )
        .bind(&episode.id)
        .bind(&episode.scope)
        .bind(episode_search_text(&episode.summary, &episode.topics))
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(episode)
    }

    async fn episode_access(
        &self,
        scope: &str,
        ids: &[String],
    ) -> Result<HashMap<String, FactAccess>> {
        self.access_of("memory_episodes", scope, ids).await
    }

    async fn episodes(&self, scope: &str) -> Result<Vec<EpisodeRecord>> {
        let sql = format!(
            "SELECT {EPISODE_COLUMNS} FROM memory_episodes WHERE scope = ? ORDER BY created_at DESC"
        );
        let rows = sqlx::query(&sql).bind(scope).fetch_all(&self.pool).await?;
        Ok(rows.iter().map(row_to_episode).collect())
    }

    async fn episode_lexical_scores(
        &self,
        scope: &str,
        query: &str,
    ) -> Result<HashMap<String, f32>> {
        self.fts_scores("memory_episodes_fts", "episode_id", scope, query)
            .await
    }

    async fn record_episode_recalls(&self, ids: &[String], now: DateTime<Utc>) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        bump_recalls(&mut tx, "memory_episodes", ids, &ts(now)).await?;
        tx.commit().await?;
        Ok(())
    }

    async fn decay(&self, now: DateTime<Utc>, config: &RetentionConfig) -> Result<DecayReport> {
        let mut report = DecayReport::default();
        let mut tx = self.pool.begin().await?;
        if let Some(window) = config.superseded_retention {
            report.superseded_deleted = sqlx::query(
                "DELETE FROM memory_facts WHERE superseded_at IS NOT NULL AND superseded_at < ?",
            )
            .bind(ts(now - window))
            .execute(&mut *tx)
            .await?
            .rows_affected();
        }
        if let Some(window) = config.episode_retention {
            let cutoff = ts(now - window);
            sqlx::query(
                "DELETE FROM memory_episodes_fts WHERE episode_id IN (
                    SELECT id FROM memory_episodes WHERE recall_count = 0 AND created_at < ?)",
            )
            .bind(&cutoff)
            .execute(&mut *tx)
            .await?;
            report.episodes_deleted = sqlx::query(
                "DELETE FROM memory_episodes WHERE recall_count = 0 AND created_at < ?",
            )
            .bind(&cutoff)
            .execute(&mut *tx)
            .await?
            .rows_affected();
        }
        if let Some(window) = config.fact_recall_decay_after {
            report.facts_decayed = sqlx::query(
                "UPDATE memory_facts
                 SET recall_count = MAX(0, recall_count - MAX(1, recall_count / 10))
                 WHERE superseded_at IS NULL AND recall_count > 0
                   AND (last_recalled_at IS NULL OR last_recalled_at < ?)",
            )
            .bind(ts(now - window))
            .execute(&mut *tx)
            .await?
            .rows_affected();
        }
        if let Some(window) = config.episode_recall_decay_after {
            report.episodes_decayed = sqlx::query(
                "UPDATE memory_episodes SET recall_count = MAX(0, recall_count - 1)
                 WHERE recall_count > 0 AND (last_recalled_at IS NULL OR last_recalled_at < ?)",
            )
            .bind(ts(now - window))
            .execute(&mut *tx)
            .await?
            .rows_affected();
        }
        tx.commit().await?;
        Ok(report)
    }

    async fn stale_embeddings(
        &self,
        model: &str,
        kind: RecordKind,
        after_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<StaleEmbedding>> {
        let (columns, table, active) = match kind {
            RecordKind::Fact => (
                "id, category, key, value, embedding_model",
                "memory_facts",
                "superseded_at IS NULL AND",
            ),
            RecordKind::Episode => (
                "id, summary, topics, embedding_model",
                "memory_episodes",
                "",
            ),
        };
        let sql = format!(
            "SELECT {columns} FROM {table}
             WHERE {active} (embedding_model IS NULL OR embedding_model != ?)
               AND (? IS NULL OR id > ?)
             ORDER BY id LIMIT ?"
        );
        let rows = sqlx::query(&sql)
            .bind(model)
            .bind(after_id)
            .bind(after_id)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .iter()
            .map(|row| StaleEmbedding {
                kind,
                id: row.get("id"),
                text: match kind {
                    RecordKind::Fact => {
                        fact_search_text(row.get("category"), row.get("key"), row.get("value"))
                    }
                    RecordKind::Episode => {
                        let topics: String = row.get("topics");
                        let topics: Vec<String> = serde_json::from_str(&topics).unwrap_or_default();
                        episode_search_text(row.get("summary"), &topics)
                    }
                },
                model: row.get("embedding_model"),
            })
            .collect())
    }

    async fn replace_embedding(
        &self,
        stale: &StaleEmbedding,
        embedding: StoredEmbedding,
    ) -> Result<bool> {
        let (table, active) = match stale.kind {
            RecordKind::Fact => ("memory_facts", "AND superseded_at IS NULL"),
            RecordKind::Episode => ("memory_episodes", ""),
        };
        let sql = format!(
            "UPDATE {table} SET embedding = ?, embedding_model = ?
             WHERE id = ? AND embedding_model IS ? {active}"
        );
        let changed = sqlx::query(&sql)
            .bind(encode_embedding(&embedding.vector))
            .bind(&embedding.model)
            .bind(&stale.id)
            .bind(&stale.model)
            .execute(&self.pool)
            .await?
            .rows_affected();
        Ok(changed > 0)
    }

    async fn stats(&self) -> Result<MemoryStats> {
        let row = sqlx::query(
            "SELECT
                (SELECT COUNT(*) FROM memory_facts WHERE superseded_at IS NULL) AS active,
                (SELECT COUNT(*) FROM memory_facts WHERE superseded_at IS NOT NULL) AS superseded,
                (SELECT COUNT(*) FROM memory_episodes) AS episodes,
                (SELECT COUNT(*) FROM (SELECT scope FROM memory_facts
                                       UNION SELECT scope FROM memory_episodes)) AS scopes",
        )
        .fetch_one(&self.pool)
        .await?;
        let count = |c: &str| row.get::<i64, _>(c).max(0) as u64;
        Ok(MemoryStats {
            active_facts: count("active"),
            superseded_facts: count("superseded"),
            episodes: count("episodes"),
            scopes: count("scopes"),
        })
    }
}
