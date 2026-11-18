//! Backend-neutral storage contract for derived memory embeddings.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use sqlx::{QueryBuilder, Row, Sqlite, SqlitePool};

use crate::embed::{decode_embedding_with_dim, encode_embedding};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryOwner {
    pub owner_type: String,
    pub owner_id: String,
}

impl MemoryOwner {
    pub fn new(owner_type: impl Into<String>, owner_id: impl ToString) -> Self {
        Self {
            owner_type: owner_type.into(),
            owner_id: owner_id.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingItem {
    pub owner: MemoryOwner,
    pub purpose: String,
    pub model: String,
    pub content_hash: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct EmbeddingSearchFilter {
    pub owner_types: Vec<String>,
    pub purpose: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingHit {
    pub owner: MemoryOwner,
    pub score: f32,
}

#[async_trait]
pub trait EmbeddingIndex: Send + Sync {
    async fn upsert(&self, item: EmbeddingItem) -> anyhow::Result<()>;
    async fn search(
        &self,
        query: &[f32],
        filter: &EmbeddingSearchFilter,
        limit: usize,
    ) -> anyhow::Result<Vec<EmbeddingHit>>;
    async fn mark_stale(&self, owner: &MemoryOwner) -> anyhow::Result<()>;
    #[allow(dead_code)]
    async fn purge_stale(&self) -> anyhow::Result<u64>;
}

/// The `memory_embeddings` table, searched exactly (every live vector is
/// scored).
///
/// Decoded vectors are kept in memory per filter and reused until the table
/// changes. A trigger installed by [`crate::sqlite::migrate`] bumps
/// `memory_embedding_generation` on every insert, update or delete, whoever
/// writes the row (this index, retention, a wipe, another process), and a
/// search first compares that number with the one its vectors were loaded at.
/// Without the generation table (a database [`crate::sqlite::migrate`] has
/// not seen) every search reads the table, as before.
///
/// Clones share the cache, so hold one index per database and clone it.
#[derive(Clone)]
pub struct SqliteEmbeddingIndex {
    pool: SqlitePool,
    cache: Arc<Mutex<HashMap<CacheKey, Arc<VectorSet>>>>,
}

/// One filter's worth of decoded vectors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    owner_types: Vec<String>,
    purpose: Option<String>,
    model: Option<String>,
}

/// Vectors of one dimension, stored contiguously with their norms so a
/// search is a dot product per row.
struct VectorSet {
    generation: i64,
    dim: usize,
    owners: Vec<MemoryOwner>,
    norms: Vec<f32>,
    values: Vec<f32>,
}

impl VectorSet {
    fn len(&self) -> usize {
        self.owners.len()
    }
}

impl SqliteEmbeddingIndex {
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            pool,
            cache: Arc::default(),
        }
    }

    /// The table's change counter, or `None` when the database has no
    /// generation table (searches then read the table every time).
    async fn generation(&self) -> Option<i64> {
        sqlx::query_scalar("SELECT generation FROM memory_embedding_generation WHERE id = 1")
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten()
    }

    async fn load(
        &self,
        filter: &EmbeddingSearchFilter,
        dim: usize,
        generation: i64,
    ) -> anyhow::Result<VectorSet> {
        let mut builder = QueryBuilder::<Sqlite>::new(
            "SELECT owner_type, owner_id, embedding_dim, embedding
             FROM memory_embeddings WHERE stale_at IS NULL",
        );
        builder.push(" AND embedding_dim = ").push_bind(dim as i64);
        if let Some(purpose) = &filter.purpose {
            builder.push(" AND embedding_purpose = ").push_bind(purpose);
        }
        if let Some(model) = &filter.model {
            builder.push(" AND embedding_model = ").push_bind(model);
        }
        if !filter.owner_types.is_empty() {
            builder.push(" AND owner_type IN (");
            let mut separated = builder.separated(", ");
            for owner_type in &filter.owner_types {
                separated.push_bind(owner_type);
            }
            separated.push_unseparated(")");
        }
        let rows = builder.build().fetch_all(&self.pool).await?;

        let mut set = VectorSet {
            generation,
            dim,
            owners: Vec::with_capacity(rows.len()),
            norms: Vec::with_capacity(rows.len()),
            values: Vec::with_capacity(rows.len() * dim),
        };
        for row in rows {
            let blob: Vec<u8> = row.get("embedding");
            let Ok(vector) = decode_embedding_with_dim(&blob, dim) else {
                continue;
            };
            if vector.len() != dim {
                continue;
            }
            set.norms
                .push(vector.iter().map(|x| x * x).sum::<f32>().sqrt());
            set.values.extend_from_slice(&vector);
            set.owners.push(MemoryOwner {
                owner_type: row.get("owner_type"),
                owner_id: row.get("owner_id"),
            });
        }
        Ok(set)
    }

    async fn vectors(
        &self,
        filter: &EmbeddingSearchFilter,
        dim: usize,
    ) -> anyhow::Result<Arc<VectorSet>> {
        let mut owner_types = filter.owner_types.clone();
        owner_types.sort();
        owner_types.dedup();
        let key = CacheKey {
            owner_types,
            purpose: filter.purpose.clone(),
            model: filter.model.clone(),
        };
        // Read the generation before the rows: a write that lands in
        // between bumps it again, so the set is reloaded next time rather
        // than served stale.
        let Some(generation) = self.generation().await else {
            return Ok(Arc::new(self.load(filter, dim, -1).await?));
        };
        if let Some(set) = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            && set.generation == generation
            && set.dim == dim
        {
            return Ok(set.clone());
        }
        let set = Arc::new(self.load(filter, dim, generation).await?);
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, set.clone());
        Ok(set)
    }
}

/// The `limit` best of `set` for `query`, best first; ties by owner id.
fn rank(set: &VectorSet, query: &[f32], limit: usize) -> Vec<EmbeddingHit> {
    let query_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut scored: Vec<(f32, usize)> = (0..set.len())
        .map(|index| {
            let norm = set.norms[index];
            let score = if query_norm <= f32::EPSILON.sqrt() || norm <= f32::EPSILON.sqrt() {
                0.0
            } else {
                let row = &set.values[index * set.dim..(index + 1) * set.dim];
                let dot: f32 = row.iter().zip(query).map(|(a, b)| a * b).sum();
                dot / (query_norm * norm)
            };
            (score, index)
        })
        .collect();
    let order = |a: &(f32, usize), b: &(f32, usize)| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| set.owners[a.1].owner_id.cmp(&set.owners[b.1].owner_id))
    };
    if scored.len() > limit {
        scored.select_nth_unstable_by(limit, order);
        scored.truncate(limit);
    }
    scored.sort_by(order);
    scored
        .into_iter()
        .map(|(score, index)| EmbeddingHit {
            owner: set.owners[index].clone(),
            score,
        })
        .collect()
}

#[async_trait]
impl EmbeddingIndex for SqliteEmbeddingIndex {
    async fn upsert(&self, item: EmbeddingItem) -> anyhow::Result<()> {
        if item.embedding.is_empty() {
            anyhow::bail!("refusing to index an empty embedding");
        }

        let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
        sqlx::query(
            "UPDATE memory_embeddings
             SET stale_at = COALESCE(stale_at, datetime('now')), updated_at = datetime('now')
             WHERE owner_type = ? AND owner_id = ? AND embedding_purpose = ?
               AND embedding_model = ? AND content_hash != ? AND stale_at IS NULL",
        )
        .bind(&item.owner.owner_type)
        .bind(&item.owner.owner_id)
        .bind(&item.purpose)
        .bind(&item.model)
        .bind(&item.content_hash)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO memory_embeddings
                (owner_type, owner_id, embedding_purpose, embedding_model,
                 embedding_dim, content_hash, embedding, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
             ON CONFLICT(owner_type, owner_id, embedding_purpose, embedding_model, content_hash)
             DO UPDATE SET embedding = excluded.embedding,
                           embedding_dim = excluded.embedding_dim,
                           stale_at = NULL,
                           updated_at = datetime('now')",
        )
        .bind(&item.owner.owner_type)
        .bind(&item.owner.owner_id)
        .bind(&item.purpose)
        .bind(&item.model)
        .bind(item.embedding.len() as i64)
        .bind(&item.content_hash)
        .bind(encode_embedding(&item.embedding))
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        filter: &EmbeddingSearchFilter,
        limit: usize,
    ) -> anyhow::Result<Vec<EmbeddingHit>> {
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let set = self.vectors(filter, query.len()).await?;
        Ok(rank(&set, query, limit))
    }

    async fn mark_stale(&self, owner: &MemoryOwner) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE memory_embeddings
             SET stale_at = COALESCE(stale_at, datetime('now')), updated_at = datetime('now')
             WHERE owner_type = ? AND owner_id = ? AND stale_at IS NULL",
        )
        .bind(&owner.owner_type)
        .bind(&owner.owner_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn purge_stale(&self) -> anyhow::Result<u64> {
        Ok(
            sqlx::query("DELETE FROM memory_embeddings WHERE stale_at IS NOT NULL")
                .execute(&self.pool)
                .await?
                .rows_affected(),
        )
    }
}

pub fn content_hash(content: &str) -> String {
    hex::encode(Sha256::digest(content.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn index() -> (SqlitePool, SqliteEmbeddingIndex) {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        (pool.clone(), SqliteEmbeddingIndex::new(pool))
    }

    fn item(owner_id: &str, vector: Vec<f32>) -> EmbeddingItem {
        EmbeddingItem {
            owner: MemoryOwner::new("claim", owner_id),
            purpose: "recall".to_string(),
            model: "synthetic-model".to_string(),
            content_hash: content_hash(&format!("{owner_id}{vector:?}")),
            embedding: vector,
        }
    }

    fn filter() -> EmbeddingSearchFilter {
        EmbeddingSearchFilter {
            owner_types: vec!["claim".to_string()],
            purpose: Some("recall".to_string()),
            model: Some("synthetic-model".to_string()),
        }
    }

    async fn top(index: &SqliteEmbeddingIndex, query: &[f32]) -> Vec<String> {
        index
            .search(query, &filter(), 10)
            .await
            .unwrap()
            .into_iter()
            .map(|hit| hit.owner.owner_id)
            .collect()
    }

    #[tokio::test]
    async fn cached_vectors_follow_every_kind_of_write() {
        let (pool, index) = index().await;
        index.upsert(item("1", vec![1.0, 0.0])).await.unwrap();
        assert_eq!(top(&index, &[1.0, 0.0]).await, ["1"]);

        // A row written behind the index's back (another module, another
        // process) is seen on the next search.
        sqlx::query(
            "INSERT INTO memory_embeddings
                (owner_type, owner_id, embedding_purpose, embedding_model,
                 embedding_dim, content_hash, embedding, created_at, updated_at)
             VALUES ('claim', '2', 'recall', 'synthetic-model', 2, 'raw', ?,
                     datetime('now'), datetime('now'))",
        )
        .bind(encode_embedding(&[0.9, 0.1]))
        .execute(&pool)
        .await
        .unwrap();
        assert_eq!(top(&index, &[1.0, 0.0]).await, ["1", "2"]);

        // Re-embedding replaces the old vector.
        index.upsert(item("1", vec![0.0, 1.0])).await.unwrap();
        assert_eq!(top(&index, &[1.0, 0.0]).await, ["2", "1"]);

        // Stale and deleted rows leave the results.
        index
            .mark_stale(&MemoryOwner::new("claim", "2"))
            .await
            .unwrap();
        assert_eq!(top(&index, &[1.0, 0.0]).await, ["1"]);
        sqlx::query("DELETE FROM memory_embeddings")
            .execute(&pool)
            .await
            .unwrap();
        assert!(top(&index, &[1.0, 0.0]).await.is_empty());
    }

    #[tokio::test]
    async fn clones_share_one_cache_and_unchanged_tables_are_not_reread() {
        let (pool, index) = index().await;
        index.upsert(item("1", vec![1.0, 0.0])).await.unwrap();
        let clone = index.clone();
        assert_eq!(top(&clone, &[1.0, 0.0]).await, ["1"]);
        let generation = index.generation().await.unwrap();

        // Corrupt the stored row without bumping the generation: a search
        // that reread the table would now skip it, a cached one still finds it.
        sqlx::query("DROP TRIGGER memory_embeddings_generation_update")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("UPDATE memory_embeddings SET embedding = x'00'")
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(index.generation().await, Some(generation));
        assert_eq!(top(&index, &[1.0, 0.0]).await, ["1"]);
    }

    #[tokio::test]
    async fn a_database_without_the_generation_table_still_searches() {
        let (pool, index) = index().await;
        index.upsert(item("1", vec![1.0, 0.0])).await.unwrap();
        sqlx::query("DROP TABLE memory_embedding_generation")
            .execute(&pool)
            .await
            .unwrap();
        for trigger in ["insert", "update", "delete"] {
            sqlx::query(sqlx::AssertSqlSafe(format!(
                "DROP TRIGGER memory_embeddings_generation_{trigger}"
            )))
            .execute(&pool)
            .await
            .unwrap();
        }
        assert_eq!(top(&index, &[1.0, 0.0]).await, ["1"]);
        index.upsert(item("2", vec![1.0, 0.1])).await.unwrap();
        assert_eq!(top(&index, &[1.0, 0.0]).await, ["1", "2"]);
    }

    #[test]
    fn ranking_matches_plain_cosine_and_breaks_ties_by_owner_id() {
        let vectors = [
            ("b", vec![1.0, 0.0, 0.0]),
            ("a", vec![1.0, 0.0, 0.0]),
            ("c", vec![0.2, 0.9, 0.1]),
            ("zero", vec![0.0, 0.0, 0.0]),
        ];
        let mut set = VectorSet {
            generation: 0,
            dim: 3,
            owners: Vec::new(),
            norms: Vec::new(),
            values: Vec::new(),
        };
        for (id, vector) in &vectors {
            set.owners.push(MemoryOwner::new("claim", *id));
            set.norms
                .push(vector.iter().map(|x| x * x).sum::<f32>().sqrt());
            set.values.extend_from_slice(vector);
        }
        let query = [0.8, 0.3, 0.0];
        let hits = rank(&set, &query, 3);
        let ids: Vec<&str> = hits.iter().map(|h| h.owner.owner_id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        for hit in &hits {
            let (_, vector) = vectors
                .iter()
                .find(|(id, _)| *id == hit.owner.owner_id)
                .unwrap();
            assert!((hit.score - crate::embed::cosine(&query, vector)).abs() < 1e-6);
        }
        assert_eq!(rank(&set, &query, 10).last().unwrap().score, 0.0);
    }
}
