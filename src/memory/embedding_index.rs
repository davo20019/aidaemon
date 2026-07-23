//! Backend-neutral storage contract for derived memory embeddings.

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use sqlx::{QueryBuilder, Row, Sqlite, SqlitePool};

use crate::memory::binary::{decode_embedding_with_dim, encode_embedding};
use crate::memory::math::cosine_similarity;

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

#[derive(Clone)]
pub struct SqliteEmbeddingIndex {
    pool: SqlitePool,
}

impl SqliteEmbeddingIndex {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl EmbeddingIndex for SqliteEmbeddingIndex {
    async fn upsert(&self, item: EmbeddingItem) -> anyhow::Result<()> {
        if item.embedding.is_empty() {
            anyhow::bail!("refusing to index an empty embedding");
        }

        let mut tx = self.pool.begin().await?;
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

        let mut builder = QueryBuilder::<Sqlite>::new(
            "SELECT owner_type, owner_id, embedding_dim, embedding
             FROM memory_embeddings WHERE stale_at IS NULL",
        );
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

        let mut hits = Vec::new();
        for row in rows {
            let owner_type: String = row.get("owner_type");
            let dim: i64 = row.get("embedding_dim");
            if dim as usize != query.len() {
                continue;
            }
            let blob: Vec<u8> = row.get("embedding");
            let Ok(vector) = decode_embedding_with_dim(&blob, dim as usize) else {
                continue;
            };
            let hit = EmbeddingHit {
                owner: MemoryOwner {
                    owner_type,
                    owner_id: row.get("owner_id"),
                },
                score: cosine_similarity(query, &vector),
            };
            let position = hits
                .binary_search_by(|existing: &EmbeddingHit| {
                    existing
                        .score
                        .partial_cmp(&hit.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .reverse()
                        .then_with(|| existing.owner.owner_id.cmp(&hit.owner.owner_id))
                })
                .unwrap_or_else(|position| position);
            hits.insert(position, hit);
            if hits.len() > limit {
                hits.pop();
            }
        }
        Ok(hits)
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
    format!("{:x}", Sha256::digest(content.as_bytes()))
}
