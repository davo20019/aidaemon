//! The SQLite implementation of the [`crate::stores`] contract: facts with
//! privacy and channel provenance, episodes, the claim graph, structured
//! personal memory, people and learned habits.
//!
//! [`SqliteMemory`] works on a pool the host opens, so memory can share one
//! database file (and its encryption) with the host's other tables.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Datelike, Utc};
use sqlx::{Row, SqlitePool};
use tokio::sync::Mutex;

use crate::embed::{decode_embedding_with_dim, encode_embedding, Embedder};
use crate::records::*;
use crate::stores::*;
use crate::text::truncate_chars as truncate_str;
use crate::LearningStore;
use crate::{BehaviorPattern, ErrorSolution, Expertise, Procedure, UserProfile};
use aidaemon_core::ChannelVisibility;

mod claims;
mod episodes;
mod export;
mod facts;
mod learning;
mod people;
mod retention;
mod rows;
mod schema;
#[cfg(test)]
mod search_tests;
mod structured;

pub(crate) use claims::memory_visible;
pub use claims::{project_message_span, MemoryClaimRecord};
pub use export::ExportFile;
pub use facts::{
    build_fact_embedding_text, fact_category_is_personal, facts_are_restatements,
    is_owner_authored_fact_source,
};
pub(crate) use facts::{facts_make_the_same_claim, FACT_RESTATEMENT_MIN_SIMILARITY};
pub use retention::{MemoryRetention, MemoryRetentionPolicy, MemoryRetentionStats};
pub use schema::{migrate, rebuild_memory_fts_projections};

/// Dimension of the vectors in databases written so far (AllMiniLM-L6-v2).
const EMBEDDING_DIM: usize = 384;

fn decode_embedding(blob: &[u8]) -> anyhow::Result<Vec<f32>> {
    decode_embedding_with_dim(blob, EMBEDDING_DIM).map_err(anyhow::Error::from)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::cosine(a, b)
}

fn episode_search_score(similarity: f32, recall_count: i32) -> f32 {
    similarity * crate::episode_recall_boost(recall_count.max(0) as u32)
}

/// How the host's channel identifiers relate. The defaults suit a host whose
/// ids are stable and carry no channel in the session id.
pub trait ChannelIds: Send + Sync {
    /// Whether a row stored under `stored` belongs to the channel `current`.
    fn same_channel(&self, stored: &str, current: &str) -> bool {
        stored == current
    }

    /// The channel a session belongs to, when its id says.
    fn channel_of_session(&self, _session_id: &str) -> Option<String> {
        None
    }
}

/// Exact channel-id comparison, no channel derived from session ids.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExactChannelIds;

impl ChannelIds for ExactChannelIds {}

/// Cross-encoder reordering of search candidates. Optional: without one, or
/// when one fails, results keep their embedding order.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// `(input index, score)` pairs, most relevant first.
    async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> anyhow::Result<Vec<(usize, f32)>>;

    /// The score below which a reranked search candidate is not relevant
    /// and is dropped (exact-key matches are kept regardless). The scale is
    /// specific to the model: cross-encoders emit logits or probabilities
    /// with different ranges, so only the implementation can say. `None`
    /// disables the floor, and the search pool caps are the only gate.
    fn relevance_floor(&self) -> Option<crate::search::RelevanceFloor> {
        None
    }

    /// How long a search waits for one rerank before it keeps the
    /// unreranked order. Reranking only refines an answer the search already
    /// has, so a model starved of CPU (or queued behind another rerank) must
    /// cost the search a bounded wait, never minutes.
    fn deadline(&self) -> std::time::Duration {
        DEFAULT_RERANK_DEADLINE
    }
}

/// [`Reranker::deadline`]'s default: a few times a cross-encoder's normal
/// latency on a few dozen short facts.
pub const DEFAULT_RERANK_DEADLINE: std::time::Duration = std::time::Duration::from_secs(8);

/// The store's embedding and reranking models.
#[derive(Clone)]
pub(crate) struct StoreModels {
    embedder: Arc<dyn Embedder>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl StoreModels {
    fn model_id(&self) -> String {
        self.embedder.model_id().to_string()
    }

    async fn embed(&self, text: String) -> anyhow::Result<Vec<f32>> {
        self.embedder
            .embed(&[text])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("embedder returned no vector"))
    }

    /// One embedder call for many texts: one model hop instead of one per
    /// text.
    async fn embed_many(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let expected = texts.len();
        let vectors = self.embedder.embed(&texts).await?;
        anyhow::ensure!(
            vectors.len() == expected,
            "embedder returned {} vectors for {expected} texts",
            vectors.len()
        );
        Ok(vectors)
    }

    async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        match &self.reranker {
            Some(reranker) => {
                let deadline = reranker.deadline();
                tokio::time::timeout(deadline, reranker.rerank(query, documents))
                    .await
                    .map_err(|_| anyhow::anyhow!("rerank missed its {deadline:?} deadline"))?
            }
            None => anyhow::bail!("no reranker configured"),
        }
    }
}

/// Long-term memory in SQLite.
#[derive(Clone)]
pub struct SqliteMemory {
    pool: SqlitePool,
    /// One index for this database, so its vector cache is shared by every
    /// search and clone.
    embedding_index: crate::embedding_index::SqliteEmbeddingIndex,
    embedding_service: StoreModels,
    fact_upsert_lock: Arc<Mutex<()>>,
    channels: Arc<dyn ChannelIds>,
    clean_error_text: fn(&str) -> String,
    messages: Option<Arc<dyn crate::history::MessageLog>>,
}

impl SqliteMemory {
    /// Memory over `pool`, whose tables [`migrate`] has created.
    pub fn new(pool: SqlitePool, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedding_index: crate::embedding_index::SqliteEmbeddingIndex::new(pool.clone()),
            pool,
            embedding_service: StoreModels {
                embedder,
                reranker: None,
            },
            fact_upsert_lock: Arc::new(Mutex::new(())),
            channels: Arc::new(ExactChannelIds),
            clean_error_text: str::to_string,
            messages: None,
        }
    }

    /// Reorders explicit fact searches with `reranker`.
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.embedding_service.reranker = Some(reranker);
        self
    }

    /// Compares and derives channel ids the host's way.
    pub fn with_channel_ids(mut self, channels: Arc<dyn ChannelIds>) -> Self {
        self.channels = channels;
        self
    }

    /// Strips host text (wrappers, notices) from an error before its
    /// recurrence pattern is derived.
    pub fn with_error_cleaner(mut self, clean: fn(&str) -> String) -> Self {
        self.clean_error_text = clean;
        self
    }

    /// Links episodes to the host's messages (see [`MessageLog`]).
    ///
    /// [`MessageLog`]: crate::history::MessageLog
    pub fn with_message_log(mut self, messages: Arc<dyn crate::history::MessageLog>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// The first and last messages a span of a session covers, when the host
    /// gave a [`MessageLog`](crate::history::MessageLog). Without one an
    /// episode simply has no bounds.
    async fn message_event_bounds(
        &self,
        session_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<(Option<i64>, Option<i64>)> {
        match &self.messages {
            Some(messages) => messages.message_bounds(session_id, start, end).await,
            None => Ok((None, None)),
        }
    }

    /// The pool the store works on.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

fn parse_dt(s: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

/// Parse a date string and calculate days until it from today (wrapping year).
fn days_until_date(value: &str, today: chrono::NaiveDate) -> Option<i64> {
    use chrono::NaiveDate;

    let trimmed = value.trim();

    // Try "YYYY-MM-DD"
    if let Ok(d) = NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        let this_year = today.with_month(d.month())?.with_day(d.day())?;
        let diff = (this_year - today).num_days();
        return Some(if diff < 0 { diff + 365 } else { diff });
    }

    // Try "MM-DD" or "MM/DD"
    for fmt in &["%m-%d", "%m/%d"] {
        if let Ok(d) =
            NaiveDate::parse_from_str(&format!("2000-{}", trimmed), &format!("2000-{}", fmt))
        {
            let this_year = today.with_month(d.month())?.with_day(d.day())?;
            let diff = (this_year - today).num_days();
            return Some(if diff < 0 { diff + 365 } else { diff });
        }
    }

    // Try "Month DD" (e.g., "March 15")
    let months = [
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("may", 5),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
    ];
    let lower = trimmed.to_lowercase();
    for (name, num) in &months {
        if let Some(rest) = lower.strip_prefix(name) {
            let rest = rest.trim().trim_start_matches([',', ' ']);
            if let Ok(day) = rest.parse::<u32>() {
                let this_year = today.with_month(*num)?.with_day(day)?;
                let diff = (this_year - today).num_days();
                return Some(if diff < 0 { diff + 365 } else { diff });
            }
        }
    }

    None
}
