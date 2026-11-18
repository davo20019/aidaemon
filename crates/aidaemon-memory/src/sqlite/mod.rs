//! The SQLite implementation of the [`crate::stores`] contract: facts with
//! privacy and channel provenance, episodes, the claim graph, structured
//! personal memory, people and learned habits.
//!
//! [`SqliteMemory`] works on a pool the host opens, so memory can share one
//! database file (and its encryption) with the host's other tables.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use tokio::sync::Mutex;

use crate::LearningStore;
use crate::embed::{Embedder, decode_embedding_with_dim, encode_embedding};
use crate::records::*;
use crate::stores::*;
use crate::text::truncate_chars as truncate_str;
use crate::{BehaviorPattern, ErrorSolution, Expertise, Procedure, UserProfile};
use aidaemon_core::ChannelVisibility;

mod claims;
mod episodes;
mod export;
mod fact_folds;
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
pub use claims::{MemoryClaimRecord, project_message_span};
pub use export::ExportFile;
pub use fact_folds::{
    FoldCandidate, FoldRule, ONE_VALUE_MIN_SIMILARITY, ONE_WORD_VALUE_MIN_SIMILARITY, PlannedFold,
    facts_hold_one_value, plan_folds,
};
pub(crate) use facts::{FACT_RESTATEMENT_MIN_SIMILARITY, facts_make_the_same_claim};
pub use facts::{
    OWNER_QUOTED_SOURCE, build_fact_embedding_text, fact_category_is_personal, fact_source_rank,
    facts_are_restatements, is_owner_authored_fact_source, keys_name_same_attribute,
};
pub use retention::{MemoryRetention, MemoryRetentionPolicy, MemoryRetentionStats};
pub use schema::{migrate, rebuild_memory_fts_projections};

/// Dimension of the test embedders' vectors.
#[cfg(test)]
const EMBEDDING_DIM: usize = 384;

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

/// The store's embedding model.
#[derive(Clone)]
pub(crate) struct StoreModels {
    embedder: Arc<dyn Embedder>,
    /// Recent single-text embeddings, shared by every clone.
    queries: Arc<QueryMemo>,
}

/// How many recent query texts keep their vectors.
const QUERY_MEMO_CAPACITY: usize = 8;

/// The last few single-text embeddings. One turn's recall searches facts,
/// episodes, procedures and error solutions for the same message at once;
/// with this they share one model run instead of queueing four behind the
/// model's lock. A text
/// always has the same vector, so an entry never goes stale; one being
/// computed is awaited, not recomputed, and a failed run leaves its entry
/// empty for the next caller to retry.
#[derive(Default)]
struct QueryMemo {
    entries: std::sync::Mutex<std::collections::VecDeque<(String, VectorCell)>>,
}

/// One text's vector, filled by whichever caller runs the model first.
type VectorCell = Arc<tokio::sync::OnceCell<Vec<f32>>>;

impl QueryMemo {
    fn cell(&self, text: &str) -> VectorCell {
        let Ok(mut entries) = self.entries.lock() else {
            return Arc::new(tokio::sync::OnceCell::new());
        };
        if let Some(position) = entries.iter().position(|(key, _)| key == text) {
            let entry = entries.remove(position).expect("position is in range");
            let cell = entry.1.clone();
            entries.push_back(entry);
            return cell;
        }
        let cell = Arc::new(tokio::sync::OnceCell::new());
        entries.push_back((text.to_string(), cell.clone()));
        while entries.len() > QUERY_MEMO_CAPACITY {
            entries.pop_front();
        }
        cell
    }
}

impl StoreModels {
    fn model_id(&self) -> String {
        self.embedder.model_id().to_string()
    }

    /// A stored vector of this model's dimension (binary, or the legacy
    /// JSON form); anything else is an error, never a vector of another
    /// length.
    fn decode(&self, blob: &[u8]) -> anyhow::Result<Vec<f32>> {
        decode_embedding_with_dim(blob, self.embedder.dimensions()).map_err(anyhow::Error::from)
    }

    async fn embed(&self, text: String) -> anyhow::Result<Vec<f32>> {
        let cell = self.queries.cell(&text);
        cell.get_or_try_init(|| async {
            self.embedder
                .embed(&[text])
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("embedder returned no vector"))
        })
        .await
        .cloned()
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
                queries: Arc::default(),
            },
            fact_upsert_lock: Arc::new(Mutex::new(())),
            channels: Arc::new(ExactChannelIds),
            clean_error_text: str::to_string,
            messages: None,
        }
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

#[cfg(test)]
mod query_memo_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::{QUERY_MEMO_CAPACITY, StoreModels};
    use crate::Embedder;

    struct Counting {
        runs: AtomicUsize,
        fail_first: bool,
    }

    #[async_trait]
    impl Embedder for Counting {
        fn model_id(&self) -> &str {
            "counting"
        }
        fn dimensions(&self) -> usize {
            1
        }
        async fn embed(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
            let run = self.runs.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            if self.fail_first && run == 0 {
                return Err(crate::MemoryError::Embedding("model not loaded yet".into()));
            }
            Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
        }
    }

    fn models(fail_first: bool) -> (StoreModels, Arc<Counting>) {
        let embedder = Arc::new(Counting {
            runs: AtomicUsize::new(0),
            fail_first,
        });
        let models = StoreModels {
            embedder: embedder.clone(),
            queries: Arc::default(),
        };
        (models, embedder)
    }

    #[tokio::test]
    async fn concurrent_searches_for_one_text_share_one_model_run() {
        let (models, embedder) = models(false);
        let text = "what is on my calendar".to_string();
        let (a, b, c, d) = tokio::join!(
            models.embed(text.clone()),
            models.embed(text.clone()),
            models.embed(text.clone()),
            models.embed(text.clone()),
        );
        let results = [a.unwrap(), b.unwrap(), c.unwrap(), d.unwrap()];
        assert!(results.iter().all(|vector| vector == &results[0]));
        assert_eq!(embedder.runs.load(Ordering::SeqCst), 1);
        // A different text runs the model again.
        models.embed("another question".to_string()).await.unwrap();
        assert_eq!(embedder.runs.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_failed_run_is_retried_and_old_texts_are_evicted() {
        let (models, embedder) = models(true);
        assert!(models.embed("first".to_string()).await.is_err());
        assert!(models.embed("first".to_string()).await.is_ok());
        assert_eq!(embedder.runs.load(Ordering::SeqCst), 2);
        for n in 0..QUERY_MEMO_CAPACITY {
            models.embed(format!("text {n}")).await.unwrap();
        }
        let before = embedder.runs.load(Ordering::SeqCst);
        models.embed("first".to_string()).await.unwrap();
        assert_eq!(embedder.runs.load(Ordering::SeqCst), before + 1);
    }
}
