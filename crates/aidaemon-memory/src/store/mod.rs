//! The [`MemoryStore`] extension point and the two bundled stores.

mod in_memory;
mod migrations;
mod sqlite;

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::error::Result;
use crate::scoring::term_overlap;
use crate::types::{
    DecayReport, Episode, EpisodeRecord, Fact, FactRecord, MemoryStats, NewEpisode, NewFact,
    RetentionConfig, ScoredEpisode, ScoredFact, StoredEmbedding, UpsertOutcome,
};

pub use in_memory::InMemoryMemoryStore;
pub use migrations::{latest_version, Migration, MIGRATIONS};
pub use sqlite::SqliteMemoryStore;

/// Persistence for facts and episodes. **This is the extension point** for
/// bringing your own backend (pgvector, Qdrant, a document store, ...).
/// [`crate::Memory`] holds an `Arc<dyn MemoryStore>`; the trait is
/// object-safe.
///
/// Contract:
/// - Facts are keyed by `(scope, category, key)` (already normalized when
///   they arrive) and at most one is active per key.
/// - `upsert_fact` with a different value marks the active row superseded
///   (sets `superseded_at`) and inserts a new one; with the same value
///   (see [`crate::same_value`]) it refreshes `updated_at`, keeps the higher
///   confidence and returns [`UpsertOutcome::Reinforced`].
/// - Superseded facts are never returned by `active_facts` or searches, but
///   remain in `fact_history` until retention deletes them.
/// - Scopes are hard boundaries: never return another scope's rows.
///
/// Lexical scoring and searching have defaults, so a minimal store only
/// stores rows. A store with its own ranking (vector DB, hybrid search
/// engine) overrides [`MemoryStore::search_facts`] /
/// [`MemoryStore::search_episodes`] and the [`crate::Scorer`] is bypassed.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Insert, reinforce or supersede (see the trait docs). `embedding`
    /// describes the new value's search text, when one was computed.
    async fn upsert_fact(
        &self,
        fact: NewFact,
        embedding: Option<StoredEmbedding>,
        now: DateTime<Utc>,
    ) -> Result<UpsertOutcome>;

    /// Mark the active fact for this key superseded. `false` if none.
    async fn supersede_fact(
        &self,
        scope: &str,
        category: &str,
        key: &str,
        now: DateTime<Utc>,
    ) -> Result<bool>;

    /// Every active fact in `scope`, with embeddings.
    async fn active_facts(&self, scope: &str) -> Result<Vec<FactRecord>>;

    /// All versions of one key, oldest first, superseded ones included.
    async fn fact_history(&self, scope: &str, category: &str, key: &str) -> Result<Vec<Fact>>;

    /// Lexical relevance (0.0-1.0) of active facts to `query`, by fact id.
    /// Facts absent from the map score 0. The default is word overlap over
    /// [`Fact::search_text`].
    async fn fact_lexical_scores(&self, scope: &str, query: &str) -> Result<HashMap<String, f32>> {
        Ok(self
            .active_facts(scope)
            .await?
            .into_iter()
            .map(|r| {
                let score = term_overlap(query, &r.fact.search_text());
                (r.fact.id, score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect())
    }

    /// Native ranked search. Return `Some` to bypass the scorer: the result
    /// is used as is (already filtered, ordered, at most `limit`). The
    /// default `None` lets [`crate::Memory`] score candidates itself.
    async fn search_facts(
        &self,
        _scope: &str,
        _query: &str,
        _query_embedding: Option<&StoredEmbedding>,
        _limit: usize,
    ) -> Result<Option<Vec<ScoredFact>>> {
        Ok(None)
    }

    /// Bump `recall_count` and set `last_recalled_at` for these facts.
    async fn record_fact_recalls(&self, ids: &[String], now: DateTime<Utc>) -> Result<()>;

    async fn insert_episode(
        &self,
        episode: NewEpisode,
        embedding: Option<StoredEmbedding>,
        now: DateTime<Utc>,
    ) -> Result<Episode>;

    /// Every episode in `scope`, with embeddings.
    async fn episodes(&self, scope: &str) -> Result<Vec<EpisodeRecord>>;

    /// Lexical relevance of episodes; default word overlap.
    async fn episode_lexical_scores(
        &self,
        scope: &str,
        query: &str,
    ) -> Result<HashMap<String, f32>> {
        Ok(self
            .episodes(scope)
            .await?
            .into_iter()
            .map(|r| {
                let score = term_overlap(query, &r.episode.search_text());
                (r.episode.id, score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect())
    }

    /// Native ranked episode search; see [`MemoryStore::search_facts`].
    async fn search_episodes(
        &self,
        _scope: &str,
        _query: &str,
        _query_embedding: Option<&StoredEmbedding>,
        _limit: usize,
    ) -> Result<Option<Vec<ScoredEpisode>>> {
        Ok(None)
    }

    async fn record_episode_recalls(&self, ids: &[String], now: DateTime<Utc>) -> Result<()>;

    /// Apply recall decay and retention (see [`RetentionConfig`]). The
    /// default does nothing, for stores whose backend handles expiry.
    async fn decay(&self, _now: DateTime<Utc>, _config: &RetentionConfig) -> Result<DecayReport> {
        Ok(DecayReport::default())
    }

    async fn stats(&self) -> Result<MemoryStats>;
}
