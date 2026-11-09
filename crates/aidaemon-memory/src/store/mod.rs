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
    DecayReport, Episode, EpisodeRecord, Fact, FactAccess, FactRecord, MemoryStats, NewEpisode,
    NewFact, RetentionConfig, ScoredEpisode, ScoredFact, StoredEmbedding, UpsertOutcome,
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
///
/// Access levels are optional too. A store that keeps the default
/// [`MemoryStore::fact_access`] cannot say who may see a record, so
/// [`crate::Memory`] shows its records only to the owner in a direct
/// conversation and never in a shared channel.
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

    /// [`MemoryStore::upsert_fact`] that records who may see the fact,
    /// atomically with the write. A reinforced fact keeps the narrowest of
    /// its access and `access` ([`FactAccess::narrowest`]); a new version
    /// takes `access`. The default ignores `access`, which leaves the fact
    /// unclassified and so private.
    async fn upsert_fact_with_access(
        &self,
        fact: NewFact,
        embedding: Option<StoredEmbedding>,
        _access: &FactAccess,
        now: DateTime<Utc>,
    ) -> Result<UpsertOutcome> {
        self.upsert_fact(fact, embedding, now).await
    }

    /// Who may see these facts, by id, for ids in `scope` the store knows. A
    /// fact absent from the map is unclassified and treated as private.
    async fn fact_access(
        &self,
        _scope: &str,
        _ids: &[String],
    ) -> Result<HashMap<String, FactAccess>> {
        Ok(HashMap::new())
    }

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

    /// [`MemoryStore::insert_episode`] that records who may see the episode.
    /// The default ignores `access`, leaving the episode private.
    async fn insert_episode_with_access(
        &self,
        episode: NewEpisode,
        embedding: Option<StoredEmbedding>,
        _access: &FactAccess,
        now: DateTime<Utc>,
    ) -> Result<Episode> {
        self.insert_episode(episode, embedding, now).await
    }

    /// Who may see these episodes; see [`MemoryStore::fact_access`].
    async fn episode_access(
        &self,
        _scope: &str,
        _ids: &[String],
    ) -> Result<HashMap<String, FactAccess>> {
        Ok(HashMap::new())
    }

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
