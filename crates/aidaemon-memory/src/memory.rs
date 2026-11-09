//! [`Memory`]: the main handle tying a store, an embedder and a scorer
//! together, plus the optional model-driven writers.

use std::collections::HashMap;
use std::sync::Arc;

use aidaemon_core::{Message, Receipt, RunOutcome, Usage};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::embed::{cosine, Embedder};
use crate::error::{MemoryError, Result};
use crate::extract::{EpisodeSummarizer, FactExtractor};
use crate::scoring::{HybridScorer, Scorer, ScoringConfig};
use crate::store::MemoryStore;
use crate::types::{
    episode_search_text, fact_search_text, normalize_category, normalize_key, truncate_chars,
    DecayReport, Episode, Fact, FactAccess, MemoryStats, NewEpisode, NewFact, Provenance,
    RecallAudience, RetentionConfig, ScoredEpisode, ScoredFact, Signals, StoredEmbedding,
    UpsertOutcome,
};

/// Maps a session id (from `ToolContext` / `HookContext`) to a memory
/// scope. The default is the identity: one memory per session id.
pub type ScopeResolver = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Maps a session id to who that conversation reaches, which decides what
/// recall may return and how new memories are classified.
pub type AudienceResolver = Arc<dyn Fn(&str) -> RecallAudience + Send + Sync>;

/// How many candidates to ask a ranking store for when some may be filtered
/// out afterwards, as a multiple of the limit.
const FILTERED_SEARCH_OVERFETCH: usize = 4;

/// The strongest query matches whose relations feed the graph signal.
const MAX_GRAPH_SEEDS: usize = 6;

/// A seed must match the query at least this well (semantic or lexical).
/// The scorer is not asked, so a custom one sees each candidate once.
const GRAPH_SEED_MIN_SIGNAL: f32 = 0.3;

struct Inner {
    store: Arc<dyn MemoryStore>,
    embedder: Arc<dyn Embedder>,
    scorer: Arc<dyn Scorer>,
    retention: RetentionConfig,
    scope_resolver: ScopeResolver,
    audience_resolver: AudienceResolver,
    extractor: Option<Arc<FactExtractor>>,
    summarizer: Option<Arc<EpisodeSummarizer>>,
}

/// Long-term memory. Depends only on traits ([`MemoryStore`],
/// [`Embedder`], [`Scorer`]), so every piece is swappable. Cheap to clone;
/// clones share everything.
#[derive(Clone)]
pub struct Memory {
    inner: Arc<Inner>,
}

/// Builder for [`Memory`].
pub struct MemoryBuilder {
    store: Arc<dyn MemoryStore>,
    embedder: Arc<dyn Embedder>,
    scorer: Arc<dyn Scorer>,
    retention: RetentionConfig,
    scope_resolver: ScopeResolver,
    audience_resolver: AudienceResolver,
    extractor: Option<Arc<FactExtractor>>,
    summarizer: Option<Arc<EpisodeSummarizer>>,
}

impl MemoryBuilder {
    /// Replace the ranking ([`HybridScorer`] by default).
    pub fn scorer(mut self, scorer: Arc<dyn Scorer>) -> Self {
        self.scorer = scorer;
        self
    }

    /// Keep the default scorer but change its weights and thresholds.
    pub fn scoring(mut self, config: ScoringConfig) -> Self {
        self.scorer = Arc::new(HybridScorer::new(config));
        self
    }

    pub fn retention(mut self, retention: RetentionConfig) -> Self {
        self.retention = retention;
        self
    }

    /// How session ids map to scopes (for example, many sessions of one
    /// user sharing `telegram:synthetic-user-1`).
    pub fn scope_resolver(mut self, f: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        self.scope_resolver = Arc::new(f);
        self
    }

    /// Who each session's conversation reaches. A host with shared channels
    /// must set this: without it every session counts as the owner in a
    /// direct conversation, so only the scope separates them.
    pub fn audience_resolver(
        mut self,
        f: impl Fn(&str) -> RecallAudience + Send + Sync + 'static,
    ) -> Self {
        self.audience_resolver = Arc::new(f);
        self
    }

    /// Enables [`Memory::ingest_turn`] / [`Memory::ingest_run`] fact
    /// extraction and the [`crate::MemoryWriter`] hook.
    pub fn extractor(mut self, extractor: FactExtractor) -> Self {
        self.extractor = Some(Arc::new(extractor));
        self
    }

    /// Enables episode summaries in [`Memory::ingest_run`].
    pub fn summarizer(mut self, summarizer: EpisodeSummarizer) -> Self {
        self.summarizer = Some(Arc::new(summarizer));
        self
    }

    pub fn build(self) -> Memory {
        Memory {
            inner: Arc::new(Inner {
                store: self.store,
                embedder: self.embedder,
                scorer: self.scorer,
                retention: self.retention,
                scope_resolver: self.scope_resolver,
                audience_resolver: self.audience_resolver,
                extractor: self.extractor,
                summarizer: self.summarizer,
            }),
        }
    }
}

/// What [`Memory::ingest_turn`] / [`Memory::ingest_run`] did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IngestReport {
    pub scope: String,
    /// One entry per stored fact.
    pub facts: Vec<UpsertOutcome>,
    /// Model entries that were malformed or failed validation.
    pub rejected: usize,
    pub episode: Option<Episode>,
    /// Model usage spent on extraction and summarization.
    pub usage: Usage,
    /// Why nothing was extracted, when skipped on purpose.
    pub skipped: Option<String>,
}

impl Memory {
    pub fn builder(store: Arc<dyn MemoryStore>, embedder: Arc<dyn Embedder>) -> MemoryBuilder {
        MemoryBuilder {
            store,
            embedder,
            scorer: Arc::new(HybridScorer::default()),
            retention: RetentionConfig::default(),
            scope_resolver: Arc::new(|session: &str| session.to_string()),
            audience_resolver: Arc::new(|_: &str| RecallAudience::owner()),
            extractor: None,
            summarizer: None,
        }
    }

    /// Memory with the default scorer, retention and identity scopes.
    pub fn new(store: impl MemoryStore + 'static, embedder: impl Embedder + 'static) -> Self {
        Self::builder(Arc::new(store), Arc::new(embedder)).build()
    }

    pub fn store(&self) -> &Arc<dyn MemoryStore> {
        &self.inner.store
    }

    pub fn embedder(&self) -> &Arc<dyn Embedder> {
        &self.inner.embedder
    }

    pub fn extractor(&self) -> Option<&FactExtractor> {
        self.inner.extractor.as_deref()
    }

    /// The scope for a session id.
    pub fn scope_for(&self, session_id: &str) -> String {
        (self.inner.scope_resolver)(session_id)
    }

    /// Who a session's conversation reaches.
    pub fn audience_for(&self, session_id: &str) -> RecallAudience {
        (self.inner.audience_resolver)(session_id)
    }

    /// Which ids `audience` may see, given the store's answer about their
    /// access. A failed lookup shows nothing rather than everything.
    fn visible_ids(
        access: Result<HashMap<String, FactAccess>>,
        audience: &RecallAudience,
    ) -> impl Fn(&str) -> bool {
        let access = access.unwrap_or_else(|error| {
            tracing::warn!(%error, "access lookup failed; hiding restricted memories");
            HashMap::new()
        });
        let audience = audience.clone();
        move |id| audience.may_see(access.get(id))
    }

    async fn embed_one(&self, text: &str) -> Option<StoredEmbedding> {
        let embedder = &self.inner.embedder;
        match embedder.embed(&[text.to_string()]).await {
            Ok(mut vectors) if vectors.len() == 1 => Some(StoredEmbedding {
                model: embedder.model_id().to_string(),
                vector: vectors.remove(0),
            }),
            Ok(_) => {
                tracing::warn!("embedder returned the wrong number of vectors");
                None
            }
            Err(error) => {
                tracing::warn!(%error, "embedding failed; falling back to lexical recall");
                None
            }
        }
    }

    fn semantic(query: Option<&StoredEmbedding>, stored: Option<&StoredEmbedding>) -> Option<f32> {
        let (q, s) = (query?, stored?);
        (q.model == s.model && q.vector.len() == s.vector.len())
            .then(|| cosine(&q.vector, &s.vector).max(0.0))
    }

    /// Store a fact now. See [`Memory::remember_at`].
    pub async fn remember(&self, fact: NewFact) -> Result<UpsertOutcome> {
        self.remember_at(fact, Utc::now()).await
    }

    /// Normalize, embed and upsert a private fact as of `now`. An embedding
    /// failure stores the fact without a vector (lexical recall still finds
    /// it).
    pub async fn remember_at(&self, fact: NewFact, now: DateTime<Utc>) -> Result<UpsertOutcome> {
        self.write_fact(fact, &FactAccess::private(), now).await
    }

    /// Store a fact now, stating who may see it.
    pub async fn remember_with(&self, fact: NewFact, access: &FactAccess) -> Result<UpsertOutcome> {
        self.write_fact(fact, access, Utc::now()).await
    }

    async fn write_fact(
        &self,
        fact: NewFact,
        access: &FactAccess,
        now: DateTime<Utc>,
    ) -> Result<UpsertOutcome> {
        let fact = fact.normalized()?;
        let embedding = self
            .embed_one(&fact_search_text(&fact.category, &fact.key, &fact.value))
            .await;
        self.inner
            .store
            .upsert_fact_with_access(fact, embedding, access, now)
            .await
    }

    /// Supersede the active fact for this key. History is kept.
    pub async fn forget(&self, scope: &str, category: &str, key: &str) -> Result<bool> {
        self.inner
            .store
            .supersede_fact(
                scope.trim(),
                &normalize_category(category),
                &normalize_key(key),
                Utc::now(),
            )
            .await
    }

    /// Active facts in a scope, most recently updated first.
    pub async fn facts(&self, scope: &str) -> Result<Vec<Fact>> {
        let mut facts: Vec<Fact> = self
            .inner
            .store
            .active_facts(scope)
            .await?
            .into_iter()
            .map(|r| r.fact)
            .collect();
        facts.sort_by_key(|f| std::cmp::Reverse(f.updated_at));
        Ok(facts)
    }

    /// Every version of a key, superseded ones included.
    pub async fn fact_history(&self, scope: &str, category: &str, key: &str) -> Result<Vec<Fact>> {
        self.inner
            .store
            .fact_history(scope, &normalize_category(category), &normalize_key(key))
            .await
    }

    /// The facts most relevant to `query`, best first, as the owner in a
    /// direct conversation sees them. Each returned fact's recall count is
    /// incremented. An empty query returns the most recently updated facts.
    ///
    /// Use [`Memory::recall_facts_for`] for anything that reaches a channel.
    pub async fn recall_facts(
        &self,
        scope: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredFact>> {
        self.recall_facts_for(scope, query, limit, &RecallAudience::owner())
            .await
    }

    /// [`Memory::recall_facts`] limited to what `audience` may see. Restricted
    /// facts are dropped before ranking, so they neither fill the limit nor
    /// gain recall counts.
    pub async fn recall_facts_for(
        &self,
        scope: &str,
        query: &str,
        limit: usize,
        audience: &RecallAudience,
    ) -> Result<Vec<ScoredFact>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let now = Utc::now();
        let store = &self.inner.store;
        let query_embedding = if query.trim().is_empty() {
            None
        } else {
            self.embed_one(query).await
        };

        let restricted = !audience.sees_everything();
        let fetch = if restricted {
            limit.saturating_mul(FILTERED_SEARCH_OVERFETCH)
        } else {
            limit
        };
        let mut hits = match store
            .search_facts(scope, query, query_embedding.as_ref(), fetch)
            .await?
        {
            Some(mut native) => {
                if restricted {
                    let ids: Vec<String> = native.iter().map(|h| h.fact.id.clone()).collect();
                    let visible = Self::visible_ids(store.fact_access(scope, &ids).await, audience);
                    native.retain(|h| visible(&h.fact.id));
                }
                native
            }
            None => {
                let mut records = store.active_facts(scope).await?;
                if restricted {
                    let ids: Vec<String> = records.iter().map(|r| r.fact.id.clone()).collect();
                    let visible = Self::visible_ids(store.fact_access(scope, &ids).await, audience);
                    records.retain(|r| visible(&r.fact.id));
                }
                if query.trim().is_empty() {
                    let mut facts: Vec<Fact> = records.into_iter().map(|r| r.fact).collect();
                    facts.sort_by_key(|f| std::cmp::Reverse(f.updated_at));
                    facts
                        .into_iter()
                        .take(limit)
                        .map(|fact| ScoredFact {
                            fact,
                            score: 0.0,
                            signals: Signals::default(),
                        })
                        .collect()
                } else {
                    let lexical = self.lexical(store.fact_lexical_scores(scope, query).await);
                    let signals: Vec<(Option<f32>, f32)> = records
                        .iter()
                        .map(|r| {
                            let semantic =
                                Self::semantic(query_embedding.as_ref(), r.embedding.as_ref());
                            let lex = lexical.get(&r.fact.id).copied().unwrap_or(0.0);
                            (semantic, lex)
                        })
                        .collect();
                    let mut seeds: Vec<(f32, String)> = records
                        .iter()
                        .zip(&signals)
                        .map(|(r, &(semantic, lex))| {
                            (semantic.unwrap_or(0.0).max(lex), r.fact.id.clone())
                        })
                        .filter(|(strength, _)| *strength > GRAPH_SEED_MIN_SIGNAL)
                        .collect();
                    seeds.sort_by(|a, b| b.0.total_cmp(&a.0));
                    seeds.truncate(MAX_GRAPH_SEEDS);
                    let seed_ids: Vec<String> = seeds.into_iter().map(|(_, id)| id).collect();
                    let graph = if seed_ids.is_empty() {
                        HashMap::new()
                    } else {
                        store
                            .fact_graph_scores(scope, &seed_ids)
                            .await
                            .unwrap_or_else(|error| {
                                tracing::warn!(%error, "graph lookup failed; ranking without it");
                                HashMap::new()
                            })
                    };
                    let mut scored: Vec<ScoredFact> = records
                        .into_iter()
                        .zip(signals)
                        .filter_map(|(r, (semantic, lex))| {
                            let linked = graph.get(&r.fact.id).copied().unwrap_or(0.0);
                            let score = self
                                .inner
                                .scorer
                                .score_fact_with_graph(&r.fact, semantic, lex, linked, now)?;
                            Some(ScoredFact {
                                fact: r.fact,
                                score: score.value,
                                signals: score.signals,
                            })
                        })
                        .collect();
                    scored.sort_by(|a, b| {
                        b.score
                            .total_cmp(&a.score)
                            .then(b.fact.updated_at.cmp(&a.fact.updated_at))
                    });
                    scored.truncate(limit);
                    scored
                }
            }
        };
        hits.truncate(limit);

        let ids: Vec<String> = hits.iter().map(|h| h.fact.id.clone()).collect();
        if !ids.is_empty() {
            match store.record_fact_recalls(&ids, now).await {
                Ok(()) => hits.iter_mut().for_each(|h| {
                    h.fact.recall_count += 1;
                    h.fact.last_recalled_at = Some(now);
                }),
                Err(error) => tracing::warn!(%error, "failed to record fact recalls"),
            }
        }
        Ok(hits)
    }

    fn lexical(&self, scores: Result<HashMap<String, f32>>) -> HashMap<String, f32> {
        scores.unwrap_or_else(|error| {
            tracing::warn!(%error, "lexical search failed; using semantic scores only");
            HashMap::new()
        })
    }

    /// Store a private episode now.
    pub async fn add_episode(&self, episode: NewEpisode) -> Result<Episode> {
        self.add_episode_with(episode, &FactAccess::private()).await
    }

    /// Store an episode now, stating who may see it.
    pub async fn add_episode_with(
        &self,
        episode: NewEpisode,
        access: &FactAccess,
    ) -> Result<Episode> {
        let mut episode = episode;
        episode.scope = episode.scope.trim().to_string();
        episode.summary = episode.summary.trim().to_string();
        if episode.scope.is_empty() || episode.summary.is_empty() {
            return Err(MemoryError::Invalid(
                "episode scope and summary must be non-empty".into(),
            ));
        }
        episode.importance = if episode.importance.is_finite() {
            episode.importance.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let embedding = self
            .embed_one(&episode_search_text(&episode.summary, &episode.topics))
            .await;
        self.inner
            .store
            .insert_episode_with_access(episode, embedding, access, Utc::now())
            .await
    }

    /// The episodes most relevant to `query`, best first, as the owner in a
    /// direct conversation sees them. Recall counts of returned episodes are
    /// incremented.
    pub async fn recall_episodes(
        &self,
        scope: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredEpisode>> {
        self.recall_episodes_for(scope, query, limit, &RecallAudience::owner())
            .await
    }

    /// [`Memory::recall_episodes`] limited to what `audience` may see.
    pub async fn recall_episodes_for(
        &self,
        scope: &str,
        query: &str,
        limit: usize,
        audience: &RecallAudience,
    ) -> Result<Vec<ScoredEpisode>> {
        if limit == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let now = Utc::now();
        let store = &self.inner.store;
        let query_embedding = self.embed_one(query).await;
        let restricted = !audience.sees_everything();
        let fetch = if restricted {
            limit.saturating_mul(FILTERED_SEARCH_OVERFETCH)
        } else {
            limit
        };
        let mut hits = match store
            .search_episodes(scope, query, query_embedding.as_ref(), fetch)
            .await?
        {
            Some(mut native) => {
                if restricted {
                    let ids: Vec<String> = native.iter().map(|h| h.episode.id.clone()).collect();
                    let visible =
                        Self::visible_ids(store.episode_access(scope, &ids).await, audience);
                    native.retain(|h| visible(&h.episode.id));
                }
                native
            }
            None => {
                let lexical = self.lexical(store.episode_lexical_scores(scope, query).await);
                let mut records = store.episodes(scope).await?;
                if restricted {
                    let ids: Vec<String> = records.iter().map(|r| r.episode.id.clone()).collect();
                    let visible =
                        Self::visible_ids(store.episode_access(scope, &ids).await, audience);
                    records.retain(|r| visible(&r.episode.id));
                }
                let mut scored: Vec<ScoredEpisode> = records
                    .into_iter()
                    .filter_map(|r| {
                        let semantic =
                            Self::semantic(query_embedding.as_ref(), r.embedding.as_ref());
                        let lex = lexical.get(&r.episode.id).copied().unwrap_or(0.0);
                        let score = self
                            .inner
                            .scorer
                            .score_episode(&r.episode, semantic, lex, now)?;
                        Some(ScoredEpisode {
                            episode: r.episode,
                            score: score.value,
                            signals: score.signals,
                        })
                    })
                    .collect();
                scored.sort_by(|a, b| {
                    b.score
                        .total_cmp(&a.score)
                        .then(b.episode.created_at.cmp(&a.episode.created_at))
                });
                scored
            }
        };
        hits.truncate(limit);
        let ids: Vec<String> = hits.iter().map(|h| h.episode.id.clone()).collect();
        if !ids.is_empty() {
            match store.record_episode_recalls(&ids, now).await {
                Ok(()) => hits.iter_mut().for_each(|h| {
                    h.episode.recall_count += 1;
                    h.episode.last_recalled_at = Some(now);
                }),
                Err(error) => tracing::warn!(%error, "failed to record episode recalls"),
            }
        }
        Ok(hits)
    }

    /// Apply recall decay and retention as of `now`. Meant to run about
    /// once a day; each call is one decay step.
    pub async fn decay(&self, now: DateTime<Utc>) -> Result<DecayReport> {
        self.inner.store.decay(now, &self.inner.retention).await
    }

    pub async fn stats(&self) -> Result<MemoryStats> {
        self.inner.store.stats().await
    }

    /// Extract durable facts from the latest user turn of `messages` and
    /// store them with provenance. Needs a [`FactExtractor`].
    pub async fn ingest_turn(
        &self,
        session_id: &str,
        run_id: Option<&str>,
        messages: &[Message],
    ) -> Result<IngestReport> {
        let extractor = self
            .inner
            .extractor
            .as_ref()
            .ok_or(MemoryError::NotConfigured("no FactExtractor configured"))?;
        let scope = self.scope_for(session_id);
        let mut report = IngestReport {
            scope: scope.clone(),
            ..IngestReport::default()
        };
        let Some(start) = aidaemon_policies::last_user_message_index(messages) else {
            report.skipped = Some("no user message".into());
            return Ok(report);
        };
        let turn = &messages[start..];
        if !extractor.should_extract(turn) {
            report.skipped = Some("user message too short to hold a durable fact".into());
            return Ok(report);
        }
        let known: Vec<(String, String)> = self
            .inner
            .store
            .active_facts(&scope)
            .await?
            .into_iter()
            .take(50)
            .map(|r| (r.fact.category, r.fact.key))
            .collect();
        let extraction = extractor.extract(turn, &known).await?;
        report.usage.add(extraction.usage);
        report.rejected = extraction.rejected;
        let default_excerpt = truncate_chars(messages[start].content.trim(), 300);
        for fact in extraction.facts {
            let new = NewFact {
                scope: scope.clone(),
                category: fact.category,
                key: fact.key,
                value: fact.value,
                confidence: fact.confidence,
                source: Provenance {
                    run_id: run_id.map(str::to_string),
                    session_id: Some(session_id.to_string()),
                    excerpt: Some(fact.evidence.unwrap_or_else(|| default_excerpt.clone())),
                },
            };
            let access = self.audience_for(session_id).access_for_writes();
            match self.remember_with(new, &access).await {
                Ok(outcome) => report.facts.push(outcome),
                Err(MemoryError::Invalid(reason)) => {
                    tracing::debug!(%reason, "extracted fact rejected");
                    report.rejected += 1;
                }
                Err(other) => return Err(other),
            }
        }
        Ok(report)
    }

    /// Summarize a session into an [`Episode`] and store it. Needs an
    /// [`EpisodeSummarizer`].
    pub async fn summarize_session(
        &self,
        session_id: &str,
        messages: &[Message],
        receipts: &[Receipt],
    ) -> Result<(Episode, Usage)> {
        let summarizer = self
            .inner
            .summarizer
            .as_ref()
            .ok_or(MemoryError::NotConfigured(
                "no EpisodeSummarizer configured",
            ))?;
        let draft = summarizer.summarize(messages, receipts).await?;
        let access = self.audience_for(session_id).access_for_writes();
        let episode = self
            .add_episode_with(
                NewEpisode {
                    scope: self.scope_for(session_id),
                    session_id: session_id.to_string(),
                    summary: draft.summary,
                    topics: draft.topics,
                    importance: draft.importance,
                },
                &access,
            )
            .await?;
        Ok((episode, draft.usage))
    }

    /// Explicit alternative to [`crate::MemoryWriter`]: extract facts from
    /// the run's latest user turn (with a [`FactExtractor`]) and, for a
    /// completed run, store an episode (with an [`EpisodeSummarizer`]).
    pub async fn ingest_run(&self, outcome: &RunOutcome) -> Result<IngestReport> {
        let state = &outcome.state;
        if self.inner.extractor.is_none() && self.inner.summarizer.is_none() {
            return Err(MemoryError::NotConfigured(
                "ingest_run needs a FactExtractor or an EpisodeSummarizer",
            ));
        }
        let run_id = state.run_id.to_string();
        let mut report = if self.inner.extractor.is_some() {
            self.ingest_turn(&state.session_id, Some(&run_id), &state.messages)
                .await?
        } else {
            IngestReport {
                scope: self.scope_for(&state.session_id),
                ..IngestReport::default()
            }
        };
        if self.inner.summarizer.is_some() && outcome.is_completed() {
            let (episode, usage) = self
                .summarize_session(&state.session_id, &state.messages, &state.receipts)
                .await?;
            report.usage.add(usage);
            report.episode = Some(episode);
        }
        Ok(report)
    }
}
