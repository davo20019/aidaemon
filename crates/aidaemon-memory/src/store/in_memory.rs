use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use super::{related_fact_scores, MemoryStore};
use crate::error::{MemoryError, Result};
use crate::types::{
    new_id, normalize_category, normalize_key, same_value, DecayReport, Episode, EpisodeRecord,
    Fact, FactAccess, FactRecord, MemoryStats, NewEpisode, NewFact, RecordKind, RetentionConfig,
    StaleEmbedding, StoredEmbedding, UpsertOutcome,
};

#[derive(Default)]
struct Inner {
    facts: Vec<FactRecord>,
    episodes: Vec<EpisodeRecord>,
    /// By record id. Rows without an entry are private.
    fact_access: HashMap<String, FactAccess>,
    episode_access: HashMap<String, FactAccess>,
}

/// A process-local [`MemoryStore`]. Same semantics as the SQLite store,
/// nothing persisted. Lexical scores use word overlap.
#[derive(Default)]
pub struct InMemoryMemoryStore {
    inner: Mutex<Inner>,
}

impl InMemoryMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>> {
        self.inner
            .lock()
            .map_err(|_| MemoryError::Store("in-memory store lock poisoned".into()))
    }
}

/// Decay a recall count by a tenth, at least one (daemon `decay_memories`).
pub(crate) fn decayed_fact_recalls(count: u32) -> u32 {
    count.saturating_sub((count / 10).max(1))
}

#[async_trait]
impl MemoryStore for InMemoryMemoryStore {
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
        let mut inner = self.lock()?;
        let Inner {
            facts, fact_access, ..
        } = &mut *inner;
        let active = facts.iter_mut().find(|r| {
            r.fact.is_active()
                && r.fact.scope == fact.scope
                && r.fact.category == fact.category
                && r.fact.key == fact.key
        });
        let mut previous = None;
        if let Some(record) = active {
            if same_value(&record.fact.value, &fact.value) {
                record.fact.updated_at = now;
                record.fact.confidence = record.fact.confidence.max(fact.confidence);
                if record.fact.source.excerpt.is_none() {
                    record.fact.source.excerpt = fact.source.excerpt;
                }
                if embedding.is_some() {
                    record.embedding = embedding;
                }
                let id = record.fact.id.clone();
                let narrowed = fact_access
                    .get(&id)
                    .cloned()
                    .unwrap_or_default()
                    .narrowest(access);
                fact_access.insert(id.clone(), narrowed);
                return Ok(UpsertOutcome::Reinforced { id });
            }
            record.fact.superseded_at = Some(now);
            previous = Some((record.fact.id.clone(), record.fact.value.clone()));
        }
        let id = new_id();
        fact_access.insert(id.clone(), access.clone());
        facts.push(FactRecord {
            fact: Fact {
                id: id.clone(),
                scope: fact.scope,
                category: fact.category,
                key: fact.key,
                value: fact.value,
                confidence: fact.confidence,
                source: fact.source,
                created_at: now,
                updated_at: now,
                superseded_at: None,
                recall_count: 0,
                last_recalled_at: None,
            },
            embedding,
        });
        Ok(match previous {
            None => UpsertOutcome::Inserted { id },
            Some((previous_id, previous_value)) => UpsertOutcome::Superseded {
                id,
                previous_id,
                previous_value,
            },
        })
    }

    async fn fact_access(
        &self,
        scope: &str,
        ids: &[String],
    ) -> Result<HashMap<String, FactAccess>> {
        let inner = self.lock()?;
        let wanted: HashSet<&String> = ids.iter().collect();
        Ok(inner
            .facts
            .iter()
            .filter(|r| r.fact.scope == scope && wanted.contains(&r.fact.id))
            .map(|r| {
                let access = inner.fact_access.get(&r.fact.id).cloned();
                (r.fact.id.clone(), access.unwrap_or_default())
            })
            .collect())
    }

    async fn supersede_fact(
        &self,
        scope: &str,
        category: &str,
        key: &str,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        let (category, key) = (normalize_category(category), normalize_key(key));
        let mut inner = self.lock()?;
        let mut changed = false;
        for r in inner.facts.iter_mut().filter(|r| {
            r.fact.is_active()
                && r.fact.scope == scope
                && r.fact.category == category
                && r.fact.key == key
        }) {
            r.fact.superseded_at = Some(now);
            changed = true;
        }
        Ok(changed)
    }

    async fn active_facts(&self, scope: &str) -> Result<Vec<FactRecord>> {
        Ok(self
            .lock()?
            .facts
            .iter()
            .filter(|r| r.fact.is_active() && r.fact.scope == scope)
            .cloned()
            .collect())
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
        let (category, key) = (normalize_category(category), normalize_key(key));
        Ok(self
            .lock()?
            .facts
            .iter()
            .filter(|r| r.fact.scope == scope && r.fact.category == category && r.fact.key == key)
            .map(|r| r.fact.clone())
            .collect())
    }

    async fn record_fact_recalls(&self, ids: &[String], now: DateTime<Utc>) -> Result<()> {
        let ids: HashSet<&String> = ids.iter().collect();
        for r in self
            .lock()?
            .facts
            .iter_mut()
            .filter(|r| ids.contains(&r.fact.id))
        {
            r.fact.recall_count += 1;
            r.fact.last_recalled_at = Some(now);
        }
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
        let mut inner = self.lock()?;
        inner
            .episode_access
            .insert(episode.id.clone(), access.clone());
        inner.episodes.push(EpisodeRecord {
            episode: episode.clone(),
            embedding,
        });
        Ok(episode)
    }

    async fn episode_access(
        &self,
        scope: &str,
        ids: &[String],
    ) -> Result<HashMap<String, FactAccess>> {
        let inner = self.lock()?;
        let wanted: HashSet<&String> = ids.iter().collect();
        Ok(inner
            .episodes
            .iter()
            .filter(|r| r.episode.scope == scope && wanted.contains(&r.episode.id))
            .map(|r| {
                let access = inner.episode_access.get(&r.episode.id).cloned();
                (r.episode.id.clone(), access.unwrap_or_default())
            })
            .collect())
    }

    async fn episodes(&self, scope: &str) -> Result<Vec<EpisodeRecord>> {
        Ok(self
            .lock()?
            .episodes
            .iter()
            .filter(|r| r.episode.scope == scope)
            .cloned()
            .collect())
    }

    async fn record_episode_recalls(&self, ids: &[String], now: DateTime<Utc>) -> Result<()> {
        let ids: HashSet<&String> = ids.iter().collect();
        for r in self
            .lock()?
            .episodes
            .iter_mut()
            .filter(|r| ids.contains(&r.episode.id))
        {
            r.episode.recall_count += 1;
            r.episode.last_recalled_at = Some(now);
        }
        Ok(())
    }

    async fn decay(&self, now: DateTime<Utc>, config: &RetentionConfig) -> Result<DecayReport> {
        let mut report = DecayReport::default();
        let mut inner = self.lock()?;
        if let Some(window) = config.superseded_retention {
            let cutoff = now - window;
            let before = inner.facts.len();
            inner
                .facts
                .retain(|r| r.fact.superseded_at.is_none_or(|at| at >= cutoff));
            report.superseded_deleted = (before - inner.facts.len()) as u64;
            let kept: HashSet<String> = inner.facts.iter().map(|r| r.fact.id.clone()).collect();
            inner.fact_access.retain(|id, _| kept.contains(id));
        }
        if let Some(window) = config.episode_retention {
            let cutoff = now - window;
            let before = inner.episodes.len();
            inner
                .episodes
                .retain(|r| r.episode.recall_count > 0 || r.episode.created_at >= cutoff);
            report.episodes_deleted = (before - inner.episodes.len()) as u64;
            let kept: HashSet<String> = inner
                .episodes
                .iter()
                .map(|r| r.episode.id.clone())
                .collect();
            inner.episode_access.retain(|id, _| kept.contains(id));
        }
        if let Some(window) = config.fact_recall_decay_after {
            let cutoff = now - window;
            for r in inner.facts.iter_mut().filter(|r| {
                r.fact.is_active()
                    && r.fact.recall_count > 0
                    && r.fact.last_recalled_at.is_none_or(|at| at < cutoff)
            }) {
                r.fact.recall_count = decayed_fact_recalls(r.fact.recall_count);
                report.facts_decayed += 1;
            }
        }
        if let Some(window) = config.episode_recall_decay_after {
            let cutoff = now - window;
            for r in inner.episodes.iter_mut().filter(|r| {
                r.episode.recall_count > 0
                    && r.episode.last_recalled_at.is_none_or(|at| at < cutoff)
            }) {
                r.episode.recall_count -= 1;
                report.episodes_decayed += 1;
            }
        }
        Ok(report)
    }

    async fn stale_embeddings(
        &self,
        model: &str,
        kind: RecordKind,
        after_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<StaleEmbedding>> {
        let inner = self.lock()?;
        let describe = |id: &str, text: String, embedding: &Option<StoredEmbedding>| {
            let current = embedding.as_ref().map(|e| e.model.clone());
            let stale = current.as_deref() != Some(model) && after_id.is_none_or(|a| id > a);
            stale.then(|| StaleEmbedding {
                kind,
                id: id.to_string(),
                text,
                model: current,
            })
        };
        let mut rows: Vec<StaleEmbedding> = match kind {
            RecordKind::Fact => inner
                .facts
                .iter()
                .filter(|r| r.fact.is_active())
                .filter_map(|r| describe(&r.fact.id, r.fact.search_text(), &r.embedding))
                .collect(),
            RecordKind::Episode => inner
                .episodes
                .iter()
                .filter_map(|r| describe(&r.episode.id, r.episode.search_text(), &r.embedding))
                .collect(),
        };
        rows.sort_by(|a, b| a.id.cmp(&b.id));
        rows.truncate(limit);
        Ok(rows)
    }

    async fn replace_embedding(
        &self,
        stale: &StaleEmbedding,
        embedding: StoredEmbedding,
    ) -> Result<bool> {
        let mut inner = self.lock()?;
        let slot = match stale.kind {
            RecordKind::Fact => inner
                .facts
                .iter_mut()
                .find(|r| r.fact.id == stale.id && r.fact.is_active())
                .map(|r| &mut r.embedding),
            RecordKind::Episode => inner
                .episodes
                .iter_mut()
                .find(|r| r.episode.id == stale.id)
                .map(|r| &mut r.embedding),
        };
        let Some(slot) = slot else {
            return Ok(false);
        };
        if slot.as_ref().map(|e| e.model.as_str()) != stale.model.as_deref() {
            return Ok(false);
        }
        *slot = Some(embedding);
        Ok(true)
    }

    async fn stats(&self) -> Result<MemoryStats> {
        let inner = self.lock()?;
        let mut scopes: HashSet<&str> = inner.facts.iter().map(|r| r.fact.scope.as_str()).collect();
        scopes.extend(inner.episodes.iter().map(|r| r.episode.scope.as_str()));
        let active = inner.facts.iter().filter(|r| r.fact.is_active()).count() as u64;
        Ok(MemoryStats {
            active_facts: active,
            superseded_facts: inner.facts.len() as u64 - active,
            episodes: inner.episodes.len() as u64,
            scopes: scopes.len() as u64,
        })
    }
}
