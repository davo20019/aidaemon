use std::collections::HashSet;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use super::MemoryStore;
use crate::error::{MemoryError, Result};
use crate::types::{
    new_id, normalize_category, normalize_key, same_value, DecayReport, Episode, EpisodeRecord,
    Fact, FactRecord, MemoryStats, NewEpisode, NewFact, RetentionConfig, StoredEmbedding,
    UpsertOutcome,
};

#[derive(Default)]
struct Inner {
    facts: Vec<FactRecord>,
    episodes: Vec<EpisodeRecord>,
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
        let mut inner = self.lock()?;
        let active = inner.facts.iter_mut().find(|r| {
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
                return Ok(UpsertOutcome::Reinforced {
                    id: record.fact.id.clone(),
                });
            }
            record.fact.superseded_at = Some(now);
            previous = Some((record.fact.id.clone(), record.fact.value.clone()));
        }
        let id = new_id();
        inner.facts.push(FactRecord {
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
        self.lock()?.episodes.push(EpisodeRecord {
            episode: episode.clone(),
            embedding,
        });
        Ok(episode)
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
        }
        if let Some(window) = config.episode_retention {
            let cutoff = now - window;
            let before = inner.episodes.len();
            inner
                .episodes
                .retain(|r| r.episode.recall_count > 0 || r.episode.created_at >= cutoff);
            report.episodes_deleted = (before - inner.episodes.len()) as u64;
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
