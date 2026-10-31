//! Shared test stores: a minimal user-defined `MemoryStore`, written outside
//! the crate against the public trait only.
#![allow(dead_code)]

use std::sync::Mutex;

use aidaemon_memory::{
    same_value, Episode, EpisodeRecord, Fact, FactRecord, MemoryStats, MemoryStore, NewEpisode,
    NewFact, Result, StoredEmbedding, UpsertOutcome,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};

/// A tiny Vec-backed store: just rows, every default method left alone
/// (lexical scoring, native search and decay use the trait defaults).
///
/// The store conformance kit (`tests/conformance_tests.rs`) originally
/// caught three contract gaps here: reinforcement did not keep the higher
/// confidence, `record_episode_recalls` was a no-op, and `stats` always
/// reported one scope.
#[derive(Default)]
pub struct VecStore {
    pub facts: Mutex<Vec<FactRecord>>,
    pub episodes: Mutex<Vec<EpisodeRecord>>,
}

#[async_trait]
impl MemoryStore for VecStore {
    async fn upsert_fact(
        &self,
        fact: NewFact,
        embedding: Option<StoredEmbedding>,
        now: DateTime<Utc>,
    ) -> Result<UpsertOutcome> {
        let mut facts = self.facts.lock().unwrap();
        let mut previous = None;
        if let Some(active) = facts.iter_mut().find(|r| {
            r.fact.superseded_at.is_none()
                && (
                    r.fact.scope.as_str(),
                    r.fact.category.as_str(),
                    r.fact.key.as_str(),
                ) == (
                    fact.scope.as_str(),
                    fact.category.as_str(),
                    fact.key.as_str(),
                )
        }) {
            if same_value(&active.fact.value, &fact.value) {
                active.fact.updated_at = now;
                // The contract keeps the higher confidence on reinforcement.
                active.fact.confidence = active.fact.confidence.max(fact.confidence);
                return Ok(UpsertOutcome::Reinforced {
                    id: active.fact.id.clone(),
                });
            }
            active.fact.superseded_at = Some(now);
            previous = Some((active.fact.id.clone(), active.fact.value.clone()));
        }
        let id = format!("vec-{}", facts.len());
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

    async fn supersede_fact(
        &self,
        scope: &str,
        category: &str,
        key: &str,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        let mut hit = false;
        for r in self.facts.lock().unwrap().iter_mut() {
            if r.fact.superseded_at.is_none()
                && r.fact.scope == scope
                && r.fact.category == category
                && r.fact.key == key
            {
                r.fact.superseded_at = Some(now);
                hit = true;
            }
        }
        Ok(hit)
    }

    async fn active_facts(&self, scope: &str) -> Result<Vec<FactRecord>> {
        Ok(self
            .facts
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.fact.scope == scope && r.fact.superseded_at.is_none())
            .cloned()
            .collect())
    }

    async fn fact_history(&self, scope: &str, category: &str, key: &str) -> Result<Vec<Fact>> {
        Ok(self
            .facts
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.fact.scope == scope && r.fact.category == category && r.fact.key == key)
            .map(|r| r.fact.clone())
            .collect())
    }

    async fn record_fact_recalls(&self, ids: &[String], now: DateTime<Utc>) -> Result<()> {
        for r in self.facts.lock().unwrap().iter_mut() {
            if ids.contains(&r.fact.id) {
                r.fact.recall_count += 1;
                r.fact.last_recalled_at = Some(now);
            }
        }
        Ok(())
    }

    async fn insert_episode(
        &self,
        episode: NewEpisode,
        embedding: Option<StoredEmbedding>,
        now: DateTime<Utc>,
    ) -> Result<Episode> {
        let mut episodes = self.episodes.lock().unwrap();
        let episode = Episode {
            id: format!("ep-{}", episodes.len()),
            scope: episode.scope,
            session_id: episode.session_id,
            summary: episode.summary,
            topics: episode.topics,
            importance: episode.importance,
            created_at: now,
            recall_count: 0,
            last_recalled_at: None,
        };
        episodes.push(EpisodeRecord {
            episode: episode.clone(),
            embedding,
        });
        Ok(episode)
    }

    async fn episodes(&self, scope: &str) -> Result<Vec<EpisodeRecord>> {
        Ok(self
            .episodes
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.episode.scope == scope)
            .cloned()
            .collect())
    }

    async fn record_episode_recalls(&self, ids: &[String], now: DateTime<Utc>) -> Result<()> {
        for r in self.episodes.lock().unwrap().iter_mut() {
            if ids.contains(&r.episode.id) {
                r.episode.recall_count += 1;
                r.episode.last_recalled_at = Some(now);
            }
        }
        Ok(())
    }

    async fn stats(&self) -> Result<MemoryStats> {
        let facts = self.facts.lock().unwrap();
        let active = facts
            .iter()
            .filter(|r| r.fact.superseded_at.is_none())
            .count() as u64;
        let episodes = self.episodes.lock().unwrap();
        let mut scopes: Vec<&str> = facts
            .iter()
            .map(|r| r.fact.scope.as_str())
            .chain(episodes.iter().map(|r| r.episode.scope.as_str()))
            .collect();
        scopes.sort_unstable();
        scopes.dedup();
        Ok(MemoryStats {
            active_facts: active,
            superseded_facts: facts.len() as u64 - active,
            episodes: episodes.len() as u64,
            scopes: scopes.len() as u64,
        })
    }
}
