//! "Bring your own memory": a user-defined store and embedder, written
//! here outside the crate, swap in through the public traits.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use aidaemon_core::testing::{call, text, tool_calls, ScriptedProvider};
use aidaemon_core::Agent;
use aidaemon_memory::{
    same_value, Embedder, Episode, EpisodeRecord, Fact, FactRecord, Memory, MemoryRecall,
    MemoryStats, MemoryStore, NewEpisode, NewFact, RecallTool, Result, Score, ScoredFact, Scorer,
    Signals, StoredEmbedding, UpsertOutcome,
};
use aidaemon_policies::Recall;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::json;

const SCOPE: &str = "telegram:synthetic-user-1";

/// A tiny Vec-backed store: just rows, every default method left alone.
#[derive(Default)]
struct VecStore {
    facts: Mutex<Vec<FactRecord>>,
    episodes: Mutex<Vec<EpisodeRecord>>,
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

    async fn record_episode_recalls(&self, _ids: &[String], _now: DateTime<Utc>) -> Result<()> {
        Ok(())
    }

    async fn stats(&self) -> Result<MemoryStats> {
        let facts = self.facts.lock().unwrap();
        let active = facts
            .iter()
            .filter(|r| r.fact.superseded_at.is_none())
            .count() as u64;
        Ok(MemoryStats {
            active_facts: active,
            superseded_facts: facts.len() as u64 - active,
            episodes: self.episodes.lock().unwrap().len() as u64,
            scopes: 1,
        })
    }
}

/// A user-defined embedder: a 3-dim "topic" space, counting its calls.
#[derive(Default)]
struct TopicEmbedder {
    calls: Mutex<usize>,
}

#[async_trait]
impl Embedder for TopicEmbedder {
    fn model_id(&self) -> &str {
        "topics-v1"
    }
    fn dimensions(&self) -> usize {
        3
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        *self.calls.lock().unwrap() += texts.len();
        Ok(texts
            .iter()
            .map(|t| {
                let t = t.to_lowercase();
                let has = |w: &str| if t.contains(w) { 1.0 } else { 0.0 };
                vec![has("family"), has("work"), has("food")]
            })
            .collect())
    }
}

#[tokio::test]
async fn custom_store_and_embedder_swap_in() {
    let store = Arc::new(VecStore::default());
    let embedder = Arc::new(TopicEmbedder::default());
    let memory = Memory::builder(store.clone(), embedder.clone()).build();

    memory
        .remember(NewFact::new(SCOPE, "family", "daughters", "Mia and Zoe"))
        .await
        .unwrap();
    memory
        .remember(NewFact::new(SCOPE, "work", "employer", "Acme Corp"))
        .await
        .unwrap();
    let outcome = memory
        .remember(NewFact::new(SCOPE, "work", "employer", "Globex"))
        .await
        .unwrap();
    assert!(matches!(outcome, UpsertOutcome::Superseded { .. }));

    // Purely semantic match through the custom embedder ("kids" shares no
    // word with the fact); lexical comes from the trait's default.
    let hits = memory
        .recall_facts(SCOPE, "my family, the kids", 5)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].fact.value, "Mia and Zoe");
    assert!(hits[0].signals.semantic > 0.9);

    let hits = memory.recall_facts(SCOPE, "work stuff", 5).await.unwrap();
    assert_eq!(hits[0].fact.value, "Globex");
    assert_eq!(store.facts.lock().unwrap()[0].fact.recall_count, 1);
    assert!(*embedder.calls.lock().unwrap() >= 5);

    // The rest of the SDK sees only the policies trait and core tools.
    let snippets = MemoryRecall::new(memory.clone())
        .recall(SCOPE, "family", 5)
        .await;
    assert_eq!(snippets, vec!["family daughters: Mia and Zoe".to_string()]);

    let agent = Agent::builder("assistant")
        .model("m")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "recall", json!({"query": "work"}))]),
            text("You work at Globex."),
        ]))
        .tool(RecallTool::new(memory.clone()))
        .build()
        .unwrap();
    let outcome = agent.run(SCOPE, "Where do I work?").await.unwrap();
    assert_eq!(outcome.receipts()[0].details["facts"][0]["value"], "Globex");

    // Default decay is a no-op for stores that expire on their own.
    assert_eq!(memory.decay(Utc::now()).await.unwrap(), Default::default());
}

/// A store with native ranking bypasses the scorer.
struct RankedStore(VecStore);

#[async_trait]
impl MemoryStore for RankedStore {
    async fn upsert_fact(
        &self,
        f: NewFact,
        e: Option<StoredEmbedding>,
        n: DateTime<Utc>,
    ) -> Result<UpsertOutcome> {
        self.0.upsert_fact(f, e, n).await
    }
    async fn supersede_fact(&self, s: &str, c: &str, k: &str, n: DateTime<Utc>) -> Result<bool> {
        self.0.supersede_fact(s, c, k, n).await
    }
    async fn active_facts(&self, s: &str) -> Result<Vec<FactRecord>> {
        self.0.active_facts(s).await
    }
    async fn fact_history(&self, s: &str, c: &str, k: &str) -> Result<Vec<Fact>> {
        self.0.fact_history(s, c, k).await
    }
    async fn search_facts(
        &self,
        scope: &str,
        _query: &str,
        query_embedding: Option<&StoredEmbedding>,
        limit: usize,
    ) -> Result<Option<Vec<ScoredFact>>> {
        assert!(
            query_embedding.is_some(),
            "query embedding is passed through"
        );
        // "Native" ranking: alphabetical by value, everything matches.
        let mut facts: Vec<Fact> = self
            .0
            .active_facts(scope)
            .await?
            .into_iter()
            .map(|r| r.fact)
            .collect();
        facts.sort_by(|a, b| a.value.cmp(&b.value));
        Ok(Some(
            facts
                .into_iter()
                .take(limit)
                .map(|fact| ScoredFact {
                    fact,
                    score: 1.0,
                    signals: Signals::default(),
                })
                .collect(),
        ))
    }
    async fn record_fact_recalls(&self, ids: &[String], n: DateTime<Utc>) -> Result<()> {
        self.0.record_fact_recalls(ids, n).await
    }
    async fn insert_episode(
        &self,
        ep: NewEpisode,
        e: Option<StoredEmbedding>,
        n: DateTime<Utc>,
    ) -> Result<Episode> {
        self.0.insert_episode(ep, e, n).await
    }
    async fn episodes(&self, s: &str) -> Result<Vec<EpisodeRecord>> {
        self.0.episodes(s).await
    }
    async fn record_episode_recalls(&self, ids: &[String], n: DateTime<Utc>) -> Result<()> {
        self.0.record_episode_recalls(ids, n).await
    }
    async fn stats(&self) -> Result<MemoryStats> {
        self.0.stats().await
    }
}

/// A scorer that would drop everything; proves the native path skips it.
struct RejectAll(Mutex<HashMap<&'static str, usize>>);

impl Scorer for RejectAll {
    fn score_fact(&self, _: &Fact, _: Option<f32>, _: f32, _: DateTime<Utc>) -> Option<Score> {
        *self.0.lock().unwrap().entry("fact").or_default() += 1;
        None
    }
    fn score_episode(
        &self,
        _: &Episode,
        _: Option<f32>,
        _: f32,
        _: DateTime<Utc>,
    ) -> Option<Score> {
        None
    }
}

#[tokio::test]
async fn native_search_bypasses_the_scorer_and_custom_scorers_apply_otherwise() {
    let scorer = Arc::new(RejectAll(Mutex::new(HashMap::new())));
    let ranked = Memory::builder(
        Arc::new(RankedStore(VecStore::default())),
        Arc::new(TopicEmbedder::default()),
    )
    .scorer(scorer.clone())
    .build();
    for value in ["zucchini", "apples"] {
        ranked
            .remember(NewFact::new(SCOPE, "food", value, value))
            .await
            .unwrap();
    }
    let hits = ranked.recall_facts(SCOPE, "anything", 5).await.unwrap();
    let values: Vec<&str> = hits.iter().map(|h| h.fact.value.as_str()).collect();
    assert_eq!(values, ["apples", "zucchini"]);
    assert_eq!(hits[0].fact.recall_count, 1);
    assert!(scorer.0.lock().unwrap().is_empty());

    // Same scorer on a plain store: it is consulted and drops everything.
    let plain = Memory::builder(
        Arc::new(VecStore::default()),
        Arc::new(TopicEmbedder::default()),
    )
    .scorer(scorer.clone())
    .build();
    plain
        .remember(NewFact::new(SCOPE, "food", "fruit", "apples"))
        .await
        .unwrap();
    assert!(plain
        .recall_facts(SCOPE, "food apples", 5)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(scorer.0.lock().unwrap()["fact"], 1);
}
