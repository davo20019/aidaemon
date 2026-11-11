//! `FactStore::search_facts` over SQLite, on the synthetic corpus.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

use super::*;
use crate::search::{FactSearchConfig, FactSearchReport};
use crate::search_fixture::{self as fx, corpus, queries};
use crate::HashEmbedder;

async fn open_at(path: &std::path::Path) -> SqliteMemory {
    // Test-only durability trade: the corpus is rebuilt on every run.
    let options = SqliteConnectOptions::new()
        .filename(path)
        .synchronous(sqlx::sqlite::SqliteSynchronous::Off);
    let pool = SqlitePoolOptions::new()
        .max_connections(2)
        .connect_with(options)
        .await
        .unwrap();
    migrate(&pool).await.unwrap();
    SqliteMemory::new(pool, Arc::new(HashEmbedder::new(EMBEDDING_DIM)))
}

async fn open() -> (SqliteMemory, tempfile::NamedTempFile) {
    let file = tempfile::NamedTempFile::new().unwrap();
    (open_at(file.path()).await, file)
}

/// The corpus written once through the normal write path (slow: every
/// write embeds and projects), with each fixture fact's recall count.
fn template() -> &'static std::path::Path {
    static TEMPLATE: std::sync::OnceLock<tempfile::TempPath> = std::sync::OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        let target = path.to_path_buf();
        std::thread::spawn(move || {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let memory = open_at(&target).await;
                for fact in corpus().into_iter().filter(|f| f.id > 0) {
                    memory
                        .upsert_fact(
                            &fact.category,
                            &fact.key,
                            &fact.value,
                            "synthetic",
                            None,
                            FactPrivacy::Global,
                        )
                        .await
                        .unwrap();
                    sqlx::query(
                        "UPDATE facts SET recall_count = ? WHERE value = ? AND superseded_at IS NULL",
                    )
                    .bind(fact.recall_count)
                    .bind(&fact.value)
                    .execute(memory.pool())
                    .await
                    .unwrap();
                }
                sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
                    .execute(memory.pool())
                    .await
                    .unwrap();
                memory.pool().close().await;
            })
        })
        .join()
        .unwrap();
        path
    })
}

/// A private copy of the seeded corpus, and fixture id → stored value.
async fn seeded() -> (SqliteMemory, tempfile::NamedTempFile, HashMap<i64, String>) {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::copy(template(), file.path()).unwrap();
    let values = corpus()
        .into_iter()
        .filter(|f| f.id > 0)
        .map(|f| (f.id, f.value))
        .collect();
    (open_at(file.path()).await, file, values)
}

fn values(report: &FactSearchReport) -> Vec<String> {
    report.hits.iter().map(|h| h.fact.value.clone()).collect()
}

#[tokio::test]
async fn search_facts_returns_port_among_flood() {
    let (memory, _file, _) = seeded().await;
    let report = memory
        .search_facts("staging deploy port", 10, &FactSearchConfig::default())
        .await
        .unwrap();
    assert_eq!(
        report.hits[0].fact.key,
        "staging_deploy_port",
        "{:?}",
        values(&report)
    );
    assert!(report.considered >= report.passed_floor);
    assert!(report.passed_floor >= report.hits.len());
}

#[tokio::test]
async fn search_facts_related_excludes_hot_cluster() {
    let (memory, _file, stored) = seeded().await;
    for q in queries()
        .into_iter()
        .filter(|q| !q.forbidden_related.is_empty())
    {
        let report = memory
            .search_facts(q.query, q.limit, &FactSearchConfig::default())
            .await
            .unwrap();
        let relevant = &stored[&q.relevant[0]];
        assert!(
            values(&report).contains(relevant),
            "{}: {:?}",
            q.query,
            values(&report)
        );
        for id in q.forbidden_related {
            let forbidden = &stored[id];
            assert!(
                !report.related.iter().any(|h| &h.fact.value == forbidden),
                "{}: hot-cluster fact {id} in related",
                q.query
            );
        }
    }
}

#[tokio::test]
async fn search_facts_family_cluster_travels_as_related() {
    let (memory, _file, _) = seeded().await;
    let report = memory
        .search_facts("mother", 1, &FactSearchConfig::default())
        .await
        .unwrap();
    assert_eq!(values(&report), vec!["Maria Rivera".to_string()]);
    let related: Vec<&str> = report
        .related
        .iter()
        .map(|h| h.fact.value.as_str())
        .collect();
    assert!(related.contains(&"Frank Rivera"), "{related:?}");
    assert!(report.related.len() <= FactSearchConfig::default().related_max);
}

struct FailingReranker;

#[async_trait]
impl Reranker for FailingReranker {
    async fn rerank(
        &self,
        _query: String,
        _documents: Vec<String>,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        anyhow::bail!("synthetic reranker outage")
    }

    fn relevance_floor(&self) -> Option<f32> {
        Some(0.5)
    }
}

#[tokio::test]
async fn search_facts_survives_reranker_error() {
    let (memory, _file, _) = seeded().await;
    let memory = memory.with_reranker(Arc::new(FailingReranker));
    let report = memory
        .search_facts("reply style", 5, &FactSearchConfig::default())
        .await
        .unwrap();
    assert!(!report.hits.is_empty());
    assert!(!report.reranked);
    assert_eq!(report.hits[0].fact.key, "reply_style");
}

/// Scores documents by how many query words they contain, with a floor of 1.
struct OverlapReranker;

#[async_trait]
impl Reranker for OverlapReranker {
    async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let words = crate::search::tokenize(&query);
        let mut scored: Vec<(usize, f32)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let doc = crate::search::tokenize(doc);
                (i, words.iter().filter(|w| doc.contains(w)).count() as f32)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored)
    }

    fn relevance_floor(&self) -> Option<f32> {
        Some(1.0)
    }
}

#[tokio::test]
async fn search_facts_applies_the_reranker_floor() {
    let (memory, _file, _) = seeded().await;
    let memory = memory.with_reranker(Arc::new(OverlapReranker));
    let report = memory
        .search_facts("code review preference", 10, &FactSearchConfig::default())
        .await
        .unwrap();
    assert!(report.reranked);
    assert_eq!(report.hits[0].fact.key, "code_review_preference");
    assert_eq!(
        report.passed_floor,
        report.hits.len().max(report.passed_floor)
    );
    assert!(report.hits.iter().all(|h| h.score >= 1.0));
}

#[tokio::test]
async fn search_facts_collapses_canonical_duplicate() {
    let (memory, _file) = open().await;
    memory
        .upsert_fact(
            "user",
            "birth_date",
            "1990-04-12",
            "synthetic",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    let write = PersonalMemoryWrite {
        entities: vec![PersonalEntityCandidate {
            local_id: "owner".into(),
            entity_type: "person".into(),
            canonical_name: "Alice Rivera".into(),
            is_reference: false,
            canonical_name_confirmed: true,
        }],
        facts: vec![PersonalFactCandidate {
            subject_local_id: "owner".into(),
            predicate: "birth_date".into(),
            value: "1990-04-12".into(),
            display_value: None,
            valid_from: None,
            valid_to: None,
        }],
        direct_user_statement: true,
        ..Default::default()
    };
    memory
        .reconcile_personal_memory(&write, "synthetic", None, None, FactPrivacy::Global)
        .await
        .unwrap();
    assert!(
        memory
            .get_canonical_memory_facts()
            .await
            .unwrap()
            .iter()
            .any(|f| f.key == "birth_date"),
        "the write must produce a canonical projection for this test to mean anything"
    );
    let report = memory
        .search_facts("birth date", 10, &FactSearchConfig::default())
        .await
        .unwrap();
    let dates = report
        .hits
        .iter()
        .chain(&report.related)
        .filter(|h| h.fact.value == "1990-04-12")
        .count();
    assert_eq!(dates, 1, "{:?}", values(&report));
}

struct BareStore;

#[async_trait]
impl FactStore for BareStore {
    async fn upsert_fact_with_provenance(
        &self,
        _category: &str,
        _key: &str,
        _value: &str,
        _source: &str,
        _channel_id: Option<&str>,
        _privacy: FactPrivacy,
        _first_seen_at: Option<DateTime<Utc>>,
        _source_excerpt: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_facts(&self, _category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
        Ok(vec![fx::fact(1, "user", "birth_date", "1990-04-12", 0)])
    }
}

#[tokio::test]
async fn default_impl_returns_empty_report() {
    let report = BareStore
        .search_facts("birth date", 10, &FactSearchConfig::default())
        .await
        .unwrap();
    assert!(report.hits.is_empty() && report.related.is_empty());
    assert_eq!(report.considered, 0);
}
