//! Store-level and retrieval behavior, run against both bundled stores.

use std::sync::Arc;

use aidaemon_memory::{
    HashEmbedder, InMemoryMemoryStore, Memory, MemoryStore, NewEpisode, NewFact, Provenance,
    SqliteMemoryStore, UpsertOutcome,
};
use chrono::{Duration, Utc};

const SCOPE: &str = "telegram:synthetic-user-1";

async fn memories() -> Vec<(&'static str, Memory)> {
    let sqlite = SqliteMemoryStore::open_in_memory().await.unwrap();
    vec![
        (
            "in-memory",
            Memory::new(InMemoryMemoryStore::new(), HashEmbedder::new(512)),
        ),
        ("sqlite", Memory::new(sqlite, HashEmbedder::new(512))),
    ]
}

fn fact(category: &str, key: &str, value: &str) -> NewFact {
    NewFact::new(SCOPE, category, key, value)
}

#[tokio::test]
async fn supersession_hides_old_values_but_keeps_history() {
    for (name, memory) in memories().await {
        let first = memory
            .remember(fact("person", "daughters", "Mia"))
            .await
            .unwrap();
        assert!(matches!(first, UpsertOutcome::Inserted { .. }), "{name}");

        let same = memory
            .remember(fact("Person", "Daughters", "  mia "))
            .await
            .unwrap();
        assert_eq!(
            same,
            UpsertOutcome::Reinforced {
                id: first.id().to_string()
            },
            "{name}"
        );

        let corrected = memory
            .remember(fact("person", "daughters", "Mia and Zoe"))
            .await
            .unwrap();
        match &corrected {
            UpsertOutcome::Superseded {
                previous_id,
                previous_value,
                ..
            } => {
                assert_eq!(previous_id, first.id(), "{name}");
                assert_eq!(previous_value, "Mia", "{name}");
            }
            other => panic!("{name}: expected supersession, got {other:?}"),
        }

        let hits = memory
            .recall_facts(SCOPE, "daughters Mia", 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "{name}");
        assert_eq!(hits[0].fact.value, "Mia and Zoe", "{name}");

        let history = memory
            .fact_history(SCOPE, "person", "daughters")
            .await
            .unwrap();
        assert_eq!(history.len(), 2, "{name}");
        assert_eq!(history[0].value, "Mia");
        assert!(history[0].superseded_at.is_some());
        assert!(history[1].is_active());

        let stats = memory.stats().await.unwrap();
        assert_eq!(
            (stats.active_facts, stats.superseded_facts),
            (1, 1),
            "{name}"
        );

        assert!(memory.forget(SCOPE, "person", "daughters").await.unwrap());
        assert!(memory
            .recall_facts(SCOPE, "daughters Mia", 10)
            .await
            .unwrap()
            .is_empty());
        assert!(!memory.forget(SCOPE, "person", "daughters").await.unwrap());
        assert_eq!(
            memory
                .fact_history(SCOPE, "person", "daughters")
                .await
                .unwrap()
                .len(),
            2
        );
    }
}

#[tokio::test]
async fn relevant_fact_outranks_unrelated_and_noise_is_dropped() {
    for (name, memory) in memories().await {
        memory
            .remember(fact("preference", "favorite_color", "deep blue"))
            .await
            .unwrap();
        memory
            .remember(fact(
                "project",
                "acme_corp_deadline",
                "invoice audit due March",
            ))
            .await
            .unwrap();
        memory
            .remember(fact("person", "manager", "Jordan Lee"))
            .await
            .unwrap();

        let hits = memory
            .recall_facts(SCOPE, "what is my favorite color", 5)
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            1,
            "{name}: unrelated facts must be dropped: {hits:?}"
        );
        assert_eq!(hits[0].fact.key, "favorite_color");
        assert!(hits[0].signals.semantic > 0.3, "{name}");
        assert!(hits[0].signals.lexical > 0.3, "{name}");

        assert!(memory
            .recall_facts(SCOPE, "zebra migration patterns", 5)
            .await
            .unwrap()
            .is_empty());
        // Scopes are hard boundaries.
        assert!(memory
            .recall_facts("telegram:synthetic-user-2", "favorite color", 5)
            .await
            .unwrap()
            .is_empty());
    }
}

#[tokio::test]
async fn freshness_and_confidence_break_ties() {
    for (name, memory) in memories().await {
        let now = Utc::now();
        memory
            .remember_at(
                fact("project", "alpha_deadline", "Friday"),
                now - Duration::days(10),
            )
            .await
            .unwrap();
        memory
            .remember_at(fact("project", "beta_deadline", "Friday"), now)
            .await
            .unwrap();
        let hits = memory
            .recall_facts(SCOPE, "deadline Friday", 5)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2, "{name}");
        assert_eq!(hits[0].fact.key, "beta_deadline", "{name}: fresher first");
        assert!(hits[0].signals.freshness > 0.9);
        assert_eq!(hits[1].signals.freshness, 0.0);

        memory
            .remember_at(
                fact("travel", "gamma_seat", "window").confidence(0.2),
                now - Duration::days(30),
            )
            .await
            .unwrap();
        memory
            .remember_at(
                fact("travel", "delta_seat", "window").confidence(0.95),
                now - Duration::days(30),
            )
            .await
            .unwrap();
        let hits = memory.recall_facts(SCOPE, "seat window", 5).await.unwrap();
        assert_eq!(hits.len(), 2, "{name}");
        assert_eq!(hits[0].fact.key, "delta_seat", "{name}: confident first");
    }
}

#[tokio::test]
async fn provenance_excerpt_raises_the_score() {
    for (name, memory) in memories().await {
        memory
            .remember(fact("pet", "first_dog", "Rex").source(Provenance {
                excerpt: Some("my first dog was Rex".into()),
                ..Provenance::default()
            }))
            .await
            .unwrap();
        memory
            .remember(fact("pet", "second_dog", "Rex"))
            .await
            .unwrap();
        let hits = memory.recall_facts(SCOPE, "dog Rex", 5).await.unwrap();
        assert_eq!(hits[0].fact.key, "first_dog", "{name}");
        assert_eq!(hits[0].signals.provenance, 1.0);
    }
}

#[tokio::test]
async fn recall_counts_increment_and_decay() {
    for (name, memory) in memories().await {
        memory
            .remember(fact("user", "home_city", "Lisbon"))
            .await
            .unwrap();
        for expected in 1..=3 {
            let hits = memory.recall_facts(SCOPE, "home city", 5).await.unwrap();
            assert_eq!(hits[0].fact.recall_count, expected, "{name}");
        }
        let stored = memory.facts(SCOPE).await.unwrap();
        assert_eq!(stored[0].recall_count, 3, "{name}");
        assert!(stored[0].last_recalled_at.is_some());

        // Recalled just now: nothing decays yet.
        let report = memory.decay(Utc::now()).await.unwrap();
        assert_eq!(report.facts_decayed, 0, "{name}");

        // Two weeks without recall: lose max(1, count/10).
        let report = memory.decay(Utc::now() + Duration::days(15)).await.unwrap();
        assert_eq!(report.facts_decayed, 1, "{name}");
        assert_eq!(memory.facts(SCOPE).await.unwrap()[0].recall_count, 2);
    }
}

#[tokio::test]
async fn retention_deletes_old_superseded_rows_and_unrecalled_episodes() {
    for (name, memory) in memories().await {
        memory
            .remember(fact("user", "editor", "vim"))
            .await
            .unwrap();
        memory
            .remember(fact("user", "editor", "helix"))
            .await
            .unwrap();
        memory
            .add_episode(NewEpisode {
                scope: SCOPE.into(),
                session_id: "s1".into(),
                summary: "Planned a trip to Porto with Alice Rivera".into(),
                topics: vec!["travel".into()],
                importance: 0.5,
            })
            .await
            .unwrap();

        let report = memory.decay(Utc::now() + Duration::days(30)).await.unwrap();
        assert_eq!(report.superseded_deleted, 0, "{name}");

        let report = memory.decay(Utc::now() + Duration::days(91)).await.unwrap();
        assert_eq!(report.superseded_deleted, 1, "{name}");
        assert_eq!(report.episodes_deleted, 0, "{name}");
        assert_eq!(
            memory
                .fact_history(SCOPE, "user", "editor")
                .await
                .unwrap()
                .len(),
            1
        );

        let report = memory
            .decay(Utc::now() + Duration::days(400))
            .await
            .unwrap();
        assert_eq!(report.episodes_deleted, 1, "{name}");
        assert_eq!(memory.stats().await.unwrap().episodes, 0);
    }
}

#[tokio::test]
async fn episodes_are_recalled_by_relevance() {
    for (name, memory) in memories().await {
        for (summary, topics) in [
            (
                "Debugged the Acme Corp billing export with Jordan Lee",
                vec!["billing"],
            ),
            (
                "Planned a birthday party for Mia and Zoe",
                vec!["party", "family"],
            ),
        ] {
            memory
                .add_episode(NewEpisode {
                    scope: SCOPE.into(),
                    session_id: "s".into(),
                    summary: summary.into(),
                    topics: topics.into_iter().map(String::from).collect(),
                    importance: 0.6,
                })
                .await
                .unwrap();
        }
        let hits = memory
            .recall_episodes(SCOPE, "birthday party plans", 5)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "{name}");
        assert!(hits[0].episode.summary.contains("birthday"));
        assert_eq!(hits[0].episode.recall_count, 1);
    }
}

#[tokio::test]
async fn embedder_failure_falls_back_to_lexical() {
    struct Broken;
    #[async_trait::async_trait]
    impl aidaemon_memory::Embedder for Broken {
        fn model_id(&self) -> &str {
            "broken"
        }
        fn dimensions(&self) -> usize {
            4
        }
        async fn embed(&self, _: &[String]) -> aidaemon_memory::Result<Vec<Vec<f32>>> {
            Err(aidaemon_memory::MemoryError::Embedding("offline".into()))
        }
    }
    let memory = Memory::new(InMemoryMemoryStore::new(), Broken);
    memory
        .remember(fact("user", "home_city", "Lisbon"))
        .await
        .unwrap();
    let hits = memory.recall_facts(SCOPE, "home city", 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].signals.semantic, 0.0);
}

#[tokio::test]
async fn vectors_from_another_model_are_ignored() {
    let store = Arc::new(InMemoryMemoryStore::new());
    let old = Memory::builder(store.clone(), Arc::new(HashEmbedder::new(64))).build();
    old.remember(fact("user", "home_city", "Lisbon"))
        .await
        .unwrap();
    let new = Memory::builder(store, Arc::new(HashEmbedder::new(128))).build();
    let hits = new.recall_facts(SCOPE, "home city", 5).await.unwrap();
    assert_eq!(hits.len(), 1, "lexical still finds it");
    assert_eq!(hits[0].signals.semantic, 0.0);
}

// ---------------------------------------------------------------- SQLite

#[tokio::test]
async fn sqlite_migrations_are_idempotent_and_data_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.db");
    {
        let store = SqliteMemoryStore::open(&path).await.unwrap();
        assert_eq!(
            store.schema_version().await.unwrap(),
            aidaemon_memory::latest_version()
        );
        let memory = Memory::new(store.clone(), HashEmbedder::default());
        memory
            .remember(
                fact("person", "partner", "Alice Rivera").source(Provenance {
                    run_id: Some("run-1".into()),
                    session_id: Some(SCOPE.into()),
                    excerpt: Some("my partner Alice Rivera".into()),
                }),
            )
            .await
            .unwrap();
        memory
            .add_episode(NewEpisode {
                scope: SCOPE.into(),
                session_id: SCOPE.into(),
                summary: "Talked about weekend plans".into(),
                topics: vec!["weekend".into()],
                importance: 0.4,
            })
            .await
            .unwrap();
        store.pool().close().await;
    }
    // Open twice more: migrations are a no-op, the data is still there.
    let _again = SqliteMemoryStore::open(&path).await.unwrap();
    let store = SqliteMemoryStore::open(&path).await.unwrap();
    let applied: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_schema_migrations")
        .fetch_one(store.pool())
        .await
        .unwrap();
    assert_eq!(applied, aidaemon_memory::MIGRATIONS.len() as i64);

    let records = store.active_facts(SCOPE).await.unwrap();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.fact.value, "Alice Rivera");
    assert_eq!(record.fact.source.run_id.as_deref(), Some("run-1"));
    assert_eq!(record.embedding.as_ref().unwrap().vector.len(), 256);

    let memory = Memory::new(store, HashEmbedder::default());
    let hits = memory
        .recall_facts(SCOPE, "who is my partner", 5)
        .await
        .unwrap();
    assert_eq!(hits[0].fact.value, "Alice Rivera");
    let episodes = memory
        .recall_episodes(SCOPE, "weekend plans", 5)
        .await
        .unwrap();
    assert_eq!(episodes[0].episode.topics, vec!["weekend".to_string()]);
}

#[tokio::test]
async fn sqlite_fts_tracks_only_active_facts() {
    let store = SqliteMemoryStore::open_in_memory().await.unwrap();
    let memory = Memory::new(store.clone(), HashEmbedder::default());

    let first = memory
        .remember(fact("user", "employer", "Acme Corp"))
        .await
        .unwrap();
    assert_eq!(
        store.fts_fact_ids(SCOPE).await.unwrap(),
        vec![first.id().to_string()]
    );

    // Reinforcing keeps the same single row.
    memory
        .remember(fact("user", "employer", "acme corp"))
        .await
        .unwrap();
    assert_eq!(
        store.fts_fact_ids(SCOPE).await.unwrap(),
        vec![first.id().to_string()]
    );

    // Superseding swaps the indexed row in the same transaction.
    let second = memory
        .remember(fact("user", "employer", "Globex Industries"))
        .await
        .unwrap();
    assert_eq!(
        store.fts_fact_ids(SCOPE).await.unwrap(),
        vec![second.id().to_string()]
    );
    let lexical = store.fact_lexical_scores(SCOPE, "Acme").await.unwrap();
    assert!(
        lexical.is_empty(),
        "old value must not be searchable: {lexical:?}"
    );
    let lexical = store.fact_lexical_scores(SCOPE, "Globex").await.unwrap();
    assert!(lexical[second.id()] > 0.3);

    memory.forget(SCOPE, "user", "employer").await.unwrap();
    assert!(store.fts_fact_ids(SCOPE).await.unwrap().is_empty());
}

#[tokio::test]
async fn sqlite_lexical_scores_are_normalized_and_injection_safe() {
    let store = SqliteMemoryStore::open_in_memory().await.unwrap();
    let memory = Memory::new(store.clone(), HashEmbedder::default());
    memory
        .remember(fact("preference", "coffee_order", "oat flat white"))
        .await
        .unwrap();
    memory
        .remember(fact("preference", "tea", "green tea"))
        .await
        .unwrap();
    let scores = store
        .fact_lexical_scores(SCOPE, "coffee order \"NEAR( * OR")
        .await
        .unwrap();
    assert_eq!(scores.len(), 1);
    assert!(scores.values().all(|s| (0.0..=1.0).contains(s)));
}
