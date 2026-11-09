//! Store-level and retrieval behavior, run against both bundled stores.

use std::sync::Arc;

use aidaemon_core::ChannelVisibility;
use aidaemon_memory::{
    FactAccess, HashEmbedder, InMemoryMemoryStore, Memory, MemoryStore, NewEpisode, NewFact,
    Provenance, RecallAudience, ScoredFact, SqliteMemoryStore, UpsertOutcome,
};
use chrono::{Duration, Utc};

mod common;

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

// ---------------------------------------------------------------- Access

fn group(channel: &str) -> RecallAudience {
    RecallAudience::new(ChannelVisibility::PrivateGroup, Some(channel.into()), true)
}

fn public(channel: &str) -> RecallAudience {
    RecallAudience::new(ChannelVisibility::Public, Some(channel.into()), true)
}

fn episode(summary: &str) -> NewEpisode {
    NewEpisode {
        scope: SCOPE.into(),
        session_id: "telegram:synthetic-session-1".into(),
        summary: summary.into(),
        topics: vec![],
        importance: 0.5,
    }
}

#[tokio::test]
async fn private_records_never_reach_shared_channels() {
    for (name, memory) in memories().await {
        memory
            .remember_with(fact("user", "home_city", "Lisbon"), &FactAccess::private())
            .await
            .unwrap();
        memory
            .add_episode_with(
                episode("Talked about moving to Lisbon"),
                &FactAccess::private(),
            )
            .await
            .unwrap();

        let audiences = [
            group("c1"),
            public("c1"),
            RecallAudience::new(ChannelVisibility::PublicExternal, Some("c1".into()), true),
            RecallAudience::new(ChannelVisibility::Private, None, false),
        ];
        for audience in &audiences {
            let facts = memory
                .recall_facts_for(SCOPE, "home city Lisbon", 5, audience)
                .await
                .unwrap();
            assert!(facts.is_empty(), "{name}: {audience:?} saw {facts:?}");
            let episodes = memory
                .recall_episodes_for(SCOPE, "moving to Lisbon", 5, audience)
                .await
                .unwrap();
            assert!(episodes.is_empty(), "{name}: {audience:?} saw {episodes:?}");
        }
        // The empty-query listing goes through the same filter.
        let listed = memory
            .recall_facts_for(SCOPE, "", 5, &group("c1"))
            .await
            .unwrap();
        assert!(listed.is_empty(), "{name}: {listed:?}");

        // Hidden recalls left no trace, and the owner still sees everything.
        let owner = memory
            .recall_facts(SCOPE, "home city Lisbon", 5)
            .await
            .unwrap();
        assert_eq!(owner.len(), 1, "{name}");
        assert_eq!(owner[0].fact.recall_count, 1, "{name}");
        assert_eq!(
            memory
                .recall_episodes(SCOPE, "moving to Lisbon", 5)
                .await
                .unwrap()
                .len(),
            1,
            "{name}"
        );
    }
}

#[tokio::test]
async fn channel_records_stay_in_their_channel_and_global_ones_are_shared() {
    for (name, memory) in memories().await {
        memory
            .remember_with(
                fact("team", "standup_time", "ten thirty"),
                &FactAccess::channel("c1"),
            )
            .await
            .unwrap();
        memory
            .remember_with(fact("team", "office_city", "Porto"), &FactAccess::global())
            .await
            .unwrap();

        let keys = |hits: Vec<ScoredFact>| -> Vec<String> {
            let mut keys: Vec<String> = hits.into_iter().map(|h| h.fact.key).collect();
            keys.sort();
            keys
        };
        let query = "team standup time office city";
        let same = memory
            .recall_facts_for(SCOPE, query, 5, &group("c1"))
            .await
            .unwrap();
        assert_eq!(keys(same), ["office_city", "standup_time"], "{name}");
        let other = memory
            .recall_facts_for(SCOPE, query, 5, &group("c2"))
            .await
            .unwrap();
        assert_eq!(keys(other), ["office_city"], "{name}");
        // Public channels get no general memory (the daemon's rule), only
        // what was said in that channel.
        let in_public = memory
            .recall_facts_for(SCOPE, query, 5, &public("c1"))
            .await
            .unwrap();
        assert_eq!(keys(in_public), ["standup_time"], "{name}");
    }
}

#[tokio::test]
async fn hidden_facts_do_not_use_up_the_limit() {
    for (name, memory) in memories().await {
        for i in 0..4 {
            memory
                .remember_with(
                    fact("note", &format!("private_{i}"), "mango harvest"),
                    &FactAccess::private(),
                )
                .await
                .unwrap();
        }
        memory
            .remember_with(
                fact("note", "shared", "mango recipe"),
                &FactAccess::global(),
            )
            .await
            .unwrap();
        let hits = memory
            .recall_facts_for(SCOPE, "mango", 1, &group("c1"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "{name}");
        assert_eq!(hits[0].fact.key, "shared", "{name}");
    }
}

#[tokio::test]
async fn memory_written_without_a_level_is_private() {
    for (name, memory) in memories().await {
        memory
            .remember(fact("user", "home_city", "Lisbon"))
            .await
            .unwrap();
        assert!(
            memory
                .recall_facts_for(SCOPE, "home city", 5, &group("c1"))
                .await
                .unwrap()
                .is_empty(),
            "{name}"
        );
    }
}

#[tokio::test]
async fn restating_in_a_narrower_place_does_not_leave_it_wider() {
    for (name, memory) in memories().await {
        memory
            .remember_with(fact("user", "home_city", "Lisbon"), &FactAccess::global())
            .await
            .unwrap();
        memory
            .remember_with(fact("user", "home_city", "lisbon"), &FactAccess::private())
            .await
            .unwrap();
        assert!(
            memory
                .recall_facts_for(SCOPE, "home city", 5, &group("c1"))
                .await
                .unwrap()
                .is_empty(),
            "{name}: a private restatement narrows the fact"
        );
    }
}

/// A store that keeps no access levels (the trait default) shows its records
/// to the owner in a direct conversation and hides them everywhere else.
#[tokio::test]
async fn a_store_without_access_levels_fails_closed() {
    let memory = Memory::new(common::VecStore::default(), HashEmbedder::new(512));
    memory
        .remember_with(fact("user", "home_city", "Lisbon"), &FactAccess::global())
        .await
        .unwrap();
    memory
        .add_episode(episode("Moved to Lisbon"))
        .await
        .unwrap();
    assert_eq!(
        memory
            .recall_facts(SCOPE, "home city", 5)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(memory
        .recall_facts_for(SCOPE, "home city", 5, &group("c1"))
        .await
        .unwrap()
        .is_empty());
    assert!(memory
        .recall_episodes_for(SCOPE, "Moved to Lisbon", 5, &group("c1"))
        .await
        .unwrap()
        .is_empty());
}

/// The tools and injected recall use the session's audience, and a restricted
/// writer is not told what it replaced.
#[tokio::test]
async fn tools_and_injection_follow_the_session_audience() {
    use aidaemon_core::{Tool, ToolContext};
    use aidaemon_memory::{MemoryRecall, RecallTool, RememberTool};
    use aidaemon_policies::Recall;

    let memory = Memory::builder(
        Arc::new(InMemoryMemoryStore::new()),
        Arc::new(HashEmbedder::new(512)),
    )
    .scope_resolver(|_| SCOPE.to_string())
    .audience_resolver(|session| {
        if session == "owner-dm" {
            RecallAudience::owner()
        } else {
            RecallAudience::new(ChannelVisibility::PrivateGroup, Some("c1".into()), true)
        }
    })
    .build();
    memory
        .remember(fact("user", "home_city", "Lisbon"))
        .await
        .unwrap();

    let in_group = ToolContext::new("group-session", "call-1");
    let recalled = RecallTool::new(memory.clone())
        .call(serde_json::json!({"query": "home city"}), &in_group)
        .await
        .unwrap();
    assert_eq!(recalled.content, "No matching memories.");
    let snippets = MemoryRecall::new(memory.clone())
        .recall("group-session", "home city", 5)
        .await;
    assert!(snippets.is_empty(), "{snippets:?}");
    assert_eq!(
        MemoryRecall::new(memory.clone())
            .recall("owner-dm", "home city", 5)
            .await
            .len(),
        1
    );

    // Written in the group: the fact stays in that channel, and the reply
    // does not repeat the private value it replaced.
    let saved = RememberTool::new(memory.clone())
        .call(
            serde_json::json!({"category": "user", "key": "home_city", "value": "Porto"}),
            &in_group,
        )
        .await
        .unwrap();
    assert!(!saved.content.contains("Lisbon"), "{}", saved.content);
    assert!(!format!("{:?}", saved.details).contains("Lisbon"));
    let hits = memory
        .recall_facts_for(SCOPE, "home city", 5, &group("c1"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].fact.value, "Porto");
    assert!(memory
        .recall_facts_for(SCOPE, "home city", 5, &group("c2"))
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn sqlite_v1_database_upgrades_and_keeps_its_rows() {
    use aidaemon_memory::{encode_embedding, latest_version, MIGRATIONS};
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1.db");
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(&path)
                .create_if_missing(true),
        )
        .await
        .unwrap();
    // Exactly what a version-1 build left behind.
    assert_eq!(MIGRATIONS[0].version, 1);
    sqlx::query(
        "CREATE TABLE memory_schema_migrations (
            version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::raw_sql(MIGRATIONS[0].sql)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO memory_schema_migrations VALUES (1, '2026-01-01T00:00:00.000Z')")
        .execute(&pool)
        .await
        .unwrap();
    let old = "2026-01-01T00:00:00.000000Z";
    let insert = "INSERT INTO memory_facts (id, scope, category, key, value, confidence,
        source_excerpt, created_at, updated_at, superseded_at, recall_count, embedding,
        embedding_model) VALUES (?, ?, 'user', 'home_city', ?, 0.9, 'said so', ?, ?, ?, 3, ?, ?)";
    let vector = encode_embedding(&[0.5, -0.25]);
    sqlx::query(insert)
        .bind("old-fact")
        .bind(SCOPE)
        .bind("Porto")
        .bind(old)
        .bind(old)
        .bind(old)
        .bind(&vector)
        .bind("old-model")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(insert)
        .bind("active-fact")
        .bind(SCOPE)
        .bind("Lisbon")
        .bind(old)
        .bind(old)
        .bind(None::<String>)
        .bind(&vector)
        .bind("old-model")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("INSERT INTO memory_facts_fts (fact_id, scope, content) VALUES (?, ?, ?)")
        .bind("active-fact")
        .bind(SCOPE)
        .bind("user home city: Lisbon")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO memory_episodes (id, scope, session_id, summary, topics, importance,
            created_at) VALUES ('old-episode', ?, 's1', 'Moved to Lisbon', '[\"move\"]', 0.7, ?)",
    )
    .bind(SCOPE)
    .bind(old)
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO memory_episodes_fts (episode_id, scope, content)
         VALUES ('old-episode', ?, 'Moved to Lisbon Topics: move')",
    )
    .bind(SCOPE)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;

    let store = SqliteMemoryStore::open(&path).await.unwrap();
    assert_eq!(store.schema_version().await.unwrap(), latest_version());
    assert!(latest_version() >= 2);

    // Every row and field survived.
    let active = store.active_facts(SCOPE).await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].fact.id, "active-fact");
    assert_eq!(active[0].fact.value, "Lisbon");
    assert_eq!(active[0].fact.recall_count, 3);
    assert_eq!(active[0].embedding.as_ref().unwrap().model, "old-model");
    assert_eq!(active[0].embedding.as_ref().unwrap().vector, [0.5, -0.25]);
    let history = store
        .fact_history(SCOPE, "user", "home_city")
        .await
        .unwrap();
    assert_eq!(history.len(), 2);
    assert!(history.iter().any(|f| f.superseded_at.is_some()));
    assert_eq!(store.episodes(SCOPE).await.unwrap().len(), 1);

    // Old rows were never classified, so they read as private.
    let ids = ["active-fact".to_string()];
    assert_eq!(
        store
            .fact_access(SCOPE, &ids)
            .await
            .unwrap()
            .get("active-fact"),
        Some(&FactAccess::private())
    );

    let memory = Memory::new(store, HashEmbedder::new(512));
    let owner = memory.recall_facts(SCOPE, "home city", 5).await.unwrap();
    assert_eq!(owner.len(), 1, "the owner keeps seeing old rows");
    assert_eq!(owner[0].signals.semantic, 0.0, "vectors of another model");
    assert!(memory
        .recall_facts_for(SCOPE, "home city", 5, &group("c1"))
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        memory
            .recall_episodes(SCOPE, "Moved Lisbon", 5)
            .await
            .unwrap()
            .len(),
        1
    );
    // The upgraded store takes new classified writes.
    memory
        .remember_with(fact("team", "office_city", "Porto"), &FactAccess::global())
        .await
        .unwrap();
    assert_eq!(
        memory
            .recall_facts_for(SCOPE, "office city", 5, &group("c1"))
            .await
            .unwrap()
            .len(),
        1
    );
}
