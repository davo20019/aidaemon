use super::*;
use crate::memory::embeddings::EmbeddingService;
use crate::traits::FactStore;
use crate::traits::{ExtractedMemoryEntity, ExtractedMemoryGraph, ExtractedMemoryRelationship};
use aidaemon_memory::embedding_index::*;
use aidaemon_memory::encode_embedding;
use aidaemon_memory::stores::MemoryGraphStore;
use std::sync::Arc;

async fn test_store() -> (tempfile::TempDir, SqliteStateStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.db");
    let service = Arc::new(EmbeddingService::new().unwrap());
    let store = SqliteStateStore::new(path.to_str().unwrap(), 20, None, service)
        .await
        .unwrap();
    (dir, store)
}

async fn insert_fact(store: &SqliteStateStore, privacy: &str) -> i64 {
    let vector = encode_embedding(&vec![0.25; 384]);
    sqlx::query(
        "INSERT INTO facts
            (category, key, value, source, created_at, updated_at, channel_id,
             privacy, embedding, recall_count)
         VALUES ('preference', 'default_editor', 'Neovim', 'owner-stated',
                 '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                 'telegram:synthetic-user-1', ?, ?, 0)",
    )
    .bind(privacy)
    .bind(vector)
    .execute(&store.pool)
    .await
    .unwrap()
    .last_insert_rowid()
}

#[tokio::test]
async fn projection_versions_embeddings_and_keeps_one_active() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "global").await;
    store.project_fact_memory(fact_id).await.unwrap();

    sqlx::query(
        "UPDATE facts SET value = 'Helix', updated_at = '2026-01-02T00:00:00Z' WHERE id = ?",
    )
    .bind(fact_id)
    .execute(&store.pool)
    .await
    .unwrap();
    store.project_fact_memory(fact_id).await.unwrap();

    let claim_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_claims WHERE source_fact_id = ?")
            .bind(fact_id)
            .fetch_one(&store.pool)
            .await
            .unwrap();
    let active_embeddings: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_embeddings WHERE owner_type = 'claim' AND stale_at IS NULL",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let stale_embeddings: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_embeddings WHERE owner_type = 'claim' AND stale_at IS NOT NULL",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(claim_count, 1);
    assert_eq!(active_embeddings, 1);
    assert_eq!(stale_embeddings, 1);
}

#[tokio::test]
async fn fact_keys_do_not_guess_semantic_entities() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "global").await;
    sqlx::query("UPDATE facts SET key = 'database_framework', value = 'SQLite' WHERE id = ?")
        .bind(fact_id)
        .execute(&store.pool)
        .await
        .unwrap();
    store.project_fact_memory(fact_id).await.unwrap();

    let guessed: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities WHERE entity_type IN ('technology', 'project')",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(guessed, 0);
}

#[tokio::test]
async fn grounded_structured_graph_is_linked_to_claim() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "global").await;
    store.project_fact_memory(fact_id).await.unwrap();
    let graph = ExtractedMemoryGraph {
        entities: vec![
            ExtractedMemoryEntity {
                local_id: "owner".to_string(),
                name: "owner".to_string(),
                entity_type: "person".to_string(),
                aliases: vec![],
                confidence: 0.99,
            },
            ExtractedMemoryEntity {
                local_id: "db".to_string(),
                name: "PostgreSQL".to_string(),
                entity_type: "technology".to_string(),
                aliases: vec!["Postgres".to_string()],
                confidence: 0.92,
            },
        ],
        relationships: vec![ExtractedMemoryRelationship {
            source_id: "owner".to_string(),
            target_id: "db".to_string(),
            relation: "uses database".to_string(),
            confidence: 0.9,
        }],
    };
    store
        .persist_extracted_fact_graph(
            "preference",
            "default_editor",
            "The owner uses PostgreSQL (Postgres) for local projects.",
            &graph,
        )
        .await
        .unwrap();

    let semantic_edges: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_edges WHERE relation = 'uses_database'
         AND source_claim_id IN (SELECT id FROM memory_claims WHERE source_fact_id = ?)",
    )
    .bind(fact_id)
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(semantic_edges, 1);

    let alias_graph = ExtractedMemoryGraph {
        entities: vec![ExtractedMemoryEntity {
            local_id: "db2".to_string(),
            name: "Postgres".to_string(),
            entity_type: "technology".to_string(),
            aliases: vec!["PostgreSQL".to_string()],
            confidence: 0.95,
        }],
        relationships: vec![],
    };
    store
        .persist_extracted_fact_graph(
            "preference",
            "default_editor",
            "Postgres remains the selected database.",
            &alias_graph,
        )
        .await
        .unwrap();
    let technology_entities: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_entities WHERE entity_type = 'technology'")
            .fetch_one(&store.pool)
            .await
            .unwrap();
    assert_eq!(technology_entities, 1);
}

#[tokio::test]
async fn ungrounded_and_low_confidence_graph_output_is_rejected() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "global").await;
    store.project_fact_memory(fact_id).await.unwrap();
    let graph = ExtractedMemoryGraph {
        entities: vec![
            ExtractedMemoryEntity {
                local_id: "invented".to_string(),
                name: "Imaginary Corp".to_string(),
                entity_type: "organization".to_string(),
                aliases: vec![],
                confidence: 0.99,
            },
            ExtractedMemoryEntity {
                local_id: "weak".to_string(),
                name: "Neovim".to_string(),
                entity_type: "technology".to_string(),
                aliases: vec![],
                confidence: 0.4,
            },
        ],
        relationships: vec![ExtractedMemoryRelationship {
            source_id: "invented".to_string(),
            target_id: "weak".to_string(),
            relation: "owns".to_string(),
            confidence: 0.99,
        }],
    };
    store
        .persist_extracted_fact_graph(
            "preference",
            "default_editor",
            "I use Neovim as my editor.",
            &graph,
        )
        .await
        .unwrap();

    let semantic_entities: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities WHERE entity_type IN ('organization', 'technology')",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(semantic_entities, 0);
}

#[tokio::test]
async fn supersession_invalidates_claim_vector_and_graph_edge() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "global").await;
    store.project_fact_memory(fact_id).await.unwrap();
    sqlx::query("UPDATE facts SET superseded_at = '2026-01-03T00:00:00Z' WHERE id = ?")
        .bind(fact_id)
        .execute(&store.pool)
        .await
        .unwrap();
    store.project_fact_memory(fact_id).await.unwrap();

    let active_claims: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_claims WHERE valid_to IS NULL")
            .fetch_one(&store.pool)
            .await
            .unwrap();
    let active_vectors: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_embeddings WHERE stale_at IS NULL")
            .fetch_one(&store.pool)
            .await
            .unwrap();
    let active_edges: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_edges WHERE valid_to IS NULL")
            .fetch_one(&store.pool)
            .await
            .unwrap();
    assert_eq!((active_claims, active_vectors, active_edges), (0, 0, 0));
}

#[tokio::test]
async fn fts_recall_applies_visibility_at_claim_load() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "private").await;
    store.project_fact_memory(fact_id).await.unwrap();

    let owner_hits = store
        .search_memory_claims("default editor", None, ChannelVisibility::Private, true, 10)
        .await
        .unwrap();
    let public_hits = store
        .search_memory_claims(
            "default editor",
            Some("telegram:synthetic-user-1"),
            ChannelVisibility::Public,
            false,
            10,
        )
        .await
        .unwrap();
    assert_eq!(owner_hits.len(), 1);
    assert_eq!(owner_hits[0].source_fact_id, Some(fact_id));
    assert!(public_hits.is_empty());
}

#[tokio::test]
async fn graph_traversal_is_bounded_to_two_hops() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "global").await;
    store.project_fact_memory(fact_id).await.unwrap();
    let owner_id: i64 =
        sqlx::query_scalar("SELECT id FROM memory_entities WHERE canonical_name = 'owner'")
            .fetch_one(&store.pool)
            .await
            .unwrap();

    let neighbors = MemoryGraphStore::neighbors(
        store.memory(),
        owner_id,
        99,
        &GraphSearchScope {
            channel_id: None,
            visibility: ChannelVisibility::Internal,
            requester_is_owner: true,
        },
    )
    .await
    .unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].display_name, "default_editor");
    assert_eq!(neighbors[0].depth, 1);
    assert!(neighbors.iter().all(|neighbor| neighbor.depth <= 2));
}

#[tokio::test]
async fn startup_backfill_does_not_promote_unselected_messages_to_memory() {
    let (_dir, store) = test_store().await;
    let mut tx = store.pool.begin().await.unwrap();
    for i in 0..1_001 {
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at)
             VALUES ('telegram:synthetic-user-1', 'user_message', ?, '2026-01-01T00:00:00Z')",
        )
        .bind(serde_json::json!({"content": format!("synthetic message {i}")}).to_string())
        .execute(&mut *tx)
        .await
        .unwrap();
    }
    tx.commit().await.unwrap();

    let (_, _, first_count, _, _) = store.backfill_missing_memory_projections().await.unwrap();
    let (_, _, second_count, _, _) = store.backfill_missing_memory_projections().await.unwrap();
    let spans: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_spans WHERE span_kind = 'message'")
            .fetch_one(&store.pool)
            .await
            .unwrap();
    assert_eq!(first_count, 0);
    assert_eq!(second_count, 0);
    assert_eq!(spans, 0);
}

#[tokio::test]
async fn selected_fact_projects_only_its_canonical_source_message() {
    let (_dir, store) = test_store().await;
    let selected_event_id: i64 = sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at)
         VALUES ('telegram:synthetic-user-1', 'user_message', ?, '2026-01-01T00:00:00Z')",
    )
    .bind(serde_json::json!({"content": "My default editor is Neovim"}).to_string())
    .execute(&store.pool)
    .await
    .unwrap()
    .last_insert_rowid();
    let control_event_id: i64 = sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at)
         VALUES ('telegram:synthetic-user-1', 'user_message', ?, '2026-01-01T00:01:00Z')",
    )
    .bind(serde_json::json!({"content": "Run the control-plane check now"}).to_string())
    .execute(&store.pool)
    .await
    .unwrap()
    .last_insert_rowid();
    let fact_id = insert_fact(&store, "private").await;
    sqlx::query(
        "UPDATE facts SET source_excerpt = 'My default editor is Neovim',
                          first_seen_at = '2026-01-01T00:00:00Z'
         WHERE id = ?",
    )
    .bind(fact_id)
    .execute(&store.pool)
    .await
    .unwrap();

    aidaemon_runtime::history::project_event_span(&store.pool, selected_event_id)
        .await
        .unwrap();
    store.project_fact_memory(fact_id).await.unwrap();

    let selected_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_spans WHERE source_event_id = ?")
            .bind(selected_event_id)
            .fetch_one(&store.pool)
            .await
            .unwrap();
    let control_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_spans WHERE source_event_id = ?")
            .bind(control_event_id)
            .fetch_one(&store.pool)
            .await
            .unwrap();
    assert_eq!(selected_count, 1);
    assert_eq!(control_count, 0);
}

#[tokio::test]
async fn startup_backfill_skips_unprojected_superseded_facts() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "private").await;
    sqlx::query(
        "UPDATE facts
         SET superseded_at = '2026-01-02T00:00:00Z'
         WHERE id = ?",
    )
    .bind(fact_id)
    .execute(&store.pool)
    .await
    .unwrap();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        store.backfill_missing_memory_projections(),
    )
    .await
    .expect("backfill must not loop on an unprojected superseded fact")
    .unwrap();

    assert_eq!(result.0, 0);
    let claim_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_claims WHERE source_fact_id = ?")
            .bind(fact_id)
            .fetch_one(&store.pool)
            .await
            .unwrap();
    assert_eq!(claim_count, 0);
}

#[tokio::test]
async fn exact_index_filters_and_ranks_thousands_of_vectors() {
    let (_dir, store) = test_store().await;
    let mut tx = store.pool.begin().await.unwrap();
    for i in 0..5_000i64 {
        let vector = if i == 3_777 {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        };
        sqlx::query(
            "INSERT INTO memory_embeddings
                (owner_type, owner_id, embedding_purpose, embedding_model,
                 embedding_dim, content_hash, embedding, created_at, updated_at)
             VALUES ('claim', ?, 'recall', ?, 2, ?, ?, datetime('now'), datetime('now'))",
        )
        .bind(i.to_string())
        .bind(EMBEDDING_MODEL_ID)
        .bind(format!("hash-{i}"))
        .bind(encode_embedding(&vector))
        .execute(&mut *tx)
        .await
        .unwrap();
    }
    // A distractor purpose must not enter the recall candidate set.
    sqlx::query(
        "INSERT INTO memory_embeddings
            (owner_type, owner_id, embedding_purpose, embedding_model,
             embedding_dim, content_hash, embedding, created_at, updated_at)
         VALUES ('claim', 'distractor', 'dedup', ?, 2, 'distractor', ?, datetime('now'), datetime('now'))",
    )
    .bind(EMBEDDING_MODEL_ID)
    .bind(encode_embedding(&[1.0, 0.0]))
    .execute(&mut *tx)
    .await
    .unwrap();
    // Corrupt payload metadata is skipped rather than failing the search.
    sqlx::query(
        "INSERT INTO memory_embeddings
            (owner_type, owner_id, embedding_purpose, embedding_model,
             embedding_dim, content_hash, embedding, created_at, updated_at)
         VALUES ('claim', 'corrupt', 'recall', ?, 2, 'corrupt', ?, datetime('now'), datetime('now'))",
    )
    .bind(EMBEDDING_MODEL_ID)
    .bind(encode_embedding(&[1.0]))
    .execute(&mut *tx)
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let hits = SqliteEmbeddingIndex::new(store.pool.clone())
        .search(
            &[1.0, 0.0],
            &EmbeddingSearchFilter {
                owner_types: vec!["claim".to_string()],
                purpose: Some("recall".to_string()),
                model: Some(EMBEDDING_MODEL_ID.to_string()),
            },
            5,
        )
        .await
        .unwrap();
    assert_eq!(hits[0].owner.owner_id, "3777");
    assert!(hits.iter().all(|hit| hit.owner.owner_id != "distractor"));
    assert!(hits.iter().all(|hit| hit.owner.owner_id != "corrupt"));
    assert_eq!(
        store
            .canonical_memory_health()
            .await
            .unwrap()
            .embedding_dimension_mismatches,
        1
    );
}

#[tokio::test]
async fn derived_projection_failure_does_not_rollback_authoritative_write() {
    let (_dir, store) = test_store().await;
    let fact_id = insert_fact(&store, "global").await;
    sqlx::query("DROP TABLE memory_claims")
        .execute(&store.pool)
        .await
        .unwrap();

    store
        .update_fact_privacy(fact_id, FactPrivacy::Private)
        .await
        .unwrap();
    let privacy: String = sqlx::query_scalar("SELECT privacy FROM facts WHERE id = ?")
        .bind(fact_id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
    assert_eq!(privacy, "private");
}
