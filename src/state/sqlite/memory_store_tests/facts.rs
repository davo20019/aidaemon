use super::*;
use crate::memory::embeddings::EmbeddingService;
use crate::traits::{FactStore, PeopleStore};
use crate::types::FactPrivacy;
use std::sync::Arc;

async fn setup_store() -> (SqliteStateStore, tempfile::NamedTempFile) {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = SqliteStateStore::new(
        db_file.path().to_str().unwrap(),
        100,
        None,
        embedding_service,
    )
    .await
    .unwrap();
    (store, db_file)
}

#[tokio::test]
async fn assemble_neighborhood_pulls_owner_family_cluster() {
    let (store, _db) = setup_store().await;

    store
        .upsert_fact(
            "user",
            "mother_name",
            "Carol Mendez",
            "test",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    store
        .upsert_fact(
            "user",
            "father",
            "Frank Mendez",
            "test",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    store
        .upsert_fact(
            "user",
            "partner_name",
            "Alice Rivera",
            "test",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Pretend the search already matched the mother fact (id resolved below).
    let mother = store
        .get_facts(Some("user"))
        .await
        .unwrap()
        .into_iter()
        .find(|f| f.key == "mother_name")
        .unwrap();
    let initial: std::collections::HashSet<i64> = [mother.id].into_iter().collect();

    let out = store
        .assemble_neighborhood(&["Carol".to_string()], &initial)
        .await
        .unwrap();
    let values: Vec<String> = out.iter().map(|f| f.value.clone()).collect();
    assert!(
        values.iter().any(|v| v.contains("Frank")),
        "father pulled into cluster: {:?}",
        values
    );
}

/// Tests the *primary* person-resolution path: entity name → resolved Person
/// with a `relationship` role → `owner_relationship = true` → cluster pulls
/// in co-relations.
///
/// Critically, `initial_ids` is EMPTY so the fallback "initial-match has a
/// relationship key" trigger cannot fire.  Only the Person-record path can
/// enable the owner cluster here.
#[tokio::test]
async fn assemble_neighborhood_resolves_person_relationship() {
    let (store, _db) = setup_store().await;

    // Seed a Person record with a relationship role.
    let person = crate::traits::Person {
        id: 0,
        name: "Carol Mendez".to_string(),
        aliases: vec![],
        relationship: Some("mother".to_string()),
        platform_ids: std::collections::HashMap::new(),
        notes: None,
        communication_style: None,
        language_preference: None,
        last_interaction_at: None,
        interaction_count: 0,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.upsert_person(&person).await.unwrap();

    // Seed flat facts that belong to the owner-relationship cluster.
    store
        .upsert_fact(
            "user",
            "father",
            "Frank Mendez",
            "test",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    store
        .upsert_fact(
            "user",
            "partner_name",
            "Alice Rivera",
            "test",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Empty initial_ids — the fallback trigger (initial fact has relationship
    // key) CANNOT fire.  Only the Person-record resolution path can enable
    // the owner cluster.
    let initial: std::collections::HashSet<i64> = std::collections::HashSet::new();

    let out = store
        .assemble_neighborhood(&["Carol".to_string()], &initial)
        .await
        .unwrap();
    let values: Vec<String> = out.iter().map(|f| f.value.clone()).collect();
    assert!(
        values.iter().any(|v| v.contains("Frank")),
        "father should be pulled into cluster via person-resolution path: {:?}",
        values
    );
}

/// Word-boundary guard: entity "Ana" must NOT resolve the person "Banana Doe".
#[tokio::test]
async fn assemble_neighborhood_person_match_is_word_boundary() {
    let (store, _db) = setup_store().await;

    // "Banana Doe" — contains "ana" as a substring but NOT as a whole word.
    let person = crate::traits::Person {
        id: 0,
        name: "Banana Doe".to_string(),
        aliases: vec![],
        relationship: Some("friend".to_string()),
        platform_ids: std::collections::HashMap::new(),
        notes: None,
        communication_style: None,
        language_preference: None,
        last_interaction_at: None,
        interaction_count: 0,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.upsert_person(&person).await.unwrap();

    // A fact whose value would only appear if the person cluster were enabled.
    store
        .upsert_fact(
            "user",
            "best_friend",
            "Banana Doe",
            "test",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    store
        .upsert_fact(
            "user",
            "hobby",
            "gardening",
            "test",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Search for "Ana" — should NOT match "Banana Doe" via word-boundary.
    let initial: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let out = store
        .assemble_neighborhood(&["Ana".to_string()], &initial)
        .await
        .unwrap();
    // The owner cluster must NOT be triggered by false substring match.
    // "hobby" fact is unrelated; if "Ana" matched "Banana Doe", the friend's
    // relationship role would enable the cluster and pull in all relationship
    // facts.  Assert the result is empty (no initial hits, no cluster).
    assert!(
        out.is_empty(),
        "entity 'Ana' must not resolve person 'Banana Doe' via substring: {:?}",
        out
    );
}
