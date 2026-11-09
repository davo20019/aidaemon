use super::*;
use crate::memory::embeddings::EmbeddingService;
use crate::traits::FactStore;
use std::sync::Arc;

async fn store() -> (tempfile::TempDir, SqliteStateStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("structured-memory.db");
    let embeddings = Arc::new(EmbeddingService::new().unwrap());
    let store = SqliteStateStore::new(path.to_str().unwrap(), 20, None, embeddings)
        .await
        .unwrap();
    (dir, store)
}

fn entity(local_id: &str, canonical_name: &str, is_reference: bool) -> PersonalEntityCandidate {
    PersonalEntityCandidate {
        local_id: local_id.to_string(),
        entity_type: "person".to_string(),
        canonical_name: canonical_name.to_string(),
        is_reference,
        canonical_name_confirmed: !is_reference,
    }
}

fn alias(local_id: &str, value: &str, kind: &str) -> PersonalAliasCandidate {
    PersonalAliasCandidate {
        entity_local_id: local_id.to_string(),
        value: value.to_string(),
        alias_type: kind.to_string(),
    }
}

fn fact(local_id: &str, predicate: &str, value: &str) -> PersonalFactCandidate {
    PersonalFactCandidate {
        subject_local_id: local_id.to_string(),
        predicate: predicate.to_string(),
        value: value.to_string(),
        display_value: None,
        valid_from: None,
        valid_to: None,
    }
}

fn edge(from: &str, relation: &str, to: &str) -> PersonalRelationshipCandidate {
    PersonalRelationshipCandidate {
        source_local_id: from.to_string(),
        relationship_type: relation.to_string(),
        target_local_id: to.to_string(),
        valid_from: None,
        valid_to: None,
    }
}

#[tokio::test]
async fn owner_identity_aliases_and_handle_are_idempotent() {
    let (_dir, store) = store().await;
    let write = PersonalMemoryWrite {
        entities: vec![entity("owner", "Alice Morgan Rivera", false)],
        aliases: vec![
            alias("owner", "Morgan", "preferred_name"),
            alias("owner", "rivera_dev_42", "online_handle"),
        ],
        direct_user_statement: true,
        ..Default::default()
    };
    store
        .reconcile_structured_personal_memory(
            &write,
            "agent",
            Some("synthetic owner statement"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    store
        .reconcile_structured_personal_memory(
            &write,
            "agent",
            Some("synthetic owner statement"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();

    let owners: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities WHERE is_owner = 1 AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let canonical: String =
        sqlx::query_scalar("SELECT canonical_name FROM memory_entities WHERE is_owner = 1")
            .fetch_one(&store.pool)
            .await
            .unwrap();
    let aliases: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_aliases
         WHERE entity_id = (SELECT id FROM memory_entities WHERE is_owner = 1)",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let separate_preferred_person: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities
         WHERE entity_type = 'person' AND canonical_name = 'morgan' AND is_owner = 0",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(owners, 1);
    assert_eq!(canonical, "alice morgan rivera");
    assert_eq!(aliases, 2);
    assert_eq!(separate_preferred_person, 0);
}

#[tokio::test]
async fn children_aliases_dates_and_inverse_relationships_are_canonical() {
    let (_dir, store) = store().await;
    let first = PersonalMemoryWrite {
        entities: vec![
            entity("owner", "Owner", false),
            entity("child_one", "Alexandra River", false),
            entity("child_two", "Jordan River", false),
        ],
        aliases: vec![
            alias("child_one", "Lexi", "nickname"),
            alias("child_two", "Jori", "nickname"),
        ],
        facts: vec![
            fact("child_one", "birth_date", "April 3, 2019"),
            fact("child_two", "birth_date", "November 12 2011"),
        ],
        relationships: vec![
            edge("owner", "PARENT_OF", "child_one"),
            edge("owner", "PARENT_OF", "child_two"),
        ],
        direct_user_statement: true,
        ..Default::default()
    };
    store
        .reconcile_structured_personal_memory(
            &first,
            "agent",
            Some("synthetic family statement"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();

    let later = PersonalMemoryWrite {
        entities: vec![entity("child", "Lexi", true)],
        facts: vec![fact("child", "favorite_color", "violet")],
        direct_user_statement: true,
        ..Default::default()
    };
    store
        .reconcile_structured_personal_memory(
            &later,
            "agent",
            Some("synthetic alias reuse"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    let residence = PersonalMemoryWrite {
        entities: vec![
            entity("owner", "Owner", false),
            entity("first", "Lexi", true),
            entity("second", "Jori", true),
            PersonalEntityCandidate {
                local_id: "city".to_string(),
                entity_type: "place".to_string(),
                canonical_name: "Riverton, VA".to_string(),
                is_reference: false,
                canonical_name_confirmed: true,
            },
        ],
        relationships: vec![
            edge("first", "LIVES_WITH", "owner"),
            edge("second", "LIVES_WITH", "owner"),
            edge("owner", "LIVES_IN", "city"),
        ],
        direct_user_statement: true,
        ..Default::default()
    };
    store
        .reconcile_structured_personal_memory(
            &residence,
            "agent",
            Some("synthetic shared residence"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    // Repeat the relationship-bearing statement to verify full idempotency.
    store
        .reconcile_structured_personal_memory(
            &first,
            "agent",
            Some("synthetic family statement"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();

    let people: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities WHERE entity_type = 'person' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let birth_dates: Vec<String> = sqlx::query_scalar(
        "SELECT normalized_value FROM memory_entity_facts
         WHERE predicate = 'birth_date' AND status = 'active' ORDER BY normalized_value",
    )
    .fetch_all(&store.pool)
    .await
    .unwrap();
    let parent_edges: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships
         WHERE relationship_type = 'PARENT_OF' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let child_edges: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships
         WHERE relationship_type = 'CHILD_OF' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let nickname_entities: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_entities WHERE canonical_name = 'lexi'")
            .fetch_one(&store.pool)
            .await
            .unwrap();
    let child_lives_with_owner: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships r
         JOIN memory_entities source ON source.id = r.source_entity_id
         JOIN memory_entities target ON target.id = r.target_entity_id
         WHERE r.relationship_type = 'LIVES_WITH' AND source.is_owner = 0
           AND target.is_owner = 1 AND r.status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let owner_lives_in: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships r
         JOIN memory_entities source ON source.id = r.source_entity_id
         WHERE r.relationship_type = 'LIVES_IN' AND source.is_owner = 1
           AND r.status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(people, 3);
    assert_eq!(birth_dates, vec!["2011-11-12", "2019-04-03"]);
    assert_eq!((parent_edges, child_edges), (2, 2));
    assert_eq!(nickname_entities, 0);
    assert_eq!(child_lives_with_owner, 2);
    assert_eq!(owner_lives_in, 1);
}

#[tokio::test]
async fn direct_correction_supersedes_legacy_and_retrieval_prefers_active() {
    let (_dir, store) = store().await;
    let legacy = PersonalMemoryWrite {
        entities: vec![entity("owner", "Owner", false)],
        facts: vec![fact("owner", "birth_date", "March 6 1991")],
        ..Default::default()
    };
    store
        .reconcile_structured_personal_memory(
            &legacy,
            "legacy",
            Some("synthetic import"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    store
        .upsert_fact(
            "user",
            "birthday",
            "1991-03-06",
            "legacy",
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    let correction = PersonalMemoryWrite {
        entities: vec![entity("owner", "Owner", false)],
        facts: vec![fact("owner", "birth_date", "February 4 1988")],
        direct_user_statement: true,
        ..Default::default()
    };
    let result = store
        .reconcile_structured_personal_memory(
            &correction,
            "agent",
            Some("synthetic correction"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    assert_eq!(result.superseded_facts, 1);

    let active: String = sqlx::query_scalar(
        "SELECT normalized_value FROM memory_entity_facts
         WHERE predicate = 'birth_date' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let superseded: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entity_facts
         WHERE predicate = 'birth_date' AND status = 'superseded'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let recalled = store
        .get_relevant_facts("what is my birth date", 10)
        .await
        .unwrap();
    assert_eq!(active, "1988-02-04");
    assert_eq!(superseded, 1);
    assert!(recalled
        .iter()
        .any(|fact| { fact.source.starts_with("canonical:") && fact.value == "February 4 1988" }));
    assert!(!recalled.iter().any(|fact| fact.value == "1991-03-06"));
}

#[tokio::test]
async fn confirmed_conflicts_and_ambiguous_aliases_require_review() {
    let (_dir, store) = store().await;
    let initial = PersonalMemoryWrite {
        entities: vec![
            entity("owner", "Owner", false),
            entity("one", "Jordan Lee", false),
            entity("two", "Jordan Patel", false),
        ],
        aliases: vec![
            alias("one", "Jo", "nickname"),
            alias("two", "Jo", "nickname"),
        ],
        facts: vec![fact("owner", "birth_date", "2000-01-01")],
        direct_user_statement: true,
        ..Default::default()
    };
    store
        .reconcile_structured_personal_memory(
            &initial,
            "agent",
            Some("synthetic confirmed facts"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    let conflict = PersonalMemoryWrite {
        entities: vec![entity("owner", "Owner", false)],
        facts: vec![fact("owner", "birth_date", "2001-01-01")],
        direct_user_statement: true,
        correction: false,
        ..Default::default()
    };
    let conflict_result = store
        .reconcile_structured_personal_memory(
            &conflict,
            "agent",
            Some("synthetic conflicting statement"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    let ambiguous = PersonalMemoryWrite {
        entities: vec![entity("reference", "Jo", true)],
        facts: vec![fact("reference", "city", "Raleigh")],
        direct_user_statement: true,
        ..Default::default()
    };
    let ambiguous_result = store
        .reconcile_structured_personal_memory(
            &ambiguous,
            "agent",
            Some("synthetic ambiguous reference"),
            None,
            FactPrivacy::Private,
        )
        .await
        .unwrap();
    let reviews: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_resolution_reviews WHERE status = 'pending'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(conflict_result.disputed_facts, 1);
    assert!(!ambiguous_result.unresolved.is_empty());
    assert_eq!(reviews, 2);
}

#[tokio::test]
async fn legacy_parent_records_backfill_separately_and_quarantine_compound_fact() {
    let (_dir, store) = store().await;
    let now = Utc::now().to_rfc3339();
    for (name, relationship, aliases) in [
        ("Alice Rivera", "owner", "[]"),
        ("Marco Rivera", "father", "[]"),
        ("Sofia Mendez", "mother", "[\"Sofi\"]"),
    ] {
        sqlx::query(
            "INSERT INTO people
             (name, aliases_json, relationship, platform_ids_json, created_at, updated_at)
             VALUES (?, ?, ?, '{}', ?, ?)",
        )
        .bind(name)
        .bind(aliases)
        .bind(relationship)
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();
    }
    sqlx::query(
        "INSERT INTO facts
         (category, key, value, source, created_at, updated_at, privacy)
         VALUES ('user', 'father_name', 'Marco Rivera and Sofia Mendez (Sofi)',
                 'legacy', ?, ?, 'private')",
    )
    .bind(&now)
    .bind(&now)
    .execute(&store.pool)
    .await
    .unwrap();

    store.backfill_structured_personal_memory().await.unwrap();
    let parents: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships
         WHERE relationship_type = 'PARENT_OF' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let combined_entity: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities
         WHERE canonical_name = 'marco rivera and sofia mendez sofi'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let reviews: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_resolution_reviews
         WHERE review_kind = 'compound_parent_identity' AND status = 'pending'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(parents, 2);
    assert_eq!(combined_entity, 0);
    assert_eq!(reviews, 1);
}

#[tokio::test]
async fn explicit_flat_relationship_facts_backfill_idempotently() {
    let (_dir, store) = store().await;
    let now = Utc::now().to_rfc3339();
    for (name, relationship) in [
        ("Alice Rivera", "owner"),
        ("Mia Rivera", ""),
        ("Zoe Rivera", ""),
    ] {
        sqlx::query(
            "INSERT INTO people
             (name, aliases_json, relationship, platform_ids_json, created_at, updated_at)
             VALUES (?, '[]', ?, '{}', ?, ?)",
        )
        .bind(name)
        .bind(relationship)
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();
    }
    for (key, value) in [
        ("dad_name", "Jordan Lee"),
        ("mother_name", "Alexandra \"Lexi\" Rivera"),
        ("daughter_names", "Mia Rivera and Zoe Rivera"),
    ] {
        sqlx::query(
            "INSERT INTO facts
             (category, key, value, source, created_at, updated_at, privacy)
             VALUES ('user', ?, ?, 'user_stated', ?, ?, 'private')",
        )
        .bind(key)
        .bind(value)
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();
    }

    store.backfill_structured_personal_memory().await.unwrap();
    store.backfill_structured_personal_memory().await.unwrap();

    let parent_edges: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships
         WHERE relationship_type = 'PARENT_OF' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let child_edges: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships
         WHERE relationship_type = 'CHILD_OF' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let nickname_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_aliases
         WHERE alias_type = 'nickname' AND normalized_value = 'lexi'
           AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!((parent_edges, child_edges), (4, 4));
    assert_eq!(nickname_count, 1);
}

#[tokio::test]
async fn explicit_nickname_and_matching_relationship_merge_legacy_short_name_entity() {
    let (_dir, store) = store().await;
    let now = Utc::now().to_rfc3339();
    for (name, relationship) in [("Alice Rivera", "owner"), ("Sol", "mother")] {
        sqlx::query(
            "INSERT INTO people
             (name, aliases_json, relationship, platform_ids_json, created_at, updated_at)
             VALUES (?, '[]', ?, '{}', ?, ?)",
        )
        .bind(name)
        .bind(relationship)
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();
    }
    sqlx::query(
        "INSERT INTO facts
         (category, key, value, source, created_at, updated_at, privacy)
         VALUES ('user', 'mother_name', 'Marisol Vega \"Sol\"',
                 'user_stated', ?, ?, 'private')",
    )
    .bind(&now)
    .bind(&now)
    .execute(&store.pool)
    .await
    .unwrap();

    store.backfill_structured_personal_memory().await.unwrap();
    store.backfill_structured_personal_memory().await.unwrap();

    let active_people: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities
         WHERE status = 'active'
           AND canonical_name IN ('sol', 'marisol vega')",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let merged_people: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities
         WHERE status = 'merged' AND canonical_name = 'sol'
           AND merged_into_entity_id IS NOT NULL",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let active_parent_edges: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships
         WHERE status = 'active' AND relationship_type = 'PARENT_OF'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(active_people, 1);
    assert_eq!(merged_people, 1);
    assert_eq!(active_parent_edges, 1);
}

#[tokio::test]
async fn legacy_full_names_promote_short_owner_and_child_names_idempotently() {
    let (_dir, store) = store().await;
    let now = Utc::now().to_rfc3339();
    for (name, relationship) in [("Morgan", "owner"), ("Zo", "daughter")] {
        sqlx::query(
            "INSERT INTO people
             (name, aliases_json, relationship, platform_ids_json, created_at, updated_at)
             VALUES (?, '[]', ?, '{}', ?, ?)",
        )
        .bind(name)
        .bind(relationship)
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();
    }
    for (key, value) in [
        ("full_name", "Alice Morgan Rivera"),
        ("daughter_zoe_full_name", "Zoe (nickname Zo)"),
        ("daughter_zo_birthday", "April 12, 2015"),
        (
            "daughters_live_with_user",
            "Both daughters live with me in Exampletown, VA",
        ),
    ] {
        sqlx::query(
            "INSERT INTO facts
             (category, key, value, source, created_at, updated_at, privacy)
             VALUES ('user', ?, ?, 'user_stated', ?, ?, 'private')",
        )
        .bind(key)
        .bind(value)
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();
    }

    store.backfill_structured_personal_memory().await.unwrap();
    store.backfill_structured_personal_memory().await.unwrap();

    let owner_name: String = sqlx::query_scalar(
        "SELECT canonical_name FROM memory_entities
         WHERE is_owner = 1 AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let zoe_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities
         WHERE canonical_name = 'zoe' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let zo_entity_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_entities
         WHERE canonical_name = 'zo' AND status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let zo_alias_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_aliases a
         JOIN memory_entities e ON e.id = a.entity_id
         WHERE e.canonical_name = 'zoe' AND e.status = 'active'
           AND a.alias_type = 'nickname' AND a.normalized_value = 'zo'
           AND a.status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let zoe_birth_date: String = sqlx::query_scalar(
        "SELECT f.normalized_value FROM memory_entity_facts f
         JOIN memory_entities e ON e.id = f.subject_entity_id
         WHERE e.canonical_name = 'zoe' AND e.status = 'active'
           AND f.predicate = 'birth_date' AND f.status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    let child_lives_with_owner: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_relationships r
         JOIN memory_entities child ON child.id = r.source_entity_id
         JOIN memory_entities owner ON owner.id = r.target_entity_id
         WHERE child.canonical_name = 'zoe' AND child.status = 'active'
           AND owner.is_owner = 1 AND owner.status = 'active'
           AND r.relationship_type = 'LIVES_WITH' AND r.status = 'active'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(owner_name, "alice morgan rivera");
    assert_eq!(zoe_count, 1);
    assert_eq!(zo_entity_count, 0);
    assert_eq!(zo_alias_count, 1);
    assert_eq!(zoe_birth_date, "2015-04-12");
    assert_eq!(child_lives_with_owner, 1);
}
