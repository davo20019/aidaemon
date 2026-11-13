use super::*;
use crate::state::types::FactPrivacy;
use crate::state::types::FactStore;
use aidaemon_memory::embeddings::EmbeddingService;
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

/// The owner's family cluster travels as related context of an explicit
/// search: a mother hit brings the father, who shares no word with it.
#[tokio::test]
async fn search_facts_relates_owner_family_cluster() {
    let (store, _db) = setup_store().await;
    for (key, value) in [
        ("mother_name", "Carol Mendez"),
        ("father", "Frank Mendez"),
        ("partner_name", "Alice Rivera"),
    ] {
        store
            .upsert_fact("user", key, value, "test", None, FactPrivacy::Global)
            .await
            .unwrap();
    }
    let report = store
        .search_facts(
            "mother",
            1,
            &aidaemon_memory::search::FactSearchConfig::default(),
        )
        .await
        .unwrap();
    assert_eq!(report.hits.len(), 1);
    assert_eq!(report.hits[0].fact.value, "Carol Mendez");
    assert!(
        report
            .related
            .iter()
            .any(|h| h.fact.value == "Frank Mendez"),
        "father pulled into the cluster: {:?}",
        report.related
    );
}
