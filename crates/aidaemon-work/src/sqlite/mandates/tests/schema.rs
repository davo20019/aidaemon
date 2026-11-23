//! Tests of the mandate tables' migrations and reopening.

use super::*;

/// A database whose `mandates` table predates the owner-given name
/// gains the nullable column on the next migration; old rows read back
/// unnamed and new ones keep their name.
#[tokio::test]
async fn an_existing_mandates_table_gains_the_name_column() {
    let (store, _database) = test_store().await;
    let pool = store.pool();
    let (old_goal, old_mandate) = controller("owner-session", 0);
    store
        .create_mandate_controller(&old_goal, &old_mandate)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE mandates DROP COLUMN name")
        .execute(&pool)
        .await
        .unwrap();

    crate::schema::migrate_work_tables(&pool, |_| false)
        .await
        .unwrap();

    let unnamed = store.get_mandate(&old_mandate.id).await.unwrap().unwrap();
    assert_eq!(unnamed.name, None);
    let (goal, mut mandate) = controller("owner-session", 0);
    mandate.name = Some("synthetic-watch".to_string());
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    let named = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(named.name.as_deref(), Some("synthetic-watch"));
    assert!(named.has_name("Synthetic-Watch"));
}

#[tokio::test]
async fn mandate_schema_and_rows_survive_an_idempotent_reopen() {
    let (store, database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    store.pool.close().await;
    drop(store);

    let reopened = open_for_tests(database.path().to_str().unwrap()).await;
    let loaded = reopened.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(loaded.goal_id, goal.id);
    assert_eq!(loaded.version, mandate.version);
    assert_eq!(loaded.next_review_at, mandate.next_review_at);
}
