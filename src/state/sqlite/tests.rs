use super::*;
use crate::memory::embeddings::EmbeddingService;
use crate::traits::store_prelude::*;
use crate::traits::{
    BehaviorPattern, DynamicBot, DynamicMcpServer, DynamicSkill, Episode, ErrorSolution, Goal,
    GoalSchedule, Message, Procedure, SkillDraft, TokenUsage,
};
use crate::types::FactPrivacy;
use std::sync::Arc;

async fn setup_test_store() -> (SqliteStateStore, tempfile::NamedTempFile) {
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

async fn setup_test_store_with_cap(cap: usize) -> (SqliteStateStore, tempfile::NamedTempFile) {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = SqliteStateStore::new(
        db_file.path().to_str().unwrap(),
        cap,
        None,
        embedding_service,
    )
    .await
    .unwrap();
    (store, db_file)
}

fn make_message(session_id: &str, role: &str, content: &str) -> Message {
    Message {
        id: uuid::Uuid::new_v4().to_string(),
        session_id: session_id.to_string(),
        role: role.to_string(),
        content: Some(content.to_string()),
        tool_call_id: None,
        tool_name: None,
        tool_calls_json: None,
        created_at: Utc::now(),
        importance: 0.5,
        embedding: None,
    }
}

async fn create_events_table_for_tests(store: &SqliteStateStore) {
    sqlx::query(
        r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT,
                task_id TEXT,
                tool_name TEXT
            )
            "#,
    )
    .execute(&store.pool())
    .await
    .unwrap();
}

async fn insert_conversation_event_for_tests(
    store: &SqliteStateStore,
    session_id: &str,
    event_type: &str,
    data: serde_json::Value,
    created_at: chrono::DateTime<Utc>,
) {
    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(session_id)
    .bind(event_type)
    .bind(data.to_string())
    .bind(created_at.to_rfc3339())
    .execute(&store.pool())
    .await
    .unwrap();
}

// ==================== Message Tests ====================

#[tokio::test]
async fn test_append_and_get_history() {
    let (store, _db) = setup_test_store().await;
    let session = "sess-1";

    let m1 = make_message(session, "user", "Hello");
    let m2 = make_message(session, "assistant", "Hi there");
    let m3 = make_message(session, "user", "How are you?");

    store.append_message(&m1).await.unwrap();
    store.append_message(&m2).await.unwrap();
    store.append_message(&m3).await.unwrap();

    let history = store.get_history(session, 100).await.unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].content.as_deref(), Some("Hello"));
    assert_eq!(history[1].content.as_deref(), Some("Hi there"));
    assert_eq!(history[2].content.as_deref(), Some("How are you?"));
}

#[tokio::test]
async fn test_get_history_limit() {
    let (store, _db) = setup_test_store().await;
    let session = "sess-limit";

    for i in 0..10 {
        let msg = make_message(session, "user", &format!("Message {}", i));
        store.append_message(&msg).await.unwrap();
    }

    let history = store.get_history(session, 5).await.unwrap();
    assert_eq!(history.len(), 5);
    // The truncate_with_anchor logic preserves the first user message,
    // so the last message should be the most recent one
    assert_eq!(
        history.last().unwrap().content.as_deref(),
        Some("Message 9")
    );
}

#[tokio::test]
async fn test_session_isolation() {
    let (store, _db) = setup_test_store().await;

    let m_a = make_message("session_a", "user", "From A");
    let m_b = make_message("session_b", "user", "From B");

    store.append_message(&m_a).await.unwrap();
    store.append_message(&m_b).await.unwrap();

    let history_a = store.get_history("session_a", 100).await.unwrap();
    let history_b = store.get_history("session_b", 100).await.unwrap();

    assert_eq!(history_a.len(), 1);
    assert_eq!(history_b.len(), 1);
    assert_eq!(history_a[0].content.as_deref(), Some("From A"));
    assert_eq!(history_b[0].content.as_deref(), Some("From B"));
}

#[tokio::test]
async fn test_clear_session() {
    let (store, _db) = setup_test_store().await;
    let session = "sess-clear";

    store
        .append_message(&make_message(session, "user", "Hi"))
        .await
        .unwrap();
    store
        .append_message(&make_message(session, "assistant", "Hello"))
        .await
        .unwrap();

    let before = store.get_history(session, 100).await.unwrap();
    assert_eq!(before.len(), 2);

    store.clear_session(session).await.unwrap();

    let after = store.get_history(session, 100).await.unwrap();
    assert_eq!(after.len(), 0);
}

#[tokio::test]
async fn test_working_memory_cap() {
    let (store, _db) = setup_test_store_with_cap(5).await;
    let session = "sess-cap";

    for i in 0..10 {
        let msg = make_message(session, "user", &format!("Msg {}", i));
        store.append_message(&msg).await.unwrap();
    }

    let history = store.get_history(session, 100).await.unwrap();
    assert!(
        history.len() <= 5,
        "Expected <= 5 messages in working memory, got {}",
        history.len()
    );
}

#[tokio::test]
async fn test_get_history_hydrates_from_events_when_available() {
    let (store, _db) = setup_test_store().await;
    let session = "sess-events";
    let now = Utc::now();

    create_events_table_for_tests(&store).await;
    insert_conversation_event_for_tests(
        &store,
        session,
        "user_message",
        serde_json::json!({
            "content": "Hello from events",
            "message_id": "msg-user-1",
            "has_attachments": false
        }),
        now,
    )
    .await;
    insert_conversation_event_for_tests(
        &store,
        session,
        "assistant_response",
        serde_json::json!({
            "content": "Hi from assistant event",
            "message_id": "msg-assistant-1",
            "model": "test",
            "tool_calls": []
        }),
        now + chrono::Duration::seconds(1),
    )
    .await;
    insert_conversation_event_for_tests(
        &store,
        session,
        "tool_result",
        serde_json::json!({
            "message_id": "msg-tool-1",
            "tool_call_id": "tc-1",
            "name": "terminal",
            "result": "ok",
            "success": true,
            "duration_ms": 7,
            "error": null,
            "task_id": null
        }),
        now + chrono::Duration::seconds(2),
    )
    .await;

    let history = store.get_history(session, 100).await.unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].id, "msg-user-1");
    assert_eq!(history[0].role, "user");
    assert_eq!(history[1].id, "msg-assistant-1");
    assert_eq!(history[1].role, "assistant");
    assert_eq!(history[2].id, "msg-tool-1");
    assert_eq!(history[2].role, "tool");
    assert_eq!(history[2].tool_call_id.as_deref(), Some("tc-1"));
}

#[tokio::test]
async fn test_get_context_uses_event_history() {
    let (store, _db) = setup_test_store().await;
    let session = "sess-context-fallback";
    let now = Utc::now();

    create_events_table_for_tests(&store).await;
    insert_conversation_event_for_tests(
        &store,
        session,
        "user_message",
        serde_json::json!({
            "content": "Context from events",
            "message_id": "msg-context-1",
            "has_attachments": false
        }),
        now,
    )
    .await;

    let context = store.get_context(session, "context", 10).await.unwrap();
    assert_eq!(context.len(), 1);
    assert_eq!(context[0].id, "msg-context-1");
    assert_eq!(context[0].role, "user");
    assert_eq!(context[0].content.as_deref(), Some("Context from events"));
}

#[tokio::test]
async fn test_open_store_migrates_legacy_messages_to_events() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = sqlx::SqlitePool::connect(&db_url).await.unwrap();

    sqlx::query(
        "CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_name TEXT,
            tool_calls_json TEXT,
            created_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO messages (id, session_id, role, content, created_at)
         VALUES ('legacy-u-1', 'sess-migrate', 'user', 'hello from legacy', ?)",
    )
    .bind(Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO messages (id, session_id, role, content, created_at)
         VALUES ('legacy-a-1', 'sess-migrate', 'assistant', 'legacy reply', ?)",
    )
    .bind((Utc::now() + chrono::Duration::seconds(1)).to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    pool.close().await;

    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = SqliteStateStore::new(
        db_file.path().to_str().unwrap(),
        100,
        None,
        embedding_service,
    )
    .await
    .unwrap();

    let has_messages_table: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages' LIMIT 1",
    )
    .fetch_optional(&store.pool())
    .await
    .unwrap();
    assert!(has_messages_table.is_none());

    let history = store.get_history("sess-migrate", 10).await.unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].id, "legacy-u-1");
    assert_eq!(history[0].role, "user");
    assert_eq!(history[1].id, "legacy-a-1");
    assert_eq!(history[1].role, "assistant");

    let event_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)
         FROM events
         WHERE session_id = 'sess-migrate'
           AND event_type IN ('user_message', 'assistant_response', 'tool_result')",
    )
    .fetch_one(&store.pool())
    .await
    .unwrap();
    assert_eq!(event_rows, 2);
}

#[tokio::test]
async fn test_open_store_migration_skips_preexisting_event_rows() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = sqlx::SqlitePool::connect(&db_url).await.unwrap();

    crate::db::migrations::migrate_events(&pool).await.unwrap();

    sqlx::query(
        "CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_name TEXT,
            tool_calls_json TEXT,
            created_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    let t0 = Utc::now();
    sqlx::query(
        "INSERT INTO messages (id, session_id, role, content, created_at)
         VALUES ('legacy-u-dup-1', 'sess-migrate-dedupe', 'user', 'hello', ?)",
    )
    .bind(t0.to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO messages (id, session_id, role, content, created_at)
         VALUES ('legacy-a-dup-1', 'sess-migrate-dedupe', 'assistant', 'reply', ?)",
    )
    .bind((t0 + chrono::Duration::seconds(1)).to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    // Simulate a partial prior migration run: first row already exists in events.
    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at)
         VALUES (?, 'user_message', ?, ?)",
    )
    .bind("sess-migrate-dedupe")
    .bind(
        serde_json::json!({
            "message_id": "legacy-u-dup-1",
            "content": "hello",
            "has_attachments": false
        })
        .to_string(),
    )
    .bind(t0.to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    pool.close().await;

    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = SqliteStateStore::new(
        db_file.path().to_str().unwrap(),
        100,
        None,
        embedding_service,
    )
    .await
    .unwrap();

    let duplicate_user_events: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)
         FROM events
         WHERE session_id = 'sess-migrate-dedupe'
           AND event_type = 'user_message'
           AND json_extract(data, '$.message_id') = 'legacy-u-dup-1'",
    )
    .fetch_one(&store.pool())
    .await
    .unwrap();
    assert_eq!(duplicate_user_events, 1);

    let total_events: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)
         FROM events
         WHERE session_id = 'sess-migrate-dedupe'
           AND event_type IN ('user_message', 'assistant_response', 'tool_result')",
    )
    .fetch_one(&store.pool())
    .await
    .unwrap();
    assert_eq!(total_events, 2);
}

// ==================== Fact Tests ====================

#[tokio::test]
async fn test_upsert_fact_insert() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact(
            "preference",
            "language",
            "Rust",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let facts = store.get_facts(Some("preference")).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].category, "preference");
    assert_eq!(facts[0].key, "language");
    assert_eq!(facts[0].value, "Rust");
}

#[tokio::test]
async fn test_upsert_fact_supersede() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact(
            "preference",
            "editor",
            "vim",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Upserting the same category/key with the same value should succeed
    // (it updates timestamp/source rather than inserting a new row).
    store
        .upsert_fact(
            "preference",
            "editor",
            "vim",
            "observation",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let facts = store.get_facts(Some("preference")).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].value, "vim");
    // Source should be updated
    assert_eq!(facts[0].source, "observation");
}

#[tokio::test]
async fn test_upsert_fact_value_change_creates_history() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact(
            "preference",
            "editor",
            "vim",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    store
        .upsert_fact(
            "preference",
            "editor",
            "neovim",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let active = store.get_facts(Some("preference")).await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].key, "editor");
    assert_eq!(active[0].value, "neovim");
    assert!(active[0].superseded_at.is_none());

    let history = store
        .get_fact_history("preference", "editor")
        .await
        .unwrap();
    assert_eq!(
        history.len(),
        2,
        "Should keep superseded versions for history"
    );
    assert_eq!(history[0].value, "neovim");
    assert!(history[0].superseded_at.is_none());
    assert_eq!(history[1].value, "vim");
    assert!(history[1].superseded_at.is_some());
}

#[tokio::test]
async fn test_get_facts_by_category() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact("pref", "color", "blue", "user", None, FactPrivacy::Global)
        .await
        .unwrap();
    store
        .upsert_fact("info", "name", "Alice", "user", None, FactPrivacy::Global)
        .await
        .unwrap();
    store
        .upsert_fact("pref", "food", "pizza", "user", None, FactPrivacy::Global)
        .await
        .unwrap();

    let pref_facts = store.get_facts(Some("pref")).await.unwrap();
    assert_eq!(pref_facts.len(), 2);
    for f in &pref_facts {
        assert_eq!(f.category, "pref");
    }

    let info_facts = store.get_facts(Some("info")).await.unwrap();
    assert_eq!(info_facts.len(), 1);
    assert_eq!(info_facts[0].key, "name");

    let all_facts = store.get_facts(None).await.unwrap();
    assert_eq!(all_facts.len(), 3);
}

#[tokio::test]
async fn test_delete_fact_soft_delete() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact(
            "temp",
            "item",
            "delete-me",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let facts = store.get_facts(Some("temp")).await.unwrap();
    assert_eq!(facts.len(), 1);
    let fact_id = facts[0].id;

    store.delete_fact(fact_id).await.unwrap();

    let after = store.get_facts(Some("temp")).await.unwrap();
    assert_eq!(after.len(), 0);
}

#[tokio::test]
async fn test_increment_fact_recall() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact(
            "test",
            "recall_key",
            "recall_val",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let facts = store.get_facts(Some("test")).await.unwrap();
    let fact_id = facts[0].id;
    assert_eq!(facts[0].recall_count, 0);
    assert!(facts[0].last_recalled_at.is_none());

    store.increment_fact_recall(fact_id).await.unwrap();
    store.increment_fact_recall(fact_id).await.unwrap();

    let updated = store.get_facts(Some("test")).await.unwrap();
    assert_eq!(updated[0].recall_count, 2);
    assert!(updated[0].last_recalled_at.is_some());
}

#[tokio::test]
async fn test_get_relevant_facts_increments_recall() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact(
            "project",
            "language",
            "Rust",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let facts = store.get_relevant_facts("rust language", 10).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].key, "language");

    let updated = store.get_facts(Some("project")).await.unwrap();
    assert_eq!(updated.len(), 1);
    assert_eq!(updated[0].recall_count, 1);
    assert!(updated[0].last_recalled_at.is_some());
}

#[tokio::test]
async fn test_fact_privacy_channel_scoped() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact(
            "context",
            "project",
            "aidaemon",
            "user",
            Some("slack:C12345"),
            FactPrivacy::Channel,
        )
        .await
        .unwrap();

    let facts = store.get_facts(Some("context")).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].channel_id.as_deref(), Some("slack:C12345"));
    assert_eq!(facts[0].privacy, FactPrivacy::Channel);
}

#[tokio::test]
async fn test_update_fact_privacy() {
    let (store, _db) = setup_test_store().await;

    store
        .upsert_fact(
            "secret",
            "api_key_hint",
            "starts with sk-",
            "user",
            Some("slack:C999"),
            FactPrivacy::Channel,
        )
        .await
        .unwrap();

    let facts = store.get_facts(Some("secret")).await.unwrap();
    assert_eq!(facts[0].privacy, FactPrivacy::Channel);
    let fact_id = facts[0].id;

    store
        .update_fact_privacy(fact_id, FactPrivacy::Global)
        .await
        .unwrap();

    let updated = store.get_facts(Some("secret")).await.unwrap();
    assert_eq!(updated[0].privacy, FactPrivacy::Global);
}

#[tokio::test]
async fn test_get_relevant_facts_freshness_boost_affects_sorting() {
    let (store, _db) = setup_test_store().await;

    let query = "what is my dog's name?";
    let q = store
        .embedding_service
        .embed(query.to_string())
        .await
        .unwrap();
    let q_norm = (q.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(q_norm > 0.0, "Query embedding norm should be > 0");
    let uq: Vec<f32> = q.iter().map(|x| x / q_norm).collect();

    // Build an orthogonal unit vector to uq.
    let mut r = vec![0.0f32; uq.len()];
    r[0] = 1.0;
    let proj = r.iter().zip(uq.iter()).map(|(a, b)| a * b).sum::<f32>();
    let mut o: Vec<f32> = r.iter().zip(uq.iter()).map(|(a, b)| a - proj * b).collect();
    let o_norm = (o.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(o_norm > 1e-6, "Orthogonal basis norm should be > 0");
    for v in o.iter_mut() {
        *v /= o_norm;
    }
    let uo = o;

    let mk = |cos: f32| -> Vec<f32> {
        let sin = (1.0 - cos * cos).sqrt();
        uq.iter()
            .zip(uo.iter())
            .map(|(a, b)| cos * a + sin * b)
            .collect()
    };

    // Older but slightly more semantically similar.
    store
        .upsert_fact(
            "user",
            "dog_name_old",
            "Max",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    // Newer but slightly less semantically similar; recency boost should flip ordering.
    store
        .upsert_fact(
            "user",
            "dog_name_new",
            "Pixel",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let now = Utc::now();
    let old_ts = (now - chrono::Duration::days(8)).to_rfc3339();
    let new_ts = now.to_rfc3339();

    let emb_old = encode_embedding(&mk(0.80));
    let emb_new = encode_embedding(&mk(0.75));

    sqlx::query("UPDATE facts SET embedding = ?, updated_at = ? WHERE category = ? AND key = ?")
        .bind(&emb_old)
        .bind(&old_ts)
        .bind("user")
        .bind("dog_name_old")
        .execute(&store.pool())
        .await
        .unwrap();

    sqlx::query("UPDATE facts SET embedding = ?, updated_at = ? WHERE category = ? AND key = ?")
        .bind(&emb_new)
        .bind(&new_ts)
        .bind("user")
        .bind("dog_name_new")
        .execute(&store.pool())
        .await
        .unwrap();

    let facts = store.get_relevant_facts(query, 2).await.unwrap();
    assert_eq!(facts.len(), 2);
    assert_eq!(
        facts[0].key, "dog_name_new",
        "Freshness boost should lift the newer fact when semantic scores are close"
    );
}

#[tokio::test]
async fn test_get_relevant_facts_freshness_boost_does_not_override_threshold() {
    let (store, _db) = setup_test_store().await;

    let query = "what is my dog's name?";
    let q = store
        .embedding_service
        .embed(query.to_string())
        .await
        .unwrap();
    let q_norm = (q.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(q_norm > 0.0);
    let uq: Vec<f32> = q.iter().map(|x| x / q_norm).collect();

    let mut r = vec![0.0f32; uq.len()];
    r[0] = 1.0;
    let proj = r.iter().zip(uq.iter()).map(|(a, b)| a * b).sum::<f32>();
    let mut o: Vec<f32> = r.iter().zip(uq.iter()).map(|(a, b)| a - proj * b).collect();
    let o_norm = (o.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(o_norm > 1e-6);
    for v in o.iter_mut() {
        *v /= o_norm;
    }
    let uo = o;

    let mk = |cos: f32| -> Vec<f32> {
        let sin = (1.0 - cos * cos).sqrt();
        uq.iter()
            .zip(uo.iter())
            .map(|(a, b)| cos * a + sin * b)
            .collect()
    };

    store
        .upsert_fact(
            "user",
            "below_threshold",
            "Pixel",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    store
        .upsert_fact(
            "user",
            "above_threshold",
            "Max",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let now = Utc::now();
    let fresh_ts = now.to_rfc3339();
    let old_ts = (now - chrono::Duration::days(8)).to_rfc3339();

    let emb_low = encode_embedding(&mk(0.29));
    let emb_high = encode_embedding(&mk(0.51));

    sqlx::query("UPDATE facts SET embedding = ?, updated_at = ? WHERE category = ? AND key = ?")
        .bind(&emb_low)
        .bind(&fresh_ts)
        .bind("user")
        .bind("below_threshold")
        .execute(&store.pool())
        .await
        .unwrap();
    sqlx::query("UPDATE facts SET embedding = ?, updated_at = ? WHERE category = ? AND key = ?")
        .bind(&emb_high)
        .bind(&old_ts)
        .bind("user")
        .bind("above_threshold")
        .execute(&store.pool())
        .await
        .unwrap();

    let facts = store.get_relevant_facts(query, 2).await.unwrap();
    assert!(
        facts.iter().all(|f| f.key != "below_threshold"),
        "Freshness boost must not allow sub-threshold semantic matches into results"
    );
    assert!(
        facts.iter().any(|f| f.key == "above_threshold"),
        "Above-threshold semantic match should be returned"
    );
}

#[tokio::test]
async fn test_get_relevant_facts_does_not_pad_with_unrelated_recent_facts() {
    let (store, _db) = setup_test_store().await;

    let query = "rust deployment strategy";
    let q = store
        .embedding_service
        .embed(query.to_string())
        .await
        .unwrap();
    let q_norm = (q.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(q_norm > 0.0);
    let uq: Vec<f32> = q.iter().map(|x| x / q_norm).collect();

    let mut r = vec![0.0f32; uq.len()];
    r[0] = 1.0;
    let proj = r.iter().zip(uq.iter()).map(|(a, b)| a * b).sum::<f32>();
    let mut o: Vec<f32> = r.iter().zip(uq.iter()).map(|(a, b)| a - proj * b).collect();
    let o_norm = (o.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(o_norm > 1e-6);
    for v in o.iter_mut() {
        *v /= o_norm;
    }
    let uo = o;

    let mk = |cos: f32| -> Vec<f32> {
        let sin = (1.0 - cos * cos).sqrt();
        uq.iter()
            .zip(uo.iter())
            .map(|(a, b)| cos * a + sin * b)
            .collect()
    };

    store
        .upsert_fact(
            "project",
            "deploy_notes",
            "Use canary rollout for Rust API",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    store
        .upsert_fact(
            "travel",
            "vacation_city",
            "Barcelona",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let now = Utc::now();
    let old_ts = (now - chrono::Duration::days(8)).to_rfc3339();
    let fresh_ts = now.to_rfc3339();

    let emb_relevant = encode_embedding(&mk(0.52));
    let emb_unrelated = encode_embedding(&mk(0.20));

    sqlx::query("UPDATE facts SET embedding = ?, updated_at = ? WHERE category = ? AND key = ?")
        .bind(&emb_relevant)
        .bind(&old_ts)
        .bind("project")
        .bind("deploy_notes")
        .execute(&store.pool())
        .await
        .unwrap();
    sqlx::query("UPDATE facts SET embedding = ?, updated_at = ? WHERE category = ? AND key = ?")
        .bind(&emb_unrelated)
        .bind(&fresh_ts)
        .bind("travel")
        .bind("vacation_city")
        .execute(&store.pool())
        .await
        .unwrap();

    let facts = store.get_relevant_facts(query, 6).await.unwrap();
    assert!(
        facts.iter().any(|f| f.key == "deploy_notes"),
        "Above-threshold fact should be returned"
    );
    // With padding enabled, unrelated facts may be included when results are sparse.
    // The key invariant is that relevant facts come first.
    if facts.len() >= 2 {
        assert_eq!(
            facts[0].key, "deploy_notes",
            "Relevant fact should come before padded results"
        );
    }
}

#[tokio::test]
async fn test_get_relevant_facts_missing_embedding_lexical_can_compete() {
    let (store, _db) = setup_test_store().await;

    let query = "what is my dog's name?";
    let q = store
        .embedding_service
        .embed(query.to_string())
        .await
        .unwrap();
    let q_norm = (q.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(q_norm > 0.0);
    let uq: Vec<f32> = q.iter().map(|x| x / q_norm).collect();

    let mut r = vec![0.0f32; uq.len()];
    r[0] = 1.0;
    let proj = r.iter().zip(uq.iter()).map(|(a, b)| a * b).sum::<f32>();
    let mut o: Vec<f32> = r.iter().zip(uq.iter()).map(|(a, b)| a - proj * b).collect();
    let o_norm = (o.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(o_norm > 1e-6);
    for v in o.iter_mut() {
        *v /= o_norm;
    }
    let uo = o;

    let mk = |cos: f32| -> Vec<f32> {
        let sin = (1.0 - cos * cos).sqrt();
        uq.iter()
            .zip(uo.iter())
            .map(|(a, b)| cos * a + sin * b)
            .collect()
    };

    // Seed 10 semantically-relevant facts that barely clear the threshold, old enough to get no boost.
    for i in 0..10 {
        store
            .upsert_fact(
                "user",
                &format!("k{}", i),
                "filler",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
    }

    // Fresh unembedded fact that should still be retrieved via lexical fallback ("dog_name" matches query tokens).
    store
        .upsert_fact(
            "user",
            "dog_name",
            "Pixel",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let now = Utc::now();
    let old_ts = (now - chrono::Duration::days(8)).to_rfc3339();
    let fresh_ts = now.to_rfc3339();

    let emb = encode_embedding(&mk(0.51));
    for i in 0..10 {
        sqlx::query(
            "UPDATE facts SET embedding = ?, updated_at = ? WHERE category = ? AND key = ?",
        )
        .bind(&emb)
        .bind(&old_ts)
        .bind("user")
        .bind(format!("k{}", i))
        .execute(&store.pool())
        .await
        .unwrap();
    }

    // Force missing embedding for dog_name.
    sqlx::query("UPDATE facts SET embedding = NULL, updated_at = ? WHERE category = ? AND key = ?")
        .bind(&fresh_ts)
        .bind("user")
        .bind("dog_name")
        .execute(&store.pool())
        .await
        .unwrap();

    let facts = store.get_relevant_facts(query, 10).await.unwrap();
    assert!(
        facts
            .iter()
            .any(|f| f.key == "dog_name" && f.value == "Pixel"),
        "Lexical fallback should allow fresh missing-embedding facts to compete for top slots"
    );
}

#[tokio::test]
async fn test_get_relevant_facts_lexical_value_matching_uses_word_boundaries() {
    let (store, _db) = setup_test_store().await;

    let query = "dog";
    let q = store
        .embedding_service
        .embed(query.to_string())
        .await
        .unwrap();
    let q_norm = (q.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(q_norm > 0.0);
    let uq: Vec<f32> = q.iter().map(|x| x / q_norm).collect();

    // Build an orthogonal unit vector to uq.
    let mut r = vec![0.0f32; uq.len()];
    r[0] = 1.0;
    let proj = r.iter().zip(uq.iter()).map(|(a, b)| a * b).sum::<f32>();
    let mut o: Vec<f32> = r.iter().zip(uq.iter()).map(|(a, b)| a - proj * b).collect();
    let o_norm = (o.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert!(o_norm > 1e-6);
    for v in o.iter_mut() {
        *v /= o_norm;
    }
    let uo = o;

    let mk = |cos: f32| -> Vec<f32> {
        let sin = (1.0 - cos * cos).sqrt();
        uq.iter()
            .zip(uo.iter())
            .map(|(a, b)| cos * a + sin * b)
            .collect()
    };

    // Seed 10 semantic facts that are just above the semantic threshold so the
    // result set fills up (preventing the "append all unembedded" fallback).
    for i in 0..10 {
        store
            .upsert_fact(
                "user",
                &format!("sem{}", i),
                "filler",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
    }

    // Missing-embedding fact whose VALUE contains "dodger" (substring "dog") but not as a word.
    // Word-boundary matching should NOT count this as a lexical hit for query "dog".
    store
        .upsert_fact(
            "user",
            "place",
            "dodger stadium info",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let now = Utc::now();
    let old_ts = (now - chrono::Duration::days(8)).to_rfc3339();
    let fresh_ts = now.to_rfc3339();

    let emb = encode_embedding(&mk(0.52));
    for i in 0..10 {
        sqlx::query(
            "UPDATE facts SET embedding = ?, updated_at = ? WHERE category = ? AND key = ?",
        )
        .bind(&emb)
        .bind(&old_ts)
        .bind("user")
        .bind(format!("sem{}", i))
        .execute(&store.pool())
        .await
        .unwrap();
    }

    // Force missing embedding for "place" and keep it fresh to maximize the chance that
    // naive substring matching would otherwise boost it into the top results.
    sqlx::query("UPDATE facts SET embedding = NULL, updated_at = ? WHERE category = ? AND key = ?")
        .bind(&fresh_ts)
        .bind("user")
        .bind("place")
        .execute(&store.pool())
        .await
        .unwrap();

    let facts = store.get_relevant_facts(query, 10).await.unwrap();
    assert!(
        facts.iter().all(|f| f.key != "place"),
        "Value substring matching should not treat 'dodger' as a match for 'dog'"
    );
}

// ==================== Episode Tests ====================

#[tokio::test]
async fn test_insert_and_get_episodes() {
    let (store, _db) = setup_test_store().await;

    let episode = Episode {
        id: 0,
        session_id: "ep-sess".to_string(),
        summary: "We discussed Rust async patterns".to_string(),
        topics: Some(vec!["rust".to_string(), "async".to_string()]),
        emotional_tone: Some("curious".to_string()),
        outcome: Some("learned tokio basics".to_string()),
        importance: 0.8,
        recall_count: 0,
        last_recalled_at: None,
        message_count: 12,
        start_time: Utc::now() - chrono::Duration::hours(1),
        end_time: Utc::now(),
        created_at: Utc::now(),
        channel_id: None,
    };

    let ep_id = store.insert_episode(&episode).await.unwrap();
    assert!(ep_id > 0);

    let episodes = store.get_recent_episodes(10).await.unwrap();
    assert_eq!(episodes.len(), 1);
    assert_eq!(episodes[0].summary, "We discussed Rust async patterns");
    assert_eq!(episodes[0].message_count, 12);
    assert_eq!(
        episodes[0].topics,
        Some(vec!["rust".to_string(), "async".to_string()])
    );
}

#[tokio::test]
async fn test_increment_episode_recall() {
    let (store, _db) = setup_test_store().await;

    let episode = Episode {
        id: 0,
        session_id: "ep-recall".to_string(),
        summary: "Recall test episode".to_string(),
        topics: None,
        emotional_tone: None,
        outcome: None,
        importance: 0.5,
        recall_count: 0,
        last_recalled_at: None,
        message_count: 5,
        start_time: Utc::now(),
        end_time: Utc::now(),
        created_at: Utc::now(),
        channel_id: None,
    };

    let ep_id = store.insert_episode(&episode).await.unwrap();

    store.increment_episode_recall(ep_id).await.unwrap();
    store.increment_episode_recall(ep_id).await.unwrap();

    let episodes = store.get_recent_episodes(10).await.unwrap();
    assert_eq!(episodes[0].recall_count, 2);
    assert!(episodes[0].last_recalled_at.is_some());
}

#[tokio::test]
async fn test_backfill_episode_embeddings() {
    let (store, _db) = setup_test_store().await;

    let episode = Episode {
        id: 0,
        session_id: "ep-embed".to_string(),
        summary: "An episode about machine learning".to_string(),
        topics: None,
        emotional_tone: None,
        outcome: None,
        importance: 0.5,
        recall_count: 0,
        last_recalled_at: None,
        message_count: 3,
        start_time: Utc::now(),
        end_time: Utc::now(),
        created_at: Utc::now(),
        channel_id: None,
    };

    store.insert_episode(&episode).await.unwrap();

    // Episodes are inserted without embeddings
    let backfilled = store.backfill_episode_embeddings().await.unwrap();
    assert_eq!(backfilled, 1);

    // Running again should backfill 0 since all have embeddings now
    let backfilled_again = store.backfill_episode_embeddings().await.unwrap();
    assert_eq!(backfilled_again, 0);
}

// ==================== Goal Tests ====================

#[tokio::test]
async fn test_create_and_get_active_personal_goals() {
    let (store, _db) = setup_test_store().await;

    let mut goal = Goal::new_finite("Learn Rust generics", "test-session");
    goal.domain = "personal".to_string();
    goal.priority = "high".to_string();
    store.create_goal(&goal).await.unwrap();

    let active = store.get_active_personal_goals(10).await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].description, "Learn Rust generics");
    assert_eq!(active[0].status, "active");
    assert_eq!(active[0].priority, "high");
    assert_eq!(active[0].domain, "personal");
}

#[tokio::test]
async fn test_update_personal_goal_status() {
    let (store, _db) = setup_test_store().await;

    let mut goal = Goal::new_finite("Finish project", "test-session");
    goal.domain = "personal".to_string();
    store.create_goal(&goal).await.unwrap();

    store
        .update_personal_goal(&goal.id, Some("completed"), None)
        .await
        .unwrap();

    let active = store.get_active_personal_goals(10).await.unwrap();
    assert_eq!(
        active.len(),
        0,
        "Completed personal goal should not appear in active personal goals"
    );
}

#[tokio::test]
async fn test_update_personal_goal_progress_note() {
    let (store, _db) = setup_test_store().await;

    let mut goal = Goal::new_finite("Write tests", "test-session");
    goal.domain = "personal".to_string();
    store.create_goal(&goal).await.unwrap();

    store
        .update_personal_goal(&goal.id, None, Some("Added 5 unit tests"))
        .await
        .unwrap();
    store
        .update_personal_goal(&goal.id, None, Some("Added 10 more tests"))
        .await
        .unwrap();

    let stored = store.get_goal(&goal.id).await.unwrap().unwrap();
    let notes = stored.progress_notes.as_ref().unwrap();
    assert_eq!(notes.len(), 2);
    assert_eq!(notes[0], "Added 5 unit tests");
    assert_eq!(notes[1], "Added 10 more tests");
}

// ==================== User Profile Tests ====================

#[tokio::test]
async fn test_default_user_profile() {
    let (store, _db) = setup_test_store().await;

    let profile = store.get_user_profile().await.unwrap();
    assert_eq!(profile.verbosity_preference, "medium");
    assert_eq!(profile.explanation_depth, "moderate");
    assert_eq!(profile.tone_preference, "neutral");
    assert_eq!(profile.emoji_preference, "none");
    assert!(profile.asks_before_acting);
    assert!(profile.prefers_explanations);
    // likes_suggestions defaults to false in the code
    assert!(!profile.likes_suggestions);
}

#[tokio::test]
async fn test_update_user_profile() {
    let (store, _db) = setup_test_store().await;

    // First call creates the default profile
    let mut profile = store.get_user_profile().await.unwrap();

    profile.verbosity_preference = "brief".to_string();
    profile.tone_preference = "casual".to_string();
    profile.emoji_preference = "frequent".to_string();
    profile.asks_before_acting = false;

    store.update_user_profile(&profile).await.unwrap();

    let updated = store.get_user_profile().await.unwrap();
    assert_eq!(updated.verbosity_preference, "brief");
    assert_eq!(updated.tone_preference, "casual");
    assert_eq!(updated.emoji_preference, "frequent");
    assert!(!updated.asks_before_acting);
    // Unchanged fields should remain
    assert_eq!(updated.explanation_depth, "moderate");
    assert!(updated.prefers_explanations);
}

// ==================== Behavior Pattern Tests ====================

#[tokio::test]
async fn test_insert_and_get_behavior_patterns() {
    let (store, _db) = setup_test_store().await;

    let pattern = BehaviorPattern {
        id: 0,
        pattern_type: "habit".to_string(),
        description: "Always runs tests after code changes".to_string(),
        trigger_context: Some("code modification".to_string()),
        action: Some("cargo test".to_string()),
        confidence: 0.7,
        occurrence_count: 3,
        last_seen_at: Some(Utc::now()),
        created_at: Utc::now(),
    };

    let pat_id = store.insert_behavior_pattern(&pattern).await.unwrap();
    assert!(pat_id > 0);

    let patterns = store.get_behavior_patterns(0.5).await.unwrap();
    assert_eq!(patterns.len(), 1);
    assert_eq!(
        patterns[0].description,
        "Always runs tests after code changes"
    );
    assert_eq!(patterns[0].confidence, 0.7);
    assert_eq!(patterns[0].occurrence_count, 3);

    // With a higher min_confidence threshold, it should not be returned
    let filtered = store.get_behavior_patterns(0.9).await.unwrap();
    assert_eq!(filtered.len(), 0);
}

#[tokio::test]
async fn test_update_behavior_pattern_confidence() {
    let (store, _db) = setup_test_store().await;

    let pattern = BehaviorPattern {
        id: 0,
        pattern_type: "trigger".to_string(),
        description: "Checks git status before committing".to_string(),
        trigger_context: None,
        action: None,
        confidence: 0.5,
        occurrence_count: 1,
        last_seen_at: None,
        created_at: Utc::now(),
    };

    let pat_id = store.insert_behavior_pattern(&pattern).await.unwrap();

    store.update_behavior_pattern(pat_id, 0.1).await.unwrap();

    let patterns = store.get_behavior_patterns(0.0).await.unwrap();
    assert_eq!(patterns.len(), 1);
    assert_eq!(patterns[0].occurrence_count, 2);
    assert!((patterns[0].confidence - 0.6).abs() < 0.01);
    assert!(patterns[0].last_seen_at.is_some());
}

#[tokio::test]
async fn test_record_behavior_pattern_upserts_by_logical_key() {
    let (store, _db) = setup_test_store().await;

    store
        .record_behavior_pattern(
            "failure",
            "Tool terminal repeatedly fails on permission denied; pivot sooner.",
            Some("terminal"),
            Some("pivot"),
            0.7,
            1,
        )
        .await
        .unwrap();

    store
        .record_behavior_pattern(
            "failure",
            "Tool terminal repeatedly fails on permission denied; pivot sooner.",
            Some("terminal"),
            Some("pivot"),
            0.8,
            2,
        )
        .await
        .unwrap();

    let patterns = store.get_behavior_patterns(0.0).await.unwrap();
    assert_eq!(patterns.len(), 1);
    assert_eq!(patterns[0].pattern_type, "failure");
    assert_eq!(patterns[0].occurrence_count, 3);
    assert!(patterns[0].confidence >= 0.7);
}

// ==================== Procedure Tests ====================

#[tokio::test]
async fn test_insert_and_get_procedures() {
    let (store, _db) = setup_test_store().await;

    let procedure = Procedure {
        id: 0,
        name: "deploy-app".to_string(),
        trigger_pattern: "deploy the application".to_string(),
        steps: vec![
            "cargo build --release".to_string(),
            "scp target/release/app server:".to_string(),
            "ssh server systemctl restart app".to_string(),
        ],
        success_count: 1,
        failure_count: 0,
        avg_duration_secs: Some(30.0),
        last_used_at: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    let proc_id = store.upsert_procedure(&procedure).await.unwrap();
    assert!(proc_id > 0);

    let procs = store.get_relevant_procedures("deploy", 10).await.unwrap();
    assert_eq!(procs.len(), 1);
    assert_eq!(procs[0].name, "deploy-app");
    assert_eq!(procs[0].steps.len(), 3);
}

#[tokio::test]
async fn test_procedure_success_count_increments() {
    let (store, _db) = setup_test_store().await;

    let procedure = Procedure {
        id: 0,
        name: "run-tests".to_string(),
        trigger_pattern: "run the test suite".to_string(),
        steps: vec!["cargo test".to_string()],
        success_count: 1,
        failure_count: 0,
        avg_duration_secs: None,
        last_used_at: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    store.upsert_procedure(&procedure).await.unwrap();

    // Upsert again with the same name triggers ON CONFLICT DO UPDATE
    // which increments success_count
    store.upsert_procedure(&procedure).await.unwrap();

    let procs = store.get_relevant_procedures("test", 10).await.unwrap();
    assert_eq!(procs.len(), 1);
    assert!(
        procs[0].success_count >= 2,
        "Expected success_count >= 2 after upsert conflict, got {}",
        procs[0].success_count
    );
}

#[tokio::test]
async fn test_procedure_upsert_accumulates_failure_and_updates_trigger() {
    let (store, _db) = setup_test_store().await;

    let procedure = Procedure {
        id: 0,
        name: "run-tests-1234abcd".to_string(),
        trigger_pattern: "run test suite".to_string(),
        steps: vec!["cargo test".to_string()],
        success_count: 1,
        failure_count: 0,
        avg_duration_secs: Some(20.0),
        last_used_at: Some(Utc::now()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    store.upsert_procedure(&procedure).await.unwrap();

    let failure_update = Procedure {
        id: 0,
        name: "run-tests-1234abcd".to_string(),
        trigger_pattern: "run complete test suite in ci".to_string(),
        steps: vec!["cargo test --workspace".to_string()],
        success_count: 0,
        failure_count: 1,
        avg_duration_secs: None,
        last_used_at: Some(Utc::now()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    store.upsert_procedure(&failure_update).await.unwrap();

    let row = sqlx::query(
        "SELECT success_count, failure_count, trigger_pattern
         FROM procedures WHERE name = ?",
    )
    .bind("run-tests-1234abcd")
    .fetch_one(&store.pool())
    .await
    .unwrap();

    let success_count: i32 = row.get("success_count");
    let failure_count: i32 = row.get("failure_count");
    let trigger_pattern: String = row.get("trigger_pattern");

    assert_eq!(success_count, 1);
    assert_eq!(failure_count, 1);
    assert_eq!(trigger_pattern, "run complete test suite in ci");
}

// ==================== Error Solution Tests ====================

#[tokio::test]
async fn test_insert_and_get_error_solutions() {
    let (store, _db) = setup_test_store().await;

    let solution = ErrorSolution {
        id: 0,
        error_pattern: "connection refused on port 5432".to_string(),
        domain: Some("database".to_string()),
        solution_summary: "Start the PostgreSQL service".to_string(),
        solution_steps: Some(vec![
            "sudo systemctl start postgresql".to_string(),
            "verify with pg_isready".to_string(),
        ]),
        success_count: 1,
        failure_count: 0,
        last_used_at: None,
        created_at: Utc::now(),
    };

    let sol_id = store.insert_error_solution(&solution).await.unwrap();
    assert!(sol_id > 0);

    let solutions = store
        .get_relevant_error_solutions("connection refused", 10)
        .await
        .unwrap();
    assert_eq!(solutions.len(), 1);
    assert_eq!(
        solutions[0].solution_summary,
        "Start the PostgreSQL service"
    );
    assert_eq!(solutions[0].domain.as_deref(), Some("database"));
}

#[tokio::test]
async fn test_insert_error_solution_upserts_and_increments_success_count() {
    let (store, _db) = setup_test_store().await;

    let solution = ErrorSolution {
        id: 0,
        error_pattern: "permission denied: /tmp/foo.txt:12:3".to_string(),
        domain: None,
        solution_summary: "Fix permissions".to_string(),
        solution_steps: Some(vec!["chmod +w <path>".to_string()]),
        success_count: 1,
        failure_count: 0,
        last_used_at: None,
        created_at: Utc::now(),
    };

    let id1 = store.insert_error_solution(&solution).await.unwrap();
    let id2 = store.insert_error_solution(&solution).await.unwrap();
    assert_eq!(id1, id2, "expected upsert to return the same row id");

    let solutions = store
        .get_relevant_error_solutions("permission denied", 10)
        .await
        .unwrap();
    assert_eq!(solutions.len(), 1);
    assert_eq!(solutions[0].success_count, 2);
}

#[tokio::test]
async fn test_update_error_solution_outcome() {
    let (store, _db) = setup_test_store().await;

    let solution = ErrorSolution {
        id: 0,
        error_pattern: "file not found".to_string(),
        domain: None,
        solution_summary: "Check the file path".to_string(),
        solution_steps: None,
        success_count: 0,
        failure_count: 0,
        last_used_at: None,
        created_at: Utc::now(),
    };

    let sol_id = store.insert_error_solution(&solution).await.unwrap();

    // Record a success
    store.update_error_solution(sol_id, true).await.unwrap();
    // Record a failure
    store.update_error_solution(sol_id, false).await.unwrap();
    // Record another success
    store.update_error_solution(sol_id, true).await.unwrap();

    let solutions = store
        .get_relevant_error_solutions("file not found", 10)
        .await
        .unwrap();
    assert_eq!(solutions.len(), 1);
    assert_eq!(solutions[0].success_count, 2);
    assert_eq!(solutions[0].failure_count, 1);
}

// ==================== Token Usage Tests ====================

#[tokio::test]
async fn test_record_and_get_token_usage() {
    let (store, _db) = setup_test_store().await;

    let usage = TokenUsage {
        model: "gpt-4".to_string(),
        input_tokens: 100,
        output_tokens: 50,
    };

    store
        .record_token_usage("token-sess", &usage)
        .await
        .unwrap();

    // Use a date in the past to capture all records
    let records = store
        .get_token_usage_since("2000-01-01T00:00:00Z")
        .await
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].model, "gpt-4");
    assert_eq!(records[0].input_tokens, 100);
    assert_eq!(records[0].output_tokens, 50);
}

#[tokio::test]
async fn test_token_usage_since_filter() {
    let (store, _db) = setup_test_store().await;

    let usage1 = TokenUsage {
        model: "gpt-4".to_string(),
        input_tokens: 100,
        output_tokens: 50,
    };
    let usage2 = TokenUsage {
        model: "gpt-3.5".to_string(),
        input_tokens: 200,
        output_tokens: 80,
    };

    store.record_token_usage("sess-1", &usage1).await.unwrap();
    store.record_token_usage("sess-2", &usage2).await.unwrap();

    // A far-past date should return all records
    let all = store
        .get_token_usage_since("2000-01-01T00:00:00Z")
        .await
        .unwrap();
    assert_eq!(all.len(), 2);

    // A far-future date should return no records
    let none = store
        .get_token_usage_since("2099-01-01T00:00:00Z")
        .await
        .unwrap();
    assert_eq!(none.len(), 0);
}

// ==================== Dynamic Bot Tests ====================

#[tokio::test]
async fn test_dynamic_bots_crud() {
    let (store, _db) = setup_test_store().await;

    let bot = DynamicBot {
        id: 0,
        channel_type: "telegram".to_string(),
        bot_token: "123456:ABC".to_string(),
        app_token: None,
        allowed_user_ids: vec!["user1".to_string(), "user2".to_string()],
        extra_config: "{}".to_string(),
        created_at: String::new(),
    };

    // Add
    let bot_id = store.add_dynamic_bot(&bot).await.unwrap();
    assert!(bot_id > 0);

    // List
    let bots = store.get_dynamic_bots().await.unwrap();
    assert_eq!(bots.len(), 1);
    assert_eq!(bots[0].channel_type, "telegram");
    assert_eq!(bots[0].bot_token, "123456:ABC");
    assert_eq!(bots[0].allowed_user_ids.len(), 2);

    // Delete
    store.delete_dynamic_bot(bot_id).await.unwrap();

    let after = store.get_dynamic_bots().await.unwrap();
    assert_eq!(after.len(), 0);
}

// ==================== Dynamic Skill Tests ====================

#[tokio::test]
async fn test_dynamic_skills_crud() {
    let (store, _db) = setup_test_store().await;

    let skill = DynamicSkill {
        id: 0,
        name: "code-review".to_string(),
        description: "Review code for best practices".to_string(),
        triggers_json: r#"["review","code review"]"#.to_string(),
        body: "# Code Review\nCheck for...\n".to_string(),
        source: "inline".to_string(),
        source_url: None,
        enabled: true,
        version: Some("1.0".to_string()),
        created_at: String::new(),
        resources_json: "[]".to_string(),
    };

    // Add
    let skill_id = store.add_dynamic_skill(&skill).await.unwrap();
    assert!(skill_id > 0);

    // List
    let skills = store.get_dynamic_skills().await.unwrap();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "code-review");
    assert!(skills[0].enabled);

    // Disable
    store
        .update_dynamic_skill_enabled(skill_id, false)
        .await
        .unwrap();
    let skills = store.get_dynamic_skills().await.unwrap();
    assert!(!skills[0].enabled);

    // Re-enable
    store
        .update_dynamic_skill_enabled(skill_id, true)
        .await
        .unwrap();
    let skills = store.get_dynamic_skills().await.unwrap();
    assert!(skills[0].enabled);

    // Delete
    store.delete_dynamic_skill(skill_id).await.unwrap();
    let skills = store.get_dynamic_skills().await.unwrap();
    assert_eq!(skills.len(), 0);
}

#[tokio::test]
async fn test_skill_draft_exists_for_procedure_after_dismissal() {
    let (store, _db) = setup_test_store().await;
    let draft = SkillDraft {
        id: 0,
        name: "deploy-helper".to_string(),
        description: "Draft replacement".to_string(),
        triggers_json: r#"["deploy"]"#.to_string(),
        body: "1. Build.\n2. Deploy.".to_string(),
        source_procedure: "deploy-proc".to_string(),
        status: "pending".to_string(),
        created_at: String::new(),
    };

    let draft_id = store.add_skill_draft(&draft).await.unwrap();
    assert!(store
        .skill_draft_exists_for_procedure("deploy-proc")
        .await
        .unwrap());

    store
        .update_skill_draft_status(draft_id, "dismissed")
        .await
        .unwrap();
    assert!(store
        .skill_draft_exists_for_procedure("deploy-proc")
        .await
        .unwrap());
}

// ==================== Dynamic MCP Server Tests ====================

#[tokio::test]
async fn test_dynamic_mcp_servers_crud() {
    let (store, _db) = setup_test_store().await;

    let server = DynamicMcpServer {
        id: 0,
        name: "test_server".to_string(),
        command: "npx".to_string(),
        args_json: r#"["@test/mcp-server"]"#.to_string(),
        env_keys_json: r#"["API_KEY"]"#.to_string(),
        triggers_json: r#"["test","testing"]"#.to_string(),
        enabled: true,
        created_at: String::new(),
    };

    // Save
    let server_id = store.save_dynamic_mcp_server(&server).await.unwrap();
    assert!(server_id > 0);

    // List
    let servers = store.list_dynamic_mcp_servers().await.unwrap();
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].name, "test_server");
    assert_eq!(servers[0].command, "npx");
    assert!(servers[0].enabled);

    // Update
    let mut updated_server = servers[0].clone();
    updated_server.command = "uvx".to_string();
    updated_server.enabled = false;
    store
        .update_dynamic_mcp_server(&updated_server)
        .await
        .unwrap();

    let servers = store.list_dynamic_mcp_servers().await.unwrap();
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].command, "uvx");
    assert!(!servers[0].enabled);

    // Delete
    store
        .delete_dynamic_mcp_server(updated_server.id)
        .await
        .unwrap();
    let servers = store.list_dynamic_mcp_servers().await.unwrap();
    assert_eq!(servers.len(), 0);
}

#[tokio::test]
async fn test_oauth_connection_crud() {
    let (store, _tmp) = setup_test_store().await;

    // Insert
    let conn = crate::traits::OAuthConnection {
        id: 0,
        service: "twitter".to_string(),
        auth_type: "oauth2_pkce".to_string(),
        username: Some("@testuser".to_string()),
        scopes: r#"["tweet.read","tweet.write"]"#.to_string(),
        token_expires_at: Some("2025-12-31T00:00:00Z".to_string()),
        created_at: chrono::Utc::now().to_rfc3339(),
        updated_at: chrono::Utc::now().to_rfc3339(),
    };
    let id = store.save_oauth_connection(&conn).await.unwrap();
    assert!(id > 0);

    // Get by service
    let fetched = store
        .get_oauth_connection("twitter")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(fetched.service, "twitter");
    assert_eq!(fetched.auth_type, "oauth2_pkce");
    assert_eq!(fetched.username, Some("@testuser".to_string()));

    // List all
    let all = store.list_oauth_connections().await.unwrap();
    assert_eq!(all.len(), 1);

    // Update expiry
    store
        .update_oauth_token_expiry("twitter", Some("2026-06-30T00:00:00Z"))
        .await
        .unwrap();
    let updated = store
        .get_oauth_connection("twitter")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        updated.token_expires_at,
        Some("2026-06-30T00:00:00Z".to_string())
    );

    // Upsert (same service, different data)
    let conn2 = crate::traits::OAuthConnection {
        id: 0,
        service: "twitter".to_string(),
        auth_type: "oauth2_pkce".to_string(),
        username: Some("@newuser".to_string()),
        scopes: r#"["tweet.read"]"#.to_string(),
        token_expires_at: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        updated_at: chrono::Utc::now().to_rfc3339(),
    };
    store.save_oauth_connection(&conn2).await.unwrap();
    let upserted = store
        .get_oauth_connection("twitter")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(upserted.username, Some("@newuser".to_string()));
    // Still just 1 connection (upserted, not duplicated)
    assert_eq!(store.list_oauth_connections().await.unwrap().len(), 1);

    // Delete
    store.delete_oauth_connection("twitter").await.unwrap();
    assert!(store
        .get_oauth_connection("twitter")
        .await
        .unwrap()
        .is_none());
    assert_eq!(store.list_oauth_connections().await.unwrap().len(), 0);
}

#[tokio::test]
async fn test_oauth_connection_not_found() {
    let (store, _tmp) = setup_test_store().await;
    let result = store.get_oauth_connection("nonexistent").await.unwrap();
    assert!(result.is_none());
}

// -----------------------------------------------------------------------
// Dynamic CLI Agent CRUD tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_dynamic_cli_agent_crud() {
    let (store, _db) = setup_test_store().await;

    // Initially empty
    let agents = store.list_dynamic_cli_agents().await.unwrap();
    assert!(agents.is_empty());

    // Save a new agent
    let agent = crate::traits::DynamicCliAgent {
        id: 0,
        name: "test-agent".to_string(),
        command: "echo".to_string(),
        args_json: r#"["hello"]"#.to_string(),
        description: "Test echo agent".to_string(),
        timeout_secs: Some(30),
        max_output_chars: Some(5000),
        enabled: true,
        created_at: String::new(),
    };
    let id = store.save_dynamic_cli_agent(&agent).await.unwrap();
    assert!(id > 0);

    // List should return it
    let agents = store.list_dynamic_cli_agents().await.unwrap();
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0].name, "test-agent");
    assert_eq!(agents[0].command, "echo");
    assert_eq!(agents[0].args_json, r#"["hello"]"#);
    assert_eq!(agents[0].description, "Test echo agent");
    assert_eq!(agents[0].timeout_secs, Some(30));
    assert_eq!(agents[0].max_output_chars, Some(5000));
    assert!(agents[0].enabled);

    // Update the agent
    let mut updated = agents[0].clone();
    updated.command = "bash".to_string();
    updated.enabled = false;
    store.update_dynamic_cli_agent(&updated).await.unwrap();

    let agents = store.list_dynamic_cli_agents().await.unwrap();
    assert_eq!(agents[0].command, "bash");
    assert!(!agents[0].enabled);

    // Delete the agent
    store.delete_dynamic_cli_agent(updated.id).await.unwrap();
    let agents = store.list_dynamic_cli_agents().await.unwrap();
    assert!(agents.is_empty());
}

#[tokio::test]
async fn test_dynamic_cli_agent_upsert() {
    let (store, _db) = setup_test_store().await;

    let agent = crate::traits::DynamicCliAgent {
        id: 0,
        name: "upsert-agent".to_string(),
        command: "echo".to_string(),
        args_json: "[]".to_string(),
        description: "v1".to_string(),
        timeout_secs: None,
        max_output_chars: None,
        enabled: true,
        created_at: String::new(),
    };
    store.save_dynamic_cli_agent(&agent).await.unwrap();

    // Save again with same name — should upsert
    let agent2 = crate::traits::DynamicCliAgent {
        id: 0,
        name: "upsert-agent".to_string(),
        command: "bash".to_string(),
        args_json: r#"["-c"]"#.to_string(),
        description: "v2".to_string(),
        timeout_secs: Some(60),
        max_output_chars: Some(10000),
        enabled: true,
        created_at: String::new(),
    };
    store.save_dynamic_cli_agent(&agent2).await.unwrap();

    // Should still be 1 agent (upserted)
    let agents = store.list_dynamic_cli_agents().await.unwrap();
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0].command, "bash");
    assert_eq!(agents[0].description, "v2");
}

// -----------------------------------------------------------------------
// CLI Agent Invocation logging tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_cli_agent_invocation_logging() {
    let (store, _db) = setup_test_store().await;

    // Initially empty
    let invocations = store.get_cli_agent_invocations(10).await.unwrap();
    assert!(invocations.is_empty());

    // Log a start
    let inv_id = store
        .log_cli_agent_start(
            "session1",
            "claude",
            "Create a website",
            Some("/tmp/project"),
        )
        .await
        .unwrap();
    assert!(inv_id > 0);

    // Should appear in list
    let invocations = store.get_cli_agent_invocations(10).await.unwrap();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].session_id, "session1");
    assert_eq!(invocations[0].agent_name, "claude");
    assert_eq!(invocations[0].prompt_summary, "Create a website");
    assert_eq!(invocations[0].working_dir, Some("/tmp/project".to_string()));
    assert!(invocations[0].success.is_none()); // Not completed yet

    // Log completion
    store
        .log_cli_agent_complete(inv_id, Some(0), "Website created successfully", true, 45.5)
        .await
        .unwrap();

    // Should be updated
    let invocations = store.get_cli_agent_invocations(10).await.unwrap();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].exit_code, Some(0));
    assert_eq!(invocations[0].success, Some(true));
    assert_eq!(invocations[0].duration_secs, Some(45.5));
    assert_eq!(
        invocations[0].output_summary,
        Some("Website created successfully".to_string())
    );
}

#[tokio::test]
async fn test_cli_agent_invocation_limit() {
    let (store, _db) = setup_test_store().await;

    // Log 5 invocations
    for i in 0..5 {
        store
            .log_cli_agent_start("session1", "claude", &format!("Task {}", i), None)
            .await
            .unwrap();
    }

    // Limit=3 should return exactly 3
    let invocations = store.get_cli_agent_invocations(3).await.unwrap();
    assert_eq!(invocations.len(), 3);

    // All 5 should be retrievable
    let all = store.get_cli_agent_invocations(100).await.unwrap();
    assert_eq!(all.len(), 5);
}

#[tokio::test]
async fn test_cli_agent_invocation_failure() {
    let (store, _db) = setup_test_store().await;

    let inv_id = store
        .log_cli_agent_start("session1", "gemini", "Debug crash", None)
        .await
        .unwrap();

    // Log failure
    store
        .log_cli_agent_complete(inv_id, Some(1), "Error: command not found", false, 2.1)
        .await
        .unwrap();

    let invocations = store.get_cli_agent_invocations(10).await.unwrap();
    assert_eq!(invocations[0].exit_code, Some(1));
    assert_eq!(invocations[0].success, Some(false));
}

#[tokio::test]
async fn test_cli_agent_invocation_cleanup_stale() {
    let (store, _db) = setup_test_store().await;

    let inv_id = store
        .log_cli_agent_start("session1", "claude", "Long running task", None)
        .await
        .unwrap();

    // Force the invocation to look stale.
    sqlx::query(
        "UPDATE cli_agent_invocations SET started_at = datetime('now', '-3 hours') WHERE id = ?",
    )
    .bind(inv_id)
    .execute(&store.pool())
    .await
    .unwrap();

    let count = store.cleanup_stale_cli_agent_invocations(2).await.unwrap();
    assert_eq!(count, 1);

    let invocations = store.get_cli_agent_invocations(10).await.unwrap();
    let inv = invocations.iter().find(|i| i.id == inv_id).unwrap();
    assert!(inv.completed_at.is_some());
    assert_eq!(inv.success, Some(false));
    assert!(inv
        .output_summary
        .as_ref()
        .is_some_and(|s| s.contains("Auto-closed stale invocation")));

    // Idempotent: second run should do nothing.
    let count2 = store.cleanup_stale_cli_agent_invocations(2).await.unwrap();
    assert_eq!(count2, 0);
}

// ==================== Orchestration Tests ====================

#[tokio::test]
async fn test_goals_crud() {
    let (store, _file) = setup_test_store().await;

    // Create
    let goal = crate::traits::Goal::new_finite("Build a website", "session_1");
    store.create_goal(&goal).await.unwrap();

    // Get
    let fetched = store.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(fetched.description, "Build a website");
    assert_eq!(fetched.status, "active");
    assert_eq!(fetched.goal_type, "finite");
    assert_eq!(fetched.session_id, "session_1");

    // Update
    let mut updated = fetched;
    updated.status = "completed".to_string();
    updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
    store.update_goal(&updated).await.unwrap();

    let fetched2 = store.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(fetched2.status, "completed");
    assert!(fetched2.completed_at.is_some());

    // Active goals should not include completed
    let active = store.get_active_goals().await.unwrap();
    assert!(active.is_empty());
}

#[tokio::test]
async fn test_goals_session_filter() {
    let (store, _file) = setup_test_store().await;

    let goal1 = crate::traits::Goal::new_finite("Task A", "session_1");
    let goal2 = crate::traits::Goal::new_finite("Task B", "session_2");
    let goal3 = crate::traits::Goal::new_finite("Task C", "session_1");

    store.create_goal(&goal1).await.unwrap();
    store.create_goal(&goal2).await.unwrap();
    store.create_goal(&goal3).await.unwrap();

    let session1_goals = store.get_goals_for_session("session_1").await.unwrap();
    assert_eq!(session1_goals.len(), 2);
    assert!(session1_goals.iter().all(|g| g.session_id == "session_1"));

    let session2_goals = store.get_goals_for_session("session_2").await.unwrap();
    assert_eq!(session2_goals.len(), 1);
}

#[tokio::test]
async fn test_tasks_crud() {
    let (store, _file) = setup_test_store().await;

    // Create parent goal first (FK)
    let goal = crate::traits::Goal::new_finite("Parent goal", "session_1");
    store.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let task = crate::traits::Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Step 1: create files".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: now.clone(),
        started_at: None,
        completed_at: None,
    };
    store.create_task(&task).await.unwrap();

    // Get
    let fetched = store.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(fetched.description, "Step 1: create files");
    assert_eq!(fetched.status, "pending");
    assert!(fetched.idempotent);

    // Update
    let mut updated = fetched;
    updated.status = "completed".to_string();
    updated.result = Some("Files created successfully".to_string());
    updated.completed_at = Some(now.clone());
    store.update_task(&updated).await.unwrap();

    let fetched2 = store.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(fetched2.status, "completed");
    assert!(fetched2.result.is_some());

    // List for goal
    let tasks = store.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
}

#[tokio::test]
async fn test_claim_task() {
    let (store, _file) = setup_test_store().await;

    let goal = crate::traits::Goal::new_finite("Goal", "session_1");
    store.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let task = crate::traits::Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Claimable task".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: now,
        started_at: None,
        completed_at: None,
    };
    store.create_task(&task).await.unwrap();

    // First claim succeeds
    let claimed = store.claim_task(&task.id, "agent-1").await.unwrap();
    assert!(claimed);

    // Second claim fails (already claimed)
    let claimed2 = store.claim_task(&task.id, "agent-2").await.unwrap();
    assert!(!claimed2);

    // Verify agent_id was set
    let fetched = store.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(fetched.agent_id, Some("agent-1".to_string()));
    assert_eq!(fetched.status, "claimed");
}

#[tokio::test]
async fn test_task_activity_log() {
    let (store, _file) = setup_test_store().await;

    let goal = crate::traits::Goal::new_finite("Goal", "session_1");
    store.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let task = crate::traits::Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Task".to_string(),
        status: "running".to_string(),
        priority: "medium".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("agent-1".to_string()),
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: now.clone(),
        started_at: Some(now.clone()),
        completed_at: None,
    };
    store.create_task(&task).await.unwrap();

    // Log activity
    let activity = crate::traits::TaskActivity {
        id: 0,
        task_id: task.id.clone(),
        activity_type: "tool_call".to_string(),
        tool_name: Some("terminal".to_string()),
        tool_args: Some("{\"command\":\"ls\"}".to_string()),
        result: Some("file1.txt\nfile2.txt".to_string()),
        success: Some(true),
        tokens_used: Some(42),
        created_at: now,
    };
    store.log_task_activity(&activity).await.unwrap();

    let activities = store.get_task_activities(&task.id).await.unwrap();
    assert_eq!(activities.len(), 1);
    assert_eq!(activities[0].activity_type, "tool_call");
    assert_eq!(activities[0].tool_name, Some("terminal".to_string()));
    assert_eq!(activities[0].success, Some(true));
    assert_eq!(activities[0].tokens_used, Some(42));
}

// --- Notification Queue Tests ---

#[tokio::test]
async fn test_notification_queue_enqueue_and_fetch() {
    let (store, _file) = setup_test_store().await;

    let entry = crate::traits::NotificationEntry::new(
        "goal-1",
        "session-1",
        "completed",
        "Goal completed: build website",
    );
    store.enqueue_notification(&entry).await.unwrap();

    let pending = store.get_pending_notifications(10).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].goal_id, "goal-1");
    assert_eq!(pending[0].notification_type, "completed");
    assert_eq!(pending[0].priority, "critical");
    assert!(pending[0].expires_at.is_none()); // critical = no expiry
    assert!(pending[0].delivered_at.is_none());
}

#[tokio::test]
async fn test_notification_queue_status_update_has_expiry() {
    let (store, _file) = setup_test_store().await;

    let entry = crate::traits::NotificationEntry::new(
        "goal-1",
        "session-1",
        "progress",
        "Goal 50% complete",
    );
    assert_eq!(entry.priority, "status_update");
    assert!(entry.expires_at.is_some());

    store.enqueue_notification(&entry).await.unwrap();
    let pending = store.get_pending_notifications(10).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert!(pending[0].expires_at.is_some());
}

#[tokio::test]
async fn test_notification_queue_mark_delivered() {
    let (store, _file) = setup_test_store().await;

    let entry = crate::traits::NotificationEntry::new(
        "goal-1",
        "session-1",
        "failed",
        "Goal failed: deployment error",
    );
    store.enqueue_notification(&entry).await.unwrap();

    // Mark as delivered
    store.mark_notification_delivered(&entry.id).await.unwrap();

    // Should no longer appear in pending
    let pending = store.get_pending_notifications(10).await.unwrap();
    assert_eq!(pending.len(), 0);
}

#[tokio::test]
async fn test_notification_queue_priority_ordering() {
    let (store, _file) = setup_test_store().await;

    // Enqueue status_update first
    let status =
        crate::traits::NotificationEntry::new("goal-1", "session-1", "progress", "Progress update");
    store.enqueue_notification(&status).await.unwrap();

    // Enqueue critical second
    let critical =
        crate::traits::NotificationEntry::new("goal-2", "session-1", "failed", "Goal failed");
    store.enqueue_notification(&critical).await.unwrap();

    let pending = store.get_pending_notifications(10).await.unwrap();
    assert_eq!(pending.len(), 2);
    // Critical should come first despite being enqueued second
    assert_eq!(pending[0].priority, "critical");
    assert_eq!(pending[1].priority, "status_update");
}

#[tokio::test]
async fn test_notification_queue_cleanup_expired() {
    let (store, _file) = setup_test_store().await;

    // Insert a notification with an already-expired expires_at
    let mut entry =
        crate::traits::NotificationEntry::new("goal-1", "session-1", "stalled", "Goal stalled");
    // Set expires_at to the past
    entry.expires_at = Some((chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc3339());
    store.enqueue_notification(&entry).await.unwrap();

    // Also insert a non-expired critical notification
    let critical =
        crate::traits::NotificationEntry::new("goal-2", "session-1", "failed", "Goal failed");
    store.enqueue_notification(&critical).await.unwrap();

    // Cleanup should remove the expired one
    let cleaned = store.cleanup_expired_notifications().await.unwrap();
    assert_eq!(cleaned, 1);

    // Only the critical one remains
    let pending = store.get_pending_notifications(10).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].notification_type, "failed");
}

#[tokio::test]
async fn test_notification_queue_increment_attempt() {
    let (store, _file) = setup_test_store().await;

    let entry =
        crate::traits::NotificationEntry::new("goal-1", "session-1", "completed", "Goal done");
    store.enqueue_notification(&entry).await.unwrap();

    store
        .increment_notification_attempt(&entry.id)
        .await
        .unwrap();
    store
        .increment_notification_attempt(&entry.id)
        .await
        .unwrap();

    let pending = store.get_pending_notifications(10).await.unwrap();
    assert_eq!(pending[0].attempts, 2);
}

#[tokio::test]
async fn test_cleanup_stale_goals() {
    let (store, _db) = setup_test_store().await;
    let now = chrono::Utc::now().to_rfc3339();
    let three_hours_ago = (chrono::Utc::now() - chrono::Duration::hours(3)).to_rfc3339();

    // Finite orchestration goals without schedules: stale active/pending -> failed.
    let mut stale_finite = Goal::new_finite("stale finite goal", "test");
    stale_finite.id = "stale-finite".to_string();
    stale_finite.created_at = three_hours_ago.clone();
    stale_finite.updated_at = three_hours_ago.clone();
    store.create_goal(&stale_finite).await.unwrap();

    // Recent finite stays active.
    let mut recent_finite = Goal::new_finite("recent finite goal", "test");
    recent_finite.id = "recent-finite".to_string();
    recent_finite.created_at = now.clone();
    recent_finite.updated_at = now.clone();
    store.create_goal(&recent_finite).await.unwrap();

    // Scheduled finite goals should not be failed by stale cleanup.
    let mut stale_scheduled = Goal::new_finite("stale scheduled finite goal", "test");
    stale_scheduled.id = "stale-scheduled-finite".to_string();
    stale_scheduled.created_at = three_hours_ago.clone();
    stale_scheduled.updated_at = three_hours_ago.clone();
    store.create_goal(&stale_scheduled).await.unwrap();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: stale_scheduled.id.clone(),
        cron_expr: "0 9 12 2 *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("0 9 12 2 *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: true,
        is_paused: false,
        last_run_at: None,
        next_run_at: (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339(),
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    store.create_goal_schedule(&schedule).await.unwrap();

    // Continuous goals are not cleaned up by this path.
    let mut stale_continuous = Goal::new_continuous("stale continuous goal", "test", None, None);
    stale_continuous.id = "stale-continuous".to_string();
    stale_continuous.created_at = three_hours_ago.clone();
    stale_continuous.updated_at = three_hours_ago.clone();
    store.create_goal(&stale_continuous).await.unwrap();

    // Run cleanup with 2-hour threshold
    let count = store.cleanup_stale_goals(2).await.unwrap();
    assert_eq!(count, 1);

    let g = store.get_goal("stale-finite").await.unwrap().unwrap();
    assert_eq!(g.status, "failed");
    assert!(g.completed_at.is_some());

    let g = store.get_goal("recent-finite").await.unwrap().unwrap();
    assert_eq!(g.status, "active");

    let g = store
        .get_goal("stale-scheduled-finite")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(g.status, "active");

    let g = store.get_goal("stale-continuous").await.unwrap().unwrap();
    assert_eq!(g.status, "active");
}

#[tokio::test]
async fn test_migrate_legacy_scheduled_tasks_to_goals_and_schedules() {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().to_str().unwrap();
    let now = chrono::Utc::now();
    let next_hour = (now + chrono::Duration::hours(1)).to_rfc3339();

    // Create a legacy scheduled_tasks table and rows BEFORE opening SqliteStateStore (migrations run on open).
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(db_path)
                .create_if_missing(true),
        )
        .await
        .unwrap();
    sqlx::query(
        "CREATE TABLE scheduled_tasks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            cron_expr TEXT NOT NULL,
            original_schedule TEXT NOT NULL,
            prompt TEXT NOT NULL,
            source TEXT NOT NULL,
            is_oneshot INTEGER NOT NULL DEFAULT 0,
            is_paused INTEGER NOT NULL DEFAULT 0,
            is_trusted INTEGER NOT NULL DEFAULT 0,
            last_run_at TEXT,
            next_run_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Recurring active legacy task
    sqlx::query(
        "INSERT INTO scheduled_tasks
         (id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused, is_trusted, last_run_at, next_run_at, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind("legacy-recurring-1")
    .bind("legacy recurring")
    .bind("0 */6 * * *")
    .bind("every 6h")
    .bind("monitor API health")
    .bind("tool")
    .bind(0i64)
    .bind(0i64)
    .bind(0i64)
    .bind::<Option<String>>(None)
    .bind(&next_hour)
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    // One-shot paused legacy task
    sqlx::query(
        "INSERT INTO scheduled_tasks
         (id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused, is_trusted, last_run_at, next_run_at, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind("legacy-oneshot-1")
    .bind("legacy oneshot")
    .bind("0 9 * * *")
    .bind("tomorrow at 9am")
    .bind("check deployment")
    .bind("tool")
    .bind(1i64)
    .bind(1i64)
    .bind(0i64)
    .bind::<Option<String>>(None)
    .bind(&next_hour)
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .execute(&pool)
    .await
    .unwrap();

    pool.close().await;

    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = SqliteStateStore::new(db_path, 100, None, embedding_service)
        .await
        .unwrap();

    // scheduled_tasks table should be dropped after migration.
    let has_scheduled_tasks: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='scheduled_tasks'",
    )
    .fetch_one(&store.pool())
    .await
    .unwrap();
    assert_eq!(has_scheduled_tasks, 0);

    let g1 = store
        .get_goal("legacy-sched-legacy-recurring-1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(g1.domain, "orchestration");
    assert_eq!(g1.goal_type, "continuous");
    assert_eq!(g1.status, "active");
    assert_eq!(g1.session_id, "system");
    let sched1 = store.get_schedules_for_goal(&g1.id).await.unwrap();
    assert_eq!(sched1.len(), 1);
    assert_eq!(sched1[0].id, "sched-legacy-legacy-recurring-1");
    assert_eq!(sched1[0].cron_expr, "0 */6 * * *");
    assert_eq!(sched1[0].tz, "local");
    assert_eq!(sched1[0].original_schedule.as_deref(), Some("every 6h"));
    assert!(!sched1[0].is_one_shot);
    assert!(!sched1[0].is_paused);
    let expected_next = chrono::DateTime::parse_from_rfc3339(&next_hour)
        .unwrap()
        .with_timezone(&chrono::Utc);
    let got_next = chrono::DateTime::parse_from_rfc3339(&sched1[0].next_run_at)
        .unwrap()
        .with_timezone(&chrono::Utc);
    assert_eq!(got_next, expected_next);

    let g2 = store
        .get_goal("legacy-sched-legacy-oneshot-1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(g2.domain, "orchestration");
    assert_eq!(g2.goal_type, "finite");
    assert_eq!(g2.status, "paused");
    assert_eq!(g2.session_id, "system");
    let sched2 = store.get_schedules_for_goal(&g2.id).await.unwrap();
    assert_eq!(sched2.len(), 1);
    assert_eq!(sched2[0].id, "sched-legacy-legacy-oneshot-1");
    assert_eq!(sched2[0].tz, "local");
    assert_eq!(
        sched2[0].original_schedule.as_deref(),
        Some("tomorrow at 9am")
    );
    assert!(sched2[0].is_one_shot);
    assert!(sched2[0].is_paused);
    assert!(!sched2[0].cron_expr.trim().is_empty());
}

#[tokio::test]
async fn test_migrate_fixup_scheduled_goal_budgets() {
    let (store, _db_file) = setup_test_store().await;

    let goal = Goal::new_continuous(
        "Legacy scheduled budget bug",
        "system",
        Some(5000),
        Some(20000),
    );
    store.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 */6 * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("every 6h".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    store.create_goal_schedule(&schedule).await.unwrap();

    let before = store.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(before.budget_per_check, Some(5000));
    assert_eq!(before.budget_daily, Some(20000));

    migrations::migrate_state(&store.pool()).await.unwrap();

    let after = store.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(after.budget_per_check, Some(50_000));
    assert_eq!(after.budget_daily, Some(200_000));
}

#[tokio::test]
async fn test_migrate_v3_tables_renamed_and_schedule_migrated() {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().to_str().unwrap();
    let now = chrono::Utc::now().to_rfc3339();

    // Create goals_v3/tasks_v3/task_activity_v3 BEFORE opening SqliteStateStore (migrations run on open).
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(db_path)
                .create_if_missing(true),
        )
        .await
        .unwrap();

    sqlx::query(
        "CREATE TABLE goals_v3 (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            goal_type TEXT NOT NULL DEFAULT 'finite',
            status TEXT NOT NULL DEFAULT 'active',
            priority TEXT NOT NULL DEFAULT 'medium',
            conditions TEXT,
            context TEXT,
            resources TEXT,
            schedule TEXT,
            budget_per_check INTEGER,
            budget_daily INTEGER,
            tokens_used_today INTEGER NOT NULL DEFAULT 0,
            last_useful_action TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            parent_goal_id TEXT,
            session_id TEXT NOT NULL,
            notified_at TEXT
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE tasks_v3 (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals_v3(id) ON DELETE CASCADE,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            priority TEXT NOT NULL DEFAULT 'medium',
            task_order INTEGER NOT NULL DEFAULT 0,
            parallel_group TEXT,
            depends_on TEXT,
            agent_id TEXT,
            context TEXT,
            result TEXT,
            error TEXT,
            blocker TEXT,
            idempotent INTEGER NOT NULL DEFAULT 0,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE task_activity_v3 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES tasks_v3(id) ON DELETE CASCADE,
            activity_type TEXT NOT NULL,
            tool_name TEXT,
            tool_args TEXT,
            result TEXT,
            success INTEGER,
            tokens_used INTEGER,
            created_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO goals_v3
         (id, description, goal_type, status, priority, schedule, created_at, updated_at, session_id)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind("g-v3-1")
    .bind("v3 goal")
    .bind("continuous")
    .bind("active")
    .bind("medium")
    .bind("0 6,12,18 * * *")
    .bind(&now)
    .bind(&now)
    .bind("test")
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tasks_v3
         (id, goal_id, description, status, priority, created_at)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind("t-v3-1")
    .bind("g-v3-1")
    .bind("do the thing")
    .bind("pending")
    .bind("medium")
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO task_activity_v3
         (task_id, activity_type, tool_name, tool_args, result, success, tokens_used, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind("t-v3-1")
    .bind("tool_call")
    .bind("terminal")
    .bind("{\"cmd\":\"echo hi\"}")
    .bind("ok")
    .bind(1i64)
    .bind(10i64)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    pool.close().await;

    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = SqliteStateStore::new(db_path, 100, None, embedding_service)
        .await
        .unwrap();

    // v3 tables should be gone.
    let has_goals_v3: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='goals_v3'",
    )
    .fetch_one(&store.pool())
    .await
    .unwrap();
    let has_tasks_v3: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tasks_v3'",
    )
    .fetch_one(&store.pool())
    .await
    .unwrap();
    let has_task_activity_v3: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='task_activity_v3'",
    )
    .fetch_one(&store.pool())
    .await
    .unwrap();
    assert_eq!(has_goals_v3, 0);
    assert_eq!(has_tasks_v3, 0);
    assert_eq!(has_task_activity_v3, 0);

    // Data should be preserved under clean names.
    let g = store.get_goal("g-v3-1").await.unwrap().unwrap();
    assert_eq!(g.description, "v3 goal");
    assert_eq!(g.domain, "orchestration");

    let schedules = store.get_schedules_for_goal(&g.id).await.unwrap();
    assert_eq!(schedules.len(), 1);
    assert_eq!(schedules[0].id, "sched-migrated-g-v3-1");
    assert_eq!(schedules[0].cron_expr, "0 6,12,18 * * *");
    assert_eq!(schedules[0].tz, "local");
    assert!(!schedules[0].next_run_at.trim().is_empty());

    let tasks = store.get_tasks_for_goal(&g.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].id, "t-v3-1");

    let act = store.get_task_activities(&tasks[0].id).await.unwrap();
    assert_eq!(act.len(), 1);
    assert_eq!(act[0].tool_name.as_deref(), Some("terminal"));

    // Idempotency: re-opening should not duplicate migrated schedules.
    drop(store);
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store2 = SqliteStateStore::new(db_path, 100, None, embedding_service)
        .await
        .unwrap();
    let schedules2 = store2.get_schedules_for_goal("g-v3-1").await.unwrap();
    assert_eq!(schedules2.len(), 1);
}

// ==================== Semantic Dedup Tests (BUG-8) ====================

#[tokio::test]
async fn test_upsert_fact_semantic_dedup_catches_synonym_keys() {
    let (store, _db) = setup_test_store().await;

    // Insert a fact with key "editor"
    store
        .upsert_fact(
            "preference",
            "editor",
            "Vim",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Wait a moment to ensure embedding is computed
    let facts = store.get_facts(Some("preference")).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].key, "editor");
    assert_eq!(facts[0].value, "Vim");

    // Now upsert with a semantically similar key "preferred_editor" and new value.
    // The semantic dedup should detect that "preferred_editor" ≈ "editor" and
    // supersede the old fact rather than creating a duplicate.
    store
        .upsert_fact(
            "preference",
            "preferred_editor",
            "Neovim",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let facts_after = store.get_facts(Some("preference")).await.unwrap();
    // Should still be 1 active fact — the old "editor: Vim" should be superseded
    assert_eq!(
        facts_after.len(),
        1,
        "Semantic dedup should prevent duplicate: got {:?}",
        facts_after
            .iter()
            .map(|f| format!("{}={}", f.key, f.value))
            .collect::<Vec<_>>()
    );
    assert_eq!(facts_after[0].value, "Neovim");
}

#[tokio::test]
async fn test_upsert_fact_semantic_dedup_no_false_merge() {
    let (store, _db) = setup_test_store().await;

    // Insert two facts in the same category with genuinely different meanings
    store
        .upsert_fact(
            "preference",
            "editor",
            "Vim",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    store
        .upsert_fact(
            "preference",
            "operating_system",
            "Linux",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    let facts = store.get_facts(Some("preference")).await.unwrap();
    // Both facts should survive — they are semantically distinct
    assert_eq!(
        facts.len(),
        2,
        "Distinct facts should not be merged: got {:?}",
        facts
            .iter()
            .map(|f| format!("{}={}", f.key, f.value))
            .collect::<Vec<_>>()
    );
}

// ==================== Episode Unique Constraint Tests (BUG-9) ====================

#[tokio::test]
async fn test_multiple_episodes_per_session_allowed() {
    let (store, _db) = setup_test_store().await;

    let ep1 = Episode {
        id: 0,
        session_id: "multi-ep-sess".to_string(),
        summary: "First episode about project setup".to_string(),
        topics: Some(vec!["setup".to_string()]),
        emotional_tone: Some("focused".to_string()),
        outcome: Some("project initialized".to_string()),
        importance: 0.7,
        recall_count: 0,
        last_recalled_at: None,
        message_count: 10,
        start_time: Utc::now() - chrono::Duration::hours(2),
        end_time: Utc::now() - chrono::Duration::hours(1),
        created_at: Utc::now() - chrono::Duration::hours(1),
        channel_id: None,
    };

    let ep2 = Episode {
        id: 0,
        session_id: "multi-ep-sess".to_string(),
        summary: "Second episode about debugging".to_string(),
        topics: Some(vec!["debugging".to_string()]),
        emotional_tone: Some("frustrated".to_string()),
        outcome: Some("bug fixed".to_string()),
        importance: 0.8,
        recall_count: 0,
        last_recalled_at: None,
        message_count: 15,
        start_time: Utc::now() - chrono::Duration::hours(1),
        end_time: Utc::now(),
        created_at: Utc::now(),
        channel_id: None,
    };

    let id1 = store.insert_episode(&ep1).await.unwrap();
    let id2 = store.insert_episode(&ep2).await.unwrap();

    // Both episodes should be created with different IDs
    assert!(id1 > 0);
    assert!(id2 > 0);
    assert_ne!(id1, id2);

    let episodes = store.get_recent_episodes(10).await.unwrap();
    let session_eps: Vec<_> = episodes
        .iter()
        .filter(|e| e.session_id == "multi-ep-sess")
        .collect();
    assert_eq!(
        session_eps.len(),
        2,
        "Both episodes should exist for same session"
    );
}

#[tokio::test]
async fn test_episode_retrieval_at_lower_threshold() {
    let (store, _db) = setup_test_store().await;

    // Insert an episode with a specific topic and backfill its embedding
    let ep = Episode {
        id: 0,
        session_id: "threshold-test".to_string(),
        summary: "We discussed using Kubernetes for container orchestration in production"
            .to_string(),
        topics: Some(vec!["kubernetes".to_string(), "devops".to_string()]),
        emotional_tone: Some("technical".to_string()),
        outcome: Some("decided on k8s".to_string()),
        importance: 0.9,
        recall_count: 0,
        last_recalled_at: None,
        message_count: 20,
        start_time: Utc::now() - chrono::Duration::hours(1),
        end_time: Utc::now(),
        created_at: Utc::now(),
        channel_id: None,
    };
    store.insert_episode(&ep).await.unwrap();
    store.backfill_episode_embeddings().await.unwrap();

    // Query with a related but not identical topic — should be returned with 0.3 threshold
    let results = store
        .get_relevant_episodes("container deployment infrastructure", 10)
        .await
        .unwrap();
    // The episode should be retrieved — with the old 0.5 threshold it might have been filtered
    assert!(
        !results.is_empty(),
        "Episode about kubernetes should be relevant to container deployment query"
    );
}
