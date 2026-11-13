// The memory jobs over the daemon's store and event log.

use super::*;
use crate::traits::Message;
use aidaemon_memory::full::MemoryJobs;
use aidaemon_runtime::history::uncaptured_sessions_sql;
use sqlx::SqlitePool;

use crate::config::ProviderKind;
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::embeddings::EmbeddingService;
use crate::state::SqliteStateStore;
use crate::testing::MockProvider;
use crate::traits::store_prelude::*;
use crate::traits::MessageAnnotation;
use chrono::Utc;
use serde_json::json;
use std::sync::Arc;

/// An event store over `store`'s database, for the gateway's rows.
async fn test_event_store(store: &Arc<SqliteStateStore>) -> Arc<crate::events::EventStore> {
    Arc::new(
        <crate::events::EventStore as crate::events::EventStoreForTests>::new(store.pool())
            .await
            .expect("event store"),
    )
}

async fn setup_memory_manager_with_responses(
    responses: Vec<crate::traits::ProviderResponse>,
) -> (
    Arc<SqliteStateStore>,
    Arc<MockProvider>,
    Arc<MemoryJobs>,
    tempfile::NamedTempFile,
) {
    let db_file = tempfile::NamedTempFile::new().expect("temp db file");
    let embedding_service = Arc::new(EmbeddingService::new().expect("embedding service"));
    let store = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().expect("db path"),
            100,
            None,
            embedding_service.clone(),
        )
        .await
        .expect("state store"),
    );
    let provider = Arc::new(MockProvider::with_responses(responses));
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone(),
        None,
        ProviderKind::OpenaiCompatible,
        "mock".to_string(),
    );
    let jobs = memory_jobs(
        store.pool(),
        store.clone(),
        store.memory_stores(),
        embedding_service,
        llm_runtime.clone(),
        SideCallGateway::new(llm_runtime, store.clone(), test_event_store(&store).await),
        Duration::from_secs(60),
        None,
    );
    (store, provider, jobs, db_file)
}

#[tokio::test]
async fn a_replacing_memory_system_brings_its_jobs_and_goal_review_stays() {
    let (store, _provider, jobs, _db) = setup_memory_manager_with_responses(vec![]).await;
    let coordinator = || {
        let (_tx, rx) = tokio::sync::mpsc::channel(1);
        crate::heartbeat::HeartbeatCoordinator::new(
            store.clone() as Arc<dyn StateStore>,
            60,
            3,
            rx,
            None,
            None,
            None,
        )
    };

    let mut own = coordinator();
    crate::startup::runtime::register_memory_jobs(&mut own, store.pool(), &jobs, true);
    let own = own.scheduled_names();
    for job in ["goal_review", "consolidation", "episodes", "memory_decay"] {
        assert!(own.iter().any(|name| name == job), "{job} in {own:?}");
    }

    let replacement = aidaemon_memory::ScopedMemory::new(aidaemon_memory::Memory::builder(
        Arc::new(aidaemon_memory::InMemoryMemoryStore::new()),
        Arc::new(aidaemon_memory::HashEmbedder::default()),
    ));
    let memory = crate::agent::memory_plugin::ConfiguredMemory::replaced_by(Arc::new(replacement));
    assert!(!memory.own_memory());
    let mut replaced = coordinator();
    crate::startup::runtime::register_memory_jobs(
        &mut replaced,
        store.pool(),
        &jobs,
        memory.own_memory(),
    );
    for service in memory.services {
        replaced.add_service(service);
    }
    let mut replaced = replaced.scheduled_names();
    replaced.sort();
    assert_eq!(replaced, ["goal_review", "memory_decay"]);
}

async fn insert_owner_session_events(
    pool: &SqlitePool,
    session_id: &str,
    count: usize,
    age: chrono::Duration,
    execution_origin: Option<&str>,
    content_sentinel: &str,
) {
    let base = Utc::now() - age;
    for index in 0..count {
        let mut data = json!({
            "content": format!("{content_sentinel} message {index}"),
            "message_id": format!("{session_id}-message-{index}"),
            "has_attachments": false,
            "user_role": "owner",
            "channel_visibility": "private"
        });
        if index == 0 {
            if let Some(origin) = execution_origin {
                data["execution_origin"] = json!(origin);
            }
        }
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at)
             VALUES (?, 'user_message', ?, ?)",
        )
        .bind(session_id)
        .bind(data.to_string())
        .bind((base + chrono::Duration::seconds(index as i64)).to_rfc3339())
        .execute(pool)
        .await
        .expect("insert owner session event");
    }
}

#[tokio::test]
async fn mandate_sessions_are_excluded_from_fact_consolidation_by_durable_origin() {
    let response = MockProvider::text_response(
        r#"[{"category":"preference","key":"editor","value":"Helix","privacy":"global"}]"#,
    );
    let (store, provider, jobs, _db_file) =
        setup_memory_manager_with_responses(vec![response]).await;
    insert_owner_session_events(
        &store.pool(),
        "ordinary-owner-fact-session",
        1,
        chrono::Duration::hours(2),
        None,
        "Important: remember that I prefer Helix as my editor.",
    )
    .await;
    insert_owner_session_events(
        &store.pool(),
        "owner-looking-mandate-fact-session",
        1,
        chrono::Duration::hours(2),
        Some("mandate"),
        "MANDATE_PRIVATE_SENTINEL must never become global memory.",
    )
    .await;

    jobs.consolidate(Utc::now())
        .await
        .expect("fact consolidation");

    assert_eq!(provider.call_count().await, 1);
    let calls = provider.call_log.lock().await;
    let prompt = serde_json::to_string(&calls[0].messages).expect("serialize call");
    assert!(prompt.contains("prefer Helix"));
    assert!(!prompt.contains("MANDATE_PRIVATE_SENTINEL"));
    drop(calls);

    let fact_value: String = sqlx::query_scalar(
        "SELECT value FROM facts
         WHERE category = 'preference' AND key = 'editor' AND superseded_at IS NULL",
    )
    .fetch_one(&store.pool())
    .await
    .expect("ordinary fact learned");
    assert_eq!(fact_value, "Helix");
}

#[tokio::test]
async fn mandate_sessions_are_excluded_from_idle_episode_candidates() {
    let response = MockProvider::text_response(
        r#"{
            "summary":"Ordinary owner idle session",
            "topics":["ordinary"],
            "emotional_tone":"productive",
            "outcome":"resolved",
            "goals_mentioned":[],
            "emotional_intensity":0.2
        }"#,
    );
    let (store, provider, jobs, _db_file) =
        setup_memory_manager_with_responses(vec![response]).await;
    insert_owner_session_events(
        &store.pool(),
        "ordinary-owner-idle-session",
        5,
        chrono::Duration::minutes(45),
        None,
        "ORDINARY_IDLE_SENTINEL",
    )
    .await;
    insert_owner_session_events(
        &store.pool(),
        "owner-looking-mandate-idle-session",
        5,
        chrono::Duration::minutes(45),
        Some("mandate"),
        "MANDATE_IDLE_SENTINEL",
    )
    .await;

    jobs.episodes(Utc::now()).await.expect("idle episode pass");

    assert_eq!(provider.call_count().await, 1);
    let ordinary_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM episodes WHERE session_id = ?")
            .bind("ordinary-owner-idle-session")
            .fetch_one(&store.pool())
            .await
            .expect("ordinary episode count");
    let mandate_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM episodes WHERE session_id = ?")
            .bind("owner-looking-mandate-idle-session")
            .fetch_one(&store.pool())
            .await
            .expect("mandate episode count");
    assert_eq!(ordinary_count, 1);
    assert_eq!(mandate_count, 0);
}

/// Episodes are read through the memory stores and joined to events as
/// a parameter: a session its latest episode covers is not a candidate,
/// and one with newer events is.
#[tokio::test]
async fn idle_candidates_respect_each_sessions_latest_episode() {
    let (store, provider, jobs, _db_file) = setup_memory_manager_with_responses(vec![]).await;
    for session in ["covered-session", "stale-episode-session"] {
        insert_owner_session_events(
            &store.pool(),
            session,
            5,
            chrono::Duration::minutes(45),
            None,
            "EPISODE_BOUND_SENTINEL",
        )
        .await;
    }
    // Ends after every event, but bound to event 0: selecting this
    // session by mistake would summarize all of its events.
    let covered_end = Utc::now().to_rfc3339();
    let stale_end = (Utc::now() - chrono::Duration::days(3)).to_rfc3339();
    for (session, end) in [
        ("covered-session", &covered_end),
        ("stale-episode-session", &stale_end),
    ] {
        sqlx::query(
            "INSERT INTO episodes (session_id, summary, importance, message_count,
                                   start_time, end_time, created_at, end_event_id)
             VALUES (?1, 'earlier', 0.5, 5, ?2, ?2, ?2, 0)",
        )
        .bind(session)
        .bind(end)
        .execute(&store.pool())
        .await
        .expect("seed episode");
    }

    jobs.episodes(Utc::now()).await.expect("idle episode pass");

    assert_eq!(provider.call_count().await, 1, "only the stale session");
}

#[tokio::test]
async fn mandate_sessions_are_excluded_from_long_episode_candidates() {
    let response = MockProvider::text_response(
        r#"{
            "summary":"Ordinary owner long session",
            "topics":["ordinary"],
            "emotional_tone":"productive",
            "outcome":"ongoing",
            "goals_mentioned":[],
            "emotional_intensity":0.2
        }"#,
    );
    let (store, provider, jobs, _db_file) =
        setup_memory_manager_with_responses(vec![response]).await;
    insert_owner_session_events(
        &store.pool(),
        "ordinary-owner-long-session",
        20,
        chrono::Duration::minutes(10),
        None,
        "ORDINARY_LONG_SENTINEL",
    )
    .await;
    insert_owner_session_events(
        &store.pool(),
        "owner-looking-mandate-long-session",
        20,
        chrono::Duration::minutes(10),
        Some("mandate"),
        "MANDATE_LONG_SENTINEL",
    )
    .await;

    jobs.episodes(Utc::now()).await.expect("long episode pass");

    assert_eq!(provider.call_count().await, 1);
    let ordinary_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM episodes WHERE session_id = ?")
            .bind("ordinary-owner-long-session")
            .fetch_one(&store.pool())
            .await
            .expect("ordinary episode count");
    let mandate_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM episodes WHERE session_id = ?")
            .bind("owner-looking-mandate-long-session")
            .fetch_one(&store.pool())
            .await
            .expect("mandate episode count");
    assert_eq!(ordinary_count, 1);
    assert_eq!(mandate_count, 0);
}

#[tokio::test]
async fn test_consolidation_skips_public_external_sessions() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service.clone(),
        )
        .await
        .unwrap(),
    );

    // Ensure events table exists for visibility lookup.
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

    let session_id = "pubext_session_consolidation";
    let msg_id = uuid::Uuid::new_v4().to_string();
    // Mark the session as PublicExternal via the user_message event metadata.
    let data = serde_json::json!({
        "content": "hello world",
        "message_id": msg_id,
        "has_attachments": false,
        "user_role": "owner",
        "channel_visibility": "public_external",
        "channel_id": "twitter:ext_999",
        "platform": "twitter",
    });
    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(session_id)
    .bind("user_message")
    .bind(data.to_string())
    .bind((Utc::now() - chrono::Duration::hours(2)).to_rfc3339())
    .execute(&store.pool())
    .await
    .unwrap();

    let provider = Arc::new(MockProvider::new());
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone(),
        None,
        ProviderKind::OpenaiCompatible,
        "mock".to_string(),
    );
    let mgr = memory_jobs(
        store.pool(),
        store.clone(),
        store.memory_stores(),
        embedding_service,
        llm_runtime.clone(),
        SideCallGateway::new(llm_runtime, store.clone(), test_event_store(&store).await),
        Duration::from_secs(60),
        None,
    );

    mgr.consolidate(Utc::now()).await.unwrap();

    // Provider should not be called for PublicExternal sessions.
    assert_eq!(provider.call_count().await, 0);

    // Cursor should advance so we don't repeatedly revisit skipped sessions.
    let cursor = store
        .get_setting(&format!("memory_fact_consolidation_cursor:{}", session_id))
        .await
        .unwrap();
    assert!(cursor.is_some());

    // No facts should be learned.
    let cnt: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM facts")
        .fetch_one(&store.pool())
        .await
        .unwrap();
    assert_eq!(cnt, 0);
}

#[tokio::test]
async fn test_consolidation_skips_non_owner_sessions() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service.clone(),
        )
        .await
        .unwrap(),
    );

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

    let session_id = "slack:D_non_owner";
    let user_data = serde_json::json!({
        "content": "My name is Alice",
        "message_id": "msg-1",
        "has_attachments": false,
        "user_role": "public",
        "channel_visibility": "private",
        "channel_id": "slack:D_non_owner",
        "platform": "slack",
    });
    let assistant_data = serde_json::json!({
        "message_id": "msg-2",
        "content": "Nice to meet you."
    });

    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(session_id)
    .bind("user_message")
    .bind(user_data.to_string())
    .bind((Utc::now() - chrono::Duration::hours(2)).to_rfc3339())
    .execute(&store.pool())
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(session_id)
    .bind("assistant_response")
    .bind(assistant_data.to_string())
    .bind((Utc::now() - chrono::Duration::hours(2) + chrono::Duration::seconds(5)).to_rfc3339())
    .execute(&store.pool())
    .await
    .unwrap();

    let provider = Arc::new(MockProvider::new());
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone(),
        None,
        ProviderKind::OpenaiCompatible,
        "mock".to_string(),
    );
    let mgr = memory_jobs(
        store.pool(),
        store.clone(),
        store.memory_stores(),
        embedding_service,
        llm_runtime.clone(),
        SideCallGateway::new(llm_runtime, store.clone(), test_event_store(&store).await),
        Duration::from_secs(60),
        None,
    );

    mgr.consolidate(Utc::now()).await.unwrap();

    assert_eq!(provider.call_count().await, 0);

    let cursor = store
        .get_setting(&format!("memory_fact_consolidation_cursor:{}", session_id))
        .await
        .unwrap();
    assert!(cursor.is_some());

    let cnt: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM facts")
        .fetch_one(&store.pool())
        .await
        .unwrap();
    assert_eq!(cnt, 0);
}

#[tokio::test]
async fn test_fetch_owner_user_messages_since_filters_non_owner_messages() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service.clone(),
        )
        .await
        .unwrap(),
    );

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

    let owner_data = serde_json::json!({
        "content": "Please keep responses short.",
        "message_id": "owner-msg",
        "has_attachments": false,
        "user_role": "owner"
    });
    let public_data = serde_json::json!({
        "content": "I like pirate responses.",
        "message_id": "public-msg",
        "has_attachments": false,
        "user_role": "public"
    });

    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind("owner-session")
    .bind("user_message")
    .bind(owner_data.to_string())
    .bind((Utc::now() - chrono::Duration::minutes(10)).to_rfc3339())
    .execute(&store.pool())
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind("public-session")
    .bind("user_message")
    .bind(public_data.to_string())
    .bind((Utc::now() - chrono::Duration::minutes(5)).to_rfc3339())
    .execute(&store.pool())
    .await
    .unwrap();

    let _ = embedding_service;
    let mgr = EventHistory::new(store.pool(), Some(store.clone()));

    let since = (Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
    let messages = mgr
        .fetch_owner_user_messages_since(&since, 10)
        .await
        .unwrap();

    assert_eq!(messages.len(), 1);
    assert_eq!(
        messages[0].content.as_deref(),
        Some("Please keep responses short.")
    );
}

#[tokio::test]
async fn test_fetch_session_messages_since_last_episode_supports_legacy_annotations() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let store = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service.clone(),
        )
        .await
        .unwrap(),
    );

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

    let session_id = "legacy_annotations_session";
    let legacy_data = json!({
        "message_id": "legacy-msg",
        "tool_call_id": "call-legacy",
        "name": "system",
        "result": "[SYSTEM] Legacy internal note"
    });
    let annotated_data = json!({
        "message_id": "annotated-msg",
        "tool_call_id": "call-annotated",
        "name": "terminal",
        "result": "cargo test\n\n[SYSTEM] Do not retry.",
        "annotations": [{"type": "appended_system_notice"}]
    });

    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(session_id)
    .bind("tool_result")
    .bind(legacy_data.to_string())
    .bind((Utc::now() - chrono::Duration::minutes(2)).to_rfc3339())
    .execute(&store.pool())
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(session_id)
    .bind("tool_result")
    .bind(annotated_data.to_string())
    .bind((Utc::now() - chrono::Duration::minutes(1)).to_rfc3339())
    .execute(&store.pool())
    .await
    .unwrap();

    let _ = embedding_service;
    let mgr = EventHistory::new(store.pool(), Some(store.clone()));

    let messages = mgr
        .session_messages_after(session_id, None, 10)
        .await
        .unwrap();

    assert_eq!(messages.len(), 2);

    let legacy = &messages[0];
    assert!(legacy.annotations.is_empty());
    assert_eq!(
        legacy.effective_annotations().as_ref(),
        [MessageAnnotation::EntireSystemNotice]
    );
    assert!(legacy.primary_content().is_none());

    let annotated = &messages[1];
    assert_eq!(
        annotated.annotations,
        vec![MessageAnnotation::AppendedSystemNotice]
    );
    assert_eq!(annotated.primary_content().as_deref(), Some("cargo test"));
}

/// The pre-rewrite query: a `LEFT JOIN` against the unindexed JSON list.
/// Kept only to prove the indexed form selects the same sessions.
fn legacy_uncaptured_sessions_sql(filter: &str, having: &str) -> String {
    format!(
        "SELECT ev.session_id
         FROM events ev
         LEFT JOIN (
             SELECT json_extract(value, '$[0]') AS session_id,
                    json_extract(value, '$[1]') AS latest_end
             FROM json_each(?1)
         ) ep ON ep.session_id = ev.session_id
         WHERE ev.event_type IN ('user_message', 'assistant_response', 'tool_result')
           AND (ep.latest_end IS NULL OR ev.created_at > ep.latest_end)
           {filter}
         GROUP BY ev.session_id
         {having}"
    )
}

#[tokio::test]
async fn uncaptured_session_scan_matches_the_left_join_it_replaced() {
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    crate::db::migrations::migrate_events(&pool).await.unwrap();
    let at = |minute: u32| format!("2026-01-01T10:{minute:02}:00+00:00");
    // (session, conversation events at minutes, latest episode end)
    let sessions: [(&str, &[u32], Option<u32>); 6] = [
        ("no-episode-many", &[1, 2, 3, 4, 5, 6], None),
        ("no-episode-few", &[1, 2], None),
        ("episode-then-many", &[1, 2, 11, 12, 13, 14, 15], Some(10)),
        ("episode-then-few", &[1, 2, 3, 4, 5, 11], Some(10)),
        ("episode-covers-all", &[1, 2, 3, 4, 5, 6], Some(30)),
        ("recent-no-episode", &[50, 51, 52, 53, 54], None),
    ];
    let mut ends = Vec::new();
    for (session, minutes, end) in sessions {
        for (i, minute) in minutes.iter().enumerate() {
            let kind = ["user_message", "assistant_response", "tool_result"][i % 3];
            sqlx::query(
                "INSERT INTO events(session_id, event_type, data, created_at)
                 VALUES (?, ?, '{}', ?)",
            )
            .bind(session)
            .bind(kind)
            .bind(at(*minute))
            .execute(&pool)
            .await
            .unwrap();
        }
        // Bookkeeping events never count toward a session.
        sqlx::query(
            "INSERT INTO events(session_id, event_type, data, created_at)
             VALUES (?, 'task_start', '{}', ?)",
        )
        .bind(session)
        .bind(at(59))
        .execute(&pool)
        .await
        .unwrap();
        if let Some(end) = end {
            ends.push((session.to_string(), at(end)));
        }
    }
    let ends = serde_json::to_string(&ends).unwrap();
    let cutoff = at(40);

    let cases = [
        (
            "AND ev.created_at < ?2",
            "HAVING COUNT(ev.id) >= 5 AND MAX(ev.created_at) < ?2",
        ),
        ("", "HAVING COUNT(ev.id) >= 5"),
    ];
    for (filter, having) in cases {
        let mut got: Vec<String> = sqlx::query_scalar(sqlx::AssertSqlSafe(
            &*uncaptured_sessions_sql(filter, having),
        ))
        .bind(&ends)
        .bind(&cutoff)
        .fetch_all(&pool)
        .await
        .unwrap();
        let mut want: Vec<String> = sqlx::query_scalar(sqlx::AssertSqlSafe(
            &*legacy_uncaptured_sessions_sql(filter, having),
        ))
        .bind(&ends)
        .bind(&cutoff)
        .fetch_all(&pool)
        .await
        .unwrap();
        got.sort();
        want.sort();
        assert_eq!(got, want, "filter={filter:?} having={having:?}");
        assert!(got.contains(&"no-episode-many".to_string()));
        assert!(got.contains(&"episode-then-many".to_string()));
        assert!(!got.contains(&"episode-then-few".to_string()));
        assert!(!got.contains(&"episode-covers-all".to_string()));
    }
}

/// The daemon's own runs talking to themselves are never owner speech:
/// an internal continuation is not among an episode's turns.
#[tokio::test]
async fn internal_continuations_are_left_out_of_episode_turns() {
    let (store, _provider, _jobs, _db) = setup_memory_manager_with_responses(vec![]).await;
    for (index, data) in [
        json!({"content": "Find the synthetic product specifications.", "user_role": "owner"}),
        json!({
            "content": "Worker output from an internal retry.",
            "user_role": "owner",
            "annotations": [{"type": "internal_continuation"}]
        }),
    ]
    .into_iter()
    .enumerate()
    {
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at)
             VALUES ('session-1', 'user_message', ?, ?)",
        )
        .bind(data.to_string())
        .bind((Utc::now() - chrono::Duration::minutes(5 - index as i64)).to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();
    }
    let history = EventHistory::new(store.pool(), Some(store.clone()));
    let messages: Vec<Message> = history
        .session_messages_after("session-1", None, 10)
        .await
        .unwrap();
    let texts: Vec<_> = messages
        .iter()
        .filter_map(|m| m.content.as_deref())
        .collect();
    assert_eq!(texts, ["Find the synthetic product specifications."]);
}
