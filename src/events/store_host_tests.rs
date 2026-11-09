//! The event store with the daemon's seams: error grouping that strips the
//! daemon's own wrappers, and memory-span projection.

use chrono::{DateTime, Duration, Utc};
use serde_json::json;
use sqlx::SqlitePool;

use super::*;

async fn setup_store() -> (EventStore, tempfile::NamedTempFile) {
    let db_file = tempfile::NamedTempFile::new().expect("temp db file");
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.expect("connect sqlite");
    let store = EventStore::new(pool).await.expect("init event store");
    (store, db_file)
}

async fn append_tool_result(
    store: &EventStore,
    session_id: &str,
    fixture: ToolResultFixture<'_>,
) {
    let mut payload = json!({
        "tool_call_id": format!(
            "tc-{}-{}",
            fixture.tool,
            fixture.created_at.timestamp_nanos_opt().unwrap_or(0)
        ),
        "name": fixture.tool,
        "result": fixture.result,
        "success": fixture.success,
        "duration_ms": fixture.duration_ms,
    });
    if let Some(err) = fixture.error {
        payload["error"] = json!(err);
    }
    append_event_at(
        store,
        session_id,
        EventType::ToolResult,
        payload,
        fixture.created_at,
    )
    .await;
}

#[tokio::test]
async fn user_message_memory_projection_requires_explicit_allowed_stage() {
    let (store, _database) = setup_store().await;
    sqlx::query(
        "CREATE TABLE memory_spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            span_kind TEXT NOT NULL,
            source_event_id INTEGER,
            source_episode_id INTEGER,
            session_id TEXT,
            channel_id TEXT,
            role TEXT,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            privacy TEXT NOT NULL DEFAULT 'global',
            observed_from TEXT,
            observed_to TEXT,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            deleted_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(&store.pool())
    .await
    .unwrap();
    sqlx::query(
        "CREATE UNIQUE INDEX idx_memory_spans_source_event
         ON memory_spans(source_event_id) WHERE source_event_id IS NOT NULL",
    )
    .execute(&store.pool())
    .await
    .unwrap();
    let event_id = store
        .append(Event::new(
            "session-memory-boundary",
            EventType::UserMessage,
            json!({
                "task_id": "task-memory-boundary",
                "turn_id": "turn-memory-boundary",
                "content": "Synthetic content that must stay outside automatic memory"
            }),
        ))
        .await
        .unwrap();

    let before_projection: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_spans WHERE source_event_id = ?")
            .bind(event_id)
            .fetch_one(&store.pool())
            .await
            .unwrap();
    assert_eq!(
        before_projection, 0,
        "canonical append must not imply memory persistence"
    );
    store
        .project_selected_user_message_memory_span_for_task("task-memory-boundary")
        .await
        .unwrap();
    let while_policy_pending: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_spans WHERE source_event_id = ?")
            .bind(event_id)
            .fetch_one(&store.pool())
            .await
            .unwrap();
    assert_eq!(
        while_policy_pending, 0,
        "projection fails closed until the task has a durable allowed policy"
    );

    store
        .append(Event::new(
            "session-memory-boundary",
            EventType::MemoryPolicyCompiled,
            json!({
                "task_id": "task-memory-boundary",
                "turn_id": "turn-memory-boundary",
                "access": "suppressed",
                "reason_code": "current_assessment_denies_user_memory",
                "retrieval_suppressed": true,
                "persistence_suppressed": true
            }),
        ))
        .await
        .unwrap();
    store
        .project_selected_user_message_memory_span_for_task("task-memory-boundary")
        .await
        .unwrap();

    let after_suppressed_projection: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memory_spans WHERE source_event_id = ?")
            .bind(event_id)
            .fetch_one(&store.pool())
            .await
            .unwrap();
    assert_eq!(after_suppressed_projection, 0);

    let allowed_event_id = store
        .append(Event::new(
            "session-memory-allowed",
            EventType::UserMessage,
            json!({
                "task_id": "task-memory-allowed",
                "turn_id": "turn-memory-allowed",
                "content": "My synthetic preference is dark mode"
            }),
        ))
        .await
        .unwrap();
    store
        .append(Event::new(
            "session-memory-allowed",
            EventType::MemoryPolicyCompiled,
            json!({
                "task_id": "task-memory-allowed",
                "turn_id": "turn-memory-allowed",
                "access": "allowed",
                "reason_code": "ordinary_allowed_memory",
                "retrieval_suppressed": false,
                "persistence_suppressed": false
            }),
        ))
        .await
        .unwrap();
    assert_eq!(
        store
            .project_selected_user_message_memory_span_for_task("task-memory-allowed")
            .await
            .unwrap(),
        Some(allowed_event_id)
    );
}

#[tokio::test]
async fn get_tool_stats_aggregates_and_groups_errors() {
    let (store, _db_file) = setup_store().await;
    let now = Utc::now();
    let session = "s-tool-stats-1";

    append_tool_result(
        &store,
        session,
        ToolResultFixture {
            tool: "terminal",
            success: true,
            duration_ms: 100,
            result: "ok",
            error: None,
            created_at: now - Duration::minutes(50),
        },
    )
    .await;
    append_tool_result(
        &store,
        session,
        ToolResultFixture {
            tool: "terminal",
            success: true,
            duration_ms: 300,
            result: "ok",
            error: None,
            created_at: now - Duration::minutes(40),
        },
    )
    .await;
    append_tool_result(
        &store,
        session,
        ToolResultFixture {
            tool: "terminal",
            success: false,
            duration_ms: 200,
            result: "Error: Connection timed out at /tmp/foo.rs:12:3",
            error: Some("Error: Connection timed out at /tmp/foo.rs:12:3"),
            created_at: now - Duration::minutes(30),
        },
    )
    .await;
    append_tool_result(
        &store,
        session,
        ToolResultFixture {
            tool: "terminal",
            success: false,
            duration_ms: 400,
            result: "Error: Connection timed out at /tmp/bar.rs:99:1",
            error: Some("Error: Connection timed out at /tmp/bar.rs:99:1"),
            created_at: now - Duration::minutes(20),
        },
    )
    .await;

    let stats = store
        .get_tool_stats("terminal", now - Duration::hours(24))
        .await
        .expect("tool stats");

    assert_eq!(stats.total_calls, 4);
    assert_eq!(stats.successful, 2);
    assert_eq!(stats.failed, 2);
    assert_eq!(stats.avg_duration_ms, 250);
    assert_eq!(stats.common_errors.len(), 1);
    assert_eq!(stats.common_errors[0].1, 2);
}

async fn append_event_at(
    store: &EventStore,
    session_id: &str,
    event_type: EventType,
    data: serde_json::Value,
    created_at: DateTime<Utc>,
) {
    let mut event = Event::new(session_id, event_type, data);
    event.created_at = created_at;
    store.append(event).await.expect("append event");
}

struct ToolResultFixture<'a> {
    tool: &'a str,
    success: bool,
    duration_ms: u64,
    result: &'a str,
    error: Option<&'a str>,
    created_at: DateTime<Utc>,
}
