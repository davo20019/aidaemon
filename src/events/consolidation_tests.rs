//! Consolidation (`aidaemon_runtime::consolidation`) over the daemon's stores.

use super::*;
use crate::config::ProviderKind;
use crate::events::{DiagnosticSeverity, Event, EventType, TaskStatus};
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::embeddings::EmbeddingService;
use crate::plans::PlanStore;
use crate::state::SqliteStateStore;
use crate::testing::MockProvider;
use aidaemon_runtime::consolidation::*;
use aidaemon_runtime::llm::gateway::SideCallGateway;
use chrono::{Duration, Utc};
use serde_json::json;
use std::sync::Arc;

async fn setup_consolidator_test() -> (
    Arc<SqliteStateStore>,
    Arc<EventStore>,
    Consolidator,
    tempfile::NamedTempFile,
) {
    let db_file = tempfile::NamedTempFile::new().expect("temp db file");
    let db_path = db_file.path().to_str().expect("db path");
    let embedding_service = Arc::new(EmbeddingService::deterministic());
    let state = Arc::new(
        SqliteStateStore::for_tests(db_path, 100, None, embedding_service)
            .await
            .expect("state store"),
    );
    let event_store = Arc::new(EventStore::new(state.pool()).await.expect("event store"));
    let plan_store = Arc::new(PlanStore::new(state.pool()).await.expect("plan store"));
    let consolidator = Consolidator::new(
        event_store.clone(),
        plan_store,
        state.pool(),
        state.memory_stores(),
        None,
        unused_side_calls(&state, &event_store),
        None,
    );
    (state, event_store, consolidator, db_file)
}

/// A gateway for a consolidator that has no model to call.
fn unused_side_calls(
    state: &Arc<SqliteStateStore>,
    event_store: &Arc<EventStore>,
) -> SideCallGateway {
    let runtime = SharedLlmRuntime::new(
        Arc::new(MockProvider::with_responses(vec![])),
        None,
        ProviderKind::OpenaiCompatible,
        "mock".to_string(),
    );
    SideCallGateway::new(runtime, state.clone(), event_store.clone())
}

async fn append_old_procedure_trace(
    event_store: &EventStore,
    session_id: &str,
    task_id: &str,
    execution_origin: Option<&str>,
) {
    let base = Utc::now() - Duration::days(9);
    let mut user_data = json!({
        "content": "Run the repeatable release workflow.",
        "message_id": format!("message-{task_id}"),
        "has_attachments": false,
        "user_role": "owner"
    });
    if let Some(origin) = execution_origin {
        user_data["execution_origin"] = json!(origin);
    }

    let events = vec![
        {
            user_data["task_id"] = json!(task_id);
            (EventType::UserMessage, user_data)
        },
        (
            EventType::TaskStart,
            json!({
                "task_id": task_id,
                "description": format!("release workflow for {task_id}")
            }),
        ),
        (
            EventType::MemoryPolicyCompiled,
            json!({
                "task_id": task_id,
                "access": "allowed",
                "reason_code": "synthetic_test_policy",
                "retrieval_suppressed": false,
                "persistence_suppressed": false
            }),
        ),
        (
            EventType::ToolCall,
            json!({
                "tool_call_id": format!("call-1-{task_id}"),
                "name": "terminal",
                "arguments": {"command": "cargo test"},
                "summary": "run tests",
                "task_id": task_id
            }),
        ),
        (
            EventType::ToolCall,
            json!({
                "tool_call_id": format!("call-2-{task_id}"),
                "name": "terminal",
                "arguments": {"command": "cargo build --release"},
                "summary": "build release",
                "task_id": task_id
            }),
        ),
        (
            EventType::TaskEnd,
            serde_json::to_value(TaskEndData {
                task_id: task_id.to_string(),
                status: TaskStatus::Completed,
                outcome: Some(TaskOutcome::Succeeded),
                duration_secs: 30,
                iterations: 2,
                tool_calls_count: 2,
                error: None,
                summary: Some("release verified".to_string()),
                efficiency: None,
                turn_id: None,
                completion_proof: None,
                harness_eval: None,
            })
            .expect("serialize task end"),
        ),
    ];

    for (offset, (event_type, data)) in events.into_iter().enumerate() {
        let mut event = Event::new(session_id, event_type, data);
        event.created_at = base + Duration::seconds(offset as i64);
        event_store.append(event).await.expect("append old event");
    }
}

/// With another memory system learning, old events must still reach
/// retention, which only deletes what consolidation has marked.
#[tokio::test]
async fn retiring_old_events_marks_them_without_learning() {
    let db_file = tempfile::NamedTempFile::new().expect("temp db file");
    let db_path = db_file.path().to_str().expect("db path");
    let embedding_service = Arc::new(EmbeddingService::deterministic());
    let state = Arc::new(
        SqliteStateStore::for_tests(db_path, 100, None, embedding_service)
            .await
            .expect("state store"),
    );
    let event_store = Arc::new(EventStore::new(state.pool()).await.expect("event store"));
    let plan_store = Arc::new(PlanStore::new(state.pool()).await.expect("plan store"));
    let provider = Arc::new(MockProvider::with_responses(vec![]));
    let runtime = SharedLlmRuntime::new(
        provider.clone(),
        None,
        ProviderKind::OpenaiCompatible,
        "mock".to_string(),
    );
    let consolidator = Arc::new(Consolidator::new(
        event_store.clone(),
        plan_store,
        state.pool(),
        state.memory_stores(),
        Some(runtime.clone()),
        SideCallGateway::new(runtime, state.clone(), event_store.clone()),
        None,
    ));
    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at)
         VALUES ('ordinary-owner-session', 'user_message', '{\"content\":\"hello\"}',
                 '2025-01-01T00:00:00+00:00')",
    )
    .execute(&state.pool())
    .await
    .expect("old event");

    let stats = Pruner::new(event_store, consolidator, 7)
        .retire_old_events()
        .await
        .expect("retire pass");

    assert_eq!(stats.consolidation_errors, 0);
    assert_eq!(provider.call_count().await, 0, "nothing is learned");
    let unmarked: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM events WHERE consolidated_at IS NULL")
            .fetch_one(&state.pool())
            .await
            .expect("count");
    assert_eq!(unmarked, 0);
}

#[tokio::test]
async fn pruner_marks_mandate_and_internal_sessions_without_learning() {
    let db_file = tempfile::NamedTempFile::new().expect("temp db file");
    let db_path = db_file.path().to_str().expect("db path");
    let embedding_service = Arc::new(EmbeddingService::deterministic());
    let state = Arc::new(
        SqliteStateStore::for_tests(db_path, 100, None, embedding_service)
            .await
            .expect("state store"),
    );
    let event_store = Arc::new(EventStore::new(state.pool()).await.expect("event store"));
    let plan_store = Arc::new(PlanStore::new(state.pool()).await.expect("plan store"));
    let provider = Arc::new(MockProvider::with_responses(vec![
        MockProvider::text_response(
            r#"{
                "name":"verified-release-workflow",
                "trigger_pattern":"when preparing a verified release",
                "steps":["run tests","build release"],
                "skip":false
            }"#,
        ),
    ]));
    let runtime = SharedLlmRuntime::new(
        provider.clone(),
        None,
        ProviderKind::OpenaiCompatible,
        "mock".to_string(),
    );
    let consolidator = Arc::new(Consolidator::new(
        event_store.clone(),
        plan_store,
        state.pool(),
        state.memory_stores(),
        Some(runtime.clone()),
        SideCallGateway::new(runtime, state.clone(), event_store.clone()),
        None,
    ));

    // Deliberately use an owner-looking session ID for the mandate: the
    // explicit event provenance, not a prefix or role, must isolate it.
    append_old_procedure_trace(
        &event_store,
        "owner-looking-mandate-session",
        "mandate-task",
        Some("mandate"),
    )
    .await;
    append_old_procedure_trace(
        &event_store,
        "specialist:executor:isolated",
        "specialist-task",
        None,
    )
    .await;
    append_old_procedure_trace(
        &event_store,
        "ordinary-owner-session",
        "ordinary-task",
        None,
    )
    .await;
    append_old_procedure_trace(
        &event_store,
        "malformed-origin-session",
        "malformed-task",
        None,
    )
    .await;
    sqlx::query(
        "UPDATE events SET data = '{not-json'
         WHERE session_id = 'malformed-origin-session'
           AND event_type = 'user_message'",
    )
    .execute(&state.pool())
    .await
    .expect("corrupt legacy user-message payload");

    let stats = Pruner::new(event_store.clone(), consolidator, 7)
        .prune()
        .await
        .expect("prune consolidation pass");
    assert_eq!(stats.consolidation_errors, 0);
    assert_eq!(provider.call_count().await, 1);

    let procedures: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM procedures")
        .fetch_one(&state.pool())
        .await
        .expect("count procedures");
    assert_eq!(procedures, 1, "only the ordinary owner session may learn");

    let learned_trigger: String =
        sqlx::query_scalar("SELECT trigger_pattern FROM procedures LIMIT 1")
            .fetch_one(&state.pool())
            .await
            .expect("ordinary procedure");
    assert_eq!(learned_trigger, "when preparing a verified release");

    for session_id in [
        "owner-looking-mandate-session",
        "specialist:executor:isolated",
        "malformed-origin-session",
        "ordinary-owner-session",
    ] {
        assert!(
            event_store
                .query_unconsolidated(session_id)
                .await
                .expect("query unconsolidated")
                .is_empty(),
            "{session_id} should not remain a perpetual prune candidate"
        );
    }
}

/// "Don't remember this": consolidation later passes over an opted-out
/// turn's events without learning, and still learns from a normal turn.
#[tokio::test]
async fn consolidation_skips_a_turn_opted_out_of_learning() {
    let (state, event_store, consolidator, _db_file) = setup_consolidator_test().await;
    append_old_procedure_trace(&event_store, "opted-out-session", "private-task", None).await;
    append_old_procedure_trace(&event_store, "normal-session", "normal-task", None).await;
    crate::agent::memory_plugin::mark_turn_opted_out(
        &crate::events::EventEmitter::new(event_store.clone(), "opted-out-session"),
        "private-task",
        "message-private-task",
    )
    .await
    .expect("mark opted out");

    let skipped = consolidator
        .consolidate_session("opted-out-session")
        .await
        .expect("consolidate opted-out session");
    assert_eq!(skipped.procedures_created, 0);
    assert!(
        event_store
            .query_unconsolidated("opted-out-session")
            .await
            .unwrap()
            .is_empty()
    );
    let procedures: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM procedures")
        .fetch_one(&state.pool())
        .await
        .unwrap();
    assert_eq!(procedures, 0, "nothing learned from the opted-out turn");

    let learned = consolidator
        .consolidate_session("normal-session")
        .await
        .expect("consolidate normal session");
    assert_eq!(learned.procedures_created, 1, "a normal turn still learns");
}

#[tokio::test]
async fn test_consolidation_records_failure_patterns_from_warning_decisions() {
    let (state, event_store, consolidator, _db_file) = setup_consolidator_test().await;

    let task_id = "task-failure-pattern";
    let mut task_start = Event::new(
        "session-failure-pattern",
        EventType::TaskStart,
        json!({
            "task_id": task_id,
            "description": "debug repetitive terminal loop"
        }),
    );
    task_start.created_at = Utc::now();
    event_store
        .append(task_start)
        .await
        .expect("append task start");

    event_store
        .append(Event::new(
            "session-failure-pattern",
            EventType::MemoryPolicyCompiled,
            json!({
                "task_id": task_id,
                "access": "allowed",
                "reason_code": "synthetic_test_policy",
                "retrieval_suppressed": false,
                "persistence_suppressed": false
            }),
        ))
        .await
        .expect("append memory policy");

    let mut decision = Event::new(
        "session-failure-pattern",
        EventType::DecisionPoint,
        serde_json::to_value(DecisionPointData {
            decision_type: crate::events::DecisionType::RepetitiveCallDetection,
            task_id: task_id.to_string(),
            iteration: 4,
            severity: DiagnosticSeverity::Warning,
            code: Some("repetitive_call_detection".to_string()),
            metadata: json!({
                "tool": "terminal",
                "count": 4,
                "action": "hard_stop"
            }),
            summary: "Repetitive tool call hard-stopped for terminal (count=4)".to_string(),
        })
        .expect("serialize decision"),
    );
    decision.created_at = Utc::now();
    event_store.append(decision).await.expect("append decision");

    let mut task_end = Event::new(
        "session-failure-pattern",
        EventType::TaskEnd,
        serde_json::to_value(TaskEndData {
            task_id: task_id.to_string(),
            status: TaskStatus::Failed,
            outcome: Some(crate::events::TaskOutcome::Failed),
            duration_secs: 3,
            iterations: 4,
            tool_calls_count: 4,
            error: Some("Repetitive tool calls".to_string()),
            summary: Some("Agent stopped due to repetitive terminal loop".to_string()),
            efficiency: None,
            turn_id: None,
            completion_proof: None,
            harness_eval: None,
        })
        .expect("serialize task end"),
    );
    task_end.created_at = Utc::now();
    event_store.append(task_end).await.expect("append task end");

    let result = consolidator
        .consolidate_session("session-failure-pattern")
        .await
        .expect("consolidate session");
    assert_eq!(result.behavior_patterns_recorded, 1);

    let patterns = state
        .get_behavior_patterns(0.0)
        .await
        .expect("get behavior patterns");
    let pattern = patterns
        .iter()
        .find(|pattern| {
            pattern.trigger_context.as_deref() == Some("repetitive_call_detection:terminal")
        })
        .expect("repetitive failure pattern");
    assert_eq!(pattern.pattern_type, "failure");
    assert_eq!(
        pattern.action.as_deref(),
        Some("stop_retrying_identical_calls")
    );
    assert!(
        pattern
            .description
            .contains("stop retrying the same operation"),
        "pattern description was: {}",
        pattern.description
    );
    assert!(pattern.confidence >= 0.8);
}

#[tokio::test]
async fn test_consolidation_records_evidence_gate_failure_patterns() {
    let (state, event_store, consolidator, _db_file) = setup_consolidator_test().await;

    let task_id = "task-evidence-pattern";
    let mut task_start = Event::new(
        "session-evidence-pattern",
        EventType::TaskStart,
        json!({
            "task_id": task_id,
            "description": "edit src/main.rs"
        }),
    );
    task_start.created_at = Utc::now();
    event_store
        .append(task_start)
        .await
        .expect("append task start");

    event_store
        .append(Event::new(
            "session-evidence-pattern",
            EventType::MemoryPolicyCompiled,
            json!({
                "task_id": task_id,
                "access": "allowed",
                "reason_code": "synthetic_test_policy",
                "retrieval_suppressed": false,
                "persistence_suppressed": false
            }),
        ))
        .await
        .expect("append memory policy");

    let mut decision = Event::new(
        "session-evidence-pattern",
        EventType::DecisionPoint,
        serde_json::to_value(DecisionPointData {
            decision_type: crate::events::DecisionType::EvidenceGate,
            task_id: task_id.to_string(),
            iteration: 1,
            severity: DiagnosticSeverity::Warning,
            code: Some("missing_pre_execution_evidence".to_string()),
            metadata: json!({
                "condition": "missing_pre_execution_evidence",
                "tool": "edit_file",
                "required_evidence_kind": "file_read",
                "target": "src/main.rs"
            }),
            summary: "Blocked edit_file until required evidence is gathered".to_string(),
        })
        .expect("serialize decision"),
    );
    decision.created_at = Utc::now();
    event_store.append(decision).await.expect("append decision");

    let mut task_end = Event::new(
        "session-evidence-pattern",
        EventType::TaskEnd,
        serde_json::to_value(TaskEndData {
            task_id: task_id.to_string(),
            status: TaskStatus::Failed,
            outcome: Some(crate::events::TaskOutcome::Failed),
            duration_secs: 2,
            iterations: 1,
            tool_calls_count: 0,
            error: Some("missing evidence".to_string()),
            summary: None,
            efficiency: None,
            turn_id: None,
            completion_proof: None,
            harness_eval: None,
        })
        .expect("serialize task end"),
    );
    task_end.created_at = Utc::now();
    event_store.append(task_end).await.expect("append task end");

    let result = consolidator
        .consolidate_session("session-evidence-pattern")
        .await
        .expect("consolidate session");
    assert_eq!(result.behavior_patterns_recorded, 1);

    let patterns = state
        .get_behavior_patterns(0.0)
        .await
        .expect("get behavior patterns");
    let pattern = patterns
        .iter()
        .find(|pattern| {
            pattern.trigger_context.as_deref()
                == Some("missing_pre_execution_evidence:edit_file:file_read")
        })
        .expect("evidence gate failure pattern");
    assert_eq!(pattern.pattern_type, "failure");
    assert_eq!(
        pattern.action.as_deref(),
        Some("gather_direct_evidence_before_mutation")
    );
    assert!(pattern.description.contains("gather file_read"));
}
