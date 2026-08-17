use super::*;
use crate::events::ToolOutcomeEvidenceSource;
use crate::testing::{setup_test_agent, MockProvider};
use crate::types::{ChannelContext, UserRole};
use serde_json::json;

#[test]
fn test_is_resume_request_detects_continue_variants() {
    assert!(is_resume_request("continue"));
    assert!(is_resume_request("Continue with next phase"));
    assert!(is_resume_request("resume the previous task"));
    assert!(is_resume_request("next phase"));
    assert!(!is_resume_request("How do I continue learning Rust?"));
}

#[tokio::test]
async fn test_continue_injects_resume_checkpoint_and_closes_orphan_task() {
    let provider = MockProvider::with_responses(vec![
        // A typed permanent failure cannot be closed by an unsupported text
        // assertion. The first resumed turn selects another executable step.
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Resumed and done."),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();
    let session_id = "resume_session";
    let orphan_task_id = "task-orphan-1";

    let emitter =
        crate::events::EventEmitter::new(harness.agent.event_store.clone(), session_id.to_string())
            .with_task_id(orphan_task_id.to_string());

    emitter
        .emit(
            EventType::TaskStart,
            TaskStartData {
                task_id: orphan_task_id.to_string(),
                description: "Build website and deploy".to_string(),
                parent_task_id: None,
                user_message: Some("Build website and deploy".to_string()),
                turn_id: None,
            },
        )
        .await
        .unwrap();
    emitter
        .emit(
            EventType::ThinkingStart,
            ThinkingStartData {
                iteration: 2,
                task_id: orphan_task_id.to_string(),
                total_tool_calls: 1,
            },
        )
        .await
        .unwrap();
    emitter
        .emit(
            EventType::AssistantResponse,
            AssistantResponseData {
                message_id: None,
                content: Some("I'll continue by checking the config.".to_string()),
                tool_calls: Some(vec![ToolCallInfo {
                    id: "call_pending".to_string(),
                    name: "system_info".to_string(),
                    arguments: json!({}),
                    extra_content: None,
                }]),
                model: "mock-model".to_string(),
                input_tokens: None,
                output_tokens: None,
                annotations: Vec::new(),
                turn_id: None,
                task_id: Some(orphan_task_id.to_string()),
                referenced_receipts: Vec::new(),
            },
        )
        .await
        .unwrap();
    emitter
        .emit(
            EventType::ToolResult,
            ToolResultData {
                message_id: None,
                tool_call_id: "call_done".to_string(),
                name: "system_info".to_string(),
                result: "ok".to_string(),
                success: true,
                duration_ms: 12,
                error: None,
                task_id: Some(orphan_task_id.to_string()),
                annotations: Vec::new(),
                turn_id: None,
                attachments: Vec::new(),
                receipt: None,
            },
        )
        .await
        .unwrap();
    emitter
        .emit(
            EventType::DecisionPoint,
            DecisionPointData {
                decision_type: DecisionType::ExecutionStateSnapshot,
                task_id: orphan_task_id.to_string(),
                iteration: 2,
                severity: crate::events::DiagnosticSeverity::Info,
                code: Some("step_completed".to_string()),
                metadata: json!({
                    "condition": "step_completed",
                    "execution_state": {
                        "execution_id": "exec-orphan-1",
                        "current_step": {
                            "step_id": "step-2-call_pending",
                            "description": "Run `system_info` against /tmp/demo",
                            "plan_version": 1,
                            "primary_tool": "system_info",
                            "expected_effect": "observation",
                            "target_scope": {
                                "allowed_targets": [{"kind":"path","value":"/tmp/demo"}],
                                "hard_fail_outside_scope": false
                            },
                            "expected_targets": [{"kind":"path","value":"/tmp/demo"}],
                            "retry_policy": {
                                "max_attempts": 1,
                                "allow_tool_invocation_retry": false
                            },
                            "approval_requirement": "not_needed",
                            "idempotency_key": null
                        },
                        "attempt_count": 1,
                        "last_tool_name": "system_info",
                        "last_outcome": "progress",
                        "background_handoff_active": false,
                        "persisted_at": null,
                        "budget_tier": "small",
                        "budget": {
                            "max_steps": 4,
                            "max_tokens": 12000,
                            "max_llm_calls": 8,
                            "max_tool_calls": 5,
                            "max_validation_rounds": 2,
                            "max_wall_clock_ms": 180000
                        },
                        "persistence": "durable",
                        "llm_calls_used": 1,
                        "tool_calls_used": 1,
                        "validation_rounds_used": 0,
                        "steps_used": 1
                    }
                }),
                summary: "Captured execution state snapshot".to_string(),
            },
        )
        .await
        .unwrap();
    let receipt = ToolReceiptV1::from_metadata(
        &crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::FailedPermanent),
            semantics: crate::traits::ToolCallSemantics::observation(),
            ..crate::traits::ToolCallMetadata::default()
        },
        crate::traits::ToolOutcomeStatus::FailedPermanent,
        ToolOutcomeEvidenceSource::ToolReported,
        Some("exec:exec-orphan-1:2:system_info:call_pending".to_string()),
    );
    emitter
        .emit(
            EventType::ToolResult,
            ToolResultData {
                message_id: None,
                tool_call_id: "call_pending".to_string(),
                name: "system_info".to_string(),
                result: "legacy success flag is weaker than the receipt".to_string(),
                success: true,
                duration_ms: 8,
                error: None,
                task_id: Some(orphan_task_id.to_string()),
                annotations: Vec::new(),
                turn_id: None,
                attachments: Vec::new(),
                receipt: Some(receipt),
            },
        )
        .await
        .unwrap();

    let checkpoint = crate::agent::resume::build_resume_checkpoint(&harness.agent, session_id)
        .await
        .unwrap()
        .expect("expected resume checkpoint");
    let execution_snapshot = checkpoint
        .execution_snapshot
        .as_ref()
        .expect("expected execution snapshot on resume checkpoint");
    assert_eq!(execution_snapshot.execution_id, "exec-orphan-1");
    assert_eq!(
        execution_snapshot.current_step_id.as_deref(),
        Some("step-2-call_pending")
    );
    assert_eq!(
        execution_snapshot.current_tool.as_deref(),
        Some("system_info")
    );
    assert_eq!(
        execution_snapshot.current_target.as_deref(),
        Some("/tmp/demo")
    );
    assert_eq!(
        execution_snapshot.last_outcome,
        Some(crate::agent::execution_state::StepExecutionOutcome::NonrecoverableFailure),
        "typed receipt must override stale snapshot/legacy success fields"
    );
    assert_eq!(
        execution_snapshot.idempotency_key.as_deref(),
        Some("exec:exec-orphan-1:2:system_info:call_pending")
    );

    let reply = harness
        .agent
        .handle_message(
            session_id,
            "continue",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(reply, "Resumed and done.");

    let calls = harness.provider.call_log.lock().await;
    assert!(!calls.is_empty());
    let first_call = &calls[0];
    // Pillar A: the resume checkpoint moved from the core to the per-task context
    // tail (also a system message), so scan ALL system messages.
    let system_prompt: String = first_call
        .messages
        .iter()
        .filter(|msg| msg.get("role").and_then(|v| v.as_str()) == Some("system"))
        .filter_map(|msg| msg.get("content").and_then(|v| v.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(system_prompt.contains("## Resume Checkpoint"));
    assert!(system_prompt.contains(orphan_task_id));

    let orphan_events = harness
        .agent
        .event_store
        .query_task_events_for_session(session_id, orphan_task_id)
        .await
        .unwrap();
    let orphan_end = orphan_events
        .iter()
        .find(|e| e.event_type == EventType::TaskEnd)
        .expect("expected orphan task_end after resume");
    let orphan_end_data = orphan_end.parse_data::<TaskEndData>().unwrap();
    assert_eq!(orphan_end_data.status, TaskStatus::Interrupted);
    assert_eq!(
        orphan_end_data.effective_outcome(),
        crate::events::TaskOutcome::Partial
    );
    // Legacy case: the orphan TaskStart had turn_id = None, so the recovery
    // TaskEnd must NOT borrow the new resume turn — it stays None.
    assert!(
        orphan_end_data.turn_id.is_none(),
        "legacy recovery TaskEnd must keep turn_id = None, not borrow resume turn"
    );
    assert!(
        orphan_end_data
            .summary
            .unwrap_or_default()
            .contains("Resumed in task"),
        "expected interruption reason to reference resumed task"
    );

    let starts = harness
        .agent
        .event_store
        .query_events_by_types(session_id, &[EventType::TaskStart], 10)
        .await
        .unwrap();
    let resumed_start = starts.into_iter().find_map(|event| {
        let data = event.parse_data::<TaskStartData>().ok()?;
        if data.parent_task_id.as_deref() == Some(orphan_task_id) {
            Some(data)
        } else {
            None
        }
    });
    assert!(
        resumed_start.is_some(),
        "expected resumed task_start to reference orphan as parent"
    );
}

#[tokio::test]
async fn recovery_task_end_uses_checkpoint_turn_id_not_resume_turn() {
    // An interrupted task whose original turn is `turn-old`. The recovery
    // TaskEnd MUST attribute to turn-old, never the new resume turn.
    let provider = MockProvider::new();
    let harness = setup_test_agent(provider).await.unwrap();
    let session_id = "resume_turn_session";
    let orphan_task_id = "task-orphan-turnid";

    let emitter =
        crate::events::EventEmitter::new(harness.agent.event_store.clone(), session_id.to_string())
            .with_task_id(orphan_task_id.to_string());
    emitter
        .emit(
            EventType::TaskStart,
            TaskStartData {
                task_id: orphan_task_id.to_string(),
                description: "Interrupted work".to_string(),
                parent_task_id: None,
                user_message: Some("do the work".to_string()),
                turn_id: Some("turn-old".to_string()),
            },
        )
        .await
        .unwrap();

    let checkpoint = crate::agent::resume::build_resume_checkpoint(&harness.agent, session_id)
        .await
        .unwrap()
        .expect("expected resume checkpoint");
    assert_eq!(
        checkpoint.turn_id.as_deref(),
        Some("turn-old"),
        "checkpoint carries the interrupted task's original turn_id"
    );

    crate::agent::resume::mark_task_interrupted_for_resume(
        &harness.agent,
        session_id,
        &checkpoint,
        "task-resume-new",
    )
    .await;

    let orphan_events = harness
        .agent
        .event_store
        .query_task_events_for_session(session_id, orphan_task_id)
        .await
        .unwrap();
    let orphan_end = orphan_events
        .iter()
        .find(|e| e.event_type == EventType::TaskEnd)
        .expect("expected recovery task_end");
    // The first-class indexed column must carry turn-old.
    assert_eq!(orphan_end.turn_id.as_deref(), Some("turn-old"));
    let end_data = orphan_end.parse_data::<TaskEndData>().unwrap();
    assert_eq!(end_data.turn_id.as_deref(), Some("turn-old"));
}

#[tokio::test]
async fn normal_task_stamps_active_turn_on_conversation_and_task_events() {
    // Drive a normal task to completion and assert UserMessage,
    // AssistantResponse, TaskStart, and TaskEnd all carry the same active
    // turn UUID (the turn's opening user-message id).
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("All done.")]);
    let harness = setup_test_agent(provider).await.unwrap();
    let session_id = "normal_turn_session";

    let reply = harness
        .agent
        .handle_message(
            session_id,
            "please do a thing",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(reply, "All done.");

    let events = harness
        .agent
        .event_store
        .query_events(session_id, chrono::Utc::now() - chrono::Duration::days(1))
        .await
        .unwrap();

    let user_turn = events
        .iter()
        .find(|e| e.event_type == EventType::UserMessage)
        .and_then(|e| e.turn_id.clone())
        .expect("UserMessage must carry a turn_id");

    for et in [
        EventType::AssistantResponse,
        EventType::TaskStart,
        EventType::TaskEnd,
    ] {
        let ev = events
            .iter()
            .find(|e| e.event_type == et)
            .unwrap_or_else(|| panic!("missing {:?} event", et));
        assert_eq!(
            ev.turn_id.as_deref(),
            Some(user_turn.as_str()),
            "{:?} must carry the active turn UUID",
            et
        );
    }
}
