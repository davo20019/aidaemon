use super::*;
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
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("Resumed and done.")]);
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
            },
        )
        .await
        .unwrap();

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
    let system_prompt = first_call
        .messages
        .iter()
        .find_map(|msg| {
            if msg.get("role").and_then(|v| v.as_str()) == Some("system") {
                return msg
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
            }
            None
        })
        .expect("expected system prompt");
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
    assert_eq!(orphan_end_data.status, TaskStatus::Failed);
    assert!(
        orphan_end_data
            .error
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
