//! Dialogue state recorded through the daemon's agent.

use chrono::Utc;

use super::dialogue_state::record_dialogue_user_message;
use crate::traits::{DialogueState, Message, OpenRequest, OpenRequestStatus};

#[tokio::test]
async fn internal_continuation_does_not_rotate_open_request() {
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::DialogueStateStore;

    let harness = setup_test_agent(MockProvider::new())
        .await
        .expect("test harness");
    let now = Utc::now();
    let mut state = DialogueState::new("test-session");
    state.open_request = Some(OpenRequest {
        user_message_id: "request-1".to_string(),
        text: "Find the specifications and tradeoffs for the synthetic product.".to_string(),
        status: OpenRequestStatus::InProgress,
        task_id: Some("task-1".to_string()),
        project_scope: None,
        completion_contract: None,
        opened_at: now,
        resolved_at: None,
    });
    harness
        .state
        .upsert_dialogue_state(&state)
        .await
        .expect("persist dialogue state");
    let continuation = Message {
        content: Some(
            "Worker output: https://goo.gle/gemini-cli-auth-docs#workspace-gca".to_string(),
        ),
        annotations: vec![crate::traits::MessageAnnotation::InternalContinuation],
        ..Message::new_runtime("continuation-1", "test-session", "user")
    };

    record_dialogue_user_message(&harness.agent, "test-session", &continuation)
        .await
        .expect("record continuation");

    let persisted = harness
        .state
        .get_dialogue_state("test-session")
        .await
        .expect("read dialogue state")
        .expect("dialogue state");
    let open = persisted.open_request.expect("open request remains");
    assert_eq!(open.user_message_id, "request-1");
    assert!(!open.text.contains("workspace-gca"));
    assert!(persisted.last_user_turn.is_none());
}
