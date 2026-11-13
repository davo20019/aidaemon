//! `search_history` lives in `aidaemon_runtime::tools::search_history`.

pub use aidaemon_runtime::tools::search_history::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventStoreForTests as _;
    use crate::events::{Event, EventStore, EventType};
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::testing::CallWithFacts;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolSemantics;
    use crate::types::ChannelVisibility;
    use crate::types::UserRole;
    use serde_json::json;
    use serde_json::Value;
    use std::sync::Arc;

    async fn tool() -> (SearchHistoryTool, Arc<EventStore>, tempfile::NamedTempFile) {
        let db = tempfile::NamedTempFile::new().unwrap();
        let path = db.path().to_path_buf();
        let embedding = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(path.to_str().unwrap(), 50, None, embedding)
                .await
                .unwrap(),
        );
        let events = Arc::new(EventStore::new(state.pool()).await.unwrap());
        (SearchHistoryTool::new(state, 90), events, db)
    }

    async fn append_tool_turn(
        events: &EventStore,
        session: &str,
        turn: &str,
        message: &str,
        output: Option<&str>,
    ) {
        let turn_id = |mut event: Event| {
            event.turn_id = Some(turn.to_string());
            event
        };
        events
            .append(turn_id(Event::new(
                session,
                EventType::UserMessage,
                json!({
                    "content": format!("request {turn}"),
                    "message_id": message,
                    "channel_visibility": "private",
                    "user_role": "owner",
                }),
            )))
            .await
            .unwrap();
        if let Some(output) = output {
            events
                .append(turn_id(Event::new(
                    session,
                    EventType::ToolCall,
                    json!({"tool_call_id": format!("call-{turn}"), "name": "terminal",
                           "arguments": {"command": "echo synthetic"}}),
                )))
                .await
                .unwrap();
            events
                .append(turn_id(Event::new(
                    session,
                    EventType::ToolResult,
                    json!({"tool_call_id": format!("call-{turn}"), "name": "terminal",
                           "result": output, "success": true, "duration_ms": 1}),
                )))
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn opening_a_turn_returns_its_tool_calls_and_output() {
        let (tool, events, _db) = tool().await;
        append_tool_turn(
            &events,
            "root-session",
            "t1",
            "m1",
            Some("synthetic-output-417"),
        )
        .await;
        let context = |session: &str, role: UserRole| crate::traits::ToolCallContext {
            session_id: session.into(),
            user_role: Some(role),
            channel_visibility: Some(ChannelVisibility::Private),
            ..Default::default()
        };
        // As a strict-schema model sends it: every unused field blank.
        let open = json!({
            "action": "open", "turn_id": "t1", "event_id": 0, "message_id": "",
            "query": "", "cursor": "", "session_id": "", "task_id": "",
            "limit": 10, "context": 0, "include_subagents": false
        })
        .to_string();
        let opened = tool
            .call_with_execution_context(&open, None, context("root-session", UserRole::Owner))
            .await
            .unwrap();
        assert!(
            opened.output.contains("synthetic-output-417"),
            "{}",
            opened.output
        );
        assert!(
            opened.output.contains("echo synthetic"),
            "{}",
            opened.output
        );

        let guest = tool
            .call_with_execution_context(&open, None, context("root-session", UserRole::Guest))
            .await;
        assert!(guest.is_err() || !guest.unwrap().output.contains("synthetic-output-417"));
    }

    #[tokio::test]
    async fn the_note_names_prior_turns_that_ran_tools_and_nothing_they_said() {
        let (tool, events, _db) = tool().await;
        assert!(tool.session_context_note("root-session").await.is_none());
        append_tool_turn(
            &events,
            "root-session",
            "t1",
            "m1",
            Some("synthetic-output-417"),
        )
        .await;
        append_tool_turn(&events, "root-session", "t2", "m2", None).await;
        append_tool_turn(&events, "other-session", "t9", "m9", Some("other")).await;
        append_tool_turn(&events, "root-session", "t3", "m3", None).await;

        let note = tool
            .session_context_note("root-session")
            .await
            .expect("t1 ran a tool");
        assert!(
            note.contains("2 turns ago (turn_id=t1): terminal x1"),
            "{note}"
        );
        assert!(!note.contains("synthetic-output-417"), "{note}");
        assert!(!note.contains("t9"), "{note}");
    }

    /// One tool result stamped with the typed stage its receipt carries.
    async fn append_tool_result(events: &EventStore, turn: &str, call: &str, stage: &str) {
        let mut event = Event::new(
            "root-session",
            EventType::ToolResult,
            json!({"tool_call_id": call, "name": "terminal", "result": "", "success": false,
                   "duration_ms": 0, "receipt": {"invocation_stage": stage}}),
        );
        event.turn_id = Some(turn.to_string());
        events.append(event).await.unwrap();
    }

    #[tokio::test]
    async fn the_note_says_a_refused_call_never_ran() {
        let (tool, events, _db) = tool().await;
        append_tool_turn(&events, "root-session", "t1", "m1", Some("output")).await;
        append_tool_result(&events, "t1", "c2", "rejected_before_dispatch").await;
        append_tool_turn(&events, "root-session", "t2", "m2", None).await;
        append_tool_result(&events, "t2", "c3", "rejected_before_dispatch").await;
        append_tool_turn(&events, "root-session", "t3", "m3", None).await;

        let note = tool.session_context_note("root-session").await.unwrap();
        assert!(
            note.contains("(turn_id=t1): terminal x1; refused before running: terminal x1"),
            "{note}"
        );
        assert!(
            note.contains("(turn_id=t2): refused before running: terminal x1"),
            "{note}"
        );
    }

    #[tokio::test]
    async fn the_note_lists_a_turn_of_merged_messages_once() {
        let (tool, events, _db) = tool().await;
        append_tool_turn(&events, "root-session", "t1", "m1", None).await;
        append_tool_turn(&events, "root-session", "t1", "m2", Some("output")).await;
        append_tool_turn(&events, "root-session", "t2", "m3", None).await;

        let note = tool.session_context_note("root-session").await.unwrap();
        assert_eq!(note.matches("turn_id=t1").count(), 1, "{note}");
        assert!(note.contains("previous turn (turn_id=t1)"), "{note}");
    }

    #[tokio::test]
    async fn exact_output_is_ephemeral_and_cursor_cannot_change_runtime_scope() {
        let (tool, events, _db) = tool().await;
        events
            .append(Event::new(
                "root-session",
                EventType::UserMessage,
                json!({
                    "content": "needle verbatim secret",
                    "message_id": "m1",
                    "channel_visibility": "private",
                    "user_role": "owner",
                    "turn_id": "t1"
                }),
            ))
            .await
            .unwrap();
        events
            .append(Event::new(
                "other-session",
                EventType::AssistantResponse,
                json!({
                    "content": "interleaved other-session message",
                    "message_id": "m-other"
                }),
            ))
            .await
            .unwrap();
        events
            .append(Event::new(
                "root-session",
                EventType::AssistantResponse,
                json!({
                    "content": "same-session followup",
                    "message_id": "m2"
                }),
            ))
            .await
            .unwrap();

        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "action": "search",
                    "query": "needle"
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "root-session".into(),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(outcome.output.contains("verbatim secret"));
        assert!(outcome.metadata.untrusted_verbatim);
        assert!(outcome.metadata.preserve_inline);
        assert!(outcome.metadata.suppress_activity_result);
        assert!(!outcome
            .metadata
            .persistent_output
            .as_deref()
            .unwrap()
            .contains("verbatim secret"));

        let value: Value = serde_json::from_str(&outcome.output).unwrap();
        let cursor = value["matches"][0]["older_cursor"].as_str().unwrap();
        let newer_cursor = value["matches"][0]["newer_cursor"].as_str().unwrap();
        let page = tool
            .call_with_execution_context(
                &json!({
                    "action": "page",
                    "cursor": newer_cursor
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "root-session".into(),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(page.output.contains("same-session followup"));
        assert!(!page.output.contains("interleaved other-session message"));

        let widened = tool
            .call_with_execution_context(
                &json!({
                    "action": "page",
                    "cursor": cursor
                })
                .to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "different-session".into(),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    user_role: Some(crate::types::UserRole::Owner),
                    ..Default::default()
                },
            )
            .await;
        assert!(widened.is_err());
    }

    #[tokio::test]
    async fn scoped_contexts_hide_global_projection_metadata_and_cannot_repair() {
        let (tool, _events, _db) = tool().await;
        let scoped_args = json!({ "action": "health" });
        let facts = crate::traits::ToolCallContext {
            session_id: "slack:C1".into(),
            channel_id: Some("slack:C1".into()),
            channel_visibility: Some(ChannelVisibility::PrivateGroup),
            user_role: Some(UserRole::Owner),
            ..Default::default()
        };
        let health = tool
            .call_with(&scoped_args.to_string(), facts.clone())
            .await
            .unwrap();
        assert!(!health.contains("canonical_messages"));
        assert!(health.contains("counts are hidden"));

        let mut repair_args = scoped_args;
        repair_args["action"] = json!("repair");
        assert!(tool
            .call_with(&repair_args.to_string(), facts)
            .await
            .is_err());
    }
}
