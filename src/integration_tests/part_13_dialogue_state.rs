// ==================== Dialogue State Projection ====================

struct DialogueStateWebSearchTool {
    queries: Arc<tokio::sync::Mutex<Vec<String>>>,
}

impl DialogueStateWebSearchTool {
    fn new(queries: Arc<tokio::sync::Mutex<Vec<String>>>) -> Self {
        Self { queries }
    }
}

#[async_trait::async_trait]
impl crate::traits::Tool for DialogueStateWebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web and return canned results for tests"
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "web_search",
            "description": "Search the web and return canned results for tests",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"],
                "additionalProperties": true
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: serde_json::Value = serde_json::from_str(arguments)?;
        let query = args["query"].as_str().unwrap_or("").to_string();
        self.queries.lock().await.push(query.clone());
        Ok(format!(
            "1. [Result](https://example.com/{})\n   Evidence for {}",
            query.replace(' ', "-"),
            query
        ))
    }
}

#[tokio::test]
async fn test_unanswered_request_followup_uses_dialogue_state_projection() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("I searched for AI news and found several results."),
        MockProvider::text_response("Here is the original answer you asked for."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let session_id = "dialogue_state_followup";

    let _ = harness
        .agent
        .handle_message(
            session_id,
            "What were the deployment regressions in yesterday's rollout?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let _ = harness
        .agent
        .handle_message(
            session_id,
            "You didn't answer my question",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let dialogue_state = harness
        .state
        .get_dialogue_state(session_id)
        .await
        .unwrap()
        .expect("dialogue state should be persisted");
    assert_eq!(
        dialogue_state
            .open_request
            .as_ref()
            .map(|request| request.text.as_str()),
        Some("What were the deployment regressions in yesterday's rollout?")
    );
    assert_eq!(
        dialogue_state.last_user_turn.as_ref().map(|turn| turn.kind),
        Some(crate::traits::UserTurnKind::Followup)
    );

    let call_log = harness.provider.call_log.lock().await;
    let second_call = call_log.last().expect("expected second LLM call");
    let user_message = second_call
        .messages
        .iter()
        .rev()
        .find(|msg| msg.get("role").and_then(|role| role.as_str()) == Some("user"))
        .and_then(|msg| msg.get("content").and_then(|content| content.as_str()))
        .unwrap_or_default();

    assert!(
        user_message.contains("Original request:"),
        "expected combined followup prompt, got: {user_message}"
    );
    assert!(
        user_message.contains("What were the deployment regressions in yesterday's rollout?"),
        "original request missing from followup prompt: {user_message}"
    );
    assert!(
        user_message.contains("Follow-up:"),
        "follow-up marker missing from prompt: {user_message}"
    );
    assert!(
        user_message.contains("You didn't answer my question"),
        "follow-up text missing from prompt: {user_message}"
    );
}

#[tokio::test]
async fn test_schedule_trigger_followup_blocks_off_topic_web_search() {
    let queries = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "Schedules trigger at:\n- 9:00 AM\n- 12:00 PM\n- 6:00 PM\nThese are daily recurring tasks for posting tweets about aidaemon.",
        ),
        MockProvider::tool_call_response(
            "web_search",
            r#"{"query":"top 3 tallest buildings in the world 2025 height"}"#,
        ),
        MockProvider::text_response(
            "I need to check the scheduled run state for that 9:00 AM trigger instead of searching the web.",
        ),
    ]);

    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(DialogueStateWebSearchTool::new(queries.clone()))
            as Arc<dyn crate::traits::Tool>],
        None,
    )
    .await
    .unwrap();
    let session_id = "dialogue_state_schedule_trigger_followup";

    let _ = harness
        .agent
        .handle_message(
            session_id,
            "What times does the tweet posting schedule trigger?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            session_id,
            "Did it trigger the 9:00 am today?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("scheduled run state"),
        "response should pivot back to schedule state instead of external web search: {response}"
    );
    assert!(
        queries.lock().await.is_empty(),
        "off-topic web_search should be blocked before execution"
    );
}

#[tokio::test]
async fn test_new_request_drops_previous_failed_search_exchange_from_prompt() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "I made some progress but wasn't able to fully complete the task.\n\nTry rephrasing your request or providing more specific guidance.",
        ),
        MockProvider::text_response("You have no scheduled tasks right now."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let session_id = "dialogue_state_new_request_prompt_isolation";

    let _ = harness
        .agent
        .handle_message(
            session_id,
            "top 3 tallest buildings in the world 2024 height",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let _ = harness
        .agent
        .handle_message(
            session_id,
            "What are your scheduled tasks?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let second_call = call_log.last().expect("expected second LLM call");

    assert!(
        second_call.messages.iter().any(|msg| {
            msg.get("role").and_then(|role| role.as_str()) == Some("user")
                && msg
                    .get("content")
                    .and_then(|content| content.as_str())
                    .is_some_and(|content| content.contains("scheduled tasks"))
        }),
        "current request should still be present in the prompt"
    );
    assert!(
        !second_call.messages.iter().any(|msg| {
            if msg.get("role").and_then(|role| role.as_str()) == Some("system") {
                return false;
            }
            msg.get("content")
                .and_then(|content| content.as_str())
                .is_some_and(|content| content.contains("tallest buildings"))
        }),
        "fresh requests should not inherit the prior failed search topic: {:?}",
        second_call.messages
    );
    assert!(
        !second_call.messages.iter().any(|msg| {
            if msg.get("role").and_then(|role| role.as_str()) == Some("system") {
                return false;
            }
            msg.get("content")
                .and_then(|content| content.as_str())
                .is_some_and(|content| content.contains("wasn't able to fully complete the task"))
        }),
        "fresh requests should not inherit the prior failure summary: {:?}",
        second_call.messages
    );
}
