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
async fn test_semantic_followup_uses_dialogue_state_projection() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("I searched for AI news and found several results."),
        MockProvider::text_response("Here is the original answer you asked for."),
    ])
    .with_task_assessments(vec![
        MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "new_request",
            "none",
        ),
        MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "continuation",
            "none",
        ),
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
        Some("You didn't answer my question")
    );
    assert_eq!(
        dialogue_state.last_user_turn.as_ref().map(|turn| turn.kind),
        Some(crate::traits::UserTurnKind::Followup)
    );

    let call_log = harness.provider.call_log.lock().await;
    let second_call = call_log.last().expect("expected second LLM call");
    let user_messages = second_call
        .messages
        .iter()
        .filter(|msg| msg.get("role").and_then(|role| role.as_str()) == Some("user"))
        .filter_map(|msg| msg.get("content").and_then(|content| content.as_str()))
        .collect::<Vec<_>>();

    assert_eq!(
        user_messages
            .iter()
            .filter(|message| **message == "You didn't answer my question")
            .count(),
        1,
        "the raw follow-up must appear exactly once: {:?}",
        second_call.messages
    );
    assert!(
        !serde_json::to_string(&second_call.messages)
            .unwrap()
            .contains("Original request:"),
        "the provider payload must not contain a synthetic combined prompt"
    );
    let parent_answer_idx = second_call
        .messages
        .iter()
        .position(|message| {
            message.get("content").and_then(|content| content.as_str())
                == Some("I searched for AI news and found several results.")
        })
        .expect("preceding assistant answer must remain in the transcript");
    assert_eq!(
        second_call.messages[parent_answer_idx + 1]
            .get("content")
            .and_then(|content| content.as_str()),
        Some("You didn't answer my question"),
        "the preceding assistant answer must be structurally adjacent to the raw follow-up"
    );
}

#[tokio::test]
async fn test_exact_source_question_keeps_adjacent_answer_without_phrase_rule() {
    let location_answer = "You live in Fairfax, VA.";
    let job_prep_answer = format!(
        "We've mainly been working on your 2026 AI job preparation and interview briefing. {}\n\nSource artifact: /tmp/synthetic-ai-job-prep/briefing.md",
        "Detailed preparation context covering agent systems, production reliability, evaluated projects, RAG, MLOps, system design, technical interviews, behavioral interviews, market updates, and application tracking. ".repeat(8)
    );
    let source_question = "Where did you get that info from?";
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(location_answer),
        MockProvider::text_response(&job_prep_answer),
        MockProvider::text_response(
            "I based that recap on the saved job-preparation notes returned in the prior turn.",
        ),
    ])
    .with_task_assessments(vec![
        MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "new_request",
            "none",
        ),
        MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "new_request",
            "none",
        ),
        MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "continuation",
            "conversation_history",
        ),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();
    let session_id = "exact_source_question_adjacency";

    for message in [
        "Where do I live?",
        "Remind me what we've mainly been working on together.",
        source_question,
    ] {
        harness
            .agent
            .handle_message(
                session_id,
                message,
                None,
                UserRole::Owner,
                ChannelContext::private("test"),
                None,
            )
            .await
            .unwrap();
    }

    // This exact wording intentionally has no source-question phrase rule. The
    // unresolved typed antecedent may classify it as a Followup without parsing
    // the wording; transcript continuity must still bind it to the immediately
    // preceding answer by canonical topology.
    let dialogue_state = harness
        .state
        .get_dialogue_state(session_id)
        .await
        .unwrap()
        .expect("dialogue state");
    assert_eq!(
        dialogue_state.last_user_turn.as_ref().map(|turn| turn.kind),
        Some(crate::traits::UserTurnKind::Followup)
    );

    let calls = harness.provider.call_log.lock().await;
    let source_call = calls.last().expect("source-question model call");
    let serialized = serde_json::to_string(&source_call.messages).unwrap();
    assert_eq!(serialized.matches(source_question).count(), 1, "{serialized}");
    assert!(!serialized.contains("Original request:"), "{serialized}");

    let job_answer_idx = source_call
        .messages
        .iter()
        .position(|message| {
            message.get("content").and_then(|content| content.as_str())
                == Some(job_prep_answer.as_str())
        })
        .expect("immediately preceding job-preparation answer");
    assert_eq!(
        source_call.messages[job_answer_idx + 1]
            .get("content")
            .and_then(|content| content.as_str()),
        Some(source_question),
        "source question must be directly adjacent to the immediately preceding answer"
    );
    assert!(
        serialized.contains("/tmp/synthetic-ai-job-prep/briefing.md"),
        "source-bearing tail of the preceding answer must survive"
    );
    let location_idx = source_call
        .messages
        .iter()
        .position(|message| {
            message.get("content").and_then(|content| content.as_str()) == Some(location_answer)
        })
        .expect("older location answer remains available as history");
    assert!(
        location_idx < job_answer_idx,
        "older Fairfax answer must not replace the structural parent"
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
    ])
    .with_task_assessments(vec![
        MockProvider::semantic_task_assessment(
            "answer",
            false,
            false,
            &[],
            "new_request",
            "goal_state",
        ),
        MockProvider::semantic_task_assessment(
            "check",
            false,
            true,
            &[],
            "continuation",
            "goal_state",
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
        response.contains("without verified success")
            && !response.contains("What I need from you"),
        "an unavailable in-scope observation must fail honestly without asking the user to manufacture a receipt: {response}"
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
    // Pillar B (Task 7): under turn-anchored whole-turn history, the prior turn
    // IS retained as ARCHIVED context (the prior user message survives verbatim),
    // so we no longer assert its absence. What MUST still be dropped is the
    // learned-helplessness failure boilerplate: `render_archived` excludes
    // `is_failure_boilerplate` assistant text and substitutes a terminal-state
    // placeholder, so the poisoning "I wasn't able to..." reply never re-enters
    // the prompt to trigger giving-up behavior.
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
