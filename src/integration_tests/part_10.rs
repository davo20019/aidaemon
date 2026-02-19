// ==================== Orchestrator Tool Presence Regression Tests ====================

#[tokio::test]
async fn test_orchestrator_first_call_has_tools() {
    // With default+fallback routing, the consultant text-only pre-pass is disabled.
    // The first LLM call ALWAYS includes tools, even at depth=0 (orchestrator).
    // After the intent gate classifies the task, tools remain available for execution.
    let provider = MockProvider::with_responses(vec![
        // Iteration 1 (tools available): intent gate classification + text response
        MockProvider::text_response("I'll check that for you."),
        // Execution loop: tool call
        MockProvider::tool_call_response("system_info", "{}"),
        // Execution loop: final response
        MockProvider::text_response("System is running macOS."),
    ]);

    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let _response = harness
        .agent
        .handle_message(
            "test_session",
            "Show me the system information",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let calls = harness.provider.call_log.lock().await;
    assert!(!calls.is_empty(), "Expected at least 1 LLM call");

    // First call: MUST have tools (no tool-stripping for iteration 1 anymore)
    assert!(
        !calls[0].tools.is_empty(),
        "First LLM call must have tools present, got 0 tools"
    );
}

#[tokio::test]
async fn test_orchestrator_executes_tool_calls_in_first_iteration() {
    // With default+fallback routing, tools are always present. If the LLM
    // returns a tool call in iteration 1, it is executed (not dropped).
    // Previously, the consultant pass had no tools and tool calls were
    // considered "hallucinated" and dropped. Now they are legitimate.
    use crate::traits::ToolCall;

    let provider = MockProvider::with_responses(vec![
        // Iteration 1 (tools present): LLM returns text + tool call
        ProviderResponse {
            content: Some("I'll look into the system details.".to_string()),
            tool_calls: vec![ToolCall {
                id: "call_system_info".to_string(),
                name: "system_info".to_string(),
                arguments: "{}".to_string(),
                extra_content: None,
            }],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        // After tool execution: final text response
        MockProvider::text_response("System is running macOS."),
    ]);

    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Check the system information now",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The tool call from iteration 1 is executed, and the final response
    // comes from the subsequent LLM call after tool execution.
    assert_eq!(response, "System is running macOS.");

    let calls = harness.provider.call_log.lock().await;
    // First call has tools (no tool-stripping anymore)
    assert!(
        !calls[0].tools.is_empty(),
        "First LLM call must have tools present"
    );
    // At least 2 calls: initial (with tool call) + post-execution
    assert!(
        calls.len() >= 2,
        "Expected at least 2 LLM calls (tool call + final), got {}",
        calls.len()
    );
}

#[tokio::test]
async fn test_orchestrator_knowledge_flow() {
    // Knowledge flow: iteration 1 emits INTENT_GATE with can_answer_now=true,
    // then the execution loop answers without tool use. With default+fallback
    // routing, tools ARE present in the first call (no tool-stripping), but
    // the model chooses not to use them for simple knowledge answers.
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "I can answer this from memory.\n[INTENT_GATE]\n{\"complexity\": \"knowledge\", \"can_answer_now\": true, \"needs_tools\": false}",
        ),
        MockProvider::text_response("The capital of France is Paris."),
    ]);

    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "What is the capital of France?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "The capital of France is Paris.");

    let call_count = harness.provider.call_count().await;
    assert_eq!(call_count, 2, "Expected intent gate classifier + executor answer");

    // Tools are present in the first call (no tool-stripping in the new architecture)
    let calls = harness.provider.call_log.lock().await;
    assert!(
        !calls[0].tools.is_empty(),
        "First LLM call should have tools present (default+fallback routing)"
    );
}

#[tokio::test]
async fn test_executor_mode_retains_tools() {
    // Contrast: an agent in executor mode (depth > 0) MUST have tools available.
    // This ensures set_test_executor_mode doesn't break tool access.
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("System info retrieved."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    // setup_test_agent calls set_test_executor_mode() → depth=1, Executor role

    let _response = harness
        .agent
        .handle_message(
            "test_session",
            "Show me the system information",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let calls = harness.provider.call_log.lock().await;
    assert!(
        !calls[0].tools.is_empty(),
        "Executor mode must have tools available in LLM calls"
    );
}

/// Scenario: Turn 1 makes tool calls, Turn 2 asks a different question.
/// The tool intermediates from Turn 1 should be collapsed so they don't
/// pollute Turn 2's context and confuse the LLM (context bleeding bug).
#[tokio::test]
async fn test_old_tool_intermediates_collapsed_in_follow_up() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1: tool call + final response
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Your system has 16GB RAM and an M1 chip."),
        // Turn 2: direct text response (different topic)
        MockProvider::text_response("Bella is your cat."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    // Turn 1: triggers a tool call (system_info)
    let r1 = harness
        .agent
        .handle_message(
            "collapse_test",
            "What system info do I have?",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r1, "Your system has 16GB RAM and an M1 chip.");

    // Turn 2: different topic — should NOT include Turn 1's tool intermediates
    let r2 = harness
        .agent
        .handle_message(
            "collapse_test",
            "Who is bella?",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r2, "Bella is your cat.");

    // Verify Turn 2's messages don't contain tool intermediates from Turn 1
    let call_log = harness.provider.call_log.lock().await;
    let turn2_call = call_log.last().unwrap();
    let turn2_msgs = &turn2_call.messages;

    // There should be NO tool-role messages from Turn 1
    let tool_msgs: Vec<&serde_json::Value> = turn2_msgs
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .collect();
    assert!(
        tool_msgs.is_empty(),
        "Turn 2 should not contain tool results from Turn 1, found {} tool messages",
        tool_msgs.len()
    );

    // There should be NO assistant messages with tool_calls from Turn 1
    let assistant_with_tc: Vec<&serde_json::Value> = turn2_msgs
        .iter()
        .filter(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && m.get("tool_calls").is_some()
        })
        .collect();
    assert!(
        assistant_with_tc.is_empty(),
        "Turn 2 should not contain assistant tool_calls from Turn 1, found {}",
        assistant_with_tc.len()
    );

    // But Turn 2 SHOULD still have the user messages and final assistant response from Turn 1
    let user_msgs: Vec<&serde_json::Value> = turn2_msgs
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .collect();
    assert!(
        user_msgs.len() >= 2,
        "Turn 2 should include user messages from both turns, found {}",
        user_msgs.len()
    );
}

/// Regression: when the final LLM response is empty after tool calls, a
/// synthesized "Done" message is returned. Before the fix it was NOT saved
/// to the DB, causing the next interaction's history to merge the two user
/// messages (missing assistant in between) and bleeding context.
#[tokio::test]
async fn test_synthesized_done_persisted() {
    // At depth=0 (orchestrator), iteration 1 is the consultant pass (no tools).
    // The mock tool_call_response triggers hallucinated-tool detection which
    // forces needs_tools=true → Simple intent → tools loaded → loop continues.
    let provider = MockProvider::with_responses(vec![
        // Turn 1, iteration 1 (consultant pass): tool_call forces needs_tools=true
        MockProvider::tool_call_response("system_info", "{}"),
        // Turn 1, iteration 2 (tools available): tool call is executed
        MockProvider::tool_call_response("system_info", "{}"),
        // Turn 1, iteration 3: empty response → "Done" synthesis at depth=0
        MockProvider::text_response(""),
        // Turn 2, iteration 1 (consultant pass): classifier output
        MockProvider::text_response(
            "I can answer this from memory.\n[INTENT_GATE] {\"complexity\":\"knowledge\",\"can_answer_now\":true,\"needs_tools\":false}",
        ),
        // Turn 2, iteration 2 (execution): final user-visible answer
        MockProvider::text_response("Weather is sunny."),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    // Reset to depth=0 so orchestrator mode + "Done" synthesis fires
    harness.agent.set_test_orchestrator_mode();

    // Turn 1: should trigger completion recovery (tool output or Done synthesis)
    let r1 = harness
        .agent
        .handle_message(
            "done_persist_test",
            "Check my system info",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    // After tool execution with empty final response, the agent recovers from
    // the latest tool output ("Here is the latest tool output:") or synthesizes "Done".
    assert!(
        r1.starts_with("Done") || r1.starts_with("Here is the latest tool output"),
        "Expected Done synthesis or tool output recovery, got: {}",
        r1
    );

    // Turn 2: different topic
    let r2 = harness
        .agent
        .handle_message(
            "done_persist_test",
            "Tell me the weather",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(!r2.is_empty(), "Turn 2 should produce a non-empty response");

    // Verify: Turn 2's first LLM call should have >= 2 separate user messages (not merged)
    let call_log = harness.provider.call_log.lock().await;
    // Turn 2 starts at the 4th call (Turn 1 consumed 3 calls).
    let turn2_call = &call_log[3];
    let user_msgs: Vec<&serde_json::Value> = turn2_call
        .messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .collect();
    assert!(
        user_msgs.len() >= 2,
        "Turn 2 should have at least 2 separate user messages (not merged), found {}",
        user_msgs.len()
    );

    // Verify: there should be a completion assistant message between the user messages
    // (either "Done" synthesis or "Here is the latest tool output" recovery)
    let completion_assistant = turn2_call.messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("assistant")
            && m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.starts_with("Done") || s.starts_with("Here is the latest tool output"))
    });
    assert!(
        completion_assistant,
        "Turn 2's history should contain the persisted completion assistant message from Turn 1"
    );
}

/// Regression: old interaction assistant responses should be truncated so
/// stale context from long prior turns doesn't pollute subsequent replies.
/// Exception: the immediately-prior assistant message (e.g., budget/timeout response)
/// is preserved untruncated to provide handoff context.
#[tokio::test]
async fn test_old_interaction_assistant_content_truncated() {
    let long_response_1 = "B".repeat(500);
    let long_response_2 = "A".repeat(500);
    let provider = MockProvider::with_responses(vec![
        // Turn 1: long response
        MockProvider::text_response(&long_response_1),
        // Turn 2: another long response
        MockProvider::text_response(&long_response_2),
        // Turn 3: direct text response
        MockProvider::text_response("Short answer."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    // Turn 1: produces a long assistant response
    let r1 = harness
        .agent
        .handle_message(
            "truncate_test",
            "First question?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r1, long_response_1);

    // Turn 2: another long response
    let r2 = harness
        .agent
        .handle_message(
            "truncate_test",
            "Second question?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r2, long_response_2);

    // Turn 3: different topic
    let r3 = harness
        .agent
        .handle_message(
            "truncate_test",
            "Third question?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r3, "Short answer.");

    // Verify: Turn 1's 500-char response (2+ turns back) is truncated in Turn 3,
    // but Turn 2's response (immediately prior) is preserved untruncated.
    let call_log = harness.provider.call_log.lock().await;
    let turn3_call = call_log.last().unwrap();
    let assistant_msgs: Vec<&serde_json::Value> = turn3_call
        .messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .collect();

    // Turn 1's response (BBB...) should be truncated (it's NOT the immediately-prior)
    let has_truncated = assistant_msgs.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.starts_with('B') && s.ends_with('…') && s.len() < 500)
    });
    assert!(
        has_truncated,
        "Turn 1's long assistant response should be truncated in Turn 3's context"
    );

    // Turn 2's response (AAA...) should be preserved untruncated (immediately-prior)
    let has_preserved = assistant_msgs.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.starts_with('A') && s.len() == 500)
    });
    assert!(
        has_preserved,
        "Turn 2's assistant response (immediately prior) should be preserved untruncated"
    );

    // Truncated content should be <= MAX_OLD_ASSISTANT_CONTENT_CHARS + ellipsis
    for m in &assistant_msgs {
        if let Some(content) = m.get("content").and_then(|c| c.as_str()) {
            if content.starts_with('B') && content.ends_with('…') {
                // 200 chars + "…" (3 bytes) = ~203 bytes max
                assert!(
                    content.len() <= 210,
                    "Truncated content should be ~203 chars max, got {} chars: {}...",
                    content.len(),
                    &content[..50.min(content.len())]
                );
            }
        }
    }
}

/// Short assistant responses from old turns should be passed through unmodified
/// (no marker text appended, since LLMs tend to echo markers back).
#[tokio::test]
async fn test_old_short_assistant_response_preserved_unmodified() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1: short response about files
        MockProvider::text_response("Here are 20 .rs files in the tools directory."),
        // Turn 2: different topic
        MockProvider::text_response("Rust 1.82.0"),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    // Turn 1
    let _ = harness
        .agent
        .handle_message(
            "prior_turn_no_marker",
            "List files in src/tools",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Turn 2: completely different topic
    let _ = harness
        .agent
        .handle_message(
            "prior_turn_no_marker",
            "What version of Rust is installed?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Verify: Turn 1's short assistant response is present without marker text
    let call_log = harness.provider.call_log.lock().await;
    let turn2_call = call_log.last().unwrap();
    let old_assistant_msgs: Vec<&serde_json::Value> = turn2_call
        .messages
        .iter()
        .filter(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("files"))
        })
        .collect();

    assert!(
        !old_assistant_msgs.is_empty(),
        "Turn 1's assistant response should be present in Turn 2's context"
    );
    // Content should be exactly what the LLM returned — no marker appended
    let content = old_assistant_msgs[0]
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap();
    assert!(
        !content.contains("[prior turn]"),
        "Old assistant responses should NOT have [prior turn] marker (causes LLM echoing). Got: {}",
        content
    );
    assert_eq!(
        content, "Here are 20 .rs files in the tools directory.",
        "Short old assistant content should be preserved unmodified"
    );
}
