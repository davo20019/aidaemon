// ==================== Task Boundary on Multi-Turn Tasks ====================

/// Regression: When a second turn runs after a previous interaction, old user messages
/// from prior interactions must not confuse the model. The task boundary marker
/// should be injected to separate old context from the current task.
///
/// Scenario: Turn 1 asks "Why?", Turn 2 asks to find a file.
/// Turn 2's LLM calls should see a [Current Task] marker separating the old "Why?"
/// from the current request, preventing the model from responding to old context.
#[tokio::test]
async fn test_task_boundary_injected_between_turns() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1: direct text answer
        MockProvider::text_response(
            "Because the previous step required it — happy to elaborate if useful.",
        ),
        // Turn 2: tool call, then answer
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Found the Spanish resume."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    // Turn 1: simple question
    let _r1 = harness
        .agent
        .handle_message(
            "boundary_test",
            "Why?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Turn 2: different task
    let _r2 = harness
        .agent
        .handle_message(
            "boundary_test",
            "Send me the resume in Spanish now.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Verify: any LLM call for Turn 2 that includes old "Why?" context
    // must also have a [Current Task] marker separating old from new.
    let call_log = harness.provider.call_log.lock().await;
    assert!(
        call_log.len() >= 2,
        "Expected at least 2 LLM calls (one per turn), got {}",
        call_log.len()
    );

    let turn2_calls: Vec<_> = call_log
        .iter()
        .filter(|call| {
            call.messages.iter().any(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.contains("Send me the resume in Spanish now."))
            })
        })
        .collect();
    assert!(
        !turn2_calls.is_empty(),
        "Expected at least one Turn 2 LLM call containing the current user request"
    );

    let turn2_calls_ok = turn2_calls.iter().all(|call| {
        let has_old_user = call.messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("user")
                && m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("Why?"))
        });
        let has_boundary = call.messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("system")
                && m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("[Current Task]"))
        });
        // If old context is present, boundary must be too. If old context was dropped, that's fine.
        !has_old_user || has_boundary
    });
    assert!(
        turn2_calls_ok,
        "All Turn 2 LLM calls must have [Current Task] when old user context is present"
    );
}

/// File-upload requests are handled by the sliding window and compaction system.
/// With the adaptive sliding window, small prior pairs may be retained if they
/// fit within the token budget. The compaction trigger fires on file uploads
/// without referential language, producing a summary for subsequent context.
/// This test verifies that the uploaded-file message is always present in the
/// Turn 2 context and that a task boundary marker separates it from any
/// retained prior conversation.
#[tokio::test]
async fn test_uploaded_artifact_request_has_task_boundary() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1 response
        MockProvider::text_response(
            "Would you like me to get more detailed information for any specific trial(s)?",
        ),
        // Compaction LLM call (file upload triggers compaction)
        MockProvider::text_response("Summary of prior conversation."),
        // Turn 2 response
        MockProvider::text_response("I reviewed the uploaded document and identified the issue."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message(
            "artifact_bleed_test",
            "These are the NCT trial numbers: NCT06737964 and NCT06737965.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let artifact_request = "[File received: 68235.png (413 KB, image/png)\nSaved to: /Users/davidloor/projects/aidaemon/.aidaemon/files/inbox/694c3943_68235.png]\nCheck the doc and fix the issue.";
    let _ = harness
        .agent
        .handle_message(
            "artifact_bleed_test",
            artifact_request,
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let turn2_call = call_log.last().expect("turn 2 call");

    // The file-upload message must be present in Turn 2 context.
    assert!(
        turn2_call.messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("user")
                && m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("[File received: 68235.png"))
        }),
        "Turn 2 should include the uploaded-file context"
    );

    // A task boundary marker should separate prior history from the current request.
    assert!(
        turn2_call.messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains("[Current Task]"))
        }),
        "Turn 2 should have a task boundary marker: {:?}",
        turn2_call.messages
    );
}

/// Regression: after tool progress exists in the current task, a generic idle
/// prompt must not be accepted as the final answer. The next LLM call should
/// also carry an execution checkpoint for continuity.
#[tokio::test]
async fn test_idle_reengagement_reply_after_tool_progress_is_recovered() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("I'm here. What would you like me to help you with?"),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "idle_reengagement_recovery",
            "Check the system details and tell me what machine this is.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        !response.contains("What would you like me to help you with"),
        "generic idle re-engagement reply should not be returned after tool progress: {}",
        response
    );
    assert!(
        response.contains("latest tool output") || response.contains("Date:"),
        "final reply should recover from concrete tool evidence: {}",
        response
    );

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        call_log.len() >= 2,
        "expected at least two LLM calls, got {}",
        call_log.len()
    );
    let second_call_has_checkpoint = call_log[1].messages.iter().any(|message| {
        message.get("role").and_then(|r| r.as_str()) == Some("system")
            && message
                .get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|content| {
                    content.contains("EXECUTION CHECKPOINT")
                        && content.contains("Check the system details")
                })
    });
    assert!(
        second_call_has_checkpoint,
        "second LLM call should include the execution checkpoint"
    );
}

// ==================== Orchestrator Tool Presence Regression Tests ====================

#[tokio::test]
async fn test_orchestrator_first_call_has_tools() {
    // With default+fallback routing, the text-only pre-pass is disabled.
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
    // Previously, the first routing pass had no tools and tool calls were
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
                cached_input_tokens: None,
                cache_creation_input_tokens: None,
                model: "mock".to_string(),
                ..Default::default()
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
    // Knowledge flow: the first reply is a short deferral, bounced by the
    // deferred-action gate; the retry answers without tool use. With
    // default+fallback routing, tools ARE present in the first call (no
    // tool-stripping), but the model chooses not to use them for simple
    // knowledge answers.
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("Let me check my memory first."),
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
    assert_eq!(call_count, 2, "Expected bounced deferral + executor answer");

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
/// The immediately preceding turn keeps bounded receipt-bearing tool evidence
/// so a follow-up can distinguish observed facts from the assistant's prose.
/// Older turns are still collapsed by the archived renderer.
#[tokio::test]
async fn test_immediate_parent_tool_evidence_is_bounded_in_follow_up() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1: tool call + final response
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Your system has 16GB RAM and an M1 chip."),
        // Turn 2: direct text response (different topic)
        MockProvider::text_response("Mia is your cat."),
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

    // Turn 2: different topic. The immediately preceding tool result remains
    // available, but only through the adjacent turn's bounded evidence form.
    let r2 = harness
        .agent
        .handle_message(
            "collapse_test",
            "Who is mia?",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r2, "Mia is your cat.");

    // Verify Turn 2's messages: Prior 1 tool results retain their strong
    // receipt and are bounded. Prior 2+ tool results use compact summaries.
    let call_log = harness.provider.call_log.lock().await;
    let turn2_call = call_log.last().unwrap();
    let turn2_msgs = &turn2_call.messages;

    // Tool results from Turn 1 (the "Prior 1" interaction) should be present
    // with their receipt identity rather than being converted into prose.
    let tool_msgs: Vec<&serde_json::Value> = turn2_msgs
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .collect();
    assert!(!tool_msgs.is_empty(), "adjacent evidence must be retained");
    for tool_msg in &tool_msgs {
        let content = tool_msg
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        assert!(
            content.contains("[Tool receipt v4:"),
            "Prior 1 tool result should retain its receipt, got: {}",
            content
        );
        assert!(
            content.chars().count() <= 4_000,
            "Prior 1 tool evidence should be bounded, got: {}",
            content
        );
    }

    // Turn 2 SHOULD still have the user messages from both turns
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
    // At depth=0 (orchestrator), iteration 1 is the first routing call.
    // The mock tool_call_response triggers hallucinated-tool detection which
    // forces needs_tools=true → Simple intent → tools loaded → loop continues.
    let provider = MockProvider::with_responses(vec![
        // Turn 1, iteration 1: tool call
        MockProvider::tool_call_response("system_info", "{}"),
        // Turn 1, iteration 2 (tools available): tool call is executed
        MockProvider::tool_call_response("system_info", "{}"),
        // Turn 1, iteration 3: empty response → "Done" synthesis at depth=0
        MockProvider::text_response(""),
        // Turn 2, iteration 1: short deferral, bounced by the deferred-action gate
        MockProvider::text_response("Let me check my memory first."),
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
        r1.starts_with("Done") || r1.starts_with("Here is the latest tool output") || r1.starts_with("Here's the command output") || r1.starts_with("Here are the results"),
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
                .is_some_and(|s| s.starts_with("Done") || s.starts_with("Here is the latest tool output") || s.starts_with("Here's the command output") || s.starts_with("Here are the results") || s.starts_with("Here's"))
    });
    assert!(
        completion_assistant,
        "Turn 2's history should contain the persisted completion assistant message from Turn 1"
    );
}

/// Regression: stale archived answers stay compact while the exact immediately
/// preceding answer remains available for natural-language follow-ups.
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
            "Also third question?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r3, "Short answer.");

    // Verify: the older turn is compact, while the structurally referenced
    // immediately preceding turn is present in full.
    let call_log = harness.provider.call_log.lock().await;
    let turn3_call = call_log.last().unwrap();
    let assistant_msgs: Vec<&serde_json::Value> = turn3_call
        .messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .collect();

    // Turn 1's response (BBB...) should be truncated (archived).
    let has_truncated_b = assistant_msgs.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.starts_with('B') && s.ends_with('…') && s.len() < 500)
    });
    assert!(
        has_truncated_b,
        "Turn 1's long assistant response should be truncated in Turn 3's context"
    );

    let has_full_a = assistant_msgs.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s == long_response_2)
    });
    assert!(
        has_full_a,
        "Turn 2's assistant response should be preserved in full as the referenced parent"
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
        // Turn 1: short direct answer
        MockProvider::text_response("It is 4."),
        // Turn 2: different topic
        MockProvider::text_response("Rust 1.82.0"),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    // Turn 1
    let _ = harness
        .agent
        .handle_message(
            "prior_turn_no_marker",
            "What is 2 + 2?",
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
                    .is_some_and(|s| s == "It is 4.")
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
        content, "It is 4.",
        "Short old assistant content should be preserved unmodified"
    );
}

// ==================== Compaction Integration Tests ====================

/// When canonical conversation history crosses the token high-water mark,
/// compaction should fire and produce structured state in the DB. Subsequent
/// turns should see that state and its coverage cursor in their LLM context.
#[tokio::test]
async fn test_compaction_fires_on_window_overflow() {
    // All responses are interchangeable because the background summary and
    // foreground completion calls can race. Supplying a generous homogeneous
    // queue makes the test assert context behavior, not scheduler ordering.
    let responses = (0..32)
        .map(|_| MockProvider::text_response("Synthetic response"))
        .collect();

    let provider = MockProvider::with_responses(responses);
    let mut harness = setup_test_agent(provider).await.unwrap();
    harness
        .agent
        .set_context_compaction_tokens_for_test(512, 256);

    // Cross the focused test watermark with ordinary-sized synthetic messages.
    // Production retains its 12k-token default; compaction is intentionally no
    // longer driven by a fixed number of messages.
    let pressure = "synthetic historical context detail ".repeat(30);
    for i in 1..=7 {
        let _ = harness
            .agent
            .handle_message(
                "compaction_test",
                &format!("Question {i} about synthetic topic {i}. {pressure}"),
                None,
                UserRole::Owner,
                ChannelContext::private("test"),
                None,
            )
            .await
            .unwrap();
    }

    // Allow the async compaction task to complete.
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // Verify: summary should exist in DB after window overflow.
    let summary = harness
        .state
        .get_conversation_summary("compaction_test")
        .await
        .unwrap();
    assert!(
        summary.is_some(),
        "Compaction summary should exist in DB after window overflow"
    );
    let summary = summary.unwrap();
    assert!(
        !summary.summary.is_empty(),
        "Compaction summary should not be empty"
    );

    // Turn 8: the summary should be injected into LLM context.
    let _ = harness
        .agent
        .handle_message(
            "compaction_test",
            "Question 8 about topic 8",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Allow Turn 8's async compaction to settle. The previous 200ms was
    // tight enough to flake under coverage instrumentation; matching the
    // 1000ms pattern used above the prior assertion makes the test
    // resilient to slower runners.
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // Verify: Turn 8's LLM call should include compacted state. Turn 8
    // can generate multiple LLM calls (an async incremental compaction plus
    // the main response) and their order on the call_log is timing-
    // dependent. Scan the calls produced during Turn 8 rather than relying
    // on `last()` so the assertion checks the message-building path
    // regardless of which call landed last.
    let call_log = harness.provider.call_log.lock().await;
    assert!(
        call_log.len() >= 8,
        "expected at least one call per turn; got {}",
        call_log.len()
    );
    // Turn 8's calls are at the tail of the log. The exact count is
    // implementation-dependent (compaction may add 0 or 1 calls), so we
    // scan the last 4 calls — more than enough to cover any plausible
    // mix and still bounded so we don't match earlier turns.
    let tail_start = call_log.len().saturating_sub(4);
    let has_summary = call_log[tail_start..].iter().any(|call| {
        call.messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("system")
                && m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| {
                        s.contains("[Context Coverage]")
                            && s.contains("[Compacted Conversation State]")
                    })
        })
    });
    assert!(
        has_summary,
        "Turn 8's LLM context should include compacted state and its coverage cursor"
    );

    // Verify: Turn 8 should have a [Current Task] boundary marker. Like the
    // summary check above, scan the tail of Turn 8's calls — compaction may
    // land last and its system prompt won't carry the boundary marker.
    let has_boundary = call_log[tail_start..].iter().any(|call| {
        call.messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains("[Current Task]"))
        })
    });
    assert!(
        has_boundary,
        "Turn 8's LLM context should include [Current Task] boundary marker"
    );
}

/// Regression: messages persisted during a `handle_message` call must be
/// stamped with a `turn_id` so boundary detection groups them deterministically.
///
/// Before turn_id, the boundary was inferred by matching `user_text` against
/// message content, which had a known race condition: when the same text was
/// sent twice in the same session, `rposition` could pick the old instance and
/// keep an unrelated tool chain as "current interaction." With turn_id, the
/// boundary is a lookup, immune to duplicate text.
#[tokio::test]
async fn test_turn_id_groups_messages_within_a_turn() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1: tool call + final response.
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Done turn 1"),
        // Turn 2: tool call + final response.
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Done turn 2"),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    // Turn 1
    let _ = harness
        .agent
        .handle_message(
            "turn_id_test",
            "First request",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Turn 2
    let _ = harness
        .agent
        .handle_message(
            "turn_id_test",
            "Second request",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Pull persisted messages from working memory. Every message that flowed
    // through `append_message_canonical` during a turn should carry a turn_id.
    let history = harness
        .state
        .get_history("turn_id_test", 100)
        .await
        .unwrap();

    let stamped: Vec<_> = history.iter().filter(|m| m.turn_id.is_some()).collect();
    assert!(
        !stamped.is_empty(),
        "expected some messages to carry a turn_id, got {} messages with none",
        history.len()
    );

    // Each user message carries a turn_id that equals its own message id
    // (set in bootstrap so the turn_id is the same as the user message id).
    // Verify the invariant on user messages.
    let user_messages: Vec<_> = history.iter().filter(|m| m.role == "user").collect();
    assert_eq!(
        user_messages.len(),
        2,
        "expected 2 user messages, got {}",
        user_messages.len()
    );
    for um in &user_messages {
        assert_eq!(
            um.turn_id.as_deref(),
            Some(um.id.as_str()),
            "user message turn_id should equal its own id; got msg id={} turn_id={:?}",
            um.id,
            um.turn_id
        );
    }

    // The two user messages have distinct turn_ids.
    assert_ne!(
        user_messages[0].turn_id, user_messages[1].turn_id,
        "two distinct user turns must have distinct turn_ids"
    );

    // Every assistant or tool message after the first user message and before
    // the second user message should carry Turn 1's turn_id. We don't assert
    // exact grouping (tool result placement can vary by code path), but we do
    // assert at least one non-user message carries each turn_id.
    let turn1_id = user_messages[0].turn_id.clone().unwrap();
    let turn2_id = user_messages[1].turn_id.clone().unwrap();
    let turn1_nonuser_count = history
        .iter()
        .filter(|m| m.role != "user" && m.turn_id.as_deref() == Some(&turn1_id))
        .count();
    let turn2_nonuser_count = history
        .iter()
        .filter(|m| m.role != "user" && m.turn_id.as_deref() == Some(&turn2_id))
        .count();
    assert!(
        turn1_nonuser_count > 0,
        "Turn 1 should have at least one non-user message stamped with its turn_id"
    );
    assert!(
        turn2_nonuser_count > 0,
        "Turn 2 should have at least one non-user message stamped with its turn_id"
    );
}

fn llm_messages_contain_input_audio(messages: &[serde_json::Value]) -> bool {
    messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user")
            && m.get("content").and_then(|c| c.as_array()).is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("input_audio"))
            })
    })
}

fn llm_messages_contain_image_url(messages: &[serde_json::Value]) -> bool {
    messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user")
            && m.get("content").and_then(|c| c.as_array()).is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("image_url"))
            })
    })
}

/// Vision-enabled image uploads should reach the LLM as multimodal content blocks.
#[tokio::test]
async fn test_vision_image_attachment_reaches_provider_as_multimodal() {
    use std::io::Write;

    use crate::channels::attachments::{build_inbound_text, message_attachment};

    let mut png = tempfile::NamedTempFile::new().unwrap();
    png.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00])
        .unwrap();

    let attachment = message_attachment(
        png.path().to_path_buf(),
        "test.png".to_string(),
        "image/png".to_string(),
        9,
    );
    let attachments = vec![attachment.clone()];
    let inbound_text = build_inbound_text("what is this?", &attachments);

    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("Prior context acknowledged."),
        MockProvider::text_response("Summary of prior conversation."),
        MockProvider::text_response("That looks like a PNG image."),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message(
            "vision_multimodal_test",
            "Hello before the image.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let _ = harness
        .agent
        .handle_message_with_attachments(
            "vision_multimodal_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let agent_calls: Vec<_> = call_log
        .iter()
        .filter(|call| {
            !call.messages.iter().any(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("conversation summarizer"))
            })
        })
        .collect();
    assert!(
        agent_calls
            .iter()
            .any(|call| llm_messages_contain_image_url(&call.messages)),
        "expected an agent LLM call with image_url content block, got: {:?}",
        agent_calls
            .iter()
            .map(|c| &c.messages)
            .collect::<Vec<_>>()
    );
}

/// When vision is disabled, attachments stay as text stubs only.
#[tokio::test]
async fn test_vision_disabled_sends_text_stub_only() {
    use std::io::Write;

    use crate::channels::attachments::{build_inbound_text, message_attachment};
    use crate::config::{FilesConfig, VisionConfig};

    let mut png = tempfile::NamedTempFile::new().unwrap();
    png.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00])
        .unwrap();

    let attachment = message_attachment(
        png.path().to_path_buf(),
        "test.png".to_string(),
        "image/png".to_string(),
        9,
    );
    let attachments = vec![attachment.clone()];
    let inbound_text = build_inbound_text("describe this", &attachments);

    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I received your file but cannot view images while vision is disabled.",
    )]);
    let mut harness = setup_test_agent(provider).await.unwrap();
    let files = FilesConfig {
        vision_enabled: false,
        ..Default::default()
    };
    harness
        .agent
        .set_test_vision_config(VisionConfig::from_files(&files));

    let _ = harness
        .agent
        .handle_message_with_attachments(
            "vision_disabled_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        !call_log
            .iter()
            .any(|call| llm_messages_contain_image_url(&call.messages)),
        "vision disabled must not send image_url blocks"
    );
    assert!(
        call_log.iter().any(|call| {
            call.messages.iter().any(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.contains("[File received: test.png"))
            })
        }),
        "user message should still include the text stub"
    );
}

/// Audio attachments reach the LLM as input_audio when model matches audio patterns.
#[tokio::test]
async fn test_audio_attachment_reaches_provider_as_input_audio() {
    use std::io::Write;

    use crate::channels::attachments::{build_inbound_text, message_attachment};
    let mut ogg = tempfile::NamedTempFile::new().unwrap();
    ogg.write_all(&[1, 2, 3, 4, 5]).unwrap();

    let attachment = message_attachment(
        ogg.path().to_path_buf(),
        "voice.ogg".to_string(),
        "audio/ogg".to_string(),
        5,
    );
    let attachments = vec![attachment.clone()];
    let inbound_text = build_inbound_text("what did they say?", &attachments);

    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "They asked about the weather.",
    )]);
    let harness = setup_test_agent(provider).await.unwrap();
    harness
        .agent
        .set_test_model("gemini-2.0-flash")
        .await;

    let _ = harness
        .agent
        .handle_message_with_attachments(
            "audio_multimodal_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        call_log
            .iter()
            .any(|call| llm_messages_contain_input_audio(&call.messages)),
        "expected input_audio block in LLM payload"
    );
}

/// When audio is disabled, attachments stay as text stubs only.
#[tokio::test]
async fn test_audio_disabled_sends_text_stub_only() {
    use std::io::Write;

    use crate::channels::attachments::{build_inbound_text, message_attachment};
    use crate::config::{AudioConfig, FilesConfig};

    let mut ogg = tempfile::NamedTempFile::new().unwrap();
    ogg.write_all(&[1, 2, 3, 4]).unwrap();

    let attachment = message_attachment(
        ogg.path().to_path_buf(),
        "voice.ogg".to_string(),
        "audio/ogg".to_string(),
        4,
    );
    let attachments = vec![attachment.clone()];
    let inbound_text = build_inbound_text("listen", &attachments);

    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I received your audio but cannot process it while audio is disabled.",
    )]);
    let mut harness = setup_test_agent(provider).await.unwrap();
    let files = FilesConfig {
        audio_enabled: false,
        ..Default::default()
    };
    harness
        .agent
        .set_test_audio_config(AudioConfig::from_files(&files));
    harness
        .agent
        .set_test_model("gemini-2.0-flash")
        .await;

    let _ = harness
        .agent
        .handle_message_with_attachments(
            "audio_disabled_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        !call_log
            .iter()
            .any(|call| llm_messages_contain_input_audio(&call.messages)),
        "audio disabled must not send input_audio blocks"
    );
}

/// Ineligible models should not receive input_audio blocks.
#[tokio::test]
async fn test_audio_ineligible_model_sends_text_stub_only() {
    use std::io::Write;

    use crate::channels::attachments::{build_inbound_text, message_attachment};

    let mut ogg = tempfile::NamedTempFile::new().unwrap();
    ogg.write_all(&[1, 2, 3, 4]).unwrap();

    let attachment = message_attachment(
        ogg.path().to_path_buf(),
        "voice.ogg".to_string(),
        "audio/ogg".to_string(),
        4,
    );
    let attachments = vec![attachment.clone()];
    let inbound_text = build_inbound_text("listen", &attachments);

    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I saved the audio but this model cannot hear it.",
    )]);
    let harness = setup_test_agent(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message_with_attachments(
            "audio_ineligible_model_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        !call_log
            .iter()
            .any(|call| llm_messages_contain_input_audio(&call.messages)),
        "mock-model must not send input_audio blocks"
    );
}

/// When native audio is skipped, Whisper STT fallback appends transcription text.
#[tokio::test]
async fn test_stt_fallback_appends_transcription_when_native_audio_skipped() {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    use crate::agent::stt::{content_has_transcription, format_transcription_line};
    use crate::channels::attachments::{build_inbound_text, message_attachment};
    use crate::config::SttConfig;

    let tmp = tempfile::tempdir().unwrap();
    let mock_cli = tmp.path().join("mock-whisper-cli.sh");
    std::fs::write(
        &mock_cli,
        r#"#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -of) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s' 'Who is my dad?' > "${out}.txt"
"#,
    )
    .unwrap();
    let mut perms = std::fs::metadata(&mock_cli).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&mock_cli, perms).unwrap();

    let model_path = tmp.path().join("model.bin");
    std::fs::write(&model_path, b"mock").unwrap();

    let mut wav = tempfile::NamedTempFile::new().unwrap();
    wav.write_all(b"RIFF....WAVEfmt ").unwrap();

    let attachment = message_attachment(
        wav.path().to_path_buf(),
        "voice.wav".to_string(),
        "audio/wav".to_string(),
        16,
    );
    let attachments = vec![attachment.clone()];
    let inbound_text = build_inbound_text("what did they say?", &attachments);

    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "They asked who their dad is.",
    )]);
    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_stt_config(SttConfig {
        enabled: true,
        cli_path: mock_cli,
        model_path,
        ffmpeg_path: std::path::PathBuf::from("ffmpeg"),
        language: "en".to_string(),
        max_audio_bytes: 25 * 1_048_576,
        timeout_secs: 30,
        mime_types: vec!["audio/wav".to_string(), "audio/ogg".to_string()],
    });

    let _ = harness
        .agent
        .handle_message_with_attachments(
            "stt_fallback_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let history = harness
        .state
        .get_history("stt_fallback_test", 5)
        .await
        .unwrap();
    let user_msg = history
        .iter()
        .find(|m| m.role == "user")
        .expect("user message persisted");
    let content = user_msg.content.as_deref().unwrap_or("");
    assert!(
        content_has_transcription(content),
        "user message should include STT transcription, got: {content}"
    );
    assert!(content.contains(&format_transcription_line("voice.wav", "Who is my dad?")));

    let call_log = harness.provider.call_log.lock().await;
    let llm_user_text = call_log
        .iter()
        .flat_map(|call| call.messages.iter())
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>();
    assert!(
        llm_user_text
            .iter()
            .any(|text| content_has_transcription(text)),
        "LLM payload should include transcription text"
    );
    assert!(
        !call_log
            .iter()
            .any(|call| llm_messages_contain_input_audio(&call.messages)),
        "STT fallback must not send input_audio blocks"
    );
}

/// Non-image attachments on ineligible models should not produce vision blocks.
#[tokio::test]
async fn test_non_image_attachment_is_text_stub_only() {
    use crate::channels::attachments::{build_inbound_text, message_attachment};

    let attachment = message_attachment(
        std::path::PathBuf::from("/tmp/voice.ogg"),
        "voice.ogg".to_string(),
        "audio/ogg".to_string(),
        1200,
    );
    let attachments = vec![attachment.clone()];
    let inbound_text = build_inbound_text("transcribe this", &attachments);

    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I saved the audio file but cannot transcribe it in this test.",
    )]);
    let harness = setup_test_agent(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message_with_attachments(
            "vision_audio_stub_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        !call_log
            .iter()
            .any(|call| llm_messages_contain_image_url(&call.messages)),
        "audio attachments must not produce image_url blocks"
    );
}

/// File uploads with structured attachments should still trigger compaction on the stub marker.
#[tokio::test]
async fn test_vision_attachment_still_triggers_compaction() {
    use std::io::Write;

    use crate::channels::attachments::{build_inbound_text, message_attachment};

    let mut png = tempfile::NamedTempFile::new().unwrap();
    png.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00])
        .unwrap();

    let attachment = message_attachment(
        png.path().to_path_buf(),
        "doc.png".to_string(),
        "image/png".to_string(),
        9,
    );
    let attachments = vec![attachment.clone()];
    let inbound_text = build_inbound_text("Check the doc and fix the issue.", &attachments);

    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "Would you like me to get more detailed information for any specific trial(s)?",
        ),
        MockProvider::text_response("Summary of prior conversation."),
        MockProvider::text_response("I reviewed the uploaded document and identified the issue."),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message(
            "vision_compaction_test",
            "These are the NCT trial numbers: NCT06737964 and NCT06737965.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let _ = harness
        .agent
        .handle_message_with_attachments(
            "vision_compaction_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        call_log.len() >= 3,
        "file upload with attachment should trigger compaction LLM call; got {} calls",
        call_log.len()
    );
    assert!(
        call_log.iter().any(|call| {
            call.messages.iter().any(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("Summary of prior conversation."))
            })
        }),
        "expected compaction summary in an LLM call"
    );
}

/// Photo/document uploads without a caption must not inherit tool requirements from
/// the channel-injected inbox path in the file stub metadata.
#[tokio::test]
async fn test_attachment_stub_metadata_does_not_force_tool_required_loop() {
    use std::io::Write;

    use crate::channels::attachments::{build_inbound_text, message_attachment};
    use crate::traits::ToolChoiceMode;

    let mut png = tempfile::NamedTempFile::new().unwrap();
    png.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00])
        .unwrap();

    let attachment = message_attachment(
        png.path().to_path_buf(),
        "photo.jpg".to_string(),
        "image/jpeg".to_string(),
        9,
    );
    let inbound_text = build_inbound_text("", std::slice::from_ref(&attachment));

    let vision_reply = MockProvider::text_response("This image looks like a small PNG file.");
    let provider = MockProvider::with_responses(vec![
        vision_reply.clone(),
        vision_reply,
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    let reply = harness
        .agent
        .handle_message_with_attachments(
            "attachment_stub_intent_test",
            &inbound_text,
            &[attachment],
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert!(
        reply.contains("PNG") || reply.contains("image"),
        "expected vision/text reply, got: {reply}"
    );

    let call_log = harness.provider.call_log.lock().await;
    let forced_tool_calls = call_log
        .iter()
        .filter(|call| call.options.tool_choice == ToolChoiceMode::Required)
        .count();
    assert_eq!(
        forced_tool_calls, 0,
        "stub inbox path must not trigger tool_choice=Required recovery loop"
    );
}
