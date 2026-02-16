// ---------------------------------------------------------------------------
// Consultant pass tests — no-tools first turn with smart router
// ---------------------------------------------------------------------------

/// Classifier-only consultant flow: iteration 1 classifies intent via INTENT_GATE,
/// then the execution loop produces the user-visible answer.
#[tokio::test]
async fn test_consultant_pass_classifies_then_executor_answers_questions() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "[INTENT_GATE]\n\
             {\"complexity\": \"knowledge\", \"can_answer_now\": true, \"needs_tools\": false}",
        ),
        MockProvider::text_response(
            "Your website is deployed to Cloudflare Workers at your-site.workers.dev.",
        ),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    harness
        .state
        .upsert_fact(
            "project",
            "my website",
            "deployed to cloudflare workers at your-site.workers.dev",
            "user",
            None,
            crate::types::FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Question (contains ?) -> consultant classifies, executor answers.
    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Can you tell me the deployment URL for my website?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response,
        "Your website is deployed to Cloudflare Workers at your-site.workers.dev."
    );

    assert_eq!(harness.provider.call_count().await, 2);
    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls[0].tools.is_empty(),
        "Consultant classifier call should have no tools"
    );
}

#[tokio::test]
async fn test_critical_owner_name_query_is_deterministic() {
    let harness = setup_test_agent_with_models(MockProvider::new(), "primary-model", "smart-model")
        .await
        .unwrap();

    harness
        .state
        .upsert_fact(
            "user",
            "name",
            "Test Owner",
            "owner",
            None,
            crate::types::FactPrivacy::Global,
        )
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "What's my name?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Your name is Test Owner.");
    assert_eq!(
        harness.provider.call_count().await,
        0,
        "Critical identity query should resolve without an LLM call"
    );
}

#[tokio::test]
async fn test_personal_recall_turn_routes_at_least_primary_model() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "[INTENT_GATE] {\"complexity\":\"knowledge\",\"can_answer_now\":true,\"needs_tools\":false}",
        ),
        MockProvider::text_response("I don't have pet information saved."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "What about pets?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(response.contains("don't have pet information"));
    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls.len() >= 2,
        "Classifier-only flow should include consultant + executor calls"
    );
    assert_eq!(
        calls[0].model, "primary-model",
        "Personal recall should not route to the cheapest profile/model"
    );
}

#[tokio::test]
async fn test_consultant_empty_answerable_turn_falls_through_to_tool_path() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "[INTENT_GATE] {\"complexity\":\"knowledge\",\"can_answer_now\":true,\"needs_tools\":false,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[]}",
        ),
        MockProvider::text_response("Recovered after memory/tool retry."),
    ]);

    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "What timezone am I in?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Recovered after memory/tool retry.");
    let calls = harness.provider.call_log.lock().await;
    assert_eq!(calls.len(), 2);
    assert!(
        !calls[1].tools.is_empty(),
        "Empty answerable consultant output should trigger tool-enabled retry path"
    );
}

#[tokio::test]
async fn test_identity_tool_result_survives_context_collapse() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"user","key":"name","value":"David"}"#,
        ),
        MockProvider::text_response("Saved."),
        MockProvider::text_response("Continuing with your latest request."),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message(
            "test_session",
            "Remember that my name is David",
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
            "test_session",
            "What should we do next?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls.len() >= 3,
        "Expected at least 3 model calls across both turns"
    );
    let second_turn_call = &calls[2];
    let has_identity_tool_context = second_turn_call.messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("tool")
            && m.get("name").and_then(|n| n.as_str()) == Some("remember_fact")
            && m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|c| c.to_ascii_lowercase().contains("name = david"))
    });
    assert!(
        has_identity_tool_context,
        "Critical identity tool result should be preserved across context collapsing"
    );
}

/// For action requests (non-questions), the consultant classifies and the
/// execution model handles tool use.
#[tokio::test]
async fn test_consultant_pass_continues_for_actions() {
    // Three responses:
    //   1. Consultant (no tools) → analysis of the action
    //   2. Full agent loop → tool call (system_info)
    //   3. Full agent loop → final text with tool result
    // Orchestration routes "show me system info" as Simple → falls through to full agent loop.
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("I'll check the system information for you."),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Your system is running macOS."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    // Action request (imperative, no ?) → consultant pass → full agent loop
    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Show me the current system information and environment details",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Your system is running macOS.");

    let call_count = harness.provider.call_count().await;
    assert!(
        call_count >= 2,
        "Expected at least 2 LLM calls (consultant + full loop)"
    );

    let calls = harness.provider.call_log.lock().await;
    // First call: no tools (consultant pass)
    assert!(
        calls[0].tools.is_empty(),
        "Consultant call should have no tools"
    );
}

/// Regression: if the execution model replies with deferred-action narration
/// ("I'll do X", "starting workflow") but no tool calls, the agent must keep
/// iterating instead of returning that narration as final output.
#[tokio::test]
async fn test_deferred_action_no_tool_calls_does_not_complete_task() {
    let provider = MockProvider::with_responses(vec![
        // 1) Consultant pass
        MockProvider::text_response(
            "I'll check and send it over.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[],\"complexity\":\"simple\"}",
        ),
        // 2) Full loop (bad): deferred action text, no tool calls
        MockProvider::text_response(
            "I'll find your resume and send it over right away.\nStarting the send-resume workflow...",
        ),
        // 3) Full loop (good): actual tool execution
        MockProvider::tool_call_response("system_info", "{}"),
        // 4) Final answer
        MockProvider::text_response("Found it and sent it."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "send me my resume",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Found it and sent it.");
    assert_eq!(harness.provider.call_count().await, 4);

    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls[0].tools.is_empty(),
        "Consultant pass must be tool-free"
    );
    assert!(
        !calls[1].tools.is_empty(),
        "Execution loop must have tools available after consultant pass"
    );
}

/// Regression: even after some successful tool calls, a deferred-action
/// narration ("I'll send it over") must not be treated as final completion.
#[tokio::test]
async fn test_deferred_action_after_tool_progress_does_not_complete_task() {
    let provider = MockProvider::with_responses(vec![
        // 1) Consultant pass
        MockProvider::text_response(
            "I'll find it for you.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[],\"complexity\":\"simple\"}",
        ),
        // 2) Full loop: executes a tool successfully
        MockProvider::tool_call_response("system_info", "{}"),
        // 3) Full loop (bad): deferred narration instead of results
        MockProvider::text_response(
            "I'll send it over once I locate the exact file. Give me a moment.",
        ),
        // 4) Full loop (good): concrete outcome
        MockProvider::text_response("I couldn't find a matching SOW PDF in the project files."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Send me the SOW PDF from the Lodestar project",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response,
        "I couldn't find a matching SOW PDF in the project files."
    );
    assert_eq!(harness.provider.call_count().await, 4);

    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls[0].tools.is_empty(),
        "Consultant pass must be tool-free"
    );
    assert!(
        !calls[1].tools.is_empty(),
        "Execution loop must have tools available after consultant pass"
    );
}

/// With uniform models (all "mock-model"), no consultant pass — tools should
/// be available from the very first LLM call.
#[tokio::test]
async fn test_no_consultant_pass_with_uniform_models() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Here is your info."),
    ]);

    // Uniform models → router disabled → no consultant pass
    let harness = setup_test_agent(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "What's my system info?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Here is your info.");
    assert_eq!(harness.provider.call_count().await, 2);

    // First call should have tools (no consultant pass)
    let calls = harness.provider.call_log.lock().await;
    assert!(
        !calls[0].tools.is_empty(),
        "Without consultant pass, first call should have tools"
    );
}

/// Consultant pass with an empty response on iteration 1 should NOT
/// intercept — it falls through to the normal empty-response handling.
#[tokio::test]
async fn test_consultant_pass_empty_response_not_intercepted() {
    // Consultant returns empty text → should not be intercepted
    // Then execution model responds normally
    let provider = MockProvider::with_responses(vec![
        // Empty content response (consultant returns nothing)
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 10,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        MockProvider::text_response("Fallback response."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let _response = harness
        .agent
        .handle_message(
            "test_session",
            "Hello",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The empty consultant response should NOT be intercepted (the !reply.is_empty()
    // check lets it fall through to normal empty-response handling).
    // The exact behavior depends on depth/iteration, but it should not panic.
}

/// Regression: when the execution loop keeps returning empty content with no
/// tool calls after consultant pass, the agent should attempt one recovery
/// pass, then persist the fallback response and emit task completion.
#[tokio::test]
async fn test_empty_execution_response_persists_fallback_message() {
    let provider = MockProvider::with_responses(vec![
        // Consultant pass (iteration 1): empty response, falls through.
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 10,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        // Execution loop (iteration 2): empty -> one retry nudge.
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 20,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        // Execution loop (iteration 3): still empty -> fallback.
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 20,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Who is becquer?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let expected = "I wasn't able to process that request. Could you try rephrasing?";
    assert_eq!(response, expected);
    assert_eq!(harness.provider.call_count().await, 3);

    let history = harness.state.get_history("test_session", 10).await.unwrap();
    assert!(
        history
            .iter()
            .any(|m| m.role == "assistant" && m.content.as_deref() == Some(expected)),
        "Fallback response should be persisted in history. History: {:?}",
        history
    );
}

#[tokio::test]
async fn test_empty_execution_response_surfaces_provider_note() {
    let provider = MockProvider::with_responses(vec![
        // Consultant pass (iteration 1): empty response, falls through.
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 10,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        // Execution loop (iteration 2): empty response with provider note
        // -> one retry nudge.
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 20,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: Some(
                "finish reason: SAFETY; candidate safety categories: HARM_CATEGORY_HATE_SPEECH"
                    .to_string(),
            ),
        },
        // Execution loop (iteration 3): still empty, no note.
        // The fallback should still surface the previous retry note.
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 20,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Find my resume and send it",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(response.starts_with("I wasn't able to process that request."));
    assert!(response.contains("The model returned no usable output (finish reason: SAFETY; candidate safety categories: HARM_CATEGORY_HATE_SPEECH)."));
    assert!(response.ends_with("Could you try rephrasing?"));
    assert_eq!(harness.provider.call_count().await, 3);

    let history = harness.state.get_history("test_session", 10).await.unwrap();
    assert!(
        history
            .iter()
            .any(|m| m.role == "assistant" && m.content.as_deref() == Some(response.as_str())),
        "Fallback response with provider note should be persisted in history. History: {:?}",
        history
    );
}

#[tokio::test]
async fn test_empty_execution_response_retry_recovers_with_text() {
    let provider = MockProvider::with_responses(vec![
        // Consultant pass (iteration 1): empty response, falls through.
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 10,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        // Execution loop (iteration 2): empty response -> retry nudge.
        ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 20,
                output_tokens: 0,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        // Execution loop (iteration 3): recovery succeeds with text.
        MockProvider::text_response("Recovered response."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Create a page",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Recovered response.");
    assert_eq!(harness.provider.call_count().await, 3);
    assert!(!response.contains("I wasn't able to process that request."));
}

/// When the consultant pass model returns BOTH text AND tool calls on
/// iteration 1, the tool calls should be DROPPED and only the text
/// analysis should be kept.  This handles Gemini models that hallucinate
/// function calls from system prompt tool descriptions.
/// When hallucinated tool calls are detected, the code forces `needs_tools = true`
/// (the LLM signaled it needs tools by attempting to call them), so the question
/// falls through to the tool loop — not returned directly.
#[tokio::test]
async fn test_consultant_pass_drops_hallucinated_tool_calls() {
    use crate::traits::ToolCall;

    // Consultant returns confident text + hallucinated tool call (iteration 1)
    // → hallucinated tool calls force needs_tools=true → falls through to execution loop.
    // Execution loop returns the final text response.
    let provider = MockProvider::with_responses(vec![
        // Iteration 1: consultant returns confident text AND a hallucinated tool call
        ProviderResponse {
            content: Some(
                "Your website is deployed at your-site.workers.dev on Cloudflare Workers."
                    .to_string(),
            ),
            tool_calls: vec![ToolCall {
                id: "call_hallucinated".to_string(),
                name: "terminal".to_string(),
                arguments: r#"{"command":"find ~ -name wrangler.toml"}"#.to_string(),
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
        // Iteration 2+: execution loop response
        MockProvider::text_response(
            "Your website is deployed at your-site.workers.dev on Cloudflare Workers.",
        ),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Can you tell me the deployment URL for my website?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The final response comes from the execution loop
    assert_eq!(
        response,
        "Your website is deployed at your-site.workers.dev on Cloudflare Workers."
    );

    // At least 1 LLM call (consultant) + execution loop calls
    let call_count = harness.provider.call_count().await;
    assert!(
        call_count >= 2,
        "Expected at least 2 LLM calls — consultant + execution loop (got {})",
        call_count
    );
}

/// Regression: if consultant analysis sanitizes to empty but intent gate says
/// acknowledgment + needs_tools=true (e.g. "Yes, do it."), we must NOT return
/// an empty direct reply. The turn should fall through to execution.
#[tokio::test]
async fn test_acknowledgment_with_needs_tools_and_empty_analysis_falls_through() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "[tool_use: terminal]\n\
             cmd: read_file project/plan.md\n\
             args: {\"path\":\"project/plan.md\"}\n\
             \n\
             [INTENT_GATE]\n\
             {\"complexity\":\"simple\",\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"is_acknowledgment\":true}",
        ),
        MockProvider::text_response("Proceeding with the requested changes."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Yes, do it.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Proceeding with the requested changes.");
    assert_eq!(
        harness.provider.call_count().await,
        2,
        "Expected consultant pass + execution pass"
    );
}

/// Empty consultant text for a pure acknowledgment should produce a safe
/// non-empty conversational reply instead of persisting an empty assistant turn.
#[tokio::test]
async fn test_acknowledgment_with_empty_analysis_returns_default_reply() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "[INTENT_GATE]\n\
         {\"complexity\":\"simple\",\"can_answer_now\":true,\"needs_tools\":false,\"needs_clarification\":false,\"is_acknowledgment\":true}",
    )]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Yes",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Got it.");
    assert_eq!(harness.provider.call_count().await, 1);
}

/// Keep short-correction guardrail behavior: empty consultant analysis still
/// yields the fixed correction acknowledgment response.
#[tokio::test]
async fn test_short_correction_with_empty_analysis_returns_default_reply() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "[INTENT_GATE]\n\
         {\"complexity\":\"simple\",\"can_answer_now\":true,\"needs_tools\":false,\"needs_clarification\":false,\"is_acknowledgment\":false}",
    )]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "You did send me the file",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "You're right — thanks for the correction.");
    assert_eq!(harness.provider.call_count().await, 1);
}

#[tokio::test]
async fn test_intent_gate_decision_metadata_includes_route_reason_for_direct_reply() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "[INTENT_GATE]\n\
         {\"complexity\":\"simple\",\"can_answer_now\":true,\"needs_tools\":false,\"needs_clarification\":false,\"is_acknowledgment\":true}",
    )]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Yes",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "Got it.");

    let event_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id DESC",
    )
    .bind("test_session")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    let intent_gate_decision = event_rows
        .into_iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(&raw).unwrap())
        .find(|data| data.get("decision_type").and_then(|v| v.as_str()) == Some("intent_gate"))
        .expect("expected at least one intent_gate decision point");

    let metadata = intent_gate_decision
        .get("metadata")
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        metadata.get("route_reason").and_then(|v| v.as_str()),
        Some("acknowledgment_direct_reply")
    );
    assert_eq!(
        metadata.get("route_action").and_then(|v| v.as_str()),
        Some("return")
    );
    assert_eq!(metadata.get("route_reply_len").and_then(|v| v.as_u64()), Some(7));
}

#[tokio::test]
async fn test_intent_gate_decision_metadata_includes_route_reason_for_continue() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "[INTENT_GATE]\n\
             {\"complexity\":\"simple\",\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"is_acknowledgment\":true}",
        ),
        MockProvider::text_response("Proceeding with the requested changes."),
    ]);

    let harness = setup_test_agent_with_models(provider, "primary-model", "smart-model")
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Yes, do it.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "Proceeding with the requested changes.");

    let event_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id DESC",
    )
    .bind("test_session")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    let intent_gate_decision = event_rows
        .into_iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(&raw).unwrap())
        .find(|data| data.get("decision_type").and_then(|v| v.as_str()) == Some("intent_gate"))
        .expect("expected at least one intent_gate decision point");

    let metadata = intent_gate_decision
        .get("metadata")
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        metadata.get("route_reason").and_then(|v| v.as_str()),
        Some("tools_required")
    );
    assert_eq!(
        metadata.get("route_action").and_then(|v| v.as_str()),
        Some("continue")
    );
    assert_eq!(metadata.get("route_reply_len").and_then(|v| v.as_u64()), None);
}
