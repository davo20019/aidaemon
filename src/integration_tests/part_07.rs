// ==================== Orchestration Integration Tests ====================

#[tokio::test]
async fn test_orchestration_uniform_models_no_routing() {
    // With uniform models (no router), consultant pass doesn't activate,
    // so orchestration routing doesn't happen — simple messages get direct responses.
    let provider = MockProvider::new(); // Returns "Mock response"
    let harness = setup_test_agent(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Hello!",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Mock response");

    // No goals — uniform models bypass consultant pass / routing
    let goals = harness.state.get_active_goals().await.unwrap();
    assert!(goals.is_empty(), "No goals with uniform models");
}

#[tokio::test]
async fn test_orchestration_simple_falls_through_to_full_loop() {
    // Orchestration enabled, non-uniform models → consultant pass activates → routing
    // Consultant says "needs tools" (not can_answer_now), simple task → full agent loop
    let provider = MockProvider::with_responses(vec![
        // 1st call: consultant pass — analysis that triggers needs_tools
        MockProvider::text_response(
            "I'll help you check the system info.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[]}",
        ),
        // 2nd call: full agent loop — tool call
        MockProvider::tool_call_response("system_info", "{}"),
        // 3rd call: full agent loop — final response
        MockProvider::text_response("Your system is running macOS."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "check system info",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Should get the full agent loop's response
    assert_eq!(response, "Your system is running macOS.");

    // No goals should be created (simple tasks don't create goals)
    let goals = harness.state.get_active_goals().await.unwrap();
    assert!(goals.is_empty(), "Simple tasks should not create goals");
}

#[tokio::test]
async fn test_orchestration_complex_creates_goal() {
    // Deterministic pre-routing classifies the request as complex based on
    // keyword heuristics (action markers + compound keywords), creates a goal,
    // and spawns a task lead synchronously (no self_ref in tests).
    // No consultant LLM call — routing is purely deterministic.
    let provider = MockProvider::with_responses(vec![
        // Task lead's LLM call — deterministic routing creates the goal
        // without an LLM call, then the task lead runs the agent loop.
        MockProvider::text_response("I'll start working on building your website."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    // User text must trigger looks_like_complex_request_fallback():
    // >=3 action markers (analyze, compare, identify, find, summarize) + compound ("and"/"then")
    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Analyze the requirements and compare authentication libraries, then identify the best frameworks, find suitable database solutions, and summarize the deployment options for a full-stack website with CI/CD pipeline",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Should get a response (the exact text depends on how many LLM calls the agent loop makes)
    assert!(!response.is_empty(), "Should return a non-empty response");

    // The key assertion: a goal should have been created
    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1, "Complex request should create a goal");
    // Task leads are always-on, so the goal is completed after the task lead succeeds
    assert_eq!(goals[0].status, "completed");
    assert!(goals[0]
        .description
        .contains("Analyze the requirements"));
}

#[tokio::test]
async fn test_orchestration_complex_internal_maintenance_does_not_create_goal() {
    // Deterministic routing classifies this as complex (action markers trigger
    // looks_like_complex_request_fallback), but handle_complex_intent detects
    // maintenance keywords and returns a canned response without creating a goal.
    // No LLM calls needed — both routing and maintenance detection are deterministic.
    let provider = MockProvider::new(); // No scripted responses needed
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    // User text triggers complex classification (analyze, identify, find, report = 4 action
    // markers + "and"/"then") AND matches is_internal_maintenance_intent (process embeddings
    // + consolidate memories + decay old facts).
    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Analyze the knowledge base and identify stale entries, then process embeddings, consolidate memories, find outdated data, and report on decay old facts",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("already runs via built-in background jobs"),
        "Expected maintenance-routing response, got: {response}"
    );

    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert!(
        goals.is_empty(),
        "Internal maintenance intent should not create a goal"
    );
}

#[tokio::test]
async fn test_orchestration_simple_stall_detection_in_full_loop() {
    // Simple tasks now go through full agent loop which has its own stall detection.
    // After the consultant pass, repeated identical tool calls should be detected.
    let provider = MockProvider::with_responses(vec![
        // 1st call: consultant pass
        MockProvider::text_response(
            "I'll run a command for you.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[]}",
        ),
        // 2nd call: intent gate narration (full loop requires narration on first tool iteration)
        MockProvider::tool_call_response("system_info", "{}"),
        // Repeated identical tool calls — stall detection should kick in
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        // Enough repetitions to trigger stall detection
        MockProvider::text_response("Should not reach here"),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "run a quick check",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Full loop stall detection produces graceful responses
    assert!(
        !response.is_empty(),
        "Should return a non-empty response even on stall"
    );
}

#[tokio::test]
async fn test_orchestration_simple_uses_full_loop_with_all_tools() {
    // Simple tasks now use the full agent loop with all tools available.
    // Verify the agent can complete a simple task through the full loop.
    let provider = MockProvider::with_responses(vec![
        // 1st call: consultant pass
        MockProvider::text_response(
            "I'll help with that.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[]}",
        ),
        // 2nd call: full agent loop makes a tool call
        MockProvider::tool_call_response("system_info", "{}"),
        // 3rd call: full agent loop returns final response
        MockProvider::text_response("Diagnostics complete. All systems normal."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "run diagnostics",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Diagnostics complete. All systems normal.");
}

#[tokio::test]
async fn test_personal_recall_challenge_scopes_tools_and_reaffirms() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "Let me verify memory first.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[]}",
        ),
        {
            let mut resp = MockProvider::tool_call_response(
                "browser",
                r#"{"action":"navigate","url":"https://example.com"}"#,
            );
            resp.content = Some("I'll check additional sources.".to_string());
            resp
        },
        {
            let mut resp = MockProvider::tool_call_response(
                "manage_people",
                r#"{"action":"view","person_name":"__unknown_person_for_recall_guardrail__"}"#,
            );
            resp.content = Some("I'll re-check your stored people data.".to_string());
            resp
        },
        MockProvider::text_response("I still do not have that information saved in memory."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Are you sure I have pets?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("do not have"),
        "Expected no-information reaffirmation after targeted memory re-check, got: {}",
        response
    );
    assert!(
        harness.provider.call_count().await <= 4,
        "Challenge turn should stay bounded and not spiral"
    );

    let history = harness.state.get_history("test_session", 50).await.unwrap();
    let browser_tool_msgs: Vec<&crate::traits::Message> = history
        .iter()
        .filter(|m| m.role == "tool" && m.tool_name.as_deref() == Some("browser"))
        .collect();
    let scoped_block = !browser_tool_msgs.is_empty()
        && browser_tool_msgs.iter().all(|m| {
            m.content.as_deref().is_some_and(|c| {
                c.contains("Personal-memory recall")
                    || c.contains("not a real tool")
                    || c.contains("Unknown tool")
            })
        });
    assert!(
        scoped_block,
        "Expected out-of-scope browser tool call to be blocked for personal recall turn"
    );
}

#[tokio::test]
async fn test_personal_recall_challenge_inherits_previous_turn_context() {
    // With deterministic routing (no consultant LLM pass), each turn makes a
    // single LLM call through the normal agent loop. The deterministic routing
    // classifies both messages as Simple (no action markers, no schedule) and
    // falls through to the execution loop.
    let provider = MockProvider::with_responses(vec![
        // Turn 1: execution loop — direct answer
        MockProvider::text_response("I don't have information about pets."),
        // Turn 2: execution loop — reaffirmation
        MockProvider::text_response("I still do not have that information saved in memory."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let first = harness
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
    assert!(
        first.contains("don't have information about pets"),
        "Expected personal-recall context, got: {}",
        first
    );

    let second = harness
        .agent
        .handle_message(
            "test_session",
            "Are you sure?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(
        second.contains("do not have") || second.contains("requires running tools"),
        "Expected no-information reaffirmation or tools-unavailable message, got: {}",
        second
    );
    // No consultant pass: 1 LLM call per turn x 2 turns = 2
    assert!(
        harness.provider.call_count().await <= 3,
        "Follow-up challenge should stay bounded and not spiral"
    );
}

#[tokio::test]
async fn test_orchestration_scheduled_one_shot_creates_pending_confirmation() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I'll schedule that.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[],\"schedule\":\"in 2h\",\"schedule_type\":\"one_shot\"}",
    )]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "deploy in 2 hours",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("Reply **confirm**"),
        "Expected confirmation prompt for scheduled goal"
    );

    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0].goal_type, "finite");
    assert_eq!(goals[0].status, "pending_confirmation");
    let schedules = harness
        .state
        .get_schedules_for_goal(&goals[0].id)
        .await
        .unwrap();
    assert_eq!(schedules.len(), 1);
    assert!(schedules[0].is_one_shot);
}

#[tokio::test]
async fn test_orchestration_scheduled_malformed_schedule_recovers_from_user_text() {
    // E2E regression: LLM sometimes emits schedule="2 minutes" while the user
    // said "in 2 minutes". The scheduler path should still be taken.
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "Sure.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[],\"schedule\":\"2 minutes\",\"schedule_type\":\"one_shot\"}",
    )]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "check disk space in 2 minutes",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("Reply **confirm**"),
        "Expected confirmation prompt for scheduled goal"
    );

    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0].status, "pending_confirmation");
    let schedules = harness
        .state
        .get_schedules_for_goal(&goals[0].id)
        .await
        .unwrap();
    assert_eq!(schedules.len(), 1);
    assert!(schedules[0].is_one_shot);
}

#[tokio::test]
async fn test_orchestration_scheduled_recurring_creates_pending_confirmation() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I'll schedule recurring monitoring.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[],\"schedule\":\"every 6h\",\"schedule_type\":\"recurring\"}",
    )]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "monitor API health every 6h",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("Reply **confirm**"),
        "Expected confirmation prompt for recurring schedule"
    );

    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0].goal_type, "continuous");
    assert_eq!(goals[0].status, "pending_confirmation");
    assert_eq!(goals[0].budget_per_check, Some(50_000));
    assert_eq!(goals[0].budget_daily, Some(200_000));
    let schedules = harness
        .state
        .get_schedules_for_goal(&goals[0].id)
        .await
        .unwrap();
    assert_eq!(schedules.len(), 1);
    assert!(!schedules[0].is_one_shot);
}

#[tokio::test]
async fn test_orchestration_schedule_confirm_activates_goal() {
    // Deterministic routing detects "in 2 hours" via schedule heuristic (no LLM call).
    // "confirm" is handled by the confirmation gate (also no LLM call).
    // Total LLM calls across both messages = 0.
    let provider = MockProvider::new(); // No scripted responses needed
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message(
            "test_session",
            "deploy in 2 hours",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let confirm_response = harness
        .agent
        .handle_message(
            "test_session",
            "confirm",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(confirm_response.contains("Scheduled:"));

    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0].status, "active");
    assert_eq!(
        harness.provider.call_count().await,
        0,
        "Both schedule creation (deterministic) and confirm (gate) should work without LLM calls"
    );
}

#[tokio::test]
async fn test_orchestration_schedule_cancel_removes_goal() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I'll schedule that.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[],\"schedule\":\"in 2h\",\"schedule_type\":\"one_shot\"}",
    )]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message(
            "test_session",
            "deploy in 2 hours",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let cancel_response = harness
        .agent
        .handle_message(
            "test_session",
            "cancel",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(cancel_response.contains("cancelled"));

    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0].status, "cancelled");
}

#[tokio::test]
async fn test_orchestration_targeted_cancel_text_does_not_auto_cancel_session_goal() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "Understood.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"cancel_intent\":true,\"cancel_scope\":\"targeted\",\"complexity\":\"simple\"}",
        ),
        // needs_tools=true blocks text-only responses, so executor must use a tool call
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Please share the goal ID to cancel that specific goal."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let morning_goal = Goal::new_continuous(
        "Send me a slack message at 7:00 am EST tomorrow with a positive message",
        "test_session",
        Some(2000),
        Some(20000),
    );
    harness.state.create_goal(&morning_goal).await.unwrap();

    let english_goal = Goal::new_continuous(
        "English Research: Researching English pronunciation/phonetics for Spanish speakers",
        "other_session",
        Some(2000),
        Some(20000),
    );
    harness.state.create_goal(&english_goal).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "cancel this goal: English Research: Researching English",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response,
        "Please share the goal ID to cancel that specific goal."
    );
    assert_eq!(
        harness.provider.call_count().await,
        3,
        "Targeted cancel text should not trigger session-wide auto-cancel shortcut"
    );

    let morning_after = harness
        .state
        .get_goal(&morning_goal.id)
        .await
        .unwrap()
        .unwrap();
    let english_after = harness
        .state
        .get_goal(&english_goal.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(morning_after.status, "active");
    assert_eq!(english_after.status, "active");
}

#[tokio::test]
async fn test_orchestration_schedule_new_message_cancels_pending() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "I'll schedule that.\n[INTENT_GATE] {\"can_answer_now\":false,\"needs_tools\":true,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[],\"schedule\":\"in 2h\",\"schedule_type\":\"one_shot\"}",
        ),
        MockProvider::text_response(
            "[INTENT_GATE] {\"can_answer_now\":true,\"needs_tools\":false,\"needs_clarification\":false,\"clarifying_question\":\"\",\"missing_info\":[],\"complexity\":\"knowledge\"}",
        ),
        MockProvider::text_response("Rust is a systems programming language."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let _ = harness
        .agent
        .handle_message(
            "test_session",
            "deploy in 2 hours",
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
            "what is rust?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0].status, "cancelled");
}
