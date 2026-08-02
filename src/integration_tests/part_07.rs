// ==================== Orchestration Integration Tests ====================

#[tokio::test]
async fn test_orchestration_uniform_models_no_routing() {
    // With uniform models, messages go directly through the normal model loop.
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

    // No goals — uniform models bypass orchestration routing
    let goals = harness.state.get_active_goals().await.unwrap();
    assert!(goals.is_empty(), "No goals with uniform models");
}

#[tokio::test]
async fn test_orchestration_simple_falls_through_to_full_loop() {
    // The request goes through the full agent loop.
    let provider = MockProvider::with_responses(vec![
        // 1st call: deferral text, bounced by the deferred-action gate
        MockProvider::text_response("I'll check the system info and get back to you."),
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
async fn test_orchestration_simple_stall_detection_in_full_loop() {
    // Simple tasks now go through full agent loop which has its own stall detection.
    // After the first routing pass, repeated identical tool calls should be detected.
    let provider = MockProvider::with_responses(vec![
        // 1st call: deferral text, bounced by the deferred-action gate
        MockProvider::text_response("I'll run a command for you."),
        // 2nd call: real tool call
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
        // 1st call: deferral text, bounced by the deferred-action gate
        MockProvider::text_response("I'll run the diagnostics and get back to you."),
        // 2nd call: deferred-action retry produces a real tool call
        MockProvider::tool_call_response("system_info", "{}"),
        // 3rd-5th calls: final response, repeated to survive mutation-contract
        // nudges ("run" triggers expects_mutation=true → up to 2 extra iterations
        // before text response is accepted).
        MockProvider::text_response("Diagnostics complete. All systems normal."),
        MockProvider::text_response("Diagnostics complete. All systems normal."),
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
        // Recall turns accept substantive text replies readily, so the first
        // response is the out-of-scope tool call this test is about.
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
                    || c.contains("should be answered directly in plain text")
            })
        });
    // The browser tool call may be blocked by the personal-recall scope
    // guard OR by the text-only prelude check (for non-mutation turns).
    // Either path prevents execution.
    assert!(
        scoped_block,
        "Expected out-of-scope browser tool call to be blocked for personal recall turn"
    );
}

#[tokio::test]
async fn test_personal_recall_challenge_inherits_previous_turn_context() {
    // Each turn makes a model call through the normal agent loop.
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
    // No text-only pre-pass: 1 LLM call per turn x 2 turns = 2
    assert!(
        harness.provider.call_count().await <= 3,
        "Follow-up challenge should stay bounded and not spiral"
    );
}

#[tokio::test]
async fn test_general_reaffirmation_challenge_injects_prior_answer_anchor() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("There are 3 R's in strawberry."),
        MockProvider::text_response("Yes — strawberry has 3 R's."),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    let first = harness
        .agent
        .handle_message(
            "reaffirm_anchor_test",
            "How many R's in strawberry?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(
        first.contains("3 R"),
        "Expected strawberry answer, got: {}",
        first
    );

    let _second = harness
        .agent
        .handle_message(
            "reaffirm_anchor_test",
            "Are you sure?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let challenge_call = call_log.last().expect("challenge turn LLM call");
    let has_anchor = challenge_call.messages.iter().any(|message| {
        message
            .get("content")
            .and_then(|content| content.as_str())
            .is_some_and(|text| {
                text.contains("REAFFIRMATION CHALLENGE")
                    && text.contains("How many R's in strawberry?")
                    && text.contains("There are 3 R's in strawberry.")
            })
    });
    assert!(
        has_anchor,
        "Challenge turn should inject reaffirmation anchor for the immediately previous exchange: {:?}",
        challenge_call.messages
    );
}

#[tokio::test]
async fn test_compound_message_with_challenge_keyword_skips_reaffirmation_anchor() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("There are 3 R's in strawberry."),
        MockProvider::text_response("Here is the blog post about Ecuador."),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    harness
        .agent
        .handle_message(
            "reaffirm_anchor_negative_test",
            "How many R's in strawberry?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Contains the word "really" but is a new task, not a vague challenge of
    // the previous answer — the anchor directive must NOT be injected.
    harness
        .agent
        .handle_message(
            "reaffirm_anchor_negative_test",
            "I really need you to write a blog post about Ecuador",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let second_call = call_log.last().expect("second turn LLM call");
    let has_anchor = second_call.messages.iter().any(|message| {
        message
            .get("content")
            .and_then(|content| content.as_str())
            .is_some_and(|text| text.contains("REAFFIRMATION CHALLENGE"))
    });
    assert!(
        !has_anchor,
        "Compound new-task message must not be pinned to the previous exchange: {:?}",
        second_call.messages
    );
}

#[tokio::test]
async fn test_orchestration_targeted_cancel_text_does_not_auto_cancel_session_goal() {
    let provider = MockProvider::with_responses(vec![
        // Deterministic cancel detection classifies this as a targeted cancel,
        // which falls through to the normal loop instead of auto-cancelling.
        // The deferral below is bounced by the deferred-action gate.
        MockProvider::text_response("I'll look into which goal you mean."),
        // Deferred-action retry produces a real tool call
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
async fn test_zero_tool_fabricated_mutation_claim_is_blocked() {
    // Reproduces the 2026-06-06 attribution-run turn-10 bug: the model
    // claimed "I have deleted the folder" without making a single tool
    // call, and the completion phase accepted it. A past-tense side-effect
    // claim in a zero-tool task with a mutation contract must be treated
    // like a deferred action: nudged with a hard tool requirement, never
    // accepted as the final answer.
    let fabrication = "I have deleted the folder /tmp/fab-test entirely.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(fabrication),
        MockProvider::text_response(fabrication),
        MockProvider::text_response(
            "I could not verify the deletion because no command was run.",
        ),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Delete the folder /tmp/fab-test entirely.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The fabricated claim must not survive as the final answer.
    assert!(
        !response.contains("I have deleted"),
        "fabricated zero-tool mutation claim was accepted as completion: {response}"
    );

    // The loop must have continued past the first reply, and the hard
    // tool-call requirement directive must have been injected.
    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls.len() >= 2,
        "completion was accepted on the first iteration (calls={})",
        calls.len()
    );
    let nudged = calls.iter().skip(1).any(|c| {
        c.messages.iter().any(|m| {
            m["content"]
                .as_str()
                .is_some_and(|t| t.contains("MUST include at least one tool call"))
        })
    });
    assert!(
        nudged,
        "DeferredToolCallRequired directive was not injected after the fabricated claim"
    );

    let event_store = crate::events::EventStore::new(harness.state.pool())
        .await
        .expect("event store from harness pool");
    let events = event_store
        .query_recent_events("test_session", 200)
        .await
        .expect("recent events");
    let saw_mutation_gate_warning = events.iter().any(|event| {
        let Ok(data) = event.parse_data::<crate::events::DecisionPointData>() else {
            return false;
        };
        data.decision_type == crate::events::DecisionType::PostExecutionValidation
            && data
                .metadata
                .get("condition")
                .and_then(serde_json::Value::as_str)
                == Some("expects_mutation_gate_evaluated")
            && data
                .metadata
                .get("assistant_claimed_mutation")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && data
                .metadata
                .get("mutation_tool_calls_count")
                .and_then(serde_json::Value::as_u64)
                == Some(0)
            && data
                .metadata
                .get("outcome")
                .and_then(serde_json::Value::as_str)
                == Some("blocked_claimed_mutation_without_tool")
    });
    assert!(
        saw_mutation_gate_warning,
        "mutation gate did not emit explicit warning telemetry for fabricated zero-tool mutation claim"
    );
}

#[tokio::test]
async fn test_zero_tool_fabricated_delegation_claim_is_blocked() {
    let fabrication =
        "I've initiated a deep analysis using a specialized review agent. I'll return shortly.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(fabrication),
        MockProvider::text_response(
            "I could not start a specialist agent, so no delegated review is running.",
        ),
    ]);
    let harness = setup_test_agent(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "fabricated_delegation",
            "Analyze that resume. Any flaws?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        !response.contains("I've initiated"),
        "fabricated zero-tool delegation claim was accepted: {response}"
    );
    let calls = harness.provider.call_log.lock().await;
    assert!(calls.len() >= 2);
    assert!(calls.iter().skip(1).any(|call| {
        call.messages.iter().any(|message| {
            message["content"]
                .as_str()
                .is_some_and(|text| text.contains("MUST include at least one tool call"))
        })
    }));
}

/// The efficacy-analysis payoff: a supervision gate fire is persisted as a
/// `GateTelemetry` decision point whose `task_id` matches the turn's `TaskEnd`
/// event (which already carries a `TaskOutcome`). That shared key is what lets
/// "which gates help vs hurt" be a query — join gate fires → task outcome —
/// instead of manual log archaeology.
///
/// Rather than depend on which supervision gate the loop happens to select
/// (gate selection is loop-internal and evolves), this drives a real turn to
/// produce a genuine `TaskEnd` + `task_id` + `TaskOutcome`, then records a
/// gate fire against that same real `task_id` and asserts they join.
#[tokio::test]
async fn test_gate_fire_event_joins_to_task_end_by_task_id() {
    use crate::events::{DecisionPointData, DecisionType, EventEmitter, EventType, TaskEndData};

    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "It is running on the host operating system.",
    )]);
    let harness = setup_test_agent(provider).await.unwrap();
    let session_id = "test_session";

    harness
        .agent
        .handle_message(
            session_id,
            "What operating system is this running on?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let event_store = crate::events::EventStore::new(harness.state.pool())
        .await
        .expect("event store from harness pool");
    let events = event_store
        .query_recent_events(session_id, 200)
        .await
        .expect("recent events");

    // The real turn emitted a TaskEnd carrying a task_id and a TaskOutcome —
    // the join target an analyst correlates gate fires against.
    let task_end = events
        .iter()
        .filter(|e| e.event_type == EventType::TaskEnd)
        .find_map(|e| e.parse_data::<TaskEndData>().ok())
        .expect("turn emitted a TaskEnd event");
    let real_task_id = task_end.task_id.clone();
    assert!(
        task_end.outcome.is_some(),
        "TaskEnd must carry a TaskOutcome for the gate-fire join to be meaningful"
    );

    // Record a supervision gate fire against that same real task_id.
    let emitter = EventEmitter::new(harness.agent.event_store().clone(), session_id.to_string());
    harness
        .agent
        .supervision_gate_enforced(
            "mutation_contract_block",
            "gemma-3-27b-it",
            &emitter,
            &real_task_id,
            4,
        )
        .await;

    // The gate fire is queryable and joins to the TaskEnd by task_id.
    let events = event_store
        .query_recent_events(session_id, 200)
        .await
        .expect("recent events after gate fire");
    let joined = events
        .iter()
        .filter_map(|e| e.parse_data::<DecisionPointData>().ok())
        .any(|d| {
            d.decision_type == DecisionType::GateTelemetry
                && d.metadata.get("code").and_then(serde_json::Value::as_str)
                    == Some("supervision_gate_fire")
                && d.task_id == real_task_id
        });
    assert!(
        joined,
        "gate fire did not persist with the real task_id {real_task_id} for the TaskEnd join"
    );
}
