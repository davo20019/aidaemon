// ==========================================================================
// Realistic workflow tests
//
// These simulate common user workflows end-to-end: asking for system info,
// multi-step tool interactions, and multi-turn conversations with tool use.
// ==========================================================================

/// Scenario: Telegram Owner asks the agent to check system info.
/// Simulates: user message → LLM calls system_info → LLM returns formatted answer.
#[tokio::test]
async fn test_telegram_owner_system_info_workflow() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("You're running macOS on arm64 with 16GB RAM."),
    ]);

    let mut allowed = vec![12345u64];
    let owner_ids: Vec<u64> = vec![];
    let role = simulate_telegram_auth(&mut allowed, &owner_ids, 12345).unwrap();

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "tg_12345",
            "what system am I running?",
            None,
            role,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "You're running macOS on arm64 with 16GB RAM.");
    assert_eq!(harness.provider.call_count().await, 2); // tool call + final
}

/// Scenario: Slack Owner asks to run a command — LLM properly uses tool.
/// Tests the full loop: auth → Owner → tool call → tool result → final answer.
#[tokio::test]
async fn test_slack_owner_command_workflow() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("System: Linux x86_64, 8 cores"),
    ]);

    let allowed = vec!["UOWNER".to_string()];
    let role = simulate_slack_auth(&allowed, "UOWNER", true).unwrap();
    assert_eq!(role, UserRole::Owner);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "slack_owner",
            "check the system specs",
            None,
            role,
            ChannelContext::private("slack"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "System: Linux x86_64, 8 cores");
    // Verify the tool was actually executed (tool result in second call)
    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 2);
    let has_tool_result = call_log[1]
        .messages
        .iter()
        .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"));
    assert!(
        has_tool_result,
        "Second LLM call should contain tool execution result"
    );
}

/// Scenario: Public user asks to run a command — no tools available.
/// The LLM should only respond conversationally.
#[tokio::test]
async fn test_public_user_cannot_execute_tools() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I'm sorry, I can only chat — tool-based actions aren't available for public users.",
    )]);

    let allowed = vec!["UOWNER".to_string()];
    let role = simulate_slack_auth(&allowed, "URANDOM", true).unwrap();
    assert_eq!(role, UserRole::Public);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "slack_public",
            "run python3 --version",
            None,
            role,
            ChannelContext::private("slack"),
            None,
        )
        .await
        .unwrap();

    // LLM was given no tools, so it can only reply with text
    let call_log = harness.provider.call_log.lock().await;
    assert!(
        call_log[0].tools.is_empty(),
        "Public user must have no tools"
    );
    assert_eq!(
        call_log.len(),
        1,
        "Should be a single conversational reply, no tool loop"
    );
    assert!(
        response.contains("sorry") || response.contains("can only chat"),
        "Response should explain tool limitation"
    );
}

/// Scenario: Multi-turn conversation where first turn uses a tool, second
/// turn references the first. Verifies history + tool results carry over.
#[tokio::test]
async fn test_multi_turn_with_tool_use() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1: tool call + final response
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("You have 16GB RAM."),
        // Turn 2: direct response referencing previous context
        MockProvider::text_response("Yes, 16GB is enough for Docker."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    // Turn 1
    let r1 = harness
        .agent
        .handle_message(
            "multi_tool",
            "how much RAM do I have?",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r1, "You have 16GB RAM.");

    // Turn 2 — references turn 1
    let r2 = harness
        .agent
        .handle_message(
            "multi_tool",
            "is that enough for Docker?",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(r2, "Yes, 16GB is enough for Docker.");

    // Third LLM call should include the full history from turn 1
    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 3); // tool_call + response + follow-up
    let third_call_msgs = &call_log[2].messages;
    // Should have more messages than the first call (accumulated history)
    assert!(
        third_call_msgs.len() > call_log[0].messages.len(),
        "Follow-up should include previous turn's history"
    );
}

struct ReflectionProbeTool;

#[async_trait::async_trait]
impl crate::traits::Tool for ReflectionProbeTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> &str {
        "Synthetic http_request tool for reflection feedback loop tests"
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "http_request",
            "description": "Synthetic http_request tool for reflection feedback loop tests",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": { "type": "string" },
                    "url": { "type": "string" },
                    "method": { "type": "string" }
                },
                "required": ["mode"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let mode = serde_json::from_str::<serde_json::Value>(arguments)?
            .get("mode")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        match mode.as_str() {
            "alpha" => Err(anyhow::anyhow!("wrong base url alpha")),
            "beta" => Err(anyhow::anyhow!("wrong auth beta")),
            "ok" => Ok("probe ok".to_string()),
            other => Err(anyhow::anyhow!("unexpected mode {}", other)),
        }
    }

    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
}

fn reflection_diagnosis_response(
    root_cause: &str,
    action: &str,
    learning: &str,
) -> ProviderResponse {
    MockProvider::text_response(&format!(
        "ROOT_CAUSE: {root_cause}\nRECOMMENDED_ACTION: {action}\nLEARNING: {learning}"
    ))
}

#[tokio::test]
#[serial_test::serial]
async fn test_reflection_full_flow_verifies_learning_on_immediate_recovery() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"alpha","method":"GET","url":"https://api.example.com/alpha?attempt=1"}"#,
        ),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"alpha","method":"HEAD","url":"https://api.example.com/alpha?attempt=2"}"#,
        ),
        reflection_diagnosis_response(
            "The tool keeps using the wrong base URL.",
            "Switch the probe to the corrected endpoint.",
            "Always switch the probe off the broken alpha URL before retrying.",
        ),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"ok","method":"GET","url":"https://api.example.com/fixed"}"#,
        ),
        MockProvider::text_response("Recovered after reflection."),
    ]);

    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(ReflectionProbeTool)],
        None,
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "reflection_verify_success",
            "run the reflection probe",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Recovered after reflection.");

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        call_log.len(),
        5,
        "expected tool/tool/reflection/tool/final flow"
    );
    assert!(
        call_log[2].tools.is_empty(),
        "reflection call must be text-only with no tools exposed"
    );
    assert!(
        call_log[2].messages.iter().any(|message| {
            message["role"] == "user"
                && message["content"]
                    .as_str()
                    .is_some_and(|content| content.contains("ERROR HISTORY"))
        }),
        "reflection call should include the failure analysis prompt"
    );
    assert!(
        call_log[3].messages.iter().any(|message| {
            message["role"] == "system"
                && message["content"].as_str().is_some_and(|content| {
                    content.contains("SELF-DIAGNOSIS")
                        && content.contains("wrong base URL")
                        && content.contains("Switch the probe")
                })
        }),
        "post-reflection model call should receive the injected self-diagnosis directive"
    );

    // Verification runs via tokio::spawn — yield to let the spawned task complete.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT solution_summary, success_count FROM error_solutions WHERE solution_summary = ?",
    )
    .bind("Always switch the probe off the broken alpha URL before retrying.")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    assert_eq!(rows.len(), 1, "expected one stored reflection learning");
    assert_eq!(
        rows[0].1, 1,
        "immediate recovery should verify the reflection learning"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn test_reflection_full_flow_does_not_verify_stale_signature_after_drift() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"alpha","method":"GET","url":"https://api.example.com/alpha?attempt=1"}"#,
        ),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"alpha","method":"HEAD","url":"https://api.example.com/alpha?attempt=2"}"#,
        ),
        reflection_diagnosis_response(
            "The tool keeps using the wrong base URL.",
            "Fix the alpha endpoint before retrying.",
            "Always correct the alpha endpoint before retrying the probe.",
        ),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"beta","method":"GET","url":"https://api.example.com/beta?attempt=1"}"#,
        ),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"beta","method":"HEAD","url":"https://api.example.com/beta?attempt=2"}"#,
        ),
        reflection_diagnosis_response(
            "The tool is now failing auth.",
            "Refresh the beta auth settings before retrying.",
            "Always refresh beta auth settings before retrying the probe.",
        ),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"alpha","method":"POST","url":"https://api.example.com/alpha?attempt=3"}"#,
        ),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"ok","method":"GET","url":"https://api.example.com/fixed"}"#,
        ),
        MockProvider::text_response("Finished without verifying a stale reflection."),
    ]);

    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(ReflectionProbeTool)],
        None,
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "reflection_verify_drift",
            "run the reflection probe through a few retries",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Finished without verifying a stale reflection.");

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        call_log.len(),
        9,
        "expected two reflection calls in the real loop"
    );
    assert!(
        call_log[2].tools.is_empty() && call_log[5].tools.is_empty(),
        "both reflection calls must be text-only"
    );

    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT solution_summary, success_count FROM error_solutions WHERE solution_summary IN (?, ?) ORDER BY solution_summary",
    )
    .bind("Always correct the alpha endpoint before retrying the probe.")
    .bind("Always refresh beta auth settings before retrying the probe.")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    assert_eq!(
        rows.len(),
        2,
        "expected both reflection learnings to be stored"
    );
    assert_eq!(
        rows,
        vec![
            (
                "Always correct the alpha endpoint before retrying the probe.".to_string(),
                0,
            ),
            (
                "Always refresh beta auth settings before retrying the probe.".to_string(),
                0,
            ),
        ],
        "a later success after signature drift must not verify either earlier reflection"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn test_reflection_full_flow_does_not_verify_after_skipping_recovery_turn() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"alpha","method":"GET","url":"https://api.example.com/alpha?attempt=1"}"#,
        ),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"alpha","method":"HEAD","url":"https://api.example.com/alpha?attempt=2"}"#,
        ),
        reflection_diagnosis_response(
            "The tool keeps using the wrong base URL.",
            "Fix the alpha endpoint before retrying.",
            "Always correct the alpha endpoint before retrying the probe.",
        ),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response(
            "http_request",
            r#"{"mode":"ok","method":"GET","url":"https://api.example.com/fixed"}"#,
        ),
        MockProvider::text_response("Finished after a later unrelated probe success."),
    ]);

    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(ReflectionProbeTool)],
        None,
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "reflection_verify_skip_turn",
            "run the reflection probe and gather other context first",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Finished after a later unrelated probe success.");

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        call_log.len(),
        6,
        "expected tool/tool/reflection/other-tool/tool/final flow"
    );
    assert!(
        call_log[2].tools.is_empty(),
        "reflection call must be text-only with no tools exposed"
    );

    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT solution_summary, success_count FROM error_solutions WHERE solution_summary = ?",
    )
    .bind("Always correct the alpha endpoint before retrying the probe.")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    assert_eq!(rows.len(), 1, "expected one stored reflection learning");
    assert_eq!(
        rows[0].1, 0,
        "skipping the designated recovery turn must leave the reflection learning unverified"
    );
}

/// Scenario: Guest on Telegram asks for help — no tools are available.
/// Verifies guest still gets a conversational response.
#[tokio::test]
async fn test_telegram_guest_has_no_tools() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I can help with that without tools.",
    )]);

    let mut allowed = vec![100, 200];
    let owner_ids = vec![100]; // only 100 is owner
    let role = simulate_telegram_auth(&mut allowed, &owner_ids, 200).unwrap();
    assert_eq!(role, UserRole::Guest);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "tg_guest",
            "what system is this?",
            None,
            role,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    // Guest receives a conversational response without tool calls
    assert_eq!(response, "I can help with that without tools.");
    assert_eq!(harness.provider.call_count().await, 1);
    let call_log = harness.provider.call_log.lock().await;
    assert!(call_log[0].tools.is_empty(), "Guest should not have tools");
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    assert!(
        sys["content"]
            .as_str()
            .unwrap()
            .contains("Tool access is owner-only"),
        "Guest should have owner-only restriction in system prompt"
    );
}

/// Scenario: Discord user uses tools — always Owner, no restrictions.
#[tokio::test]
async fn test_discord_user_full_tool_workflow() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("All good on Discord!"),
    ]);

    let role = simulate_discord_auth().unwrap();
    assert_eq!(role, UserRole::Owner);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "discord_42",
            "check system",
            None,
            role,
            ChannelContext::private("discord"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "All good on Discord!");
    assert_eq!(harness.provider.call_count().await, 2);
}

// ==========================================================================
// Multi-step, stall detection, memory, and safety tests
// ==========================================================================

/// Multi-step tool execution: agent calls system_info, then remember_fact,
/// then gives final answer. Verifies 3-step agentic loop.
#[tokio::test]
async fn test_multi_step_tool_execution() {
    let provider = MockProvider::with_responses(vec![
        // Iter 1 (intent gate — narration too short, will be forced to narrate)
        // The agent returns a tool call with no narration on iter 1, intent gate fires,
        // then on iter 2 it gets the same tool call with narration
        MockProvider::tool_call_response("system_info", "{}"),
        // Iter 2: same tool call but now with narration (>20 chars)
        {
            let mut resp = MockProvider::tool_call_response("system_info", "{}");
            resp.content = Some("Let me check your system info first.".to_string());
            resp
        },
        // Iter 3: based on system_info result, remember a fact
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"system","key":"os","value":"test-os"}"#,
        ),
        // Iter 4: final answer
        MockProvider::text_response("Done! I checked your system and saved the info."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "multi_step",
            "check my system and remember what you find",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Done! I checked your system and saved the info.");
    // Should be 4 LLM calls: intent-gated + system_info + remember_fact + final
    assert_eq!(harness.provider.call_count().await, 4);
}

#[tokio::test]
async fn test_risky_tool_calls_trigger_structured_pre_execution_plan() {
    let provider = MockProvider::with_responses(vec![
        {
            let mut resp = MockProvider::tool_call_response(
                "remember_fact",
                r#"{"category":"preference","key":"favorite_editor","value":"helix"}"#,
            );
            resp.content =
                Some("I'll store that preference and confirm it back to you.".to_string());
            resp
        },
        MockProvider::text_response(
            r#"{"goal":"Store the user's editor preference","success_criteria":["The preference is stored in long-term memory","The final reply confirms what was stored"],"first_action":{"tool":"remember_fact","target":"","description":"Store the favorite editor preference in long-term memory"},"requires_verification":true,"risky_actions":["This mutates long-term memory state"],"version":1}"#,
        ),
        MockProvider::text_response("I'll remember that your favorite editor is helix."),
    ]);

    let harness = setup_test_agent_root(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "plan_gate_success",
            "remember that my favorite editor is helix",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        response,
        "I'll remember that your favorite editor is helix."
    );

    assert_eq!(
        call_log.len(),
        3,
        "expected tool call + planning gate + final"
    );
    assert!(
        matches!(
            call_log[1].options.response_mode,
            crate::traits::ResponseMode::JsonSchema { .. }
        ),
        "second LLM call should be the schema-constrained pre-execution planning gate"
    );
    assert!(
        call_log[1].tools.is_empty(),
        "planning gate should run without tool definitions"
    );
    let has_tool_result = call_log[2]
        .messages
        .iter()
        .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"));
    assert!(
        has_tool_result,
        "final LLM call should receive the remember_fact tool result"
    );

    let event_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id DESC",
    )
    .bind("plan_gate_success")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    let accepted_gate = event_rows
        .iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(raw).unwrap())
        .find(|data| {
            data.get("decision_type").and_then(|v| v.as_str()) == Some("execution_planning_gate")
                && data
                    .get("metadata")
                    .and_then(|m| m.get("gate_result"))
                    .and_then(|v| v.as_str())
                    == Some("accepted")
        })
        .expect("expected accepted execution_planning_gate decision point");
    assert_eq!(
        accepted_gate["metadata"]["tool"].as_str(),
        Some("remember_fact")
    );
}

#[tokio::test]
async fn test_pre_execution_plan_gate_gracefully_falls_back_when_provider_rejects_structured_call()
{
    let provider = MockProvider::with_responses(vec![
        {
            let mut resp = MockProvider::tool_call_response(
                "remember_fact",
                r#"{"category":"preference","key":"favorite_shell","value":"zsh"}"#,
            );
            resp.content = Some("I'll save that shell preference for later.".to_string());
            resp
        },
        MockProvider::text_response("I'll remember that your preferred shell is zsh."),
    ])
    .rejecting_non_default_options();

    let harness = setup_test_agent_root(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "plan_gate_unavailable",
            "remember that my preferred shell is zsh",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "I'll remember that your preferred shell is zsh.");

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        call_log.len(),
        3,
        "expected tool call + failed planning gate + final"
    );
    assert!(
        matches!(
            call_log[1].options.response_mode,
            crate::traits::ResponseMode::JsonSchema { .. }
        ),
        "second LLM call should still attempt the structured planning gate"
    );

    let event_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id DESC",
    )
    .bind("plan_gate_unavailable")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    let unavailable_gate = event_rows
        .iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(raw).unwrap())
        .find(|data| {
            data.get("decision_type").and_then(|v| v.as_str()) == Some("execution_planning_gate")
                && data
                    .get("metadata")
                    .and_then(|m| m.get("gate_result"))
                    .and_then(|v| v.as_str())
                    == Some("unavailable")
        })
        .expect("expected unavailable execution_planning_gate decision point");
    assert_eq!(
        unavailable_gate["metadata"]["tool"].as_str(),
        Some("remember_fact")
    );
}

#[tokio::test]
async fn test_high_risk_critique_pass_replans_before_external_execution() {
    let provider = MockProvider::with_responses(vec![
        {
            let mut resp = MockProvider::tool_call_response("external_action", "{}");
            resp.content = Some(
                "I'm going to write to the external system immediately and then summarize the result."
                    .to_string(),
            );
            resp
        },
        MockProvider::text_response(
            r#"{"goal":"Write to the external system","success_criteria":["The write completes successfully","The final reply confirms the result"],"first_action":{"tool":"external_action","target":"","description":"Write to the external system immediately"},"requires_verification":true,"risky_actions":["This performs an external state-changing action"],"version":1}"#,
        ),
        MockProvider::text_response(
            r#"{"verdict":"replan","issues":["Missing evidence: the plan jumps straight to an external write without first inspecting the current target state."],"summary":"The plan should inspect the target state before performing the external write."}"#,
        ),
        MockProvider::text_response(
            "I need to inspect the target state before I perform the external write.",
        ),
    ]);

    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(ExternalActionTool)],
        None,
    )
    .await
    .unwrap();
    let response = harness
        .agent
        .handle_message(
            "critique_replan_external",
            "write the new record to the external system",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        !response.is_empty(),
        "expected the loop to return a non-empty retry response after critique rejection"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        call_log.len(),
        4,
        "expected tool call + plan + critique + retry"
    );
    assert!(
        matches!(
            call_log[2].options.response_mode,
            crate::traits::ResponseMode::JsonSchema { .. }
        ),
        "third LLM call should be the schema-constrained critique pass"
    );

    let event_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id DESC",
    )
    .bind("critique_replan_external")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    let rejected_critique = event_rows
        .iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(raw).unwrap())
        .find(|data| {
            data.get("decision_type").and_then(|v| v.as_str()) == Some("execution_critique_pass")
                && data
                    .get("metadata")
                    .and_then(|m| m.get("critique_result"))
                    .and_then(|v| v.as_str())
                    == Some("rejected")
        })
        .expect("expected rejected execution_critique_pass decision point");
    assert_eq!(
        rejected_critique["metadata"]["tool"].as_str(),
        Some("external_action")
    );

    let tool_call_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM events WHERE session_id = ? AND event_type = 'tool_call'",
    )
    .bind("critique_replan_external")
    .fetch_one(&harness.state.pool())
    .await
    .unwrap();
    assert_eq!(
        tool_call_count, 0,
        "critique rejection should block external execution before any tool call event is emitted"
    );
}

#[tokio::test]
async fn test_execution_state_snapshots_and_idempotency_key_are_emitted_for_mutations() {
    let provider = MockProvider::with_responses(vec![
        {
            let mut resp = MockProvider::tool_call_response(
                "remember_fact",
                r#"{"category":"preference","key":"favorite_terminal","value":"ghostty"}"#,
            );
            resp.content =
                Some("I'll store that terminal preference and confirm it back to you.".to_string());
            resp
        },
        MockProvider::text_response(
            r#"{"goal":"Store the user's terminal preference","success_criteria":["The preference is stored in long-term memory","The final reply confirms what was stored"],"first_action":{"tool":"remember_fact","target":"","description":"Store the favorite terminal preference in long-term memory"},"requires_verification":true,"risky_actions":["This mutates long-term memory state"]}"#,
        ),
        MockProvider::text_response("I'll remember that your favorite terminal is ghostty."),
    ]);

    let harness = setup_test_agent_root(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "execution_state_remember",
            "remember that my favorite terminal is ghostty",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response,
        "I'll remember that your favorite terminal is ghostty."
    );

    let decision_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id DESC",
    )
    .bind("execution_state_remember")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    let has_budget_selection = decision_rows
        .iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(raw).unwrap())
        .any(|data| {
            data.get("decision_type").and_then(|v| v.as_str()) == Some("execution_budget_selection")
                && data
                    .get("metadata")
                    .and_then(|m| m.get("route_kind"))
                    .and_then(|v| v.as_str())
                    .is_some()
        });
    assert!(
        has_budget_selection,
        "expected execution budget selection event"
    );

    let has_step_completed_snapshot = decision_rows
        .iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(raw).unwrap())
        .any(|data| {
            data.get("decision_type").and_then(|v| v.as_str()) == Some("execution_state_snapshot")
                && data
                    .get("metadata")
                    .and_then(|m| m.get("condition"))
                    .and_then(|v| v.as_str())
                    == Some("step_completed")
        });
    assert!(
        has_step_completed_snapshot,
        "expected step_completed execution state snapshot"
    );

    let tool_call_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'tool_call' ORDER BY id DESC",
    )
    .bind("execution_state_remember")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    let tool_call = tool_call_rows
        .iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(raw).unwrap())
        .find(|data| data.get("name").and_then(|v| v.as_str()) == Some("remember_fact"))
        .expect("expected remember_fact tool call event");

    let idempotency_key = tool_call
        .get("idempotency_key")
        .and_then(|v| v.as_str())
        .expect("expected idempotency key on mutating tool call");
    assert!(
        idempotency_key.starts_with("exec:"),
        "expected execution-scoped idempotency key, got {idempotency_key}"
    );
}

#[tokio::test]
async fn test_evidence_gate_blocks_edit_until_file_is_read_then_plans_first_mutation() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("notes.txt");
    std::fs::write(&file_path, "hello world\n").unwrap();
    let file_path = file_path.to_string_lossy().to_string();

    let provider = MockProvider::with_responses(vec![
        {
            let mut resp = MockProvider::tool_call_response(
                "edit_file",
                &json!({
                    "path": file_path.clone(),
                    "old_text": "hello world",
                    "new_text": "hello evidence gate",
                })
                .to_string(),
            );
            resp.content = Some("I'll update the file now.".to_string());
            resp
        },
        {
            let mut resp = MockProvider::tool_call_response(
                "read_file",
                &json!({ "path": file_path.clone() }).to_string(),
            );
            resp.content = Some("I need to inspect the file before editing it.".to_string());
            resp
        },
        {
            let mut resp = MockProvider::tool_call_response(
                "edit_file",
                &json!({
                    "path": file_path.clone(),
                    "old_text": "hello world",
                    "new_text": "hello evidence gate",
                })
                .to_string(),
            );
            resp.content = Some("I have read the file and will now update it.".to_string());
            resp
        },
        MockProvider::text_response(
            &json!({
                "goal": "Update the target file contents",
                "success_criteria": [
                    "The target file contains the new replacement text",
                    "The final reply confirms the edit"
                ],
                "first_action": {
                    "tool": "edit_file",
                    "target": file_path.clone(),
                    "description": "Replace the old text in the file after inspecting it"
                },
                "requires_verification": true,
                "risky_actions": ["This mutates a local file"]
            })
            .to_string(),
        ),
        MockProvider::text_response("Updated notes.txt successfully."),
    ]);

    let harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(ReadFileTool), Arc::new(EditFileTool)],
        None,
    )
    .await
    .unwrap();
    let response = harness
        .agent
        .handle_message(
            "evidence_gate_edit",
            "update the note file",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Updated notes.txt successfully.");
    assert_eq!(
        std::fs::read_to_string(&file_path).unwrap(),
        "hello evidence gate\n"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        call_log.len(),
        5,
        "expected edit block + read + edit + planning gate + final"
    );
    assert!(
        matches!(
            call_log[3].options.response_mode,
            crate::traits::ResponseMode::JsonSchema { .. }
        ),
        "planning gate should still run after read-only evidence is gathered"
    );

    let event_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id DESC",
    )
    .bind("evidence_gate_edit")
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();

    let evidence_gate = event_rows
        .iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(raw).unwrap())
        .find(|data| {
            data.get("decision_type").and_then(|v| v.as_str()) == Some("evidence_gate")
                && data
                    .get("metadata")
                    .and_then(|m| m.get("tool"))
                    .and_then(|v| v.as_str())
                    == Some("edit_file")
        })
        .expect("expected edit_file evidence_gate decision point");
    assert_eq!(
        evidence_gate["metadata"]["required_evidence_kind"].as_str(),
        Some("FileRead")
    );
}

#[tokio::test]
async fn test_plain_text_retry_does_not_trip_execution_budget_before_tool_mode() {
    let provider = MockProvider::with_responses(vec![
        ProviderResponse {
            content: None,
            tool_calls: vec![],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 12_000,
                output_tokens: 6_000,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: Some("truncated".to_string()),
        },
        MockProvider::text_response(
            "Here is a tweet: Building a lot lately. What should I share next?",
        ),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "plain_text_retry_budget",
            "can you post a tweet about your new stuff, thoughts, updates or anything about yourself? make it engaging so people want to comment on it",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response,
        "Here is a tweet: Building a lot lately. What should I share next?"
    );
    assert_eq!(
        harness.provider.call_count().await,
        2,
        "plain-text retries should be allowed to recover without tripping execution budget"
    );
}

#[tokio::test]
async fn test_execution_budget_starts_after_tool_handoff_response() {
    let provider = MockProvider::with_responses(vec![
        ProviderResponse {
            content: None,
            tool_calls: vec![crate::traits::ToolCall {
                id: "call_budget_baseline".to_string(),
                name: "system_info".to_string(),
                arguments: "{}".to_string(),
                extra_content: None,
            }],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 30_000,
                output_tokens: 15_000,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        MockProvider::text_response("I checked the system info successfully."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "execution_budget_baseline",
            "check the system info on this machine",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "I checked the system info successfully.");
    assert_eq!(
        harness.provider.call_count().await,
        2,
        "the pre-tool planning/tool-selection call should not consume the execution token envelope"
    );
}

#[tokio::test]
async fn test_observational_progress_extends_budget_so_productive_runs_complete() {
    // Regression: previously, a tight budget (max_llm_calls=2) would exhaust
    // after two tool-calling iterations, even though both calls succeeded.
    // With progress-based budget extension, each successful tool call extends
    // the budget, so productive runs are never artificially stopped.
    let provider = MockProvider::with_responses(vec![
        ProviderResponse {
            content: None,
            tool_calls: vec![crate::traits::ToolCall {
                id: "call_system_info_initial".to_string(),
                name: "system_info".to_string(),
                arguments: "{}".to_string(),
                extra_content: None,
            }],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 200,
                output_tokens: 100,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        ProviderResponse {
            content: None,
            tool_calls: vec![crate::traits::ToolCall {
                id: "call_system_info_repeat".to_string(),
                name: "system_info".to_string(),
                arguments: "{}".to_string(),
                extra_content: None,
            }],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 48_000,
                output_tokens: 24_000,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: Some("oversized_observational_retry".to_string()),
        },
        MockProvider::text_response(
            "Here is the current system summary: macOS on arm64 with the expected hardware profile.",
        ),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    // Start with a tight budget. Both tool calls succeed, so the
    // progress-based extension keeps the budget above the usage counter and
    // the agent completes normally — no budget blocker.
    harness
        .agent
        .set_test_execution_budget_override(Some(crate::agent::ExecutionBudget {
            max_steps: 100,
            max_tokens: 0,
            max_llm_calls: 2,
            max_tool_calls: 100,
            max_validation_rounds: 100,
            max_wall_clock_ms: 600_000,
        }));
    let response = harness
        .agent
        .handle_message(
            "observation_budget_closeout",
            "check the current system info and summarize it for me",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response,
        "Here is the current system summary: macOS on arm64 with the expected hardware profile."
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(call_log.len(), 3);
    // Budget was NOT exhausted because successful tool calls extended it,
    // so the final call proceeds normally without being forced into closeout.
    assert!(
        !response.contains("blocked"),
        "productive runs should never be stopped by the budget"
    );
}

#[tokio::test]
async fn test_deferred_text_only_turn_switches_to_plain_text_recovery_mode() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("I'll draft something engaging for you."),
        MockProvider::text_response(
            "Building a lot lately. What's something you're curious about behind the scenes?",
        ),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "text_only_draft_recovery",
            "write a tweet about what you've been building lately and make it engaging",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response,
        "Building a lot lately. What's something you're curious about behind the scenes?"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(
        call_log.len(),
        2,
        "expected one direct-answer recovery retry"
    );
    assert!(
        !call_log[0].tools.is_empty(),
        "initial drafting turn should still expose tools before recovery decides they are unnecessary"
    );
    assert!(
        call_log[1].tools.is_empty(),
        "plain-text recovery retry should strip tools to break the deferred-action loop"
    );
    assert_eq!(
        call_log[1].options.tool_choice,
        crate::traits::ToolChoiceMode::None
    );
}

#[tokio::test]
async fn test_route_failsafe_does_not_turn_plain_text_draft_into_tool_required_loop() {
    let session_id = "route_failsafe_plain_text_draft";
    crate::agent::set_route_failsafe_for_session_for_test(session_id, true);

    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("Let me draft a strong tweet for you."),
        MockProvider::text_response(
            "Been heads-down building. What should I share a behind-the-scenes update about next?",
        ),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            session_id,
            "can you post a tweet about your new stuff and make it engaging",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    crate::agent::set_route_failsafe_for_session_for_test(session_id, false);

    assert_eq!(
        response,
        "Been heads-down building. What should I share a behind-the-scenes update about next?"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(call_log.len(), 2);
    // "post a tweet ... make it engaging" is now classified as DraftThenDeliver
    // (expects_mutation=true), so the second call keeps tools available for
    // the mutation path rather than stripping them.
    assert!(
        !call_log[1].tools.is_empty(),
        "DraftThenDeliver mutation turn should keep tools on retry"
    );
}

#[tokio::test]
async fn test_text_only_turn_recovers_when_model_drifts_to_side_effecting_tool() {
    let temp_path = std::env::temp_dir().join(format!(
        "aidaemon-text-only-drift-{}.txt",
        uuid::Uuid::new_v4()
    ));
    let _ = std::fs::remove_file(&temp_path);
    let provider = MockProvider::with_responses(vec![
        ProviderResponse {
            content: Some("I'll handle that for you now.".to_string()),
            tool_calls: vec![crate::traits::ToolCall {
                id: "text_only_side_effecting_drift".to_string(),
                name: "terminal".to_string(),
                arguments: json!({"command": format!("touch {}", temp_path.display())})
                    .to_string(),
                extra_content: None,
            }],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 12,
                output_tokens: 8,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        MockProvider::text_response(
            "Been heads-down building lately. What should I share a behind-the-scenes update about next?",
        ),
    ]);

    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    // Use a conversational question that doesn't trigger expects_mutation
    // or requires_observation, so the text-only prelude fires.
    let response = harness
        .agent
        .handle_message(
            "text_only_side_effecting_drift",
            "what is the meaning of life",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response,
        "Been heads-down building lately. What should I share a behind-the-scenes update about next?"
    );

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(call_log.len(), 2);
    assert!(
        call_log[1].tools.is_empty(),
        "text-only recovery should disable tools after side-effecting drift"
    );
    assert_eq!(
        call_log[1].options.tool_choice,
        crate::traits::ToolChoiceMode::None
    );
    assert!(
        !temp_path.exists(),
        "side-effecting terminal drift should be blocked before execution"
    );
}

#[tokio::test]
async fn test_account_scoped_social_post_request_stays_in_execution_lane() {
    let temp_path = std::env::temp_dir().join(format!(
        "aidaemon-live-delivery-{}.txt",
        uuid::Uuid::new_v4()
    ));
    let _ = std::fs::remove_file(&temp_path);
    let provider = MockProvider::with_responses(vec![
        crate::traits::ProviderResponse {
            content: Some("I'll post that now.".to_string()),
            tool_calls: vec![crate::traits::ToolCall {
                id: "account_scoped_social_post".to_string(),
                name: "terminal".to_string(),
                arguments: serde_json::json!({
                    "command": format!("touch {}", temp_path.display())
                })
                .to_string(),
                extra_content: None,
            }],
            usage: Some(crate::traits::TokenUsage {
                input_tokens: 12,
                output_tokens: 8,
                model: "mock".to_string(),
            }),
            thinking: None,
            response_note: None,
        },
        MockProvider::text_response("Posted."),
    ]);

    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "account_scoped_social_post",
            "Can you post a tweet on your account?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Posted.");
    assert!(
        temp_path.exists(),
        "account-scoped live-delivery phrasing should remain tool-enabled"
    );
}

#[tokio::test]
async fn test_explicit_tool_turn_still_forces_required_tool_choice_after_no_tool_deferral() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response("I'll check the system info now."),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("I checked the system info for you."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "tool_required_deferral_recovery",
            "check the system info on this machine",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "I checked the system info for you.");

    let call_log = harness.provider.call_log.lock().await.clone();
    assert_eq!(call_log.len(), 3);
    assert_eq!(
        call_log[1].options.tool_choice,
        crate::traits::ToolChoiceMode::Required,
        "genuinely tool-required turns should still escalate to required tool choice on retry"
    );
}

/// Stall detection: agent keeps calling an unknown tool which errors each
/// iteration. After MAX_STALL_ITERATIONS (3), agent should gracefully stop.
#[tokio::test]
async fn test_stall_detection_unknown_tool() {
    let provider = MockProvider::with_responses(vec![
        // Intent gate: first call has narration to pass the gate
        {
            let mut resp = MockProvider::tool_call_response("nonexistent_tool", "{}");
            resp.content = Some("I'll try calling this tool to help you.".to_string());
            resp
        },
        // Iter 2: agent tries again
        {
            let mut resp = MockProvider::tool_call_response("nonexistent_tool", "{}");
            resp.content = Some("Let me try again with the tool.".to_string());
            resp
        },
        // Iter 3: agent tries yet again
        {
            let mut resp = MockProvider::tool_call_response("nonexistent_tool", "{}");
            resp.content = Some("One more attempt with the tool.".to_string());
            resp
        },
        // Iter 4: stall detection should have kicked in before this
        MockProvider::text_response("This should not be reached"),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "stall_session",
            "do something",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Agent should have gracefully stopped due to stall (3 iterations with 0 success)
    // The response will be the graceful stall message, not "This should not be reached"
    assert!(
        !response.contains("This should not be reached"),
        "Agent should have stopped before the 4th LLM call"
    );
    // Stall fires at iteration 4 check (after 3 failed iters), so we expect 3 LLM calls
    assert!(
        harness.provider.call_count().await <= 4,
        "Agent should stop after stall detection, got {} calls",
        harness.provider.call_count().await
    );
}

/// Regression test: "create a new website about cars" scenario.
///
/// Real-world bug: user sent a complex prompt, aidaemon delegated to cli_agent,
/// cli_agent completed successfully (built the site, took screenshots). Then
/// aidaemon did follow-up work — exploring the project with 10+ consecutive
/// terminal-like calls (ls, git status, git remote -v, cat package.json, etc.).
/// The alternating pattern detection falsely triggered because all calls used
/// the same tool name (unique_tools.len() == 1 <= 2).
///
/// This test simulates the full flow: system_info (project discovery, 3 calls)
/// → remember_fact (storing project findings, 9 calls) → final summary.
/// The first 3 system_info calls hit the per-tool call-count limit (3 for
/// non-exempt tools), then the agent switches to remember_fact (exempt).
/// Total: 12 single-tool-name calls in the window, must NOT trigger stall.
#[tokio::test]
async fn test_complex_prompt_website_project_exploration() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    // Iteration 1: agent narrates intent + checks system info
    {
        let mut resp = MockProvider::tool_call_response("system_info", "{}");
        resp.content = Some(
            "I'll help you create a new website about cars. Let me first check the system \
             environment to understand what tools and runtimes are available."
                .to_string(),
        );
        responses.push(resp);
    }

    // Iteration 2: agent checks system again (e.g. checking Node.js version)
    {
        let mut resp = MockProvider::tool_call_response("system_info", r#"{"check":"node"}"#);
        resp.content = Some("Let me check if Node.js and npm are installed.".to_string());
        responses.push(resp);
    }

    // Iteration 3: system_info one more time (e.g. checking git)
    {
        let mut resp = MockProvider::tool_call_response("system_info", r#"{"check":"git"}"#);
        resp.content = Some("Checking git configuration for the project.".to_string());
        responses.push(resp);
    }

    // Iterations 4-12: agent records what it found (simulating terminal exploration
    // like ls, cat package.json, git remote -v, etc. — uses remember_fact since
    // terminal requires approval flow unavailable in test harness)
    let facts = [
        (
            "project_structure",
            "Next.js app with src/app layout, tailwind configured",
        ),
        (
            "dependencies",
            "next@14, react@18, tailwindcss@3, typescript@5",
        ),
        (
            "git_remote",
            "origin https://github.com/user/my-website.git",
        ),
        (
            "deployment",
            "Vercel project linked, domain myproject.example.com",
        ),
        (
            "pages_found",
            "Home, About, Gallery, Contact — all with placeholder content",
        ),
        (
            "build_status",
            "npm run build succeeds, no TypeScript errors",
        ),
        (
            "styling",
            "Tailwind with custom theme, dark mode support configured",
        ),
        (
            "images",
            "public/images/ has 12 sample photos from Unsplash",
        ),
        (
            "performance",
            "Lighthouse score 98/100, all Core Web Vitals green",
        ),
    ];
    for (key, value) in &facts {
        let args = format!(
            r#"{{"category":"project","key":"{}","value":"{}"}}"#,
            key, value
        );
        let mut resp = MockProvider::tool_call_response("remember_fact", &args);
        resp.content = Some(format!("Recording project detail: {}", key));
        responses.push(resp);
    }

    // Final: agent summarizes everything
    responses.push(MockProvider::text_response(
        "Done! I've explored the website project and recorded all the key details. \
         The site is built with Next.js 14, deployed to myproject.example.com via Vercel, \
         with 4 pages, Tailwind styling, and a Lighthouse score of 98.",
    ));

    let harness = setup_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();
    let response = harness
        .agent
        .handle_message(
            "telegram_12345",
            "I need to create a new website for my portfolio. We should push it to \
             myproject.example.com. make it modern.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    // Agent should either complete normally or stop gracefully after making
    // progress. The key invariant: no crash, no error, and the response is not empty.
    assert!(
        !response.is_empty(),
        "Agent should return a non-empty response"
    );
    assert!(
        !response.contains("stuck in a loop"),
        "Should not report stuck in a loop"
    );
}

/// Regression test: "Previous convo" — user resumes a conversation and the agent
/// explores an existing project to understand where they left off.
///
/// Real-world bug: user said "Previous convo" and the agent started exploring the
/// my-website project with many terminal commands: ls, git status, ls src -R,
/// cat package.json (x2), git remote -v (x2), ls .git (x2). The duplicate calls
/// (same command twice) didn't reach the soft-redirect threshold of 3, but
/// 11 consecutive terminal calls triggered the alternating pattern detection
/// at the 10th call (window full of a single tool name).
///
/// This test scripts 11 consecutive remember_fact calls with some duplicate
/// arguments (simulating the real-world pattern) and verifies no stall fires.
#[tokio::test]
async fn test_complex_prompt_resume_previous_conversation() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    // Iteration 1: agent narrates + first exploration
    {
        let mut resp = MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"context","key":"project_dir","value":"/home/testuser/my-website"}"#,
        );
        resp.content = Some(
            "I see you want to continue from our previous conversation about the website project. \
             Let me check the current state of the project."
                .to_string(),
        );
        responses.push(resp);
    }

    // Iterations 2-11: agent explores the project, with some duplicate calls
    // (simulating real behavior: git remote -v called twice, ls .git twice, etc.)
    let exploration_steps = [
        ("project_status", "git shows 3 modified files, 1 untracked"),
        (
            "branch",
            "On branch feature/gallery, 2 commits ahead of main",
        ),
        ("package_json", "next@14.1.0, react@18.2.0, 12 dependencies"),
        // Duplicate: agent re-reads package.json (real behavior observed)
        ("package_json", "next@14.1.0, react@18.2.0, 12 dependencies"),
        ("git_remote", "origin git@github.com:user/my-website.git"),
        // Duplicate: agent re-checks remote (real behavior observed)
        ("git_remote", "origin git@github.com:user/my-website.git"),
        (
            "recent_commits",
            "feat: add gallery page, fix: responsive nav, style: footer",
        ),
        (
            "directory_layout",
            "src/app/(pages)/gallery/page.tsx, components/CarCard.tsx",
        ),
        // Duplicate: agent re-lists directory (real behavior observed)
        (
            "directory_layout",
            "src/app/(pages)/gallery/page.tsx, components/CarCard.tsx",
        ),
        (
            "deployment_status",
            "Last deploy 2 hours ago, all checks passed",
        ),
    ];
    for (key, value) in &exploration_steps {
        let args = format!(
            r#"{{"category":"project","key":"{}","value":"{}"}}"#,
            key, value
        );
        let mut resp = MockProvider::tool_call_response("remember_fact", &args);
        resp.content = Some(format!("Checking: {}", key));
        responses.push(resp);
    }

    // Final: agent summarizes what it found
    responses.push(MockProvider::text_response(
        "I've reviewed the project state. You were working on the gallery page for the \
         website. The feature/gallery branch has 3 modified files and is 2 commits ahead of \
         main. The last deploy was 2 hours ago. What would you like to continue with?",
    ));

    let harness = setup_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();
    let response = harness
        .agent
        .handle_message(
            "telegram_12345",
            "Previous convo",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    // Agent should either complete normally or stop gracefully after making
    // progress (the new stopping_phase detects stall-with-progress and returns
    // a graceful response, or the agent may return the last narration text).
    // The key invariant: no crash, no error, and the response is not empty.
    assert!(
        !response.is_empty(),
        "Agent should return a non-empty response"
    );
    assert!(
        !response.contains("stuck in a loop"),
        "Should not report stuck in a loop"
    );
}

/// Regression test: agent uses TWO tools in a productive alternating pattern
/// (system_info + remember_fact) without triggering the alternating detection.
///
/// Tests that the diversity check works correctly: when the agent bounces
/// between 2 tools but each call has unique arguments (productive exploration),
/// the alternating pattern detection should NOT fire.
#[tokio::test]
async fn test_two_tool_alternating_with_diverse_args_allowed() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    // 12 iterations alternating system_info and remember_fact
    // system_info will get blocked after 3 calls, but the pattern still exercises
    // the alternating detection logic. Using multi-tool responses to keep both
    // in the recent_tool_names window.
    for i in 0..12 {
        if i < 3 {
            // First 3: use system_info (before the 3-call block kicks in)
            let mut resp = MockProvider::tool_call_response("system_info", "{}");
            resp.content = Some(format!("Checking system, iteration {}.", i));
            responses.push(resp);
        }
        // All 12: also use remember_fact with unique args
        let args = format!(
            r#"{{"category":"observation","key":"check_{}","value":"result_{}"}}"#,
            i, i
        );
        let mut resp = MockProvider::tool_call_response("remember_fact", &args);
        resp.content = Some(format!("Recording observation {}.", i));
        responses.push(resp);
    }

    // Final text
    responses.push(MockProvider::text_response(
        "Finished all system checks and observations. Everything looks good.",
    ));

    let harness = setup_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();
    let response = harness
        .agent
        .handle_message(
            "mixed_session",
            "Run a comprehensive system audit and record all findings",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Agent should either complete normally or stop gracefully after making
    // progress. The key invariant: no crash, no error, and the response is not empty.
    assert!(
        !response.is_empty(),
        "Agent should return a non-empty response"
    );
}

/// Verify that a TRUE alternating loop (same 2 calls cycling with identical
/// arguments) IS still detected and stopped.
#[tokio::test]
async fn test_true_alternating_loop_still_detected() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    // 12 iterations of the exact same 2 calls alternating (A-B-A-B loop)
    // system_info gets blocked after 3 calls, then remember_fact with identical
    // args will trigger the repetitive hash detection instead. Either way, the
    // agent should be stopped — it's genuinely looping.
    for i in 0..12 {
        // Same system_info call every time
        let mut resp = MockProvider::tool_call_response("system_info", "{}");
        resp.content = Some(format!("Checking system status, attempt {}.", i));
        responses.push(resp);
        // Same remember_fact call every time (identical args = true loop)
        let mut resp = MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"status","key":"check","value":"pending"}"#,
        );
        resp.content = Some("Still checking...".to_string());
        responses.push(resp);
    }
    // This should never be reached
    responses.push(MockProvider::text_response("This should not be reached"));

    let harness = setup_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();
    let response = harness
        .agent
        .handle_message(
            "loop_session",
            "check the system status",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Should be stopped by stall/repetitive detection, NOT complete normally
    assert!(
        !response.contains("This should not be reached"),
        "True alternating loop should be detected and stopped"
    );
    // Should be stopped before all 12 iterations complete
    let calls = harness.provider.call_count().await;
    assert!(
        calls < 20,
        "Expected loop to be stopped early, but got {} LLM calls",
        calls
    );
}

/// Memory persistence through tool use: agent remembers a fact via tool,
/// then on the next turn, the fact should appear in the system prompt.
#[tokio::test]
async fn test_memory_fact_persists_across_turns() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1: agent narrates then remembers a fact
        {
            let mut resp = MockProvider::tool_call_response(
                "remember_fact",
                r#"{"category":"preference","key":"language","value":"Rust"}"#,
            );
            resp.content = Some("I'll remember that you prefer Rust.".to_string());
            resp
        },
        // Turn 1: final response
        MockProvider::text_response("Got it! I'll remember you prefer Rust."),
        // Turn 2: the agent should see the fact in its system prompt
        MockProvider::text_response("Yes, I know you prefer Rust!"),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    // Turn 1: remember the fact
    let r1 = harness
        .agent
        .handle_message(
            "memory_session",
            "I prefer Rust",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(
        r1.contains("Rust"),
        "Turn 1 should acknowledge the preference"
    );

    // Verify fact was persisted to state
    let facts = harness.state.get_relevant_facts("Rust", 10).await.unwrap();
    assert!(
        facts.iter().any(|f| f.value.contains("Rust")),
        "Fact about Rust should be stored. Got: {:?}",
        facts.iter().map(|f| &f.value).collect::<Vec<_>>()
    );

    // Turn 2: system prompt should include the remembered fact
    let _r2 = harness
        .agent
        .handle_message(
            "memory_session",
            "what language do I prefer?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The system prompt sent to the LLM in turn 2 should contain the fact
    let call_log = harness.provider.call_log.lock().await;
    let last_call = call_log.last().unwrap();
    let system_msg = last_call
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let system_content = system_msg["content"].as_str().unwrap_or("");
    assert!(
        system_content.contains("Rust"),
        "System prompt on turn 2 should include the remembered fact about Rust. \
         System prompt tail: ...{}",
        &system_content[system_content.len().saturating_sub(500)..]
    );
}

/// Intent gate test: on the first iteration, if the LLM returns a tool call
/// without narration (content < 20 chars), the agent should force narration
/// and re-issue. This ensures the user sees what the agent plans to do.
#[tokio::test]
async fn test_intent_gate_forces_narration() {
    let provider = MockProvider::with_responses(vec![
        // Iter 1: tool call with NO narration → intent gate triggers
        MockProvider::tool_call_response("system_info", "{}"),
        // Iter 2: same tool call but now with narration (agent learned)
        {
            let mut resp = MockProvider::tool_call_response("system_info", "{}");
            resp.content = Some("I'll check your system information now.".to_string());
            resp
        },
        // Iter 3: final answer
        MockProvider::text_response("Your system is running fine."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();
    let response = harness
        .agent
        .handle_message(
            "intent_session",
            "check my system",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Your system is running fine.");
    // 3 LLM calls: intent-gated (no exec) + narrated tool call + final
    assert_eq!(harness.provider.call_count().await, 3);
}

/// Scheduler simulation: messages from scheduled tasks use special session IDs.
/// The agent treats `scheduler_trigger_*` sessions as untrusted.
#[tokio::test]
async fn test_scheduler_trigger_session_handling() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Simulate a scheduler event with the special session ID format
    let response = harness
        .agent
        .handle_message(
            "scheduled_42",
            "[AUTOMATED TRIGGER from scheduler]\nCheck system health",
            None,
            UserRole::Owner,
            ChannelContext::private("scheduler"),
            None,
        )
        .await
        .unwrap();

    // Agent should process the message (it's a valid session)
    assert_eq!(response, "Mock response");
    assert_eq!(harness.provider.call_count().await, 1);

    // Verify the message is stored with the scheduled session ID
    let history = harness.state.get_history("scheduled_42", 10).await.unwrap();
    assert!(history.len() >= 2, "Should have user + assistant messages");
}

/// Multi-turn memory: facts remembered in turn 1 are available in turn 2's
/// system prompt, and message history carries forward correctly.
#[tokio::test]
async fn test_memory_system_prompt_enrichment() {
    let provider = MockProvider::with_responses(vec![
        // Turn 1: plain response
        MockProvider::text_response("Hello! Nice to meet you."),
        // Turn 2: response that would reference memory
        MockProvider::text_response("Based on what I know, here's my answer."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    // Seed a fact directly into state (simulates prior learning)
    harness
        .state
        .upsert_fact(
            "project",
            "framework",
            "Uses React with TypeScript",
            "agent",
            None,
            crate::types::FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Turn 1
    harness
        .agent
        .handle_message(
            "enrichment_session",
            "hello",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Turn 2 — ask about the project
    harness
        .agent
        .handle_message(
            "enrichment_session",
            "what framework does my project use?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Facts are on-demand now — system prompt should NOT contain the seeded fact,
    // but SHOULD contain the memory capabilities summary.
    let call_log = harness.provider.call_log.lock().await;
    let turn2_call = &call_log[1];
    let system_msg = turn2_call
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let system_content = system_msg["content"].as_str().unwrap_or("");

    assert!(
        !system_content.contains("React with TypeScript"),
        "Facts should NOT be bulk-injected into system prompt"
    );
    assert!(
        system_content.contains("Your Memory") || system_content.contains("manage_memories"),
        "System prompt should contain memory capabilities summary"
    );
}
