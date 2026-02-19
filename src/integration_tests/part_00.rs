// Integration tests that exercise the real agent loop with a mock LLM.
//
// These tests verify: agent loop, tool execution, memory persistence,
// multi-turn history, and session isolation — the same code path all
// channels use via `Agent::handle_message()`.

use crate::testing::{
    setup_full_stack_test_agent, setup_full_stack_test_agent_with_extra_tools, setup_test_agent,
    setup_test_agent_orchestrator, setup_test_agent_orchestrator_task_leads,
    setup_test_agent_with_models, MockProvider, MockTool,
};
use crate::traits::store_prelude::*;
use crate::traits::{
    BehaviorPattern, Episode, ErrorSolution, Goal, Procedure, ProviderResponse, StateStore,
    UserProfile,
};
use crate::types::{ChannelContext, ChannelVisibility, StatusUpdate, UserRole};
use chrono::Utc;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[tokio::test]
async fn test_basic_message_response() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

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
    assert_eq!(harness.provider.call_count().await, 1);
}

#[tokio::test]
async fn test_empty_llm_response_retried_then_fallback() {
    // With consultant pass disabled (default+fallback routing), the first empty
    // response IS the first iteration. Flow: empty -> retry with nudge -> empty -> fallback.
    // That's 2 LLM calls total.
    let provider = MockProvider::with_responses(vec![
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
            "Hello!",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let expected = "I wasn't able to process that request. Could you try rephrasing?";
    assert_eq!(harness.provider.call_count().await, 2);
    assert_eq!(response, expected);

    // Verify the retry iteration included a system nudge for the LLM.
    let calls = harness.provider.call_log.lock().await.clone();
    assert!(calls.len() >= 2, "expected at least 2 LLM calls");
    let retry_messages = &calls[1].messages;
    let saw_retry_nudge = retry_messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("system")
            && m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|c| c.contains("previous reply was empty"))
    });
    assert!(
        saw_retry_nudge,
        "expected retry nudge system message on second call"
    );
}

#[tokio::test]
async fn test_tool_execution() {
    // Script: first call → tool call, second call → final text response
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Here is your system info."),
    ]);

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

    // Agent should have called the LLM twice (tool call + final response)
    assert_eq!(harness.provider.call_count().await, 2);
    assert_eq!(response, "Here is your system info.");
}

#[tokio::test]
async fn test_memory_persistence() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness
        .agent
        .handle_message(
            "persist_session",
            "Remember this",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Verify messages were persisted to state store
    let history = harness
        .state
        .get_history("persist_session", 10)
        .await
        .unwrap();

    // Should have at least a user message and an assistant message
    assert!(
        history.len() >= 2,
        "Expected at least 2 messages, got {}",
        history.len()
    );

    let roles: Vec<&str> = history.iter().map(|m| m.role.as_str()).collect();
    assert!(roles.contains(&"user"), "Missing user message in history");
    assert!(
        roles.contains(&"assistant"),
        "Missing assistant message in history"
    );
}

#[tokio::test]
async fn test_multi_turn_conversation() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // First turn
    harness
        .agent
        .handle_message(
            "multi_session",
            "First message",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Second turn
    harness
        .agent
        .handle_message(
            "multi_session",
            "Second message",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The second LLM call should include history from the first turn
    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 2);

    // Second call should have more messages than the first (includes first turn)
    let first_msg_count = call_log[0].messages.len();
    let second_msg_count = call_log[1].messages.len();
    assert!(
        second_msg_count > first_msg_count,
        "Second call should include first turn's history: {} vs {}",
        second_msg_count,
        first_msg_count,
    );
}

#[tokio::test]
async fn test_session_isolation() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Send to session A
    harness
        .agent
        .handle_message(
            "session_a",
            "Message for A",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Send to session B
    harness
        .agent
        .handle_message(
            "session_b",
            "Message for B",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Check histories are isolated
    let history_a = harness.state.get_history("session_a", 10).await.unwrap();
    let history_b = harness.state.get_history("session_b", 10).await.unwrap();

    // Each session should have its own messages
    let a_contents: Vec<String> = history_a.iter().filter_map(|m| m.content.clone()).collect();
    let b_contents: Vec<String> = history_b.iter().filter_map(|m| m.content.clone()).collect();

    // Session A should contain "Message for A" but not "Message for B"
    assert!(
        a_contents
            .iter()
            .any(|c: &String| c.contains("Message for A")),
        "Session A missing its own message"
    );
    assert!(
        !a_contents
            .iter()
            .any(|c: &String| c.contains("Message for B")),
        "Session A contains session B's message"
    );

    // Session B should contain "Message for B" but not "Message for A"
    assert!(
        b_contents
            .iter()
            .any(|c: &String| c.contains("Message for B")),
        "Session B missing its own message"
    );
    assert!(
        !b_contents
            .iter()
            .any(|c: &String| c.contains("Message for A")),
        "Session B contains session A's message"
    );
}

/// Reproduces the Slack bug: user asks "can you run python3 --version" and
/// the LLM returns a generic text response instead of calling the terminal tool.
///
/// This test proves that tools ARE provided to the LLM for Owner users,
/// so the bug is in the LLM's decision-making — not the agent wiring.
#[tokio::test]
async fn test_owner_receives_tools_for_command_request() {
    // Simulate the buggy LLM behavior: return text instead of calling a tool
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "I've received your message. I'm currently processing it via my daemon core.",
    )]);

    let harness = setup_test_agent(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "slack_session",
            "can you run python3 --version",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "slack".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: None,
                sender_id: None,
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    // The agent faithfully returns the LLM's response (the bug)
    assert_eq!(
        response,
        "I've received your message. I'm currently processing it via my daemon core."
    );

    // But crucially: the LLM was given tools to use (it just chose not to)
    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 1);
    assert!(
        !call_log[0].tools.is_empty(),
        "Owner user should have tools available, but tools were empty"
    );
}

/// Verify that Public users get NO tools — the LLM can only respond
/// conversationally. This is the expected behavior for non-whitelisted
/// users on Slack who @mention the bot or DM it.
#[tokio::test]
async fn test_public_user_gets_no_tools() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let _response = harness
        .agent
        .handle_message(
            "public_session",
            "can you run python3 --version",
            None,
            UserRole::Public,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "slack".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: None,
                sender_id: None,
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    // Public user: LLM should have received ZERO tools
    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 1);
    assert!(
        call_log[0].tools.is_empty(),
        "Public user should have NO tools, but got: {:?}",
        call_log[0].tools
    );
}

/// The correct behavior: when the LLM properly calls the terminal tool,
/// the agent executes it and returns the final response.
/// (Uses system_info as proxy since terminal has side effects.)
#[tokio::test]
async fn test_command_request_with_tool_use() {
    // Script the correct behavior: LLM calls a tool, then responds with output
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Python 3.11.5"),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "slack_session",
            "can you run python3 --version",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "slack".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: None,
                sender_id: None,
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    // Agent should loop: tool call → execute → LLM again → final text
    assert_eq!(harness.provider.call_count().await, 2);
    assert_eq!(response, "Python 3.11.5");

    // Second LLM call should include the tool result in messages
    let call_log = harness.provider.call_log.lock().await;
    let second_call_msgs = &call_log[1].messages;
    let has_tool_result = second_call_msgs
        .iter()
        .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"));
    assert!(
        has_tool_result,
        "Second LLM call should include tool result message"
    );
}

/// Diagnostic test: inspect the EXACT payload sent to the LLM when an Owner
/// asks "can you run python3 --version" on Slack. Verifies:
/// 1. System prompt includes terminal tool instructions
/// 2. Tool definitions include "terminal" tool
/// 3. User message is passed correctly
/// 4. No confusing context that might cause the LLM to skip tool use
#[tokio::test]
async fn test_slack_command_request_llm_payload() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let _response = harness
        .agent
        .handle_message(
            "slack_diag",
            "can you run python3 --version",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "slack".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: None,
                sender_id: None,
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 1, "Expected exactly 1 LLM call");

    let call = &call_log[0];

    // Check tool definitions are present
    let tool_names: Vec<String> = call
        .tools
        .iter()
        .filter_map(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .map(String::from)
        })
        .collect();
    assert!(!tool_names.is_empty(), "No tools provided to LLM");
    // In test harness we only have system_info, but in production terminal would be here
    assert!(
        tool_names.contains(&"system_info".to_string()),
        "system_info tool missing from tool defs: {:?}",
        tool_names
    );

    // Check system prompt is the first message with role "system"
    let system_msg = call
        .messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"));
    assert!(system_msg.is_some(), "No system message found in LLM call");
    let system_content = system_msg
        .unwrap()
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    // System prompt should mention the user role
    assert!(
        system_content.contains("[User Role: owner]"),
        "System prompt missing Owner role tag. Got: ...{}...",
        &system_content[system_content.len().saturating_sub(200)..]
    );
    // Should NOT contain the Public user restriction
    assert!(
        !system_content.contains("You have NO tools available"),
        "System prompt incorrectly contains Public user restriction for Owner"
    );

    // Check user message is present
    let user_msg = call
        .messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"));
    assert!(user_msg.is_some(), "No user message found in LLM call");
    let user_content = user_msg
        .unwrap()
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    assert!(
        user_content.contains("python3 --version"),
        "User message doesn't contain the original request: {}",
        user_content
    );
}

/// Verify that the Google GenAI provider does NOT include google_search
/// grounding when function tools are present. Grounding alongside function
/// declarations caused the model to skip tool calls and hallucinate text.
#[tokio::test]
async fn test_gemini_no_grounding_with_function_tools() {
    use serde_json::json;

    let provider = crate::providers::GoogleGenAiProvider::new("fake-key");

    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                }
            }
        }
    })];

    // When function tools are present, grounding must be disabled
    let converted = provider.convert_tools_for_test(&tools, false);
    assert!(converted.is_some(), "Should have tools");
    let tools_array = converted.unwrap();

    // Should have only 1 entry: function_declarations (NO google_search)
    assert_eq!(
        tools_array.len(),
        1,
        "Expected only function_declarations, got: {:?}",
        tools_array
    );
    assert!(
        tools_array[0].get("function_declarations").is_some(),
        "Missing function_declarations"
    );
    assert!(
        tools_array[0].get("google_search").is_none(),
        "google_search should NOT be present alongside function tools"
    );

    // When no function tools, grounding IS included
    let converted_grounding = provider.convert_tools_for_test(&[], true);
    assert!(converted_grounding.is_some());
    let grounding_array = converted_grounding.unwrap();
    assert_eq!(grounding_array.len(), 1);
    assert!(
        grounding_array[0].get("google_search").is_some(),
        "google_search should be present when no function tools"
    );
}

/// Guests should not receive tools under owner-only policy, and the system
/// prompt should include the owner-only restriction.
#[tokio::test]
async fn test_guest_user_has_no_tools_with_owner_only_warning() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let _response = harness
        .agent
        .handle_message(
            "guest_session",
            "run some command",
            None,
            UserRole::Guest,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 1);

    // Guest should have no tools under owner-only policy
    assert!(
        call_log[0].tools.is_empty(),
        "Guest user should not have tools available"
    );

    // System prompt should contain the guest warning
    let system_msg = call_log[0]
        .messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"));
    let system_content = system_msg
        .unwrap()
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    assert!(
        system_content.contains("[User Role: guest]"),
        "System prompt missing Guest role tag"
    );
    assert!(
        system_content.contains("Tool access is owner-only"),
        "System prompt missing owner-only tool-access warning"
    );
}

/// Owner system prompt should NOT contain guest/public warnings.
#[tokio::test]
async fn test_owner_system_prompt_has_no_restrictions() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let _response = harness
        .agent
        .handle_message(
            "owner_session",
            "hello",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let system_msg = call_log[0]
        .messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"));
    let system_content = system_msg
        .unwrap()
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");

    assert!(
        system_content.contains("[User Role: owner]"),
        "System prompt missing Owner role tag"
    );
    assert!(
        !system_content.contains("current user is a guest"),
        "Owner should NOT have guest warning"
    );
    assert!(
        !system_content.contains("NO tools available"),
        "Owner should NOT have public restriction"
    );
}

#[tokio::test]
async fn test_system_prompt_pins_critical_facts_for_owner_dm() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

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
    harness
        .state
        .upsert_fact(
            "assistant",
            "bot_name",
            "TestBot",
            "owner",
            None,
            crate::types::FactPrivacy::Global,
        )
        .await
        .unwrap();
    harness
        .state
        .upsert_fact(
            "user",
            "spouse",
            "Alice",
            "owner",
            None,
            crate::types::FactPrivacy::Global,
        )
        .await
        .unwrap();

    let _response = harness
        .agent
        .handle_message(
            "owner_session",
            "Give me a quick recap.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let system_msg = call_log[0]
        .messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .expect("system message should be present");
    let system_content = system_msg
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");

    assert!(
        system_content.contains("[Critical Facts — Highest Priority For Recall]"),
        "Critical facts block should be injected in owner DM prompts"
    );
    assert!(system_content.contains("Owner name: Test Owner"));
    assert!(system_content.contains("Assistant name: TestBot"));
    assert!(system_content.contains("partner: Alice"));
}

// ==========================================================================
// End-to-end channel simulation tests
//
// These replicate the auth + role logic from each channel (Telegram, Slack,
// Discord) and verify the agent produces the correct outcome for each user
// scenario. This catches regressions where a channel config change would
// silently break the user experience.
// ==========================================================================

/// Helper: replicate Telegram's auth + role logic.
/// Returns None if the user would be rejected, or Some(UserRole).
fn simulate_telegram_auth(
    allowed_user_ids: &mut Vec<u64>,
    owner_user_ids: &[u64],
    user_id: u64,
) -> Option<UserRole> {
    // Step 1: auth check (mirrors telegram.rs handle_message)
    use crate::channels::telegram::{check_auth, determine_role, AuthResult};
    let auth = if allowed_user_ids.is_empty() {
        check_auth(allowed_user_ids, user_id)
    } else if allowed_user_ids.contains(&user_id) {
        AuthResult::Authorized
    } else {
        AuthResult::Unauthorized
    };

    match auth {
        AuthResult::Unauthorized => None,
        AuthResult::Authorized | AuthResult::AutoClaimed => {
            Some(determine_role(owner_user_ids, user_id))
        }
    }
}

/// Helper: replicate Slack's auth logic.
fn simulate_slack_auth(
    allowed_user_ids: &[String],
    user_id: &str,
    is_dm: bool,
) -> Option<UserRole> {
    let is_whitelisted =
        !allowed_user_ids.is_empty() && allowed_user_ids.contains(&user_id.to_string());
    if is_whitelisted {
        Some(UserRole::Owner)
    } else if is_dm {
        Some(UserRole::Public)
    } else {
        None // silently ignored
    }
}

/// Helper: replicate Discord's auth logic (always Owner).
fn simulate_discord_auth() -> Option<UserRole> {
    Some(UserRole::Owner)
}

// --- Telegram scenarios ---

/// Scenario: New user sets up Telegram, no allowed_user_ids in config.
/// Expected: auto-claimed as Owner, gets tools.
#[tokio::test]
async fn test_telegram_first_user_auto_claim() {
    let mut allowed: Vec<u64> = vec![];
    let owner_ids: Vec<u64> = vec![];
    let role = simulate_telegram_auth(&mut allowed, &owner_ids, 12345);

    assert_eq!(
        role,
        Some(UserRole::Owner),
        "First user should be auto-claimed as Owner"
    );
    assert_eq!(
        allowed,
        vec![12345],
        "User ID should be persisted to allow list"
    );

    // Verify agent treats them as full Owner
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message(
            "tg_123",
            "hello",
            None,
            role.unwrap(),
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        !call_log[0].tools.is_empty(),
        "Auto-claimed user should have tools"
    );
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    assert!(
        sys["content"]
            .as_str()
            .unwrap()
            .contains("[User Role: owner]"),
        "Auto-claimed user should be Owner"
    );
}

/// Scenario: Telegram user in allowed_user_ids, no owner_ids configured.
/// Expected: Owner (the fix — previously was Guest).
#[tokio::test]
async fn test_telegram_allowed_user_no_owner_ids() {
    let mut allowed = vec![111];
    let owner_ids: Vec<u64> = vec![];
    let role = simulate_telegram_auth(&mut allowed, &owner_ids, 111);

    assert_eq!(
        role,
        Some(UserRole::Owner),
        "Allowed user with no owner_ids should be Owner"
    );

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message(
            "tg_111",
            "run a command",
            None,
            role.unwrap(),
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log[0].tools.is_empty(), "Should have tools as Owner");
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    assert!(
        !sys["content"]
            .as_str()
            .unwrap()
            .contains("current user is a guest"),
        "Should NOT have guest warning"
    );
}

/// Scenario: Telegram user in allowed_user_ids AND owner_ids.
/// Expected: Owner.
#[tokio::test]
async fn test_telegram_user_in_owner_ids() {
    let mut allowed = vec![111, 222];
    let owner_ids = vec![111];
    let role = simulate_telegram_auth(&mut allowed, &owner_ids, 111);

    assert_eq!(role, Some(UserRole::Owner));
}

/// Scenario: Telegram user in allowed_user_ids but NOT in owner_ids.
/// Expected: Guest role with no tools.
#[tokio::test]
async fn test_telegram_guest_user() {
    let mut allowed = vec![111, 222];
    let owner_ids = vec![111]; // only 111 is owner
    let role = simulate_telegram_auth(&mut allowed, &owner_ids, 222);

    assert_eq!(
        role,
        Some(UserRole::Guest),
        "Non-owner allowed user should be Guest"
    );

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message(
            "tg_222",
            "hello",
            None,
            role.unwrap(),
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

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
        "Guest should see owner-only warning"
    );
}

/// Scenario: Telegram user NOT in allowed_user_ids.
/// Expected: Rejected (no agent call).
#[tokio::test]
async fn test_telegram_unauthorized_user() {
    let mut allowed = vec![111];
    let owner_ids: Vec<u64> = vec![];
    let role = simulate_telegram_auth(&mut allowed, &owner_ids, 999);

    assert_eq!(role, None, "Unauthorized user should be rejected");
}

/// Scenario: After auto-claim, a second unknown user tries to message.
/// Expected: Rejected.
#[tokio::test]
async fn test_telegram_second_user_after_auto_claim_rejected() {
    let mut allowed: Vec<u64> = vec![];
    let owner_ids: Vec<u64> = vec![];

    // First user auto-claims
    let role1 = simulate_telegram_auth(&mut allowed, &owner_ids, 111);
    assert_eq!(role1, Some(UserRole::Owner));

    // Second user is rejected
    let role2 = simulate_telegram_auth(&mut allowed, &owner_ids, 222);
    assert_eq!(
        role2, None,
        "Second user should be rejected after auto-claim"
    );
}

// --- Slack scenarios ---

/// Scenario: Slack whitelisted user DMs bot.
/// Expected: Owner with full tool access.
#[tokio::test]
async fn test_slack_whitelisted_user_is_owner() {
    let allowed = vec!["U12345".to_string()];
    let role = simulate_slack_auth(&allowed, "U12345", true);

    assert_eq!(role, Some(UserRole::Owner));

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message(
            "slack_U12345",
            "do something",
            None,
            role.unwrap(),
            ChannelContext::private("slack"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        !call_log[0].tools.is_empty(),
        "Slack owner should have tools"
    );
}

/// Scenario: Non-whitelisted Slack user DMs the bot.
/// Expected: Public (no tools).
#[tokio::test]
async fn test_slack_non_whitelisted_dm_is_public() {
    let allowed = vec!["U12345".to_string()];
    let role = simulate_slack_auth(&allowed, "U99999", true);

    assert_eq!(role, Some(UserRole::Public));

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message(
            "slack_U99999",
            "run command",
            None,
            role.unwrap(),
            ChannelContext::private("slack"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        call_log[0].tools.is_empty(),
        "Public Slack user should have no tools"
    );
}

/// Scenario: Non-whitelisted Slack user in a channel (not DM, no @mention).
/// Expected: Silently ignored (no agent call).
#[tokio::test]
async fn test_slack_non_whitelisted_channel_ignored() {
    let allowed = vec!["U12345".to_string()];
    let role = simulate_slack_auth(&allowed, "U99999", false);

    assert_eq!(
        role, None,
        "Non-whitelisted channel message should be ignored"
    );
}

// --- Discord scenarios ---

/// Scenario: Any Discord user sends a message.
/// Expected: Always Owner (Discord has no auth check currently).
#[tokio::test]
async fn test_discord_always_owner() {
    let role = simulate_discord_auth();

    assert_eq!(role, Some(UserRole::Owner));

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message(
            "discord_123",
            "hello",
            None,
            role.unwrap(),
            ChannelContext::private("discord"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(
        !call_log[0].tools.is_empty(),
        "Discord user should have tools"
    );
}

/// Verify tool access tiers: Owner gets tools, Guest/Public get none.
/// This is the key regression test for the role system.
#[tokio::test]
async fn test_role_tool_access_tiers() {
    let roles_and_expected = vec![
        (UserRole::Owner, true, "owner"),
        (UserRole::Guest, false, "guest"),
        (UserRole::Public, false, "public"),
    ];

    for (role, should_have_tools, label) in roles_and_expected {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();

        let _response = harness
            .agent
            .handle_message(
                &format!("{}_session", label),
                "test message",
                None,
                role,
                ChannelContext::private("test"),
                None,
            )
            .await
            .unwrap();

        let call_log = harness.provider.call_log.lock().await;
        let has_tools = !call_log[0].tools.is_empty();
        assert_eq!(
            has_tools, should_have_tools,
            "{} user: expected tools={}, got tools={}",
            label, should_have_tools, has_tools
        );
    }
}
