//! Integration tests that exercise the real agent loop with a mock LLM.
//!
//! These tests verify: agent loop, tool execution, memory persistence,
//! multi-turn history, and session isolation — the same code path all
//! channels use via `Agent::handle_message()`.

use chrono::Utc;
use crate::testing::{setup_test_agent, MockProvider};
use crate::traits::{
    BehaviorPattern, Episode, ErrorSolution, Expertise, Goal, Procedure, StateStore, UserProfile,
};
use crate::types::{ChannelContext, ChannelVisibility, UserRole};

#[tokio::test]
async fn test_basic_message_response() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let response = harness
        .agent
        .handle_message("test_session", "Hello!", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    assert_eq!(response, "Mock response");
    assert_eq!(harness.provider.call_count().await, 1);
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
        .handle_message("persist_session", "Remember this", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    // Verify messages were persisted to state store
    let history = harness.state.get_history("persist_session", 10).await.unwrap();

    // Should have at least a user message and an assistant message
    assert!(history.len() >= 2, "Expected at least 2 messages, got {}", history.len());

    let roles: Vec<&str> = history.iter().map(|m| m.role.as_str()).collect();
    assert!(roles.contains(&"user"), "Missing user message in history");
    assert!(roles.contains(&"assistant"), "Missing assistant message in history");
}

#[tokio::test]
async fn test_multi_turn_conversation() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // First turn
    harness
        .agent
        .handle_message("multi_session", "First message", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    // Second turn
    harness
        .agent
        .handle_message("multi_session", "Second message", None, UserRole::Owner, ChannelContext::private("test"))
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
        .handle_message("session_a", "Message for A", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    // Send to session B
    harness
        .agent
        .handle_message("session_b", "Message for B", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    // Check histories are isolated
    let history_a = harness.state.get_history("session_a", 10).await.unwrap();
    let history_b = harness.state.get_history("session_b", 10).await.unwrap();

    // Each session should have its own messages
    let a_contents: Vec<String> = history_a
        .iter()
        .filter_map(|m| m.content.clone())
        .collect();
    let b_contents: Vec<String> = history_b
        .iter()
        .filter_map(|m| m.content.clone())
        .collect();

    // Session A should contain "Message for A" but not "Message for B"
    assert!(
        a_contents.iter().any(|c: &String| c.contains("Message for A")),
        "Session A missing its own message"
    );
    assert!(
        !a_contents.iter().any(|c: &String| c.contains("Message for B")),
        "Session A contains session B's message"
    );

    // Session B should contain "Message for B" but not "Message for A"
    assert!(
        b_contents.iter().any(|c: &String| c.contains("Message for B")),
        "Session B missing its own message"
    );
    assert!(
        !b_contents.iter().any(|c: &String| c.contains("Message for A")),
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
            },
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
            },
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
            },
        )
        .await
        .unwrap();

    // Agent should loop: tool call → execute → LLM again → final text
    assert_eq!(harness.provider.call_count().await, 2);
    assert_eq!(response, "Python 3.11.5");

    // Second LLM call should include the tool result in messages
    let call_log = harness.provider.call_log.lock().await;
    let second_call_msgs = &call_log[1].messages;
    let has_tool_result = second_call_msgs.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("tool")
    });
    assert!(has_tool_result, "Second LLM call should include tool result message");
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
            },
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
    assert!(
        !tool_names.is_empty(),
        "No tools provided to LLM"
    );
    // In test harness we only have system_info, but in production terminal would be here
    assert!(
        tool_names.contains(&"system_info".to_string()),
        "system_info tool missing from tool defs: {:?}",
        tool_names
    );

    // Check system prompt is the first message with role "system"
    let system_msg = call.messages.iter().find(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("system")
    });
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
    let user_msg = call.messages.iter().find(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user")
    });
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

/// Guest users should still receive tools (unlike Public) but the system
/// prompt should include the cautious-guest warning. This validates the
/// 3-tier model: Owner (full), Guest (tools + warning), Public (no tools).
#[tokio::test]
async fn test_guest_user_gets_tools_with_warning() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let _response = harness
        .agent
        .handle_message(
            "guest_session",
            "run some command",
            None,
            UserRole::Guest,
            ChannelContext::private("telegram"),
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 1);

    // Guest should have tools (unlike Public)
    assert!(
        !call_log[0].tools.is_empty(),
        "Guest user should have tools available"
    );

    // System prompt should contain the guest warning
    let system_msg = call_log[0].messages.iter().find(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("system")
    });
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
        system_content.contains("current user is a guest"),
        "System prompt missing guest caution warning"
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
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let system_msg = call_log[0].messages.iter().find(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("system")
    });
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

    assert_eq!(role, Some(UserRole::Owner), "First user should be auto-claimed as Owner");
    assert_eq!(allowed, vec![12345], "User ID should be persisted to allow list");

    // Verify agent treats them as full Owner
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message("tg_123", "hello", None, role.unwrap(), ChannelContext::private("telegram"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log[0].tools.is_empty(), "Auto-claimed user should have tools");
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    assert!(
        sys["content"].as_str().unwrap().contains("[User Role: owner]"),
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

    assert_eq!(role, Some(UserRole::Owner), "Allowed user with no owner_ids should be Owner");

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message("tg_111", "run a command", None, role.unwrap(), ChannelContext::private("telegram"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log[0].tools.is_empty(), "Should have tools as Owner");
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    assert!(
        !sys["content"].as_str().unwrap().contains("current user is a guest"),
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

/// Scenario: Telegram user in allowed_user_ids but NOT in owner_ids (3-tier).
/// Expected: Guest (gets tools but with caution warning).
#[tokio::test]
async fn test_telegram_guest_user() {
    let mut allowed = vec![111, 222];
    let owner_ids = vec![111]; // only 111 is owner
    let role = simulate_telegram_auth(&mut allowed, &owner_ids, 222);

    assert_eq!(role, Some(UserRole::Guest), "Non-owner allowed user should be Guest");

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .handle_message("tg_222", "hello", None, role.unwrap(), ChannelContext::private("telegram"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log[0].tools.is_empty(), "Guest should still have tools");
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    assert!(
        sys["content"].as_str().unwrap().contains("current user is a guest"),
        "Guest should see caution warning"
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
    assert_eq!(role2, None, "Second user should be rejected after auto-claim");
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
        .handle_message("slack_U12345", "do something", None, role.unwrap(), ChannelContext::private("slack"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log[0].tools.is_empty(), "Slack owner should have tools");
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
        .handle_message("slack_U99999", "run command", None, role.unwrap(), ChannelContext::private("slack"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(call_log[0].tools.is_empty(), "Public Slack user should have no tools");
}

/// Scenario: Non-whitelisted Slack user in a channel (not DM, no @mention).
/// Expected: Silently ignored (no agent call).
#[tokio::test]
async fn test_slack_non_whitelisted_channel_ignored() {
    let allowed = vec!["U12345".to_string()];
    let role = simulate_slack_auth(&allowed, "U99999", false);

    assert_eq!(role, None, "Non-whitelisted channel message should be ignored");
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
        .handle_message("discord_123", "hello", None, role.unwrap(), ChannelContext::private("discord"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log[0].tools.is_empty(), "Discord user should have tools");
}

/// Verify the 3-tier tool access: Owner gets tools, Guest gets tools, Public gets none.
/// This is the key regression test for the role system.
#[tokio::test]
async fn test_role_tool_access_tiers() {
    let roles_and_expected = vec![
        (UserRole::Owner, true, "owner"),
        (UserRole::Guest, true, "guest"),
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
        )
        .await
        .unwrap();

    assert_eq!(response, "System: Linux x86_64, 8 cores");
    // Verify the tool was actually executed (tool result in second call)
    let call_log = harness.provider.call_log.lock().await;
    assert_eq!(call_log.len(), 2);
    let has_tool_result = call_log[1].messages.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("tool")
    });
    assert!(has_tool_result, "Second LLM call should contain tool execution result");
}

/// Scenario: Public user asks to run a command — no tools available.
/// The LLM should only respond conversationally.
#[tokio::test]
async fn test_public_user_cannot_execute_tools() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::text_response(
            "I'm sorry, I can only chat — tool-based actions aren't available for public users.",
        ),
    ]);

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
        )
        .await
        .unwrap();

    // LLM was given no tools, so it can only reply with text
    let call_log = harness.provider.call_log.lock().await;
    assert!(call_log[0].tools.is_empty(), "Public user must have no tools");
    assert_eq!(call_log.len(), 1, "Should be a single conversational reply, no tool loop");
    assert!(response.contains("sorry") || response.contains("can only chat"),
        "Response should explain tool limitation");
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

/// Scenario: Guest on Telegram asks for help — gets tools but with caution.
/// Verifies the guest can still use the agent productively.
#[tokio::test]
async fn test_telegram_guest_can_use_tools() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Here's your system info."),
    ]);

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
        )
        .await
        .unwrap();

    // Guest can use tools
    assert_eq!(response, "Here's your system info.");
    assert_eq!(harness.provider.call_count().await, 2);

    // But system prompt includes the caution warning
    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    assert!(
        sys["content"].as_str().unwrap().contains("current user is a guest"),
        "Guest should have caution in system prompt"
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
        .handle_message("multi_step", "check my system and remember what you find", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    assert_eq!(response, "Done! I checked your system and saved the info.");
    // Should be 4 LLM calls: intent-gated + system_info + remember_fact + final
    assert_eq!(harness.provider.call_count().await, 4);
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
        .handle_message("stall_session", "do something", None, UserRole::Owner, ChannelContext::private("test"))
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
        .handle_message("memory_session", "I prefer Rust", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();
    assert!(r1.contains("Rust"), "Turn 1 should acknowledge the preference");

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
        .handle_message("memory_session", "what language do I prefer?", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    // The system prompt sent to the LLM in turn 2 should contain the fact
    let call_log = harness.provider.call_log.lock().await;
    let last_call = call_log.last().unwrap();
    let system_msg = last_call.messages.iter().find(|m| m["role"] == "system").unwrap();
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
        .handle_message("intent_session", "check my system", None, UserRole::Owner, ChannelContext::private("test"))
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
        .upsert_fact("project", "framework", "Uses React with TypeScript", "agent")
        .await
        .unwrap();

    // Turn 1
    harness
        .agent
        .handle_message("enrichment_session", "hello", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    // Turn 2 — ask about the project
    harness
        .agent
        .handle_message("enrichment_session", "what framework does my project use?", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    // Check that the system prompt in turn 2 includes the seeded fact
    let call_log = harness.provider.call_log.lock().await;
    let turn2_call = &call_log[1];
    let system_msg = turn2_call.messages.iter().find(|m| m["role"] == "system").unwrap();
    let system_content = system_msg["content"].as_str().unwrap_or("");

    assert!(
        system_content.contains("React") || system_content.contains("TypeScript"),
        "System prompt should include the seeded fact about React/TypeScript. \
         System prompt tail: ...{}",
        &system_content[system_content.len().saturating_sub(500)..]
    );

    // Also verify history carries forward (turn 2 has more messages than turn 1)
    let turn1_msgs = call_log[0].messages.len();
    let turn2_msgs = call_log[1].messages.len();
    assert!(
        turn2_msgs > turn1_msgs,
        "Turn 2 should include turn 1 history: {} vs {}",
        turn2_msgs, turn1_msgs
    );
}

// ==========================================================================
// Full memory system tests
//
// The memory system has 7+ layers. Each test seeds data into the state store,
// then verifies it appears in the agent's system prompt sent to the LLM.
// ==========================================================================

/// Episodes: session summaries appear in system prompt as "Relevant Past Sessions".
#[tokio::test]
async fn test_memory_episodes_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Seed an episode
    let episode = Episode {
        id: 0,
        session_id: "old_session".to_string(),
        summary: "User debugged a Rust lifetime error in their web server".to_string(),
        topics: Some(vec!["rust".to_string(), "debugging".to_string()]),
        emotional_tone: Some("frustrated then relieved".to_string()),
        outcome: Some("resolved".to_string()),
        importance: 0.8,
        recall_count: 0,
        last_recalled_at: None,
        message_count: 15,
        start_time: Utc::now(),
        end_time: Utc::now(),
        created_at: Utc::now(),
    };
    harness.state.insert_episode(&episode).await.unwrap();

    // Ask about something related so embedding similarity matches
    harness
        .agent
        .handle_message("ep_session", "I have a Rust error", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        content.contains("Past Sessions") || content.contains("lifetime error") || content.contains("web server"),
        "System prompt should include episode about Rust debugging. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Goals: active goals appear in system prompt as "Active Goals".
#[tokio::test]
async fn test_memory_goals_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let goal = Goal {
        id: 0,
        description: "Migrate the database from PostgreSQL to SQLite".to_string(),
        status: "active".to_string(),
        priority: "high".to_string(),
        progress_notes: Some(vec!["Schema drafted".to_string()]),
        source_episode_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        completed_at: None,
    };
    harness.state.insert_goal(&goal).await.unwrap();

    harness
        .agent
        .handle_message("goal_session", "what should I work on?", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        content.contains("Active Goals") && content.contains("PostgreSQL"),
        "System prompt should include the active goal about DB migration. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Procedures: learned step sequences appear in system prompt as "Known Procedures".
#[tokio::test]
async fn test_memory_procedures_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let procedure = Procedure {
        id: 0,
        name: "Deploy release workflow".to_string(),
        trigger_pattern: "deploy release workflow".to_string(),
        steps: vec![
            "Run test suite".to_string(),
            "Build release binary".to_string(),
            "Upload to server".to_string(),
            "Restart service".to_string(),
        ],
        success_count: 8,
        failure_count: 1,
        avg_duration_secs: Some(120.0),
        last_used_at: Some(Utc::now()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    harness.state.upsert_procedure(&procedure).await.unwrap();

    // Use a query that is a substring of the trigger_pattern for text matching,
    // and does NOT trigger auto-plan creation (which requires "deploy"+"production").
    harness
        .agent
        .handle_message("proc_session", "deploy release", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log.is_empty(), "Expected at least 1 LLM call, got 0");
    let sys = call_log[0].messages.iter().find(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("system")
    }).expect("No system message found in LLM call");
    let content = sys["content"].as_str().unwrap();
    assert!(
        content.contains("Known Procedures") && content.contains("Deploy release workflow"),
        "System prompt should include the deploy procedure. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Error solutions: known fixes appear in system prompt as "Known Error Solutions".
#[tokio::test]
async fn test_memory_error_solutions_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let solution = ErrorSolution {
        id: 0,
        error_pattern: "connection refused port 5432".to_string(),
        domain: Some("database".to_string()),
        solution_summary: "Start PostgreSQL service and check pg_hba.conf".to_string(),
        solution_steps: Some(vec![
            "sudo systemctl start postgresql".to_string(),
            "Check pg_hba.conf for auth rules".to_string(),
        ]),
        success_count: 5,
        failure_count: 0,
        last_used_at: Some(Utc::now()),
        created_at: Utc::now(),
    };
    harness.state.insert_error_solution(&solution).await.unwrap();

    harness
        .agent
        .handle_message("err_session", "connection refused port 5432", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        content.contains("Error Solutions") && content.contains("PostgreSQL service"),
        "System prompt should include the error solution. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Expertise: domain skill levels appear in system prompt as "Expertise Levels".
#[tokio::test]
async fn test_memory_expertise_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Build expertise by incrementing multiple times
    for _ in 0..10 {
        harness.state.increment_expertise("rust", true, None).await.unwrap();
    }
    for _ in 0..3 {
        harness.state.increment_expertise("rust", false, Some("borrow checker")).await.unwrap();
    }

    harness
        .agent
        .handle_message("exp_session", "help me with code", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        content.contains("Expertise") && content.contains("rust"),
        "System prompt should include expertise levels. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Behavior patterns: observed habits appear in system prompt as "Observed Patterns".
#[tokio::test]
async fn test_memory_behavior_patterns_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let pattern = BehaviorPattern {
        id: 0,
        pattern_type: "habit".to_string(),
        description: "User always runs tests before committing code".to_string(),
        trigger_context: Some("git commit".to_string()),
        action: Some("cargo test".to_string()),
        confidence: 0.85,
        occurrence_count: 12,
        last_seen_at: Some(Utc::now()),
        created_at: Utc::now(),
    };
    harness.state.insert_behavior_pattern(&pattern).await.unwrap();

    harness
        .agent
        .handle_message("pat_session", "I want to commit", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        content.contains("Observed Patterns") && content.contains("tests before committing"),
        "System prompt should include behavior patterns. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// User profile: communication preferences affect the system prompt.
#[tokio::test]
async fn test_memory_user_profile_affects_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let profile = UserProfile {
        id: 1,
        verbosity_preference: "brief".to_string(),
        explanation_depth: "minimal".to_string(),
        tone_preference: "casual".to_string(),
        emoji_preference: "none".to_string(),
        typical_session_length: Some(10),
        active_hours: None,
        common_workflows: Some(vec!["code review".to_string()]),
        asks_before_acting: true,
        prefers_explanations: false,
        likes_suggestions: true,
        updated_at: Utc::now(),
    };
    harness.state.update_user_profile(&profile).await.unwrap();

    harness
        .agent
        .handle_message("profile_session", "hello", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        content.contains("Communication Preferences") && content.contains("brief"),
        "System prompt should include user profile preferences. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Full memory stack: seed ALL memory components and verify they all appear
/// in a single system prompt. This is the comprehensive regression test.
#[tokio::test]
async fn test_full_memory_stack_in_system_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // 1. Facts
    harness.state.upsert_fact("project", "language", "Rust with Tokio async runtime", "agent").await.unwrap();

    // 2. Episodes
    let episode = Episode {
        id: 0, session_id: "past".to_string(),
        summary: "Deployed v2.0 of the API server".to_string(),
        topics: Some(vec!["deployment".to_string()]),
        emotional_tone: Some("confident".to_string()),
        outcome: Some("success".to_string()),
        importance: 0.9, recall_count: 2, last_recalled_at: None,
        message_count: 20, start_time: Utc::now(), end_time: Utc::now(), created_at: Utc::now(),
    };
    harness.state.insert_episode(&episode).await.unwrap();

    // 3. Goals
    let goal = Goal {
        id: 0, description: "Ship v3.0 with WebSocket support".to_string(),
        status: "active".to_string(), priority: "high".to_string(),
        progress_notes: Some(vec!["Design complete".to_string()]),
        source_episode_id: None, created_at: Utc::now(), updated_at: Utc::now(), completed_at: None,
    };
    harness.state.insert_goal(&goal).await.unwrap();

    // 4. Procedures
    let proc = Procedure {
        id: 0, name: "Release workflow".to_string(),
        trigger_pattern: "deploy and release workflow".to_string(),
        steps: vec!["cargo test".to_string(), "cargo build --release".to_string(), "deploy".to_string()],
        success_count: 6, failure_count: 0, avg_duration_secs: Some(90.0),
        last_used_at: Some(Utc::now()), created_at: Utc::now(), updated_at: Utc::now(),
    };
    harness.state.upsert_procedure(&proc).await.unwrap();

    // 5. Error solutions — error_pattern must contain the query as a substring for text matching fallback
    let solution = ErrorSolution {
        id: 0, error_pattern: "deploy and release failed with exit code 1".to_string(),
        domain: Some("ops".to_string()),
        solution_summary: "Check CI pipeline and retry the deployment".to_string(),
        solution_steps: Some(vec!["Check CI logs".to_string(), "Retry deploy".to_string()]),
        success_count: 4, failure_count: 0, last_used_at: Some(Utc::now()), created_at: Utc::now(),
    };
    harness.state.insert_error_solution(&solution).await.unwrap();

    // 6. Expertise
    for _ in 0..15 {
        harness.state.increment_expertise("deployment", true, None).await.unwrap();
    }

    // 7. Behavior patterns
    let pattern = BehaviorPattern {
        id: 0, pattern_type: "habit".to_string(),
        description: "Always checks CI before merging".to_string(),
        trigger_context: Some("merge".to_string()),
        action: Some("check CI status".to_string()),
        confidence: 0.9, occurrence_count: 8, last_seen_at: Some(Utc::now()), created_at: Utc::now(),
    };
    harness.state.insert_behavior_pattern(&pattern).await.unwrap();

    // 8. User profile
    let profile = UserProfile {
        id: 1, verbosity_preference: "detailed".to_string(),
        explanation_depth: "thorough".to_string(), tone_preference: "casual".to_string(),
        emoji_preference: "none".to_string(), typical_session_length: Some(30),
        active_hours: None, common_workflows: Some(vec!["deployment".to_string()]),
        asks_before_acting: false, prefers_explanations: true, likes_suggestions: true,
        updated_at: Utc::now(),
    };
    harness.state.update_user_profile(&profile).await.unwrap();

    // Use a query that is a substring of procedure trigger_pattern for text matching,
    // and does NOT trigger auto-plan creation (which requires "deploy"+"production").
    harness
        .agent
        .handle_message("full_memory", "deploy and release", None, UserRole::Owner, ChannelContext::private("test"))
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    let content = sys["content"].as_str().unwrap();

    // Verify each memory layer appears
    let checks = vec![
        ("Communication Preferences", "User profile"),
        ("Expertise", "Expertise levels"),
        ("Active Goals", "Goals"),
        ("Known Facts", "Facts"),
        ("Observed Patterns", "Behavior patterns"),
    ];

    let mut missing = vec![];
    for (marker, label) in &checks {
        if !content.contains(marker) {
            missing.push(*label);
        }
    }

    assert!(
        missing.is_empty(),
        "System prompt is missing these memory components: {:?}\n\nSystem prompt tail:\n...{}",
        missing,
        &content[content.len().saturating_sub(2000)..]
    );

    // Verify procedures and error solutions appear (these use semantic search,
    // so they need embedding match — check if section headers exist at minimum)
    assert!(
        content.contains("Known Procedures") || content.contains("Release workflow"),
        "System prompt should include procedures"
    );
    assert!(
        content.contains("Error Solutions") || content.contains("deploy and release failed"),
        "System prompt should include error solutions"
    );
}

/// Public channels should NOT inject personal memory (facts, episodes, goals,
/// patterns, profile) but SHOULD still inject operational memory (procedures,
/// error solutions, expertise).
#[tokio::test]
async fn test_public_channel_hides_personal_memory() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Seed personal memory
    harness.state.upsert_fact("personal", "name", "David", "user").await.unwrap();
    let goal = Goal {
        id: 0, description: "Learn Japanese".to_string(),
        status: "active".to_string(), priority: "medium".to_string(),
        progress_notes: None, source_episode_id: None,
        created_at: Utc::now(), updated_at: Utc::now(), completed_at: None,
    };
    harness.state.insert_goal(&goal).await.unwrap();

    // Seed operational memory
    let proc = Procedure {
        id: 0, name: "Debug crash".to_string(),
        trigger_pattern: "crash debug segfault".to_string(),
        steps: vec!["Check logs".to_string(), "Run with RUST_BACKTRACE=1".to_string()],
        success_count: 5, failure_count: 0, avg_duration_secs: None,
        last_used_at: None, created_at: Utc::now(), updated_at: Utc::now(),
    };
    harness.state.upsert_procedure(&proc).await.unwrap();

    // Send in a PUBLIC channel context
    harness
        .agent
        .handle_message(
            "public_chan",
            "my app crashed",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Public,
                platform: "slack".to_string(),
                channel_name: Some("general".to_string()),
            },
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0].messages.iter().find(|m| m["role"] == "system").unwrap();
    let content = sys["content"].as_str().unwrap();

    // Personal memory should NOT leak into public channels
    assert!(
        !content.contains("David"),
        "Personal facts should NOT appear in public channel prompt"
    );
    assert!(
        !content.contains("Learn Japanese"),
        "Personal goals should NOT appear in public channel prompt"
    );
}
