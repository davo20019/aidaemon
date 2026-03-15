// ==========================================================================
// Minimal context system prompt tests
//
// Memory data (facts, goals, expertise, procedures, episodes, patterns) is
// retrieved on demand via tools, NOT bulk-injected into the system prompt.
// These tests verify the system prompt contains capability descriptions
// instead of raw data.
// ==========================================================================

/// System prompt contains memory capabilities summary instead of bulk data.
#[tokio::test]
async fn test_memory_episodes_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness
        .agent
        .handle_message(
            "ep_session",
            "I have a Rust error",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();
    // Episodes are no longer injected — memory is on-demand
    assert!(
        !content.contains("Past Sessions"),
        "System prompt should NOT contain bulk episode data"
    );
    assert!(
        content.contains("Your Memory"),
        "System prompt should contain memory capabilities summary"
    );
}

/// Goals are NOT bulk-injected — accessed on demand via scheduled_goals tool.
#[tokio::test]
async fn test_memory_goals_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let mut goal = Goal::new_finite(
        "Migrate the database from PostgreSQL to SQLite",
        "goal_session",
    );
    goal.domain = "personal".to_string();
    goal.priority = "high".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    harness
        .agent
        .handle_message(
            "goal_session",
            "what should I work on?",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        !content.contains("PostgreSQL"),
        "Goal descriptions should NOT be bulk-injected into prompt"
    );
    assert!(
        content.contains("scheduled_goals"),
        "System prompt should reference scheduled_goals tool for on-demand access"
    );
}

/// Procedures are NOT bulk-injected — model learns from tool descriptions.
#[tokio::test]
async fn test_memory_procedures_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness
        .agent
        .handle_message(
            "proc_session",
            "deploy release",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log.is_empty(), "Expected at least 1 LLM call");
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .expect("No system message found");
    let content = sys["content"].as_str().unwrap();
    assert!(
        !content.contains("Known Procedures"),
        "Procedures should NOT be bulk-injected into prompt"
    );
}

/// Error solutions are NOT bulk-injected — model uses tool error messages directly.
#[tokio::test]
async fn test_memory_error_solutions_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness
        .agent
        .handle_message(
            "err_session",
            "connection refused port 5432",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        !content.contains("Error Solutions"),
        "Error solutions should NOT be bulk-injected into prompt"
    );
}

/// Expertise levels are NOT bulk-injected — model adapts from conversation.
#[tokio::test]
async fn test_memory_expertise_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness
        .agent
        .handle_message(
            "exp_session",
            "help me with code",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        !content.contains("Expertise Levels"),
        "Expertise levels should NOT be bulk-injected into prompt"
    );
}

/// Behavior patterns are NOT bulk-injected.
#[tokio::test]
async fn test_memory_behavior_patterns_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness
        .agent
        .handle_message(
            "pat_session",
            "I want to commit",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        !content.contains("Observed Patterns"),
        "Behavior patterns should NOT be bulk-injected into prompt"
    );
}

/// Failure patterns are NOT bulk-injected.
#[tokio::test]
async fn test_memory_failure_patterns_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness
        .agent
        .handle_message(
            "failure_pattern_session",
            "help me debug this command issue",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        !content.contains("Failure Patterns To Avoid"),
        "Failure patterns should NOT be bulk-injected into prompt"
    );
}

/// Failure patterns are also not injected in public channels.
#[tokio::test]
async fn test_memory_failure_patterns_injected_into_public_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness
        .agent
        .handle_message(
            "public_failure_pattern_session",
            "help me debug this search workflow",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Public,
                platform: "test".to_string(),
                channel_name: Some("#eng".to_string()),
                channel_id: Some("test:eng".to_string()),
                sender_name: Some("Alice".to_string()),
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
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        !content.contains("Failure Patterns To Avoid"),
        "Failure patterns should NOT be bulk-injected into public prompt"
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
        .handle_message(
            "profile_session",
            "hello",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();
    assert!(
        content.contains("Communication Preferences") && content.contains("brief"),
        "System prompt should include user profile preferences. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Minimal context: system prompt has capability summary + profile, NOT bulk data.
#[tokio::test]
async fn test_full_memory_stack_in_system_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Seed user profile (this SHOULD still appear)
    let profile = UserProfile {
        id: 1,
        verbosity_preference: "detailed".to_string(),
        explanation_depth: "thorough".to_string(),
        tone_preference: "casual".to_string(),
        emoji_preference: "none".to_string(),
        typical_session_length: Some(30),
        active_hours: None,
        common_workflows: Some(vec!["deployment".to_string()]),
        asks_before_acting: false,
        prefers_explanations: true,
        likes_suggestions: true,
        updated_at: Utc::now(),
    };
    harness.state.update_user_profile(&profile).await.unwrap();

    harness
        .agent
        .handle_message(
            "full_memory",
            "deploy and release",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let call_log = harness.provider.call_log.lock().await;
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();

    // Profile SHOULD be present (stays in every prompt)
    assert!(
        content.contains("Communication Preferences"),
        "User profile should still be injected"
    );

    // Memory capabilities summary SHOULD be present
    assert!(
        content.contains("Your Memory"),
        "Memory capabilities summary should be present"
    );
    assert!(
        content.contains("manage_memories"),
        "Should reference manage_memories tool"
    );

    // Bulk data should NOT be present
    let should_not_contain = vec![
        "Known Procedures",
        "Error Solutions",
        "Expertise Levels",
        "Active Goals",
        "Known Facts",
        "Observed Patterns",
        "Failure Patterns",
        "Trusted Command Patterns",
        "Contextual Suggestions",
        "Relevant Past Sessions",
    ];
    for marker in &should_not_contain {
        assert!(
            !content.contains(marker),
            "System prompt should NOT contain '{}' — memory is on-demand now",
            marker
        );
    }
}

/// Public channels: Global facts are accessible, Private facts are not.
/// Goals, patterns, and profile are DM-only and should NOT appear.
#[tokio::test]
async fn test_public_channel_hides_personal_memory() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Global fact — should appear in public channels (accessible everywhere)
    harness
        .state
        .upsert_fact(
            "personal",
            "name",
            "Alice",
            "user",
            None,
            crate::types::FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Private fact — should NOT appear in public channels
    harness
        .state
        .upsert_fact(
            "preference",
            "hobby",
            "Enjoys photography",
            "user",
            None,
            crate::types::FactPrivacy::Private,
        )
        .await
        .unwrap();

    // Personal goal — DM-only, should NOT appear in public channels
    let mut goal = Goal::new_finite("Learn Japanese", "public_chan");
    goal.domain = "personal".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    // Seed operational memory
    let proc = Procedure {
        id: 0,
        name: "Debug crash".to_string(),
        trigger_pattern: "crash debug segfault".to_string(),
        steps: vec![
            "Check logs".to_string(),
            "Run with RUST_BACKTRACE=1".to_string(),
        ],
        success_count: 5,
        failure_count: 0,
        avg_duration_secs: None,
        last_used_at: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
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
                channel_id: Some("slack:C_TEST".to_string()),
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
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    let content = sys["content"].as_str().unwrap();

    // Private facts should NOT appear in public channels
    assert!(
        !content.contains("Enjoys photography"),
        "Private facts should NOT appear in public channel prompt"
    );
    // Goals are DM-only — should NOT appear in public channels
    assert!(
        !content.contains("Learn Japanese"),
        "Personal goals should NOT appear in public channel prompt"
    );
}

/// Verify that all tool schemas the agent sends to the LLM are compatible
/// with the Google Gemini API. Gemini rejects `$schema` fields anywhere in
/// function declaration parameters. This test catches any tool (built-in or
/// MCP) that introduces `$schema` before it causes a 400 error in production.
#[tokio::test]
async fn test_tool_schemas_gemini_compatible() {
    use serde_json::json;

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Trigger a normal message so the agent builds tool definitions
    let _response = harness
        .agent
        .handle_message(
            "schema_check",
            "hello",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Grab the actual tools sent to the LLM
    let call_log = harness.provider.call_log.lock().await;
    assert!(!call_log.is_empty(), "Expected at least 1 LLM call");
    let tools = &call_log[0].tools;
    assert!(!tools.is_empty(), "Expected tools in the LLM call");

    // Run them through the Gemini convert_tools pipeline
    let gemini = crate::providers::GoogleGenAiProvider::new("fake-key");
    let converted = gemini.convert_tools_for_test(tools, false);
    assert!(converted.is_some(), "convert_tools should return Some");

    // Recursively check no $schema anywhere in the output
    fn assert_no_schema(value: &serde_json::Value, path: &str) {
        match value {
            serde_json::Value::Object(map) => {
                assert!(
                    !map.contains_key("$schema"),
                    "Gemini-incompatible '$schema' found at: {}",
                    path
                );
                for (k, v) in map {
                    assert_no_schema(v, &format!("{}.{}", path, k));
                }
            }
            serde_json::Value::Array(arr) => {
                for (i, v) in arr.iter().enumerate() {
                    assert_no_schema(v, &format!("{}[{}]", path, i));
                }
            }
            _ => {}
        }
    }

    assert_no_schema(&json!(converted.unwrap()), "gemini_tools");
}
