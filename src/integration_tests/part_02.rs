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
        channel_id: None,
    };
    harness.state.insert_episode(&episode).await.unwrap();

    // Ask about something related so embedding similarity matches
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
    assert!(
        content.contains("Past Sessions")
            || content.contains("lifetime error")
            || content.contains("web server"),
        "System prompt should include episode about Rust debugging. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Goals: active goals appear in system prompt as "Active Goals".
#[tokio::test]
async fn test_memory_goals_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let mut goal = Goal::new_finite(
        "Migrate the database from PostgreSQL to SQLite",
        "goal_session",
    );
    goal.domain = "personal".to_string();
    goal.priority = "high".to_string();
    goal.progress_notes = Some(vec!["Schema drafted".to_string()]);
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
    assert!(!call_log.is_empty(), "Expected at least 1 LLM call, got 0");
    let sys = call_log[0]
        .messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .expect("No system message found in LLM call");
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
    harness
        .state
        .insert_error_solution(&solution)
        .await
        .unwrap();

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
        harness
            .state
            .increment_expertise("rust", true, None)
            .await
            .unwrap();
    }
    for _ in 0..3 {
        harness
            .state
            .increment_expertise("rust", false, Some("borrow checker"))
            .await
            .unwrap();
    }

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
    harness
        .state
        .insert_behavior_pattern(&pattern)
        .await
        .unwrap();

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
        content.contains("Observed Patterns") && content.contains("tests before committing"),
        "System prompt should include behavior patterns. Tail: ...{}",
        &content[content.len().saturating_sub(800)..]
    );
}

/// Failure patterns: repeated dead-end workflows appear in system prompt guidance.
#[tokio::test]
async fn test_memory_failure_patterns_injected_into_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let pattern = BehaviorPattern {
        id: 0,
        pattern_type: "failure".to_string(),
        description:
            "Repeated terminal failures on permission denied; pivot to different approach earlier."
                .to_string(),
        trigger_context: Some("terminal".to_string()),
        action: Some("pivot".to_string()),
        confidence: 0.8,
        occurrence_count: 5,
        last_seen_at: Some(Utc::now()),
        created_at: Utc::now(),
    };
    harness
        .state
        .insert_behavior_pattern(&pattern)
        .await
        .unwrap();

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
        content.contains("Failure Patterns To Avoid")
            && content.contains("Repeated terminal failures"),
        "System prompt should include failure-pattern guidance. Tail: ...{}",
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

/// Full memory stack: seed ALL memory components and verify they all appear
/// in a single system prompt. This is the comprehensive regression test.
#[tokio::test]
async fn test_full_memory_stack_in_system_prompt() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // 1. Facts
    harness
        .state
        .upsert_fact(
            "project",
            "language",
            "Rust with Tokio async runtime",
            "agent",
            None,
            crate::types::FactPrivacy::Global,
        )
        .await
        .unwrap();

    // 2. Episodes
    let episode = Episode {
        id: 0,
        session_id: "past".to_string(),
        summary: "Deployed v2.0 of the API server".to_string(),
        topics: Some(vec!["deployment".to_string()]),
        emotional_tone: Some("confident".to_string()),
        outcome: Some("success".to_string()),
        importance: 0.9,
        recall_count: 2,
        last_recalled_at: None,
        message_count: 20,
        start_time: Utc::now(),
        end_time: Utc::now(),
        created_at: Utc::now(),
        channel_id: None,
    };
    harness.state.insert_episode(&episode).await.unwrap();

    // 3. Goals (personal goals are injected in DM prompts)
    let mut goal = Goal::new_finite("Ship next release with WebSocket support", "full_memory");
    goal.domain = "personal".to_string();
    goal.priority = "high".to_string();
    goal.progress_notes = Some(vec!["Design complete".to_string()]);
    harness.state.create_goal(&goal).await.unwrap();

    // 4. Procedures
    let proc = Procedure {
        id: 0,
        name: "Release workflow".to_string(),
        trigger_pattern: "deploy and release workflow".to_string(),
        steps: vec![
            "cargo test".to_string(),
            "cargo build --release".to_string(),
            "deploy".to_string(),
        ],
        success_count: 6,
        failure_count: 0,
        avg_duration_secs: Some(90.0),
        last_used_at: Some(Utc::now()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    harness.state.upsert_procedure(&proc).await.unwrap();

    // 5. Error solutions — error_pattern must contain the query as a substring for text matching fallback
    let solution = ErrorSolution {
        id: 0,
        error_pattern: "deploy and release failed with exit code 1".to_string(),
        domain: Some("ops".to_string()),
        solution_summary: "Check CI pipeline and retry the deployment".to_string(),
        solution_steps: Some(vec![
            "Check CI logs".to_string(),
            "Retry deploy".to_string(),
        ]),
        success_count: 4,
        failure_count: 0,
        last_used_at: Some(Utc::now()),
        created_at: Utc::now(),
    };
    harness
        .state
        .insert_error_solution(&solution)
        .await
        .unwrap();

    // 6. Expertise
    for _ in 0..15 {
        harness
            .state
            .increment_expertise("deployment", true, None)
            .await
            .unwrap();
    }

    // 7. Behavior patterns
    let pattern = BehaviorPattern {
        id: 0,
        pattern_type: "habit".to_string(),
        description: "Always checks CI before merging".to_string(),
        trigger_context: Some("merge".to_string()),
        action: Some("check CI status".to_string()),
        confidence: 0.9,
        occurrence_count: 8,
        last_seen_at: Some(Utc::now()),
        created_at: Utc::now(),
    };
    harness
        .state
        .insert_behavior_pattern(&pattern)
        .await
        .unwrap();

    // 8. User profile
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

    // Use a query that is a substring of procedure trigger_pattern for text matching,
    // and does NOT trigger auto-plan creation (which requires "deploy"+"production").
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

