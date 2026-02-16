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

    // Agent must complete normally — no stall detection
    assert!(
        response.contains("Done!"),
        "Agent should complete the full exploration without stall. Got: {}",
        response.chars().take(300).collect::<String>()
    );
    // 3 system_info + 9 remember_fact + 1 final text = 13 LLM calls
    // (system_info calls 4+ get blocked but the iteration still counts)
    let calls = harness.provider.call_count().await;
    assert!(
        calls >= 12,
        "Expected at least 12 LLM calls for full exploration, got {}",
        calls
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

    // Agent must complete normally despite 11 consecutive same-tool calls
    // (some with duplicate arguments)
    assert!(
        response.contains("gallery page"),
        "Agent should complete exploration without stall. Got: {}",
        response.chars().take(300).collect::<String>()
    );
    let calls = harness.provider.call_count().await;
    assert!(
        calls >= 11,
        "Expected at least 11 LLM calls for project exploration, got {}",
        calls
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

    assert!(
        response.contains("Finished all system checks"),
        "Agent should complete two-tool exploration without stall. Got: {}",
        response.chars().take(300).collect::<String>()
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

    // Check that the system prompt in turn 2 includes the seeded fact
    let call_log = harness.provider.call_log.lock().await;
    let turn2_call = &call_log[1];
    let system_msg = turn2_call
        .messages
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
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
        turn2_msgs,
        turn1_msgs
    );
}

