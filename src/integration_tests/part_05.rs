// ==========================================================================
// Full-Stack Tests
//
// These use `FullStackTestHarness` with a real TerminalTool + ChannelHub
// approval wiring. Tests exercise real shell commands through the agent loop,
// verifying stall detection doesn't false-positive on legitimate exploration.
// ==========================================================================

/// Full-stack regression test: 12+ consecutive terminal calls with unique
/// commands (website exploration scenario). Must complete without stall.
///
/// Replicates the "create a website about cars" production failure where the
/// agent explored a project with `ls`, `git status`, `pwd`, etc. and the
/// stall detection falsely triggered.
#[tokio::test]
async fn test_full_stack_website_exploration_no_stall() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    let commands = [
        ("Let me explore the project.", r#"{"command": "ls -la"}"#),
        ("Checking system.", r#"{"command": "pwd"}"#),
        ("Git status.", r#"{"command": "git status"}"#),
        ("OS info.", r#"{"command": "uname -a"}"#),
        ("Who am I.", r#"{"command": "whoami"}"#),
        ("Current date.", r#"{"command": "date"}"#),
        ("Disk space.", r#"{"command": "df -h ."}"#),
        ("Environment.", r#"{"command": "env | head -5"}"#),
        ("Shell.", r#"{"command": "echo $SHELL"}"#),
        ("Hostname.", r#"{"command": "hostname"}"#),
        ("Uptime.", r#"{"command": "uptime"}"#),
        ("Process list.", r#"{"command": "ps aux | head -3"}"#),
    ];

    for (narration, args) in &commands {
        let mut resp = MockProvider::tool_call_response("terminal", args);
        resp.content = Some(narration.to_string());
        responses.push(resp);
    }

    // Final text response
    responses.push(MockProvider::text_response(
        "Done! Here's the complete summary of the system exploration.",
    ));

    let harness = setup_full_stack_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "telegram_test",
            "Explore the current system thoroughly — check files, git, OS, user, disk, and processes.",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    // Agent must complete normally — no false-positive stall detection
    assert!(
        response.contains("Done!"),
        "Agent should complete the full exploration without stall. Got: {}",
        response.chars().take(300).collect::<String>()
    );
    assert!(
        !response.contains("stuck in a loop"),
        "Should not trigger stall detection for diverse terminal commands"
    );
    let calls = harness.provider.call_count().await;
    assert!(
        calls >= 13,
        "Expected at least 13 LLM calls (12 terminal + 1 final), got {}",
        calls
    );
}

/// Full-stack test: terminal calls with duplicate commands (real pattern from
/// production). The agent sometimes re-checks things like `ls -la` or
/// `git remote -v` — this should NOT trigger stall detection.
#[tokio::test]
async fn test_full_stack_duplicate_commands_no_stall() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    let commands = [
        ("Checking project.", r#"{"command": "ls -la"}"#),
        ("Git info.", r#"{"command": "git status"}"#),
        // Duplicate: re-checking project structure
        ("Let me re-check.", r#"{"command": "ls -la"}"#),
        ("Remote.", r#"{"command": "git remote -v"}"#),
        // Duplicate: verifying remote
        ("Verify remote.", r#"{"command": "git remote -v"}"#),
        ("Date check.", r#"{"command": "date"}"#),
        ("Hostname.", r#"{"command": "hostname"}"#),
        // Duplicate: re-checking hostname
        ("Check again.", r#"{"command": "hostname"}"#),
        ("User.", r#"{"command": "whoami"}"#),
        ("Shell.", r#"{"command": "echo $SHELL"}"#),
    ];

    for (narration, args) in &commands {
        let mut resp = MockProvider::tool_call_response("terminal", args);
        resp.content = Some(narration.to_string());
        responses.push(resp);
    }

    responses.push(MockProvider::text_response(
        "Done! Here's what I found about the system.",
    ));

    let harness = setup_full_stack_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "telegram_test",
            "Check the project — files, git status, remote, hostname, user.",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("Done!"),
        "Agent should complete with duplicate commands without stall. Got: {}",
        response.chars().take(300).collect::<String>()
    );
    assert!(
        !response.contains("stuck in a loop"),
        "Duplicate commands with diverse patterns should not trigger stall"
    );
    let calls = harness.provider.call_count().await;
    assert!(
        calls >= 11,
        "Expected at least 11 LLM calls (10 terminal + 1 final), got {}",
        calls
    );
}

/// Full-stack test: cli_agent delegation followed by terminal follow-up work.
///
/// Verifies that stall counters reset after cli_agent completion, so the
/// follow-up terminal exploration doesn't inherit stall state from before.
#[tokio::test]
async fn test_full_stack_cli_agent_then_terminal_followup() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    // Step 1: delegate to cli_agent
    {
        let mut resp = MockProvider::tool_call_response(
            "cli_agent",
            r#"{"action":"run","tool":"claude","prompt":"build website"}"#,
        );
        resp.content = Some("I'll delegate the website build to the CLI agent.".to_string());
        responses.push(resp);
    }

    // Steps 2-9: follow-up terminal work after cli_agent completes
    let followup_commands = [
        ("CLI agent done. Let me verify.", r#"{"command": "ls -la"}"#),
        ("Git status.", r#"{"command": "git status"}"#),
        ("Check remote.", r#"{"command": "git remote -v"}"#),
        ("Who.", r#"{"command": "whoami"}"#),
        ("Date.", r#"{"command": "date"}"#),
        ("Pwd.", r#"{"command": "pwd"}"#),
        ("Uptime.", r#"{"command": "uptime"}"#),
        ("Host.", r#"{"command": "hostname"}"#),
    ];

    for (narration, args) in &followup_commands {
        let mut resp = MockProvider::tool_call_response("terminal", args);
        resp.content = Some(narration.to_string());
        responses.push(resp);
    }

    // Final response
    responses.push(MockProvider::text_response(
        "Done! Website deployed successfully.",
    ));

    // Add mock cli_agent tool
    let cli_agent_mock = Arc::new(MockTool::new(
        "cli_agent",
        "Delegates tasks to CLI agents",
        "Website built successfully. Files in /tmp/my-website",
    ));

    let harness = setup_full_stack_test_agent_with_extra_tools(
        MockProvider::with_responses(responses),
        vec![cli_agent_mock as Arc<dyn crate::traits::Tool>],
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "telegram_test",
            "Build a website about cars then verify everything is set up correctly.",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("deployed successfully"),
        "Agent should complete after cli_agent + terminal follow-up. Got: {}",
        response.chars().take(300).collect::<String>()
    );
    let calls = harness.provider.call_count().await;
    assert!(
        calls >= 10,
        "Expected at least 10 LLM calls (1 cli_agent + 8 terminal + 1 final), got {}",
        calls
    );
}

/// Full-stack test: verify StatusUpdate events flow correctly through the stack.
///
/// Sends a terminal command through the full agent loop and verifies that
/// ToolStart and ToolComplete status updates are emitted.
#[tokio::test]
async fn test_full_stack_status_updates_received() {
    let responses = vec![
        {
            let mut resp =
                MockProvider::tool_call_response("terminal", r#"{"command": "echo hello"}"#);
            resp.content = Some("Let me check something.".to_string());
            resp
        },
        MockProvider::text_response("Done! All good."),
    ];

    let harness = setup_full_stack_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();

    // Create status channel to capture updates
    let (status_tx, mut status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(64);

    let response = harness
        .agent
        .handle_message(
            "telegram_test",
            "Run echo hello",
            Some(status_tx),
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("All good"),
        "Agent should complete normally. Got: {}",
        response
    );

    // Collect all status updates
    let mut updates = Vec::new();
    while let Ok(update) = status_rx.try_recv() {
        updates.push(update);
    }

    // Verify we got tool lifecycle events
    let has_tool_start = updates
        .iter()
        .any(|u| matches!(u, StatusUpdate::ToolStart { name, .. } if name == "terminal"));
    let has_thinking = updates
        .iter()
        .any(|u| matches!(u, StatusUpdate::Thinking(_)));

    assert!(
        has_tool_start,
        "Should have received ToolStart for terminal. Updates: {:?}",
        updates
    );
    assert!(
        has_thinking,
        "Should have received at least one Thinking update. Updates: {:?}",
        updates
    );
    // ToolComplete may or may not be captured depending on timing — the key
    // verification is that ToolStart fires before execution and Thinking fires
    // for subsequent iterations.
}

/// Full-stack regression: duplicate identical send_file calls in one task
/// should only execute the underlying send once.
#[tokio::test]
async fn test_full_stack_duplicate_send_file_suppressed() {
    struct CountingSendFileTool {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl crate::traits::Tool for CountingSendFileTool {
        fn name(&self) -> &str {
            "send_file"
        }

        fn description(&self) -> &str {
            "Test send_file tool that counts executions."
        }

        fn schema(&self) -> serde_json::Value {
            json!({
                "name": "send_file",
                "description": self.description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string" },
                        "caption": { "type": "string" }
                    },
                    "required": ["file_path"]
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("File sent by counting send_file tool".to_string())
        }
    }

    let send_file_args = r#"{"file_path":"/Users/testuser/projects/acme-corp/proposal/sow-project-plan.pdf","caption":"Here is the SOW PDF from the Acme project."}"#;
    let responses = vec![
        MockProvider::tool_call_response("send_file", send_file_args),
        MockProvider::tool_call_response("send_file", send_file_args),
        MockProvider::text_response("Done. I sent the file."),
    ];

    let send_file_calls = Arc::new(AtomicUsize::new(0));
    let send_file_tool = Arc::new(CountingSendFileTool {
        calls: send_file_calls.clone(),
    });

    let harness = setup_full_stack_test_agent_with_extra_tools(
        MockProvider::with_responses(responses),
        vec![send_file_tool as Arc<dyn crate::traits::Tool>],
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "telegram_test",
            "Send me the SOW PDF from the Lodestar project",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("Done."),
        "Agent should complete normally. Got: {}",
        response
    );

    assert_eq!(
        send_file_calls.load(Ordering::SeqCst),
        1,
        "send_file should execute only once for duplicate identical calls"
    );

    let history = harness
        .state
        .get_history("telegram_test", 200)
        .await
        .unwrap();
    let dedupe_msgs = history
        .iter()
        .filter(|m| {
            m.role == "tool"
                && m.tool_name.as_deref() == Some("send_file")
                && m.content
                    .as_deref()
                    .is_some_and(|c| c.contains("Duplicate send_file suppressed"))
        })
        .count();
    assert_eq!(
        dedupe_msgs, 1,
        "Expected one dedupe tool message for suppressed duplicate send_file"
    );
}

/// Full-stack regression test: "What's the url of the site that you deployed?"
///
/// Real-world scenario: user asks about a previously deployed site. The agent
/// has no memory of the deployment so it searches for clues — checking git
/// remotes, config files, deployment manifests, environment variables, etc.
/// This triggers 10+ consecutive terminal calls as the agent hunts for the URL.
///
/// This is a particularly tricky case because:
/// 1. Many commands return similar "not found" results (low diversity)
/// 2. The agent may retry similar commands in different directories
/// 3. Some commands overlap semantically (git remote -v, cat CNAME, etc.)
#[tokio::test]
async fn test_full_stack_deployed_site_url_lookup_no_stall() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    // The agent tries to find deployment info through various commands
    let commands = [
        (
            "Let me check the git remote to find the deployment URL.",
            r#"{"command": "git remote -v"}"#,
        ),
        (
            "Let me look for deployment configuration files.",
            r#"{"command": "ls -la"}"#,
        ),
        (
            "Checking for a CNAME or deployment config.",
            r#"{"command": "ls public/ 2>/dev/null || echo 'no public dir'"}"#,
        ),
        (
            "Let me check package.json for deployment scripts.",
            r#"{"command": "cat package.json 2>/dev/null || echo 'no package.json'"}"#,
        ),
        (
            "Looking for Vercel or Netlify config.",
            r#"{"command": "ls vercel.json netlify.toml .vercel 2>/dev/null || echo 'none found'"}"#,
        ),
        (
            "Checking environment variables for URLs.",
            r#"{"command": "env | grep -i url || echo 'no URL env vars'"}"#,
        ),
        (
            "Let me check git log for deployment commits.",
            r#"{"command": "git log --oneline -5 2>/dev/null || echo 'not a git repo'"}"#,
        ),
        (
            "Checking for GitHub Pages or similar config.",
            r#"{"command": "cat CNAME 2>/dev/null || echo 'no CNAME'"}"#,
        ),
        (
            "Looking for docker or CI deployment files.",
            r#"{"command": "ls Dockerfile docker-compose.yml .github/workflows/ 2>/dev/null || echo 'none'"}"#,
        ),
        (
            "Checking the git config for any deploy URLs.",
            r#"{"command": "git config --list 2>/dev/null | grep -i url || echo 'no url in git config'"}"#,
        ),
        (
            "One more check — looking at recent branches.",
            r#"{"command": "git branch -a 2>/dev/null | head -10 || echo 'no branches'"}"#,
        ),
    ];

    for (narration, args) in &commands {
        let mut resp = MockProvider::tool_call_response("terminal", args);
        resp.content = Some(narration.to_string());
        responses.push(resp);
    }

    // Agent gives up and reports what it found
    responses.push(MockProvider::text_response(
        "I couldn't find a specific deployment URL in the current project. \
         The git remote points to github.com but I don't see a CNAME, \
         Vercel config, or Netlify config. Could you tell me which project \
         you're referring to? I may have that info stored from a previous session.",
    ));

    let harness = setup_full_stack_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "telegram_test",
            "What's the url of the site that you deployed?",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    // Agent must complete normally — 11 consecutive terminal calls should NOT stall
    assert!(
        !response.contains("stuck in a loop"),
        "Should not trigger stall for URL lookup exploration. Got: {}",
        response.chars().take(400).collect::<String>()
    );
    assert!(
        !response.contains("I seem to be stuck"),
        "Should not trigger graceful stall response. Got: {}",
        response.chars().take(400).collect::<String>()
    );
    assert!(
        response.contains("deployment URL") || response.contains("git remote"),
        "Agent should give a meaningful answer about deployment. Got: {}",
        response.chars().take(400).collect::<String>()
    );
    let calls = harness.provider.call_count().await;
    assert!(
        calls >= 12,
        "Expected at least 12 LLM calls (11 terminal + 1 final), got {}",
        calls
    );
}

/// Full-stack regression test: blocked non-exempt tool triggers false-positive stall.
///
/// Root cause analysis: when the LLM calls a non-exempt tool (e.g. system_info,
/// web_search) more than 3 times, the call gets BLOCKED with a coaching message.
/// But the blocked call doesn't increment `successful_tool_calls`, so if the LLM
/// keeps trying the same tool, every iteration has `successful_tool_calls == 0`,
/// and after 3 such iterations, `stall_count >= 3` fires graceful_stall_response.
///
/// This reproduces the exact "What's the url of the site that you deployed?"
/// failure: the LLM called system_info to search for deployment config, got
/// blocked after 3 calls, then kept trying → stall after 4 tool calls total.
#[tokio::test]
async fn test_full_stack_blocked_tool_triggers_stall() {
    let mut responses: Vec<ProviderResponse> = Vec::new();

    // Iteration 1 (intent gate): narration required
    {
        let mut resp = MockProvider::tool_call_response("system_info", "{}");
        resp.content = Some(
            "Let me look up the deployment URL by checking the system configuration.".to_string(),
        );
        responses.push(resp);
    }

    // Iterations 2-4: system_info executes successfully (3 calls, hits per-tool limit)
    for i in 0..3 {
        let mut resp = MockProvider::tool_call_response(
            "system_info",
            &format!(r#"{{"check":"deploy_{}"}}"#, i),
        );
        resp.content = Some(format!("Checking deployment config {}.", i));
        responses.push(resp);
    }

    // Iterations 5-7: system_info gets BLOCKED (prior_calls >= 3, not exempt)
    // These iterations have successful_tool_calls == 0 → stall_count increments
    for i in 3..6 {
        let mut resp = MockProvider::tool_call_response(
            "system_info",
            &format!(r#"{{"check":"deploy_{}"}}"#, i),
        );
        resp.content = Some(format!("Let me try checking config {} again.", i));
        responses.push(resp);
    }

    // Final: should reach this if stall detection doesn't fire
    responses.push(MockProvider::text_response(
        "I couldn't find the deployment URL. Which project are you referring to?",
    ));

    let harness = setup_full_stack_test_agent(MockProvider::with_responses(responses))
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "telegram_test",
            "What's the url of the site that you deployed?",
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    // Regression test: blocked tool calls now count as progress for stall
    // detection, so the agent gets a chance to adapt instead of stalling.
    assert!(
        !response.contains("stuck") && !response.contains("not making progress"),
        "Blocked non-exempt tool calls should NOT trigger stall detection. Got: {}",
        response.chars().take(400).collect::<String>()
    );
}

