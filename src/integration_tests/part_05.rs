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
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            },
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
        "Should not trigger stall detection for diverse terminal commands"
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
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    // Agent should either complete normally or gracefully stall after making
    // meaningful progress (the new stopping_phase detects stall-with-progress
    // when total_successful_tool_calls >= 3 and returns a partial stall response).
    assert!(
        !response.is_empty(),
        "Agent should return a non-empty response"
    );
    assert!(
        !response.contains("stuck in a loop"),
        "Duplicate commands with diverse patterns should not trigger stall"
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
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            },
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
    // Status pings carry user-facing labels, not raw internal tool names —
    // `terminal` is relabeled to "running a command" (see
    // `sanitize::user_facing_tool_activity`).
    let has_tool_start = updates
        .iter()
        .any(|u| matches!(u, StatusUpdate::ToolStart { name, .. } if name == "running a command"));
    let dm_start_shows_command = updates.iter().any(|u| {
        matches!(
            u,
            StatusUpdate::ToolStart { name, summary }
                if name == "running a command" && summary.is_empty()
        )
    });
    let has_tool_complete = updates.iter().any(
        |u| matches!(u, StatusUpdate::ToolComplete { name, .. } if name == "running a command"),
    );
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
    assert!(
        has_tool_complete,
        "Should have received ToolComplete for terminal. Updates: {:?}",
        updates
    );
    assert!(
        dm_start_shows_command,
        "Private-DM ToolStart should suppress the command summary. Updates: {:?}",
        updates
    );
}

struct ExternalActionTool;

#[async_trait::async_trait]
impl crate::traits::Tool for ExternalActionTool {
    fn name(&self) -> &str {
        "external_action"
    }

    fn description(&self) -> &str {
        "Writes to an external service for testing."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "external_action",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        })
    }

    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        Ok("Created remote record id=abc123".to_string())
    }
}

struct PreWrappedExternalActionTool;

#[async_trait::async_trait]
impl crate::traits::Tool for PreWrappedExternalActionTool {
    fn name(&self) -> &str {
        "prewrapped_external_action"
    }

    fn description(&self) -> &str {
        "Writes to an external service and returns already-wrapped output."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "prewrapped_external_action",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        })
    }

    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        Ok(crate::tools::sanitize::wrap_untrusted_output(
            "prewrapped_external_action",
            "Created remote record id=wrapped123",
        ))
    }
}

struct StructuredExternalActionTool;

#[async_trait::async_trait]
impl crate::traits::Tool for StructuredExternalActionTool {
    fn name(&self) -> &str {
        "structured_external_action"
    }

    fn description(&self) -> &str {
        "Creates an external record and returns a structured HTTP response."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "structured_external_action",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        Ok(
            r#"HTTP 201 Created
content-type: application/json; charset=utf-8

JSON summary:
Top-level keys: data

{
  "data": {
    "edit_history_tweet_ids": [
      ""
    ],
    "id": "2082623804740620701",
    "text": "Synthetic engagement prompt"
  }
}"#
            .to_string(),
        )
    }
}

#[tokio::test]
async fn test_successful_mutation_receipt_is_not_reopened_by_candidate_wording() {
    let responses = vec![
        MockProvider::tool_call_response("structured_external_action", "{}"),
        MockProvider::text_response("I'll handle that now."),
    ];

    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        MockProvider::with_responses(responses),
        vec![Arc::new(StructuredExternalActionTool)],
        None,
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "structured_external_action_deferral",
            "Create the external post.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "I'll handle that now.");
    let tool_results = harness
        .agent
        .event_store()
        .query_events_by_types(
            "structured_external_action_deferral",
            &[crate::events::EventType::ToolResult],
            20,
        )
        .await
        .unwrap();
    assert!(tool_results.iter().any(|event| {
        event
            .parse_data::<crate::events::ToolResultData>()
            .is_ok_and(|result| result.receipt.is_some_and(|receipt| {
                receipt.outcome_status == crate::traits::ToolOutcomeStatus::Succeeded
                    && receipt.invocation_stage.reached_dispatch()
            }))
    }));
}

struct UrlProbeTool;

#[async_trait::async_trait]
impl crate::traits::Tool for UrlProbeTool {
    fn name(&self) -> &str {
        "url_probe"
    }

    fn description(&self) -> &str {
        "Reads a URL for verification testing."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "url_probe",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string" }
                },
                "required": ["url"],
                "additionalProperties": false
            }
        })
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: serde_json::Value = serde_json::from_str(arguments)?;
        let url = args
            .get("url")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing url"))?;
        Ok(format!(
            "Checked {} and confirmed the posts are visible.",
            url
        ))
    }
}

#[tokio::test]
async fn test_successful_external_action_timeout_returns_deterministic_completion() {
    let responses = vec![
        {
            let mut resp = MockProvider::tool_call_response("external_action", "{}");
            resp.content = Some("I'll handle that.".to_string());
            resp
        },
        MockProvider::text_response("This reply should time out before it is used."),
    ];

    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        MockProvider::with_delayed_responses(
            responses,
            vec![std::time::Duration::ZERO, std::time::Duration::from_secs(2)],
        ),
        vec![Arc::new(ExternalActionTool)],
        Some(1),
    )
    .await
    .unwrap();

    let (status_tx, mut status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(64);
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        harness.agent.handle_message(
            "external_action_timeout",
            "Create the remote record.",
            Some(status_tx),
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        ),
    )
    .await
    .expect("request should not hang")
    .unwrap();

    assert!(
        response.contains("The requested action completed successfully."),
        "response should use deterministic completion ack: {}",
        response
    );
    assert!(
        response.contains("Created remote record id=abc123"),
        "response should include the latest external action result: {}",
        response
    );

    let mut updates = Vec::new();
    while let Ok(update) = status_rx.try_recv() {
        updates.push(update);
    }

    assert!(
        updates.iter().any(|update| matches!(
            update,
            StatusUpdate::ToolComplete { name, summary }
                if name == "external_action" && summary.contains("Created remote record id=abc123")
        )),
        "expected ToolComplete status update for external_action, got: {:?}",
        updates
    );
}

#[tokio::test]
async fn test_prewrapped_external_action_timeout_keeps_latest_result() {
    let responses = vec![
        {
            let mut resp = MockProvider::tool_call_response("prewrapped_external_action", "{}");
            resp.content = Some("I'll handle that.".to_string());
            resp
        },
        MockProvider::text_response("This reply should time out before it is used."),
    ];

    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        MockProvider::with_delayed_responses(
            responses,
            vec![std::time::Duration::ZERO, std::time::Duration::from_secs(2)],
        ),
        vec![Arc::new(PreWrappedExternalActionTool)],
        Some(1),
    )
    .await
    .unwrap();

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        harness.agent.handle_message(
            "prewrapped_external_action_timeout",
            "Create the remote record.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        ),
    )
    .await
    .expect("request should not hang")
    .unwrap();

    assert!(
        response.contains("The requested action completed successfully."),
        "response should use deterministic completion ack: {}",
        response
    );
    assert!(
        response.contains("Created remote record id=wrapped123"),
        "response should preserve the latest external action result: {}",
        response
    );
    assert!(
        !response.contains("UNTRUSTED EXTERNAL DATA"),
        "response should not leak wrapper markers into the final reply: {}",
        response
    );
}

#[tokio::test]
async fn test_visible_outcome_request_requires_matching_verification_before_completion() {
    let responses = vec![
        {
            let mut resp = MockProvider::tool_call_response("external_action", "{}");
            resp.content = Some("I'll fix that.".to_string());
            resp
        },
        MockProvider::text_response("Done."),
        MockProvider::tool_call_response("url_probe", r#"{"url":"https://blog.aidaemon.ai"}"#),
        MockProvider::text_response(
            "I checked https://blog.aidaemon.ai and the posts are now visible.",
        ),
    ];

    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        MockProvider::with_responses(responses),
        vec![Arc::new(ExternalActionTool), Arc::new(UrlProbeTool)],
        None,
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "visible_outcome_verification",
            "I still don't see the posts here: https://blog.aidaemon.ai",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("posts are now visible"),
        "final response should reflect a verified outcome: {}",
        response
    );
    assert!(
        !response.contains("The requested action completed successfully."),
        "generic success ack should be blocked until verification completes: {}",
        response
    );
    assert_eq!(
        harness.provider.call_count().await,
        4,
        "agent should continue past the low-signal completion and perform verification"
    );
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

        fn capabilities(&self) -> crate::traits::ToolCapabilities {
            crate::traits::ToolCapabilities {
                read_only: false,
                external_side_effect: true,
                needs_approval: false,
                idempotent: false,
                high_impact_write: false,
            }
        }

        fn call_semantics(&self, arguments: &str) -> crate::traits::ToolCallSemantics {
            let path = serde_json::from_str::<serde_json::Value>(arguments)
                .ok()
                .and_then(|args| {
                    args.get("file_path")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_default();
            crate::traits::ToolCallSemantics::mutation_with(
                crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY,
            )
            .with_target_hint(crate::traits::ToolTargetHintKind::Path, path)
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

/// Regression: once a duplicate send_file is suppressed, the task should be
/// forced into text-only mode instead of repeatedly attempting more file sends.
#[tokio::test]
async fn test_duplicate_send_file_forces_text_closeout() {
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

        fn capabilities(&self) -> crate::traits::ToolCapabilities {
            crate::traits::ToolCapabilities {
                read_only: false,
                external_side_effect: true,
                needs_approval: false,
                idempotent: false,
                high_impact_write: false,
            }
        }

        fn call_semantics(&self, arguments: &str) -> crate::traits::ToolCallSemantics {
            let path = serde_json::from_str::<serde_json::Value>(arguments)
                .ok()
                .and_then(|args| {
                    args.get("file_path")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_default();
            crate::traits::ToolCallSemantics::mutation_with(
                crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY,
            )
            .with_target_hint(crate::traits::ToolTargetHintKind::Path, path)
        }
    }

    let send_file_args = r#"{"file_path":"/Users/testuser/projects/acme-corp/proposal/sow-project-plan.pdf","caption":"Here is the SOW PDF from the Acme project."}"#;
    let responses = vec![
        MockProvider::tool_call_response("send_file", send_file_args),
        MockProvider::tool_call_response("send_file", send_file_args),
        MockProvider::tool_call_response("send_file", send_file_args),
        MockProvider::text_response("Done. I already sent the file."),
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
            "telegram_test_force_text",
            "Send me the SOW PDF from the Lodestar project",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("Done. I already sent the file."),
        "Agent should close out with plain text after duplicate send_file. Got: {}",
        response
    );
    assert_eq!(
        send_file_calls.load(Ordering::SeqCst),
        1,
        "Only the first send_file call should execute"
    );

    let history = harness
        .state
        .get_history("telegram_test_force_text", 200)
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
        "Expected exactly one duplicate suppression message"
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
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    // Agent should either complete normally or gracefully stall after making
    // meaningful progress (the new stopping_phase detects stall-with-progress
    // when total_successful_tool_calls >= 3 and returns a partial stall response).
    assert!(
        !response.contains("stuck in a loop"),
        "Should not trigger stuck-in-a-loop message. Got: {}",
        response.chars().take(400).collect::<String>()
    );
    // Agent should either complete with a meaningful answer or stop gracefully
    // after making progress. The key invariant: no crash, no error, and not empty.
    assert!(
        !response.is_empty(),
        "Agent should return a non-empty response"
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
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:12345".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
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

struct FailingExternalActionTool;

#[async_trait::async_trait]
impl crate::traits::Tool for FailingExternalActionTool {
    fn name(&self) -> &str {
        "failing_external_action"
    }
    fn description(&self) -> &str {
        "Writes to an external service; always fails (testing)."
    }
    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "failing_external_action",
            "description": self.description(),
            "parameters": {"type": "object", "properties": {}}
        })
    }
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
    fn call_semantics(&self, _arguments: &str) -> crate::traits::ToolCallSemantics {
        crate::traits::ToolCallSemantics::mutation()
    }
    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        anyhow::bail!("synthetic external action failure")
    }
}

/// A model that gives up with an UNFIXED failed external mutation gets ONE
/// evidence-fed recovery pass demanding a different approach BEFORE the
/// honest failure report (live repro: task 2e87a458 — quoting failure,
/// retry "succeeded" with a traceback in stdout, model answered, user got
/// a report when a strategy change would likely have worked).
#[tokio::test]
async fn test_failed_mutation_gets_recovery_pass_before_report() {
    // Uncorrected failed mutations short-circuit straight into the
    // reconciliation zone, so the recovery pass fires on the FIRST give-up.
    let give_up = "I couldn't finish creating the record due to a parse error.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("failing_external_action", "{}"),
        MockProvider::text_response(give_up),
        // Recovery pass fires here: model pivots to a different tool that works.
        MockProvider::tool_call_response("external_action", "{}"),
        MockProvider::text_response(
            "The first method failed, so I used the alternative service — the record is created (id=abc123).",
        ),
        MockProvider::text_response(
            "1 attempt failed initially, but the retry succeeded — the record is created (id=abc123).",
        ),
        MockProvider::text_response(
            "1 attempt failed initially, but the retry succeeded — the record is created (id=abc123).",
        ),
    ]);
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![
            std::sync::Arc::new(FailingExternalActionTool)
                as std::sync::Arc<dyn crate::traits::Tool>,
            std::sync::Arc::new(ExternalActionTool) as std::sync::Arc<dyn crate::traits::Tool>,
        ],
        None,
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "tg_mutation_recovery",
            "Create the record in the external service",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The runtime observed a typed failure and a later satisfying receipt;
    // recovery never depends on a coaching phrase.
    let results = harness
        .agent
        .event_store()
        .query_events_by_types(
            "tg_mutation_recovery",
            &[crate::events::EventType::ToolResult],
            20,
        )
        .await
        .unwrap();
    let statuses = results
        .iter()
        .filter_map(|event| event.parse_data::<crate::events::ToolResultData>().ok())
        .filter_map(|result| result.receipt.map(|receipt| receipt.outcome_status))
        .collect::<Vec<_>>();
    assert!(statuses.contains(&crate::traits::ToolOutcomeStatus::FailedPermanent));
    assert!(statuses.contains(&crate::traits::ToolOutcomeStatus::Succeeded));

    // The pivot's success ships — not a failure report.
    assert!(
        response.contains("record is created") || response.contains("abc123"),
        "recovered outcome must ship: {response:?}"
    );
}

/// End-to-end proof of the runtime self-repair contract: a directory
/// capability the execution prelude projected on the model's behalf cannot be
/// prepared (its parent is a file), so the terminal adapter reports a typed
/// `RuntimePreparationFailure`; the dispatcher drops its own projection and
/// re-runs the prepared call without consuming a model iteration. The model's
/// command then runs exactly as declared and the model never sees the
/// runtime's mistake as a tool error.
#[tokio::test]
async fn test_full_stack_runtime_repairs_its_own_unpreparable_projection() {
    let workspace = tempfile::tempdir().expect("workspace");
    let blocker = workspace.path().join("blocker");
    std::fs::write(&blocker, "a file, not a directory").expect("blocker");
    // A future root beneath an existing FILE: passes the pre-projection disk
    // check (it does not exist) but can never be materialized (ENOTDIR).
    let impossible_root = blocker.join("sub").to_string_lossy().to_string();
    let output = workspace.path().join("out.txt").to_string_lossy().to_string();
    let cwd = workspace.path().to_string_lossy().to_string();

    // The typed assessment declares the impossible root as a write root, so
    // the contract compiler classifies it as a directory capability and the
    // prelude projects it into the terminal call's `write_roots`.
    let base = MockProvider::semantic_task_assessment(
        "change",
        true,
        false,
        &["local_workspace_write"],
        "new_request",
        "local_workspace",
    );
    let mut assessment: serde_json::Value =
        serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
    assessment["contract"]["filesystem_access"] = serde_json::json!({
        "execution_cwd": cwd,
        "read_paths": [],
        "write_paths": [output],
        "read_roots": [],
        "write_roots": [impossible_root],
    });
    let assessment = MockProvider::text_response(&assessment.to_string());

    let terminal_args = serde_json::json!({
        "command": format!("printf SYNTHETIC_REPAIR_OK > '{output}'"),
        "working_dir": cwd,
        "write_paths": [output],
    })
    .to_string();
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("terminal", &terminal_args),
        MockProvider::text_response("Wrote the file."),
    ])
    .with_task_assessments(vec![assessment]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();

    let session_id = "runtime_projection_repair";
    let response = harness
        .agent
        .handle_message(
            session_id,
            &format!("Write SYNTHETIC_REPAIR_OK to {output} (output root {impossible_root})."),
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:synthetic-owner".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();
    assert!(!response.is_empty());

    // The model's command ran exactly as declared.
    assert_eq!(
        std::fs::read_to_string(&output).expect("command output must exist"),
        "SYNTHETIC_REPAIR_OK"
    );

    let decision_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id",
    )
    .bind(session_id)
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    let repaired = decision_rows
        .iter()
        .map(|raw| serde_json::from_str::<serde_json::Value>(raw).unwrap())
        .find(|data| {
            data["metadata"]["condition"].as_str() == Some("runtime_projection_repaired")
        })
        .expect("the dispatcher must record that it repaired its own projection");
    assert_eq!(repaired["metadata"]["field"], "write_roots");
    assert_eq!(repaired["metadata"]["value"], impossible_root);
    assert_eq!(repaired["metadata"]["model_iteration_consumed"], false);

    // No tool result carrying the runtime's own preparation error ever
    // reached the model; the single durable terminal receipt is the
    // dispatched command.
    let tool_results: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'tool_result' AND tool_name = 'terminal' ORDER BY id",
    )
    .bind(session_id)
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    assert_eq!(tool_results.len(), 1, "{tool_results:?}");
    let receipt: serde_json::Value = serde_json::from_str(&tool_results[0]).unwrap();
    assert!(
        !receipt.to_string().contains("could not prepare"),
        "runtime preparation failure leaked to the model: {receipt}"
    );
    assert_eq!(receipt["receipt"]["invocation_stage"], "dispatched");
    assert_eq!(receipt["receipt"]["exit_code"], 0);
}

/// A pre-dispatch denial the contract itself asked for (`non_success_terminal`
/// receipt, read-only contract) is the terminal evidence. The completion gate
/// must let the model's typed recovery reply through instead of demanding an
/// evidence-seeking pass that can never succeed.
#[tokio::test]
async fn test_full_stack_typed_recovery_reply_after_negative_contract_denial() {
    let workspace = tempfile::tempdir().expect("workspace");
    let target = workspace.path().join("denied.txt").to_string_lossy().to_string();
    let cwd = workspace.path().to_string_lossy().to_string();

    let base = MockProvider::semantic_task_assessment(
        "check",
        false,
        true,
        &[],
        "new_request",
        "local_workspace",
    );
    let mut assessment: serde_json::Value =
        serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
    assessment["contract"]["mutation_scope"] = serde_json::json!("read_only");
    assessment["contract"]["evidence_requirements"] = serde_json::json!([{
        "summary": "Complete the exact requested machine invocation",
        "acceptable_scopes": ["local_workspace"],
        "purpose": "outcome",
        "minimum_authority": "direct",
        "temporal_scope": "historical",
        "receipt": {
            "tool_names": ["terminal"],
            "outcome_condition": "non_success_terminal",
            "requires_output": false,
            "min_invocations": 1,
            "max_invocations": 1
        }
    }]);
    let assessment = MockProvider::text_response(&assessment.to_string());

    let write_args = serde_json::json!({
        "command": format!("printf SYNTHETIC_DENIED > '{target}'"),
        "working_dir": cwd,
        "write_paths": [target],
    })
    .to_string();
    let marker = "phase=SYNTHETIC-RR; protected_write=denied; dependent_terminal=stopped";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("terminal", &write_args),
        MockProvider::text_response(marker),
        // If the gate wrongly demands another pass, the mock keeps answering
        // with the same typed reply; the test then fails on the final text.
        MockProvider::text_response(marker),
        MockProvider::text_response(marker),
    ])
    .with_task_assessments(vec![assessment]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();

    let session_id = "typed_recovery_after_denial";
    let response = harness
        .agent
        .handle_message(
            session_id,
            &format!(
                "Attempt exactly one write to {target}. If denied, stop and reply exactly {marker}"
            ),
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:synthetic-owner".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    assert!(
        !std::path::Path::new(&target).exists(),
        "the read-only contract must block the write"
    );
    assert!(
        response.contains(marker),
        "typed recovery reply must be delivered, got: {response}"
    );
    assert!(
        !response.contains("couldn't complete"),
        "generic failure shipped instead of the typed reply: {response}"
    );
    // The gate did not loop: no evidence-seeking pass was demanded after the
    // denial, and the denial was counted as the terminal policy observation.
    let decision_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id",
    )
    .bind(session_id)
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    let conditions: Vec<String> = decision_rows
        .iter()
        .filter_map(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .filter_map(|data| data["metadata"]["condition"].as_str().map(str::to_string))
        .collect();
    assert!(
        !conditions.iter().any(|c| c == "tools_required_no_tool_response"),
        "gate looped: {conditions:?}"
    );
    assert!(
        conditions.iter().any(|c| c == "negative_completion_contract"),
        "{conditions:?}"
    );
    assert!(
        conditions.iter().any(|c| c == "policy_denial_terminal_observation"),
        "denial must be accepted as the terminal observation: {conditions:?}"
    );
    // The ledger-first arbiter (shadow by default) must agree: nothing is
    // reachable after the credited denial, so the run is closed on evidence.
    let closeout = decision_rows
        .iter()
        .filter_map(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .find(|data| data["metadata"]["condition"].as_str() == Some("ledger_closeout"))
        .expect("ledger closeout verdict must be recorded");
    assert_eq!(closeout["metadata"]["verdict"], "closed", "{closeout}");
    assert!(
        matches!(
            closeout["metadata"]["proof_basis"].as_str(),
            Some("contract") | Some("credited_denial")
        ),
        "{closeout}"
    );
    assert!(closeout["metadata"]["reachable_obligation_ids"]
        .as_array()
        .is_some_and(Vec::is_empty));
}

/// The executor's own `track_requirements` checklist is the expectations
/// lane. The assessor under-describes the request (no contract obligations at
/// all — the "all unknown" class seen live), the model declares three typed
/// items, performs one, and tries to stop. The ledger holds it to its own
/// declaration: the arbiter demands the open reachable items, the model does
/// them, and only then does the reply ship.
#[tokio::test]
async fn test_full_stack_executor_declared_checklist_is_demanded_until_receipts_close_it() {
    let workspace = tempfile::tempdir().expect("workspace");
    let cwd = workspace.path().to_string_lossy().to_string();
    let dir = workspace.path().join("declared").to_string_lossy().to_string();
    let file = workspace
        .path()
        .join("declared/note.txt")
        .to_string_lossy()
        .to_string();

    // Assessor: nothing typed at all — no mutation, no observation.
    let assessment = MockProvider::semantic_task_assessment(
        "check",
        false,
        false,
        &[],
        "new_request",
        "local_workspace",
    );

    let checklist_args = serde_json::json!({
        "items": [
            {"text": "create the directory", "status": "pending",
             "mutation_effects": ["local_workspace_write"], "targets": [dir]},
            {"text": "write the note", "status": "pending",
             "mutation_effects": ["local_workspace_write"], "targets": [file]},
            {"text": "read the note back", "status": "pending",
             "requires_observation": true, "targets": [file]}
        ]
    })
    .to_string();
    let mkdir_args = serde_json::json!({
        "command": format!("mkdir -p '{dir}'"),
        "working_dir": cwd,
        "write_paths": [dir],
    })
    .to_string();
    let write_args = serde_json::json!({
        "command": format!("printf SYNTHETIC_NOTE > '{file}'"),
        "working_dir": cwd,
        "write_paths": [file],
    })
    .to_string();
    let read_args = serde_json::json!({
        "command": format!("cat '{file}'"),
        "working_dir": cwd,
        "read_paths": [file],
    })
    .to_string();
    let early = "phase=EARLY; directory created, stopping here.";
    let done = "phase=DONE; directory, note, and read-back all completed.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", &checklist_args),
        MockProvider::tool_call_response("terminal", &mkdir_args),
        // The model stops after one of its three declared steps.
        MockProvider::text_response(early),
        // Demanded by its own declaration, it finishes the rest.
        MockProvider::tool_call_response("terminal", &write_args),
        MockProvider::tool_call_response("terminal", &read_args),
        MockProvider::text_response(done),
        MockProvider::text_response(done),
    ])
    .with_task_assessments(vec![assessment]);

    let plan_pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let plan_store = Arc::new(crate::plans::PlanStore::new(plan_pool).await.unwrap());
    let track = Arc::new(crate::tools::track_requirements::TrackRequirementsTool::new(
        plan_store,
    ));
    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![track as Arc<dyn crate::traits::Tool>],
    )
    .await
    .unwrap();

    let session_id = "executor_declared_checklist";
    let response = harness
        .agent
        .handle_message(
            session_id,
            &format!("Create {dir}, write a note file inside it, then read it back to confirm."),
            None,
            UserRole::Owner,
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "telegram".to_string(),
                channel_name: None,
                channel_id: None,
                workspace_id: None,
                sender_name: Some("Alice".to_string()),
                sender_id: Some("telegram:synthetic-owner".to_string()),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                workspace_grant: None,
                trusted: false,
            },
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(&file).ok().as_deref(),
        Some("SYNTHETIC_NOTE"),
        "the declared write must have happened"
    );
    assert!(
        response.contains(done) && !response.contains(early),
        "the early stop must not ship: {response}"
    );
    let decision_rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id",
    )
    .bind(session_id)
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    let conditions: Vec<String> = decision_rows
        .iter()
        .filter_map(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .filter_map(|data| data["metadata"]["condition"].as_str().map(str::to_string))
        .collect();
    assert!(
        conditions.iter().any(|c| c == "ledger_expectations_required"),
        "the executor's open items must have been demanded: {conditions:?}"
    );
    let closeouts: Vec<serde_json::Value> = decision_rows
        .iter()
        .filter_map(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .filter(|data| data["metadata"]["condition"].as_str() == Some("ledger_closeout"))
        .collect();
    let last = closeouts.last().expect("ledger closeout recorded");
    assert_eq!(last["metadata"]["verdict"], "closed", "{last}");
    assert!(
        closeouts.iter().any(|c| c["metadata"]["reachable_obligation_ids"]
            .as_array()
            .is_some_and(|ids| ids
                .iter()
                .any(|id| id.as_str().is_some_and(|id| id.contains("obligation:checklist:"))))),
        "checklist obligations must have been the reachable expectations: {closeouts:?}"
    );
}
