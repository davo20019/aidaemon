// ==========================================================================
// Explicit authority
//
// Tool authority comes from explicit facts (role, workspace grant, a goal's
// bound workspace, the owner's level and session mode), never from the
// request assessment.
// ==========================================================================

/// A fresh synthetic directory under `/tmp/aidaemon-test`.
fn authority_test_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from("/tmp/aidaemon-test").join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every tool result the model saw, across all model calls.
async fn authority_tool_results(harness: &crate::testing::TestHarness) -> Vec<String> {
    let log = harness.provider.call_log.lock().await;
    let mut results = Vec::new();
    for call in log.iter() {
        for message in call.messages.iter().filter(|m| m["role"] == "tool") {
            if let Some(content) = message["content"].as_str() {
                if !results.iter().any(|seen| seen == content) {
                    results.push(content.to_string());
                }
            }
        }
    }
    results
}

#[tokio::test]
async fn assessed_read_only_contract_no_longer_blocks_owner_writes() {
    let dir = authority_test_dir();
    let target = dir.join("a.txt");
    let base =
        MockProvider::semantic_task_assessment("action", true, false, &[], "new_request", "none");
    let mut assessment: serde_json::Value =
        serde_json::from_str(base.content.as_deref().unwrap()).unwrap();
    assessment["contract"]["mutation_scope"] = json!("read_only");
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": target, "content": "hello"}).to_string(),
        ),
        MockProvider::text_response("Wrote the file."),
    ])
    .with_task_assessments(vec![MockProvider::text_response(&assessment.to_string())]);
    let tool: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::WriteFileTool);
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(provider, vec![tool], None)
        .await
        .unwrap();

    harness
        .agent
        .handle_message(
            "tg_authority_owner_write",
            &format!("Write hello to {}", target.display()),
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    let results = authority_tool_results(&harness).await;
    assert!(
        target.is_file(),
        "the owner's write must run despite a read-only assessment; the model saw {results:?}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn goal_run_writes_confined_reads_free() {
    let dir = authority_test_dir();
    let workspace = dir.join("ws");
    std::fs::create_dir_all(&workspace).unwrap();
    let inside = workspace.join("a.txt");
    let outside = dir.join("outside.txt");
    let fixture = dir.join("fixture.txt");
    std::fs::write(&fixture, "synthetic fixture marker").unwrap();

    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": inside, "content": "inside"}).to_string(),
        ),
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": outside, "content": "outside"}).to_string(),
        ),
        MockProvider::tool_call_response("read_file", &json!({"path": fixture}).to_string()),
        MockProvider::text_response("Goal step done."),
    ]);
    let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
        Arc::new(crate::tools::WriteFileTool),
        Arc::new(crate::tools::ReadFileTool),
    ];
    let mut harness = setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, tools, None)
        .await
        .unwrap();
    let mut goal = Goal::new_finite("Maintain the synthetic workspace", "goal_authority_session");
    goal.status = "active".to_string();
    goal.context = Some(
        json!({
            "project_scope": workspace.to_string_lossy(),
            "project_scope_bound_at": "2026-11-03T00:00:00Z",
        })
        .to_string(),
    );
    harness.state.create_goal(&goal).await.unwrap();
    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    harness
        .agent
        .handle_message(
            "goal_authority_session",
            "Run the next step of the goal.",
            None,
            UserRole::Owner,
            ChannelContext::internal(),
            None,
        )
        .await
        .unwrap();

    let results = authority_tool_results(&harness).await;
    assert!(inside.is_file(), "a write inside the workspace runs; saw {results:?}");
    assert!(
        !outside.exists(),
        "a write outside the workspace is refused; saw {results:?}"
    );
    assert!(
        results
            .iter()
            .any(|result| result.contains("synthetic fixture marker")),
        "a read outside the workspace runs; saw {results:?}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn guest_grants_unchanged() {
    let dir = authority_test_dir();
    let root = std::fs::canonicalize(&dir).unwrap();
    std::fs::write(root.join("package.json"), r#"{"name":"synthetic-project"}"#).unwrap();
    let grant = crate::types::WorkspaceGrant {
        platform: "telegram".to_string(),
        workspace_id: "synthetic-workspace".to_string(),
        channel_id: "synthetic-group".to_string(),
        user_id: "synthetic-user-1".to_string(),
        project_root: root.to_string_lossy().into_owned(),
        access: crate::types::WorkspaceAccessLevel::Edit,
        expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
    };
    let group_ctx = |grant: Option<crate::types::WorkspaceGrant>| {
        let mut ctx = ChannelContext::private("telegram");
        ctx.visibility = ChannelVisibility::PrivateGroup;
        ctx.channel_id = Some("telegram:synthetic-group".to_string());
        ctx.workspace_id = Some("synthetic-workspace".to_string());
        ctx.sender_id = Some("telegram:synthetic-user-1".to_string());
        ctx.workspace_grant = grant;
        ctx
    };

    // With an Edit grant the guest writes inside its root.
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": root.join("notes.txt"), "content": "guest"}).to_string(),
        ),
        MockProvider::text_response("Saved."),
    ]);
    let tool: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::WriteFileTool);
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(provider, vec![tool], None)
        .await
        .unwrap();
    harness
        .agent
        .handle_message(
            "tg_authority_guest_grant",
            "Save a note in the project.",
            None,
            UserRole::Guest,
            group_ctx(Some(grant)),
            None,
        )
        .await
        .unwrap();
    let results = authority_tool_results(&harness).await;
    assert!(
        root.join("notes.txt").is_file(),
        "an Edit grant still writes inside its root; saw {results:?}"
    );

    // Without a grant the guest has no tools at all, so nothing is written.
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": root.join("other.txt"), "content": "guest"}).to_string(),
        ),
        MockProvider::text_response("Could not save."),
    ]);
    let tool: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::WriteFileTool);
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(provider, vec![tool], None)
        .await
        .unwrap();
    harness
        .agent
        .handle_message(
            "tg_authority_guest_no_grant",
            "Save a note in the project.",
            None,
            UserRole::Guest,
            group_ctx(None),
            None,
        )
        .await
        .unwrap();
    assert!(
        harness.provider.call_log.lock().await[0].tools.is_empty(),
        "a guest without a grant is offered no tools"
    );
    assert!(!root.join("other.txt").exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn mandate_allowed_tools_still_restrict() {
    let dir = authority_test_dir();
    let target = dir.join("mandate.txt");
    let create_args = json!({
        "action": "create",
        "objective": "Steward a synthetic account as an authentic presence",
        "operation_scopes": [{
            "tool": "http_request",
            "operation": "POST",
            "kind": "mutation",
            "target_prefixes": ["https://api.example.test/posts"],
            "mutation_effects": ["remote_mutation", "external_delivery"]
        }],
        "max_mutating_actions_per_cycle": 1,
        "max_mutating_actions_per_rolling_24h": 4,
        "min_seconds_between_mutations": 900,
        "constraints": ["No fabrication"],
        "success_criteria": ["Each post is grounded and useful"],
        "stop_conditions": ["Authentication is revoked"]
    })
    .to_string();
    let draft_args = create_args.replace("\"create\"", "\"draft\"");
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("manage_mandates", &draft_args),
        MockProvider::tool_call_response("manage_mandates", &create_args),
        MockProvider::text_response("The mandate is active."),
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": target, "content": "outside the mandate"}).to_string(),
        ),
        MockProvider::text_response("That is outside the mandate."),
    ]);
    let mut harness = setup_test_agent_with_mandates(provider).await.unwrap();
    harness
        .agent
        .handle_message(
            "owner-chat",
            "Manage my synthetic account as an ongoing presence; decide when to post or wait.",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    let mandate =
        retry_test_sqlite_busy(|| harness.state.claim_due_mandates(1, "test-heartbeat", 300))
            .await
            .unwrap()
            .into_iter()
            .next()
            .expect("the mandate is due");
    assert_eq!(mandate.authority.allowed_tools, ["http_request"]);
    let root_task_id = uuid::Uuid::new_v4().to_string();
    let root_task = crate::traits::Task {
        id: root_task_id.clone(),
        goal_id: mandate.goal_id.clone(),
        description: "Review the mandate".to_string(),
        status: "pending".to_string(),
        priority: "high".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 0,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    let goal_run_id = uuid::Uuid::new_v4().to_string();
    retry_test_sqlite_busy(|| {
        harness.state.create_mandate_review_run(
            &mandate.id,
            mandate.review_lease_token.as_deref().unwrap(),
            &goal_run_id,
            &root_task,
        )
    })
    .await
    .unwrap();
    let root_attempt = retry_test_sqlite_busy(|| {
        harness
            .state
            .claim_task_with_lease(&root_task_id, "test-task-lead", Some("profile-task-lead"), 7_200)
    })
    .await
    .unwrap()
    .unwrap();
    harness
        .agent
        .push_test_tool(Arc::new(crate::tools::WriteFileTool));
    harness.agent.set_test_mandate_execution(
        &mandate.id,
        mandate.version,
        mandate.authority.clone(),
        &mandate.goal_id,
        &root_task_id,
        &root_attempt.id,
        &root_attempt,
    );
    // The review ends without a committed decision; only the refusal matters.
    let _ = harness
        .agent
        .handle_message(
            "mandate-review",
            &format!("Review the mandate. Write a note to {}.", target.display()),
            None,
            UserRole::Owner,
            ChannelContext::internal(),
            None,
        )
        .await;

    let results = authority_tool_results(&harness).await;
    assert!(
        !target.exists(),
        "the mandate's allowed tools still restrict its worker; saw {results:?}"
    );
    assert!(
        results
            .iter()
            .any(|result| result.contains("[MANDATE AUTHORITY BLOCKED]")),
        "the refusal comes from the mandate gate; saw {results:?}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn readonly_blocks_writes_then_full_restores() {
    let dir = authority_test_dir();
    let blocked = dir.join("blocked.txt");
    let restored = dir.join("restored.txt");
    std::fs::write(dir.join("listing_marker.txt"), "x").unwrap();
    let session = "tg_authority_readonly";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": blocked, "content": "no"}).to_string(),
        ),
        MockProvider::text_response("I could not write the file."),
        MockProvider::tool_call_response(
            "terminal",
            &json!({"command": format!("ls {}", dir.display())}).to_string(),
        ),
        MockProvider::text_response("Listed the directory."),
        MockProvider::tool_call_response(
            "write_file",
            &json!({"path": restored, "content": "yes"}).to_string(),
        ),
        MockProvider::text_response("Wrote the file."),
    ]);
    let tool: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::WriteFileTool);
    let harness = crate::testing::setup_full_stack_test_agent_with_extra_tools(provider, vec![tool])
        .await
        .unwrap();
    let state = Arc::clone(&harness.state);
    let provider = Arc::clone(&harness.provider);
    let agent = Arc::new(harness.agent);
    let commands = crate::channels::commands::CommandContext {
        agent: agent.clone(),
        state: state.clone(),
        task_registry: Arc::new(crate::tasks::TaskRegistry::new(10)),
        config_path: std::path::PathBuf::new(),
    };
    let turn = |text: &'static str| {
        let agent = agent.clone();
        async move {
            agent
                .handle_message(
                    session,
                    text,
                    None,
                    UserRole::Owner,
                    ChannelContext::private("telegram"),
                    None,
                )
                .await
                .unwrap();
        }
    };

    let reply = commands
        .dispatch("/readonly", "", session, UserRole::Owner)
        .await
        .unwrap();
    assert!(reply.starts_with("Read-only for this chat until "), "{reply}");

    turn("Write a file.").await;
    assert!(!blocked.exists(), "a write is refused in read-only mode");
    turn("List the directory.").await;

    let results = {
        let log = provider.call_log.lock().await;
        let mut seen = Vec::new();
        for call in log.iter() {
            for message in call.messages.iter().filter(|m| m["role"] == "tool") {
                if let Some(content) = message["content"].as_str() {
                    if !seen.iter().any(|s: &String| s == content) {
                        seen.push(content.to_string());
                    }
                }
            }
        }
        seen
    };
    assert!(
        results.iter().any(|r| r.contains("[AUTHORITY BLOCKED]")),
        "the write was refused by the authority gate; saw {results:?}"
    );
    assert!(
        results.iter().any(|r| r.contains("listing_marker.txt")),
        "terminal ls ran; saw {results:?}"
    );
    assert!(
        !results
            .iter()
            .any(|r| r.contains("[AUTHORITY BLOCKED]") && r.contains("terminal")),
        "terminal ls is not refused; saw {results:?}"
    );

    // The denial is part of the run's record: a blocked, pre-dispatch tool
    // result for the write.
    let events = agent
        .event_store()
        .query_events_by_types(session, &[crate::events::EventType::ToolResult], 20)
        .await
        .unwrap();
    assert!(
        events.iter().any(|event| event.data["name"] == "write_file"
            && event.data["result"]
                .as_str()
                .is_some_and(|r| r.starts_with("[AUTHORITY BLOCKED]"))
            && event.data["success"] == false),
        "the authority denial is recorded in the run; saw {:?}",
        events.iter().map(|e| e.data.clone()).collect::<Vec<_>>()
    );

    assert_eq!(
        commands
            .dispatch("/mode", "full", session, UserRole::Owner)
            .await
            .as_deref(),
        Some("Back to full access.")
    );
    turn("Write a file again.").await;
    assert!(restored.is_file(), "a write runs again after /mode full");
    let _ = std::fs::remove_dir_all(dir);
}
