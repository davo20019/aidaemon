use crate::traits::AgentRole;

fn extract_tool_names(defs: &[serde_json::Value]) -> Vec<String> {
    defs.iter()
        .filter_map(|def| def.get("function"))
        .filter_map(|f| f.get("name"))
        .filter_map(|n| n.as_str())
        .map(ToString::to_string)
        .collect()
}

struct HighImpactCliAgentMock;

#[async_trait::async_trait]
impl crate::traits::Tool for HighImpactCliAgentMock {
    fn name(&self) -> &str {
        "cli_agent"
    }

    fn description(&self) -> &str {
        "high-impact delegation tool for policy-filter regression tests"
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "cli_agent",
            "description": "high-impact delegation tool for policy-filter regression tests",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": { "type": "string" }
                },
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        Ok("cli agent executed".to_string())
    }

    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }
}

#[tokio::test]
async fn test_delegation_executor_hides_competing_tools_when_cli_agent_available() {
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("executor finished")]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("browser", "browser tool", "ok")) as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("run_command", "run command tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
    ];
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, extra_tools)
        .await
        .unwrap();
    let provider_log = harness.provider.clone();
    let agent = Arc::new(harness.agent);

    let response = agent
        .spawn_child(
            "executor test",
            "inspect workspace",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "executor finished");

    let calls = provider_log.call_log.lock().await;
    assert!(!calls.is_empty(), "Expected at least one LLM call");
    let names = extract_tool_names(&calls.last().unwrap().tools);

    assert!(names.contains(&"cli_agent".to_string()));
    assert!(!names.contains(&"terminal".to_string()));
    assert!(!names.contains(&"browser".to_string()));
    assert!(!names.contains(&"run_command".to_string()));
}

#[tokio::test]
async fn test_delegation_executor_keeps_competing_tools_when_cli_agent_unavailable() {
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("executor finished")]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "ok").with_availability(false))
            as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("browser", "browser tool", "ok")) as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("run_command", "run command tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
    ];
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, extra_tools)
        .await
        .unwrap();
    let provider_log = harness.provider.clone();
    let agent = Arc::new(harness.agent);

    let response = agent
        .spawn_child(
            "executor test",
            "inspect workspace",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "executor finished");

    let calls = provider_log.call_log.lock().await;
    assert!(!calls.is_empty(), "Expected at least one LLM call");
    let names = extract_tool_names(&calls.last().unwrap().tools);

    assert!(!names.contains(&"cli_agent".to_string()));
    assert!(names.contains(&"terminal".to_string()));
    assert!(names.contains(&"browser".to_string()));
    assert!(names.contains(&"run_command".to_string()));
}

#[tokio::test]
async fn test_spawn_child_task_lead_scopes_tools_via_shared_builder() {
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("task lead finished")]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("browser", "browser tool", "ok")) as Arc<dyn crate::traits::Tool>,
    ];
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, extra_tools)
        .await
        .unwrap();
    let provider_log = harness.provider.clone();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("audit workspace", "task_lead_test_session");
    harness.state.create_goal(&goal).await.unwrap();

    let response = agent
        .spawn_child(
            "goal orchestration",
            "audit workspace",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::TaskLead),
            Some(goal.id.as_str()),
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "task lead finished");

    let calls = provider_log.call_log.lock().await;
    assert!(!calls.is_empty(), "Expected at least one LLM call");
    let names = extract_tool_names(&calls.last().unwrap().tools);

    assert!(
        names.contains(&"manage_goal_tasks".to_string()),
        "tool names: {:?}",
        names
    );
    assert!(
        names.contains(&"cli_agent".to_string()),
        "tool names: {:?}",
        names
    );
    // TaskLeads now include essential Action tools as fallback for when delegation fails.
    assert!(
        names.contains(&"terminal".to_string()),
        "TaskLead should have terminal as fallback: {:?}",
        names
    );
    assert!(
        !names.contains(&"browser".to_string()),
        "browser should NOT be in essential action tools: {:?}",
        names
    );
}

#[tokio::test]
async fn test_advisory_policy_filter_keeps_registered_tool_visible() {
    let provider = MockProvider::with_responses(vec![
        ProviderResponse {
            content: None,
            tool_calls: vec![crate::traits::ToolCall {
                id: "call_hidden_cli_agent".to_string(),
                name: "cli_agent".to_string(),
                arguments: r#"{"task":"inspect workspace"}"#.to_string(),
                extra_content: None,
            }],
            usage: None,
            thinking: None,
            response_note: None,
        },
        MockProvider::text_response("Used the currently exposed tools instead."),
    ]);
    let extra_tools = vec![Arc::new(HighImpactCliAgentMock) as Arc<dyn crate::traits::Tool>];
    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        extra_tools,
        None,
    )
    .await
    .unwrap();

    let response = harness
        .agent
        .handle_message(
            "hidden_tool_guess_test",
            "check the system specs on this machine",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "Used the currently exposed tools instead.");

    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls.len() >= 2,
        "expected at least two LLM calls, got {}",
        calls.len()
    );

    let first_call_tool_names = extract_tool_names(&calls[0].tools);
    assert!(
        first_call_tool_names.contains(&"cli_agent".to_string()),
        "an advisory policy score must not hide a registered tool, got {:?}",
        first_call_tool_names
    );

    let second_call_has_tool_result = calls[1].messages.iter().any(|message| {
        message.get("role").and_then(|role| role.as_str()) == Some("tool")
            && message.get("name").and_then(|name| name.as_str()) == Some("cli_agent")
            && message
                .get("content")
                .and_then(|content| content.as_str())
                .is_some_and(|content| content.contains("cli agent executed"))
    });
    assert!(
        second_call_has_tool_result,
        "second call messages did not contain the execution result: {:?}",
        calls[1].messages
    );
}

#[tokio::test]
async fn test_executor_spawn_persists_structured_handoff_and_result_on_task() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "Updated /tmp/demo/src/main.rs and reran the scoped checks successfully.",
    )]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Patch the regression", "delegation-task-context");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-structured-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Patch /tmp/demo/src/main.rs".to_string(),
        status: "claimed".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&task).await.unwrap();

    let response = agent
        .spawn_child(
            "Patch the scoped regression in /tmp/demo",
            "Patch /tmp/demo/src/main.rs",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            Some(goal.id.as_str()),
            Some(task.id.as_str()),
            Some("/tmp/demo"),
            None,
        )
        .await
        .unwrap();

    assert!(response.contains("Updated /tmp/demo/src/main.rs"));

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "completed");
    let context: serde_json::Value = serde_json::from_str(updated.context.as_deref().unwrap())
        .expect("task context should be json");
    assert_eq!(
        context["executor_handoff"]["task_id"].as_str(),
        Some(task.id.as_str())
    );
    assert_eq!(
        context["executor_result"]["task_outcome"].as_str(),
        Some("task_done")
    );
    assert_eq!(
        context["executor_handoff"]["target_scope"]["allowed_targets"][0]["value"].as_str(),
        Some("/tmp/demo")
    );
}

#[tokio::test]
async fn test_report_blocker_terminates_executor_loop() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "report_blocker",
            r#"{"reason":"Docker daemon is not reachable","outcome":"blocked","exact_need":"Start Docker so ddev can connect.","next_step":"Re-run ddev composer update once Docker is up."}"#,
        ),
        MockProvider::text_response("SHOULD NOT BE REACHED - loop must end at report_blocker."),
    ]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Update Drupal modules", "delegation-blocker-terminal");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-blocker-terminal-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Run ddev composer update".to_string(),
        status: "claimed".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&task).await.unwrap();

    let response = agent
        .spawn_child(
            "Run the ddev composer update for the Drupal site",
            "Run ddev composer update",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            Some(goal.id.as_str()),
            Some(task.id.as_str()),
            Some("/tmp/demo"),
            None,
        )
        .await
        .unwrap();

    // The executor's final response is the structured blocker summary,
    // not a follow-up LLM reply.
    assert!(
        response.contains("Executor outcome: blocked"),
        "expected blocker summary as final response, got: {response}"
    );
    assert!(
        !response.contains("SHOULD NOT BE REACHED"),
        "loop continued past report_blocker"
    );

    // Exactly one LLM call: the one that emitted report_blocker. No
    // reconciliation/verification iterations after a declared blocker.
    assert_eq!(
        harness.provider.call_count().await,
        1,
        "report_blocker must terminate the executor loop without further LLM calls"
    );

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "blocked");
}

#[tokio::test]
async fn test_spawn_timeout_salvages_persisted_executor_outcome() {
    // Child future will be cancelled instantly (timeout_secs = 0), simulating
    // the parent timing out while the executor had already persisted a
    // terminal outcome via report_blocker.
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "child reply that the timeout will discard",
    )]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Update Drupal modules", "delegation-timeout-salvage");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-timeout-salvage-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Run ddev composer update".to_string(),
        status: "blocked".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: Some(
            "Executor outcome: blocked\nSummary: Docker daemon is not reachable.".to_string(),
        ),
        error: None,
        blocker: Some("BLOCKED: Docker daemon is not reachable".to_string()),
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    harness.state.create_task(&task).await.unwrap();

    let spawn_tool = crate::tools::spawn::SpawnAgentTool::new(Arc::downgrade(&agent), 4000, 0);
    let result = crate::traits::Tool::call(
        &spawn_tool,
        &json!({
            "mission": "Run the ddev composer update",
            "task": "Run ddev composer update",
            "task_id": task.id,
        })
        .to_string(),
    )
    .await
    .unwrap();

    assert!(
        result.contains("Docker daemon is not reachable"),
        "timeout must salvage the persisted executor outcome, got: {result}"
    );
    assert!(
        !result.contains("timed out after"),
        "salvaged outcome must replace the generic timeout error, got: {result}"
    );

    // The persisted terminal status must survive the timeout handling.
    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "blocked");
}

#[tokio::test]
async fn test_background_spawn_timeout_salvages_persisted_executor_outcome() {
    use crate::traits::NotificationStore;

    let provider = MockProvider::new();
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Update Drupal modules", "delegation-bg-salvage");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-bg-salvage-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Run ddev composer update".to_string(),
        status: "blocked".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: Some(
            "Executor outcome: blocked\nSummary: Docker daemon is not reachable.".to_string(),
        ),
        error: None,
        blocker: Some("BLOCKED: Docker daemon is not reachable".to_string()),
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    harness.state.create_task(&task).await.unwrap();

    let spawn_tool = crate::tools::spawn::SpawnAgentTool::new(Arc::downgrade(&agent), 4000, 0)
        .with_state(harness.state.clone() as Arc<dyn crate::traits::StateStore>);
    let ack = crate::traits::Tool::call(
        &spawn_tool,
        &json!({
            "mission": "Run the ddev composer update",
            "task": "Run ddev composer update",
            "task_id": task.id,
            "background": true,
            "_session_id": "delegation-bg-salvage",
            "_goal_id": goal.id,
        })
        .to_string(),
    )
    .await
    .unwrap();
    assert!(ack.contains("background"));

    let mut completion_message = None;
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let pending = harness.state.get_pending_notifications(20).await.unwrap();
        if let Some(entry) = pending
            .iter()
            .find(|n| n.message.contains("Mission: Run the ddev composer update"))
        {
            completion_message = Some(entry.message.clone());
            break;
        }
    }
    let message = completion_message.expect("background completion notification should be queued");
    assert!(
        message.contains("Docker daemon is not reachable"),
        "background timeout must salvage the persisted executor outcome, got: {message}"
    );
    assert!(
        !message.contains("timed out"),
        "salvaged outcome must replace the generic timeout message, got: {message}"
    );
}

#[tokio::test]
async fn test_executor_timeout_does_not_clobber_terminal_task_status() {
    let provider = MockProvider::new();
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Update Drupal modules", "delegation-timeout-clobber");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-timeout-clobber-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Run ddev composer update".to_string(),
        status: "blocked".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: Some(
            "Executor outcome: blocked\nSummary: Docker daemon is not reachable.".to_string(),
        ),
        error: None,
        blocker: Some("BLOCKED: Docker daemon is not reachable".to_string()),
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    harness.state.create_task(&task).await.unwrap();

    agent.mark_executor_task_timeout(&task.id, 300).await;

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(
        updated.status, "blocked",
        "timeout finalization must not overwrite an already-terminal task status"
    );
    assert!(
        updated
            .blocker
            .as_deref()
            .unwrap_or_default()
            .contains("Docker daemon is not reachable"),
        "persisted blocker must be preserved"
    );
}

#[tokio::test]
async fn test_executor_spawn_persists_needs_approval_blocker_result() {
    let provider = MockProvider::with_responses(vec![MockProvider::tool_call_response(
        "report_blocker",
        r#"{"reason":"Need approval to rotate the production credentials","outcome":"needs_approval","partial_work":"Validated the rotation script and staged the rollout notes","exact_need":"Owner approval to rotate the production credentials.","next_step":"Run the approved credential rotation and verify the service health.","target":"production credentials"}"#,
    )]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Rotate production credentials", "delegation-approval");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-approval-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Rotate the production credentials".to_string(),
        status: "claimed".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&task).await.unwrap();

    let response = agent
        .spawn_child(
            "Rotate the production credentials safely",
            "Rotate the production credentials",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            Some(goal.id.as_str()),
            Some(task.id.as_str()),
            Some("/tmp/demo"),
            None,
        )
        .await
        .unwrap();

    // report_blocker is terminal: the structured summary is the final response.
    assert!(response.contains("Executor outcome: needs_approval"));

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "blocked");
    assert!(updated
        .result
        .as_deref()
        .unwrap_or_default()
        .contains("Executor outcome: needs_approval"));
    let context: serde_json::Value = serde_json::from_str(updated.context.as_deref().unwrap())
        .expect("task context should be json");
    assert_eq!(
        context["executor_result"]["task_outcome"].as_str(),
        Some("needs_approval")
    );
}
