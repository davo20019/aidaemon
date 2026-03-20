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
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "executor finished",
    )]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "ok")) as Arc<dyn crate::traits::Tool>,
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
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "executor finished",
    )]);
    let extra_tools = vec![
        Arc::new(
            MockTool::new("cli_agent", "delegation tool", "ok").with_availability(false),
        ) as Arc<dyn crate::traits::Tool>,
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
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "task lead finished",
    )]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("browser", "browser tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
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
async fn test_hidden_tool_guess_is_blocked_when_not_in_current_tool_defs() {
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
        !first_call_tool_names.contains(&"cli_agent".to_string()),
        "cli_agent should be hidden by policy filter, got {:?}",
        first_call_tool_names
    );

    // The hidden cli_agent call is intercepted before tool execution:
    // either by the text-only prelude check (plain-text redirect for
    // non-mutation turns) or by the hidden-tool guard. Both inject a
    // tool-role message for cli_agent that prevents execution.
    let second_call_has_tool_block = calls[1].messages.iter().any(|message| {
        message.get("role").and_then(|role| role.as_str()) == Some("tool")
            && message.get("name").and_then(|name| name.as_str()) == Some("cli_agent")
            && message
                .get("content")
                .and_then(|content| content.as_str())
                .is_some_and(|content| {
                    content.contains("not available in your current tool list")
                        || content.contains("should be answered directly in plain text")
                })
    });
    assert!(
        second_call_has_tool_block,
        "second call messages did not contain a tool block notice: {:?}",
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
        )
        .await
        .unwrap();

    assert!(response.contains("Updated /tmp/demo/src/main.rs"));

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "completed");
    let context: serde_json::Value =
        serde_json::from_str(updated.context.as_deref().unwrap()).expect("task context should be json");
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
async fn test_executor_spawn_persists_needs_approval_blocker_result() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "report_blocker",
            r#"{"reason":"Need approval to rotate the production credentials","outcome":"needs_approval","partial_work":"Validated the rotation script and staged the rollout notes","exact_need":"Owner approval to rotate the production credentials.","next_step":"Run the approved credential rotation and verify the service health.","target":"production credentials"}"#,
        ),
        MockProvider::text_response("Stopping after reporting the approval blocker."),
    ]);
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
        )
        .await
        .unwrap();

    assert!(response.contains("Stopping after reporting"));

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "blocked");
    assert!(updated
        .result
        .as_deref()
        .unwrap_or_default()
        .contains("Executor outcome: needs_approval"));
    let context: serde_json::Value =
        serde_json::from_str(updated.context.as_deref().unwrap()).expect("task context should be json");
    assert_eq!(
        context["executor_result"]["task_outcome"].as_str(),
        Some("needs_approval")
    );
}
