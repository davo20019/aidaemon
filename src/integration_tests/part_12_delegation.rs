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
    assert!(
        !names.contains(&"terminal".to_string()),
        "tool names: {:?}",
        names
    );
    assert!(
        !names.contains(&"browser".to_string()),
        "tool names: {:?}",
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
            "what time is it?",
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

    let second_call_has_hidden_tool_block = calls[1].messages.iter().any(|message| {
        message.get("role").and_then(|role| role.as_str()) == Some("tool")
            && message.get("name").and_then(|name| name.as_str()) == Some("cli_agent")
            && message
                .get("content")
                .and_then(|content| content.as_str())
                .is_some_and(|content| {
                    content.contains("not available in your current tool list")
                        && content.contains("Do NOT guess or force hidden tool names")
                })
    });
    assert!(
        second_call_has_hidden_tool_block,
        "second call messages did not contain the hidden-tool block notice: {:?}",
        calls[1].messages
    );
}
