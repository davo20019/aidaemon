use crate::traits::AgentRole;

fn extract_tool_names(defs: &[serde_json::Value]) -> Vec<String> {
    defs.iter()
        .filter_map(|def| def.get("function"))
        .filter_map(|f| f.get("name"))
        .filter_map(|n| n.as_str())
        .map(ToString::to_string)
        .collect()
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
