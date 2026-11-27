//! Discovering, adding and removing agents.

use super::*;

#[tokio::test]
async fn test_add_agent_with_echo() {
    let (tool, _db) = setup_echo_tool().await;

    // `echo` exists on all systems
    let result = tool
        .add_agent("my-echo", "echo", vec![], "Custom echo", None, None)
        .await
        .unwrap();
    assert!(
        result.contains("added successfully"),
        "Expected success, got: {}",
        result
    );

    // Should be in the tools map
    let agents = tool.list_agents();
    assert!(
        agents.iter().any(|(name, _, _, _)| name == "my-echo"),
        "Expected my-echo in agent list"
    );
}

#[tokio::test]
async fn test_add_agent_nonexistent_command() {
    let (tool, _db) = setup_echo_tool().await;

    let result = tool
        .add_agent(
            "fake",
            "aidaemon-nonexistent-cmd-xyz",
            vec![],
            "",
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        result.contains("not found"),
        "Expected not found error, got: {}",
        result
    );
}

#[tokio::test]
async fn test_remove_agent() {
    let (tool, _db) = setup_echo_tool().await;

    // Add then remove
    tool.add_agent("removeme", "echo", vec![], "", None, None)
        .await
        .unwrap();
    let result = tool.remove_agent("removeme").await.unwrap();
    assert!(result.contains("removed"));

    let agents = tool.list_agents();
    assert!(
        !agents.iter().any(|(name, _, _, _)| name == "removeme"),
        "Agent should have been removed"
    );
}

#[tokio::test]
async fn test_remove_nonexistent_agent() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool.remove_agent("nonexistent").await.unwrap();
    assert!(result.contains("not found"));
}

// -----------------------------------------------------------------------
// Auth error detection tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_discover_finds_echo() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(TestStore::open(db_file.path().to_str().unwrap()).await);

    // Config with echo as a custom tool
    let mut tools = HashMap::new();
    tools.insert(
        "echo".to_string(),
        CliToolConfig {
            command: "echo".to_string(),
            args: vec![],
            description: "Echo for test".to_string(),
            timeout_secs: None,
            max_output_chars: None,
            protocol: None,
        },
    );
    let config = CliAgentsConfig {
        enabled: true,
        timeout_secs: 30,
        max_output_chars: 10000,
        tools,
        permissions: HashMap::new(),
        ..CliAgentsConfig::default()
    };

    let tool = CliAgentTool::discover(
        aidaemon_tools::testing::active_execution_backend(),
        config,
        state as Arc<dyn CliAgentStore>,
    )
    .await;
    assert!(tool.has_tools());

    let agents = tool.list_agents();
    assert!(agents.iter().any(|(name, _, _, _)| name == "echo"));
}

#[tokio::test]
async fn test_discover_skips_nonexistent() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(TestStore::open(db_file.path().to_str().unwrap()).await);

    let mut tools = HashMap::new();
    tools.insert(
        "fake-tool".to_string(),
        CliToolConfig {
            command: "aidaemon-nonexistent-12345".to_string(),
            args: vec![],
            description: "".to_string(),
            timeout_secs: None,
            max_output_chars: None,
            protocol: None,
        },
    );
    let config = CliAgentsConfig {
        enabled: true,
        timeout_secs: 30,
        max_output_chars: 10000,
        tools,
        permissions: HashMap::new(),
        ..CliAgentsConfig::default()
    };

    let tool = CliAgentTool::discover(
        aidaemon_tools::testing::active_execution_backend(),
        config,
        state as Arc<dyn CliAgentStore>,
    )
    .await;
    assert!(!tool.has_tools());
}

// -----------------------------------------------------------------------
// Invocation logging integration test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_has_tools() {
    let (tool, _db) = setup_echo_tool().await;
    assert!(tool.has_tools());
    assert!(ToolSemantics::is_available(&tool));

    // Clear all tools
    tool.tools.write().unwrap().clear();
    assert!(!tool.has_tools());
    assert!(!ToolSemantics::is_available(&tool));
}

#[tokio::test]
async fn test_run_without_tool_and_no_agents_errors() {
    let (tool, _db) = setup_echo_tool().await;
    tool.tools.write().unwrap().clear();
    tool.tool_names.write().unwrap().clear();

    let result = tool.call(r#"{"action":"run","prompt":"test"}"#).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("No CLI agents available"),
        "Expected no-agents error, got: {}",
        err
    );
}

// -----------------------------------------------------------------------
// Enriched prompt test
// -----------------------------------------------------------------------

/// Antigravity on its own (consumer) sign-in: Google's terms bar another
/// agent handing it work. The automatic pick passes it over, and a run
/// that names it is refused before launch; the owner's own prompts in Code
/// mode (`direct_agent`) still reach it.
#[tokio::test]
async fn agy_on_its_own_sign_in_is_never_handed_work_but_still_runs_directly() {
    let (tool, _db) = setup_bash_tool().await;
    tool.tools.write().unwrap().insert(
        "agy".to_string(),
        CliToolEntry {
            command: "agy".to_string(),
            args: vec!["acp".to_string()],
            description: "Antigravity CLI".to_string(),
            timeout: Duration::from_secs(10),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );
    // Ranked above an unknown agent, yet passed over.
    assert!(CliAgentTool::tool_priority("agy") < CliAgentTool::tool_priority("bash-agent"));
    assert_eq!(tool.default_tool_name().as_deref(), Some("bash-agent"));

    let prepare = |direct_agent: bool| {
        let tool = &tool;
        async move {
            tool.prepare_run(
                "agy",
                "agyrun12",
                "fix the build",
                "fix the build",
                None,
                "synthetic-session",
                None,
                None,
                CliWorkspaceMode::ReadWrite,
                &[],
                &[],
                false,
                &crate::helper_run::HelperSpec::default(),
                None,
                None,
                None,
                None,
                false,
                false,
                direct_agent,
                &[],
                &[],
                None,
            )
            .await
            .unwrap()
        }
    };
    let Err(message) = prepare(false).await else {
        panic!("delegated work must not reach agy on its own sign-in");
    };
    assert!(message.contains("can't be handed work"), "{message}");
    if let Err(message) = prepare(true).await {
        assert!(!message.contains("can't be handed work"), "{message}");
    }
}
