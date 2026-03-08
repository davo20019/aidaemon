use super::*;
use crate::testing::{setup_test_agent, MockProvider};

struct SlowTool;

#[async_trait::async_trait]
impl Tool for SlowTool {
    fn name(&self) -> &str {
        "slow_tool"
    }

    fn description(&self) -> &str {
        "Sleeps before returning"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "slow_tool",
            "description": "Sleeps before returning",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        })
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        tokio::time::sleep(Duration::from_millis(150)).await;
        Ok("done".to_string())
    }
}

struct SlowCliAgentTool;

#[async_trait::async_trait]
impl Tool for SlowCliAgentTool {
    fn name(&self) -> &str {
        "cli_agent"
    }

    fn description(&self) -> &str {
        "Simulates a long-running cli_agent tool call"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "cli_agent",
            "description": "Simulates a long-running cli_agent tool call",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        })
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        tokio::time::sleep(Duration::from_millis(150)).await;
        Ok("ok".to_string())
    }
}

struct EchoSpawnAgentTool;

#[async_trait::async_trait]
impl Tool for EchoSpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Echoes enriched spawn_agent arguments for regression tests"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "spawn_agent",
            "description": "Echoes enriched spawn_agent arguments for regression tests",
            "parameters": {
                "type": "object",
                "properties": {
                    "mission": { "type": "string" },
                    "task": { "type": "string" }
                },
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(arguments.to_string())
    }
}

#[tokio::test]
async fn execute_tool_watchdog_times_out_slow_tool() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    harness.agent.tools.push(Arc::new(SlowTool));
    harness.agent.llm_call_timeout = Some(Duration::from_millis(30));

    let result = harness
        .agent
        .execute_tool_with_watchdog(
            "slow_tool",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-1"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
            },
        )
        .await;

    let err = result.expect_err("slow tool should time out");
    assert!(
        err.to_string().contains("timed out"),
        "timeout error expected, got: {}",
        err
    );
}

#[tokio::test]
async fn execute_tool_watchdog_skips_cli_agent() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    harness.agent.tools.push(Arc::new(SlowCliAgentTool));
    harness.agent.llm_call_timeout = Some(Duration::from_millis(30));

    let result = harness
        .agent
        .execute_tool_with_watchdog(
            "cli_agent",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-3"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
            },
        )
        .await
        .expect("cli_agent should bypass watchdog");

    assert_eq!(result, "ok");
}

#[tokio::test]
async fn execute_tool_watchdog_allows_fast_tool() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    // system_info runs multiple subprocesses; allow a bit of slack to avoid
    // flakiness on slower/loaded machines.
    harness.agent.llm_call_timeout = Some(Duration::from_secs(5));

    let result = harness
        .agent
        .execute_tool_with_watchdog(
            "system_info",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-2"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
            },
        )
        .await
        .expect("fast tool should succeed");

    assert!(
        !result.is_empty(),
        "system_info should return a non-empty payload"
    );
}

#[tokio::test]
async fn execute_tool_watchdog_injects_project_scope_into_spawn_agent() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    harness.agent.tools.push(Arc::new(EchoSpawnAgentTool));

    let result = harness
        .agent
        .execute_tool_with_watchdog(
            "spawn_agent",
            r#"{"mission":"delegate log analysis","task":"inspect the logs","_project_scope":"/tmp/spoofed"}"#,
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-4"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: Some("/Users/davidloor/Library/Logs/aidaemon"),
                trusted: false,
                user_role: UserRole::Owner,
            },
        )
        .await
        .expect("spawn_agent call should succeed");

    let payload: Value = serde_json::from_str(&result).expect("spawn_agent args should be JSON");
    assert_eq!(
        payload.get("_project_scope").and_then(Value::as_str),
        Some("/Users/davidloor/Library/Logs/aidaemon")
    );
    assert_eq!(
        payload.get("mission").and_then(Value::as_str),
        Some("delegate log analysis")
    );
    assert_eq!(
        payload.get("_session_id").and_then(Value::as_str),
        Some("test-session")
    );
}
