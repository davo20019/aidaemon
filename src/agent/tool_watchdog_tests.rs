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
            "test-session",
            Some("task-1"),
            None,
            ChannelVisibility::Private,
            None,
            false,
            UserRole::Owner,
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
            "test-session",
            Some("task-3"),
            None,
            ChannelVisibility::Private,
            None,
            false,
            UserRole::Owner,
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
    harness.agent.llm_call_timeout = Some(Duration::from_secs(1));

    let result = harness
        .agent
        .execute_tool_with_watchdog(
            "system_info",
            "{}",
            "test-session",
            Some("task-2"),
            None,
            ChannelVisibility::Private,
            None,
            false,
            UserRole::Owner,
        )
        .await
        .expect("fast tool should succeed");

    assert!(
        !result.is_empty(),
        "system_info should return a non-empty payload"
    );
}
