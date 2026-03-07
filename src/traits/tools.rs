use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::types::StatusUpdate;

/// Role assigned to an agent for role-based tool scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    /// Root agent — routes, classifies, full tool access (legacy behavior).
    Orchestrator,
    /// Plans & delegates — management tools only.
    TaskLead,
    /// Executes a single task — action tools + report_blocker.
    Executor,
}

/// Categorization of a tool for role-based scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRole {
    /// Terminal, web_search, web_fetch, browser, etc.
    Action,
    /// ManageGoalTasksTool, ReportBlockerTool — task lead tools.
    Management,
    /// SystemInfoTool, RememberFactTool — available to all roles.
    Universal,
}

/// Safety and execution metadata for policy-driven tool selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub read_only: bool,
    pub external_side_effect: bool,
    pub needs_approval: bool,
    pub idempotent: bool,
    pub high_impact_write: bool,
}

/// Structured execution metadata returned by tools.
///
/// This is intentionally minimal and backward-compatible: tools can continue
/// returning plain text while selectively populating structured fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallMetadata {
    /// Process exit code when applicable (e.g. terminal/run_command style tools).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// True when tool execution exceeded a timeout threshold.
    #[serde(default)]
    pub timed_out: bool,
    /// True when execution moved to background tracking.
    #[serde(default)]
    pub background_started: bool,
    /// True when the process is detached and intentionally long-lived.
    #[serde(default)]
    pub detached: bool,
    /// True when the tool guarantees automatic completion delivery for a
    /// backgrounded operation in the current run.
    #[serde(default)]
    pub completion_notifications_enabled: bool,
    /// Transport/runtime failure outside normal tool semantics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallOutcome {
    pub output: String,
    #[serde(default)]
    pub metadata: ToolCallMetadata,
}

impl ToolCallOutcome {
    pub fn from_output(output: String) -> Self {
        Self {
            output,
            metadata: ToolCallMetadata::default(),
        }
    }
}

impl Default for ToolCapabilities {
    fn default() -> Self {
        Self {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }
}

/// Tool trait — system tools, terminal, MCP-proxied tools.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// Returns the OpenAI-format function schema as a JSON Value.
    fn schema(&self) -> Value;
    /// Execute the tool with the given JSON arguments string, returns result text.
    async fn call(&self, arguments: &str) -> anyhow::Result<String>;

    /// Execute the tool with access to a status update channel for streaming feedback.
    /// Default implementation just calls `call()` - override for tools that emit progress.
    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        // Default: ignore status channel and just call the basic method
        let _ = status_tx;
        self.call(arguments).await
    }

    /// Structured execution path used by the agent loop.
    ///
    /// Default behavior preserves compatibility for existing tools by wrapping
    /// plain text output with empty metadata.
    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let output = self.call_with_status(arguments, status_tx).await?;
        Ok(ToolCallOutcome::from_output(output))
    }

    /// Task lifecycle callback fired after the agent emits `TaskEnd`.
    /// Tools that spawn background activity can use this to clean up
    /// task-scoped resources.
    async fn on_task_end(&self, _task_id: &str, _session_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Categorize this tool for role-based scoping.
    /// Default: Action (most tools are action tools).
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    /// Capability metadata used by the execution policy and risk gate.
    /// Defaults are intentionally conservative.
    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::default()
    }

    /// Whether this tool is currently operational.
    ///
    /// Default: true. Override for tools with dynamic backends that may be
    /// temporarily unavailable at runtime.
    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct AlwaysAvailableTool;

    #[async_trait]
    impl Tool for AlwaysAvailableTool {
        fn name(&self) -> &str {
            "always_available"
        }

        fn description(&self) -> &str {
            "test"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "always_available",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }
    }

    struct UnavailableTool;

    #[async_trait]
    impl Tool for UnavailableTool {
        fn name(&self) -> &str {
            "unavailable"
        }

        fn description(&self) -> &str {
            "test"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "unavailable",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }

        fn is_available(&self) -> bool {
            false
        }
    }

    #[test]
    fn default_is_available_returns_true() {
        let tool = AlwaysAvailableTool;
        assert!(tool.is_available());
    }

    #[test]
    fn override_is_available_returns_false() {
        let tool = UnavailableTool;
        assert!(!tool.is_available());
    }
}
