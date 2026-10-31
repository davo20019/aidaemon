//! Synthetic tools, a policy hook and a factory for the shipped
//! `scenarios/` directory. Enabled with the `examples` feature.

use std::sync::Arc;

use aidaemon_core::testing::spec;
use aidaemon_core::{
    Agent, FnTool, Hook, HookContext, Provider, Tool, ToolAnnotations, ToolCall, ToolDecision,
    ToolInfo, ToolOutput,
};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::runner::{AgentFactory, FactoryError};
use crate::scenario::Scenario;

fn arg(args: &Value, key: &str) -> String {
    args[key].as_str().unwrap_or_default().to_string()
}

/// Four small tools with canned output:
///
/// - `get_weather` (read-only, `city`): "Sunny and 21C in {city}".
/// - `read_file` (read-only, `path`): "(contents of {path})".
/// - `check_status` (read-only, `job`): always "status: pending", so
///   repeated polling looks stuck.
/// - `delete_file` (destructive, `path`): "deleted {path}".
pub fn example_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(FnTool::new(
            spec("get_weather", "Current weather for a city", &["city"]),
            ToolAnnotations::READ_ONLY,
            |args| async move {
                Ok(ToolOutput::text(format!(
                    "Sunny and 21C in {}",
                    arg(&args, "city")
                )))
            },
        )),
        Arc::new(FnTool::new(
            spec("read_file", "Read a UTF-8 text file", &["path"]),
            ToolAnnotations::READ_ONLY,
            |args| async move {
                Ok(ToolOutput::text(format!(
                    "(contents of {})",
                    arg(&args, "path")
                )))
            },
        )),
        Arc::new(FnTool::new(
            spec(
                "check_status",
                "Check the status of a background job",
                &["job"],
            ),
            ToolAnnotations::READ_ONLY,
            |_args| async move { Ok(ToolOutput::text("status: pending")) },
        )),
        Arc::new(FnTool::new(
            spec("delete_file", "Delete a file", &["path"]),
            ToolAnnotations::DESTRUCTIVE,
            |args| async move {
                let path = arg(&args, "path");
                Ok(ToolOutput::text(format!("deleted {path}")).with_details(json!({"path": path})))
            },
        )),
    ]
}

/// Policy hook: deny every call to a tool annotated as destructive.
pub struct DenyDestructive;

#[async_trait]
impl Hook for DenyDestructive {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        if tool.annotations.destructive {
            ToolDecision::Deny {
                reason: format!("`{}` is destructive and not allowed here", call.name),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

/// Builds an agent with [`example_tools`] and, unless the scenario has the
/// tag `allow_destructive`, the [`DenyDestructive`] hook.
#[derive(Default)]
pub struct ExampleFactory {
    live: Option<Arc<dyn Provider>>,
}

impl ExampleFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_live_provider(provider: Arc<dyn Provider>) -> Self {
        Self {
            live: Some(provider),
        }
    }
}

impl AgentFactory for ExampleFactory {
    fn build(
        &self,
        scenario: &Scenario,
        provider: Arc<dyn Provider>,
    ) -> Result<Agent, FactoryError> {
        let mut builder = Agent::builder("example-agent")
            .model("example-model")
            .provider(provider);
        for tool in example_tools() {
            builder = builder.tool_arc(tool);
        }
        if !scenario.tags.iter().any(|t| t == "allow_destructive") {
            builder = builder.hook(DenyDestructive);
        }
        Ok(scenario.configure(builder).build()?)
    }

    fn live_provider(&self, _scenario: &Scenario) -> Option<Arc<dyn Provider>> {
        self.live.clone()
    }
}
