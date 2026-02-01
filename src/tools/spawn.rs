use std::sync::{OnceLock, Weak};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::agent::Agent;
use crate::traits::Tool;

/// A tool that allows the LLM to spawn a sub-agent for a focused task.
///
/// The sub-agent runs its own agentic loop with a dedicated session and a
/// system prompt that includes the given mission. Its final text response is
/// returned as the tool result.
///
/// The circular dependency (Agent → tools → SpawnAgentTool → Agent) is broken
/// by storing a `Weak<Agent>` inside a `OnceLock`. The weak reference is set
/// after the owning `Arc<Agent>` is constructed.
pub struct SpawnAgentTool {
    agent: OnceLock<Weak<Agent>>,
}

impl SpawnAgentTool {
    /// Create a SpawnAgentTool with a known agent reference.
    #[allow(dead_code)]
    pub fn new(agent: Weak<Agent>) -> Self {
        let lock = OnceLock::new();
        let _ = lock.set(agent);
        Self { agent: lock }
    }

    /// Create a SpawnAgentTool with a deferred agent reference.
    /// Call [`set_agent`] after constructing the `Arc<Agent>`.
    pub fn new_deferred() -> Self {
        Self {
            agent: OnceLock::new(),
        }
    }

    /// Set the agent reference. Must be called exactly once after the owning
    /// `Arc<Agent>` is constructed. Panics if called more than once.
    pub fn set_agent(&self, agent: Weak<Agent>) {
        self.agent
            .set(agent)
            .expect("SpawnAgentTool::set_agent called more than once");
    }

    fn get_agent(&self) -> anyhow::Result<std::sync::Arc<Agent>> {
        let weak = self
            .agent
            .get()
            .ok_or_else(|| anyhow::anyhow!("SpawnAgentTool: agent reference not set"))?;
        weak.upgrade()
            .ok_or_else(|| anyhow::anyhow!("SpawnAgentTool: parent agent has been dropped"))
    }
}

#[derive(Deserialize)]
struct SpawnArgs {
    /// High-level mission / role description for the sub-agent.
    mission: String,
    /// The concrete task or question the sub-agent should work on.
    task: String,
}

#[async_trait]
impl Tool for SpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Spawn a sub-agent to handle a focused task autonomously. \
         The sub-agent has access to all tools and runs its own reasoning loop. \
         Use this for tasks that benefit from isolated, parallel reasoning."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "spawn_agent",
            "description": "Spawn a sub-agent to handle a focused task autonomously. \
                The sub-agent has access to all tools and runs its own reasoning loop. \
                Use this for complex sub-tasks that benefit from isolated context and focused reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mission": {
                        "type": "string",
                        "description": "High-level mission or role for the sub-agent \
                            (e.g. 'Research assistant focused on Python packaging')"
                    },
                    "task": {
                        "type": "string",
                        "description": "The specific task or question the sub-agent should accomplish"
                    }
                },
                "required": ["mission", "task"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: SpawnArgs = serde_json::from_str(arguments)?;
        let agent = self.get_agent()?;

        info!(
            depth = agent.depth(),
            max_depth = agent.max_depth(),
            mission = %args.mission,
            "spawn_agent tool invoked"
        );

        let result = agent.spawn_child(&args.mission, &args.task).await;

        match result {
            Ok(response) => {
                // Truncate very long sub-agent responses to avoid blowing up context.
                const MAX_RESPONSE_LEN: usize = 8000;
                if response.len() > MAX_RESPONSE_LEN {
                    let truncated = &response[..MAX_RESPONSE_LEN];
                    Ok(format!(
                        "{}\n\n[Sub-agent response truncated at {} chars]",
                        truncated, MAX_RESPONSE_LEN
                    ))
                } else {
                    Ok(response)
                }
            }
            Err(e) => Ok(format!("Sub-agent error: {}", e)),
        }
    }
}
