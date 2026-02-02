use std::sync::{OnceLock, Weak};
use std::time::Duration;

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
    max_response_chars: usize,
    timeout_secs: u64,
}

impl SpawnAgentTool {
    /// Create a SpawnAgentTool with a known agent reference.
    #[allow(dead_code)]
    pub fn new(agent: Weak<Agent>, max_response_chars: usize, timeout_secs: u64) -> Self {
        let lock = OnceLock::new();
        let _ = lock.set(agent);
        Self {
            agent: lock,
            max_response_chars,
            timeout_secs,
        }
    }

    /// Create a SpawnAgentTool with a deferred agent reference.
    /// Call [`set_agent`] after constructing the `Arc<Agent>`.
    pub fn new_deferred(max_response_chars: usize, timeout_secs: u64) -> Self {
        Self {
            agent: OnceLock::new(),
            max_response_chars,
            timeout_secs,
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

/// Truncate a string to at most `max_chars` bytes without splitting a
/// multi-byte UTF-8 character. Returns the original string when it fits.
fn truncate_utf8(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        return s;
    }
    // Find the last char boundary at or before `max_chars`.
    let boundary = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= max_chars)
        .last()
        .unwrap_or(0);
    &s[..boundary]
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

        let timeout_duration = Duration::from_secs(self.timeout_secs);
        let result = tokio::time::timeout(
            timeout_duration,
            agent.spawn_child(&args.mission, &args.task),
        )
        .await;

        match result {
            Ok(Ok(response)) => {
                let max_len = self.max_response_chars;
                if response.len() > max_len {
                    let truncated = truncate_utf8(&response, max_len);
                    Ok(format!(
                        "{}\n\n[Sub-agent response truncated at {} chars]",
                        truncated, max_len
                    ))
                } else {
                    Ok(response)
                }
            }
            Ok(Err(e)) => Ok(format!("Sub-agent error: {}", e)),
            Err(_) => Ok(format!(
                "Sub-agent timed out after {} seconds",
                self.timeout_secs
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_utf8_ascii() {
        assert_eq!(truncate_utf8("hello world", 5), "hello");
        assert_eq!(truncate_utf8("hello", 10), "hello");
        assert_eq!(truncate_utf8("hello", 5), "hello");
    }

    #[test]
    fn truncate_utf8_multibyte() {
        // Each emoji is 4 bytes
        let s = "🔥🔥🔥";
        assert_eq!(s.len(), 12);
        // Limit 4 should include exactly the first emoji
        assert_eq!(truncate_utf8(s, 4), "🔥");
        // Limit 5 should still only include the first emoji (next starts at 4, ends at 8)
        assert_eq!(truncate_utf8(s, 5), "🔥");
        // Limit 8 should include two emojis
        assert_eq!(truncate_utf8(s, 8), "🔥🔥");
        // Limit 1 — no full character fits, but char_indices first is (0, '🔥')
        // take_while(|&i| i <= 1) → only i=0 qualifies
        assert_eq!(truncate_utf8(s, 1), "");
    }

    #[test]
    fn truncate_utf8_mixed() {
        let s = "hi🌍!";
        // 'h'=1, 'i'=1, '🌍'=4, '!'=1 → total 7
        assert_eq!(truncate_utf8(s, 3), "hi");
        assert_eq!(truncate_utf8(s, 6), "hi🌍");
        assert_eq!(truncate_utf8(s, 7), "hi🌍!");
    }

    #[test]
    fn truncate_utf8_empty() {
        assert_eq!(truncate_utf8("", 10), "");
        assert_eq!(truncate_utf8("", 0), "");
    }

    #[test]
    fn deferred_initialization_not_set() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        let result = tool.get_agent();
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not set"));
    }

    #[test]
    fn config_defaults() {
        use crate::config::SubagentsConfig;
        let cfg = SubagentsConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.max_depth, 3);
        assert_eq!(cfg.max_iterations, 10);
        assert_eq!(cfg.max_response_chars, 8000);
        assert_eq!(cfg.timeout_secs, 300);
    }
}
