use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::agent::{Agent, StatusUpdate};
use crate::channels::ChannelHub;
use crate::traits::{AgentRole, Tool};
use crate::types::{ChannelContext, ChannelVisibility, UserRole};

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
    hub: OnceLock<Weak<ChannelHub>>,
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
            hub: OnceLock::new(),
            max_response_chars,
            timeout_secs,
        }
    }

    /// Create a SpawnAgentTool with a deferred agent reference.
    /// Call [`set_agent`] after constructing the `Arc<Agent>`.
    pub fn new_deferred(max_response_chars: usize, timeout_secs: u64) -> Self {
        Self {
            agent: OnceLock::new(),
            hub: OnceLock::new(),
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

    /// Set the channel hub reference for background mode notifications.
    pub fn set_hub(&self, hub: Weak<ChannelHub>) {
        let _ = self.hub.set(hub);
    }

    fn get_hub(&self) -> Option<Arc<ChannelHub>> {
        self.hub.get().and_then(|w| w.upgrade())
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
    /// When true, spawn the sub-agent in the background and return immediately.
    #[serde(default)]
    background: bool,
    /// Task ID — when provided by a task lead, the executor tracks activity against this task.
    #[serde(default)]
    task_id: Option<String>,
    /// Session ID injected by execute_tool — used for background completion notifications.
    #[serde(default)]
    _session_id: Option<String>,
    /// Channel visibility injected by execute_tool — propagated to child agents.
    #[serde(default)]
    _channel_visibility: Option<String>,
    /// User role injected by execute_tool — propagated to child agents to prevent
    /// privilege escalation (e.g., Guest user spawning Owner-level sub-agent).
    #[serde(default)]
    _user_role: Option<String>,
    /// Task ID injected by execute_tool for task-lead → executor spawning.
    #[serde(default)]
    _task_id: Option<String>,
    /// Goal ID injected by execute_tool for task-lead → executor spawning.
    #[serde(default)]
    _goal_id: Option<String>,
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
                    },
                    "background": {
                        "type": "boolean",
                        "description": "When true, spawn the sub-agent in the background and return immediately. \
                            The result will be sent as a message when the sub-agent finishes. \
                            Use this for long-running tasks where the user doesn't need to wait.",
                        "default": false
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to associate with this executor (used by task leads to connect executor work to task tracking)"
                    }
                },
                "required": ["mission", "task"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.call_with_status(arguments, None).await
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let args: SpawnArgs = serde_json::from_str(arguments)?;
        let agent = self.get_agent()?;

        info!(
            depth = agent.depth(),
            max_depth = agent.max_depth(),
            mission = %args.mission,
            background = args.background,
            "spawn_agent tool invoked"
        );

        // Reconstruct channel context from injected visibility string
        let channel_ctx = {
            let visibility = args
                ._channel_visibility
                .as_deref()
                .map(ChannelVisibility::from_str_lossy)
                .unwrap_or(ChannelVisibility::Internal);
            ChannelContext {
                visibility,
                platform: "internal".to_string(),
                channel_name: None,
                channel_id: None,
                sender_name: None,
                sender_id: None,
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            }
        };

        // Propagate parent's user role to prevent privilege escalation.
        // Default to Guest (least privilege) if not provided.
        let user_role = match args._user_role.as_deref() {
            Some("Owner") => UserRole::Owner,
            Some("Guest") => UserRole::Guest,
            _ => UserRole::Guest,
        };

        // Determine child role based on parent's role.
        let child_role = if agent.role() == AgentRole::TaskLead {
            Some(AgentRole::Executor)
        } else {
            None
        };
        // LLM-provided task_id takes priority; fall back to injected _task_id
        let task_id_ref = args.task_id.or(args._task_id.clone());
        let goal_id_ref = args._goal_id.clone();

        if !args.background {
            return self
                .run_sync(
                    agent,
                    &args.mission,
                    &args.task,
                    status_tx,
                    channel_ctx,
                    user_role,
                    child_role,
                    goal_id_ref.as_deref(),
                    task_id_ref.as_deref(),
                )
                .await;
        }

        // Background mode: need hub + session_id, fall back to sync if unavailable
        let hub = match self.get_hub() {
            Some(h) => h,
            None => {
                info!("Background mode requested but hub not available, falling back to sync");
                return self
                    .run_sync(
                        agent,
                        &args.mission,
                        &args.task,
                        status_tx,
                        channel_ctx,
                        user_role,
                        child_role,
                        goal_id_ref.as_deref(),
                        task_id_ref.as_deref(),
                    )
                    .await;
            }
        };
        let session_id = match args._session_id {
            Some(ref id) if !id.is_empty() => id.clone(),
            _ => {
                info!("Background mode requested but no session_id, falling back to sync");
                return self
                    .run_sync(
                        agent,
                        &args.mission,
                        &args.task,
                        status_tx,
                        channel_ctx,
                        user_role,
                        child_role,
                        goal_id_ref.as_deref(),
                        task_id_ref.as_deref(),
                    )
                    .await;
            }
        };

        let mission = args.mission.clone();
        let timeout_secs = self.timeout_secs;
        let max_response_chars = self.max_response_chars;

        tokio::spawn(async move {
            let timeout_duration = Duration::from_secs(timeout_secs);
            let result = tokio::time::timeout(
                timeout_duration,
                agent.spawn_child(
                    &mission,
                    &args.task,
                    status_tx,
                    channel_ctx,
                    user_role,
                    child_role,
                    goal_id_ref.as_deref(),
                    task_id_ref.as_deref(),
                ),
            )
            .await;

            let message = match result {
                Ok(Ok(response)) => {
                    let text = if response.len() > max_response_chars {
                        truncate_utf8(&response, max_response_chars).to_string()
                    } else {
                        response
                    };
                    format!(
                        "\u{2705} Background task complete\nMission: {}\n\n{}",
                        mission, text
                    )
                }
                Ok(Err(e)) => {
                    format!(
                        "\u{274c} Background task failed\nMission: {}\nError: {}",
                        mission, e
                    )
                }
                Err(_) => {
                    format!(
                        "\u{23f1} Background task timed out\nMission: {}\nTimed out after {}s",
                        mission, timeout_secs
                    )
                }
            };

            if let Err(e) = hub.send_text(&session_id, &message).await {
                error!(
                    session_id = %session_id,
                    error = %e,
                    "Failed to send background task result"
                );
            }
        });

        Ok(format!(
            "Sub-agent spawned in background for mission: \"{}\". \
             The result will be sent as a message when it completes.",
            args.mission
        ))
    }
}

impl SpawnAgentTool {
    /// Run the sub-agent synchronously (blocking until completion or timeout).
    #[allow(clippy::too_many_arguments)]
    async fn run_sync(
        &self,
        agent: Arc<Agent>,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
        child_role: Option<AgentRole>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let timeout_duration = Duration::from_secs(self.timeout_secs);
        let result = tokio::time::timeout(
            timeout_duration,
            agent.spawn_child(
                mission,
                task,
                status_tx,
                channel_ctx,
                user_role,
                child_role,
                goal_id,
                task_id,
            ),
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

    #[test]
    fn deferred_hub_not_set() {
        let tool = SpawnAgentTool::new_deferred(8000, 300);
        assert!(tool.get_hub().is_none());
    }

    #[test]
    fn spawn_args_background_default() {
        let json = r#"{"mission": "test", "task": "do stuff"}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert!(!args.background);
        assert!(args._session_id.is_none());
        assert!(args._channel_visibility.is_none());
    }

    #[test]
    fn spawn_args_background_true() {
        let json = r#"{"mission": "test", "task": "do stuff", "background": true, "_session_id": "tg:123"}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert!(args.background);
        assert_eq!(args._session_id.as_deref(), Some("tg:123"));
    }

    #[test]
    fn spawn_args_with_channel_visibility() {
        let json = r#"{"mission": "test", "task": "do stuff", "_channel_visibility": "public"}"#;
        let args: SpawnArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args._channel_visibility.as_deref(), Some("public"));
    }
}
