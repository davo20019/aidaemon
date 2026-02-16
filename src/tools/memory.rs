use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::traits::{StateStore, Tool, ToolRole};
use crate::types::FactPrivacy;

pub struct RememberFactTool {
    state: Arc<dyn StateStore>,
    /// Current channel_id set by the agent before tool execution.
    pub(crate) current_channel_id: Arc<RwLock<Option<String>>>,
}

impl RememberFactTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self {
            state,
            current_channel_id: Arc::new(RwLock::new(None)),
        }
    }
}

#[derive(Deserialize)]
struct RememberArgs {
    category: String,
    key: String,
    value: String,
}

#[async_trait]
impl Tool for RememberFactTool {
    fn name(&self) -> &str {
        "remember_fact"
    }

    fn description(&self) -> &str {
        "Store a long-lived fact (not goals or schedules) for long-term memory"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "remember_fact",
            "description": "Store a stable, long-term fact about the user or their environment. Facts are injected into your system prompt on every request, so only store things that are persistently useful — user preferences, personal info, environment details, communication patterns. Do NOT store task-scoped research, reference data gathered for a specific project, or content being built (e.g., product prices, API docs, website copy). Do NOT use this for personal goals or scheduled work; use the manage_memories tool (create_personal_goal / create_scheduled_goal) for goals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category for the fact (e.g. 'user', 'preference', 'project')"
                    },
                    "key": {
                        "type": "string",
                        "description": "A unique key for this fact within the category"
                    },
                    "value": {
                        "type": "string",
                        "description": "The fact to remember"
                    }
                },
                "required": ["category", "key", "value"]
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: RememberArgs = serde_json::from_str(arguments)?;

        // Reject persona/identity manipulation saves
        let combined =
            format!("{} {} {}", args.category, args.key, args.value).to_ascii_lowercase();
        let persona_patterns = [
            "talk like",
            "speak like",
            "act like",
            "act as",
            "pretend to be",
            "roleplay",
            "persona",
            "character voice",
            "pirate",
            "accent",
            "from now on",
            "new identity",
            "speak in character",
            "respond as",
        ];
        if persona_patterns.iter().any(|p| combined.contains(p)) {
            return Ok(
                "Rejected: Cannot save persona or identity changes as facts. \
                 I maintain a consistent identity across all interactions."
                    .to_string(),
            );
        }

        // Reject personal goal tracking in facts. The goal registry is the source of truth.
        let category_lower = args.category.trim().to_ascii_lowercase();
        let key_lower = args.key.trim().to_ascii_lowercase();
        let value_lower = args.value.trim().to_ascii_lowercase();
        let looks_like_personal_goal_key =
            key_lower.starts_with("personal_goal") || key_lower.contains("personal_goal");
        let looks_like_user_goal_key = key_lower.starts_with("goal_")
            && matches!(category_lower.as_str(), "user" | "preference");
        let looks_like_goal_value = matches!(category_lower.as_str(), "user" | "preference")
            && (value_lower.contains("my goal")
                || value_lower.contains("personal goal")
                || value_lower.contains("goal is to")
                || value_lower.starts_with("goal:"));
        if looks_like_personal_goal_key || looks_like_user_goal_key || looks_like_goal_value {
            return Ok(
                "Rejected: Personal goals should be tracked in the goal registry (not facts). \
                 Use manage_memories(action='create_personal_goal', goal='...') instead."
                    .to_string(),
            );
        }

        let channel_id = self.current_channel_id.read().await.clone();
        // When explicitly remembered by the agent, default to global privacy
        self.state
            .upsert_fact(
                &args.category,
                &args.key,
                &args.value,
                "agent",
                channel_id.as_deref(),
                FactPrivacy::Global,
            )
            .await?;
        Ok(format!(
            "Remembered: [{}] {} = {}",
            args.category, args.key, args.value
        ))
    }
}
