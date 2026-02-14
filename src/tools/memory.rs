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
        "Store a fact about the user or environment for long-term memory"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "remember_fact",
            "description": "Store a fact about the user or environment for long-term memory. Facts are injected into your system prompt on every request.",
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
        let combined = format!("{} {} {}", args.category, args.key, args.value).to_ascii_lowercase();
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
