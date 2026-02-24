use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::traits::{StateStore, Tool, ToolCapabilities, ToolRole};
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
    category: Option<String>,
    key: Option<String>,
    value: Option<String>,
    #[serde(default)]
    facts: Option<Vec<FactEntry>>,
}

#[derive(Deserialize)]
struct FactEntry {
    category: String,
    key: String,
    value: String,
}

const PERSONA_PATTERNS: &[&str] = &[
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

fn is_persona_manipulation(category: &str, key: &str, value: &str) -> bool {
    let combined = format!("{} {} {}", category, key, value).to_ascii_lowercase();
    PERSONA_PATTERNS.iter().any(|p| combined.contains(p))
}

fn is_goal_fact(category: &str, key: &str, value: &str) -> bool {
    let category_lower = category.trim().to_ascii_lowercase();
    let key_lower = key.trim().to_ascii_lowercase();
    let value_lower = value.trim().to_ascii_lowercase();
    let looks_like_personal_goal_key =
        key_lower.starts_with("personal_goal") || key_lower.contains("personal_goal");
    let looks_like_user_goal_key =
        key_lower.starts_with("goal_") && matches!(category_lower.as_str(), "user" | "preference");
    let looks_like_goal_value = matches!(category_lower.as_str(), "user" | "preference")
        && (value_lower.contains("my goal")
            || value_lower.contains("personal goal")
            || value_lower.contains("goal is to")
            || value_lower.starts_with("goal:"));
    looks_like_personal_goal_key || looks_like_user_goal_key || looks_like_goal_value
}

#[async_trait]
impl Tool for RememberFactTool {
    fn name(&self) -> &str {
        "remember_fact"
    }

    fn description(&self) -> &str {
        "Store one or more long-lived facts (not goals or schedules) for long-term memory; use when user says learn/remember/save this"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "remember_fact",
            "description": "Store one or more stable, long-term facts about the user or their environment. Use this when the user asks you to learn, remember, or save facts for later. Facts are injected into your system prompt on every request, so only store things that are persistently useful — user preferences, personal info, environment details, communication patterns. Do NOT store task-scoped research, reference data gathered for a specific project, or content being built (e.g., product prices, API docs, website copy). Do NOT use this for personal goals or scheduled work; use the manage_memories tool (create_personal_goal / create_scheduled_goal) for goals. For multiple facts, use the 'facts' array parameter instead of making separate calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category for a single fact (e.g. 'user', 'preference', 'project')"
                    },
                    "key": {
                        "type": "string",
                        "description": "A unique key for a single fact within the category"
                    },
                    "value": {
                        "type": "string",
                        "description": "The single fact to remember"
                    },
                    "facts": {
                        "type": "array",
                        "description": "Batch mode: an array of facts to store at once. Use this when the user mentions multiple facts in one message.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "Category for this fact"
                                },
                                "key": {
                                    "type": "string",
                                    "description": "A unique key for this fact"
                                },
                                "value": {
                                    "type": "string",
                                    "description": "The fact to remember"
                                }
                            },
                            "required": ["category", "key", "value"]
                        }
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: RememberArgs = serde_json::from_str(arguments)?;

        // Build the list of facts to store (batch or single)
        let entries: Vec<FactEntry> = if let Some(facts) = args.facts {
            if facts.is_empty() {
                anyhow::bail!("'facts' array is empty — provide at least one fact");
            }
            facts
        } else {
            // Single-fact mode: require all three fields
            let category = args
                .category
                .ok_or_else(|| anyhow::anyhow!("'category' is required (or use 'facts' array)"))?;
            let key = args
                .key
                .ok_or_else(|| anyhow::anyhow!("'key' is required (or use 'facts' array)"))?;
            let value = args
                .value
                .ok_or_else(|| anyhow::anyhow!("'value' is required (or use 'facts' array)"))?;
            vec![FactEntry {
                category,
                key,
                value,
            }]
        };

        let channel_id = self.current_channel_id.read().await.clone();
        let mut results = Vec::new();

        for entry in &entries {
            // Reject persona/identity manipulation saves
            if is_persona_manipulation(&entry.category, &entry.key, &entry.value) {
                results.push(format!(
                    "Rejected [{}] {}: cannot save persona/identity changes",
                    entry.category, entry.key
                ));
                continue;
            }

            // Reject personal goal tracking in facts
            if is_goal_fact(&entry.category, &entry.key, &entry.value) {
                results.push(format!(
                    "Rejected [{}] {}: use manage_memories(create_personal_goal) for goals",
                    entry.category, entry.key
                ));
                continue;
            }

            self.state
                .upsert_fact(
                    &entry.category,
                    &entry.key,
                    &entry.value,
                    "agent",
                    channel_id.as_deref(),
                    FactPrivacy::Global,
                )
                .await?;
            results.push(format!(
                "Remembered: [{}] {} = {}",
                entry.category, entry.key, entry.value
            ));
        }

        Ok(results.join("\n"))
    }
}
