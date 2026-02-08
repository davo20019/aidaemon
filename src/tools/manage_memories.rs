use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{StateStore, Tool};
use crate::types::FactPrivacy;

pub struct ManageMemoriesTool {
    state: Arc<dyn StateStore>,
}

impl ManageMemoriesTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self { state }
    }
}

#[derive(Deserialize)]
struct ManageArgs {
    action: String,
    category: Option<String>,
    key: Option<String>,
    privacy: Option<String>,
    query: Option<String>,
    goal_id: Option<i64>,
}

#[async_trait]
impl Tool for ManageMemoriesTool {
    fn name(&self) -> &str {
        "manage_memories"
    }

    fn description(&self) -> &str {
        "List, search, forget, or change privacy of stored memories and goals"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_memories",
            "description": "List, search, forget, or change privacy of stored memories and goals. Use when the user asks 'what do you remember?', 'forget about X', wants to manage memory privacy, or manage tracked goals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "forget", "set_privacy", "search", "list_goals", "complete_goal", "abandon_goal"],
                        "description": "Action to perform"
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (for list/forget/set_privacy)"
                    },
                    "key": {
                        "type": "string",
                        "description": "Fact key (for forget/set_privacy)"
                    },
                    "privacy": {
                        "type": "string",
                        "enum": ["global", "channel", "private"],
                        "description": "Target privacy level (for set_privacy)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search term (for search action)"
                    },
                    "goal_id": {
                        "type": "integer",
                        "description": "Goal ID (for complete_goal/abandon_goal)"
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            "list" => {
                let facts = self.state.get_all_facts_with_provenance().await?;
                if facts.is_empty() {
                    return Ok("No memories stored.".to_string());
                }

                let filtered = if let Some(ref cat) = args.category {
                    facts.into_iter().filter(|f| f.category == *cat).collect::<Vec<_>>()
                } else {
                    facts
                };

                let mut output = format!("**Stored Memories** ({} facts)\n\n", filtered.len());
                let mut current_cat = String::new();
                for f in &filtered {
                    if f.category != current_cat {
                        current_cat = f.category.clone();
                        output.push_str(&format!("### {}\n", current_cat));
                    }
                    let privacy_label = f.privacy.to_string();
                    let channel_label = f.channel_id.as_deref().unwrap_or("global");
                    let age = chrono::Utc::now().signed_duration_since(f.updated_at);
                    let age_str = if age.num_days() > 0 {
                        format!("{}d ago", age.num_days())
                    } else if age.num_hours() > 0 {
                        format!("{}h ago", age.num_hours())
                    } else {
                        "just now".to_string()
                    };
                    output.push_str(&format!(
                        "- **{}**: {} (privacy: {}, from: {}, updated: {})\n",
                        f.key, f.value, privacy_label, channel_label, age_str
                    ));
                }
                Ok(output)
            }
            "forget" => {
                let key = args.key.as_deref().ok_or_else(|| anyhow::anyhow!("'key' is required for forget action"))?;
                let category = args.category.as_deref().ok_or_else(|| anyhow::anyhow!("'category' is required for forget action"))?;

                let facts = self.state.get_facts(Some(category)).await?;
                let fact = facts.iter().find(|f| f.key == key && f.superseded_at.is_none());

                match fact {
                    Some(f) => {
                        self.state.delete_fact(f.id).await?;
                        Ok(format!("Forgotten: [{}] {}", category, key))
                    }
                    None => Ok(format!("No active fact found: [{}] {}", category, key)),
                }
            }
            "set_privacy" => {
                let key = args.key.as_deref().ok_or_else(|| anyhow::anyhow!("'key' is required for set_privacy action"))?;
                let category = args.category.as_deref().ok_or_else(|| anyhow::anyhow!("'category' is required for set_privacy action"))?;
                let privacy_str = args.privacy.as_deref().ok_or_else(|| anyhow::anyhow!("'privacy' is required for set_privacy action"))?;
                let privacy = FactPrivacy::from_str_lossy(privacy_str);

                let facts = self.state.get_facts(Some(category)).await?;
                let fact = facts.iter().find(|f| f.key == key && f.superseded_at.is_none());

                match fact {
                    Some(f) => {
                        self.state.update_fact_privacy(f.id, privacy).await?;
                        Ok(format!("Updated privacy: [{}] {} → {}", category, key, privacy))
                    }
                    None => Ok(format!("No active fact found: [{}] {}", category, key)),
                }
            }
            "search" => {
                let query = args.query.as_deref().unwrap_or("");
                if query.is_empty() {
                    return Ok("Please provide a search query.".to_string());
                }

                let facts = self.state.get_all_facts_with_provenance().await?;
                let query_lower = query.to_lowercase();
                let matches: Vec<_> = facts
                    .iter()
                    .filter(|f| {
                        f.key.to_lowercase().contains(&query_lower)
                            || f.value.to_lowercase().contains(&query_lower)
                            || f.category.to_lowercase().contains(&query_lower)
                    })
                    .collect();

                if matches.is_empty() {
                    return Ok(format!("No memories matching '{}'.", query));
                }

                let mut output = format!("**Search results for '{}'** ({} matches)\n\n", query, matches.len());
                for f in matches.iter().take(20) {
                    let privacy_label = f.privacy.to_string();
                    let channel_label = f.channel_id.as_deref().unwrap_or("global");
                    output.push_str(&format!(
                        "- [{}] **{}**: {} (privacy: {}, from: {})\n",
                        f.category, f.key, f.value, privacy_label, channel_label
                    ));
                }
                Ok(output)
            }
            "list_goals" => {
                let goals = self.state.get_active_goals().await?;
                if goals.is_empty() {
                    return Ok("No active goals.".to_string());
                }

                let mut output = format!("**Active Goals** ({} goals)\n\n", goals.len());
                for g in &goals {
                    let age = chrono::Utc::now().signed_duration_since(g.created_at);
                    let age_str = if age.num_days() > 0 {
                        format!("{}d ago", age.num_days())
                    } else if age.num_hours() > 0 {
                        format!("{}h ago", age.num_hours())
                    } else {
                        "just now".to_string()
                    };
                    let notes_count = g.progress_notes.as_ref().map_or(0, |n| n.len());
                    output.push_str(&format!(
                        "- **[ID: {}]** {} (priority: {}, created: {}, {} progress notes)\n",
                        g.id, g.description, g.priority, age_str, notes_count
                    ));
                }
                Ok(output)
            }
            "complete_goal" => {
                let goal_id = args.goal_id.ok_or_else(|| anyhow::anyhow!("'goal_id' is required for complete_goal action"))?;
                self.state.update_goal(goal_id, Some("completed"), None).await?;
                Ok(format!("Goal {} marked as completed.", goal_id))
            }
            "abandon_goal" => {
                let goal_id = args.goal_id.ok_or_else(|| anyhow::anyhow!("'goal_id' is required for abandon_goal action"))?;
                self.state.update_goal(goal_id, Some("abandoned"), None).await?;
                Ok(format!("Goal {} marked as abandoned. It will not be re-created by automatic analysis.", goal_id))
            }
            other => Ok(format!("Unknown action: '{}'. Use list, forget, set_privacy, search, list_goals, complete_goal, or abandon_goal.", other)),
        }
    }
}
