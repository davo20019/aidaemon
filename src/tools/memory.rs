use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::traits::{
    FactStore, PersonalAliasCandidate, PersonalEntityCandidate, PersonalFactCandidate,
    PersonalMemoryWrite, Tool, ToolCapabilities, ToolRole,
};
use crate::types::FactPrivacy;

pub struct RememberFactTool {
    state: Arc<dyn FactStore>,
    /// Current channel_id set by the agent before tool execution.
    pub(crate) current_channel_id: Arc<RwLock<Option<String>>>,
}

impl RememberFactTool {
    pub fn new(state: Arc<dyn FactStore>) -> Self {
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
    #[serde(default)]
    personal_memory: Option<PersonalMemoryWrite>,
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
            "description": "Store stable long-term user/environment facts. Exclude task-scoped research, reference data, generated content, goals, and schedules. Use facts for batches. Use personal_memory for identity, aliases, people, dates, and relationships.",
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
                        "description": "Value; empty or 'delete' removes it."
                    },
                    "facts": {
                        "type": "array",
                        "description": "Batch of legacy/general facts.",
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
                                    "description": "Value; empty or 'delete' removes it."
                                }
                            },
                            "required": ["category", "key", "value"]
                        }
                    },
                    "personal_memory": {
                        "type": "object",
                        "description": "Canonical entity-aware personal/profile memory.",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "local_id": {"type": "string"},
                                        "entity_type": {"type": "string", "enum": ["person", "organization", "project", "place", "account"]},
                                        "canonical_name": {"type": "string"},
                                        "is_reference": {"type": "boolean", "description": "Resolve existing alias/name; false declares."},
                                        "canonical_name_confirmed": {"type": "boolean"}
                                    },
                                    "required": ["local_id", "entity_type", "canonical_name"]
                                }
                            },
                            "aliases": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "entity_local_id": {"type": "string"},
                                        "value": {"type": "string"},
                                        "alias_type": {"type": "string", "enum": ["legal_name_variant", "preferred_name", "nickname", "username", "online_handle", "account_name"]}
                                    },
                                    "required": ["entity_local_id", "value", "alias_type"]
                                }
                            },
                            "facts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subject_local_id": {"type": "string"},
                                        "predicate": {"type": "string"},
                                        "value": {"type": "string"},
                                        "display_value": {"type": "string"},
                                        "valid_from": {"type": "string"},
                                        "valid_to": {"type": "string"}
                                    },
                                    "required": ["subject_local_id", "predicate", "value"]
                                }
                            },
                            "relationships": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_local_id": {"type": "string"},
                                        "relationship_type": {"type": "string", "enum": ["PARENT_OF", "CHILD_OF", "LIVES_WITH", "LIVES_IN", "USES_ALIAS", "USES_HANDLE", "HAS_ACCOUNT"]},
                                        "target_local_id": {"type": "string"},
                                        "valid_from": {"type": "string"},
                                        "valid_to": {"type": "string"}
                                    },
                                    "required": ["source_local_id", "relationship_type", "target_local_id"]
                                }
                            },
                            "direct_user_statement": {"type": "boolean"},
                            "correction": {"type": "boolean"}
                        },
                        "additionalProperties": false
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: RememberArgs = serde_json::from_str(arguments)?;
        let channel_id = self.current_channel_id.read().await.clone();

        if let Some(mut personal) = args.personal_memory {
            personal.direct_user_statement = true;
            let result = self
                .state
                .reconcile_personal_memory(
                    &personal,
                    "agent",
                    None,
                    channel_id.as_deref(),
                    FactPrivacy::Private,
                )
                .await?;
            return Ok(format!("Personal memory: {}", result.concise_summary()));
        }

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

        let mut structured = PersonalMemoryWrite {
            direct_user_statement: true,
            ..Default::default()
        };
        let mut owner_added = false;
        let mut structured_indices = std::collections::BTreeSet::new();
        for (index, entry) in entries.iter().enumerate() {
            if !matches!(
                entry.category.trim().to_ascii_lowercase().as_str(),
                "user" | "personal" | "profile"
            ) {
                continue;
            }
            let key = entry.key.trim().to_ascii_lowercase();
            let supported = matches!(
                key.as_str(),
                "name"
                    | "full_name"
                    | "preferred_name"
                    | "nickname"
                    | "username"
                    | "online_nickname"
                    | "online_handle"
                    | "handle"
                    | "birthday"
                    | "birth_date"
                    | "date_of_birth"
                    | "residence"
                    | "current_residence"
                    | "birthplace"
                    | "place_of_birth"
            );
            if !supported || entry.value.trim().is_empty() {
                continue;
            }
            if !owner_added {
                structured.entities.push(PersonalEntityCandidate {
                    local_id: "owner".to_string(),
                    entity_type: "person".to_string(),
                    canonical_name: if matches!(key.as_str(), "name" | "full_name") {
                        entry.value.clone()
                    } else {
                        "Owner".to_string()
                    },
                    is_reference: false,
                    canonical_name_confirmed: matches!(key.as_str(), "name" | "full_name"),
                });
                owner_added = true;
            } else if matches!(key.as_str(), "name" | "full_name") {
                if let Some(owner) = structured.entities.first_mut() {
                    owner.canonical_name = entry.value.clone();
                    owner.canonical_name_confirmed = true;
                }
            }
            match key.as_str() {
                "name" | "full_name" => structured.aliases.push(PersonalAliasCandidate {
                    entity_local_id: "owner".to_string(),
                    value: entry.value.clone(),
                    alias_type: "legal_name_variant".to_string(),
                }),
                "preferred_name" | "nickname" => structured.aliases.push(PersonalAliasCandidate {
                    entity_local_id: "owner".to_string(),
                    value: entry.value.clone(),
                    alias_type: key,
                }),
                "username" | "online_nickname" | "online_handle" | "handle" => {
                    structured.aliases.push(PersonalAliasCandidate {
                        entity_local_id: "owner".to_string(),
                        value: entry.value.clone(),
                        alias_type: "online_handle".to_string(),
                    })
                }
                _ => structured.facts.push(PersonalFactCandidate {
                    subject_local_id: "owner".to_string(),
                    predicate: key,
                    value: entry.value.clone(),
                    display_value: None,
                    valid_from: None,
                    valid_to: None,
                }),
            }
            structured_indices.insert(index);
        }

        let mut results = Vec::new();
        if !structured.entities.is_empty() {
            let result = self
                .state
                .reconcile_personal_memory(
                    &structured,
                    "agent",
                    None,
                    channel_id.as_deref(),
                    FactPrivacy::Private,
                )
                .await?;
            results.push(format!("Personal memory: {}", result.concise_summary()));
            for index in &structured_indices {
                let entry = &entries[*index];
                results.push(format!(
                    "Remembered: [{}] {} = {} (canonical)",
                    entry.category, entry.key, entry.value
                ));
            }
        }

        for (index, entry) in entries.iter().enumerate() {
            if structured_indices.contains(&index) {
                continue;
            }
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

            // Empty/whitespace-only value or explicit deletion markers → delete the fact
            let trimmed = entry.value.trim();
            let is_deletion = trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("none")
                || trimmed.eq_ignore_ascii_case("null")
                || trimmed.eq_ignore_ascii_case("n/a")
                || trimmed.eq_ignore_ascii_case("delete")
                || trimmed.eq_ignore_ascii_case("remove")
                || trimmed.eq_ignore_ascii_case("deleted")
                || trimmed.eq_ignore_ascii_case("removed");
            if is_deletion {
                let deleted = self
                    .state
                    .delete_fact_by_key(&entry.category, &entry.key)
                    .await?;
                if deleted {
                    results.push(format!("Deleted: [{}] {}", entry.category, entry.key));
                } else {
                    results.push(format!(
                        "Not found (nothing to delete): [{}] {}",
                        entry.category, entry.key
                    ));
                }
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

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
}
