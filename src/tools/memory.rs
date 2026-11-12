use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::traits::{
    FactStore, PersonalAliasCandidate, PersonalEntityCandidate, PersonalFactCandidate,
    PersonalMemoryWrite, Tool, ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolRole,
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
    // Word-boundary match: a substring check rejected every fact in the
    // "personal" category because it contains "persona". Keys are snake_case,
    // so split them into words first.
    let combined = format!("{} {} {}", category, key, value).replace(['_', '-'], " ");
    PERSONA_PATTERNS
        .iter()
        .any(|p| crate::agent::keyword_match(&combined, p))
}

/// Empty/whitespace-only values and explicit markers delete the fact.
fn is_deletion_marker(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.is_empty()
        || [
            "none", "null", "n/a", "delete", "remove", "deleted", "removed",
        ]
        .iter()
        .any(|marker| trimmed.eq_ignore_ascii_case(marker))
}

/// True when this exact call would delete at least one fact. Mirrors the
/// entry selection in `call`: a non-empty `facts` batch wins, otherwise the
/// single key/value pair.
pub(crate) fn call_deletes_facts(arguments: &str) -> bool {
    let Ok(args) = serde_json::from_str::<RememberArgs>(arguments) else {
        return false;
    };
    match args.facts.filter(|facts| !facts.is_empty()) {
        Some(facts) => facts.iter().any(|entry| is_deletion_marker(&entry.value)),
        None => {
            args.key.is_some_and(|key| !key.trim().is_empty())
                && args.value.as_deref().is_some_and(is_deletion_marker)
        }
    }
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
        "Store stable user/environment facts, not task research, generated content, goals, or schedules. Use personal_memory for people and relationships."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "remember_fact",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string"
                    },
                    "key": {
                        "type": "string"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value; empty or 'delete' removes it."
                    },
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string"
                                },
                                "key": {
                                    "type": "string"
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

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: RememberArgs = serde_json::from_str(arguments)?;
            let channel_id = self.current_channel_id.read().await.clone();

            let mut results = Vec::new();

            // An empty personal_memory block (models often send every schema
            // section) carries nothing to reconcile. Treating it as present used
            // to short-circuit the call and silently drop the plain facts beside
            // it while reporting "No changes (already current)".
            let personal = args.personal_memory.filter(|personal| {
                !(personal.entities.is_empty()
                    && personal.aliases.is_empty()
                    && personal.facts.is_empty()
                    && personal.relationships.is_empty())
            });
            let has_personal = personal.is_some();
            if let Some(mut personal) = personal {
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
                results.push(format!("Personal memory: {}", result.concise_summary()));
            }

            let facts = args.facts.filter(|facts| !facts.is_empty());
            let single_key = args.key.filter(|key| !key.trim().is_empty());
            if has_personal && facts.is_none() && single_key.is_none() {
                return Ok(results.join("\n"));
            }

            // Build the list of facts to store (batch or single)
            let entries: Vec<FactEntry> = if let Some(facts) = facts {
                facts
            } else {
                // Single-fact mode: require all three fields
                let category = args.category.ok_or_else(|| {
                    anyhow::anyhow!("'category' is required (or use 'facts' array)")
                })?;
                let key = single_key
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
                // A deletion marker is a forget, not a value: it takes the
                // delete path below, which also retracts the canonical record.
                if !supported || is_deletion_marker(&entry.value) {
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
                    "preferred_name" | "nickname" => {
                        structured.aliases.push(PersonalAliasCandidate {
                            entity_local_id: "owner".to_string(),
                            value: entry.value.clone(),
                            alias_type: key,
                        })
                    }
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
                if is_deletion_marker(&entry.value) {
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
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for RememberFactTool {
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            always_offered: true,
            ..Default::default()
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }

    /// Memory is persistent agent state: a write is typed `configuration`
    /// and a deletion is also `destructive`, so the receipt can prove a
    /// checklist step declared with either label. An untyped receipt could
    /// not, which made the model report a deletion it had just performed as
    /// "already deleted".
    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        if call_deletes_facts(arguments) {
            ToolCallSemantics::mutation_with(
                ToolMutationEffects::CONFIGURATION
                    .union(ToolMutationEffects::MEMORY_WRITE)
                    .union(ToolMutationEffects::DESTRUCTIVE),
            )
        } else {
            ToolCallSemantics::mutation_with(
                ToolMutationEffects::CONFIGURATION.union(ToolMutationEffects::MEMORY_WRITE),
            )
        }
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

    /// Only the agent's own memory store changes, so a decline earlier in
    /// the run never puts a fact to the owner.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects::none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolSemantics;

    async fn setup() -> (
        RememberFactTool,
        Arc<SqliteStateStore>,
        tempfile::NamedTempFile,
    ) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        (RememberFactTool::new(state.memory_stores()), state, db_file)
    }

    #[tokio::test]
    async fn empty_personal_memory_block_does_not_drop_batch_facts() {
        let (tool, state, _db) = setup().await;
        // Shape a live model sent: every schema section filled in, with the
        // real content only in the top-level `facts` array.
        let args = json!({
            "category": "preference",
            "key": "",
            "value": "",
            "facts": [
                {"category": "personal", "key": "qa_codename", "value": "SYNTH-CODE-7"},
                {"category": "preference", "key": "favorite_test_number", "value": "1729"}
            ],
            "personal_memory": {
                "aliases": [], "entities": [], "facts": [], "relationships": [],
                "correction": false, "direct_user_statement": true
            }
        });
        let out = tool.call(&args.to_string()).await.unwrap();
        assert!(!out.contains("already current"), "{out}");
        assert!(out.contains("qa_codename = SYNTH-CODE-7"), "{out}");

        let facts = state.get_facts(None).await.unwrap();
        assert!(facts
            .iter()
            .any(|f| f.key == "qa_codename" && f.value == "SYNTH-CODE-7"));
        assert!(facts
            .iter()
            .any(|f| f.key == "favorite_test_number" && f.value == "1729"));
    }

    #[tokio::test]
    async fn empty_personal_memory_and_no_facts_is_an_error_not_a_noop() {
        let (tool, _state, _db) = setup().await;
        let args = json!({
            "key": "",
            "value": "",
            "personal_memory": {"entities": [], "aliases": [], "facts": [], "relationships": []}
        });
        assert!(tool.call(&args.to_string()).await.is_err());
    }

    #[test]
    fn persona_guard_uses_word_boundaries() {
        assert!(!is_persona_manipulation(
            "personal",
            "qa_codename",
            "SYNTH-CODE-7"
        ));
        assert!(!is_persona_manipulation(
            "user",
            "favorite_drink",
            "Sharpie accents"
        ));
        assert!(is_persona_manipulation("preference", "persona", "grumpy"));
        assert!(is_persona_manipulation(
            "preference",
            "style",
            "talk like a pirate"
        ));
        assert!(is_persona_manipulation("preference", "act_as", "my lawyer"));
    }

    #[tokio::test]
    async fn memory_receipts_are_typed_for_checklist_credit() {
        let (tool, _state, _db) = setup().await;
        let effects = |args: Value| tool.call_semantics(&args.to_string()).mutation_effects;

        let batch_delete = effects(json!({
            "facts": [{"category": "user", "key": "qa_codename", "value": "delete"}],
            "key": "", "value": ""
        }));
        assert!(batch_delete.satisfies(ToolMutationEffects::DESTRUCTIVE));
        assert!(batch_delete.satisfies(ToolMutationEffects::CONFIGURATION));
        assert!(
            effects(json!({"category": "user", "key": "qa_codename", "value": ""}))
                .satisfies(ToolMutationEffects::DESTRUCTIVE)
        );

        let store = effects(json!({
            "facts": [{"category": "user", "key": "qa_codename", "value": "SYNTH-CODE-7"}],
            "key": "", "value": ""
        }));
        assert!(store.satisfies(ToolMutationEffects::CONFIGURATION));
        assert!(!store.satisfies(ToolMutationEffects::DESTRUCTIVE));
        // A plan that calls saving a fact a derived write can close, but a
        // saved fact is never a workspace file, and a config change alone is
        // not a saved fact.
        assert!(store.satisfies(ToolMutationEffects::LOCAL_DERIVED_WRITE));
        assert!(!store.satisfies(ToolMutationEffects::LOCAL_WORKSPACE_WRITE));
        assert!(
            !ToolMutationEffects::CONFIGURATION.satisfies(ToolMutationEffects::LOCAL_DERIVED_WRITE)
        );
        // An empty key/value pair names nothing to delete.
        assert!(
            !effects(json!({"key": "", "value": ""})).satisfies(ToolMutationEffects::DESTRUCTIVE)
        );
    }
}
