use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::semantics::{
    ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolRole,
};

use crate::records::{
    FactPrivacy, PersonalAliasCandidate, PersonalEntityCandidate, PersonalFactCandidate,
    PersonalMemoryWrite,
};
use crate::stores::FactStore;

pub struct RememberFactTool {
    state: Arc<dyn FactStore>,
    /// Current channel_id set by the agent before tool execution.
    pub current_channel_id: Arc<RwLock<Option<String>>>,
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
    /// Forget the single `category`/`key` fact instead of storing a value.
    #[serde(default)]
    delete: bool,
    #[serde(default)]
    facts: Option<Vec<FactEntry>>,
    /// People and relationships moved to `manage_people` `remember`; a call
    /// that still sends them is answered with that and stores nothing.
    #[serde(default)]
    personal_memory: Option<PersonalMemoryWrite>,
}

/// How a call that still carries people is answered.
pub const PEOPLE_MOVED: &str = "remember_fact no longer stores people; nothing was stored. Send the same personal_memory to manage_people with action \"remember\", and any owner facts here without it.";

#[derive(Deserialize)]
struct FactEntry {
    category: String,
    key: String,
    #[serde(default)]
    value: String,
    #[serde(default)]
    delete: bool,
}

/// True when this exact call would delete at least one fact. Mirrors the
/// entry selection in `call`: a non-empty `facts` batch wins, otherwise the
/// single key/value pair. Deletion is the typed `delete` flag, never a value.
pub fn call_deletes_facts(arguments: &str) -> bool {
    let Ok(args) = serde_json::from_str::<RememberArgs>(arguments) else {
        return false;
    };
    match args.facts.filter(|facts| !facts.is_empty()) {
        Some(facts) => facts.iter().any(|entry| entry.delete),
        None => args.delete && args.key.is_some_and(|key| !key.trim().is_empty()),
    }
}

#[async_trait]
impl Tool for RememberFactTool {
    fn name(&self) -> &str {
        "remember_fact"
    }

    fn description(&self) -> &str {
        "Store or forget stable facts about the owner and their environment, not task research, generated content, goals, or schedules. Other people, family and relationships: manage_people remember."
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
                        "description": "Stored as written, including \"none\"."
                    },
                    "delete": {
                        "type": "boolean",
                        "description": "Forget category/key; value is ignored."
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
                                    "description": "Stored as written, including \"none\"."
                                },
                                "delete": {
                                    "type": "boolean",
                                    "description": "Forget this fact; value is ignored."
                                }
                            },
                            "required": ["category", "key"]
                        }
                    }
                },
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: RememberArgs = serde_json::from_str(arguments)?;
            let channel_id = self.current_channel_id.read().await.clone();

            let mut results = Vec::new();

            // An empty personal_memory block (models often send every schema
            // section) carries nothing; one with people is answered with
            // where people went, and nothing in the call is stored.
            if call_carries_people(arguments) {
                return Ok(PEOPLE_MOVED.to_string());
            }
            let facts = args.facts.filter(|facts| !facts.is_empty());
            let single_key = args.key.filter(|key| !key.trim().is_empty());

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
                let value = args.value.unwrap_or_default();
                vec![FactEntry {
                    category,
                    key,
                    value,
                    delete: args.delete,
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
                // A forget takes the delete path below, which also retracts
                // the canonical record. An empty value is reported there.
                if !supported || entry.delete || entry.value.trim().is_empty() {
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
                } else if matches!(key.as_str(), "name" | "full_name")
                    && let Some(owner) = structured.entities.first_mut()
                {
                    owner.canonical_name = entry.value.clone();
                    owner.canonical_name_confirmed = true;
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
                        retract: false,
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
                if entry.delete {
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
                if entry.value.trim().is_empty() {
                    results.push(format!(
                        "Not stored [{}] {}: value is empty (set delete: true to forget it)",
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
        .await;
        let people_moved = call_carries_people(arguments);
        output.map(|output| {
            let mut outcome = aidaemon_tools::semantics::ToolCallOutcome::from_output(output);
            if people_moved {
                outcome.metadata.outcome_status =
                    Some(aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent);
            }
            outcome
        })
    }
}

/// Whether the call still sends people (a non-empty `personal_memory`).
fn call_carries_people(arguments: &str) -> bool {
    serde_json::from_str::<RememberArgs>(arguments)
        .ok()
        .and_then(|args| args.personal_memory)
        .is_some_and(|personal| {
            !(personal.entities.is_empty()
                && personal.aliases.is_empty()
                && personal.facts.is_empty()
                && personal.relationships.is_empty())
        })
}

impl aidaemon_tools::semantics::ToolSemantics for RememberFactTool {
    fn dispatch_facts(&self) -> aidaemon_tools::semantics::ToolDispatchFacts {
        aidaemon_tools::semantics::ToolDispatchFacts {
            offering: aidaemon_tools::semantics::Offering::Core,
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
    use crate::full::tools::test_support::{ToolCalls as _, memory};
    use crate::sqlite::SqliteMemory;
    use aidaemon_tools::semantics::ToolSemantics;

    async fn setup() -> (RememberFactTool, Arc<SqliteMemory>, ()) {
        let state = memory().await;
        (RememberFactTool::new(state.clone()), state, ())
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
        assert!(
            facts
                .iter()
                .any(|f| f.key == "qa_codename" && f.value == "SYNTH-CODE-7")
        );
        assert!(
            facts
                .iter()
                .any(|f| f.key == "favorite_test_number" && f.value == "1729")
        );
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

    /// People moved to manage_people: a call that still sends them stores
    /// nothing (not even the facts beside them) and says where they went.
    #[tokio::test]
    async fn a_call_with_people_names_manage_people_and_stores_nothing() {
        let (tool, state, _db) = setup().await;
        let args = json!({
            "facts": [{"category": "preference", "key": "editor", "value": "synthetic-editor"}],
            "personal_memory": {
                "entities": [{"local_id": "zoe", "entity_type": "person", "canonical_name": "Zoe"}]
            }
        });
        let outcome = tool
            .execute(&args.to_string(), &Default::default())
            .await
            .unwrap();
        assert!(
            outcome.output.contains("manage_people"),
            "{}",
            outcome.output
        );
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent)
        );
        assert!(state.get_facts(None).await.unwrap().is_empty());
        assert!(state.get_canonical_memory_facts().await.unwrap().is_empty());
    }

    /// The owner's own profile keys still land in structured memory.
    #[tokio::test]
    async fn owner_profile_keys_still_reach_structured_memory() {
        let (tool, state, _db) = setup().await;
        let out = tool
            .call(
                &json!({"category": "user", "key": "birthday", "value": "1990-01-02"}).to_string(),
            )
            .await
            .unwrap();
        assert!(out.contains("(canonical)"), "{out}");
        assert!(
            state
                .get_canonical_memory_facts()
                .await
                .unwrap()
                .iter()
                .any(|fact| fact.category == "user" && fact.value == "1990-01-02")
        );
    }

    /// A value is stored as written: "none" is an answer, and wording such
    /// as "from now on" is not an attack. Forgetting is the typed flag.
    #[tokio::test]
    async fn values_are_stored_as_written_and_delete_is_typed() {
        let (tool, state, _db) = setup().await;
        let stored = |key: &'static str| {
            let state = state.clone();
            async move {
                state
                    .get_facts(None)
                    .await
                    .unwrap()
                    .into_iter()
                    .find(|f| f.key == key)
                    .map(|f| f.value)
            }
        };
        for (key, value) in [
            ("allergies", "none"),
            ("middle_name", "N/A"),
            ("work_schedule", "From now on I work remote Fridays"),
            ("quarter_goal", "my goal is to ship the synthetic widget"),
        ] {
            let out = tool
                .call(&json!({"category": "user", "key": key, "value": value}).to_string())
                .await
                .unwrap();
            assert!(out.contains("Remembered"), "{out}");
            assert_eq!(stored(key).await.as_deref(), Some(value));
        }

        let out = tool
            .call(&json!({"category": "user", "key": "allergies", "delete": true}).to_string())
            .await
            .unwrap();
        assert!(out.contains("Deleted"), "{out}");
        assert_eq!(stored("allergies").await, None);

        let out = tool
            .call(&json!({"category": "user", "key": "middle_name", "value": ""}).to_string())
            .await
            .unwrap();
        assert!(out.contains("Not stored"), "{out}");
        assert_eq!(stored("middle_name").await.as_deref(), Some("N/A"));
    }

    #[tokio::test]
    async fn memory_receipts_are_typed_for_checklist_credit() {
        let (tool, _state, _db) = setup().await;
        let effects = |args: Value| tool.call_semantics(&args.to_string()).mutation_effects;

        let batch_delete = effects(json!({
            "facts": [{"category": "user", "key": "qa_codename", "delete": true}],
            "key": "", "value": ""
        }));
        assert!(batch_delete.satisfies(ToolMutationEffects::DESTRUCTIVE));
        assert!(batch_delete.satisfies(ToolMutationEffects::CONFIGURATION));
        assert!(
            effects(json!({"category": "user", "key": "qa_codename", "delete": true}))
                .satisfies(ToolMutationEffects::DESTRUCTIVE)
        );
        // A value that reads like a forget is a value.
        assert!(
            !effects(json!({"category": "user", "key": "qa_codename", "value": "delete"}))
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
        // An empty key names nothing to delete.
        assert!(
            !effects(json!({"key": "", "delete": true}))
                .satisfies(ToolMutationEffects::DESTRUCTIVE)
        );
    }
}
