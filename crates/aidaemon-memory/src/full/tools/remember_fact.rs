use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::semantics::{
    ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolRole,
};

use crate::project::{MemoryScope, NoteKind, NoteSource, NoteWrite, ProjectNotes};
use crate::records::{
    FactPrivacy, PersonalAliasCandidate, PersonalEntityCandidate, PersonalFactCandidate,
    PersonalMemoryWrite,
};
use crate::stores::FactStore;
use aidaemon_core::ProjectRef;

pub struct RememberFactTool {
    state: Arc<dyn FactStore>,
    /// Current channel_id set by the agent before tool execution.
    pub current_channel_id: Arc<RwLock<Option<String>>>,
    /// Project memory: on a turn that works in a project, what the turn
    /// learns about the project is kept here instead of in `state`.
    project_notes: Option<Arc<ProjectNotes>>,
    /// The project of the one turn this instance serves, for a host whose
    /// dispatcher does not set `ToolCallContext::project`.
    turn_project: Option<ProjectRef>,
}

impl RememberFactTool {
    pub fn new(state: Arc<dyn FactStore>) -> Self {
        Self {
            state,
            current_channel_id: Arc::new(RwLock::new(None)),
            project_notes: None,
            turn_project: None,
        }
    }

    /// Keep project facts in `notes` on turns that work in a project.
    pub fn with_project_notes(mut self, notes: Arc<ProjectNotes>) -> Self {
        self.project_notes = Some(notes);
        self
    }

    /// This instance serves one turn that works in `project`.
    pub fn for_project(mut self, project: Option<ProjectRef>) -> Self {
        self.turn_project = project;
        self
    }

    /// The project the call's turn works in, when project memory is on.
    fn project(
        &self,
        facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> Option<(Arc<ProjectNotes>, ProjectRef)> {
        let notes = self.project_notes.clone()?;
        let project = facts
            .project
            .clone()
            .or_else(|| self.turn_project.clone())?;
        Some((notes, project))
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
    /// `project` or `personal`, for the single fact and as the default of
    /// `facts` entries without their own.
    #[serde(default)]
    scope: Option<String>,
}

/// The `scope` parameter, shared by the single fact and `facts` entries.
static SCOPE_SCHEMA: std::sync::LazyLock<Value> = std::sync::LazyLock::new(|| {
    json!({
        "type": "string",
        "enum": ["project", "personal"],
        "description": "In a project conversation: project (default) keeps it in the project's notes, personal keeps it for the owner. Facts about the owner are always personal."
    })
});

/// The `evidence` parameter, shared by the single fact and `facts` entries.
/// The learner reads it after the run: a quote found in the owner's own
/// message marks the fact as stated by the owner, anything else as derived
/// (see `TaskLearner`'s provenance stamping).
static EVIDENCE_SCHEMA: std::sync::LazyLock<Value> = std::sync::LazyLock::new(|| {
    json!({
        "type": "string",
        "description": "The owner's own words stating this fact, copied verbatim from their message. Omit when they did not state it (inferred, asked, negated, or from a tool, file or page)."
    })
});

/// The `evidence` of a `facts` entry: as [`EVIDENCE_SCHEMA`], described once.
static ENTRY_EVIDENCE_SCHEMA: std::sync::LazyLock<Value> =
    std::sync::LazyLock::new(|| json!({"type": "string"}));

/// The `scope` of a `facts` entry: as [`SCOPE_SCHEMA`], described once.
static ENTRY_SCOPE_SCHEMA: std::sync::LazyLock<Value> =
    std::sync::LazyLock::new(|| json!({"type": "string", "enum": ["project", "personal"]}));

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
    #[serde(default)]
    scope: Option<String>,
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
        "Store or forget stable facts about the owner and their environment, not task research, generated content, goals, schedules, or a rule the owner limits to this chat or task (it already applies here). Other people, family and relationships: manage_people remember."
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
                    "scope": SCOPE_SCHEMA.clone(),
                    "evidence": EVIDENCE_SCHEMA.clone(),
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
                                },
                                "scope": ENTRY_SCOPE_SCHEMA.clone(),
                                "evidence": ENTRY_EVIDENCE_SCHEMA.clone()
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
        call: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: RememberArgs = serde_json::from_str(arguments)?;
            let set_channel = self.current_channel_id.read().await.clone();
            let (fact_privacy, channel_id) = write_access(call, set_channel);

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
                    scope: None,
                }]
            };
            // Each entry's own scope, else the call's.
            let call_scope = args.scope.clone();
            let entries: Vec<FactEntry> = entries
                .into_iter()
                .map(|mut entry| {
                    if entry.scope.is_none() {
                        entry.scope = call_scope.clone();
                    }
                    entry
                })
                .collect();

            // On a turn in a project, what belongs to the project becomes
            // its notes and never reaches the fact store, so it cannot
            // replace or merge with any of the owner's facts.
            let project = self.project(call);
            let mut project_indices = std::collections::BTreeSet::new();
            for (index, entry) in entries.iter().enumerate() {
                let requested = match entry.scope.as_deref() {
                    None => None,
                    Some(raw) => match MemoryScope::parse(raw) {
                        Some(scope) => Some(scope),
                        None => {
                            results.push(format!(
                                "Not stored [{}] {}: scope must be \"project\" or \"personal\"",
                                entry.category, entry.key
                            ));
                            project_indices.insert(index);
                            continue;
                        }
                    },
                };
                let in_project = project.is_some();
                if requested == Some(MemoryScope::Project) && !in_project {
                    results.push(format!(
                        "Not stored [{}] {}: project memory is not available in this conversation; leave scope out to remember it for the owner",
                        entry.category, entry.key
                    ));
                    project_indices.insert(index);
                    continue;
                }
                if crate::project::route(&entry.category, requested, in_project)
                    != MemoryScope::Project
                {
                    continue;
                }
                project_indices.insert(index);
                let Some((notes, project)) = project.clone() else {
                    continue;
                };
                results.push(project_note(notes, project, entry).await);
            }

            let mut structured = PersonalMemoryWrite {
                direct_user_statement: true,
                ..Default::default()
            };
            let mut owner_added = false;
            let mut structured_indices = std::collections::BTreeSet::new();
            for (index, entry) in entries.iter().enumerate() {
                if project_indices.contains(&index)
                    || !matches!(
                        entry.category.trim().to_ascii_lowercase().as_str(),
                        "user" | "personal" | "profile"
                    )
                {
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
                if structured_indices.contains(&index) || project_indices.contains(&index) {
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
                        fact_privacy,
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

/// Writes or deletes one entry in `project`'s notes, as a result line.
async fn project_note(notes: Arc<ProjectNotes>, project: ProjectRef, entry: &FactEntry) -> String {
    let kind = crate::project::kind_for_category(&entry.category);
    let label = format!("[{}] {}", entry.category, entry.key);
    let name = project.name.clone();
    let category = entry.category.clone();
    let key = entry.key.clone();
    if entry.delete {
        let deleted = tokio::task::spawn_blocking(move || notes.delete(&project, Some(kind), &key))
            .await
            .map_err(|error| error.to_string())
            .and_then(|result| result.map_err(|error| error.to_string()));
        return match deleted {
            Ok(true) => format!("Deleted from project {name}: {label}"),
            Ok(false) => format!("Not found in project {name} (nothing to delete): {label}"),
            Err(error) => format!("Not deleted {label}: {error}"),
        };
    }
    if entry.value.trim().is_empty() {
        return format!("Not stored {label}: value is empty (set delete: true to forget it)");
    }
    let value = entry.value.clone();
    let write = NoteWrite {
        kind,
        key,
        category: Some(category),
        description: None,
        body: value.clone(),
        source: NoteSource::Agent,
    };
    let written = tokio::task::spawn_blocking(move || notes.write(&project, &write))
        .await
        .map_err(|error| error.to_string())
        .and_then(|result| result.map_err(|error| error.to_string()));
    match written {
        Ok(_) => {
            let kind = if kind == NoteKind::Fact {
                String::new()
            } else {
                format!(" ({})", kind.as_str())
            };
            format!("Remembered for project {name}{kind}: {label} = {value}")
        }
        Err(error) => format!("Not stored {label}: {error}"),
    }
}

/// Whether the call still sends people (a non-empty `personal_memory`).
/// Who may later see a fact this call stores, from the dispatcher's typed
/// channel facts: the owner's direct or internal conversation keeps it
/// general (only the owner's own turns see `Global`), a shared channel keeps
/// it in that channel, and a call with no known channel keeps it private.
fn write_access(
    facts: &aidaemon_tools::semantics::ToolCallContext,
    set_channel: Option<String>,
) -> (FactPrivacy, Option<String>) {
    use aidaemon_core::ChannelVisibility;
    let channel_id = facts.channel_id.clone().or(set_channel);
    match (facts.channel_visibility, channel_id) {
        (Some(ChannelVisibility::Private | ChannelVisibility::Internal), channel_id) => {
            (FactPrivacy::Global, channel_id)
        }
        (Some(ChannelVisibility::PrivateGroup | ChannelVisibility::Public), Some(channel_id)) => {
            (FactPrivacy::Channel, Some(channel_id))
        }
        (_, channel_id) => (FactPrivacy::Private, channel_id),
    }
}

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
            activity: aidaemon_tools::semantics::ToolActivity::Fixed(
                aidaemon_core::card::ActivityKind::Remember,
            ),
            channel_reach: aidaemon_tools::semantics::ChannelReach::Public,
            status_label: Some("updating memory"),
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
        ToolCapabilities::IDEMPOTENT_LOCAL_WRITE
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

    #[test]
    fn stored_privacy_follows_the_channel() {
        use aidaemon_core::ChannelVisibility;
        use aidaemon_tools::semantics::ToolCallContext;
        let ctx = |visibility, channel: Option<&str>| ToolCallContext {
            channel_visibility: visibility,
            channel_id: channel.map(str::to_string),
            ..Default::default()
        };
        assert_eq!(
            write_access(&ctx(Some(ChannelVisibility::Private), None), None).0,
            FactPrivacy::Global
        );
        assert_eq!(
            write_access(
                &ctx(
                    Some(ChannelVisibility::PrivateGroup),
                    Some("slack:C-synthetic")
                ),
                None
            ),
            (FactPrivacy::Channel, Some("slack:C-synthetic".to_string()))
        );
        // A shared channel without an id, or no channel facts at all, keeps
        // the fact private rather than general.
        assert_eq!(
            write_access(&ctx(Some(ChannelVisibility::Public), None), None).0,
            FactPrivacy::Private
        );
        assert_eq!(write_access(&ctx(None, None), None).0, FactPrivacy::Private);
    }

    mod project_scope {
        use super::*;
        use crate::project::{NoteKind, ProjectNotes};
        use aidaemon_tools::semantics::ToolCallContext;

        struct Fixture {
            tool: RememberFactTool,
            state: Arc<SqliteMemory>,
            notes: Arc<ProjectNotes>,
            project: ProjectRef,
            _dir: tempfile::TempDir,
        }

        async fn fixture() -> Fixture {
            let state = memory().await;
            let dir = tempfile::tempdir().unwrap();
            let notes = Arc::new(ProjectNotes::new(dir.path()));
            let tool = RememberFactTool::new(state.clone()).with_project_notes(notes.clone());
            let project = ProjectRef::new(
                "project-00000000-0000-4000-8000-000000000001",
                "Acme Site",
                None,
            )
            .unwrap();
            Fixture {
                tool,
                state,
                notes,
                project,
                _dir: dir,
            }
        }

        async fn call_in(
            tool: &RememberFactTool,
            project: Option<&ProjectRef>,
            args: Value,
        ) -> String {
            let facts = ToolCallContext {
                project: project.cloned(),
                ..Default::default()
            };
            tool.execute(&args.to_string(), &facts)
                .await
                .unwrap()
                .output
        }

        #[tokio::test]
        async fn a_project_fact_never_touches_personal_memory() {
            let f = fixture().await;
            for (category, key, value) in [
                ("user", "name", "Alice Rivera"),
                ("preference", "editor", "vim"),
                ("technical", "test_command", "pytest -q"),
            ] {
                f.state
                    .upsert_fact(category, key, value, "user", None, FactPrivacy::Global)
                    .await
                    .unwrap();
            }
            let before = f.state.get_facts(None).await.unwrap();
            // Same key as a personal fact, and a near-synonym of another.
            let out = call_in(
                &f.tool,
                Some(&f.project),
                json!({"facts": [
                    {"category": "technical", "key": "test_command", "value": "make test"},
                    {"category": "technical", "key": "editor_config", "value": "vim modelines are off here"}
                ]}),
            )
            .await;
            assert!(out.contains("Remembered for project Acme Site"), "{out}");
            let after = f.state.get_facts(None).await.unwrap();
            let ids = |facts: &[crate::records::Fact]| {
                let mut v: Vec<_> = facts.iter().map(|x| (x.id, x.value.clone())).collect();
                v.sort();
                v
            };
            assert_eq!(ids(&before), ids(&after));
            assert!(
                f.notes
                    .read(&f.project, NoteKind::Fact, "test_command")
                    .unwrap()
                    .contains("make test")
            );
            assert_eq!(f.notes.entries(&f.project).len(), 2);
        }

        #[tokio::test]
        async fn without_a_project_nothing_changes() {
            let f = fixture().await;
            let out = call_in(
                &f.tool,
                None,
                json!({"category": "technical", "key": "test_command", "value": "make test"}),
            )
            .await;
            assert_eq!(out, "Remembered: [technical] test_command = make test");
            assert!(f.notes.entries(&f.project).is_empty());
            let out = call_in(
                &f.tool,
                None,
                json!({"category": "technical", "key": "db", "value": "Postgres", "scope": "project"}),
            )
            .await;
            assert!(out.contains("project memory is not available"), "{out}");
            assert!(
                f.state
                    .get_facts(None)
                    .await
                    .unwrap()
                    .iter()
                    .all(|x| x.key != "db")
            );
        }

        #[tokio::test]
        async fn the_owner_stays_personal_in_a_project() {
            let f = fixture().await;
            let out = call_in(
                &f.tool,
                Some(&f.project),
                json!({"facts": [
                    {"category": "user", "key": "residence", "value": "Springfield", "scope": "project"},
                    {"category": "preference", "key": "indentation", "value": "tabs"},
                    {"category": "technical", "key": "laptop", "value": "a synthetic laptop", "scope": "personal"}
                ]}),
            )
            .await;
            assert!(out.contains("canonical"), "{out}");
            let facts = f.state.get_facts(None).await.unwrap();
            assert!(facts.iter().any(|x| x.key == "indentation"), "{facts:?}");
            assert!(facts.iter().any(|x| x.key == "laptop"), "{facts:?}");
            assert!(f.notes.entries(&f.project).is_empty());
        }

        #[tokio::test]
        async fn project_notes_are_deleted_by_key_and_bad_scopes_store_nothing() {
            let f = fixture().await;
            call_in(
                &f.tool,
                Some(&f.project),
                json!({"category": "pitfall", "key": "migrations", "value": "Run migrations before seeding."}),
            )
            .await;
            assert!(
                f.notes
                    .read(&f.project, NoteKind::Pitfall, "migrations")
                    .is_some()
            );
            let out = call_in(
                &f.tool,
                Some(&f.project),
                json!({"category": "pitfall", "key": "migrations", "delete": true}),
            )
            .await;
            assert!(out.starts_with("Deleted from project"), "{out}");
            assert!(f.notes.entries(&f.project).is_empty());
            let out = call_in(
                &f.tool,
                Some(&f.project),
                json!({"category": "technical", "key": "db", "value": "Postgres", "scope": "team"}),
            )
            .await;
            assert!(out.contains("scope must be"), "{out}");
            assert!(f.notes.entries(&f.project).is_empty());
            assert!(f.state.get_facts(None).await.unwrap().is_empty());
        }

        #[tokio::test]
        async fn a_tool_without_notes_keeps_todays_behaviour_in_a_project() {
            let state = memory().await;
            let tool = RememberFactTool::new(state.clone());
            let project = ProjectRef::new("project-x", "X", None).unwrap();
            let out = call_in(
                &tool,
                Some(&project),
                json!({"category": "technical", "key": "db", "value": "Postgres"}),
            )
            .await;
            assert_eq!(out, "Remembered: [technical] db = Postgres");
        }
    }
}
