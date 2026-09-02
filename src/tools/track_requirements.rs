//! `track_requirements` — model-facing tool to register and update the durable
//! checklist of requirements for the current multi-step / deferred-action turn.
//! Full-set replace each call (like a todo tool). Backed by the `plans/` store.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::events::{ExecutorExpectationItem, ExecutorExpectationsDeclaredData};
use crate::plans::{ChecklistItem, PlanStore, StepStatus};
use crate::traits::{
    RequestObservationTarget, RequestVerificationTargetKind, StableObservationSubject, Tool,
    ToolArgumentContractViolation, ToolCallSemantics, ToolCapabilities, ToolMutationEffects,
    ToolSemanticFacet,
};

/// Item-level key under which canonicalization records resource-ID targets
/// that no registered tool advertises. Kept on the canonical arguments so the
/// persisted checklist and the durable expectation event see the same drop.
pub const UNBOUND_TARGETS_KEY: &str = "_unbound_targets";

/// Start of the user-facing render in this tool's output. The agent loop
/// locates it to lift the checklist onto the live status surface, so any
/// model-only notice must precede it.
pub const CHECKLIST_HEADER: &str = "Checklist updated";

/// Union of the exact receipt subjects every registered tool advertises via
/// `Tool::stable_observation_subjects`. Checklist resource-ID targets are bound
/// against it: a value outside every advertised ID and member namespace can
/// never be credited by a receipt, so it is never compiled into an obligation.
#[derive(Debug, Clone, Default)]
pub struct StableSubjectVocabulary {
    subjects: Vec<StableObservationSubject>,
}

impl StableSubjectVocabulary {
    pub fn from_tools(tools: &[Arc<dyn Tool>]) -> Self {
        Self::from_subjects(
            tools
                .iter()
                .flat_map(|tool| tool.stable_observation_subjects())
                .collect(),
        )
    }

    pub fn from_subjects(subjects: Vec<StableObservationSubject>) -> Self {
        let mut merged: Vec<StableObservationSubject> = Vec::new();
        for subject in subjects {
            let existing = match subject.resource_id.as_deref() {
                Some(resource_id) => merged
                    .iter_mut()
                    .find(|known| known.resource_id.as_deref() == Some(resource_id)),
                None => merged.iter_mut().find(|known| {
                    known.resource_id.is_none()
                        && known.member_namespaces == subject.member_namespaces
                }),
            };
            match existing {
                Some(known) => {
                    for facet in subject.facets {
                        if !known.facets.contains(&facet) {
                            known.facets.push(facet);
                        }
                    }
                    for namespace in subject.member_namespaces {
                        if !known.member_namespaces.contains(&namespace) {
                            known.member_namespaces.push(namespace);
                        }
                    }
                }
                None => merged.push(subject),
            }
        }
        Self { subjects: merged }
    }

    pub fn is_empty(&self) -> bool {
        self.subjects.is_empty()
    }

    /// Whether some registered tool can report `value` verbatim.
    pub fn binds(&self, value: &str) -> bool {
        self.subjects.iter().any(|subject| subject.binds(value))
    }

    fn collections(&self) -> impl Iterator<Item = &StableObservationSubject> {
        self.subjects
            .iter()
            .filter(|subject| subject.resource_id.is_some())
    }

    fn member_namespaces(&self) -> Vec<&str> {
        let mut namespaces: Vec<&str> = Vec::new();
        for subject in &self.subjects {
            for namespace in &subject.member_namespaces {
                if !namespaces.contains(&namespace.as_str()) {
                    namespaces.push(namespace);
                }
            }
        }
        namespaces
    }

    /// Schema-time description of what the model may bind to.
    fn render_schema_hint(&self) -> String {
        let mut hint = String::new();
        let collections: Vec<String> = self
            .collections()
            .map(|subject| {
                let facets = subject
                    .facets
                    .iter()
                    .map(|facet| facet.as_str())
                    .collect::<Vec<_>>()
                    .join("/");
                let resource_id = subject.resource_id.as_deref().unwrap_or_default();
                if facets.is_empty() {
                    format!("{resource_id} ({})", subject.summary)
                } else {
                    format!("{resource_id} ({}; facets {facets})", subject.summary)
                }
            })
            .collect();
        if !collections.is_empty() {
            hint.push_str(" Stable collection subjects: ");
            hint.push_str(&collections.join("; "));
            hint.push('.');
        }
        let namespaces = self.member_namespaces();
        if !namespaces.is_empty() {
            hint.push_str(" Exact member IDs start with ");
            hint.push_str(&namespaces.join(", "));
            hint.push_str(" and come from tool output.");
        }
        hint.push_str(
            " Any other resource ID is dropped from its item (the item becomes an untargeted \
             observation) and reported back.",
        );
        hint
    }

    /// Call-time notice naming what was dropped and what binds instead.
    fn render_notice(&self, dropped: &[(usize, Vec<String>)]) -> String {
        let mut lines = Vec::with_capacity(dropped.len() + 2);
        for (index, values) in dropped {
            lines.push(format!(
                "Item {index}: no tool reports {}; kept as an untargeted observation.",
                values
                    .iter()
                    .map(|value| format!("`{value}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let collections = self
            .collections()
            .filter_map(|subject| subject.resource_id.as_deref())
            .collect::<Vec<_>>();
        if !collections.is_empty() {
            lines.push(format!(
                "Bindable collection subjects are {}.",
                collections.join(", ")
            ));
        }
        let namespaces = self.member_namespaces();
        if !namespaces.is_empty() {
            lines.push(format!(
                "Exact member IDs start with {} and must be taken from tool output.",
                namespaces.join(", ")
            ));
        }
        lines.join("\n")
    }
}

pub struct TrackRequirementsTool {
    plan_store: Arc<PlanStore>,
    vocabulary: StableSubjectVocabulary,
}

impl TrackRequirementsTool {
    pub fn new(plan_store: Arc<PlanStore>) -> Self {
        Self {
            plan_store,
            vocabulary: StableSubjectVocabulary::default(),
        }
    }

    /// Bind checklist resource-ID targets against the subjects these tools
    /// advertise. With an empty vocabulary every resource ID stays exact.
    pub fn with_stable_subjects(mut self, vocabulary: StableSubjectVocabulary) -> Self {
        self.vocabulary = vocabulary;
        self
    }

    fn targets_description(&self) -> String {
        let mut description = String::from(
            "Exact receipt evidence targets. Prefer a structured subject/facet binding and never \
             invent a resource ID; use an ID returned by a tool.",
        );
        if !self.vocabulary.is_empty() {
            description.push_str(&self.vocabulary.render_schema_hint());
        }
        description.push_str(
            " Legacy strings remain exact: absolute paths and URLs retain their type; every other \
             value is an opaque resource ID that must exactly match a receipt.",
        );
        description
    }
}

fn parse_status(s: &str) -> StepStatus {
    StepStatus::from_str(s).unwrap_or(StepStatus::Pending)
}

fn unbound_targets_of(item: &Value) -> Vec<String> {
    item.get(UNBOUND_TARGETS_KEY)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Rewrite checklist arguments so every resource-ID target is one a registered
/// tool can report. Unbound values move from `targets` into
/// `_unbound_targets` on the same item; paths, URLs, structurally invalid
/// targets, and everything else pass through untouched (validation still
/// rejects the invalid ones). Idempotent, and the identity when `vocabulary`
/// is empty.
pub(crate) fn bind_checklist_targets(
    arguments: &str,
    vocabulary: &StableSubjectVocabulary,
) -> Result<String, ToolArgumentContractViolation> {
    if vocabulary.is_empty() {
        return Ok(arguments.to_string());
    }
    let mut value: Value = serde_json::from_str(arguments).map_err(|error| {
        ToolArgumentContractViolation::new(format!("invalid JSON arguments: {error}"))
    })?;
    let Some(items) = value.get_mut("items").and_then(Value::as_array_mut) else {
        return Ok(arguments.to_string());
    };
    let mut changed = false;
    for item in items.iter_mut() {
        let Some(targets) = item.get("targets").and_then(Value::as_array).cloned() else {
            continue;
        };
        let mut kept = Vec::with_capacity(targets.len());
        let mut unbound = unbound_targets_of(item);
        for raw in targets {
            // Structurally invalid targets stay in place so validation still
            // rejects them; only well-formed, uncreditable IDs are dropped.
            let Ok(target) = parse_expectation_target(&raw) else {
                kept.push(raw);
                continue;
            };
            let subject_bound = target.subject.kind != RequestVerificationTargetKind::ResourceId
                || vocabulary.binds(&target.subject.value);
            let collection_bound = target.collection_coverage.as_ref().is_none_or(|coverage| {
                coverage.collection.kind != RequestVerificationTargetKind::ResourceId
                    || vocabulary.binds(&coverage.collection.value)
            });
            if subject_bound && collection_bound {
                kept.push(raw);
            } else {
                changed = true;
                if !unbound.contains(&target.subject.value) {
                    unbound.push(target.subject.value.clone());
                }
            }
        }
        if changed {
            item["targets"] = Value::Array(kept);
            if !unbound.is_empty() {
                item[UNBOUND_TARGETS_KEY] =
                    Value::Array(unbound.into_iter().map(Value::String).collect());
            }
        }
    }
    if !changed {
        return Ok(arguments.to_string());
    }
    serde_json::to_string(&value).map_err(|error| {
        ToolArgumentContractViolation::new(format!("failed to re-serialize arguments: {error}"))
    })
}

/// Project a `track_requirements` call into the executor's typed expectation
/// declaration. Only typed content survives: effects, the observation flag,
/// exact targets, the recorded unbound targets, and the declared status.
pub(crate) fn executor_expectations_from_checklist_arguments(
    task_id: &str,
    arguments: &str,
) -> Option<ExecutorExpectationsDeclaredData> {
    let value: Value = serde_json::from_str(arguments).ok()?;
    let raw_items = value.get("items")?.as_array()?;
    let mut items = Vec::with_capacity(raw_items.len());
    for (index, item) in raw_items.iter().enumerate() {
        let description = item.get("text")?.as_str()?.to_string();
        let mut mutation_effects = ToolMutationEffects::NONE;
        if let Some(effects) = item.get("mutation_effects").and_then(Value::as_array) {
            for effect in effects.iter().filter_map(Value::as_str) {
                if let Some(effect) = ToolMutationEffects::from_protocol_name(effect) {
                    mutation_effects = mutation_effects.union(effect);
                }
            }
        }
        let (targets, observation_targets) = parse_expectation_targets(item.get("targets")).ok()?;
        items.push(ExecutorExpectationItem {
            index,
            description,
            requires_observation: item
                .get("requires_observation")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            mutation_effects,
            targets,
            observation_targets,
            unbound_targets: unbound_targets_of(item),
            status: item
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending")
                .to_string(),
        });
    }
    (!items.is_empty()).then(|| ExecutorExpectationsDeclaredData {
        schema_version: ExecutorExpectationsDeclaredData::SCHEMA_VERSION,
        task_id: task_id.to_string(),
        items,
    })
}

/// Decode checklist evidence targets once for both persistence and the durable
/// executor-expectation event. String targets retain the original path/URL
/// behavior; every other string is an exact opaque resource ID and therefore
/// cannot silently degrade into an untargeted observation.
pub(crate) fn parse_expectation_targets(
    raw_targets: Option<&Value>,
) -> Result<(Vec<String>, Vec<RequestObservationTarget>), String> {
    let Some(raw_targets) = raw_targets else {
        return Ok((Vec::new(), Vec::new()));
    };
    let values = raw_targets
        .as_array()
        .ok_or_else(|| "targets must be an array".to_string())?;
    let mut legacy_targets = Vec::with_capacity(values.len());
    let mut observation_targets = Vec::with_capacity(values.len());
    for value in values {
        let target = parse_expectation_target(value)?;
        legacy_targets.push(target.subject.value.clone());
        observation_targets.push(target);
    }
    Ok((legacy_targets, observation_targets))
}

fn parse_expectation_target(value: &Value) -> Result<RequestObservationTarget, String> {
    let target = if let Some(value) = value.as_str() {
        RequestObservationTarget::from_legacy_exact(value)
            .ok_or_else(|| "target strings must not be empty".to_string())?
    } else {
        serde_json::from_value::<RequestObservationTarget>(value.clone())
            .map_err(|error| format!("invalid structured target: {error}"))?
    };
    if target.subject.value.trim().is_empty() {
        return Err("target subject must not be empty".to_string());
    }
    if let Some(coverage) = target.collection_coverage.as_ref() {
        if coverage.collection.value.trim().is_empty() {
            return Err("collection target must not be empty".to_string());
        }
        if target.facets.is_empty() {
            return Err(
                "collection-backed targets must declare at least one semantic facet".to_string(),
            );
        }
    }
    Ok(target)
}

#[async_trait]
impl Tool for TrackRequirementsTool {
    fn name(&self) -> &str {
        "track_requirements"
    }

    fn description(&self) -> &str {
        "Declare the typed checklist of what the current request requires, BEFORE your first \
         operation whenever the request names two or more distinct operations, deliverables, or \
         targets (an ordered sequence like create/write/read/remove/verify is one item per step). \
         Give each item its exact targets (paths/URLs) and mutation_effects, or requires_observation \
         for reads/verifications. The runtime compiles typed items into obligations closed only by \
         real tool receipts and will keep asking for open reachable items, so declare exactly what \
         you will do and do all of it. Pass the FULL list every time; it replaces the previous list. \
         Mark items 'completed' as you finish them, or 'deferred' if you intentionally skip one. \
         Skip this only for a single direct operation or when the user forbids it."
    }

    fn schema(&self) -> Value {
        let exact_target = json!({
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["url", "path", "resource_id"]
                },
                "value": { "type": "string", "minLength": 1 }
            },
            "required": ["kind", "value"],
            "additionalProperties": false
        });
        json!({
            "name": "track_requirements",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "The full ordered checklist of requirements for this request.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "One concrete requirement."
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "deferred"]
                                },
                                "note": {
                                    "type": "string"
                                },
                                "depends_on": {
                                    "type": "array",
                                    "description": "Zero-based prerequisite indices.",
                                    "items": { "type": "integer", "minimum": 0 },
                                    "uniqueItems": true
                                },
                                "mutation_effects": {
                                    "type": "array",
                                    "description": "Typed outcomes required for receipt-based completion.",
                                    "items": {
                                        "type": "string",
                                        "enum": ToolMutationEffects::PROTOCOL_NAMES
                                    },
                                    "uniqueItems": true
                                },
                                "requires_observation": {
                                    "type": "boolean",
                                    "description": "Require an observation receipt."
                                },
                                "targets": {
                                    "type": "array",
                                    "description": self.targets_description(),
                                    "items": {
                                        "oneOf": [
                                            { "type": "string", "minLength": 1 },
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "subject": exact_target.clone(),
                                                    "facets": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "string",
                                                            "enum": ToolSemanticFacet::PROTOCOL_NAMES
                                                        },
                                                        "uniqueItems": true
                                                    },
                                                    "collection_coverage": {
                                                        "type": "object",
                                                        "description": "Require coverage of an exact collection. Set subject equal to collection to require the whole collection; otherwise complete coverage can prove exact membership or non-membership, while partial coverage cannot prove absence.",
                                                        "properties": {
                                                            "collection": exact_target,
                                                            "minimum_completeness": {
                                                                "type": "string",
                                                                "enum": ["complete"]
                                                            }
                                                        },
                                                        "required": ["collection", "minimum_completeness"],
                                                        "additionalProperties": false
                                                    }
                                                },
                                                "required": ["subject", "facets"],
                                                "additionalProperties": false
                                            }
                                        ]
                                    },
                                    "uniqueItems": true
                                }
                            },
                            "required": ["text", "status"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": false
            }
        })
    }

    fn canonicalize_arguments(
        &self,
        arguments: &str,
    ) -> Result<String, ToolArgumentContractViolation> {
        bind_checklist_targets(arguments, &self.vocabulary)
    }

    fn validate_arguments(&self, arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        let value: Value = serde_json::from_str(arguments).map_err(|error| {
            ToolArgumentContractViolation::new(format!("invalid JSON arguments: {error}"))
        })?;
        if let Some(items) = value.get("items").and_then(Value::as_array) {
            for (index, item) in items.iter().enumerate() {
                parse_expectation_targets(item.get("targets")).map_err(|error| {
                    ToolArgumentContractViolation::new(format!(
                        "item {index} has invalid evidence targets: {error}"
                    ))
                })?;
            }
        }
        Ok(())
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let v: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
        let session_id = match v.get("_session_id").and_then(Value::as_str) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => {
                return Ok(
                    "track_requirements: no session context available; checklist not saved."
                        .to_string(),
                )
            }
        };
        let task_id = v
            .get("_task_id")
            .and_then(Value::as_str)
            .map(|s| s.to_string());

        let Some(raw_items) = v.get("items").and_then(Value::as_array) else {
            return Ok("track_requirements: no items provided.".to_string());
        };
        let mut items = Vec::with_capacity(raw_items.len());
        let mut dropped_targets: Vec<(usize, Vec<String>)> = Vec::new();
        for (item_index, item) in raw_items.iter().enumerate() {
            let unbound = unbound_targets_of(item);
            if !unbound.is_empty() {
                dropped_targets.push((item_index, unbound));
            }
            let Some(text) = item.get("text").and_then(Value::as_str) else {
                return Ok(format!(
                    "track_requirements: item {item_index} is missing text."
                ));
            };
            let status = item
                .get("status")
                .and_then(Value::as_str)
                .map(parse_status)
                .unwrap_or(StepStatus::Pending);
            let depends_on = match item.get("depends_on") {
                None => Vec::new(),
                Some(Value::Array(values)) => {
                    let Some(indices) = values
                        .iter()
                        .map(|value| value.as_u64().and_then(|n| usize::try_from(n).ok()))
                        .collect::<Option<Vec<_>>>()
                    else {
                        return Ok(format!(
                            "track_requirements: item {item_index} has an invalid dependency index."
                        ));
                    };
                    indices
                }
                Some(_) => {
                    return Ok(format!(
                        "track_requirements: item {item_index} dependencies must be an array."
                    ))
                }
            };
            let mut required_mutation_effects = ToolMutationEffects::NONE;
            if let Some(raw_effects) = item.get("mutation_effects") {
                let Some(values) = raw_effects.as_array() else {
                    return Ok(format!(
                        "track_requirements: item {item_index} mutation_effects must be an array."
                    ));
                };
                for value in values {
                    let Some(effect) = value
                        .as_str()
                        .and_then(ToolMutationEffects::from_protocol_name)
                    else {
                        return Ok(format!(
                            "track_requirements: item {item_index} has an unknown mutation effect."
                        ));
                    };
                    required_mutation_effects = required_mutation_effects.union(effect);
                }
            }
            let expected_targets = match parse_expectation_targets(item.get("targets")) {
                Ok((targets, _)) => targets,
                Err(error) => {
                    return Ok(format!(
                    "track_requirements: item {item_index} has invalid evidence targets: {error}."
                ))
                }
            };
            items.push(ChecklistItem {
                description: text.to_string(),
                status,
                depends_on,
                required_mutation_effects,
                requires_observation: item
                    .get("requires_observation")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                expected_targets,
            });
        }

        if items.is_empty() {
            return Ok("track_requirements: no items provided.".to_string());
        }

        let plan = match self
            .plan_store
            .upsert_checklist_graph(
                &session_id,
                task_id.as_deref(),
                "track_requirements",
                &items,
            )
            .await
        {
            Ok(p) => p,
            Err(e) => {
                // Graceful degradation: never break the loop on a storage error.
                tracing::warn!(error = %e, "track_requirements: failed to persist checklist");
                return Ok("Checklist noted (not persisted).".to_string());
            }
        };

        // The rendered checklist is returned to the agent loop, which surfaces it
        // to the user via the single live status surface (StatusUpdate::Checklist).
        // This tool no longer posts/edits channel messages directly. Anything
        // meant only for the model goes before the "Checklist updated" header,
        // which is where the loop starts reading the user-facing render.
        let mut output = String::new();
        if !dropped_targets.is_empty() {
            output.push_str(&self.vocabulary.render_notice(&dropped_targets));
            output.push_str("\n\n");
        }
        output.push_str(&format!(
            "{CHECKLIST_HEADER} ({}/{} done):\n{}",
            plan.completed_steps(),
            plan.steps.len(),
            plan.render_compact_checklist()
        ));
        Ok(output)
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        // Checklist bookkeeping cannot prove a requested mutation occurred.
        ToolCallSemantics::administrative()
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

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn test_tool() -> (TrackRequirementsTool, Arc<PlanStore>) {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let plan_store = Arc::new(PlanStore::new(pool).await.unwrap());
        // The tool only persists checklist state; UI delivery is handled by the
        // agent loop via StatusUpdate::Checklist, so there is no hub to wire here.
        (TrackRequirementsTool::new(plan_store.clone()), plan_store)
    }

    #[tokio::test]
    async fn checklist_bookkeeping_is_not_a_contract_mutation() {
        // Live 2026-07-12: a track_requirements call incremented
        // completion_progress.mutation_count, which satisfied the "did the
        // requested mutation happen?" gates for a "Send me my resume" turn —
        // letting a tool-output paste ship in place of the missing send_file.
        // The checklist is internal bookkeeping: Administrative, not Mutation.
        let (tool, _plan_store) = test_tool().await;
        let semantics = tool.call_semantics(r#"{"items":[{"text":"send the file"}]}"#);
        assert!(
            !semantics.mutates_state(),
            "checklist writes must not count as contract mutations"
        );
    }

    #[test]
    fn legacy_free_form_target_is_an_exact_resource_not_untargeted() {
        let raw = json!(["opaque-subject-that-no-adapter-reported"]);
        let (legacy, typed) = parse_expectation_targets(Some(&raw)).unwrap();
        assert_eq!(legacy, vec!["opaque-subject-that-no-adapter-reported"]);
        assert_eq!(typed.len(), 1);
        assert_eq!(
            typed[0].subject.kind,
            crate::traits::RequestVerificationTargetKind::ResourceId
        );
        assert_eq!(
            typed[0].subject.value,
            "opaque-subject-that-no-adapter-reported"
        );
    }

    #[test]
    fn structured_target_retains_subject_facet_and_complete_collection() {
        let raw = json!([{
            "subject": {"kind": "resource_id", "value": "goal:synthetic-goal-1"},
            "facets": ["schedule", "recovery"],
            "collection_coverage": {
                "collection": {
                    "kind": "resource_id",
                    "value": "goal_collection:scheduled_goals"
                },
                "minimum_completeness": "complete"
            }
        }]);
        let (_, typed) = parse_expectation_targets(Some(&raw)).unwrap();
        assert_eq!(
            typed[0].facets,
            vec![
                crate::traits::ToolSemanticFacet::Schedule,
                crate::traits::ToolSemanticFacet::Recovery
            ]
        );
        assert_eq!(
            typed[0]
                .collection_coverage
                .as_ref()
                .map(|coverage| coverage.collection.value.as_str()),
            Some("goal_collection:scheduled_goals")
        );
    }

    #[test]
    fn collection_target_without_facets_is_rejected() {
        let raw = json!([{
            "subject": {"kind": "resource_id", "value": "goal:synthetic-goal-1"},
            "facets": [],
            "collection_coverage": {
                "collection": {
                    "kind": "resource_id",
                    "value": "goal_collection:scheduled_goals"
                },
                "minimum_completeness": "complete"
            }
        }]);
        assert!(parse_expectation_targets(Some(&raw)).is_err());
    }

    #[tokio::test]
    async fn test_track_requirements_persists_and_returns_rendered_list() {
        let (tool, plan_store) = test_tool().await;
        let args = json!({
            "_session_id": "sess-1",
            "items": [
                {"text": "create script", "status": "in_progress"},
                {"text": "send the file", "status": "pending"}
            ]
        })
        .to_string();
        let out = tool.call(&args).await.unwrap();
        assert!(out.contains("create script"));
        assert!(out.contains("send the file"));
        let plan = plan_store
            .get_incomplete_for_session("sess-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.unchecked_steps().len(), 2);
    }

    #[tokio::test]
    async fn test_track_requirements_marks_completed_on_update() {
        let (tool, plan_store) = test_tool().await;
        let mk = |status: &str| {
            json!({
                "_session_id": "sess-2",
                "items": [{"text": "send the file", "status": status}]
            })
            .to_string()
        };
        tool.call(&mk("pending")).await.unwrap();
        // While still pending it is the session's active (incomplete) checklist.
        let id = plan_store
            .get_incomplete_for_session("sess-2")
            .await
            .unwrap()
            .unwrap()
            .id;

        tool.call(&mk("completed")).await.unwrap();
        // Once all items are completed the plan is auto-completed (no longer
        // "incomplete"), so fetch it by id to verify the final state.
        let plan = plan_store.get(&id).await.unwrap().unwrap();
        assert_eq!(plan.unchecked_steps().len(), 0);
        assert_eq!(plan.completed_steps(), 1);
    }

    #[tokio::test]
    async fn test_track_requirements_missing_session_errs_gracefully() {
        let (tool, _ps) = test_tool().await;
        let args = json!({"items": []}).to_string();
        let out = tool.call(&args).await.unwrap();
        assert!(out.to_lowercase().contains("session"));
    }

    fn synthetic_vocabulary() -> StableSubjectVocabulary {
        StableSubjectVocabulary::from_subjects(vec![
            StableObservationSubject::collection(
                "objective_collection:scheduled_goals",
                &[ToolSemanticFacet::Schedule, ToolSemanticFacet::RunState],
                "objective:",
                "every scheduled goal",
            ),
            StableObservationSubject::collection(
                "objective_collection:mandate_controllers",
                &[ToolSemanticFacet::Control],
                "objective:",
                "every mandate controller",
            ),
            StableObservationSubject::collection(
                "auth_profile_collection:configured",
                &[ToolSemanticFacet::Authorization],
                "auth_profile:",
                "every configured API auth profile",
            ),
            StableObservationSubject::namespace("res_", "session resource handles"),
        ])
    }

    /// The live 2026-09-01 audit shape: two items bound to advertised
    /// collections and a third bound to an invented credential-state ID that
    /// no adapter reports.
    fn audit_checklist_arguments(status: &str) -> Value {
        json!({
            "_session_id": "sess-audit",
            "_task_id": "task-audit",
            "items": [
                {
                    "text": "Audit the schedule state and latest run outcome.",
                    "status": status,
                    "requires_observation": true,
                    "targets": [{
                        "subject": {"kind": "resource_id", "value": "objective_collection:scheduled_goals"},
                        "facets": ["schedule", "run_state"],
                        "collection_coverage": {
                            "collection": {"kind": "resource_id", "value": "objective_collection:scheduled_goals"},
                            "minimum_completeness": "complete"
                        }
                    }]
                },
                {
                    "text": "Audit objective-control presence.",
                    "status": status,
                    "requires_observation": true,
                    "targets": [{
                        "subject": {"kind": "resource_id", "value": "objective_collection:mandate_controllers"},
                        "facets": ["control"],
                        "collection_coverage": {
                            "collection": {"kind": "resource_id", "value": "objective_collection:mandate_controllers"},
                            "minimum_completeness": "complete"
                        }
                    }]
                },
                {
                    "text": "Audit owner configuration and credential readiness.",
                    "status": status,
                    "requires_observation": true,
                    "targets": ["x_owner_credential_state"]
                }
            ]
        })
    }

    #[tokio::test]
    async fn unbound_resource_id_target_is_dropped_and_reported() {
        let (tool, plan_store) = test_tool().await;
        let tool = tool.with_stable_subjects(synthetic_vocabulary());
        let raw = audit_checklist_arguments("in_progress").to_string();

        let canonical = tool.canonicalize_arguments(&raw).unwrap();
        let canonical_value: Value = serde_json::from_str(&canonical).unwrap();
        let items = canonical_value["items"].as_array().unwrap();
        assert_eq!(
            items[0]["targets"],
            audit_checklist_arguments("in_progress")["items"][0]["targets"],
            "an advertised collection binding must survive untouched"
        );
        assert_eq!(items[1]["targets"].as_array().unwrap().len(), 1);
        assert_eq!(
            items[2]["targets"].as_array().map(Vec::len),
            Some(0),
            "an invented resource ID can never be credited, so it must not become an exact obligation"
        );
        assert_eq!(
            items[2][UNBOUND_TARGETS_KEY],
            json!(["x_owner_credential_state"]),
            "the drop must be recorded, never silent"
        );

        let out = tool.call(&canonical).await.unwrap();
        assert!(
            out.contains("x_owner_credential_state"),
            "the model must be told which target was unbound: {out}"
        );
        assert!(
            out.contains("auth_profile_collection:configured") && out.contains("objective:"),
            "the notice must name the stable subjects the model can bind instead: {out}"
        );
        assert!(
            out.contains("Checklist updated (0/3 done):\n"),
            "the checklist header must remain the live-surface delimiter: {out}"
        );
        let plan = plan_store
            .get_incomplete_for_session("sess-audit")
            .await
            .unwrap()
            .unwrap();
        assert!(plan.steps[2].expected_targets.is_empty());
        assert_eq!(
            plan.steps[0].expected_targets,
            ["objective_collection:scheduled_goals"]
        );
    }

    #[test]
    fn advertised_subjects_and_namespaced_members_stay_exact() {
        let vocabulary = synthetic_vocabulary();
        let raw = json!({
            "items": [{
                "text": "bind exact identities",
                "status": "pending",
                "requires_observation": true,
                "targets": [
                    "auth_profile:twitter",
                    "objective:sha256:0123abcd",
                    "res_0123abcd",
                    "/tmp/synthetic/report.md",
                    "https://example.test/status",
                    {
                        "subject": {"kind": "resource_id", "value": "objective:sha256:0123abcd"},
                        "facets": ["schedule"],
                        "collection_coverage": {
                            "collection": {"kind": "resource_id", "value": "objective_collection:scheduled_goals"},
                            "minimum_completeness": "complete"
                        }
                    }
                ]
            }]
        })
        .to_string();
        let canonical = bind_checklist_targets(&raw, &vocabulary).unwrap();
        let value: Value = serde_json::from_str(&canonical).unwrap();
        assert_eq!(value["items"][0]["targets"].as_array().unwrap().len(), 6);
        assert!(value["items"][0].get(UNBOUND_TARGETS_KEY).is_none());
    }

    #[test]
    fn bare_namespace_prefix_and_unknown_collection_are_unbound() {
        let vocabulary = synthetic_vocabulary();
        let raw = json!({
            "items": [{
                "text": "bind exact identities",
                "status": "pending",
                "requires_observation": true,
                "targets": [
                    "auth_profile:",
                    {
                        "subject": {"kind": "resource_id", "value": "objective:sha256:0123abcd"},
                        "facets": ["schedule"],
                        "collection_coverage": {
                            "collection": {"kind": "resource_id", "value": "goal_collection:invented"},
                            "minimum_completeness": "complete"
                        }
                    }
                ]
            }]
        })
        .to_string();
        let canonical = bind_checklist_targets(&raw, &vocabulary).unwrap();
        let value: Value = serde_json::from_str(&canonical).unwrap();
        assert_eq!(value["items"][0]["targets"], json!([]));
        assert_eq!(
            value["items"][0][UNBOUND_TARGETS_KEY],
            json!(["auth_profile:", "objective:sha256:0123abcd"])
        );
    }

    #[test]
    fn binding_is_idempotent_and_keeps_the_unbound_record() {
        let vocabulary = synthetic_vocabulary();
        let raw = audit_checklist_arguments("pending").to_string();
        let once = bind_checklist_targets(&raw, &vocabulary).unwrap();
        let twice = bind_checklist_targets(&once, &vocabulary).unwrap();
        assert_eq!(once, twice);
        let value: Value = serde_json::from_str(&twice).unwrap();
        assert_eq!(
            value["items"][2][UNBOUND_TARGETS_KEY],
            json!(["x_owner_credential_state"])
        );
    }

    #[test]
    fn empty_vocabulary_keeps_legacy_exact_behavior() {
        let raw = audit_checklist_arguments("pending").to_string();
        let canonical = bind_checklist_targets(&raw, &StableSubjectVocabulary::default()).unwrap();
        assert_eq!(canonical, raw);
    }

    #[tokio::test]
    async fn structurally_invalid_targets_still_fail_validation_after_binding() {
        let (tool, _plan_store) = test_tool().await;
        let tool = tool.with_stable_subjects(synthetic_vocabulary());
        let raw = json!({
            "items": [{
                "text": "broken",
                "status": "pending",
                "targets": [{"subject": {"kind": "resource_id", "value": "   "}, "facets": []}]
            }]
        })
        .to_string();
        let violation = tool.prepare_invocation(&raw).unwrap_err();
        assert!(violation.reason.contains("item 0"), "{violation:?}");
    }

    #[tokio::test]
    async fn schema_lists_advertised_stable_subjects() {
        let (tool, _plan_store) = test_tool().await;
        let tool = tool.with_stable_subjects(synthetic_vocabulary());
        let schema = tool.schema();
        let description = schema["parameters"]["properties"]["items"]["items"]["properties"]
            ["targets"]["description"]
            .as_str()
            .unwrap();
        for expected in [
            "objective_collection:scheduled_goals",
            "objective_collection:mandate_controllers",
            "auth_profile_collection:configured",
            "objective:",
            "auth_profile:",
            "res_",
        ] {
            assert!(
                description.contains(expected),
                "schema must advertise {expected}: {description}"
            );
        }
    }

    #[test]
    fn structured_subject_facets_survive_into_the_durable_expectation_event() {
        let declared = executor_expectations_from_checklist_arguments(
            "task-synthetic",
            r#"{
                "items": [{
                    "text": "Audit one objective",
                    "requires_observation": true,
                    "targets": [{
                        "subject": {
                            "kind": "resource_id",
                            "value": "objective:sha256:synthetic"
                        },
                        "facets": ["schedule", "recovery"],
                        "collection_coverage": {
                            "collection": {
                                "kind": "resource_id",
                                "value": "objective_collection:scheduled_goals"
                            },
                            "minimum_completeness": "complete"
                        }
                    }],
                    "status": "pending"
                }]
            }"#,
        )
        .expect("typed declaration");
        let item = &declared.items[0];
        assert_eq!(item.targets, ["objective:sha256:synthetic"]);
        assert_eq!(item.observation_targets.len(), 1);
        assert_eq!(
            item.observation_targets[0].facets,
            [ToolSemanticFacet::Schedule, ToolSemanticFacet::Recovery]
        );
        assert_eq!(
            item.observation_targets[0]
                .collection_coverage
                .as_ref()
                .map(|coverage| coverage.collection.value.as_str()),
            Some("objective_collection:scheduled_goals")
        );
        assert!(item.unbound_targets.is_empty());
    }

    #[test]
    fn unbound_targets_survive_into_the_durable_expectation_event() {
        let canonical = bind_checklist_targets(
            &audit_checklist_arguments("pending").to_string(),
            &synthetic_vocabulary(),
        )
        .unwrap();
        let declared =
            executor_expectations_from_checklist_arguments("task-audit", &canonical).unwrap();
        assert_eq!(declared.items.len(), 3);
        assert_eq!(declared.items[0].observation_targets.len(), 1);
        assert_eq!(declared.items[1].observation_targets.len(), 1);
        assert!(declared.items[2].targets.is_empty());
        assert!(declared.items[2].observation_targets.is_empty());
        assert_eq!(
            declared.items[2].unbound_targets,
            ["x_owner_credential_state"]
        );
        assert!(declared.items[2].requires_observation);
    }
}
