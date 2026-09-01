//! `track_requirements` — model-facing tool to register and update the durable
//! checklist of requirements for the current multi-step / deferred-action turn.
//! Full-set replace each call (like a todo tool). Backed by the `plans/` store.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::plans::{ChecklistItem, PlanStore, StepStatus};
use crate::traits::{
    RequestObservationTarget, Tool, ToolArgumentContractViolation, ToolCallSemantics,
    ToolCapabilities, ToolMutationEffects, ToolSemanticFacet,
};

pub struct TrackRequirementsTool {
    plan_store: Arc<PlanStore>,
}

impl TrackRequirementsTool {
    pub fn new(plan_store: Arc<PlanStore>) -> Self {
        Self { plan_store }
    }
}

fn parse_status(s: &str) -> StepStatus {
    StepStatus::from_str(s).unwrap_or(StepStatus::Pending)
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
                    "collection-backed targets must declare at least one semantic facet"
                        .to_string(),
                );
            }
        }
        legacy_targets.push(target.subject.value.clone());
        observation_targets.push(target);
    }
    Ok((legacy_targets, observation_targets))
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
                                    "description": "Exact receipt evidence targets. Prefer a structured subject/facet binding and never invent a resource ID. Use an ID returned by a tool; for objective discovery, the stable collection subjects are objective_collection:scheduled_goals and objective_collection:mandate_controllers. Legacy strings remain exact: absolute paths and URLs retain their type; every other value is an opaque resource ID that must exactly match a receipt.",
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
        for (item_index, item) in raw_items.iter().enumerate() {
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
        // This tool no longer posts/edits channel messages directly.
        Ok(format!(
            "Checklist updated ({}/{} done):\n{}",
            plan.completed_steps(),
            plan.steps.len(),
            plan.render_compact_checklist()
        ))
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
}
