//! `track_requirements` — model-facing tool to register and update the durable
//! checklist of requirements for the current multi-step / deferred-action turn.
//! Full-set replace each call (like a todo tool). Backed by the `plans/` store.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::plans::{ChecklistItem, PlanStore, StepStatus};
use crate::traits::{Tool, ToolCallSemantics, ToolCapabilities, ToolMutationEffects};

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

#[async_trait]
impl Tool for TrackRequirementsTool {
    fn name(&self) -> &str {
        "track_requirements"
    }

    fn description(&self) -> &str {
        "Register and update the checklist of concrete requirements for the current request. \
         Call this FIRST when a request has multiple steps or a deferred action (e.g. 'do X, then send me the file'). \
         Pass the FULL list every time with each item's current status; it replaces the previous list. \
         Mark each item 'completed' as you finish it, or 'deferred' if you intentionally skip it."
    }

    fn schema(&self) -> Value {
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
                                    "description": "Exact canonical receipt targets.",
                                    "items": { "type": "string", "minLength": 1 },
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
            let expected_targets = match item.get("targets") {
                None => Vec::new(),
                Some(Value::Array(values)) => {
                    let Some(targets) = values
                        .iter()
                        .map(|value| value.as_str().map(str::to_string))
                        .collect::<Option<Vec<_>>>()
                    else {
                        return Ok(format!(
                            "track_requirements: item {item_index} has an invalid exact target."
                        ));
                    };
                    targets
                }
                Some(_) => {
                    return Ok(format!(
                        "track_requirements: item {item_index} targets must be an array."
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
