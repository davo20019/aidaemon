//! `track_requirements` — model-facing tool to register and update the durable
//! checklist of requirements for the current multi-step / deferred-action turn.
//! Full-set replace each call (like a todo tool). Backed by the `plans/` store.

use std::sync::{Arc, OnceLock, Weak};

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::channels::ChannelHub;
use crate::plans::{PlanStore, StepStatus};
use crate::traits::{Tool, ToolCapabilities};

pub struct TrackRequirementsTool {
    plan_store: Arc<PlanStore>,
    /// Set after ChannelHub creation (core.rs ordering), mirroring the terminal
    /// tool's deferred hub wiring. `None` until set; channel posts are skipped
    /// until then (and in tests).
    hub: OnceLock<Weak<ChannelHub>>,
}

impl TrackRequirementsTool {
    pub fn new(plan_store: Arc<PlanStore>) -> Self {
        Self {
            plan_store,
            hub: OnceLock::new(),
        }
    }

    pub fn set_hub(&self, hub: Weak<ChannelHub>) {
        let _ = self.hub.set(hub);
    }

    fn get_hub(&self) -> Option<Arc<ChannelHub>> {
        self.hub.get().and_then(|w| w.upgrade())
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
                                    "description": "One concrete requirement, e.g. 'send the report file to the user'."
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "deferred"]
                                },
                                "note": {
                                    "type": "string",
                                    "description": "Optional short note (e.g. why deferred)."
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

    fn capabilities(&self) -> ToolCapabilities {
        // Internal bookkeeping only: writes the checklist to the local plan store.
        // No external side effect, no approval, full-set replace is idempotent.
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
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

        let items: Vec<(String, StepStatus)> = v
            .get("items")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|it| {
                        let text = it.get("text").and_then(Value::as_str)?.to_string();
                        let status = it
                            .get("status")
                            .and_then(Value::as_str)
                            .map(parse_status)
                            .unwrap_or(StepStatus::Pending);
                        Some((text, status))
                    })
                    .collect()
            })
            .unwrap_or_default();

        if items.is_empty() {
            return Ok("track_requirements: no items provided.".to_string());
        }

        // Was there already a checklist for this session? (decides whether to post.)
        let had_existing = matches!(
            self.plan_store
                .get_incomplete_for_session(&session_id)
                .await,
            Ok(Some(_))
        );

        let plan = match self
            .plan_store
            .upsert_checklist(
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

        // Post the compact checklist to the channel ONCE, on first creation.
        if !had_existing {
            if let Some(hub) = self.get_hub() {
                let _ = hub
                    .send_text(&session_id, &plan.render_compact_checklist())
                    .await;
            }
        }

        Ok(format!(
            "Checklist updated ({}/{} done):\n{}",
            plan.completed_steps(),
            plan.steps.len(),
            plan.render_compact_checklist()
        ))
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
        // No hub set: channel posts are skipped (deferred wiring happens in core.rs).
        (TrackRequirementsTool::new(plan_store.clone()), plan_store)
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
