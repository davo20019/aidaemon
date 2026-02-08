//! Plan Manager Tool - LLM interface for managing task plans.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tracing::info;

use crate::plans::{generate_plan_steps, PlanStatus, PlanStore, StepStatus, StepTracker, TaskPlan};
use crate::traits::{ModelProvider, Tool};

/// Tool for LLM to manage task plans.
pub struct PlanManagerTool {
    plan_store: Arc<PlanStore>,
    step_tracker: Arc<StepTracker>,
    provider: Arc<dyn ModelProvider>,
    model: String,
}

impl PlanManagerTool {
    pub fn new(
        plan_store: Arc<PlanStore>,
        step_tracker: Arc<StepTracker>,
        provider: Arc<dyn ModelProvider>,
        model: String,
    ) -> Self {
        Self {
            plan_store,
            step_tracker,
            provider,
            model,
        }
    }
}

#[async_trait]
impl Tool for PlanManagerTool {
    fn name(&self) -> &str {
        "plan_manager"
    }

    fn description(&self) -> &str {
        "Manage multi-step task plans. ONLY use for complex coding/engineering tasks requiring 5+ tool calls \
         (e.g., building features, refactoring, deployments). Do NOT use for: questions, lookups, \
         single commands, or tasks completable in 1-4 tool calls. \
         Actions: create, revise, complete_step, fail_step, skip_step, retry_step, checkpoint, abandon, get, resume"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "plan_manager",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "revise", "complete_step", "fail_step", "skip_step", "retry_step", "checkpoint", "abandon", "get", "resume"],
                        "description": "Action to perform on the plan"
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description (for create action, or updated description for revise)"
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Step descriptions (for create action). If omitted, steps will be auto-generated."
                    },
                    "new_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New steps to REPLACE remaining pending steps (for revise action). Completed steps are preserved."
                    },
                    "revision_reason": {
                        "type": "string",
                        "description": "Why the plan is being revised (for revise action)"
                    },
                    "result_summary": {
                        "type": "string",
                        "description": "Summary of step result (for complete_step)"
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message (for fail_step)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for skipping (for skip_step)"
                    },
                    "checkpoint_key": {
                        "type": "string",
                        "description": "Key for checkpoint data (for checkpoint action)"
                    },
                    "checkpoint_value": {
                        "description": "Value to checkpoint (for checkpoint action)"
                    },
                    "plan_id": {
                        "type": "string",
                        "description": "Plan ID to look up (for get action). If omitted, returns the current session's active plan."
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;

        // Extract session_id injected by the agent
        let session_id = args["_session_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing _session_id in arguments"))?;

        let action = args["action"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing action"))?;

        match action {
            "create" => self.create_plan(session_id, &args).await,
            "revise" => self.revise_plan(session_id, &args).await,
            "complete_step" => self.complete_step(session_id, &args).await,
            "fail_step" => self.fail_step(session_id, &args).await,
            "skip_step" => self.skip_step(session_id, &args).await,
            "retry_step" => self.retry_step(session_id).await,
            "checkpoint" => self.checkpoint(session_id, &args).await,
            "abandon" => self.abandon_plan(session_id).await,
            "get" => {
                let plan_id = args["plan_id"].as_str();
                self.get_plan(session_id, plan_id).await
            }
            "resume" => self.resume_plan(session_id).await,
            _ => Err(anyhow::anyhow!("Unknown action: {}", action)),
        }
    }
}

impl PlanManagerTool {
    async fn create_plan(&self, session_id: &str, args: &Value) -> anyhow::Result<String> {
        // Check for existing incomplete plan
        if let Some(existing) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        {
            return Ok(format!(
                "Cannot create new plan - there's already an incomplete plan: '{}' ({:?}, step {}/{}). \
                 Use action='abandon' first if you want to start a new plan.",
                existing.description,
                existing.status,
                existing.current_step + 1,
                existing.steps.len()
            ));
        }

        let description = args["description"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing description for create action"))?;

        // Get steps - either from args or generate via LLM
        let steps: Vec<String> = if let Some(steps_array) = args["steps"].as_array() {
            steps_array
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            // Generate steps via LLM
            info!(description = %description, "Generating plan steps via LLM");
            generate_plan_steps(&*self.provider, &self.model, description).await?
        };

        if steps.is_empty() {
            return Err(anyhow::anyhow!("Cannot create plan with no steps"));
        }

        let plan = TaskPlan::new(
            session_id,
            description, // trigger_message
            description, // description
            steps.clone(),
            "explicit_creation",
        );

        self.plan_store.create(&plan).await?;

        info!(
            plan_id = %plan.id,
            session_id = %session_id,
            steps = steps.len(),
            "Created new task plan"
        );

        let steps_formatted: Vec<String> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s))
            .collect();

        Ok(format!(
            "Created plan '{}' with {} steps:\n{}\n\nCurrently on step 1. \
             Use complete_step when done, fail_step if it fails, or skip_step to skip.",
            plan.description,
            steps.len(),
            steps_formatted.join("\n")
        ))
    }

    async fn revise_plan(&self, session_id: &str, args: &Value) -> anyhow::Result<String> {
        let mut plan = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("No active plan to revise"))?;

        let revision_reason = args["revision_reason"]
            .as_str()
            .unwrap_or("User requested changes");

        // Get new steps - either from args or generate via LLM
        let new_steps: Vec<String> = if let Some(steps_array) = args["new_steps"].as_array() {
            steps_array
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            // Generate new steps via LLM, incorporating the revision reason
            let revised_description = if let Some(desc) = args["description"].as_str() {
                desc.to_string()
            } else {
                format!("{} (revised: {})", plan.description, revision_reason)
            };
            info!(description = %revised_description, "Generating revised plan steps via LLM");
            generate_plan_steps(&*self.provider, &self.model, &revised_description).await?
        };

        if new_steps.is_empty() {
            return Err(anyhow::anyhow!("Cannot revise plan with no new steps"));
        }

        // Preserve completed/in-progress steps, replace pending ones
        let current_idx = plan.current_step;
        let preserved_steps: Vec<_> = plan.steps.iter().take(current_idx + 1).cloned().collect();
        let preserved_count = preserved_steps.len();

        // Build new steps list: preserved + new
        let mut revised_steps = preserved_steps;
        for (i, description) in new_steps.into_iter().enumerate() {
            revised_steps.push(crate::plans::PlanStep {
                index: preserved_count + i,
                description,
                status: StepStatus::Pending,
                tool_call_ids: Vec::new(),
                result_summary: None,
                error: None,
                retry_count: 0,
                started_at: None,
                completed_at: None,
            });
        }

        // Update step indices
        for (i, step) in revised_steps.iter_mut().enumerate() {
            step.index = i;
        }

        let old_step_count = plan.steps.len();
        plan.steps = revised_steps;

        // Optionally update description
        if let Some(desc) = args["description"].as_str() {
            plan.description = desc.to_string();
        }

        plan.updated_at = chrono::Utc::now();
        self.plan_store.update(&plan).await?;

        info!(
            plan_id = %plan.id,
            session_id = %session_id,
            old_steps = old_step_count,
            new_steps = plan.steps.len(),
            preserved = preserved_count,
            reason = %revision_reason,
            "Revised task plan"
        );

        // Format output showing the revised plan
        let steps_formatted: Vec<String> = plan
            .steps
            .iter()
            .map(|s| {
                let marker = match s.status {
                    StepStatus::Completed => "[x]",
                    StepStatus::InProgress => "[>]",
                    StepStatus::Failed => "[!]",
                    StepStatus::Skipped => "[-]",
                    StepStatus::Pending => "[ ]",
                };
                format!("{} {}. {}", marker, s.index + 1, s.description)
            })
            .collect();

        Ok(format!(
            "Revised plan '{}' (reason: {}).\n\
             Preserved {} completed/current step(s), added {} new step(s).\n\
             Total: {} steps\n\n{}\n\nCurrently on step {}.",
            plan.description,
            revision_reason,
            preserved_count,
            plan.steps.len() - preserved_count,
            plan.steps.len(),
            steps_formatted.join("\n"),
            plan.current_step + 1
        ))
    }

    async fn complete_step(&self, session_id: &str, args: &Value) -> anyhow::Result<String> {
        let result_summary = args["result_summary"].as_str().map(String::from);

        let plan = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?;
        let current_step = plan.as_ref().map(|p| p.current_step).unwrap_or(0);

        match self
            .step_tracker
            .complete_step(session_id, current_step, result_summary)
            .await?
        {
            Some(plan) => {
                if plan.status == PlanStatus::Completed {
                    Ok(format!(
                        "Step {} completed. All steps done! Plan '{}' is complete.",
                        current_step + 1,
                        plan.description
                    ))
                } else {
                    let next_step = &plan.steps[plan.current_step];
                    Ok(format!(
                        "Step {} completed. Now on step {}: '{}'",
                        current_step + 1,
                        plan.current_step + 1,
                        next_step.description
                    ))
                }
            }
            None => Ok("No active plan found.".to_string()),
        }
    }

    async fn fail_step(&self, session_id: &str, args: &Value) -> anyhow::Result<String> {
        let error = args["error"]
            .as_str()
            .unwrap_or("Unspecified error")
            .to_string();

        let plan = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?;
        let current_step = plan.as_ref().map(|p| p.current_step).unwrap_or(0);

        match self
            .step_tracker
            .fail_step(session_id, current_step, error.clone())
            .await?
        {
            Some(plan) => {
                let step = &plan.steps[current_step];
                Ok(format!(
                    "Step {} '{}' marked as failed: {}. \
                     You can use retry_step to try again, skip_step to move on, or abandon to cancel the plan.",
                    current_step + 1,
                    step.description,
                    error
                ))
            }
            None => Ok("No active plan found.".to_string()),
        }
    }

    async fn skip_step(&self, session_id: &str, args: &Value) -> anyhow::Result<String> {
        let reason = args["reason"].as_str().map(String::from);

        match self.step_tracker.skip_step(session_id, reason).await? {
            Some(plan) => {
                if plan.status == PlanStatus::Completed {
                    Ok(format!(
                        "Step skipped. All steps done! Plan '{}' is complete.",
                        plan.description
                    ))
                } else {
                    let next_step = &plan.steps[plan.current_step];
                    Ok(format!(
                        "Step skipped. Now on step {}: '{}'",
                        plan.current_step + 1,
                        next_step.description
                    ))
                }
            }
            None => Ok("No active plan found.".to_string()),
        }
    }

    async fn retry_step(&self, session_id: &str) -> anyhow::Result<String> {
        match self.step_tracker.retry_step(session_id).await? {
            Some(plan) => {
                let step = &plan.steps[plan.current_step];
                Ok(format!(
                    "Retrying step {}: '{}' (attempt #{})",
                    plan.current_step + 1,
                    step.description,
                    step.retry_count + 1
                ))
            }
            None => Ok("No active plan found or current step is not failed.".to_string()),
        }
    }

    async fn checkpoint(&self, session_id: &str, args: &Value) -> anyhow::Result<String> {
        let key = args["checkpoint_key"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing checkpoint_key"))?;

        let value = args.get("checkpoint_value").cloned().unwrap_or(Value::Null);

        let plan = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("No active plan found"))?;

        self.plan_store
            .set_checkpoint(&plan.id, key, value.clone())
            .await?;

        Ok(format!(
            "Checkpointed '{}' = {} for plan '{}'",
            key, value, plan.description
        ))
    }

    async fn abandon_plan(&self, session_id: &str) -> anyhow::Result<String> {
        match self.step_tracker.abandon_plan(session_id).await? {
            Some(plan) => Ok(format!(
                "Abandoned plan '{}' at step {}/{}.",
                plan.description,
                plan.current_step + 1,
                plan.steps.len()
            )),
            None => Ok("No active plan found.".to_string()),
        }
    }

    async fn get_plan(&self, session_id: &str, plan_id: Option<&str>) -> anyhow::Result<String> {
        // Look up by plan_id if provided, otherwise find the active plan for this session
        let plan = if let Some(id) = plan_id {
            self.plan_store.get(id).await?
        } else {
            self.plan_store
                .get_incomplete_for_session(session_id)
                .await?
        };

        match plan {
            Some(plan) => {
                let mut output = format!(
                    "Plan: {}\nStatus: {:?}\nProgress: {}/{} steps\n\nSteps:\n",
                    plan.description,
                    plan.status,
                    plan.completed_steps(),
                    plan.steps.len()
                );

                for step in &plan.steps {
                    let marker = match step.status {
                        StepStatus::Completed => "[x]",
                        StepStatus::InProgress => "[>]",
                        StepStatus::Failed => "[!]",
                        StepStatus::Skipped => "[-]",
                        StepStatus::Pending => "[ ]",
                    };
                    let duration_str = match step.duration_secs() {
                        Some(secs) => format!(" ({}s)", secs),
                        None => String::new(),
                    };
                    output.push_str(&format!(
                        "  {} {}. {}{}\n",
                        marker,
                        step.index + 1,
                        step.description,
                        duration_str
                    ));
                    if let Some(ref error) = step.error {
                        output.push_str(&format!("      Error: {}\n", error));
                    }
                }

                if !plan.checkpoint.is_null()
                    && plan
                        .checkpoint
                        .as_object()
                        .map(|o| !o.is_empty())
                        .unwrap_or(false)
                {
                    output.push_str(&format!("\nCheckpoint data: {}\n", plan.checkpoint));
                }

                Ok(output)
            }
            None => {
                if plan_id.is_some() {
                    Ok("Plan not found.".to_string())
                } else {
                    Ok("No active plan found.".to_string())
                }
            }
        }
    }

    async fn resume_plan(&self, session_id: &str) -> anyhow::Result<String> {
        match self.step_tracker.resume_plan(session_id).await? {
            Some(plan) => {
                let step = &plan.steps[plan.current_step];
                Ok(format!(
                    "Resumed plan '{}'. Currently on step {}: '{}'",
                    plan.description,
                    plan.current_step + 1,
                    step.description
                ))
            }
            None => Ok("No paused plan found to resume.".to_string()),
        }
    }
}
