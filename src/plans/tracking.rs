//! Step tracking and progress updates.
//!
//! Monitors tool execution and updates plan progress.

use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, info};

use super::{PlanStatus, PlanStore, StepStatus, TaskPlan};
use crate::utils::truncate_str;

/// Tracks step progress based on tool execution.
pub struct StepTracker {
    plan_store: Arc<PlanStore>,
}

impl StepTracker {
    pub fn new(plan_store: Arc<PlanStore>) -> Self {
        Self { plan_store }
    }

    /// Record a tool call for the current step.
    /// Returns the updated plan if changes were made.
    pub async fn record_tool_call(
        &self,
        session_id: &str,
        tool_name: &str,
        tool_call_id: &str,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        // Only track if plan is actively executing
        if plan.status != PlanStatus::InProgress {
            return Ok(None);
        }

        // Record the tool call on the current step
        if let Some(step) = plan.current_step_mut() {
            let tool_ref = format!("{}:{}", tool_name, tool_call_id);
            step.tool_call_ids.push(tool_ref);
            debug!(
                plan_id = %plan.id,
                step = plan.current_step,
                tool = tool_name,
                "Recorded tool call for plan step"
            );
        }

        plan.updated_at = Utc::now();
        self.plan_store.update(&plan).await?;
        Ok(Some(plan))
    }

    /// Called after a tool result. May auto-advance steps based on heuristics.
    /// Returns (updated_plan, step_completed) if changes were made.
    pub async fn on_tool_result(
        &self,
        session_id: &str,
        tool_name: &str,
        success: bool,
        result_summary: &str,
    ) -> anyhow::Result<Option<(TaskPlan, bool)>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        // Only track if plan is actively executing
        if plan.status != PlanStatus::InProgress {
            return Ok(None);
        }

        let current_idx = plan.current_step;
        let step = match plan.steps.get(current_idx) {
            Some(s) if s.status == StepStatus::InProgress => s,
            _ => return Ok(Some((plan, false))),
        };

        // Check if this tool result indicates step failure
        if !success {
            // Don't auto-fail the step - let the LLM decide if this is recoverable
            // Just log it
            debug!(
                plan_id = %plan.id,
                step = current_idx,
                tool = tool_name,
                "Tool failed during plan step (not auto-failing step)"
            );
            return Ok(Some((plan, false)));
        }

        // Heuristic: Check if this looks like a "final" action for the step
        // This is conservative - the LLM should explicitly complete steps in most cases
        let step_desc_lower = step.description.to_lowercase();
        let result_lower = result_summary.to_lowercase();

        let looks_complete = if tool_name == "terminal" {
            // Test steps
            (step_desc_lower.contains("test")
                && (result_lower.contains("passed") || result_lower.contains("success")))
                // Build steps
                || (step_desc_lower.contains("build") && result_lower.contains("success"))
                // Deploy steps
                || (step_desc_lower.contains("deploy")
                    && (result_lower.contains("success") || result_lower.contains("deployed")))
        } else {
            false
        };

        if looks_complete {
            info!(
                plan_id = %plan.id,
                step = current_idx,
                step_description = %step.description,
                "Auto-completing plan step based on tool result"
            );

            plan.complete_current_step(Some(truncate_str(result_summary, 200)));

            // Advance to next step
            let had_next = plan.advance_to_next_step();

            if !had_next {
                info!(plan_id = %plan.id, "Plan completed - all steps done");
            }

            self.plan_store.update(&plan).await?;
            return Ok(Some((plan, true)));
        }

        Ok(Some((plan, false)))
    }

    /// Explicitly complete the current step (called by LLM via plan_manager tool).
    pub async fn complete_step(
        &self,
        session_id: &str,
        step_index: usize,
        result_summary: Option<String>,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        // Validate step index
        if step_index != plan.current_step {
            anyhow::bail!(
                "Cannot complete step {} - current step is {}",
                step_index,
                plan.current_step
            );
        }

        info!(
            plan_id = %plan.id,
            step = step_index,
            "Completing plan step (explicit)"
        );

        plan.complete_current_step(result_summary);
        let had_next = plan.advance_to_next_step();

        if !had_next {
            info!(plan_id = %plan.id, "Plan completed - all steps done");
        }

        self.plan_store.update(&plan).await?;
        Ok(Some(plan))
    }

    /// Explicitly fail the current step.
    pub async fn fail_step(
        &self,
        session_id: &str,
        step_index: usize,
        error: String,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        if step_index != plan.current_step {
            anyhow::bail!(
                "Cannot fail step {} - current step is {}",
                step_index,
                plan.current_step
            );
        }

        info!(
            plan_id = %plan.id,
            step = step_index,
            error = %error,
            "Failing plan step"
        );

        plan.fail_current_step(error);
        // Don't auto-fail the whole plan - let LLM decide what to do

        self.plan_store.update(&plan).await?;
        Ok(Some(plan))
    }

    /// Retry the current step.
    pub async fn retry_step(&self, session_id: &str) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        let step = plan.current_step_ref();
        if !matches!(step.map(|s| s.status), Some(StepStatus::Failed)) {
            anyhow::bail!("Can only retry failed steps");
        }

        info!(
            plan_id = %plan.id,
            step = plan.current_step,
            "Retrying plan step"
        );

        plan.retry_current_step();
        self.plan_store.update(&plan).await?;
        Ok(Some(plan))
    }

    /// Skip the current step.
    pub async fn skip_step(
        &self,
        session_id: &str,
        reason: Option<String>,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        info!(
            plan_id = %plan.id,
            step = plan.current_step,
            reason = ?reason,
            "Skipping plan step"
        );

        if let Some(step) = plan.current_step_mut() {
            step.status = StepStatus::Skipped;
            step.completed_at = Some(Utc::now());
            step.result_summary = reason;
        }

        let had_next = plan.advance_to_next_step();
        if !had_next {
            info!(plan_id = %plan.id, "Plan completed - all steps done (some skipped)");
        }

        self.plan_store.update(&plan).await?;
        Ok(Some(plan))
    }

    /// Pause all in-progress plans (e.g., on graceful shutdown).
    /// Returns the number of plans paused.
    pub async fn pause_all_plans(&self) -> anyhow::Result<usize> {
        let plans = self.plan_store.get_all_in_progress().await?;
        let mut count = 0;
        for mut plan in plans {
            info!(plan_id = %plan.id, session_id = %plan.session_id, "Pausing plan for shutdown");
            plan.status = PlanStatus::Paused;
            self.plan_store.update(&plan).await?;
            count += 1;
        }
        Ok(count)
    }

    /// Pause the plan (e.g., when session ends).
    pub async fn pause_plan(&self, session_id: &str) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        if plan.status == PlanStatus::InProgress {
            info!(plan_id = %plan.id, "Pausing plan");
            plan.status = PlanStatus::Paused;
            self.plan_store.update(&plan).await?;
        }

        Ok(Some(plan))
    }

    /// Resume a paused plan.
    pub async fn resume_plan(&self, session_id: &str) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        if plan.status == PlanStatus::Paused {
            info!(plan_id = %plan.id, "Resuming plan");
            plan.status = PlanStatus::InProgress;

            // Ensure current step is in progress
            if let Some(step) = plan.current_step_mut() {
                if step.status == StepStatus::Pending {
                    step.status = StepStatus::InProgress;
                    step.started_at = Some(Utc::now());
                }
            }

            self.plan_store.update(&plan).await?;
        }

        Ok(Some(plan))
    }

    /// Abandon a plan (user cancelled).
    pub async fn abandon_plan(&self, session_id: &str) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        info!(plan_id = %plan.id, "Abandoning plan");
        plan.status = PlanStatus::Abandoned;
        self.plan_store.update(&plan).await?;
        Ok(Some(plan))
    }

    /// Mark plan as completely failed (unrecoverable).
    pub async fn fail_plan(
        &self,
        session_id: &str,
        error: String,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let Some(mut plan) = self
            .plan_store
            .get_incomplete_for_session(session_id)
            .await?
        else {
            return Ok(None);
        };

        info!(plan_id = %plan.id, error = %error, "Failing plan");
        plan.status = PlanStatus::Failed;

        // Mark current step as failed too
        if let Some(step) = plan.current_step_mut() {
            if step.status == StepStatus::InProgress {
                step.status = StepStatus::Failed;
                step.error = Some(error);
                step.completed_at = Some(Utc::now());
            }
        }

        self.plan_store.update(&plan).await?;
        Ok(Some(plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plans::TaskPlan;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn create_test_tracker() -> (StepTracker, Arc<PlanStore>) {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let store = Arc::new(PlanStore::new(pool).await.unwrap());
        let tracker = StepTracker::new(store.clone());
        (tracker, store)
    }

    #[tokio::test]
    async fn test_complete_step() {
        let (tracker, store) = create_test_tracker().await;

        let plan = TaskPlan::new(
            "session_123",
            "Test",
            "Test task",
            vec!["Step 1".to_string(), "Step 2".to_string()],
            "test",
        );
        store.create(&plan).await.unwrap();

        // Complete first step
        let updated = tracker
            .complete_step("session_123", 0, Some("Done".to_string()))
            .await
            .unwrap()
            .unwrap();

        assert_eq!(updated.current_step, 1);
        assert_eq!(updated.steps[0].status, StepStatus::Completed);
        assert_eq!(updated.steps[1].status, StepStatus::InProgress);
    }

    #[tokio::test]
    async fn test_pause_and_resume() {
        let (tracker, store) = create_test_tracker().await;

        let plan = TaskPlan::new(
            "session_123",
            "Test",
            "Test task",
            vec!["Step 1".to_string()],
            "test",
        );
        store.create(&plan).await.unwrap();

        // Pause
        let paused = tracker.pause_plan("session_123").await.unwrap().unwrap();
        assert_eq!(paused.status, PlanStatus::Paused);

        // Resume
        let resumed = tracker.resume_plan("session_123").await.unwrap().unwrap();
        assert_eq!(resumed.status, PlanStatus::InProgress);
    }

    #[tokio::test]
    async fn test_abandon_plan() {
        let (tracker, store) = create_test_tracker().await;

        let plan = TaskPlan::new(
            "session_123",
            "Test",
            "Test task",
            vec!["Step 1".to_string()],
            "test",
        );
        store.create(&plan).await.unwrap();

        let abandoned = tracker.abandon_plan("session_123").await.unwrap().unwrap();
        assert_eq!(abandoned.status, PlanStatus::Abandoned);

        // Should no longer find incomplete plan
        let incomplete = store
            .get_incomplete_for_session("session_123")
            .await
            .unwrap();
        assert!(incomplete.is_none());
    }
}
