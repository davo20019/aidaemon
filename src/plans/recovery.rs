//! Plan recovery after process restart.
//!
//! Handles plans that were interrupted by crashes or restarts.

use std::sync::Arc;

use chrono::Utc;
use tracing::{info, warn};

use super::{PlanStatus, PlanStore, StepStatus};

/// Statistics from plan recovery.
#[derive(Debug, Default)]
pub struct RecoveryStats {
    /// Plans that were already completed (all steps done despite in_progress status)
    pub completed: usize,
    /// Plans that were paused (interrupted mid-execution)
    pub paused: usize,
    /// Plans that failed during recovery
    pub failed: usize,
}

/// Handles recovery of interrupted plans.
pub struct PlanRecovery {
    plan_store: Arc<PlanStore>,
}

impl PlanRecovery {
    pub fn new(plan_store: Arc<PlanStore>) -> Self {
        Self { plan_store }
    }

    /// Recover plans that were interrupted by process restart.
    ///
    /// This should be called on startup. It:
    /// 1. Finds all plans that were "in_progress"
    /// 2. Marks them as "paused" so the agent knows to ask about resuming
    /// 3. Adds a note to the current step about the interruption
    pub async fn recover_interrupted_plans(&self) -> anyhow::Result<RecoveryStats> {
        let mut stats = RecoveryStats::default();

        // Find all plans that were in progress
        let in_progress_plans = self.plan_store.get_all_in_progress().await?;

        if in_progress_plans.is_empty() {
            return Ok(stats);
        }

        info!(
            count = in_progress_plans.len(),
            "Found in-progress plans to recover"
        );

        for mut plan in in_progress_plans {
            // If all steps are already done, mark as completed instead of paused
            if plan.is_finished() {
                plan.status = PlanStatus::Completed;
                plan.updated_at = Utc::now();
                match self.plan_store.update(&plan).await {
                    Ok(()) => {
                        info!(
                            plan_id = %plan.id,
                            session_id = %plan.session_id,
                            "Marked finished plan as completed during recovery"
                        );
                        stats.completed += 1;
                    }
                    Err(e) => {
                        warn!(
                            plan_id = %plan.id,
                            error = %e,
                            "Failed to update finished plan"
                        );
                        stats.failed += 1;
                    }
                }
                continue;
            }

            // Mark the plan as paused
            plan.status = PlanStatus::Paused;
            plan.updated_at = Utc::now();

            // Add a note to the current step about the interruption
            if let Some(step) = plan.steps.get_mut(plan.current_step) {
                if step.status == StepStatus::InProgress {
                    // Keep it as InProgress but note the interruption
                    let existing_error = step.error.take().unwrap_or_default();
                    let interrupt_note = if existing_error.is_empty() {
                        "Interrupted by process restart".to_string()
                    } else {
                        format!("{}; Interrupted by process restart", existing_error)
                    };
                    step.error = Some(interrupt_note);
                }
            }

            // Save the updated plan
            match self.plan_store.update(&plan).await {
                Ok(()) => {
                    info!(
                        plan_id = %plan.id,
                        session_id = %plan.session_id,
                        step = plan.current_step,
                        "Marked interrupted plan as paused"
                    );
                    stats.paused += 1;
                }
                Err(e) => {
                    warn!(
                        plan_id = %plan.id,
                        error = %e,
                        "Failed to update interrupted plan"
                    );
                    stats.failed += 1;
                }
            }
        }

        Ok(stats)
    }

    /// Get a summary of plans needing attention (for logging).
    pub async fn get_recovery_summary(&self) -> anyhow::Result<String> {
        let in_progress = self.plan_store.get_all_in_progress().await?;

        if in_progress.is_empty() {
            return Ok("No plans need recovery".to_string());
        }

        let mut summary = format!("{} plan(s) were in progress:\n", in_progress.len());
        for plan in in_progress {
            summary.push_str(&format!(
                "  - [{}] {} (step {}/{})\n",
                plan.session_id,
                plan.description,
                plan.current_step + 1,
                plan.steps.len()
            ));
        }

        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plans::TaskPlan;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn create_test_recovery() -> (PlanRecovery, Arc<PlanStore>) {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let store = Arc::new(PlanStore::new(pool).await.unwrap());
        let recovery = PlanRecovery::new(store.clone());
        (recovery, store)
    }

    #[tokio::test]
    async fn test_recover_interrupted_plans() {
        let (recovery, store) = create_test_recovery().await;

        // Create an in-progress plan
        let mut plan = TaskPlan::new(
            "session_123",
            "Test",
            "Test task",
            vec!["Step 1".to_string(), "Step 2".to_string()],
            "test",
        );
        // Simulate being mid-execution
        plan.complete_current_step(Some("Done".to_string()));
        plan.advance_to_next_step();
        store.create(&plan).await.unwrap();

        // Simulate restart - recover plans
        let stats = recovery.recover_interrupted_plans().await.unwrap();

        assert_eq!(stats.paused, 1);
        assert_eq!(stats.failed, 0);

        // Check the plan was marked as paused
        let recovered = store.get(&plan.id).await.unwrap().unwrap();
        assert_eq!(recovered.status, PlanStatus::Paused);
        assert!(recovered.steps[1]
            .error
            .as_ref()
            .unwrap()
            .contains("Interrupted"));
    }

    #[tokio::test]
    async fn test_no_plans_to_recover() {
        let (recovery, store) = create_test_recovery().await;

        // Create a completed plan (should not be recovered)
        let mut plan = TaskPlan::new(
            "session_123",
            "Test",
            "Test task",
            vec!["Step 1".to_string()],
            "test",
        );
        plan.status = PlanStatus::Completed;
        store.create(&plan).await.unwrap();

        let stats = recovery.recover_interrupted_plans().await.unwrap();
        assert_eq!(stats.paused, 0);
    }
}
