//! How a finished run becomes the turn's reply and task outcome.

use super::*;
#[cfg(any(test, feature = "testing"))]
pub use crate::engine::outcome::*;

/// Defer the checklist's advisory steps once the run is done: a step with
/// no typed mutation effect and no observation requirement is display only,
/// and left pending it would make restart recovery pause the plan and ask to
/// resume work that is finished. Typed steps keep their state. Returns
/// whether anything changed.
pub fn reconcile_plan_projection(plan: &mut aidaemon_work::plans::TaskPlan) -> bool {
    use aidaemon_work::plans::StepStatus;
    let mut changed = false;
    for step in &mut plan.steps {
        // Marked as handed to the background, but no launch took it: nothing
        // is working on it, so it was not done.
        if step.status == StepStatus::Background && !step.has_background_jobs() {
            step.status = StepStatus::Deferred;
            step.completed_at = Some(chrono::Utc::now());
            changed = true;
            continue;
        }
        if matches!(step.status, StepStatus::Pending | StepStatus::InProgress)
            && step.required_mutation_effects.is_empty()
            && !step.requires_observation
        {
            step.status = StepStatus::Deferred;
            step.result_summary = Some(
                "Advisory checklist item; authoritative completion is owned by the typed task proof"
                    .to_string(),
            );
            step.completed_at = Some(chrono::Utc::now());
            changed = true;
        }
    }
    if changed {
        plan.sync_active_step();
    }
    changed
}

impl TurnEngine<'_> {
    /// Mark the steps of this task's checklist that the run's receipts
    /// proved as confirmed (no longer only the model's word). Returns the
    /// card to show when anything changed.
    /// The card for this task's checklist as stored now. A one-step plan
    /// gets no card (`track_requirements`): the step is the request.
    pub async fn plan_card(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Option<aidaemon_core::card::PlanCard> {
        let plan_store = self.plan_store().await?;
        match plan_store.for_task(session_id, task_id).await {
            Ok(plan) => plan
                .filter(|plan| plan.steps.len() > 1)
                .map(|plan| plan.card()),
            Err(error) => {
                warn!(task_id, %error, "SDK engine: could not read the checklist");
                None
            }
        }
    }

    pub async fn confirm_checklist_from_receipts(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Option<aidaemon_core::card::PlanCard> {
        let plan_store = self.plan_store().await?;
        // Only a step still on the model's word can be proven: a checklist
        // with none skips reading the run's evidence.
        plan_store
            .for_task(session_id, task_id)
            .await
            .ok()
            .flatten()
            .filter(|plan| {
                plan.steps
                    .iter()
                    .any(|step| step.evidence == aidaemon_work::plans::StepEvidence::ModelSaid)
            })?;
        let proven = self
            .events()
            .task_run_aggregate(session_id, task_id)
            .await
            .ok()?
            .satisfied_checklist_items();
        if proven.is_empty() {
            return None;
        }
        match plan_store.confirm_steps(session_id, task_id, &proven).await {
            // A one-step plan gets no card (`track_requirements`).
            Ok(plan) => plan
                .filter(|plan| plan.steps.len() > 1)
                .map(|plan| plan.card()),
            Err(error) => {
                warn!(task_id, %error, "SDK engine: could not confirm checklist steps");
                None
            }
        }
    }

    /// Reconcile this task's checklist after a completed run (see
    /// [`reconcile_plan_projection`]) and record its final shape.
    pub async fn reconcile_checklist_projection(
        &self,
        emitter: &EventEmitter,
        session_id: &str,
        task_id: &str,
    ) {
        // Receipts that landed after the last checklist update may prove
        // steps the model ticked earlier.
        let _ = self
            .confirm_checklist_from_receipts(session_id, task_id)
            .await;
        let Some(plan_store) = self.plan_store().await else {
            return;
        };
        let Some(plan_id) = plan_store
            .for_task(session_id, task_id)
            .await
            .ok()
            .flatten()
            .map(|plan| plan.id)
        else {
            return;
        };
        // Changed under the plan write lock: a helper's launch or end may
        // be changing the same plan now.
        let plan = match plan_store.modify(&plan_id, reconcile_plan_projection).await {
            Ok(Some(plan)) => plan,
            Ok(None) => return,
            Err(error) => {
                warn!(task_id, %error, "SDK engine: could not reconcile the checklist");
                return;
            }
        };
        let unchecked = plan.unchecked_steps().len();
        let deferred = plan
            .steps
            .iter()
            .filter(|step| step.status == aidaemon_work::plans::StepStatus::Deferred)
            .count();
        self.emit_decision_point(
            emitter,
            task_id,
            0,
            aidaemon_events::payloads::DecisionType::GateTelemetry,
            "checklist turn completed",
            serde_json::json!({
                "event": "checklist_complete",
                "total": plan.steps.len(),
                "completed": plan.completed_steps(),
                "deferred": deferred,
                "unchecked": unchecked,
                "engine": "sdk",
            }),
        )
        .await;
    }
}
