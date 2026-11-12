//! How a finished run becomes the turn's reply and task outcome.

use super::*;
#[cfg(test)]
pub(super) use aidaemon_runtime::engine::outcome::*;

/// Defer the checklist's advisory steps once the run is done: a step with
/// no typed mutation effect and no observation requirement is display only,
/// and left pending it would make restart recovery pause the plan and ask to
/// resume work that is finished. Typed steps keep their state. Returns
/// whether anything changed.
pub(super) fn reconcile_plan_projection(plan: &mut crate::plans::TaskPlan) -> bool {
    use crate::plans::StepStatus;
    let mut changed = false;
    for step in &mut plan.steps {
        // Marked as handed to the background, but no launch took it: nothing
        // is working on it, so it was not done.
        if step.status == StepStatus::Background && step.background_jobs.is_empty() {
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

impl Agent {
    /// Reconcile this task's checklist after a completed run (see
    /// [`reconcile_plan_projection`]) and record its final shape.
    pub(super) async fn reconcile_checklist_projection(
        &self,
        emitter: &EventEmitter,
        session_id: &str,
        task_id: &str,
    ) {
        let Some(plan_store) = self.plan_store.read().await.clone() else {
            return;
        };
        let Some(mut plan) = plan_store
            .get_recent_for_session(session_id, 1)
            .await
            .ok()
            .and_then(|plans| plans.into_iter().next())
            .filter(|plan| plan.task_id.as_deref() == Some(task_id))
        else {
            return;
        };
        if reconcile_plan_projection(&mut plan) {
            if let Err(error) = plan_store.update(&plan).await {
                warn!(task_id, %error, "SDK engine: could not reconcile the checklist");
            }
        }
        let unchecked = plan.unchecked_steps().len();
        let deferred = plan
            .steps
            .iter()
            .filter(|step| step.status == crate::plans::StepStatus::Deferred)
            .count();
        self.emit_decision_point(
            emitter,
            task_id,
            0,
            crate::events::DecisionType::GateTelemetry,
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
