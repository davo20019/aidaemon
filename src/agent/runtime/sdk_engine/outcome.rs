//! How a finished run becomes the turn's reply and task outcome.

use super::*;
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

/// What a run settled, beyond its kernel status, that decides the turn's
/// final status and outcome. Shared with run-end hooks (memory writers), so
/// they judge a run exactly as the turn does.
#[derive(Default)]
pub(crate) struct TurnVerdict {
    /// The reply and outcome a used-up goal or mandate budget closes the run
    /// with; the run itself ends through the provider port.
    budget_stop: std::sync::Mutex<Option<(String, TaskOutcome)>>,
    /// Confirmation of the latest successful external change.
    pub(super) external_action_ack: std::sync::Mutex<Option<String>>,
    /// Set when the run must end failed despite a final answer.
    pub(super) run_failure: std::sync::Mutex<Option<String>>,
    /// The latest provider note (for example a safety finish reason).
    pub(super) response_note: std::sync::Mutex<Option<String>>,
    /// The latest well-formed final answer a closeout demand sent back for
    /// more work. If the model then replies with nothing, this ships instead
    /// of a generic activity summary.
    interrupted_answer: std::sync::Mutex<Option<String>>,
    /// The handoff reply for a run that stopped while a job it launched was
    /// still running: that job's notifier delivers the result, so the run
    /// tells the user the work continues instead of shipping the model's own
    /// verdict on unfinished work.
    handoff_reply: std::sync::Mutex<Option<String>>,
}

impl TurnVerdict {
    /// The run ends on a used-up budget: the next model call fails with the
    /// budget's marker, and the turn ships `reply` with `outcome`.
    pub(super) fn record_budget_stop(&self, reply: String, outcome: TaskOutcome) {
        if let Ok(mut stop) = self.budget_stop.lock() {
            *stop = Some((reply, outcome));
        }
    }

    pub(super) fn record_handoff_reply(&self, reply: String) {
        if let Ok(mut handoff) = self.handoff_reply.lock() {
            *handoff = Some(reply);
        }
    }

    pub(crate) fn record_interrupted_answer(&self, answer: String) {
        if let Ok(mut interrupted) = self.interrupted_answer.lock() {
            *interrupted = Some(answer);
        }
    }

    /// The reply the user gets, with its status and outcome: [`Self::resolve`],
    /// then for a completed run the interrupted-answer fallback and the final
    /// reply checks (reply sanitization alone for a stopped run), and for
    /// every run secret redaction on public channels.
    pub(crate) fn final_reply(
        &self,
        outcome: &sdk::RunOutcome,
        tool_calls: &[String],
        visibility: crate::types::ChannelVisibility,
    ) -> (String, TaskStatus, TaskOutcome) {
        let (reply, status, task_outcome) = self.resolve(outcome);
        let handoff = self
            .handoff_reply
            .lock()
            .ok()
            .and_then(|reply| reply.clone());
        // A handed-off run stopped with its work still running: partial work,
        // whatever the kernel's completed status would otherwise credit.
        let handed_off = matches!(outcome.status, sdk::RunStatus::Completed) && handoff.is_some();
        let task_outcome = if handed_off && task_outcome == TaskOutcome::Succeeded {
            TaskOutcome::Partial
        } else {
            task_outcome
        };
        let reply = if let (true, Some(handoff)) = (handed_off, handoff) {
            handoff
        } else if matches!(outcome.status, sdk::RunStatus::Completed) {
            let note = self.response_note.lock().ok().and_then(|note| note.clone());
            let interrupted = self
                .interrupted_answer
                .lock()
                .ok()
                .and_then(|answer| answer.clone());
            let reply = match interrupted {
                Some(answer) if reply.trim().is_empty() => answer,
                _ => reply,
            };
            finalize_reply(&reply, tool_calls, note.as_deref())
        } else {
            // A stopped run's text (the model's partial answer, or the
            // daemon's own explanation) gets the same reply sanitization; a
            // partial answer that was mostly scaffolding is not shipped.
            let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(&reply);
            if crate::tools::sanitize::reply_gutted_by_sanitization(
                reply.trim().chars().count(),
                &sanitized,
            ) {
                "I stopped before finishing and couldn't put together a clean summary of where \
                 I got to. Ask me to continue and I'll pick up from here."
                    .to_string()
            } else {
                sanitized
            }
        };
        let reply = match visibility {
            crate::types::ChannelVisibility::Public
            | crate::types::ChannelVisibility::PublicExternal => {
                let (sanitized, redacted) = crate::tools::sanitize::sanitize_output(&reply);
                if redacted && visibility == crate::types::ChannelVisibility::PublicExternal {
                    format!("{sanitized}\n\n(Some content was filtered for security)")
                } else {
                    sanitized
                }
            }
            _ => reply,
        };
        (reply, status, task_outcome)
    }

    pub(super) fn run_failure(&self) -> Option<String> {
        self.run_failure
            .lock()
            .ok()
            .and_then(|failure| failure.clone())
    }

    /// The turn's reply, status and outcome for a finished run.
    pub(in crate::agent) fn resolve(
        &self,
        outcome: &sdk::RunOutcome,
    ) -> (String, TaskStatus, TaskOutcome) {
        let (reply, status, task_outcome) = reply_for(outcome);
        let external_ack = self
            .external_action_ack
            .lock()
            .ok()
            .and_then(|ack| ack.clone());
        let budget_stop = self.budget_stop.lock().ok().and_then(|stop| stop.clone());
        let (reply, status, task_outcome) = match (&outcome.status, budget_stop) {
            // The budget hook stopped the run, or the mandate's balance
            // could not cover the next call.
            (sdk::RunStatus::Stopped { reason }, Some((reply, outcome)))
                if reason == GOAL_BUDGET_EXHAUSTED =>
            {
                (reply, TaskStatus::Completed, outcome)
            }
            (sdk::RunStatus::ProviderFailed { error }, Some((reply, outcome)))
                if error.contains(MANDATE_BUDGET_EXHAUSTED) =>
            {
                (reply, TaskStatus::Completed, outcome)
            }
            _ => (reply, status, task_outcome),
        };
        let (reply, status, task_outcome) = match (&outcome.status, external_ack) {
            (
                sdk::RunStatus::ProviderFailed { .. } | sdk::RunStatus::LimitReached { .. },
                Some(ack),
            )
            | (sdk::RunStatus::Stopped { .. }, Some(ack)) => (
                crate::agent::tool_execution::user_facing_external_action_ack(&ack),
                TaskStatus::Completed,
                TaskOutcome::Succeeded,
            ),
            _ => (reply, status, task_outcome),
        };
        // The owner declined part of what the run set out to do. However the
        // run ended, and whatever acknowledged it (a tool's finish, or an
        // external action's ack on a limit or provider stop), it never
        // succeeds over that decline.
        let task_outcome = match task_outcome {
            TaskOutcome::Succeeded if !outcome.state.declined.is_empty() => TaskOutcome::Partial,
            task_outcome => task_outcome,
        };
        if self.run_failure().is_some() {
            (reply, TaskStatus::Failed, TaskOutcome::Failed)
        } else {
            (reply, status, task_outcome)
        }
    }
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
