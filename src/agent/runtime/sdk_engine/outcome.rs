//! How a finished run becomes the turn's reply and task outcome.

use super::*;

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

/// The user-facing reply for a run that did not end with a final answer.
pub(super) fn reply_for(outcome: &sdk::RunOutcome) -> (String, TaskStatus, TaskOutcome) {
    let text = outcome
        .final_text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string);
    match &outcome.status {
        sdk::RunStatus::Completed => (
            text.unwrap_or_default(),
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
        ),
        sdk::RunStatus::LimitReached { limit } => (
            text.unwrap_or_else(|| {
                let what = match limit {
                    sdk::LimitKind::Turns => "the number of steps",
                    sdk::LimitKind::Tokens => "the token budget",
                    sdk::LimitKind::ToolCalls => "the number of tool calls",
                    sdk::LimitKind::WallClock => "the time",
                };
                format!(
                    "I stopped before finishing: I reached the limit on {what} for one request. \
                     Ask me to continue and I'll pick up from here."
                )
            }),
            TaskStatus::Completed,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Stuck { signal } => (
            text.unwrap_or_else(|| {
                let _ = signal;
                "I stopped because I kept repeating the same step without making progress. \
                 Tell me how you'd like to proceed, or ask me to try a different approach."
                    .to_string()
            }),
            TaskStatus::Completed,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Stopped { reason } => (
            text.unwrap_or_else(|| format!("I stopped: {reason}")),
            TaskStatus::Completed,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Interrupted { interruption } => (
            format!(
                "I paused before running `{}` and could not ask for approval here.",
                interruption.call.name
            ),
            TaskStatus::Interrupted,
            TaskOutcome::Partial,
        ),
        sdk::RunStatus::Cancelled => (
            "Cancelled.".to_string(),
            TaskStatus::Cancelled,
            TaskOutcome::Failed,
        ),
        sdk::RunStatus::ProviderFailed { error } => (
            // Provider errors can carry paths and credentials: the user gets
            // the key line, redacted, as the legacy error summary did.
            format!(
                "I couldn't finish because the model provider returned an error: {}",
                crate::agent::post_task::redact_error_line_for_summary(
                    error.to_string().lines().next().unwrap_or_default()
                )
            ),
            TaskStatus::Failed,
            TaskOutcome::Failed,
        ),
        other => (
            text.unwrap_or_else(|| format!("The run ended without an answer ({other:?}).")),
            TaskStatus::Failed,
            TaskOutcome::Failed,
        ),
    }
}

/// What a run settled, beyond its kernel status, that decides the turn's
/// final status and outcome. Shared with run-end hooks (memory writers), so
/// they judge a run exactly as the turn does.
#[derive(Default)]
pub(crate) struct TurnVerdict {
    /// The neutral handoff reply once work moved to the background.
    pub(super) terminal_reply: std::sync::Mutex<Option<(String, TaskOutcome)>>,
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
}

impl TurnVerdict {
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
        let reply = if matches!(outcome.status, sdk::RunStatus::Completed) {
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
        let terminal = self
            .terminal_reply
            .lock()
            .ok()
            .and_then(|reply| reply.clone());
        let (reply, status, task_outcome) = match (&outcome.status, terminal) {
            (sdk::RunStatus::Stopped { reason }, Some((reply, outcome)))
                if reason == TERMINAL_REPLY_STOP =>
            {
                (reply, TaskStatus::Completed, outcome)
            }
            // A budget stop ends the run through the provider port.
            (sdk::RunStatus::ProviderFailed { error }, Some((reply, outcome)))
                if error.contains(GOAL_BUDGET_EXHAUSTED)
                    || error.contains(MANDATE_BUDGET_EXHAUSTED) =>
            {
                (reply, TaskStatus::Completed, outcome)
            }
            _ => (reply, status, task_outcome),
        };
        let (reply, status, task_outcome) = match (&outcome.status, external_ack) {
            (
                sdk::RunStatus::ProviderFailed { .. } | sdk::RunStatus::LimitReached { .. },
                Some(ack),
            ) => (
                crate::agent::tool_execution_phase::user_facing_external_action_ack(&ack),
                TaskStatus::Completed,
                TaskOutcome::Succeeded,
            ),
            _ => (reply, status, task_outcome),
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
