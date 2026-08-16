use super::completion_phase::CompletionCtx;
use super::turn_transition::{TurnRestartReason, TurnTransition};
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;

pub(super) enum ResponsePhaseOutcome {
    ContinueLoop,
    Return(anyhow::Result<String>),
    ProceedToToolExecution,
}

impl ResponsePhaseOutcome {
    pub(super) fn into_response_transition(self) -> TurnTransition<()> {
        self.into_turn_transition(TurnRestartReason::ResponsePhaseRecovery)
    }

    fn into_turn_transition(self, restart_reason: TurnRestartReason) -> TurnTransition<()> {
        match self {
            Self::ContinueLoop => TurnTransition::Restart(restart_reason),
            Self::Return(result) => TurnTransition::Finish(result),
            Self::ProceedToToolExecution => TurnTransition::Advance(()),
        }
    }
}

#[allow(dead_code)]
pub(super) struct ResponsePhaseCtx<'a> {
    pub resp: &'a mut ProviderResponse,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub task_tokens_used: u64,
    pub learning_ctx: &'a mut LearningContext,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub tool_defs: &'a mut Vec<Value>,
    pub base_tool_defs: &'a mut Vec<Value>,
    pub available_capabilities: &'a mut HashMap<String, ToolCapabilities>,
    pub policy_bundle: &'a mut PolicyBundle,
    pub tools_allowed_for_user: bool,
    pub llm_provider: Arc<dyn ModelProvider>,
    pub llm_router: Option<Router>,
    pub model: &'a mut String,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub total_successful_tool_calls: usize,
    pub stall_count: &'a mut usize,
    pub consecutive_clean_iterations: &'a mut usize,
    pub deferred_no_tool_streak: &'a mut usize,
    pub deferred_no_tool_model_switches: &'a mut usize,
    pub fallback_expanded_once: &'a mut bool,
    pub empty_response_retry_used: &'a mut bool,
    pub empty_response_retry_pending: &'a mut bool,
    pub empty_response_retry_note: &'a mut Option<String>,
    pub identity_prefill_text: &'a mut Option<String>,
    pub pending_background_ack: &'a mut Option<String>,
    pub pending_external_action_ack: &'a mut Option<String>,
    pub require_file_recheck_before_answer: &'a mut bool,
    pub completion_progress: &'a mut CompletionProgress,
    pub turn_context: &'a TurnContext,
    pub execution_requirement: &'a ExecutionRequirement,
    pub force_text_response: &'a mut bool,
    pub execution_state: &'a mut ExecutionState,
    pub validation_state: &'a mut ValidationState,
}

pub(super) async fn run_response_phase(
    services: &super::services::AgentServices<'_>,
    ctx: &mut ResponsePhaseCtx<'_>,
) -> anyhow::Result<ResponsePhaseOutcome> {
    let completion_outcome = super::completion_phase::run_completion_phase(
        services,
        &mut CompletionCtx {
            resp: &mut *ctx.resp,
            emitter: ctx.emitter,
            task_id: ctx.task_id,
            session_id: ctx.session_id,
            user_text: ctx.user_text,
            iteration: ctx.iteration,
            task_start: ctx.task_start,
            learning_ctx: &mut *ctx.learning_ctx,
            pending_system_messages: &mut *ctx.pending_system_messages,
            tool_defs: &mut *ctx.tool_defs,
            base_tool_defs: &mut *ctx.base_tool_defs,
            available_capabilities: &mut *ctx.available_capabilities,
            policy_bundle: &mut *ctx.policy_bundle,
            llm_provider: ctx.llm_provider.clone(),
            llm_router: ctx.llm_router.clone(),
            model: &mut *ctx.model,
            channel_ctx: ctx.channel_ctx.clone(),
            user_role: ctx.user_role,
            total_successful_tool_calls: ctx.total_successful_tool_calls,
            stall_count: &mut *ctx.stall_count,
            consecutive_clean_iterations: &mut *ctx.consecutive_clean_iterations,
            deferred_no_tool_streak: &mut *ctx.deferred_no_tool_streak,
            deferred_no_tool_model_switches: &mut *ctx.deferred_no_tool_model_switches,
            fallback_expanded_once: &mut *ctx.fallback_expanded_once,
            empty_response_retry_used: &mut *ctx.empty_response_retry_used,
            empty_response_retry_pending: &mut *ctx.empty_response_retry_pending,
            empty_response_retry_note: &mut *ctx.empty_response_retry_note,
            identity_prefill_text: &mut *ctx.identity_prefill_text,
            pending_background_ack: &mut *ctx.pending_background_ack,
            pending_external_action_ack: &mut *ctx.pending_external_action_ack,
            require_file_recheck_before_answer: &mut *ctx.require_file_recheck_before_answer,
            completion_progress: &mut *ctx.completion_progress,
            turn_context: ctx.turn_context,
            execution_requirement: ctx.execution_requirement,
            force_text_response: &mut *ctx.force_text_response,
            execution_state: &mut *ctx.execution_state,
            validation_state: &mut *ctx.validation_state,
        },
    )
    .await?;
    if let Some(outcome) = completion_outcome {
        // Checklists are a projection of the typed completion graph, not a
        // second completion authority.
        if matches!(outcome, ResponsePhaseOutcome::Return(Ok(_))) {
            reconcile_checklist_projection(
                services.agent,
                ctx.emitter,
                ctx.session_id,
                ctx.task_id,
                ctx.iteration,
                ctx.model,
            )
            .await;
        }
        return Ok(outcome);
    }

    Ok(ResponsePhaseOutcome::ProceedToToolExecution)
}

/// Reconcile the display checklist after the authoritative task proof closes.
/// Untyped prose-only steps are deferred rather than promoted to evidence or
/// allowed to trigger another model/validation cycle.
async fn reconcile_checklist_projection(
    agent: &crate::agent::Agent,
    emitter: &crate::events::EventEmitter,
    session_id: &str,
    task_id: &str,
    iteration: usize,
    _model: &str,
) {
    use crate::plans::StepStatus;
    if task_id.is_empty() {
        return;
    }
    let Some(plan_store) = agent.plan_store.read().await.clone() else {
        return;
    };
    // Most recent checklist for this session, scoped to the current turn.
    let Some(mut plan) = plan_store
        .get_recent_for_session(session_id, 1)
        .await
        .ok()
        .and_then(|plans| plans.into_iter().next())
        .filter(|p| p.task_id.as_deref() == Some(task_id))
    else {
        return;
    };
    let projection_changed = reconcile_plan_projection(&mut plan);
    if projection_changed {
        if let Err(error) = plan_store.update(&plan).await {
            tracing::warn!(task_id, %error, "Failed to reconcile checklist projection");
        }
    }
    let unchecked: Vec<String> = plan
        .unchecked_steps()
        .iter()
        .map(|s| s.description.clone())
        .collect();
    if !unchecked.is_empty() {
        agent
            .emit_decision_point(
                emitter,
                task_id,
                iteration,
                crate::events::DecisionType::GateTelemetry,
                "checklist projection remains incomplete",
                serde_json::json!({
                    "event": "checklist_projection_incomplete",
                    "unchecked_count": unchecked.len(),
                    "items": unchecked,
                }),
            )
            .await;
    }
    // Allow finishing — emit completion stats + post the recap once per task.
    if agent
        .checklist_turn_flags
        .write()
        .await
        .insert(format!("{task_id}:recapped"))
    {
        let completed = plan.completed_steps();
        let deferred = plan
            .steps
            .iter()
            .filter(|s| s.status == StepStatus::Deferred)
            .count();
        agent
            .emit_decision_point(
                emitter,
                task_id,
                iteration,
                crate::events::DecisionType::GateTelemetry,
                "checklist turn completed",
                serde_json::json!({
                    "event": "checklist_complete",
                    "total": plan.steps.len(),
                    "completed": completed,
                    "deferred": deferred,
                    "unchecked": unchecked.len(),
                }),
            )
            .await;
        // The final checklist state is surfaced through the single live status
        // surface (StatusUpdate::Checklist), emitted from the tool-execution loop
        // as the checklist evolves. Final-answer delivery is owned elsewhere, so
        // this phase no longer posts/edits channel messages directly.
    }
}

fn reconcile_plan_projection(plan: &mut crate::plans::TaskPlan) -> bool {
    use crate::plans::StepStatus;
    let mut projection_changed = false;
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
            projection_changed = true;
        }
    }
    if projection_changed {
        plan.sync_active_step();
    }
    projection_changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advisory_checklist_cannot_reopen_a_closed_typed_task() {
        let mut plan = crate::plans::TaskPlan::new(
            "telegram:synthetic",
            "synthetic request",
            "synthetic plan",
            vec!["Summarize the already proven result".to_string()],
            "test",
        );
        assert!(reconcile_plan_projection(&mut plan));
        assert_eq!(plan.steps[0].status, crate::plans::StepStatus::Deferred);
        assert!(plan.is_finished());
    }

    #[test]
    fn typed_checklist_step_remains_telemetry_incomplete_without_gating() {
        let mut plan = crate::plans::TaskPlan::new(
            "telegram:synthetic",
            "synthetic request",
            "synthetic plan",
            vec!["Observe canonical state".to_string()],
            "test",
        );
        plan.steps[0].requires_observation = true;
        assert!(!reconcile_plan_projection(&mut plan));
        assert_eq!(plan.steps[0].status, crate::plans::StepStatus::InProgress);
    }
}
