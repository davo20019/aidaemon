use super::completion_phase::CompletionCtx;
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;

pub(super) enum ResponsePhaseOutcome {
    ContinueLoop,
    Return(anyhow::Result<String>),
    ProceedToToolExecution,
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
    pub is_personal_memory_recall_turn: bool,
    pub is_reaffirmation_challenge_turn: bool,
    pub requests_external_verification: bool,
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
        // Soft requirement-checklist verification: when the model is finishing
        // but its self-registered checklist still has unchecked items, nudge once
        // and continue the loop; otherwise allow the finish (and post the recap).
        if matches!(outcome, ResponsePhaseOutcome::Return(Ok(_))) {
            if let Some(directive) = checklist_completion_gate(
                services.agent,
                ctx.emitter,
                ctx.session_id,
                ctx.task_id,
                ctx.iteration,
                ctx.model,
            )
            .await
            {
                ctx.pending_system_messages.push(directive);
                return Ok(ResponsePhaseOutcome::ContinueLoop);
            }
        }
        return Ok(outcome);
    }

    Ok(ResponsePhaseOutcome::ProceedToToolExecution)
}

/// Soft completion verification for the requirement checklist. Returns
/// `Some(directive)` to inject and block finishing when the current turn's
/// checklist still has unchecked items and this is the first such encounter for
/// the task; otherwise posts the done-vs-deferred recap once and returns `None`.
/// Degrades to `None` (current behavior) when no plan store / checklist exists.
async fn checklist_completion_gate(
    agent: &crate::agent::Agent,
    emitter: &crate::events::EventEmitter,
    session_id: &str,
    task_id: &str,
    iteration: usize,
    model: &str,
) -> Option<super::system_directives::SystemDirective> {
    use crate::plans::StepStatus;
    if task_id.is_empty() {
        return None;
    }
    let plan_store = agent.plan_store.read().await.clone()?;
    // Most recent checklist for this session, scoped to the current turn.
    let plan = plan_store
        .get_recent_for_session(session_id, 1)
        .await
        .ok()?
        .into_iter()
        .next()
        .filter(|p| p.task_id.as_deref() == Some(task_id))?;
    let unchecked: Vec<String> = plan
        .unchecked_steps()
        .iter()
        .map(|s| s.description.clone())
        .collect();
    let guided =
        agent.trust_tier_for_model(model) == crate::agent::trust_tier::ModelTrustTier::Guided;
    if guided
        && !unchecked.is_empty()
        && agent
            .checklist_turn_flags
            .write()
            .await
            .insert(format!("{task_id}:nudged"))
    {
        // Telemetry: the soft-verification nudge fired (model tried to finish
        // with items still unchecked). Joinable to TaskEnd by task_id.
        agent
            .emit_decision_point(
                emitter,
                task_id,
                iteration,
                crate::events::DecisionType::GateTelemetry,
                "checklist soft-verification nudge fired",
                serde_json::json!({
                    "event": "checklist_nudge",
                    "unchecked_count": unchecked.len(),
                    "items": unchecked,
                }),
            )
            .await;
        return Some(
            super::system_directives::SystemDirective::ChecklistVerificationRequired {
                items: unchecked,
            },
        );
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
    None
}
