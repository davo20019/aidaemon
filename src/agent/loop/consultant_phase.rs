use super::consultant_completion_phase::ConsultantCompletionCtx;
use super::consultant_decision_phase::ConsultantDecisionCtx;
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;

pub(super) enum ConsultantPhaseOutcome {
    ContinueLoop,
    Return(anyhow::Result<String>),
    ProceedToToolExecution,
}

pub(super) struct ConsultantPhaseCtx<'a> {
    pub resp: &'a mut ProviderResponse,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub iteration: usize,
    pub consultant_pass_active: bool,
    pub task_start: Instant,
    pub task_tokens_used: u64,
    pub learning_ctx: &'a mut LearningContext,
    pub pending_system_messages: &'a mut Vec<String>,
    pub tool_defs: &'a mut Vec<Value>,
    pub base_tool_defs: &'a mut Vec<Value>,
    pub available_capabilities: &'a mut HashMap<String, ToolCapabilities>,
    pub policy_bundle: &'a mut PolicyBundle,
    pub tools_allowed_for_user: bool,
    pub restrict_to_personal_memory_tools: bool,
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
    pub require_file_recheck_before_answer: &'a mut bool,
    pub turn_context: &'a TurnContext,
}

impl Agent {
    pub(super) async fn run_consultant_phase(
        &self,
        ctx: &mut ConsultantPhaseCtx<'_>,
    ) -> anyhow::Result<ConsultantPhaseOutcome> {
        let decision_outcome = self
            .run_consultant_decision_phase(&mut ConsultantDecisionCtx {
                resp: &mut *ctx.resp,
                emitter: ctx.emitter,
                task_id: ctx.task_id,
                session_id: ctx.session_id,
                user_text: ctx.user_text,
                iteration: ctx.iteration,
                consultant_pass_active: ctx.consultant_pass_active,
                task_start: ctx.task_start,
                task_tokens_used: ctx.task_tokens_used,
                learning_ctx: &mut *ctx.learning_ctx,
                pending_system_messages: &mut *ctx.pending_system_messages,
                tool_defs: &mut *ctx.tool_defs,
                base_tool_defs: &mut *ctx.base_tool_defs,
                available_capabilities: &mut *ctx.available_capabilities,
                policy_bundle: &mut *ctx.policy_bundle,
                tools_allowed_for_user: ctx.tools_allowed_for_user,
                restrict_to_personal_memory_tools: ctx.restrict_to_personal_memory_tools,
                is_personal_memory_recall_turn: ctx.is_personal_memory_recall_turn,
                is_reaffirmation_challenge_turn: ctx.is_reaffirmation_challenge_turn,
                requests_external_verification: ctx.requests_external_verification,
                llm_provider: ctx.llm_provider.clone(),
                llm_router: ctx.llm_router.clone(),
                model: &*ctx.model,
                user_role: ctx.user_role,
                channel_ctx: ctx.channel_ctx.clone(),
                status_tx: ctx.status_tx.clone(),
                turn_context: ctx.turn_context,
            })
            .await?;
        if let Some(outcome) = decision_outcome {
            return Ok(outcome);
        }

        let completion_outcome = self
            .run_consultant_completion_phase(&mut ConsultantCompletionCtx {
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
                restrict_to_personal_memory_tools: ctx.restrict_to_personal_memory_tools,
                llm_provider: ctx.llm_provider.clone(),
                llm_router: ctx.llm_router.clone(),
                model: &mut *ctx.model,
                channel_ctx: ctx.channel_ctx.clone(),
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
                require_file_recheck_before_answer: &mut *ctx.require_file_recheck_before_answer,
            })
            .await?;
        if let Some(outcome) = completion_outcome {
            return Ok(outcome);
        }

        Ok(ConsultantPhaseOutcome::ProceedToToolExecution)
    }
}
