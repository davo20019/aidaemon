use super::consultant_intent_gate_phase::{ConsultantIntentGateCtx, ConsultantIntentGateOutcome};
use super::consultant_orchestration_phase::ConsultantOrchestrationCtx;
use super::consultant_phase::ConsultantPhaseOutcome;
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;

pub(super) struct ConsultantDecisionCtx<'a> {
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
    pub model: &'a String,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub turn_context: &'a TurnContext,
}

impl Agent {
    pub(super) async fn run_consultant_decision_phase(
        &self,
        ctx: &mut ConsultantDecisionCtx<'_>,
    ) -> anyhow::Result<Option<ConsultantPhaseOutcome>> {
        // Consultant decision routing only runs on iteration 1 for top-level orchestrator.
        if ctx.iteration != 1 || !ctx.consultant_pass_active {
            return Ok(None);
        }

        let intent_outcome = self
            .run_consultant_intent_gate_phase(&mut ConsultantIntentGateCtx {
                resp: &mut *ctx.resp,
                emitter: ctx.emitter,
                task_id: ctx.task_id,
                session_id: ctx.session_id,
                user_text: ctx.user_text,
                iteration: ctx.iteration,
                task_start: ctx.task_start,
                learning_ctx: &mut *ctx.learning_ctx,
                is_personal_memory_recall_turn: ctx.is_personal_memory_recall_turn,
                is_reaffirmation_challenge_turn: ctx.is_reaffirmation_challenge_turn,
                requests_external_verification: ctx.requests_external_verification,
            })
            .await?;

        let intent_gate = match intent_outcome {
            ConsultantIntentGateOutcome::Return(outcome) => return Ok(Some(outcome)),
            ConsultantIntentGateOutcome::Continue(data) => data.intent_gate,
        };

        self.run_consultant_orchestration_phase(&mut ConsultantOrchestrationCtx {
            emitter: ctx.emitter,
            task_id: ctx.task_id,
            session_id: ctx.session_id,
            user_text: ctx.user_text,
            iteration: ctx.iteration,
            task_start: ctx.task_start,
            task_tokens_used: ctx.task_tokens_used,
            pending_system_messages: &mut *ctx.pending_system_messages,
            tool_defs: &mut *ctx.tool_defs,
            base_tool_defs: &mut *ctx.base_tool_defs,
            available_capabilities: &mut *ctx.available_capabilities,
            policy_bundle: &mut *ctx.policy_bundle,
            tools_allowed_for_user: ctx.tools_allowed_for_user,
            restrict_to_personal_memory_tools: ctx.restrict_to_personal_memory_tools,
            llm_provider: ctx.llm_provider.clone(),
            llm_router: ctx.llm_router.clone(),
            model: ctx.model,
            user_role: ctx.user_role,
            channel_ctx: ctx.channel_ctx.clone(),
            status_tx: ctx.status_tx.clone(),
            intent_gate: &intent_gate,
            turn_context: ctx.turn_context,
        })
        .await
    }
}
