use crate::agent::*;
use crate::execution_policy::PolicyBundle;

pub(in crate::agent) struct OrchestrationCtx<'a> {
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub task_tokens_used: u64,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub tool_defs: &'a mut Vec<Value>,
    pub base_tool_defs: &'a mut Vec<Value>,
    pub available_capabilities: &'a mut HashMap<String, ToolCapabilities>,
    pub policy_bundle: &'a mut PolicyBundle,
    pub tools_allowed_for_user: bool,
    pub llm_provider: Arc<dyn ModelProvider>,
    pub llm_router: Option<Router>,
    pub model: &'a String,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub intent_gate: &'a IntentGateDecision,
    pub intent_complexity: &'a IntentComplexity,
    pub structured_complexity_used: bool,
    pub turn_context: &'a TurnContext,
    pub execution_requirement: &'a ExecutionRequirement,
}
