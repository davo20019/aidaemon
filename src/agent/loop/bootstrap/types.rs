use crate::agent::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ConversationSummary;

pub(in crate::agent) enum BootstrapOutcome {
    Return(anyhow::Result<String>),
    Continue(Box<BootstrapData>),
}

pub(in crate::agent) struct BootstrapCtx<'a> {
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub user_role: UserRole,
    pub channel_ctx: &'a ChannelContext,
}

pub(in crate::agent) struct BootstrapData {
    pub task_id: String,
    pub emitter: crate::events::EventEmitter,
    pub learning_ctx: LearningContext,
    pub is_personal_memory_recall_turn: bool,
    pub is_reaffirmation_challenge_turn: bool,
    pub requests_external_verification: bool,
    pub restrict_to_personal_memory_tools: bool,
    pub active_skill_names: Vec<String>,
    pub active_untrusted_external_reference_skills: Vec<String>,
    pub restrict_untrusted_external_reference_tools: bool,
    pub personal_memory_tool_call_cap: usize,
    pub tools_allowed_for_user: bool,
    pub available_capabilities: HashMap<String, ToolCapabilities>,
    pub base_tool_defs: Vec<Value>,
    pub tool_defs: Vec<Value>,
    pub policy_bundle: PolicyBundle,
    pub llm_provider: Arc<dyn ModelProvider>,
    pub llm_router: Option<Router>,
    pub model: String,
    pub route_failsafe_active: bool,
    /// Pillar A: message-zero bytes (the session-static CORE prompt). Cacheable
    /// prefix; rendered once per task by `render_core_prompt`.
    pub core_prompt_bytes: String,
    /// Pillar A: the per-task volatile context tail (timestamp, session context,
    /// memories, matched skill bodies, resume checkpoint, …). Compiled once per
    /// task and reused byte-identically across the within-task loop. Inserted at
    /// the boundary − 1 position (immediately before the current user message).
    pub task_context_tail: String,
    pub pinned_memories: Vec<Message>,
    pub session_summary: Option<ConversationSummary>,
}
