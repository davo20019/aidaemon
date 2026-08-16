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
    pub attachments: &'a [crate::traits::MessageAttachment],
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub user_role: UserRole,
    pub channel_ctx: &'a ChannelContext,
    /// True for runtime-generated background-result continuations. These turns
    /// carry evidence for the existing request, not fresh user intent.
    pub internal_continuation: bool,
    /// Explicit parent for a runtime-generated continuation. Ordinary ingress
    /// and language-classified follow-ups leave this empty.
    pub parent_task_id: Option<&'a str>,
}

pub(in crate::agent) struct BootstrapData {
    /// Canonical user message persisted for this turn. This is normally the
    /// user's exact text; when STT fallback runs it also contains the appended
    /// transcription. Downstream phases must use this value rather than the
    /// pre-bootstrap input so prompt assembly matches durable history.
    pub user_text: String,
    pub task_id: String,
    /// One semantic assessment compiled before optional memory access. The
    /// main loop consumes this exact artifact rather than issuing a second
    /// classifier call that could disagree with the memory gate.
    pub task_plan: Option<super::task_planning::TaskPlan>,
    pub task_assessment_attempted: bool,
    pub memory_pipeline_policy: super::task_planning::MemoryPipelinePolicy,
    /// Durable execution identity recovered from an interrupted run. The new
    /// turn reuses it so operation-derived idempotency keys remain stable.
    pub resume_execution_snapshot: Option<ResumeExecutionSnapshot>,
    pub emitter: crate::events::EventEmitter,
    pub learning_ctx: LearningContext,
    pub is_reaffirmation_challenge_turn: bool,
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
    /// Canonical per-turn scope/follow-up/contract snapshot. Bootstrap resolves
    /// it before prompt construction so scoped project instructions and the
    /// execution loop use the exact same project decision.
    pub turn_context: TurnContext,
    /// Authorized, task-local instruction hierarchy state. `None` for turns
    /// without a single trusted project scope; otherwise used to discover
    /// deeper instruction files before the first tool action in that subtree.
    pub project_instruction_tracker: Option<crate::project_instructions::ProjectInstructionTracker>,
    /// Pillar A: message-zero bytes (the session-static CORE prompt). Cacheable
    /// prefix; rendered once per task by `render_core_prompt`.
    pub core_prompt_bytes: String,
    /// Pillar A: context tails compiled from one snapshot. The finalized typed
    /// task relationship selects one after semantic assessment: fresh tasks
    /// exclude conversation summaries/activity, while continuations retain the
    /// current task thread. Just-in-time project instructions are appended only
    /// to the selected tail.
    pub fresh_task_context_tail: String,
    pub continuation_task_context_tail: String,
    pub session_summary: Option<ConversationSummary>,
    pub harness_eval: HarnessEvalAccumulator,
}
