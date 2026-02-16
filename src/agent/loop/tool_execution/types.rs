use crate::agent::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;

pub(in crate::agent) enum ToolExecutionOutcome {
    Return(anyhow::Result<String>),
    NextIteration,
}

/// A JoinHandle wrapper that aborts the task when dropped.
/// Standard `JoinHandle::drop()` detaches the task (it keeps running);
/// this ensures background tasks like the heartbeat keeper are cleaned up
/// if the parent future is cancelled by an outer `select!`.
pub(super) struct AbortOnDrop(pub(super) tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

pub(in crate::agent) struct ToolExecutionCtx<'a> {
    pub resp: &'a ProviderResponse,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub learning_ctx: &'a mut LearningContext,
    pub task_tokens_used: u64,
    pub user_text: &'a str,
    pub restrict_to_personal_memory_tools: bool,
    pub is_reaffirmation_challenge_turn: bool,
    pub personal_memory_tool_call_cap: usize,
    pub base_tool_defs: &'a Vec<Value>,
    pub available_capabilities: &'a HashMap<String, ToolCapabilities>,
    pub policy_bundle: &'a PolicyBundle,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub channel_ctx: &'a ChannelContext,
    pub user_role: UserRole,
    pub heartbeat: &'a Option<Arc<AtomicU64>>,
    pub tool_defs: &'a mut Vec<Value>,
    pub total_tool_calls_attempted: &'a mut usize,
    pub total_successful_tool_calls: &'a mut usize,
    pub tool_failure_count: &'a mut HashMap<String, usize>,
    pub tool_call_count: &'a mut HashMap<String, usize>,
    pub personal_memory_tool_calls: &'a mut usize,
    pub no_evidence_result_streak: &'a mut usize,
    pub no_evidence_tools_seen: &'a mut HashSet<String>,
    pub evidence_gain_count: &'a mut usize,
    pub pending_error_solution_ids: &'a mut Vec<i64>,
    pub tool_failure_patterns: &'a mut HashMap<(String, String), usize>,
    pub last_tool_failure: &'a mut Option<(String, String)>,
    pub in_session_learned: &'a mut HashSet<(String, String)>,
    pub unknown_tools: &'a mut HashSet<String>,
    pub recent_tool_calls: &'a mut VecDeque<u64>,
    pub consecutive_same_tool: &'a mut (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a mut HashSet<u64>,
    pub force_text_response: &'a mut bool,
    pub pending_system_messages: &'a mut Vec<String>,
    pub recent_tool_names: &'a mut VecDeque<String>,
    pub successful_send_file_keys: &'a mut HashSet<String>,
    pub cli_agent_boundary_injected: &'a mut bool,
    pub stall_count: &'a mut usize,
    pub deferred_no_tool_streak: &'a mut usize,
    pub consecutive_clean_iterations: &'a mut usize,
    pub fallback_expanded_once: &'a mut bool,
    pub known_project_dir: &'a mut Option<String>,
    pub dirs_with_project_inspect_file_evidence: &'a mut HashSet<String>,
    pub dirs_with_search_no_matches: &'a mut HashSet<String>,
    pub require_file_recheck_before_answer: &'a mut bool,
}
