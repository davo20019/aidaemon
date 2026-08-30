use crate::agent::loop_state::ReadFileObservationTracker;
use crate::agent::turn_transition::{TurnRestartReason, TurnTransition};
use crate::agent::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;

/// Ledger of argument values the execution prelude added to a tool call on the
/// model's behalf (directory capability projections). The dispatcher consults
/// it when the tool reports a runtime preparation failure: a failing value that
/// the prelude authored is the runtime's own mistake and is repaired without
/// involving the model; a failing value the model declared is surfaced to it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(in crate::agent) struct PreludeProjections {
    pub write_roots: Vec<String>,
    pub read_roots: Vec<String>,
    pub write_paths: Vec<String>,
}

impl PreludeProjections {
    pub(in crate::agent) fn contains(&self, field: &str, value: &str) -> bool {
        let values = match field {
            "write_roots" => &self.write_roots,
            "read_roots" => &self.read_roots,
            "write_paths" => &self.write_paths,
            _ => return false,
        };
        values.iter().any(|entry| entry == value)
    }
}

/// Remove one projected value from an array argument, returning the repaired
/// argument JSON. `None` when the arguments do not carry that value.
pub(in crate::agent) fn remove_projected_value(
    arguments: &str,
    field: &str,
    value: &str,
) -> Option<String> {
    let mut parsed: Value = serde_json::from_str(arguments).ok()?;
    let object = parsed.as_object_mut()?;
    let entries = object.get_mut(field)?.as_array_mut()?;
    let before = entries.len();
    entries.retain(|entry| entry.as_str() != Some(value));
    if entries.len() == before {
        return None;
    }
    serde_json::to_string(&parsed).ok()
}

/// A prelude projection that names a directory capability must describe a
/// directory. A path that already exists as something else contradicts that
/// claim, and enforcing it would break the very call it was meant to
/// authorize. Unknown (future) paths keep the projected role.
pub(in crate::agent) fn projection_conflicts_with_disk(path: &str) -> bool {
    let candidate = std::path::Path::new(path);
    candidate.exists() && !candidate.is_dir()
}

pub(in crate::agent) enum ToolExecutionOutcome {
    Return(anyhow::Result<String>),
    NextIteration,
}

impl ToolExecutionOutcome {
    pub(in crate::agent) fn into_turn_transition(self) -> TurnTransition<std::convert::Infallible> {
        match self {
            Self::Return(result) => TurnTransition::Finish(result),
            Self::NextIteration => {
                TurnTransition::Restart(TurnRestartReason::ToolExecutionCompleted)
            }
        }
    }
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
    /// Model selected for this turn — used to resolve per-model tool result caps.
    pub model: &'a str,
    pub active_skill_names: &'a [String],
    pub active_untrusted_external_reference_skills: &'a [String],
    pub restrict_untrusted_external_reference_tools: bool,
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
    pub tool_failure_signatures: &'a mut HashMap<(String, String), usize>,
    pub tool_transient_failure_count: &'a mut HashMap<String, usize>,
    pub tool_cooldown_until_iteration: &'a mut HashMap<String, usize>,
    pub tool_call_count: &'a mut HashMap<String, usize>,
    pub no_evidence_result_streak: &'a mut usize,
    pub no_evidence_tools_seen: &'a mut HashSet<String>,
    pub evidence_gain_count: &'a mut usize,
    pub pending_error_solution_ids: &'a mut Vec<i64>,
    pub tool_error_history:
        &'a mut HashMap<(String, String), Vec<super::reflection::ToolErrorEntry>>,
    pub reflection_completed: &'a mut HashSet<(String, String)>,
    pub pending_reflection_recoveries:
        &'a mut HashMap<String, super::reflection::PendingReflectionRecovery>,
    pub tool_failure_patterns: &'a mut HashMap<(String, String), usize>,
    pub last_tool_failure: &'a mut Option<(String, String)>,
    pub last_failure_class: &'a mut Option<ToolFailureClass>,
    pub in_session_learned: &'a mut HashSet<(String, String)>,
    pub unknown_tools: &'a mut HashSet<String>,
    pub recent_tool_calls: &'a mut VecDeque<u64>,
    pub consecutive_same_tool: &'a mut (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a mut HashSet<u64>,
    pub force_text_response: &'a mut bool,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub recent_tool_names: &'a mut VecDeque<String>,
    pub successful_send_file_keys: &'a mut HashSet<String>,
    pub cli_agent_boundary_injected: &'a mut bool,
    pub evidence_state: &'a mut EvidenceState,
    pub pending_background_ack: &'a mut Option<String>,
    pub pending_external_action_ack: &'a mut Option<String>,
    pub stall_count: &'a mut usize,
    pub deferred_no_tool_streak: &'a mut usize,
    pub consecutive_clean_iterations: &'a mut usize,
    pub fallback_expanded_once: &'a mut bool,
    pub known_project_dir: &'a mut Option<String>,
    pub dirs_with_project_inspect_file_evidence: &'a mut HashSet<String>,
    pub dirs_with_search_no_matches: &'a mut HashSet<String>,
    pub require_file_recheck_before_answer: &'a mut bool,
    pub completion_progress: &'a mut CompletionProgress,
    pub turn_context: &'a TurnContext,
    /// Authorized project-instruction state for pre-action nested discovery.
    pub project_instruction_tracker:
        &'a mut Option<crate::project_instructions::ProjectInstructionTracker>,
    /// Persistent system-role task tail. Newly discovered nested guidance is
    /// appended here before the triggering tool call is allowed to retry.
    pub task_context_tail: &'a mut String,
    pub resolved_goal_id: Option<&'a str>,
    /// Durable scheduled-goal provenance, resolved from goal/task state rather
    /// than inferred from the current prompt text.
    pub is_scheduled_goal: bool,
    pub execution_state: &'a mut ExecutionState,
    pub validation_state: &'a mut ValidationState,
    pub read_file_tracker: &'a mut ReadFileObservationTracker,
    /// Cache of last successful tool results keyed by call hash.
    /// Used to replay read_file/search_files content when the repetitive
    /// redirect fires, so the model retains data lost to context truncation.
    pub tool_result_cache: &'a mut HashMap<u64, String>,
    /// Correction context for this task. `None` on all normal/user-initiated
    /// paths. The P2.4 sandbox gate reads this to apply the default-deny policy.
    #[allow(dead_code)]
    pub correction:
        Option<std::sync::Arc<crate::agent::correction_execution::CorrectionExecutionContext>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_ledger_attributes_values_by_field() {
        let ledger = PreludeProjections {
            write_roots: vec!["/tmp/synthetic-root".to_string()],
            ..PreludeProjections::default()
        };
        assert!(ledger.contains("write_roots", "/tmp/synthetic-root"));
        assert!(!ledger.contains("write_paths", "/tmp/synthetic-root"));
        assert!(!ledger.contains("write_roots", "/tmp/other"));
        assert!(!ledger.contains("unknown", "/tmp/synthetic-root"));
    }

    #[test]
    fn repairing_removes_only_the_offending_projection() {
        let arguments = r#"{"action":"run","command":"true","write_paths":["/tmp/f"],"write_roots":["/tmp/d","/tmp/f"]}"#;
        let repaired = remove_projected_value(arguments, "write_roots", "/tmp/f").unwrap();
        let parsed: Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["write_roots"], serde_json::json!(["/tmp/d"]));
        assert_eq!(parsed["write_paths"], serde_json::json!(["/tmp/f"]));
        assert_eq!(parsed["command"], "true");
        assert!(remove_projected_value(arguments, "write_roots", "/tmp/absent").is_none());
        assert!(remove_projected_value(arguments, "read_roots", "/tmp/f").is_none());
    }

    #[test]
    fn disk_reality_overrides_a_directory_projection() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("result.txt");
        std::fs::write(&file, "x").unwrap();
        assert!(projection_conflicts_with_disk(&file.to_string_lossy()));
        assert!(!projection_conflicts_with_disk(
            &dir.path().to_string_lossy()
        ));
        assert!(!projection_conflicts_with_disk(
            &dir.path().join("future").to_string_lossy()
        ));
    }
}
