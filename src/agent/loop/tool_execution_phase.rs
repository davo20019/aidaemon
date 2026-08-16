#[path = "tool_execution/phase_impl.rs"]
mod phase_impl;

pub(super) use phase_impl::project_instruction_targets_for_tool_call;
pub(super) use phase_impl::user_facing_external_action_ack;
pub(super) use phase_impl::{
    accumulate_evidence_requirement_marker_matches, complete_tool_result_semantics,
    matching_evidence_requirement_indices, observation_matches_completion_contract,
    tool_result_or_metadata_contains_verifiable_evidence,
};
pub(super) use phase_impl::{fallback_tool_semantic_scope, run_tool_execution_phase};
pub(super) use phase_impl::{PendingReflectionRecovery, ToolErrorEntry, ToolExecutionCtx};
