#[path = "tool_execution/phase_impl.rs"]
mod phase_impl;

pub(super) use phase_impl::project_instruction_targets_for_tool_call;
pub(super) use phase_impl::user_facing_external_action_ack;
pub(super) use phase_impl::{
    accumulate_evidence_requirement_marker_matches, complete_tool_result_semantics,
    evidence_requirement_accepts_nonstandard_outcome, matching_evidence_requirement_indices,
    observation_matches_completion_contract, tool_result_or_metadata_contains_verifiable_evidence,
};
#[cfg(feature = "sdk-engine")]
pub(super) use phase_impl::{build_background_detach_ack, tool_result_indicates_background_detach};
#[cfg(feature = "sdk-engine")]
pub(super) use phase_impl::{
    build_external_action_completion_ack, mandate_gate_for_tool_call, parse_checklist_from_result,
    should_build_external_action_ack, summarize_completed_tool_result, MandateGateDecision,
};
pub(super) use phase_impl::{fallback_tool_semantic_scope, run_tool_execution_phase};
pub(super) use phase_impl::{PendingReflectionRecovery, ToolErrorEntry, ToolExecutionCtx};
