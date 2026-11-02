mod budget_blocking;
mod execution_io;
mod guards;
mod post_loop;
mod project_dir;
mod reflection;
mod result_learning;
mod run;
mod run_helpers;
mod types;

pub(in crate::agent) use execution_io::{durable_replay_preflight, DurableReplayPreflight};
pub(in crate::agent) use project_dir::{
    maybe_inject_project_dir_into_tool_args, project_dir_from_tool_args,
    project_instruction_targets_for_tool_call,
};
pub(in crate::agent) use reflection::{PendingReflectionRecovery, ToolErrorEntry};
pub(in crate::agent) use run::run_tool_execution_phase;
pub(in crate::agent) use run::{
    mandate_gate_for_tool_call, parse_checklist_from_result, MandateGateDecision,
};
pub(in crate::agent) use run_helpers::{
    access_manifest_scope_check, attenuate_unauthorized_read_roots, ScopeCheck,
};
pub(in crate::agent) use run_helpers::{
    accumulate_evidence_requirement_marker_matches, complete_tool_result_semantics,
    evidence_requirement_accepts_nonstandard_outcome, fallback_tool_semantic_scope,
    matching_evidence_requirement_indices, observation_matches_completion_contract,
    tool_result_or_metadata_contains_verifiable_evidence, user_facing_external_action_ack,
};
pub(in crate::agent) use run_helpers::{
    blocked_for_untrusted_external_reference_message, build_background_detach_ack,
    build_external_action_completion_ack, should_build_external_action_ack,
    summarize_completed_tool_result, tool_result_indicates_background_detach,
};
pub(in crate::agent) use types::ToolExecutionCtx;
