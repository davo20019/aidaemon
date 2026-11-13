/// Project directories of tool calls live in `aidaemon-runtime`.
mod project_dir {
    pub(in crate::agent) use aidaemon_runtime::turn::project_dir::*;
}
/// Scope checks, evidence matching and acks of a dispatched call live in
/// `aidaemon-runtime`.
mod run_helpers {
    pub(in crate::agent) use aidaemon_runtime::turn::tool_dispatch::*;
}
#[cfg(test)]
mod send_file_scope_tests;

pub(in crate::agent) use aidaemon_runtime::engine::tool_gate::{
    durable_replay_preflight, mandate_gate_for_tool_call, DurableReplayPreflight,
    MandateGateDecision,
};
pub(in crate::agent) use project_dir::project_instruction_targets_for_tool_call;
#[cfg(test)]
pub(in crate::agent) use run_helpers::access_manifest_scope_check;
pub(in crate::agent) use run_helpers::ScopeCheck;
pub(in crate::agent) use run_helpers::{
    accumulate_evidence_requirement_marker_matches, complete_tool_result_semantics,
    evidence_requirement_accepts_nonstandard_outcome, matching_evidence_requirement_indices,
    observation_matches_completion_contract, tool_result_or_metadata_contains_verifiable_evidence,
};
pub(in crate::agent) use run_helpers::{
    blocked_for_untrusted_external_reference_message, build_background_detach_ack,
    build_external_action_completion_ack, should_build_external_action_ack,
    summarize_completed_tool_result,
};
