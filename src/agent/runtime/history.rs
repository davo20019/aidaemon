//! History/turn-context facade.
//!
//! Phase 4 of the decoupling campaign split the former monolithic
//! `runtime/history.rs` into four concern-focused siblings:
//!
//! - [`followup`](super::followup) — follow-up / task-switch classification.
//! - [`completion_contract`](super::completion_contract) — completion-contract inference.
//! - [`turn_context`](super::turn_context) — turn-context assembly + history I/O.
//! - [`notes`](super::notes) — auxiliary note recording.
//!
//! This module is now a re-export shim so consumers can keep importing the
//! same names through `history::`.

pub(super) use super::completion_contract::{
    authored_artifact_still_needs_delivery_recovery, completion_contract_allows_force_text,
    inherit_unfinished_request_contract, install_semantic_completion_contract,
    mutation_contract_fulfilled, parse_planned_forbidden_action, parse_planned_mutation_effects,
    parse_planned_task_kind, retain_structural_completion_contract, CompletionContract,
    CompletionProgress, CompletionTaskKind, ExecutionRequirement, ForbiddenMutationAction,
    SemanticCompletionRequirements, VerificationTarget, VerificationTargetKind,
};
pub(super) use super::followup::{assistant_message_looks_like_clarifying_question, FollowupMode};
pub(super) use super::turn_context::TurnContext;
