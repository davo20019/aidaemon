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
    apply_planned_contract_signals, apply_planned_mutation_constraints,
    apply_planned_required_mutation_effects, completion_contract_allows_force_text,
    infer_completion_contract, mutation_contract_fulfilled, parse_planned_forbidden_action,
    parse_planned_mutation_effects, parse_planned_task_kind, CompletionContract,
    CompletionProgress, CompletionTaskKind, ExecutionRequirement, ForbiddenMutationAction,
    VerificationTarget, VerificationTargetKind,
};
pub(super) use super::followup::{
    assistant_message_looks_like_clarifying_question,
    looks_like_context_dependent_followup_question, looks_like_explicit_task_switch,
    looks_like_self_contained_mutation_request, looks_like_short_command_request,
    looks_like_standalone_goal_request, looks_like_unanswered_request_reference, FollowupMode,
};
pub(super) use super::turn_context::TurnContext;
