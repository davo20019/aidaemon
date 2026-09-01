//! Ledger-first closeout: Authority ∩ visible tools decides what an
//! expectation could still be satisfied by; the receipt ledger decides what
//! is proven. This module owns the admissibility predicate the ledger
//! arbiter (`RunAggregate::closeout`) is parameterized with.
//!
//! Authority (forbids, allowed tools, forbidden scopes) is fail-closed and
//! contract-owned. Expectations (obligations) are proposals: they may ask the
//! loop for more work only when a tool admissible under Authority could
//! satisfy them. Nothing here reads request wording.

use crate::agent::completion_contract::RequestAuthority;
use crate::events::RunObligation;
use crate::traits::ToolSemanticScope;

/// Static authority view used to test whether a tool may still be invoked.
pub(super) struct ClosingAuthority<'a> {
    pub forbids_tool_use: bool,
    pub forbids_mutation: bool,
    pub allowed_tool_names: &'a [String],
    pub forbidden_tool_scopes: &'a [ToolSemanticScope],
}

impl<'a> ClosingAuthority<'a> {
    pub(super) fn from_authority(authority: RequestAuthority<'a>) -> Self {
        Self {
            forbids_tool_use: authority.forbids_tool_use,
            forbids_mutation: authority.forbids_mutation,
            allowed_tool_names: authority.allowed_tool_names,
            forbidden_tool_scopes: authority.forbidden_tool_scopes,
        }
    }

    /// Whether Authority admits invoking `tool` at all. `read_only` is the
    /// registered tool's declared capability (None for unknown tools, which
    /// stay admissible so a custom tool is never declared unavailable
    /// prematurely).
    pub(super) fn admits_tool(&self, tool: &str, read_only: Option<bool>) -> bool {
        if self.forbids_tool_use {
            return false;
        }
        let explicitly_allowed = self.allowed_tool_names.iter().any(|name| name == tool);
        if !self.allowed_tool_names.is_empty() && !explicitly_allowed {
            return false;
        }
        // Exact beats broad: a tool the request named explicitly is admitted
        // even when a scope-level prohibition would otherwise cover it (the
        // same rule as an exact write path beating a directory root).
        if !explicitly_allowed {
            if let Some(scope) =
                crate::agent::tool_execution_phase::fallback_tool_semantic_scope(tool)
            {
                if self.forbidden_tool_scopes.contains(&scope) {
                    return false;
                }
            }
        }
        if self.forbids_mutation && read_only == Some(false) {
            return false;
        }
        true
    }
}

/// Whether some visible tool admissible under `authority` could still
/// satisfy `obligation`.
pub(super) fn obligation_admissible(
    obligation: &RunObligation,
    authority: &ClosingAuthority<'_>,
    visible_tools: &[&str],
    read_only: impl Fn(&str) -> Option<bool>,
) -> bool {
    let admits = |tool: &str| authority.admits_tool(tool, read_only(tool));
    if let Some(predicate) = obligation.receipt.as_ref() {
        if !predicate.tool_names.is_empty() {
            return predicate
                .tool_names
                .iter()
                .any(|tool| visible_tools.contains(&tool.as_str()) && admits(tool));
        }
    }
    if let Some(requirement) = obligation.evidence_requirement.as_ref() {
        let candidates = crate::agent::inquiry::candidate_tools_for_requirements(
            std::slice::from_ref(requirement),
            visible_tools.iter().copied(),
        );
        if !candidates.is_empty() {
            return candidates.iter().any(|tool| admits(tool));
        }
        // Unknown/dynamic tools without a static evidence model remain
        // possible candidates rather than being declared unavailable.
        return visible_tools
            .iter()
            .any(|tool| !crate::agent::inquiry::has_static_evidence_model(tool) && admits(tool));
    }
    // A mutation obligation with no exact predicate: any admissible mutating
    // tool could still perform it unless Authority forbids mutation outright.
    if !obligation.required_effect.is_empty() {
        return !authority.forbids_mutation
            && visible_tools
                .iter()
                .any(|tool| read_only(tool) != Some(true) && admits(tool));
    }
    visible_tools.iter().any(|tool| admits(tool))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{RunObligationClass, RunObligationState};
    use crate::traits::{
        EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
        RequestReceiptPredicate, ToolMutationEffects,
    };

    fn obligation(
        receipt: Option<RequestReceiptPredicate>,
        requirement: Option<RequestEvidenceRequirement>,
        effect: ToolMutationEffects,
    ) -> RunObligation {
        RunObligation {
            id: "task:t/obligation:evidence:0".to_string(),
            class: RunObligationClass::Observe,
            state: RunObligationState::Pending,
            receipt,
            evidence_requirement: requirement,
            required_effect: effect,
            satisfied_at_revision: None,
            satisfying_receipt_ids: Vec::new(),
            required_target: None,
            observation_targets: Vec::new(),
            summary: None,
        }
    }

    fn authority<'a>(
        forbids_mutation: bool,
        forbidden: &'a [ToolSemanticScope],
    ) -> ClosingAuthority<'a> {
        ClosingAuthority {
            forbids_tool_use: false,
            forbids_mutation,
            allowed_tool_names: &[],
            forbidden_tool_scopes: forbidden,
        }
    }

    #[test]
    fn exact_invocation_is_reachable_only_through_a_visible_admissible_tool() {
        let predicate = RequestReceiptPredicate {
            tool_names: vec!["write_file".to_string()],
            ..RequestReceiptPredicate::default()
        };
        let obligation = obligation(Some(predicate), None, ToolMutationEffects::NONE);
        let read_only = |tool: &str| Some(tool != "write_file");
        assert!(obligation_admissible(
            &obligation,
            &authority(false, &[]),
            &["write_file", "read_file"],
            read_only
        ));
        // Not visible.
        assert!(!obligation_admissible(
            &obligation,
            &authority(false, &[]),
            &["read_file"],
            read_only
        ));
        // Visible but the read-only contract forbids the mutating tool.
        assert!(!obligation_admissible(
            &obligation,
            &authority(true, &[]),
            &["write_file"],
            read_only
        ));
    }

    #[test]
    fn an_explicitly_allowed_tool_beats_a_scope_level_prohibition() {
        let predicate = RequestReceiptPredicate {
            tool_names: vec!["terminal".to_string()],
            ..RequestReceiptPredicate::default()
        };
        let obligation = obligation(Some(predicate), None, ToolMutationEffects::NONE);
        let allowed = vec!["terminal".to_string()];
        let authority = ClosingAuthority {
            forbids_tool_use: false,
            forbids_mutation: false,
            allowed_tool_names: &allowed,
            forbidden_tool_scopes: &[ToolSemanticScope::LocalWorkspace],
        };
        assert!(obligation_admissible(
            &obligation,
            &authority,
            &["terminal", "read_file"],
            |_| Some(false)
        ));
        let none_allowed: Vec<String> = Vec::new();
        let authority = ClosingAuthority {
            allowed_tool_names: &none_allowed,
            ..authority
        };
        assert!(!obligation_admissible(
            &obligation,
            &authority,
            &["terminal", "read_file"],
            |_| Some(false)
        ));
    }

    #[test]
    fn observation_is_reachable_through_the_static_evidence_model_and_scopes() {
        let requirement = RequestEvidenceRequirement {
            summary: "current goal state".to_string(),
            acceptable_scopes: vec![ToolSemanticScope::GoalState],
            purpose: EvidencePurpose::CurrentState,
            minimum_authority: EvidenceAuthority::Canonical,
            temporal_scope: EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        };
        let obligation = obligation(None, Some(requirement), ToolMutationEffects::NONE);
        let read_only = |_: &str| Some(true);
        assert!(obligation_admissible(
            &obligation,
            &authority(true, &[]),
            &["manage_mandates", "read_file"],
            read_only
        ));
        // The scope is forbidden by authority: unreachable.
        assert!(!obligation_admissible(
            &obligation,
            &authority(true, &[ToolSemanticScope::GoalState]),
            &["manage_mandates", "read_file"],
            read_only
        ));
        // No tool for that scope is visible.
        assert!(!obligation_admissible(
            &obligation,
            &authority(true, &[]),
            &["read_file"],
            read_only
        ));
    }

    #[test]
    fn mutation_obligation_is_unreachable_under_a_read_only_contract() {
        let obligation = obligation(None, None, ToolMutationEffects::LOCAL_WORKSPACE_WRITE);
        let read_only = |tool: &str| Some(tool == "read_file");
        assert!(obligation_admissible(
            &obligation,
            &authority(false, &[]),
            &["terminal", "read_file"],
            read_only
        ));
        assert!(!obligation_admissible(
            &obligation,
            &authority(true, &[]),
            &["terminal", "read_file"],
            read_only
        ));
    }
}
