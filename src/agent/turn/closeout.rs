//! Ledger-first closeout: Authority ∩ visible tools decides what an
//! expectation could still be satisfied by; the receipt ledger decides what
//! is proven. This module owns the admissibility predicate the ledger
//! arbiter (`RunAggregate::closeout`) is parameterized with.
//!
//! Authority is the turn's resolved authority (`agent::authority`, see
//! `ClosingAuthority::from_resolved`), never the assessed contract.
//! Expectations (obligations) are proposals: they may ask the
//! loop for more work only when a tool admissible under Authority could
//! satisfy them. Nothing here reads request wording.

use crate::agent::completion_contract::RequestAuthority;
use crate::events::RunObligation;
use crate::traits::ToolSemanticScope;

/// Static authority view used to test whether a tool may still be invoked.
pub(in crate::agent) struct ClosingAuthority<'a> {
    pub forbids_tool_use: bool,
    pub forbids_mutation: bool,
    pub allowed_tool_names: &'a [String],
    pub forbidden_tool_scopes: &'a [ToolSemanticScope],
    pub owner_control_plane_tools: &'a [String],
}

impl ClosingAuthority<'static> {
    /// The turn's resolved authority: read-only forbids mutation, an empty
    /// tool set forbids tool use. Offered tools were already filtered to the
    /// authority's tool set, so no allowlist or scope applies here.
    pub(in crate::agent) fn from_resolved(authority: &aidaemon_core::Authority) -> Self {
        Self {
            forbids_tool_use: authority
                .tools
                .as_ref()
                .is_some_and(|tools| tools.is_empty()),
            forbids_mutation: authority.level == aidaemon_core::AccessLevel::ReadOnly,
            allowed_tool_names: &[],
            forbidden_tool_scopes: &[],
            owner_control_plane_tools: &[],
        }
    }
}

impl<'a> ClosingAuthority<'a> {
    /// The assessed contract's authority view. No longer enforced; kept for
    /// shadow comparison against the resolved authority.
    #[allow(dead_code)]
    pub(in crate::agent) fn from_authority(authority: RequestAuthority<'a>) -> Self {
        Self {
            forbids_tool_use: authority.forbids_tool_use,
            forbids_mutation: authority.forbids_mutation,
            allowed_tool_names: authority.allowed_tool_names,
            forbidden_tool_scopes: authority.forbidden_tool_scopes,
            owner_control_plane_tools: authority.owner_control_plane_tools,
        }
    }

    /// The tool is admitted only because the runtime kept it as an owner
    /// control-plane tool; the caller must also check the exact call
    /// (`Tool::owner_control_plane_call`).
    pub(in crate::agent) fn admits_only_as_control_plane(&self, tool: &str) -> bool {
        !self.allowed_tool_names.is_empty()
            && !self.allowed_tool_names.iter().any(|name| name == tool)
            && self
                .owner_control_plane_tools
                .iter()
                .any(|name| name == tool)
    }

    /// Whether Authority admits invoking `tool` at all. `read_only` is the
    /// registered tool's declared capability (None for unknown tools, which
    /// stay admissible so a custom tool is never declared unavailable
    /// prematurely).
    pub(in crate::agent) fn admits_tool(&self, tool: &str, read_only: Option<bool>) -> bool {
        if self.forbids_tool_use {
            return false;
        }
        let explicitly_allowed = self.allowed_tool_names.iter().any(|name| name == tool);
        let control_plane = self
            .owner_control_plane_tools
            .iter()
            .any(|name| name == tool);
        if !self.allowed_tool_names.is_empty() && !explicitly_allowed && !control_plane {
            return false;
        }
        // Exact beats broad: a tool the request named explicitly is admitted
        // even when a scope-level prohibition would otherwise cover it (the
        // same rule as an exact write path beating a directory root).
        if !explicitly_allowed {
            if let Some(scope) = crate::agent::tool_execution::fallback_tool_semantic_scope(tool) {
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
pub(in crate::agent) fn obligation_admissible(
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
        // Candidate routing is ranked and capped for model guidance. Apply
        // authority first so unavailable candidates cannot crowd out an
        // admissible route or suppress the custom-tool fallback below.
        let candidates = crate::agent::inquiry::candidate_tools_for_requirements(
            std::slice::from_ref(requirement),
            visible_tools.iter().copied().filter(|tool| admits(tool)),
        );
        if !candidates.is_empty() {
            return true;
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

/// A bounded extra pass requested at stop time by something other than the
/// ledger's expectations. These share the ledger's rule: the loop may ask for
/// more only when something admissible could satisfy the request, and only
/// within the demand's budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::agent) enum StopDemand {
    /// A mandate task lead must commit its decision with `manage_mandates
    /// record_decision`; its prose is never the decision.
    MandateDecisionCommit,
    /// The run's last GUI coordinate click is unverified; an observation by
    /// the GUI tool that made the click confirms it.
    #[cfg_attr(not(feature = "computer_use"), allow(dead_code))]
    GuiClickVerification,
    /// The final reply cannot ship (empty, internal markers, pasted tool
    /// output); a rewrite needs no tool.
    ReplyRepair,
}

/// The exact call that satisfies `StopDemand::MandateDecisionCommit`.
pub(in crate::agent) const MANDATE_DECISION_TOOL: &str = "manage_mandates";
pub(in crate::agent) const MANDATE_DECISION_CALL: &str = r#"{"action":"record_decision"}"#;

impl StopDemand {
    /// The decision-point condition each verdict is recorded under.
    pub(in crate::agent) fn condition(self) -> &'static str {
        match self {
            Self::MandateDecisionCommit => "mandate_decision_commit_missing",
            Self::GuiClickVerification => "gui_coordinate_click_unverified",
            Self::ReplyRepair => "final_reply_defect",
        }
    }

    /// How many times one run may ask for this demand.
    pub(in crate::agent) fn budget(self) -> u32 {
        match self {
            Self::MandateDecisionCommit | Self::ReplyRepair => 1,
            Self::GuiClickVerification => 3,
        }
    }
}

/// The arbiter's answer for one stop demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::agent) enum StopDemandVerdict {
    /// Ask the model for one more pass.
    Demand,
    /// This run already used the demand's budget.
    Exhausted,
    /// Nothing admissible could satisfy it, so asking would only spend a
    /// model call on a refusal.
    Unreachable,
}

impl StopDemandVerdict {
    pub(in crate::agent) fn as_str(self) -> &'static str {
        match self {
            Self::Demand => "demand",
            Self::Exhausted => "exhausted",
            Self::Unreachable => "unreachable",
        }
    }
}

/// Whether the loop may ask for `demand`, after `used` earlier asks this run.
/// `control_plane_call(tool, arguments)` answers whether a tool kept only as
/// an owner control-plane tool admits that exact call, as the call gate asks.
pub(in crate::agent) fn stop_demand_verdict(
    demand: StopDemand,
    used: u32,
    authority: &ClosingAuthority<'_>,
    visible_tools: &[&str],
    read_only: impl Fn(&str) -> Option<bool>,
    control_plane_call: impl Fn(&str, &str) -> bool,
) -> StopDemandVerdict {
    let reachable = match demand {
        StopDemand::MandateDecisionCommit => {
            let tool = MANDATE_DECISION_TOOL;
            visible_tools.contains(&tool)
                && authority.admits_tool(tool, read_only(tool))
                && (!authority.admits_only_as_control_plane(tool)
                    || control_plane_call(tool, MANDATE_DECISION_CALL))
        }
        // The click came from a tool this run could call, and observing is
        // read-only, so only a request that forbids tools outright blocks it.
        StopDemand::GuiClickVerification => !authority.forbids_tool_use,
        StopDemand::ReplyRepair => true,
    };
    if !reachable {
        StopDemandVerdict::Unreachable
    } else if used >= demand.budget() {
        StopDemandVerdict::Exhausted
    } else {
        StopDemandVerdict::Demand
    }
}

#[cfg(test)]
mod stop_demand_tests {
    use super::*;

    /// The resolved authority's level and tool set become the closing view:
    /// read-only forbids mutation, an empty set forbids tools, nothing else.
    #[test]
    fn closing_authority_from_resolved_reads_level_and_tool_set() {
        let full = ClosingAuthority::from_resolved(&aidaemon_core::Authority::full());
        assert!(!full.forbids_mutation && !full.forbids_tool_use);
        assert!(full.admits_tool("write_file", Some(false)));

        let read_only = ClosingAuthority::from_resolved(&aidaemon_core::Authority::read_only());
        assert!(read_only.forbids_mutation && !read_only.forbids_tool_use);
        assert!(!read_only.admits_tool("write_file", Some(false)));
        assert!(read_only.admits_tool("read_file", Some(true)));

        let none = ClosingAuthority::from_resolved(&aidaemon_core::Authority::no_tools());
        assert!(none.forbids_tool_use);
        assert!(!none.admits_tool("read_file", Some(true)));
    }

    fn authority<'a>(allowed: &'a [String], control_plane: &'a [String]) -> ClosingAuthority<'a> {
        ClosingAuthority {
            forbids_tool_use: false,
            forbids_mutation: false,
            allowed_tool_names: allowed,
            forbidden_tool_scopes: &[],
            owner_control_plane_tools: control_plane,
        }
    }

    /// The commit demand is made only when the exact `record_decision` call
    /// would be admitted; a tool kept only for the owner's control-plane
    /// actions does not admit it, so asking would spend the retry on a refusal.
    #[test]
    fn mandate_commit_is_demanded_only_when_record_decision_is_admitted() {
        let visible = ["manage_mandates", "http_request"];
        let read_only = |_: &str| Some(false);
        let real_rule = |tool: &str, arguments: &str| {
            tool == MANDATE_DECISION_TOOL && !arguments.contains("record_decision")
        };
        let demand = StopDemand::MandateDecisionCommit;

        let open = authority(&[], &[]);
        assert_eq!(
            stop_demand_verdict(demand, 0, &open, &visible, read_only, real_rule),
            StopDemandVerdict::Demand
        );
        assert_eq!(
            stop_demand_verdict(demand, 1, &open, &visible, read_only, real_rule),
            StopDemandVerdict::Exhausted
        );

        let allowed = vec!["http_request".to_string()];
        let control_plane = vec![MANDATE_DECISION_TOOL.to_string()];
        let narrowed = authority(&allowed, &control_plane);
        assert_eq!(
            stop_demand_verdict(demand, 0, &narrowed, &visible, read_only, real_rule),
            StopDemandVerdict::Unreachable
        );
        assert_eq!(
            stop_demand_verdict(demand, 0, &open, &["http_request"], read_only, real_rule),
            StopDemandVerdict::Unreachable,
            "an unregistered decision tool cannot be demanded"
        );
    }

    #[test]
    fn tool_free_and_gui_demands_follow_only_their_budgets_and_tool_use() {
        let open = authority(&[], &[]);
        let none = |_: &str| None;
        let never = |_: &str, _: &str| false;
        let reply = StopDemand::ReplyRepair;
        assert_eq!(
            stop_demand_verdict(reply, 0, &open, &[], none, never),
            StopDemandVerdict::Demand
        );
        assert_eq!(
            stop_demand_verdict(reply, 1, &open, &[], none, never),
            StopDemandVerdict::Exhausted
        );
        let gui = StopDemand::GuiClickVerification;
        assert_eq!(
            stop_demand_verdict(gui, 2, &open, &[], none, never),
            StopDemandVerdict::Demand
        );
        assert_eq!(
            stop_demand_verdict(gui, 3, &open, &[], none, never),
            StopDemandVerdict::Exhausted
        );
        let forbidden = ClosingAuthority {
            forbids_tool_use: true,
            ..authority(&[], &[])
        };
        assert_eq!(
            stop_demand_verdict(gui, 0, &forbidden, &[], none, never),
            StopDemandVerdict::Unreachable
        );
    }
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
            prerequisite: false,
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
            owner_control_plane_tools: &[],
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
            &["read_file", "check_remote"],
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
            owner_control_plane_tools: &[],
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
    fn disallowed_static_candidates_do_not_hide_an_admissible_dynamic_tool() {
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
        let authority = authority(true, &[]);
        let read_only = |tool: &str| Some(tool == "check_remote");
        for visible in [
            vec!["check_remote"],
            vec!["manage_mandates", "check_remote"],
            vec!["check_remote", "manage_mandates"],
        ] {
            assert!(obligation_admissible(
                &obligation,
                &authority,
                &visible,
                read_only
            ));
        }
        assert!(!obligation_admissible(
            &obligation,
            &authority,
            &["manage_mandates"],
            read_only
        ));
        assert!(!obligation_admissible(
            &obligation,
            &ClosingAuthority {
                forbids_tool_use: true,
                ..authority
            },
            &["manage_mandates", "check_remote"],
            read_only
        ));
    }

    #[test]
    fn candidate_ranking_limit_cannot_hide_the_only_authorized_tool() {
        let requirement = RequestEvidenceRequirement {
            summary: "inspect current state".to_string(),
            acceptable_scopes: vec![
                ToolSemanticScope::GoalState,
                ToolSemanticScope::ExternalRemote,
                ToolSemanticScope::LocalWorkspace,
                ToolSemanticScope::HostLocal,
                ToolSemanticScope::UserMemory,
            ],
            purpose: EvidencePurpose::CurrentState,
            minimum_authority: EvidenceAuthority::Advisory,
            temporal_scope: EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        };
        let visible = [
            "browser",
            "goal_trace",
            "http_request",
            "manage_goal_tasks",
            "manage_mandates",
            "manage_memories",
            "manage_people",
            "read_file",
            "read_node_health",
            "read_node_sensors",
            "web_search",
        ];
        let candidates = crate::agent::inquiry::candidate_tools_for_requirements(
            std::slice::from_ref(&requirement),
            visible,
        );
        assert_eq!(candidates.len(), 10);
        assert!(!candidates.iter().any(|tool| tool == "web_search"));
        let obligation = obligation(None, Some(requirement), ToolMutationEffects::NONE);
        let allowed = vec!["web_search".to_string()];
        let authority = ClosingAuthority {
            allowed_tool_names: &allowed,
            ..authority(true, &[])
        };
        assert!(obligation_admissible(
            &obligation,
            &authority,
            &visible,
            |_| Some(true)
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
