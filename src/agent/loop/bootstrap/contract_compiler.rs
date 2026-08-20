use super::task_planning::{
    planned_contract_is_confident, PlannedContractSignals, PlannedFilesystemAccess,
    PlannedTaskShape,
};
use crate::agent::CompletionTaskKind;
use crate::traits::{
    EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
    RequestReceiptPredicate, RequestResponseContract, RequestedOutcomeCondition,
    ToolCallAccessManifest, ToolMutationEffects, ToolSemanticScope, ToolTargetHint,
    ToolTargetHintKind,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// One independently compiled contract lane. Reason codes are stable telemetry
/// values; they never contain or classify user prose.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ContractLaneDecision {
    pub lane: &'static str,
    pub accepted: bool,
    pub reason_code: &'static str,
    pub candidate_count: usize,
    pub installed_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct CompiledCompletionCore {
    pub expects_mutation: bool,
    pub requires_observation: bool,
    pub task_kind: CompletionTaskKind,
    pub required_mutation_effects: ToolMutationEffects,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CompiledEvidencePolicy {
    pub minimum_sources: usize,
    pub requires_primary_sources: bool,
    pub requires_exact_history: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct CompiledAuthority {
    pub mutation_scope: &'static str,
    pub forbidden_actions: Vec<crate::agent::ForbiddenMutationAction>,
    pub forbids_tool_use: bool,
    pub allowed_tool_names: Vec<String>,
    pub forbidden_tool_scopes: Vec<ToolSemanticScope>,
}

impl Default for CompiledAuthority {
    fn default() -> Self {
        Self {
            mutation_scope: "allowed",
            forbidden_actions: Vec::new(),
            forbids_tool_use: false,
            allowed_tool_names: Vec::new(),
            forbidden_tool_scopes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CompiledTaskContract {
    pub core: Option<CompiledCompletionCore>,
    pub authority: CompiledAuthority,
    /// Authority is itself a product: an invalid mutation lane must not erase
    /// a valid tool restriction (or vice versa). Invalid lanes retain the
    /// no-op defaults in `authority`, so applying either valid sibling can
    /// only attenuate the pre-existing runtime contract.
    pub mutation_authority_valid: bool,
    pub tool_authority_valid: bool,
    pub evidence_policy: CompiledEvidencePolicy,
    pub evidence_requirements: Vec<RequestEvidenceRequirement>,
    pub required_invocations: Vec<RequestReceiptPredicate>,
    pub response_contract: Option<Box<RequestResponseContract>>,
    pub filesystem_access: Option<ToolCallAccessManifest>,
    pub project_scope: Option<String>,
    pub decisions: Vec<ContractLaneDecision>,
}

pub(crate) struct ContractCompilerInput<'a> {
    pub signals: &'a PlannedContractSignals,
    pub task_shape: Option<&'a PlannedTaskShape>,
    pub available_tool_names: &'a [String],
    pub available_tool_receipt_kinds: &'a [(String, crate::traits::ToolReceiptKind)],
    pub structural_filesystem_resources: &'a [String],
    pub structural_project_scopes: &'a [String],
    pub project_alias_roots: &'a [String],
    /// Current user turn used only to bind an assessor-produced response
    /// contract to its request identity. Rust never classifies its wording.
    pub current_user_text: &'a str,
}

fn decision(
    lane: &'static str,
    accepted: bool,
    reason_code: &'static str,
    candidate_count: usize,
    installed_count: usize,
) -> ContractLaneDecision {
    ContractLaneDecision {
        lane,
        accepted,
        reason_code,
        candidate_count,
        installed_count,
    }
}

fn compile_core(signals: &PlannedContractSignals) -> Result<CompiledCompletionCore, &'static str> {
    let expects_mutation = signals.expects_mutation.ok_or("missing_expects_mutation")?;
    let requires_observation = signals
        .requires_observation
        .ok_or("missing_requires_observation")?;
    let task_kind = signals
        .task_kind
        .as_deref()
        .and_then(crate::agent::parse_planned_task_kind)
        .ok_or("invalid_task_kind")?;
    let mutation_capable = matches!(
        task_kind,
        CompletionTaskKind::Change
            | CompletionTaskKind::Deliver
            | CompletionTaskKind::Schedule
            | CompletionTaskKind::Monitor
    );
    if expects_mutation != mutation_capable {
        return Err("task_kind_mutation_mismatch");
    }

    let effect_names = signals
        .required_effects
        .as_deref()
        .ok_or("missing_required_effects")?;
    let required_mutation_effects = if expects_mutation {
        let effects = crate::agent::parse_planned_mutation_effects(effect_names)
            .ok_or("invalid_required_effect")?;
        if effects.is_empty() {
            return Err("missing_mutation_effect");
        }
        // Destructive is an operation/risk qualifier, not a postcondition.
        // Requiring it as an achieved effect creates an obligation no generic
        // adapter can prove independently of the actual domain mutation.
        if effects.contains(ToolMutationEffects::DESTRUCTIVE) {
            return Err("risk_qualifier_is_not_completion_effect");
        }
        if effects.contains(ToolMutationEffects::REMOTE_MUTATION)
            && effects.intersects(
                ToolMutationEffects::REMOTE_DEPLOY.union(ToolMutationEffects::EXTERNAL_DELIVERY),
            )
        {
            return Err("overlapping_remote_effects");
        }
        if task_kind == CompletionTaskKind::Deliver
            && !effects.contains(ToolMutationEffects::EXTERNAL_DELIVERY)
        {
            return Err("delivery_effect_missing");
        }
        effects
    } else {
        if !effect_names.is_empty() {
            return Err("observational_task_has_mutation_effect");
        }
        ToolMutationEffects::NONE
    };

    Ok(CompiledCompletionCore {
        expects_mutation,
        requires_observation,
        task_kind,
        required_mutation_effects,
    })
}

fn compile_response_contract(
    signals: &PlannedContractSignals,
    current_user_text: &str,
) -> (Option<Box<RequestResponseContract>>, ContractLaneDecision) {
    const MAX_EXACT_RESPONSE_BYTES: usize = 4096;
    let Some(candidate) = signals.response_contract.as_ref() else {
        return (
            None,
            decision("response_presentation", true, "not_requested", 0, 0),
        );
    };
    if candidate.mode != "exact_text" {
        return (
            None,
            decision(
                "response_presentation",
                false,
                "unsupported_presentation_mode",
                1,
                0,
            ),
        );
    }
    let Some(success_text) = candidate
        .success_text
        .as_deref()
        .filter(|text| !text.is_empty())
    else {
        return (
            None,
            decision("response_presentation", false, "exact_text_missing", 1, 0),
        );
    };
    if success_text.len() > MAX_EXACT_RESPONSE_BYTES {
        return (
            None,
            decision("response_presentation", false, "exact_text_too_large", 1, 0),
        );
    }
    if success_text
        .chars()
        .any(|ch| ch.is_control() && !matches!(ch, '\n' | '\r' | '\t'))
    {
        return (
            None,
            decision(
                "response_presentation",
                false,
                "exact_text_contains_control",
                1,
                0,
            ),
        );
    }
    // The semantic producer decides whether the request asks for exact text.
    // This deterministic boundary validates only the typed value's shape and
    // binds it to the current request; it deliberately does not re-interpret
    // natural language with keywords, phrase lists, or substring rules.
    let source_message_hash = format!("{:x}", Sha256::digest(current_user_text.as_bytes()));
    (
        Some(Box::new(RequestResponseContract::ExactText {
            success_text: success_text.to_string(),
            source_message_hash,
        })),
        decision("response_presentation", true, "accepted", 1, 1),
    )
}

fn compile_evidence_policy(
    signals: &PlannedContractSignals,
    evidence_requirements: &[RequestEvidenceRequirement],
) -> (CompiledEvidencePolicy, Vec<ContractLaneDecision>) {
    let mut policy = CompiledEvidencePolicy::default();
    let mut decisions = Vec::new();
    let minimum_sources = signals.minimum_sources.map(usize::from);
    let requires_primary_sources = signals.requires_primary_sources;
    let has_remote_subject_evidence = evidence_requirements.iter().any(|requirement| {
        requirement
            .acceptable_scopes
            .contains(&ToolSemanticScope::ExternalRemote)
            && matches!(
                requirement.purpose,
                EvidencePurpose::CurrentState
                    | EvidencePurpose::HistoricalRecord
                    | EvidencePurpose::Content
                    | EvidencePurpose::Attribution
                    | EvidencePurpose::CausalExplanation
            )
    });
    match (minimum_sources, requires_primary_sources) {
        (Some(0), Some(false)) => {
            decisions.push(decision("research_evidence_policy", true, "empty", 0, 0))
        }
        (Some(minimum), Some(primary))
            if (1..=20).contains(&minimum) && has_remote_subject_evidence =>
        {
            policy.minimum_sources = minimum;
            policy.requires_primary_sources = primary;
            decisions.push(decision("research_evidence_policy", true, "accepted", 1, 1));
        }
        (Some(_), Some(_)) => decisions.push(decision(
            "research_evidence_policy",
            false,
            "missing_remote_subject_evidence",
            1,
            0,
        )),
        _ => decisions.push(decision(
            "research_evidence_policy",
            false,
            "missing_typed_policy",
            usize::from(minimum_sources.is_some() || requires_primary_sources.is_some()),
            0,
        )),
    }

    let requested_exact_history = signals.requires_exact_history.unwrap_or(false);
    let has_exact_history_evidence = evidence_requirements.iter().any(|requirement| {
        requirement
            .acceptable_scopes
            .contains(&ToolSemanticScope::ConversationHistory)
            && requirement.purpose == EvidencePurpose::HistoricalRecord
            && requirement.minimum_authority == EvidenceAuthority::Canonical
    });
    if requested_exact_history && has_exact_history_evidence {
        policy.requires_exact_history = true;
        decisions.push(decision("exact_history_policy", true, "accepted", 1, 1));
    } else if requested_exact_history {
        decisions.push(decision(
            "exact_history_policy",
            false,
            "missing_canonical_history_obligation",
            1,
            0,
        ));
    } else {
        decisions.push(decision("exact_history_policy", true, "empty", 0, 0));
    }
    (policy, decisions)
}

fn valid_receipt(
    receipt: &RequestReceiptPredicate,
    available_tool_names: &[String],
    available_tool_receipt_kinds: &[(String, crate::traits::ToolReceiptKind)],
) -> bool {
    let process_fields_requested = !receipt.exit_codes.is_empty()
        || receipt
            .outcome_statuses
            .contains(&crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult);
    let tool_is_process = receipt.tool_names.first().and_then(|name| {
        available_tool_receipt_kinds
            .iter()
            .find(|(tool, _)| tool == name)
            .map(|(_, kind)| *kind == crate::traits::ToolReceiptKind::Process)
    });
    receipt.tool_names.len() == 1
        && receipt.tool_names.iter().all(|name| {
            !name.is_empty()
                && name.len() <= 80
                && name.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'_' | b'-')
                })
                && available_tool_names.contains(name)
        })
        // Process-only fields must not make a generic/API receipt obligation
        // impossible to satisfy. This uses the registered adapter protocol,
        // never tool names as a language classifier.
        && (!process_fields_requested || tool_is_process == Some(true))
        && receipt.exit_codes.len() <= 16
        && receipt.outcome_statuses.len() <= 6
        && receipt
            .min_invocations
            .is_none_or(|limit| (1..=16).contains(&limit))
        && receipt
            .max_invocations
            .is_none_or(|limit| (1..=16).contains(&limit))
        && receipt
            .min_invocations
            .zip(receipt.max_invocations)
            .is_none_or(|(minimum, maximum)| minimum <= maximum)
}

/// Normalize dependent receipt fields against the runtime's typed process
/// protocol. Exit code and outcome are not independent facts: a normal process
/// exit of zero is `succeeded`, a positive exit is a completed negative
/// observation, and the backend's negative no-status sentinel is an execution
/// failure. Leaving a model-produced incompatible conjunction in the contract
/// creates an obligation no real receipt can satisfy.
fn normalize_receipt_predicate(mut receipt: RequestReceiptPredicate) -> RequestReceiptPredicate {
    if let Some(condition) = receipt.outcome_condition {
        // The semantic producer owns only the request-level condition. Clear
        // any concurrently emitted protocol fields so they cannot turn an OR
        // policy such as `non_success_terminal` into an impossible conjunction.
        receipt.outcome_statuses.clear();
        receipt.contract_rejected = None;
        if condition == RequestedOutcomeCondition::ContractRejected {
            receipt.exit_codes.clear();
        }
    } else if receipt.contract_rejected == Some(true) {
        // Rejection is an independent typed disposition that may occur at an
        // adapter-validation or dispatcher-policy boundary. It causally rules
        // out a process exit code, but it does not imply one universal domain
        // outcome across receipt schema versions or adapter classes.
        receipt.exit_codes.clear();
        receipt.outcome_statuses.clear();
    } else if !receipt.exit_codes.is_empty() {
        receipt.outcome_statuses = receipt
            .exit_codes
            .iter()
            .copied()
            .map(crate::traits::ToolOutcomeStatus::from_process_exit_code)
            .collect();
        receipt
            .outcome_statuses
            .sort_by_key(|status| status.as_str());
        receipt.outcome_statuses.dedup();
    }
    receipt
}

fn normalize_receipt_for_tool_protocol(
    mut receipt: RequestReceiptPredicate,
    available_tool_receipt_kinds: &[(String, crate::traits::ToolReceiptKind)],
) -> RequestReceiptPredicate {
    let Some(tool_name) = receipt.tool_names.first() else {
        return receipt;
    };
    let Some((_, kind)) = available_tool_receipt_kinds
        .iter()
        .find(|(name, _)| name == tool_name)
    else {
        return receipt;
    };
    if *kind != crate::traits::ToolReceiptKind::Process {
        // Exit codes are not part of a generic/HTTP receipt. Preserve the
        // named invocation and all status values that the adapter can emit,
        // but remove only the impossible process-specific fields so one valid
        // management read is not turned into a permanently open obligation.
        receipt.exit_codes.clear();
        receipt.outcome_statuses.retain(|status| {
            *status != crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult
        });
        if receipt.outcome_condition == Some(RequestedOutcomeCondition::CompletedWithNegativeResult)
        {
            // A "completed negative" is specifically a normal process exit.
            // Generic and HTTP adapters expose only terminal success,
            // failure, blocking, or contract rejection. Preserve the user's
            // negative-result intent in the adapter's actual outcome algebra
            // instead of installing a predicate no receipt can satisfy.
            receipt.outcome_condition = Some(RequestedOutcomeCondition::NonSuccessTerminal);
            receipt.requires_output = false;
        } else if matches!(
            receipt.outcome_condition,
            Some(RequestedOutcomeCondition::ContractRejected)
                | Some(RequestedOutcomeCondition::Blocked)
        ) {
            // These dispositions can occur before the adapter produces result
            // bytes. Their typed receipt is authoritative evidence by itself.
            receipt.requires_output = false;
        }
    }
    receipt
}

fn canonical_invocation_requirement(
    receipt: RequestReceiptPredicate,
) -> RequestEvidenceRequirement {
    RequestEvidenceRequirement {
        summary: "Complete the exact requested machine invocation".to_string(),
        acceptable_scopes: Vec::new(),
        purpose: EvidencePurpose::Outcome,
        minimum_authority: EvidenceAuthority::Direct,
        // An invocation outcome remains true after later operations. Current
        // scope is reserved for observations whose subject state can change.
        temporal_scope: EvidenceTemporalScope::Historical,
        required_content_markers: Vec::new(),
        receipt: Some(receipt),
        target: None,
    }
}

fn compile_obligations(
    signals: &PlannedContractSignals,
    available_tool_names: &[String],
    available_tool_receipt_kinds: &[(String, crate::traits::ToolReceiptKind)],
) -> (
    Vec<RequestEvidenceRequirement>,
    Vec<RequestReceiptPredicate>,
    ContractLaneDecision,
) {
    let evidence_candidates = signals.evidence_requirements.as_deref().unwrap_or_default();
    let invocation_candidates = signals.required_invocations.as_deref().unwrap_or_default();
    let candidate_count = evidence_candidates.len() + invocation_candidates.len();
    let mut evidence = Vec::new();
    let mut invocations = Vec::new();

    // A receipt predicate is the only durable identity available for an
    // invocation obligation.  If a producer emits several obligations with
    // the same predicate but different cardinality fields, those obligations
    // are observationally indistinguishable: no later receipt can prove which
    // one it belonged to.  Keeping them as separate proof nodes makes the
    // runtime derive a shared retry gate from the smallest limit, so an
    // exact-three obligation paired with two redundant exact-one obligations
    // is incorrectly closed after the first call.  Coalesce the equivalent
    // predicates here and retain the strongest required cardinality.  Truly
    // distinct obligations must carry a distinct typed predicate (tool,
    // outcome, target, or another protocol field); they are not merged.
    let coalesce_invocation = |invocations: &mut Vec<RequestReceiptPredicate>,
                               receipt: RequestReceiptPredicate| {
        let normalized = receipt;
        if let Some(existing) = invocations.iter_mut().find(|existing| {
            let mut existing_identity = (*existing).clone();
            let mut candidate_identity = normalized.clone();
            existing_identity.min_invocations = None;
            existing_identity.max_invocations = None;
            candidate_identity.min_invocations = None;
            candidate_identity.max_invocations = None;
            existing_identity == candidate_identity
        }) {
            let minimum = existing
                .min_invocations
                .into_iter()
                .chain(normalized.min_invocations)
                .max();
            let maximum = match (existing.max_invocations, normalized.max_invocations) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (None, _) | (_, None) => None,
            };
            // A malformed overlapping pair such as min=3/max=3 together
            // with min=1/max=1 must not become an impossible min=3/max=1
            // predicate. The largest explicit minimum is the only
            // distinguishable requirement, so lift the maximum to it.
            existing.min_invocations = minimum;
            existing.max_invocations = match (maximum, minimum) {
                (Some(maximum), Some(minimum)) => Some(maximum.max(minimum)),
                (maximum, _) => maximum,
            };
            true
        } else {
            // Keep the vector's ownership model simple: only append after
            // checking the canonical identity above.
            invocations.push(normalized);
            false
        }
    };

    let mut add_invocation = |receipt: &RequestReceiptPredicate| {
        let receipt = normalize_receipt_for_tool_protocol(
            normalize_receipt_predicate(receipt.clone()),
            available_tool_receipt_kinds,
        );
        if valid_receipt(&receipt, available_tool_names, available_tool_receipt_kinds) {
            coalesce_invocation(&mut invocations, receipt);
        }
    };
    for receipt in invocation_candidates {
        add_invocation(receipt);
    }

    for candidate in evidence_candidates {
        // A receipt-bound need is one machine-observation obligation. Its
        // receipt predicate, not a second scope/authority tuple, is its stable
        // proof identity. Canonicalizing here prevents the same requested call
        // from becoming two incompatible proof nodes.
        if let Some(receipt) = candidate.receipt.as_ref() {
            // Receipt predicates prove execution outcomes. Do not let a
            // malformed subject-matter evidence item (for example, a memory
            // fact) silently weaken into invocation-only proof merely because
            // it also carries a receipt-shaped object.
            if candidate.purpose == EvidencePurpose::Outcome {
                add_invocation(receipt);
            }
            continue;
        }

        let summary_len = candidate.summary.trim().chars().count();
        if summary_len == 0
            || summary_len > 240
            || candidate.acceptable_scopes.is_empty()
            || candidate.acceptable_scopes.len() > 3
            || candidate.target.is_some()
        {
            continue;
        }
        let mut normalized = candidate.clone();
        if !crate::agent::inquiry::requirement_has_builtin_evidence_route(&normalized) {
            continue;
        }
        normalized.required_content_markers.clear();
        normalized
            .acceptable_scopes
            .sort_by_key(|scope| scope.as_str());
        normalized.acceptable_scopes.dedup();
        let duplicate = evidence
            .iter()
            .any(|existing: &RequestEvidenceRequirement| {
                existing.acceptable_scopes == normalized.acceptable_scopes
                    && existing.purpose == normalized.purpose
                    && existing.minimum_authority == normalized.minimum_authority
                    && existing.temporal_scope == normalized.temporal_scope
                    && existing.target == normalized.target
            });
        if !duplicate {
            evidence.push(normalized);
        }
    }

    // Keep the invocation lane separate in the compiled value. Installation
    // creates its canonical proof nodes once, after every producer has passed
    // through this same compiler.
    let installed_count = evidence.len() + invocations.len();
    let accepted = candidate_count == 0 || installed_count > 0;
    let reason = if candidate_count == 0 {
        "empty"
    } else if installed_count == candidate_count {
        "accepted"
    } else if installed_count > 0 {
        "partially_accepted"
    } else {
        "no_valid_typed_obligation"
    };
    (
        evidence,
        invocations,
        decision(
            "obligations",
            accepted,
            reason,
            candidate_count,
            installed_count,
        ),
    )
}

fn compile_authority(
    signals: &PlannedContractSignals,
    available_tool_names: &[String],
) -> (CompiledAuthority, bool, bool, Vec<ContractLaneDecision>) {
    let mut authority = CompiledAuthority::default();
    let mut decisions = Vec::new();
    let mut mutation_authority_valid = false;
    let mut tool_authority_valid = false;

    let forbidden_actions = signals
        .forbidden_actions
        .iter()
        .filter_map(|action| crate::agent::parse_planned_forbidden_action(action))
        .collect::<Vec<_>>();
    let mutation_scope = signals
        .mutation_scope
        .as_deref()
        .map(|scope| scope.trim().to_ascii_lowercase());
    match mutation_scope.as_deref() {
        Some("allowed") if forbidden_actions.is_empty() => {
            mutation_authority_valid = true;
            decisions.push(decision("mutation_authority", true, "accepted", 1, 1));
        }
        Some("read_only" | "read-only") => {
            mutation_authority_valid = true;
            authority.mutation_scope = "read_only";
            decisions.push(decision("mutation_authority", true, "accepted", 1, 1));
        }
        Some("scoped") if !forbidden_actions.is_empty() => {
            mutation_authority_valid = true;
            authority.mutation_scope = "scoped";
            authority.forbidden_actions = forbidden_actions;
            decisions.push(decision("mutation_authority", true, "accepted", 1, 1));
        }
        _ => decisions.push(decision(
            "mutation_authority",
            false,
            "invalid_typed_policy",
            usize::from(mutation_scope.is_some()),
            0,
        )),
    }

    authority.forbidden_tool_scopes = signals.forbidden_tool_scopes.clone();
    authority
        .forbidden_tool_scopes
        .sort_by_key(|scope| scope.as_str());
    authority.forbidden_tool_scopes.dedup();
    let tool_scope = signals
        .tool_scope
        .as_deref()
        .map(|scope| scope.trim().to_ascii_lowercase());
    match tool_scope.as_deref() {
        Some("allowed") if signals.allowed_tool_names.is_empty() => {
            tool_authority_valid = true;
            decisions.push(decision(
                "tool_authority",
                true,
                "accepted",
                1 + signals.forbidden_tool_scopes.len(),
                1 + authority.forbidden_tool_scopes.len(),
            ));
        }
        Some("forbidden") if signals.allowed_tool_names.is_empty() => {
            tool_authority_valid = true;
            authority.forbids_tool_use = true;
            decisions.push(decision("tool_authority", true, "accepted", 1, 1));
        }
        Some("restricted") => {
            let mut names = signals
                .allowed_tool_names
                .iter()
                .filter(|name| available_tool_names.contains(name))
                .cloned()
                .collect::<Vec<_>>();
            names.sort();
            names.dedup();
            if names.is_empty() || names.len() > 8 {
                decisions.push(decision(
                    "tool_authority",
                    false,
                    "no_registered_allowed_capability",
                    signals.allowed_tool_names.len(),
                    0,
                ));
            } else {
                tool_authority_valid = true;
                authority.allowed_tool_names = names;
                decisions.push(decision(
                    "tool_authority",
                    true,
                    if authority.allowed_tool_names.len() == signals.allowed_tool_names.len() {
                        "accepted"
                    } else {
                        "partially_accepted"
                    },
                    signals.allowed_tool_names.len(),
                    authority.allowed_tool_names.len(),
                ));
            }
        }
        _ => decisions.push(decision(
            "tool_authority",
            false,
            "invalid_typed_policy",
            usize::from(tool_scope.is_some()),
            0,
        )),
    }
    if !tool_authority_valid {
        authority.forbidden_tool_scopes.clear();
    }
    (
        authority,
        mutation_authority_valid,
        tool_authority_valid,
        decisions,
    )
}

fn resolve_structural_path(
    raw: &str,
    structural_resources: &[String],
    alias_roots: &[String],
) -> Option<String> {
    crate::tools::fs_utils::resolve_structural_filesystem_reference(raw, alias_roots)
        .map(|path| path.to_string_lossy().to_string())
        .filter(|path| structural_resources.contains(path))
}

fn compile_filesystem_access(
    candidate: Option<&PlannedFilesystemAccess>,
    structural_resources: &[String],
    alias_roots: &[String],
    project_reference: Option<&str>,
    required_mutation_effects: ToolMutationEffects,
) -> (Option<ToolCallAccessManifest>, ContractLaneDecision) {
    let candidate = candidate.cloned().unwrap_or_default();
    let resolved_project_scope = project_reference.and_then(|reference| {
        resolve_structural_path(reference, structural_resources, alias_roots)
    });
    let candidate_count = usize::from(candidate.execution_cwd.is_some())
        + candidate.read_paths.len()
        + candidate.write_paths.len()
        + candidate.read_roots.len()
        + candidate.write_roots.len();
    let execution_cwd = candidate
        .execution_cwd
        .as_deref()
        .and_then(|path| resolve_structural_path(path, structural_resources, alias_roots));
    let resolved_read_roots = candidate
        .read_roots
        .iter()
        .filter_map(|path| resolve_structural_path(path, structural_resources, alias_roots))
        .collect::<Vec<_>>();
    let resolved_write_roots = candidate
        .write_roots
        .iter()
        .filter_map(|path| resolve_structural_path(path, structural_resources, alias_roots))
        .collect::<Vec<_>>();

    // A contract may name both a disposable directory and one of its future
    // children.  The child is still an exact resource, but the explicitly
    // named ancestor is also a directory capability: otherwise a later
    // `write_file` or terminal invocation cannot create the child before it
    // exists.  Derive this only from the structural resource graph supplied by
    // the current request and only for an access role that the assessor marked
    // read/write.  We never infer a root from shell text, a filename suffix,
    // or an execution cwd (cwd is traversal context, not data authority).
    let resolved_structural_resources = structural_resources
        .iter()
        .filter_map(|path| resolve_structural_path(path, structural_resources, alias_roots))
        .collect::<Vec<_>>();
    let structural_ancestors_for = |path: &str| {
        let candidate_path = std::path::Path::new(path);
        resolved_structural_resources
            .iter()
            .filter(|ancestor| {
                ancestor.as_str() != path
                    && execution_cwd.as_deref() != Some(ancestor.as_str())
                    && candidate_path.starts_with(std::path::Path::new(ancestor.as_str()))
            })
            .max_by_key(|ancestor| std::path::Path::new(ancestor.as_str()).components().count())
            .cloned()
    };
    let is_directory_capability = |path: &str, explicit_roots: &[String]| {
        explicit_roots.iter().any(|root| root == path)
            || execution_cwd.as_deref() == Some(path)
            || resolved_project_scope.as_deref() == Some(path)
            || std::path::Path::new(path).is_dir()
    };
    let mut read_targets = candidate
        .read_paths
        .iter()
        .filter_map(|path| resolve_structural_path(path, structural_resources, alias_roots))
        .filter_map(|path| {
            let kind = if is_directory_capability(&path, &resolved_read_roots) {
                ToolTargetHintKind::ProjectScope
            } else {
                ToolTargetHintKind::Path
            };
            ToolTargetHint::new(kind, path)
        })
        .collect::<Vec<_>>();
    let mut write_targets = candidate
        .write_paths
        .iter()
        .filter_map(|path| resolve_structural_path(path, structural_resources, alias_roots))
        .filter_map(|path| {
            let kind = if is_directory_capability(&path, &resolved_write_roots) {
                ToolTargetHintKind::ProjectScope
            } else {
                ToolTargetHintKind::Path
            };
            ToolTargetHint::new(kind, path)
        })
        .collect::<Vec<_>>();
    // A project identity is deliberately not a write-capability wildcard. If
    // the semantic producer supplied an exact child inside an identified
    // project, retain that exact grant and require an explicit `write_roots`
    // capability for broader descendant creation. Ancestor derivation is only
    // useful for an untyped disposable output graph (where the request names a
    // future root and child but has no project identity); keeping it out of
    // identified projects prevents a source-file request from silently
    // becoming a repository-wide write grant.
    let derived_read_roots = resolved_project_scope.is_none().then(|| {
        read_targets
            .iter()
            .filter_map(|target| structural_ancestors_for(&target.value))
            .filter(|ancestor| !resolved_read_roots.iter().any(|root| root == ancestor))
            .collect::<Vec<_>>()
    });
    let derived_write_roots = resolved_project_scope.is_none().then(|| {
        write_targets
            .iter()
            .filter_map(|target| structural_ancestors_for(&target.value))
            .filter(|ancestor| !resolved_write_roots.iter().any(|root| root == ancestor))
            .collect::<Vec<_>>()
    });
    read_targets.extend(
        derived_read_roots
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, path)),
    );
    write_targets.extend(
        derived_write_roots
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, path)),
    );
    read_targets.extend(
        resolved_read_roots
            .iter()
            .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, path)),
    );
    write_targets.extend(
        resolved_write_roots
            .iter()
            .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, path)),
    );
    // A semantic project reference is a typed scope identity, not a prose
    // hint. If a confident local-mutation contract omitted a write role, the
    // reference is the only safe capability source we can use to keep the
    // request autonomous. Add it as a directory grant only when no write
    // target was supplied at all; an explicit exact child remains exact.
    let local_mutation = required_mutation_effects.intersects(
        ToolMutationEffects::LOCAL_SOURCE_WRITE
            .union(ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
            .union(ToolMutationEffects::LOCAL_DERIVED_WRITE)
            .union(ToolMutationEffects::REPOSITORY_WRITE)
            .union(ToolMutationEffects::CONFIGURATION)
            .union(ToolMutationEffects::DESTRUCTIVE),
    );
    let project_scope_fallback_installed =
        local_mutation && write_targets.is_empty() && resolved_project_scope.is_some();
    if project_scope_fallback_installed {
        if let Some(path) = resolved_project_scope.as_deref() {
            if let Some(target) = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, path) {
                write_targets.push(target);
            }
        }
    }
    read_targets.sort_by(|left, right| left.value.cmp(&right.value));
    read_targets.dedup();
    write_targets.sort_by(|left, right| left.value.cmp(&right.value));
    write_targets.dedup();
    let installed_count =
        usize::from(execution_cwd.is_some()) + read_targets.len() + write_targets.len();
    let manifest = (installed_count > 0).then_some(ToolCallAccessManifest {
        execution_cwd,
        read_targets,
        write_targets,
        adapter_read_targets: Vec::new(),
    });
    (
        manifest,
        decision(
            "filesystem_authority",
            candidate_count == 0 || installed_count > 0,
            if project_scope_fallback_installed {
                "project_scope_fallback"
            } else if candidate_count == installed_count {
                "accepted"
            } else if installed_count > 0 {
                "partially_accepted"
            } else if candidate_count == 0 {
                "empty"
            } else {
                "no_structural_resource_identity"
            },
            candidate_count,
            installed_count,
        ),
    )
}

fn compile_project_scope(
    project_reference: Option<&str>,
    structural_project_scopes: &[String],
    alias_roots: &[String],
) -> (Option<String>, ContractLaneDecision) {
    let Some(reference) = project_reference
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return (None, decision("project_scope", true, "empty", 0, 0));
    };
    let resolved = crate::tools::fs_utils::resolve_project_scope_reference(reference, alias_roots)
        .map(|path| path.to_string_lossy().to_string())
        .filter(|path| structural_project_scopes.contains(path));
    let accepted = resolved.is_some();
    (
        resolved,
        decision(
            "project_scope",
            accepted,
            if accepted {
                "accepted"
            } else {
                "no_structural_resource_identity"
            },
            1,
            usize::from(accepted),
        ),
    )
}

/// Compile independent typed products from one semantic candidate. No user
/// prose is classified here: registry membership and canonical structural
/// resource identities are the only grounding inputs.
pub(crate) fn compile_task_contract(input: ContractCompilerInput<'_>) -> CompiledTaskContract {
    let mut compiled = CompiledTaskContract::default();
    if !planned_contract_is_confident(input.signals, input.task_shape) {
        compiled.decisions.push(decision(
            "candidate",
            false,
            "insufficient_semantic_confidence",
            1,
            0,
        ));
        return compiled;
    }
    compiled
        .decisions
        .push(decision("candidate", true, "accepted", 1, 1));

    match compile_core(input.signals) {
        Ok(core) => {
            compiled.core = Some(core);
            compiled
                .decisions
                .push(decision("completion_core", true, "accepted", 1, 1));
        }
        Err(reason) => compiled
            .decisions
            .push(decision("completion_core", false, reason, 1, 0)),
    }

    let (response_contract, response_decision) =
        compile_response_contract(input.signals, input.current_user_text);
    compiled.response_contract = response_contract;
    compiled.decisions.push(response_decision);

    let (evidence, invocations, obligation_decision) = compile_obligations(
        input.signals,
        input.available_tool_names,
        input.available_tool_receipt_kinds,
    );
    compiled.evidence_requirements = evidence;
    compiled.required_invocations = invocations;
    compiled.decisions.push(obligation_decision);

    let (evidence_policy, evidence_policy_decisions) =
        compile_evidence_policy(input.signals, &compiled.evidence_requirements);
    compiled.evidence_policy = evidence_policy;
    compiled.decisions.extend(evidence_policy_decisions);

    let (authority, mutation_authority_valid, tool_authority_valid, authority_decisions) =
        compile_authority(input.signals, input.available_tool_names);
    compiled.authority = authority;
    compiled.mutation_authority_valid = mutation_authority_valid;
    compiled.tool_authority_valid = tool_authority_valid;
    compiled.decisions.extend(authority_decisions);

    if compiled.core.as_ref().is_some_and(|core| {
        core.expects_mutation
            && ((compiled.mutation_authority_valid
                && compiled.authority.mutation_scope == "read_only")
                || (compiled.tool_authority_valid && compiled.authority.forbids_tool_use))
    }) {
        let reason = if compiled.authority.forbids_tool_use {
            "mutation_lifecycle_conflicts_with_tool_prohibition"
        } else {
            "mutation_lifecycle_conflicts_with_read_only_authority"
        };
        compiled.core = None;
        compiled
            .decisions
            .push(decision("composition", false, reason, 2, 1));
    } else {
        compiled
            .decisions
            .push(decision("composition", true, "accepted", 2, 2));
    }

    if compiled.core.as_ref().is_some_and(|core| {
        core.requires_observation
            && compiled.evidence_requirements.is_empty()
            && compiled.required_invocations.is_empty()
    }) {
        compiled.core = None;
        compiled.decisions.push(decision(
            "composition",
            false,
            "observation_lifecycle_missing_typed_obligation",
            1,
            0,
        ));
    }

    compiled.decisions.push(decision(
        "legacy_response_markers",
        true,
        "non_authoritative_ignored",
        input.signals.required_response_fields.len(),
        0,
    ));

    let (filesystem_access, filesystem_decision) = compile_filesystem_access(
        input.signals.filesystem_access.as_ref(),
        input.structural_filesystem_resources,
        input.project_alias_roots,
        input.signals.project_reference.as_deref(),
        compiled
            .core
            .as_ref()
            .map(|core| core.required_mutation_effects)
            .unwrap_or(ToolMutationEffects::NONE),
    );
    compiled.filesystem_access = filesystem_access;
    compiled.decisions.push(filesystem_decision);

    let (project_scope, project_decision) = compile_project_scope(
        input.signals.project_reference.as_deref(),
        input.structural_project_scopes,
        input.project_alias_roots,
    );
    compiled.project_scope = project_scope;
    compiled.decisions.push(project_decision);

    if let Some(core) = compiled.core.as_mut() {
        core.requires_observation |=
            !compiled.evidence_requirements.is_empty() || !compiled.required_invocations.is_empty();
        if core.requires_observation && core.task_kind == CompletionTaskKind::Conversational {
            core.task_kind = CompletionTaskKind::Check;
        }
    }

    // Ensure the helper remains the sole canonical constructor for receipt
    // obligations; this assertion also makes accidental divergence visible in
    // focused tests without affecting runtime state.
    debug_assert!(compiled
        .required_invocations
        .iter()
        .cloned()
        .map(canonical_invocation_requirement)
        .all(|requirement| crate::agent::inquiry::requirement_is_exact_invocation(&requirement)));
    compiled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_signals() -> PlannedContractSignals {
        PlannedContractSignals {
            confidence: Some("high".to_string()),
            expects_mutation: Some(false),
            requires_observation: Some(true),
            required_effects: Some(Vec::new()),
            task_kind: Some("check".to_string()),
            mutation_scope: Some("allowed".to_string()),
            forbidden_actions: Vec::new(),
            constraint_evidence: Vec::new(),
            tool_scope: Some("allowed".to_string()),
            allowed_tool_names: Vec::new(),
            forbidden_tool_scopes: Vec::new(),
            tool_constraint_evidence: Vec::new(),
            required_response_fields: Vec::new(),
            response_contract: None,
            minimum_sources: Some(0),
            requires_primary_sources: Some(false),
            requires_exact_history: Some(false),
            evidence_requirements: Some(Vec::new()),
            required_invocations: Some(Vec::new()),
            filesystem_access: Some(PlannedFilesystemAccess::default()),
            project_reference: None,
        }
    }

    fn compile(signals: &PlannedContractSignals) -> CompiledTaskContract {
        let receipt_kinds = vec![
            (
                "manage_mandates".to_string(),
                crate::traits::ToolReceiptKind::Generic,
            ),
            (
                "terminal".to_string(),
                crate::traits::ToolReceiptKind::Process,
            ),
        ];
        compile_task_contract(ContractCompilerInput {
            signals,
            task_shape: None,
            available_tool_names: &["manage_mandates".to_string(), "terminal".to_string()],
            available_tool_receipt_kinds: &receipt_kinds,
            structural_filesystem_resources: &[],
            structural_project_scopes: &[],
            project_alias_roots: &[],
            current_user_text: "synthetic request",
        })
    }

    #[test]
    fn malformed_authority_does_not_erase_valid_obligation() {
        let mut signals = base_signals();
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["manage_mandates".to_string()],
            requires_output: true,
            ..RequestReceiptPredicate::default()
        }]);
        signals.tool_scope = Some("restricted".to_string());
        signals.allowed_tool_names = vec!["unregistered_capability".to_string()];
        signals.forbidden_tool_scopes = vec![ToolSemanticScope::ExternalRemote];

        let compiled = compile(&signals);
        assert!(compiled.core.is_some());
        assert_eq!(compiled.required_invocations.len(), 1);
        assert!(compiled.mutation_authority_valid);
        assert!(!compiled.tool_authority_valid);
        assert!(compiled.authority.allowed_tool_names.is_empty());
        assert!(compiled.authority.forbidden_tool_scopes.is_empty());
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "tool_authority"
                && !decision.accepted
                && decision.reason_code == "no_registered_allowed_capability"
        }));
    }

    #[test]
    fn valid_tool_restriction_survives_an_invalid_mutation_sibling() {
        let mut signals = base_signals();
        signals.mutation_scope = Some("invalid".to_string());
        signals.tool_scope = Some("restricted".to_string());
        signals.allowed_tool_names = vec!["manage_mandates".to_string()];
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["manage_mandates".to_string()],
            ..RequestReceiptPredicate::default()
        }]);

        let compiled = compile(&signals);
        assert!(compiled.core.is_some());
        assert!(!compiled.mutation_authority_valid);
        assert!(compiled.tool_authority_valid);
        assert_eq!(
            compiled.authority.allowed_tool_names,
            vec!["manage_mandates".to_string()]
        );
    }

    #[test]
    fn ungrounded_observation_lane_cannot_install_a_generic_success_gate() {
        let compiled = compile(&base_signals());
        assert!(compiled.core.is_none());
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "composition"
                && !decision.accepted
                && decision.reason_code == "observation_lifecycle_missing_typed_obligation"
        }));
    }

    #[test]
    fn named_invocation_forces_observation_even_when_core_flag_is_false() {
        let mut signals = base_signals();
        signals.task_kind = Some("conversational".to_string());
        signals.requires_observation = Some(false);
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["manage_mandates".to_string()],
            exit_codes: vec![0],
            outcome_statuses: vec![crate::traits::ToolOutcomeStatus::Succeeded],
            outcome_condition: None,
            requires_output: true,
            contract_rejected: Some(false),
            min_invocations: None,
            max_invocations: None,
        }]);

        let compiled = compile(&signals);
        assert_eq!(compiled.required_invocations.len(), 1);
        assert!(compiled.required_invocations[0].exit_codes.is_empty());
        assert!(compiled.core.expect("valid core").requires_observation);
    }

    #[test]
    fn generic_receipts_drop_only_impossible_process_negative_statuses() {
        let mut signals = base_signals();
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["manage_mandates".to_string()],
            exit_codes: vec![0],
            outcome_statuses: vec![
                crate::traits::ToolOutcomeStatus::Succeeded,
                crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult,
            ],
            requires_output: true,
            ..RequestReceiptPredicate::default()
        }]);

        let compiled = compile(&signals);
        let predicate = &compiled.required_invocations[0];
        assert!(predicate.exit_codes.is_empty());
        assert_eq!(
            predicate.outcome_statuses,
            [crate::traits::ToolOutcomeStatus::Succeeded]
        );
    }

    #[test]
    fn generic_adapter_translates_process_only_negative_condition() {
        let mut signals = base_signals();
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["manage_mandates".to_string()],
            outcome_condition: Some(RequestedOutcomeCondition::CompletedWithNegativeResult),
            requires_output: true,
            max_invocations: Some(1),
            ..RequestReceiptPredicate::default()
        }]);

        let compiled = compile(&signals);
        let predicate = &compiled.required_invocations[0];
        assert_eq!(
            predicate.outcome_condition,
            Some(RequestedOutcomeCondition::NonSuccessTerminal)
        );
        assert!(!predicate.requires_output);
    }

    #[test]
    fn process_adapter_preserves_completed_negative_condition_and_output() {
        let mut signals = base_signals();
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["terminal".to_string()],
            outcome_condition: Some(RequestedOutcomeCondition::CompletedWithNegativeResult),
            requires_output: true,
            max_invocations: Some(1),
            ..RequestReceiptPredicate::default()
        }]);

        let compiled = compile(&signals);
        let predicate = &compiled.required_invocations[0];
        assert_eq!(
            predicate.outcome_condition,
            Some(RequestedOutcomeCondition::CompletedWithNegativeResult)
        );
        assert!(predicate.requires_output);
    }

    #[test]
    fn process_exit_code_normalizes_an_impossible_model_outcome_conjunction() {
        let mut signals = base_signals();
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["terminal".to_string()],
            exit_codes: vec![1],
            outcome_statuses: vec![crate::traits::ToolOutcomeStatus::FailedPermanent],
            contract_rejected: Some(false),
            max_invocations: Some(1),
            ..RequestReceiptPredicate::default()
        }]);

        let compiled = compile(&signals);
        assert_eq!(compiled.required_invocations.len(), 1);
        assert_eq!(
            compiled.required_invocations[0].outcome_statuses,
            [crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult]
        );
        assert_eq!(compiled.required_invocations[0].exit_codes, [1]);
    }

    #[test]
    fn equivalent_receipt_obligations_coalesce_to_the_strongest_cardinality() {
        let mut signals = base_signals();
        let receipt = |minimum: usize, maximum: usize| RequestReceiptPredicate {
            tool_names: vec!["terminal".to_string()],
            exit_codes: vec![0],
            outcome_condition: Some(RequestedOutcomeCondition::Succeeded),
            min_invocations: Some(minimum),
            max_invocations: Some(maximum),
            ..RequestReceiptPredicate::default()
        };
        // These predicates carry no typed distinction that could identify
        // which terminal receipt belongs to which obligation. Keeping them as
        // three independent gates would make the runtime choose the smallest
        // max (one) and reject the second concrete operation. The exact-three
        // requirement is the strongest distinguishable contract.
        signals.required_invocations = Some(vec![receipt(3, 3), receipt(1, 1), receipt(1, 1)]);

        let compiled = compile(&signals);

        assert_eq!(compiled.required_invocations.len(), 1);
        let predicate = &compiled.required_invocations[0];
        assert_eq!(predicate.min_invocations, Some(3));
        assert_eq!(predicate.max_invocations, Some(3));
    }

    #[test]
    fn rejected_before_dispatch_predicate_cannot_also_require_an_exit_code() {
        let mut signals = base_signals();
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["terminal".to_string()],
            exit_codes: vec![1],
            outcome_statuses: vec![crate::traits::ToolOutcomeStatus::FailedPermanent],
            contract_rejected: Some(true),
            ..RequestReceiptPredicate::default()
        }]);

        let compiled = compile(&signals);
        assert!(compiled.required_invocations[0].exit_codes.is_empty());
        assert!(compiled.required_invocations[0].outcome_statuses.is_empty());
    }

    #[test]
    fn semantic_non_success_condition_replaces_low_level_protocol_conjunctions() {
        let mut signals = base_signals();
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["manage_mandates".to_string()],
            outcome_statuses: vec![crate::traits::ToolOutcomeStatus::Succeeded],
            outcome_condition: Some(RequestedOutcomeCondition::NonSuccessTerminal),
            contract_rejected: Some(true),
            max_invocations: Some(1),
            ..RequestReceiptPredicate::default()
        }]);

        let compiled = compile(&signals);
        let predicate = &compiled.required_invocations[0];
        assert_eq!(
            predicate.outcome_condition,
            Some(RequestedOutcomeCondition::NonSuccessTerminal)
        );
        assert!(predicate.outcome_statuses.is_empty());
        assert_eq!(predicate.contract_rejected, None);
    }

    #[test]
    fn contradictory_read_only_authority_cannot_install_mutation_lifecycle() {
        let mut signals = base_signals();
        signals.task_kind = Some("change".to_string());
        signals.expects_mutation = Some(true);
        signals.required_effects = Some(vec!["local_source_write".to_string()]);
        signals.mutation_scope = Some("read_only".to_string());

        let compiled = compile(&signals);
        assert!(compiled.core.is_none());
        assert!(compiled.mutation_authority_valid);
        assert!(compiled.tool_authority_valid);
        assert_eq!(compiled.authority.mutation_scope, "read_only");
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "composition"
                && decision.reason_code == "mutation_lifecycle_conflicts_with_read_only_authority"
        }));
    }

    #[test]
    fn tool_prohibition_cannot_install_mutation_lifecycle() {
        let mut signals = base_signals();
        signals.task_kind = Some("change".to_string());
        signals.expects_mutation = Some(true);
        signals.required_effects = Some(vec!["local_source_write".to_string()]);
        signals.tool_scope = Some("forbidden".to_string());

        let compiled = compile(&signals);
        assert!(compiled.core.is_none());
        assert!(compiled.authority.forbids_tool_use);
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "composition"
                && decision.reason_code == "mutation_lifecycle_conflicts_with_tool_prohibition"
        }));
    }

    #[test]
    fn destructive_risk_qualifier_is_not_accepted_as_completion_effect() {
        let mut signals = base_signals();
        signals.task_kind = Some("change".to_string());
        signals.expects_mutation = Some(true);
        signals.required_effects = Some(vec![
            "local_source_write".to_string(),
            "destructive".to_string(),
        ]);

        let compiled = compile(&signals);
        assert!(compiled.core.is_none());
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "completion_core"
                && decision.reason_code == "risk_qualifier_is_not_completion_effect"
        }));
    }

    #[test]
    fn exact_success_response_is_accepted_as_a_typed_artifact() {
        let mut signals = base_signals();
        signals.response_contract = Some(super::super::task_planning::PlannedResponseContract {
            mode: "exact_text".to_string(),
            success_text: Some("synthetic".to_string()),
        });

        let compiled = compile(&signals);
        assert_eq!(
            compiled
                .response_contract
                .as_ref()
                .map(|contract| contract.success_text()),
            Some("synthetic")
        );
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "response_presentation"
                && decision.accepted
                && decision.reason_code == "accepted"
        }));
    }

    #[test]
    fn malformed_response_artifact_is_rejected_without_affecting_execution() {
        let mut signals = base_signals();
        signals.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["manage_mandates".to_string()],
            outcome_condition: Some(RequestedOutcomeCondition::Succeeded),
            ..RequestReceiptPredicate::default()
        }]);
        signals.response_contract = Some(super::super::task_planning::PlannedResponseContract {
            mode: "unsupported_mode".to_string(),
            success_text: Some("synthetic".to_string()),
        });

        let compiled = compile(&signals);
        assert!(compiled.core.is_some());
        assert!(compiled.response_contract.is_none());
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "response_presentation"
                && !decision.accepted
                && decision.reason_code == "unsupported_presentation_mode"
        }));
    }

    #[test]
    fn receipt_bound_evidence_and_invocation_share_one_proof_identity() {
        let receipt = RequestReceiptPredicate {
            tool_names: vec!["terminal".to_string()],
            outcome_statuses: vec![crate::traits::ToolOutcomeStatus::FailedPermanent],
            ..RequestReceiptPredicate::default()
        };
        let mut signals = base_signals();
        signals.evidence_requirements = Some(vec![RequestEvidenceRequirement {
            summary: "Observe an unavailable process working directory".to_string(),
            acceptable_scopes: vec![ToolSemanticScope::HostLocal],
            purpose: EvidencePurpose::Outcome,
            minimum_authority: EvidenceAuthority::Canonical,
            temporal_scope: EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: Some(receipt.clone()),
            target: None,
        }]);
        signals.required_invocations = Some(vec![receipt.clone()]);

        let compiled = compile(&signals);
        assert!(compiled.evidence_requirements.is_empty());
        assert_eq!(compiled.required_invocations, [receipt]);
    }

    #[test]
    fn prose_evidence_fields_do_not_participate_in_compilation() {
        let mut left = base_signals();
        left.required_invocations = Some(vec![RequestReceiptPredicate {
            tool_names: vec!["manage_mandates".to_string()],
            ..RequestReceiptPredicate::default()
        }]);
        let mut right = left.clone();
        left.constraint_evidence = vec!["one arbitrary sentence".to_string()];
        right.constraint_evidence = vec!["completely different wording".to_string()];
        left.tool_constraint_evidence = vec!["first wording".to_string()];
        right.tool_constraint_evidence = vec!["second wording".to_string()];

        let left = compile(&left);
        let right = compile(&right);
        assert_eq!(left.required_invocations, right.required_invocations);
        assert_eq!(
            left.authority.allowed_tool_names,
            right.authority.allowed_tool_names
        );
        assert_eq!(
            left.authority.forbids_tool_use,
            right.authority.forbids_tool_use
        );
    }

    #[test]
    fn filesystem_roles_are_grounded_without_dropping_exact_request_resources() {
        let mut signals = base_signals();
        signals.filesystem_access = Some(PlannedFilesystemAccess {
            execution_cwd: Some("/tmp".to_string()),
            read_paths: vec![
                "/Users/synthetic/.cargo".to_string(),
                "/Users/synthetic/.rustup".to_string(),
            ],
            write_paths: vec!["/tmp/synthetic-future-root".to_string()],
            read_roots: Vec::new(),
            write_roots: Vec::new(),
        });
        let resources = vec![
            "/tmp".to_string(),
            "/Users/synthetic/.cargo".to_string(),
            "/Users/synthetic/.rustup".to_string(),
            "/tmp/synthetic-future-root".to_string(),
        ];
        let compiled = compile_task_contract(ContractCompilerInput {
            signals: &signals,
            task_shape: None,
            available_tool_names: &["terminal".to_string()],
            available_tool_receipt_kinds: &[(
                "terminal".to_string(),
                crate::traits::ToolReceiptKind::Process,
            )],
            structural_filesystem_resources: &resources,
            structural_project_scopes: &resources,
            project_alias_roots: &[],
            current_user_text: "synthetic request",
        });

        let access = compiled.filesystem_access.expect("filesystem authority");
        assert_eq!(access.execution_cwd.as_deref(), Some("/tmp"));
        assert_eq!(access.read_targets.len(), 2);
        assert_eq!(access.write_targets.len(), 1);
        assert_eq!(access.write_targets[0].value, "/tmp/synthetic-future-root");
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "filesystem_authority"
                && decision.accepted
                && decision.reason_code == "accepted"
        }));
    }

    #[test]
    fn directory_resources_compile_as_descendant_capabilities_but_files_stay_exact() {
        let dir = tempfile::tempdir().expect("directory");
        let directory = dir.path().join("output");
        let file = dir.path().join("input.txt");
        std::fs::create_dir_all(&directory).expect("output directory");
        std::fs::write(&file, "synthetic").expect("input file");
        let directory = directory.to_string_lossy().to_string();
        let file = file.to_string_lossy().to_string();
        let mut signals = base_signals();
        signals.task_kind = Some("change".to_string());
        signals.expects_mutation = Some(true);
        signals.requires_observation = Some(false);
        signals.required_effects = Some(vec!["local_source_write".to_string()]);
        signals.filesystem_access = Some(PlannedFilesystemAccess {
            read_paths: vec![file.clone()],
            write_paths: vec![directory.clone()],
            ..PlannedFilesystemAccess::default()
        });
        let compiled = compile_task_contract(ContractCompilerInput {
            signals: &signals,
            task_shape: None,
            available_tool_names: &["terminal".to_string()],
            available_tool_receipt_kinds: &[(
                "terminal".to_string(),
                crate::traits::ToolReceiptKind::Process,
            )],
            structural_filesystem_resources: &[file.clone(), directory.clone()],
            structural_project_scopes: &[],
            project_alias_roots: &[],
            current_user_text: "synthetic request",
        });
        let access = compiled.filesystem_access.expect("filesystem authority");
        assert_eq!(access.read_targets[0].kind, ToolTargetHintKind::Path);
        assert_eq!(
            access.write_targets[0].kind,
            ToolTargetHintKind::ProjectScope
        );
    }

    #[test]
    fn explicit_future_root_compiles_as_a_directory_capability() {
        let root = "/tmp/synthetic-future-root-for-contract";
        let mut signals = base_signals();
        signals.task_kind = Some("change".to_string());
        signals.expects_mutation = Some(true);
        signals.requires_observation = Some(false);
        signals.required_effects = Some(vec!["local_source_write".to_string()]);
        signals.filesystem_access = Some(PlannedFilesystemAccess {
            write_roots: vec![root.to_string()],
            ..PlannedFilesystemAccess::default()
        });
        let compiled = compile_task_contract(ContractCompilerInput {
            signals: &signals,
            task_shape: None,
            available_tool_names: &["terminal".to_string()],
            available_tool_receipt_kinds: &[(
                "terminal".to_string(),
                crate::traits::ToolReceiptKind::Process,
            )],
            structural_filesystem_resources: &[root.to_string()],
            structural_project_scopes: &[],
            project_alias_roots: &[],
            current_user_text: "synthetic request",
        });
        let access = compiled.filesystem_access.expect("filesystem authority");
        assert_eq!(
            access.write_targets[0].kind,
            ToolTargetHintKind::ProjectScope
        );
    }

    #[test]
    fn project_reference_promotes_future_write_root_without_prose_inference() {
        let root = "/tmp/synthetic-project-reference-root";
        let mut signals = base_signals();
        signals.task_kind = Some("change".to_string());
        signals.expects_mutation = Some(true);
        signals.requires_observation = Some(false);
        signals.required_effects = Some(vec!["local_derived_write".to_string()]);
        signals.project_reference = Some(root.to_string());
        // The semantic producer omitted filesystem roles. The current request
        // still supplied one exact, structural project identity, so the
        // compiler may install only that bounded directory capability.
        signals.filesystem_access = Some(PlannedFilesystemAccess::default());
        let compiled = compile_task_contract(ContractCompilerInput {
            signals: &signals,
            task_shape: None,
            available_tool_names: &["terminal".to_string()],
            available_tool_receipt_kinds: &[(
                "terminal".to_string(),
                crate::traits::ToolReceiptKind::Process,
            )],
            structural_filesystem_resources: &[root.to_string()],
            structural_project_scopes: &[root.to_string()],
            project_alias_roots: &[],
            current_user_text: "synthetic request",
        });
        let access = compiled.filesystem_access.expect("filesystem authority");
        assert_eq!(access.write_targets.len(), 1);
        assert_eq!(
            access.write_targets[0].kind,
            ToolTargetHintKind::ProjectScope
        );
        assert_eq!(access.write_targets[0].value, root);
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "filesystem_authority"
                && decision.reason_code == "project_scope_fallback"
                && decision.accepted
        }));
    }

    #[test]
    fn exact_child_write_remains_exact_when_project_reference_is_an_ancestor() {
        let root = "/tmp/synthetic-project-reference-root";
        let child = format!("{root}/src/main.rs");
        let mut signals = base_signals();
        signals.task_kind = Some("change".to_string());
        signals.expects_mutation = Some(true);
        signals.requires_observation = Some(false);
        signals.required_effects = Some(vec!["local_source_write".to_string()]);
        signals.project_reference = Some(root.to_string());
        signals.filesystem_access = Some(PlannedFilesystemAccess {
            write_paths: vec![child.clone()],
            ..PlannedFilesystemAccess::default()
        });
        let compiled = compile_task_contract(ContractCompilerInput {
            signals: &signals,
            task_shape: None,
            available_tool_names: &["write_file".to_string()],
            available_tool_receipt_kinds: &[(
                "write_file".to_string(),
                crate::traits::ToolReceiptKind::Generic,
            )],
            structural_filesystem_resources: &[root.to_string(), child.clone()],
            structural_project_scopes: &[root.to_string()],
            project_alias_roots: &[],
            current_user_text: "synthetic request",
        });
        let access = compiled.filesystem_access.expect("filesystem authority");
        assert!(access
            .write_targets
            .iter()
            .any(|target| target.kind == ToolTargetHintKind::Path && target.value == child));
        assert!(!access.write_targets.iter().any(|target| {
            target.kind == ToolTargetHintKind::ProjectScope && target.value == root
        }));
    }

    #[test]
    fn explicitly_named_future_ancestor_authorizes_named_descendant_creation() {
        let root = "/tmp/synthetic-future-root-for-descendant";
        let child = format!("{root}/.keep");
        let mut signals = base_signals();
        signals.task_kind = Some("change".to_string());
        signals.expects_mutation = Some(true);
        signals.requires_observation = Some(false);
        signals.required_effects = Some(vec!["local_source_write".to_string()]);
        signals.filesystem_access = Some(PlannedFilesystemAccess {
            write_paths: vec![child.clone()],
            ..PlannedFilesystemAccess::default()
        });
        let resources = vec![root.to_string(), child.clone()];
        let compiled = compile_task_contract(ContractCompilerInput {
            signals: &signals,
            task_shape: None,
            available_tool_names: &["write_file".to_string()],
            available_tool_receipt_kinds: &[(
                "write_file".to_string(),
                crate::traits::ToolReceiptKind::Generic,
            )],
            structural_filesystem_resources: &resources,
            structural_project_scopes: &[],
            project_alias_roots: &[],
            current_user_text: "synthetic request",
        });
        let access = compiled.filesystem_access.expect("filesystem authority");
        assert!(access
            .write_targets
            .iter()
            .any(|target| target.kind == ToolTargetHintKind::Path && target.value == child));
        assert!(access.write_targets.iter().any(|target| {
            target.kind == ToolTargetHintKind::ProjectScope && target.value == root
        }));
    }

    #[test]
    fn research_policy_cannot_attach_to_host_local_process_evidence() {
        let mut signals = base_signals();
        signals.minimum_sources = Some(1);
        signals.requires_primary_sources = Some(false);
        signals.evidence_requirements = Some(vec![RequestEvidenceRequirement {
            summary: "Observe the synthetic process outcome".to_string(),
            acceptable_scopes: vec![ToolSemanticScope::HostLocal],
            purpose: EvidencePurpose::Outcome,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        }]);

        let compiled = compile(&signals);

        assert_eq!(compiled.evidence_policy.minimum_sources, 0);
        assert!(!compiled.evidence_policy.requires_primary_sources);
        assert_eq!(compiled.evidence_requirements.len(), 1);
        assert!(compiled.core.is_some());
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "research_evidence_policy"
                && !decision.accepted
                && decision.reason_code == "missing_remote_subject_evidence"
        }));
    }

    #[test]
    fn research_policy_composes_with_external_subject_evidence() {
        let mut signals = base_signals();
        signals.minimum_sources = Some(2);
        signals.requires_primary_sources = Some(true);
        signals.evidence_requirements = Some(vec![RequestEvidenceRequirement {
            summary: "Establish the synthetic release state".to_string(),
            acceptable_scopes: vec![ToolSemanticScope::ExternalRemote],
            purpose: EvidencePurpose::CurrentState,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        }]);

        let compiled = compile(&signals);

        assert_eq!(compiled.evidence_policy.minimum_sources, 2);
        assert!(compiled.evidence_policy.requires_primary_sources);
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "research_evidence_policy"
                && decision.accepted
                && decision.reason_code == "accepted"
        }));
    }
}
