use super::task_planning::{
    planned_contract_is_confident, PlannedContractSignals, PlannedFilesystemAccess,
    PlannedTaskShape,
};
use crate::agent::CompletionTaskKind;
use crate::traits::{
    EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
    RequestReceiptPredicate, ToolCallAccessManifest, ToolMutationEffects, ToolSemanticScope,
    ToolTargetHint, ToolTargetHintKind,
};
use serde::Serialize;

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
    pub evidence_requirements: Vec<RequestEvidenceRequirement>,
    pub required_invocations: Vec<RequestReceiptPredicate>,
    pub filesystem_access: Option<ToolCallAccessManifest>,
    pub project_scope: Option<String>,
    pub decisions: Vec<ContractLaneDecision>,
}

pub(crate) struct ContractCompilerInput<'a> {
    pub signals: &'a PlannedContractSignals,
    pub task_shape: Option<&'a PlannedTaskShape>,
    pub available_tool_names: &'a [String],
    pub structural_filesystem_resources: &'a [String],
    pub structural_project_scopes: &'a [String],
    pub project_alias_roots: &'a [String],
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

    let minimum_sources = signals.minimum_sources.ok_or("missing_minimum_sources")? as usize;
    let requires_primary_sources = signals
        .requires_primary_sources
        .ok_or("missing_primary_source_policy")?;
    if minimum_sources > 20 || requires_primary_sources && minimum_sources == 0 {
        return Err("invalid_source_policy");
    }

    Ok(CompiledCompletionCore {
        expects_mutation,
        requires_observation,
        task_kind,
        required_mutation_effects,
        minimum_sources,
        requires_primary_sources,
        requires_exact_history: signals.requires_exact_history.unwrap_or(false),
    })
}

fn valid_receipt(receipt: &RequestReceiptPredicate, available_tool_names: &[String]) -> bool {
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
        && receipt.exit_codes.len() <= 16
        && receipt.outcome_statuses.len() <= 6
}

fn canonical_invocation_requirement(
    receipt: RequestReceiptPredicate,
) -> RequestEvidenceRequirement {
    RequestEvidenceRequirement {
        summary: "Complete the exact requested machine invocation".to_string(),
        acceptable_scopes: Vec::new(),
        purpose: EvidencePurpose::Outcome,
        minimum_authority: EvidenceAuthority::Direct,
        temporal_scope: EvidenceTemporalScope::Current,
        required_content_markers: Vec::new(),
        receipt: Some(receipt),
        target: None,
    }
}

fn compile_obligations(
    signals: &PlannedContractSignals,
    available_tool_names: &[String],
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

    let mut add_invocation = |receipt: &RequestReceiptPredicate| {
        if valid_receipt(receipt, available_tool_names) && !invocations.contains(receipt) {
            invocations.push(receipt.clone());
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
            || !crate::agent::inquiry::requirement_has_builtin_evidence_route(candidate)
        {
            continue;
        }
        let mut normalized = candidate.clone();
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
) -> (CompiledAuthority, Vec<ContractLaneDecision>) {
    let mut authority = CompiledAuthority::default();
    let mut decisions = Vec::new();

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
            decisions.push(decision("mutation_authority", true, "accepted", 1, 1));
        }
        Some("read_only" | "read-only") => {
            authority.mutation_scope = "read_only";
            decisions.push(decision("mutation_authority", true, "accepted", 1, 1));
        }
        Some("scoped") if !forbidden_actions.is_empty() => {
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
        Some("allowed") if signals.allowed_tool_names.is_empty() => decisions.push(decision(
            "tool_authority",
            true,
            "accepted",
            1 + signals.forbidden_tool_scopes.len(),
            1 + authority.forbidden_tool_scopes.len(),
        )),
        Some("forbidden") if signals.allowed_tool_names.is_empty() => {
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
    (authority, decisions)
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
) -> (Option<ToolCallAccessManifest>, ContractLaneDecision) {
    let Some(candidate) = candidate else {
        return (
            None,
            decision("filesystem_authority", false, "missing", 0, 0),
        );
    };
    let candidate_count = usize::from(candidate.execution_cwd.is_some())
        + candidate.read_paths.len()
        + candidate.write_paths.len();
    let execution_cwd = candidate
        .execution_cwd
        .as_deref()
        .and_then(|path| resolve_structural_path(path, structural_resources, alias_roots));
    let mut read_targets = candidate
        .read_paths
        .iter()
        .filter_map(|path| resolve_structural_path(path, structural_resources, alias_roots))
        .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::Path, path))
        .collect::<Vec<_>>();
    let mut write_targets = candidate
        .write_paths
        .iter()
        .filter_map(|path| resolve_structural_path(path, structural_resources, alias_roots))
        .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::Path, path))
        .collect::<Vec<_>>();
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
    });
    (
        manifest,
        decision(
            "filesystem_authority",
            candidate_count == 0 || installed_count > 0,
            if candidate_count == installed_count {
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

    let (evidence, invocations, obligation_decision) =
        compile_obligations(input.signals, input.available_tool_names);
    compiled.evidence_requirements = evidence;
    compiled.required_invocations = invocations;
    compiled.decisions.push(obligation_decision);

    let (authority, authority_decisions) =
        compile_authority(input.signals, input.available_tool_names);
    compiled.authority = authority;
    compiled.decisions.extend(authority_decisions);

    if compiled.core.as_ref().is_some_and(|core| {
        core.expects_mutation && compiled.authority.mutation_scope == "read_only"
    }) {
        compiled.core = None;
        compiled.decisions.push(decision(
            "composition",
            false,
            "mutation_lifecycle_conflicts_with_read_only_authority",
            2,
            1,
        ));
    } else {
        compiled
            .decisions
            .push(decision("composition", true, "accepted", 2, 2));
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
        if core.requires_exact_history
            && !compiled.evidence_requirements.iter().any(|requirement| {
                requirement
                    .acceptable_scopes
                    .contains(&ToolSemanticScope::ConversationHistory)
                    && requirement.purpose == EvidencePurpose::HistoricalRecord
                    && requirement.minimum_authority == EvidenceAuthority::Canonical
            })
        {
            core.requires_exact_history = false;
            compiled.decisions.push(decision(
                "exact_history",
                false,
                "missing_canonical_history_obligation",
                1,
                0,
            ));
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
        compile_task_contract(ContractCompilerInput {
            signals,
            task_shape: None,
            available_tool_names: &["manage_mandates".to_string(), "terminal".to_string()],
            structural_filesystem_resources: &[],
            structural_project_scopes: &[],
            project_alias_roots: &[],
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

        let compiled = compile(&signals);
        assert!(compiled.core.is_some());
        assert_eq!(compiled.required_invocations.len(), 1);
        assert!(compiled.authority.allowed_tool_names.is_empty());
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "tool_authority"
                && !decision.accepted
                && decision.reason_code == "no_registered_allowed_capability"
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
            requires_output: true,
            contract_rejected: Some(false),
        }]);

        let compiled = compile(&signals);
        assert_eq!(compiled.required_invocations.len(), 1);
        assert!(compiled.core.expect("valid core").requires_observation);
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
        assert_eq!(compiled.authority.mutation_scope, "read_only");
        assert!(compiled.decisions.iter().any(|decision| {
            decision.lane == "composition"
                && decision.reason_code == "mutation_lifecycle_conflicts_with_read_only_authority"
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
}
