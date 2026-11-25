//! Completion-contract inference: deriving what a turn must observe/mutate/verify.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged. Owns the completion types ([`CompletionContract`],
//! [`CompletionProgress`], [`CompletionTaskKind`], [`VerificationTarget`],
//! [`VerificationTargetKind`], `CompletionSignals`) and the `infer_*` /
//! `extract_verification_*` helpers.

use regex::Regex;
use std::sync::LazyLock;

use aidaemon_events::execution_graph::{
    ExecutionEdgeKind, ExecutionGraph, ExecutionNodeKind, ExecutionNodeState,
};
use aidaemon_events::traits::{
    RequestCompletionContract, RequestEvidenceRequirement, RequestForbiddenAction,
    RequestResponseContract, RequestTaskKind, RequestVerificationTarget,
    RequestVerificationTargetKind,
};
use aidaemon_events::traits::{ToolCallSemantics, ToolMutationEffects, ToolOutcomeStatus};

/// How a request treats connected-service content (drafting copy, delivering
/// it live, or both). Advisory only: nothing in the runtime decides tools,
/// mutation or completion from it, and production turns leave it `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectedContentMode {
    #[default]
    None,
    DraftOnly,
    DeliverOnly,
    DraftThenDeliver,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompletionTaskKind {
    #[default]
    Conversational,
    Answer,
    Check,
    Find,
    Change,
    Deliver,
    Schedule,
    Monitor,
    Diagnose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationTargetKind {
    Url,
    Path,
    ResourceId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationTarget {
    pub kind: VerificationTargetKind,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForbiddenMutationAction {
    Create,
    Delete,
    Deploy,
    Publish,
    Post,
    Send,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompletionContract {
    /// Current task that owns every field, target, requirement, and constraint
    /// in this contract. The container is the authoritative scope boundary: no
    /// unbound or differently-bound contract may drive execution/finalization.
    pub scope_task_id: Option<String>,
    /// Explicit lineage for prior contracts adopted by a typed follow-up edge.
    pub adopted_from_task_ids: Vec<String>,
    pub task_kind: CompletionTaskKind,
    pub expects_mutation: bool,
    /// Typed outcomes that must be observed before a mutating request is
    /// considered fulfilled. An empty set retains legacy count-based behavior.
    pub required_mutation_effects: ToolMutationEffects,
    /// The user explicitly constrained this turn to observation/reporting.
    /// Unlike `expects_mutation = false`, this is a hard negative obligation:
    /// mutation attempts are contract violations and must be blocked.
    pub forbids_mutation: bool,
    /// Explicit, grounded prohibition on every tool call for this request.
    pub forbids_tool_use: bool,
    /// Explicit allow-only tool boundary. Empty means no name-level allowlist.
    pub allowed_tool_names: Vec<String>,
    /// Owner control-plane tools the runtime keeps outside a narrowed
    /// allowlist. Kept so older persisted contracts keep round-tripping.
    pub owner_control_plane_tools: Vec<String>,
    /// Explicit, grounded capability deny-set for this request. Enforcement
    /// happens both when definitions are offered and again before dispatch.
    pub forbidden_tool_scopes: Vec<aidaemon_events::traits::ToolSemanticScope>,
    /// Typed receipt-triggered lane closures for conditional workflows.
    /// These are task-local attenuation rules, never inferred from prose at
    /// execution time and never allowed to grant a capability.
    pub dispatch_stop_rules: Vec<aidaemon_events::traits::RequestDispatchStopRule>,
    /// Retained for persisted-schema compatibility. Final prose is not proof;
    /// runtime completion deliberately ignores these legacy labels.
    pub required_response_fields: Vec<String>,
    /// Exact successful response artifact produced for the current request.
    /// It is presentation state and never execution evidence.
    pub response_contract: Option<Box<RequestResponseContract>>,
    /// Operation-specific negative obligations. These do not turn an otherwise
    /// mutating task into a report-only task; they block only the named action.
    pub forbidden_mutation_actions: Vec<ForbiddenMutationAction>,
    pub requires_observation: bool,
    pub requires_reverification_after_mutation: bool,
    pub explicit_verification_requested: bool,
    /// Semantic evidence requirements from the validated task contract.
    pub minimum_sources: usize,
    pub requires_primary_sources: bool,
    pub requires_exact_history: bool,
    /// Material information needs whose typed evidence must be closed before
    /// successful completion. Empty retains legacy generic-observation behavior.
    pub evidence_requirements: Vec<RequestEvidenceRequirement>,
    /// Durable source obligation -> semantic requirement bindings for typed
    /// continuation adoption. Never infer these from local vector positions.
    pub adopted_evidence_bindings: Vec<aidaemon_events::traits::AdoptedEvidenceBinding>,
    pub connected_content_mode: ConnectedContentMode,
    pub verification_targets: Vec<VerificationTarget>,
    /// The owner's units of work, compiled from the owner's own message by
    /// the request compiler ([`crate::turn::task_planning::compile_owner_units`]),
    /// apart from the model that carries them out. The closeout judges the
    /// run against them, never only against the executor's own checklist.
    pub owner_units: Vec<aidaemon_events::traits::RequestWorkUnit>,
}

impl CompletionContract {
    pub fn adopt_for_task(&mut self, task_id: &str) {
        if let Some(prior) = self
            .scope_task_id
            .as_deref()
            .filter(|prior| *prior != task_id)
            && !self.adopted_from_task_ids.iter().any(|id| id == prior)
        {
            self.adopted_from_task_ids.push(prior.to_string());
        }
        self.scope_task_id = Some(task_id.to_string());
    }

    pub fn belongs_to_task(&self, task_id: &str) -> bool {
        self.scope_task_id.as_deref() == Some(task_id)
    }
}

pub fn persistable_completion_contract(contract: &CompletionContract) -> RequestCompletionContract {
    RequestCompletionContract {
        scope_task_id: contract.scope_task_id.clone(),
        adopted_from_task_ids: contract.adopted_from_task_ids.clone(),
        task_kind: match contract.task_kind {
            CompletionTaskKind::Conversational => RequestTaskKind::Conversational,
            CompletionTaskKind::Answer => RequestTaskKind::Answer,
            CompletionTaskKind::Check => RequestTaskKind::Check,
            CompletionTaskKind::Find => RequestTaskKind::Find,
            CompletionTaskKind::Change => RequestTaskKind::Change,
            CompletionTaskKind::Deliver => RequestTaskKind::Deliver,
            CompletionTaskKind::Schedule => RequestTaskKind::Schedule,
            CompletionTaskKind::Monitor => RequestTaskKind::Monitor,
            CompletionTaskKind::Diagnose => RequestTaskKind::Diagnose,
        },
        expects_mutation: contract.expects_mutation,
        required_mutation_effects: contract.required_mutation_effects,
        forbids_mutation: contract.forbids_mutation,
        forbids_tool_use: contract.forbids_tool_use,
        allowed_tool_names: contract.allowed_tool_names.clone(),
        forbidden_tool_scopes: contract.forbidden_tool_scopes.clone(),
        dispatch_stop_rules: contract.dispatch_stop_rules.clone(),
        required_response_fields: contract.required_response_fields.clone(),
        response_contract: contract.response_contract.clone(),
        forbidden_actions: contract
            .forbidden_mutation_actions
            .iter()
            .map(|action| match action {
                ForbiddenMutationAction::Create => RequestForbiddenAction::Create,
                ForbiddenMutationAction::Delete => RequestForbiddenAction::Delete,
                ForbiddenMutationAction::Deploy => RequestForbiddenAction::Deploy,
                ForbiddenMutationAction::Publish => RequestForbiddenAction::Publish,
                ForbiddenMutationAction::Post => RequestForbiddenAction::Post,
                ForbiddenMutationAction::Send => RequestForbiddenAction::Send,
            })
            .collect(),
        requires_observation: contract.requires_observation,
        requires_reverification_after_mutation: contract.requires_reverification_after_mutation,
        explicit_verification_requested: contract.explicit_verification_requested,
        minimum_sources: contract.minimum_sources,
        requires_primary_sources: contract.requires_primary_sources,
        requires_exact_history: contract.requires_exact_history,
        evidence_requirements: contract.evidence_requirements.clone(),
        adopted_evidence_bindings: contract.adopted_evidence_bindings.clone(),
        verification_targets: contract
            .verification_targets
            .iter()
            .map(|target| RequestVerificationTarget {
                kind: match target.kind {
                    VerificationTargetKind::Url => RequestVerificationTargetKind::Url,
                    VerificationTargetKind::Path => RequestVerificationTargetKind::Path,
                    VerificationTargetKind::ResourceId => RequestVerificationTargetKind::ResourceId,
                },
                value: target.value.clone(),
            })
            .collect(),
        owner_units: contract.owner_units.clone(),
    }
}

pub fn completion_contract_from_persisted(
    contract: &RequestCompletionContract,
) -> CompletionContract {
    // Durable open requests can outlive an evidence-ontology upgrade. Recheck
    // their typed tuples at hydration so a legacy impossible obligation cannot
    // survive a restart and keep an otherwise answerable task in validation
    // forever. This never guesses a replacement requirement; the generic
    // observation bit remains available as a conservative fallback.
    let reachable_evidence_requirements = contract
        .evidence_requirements
        .iter()
        .filter(|requirement| super::inquiry::requirement_has_builtin_evidence_route(requirement))
        .cloned()
        .collect();
    CompletionContract {
        scope_task_id: contract.scope_task_id.clone(),
        adopted_from_task_ids: contract.adopted_from_task_ids.clone(),
        task_kind: match contract.task_kind {
            RequestTaskKind::Conversational => CompletionTaskKind::Conversational,
            RequestTaskKind::Answer => CompletionTaskKind::Answer,
            RequestTaskKind::Check => CompletionTaskKind::Check,
            RequestTaskKind::Find => CompletionTaskKind::Find,
            RequestTaskKind::Change => CompletionTaskKind::Change,
            RequestTaskKind::Deliver => CompletionTaskKind::Deliver,
            RequestTaskKind::Schedule => CompletionTaskKind::Schedule,
            RequestTaskKind::Monitor => CompletionTaskKind::Monitor,
            RequestTaskKind::Diagnose => CompletionTaskKind::Diagnose,
        },
        expects_mutation: contract.expects_mutation,
        required_mutation_effects: contract.required_mutation_effects,
        forbids_mutation: contract.forbids_mutation,
        forbids_tool_use: contract.forbids_tool_use,
        allowed_tool_names: contract.allowed_tool_names.clone(),
        owner_control_plane_tools: Vec::new(),
        forbidden_tool_scopes: contract.forbidden_tool_scopes.clone(),
        dispatch_stop_rules: contract.dispatch_stop_rules.clone(),
        required_response_fields: contract.required_response_fields.clone(),
        response_contract: contract.response_contract.clone(),
        forbidden_mutation_actions: contract
            .forbidden_actions
            .iter()
            .map(|action| match action {
                RequestForbiddenAction::Create => ForbiddenMutationAction::Create,
                RequestForbiddenAction::Delete => ForbiddenMutationAction::Delete,
                RequestForbiddenAction::Deploy => ForbiddenMutationAction::Deploy,
                RequestForbiddenAction::Publish => ForbiddenMutationAction::Publish,
                RequestForbiddenAction::Post => ForbiddenMutationAction::Post,
                RequestForbiddenAction::Send => ForbiddenMutationAction::Send,
            })
            .collect(),
        requires_observation: contract.requires_observation,
        requires_reverification_after_mutation: contract.requires_reverification_after_mutation,
        explicit_verification_requested: contract.explicit_verification_requested,
        minimum_sources: contract.minimum_sources,
        requires_primary_sources: contract.requires_primary_sources,
        requires_exact_history: contract.requires_exact_history,
        evidence_requirements: reachable_evidence_requirements,
        adopted_evidence_bindings: contract
            .adopted_evidence_bindings
            .iter()
            .filter(|binding| {
                super::inquiry::requirement_has_builtin_evidence_route(&binding.requirement)
                    && contract
                        .evidence_requirements
                        .contains(&binding.requirement)
            })
            .cloned()
            .collect(),
        connected_content_mode: ConnectedContentMode::None,
        verification_targets: contract
            .verification_targets
            .iter()
            .map(|target| VerificationTarget {
                kind: match target.kind {
                    RequestVerificationTargetKind::Url => VerificationTargetKind::Url,
                    RequestVerificationTargetKind::Path => VerificationTargetKind::Path,
                    RequestVerificationTargetKind::ResourceId => VerificationTargetKind::ResourceId,
                },
                value: target.value.clone(),
            })
            .collect(),
        owner_units: contract.owner_units.clone(),
    }
}

/// Preserve the typed obligations of a persisted request that has not reached
/// a terminal outcome. The caller decides whether the requests are related
/// from dialogue state; this function deliberately performs no text matching.
pub fn inherit_unfinished_request_contract(
    mut current: CompletionContract,
    unfinished: &CompletionContract,
) -> CompletionContract {
    if let Some(origin) = unfinished.scope_task_id.as_ref()
        && current.scope_task_id.as_ref() != Some(origin)
        && !current.adopted_from_task_ids.contains(origin)
    {
        current.adopted_from_task_ids.push(origin.clone());
    }
    for origin in &unfinished.adopted_from_task_ids {
        if current.scope_task_id.as_ref() != Some(origin)
            && !current.adopted_from_task_ids.contains(origin)
        {
            current.adopted_from_task_ids.push(origin.clone());
        }
    }
    let current_requires_execution = current.expects_mutation || current.requires_observation;
    let may_carry_mutation = !current.forbids_mutation;

    if may_carry_mutation && unfinished.expects_mutation {
        current.expects_mutation = true;
        current.required_mutation_effects = current
            .required_mutation_effects
            .union(unfinished.required_mutation_effects);
        current.requires_reverification_after_mutation |=
            unfinished.requires_reverification_after_mutation;
    }

    current.requires_observation |= unfinished.requires_observation;
    current.explicit_verification_requested |= unfinished.explicit_verification_requested;
    // Evidence-quality policy belongs to the same unfinished lifecycle as its
    // proof obligations. Carry it monotonically with those obligations so a
    // partial continuation assessment cannot retain the work while silently
    // weakening the standard that closes it.
    current.minimum_sources = current.minimum_sources.max(unfinished.minimum_sources);
    current.requires_primary_sources |= unfinished.requires_primary_sources;
    current.requires_exact_history |= unfinished.requires_exact_history;
    for (index, requirement) in unfinished.evidence_requirements.iter().enumerate() {
        if !current.evidence_requirements.contains(requirement) {
            current.evidence_requirements.push(requirement.clone());
        }
        if let Some(origin) = unfinished.scope_task_id.as_deref() {
            let source_obligation_id = format!("task:{origin}/obligation:evidence:{index}");
            if !current
                .adopted_evidence_bindings
                .iter()
                .any(|binding| binding.source_obligation_id == source_obligation_id)
            {
                current.adopted_evidence_bindings.push(
                    aidaemon_events::traits::AdoptedEvidenceBinding {
                        source_obligation_id,
                        requirement: requirement.clone(),
                    },
                );
            }
        }
    }
    for binding in &unfinished.adopted_evidence_bindings {
        if current.evidence_requirements.contains(&binding.requirement)
            && !current
                .adopted_evidence_bindings
                .iter()
                .any(|existing| existing.source_obligation_id == binding.source_obligation_id)
        {
            current.adopted_evidence_bindings.push(binding.clone());
        }
    }

    if !current_requires_execution
        && (current.expects_mutation || current.requires_observation)
        && !current.forbids_mutation
    {
        current.task_kind = unfinished.task_kind;
    }

    // Persist scoped safety constraints across a continuation. A current turn
    // that explicitly requests mutation supersedes an older blanket
    // observation-only constraint, while operation-specific exclusions remain.
    if !current.expects_mutation {
        current.forbids_mutation |= unfinished.forbids_mutation;
    }
    // A current explicit prohibition wins. An older prohibition is inherited
    // only while this related continuation has no positive execution request.
    if !current_requires_execution {
        current.forbids_tool_use |= unfinished.forbids_tool_use;
        if current.allowed_tool_names.is_empty() {
            current.allowed_tool_names = unfinished.allowed_tool_names.clone();
        } else if !unfinished.allowed_tool_names.is_empty() {
            current
                .allowed_tool_names
                .retain(|name| unfinished.allowed_tool_names.contains(name));
        }
    }
    for scope in &unfinished.forbidden_tool_scopes {
        if !current.forbidden_tool_scopes.contains(scope) {
            current.forbidden_tool_scopes.push(*scope);
        }
    }
    for rule in &unfinished.dispatch_stop_rules {
        if !current.dispatch_stop_rules.contains(rule) {
            current.dispatch_stop_rules.push(rule.clone());
        }
    }
    for field in &unfinished.required_response_fields {
        if !current.required_response_fields.contains(field) {
            current.required_response_fields.push(field.clone());
        }
    }
    if current.response_contract.is_none() {
        current.response_contract = unfinished.response_contract.clone();
    }
    if current.forbids_tool_use {
        current.allowed_tool_names.clear();
        current.requires_observation = false;
        current.explicit_verification_requested = false;
    }
    for action in &unfinished.forbidden_mutation_actions {
        if !current.forbidden_mutation_actions.contains(action) {
            current.forbidden_mutation_actions.push(*action);
        }
    }
    for target in &unfinished.verification_targets {
        if !current.verification_targets.contains(target) {
            current.verification_targets.push(target.clone());
        }
    }

    current
}

/// Carry only durable authority and safety constraints from a completed
/// antecedent into a semantically related new turn. Satisfied response fields,
/// evidence requirements, mutation effects, and verification targets belong
/// to the completed lifecycle and must never become fresh obligations.
pub fn inherit_request_constraints(
    mut current: CompletionContract,
    antecedent: &CompletionContract,
) -> CompletionContract {
    if let Some(origin) = antecedent.scope_task_id.as_ref()
        && current.scope_task_id.as_ref() != Some(origin)
        && !current.adopted_from_task_ids.contains(origin)
    {
        current.adopted_from_task_ids.push(origin.clone());
    }
    for origin in &antecedent.adopted_from_task_ids {
        if current.scope_task_id.as_ref() != Some(origin)
            && !current.adopted_from_task_ids.contains(origin)
        {
            current.adopted_from_task_ids.push(origin.clone());
        }
    }

    let current_requires_execution = current.expects_mutation || current.requires_observation;
    // Blanket execution constraints are request-local. A later, semantically
    // related turn may explicitly authorize a new observation or mutation;
    // that current contract outranks a completed antecedent. Specific denied
    // scopes/actions still carry unless the policy gains an explicit typed
    // representation for revoking them.
    if !current_requires_execution {
        current.forbids_tool_use |= antecedent.forbids_tool_use;
        if current.allowed_tool_names.is_empty() {
            current.allowed_tool_names = antecedent.allowed_tool_names.clone();
        } else if !antecedent.allowed_tool_names.is_empty() {
            current
                .allowed_tool_names
                .retain(|name| antecedent.allowed_tool_names.contains(name));
        }
    }
    if !current.expects_mutation {
        current.forbids_mutation |= antecedent.forbids_mutation;
    }
    for scope in &antecedent.forbidden_tool_scopes {
        if !current.forbidden_tool_scopes.contains(scope) {
            current.forbidden_tool_scopes.push(*scope);
        }
    }
    for action in &antecedent.forbidden_mutation_actions {
        if !current.forbidden_mutation_actions.contains(action) {
            current.forbidden_mutation_actions.push(*action);
        }
    }
    if current.forbids_tool_use {
        current.allowed_tool_names.clear();
        current.requires_observation = false;
        current.explicit_verification_requested = false;
        current.evidence_requirements.clear();
    }
    current
}

fn structural_evidence_requirement(
    targets: &[VerificationTarget],
) -> Option<RequestEvidenceRequirement> {
    if targets.is_empty() {
        return None;
    }
    let mut acceptable_scopes = Vec::new();
    if targets
        .iter()
        .any(|target| target.kind == VerificationTargetKind::Path)
    {
        acceptable_scopes.push(aidaemon_events::traits::ToolSemanticScope::LocalWorkspace);
    }
    if targets
        .iter()
        .any(|target| target.kind == VerificationTargetKind::Url)
    {
        acceptable_scopes.push(aidaemon_events::traits::ToolSemanticScope::ExternalRemote);
    }
    Some(RequestEvidenceRequirement {
        summary: "Observe the current state of one exact resource named by the request".to_string(),
        acceptable_scopes,
        purpose: aidaemon_events::traits::EvidencePurpose::CurrentState,
        minimum_authority: aidaemon_events::traits::EvidenceAuthority::Direct,
        temporal_scope: aidaemon_events::traits::EvidenceTemporalScope::Current,
        required_content_markers: Vec::new(),
        receipt: None,
        // Target matching remains a separate OR across structural identities.
        // Binding this requirement to one arbitrarily selected path would make
        // incidental working directories or executable names authoritative.
        target: None,
    })
}

/// Reduce the bootstrap contract to what the request structurally names: exact
/// URLs and structurally resolved paths remain target hints for one generic
/// observation obligation.
///
/// A resource identity is not itself a material information need. A request can
/// mention a working directory, an executable path, an earlier comparison
/// location, and the resource to inspect. Promoting every extracted identity to
/// a separate proof node made incidental and historical paths authoritative.
/// The successful observation must still match at least one exact target, but
/// the fallback creates only one evidence need regardless of how many
/// identities were extracted.
pub fn retain_structural_completion_contract(contract: &mut CompletionContract) {
    let verification_targets = std::mem::take(&mut contract.verification_targets);
    let scope_task_id = contract.scope_task_id.take();
    let adopted_from_task_ids = std::mem::take(&mut contract.adopted_from_task_ids);
    let requires_observation = !verification_targets.is_empty();
    let evidence_requirements = structural_evidence_requirement(&verification_targets)
        .into_iter()
        .collect();
    *contract = CompletionContract {
        scope_task_id,
        adopted_from_task_ids,
        task_kind: if requires_observation {
            CompletionTaskKind::Check
        } else {
            CompletionTaskKind::Conversational
        },
        requires_observation,
        explicit_verification_requested: requires_observation,
        evidence_requirements,
        verification_targets,
        ..CompletionContract::default()
    };
}

/// Construct the bootstrap contract from concrete resource identities only,
/// with relative filesystem targets anchored to the turn's typed workspace
/// (or the daemon cwd when the turn has none).
pub fn infer_structural_completion_contract_anchored(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    text: &str,
    alias_roots: &[String],
    workspace_anchor: Option<&std::path::Path>,
) -> CompletionContract {
    let mut contract = CompletionContract {
        verification_targets: extract_verification_targets_anchored(
            backend,
            text,
            alias_roots,
            workspace_anchor,
        ),
        ..CompletionContract::default()
    };
    retain_structural_completion_contract(&mut contract);
    contract
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompletionProgress {
    /// Stable namespace for every node in this task's completion graph.
    /// Runtime-created graphs never use a process-global "current" request
    /// node, so evidence from another task cannot become executable merely
    /// because its local obligation label happens to match.
    pub task_scope: String,
    pub observation_count: usize,
    /// Mutation-capable invocations that crossed the adapter's pre-I/O
    /// contract boundary. Unlike `mutation_count`, this records attempts even
    /// when their final state cannot be proven successful.
    pub mutation_attempt_count: usize,
    /// Mutation attempts whose terminal receipt cannot prove that no partial
    /// effect occurred. This is epistemic state, not confirmed completion.
    pub indeterminate_mutation_count: usize,
    pub mutation_count: usize,
    /// Effects observed from successful calls. This is separate from the raw
    /// count so incidental cache writes cannot fulfill source/deploy outcomes.
    pub observed_mutation_effects: ToolMutationEffects,
    pub verification_count: usize,
    pub verification_pending: bool,
    /// Count of external mutation attempts that failed (4xx/5xx, etc.)
    pub failed_external_mutation_count: usize,
    /// Count of external mutation attempts that succeeded
    pub successful_external_mutation_count: usize,
    /// True after we have already given the LLM one reconciliation-informed retry.
    pub external_mutation_reconciliation_attempted: bool,
    /// True after the one-shot RECOVERY pass: before asking the model to
    /// report failed external mutations honestly, it first gets one
    /// evidence-fed chance to fix them with a DIFFERENT approach.
    pub external_mutation_recovery_attempted: bool,
    /// True after the one-shot recovery for an authored local artifact that
    /// has not yet been delivered to the user.
    pub undelivered_artifact_recovery_attempted: bool,
    /// Cumulative count of times the verification guard blocked completion.
    /// Unlike `stall_count` (which is shared with other stall mechanisms and
    /// gets reset), this counter only increments and is used as a safety valve
    /// to prevent infinite verification loops.
    pub verification_block_count: usize,
    /// Count of compatible verification tool calls that actually executed
    /// while an observation obligation was pending. Completion blocks are
    /// tracked separately and must never masquerade as exhausted retries.
    pub verification_attempt_count: usize,
    /// `completed_operation_results` at the last verification re-ask. The
    /// re-ask is bounded by receipts: it fires again only after new completed
    /// work changed what the ledger can credit, never merely because an
    /// expectation is still open.
    pub verification_demanded_at_completed: Option<usize>,
    /// `completed_operation_results` at the last demand for the executor's
    /// own declared checklist items. Bounded the same way: one demand per
    /// ledger state, never a second one for the same unchanged receipts.
    pub executor_demanded_at_completed: Option<usize>,
    /// Count of times the response-quality nudge has been injected.
    /// Used to prevent infinite nudge loops — only fire once.
    pub quality_nudge_count: usize,
    /// Count of times the locate-file retry nudge has been injected (model
    /// asked the user to upload a file it could find itself). Fires once.
    pub file_access_retry_count: usize,
    /// Count of retries issued to satisfy an explicit source-count and direct
    /// citation requirement from the user.
    pub source_evidence_nudge_count: usize,
    /// Count of retries issued when the draft claimed exact conversation
    /// history was unavailable without attempting canonical retrieval.
    pub history_lookup_nudge_count: usize,
    /// Count of bounded retries issued after a completion claim lacked the
    /// typed mutation receipt required by the request.
    pub mutation_claim_nudge_count: usize,
    /// Count of bounded retries issued because a mandate task lead attempted
    /// to finish without an exact durable decision for its current run.
    pub mandate_decision_retry_count: usize,
    /// Per-turn proof graph. Legacy/default-constructed progress values keep
    /// using counters; runtime construction initializes this graph and makes
    /// typed receipt evidence authoritative for completion.
    pub proof_graph: ExecutionGraph,
    pub proof_graph_initialized: bool,
    pub mutation_obligation_ids: Vec<String>,
    pub verification_obligation_id: Option<String>,
    /// One proof-graph obligation per `CompletionContract::evidence_requirements`
    /// entry, preserving index alignment for receipt matching.
    pub evidence_obligation_ids: Vec<String>,
    /// Distinct task-local receipts accumulated for each evidence obligation.
    /// The graph edge is installed only after the predicate's typed minimum
    /// cardinality is reached.
    pub evidence_receipt_ids: Vec<Vec<String>>,
}

impl CompletionProgress {
    pub fn new(contract: &CompletionContract, task_id: &str) -> Self {
        let mut progress = Self {
            task_scope: format!("task:{task_id}"),
            verification_pending: contract.requires_observation,
            ..Self::default()
        };
        // Runtime-created progress always uses graph proof. If construction
        // ever hits an internal invariant, keep the partial graph authoritative
        // (fail closed) rather than falling back to mutation counters.
        progress.proof_graph_initialized = true;
        if let Err(error) = progress.initialize_proof_graph(contract) {
            tracing::error!(%error, "Completion proof graph initialization failed closed");
        }
        progress
    }

    fn initialize_proof_graph(&mut self, contract: &CompletionContract) -> Result<(), String> {
        let request_id = self.scoped_node_id("request");
        self.proof_graph.add_node(
            request_id.clone(),
            ExecutionNodeKind::Request,
            ExecutionNodeState::Running,
        )?;

        if contract.expects_mutation {
            let task_scope = self.task_scope.clone();
            let mut add_mutation_obligation = |id: String| -> Result<(), String> {
                self.proof_graph.add_node(
                    id.clone(),
                    ExecutionNodeKind::Obligation,
                    ExecutionNodeState::Pending,
                )?;
                self.proof_graph
                    .add_edge(&request_id, &id, ExecutionEdgeKind::Requires, None)?;
                self.mutation_obligation_ids.push(id);
                Ok(())
            };

            if contract.required_mutation_effects.is_empty() {
                add_mutation_obligation(format!("{task_scope}/obligation:mutation:any"))?;
            } else {
                for (label, effect) in mutation_effect_obligations() {
                    if contract.required_mutation_effects.intersects(effect) {
                        add_mutation_obligation(format!(
                            "{task_scope}/obligation:mutation:{label}"
                        ))?;
                    }
                }
            }
        }

        if contract.requires_observation && !contract.evidence_requirements.is_empty() {
            self.evidence_receipt_ids = vec![Vec::new(); contract.evidence_requirements.len()];
            for (index, _) in contract.evidence_requirements.iter().enumerate() {
                let id = self.scoped_node_id(&format!("obligation:evidence:{index}"));
                self.proof_graph.add_node(
                    id.clone(),
                    ExecutionNodeKind::Obligation,
                    ExecutionNodeState::Pending,
                )?;
                self.proof_graph
                    .add_edge(&request_id, &id, ExecutionEdgeKind::Requires, None)?;
                self.evidence_obligation_ids.push(id);
            }
        } else if contract.requires_observation {
            let id = self.scoped_node_id("obligation:verification");
            self.proof_graph.add_node(
                id.clone(),
                ExecutionNodeKind::Obligation,
                ExecutionNodeState::Pending,
            )?;
            self.proof_graph
                .add_edge(&request_id, &id, ExecutionEdgeKind::Requires, None)?;
            self.verification_obligation_id = Some(id);
        }

        Ok(())
    }

    fn scoped_node_id(&self, local_id: &str) -> String {
        format!("{}/{}", self.task_scope, local_id)
    }

    fn record_receipt_node(&mut self, tool_call_id: &str) -> Option<String> {
        if !self.proof_graph_initialized {
            return None;
        }
        let id = self.scoped_node_id(&format!("receipt:{tool_call_id}"));
        if self.proof_graph.node_kind(&id) == Some(ExecutionNodeKind::Receipt) {
            return Some(id);
        }
        match self.proof_graph.add_node(
            id.clone(),
            ExecutionNodeKind::Receipt,
            ExecutionNodeState::Satisfied,
        ) {
            Ok(()) => Some(id),
            Err(error) => {
                tracing::warn!(%error, "Completion proof graph rejected a tool receipt");
                None
            }
        }
    }

    /// Retain successful task-local evidence in the durable completion proof
    /// even when the assessment model omitted a blocking requirement for it.
    /// This edge is intentionally not a dependency of the request node: it
    /// improves causal audit coverage without allowing incidental activity to
    /// decide whether the task is complete.
    fn record_supporting_receipt(&mut self, tool_call_id: &str) {
        if !self.proof_graph_initialized {
            return;
        }
        let obligation_id = self.scoped_node_id(&format!("support:evidence:{tool_call_id}"));
        if self.proof_graph.node_kind(&obligation_id).is_some() {
            return;
        }
        let Some(receipt_id) = self.record_receipt_node(tool_call_id) else {
            return;
        };
        if let Err(error) = self.proof_graph.add_node(
            obligation_id.clone(),
            ExecutionNodeKind::Obligation,
            ExecutionNodeState::Pending,
        ) {
            tracing::warn!(%error, "Completion graph rejected supporting evidence obligation");
            return;
        }
        if let Err(error) = self.proof_graph.satisfy_with_evidence(
            &obligation_id,
            &receipt_id,
            Some(tool_call_id.to_string()),
        ) {
            tracing::warn!(%error, "Completion graph rejected supporting evidence receipt");
        }
    }

    pub fn record_mutation_attempt(&mut self, outcome: ToolOutcomeStatus) {
        if outcome == ToolOutcomeStatus::Blocked {
            return;
        }
        self.mutation_attempt_count = self.mutation_attempt_count.saturating_add(1);
        if matches!(
            outcome,
            ToolOutcomeStatus::CompletedWithNegativeResult
                | ToolOutcomeStatus::FailedRetryable
                | ToolOutcomeStatus::FailedPermanent
        ) {
            self.indeterminate_mutation_count = self.indeterminate_mutation_count.saturating_add(1);
        }
    }

    pub fn mark_mutation_receipt(
        &mut self,
        contract: &CompletionContract,
        semantics: &ToolCallSemantics,
        tool_call_id: &str,
    ) {
        self.mutation_count = self.mutation_count.saturating_add(1);
        self.record_supporting_receipt(tool_call_id);
        let effects = if semantics.mutation_effects.is_empty() {
            ToolMutationEffects::UNSPECIFIED
        } else {
            semantics.mutation_effects
        };
        self.observed_mutation_effects = self.observed_mutation_effects.union(effects);
        if let Some(receipt_id) = self.record_receipt_node(tool_call_id) {
            for (label, required_effect) in mutation_effect_obligations() {
                let obligation_id = self.scoped_node_id(&format!("obligation:mutation:{label}"));
                let observed_matches = if required_effect == ToolMutationEffects::UNSPECIFIED {
                    !effects.is_empty()
                } else {
                    effects.contains(required_effect)
                };
                if observed_matches
                    && self.mutation_obligation_ids.contains(&obligation_id)
                    && let Err(error) = self.proof_graph.satisfy_with_evidence(
                        &obligation_id,
                        &receipt_id,
                        Some(tool_call_id.to_string()),
                    )
                {
                    tracing::warn!(%error, %obligation_id, "Completion proof was rejected");
                }
            }
            let generic_id = self.scoped_node_id("obligation:mutation:any");
            if self.proof_graph_initialized
                && self
                    .mutation_obligation_ids
                    .iter()
                    .any(|id| id == &generic_id)
                && let Err(error) = self.proof_graph.satisfy_with_evidence(
                    &generic_id,
                    &receipt_id,
                    Some(tool_call_id.to_string()),
                )
            {
                tracing::warn!(%error, "Generic mutation completion proof was rejected");
            }
        }
        if contract.requires_reverification_after_mutation {
            self.verification_pending = true;
            if self.proof_graph_initialized
                && let Some(receipt_id) = self.record_receipt_node(tool_call_id)
            {
                let mut invalidated = self
                    .verification_obligation_id
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>();
                invalidated.extend(
                    contract
                        .evidence_requirements
                        .iter()
                        .enumerate()
                        .filter_map(|(index, requirement)| {
                            matches!(
                                requirement.temporal_scope,
                                aidaemon_events::traits::EvidenceTemporalScope::Current
                                    | aidaemon_events::traits::EvidenceTemporalScope::Both
                            )
                            .then(|| self.evidence_obligation_ids.get(index).cloned())
                            .flatten()
                        }),
                );
                for obligation_id in invalidated {
                    if let Err(error) = self.proof_graph.invalidate(&receipt_id, &obligation_id) {
                        tracing::warn!(%error, %obligation_id, "Verification invalidation was rejected");
                    }
                }
            }
        }
    }

    pub fn mark_observation_receipt(
        &mut self,
        contract: &CompletionContract,
        matched_requirement_indices: &[usize],
        generic_match: bool,
        tool_call_id: &str,
    ) {
        self.observation_count = self.observation_count.saturating_add(1);
        self.record_supporting_receipt(tool_call_id);
        if !contract.requires_observation {
            return;
        }

        if !contract.evidence_requirements.is_empty() {
            for index in matched_requirement_indices {
                let Some(receipts) = self.evidence_receipt_ids.get_mut(*index) else {
                    continue;
                };
                if !receipts.iter().any(|receipt| receipt == tool_call_id) {
                    receipts.push(tool_call_id.to_string());
                }
            }
            let obligation_ids = matched_requirement_indices
                .iter()
                .filter(|index| {
                    let minimum = contract
                        .evidence_requirements
                        .get(**index)
                        .and_then(|requirement| requirement.receipt.as_ref())
                        .and_then(|predicate| predicate.min_invocations)
                        .unwrap_or(1);
                    self.evidence_receipt_ids
                        .get(**index)
                        .is_some_and(|receipts| receipts.len() >= minimum)
                })
                .filter_map(|index| self.evidence_obligation_ids.get(*index))
                .filter(|id| self.proof_graph.state(id) != Some(ExecutionNodeState::Satisfied))
                .cloned()
                .collect::<Vec<_>>();
            if !obligation_ids.is_empty() {
                self.verification_count = self.verification_count.saturating_add(1);
                self.record_verification_evidence(tool_call_id, &obligation_ids);
            }
            self.verification_pending = self
                .evidence_obligation_ids
                .iter()
                .any(|id| self.proof_graph.state(id) != Some(ExecutionNodeState::Satisfied));
        } else if generic_match || contract.verification_targets.is_empty() {
            self.verification_pending = false;
            self.verification_count = self.verification_count.saturating_add(1);
            let obligation_ids = self
                .verification_obligation_id
                .iter()
                .cloned()
                .collect::<Vec<_>>();
            self.record_verification_evidence(tool_call_id, &obligation_ids);
        }
    }

    pub fn mark_verification_attempt(&mut self) {
        if self.verification_pending {
            self.verification_attempt_count = self.verification_attempt_count.saturating_add(1);
        }
    }

    fn record_verification_evidence(&mut self, tool_call_id: &str, obligation_ids: &[String]) {
        if !self.proof_graph_initialized {
            return;
        }
        if obligation_ids.is_empty() {
            return;
        }
        let id = self.scoped_node_id(&format!("verification:{tool_call_id}"));
        if let Err(error) = self.proof_graph.add_node(
            id.clone(),
            ExecutionNodeKind::Verification,
            ExecutionNodeState::Satisfied,
        ) {
            tracing::warn!(%error, "Completion proof graph rejected verification evidence");
            return;
        }
        for obligation_id in obligation_ids {
            if let Err(error) = self.proof_graph.satisfy_with_evidence(
                obligation_id,
                &id,
                Some(tool_call_id.to_string()),
            ) {
                tracing::warn!(%error, %obligation_id, "Completion proof graph rejected verification evidence");
            }
        }
    }

    pub fn completion_obligations_for_receipt(&self, tool_call_id: &str) -> Vec<String> {
        self.proof_graph
            .obligations_satisfied_by_receipt(tool_call_id)
    }
}

fn mutation_effect_obligations() -> [(&'static str, ToolMutationEffects); 11] {
    [
        (
            "local_source_write",
            ToolMutationEffects::LOCAL_SOURCE_WRITE,
        ),
        (
            "local_workspace_write",
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
        ),
        (
            "local_derived_write",
            ToolMutationEffects::LOCAL_DERIVED_WRITE,
        ),
        ("repository_write", ToolMutationEffects::REPOSITORY_WRITE),
        ("remote_mutation", ToolMutationEffects::REMOTE_MUTATION),
        ("remote_deploy", ToolMutationEffects::REMOTE_DEPLOY),
        ("external_delivery", ToolMutationEffects::EXTERNAL_DELIVERY),
        ("process_state", ToolMutationEffects::PROCESS_STATE),
        ("configuration", ToolMutationEffects::CONFIGURATION),
        ("destructive", ToolMutationEffects::DESTRUCTIVE),
        ("unspecified", ToolMutationEffects::UNSPECIFIED),
    ]
}

pub static HTTP_URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)\bhttps?://[^\s"'()<>]+"#).expect("valid http url regex"));

fn extract_verification_filesystem_targets(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    text: &str,
    targets: &mut Vec<VerificationTarget>,
    max_targets: usize,
    alias_roots: &[String],
    workspace_anchor: Option<&std::path::Path>,
) {
    for value in super::project_scope::extract_exact_filesystem_resources_from_text_anchored(
        backend,
        text,
        alias_roots,
        workspace_anchor,
    ) {
        if targets.len() >= max_targets {
            break;
        }
        if !targets
            .iter()
            .any(|target| target.kind == VerificationTargetKind::Path && target.value == value)
        {
            targets.push(VerificationTarget {
                kind: VerificationTargetKind::Path,
                value,
            });
        }
    }
}

fn extract_verification_targets_anchored(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    text: &str,
    alias_roots: &[String],
    workspace_anchor: Option<&std::path::Path>,
) -> Vec<VerificationTarget> {
    let mut targets = Vec::new();

    for capture in HTTP_URL_RE.captures_iter(text) {
        let raw = capture
            .get(0)
            .map(|m| m.as_str())
            .unwrap_or_default()
            .trim_end_matches(['.', ',', ';', ')', ']', '}'])
            .to_string();
        if raw.is_empty()
            || targets.iter().any(|existing: &VerificationTarget| {
                existing.kind == VerificationTargetKind::Url && existing.value == raw
            })
        {
            continue;
        }
        targets.push(VerificationTarget {
            kind: VerificationTargetKind::Url,
            value: raw,
        });
    }

    extract_verification_filesystem_targets(
        backend,
        text,
        &mut targets,
        4,
        alias_roots,
        workspace_anchor,
    );

    targets
}

/// Narrow a freshly inferred contract to what a delegated executor child can
/// actually satisfy on its own.
///
/// An executor runs one narrowly-scoped step of a decomposed plan; the
/// orchestrating parent owns goal-level verification and routinely delegates
/// it to a sibling task. Reverification inferred from mutation-flavored
/// phrases ("deploy", "publish") in the mission would demand a verification
/// the child was never asked to perform — and stamp a fully successful step
/// `partial`. Reverification survives only when the mission itself asks for
/// it.
pub fn scope_contract_for_delegated_executor(
    mut contract: CompletionContract,
) -> CompletionContract {
    if !contract.explicit_verification_requested {
        contract.requires_reverification_after_mutation = false;
    }
    contract
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_impossible_evidence_is_quarantined_during_hydration() {
        let mut persisted = persistable_completion_contract(&CompletionContract::default());
        persisted.requires_observation = true;
        persisted.evidence_requirements = vec![RequestEvidenceRequirement {
            summary: "Establish a canonical historical cause from personal memory".to_string(),
            acceptable_scopes: vec![aidaemon_events::traits::ToolSemanticScope::UserMemory],
            purpose: aidaemon_events::traits::EvidencePurpose::CausalExplanation,
            minimum_authority: aidaemon_events::traits::EvidenceAuthority::Canonical,
            temporal_scope: aidaemon_events::traits::EvidenceTemporalScope::Historical,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        }];

        let hydrated = completion_contract_from_persisted(&persisted);
        assert!(hydrated.requires_observation);
        assert!(hydrated.evidence_requirements.is_empty());
        assert!(CompletionProgress::new(&hydrated, "synthetic-current-task").verification_pending);
    }

    #[test]
    fn failed_assessment_retains_only_structural_resource_obligation() {
        let target = VerificationTarget {
            kind: VerificationTargetKind::Path,
            value: "/tmp/synthetic-target".to_string(),
        };
        let mut contract = CompletionContract {
            task_kind: CompletionTaskKind::Change,
            expects_mutation: true,
            required_mutation_effects: ToolMutationEffects::LOCAL_SOURCE_WRITE,
            forbids_mutation: true,
            forbidden_mutation_actions: vec![ForbiddenMutationAction::Delete],
            verification_targets: vec![target.clone()],
            ..CompletionContract::default()
        };

        retain_structural_completion_contract(&mut contract);

        assert_eq!(contract.task_kind, CompletionTaskKind::Check);
        assert!(contract.requires_observation);
        assert!(contract.explicit_verification_requested);
        assert!(!contract.expects_mutation);
        assert!(!contract.forbids_mutation);
        assert_eq!(contract.verification_targets, vec![target]);
        assert_eq!(contract.evidence_requirements.len(), 1);
        assert!(contract.evidence_requirements[0].target.is_none());
    }

    #[test]
    fn incidental_structural_paths_share_one_generic_observation_obligation() {
        let mut contract = CompletionContract {
            verification_targets: vec![
                VerificationTarget {
                    kind: VerificationTargetKind::Path,
                    value: "/tmp".to_string(),
                },
                VerificationTarget {
                    kind: VerificationTargetKind::Path,
                    value: "/synthetic/project".to_string(),
                },
                VerificationTarget {
                    kind: VerificationTargetKind::Path,
                    value: "/bin/pwd".to_string(),
                },
            ],
            ..CompletionContract::default()
        };
        retain_structural_completion_contract(&mut contract);
        let progress = CompletionProgress::new(&contract, "synthetic-task");

        assert!(contract.requires_observation);
        assert_eq!(contract.evidence_requirements.len(), 1);
        assert!(contract.evidence_requirements[0].target.is_none());
        assert!(progress.verification_obligation_id.is_none());
        assert_eq!(progress.evidence_obligation_ids.len(), 1);
        assert!(progress.mutation_obligation_ids.is_empty());
    }

    #[test]
    fn unfinished_contract_inheritance_uses_typed_state_not_followup_words() {
        let current = CompletionContract {
            task_kind: CompletionTaskKind::Answer,
            ..CompletionContract::default()
        };
        let unfinished = CompletionContract {
            task_kind: CompletionTaskKind::Change,
            expects_mutation: true,
            required_mutation_effects: ToolMutationEffects::LOCAL_SOURCE_WRITE,
            requires_observation: true,
            requires_reverification_after_mutation: true,
            explicit_verification_requested: true,
            ..CompletionContract::default()
        };

        let inherited = inherit_unfinished_request_contract(current, &unfinished);

        assert_eq!(inherited.task_kind, CompletionTaskKind::Change);
        assert!(inherited.expects_mutation);
        assert!(
            inherited
                .required_mutation_effects
                .contains(ToolMutationEffects::LOCAL_SOURCE_WRITE)
        );
        assert!(inherited.requires_observation);
        assert!(inherited.requires_reverification_after_mutation);
        assert!(inherited.explicit_verification_requested);
    }

    #[test]
    fn unfinished_contract_inheritance_preserves_evidence_policy_monotonically() {
        let current = CompletionContract {
            requires_observation: true,
            minimum_sources: 1,
            ..CompletionContract::default()
        };
        let unfinished = CompletionContract {
            requires_observation: true,
            minimum_sources: 3,
            requires_primary_sources: true,
            requires_exact_history: true,
            ..CompletionContract::default()
        };

        let inherited = inherit_unfinished_request_contract(current, &unfinished);

        assert_eq!(inherited.minimum_sources, 3);
        assert!(inherited.requires_primary_sources);
        assert!(inherited.requires_exact_history);
    }

    #[test]
    fn unfinished_request_retains_its_typed_response_artifact() {
        let expected = RequestResponseContract::ExactText {
            success_text: "phase=synthetic; outcome=complete".to_string(),
            source_message_hash: "synthetic-hash".to_string(),
        };
        let current = CompletionContract::default();
        let unfinished = CompletionContract {
            response_contract: Some(Box::new(expected.clone())),
            ..CompletionContract::default()
        };

        let inherited = inherit_unfinished_request_contract(current, &unfinished);

        assert_eq!(inherited.response_contract, Some(Box::new(expected)));
    }

    #[test]
    fn inherited_evidence_keeps_stable_source_identity_across_reorder_and_hydration() {
        let child_only = RequestEvidenceRequirement {
            summary: "Observe child state".to_string(),
            acceptable_scopes: vec![aidaemon_events::traits::ToolSemanticScope::HostLocal],
            purpose: aidaemon_events::traits::EvidencePurpose::CurrentState,
            minimum_authority: aidaemon_events::traits::EvidenceAuthority::Direct,
            temporal_scope: aidaemon_events::traits::EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        };
        let parent = RequestEvidenceRequirement {
            summary: "Observe parent command output".to_string(),
            acceptable_scopes: vec![aidaemon_events::traits::ToolSemanticScope::HostLocal],
            purpose: aidaemon_events::traits::EvidencePurpose::Content,
            minimum_authority: aidaemon_events::traits::EvidenceAuthority::Direct,
            temporal_scope: aidaemon_events::traits::EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        };
        let current = CompletionContract {
            scope_task_id: Some("task-child".to_string()),
            requires_observation: true,
            evidence_requirements: vec![child_only],
            ..CompletionContract::default()
        };
        let unfinished = CompletionContract {
            scope_task_id: Some("task-parent".to_string()),
            requires_observation: true,
            evidence_requirements: vec![parent.clone()],
            ..CompletionContract::default()
        };

        let inherited = inherit_unfinished_request_contract(current, &unfinished);
        assert_eq!(inherited.evidence_requirements[1], parent);
        assert_eq!(
            inherited.adopted_evidence_bindings[0].source_obligation_id,
            "task:task-parent/obligation:evidence:0"
        );
        assert_eq!(inherited.adopted_evidence_bindings[0].requirement, parent);

        let hydrated =
            completion_contract_from_persisted(&persistable_completion_contract(&inherited));
        assert_eq!(
            hydrated.adopted_evidence_bindings,
            inherited.adopted_evidence_bindings
        );
    }

    #[test]
    fn completed_antecedent_carries_constraints_without_reopening_satisfied_obligations() {
        let current = CompletionContract {
            scope_task_id: Some("task-current".to_string()),
            required_response_fields: vec!["current_field".to_string()],
            ..CompletionContract::default()
        };
        let antecedent = CompletionContract {
            scope_task_id: Some("task-prior".to_string()),
            forbids_tool_use: true,
            forbidden_tool_scopes: vec![aidaemon_events::traits::ToolSemanticScope::UserMemory],
            required_response_fields: vec!["old_field".to_string()],
            requires_observation: true,
            evidence_requirements: vec![RequestEvidenceRequirement {
                summary: "Satisfied prior evidence".to_string(),
                acceptable_scopes: vec![aidaemon_events::traits::ToolSemanticScope::HostLocal],
                purpose: aidaemon_events::traits::EvidencePurpose::Outcome,
                minimum_authority: aidaemon_events::traits::EvidenceAuthority::Direct,
                temporal_scope: aidaemon_events::traits::EvidenceTemporalScope::Current,
                required_content_markers: vec!["old_field".to_string()],
                receipt: None,
                target: None,
            }],
            verification_targets: vec![VerificationTarget {
                kind: VerificationTargetKind::Path,
                value: "/tmp/prior".to_string(),
            }],
            ..CompletionContract::default()
        };

        let inherited = inherit_request_constraints(current, &antecedent);
        assert!(inherited.forbids_tool_use);
        assert!(
            inherited
                .forbidden_tool_scopes
                .contains(&aidaemon_events::traits::ToolSemanticScope::UserMemory)
        );
        assert_eq!(inherited.required_response_fields, ["current_field"]);
        assert!(inherited.evidence_requirements.is_empty());
        assert!(inherited.verification_targets.is_empty());
        assert_eq!(inherited.adopted_from_task_ids, ["task-prior"]);
    }

    #[test]
    fn explicit_current_mutation_outranks_completed_read_only_antecedent() {
        let current = CompletionContract {
            expects_mutation: true,
            required_mutation_effects: ToolMutationEffects::LOCAL_SOURCE_WRITE,
            requires_observation: true,
            ..CompletionContract::default()
        };
        let antecedent = CompletionContract {
            forbids_mutation: true,
            forbids_tool_use: true,
            ..CompletionContract::default()
        };

        let inherited = inherit_request_constraints(current, &antecedent);

        assert!(!inherited.forbids_mutation);
        assert!(!inherited.forbids_tool_use);
        assert!(inherited.expects_mutation);
        assert!(inherited.requires_observation);
    }

    #[test]
    fn task_binding_requires_explicit_adoption_and_namespaces_proof_nodes() {
        let mut prior = CompletionContract {
            scope_task_id: Some("task-prior".to_string()),
            required_response_fields: vec!["synthetic_field".to_string()],
            requires_observation: true,
            ..CompletionContract::default()
        };
        assert!(!prior.belongs_to_task("task-current"));

        prior.adopt_for_task("task-current");
        assert!(prior.belongs_to_task("task-current"));
        assert_eq!(prior.adopted_from_task_ids, ["task-prior"]);

        let current_progress = CompletionProgress::new(&prior, "task-current");
        let other_progress = CompletionProgress::new(&prior, "task-other");
        assert_ne!(current_progress.task_scope, other_progress.task_scope);
        assert_eq!(
            current_progress
                .proof_graph
                .node_kind("task:task-current/request"),
            Some(ExecutionNodeKind::Request)
        );
        assert_eq!(
            other_progress
                .proof_graph
                .node_kind("task:task-other/request"),
            Some(ExecutionNodeKind::Request)
        );
        assert!(
            current_progress
                .proof_graph
                .node_kind("task:task-other/request")
                .is_none()
        );
    }

    #[test]
    fn current_observation_only_constraint_blocks_inherited_mutation() {
        let current = CompletionContract {
            task_kind: CompletionTaskKind::Check,
            forbids_mutation: true,
            requires_observation: true,
            ..CompletionContract::default()
        };
        let unfinished = CompletionContract {
            task_kind: CompletionTaskKind::Change,
            expects_mutation: true,
            required_mutation_effects: ToolMutationEffects::LOCAL_SOURCE_WRITE,
            ..CompletionContract::default()
        };

        let inherited = inherit_unfinished_request_contract(current, &unfinished);

        assert!(!inherited.expects_mutation);
        assert!(inherited.forbids_mutation);
        assert!(inherited.required_mutation_effects.is_empty());
        assert_eq!(inherited.task_kind, CompletionTaskKind::Check);
    }

    // Regression tests from the 2026-06-06 attribution run: the mutation
    // expectation misfired in BOTH directions across builds. Read-only
    // phrasings were blocked with expects_mutation=true on the older build,
    // while the turn-10 delete request must keep expects_mutation=true so
    // the zero-tool fabrication guard can fire.

    #[test]
    fn failed_mutation_attempt_is_visible_without_claiming_success() {
        let mut progress = CompletionProgress::default();
        progress.record_mutation_attempt(ToolOutcomeStatus::CompletedWithNegativeResult);

        assert_eq!(progress.mutation_attempt_count, 1);
        assert_eq!(progress.indeterminate_mutation_count, 1);
        assert_eq!(progress.mutation_count, 0);
    }
}
