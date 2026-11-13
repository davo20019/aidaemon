//! Deterministic request-run lifecycle reducer.
//!
//! The immutable event history is the source of truth. This aggregate is a
//! discardable projection: rebuilding it from the same ordered events must
//! always yield the same result. LLM output can propose contracts and tool
//! operations, but only typed events change lifecycle state here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::TaskKernelOperationClaim;
use super::{
    AssistantResponseData, AssistantResponseDisposition, Event, EventType, ExecutorExpectationItem,
    ExecutorExpectationsDeclaredData, ResponseDeliveryData, ResponseDeliveryState,
    TaskContractCompiledData, TaskEndData, TaskOutcome, TaskStartData, TaskStatus, ToolCallData,
    ToolResultData,
};
use crate::traits::{
    EvidenceTemporalScope, RequestDispatchStopRule, RequestEvidenceRequirement,
    RequestObservationTarget, RequestReceiptPredicate, RequestResponseContract,
    RequestVerificationTargetKind, ToolCollectionObservation, ToolMutationEffects,
    ToolObservationEvidence, ToolOutcomeStatus, ToolSemanticFacet, ToolTargetHint,
    ToolTargetHintKind,
};
use aidaemon_tools::LexicalPaths;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunObligationClass {
    /// A requested invocation/result occurred. This is historical and monotonic.
    Perform,
    /// A requested effect was achieved. This is historical and monotonic.
    Achieve,
    /// A state assertion holds at one effect revision and may be invalidated.
    Observe,
    /// The canonical response artifact was durably prepared for the transport
    /// outbox. Platform acknowledgement remains a separate delivery event.
    Deliver,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunObligationState {
    Pending,
    Satisfied,
    Invalidated,
    /// The executor stopped working on a checklist item: it marked it
    /// `deferred`/`skipped` or dropped it from its list. Deferral is not
    /// evidence: the run may stop over a deferred item (the closeout never
    /// demands it), but it never counts toward fulfillment, and a receipt
    /// that proves it still closes it. Serialized projections written before
    /// this distinction called it `abandoned`.
    #[serde(alias = "abandoned")]
    Deferred,
    Unverifiable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunObligation {
    pub id: String,
    pub class: RunObligationClass,
    pub state: RunObligationState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<RequestReceiptPredicate>,
    /// Full typed evidence identity for observation obligations. Earlier
    /// projections retained only the class and accidentally let any successful
    /// read satisfy any subject scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_requirement: Option<RequestEvidenceRequirement>,
    #[serde(default)]
    pub required_effect: ToolMutationEffects,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub satisfied_at_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub satisfying_receipt_ids: Vec<String>,
    /// Exact subject the proving receipt must have touched (declared path or
    /// URL). Binds both mutation and observation proof to that subject so one
    /// write cannot close every "write something" item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_target: Option<crate::traits::RequestVerificationTarget>,
    /// Exact subject/facet selectors declared by the executor. Every selector
    /// must be supported by the same completed receipt; opaque targets never
    /// degrade into a generic observation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observation_targets: Vec<RequestObservationTarget>,
    /// Human-readable statement of the obligation for demand rendering
    /// (executor-declared items carry their own description).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// The step in the executor's own words, for the owner: an executor
    /// checklist item's description, never the demand text in `summary`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
    /// A checklist observation that a later item depends on (`depends_on`):
    /// it is satisfied by having happened before that work, so mutations
    /// made for the dependent work do not make it stale.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub prerequisite: bool,
}

/// The typed reason code of a refusal the owner declined.
pub const OWNER_APPROVAL_DECLINED: &str = "owner_approval_declined";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunOperation {
    pub operation_id: String,
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_operation_key: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obligation_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_attempts: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_invocations: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<ToolOutcomeStatus>,
    #[serde(default)]
    pub dispatched: bool,
    /// The operation was refused before dispatch by a typed policy boundary
    /// (scope lock, read-only contract, mandate gate, capability constraint).
    /// Retrying the same operation cannot cross that boundary.
    #[serde(default)]
    pub policy_denied: bool,
    /// The owner declined the operation (its refusal is typed
    /// `owner_approval_declined`): the owner's answer, not an authority
    /// boundary the run closed at.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub owner_declined: bool,
    /// The proposed operation carried mutating semantics.
    #[serde(default)]
    pub mutating: bool,
    /// Evidence the operation would have produced (from its receipt
    /// semantics), retained so a refused operation can be bound to the
    /// observation obligation it was attempting to satisfy.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_capabilities: Vec<crate::traits::ToolEvidenceCapability>,
    /// Subjects the operation touched or proposed to touch (semantic target
    /// hints plus the declared access manifest), retained so a refused
    /// operation can be bound to the resource an obligation concerns.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<ToolTargetHint>,
    /// Exact result assertions retained for declaration-order-independent
    /// recredit. These are terminal receipt facts, not proposed semantics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<ToolObservationEvidence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collection_observations: Vec<ToolCollectionObservation>,
    /// Receipt-typed mutation effects, retained so an obligation declared or
    /// retargeted after this receipt landed can still be verified against it.
    #[serde(default)]
    pub mutation_effects: crate::traits::ToolMutationEffects,
    /// The terminal receipt carried observation semantics.
    #[serde(default)]
    pub observes: bool,
    /// Effect revision after this operation's terminal receipt was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_revision: Option<u64>,
    /// Position of this operation's latest receipt among the run's receipts.
    /// Unlike `result_revision` it orders a launch receipt and observations
    /// that changed nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation_lineage: Option<super::ToolOperationLineage>,
    /// The declared checklist items this operation took on (a launch's
    /// `checklist_items`, from its receipt): the typed link from work to
    /// the item it serves.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checklist_items: Vec<usize>,
}

impl RunOperation {
    /// A background handoff is an intermediate receipt for an invocation that
    /// is still running. Derive the state from the existing outcome field so
    /// aggregates serialized before this lifecycle distinction remain valid.
    fn is_background_in_flight(&self) -> bool {
        self.outcome == Some(ToolOutcomeStatus::Backgrounded)
    }

    fn has_terminal_receipt(&self) -> bool {
        self.result_id.is_some() && !self.is_background_in_flight()
    }

    /// The operation's mutation is a change a later observation could
    /// verify: anything but the delivery of the call's own local capture
    /// (see [`delivers_own_capture`]).
    pub fn changes_verifiable_state(&self) -> bool {
        self.mutating && !delivers_own_capture(self.observes, self.mutation_effects, &self.targets)
    }
}

/// A call that observed something and delivered its own local capture of it
/// (a screenshot the chat confirmed receiving): its only effect is the
/// delivery, it names only local files, and its receipt already carries
/// what it observed and that the delivery arrived. The captured file is
/// unchanged and no observation can read the chat back, so the delivery is
/// no change that needs verifying and leaves earlier observations current.
/// A delivery that names no local file (a node restart request, a message
/// to a remote service), a plain delivery that observed nothing
/// (`send_file`), and any delivery combined with another effect (a click
/// that posts) still count as changes.
pub fn delivers_own_capture(
    observes: bool,
    effects: crate::traits::ToolMutationEffects,
    targets: &[ToolTargetHint],
) -> bool {
    observes
        && effects == crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY
        && !targets.is_empty()
        && targets
            .iter()
            .all(|target| target.kind == ToolTargetHintKind::Path)
}

/// End-to-end lifecycle projection. Execution completion and user-visible
/// delivery are deliberately separate states; a generated response is not a
/// transport acknowledgement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskKernelPhase {
    Initialized,
    Running,
    WorkPending,
    WorkSucceeded,
    ResponsePrepared,
    DeliveryQueued,
    DeliveryFailed,
    Delivered,
    Partial,
    Failed,
    Cancelled,
    Interrupted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskKernelAdmission {
    Admitted,
    Rejected { code: &'static str, detail: String },
}

/// Why a pending expectation can no longer be satisfied in this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnreachableReason {
    /// An operation bound to it was refused by a typed policy boundary.
    PolicyDenied,
    /// Its bounded invocation budget is spent.
    CardinalityExhausted,
    /// No visible tool admissible under the request's authority can satisfy it.
    NoAdmissibleTool,
    /// The executor deferred or dropped its own checklist item. The loop may
    /// not demand it (stopping is permitted), but it is not proven either.
    ExecutorDeferred,
}

/// The ledger-first closeout verdict. See [`RunAggregate::closeout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloseoutDecision {
    /// Nothing more is owed: proven by contract, by evidence, or by a
    /// credited denial; or the contract carried no obligations.
    Closed { proof_basis: &'static str },
    /// These expectations are still satisfiable; the loop may ask for them.
    Reachable { obligation_ids: Vec<String> },
    /// Expectations remain but none can be satisfied; the run closes on the
    /// evidence it has and reports exactly what blocked the rest.
    Unreachable {
        blocked: Vec<(String, UnreachableReason)>,
    },
}

impl CloseoutDecision {
    pub fn verdict(&self) -> &'static str {
        match self {
            Self::Closed { .. } => "closed",
            Self::Reachable { .. } => "reachable",
            Self::Unreachable { .. } => "unreachable",
        }
    }

    /// Whether the loop may still demand more work from the model.
    pub fn work_reachable(&self) -> bool {
        matches!(self, Self::Reachable { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunTerminalDecision {
    /// Every accepted obligation has durable proof.
    Succeeded,
    /// The compiled contract is exhausted without crediting the work, but
    /// every terminal operation receipt succeeded or was credited. Evidence of
    /// completed work beats a contract that could not describe it; this is a
    /// success whose proof basis is the receipt set rather than the contract.
    SucceededByEvidence,
    /// Every open obligation was refused by a typed authority boundary before
    /// dispatch (nothing dispatched). The request could not be carried out
    /// and the run is complete at that boundary; the reply narrates it. The
    /// work did not happen, so this is final but not a success: the task
    /// ends `Partial` and the phase is `TaskKernelPhase::Partial`.
    ClosedByPolicyDenial,
    /// At least one exact bounded invocation is exhausted without proof.
    Failed,
    /// Work remains and the aggregate still permits progress or recovery.
    Pending,
    /// No task-owned contract was installed; preserve the caller's legacy path.
    Unspecified,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunAggregate {
    pub schema_version: u16,
    pub task_id: String,
    #[serde(default)]
    pub contract_present: bool,
    /// The executing model declared typed expectations of its own.
    #[serde(default)]
    pub executor_expectations_present: bool,
    /// The executing model's own verdict on the request, from its latest
    /// checklist declaration. `partial`/`blocked` cap the run below success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_claimed_outcome: Option<super::ExecutorClaimedOutcome>,
    /// The resource the executor declared its work ran out of, with an
    /// unfinished outcome ([`super::ShortResource`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_short_of: Option<super::ShortResource>,
    /// Exact resources the request named for verification (from the
    /// compiled contract). A refusal touching one of them binds every
    /// untargeted observation obligation to that boundary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verification_targets: Vec<crate::traits::RequestVerificationTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_fingerprint: Option<String>,
    #[serde(default)]
    pub effect_revision: u64,
    /// Receipts recorded so far; stamps each operation's `receipt_sequence`.
    #[serde(default)]
    pub receipt_sequence: u64,
    #[serde(default)]
    pub obligations: BTreeMap<String, RunObligation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_contract: Option<Box<RequestResponseContract>>,
    /// Empty means unrestricted. A non-empty set is the request's exact
    /// capability allowlist and is replay-checked against every operation.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub allowed_tool_names: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dispatch_stop_rules: Vec<RequestDispatchStopRule>,
    #[serde(default)]
    pub operations: BTreeMap<String, RunOperation>,
    #[serde(default)]
    pub cardinality_violations: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_causal_operation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub invariant_violations: Vec<String>,
    #[serde(default)]
    pub started: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_task_status: Option<TaskStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_task_outcome: Option<TaskOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prepared_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub delivery_states: BTreeMap<String, ResponseDeliveryState>,
    /// Source obligation id (from a task this contract adopted) -> the
    /// obligation id this run compiled for the same requirement. An adopted
    /// durable replay of the source receipt claims and completes the source
    /// id; this alias is the typed bridge that lets it prove ours.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub adopted_obligation_aliases: BTreeMap<String, String>,
    /// How this run's path targets and hints compare. The backend's fixed
    /// facts, not part of the projection, so never serialized.
    #[serde(skip, default = "LexicalPaths::daemon_host")]
    paths: LexicalPaths,
}

impl RunAggregate {
    /// The indices of the executing model's checklist items that receipts
    /// have proven (their `obligation:checklist:<n>` is satisfied).
    pub fn satisfied_checklist_items(&self) -> BTreeSet<usize> {
        self.obligations
            .iter()
            .filter(|(_, obligation)| obligation.state == RunObligationState::Satisfied)
            .filter_map(|(id, _)| {
                id.rsplit_once("/obligation:checklist:")
                    .and_then(|(_, index)| index.parse().ok())
            })
            .collect()
    }

    pub const SCHEMA_VERSION: u16 = 4;

    #[cfg(any(test, feature = "testing"))]
    pub fn new(task_id: impl Into<String>) -> Self {
        Self::on_paths(task_id, LexicalPaths::daemon_host())
    }

    pub fn on_paths(task_id: impl Into<String>, paths: LexicalPaths) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            task_id: task_id.into(),
            contract_present: false,
            executor_expectations_present: false,
            executor_claimed_outcome: None,
            executor_short_of: None,
            verification_targets: Vec::new(),
            contract_fingerprint: None,
            effect_revision: 0,
            receipt_sequence: 0,
            obligations: BTreeMap::new(),
            response_contract: None,
            allowed_tool_names: BTreeSet::new(),
            dispatch_stop_rules: Vec::new(),
            operations: BTreeMap::new(),
            cardinality_violations: 0,
            primary_causal_operation_id: None,
            invariant_violations: Vec::new(),
            started: false,
            parent_task_id: None,
            recorded_task_status: None,
            recorded_task_outcome: None,
            prepared_response_id: None,
            delivery_states: BTreeMap::new(),
            adopted_obligation_aliases: BTreeMap::new(),
            paths,
        }
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn replay(task_id: &str, events: &[Event]) -> Self {
        Self::replay_on_paths(task_id, events, LexicalPaths::daemon_host())
    }

    pub fn replay_on_paths(task_id: &str, events: &[Event], paths: LexicalPaths) -> Self {
        let mut aggregate = Self::on_paths(task_id, paths);
        let mut ordered = events.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|event| event.id);
        for event in ordered {
            aggregate.apply(event);
        }
        aggregate.reconcile_cardinality();
        aggregate
    }

    pub fn apply(&mut self, event: &Event) {
        match event.event_type {
            EventType::TaskStart => {
                if let Ok(start) = event.parse_data::<TaskStartData>() {
                    self.record_task_start(start);
                } else {
                    self.record_invariant("task_start_payload_invalid");
                }
            }
            EventType::TaskContractCompiled => {
                if let Ok(compiled) = event.parse_data::<TaskContractCompiledData>() {
                    if compiled.task_id == self.task_id {
                        self.install_contract(compiled);
                    } else {
                        self.record_invariant("contract_task_identity_mismatch");
                    }
                } else {
                    self.record_invariant("contract_payload_invalid");
                }
            }
            EventType::ExecutorExpectationsDeclared => {
                if let Ok(declared) = event.parse_data::<ExecutorExpectationsDeclaredData>() {
                    if declared.task_id == self.task_id {
                        self.install_executor_expectations(declared);
                    } else {
                        self.record_invariant("executor_expectations_task_identity_mismatch");
                    }
                } else {
                    self.record_invariant("executor_expectations_payload_invalid");
                }
            }
            EventType::ToolCall => {
                if let Ok(call) = event.parse_data::<ToolCallData>() {
                    self.record_call(call);
                } else {
                    self.record_invariant("tool_call_payload_invalid");
                }
            }
            EventType::ToolResult => {
                if let Ok(result) = event.parse_data::<ToolResultData>() {
                    self.record_result(result);
                } else {
                    self.record_invariant("tool_result_payload_invalid");
                }
            }
            EventType::AssistantResponse => {
                if let Ok(response) = event.parse_data::<AssistantResponseData>() {
                    self.record_assistant_response(response);
                } else {
                    self.record_invariant("assistant_response_payload_invalid");
                }
            }
            EventType::ResponseDelivery => {
                if let Ok(delivery) = event.parse_data::<ResponseDeliveryData>() {
                    self.record_delivery(delivery);
                } else {
                    self.record_invariant("response_delivery_payload_invalid");
                }
            }
            EventType::TaskEnd => {
                if let Ok(end) = event.parse_data::<TaskEndData>() {
                    self.record_task_end(end);
                } else {
                    self.record_invariant("task_end_payload_invalid");
                }
            }
            _ => {}
        }
    }

    fn record_task_start(&mut self, start: TaskStartData) {
        if start.task_id != self.task_id {
            self.record_invariant("task_start_identity_mismatch");
            return;
        }
        if self.started {
            if self.parent_task_id != start.parent_task_id {
                self.record_invariant("conflicting_task_start");
            }
            return;
        }
        self.started = true;
        self.parent_task_id = start.parent_task_id;
    }

    fn install_contract(&mut self, compiled: TaskContractCompiledData) {
        let fingerprint = serde_json::to_vec(&compiled).ok().map(|bytes| {
            let digest = Sha256::digest(bytes);
            hex::encode(digest)
        });
        if self.contract_present {
            if self.contract_fingerprint != fingerprint {
                self.record_invariant("conflicting_contract_installation");
            }
            return;
        }
        self.contract_present = true;
        self.contract_fingerprint = fingerprint;
        self.verification_targets = compiled.contract.verification_targets.clone();
        self.obligations
            .retain(|id, _| id.contains("/obligation:checklist:"));
        self.response_contract = compiled.contract.response_contract.clone();
        self.allowed_tool_names = compiled
            .contract
            .allowed_tool_names
            .iter()
            .cloned()
            .collect();
        self.dispatch_stop_rules = compiled.contract.dispatch_stop_rules.clone();
        self.cardinality_violations = 0;

        for (index, requirement) in compiled.contract.evidence_requirements.iter().enumerate() {
            let id = format!("task:{}/obligation:evidence:{index}", self.task_id);
            let class = if requirement.receipt.is_some() {
                // Requested invocation outcomes are immutable historical facts.
                RunObligationClass::Perform
            } else if requirement.temporal_scope == EvidenceTemporalScope::Historical {
                // Historical subject facts do not become false because a
                // later mutation advances the current-state revision.
                RunObligationClass::Achieve
            } else {
                // Current/Both subject observations are revision-bound,
                // regardless of their semantic purpose.
                RunObligationClass::Observe
            };
            self.insert_obligation(RunObligation {
                id,
                class,
                state: RunObligationState::Pending,
                receipt: requirement.receipt.clone(),
                evidence_requirement: Some(requirement.clone()),
                required_effect: ToolMutationEffects::NONE,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
                required_target: None,
                observation_targets: Vec::new(),
                summary: None,
                step: None,
                prerequisite: false,
            });
        }

        self.adopted_obligation_aliases.clear();
        for binding in &compiled.contract.adopted_evidence_bindings {
            let Some(index) = compiled
                .contract
                .evidence_requirements
                .iter()
                .position(|requirement| requirement == &binding.requirement)
            else {
                continue;
            };
            self.adopted_obligation_aliases.insert(
                binding.source_obligation_id.clone(),
                format!("task:{}/obligation:evidence:{index}", self.task_id),
            );
        }

        // Older compiled rows can carry invocation predicates separately from
        // the canonical evidence list. Preserve them once by value.
        for predicate in compiled.required_invocations {
            let duplicate = self
                .obligations
                .values()
                .any(|obligation| obligation.receipt.as_ref() == Some(&predicate));
            if duplicate {
                continue;
            }
            let index = self.obligations.len();
            self.insert_obligation(RunObligation {
                id: format!("task:{}/obligation:invocation:{index}", self.task_id),
                class: RunObligationClass::Perform,
                state: RunObligationState::Pending,
                receipt: Some(predicate),
                evidence_requirement: None,
                required_effect: ToolMutationEffects::NONE,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
                required_target: None,
                observation_targets: Vec::new(),
                summary: None,
                step: None,
                prerequisite: false,
            });
        }

        let required_effects = if compiled.contract.expects_mutation
            && compiled.contract.required_mutation_effects.is_empty()
        {
            ToolMutationEffects::UNSPECIFIED
        } else {
            compiled.contract.required_mutation_effects
        };
        for effect_name in required_effects.protocol_names() {
            let Some(effect) = ToolMutationEffects::from_protocol_name(effect_name) else {
                continue;
            };
            self.insert_obligation(RunObligation {
                id: format!("task:{}/obligation:mutation:{effect_name}", self.task_id),
                class: RunObligationClass::Achieve,
                state: RunObligationState::Pending,
                receipt: None,
                evidence_requirement: None,
                required_effect: effect,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
                required_target: None,
                observation_targets: Vec::new(),
                summary: None,
                step: None,
                prerequisite: false,
            });
        }

        if compiled.contract.requires_observation
            && compiled.contract.evidence_requirements.is_empty()
        {
            self.insert_obligation(RunObligation {
                id: format!("task:{}/obligation:verification", self.task_id),
                class: RunObligationClass::Observe,
                // The semantic contract lost or omitted the proposition that
                // evidence must prove. Any-observation fallback would allow a
                // single unrelated read to close an arbitrary multi-fact
                // request. Fail closed at the persisted lifecycle boundary.
                state: RunObligationState::Unverifiable,
                receipt: None,
                evidence_requirement: None,
                required_effect: ToolMutationEffects::NONE,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
                required_target: None,
                observation_targets: Vec::new(),
                summary: None,
                step: None,
                prerequisite: false,
            });
        }

        if self.response_contract.is_some() {
            self.insert_obligation(RunObligation {
                id: format!("task:{}/obligation:deliver:response", self.task_id),
                class: RunObligationClass::Deliver,
                state: RunObligationState::Pending,
                receipt: None,
                evidence_requirement: None,
                required_effect: ToolMutationEffects::NONE,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
                required_target: None,
                observation_targets: Vec::new(),
                summary: None,
                step: None,
                prerequisite: false,
            });
        }
    }

    /// Compile the executing model's typed checklist into obligations. Each
    /// typed item becomes one obligation keyed by its index: a mutation item
    /// is `Achieve` (closed by a succeeded receipt carrying the effect); an
    /// observation/targeted item is `Observe` (closed by a compatible exact
    /// subject/facet receipt). Opaque identifiers are exact resource IDs; they
    /// never fall back to an untargeted observation.
    /// Untyped free-text items are not obligations. A later declaration
    /// replaces the set: kept indices retain their proof; `deferred`/`skipped`/
    /// `failed` items and removed indices that are not already proven become
    /// `Deferred`. Editing the checklist never turns unperformed work into
    /// accomplished work: only receipts satisfy an obligation.
    fn install_executor_expectations(&mut self, declared: ExecutorExpectationsDeclaredData) {
        self.executor_expectations_present = true;
        self.executor_claimed_outcome = declared.outcome;
        self.executor_short_of = declared
            .outcome
            .filter(|outcome| outcome.unfinished())
            .and(declared.short_of);
        let exact_resource_targets = declared.schema_version >= 2;
        let depended_on: BTreeSet<usize> = declared
            .items
            .iter()
            .flat_map(|item| item.depends_on.iter().copied())
            .collect();
        let mut declared_ids = BTreeSet::new();
        for item in &declared.items {
            let id = format!("task:{}/obligation:checklist:{}", self.task_id, item.index);
            // `failed`: tried and not achieved. Not demanded again, never done.
            // `background`: handed to a job that reports back; this run can
            // neither demand nor prove it, so it is not done here.
            let deferred = matches!(
                item.status.as_str(),
                "deferred" | "skipped" | "failed" | "background"
            );
            let has_effect = !item.mutation_effects.is_empty();
            let declared_observation_targets = if item.observation_targets.is_empty() {
                item.targets
                    .iter()
                    .filter_map(|target| {
                        let parsed = RequestObservationTarget::from_legacy_exact(target)?;
                        (exact_resource_targets
                            || parsed.subject.kind != RequestVerificationTargetKind::ResourceId)
                            .then_some(parsed)
                    })
                    .collect::<Vec<_>>()
            } else {
                item.observation_targets.clone()
            };
            let required_target = declared_observation_targets
                .first()
                .map(|target| target.subject.clone());
            let (class, required_effect) = if has_effect {
                (RunObligationClass::Achieve, item.mutation_effects)
            } else if item.requires_observation || !declared_observation_targets.is_empty() {
                (RunObligationClass::Observe, ToolMutationEffects::NONE)
            } else {
                // Free text without typed content is a note, not an obligation.
                continue;
            };
            let observation_targets = if class == RunObligationClass::Observe {
                declared_observation_targets
            } else {
                Vec::new()
            };
            declared_ids.insert(id.clone());
            let existing = self.obligations.get(&id);
            // An index is stable presentation identity, not evidence identity.
            // Preserve proof only when the complete typed obligation is the
            // same; retargeting index 0 from goal A to goal B must reopen it.
            let same_contract = existing.is_some_and(|obligation| {
                obligation.class == class
                    && obligation.required_effect == required_effect
                    && obligation.required_target == required_target
                    && obligation.observation_targets == observation_targets
            });
            // Naming where already-proven work lives narrows the item; it is
            // not a different item. The executor adds an item's target when
            // it ticks the item, after the receipts that proved it, and that
            // edit must not turn accomplished work into unperformed work.
            // The proof is dropped only when it contradicts the new target:
            // its receipts name subjects of that kind and none is the target.
            let same_contract = same_contract
                || existing.is_some_and(|obligation| {
                    class == RunObligationClass::Achieve
                        && obligation.class == class
                        && obligation.required_effect == required_effect
                        && obligation.required_target.is_none()
                        && obligation.state == RunObligationState::Satisfied
                        && required_target.as_ref().is_none_or(|target| {
                            let named: Vec<&ToolTargetHint> = obligation
                                .satisfying_receipt_ids
                                .iter()
                                .filter_map(|id| self.operations.get(id))
                                .flat_map(|operation| operation.targets.iter())
                                .filter(|hint| hint_is_of_target_kind(target.kind, hint.kind))
                                .collect();
                            named.is_empty()
                                || named
                                    .iter()
                                    .any(|hint| hint_matches_target(&self.paths, target, hint))
                        })
                });
            let existing_proof = existing
                .filter(|_| same_contract)
                .map(|o| o.satisfying_receipt_ids.clone())
                .unwrap_or_default();
            let existing_revision = existing
                .filter(|_| same_contract)
                .and_then(|o| o.satisfied_at_revision);
            // Proof wins over the label in both directions: a proven item
            // stays satisfied even when re-posted as deferred, and a deferred
            // label never stands in for proof.
            let state = if same_contract
                && existing.is_some_and(|o| o.state == RunObligationState::Satisfied)
            {
                RunObligationState::Satisfied
            } else if deferred {
                RunObligationState::Deferred
            } else {
                RunObligationState::Pending
            };
            self.insert_obligation(RunObligation {
                id,
                class,
                state,
                receipt: None,
                evidence_requirement: None,
                required_effect,
                satisfied_at_revision: existing_revision,
                satisfying_receipt_ids: existing_proof,
                summary: Some(executor_item_summary(
                    item,
                    class,
                    required_effect,
                    required_target.as_ref(),
                )),
                step: Some(item.description.clone()),
                required_target,
                observation_targets,
                prerequisite: class == RunObligationClass::Observe
                    && depended_on.contains(&item.index),
            });
        }
        for (id, obligation) in self.obligations.iter_mut() {
            if id.contains("/obligation:checklist:")
                && !declared_ids.contains(id)
                && obligation.state != RunObligationState::Satisfied
            {
                obligation.state = RunObligationState::Deferred;
            }
        }
        // The executor's natural flow is do-the-work-then-tick-the-item, so
        // an item's proof must not depend on whether its receipt landed
        // before or after the declaration that named its targets.
        self.recredit_checklist_obligations_from_recorded_receipts();
        self.close_remote_effect_obligations_on_verified_remote_observation();
    }

    /// Verify still-pending checklist obligations against every receipt the
    /// run has recorded, whether it landed before or after the item (or its
    /// exact targets) was declared. Receipts remain the only thing that can
    /// close an obligation.
    ///
    /// An `Achieve` item is proven by the receipts together, the way the
    /// work is done: its effects may be spread over several succeeded
    /// receipts on its target, and a launch receipt proves the process-state
    /// change it declared while that process is still running. Work whose
    /// receipts cannot name the item's target (a local process behind a
    /// public URL) is proven by the effect plus a later completed
    /// observation of that target.
    fn recredit_checklist_obligations_from_recorded_receipts(&mut self) {
        struct RecordedReceipt {
            operation_id: String,
            succeeded: bool,
            completed_observation: bool,
            dispatched: bool,
            mutation_effects: crate::traits::ToolMutationEffects,
            /// What this receipt proves happened: all its effects once it
            /// succeeded, only the process-state change while in flight.
            proven_effects: crate::traits::ToolMutationEffects,
            observes: bool,
            targets: Vec<ToolTargetHint>,
            observations: Vec<ToolObservationEvidence>,
            collection_observations: Vec<ToolCollectionObservation>,
            revision: u64,
            sequence: u64,
        }
        let mut receipts: Vec<RecordedReceipt> = self
            .operations
            .values()
            .filter(|operation| operation.result_id.is_some())
            .map(|operation| {
                let terminal = operation.has_terminal_receipt();
                let succeeded =
                    terminal && matches!(operation.outcome, Some(ToolOutcomeStatus::Succeeded));
                let proven_effects = if !operation.dispatched {
                    crate::traits::ToolMutationEffects::NONE
                } else if succeeded {
                    operation.mutation_effects
                } else if operation.is_background_in_flight() {
                    operation
                        .mutation_effects
                        .intersect(crate::traits::ToolMutationEffects::PROCESS_STATE)
                } else {
                    crate::traits::ToolMutationEffects::NONE
                };
                RecordedReceipt {
                    operation_id: operation.operation_id.clone(),
                    succeeded,
                    completed_observation: terminal
                        && matches!(
                            operation.outcome,
                            Some(
                                ToolOutcomeStatus::Succeeded
                                    | ToolOutcomeStatus::CompletedWithNegativeResult
                            )
                        ),
                    dispatched: operation.dispatched,
                    mutation_effects: operation.mutation_effects,
                    proven_effects,
                    observes: operation.observes,
                    targets: operation.targets.clone(),
                    observations: operation.observations.clone(),
                    collection_observations: operation.collection_observations.clone(),
                    revision: operation.result_revision.unwrap_or(self.effect_revision),
                    sequence: operation.receipt_sequence.unwrap_or(0),
                }
            })
            .collect();
        if receipts.is_empty() {
            return;
        }
        receipts.sort_by_key(|receipt| receipt.sequence);
        for (id, obligation) in self.obligations.iter_mut() {
            if !id.contains("/obligation:checklist:")
                || !matches!(
                    obligation.state,
                    RunObligationState::Pending | RunObligationState::Deferred
                )
                || obligation.receipt.is_some()
            {
                continue;
            }
            if obligation.class == RunObligationClass::Achieve {
                let required = obligation.required_effect;
                if required.is_empty() {
                    continue;
                }
                let target = obligation.required_target.as_ref();
                // The earliest receipts that together carry the effect. A
                // receipt counts only when it adds to what is proven so far.
                let covering = |on_target: bool| {
                    let mut proven = crate::traits::ToolMutationEffects::NONE;
                    let mut proof = Vec::new();
                    for receipt in &receipts {
                        if receipt.proven_effects.is_empty()
                            || (on_target
                                && target.is_some_and(|target| {
                                    !hints_touch_target(&self.paths, target, &receipt.targets)
                                }))
                        {
                            continue;
                        }
                        let widened = proven.union(receipt.proven_effects);
                        // Judged per required effect through `satisfies`, so
                        // a more specific receipt (a source-file write for a
                        // workspace write) counts as the progress it is.
                        let adds = (0..u32::BITS)
                            .map(|bit| {
                                crate::traits::ToolMutationEffects::from_bits(
                                    required.bits() & (1 << bit),
                                )
                            })
                            .any(|part| {
                                !part.is_empty()
                                    && widened.satisfies(part)
                                    && !proven.satisfies(part)
                            });
                        if !adds {
                            continue;
                        }
                        proven = widened;
                        proof.push(receipt);
                        if proven.satisfies(required) {
                            return Some(proof);
                        }
                    }
                    None
                };
                let proof = covering(true).or_else(|| {
                    let target = target?;
                    let mut proof = covering(false)?;
                    let done_at = proof.iter().map(|receipt| receipt.sequence).max()?;
                    let observed = receipts.iter().find(|receipt| {
                        receipt.sequence > done_at
                            && receipt.completed_observation
                            && receipt.dispatched
                            && receipt.observes
                            && hints_touch_target(&self.paths, target, &receipt.targets)
                    })?;
                    proof.push(observed);
                    Some(proof)
                });
                let Some(proof) = proof else {
                    continue;
                };
                for receipt in &proof {
                    if !obligation
                        .satisfying_receipt_ids
                        .contains(&receipt.operation_id)
                    {
                        obligation
                            .satisfying_receipt_ids
                            .push(receipt.operation_id.clone());
                    }
                }
                obligation.state = RunObligationState::Satisfied;
                obligation.satisfied_at_revision =
                    proof.iter().map(|receipt| receipt.revision).max();
                continue;
            }
            if obligation.class != RunObligationClass::Observe {
                continue;
            }
            // Mirror the live staleness rule: an observation credited at an
            // older revision is stale only when a later mutation could have
            // changed the observed state.
            let fresh = |receipt: &RecordedReceipt| {
                obligation.prerequisite
                    || !receipts.iter().any(|later| {
                        later.succeeded
                            && !later.mutation_effects.is_empty()
                            && later.revision > receipt.revision
                            && obligation_touched_by_targets(
                                &self.paths,
                                obligation,
                                &later.targets,
                            )
                    })
            };
            let observes = |receipt: &RecordedReceipt| {
                receipt.completed_observation && receipt.dispatched && receipt.observes
            };
            // Each target needs one fresh observation that supports it; one
            // receipt may cover several, and an item naming two files is
            // observed by reading each, not only by one call touching both.
            let proof: Option<Vec<&RecordedReceipt>> = if obligation.observation_targets.is_empty()
            {
                receipts
                    .iter()
                    .find(|receipt| {
                        observes(receipt)
                            && observation_targets_supported(
                                &self.paths,
                                &[],
                                &receipt.targets,
                                &receipt.observations,
                                &receipt.collection_observations,
                            )
                            && fresh(receipt)
                    })
                    .map(|receipt| vec![receipt])
            } else {
                obligation
                    .observation_targets
                    .iter()
                    .map(|target| {
                        receipts.iter().find(|receipt| {
                            observes(receipt)
                                && observation_target_supported(
                                    &self.paths,
                                    target,
                                    &receipt.targets,
                                    &receipt.observations,
                                    &receipt.collection_observations,
                                )
                                && fresh(receipt)
                        })
                    })
                    .collect()
            };
            let Some(proof) = proof else {
                continue;
            };
            for receipt in &proof {
                if !obligation
                    .satisfying_receipt_ids
                    .contains(&receipt.operation_id)
                {
                    obligation
                        .satisfying_receipt_ids
                        .push(receipt.operation_id.clone());
                }
            }
            obligation.state = RunObligationState::Satisfied;
            // The oldest observation bounds it: a later mutation of what any
            // of them saw invalidates the item.
            obligation.satisfied_at_revision = proof.iter().map(|receipt| receipt.revision).min();
        }
    }

    /// A shell adapter deliberately never carries remote effects in its own
    /// receipt (it cannot masquerade as a remote mutation), yet a
    /// repository-documented deploy IS a shell command. Prove the remote
    /// effect the way an operator proves it: a succeeded mutation happened,
    /// and an independently declared URL observation verified the remote
    /// state afterward. The adapter's self-claim alone still proves nothing;
    /// without a post-mutation URL verification these obligations stay open.
    fn close_remote_effect_obligations_on_verified_remote_observation(&mut self) {
        const REMOTE: crate::traits::ToolMutationEffects =
            crate::traits::ToolMutationEffects::REMOTE_DEPLOY
                .union(crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY);
        let Some(observation) = self
            .obligations
            .values()
            .filter(|obligation| {
                obligation.class == RunObligationClass::Observe
                    && obligation.state == RunObligationState::Satisfied
                    && obligation.required_target.as_ref().is_some_and(|target| {
                        target.kind == crate::traits::RequestVerificationTargetKind::Url
                    })
                    && obligation
                        .satisfied_at_revision
                        .is_some_and(|revision| revision >= 1)
            })
            .max_by_key(|obligation| obligation.satisfied_at_revision)
        else {
            return;
        };
        let revision = observation.satisfied_at_revision;
        let proof = observation.satisfying_receipt_ids.clone();
        for obligation in self.obligations.values_mut() {
            if obligation.state == RunObligationState::Pending
                && obligation.class == RunObligationClass::Achieve
                && obligation.receipt.is_none()
                && !obligation.required_effect.is_empty()
                && REMOTE.contains(obligation.required_effect)
            {
                obligation.state = RunObligationState::Satisfied;
                obligation.satisfied_at_revision = revision;
                for id in &proof {
                    if !obligation.satisfying_receipt_ids.contains(id) {
                        obligation.satisfying_receipt_ids.push(id.clone());
                    }
                }
            }
        }
    }

    /// Whether any typed expectations exist to arbitrate: from the compiled
    /// contract, the executor's own declaration, or both.
    pub fn expectations_present(&self) -> bool {
        self.contract_present || self.executor_expectations_present
    }

    /// Executor-declared obligations not yet proven, deferred ones included.
    /// These are the model's own statement of unmet work; a run that ends
    /// with any is incomplete, not "closed by evidence".
    /// The executing model said the request is not fully done (`partial`
    /// or `blocked`). Its own verdict: no receipt upgrades it to a success.
    pub fn executor_claimed_unfinished(&self) -> bool {
        self.executor_claimed_outcome
            .is_some_and(super::ExecutorClaimedOutcome::unfinished)
    }

    /// The executing model declared work (a checklist item that changes
    /// something) and the run attempted part of it or none of it: these are
    /// the declared items, in order, that nothing took on. Judged per item,
    /// from receipts only, never from the reply:
    ///
    /// - an item a receipt proved, or one a call named in its typed
    ///   `checklist_items` (a launch handing it to a job, or a call refused
    ///   by policy or declined by the owner), was attempted;
    /// - every other call that changed something, was refused or declined,
    ///   or left a job running without naming its items took on one item at
    ///   most: the items that remain beyond those calls were never started.
    ///
    /// A resource the model reported running out of is a typed reason and
    /// holds nothing. Nothing else stood between such an item and the work,
    /// so a deferral, a handoff or an unfinished verdict is no reason to
    /// leave it: it is carried out, or the owner is asked
    /// ([`super::ExecutorClaimedOutcome::NeedsOwner`]). Live 2026-11-13: two
    /// installs declared, one helper launched for the first, and the turn
    /// handed off with the second never started and no reason given.
    pub fn unattempted_declared_work(&self) -> Vec<String> {
        if self.executor_short_of.is_some() {
            return Vec::new();
        }
        let attempted = |operation: &&RunOperation| {
            operation.mutating
                || operation.policy_denied
                || operation.owner_declined
                || operation.is_background_in_flight()
        };
        let named: BTreeSet<usize> = self
            .operations
            .values()
            .filter(|operation| {
                operation.dispatched || operation.policy_denied || operation.owner_declined
            })
            .flat_map(|operation| operation.checklist_items.iter().copied())
            .collect();
        let unnamed_attempts = self
            .operations
            .values()
            .filter(|operation| operation.checklist_items.is_empty())
            .filter(attempted)
            .count();
        let mut open: Vec<(usize, String)> = self
            .obligations
            .iter()
            .filter(|(_, obligation)| {
                obligation.class == RunObligationClass::Achieve
                    && obligation.state != RunObligationState::Satisfied
            })
            .filter_map(|(id, _)| {
                let index: usize = id
                    .rsplit_once("/obligation:checklist:")
                    .and_then(|(_, index)| index.parse().ok())?;
                (!named.contains(&index)).then(|| (index, id.clone()))
            })
            .collect();
        open.sort_by_key(|(index, _)| *index);
        open.into_iter()
            .skip(unnamed_attempts)
            .map(|(_, id)| id)
            .collect()
    }

    /// Record a launch whose receipt is being written now: the job it left
    /// running took on `checklist_items` (none named: one item at most).
    /// Lets a handoff be judged with the launch it makes, before its
    /// receipt reaches the ledger.
    pub fn with_pending_launch(mut self, operation_id: &str, checklist_items: &[usize]) -> Self {
        let operation = self
            .operations
            .entry(operation_id.to_string())
            .or_insert_with(|| RunOperation {
                operation_id: operation_id.to_string(),
                tool_name: String::new(),
                stable_operation_key: None,
                obligation_ids: Vec::new(),
                max_attempts: None,
                max_invocations: None,
                idempotency_key: None,
                outcome: None,
                dispatched: false,
                policy_denied: false,
                owner_declined: false,
                mutating: false,
                evidence_capabilities: Vec::new(),
                targets: Vec::new(),
                observations: Vec::new(),
                collection_observations: Vec::new(),
                mutation_effects: crate::traits::ToolMutationEffects::NONE,
                observes: false,
                result_revision: None,
                receipt_sequence: None,
                result_id: None,
                operation_lineage: None,
                checklist_items: Vec::new(),
            });
        operation.dispatched = true;
        operation.outcome = Some(ToolOutcomeStatus::Backgrounded);
        for item in checklist_items {
            if !operation.checklist_items.contains(item) {
                operation.checklist_items.push(*item);
            }
        }
        self
    }

    /// The checklist indices of [`Self::unattempted_declared_work`].
    pub fn unattempted_checklist_items(&self) -> BTreeSet<usize> {
        self.unattempted_declared_work()
            .iter()
            .filter_map(|id| {
                id.rsplit_once("/obligation:checklist:")
                    .and_then(|(_, index)| index.parse().ok())
            })
            .collect()
    }

    /// Whether any declared work was never attempted
    /// ([`Self::unattempted_declared_work`]).
    pub fn declared_work_unattempted(&self) -> bool {
        !self.unattempted_declared_work().is_empty()
    }

    pub fn open_executor_expectations(&self) -> usize {
        self.obligations
            .iter()
            .filter(|(id, obligation)| {
                id.contains("/obligation:checklist:")
                    && obligation.state != RunObligationState::Satisfied
            })
            .count()
    }

    /// One line per obligation for a "still expected" directive: its
    /// summary, its evidence requirement, the tools that would satisfy it,
    /// or its id.
    pub fn expectation_summaries(&self, obligation_ids: &[String]) -> Vec<String> {
        obligation_ids
            .iter()
            .filter_map(|id| self.obligations.get(id))
            .map(|obligation| {
                if let Some(summary) = obligation.summary.as_ref() {
                    summary.clone()
                } else if let Some(requirement) = obligation.evidence_requirement.as_ref() {
                    requirement.summary.clone()
                } else if let Some(predicate) = obligation.receipt.as_ref() {
                    format!("invoke {}", predicate.tool_names.join("/"))
                } else {
                    obligation.id.clone()
                }
            })
            .collect()
    }

    fn insert_obligation(&mut self, obligation: RunObligation) {
        self.obligations.insert(obligation.id.clone(), obligation);
    }

    fn record_call(&mut self, mut call: ToolCallData) {
        if call
            .task_id
            .as_deref()
            .is_some_and(|task_id| task_id != self.task_id)
        {
            self.record_invariant("tool_call_task_identity_mismatch");
            return;
        }
        if let Some(existing) = self.operations.get(&call.tool_call_id) {
            if existing.tool_name != call.name
                || existing.stable_operation_key != call.stable_operation_key
                || existing.obligation_ids != call.obligation_ids
            {
                self.record_invariant("conflicting_operation_claim");
            }
            return;
        }
        if matches!(
            call.operation_lineage,
            Some(super::ToolOperationLineage::DurableReplay { .. })
        ) {
            call.obligation_ids = self.resolve_adopted_obligation_ids(&call.obligation_ids);
        }
        call.obligation_ids = self.effective_obligation_ids(&call.obligation_ids);
        self.operations.insert(
            call.tool_call_id.clone(),
            RunOperation {
                operation_id: call.tool_call_id,
                tool_name: call.name,
                stable_operation_key: call.stable_operation_key,
                obligation_ids: call.obligation_ids,
                max_attempts: call.max_operation_attempts,
                max_invocations: call.max_operation_invocations,
                idempotency_key: call.idempotency_key,
                outcome: None,
                dispatched: false,
                policy_denied: false,
                owner_declined: false,
                mutating: false,
                evidence_capabilities: Vec::new(),
                targets: Vec::new(),
                observations: Vec::new(),
                collection_observations: Vec::new(),
                mutation_effects: crate::traits::ToolMutationEffects::NONE,
                observes: false,
                result_revision: None,
                receipt_sequence: None,
                result_id: None,
                operation_lineage: call.operation_lineage,
                checklist_items: Vec::new(),
            },
        );
    }

    /// Bind a proposal to the currently open subset of its eligible proof
    /// obligations. One call may be structurally compatible with several
    /// predicates, but a predicate that was already satisfied by an earlier
    /// receipt must not consume or veto a later operation needed by another
    /// still-open predicate. If every eligible obligation is already closed,
    /// retain the original set so an extra proposal is still checked against
    /// the user's maximum cardinality instead of escaping the contract.
    pub fn effective_operation_claim(
        &self,
        claim: &TaskKernelOperationClaim,
    ) -> TaskKernelOperationClaim {
        let mut effective = claim.clone();
        effective.obligation_ids = self.effective_obligation_ids(&claim.obligation_ids);
        if effective.obligation_ids != claim.obligation_ids
            && let Some(active_ceiling) = effective
                .obligation_ids
                .iter()
                .filter_map(|id| self.obligations.get(id))
                .filter_map(|obligation| obligation.receipt.as_ref())
                .filter_map(|predicate| predicate.max_invocations)
                .min()
        {
            // The producer computed its operation ceiling from every
            // statically compatible predicate. Once a stricter sibling is
            // closed, its smaller ceiling no longer governs the remaining
            // proof. Re-derive the ceiling from open obligations.
            effective.max_invocations = active_ceiling.max(1);
            effective.max_attempts = effective.max_attempts.max(effective.max_invocations);
        }
        effective
    }

    fn effective_obligation_ids(&self, eligible: &[String]) -> Vec<String> {
        let open = eligible
            .iter()
            .filter(|id| {
                self.obligations.get(*id).is_some_and(|obligation| {
                    matches!(
                        obligation.state,
                        RunObligationState::Pending | RunObligationState::Invalidated
                    )
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        if open.is_empty() {
            eligible.to_vec()
        } else {
            open
        }
    }

    /// Map obligation ids named by an adopted durable replay onto this run's
    /// own obligations. Only ids bound by the compiled contract translate;
    /// anything else is kept verbatim so an unrelated claim still cannot
    /// prove a local obligation.
    fn resolve_adopted_obligation_ids(&self, ids: &[String]) -> Vec<String> {
        let mut resolved = Vec::with_capacity(ids.len());
        for id in ids {
            let local = self
                .adopted_obligation_aliases
                .get(id)
                .cloned()
                .unwrap_or_else(|| id.clone());
            if !resolved.contains(&local) {
                resolved.push(local);
            }
        }
        resolved
    }

    fn record_result(&mut self, result: ToolResultData) {
        let terminal_before = self.terminal_decision();
        let operation_id = result.tool_call_id.clone();
        if result
            .task_id
            .as_deref()
            .is_some_and(|task_id| task_id != self.task_id)
        {
            self.record_invariant("tool_result_task_identity_mismatch");
            return;
        }
        let Some(receipt) = result.receipt.as_ref() else {
            self.record_invariant("typed_receipt_missing");
            return;
        };
        let result_id = receipt
            .result_provenance
            .result_id
            .clone()
            .unwrap_or_else(|| format!("receipt:{}", result.tool_call_id));
        if let Some(operation) = self.operations.get(&result.tool_call_id) {
            let same_result = operation.result_id.as_deref() == Some(result_id.as_str());
            if operation.is_background_in_flight() {
                if receipt.outcome_status == ToolOutcomeStatus::Backgrounded {
                    // Replayed or refreshed handoff telemetry describes the
                    // same nonterminal transition, regardless of whether an
                    // older adapter assigned it a fresh provenance ID.
                    return;
                }
                // The background receipt is provisional. Exactly one later
                // terminal receipt advances this same invocation; it does not
                // create a second operation or consume more cardinality. A
                // legacy receipt may reuse the fallback result ID, so outcome
                // state, not provenance inequality, authorizes the transition.
            } else if same_result {
                return;
            } else if operation.result_id.is_some() {
                self.record_invariant("operation_has_multiple_terminal_receipts");
                return;
            }
        }
        let operation = self
            .operations
            .entry(result.tool_call_id.clone())
            .or_insert(RunOperation {
                operation_id: result.tool_call_id.clone(),
                tool_name: result.name.clone(),
                stable_operation_key: None,
                obligation_ids: Vec::new(),
                max_attempts: None,
                max_invocations: None,
                idempotency_key: receipt.idempotency_key.clone(),
                outcome: None,
                dispatched: false,
                policy_denied: false,
                owner_declined: false,
                mutating: false,
                evidence_capabilities: Vec::new(),
                targets: Vec::new(),
                observations: Vec::new(),
                collection_observations: Vec::new(),
                mutation_effects: crate::traits::ToolMutationEffects::NONE,
                observes: false,
                result_revision: None,
                receipt_sequence: None,
                result_id: None,
                operation_lineage: None,
                checklist_items: Vec::new(),
            });
        operation.outcome = Some(receipt.outcome_status);
        for item in &receipt.checklist_items {
            if !operation.checklist_items.contains(item) {
                operation.checklist_items.push(*item);
            }
        }
        operation.dispatched = operation.dispatched || receipt.invocation_stage.reached_dispatch();
        operation.policy_denied =
            operation.policy_denied || (!operation.dispatched && receipt.access_denial.is_some());
        operation.owner_declined = operation.owner_declined
            || receipt
                .access_denial
                .as_ref()
                .is_some_and(|denial| denial.reason_code == OWNER_APPROVAL_DECLINED);
        operation.mutating = operation.mutating || receipt.semantics.mutates_state();
        if operation.evidence_capabilities.is_empty() {
            operation.evidence_capabilities = receipt.semantics.evidence.clone();
        }
        if operation.evidence_capabilities.is_empty()
            && let Some(denial) = receipt.access_denial.as_ref()
        {
            operation.evidence_capabilities = denial.proposed_evidence.clone();
        }
        if operation.targets.is_empty() {
            operation.targets = receipt_targets(receipt);
        }
        operation.observations = receipt.observations.clone();
        operation.collection_observations = receipt.collection_observations.clone();
        operation.mutation_effects = operation
            .mutation_effects
            .union(receipt.semantics.mutation_effects);
        operation.observes = operation.observes || receipt.semantics.observes_state();
        operation.result_id = Some(result_id.clone());
        self.receipt_sequence = self.receipt_sequence.saturating_add(1);
        operation.receipt_sequence = Some(self.receipt_sequence);
        let claimed_obligation_ids = operation.obligation_ids.clone();
        // Cardinality is about distinct durable invocations, not distinct
        // output bytes. Two real calls may legitimately return identical (or
        // empty) content. A durable replay reuses its source invocation's proof
        // identity so replay remains idempotent and consumes no new cardinality.
        let proof_receipt_id = match operation.operation_lineage.as_ref() {
            Some(super::ToolOperationLineage::DurableReplay {
                source_operation_id,
                ..
            }) => source_operation_id.clone(),
            _ => result.tool_call_id.clone(),
        };
        let is_durable_replay = matches!(
            operation.operation_lineage.as_ref(),
            Some(super::ToolOperationLineage::DurableReplay { .. })
        );
        if operation.dispatched
            && !self.allowed_tool_names.is_empty()
            && !self.allowed_tool_names.contains(&operation.tool_name)
        {
            self.record_invariant("dispatched_operation_outside_allowed_tool_set");
        }

        // `Backgrounded` means the adapter dispatched one invocation and
        // handed its still-running lifecycle to a durable notifier. Retain the
        // provisional receipt for replay/idempotence and cardinality, but do
        // not credit completion evidence, advance effect revision, or exhaust
        // the run while its terminal result is outstanding. The one thing a
        // launch does prove is the process-state change it declared, which
        // the checklist re-credit reads from the recorded operation.
        if receipt.outcome_status == ToolOutcomeStatus::Backgrounded {
            self.recredit_checklist_obligations_from_recorded_receipts();
            self.reconcile_cardinality();
            // The call proposal may have looked cardinality-exhausted before
            // its first receipt arrived. The handoff reopens it as in-flight,
            // so this operation is the current causal frontier regardless of
            // that pre-receipt projection.
            self.primary_causal_operation_id = Some(operation_id);
            return;
        }

        let completed_mutation = !is_durable_replay
            && result.succeeded()
            && receipt.invocation_stage.reached_dispatch()
            && receipt.semantics.mutates_state()
            && !delivers_own_capture(
                receipt.semantics.observes_state(),
                receipt.semantics.mutation_effects,
                &receipt_targets(receipt),
            );
        if completed_mutation {
            self.effect_revision = self.effect_revision.saturating_add(1);
        }
        if let Some(operation) = self.operations.get_mut(&operation_id) {
            operation.result_revision = Some(self.effect_revision);
        }
        if completed_mutation {
            // A mutation only invalidates observations of state it could have
            // changed: an earlier research read of an external page does not
            // become stale because a local post file was written afterward.
            // Target-bound observations survive mutations whose receipt
            // touches disjoint targets; untargeted observations remain
            // conservatively revision-bound.
            let mutation_targets = receipt_targets(receipt);
            for obligation in self.obligations.values_mut() {
                if obligation.class == RunObligationClass::Observe
                    && !obligation.prerequisite
                    && obligation.state == RunObligationState::Satisfied
                    && obligation
                        .satisfied_at_revision
                        .is_some_and(|revision| revision < self.effect_revision)
                    && obligation_touched_by_targets(&self.paths, obligation, &mutation_targets)
                {
                    obligation.state = RunObligationState::Invalidated;
                }
            }
        }

        let explicit_ids = if is_durable_replay {
            self.resolve_adopted_obligation_ids(&receipt.completion_obligation_ids)
        } else {
            receipt.completion_obligation_ids.clone()
        }
        .into_iter()
        .collect::<BTreeSet<_>>();
        for obligation in self.obligations.values_mut() {
            if obligation.state == RunObligationState::Unverifiable {
                continue;
            }
            let observation_compatible =
                obligation
                    .evidence_requirement
                    .as_ref()
                    .is_none_or(|requirement| {
                        evidence_receipt_supports(&self.paths, requirement, receipt)
                    });
            let target_compatible = obligation
                .required_target
                .as_ref()
                .is_none_or(|target| receipt_touches_target(&self.paths, target, receipt));
            let observation_selector_compatible = observation_targets_supported(
                &self.paths,
                &obligation.observation_targets,
                &receipt_targets(receipt),
                &receipt.observations,
                &receipt.collection_observations,
            );
            let explicitly_proven = explicit_ids.contains(&obligation.id)
                && (obligation.class != RunObligationClass::Observe
                    || (observation_compatible && observation_selector_compatible));
            let claim_allows_proof = claimed_obligation_ids.is_empty()
                || claimed_obligation_ids.contains(&obligation.id);
            let predicate_proven = claim_allows_proof
                && obligation
                    .receipt
                    .as_ref()
                    .is_some_and(|predicate| receipt_matches_predicate(&result, predicate));
            let effect_proven = obligation.class == RunObligationClass::Achieve
                && !obligation.required_effect.is_empty()
                && target_compatible
                && result.succeeded()
                && receipt
                    .semantics
                    .mutation_effects
                    .satisfies(obligation.required_effect);
            let generic_observation = obligation.class == RunObligationClass::Observe
                && obligation.receipt.is_none()
                && result.completed_observation()
                && receipt.invocation_stage.reached_dispatch()
                && receipt.semantics.observes_state()
                && observation_compatible
                && observation_selector_compatible;
            // Receipt-bound obligations are replayed from their predicate,
            // never trusted from an upstream completion-ID annotation. This
            // keeps the reducer authoritative if an intermediate matcher is
            // stale or overly broad.
            let proven = if obligation.receipt.is_some() {
                predicate_proven
            } else {
                explicitly_proven || effect_proven || generic_observation
            };
            if proven {
                if !obligation
                    .satisfying_receipt_ids
                    .contains(&proof_receipt_id)
                {
                    obligation
                        .satisfying_receipt_ids
                        .push(proof_receipt_id.clone());
                }
                let minimum = obligation
                    .receipt
                    .as_ref()
                    .and_then(|predicate| predicate.min_invocations)
                    .unwrap_or(1);
                if obligation.satisfying_receipt_ids.len() >= minimum {
                    obligation.state = RunObligationState::Satisfied;
                    obligation.satisfied_at_revision = Some(self.effect_revision);
                }
            }
        }
        // One receipt rarely carries a whole checklist item; let the recorded
        // receipts prove it together.
        self.recredit_checklist_obligations_from_recorded_receipts();
        self.close_remote_effect_obligations_on_verified_remote_observation();
        self.reconcile_cardinality();
        // Keep the last operation that advanced an open aggregate. Once the
        // run is terminal, unrelated late telemetry cannot steal causality.
        if matches!(
            terminal_before,
            RunTerminalDecision::Pending | RunTerminalDecision::Unspecified
        ) {
            self.primary_causal_operation_id = Some(operation_id);
        }
    }

    fn record_assistant_response(&mut self, response: AssistantResponseData) {
        if response
            .task_id
            .as_deref()
            .is_some_and(|task_id| task_id != self.task_id)
        {
            self.record_invariant("assistant_response_task_identity_mismatch");
            return;
        }
        if response
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
        {
            return;
        }
        if response.disposition == AssistantResponseDisposition::BackgroundHandoff {
            return;
        }
        // Any terminal assistant artifact is prepared output. The response
        // contract is a presentation hint owned by the semantic producer; it
        // never gates delivery proof. Delivery closes on any non-empty terminal
        // artifact once the execution work is proved, so a model-authored
        // answer grounded in receipts is never displaced by a byte-exact
        // projection requirement.
        self.prepared_response_id = response.message_id.clone();
        let non_empty = response
            .content
            .as_deref()
            .is_some_and(|content| !content.trim().is_empty());
        if !non_empty || !self.work_is_fulfilled() {
            return;
        }
        let proof_id = response
            .message_id
            .as_deref()
            .map(|id| format!("response:{id}"))
            .unwrap_or_else(|| "response:unidentified".to_string());
        for obligation in self
            .obligations
            .values_mut()
            .filter(|obligation| obligation.class == RunObligationClass::Deliver)
        {
            obligation.state = RunObligationState::Satisfied;
            obligation.satisfied_at_revision = Some(self.effect_revision);
            if !obligation.satisfying_receipt_ids.contains(&proof_id) {
                obligation.satisfying_receipt_ids.push(proof_id.clone());
            }
        }
    }

    fn record_delivery(&mut self, delivery: ResponseDeliveryData) {
        if delivery.task_id != self.task_id {
            self.record_invariant("response_delivery_task_identity_mismatch");
            return;
        }
        if self
            .prepared_response_id
            .as_deref()
            .is_some_and(|response_id| response_id != delivery.response_id)
        {
            self.record_invariant("response_delivery_identity_mismatch");
            return;
        }
        self.delivery_states
            .insert(delivery.response_id, delivery.state);
    }

    fn record_task_end(&mut self, end: TaskEndData) {
        if !end.task_id.is_empty() && end.task_id != self.task_id {
            self.record_invariant("task_end_identity_mismatch");
            return;
        }
        if self.recorded_task_status.is_some() {
            if self.recorded_task_status != Some(end.status)
                || self.recorded_task_outcome != Some(end.effective_outcome())
            {
                self.record_invariant("conflicting_task_end");
            }
            return;
        }
        self.recorded_task_status = Some(end.status);
        self.recorded_task_outcome = Some(end.effective_outcome());
    }

    fn reconcile_cardinality(&mut self) {
        self.cardinality_violations = 0;
        for obligation in self.obligations.values() {
            let Some(predicate) = obligation.receipt.as_ref() else {
                continue;
            };
            let Some(limit) = predicate.max_invocations else {
                continue;
            };
            let proposals = self
                .operations
                .values()
                .filter(|operation| {
                    if matches!(
                        operation.operation_lineage,
                        Some(super::ToolOperationLineage::DurableReplay { .. })
                    ) {
                        return false;
                    }
                    if !operation.obligation_ids.is_empty() {
                        operation.obligation_ids.contains(&obligation.id)
                    } else {
                        // Compatibility for events written before kernel-owned
                        // obligation bindings. New rows never infer ownership from
                        // overlapping tool predicates.
                        operation.result_id.is_some()
                            && (predicate.tool_names.is_empty()
                                || predicate.tool_names.contains(&operation.tool_name))
                    }
                })
                .count();
            self.cardinality_violations = self
                .cardinality_violations
                .saturating_add(proposals.saturating_sub(limit));
        }
    }

    fn record_invariant(&mut self, code: &str) {
        if !self.invariant_violations.iter().any(|value| value == code) {
            self.invariant_violations.push(code.to_string());
        }
    }

    /// Every obligation counts as required: a deferred item is unmet work,
    /// not a smaller denominator.
    pub fn required_count(&self) -> usize {
        self.obligations.len()
    }

    pub fn satisfied_count(&self) -> usize {
        self.obligations
            .values()
            .filter(|obligation| obligation.state == RunObligationState::Satisfied)
            .count()
    }

    pub fn is_fulfilled(&self) -> bool {
        self.expectations_present()
            && self.cardinality_violations == 0
            && self.invariant_violations.is_empty()
            && !self.obligations.is_empty()
            && self
                .obligations
                .values()
                .all(|obligation| obligation.state == RunObligationState::Satisfied)
    }

    /// Whether all execution/evidence work is proved and only presentation may
    /// remain. This is deliberately separate from terminal success: preparing
    /// the requested response artifact closes the `Deliver` obligation later.
    pub fn work_is_fulfilled(&self) -> bool {
        if !self.contract_present
            || self.cardinality_violations > 0
            || !self.invariant_violations.is_empty()
        {
            return false;
        }
        let mut work_obligations = self
            .obligations
            .values()
            .filter(|obligation| obligation.class != RunObligationClass::Deliver)
            .peekable();
        if work_obligations.peek().is_none() {
            // A response-only conversational request has no execution work.
            // Once any operation exists, however, absence of a corresponding
            // work obligation is not proof: never project success from a 0/0
            // contract merely because an untracked tool happened to return.
            return self.operations.is_empty();
        }
        work_obligations.all(|obligation| obligation.state == RunObligationState::Satisfied)
    }

    /// Deterministic successful response projection. A semantic producer owns
    /// the presentation choice; the reducer exposes the typed artifact only
    /// after non-delivery proof has closed.
    pub fn projected_success_response(&self) -> Option<&str> {
        self.work_is_fulfilled()
            .then(|| {
                self.response_contract
                    .as_ref()
                    .map(|contract| contract.success_text())
            })
            .flatten()
    }

    /// Whether the durable receipt set alone proves completed work: at least
    /// one operation reached a terminal receipt, no proposal is still dangling
    /// without one, and every terminal receipt either succeeded outright or
    /// was credited to an obligation (an expected negative result or denial).
    ///
    /// This deliberately ignores whether the compiled obligations were
    /// credited. A contract is an LLM-proposed description of the work; when
    /// it cannot describe work that demonstrably completed, the receipts win.
    /// Integrity invariants and explicit cardinality limits are not contract
    /// descriptions and still fail closed.
    /// Whether a pending obligation's bounded invocation budget is spent.
    fn obligation_exhausted(&self, obligation: &RunObligation) -> bool {
        if obligation.state == RunObligationState::Unverifiable {
            return true;
        }
        let Some(predicate) = obligation.receipt.as_ref() else {
            return false;
        };
        predicate.max_invocations.is_some_and(|limit| {
            self.operations
                .values()
                .filter(|operation| {
                    if matches!(
                        operation.operation_lineage,
                        Some(super::ToolOperationLineage::DurableReplay { .. })
                    ) {
                        return false;
                    }
                    if !operation.obligation_ids.is_empty() {
                        operation.obligation_ids.contains(&obligation.id)
                    } else {
                        operation.result_id.is_some()
                            && (predicate.tool_names.is_empty()
                                || predicate.tool_names.contains(&operation.tool_name))
                    }
                })
                .count()
                >= limit
        })
    }

    /// Whether an operation bound to this obligation was refused before
    /// dispatch by a typed policy boundary. Retrying cannot cross it.
    fn obligation_policy_denied(&self, obligation: &RunObligation) -> bool {
        self.operations.values().any(|operation| {
            if !operation.policy_denied {
                return false;
            }
            // A denied mutation attempt blocks every effect obligation: the
            // authority boundary that refused it does not move on retry.
            if obligation.class == RunObligationClass::Achieve
                && !obligation.required_effect.is_empty()
                && operation.mutating
            {
                return true;
            }
            if !operation.obligation_ids.is_empty() {
                return operation.obligation_ids.contains(&obligation.id);
            }
            // An observation obligation is bound to a refused operation by
            // the evidence that operation would have produced.
            if let Some(requirement) = obligation.evidence_requirement.as_ref() {
                if operation
                    .evidence_capabilities
                    .iter()
                    .any(|capability| requirement.supports_capability(capability))
                {
                    return true;
                }
                // …or by the resource: a refusal on one of the request's
                // named verification targets binds an observation obligation
                // about "the exact resource named by the request" (typed
                // target when present, otherwise any verification target).
                let bound_targets: Vec<&crate::traits::RequestVerificationTarget> =
                    match requirement.target.as_ref() {
                        Some(target) => vec![target],
                        None => self.verification_targets.iter().collect(),
                    };
                if bound_targets
                    .iter()
                    .any(|target| hints_touch_target(&self.paths, target, &operation.targets))
                {
                    return true;
                }
            }
            // Compatibility for pre-kernel claims only: bind by predicate.
            obligation.receipt.as_ref().is_some_and(|predicate| {
                predicate.tool_names.is_empty()
                    || predicate.tool_names.contains(&operation.tool_name)
            })
        })
    }

    /// Every obligation still open was refused by a typed authority boundary
    /// and nothing dispatched: the run ended at that boundary.
    fn closed_by_policy_denial(&self) -> bool {
        if !self.terminal_policy_denial() {
            return false;
        }
        let mut open = 0;
        for obligation in self.obligations.values() {
            if obligation.state == RunObligationState::Satisfied
                || obligation.class == RunObligationClass::Deliver
            {
                continue;
            }
            open += 1;
            if !self.obligation_policy_denied(obligation) {
                return false;
            }
        }
        open > 0
    }

    /// Ledger-first closeout arbiter.
    ///
    /// The contract's expectations are proposals; the receipt ledger decides
    /// what is proven and — via `admissible` (Authority ∩ visible tools) —
    /// what could still be proven. An expectation is *reachable* only when no
    /// recorded denial or spent budget blocks it and some admissible tool can
    /// satisfy it. Only a reachable expectation may ask the loop for more
    /// work; everything else closes on the evidence that exists.
    pub fn closeout(&self, admissible: impl Fn(&RunObligation) -> bool) -> CloseoutDecision {
        if !self.expectations_present() || self.obligations.is_empty() {
            return CloseoutDecision::Closed {
                proof_basis: "no_obligations",
            };
        }
        if self.is_fulfilled() {
            return CloseoutDecision::Closed {
                proof_basis: "contract",
            };
        }
        let mut reachable = Vec::new();
        let mut blocked = Vec::new();
        for obligation in self.obligations.values() {
            if obligation.state == RunObligationState::Satisfied {
                continue;
            }
            // Delivery obligations are satisfied by the closeout itself (the
            // response being prepared); they can never ask for more work.
            if obligation.class == RunObligationClass::Deliver {
                continue;
            }
            let reason = if obligation.state == RunObligationState::Deferred {
                Some(UnreachableReason::ExecutorDeferred)
            } else if self.obligation_policy_denied(obligation) {
                Some(UnreachableReason::PolicyDenied)
            } else if self.obligation_exhausted(obligation) {
                Some(UnreachableReason::CardinalityExhausted)
            } else if !admissible(obligation) {
                Some(UnreachableReason::NoAdmissibleTool)
            } else {
                None
            };
            match reason {
                Some(reason) => blocked.push((obligation.id.clone(), reason)),
                None => reachable.push(obligation.id.clone()),
            }
        }
        reachable.sort();
        blocked.sort_by(|left, right| left.0.cmp(&right.0));
        if !reachable.is_empty() {
            return CloseoutDecision::Reachable {
                obligation_ids: reachable,
            };
        }
        if self.evidence_closed() {
            return CloseoutDecision::Closed {
                proof_basis: if self.terminal_policy_denial() {
                    "credited_denial"
                } else {
                    "evidence"
                },
            };
        }
        CloseoutDecision::Unreachable { blocked }
    }

    /// The owner declined at least one operation of the run: what it was
    /// asked to do did not all happen, so it never closes as a success.
    pub fn owner_declined(&self) -> bool {
        self.operations
            .values()
            .any(|operation| operation.owner_declined)
    }

    /// Operations refused before dispatch by a typed policy boundary.
    pub fn policy_denied_operations(&self) -> usize {
        self.operations
            .values()
            .filter(|operation| operation.policy_denied)
            .count()
    }

    /// Refused operations that leave work the request asked for undone: a
    /// refused mutation, or a refused operation bound to one of the run's
    /// obligations. A refused read bound to nothing (the model probing a
    /// path it guessed) owes nothing; whether the request was met is
    /// judged by the obligations and receipts.
    pub fn owed_policy_denials(&self) -> usize {
        self.operations
            .values()
            .filter(|operation| {
                operation.policy_denied
                    && (operation.mutating || !operation.obligation_ids.is_empty())
            })
            .count()
    }

    /// Operations whose adapter actually ran.
    pub fn dispatched_operations(&self) -> usize {
        self.operations
            .values()
            .filter(|operation| operation.dispatched)
            .count()
    }

    /// A dispatched invocation has handed execution to durable background
    /// tracking and has not emitted its terminal receipt yet.
    pub fn has_background_in_flight(&self) -> bool {
        self.operations
            .values()
            .any(RunOperation::is_background_in_flight)
    }

    /// The run's only terminal receipts are typed policy denials: the model
    /// observed the request's authority boundary and nothing it can retry
    /// would cross it. Finalization treats the model's narration of that
    /// boundary as the closeout instead of demanding more evidence.
    pub fn terminal_policy_denial(&self) -> bool {
        self.policy_denied_operations() > 0 && self.dispatched_operations() == 0
    }

    pub fn evidence_closed(&self) -> bool {
        // Receipts close a run the contract could not credit — but not past
        // the executor's own declared, still-open work. Those items are the
        // model's typed statement that more is required, so "every receipt
        // succeeded" is incompleteness, not proof.
        if self.open_executor_expectations() > 0 || self.executor_claimed_unfinished() {
            return false;
        }
        if !self.expectations_present()
            || !self.invariant_violations.is_empty()
            || self.cardinality_violations > 0
            || self.operations.is_empty()
        {
            return false;
        }
        let credited = self
            .obligations
            .values()
            .flat_map(|obligation| obligation.satisfying_receipt_ids.iter())
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        self.operations.values().all(|operation| {
            if !operation.has_terminal_receipt() {
                return false;
            }
            let proof_id = match operation.operation_lineage.as_ref() {
                Some(super::ToolOperationLineage::DurableReplay {
                    source_operation_id,
                    ..
                }) => source_operation_id.as_str(),
                _ => operation.operation_id.as_str(),
            };
            operation.outcome == Some(ToolOutcomeStatus::Succeeded) || credited.contains(proof_id)
        })
    }

    /// The run did work and every operation that reached a terminal
    /// receipt failed: nothing succeeded and nothing was refused by policy.
    /// A run with no contract has no other proof that it accomplished
    /// anything, so its success cannot be credited.
    pub fn only_failed_operations(&self) -> bool {
        let mut terminal = self
            .operations
            .values()
            .filter(|operation| operation.has_terminal_receipt() && !operation.policy_denied)
            .peekable();
        terminal.peek().is_some()
            && terminal.all(|operation| operation.outcome.is_some_and(|status| status.is_failure()))
    }

    pub fn terminal_decision(&self) -> RunTerminalDecision {
        if !self.expectations_present() || self.obligations.is_empty() {
            return RunTerminalDecision::Unspecified;
        }
        if !self.invariant_violations.is_empty() {
            return RunTerminalDecision::Failed;
        }
        // A detached/background process has consumed one durable invocation,
        // but it has not produced a terminal receipt yet. It is neither
        // success nor exhausted failure, even if an older serialized
        // projection credited its provisional handoff as evidence.
        if self.has_background_in_flight() {
            return RunTerminalDecision::Pending;
        }
        if self.is_fulfilled() {
            return RunTerminalDecision::Succeeded;
        }
        // Evidence overrides only an otherwise-failed contract. A still
        // achievable contract stays pending so the loop can finish the work.
        let evidence_closed = self.evidence_closed();
        let exhausted = self.obligations.values().any(|obligation| {
            if !matches!(
                obligation.state,
                RunObligationState::Pending
                    | RunObligationState::Invalidated
                    | RunObligationState::Unverifiable
            ) {
                return false;
            }
            if obligation.state == RunObligationState::Unverifiable {
                return true;
            }
            let Some(predicate) = obligation.receipt.as_ref() else {
                return false;
            };
            predicate.max_invocations.is_some_and(|limit| {
                self.operations
                    .values()
                    .filter(|operation| {
                        if matches!(
                            operation.operation_lineage,
                            Some(super::ToolOperationLineage::DurableReplay { .. })
                        ) {
                            return false;
                        }
                        if !operation.obligation_ids.is_empty() {
                            operation.obligation_ids.contains(&obligation.id)
                        } else {
                            // Compatibility for pre-kernel claims only.
                            operation.result_id.is_some()
                                && (predicate.tool_names.is_empty()
                                    || predicate.tool_names.contains(&operation.tool_name))
                        }
                    })
                    .count()
                    >= limit
            })
        });
        // A typed policy denial that the contract itself credited (the request
        // asked for the attempt and anticipated its refusal) is the run's
        // terminal observation: nothing dispatched, nothing can be retried
        // across that boundary, and the remaining obligations describe work
        // the request placed beyond it ("if denied, stop"). Evidence closes
        // the run. An uncredited denial never qualifies because
        // `evidence_closed` requires every terminal receipt to be credited.
        let credited_terminal_denial = self.terminal_policy_denial() && evidence_closed;
        if (exhausted || credited_terminal_denial) && evidence_closed {
            RunTerminalDecision::SucceededByEvidence
        } else if self.closed_by_policy_denial() {
            RunTerminalDecision::ClosedByPolicyDenial
        } else if exhausted || self.cardinality_violations > 0 {
            RunTerminalDecision::Failed
        } else {
            RunTerminalDecision::Pending
        }
    }

    /// Decide admission from durable history. This is the only correctness
    /// gate for operation retry and request cardinality; in-memory ledgers are
    /// compatibility telemetry and cannot veto an admitted claim.
    pub fn admit_operation(&self, claim: &TaskKernelOperationClaim) -> TaskKernelAdmission {
        if self.recorded_task_status.is_some() {
            return TaskKernelAdmission::Rejected {
                code: "task_already_terminal",
                detail: "The task already has a durable terminal transition.".to_string(),
            };
        }
        if !self.allowed_tool_names.is_empty()
            && !self.allowed_tool_names.contains(&claim.tool_name)
        {
            return TaskKernelAdmission::Rejected {
                code: "tool_outside_authority",
                detail: format!(
                    "`{}` is outside the compiled task capability set.",
                    claim.tool_name
                ),
            };
        }
        if self
            .triggered_dispatch_stop_tools()
            .contains(&claim.tool_name)
        {
            return TaskKernelAdmission::Rejected {
                code: "dispatch_stop_rule_triggered",
                detail: format!(
                    "`{}` is closed by a satisfied task-local prerequisite transition.",
                    claim.tool_name
                ),
            };
        }
        let lineage_source = claim
            .operation_lineage
            .as_ref()
            .map(|lineage| match lineage {
                super::ToolOperationLineage::DurableReplay {
                    source_operation_id,
                    source_result_id,
                }
                | super::ToolOperationLineage::ReconcileInvalidated {
                    source_operation_id,
                    source_result_id,
                } => (source_operation_id, source_result_id),
            });
        if let Some((source_operation_id, source_result_id)) = lineage_source {
            let Some(source) = self.operations.get(source_operation_id) else {
                return TaskKernelAdmission::Rejected {
                    code: "operation_lineage_source_missing",
                    detail: "The claimed source operation is not in durable task history."
                        .to_string(),
                };
            };
            if source.result_id.as_deref() != Some(source_result_id.as_str())
                || source.tool_name != claim.tool_name
                || source.idempotency_key != claim.idempotency_key
                || source.obligation_ids != claim.obligation_ids
            {
                return TaskKernelAdmission::Rejected {
                    code: "operation_lineage_mismatch",
                    detail: "The replay/reconciliation claim does not match its durable source."
                        .to_string(),
                };
            }
            if matches!(
                claim.operation_lineage,
                Some(super::ToolOperationLineage::DurableReplay { .. })
            ) {
                return TaskKernelAdmission::Admitted;
            }
        }
        let is_reconciliation = matches!(
            claim.operation_lineage,
            Some(super::ToolOperationLineage::ReconcileInvalidated { .. })
        );
        let operation_invocations = self
            .operations
            .values()
            .filter(|operation| {
                !matches!(
                    operation.operation_lineage,
                    Some(super::ToolOperationLineage::DurableReplay { .. })
                ) && operation.stable_operation_key.as_deref()
                    == Some(claim.stable_operation_key.as_str())
            })
            .count();
        if !is_reconciliation && operation_invocations >= claim.max_invocations.max(1) {
            return TaskKernelAdmission::Rejected {
                code: "operation_invocations_exhausted",
                detail: format!(
                    "The durable operation has reached its {}-invocation ceiling.",
                    claim.max_invocations.max(1)
                ),
            };
        }
        let dispatched_attempts = self
            .operations
            .values()
            .filter(|operation| {
                !matches!(
                    operation.operation_lineage,
                    Some(super::ToolOperationLineage::DurableReplay { .. })
                ) && operation.stable_operation_key.as_deref()
                    == Some(claim.stable_operation_key.as_str())
                    && operation.dispatched
            })
            .count();
        if !is_reconciliation && dispatched_attempts >= claim.max_attempts.max(1) {
            return TaskKernelAdmission::Rejected {
                code: "operation_attempts_exhausted",
                detail: format!(
                    "The durable operation has reached its {}-attempt ceiling.",
                    claim.max_attempts.max(1)
                ),
            };
        }
        for obligation_id in &claim.obligation_ids {
            let Some(obligation) = self.obligations.get(obligation_id) else {
                return TaskKernelAdmission::Rejected {
                    code: "unknown_obligation",
                    detail: format!("Operation references unknown obligation `{obligation_id}`."),
                };
            };
            let Some(limit) = obligation
                .receipt
                .as_ref()
                .and_then(|predicate| predicate.max_invocations)
            else {
                continue;
            };
            let used = self
                .operations
                .values()
                .filter(|operation| operation.obligation_ids.contains(obligation_id))
                .filter(|operation| {
                    !matches!(
                        operation.operation_lineage,
                        Some(super::ToolOperationLineage::DurableReplay { .. })
                    )
                })
                .count();
            if used >= limit.max(1) {
                return TaskKernelAdmission::Rejected {
                    code: "obligation_cardinality_exhausted",
                    detail: format!(
                        "Obligation `{obligation_id}` has reached its {}-invocation ceiling.",
                        limit.max(1)
                    ),
                };
            }
        }
        TaskKernelAdmission::Admitted
    }

    /// Exact tool lanes closed by already-satisfied receipt triggers. The
    /// projection is deterministic from durable obligations and is safe to use
    /// both for model tool visibility and atomic admission.
    pub fn triggered_dispatch_stop_tools(&self) -> BTreeSet<String> {
        self.dispatch_stop_rules
            .iter()
            .filter(|rule| {
                self.obligations.values().any(|obligation| {
                    obligation.state == RunObligationState::Satisfied
                        && obligation.receipt.as_ref() == Some(&rule.trigger)
                })
            })
            .flat_map(|rule| rule.blocked_tool_names.iter().cloned())
            .collect()
    }

    pub fn lifecycle_phase(&self) -> TaskKernelPhase {
        if let Some(status) = self.recorded_task_status {
            match status {
                TaskStatus::Cancelled => return TaskKernelPhase::Cancelled,
                TaskStatus::Interrupted => return TaskKernelPhase::Interrupted,
                TaskStatus::Failed => return TaskKernelPhase::Failed,
                TaskStatus::Completed => {}
            }
            match self.recorded_task_outcome {
                Some(TaskOutcome::Failed) => return TaskKernelPhase::Failed,
                Some(TaskOutcome::Partial) => return TaskKernelPhase::Partial,
                _ => {}
            }
        }
        if self.delivery_states.values().any(|state| {
            matches!(
                state,
                ResponseDeliveryState::PlatformAcknowledged
                    | ResponseDeliveryState::Edited
                    | ResponseDeliveryState::Rendered
                    | ResponseDeliveryState::Read
            )
        }) {
            return TaskKernelPhase::Delivered;
        }
        if self.delivery_states.values().any(|state| {
            matches!(
                state,
                ResponseDeliveryState::Queued | ResponseDeliveryState::Sent
            )
        }) {
            return TaskKernelPhase::DeliveryQueued;
        }
        if self
            .delivery_states
            .values()
            .any(|state| *state == ResponseDeliveryState::Failed)
        {
            return TaskKernelPhase::DeliveryFailed;
        }
        if self.prepared_response_id.is_some() {
            return TaskKernelPhase::ResponsePrepared;
        }
        match self.terminal_decision() {
            // The agent's own `partial`/`blocked` is never upgraded by
            // receipts that happen to cover the declared work.
            RunTerminalDecision::Succeeded | RunTerminalDecision::SucceededByEvidence
                if self.executor_claimed_unfinished() =>
            {
                TaskKernelPhase::Partial
            }
            RunTerminalDecision::Succeeded | RunTerminalDecision::SucceededByEvidence => {
                TaskKernelPhase::WorkSucceeded
            }
            // The requested work was refused, so it did not happen.
            RunTerminalDecision::ClosedByPolicyDenial => TaskKernelPhase::Partial,
            RunTerminalDecision::Failed => TaskKernelPhase::Failed,
            RunTerminalDecision::Pending => TaskKernelPhase::WorkPending,
            RunTerminalDecision::Unspecified if self.started => TaskKernelPhase::Running,
            RunTerminalDecision::Unspecified => TaskKernelPhase::Initialized,
        }
    }
}

pub fn receipt_matches_predicate(
    result: &ToolResultData,
    predicate: &RequestReceiptPredicate,
) -> bool {
    let Some(receipt) = result.receipt.as_ref() else {
        return false;
    };
    let tool_matches = predicate.tool_names.is_empty()
        || predicate.tool_names.iter().any(|name| {
            name == &result.name || receipt.effective_tool_name.as_deref() == Some(name)
        });
    if !tool_matches {
        return false;
    }
    if !predicate.rejection_matches(receipt.contract_rejected) {
        return false;
    }
    if !predicate.exit_matches(receipt.receipt_kind, receipt.exit_code) {
        return false;
    }
    if !predicate.outcome_matches(
        receipt.outcome_status,
        receipt.invocation_stage,
        receipt.contract_rejected,
        receipt.receipt_kind,
        receipt.exit_code,
    ) {
        return false;
    }
    !predicate.requires_output || receipt.result_provenance.authoritative_chars > 0
}

fn evidence_receipt_supports(
    paths: &LexicalPaths,
    requirement: &RequestEvidenceRequirement,
    receipt: &super::ToolReceiptV1,
) -> bool {
    if !receipt
        .semantics
        .evidence
        .iter()
        .any(|capability| requirement.supports_capability(capability))
    {
        return false;
    }
    let Some(target) = requirement.target.as_ref() else {
        return true;
    };
    receipt_touches_target(paths, target, receipt)
}

/// Subjects a receipt touched or proposed to touch: semantic target hints
/// plus the declared access manifest (read/write paths and scope roots).
fn receipt_targets(receipt: &super::ToolReceiptV1) -> Vec<ToolTargetHint> {
    let mut targets = receipt.semantics.target_hints.clone();
    if let Some(manifest) = receipt.access_manifest.as_ref() {
        for hint in manifest
            .read_targets
            .iter()
            .chain(manifest.write_targets.iter())
        {
            if !targets.contains(hint) {
                targets.push(hint.clone());
            }
        }
    }
    targets
}

/// Whether the receipt's dispatched subject set contains the target. A path
/// target is touched by an equal `Path` hint or by a `ProjectScope` root that
/// is the target itself or one of its ancestors (a directory declared as a
/// write root covers the directory and everything created inside it).
fn receipt_touches_target(
    paths: &LexicalPaths,
    target: &crate::traits::RequestVerificationTarget,
    receipt: &super::ToolReceiptV1,
) -> bool {
    hints_touch_target(paths, target, &receipt_targets(receipt))
}

fn hints_touch_target(
    paths: &LexicalPaths,
    target: &crate::traits::RequestVerificationTarget,
    hints: &[ToolTargetHint],
) -> bool {
    hints
        .iter()
        .any(|hint| hint_matches_target(paths, target, hint))
}

fn hint_matches_target(
    paths: &LexicalPaths,
    target: &crate::traits::RequestVerificationTarget,
    hint: &ToolTargetHint,
) -> bool {
    match (target.kind, hint.kind) {
        (RequestVerificationTargetKind::Url, ToolTargetHintKind::Url) => target.value == hint.value,
        (RequestVerificationTargetKind::Path, ToolTargetHintKind::Path) => {
            // The target itself, or a path under it: only a directory has
            // paths under it, so a folder-scoped item is met by work on its
            // files while a file-bound item still needs that exact file.
            let expected = paths.normalize(&target.value);
            let actual = paths.normalize(&hint.value);
            expected
                .ok()
                .zip(actual.ok())
                .is_some_and(|(target, actual)| {
                    let root = target.as_str().trim_end_matches('/');
                    actual.as_str() == target.as_str()
                        || actual.as_str().starts_with(&format!("{root}/"))
                })
        }
        (RequestVerificationTargetKind::Path, ToolTargetHintKind::ProjectScope) => {
            let expected = paths.normalize(&target.value);
            let root = paths.normalize(&hint.value);
            expected.ok().zip(root.ok()).is_some_and(|(target, root)| {
                let target = target.as_str();
                let root = root.as_str().trim_end_matches('/');
                target == root || target.starts_with(&format!("{root}/"))
            })
        }
        (RequestVerificationTargetKind::ResourceId, ToolTargetHintKind::ResourceId) => {
            target.value == hint.value
        }
        _ => false,
    }
}

/// Whether a receipt subject of `hint` kind could have been a target of
/// `target` kind (a path target is named by a path or by a root over it).
fn hint_is_of_target_kind(target: RequestVerificationTargetKind, hint: ToolTargetHintKind) -> bool {
    matches!(
        (target, hint),
        (RequestVerificationTargetKind::Url, ToolTargetHintKind::Url)
            | (
                RequestVerificationTargetKind::Path,
                ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope
            )
            | (
                RequestVerificationTargetKind::ResourceId,
                ToolTargetHintKind::ResourceId
            )
    )
}

fn facets_cover(observed: &[ToolSemanticFacet], required: &[ToolSemanticFacet]) -> bool {
    required.iter().all(|facet| observed.contains(facet))
}

fn direct_observation_supports(
    paths: &LexicalPaths,
    target: &RequestObservationTarget,
    observations: &[ToolObservationEvidence],
) -> bool {
    observations.iter().any(|observation| {
        hint_matches_target(paths, &target.subject, &observation.subject)
            && facets_cover(&observation.facets, &target.facets)
    })
}

/// Verify one exact subject/facet selector against result assertions. Complete
/// collection coverage can prove non-membership; partial coverage can only
/// prove rows it actually returned, and those rows still need direct facets.
fn observation_target_supported(
    paths: &LexicalPaths,
    target: &RequestObservationTarget,
    receipt_targets: &[ToolTargetHint],
    observations: &[ToolObservationEvidence],
    collections: &[ToolCollectionObservation],
) -> bool {
    if let Some(requirement) = target.collection_coverage.as_ref() {
        let Some(collection) = collections.iter().find(|collection| {
            hint_matches_target(paths, &requirement.collection, &collection.collection)
                && collection
                    .completeness
                    .satisfies(requirement.minimum_completeness)
                && facets_cover(&collection.facets, &target.facets)
        }) else {
            return false;
        };
        // A selector may name the collection itself when the exact member
        // subject is not known until discovery. This binds the obligation to
        // complete typed coverage, never to an invented display-name alias.
        if hint_matches_target(paths, &target.subject, &collection.collection) {
            return true;
        }
        let subject_is_member = collection
            .members
            .iter()
            .any(|member| hint_matches_target(paths, &target.subject, member));
        if !subject_is_member {
            return collection.completeness == crate::traits::ToolCollectionCompleteness::Complete;
        }
        return direct_observation_supports(paths, target, observations);
    }

    if direct_observation_supports(paths, target, observations) {
        return true;
    }
    // Legacy path/URL/resource checklist declarations did not include facets.
    // Preserve their exact-target behavior while refusing to infer any typed
    // facet merely from proposed call semantics.
    target.facets.is_empty() && hints_touch_target(paths, &target.subject, receipt_targets)
}

fn observation_targets_supported(
    paths: &LexicalPaths,
    targets: &[RequestObservationTarget],
    receipt_targets: &[ToolTargetHint],
    observations: &[ToolObservationEvidence],
    collections: &[ToolCollectionObservation],
) -> bool {
    targets.iter().all(|target| {
        observation_target_supported(paths, target, receipt_targets, observations, collections)
    })
}

fn obligation_touched_by_targets(
    paths: &LexicalPaths,
    obligation: &RunObligation,
    hints: &[ToolTargetHint],
) -> bool {
    if !obligation.observation_targets.is_empty() {
        return obligation.observation_targets.iter().any(|target| {
            hints_touch_target(paths, &target.subject, hints)
                || target
                    .collection_coverage
                    .as_ref()
                    .is_some_and(|coverage| hints_touch_target(paths, &coverage.collection, hints))
        });
    }
    obligation
        .required_target
        .as_ref()
        .is_none_or(|target| hints_touch_target(paths, target, hints))
}

fn executor_item_summary(
    item: &ExecutorExpectationItem,
    class: RunObligationClass,
    required_effect: ToolMutationEffects,
    required_target: Option<&crate::traits::RequestVerificationTarget>,
) -> String {
    let target = item
        .targets
        .first()
        .map(|target| format!(" on {target}"))
        .unwrap_or_default();
    match class {
        RunObligationClass::Achieve => {
            // How a receipt names this kind of target, then the other proof
            // `recredit_checklist_obligations_from_recorded_receipts` accepts.
            let how = match required_target.map(|target| target.kind) {
                None => String::new(),
                Some(RequestVerificationTargetKind::Path) => {
                    " (declare the path as write access); or the effect, then a later completed \
                     observation of the path"
                        .to_string()
                }
                Some(RequestVerificationTargetKind::Url) => {
                    " that names the URL; or the effect, then a later completed observation of \
                     the URL (an HTTP request, browser load or successful curl/wget of it)"
                        .to_string()
                }
                Some(RequestVerificationTargetKind::ResourceId) => {
                    " that names the resource; or the effect, then a later completed \
                     observation of the resource"
                        .to_string()
                }
            };
            format!(
                "checklist item {}: {} — needs a succeeded receipt carrying {}{}{}",
                item.index + 1,
                item.description,
                required_effect.protocol_names().join("|"),
                target,
                how
            )
        }
        _ => format!(
            "checklist item {}: {} — needs a completed observation receipt{}",
            item.index + 1,
            item.description,
            target
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{
        EvidenceAuthority, EvidencePurpose, RequestCompletionContract, RequestEvidenceRequirement,
        RequestedOutcomeCondition, ToolCallEffect, ToolCallSemantics, ToolEvidenceCapability,
        ToolInvocationStage, ToolReceiptKind, ToolResultProvenance, ToolSemanticScope,
    };
    use crate::{
        Event, ExecutorExpectationItem, ExecutorExpectationsDeclaredData, TaskContractCompiledData,
        ToolOutcomeEvidenceSource, ToolReceiptV1,
    };
    use serde_json::json;

    #[test]
    fn path_targets_compare_on_the_backend_the_store_was_given() {
        let paths = LexicalPaths::posix("/remote/work", "/home/synthetic-user");
        let target = |value: &str| {
            crate::traits::RequestVerificationTarget::new(
                RequestVerificationTargetKind::Path,
                value,
            )
            .unwrap()
        };
        let hint = |kind, value: &str| ToolTargetHint::new(kind, value).unwrap();
        // Relative targets and `~` resolve on the execution backend, not on
        // the daemon host running this test.
        assert!(hint_matches_target(
            &paths,
            &target("notes/todo.md"),
            &hint(ToolTargetHintKind::Path, "/remote/work/notes/todo.md"),
        ));
        assert!(hint_matches_target(
            &paths,
            &target("~/plan.md"),
            &hint(ToolTargetHintKind::Path, "/home/synthetic-user/plan.md"),
        ));
        assert!(hint_matches_target(
            &paths,
            &target("/remote/work/a/../notes/todo.md"),
            &hint(ToolTargetHintKind::ProjectScope, "/remote/work/notes"),
        ));
        assert!(!hint_matches_target(
            &paths,
            &target("notes/todo.md"),
            &hint(ToolTargetHintKind::Path, "/elsewhere/notes/todo.md"),
        ));
    }

    fn event(event_type: EventType, value: impl Serialize) -> Event {
        let mut event = Event::new(
            "session",
            event_type,
            serde_json::to_value(value).expect("serialize event"),
        );
        event.task_id = Some("task-1".to_string());
        event
    }

    fn predicate(tool: &str, exit_code: i32) -> RequestReceiptPredicate {
        RequestReceiptPredicate {
            tool_names: vec![tool.to_string()],
            exit_codes: vec![exit_code],
            max_invocations: Some(1),
            ..RequestReceiptPredicate::default()
        }
    }

    fn contract(requirements: Vec<RequestEvidenceRequirement>) -> Event {
        contract_with_response(requirements, None)
    }

    fn contract_with_response(
        requirements: Vec<RequestEvidenceRequirement>,
        response_contract: Option<Box<RequestResponseContract>>,
    ) -> Event {
        contract_with_response_and_tools(requirements, response_contract, Vec::new())
    }

    fn contract_with_response_and_tools(
        requirements: Vec<RequestEvidenceRequirement>,
        response_contract: Option<Box<RequestResponseContract>>,
        allowed_tool_names: Vec<String>,
    ) -> Event {
        event(
            EventType::TaskContractCompiled,
            TaskContractCompiledData {
                schema_version: TaskContractCompiledData::SCHEMA_VERSION,
                task_id: "task-1".to_string(),
                contract: RequestCompletionContract {
                    scope_task_id: Some("task-1".to_string()),
                    adopted_from_task_ids: Vec::new(),
                    task_kind: crate::traits::RequestTaskKind::Check,
                    expects_mutation: false,
                    required_mutation_effects: ToolMutationEffects::NONE,
                    forbids_mutation: false,
                    forbids_tool_use: false,
                    allowed_tool_names,
                    forbidden_tool_scopes: Vec::new(),
                    dispatch_stop_rules: Vec::new(),
                    required_response_fields: Vec::new(),
                    response_contract,
                    forbidden_actions: Vec::new(),
                    requires_observation: !requirements.is_empty(),
                    requires_reverification_after_mutation: false,
                    explicit_verification_requested: !requirements.is_empty(),
                    minimum_sources: 0,
                    requires_primary_sources: false,
                    requires_exact_history: false,
                    evidence_requirements: requirements,
                    adopted_evidence_bindings: Vec::new(),
                    verification_targets: Vec::new(),
                },
                required_invocations: Vec::new(),
            },
        )
    }

    fn requirement(tool: &str, exit_code: i32) -> RequestEvidenceRequirement {
        RequestEvidenceRequirement {
            summary: "synthetic invocation".to_string(),
            acceptable_scopes: Vec::new(),
            purpose: EvidencePurpose::Outcome,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Historical,
            required_content_markers: Vec::new(),
            receipt: Some(predicate(tool, exit_code)),
            target: None,
        }
    }

    fn call(id: &str, tool: &str) -> Event {
        event(
            EventType::ToolCall,
            ToolCallData {
                tool_call_id: id.to_string(),
                name: tool.to_string(),
                arguments: json!({}),
                summary: None,
                task_id: Some("task-1".to_string()),
                idempotency_key: Some(format!("operation:{id}")),
                policy_rev: None,
                risk_score: None,
                stable_operation_key: None,
                obligation_ids: Vec::new(),
                max_operation_attempts: None,
                max_operation_invocations: None,
                operation_lineage: None,
                turn_id: None,
            },
        )
    }

    fn claimed_call(id: &str, tool: &str, stable_key: &str, obligation_ids: &[&str]) -> Event {
        event(
            EventType::ToolCall,
            ToolCallData::from_tool_call(id, tool, json!({}), Some("task-1".to_string()))
                .with_policy_metadata(Some(format!("operation:{id}")), None, None)
                .with_kernel_claim(
                    stable_key,
                    obligation_ids.iter().map(|id| (*id).to_string()).collect(),
                    1,
                    1,
                ),
        )
    }

    #[test]
    fn operation_claim_projects_away_closed_sibling_cardinality() {
        let exact_three = RequestEvidenceRequirement {
            summary: "Observe three synthetic receipts".to_string(),
            acceptable_scopes: Vec::new(),
            purpose: EvidencePurpose::Outcome,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Historical,
            required_content_markers: Vec::new(),
            receipt: Some(RequestReceiptPredicate {
                tool_names: vec!["terminal".to_string()],
                exit_codes: vec![0],
                min_invocations: Some(3),
                max_invocations: Some(3),
                ..RequestReceiptPredicate::default()
            }),
            target: None,
        };
        let exact_one_with_output = RequestEvidenceRequirement {
            summary: "Observe one synthetic output receipt".to_string(),
            receipt: Some(RequestReceiptPredicate {
                tool_names: vec!["terminal".to_string()],
                exit_codes: vec![0],
                requires_output: true,
                min_invocations: Some(1),
                max_invocations: Some(1),
                ..RequestReceiptPredicate::default()
            }),
            ..exact_three.clone()
        };
        let first_obligation = "task:task-1/obligation:evidence:0";
        let second_obligation = "task:task-1/obligation:evidence:1";
        let mut events = vec![
            contract(vec![exact_three, exact_one_with_output]),
            claimed_call(
                "first",
                "terminal",
                "operation:first",
                &[first_obligation, second_obligation],
            ),
            result(
                "first",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations[second_obligation].state,
            RunObligationState::Satisfied
        );
        assert_eq!(
            aggregate.obligations[first_obligation].state,
            RunObligationState::Pending
        );

        let proposed = TaskKernelOperationClaim {
            operation_id: "second".to_string(),
            stable_operation_key: "operation:second".to_string(),
            tool_name: "terminal".to_string(),
            obligation_ids: vec![first_obligation.to_string(), second_obligation.to_string()],
            max_attempts: 1,
            max_invocations: 1,
            idempotency_key: None,
            operation_lineage: None,
        };
        let effective = aggregate.effective_operation_claim(&proposed);
        assert_eq!(effective.obligation_ids, [first_obligation]);
        assert_eq!(effective.max_invocations, 3);
        assert_eq!(effective.max_attempts, 3);
        assert_eq!(
            aggregate.admit_operation(&effective),
            TaskKernelAdmission::Admitted
        );
    }

    fn result(
        id: &str,
        tool: &str,
        status: ToolOutcomeStatus,
        exit_code: i32,
        semantics: ToolCallSemantics,
    ) -> Event {
        event(
            EventType::ToolResult,
            ToolResultData {
                message_id: None,
                tool_call_id: id.to_string(),
                name: tool.to_string(),
                result: "synthetic result".to_string(),
                success: status == ToolOutcomeStatus::Succeeded,
                duration_ms: 1,
                error: None,
                task_id: Some("task-1".to_string()),
                annotations: Vec::new(),
                turn_id: None,
                attachments: Vec::new(),
                receipt: Some(ToolReceiptV1 {
                    schema_version: ToolReceiptV1::SCHEMA_VERSION,
                    outcome_status: status,
                    invocation_stage: ToolInvocationStage::Dispatched,
                    outcome_evidence: ToolOutcomeEvidenceSource::StructuredMetadata,
                    receipt_kind: ToolReceiptKind::Process,
                    access_manifest: None,
                    access_enforcement: Default::default(),
                    access_denial: None,
                    contract_rejected: false,
                    effective_tool_name: None,
                    idempotency_key: Some(format!("operation:{id}")),
                    exit_code: Some(exit_code),
                    timed_out: false,
                    background_started: false,
                    detached: false,
                    completion_notifications_enabled: false,
                    transport_error: None,
                    http_status: None,
                    truncation: None,
                    result_provenance: ToolResultProvenance {
                        result_id: Some(format!("result:{id}")),
                        authoritative_chars: 16,
                        ..ToolResultProvenance::default()
                    },
                    authorization_preflight: None,
                    completion_obligation_ids: Vec::new(),
                    continuation_obligation_ids: Vec::new(),
                    semantics,
                    observations: Vec::new(),
                    collection_observations: Vec::new(),
                    mandate_authority: None,
                    checklist_items: Vec::new(),
                }),
            },
        )
    }

    fn with_result_id(mut event: Event, result_id: &str) -> Event {
        event.data["receipt"]["result_provenance"]["result_id"] = json!(result_id);
        event
    }

    fn background_result(id: &str, result_id: &str) -> Event {
        let mut event = with_result_id(
            result(
                id,
                "terminal",
                ToolOutcomeStatus::Backgrounded,
                0,
                ToolCallSemantics::observation(),
            ),
            result_id,
        );
        event.data["receipt"]["background_started"] = json!(true);
        event.data["receipt"]["detached"] = json!(true);
        event
    }

    #[test]
    fn backgrounded_operation_transitions_to_succeeded_without_new_invocation() {
        let obligation = "task:task-1/obligation:evidence:0";
        let mut events = vec![
            contract(vec![requirement("terminal", 0)]),
            claimed_call("run", "terminal", "operation:run", &[obligation]),
            background_result("run", "result:run:background"),
        ];
        renumber(&mut events);

        let in_flight = RunAggregate::replay("task-1", &events);
        assert_eq!(in_flight.operations.len(), 1);
        assert_eq!(in_flight.dispatched_operations(), 1);
        assert_eq!(in_flight.cardinality_violations, 0);
        assert_eq!(
            in_flight.primary_causal_operation_id.as_deref(),
            Some("run")
        );
        assert_eq!(
            in_flight.obligations[obligation].state,
            RunObligationState::Pending
        );
        assert_eq!(in_flight.terminal_decision(), RunTerminalDecision::Pending);
        assert!(!in_flight.evidence_closed());

        let terminal = with_result_id(
            result(
                "run",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
            "result:run:terminal",
        );
        // No new projection field is needed: an aggregate persisted while the
        // process is detached derives its in-flight state from the existing
        // `outcome` and accepts the later terminal event after deserialization.
        let mut completed: RunAggregate = serde_json::from_value(
            serde_json::to_value(&in_flight).expect("serialize in-flight aggregate"),
        )
        .expect("deserialize in-flight aggregate");
        completed.apply(&terminal);
        assert_eq!(completed.operations.len(), 1);
        assert_eq!(completed.dispatched_operations(), 1);
        assert_eq!(completed.cardinality_violations, 0);
        assert_eq!(
            completed.operations["run"].outcome,
            Some(ToolOutcomeStatus::Succeeded)
        );
        assert_eq!(
            completed.operations["run"].result_id.as_deref(),
            Some("result:run:terminal")
        );
        assert_eq!(
            completed.obligations[obligation].state,
            RunObligationState::Satisfied
        );
        assert_eq!(
            completed.terminal_decision(),
            RunTerminalDecision::Succeeded
        );
        assert!(completed.invariant_violations.is_empty());
    }

    #[test]
    fn backgrounded_operation_transitions_to_failed_without_duplicate_invariant() {
        let obligation = "task:task-1/obligation:evidence:0";
        let mut events = vec![
            contract(vec![requirement("terminal", 0)]),
            claimed_call("run", "terminal", "operation:run", &[obligation]),
            background_result("run", "result:run:background"),
            with_result_id(
                result(
                    "run",
                    "terminal",
                    ToolOutcomeStatus::FailedPermanent,
                    -1,
                    ToolCallSemantics::observation(),
                ),
                "result:run:terminal",
            ),
        ];
        renumber(&mut events);

        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.operations.len(), 1);
        assert_eq!(aggregate.cardinality_violations, 0);
        assert_eq!(
            aggregate.operations["run"].outcome,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Failed);
        assert!(aggregate.invariant_violations.is_empty());
    }

    #[test]
    fn duplicate_background_and_terminal_transition_receipts_are_idempotent() {
        let obligation = "task:task-1/obligation:evidence:0";
        let background = background_result("run", "result:run:background");
        let refreshed_background = background_result("run", "result:run:background-refresh");
        let terminal = with_result_id(
            result(
                "run",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
            "result:run:terminal",
        );
        let mut events = vec![
            contract(vec![requirement("terminal", 0)]),
            claimed_call("run", "terminal", "operation:run", &[obligation]),
            background,
            refreshed_background,
            terminal.clone(),
            terminal,
        ];
        renumber(&mut events);

        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.operations.len(), 1);
        assert_eq!(aggregate.dispatched_operations(), 1);
        assert_eq!(aggregate.cardinality_violations, 0);
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::Succeeded
        );
        assert!(aggregate.invariant_violations.is_empty());
    }

    #[test]
    fn distinct_second_terminal_receipt_after_background_transition_is_rejected() {
        let obligation = "task:task-1/obligation:evidence:0";
        let mut events = vec![
            contract(vec![requirement("terminal", 0)]),
            claimed_call("run", "terminal", "operation:run", &[obligation]),
            background_result("run", "result:run:background"),
            with_result_id(
                result(
                    "run",
                    "terminal",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    ToolCallSemantics::observation(),
                ),
                "result:run:terminal-1",
            ),
            with_result_id(
                result(
                    "run",
                    "terminal",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    ToolCallSemantics::observation(),
                ),
                "result:run:terminal-2",
            ),
        ];
        renumber(&mut events);

        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.operations.len(), 1);
        assert_eq!(
            aggregate.operations["run"].result_id.as_deref(),
            Some("result:run:terminal-1")
        );
        assert_eq!(
            aggregate.invariant_violations,
            ["operation_has_multiple_terminal_receipts"]
        );
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Failed);
    }

    fn assistant_response(id: &str, content: &str) -> Event {
        event(
            EventType::AssistantResponse,
            AssistantResponseData {
                message_id: Some(id.to_string()),
                content: Some(content.to_string()),
                tool_calls: None,
                model: "synthetic-model".to_string(),
                input_tokens: None,
                output_tokens: None,
                annotations: Vec::new(),
                turn_id: None,
                task_id: Some("task-1".to_string()),
                referenced_receipts: Vec::new(),
                disposition: crate::AssistantResponseDisposition::Terminal,
            },
        )
    }

    #[test]
    fn triggered_dispatch_stop_rule_closes_only_the_dependent_lane() {
        let trigger = RequestReceiptPredicate {
            tool_names: vec!["write_file".to_string()],
            outcome_condition: Some(RequestedOutcomeCondition::NonSuccessTerminal),
            min_invocations: Some(1),
            max_invocations: Some(1),
            ..RequestReceiptPredicate::default()
        };
        let mut contract_event = contract(vec![RequestEvidenceRequirement {
            summary: "Observe the prerequisite terminal result".to_string(),
            acceptable_scopes: Vec::new(),
            purpose: EvidencePurpose::Outcome,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Historical,
            required_content_markers: Vec::new(),
            receipt: Some(trigger.clone()),
            target: None,
        }]);
        contract_event.data["contract"]["dispatch_stop_rules"] =
            serde_json::to_value(vec![RequestDispatchStopRule {
                trigger,
                blocked_receipt_kinds: vec![ToolReceiptKind::Process],
                blocked_tool_names: vec!["run_command".to_string(), "terminal".to_string()],
            }])
            .expect("serialize stop rule");
        let obligation = "task:task-1/obligation:evidence:0";
        let mut events = vec![
            contract_event,
            claimed_call("denied", "write_file", "operation:denied", &[obligation]),
            result(
                "denied",
                "write_file",
                ToolOutcomeStatus::Blocked,
                1,
                ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.triggered_dispatch_stop_tools(),
            BTreeSet::from(["run_command".to_string(), "terminal".to_string()])
        );

        let claim = |tool_name: &str| TaskKernelOperationClaim {
            operation_id: format!("operation-{tool_name}"),
            stable_operation_key: format!("stable-{tool_name}"),
            tool_name: tool_name.to_string(),
            obligation_ids: Vec::new(),
            max_attempts: 1,
            max_invocations: 1,
            idempotency_key: None,
            operation_lineage: None,
        };
        assert!(matches!(
            aggregate.admit_operation(&claim("run_command")),
            TaskKernelAdmission::Rejected {
                code: "dispatch_stop_rule_triggered",
                ..
            }
        ));
        assert_eq!(
            aggregate.admit_operation(&claim("read_file")),
            TaskKernelAdmission::Admitted,
            "an independent recovery tool must remain available"
        );
    }

    #[test]
    fn observation_obligation_preserves_scope_in_durable_reducer() {
        let requirement = RequestEvidenceRequirement {
            summary: "Observe one local workspace resource".to_string(),
            acceptable_scopes: vec![ToolSemanticScope::LocalWorkspace],
            purpose: EvidencePurpose::CurrentState,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        };
        let mut events = vec![
            contract(vec![requirement]),
            call("host", "system_info"),
            result(
                "host",
                "system_info",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation().with_evidence(vec![ToolEvidenceCapability::new(
                    ToolSemanticScope::HostLocal,
                    &[EvidencePurpose::CurrentState],
                    EvidenceAuthority::Direct,
                    EvidenceTemporalScope::Current,
                )]),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        let obligation = &aggregate.obligations["task:task-1/obligation:evidence:0"];
        assert_eq!(obligation.state, RunObligationState::Pending);

        let next_id = events.len() as i64 + 1;
        let mut read_call = call("workspace", "read_file");
        read_call.id = next_id;
        events.push(read_call);
        let mut read_result = result(
            "workspace",
            "read_file",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics::observation().with_evidence(vec![ToolEvidenceCapability::new(
                ToolSemanticScope::LocalWorkspace,
                &[EvidencePurpose::CurrentState],
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            )]),
        );
        read_result.id = next_id + 1;
        events.push(read_result);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:evidence:0"].state,
            RunObligationState::Satisfied
        );
    }

    #[test]
    fn invocation_accomplishments_accumulate_across_later_mutations() {
        let mut events = vec![
            contract(vec![
                requirement("write_file", 0),
                requirement("read_file", 0),
                requirement("terminal", 0),
            ]),
            call("write", "write_file"),
            result(
                "write",
                "write_file",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            ),
            call("read", "read_file"),
            result(
                "read",
                "read_file",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics {
                    effect: ToolCallEffect::Observation,
                    ..ToolCallSemantics::default()
                },
            ),
            call("cleanup", "terminal"),
            result(
                "cleanup",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation_and_mutation_with(
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                ),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.satisfied_count(), 3);
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::Succeeded
        );
    }

    #[test]
    fn expected_negative_is_a_completed_perform_obligation() {
        let mut events = vec![
            contract(vec![requirement("terminal", 1)]),
            call("negative", "terminal"),
            result(
                "negative",
                "terminal",
                ToolOutcomeStatus::CompletedWithNegativeResult,
                1,
                ToolCallSemantics {
                    effect: ToolCallEffect::Observation,
                    ..ToolCallSemantics::default()
                },
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        assert_eq!(
            RunAggregate::replay("task-1", &events).terminal_decision(),
            RunTerminalDecision::Succeeded
        );
    }

    #[test]
    fn cardinality_counts_distinct_invocations_even_when_output_digests_match() {
        let mut required = requirement("terminal", 0);
        let predicate = required.receipt.as_mut().expect("receipt predicate");
        predicate.min_invocations = Some(3);
        predicate.max_invocations = Some(3);
        let mut events = vec![
            contract(vec![required]),
            call("first", "terminal"),
            result(
                "first",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }

        let one = RunAggregate::replay("task-1", &events);
        assert_eq!(one.satisfied_count(), 0);
        assert_eq!(one.terminal_decision(), RunTerminalDecision::Pending);

        for id in ["second", "third"] {
            events.push(call(id, "terminal"));
            events.push(result(
                id,
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ));
        }
        // Create/cleanup-style calls often both have empty output. Their
        // content identity is equal, but their durable invocation receipts are
        // distinct and must each count toward an exact call cardinality.
        for event in events
            .iter_mut()
            .filter(|event| event.event_type == EventType::ToolResult)
        {
            event.data["receipt"]["result_provenance"]["result_id"] =
                json!("sha256:synthetic-shared-output");
        }
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let three = RunAggregate::replay("task-1", &events);
        assert_eq!(three.satisfied_count(), 1);
        assert_eq!(three.terminal_decision(), RunTerminalDecision::Succeeded);
    }

    #[test]
    fn observation_without_a_compiled_proposition_is_unverifiable() {
        let mut compiled = contract(Vec::new());
        compiled.data["contract"]["requires_observation"] = json!(true);
        let mut events = vec![
            compiled,
            call("unrelated-read", "manage_mandates"),
            result(
                "unrelated-read",
                "manage_mandates",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }

        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.satisfied_count(), 0);
        assert!(aggregate.obligations.values().any(|obligation| {
            obligation.class == RunObligationClass::Observe
                && obligation.state == RunObligationState::Unverifiable
        }));
        // The unverifiable proposition can never be credited, but the read
        // demonstrably completed. A malformed contract loses to the receipt
        // instead of reporting completed work as failed.
        assert!(!aggregate.work_is_fulfilled());
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::SucceededByEvidence
        );

        // Without any completed operation the unverifiable contract still
        // fails closed: absence of work is not evidence.
        let mut idle = vec![contract(Vec::new())];
        idle[0].data["contract"]["requires_observation"] = json!(true);
        idle[0].id = 1;
        assert_eq!(
            RunAggregate::replay("task-1", &idle).terminal_decision(),
            RunTerminalDecision::Failed
        );
    }

    #[test]
    fn successful_work_requires_the_typed_response_artifact_for_delivery() {
        let response = RequestResponseContract::ExactText {
            success_text: "phase=synthetic; outcome=complete".to_string(),
            source_message_hash: "synthetic-hash".to_string(),
        };
        let mut events = vec![
            contract_with_response(vec![requirement("terminal", 0)], Some(Box::new(response))),
            call("check", "terminal"),
            result(
                "check",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics {
                    effect: ToolCallEffect::Observation,
                    ..ToolCallSemantics::default()
                },
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let work_complete = RunAggregate::replay("task-1", &events);
        assert!(work_complete.work_is_fulfilled());
        assert_eq!(
            work_complete.projected_success_response(),
            Some("phase=synthetic; outcome=complete")
        );
        assert_eq!(
            work_complete.terminal_decision(),
            RunTerminalDecision::Pending
        );

        // An empty terminal artifact is not a delivery.
        events.push(assistant_response("empty", "   "));
        events.last_mut().unwrap().id = 4;
        assert_eq!(
            RunAggregate::replay("task-1", &events).terminal_decision(),
            RunTerminalDecision::Pending
        );

        // The response contract is a presentation hint, not a proof gate. A
        // model-authored answer grounded in the proved work closes delivery
        // without byte-exact agreement with the planner's proposed text.
        events.push(assistant_response("grounded", "generic summary"));
        events.last_mut().unwrap().id = 5;
        let delivered = RunAggregate::replay("task-1", &events);
        assert_eq!(delivered.satisfied_count(), 2);
        assert_eq!(
            delivered.terminal_decision(),
            RunTerminalDecision::Succeeded
        );
    }

    #[test]
    fn expected_non_success_is_objective_success_and_projects_its_response() {
        let response = RequestResponseContract::ExactText {
            success_text: "phase=synthetic; prerequisite=blocked".to_string(),
            source_message_hash: "synthetic-hash".to_string(),
        };
        let requirement = RequestEvidenceRequirement {
            summary: "Observe one terminal non-success".to_string(),
            acceptable_scopes: Vec::new(),
            purpose: EvidencePurpose::Outcome,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Historical,
            required_content_markers: Vec::new(),
            receipt: Some(RequestReceiptPredicate {
                tool_names: vec!["write_file".to_string()],
                outcome_condition: Some(
                    crate::traits::RequestedOutcomeCondition::NonSuccessTerminal,
                ),
                max_invocations: Some(1),
                ..RequestReceiptPredicate::default()
            }),
            target: None,
        };
        let mut events = vec![
            contract_with_response_and_tools(
                vec![requirement],
                Some(Box::new(response)),
                vec!["write_file".to_string()],
            ),
            call("write", "write_file"),
            result(
                "write",
                "write_file",
                ToolOutcomeStatus::FailedPermanent,
                -1,
                ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }

        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.work_is_fulfilled());
        assert_eq!(
            aggregate.projected_success_response(),
            Some("phase=synthetic; prerequisite=blocked")
        );
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);

        events.push(assistant_response(
            "exact-negative",
            "phase=synthetic; prerequisite=blocked",
        ));
        events.last_mut().unwrap().id = 4;
        assert_eq!(
            RunAggregate::replay("task-1", &events).terminal_decision(),
            RunTerminalDecision::Succeeded
        );
    }

    #[test]
    fn restricted_tool_contract_rejects_out_of_set_operation_during_replay() {
        let mut events = vec![
            contract_with_response_and_tools(
                Vec::new(),
                Some(Box::new(RequestResponseContract::ExactText {
                    success_text: "synthetic".to_string(),
                    source_message_hash: "synthetic-hash".to_string(),
                })),
                vec!["write_file".to_string()],
            ),
            call("unexpected", "terminal"),
            result(
                "unexpected",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }

        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(
            aggregate
                .invariant_violations
                .contains(&"dispatched_operation_outside_allowed_tool_set".to_string())
        );
        assert!(aggregate.projected_success_response().is_none());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Failed);
    }

    #[test]
    fn uncontracted_operation_cannot_unlock_a_success_response() {
        let response = RequestResponseContract::ExactText {
            success_text: "phase=synthetic; outcome=complete".to_string(),
            source_message_hash: "synthetic-hash".to_string(),
        };
        let mut events = vec![
            contract_with_response(Vec::new(), Some(Box::new(response))),
            call("untracked", "terminal"),
            result(
                "untracked",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics {
                    effect: ToolCallEffect::Observation,
                    ..ToolCallSemantics::default()
                },
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(!aggregate.work_is_fulfilled());
        assert!(aggregate.projected_success_response().is_none());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
    }

    #[test]
    fn completed_operations_close_an_exhausted_contract_by_evidence() {
        // The producer compiled a wrong contract: it expects a negative exit
        // from `terminal`, but the single bounded invocation succeeded. The
        // work happened; the contract cannot credit it. Evidence must beat the
        // contract here instead of reporting the completed work as failed.
        let mut events = vec![
            contract(vec![requirement("terminal", 1)]),
            call("run", "terminal"),
            result(
                "run",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics {
                    effect: ToolCallEffect::Observation,
                    ..ToolCallSemantics::default()
                },
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.satisfied_count(), 0);
        assert!(aggregate.evidence_closed());
        assert!(!aggregate.work_is_fulfilled());
        assert_eq!(aggregate.projected_success_response(), None);
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::SucceededByEvidence
        );
        assert_eq!(aggregate.lifecycle_phase(), TaskKernelPhase::WorkSucceeded);
    }

    #[test]
    fn closeout_asks_only_for_reachable_expectations() {
        // Two obligations: a write that was refused by policy (credited) and
        // an observation the contract still wants.
        let mut denial_requirement = requirement("write_file", 1);
        {
            let predicate = denial_requirement.receipt.as_mut().unwrap();
            predicate.outcome_condition =
                Some(crate::traits::RequestedOutcomeCondition::NonSuccessTerminal);
            predicate.exit_codes.clear();
        }
        let observe = crate::traits::RequestEvidenceRequirement {
            summary: "Dependent path remains absent".to_string(),
            acceptable_scopes: vec![crate::traits::ToolSemanticScope::LocalWorkspace],
            purpose: crate::traits::EvidencePurpose::CurrentState,
            minimum_authority: crate::traits::EvidenceAuthority::Direct,
            temporal_scope: crate::traits::EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        };
        let mut events = vec![
            contract(vec![denial_requirement, observe]),
            call("attempt", "write_file"),
            result(
                "attempt",
                "write_file",
                ToolOutcomeStatus::Blocked,
                0,
                ToolCallSemantics::default(),
            ),
        ];
        events[2].data["receipt"]["invocation_stage"] =
            serde_json::json!("rejected_before_dispatch");
        events[2].data["receipt"]["exit_code"] = serde_json::Value::Null;
        events[2].data["receipt"]["receipt_kind"] = serde_json::json!("generic");
        events[2].data["receipt"]["access_denial"] = serde_json::json!({
            "reason_code": "negative_completion_contract:mutation_forbidden",
            "enforcement": "controller_enforced"
        });
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);

        // Authority still admits a read-only observation: the observation is
        // reachable, so the loop may ask for exactly that obligation.
        let decision = aggregate.closeout(|_| true);
        assert_eq!(
            decision,
            CloseoutDecision::Reachable {
                obligation_ids: vec!["task:task-1/obligation:evidence:1".to_string()],
            }
        );
        assert!(decision.work_reachable());

        // Authority admits nothing ("if denied, stop"): the credited denial
        // closes the run; nothing is demanded.
        let decision = aggregate.closeout(|_| false);
        assert_eq!(
            decision,
            CloseoutDecision::Closed {
                proof_basis: "credited_denial"
            }
        );
        assert!(!decision.work_reachable());

        // An expectation whose bound operation was denied is never reachable
        // even when its tool is admissible: retrying cannot cross the boundary.
        let mut uncredited = events.clone();
        let mut expected_success = requirement("write_file", 0);
        {
            let predicate = expected_success.receipt.as_mut().unwrap();
            predicate.outcome_condition = Some(crate::traits::RequestedOutcomeCondition::Succeeded);
            predicate.exit_codes.clear();
        }
        uncredited[0] = contract(vec![expected_success]);
        uncredited[0].id = 1;
        let aggregate = RunAggregate::replay("task-1", &uncredited);
        assert_eq!(
            aggregate.closeout(|_| true),
            CloseoutDecision::Unreachable {
                blocked: vec![(
                    "task:task-1/obligation:evidence:0".to_string(),
                    UnreachableReason::PolicyDenied
                )],
            }
        );
    }

    #[test]
    fn closeout_ignores_delivery_and_attributes_a_denied_mutation() {
        // Live shape: "attempt one protected write; if denied stop". The
        // contract expects a mutation (an Achieve obligation), credits the
        // denial through a non_success_terminal predicate, and carries a
        // response contract (a Deliver obligation). Only the denial receipt
        // exists. Nothing is reachable; the run closes on the credited denial.
        let mut denial_requirement = requirement("write_file", 1);
        {
            let predicate = denial_requirement.receipt.as_mut().unwrap();
            predicate.outcome_condition =
                Some(crate::traits::RequestedOutcomeCondition::NonSuccessTerminal);
            predicate.exit_codes.clear();
        }
        let contract_event = event(
            EventType::TaskContractCompiled,
            TaskContractCompiledData {
                schema_version: TaskContractCompiledData::SCHEMA_VERSION,
                task_id: "task-1".to_string(),
                contract: RequestCompletionContract {
                    scope_task_id: Some("task-1".to_string()),
                    adopted_from_task_ids: Vec::new(),
                    task_kind: crate::traits::RequestTaskKind::Change,
                    expects_mutation: true,
                    required_mutation_effects: ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    forbids_mutation: false,
                    forbids_tool_use: false,
                    allowed_tool_names: Vec::new(),
                    forbidden_tool_scopes: Vec::new(),
                    dispatch_stop_rules: Vec::new(),
                    required_response_fields: Vec::new(),
                    response_contract: Some(Box::new(RequestResponseContract::ExactText {
                        success_text: "phase=SYNTHETIC-RR; protected_write=denied".to_string(),
                        source_message_hash: "synthetic-hash".to_string(),
                    })),
                    forbidden_actions: Vec::new(),
                    requires_observation: true,
                    requires_reverification_after_mutation: false,
                    explicit_verification_requested: true,
                    minimum_sources: 0,
                    requires_primary_sources: false,
                    requires_exact_history: false,
                    evidence_requirements: vec![denial_requirement],
                    adopted_evidence_bindings: Vec::new(),
                    verification_targets: Vec::new(),
                },
                required_invocations: Vec::new(),
            },
        );
        let mut events = vec![
            contract_event,
            call("attempt", "write_file"),
            result(
                "attempt",
                "write_file",
                ToolOutcomeStatus::Blocked,
                0,
                ToolCallSemantics::mutation(),
            ),
        ];
        events[2].data["receipt"]["invocation_stage"] =
            serde_json::json!("rejected_before_dispatch");
        events[2].data["receipt"]["exit_code"] = serde_json::Value::Null;
        events[2].data["receipt"]["receipt_kind"] = serde_json::json!("generic");
        events[2].data["receipt"]["access_denial"] = serde_json::json!({
            "reason_code": "controller_scope_contract_rejected",
            "enforcement": "controller_enforced"
        });
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(
            aggregate
                .obligations
                .contains_key("task:task-1/obligation:mutation:local_workspace_write")
        );
        assert!(
            aggregate
                .obligations
                .contains_key("task:task-1/obligation:deliver:response")
        );
        assert!(aggregate.terminal_policy_denial());
        let decision = aggregate.closeout(|_| true);
        assert_eq!(
            decision,
            CloseoutDecision::Closed {
                proof_basis: "credited_denial"
            },
            "{decision:?}"
        );
    }

    #[test]
    fn uncredited_denial_of_the_only_observation_closes_by_policy_denial() {
        // "Read /etc/hosts" compiled as a plain observation requirement (no
        // predicate anticipating refusal). The one attempt is refused by the
        // protected host-data boundary. The refused operation would have
        // produced exactly the evidence the obligation needs, so the ledger
        // binds them: the obligation is unreachable and the run closes at
        // the authority boundary rather than lingering "pending".
        let observe = crate::traits::RequestEvidenceRequirement {
            summary: "Observe the current state of one exact resource".to_string(),
            acceptable_scopes: vec![crate::traits::ToolSemanticScope::LocalWorkspace],
            purpose: crate::traits::EvidencePurpose::CurrentState,
            minimum_authority: crate::traits::EvidenceAuthority::Direct,
            temporal_scope: crate::traits::EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        };
        // Live rejection receipts carry no observed semantics (nothing ran);
        // the typed denial names the evidence the refused call proposed.
        let mut events = vec![
            contract(vec![observe]),
            call("read", "terminal"),
            result(
                "read",
                "terminal",
                ToolOutcomeStatus::Blocked,
                0,
                ToolCallSemantics::default(),
            ),
        ];
        events[2].data["receipt"]["invocation_stage"] =
            serde_json::json!("rejected_before_dispatch");
        events[2].data["receipt"]["exit_code"] = serde_json::Value::Null;
        events[2].data["receipt"]["receipt_kind"] = serde_json::json!("generic");
        events[2].data["receipt"]["semantics"] = serde_json::json!({});
        events[2].data["receipt"]["access_denial"] = serde_json::json!({
            "reason_code": "protected_host_data",
            "enforcement": "controller_enforced",
            "proposed_evidence": [{
                "scope": "local_workspace",
                "purposes": ["current_state"],
                "authority": "direct",
                "temporal_scope": "current"
            }]
        });
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.terminal_policy_denial());
        assert!(!aggregate.evidence_closed(), "the denial credited nothing");
        assert_eq!(
            aggregate.closeout(|_| true),
            CloseoutDecision::Unreachable {
                blocked: vec![(
                    "task:task-1/obligation:evidence:0".to_string(),
                    UnreachableReason::PolicyDenied
                )],
            }
        );
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::ClosedByPolicyDenial
        );
        assert_eq!(aggregate.lifecycle_phase(), TaskKernelPhase::Partial);

        // A refusal of an unrelated capability does not bind: the obligation
        // stays reachable and the run stays pending.
        let mut unrelated = events.clone();
        unrelated[2].data["receipt"]["access_denial"]["proposed_evidence"] = serde_json::json!([{
            "scope": "external_remote",
            "purposes": ["current_state"],
            "authority": "direct",
            "temporal_scope": "current"
        }]);
        let aggregate = RunAggregate::replay("task-1", &unrelated);
        assert!(aggregate.closeout(|_| true).work_reachable());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
    }

    #[test]
    fn closeout_reports_spent_budgets_and_missing_tools() {
        let mut events = vec![
            contract(vec![requirement("terminal", 0)]),
            call("run", "terminal"),
            result(
                "run",
                "terminal",
                ToolOutcomeStatus::CompletedWithNegativeResult,
                1,
                ToolCallSemantics::default(),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        // max_invocations=1 is spent by the negative result: unreachable.
        assert_eq!(
            aggregate.closeout(|_| true),
            CloseoutDecision::Unreachable {
                blocked: vec![(
                    "task:task-1/obligation:evidence:0".to_string(),
                    UnreachableReason::CardinalityExhausted
                )],
            }
        );
        // Fresh contract, no operations, but authority admits no tool.
        let fresh = vec![{
            let mut event = contract(vec![requirement("terminal", 0)]);
            event.id = 1;
            event
        }];
        let aggregate = RunAggregate::replay("task-1", &fresh);
        assert_eq!(
            aggregate.closeout(|_| false),
            CloseoutDecision::Unreachable {
                blocked: vec![(
                    "task:task-1/obligation:evidence:0".to_string(),
                    UnreachableReason::NoAdmissibleTool
                )],
            }
        );
        assert!(aggregate.closeout(|_| true).work_reachable());
    }

    #[test]
    fn credited_policy_denial_closes_the_run_by_evidence() {
        // "Attempt one write; if denied, stop." The contract credits the
        // denial (non_success_terminal) and also carries an observation
        // obligation the request forbids performing after the denial. The
        // denial is the terminal observation; evidence closes the run.
        let mut denial_requirement = requirement("write_file", 1);
        {
            let predicate = denial_requirement.receipt.as_mut().unwrap();
            predicate.outcome_condition =
                Some(crate::traits::RequestedOutcomeCondition::NonSuccessTerminal);
            // A refusal has no process exit code; the request did not
            // predetermine the boundary that produces the refusal.
            predicate.exit_codes.clear();
        }
        let mut events = vec![
            contract(vec![
                denial_requirement,
                crate::traits::RequestEvidenceRequirement {
                    summary: "Dependent path remains absent".to_string(),
                    acceptable_scopes: vec![crate::traits::ToolSemanticScope::LocalWorkspace],
                    purpose: crate::traits::EvidencePurpose::CurrentState,
                    minimum_authority: crate::traits::EvidenceAuthority::Direct,
                    temporal_scope: crate::traits::EvidenceTemporalScope::Current,
                    required_content_markers: Vec::new(),
                    receipt: None,
                    target: None,
                },
            ]),
            call("attempt", "write_file"),
            result(
                "attempt",
                "write_file",
                ToolOutcomeStatus::Blocked,
                0,
                ToolCallSemantics::default(),
            ),
        ];
        events[2].data["receipt"]["invocation_stage"] =
            serde_json::json!("rejected_before_dispatch");
        events[2].data["receipt"]["exit_code"] = serde_json::Value::Null;
        events[2].data["receipt"]["receipt_kind"] = serde_json::json!("generic");
        events[2].data["receipt"]["access_denial"] = serde_json::json!({
            "reason_code": "negative_completion_contract:mutation_forbidden",
            "enforcement": "controller_enforced"
        });
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.terminal_policy_denial());
        assert!(
            aggregate.evidence_closed(),
            "credited denial must close evidence"
        );
        assert!(!aggregate.is_fulfilled());
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::SucceededByEvidence
        );

        // The same denial that NO obligation anticipated stays pending: the
        // request expected the write to succeed.
        let mut expected_success = requirement("write_file", 0);
        {
            let predicate = expected_success.receipt.as_mut().unwrap();
            predicate.outcome_condition = Some(crate::traits::RequestedOutcomeCondition::Succeeded);
            predicate.exit_codes.clear();
        }
        let mut unexpected = events.clone();
        unexpected[0] = contract(vec![expected_success]);
        unexpected[0].id = 1;
        let aggregate = RunAggregate::replay("task-1", &unexpected);
        assert!(aggregate.terminal_policy_denial());
        assert!(!aggregate.evidence_closed());
        assert_ne!(
            aggregate.terminal_decision(),
            RunTerminalDecision::SucceededByEvidence
        );
    }

    #[test]
    fn evidence_closure_requires_every_terminal_receipt_to_be_credited_or_successful() {
        // A pre-dispatch denial that no obligation expected is not evidence of
        // completed work, even when a sibling operation succeeded.
        let mut events = vec![
            contract(vec![requirement("terminal", 1)]),
            call("run", "terminal"),
            result(
                "run",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics {
                    effect: ToolCallEffect::Observation,
                    ..ToolCallSemantics::default()
                },
            ),
            call("write", "write_file"),
            result(
                "write",
                "write_file",
                ToolOutcomeStatus::Blocked,
                -1,
                ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(!aggregate.evidence_closed());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Failed);

        // A proposal that never produced a terminal receipt is not closed work.
        let mut dangling = vec![
            contract(vec![requirement("terminal", 1)]),
            call("run", "terminal"),
            result(
                "run",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics {
                    effect: ToolCallEffect::Observation,
                    ..ToolCallSemantics::default()
                },
            ),
            call("second", "terminal"),
        ];
        for (index, event) in dangling.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        assert!(!RunAggregate::replay("task-1", &dangling).evidence_closed());

        // Zero operations are never evidence; a contract with no work stays
        // pending rather than succeeding by absence.
        let mut empty = vec![contract(vec![requirement("terminal", 1)])];
        empty[0].id = 1;
        let aggregate = RunAggregate::replay("task-1", &empty);
        assert!(!aggregate.evidence_closed());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
    }

    #[test]
    fn evidence_closure_does_not_override_a_pending_contract() {
        // One of two required invocations happened. The contract is still
        // achievable, so the run stays pending and the loop may continue.
        let mut events = vec![
            contract(vec![
                requirement("terminal", 0),
                requirement("read_file", 0),
            ]),
            call("run", "terminal"),
            result(
                "run",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics {
                    effect: ToolCallEffect::Observation,
                    ..ToolCallSemantics::default()
                },
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.evidence_closed());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
    }

    #[test]
    fn bounded_failed_invocation_is_terminal_without_an_llm_finalizer() {
        let mut events = vec![
            contract(vec![requirement("write_file", 0)]),
            call("write", "write_file"),
            result(
                "write",
                "write_file",
                ToolOutcomeStatus::FailedPermanent,
                1,
                ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        assert_eq!(
            RunAggregate::replay("task-1", &events).terminal_decision(),
            RunTerminalDecision::Failed
        );
    }

    #[test]
    fn typed_pre_dispatch_denials_are_terminal_for_the_ledger() {
        let mut denied = result(
            "call-denied",
            "write_file",
            ToolOutcomeStatus::Blocked,
            0,
            ToolCallSemantics::default(),
        );
        denied.data["receipt"]["invocation_stage"] = serde_json::json!("rejected_before_dispatch");
        denied.data["receipt"]["access_denial"] = serde_json::json!({
            "reason_code": "negative_completion_contract",
            "enforcement": "controller_enforced"
        });
        let aggregate = RunAggregate::replay("task-1", &[denied.clone()]);
        assert_eq!(aggregate.policy_denied_operations(), 1);
        assert_eq!(aggregate.dispatched_operations(), 0);
        assert!(aggregate.terminal_policy_denial());

        // A pre-dispatch block WITHOUT a typed denial (a deferral, a budget
        // stop, a retry hint) is not a policy boundary.
        let mut deferred = denied.clone();
        deferred.data["tool_call_id"] = serde_json::json!("call-deferred");
        deferred.data["receipt"]["access_denial"] = serde_json::Value::Null;
        let aggregate = RunAggregate::replay("task-1", &[deferred]);
        assert_eq!(aggregate.policy_denied_operations(), 0);
        assert!(!aggregate.terminal_policy_denial());

        // Once anything actually dispatched, the denial is no longer the
        // run's terminal observation.
        let ran = result(
            "call-ran",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics::default(),
        );
        let aggregate = RunAggregate::replay("task-1", &[denied, ran]);
        assert_eq!(aggregate.policy_denied_operations(), 1);
        assert!(!aggregate.terminal_policy_denial());
    }

    #[test]
    fn replay_is_idempotent_for_duplicate_result_event() {
        let contract = contract(vec![requirement("terminal", 0)]);
        let call = call("run", "terminal");
        let result = result(
            "run",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics {
                effect: ToolCallEffect::Observation,
                ..ToolCallSemantics::default()
            },
        );
        let mut events = vec![contract, call, result.clone(), result];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.satisfied_count(), 1);
        assert_eq!(aggregate.operations.len(), 1);
    }

    #[test]
    fn typed_replay_reuses_proof_without_consuming_cardinality() {
        let obligation = "task:task-1/obligation:evidence:0";
        let source_call = claimed_call("run", "terminal", "operation:run", &[obligation]);
        let source_result = result(
            "run",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics::observation(),
        );
        let lineage = crate::ToolOperationLineage::DurableReplay {
            source_operation_id: "run".to_string(),
            source_result_id: "result:run".to_string(),
        };
        let replay_call = event(
            EventType::ToolCall,
            ToolCallData::from_tool_call(
                "run-replay",
                "terminal",
                json!({}),
                Some("task-1".to_string()),
            )
            .with_policy_metadata(Some("operation:run".to_string()), None, None)
            .with_kernel_claim("operation:run", vec![obligation.to_string()], 1, 1)
            .with_operation_lineage(Some(lineage.clone())),
        );
        let mut replay_result = result(
            "run-replay",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics::observation(),
        );
        replay_result.data["receipt"]["invocation_stage"] = json!("replayed");
        replay_result.data["receipt"]["outcome_evidence"] = json!("durable_replay");
        replay_result.data["receipt"]["result_provenance"]["result_id"] = json!("result:run");
        replay_result.data["receipt"]["idempotency_key"] = json!("operation:run");

        let mut source_events = vec![
            contract(vec![requirement("terminal", 0)]),
            source_call,
            source_result,
        ];
        for (index, event) in source_events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let source = RunAggregate::replay("task-1", &source_events);
        assert_eq!(
            source.admit_operation(&TaskKernelOperationClaim {
                operation_id: "run-replay".to_string(),
                stable_operation_key: "operation:run".to_string(),
                tool_name: "terminal".to_string(),
                obligation_ids: vec![obligation.to_string()],
                max_attempts: 1,
                max_invocations: 1,
                idempotency_key: Some("operation:run".to_string()),
                operation_lineage: Some(lineage),
            }),
            TaskKernelAdmission::Admitted
        );

        source_events.extend([replay_call, replay_result]);
        for (index, event) in source_events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let replayed = RunAggregate::replay("task-1", &source_events);
        assert_eq!(replayed.cardinality_violations, 0);
        assert_eq!(replayed.satisfied_count(), 1);
        assert_eq!(replayed.operations.len(), 2);
    }

    #[test]
    fn adopted_parent_receipt_credits_the_bound_child_obligation() {
        // A background continuation child inherits the parent's invocation
        // requirement. The adopted durable replay of the parent's terminal
        // receipt still names the *parent's* obligation id, both in the call
        // claim and in the receipt's completion ids; the child contract's
        // adopted binding is the typed bridge to the child's own obligation.
        let parent_obligation = "task:parent-1/obligation:evidence:0";
        let child_obligation = "task:task-1/obligation:evidence:0";
        let inherited = requirement("terminal", 0);
        let mut contract = contract(vec![inherited.clone()]);
        contract.data["contract"]["adopted_from_task_ids"] = json!(["parent-1"]);
        contract.data["contract"]["adopted_evidence_bindings"] = json!([{
            "source_obligation_id": parent_obligation,
            "requirement": inherited,
        }]);
        contract.data["contract"]["expects_mutation"] = json!(true);
        contract.data["contract"]["required_mutation_effects"] =
            json!(ToolMutationEffects::PROCESS_STATE.bits());

        let lineage = crate::ToolOperationLineage::DurableReplay {
            source_operation_id: "run".to_string(),
            source_result_id: "result:run".to_string(),
        };
        let adopted_call = event(
            EventType::ToolCall,
            ToolCallData::from_tool_call("run", "terminal", json!({}), Some("task-1".to_string()))
                .with_policy_metadata(Some("operation:run".to_string()), None, None)
                .with_kernel_claim("operation:run", vec![parent_obligation.to_string()], 1, 1)
                .with_operation_lineage(Some(lineage)),
        );
        // The terminal receipt of a backgrounded process carries the
        // process-state mutation the launch declared.
        let mut adopted_result = result(
            "run",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics::observation_and_mutation_with(ToolMutationEffects::PROCESS_STATE),
        );
        adopted_result.data["receipt"]["completion_obligation_ids"] = json!([parent_obligation]);
        adopted_result.data["receipt"]["result_provenance"]["result_id"] = json!("result:run");

        let mut events = vec![contract, adopted_call, adopted_result];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.invariant_violations.is_empty());
        assert_eq!(
            aggregate.obligations[child_obligation].state,
            RunObligationState::Satisfied,
            "adopted receipt must close the child's bound invocation obligation"
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:mutation:process_state"].state,
            RunObligationState::Satisfied,
            "the completed background process proves the inherited process-state mutation"
        );
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::Succeeded
        );
        assert_eq!(
            aggregate.closeout(|_| true),
            CloseoutDecision::Closed {
                proof_basis: "contract"
            }
        );
        // The replay stays idempotent and never advances the child's effect
        // revision: the mutation happened in the parent's lifecycle.
        assert_eq!(aggregate.effect_revision, 0);
    }

    #[test]
    fn later_mutation_invalidates_current_observation_not_historical_accomplishment() {
        let current = RequestEvidenceRequirement {
            summary: "synthetic current state".to_string(),
            acceptable_scopes: vec![crate::traits::ToolSemanticScope::HostLocal],
            purpose: EvidencePurpose::CurrentState,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        };
        let mut observation = result(
            "observe",
            "system_info",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics {
                effect: ToolCallEffect::Observation,
                ..ToolCallSemantics::default()
            }
            .with_evidence(vec![ToolEvidenceCapability::new(
                ToolSemanticScope::HostLocal,
                &[EvidencePurpose::CurrentState],
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            )]),
        );
        observation.data["receipt"]["completion_obligation_ids"] =
            json!(["task:task-1/obligation:evidence:0"]);
        let mut events = vec![
            contract(vec![current]),
            call("observe", "system_info"),
            observation,
            call("mutate", "write_file"),
            result(
                "mutate",
                "write_file",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }

        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.satisfied_count(), 0);
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
        assert!(aggregate.obligations.values().any(|obligation| {
            obligation.class == RunObligationClass::Observe
                && obligation.state == RunObligationState::Invalidated
        }));
    }

    #[test]
    fn durable_claims_allocate_overlapping_obligations_without_cross_satisfaction() {
        let obligation_zero = "task:task-1/obligation:evidence:0";
        let obligation_one = "task:task-1/obligation:evidence:1";
        let mut events = vec![
            contract(vec![requirement("terminal", 0), requirement("terminal", 0)]),
            claimed_call(
                "phase-1",
                "terminal",
                "operation:phase-1",
                &[obligation_zero],
            ),
            result(
                "phase-1",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }

        let first = RunAggregate::replay("task-1", &events);
        assert_eq!(first.satisfied_count(), 1);
        assert_eq!(first.terminal_decision(), RunTerminalDecision::Pending);
        assert_eq!(
            first.admit_operation(&TaskKernelOperationClaim {
                operation_id: "phase-1-retry".to_string(),
                stable_operation_key: "operation:phase-1".to_string(),
                tool_name: "terminal".to_string(),
                obligation_ids: vec![obligation_zero.to_string()],
                max_attempts: 1,
                max_invocations: 1,
                idempotency_key: None,
                operation_lineage: None,
            }),
            TaskKernelAdmission::Rejected {
                code: "operation_invocations_exhausted",
                detail: "The durable operation has reached its 1-invocation ceiling.".to_string(),
            }
        );
        assert_eq!(
            first.admit_operation(&TaskKernelOperationClaim {
                operation_id: "phase-2".to_string(),
                stable_operation_key: "operation:phase-2".to_string(),
                tool_name: "terminal".to_string(),
                obligation_ids: vec![obligation_one.to_string()],
                max_attempts: 1,
                max_invocations: 1,
                idempotency_key: None,
                operation_lineage: None,
            }),
            TaskKernelAdmission::Admitted
        );

        events.push(claimed_call(
            "phase-2",
            "terminal",
            "operation:phase-2",
            &[obligation_one],
        ));
        events.push(result(
            "phase-2",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics::observation(),
        ));
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        assert_eq!(
            RunAggregate::replay("task-1", &events).terminal_decision(),
            RunTerminalDecision::Succeeded
        );
    }

    #[test]
    fn response_preparation_and_transport_ack_are_distinct_lifecycle_states() {
        let response = RequestResponseContract::ExactText {
            success_text: "synthetic complete".to_string(),
            source_message_hash: "synthetic-hash".to_string(),
        };
        let mut events = vec![
            event(
                EventType::TaskStart,
                TaskStartData {
                    task_id: "task-1".to_string(),
                    description: "synthetic lifecycle".to_string(),
                    parent_task_id: None,
                    user_message: None,
                    turn_id: None,
                    spawned_by: None,
                },
            ),
            contract_with_response(Vec::new(), Some(Box::new(response))),
            assistant_response("response-1", "synthetic complete"),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        assert_eq!(
            RunAggregate::replay("task-1", &events).lifecycle_phase(),
            TaskKernelPhase::ResponsePrepared
        );

        events.push(event(
            EventType::ResponseDelivery,
            ResponseDeliveryData {
                response_id: "response-1".to_string(),
                task_id: "task-1".to_string(),
                turn_id: None,
                platform: "synthetic".to_string(),
                state: ResponseDeliveryState::PlatformAcknowledged,
                platform_message_ids: vec!["platform-1".to_string()],
                error_code: None,
                occurred_at: "2026-01-01T00:00:00Z".to_string(),
            },
        ));
        events.last_mut().unwrap().id = 4;
        assert_eq!(
            RunAggregate::replay("task-1", &events).lifecycle_phase(),
            TaskKernelPhase::Delivered
        );
    }

    fn executor_expectations(items: Vec<ExecutorExpectationItem>) -> Event {
        event(
            EventType::ExecutorExpectationsDeclared,
            ExecutorExpectationsDeclaredData {
                schema_version: ExecutorExpectationsDeclaredData::SCHEMA_VERSION,
                task_id: "task-1".to_string(),
                items,
                outcome: None,
                short_of: None,
            },
        )
    }

    fn checklist_item(
        index: usize,
        effects: ToolMutationEffects,
        requires_observation: bool,
        targets: &[&str],
        status: &str,
    ) -> ExecutorExpectationItem {
        ExecutorExpectationItem {
            index,
            description: format!("step {index}"),
            requires_observation,
            mutation_effects: effects,
            targets: targets.iter().map(|t| t.to_string()).collect(),
            observation_targets: Vec::new(),
            unbound_targets: Vec::new(),
            depends_on: Vec::new(),
            status: status.to_string(),
        }
    }

    fn objective_observation_target(
        objective_id: &str,
        facets: &[ToolSemanticFacet],
        collection_id: Option<&str>,
    ) -> RequestObservationTarget {
        RequestObservationTarget {
            subject: crate::traits::RequestVerificationTarget::new(
                RequestVerificationTargetKind::ResourceId,
                objective_id,
            )
            .unwrap(),
            facets: facets.to_vec(),
            collection_coverage: collection_id.map(|collection_id| {
                crate::traits::RequestCollectionCoverageRequirement {
                    collection: crate::traits::RequestVerificationTarget::new(
                        RequestVerificationTargetKind::ResourceId,
                        collection_id,
                    )
                    .unwrap(),
                    minimum_completeness: crate::traits::ToolCollectionCompleteness::Complete,
                }
            }),
        }
    }

    fn with_exact_observations(
        mut event: Event,
        observations: Vec<ToolObservationEvidence>,
        collections: Vec<ToolCollectionObservation>,
    ) -> Event {
        event.data["receipt"]["observations"] = serde_json::to_value(observations).unwrap();
        event.data["receipt"]["collection_observations"] =
            serde_json::to_value(collections).unwrap();
        event
    }

    fn objective_observation(
        objective_id: &str,
        facets: &[ToolSemanticFacet],
    ) -> ToolObservationEvidence {
        ToolObservationEvidence::new(
            ToolTargetHint::new(ToolTargetHintKind::ResourceId, objective_id).unwrap(),
            facets,
        )
    }

    /// Mirrors the live shape: the runtime fills terminal receipts with
    /// direct, current local-workspace capabilities.
    fn terminal_observation_semantics(path: Option<&str>) -> ToolCallSemantics {
        let mut semantics =
            ToolCallSemantics::observation().with_evidence(vec![ToolEvidenceCapability {
                scope: ToolSemanticScope::LocalWorkspace,
                purposes: vec![EvidencePurpose::CurrentState, EvidencePurpose::Outcome],
                authority: EvidenceAuthority::Direct,
                temporal_scope: EvidenceTemporalScope::Current,
            }]);
        if let Some(path) = path {
            semantics = semantics.with_target_hint(crate::traits::ToolTargetHintKind::Path, path);
        }
        semantics
    }

    fn workspace_write_semantics(path: &str) -> ToolCallSemantics {
        ToolCallSemantics {
            effect: ToolCallEffect::Mutation,
            mutation_effects: ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
            ..ToolCallSemantics::default()
        }
        .with_target_hint(crate::traits::ToolTargetHintKind::Path, path)
    }

    fn renumber(events: &mut [Event]) {
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
    }

    /// The agent's own `blocked` caps the run below success: receipts that
    /// cover every declared item close nothing, and the phase is partial.
    #[test]
    fn claimed_blocked_is_never_upgraded_by_covering_receipts() {
        let declared = |outcome| {
            let mut event = executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::NONE,
                true,
                &["/tmp/proj/routes.yml"],
                "completed",
            )]);
            event.data["outcome"] = serde_json::json!(outcome);
            event
        };
        let read = [
            call("routes", "read_file"),
            result(
                "routes",
                "read_file",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(Some("/tmp/proj/routes.yml")),
            ),
        ];
        for (outcome, unfinished) in [("done", false), ("blocked", true)] {
            let mut events = vec![declared(outcome)];
            events.extend(read.clone());
            renumber(&mut events);
            let aggregate = RunAggregate::replay("task-1", &events);
            assert_eq!(aggregate.executor_claimed_unfinished(), unfinished);
            assert_eq!(aggregate.evidence_closed(), !unfinished, "{outcome}");
            assert_eq!(
                aggregate.lifecycle_phase() == TaskKernelPhase::Partial,
                unfinished,
                "{outcome}: {:?}",
                aggregate.lifecycle_phase()
            );
        }
    }

    /// The resource a run ran out of is kept only with an unfinished
    /// verdict: a `done` never carries a reason it stopped.
    #[test]
    fn the_resource_a_run_ran_out_of_comes_with_an_unfinished_verdict() {
        for (outcome, kept) in [("blocked", true), ("partial", true), ("done", false)] {
            let mut event = executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::NONE,
                true,
                &["/tmp/proj/routes.yml"],
                "pending",
            )]);
            event.data["outcome"] = serde_json::json!(outcome);
            event.data["short_of"] = serde_json::json!("docker_disk_space");
            let aggregate = RunAggregate::replay("task-1", &[event]);
            assert_eq!(
                aggregate.executor_short_of,
                kept.then_some(crate::ShortResource::DockerDiskSpace),
                "{outcome}"
            );
        }
    }

    /// An item that observes two resources is done when each was observed,
    /// by separate calls (two reads, or two sub-agents' reads); one read
    /// leaves it owed.
    #[test]
    fn multi_target_observation_item_is_closed_by_one_read_per_target() {
        let item = || {
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::NONE,
                true,
                &["/tmp/proj/routes.yml", "/tmp/proj/tests"],
                "pending",
            )])
        };
        let read = |id: &str, path: &str| {
            [
                call(id, "read_file"),
                result(
                    id,
                    "read_file",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    terminal_observation_semantics(Some(path)),
                ),
            ]
        };
        let obligation = "task:task-1/obligation:checklist:0";

        let mut one = vec![item()];
        one.extend(read("routes", "/tmp/proj/routes.yml"));
        renumber(&mut one);
        assert_eq!(
            RunAggregate::replay("task-1", &one).obligations[obligation].state,
            RunObligationState::Pending,
            "the tests folder was never observed"
        );

        let mut both = vec![item()];
        both.extend(read("routes", "/tmp/proj/routes.yml"));
        both.extend(read("tests", "/tmp/proj/tests"));
        renumber(&mut both);
        let aggregate = RunAggregate::replay("task-1", &both);
        assert_eq!(
            aggregate.obligations[obligation].state,
            RunObligationState::Satisfied
        );
        let mut proof = aggregate.obligations[obligation]
            .satisfying_receipt_ids
            .clone();
        proof.sort();
        assert_eq!(proof, vec!["routes".to_string(), "tests".to_string()]);
    }

    /// A checklist item scoped to a folder is done by writes to files in
    /// it; an item bound to one file is not done by a write to its parent.
    #[test]
    fn folder_scoped_write_item_is_closed_by_a_write_inside_the_folder() {
        let mut events = vec![
            executor_expectations(vec![
                checklist_item(
                    0,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &["/tmp/proj/calc"],
                    "pending",
                ),
                checklist_item(
                    1,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &["/tmp/proj/calc/other.py"],
                    "pending",
                ),
            ]),
            call("edit", "edit_file"),
            result(
                "edit",
                "edit_file",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/proj/calc/calc.py"),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:1"].state,
            RunObligationState::Pending,
            "a different file in the folder is still owed"
        );
        let _ = &mut events;
    }

    /// "Inspect, then edit, then verify": the edit (which depends on the
    /// inspection) must not reopen the inspection, while the verification,
    /// which nothing depends on, still needs a read after the edit.
    #[test]
    fn prerequisite_inspection_is_not_staled_by_the_work_that_depends_on_it() {
        let path = "/tmp/proj/calc.py";
        let declare = |statuses: [&str; 3]| {
            let inspect = checklist_item(0, ToolMutationEffects::NONE, true, &[path], statuses[0]);
            let mut edit = checklist_item(
                1,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &[path],
                statuses[1],
            );
            edit.depends_on = vec![0];
            let mut verify =
                checklist_item(2, ToolMutationEffects::NONE, true, &[path], statuses[2]);
            verify.depends_on = vec![1];
            executor_expectations(vec![inspect, edit, verify])
        };
        let mut events = vec![
            declare(["in_progress", "pending", "pending"]),
            call("read", "read_file"),
            result(
                "read",
                "read_file",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(Some(path)),
            ),
            call("edit", "edit_file"),
            result(
                "edit",
                "edit_file",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics(path),
            ),
            declare(["completed", "completed", "in_progress"]),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied,
            "the edit that builds on the inspection does not make it stale"
        );
        match aggregate.closeout(|_| true) {
            CloseoutDecision::Reachable { obligation_ids } => assert_eq!(
                obligation_ids,
                vec!["task:task-1/obligation:checklist:2".to_string()],
                "only the post-edit verification is still owed"
            ),
            other => panic!("expected reachable, got {other:?}"),
        }

        events.push(call("verify", "read_file"));
        events.push(result(
            "verify",
            "read_file",
            ToolOutcomeStatus::Succeeded,
            0,
            terminal_observation_semantics(Some(path)),
        ));
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.open_executor_expectations(), 0);
        assert!(aggregate.evidence_closed());
    }

    #[test]
    fn executor_declared_items_become_obligations_closed_only_by_receipts() {
        // The model declares three typed steps with no assessor contract at
        // all. Each write closes only the item bound to its path; the
        // observation stays reachable, so the run is neither fulfilled nor
        // evidence-closed even though every receipt succeeded.
        let mut events = vec![
            executor_expectations(vec![
                checklist_item(
                    0,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &["/tmp/a"],
                    "pending",
                ),
                checklist_item(
                    1,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &["/tmp/a/f"],
                    "pending",
                ),
                checklist_item(2, ToolMutationEffects::NONE, true, &[], "pending"),
                // Free text without typed content is never an obligation.
                checklist_item(3, ToolMutationEffects::NONE, false, &[], "pending"),
            ]),
            call("mk", "terminal"),
            result(
                "mk",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/a"),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.executor_expectations_present);
        assert!(!aggregate.contract_present);
        assert!(
            !aggregate
                .obligations
                .contains_key("task:task-1/obligation:checklist:3")
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied
        );
        // One write does not close every "write" item: the second is bound
        // to a different path.
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:1"].state,
            RunObligationState::Pending
        );
        assert_eq!(aggregate.open_executor_expectations(), 2);
        assert!(!aggregate.is_fulfilled());
        assert!(!aggregate.evidence_closed());
        match aggregate.closeout(|_| true) {
            CloseoutDecision::Reachable { obligation_ids } => {
                assert_eq!(
                    obligation_ids,
                    vec![
                        "task:task-1/obligation:checklist:1".to_string(),
                        "task:task-1/obligation:checklist:2".to_string()
                    ]
                );
            }
            other => panic!("expected reachable, got {other:?}"),
        }

        events.push(call("wr", "terminal"));
        events.push(result(
            "wr",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            workspace_write_semantics("/tmp/a/f"),
        ));
        events.push(call("ls", "terminal"));
        events.push(result(
            "ls",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            terminal_observation_semantics(None),
        ));
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.open_executor_expectations(), 0);
        assert!(aggregate.is_fulfilled());
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::Succeeded
        );
    }

    #[test]
    fn destructive_checklist_item_closes_on_destructive_receipt_not_on_creation() {
        // Mirrors the live create/remove workflow: the executor declares the
        // removal step `destructive` on the same path it earlier created. The
        // creation receipt touching that path must NOT close the removal
        // obligation, and a deletion receipt typed DESTRUCTIVE (the terminal
        // adapter types `rm`/`rmdir`/`unlink` syntax) must.
        let destructive_semantics = ToolCallSemantics {
            effect: ToolCallEffect::Mutation,
            mutation_effects: ToolMutationEffects::LOCAL_WORKSPACE_WRITE
                .union(ToolMutationEffects::DESTRUCTIVE),
            ..ToolCallSemantics::default()
        }
        .with_target_hint(crate::traits::ToolTargetHintKind::Path, "/tmp/card");
        let mut events = vec![
            executor_expectations(vec![
                checklist_item(
                    0,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &["/tmp/card"],
                    "pending",
                ),
                checklist_item(
                    1,
                    ToolMutationEffects::DESTRUCTIVE,
                    false,
                    &["/tmp/card"],
                    "pending",
                ),
            ]),
            call("mk", "terminal"),
            result(
                "mk",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/card"),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:1"].state,
            RunObligationState::Pending,
            "a creation receipt on the same path must not prove the removal"
        );

        events.push(call("rm", "terminal"));
        events.push(result(
            "rm",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            destructive_semantics,
        ));
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:1"].state,
            RunObligationState::Satisfied
        );
        assert_eq!(aggregate.open_executor_expectations(), 0);
        assert!(aggregate.is_fulfilled());
    }

    #[test]
    fn executor_redeclaration_keeps_proof_and_defers_without_fulfilling() {
        let mut events = vec![
            executor_expectations(vec![
                checklist_item(
                    0,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &[],
                    "pending",
                ),
                checklist_item(1, ToolMutationEffects::NONE, true, &[], "pending"),
                checklist_item(2, ToolMutationEffects::NONE, true, &["/tmp/x"], "pending"),
            ]),
            call("w", "write_file"),
            result(
                "w",
                "write_file",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/w"),
            ),
            // The model re-posts its list: item 0 marked completed (proof is
            // retained from the receipt, not from the label), item 1 deferred,
            // item 2 dropped entirely.
            executor_expectations(vec![
                checklist_item(
                    0,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &[],
                    "completed",
                ),
                checklist_item(1, ToolMutationEffects::NONE, true, &[], "deferred"),
            ]),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:1"].state,
            RunObligationState::Deferred
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:2"].state,
            RunObligationState::Deferred
        );
        // Deferral is not evidence: the items stay unmet work.
        assert_eq!(aggregate.open_executor_expectations(), 2);
        assert_eq!(aggregate.required_count(), 3);
        assert_eq!(aggregate.satisfied_count(), 1);
        assert!(!aggregate.is_fulfilled());
        assert!(!aggregate.evidence_closed());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
        // Stopping is still permitted: the closeout never demands a deferred
        // item, it reports it as blocked by the executor's own deferral.
        assert_eq!(
            aggregate.closeout(|_| true),
            CloseoutDecision::Unreachable {
                blocked: vec![
                    (
                        "task:task-1/obligation:checklist:1".to_string(),
                        UnreachableReason::ExecutorDeferred
                    ),
                    (
                        "task:task-1/obligation:checklist:2".to_string(),
                        UnreachableReason::ExecutorDeferred
                    ),
                ],
            }
        );
    }

    /// The vacuous case: declare work, defer all of it, run nothing. Before
    /// deferral counted as satisfaction this replayed as `Succeeded`.
    #[test]
    fn deferring_every_item_without_receipts_never_succeeds() {
        let mut events = vec![
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &[],
                "pending",
            )]),
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &[],
                "deferred",
            )]),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.operations.is_empty());
        assert!(!aggregate.is_fulfilled());
        assert!(!aggregate.work_is_fulfilled());
        assert!(aggregate.projected_success_response().is_none());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
    }

    /// Successful receipts for other work cannot evidence-close a run past
    /// a deferred checklist item (the `SucceededByEvidence` path).
    #[test]
    fn deferred_items_block_evidence_closure() {
        let mut events = vec![
            executor_expectations(vec![
                checklist_item(
                    0,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &["/tmp/report"],
                    "pending",
                ),
                checklist_item(
                    1,
                    ToolMutationEffects::NONE,
                    true,
                    &["/tmp/sent"],
                    "pending",
                ),
            ]),
            call("w", "write_file"),
            result(
                "w",
                "write_file",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/report"),
            ),
            executor_expectations(vec![
                checklist_item(
                    0,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &["/tmp/report"],
                    "completed",
                ),
                checklist_item(
                    1,
                    ToolMutationEffects::NONE,
                    true,
                    &["/tmp/sent"],
                    "deferred",
                ),
            ]),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(
            aggregate
                .operations
                .values()
                .all(|op| op.outcome == Some(ToolOutcomeStatus::Succeeded))
        );
        assert!(!aggregate.evidence_closed());
        assert_ne!(
            aggregate.terminal_decision(),
            RunTerminalDecision::SucceededByEvidence
        );
        assert_ne!(
            aggregate.terminal_decision(),
            RunTerminalDecision::Succeeded
        );
    }

    /// Receipts decide achievement in both directions: an item proven by a
    /// receipt stays satisfied when re-posted as deferred or dropped, and a
    /// deferred item a later receipt proves becomes satisfied.
    #[test]
    fn proof_outranks_the_deferred_label() {
        let declare = |status: &str| {
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &["/tmp/out"],
                status,
            )])
        };
        let write = || {
            vec![
                call("w", "write_file"),
                result(
                    "w",
                    "write_file",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    workspace_write_semantics("/tmp/out"),
                ),
            ]
        };

        let mut proven_then_deferred = vec![declare("pending")];
        proven_then_deferred.extend(write());
        proven_then_deferred.push(declare("deferred"));
        renumber(&mut proven_then_deferred);
        let aggregate = RunAggregate::replay("task-1", &proven_then_deferred);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied
        );
        assert!(aggregate.is_fulfilled());

        let mut deferred_then_proven = vec![declare("deferred")];
        deferred_then_proven.extend(write());
        renumber(&mut deferred_then_proven);
        let aggregate = RunAggregate::replay("task-1", &deferred_then_proven);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied
        );
        assert!(aggregate.is_fulfilled());
    }

    /// An item the executor tried and could not achieve (`failed`) is not
    /// demanded again and never counts as done, like a deferred one.
    #[test]
    fn a_failed_item_is_not_demanded_and_not_done() {
        let mut events = vec![executor_expectations(vec![checklist_item(
            0,
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
            false,
            &["/tmp/out"],
            "failed",
        )])];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Deferred
        );
        assert!(!aggregate.is_fulfilled());
    }

    /// Serialized projections from before the rename still load.
    #[test]
    fn abandoned_projections_deserialize_as_deferred() {
        let state: RunObligationState = serde_json::from_str("\"abandoned\"").unwrap();
        assert_eq!(state, RunObligationState::Deferred);
    }

    #[test]
    fn executor_path_bound_observation_needs_a_receipt_for_that_path() {
        let declared = executor_expectations(vec![checklist_item(
            0,
            ToolMutationEffects::NONE,
            true,
            &["/tmp/target.txt"],
            "pending",
        )]);
        let mut wrong = vec![
            declared.clone(),
            call("cat", "terminal"),
            result(
                "cat",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(Some("/tmp/other.txt")),
            ),
        ];
        for (index, event) in wrong.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        assert_eq!(
            RunAggregate::replay("task-1", &wrong).open_executor_expectations(),
            1
        );

        let mut right = vec![
            declared,
            call("cat", "terminal"),
            result(
                "cat",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(Some("/tmp/target.txt")),
            ),
        ];
        for (index, event) in right.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        assert_eq!(
            RunAggregate::replay("task-1", &right).open_executor_expectations(),
            0
        );
    }

    #[test]
    fn executor_completed_label_does_not_prove_an_item_without_a_receipt() {
        // Marking an item "completed" is a claim; only a receipt proves it.
        let mut events = vec![executor_expectations(vec![checklist_item(
            0,
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
            false,
            &[],
            "completed",
        )])];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.open_executor_expectations(), 1);
        assert!(!aggregate.is_fulfilled());
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
    }

    #[test]
    fn contract_arriving_after_executor_declaration_keeps_checklist_obligations() {
        let mut events = vec![
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &[],
                "pending",
            )]),
            contract(vec![requirement("terminal", 0)]),
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(
            aggregate
                .obligations
                .contains_key("task:task-1/obligation:checklist:0")
        );
        assert!(
            aggregate
                .obligations
                .contains_key("task:task-1/obligation:evidence:0")
        );
    }

    #[test]
    fn a_project_scope_root_hint_proves_a_path_bound_item_under_it() {
        // Terminal receipts carry `write_roots`/`read_roots` as ProjectScope
        // hints: `mkdir -p /tmp/x` declared with write_roots=[/tmp/x].
        let declared = executor_expectations(vec![
            checklist_item(
                0,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &["/tmp/x"],
                "pending",
            ),
            checklist_item(1, ToolMutationEffects::NONE, true, &["/tmp/x"], "pending"),
        ]);
        let root_hint = |semantics: ToolCallSemantics| {
            semantics.with_target_hint(crate::traits::ToolTargetHintKind::ProjectScope, "/tmp/x")
        };
        let mut events = vec![
            declared,
            call("mk", "terminal"),
            result(
                "mk",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                root_hint(ToolCallSemantics {
                    effect: ToolCallEffect::ObservationAndMutation,
                    mutation_effects: ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    ..ToolCallSemantics::default()
                }),
            ),
            call("chk", "terminal"),
            result(
                "chk",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                root_hint(terminal_observation_semantics(None)),
            ),
        ];
        renumber(&mut events);
        assert_eq!(
            RunAggregate::replay("task-1", &events).open_executor_expectations(),
            0
        );
    }

    #[test]
    fn a_typed_source_write_proves_a_workspace_write_item() {
        let mut events = vec![
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &["/tmp/x/f.txt"],
                "pending",
            )]),
            call("w", "write_file"),
            result(
                "w",
                "write_file",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics {
                    effect: ToolCallEffect::Mutation,
                    mutation_effects: ToolMutationEffects::LOCAL_SOURCE_WRITE,
                    ..ToolCallSemantics::default()
                }
                .with_target_hint(crate::traits::ToolTargetHintKind::Path, "/tmp/x/f.txt"),
            ),
        ];
        renumber(&mut events);
        assert_eq!(
            RunAggregate::replay("task-1", &events).open_executor_expectations(),
            0
        );
    }

    #[test]
    fn opaque_checklist_target_never_degrades_to_an_unrelated_observation() {
        let mut events = vec![
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::NONE,
                true,
                &["scheduled_task_state:synthetic_blog"],
                "pending",
            )]),
            result(
                "read-other",
                "manage_mandates",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        let obligation = &aggregate.obligations["task:task-1/obligation:checklist:0"];
        assert_eq!(obligation.state, RunObligationState::Pending);
        assert_eq!(
            obligation.observation_targets[0].subject.kind,
            RequestVerificationTargetKind::ResourceId
        );
    }

    #[test]
    fn legacy_executor_events_keep_their_pre_resource_id_replay_semantics() {
        let mut declaration = executor_expectations(vec![checklist_item(
            0,
            ToolMutationEffects::NONE,
            true,
            &["legacy-display-label"],
            "pending",
        )]);
        declaration.data["schema_version"] = serde_json::json!(1);
        let mut events = vec![
            declaration,
            result(
                "legacy-read",
                "system_info",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied,
            "schema-v1 history must not be reclassified by the additive exact-target protocol"
        );
    }

    #[test]
    fn exact_objective_subject_and_facets_prevent_cross_object_credit() {
        let requested_facets = [
            ToolSemanticFacet::Schedule,
            ToolSemanticFacet::RunState,
            ToolSemanticFacet::Recovery,
        ];
        let mut item = checklist_item(0, ToolMutationEffects::NONE, true, &[], "pending");
        item.observation_targets = vec![objective_observation_target(
            "objective:goal-a",
            &requested_facets,
            None,
        )];
        let declaration = executor_expectations(vec![item]);
        let wrong = with_exact_observations(
            result(
                "read-b",
                "manage_mandates",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
            vec![objective_observation("objective:goal-b", &requested_facets)],
            Vec::new(),
        );
        let mut wrong_events = vec![declaration.clone(), wrong];
        renumber(&mut wrong_events);
        assert_eq!(
            RunAggregate::replay("task-1", &wrong_events).obligations
                ["task:task-1/obligation:checklist:0"]
                .state,
            RunObligationState::Pending
        );

        let right = with_exact_observations(
            result(
                "read-a",
                "scheduled_goal_runs",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
            vec![objective_observation("objective:goal-a", &requested_facets)],
            Vec::new(),
        );
        let mut right_events = vec![declaration, right];
        renumber(&mut right_events);
        assert_eq!(
            RunAggregate::replay("task-1", &right_events).obligations
                ["task:task-1/obligation:checklist:0"]
                .state,
            RunObligationState::Satisfied
        );
    }

    #[test]
    fn complete_collection_coverage_proves_absence_but_partial_coverage_does_not() {
        let collection_id = "objective_collection:scheduled_goals";
        let facets = [ToolSemanticFacet::Schedule];
        let mut item = checklist_item(0, ToolMutationEffects::NONE, true, &[], "pending");
        item.observation_targets = vec![objective_observation_target(
            "objective:goal-missing",
            &facets,
            Some(collection_id),
        )];
        let declaration = executor_expectations(vec![item]);
        let collection_hint =
            ToolTargetHint::new(ToolTargetHintKind::ResourceId, collection_id).unwrap();
        let partial = with_exact_observations(
            result(
                "partial",
                "scheduled_goal_runs",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
            Vec::new(),
            vec![ToolCollectionObservation {
                collection: collection_hint.clone(),
                facets: facets.to_vec(),
                completeness: crate::traits::ToolCollectionCompleteness::Partial,
                returned_count: 1,
                total_count: Some(2),
                members: vec![
                    ToolTargetHint::new(ToolTargetHintKind::ResourceId, "objective:goal-other")
                        .unwrap(),
                ],
            }],
        );
        let mut partial_events = vec![declaration.clone(), partial];
        renumber(&mut partial_events);
        assert_eq!(
            RunAggregate::replay("task-1", &partial_events).obligations
                ["task:task-1/obligation:checklist:0"]
                .state,
            RunObligationState::Pending
        );

        let complete = with_exact_observations(
            result(
                "complete",
                "scheduled_goal_runs",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
            Vec::new(),
            vec![ToolCollectionObservation::complete(
                collection_hint,
                &facets,
                vec![
                    ToolTargetHint::new(ToolTargetHintKind::ResourceId, "objective:goal-other")
                        .unwrap(),
                ],
            )],
        );
        let mut complete_events = vec![declaration, complete];
        renumber(&mut complete_events);
        assert_eq!(
            RunAggregate::replay("task-1", &complete_events).obligations
                ["task:task-1/obligation:checklist:0"]
                .state,
            RunObligationState::Satisfied
        );
    }

    /// Regression for the 2026-09-01 audit runs: every receipt succeeded, but
    /// the model bound one checklist item to an invented credential-state ID
    /// no adapter reports. That obligation stayed pending and reachable, so
    /// the run closed `partial`. Binding checklist targets against the stable
    /// subjects tools actually advertise turns the invented ID into a
    /// recorded, untargeted observation that the credential read satisfies.

    #[test]
    fn mutation_of_a_covered_collection_invalidates_prior_absence_proof() {
        let collection_id = "objective_collection:scheduled_goals";
        let facets = [ToolSemanticFacet::Schedule];
        let mut item = checklist_item(0, ToolMutationEffects::NONE, true, &[], "pending");
        item.observation_targets = vec![objective_observation_target(
            "objective:goal-missing",
            &facets,
            Some(collection_id),
        )];
        let collection_hint =
            ToolTargetHint::new(ToolTargetHintKind::ResourceId, collection_id).unwrap();
        let complete_absence = with_exact_observations(
            result(
                "complete",
                "scheduled_goal_runs",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
            Vec::new(),
            vec![ToolCollectionObservation::complete(
                collection_hint,
                &facets,
                Vec::new(),
            )],
        );
        let collection_mutation = result(
            "create-schedule",
            "manage_goal",
            ToolOutcomeStatus::Succeeded,
            0,
            ToolCallSemantics::mutation_with(ToolMutationEffects::CONFIGURATION)
                .with_target_hint(ToolTargetHintKind::ResourceId, collection_id),
        );
        let mut events = vec![
            executor_expectations(vec![item]),
            complete_absence,
            collection_mutation,
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Invalidated
        );
    }

    #[test]
    fn retargeting_a_checklist_index_does_not_inherit_the_old_subjects_proof() {
        let facets = [ToolSemanticFacet::Control];
        let mut item_a = checklist_item(0, ToolMutationEffects::NONE, true, &[], "pending");
        item_a.observation_targets = vec![objective_observation_target(
            "objective:goal-a",
            &facets,
            None,
        )];
        let mut item_b = item_a.clone();
        item_b.observation_targets = vec![objective_observation_target(
            "objective:goal-b",
            &facets,
            None,
        )];
        let mut events = vec![
            executor_expectations(vec![item_a]),
            with_exact_observations(
                result(
                    "read-a",
                    "manage_mandates",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    ToolCallSemantics::observation(),
                ),
                vec![objective_observation("objective:goal-a", &facets)],
                Vec::new(),
            ),
            executor_expectations(vec![item_b]),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        let obligation = &aggregate.obligations["task:task-1/obligation:checklist:0"];
        assert_eq!(obligation.state, RunObligationState::Pending);
        assert!(obligation.satisfying_receipt_ids.is_empty());
    }

    #[test]
    fn upstream_completion_id_cannot_bypass_an_exact_observation_selector() {
        let facets = [ToolSemanticFacet::Measurement];
        let mut item = checklist_item(0, ToolMutationEffects::NONE, true, &[], "pending");
        item.observation_targets = vec![objective_observation_target(
            "objective:goal-a",
            &facets,
            None,
        )];
        let mut wrong = with_exact_observations(
            result(
                "read-b",
                "manage_mandates",
                ToolOutcomeStatus::Succeeded,
                0,
                ToolCallSemantics::observation(),
            ),
            vec![objective_observation("objective:goal-b", &facets)],
            Vec::new(),
        );
        wrong.data["receipt"]["completion_obligation_ids"] =
            serde_json::json!(["task:task-1/obligation:checklist:0"]);
        let mut events = vec![executor_expectations(vec![item]), wrong];
        renumber(&mut events);
        assert_eq!(
            RunAggregate::replay("task-1", &events).obligations
                ["task:task-1/obligation:checklist:0"]
                .state,
            RunObligationState::Pending
        );
    }

    #[test]
    fn a_refusal_on_a_named_verification_target_closes_the_run_at_that_boundary() {
        // "Attempt exactly one protected write to P; expected to be denied;
        // stop immediately." The assessor compiled an untargeted observation
        // ("the exact resource named by the request") with P as a
        // verification target. The denied write touched P, so the
        // observation is bound to the boundary and the run closes there.
        let mut contract_event = contract(vec![RequestEvidenceRequirement {
            summary: "Observe the current state of one exact resource".to_string(),
            acceptable_scopes: vec![ToolSemanticScope::LocalWorkspace],
            purpose: EvidencePurpose::CurrentState,
            minimum_authority: EvidenceAuthority::Direct,
            temporal_scope: EvidenceTemporalScope::Current,
            required_content_markers: Vec::new(),
            receipt: None,
            target: None,
        }]);
        contract_event.data["contract"]["verification_targets"] =
            serde_json::json!([{"kind": "path", "value": "/private/var/root/denied.txt"}]);
        let mut events = vec![
            contract_event,
            call("w", "write_file"),
            result(
                "w",
                "write_file",
                ToolOutcomeStatus::Blocked,
                0,
                ToolCallSemantics::mutation().with_target_hint(
                    crate::traits::ToolTargetHintKind::Path,
                    "/private/var/root/denied.txt",
                ),
            ),
        ];
        events[2].data["receipt"]["invocation_stage"] =
            serde_json::json!("rejected_before_dispatch");
        events[2].data["receipt"]["exit_code"] = serde_json::Value::Null;
        events[2].data["receipt"]["receipt_kind"] = serde_json::json!("generic");
        events[2].data["receipt"]["access_denial"] = serde_json::json!({
            "reason_code": "controller_scope_contract_rejected",
            "enforcement": "controller_enforced"
        });
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.terminal_policy_denial());
        assert_eq!(
            aggregate.terminal_decision(),
            RunTerminalDecision::ClosedByPolicyDenial
        );

        // A refusal on an unrelated path does not bind: the run stays pending.
        events[2].data["receipt"]["semantics"]["target_hints"] =
            serde_json::json!([{"kind": "Path", "value": "/private/var/root/other.txt"}]);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Pending);
    }

    fn contract_requiring_effects(effects: ToolMutationEffects) -> Event {
        let mut contract_event = contract(Vec::new());
        contract_event.data["contract"]["expects_mutation"] = serde_json::json!(true);
        contract_event.data["contract"]["required_mutation_effects"] =
            serde_json::json!(effects.bits());
        contract_event
    }

    fn url_observation_semantics(url: &str) -> ToolCallSemantics {
        ToolCallSemantics::observation()
            .with_target_hint(crate::traits::ToolTargetHintKind::Url, url)
    }

    /// Live-shape regression (blog run f535f390, 2026-08-31): a repository's
    /// documented deploy IS a shell command, and the shell adapter can never
    /// carry remote effects in its own receipt. The remote obligations close
    /// on the operator-grade pair — succeeded mutation, then an independently
    /// declared URL observation verified afterward — and stay open without it.
    #[test]
    fn remote_effect_obligations_close_on_post_mutation_url_verification() {
        let url = "https://blog.example.test/posts/synthetic-post/";
        let base = vec![
            contract_requiring_effects(
                ToolMutationEffects::REMOTE_DEPLOY.union(ToolMutationEffects::EXTERNAL_DELIVERY),
            ),
            executor_expectations(vec![
                checklist_item(
                    0,
                    ToolMutationEffects::REMOTE_DEPLOY,
                    false,
                    &["/tmp/blog"],
                    "pending",
                ),
                checklist_item(1, ToolMutationEffects::NONE, true, &[url], "pending"),
            ]),
            call("deploy", "terminal"),
            result(
                "deploy",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/blog"),
            ),
        ];

        // Without the URL verification the adapter's local receipt proves
        // nothing remote: every remote obligation stays open.
        let mut events = base.clone();
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        for id in [
            "task:task-1/obligation:checklist:0",
            "task:task-1/obligation:mutation:remote_deploy",
            "task:task-1/obligation:mutation:external_delivery",
        ] {
            assert_eq!(
                aggregate.obligations[id].state,
                RunObligationState::Pending,
                "{id} must stay open without a post-mutation URL verification"
            );
        }

        let mut events = base;
        events.push(call("verify", "http_request"));
        events.push(result(
            "verify",
            "http_request",
            ToolOutcomeStatus::Succeeded,
            0,
            url_observation_semantics(url),
        ));
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        for id in [
            "task:task-1/obligation:checklist:0",
            "task:task-1/obligation:checklist:1",
            "task:task-1/obligation:mutation:remote_deploy",
            "task:task-1/obligation:mutation:external_delivery",
        ] {
            assert_eq!(
                aggregate.obligations[id].state,
                RunObligationState::Satisfied,
                "{id} must close once the URL observation verified the deploy"
            );
        }
    }

    /// Live-shape regression: the executor researched two external pages and
    /// only afterwards declared the checklist item naming those URLs as its
    /// targets. Proof must not depend on declaration order, and a later
    /// unrelated local write must not stale an observation of a disjoint
    /// external target.
    #[test]
    fn checklist_targets_declared_after_the_receipt_still_credit() {
        let url = "https://example.test/observances/synthetic-day";
        let mut events = vec![
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::NONE,
                true,
                &[],
                "pending",
            )]),
            call("research", "web_fetch"),
            result(
                "research",
                "web_fetch",
                ToolOutcomeStatus::Succeeded,
                0,
                url_observation_semantics(url),
            ),
            call("write", "write_file"),
            result(
                "write",
                "write_file",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/blog/post.md"),
            ),
            // The item gains its exact researched targets only after the
            // receipts landed — the executor's natural tick-after-work flow.
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::NONE,
                true,
                &[url],
                "completed",
            )]),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied,
            "the earlier research receipt must credit the later-targeted item"
        );
    }

    #[test]
    fn mutation_invalidates_only_observations_of_state_it_could_have_changed() {
        let url = "https://blog.example.test/posts/synthetic-post/";
        let mut events = vec![
            executor_expectations(vec![
                checklist_item(0, ToolMutationEffects::NONE, true, &[url], "pending"),
                checklist_item(
                    1,
                    ToolMutationEffects::NONE,
                    true,
                    &["/tmp/blog/post.md"],
                    "pending",
                ),
            ]),
            call("verify", "http_request"),
            result(
                "verify",
                "http_request",
                ToolOutcomeStatus::Succeeded,
                0,
                url_observation_semantics(url),
            ),
            call("read", "read_file"),
            result(
                "read",
                "read_file",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(Some("/tmp/blog/post.md")),
            ),
            call("write", "write_file"),
            result(
                "write",
                "write_file",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/blog/post.md"),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied,
            "a local write cannot stale an observation of a disjoint URL"
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:1"].state,
            RunObligationState::Invalidated,
            "a write to the observed path must stale that observation"
        );
    }

    fn effect_semantics(
        effects: ToolMutationEffects,
        hint: Option<(crate::traits::ToolTargetHintKind, &str)>,
    ) -> ToolCallSemantics {
        let semantics = ToolCallSemantics {
            effect: ToolCallEffect::ObservationAndMutation,
            mutation_effects: effects,
            ..ToolCallSemantics::default()
        };
        match hint {
            Some((kind, value)) => semantics.with_target_hint(kind, value),
            None => semantics,
        }
    }

    /// The shape of a real run: build a site, send a screenshot of it, start
    /// a long-lived tunnel and check its public URL. The executor names each
    /// item's target only when it ticks the item, the site is written by
    /// several commands, the screenshot tool delivers its own capture, the
    /// tunnel is still running (its success state) when the run ends, and
    /// its URL is checked by a shell command whose receipt names no URL.
    #[test]
    fn site_setup_checklist_closes_on_the_receipts_of_the_work_that_did_it() {
        const SITE: &str = "/tmp/proj/site";
        const SHOT: &str = "/tmp/inbox/shot.png";
        const TUNNEL: &str = "https://synthetic-tunnel.example/";
        let site_effects =
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE.union(ToolMutationEffects::PROCESS_STATE);
        let items = |site: &[&str], shot: &[&str], tunnel: &[&str]| {
            executor_expectations(vec![
                checklist_item(0, site_effects, false, site, "completed"),
                checklist_item(
                    1,
                    ToolMutationEffects::EXTERNAL_DELIVERY,
                    false,
                    shot,
                    "completed",
                ),
                checklist_item(
                    2,
                    ToolMutationEffects::PROCESS_STATE,
                    false,
                    tunnel,
                    "completed",
                ),
            ])
        };
        let mut events = vec![
            items(&[], &[], &[]),
            call("install", "terminal"),
            result(
                "install",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                workspace_write_semantics("/tmp/proj/site/wp-config.php"),
            ),
            items(&[SITE], &[], &[]),
            call("shot", "browser"),
            result(
                "shot",
                "browser",
                ToolOutcomeStatus::Succeeded,
                0,
                effect_semantics(
                    ToolMutationEffects::EXTERNAL_DELIVERY,
                    Some((crate::traits::ToolTargetHintKind::Path, SHOT)),
                ),
            ),
            items(&[SITE], &[SHOT], &[]),
            call("tunnel", "terminal"),
            result(
                "tunnel",
                "terminal",
                ToolOutcomeStatus::Backgrounded,
                0,
                effect_semantics(
                    site_effects,
                    Some((
                        crate::traits::ToolTargetHintKind::Path,
                        "/tmp/proj/site/tunnel.log",
                    )),
                ),
            ),
            call("probe", "terminal"),
            result(
                "probe",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(None),
            ),
            items(&[SITE], &[SHOT], &[TUNNEL]),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        for index in 0..3 {
            assert_eq!(
                aggregate.obligations[&format!("task:task-1/obligation:checklist:{index}")].state,
                RunObligationState::Satisfied,
                "checklist item {index}"
            );
        }
        assert_eq!(aggregate.open_executor_expectations(), 0);
    }

    /// Naming the target of a proven item keeps its proof; a target the
    /// proving receipts contradict, or a move to another target, does not.
    #[test]
    fn narrowing_a_proven_effect_item_keeps_its_proof_and_retargeting_reopens_it() {
        let item = |targets: &[&str]| {
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                targets,
                "completed",
            )])
        };
        let state = |declarations: Vec<Event>| {
            let mut events = vec![
                item(&[]),
                call("write", "write_file"),
                result(
                    "write",
                    "write_file",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    workspace_write_semantics("/tmp/proj/a.txt"),
                ),
            ];
            events.extend(declarations);
            renumber(&mut events);
            RunAggregate::replay("task-1", &events).obligations
                ["task:task-1/obligation:checklist:0"]
                .state
        };
        assert_eq!(
            state(vec![item(&["/tmp/proj"])]),
            RunObligationState::Satisfied,
            "the write was proven before the item named its folder"
        );
        assert_eq!(
            state(vec![item(&["https://synthetic-site.example/"])]),
            RunObligationState::Satisfied,
            "a file write says nothing against a URL the item now names"
        );
        assert_eq!(
            state(vec![item(&["/tmp/elsewhere/b.txt"])]),
            RunObligationState::Pending,
            "the proving write was to another file"
        );
        assert_eq!(
            state(vec![
                item(&["/tmp/proj/a.txt"]),
                item(&["/tmp/elsewhere/b.txt"])
            ]),
            RunObligationState::Pending,
            "moving a targeted item to another target is a different item"
        );
    }

    /// A launch receipt proves that a process was started and nothing else:
    /// the files the command will write are not written yet.
    #[test]
    fn background_launch_proves_process_state_but_not_the_command_s_writes() {
        let mut events = vec![
            executor_expectations(vec![
                checklist_item(0, ToolMutationEffects::PROCESS_STATE, false, &[], "pending"),
                checklist_item(
                    1,
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    false,
                    &[],
                    "pending",
                ),
            ]),
            call("build", "terminal"),
            result(
                "build",
                "terminal",
                ToolOutcomeStatus::Backgrounded,
                0,
                effect_semantics(
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE
                        .union(ToolMutationEffects::PROCESS_STATE),
                    None,
                ),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:1"].state,
            RunObligationState::Pending,
            "the build has not finished writing"
        );
    }

    /// A launch that was never declared as a process-state change (a status
    /// check on a running process) proves nothing.
    #[test]
    fn background_receipt_without_a_process_effect_proves_nothing() {
        let mut events = vec![
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::PROCESS_STATE,
                false,
                &[],
                "pending",
            )]),
            call("check", "terminal"),
            result(
                "check",
                "terminal",
                ToolOutcomeStatus::Backgrounded,
                0,
                terminal_observation_semantics(None),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Pending
        );
    }

    /// The live Drupal helper (2026-11-13): it launched `ddev start`, looked
    /// at it while it ran, and the launch's completion ended it. The check
    /// is a finished observation, so nothing is left in flight once the
    /// launch ended, and the run reads as having finished its work.
    #[test]
    fn checking_a_running_launch_leaves_nothing_in_flight_once_it_ends() {
        let mut events = vec![
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::PROCESS_STATE,
                false,
                &[],
                "completed",
            )]),
            call("start", "terminal"),
            result(
                "start",
                "terminal",
                ToolOutcomeStatus::Backgrounded,
                0,
                effect_semantics(ToolMutationEffects::PROCESS_STATE, None),
            ),
            call("check", "terminal"),
            result(
                "check",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(None),
            ),
        ];
        renumber(&mut events);
        assert!(
            RunAggregate::replay("task-1", &events).has_background_in_flight(),
            "the launch runs until its completion"
        );
        events.push(result(
            "start",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            effect_semantics(ToolMutationEffects::PROCESS_STATE, None),
        ));
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(!aggregate.has_background_in_flight());
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Satisfied
        );
    }

    /// An effect whose receipts do not touch the item's target is credited
    /// only when the target was observed after the work, never by an
    /// observation that predates it or by one of another target.
    #[test]
    fn effect_away_from_its_target_needs_a_later_observation_of_that_target() {
        const TUNNEL: &str = "https://synthetic-tunnel.example/";
        let item = || {
            executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::PROCESS_STATE,
                false,
                &[TUNNEL],
                "completed",
            )])
        };
        let launch = || {
            [
                call("tunnel", "terminal"),
                result(
                    "tunnel",
                    "terminal",
                    ToolOutcomeStatus::Backgrounded,
                    0,
                    effect_semantics(ToolMutationEffects::PROCESS_STATE, None),
                ),
            ]
        };
        let probe = |id: &str, url: &str| {
            [
                call(id, "terminal"),
                result(
                    id,
                    "terminal",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    url_observation_semantics(url),
                ),
            ]
        };
        let state = |mut events: Vec<Event>| {
            renumber(&mut events);
            RunAggregate::replay("task-1", &events).obligations
                ["task:task-1/obligation:checklist:0"]
                .state
        };

        let unobserved = [vec![item()], launch().to_vec()].concat();
        assert_eq!(state(unobserved), RunObligationState::Pending);

        let observed_before = [
            vec![item()],
            probe("early", TUNNEL).to_vec(),
            launch().to_vec(),
        ]
        .concat();
        assert_eq!(state(observed_before), RunObligationState::Pending);

        let other_target = [
            vec![item()],
            launch().to_vec(),
            probe("other", "https://elsewhere.example/").to_vec(),
        ]
        .concat();
        assert_eq!(state(other_target), RunObligationState::Pending);

        let no_effect = [vec![item()], probe("only", TUNNEL).to_vec()].concat();
        assert_eq!(state(no_effect), RunObligationState::Pending);

        let observed_after = [
            vec![item()],
            launch().to_vec(),
            probe("late", TUNNEL).to_vec(),
        ]
        .concat();
        assert_eq!(state(observed_after), RunObligationState::Satisfied);
    }

    /// Two receipts that each carry part of an item's effects prove it
    /// together, but only receipts on the item's target count.
    #[test]
    fn several_receipts_on_the_target_jointly_cover_a_multi_effect_item() {
        let both =
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE.union(ToolMutationEffects::CONFIGURATION);
        let config = |path: &str| {
            effect_semantics(
                ToolMutationEffects::CONFIGURATION,
                Some((crate::traits::ToolTargetHintKind::Path, path)),
            )
        };
        let run = |config_path: &str| {
            let mut events = vec![
                executor_expectations(vec![checklist_item(
                    0,
                    both,
                    false,
                    &["/tmp/proj/site"],
                    "pending",
                )]),
                call("write", "write_file"),
                result(
                    "write",
                    "write_file",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    workspace_write_semantics("/tmp/proj/site/index.php"),
                ),
                call("config", "terminal"),
                result(
                    "config",
                    "terminal",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    config(config_path),
                ),
            ];
            renumber(&mut events);
            RunAggregate::replay("task-1", &events).obligations
                ["task:task-1/obligation:checklist:0"]
                .clone()
        };
        let joint = run("/tmp/proj/site/settings.php");
        assert_eq!(joint.state, RunObligationState::Satisfied);
        assert_eq!(joint.satisfying_receipt_ids, vec!["write", "config"]);
        let source_write = |effects| {
            let mut events = vec![
                executor_expectations(vec![checklist_item(0, both, false, &[], "pending")]),
                call("edit", "edit_file"),
                result(
                    "edit",
                    "edit_file",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    effect_semantics(ToolMutationEffects::LOCAL_SOURCE_WRITE, None),
                ),
                call("config", "terminal"),
                result(
                    "config",
                    "terminal",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    effect_semantics(effects, None),
                ),
            ];
            renumber(&mut events);
            RunAggregate::replay("task-1", &events).obligations
                ["task:task-1/obligation:checklist:0"]
                .state
        };
        assert_eq!(
            source_write(ToolMutationEffects::CONFIGURATION),
            RunObligationState::Satisfied,
            "a source-file write is a workspace write"
        );
        assert_eq!(
            source_write(ToolMutationEffects::EXTERNAL_DELIVERY),
            RunObligationState::Pending
        );
        assert_eq!(
            run("/tmp/other/settings.php").state,
            RunObligationState::Pending,
            "configuration somewhere else is not this item's work"
        );
    }

    /// An effect item's demand names how its target kind is proven, never
    /// the path advice for a URL target.
    #[test]
    fn effect_item_summary_matches_its_target_kind() {
        let summary = |target: &str| {
            let mut events = vec![executor_expectations(vec![checklist_item(
                0,
                ToolMutationEffects::PROCESS_STATE,
                false,
                &[target],
                "pending",
            )])];
            renumber(&mut events);
            RunAggregate::replay("task-1", &events).obligations
                ["task:task-1/obligation:checklist:0"]
                .summary
                .clone()
                .unwrap()
        };
        let url = summary("https://synthetic-tunnel.example/");
        assert!(url.contains("names the URL"), "{url}");
        assert!(url.contains("later completed observation"), "{url}");
        assert!(!url.contains("declare the path"), "{url}");
        let path = summary("/tmp/proj/site");
        assert!(path.contains("declare the path as write access"), "{path}");
        assert!(path.contains("later completed observation"), "{path}");
    }

    /// Delivering a call's own capture to the chat is no change an
    /// observation could verify: it neither advances the effect revision nor
    /// stales an earlier observation, and it still proves a delivery item.
    /// A plain delivery, a delivery to something other than a local file
    /// and a delivery with another effect stay changes.
    #[test]
    fn delivering_an_own_capture_is_not_a_change_to_verify() {
        const SHOT: &str = "/tmp/inbox/shot.png";
        let path = crate::traits::ToolTargetHintKind::Path;
        let run = |delivery: ToolCallSemantics| {
            let mut events = vec![
                executor_expectations(vec![
                    checklist_item(0, ToolMutationEffects::NONE, true, &[], "pending"),
                    checklist_item(
                        1,
                        ToolMutationEffects::EXTERNAL_DELIVERY,
                        false,
                        &[],
                        "pending",
                    ),
                ]),
                call("look", "browser"),
                result(
                    "look",
                    "browser",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    terminal_observation_semantics(None),
                ),
                call("deliver", "browser"),
                result(
                    "deliver",
                    "browser",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    delivery,
                ),
            ];
            renumber(&mut events);
            RunAggregate::replay("task-1", &events)
        };

        let capture = run(effect_semantics(
            ToolMutationEffects::EXTERNAL_DELIVERY,
            Some((path, SHOT)),
        ));
        assert_eq!(capture.effect_revision, 0);
        assert!(!capture.operations["deliver"].changes_verifiable_state());
        assert_eq!(
            capture.obligations["task:task-1/obligation:checklist:0"].satisfied_at_revision,
            Some(0),
            "the earlier observation stays current"
        );
        assert_eq!(
            capture.obligations["task:task-1/obligation:checklist:1"].state,
            RunObligationState::Satisfied,
            "the delivery still proves the delivery item"
        );

        let send_file = ToolCallSemantics {
            effect: ToolCallEffect::Mutation,
            mutation_effects: ToolMutationEffects::EXTERNAL_DELIVERY,
            ..ToolCallSemantics::default()
        }
        .with_target_hint(path, SHOT);
        let remote = effect_semantics(
            ToolMutationEffects::EXTERNAL_DELIVERY,
            Some((
                crate::traits::ToolTargetHintKind::Url,
                "https://chat.example/",
            )),
        );
        let untargeted = effect_semantics(ToolMutationEffects::EXTERNAL_DELIVERY, None);
        let post = effect_semantics(
            ToolMutationEffects::EXTERNAL_DELIVERY.union(ToolMutationEffects::REMOTE_MUTATION),
            Some((path, SHOT)),
        );
        for (name, semantics) in [
            ("plain delivery", send_file.clone()),
            ("remote target", remote),
            ("no target", untargeted),
            ("delivery with another effect", post),
        ] {
            let aggregate = run(semantics);
            assert_eq!(aggregate.effect_revision, 1, "{name}");
            assert!(
                aggregate.operations["deliver"].changes_verifiable_state(),
                "{name}"
            );
        }
        assert_eq!(
            run(send_file).obligations["task:task-1/obligation:checklist:0"].state,
            RunObligationState::Invalidated,
            "a plain delivery still stales an untargeted observation"
        );
    }

    fn declared_with_outcome(
        items: Vec<ExecutorExpectationItem>,
        outcome: Option<crate::ExecutorClaimedOutcome>,
    ) -> Event {
        event(
            EventType::ExecutorExpectationsDeclared,
            ExecutorExpectationsDeclaredData {
                schema_version: ExecutorExpectationsDeclaredData::SCHEMA_VERSION,
                task_id: "task-1".to_string(),
                items,
                outcome,
                short_of: None,
            },
        )
    }

    /// Live 2026-11-13: "install Drupal and WordPress" was declared as two
    /// installs, the run only listed the sites a past run had made, then
    /// re-declared the installs deferred with a `blocked` verdict. Nothing
    /// was attempted: the aggregate says so from receipts alone.
    #[test]
    fn declared_work_with_only_observations_is_unattempted() {
        let install = |index, status: &str| {
            checklist_item(
                index,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &[],
                status,
            )
        };
        let mut events = vec![
            declared_with_outcome(vec![install(0, "pending"), install(1, "pending")], None),
            call("list", "terminal"),
            result(
                "list",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(None),
            ),
            declared_with_outcome(
                vec![install(0, "deferred"), install(1, "deferred")],
                Some(crate::ExecutorClaimedOutcome::Blocked),
            ),
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate.declared_work_unattempted());

        // One attempted change, even a failed one, is an attempt, but at
        // one item: the second install was still never started.
        events.push(call("install", "terminal"));
        events.push(result(
            "install",
            "terminal",
            ToolOutcomeStatus::FailedPermanent,
            1,
            workspace_write_semantics("/tmp/site"),
        ));
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(aggregate.unattempted_checklist_items(), BTreeSet::from([1]));

        events.push(call("install-2", "terminal"));
        events.push(result(
            "install-2",
            "terminal",
            ToolOutcomeStatus::Succeeded,
            0,
            workspace_write_semantics("/tmp/site-2"),
        ));
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(!aggregate.declared_work_unattempted());
    }

    /// Live 2026-11-13: Drupal and WordPress installs declared, one helper
    /// launched naming the Drupal item, and the turn handed off. The
    /// WordPress item is judged on its own: nothing took it on.
    #[test]
    fn a_launch_covers_only_the_items_it_names() {
        let install = |index, status: &str| {
            checklist_item(
                index,
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                false,
                &[],
                status,
            )
        };
        let mut launch = result(
            "spawn",
            "spawn_agent",
            ToolOutcomeStatus::Backgrounded,
            0,
            ToolCallSemantics::default(),
        );
        launch.data["receipt"]["checklist_items"] = serde_json::json!([0]);
        let mut events = vec![
            declared_with_outcome(vec![install(0, "background"), install(1, "pending")], None),
            call("spawn", "spawn_agent"),
            launch,
        ];
        renumber(&mut events);
        let aggregate = RunAggregate::replay("task-1", &events);
        assert_eq!(
            aggregate.unattempted_declared_work(),
            vec!["task:task-1/obligation:checklist:1".to_string()]
        );

        // A deferral is no typed reason; running out of a resource is.
        events.push(declared_with_outcome(
            vec![install(0, "background"), install(1, "deferred")],
            Some(crate::ExecutorClaimedOutcome::Partial),
        ));
        renumber(&mut events);
        assert!(RunAggregate::replay("task-1", &events).declared_work_unattempted());
        events.push(event(
            EventType::ExecutorExpectationsDeclared,
            ExecutorExpectationsDeclaredData {
                schema_version: ExecutorExpectationsDeclaredData::SCHEMA_VERSION,
                task_id: "task-1".to_string(),
                items: vec![install(0, "background"), install(1, "deferred")],
                outcome: Some(crate::ExecutorClaimedOutcome::Partial),
                short_of: Some(crate::ShortResource::DiskSpace),
            },
        ));
        renumber(&mut events);
        assert!(!RunAggregate::replay("task-1", &events).declared_work_unattempted());
    }

    /// A request that only observes (a status question) declares no work,
    /// so it is never held to work it did not ask for.
    #[test]
    fn an_observation_only_checklist_owes_no_work() {
        let mut events = vec![
            declared_with_outcome(
                vec![checklist_item(
                    0,
                    ToolMutationEffects::NONE,
                    true,
                    &[],
                    "pending",
                )],
                None,
            ),
            call("list", "terminal"),
            result(
                "list",
                "terminal",
                ToolOutcomeStatus::Succeeded,
                0,
                terminal_observation_semantics(None),
            ),
        ];
        renumber(&mut events);
        assert!(!RunAggregate::replay("task-1", &events).declared_work_unattempted());
    }
}
