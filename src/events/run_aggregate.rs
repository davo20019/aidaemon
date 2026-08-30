//! Deterministic request-run lifecycle reducer.
//!
//! The immutable event history is the source of truth. This aggregate is a
//! discardable projection: rebuilding it from the same ordered events must
//! always yield the same result. LLM output can propose contracts and tool
//! operations, but only typed events change lifecycle state here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    AssistantResponseData, AssistantResponseDisposition, Event, EventType,
    ExecutorExpectationsDeclaredData, ResponseDeliveryData, ResponseDeliveryState,
    TaskContractCompiledData, TaskEndData, TaskOutcome, TaskStartData, TaskStatus, ToolCallData,
    ToolResultData,
};
use crate::traits::{
    EvidenceTemporalScope, RequestDispatchStopRule, RequestEvidenceRequirement,
    RequestReceiptPredicate, RequestResponseContract, RequestVerificationTargetKind,
    ToolMutationEffects, ToolOutcomeStatus, ToolTargetHintKind,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunObligationClass {
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
pub(crate) enum RunObligationState {
    Pending,
    Satisfied,
    Invalidated,
    Abandoned,
    Unverifiable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct RunObligation {
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct RunOperation {
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
    /// The proposed operation carried mutating semantics.
    #[serde(default)]
    pub mutating: bool,
    /// Evidence the operation would have produced (from its receipt
    /// semantics), retained so a refused operation can be bound to the
    /// observation obligation it was attempting to satisfy.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_capabilities: Vec<crate::traits::ToolEvidenceCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation_lineage: Option<super::ToolOperationLineage>,
}

/// End-to-end lifecycle projection. Execution completion and user-visible
/// delivery are deliberately separate states; a generated response is not a
/// transport acknowledgement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TaskKernelPhase {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TaskKernelOperationClaim {
    pub operation_id: String,
    pub stable_operation_key: String,
    pub tool_name: String,
    #[serde(default)]
    pub obligation_ids: Vec<String>,
    pub max_attempts: usize,
    pub max_invocations: usize,
    pub idempotency_key: Option<String>,
    pub operation_lineage: Option<super::ToolOperationLineage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TaskKernelAdmission {
    Admitted,
    Rejected { code: &'static str, detail: String },
}

/// Why a pending expectation can no longer be satisfied in this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UnreachableReason {
    /// An operation bound to it was refused by a typed policy boundary.
    PolicyDenied,
    /// Its bounded invocation budget is spent.
    CardinalityExhausted,
    /// No visible tool admissible under the request's authority can satisfy it.
    NoAdmissibleTool,
}

/// The ledger-first closeout verdict. See [`RunAggregate::closeout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CloseoutDecision {
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
    pub(crate) fn verdict(&self) -> &'static str {
        match self {
            Self::Closed { .. } => "closed",
            Self::Reachable { .. } => "reachable",
            Self::Unreachable { .. } => "unreachable",
        }
    }

    /// Whether the loop may still demand more work from the model.
    pub(crate) fn work_reachable(&self) -> bool {
        matches!(self, Self::Reachable { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunTerminalDecision {
    /// Every accepted obligation has durable proof.
    Succeeded,
    /// The compiled contract is exhausted without crediting the work, but
    /// every terminal operation receipt succeeded or was credited. Evidence of
    /// completed work beats a contract that could not describe it; this is a
    /// success whose proof basis is the receipt set rather than the contract.
    SucceededByEvidence,
    /// Every open obligation was refused by a typed authority boundary before
    /// dispatch (nothing dispatched). The request could not be carried out
    /// and the run is complete at that boundary; the reply narrates it.
    ClosedByPolicyDenial,
    /// At least one exact bounded invocation is exhausted without proof.
    Failed,
    /// Work remains and the aggregate still permits progress or recovery.
    Pending,
    /// No task-owned contract was installed; preserve the caller's legacy path.
    Unspecified,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct RunAggregate {
    pub schema_version: u16,
    pub task_id: String,
    #[serde(default)]
    pub contract_present: bool,
    /// The executing model declared typed expectations of its own.
    #[serde(default)]
    pub executor_expectations_present: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_fingerprint: Option<String>,
    #[serde(default)]
    pub effect_revision: u64,
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
}

impl RunAggregate {
    pub const SCHEMA_VERSION: u16 = 4;

    pub(crate) fn new(task_id: impl Into<String>) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            task_id: task_id.into(),
            contract_present: false,
            executor_expectations_present: false,
            contract_fingerprint: None,
            effect_revision: 0,
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
        }
    }

    pub(crate) fn replay(task_id: &str, events: &[Event]) -> Self {
        let mut aggregate = Self::new(task_id);
        let mut ordered = events.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|event| event.id);
        for event in ordered {
            aggregate.apply(event);
        }
        aggregate.reconcile_cardinality();
        aggregate
    }

    pub(crate) fn apply(&mut self, event: &Event) {
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
            format!("{digest:x}")
        });
        if self.contract_present {
            if self.contract_fingerprint != fingerprint {
                self.record_invariant("conflicting_contract_installation");
            }
            return;
        }
        self.contract_present = true;
        self.contract_fingerprint = fingerprint;
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
            });
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
            });
        }
    }

    /// Compile the executing model's typed checklist into obligations. Each
    /// typed item becomes one obligation keyed by its index: a mutation item
    /// is `Achieve` (closed by a succeeded receipt carrying the effect); an
    /// observation/targeted item is `Observe` (closed by a compatible
    /// observation receipt, target-bound when a path target was declared).
    /// Untyped free-text items are not obligations. A later declaration
    /// replaces the set: kept indices retain their proof, `deferred`/`skipped`
    /// items are abandoned, removed indices are abandoned.
    fn install_executor_expectations(&mut self, declared: ExecutorExpectationsDeclaredData) {
        self.executor_expectations_present = true;
        let mut declared_ids = BTreeSet::new();
        for item in &declared.items {
            let id = format!("task:{}/obligation:checklist:{}", self.task_id, item.index);
            let abandoned = matches!(item.status.as_str(), "deferred" | "skipped");
            let has_effect = !item.mutation_effects.is_empty();
            let required_target = item.targets.iter().find_map(|target| {
                let target = target.trim();
                if target.starts_with('/') || target.starts_with("~/") {
                    Some(crate::traits::RequestVerificationTarget {
                        kind: crate::traits::RequestVerificationTargetKind::Path,
                        value: target.to_string(),
                    })
                } else if target.starts_with("http://") || target.starts_with("https://") {
                    Some(crate::traits::RequestVerificationTarget {
                        kind: crate::traits::RequestVerificationTargetKind::Url,
                        value: target.to_string(),
                    })
                } else {
                    None
                }
            });
            let (class, required_effect) = if has_effect {
                (RunObligationClass::Achieve, item.mutation_effects)
            } else if item.requires_observation || required_target.is_some() {
                (RunObligationClass::Observe, ToolMutationEffects::NONE)
            } else {
                // Free text without typed content is a note, not an obligation.
                continue;
            };
            declared_ids.insert(id.clone());
            let existing = self.obligations.get(&id);
            let existing_proof = existing
                .map(|o| o.satisfying_receipt_ids.clone())
                .unwrap_or_default();
            let existing_revision = existing.and_then(|o| o.satisfied_at_revision);
            let state = if abandoned {
                RunObligationState::Abandoned
            } else if existing.is_some_and(|o| o.state == RunObligationState::Satisfied) {
                RunObligationState::Satisfied
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
                required_target,
            });
        }
        for (id, obligation) in self.obligations.iter_mut() {
            if id.contains("/obligation:checklist:") && !declared_ids.contains(id) {
                obligation.state = RunObligationState::Abandoned;
            }
        }
    }

    /// Whether any typed expectations exist to arbitrate: from the compiled
    /// contract, the executor's own declaration, or both.
    pub(crate) fn expectations_present(&self) -> bool {
        self.contract_present || self.executor_expectations_present
    }

    /// Executor-declared obligations still open. These are the model's own
    /// statement of unmet work; a run that ends with any open is incomplete,
    /// not "closed by evidence".
    pub(crate) fn open_executor_expectations(&self) -> usize {
        self.obligations
            .iter()
            .filter(|(id, obligation)| {
                id.contains("/obligation:checklist:")
                    && !matches!(
                        obligation.state,
                        RunObligationState::Satisfied | RunObligationState::Abandoned
                    )
            })
            .count()
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
                mutating: false,
                evidence_capabilities: Vec::new(),
                result_id: None,
                operation_lineage: call.operation_lineage,
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
    pub(crate) fn effective_operation_claim(
        &self,
        claim: &TaskKernelOperationClaim,
    ) -> TaskKernelOperationClaim {
        let mut effective = claim.clone();
        effective.obligation_ids = self.effective_obligation_ids(&claim.obligation_ids);
        if effective.obligation_ids != claim.obligation_ids {
            if let Some(active_ceiling) = effective
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
        if self
            .operations
            .get(&result.tool_call_id)
            .and_then(|operation| operation.result_id.as_deref())
            == Some(result_id.as_str())
        {
            return;
        }
        if self
            .operations
            .get(&result.tool_call_id)
            .and_then(|operation| operation.result_id.as_ref())
            .is_some()
        {
            self.record_invariant("operation_has_multiple_terminal_receipts");
            return;
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
                mutating: false,
                evidence_capabilities: Vec::new(),
                result_id: None,
                operation_lineage: None,
            });
        operation.outcome = Some(receipt.outcome_status);
        operation.dispatched = receipt.invocation_stage.reached_dispatch();
        operation.policy_denied = !operation.dispatched && receipt.access_denial.is_some();
        operation.mutating = operation.mutating || receipt.semantics.mutates_state();
        if operation.evidence_capabilities.is_empty() {
            operation.evidence_capabilities = receipt.semantics.evidence.clone();
        }
        if operation.evidence_capabilities.is_empty() {
            if let Some(denial) = receipt.access_denial.as_ref() {
                operation.evidence_capabilities = denial.proposed_evidence.clone();
            }
        }
        operation.result_id = Some(result_id.clone());
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

        let completed_mutation = !is_durable_replay
            && result.succeeded()
            && receipt.invocation_stage.reached_dispatch()
            && receipt.semantics.mutates_state();
        if completed_mutation {
            self.effect_revision = self.effect_revision.saturating_add(1);
            for obligation in self.obligations.values_mut() {
                if obligation.class == RunObligationClass::Observe
                    && obligation.state == RunObligationState::Satisfied
                    && obligation
                        .satisfied_at_revision
                        .is_some_and(|revision| revision < self.effect_revision)
                {
                    obligation.state = RunObligationState::Invalidated;
                }
            }
        }

        let explicit_ids = receipt
            .completion_obligation_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        for obligation in self.obligations.values_mut() {
            if obligation.state == RunObligationState::Unverifiable {
                continue;
            }
            let observation_compatible = obligation
                .evidence_requirement
                .as_ref()
                .is_none_or(|requirement| evidence_receipt_supports(requirement, receipt));
            let target_compatible = obligation
                .required_target
                .as_ref()
                .is_none_or(|target| receipt_touches_target(target, receipt));
            let explicitly_proven = explicit_ids.contains(&obligation.id)
                && (obligation.class != RunObligationClass::Observe || observation_compatible);
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
                && target_compatible;
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

    pub(crate) fn required_count(&self) -> usize {
        self.obligations
            .values()
            .filter(|obligation| obligation.state != RunObligationState::Abandoned)
            .count()
    }

    pub(crate) fn satisfied_count(&self) -> usize {
        self.obligations
            .values()
            .filter(|obligation| obligation.state == RunObligationState::Satisfied)
            .count()
    }

    pub(crate) fn is_fulfilled(&self) -> bool {
        self.expectations_present()
            && self.cardinality_violations == 0
            && self.invariant_violations.is_empty()
            && !self.obligations.is_empty()
            && self.obligations.values().all(|obligation| {
                matches!(
                    obligation.state,
                    RunObligationState::Satisfied | RunObligationState::Abandoned
                )
            })
    }

    /// Whether all execution/evidence work is proved and only presentation may
    /// remain. This is deliberately separate from terminal success: preparing
    /// the requested response artifact closes the `Deliver` obligation later.
    pub(crate) fn work_is_fulfilled(&self) -> bool {
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
        work_obligations.all(|obligation| {
            matches!(
                obligation.state,
                RunObligationState::Satisfied | RunObligationState::Abandoned
            )
        })
    }

    /// Deterministic successful response projection. A semantic producer owns
    /// the presentation choice; the reducer exposes the typed artifact only
    /// after non-delivery proof has closed.
    pub(crate) fn projected_success_response(&self) -> Option<&str> {
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
            if matches!(
                obligation.state,
                RunObligationState::Satisfied | RunObligationState::Abandoned
            ) || obligation.class == RunObligationClass::Deliver
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
    pub(crate) fn closeout(&self, admissible: impl Fn(&RunObligation) -> bool) -> CloseoutDecision {
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
            if matches!(
                obligation.state,
                RunObligationState::Satisfied | RunObligationState::Abandoned
            ) {
                continue;
            }
            // Delivery obligations are satisfied by the closeout itself (the
            // response being prepared); they can never ask for more work.
            if obligation.class == RunObligationClass::Deliver {
                continue;
            }
            let reason = if self.obligation_policy_denied(obligation) {
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

    /// Operations refused before dispatch by a typed policy boundary.
    pub(crate) fn policy_denied_operations(&self) -> usize {
        self.operations
            .values()
            .filter(|operation| operation.policy_denied)
            .count()
    }

    /// Operations whose adapter actually ran.
    pub(crate) fn dispatched_operations(&self) -> usize {
        self.operations
            .values()
            .filter(|operation| operation.dispatched)
            .count()
    }

    /// The run's only terminal receipts are typed policy denials: the model
    /// observed the request's authority boundary and nothing it can retry
    /// would cross it. Finalization treats the model's narration of that
    /// boundary as the closeout instead of demanding more evidence.
    pub(crate) fn terminal_policy_denial(&self) -> bool {
        self.policy_denied_operations() > 0 && self.dispatched_operations() == 0
    }

    pub(crate) fn evidence_closed(&self) -> bool {
        // Receipts close a run the contract could not credit — but not past
        // the executor's own declared, still-open work. Those items are the
        // model's typed statement that more is required, so "every receipt
        // succeeded" is incompleteness, not proof.
        if self.open_executor_expectations() > 0 {
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
            if operation.result_id.is_none() {
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

    pub(crate) fn terminal_decision(&self) -> RunTerminalDecision {
        if !self.expectations_present() || self.obligations.is_empty() {
            return RunTerminalDecision::Unspecified;
        }
        if !self.invariant_violations.is_empty() {
            return RunTerminalDecision::Failed;
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
    pub(crate) fn admit_operation(&self, claim: &TaskKernelOperationClaim) -> TaskKernelAdmission {
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
    pub(crate) fn triggered_dispatch_stop_tools(&self) -> BTreeSet<String> {
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

    pub(crate) fn lifecycle_phase(&self) -> TaskKernelPhase {
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
            RunTerminalDecision::Succeeded
            | RunTerminalDecision::SucceededByEvidence
            | RunTerminalDecision::ClosedByPolicyDenial => TaskKernelPhase::WorkSucceeded,
            RunTerminalDecision::Failed => TaskKernelPhase::Failed,
            RunTerminalDecision::Pending => TaskKernelPhase::WorkPending,
            RunTerminalDecision::Unspecified if self.started => TaskKernelPhase::Running,
            RunTerminalDecision::Unspecified => TaskKernelPhase::Initialized,
        }
    }
}

pub(crate) fn receipt_matches_predicate(
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
    receipt_touches_target(target, receipt)
}

/// Whether the receipt's dispatched subject set — semantic target hints plus
/// the declared access manifest (read/write paths) — contains the target.
fn receipt_touches_target(
    target: &crate::traits::RequestVerificationTarget,
    receipt: &super::ToolReceiptV1,
) -> bool {
    let manifest_targets = receipt.access_manifest.iter().flat_map(|manifest| {
        manifest
            .read_targets
            .iter()
            .chain(manifest.write_targets.iter())
    });
    receipt
        .semantics
        .target_hints
        .iter()
        .chain(manifest_targets)
        .any(|hint| match (target.kind, hint.kind) {
            (RequestVerificationTargetKind::Url, ToolTargetHintKind::Url) => {
                target.value == hint.value
            }
            (RequestVerificationTargetKind::Path, ToolTargetHintKind::Path) => {
                let expected = crate::execution::normalize_active_path_lexically(&target.value);
                let actual = crate::execution::normalize_active_path_lexically(&hint.value);
                expected
                    .ok()
                    .zip(actual.ok())
                    .is_some_and(|(left, right)| left == right)
            }
            _ => false,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{
        Event, ExecutorExpectationItem, ExecutorExpectationsDeclaredData, TaskContractCompiledData,
        ToolOutcomeEvidenceSource, ToolReceiptV1,
    };
    use crate::traits::{
        EvidenceAuthority, EvidencePurpose, RequestCompletionContract, RequestEvidenceRequirement,
        RequestedOutcomeCondition, ToolCallEffect, ToolCallSemantics, ToolEvidenceCapability,
        ToolInvocationStage, ToolReceiptKind, ToolResultProvenance, ToolSemanticScope,
    };
    use serde_json::json;

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
                    mandate_authority: None,
                }),
            },
        )
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
                disposition: crate::events::AssistantResponseDisposition::Terminal,
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
        assert!(aggregate
            .invariant_violations
            .contains(&"dispatched_operation_outside_allowed_tool_set".to_string()));
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
        assert!(aggregate
            .obligations
            .contains_key("task:task-1/obligation:mutation:local_workspace_write"));
        assert!(aggregate
            .obligations
            .contains_key("task:task-1/obligation:deliver:response"));
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
        assert_eq!(aggregate.lifecycle_phase(), TaskKernelPhase::WorkSucceeded);

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
        let lineage = crate::events::ToolOperationLineage::DurableReplay {
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
            status: status.to_string(),
        }
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
        assert!(!aggregate
            .obligations
            .contains_key("task:task-1/obligation:checklist:3"));
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
    fn executor_redeclaration_keeps_proof_abandons_deferred_and_dropped_items() {
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
            RunObligationState::Abandoned
        );
        assert_eq!(
            aggregate.obligations["task:task-1/obligation:checklist:2"].state,
            RunObligationState::Abandoned
        );
        assert_eq!(aggregate.open_executor_expectations(), 0);
        assert!(aggregate.is_fulfilled());
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
        assert!(aggregate
            .obligations
            .contains_key("task:task-1/obligation:checklist:0"));
        assert!(aggregate
            .obligations
            .contains_key("task:task-1/obligation:evidence:0"));
    }
}
