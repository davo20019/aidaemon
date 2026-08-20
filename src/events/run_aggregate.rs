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
    AssistantResponseData, AssistantResponseDisposition, Event, EventType, ResponseDeliveryData,
    ResponseDeliveryState, TaskContractCompiledData, TaskEndData, TaskOutcome, TaskStartData,
    TaskStatus, ToolCallData, ToolResultData,
};
use crate::traits::{
    EvidenceTemporalScope, RequestReceiptPredicate, RequestResponseContract, ToolMutationEffects,
    ToolOutcomeStatus,
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
    #[serde(default)]
    pub required_effect: ToolMutationEffects,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub satisfied_at_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub satisfying_receipt_ids: Vec<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunTerminalDecision {
    /// Every accepted obligation has durable proof.
    Succeeded,
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
            contract_fingerprint: None,
            effect_revision: 0,
            obligations: BTreeMap::new(),
            response_contract: None,
            allowed_tool_names: BTreeSet::new(),
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
        self.obligations.clear();
        self.response_contract = compiled.contract.response_contract.clone();
        self.allowed_tool_names = compiled
            .contract
            .allowed_tool_names
            .iter()
            .cloned()
            .collect();
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
                required_effect: ToolMutationEffects::NONE,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
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
                required_effect: ToolMutationEffects::NONE,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
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
                required_effect: effect,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
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
                required_effect: ToolMutationEffects::NONE,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
            });
        }

        if self.response_contract.is_some() {
            self.insert_obligation(RunObligation {
                id: format!("task:{}/obligation:deliver:response", self.task_id),
                class: RunObligationClass::Deliver,
                state: RunObligationState::Pending,
                receipt: None,
                required_effect: ToolMutationEffects::NONE,
                satisfied_at_revision: None,
                satisfying_receipt_ids: Vec::new(),
            });
        }
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
                result_id: None,
                operation_lineage: None,
            });
        operation.outcome = Some(receipt.outcome_status);
        operation.dispatched = receipt.invocation_stage.reached_dispatch();
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
            let explicitly_proven = explicit_ids.contains(&obligation.id);
            let claim_allows_proof = claimed_obligation_ids.is_empty()
                || claimed_obligation_ids.contains(&obligation.id);
            let predicate_proven = claim_allows_proof
                && obligation
                    .receipt
                    .as_ref()
                    .is_some_and(|predicate| receipt_matches_predicate(&result, predicate));
            let effect_proven = obligation.class == RunObligationClass::Achieve
                && !obligation.required_effect.is_empty()
                && result.succeeded()
                && receipt
                    .semantics
                    .mutation_effects
                    .satisfies(obligation.required_effect);
            let generic_observation = obligation.class == RunObligationClass::Observe
                && obligation.receipt.is_none()
                && result.completed_observation()
                && receipt.invocation_stage.reached_dispatch()
                && receipt.semantics.observes_state();
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
        // Any terminal assistant artifact is prepared output. Whether it
        // satisfies an exact response obligation is decided separately below.
        self.prepared_response_id = response.message_id.clone();
        let Some(expected) = self
            .response_contract
            .as_ref()
            .map(|contract| contract.success_text())
        else {
            return;
        };
        if response.content.as_deref() != Some(expected) || !self.work_is_fulfilled() {
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
        self.contract_present
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

    pub(crate) fn terminal_decision(&self) -> RunTerminalDecision {
        if !self.contract_present || self.obligations.is_empty() {
            return RunTerminalDecision::Unspecified;
        }
        if !self.invariant_violations.is_empty() {
            return RunTerminalDecision::Failed;
        }
        if self.is_fulfilled() {
            return RunTerminalDecision::Succeeded;
        }
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
        if exhausted || self.cardinality_violations > 0 {
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
            RunTerminalDecision::Succeeded => TaskKernelPhase::WorkSucceeded,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{
        Event, TaskContractCompiledData, ToolOutcomeEvidenceSource, ToolReceiptV1,
    };
    use crate::traits::{
        EvidenceAuthority, EvidencePurpose, RequestCompletionContract, RequestEvidenceRequirement,
        ToolCallEffect, ToolCallSemantics, ToolInvocationStage, ToolReceiptKind,
        ToolResultProvenance,
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
        assert_eq!(aggregate.terminal_decision(), RunTerminalDecision::Failed);
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

        events.push(assistant_response("wrong", "generic summary"));
        events.last_mut().unwrap().id = 4;
        assert_eq!(
            RunAggregate::replay("task-1", &events).terminal_decision(),
            RunTerminalDecision::Pending
        );

        events.push(assistant_response(
            "exact",
            "phase=synthetic; outcome=complete",
        ));
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
            },
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
}
