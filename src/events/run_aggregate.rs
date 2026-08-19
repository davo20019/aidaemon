//! Deterministic request-run lifecycle reducer.
//!
//! The immutable event history is the source of truth. This aggregate is a
//! discardable projection: rebuilding it from the same ordered events must
//! always yield the same result. LLM output can propose contracts and tool
//! operations, but only typed events change lifecycle state here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    AssistantResponseData, Event, EventType, TaskContractCompiledData, ToolCallData, ToolResultData,
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
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<ToolOutcomeStatus>,
    #[serde(default)]
    pub dispatched: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_id: Option<String>,
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
}

impl RunAggregate {
    pub const SCHEMA_VERSION: u16 = 3;

    pub(crate) fn new(task_id: impl Into<String>) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            task_id: task_id.into(),
            contract_present: false,
            effect_revision: 0,
            obligations: BTreeMap::new(),
            response_contract: None,
            allowed_tool_names: BTreeSet::new(),
            operations: BTreeMap::new(),
            cardinality_violations: 0,
            primary_causal_operation_id: None,
            invariant_violations: Vec::new(),
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
            _ => {}
        }
    }

    fn install_contract(&mut self, compiled: TaskContractCompiledData) {
        if self.contract_present {
            self.record_invariant("duplicate_contract_installation");
            return;
        }
        self.contract_present = true;
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
                state: RunObligationState::Pending,
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

    fn record_call(&mut self, call: ToolCallData) {
        if call
            .task_id
            .as_deref()
            .is_some_and(|task_id| task_id != self.task_id)
        {
            self.record_invariant("tool_call_task_identity_mismatch");
            return;
        }
        if !self.allowed_tool_names.is_empty() && !self.allowed_tool_names.contains(&call.name) {
            self.record_invariant("operation_outside_allowed_tool_set");
        }
        self.operations
            .entry(call.tool_call_id.clone())
            .or_insert(RunOperation {
                operation_id: call.tool_call_id,
                tool_name: call.name,
                idempotency_key: call.idempotency_key,
                outcome: None,
                dispatched: false,
                result_id: None,
            });
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
                idempotency_key: receipt.idempotency_key.clone(),
                outcome: None,
                dispatched: false,
                result_id: None,
            });
        operation.outcome = Some(receipt.outcome_status);
        operation.dispatched = receipt.invocation_stage.reached_dispatch();
        operation.result_id = Some(result_id.clone());

        let completed_mutation = result.succeeded()
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
            let explicitly_proven = explicit_ids.contains(&obligation.id);
            let predicate_proven = obligation
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
            if explicitly_proven || predicate_proven || effect_proven || generic_observation {
                obligation.state = RunObligationState::Satisfied;
                obligation.satisfied_at_revision = Some(self.effect_revision);
                if !obligation.satisfying_receipt_ids.contains(&result_id) {
                    obligation.satisfying_receipt_ids.push(result_id.clone());
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
                    operation.result_id.is_some()
                        && (predicate.tool_names.is_empty()
                            || predicate.tool_names.contains(&operation.tool_name))
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
                        operation.result_id.is_some()
                            && (predicate.tool_names.is_empty()
                                || predicate.tool_names.contains(&operation.tool_name))
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
                turn_id: None,
            },
        )
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
        ];
        for (index, event) in events.iter_mut().enumerate() {
            event.id = index as i64 + 1;
        }

        let aggregate = RunAggregate::replay("task-1", &events);
        assert!(aggregate
            .invariant_violations
            .contains(&"operation_outside_allowed_tool_set".to_string()));
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
}
