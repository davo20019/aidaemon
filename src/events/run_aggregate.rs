//! Deterministic request-run lifecycle reducer.
//!
//! The immutable event history is the source of truth. This aggregate is a
//! discardable projection: rebuilding it from the same ordered events must
//! always yield the same result. LLM output can propose contracts and tool
//! operations, but only typed events change lifecycle state here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{Event, EventType, TaskContractCompiledData, ToolCallData, ToolResultData};
use crate::traits::{
    EvidenceTemporalScope, RequestReceiptPredicate, ToolMutationEffects, ToolOutcomeStatus,
    ToolReceiptKind,
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
    /// A response/artifact reached its requested delivery boundary.
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
    pub const SCHEMA_VERSION: u16 = 1;

    pub(crate) fn new(task_id: impl Into<String>) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            task_id: task_id.into(),
            contract_present: false,
            effect_revision: 0,
            obligations: BTreeMap::new(),
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

    fn reconcile_cardinality(&mut self) {
        self.cardinality_violations = 0;
        for obligation in self.obligations.values() {
            let Some(predicate) = obligation.receipt.as_ref() else {
                continue;
            };
            let Some(limit) = predicate.max_invocations else {
                continue;
            };
            let dispatches = self
                .operations
                .values()
                .filter(|operation| {
                    operation.dispatched
                        && (predicate.tool_names.is_empty()
                            || predicate.tool_names.contains(&operation.tool_name))
                })
                .count();
            self.cardinality_violations = self
                .cardinality_violations
                .saturating_add(dispatches.saturating_sub(limit));
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
            && !self.obligations.is_empty()
            && self.obligations.values().all(|obligation| {
                matches!(
                    obligation.state,
                    RunObligationState::Satisfied | RunObligationState::Abandoned
                )
            })
    }

    pub(crate) fn terminal_decision(&self) -> RunTerminalDecision {
        if !self.contract_present || self.obligations.is_empty() {
            return RunTerminalDecision::Unspecified;
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
                        operation.dispatched
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
    if let Some(expected_rejection) = predicate.contract_rejected {
        if receipt.contract_rejected != expected_rejection {
            return false;
        }
    } else if receipt.contract_rejected
        && !predicate
            .outcome_statuses
            .contains(&ToolOutcomeStatus::Blocked)
    {
        return false;
    }
    if !predicate.exit_codes.is_empty()
        && !receipt
            .exit_code
            .is_some_and(|code| predicate.exit_codes.contains(&code))
    {
        return false;
    }
    if !predicate.outcome_statuses.is_empty()
        && !predicate.outcome_statuses.contains(&receipt.outcome_status)
    {
        let legacy_process_negative = receipt.outcome_status
            == ToolOutcomeStatus::CompletedWithNegativeResult
            && receipt.receipt_kind == ToolReceiptKind::Process
            && receipt.exit_code.is_some_and(|code| code != 0)
            && predicate
                .outcome_statuses
                .contains(&ToolOutcomeStatus::FailedPermanent);
        if !legacy_process_negative {
            return false;
        }
    }
    if predicate.contract_rejected != Some(true)
        && !receipt.invocation_stage.reached_dispatch()
        && !predicate
            .outcome_statuses
            .contains(&ToolOutcomeStatus::Blocked)
    {
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
        ToolCallEffect, ToolCallSemantics, ToolInvocationStage, ToolResultProvenance,
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
                    allowed_tool_names: Vec::new(),
                    forbidden_tool_scopes: Vec::new(),
                    required_response_fields: Vec::new(),
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
