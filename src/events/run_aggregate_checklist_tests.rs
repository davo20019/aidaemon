//! The run ledger over the daemon's own checklist binding, which is agent
//! logic the crate does not have.

use serde::Serialize;
use serde_json::json;

use super::*;
use crate::traits::*;

#[test]
fn unbound_checklist_target_no_longer_leaves_an_all_success_audit_partial() {
    use crate::tools::track_requirements::{StableSubjectVocabulary, bind_checklist_targets};
    use crate::traits::StableObservationSubject;
    use aidaemon_work::plans::checklist_args::executor_expectations_from_checklist_arguments;

    let objective_facets = [
        ToolSemanticFacet::Schedule,
        ToolSemanticFacet::RunState,
        ToolSemanticFacet::Recovery,
        ToolSemanticFacet::Control,
        ToolSemanticFacet::Measurement,
        ToolSemanticFacet::Ownership,
    ];
    let vocabulary = StableSubjectVocabulary::from_subjects(vec![
        StableObservationSubject::collection(
            "objective_collection:scheduled_goals",
            &objective_facets,
            "objective:",
            "every scheduled goal",
        ),
        StableObservationSubject::collection(
            "objective_collection:mandate_controllers",
            &objective_facets,
            "objective:",
            "every mandate controller",
        ),
        StableObservationSubject::collection(
            "auth_profile_collection:configured",
            &[ToolSemanticFacet::Authorization],
            "auth_profile:",
            "every configured auth profile",
        ),
    ]);
    let raw_arguments = serde_json::json!({
            "items": [
                {
                    "text": "Audit the schedule state and latest run outcome.",
                    "status": "completed",
                    "requires_observation": true,
                    "targets": [{
                        "subject": {"kind": "resource_id", "value": "objective_collection:scheduled_goals"},
                        "facets": ["schedule", "run_state"],
                        "collection_coverage": {
                            "collection": {"kind": "resource_id", "value": "objective_collection:scheduled_goals"},
                            "minimum_completeness": "complete"
                        }
                    }]
                },
                {
                    "text": "Audit objective-control presence.",
                    "status": "completed",
                    "requires_observation": true,
                    "targets": [{
                        "subject": {"kind": "resource_id", "value": "objective_collection:mandate_controllers"},
                        "facets": ["control"],
                        "collection_coverage": {
                            "collection": {"kind": "resource_id", "value": "objective_collection:mandate_controllers"},
                            "minimum_completeness": "complete"
                        }
                    }]
                },
                {
                    "text": "Audit owner configuration and credential readiness.",
                    "status": "completed",
                    "requires_observation": true,
                    "targets": ["x_owner_credential_state"]
                }
            ]
        })
        .to_string();

    let goals_hint = ToolTargetHint::new(
        ToolTargetHintKind::ResourceId,
        "objective_collection:scheduled_goals",
    )
    .unwrap();
    let mandates_hint = ToolTargetHint::new(
        ToolTargetHintKind::ResourceId,
        "objective_collection:mandate_controllers",
    )
    .unwrap();
    let profile_collection_hint = ToolTargetHint::new(
        ToolTargetHintKind::ResourceId,
        "auth_profile_collection:configured",
    )
    .unwrap();
    let profile_hint =
        ToolTargetHint::new(ToolTargetHintKind::ResourceId, "auth_profile:twitter").unwrap();
    let goal_member =
        ToolTargetHint::new(ToolTargetHintKind::ResourceId, "objective:sha256:synthetic").unwrap();
    let receipts = || {
        vec![
            call("goal-runs", "scheduled_goal_runs"),
            with_exact_observations(
                result(
                    "goal-runs",
                    "scheduled_goal_runs",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    ToolCallSemantics::observation(),
                ),
                vec![objective_observation(
                    "objective:sha256:synthetic",
                    &objective_facets,
                )],
                vec![ToolCollectionObservation::complete(
                    goals_hint.clone(),
                    &objective_facets,
                    vec![goal_member.clone()],
                )],
            ),
            call("mandates", "manage_mandates"),
            with_exact_observations(
                result(
                    "mandates",
                    "manage_mandates",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    ToolCallSemantics::observation(),
                ),
                Vec::new(),
                vec![
                    ToolCollectionObservation::complete(
                        goals_hint.clone(),
                        &objective_facets,
                        vec![goal_member.clone()],
                    ),
                    ToolCollectionObservation::complete(
                        mandates_hint.clone(),
                        &objective_facets,
                        Vec::new(),
                    ),
                ],
            ),
            call("auth", "manage_http_auth"),
            with_exact_observations(
                result(
                    "auth",
                    "manage_http_auth",
                    ToolOutcomeStatus::Succeeded,
                    0,
                    ToolCallSemantics::observation(),
                ),
                vec![ToolObservationEvidence::new(
                    profile_hint.clone(),
                    &[ToolSemanticFacet::Authorization],
                )],
                vec![ToolCollectionObservation::complete(
                    profile_collection_hint.clone(),
                    &[ToolSemanticFacet::Authorization],
                    vec![profile_hint.clone()],
                )],
            ),
        ]
    };

    // Negative control: the pre-fix declaration keeps the invented ID as an
    // exact obligation, which no receipt can ever credit.
    let unbound_declaration =
        executor_expectations_from_checklist_arguments("task-1", &raw_arguments).unwrap();
    assert_eq!(
        unbound_declaration.items[2].targets,
        ["x_owner_credential_state"]
    );
    let mut control_events = vec![event(
        EventType::ExecutorExpectationsDeclared,
        unbound_declaration,
    )];
    control_events.extend(receipts());
    renumber(&mut control_events);
    let control = RunAggregate::replay("task-1", &control_events);
    assert_eq!(
        control.obligations["task:task-1/obligation:checklist:2"].state,
        RunObligationState::Pending
    );
    assert_eq!(control.open_executor_expectations(), 1);
    assert!(!control.evidence_closed());

    // Bound declaration: the invented ID is recorded as unbound and the
    // item becomes an untargeted observation expectation.
    let bound_arguments = bind_checklist_targets(&raw_arguments, &vocabulary).unwrap();
    let bound_declaration =
        executor_expectations_from_checklist_arguments("task-1", &bound_arguments).unwrap();
    assert!(bound_declaration.items[2].targets.is_empty());
    assert!(bound_declaration.items[2].observation_targets.is_empty());
    assert_eq!(
        bound_declaration.items[2].unbound_targets,
        ["x_owner_credential_state"]
    );
    assert_eq!(bound_declaration.items[0].observation_targets.len(), 1);
    assert_eq!(bound_declaration.items[1].observation_targets.len(), 1);

    let mut events = vec![event(
        EventType::ExecutorExpectationsDeclared,
        bound_declaration,
    )];
    events.extend(receipts());
    renumber(&mut events);
    let aggregate = RunAggregate::replay("task-1", &events);
    for index in 0..3 {
        assert_eq!(
            aggregate.obligations[&format!("task:task-1/obligation:checklist:{index}")].state,
            RunObligationState::Satisfied,
            "checklist item {index} must be credited by the typed receipts"
        );
    }
    assert_eq!(aggregate.open_executor_expectations(), 0);
    assert!(aggregate.evidence_closed());
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

fn objective_observation(
    objective_id: &str,
    facets: &[ToolSemanticFacet],
) -> ToolObservationEvidence {
    ToolObservationEvidence::new(
        ToolTargetHint::new(ToolTargetHintKind::ResourceId, objective_id).unwrap(),
        facets,
    )
}

fn with_exact_observations(
    mut event: Event,
    observations: Vec<ToolObservationEvidence>,
    collections: Vec<ToolCollectionObservation>,
) -> Event {
    event.data["receipt"]["observations"] = serde_json::to_value(observations).unwrap();
    event.data["receipt"]["collection_observations"] = serde_json::to_value(collections).unwrap();
    event
}

fn renumber(events: &mut [Event]) {
    for (index, event) in events.iter_mut().enumerate() {
        event.id = index as i64 + 1;
    }
}
