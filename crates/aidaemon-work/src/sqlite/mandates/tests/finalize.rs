//! Tests of finalizing runs from receipt proof.

use super::*;

#[tokio::test]
async fn proof_distinguishes_never_dispatched_from_claimed_ambiguity() {
    for claim_before_finalization in [false, true] {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "Exercise the finalizer crash-window classification",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Attempt one governed mutation",
            "The proof must classify its exact dispatch state",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let now = chrono::Utc::now();
        let reserved = reservation(&mandate, &act, &run, &fence, 1, 451, now);
        assert!(
            store
                .reserve_mandate_action_attempt(&reserved)
                .await
                .unwrap()
                .is_some()
        );
        if claim_before_finalization {
            assert!(
                store
                    .claim_mandate_mutation_dispatch(&dispatch_claim(
                        &reserved,
                        now + chrono::Duration::milliseconds(1),
                    ))
                    .await
                    .unwrap()
            );
        }
        complete_mutation_fence(&store, &fence).await;

        let result = store
            .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                mandate_id: mandate.id.clone(),
                expected_mandate_version: mandate.version,
                goal_run_id: run.id.clone(),
                finalized_at: (now + chrono::Duration::seconds(1)).to_rfc3339(),
            })
            .await
            .unwrap();
        if claim_before_finalization {
            let MandateRunFinalizationResult::ReconciliationRequired { reason, counts } = result
            else {
                panic!("a claimed dispatch with no outcome must fail closed: {result:?}")
            };
            assert_eq!(
                reason,
                MandateReconciliationReason::MutationOutcomeAmbiguous
            );
            assert_eq!(counts.ambiguous_or_reserved_mutations, 1);
            assert_eq!(counts.never_dispatched_mutations, 0);
        } else {
            // Reserved but never dispatched: provably no effect.
            assert_eq!(
                result,
                MandateRunFinalizationResult::Rejected {
                    reason: MandateFinalizationRejectReason::ActChangedNothing,
                }
            );
        }
        let attempts = store
            .list_mandate_mutation_attempts_for_run(&run.id)
            .await
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(
            attempts[0].status,
            if claim_before_finalization {
                MandateMutationAttemptStatus::Ambiguous
            } else {
                MandateMutationAttemptStatus::NeverDispatched
            }
        );
    }
}

#[tokio::test]
async fn evidence_receipts_accept_either_tool_call_id_or_receipt_result_id() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 1);
    mandate.objective_control = Some(objective_control());
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let task_id = run.root_task_id.as_deref().unwrap();
    sqlx::query(
        "INSERT INTO events
                (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES (?, 'tool_result', ?, ?, ?, 'http_request')",
    )
    .bind("owner-session")
    .bind(
        serde_json::json!({
            "task_id": task_id,
            "tool_call_id": "call_synthetic_metric",
            "name": "http_request",
            "result": "synthetic metric value",
            "success": true,
            "duration_ms": 1,
            "receipt": {
                "schema_version": aidaemon_events::ToolReceiptV1::SCHEMA_VERSION,
                "outcome_status": "succeeded",
                "outcome_evidence": "structured_metadata",
                "result_provenance": { "result_id": "sha256:synthetic-metric-digest" }
            }
        })
        .to_string(),
    )
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(task_id)
    .execute(&store.pool)
    .await
    .unwrap();

    for receipt_id in ["call_synthetic_metric", "sha256:synthetic-metric-digest"] {
        let measurement = MandateObjectiveMeasurement::new(
            &mandate.id,
            mandate.version,
            &run.id,
            12_000_000,
            9_500,
            vec![receipt_id.to_string()],
            &chrono::Utc::now().to_rfc3339(),
        );
        store
            .record_mandate_objective_measurement(&measurement)
            .await
            .unwrap_or_else(|error| panic!("{receipt_id} should be accepted: {error}"));
    }
    let invented = MandateObjectiveMeasurement::new(
        &mandate.id,
        mandate.version,
        &run.id,
        12_000_000,
        9_500,
        vec!["sha256:not-a-receipt".to_string()],
        &chrono::Utc::now().to_rfc3339(),
    );
    assert!(
        store
            .record_mandate_objective_measurement(&invented)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn receipt_backed_stop_and_learning_are_same_mandate_only() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 0);
    mandate.success_criteria = vec!["Account stewardship is no longer needed".to_string()];
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let root_task_id = run.root_task_id.as_deref().unwrap();
    let root_attempt = store
        .claim_task_with_lease(
            root_task_id,
            "receipt-backed-stop",
            Some("profile-task-lead"),
            7_200,
        )
        .await
        .unwrap()
        .unwrap();

    let receipt = aidaemon_events::ToolReceiptV1::from_metadata(
        &crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
            semantics: crate::traits::ToolCallSemantics::observation(),
            ..Default::default()
        },
        crate::traits::ToolOutcomeStatus::Succeeded,
        aidaemon_events::ToolOutcomeEvidenceSource::StructuredMetadata,
        None,
    );
    let result = aidaemon_events::ToolResultData {
        message_id: None,
        tool_call_id: "observation-receipt-1".to_string(),
        name: "web_fetch".to_string(),
        result: "bounded observation".to_string(),
        success: true,
        duration_ms: 1,
        error: None,
        task_id: Some(root_task_id.to_string()),
        annotations: vec![],
        turn_id: None,
        attachments: vec![],
        receipt: Some(receipt),
    };
    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES (?, 'tool_result', ?, ?, ?, 'web_fetch')",
    )
    .bind("specialist:mandate")
    .bind(serde_json::to_string(&result).unwrap())
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(root_task_id)
    .execute(&store.pool)
    .await
    .unwrap();

    let mut decision = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Stop,
        "A current structured observation proves the owner criterion",
        mandate.version,
    );
    decision.termination_kind = Some(MandateTerminationKind::SuccessCriteriaSatisfied);
    decision.activity_level = MandateActivityLevel::Urgent;
    decision.termination_match = mandate.success_criteria.first().cloned();
    decision.evidence_receipt_ids = vec!["observation-receipt-1".to_string()];
    let note = MandateLearningNote::new(
        &mandate.id,
        mandate.version,
        &decision.id,
        "Explicit verification is more reliable than inferred completion",
        vec!["observation-receipt-1".to_string()],
    );
    store
        .record_mandate_decision_with_updates(
            &decision,
            None,
            Some(&MandateOperatingUpdates {
                learning_note: Some(note.clone()),
            }),
            Some(&root_attempt.id),
        )
        .await
        .unwrap();
    assert_eq!(
        store
            .list_mandate_learning_notes(&mandate.id, 10)
            .await
            .unwrap(),
        vec![note]
    );
    assert_eq!(
        store
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .unwrap()
            .activity_level,
        MandateActivityLevel::Urgent
    );

    assert!(
        store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("receipt-backed STOP recorded".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    );
    let finalized = store
        .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
            mandate_id: mandate.id.clone(),
            expected_mandate_version: mandate.version,
            goal_run_id: run.id,
            finalized_at: chrono::Utc::now().to_rfc3339(),
        })
        .await
        .unwrap();
    assert!(matches!(
        finalized,
        MandateRunFinalizationResult::NonActionSatisfied {
            outcome: MandateDecisionOutcome::Stop,
            ..
        }
    ));
}

#[tokio::test]
async fn canonical_tool_result_atomically_projects_the_mandate_ledger() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let decision = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "Exercise atomic receipt projection",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &decision.id,
        &run.id,
        "Issue one governed request",
        "The event and ledger must commit together",
    );
    store
        .record_mandate_decision(&decision, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let now = chrono::Utc::now();
    let reservation = reservation(&mandate, &decision, &run, &fence, 1, 777, now);
    store
        .reserve_mandate_action_attempt(&reservation)
        .await
        .unwrap()
        .expect("reserved mutation");
    assert!(
        store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &reservation,
                now + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap()
    );

    let semantics = crate::traits::ToolCallSemantics::mutation_with(
        crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY,
    );
    let mut receipt = aidaemon_events::ToolReceiptV1::from_metadata(
        &crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
            http_status: Some(201),
            semantics,
            ..Default::default()
        },
        crate::traits::ToolOutcomeStatus::Succeeded,
        aidaemon_events::ToolOutcomeEvidenceSource::StructuredMetadata,
        Some("mandate-atomic-receipt".to_string()),
    );
    receipt.mandate_authority = Some(reservation.grant.clone());
    let result = aidaemon_events::ToolResultData {
        message_id: None,
        tool_call_id: reservation.tool_call_id.clone(),
        name: "http_request".to_string(),
        result: "created".to_string(),
        success: true,
        duration_ms: 1,
        error: None,
        task_id: Some(fence.worker_task.id.clone()),
        annotations: vec![],
        turn_id: None,
        attachments: vec![],
        receipt: Some(receipt),
    };
    let event_store = aidaemon_events::EventStore::on_paths(
        store.pool(),
        aidaemon_tools::LexicalPaths::daemon_host(),
        Arc::new(aidaemon_events::NoEventHost),
    )
    .await
    .unwrap();
    event_store
        .append(aidaemon_events::Event::new(
            "specialist:mandate",
            aidaemon_events::EventType::ToolResult,
            serde_json::to_value(result).unwrap(),
        ))
        .await
        .unwrap();

    let attempts = store
        .list_mandate_mutation_attempts_for_run(&run.id)
        .await
        .unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, MandateMutationAttemptStatus::Succeeded);
}

#[tokio::test]
async fn strict_receipts_finalize_act_and_mixed_failure_fails_closed() {
    for include_failure in [false, true] {
        let (store, _database) = test_store().await;
        let cycle_cap = if include_failure { 2 } else { 1 };
        let (goal, mut mandate) = controller("owner-session", cycle_cap);
        mandate.authority.max_mutating_actions_per_rolling_24h = cycle_cap;
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "Execute and prove the governed action",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Execute governed mutation",
            "Strict receipts determine completion",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let base = chrono::Utc::now();
        let first = reservation(&mandate, &act, &run, &fence, 1, 401, base);
        assert!(
            store
                .reserve_mandate_action_attempt(&first)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            store
                .claim_mandate_mutation_dispatch(&dispatch_claim(
                    &first,
                    base + chrono::Duration::milliseconds(1),
                ))
                .await
                .unwrap()
        );
        assert!(
            finish_attempt(
                &store,
                &first,
                MandateMutationAttemptStatus::Succeeded,
                Some(200),
                base + chrono::Duration::seconds(1)
            )
            .await
        );
        if include_failure {
            let second = reservation(
                &mandate,
                &act,
                &run,
                &fence,
                2,
                402,
                base + chrono::Duration::seconds(900),
            );
            assert!(
                store
                    .reserve_mandate_action_attempt(&second)
                    .await
                    .unwrap()
                    .is_some()
            );
            assert!(
                store
                    .claim_mandate_mutation_dispatch(&dispatch_claim(
                        &second,
                        base + chrono::Duration::milliseconds(900_001),
                    ))
                    .await
                    .unwrap()
            );
            assert!(
                finish_attempt(
                    &store,
                    &second,
                    MandateMutationAttemptStatus::Failed,
                    Some(400),
                    base + chrono::Duration::seconds(901)
                )
                .await
            );
        }
        complete_mutation_fence(&store, &fence).await;
        let result = store
            .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                mandate_id: mandate.id.clone(),
                expected_mandate_version: mandate.version,
                goal_run_id: run.id.clone(),
                finalized_at: (base + chrono::Duration::seconds(1_000)).to_rfc3339(),
            })
            .await
            .unwrap();
        if include_failure {
            assert!(matches!(
                result,
                MandateRunFinalizationResult::ReconciliationRequired {
                    reason: MandateReconciliationReason::MutationOutcomeFailed,
                    ..
                }
            ));
            assert_eq!(
                store.get_goal_runs(&goal.id).await.unwrap()[0].status,
                "failed"
            );
            assert!(
                store
                    .get_current_goal_run(&goal.id)
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_eq!(
                store.list_intentions(&mandate.id, 1).await.unwrap()[0].status,
                IntentionStatus::Suspended
            );
            assert_eq!(
                store
                    .get_mandate(&mandate.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                MandateStatus::AwaitingInput
            );
            assert_eq!(
                store.get_goal(&goal.id).await.unwrap().unwrap().status,
                "paused"
            );
        } else {
            assert!(matches!(
                result,
                MandateRunFinalizationResult::ActSatisfied { .. }
            ));
            assert_eq!(
                store.get_goal_runs(&goal.id).await.unwrap()[0].status,
                "completed"
            );
            assert_eq!(
                store.list_intentions(&mandate.id, 1).await.unwrap()[0].status,
                IntentionStatus::Satisfied
            );
        }
    }
}
