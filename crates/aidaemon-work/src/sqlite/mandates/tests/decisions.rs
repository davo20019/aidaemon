//! Tests of recording decision cycles.

use super::*;

#[tokio::test]
async fn decision_validates_policy_version_and_commits_transition_atomically() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    let run = claim_and_start_run(&store, &goal, &mandate).await;

    let stale = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Wait,
        "Nothing useful yet",
        mandate.version + 1,
    );
    assert!(
        store
            .record_mandate_decision(&stale, None, None)
            .await
            .is_err()
    );
    assert!(
        store
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .get_mandate(&mandate.id)
            .await
            .unwrap()
            .unwrap()
            .review_lease_token
            .is_some()
    );

    let mut wait = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Wait,
        "No action clears the quality threshold",
        mandate.version,
    );
    wait.reconsider_at = Some((chrono::Utc::now() + chrono::Duration::seconds(1)).to_rfc3339());
    let before = chrono::Utc::now();
    store
        .record_mandate_decision(&wait, None, None)
        .await
        .unwrap();

    let loaded = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(loaded.status, MandateStatus::Active);
    assert!(loaded.review_lease_token.is_none());
    let next = chrono::DateTime::parse_from_rfc3339(&loaded.next_review_at)
        .unwrap()
        .with_timezone(&chrono::Utc);
    assert!(next >= before + chrono::Duration::seconds(mandate.min_review_secs - 1));
    let stored_decision = store
        .get_mandate_decision_for_run(&run.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored_decision.mandate_version, mandate.version);
    assert_eq!(stored_decision.reconsider_at, Some(loaded.next_review_at));
    let history = store.list_mandate_decisions(&mandate.id, 10).await.unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].id, stored_decision.id);
    assert_eq!(history[0].outcome, MandateDecisionOutcome::Wait);
}

/// A bounded mandate's objective control is advisory: its decisions do
/// not wait on a measurement. Autopilot keeps the measured control loop.
#[tokio::test]
async fn bounded_mandate_with_objective_control_decides_without_a_measurement() {
    for (mode, gated) in [
        (MandateAutonomyMode::Bounded, false),
        (MandateAutonomyMode::Autopilot, true),
    ] {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 1);
        mandate.autonomy_mode = mode;
        mandate.objective_control = Some(objective_control());
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let wait = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Wait,
            "Nothing new is worth acting on yet",
            mandate.version,
        );
        let result = store.record_mandate_decision(&wait, None, None).await;
        if gated {
            let error = result.expect_err("autopilot decisions need a measurement");
            assert!(
                error
                    .to_string()
                    .contains("receipt-backed metric measurement"),
                "{error}"
            );
        } else {
            result.expect("a bounded mandate decides without a measurement");
            let stored = store
                .get_mandate_decision_for_run(&run.id)
                .await
                .unwrap()
                .expect("decision stored");
            assert_eq!(stored.outcome, MandateDecisionOutcome::Wait);
        }
    }
}

#[tokio::test]
async fn act_for_value_contract_requires_exact_owner_criterion_and_full_judgment() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 1);
    mandate.success_criteria =
        vec!["Each intervention provides verified useful information to the audience".to_string()];
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "Current evidence supports one bounded useful intervention",
        mandate.version,
    );
    let mut intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Provide one verified useful update",
        "The intervention advances the confirmed value criterion",
    );
    intention.value_criterion = Some("Publish something on every review".to_string());
    intention.expected_benefit =
        Some("Give the audience current, grounded information".to_string());
    intention.risk = Some("Low volume limits attention and reputation costs".to_string());
    intention.invalidation_criteria =
        Some("The information is stale, duplicate, or unsupported".to_string());

    let error = store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("not an exact owner-authored success criterion")
    );

    intention.value_criterion = mandate.success_criteria.first().cloned();
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    assert_eq!(
        store.list_intentions(&mandate.id, 1).await.unwrap()[0],
        intention
    );
}

#[tokio::test]
async fn runtime_fallback_wait_bypasses_semantic_gates_and_finalizes_as_deliberator_failure() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 1);
    // The measured control loop (gate, stagnation) is autopilot's.
    mandate.autonomy_mode = MandateAutonomyMode::Autopilot;
    mandate.objective_control = Some(objective_control());
    let run = claim_and_start_run(&store, &goal, &mandate).await;

    // A model-authored WAIT on a controlled mandate still needs a
    // current-run measurement.
    let model_wait = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Wait,
        "model decided to wait without measuring",
        mandate.version,
    );
    let error = store
        .record_mandate_decision(&model_wait, None, None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("metric measurement"), "{error}");

    // The runtime fallback carries no authority and must always persist,
    // regardless of objective control state.
    let fallback =
        MandateDecisionCycle::runtime_fallback_wait(&mandate, &run.id, chrono::Utc::now());
    assert!(fallback.is_runtime_fallback());
    store
        .record_mandate_decision(&fallback, None, None)
        .await
        .unwrap();
    let mut spoofed = fallback.clone();
    spoofed.id = uuid::Uuid::new_v4().to_string();
    spoofed.evidence_receipt_ids = vec!["call_anything".to_string()];
    assert!(
        store
            .record_mandate_decision(&spoofed, None, None)
            .await
            .is_err()
    );

    let root_attempt = store
        .claim_task_with_lease(
            run.root_task_id.as_deref().unwrap(),
            "fallback-root",
            Some("profile-task-lead"),
            7_200,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(
        store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("task lead returned without a decision".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    );
    let result = store
        .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
            mandate_id: mandate.id.clone(),
            expected_mandate_version: mandate.version,
            goal_run_id: run.id.clone(),
            finalized_at: chrono::Utc::now().to_rfc3339(),
        })
        .await
        .unwrap();
    assert_eq!(
        result,
        MandateRunFinalizationResult::Rejected {
            reason: MandateFinalizationRejectReason::DeliberatorFailed
        }
    );
    let current = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(current.status, MandateStatus::Active);
    assert_eq!(
        store.get_goal_runs(&goal.id).await.unwrap()[0]
            .outcome_summary
            .as_deref(),
        Some("mandate_review_failed:deliberator_failed")
    );
}

#[tokio::test]
async fn missing_decision_closes_run_schedules_retry_and_commits_review_failed_notice() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let root_attempt = store
        .claim_task_with_lease(
            run.root_task_id.as_deref().unwrap(),
            "decision-missing-root",
            Some("profile-task-lead"),
            7_200,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(
        store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("task lead returned without a decision".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    );
    let finalized_at = chrono::Utc::now();

    let result = store
        .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
            mandate_id: mandate.id.clone(),
            expected_mandate_version: mandate.version,
            goal_run_id: run.id.clone(),
            finalized_at: finalized_at.to_rfc3339(),
        })
        .await
        .unwrap();
    assert_eq!(
        result,
        MandateRunFinalizationResult::Rejected {
            reason: MandateFinalizationRejectReason::DecisionMissing
        }
    );
    assert_eq!(
        store.get_goal_runs(&goal.id).await.unwrap()[0].status,
        "failed"
    );
    let current = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(current.status, MandateStatus::Active);
    assert!(
        chrono::DateTime::parse_from_rfc3339(&current.next_review_at)
            .unwrap()
            .with_timezone(&chrono::Utc)
            >= finalized_at + chrono::Duration::seconds(mandate.min_review_secs)
    );
    let controller = store.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(controller.status, "active");
    assert_eq!(controller.dispatch_failures, 1);
    let queued: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM notification_queue
             WHERE id = ? AND notification_type = 'mandate_review_failed'
               AND priority = 'critical' AND expires_at IS NULL",
    )
    .bind(format!("mandate-run-notice:{}", run.id))
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(queued, 1);
}

#[tokio::test]
async fn invalid_decision_state_pauses_controller_and_commits_review_failed_notice() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let root_attempt = store
        .claim_task_with_lease(
            run.root_task_id.as_deref().unwrap(),
            "invalid-decision-root",
            Some("profile-task-lead"),
            7_200,
        )
        .await
        .unwrap()
        .unwrap();
    let decision = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "Persist a valid row before simulating corruption",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &decision.id,
        &run.id,
        "A committed intention that cannot belong to WAIT",
        "It creates a valid but inconsistent persisted decision state",
    );
    store
        .record_mandate_decision(&decision, Some(&intention), Some(&root_attempt.id))
        .await
        .unwrap();
    sqlx::query("UPDATE mandate_decision_cycles SET outcome = 'wait' WHERE id = ?")
        .bind(&decision.id)
        .execute(&store.pool)
        .await
        .unwrap();
    assert!(
        store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("invalid decision state simulated".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    );

    let result = store
        .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
            mandate_id: mandate.id.clone(),
            expected_mandate_version: mandate.version,
            goal_run_id: run.id.clone(),
            finalized_at: chrono::Utc::now().to_rfc3339(),
        })
        .await
        .unwrap();
    assert_eq!(
        result,
        MandateRunFinalizationResult::Rejected {
            reason: MandateFinalizationRejectReason::InvalidDecisionState
        }
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
    assert_eq!(
        store.get_goal_runs(&goal.id).await.unwrap()[0].status,
        "failed"
    );
    let queued: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM notification_queue
             WHERE id = ? AND notification_type = 'mandate_review_failed'
               AND priority = 'critical' AND expires_at IS NULL",
    )
    .bind(format!("mandate-run-notice:{}", run.id))
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(queued, 1);
}
