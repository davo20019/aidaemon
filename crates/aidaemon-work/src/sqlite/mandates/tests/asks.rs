//! Tests of asks, owner alerts and suspensions.

use super::*;

#[tokio::test]
async fn ask_and_stop_atomically_change_mandate_and_controller_status() {
    for (outcome, expected_mandate, expected_goal) in [
        (
            MandateDecisionOutcome::Ask,
            MandateStatus::AwaitingInput,
            "paused",
        ),
        (
            MandateDecisionOutcome::Stop,
            MandateStatus::Completed,
            "completed",
        ),
    ] {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let root_attempt = store
            .claim_task_with_lease(
                run.root_task_id.as_deref().unwrap(),
                "non-action-root",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        let mut decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            outcome,
            "The mandate cannot safely proceed",
            mandate.version,
        );
        if outcome == MandateDecisionOutcome::Ask {
            decision.question = Some("May I broaden the allowed target?".to_string());
        } else {
            decision.termination_kind = Some(MandateTerminationKind::SafetyTermination);
        }
        store
            .record_mandate_decision(&decision, None, Some(&root_attempt.id))
            .await
            .unwrap();
        assert_eq!(
            store
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::Active,
            "the lifecycle transition remains provisional until proof finalization"
        );
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
        assert!(
            store
                .patch_task_from_attempt(
                    &root_attempt.id,
                    &root_attempt.lease_token,
                    &crate::traits::TaskAttemptPatch {
                        status: "completed".to_string(),
                        result: Some("non-action decision durably recorded".to_string()),
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
        assert!(matches!(
            result,
            MandateRunFinalizationResult::NonActionSatisfied {
                outcome: finalized_outcome,
                ..
            } if finalized_outcome == outcome
        ));
        assert_eq!(
            store
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            expected_mandate
        );
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            expected_goal
        );
    }
}

#[tokio::test]
async fn ask_and_stop_crashes_remain_reconcilable_until_proof_finalization() {
    for outcome in [MandateDecisionOutcome::Ask, MandateDecisionOutcome::Stop] {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let root_attempt = store
            .claim_task_with_lease(
                run.root_task_id.as_deref().unwrap(),
                "crashing-non-action-root",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        let mut decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            outcome,
            "Record the non-action decision before the simulated crash",
            mandate.version,
        );
        if outcome == MandateDecisionOutcome::Ask {
            decision.question = Some("May I broaden the exact target?".to_string());
        } else {
            decision.termination_kind = Some(MandateTerminationKind::SafetyTermination);
        }
        store
            .record_mandate_decision(&decision, None, Some(&root_attempt.id))
            .await
            .unwrap();
        assert_eq!(
            store
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::Active
        );
        sqlx::query("UPDATE task_attempts SET lease_expires_at = ? WHERE id = ?")
            .bind((chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339())
            .bind(&root_attempt.id)
            .execute(&store.pool)
            .await
            .unwrap();

        assert!(
            store
                .claim_due_mandates(1, "post-crash-reconciler", 300)
                .await
                .unwrap()
                .is_empty()
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
            "cancelled"
        );
    }
}

/// Drive one review to a finalized ASK exactly as the task lead would,
/// leaving the mandate parked in `awaiting_input` with its question.
async fn park_mandate_on_ask(
    store: &SqliteWorkStore,
    goal: &Goal,
    mandate: &Mandate,
    question: &str,
) -> GoalRun {
    let run = claim_and_start_run(store, goal, mandate).await;
    let root_attempt = store
        .claim_task_with_lease(
            run.root_task_id.as_deref().unwrap(),
            "ask-root",
            Some("profile-task-lead"),
            7_200,
        )
        .await
        .unwrap()
        .unwrap();
    let mut decision = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Ask,
        "owner judgment requested",
        mandate.version,
    );
    decision.question = Some(question.to_string());
    store
        .record_mandate_decision(&decision, None, Some(&root_attempt.id))
        .await
        .unwrap();
    assert!(
        store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("ASK recorded".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    );
    store
        .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
            mandate_id: mandate.id.clone(),
            expected_mandate_version: mandate.version,
            goal_run_id: run.id.clone(),
            finalized_at: chrono::Utc::now().to_rfc3339(),
        })
        .await
        .unwrap();
    let parked = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(parked.status, MandateStatus::AwaitingInput);
    run
}

async fn make_review_due(store: &SqliteWorkStore, mandate: &Mandate) {
    sqlx::query("UPDATE mandates SET next_review_at = ? WHERE id = ?")
        .bind((chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339())
        .bind(&mandate.id)
        .execute(&store.pool)
        .await
        .unwrap();
}

/// Drive one review to a finalized WAIT carrying `owner_alert`, as the
/// task lead would. Returns the commit error when the decision is refused.
async fn finalize_wait_with_alert(
    store: &SqliteWorkStore,
    goal: &Goal,
    mandate: &Mandate,
    alert: &str,
) -> anyhow::Result<()> {
    let run = claim_and_start_run(store, goal, mandate).await;
    let root_attempt = store
        .claim_task_with_lease(
            run.root_task_id.as_deref().unwrap(),
            "alert-root",
            Some("profile-task-lead"),
            7_200,
        )
        .await
        .unwrap()
        .unwrap();
    let mut decision = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Wait,
        "the watched page answered 503",
        mandate.version,
    );
    decision.owner_alert = Some(alert.to_string());
    store
        .record_mandate_decision(&decision, None, Some(&root_attempt.id))
        .await?;
    assert!(
        store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("WAIT recorded".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    );
    store
        .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
            mandate_id: mandate.id.clone(),
            expected_mandate_version: mandate.version,
            goal_run_id: run.id.clone(),
            finalized_at: chrono::Utc::now().to_rfc3339(),
        })
        .await
        .unwrap();
    Ok(())
}

#[tokio::test]
async fn a_granted_owner_alert_reaches_the_owner_and_the_mandate_keeps_running() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 0);
    mandate.authority.owner_alerts = true;
    finalize_wait_with_alert(
        &store,
        &goal,
        &mandate,
        "The watched page is down (HTTP 503).",
    )
    .await
    .expect("a granted alert commits with its WAIT");
    let message = sqlx::query_scalar::<_, String>(
        "SELECT message FROM notification_queue
             WHERE goal_id = ? AND notification_type = 'mandate_alert'",
    )
    .bind(&goal.id)
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert!(
        message.contains("The watched page is down (HTTP 503)."),
        "{message}"
    );
    let after = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(after.status, MandateStatus::Active);
}

#[tokio::test]
async fn an_owner_alert_without_the_grant_is_refused() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 0);
    assert!(!mandate.authority.owner_alerts);
    let refused =
        finalize_wait_with_alert(&store, &goal, &mandate, "The watched page is down.").await;
    assert!(refused.is_err(), "an alert needs the owner's grant");
    let alerts = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM notification_queue WHERE notification_type = 'mandate_alert'",
    )
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(alerts, 0);
}

#[tokio::test]
async fn ask_notice_carries_the_generated_question_for_the_owner() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    park_mandate_on_ask(&store, &goal, &mandate, "May I also read reply counts?").await;
    let message = sqlx::query_scalar::<_, String>(
        "SELECT message FROM notification_queue
             WHERE goal_id = ? AND notification_type = 'mandate_ask'",
    )
    .bind(&goal.id)
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert!(
        message.contains("May I also read reply counts?"),
        "{message}"
    );
}

#[tokio::test]
async fn unanswered_ask_resumes_within_existing_authority_at_its_reconsider_time() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    park_mandate_on_ask(&store, &goal, &mandate, "May I also read reply counts?").await;

    // Not yet due: the owner still has the full reconsider window.
    assert!(
        store
            .claim_due_mandates(10, "synthetic-heartbeat", 300)
            .await
            .unwrap()
            .is_empty()
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

    make_review_due(&store, &mandate).await;
    let claimed = store
        .claim_due_mandates(10, "synthetic-heartbeat", 300)
        .await
        .unwrap();
    assert_eq!(claimed.len(), 1, "an unanswered ASK must not park forever");
    let resumed = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(resumed.status, MandateStatus::Active);
    assert!(resumed.suspension.is_none());
    assert_eq!(
        resumed.authority, mandate.authority,
        "resuming without an answer must not change authority"
    );
    assert_eq!(
        store.get_goal(&goal.id).await.unwrap().unwrap().status,
        "active"
    );
    let notice = sqlx::query_scalar::<_, String>(
        "SELECT message FROM notification_queue
             WHERE goal_id = ? AND notification_type = 'mandate_ask_timeout'",
    )
    .bind(&goal.id)
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert!(notice.contains("May I also read reply counts?"), "{notice}");
    // Named by the objective the owner confirmed, in plain language.
    assert!(notice.starts_with("Your automation \""), "{notice}");
    assert!(notice.contains("No reply is needed"), "{notice}");
    assert!(!notice.contains("within its current authority"), "{notice}");
}

#[tokio::test]
async fn safety_suspensions_never_auto_resume() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    let suspension = MandateSuspension::new(
        MandateSuspensionKind::ReconciliationRequired,
        Some("synthetic".to_string()),
    );
    sqlx::query(
        "UPDATE mandates SET status = 'awaiting_input', suspension_json = ?,
                 confirmed_at = COALESCE(confirmed_at, ?)
             WHERE id = ?",
    )
    .bind(serde_json::to_string(&suspension).unwrap())
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(&mandate.id)
    .execute(&store.pool)
    .await
    .unwrap();
    make_review_due(&store, &mandate).await;
    assert!(
        store
            .claim_due_mandates(10, "synthetic-heartbeat", 300)
            .await
            .unwrap()
            .is_empty()
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
}
