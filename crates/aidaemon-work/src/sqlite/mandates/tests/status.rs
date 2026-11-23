//! Tests of lifecycle transitions and wake signals.

use super::*;

#[tokio::test]
async fn structured_signal_wakes_only_matching_autopilot_once() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 0);
    mandate.autonomy_mode = MandateAutonomyMode::Autopilot;
    mandate.objective_control = Some(objective_control());
    mandate.authority = MandateAuthority::from_operation_scopes(
        true,
        vec![MandateOperationScope {
            tool: "http_request".to_string(),
            operation: ToolCallOperation::Get,
            kind: MandateOperationKind::Observation,
            target_prefixes: vec![
                "https://api.example.test/v1/mentions".to_string(),
                "auth_profile:synthetic-social".to_string(),
                "account:synthetic-1".to_string(),
            ],
            allowed_query_params: vec!["since_id".to_string()],
            mutation_effects: Vec::new(),
        }],
        0,
        0,
        0,
    );
    mandate.next_review_at = (chrono::Utc::now() + chrono::Duration::hours(3)).to_rfc3339();
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    let signal = MandateWakeSignal {
        kind: MandateWakeSignalKind::Mention,
        source: "synthetic_social".to_string(),
        target_url: "https://api.example.test/v1/mentions?since_id=42".to_string(),
        account_id: Some("account:synthetic-1".to_string()),
        dedupe_key: "mention:42".to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339(),
    };

    assert_eq!(
        store.wake_mandates_for_signal(&signal).await.unwrap(),
        vec![mandate.id.clone()]
    );
    let awakened = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert!(
        chrono::DateTime::parse_from_rfc3339(&awakened.next_review_at).unwrap()
            <= chrono::Utc::now() + chrono::Duration::seconds(1)
    );
    assert!(
        store
            .wake_mandates_for_signal(&signal)
            .await
            .unwrap()
            .is_empty()
    );

    let mut wrong_account = signal;
    wrong_account.dedupe_key = "mention:43".to_string();
    wrong_account.account_id = Some("account:synthetic-2".to_string());
    assert!(
        store
            .wake_mandates_for_signal(&wrong_account)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn lifecycle_transition_revokes_open_cycle_and_stale_owner_snapshot() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "A bounded action was justified before the pause",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Perform one bounded action",
        "It advances the mandate",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;

    let mut stale_update = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    stale_update.version += 1;
    stale_update.objective = "A stale owner snapshot".to_string();

    assert!(
        store
            .transition_mandate_status(&mandate.id, MandateStatus::Active, MandateStatus::Paused,)
            .await
            .unwrap()
    );
    let paused = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(paused.status, MandateStatus::Paused);
    assert_eq!(paused.version, mandate.version + 1);
    assert!(
        store
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        store.get_goal_runs(&goal.id).await.unwrap()[0].status,
        "cancelled"
    );
    assert_eq!(
        store.list_intentions(&mandate.id, 1).await.unwrap()[0].status,
        IntentionStatus::Suspended
    );
    assert!(
        store
            .reserve_mandate_action_attempt(&reservation(
                &mandate,
                &act,
                &run,
                &fence,
                1,
                1,
                chrono::Utc::now(),
            ))
            .await
            .unwrap()
            .is_none()
    );

    assert!(
        store
            .transition_mandate_status(&mandate.id, MandateStatus::Paused, MandateStatus::Active,)
            .await
            .unwrap()
    );
    let resumed = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(resumed.version, mandate.version + 2);
    assert!(
        store
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .is_none()
    );

    // The update was prepared before the lifecycle epoch changed, so its
    // CAS must fail instead of restoring its stale active status/policy.
    assert!(store.update_mandate(&stale_update).await.is_err());

    let mut stale_cancel_racer = resumed;
    stale_cancel_racer.version += 1;
    stale_cancel_racer.objective = "Must not resurrect after cancel".to_string();
    assert!(
            store
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Active,
                    MandateStatus::Cancelled,
                )
                .await
                .unwrap()
        );
    assert!(store.update_mandate(&stale_cancel_racer).await.is_err());
    let cancelled = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(cancelled.status, MandateStatus::Cancelled);
    assert_eq!(cancelled.version, mandate.version + 3);
}
