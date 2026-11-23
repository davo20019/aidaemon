//! Tests of creating, authorizing, updating and confirming mandates.

use super::*;

#[tokio::test]
async fn autopilot_without_a_measured_objective_stays_active_and_due() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 0);
    mandate.autonomy_mode = MandateAutonomyMode::Autopilot;
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    let claimed = store
        .claim_due_mandates(10, "synthetic-heartbeat", 300)
        .await
        .unwrap();
    assert_eq!(claimed.len(), 1);
    let current = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(current.status, MandateStatus::Active);
    assert!(current.objective_control.is_none());
    assert!(current.suspension.is_none());
}

#[tokio::test]
async fn controller_and_mandate_are_created_atomically() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    let loaded = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(loaded.goal_id, goal.id);
    assert_eq!(loaded.version, 1);
    assert!(loaded.review_lease_token.is_none());
    assert!(store.get_goal(&goal.id).await.unwrap().is_some());
    let channel_links: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM work_channel_links WHERE goal_id = ? AND channel_session_id = ?",
    )
    .bind(&goal.id)
    .bind(&goal.session_id)
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(channel_links, 1);

    let pending_goal =
        Goal::new_continuous_pending("Mandate awaiting confirmation", "owner-session", None, None);
    let mut pending_mandate = Mandate::new(
        &pending_goal.id,
        None,
        "Await explicit owner confirmation",
        "owner-session",
        authority(0),
        60,
        3_600,
        300,
    );
    pending_mandate.status = MandateStatus::Paused;
    pending_mandate.confirmed_at = None;
    store
        .create_mandate_controller(&pending_goal, &pending_mandate)
        .await
        .unwrap();
    assert_eq!(
        store
            .get_goal(&pending_goal.id)
            .await
            .unwrap()
            .unwrap()
            .status,
        "pending_confirmation"
    );
    assert_eq!(
        store
            .get_mandate(&pending_mandate.id)
            .await
            .unwrap()
            .unwrap()
            .status,
        MandateStatus::Paused
    );

    let (bad_goal, mut bad_mandate) = controller("owner-session", 2);
    bad_mandate.source_goal_id = Some("missing-personal-goal".to_string());
    assert!(
        store
            .create_mandate_controller(&bad_goal, &bad_mandate)
            .await
            .is_err()
    );
    assert!(store.get_goal(&bad_goal.id).await.unwrap().is_none());

    let same_owner_source = Goal::new_personal("Grow an audience", "owner-session");
    store.create_goal(&same_owner_source).await.unwrap();
    let (linked_goal, mut linked_mandate) = controller("owner-session", 2);
    linked_mandate.source_goal_id = Some(same_owner_source.id);
    store
        .create_mandate_controller(&linked_goal, &linked_mandate)
        .await
        .unwrap();
    assert!(store.get_goal(&linked_goal.id).await.unwrap().is_some());

    let foreign_source = Goal::new_personal("Grow an audience", "different-owner-session");
    store.create_goal(&foreign_source).await.unwrap();
    let (foreign_goal, mut foreign_mandate) = controller("owner-session", 2);
    foreign_mandate.source_goal_id = Some(foreign_source.id);
    assert!(
        store
            .create_mandate_controller(&foreign_goal, &foreign_mandate)
            .await
            .is_err()
    );
    assert!(store.get_goal(&foreign_goal.id).await.unwrap().is_none());
}

#[tokio::test]
async fn verified_private_owner_is_authorized_across_bot_routes() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("alpha_bot:12345", 1);
    assert_eq!(mandate.owner_principal_id, "principal:telegram:12345");
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    assert!(
        store
            .is_mandate_session_authorized(&mandate.id, "beta_bot:12345")
            .await
            .unwrap()
    );
    assert_eq!(
        store
            .list_mandates(Some("beta_bot:12345"), false)
            .await
            .unwrap()
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        [mandate.id.as_str()]
    );
    assert!(
        !store
            .is_mandate_session_authorized(&mandate.id, "beta_bot:67890")
            .await
            .unwrap()
    );

    let (group_goal, group_mandate) = controller("alpha_bot:-10012345", 1);
    assert_ne!(
        group_mandate.owner_principal_id,
        "principal:telegram:-10012345"
    );
    store
        .create_mandate_controller(&group_goal, &group_mandate)
        .await
        .unwrap();
    assert!(
        !store
            .is_mandate_session_authorized(&group_mandate.id, "beta_bot:-10012345")
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn persistence_rejects_text_bound_bypasses_before_writing() {
    let (store, _database) = test_store().await;
    let (bad_goal, mut bad_mandate) = controller("owner-session", 2);
    bad_mandate.objective = "é".repeat(1_025);
    assert!(
        store
            .create_mandate_controller(&bad_goal, &bad_mandate)
            .await
            .is_err()
    );
    assert!(store.get_goal(&bad_goal.id).await.unwrap().is_none());

    let (goal, mandate) = controller("owner-session", 2);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let oversized_decision = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Wait,
        &"é".repeat(1_025),
        mandate.version,
    );
    assert!(
        store
            .record_mandate_decision(&oversized_decision, None, None)
            .await
            .is_err()
    );

    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "A bounded governed action is useful",
        mandate.version,
    );
    let oversized_intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        &"é".repeat(513),
        "A bounded governed action is useful",
    );
    assert!(
        store
            .record_mandate_decision(&act, Some(&oversized_intention), None)
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
}

#[tokio::test]
async fn owner_update_is_cas_and_clears_a_live_lease() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    assert_eq!(
        store
            .claim_due_mandates(1, "heartbeat", 300)
            .await
            .unwrap()
            .len(),
        1
    );

    let mut update = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    update.version += 1;
    update.objective = "Updated owner objective".to_string();
    store.update_mandate(&update).await.unwrap();

    let loaded = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(loaded.version, 2);
    assert_eq!(loaded.objective, "Updated owner objective");
    assert!(loaded.review_lease_token.is_none());
    assert!(store.update_mandate(&update).await.is_err());

    let mut attempted_pause = loaded;
    attempted_pause.version += 1;
    attempted_pause.status = MandateStatus::Paused;
    assert!(store.update_mandate(&attempted_pause).await.is_err());
    assert!(
        store
            .transition_mandate_status(&mandate.id, MandateStatus::Active, MandateStatus::Paused,)
            .await
            .unwrap()
    );
    assert_eq!(
        store.get_goal(&goal.id).await.unwrap().unwrap().status,
        "paused"
    );
}

#[tokio::test]
async fn confirmation_is_durable_atomic_and_required_for_activation() {
    let (store, _database) = test_store().await;
    let goal = Goal::new_continuous_pending(
        "Mandate awaiting owner confirmation",
        "owner-session",
        None,
        None,
    );
    let mut mandate = Mandate::new(
        &goal.id,
        None,
        "Await explicit owner confirmation",
        "owner-session",
        authority(0),
        60,
        3_600,
        300,
    );
    mandate.status = MandateStatus::Paused;
    mandate.confirmed_at = None;
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    assert!(
        !store
            .transition_mandate_status(&mandate.id, MandateStatus::Paused, MandateStatus::Active,)
            .await
            .unwrap()
    );
    // Confirmation is a CAS against the exact policy version displayed to
    // the owner; a stale or speculative callback cannot activate it.
    assert!(
        !store
            .confirm_mandate(&mandate.id, mandate.version + 1, None)
            .await
            .unwrap()
    );
    assert_eq!(
        store
            .get_mandate(&mandate.id)
            .await
            .unwrap()
            .unwrap()
            .status,
        MandateStatus::Paused
    );
    assert!(
        store
            .confirm_mandate(&mandate.id, mandate.version, None)
            .await
            .unwrap()
    );
    assert!(
        !store
            .confirm_mandate(&mandate.id, mandate.version, None)
            .await
            .unwrap()
    );

    let confirmed = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(confirmed.status, MandateStatus::Active);
    assert_eq!(confirmed.version, 2);
    assert!(confirmed.confirmed_at.is_some());
    assert!(confirmed.is_active());
    let controller = store.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(controller.status, "active");
    assert_eq!(controller.dispatch_failures, 0);
}

#[tokio::test]
async fn stale_confirmation_cleanup_cancels_linked_mandate_and_cannot_activate_it() {
    let (store, _database) = test_store().await;
    let goal =
        Goal::new_continuous_pending("Stale mandate confirmation", "owner-session", None, None);
    let mut mandate = Mandate::new(
        &goal.id,
        None,
        "Never activate without confirmation",
        "owner-session",
        authority(0),
        60,
        3_600,
        300,
    );
    mandate.status = MandateStatus::Paused;
    mandate.confirmed_at = None;
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    sqlx::query("UPDATE goals SET created_at = ? WHERE id = ?")
        .bind((chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339())
        .bind(&goal.id)
        .execute(&store.pool)
        .await
        .unwrap();

    assert_eq!(
        store
            .cancel_stale_pending_confirmation_goals(3_600)
            .await
            .unwrap(),
        1
    );
    let cancelled = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(cancelled.status, MandateStatus::Cancelled);
    assert!(cancelled.confirmed_at.is_none());
    assert_eq!(
        store.get_goal(&goal.id).await.unwrap().unwrap().status,
        "cancelled"
    );
    assert!(
        !store
            .confirm_mandate(&mandate.id, mandate.version, None)
            .await
            .unwrap()
    );
    assert!(
            !store
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Cancelled,
                    MandateStatus::Active,
                )
                .await
                .unwrap()
        );
}
