//! Tests of review-run creation, claiming and leases.

use super::*;

#[tokio::test]
async fn atomic_review_creation_rolls_back_run_when_root_insert_fails() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    let claimed = store
        .claim_due_mandates(1, "test-heartbeat", 300)
        .await
        .unwrap();
    let lease = claimed[0].review_lease_token.as_deref().unwrap();

    let other_goal = Goal::new_finite("Existing task owner", "owner-session");
    store.create_goal(&other_goal).await.unwrap();
    let existing_task = review_root(&other_goal, "other-run", "duplicate-root-id");
    store.create_task(&existing_task).await.unwrap();

    let attempted_run_id = uuid::Uuid::new_v4().to_string();
    let duplicate_root = review_root(&goal, &attempted_run_id, "duplicate-root-id");
    assert!(
        store
            .create_mandate_review_run(&mandate.id, lease, &attempted_run_id, &duplicate_root,)
            .await
            .is_err()
    );
    let orphan_count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM goal_runs WHERE id = ?")
        .bind(&attempted_run_id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
    assert_eq!(orphan_count, 0);

    let successful_run_id = uuid::Uuid::new_v4().to_string();
    let fresh_root = review_root(&goal, &successful_run_id, &uuid::Uuid::new_v4().to_string());
    let run = store
        .create_mandate_review_run(&mandate.id, lease, &successful_run_id, &fresh_root)
        .await
        .unwrap();
    assert_eq!(run.root_task_id.as_deref(), Some(fresh_root.id.as_str()));
    assert_eq!(
        store.get_tasks_for_goal_run(&run.id).await.unwrap().len(),
        1
    );
}

#[tokio::test]
async fn claiming_recovers_pre_atomic_run_created_without_a_root_task() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    let first_claim = store
        .claim_due_mandates(1, "old-heartbeat", 300)
        .await
        .unwrap();
    assert_eq!(first_claim.len(), 1);
    let interrupted = store
        .start_goal_run(&goal.id, "mandate", None, Some("missing-root"))
        .await
        .unwrap();

    let recovered = store
        .claim_due_mandates(1, "new-heartbeat", 300)
        .await
        .unwrap();
    assert_eq!(recovered.len(), 1);
    let old_run = store
        .get_goal_runs(&goal.id)
        .await
        .unwrap()
        .into_iter()
        .find(|run| run.id == interrupted.id)
        .unwrap();
    assert_eq!(old_run.status, "failed");
    assert!(
        old_run
            .outcome_summary
            .as_deref()
            .unwrap()
            .contains("interrupted mandate dispatch")
    );

    let run_id = uuid::Uuid::new_v4().to_string();
    let root = review_root(&goal, &run_id, &uuid::Uuid::new_v4().to_string());
    let new_run = store
        .create_mandate_review_run(
            &mandate.id,
            recovered[0].review_lease_token.as_deref().unwrap(),
            &run_id,
            &root,
        )
        .await
        .unwrap();
    assert_eq!(
        store
            .get_tasks_for_goal_run(&new_run.id)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn due_review_lease_has_one_winner_and_requires_its_token_to_release() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    let (left, right) = tokio::join!(
        store.claim_due_mandates(1, "heartbeat-a", 300),
        store.claim_due_mandates(1, "heartbeat-b", 300)
    );
    let mut winners = left.unwrap();
    winners.extend(right.unwrap());
    assert_eq!(winners.len(), 1);
    let lease_token = winners[0].review_lease_token.clone().unwrap();
    assert!(
        !store
            .release_mandate_review_lease(
                &mandate.id,
                "wrong-token",
                &(chrono::Utc::now() + chrono::Duration::minutes(5)).to_rfc3339(),
            )
            .await
            .unwrap()
    );

    let retry_at = (chrono::Utc::now() + chrono::Duration::minutes(5)).to_rfc3339();
    assert!(
        store
            .release_mandate_review_lease(&mandate.id, &lease_token, &retry_at)
            .await
            .unwrap()
    );
    let released = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(released.next_review_at, retry_at);
    assert!(released.review_lease_token.is_none());
    assert!(
        store
            .claim_due_mandates(1, "heartbeat-c", 300)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn claiming_reviews_atomically_closes_expired_mandates_and_controllers() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 2);
    mandate.expires_at = Some((chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339());
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    assert!(
        store
            .claim_due_mandates(1, "heartbeat", 300)
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
        MandateStatus::Completed
    );
    let controller = store.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(controller.status, "completed");
    assert!(controller.completed_at.is_some());
}
