//! Tests of recovering runs a crash left open.

use super::*;

#[tokio::test]
async fn completed_root_crash_without_dispatch_retries_after_grace_and_cancels_pending_child() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let root_task_id = run.root_task_id.as_deref().unwrap();
    let root_attempt = store
        .claim_task_with_lease(
            root_task_id,
            "crashing-root",
            Some("profile-task-lead"),
            7_200,
        )
        .await
        .unwrap()
        .unwrap();
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "One bounded child should perform the action",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Perform the bounded child action",
        "It advances the mandate",
    );
    store
        .record_mandate_decision(&act, Some(&intention), Some(&root_attempt.id))
        .await
        .unwrap();
    let child_id = uuid::Uuid::new_v4().to_string();
    let mut child = review_root(&goal, &run.id, &child_id);
    child.task_order = 1;
    child.description = "Pending governed child".to_string();
    assert!(
        store
            .create_mandate_task_from_attempt(
                &child,
                &mandate.id,
                mandate.version,
                &run.id,
                &root_attempt.id,
                16,
            )
            .await
            .unwrap()
    );
    assert!(
        store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("decision recorded; child queued".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
    );

    // The immediate finalizer gets a short uncontended window.
    assert!(
        store
            .claim_due_mandates(1, "grace-probe", 300)
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
        MandateStatus::Active
    );
    assert_eq!(
        store.get_task(&child_id).await.unwrap().unwrap().status,
        "pending"
    );

    let stale_completion = (chrono::Utc::now() - chrono::Duration::seconds(61)).to_rfc3339();
    sqlx::query("UPDATE tasks SET completed_at = ? WHERE id = ?")
        .bind(&stale_completion)
        .bind(root_task_id)
        .execute(&store.pool)
        .await
        .unwrap();
    assert!(
        store
            .claim_due_mandates(1, "reconciliation-probe", 300)
            .await
            .unwrap()
            .is_empty()
    );

    // The child never reserved a mutation, so nothing external can have
    // happened: the run fails and the mandate retries, unsuspended.
    assert_eq!(
        store
            .get_mandate(&mandate.id)
            .await
            .unwrap()
            .unwrap()
            .status,
        MandateStatus::Active
    );
    assert_eq!(
        store.get_goal(&goal.id).await.unwrap().unwrap().status,
        "active"
    );
    assert_eq!(
        store.get_task(&child_id).await.unwrap().unwrap().status,
        "cancelled"
    );
    assert_eq!(
        store.get_goal_runs(&goal.id).await.unwrap()[0]
            .outcome_summary
            .as_deref(),
        Some("mandate_review_failed:act_interrupted_before_dispatch")
    );
    let notification_message: String = sqlx::query_scalar(
        "SELECT message FROM notification_queue
             WHERE goal_id = ? AND notification_type = 'mandate_review_failed'
               AND priority = 'critical' AND expires_at IS NULL
             LIMIT 1",
    )
    .bind(&goal.id)
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert!(
        notification_message.contains("decided to act"),
        "{notification_message}"
    );
    assert!(
        notification_message.contains("interrupted before any action was dispatched"),
        "{notification_message}"
    );
    assert!(!notification_message.contains("stopped before it finished"));
    assert!(notification_message.contains("will retry automatically"));
    let intention_status =
        sqlx::query_scalar::<_, String>("SELECT status FROM intentions WHERE id = ?")
            .bind(&intention.id)
            .fetch_one(&store.pool)
            .await
            .unwrap();
    assert_eq!(intention_status, "suspended");
}

/// An ACT committed by a root whose daemon then stopped: no action task,
/// no reservation, root attempt gone.
async fn orphan_act_without_task(
    store: &SqliteWorkStore,
) -> (Goal, Mandate, crate::traits::GoalRun, Intention) {
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(store, &goal, &mandate).await;
    let root_task_id = run.root_task_id.clone().unwrap();
    let root_attempt = store
        .claim_task_with_lease(
            &root_task_id,
            "crashing-root",
            Some("profile-task-lead"),
            600,
        )
        .await
        .unwrap()
        .unwrap();
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "Post the drafted update",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Publish: Synthetic launch note for Acme Corp",
        "It advances the mandate",
    );
    store
        .record_mandate_decision(&act, Some(&intention), Some(&root_attempt.id))
        .await
        .unwrap();
    sqlx::query("UPDATE task_attempts SET lease_expires_at = ? WHERE id = ?")
        .bind((chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339())
        .bind(&root_attempt.id)
        .execute(&store.pool)
        .await
        .unwrap();
    sqlx::query("UPDATE tasks SET status = 'interrupted' WHERE id = ?")
        .bind(&root_task_id)
        .execute(&store.pool)
        .await
        .unwrap();
    (goal, mandate, run, intention)
}

async fn recover(store: &SqliteWorkStore) {
    let _ = store
        .claim_due_mandates(1, "recovery-probe", 300)
        .await
        .unwrap();
}

#[tokio::test]
async fn orphaned_act_without_task_resumes_once_with_saved_intention() {
    let (store, _database) = test_store().await;
    let (goal, mandate, run, intention) = orphan_act_without_task(&store).await;
    recover(&store).await;
    let root = store
        .get_task(run.root_task_id.as_deref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(root.status, "pending");
    assert_eq!(root.retry_count, 1);
    assert!(
        root.description.contains(&intention.description),
        "{}",
        root.description
    );
    let runs = store.get_goal_runs(&goal.id).await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].status, "pending");
    assert_eq!(
        sqlx::query_scalar::<_, String>("SELECT status FROM intentions WHERE id = ?")
            .bind(&intention.id)
            .fetch_one(&store.pool)
            .await
            .unwrap(),
        "committed"
    );
    assert_eq!(
        store
            .get_mandate(&mandate.id)
            .await
            .unwrap()
            .unwrap()
            .status,
        MandateStatus::Active
    );
    // A second recovery while it waits for dispatch changes nothing.
    recover(&store).await;
    let tasks = store.get_tasks_for_goal_run(&run.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].status, "pending");
    assert_eq!(store.get_goal_runs(&goal.id).await.unwrap().len(), 1);
}

#[tokio::test]
async fn resumed_act_interrupted_again_falls_back_instead_of_resuming_twice() {
    let (store, _database) = test_store().await;
    let (goal, _mandate, run, intention) = orphan_act_without_task(&store).await;
    let root_task_id = run.root_task_id.clone().unwrap();
    recover(&store).await;
    // The resumed lead is interrupted too, again before any task.
    let attempt = store
        .claim_task_with_lease(
            &root_task_id,
            "resumed-root",
            Some("profile-task-lead"),
            600,
        )
        .await
        .unwrap()
        .unwrap();
    sqlx::query("UPDATE task_attempts SET lease_expires_at = ? WHERE id = ?")
        .bind((chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339())
        .bind(&attempt.id)
        .execute(&store.pool)
        .await
        .unwrap();
    sqlx::query("UPDATE tasks SET status = 'interrupted' WHERE id = ?")
        .bind(&root_task_id)
        .execute(&store.pool)
        .await
        .unwrap();
    recover(&store).await;
    assert_eq!(
        store.get_goal_runs(&goal.id).await.unwrap()[0]
            .outcome_summary
            .as_deref(),
        Some("mandate_review_failed:act_interrupted_before_dispatch")
    );
    assert_eq!(
        sqlx::query_scalar::<_, String>("SELECT status FROM intentions WHERE id = ?")
            .bind(&intention.id)
            .fetch_one(&store.pool)
            .await
            .unwrap(),
        "suspended"
    );
}

#[tokio::test]
async fn orphaned_act_with_existing_action_task_is_not_resumed() {
    let (store, _database) = test_store().await;
    let (goal, _mandate, run, _intention) = orphan_act_without_task(&store).await;
    // An action task already exists for the cycle: never resume.
    sqlx::query("UPDATE tasks SET retry_count = 0 WHERE id = ?")
        .bind(run.root_task_id.as_deref().unwrap())
        .execute(&store.pool)
        .await
        .unwrap();
    let child_id = uuid::Uuid::new_v4().to_string();
    let mut child = review_root(&goal, &run.id, &child_id);
    child.task_order = 1;
    store.create_task(&child).await.unwrap();
    recover(&store).await;
    let runs = store.get_goal_runs(&goal.id).await.unwrap();
    assert_ne!(runs[0].status, "pending");
    assert_eq!(
        runs[0].outcome_summary.as_deref(),
        Some("mandate_review_failed:act_interrupted_before_dispatch")
    );
}

#[tokio::test]
async fn stale_or_version_changed_act_is_not_resumed() {
    for case in ["stale", "version"] {
        let (store, _database) = test_store().await;
        let (goal, mandate, run, _intention) = orphan_act_without_task(&store).await;
        match case {
            "stale" => {
                sqlx::query(
                    "UPDATE mandate_decision_cycles SET created_at = ? WHERE goal_run_id = ?",
                )
                .bind((chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339())
                .bind(&run.id)
                .execute(&store.pool)
                .await
                .unwrap();
            }
            _ => {
                sqlx::query("UPDATE mandates SET version = version + 1 WHERE id = ?")
                    .bind(&mandate.id)
                    .execute(&store.pool)
                    .await
                    .unwrap();
            }
        }
        recover(&store).await;
        let root = store
            .get_task(run.root_task_id.as_deref().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(root.retry_count, 0, "{case}");
        assert_ne!(root.status, "pending", "{case}");
        if case == "stale" {
            assert_eq!(
                store.get_goal_runs(&goal.id).await.unwrap()[0]
                    .outcome_summary
                    .as_deref(),
                Some("mandate_review_failed:act_interrupted_before_dispatch")
            );
        }
    }
}

#[tokio::test]
async fn orphaned_review_without_decision_retries_without_reconciliation_pause() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let root_attempt = store
        .claim_task_with_lease(
            run.root_task_id.as_deref().unwrap(),
            "crashing-reviewer",
            Some("profile-task-lead"),
            180,
        )
        .await
        .unwrap()
        .unwrap();
    sqlx::query("UPDATE task_attempts SET lease_expires_at = ? WHERE id = ?")
        .bind((chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339())
        .bind(&root_attempt.id)
        .execute(&store.pool)
        .await
        .unwrap();

    assert!(
        store
            .claim_due_mandates(1, "orphan-recovery", 300)
            .await
            .unwrap()
            .is_empty()
    );

    let recovered_mandate = store.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert_eq!(recovered_mandate.status, MandateStatus::Active);
    assert!(recovered_mandate.next_review_at > chrono::Utc::now().to_rfc3339());
    assert_eq!(
        store.get_goal(&goal.id).await.unwrap().unwrap().status,
        "active"
    );
    let recovered_run = store
        .get_goal_runs(&goal.id)
        .await
        .unwrap()
        .into_iter()
        .find(|candidate| candidate.id == run.id)
        .unwrap();
    assert_eq!(recovered_run.status, "failed");
    assert_eq!(
        recovered_run.outcome_summary.as_deref(),
        Some("mandate_review_failed:decision_missing")
    );

    let root = store
        .get_task(run.root_task_id.as_deref().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(root.status, "cancelled");
    let notification: (String, String) = sqlx::query_as(
        "SELECT notification_type, message FROM notification_queue
             WHERE id = ?",
    )
    .bind(format!("mandate-run-notice:{}", run.id))
    .fetch_one(&store.pool)
    .await
    .unwrap();
    assert_eq!(notification.0, "mandate_review_failed");
    assert!(notification.1.contains("before it decided anything"));
    assert!(!notification.1.contains("decision missing"));
    assert!(notification.1.contains("Nothing was done"));
    assert!(!notification.1.contains("check the account"));
}

#[tokio::test]
async fn stale_controller_repair_cannot_revive_an_owner_cancellation() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    store
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
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

    assert!(
        !store
            .keep_mandate_controller_active(&mandate.id, mandate.version)
            .await
            .unwrap()
    );
    assert_eq!(
        store.get_goal(&goal.id).await.unwrap().unwrap().status,
        "cancelled"
    );
}
