//! Tests of mandate action tasks and their dependencies.

use super::*;

/// A mandate step that depends on another has its dependency row from the
/// moment it is written, so a claim waits for the prerequisite (live
/// 2026-11-21: the rows came only from a rebuild at the next start).
#[tokio::test]
async fn a_mandate_step_waits_for_its_prerequisite_from_its_write() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "Two ordered steps are warranted",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Draft, then post, one bounded note",
        "It advances the mandate",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let root_task_id = run.root_task_id.as_deref().unwrap();
    let root_attempt = store
        .claim_task_with_lease(
            root_task_id,
            "root-worker",
            Some("profile-task-lead"),
            7_200,
        )
        .await
        .unwrap()
        .unwrap();
    let mut first = review_root(&goal, &run.id, "step-first");
    first.task_order = 1;
    let mut second = review_root(&goal, &run.id, "step-second");
    second.task_order = 2;
    second.depends_on = Some(serde_json::json!(["step-first"]).to_string());
    for task in [&first, &second] {
        assert!(
            store
                .create_mandate_task_from_attempt(
                    task,
                    &mandate.id,
                    mandate.version,
                    &run.id,
                    &root_attempt.id,
                    16,
                )
                .await
                .unwrap()
        );
    }
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = 'step-second'",
    )
    .fetch_all(&store.pool)
    .await
    .unwrap();
    assert_eq!(rows, ["step-first"]);
    assert!(
        store
            .claim_mandate_task_from_attempt(
                "step-second",
                "executor-worker",
                &mandate.id,
                mandate.version,
                &run.id,
                &root_attempt.id,
                7_200,
            )
            .await
            .unwrap()
            .is_none(),
        "not claimed before its prerequisite is done"
    );
}

/// An executor's blocker is input to the task lead, which owns the run's
/// outcome: the mandate run stays running so the lead can still call the
/// model (budget fence) and finalize. The lead's own blocker still blocks.
#[tokio::test]
async fn executor_blocker_leaves_the_mandate_run_to_its_task_lead() {
    let (store, _database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 2);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "One bounded note is warranted",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &act.id,
        &run.id,
        "Post one bounded note",
        "It advances the mandate",
    );
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    let blocked = |reason: &str| crate::traits::TaskAttemptPatch {
        status: "blocked".to_string(),
        result: None,
        error: None,
        blocker: Some(reason.to_string()),
        context: None,
        handoff: None,
    };
    async fn run_status(store: &SqliteWorkStore, run_id: &str) -> String {
        sqlx::query_scalar::<_, String>("SELECT status FROM goal_runs WHERE id = ?")
            .bind(run_id)
            .fetch_one(&store.pool)
            .await
            .unwrap()
    }

    assert!(
        store
            .patch_task_from_attempt(
                &fence.worker_attempt.id,
                &fence.worker_attempt.lease_token,
                &blocked("the target is unreachable"),
            )
            .await
            .unwrap()
    );
    assert_eq!(run_status(&store, &run.id).await, "running");

    assert!(
        store
            .patch_task_from_attempt(
                &fence.root_attempt.id,
                &fence.root_attempt.lease_token,
                &blocked("the lead cannot continue"),
            )
            .await
            .unwrap()
    );
    assert_eq!(run_status(&store, &run.id).await, "blocked");
}
