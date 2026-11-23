//! Tests of mutations refused before dispatch.

use super::*;

async fn record_refused_receipt(
    store: &SqliteWorkStore,
    task_id: &str,
    reason_code: &str,
    effect: &str,
) {
    let data = serde_json::json!({
        "task_id": task_id,
        "receipt": {
            "invocation_stage": "rejected_before_dispatch",
            "outcome_status": "blocked",
            "access_denial": {"reason_code": reason_code},
            "semantics": {"effect": effect},
        },
    });
    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at, task_id)
             VALUES ('owner-session', 'tool_result', ?, ?, ?)",
    )
    .bind(data.to_string())
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(task_id)
    .execute(&store.pool)
    .await
    .unwrap();
}

/// An ACT whose worker was refused before dispatch and reported success.
/// `receipt` is the refusal's (reason code, effect), when there is one.
async fn finalize_refused_act(
    receipt: Option<(&str, &str)>,
) -> (
    SqliteWorkStore,
    tempfile::NamedTempFile,
    Task,
    GoalRun,
    MandateRunFinalizationResult,
) {
    let (store, database) = test_store().await;
    let (goal, mandate) = controller("owner-session", 1);
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let act = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "Post one update",
        mandate.version,
    );
    let intention = Intention::new(&mandate.id, &act.id, &run.id, "Post", "Useful");
    store
        .record_mandate_decision(&act, Some(&intention), None)
        .await
        .unwrap();
    let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
    if let Some((reason_code, effect)) = receipt {
        record_refused_receipt(&store, &fence.worker_task.id, reason_code, effect).await;
    }
    complete_mutation_fence(&store, &fence).await;
    let result = store
        .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
            mandate_id: mandate.id.clone(),
            expected_mandate_version: mandate.version,
            goal_run_id: run.id.clone(),
            finalized_at: chrono::Utc::now().to_rfc3339(),
        })
        .await
        .unwrap();
    (store, database, fence.worker_task, run, result)
}

#[tokio::test]
async fn a_mutation_refused_before_dispatch_names_its_gate() {
    let (store, _database, worker, run, result) =
        finalize_refused_act(Some(("untrusted_egress_denied", "Mutation"))).await;

    let gate = MandateDispatchGate::UntrustedContentCheck;
    let reason = MandateReconciliationReason::ActBlockedBeforeDispatch { gate };
    let MandateRunFinalizationResult::ReconciliationRequired {
        reason: actual,
        counts,
    } = result
    else {
        panic!("expected reconciliation, got {result:?}");
    };
    assert_eq!(actual, reason);
    assert_eq!(counts.mutation_reservations, 0);
    assert_eq!(counts.failed_or_blocked_tasks, 1, "{counts:?}");

    let (status, blocker): (String, Option<String>) =
        sqlx::query_as("SELECT status, blocker FROM tasks WHERE id = ?")
            .bind(&worker.id)
            .fetch_one(&store.pool)
            .await
            .unwrap();
    assert_eq!(status, "blocked");
    assert!(
        blocker.unwrap().contains(gate.owner_label()),
        "the task records its gate"
    );
    let outcome: String = sqlx::query_scalar("SELECT outcome_summary FROM goal_runs WHERE id = ?")
        .bind(&run.id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
    assert_eq!(
        outcome,
        "mandate_reconciliation_required:act_blocked_before_dispatch:untrusted_content_check"
    );
}

/// Without a refused mutation on a receipt there is no gate to name: a
/// refused read, or no receipt at all, leaves an ACT whose ledger proves
/// it changed nothing, which is retried without asking the owner.
#[tokio::test]
async fn only_a_refused_mutation_counts_as_blocked_before_dispatch() {
    for receipt in [None, Some(("untrusted_egress_denied", "ReadOnly"))] {
        let (store, _database, worker, run, result) = finalize_refused_act(receipt).await;
        assert_eq!(
            result,
            MandateRunFinalizationResult::Rejected {
                reason: MandateFinalizationRejectReason::ActChangedNothing,
            },
            "{receipt:?}"
        );
        let (mandate_status, intention_status): (String, String) = sqlx::query_as(
            "SELECT m.status, i.status FROM goal_runs r
                 JOIN mandates m ON m.goal_id = r.goal_id
                 JOIN intentions i ON i.goal_run_id = r.id
                 WHERE r.id = ?",
        )
        .bind(&run.id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(mandate_status, "active", "no owner reconciliation");
        assert_eq!(intention_status, "failed");
        let status: String = sqlx::query_scalar("SELECT status FROM tasks WHERE id = ?")
            .bind(&worker.id)
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(status, "completed");
    }
}

#[test]
fn the_owner_notice_names_the_gate_and_says_nothing_went_out() {
    let notice = crate::traits::MandateRunNotification::new(
        "mandate-synthetic",
        1,
        "goal-synthetic",
        "run-synthetic",
        "owner-session",
        crate::traits::MandateRunNotificationKind::ReconciliationRequired {
            reason: MandateReconciliationReason::ActBlockedBeforeDispatch {
                gate: MandateDispatchGate::UntrustedContentCheck,
            },
        },
        MandateRunProofCounts::default(),
        "2026-11-10T00:00:00Z",
    );

    let message = notice.message();

    assert!(message.contains("nothing was published"), "{message}");
    assert!(
        message.contains("untrusted-content safety check"),
        "{message}"
    );
}
