//! Tests of the mandate store. Shared helpers live here; the tests are
//! grouped by concern in the child modules.

use super::*;
use crate::traits::*;
use crate::traits::{
    Goal, Intention, Mandate, MandateAuthority, MandateDecisionCycle, MandateMutationTarget,
    MandateObjectiveControl, MandateOperationKind, MandateOperationScope, MandateWakeSignalKind,
    ObjectiveMetricDirection, Task, ToolCallOperation,
};
use std::sync::Arc;

mod asks;
mod controller;
mod decisions;
mod finalize;
mod measurements;
mod mutations;
mod recovery;
mod refusals;
mod reviews;
mod schema;
mod status;
mod tasks;

async fn test_store() -> (SqliteWorkStore, tempfile::NamedTempFile) {
    let database = tempfile::NamedTempFile::new().unwrap();
    let store = open_for_tests(database.path().to_str().unwrap()).await;
    (store, database)
}

fn authority(max_actions: u32) -> MandateAuthority {
    MandateAuthority {
        allowed_tools: vec!["http_request".to_string()],
        allowed_mutation_effects: vec!["external_delivery".to_string()],
        allowed_target_prefixes: vec!["https://api.x.com/2/".to_string()],
        max_mutating_actions_per_cycle: max_actions,
        max_mutating_actions_per_rolling_24h: if max_actions == 0 { 0 } else { 24 },
        min_seconds_between_mutations: if max_actions == 0 { 0 } else { 900 },
        ..MandateAuthority::default()
    }
}

fn controller(session_id: &str, max_actions: u32) -> (Goal, Mandate) {
    let goal = Goal::new_continuous(
        "Steward the account",
        session_id,
        Some(10_000),
        Some(50_000),
    );
    let mut mandate = Mandate::new(
        &goal.id,
        None,
        "Maintain a useful, authentic account presence",
        session_id,
        authority(max_actions),
        60,
        3_600,
        300,
    );
    mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
    (goal, mandate)
}

fn objective_control() -> MandateObjectiveControl {
    MandateObjectiveControl {
        schema_version: MandateObjectiveControl::SCHEMA_VERSION,
        metric_name: "synthetic useful interactions".to_string(),
        unit: "count".to_string(),
        baseline_micros: 10_000_000,
        target_micros: 20_000_000,
        direction: ObjectiveMetricDirection::AtLeast,
        measurement_source: "metric_source:synthetic-analytics".to_string(),
        measurement_cadence_secs: 3_600,
        baseline_observed_at: chrono::Utc::now().to_rfc3339(),
    }
}

fn review_root(goal: &Goal, run_id: &str, task_id: &str) -> Task {
    Task {
        id: task_id.to_string(),
        goal_id: goal.id.clone(),
        description: format!("Mandate review: test controller for goal run {run_id}"),
        status: "pending".to_string(),
        priority: goal.priority.clone(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 0,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    }
}

async fn claim_and_start_run(
    store: &SqliteWorkStore,
    goal: &Goal,
    mandate: &Mandate,
) -> crate::traits::GoalRun {
    store
        .create_mandate_controller(goal, mandate)
        .await
        .unwrap();
    let claimed = store
        .claim_due_mandates(1, "test-heartbeat", 300)
        .await
        .unwrap();
    assert_eq!(claimed.len(), 1);
    let root_task_id = uuid::Uuid::new_v4().to_string();
    let run = store
        .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
        .await
        .unwrap();
    store
        .create_task(&review_root(goal, &run.id, &root_task_id))
        .await
        .unwrap();
    // Goal runs now remain pending until a worker claim transitions them.
    // These storage tests exercise post-claim mandate invariants without a
    // task worker, so model that already-claimed state explicitly.
    sqlx::query("UPDATE goal_runs SET status = 'running' WHERE id = ?")
        .bind(&run.id)
        .execute(&store.pool)
        .await
        .unwrap();
    run
}

struct MutationFence {
    root_attempt: crate::traits::TaskAttempt,
    worker_task: Task,
    worker_attempt: crate::traits::TaskAttempt,
}

async fn claim_mutation_fence(
    store: &SqliteWorkStore,
    goal: &Goal,
    mandate: &Mandate,
    run: &GoalRun,
) -> MutationFence {
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
    let worker_task_id = uuid::Uuid::new_v4().to_string();
    let mut worker_task = review_root(goal, &run.id, &worker_task_id);
    worker_task.description = "Perform one governed HTTP mutation".to_string();
    worker_task.task_order = 1;
    assert!(
        store
            .create_mandate_task_from_attempt(
                &worker_task,
                &mandate.id,
                mandate.version,
                &run.id,
                &root_attempt.id,
                16,
            )
            .await
            .unwrap()
    );
    let worker_attempt = store
        .claim_mandate_task_from_attempt(
            &worker_task_id,
            "executor-worker",
            &mandate.id,
            mandate.version,
            &run.id,
            &root_attempt.id,
            7_200,
        )
        .await
        .unwrap()
        .unwrap();
    MutationFence {
        root_attempt,
        worker_task,
        worker_attempt,
    }
}

fn reservation(
    mandate: &Mandate,
    decision: &MandateDecisionCycle,
    run: &GoalRun,
    fence: &MutationFence,
    reserved_action_attempt: i64,
    sequence: u64,
    reserved_at: chrono::DateTime<chrono::Utc>,
) -> MandateMutationReservation {
    MandateMutationReservation {
        grant: crate::traits::MandateAuthorityGrant {
            mandate_id: mandate.id.clone(),
            mandate_version: mandate.version,
            owner_principal_id: mandate.owner_principal_id.clone(),
            decision_cycle_id: decision.id.clone(),
            action_digest: format!("{sequence:064x}"),
            counts_toward_cycle_budget: true,
            reserved_action_attempt,
            tool_call_id: Some(format!("tool-call-{sequence}")),
        },
        goal_run_id: run.id.clone(),
        root_task_id: run.root_task_id.clone().unwrap(),
        root_task_attempt_id: fence.root_attempt.id.clone(),
        task_id: fence.worker_task.id.clone(),
        task_attempt_id: fence.worker_attempt.id.clone(),
        tool_call_id: format!("tool-call-{sequence}"),
        tool_name: "http_request".to_string(),
        mutation_effects: vec!["external_delivery".to_string()],
        targets: vec![MandateMutationTarget {
            kind: "url".to_string(),
            identifier: "https://api.x.com/2/tweets".to_string(),
        }],
        account_identifiers: Vec::new(),
        reserved_at: reserved_at.to_rfc3339(),
    }
}

fn dispatch_claim(
    reservation: &MandateMutationReservation,
    claimed_at: chrono::DateTime<chrono::Utc>,
) -> MandateMutationDispatchClaim {
    MandateMutationDispatchClaim {
        grant: reservation.grant.clone(),
        goal_run_id: reservation.goal_run_id.clone(),
        root_task_id: reservation.root_task_id.clone(),
        root_task_attempt_id: reservation.root_task_attempt_id.clone(),
        task_id: reservation.task_id.clone(),
        task_attempt_id: reservation.task_attempt_id.clone(),
        tool_call_id: reservation.tool_call_id.clone(),
        tool_name: reservation.tool_name.clone(),
        claimed_at: claimed_at.to_rfc3339(),
    }
}

async fn complete_mutation_fence(store: &SqliteWorkStore, fence: &MutationFence) {
    let complete = crate::traits::TaskAttemptPatch {
        status: "completed".to_string(),
        result: Some("verified".to_string()),
        ..Default::default()
    };
    assert!(
        store
            .patch_task_from_attempt(
                &fence.worker_attempt.id,
                &fence.worker_attempt.lease_token,
                &complete,
            )
            .await
            .unwrap()
    );
    assert!(
        store
            .patch_task_from_attempt(
                &fence.root_attempt.id,
                &fence.root_attempt.lease_token,
                &complete,
            )
            .await
            .unwrap()
    );
}

/// Settle a dispatched attempt with an explicit completion time, as the
/// canonical tool-result append does at its own event time. Quota and
/// cooldown tests need the time chosen; the transition is the same.
async fn finish_attempt(
    store: &SqliteWorkStore,
    reservation: &MandateMutationReservation,
    status: MandateMutationAttemptStatus,
    http_status: Option<u16>,
    completed_at: chrono::DateTime<chrono::Utc>,
) -> bool {
    sqlx::query(
        "UPDATE mandate_mutation_attempts
             SET status = ?, outcome_evidence = ?, http_status = ?, completed_at = ?
             WHERE mandate_id = ? AND mandate_version = ?
               AND decision_cycle_id = ? AND reserved_action_attempt = ?
               AND action_digest = ? AND goal_run_id = ?
               AND task_id = ? AND task_attempt_id = ? AND tool_call_id = ?
               AND status = 'reserved' AND dispatch_claimed_at IS NOT NULL",
    )
    .bind(match status {
        MandateMutationAttemptStatus::Succeeded => "succeeded",
        MandateMutationAttemptStatus::Failed => "failed",
        MandateMutationAttemptStatus::Ambiguous => "ambiguous",
        other => panic!("not a post-dispatch outcome: {other:?}"),
    })
    .bind("structured_metadata")
    .bind(http_status.map(i64::from))
    .bind(completed_at.to_rfc3339())
    .bind(&reservation.grant.mandate_id)
    .bind(reservation.grant.mandate_version)
    .bind(&reservation.grant.decision_cycle_id)
    .bind(reservation.grant.reserved_action_attempt)
    .bind(&reservation.grant.action_digest)
    .bind(&reservation.goal_run_id)
    .bind(&reservation.task_id)
    .bind(&reservation.task_attempt_id)
    .bind(&reservation.tool_call_id)
    .execute(&store.pool)
    .await
    .unwrap()
    .rows_affected()
        == 1
}
