//! Finalizing a mandate run from the proof its receipts give.

use super::*;

pub(super) async fn finalize_mandate_run_from_proof<S: WorkDatabase>(
    store: &S,
    request: &MandateRunFinalizationRequest,
) -> anyhow::Result<MandateRunFinalizationResult> {
    if validate_canonical_identifier("mandate id", &request.mandate_id, 256).is_err()
        || validate_canonical_identifier("goal run id", &request.goal_run_id, 256).is_err()
        || request.expected_mandate_version <= 0
        || validate_timestamp("mandate finalization timestamp", &request.finalized_at).is_err()
    {
        return Ok(MandateRunFinalizationResult::Rejected {
            reason: MandateFinalizationRejectReason::InvalidRequest,
        });
    }

    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
    // Acquire SQLite's writer lock before reading the proof snapshot. All
    // subsequent task, ledger, policy, and lifecycle checks therefore
    // linearize with owner updates and worker finalization writes.
    let fenced = sqlx::query(
        "UPDATE goal_runs SET updated_at = updated_at
             WHERE id = ? AND trigger_type = 'mandate' AND status = 'running'",
    )
    .bind(&request.goal_run_id)
    .execute(&mut *tx)
    .await?;
    if fenced.rows_affected() != 1 {
        tx.rollback().await?;
        return Ok(MandateRunFinalizationResult::Stale {
            reason: MandateFinalizationStaleReason::RunNotCurrent,
        });
    }

    let row = sqlx::query(
        "SELECT m.version AS mandate_version, m.status AS mandate_status,
                    m.min_review_secs AS min_review_secs,
                    m.max_review_secs AS max_review_secs,
                    g.session_id AS owner_session_id,
                    g.dispatch_failures AS review_failures,
                    gr.goal_id AS goal_id, gr.root_task_id AS root_task_id,
                    dc.id AS decision_cycle_id, dc.mandate_version AS decision_version,
                    dc.outcome AS decision_outcome, dc.origin AS decision_origin,
                    dc.action_attempts AS decision_action_attempts
             FROM goal_runs gr
             JOIN mandates m ON m.goal_id = gr.goal_id
             JOIN goals g ON g.id = gr.goal_id
             LEFT JOIN mandate_decision_cycles dc ON dc.goal_run_id = gr.id
             WHERE gr.id = ? AND m.id = ?",
    )
    .bind(&request.goal_run_id)
    .bind(&request.mandate_id)
    .fetch_optional(&mut *tx)
    .await?;
    let Some(row) = row else {
        tx.rollback().await?;
        return Ok(MandateRunFinalizationResult::Stale {
            reason: MandateFinalizationStaleReason::MandateMissingOrVersionChanged,
        });
    };
    let mandate_version: i64 = row.get("mandate_version");
    let min_review_secs: i64 = row.get("min_review_secs");
    let max_review_secs: i64 = row.get("max_review_secs");
    let review_failures: i32 = row.get("review_failures");
    let goal_id: String = row.get("goal_id");
    let owner_session_id: String = row.get("owner_session_id");
    if mandate_version != request.expected_mandate_version {
        tx.rollback().await?;
        return Ok(MandateRunFinalizationResult::Stale {
            reason: MandateFinalizationStaleReason::MandateMissingOrVersionChanged,
        });
    }
    let open_run_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM goal_runs
             WHERE goal_id = ? AND status IN ('pending', 'running', 'blocked')",
    )
    .bind(&goal_id)
    .fetch_one(&mut *tx)
    .await?;
    if open_run_count != 1 {
        tx.rollback().await?;
        return Ok(MandateRunFinalizationResult::Stale {
            reason: MandateFinalizationStaleReason::RunNotCurrent,
        });
    }

    macro_rules! close_review_failed_retry {
            ($reason:expr) => {{
                let reason = $reason;
                let (intention_from, intention_to) =
                    intention_transition(IntentionStatus::Committed, IntentionStatus::Failed)?;
                sqlx::query(
                    "UPDATE intentions
                     SET status = ?, updated_at = ?, completed_at = ?
                     WHERE goal_run_id = ? AND status = ?",
                )
                .bind(intention_to)
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .bind(intention_from)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE task_attempts
                     SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
                     WHERE goal_run_id = ? AND status IN ('claimed', 'running')",
                )
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE tasks
                     SET status = 'cancelled', current_attempt_id = NULL,
                         completed_at = COALESCE(completed_at, ?), updated_at = ?,
                         version = version + 1
                     WHERE goal_run_id = ?
                       AND status IN ('pending', 'claimed', 'running', 'blocked')",
                )
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let run_updated = sqlx::query(
                    "UPDATE goal_runs
                     SET status = 'failed', outcome_summary = ?, completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
                )
                .bind(format!("mandate_review_failed:{}", reason.as_str()))
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let finalized_at = chrono::DateTime::parse_from_rfc3339(&request.finalized_at)
                    .expect("validated mandate finalization timestamp")
                    .with_timezone(&chrono::Utc);
                let next_review_failures = review_failures.saturating_add(1);
                let backoff_shift = u32::try_from(next_review_failures.saturating_sub(1))
                    .unwrap_or(0)
                    .min(6);
                let retry_delay_secs = min_review_secs
                    .saturating_mul(1_i64 << backoff_shift)
                    .min(max_review_secs);
                let retry_at =
                    (finalized_at + chrono::Duration::seconds(retry_delay_secs)).to_rfc3339();
                let mandate_updated = sqlx::query(
                    "UPDATE mandates
                     SET next_review_at = ?, review_lease_token = NULL,
                         review_lease_expires_at = NULL, updated_at = ?
                     WHERE id = ? AND version = ? AND status = 'active'",
                )
                .bind(&retry_at)
                .bind(&request.finalized_at)
                .bind(&request.mandate_id)
                .bind(request.expected_mandate_version)
                .execute(&mut *tx)
                .await?;
                anyhow::ensure!(
                    run_updated.rows_affected() == 1 && mandate_updated.rows_affected() == 1,
                    "mandate review-failure retry state changed during finalization"
                );
                update_controller_status(
                    &mut tx,
                    &goal_id,
                    MandateStatus::Active,
                    &request.finalized_at,
                )
                .await?;
                sqlx::query(
                    "UPDATE goals SET dispatch_failures = ?, updated_at = ? WHERE id = ?",
                )
                .bind(next_review_failures)
                .bind(&request.finalized_at)
                .bind(&goal_id)
                .execute(&mut *tx)
                .await?;
                if crate::mandates::review_failure_is_notified(i64::from(next_review_failures)) {
                    let notice = crate::traits::MandateRunNotification::new(
                        &request.mandate_id,
                        request.expected_mandate_version,
                        &goal_id,
                        &request.goal_run_id,
                        &owner_session_id,
                        crate::traits::MandateRunNotificationKind::ReviewFailed { reason },
                        crate::traits::MandateRunProofCounts::default(),
                        &request.finalized_at,
                    );
                    super::notifications::enqueue_mandate_run_notification_on_connection(
                        &mut *tx,
                        &notice,
                    )
                    .await?;
                }
                tx.commit().await?;
                return Ok(MandateRunFinalizationResult::Rejected { reason });
            }};
        }

    macro_rules! close_invalid_decision_state {
            () => {{
                let reason = MandateFinalizationRejectReason::InvalidDecisionState;
                sqlx::query(
                    "UPDATE mandate_mutation_attempts
                     SET status = CASE
                             WHEN dispatch_claimed_at IS NULL THEN 'never_dispatched'
                             ELSE 'ambiguous'
                         END,
                         completed_at = COALESCE(completed_at, ?)
                     WHERE goal_run_id = ? AND mandate_id = ? AND status = 'reserved'",
                )
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .bind(&request.mandate_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE task_attempts
                     SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
                     WHERE goal_run_id = ? AND status IN ('claimed', 'running')",
                )
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE tasks
                     SET status = 'cancelled', current_attempt_id = NULL,
                         completed_at = COALESCE(completed_at, ?), updated_at = ?,
                         version = version + 1
                     WHERE goal_run_id = ?
                       AND status IN ('pending', 'claimed', 'running', 'blocked')",
                )
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let (intention_from, intention_to) = intention_transition(
                    IntentionStatus::Committed,
                    IntentionStatus::Suspended,
                )?;
                sqlx::query(
                    "UPDATE intentions
                     SET status = ?, updated_at = ?, completed_at = ?
                     WHERE goal_run_id = ? AND status = ?",
                )
                .bind(intention_to)
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .bind(intention_from)
                .execute(&mut *tx)
                .await?;
                let run_updated = sqlx::query(
                    "UPDATE goal_runs
                     SET status = 'failed', outcome_summary = ?, completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
                )
                .bind(format!("mandate_review_failed:{}", reason.as_str()))
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let mut suspension = MandateSuspension::new(
                    MandateSuspensionKind::ReviewFailed,
                    Some(reason.as_str().to_string()),
                );
                suspension.goal_run_id = Some(request.goal_run_id.clone());
                let mandate_updated = sqlx::query(
                    "UPDATE mandates
                     SET status = 'awaiting_input', review_lease_token = NULL,
                         review_lease_expires_at = NULL, suspension_json = ?, updated_at = ?
                     WHERE id = ? AND version = ? AND status = 'active'",
                )
                .bind(serde_json::to_string(&suspension)?)
                .bind(&request.finalized_at)
                .bind(&request.mandate_id)
                .bind(request.expected_mandate_version)
                .execute(&mut *tx)
                .await?;
                anyhow::ensure!(
                    run_updated.rows_affected() == 1 && mandate_updated.rows_affected() == 1,
                    "invalid mandate decision state changed during finalization"
                );
                update_controller_status(
                    &mut tx,
                    &goal_id,
                    MandateStatus::AwaitingInput,
                    &request.finalized_at,
                )
                .await?;
                let notice = crate::traits::MandateRunNotification::new(
                    &request.mandate_id,
                    request.expected_mandate_version,
                    &goal_id,
                    &request.goal_run_id,
                    &owner_session_id,
                    crate::traits::MandateRunNotificationKind::ReviewFailed { reason },
                    crate::traits::MandateRunProofCounts::default(),
                    &request.finalized_at,
                );
                super::notifications::enqueue_mandate_run_notification_on_connection(
                    &mut *tx,
                    &notice,
                )
                .await?;
                tx.commit().await?;
                return Ok(MandateRunFinalizationResult::Rejected { reason });
            }};
        }
    let decision_cycle_id: Option<String> = row.get("decision_cycle_id");
    let decision_version: Option<i64> = row.get("decision_version");
    let decision_outcome_raw: Option<String> = row.get("decision_outcome");
    let decision_origin = row
        .get::<Option<String>, _>("decision_origin")
        .as_deref()
        .and_then(MandateDecisionOrigin::parse);
    let decision_action_attempts: Option<i64> = row.get("decision_action_attempts");
    let (Some(decision_cycle_id), Some(decision_version), Some(decision_outcome_raw)) =
        (decision_cycle_id, decision_version, decision_outcome_raw)
    else {
        close_review_failed_retry!(MandateFinalizationRejectReason::DecisionMissing);
    };
    if decision_version != mandate_version {
        tx.rollback().await?;
        return Ok(MandateRunFinalizationResult::Stale {
            reason: MandateFinalizationStaleReason::DecisionVersionChanged,
        });
    }
    let Some(decision_outcome) = MandateDecisionOutcome::parse(&decision_outcome_raw) else {
        close_invalid_decision_state!();
    };
    if decision_origin == Some(MandateDecisionOrigin::RuntimeFallback) {
        close_review_failed_retry!(MandateFinalizationRejectReason::DeliberatorFailed);
    }
    let Some(decision_action_attempts) = decision_action_attempts else {
        close_invalid_decision_state!();
    };
    let mandate_status_raw: String = row.get("mandate_status");
    let Some(mandate_status) = MandateStatus::parse(&mandate_status_raw) else {
        close_invalid_decision_state!();
    };
    let expected_status = MandateStatus::Active;
    if mandate_status != expected_status {
        tx.rollback().await?;
        return Ok(MandateRunFinalizationResult::Stale {
            reason: MandateFinalizationStaleReason::MandateMissingOrVersionChanged,
        });
    }

    let root_task_id: Option<String> = row.get("root_task_id");
    let root_success = if let Some(root_task_id) = root_task_id.as_deref() {
        sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM tasks
                 WHERE id = ? AND goal_run_id = ? AND status = 'completed'
                   AND COALESCE(NULLIF(trim(error), ''), '') = ''
                   AND COALESCE(NULLIF(trim(blocker), ''), '') = ''",
        )
        .bind(root_task_id)
        .bind(&request.goal_run_id)
        .fetch_one(&mut *tx)
        .await?
            == 1
    } else {
        false
    };

    // Before the task counts: a worker refused before dispatch did not do
    // its work, whatever status it closed with.
    let reserved_mutations: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM mandate_mutation_attempts
             WHERE goal_run_id = ? AND mandate_id = ?",
    )
    .bind(&request.goal_run_id)
    .bind(&request.mandate_id)
    .fetch_one(&mut *tx)
    .await?;
    let refused_by = if decision_outcome == MandateDecisionOutcome::Act && reserved_mutations == 0 {
        block_worker_refused_before_dispatch(
            &mut tx,
            &request.goal_run_id,
            root_task_id.as_deref().unwrap_or(""),
            &request.finalized_at,
        )
        .await?
    } else {
        None
    };

    let task_counts = sqlx::query(
        "SELECT COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN status = 'completed'
                        AND COALESCE(NULLIF(trim(error), ''), '') = ''
                        AND COALESCE(NULLIF(trim(blocker), ''), '') = ''
                        THEN 1 ELSE 0 END), 0) AS completed,
                    COALESCE(SUM(CASE WHEN status IN ('failed', 'blocked', 'interrupted')
                        OR COALESCE(NULLIF(trim(error), ''), '') != ''
                        OR COALESCE(NULLIF(trim(blocker), ''), '') != ''
                        THEN 1 ELSE 0 END), 0) AS failed_or_blocked
             FROM tasks WHERE goal_run_id = ? AND id != ?",
    )
    .bind(&request.goal_run_id)
    .bind(root_task_id.as_deref().unwrap_or(""))
    .fetch_one(&mut *tx)
    .await?;
    let non_root_tasks: i64 = task_counts.get("total");
    let completed_tasks: i64 = task_counts.get("completed");
    let failed_or_blocked_tasks: i64 = task_counts.get("failed_or_blocked");

    // Close any crash window between durable reservation, the one-use
    // final-dispatch claim, and strict receipt projection before proving
    // the run. Unclaimed rows prove no adapter I/O was authorized; claimed
    // rows without a receipt are necessarily ambiguous.
    sqlx::query(
        "UPDATE mandate_mutation_attempts
             SET status = CASE
                     WHEN dispatch_claimed_at IS NULL THEN 'never_dispatched'
                     ELSE 'ambiguous'
                 END,
                 completed_at = COALESCE(completed_at, ?)
             WHERE goal_run_id = ? AND mandate_id = ? AND status = 'reserved'",
    )
    .bind(&request.finalized_at)
    .bind(&request.goal_run_id)
    .bind(&request.mandate_id)
    .execute(&mut *tx)
    .await?;

    let mutation_counts = sqlx::query(
        "SELECT COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END), 0) AS succeeded,
                    COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed,
                    COALESCE(SUM(CASE WHEN status = 'never_dispatched' THEN 1 ELSE 0 END), 0)
                        AS never_dispatched,
                    COALESCE(SUM(CASE WHEN status = 'ambiguous' THEN 1 ELSE 0 END), 0)
                        AS ambiguous,
                    COALESCE(SUM(CASE WHEN mandate_version != ? OR decision_cycle_id != ?
                        THEN 1 ELSE 0 END), 0) AS mismatched
             FROM mandate_mutation_attempts
             WHERE goal_run_id = ? AND mandate_id = ?",
    )
    .bind(mandate_version)
    .bind(&decision_cycle_id)
    .bind(&request.goal_run_id)
    .bind(&request.mandate_id)
    .fetch_one(&mut *tx)
    .await?;
    let mutation_total: i64 = mutation_counts.get("total");
    let mutation_succeeded: i64 = mutation_counts.get("succeeded");
    let mutation_failed: i64 = mutation_counts.get("failed");
    let mutation_never_dispatched: i64 = mutation_counts.get("never_dispatched");
    let mutation_ambiguous: i64 = mutation_counts.get("ambiguous");
    let mutation_mismatched: i64 = mutation_counts.get("mismatched");
    let count = |value: i64| u32::try_from(value.max(0)).unwrap_or(u32::MAX);
    let counts = MandateRunProofCounts {
        non_root_tasks: count(non_root_tasks),
        completed_tasks: count(completed_tasks),
        incomplete_tasks: count(non_root_tasks.saturating_sub(completed_tasks)),
        failed_or_blocked_tasks: count(failed_or_blocked_tasks),
        mutation_reservations: count(mutation_total),
        succeeded_mutations: count(mutation_succeeded),
        failed_mutations: count(mutation_failed),
        never_dispatched_mutations: count(mutation_never_dispatched),
        ambiguous_or_reserved_mutations: count(mutation_ambiguous),
    };
    macro_rules! reconcile_fail_closed {
            ($reason:expr) => {{
                let reason = $reason;
                sqlx::query(
                    "UPDATE task_attempts
                     SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
                     WHERE goal_run_id = ? AND status IN ('claimed', 'running')",
                )
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE tasks
                     SET status = 'cancelled', current_attempt_id = NULL,
                         completed_at = COALESCE(completed_at, ?), updated_at = ?, version = version + 1
                     WHERE goal_run_id = ? AND status IN ('pending', 'claimed', 'running')",
                )
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let (intention_from, intention_to) = intention_transition(
                    IntentionStatus::Committed,
                    IntentionStatus::Suspended,
                )?;
                sqlx::query(
                    "UPDATE intentions
                     SET status = ?, updated_at = ?, completed_at = ?
                     WHERE goal_run_id = ? AND status = ?",
                )
                .bind(intention_to)
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .bind(intention_from)
                .execute(&mut *tx)
                .await?;
                let run_updated = sqlx::query(
                    "UPDATE goal_runs
                     SET status = 'failed', outcome_summary = ?, completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
                )
                .bind(format!("mandate_reconciliation_required:{}", reason.as_str()))
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let mut suspension = MandateSuspension::new(
                    MandateSuspensionKind::ReconciliationRequired,
                    Some(reason.as_str().to_string()),
                );
                suspension.decision_cycle_id = Some(decision_cycle_id.clone());
                suspension.goal_run_id = Some(request.goal_run_id.clone());
                let mandate_updated = sqlx::query(
                    "UPDATE mandates
                     SET status = 'awaiting_input', review_lease_token = NULL,
                         review_lease_expires_at = NULL, suspension_json = ?, updated_at = ?
                     WHERE id = ? AND version = ? AND status = ?",
                )
                .bind(serde_json::to_string(&suspension)?)
                .bind(&request.finalized_at)
                .bind(&request.mandate_id)
                .bind(request.expected_mandate_version)
                .bind(expected_status.as_str())
                .execute(&mut *tx)
                .await?;
                if run_updated.rows_affected() != 1 || mandate_updated.rows_affected() != 1 {
                    tx.rollback().await?;
                    return Ok(MandateRunFinalizationResult::Stale {
                        reason: MandateFinalizationStaleReason::RunNotCurrent,
                    });
                }
                update_controller_status(
                    &mut tx,
                    &goal_id,
                    MandateStatus::AwaitingInput,
                    &request.finalized_at,
                )
                .await?;
                let notice = crate::traits::MandateRunNotification::new(
                    &request.mandate_id,
                    request.expected_mandate_version,
                    &goal_id,
                    &request.goal_run_id,
                    &owner_session_id,
                    crate::traits::MandateRunNotificationKind::ReconciliationRequired { reason },
                    counts.clone(),
                    &request.finalized_at,
                );
                super::notifications::enqueue_mandate_run_notification_on_connection(
                    &mut *tx,
                    &notice,
                )
                .await?;
                tx.commit().await?;
                return Ok(MandateRunFinalizationResult::ReconciliationRequired {
                    reason,
                    counts: counts.clone(),
                });
            }};
        }
    if let Some(gate) = refused_by {
        reconcile_fail_closed!(MandateReconciliationReason::ActBlockedBeforeDispatch { gate });
    }
    // Every external effect of an ACT goes through a mutation
    // reservation. When none succeeded, none has an unknown outcome and
    // none belongs to another decision, the ledger proves the review
    // changed nothing: retry it rather than ask the owner to confirm
    // what is already known.
    if decision_outcome == MandateDecisionOutcome::Act
        && mutation_succeeded == 0
        && mutation_ambiguous == 0
        && mutation_mismatched == 0
    {
        close_review_failed_retry!(MandateFinalizationRejectReason::ActChangedNothing);
    }
    if !root_success {
        reconcile_fail_closed!(MandateReconciliationReason::RootTaskNotSuccessful);
    }

    match decision_outcome {
        MandateDecisionOutcome::Act => {
            let intention_id = sqlx::query_scalar::<_, String>(
                "SELECT id FROM intentions
                     WHERE mandate_id = ? AND decision_cycle_id = ? AND goal_run_id = ?
                       AND status = 'committed' AND completed_at IS NULL",
            )
            .bind(&request.mandate_id)
            .bind(&decision_cycle_id)
            .bind(&request.goal_run_id)
            .fetch_optional(&mut *tx)
            .await?;
            let Some(intention_id) = intention_id else {
                reconcile_fail_closed!(MandateReconciliationReason::ActMissingIntention);
            };
            if non_root_tasks == 0 {
                reconcile_fail_closed!(MandateReconciliationReason::ActMissingWorkTask);
            }
            if completed_tasks != non_root_tasks || failed_or_blocked_tasks > 0 {
                reconcile_fail_closed!(MandateReconciliationReason::WorkTasksIncomplete);
            }
            if mutation_mismatched > 0 || decision_action_attempts != mutation_total {
                reconcile_fail_closed!(MandateReconciliationReason::ActionLedgerMismatch);
            }
            if mutation_failed > 0 {
                reconcile_fail_closed!(MandateReconciliationReason::MutationOutcomeFailed);
            }
            if mutation_never_dispatched > 0 {
                reconcile_fail_closed!(MandateReconciliationReason::MutationOutcomeFailed);
            }
            if mutation_ambiguous > 0 {
                reconcile_fail_closed!(MandateReconciliationReason::MutationOutcomeAmbiguous);
            }
            if mutation_succeeded == 0 {
                reconcile_fail_closed!(MandateReconciliationReason::ActMissingVerifiedMutation);
            }
            let (intention_from, intention_to) =
                intention_transition(IntentionStatus::Committed, IntentionStatus::Satisfied)?;
            let intention_updated = sqlx::query(
                "UPDATE intentions SET status = ?, updated_at = ?, completed_at = ?
                     WHERE id = ? AND status = ? AND completed_at IS NULL",
            )
            .bind(intention_to)
            .bind(&request.finalized_at)
            .bind(&request.finalized_at)
            .bind(&intention_id)
            .bind(intention_from)
            .execute(&mut *tx)
            .await?;
            if intention_updated.rows_affected() != 1 {
                tx.rollback().await?;
                return Ok(MandateRunFinalizationResult::Stale {
                    reason: MandateFinalizationStaleReason::RunNotCurrent,
                });
            }
            let run_updated = sqlx::query(
                "UPDATE goal_runs
                     SET status = 'completed', outcome_summary = 'mandate_act_satisfied',
                         completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
            )
            .bind(&request.finalized_at)
            .bind(&request.finalized_at)
            .bind(&request.goal_run_id)
            .execute(&mut *tx)
            .await?;
            if run_updated.rows_affected() != 1 {
                tx.rollback().await?;
                return Ok(MandateRunFinalizationResult::Stale {
                    reason: MandateFinalizationStaleReason::RunNotCurrent,
                });
            }
            let notice = crate::traits::MandateRunNotification::new(
                &request.mandate_id,
                request.expected_mandate_version,
                &goal_id,
                &request.goal_run_id,
                &owner_session_id,
                crate::traits::MandateRunNotificationKind::ActSatisfied,
                counts.clone(),
                &request.finalized_at,
            );
            super::notifications::enqueue_mandate_run_notification_on_connection(&mut tx, &notice)
                .await?;
            sqlx::query("UPDATE goals SET dispatch_failures = 0, updated_at = ? WHERE id = ?")
                .bind(&request.finalized_at)
                .bind(&goal_id)
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            Ok(MandateRunFinalizationResult::ActSatisfied { counts })
        }
        MandateDecisionOutcome::Wait
        | MandateDecisionOutcome::Ask
        | MandateDecisionOutcome::Stop => {
            if non_root_tasks != 0 {
                reconcile_fail_closed!(MandateReconciliationReason::NonActionCreatedWork);
            }
            if mutation_total != 0 || decision_action_attempts != 0 {
                reconcile_fail_closed!(MandateReconciliationReason::NonActionReservedMutation);
            }
            let intention_count: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM intentions
                     WHERE mandate_id = ? AND decision_cycle_id = ? AND goal_run_id = ?",
            )
            .bind(&request.mandate_id)
            .bind(&decision_cycle_id)
            .bind(&request.goal_run_id)
            .fetch_one(&mut *tx)
            .await?;
            if intention_count != 0 {
                close_invalid_decision_state!();
            }
            let final_mandate_status = match decision_outcome {
                MandateDecisionOutcome::Wait => MandateStatus::Active,
                MandateDecisionOutcome::Ask => MandateStatus::AwaitingInput,
                MandateDecisionOutcome::Stop => MandateStatus::Completed,
                MandateDecisionOutcome::Act => unreachable!(),
            };
            let suspension = (decision_outcome == MandateDecisionOutcome::Ask).then(|| {
                let mut value = MandateSuspension::new(
                    MandateSuspensionKind::AwaitingAnswer,
                    Some("agent_question".to_string()),
                );
                value.decision_cycle_id = Some(decision_cycle_id.clone());
                value.goal_run_id = Some(request.goal_run_id.clone());
                value
            });
            let mandate_updated = sqlx::query(
                "UPDATE mandates
                     SET status = ?, review_lease_token = NULL,
                         review_lease_expires_at = NULL, suspension_json = ?, updated_at = ?
                     WHERE id = ? AND version = ? AND status = 'active'",
            )
            .bind(final_mandate_status.as_str())
            .bind(suspension.as_ref().map(serde_json::to_string).transpose()?)
            .bind(&request.finalized_at)
            .bind(&request.mandate_id)
            .bind(request.expected_mandate_version)
            .execute(&mut *tx)
            .await?;
            if mandate_updated.rows_affected() != 1 {
                tx.rollback().await?;
                return Ok(MandateRunFinalizationResult::Stale {
                    reason: MandateFinalizationStaleReason::MandateMissingOrVersionChanged,
                });
            }
            update_controller_status(
                &mut tx,
                &goal_id,
                final_mandate_status,
                &request.finalized_at,
            )
            .await?;
            sqlx::query("UPDATE goals SET dispatch_failures = 0, updated_at = ? WHERE id = ?")
                .bind(&request.finalized_at)
                .bind(&goal_id)
                .execute(&mut *tx)
                .await?;
            let run_updated = sqlx::query(
                "UPDATE goal_runs
                     SET status = 'completed', outcome_summary = 'mandate_non_action_satisfied',
                         completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
            )
            .bind(&request.finalized_at)
            .bind(&request.finalized_at)
            .bind(&request.goal_run_id)
            .execute(&mut *tx)
            .await?;
            if run_updated.rows_affected() != 1 {
                tx.rollback().await?;
                return Ok(MandateRunFinalizationResult::Stale {
                    reason: MandateFinalizationStaleReason::RunNotCurrent,
                });
            }
            let owner_alert = if decision_outcome == MandateDecisionOutcome::Wait {
                sqlx::query_scalar::<_, Option<String>>(
                    "SELECT owner_alert FROM mandate_decision_cycles WHERE id = ?",
                )
                .bind(&decision_cycle_id)
                .fetch_optional(&mut *tx)
                .await?
                .flatten()
            } else {
                None
            };
            if let Some(kind) = match decision_outcome {
                MandateDecisionOutcome::Wait => owner_alert
                    .is_some()
                    .then_some(crate::traits::MandateRunNotificationKind::Alert),
                MandateDecisionOutcome::Ask => Some(crate::traits::MandateRunNotificationKind::Ask),
                MandateDecisionOutcome::Stop => {
                    Some(crate::traits::MandateRunNotificationKind::Stopped)
                }
                MandateDecisionOutcome::Act => unreachable!(),
            } {
                let question = if decision_outcome == MandateDecisionOutcome::Ask {
                    sqlx::query_scalar::<_, Option<String>>(
                        "SELECT question FROM mandate_decision_cycles WHERE id = ?",
                    )
                    .bind(&decision_cycle_id)
                    .fetch_optional(&mut *tx)
                    .await?
                    .flatten()
                } else {
                    None
                };
                let notice = crate::traits::MandateRunNotification::new(
                    &request.mandate_id,
                    request.expected_mandate_version,
                    &goal_id,
                    &request.goal_run_id,
                    &owner_session_id,
                    kind,
                    counts.clone(),
                    &request.finalized_at,
                )
                .with_owner_question(question.as_deref())
                .with_owner_alert(owner_alert.as_deref());
                super::notifications::enqueue_mandate_run_notification_on_connection(
                    &mut tx, &notice,
                )
                .await?;
            }
            tx.commit().await?;
            Ok(MandateRunFinalizationResult::NonActionSatisfied {
                outcome: decision_outcome,
                counts,
            })
        }
    }
}
