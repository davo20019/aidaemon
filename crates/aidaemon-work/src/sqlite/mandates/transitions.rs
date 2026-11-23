//! Status transitions shared by the store: controller goal status, intention transitions and invalidating open runs.

use super::*;

pub(super) fn controller_goal_status(mandate_status: MandateStatus) -> &'static str {
    match mandate_status {
        MandateStatus::Active => "active",
        MandateStatus::Paused | MandateStatus::AwaitingInput => "paused",
        MandateStatus::Completed => "completed",
        MandateStatus::Cancelled => "cancelled",
    }
}

pub(super) fn intention_transition(
    from: IntentionStatus,
    to: IntentionStatus,
) -> anyhow::Result<(&'static str, &'static str)> {
    anyhow::ensure!(
        from.can_transition_to(to),
        "invalid intention status transition {from} -> {to}"
    );
    Ok((from.as_str(), to.as_str()))
}

pub(super) async fn update_controller_status(
    connection: &mut sqlx::SqliteConnection,
    goal_id: &str,
    mandate_status: MandateStatus,
    now: &str,
) -> anyhow::Result<()> {
    let goal_status = controller_goal_status(mandate_status);
    let result = sqlx::query(
        "UPDATE goals
         SET status = ?, updated_at = ?,
             completed_at = CASE WHEN ? IN ('completed', 'cancelled') THEN ? ELSE NULL END
         WHERE id = ? AND domain = 'orchestration' AND goal_type = 'continuous'",
    )
    .bind(goal_status)
    .bind(now)
    .bind(goal_status)
    .bind(now)
    .bind(goal_id)
    .execute(connection)
    .await?;
    anyhow::ensure!(
        result.rows_affected() == 1,
        "mandate controller goal not found"
    );
    Ok(())
}

pub(super) async fn invalidate_open_mandate_runs(
    connection: &mut sqlx::SqliteConnection,
    goal_id: &str,
    now: &str,
) -> anyhow::Result<i64> {
    let claimed_unresolved: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)
         FROM mandate_mutation_attempts ma
         JOIN goal_runs gr ON gr.id = ma.goal_run_id
         WHERE gr.goal_id = ? AND gr.trigger_type = 'mandate'
           AND gr.status IN ('pending', 'running', 'blocked')
           AND ma.status = 'reserved' AND ma.dispatch_claimed_at IS NOT NULL",
    )
    .bind(goal_id)
    .fetch_one(&mut *connection)
    .await?;

    // Linearize authority revocation with the mutation ledger before leases
    // and runs are closed. A reservation that never reached the final
    // dispatcher is explicitly non-quota/non-ambiguous; once the one-use
    // dispatch claim exists, absence of a strict receipt is externally
    // indeterminate and must remain reconciliation-required.
    sqlx::query(
        "UPDATE mandate_mutation_attempts
         SET status = CASE
                 WHEN dispatch_claimed_at IS NULL THEN 'never_dispatched'
                 ELSE 'ambiguous'
             END,
             completed_at = COALESCE(completed_at, ?)
         WHERE status = 'reserved'
           AND goal_run_id IN (
               SELECT id FROM goal_runs
               WHERE goal_id = ? AND trigger_type = 'mandate'
                 AND status IN ('pending', 'running', 'blocked')
           )",
    )
    .bind(now)
    .bind(goal_id)
    .execute(&mut *connection)
    .await?;

    if claimed_unresolved > 0 {
        let rows = sqlx::query(
            "SELECT m.id AS mandate_id, m.version AS mandate_version,
                    gr.goal_id, gr.id AS goal_run_id, g.session_id,
                    (SELECT COUNT(*) FROM tasks t
                     WHERE t.goal_run_id = gr.id AND t.id != COALESCE(gr.root_task_id, ''))
                        AS non_root_tasks,
                    (SELECT COUNT(*) FROM tasks t
                     WHERE t.goal_run_id = gr.id AND t.id != COALESCE(gr.root_task_id, '')
                       AND t.status = 'completed'
                       AND COALESCE(NULLIF(trim(t.error), ''), '') = ''
                       AND COALESCE(NULLIF(trim(t.blocker), ''), '') = '')
                        AS completed_tasks,
                    (SELECT COUNT(*) FROM tasks t
                     WHERE t.goal_run_id = gr.id AND t.id != COALESCE(gr.root_task_id, '')
                       AND (t.status IN ('failed', 'blocked', 'interrupted')
                         OR COALESCE(NULLIF(trim(t.error), ''), '') != ''
                         OR COALESCE(NULLIF(trim(t.blocker), ''), '') != ''))
                        AS failed_or_blocked_tasks,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id) AS mutation_reservations,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id AND x.status = 'succeeded')
                        AS succeeded_mutations,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id AND x.status = 'failed')
                        AS failed_mutations,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id AND x.status = 'never_dispatched')
                        AS never_dispatched_mutations,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id AND x.status IN ('reserved', 'ambiguous'))
                        AS ambiguous_mutations
             FROM goal_runs gr
             JOIN mandates m ON m.goal_id = gr.goal_id
             JOIN goals g ON g.id = gr.goal_id
             WHERE gr.goal_id = ? AND gr.trigger_type = 'mandate'
               AND gr.status IN ('pending', 'running', 'blocked')
               AND EXISTS (
                   SELECT 1 FROM mandate_mutation_attempts ma
                   WHERE ma.goal_run_id = gr.id AND ma.status = 'ambiguous'
               )",
        )
        .bind(goal_id)
        .fetch_all(&mut *connection)
        .await?;
        let count = |row: &sqlx::sqlite::SqliteRow, name: &str| {
            u32::try_from(row.get::<i64, _>(name).max(0)).unwrap_or(u32::MAX)
        };
        for row in rows {
            let non_root_tasks = count(&row, "non_root_tasks");
            let completed_tasks = count(&row, "completed_tasks");
            let notice = crate::traits::MandateRunNotification::new(
                &row.get::<String, _>("mandate_id"),
                row.get("mandate_version"),
                &row.get::<String, _>("goal_id"),
                &row.get::<String, _>("goal_run_id"),
                &row.get::<String, _>("session_id"),
                crate::traits::MandateRunNotificationKind::AuthorityRevokedWithUnresolvedMutation,
                crate::traits::MandateRunProofCounts {
                    non_root_tasks,
                    completed_tasks,
                    incomplete_tasks: non_root_tasks.saturating_sub(completed_tasks),
                    failed_or_blocked_tasks: count(&row, "failed_or_blocked_tasks"),
                    mutation_reservations: count(&row, "mutation_reservations"),
                    succeeded_mutations: count(&row, "succeeded_mutations"),
                    failed_mutations: count(&row, "failed_mutations"),
                    never_dispatched_mutations: count(&row, "never_dispatched_mutations"),
                    ambiguous_or_reserved_mutations: count(&row, "ambiguous_mutations"),
                },
                now,
            );
            super::notifications::enqueue_mandate_run_notification_on_connection(
                &mut *connection,
                &notice,
            )
            .await?;
        }
    }
    sqlx::query(
        "UPDATE task_attempts
         SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
         WHERE goal_run_id IN (
             SELECT id FROM goal_runs
             WHERE goal_id = ? AND trigger_type = 'mandate'
               AND status IN ('pending', 'running', 'blocked')
         ) AND status IN ('claimed', 'running')",
    )
    .bind(now)
    .bind(goal_id)
    .execute(&mut *connection)
    .await?;
    sqlx::query(
        "UPDATE tasks
         SET status = 'cancelled', current_attempt_id = NULL,
             completed_at = COALESCE(completed_at, ?), updated_at = ?, version = version + 1
         WHERE goal_run_id IN (
             SELECT id FROM goal_runs
             WHERE goal_id = ? AND trigger_type = 'mandate'
               AND status IN ('pending', 'running', 'blocked')
         ) AND status IN ('pending', 'claimed', 'running', 'blocked')",
    )
    .bind(now)
    .bind(now)
    .bind(goal_id)
    .execute(&mut *connection)
    .await?;
    let (intention_from, intention_to) =
        intention_transition(IntentionStatus::Committed, IntentionStatus::Suspended)?;
    sqlx::query(
        "UPDATE intentions
         SET status = ?, completed_at = COALESCE(completed_at, ?), updated_at = ?
         WHERE goal_run_id IN (
             SELECT id FROM goal_runs
             WHERE goal_id = ? AND trigger_type = 'mandate'
               AND status IN ('pending', 'running', 'blocked')
         ) AND status = ?",
    )
    .bind(intention_to)
    .bind(now)
    .bind(now)
    .bind(goal_id)
    .bind(intention_from)
    .execute(&mut *connection)
    .await?;
    sqlx::query(
        "UPDATE goal_runs
         SET status = 'cancelled',
             outcome_summary = CASE WHEN EXISTS (
                 SELECT 1 FROM mandate_mutation_attempts ma
                 WHERE ma.goal_run_id = goal_runs.id AND ma.status = 'ambiguous'
             ) THEN 'mandate_reconciliation_required:lifecycle_invalidated_after_dispatch_claim'
             ELSE 'Mandate lifecycle changed; a fresh decision cycle is required.' END,
             completed_at = COALESCE(completed_at, ?), updated_at = ?
         WHERE goal_id = ? AND trigger_type = 'mandate'
           AND status IN ('pending', 'running', 'blocked')",
    )
    .bind(now)
    .bind(now)
    .bind(goal_id)
    .execute(&mut *connection)
    .await?;
    Ok(claimed_unresolved)
}
