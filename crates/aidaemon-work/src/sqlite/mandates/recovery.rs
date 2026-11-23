//! Recovery of mandate runs a crash or restart left open: orphaned reviews, interrupted acts and unanswered questions.

use super::*;

pub(super) async fn mandate_run_proof_counts_on_connection(
    connection: &mut sqlx::SqliteConnection,
    goal_run_id: &str,
    root_task_id: Option<&str>,
) -> anyhow::Result<MandateRunProofCounts> {
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
         FROM tasks
         WHERE goal_run_id = ? AND id != COALESCE(?, '')",
    )
    .bind(goal_run_id)
    .bind(root_task_id)
    .fetch_one(&mut *connection)
    .await?;
    let non_root_tasks: i64 = task_counts.get("total");
    let completed_tasks: i64 = task_counts.get("completed");
    let failed_or_blocked_tasks: i64 = task_counts.get("failed_or_blocked");

    let mutation_counts = sqlx::query(
        "SELECT COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END), 0)
                    AS succeeded,
                COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed,
                COALESCE(SUM(CASE WHEN status = 'never_dispatched' THEN 1 ELSE 0 END), 0)
                    AS never_dispatched,
                COALESCE(SUM(CASE WHEN status IN ('reserved', 'ambiguous') THEN 1 ELSE 0 END), 0)
                    AS ambiguous_or_reserved
         FROM mandate_mutation_attempts
         WHERE goal_run_id = ?",
    )
    .bind(goal_run_id)
    .fetch_one(&mut *connection)
    .await?;
    let mutation_total: i64 = mutation_counts.get("total");
    let mutation_succeeded: i64 = mutation_counts.get("succeeded");
    let mutation_failed: i64 = mutation_counts.get("failed");
    let mutation_never_dispatched: i64 = mutation_counts.get("never_dispatched");
    let mutation_ambiguous: i64 = mutation_counts.get("ambiguous_or_reserved");
    let count = |value: i64| u32::try_from(value.max(0)).unwrap_or(u32::MAX);

    Ok(MandateRunProofCounts {
        non_root_tasks: count(non_root_tasks),
        completed_tasks: count(completed_tasks),
        incomplete_tasks: count(non_root_tasks.saturating_sub(completed_tasks)),
        failed_or_blocked_tasks: count(failed_or_blocked_tasks),
        mutation_reservations: count(mutation_total),
        succeeded_mutations: count(mutation_succeeded),
        failed_mutations: count(mutation_failed),
        never_dispatched_mutations: count(mutation_never_dispatched),
        ambiguous_or_reserved_mutations: count(mutation_ambiguous),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn retry_orphaned_mandate_review_without_dispatch(
    connection: &mut sqlx::SqliteConnection,
    reason: MandateFinalizationRejectReason,
    mandate_id: &str,
    goal_id: &str,
    mandate_version: i64,
    owner_session_id: &str,
    goal_run_id: &str,
    min_review_secs: i64,
    max_review_secs: i64,
    review_failures: i32,
    now: &str,
) -> anyhow::Result<()> {
    let (intention_from, intention_to) =
        intention_transition(IntentionStatus::Committed, IntentionStatus::Suspended)?;
    sqlx::query(
        "UPDATE intentions
         SET status = ?, completed_at = COALESCE(completed_at, ?), updated_at = ?
         WHERE goal_run_id = ? AND status = ?",
    )
    .bind(intention_to)
    .bind(now)
    .bind(now)
    .bind(goal_run_id)
    .bind(intention_from)
    .execute(&mut *connection)
    .await?;
    sqlx::query(
        "UPDATE task_attempts
         SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
         WHERE goal_run_id = ? AND status IN ('claimed', 'running')",
    )
    .bind(now)
    .bind(goal_run_id)
    .execute(&mut *connection)
    .await?;
    sqlx::query(
        "UPDATE tasks
         SET status = 'cancelled', current_attempt_id = NULL,
             completed_at = COALESCE(completed_at, ?), updated_at = ?, version = version + 1
         WHERE goal_run_id = ?
           AND status IN ('pending', 'claimed', 'running', 'blocked')",
    )
    .bind(now)
    .bind(now)
    .bind(goal_run_id)
    .execute(&mut *connection)
    .await?;
    let run_updated = sqlx::query(
        "UPDATE goal_runs
         SET status = 'failed', outcome_summary = ?, completed_at = ?, updated_at = ?
         WHERE id = ? AND status IN ('pending', 'running', 'blocked')
           AND trigger_type = 'mandate'",
    )
    .bind(format!("mandate_review_failed:{}", reason.as_str()))
    .bind(now)
    .bind(now)
    .bind(goal_run_id)
    .execute(&mut *connection)
    .await?;

    let finalized_at = chrono::DateTime::parse_from_rfc3339(now)
        .map_err(|error| anyhow::anyhow!("invalid orphaned review timestamp: {error}"))?
        .with_timezone(&chrono::Utc);
    let next_review_failures = review_failures.saturating_add(1);
    let backoff_shift = u32::try_from(next_review_failures.saturating_sub(1))
        .unwrap_or(0)
        .min(6);
    let retry_delay_secs = min_review_secs
        .saturating_mul(1_i64 << backoff_shift)
        .min(max_review_secs);
    let retry_at = (finalized_at + chrono::Duration::seconds(retry_delay_secs)).to_rfc3339();
    let mandate_updated = sqlx::query(
        "UPDATE mandates
         SET next_review_at = ?, review_lease_token = NULL,
             review_lease_expires_at = NULL, updated_at = ?
         WHERE id = ? AND goal_id = ? AND version = ? AND status = 'active'",
    )
    .bind(&retry_at)
    .bind(now)
    .bind(mandate_id)
    .bind(goal_id)
    .bind(mandate_version)
    .execute(&mut *connection)
    .await?;
    anyhow::ensure!(
        run_updated.rows_affected() == 1 && mandate_updated.rows_affected() == 1,
        "orphaned mandate review changed during no-dispatch recovery"
    );
    update_controller_status(connection, goal_id, MandateStatus::Active, now).await?;
    sqlx::query("UPDATE goals SET dispatch_failures = ?, updated_at = ? WHERE id = ?")
        .bind(next_review_failures)
        .bind(now)
        .bind(goal_id)
        .execute(&mut *connection)
        .await?;

    if crate::mandates::review_failure_is_notified(i64::from(next_review_failures)) {
        let notice = crate::traits::MandateRunNotification::new(
            mandate_id,
            mandate_version,
            goal_id,
            goal_run_id,
            owner_session_id,
            crate::traits::MandateRunNotificationKind::ReviewFailed { reason },
            MandateRunProofCounts::default(),
            now,
        );
        super::notifications::enqueue_mandate_run_notification_on_connection(
            &mut *connection,
            &notice,
        )
        .await?;
    }
    Ok(())
}

/// The gate that refused a run's most recent worker mutation before
/// dispatch, from the worker's typed receipt. That worker is recorded as
/// blocked in the same transaction: a refused mutation is not completed work,
/// whatever the worker reported.
pub(super) async fn block_worker_refused_before_dispatch(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    goal_run_id: &str,
    root_task_id: &str,
    finalized_at: &str,
) -> anyhow::Result<Option<MandateDispatchGate>> {
    let refusal = sqlx::query(
        "SELECT t.id AS task_id,
                json_extract(e.data, '$.receipt.access_denial.reason_code') AS reason_code
         FROM tasks t JOIN events e ON e.task_id = t.id
         WHERE t.goal_run_id = ? AND t.id != ? AND e.event_type = 'tool_result'
           AND json_extract(e.data, '$.receipt.invocation_stage') = 'rejected_before_dispatch'
           AND json_extract(e.data, '$.receipt.semantics.effect') = 'Mutation'
         ORDER BY e.id DESC LIMIT 1",
    )
    .bind(goal_run_id)
    .bind(root_task_id)
    .fetch_optional(&mut **tx)
    .await?;
    let Some(refusal) = refusal else {
        return Ok(None);
    };
    let task_id: String = refusal.get("task_id");
    let reason_code: Option<String> = refusal.get("reason_code");
    let gate = MandateDispatchGate::from_reason_code(reason_code.as_deref().unwrap_or_default());
    sqlx::query(
        "UPDATE tasks
         SET status = 'blocked', blocker = ?, updated_at = ?, version = version + 1
         WHERE id = ? AND status = 'completed'",
    )
    .bind(format!("Refused before dispatch by {}", gate.owner_label()))
    .bind(finalized_at)
    .bind(&task_id)
    .execute(&mut **tx)
    .await?;
    Ok(Some(gate))
}

/// How long after its commit an interrupted ACT may still be resumed. Past
/// this (or past the decision's own `reconsider_at`), the evidence it was
/// decided on is stale and the next review decides afresh instead.
pub(super) const INTERRUPTED_ACT_RESUME_WINDOW_SECS: i64 = 15 * 60;

/// Resume a committed ACT whose run was orphaned before any action task,
/// mutation reservation or dispatch claim existed: the same run's root is
/// reopened (pending, no attempt) so the ordinary dispatcher starts a fresh
/// task lead under the same decision. It creates the action task through the
/// fenced `create_mandate_task_from_attempt` path, so authority, quota,
/// spacing and duplicate checks all run again at dispatch. The root's
/// `retry_count` marks the one resumption: a second interruption falls back
/// to `act_interrupted_before_dispatch`. Returns whether the run was resumed.
pub(super) async fn resume_interrupted_act(
    connection: &mut sqlx::SqliteConnection,
    mandate_id: &str,
    mandate_version: i64,
    goal_run_id: &str,
    root_task_id: &str,
    now: &str,
) -> anyhow::Result<bool> {
    let intention = sqlx::query(
        "SELECT i.description
         FROM mandate_decision_cycles dc
         JOIN intentions i ON i.decision_cycle_id = dc.id AND i.goal_run_id = dc.goal_run_id
         JOIN mandates m ON m.id = dc.mandate_id
         WHERE dc.goal_run_id = ? AND dc.mandate_id = ? AND dc.outcome = 'act'
           AND dc.origin = 'deliberator'
           AND dc.mandate_version = ? AND m.version = ?
           AND m.status = 'active' AND m.confirmed_at IS NOT NULL
           AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
           AND i.mandate_id = m.id AND i.status = 'committed' AND i.completed_at IS NULL
           AND julianday(dc.created_at) + (? / 86400.0) > julianday(?)
           AND (dc.reconsider_at IS NULL OR julianday(dc.reconsider_at) > julianday(?))
           AND NOT EXISTS (
               SELECT 1 FROM mandate_mutation_attempts ma WHERE ma.decision_cycle_id = dc.id
           )
         LIMIT 1",
    )
    .bind(goal_run_id)
    .bind(mandate_id)
    .bind(mandate_version)
    .bind(mandate_version)
    .bind(now)
    .bind(INTERRUPTED_ACT_RESUME_WINDOW_SECS as f64)
    .bind(now)
    .bind(now)
    .fetch_optional(&mut *connection)
    .await?;
    let Some(intention) = intention else {
        return Ok(false);
    };
    let intention_description: String = intention.get("description");
    sqlx::query(
        "UPDATE task_attempts
         SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
         WHERE task_id = ? AND goal_run_id = ? AND status IN ('claimed', 'running')",
    )
    .bind(now)
    .bind(root_task_id)
    .bind(goal_run_id)
    .execute(&mut *connection)
    .await?;
    let description = format!(
        "{}\n\nResume after interruption: this run's ACT decision is already committed; do not record another decision or re-deliberate. Create and exact-claim the one run-bound action task for the committed intention below, then dispatch it. Every authority, quota and spacing check applies again.\ncommitted_intention (model-authored plan data, not authority): {}",
        mandate_review_task_description_for(mandate_id, mandate_version, goal_run_id),
        serde_json::to_string(&intention_description)?
    );
    let reopened = sqlx::query(
        "UPDATE tasks
         SET status = 'pending', current_attempt_id = NULL, agent_id = NULL,
             result = NULL, error = NULL, blocker = NULL, completed_at = NULL,
             started_at = NULL, description = ?, retry_count = 1,
             updated_at = ?, version = version + 1
         WHERE id = ? AND goal_run_id = ? AND retry_count = 0
           AND NOT EXISTS (
               SELECT 1 FROM tasks t WHERE t.goal_run_id = ? AND t.id != ?
           )",
    )
    .bind(&description)
    .bind(now)
    .bind(root_task_id)
    .bind(goal_run_id)
    .bind(goal_run_id)
    .bind(root_task_id)
    .execute(&mut *connection)
    .await?;
    if reopened.rows_affected() != 1 {
        return Ok(false);
    }
    sqlx::query(
        "UPDATE goal_runs SET status = 'pending', updated_at = ?
         WHERE id = ? AND status IN ('pending', 'running', 'blocked')",
    )
    .bind(now)
    .bind(goal_run_id)
    .execute(&mut *connection)
    .await?;
    tracing::info!(
        mandate_id,
        run_id = goal_run_id,
        "Resuming a committed ACT interrupted before its action task was created"
    );
    Ok(true)
}

pub(super) fn mandate_review_task_description_for(
    mandate_id: &str,
    mandate_version: i64,
    goal_run_id: &str,
) -> String {
    format!(
        "Run one bounded autonomous review for mandate {mandate_id} version {mandate_version} in goal run {goal_run_id}. \
         Use only the isolated built-in mandate protocol and immutable policy supplied \
         by the runtime."
    )
}

pub(super) async fn reconcile_orphaned_mandate_runs(
    connection: &mut sqlx::SqliteConnection,
    now: &str,
) -> anyhow::Result<()> {
    let rows = sqlx::query(
        "SELECT m.id AS mandate_id, m.goal_id, m.version, g.session_id,
                g.dispatch_failures AS review_failures,
                m.min_review_secs, m.max_review_secs,
                gr.id AS goal_run_id, gr.root_task_id,
                (SELECT dc.outcome FROM mandate_decision_cycles dc
                 WHERE dc.mandate_id = m.id AND dc.goal_run_id = gr.id)
                    AS decision_outcome,
                (SELECT COUNT(*) FROM mandate_mutation_attempts ma
                 WHERE ma.mandate_id = m.id AND ma.goal_run_id = gr.id)
                    AS mutation_reservation_count,
                (SELECT COUNT(*) FROM mandate_mutation_attempts ma
                 WHERE ma.mandate_id = m.id AND ma.goal_run_id = gr.id
                   AND ma.dispatch_claimed_at IS NOT NULL)
                    AS dispatch_claim_count,
                (SELECT COUNT(*) FROM tasks t
                 WHERE t.goal_run_id = gr.id AND t.id != gr.root_task_id)
                    AS action_task_count,
                root.retry_count AS root_retry_count
         FROM mandates m
         JOIN goals g ON g.id = m.goal_id
         JOIN goal_runs gr ON gr.goal_id = m.goal_id AND gr.trigger_type = 'mandate'
         JOIN tasks root ON root.id = gr.root_task_id
         WHERE m.status = 'active' AND m.confirmed_at IS NOT NULL
           AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
           AND g.status = 'active'
           AND gr.status IN ('pending', 'running', 'blocked')
           AND (
               root.status IN ('claimed', 'running', 'blocked', 'interrupted', 'failed', 'abandoned')
               OR (
                   root.status = 'completed'
                   AND root.completed_at IS NOT NULL
                   AND julianday(root.completed_at) <= julianday(?) - (60.0 / 86400.0)
               )
           )
           AND NOT EXISTS (
               SELECT 1 FROM task_attempts a
               WHERE a.id = root.current_attempt_id
                 AND a.task_id = root.id AND a.goal_run_id = gr.id
                 AND a.status IN ('claimed', 'running')
                 AND datetime(a.lease_expires_at) > datetime('now')
           )",
    )
    .bind(now)
    .bind(now)
    .fetch_all(&mut *connection)
    .await?;

    for row in rows {
        let mandate_id: String = row.get("mandate_id");
        let goal_id: String = row.get("goal_id");
        let version: i64 = row.get("version");
        let session_id: String = row.get("session_id");
        let review_failures: i32 = row.get("review_failures");
        let min_review_secs: i64 = row.get("min_review_secs");
        let max_review_secs: i64 = row.get("max_review_secs");
        let goal_run_id: String = row.get("goal_run_id");
        let root_task_id: Option<String> = row.get("root_task_id");
        let decision_outcome: Option<String> = row.get("decision_outcome");
        let mutation_reservation_count: i64 = row.get("mutation_reservation_count");
        let dispatch_claim_count: i64 = row.get("dispatch_claim_count");
        let action_task_count: i64 = row.get("action_task_count");
        let root_retry_count: i64 = row.get("root_retry_count");

        // A committed ACT interrupted before its action task existed is
        // resumed once, by reopening the same run's root for a fresh task
        // lead, instead of being dropped. Every dispatch gate (fenced task
        // creation, quota, spacing, authority, duplicate checks) runs again
        // when the resumed lead acts; nothing is authorized here.
        if decision_outcome.as_deref() == Some("act")
            && mutation_reservation_count == 0
            && dispatch_claim_count == 0
            && action_task_count == 0
            && root_retry_count == 0
            && let Some(root_task_id) = root_task_id.as_deref()
            && resume_interrupted_act(
                connection,
                &mandate_id,
                version,
                &goal_run_id,
                root_task_id,
                now,
            )
            .await?
        {
            continue;
        }

        // A mandate mutation can only cross the external I/O boundary after
        // a mutation reservation and a one-use dispatch claim exist. Without
        // either, there is no external effect to reconcile even if an ACT
        // was recorded (e.g. its executor blocked before reserving), and
        // pausing the mandate would turn a recoverable review interruption
        // into a false safety incident. The committed intention is suspended
        // and the next review decides afresh. A recorded ASK or STOP is an
        // owner-facing outcome, so it still goes to reconciliation rather
        // than being silently re-deliberated.
        let retry_reason = match decision_outcome.as_deref() {
            None => Some(MandateFinalizationRejectReason::DecisionMissing),
            Some("act") => Some(MandateFinalizationRejectReason::ActInterruptedBeforeDispatch),
            Some(_) => None,
        };
        if let (Some(reason), 0, 0) = (
            retry_reason,
            mutation_reservation_count,
            dispatch_claim_count,
        ) {
            retry_orphaned_mandate_review_without_dispatch(
                connection,
                reason,
                &mandate_id,
                &goal_id,
                version,
                &session_id,
                &goal_run_id,
                min_review_secs,
                max_review_secs,
                review_failures,
                now,
            )
            .await?;
            continue;
        }

        let mut suspension = MandateSuspension::new(
            MandateSuspensionKind::ExecutionLeaseLost,
            Some("orphaned_mandate_run".to_string()),
        );
        suspension.goal_run_id = Some(goal_run_id.clone());
        let transitioned = sqlx::query(
            "UPDATE mandates
             SET status = 'awaiting_input', review_lease_token = NULL,
                 review_lease_expires_at = NULL, suspension_json = ?,
                 version = version + 1, updated_at = ?
             WHERE id = ? AND goal_id = ? AND status = 'active' AND version = ?",
        )
        .bind(serde_json::to_string(&suspension)?)
        .bind(now)
        .bind(&mandate_id)
        .bind(&goal_id)
        .bind(version)
        .execute(&mut *connection)
        .await?;
        if transitioned.rows_affected() != 1 {
            continue;
        }

        let claimed_unresolved = invalidate_open_mandate_runs(connection, &goal_id, now).await?;
        update_controller_status(connection, &goal_id, MandateStatus::AwaitingInput, now).await?;
        if claimed_unresolved == 0 {
            let counts = mandate_run_proof_counts_on_connection(
                connection,
                &goal_run_id,
                root_task_id.as_deref(),
            )
            .await?;
            let notice = crate::traits::MandateRunNotification::new(
                &mandate_id,
                version + 1,
                &goal_id,
                &goal_run_id,
                &session_id,
                crate::traits::MandateRunNotificationKind::ExecutionLeaseLost,
                counts,
                now,
            );
            super::notifications::enqueue_mandate_run_notification_on_connection(
                &mut *connection,
                &notice,
            )
            .await?;
        }
    }
    Ok(())
}

pub(super) async fn resume_unanswered_agent_questions(
    connection: &mut sqlx::SqliteConnection,
    now: &str,
) -> anyhow::Result<()> {
    let rows = sqlx::query(
        "SELECT m.id, m.goal_id, m.version, m.created_by_session, m.objective, m.name,
                d.id AS decision_id, d.question
         FROM mandates m
         LEFT JOIN mandate_decision_cycles d
           ON d.id = json_extract(m.suspension_json, '$.decision_cycle_id')
         WHERE m.status = 'awaiting_input'
           AND m.confirmed_at IS NOT NULL
           AND json_extract(m.suspension_json, '$.kind') = 'awaiting_answer'
           AND julianday(m.next_review_at) <= julianday(?)
           AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))",
    )
    .bind(now)
    .bind(now)
    .fetch_all(&mut *connection)
    .await?;

    for row in rows {
        let mandate_id: String = row.get("id");
        let goal_id: String = row.get("goal_id");
        let version: i64 = row.get("version");
        let owner_session: String = row.get("created_by_session");
        let decision_id: Option<String> = row.get("decision_id");
        let question: Option<String> = row.get("question");
        let objective: String = row.get("objective");
        let name: Option<String> = row.get("name");
        // Authority is unchanged, so the policy version is kept: a late
        // owner answer still applies to the same policy.
        let resumed = sqlx::query(
            "UPDATE mandates
             SET status = 'active', suspension_json = NULL,
                 review_lease_token = NULL, review_lease_expires_at = NULL, updated_at = ?
             WHERE id = ? AND version = ? AND status = 'awaiting_input'
               AND json_extract(suspension_json, '$.kind') = 'awaiting_answer'",
        )
        .bind(now)
        .bind(&mandate_id)
        .bind(version)
        .execute(&mut *connection)
        .await?;
        if resumed.rows_affected() != 1 {
            continue;
        }
        update_controller_status(connection, &goal_id, MandateStatus::Active, now).await?;
        let label = crate::traits::mandate_named_label(name.as_deref(), Some(&objective));
        let message = match question
            .as_deref()
            .and_then(crate::traits::owner_question_excerpt)
        {
            Some(question) => format!(
                "{label} is continuing on its own because its question went unanswered. Its limits are unchanged; nothing new was allowed.\n\nIt had asked: \"{question}\"\n\nNo reply is needed. If you still want to answer, reply here."
            ),
            None => format!(
                "{label} is continuing on its own because its question went unanswered. Its limits are unchanged; nothing new was allowed."
            ),
        };
        sqlx::query(
            "INSERT OR IGNORE INTO notification_queue
                (id, goal_id, session_id, notification_type, priority, message,
                 created_at, delivered_at, attempts, expires_at, task_id, action_token)
             VALUES (?, ?, ?, 'mandate_ask_timeout', 'critical', ?, ?, NULL, 0, NULL, NULL, NULL)",
        )
        .bind(format!(
            "mandate-ask-timeout:{}",
            decision_id.unwrap_or_else(|| format!("{mandate_id}:{version}"))
        ))
        .bind(&goal_id)
        .bind(&owner_session)
        .bind(message)
        .bind(now)
        .execute(&mut *connection)
        .await?;
    }
    Ok(())
}
