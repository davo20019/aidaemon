//! Review runs: creating them, claiming due mandates and releasing review leases.

use super::*;

pub(super) async fn create_mandate_review_run<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    review_lease_token: &str,
    goal_run_id: &str,
    root_task: &Task,
) -> anyhow::Result<GoalRun> {
    anyhow::ensure!(!mandate_id.trim().is_empty(), "mandate id is required");
    anyhow::ensure!(
        !review_lease_token.trim().is_empty(),
        "review lease token is required"
    );
    anyhow::ensure!(!goal_run_id.trim().is_empty(), "goal run id is required");
    anyhow::ensure!(!root_task.id.trim().is_empty(), "root task id is required");
    anyhow::ensure!(
        !root_task.description.trim().is_empty(),
        "root task description is required"
    );
    anyhow::ensure!(
        root_task.status == "pending"
            && root_task.started_at.is_none()
            && root_task.completed_at.is_none(),
        "a mandate review root task must start pending and unstarted"
    );

    let now = chrono::Utc::now();
    let now_string = now.to_rfc3339();
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
    let row = sqlx::query(
        "SELECT m.goal_id, g.project_id
             FROM mandates m
             JOIN goals g ON g.id = m.goal_id
             WHERE m.id = ?
               AND m.status = 'active'
               AND m.confirmed_at IS NOT NULL
               AND m.review_lease_token = ?
               AND m.review_lease_expires_at IS NOT NULL
               AND julianday(m.review_lease_expires_at) > julianday(?)
               AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
               AND g.domain = 'orchestration'
               AND g.goal_type = 'continuous'
               AND g.status = 'active'",
    )
    .bind(mandate_id)
    .bind(review_lease_token)
    .bind(&now_string)
    .bind(&now_string)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(|| {
        anyhow::anyhow!("mandate is not confirmed, active, or leased by this dispatcher")
    })?;
    let goal_id: String = row.get("goal_id");
    let project_id: String = row.get("project_id");
    anyhow::ensure!(
        root_task.goal_id == goal_id,
        "root task does not belong to the mandate controller"
    );

    let open_run_count = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM goal_runs
             WHERE goal_id = ? AND status IN ('pending', 'running', 'blocked')",
    )
    .bind(&goal_id)
    .fetch_one(&mut *tx)
    .await?;
    anyhow::ensure!(
        open_run_count == 0,
        "mandate controller already has an open goal run"
    );

    let mut run = GoalRun::new(&goal_id, &project_id, "mandate");
    run.id = goal_run_id.to_string();
    run.root_task_id = Some(root_task.id.clone());
    sqlx::query(
        "INSERT INTO goal_runs
                (id, project_id, goal_id, trigger_type, schedule_id, root_task_id,
                 status, outcome_summary, started_at, completed_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&run.id)
    .bind(&run.project_id)
    .bind(&run.goal_id)
    .bind(&run.trigger_type)
    .bind(&run.schedule_id)
    .bind(&run.root_task_id)
    .bind(&run.status)
    .bind(&run.outcome_summary)
    .bind(&run.started_at)
    .bind(&run.completed_at)
    .bind(&run.created_at)
    .bind(&run.updated_at)
    .execute(&mut *tx)
    .await?;

    let result = root_task
        .result
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    let error = root_task
        .error
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    let blocker = root_task
        .blocker
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    let root_insert = sqlx::query(
        "INSERT INTO tasks (
                id, goal_id, goal_run_id, description, status, priority, task_order,
                parallel_group, depends_on, agent_id, context, result, error, blocker,
                idempotent, retry_count, max_retries, created_at, started_at, completed_at,
                updated_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&root_task.id)
    .bind(&root_task.goal_id)
    .bind(&run.id)
    .bind(&root_task.description)
    .bind(&root_task.status)
    .bind(&root_task.priority)
    .bind(root_task.task_order)
    .bind(&root_task.parallel_group)
    .bind(&root_task.depends_on)
    .bind(&root_task.agent_id)
    .bind(&root_task.context)
    .bind(result)
    .bind(error)
    .bind(blocker)
    .bind(root_task.idempotent as i32)
    .bind(root_task.retry_count)
    .bind(root_task.max_retries)
    .bind(&root_task.created_at)
    .bind(&root_task.started_at)
    .bind(&root_task.completed_at)
    .bind(&root_task.created_at)
    .execute(&mut *tx)
    .await;
    if let Err(insert_error) = root_insert {
        // A failed statement leaves SQLite's write transaction active until
        // rollback completes. Await it here so an immediate recovery query
        // cannot race the drop-triggered asynchronous rollback and observe
        // SQLITE_BUSY under load.
        if let Err(rollback_error) = tx.rollback().await {
            return Err(anyhow::anyhow!(
                "mandate review root insert failed ({insert_error}); rollback failed ({rollback_error})"
            ));
        }
        return Err(insert_error.into());
    }
    let root_dependencies = root_task.dependency_ids().map_err(anyhow::Error::msg)?;
    super::goals::replace_task_dependencies(&mut tx, &root_task.id, &root_dependencies).await?;

    tx.commit().await?;
    Ok(run)
}

pub(super) async fn claim_due_mandates<S: WorkDatabase>(
    store: &S,
    limit: i64,
    lease_owner: &str,
    lease_secs: i64,
) -> anyhow::Result<Vec<Mandate>> {
    if limit <= 0 {
        return Ok(Vec::new());
    }
    anyhow::ensure!(!lease_owner.trim().is_empty(), "lease owner is required");
    anyhow::ensure!(
        (1..=86_400).contains(&lease_secs),
        "review lease duration must be between 1 second and 24 hours"
    );
    let now = chrono::Utc::now();
    let now_string = now.to_rfc3339();
    let lease_expires_at = (now + chrono::Duration::seconds(lease_secs)).to_rfc3339();
    let lease_token = format!("{}:{}", lease_owner.trim(), uuid::Uuid::new_v4());
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;

    // Recover the only crash state produced by the pre-atomic dispatcher:
    // an open mandate run whose root task was never inserted. No work can
    // have executed in such a run, so close it and make the mandate due in
    // the same transaction before selecting claims.
    sqlx::query(
        "UPDATE mandates
             SET next_review_at = ?, review_lease_token = NULL,
                 review_lease_expires_at = NULL, updated_at = ?
             WHERE status = 'active' AND confirmed_at IS NOT NULL
               AND EXISTS (
                   SELECT 1 FROM goal_runs gr
                   WHERE gr.goal_id = mandates.goal_id
                     AND gr.trigger_type = 'mandate'
                     AND gr.status IN ('pending', 'running', 'blocked')
                     AND NOT EXISTS (
                         SELECT 1 FROM tasks t WHERE t.goal_run_id = gr.id
                     )
               )",
    )
    .bind(&now_string)
    .bind(&now_string)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
            "UPDATE goal_runs
             SET status = 'failed',
                 outcome_summary = 'Recovered an interrupted mandate dispatch before its root task was created.',
                 completed_at = COALESCE(completed_at, ?), updated_at = ?
             WHERE trigger_type = 'mandate'
               AND status IN ('pending', 'running', 'blocked')
               AND NOT EXISTS (
                   SELECT 1 FROM tasks t WHERE t.goal_run_id = goal_runs.id
               )",
        )
        .bind(&now_string)
        .bind(&now_string)
        .execute(&mut *tx)
        .await?;

    // A root whose execution lease vanished may have performed an
    // externally visible effect before crashing. Never auto-retry that
    // ambiguous cycle: atomically pause it for owner reconciliation.
    reconcile_orphaned_mandate_runs(&mut tx, &now_string).await?;

    // An agent question is a request for owner judgment, not a safety
    // fence. Once its bounded reconsider time passes without an answer,
    // the mandate continues within its unchanged authority instead of
    // parking forever; the question stays visible in decision history.
    resume_unanswered_agent_questions(&mut tx, &now_string).await?;

    // Expiry is a hard authority boundary, not merely a query filter. Keep
    // the visible lifecycle coherent so an expired mandate cannot remain
    // labelled active indefinitely with an active backing controller. It
    // revokes attempts/runs/intentions in the same transaction, exactly as
    // an explicit lifecycle transition does.
    let expired_goal_ids = sqlx::query_scalar::<_, String>(
        "SELECT goal_id FROM mandates
             WHERE status = 'active' AND expires_at IS NOT NULL
               AND julianday(expires_at) <= julianday(?)",
    )
    .bind(&now_string)
    .fetch_all(&mut *tx)
    .await?;
    for goal_id in &expired_goal_ids {
        invalidate_open_mandate_runs(&mut tx, goal_id, &now_string).await?;
    }
    sqlx::query(
        "UPDATE goals
             SET status = 'completed', completed_at = COALESCE(completed_at, ?), updated_at = ?
             WHERE id IN (
                 SELECT goal_id FROM mandates
                 WHERE status = 'active' AND expires_at IS NOT NULL
                   AND julianday(expires_at) <= julianday(?)
             )",
    )
    .bind(&now_string)
    .bind(&now_string)
    .bind(&now_string)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "UPDATE mandates
             SET status = 'completed', review_lease_token = NULL,
                 review_lease_expires_at = NULL, version = version + 1, updated_at = ?
             WHERE status = 'active' AND expires_at IS NOT NULL
               AND julianday(expires_at) <= julianday(?)",
    )
    .bind(&now_string)
    .bind(&now_string)
    .execute(&mut *tx)
    .await?;
    let query = format!(
        "UPDATE mandates
             SET review_lease_token = ?, review_lease_expires_at = ?, updated_at = ?
             WHERE id IN (
                 SELECT m.id
                 FROM mandates m
                 JOIN goals g ON g.id = m.goal_id
                 WHERE m.status = 'active'
                   AND m.confirmed_at IS NOT NULL
                   AND g.domain = 'orchestration'
                   AND g.goal_type = 'continuous'
                   AND g.status = 'active'
                   AND julianday(m.next_review_at) <= julianday(?)
                   AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
                   AND (
                       m.review_lease_token IS NULL
                       OR m.review_lease_expires_at IS NULL
                       OR julianday(m.review_lease_expires_at) <= julianday(?)
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM goal_runs gr
                       WHERE gr.goal_id = m.goal_id
                         AND gr.status IN ('pending', 'running', 'blocked')
                   )
                 ORDER BY julianday(m.next_review_at), m.id
                 LIMIT ?
             )
               AND status = 'active'
               AND confirmed_at IS NOT NULL
               AND (
                   review_lease_token IS NULL
                   OR review_lease_expires_at IS NULL
                   OR julianday(review_lease_expires_at) <= julianday(?)
               )
             RETURNING {MANDATE_COLUMNS}"
    );
    let rows = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(&lease_token)
        .bind(&lease_expires_at)
        .bind(&now_string)
        .bind(&now_string)
        .bind(&now_string)
        .bind(&now_string)
        .bind(limit.min(100))
        .bind(&now_string)
        .fetch_all(&mut *tx)
        .await?;
    tx.commit().await?;
    let mut mandates = rows
        .iter()
        .map(mandate_from_row)
        .collect::<anyhow::Result<Vec<_>>>()?;
    mandates.sort_by(|left, right| {
        left.next_review_at
            .cmp(&right.next_review_at)
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(mandates)
}

pub(super) async fn release_mandate_review_lease<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    lease_token: &str,
    retry_at: &str,
) -> anyhow::Result<bool> {
    anyhow::ensure!(
        !lease_token.trim().is_empty(),
        "review lease token is required"
    );
    validate_timestamp("retry_at", retry_at)?;
    let result = sqlx::query(
        "UPDATE mandates
             SET next_review_at = ?, review_lease_token = NULL,
                 review_lease_expires_at = NULL, updated_at = ?
             WHERE id = ? AND review_lease_token = ?",
    )
    .bind(retry_at)
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(mandate_id)
    .bind(lease_token)
    .execute(store.work_pool())
    .await?;
    Ok(result.rows_affected() == 1)
}
