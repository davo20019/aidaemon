//! Action tasks created and claimed from a reserved mutation attempt.

use super::*;

pub(super) async fn create_mandate_task_from_attempt<S: WorkDatabase>(
    store: &S,
    task: &Task,
    mandate_id: &str,
    mandate_version: i64,
    goal_run_id: &str,
    root_task_attempt_id: &str,
    max_non_root_tasks: i64,
) -> anyhow::Result<bool> {
    anyhow::ensure!(mandate_version > 0, "mandate version must be positive");
    anyhow::ensure!(
        (1..=64).contains(&max_non_root_tasks),
        "mandate task cap must be between 1 and 64"
    );
    anyhow::ensure!(task.status == "pending", "mandate tasks must start pending");
    anyhow::ensure!(
        !task.idempotent && task.retry_count == 0 && task.max_retries == 0,
        "mandate tasks cannot be replayable"
    );
    anyhow::ensure!(
        !task.id.trim().is_empty()
            && !task.goal_id.trim().is_empty()
            && !task.description.trim().is_empty()
            && task.description.trim() == task.description
            && task.description.chars().count() <= 8_000,
        "mandate task identity and description must be canonical"
    );
    anyhow::ensure!(
        task.agent_id.is_none()
            && task.result.is_none()
            && task.error.is_none()
            && task.blocker.is_none()
            && task.started_at.is_none()
            && task.completed_at.is_none(),
        "new mandate tasks cannot carry execution state"
    );

    let now = chrono::Utc::now().to_rfc3339();
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
    // First write: acquire the writer lock and prove the complete authority
    // chain in one CAS. A pause, policy revision, run replacement, or root
    // attempt replacement either linearizes before this statement (and
    // rejects it) or after this transaction commits.
    let fenced = sqlx::query(
        "UPDATE task_attempts
             SET last_heartbeat_at = datetime('now')
             WHERE id = ? AND goal_run_id = ?
               AND status IN ('claimed', 'running')
               AND datetime(lease_expires_at) > datetime('now')
               AND EXISTS (
                   SELECT 1
                   FROM tasks root
                   JOIN goal_runs gr ON gr.id = task_attempts.goal_run_id
                   JOIN mandates m ON m.goal_id = gr.goal_id
                   JOIN goals g ON g.id = m.goal_id
                   JOIN mandate_decision_cycles dc ON dc.goal_run_id = gr.id
                   JOIN intentions i ON i.decision_cycle_id = dc.id
                   WHERE root.id = task_attempts.task_id
                     AND root.current_attempt_id = task_attempts.id
                     AND root.status IN ('claimed', 'running')
                     AND root.error IS NULL AND root.blocker IS NULL
                     AND gr.id = ? AND gr.root_task_id = root.id
                     AND gr.trigger_type = 'mandate' AND gr.status = 'running'
                     AND m.id = ? AND m.goal_id = ? AND m.version = ?
                     AND m.status = 'active' AND m.confirmed_at IS NOT NULL
                     AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
                     AND g.status = 'active' AND g.domain = 'orchestration'
                     AND g.goal_type = 'continuous'
                     AND dc.mandate_id = m.id AND dc.mandate_version = m.version
                     AND dc.outcome = 'act'
                     AND i.mandate_id = m.id AND i.goal_run_id = gr.id
                     AND i.status = 'committed'
               )",
    )
    .bind(root_task_attempt_id)
    .bind(goal_run_id)
    .bind(goal_run_id)
    .bind(mandate_id)
    .bind(&task.goal_id)
    .bind(mandate_version)
    .bind(&now)
    .execute(&mut *tx)
    .await?;
    if fenced.rows_affected() != 1 {
        tx.rollback().await?;
        return Ok(false);
    }

    let task_count = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM tasks
             WHERE goal_run_id = ?
               AND id != (SELECT root_task_id FROM goal_runs WHERE id = ?)",
    )
    .bind(goal_run_id)
    .bind(goal_run_id)
    .fetch_one(&mut *tx)
    .await?;
    if task_count >= max_non_root_tasks {
        tx.rollback().await?;
        return Ok(false);
    }

    if let Some(raw_dependencies) = task.depends_on.as_deref() {
        let dependencies = serde_json::from_str::<Vec<String>>(raw_dependencies)?;
        let unique = dependencies
            .iter()
            .collect::<std::collections::HashSet<_>>();
        anyhow::ensure!(
            unique.len() == dependencies.len(),
            "mandate task dependencies must be unique"
        );
        for dependency in dependencies {
            let in_run = sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM tasks WHERE id = ? AND goal_run_id = ?",
            )
            .bind(&dependency)
            .bind(goal_run_id)
            .fetch_one(&mut *tx)
            .await?;
            anyhow::ensure!(in_run == 1, "mandate task dependency is outside this run");
        }
    }

    let inserted = sqlx::query(
        "INSERT INTO tasks (
                id, goal_id, goal_run_id, description, status, priority, task_order,
                parallel_group, depends_on, agent_id, context, result, error, blocker,
                idempotent, retry_count, max_retries, created_at, started_at, completed_at,
                updated_at
             ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL,
                       0, 0, 0, ?, NULL, NULL, ?)",
    )
    .bind(&task.id)
    .bind(&task.goal_id)
    .bind(goal_run_id)
    .bind(&task.description)
    .bind(&task.priority)
    .bind(task.task_order)
    .bind(&task.parallel_group)
    .bind(&task.depends_on)
    .bind(&task.created_at)
    .bind(&task.created_at)
    .execute(&mut *tx)
    .await?;
    anyhow::ensure!(
        inserted.rows_affected() == 1,
        "mandate task was not inserted"
    );
    let dependencies = task.dependency_ids().map_err(anyhow::Error::msg)?;
    super::goals::replace_task_dependencies(&mut tx, &task.id, &dependencies).await?;
    tx.commit().await?;
    Ok(true)
}

// The trait method's parameters, passed through.
#[allow(clippy::too_many_arguments)]
pub(super) async fn claim_mandate_task_from_attempt<S: WorkDatabase>(
    store: &S,
    task_id: &str,
    worker_instance_id: &str,
    mandate_id: &str,
    mandate_version: i64,
    goal_run_id: &str,
    root_task_attempt_id: &str,
    lease_secs: i64,
) -> anyhow::Result<Option<crate::traits::TaskAttempt>> {
    anyhow::ensure!(mandate_version > 0, "mandate version must be positive");
    anyhow::ensure!(
        !task_id.trim().is_empty()
            && !worker_instance_id.trim().is_empty()
            && !root_task_attempt_id.trim().is_empty(),
        "mandate claim identity is required"
    );
    let lease_secs = lease_secs.clamp(1, 86_400);
    let now = chrono::Utc::now();
    let now_string = now.to_rfc3339();
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
    let fenced = sqlx::query(
        "UPDATE task_attempts
             SET last_heartbeat_at = datetime('now')
             WHERE id = ? AND goal_run_id = ?
               AND status IN ('claimed', 'running')
               AND datetime(lease_expires_at) > datetime('now')
               AND EXISTS (
                   SELECT 1
                   FROM tasks root
                   JOIN goal_runs gr ON gr.id = task_attempts.goal_run_id
                   JOIN mandates m ON m.goal_id = gr.goal_id
                   JOIN goals g ON g.id = m.goal_id
                   JOIN mandate_decision_cycles dc ON dc.goal_run_id = gr.id
                   JOIN intentions i ON i.decision_cycle_id = dc.id
                   WHERE root.id = task_attempts.task_id
                     AND root.current_attempt_id = task_attempts.id
                     AND root.status IN ('claimed', 'running')
                     AND root.error IS NULL AND root.blocker IS NULL
                     AND gr.id = ? AND gr.root_task_id = root.id
                     AND gr.trigger_type = 'mandate' AND gr.status = 'running'
                     AND m.id = ?
                     AND m.goal_id = (SELECT goal_id FROM tasks WHERE id = ?)
                     AND m.version = ?
                     AND m.status = 'active' AND m.confirmed_at IS NOT NULL
                     AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
                     AND g.status = 'active' AND g.domain = 'orchestration'
                     AND g.goal_type = 'continuous'
                     AND dc.mandate_id = m.id AND dc.mandate_version = m.version
                     AND dc.outcome = 'act'
                     AND i.mandate_id = m.id AND i.goal_run_id = gr.id
                     AND i.status = 'committed'
               )",
    )
    .bind(root_task_attempt_id)
    .bind(goal_run_id)
    .bind(goal_run_id)
    .bind(mandate_id)
    .bind(task_id)
    .bind(mandate_version)
    .bind(&now_string)
    .execute(&mut *tx)
    .await?;
    if fenced.rows_affected() != 1 {
        tx.rollback().await?;
        return Ok(None);
    }

    let claimable = sqlx::query_scalar::<_, String>(
        "SELECT t.goal_id
             FROM tasks t JOIN goal_runs gr ON gr.id = t.goal_run_id
             WHERE t.id = ? AND t.goal_run_id = ? AND t.id != gr.root_task_id
               AND t.status = 'pending' AND t.current_attempt_id IS NULL
               AND t.worker_profile_id IS NULL
               AND COALESCE(t.workspace_policy_explicit, 0) = 0
               AND NOT EXISTS (SELECT 1 FROM task_workspaces w WHERE w.task_id = t.id)",
    )
    .bind(task_id)
    .bind(goal_run_id)
    .fetch_optional(&mut *tx)
    .await?;
    if claimable.is_none() {
        tx.rollback().await?;
        return Ok(None);
    }
    let unmet = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*)
             FROM json_each(COALESCE((SELECT depends_on FROM tasks WHERE id = ?), '[]')) dep
             WHERE NOT EXISTS (
                SELECT 1 FROM tasks d
                WHERE d.id = CAST(dep.value AS TEXT)
                  AND d.goal_run_id = ?
                  AND d.status IN ('completed', 'skipped', 'superseded')
             )",
    )
    .bind(task_id)
    .bind(goal_run_id)
    .fetch_one(&mut *tx)
    .await?;
    if unmet > 0 {
        tx.rollback().await?;
        return Ok(None);
    }

    let attempt = crate::traits::TaskAttempt {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task_id.to_string(),
        goal_run_id: goal_run_id.to_string(),
        worker_profile_id: None,
        worker_instance_id: worker_instance_id.to_string(),
        lease_token: uuid::Uuid::new_v4().to_string(),
        status: "claimed".to_string(),
        lease_expires_at: (now + chrono::Duration::seconds(lease_secs)).to_rfc3339(),
        last_heartbeat_at: now_string.clone(),
        workspace_id: None,
        started_at: now_string,
        completed_at: None,
    };
    sqlx::query(
        "INSERT INTO task_attempts
                (id, task_id, goal_run_id, worker_profile_id, worker_instance_id,
                 lease_token, status, lease_expires_at, last_heartbeat_at,
                 workspace_id, started_at, completed_at)
             VALUES (?, ?, ?, NULL, ?, ?, 'claimed', ?, ?, NULL, ?, NULL)",
    )
    .bind(&attempt.id)
    .bind(&attempt.task_id)
    .bind(&attempt.goal_run_id)
    .bind(&attempt.worker_instance_id)
    .bind(&attempt.lease_token)
    .bind(&attempt.lease_expires_at)
    .bind(&attempt.last_heartbeat_at)
    .bind(&attempt.started_at)
    .execute(&mut *tx)
    .await?;
    let updated = sqlx::query(
        "UPDATE tasks
             SET status = 'claimed', agent_id = ?, current_attempt_id = ?,
                 worker_profile_id = NULL, started_at = COALESCE(started_at, datetime('now')),
                 result = NULL, error = NULL, blocker = NULL, completed_at = NULL,
                 updated_at = datetime('now'), version = version + 1
             WHERE id = ? AND goal_run_id = ? AND status = 'pending'
               AND current_attempt_id IS NULL",
    )
    .bind(worker_instance_id)
    .bind(&attempt.id)
    .bind(task_id)
    .bind(goal_run_id)
    .execute(&mut *tx)
    .await?;
    if updated.rows_affected() != 1 {
        tx.rollback().await?;
        return Ok(None);
    }
    tx.commit().await?;
    Ok(Some(attempt))
}
