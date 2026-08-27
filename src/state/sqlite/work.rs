use super::*;
use crate::traits::{
    GoalRun, HandoffArtifact, TaskAttempt, TaskAttemptPatch, TaskHandoff, TaskJournalEntry,
    TaskWorkspace, WorkCoordinationStore, WorkGoalSummary, WorkProject, WorkTaskSummary,
    WorkerProfile, DEFAULT_PROJECT_ID,
};

const DEFAULT_LEASE_SECS: i64 = 180;

fn normalized_lease_secs(lease_secs: i64) -> i64 {
    if lease_secs <= 0 {
        DEFAULT_LEASE_SECS
    } else {
        lease_secs.clamp(30, 3600)
    }
}

fn goal_run_from_row(row: &sqlx::sqlite::SqliteRow) -> GoalRun {
    GoalRun {
        id: row.get("id"),
        project_id: row.get("project_id"),
        goal_id: row.get("goal_id"),
        trigger_type: row.get("trigger_type"),
        schedule_id: row.get("schedule_id"),
        root_task_id: row.get("root_task_id"),
        status: row.get("status"),
        outcome_summary: row.get("outcome_summary"),
        started_at: row.get("started_at"),
        completed_at: row.get("completed_at"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

fn task_attempt_from_row(row: &sqlx::sqlite::SqliteRow) -> TaskAttempt {
    TaskAttempt {
        id: row.get("id"),
        task_id: row.get("task_id"),
        goal_run_id: row.get("goal_run_id"),
        worker_profile_id: row.get("worker_profile_id"),
        worker_instance_id: row.get("worker_instance_id"),
        lease_token: row.get("lease_token"),
        status: row.get("status"),
        lease_expires_at: row.get("lease_expires_at"),
        last_heartbeat_at: row.get("last_heartbeat_at"),
        workspace_id: row.get("workspace_id"),
        started_at: row.get("started_at"),
        completed_at: row.get("completed_at"),
    }
}

fn worker_profile_from_row(row: &sqlx::sqlite::SqliteRow) -> WorkerProfile {
    WorkerProfile {
        id: row.get("id"),
        project_id: row.get("project_id"),
        name: row.get("name"),
        specialist: row.get("specialist"),
        model: row.get("model"),
        tools_json: row.get("tools_json"),
        max_iterations: row.get("max_iterations"),
        tool_budget: row.get("tool_budget"),
        timeout_secs: row.get("timeout_secs"),
        max_concurrency: row.get("max_concurrency"),
        workspace_policy: row.get("workspace_policy"),
        memory_scope: row.get("memory_scope"),
        version: row.get("version"),
        enabled: row.get::<i64, _>("enabled") != 0,
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

fn task_from_row(row: &sqlx::sqlite::SqliteRow) -> Task {
    Task {
        id: row.get("id"),
        goal_id: row.get("goal_id"),
        description: row.get("description"),
        status: row.get("status"),
        priority: row.get("priority"),
        task_order: row.get("task_order"),
        parallel_group: row.get("parallel_group"),
        depends_on: row.get("depends_on"),
        agent_id: row.get("agent_id"),
        context: row.get("context"),
        result: row.get("result"),
        error: row.get("error"),
        blocker: row.get("blocker"),
        idempotent: row.get::<i64, _>("idempotent") != 0,
        retry_count: row.get("retry_count"),
        max_retries: row.get("max_retries"),
        created_at: row.get("created_at"),
        started_at: row.get("started_at"),
        completed_at: row.get("completed_at"),
    }
}

async fn insert_journal(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    entry: &TaskJournalEntry,
) -> anyhow::Result<()> {
    let payload = entry
        .payload
        .as_deref()
        .map(crate::tools::sanitize::redact_secrets);
    sqlx::query(
        "INSERT INTO task_journal
            (id, project_id, goal_id, goal_run_id, task_id, attempt_id,
             entry_type, actor_type, actor_id, source_channel, body, payload, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&entry.id)
    .bind(&entry.project_id)
    .bind(&entry.goal_id)
    .bind(&entry.goal_run_id)
    .bind(&entry.task_id)
    .bind(&entry.attempt_id)
    .bind(&entry.entry_type)
    .bind(&entry.actor_type)
    .bind(&entry.actor_id)
    .bind(&entry.source_channel)
    .bind(crate::tools::sanitize::redact_secrets(&entry.body))
    .bind(payload)
    .bind(&entry.created_at)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn task_scope(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    task_id: &str,
) -> anyhow::Result<Option<(String, String, String)>> {
    let row = sqlx::query(
        "SELECT g.project_id, t.goal_id, t.goal_run_id
         FROM tasks t
         JOIN goals g ON g.id = t.goal_id
         WHERE t.id = ?",
    )
    .bind(task_id)
    .fetch_optional(&mut **tx)
    .await?;
    Ok(row.map(|row| {
        (
            row.get("project_id"),
            row.get("goal_id"),
            row.get("goal_run_id"),
        )
    }))
}

async fn close_pending_task_escalations(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    task_id: &str,
) -> anyhow::Result<()> {
    sqlx::query(
        "UPDATE notification_queue
         SET delivered_at = datetime('now')
         WHERE task_id = ? AND notification_type = 'escalation'
           AND delivered_at IS NULL",
    )
    .bind(task_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

#[async_trait]
impl WorkCoordinationStore for SqliteStateStore {
    async fn create_work_project(
        &self,
        name: &str,
        description: Option<&str>,
    ) -> anyhow::Result<WorkProject> {
        let name = name.trim();
        anyhow::ensure!(!name.is_empty(), "project name cannot be empty");
        anyhow::ensure!(name.chars().count() <= 80, "project name is too long");
        let now = chrono::Utc::now().to_rfc3339();
        let project = WorkProject {
            id: format!("project-{}", uuid::Uuid::new_v4()),
            name: name.to_string(),
            description: description
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_string),
            created_at: now.clone(),
            updated_at: now,
        };
        sqlx::query(
            "INSERT INTO work_projects (id, name, description, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&project.id)
        .bind(&project.name)
        .bind(&project.description)
        .bind(&project.created_at)
        .bind(&project.updated_at)
        .execute(&self.pool)
        .await?;
        Ok(project)
    }

    async fn list_work_projects(&self) -> anyhow::Result<Vec<WorkProject>> {
        let rows = sqlx::query(
            "SELECT id, name, description, created_at, updated_at
             FROM work_projects ORDER BY name COLLATE NOCASE",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|row| WorkProject {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            })
            .collect())
    }

    async fn get_session_work_project(&self, session_id: &str) -> anyhow::Result<String> {
        let project = sqlx::query_scalar::<_, String>(
            "SELECT project_id FROM session_work_projects WHERE session_id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(project.unwrap_or_else(|| DEFAULT_PROJECT_ID.to_string()))
    }

    async fn set_session_work_project(
        &self,
        session_id: &str,
        project_id: &str,
    ) -> anyhow::Result<bool> {
        let exists =
            sqlx::query_scalar::<_, i64>("SELECT 1 FROM work_projects WHERE id = ? LIMIT 1")
                .bind(project_id)
                .fetch_optional(&self.pool)
                .await?
                .is_some();
        if !exists {
            return Ok(false);
        }
        sqlx::query(
            "INSERT INTO session_work_projects (session_id, project_id, updated_at)
             VALUES (?, ?, datetime('now'))
             ON CONFLICT(session_id) DO UPDATE
             SET project_id = excluded.project_id, updated_at = excluded.updated_at",
        )
        .bind(session_id)
        .bind(project_id)
        .execute(&self.pool)
        .await?;
        Ok(true)
    }

    async fn start_goal_run(
        &self,
        goal_id: &str,
        trigger_type: &str,
        schedule_id: Option<&str>,
        root_task_id: Option<&str>,
    ) -> anyhow::Result<GoalRun> {
        anyhow::ensure!(
            matches!(
                trigger_type,
                "finite" | "scheduled" | "manual" | "mandate" | "recovery" | "legacy"
            ),
            "unsupported goal run trigger type"
        );
        let mut tx = self.pool.begin().await?;
        let project_id = sqlx::query_scalar::<_, String>(
            "SELECT project_id FROM goals
             WHERE id = ? AND domain = 'orchestration'",
        )
        .bind(goal_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or_else(|| anyhow::anyhow!("goal not found or not dispatchable"))?;

        if let Some(row) = sqlx::query(
            "SELECT id, project_id, goal_id, trigger_type, schedule_id, root_task_id,
                    status, outcome_summary, started_at, completed_at, created_at, updated_at
             FROM goal_runs
             WHERE goal_id = ? AND status IN ('pending', 'running', 'blocked')
             ORDER BY julianday(started_at) DESC, id DESC LIMIT 1",
        )
        .bind(goal_id)
        .fetch_optional(&mut *tx)
        .await?
        {
            let mut existing = goal_run_from_row(&row);
            let task_count =
                sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM tasks WHERE goal_run_id = ?")
                    .bind(&existing.id)
                    .fetch_one(&mut *tx)
                    .await?;
            if task_count == 0 {
                sqlx::query(
                    "UPDATE goal_runs
                     SET trigger_type = ?, schedule_id = ?, root_task_id = COALESCE(?, root_task_id),
                         status = 'pending', updated_at = datetime('now')
                     WHERE id = ?",
                )
                .bind(trigger_type)
                .bind(schedule_id)
                .bind(root_task_id)
                .bind(&existing.id)
                .execute(&mut *tx)
                .await?;
                existing.trigger_type = trigger_type.to_string();
                existing.schedule_id = schedule_id.map(str::to_string);
                if root_task_id.is_some() {
                    existing.root_task_id = root_task_id.map(str::to_string);
                }
                existing.status = "pending".to_string();
                existing.updated_at = chrono::Utc::now().to_rfc3339();
                tx.commit().await?;
                return Ok(existing);
            }
            if trigger_type != "scheduled" {
                anyhow::bail!(
                    "goal already has an open run {}; finish or cancel it before starting another",
                    existing.id
                );
            }
        }

        let mut run = GoalRun::new(goal_id, &project_id, trigger_type);
        run.schedule_id = schedule_id.map(str::to_string);
        run.root_task_id = root_task_id.map(str::to_string);
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
        tx.commit().await?;
        Ok(run)
    }

    async fn get_current_goal_run(&self, goal_id: &str) -> anyhow::Result<Option<GoalRun>> {
        let row = sqlx::query(
            "SELECT id, project_id, goal_id, trigger_type, schedule_id, root_task_id,
                    status, outcome_summary, started_at, completed_at, created_at, updated_at
             FROM goal_runs
             WHERE goal_id = ? AND status IN ('pending', 'running', 'blocked')
             ORDER BY julianday(started_at) DESC, id DESC LIMIT 1",
        )
        .bind(goal_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.as_ref().map(goal_run_from_row))
    }

    async fn get_goal_runs(&self, goal_id: &str) -> anyhow::Result<Vec<GoalRun>> {
        let rows = sqlx::query(
            "SELECT id, project_id, goal_id, trigger_type, schedule_id, root_task_id,
                    status, outcome_summary, started_at, completed_at, created_at, updated_at
             FROM goal_runs WHERE goal_id = ?
             ORDER BY julianday(started_at) DESC, id DESC",
        )
        .bind(goal_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(goal_run_from_row).collect())
    }

    async fn get_goal_run_for_task(&self, task_id: &str) -> anyhow::Result<Option<GoalRun>> {
        let row = sqlx::query(
            "SELECT gr.id, gr.project_id, gr.goal_id, gr.trigger_type, gr.schedule_id,
                    gr.root_task_id, gr.status, gr.outcome_summary, gr.started_at,
                    gr.completed_at, gr.created_at, gr.updated_at
             FROM tasks AS t
             JOIN goal_runs AS gr ON gr.id = t.goal_run_id
             WHERE t.id = ?",
        )
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.as_ref().map(goal_run_from_row))
    }

    async fn get_tasks_for_goal_run(&self, run_id: &str) -> anyhow::Result<Vec<Task>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
                    parallel_group, depends_on, agent_id, context, result, error, blocker,
                    idempotent, retry_count, max_retries, created_at, started_at, completed_at
             FROM tasks WHERE goal_run_id = ?
             ORDER BY task_order ASC, julianday(created_at) ASC, id ASC",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(task_from_row).collect())
    }

    async fn finish_goal_run(
        &self,
        run_id: &str,
        status: &str,
        summary: Option<&str>,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            matches!(status, "completed" | "failed" | "blocked" | "cancelled"),
            "unsupported terminal run status"
        );
        let completed_at = if status == "blocked" {
            None
        } else {
            Some(chrono::Utc::now().to_rfc3339())
        };
        let now = chrono::Utc::now().to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let run = sqlx::query(
            "SELECT gr.goal_id, gr.status AS prior_status, gr.trigger_type, gr.root_task_id,
                    g.goal_type, g.session_id,
                    EXISTS(SELECT 1 FROM goal_schedules s WHERE s.goal_id = gr.goal_id) AS has_schedule,
                    json_extract(t.context, '$.recovery_for_run') AS recovery_for_run
             FROM goal_runs gr
             JOIN goals g ON g.id = gr.goal_id
             LEFT JOIN tasks t ON t.id = gr.root_task_id
             WHERE gr.id = ?",
        )
        .bind(run_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(run) = run else {
            tx.rollback().await?;
            return Ok(false);
        };
        let goal_id: String = run.get("goal_id");
        let prior_status: String = run.get("prior_status");
        let trigger_type: String = run.get("trigger_type");
        let goal_type: String = run.get("goal_type");
        let session_id: String = run.get("session_id");
        let has_schedule = run.get::<i64, _>("has_schedule") != 0;
        let recovery_for_run: Option<String> = run.get("recovery_for_run");
        if trigger_type == "scheduled" && status == "completed" {
            let incomplete_task_count = sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM tasks
                 WHERE goal_run_id = ?
                   AND (status NOT IN ('completed', 'skipped', 'superseded')
                        OR length(trim(COALESCE(error, ''))) > 0
                        OR length(trim(COALESCE(blocker, ''))) > 0)",
            )
            .bind(run_id)
            .fetch_one(&mut *tx)
            .await?;
            anyhow::ensure!(
                incomplete_task_count == 0,
                "scheduled run cannot complete while {incomplete_task_count} task obligation(s) remain unresolved"
            );
        }
        let result = sqlx::query(
            "UPDATE goal_runs
             SET status = ?, outcome_summary = ?, completed_at = ?,
                 updated_at = datetime('now')
             WHERE id = ? AND status IN ('pending', 'running', 'blocked')",
        )
        .bind(status)
        .bind(summary)
        .bind(completed_at)
        .bind(run_id)
        .execute(&mut *tx)
        .await?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        if trigger_type == "scheduled" && matches!(status, "failed" | "cancelled") {
            sqlx::query(
                "UPDATE tasks
                 SET status = 'cancelled',
                     error = COALESCE(NULLIF(error, ''),
                                      'Parent scheduled occurrence reached a terminal state.'),
                     completed_at = COALESCE(completed_at, ?)
                 WHERE goal_run_id = ?
                   AND status IN ('pending', 'claimed', 'running')",
            )
            .bind(&now)
            .bind(run_id)
            .execute(&mut *tx)
            .await?;
        }

        let scheduled_objective = goal_type == "continuous" && has_schedule;
        if scheduled_objective {
            let failure_budget = sqlx::query_scalar::<_, i64>(
                "SELECT COALESCE(
                    (SELECT json_extract(objective_control_json, '$.run_failure_budget')
                     FROM mandates WHERE goal_id = ?),
                    3
                 )",
            )
            .bind(&goal_id)
            .fetch_one(&mut *tx)
            .await?
            .clamp(1, 10);
            let recovery_proof_receipt_ids = if recovery_for_run.is_some() {
                sqlx::query_scalar::<_, String>(
                    "SELECT json_extract(e.data, '$.tool_call_id') FROM events e
                     JOIN tasks t ON t.id = json_extract(e.data, '$.task_id')
                     WHERE t.goal_run_id = ? AND e.event_type = 'tool_result'
                       AND json_extract(t.context, '$.terminal_recovery') = 1
                       AND json_extract(t.context, '$.recovery_for_run') = ?
                       AND json_extract(e.data, '$.receipt.outcome_status') = 'succeeded'
                       AND json_extract(e.data, '$.receipt.outcome_evidence')
                           IN ('tool_reported', 'structured_metadata', 'durable_replay')",
                )
                .bind(run_id)
                .bind(recovery_for_run.as_deref())
                .fetch_all(&mut *tx)
                .await?
            } else {
                Vec::new()
            };
            let recovery_completion_verified =
                recovery_for_run.is_none() || !recovery_proof_receipt_ids.is_empty();

            if status == "completed" && recovery_completion_verified {
                sqlx::query(
                    "INSERT INTO scheduled_recovery_state
                        (goal_id, consecutive_failures, failure_budget, disposition,
                         latest_failure_kind, last_failed_run_id, last_recovery_run_id, updated_at)
                     VALUES (?, 0, ?, 'healthy', NULL, NULL, ?, ?)
                     ON CONFLICT(goal_id) DO UPDATE SET
                        consecutive_failures = 0,
                        failure_budget = excluded.failure_budget,
                        disposition = 'healthy',
                        latest_failure_kind = NULL,
                        last_recovery_run_id = COALESCE(excluded.last_recovery_run_id,
                                                        scheduled_recovery_state.last_recovery_run_id),
                        updated_at = excluded.updated_at",
                )
                .bind(&goal_id)
                .bind(failure_budget)
                .bind(recovery_for_run.as_ref().map(|_| run_id))
                .bind(&now)
                .execute(&mut *tx)
                .await?;
                // Resume only schedules this state machine paused. An owner's
                // independent pause is never silently reversed.
                sqlx::query(
                    "UPDATE goal_schedules SET is_paused = 0, updated_at = ?
                     WHERE id IN (
                         SELECT schedule_id FROM scheduled_recovery_paused_schedules
                         WHERE goal_id = ?
                     )",
                )
                .bind(&now)
                .bind(&goal_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query("DELETE FROM scheduled_recovery_paused_schedules WHERE goal_id = ?")
                    .bind(&goal_id)
                    .execute(&mut *tx)
                    .await?;
            } else if (matches!(status, "failed" | "blocked")
                || status == "completed" && !recovery_completion_verified)
                // A run already counted when it blocked must not be counted a
                // second time when its close-out relabels it `failed`;
                // otherwise a three-failure budget escalates after one or two
                // real occurrences.
                && prior_status != "blocked"
            {
                let authorization_blocked = sqlx::query_scalar::<_, i64>(
                    "SELECT 1 FROM events e
                     JOIN tasks t ON t.id = json_extract(e.data, '$.task_id')
                     WHERE t.goal_run_id = ? AND e.event_type = 'tool_result'
                       AND json_extract(e.data, '$.receipt.authorization_preflight.status')
                           IN ('blocked', 'unverifiable') LIMIT 1",
                )
                .bind(run_id)
                .fetch_optional(&mut *tx)
                .await?
                .is_some();
                let retryable_tool = sqlx::query_scalar::<_, i64>(
                    "SELECT 1 FROM events e
                     JOIN tasks t ON t.id = json_extract(e.data, '$.task_id')
                     WHERE t.goal_run_id = ? AND e.event_type = 'tool_result'
                       AND json_extract(e.data, '$.receipt.outcome_status') = 'failed_retryable'
                     LIMIT 1",
                )
                .bind(run_id)
                .fetch_optional(&mut *tx)
                .await?
                .is_some();
                let permanent_tool = sqlx::query_scalar::<_, i64>(
                    "SELECT 1 FROM events e
                     JOIN tasks t ON t.id = json_extract(e.data, '$.task_id')
                     WHERE t.goal_run_id = ? AND e.event_type = 'tool_result'
                       AND json_extract(e.data, '$.receipt.outcome_status')
                           IN ('failed_permanent', 'blocked') LIMIT 1",
                )
                .bind(run_id)
                .fetch_optional(&mut *tx)
                .await?
                .is_some();
                let task_failed = sqlx::query_scalar::<_, i64>(
                    "SELECT 1 FROM tasks WHERE goal_run_id = ?
                       AND (status IN ('failed', 'interrupted')
                            OR error IS NOT NULL OR blocker IS NOT NULL) LIMIT 1",
                )
                .bind(run_id)
                .fetch_optional(&mut *tx)
                .await?
                .is_some();
                let failure_kind = if authorization_blocked {
                    crate::traits::ScheduledFailureKind::AuthorizationBlocked
                } else if retryable_tool {
                    crate::traits::ScheduledFailureKind::RetryableTool
                } else if permanent_tool {
                    crate::traits::ScheduledFailureKind::PermanentTool
                } else if task_failed {
                    crate::traits::ScheduledFailureKind::TaskFailed
                } else {
                    crate::traits::ScheduledFailureKind::OutcomeUnproven
                };
                let previous_failures = sqlx::query_scalar::<_, i64>(
                    "SELECT consecutive_failures FROM scheduled_recovery_state
                     WHERE goal_id = ?",
                )
                .bind(&goal_id)
                .fetch_optional(&mut *tx)
                .await?
                .unwrap_or(0);
                let consecutive_failures = previous_failures.saturating_add(1);
                let escalated = consecutive_failures >= failure_budget;
                let newly_escalated = escalated && previous_failures < failure_budget;
                let disposition = if escalated {
                    crate::traits::ScheduledRecoveryDisposition::Escalated
                } else {
                    crate::traits::ScheduledRecoveryDisposition::Recovering
                };
                sqlx::query(
                    "INSERT INTO scheduled_recovery_state
                        (goal_id, consecutive_failures, failure_budget, disposition,
                         latest_failure_kind, last_failed_run_id, last_recovery_run_id, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(goal_id) DO UPDATE SET
                        consecutive_failures = excluded.consecutive_failures,
                        failure_budget = excluded.failure_budget,
                        disposition = excluded.disposition,
                        latest_failure_kind = excluded.latest_failure_kind,
                        last_failed_run_id = excluded.last_failed_run_id,
                        last_recovery_run_id = COALESCE(excluded.last_recovery_run_id,
                                                        scheduled_recovery_state.last_recovery_run_id),
                        updated_at = excluded.updated_at",
                )
                .bind(&goal_id)
                .bind(consecutive_failures)
                .bind(failure_budget)
                .bind(disposition.as_str())
                .bind(failure_kind.as_str())
                .bind(run_id)
                .bind(recovery_for_run.as_ref().map(|_| run_id))
                .bind(&now)
                .execute(&mut *tx)
                .await?;

                if newly_escalated {
                    sqlx::query(
                        "INSERT OR IGNORE INTO scheduled_recovery_paused_schedules
                            (schedule_id, goal_id, paused_at)
                         SELECT id, goal_id, ? FROM goal_schedules
                         WHERE goal_id = ? AND is_paused = 0",
                    )
                    .bind(&now)
                    .bind(&goal_id)
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(
                        "UPDATE goal_schedules SET is_paused = 1, updated_at = ?
                         WHERE goal_id = ?",
                    )
                    .bind(&now)
                    .bind(&goal_id)
                    .execute(&mut *tx)
                    .await?;
                    let notification_id = format!(
                        "scheduled-recovery-budget:{}:{}",
                        goal_id, consecutive_failures
                    );
                    sqlx::query(
                        "INSERT OR IGNORE INTO notification_queue
                            (id, goal_id, session_id, notification_type, priority, message,
                             created_at, delivered_at, attempts, expires_at, task_id, action_token)
                         VALUES (?, ?, ?, 'escalation', 'critical', ?, ?, NULL, 0, NULL, NULL, NULL)",
                    )
                    .bind(notification_id)
                    .bind(&goal_id)
                    .bind(&session_id)
                    .bind(format!(
                        "Scheduled objective paused after {consecutive_failures} consecutive failed runs (budget {failure_budget}; typed cause {}). The parent objective remains active for explicit recovery.",
                        failure_kind.as_str()
                    ))
                    .bind(&now)
                    .execute(&mut *tx)
                    .await?;
                }
            }

            if let Some(failed_run_id) = recovery_for_run.as_deref() {
                let link_status = if status == "completed" && recovery_completion_verified {
                    "verified"
                } else if matches!(status, "completed" | "failed" | "cancelled") {
                    "failed"
                } else {
                    "recovering"
                };
                sqlx::query(
                    "INSERT INTO goal_run_recovery_links
                        (failed_run_id, recovery_run_id, outcome_status,
                         proof_receipt_ids_json, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?)
                     ON CONFLICT(failed_run_id, recovery_run_id) DO UPDATE SET
                        outcome_status = excluded.outcome_status,
                        proof_receipt_ids_json = excluded.proof_receipt_ids_json,
                        updated_at = excluded.updated_at",
                )
                .bind(failed_run_id)
                .bind(run_id)
                .bind(link_status)
                .bind(serde_json::to_string(&recovery_proof_receipt_ids)?)
                .bind(&now)
                .bind(&now)
                .execute(&mut *tx)
                .await?;
            }
        }
        tx.commit().await?;
        Ok(true)
    }

    async fn claim_task_with_lease(
        &self,
        task_id: &str,
        worker_instance_id: &str,
        worker_profile_id: Option<&str>,
        lease_secs: i64,
    ) -> anyhow::Result<Option<TaskAttempt>> {
        let lease_secs = normalized_lease_secs(lease_secs);
        let now = chrono::Utc::now();
        let expires = now + chrono::Duration::seconds(lease_secs);
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query(
            "SELECT t.goal_id, t.goal_run_id, t.worker_profile_id, g.project_id
             FROM tasks t JOIN goals g ON g.id = t.goal_id
             WHERE t.id = ? AND t.status IN ('pending', 'claimed')
               AND t.current_attempt_id IS NULL",
        )
        .bind(task_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let goal_id: String = row.get("goal_id");
        let goal_run_id: String = row.get("goal_run_id");
        let project_id: String = row.get("project_id");
        let assigned_profile_id: Option<String> = row.get("worker_profile_id");
        let profile_id = assigned_profile_id
            .or_else(|| worker_profile_id.map(str::to_string))
            .unwrap_or_else(|| "profile-executor".to_string());

        let unmet = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*)
             FROM task_dependencies dep
             WHERE dep.task_id = ?
               AND NOT EXISTS (
                SELECT 1 FROM tasks d
                WHERE d.id = dep.depends_on_task_id
                  AND d.goal_run_id = ?
                  AND d.status IN ('completed', 'skipped', 'superseded')
             )",
        )
        .bind(task_id)
        .bind(&goal_run_id)
        .fetch_one(&mut *tx)
        .await?;
        if unmet > 0 {
            return Ok(None);
        }

        let profile = sqlx::query(
            "SELECT max_concurrency FROM worker_profiles
             WHERE id = ? AND enabled = 1
               AND (project_id IS NULL OR project_id = ?)",
        )
        .bind(&profile_id)
        .bind(&project_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(profile) = profile else {
            anyhow::bail!("worker profile '{}' is unavailable", profile_id);
        };
        // Acquire SQLite's write reservation before counting active slots so
        // two concurrent claims cannot both observe the same final capacity.
        sqlx::query("UPDATE worker_profiles SET updated_at = updated_at WHERE id = ?")
            .bind(&profile_id)
            .execute(&mut *tx)
            .await?;
        let max_concurrency: i64 = profile.get("max_concurrency");
        let active = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM task_attempts
             WHERE worker_profile_id = ?
               AND status IN ('claimed', 'running')
               AND datetime(lease_expires_at) > datetime('now')",
        )
        .bind(&profile_id)
        .fetch_one(&mut *tx)
        .await?;
        if active >= max_concurrency {
            return Ok(None);
        }

        let attempt = TaskAttempt {
            id: uuid::Uuid::new_v4().to_string(),
            task_id: task_id.to_string(),
            goal_run_id: goal_run_id.clone(),
            worker_profile_id: Some(profile_id.clone()),
            worker_instance_id: worker_instance_id.to_string(),
            lease_token: uuid::Uuid::new_v4().to_string(),
            status: "claimed".to_string(),
            lease_expires_at: expires.to_rfc3339(),
            last_heartbeat_at: now.to_rfc3339(),
            workspace_id: None,
            started_at: now.to_rfc3339(),
            completed_at: None,
        };
        sqlx::query(
            "INSERT INTO task_attempts
                (id, task_id, goal_run_id, worker_profile_id, worker_instance_id,
                 lease_token, status, lease_expires_at, last_heartbeat_at,
                 workspace_id, started_at, completed_at)
             VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, ?, NULL, ?, NULL)",
        )
        .bind(&attempt.id)
        .bind(&attempt.task_id)
        .bind(&attempt.goal_run_id)
        .bind(&attempt.worker_profile_id)
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
                 worker_profile_id = ?, started_at = COALESCE(started_at, datetime('now')),
                 result = NULL, error = NULL, blocker = NULL, completed_at = NULL,
                 updated_at = datetime('now'), version = version + 1
             WHERE id = ? AND status IN ('pending', 'claimed')
               AND current_attempt_id IS NULL",
        )
        .bind(worker_instance_id)
        .bind(&attempt.id)
        .bind(&profile_id)
        .bind(task_id)
        .execute(&mut *tx)
        .await?;
        if updated.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }
        sqlx::query(
            "UPDATE goal_runs SET status = 'running', updated_at = datetime('now')
             WHERE id = ? AND status IN ('pending', 'blocked', 'claimed')",
        )
        .bind(&goal_run_id)
        .execute(&mut *tx)
        .await?;
        let journal = TaskJournalEntry::new(
            &project_id,
            &goal_id,
            &goal_run_id,
            "assigned",
            "system",
            worker_instance_id,
            &format!("Claimed by worker profile {}", profile_id),
        )
        .with_task(Some(task_id), Some(&attempt.id));
        insert_journal(&mut tx, &journal).await?;
        tx.commit().await?;
        Ok(Some(attempt))
    }

    async fn get_current_task_attempt(&self, task_id: &str) -> anyhow::Result<Option<TaskAttempt>> {
        let row = sqlx::query(
            "SELECT a.id, a.task_id, a.goal_run_id, a.worker_profile_id,
                    a.worker_instance_id, a.lease_token, a.status,
                    a.lease_expires_at, a.last_heartbeat_at, a.workspace_id,
                    a.started_at, a.completed_at
             FROM task_attempts a
             JOIN tasks t ON t.current_attempt_id = a.id
             WHERE t.id = ? AND a.status IN ('claimed', 'running')",
        )
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.as_ref().map(task_attempt_from_row))
    }

    async fn bind_task_attempt_worker(
        &self,
        attempt_id: &str,
        lease_token: &str,
        worker_instance_id: &str,
        worker_profile_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        // This operation reads the current lease/profile and then writes the
        // bound worker. A deferred transaction can acquire its read snapshot,
        // lose a race to another WAL writer, and fail on promotion with
        // SQLITE_BUSY_SNAPSHOT (extended code 517). Reserve the writer slot
        // before reading so the snapshot can always be committed.
        let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
        let row = sqlx::query(
            "SELECT a.worker_profile_id, g.project_id
             FROM task_attempts a
             JOIN tasks t ON t.current_attempt_id = a.id
             JOIN goals g ON g.id = t.goal_id
             WHERE a.id = ? AND a.lease_token = ?
               AND a.status IN ('claimed', 'running')
               AND datetime(a.lease_expires_at) > datetime('now')",
        )
        .bind(attempt_id)
        .bind(lease_token)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(row) = row else {
            tx.rollback().await?;
            return Ok(false);
        };
        let current_profile_id: Option<String> = row.get("worker_profile_id");
        let project_id: String = row.get("project_id");
        let target_profile_id = worker_profile_id
            .map(str::to_string)
            .or_else(|| current_profile_id.clone());

        if target_profile_id != current_profile_id {
            let Some(target_profile_id) = target_profile_id.as_deref() else {
                tx.rollback().await?;
                return Ok(false);
            };
            let profile = sqlx::query(
                "SELECT max_concurrency FROM worker_profiles
                 WHERE id = ? AND enabled = 1
                   AND (project_id IS NULL OR project_id = ?)",
            )
            .bind(target_profile_id)
            .bind(&project_id)
            .fetch_optional(&mut *tx)
            .await?;
            let Some(profile) = profile else {
                tx.rollback().await?;
                return Ok(false);
            };
            sqlx::query("UPDATE worker_profiles SET updated_at = updated_at WHERE id = ?")
                .bind(target_profile_id)
                .execute(&mut *tx)
                .await?;
            let active = sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM task_attempts
                 WHERE worker_profile_id = ? AND id != ?
                   AND status IN ('claimed', 'running')
                   AND datetime(lease_expires_at) > datetime('now')",
            )
            .bind(target_profile_id)
            .bind(attempt_id)
            .fetch_one(&mut *tx)
            .await?;
            let max_concurrency: i64 = profile.get("max_concurrency");
            if active >= max_concurrency {
                tx.rollback().await?;
                return Ok(false);
            }
        }

        let result = sqlx::query(
            "UPDATE task_attempts
             SET worker_instance_id = ?,
                 worker_profile_id = COALESCE(?, worker_profile_id)
             WHERE id = ? AND lease_token = ?
               AND status IN ('claimed', 'running')
               AND datetime(lease_expires_at) > datetime('now')",
        )
        .bind(worker_instance_id)
        .bind(worker_profile_id)
        .bind(attempt_id)
        .bind(lease_token)
        .execute(&mut *tx)
        .await?;
        if result.rows_affected() > 0 {
            sqlx::query(
                "UPDATE tasks
                 SET agent_id = ?,
                     worker_profile_id = COALESCE(?, worker_profile_id),
                     updated_at = datetime('now'), version = version + 1
                 WHERE current_attempt_id = ?",
            )
            .bind(worker_instance_id)
            .bind(worker_profile_id)
            .bind(attempt_id)
            .execute(&mut *tx)
            .await?;
            tx.commit().await?;
            return Ok(true);
        }
        tx.rollback().await?;
        Ok(false)
    }

    async fn heartbeat_task_attempt(
        &self,
        attempt_id: &str,
        lease_token: &str,
        lease_secs: i64,
    ) -> anyhow::Result<bool> {
        let lease_secs = normalized_lease_secs(lease_secs);
        let result = sqlx::query(
            "UPDATE task_attempts
             SET last_heartbeat_at = datetime('now'),
                 lease_expires_at = datetime('now', '+' || ? || ' seconds')
             WHERE id = ? AND lease_token = ?
               AND status IN ('claimed', 'running')
               AND datetime(lease_expires_at) > datetime('now')",
        )
        .bind(lease_secs)
        .bind(attempt_id)
        .bind(lease_token)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn patch_task_from_attempt(
        &self,
        attempt_id: &str,
        lease_token: &str,
        patch: &TaskAttemptPatch,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            matches!(
                patch.status.as_str(),
                "running" | "completed" | "failed" | "blocked" | "cancelled"
            ),
            "unsupported attempt task transition"
        );
        // Reserve the writer slot before reading the attempt/task fence. A
        // deferred transaction can read a valid lease, lose a race to another
        // WAL writer, and then fail promotion with SQLITE_BUSY_SNAPSHOT. That
        // previously stranded freshly dispatched work in `claimed` state.
        let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
        let row = sqlx::query(
            "SELECT a.task_id, a.goal_run_id, a.status AS attempt_status,
                    t.goal_id, g.project_id
             FROM task_attempts a
             JOIN tasks t ON t.id = a.task_id AND t.current_attempt_id = a.id
             JOIN goals g ON g.id = t.goal_id
             WHERE a.id = ? AND a.lease_token = ?
               AND a.status IN ('claimed', 'running')
               AND datetime(a.lease_expires_at) > datetime('now')",
        )
        .bind(attempt_id)
        .bind(lease_token)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(row) = row else {
            return Ok(false);
        };
        let task_id: String = row.get("task_id");
        let goal_run_id: String = row.get("goal_run_id");
        let goal_id: String = row.get("goal_id");
        let project_id: String = row.get("project_id");
        let terminal = matches!(
            patch.status.as_str(),
            "completed" | "failed" | "blocked" | "cancelled"
        );
        let completed_at = terminal.then(|| chrono::Utc::now().to_rfc3339());
        let result = patch
            .result
            .as_deref()
            .filter(|value| !value.trim().is_empty());
        let error = patch
            .error
            .as_deref()
            .filter(|value| !value.trim().is_empty());
        let blocker = patch
            .blocker
            .as_deref()
            .filter(|value| !value.trim().is_empty());
        let updated = sqlx::query(
            "UPDATE tasks
             SET status = ?,
                 result = COALESCE(?, result),
                 error = CASE WHEN ? = 'completed' THEN NULL ELSE COALESCE(?, error) END,
                 blocker = CASE WHEN ? IN ('completed', 'running') THEN NULL ELSE COALESCE(?, blocker) END,
                 context = COALESCE(?, context),
                 current_attempt_id = CASE WHEN ? THEN NULL ELSE current_attempt_id END,
                 completed_at = CASE WHEN ? THEN ? ELSE NULL END,
                 updated_at = datetime('now'), version = version + 1
             WHERE id = ? AND current_attempt_id = ?",
        )
        .bind(&patch.status)
        .bind(result)
        .bind(&patch.status)
        .bind(error)
        .bind(&patch.status)
        .bind(blocker)
        .bind(&patch.context)
        .bind(terminal)
        .bind(terminal)
        .bind(&completed_at)
        .bind(&task_id)
        .bind(attempt_id)
        .execute(&mut *tx)
        .await?;
        if updated.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        sqlx::query(
            "UPDATE task_attempts
             SET status = ?, completed_at = CASE WHEN ? THEN ? ELSE completed_at END
             WHERE id = ? AND lease_token = ?",
        )
        .bind(&patch.status)
        .bind(terminal)
        .bind(&completed_at)
        .bind(attempt_id)
        .bind(lease_token)
        .execute(&mut *tx)
        .await?;

        if patch.status == "completed" {
            sqlx::query(
                "UPDATE goals SET last_useful_action = datetime('now'), updated_at = datetime('now')
                 WHERE id = ?",
            )
            .bind(&goal_id)
            .execute(&mut *tx)
            .await?;
        } else if patch.status == "blocked" {
            sqlx::query(
                "UPDATE goal_runs SET status = 'blocked', updated_at = datetime('now')
                 WHERE id = ? AND status IN ('pending', 'running')",
            )
            .bind(&goal_run_id)
            .execute(&mut *tx)
            .await?;
        }
        if terminal && patch.status != "blocked" {
            close_pending_task_escalations(&mut tx, &task_id).await?;
        }

        if let Some(handoff) = &patch.handoff {
            anyhow::ensure!(
                handoff.task_id == task_id && handoff.attempt_id == attempt_id,
                "handoff does not match current task attempt"
            );
            let artifacts = serde_json::to_string(&handoff.artifacts)?;
            let verification = serde_json::to_string(&handoff.verification)?;
            sqlx::query(
                "INSERT INTO task_handoffs
                    (id, task_id, attempt_id, summary, artifacts_json,
                     verification_json, remaining_risk, next_step, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&handoff.id)
            .bind(&handoff.task_id)
            .bind(&handoff.attempt_id)
            .bind(crate::tools::sanitize::redact_secrets(&handoff.summary))
            .bind(artifacts)
            .bind(verification)
            .bind(
                handoff
                    .remaining_risk
                    .as_deref()
                    .map(crate::tools::sanitize::redact_secrets),
            )
            .bind(
                handoff
                    .next_step
                    .as_deref()
                    .map(crate::tools::sanitize::redact_secrets),
            )
            .bind(&handoff.created_at)
            .execute(&mut *tx)
            .await?;
        }

        let entry_type = match patch.status.as_str() {
            "blocked" => "blocked",
            "completed" | "failed" | "cancelled" => "handoff",
            _ => "transition",
        };
        let body = patch
            .handoff
            .as_ref()
            .map(|handoff| handoff.summary.as_str())
            .or(result)
            .or(error)
            .or(blocker)
            .unwrap_or(&patch.status);
        let journal = TaskJournalEntry::new(
            &project_id,
            &goal_id,
            &goal_run_id,
            entry_type,
            "agent",
            attempt_id,
            body,
        )
        .with_task(Some(&task_id), Some(attempt_id));
        insert_journal(&mut tx, &journal).await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn recover_expired_task_attempts(&self) -> anyhow::Result<Vec<String>> {
        let mut tx = self.pool.begin().await?;
        let rows = sqlx::query(
            "SELECT a.id AS attempt_id, a.task_id, a.goal_run_id,
                    t.goal_id, g.project_id, t.idempotent, t.retry_count,
                    t.max_retries, t.current_attempt_id
             FROM task_attempts a
             JOIN tasks t ON t.id = a.task_id
             JOIN goals g ON g.id = t.goal_id
             WHERE a.status IN ('claimed', 'running')
               AND datetime(a.lease_expires_at) <= datetime('now')",
        )
        .fetch_all(&mut *tx)
        .await?;
        let mut affected = Vec::new();
        for row in rows {
            let attempt_id: String = row.get("attempt_id");
            let task_id: String = row.get("task_id");
            let goal_run_id: String = row.get("goal_run_id");
            let goal_id: String = row.get("goal_id");
            let project_id: String = row.get("project_id");
            let is_current = row
                .get::<Option<String>, _>("current_attempt_id")
                .as_deref()
                == Some(attempt_id.as_str());
            let recovery = crate::traits::ExpiredAttemptRecovery::classify(
                row.get::<i64, _>("idempotent") != 0,
                row.get::<i64, _>("retry_count") as i32,
                row.get::<i64, _>("max_retries") as i32,
            );
            let attempt_status = match recovery {
                crate::traits::ExpiredAttemptRecovery::Requeue => "expired",
                crate::traits::ExpiredAttemptRecovery::RequireVerification => "needs_verification",
            };
            let expired = sqlx::query(
                "UPDATE task_attempts
                 SET status = ?, completed_at = datetime('now')
                 WHERE id = ? AND status IN ('claimed', 'running')
                   AND datetime(lease_expires_at) <= datetime('now')",
            )
            .bind(attempt_status)
            .bind(&attempt_id)
            .execute(&mut *tx)
            .await?;
            if expired.rows_affected() == 0 {
                continue;
            }
            sqlx::query(
                "UPDATE task_workspaces SET status = 'preserved'
                 WHERE attempt_id = ? AND status = 'active'",
            )
            .bind(&attempt_id)
            .execute(&mut *tx)
            .await?;
            if is_current {
                if recovery == crate::traits::ExpiredAttemptRecovery::Requeue {
                    sqlx::query(
                        "UPDATE tasks
                         SET status = 'pending', current_attempt_id = NULL, agent_id = NULL,
                             retry_count = retry_count + 1, started_at = NULL,
                             completed_at = NULL, result = NULL, error = NULL, blocker = NULL,
                             updated_at = datetime('now'), version = version + 1
                         WHERE id = ? AND current_attempt_id = ?",
                    )
                    .bind(&task_id)
                    .bind(&attempt_id)
                    .execute(&mut *tx)
                    .await?;
                } else {
                    sqlx::query(
                        "UPDATE tasks
                         SET status = 'blocked', current_attempt_id = NULL,
                             blocker = 'Execution lease expired. Verify external side effects before retrying.',
                             result = NULL, error = NULL,
                             completed_at = datetime('now'), updated_at = datetime('now'),
                             version = version + 1
                         WHERE id = ? AND current_attempt_id = ?",
                    )
                    .bind(&task_id)
                    .bind(&attempt_id)
                    .execute(&mut *tx)
                    .await?;
                    sqlx::query(
                        "UPDATE goal_runs SET status = 'blocked', updated_at = datetime('now')
                         WHERE id = ? AND status IN ('pending', 'running')",
                    )
                    .bind(&goal_run_id)
                    .execute(&mut *tx)
                    .await?;
                }
                affected.push(task_id.clone());
            }
            let body = match recovery {
                crate::traits::ExpiredAttemptRecovery::Requeue => {
                    "Worker lease expired; task re-queued because the task is retryable."
                }
                crate::traits::ExpiredAttemptRecovery::RequireVerification => {
                    "Worker lease expired; task requires human verification before another attempt."
                }
            };
            let journal = TaskJournalEntry::new(
                &project_id,
                &goal_id,
                &goal_run_id,
                "lease_lost",
                "system",
                "lease-recovery",
                body,
            )
            .with_task(Some(&task_id), Some(&attempt_id));
            insert_journal(&mut tx, &journal).await?;
        }
        tx.commit().await?;
        Ok(affected)
    }

    async fn append_task_journal(&self, entry: &TaskJournalEntry) -> anyhow::Result<()> {
        let mut tx = self.pool.begin().await?;
        insert_journal(&mut tx, entry).await?;
        tx.commit().await?;
        Ok(())
    }

    async fn get_task_journal(
        &self,
        task_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<TaskJournalEntry>> {
        let rows = sqlx::query(
            "SELECT id, project_id, goal_id, goal_run_id, task_id, attempt_id,
                    entry_type, actor_type, actor_id, source_channel, body, payload, created_at
             FROM task_journal WHERE task_id = ?
             ORDER BY julianday(created_at) DESC, id DESC LIMIT ?",
        )
        .bind(task_id)
        .bind(limit.clamp(1, 200))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|row| TaskJournalEntry {
                id: row.get("id"),
                project_id: row.get("project_id"),
                goal_id: row.get("goal_id"),
                goal_run_id: row.get("goal_run_id"),
                task_id: row.get("task_id"),
                attempt_id: row.get("attempt_id"),
                entry_type: row.get("entry_type"),
                actor_type: row.get("actor_type"),
                actor_id: row.get("actor_id"),
                source_channel: row.get("source_channel"),
                body: row.get("body"),
                payload: row.get("payload"),
                created_at: row.get("created_at"),
            })
            .collect())
    }

    async fn unblock_task(
        &self,
        task_id: &str,
        resolution: &str,
        actor_id: &str,
        source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(!resolution.trim().is_empty(), "resolution cannot be empty");
        let mut tx = self.pool.begin().await?;
        let Some((project_id, goal_id, goal_run_id)) = task_scope(&mut tx, task_id).await? else {
            return Ok(false);
        };
        let attempt_id = sqlx::query_scalar::<_, Option<String>>(
            "SELECT current_attempt_id FROM tasks
             WHERE id = ? AND status = 'blocked'",
        )
        .bind(task_id)
        .fetch_optional(&mut *tx)
        .await?
        .flatten();
        if let Some(attempt_id) = &attempt_id {
            sqlx::query(
                "UPDATE task_attempts SET status = 'cancelled', completed_at = datetime('now')
                 WHERE id = ? AND status IN ('claimed', 'running')",
            )
            .bind(attempt_id)
            .execute(&mut *tx)
            .await?;
        }
        let updated = sqlx::query(
            "UPDATE tasks
             SET status = 'pending', result = NULL, blocker = NULL, error = NULL, completed_at = NULL,
                 started_at = NULL, current_attempt_id = NULL, agent_id = NULL,
                 updated_at = datetime('now'), version = version + 1
             WHERE id = ? AND status = 'blocked'",
        )
        .bind(task_id)
        .execute(&mut *tx)
        .await?;
        if updated.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        close_pending_task_escalations(&mut tx, task_id).await?;
        sqlx::query(
            "UPDATE goal_runs SET status = 'running', updated_at = datetime('now')
             WHERE id = ? AND status = 'blocked'",
        )
        .bind(&goal_run_id)
        .execute(&mut *tx)
        .await?;
        let journal = TaskJournalEntry::new(
            &project_id,
            &goal_id,
            &goal_run_id,
            "unblocked",
            "human",
            actor_id,
            resolution.trim(),
        )
        .with_task(Some(task_id), attempt_id.as_deref())
        .with_source_channel(source_channel);
        insert_journal(&mut tx, &journal).await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn retry_work_task(
        &self,
        task_id: &str,
        actor_id: &str,
        source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        let mut tx = self.pool.begin().await?;
        let Some((project_id, goal_id, goal_run_id)) = task_scope(&mut tx, task_id).await? else {
            return Ok(false);
        };
        let updated = sqlx::query(
            "UPDATE tasks
             SET status = 'pending', result = NULL, blocker = NULL, error = NULL, completed_at = NULL,
                 started_at = NULL, current_attempt_id = NULL, agent_id = NULL,
                 retry_count = retry_count + 1,
                 updated_at = datetime('now'), version = version + 1
             WHERE id = ? AND status IN ('failed', 'blocked', 'interrupted', 'cancelled')
               AND current_attempt_id IS NULL",
        )
        .bind(task_id)
        .execute(&mut *tx)
        .await?;
        if updated.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        close_pending_task_escalations(&mut tx, task_id).await?;
        sqlx::query(
            "UPDATE goal_runs SET status = 'running', updated_at = datetime('now')
             WHERE id = ? AND status = 'blocked'",
        )
        .bind(&goal_run_id)
        .execute(&mut *tx)
        .await?;
        let journal = TaskJournalEntry::new(
            &project_id,
            &goal_id,
            &goal_run_id,
            "transition",
            "human",
            actor_id,
            "Task explicitly queued for another attempt.",
        )
        .with_task(Some(task_id), None)
        .with_source_channel(source_channel);
        insert_journal(&mut tx, &journal).await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn cancel_work_task(
        &self,
        task_id: &str,
        actor_id: &str,
        source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        let mut tx = self.pool.begin().await?;
        let Some((project_id, goal_id, goal_run_id)) = task_scope(&mut tx, task_id).await? else {
            return Ok(false);
        };
        let attempt_id = sqlx::query_scalar::<_, Option<String>>(
            "SELECT current_attempt_id FROM tasks WHERE id = ?",
        )
        .bind(task_id)
        .fetch_optional(&mut *tx)
        .await?
        .flatten();
        if let Some(attempt_id) = &attempt_id {
            sqlx::query(
                "UPDATE task_attempts SET status = 'cancelled', completed_at = datetime('now')
                 WHERE id = ? AND status IN ('claimed', 'running')",
            )
            .bind(attempt_id)
            .execute(&mut *tx)
            .await?;
        }
        let updated = sqlx::query(
            "UPDATE tasks
             SET status = 'cancelled', current_attempt_id = NULL,
                 completed_at = datetime('now'), updated_at = datetime('now'),
                 version = version + 1
             WHERE id = ? AND status NOT IN ('completed', 'cancelled', 'skipped', 'superseded')",
        )
        .bind(task_id)
        .execute(&mut *tx)
        .await?;
        if updated.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        close_pending_task_escalations(&mut tx, task_id).await?;
        let journal = TaskJournalEntry::new(
            &project_id,
            &goal_id,
            &goal_run_id,
            "transition",
            "human",
            actor_id,
            "Task cancelled.",
        )
        .with_task(Some(task_id), attempt_id.as_deref())
        .with_source_channel(source_channel);
        insert_journal(&mut tx, &journal).await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn upsert_worker_profile(&self, profile: &WorkerProfile) -> anyhow::Result<()> {
        anyhow::ensure!(
            profile.max_concurrency > 0,
            "profile concurrency must be positive"
        );
        anyhow::ensure!(
            matches!(
                profile.workspace_policy.as_str(),
                "shared" | "isolated" | "worktree"
            ),
            "unsupported workspace policy"
        );
        sqlx::query(
            "INSERT INTO worker_profiles
                (id, project_id, name, specialist, model, tools_json, max_iterations,
                 tool_budget, timeout_secs, max_concurrency, workspace_policy,
                 memory_scope, version, enabled, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(id) DO UPDATE SET
                project_id = excluded.project_id, name = excluded.name,
                specialist = excluded.specialist, model = excluded.model,
                tools_json = excluded.tools_json, max_iterations = excluded.max_iterations,
                tool_budget = excluded.tool_budget, timeout_secs = excluded.timeout_secs,
                max_concurrency = excluded.max_concurrency,
                workspace_policy = excluded.workspace_policy,
                memory_scope = excluded.memory_scope, version = excluded.version,
                enabled = excluded.enabled, updated_at = excluded.updated_at",
        )
        .bind(&profile.id)
        .bind(&profile.project_id)
        .bind(&profile.name)
        .bind(&profile.specialist)
        .bind(&profile.model)
        .bind(&profile.tools_json)
        .bind(profile.max_iterations)
        .bind(profile.tool_budget)
        .bind(profile.timeout_secs)
        .bind(profile.max_concurrency)
        .bind(&profile.workspace_policy)
        .bind(&profile.memory_scope)
        .bind(profile.version)
        .bind(profile.enabled as i64)
        .bind(&profile.created_at)
        .bind(&profile.updated_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_worker_profile(&self, profile_id: &str) -> anyhow::Result<Option<WorkerProfile>> {
        let row = sqlx::query(
            "SELECT id, project_id, name, specialist, model, tools_json,
                    max_iterations, tool_budget, timeout_secs, max_concurrency,
                    workspace_policy, memory_scope, version, enabled, created_at, updated_at
             FROM worker_profiles WHERE id = ?",
        )
        .bind(profile_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.as_ref().map(worker_profile_from_row))
    }

    async fn list_worker_profiles(
        &self,
        project_id: Option<&str>,
    ) -> anyhow::Result<Vec<WorkerProfile>> {
        let rows = sqlx::query(
            "SELECT id, project_id, name, specialist, model, tools_json,
                    max_iterations, tool_budget, timeout_secs, max_concurrency,
                    workspace_policy, memory_scope, version, enabled, created_at, updated_at
             FROM worker_profiles
             WHERE enabled = 1 AND (project_id IS NULL OR project_id = ?)
             ORDER BY name COLLATE NOCASE",
        )
        .bind(project_id.unwrap_or(DEFAULT_PROJECT_ID))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(worker_profile_from_row).collect())
    }

    async fn assign_task_worker_profile(
        &self,
        task_id: &str,
        profile_id: &str,
        actor_id: &str,
        source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        let mut tx = self.pool.begin().await?;
        let Some((project_id, goal_id, goal_run_id)) = task_scope(&mut tx, task_id).await? else {
            return Ok(false);
        };
        let profile_exists = sqlx::query_scalar::<_, i64>(
            "SELECT 1 FROM worker_profiles
             WHERE id = ? AND enabled = 1
               AND (project_id IS NULL OR project_id = ?) LIMIT 1",
        )
        .bind(profile_id)
        .bind(&project_id)
        .fetch_optional(&mut *tx)
        .await?
        .is_some();
        if !profile_exists {
            return Ok(false);
        }
        let updated = sqlx::query(
            "UPDATE tasks SET worker_profile_id = ?, updated_at = datetime('now'),
                              version = version + 1
             WHERE id = ? AND status IN ('pending', 'blocked', 'failed', 'interrupted')",
        )
        .bind(profile_id)
        .bind(task_id)
        .execute(&mut *tx)
        .await?;
        if updated.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        let journal = TaskJournalEntry::new(
            &project_id,
            &goal_id,
            &goal_run_id,
            "assigned",
            if source_channel.is_some() {
                "human"
            } else {
                "agent"
            },
            actor_id,
            &format!("Assigned worker profile {}", profile_id),
        )
        .with_task(Some(task_id), None)
        .with_source_channel(source_channel);
        insert_journal(&mut tx, &journal).await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn set_task_workspace_policy(&self, task_id: &str, policy: &str) -> anyhow::Result<bool> {
        anyhow::ensure!(
            matches!(policy, "shared" | "isolated" | "worktree"),
            "unsupported workspace policy"
        );
        let result = sqlx::query(
            "UPDATE tasks SET workspace_policy = ?, workspace_policy_explicit = 1,
                              updated_at = datetime('now'),
                              version = version + 1
             WHERE id = ? AND status IN ('pending', 'blocked', 'claimed')
               AND NOT EXISTS (
                   SELECT 1 FROM task_workspaces w
                   WHERE w.task_id = tasks.id AND w.status != 'failed'
               )",
        )
        .bind(policy)
        .bind(task_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn get_task_workspace_policy(&self, task_id: &str) -> anyhow::Result<String> {
        Ok(sqlx::query_scalar::<_, String>(
            "SELECT CASE
                      WHEN t.workspace_policy_explicit = 1
                        THEN t.workspace_policy
                      ELSE COALESCE(wp.workspace_policy, 'shared')
                    END
             FROM tasks t
             LEFT JOIN task_attempts a ON a.id = t.current_attempt_id
             LEFT JOIN worker_profiles wp
               ON wp.id = COALESCE(a.worker_profile_id, t.worker_profile_id)
             WHERE t.id = ?",
        )
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?
        .unwrap_or_else(|| "shared".to_string()))
    }

    async fn create_task_workspace(&self, workspace: &TaskWorkspace) -> anyhow::Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "INSERT INTO task_workspaces
                (id, task_id, attempt_id, backend_id, policy, root_path,
                 branch_name, base_ref, head_ref, status, created_at, released_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&workspace.id)
        .bind(&workspace.task_id)
        .bind(&workspace.attempt_id)
        .bind(&workspace.backend_id)
        .bind(&workspace.policy)
        .bind(&workspace.root_path)
        .bind(&workspace.branch_name)
        .bind(&workspace.base_ref)
        .bind(&workspace.head_ref)
        .bind(&workspace.status)
        .bind(&workspace.created_at)
        .bind(&workspace.released_at)
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE task_attempts SET workspace_id = ? WHERE id = ?")
            .bind(&workspace.id)
            .bind(&workspace.attempt_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn update_task_workspace(&self, workspace: &TaskWorkspace) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE task_workspaces
             SET root_path = ?, branch_name = ?, base_ref = ?, head_ref = ?,
                 status = ?, released_at = ?
             WHERE id = ? AND task_id = ? AND attempt_id = ?",
        )
        .bind(&workspace.root_path)
        .bind(&workspace.branch_name)
        .bind(&workspace.base_ref)
        .bind(&workspace.head_ref)
        .bind(&workspace.status)
        .bind(&workspace.released_at)
        .bind(&workspace.id)
        .bind(&workspace.task_id)
        .bind(&workspace.attempt_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_task_workspace(&self, task_id: &str) -> anyhow::Result<Option<TaskWorkspace>> {
        let row = sqlx::query(
            "SELECT id, task_id, attempt_id, backend_id, policy, root_path,
                    branch_name, base_ref, head_ref, status, created_at, released_at
             FROM task_workspaces WHERE task_id = ?
             ORDER BY julianday(created_at) DESC, id DESC LIMIT 1",
        )
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| TaskWorkspace {
            id: row.get("id"),
            task_id: row.get("task_id"),
            attempt_id: row.get("attempt_id"),
            backend_id: row.get("backend_id"),
            policy: row.get("policy"),
            root_path: row.get("root_path"),
            branch_name: row.get("branch_name"),
            base_ref: row.get("base_ref"),
            head_ref: row.get("head_ref"),
            status: row.get("status"),
            created_at: row.get("created_at"),
            released_at: row.get("released_at"),
        }))
    }

    async fn get_latest_task_handoff(&self, task_id: &str) -> anyhow::Result<Option<TaskHandoff>> {
        let row = sqlx::query(
            "SELECT id, task_id, attempt_id, summary, artifacts_json,
                    verification_json, remaining_risk, next_step, created_at
             FROM task_handoffs WHERE task_id = ?
             ORDER BY julianday(created_at) DESC, id DESC LIMIT 1",
        )
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| {
            let artifacts = row
                .get::<String, _>("artifacts_json")
                .parse::<serde_json::Value>()
                .ok()
                .and_then(|value| serde_json::from_value::<Vec<HandoffArtifact>>(value).ok())
                .unwrap_or_default();
            let verification =
                serde_json::from_str::<Vec<String>>(&row.get::<String, _>("verification_json"))
                    .unwrap_or_default();
            TaskHandoff {
                id: row.get("id"),
                task_id: row.get("task_id"),
                attempt_id: row.get("attempt_id"),
                summary: row.get("summary"),
                artifacts,
                verification,
                remaining_risk: row.get("remaining_risk"),
                next_step: row.get("next_step"),
                created_at: row.get("created_at"),
            }
        }))
    }

    async fn list_work_goals(
        &self,
        project_id: &str,
        include_terminal: bool,
        limit: i64,
    ) -> anyhow::Result<Vec<WorkGoalSummary>> {
        let rows = sqlx::query(
            "WITH ranked_runs AS (
                SELECT gr.*, rg.goal_type,
                       ROW_NUMBER() OVER (
                           PARTITION BY gr.goal_id
                           ORDER BY julianday(gr.started_at) DESC, gr.id DESC
                       ) AS recency,
                       SUM(CASE WHEN gr.status IN ('pending', 'running', 'blocked')
                                THEN 1 ELSE 0 END) OVER (
                           PARTITION BY gr.goal_id
                       ) AS active_count
                FROM goal_runs gr
                JOIN goals rg ON rg.id = gr.goal_id
             ),
             visible_runs AS (
                SELECT * FROM ranked_runs
                WHERE status IN ('pending', 'running', 'blocked')
                   OR (active_count = 0 AND recency = 1
                       AND NOT (trigger_type = 'legacy' AND goal_type = 'continuous'))
             ),
             lanes AS (
                SELECT t.id, t.goal_id, t.goal_run_id,
                    CASE
                        WHEN t.status = 'pending' AND EXISTS (
                            SELECT 1 FROM task_dependencies dep
                            WHERE dep.task_id = t.id
                              AND NOT EXISTS (
                                SELECT 1 FROM tasks d
                                WHERE d.id = dep.depends_on_task_id
                                  AND d.goal_run_id = t.goal_run_id
                                  AND d.status IN ('completed', 'skipped', 'superseded')
                            )
                        ) THEN 'waiting'
                        WHEN t.status = 'pending' THEN 'ready'
                        WHEN t.status IN ('claimed', 'running') AND EXISTS (
                            SELECT 1 FROM task_attempts a
                            WHERE a.id = t.current_attempt_id
                              AND a.status IN ('claimed', 'running')
                              AND datetime(a.lease_expires_at) > datetime('now')
                        ) THEN 'in_progress'
                        WHEN t.status = 'blocked' THEN 'blocked'
                        WHEN t.status IN ('completed', 'cancelled', 'skipped', 'superseded') THEN 'done'
                        ELSE 'needs_attention'
                    END AS lane
                FROM tasks t
             )
             SELECT g.project_id, g.id AS goal_id, g.description, g.status AS goal_status,
                    vr.id AS run_id, vr.status AS run_status,
                    SUM(CASE WHEN l.lane = 'waiting' THEN 1 ELSE 0 END) AS waiting,
                    SUM(CASE WHEN l.lane = 'ready' THEN 1 ELSE 0 END) AS ready,
                    SUM(CASE WHEN l.lane = 'in_progress' THEN 1 ELSE 0 END) AS in_progress,
                    SUM(CASE WHEN l.lane = 'blocked' THEN 1 ELSE 0 END) AS blocked,
                    SUM(CASE WHEN l.lane = 'needs_attention' THEN 1 ELSE 0 END) AS needs_attention,
                    SUM(CASE WHEN l.lane = 'done' THEN 1 ELSE 0 END) AS done,
                    datetime(MAX(julianday(COALESCE(t.updated_at, g.updated_at)))) AS updated_at,
                    MAX(julianday(COALESCE(t.updated_at, g.updated_at))) AS updated_order
             FROM goals g
             LEFT JOIN visible_runs vr ON vr.goal_id = g.id
             LEFT JOIN tasks t ON t.goal_run_id = vr.id
             LEFT JOIN lanes l ON l.id = t.id
             WHERE g.project_id = ? AND g.domain = 'orchestration'
               AND (? OR g.status NOT IN ('completed', 'failed', 'cancelled', 'abandoned'))
             GROUP BY g.id, vr.id
             ORDER BY
                CASE g.priority WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                                WHEN 'medium' THEN 3 ELSE 4 END,
                updated_order DESC
             LIMIT ?",
        )
        .bind(project_id)
        .bind(include_terminal)
        .bind(limit.clamp(1, 200))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|row| WorkGoalSummary {
                project_id: row.get("project_id"),
                goal_id: row.get("goal_id"),
                description: row.get("description"),
                goal_status: row.get("goal_status"),
                run_id: row.get("run_id"),
                run_status: row.get("run_status"),
                waiting: row.get::<i64, _>("waiting"),
                ready: row.get::<i64, _>("ready"),
                in_progress: row.get::<i64, _>("in_progress"),
                blocked: row.get::<i64, _>("blocked"),
                needs_attention: row.get::<i64, _>("needs_attention"),
                done: row.get::<i64, _>("done"),
                updated_at: row.get("updated_at"),
            })
            .collect())
    }

    async fn list_work_tasks(
        &self,
        project_id: &str,
        lane: Option<&str>,
        limit: i64,
    ) -> anyhow::Result<Vec<WorkTaskSummary>> {
        let rows = sqlx::query(
            "WITH ranked_runs AS (
                SELECT gr.*, rg.goal_type,
                       ROW_NUMBER() OVER (
                           PARTITION BY gr.goal_id
                           ORDER BY julianday(gr.started_at) DESC, gr.id DESC
                       ) AS recency,
                       SUM(CASE WHEN gr.status IN ('pending', 'running', 'blocked')
                                THEN 1 ELSE 0 END) OVER (
                           PARTITION BY gr.goal_id
                       ) AS active_count
                FROM goal_runs gr
                JOIN goals rg ON rg.id = gr.goal_id
             ),
             visible_runs AS (
                SELECT * FROM ranked_runs
                WHERE status IN ('pending', 'running', 'blocked')
                   OR (active_count = 0 AND recency = 1
                       AND NOT (trigger_type = 'legacy' AND goal_type = 'continuous'))
             ),
             projected AS (
                SELECT g.project_id, t.goal_id, g.description AS goal_description,
                       t.goal_run_id, t.id AS task_id, t.description, t.status,
                       t.priority, t.blocker, COALESCE(t.updated_at, t.created_at) AS updated_at,
                       wp.name AS worker_profile, a.worker_instance_id, a.lease_expires_at,
                       CASE
                           WHEN t.status = 'pending' AND EXISTS (
                               SELECT 1 FROM task_dependencies dep
                               WHERE dep.task_id = t.id
                                 AND NOT EXISTS (
                                   SELECT 1 FROM tasks d
                                   WHERE d.id = dep.depends_on_task_id
                                     AND d.goal_run_id = t.goal_run_id
                                     AND d.status IN ('completed', 'skipped', 'superseded')
                               )
                           ) THEN 'waiting'
                           WHEN t.status = 'pending' THEN 'ready'
                           WHEN t.status IN ('claimed', 'running') AND a.id IS NOT NULL
                                AND a.status IN ('claimed', 'running')
                                AND datetime(a.lease_expires_at) > datetime('now')
                               THEN 'in_progress'
                           WHEN t.status = 'blocked' THEN 'blocked'
                           WHEN t.status IN ('completed', 'cancelled', 'skipped', 'superseded') THEN 'done'
                           ELSE 'needs_attention'
                       END AS lane
                FROM tasks t
                JOIN goals g ON g.id = t.goal_id
                JOIN visible_runs vr ON vr.id = t.goal_run_id
                LEFT JOIN task_attempts a ON a.id = t.current_attempt_id
                LEFT JOIN worker_profiles wp ON wp.id = COALESCE(a.worker_profile_id, t.worker_profile_id)
                WHERE g.project_id = ? AND g.domain = 'orchestration'
                  AND g.status NOT IN ('completed', 'failed', 'cancelled', 'abandoned')
             )
             SELECT * FROM projected
             WHERE (? IS NULL OR lane = ?)
             ORDER BY
                CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
                julianday(updated_at) DESC
             LIMIT ?",
        )
        .bind(project_id)
        .bind(lane)
        .bind(lane)
        .bind(limit.clamp(1, 500))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|row| WorkTaskSummary {
                project_id: row.get("project_id"),
                goal_id: row.get("goal_id"),
                goal_description: row.get("goal_description"),
                goal_run_id: row.get("goal_run_id"),
                task_id: row.get("task_id"),
                description: row.get("description"),
                status: row.get("status"),
                lane: row.get("lane"),
                priority: row.get("priority"),
                worker_profile: row.get("worker_profile"),
                worker_instance_id: row.get("worker_instance_id"),
                lease_expires_at: row.get("lease_expires_at"),
                blocker: row.get("blocker"),
                updated_at: row.get("updated_at"),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::traits::store_prelude::*;
    use crate::traits::{Goal, GoalSchedule, Task};
    use std::sync::Arc;

    async fn test_store() -> (SqliteStateStore, tempfile::NamedTempFile) {
        let database = tempfile::NamedTempFile::new().unwrap();
        let store = SqliteStateStore::new(
            database.path().to_str().unwrap(),
            100,
            None,
            Arc::new(EmbeddingService::new().unwrap()),
        )
        .await
        .unwrap();
        (store, database)
    }

    fn task(goal_id: &str, description: &str, depends_on: Option<Vec<String>>) -> Task {
        Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            description: description.to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: depends_on.map(|deps| serde_json::to_string(&deps).unwrap()),
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 2,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        }
    }

    fn handoff(task_id: &str, attempt_id: &str, summary: &str) -> TaskHandoff {
        TaskHandoff {
            id: uuid::Uuid::new_v4().to_string(),
            task_id: task_id.to_string(),
            attempt_id: attempt_id.to_string(),
            summary: summary.to_string(),
            artifacts: vec![HandoffArtifact {
                kind: "path".to_string(),
                reference: "/tmp/result".to_string(),
                digest: None,
                metadata: None,
            }],
            verification: vec!["check passed".to_string()],
            remaining_risk: None,
            next_step: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    #[tokio::test]
    async fn goal_run_is_pending_until_a_worker_claims_its_task() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("execute exactly once", "session-a");
        store.create_goal(&goal).await.unwrap();
        let run = store
            .start_goal_run(&goal.id, "manual", None, None)
            .await
            .unwrap();
        assert_eq!(run.status, "pending");

        let root = task(&goal.id, "manual root", None);
        store.create_task(&root).await.unwrap();
        assert_eq!(
            store
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            "pending"
        );

        store
            .claim_task_with_lease(&root.id, "worker-a", Some("profile-task-lead"), 180)
            .await
            .unwrap()
            .expect("claim succeeds");
        assert_eq!(
            store
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            "running"
        );
    }

    #[tokio::test]
    async fn scheduled_run_terminalization_reconciles_child_obligations() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_continuous("synthetic recurrence", "session-a", None, None);
        store.create_goal(&goal).await.unwrap();
        let run = store
            .start_goal_run(&goal.id, "scheduled", None, None)
            .await
            .unwrap();
        let child = task(&goal.id, "unfinished scheduled child", None);
        store.create_task(&child).await.unwrap();

        let premature = store
            .finish_goal_run(&run.id, "completed", Some("premature"))
            .await;
        assert!(premature.is_err());
        assert_eq!(
            store
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .unwrap()
                .id,
            run.id
        );

        assert!(store
            .finish_goal_run(&run.id, "failed", Some("terminal failure"))
            .await
            .unwrap());
        let reconciled = store.get_tasks_for_goal_run(&run.id).await.unwrap();
        assert_eq!(reconciled[0].status, "cancelled");
        assert!(reconciled[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("Parent scheduled occurrence")));
    }

    #[tokio::test]
    async fn claims_are_dependency_aware_fenced_and_handed_off() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("coordinate a release", "session-a");
        store.create_goal(&goal).await.unwrap();
        let first = task(&goal.id, "prepare", None);
        let second = task(&goal.id, "verify", Some(vec![first.id.clone()]));
        store.create_task(&first).await.unwrap();
        store.create_task(&second).await.unwrap();

        assert!(store
            .claim_task_with_lease(&second.id, "worker-b", Some("profile-review"), 180)
            .await
            .unwrap()
            .is_none());
        let first_attempt = store
            .claim_task_with_lease(&first.id, "worker-a", Some("profile-code"), 180)
            .await
            .unwrap()
            .unwrap();
        let patch = TaskAttemptPatch {
            status: "completed".to_string(),
            result: Some("prepared".to_string()),
            handoff: Some(handoff(
                &first.id,
                &first_attempt.id,
                "Preparation complete",
            )),
            ..Default::default()
        };
        assert!(store
            .patch_task_from_attempt(&first_attempt.id, &first_attempt.lease_token, &patch,)
            .await
            .unwrap());
        assert_eq!(
            store
                .get_latest_task_handoff(&first.id)
                .await
                .unwrap()
                .unwrap()
                .verification,
            vec!["check passed"]
        );
        assert!(store
            .claim_task_with_lease(&second.id, "worker-b", Some("profile-review"), 180)
            .await
            .unwrap()
            .is_some());
        assert!(!store
            .patch_task_from_attempt(
                &first_attempt.id,
                &first_attempt.lease_token,
                &TaskAttemptPatch {
                    status: "failed".to_string(),
                    error: Some("late write".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn normalized_dependency_edges_are_atomic_and_reject_cycles() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("dependency graph", "session-a");
        store.create_goal(&goal).await.unwrap();
        let first = task(&goal.id, "first", None);
        let second = task(&goal.id, "second", Some(vec![first.id.clone()]));
        store.create_task(&first).await.unwrap();
        store.create_task(&second).await.unwrap();

        let edge_count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM task_dependencies WHERE task_id = ?",
        )
        .bind(&second.id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(edge_count, 1);

        let mut cyclic = first.clone();
        cyclic.depends_on = Some(serde_json::json!([second.id]).to_string());
        let error = store.update_task(&cyclic).await.unwrap_err();
        assert!(error.to_string().contains("task dependency cycle"));

        let unchanged = store.get_task(&first.id).await.unwrap().unwrap();
        assert!(unchanged.depends_on.is_none());
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM task_dependencies WHERE task_id = ?",
            )
            .bind(&first.id)
            .fetch_one(&store.pool)
            .await
            .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn retries_and_claims_clear_stale_task_outcomes() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("retry cleanly", "session-a");
        store.create_goal(&goal).await.unwrap();

        let failed = task(&goal.id, "failed once", None);
        store.create_task(&failed).await.unwrap();
        sqlx::query(
            "UPDATE tasks
             SET status = 'failed', result = 'old result', error = 'old error',
                 blocker = 'old blocker', completed_at = datetime('now')
             WHERE id = ?",
        )
        .bind(&failed.id)
        .execute(&store.pool)
        .await
        .unwrap();
        assert!(store
            .retry_work_task(&failed.id, "owner", Some("telegram"))
            .await
            .unwrap());
        let retried = store.get_task(&failed.id).await.unwrap().unwrap();
        assert_eq!(retried.status, "pending");
        assert!(retried.result.is_none());
        assert!(retried.error.is_none());
        assert!(retried.blocker.is_none());
        assert!(retried.completed_at.is_none());

        let stale_pending = task(&goal.id, "legacy pending", None);
        store.create_task(&stale_pending).await.unwrap();
        sqlx::query(
            "UPDATE tasks
             SET result = 'stale result', error = 'stale error',
                 blocker = 'stale blocker', completed_at = datetime('now')
             WHERE id = ?",
        )
        .bind(&stale_pending.id)
        .execute(&store.pool)
        .await
        .unwrap();
        assert!(store
            .claim_task_with_lease(&stale_pending.id, "worker-a", None, 180)
            .await
            .unwrap()
            .is_some());
        let claimed = store.get_task(&stale_pending.id).await.unwrap().unwrap();
        assert_eq!(claimed.status, "claimed");
        assert!(claimed.result.is_none());
        assert!(claimed.error.is_none());
        assert!(claimed.blocker.is_none());
        assert!(claimed.completed_at.is_none());
    }

    #[tokio::test]
    async fn expired_leases_retry_safe_work_and_block_ambiguous_work() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("recover work", "session-a");
        store.create_goal(&goal).await.unwrap();
        let safe = task(&goal.id, "safe read", None);
        let mut ambiguous = task(&goal.id, "external write", None);
        ambiguous.idempotent = false;
        ambiguous.max_retries = 0;
        store.create_task(&safe).await.unwrap();
        store.create_task(&ambiguous).await.unwrap();
        let safe_attempt = store
            .claim_task_with_lease(&safe.id, "worker-a", None, 180)
            .await
            .unwrap()
            .unwrap();
        let ambiguous_attempt = store
            .claim_task_with_lease(&ambiguous.id, "worker-b", None, 180)
            .await
            .unwrap()
            .unwrap();
        sqlx::query(
            "UPDATE task_attempts
             SET lease_expires_at = datetime('now', '-1 second')
             WHERE id IN (?, ?)",
        )
        .bind(&safe_attempt.id)
        .bind(&ambiguous_attempt.id)
        .execute(&store.pool)
        .await
        .unwrap();

        let affected = store.recover_expired_task_attempts().await.unwrap();
        assert_eq!(affected.len(), 2);
        let safe_after = store.get_task(&safe.id).await.unwrap().unwrap();
        assert_eq!(safe_after.status, "pending");
        assert_eq!(safe_after.retry_count, 1);
        let ambiguous_after = store.get_task(&ambiguous.id).await.unwrap().unwrap();
        assert_eq!(ambiguous_after.status, "blocked");
        assert!(ambiguous_after
            .blocker
            .unwrap()
            .contains("Verify external side effects"));
        assert!(!store
            .heartbeat_task_attempt(&safe_attempt.id, &safe_attempt.lease_token, 180,)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn worker_profile_capacity_is_enforced() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("bounded workers", "session-a");
        store.create_goal(&goal).await.unwrap();
        let first = task(&goal.id, "one", None);
        let second = task(&goal.id, "two", None);
        let third = task(&goal.id, "three", None);
        store.create_task(&first).await.unwrap();
        store.create_task(&second).await.unwrap();
        store.create_task(&third).await.unwrap();
        assert!(store
            .claim_task_with_lease(&first.id, "one", Some("profile-code"), 180)
            .await
            .unwrap()
            .is_some());
        assert!(store
            .claim_task_with_lease(&second.id, "two", Some("profile-code"), 180)
            .await
            .unwrap()
            .is_some());
        assert!(store
            .claim_task_with_lease(&third.id, "three", Some("profile-code"), 180)
            .await
            .unwrap()
            .is_none());
        let generic_attempt = store
            .claim_task_with_lease(&third.id, "three", Some("profile-executor"), 180)
            .await
            .unwrap()
            .unwrap();
        assert!(!store
            .bind_task_attempt_worker(
                &generic_attempt.id,
                &generic_attempt.lease_token,
                "three",
                Some("profile-code"),
            )
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn worker_binding_waits_for_a_wal_writer_instead_of_losing_its_snapshot() {
        let (store, _database) = test_store().await;
        let mode: String = sqlx::query_scalar("PRAGMA journal_mode = WAL")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(mode.to_ascii_lowercase(), "wal");

        let goal = Goal::new_finite("bind after contention", "session-a");
        store.create_goal(&goal).await.unwrap();
        let work = task(&goal.id, "scheduled root", None);
        store.create_task(&work).await.unwrap();
        let attempt = store
            .claim_task_with_lease(&work.id, "heartbeat-dispatch", None, 180)
            .await
            .unwrap()
            .unwrap();

        let mut competing_writer = store.pool.begin_with("BEGIN IMMEDIATE").await.unwrap();
        sqlx::query("UPDATE goals SET updated_at = datetime('now') WHERE id = ?")
            .bind(&goal.id)
            .execute(&mut *competing_writer)
            .await
            .unwrap();

        let store = Arc::new(store);
        let binding_store = store.clone();
        let attempt_id = attempt.id.clone();
        let lease_token = attempt.lease_token.clone();
        let binding = tokio::spawn(async move {
            binding_store
                .bind_task_attempt_worker(
                    &attempt_id,
                    &lease_token,
                    "specialist:task-lead:test",
                    None,
                )
                .await
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        competing_writer.commit().await.unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(2), binding)
                .await
                .expect("binding should resume after the competing writer commits")
                .unwrap()
                .unwrap()
        );

        let rebound = store
            .get_current_task_attempt(&work.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(rebound.worker_instance_id, "specialist:task-lead:test");
    }

    #[tokio::test]
    async fn explicit_profile_and_workspace_choices_override_fallbacks() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("honor task policy", "session-a");
        store.create_goal(&goal).await.unwrap();
        let assigned = task(&goal.id, "review this", None);
        let workspace = task(&goal.id, "edit safely", None);
        store.create_task(&assigned).await.unwrap();
        store.create_task(&workspace).await.unwrap();

        assert!(store
            .assign_task_worker_profile(&assigned.id, "profile-review", "owner", Some("telegram"))
            .await
            .unwrap());
        let attempt = store
            .claim_task_with_lease(&assigned.id, "worker-a", Some("profile-code"), 180)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(attempt.worker_profile_id.as_deref(), Some("profile-review"));

        let mut code_profile = store
            .get_worker_profile("profile-code")
            .await
            .unwrap()
            .unwrap();
        code_profile.workspace_policy = "worktree".to_string();
        code_profile.version += 1;
        code_profile.updated_at = chrono::Utc::now().to_rfc3339();
        store.upsert_worker_profile(&code_profile).await.unwrap();
        assert!(store
            .assign_task_worker_profile(&workspace.id, "profile-code", "owner", Some("telegram"))
            .await
            .unwrap());
        assert_eq!(
            store
                .get_task_workspace_policy(&workspace.id)
                .await
                .unwrap(),
            "worktree"
        );
        assert!(store
            .set_task_workspace_policy(&workspace.id, "shared")
            .await
            .unwrap());
        assert_eq!(
            store
                .get_task_workspace_policy(&workspace.id)
                .await
                .unwrap(),
            "shared"
        );
    }

    #[tokio::test]
    async fn projects_runs_profiles_and_human_unblocks_are_isolated() {
        let (store, _database) = test_store().await;
        let project = store
            .create_work_project("Operations", Some("production work"))
            .await
            .unwrap();
        assert!(store
            .set_session_work_project("session-ops", &project.id)
            .await
            .unwrap());
        let goal = Goal::new_finite("operate service", "session-ops");
        store.create_goal(&goal).await.unwrap();
        let first_run = store.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        assert_eq!(first_run.project_id, project.id);
        let blocked = task(&goal.id, "needs input", None);
        store.create_task(&blocked).await.unwrap();
        let attempt = store
            .claim_task_with_lease(&blocked.id, "worker-a", None, 180)
            .await
            .unwrap()
            .unwrap();
        assert!(store
            .patch_task_from_attempt(
                &attempt.id,
                &attempt.lease_token,
                &TaskAttemptPatch {
                    status: "blocked".to_string(),
                    blocker: Some("Need approval".to_string()),
                    handoff: Some(handoff(&blocked.id, &attempt.id, "Waiting for approval")),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        let escalation = crate::traits::NotificationEntry::new(
            &goal.id,
            &goal.session_id,
            "escalation",
            "Approval required",
        )
        .with_task(&blocked.id);
        store.enqueue_notification(&escalation).await.unwrap();
        assert!(store
            .get_pending_notifications(20)
            .await
            .unwrap()
            .iter()
            .any(|entry| entry.id == escalation.id));
        assert!(store
            .unblock_task(&blocked.id, "Approved by owner", "owner", Some("telegram"))
            .await
            .unwrap());
        assert!(!store
            .get_pending_notifications(20)
            .await
            .unwrap()
            .iter()
            .any(|entry| entry.id == escalation.id));
        let journal = store.get_task_journal(&blocked.id, 20).await.unwrap();
        assert!(journal.iter().any(|entry| entry.entry_type == "unblocked"));

        store
            .cancel_work_task(&blocked.id, "owner", Some("telegram"))
            .await
            .unwrap();
        store
            .finish_goal_run(&first_run.id, "cancelled", Some("closed"))
            .await
            .unwrap();
        let second_run = store
            .start_goal_run(&goal.id, "manual", None, None)
            .await
            .unwrap();
        let current = task(&goal.id, "current run", None);
        store.create_task(&current).await.unwrap();
        let current_tasks = store.get_tasks_for_goal_run(&second_run.id).await.unwrap();
        assert_eq!(current_tasks.len(), 1);
        assert_eq!(current_tasks[0].id, current.id);
        let board = store.list_work_tasks(&project.id, None, 100).await.unwrap();
        assert_eq!(board.len(), 1);
        assert_eq!(board[0].task_id, current.id);

        let terminal_goal = Goal::new_finite("archived work", "session-ops");
        store.create_goal(&terminal_goal).await.unwrap();
        let terminal_task = task(&terminal_goal.id, "old completed task", None);
        store.create_task(&terminal_task).await.unwrap();
        sqlx::query(
            "UPDATE goals
             SET status = 'completed', completed_at = datetime('now'),
                 updated_at = datetime('now')
             WHERE id = ?",
        )
        .bind(&terminal_goal.id)
        .execute(&store.pool)
        .await
        .unwrap();
        let terminal_run = store
            .get_current_goal_run(&terminal_goal.id)
            .await
            .unwrap()
            .unwrap();
        store
            .finish_goal_run(&terminal_run.id, "completed", Some("archived"))
            .await
            .unwrap();
        let active_board = store.list_work_tasks(&project.id, None, 100).await.unwrap();
        assert_eq!(active_board.len(), 1);
        assert_eq!(active_board[0].task_id, current.id);
    }

    #[tokio::test]
    async fn board_projection_includes_every_active_scheduled_run() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("process every trigger", "session-a");
        store.create_goal(&goal).await.unwrap();
        let initial = task(&goal.id, "initial work", None);
        store.create_task(&initial).await.unwrap();

        let first_scheduled = store
            .start_goal_run(&goal.id, "scheduled", None, None)
            .await
            .unwrap();
        let first_task = task(&goal.id, "first scheduled work", None);
        store.create_task(&first_task).await.unwrap();
        assert_eq!(
            store
                .get_current_goal_run(&goal.id)
                .await
                .unwrap()
                .unwrap()
                .id,
            first_scheduled.id
        );

        let second_scheduled = store
            .start_goal_run(&goal.id, "scheduled", None, None)
            .await
            .unwrap();
        let second_task = task(&goal.id, "second scheduled work", None);
        store.create_task(&second_task).await.unwrap();

        let tasks = store
            .list_work_tasks(DEFAULT_PROJECT_ID, None, 100)
            .await
            .unwrap();
        assert!(tasks.iter().any(|item| item.task_id == initial.id));
        assert!(tasks.iter().any(|item| item.task_id == first_task.id));
        assert!(tasks.iter().any(|item| item.task_id == second_task.id));
        let goals = store
            .list_work_goals(DEFAULT_PROJECT_ID, false, 100)
            .await
            .unwrap();
        assert!(goals
            .iter()
            .any(|item| item.run_id.as_deref() == Some(first_scheduled.id.as_str())));
        assert!(goals
            .iter()
            .any(|item| item.run_id.as_deref() == Some(second_scheduled.id.as_str())));
    }

    #[tokio::test]
    async fn cancelled_tasks_are_archived_in_the_done_lane() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_finite("cancel safely", "session-a");
        store.create_goal(&goal).await.unwrap();
        let cancelled = task(&goal.id, "no longer needed", None);
        store.create_task(&cancelled).await.unwrap();
        assert!(store
            .cancel_work_task(&cancelled.id, "owner", Some("telegram"))
            .await
            .unwrap());

        let done = store
            .list_work_tasks(DEFAULT_PROJECT_ID, Some("done"), 100)
            .await
            .unwrap();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].task_id, cancelled.id);
        assert_eq!(done[0].lane, "done");
        assert!(store
            .list_work_tasks(DEFAULT_PROJECT_ID, Some("needs_attention"), 100)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn terminal_legacy_history_is_not_the_current_continuous_run() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_continuous("run every day", "session-a", None, None);
        store.create_goal(&goal).await.unwrap();
        let legacy = store
            .start_goal_run(&goal.id, "legacy", None, None)
            .await
            .unwrap();
        let historical = task(&goal.id, "old cycle", None);
        store.create_task(&historical).await.unwrap();
        sqlx::query("UPDATE tasks SET status='completed', completed_at=datetime('now') WHERE id=?")
            .bind(&historical.id)
            .execute(&store.pool)
            .await
            .unwrap();
        store
            .finish_goal_run(&legacy.id, "completed", Some("legacy history"))
            .await
            .unwrap();

        assert!(store
            .list_work_tasks(DEFAULT_PROJECT_ID, None, 100)
            .await
            .unwrap()
            .is_empty());
        let goals = store
            .list_work_goals(DEFAULT_PROJECT_ID, false, 100)
            .await
            .unwrap();
        assert_eq!(goals.len(), 1);
        assert!(goals[0].run_id.is_none());
        assert_eq!(goals[0].done, 0);
    }

    #[tokio::test]
    async fn scheduled_failure_budget_escalates_and_resumes_only_owned_pauses() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_continuous("publish synthetic report", "session-a", None, None);
        store.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let active_schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: now.clone(),
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        let owner_paused_schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            is_paused: true,
            ..active_schedule.clone()
        };
        store.create_goal_schedule(&active_schedule).await.unwrap();
        store
            .create_goal_schedule(&owner_paused_schedule)
            .await
            .unwrap();

        for expected_failures in 1..=3 {
            let run = store
                .start_goal_run(&goal.id, "scheduled", Some(&active_schedule.id), None)
                .await
                .unwrap();
            assert!(store
                .finish_goal_run(&run.id, "failed", Some("synthetic failed run"))
                .await
                .unwrap());
            let recovery = store
                .get_scheduled_recovery_state(&goal.id)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(recovery.consecutive_failures, expected_failures);
            assert_eq!(
                recovery.latest_failure_kind,
                Some(crate::traits::ScheduledFailureKind::OutcomeUnproven)
            );
        }

        let escalated = store
            .get_scheduled_recovery_state(&goal.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            escalated.disposition,
            crate::traits::ScheduledRecoveryDisposition::Escalated
        );
        let schedules = store.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(schedules.iter().all(|schedule| schedule.is_paused));
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
        assert!(store
            .get_pending_notifications(20)
            .await
            .unwrap()
            .iter()
            .any(|notification| {
                notification.goal_id == goal.id && notification.notification_type == "escalation"
            }));

        let recovery_run = store
            .start_goal_run(&goal.id, "scheduled", Some(&active_schedule.id), None)
            .await
            .unwrap();
        let mut recovery_task = task(&goal.id, "verify synthetic recovery", None);
        recovery_task.context = Some(
            serde_json::json!({
                "recovery_for_run": escalated.last_failed_run_id,
                "terminal_recovery": true
            })
            .to_string(),
        );
        store.create_task(&recovery_task).await.unwrap();
        sqlx::query("UPDATE goal_runs SET root_task_id = ? WHERE id = ?")
            .bind(&recovery_task.id)
            .bind(&recovery_run.id)
            .execute(&store.pool)
            .await
            .unwrap();
        let proof_receipt_id = "synthetic-recovery-proof";
        sqlx::query(
            "INSERT INTO events
                (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES (?, 'tool_result', ?, ?, ?, 'http_request')",
        )
        .bind(&goal.session_id)
        .bind(
            serde_json::json!({
                "task_id": recovery_task.id,
                "tool_call_id": proof_receipt_id,
                "receipt": {
                    "schema_version": crate::events::ToolReceiptV1::SCHEMA_VERSION,
                    "outcome_status": "succeeded",
                    "outcome_evidence": "structured_metadata"
                }
            })
            .to_string(),
        )
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(&recovery_task.id)
        .execute(&store.pool)
        .await
        .unwrap();
        recovery_task.status = "completed".to_string();
        recovery_task.result = Some("synthetic recovery verified".to_string());
        recovery_task.completed_at = Some(chrono::Utc::now().to_rfc3339());
        store.update_task(&recovery_task).await.unwrap();
        assert!(store
            .finish_goal_run(
                &recovery_run.id,
                "completed",
                Some("synthetic verified recovery"),
            )
            .await
            .unwrap());
        let healthy = store
            .get_scheduled_recovery_state(&goal.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(healthy.consecutive_failures, 0);
        assert_eq!(
            healthy.disposition,
            crate::traits::ScheduledRecoveryDisposition::Healthy
        );
        let recovery_link: (String, String) = sqlx::query_as(
            "SELECT outcome_status, proof_receipt_ids_json
             FROM goal_run_recovery_links WHERE recovery_run_id = ?",
        )
        .bind(&recovery_run.id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(recovery_link.0, "verified");
        assert_eq!(
            serde_json::from_str::<Vec<String>>(&recovery_link.1).unwrap(),
            vec![proof_receipt_id]
        );
        let schedules = store.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(
            !schedules
                .iter()
                .find(|schedule| schedule.id == active_schedule.id)
                .unwrap()
                .is_paused
        );
        assert!(
            schedules
                .iter()
                .find(|schedule| schedule.id == owner_paused_schedule.id)
                .unwrap()
                .is_paused
        );
    }
}
