use super::*;

#[async_trait]
impl crate::traits::GoalStore for SqliteStateStore {
    async fn create_goal(&self, goal: &Goal) -> anyhow::Result<()> {
        // Enforce hard cap of 10 active evergreen goals (orchestration only).
        if goal.domain == "orchestration" && goal.goal_type == "continuous" {
            let count = self.count_active_evergreen_goals().await?;
            if count >= 10 {
                anyhow::bail!(
                    "Cannot create evergreen goal: hard cap of 10 active evergreen goals reached (current: {})",
                    count
                );
            }
        }

        let progress_notes_json = goal
            .progress_notes
            .as_ref()
            .map(|p| serde_json::to_string(p).unwrap_or_default());

        sqlx::query(
            "INSERT INTO goals (
                id, description, domain, goal_type, status, priority, conditions, context, resources,
                budget_per_check, budget_daily, tokens_used_today, tokens_used_day, last_useful_action,
                created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at,
                notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id
             )
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&goal.id)
        .bind(&goal.description)
        .bind(&goal.domain)
        .bind(&goal.goal_type)
        .bind(&goal.status)
        .bind(&goal.priority)
        .bind(&goal.conditions)
        .bind(&goal.context)
        .bind(&goal.resources)
        .bind(goal.budget_per_check)
        .bind(goal.budget_daily)
        .bind(goal.tokens_used_today)
        .bind(&goal.tokens_used_day)
        .bind(&goal.last_useful_action)
        .bind(&goal.created_at)
        .bind(&goal.updated_at)
        .bind(&goal.completed_at)
        .bind(&goal.parent_goal_id)
        .bind(&goal.session_id)
        .bind(&goal.notified_at)
        .bind(goal.notification_attempts)
        .bind(goal.dispatch_failures)
        .bind(&progress_notes_json)
        .bind(goal.source_episode_id)
        .bind(goal.legacy_int_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_goal(&self, id: &str) -> anyhow::Result<Option<Goal>> {
        let row = sqlx::query(
            "SELECT id, description, domain, goal_type, status, priority, conditions,
             context, resources, budget_per_check, budget_daily, tokens_used_today, tokens_used_day,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id,
             notified_at, notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id
             FROM goals WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| {
            let progress_notes_json: Option<String> = r.get("progress_notes");
            let progress_notes = progress_notes_json.and_then(|j| serde_json::from_str(&j).ok());

            Goal {
                id: r.get("id"),
                description: r.get("description"),
                domain: r.get("domain"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                tokens_used_day: r.get("tokens_used_day"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
                progress_notes,
                source_episode_id: r.get("source_episode_id"),
                legacy_int_id: r.get("legacy_int_id"),
            }
        }))
    }

    async fn update_goal(&self, goal: &Goal) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let progress_notes_json = goal
            .progress_notes
            .as_ref()
            .map(|p| serde_json::to_string(p).unwrap_or_default());
        sqlx::query(
            "UPDATE goals SET description = ?, domain = ?, goal_type = ?, status = ?, priority = ?,
             conditions = ?, context = ?, resources = ?,
             budget_per_check = ?, budget_daily = ?, tokens_used_today = ?, tokens_used_day = ?,
             last_useful_action = ?, updated_at = ?, completed_at = ?,
             parent_goal_id = ?, session_id = ?, notified_at = ?, notification_attempts = ?, dispatch_failures = ?,
             progress_notes = ?, source_episode_id = ?, legacy_int_id = ?
             WHERE id = ?",
        )
        .bind(&goal.description)
        .bind(&goal.domain)
        .bind(&goal.goal_type)
        .bind(&goal.status)
        .bind(&goal.priority)
        .bind(&goal.conditions)
        .bind(&goal.context)
        .bind(&goal.resources)
        .bind(goal.budget_per_check)
        .bind(goal.budget_daily)
        .bind(goal.tokens_used_today)
        .bind(&goal.tokens_used_day)
        .bind(&goal.last_useful_action)
        .bind(&now)
        .bind(&goal.completed_at)
        .bind(&goal.parent_goal_id)
        .bind(&goal.session_id)
        .bind(&goal.notified_at)
        .bind(goal.notification_attempts)
        .bind(goal.dispatch_failures)
        .bind(&progress_notes_json)
        .bind(goal.source_episode_id)
        .bind(goal.legacy_int_id)
        .bind(&goal.id)
        .execute(&self.pool)
        .await?;

        // If a goal is terminal, purge schedules so they don't linger as dead rows.
        // This keeps the DB consistent even when cancellation/completion happens via
        // bulk tools or non-tool code paths.
        if goal.domain == "orchestration"
            && matches!(goal.status.as_str(), "cancelled" | "completed")
        {
            sqlx::query("DELETE FROM goal_schedules WHERE goal_id = ?")
                .bind(&goal.id)
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    async fn get_active_goals(&self) -> anyhow::Result<Vec<Goal>> {
        let rows = sqlx::query(
            "SELECT id, description, domain, goal_type, status, priority, conditions,
             context, resources, budget_per_check, budget_daily, tokens_used_today, tokens_used_day,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id,
             notified_at, notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id
             FROM goals
             WHERE domain = 'orchestration' AND status IN ('active', 'pending')
             ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let progress_notes_json: Option<String> = r.get("progress_notes");
                let progress_notes =
                    progress_notes_json.and_then(|j| serde_json::from_str(&j).ok());
                Goal {
                    id: r.get("id"),
                    description: r.get("description"),
                    domain: r.get("domain"),
                    goal_type: r.get("goal_type"),
                    status: r.get("status"),
                    priority: r.get("priority"),
                    conditions: r.get("conditions"),
                    context: r.get("context"),
                    resources: r.get("resources"),
                    budget_per_check: r.get("budget_per_check"),
                    budget_daily: r.get("budget_daily"),
                    tokens_used_today: r.get("tokens_used_today"),
                    tokens_used_day: r.get("tokens_used_day"),
                    last_useful_action: r.get("last_useful_action"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                    completed_at: r.get("completed_at"),
                    parent_goal_id: r.get("parent_goal_id"),
                    session_id: r.get("session_id"),
                    notified_at: r.get("notified_at"),
                    notification_attempts: r.get::<i32, _>("notification_attempts"),
                    dispatch_failures: r.get::<i32, _>("dispatch_failures"),
                    progress_notes,
                    source_episode_id: r.get("source_episode_id"),
                    legacy_int_id: r.get("legacy_int_id"),
                }
            })
            .collect())
    }

    async fn get_active_personal_goals(&self, limit: i64) -> anyhow::Result<Vec<Goal>> {
        let limit = limit.clamp(0, 100);
        let rows = sqlx::query(
            "SELECT id, description, domain, goal_type, status, priority, conditions,
             context, resources, budget_per_check, budget_daily, tokens_used_today, tokens_used_day,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id,
             notified_at, notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id
             FROM goals
             WHERE domain = 'personal' AND status = 'active'
             ORDER BY
               CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
               created_at DESC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let progress_notes_json: Option<String> = r.get("progress_notes");
                let progress_notes =
                    progress_notes_json.and_then(|j| serde_json::from_str(&j).ok());
                Goal {
                    id: r.get("id"),
                    description: r.get("description"),
                    domain: r.get("domain"),
                    goal_type: r.get("goal_type"),
                    status: r.get("status"),
                    priority: r.get("priority"),
                    conditions: r.get("conditions"),
                    context: r.get("context"),
                    resources: r.get("resources"),
                    budget_per_check: r.get("budget_per_check"),
                    budget_daily: r.get("budget_daily"),
                    tokens_used_today: r.get("tokens_used_today"),
                    tokens_used_day: r.get("tokens_used_day"),
                    last_useful_action: r.get("last_useful_action"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                    completed_at: r.get("completed_at"),
                    parent_goal_id: r.get("parent_goal_id"),
                    session_id: r.get("session_id"),
                    notified_at: r.get("notified_at"),
                    notification_attempts: r.get::<i32, _>("notification_attempts"),
                    dispatch_failures: r.get::<i32, _>("dispatch_failures"),
                    progress_notes,
                    source_episode_id: r.get("source_episode_id"),
                    legacy_int_id: r.get("legacy_int_id"),
                }
            })
            .collect())
    }

    async fn update_personal_goal(
        &self,
        goal_id: &str,
        status: Option<&str>,
        progress_note: Option<&str>,
    ) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();

        let mut tx = self.pool.begin().await?;

        if let Some(note) = progress_note {
            let row = sqlx::query(
                "SELECT progress_notes FROM goals WHERE id = ? AND domain = 'personal'",
            )
            .bind(goal_id)
            .fetch_optional(&mut *tx)
            .await?;

            let mut notes: Vec<String> = row
                .and_then(|r| r.get::<Option<String>, _>("progress_notes"))
                .and_then(|j| serde_json::from_str(&j).ok())
                .unwrap_or_default();
            notes.push(note.to_string());
            let notes_json = serde_json::to_string(&notes)?;

            sqlx::query(
                "UPDATE goals
                 SET progress_notes = ?, updated_at = ?
                 WHERE id = ? AND domain = 'personal'",
            )
            .bind(&notes_json)
            .bind(&now)
            .bind(goal_id)
            .execute(&mut *tx)
            .await?;
        }

        if let Some(s) = status {
            let completed_at = if s == "completed" {
                Some(now.clone())
            } else {
                None
            };
            sqlx::query(
                "UPDATE goals
                 SET status = ?, updated_at = ?, completed_at = COALESCE(?, completed_at)
                 WHERE id = ? AND domain = 'personal'",
            )
            .bind(s)
            .bind(&now)
            .bind(&completed_at)
            .bind(goal_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    async fn get_goals_for_session(&self, session_id: &str) -> anyhow::Result<Vec<Goal>> {
        let rows = sqlx::query(
            "SELECT id, description, domain, goal_type, status, priority, conditions,
             context, resources, budget_per_check, budget_daily, tokens_used_today, tokens_used_day,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id,
             notified_at, notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id
             FROM goals
             WHERE domain = 'orchestration' AND session_id = ?
             ORDER BY created_at DESC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let progress_notes_json: Option<String> = r.get("progress_notes");
                let progress_notes =
                    progress_notes_json.and_then(|j| serde_json::from_str(&j).ok());
                Goal {
                    id: r.get("id"),
                    description: r.get("description"),
                    domain: r.get("domain"),
                    goal_type: r.get("goal_type"),
                    status: r.get("status"),
                    priority: r.get("priority"),
                    conditions: r.get("conditions"),
                    context: r.get("context"),
                    resources: r.get("resources"),
                    budget_per_check: r.get("budget_per_check"),
                    budget_daily: r.get("budget_daily"),
                    tokens_used_today: r.get("tokens_used_today"),
                    tokens_used_day: r.get("tokens_used_day"),
                    last_useful_action: r.get("last_useful_action"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                    completed_at: r.get("completed_at"),
                    parent_goal_id: r.get("parent_goal_id"),
                    session_id: r.get("session_id"),
                    notified_at: r.get("notified_at"),
                    notification_attempts: r.get::<i32, _>("notification_attempts"),
                    dispatch_failures: r.get::<i32, _>("dispatch_failures"),
                    progress_notes,
                    source_episode_id: r.get("source_episode_id"),
                    legacy_int_id: r.get("legacy_int_id"),
                }
            })
            .collect())
    }

    async fn get_pending_confirmation_goals(&self, session_id: &str) -> anyhow::Result<Vec<Goal>> {
        let rows = sqlx::query(
            "SELECT id, description, domain, goal_type, status, priority, conditions,
             context, resources, budget_per_check, budget_daily, tokens_used_today, tokens_used_day,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id,
             notified_at, notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id
             FROM goals
             WHERE domain = 'orchestration' AND session_id = ? AND status = 'pending_confirmation'
             ORDER BY created_at DESC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let progress_notes_json: Option<String> = r.get("progress_notes");
                let progress_notes =
                    progress_notes_json.and_then(|j| serde_json::from_str(&j).ok());
                Goal {
                    id: r.get("id"),
                    description: r.get("description"),
                    domain: r.get("domain"),
                    goal_type: r.get("goal_type"),
                    status: r.get("status"),
                    priority: r.get("priority"),
                    conditions: r.get("conditions"),
                    context: r.get("context"),
                    resources: r.get("resources"),
                    budget_per_check: r.get("budget_per_check"),
                    budget_daily: r.get("budget_daily"),
                    tokens_used_today: r.get("tokens_used_today"),
                    tokens_used_day: r.get("tokens_used_day"),
                    last_useful_action: r.get("last_useful_action"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                    completed_at: r.get("completed_at"),
                    parent_goal_id: r.get("parent_goal_id"),
                    session_id: r.get("session_id"),
                    notified_at: r.get("notified_at"),
                    notification_attempts: r.get::<i32, _>("notification_attempts"),
                    dispatch_failures: r.get::<i32, _>("dispatch_failures"),
                    progress_notes,
                    source_episode_id: r.get("source_episode_id"),
                    legacy_int_id: r.get("legacy_int_id"),
                }
            })
            .collect())
    }

    async fn activate_goal(&self, goal_id: &str) -> anyhow::Result<bool> {
        let goal_row = sqlx::query(
            "SELECT goal_type
             FROM goals
             WHERE id = ? AND domain = 'orchestration' AND status = 'pending_confirmation'",
        )
        .bind(goal_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = goal_row else {
            return Ok(false);
        };

        let goal_type: String = row.get("goal_type");
        if goal_type == "continuous" {
            let active_evergreen = self.count_active_evergreen_goals().await?;
            if active_evergreen >= 10 {
                anyhow::bail!(
                    "Cannot activate recurring goal: hard cap of 10 active evergreen goals reached (current: {})",
                    active_evergreen
                );
            }
        }

        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE goals
             SET status = 'active', updated_at = ?
             WHERE id = ? AND domain = 'orchestration' AND status = 'pending_confirmation'",
        )
        .bind(&now)
        .bind(goal_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn create_task(&self, task: &Task) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO tasks (id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error, blocker,
             idempotent, retry_count, max_retries, created_at, started_at, completed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&task.id)
        .bind(&task.goal_id)
        .bind(&task.description)
        .bind(&task.status)
        .bind(&task.priority)
        .bind(task.task_order)
        .bind(&task.parallel_group)
        .bind(&task.depends_on)
        .bind(&task.agent_id)
        .bind(&task.context)
        .bind(&task.result)
        .bind(&task.error)
        .bind(&task.blocker)
        .bind(task.idempotent as i32)
        .bind(task.retry_count)
        .bind(task.max_retries)
        .bind(&task.created_at)
        .bind(&task.started_at)
        .bind(&task.completed_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_task(&self, id: &str) -> anyhow::Result<Option<Task>> {
        let row = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error, blocker,
             idempotent, retry_count, max_retries, created_at, started_at, completed_at
             FROM tasks WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| Task {
            id: r.get("id"),
            goal_id: r.get("goal_id"),
            description: r.get("description"),
            status: r.get("status"),
            priority: r.get("priority"),
            task_order: r.get("task_order"),
            parallel_group: r.get("parallel_group"),
            depends_on: r.get("depends_on"),
            agent_id: r.get("agent_id"),
            context: r.get("context"),
            result: r.get("result"),
            error: r.get("error"),
            blocker: r.get("blocker"),
            idempotent: r.get::<i32, _>("idempotent") != 0,
            retry_count: r.get("retry_count"),
            max_retries: r.get("max_retries"),
            created_at: r.get("created_at"),
            started_at: r.get("started_at"),
            completed_at: r.get("completed_at"),
        }))
    }

    async fn update_task(&self, task: &Task) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE tasks SET description = ?, status = ?, priority = ?, task_order = ?,
             parallel_group = ?, depends_on = ?, agent_id = ?, context = ?,
             result = ?, error = ?, blocker = ?, idempotent = ?,
             retry_count = ?, max_retries = ?, started_at = ?, completed_at = ?
             WHERE id = ?",
        )
        .bind(&task.description)
        .bind(&task.status)
        .bind(&task.priority)
        .bind(task.task_order)
        .bind(&task.parallel_group)
        .bind(&task.depends_on)
        .bind(&task.agent_id)
        .bind(&task.context)
        .bind(&task.result)
        .bind(&task.error)
        .bind(&task.blocker)
        .bind(task.idempotent as i32)
        .bind(task.retry_count)
        .bind(task.max_retries)
        .bind(&task.started_at)
        .bind(&task.completed_at)
        .bind(&task.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_tasks_for_goal(&self, goal_id: &str) -> anyhow::Result<Vec<Task>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error, blocker,
             idempotent, retry_count, max_retries, created_at, started_at, completed_at
             FROM tasks WHERE goal_id = ?
             ORDER BY task_order ASC",
        )
        .bind(goal_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| Task {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                description: r.get("description"),
                status: r.get("status"),
                priority: r.get("priority"),
                task_order: r.get("task_order"),
                parallel_group: r.get("parallel_group"),
                depends_on: r.get("depends_on"),
                agent_id: r.get("agent_id"),
                context: r.get("context"),
                result: r.get("result"),
                error: r.get("error"),
                blocker: r.get("blocker"),
                idempotent: r.get::<i32, _>("idempotent") != 0,
                retry_count: r.get("retry_count"),
                max_retries: r.get("max_retries"),
                created_at: r.get("created_at"),
                started_at: r.get("started_at"),
                completed_at: r.get("completed_at"),
            })
            .collect())
    }

    async fn count_completed_tasks_for_goal(&self, goal_id: &str) -> anyhow::Result<i64> {
        let row = sqlx::query(
            "SELECT COUNT(*) as cnt FROM tasks
             WHERE goal_id = ? AND status IN ('completed', 'skipped')",
        )
        .bind(goal_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("cnt"))
    }

    async fn claim_task(&self, task_id: &str, agent_id: &str) -> anyhow::Result<bool> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE tasks SET status = 'claimed', agent_id = ?, started_at = ?
             WHERE id = ? AND status = 'pending'",
        )
        .bind(agent_id)
        .bind(&now)
        .bind(task_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn log_task_activity(&self, activity: &TaskActivity) -> anyhow::Result<()> {
        // Redact secrets from tool_args and result before persisting
        let redacted_args = activity
            .tool_args
            .as_deref()
            .map(crate::tools::sanitize::redact_secrets);
        let redacted_result = activity
            .result
            .as_deref()
            .map(crate::tools::sanitize::redact_secrets);

        sqlx::query(
            "INSERT INTO task_activity (task_id, activity_type, tool_name, tool_args,
             result, success, tokens_used, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&activity.task_id)
        .bind(&activity.activity_type)
        .bind(&activity.tool_name)
        .bind(&redacted_args)
        .bind(&redacted_result)
        .bind(activity.success.map(|b| b as i32))
        .bind(activity.tokens_used)
        .bind(&activity.created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_task_activities(&self, task_id: &str) -> anyhow::Result<Vec<TaskActivity>> {
        let rows = sqlx::query(
            "SELECT id, task_id, activity_type, tool_name, tool_args, result, success,
             tokens_used, created_at
             FROM task_activity WHERE task_id = ?
             ORDER BY created_at ASC",
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskActivity {
                id: r.get("id"),
                task_id: r.get("task_id"),
                activity_type: r.get("activity_type"),
                tool_name: r.get("tool_name"),
                tool_args: r.get("tool_args"),
                result: r.get("result"),
                success: r.get::<Option<i32>, _>("success").map(|v| v != 0),
                tokens_used: r.get("tokens_used"),
                created_at: r.get("created_at"),
            })
            .collect())
    }

    async fn create_goal_schedule(&self, schedule: &GoalSchedule) -> anyhow::Result<()> {
        if schedule.tz != "local" {
            anyhow::bail!(
                "Only tz='local' is supported for schedules (got tz='{}')",
                schedule.tz
            );
        }

        // Safety: schedules only apply to orchestration goals.
        let domain_row = sqlx::query("SELECT domain FROM goals WHERE id = ?")
            .bind(&schedule.goal_id)
            .fetch_optional(&self.pool)
            .await?;
        if let Some(row) = domain_row {
            let domain: String = row.get("domain");
            if domain != "orchestration" {
                anyhow::bail!(
                    "Cannot create schedule for non-orchestration goal {} (domain={})",
                    schedule.goal_id,
                    domain
                );
            }
        }

        sqlx::query(
            "INSERT INTO goal_schedules
                (id, goal_id, cron_expr, tz, original_schedule, fire_policy, is_one_shot, is_paused, last_run_at, next_run_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&schedule.id)
        .bind(&schedule.goal_id)
        .bind(&schedule.cron_expr)
        .bind(&schedule.tz)
        .bind(&schedule.original_schedule)
        .bind(&schedule.fire_policy)
        .bind(schedule.is_one_shot as i32)
        .bind(schedule.is_paused as i32)
        .bind(&schedule.last_run_at)
        .bind(&schedule.next_run_at)
        .bind(&schedule.created_at)
        .bind(&schedule.updated_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_goal_schedule(&self, schedule_id: &str) -> anyhow::Result<Option<GoalSchedule>> {
        let row = sqlx::query(
            "SELECT id, goal_id, cron_expr, tz, original_schedule, fire_policy, is_one_shot, is_paused, last_run_at, next_run_at, created_at, updated_at
             FROM goal_schedules WHERE id = ?",
        )
        .bind(schedule_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| GoalSchedule {
            id: r.get("id"),
            goal_id: r.get("goal_id"),
            cron_expr: r.get("cron_expr"),
            tz: r.get("tz"),
            original_schedule: r.get("original_schedule"),
            fire_policy: r.get("fire_policy"),
            is_one_shot: r.get::<i64, _>("is_one_shot") != 0,
            is_paused: r.get::<i64, _>("is_paused") != 0,
            last_run_at: r.get("last_run_at"),
            next_run_at: r.get("next_run_at"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        }))
    }

    async fn get_schedules_for_goal(&self, goal_id: &str) -> anyhow::Result<Vec<GoalSchedule>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, cron_expr, tz, original_schedule, fire_policy, is_one_shot, is_paused, last_run_at, next_run_at, created_at, updated_at
             FROM goal_schedules
             WHERE goal_id = ?
             ORDER BY next_run_at ASC",
        )
        .bind(goal_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalSchedule {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                cron_expr: r.get("cron_expr"),
                tz: r.get("tz"),
                original_schedule: r.get("original_schedule"),
                fire_policy: r.get("fire_policy"),
                is_one_shot: r.get::<i64, _>("is_one_shot") != 0,
                is_paused: r.get::<i64, _>("is_paused") != 0,
                last_run_at: r.get("last_run_at"),
                next_run_at: r.get("next_run_at"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
            .collect())
    }

    async fn get_due_goal_schedules(&self, limit: i64) -> anyhow::Result<Vec<GoalSchedule>> {
        let limit = limit.clamp(0, 500);
        let now = chrono::Utc::now().to_rfc3339();
        let rows = sqlx::query(
            "SELECT s.id, s.goal_id, s.cron_expr, s.tz, s.original_schedule, s.fire_policy, s.is_one_shot, s.is_paused, s.last_run_at, s.next_run_at, s.created_at, s.updated_at
             FROM goal_schedules s
             JOIN goals g ON g.id = s.goal_id
             WHERE s.is_paused = 0
               AND s.tz = 'local'
               AND s.next_run_at <= ?
               AND g.domain = 'orchestration'
               AND g.status = 'active'
             ORDER BY s.next_run_at ASC
             LIMIT ?",
        )
        .bind(&now)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalSchedule {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                cron_expr: r.get("cron_expr"),
                tz: r.get("tz"),
                original_schedule: r.get("original_schedule"),
                fire_policy: r.get("fire_policy"),
                is_one_shot: r.get::<i64, _>("is_one_shot") != 0,
                is_paused: r.get::<i64, _>("is_paused") != 0,
                last_run_at: r.get("last_run_at"),
                next_run_at: r.get("next_run_at"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
            .collect())
    }

    async fn update_goal_schedule(&self, schedule: &GoalSchedule) -> anyhow::Result<()> {
        if schedule.tz != "local" {
            anyhow::bail!(
                "Only tz='local' is supported for schedules (got tz='{}')",
                schedule.tz
            );
        }

        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE goal_schedules
             SET goal_id = ?, cron_expr = ?, tz = ?, original_schedule = ?, fire_policy = ?,
                 is_one_shot = ?, is_paused = ?, last_run_at = ?, next_run_at = ?, updated_at = ?
             WHERE id = ?",
        )
        .bind(&schedule.goal_id)
        .bind(&schedule.cron_expr)
        .bind(&schedule.tz)
        .bind(&schedule.original_schedule)
        .bind(&schedule.fire_policy)
        .bind(schedule.is_one_shot as i32)
        .bind(schedule.is_paused as i32)
        .bind(&schedule.last_run_at)
        .bind(&schedule.next_run_at)
        .bind(&now)
        .bind(&schedule.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn delete_goal_schedule(&self, schedule_id: &str) -> anyhow::Result<bool> {
        let result = sqlx::query("DELETE FROM goal_schedules WHERE id = ?")
            .bind(schedule_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn cancel_stale_pending_confirmation_goals(
        &self,
        max_age_secs: i64,
    ) -> anyhow::Result<u64> {
        let now = chrono::Utc::now().to_rfc3339();
        let cutoff = (chrono::Utc::now() - chrono::Duration::seconds(max_age_secs)).to_rfc3339();
        let mut tx = self.pool.begin().await?;

        // If we're cancelling stale pending confirmations, remove their schedules
        // so they don't appear as "scheduled" (zombie schedules).
        sqlx::query(
            "DELETE FROM goal_schedules
             WHERE goal_id IN (
               SELECT id FROM goals WHERE status = 'pending_confirmation' AND created_at < ?
             )",
        )
        .bind(&cutoff)
        .execute(&mut *tx)
        .await?;

        let result = sqlx::query(
            "UPDATE goals
             SET status = 'cancelled', updated_at = ?, completed_at = ?
             WHERE status = 'pending_confirmation' AND created_at < ?",
        )
        .bind(&now)
        .bind(&now)
        .bind(&cutoff)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(result.rows_affected())
    }

    async fn get_scheduled_goals(&self) -> anyhow::Result<Vec<Goal>> {
        let rows = sqlx::query(
            "SELECT id, description, domain, goal_type, status, priority, conditions,
             context, resources, budget_per_check, budget_daily, tokens_used_today, tokens_used_day,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id,
             notified_at, notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id
             FROM goals
             WHERE domain = 'orchestration'
               AND (
                 EXISTS (SELECT 1 FROM goal_schedules s WHERE s.goal_id = goals.id)
                 OR status = 'pending_confirmation'
               )
             ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let progress_notes_json: Option<String> = r.get("progress_notes");
                let progress_notes =
                    progress_notes_json.and_then(|j| serde_json::from_str(&j).ok());
                Goal {
                    id: r.get("id"),
                    description: r.get("description"),
                    domain: r.get("domain"),
                    goal_type: r.get("goal_type"),
                    status: r.get("status"),
                    priority: r.get("priority"),
                    conditions: r.get("conditions"),
                    context: r.get("context"),
                    resources: r.get("resources"),
                    budget_per_check: r.get("budget_per_check"),
                    budget_daily: r.get("budget_daily"),
                    tokens_used_today: r.get("tokens_used_today"),
                    tokens_used_day: r.get("tokens_used_day"),
                    last_useful_action: r.get("last_useful_action"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                    completed_at: r.get("completed_at"),
                    parent_goal_id: r.get("parent_goal_id"),
                    session_id: r.get("session_id"),
                    notified_at: r.get("notified_at"),
                    notification_attempts: r.get::<i32, _>("notification_attempts"),
                    dispatch_failures: r.get::<i32, _>("dispatch_failures"),
                    progress_notes,
                    source_episode_id: r.get("source_episode_id"),
                    legacy_int_id: r.get("legacy_int_id"),
                }
            })
            .collect())
    }

    async fn reset_daily_token_budgets(&self) -> anyhow::Result<u64> {
        let result = sqlx::query(
            "UPDATE goals
             SET tokens_used_today = 0, tokens_used_day = date('now')
             WHERE domain = 'orchestration' AND status = 'active'",
        )
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn set_goal_budgets(
        &self,
        goal_id: &str,
        budget_per_check: Option<i64>,
        budget_daily: Option<i64>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE goals SET budget_per_check = COALESCE(?, budget_per_check),
                              budget_daily = COALESCE(?, budget_daily),
                              updated_at = ? WHERE id = ?",
        )
        .bind(budget_per_check)
        .bind(budget_daily)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(goal_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn add_goal_tokens_and_get_budget_status(
        &self,
        goal_id: &str,
        delta_tokens: i64,
    ) -> anyhow::Result<Option<GoalTokenBudgetStatus>> {
        let mut tx = self.pool.begin().await?;
        let today = chrono::Utc::now().date_naive().to_string();

        // Lazy daily reset keyed by UTC day anchor.
        let _ = sqlx::query(
            "UPDATE goals
             SET tokens_used_today = 0, tokens_used_day = ?
             WHERE id = ? AND (tokens_used_day IS NULL OR tokens_used_day != ?)",
        )
        .bind(&today)
        .bind(goal_id)
        .bind(&today)
        .execute(&mut *tx)
        .await;

        if delta_tokens != 0 {
            sqlx::query(
                "UPDATE goals
                 SET tokens_used_today = MAX(0, tokens_used_today + ?)
                 WHERE id = ?",
            )
            .bind(delta_tokens)
            .bind(goal_id)
            .execute(&mut *tx)
            .await?;
        }

        let row = sqlx::query(
            "SELECT budget_per_check, budget_daily, tokens_used_today
             FROM goals
             WHERE id = ?",
        )
        .bind(goal_id)
        .fetch_optional(&mut *tx)
        .await?;

        tx.commit().await?;

        Ok(row.map(|r| GoalTokenBudgetStatus {
            budget_per_check: r.get("budget_per_check"),
            budget_daily: r.get("budget_daily"),
            tokens_used_today: r.get("tokens_used_today"),
        }))
    }

    async fn get_pending_tasks_by_priority(&self, limit: i64) -> anyhow::Result<Vec<Task>> {
        let rows = sqlx::query(
            "SELECT t.id, t.goal_id, t.description, t.status, t.priority, t.task_order,
             t.parallel_group, t.depends_on, t.agent_id, t.context, t.result, t.error,
             t.blocker, t.idempotent, t.retry_count, t.max_retries, t.created_at,
             t.started_at, t.completed_at
             FROM tasks t
             JOIN goals g ON t.goal_id = g.id AND g.domain = 'orchestration' AND g.status = 'active'
             WHERE t.status = 'pending'
             AND NOT EXISTS (
                 SELECT 1 FROM json_each(COALESCE(t.depends_on, '[]')) AS dep
                 WHERE NOT EXISTS (
                     SELECT 1 FROM tasks d WHERE d.id = dep.value AND d.status = 'completed'
                 )
             )
             ORDER BY
                 CASE t.priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
                 t.task_order ASC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| Task {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                description: r.get("description"),
                status: r.get("status"),
                priority: r.get("priority"),
                task_order: r.get("task_order"),
                parallel_group: r.get("parallel_group"),
                depends_on: r.get("depends_on"),
                agent_id: r.get("agent_id"),
                context: r.get("context"),
                result: r.get("result"),
                error: r.get("error"),
                blocker: r.get("blocker"),
                idempotent: r.get::<i32, _>("idempotent") != 0,
                retry_count: r.get("retry_count"),
                max_retries: r.get("max_retries"),
                created_at: r.get("created_at"),
                started_at: r.get("started_at"),
                completed_at: r.get("completed_at"),
            })
            .collect())
    }

    async fn get_stuck_tasks(&self, timeout_secs: i64) -> anyhow::Result<Vec<Task>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error,
             blocker, idempotent, retry_count, max_retries, created_at,
             started_at, completed_at
             FROM tasks
             WHERE status IN ('running', 'claimed')
             AND datetime(started_at) < datetime('now', '-' || ? || ' seconds')
             ORDER BY started_at ASC",
        )
        .bind(timeout_secs)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| Task {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                description: r.get("description"),
                status: r.get("status"),
                priority: r.get("priority"),
                task_order: r.get("task_order"),
                parallel_group: r.get("parallel_group"),
                depends_on: r.get("depends_on"),
                agent_id: r.get("agent_id"),
                context: r.get("context"),
                result: r.get("result"),
                error: r.get("error"),
                blocker: r.get("blocker"),
                idempotent: r.get::<i32, _>("idempotent") != 0,
                retry_count: r.get("retry_count"),
                max_retries: r.get("max_retries"),
                created_at: r.get("created_at"),
                started_at: r.get("started_at"),
                completed_at: r.get("completed_at"),
            })
            .collect())
    }

    async fn get_recently_completed_tasks(&self, since: &str) -> anyhow::Result<Vec<Task>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error,
             blocker, idempotent, retry_count, max_retries, created_at,
             started_at, completed_at
             FROM tasks
             WHERE status = 'completed' AND completed_at > ?
             ORDER BY completed_at DESC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| Task {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                description: r.get("description"),
                status: r.get("status"),
                priority: r.get("priority"),
                task_order: r.get("task_order"),
                parallel_group: r.get("parallel_group"),
                depends_on: r.get("depends_on"),
                agent_id: r.get("agent_id"),
                context: r.get("context"),
                result: r.get("result"),
                error: r.get("error"),
                blocker: r.get("blocker"),
                idempotent: r.get::<i32, _>("idempotent") != 0,
                retry_count: r.get("retry_count"),
                max_retries: r.get("max_retries"),
                created_at: r.get("created_at"),
                started_at: r.get("started_at"),
                completed_at: r.get("completed_at"),
            })
            .collect())
    }

    async fn mark_task_interrupted(&self, task_id: &str) -> anyhow::Result<bool> {
        let result = sqlx::query(
            "UPDATE tasks SET status = 'interrupted',
             completed_at = datetime('now')
             WHERE id = ? AND status IN ('running', 'claimed')",
        )
        .bind(task_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn count_active_evergreen_goals(&self) -> anyhow::Result<i64> {
        let row = sqlx::query(
            "SELECT COUNT(*) as cnt FROM goals
             WHERE domain = 'orchestration' AND goal_type = 'continuous' AND status = 'active'",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("cnt"))
    }

    async fn get_goals_needing_notification(&self) -> anyhow::Result<Vec<Goal>> {
        let rows = sqlx::query(
            "SELECT id, description, domain, goal_type, status, priority, conditions,
             context, resources, budget_per_check, budget_daily, tokens_used_today, tokens_used_day,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id,
             session_id, notified_at, notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id
             FROM goals
             WHERE domain = 'orchestration'
               AND status IN ('completed', 'failed', 'stalled')
               AND notified_at IS NULL
               AND goal_type = 'finite'
               AND notification_attempts < 3",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let progress_notes_json: Option<String> = r.get("progress_notes");
                let progress_notes =
                    progress_notes_json.and_then(|j| serde_json::from_str(&j).ok());
                Goal {
                    id: r.get("id"),
                    description: r.get("description"),
                    domain: r.get("domain"),
                    goal_type: r.get("goal_type"),
                    status: r.get("status"),
                    priority: r.get("priority"),
                    conditions: r.get("conditions"),
                    context: r.get("context"),
                    resources: r.get("resources"),
                    budget_per_check: r.get("budget_per_check"),
                    budget_daily: r.get("budget_daily"),
                    tokens_used_today: r.get("tokens_used_today"),
                    tokens_used_day: r.get("tokens_used_day"),
                    last_useful_action: r.get("last_useful_action"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                    completed_at: r.get("completed_at"),
                    parent_goal_id: r.get("parent_goal_id"),
                    session_id: r.get("session_id"),
                    notified_at: r.get("notified_at"),
                    notification_attempts: r.get::<i32, _>("notification_attempts"),
                    dispatch_failures: r.get::<i32, _>("dispatch_failures"),
                    progress_notes,
                    source_episode_id: r.get("source_episode_id"),
                    legacy_int_id: r.get("legacy_int_id"),
                }
            })
            .collect())
    }

    async fn mark_goal_notified(&self, goal_id: &str) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query("UPDATE goals SET notified_at = ? WHERE id = ?")
            .bind(&now)
            .bind(goal_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn cleanup_stale_goals(&self, stale_hours: i64) -> anyhow::Result<u64> {
        let cutoff = (chrono::Utc::now() - chrono::Duration::hours(stale_hours)).to_rfc3339();
        let now = chrono::Utc::now().to_rfc3339();

        // Finite orchestration goals without schedules: stale active/pending -> failed.
        let result = sqlx::query(
            "UPDATE goals
             SET status = 'failed', updated_at = ?, completed_at = ?
             WHERE domain = 'orchestration'
               AND status IN ('active', 'pending')
               AND goal_type = 'finite'
               AND updated_at < ?
               AND NOT EXISTS (
                   SELECT 1 FROM goal_schedules s WHERE s.goal_id = goals.id
               )",
        )
        .bind(&now)
        .bind(&now)
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }
}
