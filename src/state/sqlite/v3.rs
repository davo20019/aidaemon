use super::*;

#[async_trait]
impl crate::traits::V3Store for SqliteStateStore {
    async fn create_goal_v3(&self, goal: &GoalV3) -> anyhow::Result<()> {
        // Enforce hard cap of 10 active evergreen goals
        if goal.goal_type == "continuous" {
            let count = self.count_active_evergreen_goals().await?;
            if count >= 10 {
                anyhow::bail!(
                    "Cannot create evergreen goal: hard cap of 10 active evergreen goals reached (current: {})",
                    count
                );
            }
        }

        sqlx::query(
            "INSERT INTO goals_v3 (id, description, goal_type, status, priority, conditions,
             schedule, context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&goal.id)
        .bind(&goal.description)
        .bind(&goal.goal_type)
        .bind(&goal.status)
        .bind(&goal.priority)
        .bind(&goal.conditions)
        .bind(&goal.schedule)
        .bind(&goal.context)
        .bind(&goal.resources)
        .bind(goal.budget_per_check)
        .bind(goal.budget_daily)
        .bind(goal.tokens_used_today)
        .bind(&goal.last_useful_action)
        .bind(&goal.created_at)
        .bind(&goal.updated_at)
        .bind(&goal.completed_at)
        .bind(&goal.parent_goal_id)
        .bind(&goal.session_id)
        .bind(&goal.notified_at)
        .bind(goal.notification_attempts)
        .bind(goal.dispatch_failures)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_goal_v3(&self, id: &str) -> anyhow::Result<Option<GoalV3>> {
        let row = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3 WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| GoalV3 {
            id: r.get("id"),
            description: r.get("description"),
            goal_type: r.get("goal_type"),
            status: r.get("status"),
            priority: r.get("priority"),
            conditions: r.get("conditions"),
            schedule: r.get("schedule"),
            context: r.get("context"),
            resources: r.get("resources"),
            budget_per_check: r.get("budget_per_check"),
            budget_daily: r.get("budget_daily"),
            tokens_used_today: r.get("tokens_used_today"),
            last_useful_action: r.get("last_useful_action"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
            completed_at: r.get("completed_at"),
            parent_goal_id: r.get("parent_goal_id"),
            session_id: r.get("session_id"),
            notified_at: r.get("notified_at"),
            notification_attempts: r.get::<i32, _>("notification_attempts"),
            dispatch_failures: r.get::<i32, _>("dispatch_failures"),
        }))
    }

    async fn update_goal_v3(&self, goal: &GoalV3) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE goals_v3 SET description = ?, goal_type = ?, status = ?, priority = ?,
             conditions = ?, schedule = ?, context = ?, resources = ?,
             budget_per_check = ?, budget_daily = ?, tokens_used_today = ?,
             last_useful_action = ?, updated_at = ?, completed_at = ?,
             parent_goal_id = ?, session_id = ?, notified_at = ?, notification_attempts = ?, dispatch_failures = ?
             WHERE id = ?",
        )
        .bind(&goal.description)
        .bind(&goal.goal_type)
        .bind(&goal.status)
        .bind(&goal.priority)
        .bind(&goal.conditions)
        .bind(&goal.schedule)
        .bind(&goal.context)
        .bind(&goal.resources)
        .bind(goal.budget_per_check)
        .bind(goal.budget_daily)
        .bind(goal.tokens_used_today)
        .bind(&goal.last_useful_action)
        .bind(&now)
        .bind(&goal.completed_at)
        .bind(&goal.parent_goal_id)
        .bind(&goal.session_id)
        .bind(&goal.notified_at)
        .bind(goal.notification_attempts)
        .bind(goal.dispatch_failures)
        .bind(&goal.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_active_goals_v3(&self) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3 WHERE status IN ('active', 'pending')
             ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn get_goals_for_session_v3(&self, session_id: &str) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3 WHERE session_id = ?
             ORDER BY created_at DESC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn migrate_legacy_scheduled_tasks_to_v3(&self) -> anyhow::Result<u64> {
        let rows = sqlx::query(
            "SELECT id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused,
                    last_run_at, next_run_at, created_at, updated_at
             FROM scheduled_tasks
             ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut migrated = 0u64;
        let now_local = chrono::Local::now();
        let now_rfc3339 = chrono::Utc::now().to_rfc3339();

        for r in &rows {
            let legacy_id: String = r.get("id");
            let legacy_name: String = r.get("name");
            let legacy_cron: String = r.get("cron_expr");
            let legacy_original_schedule: String = r.get("original_schedule");
            let legacy_prompt: String = r.get("prompt");
            let legacy_source: String = r.get("source");
            let legacy_is_oneshot: bool = r.get::<i64, _>("is_oneshot") != 0;
            let legacy_is_paused: bool = r.get::<i64, _>("is_paused") != 0;
            let legacy_last_run: Option<String> = r.get("last_run_at");
            let legacy_next_run: String = r.get("next_run_at");

            // Deterministic migrated goal ID keeps migration idempotent.
            let migrated_goal_id = format!("legacy-sched-{}", legacy_id);
            if self.get_goal_v3(&migrated_goal_id).await?.is_some() {
                continue;
            }

            let description = if !legacy_prompt.trim().is_empty() {
                legacy_prompt.trim().to_string()
            } else {
                legacy_name.clone()
            };

            let schedule_for_goal = if legacy_is_oneshot {
                // Legacy one-shot rows carry exact next_run_at. Convert to an
                // absolute one-shot cron expression in system-local timezone.
                let target_local = parse_legacy_datetime_to_local(&legacy_next_run)
                    .unwrap_or_else(|| now_local + chrono::Duration::minutes(1));
                let effective_target = if target_local <= now_local {
                    // Catch up overdue legacy one-shots quickly.
                    now_local + chrono::Duration::minutes(1)
                } else {
                    target_local
                };
                format!(
                    "{} {} {} {} *",
                    effective_target.minute(),
                    effective_target.hour(),
                    effective_target.day(),
                    effective_target.month()
                )
            } else {
                legacy_cron.clone()
            };

            let mut goal = if legacy_is_oneshot {
                GoalV3::new_finite(&description, "system")
            } else {
                GoalV3::new_continuous(
                    &description,
                    "system",
                    &schedule_for_goal,
                    Some(5000),
                    Some(20000),
                )
            };

            goal.id = migrated_goal_id;
            goal.schedule = Some(schedule_for_goal.clone());
            goal.status = if legacy_is_paused {
                "paused".to_string()
            } else {
                "active".to_string()
            };

            // Preserve historical timing where possible. If legacy timestamps
            // are non-RFC3339, keep goal timestamps in canonical RFC3339 now.
            goal.created_at = now_rfc3339.clone();
            goal.updated_at = now_rfc3339.clone();
            goal.last_useful_action = legacy_last_run
                .as_deref()
                .and_then(parse_legacy_datetime_to_local)
                .map(|dt| dt.with_timezone(&chrono::Utc).to_rfc3339());
            goal.context = Some(
                serde_json::json!({
                    "migrated_from": "scheduled_tasks",
                    "legacy_task_id": legacy_id,
                    "legacy_name": legacy_name,
                    "legacy_source": legacy_source,
                    "legacy_original_schedule": legacy_original_schedule,
                    "legacy_next_run_at": legacy_next_run,
                })
                .to_string(),
            );

            match self.create_goal_v3(&goal).await {
                Ok(()) => migrated += 1,
                Err(e) => {
                    tracing::warn!(
                        legacy_task_id = %legacy_id,
                        error = %e,
                        "Failed to migrate legacy scheduled task to V3 goal"
                    );
                }
            }
        }

        Ok(migrated)
    }

    async fn get_pending_confirmation_goals(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE session_id = ? AND status = 'pending_confirmation'
             ORDER BY created_at DESC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn activate_goal_v3(&self, goal_id: &str) -> anyhow::Result<bool> {
        let goal_row = sqlx::query(
            "SELECT goal_type
             FROM goals_v3
             WHERE id = ? AND status = 'pending_confirmation'",
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
            "UPDATE goals_v3
             SET status = 'active', updated_at = ?
             WHERE id = ? AND status = 'pending_confirmation'",
        )
        .bind(&now)
        .bind(goal_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn create_task_v3(&self, task: &TaskV3) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO tasks_v3 (id, goal_id, description, status, priority, task_order,
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

    async fn get_task_v3(&self, id: &str) -> anyhow::Result<Option<TaskV3>> {
        let row = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error, blocker,
             idempotent, retry_count, max_retries, created_at, started_at, completed_at
             FROM tasks_v3 WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| TaskV3 {
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

    async fn update_task_v3(&self, task: &TaskV3) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE tasks_v3 SET description = ?, status = ?, priority = ?, task_order = ?,
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

    async fn get_tasks_for_goal_v3(&self, goal_id: &str) -> anyhow::Result<Vec<TaskV3>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error, blocker,
             idempotent, retry_count, max_retries, created_at, started_at, completed_at
             FROM tasks_v3 WHERE goal_id = ?
             ORDER BY task_order ASC",
        )
        .bind(goal_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskV3 {
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
            "SELECT COUNT(*) as cnt FROM tasks_v3
             WHERE goal_id = ? AND status IN ('completed', 'skipped')",
        )
        .bind(goal_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("cnt"))
    }

    async fn claim_task_v3(&self, task_id: &str, agent_id: &str) -> anyhow::Result<bool> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE tasks_v3 SET status = 'claimed', agent_id = ?, started_at = ?
             WHERE id = ? AND status = 'pending'",
        )
        .bind(agent_id)
        .bind(&now)
        .bind(task_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn log_task_activity_v3(&self, activity: &TaskActivityV3) -> anyhow::Result<()> {
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
            "INSERT INTO task_activity_v3 (task_id, activity_type, tool_name, tool_args,
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

    async fn get_task_activities_v3(&self, task_id: &str) -> anyhow::Result<Vec<TaskActivityV3>> {
        let rows = sqlx::query(
            "SELECT id, task_id, activity_type, tool_name, tool_args, result, success,
             tokens_used, created_at
             FROM task_activity_v3 WHERE task_id = ?
             ORDER BY created_at ASC",
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskActivityV3 {
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

    async fn get_due_evergreen_goals(&self) -> anyhow::Result<Vec<GoalV3>> {
        // Get all active continuous goals
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE goal_type = 'continuous' AND status = 'active' AND schedule IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await?;

        let now = chrono::Local::now();
        let mut due_goals = Vec::new();

        for r in &rows {
            let goal = GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            };

            // Check if the schedule is due using croner
            let schedule_str = match &goal.schedule {
                Some(s) => s,
                None => continue,
            };

            // Parse cron expression
            let cron: croner::Cron = match schedule_str.parse() {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Determine last run time — use last_useful_action or created_at
            let last_run = goal
                .last_useful_action
                .as_deref()
                .or(Some(goal.created_at.as_str()));
            let last_run_dt = match last_run {
                Some(ts) => match chrono::DateTime::parse_from_rfc3339(ts) {
                    Ok(dt) => dt.with_timezone(&chrono::Local),
                    Err(_) => continue,
                },
                None => continue,
            };

            // Check if there's a scheduled time between last_run and now
            if let Ok(next) = cron.find_next_occurrence(&last_run_dt, false) {
                if next <= now {
                    // Check daily budget
                    if let Some(daily_budget) = goal.budget_daily {
                        if goal.tokens_used_today >= daily_budget {
                            continue; // Over budget
                        }
                    }
                    due_goals.push(goal);
                }
            }
        }

        Ok(due_goals)
    }

    async fn get_due_scheduled_finite_goals(&self) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE goal_type = 'finite' AND status = 'active' AND schedule IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await?;

        let now = chrono::Local::now();
        let mut due_goals = Vec::new();

        for r in &rows {
            let goal = GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            };

            let schedule_str = match &goal.schedule {
                Some(s) => s,
                None => continue,
            };

            let cron: croner::Cron = match schedule_str.parse() {
                Ok(c) => c,
                Err(_) => continue,
            };

            let anchor = goal
                .last_useful_action
                .as_deref()
                .or(Some(goal.created_at.as_str()));
            let anchor_dt = match anchor {
                Some(ts) => match chrono::DateTime::parse_from_rfc3339(ts) {
                    Ok(dt) => dt.with_timezone(&chrono::Local),
                    Err(_) => continue,
                },
                None => continue,
            };

            if let Ok(next) = cron.find_next_occurrence(&anchor_dt, false) {
                if next <= now {
                    due_goals.push(goal);
                }
            }
        }

        Ok(due_goals)
    }

    async fn cancel_stale_pending_confirmation_goals(
        &self,
        max_age_secs: i64,
    ) -> anyhow::Result<u64> {
        let cutoff = (chrono::Utc::now() - chrono::Duration::seconds(max_age_secs)).to_rfc3339();
        let result = sqlx::query(
            "UPDATE goals_v3
             SET status = 'cancelled', updated_at = datetime('now'), completed_at = datetime('now')
             WHERE status = 'pending_confirmation' AND created_at < ?",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn get_scheduled_goals_v3(&self) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE schedule IS NOT NULL OR status = 'pending_confirmation'
             ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn reset_daily_token_budgets(&self) -> anyhow::Result<u64> {
        let result = sqlx::query(
            "UPDATE goals_v3 SET tokens_used_today = 0
             WHERE goal_type = 'continuous' AND status = 'active'",
        )
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn add_goal_tokens_and_get_budget_status(
        &self,
        goal_id: &str,
        delta_tokens: i64,
    ) -> anyhow::Result<Option<GoalTokenBudgetStatus>> {
        let mut tx = self.pool.begin().await?;

        if delta_tokens != 0 {
            sqlx::query(
                "UPDATE goals_v3
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
             FROM goals_v3
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

    async fn get_pending_tasks_by_priority(&self, limit: i64) -> anyhow::Result<Vec<TaskV3>> {
        let rows = sqlx::query(
            "SELECT t.id, t.goal_id, t.description, t.status, t.priority, t.task_order,
             t.parallel_group, t.depends_on, t.agent_id, t.context, t.result, t.error,
             t.blocker, t.idempotent, t.retry_count, t.max_retries, t.created_at,
             t.started_at, t.completed_at
             FROM tasks_v3 t
             JOIN goals_v3 g ON t.goal_id = g.id AND g.status = 'active'
             WHERE t.status = 'pending'
             AND NOT EXISTS (
                 SELECT 1 FROM json_each(COALESCE(t.depends_on, '[]')) AS dep
                 WHERE NOT EXISTS (
                     SELECT 1 FROM tasks_v3 d WHERE d.id = dep.value AND d.status = 'completed'
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
            .map(|r| TaskV3 {
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

    async fn get_stuck_tasks(&self, timeout_secs: i64) -> anyhow::Result<Vec<TaskV3>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error,
             blocker, idempotent, retry_count, max_retries, created_at,
             started_at, completed_at
             FROM tasks_v3
             WHERE status IN ('running', 'claimed')
             AND datetime(started_at) < datetime('now', '-' || ? || ' seconds')
             ORDER BY started_at ASC",
        )
        .bind(timeout_secs)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskV3 {
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

    async fn get_recently_completed_tasks(&self, since: &str) -> anyhow::Result<Vec<TaskV3>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error,
             blocker, idempotent, retry_count, max_retries, created_at,
             started_at, completed_at
             FROM tasks_v3
             WHERE status = 'completed' AND completed_at > ?
             ORDER BY completed_at DESC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskV3 {
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
            "UPDATE tasks_v3 SET status = 'interrupted',
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
            "SELECT COUNT(*) as cnt FROM goals_v3
             WHERE goal_type = 'continuous' AND status = 'active'",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("cnt"))
    }

    async fn get_goals_needing_notification(&self) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id,
             session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE status IN ('completed', 'failed', 'stalled') AND notified_at IS NULL
             AND goal_type = 'finite' AND notification_attempts < 3",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn mark_goal_notified(&self, goal_id: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE goals_v3 SET notified_at = datetime('now') WHERE id = ?")
            .bind(goal_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn cleanup_stale_goals(&self, stale_hours: i64) -> anyhow::Result<(u64, u64)> {
        let cutoff = (chrono::Utc::now() - chrono::Duration::hours(stale_hours)).to_rfc3339();

        // Legacy goals table: active → abandoned
        let legacy = sqlx::query(
            "UPDATE goals SET status = 'abandoned', updated_at = datetime('now')
             WHERE status = 'active' AND updated_at < ?",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;

        // V3 finite goals: active/pending without schedule → failed.
        // Scheduled finite goals are handled by dedicated scheduler phases.
        let v3 = sqlx::query(
            "UPDATE goals_v3 SET status = 'failed',
                    updated_at = datetime('now'),
                    completed_at = datetime('now')
             WHERE status IN ('active', 'pending')
               AND goal_type = 'finite'
               AND schedule IS NULL
               AND updated_at < ?",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;

        Ok((legacy.rows_affected(), v3.rows_affected()))
    }
}
