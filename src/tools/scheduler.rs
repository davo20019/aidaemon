use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::{Row, SqlitePool};
use tracing::info;

use crate::scheduler::{compute_next_run, parse_schedule};
use crate::traits::Tool;

pub struct SchedulerTool {
    pool: SqlitePool,
}

impl SchedulerTool {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[derive(Debug, Deserialize)]
struct SchedulerArgs {
    action: String,
    name: Option<String>,
    schedule: Option<String>,
    prompt: Option<String>,
    #[serde(default)]
    oneshot: bool,
    #[serde(default)]
    trusted: bool,
    id: Option<String>,
}

#[async_trait]
impl Tool for SchedulerTool {
    fn name(&self) -> &str {
        "scheduler"
    }

    fn description(&self) -> &str {
        "Create, list, delete, pause, and resume scheduled tasks and reminders"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "scheduler",
            "description": "Create, list, delete, pause, and resume scheduled tasks and reminders. \
                Supports natural schedule formats like 'daily at 9am', 'every 2h', 'weekdays at 8:30', \
                'in 30m', or raw 5-field cron expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "delete", "pause", "resume"],
                        "description": "The action to perform"
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable label for the task (required for create)"
                    },
                    "schedule": {
                        "type": "string",
                        "description": "When to run. Natural: 'daily at 9am', 'every 5m', 'weekdays at 8:30', 'in 2h'. Or 5-field cron: '0 9 * * 1-5' (required for create)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What the agent should do when the schedule fires (required for create)"
                    },
                    "oneshot": {
                        "type": "boolean",
                        "description": "If true, the task fires once then auto-deletes (default: false)"
                    },
                    "trusted": {
                        "type": "boolean",
                        "description": "If true, task runs with full autonomy (no terminal approval needed). Set to true when the prompt needs terminal access (e.g. 'check disk usage'). Default: false"
                    },
                    "id": {
                        "type": "string",
                        "description": "Task UUID (required for delete, pause, resume)"
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: SchedulerArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            "create" => self.create(args).await,
            "list" => self.list().await,
            "delete" => self.delete(args).await,
            "pause" => self.pause(args).await,
            "resume" => self.resume(args).await,
            other => Ok(format!("Unknown action '{}'. Use: create, list, delete, pause, resume", other)),
        }
    }
}

impl SchedulerTool {
    async fn create(&self, args: SchedulerArgs) -> anyhow::Result<String> {
        let name = args.name.as_deref().unwrap_or("").trim();
        if name.is_empty() {
            return Ok("Error: 'name' is required for create".to_string());
        }

        let schedule = args.schedule.as_deref().unwrap_or("").trim();
        if schedule.is_empty() {
            return Ok("Error: 'schedule' is required for create".to_string());
        }

        let prompt = args.prompt.as_deref().unwrap_or("").trim();
        if prompt.is_empty() {
            return Ok("Error: 'prompt' is required for create".to_string());
        }

        let cron_expr = match parse_schedule(schedule) {
            Ok(expr) => expr,
            Err(e) => return Ok(format!("Error parsing schedule '{}': {}", schedule, e)),
        };

        let next_run = match compute_next_run(&cron_expr) {
            Ok(dt) => dt,
            Err(e) => return Ok(format!("Error computing next run: {}", e)),
        };

        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        let now_str = now.to_rfc3339();
        let next_str = next_run.to_rfc3339();

        sqlx::query(
            "INSERT INTO scheduled_tasks (id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused, is_trusted, next_run_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, 'tool', ?, 0, ?, ?, ?, ?)"
        )
        .bind(&id)
        .bind(name)
        .bind(&cron_expr)
        .bind(schedule)
        .bind(prompt)
        .bind(args.oneshot as i32)
        .bind(args.trusted as i32)
        .bind(&next_str)
        .bind(&now_str)
        .bind(&now_str)
        .execute(&self.pool)
        .await?;

        info!(name = %name, cron = %cron_expr, id = %id, "Created scheduled task");

        Ok(format!(
            "Created scheduled task:\n  ID: {}\n  Name: {}\n  Schedule: {} (cron: {})\n  Prompt: {}\n  One-shot: {}\n  Trusted: {}\n  Next run: {}",
            id, name, schedule, cron_expr, prompt, args.oneshot, args.trusted, next_run.format("%Y-%m-%d %H:%M UTC")
        ))
    }

    async fn list(&self) -> anyhow::Result<String> {
        let rows = sqlx::query(
            "SELECT id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused, is_trusted, last_run_at, next_run_at, created_at
             FROM scheduled_tasks
             ORDER BY next_run_at ASC"
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok("No scheduled tasks found.".to_string());
        }

        let mut output = format!("Scheduled tasks ({}):\n", rows.len());
        for row in rows {
            let id: String = row.get("id");
            let name: String = row.get("name");
            let original_schedule: String = row.get("original_schedule");
            let prompt: String = row.get("prompt");
            let source: String = row.get("source");
            let is_oneshot: bool = row.get::<i32, _>("is_oneshot") != 0;
            let is_paused: bool = row.get::<i32, _>("is_paused") != 0;
            let is_trusted: bool = row.get::<i32, _>("is_trusted") != 0;
            let last_run_at: Option<String> = row.get("last_run_at");
            let next_run_at: String = row.get("next_run_at");

            let status = if is_paused { "PAUSED" } else { "active" };
            let oneshot_label = if is_oneshot { " [one-shot]" } else { "" };
            let trusted_label = if is_trusted { " [trusted]" } else { "" };
            let last_run = last_run_at.as_deref().unwrap_or("never");

            output.push_str(&format!(
                "\n• {} ({})\n  ID: {}\n  Schedule: {}\n  Prompt: {}\n  Status: {}{}{}\n  Source: {}\n  Last run: {}\n  Next run: {}\n",
                name, status, id, original_schedule, prompt, status, oneshot_label, trusted_label, source, last_run, next_run_at
            ));
        }

        Ok(output)
    }

    async fn delete(&self, args: SchedulerArgs) -> anyhow::Result<String> {
        let id = match args.id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => return Ok("Error: 'id' is required for delete".to_string()),
        };

        let result = sqlx::query("DELETE FROM scheduled_tasks WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            Ok(format!("No scheduled task found with ID '{}'", id))
        } else {
            info!(id = %id, "Deleted scheduled task");
            Ok(format!("Deleted scheduled task {}", id))
        }
    }

    async fn pause(&self, args: SchedulerArgs) -> anyhow::Result<String> {
        let id = match args.id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => return Ok("Error: 'id' is required for pause".to_string()),
        };

        let now_str = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query("UPDATE scheduled_tasks SET is_paused = 1, updated_at = ? WHERE id = ?")
            .bind(&now_str)
            .bind(id)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            Ok(format!("No scheduled task found with ID '{}'", id))
        } else {
            info!(id = %id, "Paused scheduled task");
            Ok(format!("Paused scheduled task {}", id))
        }
    }

    async fn resume(&self, args: SchedulerArgs) -> anyhow::Result<String> {
        let id = match args.id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => return Ok("Error: 'id' is required for resume".to_string()),
        };

        // When resuming, recompute next_run_at from now
        let row = sqlx::query("SELECT cron_expr FROM scheduled_tasks WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;

        let cron_expr: String = match row {
            Some(r) => r.get("cron_expr"),
            None => return Ok(format!("No scheduled task found with ID '{}'", id)),
        };

        let next_run = compute_next_run(&cron_expr)?;
        let now_str = chrono::Utc::now().to_rfc3339();
        let next_str = next_run.to_rfc3339();

        sqlx::query("UPDATE scheduled_tasks SET is_paused = 0, next_run_at = ?, updated_at = ? WHERE id = ?")
            .bind(&next_str)
            .bind(&now_str)
            .bind(id)
            .execute(&self.pool)
            .await?;

        info!(id = %id, "Resumed scheduled task");
        Ok(format!("Resumed scheduled task {}. Next run: {}", id, next_run.format("%Y-%m-%d %H:%M UTC")))
    }
}
