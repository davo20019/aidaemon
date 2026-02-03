use chrono::{DateTime, Datelike, Timelike, Utc};
use croner::Cron;
use regex::Regex;
use sqlx::{Row, SqlitePool};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

use crate::config::ScheduledTaskConfig;
use crate::traits::Event;

pub struct SchedulerManager {
    pool: SqlitePool,
    sender: broadcast::Sender<Event>,
    tick_interval: Duration,
}

impl SchedulerManager {
    pub fn new(
        pool: SqlitePool,
        sender: broadcast::Sender<Event>,
        tick_interval_secs: u64,
    ) -> Self {
        Self {
            pool,
            sender,
            tick_interval: Duration::from_secs(tick_interval_secs),
        }
    }

    /// Seed scheduled tasks from config into the database.
    /// Uses upsert by name+source='config'. Removes stale config entries.
    pub async fn seed_from_config(&self, tasks: &[ScheduledTaskConfig]) {
        let now = Utc::now();

        for task in tasks {
            let cron_expr = match parse_schedule(&task.schedule) {
                Ok(expr) => expr,
                Err(e) => {
                    error!(
                        name = %task.name,
                        schedule = %task.schedule,
                        "Failed to parse config schedule: {}",
                        e
                    );
                    continue;
                }
            };

            let next_run = match compute_next_run(&cron_expr) {
                Ok(dt) => dt,
                Err(e) => {
                    error!(name = %task.name, "Failed to compute next run: {}", e);
                    continue;
                }
            };

            let id = uuid::Uuid::new_v4().to_string();
            let now_str = now.to_rfc3339();
            let next_str = next_run.to_rfc3339();

            let result = sqlx::query(
                "INSERT INTO scheduled_tasks (id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused, is_trusted, next_run_at, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, 'config', ?, 0, ?, ?, ?, ?)
                 ON CONFLICT(name) WHERE source='config' DO UPDATE SET
                   cron_expr = excluded.cron_expr,
                   original_schedule = excluded.original_schedule,
                   prompt = excluded.prompt,
                   is_oneshot = excluded.is_oneshot,
                   is_trusted = excluded.is_trusted,
                   updated_at = excluded.updated_at"
            )
            .bind(&id)
            .bind(&task.name)
            .bind(&cron_expr)
            .bind(&task.schedule)
            .bind(&task.prompt)
            .bind(task.oneshot as i32)
            .bind(task.trusted as i32)
            .bind(&next_str)
            .bind(&now_str)
            .bind(&now_str)
            .execute(&self.pool)
            .await;

            match result {
                Ok(_) => info!(name = %task.name, cron = %cron_expr, "Seeded config schedule"),
                Err(e) => error!(name = %task.name, "Failed to seed config schedule: {}", e),
            }
        }

        // Remove config-sourced entries that are no longer in config
        let config_names: Vec<&str> = tasks.iter().map(|t| t.name.as_str()).collect();
        if config_names.is_empty() {
            // Delete all config-sourced tasks
            let _ = sqlx::query("DELETE FROM scheduled_tasks WHERE source = 'config'")
                .execute(&self.pool)
                .await;
        } else {
            // Build placeholders for the IN clause
            let placeholders: Vec<String> = config_names.iter().map(|_| "?".to_string()).collect();
            let query_str = format!(
                "DELETE FROM scheduled_tasks WHERE source = 'config' AND name NOT IN ({})",
                placeholders.join(", ")
            );
            let mut query = sqlx::query(&query_str);
            for name in &config_names {
                query = query.bind(name);
            }
            let _ = query.execute(&self.pool).await;
        }
    }

    /// Spawn the scheduler tick loop as a background task.
    pub fn spawn(self: Arc<Self>) {
        tokio::spawn(async move {
            // Recover missed tasks on startup
            if let Err(e) = self.recover_missed().await {
                error!("Scheduler crash recovery failed: {}", e);
            }

            loop {
                tokio::time::sleep(self.tick_interval).await;
                if let Err(e) = self.tick().await {
                    error!("Scheduler tick error: {}", e);
                }
            }
        });

        info!("Scheduler manager spawned");
    }

    /// Check for due tasks and fire them.
    async fn tick(&self) -> anyhow::Result<()> {
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        let rows = sqlx::query(
            "SELECT id, name, cron_expr, prompt, is_oneshot, is_trusted
             FROM scheduled_tasks
             WHERE next_run_at <= ? AND is_paused = 0"
        )
        .bind(&now_str)
        .fetch_all(&self.pool)
        .await?;

        for row in rows {
            let id: String = row.get("id");
            let name: String = row.get("name");
            let cron_expr: String = row.get("cron_expr");
            let prompt: String = row.get("prompt");
            let is_oneshot: bool = row.get::<i32, _>("is_oneshot") != 0;
            let is_trusted: bool = row.get::<i32, _>("is_trusted") != 0;

            // Build session_id based on trust level
            let session_id = if is_trusted {
                format!("scheduled_{}", id)
            } else {
                format!("scheduler_trigger_{}", id)
            };

            let event = Event {
                source: "scheduler".to_string(),
                session_id,
                content: prompt.clone(),
            };

            if self.sender.send(event).is_err() {
                warn!(name = %name, "No event receivers active for scheduled task");
            } else {
                info!(name = %name, "Fired scheduled task");
            }

            if is_oneshot {
                sqlx::query("DELETE FROM scheduled_tasks WHERE id = ?")
                    .bind(&id)
                    .execute(&self.pool)
                    .await?;
            } else {
                // Recompute next run
                match compute_next_run(&cron_expr) {
                    Ok(next) => {
                        let next_str = next.to_rfc3339();
                        sqlx::query(
                            "UPDATE scheduled_tasks SET last_run_at = ?, next_run_at = ?, updated_at = ? WHERE id = ?"
                        )
                        .bind(&now_str)
                        .bind(&next_str)
                        .bind(&now_str)
                        .bind(&id)
                        .execute(&self.pool)
                        .await?;
                    }
                    Err(e) => {
                        error!(name = %name, "Failed to compute next run, pausing task: {}", e);
                        sqlx::query("UPDATE scheduled_tasks SET is_paused = 1, updated_at = ? WHERE id = ?")
                            .bind(&now_str)
                            .bind(&id)
                            .execute(&self.pool)
                            .await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// On startup, fire tasks that were missed while the daemon was down.
    async fn recover_missed(&self) -> anyhow::Result<()> {
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        let rows = sqlx::query(
            "SELECT id, name, cron_expr, prompt, is_oneshot, is_trusted
             FROM scheduled_tasks
             WHERE next_run_at < ? AND is_paused = 0"
        )
        .bind(&now_str)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(());
        }

        info!(count = rows.len(), "Recovering missed scheduled tasks");

        for row in rows {
            let id: String = row.get("id");
            let name: String = row.get("name");
            let cron_expr: String = row.get("cron_expr");
            let prompt: String = row.get("prompt");
            let is_oneshot: bool = row.get::<i32, _>("is_oneshot") != 0;
            let is_trusted: bool = row.get::<i32, _>("is_trusted") != 0;

            let session_id = if is_trusted {
                format!("scheduled_{}", id)
            } else {
                format!("scheduler_trigger_{}", id)
            };

            let event = Event {
                source: "scheduler".to_string(),
                session_id,
                content: prompt.clone(),
            };

            if self.sender.send(event).is_err() {
                warn!(name = %name, "No event receivers for missed task recovery");
            } else {
                info!(name = %name, "Recovered missed scheduled task");
            }

            if is_oneshot {
                sqlx::query("DELETE FROM scheduled_tasks WHERE id = ?")
                    .bind(&id)
                    .execute(&self.pool)
                    .await?;
            } else {
                match compute_next_run(&cron_expr) {
                    Ok(next) => {
                        let next_str = next.to_rfc3339();
                        sqlx::query(
                            "UPDATE scheduled_tasks SET last_run_at = ?, next_run_at = ?, updated_at = ? WHERE id = ?"
                        )
                        .bind(&now_str)
                        .bind(&next_str)
                        .bind(&now_str)
                        .bind(&id)
                        .execute(&self.pool)
                        .await?;
                    }
                    Err(e) => {
                        error!(name = %name, "Failed to recompute next run after recovery: {}", e);
                        sqlx::query("UPDATE scheduled_tasks SET is_paused = 1, updated_at = ? WHERE id = ?")
                            .bind(&now_str)
                            .bind(&id)
                            .execute(&self.pool)
                            .await?;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Parse a human-friendly schedule string into a 5-field cron expression.
/// Supports natural shortcuts and raw cron pass-through.
pub fn parse_schedule(input: &str) -> anyhow::Result<String> {
    let input = input.trim();

    // Simple keyword shortcuts
    match input.to_lowercase().as_str() {
        "hourly" => return Ok("0 * * * *".to_string()),
        "daily" => return Ok("0 0 * * *".to_string()),
        "weekly" => return Ok("0 0 * * 0".to_string()),
        "monthly" => return Ok("0 0 1 * *".to_string()),
        _ => {}
    }

    // "every Nm" / "every N minutes"
    let re_minutes = Regex::new(r"(?i)^every\s+(\d+)\s*(?:m|min|mins|minutes?)$")?;
    if let Some(caps) = re_minutes.captures(input) {
        let n: u32 = caps[1].parse()?;
        if n == 0 || n > 59 {
            anyhow::bail!("Minutes interval must be between 1 and 59");
        }
        return Ok(format!("*/{} * * * *", n));
    }

    // "every Nh" / "every N hours"
    let re_hours = Regex::new(r"(?i)^every\s+(\d+)\s*(?:h|hrs?|hours?)$")?;
    if let Some(caps) = re_hours.captures(input) {
        let n: u32 = caps[1].parse()?;
        if n == 0 || n > 23 {
            anyhow::bail!("Hours interval must be between 1 and 23");
        }
        return Ok(format!("0 */{} * * *", n));
    }

    // "daily at 9am" / "daily at 14:30" / "daily at 2pm" / "daily at 2:30pm"
    let re_daily = Regex::new(r"(?i)^daily\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")?;
    if let Some(caps) = re_daily.captures(input) {
        let (hour, minute) = parse_time_captures(&caps)?;
        return Ok(format!("{} {} * * *", minute, hour));
    }

    // "weekdays at 8:30" / "weekdays at 9am"
    let re_weekdays = Regex::new(r"(?i)^weekdays?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")?;
    if let Some(caps) = re_weekdays.captures(input) {
        let (hour, minute) = parse_time_captures(&caps)?;
        return Ok(format!("{} {} * * 1-5", minute, hour));
    }

    // "weekends at 10am"
    let re_weekends = Regex::new(r"(?i)^weekends?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")?;
    if let Some(caps) = re_weekends.captures(input) {
        let (hour, minute) = parse_time_captures(&caps)?;
        return Ok(format!("{} {} * * 0,6", minute, hour));
    }

    // "in 2h" / "in 30m" / "in 90 minutes" — compute a one-shot absolute cron
    let re_in_time = Regex::new(r"(?i)^in\s+(\d+)\s*(m|min|mins|minutes?|h|hrs?|hours?)$")?;
    if let Some(caps) = re_in_time.captures(input) {
        let n: i64 = caps[1].parse()?;
        let unit = caps[2].to_lowercase();
        let duration = if unit.starts_with('h') {
            chrono::Duration::hours(n)
        } else {
            chrono::Duration::minutes(n)
        };
        let target = Utc::now() + duration;
        return Ok(format!(
            "{} {} {} {} *",
            target.minute(),
            target.hour(),
            target.day(),
            target.month()
        ));
    }

    // Raw cron pass-through: validate with croner
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.len() == 5 {
        // Try to parse as cron
        input.parse::<Cron>()
            .map_err(|e| anyhow::anyhow!("Invalid cron expression '{}': {}", input, e))?;
        return Ok(input.to_string());
    }

    anyhow::bail!(
        "Unrecognized schedule format '{}'. Use natural shortcuts (e.g. 'daily at 9am', 'every 5m', 'in 2h') or a 5-field cron expression.",
        input
    )
}

/// Extract hour and minute from regex captures with optional AM/PM.
fn parse_time_captures(caps: &regex::Captures) -> anyhow::Result<(u32, u32)> {
    let mut hour: u32 = caps[1].parse()?;
    let minute: u32 = caps.get(2).map_or(Ok(0), |m| m.as_str().parse())?;
    if let Some(ampm) = caps.get(3) {
        let ampm = ampm.as_str().to_lowercase();
        if ampm == "pm" && hour < 12 {
            hour += 12;
        } else if ampm == "am" && hour == 12 {
            hour = 0;
        }
    }
    if hour > 23 {
        anyhow::bail!("Hour must be between 0 and 23");
    }
    if minute > 59 {
        anyhow::bail!("Minute must be between 0 and 59");
    }
    Ok((hour, minute))
}

/// Compute the next occurrence from a cron expression using croner.
pub fn compute_next_run(cron_expr: &str) -> anyhow::Result<DateTime<Utc>> {
    let cron: Cron = cron_expr
        .parse()
        .map_err(|e| anyhow::anyhow!("Failed to parse cron '{}': {}", cron_expr, e))?;

    cron.find_next_occurrence(&Utc::now(), false)
        .map_err(|e| anyhow::anyhow!("No next occurrence for '{}': {}", cron_expr, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_schedule_keywords() {
        assert_eq!(parse_schedule("hourly").unwrap(), "0 * * * *");
        assert_eq!(parse_schedule("daily").unwrap(), "0 0 * * *");
        assert_eq!(parse_schedule("weekly").unwrap(), "0 0 * * 0");
        assert_eq!(parse_schedule("monthly").unwrap(), "0 0 1 * *");
    }

    #[test]
    fn test_parse_schedule_every_minutes() {
        assert_eq!(parse_schedule("every 5m").unwrap(), "*/5 * * * *");
        assert_eq!(parse_schedule("every 15 minutes").unwrap(), "*/15 * * * *");
        assert_eq!(parse_schedule("every 1 min").unwrap(), "*/1 * * * *");
    }

    #[test]
    fn test_parse_schedule_every_hours() {
        assert_eq!(parse_schedule("every 2h").unwrap(), "0 */2 * * *");
        assert_eq!(parse_schedule("every 4 hours").unwrap(), "0 */4 * * *");
    }

    #[test]
    fn test_parse_schedule_daily_at() {
        assert_eq!(parse_schedule("daily at 9am").unwrap(), "0 9 * * *");
        assert_eq!(parse_schedule("daily at 14:30").unwrap(), "30 14 * * *");
        assert_eq!(parse_schedule("daily at 2pm").unwrap(), "0 14 * * *");
        assert_eq!(parse_schedule("daily at 2:30pm").unwrap(), "30 14 * * *");
        assert_eq!(parse_schedule("daily at 12am").unwrap(), "0 0 * * *");
    }

    #[test]
    fn test_parse_schedule_weekdays() {
        assert_eq!(parse_schedule("weekdays at 8:30").unwrap(), "30 8 * * 1-5");
        assert_eq!(parse_schedule("weekdays at 9am").unwrap(), "0 9 * * 1-5");
    }

    #[test]
    fn test_parse_schedule_weekends() {
        assert_eq!(parse_schedule("weekends at 10am").unwrap(), "0 10 * * 0,6");
    }

    #[test]
    fn test_parse_schedule_cron_passthrough() {
        assert_eq!(parse_schedule("0 9 * * 1-5").unwrap(), "0 9 * * 1-5");
        assert_eq!(parse_schedule("*/5 * * * *").unwrap(), "*/5 * * * *");
    }

    #[test]
    fn test_parse_schedule_invalid() {
        assert!(parse_schedule("never").is_err());
        assert!(parse_schedule("every 0m").is_err());
        assert!(parse_schedule("daily at 25:00").is_err());
    }

    #[test]
    fn test_compute_next_run() {
        let next = compute_next_run("* * * * *").unwrap();
        assert!(next > Utc::now());
    }
}
