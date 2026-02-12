use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::cron_utils::is_one_shot_schedule;
use crate::traits::{GoalV3, StateStore, TaskActivityV3, TaskV3, Tool};

pub struct ScheduledGoalRunsTool {
    state: Arc<dyn StateStore>,
}

impl ScheduledGoalRunsTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self { state }
    }

    async fn resolve_goal_id_v3(&self, input_id: &str) -> anyhow::Result<String> {
        let trimmed = input_id.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Empty goal ID");
        }

        if self.state.get_goal_v3(trimmed).await?.is_some() {
            return Ok(trimmed.to_string());
        }

        let goals = self.state.get_scheduled_goals_v3().await?;
        let mut matches: Vec<&GoalV3> =
            goals.iter().filter(|g| g.id.starts_with(trimmed)).collect();

        if matches.is_empty() {
            anyhow::bail!("Scheduled goal not found: {}", trimmed);
        }
        if matches.len() == 1 {
            return Ok(matches.remove(0).id.clone());
        }

        matches.sort_by_key(|g| match g.status.as_str() {
            "active" => 0usize,
            "failed" => 1,
            "paused" => 2,
            "pending_confirmation" => 3,
            "cancelled" => 4,
            "completed" => 5,
            _ => 6,
        });

        let preview = matches
            .iter()
            .take(5)
            .map(|g| {
                let short = Self::short_id(&g.id);
                format!(
                    "{} ({}, {})",
                    short,
                    g.status,
                    Self::truncate(&g.description, 40)
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::bail!(
            "Goal ID prefix '{}' is ambiguous ({} matches): {}. Use full goal_id_v3.",
            trimmed,
            matches.len(),
            preview
        );
    }

    fn short_id(id: &str) -> String {
        id.chars().take(8).collect()
    }

    fn truncate(s: &str, max: usize) -> String {
        s.chars().take(max).collect()
    }

    fn parse_ts(ts: &str) -> Option<chrono::DateTime<chrono::Utc>> {
        chrono::DateTime::parse_from_rfc3339(ts)
            .ok()
            .map(|d| d.with_timezone(&chrono::Utc))
    }

    fn format_duration(started_at: Option<&str>, completed_at: Option<&str>) -> String {
        let Some(start_raw) = started_at else {
            return "n/a".to_string();
        };
        let Some(started) = Self::parse_ts(start_raw) else {
            return "n/a".to_string();
        };
        let Some(end_raw) = completed_at else {
            return "running".to_string();
        };
        let Some(ended) = Self::parse_ts(end_raw) else {
            return "n/a".to_string();
        };

        let secs = (ended - started).num_seconds().max(0);
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            let h = secs / 3600;
            let m = (secs % 3600) / 60;
            format!("{}h {}m", h, m)
        }
    }

    fn latest_problem_task(tasks: &[TaskV3]) -> Option<&TaskV3> {
        let latest_failed = tasks
            .iter()
            .filter(|t| t.status == "failed")
            .max_by_key(|t| {
                t.completed_at
                    .as_deref()
                    .or(t.started_at.as_deref())
                    .unwrap_or(&t.created_at)
            });
        if latest_failed.is_some() {
            return latest_failed;
        }

        tasks
            .iter()
            .filter(|t| t.status == "blocked")
            .max_by_key(|t| {
                t.completed_at
                    .as_deref()
                    .or(t.started_at.as_deref())
                    .unwrap_or(&t.created_at)
            })
    }

    fn infer_hints(problem_text: &str, has_blocked: bool) -> Vec<&'static str> {
        let mut hints = Vec::new();
        let text = problem_text.to_ascii_lowercase();

        if text.contains("timeout")
            || text.contains("timed out")
            || text.contains("deadline exceeded")
            || text.contains("connection reset")
            || text.contains("temporarily unavailable")
        {
            hints.push(
                "Likely transient service/network failure. Retry now, then reduce schedule frequency if it repeats.",
            );
        }
        if text.contains("429") || text.contains("rate limit") || text.contains("too many requests")
        {
            hints.push("Rate limited. Increase interval/backoff and reduce parallel API calls.");
        }
        if text.contains("401")
            || text.contains("403")
            || text.contains("unauthorized")
            || text.contains("forbidden")
            || text.contains("token")
            || text.contains("oauth")
            || text.contains("permission denied")
        {
            hints.push("Auth/permission issue. Reconnect credentials (manage_oauth/manage_config) and retry.");
        }
        if text.contains("404")
            || text.contains("not found")
            || text.contains("no such file")
            || text.contains("does not exist")
        {
            hints.push("Target missing/renamed. Re-validate resource IDs, URLs, and file paths.");
        }
        if text.contains("json")
            || text.contains("parse")
            || text.contains("schema")
            || text.contains("invalid format")
        {
            hints.push(
                "Data contract mismatch. Validate request payload/response parsing assumptions.",
            );
        }
        if text.contains("cap")
            || text.contains("budget")
            || text.contains("tokens_used_today")
            || text.contains("active evergreen goals reached")
        {
            hints.push(
                "Capacity/budget cap hit. Pause lower-priority recurring goals or raise limits.",
            );
        }
        if has_blocked || text.contains("dependency") || text.contains("blocked") {
            hints.push(
                "There are blocked dependencies. Resolve blocker tasks first, then retry the failed task.",
            );
        }

        if hints.is_empty() {
            hints.push(
                "No obvious failure signature found. Use goal_trace(tool_trace) to inspect exact tool-call errors.",
            );
        }
        hints
    }

    async fn run_now(&self, goal_id_input: &str) -> anyhow::Result<String> {
        let resolved_goal_id = match self.resolve_goal_id_v3(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };

        let Some(mut goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };
        if goal.schedule.is_none() {
            return Ok("Only scheduled goals can be run with scheduled_goal_runs.".to_string());
        }

        match goal.status.as_str() {
            "cancelled" | "completed" | "pending_confirmation" => {
                return Ok(format!(
                    "Cannot run goal {} in status '{}'.",
                    resolved_goal_id, goal.status
                ));
            }
            "paused" => {
                return Ok(format!(
                    "Goal {} is paused. Resume it first, then run_now.",
                    resolved_goal_id
                ));
            }
            _ => {}
        }

        let existing_tasks = self.state.get_tasks_for_goal_v3(&goal.id).await?;
        let open: Vec<&TaskV3> = existing_tasks
            .iter()
            .filter(|t| matches!(t.status.as_str(), "pending" | "claimed" | "running"))
            .collect();
        if !open.is_empty() {
            let preview = open
                .iter()
                .take(5)
                .map(|t| format!("{} ({})", Self::short_id(&t.id), t.status))
                .collect::<Vec<_>>()
                .join(", ");
            return Ok(format!(
                "Skipped run_now for {}: goal already has {} open task(s): {}.",
                resolved_goal_id,
                open.len(),
                preview
            ));
        }

        let now = chrono::Utc::now().to_rfc3339();
        let task = TaskV3 {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: format!("Manual scheduled run: {}", goal.description),
            status: "pending".to_string(),
            priority: if goal.goal_type == "continuous" {
                "low".to_string()
            } else {
                "medium".to_string()
            },
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: goal.context.clone(),
            result: None,
            error: None,
            blocker: None,
            idempotent: goal.goal_type == "continuous",
            retry_count: 0,
            max_retries: 1,
            created_at: now.clone(),
            started_at: None,
            completed_at: None,
        };

        if goal.status == "failed" {
            goal.status = "active".to_string();
            goal.completed_at = None;
        }
        let schedule_was_cleared = if goal.goal_type == "finite"
            && goal
                .schedule
                .as_ref()
                .is_some_and(|s| is_one_shot_schedule(s))
        {
            goal.schedule = None;
            true
        } else {
            false
        };
        goal.last_useful_action = Some(now.clone());
        goal.updated_at = now;
        self.state.update_goal_v3(&goal).await?;
        self.state.create_task_v3(&task).await?;

        let mut out = format!(
            "Triggered manual run for scheduled goal {}.\n- Created task: {}\n- Goal status: {}",
            resolved_goal_id, task.id, goal.status
        );
        if schedule_was_cleared {
            out.push_str("\n- One-shot schedule consumed: schedule cleared.");
        }
        Ok(out)
    }

    async fn run_history(&self, goal_id_input: &str, limit: usize) -> anyhow::Result<String> {
        let resolved_goal_id = match self.resolve_goal_id_v3(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };
        if goal.schedule.is_none() {
            return Ok("Only scheduled goals support run history in this tool.".to_string());
        }

        let mut tasks = self.state.get_tasks_for_goal_v3(&goal.id).await?;
        if tasks.is_empty() {
            return Ok(format!("No runs found yet for scheduled goal {}.", goal.id));
        }
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        let cap = limit.clamp(1, 50);
        tasks.truncate(cap);

        let mut failed = 0usize;
        let mut completed = 0usize;
        let mut pending = 0usize;
        let mut running = 0usize;
        let mut blocked = 0usize;
        for t in &tasks {
            match t.status.as_str() {
                "failed" => failed += 1,
                "completed" => completed += 1,
                "pending" => pending += 1,
                "running" | "claimed" => running += 1,
                "blocked" => blocked += 1,
                _ => {}
            }
        }

        let mut out = format!(
            "**Scheduled Run History**\n\n- Goal: {}\n- ID: {}\n- Showing: {} run(s)\n- Status mix: completed {}, failed {}, running {}, pending {}, blocked {}\n",
            goal.description,
            goal.id,
            tasks.len(),
            completed,
            failed,
            running,
            pending,
            blocked
        );

        for t in &tasks {
            let activities = self
                .state
                .get_task_activities_v3(&t.id)
                .await
                .unwrap_or_default();
            let last_activity = activities.last();
            let last_tool = last_activity
                .and_then(|a| a.tool_name.as_deref())
                .unwrap_or("-");
            let last_ok = last_activity
                .and_then(|a| a.success)
                .map(|s| if s { "ok" } else { "err" })
                .unwrap_or("n/a");

            out.push_str(&format!(
                "\n- **{}** status={} retry={}/{} created={} duration={} last_tool={}({})",
                t.id,
                t.status,
                t.retry_count,
                t.max_retries,
                t.created_at,
                Self::format_duration(t.started_at.as_deref(), t.completed_at.as_deref()),
                last_tool,
                last_ok
            ));
            if t.status == "failed" {
                if let Some(err) = &t.error {
                    out.push_str(&format!("\n  error: {}", Self::truncate(err, 160)));
                }
            } else if t.status == "blocked" {
                if let Some(blocker) = &t.blocker {
                    out.push_str(&format!("\n  blocker: {}", Self::truncate(blocker, 160)));
                }
            }
        }
        Ok(out)
    }

    async fn last_failure(&self, goal_id_input: &str) -> anyhow::Result<String> {
        let resolved_goal_id = match self.resolve_goal_id_v3(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };
        let tasks = self.state.get_tasks_for_goal_v3(&goal.id).await?;
        let Some(task) = Self::latest_problem_task(&tasks) else {
            return Ok(format!(
                "No failed/blocked runs found for scheduled goal {}.",
                goal.id
            ));
        };

        let activities = self.state.get_task_activities_v3(&task.id).await?;
        let mut out = format!(
            "**Last Failure**\n\n- Goal: {}\n- Goal ID: {}\n- Task ID: {}\n- Task status: {}\n- Retry: {}/{}\n- Created: {}\n- Duration: {}",
            goal.description,
            goal.id,
            task.id,
            task.status,
            task.retry_count,
            task.max_retries,
            task.created_at,
            Self::format_duration(task.started_at.as_deref(), task.completed_at.as_deref())
        );

        if let Some(err) = &task.error {
            out.push_str(&format!("\n- Error: {}", Self::truncate(err, 300)));
        }
        if let Some(blocker) = &task.blocker {
            out.push_str(&format!("\n- Blocker: {}", Self::truncate(blocker, 300)));
        }

        if !activities.is_empty() {
            out.push_str("\n\n**Recent Activity**");
            for a in activities.iter().rev().take(5).rev() {
                let tool = a.tool_name.as_deref().unwrap_or("-");
                let ok = a
                    .success
                    .map(|v| if v { "ok" } else { "err" })
                    .unwrap_or("n/a");
                let result = a
                    .result
                    .as_deref()
                    .map(|r| Self::truncate(r, 140))
                    .unwrap_or_default();
                if result.is_empty() {
                    out.push_str(&format!(
                        "\n- {} [{}] tool={} at {}",
                        a.activity_type, ok, tool, a.created_at
                    ));
                } else {
                    out.push_str(&format!(
                        "\n- {} [{}] tool={} at {} => {}",
                        a.activity_type, ok, tool, a.created_at, result
                    ));
                }
            }
        }

        Ok(out)
    }

    async fn unblock_hints(&self, goal_id_input: &str) -> anyhow::Result<String> {
        let resolved_goal_id = match self.resolve_goal_id_v3(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };
        let tasks = self.state.get_tasks_for_goal_v3(&goal.id).await?;
        let Some(problem_task) = Self::latest_problem_task(&tasks) else {
            return Ok(format!(
                "No failed/blocked runs found for {}. No unblock hints needed.",
                goal.id
            ));
        };

        let activities: Vec<TaskActivityV3> = self
            .state
            .get_task_activities_v3(&problem_task.id)
            .await
            .unwrap_or_default();
        let has_blocked = tasks.iter().any(|t| t.status == "blocked");

        let mut problem_text = String::new();
        if let Some(err) = &problem_task.error {
            problem_text.push_str(err);
            problem_text.push('\n');
        }
        if let Some(blocker) = &problem_task.blocker {
            problem_text.push_str(blocker);
            problem_text.push('\n');
        }
        for a in activities.iter().rev().take(10) {
            if let Some(result) = &a.result {
                problem_text.push_str(result);
                problem_text.push('\n');
            }
        }

        let hints = Self::infer_hints(&problem_text, has_blocked);
        let mut out = format!(
            "**Unblock Hints**\n\n- Goal: {}\n- Goal ID: {}\n- Problem task: {} ({})\n",
            goal.description, goal.id, problem_task.id, problem_task.status
        );
        if let Some(err) = &problem_task.error {
            out.push_str(&format!("- Latest error: {}\n", Self::truncate(err, 220)));
        }
        if let Some(blocker) = &problem_task.blocker {
            out.push_str(&format!(
                "- Latest blocker: {}\n",
                Self::truncate(blocker, 220)
            ));
        }

        out.push_str("\nLikely fixes:\n");
        for hint in hints {
            out.push_str(&format!("- {}\n", hint));
        }
        out.push_str("\nNext actions:\n");
        out.push_str(&format!(
            "- Retry immediately: scheduled_goal_runs(action='run_now', goal_id_v3='{}')\n",
            goal.id
        ));
        out.push_str(&format!(
            "- Inspect full timeline: goal_trace(action='goal_trace', goal_id_v3='{}')\n",
            goal.id
        ));
        Ok(out)
    }
}

#[derive(Deserialize)]
struct ScheduledGoalRunsArgs {
    action: String,
    #[serde(default)]
    goal_id_v3: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ScheduledGoalRunsTool {
    fn name(&self) -> &str {
        "scheduled_goal_runs"
    }

    fn description(&self) -> &str {
        "Run scheduled goals now and inspect run history/failures without terminal/sqlite access"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "scheduled_goal_runs",
            "description": "Run scheduled goals now and inspect execution diagnostics. Use this instead of terminal/sqlite for scheduled-goal run forensics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run_now", "run_history", "last_failure", "unblock_hints"],
                        "description": "Action to perform"
                    },
                    "goal_id_v3": {
                        "type": "string",
                        "description": "Scheduled goal ID (full or unique prefix)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max runs to show for run_history (default 10, max 50)"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ScheduledGoalRunsArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            "run_now" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for run_now"))?;
                self.run_now(goal_id).await
            }
            "run_history" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for run_history"))?;
                self.run_history(goal_id, args.limit.unwrap_or(10)).await
            }
            "last_failure" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for last_failure"))?;
                self.last_failure(goal_id).await
            }
            "unblock_hints" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for unblock_hints"))?;
                self.unblock_hints(goal_id).await
            }
            other => Ok(format!(
                "Unknown action: '{}'. Use run_now, run_history, last_failure, or unblock_hints.",
                other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::GoalV3;

    async fn setup_state() -> Arc<dyn StateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        state as Arc<dyn StateStore>
    }

    #[tokio::test]
    async fn run_now_creates_pending_task() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = GoalV3::new_continuous(
            "Run diagnostics job",
            "user-session",
            "0 */6 * * *",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal_v3(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "run_now",
                    "goal_id_v3": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Triggered manual run"));

        let tasks = state.get_tasks_for_goal_v3(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status, "pending");
    }

    #[tokio::test]
    async fn unblock_hints_reports_timeout_guidance() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = GoalV3::new_continuous(
            "Knowledge base maintenance",
            "system",
            "0 */6 * * *",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal_v3(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = TaskV3 {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Run scheduled check".to_string(),
            status: "failed".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: Some("Timeout while contacting embeddings service".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 1,
            max_retries: 3,
            created_at: now.clone(),
            started_at: Some(now.clone()),
            completed_at: Some(now),
        };
        state.create_task_v3(&task).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "unblock_hints",
                    "goal_id_v3": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Unblock Hints"));
        assert!(result.contains("transient service/network"));
    }
}
