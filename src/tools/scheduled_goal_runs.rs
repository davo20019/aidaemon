use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{Goal, StateStore, Task, TaskActivity, Tool, ToolCapabilities};

pub struct ScheduledGoalRunsTool {
    state: Arc<dyn StateStore>,
}

impl ScheduledGoalRunsTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self { state }
    }

    async fn set_budget(
        &self,
        goal_id_input: &str,
        budget_per_check: Option<i64>,
        budget_daily: Option<i64>,
    ) -> anyhow::Result<String> {
        const MAX_BUDGET: i64 = 2_000_000;

        if budget_per_check.is_none() && budget_daily.is_none() {
            return Ok("Provide budget_per_check and/or budget_daily.".to_string());
        }
        if let Some(v) = budget_per_check {
            if v < 0 {
                return Ok("budget_per_check must be >= 0.".to_string());
            }
            if v > MAX_BUDGET {
                return Ok(format!(
                    "budget_per_check is too large (max {}).",
                    MAX_BUDGET
                ));
            }
        }
        if let Some(v) = budget_daily {
            if v < 0 {
                return Ok("budget_daily must be >= 0.".to_string());
            }
            if v > MAX_BUDGET {
                return Ok(format!("budget_daily is too large (max {}).", MAX_BUDGET));
            }
        }

        let resolved_goal_id = match self.resolve_goal_id(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };

        let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
        if schedules.is_empty() {
            return Ok("Only scheduled goals can be updated with scheduled_goal_runs.".to_string());
        }

        let old_per_check = goal.budget_per_check;
        let old_daily = goal.budget_daily;

        if let Some(v) = budget_per_check {
            goal.budget_per_check = Some(v);
        }
        if let Some(v) = budget_daily {
            goal.budget_daily = Some(v);
        }

        if let (Some(per_check), Some(daily)) = (goal.budget_per_check, goal.budget_daily) {
            if per_check > daily {
                return Ok(format!(
                    "Invalid budgets: budget_per_check ({}) cannot exceed budget_daily ({}).",
                    per_check, daily
                ));
            }
        }

        self.state
            .set_goal_budgets(&goal.id, budget_per_check, budget_daily)
            .await?;

        let mut out = format!(
            "Updated budget for scheduled goal {}.\n- budget_per_check: {:?} -> {:?}\n- budget_daily: {:?} -> {:?}",
            goal.id, old_per_check, goal.budget_per_check, old_daily, goal.budget_daily
        );
        if let Some(budget_daily) = goal.budget_daily {
            if goal.tokens_used_today >= budget_daily {
                out.push_str(&format!(
                    "\nNote: tokens_used_today={} already exceeds the new budget_daily={}, so runs may stop until the daily reset.",
                    goal.tokens_used_today, budget_daily
                ));
            }
        }
        Ok(out)
    }

    async fn resolve_goal_id(&self, input_id: &str) -> anyhow::Result<String> {
        let trimmed = input_id.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Empty goal ID");
        }

        if self.state.get_goal(trimmed).await?.is_some() {
            return Ok(trimmed.to_string());
        }

        let goals = self.state.get_scheduled_goals().await?;
        let mut matches: Vec<&Goal> = goals.iter().filter(|g| g.id.starts_with(trimmed)).collect();

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
            "Goal ID prefix '{}' is ambiguous ({} matches): {}. Use full goal_id.",
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

    fn latest_problem_task(tasks: &[Task]) -> Option<&Task> {
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

    async fn run_now(
        &self,
        goal_id_input: &str,
        schedule_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let resolved_goal_id = match self.resolve_goal_id(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };

        let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };

        let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
        if schedules.is_empty() {
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

        let existing_tasks = self.state.get_tasks_for_goal(&goal.id).await?;
        let open: Vec<&Task> = existing_tasks
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
        let task = Task {
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

        let mut schedule_consumed = false;
        if let Some(sid) = schedule_id {
            let Some(s) = self.state.get_goal_schedule(sid).await? else {
                return Ok(format!("Schedule not found: {}", sid));
            };
            if s.goal_id != goal.id {
                return Ok(format!(
                    "Schedule {} does not belong to goal {}.",
                    sid, goal.id
                ));
            }
            if s.is_one_shot {
                let _ = self.state.delete_goal_schedule(&s.id).await;
                schedule_consumed = true;
            }
        } else if schedules.len() == 1 && schedules[0].is_one_shot {
            let _ = self.state.delete_goal_schedule(&schedules[0].id).await;
            schedule_consumed = true;
        }

        goal.last_useful_action = Some(now.clone());
        goal.updated_at = now;
        self.state.update_goal(&goal).await?;
        self.state.create_task(&task).await?;

        let mut out = format!(
            "Triggered manual run for scheduled goal {}.\n- Created task: {}\n- Goal status: {}",
            resolved_goal_id, task.id, goal.status
        );
        if schedule_consumed {
            out.push_str("\n- One-shot schedule consumed: schedule deleted.");
        }
        Ok(out)
    }

    async fn run_history(&self, goal_id_input: &str, limit: usize) -> anyhow::Result<String> {
        let resolved_goal_id = match self.resolve_goal_id(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(goal) = self.state.get_goal(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };

        let mut tasks = self.state.get_tasks_for_goal(&goal.id).await?;
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
                .get_task_activities(&t.id)
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
        let resolved_goal_id = match self.resolve_goal_id(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(goal) = self.state.get_goal(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };
        let tasks = self.state.get_tasks_for_goal(&goal.id).await?;
        let Some(task) = Self::latest_problem_task(&tasks) else {
            return Ok(format!(
                "No failed/blocked runs found for scheduled goal {}.",
                goal.id
            ));
        };

        let activities = self.state.get_task_activities(&task.id).await?;
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
        let resolved_goal_id = match self.resolve_goal_id(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(goal) = self.state.get_goal(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
        };
        let tasks = self.state.get_tasks_for_goal(&goal.id).await?;
        let Some(problem_task) = Self::latest_problem_task(&tasks) else {
            return Ok(format!(
                "No failed/blocked runs found for {}. No unblock hints needed.",
                goal.id
            ));
        };

        let activities: Vec<TaskActivity> = self
            .state
            .get_task_activities(&problem_task.id)
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
            "- Retry immediately: scheduled_goal_runs(action='run_now', goal_id='{}')\n",
            goal.id
        ));
        out.push_str(&format!(
            "- Inspect full timeline: goal_trace(action='goal_trace', goal_id='{}')\n",
            goal.id
        ));
        Ok(out)
    }
}

#[derive(Deserialize)]
struct ScheduledGoalRunsArgs {
    action: String,
    #[serde(default)]
    goal_id: Option<String>,
    #[serde(default)]
    schedule_id: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    budget_per_check: Option<i64>,
    #[serde(default)]
    budget_daily: Option<i64>,
    #[serde(default)]
    _user_role: Option<String>,
}

#[async_trait]
impl Tool for ScheduledGoalRunsTool {
    fn name(&self) -> &str {
        "scheduled_goal_runs"
    }

    fn description(&self) -> &str {
        "Run scheduled goals now and inspect run history/failures without terminal/sqlite access (not for storing facts)"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "scheduled_goal_runs",
            "description": "Run scheduled goals now and inspect execution diagnostics. Use this instead of terminal/sqlite for scheduled-goal run forensics. ONLY for recurring/scheduled goals; NOT for learning or storing facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run_now", "run_history", "last_failure", "unblock_hints", "set_budget"],
                        "description": "Action to perform. All actions require goal_id."
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Scheduled goal ID (full or unique prefix)"
                    },
                    "schedule_id": {
                        "type": "string",
                        "description": "Optional schedule ID. For one-shot schedules, run_now can consume a specific schedule when provided."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max runs to show for run_history (default 10, max 50)"
                    },
                    "budget_per_check": {
                        "type": "integer",
                        "description": "New per-check budget (tokens). Only used for set_budget."
                    },
                    "budget_daily": {
                        "type": "integer",
                        "description": "New daily budget (tokens). Only used for set_budget."
                    }
                },
                "required": ["action", "goal_id"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ScheduledGoalRunsArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
	            "run_now" => {
	                let goal_id = args
	                    .goal_id
	                    .as_deref()
	                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for run_now"))?;
	                self.run_now(goal_id, args.schedule_id.as_deref()).await
	            }
	            "run_history" => {
	                let goal_id = args
	                    .goal_id
	                    .as_deref()
	                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for run_history"))?;
	                self.run_history(goal_id, args.limit.unwrap_or(10)).await
	            }
	            "last_failure" => {
	                let goal_id = args
	                    .goal_id
	                    .as_deref()
	                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for last_failure"))?;
	                self.last_failure(goal_id).await
	            }
	            "unblock_hints" => {
	                let goal_id = args
	                    .goal_id
	                    .as_deref()
	                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for unblock_hints"))?;
	                self.unblock_hints(goal_id).await
	            }
	            "set_budget" => {
	                let is_owner = args
	                    ._user_role
	                    .as_deref()
	                    .is_some_and(|r| r.eq_ignore_ascii_case("owner"));
	                if !is_owner {
	                    return Ok("Only owners can change scheduled goal budgets.".to_string());
	                }
	                let goal_id = args
	                    .goal_id
	                    .as_deref()
	                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for set_budget"))?;
	                self.set_budget(goal_id, args.budget_per_check, args.budget_daily)
	                    .await
	            }
	            other => Ok(format!(
	                "Unknown action: '{}'. Use run_now, run_history, last_failure, unblock_hints, or set_budget.",
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
    use crate::traits::{Goal, GoalSchedule, Task};

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
    async fn schema_marks_goal_id_required() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state);
        let schema = tool.schema();
        let required = schema
            .get("parameters")
            .and_then(|p| p.get("required"))
            .and_then(|r| r.as_array())
            .expect("required array exists");
        let required_values: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required_values.contains(&"action"));
        assert!(required_values.contains(&"goal_id"));
    }

    #[tokio::test]
    async fn run_now_creates_pending_task() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "run_now",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Triggered manual run"));

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status, "pending");
    }

    #[tokio::test]
    async fn unblock_hints_reports_timeout_guidance() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Knowledge base maintenance",
            "system",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let task = Task {
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
        state.create_task(&task).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "unblock_hints",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Unblock Hints"));
        assert!(result.contains("transient service/network"));
    }

    #[tokio::test]
    async fn set_budget_updates_scheduled_goal() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_per_check": 1234,
                    "budget_daily": 5678,
                    "_user_role": "Owner"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Updated budget"));

        let updated = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(updated.budget_per_check, Some(1234));
        assert_eq!(updated.budget_daily, Some(5678));
    }

    #[tokio::test]
    async fn set_budget_rejects_non_owner() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_daily": 9999,
                    "_user_role": "Guest"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Only owners"));
    }

    #[tokio::test]
    async fn set_budget_validates_values() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 */6 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 */6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("every 6h".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let negative = tool
            .call(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_daily": -1,
                    "_user_role": "Owner"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(negative.contains("budget_daily must be >="));

        let too_large = tool
            .call(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_daily": 2000001,
                    "_user_role": "Owner"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(too_large.contains("max"));

        let invalid_relation = tool
            .call(
                &json!({
                    "action": "set_budget",
                    "goal_id": goal_id,
                    "budget_per_check": 200,
                    "budget_daily": 100,
                    "_user_role": "Owner"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(invalid_relation.contains("cannot exceed"));
    }
}
