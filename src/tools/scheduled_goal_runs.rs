use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::traits::{
    semantics_for_exact_read_actions, Goal, StateStore, Task, TaskActivity, Tool, ToolCallMetadata,
    ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolMutationEffects,
    ToolResultPresentation,
};
use crate::types::StatusUpdate;

pub struct ScheduledGoalRunsTool {
    state: Arc<dyn StateStore>,
}

impl ScheduledGoalRunsTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self { state }
    }

    const RUN_INSTRUCTIONS_MARKER: &'static str = "\n\nLATEST RUN INSTRUCTIONS:\n";

    fn description_with_run_instructions(
        description: &str,
        instructions: &str,
        revision: u64,
        run_requirement: Option<&str>,
        insufficient_evidence: Option<&str>,
    ) -> String {
        let base = description
            .split_once(Self::RUN_INSTRUCTIONS_MARKER)
            .map_or(description, |(base, _)| base)
            .trim_end();
        let mut compiled = format!(
            "{base}{}REVISION {revision} (authoritative for this run; it supersedes any conflicting earlier execution policy while preserving the base objective and non-conflicting safety constraints).\n",
            Self::RUN_INSTRUCTIONS_MARKER
        );
        if let Some(requirement) = run_requirement {
            compiled.push_str(&format!("RUN REQUIREMENT: {requirement}\n"));
        }
        if let Some(fallback) = insufficient_evidence {
            compiled.push_str(&format!("INSUFFICIENT EVIDENCE: {fallback}\n"));
        }
        compiled.push_str(instructions);
        compiled
    }

    async fn update_instructions(
        &self,
        goal_id_input: &str,
        instructions: &str,
        run_requirement: Option<&str>,
        insufficient_evidence: Option<&str>,
    ) -> anyhow::Result<String> {
        let instructions = instructions.trim();
        if instructions.is_empty() {
            return Ok("Provide non-empty instructions.".to_string());
        }
        if instructions.chars().count() > 6000 {
            return Ok("Instructions are too long (maximum 6000 characters).".to_string());
        }
        if run_requirement == Some("must_complete") && insufficient_evidence == Some("skip_run") {
            return Ok(
                "Conflicting run policy: must_complete cannot use skip_run when evidence is insufficient. Choose use_public_sources, use_best_available, or best_effort."
                    .to_string(),
            );
        }

        let resolved_goal_id = match self.resolve_goal_id(goal_id_input).await {
            Ok(id) => id,
            Err(error) => return Ok(error.to_string()),
        };
        let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {resolved_goal_id}"));
        };
        if self
            .state
            .get_schedules_for_goal(&resolved_goal_id)
            .await?
            .is_empty()
        {
            return Ok("Only scheduled goals can have run instructions updated.".to_string());
        }

        let mut context = goal
            .context
            .as_deref()
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        let revision = context
            .get("run_instruction_spec")
            .and_then(|spec| spec.get("revision"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .saturating_add(1);
        goal.description = Self::description_with_run_instructions(
            &goal.description,
            instructions,
            revision,
            run_requirement,
            insufficient_evidence,
        );
        context["run_instructions"] = Value::String(instructions.to_string());
        context["run_instruction_spec"] = json!({
            "version": 1,
            "revision": revision,
            "instructions": instructions,
            "run_requirement": run_requirement,
            "insufficient_evidence": insufficient_evidence,
            "conflict_rule": "latest_revision_wins",
            "updated_at": chrono::Utc::now().to_rfc3339(),
        });
        goal.context = Some(context.to_string());
        goal.updated_at = chrono::Utc::now().to_rfc3339();
        self.state.update_goal(&goal).await?;

        Ok(format!(
            "Updated run instructions for scheduled goal {}. No run was triggered.",
            goal.id
        ))
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

        self.state
            .set_goal_budgets(&goal.id, budget_per_check, budget_daily)
            .await?;
        if budget_daily.is_some() {
            crate::goal_tokens::clear_goal_daily_budget_override(self.state.as_ref(), &goal.id)
                .await?;
        }

        let mut out = format!(
            "Updated budget for scheduled goal {}.\n- budget_per_check: {:?} -> {:?}\n- budget_daily: {:?} -> {:?}",
            goal.id, old_per_check, goal.budget_per_check, old_daily, goal.budget_daily
        );
        if let Some(budget_daily) = goal.budget_daily {
            if goal.tokens_used_today >= budget_daily {
                out.push_str(&format!(
                    "\nNote: tokens_used_today={} already exceeds the new budget_daily={}, so new scheduled runs may be skipped until the UTC daily reset.",
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

    fn format_duration(started_at: Option<&str>, completed_at: Option<&str>) -> String {
        crate::duration_format::compact_elapsed_timestamps(
            started_at,
            completed_at,
            crate::duration_format::ZeroUnitStyle::Keep,
        )
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
                "No obvious failure signature found. Use goal_trace(action='tool_trace') to inspect exact tool-call errors.",
            );
        }
        hints
    }

    pub(crate) async fn run_now(
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

        let mut adaptive_extension = None;
        if let Some(configured_budget) = goal.budget_daily.filter(|budget| *budget > 0) {
            let today = chrono::Utc::now().date_naive().to_string();
            let used_today = if goal.tokens_used_day == today {
                goal.tokens_used_today.max(0)
            } else {
                0
            };
            let durable = crate::goal_tokens::load_goal_daily_budget_override(
                self.state.as_ref(),
                &goal.id,
                configured_budget,
                crate::agent::SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
            )
            .await;
            let mut effective_budget = durable
                .as_ref()
                .map(|value| value.budget_daily)
                .unwrap_or(configured_budget);
            let extensions_count = durable
                .as_ref()
                .map(|value| value.extensions_count)
                .unwrap_or(0);

            if used_today >= effective_budget
                && extensions_count < crate::agent::SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS
            {
                if let Some(next_budget) = crate::goal_tokens::next_goal_daily_budget(
                    effective_budget,
                    used_today,
                    crate::agent::SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
                ) {
                    let persisted = crate::goal_tokens::persist_goal_daily_budget_override(
                        self.state.as_ref(),
                        &goal.id,
                        next_budget,
                        extensions_count.saturating_add(1),
                    )
                    .await?;
                    adaptive_extension = Some((effective_budget, persisted.budget_daily));
                    effective_budget = persisted.budget_daily;
                }
            }

            if used_today >= effective_budget {
                return Ok(format!(
                    "Skipped run_now for {}: cumulative usage for this goal across today's runs is {} / {} tokens, so its bounded same-day hard cap is exhausted. This is not the cost of one run. It can run after the UTC reset.",
                    resolved_goal_id, used_today, effective_budget
                ));
            }
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

        // Manual execution is an explicit request to resume this scheduled
        // goal. A stalled/failed goal is otherwise invisible to the heartbeat
        // dispatcher, which can strand the newly created task forever while
        // its run misleadingly appears "running". Reactivate before checking
        // for existing work so a repeated run_now repairs the original run
        // instead of creating a duplicate.
        let reactivated = matches!(goal.status.as_str(), "failed" | "stalled");
        if reactivated {
            let now = chrono::Utc::now().to_rfc3339();
            goal.status = "active".to_string();
            goal.completed_at = None;
            goal.updated_at = now;
            self.state.update_goal(&goal).await?;
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
            let run_id = self
                .state
                .get_current_goal_run(&goal.id)
                .await?
                .map(|run| run.id)
                .unwrap_or_else(|| "unknown".to_string());
            let action = if reactivated {
                "Resumed the existing manual run by reactivating its stalled goal"
            } else {
                "Kept the existing open run"
            };
            return Ok(format!(
                "{action}; no duplicate run was created.\n- Durable run ID: {run_id}\n- Open task(s): {}\n- Goal status: {}",
                preview,
                goal.status,
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
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: now.clone(),
            started_at: None,
            completed_at: None,
        };

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

        if let Some(open_run) = self.state.get_current_goal_run(&goal.id).await? {
            let run_tasks = self.state.get_tasks_for_goal_run(&open_run.id).await?;
            if !run_tasks.is_empty() {
                let status = if run_tasks.iter().any(|existing| {
                    matches!(
                        existing.status.as_str(),
                        "failed" | "interrupted" | "cancelled"
                    )
                }) {
                    "failed"
                } else {
                    "completed"
                };
                self.state
                    .finish_goal_run(
                        &open_run.id,
                        status,
                        Some("Closed before an explicit manual run."),
                    )
                    .await?;
            }
        }
        let run = self
            .state
            .start_goal_run(&goal.id, "manual", schedule_id, Some(&task.id))
            .await?;

        goal.last_useful_action = Some(now.clone());
        goal.updated_at = now;
        self.state.update_goal(&goal).await?;
        self.state.create_task(&task).await?;

        let mut out = format!(
            "Queued manual run for scheduled goal {}.\n- Durable run ID: {}\n- Created task: {}\n- Run status: {}\n- Goal status: {}",
            resolved_goal_id, run.id, task.id, run.status, goal.status
        );
        if schedule_consumed {
            out.push_str("\n- One-shot schedule consumed: schedule deleted.");
        }
        if let Some((old_budget, new_budget)) = adaptive_extension {
            out.push_str(&format!(
                "\n- Same-day adaptive capacity: {old_budget} -> {new_budget}; restart-safe and bounded by the hard cap."
            ));
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

        let mut runs = self.state.get_goal_runs(&goal.id).await?;
        if runs.is_empty() {
            return Ok(format!("No runs found yet for scheduled goal {}.", goal.id));
        }
        let cap = limit.clamp(1, 50);
        runs.truncate(cap);

        let mut failed = 0usize;
        let mut completed = 0usize;
        let mut pending = 0usize;
        let mut running = 0usize;
        let mut blocked = 0usize;
        for run in &runs {
            match run.status.as_str() {
                "failed" => failed += 1,
                "completed" => completed += 1,
                "pending" => pending += 1,
                "running" => running += 1,
                "blocked" => blocked += 1,
                _ => {}
            }
        }

        let mut out = format!(
            "**Scheduled Run History**\n\n- Goal: {}\n- ID: {}\n- Showing: {} run(s)\n- Status mix: completed {}, failed {}, running {}, pending {}, blocked {}\n",
            goal.description,
            goal.id,
            runs.len(),
            completed,
            failed,
            running,
            pending,
            blocked
        );

        let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
        let coalesces = schedules.iter().any(|s| s.fire_policy != "always_fire");
        let mut tasks_by_run = Vec::with_capacity(runs.len());
        for run in &runs {
            tasks_by_run.push(
                self.state
                    .get_tasks_for_goal_run(&run.id)
                    .await
                    .unwrap_or_default(),
            );
        }
        let open_tasks = tasks_by_run
            .iter()
            .flatten()
            .filter(|task| {
                matches!(
                    task.status.as_str(),
                    "pending" | "claimed" | "running" | "blocked"
                )
            })
            .collect::<Vec<_>>();
        if coalesces && !open_tasks.is_empty() {
            out.push_str("\n**Open task(s) blocking coalesced schedule fires**");
            for task in open_tasks.iter().take(5) {
                out.push_str(&format!(
                    "\n- **{}** status={} created={} desc={}",
                    task.id,
                    task.status,
                    task.created_at,
                    Self::truncate(&task.description, 160)
                ));
            }
            out.push('\n');
        }

        for (run, tasks) in runs.iter().zip(tasks_by_run.iter()) {
            out.push_str(&format!(
                "\n- **{}** trigger={} status={} started={} duration={} tasks={}",
                run.id,
                run.trigger_type,
                run.status,
                run.started_at,
                Self::format_duration(Some(&run.started_at), run.completed_at.as_deref()),
                tasks.len()
            ));
            if let Some(summary) = &run.outcome_summary {
                out.push_str(&format!("\n  summary: {}", Self::truncate(summary, 220)));
            }
            for task in tasks
                .iter()
                .filter(|task| matches!(task.status.as_str(), "failed" | "blocked" | "interrupted"))
            {
                out.push_str(&format!(
                    "\n  task {} status={} desc={}",
                    Self::short_id(&task.id),
                    task.status,
                    Self::truncate(&task.description, 120)
                ));
                if let Some(err) = &task.error {
                    out.push_str(&format!("\n  error: {}", Self::truncate(err, 160)));
                }
                if let Some(blocker) = &task.blocker {
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
    instructions: Option<String>,
    #[serde(default)]
    run_requirement: Option<String>,
    #[serde(default)]
    insufficient_evidence: Option<String>,
    #[serde(default)]
    include_diagnostics: Option<bool>,
    #[serde(default)]
    _user_role: Option<String>,
}

fn string_enum(values: &[&str]) -> Value {
    json!({"type": "string", "enum": values})
}

fn scheduled_goal_runs_schema() -> Value {
    json!({
        "name": "scheduled_goal_runs",
        "description": "Manage runs.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run_now", "run_history", "last_failure", "unblock_hints", "set_budget", "update_instructions"]
                },
                "goal_id": {
                    "type": "string"
                },
                "schedule_id": {
                    "type": "string"
                },
                "limit": {
                    "type": "integer"
                },
                "budget_per_check": {
                    "type": "integer"
                },
                "budget_daily": {
                    "type": "integer"
                },
                "instructions": {
                    "type": "string"
                },
                "run_requirement": string_enum(&["must_complete", "best_effort"]),
                "insufficient_evidence": string_enum(&["use_public_sources", "use_best_available", "skip_run"]),
                "include_diagnostics": {
                    "type": "boolean",
                    "description": "True only for user-requested internal IDs."
                }
            },
            "required": ["action", "goal_id"],
            "additionalProperties": false
        }
    })
}

#[async_trait]
impl Tool for ScheduledGoalRunsTool {
    fn name(&self) -> &str {
        "scheduled_goal_runs"
    }

    fn description(&self) -> &str {
        "Run, update, and inspect scheduled goals without terminal/sqlite access"
    }

    fn schema(&self) -> Value {
        scheduled_goal_runs_schema()
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        semantics_for_exact_read_actions(
            arguments,
            &["run_history", "last_failure", "unblock_hints"],
            ToolMutationEffects::NONE,
        )
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
	            "update_instructions" => {
	                let is_owner = args
	                    ._user_role
	                    .as_deref()
	                    .is_some_and(|role| role.eq_ignore_ascii_case("owner"));
	                if !is_owner {
	                    return Ok("Only owners can update scheduled goal instructions.".to_string());
	                }
	                let goal_id = args
	                    .goal_id
	                    .as_deref()
	                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for update_instructions"))?;
	                let instructions = args.instructions.as_deref().ok_or_else(|| {
	                    anyhow::anyhow!("'instructions' is required for update_instructions")
	                })?;
	                self.update_instructions(
                        goal_id,
                        instructions,
                        args.run_requirement.as_deref(),
                        args.insufficient_evidence.as_deref(),
                    )
                    .await
	            }
	            other => Ok(format!(
	                "Unknown action: '{}'. Use run_now, run_history, last_failure, unblock_hints, set_budget, or update_instructions.",
	                other
	            )),
	        }
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: ScheduledGoalRunsArgs = serde_json::from_str(arguments)?;
        let output = self.call(arguments).await?;
        let mut internal_identifiers = crate::tools::sanitize::extract_uuid_identifiers(&output);
        for candidate in [args.goal_id.as_deref(), args.schedule_id.as_deref()]
            .into_iter()
            .flatten()
        {
            internal_identifiers
                .extend(crate::tools::sanitize::extract_uuid_identifiers(candidate));
        }
        internal_identifiers.sort();
        internal_identifiers.dedup();

        let presentation = if args.include_diagnostics.unwrap_or(false) {
            Some(ToolResultPresentation::DiagnosticDetail)
        } else if matches!(
            args.action.as_str(),
            "run_now" | "set_budget" | "update_instructions"
        ) {
            Some(ToolResultPresentation::NaturalSummary)
        } else {
            None
        };
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                presentation,
                internal_identifiers,
                ..ToolCallMetadata::default()
            },
        })
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
        assert!(schema["parameters"]["properties"]["include_diagnostics"].is_object());
    }

    #[tokio::test]
    async fn run_now_creates_pending_task() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let mut goal = Goal::new_continuous(
            "Run diagnostics job",
            "user-session",
            Some(1000),
            Some(5000),
        );
        goal.context = Some(
            serde_json::json!({
                "instructions": "Search a current authoritative source before drafting."
            })
            .to_string(),
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
        assert!(result.contains("Queued manual run"));

        let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status, "pending");
        assert!(!tasks[0].idempotent);
        assert_eq!(tasks[0].max_retries, 0);
        assert!(tasks[0].description.starts_with("Manual scheduled run:"));
        assert_eq!(tasks[0].context, goal.context);

        let run = state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("manual trigger creates an open goal run");
        assert_eq!(run.trigger_type, "manual");
        assert_eq!(run.status, "pending");
        assert_eq!(run.root_task_id.as_deref(), Some(tasks[0].id.as_str()));
        assert_eq!(
            state.get_tasks_for_goal_run(&run.id).await.unwrap()[0].id,
            tasks[0].id
        );
    }

    #[tokio::test]
    async fn run_now_durably_extends_an_exhausted_daily_budget_once() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let mut goal = Goal::new_continuous(
            "Publish a synthetic daily entry",
            "synthetic-session",
            Some(400_000),
            Some(1_000_000),
        );
        goal.tokens_used_today = 1_253_197;
        goal.tokens_used_day = chrono::Utc::now().date_naive().to_string();
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: crate::cron_utils::compute_next_run("0 6 * * *")
                    .unwrap()
                    .to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(&json!({"action": "run_now", "goal_id": goal.id}).to_string())
            .await
            .unwrap();

        assert!(result.contains("Queued manual run"), "result: {result}");
        assert!(result.contains("Same-day adaptive capacity"));
        let durable = crate::goal_tokens::load_goal_daily_budget_override(
            state.as_ref(),
            &goal.id,
            1_000_000,
            crate::agent::SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
        )
        .await
        .expect("manual run should persist the adaptive budget");
        assert_eq!(durable.budget_daily, 2_000_000);
        assert_eq!(durable.extensions_count, 1);
    }

    #[tokio::test]
    async fn run_now_still_stops_at_the_bounded_daily_hard_cap() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let mut goal = Goal::new_continuous(
            "Run bounded synthetic work",
            "synthetic-session",
            Some(400_000),
            Some(1_000_000),
        );
        goal.tokens_used_today = 2_100_000;
        goal.tokens_used_day = chrono::Utc::now().date_naive().to_string();
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: crate::cron_utils::compute_next_run("0 6 * * *")
                    .unwrap()
                    .to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(&json!({"action": "run_now", "goal_id": goal.id}).to_string())
            .await
            .unwrap();

        assert!(result.contains("hard cap is exhausted"), "result: {result}");
        assert!(state.get_tasks_for_goal(&goal.id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn run_now_marks_internal_ids_for_natural_owner_presentation() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous("Run synthetic diagnostics", "session", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
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
            next_run_at: crate::cron_utils::compute_next_run("0 */6 * * *")
                .unwrap()
                .to_rfc3339(),
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let outcome = tool
            .call_with_status_outcome(
                &json!({"action": "run_now", "goal_id": goal.id}).to_string(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.presentation,
            Some(ToolResultPresentation::NaturalSummary)
        );
        assert!(outcome.metadata.internal_identifiers.contains(&goal.id));
        let run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        assert!(outcome.metadata.internal_identifiers.contains(&run.id));
        assert!(!outcome.metadata.internal_identifiers.is_empty());
    }

    #[tokio::test]
    async fn explicit_run_diagnostics_select_diagnostic_presentation() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous("Run synthetic diagnostics", "session", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 */6 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("every 6h".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: crate::cron_utils::compute_next_run("0 */6 * * *")
                    .unwrap()
                    .to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "action": "run_now",
                    "goal_id": goal.id,
                    "include_diagnostics": true
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.presentation,
            Some(ToolResultPresentation::DiagnosticDetail)
        );
    }

    #[tokio::test]
    async fn run_now_reactivates_stalled_goal_without_duplicate_run() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let mut goal = Goal::new_continuous(
            "Publish one diary entry",
            "user-session",
            Some(1000),
            Some(5000),
        );
        goal.status = "stalled".to_string();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
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
            next_run_at: crate::cron_utils::compute_next_run("0 */6 * * *")
                .unwrap()
                .to_rfc3339(),
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: format!("Manual scheduled run: {}", goal.description),
            status: "pending".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: goal.context.clone(),
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        let original_run = state
            .start_goal_run(&goal.id, "manual", Some(&schedule.id), Some(&task.id))
            .await
            .unwrap();
        state.create_task(&task).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "run_now",
                    "goal_id": goal.id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Resumed the existing manual run"));
        assert!(result.contains("no duplicate run was created"));
        assert!(result.contains(&original_run.id));
        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
        assert_eq!(state.get_tasks_for_goal(&goal.id).await.unwrap().len(), 1);
        assert_eq!(state.get_goal_runs(&goal.id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn update_instructions_persists_without_triggering_and_replaces_prior_update() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let mut goal = Goal::new_continuous(
            "Publish one daily insight",
            "user-session",
            Some(1000),
            Some(5000),
        );
        goal.context = Some(json!({"existing": "preserved"}).to_string());
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily at 9am".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: crate::cron_utils::compute_next_run("0 9 * * *")
                .unwrap()
                .to_rfc3339(),
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        for instructions in [
            "Search a current authoritative source before drafting.",
            "Use one current primary source and write a human-useful insight.",
        ] {
            let result = tool
                .call(
                    &json!({
                        "action": "update_instructions",
                        "goal_id": goal.id,
                        "instructions": instructions,
                        "_user_role": "Owner"
                    })
                    .to_string(),
                )
                .await
                .unwrap();
            assert!(result.contains("No run was triggered"));
        }

        let updated = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert!(updated
            .description
            .contains("Use one current primary source and write a human-useful insight."));
        assert!(!updated
            .description
            .contains("Search a current authoritative source before drafting."));
        assert_eq!(
            updated
                .description
                .matches(ScheduledGoalRunsTool::RUN_INSTRUCTIONS_MARKER)
                .count(),
            1
        );
        let context: Value = serde_json::from_str(updated.context.as_deref().unwrap()).unwrap();
        assert_eq!(context["existing"], "preserved");
        assert_eq!(
            context["run_instructions"],
            "Use one current primary source and write a human-useful insight."
        );
        assert_eq!(context["run_instruction_spec"]["version"], 1);
        assert_eq!(context["run_instruction_spec"]["revision"], 2);
        assert_eq!(
            context["run_instruction_spec"]["conflict_rule"],
            "latest_revision_wins"
        );
        assert!(state.get_tasks_for_goal(&goal.id).await.unwrap().is_empty());
        assert!(state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            state.get_schedules_for_goal(&goal.id).await.unwrap()[0].id,
            schedule.id
        );
    }

    #[tokio::test]
    async fn update_instructions_rejects_structurally_conflicting_policy() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());
        let goal = Goal::new_continuous("Publish one verified note", "session", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .create_goal_schedule(&GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: "0 9 * * *".to_string(),
                tz: "local".to_string(),
                original_schedule: Some("daily".to_string()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: false,
                is_paused: false,
                last_run_at: None,
                next_run_at: crate::cron_utils::compute_next_run("0 9 * * *")
                    .unwrap()
                    .to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "update_instructions",
                    "goal_id": goal.id,
                    "instructions": "Always finish the run.",
                    "run_requirement": "must_complete",
                    "insufficient_evidence": "skip_run",
                    "_user_role": "Owner"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Conflicting run policy"));
        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().description,
            goal.description
        );
    }

    #[tokio::test]
    async fn run_history_reports_open_tasks_as_schedule_blockers() {
        let state = setup_state().await;
        let tool = ScheduledGoalRunsTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Post daily optimized tweets",
            "user-session",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run("0 9 * * *")
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 9 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily at 9am".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();

        let open_task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Analyze fetched engagement metrics".to_string(),
            status: "pending".to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        state.create_task(&open_task).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "run_history",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Open task(s) blocking coalesced schedule fires"));
        assert!(result.contains("Analyze fetched engagement metrics"));
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

        let allowed_relation = tool
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
        assert!(allowed_relation.contains("Updated budget"));

        let updated = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(updated.budget_per_check, Some(200));
        assert_eq!(updated.budget_daily, Some(100));
    }

    #[test]
    fn schema_fits_payload_budget() {
        // Pillar C of 2026-06-06-cross-turn-prefix-stability-design.md:
        // admin-tool schemas ride in EVERY provider call; this ceiling is the
        // per-tool payload budget. If you trip this assert by adding features,
        // compress the description text — do not raise the ceiling without
        // updating the Pillar C implementation plan.
        let bytes = serde_json::to_string(&scheduled_goal_runs_schema())
            .unwrap()
            .len();
        assert!(
            bytes <= 800,
            "scheduled_goal_runs schema is {bytes} bytes, budget is 800"
        );
    }
}
