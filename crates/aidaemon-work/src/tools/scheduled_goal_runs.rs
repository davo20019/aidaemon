//! Runs of scheduled work: run one now, its run history and last failure,
//! its budget, run instructions and workspace binding, and the canonical
//! snapshot of every scheduled objective. Not a tool: `manage_automations`
//! offers these to the model.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::traits::{ScheduledGoalRunsStore, Task};

pub struct ScheduledGoalRuns {
    state: Arc<dyn ScheduledGoalRunsStore>,
}

impl ScheduledGoalRuns {
    pub fn new(state: Arc<dyn ScheduledGoalRunsStore>) -> Self {
        Self { state }
    }

    /// Separates a goal's description from the latest run's instructions.
    pub const RUN_INSTRUCTIONS_MARKER: &'static str = "\n\nLATEST RUN INSTRUCTIONS:\n";

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

    /// Bind every run of a scheduled goal to one existing project directory.
    /// The binding is typed durable state (`context.project_scope`) that the
    /// task lead honors before any mission-text inference, so a run whose
    /// objective lives in a sibling repository never starts in an empty
    /// synthetic workspace again.
    pub async fn bind_workspace(
        &self,
        goal_id_input: &str,
        workspace_path: &str,
    ) -> anyhow::Result<String> {
        let resolved_goal_id = match self.resolve_goal_id(goal_id_input).await {
            Ok(id) => id,
            Err(error) => return Ok(error.to_string()),
        };
        let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
            return Ok(format!("Scheduled goal not found: {resolved_goal_id}"));
        };
        let raw = workspace_path.trim();
        if raw.is_empty() {
            return Ok("Provide a non-empty workspace_path.".to_string());
        }
        let expanded = if let Some(rest) = raw.strip_prefix("~/") {
            match dirs::home_dir() {
                Some(home) => home.join(rest),
                None => return Ok("Cannot resolve ~ without a home directory.".to_string()),
            }
        } else {
            std::path::PathBuf::from(raw)
        };
        if !expanded.is_absolute() {
            return Ok("workspace_path must be absolute (or start with ~/).".to_string());
        }
        let canonical = match std::fs::canonicalize(&expanded) {
            Ok(path) => path,
            Err(error) => {
                return Ok(format!(
                    "workspace_path does not exist or is not accessible: {} ({error})",
                    expanded.display()
                ));
            }
        };
        if !canonical.is_dir() {
            return Ok(format!(
                "workspace_path is not a directory: {}",
                canonical.display()
            ));
        }
        if aidaemon_tools::fs_utils::is_protected_host_data_path(&canonical) {
            return Ok(
                "workspace_path points at protected host data; choose a project directory."
                    .to_string(),
            );
        }
        let mut context = goal
            .context
            .as_deref()
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        let scope = canonical.to_string_lossy().to_string();
        let previous = context
            .get("project_scope")
            .and_then(Value::as_str)
            .map(str::to_string);
        context["project_scope"] = Value::String(scope.clone());
        context["project_scope_bound_at"] = Value::String(chrono::Utc::now().to_rfc3339());
        goal.context = Some(context.to_string());
        goal.updated_at = chrono::Utc::now().to_rfc3339();
        self.state.update_goal(&goal).await?;
        Ok(format!(
            "Bound scheduled goal workspace to {scope}{}. Future runs (including automatic recovery) execute there.",
            previous
                .filter(|prior| prior != &scope)
                .map(|prior| format!(" (previously {prior})"))
                .unwrap_or_default()
        ))
    }

    pub async fn update_instructions(
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

    pub async fn set_budget(
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
            return Ok("Only scheduled work can have a budget.".to_string());
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
        if let Some(budget_daily) = goal.budget_daily
            && goal.tokens_used_today >= budget_daily
        {
            out.push_str(&format!(
                    "\nNote: tokens_used_today={} already exceeds the new budget_daily={}, so new scheduled runs may be skipped until the UTC daily reset.",
                    goal.tokens_used_today, budget_daily
                ));
        }
        Ok(out)
    }

    /// The scheduled goal a handle names: its id, an id prefix, or a distinctive
    /// phrase from its objective.
    pub async fn resolve_goal_id(&self, input_id: &str) -> anyhow::Result<String> {
        let trimmed = input_id.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Empty goal ID");
        }

        if self.state.get_goal(trimmed).await?.is_some() {
            return Ok(trimmed.to_string());
        }

        let goals = self.state.get_scheduled_goals().await?;
        super::goal_ids::pick_goal(trimmed, &goals, "Scheduled goal")
    }

    fn short_id(id: &str) -> String {
        super::goal_ids::short_goal_id(id)
    }

    fn truncate(s: &str, max: usize) -> String {
        s.chars().take(max).collect()
    }

    fn format_duration(started_at: Option<&str>, completed_at: Option<&str>) -> String {
        aidaemon_tools::duration_format::compact_elapsed_timestamps(
            started_at,
            completed_at,
            aidaemon_tools::duration_format::ZeroUnitStyle::Keep,
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

    /// Return one canonical, read-only snapshot of every scheduled objective,
    /// its schedule state, and its latest durable execution. Discovery and run
    /// inspection belong on the same capability surface: requiring callers to
    /// first guess an ID (or route through the memory tool) makes absence and
    /// cross-object audits impossible to establish reliably.
    pub async fn overview_snapshot(
        &self,
        limit: usize,
        include_diagnostics: bool,
    ) -> anyhow::Result<(String, crate::tools::objective_status::ObjectivePortfolio)> {
        // Shared with `manage_mandates list`: both owner audit surfaces must
        // report one membership for `objective_collection:scheduled_goals`.
        let (goals, coverage) = crate::tools::objective_status::load_scheduled_goal_collection(
            self.state.as_ref(),
            limit,
        )
        .await?;
        let mut portfolio = crate::tools::objective_status::ObjectivePortfolio::default();
        portfolio.record_collection(coverage)?;
        let mut records = Vec::new();
        for goal in goals {
            let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
            let mut tasks = self.state.get_tasks_for_goal(&goal.id).await?;
            tasks.sort_by(|left, right| {
                let left_time = left
                    .completed_at
                    .as_deref()
                    .or(left.started_at.as_deref())
                    .unwrap_or(&left.created_at);
                let right_time = right
                    .completed_at
                    .as_deref()
                    .or(right.started_at.as_deref())
                    .unwrap_or(&right.created_at);
                right_time.cmp(left_time)
            });
            let latest_run = tasks.first().map(|task| {
                let mut run = json!({
                    "status": task.status,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                });
                if include_diagnostics {
                    run["task_id"] = Value::String(task.id.clone());
                }
                run
            });
            let last_run_at = schedules
                .iter()
                .filter_map(|schedule| schedule.last_run_at.as_deref())
                .max();
            let workspace_binding = goal
                .context
                .as_deref()
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .and_then(|context| {
                    context
                        .get("project_scope")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                });
            let runs = self.state.get_goal_runs(&goal.id).await?;
            let latest_run_outcome = runs.first().map(|run| {
                json!({
                    "status": run.status,
                    "completed_at": run.completed_at,
                    "summary": run
                        .outcome_summary
                        .as_deref()
                        .map(|summary| Self::truncate(summary, 200)),
                })
            });
            let mandate = self.state.get_mandate_for_goal(&goal.id).await?;
            let measurement_count = match mandate.as_ref() {
                Some(mandate) => self
                    .state
                    .list_mandate_objective_measurements(&mandate.id, 500)
                    .await?
                    .len(),
                None => 0,
            };
            let recovery = self.state.get_scheduled_recovery_state(&goal.id).await?;
            let row = crate::tools::objective_status::objective_portfolio_row(
                &goal.id,
                crate::tools::objective_status::ObjectiveCollection::ScheduledGoals,
                &schedules,
                &runs,
                recovery.as_ref(),
                mandate.as_ref(),
                measurement_count,
            )?;
            let status = &row.status;
            let active_schedule_count = schedules
                .iter()
                .filter(|schedule| !schedule.is_paused)
                .count();
            let mut record = json!({
                "objective_id": crate::tools::objective_status::objective_resource_id(&goal.id),
                "objective": Self::truncate(&goal.description, 240),
                "goal_status": goal.status,
                "workspace_binding": workspace_binding,
                "source_membership": row.source_membership_json(),
                "run_history": {
                    "completed": status.runs_completed,
                    "failed": status.runs_failed,
                    "total": runs.len(),
                },
                "latest_run_outcome": latest_run_outcome,
                "schedule_count": schedules.len(),
                "active_schedule_count": active_schedule_count,
                "schedule_state": status.wake.state(),
                "wake_source": status.wake.source(),
                "next_run_at": status.wake.next_at(),
                "last_run_at": last_run_at,
                "latest_run": latest_run,
                "objective_control":
                    crate::tools::objective_status::objective_control_json(status.control.as_ref()),
                "objective_measurement_count": measurement_count,
            });
            if let Some(recovery) = recovery {
                record["recovery"] = json!({
                    "disposition": recovery.disposition.as_str(),
                    "consecutive_failures": recovery.consecutive_failures,
                    "failure_budget": recovery.failure_budget,
                    "latest_failure_kind": recovery
                        .latest_failure_kind
                        .map(crate::traits::ScheduledFailureKind::as_str),
                    "recovery_run_linked": recovery.last_recovery_run_id.is_some(),
                });
            }
            if include_diagnostics {
                record["goal_id"] = Value::String(goal.id);
                record["schedule_ids"] = Value::Array(
                    schedules
                        .iter()
                        .map(|schedule| Value::String(schedule.id.clone()))
                        .collect(),
                );
            }
            portfolio.insert(row)?;
            records.push(record);
        }

        let output = serde_json::to_string_pretty(&json!({
            "snapshot": "scheduled_objectives",
            "complete": coverage.is_complete(),
            "total": coverage.total,
            "returned": coverage.returned,
            "collection_scope": portfolio.collection_scope_json(
                crate::tools::objective_status::ObjectiveCollection::ScheduledGoals,
            )?,
            "objectives": records,
        }))?;
        Ok((output, portfolio))
    }

    pub async fn overview(&self, limit: usize, include_diagnostics: bool) -> anyhow::Result<String> {
        self.overview_snapshot(limit, include_diagnostics)
            .await
            .map(|(output, _)| output)
    }

    pub async fn run_now(
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
            return Ok("Only scheduled work can be run now.".to_string());
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
                crate::goal_tokens::SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
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
                && extensions_count < crate::goal_tokens::SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS
                && let Some(next_budget) = crate::goal_tokens::next_goal_daily_budget(
                    effective_budget,
                    used_today,
                    crate::goal_tokens::SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
                )
            {
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
                    "Goal {} is paused. Resume it first, then run it now.",
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
                preview, goal.status,
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

    pub async fn run_history(&self, goal_id_input: &str, limit: usize) -> anyhow::Result<String> {
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

        if let Some(recovery) = self.state.get_scheduled_recovery_state(&goal.id).await? {
            out.push_str(&format!(
                "- Recovery: {} (failures {}/{}, latest cause {}, last failed run {}, last recovery run {})\n",
                recovery.disposition.as_str(),
                recovery.consecutive_failures,
                recovery.failure_budget,
                recovery
                    .latest_failure_kind
                    .map_or("none", crate::traits::ScheduledFailureKind::as_str),
                recovery.last_failed_run_id.as_deref().unwrap_or("none"),
                recovery.last_recovery_run_id.as_deref().unwrap_or("none")
            ));
        }

        let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
        if schedules.is_empty() {
            out.push_str("- Schedule: unavailable\n");
        } else {
            out.push_str("- Schedule state:\n");
            for schedule in schedules.iter().take(5) {
                out.push_str(&format!(
                    "  - id={} next_run={} last_run={} paused={} policy={}\n",
                    schedule.id,
                    schedule.next_run_at,
                    schedule.last_run_at.as_deref().unwrap_or("never"),
                    schedule.is_paused,
                    schedule.fire_policy,
                ));
            }
        }
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
        let open_tasks = runs
            .iter()
            .zip(tasks_by_run.iter())
            .flat_map(|(run, tasks)| {
                tasks.iter().filter(move |task| {
                    crate::scheduler::task_blocks_later_schedule_fire(run, task)
                })
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
                "\n- **{}** trigger={} status={} started={} completed={} duration={} tasks={}",
                run.id,
                run.trigger_type,
                run.status,
                run.started_at,
                run.completed_at.as_deref().unwrap_or("not_completed"),
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

    pub async fn last_failure(&self, goal_id_input: &str) -> anyhow::Result<String> {
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
}
