use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::traits::{GoalTraceStore, TaskActivity, ToolCapabilities};
use aidaemon_tools::OutcomeTool as Tool;

pub struct GoalTraceTool {
    state: Arc<dyn GoalTraceStore>,
}

impl GoalTraceTool {
    pub fn new(state: Arc<dyn GoalTraceStore>) -> Self {
        Self { state }
    }

    async fn resolve_goal_id(&self, input_id: &str) -> anyhow::Result<String> {
        let trimmed = input_id.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Empty goal ID");
        }
        if self.state.get_goal(trimmed).await?.is_some() {
            return Ok(trimmed.to_string());
        }

        if let Some(mandate) = self.state.get_mandate(trimmed).await? {
            return Ok(mandate.goal_id);
        }
        let mandate_matches = self
            .state
            .list_mandates(None, true)
            .await?
            .into_iter()
            .filter(|mandate| mandate.id.starts_with(trimmed))
            .collect::<Vec<_>>();
        match mandate_matches.as_slice() {
            [mandate] => return Ok(mandate.goal_id.clone()),
            [] => {}
            _ => {
                let preview = mandate_matches
                    .iter()
                    .take(5)
                    .map(|mandate| format!("{} ({})", Self::short_id(&mandate.id), mandate.status))
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow::bail!(
                    "Mandate ID prefix '{}' is ambiguous ({} matches): {}. Use the full mandate_id.",
                    trimmed,
                    mandate_matches.len(),
                    preview
                );
            }
        }

        let mut candidates = self.state.get_active_goals().await?;
        for g in self.state.get_scheduled_goals().await? {
            if !candidates.iter().any(|x| x.id == g.id) {
                candidates.push(g);
            }
        }
        super::goal_ids::pick_goal(trimmed, &candidates, "Goal")
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

    async fn recent_goal_summary(&self, limit: usize) -> anyhow::Result<String> {
        let mut goals = self.state.get_active_goals().await?;
        for g in self.state.get_scheduled_goals().await? {
            if !goals.iter().any(|existing| existing.id == g.id) {
                goals.push(g);
            }
        }

        goals.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        goals.truncate(limit.clamp(1, 20));

        if goals.is_empty() {
            return Ok(
                "No recent goals found. For simple chat recall questions, use conversation history instead of goal_trace."
                    .to_string(),
            );
        }

        let mut out = String::from(
            "**Goal Trace (Recent Goals)**\n\n\
             No `goal_id` was provided, so here are the most recent active/scheduled goals:\n",
        );
        for goal in goals {
            out.push_str(&format!(
                "- {} ({}) updated {}\n  id: {}",
                Self::truncate(&goal.description, 100),
                goal.status,
                goal.updated_at,
                goal.id
            ));
            out.push('\n');
        }
        out.push_str(
            "\nTip: call `goal_trace` again with `goal_id` (full or unique prefix) for full task/tool timeline details.",
        );
        Ok(out)
    }

    async fn gather_goal_activities(
        &self,
        goal_id: &str,
    ) -> anyhow::Result<Vec<(String, TaskActivity)>> {
        let mut rows = Vec::new();
        let tasks = self.state.get_tasks_for_goal(goal_id).await?;
        for t in &tasks {
            let activities = self.state.get_task_activities(&t.id).await?;
            for a in activities {
                rows.push((t.id.clone(), a));
            }
        }
        rows.sort_by(|a, b| a.1.created_at.cmp(&b.1.created_at));
        Ok(rows)
    }

    async fn goal_trace(
        &self,
        goal_id_input: &str,
        max_tasks: usize,
        max_activities_per_task: usize,
    ) -> anyhow::Result<String> {
        let resolved_goal_id = match self.resolve_goal_id(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(goal) = self.state.get_goal(&resolved_goal_id).await? else {
            return Ok(format!("Goal not found: {}", resolved_goal_id));
        };

        let mut tasks = self.state.get_tasks_for_goal(&goal.id).await?;
        tasks.sort_by(|a, b| a.created_at.cmp(&b.created_at));

        let mut status_counts: HashMap<String, usize> = HashMap::new();
        let mut retries_total = 0i32;
        let mut tokens_total = 0i64;
        for t in &tasks {
            *status_counts.entry(t.status.clone()).or_insert(0) += 1;
            retries_total += t.retry_count;
            let activities = self.state.get_task_activities(&t.id).await?;
            tokens_total += activities.iter().filter_map(|a| a.tokens_used).sum::<i64>();
        }

        let mut out = format!(
            "**Goal Trace**\n\n- Goal: {}\n- Goal ID: {}\n- Status: {}\n- Type: {}\n- Created: {}\n- Updated: {}\n- Tasks: {}\n- Retries used: {}\n- Activity tokens (logged): {}\n",
            goal.description,
            goal.id,
            goal.status,
            goal.goal_type,
            goal.created_at,
            goal.updated_at,
            tasks.len(),
            retries_total,
            tokens_total
        );

        if !status_counts.is_empty() {
            let mut keys: Vec<_> = status_counts.keys().cloned().collect();
            keys.sort();
            let status_str = keys
                .iter()
                .map(|k| format!("{} {}", k, status_counts[k]))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("- Task status mix: {}\n", status_str));
        }

        // Shared authoritative lifecycle readout: the same schedule/run/
        // recovery/control answer every other objective surface projects, so
        // an audit consulting the trace lane sees identical typed state.
        let mandate = self.state.get_mandate_for_goal(&goal.id).await?;
        {
            let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
            let runs = self.state.get_goal_runs(&goal.id).await?;
            let recovery = self.state.get_scheduled_recovery_state(&goal.id).await?;
            if !schedules.is_empty() || !runs.is_empty() || recovery.is_some() || mandate.is_some()
            {
                let measurement_count = match mandate.as_ref() {
                    Some(mandate) => self
                        .state
                        .list_mandate_objective_measurements(&mandate.id, 500)
                        .await?
                        .len(),
                    None => 0,
                };
                let enumerated_from = if mandate.is_some() && schedules.is_empty() {
                    crate::tools::objective_status::ObjectiveCollection::MandateControllers
                } else {
                    crate::tools::objective_status::ObjectiveCollection::ScheduledGoals
                };
                let row = crate::tools::objective_status::objective_portfolio_row(
                    &goal.id,
                    enumerated_from,
                    &schedules,
                    &runs,
                    recovery.as_ref(),
                    mandate.as_ref(),
                    measurement_count,
                )?;
                out.push_str(&format!(
                    "- Objective status: {}\n",
                    crate::tools::objective_status::render_objective_status_line(&row.status)
                ));
            }
        }

        if let Some(mandate) = mandate {
            let today = chrono::Utc::now().date_naive().to_string();
            let used_today = if goal.tokens_used_day == today {
                goal.tokens_used_today.max(0)
            } else {
                0
            };
            let remaining = goal
                .budget_daily
                .map(|daily| daily.saturating_sub(used_today).max(0));
            let can_fund_cycle = match (remaining, goal.budget_per_check) {
                (Some(remaining), Some(per_cycle)) => remaining >= per_cycle.max(0),
                _ => true,
            };
            let admission = if can_fund_cycle {
                "ready".to_string()
            } else {
                let reset = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
                    chrono::Utc::now()
                        .date_naive()
                        .succ_opt()
                        .expect("the next UTC date is representable")
                        .and_hms_opt(0, 0, 0)
                        .expect("UTC midnight is representable"),
                    chrono::Utc,
                );
                format!("budget-blocked until {}", reset.to_rfc3339())
            };
            let latest_run = self.state.get_goal_runs(&goal.id).await?.into_iter().next();
            let latest_decision = self
                .state
                .list_mandate_decisions(&mandate.id, 1)
                .await?
                .into_iter()
                .next();
            out.push_str(&format!(
                "\n**Mandate Controller**\n\n- Mandate: {} [{} v{}]\n- Next review: {}\n- Review admission: {}\n- Token budget: {:?}/cycle, {:?}/UTC day, {} used today, {:?} remaining\n- Latest run: {}\n- Latest explicit decision: {}\n",
                mandate.id,
                mandate.status,
                mandate.version,
                mandate.scheduled_next_review().unwrap_or("none, not active"),
                admission,
                goal.budget_per_check,
                goal.budget_daily,
                used_today,
                remaining,
                latest_run.as_ref().map_or_else(
                    || "none".to_string(),
                    |run| format!("{} ({})", run.id, run.status)
                ),
                latest_decision.as_ref().map_or_else(
                    || "none".to_string(),
                    |decision| format!("{} ({})", decision.outcome.as_str(), decision.created_at)
                ),
            ));
        }

        out.push_str("\n**Task Timeline**");
        for t in tasks.iter().take(max_tasks.clamp(1, 100)) {
            let activities = self.state.get_task_activities(&t.id).await?;
            let mut tool_chain = Vec::new();
            for a in activities.iter().take(max_activities_per_task.clamp(1, 20)) {
                if let Some(tool) = &a.tool_name {
                    let ok = a
                        .success
                        .map(|v| if v { "ok" } else { "err" })
                        .unwrap_or("n/a");
                    tool_chain.push(format!("{}({})", tool, ok));
                }
            }
            let tool_seq = if tool_chain.is_empty() {
                "-".to_string()
            } else {
                tool_chain.join(" -> ")
            };

            out.push_str(&format!(
                "\n- **{}** status={} retry={}/{} created={} duration={} agent={} tools={}",
                t.id,
                t.status,
                t.retry_count,
                t.max_retries,
                t.created_at,
                Self::format_duration(t.started_at.as_deref(), t.completed_at.as_deref()),
                t.agent_id
                    .as_deref()
                    .map(Self::short_id)
                    .unwrap_or_else(|| "-".to_string()),
                tool_seq
            ));

            if t.status == "failed" {
                if let Some(err) = &t.error {
                    out.push_str(&format!("\n  error: {}", Self::truncate(err, 180)));
                }
            } else if t.status == "blocked"
                && let Some(blocker) = &t.blocker
            {
                out.push_str(&format!("\n  blocker: {}", Self::truncate(blocker, 180)));
            }
        }

        Ok(out)
    }

    async fn tool_trace(
        &self,
        goal_id_input: Option<&str>,
        task_id: Option<&str>,
        tool_name: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<String> {
        let mut rows: Vec<(String, TaskActivity)> = if let Some(task_id) = task_id {
            let activities = self.state.get_task_activities(task_id).await?;
            activities
                .into_iter()
                .map(|a| (task_id.to_string(), a))
                .collect::<Vec<_>>()
        } else {
            let goal_input = goal_id_input
                .ok_or_else(|| anyhow::anyhow!("'goal_id' or 'task_id' is required"))?;
            let resolved_goal_id = match self.resolve_goal_id(goal_input).await {
                Ok(id) => id,
                Err(e) => return Ok(e.to_string()),
            };
            self.gather_goal_activities(&resolved_goal_id).await?
        };

        if let Some(filter_tool) = tool_name {
            let needle = filter_tool.to_ascii_lowercase();
            rows.retain(|(_, a)| {
                a.tool_name
                    .as_deref()
                    .unwrap_or("")
                    .to_ascii_lowercase()
                    .contains(&needle)
            });
        }

        if rows.is_empty() {
            return Ok("No matching tool activity found.".to_string());
        }

        rows.sort_by(|a, b| b.1.created_at.cmp(&a.1.created_at));
        rows.truncate(limit.clamp(1, 200));

        let mut by_tool: HashMap<String, (usize, usize, usize, i64)> = HashMap::new();
        for (_, a) in &rows {
            let key = a.tool_name.clone().unwrap_or_else(|| "-".to_string());
            let entry = by_tool.entry(key).or_insert((0, 0, 0, 0));
            entry.0 += 1;
            match a.success {
                Some(true) => entry.1 += 1,
                Some(false) => entry.2 += 1,
                None => {}
            }
            entry.3 += a.tokens_used.unwrap_or(0);
        }

        let mut tools: Vec<_> = by_tool.keys().cloned().collect();
        tools.sort();
        let mut out = format!("**Tool Trace**\n\n- Events: {}\n", rows.len());
        if let Some(tn) = tool_name {
            out.push_str(&format!("- Filter tool: {}\n", tn));
        }
        out.push_str("\n**By Tool**");
        for t in &tools {
            let (calls, ok, err, tokens) = by_tool[t];
            out.push_str(&format!(
                "\n- {}: calls {}, ok {}, err {}, tokens {}",
                t, calls, ok, err, tokens
            ));
        }

        out.push_str("\n\n**Recent Events**");
        for (task_id, a) in &rows {
            let tool = a.tool_name.as_deref().unwrap_or("-");
            let ok = a
                .success
                .map(|v| if v { "ok" } else { "err" })
                .unwrap_or("n/a");
            let result = a
                .result
                .as_deref()
                .map(|r| Self::truncate(r, 120))
                .unwrap_or_default();
            if result.is_empty() {
                out.push_str(&format!(
                    "\n- {} task={} {} tool={} [{}] tokens={}",
                    a.created_at,
                    Self::short_id(task_id),
                    a.activity_type,
                    tool,
                    ok,
                    a.tokens_used.unwrap_or(0)
                ));
            } else {
                out.push_str(&format!(
                    "\n- {} task={} {} tool={} [{}] tokens={} => {}",
                    a.created_at,
                    Self::short_id(task_id),
                    a.activity_type,
                    tool,
                    ok,
                    a.tokens_used.unwrap_or(0),
                    result
                ));
            }
        }

        Ok(out)
    }
}

#[derive(Deserialize)]
struct GoalTraceArgs {
    action: String,
    #[serde(default, alias = "goal_id_v3")]
    goal_id: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    max_tasks: Option<usize>,
    #[serde(default)]
    max_activities_per_task: Option<usize>,
}

/// The tool's one description: the schema the model reads and
/// `description()` both use it.
const TOOL_DESCRIPTION: &str =
    "Goal/task traces and tool-call timelines. Omit goal_id for recent-goal list.";

fn goal_trace_schema() -> Value {
    json!({
        "name": "goal_trace",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["goal_trace", "tool_trace"]
                },
                "goal_id": {
                    "type": "string",
                    "description": "Goal or mandate ID. Required for goal_trace; optional for tool_trace with task_id"
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (tool_trace)"
                },
                "tool_name": {
                    "type": "string",
                    "description": "Tool filter (tool_trace)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max events; default 30, max 200"
                },
                "max_tasks": {
                    "type": "integer",
                    "description": "Max tasks; default 20, max 100"
                },
                "max_activities_per_task": {
                    "type": "integer",
                    "description": "Max per task; default 6, max 20"
                }
            },
            "required": ["action"],
            "additionalProperties": false
        }
    })
}

#[async_trait]
impl Tool for GoalTraceTool {
    fn name(&self) -> &str {
        "goal_trace"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        goal_trace_schema()
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: GoalTraceArgs = serde_json::from_str(arguments)?;
            match args.action.as_str() {
                "goal_trace" => {
                    if let Some(goal_id) = args.goal_id.as_deref() {
                        self.goal_trace(
                            goal_id,
                            args.max_tasks.unwrap_or(20),
                            args.max_activities_per_task.unwrap_or(6),
                        )
                        .await
                    } else {
                        self.recent_goal_summary(8).await
                    }
                }
                "tool_trace" => {
                    if args.goal_id.is_none() && args.task_id.is_none() {
                        self.recent_goal_summary(8).await
                    } else {
                        self.tool_trace(
                            args.goal_id.as_deref(),
                            args.task_id.as_deref(),
                            args.tool_name.as_deref(),
                            args.limit.unwrap_or(30),
                        )
                        .await
                    }
                }
                other => Ok(format!(
                    "Unknown action: '{}'. Use goal_trace or tool_trace.",
                    other
                )),
            }
        }
        .await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for GoalTraceTool {
    fn dispatch_facts(&self) -> aidaemon_tools::semantics::ToolDispatchFacts {
        aidaemon_tools::semantics::ToolDispatchFacts {
            evidence: aidaemon_tools::semantics::EvidenceSource::GoalTrace,
            semantic_scope: Some(aidaemon_tools::semantics::ToolSemanticScope::GoalState),
            ..Default::default()
        }
    }

    /// Goal traces replay past tool results and messages, whoever wrote them.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Untrusted
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
}
