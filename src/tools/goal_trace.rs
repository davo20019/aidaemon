use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{GoalV3, StateStore, TaskActivityV3, Tool, ToolCapabilities};

pub struct GoalTraceTool {
    state: Arc<dyn StateStore>,
}

impl GoalTraceTool {
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

        let mut candidates = self.state.get_active_goals_v3().await?;
        for g in self.state.get_scheduled_goals_v3().await? {
            if !candidates.iter().any(|x| x.id == g.id) {
                candidates.push(g);
            }
        }

        let mut matches: Vec<&GoalV3> = candidates
            .iter()
            .filter(|g| g.id.starts_with(trimmed))
            .collect();
        if matches.is_empty() {
            anyhow::bail!("Goal not found: {}", trimmed);
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
            .map(|g| format!("{} ({})", Self::short_id(&g.id), g.status))
            .collect::<Vec<_>>()
            .join(", ");
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

    async fn gather_goal_activities(
        &self,
        goal_id: &str,
    ) -> anyhow::Result<Vec<(String, TaskActivityV3)>> {
        let mut rows = Vec::new();
        let tasks = self.state.get_tasks_for_goal_v3(goal_id).await?;
        for t in &tasks {
            let activities = self.state.get_task_activities_v3(&t.id).await?;
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
        let resolved_goal_id = match self.resolve_goal_id_v3(goal_id_input).await {
            Ok(id) => id,
            Err(e) => return Ok(e.to_string()),
        };
        let Some(goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
            return Ok(format!("Goal not found: {}", resolved_goal_id));
        };

        let mut tasks = self.state.get_tasks_for_goal_v3(&goal.id).await?;
        tasks.sort_by(|a, b| a.created_at.cmp(&b.created_at));

        let mut status_counts: HashMap<String, usize> = HashMap::new();
        let mut retries_total = 0i32;
        let mut tokens_total = 0i64;
        for t in &tasks {
            *status_counts.entry(t.status.clone()).or_insert(0) += 1;
            retries_total += t.retry_count;
            let activities = self.state.get_task_activities_v3(&t.id).await?;
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

        out.push_str("\n**Task Timeline**");
        for t in tasks.iter().take(max_tasks.clamp(1, 100)) {
            let activities = self.state.get_task_activities_v3(&t.id).await?;
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
            } else if t.status == "blocked" {
                if let Some(blocker) = &t.blocker {
                    out.push_str(&format!("\n  blocker: {}", Self::truncate(blocker, 180)));
                }
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
        let mut rows: Vec<(String, TaskActivityV3)> = if let Some(task_id) = task_id {
            let activities = self.state.get_task_activities_v3(task_id).await?;
            activities
                .into_iter()
                .map(|a| (task_id.to_string(), a))
                .collect::<Vec<_>>()
        } else {
            let goal_input = goal_id_input
                .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' or 'task_id' is required"))?;
            let resolved_goal_id = match self.resolve_goal_id_v3(goal_input).await {
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
    #[serde(default)]
    goal_id_v3: Option<String>,
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

#[async_trait]
impl Tool for GoalTraceTool {
    fn name(&self) -> &str {
        "goal_trace"
    }

    fn description(&self) -> &str {
        "Inspect goal/task execution traces and tool-call timelines without terminal/sqlite forensics"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "goal_trace",
            "description": "Execution observability for goals and tools. Use this for what-happened forensics instead of terminal/sqlite log digging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["goal_trace", "tool_trace"],
                        "description": "Trace view type"
                    },
                    "goal_id_v3": {
                        "type": "string",
                        "description": "Goal ID (full or unique prefix). Required for goal_trace, optional for tool_trace when task_id is provided."
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID for task-scoped tool trace"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Optional tool name filter for tool_trace"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max events for tool_trace (default 30, max 200)"
                    },
                    "max_tasks": {
                        "type": "integer",
                        "description": "Max tasks to include in goal_trace (default 20, max 100)"
                    },
                    "max_activities_per_task": {
                        "type": "integer",
                        "description": "Max activity points per task in goal_trace (default 6, max 20)"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: GoalTraceArgs = serde_json::from_str(arguments)?;
        match args.action.as_str() {
            "goal_trace" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for goal_trace"))?;
                self.goal_trace(
                    goal_id,
                    args.max_tasks.unwrap_or(20),
                    args.max_activities_per_task.unwrap_or(6),
                )
                .await
            }
            "tool_trace" => {
                self.tool_trace(
                    args.goal_id_v3.as_deref(),
                    args.task_id.as_deref(),
                    args.tool_name.as_deref(),
                    args.limit.unwrap_or(30),
                )
                .await
            }
            other => Ok(format!(
                "Unknown action: '{}'. Use goal_trace or tool_trace.",
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
    use crate::traits::{GoalV3, TaskV3};

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
    async fn goal_trace_includes_task_and_tool_sequence() {
        let state = setup_state().await;
        let tool = GoalTraceTool::new(state.clone());

        let goal = GoalV3::new_finite("Trace this goal", "user-session");
        let goal_id = goal.id.clone();
        state.create_goal_v3(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = TaskV3 {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.clone(),
            description: "Inspect state".to_string(),
            status: "failed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: Some("agent-1".to_string()),
            context: None,
            result: None,
            error: Some("command timed out".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 1,
            max_retries: 3,
            created_at: now.clone(),
            started_at: Some(now.clone()),
            completed_at: Some(now.clone()),
        };
        state.create_task_v3(&task).await.unwrap();

        state
            .log_task_activity_v3(&TaskActivityV3 {
                id: 0,
                task_id: task.id.clone(),
                activity_type: "tool_call".to_string(),
                tool_name: Some("terminal".to_string()),
                tool_args: Some("{\"command\":\"ls\"}".to_string()),
                result: None,
                success: None,
                tokens_used: Some(0),
                created_at: now.clone(),
            })
            .await
            .unwrap();
        state
            .log_task_activity_v3(&TaskActivityV3 {
                id: 0,
                task_id: task.id.clone(),
                activity_type: "tool_result".to_string(),
                tool_name: Some("terminal".to_string()),
                tool_args: None,
                result: Some("timed out".to_string()),
                success: Some(false),
                tokens_used: Some(12),
                created_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "goal_trace",
                    "goal_id_v3": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Goal Trace"));
        assert!(result.contains("terminal(err)"));
        assert!(result.contains("retry=1/3"));
    }

    #[tokio::test]
    async fn tool_trace_filters_by_tool_name() {
        let state = setup_state().await;
        let tool = GoalTraceTool::new(state.clone());

        let goal = GoalV3::new_finite("Trace tool filter", "user-session");
        state.create_goal_v3(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = TaskV3 {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Do work".to_string(),
            status: "completed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some("done".to_string()),
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: now.clone(),
            started_at: Some(now.clone()),
            completed_at: Some(now.clone()),
        };
        state.create_task_v3(&task).await.unwrap();

        state
            .log_task_activity_v3(&TaskActivityV3 {
                id: 0,
                task_id: task.id.clone(),
                activity_type: "tool_result".to_string(),
                tool_name: Some("web_fetch".to_string()),
                tool_args: None,
                result: Some("ok".to_string()),
                success: Some(true),
                tokens_used: Some(8),
                created_at: now.clone(),
            })
            .await
            .unwrap();
        state
            .log_task_activity_v3(&TaskActivityV3 {
                id: 0,
                task_id: task.id.clone(),
                activity_type: "tool_result".to_string(),
                tool_name: Some("terminal".to_string()),
                tool_args: None,
                result: Some("ok".to_string()),
                success: Some(true),
                tokens_used: Some(5),
                created_at: now,
            })
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "tool_trace",
                    "goal_id_v3": goal.id,
                    "tool_name": "web_fetch"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Tool Trace"));
        assert!(result.contains("Filter tool: web_fetch"));
        assert!(result.contains("web_fetch: calls 1"));
    }
}
