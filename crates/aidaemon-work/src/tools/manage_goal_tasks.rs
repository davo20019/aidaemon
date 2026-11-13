use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use crate::traits::{
    HandoffArtifact, ManageGoalTasksStore, Task, TaskAttemptPatch, TaskHandoff, ToolCallContext,
    ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolRole,
    semantics_for_exact_read_actions,
};
use aidaemon_tools::OutcomeTool as Tool;

/// Tool for task leads to manage tasks within their assigned goal.
pub struct ManageGoalTasksTool {
    goal_id: String,
    state: Arc<dyn ManageGoalTasksStore>,
    goal_run_id: Option<String>,
}

impl ManageGoalTasksTool {
    pub fn new(goal_id: String, state: Arc<dyn ManageGoalTasksStore>) -> Self {
        Self {
            goal_id,
            state,
            goal_run_id: None,
        }
    }

    pub fn with_goal_run_id(mut self, goal_run_id: Option<String>) -> Self {
        self.goal_run_id = goal_run_id;
        self
    }

    fn task_in_scope(&self, task: &Task) -> bool {
        task.goal_id == self.goal_id
    }

    async fn scoped_tasks(&self) -> anyhow::Result<Vec<Task>> {
        let tasks = if let Some(run_id) = self.goal_run_id.as_deref() {
            self.state.get_tasks_for_goal_run(run_id).await?
        } else {
            self.state.get_tasks_for_goal(&self.goal_id).await?
        };
        let root_task_ids = self
            .state
            .get_goal_runs(&self.goal_id)
            .await?
            .into_iter()
            .filter(|run| {
                self.goal_run_id
                    .as_deref()
                    .is_none_or(|run_id| run.id == run_id)
            })
            .filter_map(|run| run.root_task_id)
            .collect::<std::collections::HashSet<_>>();
        Ok(tasks
            .into_iter()
            .filter(|task| self.task_in_scope(task) && !root_task_ids.contains(&task.id))
            .collect())
    }

    async fn task_id_in_scope(&self, task_id: &str) -> anyhow::Result<bool> {
        Ok(self
            .scoped_tasks()
            .await?
            .iter()
            .any(|task| task.id == task_id))
    }

    fn out_of_scope_message(&self, task_id: &str) -> String {
        format!(
            "Task {task_id} belongs to an earlier run and cannot be changed by the current run. Use manage_goal_tasks(action=\"list_tasks\") for current task IDs."
        )
    }

    async fn task_not_found_message(&self, task_id: &str) -> String {
        let list = self
            .list_tasks()
            .await
            .unwrap_or_else(|_| "(failed to list tasks)".to_string());
        format!(
            "Task not found: {}.\n\n\
             Use manage_goal_tasks(action=\"list_tasks\") to see valid task IDs, then retry with an existing task_id.\n\n\
             {}",
            task_id, list
        )
    }

    /// Resolve a potentially short task ID to the full UUID by prefix-matching
    /// against all tasks in the current goal. Returns the full ID if exactly one
    /// task matches the prefix, otherwise returns the original string unchanged.
    async fn resolve_task_id(&self, task_id: &str) -> String {
        // If it's already a full UUID (36 chars with dashes), skip resolution
        if task_id.len() >= 36 {
            return task_id.to_string();
        }
        // Try exact lookup first
        if let Ok(Some(task)) = self.state.get_task(task_id).await
            && self.task_in_scope(&task)
            && self.task_id_in_scope(task_id).await.unwrap_or(false)
        {
            return task_id.to_string();
        }
        // Prefix-match against all tasks in this goal
        if let Ok(tasks) = self.scoped_tasks().await {
            let matches: Vec<&Task> = tasks.iter().filter(|t| t.id.starts_with(task_id)).collect();
            if matches.len() == 1 {
                return matches[0].id.clone();
            }
        }
        task_id.to_string()
    }

    fn truncate_result_for_tool_output(text: &str, max_chars: usize) -> String {
        let truncated: String = text.chars().take(max_chars).collect();
        if text.chars().count() > max_chars {
            format!("{truncated}...")
        } else {
            truncated
        }
    }

    async fn build_completed_task_result_excerpt(&self) -> anyhow::Result<Option<String>> {
        let tasks = self.scoped_tasks().await?;
        if tasks.is_empty() {
            return Ok(None);
        }

        let mut successful: Vec<&Task> = tasks
            .iter()
            .filter(|t| t.completed_successfully())
            .filter(|t| t.result.as_deref().is_some_and(|r| !r.trim().is_empty()))
            .collect();
        if successful.is_empty() {
            return Ok(None);
        }

        successful.sort_by(|a, b| {
            let a_key = a.completed_at.as_deref().unwrap_or(a.created_at.as_str());
            let b_key = b.completed_at.as_deref().unwrap_or(b.created_at.as_str());
            a_key
                .cmp(b_key)
                .then_with(|| a.task_order.cmp(&b.task_order))
                .then_with(|| a.id.cmp(&b.id))
        });

        let successful_count = successful.len();
        let total_count = tasks.len();
        const MAX_INCLUDED_RESULTS: usize = 3;
        const MAX_RESULT_CHARS_PER_TASK: usize = 700;
        let mut selected: Vec<&Task> = successful
            .iter()
            .rev()
            .take(MAX_INCLUDED_RESULTS)
            .copied()
            .collect();
        selected.reverse();

        let mut excerpt = String::new();
        if successful_count > 1 {
            excerpt.push_str(&format!(
                "{successful_count}/{total_count} tasks completed.\n\n"
            ));
        }
        if selected.len() == 1 {
            let task = selected[0];
            excerpt.push_str("Task result:\n");
            excerpt.push_str(&format!(
                "**{}**\n{}",
                task.description,
                Self::truncate_result_for_tool_output(
                    task.result.as_deref().unwrap_or(""),
                    MAX_RESULT_CHARS_PER_TASK
                )
            ));
        } else {
            excerpt.push_str("Recent task results:\n\n");
            for (idx, task) in selected.iter().enumerate() {
                if idx > 0 {
                    excerpt.push_str("\n\n");
                }
                excerpt.push_str(&format!(
                    "**{}**\n{}",
                    task.description,
                    Self::truncate_result_for_tool_output(
                        task.result.as_deref().unwrap_or(""),
                        MAX_RESULT_CHARS_PER_TASK
                    )
                ));
            }

            let omitted = successful_count.saturating_sub(selected.len());
            if omitted > 0 {
                let suffix = if omitted == 1 { "" } else { "s" };
                excerpt.push_str(&format!(
                    "\n\n(+{} earlier completed task result{} omitted)",
                    omitted, suffix
                ));
            }
        }
        Ok(Some(excerpt))
    }
}

pub use crate::goals::{is_terminal_task_status, tasks_satisfy_goal_completion};

#[derive(Clone, Deserialize)]
struct ManageGoalTasksArgs {
    action: String,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    priority: Option<String>,
    #[serde(default)]
    task_order: Option<i32>,
    #[serde(default)]
    parallel_group: Option<String>,
    #[serde(default)]
    depends_on: Option<Vec<String>>,
    #[serde(default)]
    idempotent: Option<bool>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    result: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    worker_profile: Option<String>,
    #[serde(default)]
    workspace_policy: Option<String>,
    #[serde(default)]
    artifacts: Option<Vec<String>>,
    #[serde(default)]
    verification: Option<Vec<String>>,
    #[serde(default)]
    remaining_risk: Option<String>,
    #[serde(default)]
    next_step: Option<String>,
    /// The dispatcher's mandate fence (from the call's facts, never an
    /// argument).
    #[serde(skip)]
    fence_mandate_id: Option<String>,
    #[serde(skip)]
    fence_mandate_version: Option<i64>,
    #[serde(skip)]
    fence_goal_run_id: Option<String>,
    #[serde(skip)]
    fence_task_attempt_id: Option<String>,
}

#[async_trait]
impl Tool for ManageGoalTasksTool {
    fn name(&self) -> &str {
        "manage_goal_tasks"
    }

    fn description(&self) -> &str {
        "Manage tasks within your assigned goal. Use create_task to break work into steps, \
         claim_task before spawning an executor, list_tasks to check progress, update_task to record results."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_goal_tasks",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create_task", "list_tasks", "update_task", "claim_task", "retry_task", "resolve_blocker", "complete_goal", "fail_goal"]
                    },
                    "task_id": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    },
                    "task_order": {
                        "type": "integer"
                    },
                    "parallel_group": {
                        "type": "string"
                    },
                    "depends_on": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "idempotent": {
                        "type": "boolean"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "running", "completed", "failed", "blocked", "skipped", "superseded"],
                        "description": "New status (for update_task). Use superseded only when another named task replaced this task's required work."
                    },
                    "result": {
                        "type": "string"
                    },
                    "error": {
                        "type": "string"
                    },
                    "summary": {
                        "type": "string"
                    },
                    "agent_id": {
                        "type": "string"
                    },
                    "worker_profile": {
                        "type": "string",
                        "description": "Worker profile ID."
                    },
                    "workspace_policy": {
                        "type": "string",
                        "enum": ["shared", "isolated", "worktree"],
                        "description": "shared, isolated, or Git worktree."
                    },
                    "artifacts": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Produced paths, URLs, commits, or messages."
                    },
                    "verification": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Checks and outcomes."
                    },
                    "remaining_risk": {
                        "type": "string",
                        "description": "Residual risk."
                    },
                    "next_step": {
                        "type": "string",
                        "description": "Next action."
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let facts = facts.clone();

        self.run(arguments, &facts)
            .await
            .map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl ManageGoalTasksTool {
    /// Run one call with the dispatcher's `facts`.
    pub(crate) async fn run(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<String> {
        let mut args: ManageGoalTasksArgs = serde_json::from_str(arguments)?;
        // The mandate fence comes from the dispatcher, never the arguments.
        args.fence_mandate_id = facts.mandate_id.clone();
        args.fence_mandate_version = facts.mandate_version;
        args.fence_goal_run_id = facts.goal_run_id.clone();
        args.fence_task_attempt_id = facts.task_attempt_id.clone();

        let mandate_context = args.fence_mandate_id.is_some()
            || self
                .state
                .get_mandate_for_goal(&self.goal_id)
                .await?
                .is_some();
        if mandate_context
            && !matches!(
                args.action.as_str(),
                "create_task" | "claim_task" | "list_tasks"
            )
        {
            return Ok(format!(
                "Mandate task control rejects '{}': only exact-run create_task/claim_task and read-only list_tasks are supported; executor outcomes are finalized through their own leases.",
                args.action
            ));
        }

        match args.action.as_str() {
            "create_task" => self.create_task(&args).await,
            "list_tasks" => self.list_tasks().await,
            "update_task" => self.update_task(&args).await,
            "claim_task" => self.claim_task(&args).await,
            "retry_task" => self.retry_task(&args).await,
            "resolve_blocker" => self.resolve_blocker(&args).await,
            "complete_goal" => self.complete_goal(&args).await,
            "fail_goal" => self.fail_goal(&args).await,
            other => Ok(format!(
                "Unknown action: {}. Use: create_task, list_tasks, update_task, claim_task, retry_task, resolve_blocker, complete_goal, fail_goal",
                other
            )),
        }
    }
}

impl crate::traits::ToolSemantics for ManageGoalTasksTool {
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            task_lead_delegation: true,
            same_tool_streak: crate::traits::SameToolStreak::Versatile,
            goal_context: crate::traits::GoalContext::MandateRun,
            ..Default::default()
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Management
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        semantics_for_exact_read_actions(arguments, &["list_tasks"], ToolMutationEffects::NONE)
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    /// `list_tasks`, `claim_task` and an `update_task` that does not put a
    /// task back to `pending` only write the daemon's own task records.
    /// Creating, retrying or unblocking a task queues work, and completing
    /// or failing a goal can alert the owner, so those stay undeclared.
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        let args = serde_json::from_str::<Value>(arguments).ok();
        let field = |key: &str| {
            args.as_ref()
                .and_then(|args| args.get(key))
                .and_then(Value::as_str)
        };
        let internal = match field("action") {
            Some("list_tasks" | "claim_task") => true,
            Some("update_task") => field("status") != Some("pending"),
            _ => false,
        };
        if internal {
            aidaemon_core::CallEffects::none()
        } else {
            aidaemon_core::CallEffects::unknown()
        }
    }
}

/// Check if all dependencies for a task are completed. Returns Ok(()) if all are met,
/// or an error message string describing which dependencies are unmet.
async fn check_dependencies_met(
    state: &dyn ManageGoalTasksStore,
    task: &Task,
) -> Result<(), String> {
    let tasks = state
        .get_tasks_for_goal(&task.goal_id)
        .await
        .map_err(|error| format!("could not read task dependency graph: {error}"))?;
    let graph = crate::traits::task_execution_graph(&tasks)?;
    if let Some(dependency) = graph.unresolved_dependencies(&task.id).first() {
        return Err(format!(
            "dependency {} is not completed or otherwise accepted (state: {:?})",
            dependency.id, dependency.state
        ));
    }
    Ok(())
}

/// Compress older task result entries to save space.
/// Keeps last 10 entries in full detail, compresses older ones to one-line summaries.
fn compress_old_entries(entries: &mut [Value]) {
    if entries.len() <= 10 {
        return;
    }
    let keep_full = entries.len() - 10;
    for entry in entries.iter_mut().take(keep_full) {
        if let Some(obj) = entry.as_object() {
            let task_id = obj.get("task_id").and_then(|v| v.as_str()).unwrap_or("?");
            let desc = obj
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let short_id = &task_id[..task_id.len().min(8)];
            *entry = json!(format!("{}: {} (completed)", short_id, desc));
        }
    }
}

/// Validate that adding a new task with the given dependencies won't create a cycle.
/// Uses the shared typed execution graph so scheduling and validation enforce
/// the same dependency and cycle invariants.
fn validate_no_cycles(
    existing: &[Task],
    new_task_id: &str,
    new_deps: &[String],
) -> Result<(), String> {
    let Some(template) = existing.first() else {
        return if new_deps.is_empty() {
            Ok(())
        } else {
            Err("Dependencies cannot be added before their tasks exist".to_string())
        };
    };
    let mut tasks = existing.to_vec();
    let mut candidate = template.clone();
    candidate.id = new_task_id.to_string();
    candidate.status = "pending".to_string();
    candidate.depends_on = Some(
        serde_json::to_string(new_deps)
            .map_err(|error| format!("Could not encode dependencies: {error}"))?,
    );
    tasks.push(candidate);
    crate::traits::task_execution_graph(&tasks).map(|_| ())
}

impl ManageGoalTasksTool {
    async fn create_task(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let persisted_mandate = self.state.get_mandate_for_goal(&self.goal_id).await?;
        let mandate_context = args.fence_mandate_id.is_some() || persisted_mandate.is_some();
        let mandate_fence = if mandate_context {
            let mandate_id = args.fence_mandate_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task creation is missing its dispatcher-owned mandate id")
            })?;
            let mandate_version = args.fence_mandate_version.ok_or_else(|| {
                anyhow::anyhow!(
                    "mandate task creation is missing its dispatcher-owned policy version"
                )
            })?;
            let goal_run_id = args.fence_goal_run_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task creation is missing its dispatcher-owned goal run")
            })?;
            let task_attempt_id = args.fence_task_attempt_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "mandate task creation is missing its dispatcher-owned task attempt"
                )
            })?;
            anyhow::ensure!(
                self.goal_run_id.as_deref() == Some(goal_run_id),
                "mandate task creation run does not match the task-lead binding"
            );
            if let Some(mandate) = persisted_mandate.as_ref() {
                anyhow::ensure!(
                    mandate.id == mandate_id && mandate.version == mandate_version,
                    "mandate task creation policy binding is stale"
                );
            }
            Some((mandate_id, mandate_version, goal_run_id, task_attempt_id))
        } else {
            None
        };
        // Mandate work placement is runtime-owned: the fenced dispatcher picks
        // the worker and workspace. The schema still offers these hints, so a
        // task lead that fills them in must not have its committed ACT
        // rejected; the hints are dropped instead of honored.
        let without_hints;
        let args = if mandate_fence.is_some() {
            without_hints = ManageGoalTasksArgs {
                workspace_policy: None,
                worker_profile: None,
                ..args.clone()
            };
            &without_hints
        } else {
            args
        };

        let description = args
            .description
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("create_task requires 'description'"))?;
        if let Some(profile_id) = args.worker_profile.as_deref()
            && self.state.get_worker_profile(profile_id).await?.is_none()
        {
            return Ok(format!(
                "Cannot create task: worker profile {profile_id} not found"
            ));
        }
        if let Some(policy) = args.workspace_policy.as_deref() {
            anyhow::ensure!(
                matches!(policy, "shared" | "isolated" | "worktree"),
                "workspace_policy must be shared, isolated, or worktree"
            );
        }

        let now = chrono::Utc::now().to_rfc3339();
        let task_id = uuid::Uuid::new_v4().to_string();

        // Validate dependencies don't create cycles
        if let Some(ref dep_ids) = args.depends_on
            && !dep_ids.is_empty()
        {
            let existing = self.scoped_tasks().await?;
            if let Err(reason) = validate_no_cycles(&existing, &task_id, dep_ids) {
                return Ok(format!("Cannot create task: {}", reason));
            }
        }

        let task = Task {
            id: task_id,
            goal_id: self.goal_id.clone(),
            description: description.to_string(),
            status: "pending".to_string(),
            priority: args
                .priority
                .clone()
                .unwrap_or_else(|| "medium".to_string()),
            task_order: args.task_order.unwrap_or(0),
            parallel_group: args.parallel_group.clone(),
            depends_on: args
                .depends_on
                .as_ref()
                .map(|v| serde_json::to_string(v).unwrap_or_default()),
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: if mandate_fence.is_some() {
                false
            } else {
                args.idempotent.unwrap_or(false)
            },
            retry_count: 0,
            max_retries: if mandate_fence.is_some() { 0 } else { 3 },
            created_at: now,
            started_at: None,
            completed_at: None,
        };

        if let Some((mandate_id, mandate_version, goal_run_id, task_attempt_id)) = mandate_fence {
            const MAX_MANDATE_ACTION_TASKS_PER_RUN: i64 = 16;
            if !self
                .state
                .create_mandate_task_from_attempt(
                    &task,
                    mandate_id,
                    mandate_version,
                    goal_run_id,
                    task_attempt_id,
                    MAX_MANDATE_ACTION_TASKS_PER_RUN,
                )
                .await?
            {
                return Ok(
                    "Cannot create mandate task: authority/run/attempt changed or the per-run task cap was reached. Reconsider in a fresh cycle."
                        .to_string(),
                );
            }
        } else {
            self.state.create_task(&task).await?;
        }
        if mandate_fence.is_none() {
            if let Some(profile_id) = args.worker_profile.as_deref()
                && !self
                    .state
                    .assign_task_worker_profile(&task.id, profile_id, "task-lead", None)
                    .await?
            {
                anyhow::bail!("worker profile '{}' is unavailable", profile_id);
            }
            if let Some(policy) = args.workspace_policy.as_deref()
                && !self
                    .state
                    .set_task_workspace_policy(&task.id, policy)
                    .await?
            {
                anyhow::bail!("workspace policy could not be applied to the new task");
            }
        }
        info!(goal_id = %self.goal_id, task_id = %task.id, "Created task");

        Ok(format!(
            "Created task {} (order: {}, priority: {}): {}",
            task.id, task.task_order, task.priority, task.description
        ))
    }

    async fn list_tasks(&self) -> anyhow::Result<String> {
        let tasks = self.scoped_tasks().await?;

        if tasks.is_empty() {
            return Ok(format!("No tasks for goal {}", self.goal_id));
        }

        let mut output = format!("Tasks for goal {} ({} total):\n", self.goal_id, tasks.len());
        for task in &tasks {
            let short_id = &task.id[..8.min(task.id.len())];

            // Build detail parts
            let mut details = vec![
                format!("order: {}", task.task_order),
                format!("status: {}", task.status),
            ];
            if let Some(ref pg) = task.parallel_group {
                details.push(format!("group: {}", pg));
            }
            if let Some(ref deps) = task.depends_on
                && let Ok(dep_ids) = serde_json::from_str::<Vec<String>>(deps)
                && !dep_ids.is_empty()
            {
                let short_deps: Vec<String> = dep_ids
                    .iter()
                    .map(|d| d[..8.min(d.len())].to_string())
                    .collect();
                details.push(format!("deps: [{}]", short_deps.join(", ")));
            }
            if let Some(ref aid) = task.agent_id {
                details.push(format!("agent: {}", aid));
            }
            if task.idempotent && task.max_retries > 0 {
                details.push(format!(
                    "retries: {}/{}",
                    task.retry_count, task.max_retries
                ));
            }
            if let Ok(journal) = self.state.get_task_journal(&task.id, 12).await
                && let Some(entry) = journal.iter().find(|entry| entry.actor_type == "human")
            {
                let end = aidaemon_tools::text::floor_char_boundary(&entry.body, 120);
                details.push(format!(
                    "latest human {}: {}",
                    entry.entry_type,
                    &entry.body[..end]
                ));
            }

            let result_suffix = task
                .result
                .as_deref()
                .map(|r| {
                    let end = aidaemon_tools::text::floor_char_boundary(r, 200);
                    format!(" → {}", &r[..end])
                })
                .unwrap_or_default();

            output.push_str(&format!(
                "- [{}] {} ({}){}\n",
                short_id,
                task.description,
                details.join(", "),
                result_suffix,
            ));
        }

        Ok(output)
    }

    async fn update_task(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let raw_task_id = args
            .task_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("update_task requires 'task_id'"))?;
        let task_id = self.resolve_task_id(raw_task_id).await;
        let task_id = task_id.as_str();

        let Some(mut task) = self.state.get_task(task_id).await? else {
            return Ok(self.task_not_found_message(task_id).await);
        };
        let original_status = task.status.clone();

        if task.goal_id != self.goal_id {
            anyhow::bail!("Task {} does not belong to goal {}", task_id, self.goal_id);
        }
        if !self.task_id_in_scope(task_id).await? {
            return Ok(self.out_of_scope_message(task_id));
        }

        if original_status == "blocked"
            && args
                .status
                .as_deref()
                .is_some_and(|status| status != "blocked")
        {
            return Ok(format!(
                "Blocked: task {} is waiting on a recorded blocker and cannot transition directly to another status. Use resolve_blocker with the concrete resolution, or retry_task, before completing it.",
                task_id
            ));
        }

        // Dependency enforcement: prevent moving to "running" or "claimed" if deps unmet
        if let Some(ref new_status) = args.status
            && (new_status == "running" || new_status == "claimed")
            && let Err(reason) = check_dependencies_met(self.state.as_ref(), &task).await
        {
            return Ok(format!(
                "Cannot set task {} to {}: {}",
                task_id, new_status, reason
            ));
        }

        if let Some(status) = &args.status {
            task.status = status.clone();
            if matches!(
                status.as_str(),
                "completed" | "failed" | "blocked" | "skipped" | "superseded"
            ) {
                task.completed_at = Some(chrono::Utc::now().to_rfc3339());
            }
            if status == "running" {
                task.started_at = Some(chrono::Utc::now().to_rfc3339());
            }
        }
        if let Some(result) = &args.result {
            task.result = Some(result.clone());
        }
        // A completed task's outcome is success. Error text left by a prior
        // attempt (e.g. a failed specialist handoff the lead then finished
        // directly), or supplied alongside the completion, is attempt history
        // rather than an open failure. Keeping it in `error` made the task
        // read as "completed but failed" and vetoed the whole run, so it is
        // carried as handoff risk instead.
        let attempt_history = if task.status == "completed" {
            let note = args
                .error
                .clone()
                .or_else(|| task.error.take())
                .filter(|note| !note.trim().is_empty());
            task.error = None;
            note
        } else {
            if let Some(error) = &args.error {
                task.error = (!error.trim().is_empty()).then(|| error.clone());
            }
            None
        };
        let remaining_risk = match (args.remaining_risk.clone(), attempt_history) {
            (Some(risk), Some(note)) => Some(format!("{risk}\nPrior attempt: {note}")),
            (Some(risk), None) => Some(risk),
            (None, Some(note)) => Some(format!("Prior attempt: {note}")),
            (None, None) => None,
        };

        let fenced_status = matches!(
            task.status.as_str(),
            "running" | "completed" | "failed" | "blocked" | "cancelled"
        );
        let mut attempt = self.state.get_current_task_attempt(task_id).await?;
        if attempt.is_none() && original_status == "pending" && fenced_status {
            attempt = self
                .state
                .claim_task_with_lease(
                    task_id,
                    args.agent_id.as_deref().unwrap_or("task-lead"),
                    args.worker_profile.as_deref().or(Some("profile-task-lead")),
                    180,
                )
                .await?;
        }
        if let Some(attempt) = attempt {
            if !fenced_status {
                return Ok(
                    "A claimed task cannot be skipped or superseded by an unfenced update; finish or cancel its current attempt first."
                        .to_string(),
                );
            }
            let terminal = matches!(
                task.status.as_str(),
                "completed" | "failed" | "blocked" | "cancelled"
            );
            let handoff = terminal.then(|| TaskHandoff {
                id: uuid::Uuid::new_v4().to_string(),
                task_id: task_id.to_string(),
                attempt_id: attempt.id.clone(),
                summary: args
                    .summary
                    .clone()
                    .or_else(|| args.result.clone())
                    .or_else(|| args.error.clone())
                    .unwrap_or_else(|| format!("Task {}", task.status)),
                artifacts: args
                    .artifacts
                    .clone()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|reference| HandoffArtifact {
                        kind: if reference.starts_with("http://")
                            || reference.starts_with("https://")
                        {
                            "url".to_string()
                        } else {
                            "path".to_string()
                        },
                        reference,
                        digest: None,
                        metadata: None,
                    })
                    .collect(),
                verification: args.verification.clone().unwrap_or_default(),
                remaining_risk: remaining_risk.clone(),
                next_step: args.next_step.clone(),
                created_at: chrono::Utc::now().to_rfc3339(),
            });
            let patch = TaskAttemptPatch {
                status: task.status.clone(),
                result: task.result.clone(),
                error: task.error.clone(),
                blocker: (task.status == "blocked").then(|| {
                    args.error
                        .clone()
                        .or_else(|| args.remaining_risk.clone())
                        .or_else(|| args.result.clone())
                        .unwrap_or_else(|| "Task is blocked.".to_string())
                }),
                context: task.context.clone(),
                handoff,
            };
            if !self
                .state
                .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
                .await?
            {
                return Ok(
                    "Update rejected because the task lease is no longer current.".to_string(),
                );
            }
            task = self
                .state
                .get_task(task_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Task disappeared after update"))?;
        } else {
            self.state.update_task(&task).await?;
        }
        info!(goal_id = %self.goal_id, task_id, status = %task.status, "Updated task");

        // Accumulate context when a task completes with a result
        if task.status == "completed"
            && task.result.is_some()
            && let Err(e) = self.accumulate_goal_context(&task).await
        {
            tracing::warn!(goal_id = %self.goal_id, "Failed to accumulate goal context: {}", e);
        }

        Ok(format!(
            "Updated task {} → status: {}",
            task_id, task.status
        ))
    }

    async fn claim_task(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let raw_task_id = args
            .task_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("claim_task requires 'task_id'"))?;
        let task_id = self.resolve_task_id(raw_task_id).await;
        let task_id = task_id.as_str();

        let Some(task) = self.state.get_task(task_id).await? else {
            return Ok(self.task_not_found_message(task_id).await);
        };

        if task.goal_id != self.goal_id {
            anyhow::bail!("Task {} does not belong to goal {}", task_id, self.goal_id);
        }
        if !self.task_id_in_scope(task_id).await? {
            return Ok(self.out_of_scope_message(task_id));
        }

        // Give the task lead a precise explanation while retaining the same
        // dependency check inside the atomic lease claim for race safety.
        if let Err(reason) = check_dependencies_met(self.state.as_ref(), &task).await {
            return Ok(format!("Cannot claim task {}: {}", task_id, reason));
        }

        let agent_id = args.agent_id.as_deref().unwrap_or("executor");
        if args.fence_mandate_id.is_some()
            || self
                .state
                .get_mandate_for_goal(&self.goal_id)
                .await?
                .is_some()
        {
            let mandate_id = args.fence_mandate_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task claim is missing its dispatcher-owned mandate id")
            })?;
            let mandate_version = args.fence_mandate_version.ok_or_else(|| {
                anyhow::anyhow!("mandate task claim is missing its dispatcher-owned policy version")
            })?;
            let goal_run_id = args.fence_goal_run_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task claim is missing its dispatcher-owned goal run")
            })?;
            let root_attempt_id = args.fence_task_attempt_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task claim is missing its dispatcher-owned root attempt")
            })?;
            anyhow::ensure!(
                self.goal_run_id.as_deref() == Some(goal_run_id),
                "mandate task claim run does not match the task-lead binding"
            );
            // worker_profile is a runtime-owned placement hint here; the
            // claim below never binds one, so a supplied value is ignored.
            return match self
                .state
                .claim_mandate_task_from_attempt(
                    task_id,
                    agent_id,
                    mandate_id,
                    mandate_version,
                    goal_run_id,
                    root_attempt_id,
                    180,
                )
                .await?
            {
                Some(attempt) => Ok(format!(
                    "Claimed mandate task {} for agent {} with attempt {} (no worker-profile or workspace binding)",
                    task_id, agent_id, attempt.id
                )),
                None => Ok(format!(
                    "Failed to claim mandate task {} — authority/run/attempt changed, dependencies are unmet, or another worker owns it",
                    task_id
                )),
            };
        }
        match self
            .state
            .claim_task_with_lease(task_id, agent_id, args.worker_profile.as_deref(), 180)
            .await?
        {
            Some(attempt) => {
                info!(
                    goal_id = %self.goal_id,
                    task_id,
                    agent_id,
                    attempt_id = %attempt.id,
                    "Claimed task"
                );
                Ok(format!(
                    "Claimed task {} for agent {} with attempt {} and profile {}",
                    task_id,
                    agent_id,
                    attempt.id,
                    attempt.worker_profile_id.as_deref().unwrap_or("default")
                ))
            }
            None => Ok(format!(
                "Failed to claim task {} — dependencies may be unmet, the profile may be at capacity, or another worker owns it",
                task_id
            )),
        }
    }

    async fn retry_task(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let raw_task_id = args
            .task_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("retry_task requires 'task_id'"))?;
        let task_id = self.resolve_task_id(raw_task_id).await;
        let task_id = task_id.as_str();

        let Some(task) = self.state.get_task(task_id).await? else {
            return Ok(self.task_not_found_message(task_id).await);
        };

        if task.goal_id != self.goal_id {
            anyhow::bail!("Task {} does not belong to goal {}", task_id, self.goal_id);
        }
        if !self.task_id_in_scope(task_id).await? {
            return Ok(self.out_of_scope_message(task_id));
        }
        if task.status != "failed" && task.status != "blocked" {
            return Ok(format!(
                "Cannot retry task {} — status is '{}'",
                task_id, task.status
            ));
        }
        if !task.idempotent {
            return Ok(format!(
                "Cannot retry task {} — not marked as idempotent",
                task_id
            ));
        }
        if task.retry_count >= task.max_retries {
            return Ok(format!(
                "Cannot retry task {} — max retries reached ({}/{})",
                task_id, task.retry_count, task.max_retries
            ));
        }

        if !self
            .state
            .retry_work_task(task_id, "task-lead", None)
            .await?
        {
            return Ok(format!(
                "Cannot retry task {} — it may still have a live attempt",
                task_id
            ));
        }
        let retry_count = task.retry_count + 1;

        info!(
            goal_id = %self.goal_id,
            task_id,
            retry_count,
            max_retries = task.max_retries,
            "Retried task"
        );

        Ok(format!(
            "Task {} reset to pending for retry ({}/{})",
            task_id, retry_count, task.max_retries
        ))
    }

    async fn complete_goal(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let summary = args
            .summary
            .as_deref()
            .unwrap_or("Goal completed successfully");

        let tasks = self.scoped_tasks().await?;
        if tasks.is_empty() {
            return Ok(
                "Blocked: do not call manage_goal_tasks(action=\"complete_goal\") before creating and completing concrete tasks for this goal. Create the task plan first, then complete the goal only after the task list is actually done."
                    .to_string(),
            );
        }
        let mut goal = self
            .state
            .get_goal(&self.goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Goal not found: {}", self.goal_id))?;

        // Identify genuinely active work first. A dead task never transitions to
        // completed on its own; strict success handling for finite/current runs
        // follows below, while legacy recurring calls may ignore historical dead
        // rows so one old run cannot wedge every future schedule fire.
        // (Observed live on the daily-tweet goal 9a744834: watchdog-interrupted
        // tasks from prior fires blocked complete_goal, so every new fire looped
        // trying to "resolve" them, got interrupted in turn, and piled up 178 dead
        // tasks.) This mirrors heartbeat's own `terminal_failed` classification.
        if let Some(task) = tasks
            .iter()
            .find(|task| !is_terminal_task_status(&task.status))
        {
            return Ok(format!(
                "Blocked: do not call manage_goal_tasks(action=\"complete_goal\") while tasks are still incomplete. '{}' is still {}. Finish or explicitly resolve every active task first, or use fail_goal if the goal cannot be completed.",
                task.description, task.status
            ));
        }

        // A finite goal, and every explicitly scoped run, may complete only
        // from successful terminal task state. Failed/blocked/interrupted work
        // is terminal for scheduling purposes but is not completion evidence.
        // Legacy recurring calls without a run id retain their historical-task
        // tolerance so an old dead task cannot wedge every future schedule fire.
        let strict_run_completion = goal.goal_type == "finite" || self.goal_run_id.is_some();
        if strict_run_completion && !tasks_satisfy_goal_completion(&tasks) {
            let task = tasks
                .iter()
                .find(|task| !task.satisfies_run_completion())
                .expect("an unsatisfied task must exist");
            return Ok(format!(
                "Blocked: goal completion requires successful task state. '{}' ended as {}. Retry or replace that task successfully, or use fail_goal.",
                task.description, task.status
            ));
        }

        // A continuous (recurring) goal is open-ended: a successful run does not
        // complete the goal. Keep it active for the next cycle and clear the
        // failure streak. Only finite goals are marked completed.
        let is_continuous = goal.goal_type == "continuous";
        if is_continuous {
            goal.dispatch_failures = 0;
            if let Some(raw_context) = goal.context.as_deref()
                && let Ok(mut context) = serde_json::from_str::<Value>(raw_context)
                && let Some(object) = context.as_object_mut()
            {
                object.remove("failure_summary");
                goal.context = if object.is_empty() {
                    None
                } else {
                    Some(context.to_string())
                };
            }
        } else {
            goal.status = "completed".to_string();
            goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
        }
        goal.updated_at = chrono::Utc::now().to_rfc3339();

        // Finish the run before touching the goal: if the store refuses the
        // run, the goal must not already claim a successful cycle.
        let run_id = match self.goal_run_id.clone() {
            Some(run_id) => Some(run_id),
            None => self
                .state
                .get_current_goal_run(&self.goal_id)
                .await?
                .map(|run| run.id),
        };
        if let Some(run_id) = run_id {
            let _ = self
                .state
                .finish_goal_run(&run_id, "completed", Some(summary))
                .await?;
        }
        self.state.update_goal(&goal).await?;
        info!(goal_id = %self.goal_id, is_continuous, "Goal run completed");

        let mut response = if is_continuous {
            format!(
                "Recurring goal {} run completed (goal stays active for the next cycle): {}",
                self.goal_id, summary
            )
        } else {
            format!("Goal {} completed: {}", self.goal_id, summary)
        };
        if let Some(excerpt) = self.build_completed_task_result_excerpt().await? {
            response.push_str("\n\n");
            response.push_str(&excerpt);
        }
        Ok(response)
    }

    /// Append a completed task's summary to the goal's context JSON,
    /// so later executors and the task lead can see what was accomplished.
    async fn accumulate_goal_context(&self, task: &Task) -> anyhow::Result<()> {
        let mut goal = self
            .state
            .get_goal(&self.goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Goal not found"))?;

        // Parse existing context or create new
        let mut ctx: serde_json::Value = goal
            .context
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_else(|| json!({"task_results": []}));

        let result_summary = task
            .result
            .as_deref()
            .map(|r| {
                let end = aidaemon_tools::text::floor_char_boundary(r, 500);
                &r[..end]
            })
            .unwrap_or("");

        let entry = json!({
            "task_id": task.id,
            "description": task.description,
            "result_summary": result_summary,
            "completed_at": task.completed_at,
        });

        if let Some(arr) = ctx.get_mut("task_results").and_then(|v| v.as_array_mut()) {
            arr.push(entry);
        }

        // Compress if context > 32KB: older entries get one-line summaries
        let serialized = serde_json::to_string(&ctx)?;
        if serialized.len() > 32_000
            && let Some(arr) = ctx.get_mut("task_results").and_then(|v| v.as_array_mut())
        {
            compress_old_entries(arr);
        }

        goal.context = Some(serde_json::to_string(&ctx)?);
        goal.updated_at = chrono::Utc::now().to_rfc3339();
        self.state.update_goal(&goal).await?;
        Ok(())
    }

    async fn resolve_blocker(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let raw_task_id = args
            .task_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("resolve_blocker requires 'task_id'"))?;
        let task_id = self.resolve_task_id(raw_task_id).await;
        let task_id = task_id.as_str();

        let Some(task) = self.state.get_task(task_id).await? else {
            return Ok(self.task_not_found_message(task_id).await);
        };

        if task.goal_id != self.goal_id {
            anyhow::bail!("Task {} does not belong to goal {}", task_id, self.goal_id);
        }
        if !self.task_id_in_scope(task_id).await? {
            return Ok(self.out_of_scope_message(task_id));
        }
        if task.status != "blocked" {
            return Ok(format!(
                "Task {} is not blocked (status: {})",
                task_id, task.status
            ));
        }

        let Some(resolution) = args
            .result
            .as_deref()
            .or(args.summary.as_deref())
            .or(args.error.as_deref())
            .filter(|value| !value.trim().is_empty())
        else {
            return Ok(
                "resolve_blocker requires the concrete unblock message in result or summary."
                    .to_string(),
            );
        };
        if !self
            .state
            .unblock_task(task_id, resolution, "task-lead", None)
            .await?
        {
            return Ok(format!("Task {} could not be unblocked.", task_id));
        }
        info!(goal_id = %self.goal_id, task_id, "Blocker resolved; task reset to pending");

        Ok(format!(
            "Blocker resolved for task {}. Task reset to pending.",
            task_id
        ))
    }

    async fn fail_goal(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let mut goal = self
            .state
            .get_goal(&self.goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Goal not found: {}", self.goal_id))?;
        let summary = args.summary.as_deref().unwrap_or("Goal failed");

        let mut context = goal
            .context
            .as_deref()
            .and_then(|ctx| serde_json::from_str::<Value>(ctx).ok())
            .unwrap_or_else(|| json!({}));
        if !context.is_object() {
            context = json!({
                "prior_context_raw": goal.context.clone().unwrap_or_default(),
            });
        }
        if let Some(obj) = context.as_object_mut() {
            obj.insert("failure_summary".to_string(), json!(summary));
        }

        // A continuous (recurring) goal is never terminally failed by a single
        // run: keep it active so its schedule retries next cycle, and record the
        // failure (dispatch_failures) for repeated-failure detection. Only finite
        // goals are marked "failed".
        let is_continuous = goal.goal_type == "continuous";
        if is_continuous {
            goal.dispatch_failures += 1;
        } else {
            goal.status = "failed".to_string();
        }
        goal.context = Some(context.to_string());
        goal.updated_at = chrono::Utc::now().to_rfc3339();

        self.state.update_goal(&goal).await?;
        if is_continuous {
            info!(
                goal_id = %self.goal_id,
                failures = goal.dispatch_failures,
                "Continuous goal run failed; goal kept active for next cycle"
            );
        } else {
            info!(goal_id = %self.goal_id, "Goal failed");
        }

        // Surface a recurring goal that keeps failing, so it doesn't retry
        // forever in silence. Alert on every Nth consecutive failure (the streak
        // is cleared by a successful run in complete_goal).
        const REPEATED_FAILURE_ALERT_THRESHOLD: i32 = 3;
        if is_continuous
            && goal.dispatch_failures >= REPEATED_FAILURE_ALERT_THRESHOLD
            && goal.dispatch_failures % REPEATED_FAILURE_ALERT_THRESHOLD == 0
        {
            let goal_label = aidaemon_tools::sanitize::short_goal_label(&goal.description);
            let msg = format!(
                "Heads up: your recurring goal \"{}\" has failed {} runs in a row. It is still \
                 scheduled and will keep retrying, but it likely needs attention. Latest failure: {}",
                goal_label, goal.dispatch_failures, summary
            );
            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                "evergreen_alert",
                &msg,
            );
            let _ = self.state.enqueue_notification(&entry).await;
            info!(
                goal_id = %self.goal_id,
                failures = goal.dispatch_failures,
                "Alerted user: recurring goal failing repeatedly"
            );
        }

        // Cancel remaining pending/claimed tasks so they don't get re-dispatched
        let tasks = self.scoped_tasks().await.unwrap_or_default();
        let mut cancelled = 0;
        for task in &tasks {
            if (task.status == "pending" || task.status == "claimed")
                && self
                    .state
                    .cancel_work_task(&task.id, "task-lead", None)
                    .await
                    .unwrap_or(false)
            {
                cancelled += 1;
            }
        }
        if cancelled > 0 {
            info!(goal_id = %self.goal_id, cancelled, "Cancelled pending tasks for failed goal");
        }
        let run_id = match self.goal_run_id.clone() {
            Some(run_id) => Some(run_id),
            None => self
                .state
                .get_current_goal_run(&self.goal_id)
                .await?
                .map(|run| run.id),
        };
        if let Some(run_id) = run_id {
            let _ = self
                .state
                .finish_goal_run(&run_id, "failed", Some(summary))
                .await?;
        }

        if is_continuous {
            Ok(format!(
                "Recurring goal {} run failed (cancelled {} pending tasks); the goal stays active and will retry next cycle: {}",
                self.goal_id, cancelled, summary
            ))
        } else {
            Ok(format!(
                "Goal {} failed (cancelled {} pending tasks): {}",
                self.goal_id, cancelled, summary
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_no_cycles_simple() {
        let task_a = Task {
            id: "a".to_string(),
            goal_id: "g".to_string(),
            description: "A".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: String::new(),
            started_at: None,
            completed_at: None,
        };

        // B depends on A — no cycle
        assert!(
            super::validate_no_cycles(std::slice::from_ref(&task_a), "b", &["a".to_string()])
                .is_ok()
        );
    }

    #[test]
    fn test_validate_no_cycles_self_reference() {
        // Task tries to depend on itself
        assert!(super::validate_no_cycles(&[], "a", &["a".to_string()]).is_err());
    }

    #[test]
    fn test_validate_no_cycles_circular() {
        // A depends on B, B (new) depends on A → cycle
        let task_a = Task {
            id: "a".to_string(),
            goal_id: "g".to_string(),
            description: "A".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: Some(serde_json::to_string(&vec!["b"]).unwrap()),
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: String::new(),
            started_at: None,
            completed_at: None,
        };

        let task_b = Task {
            id: "b".to_string(),
            goal_id: "g".to_string(),
            description: "B".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: String::new(),
            started_at: None,
            completed_at: None,
        };

        // C depends on A, but A already depends on B → valid (A→B, C→A)
        assert!(
            super::validate_no_cycles(&[task_a.clone(), task_b.clone()], "c", &["a".to_string()])
                .is_ok()
        );

        // New task "b2" depends on A, but A depends on B, and B already exists → no cycle
        // But if we create a NEW "b" that depends on "a" when "a" depends on "b" → cycle
        // Let's test: existing has A depends on C, C exists. New task C depends on A → cycle
        let task_a_dep_c = Task {
            id: "a".to_string(),
            depends_on: Some(serde_json::to_string(&vec!["c"]).unwrap()),
            ..task_a.clone()
        };
        assert!(super::validate_no_cycles(&[task_a_dep_c], "c", &["a".to_string()]).is_err());
    }

    #[test]
    fn test_validate_nonexistent_dependency() {
        assert!(super::validate_no_cycles(&[], "a", &["nonexistent".to_string()]).is_err());
    }

    #[test]
    fn test_compress_old_entries() {
        let mut entries: Vec<Value> = (0..15)
            .map(|i| {
                json!({
                    "task_id": format!("task-{:04}", i),
                    "description": format!("Task number {}", i),
                    "result_summary": "Done",
                    "completed_at": "2025-01-01T00:00:00Z",
                })
            })
            .collect();

        super::compress_old_entries(&mut entries);

        // First 5 (15-10) should be compressed to strings
        for entry in entries.iter().take(5) {
            assert!(
                entry.is_string(),
                "Old entries should be compressed to strings"
            );
            let s = entry.as_str().unwrap();
            assert!(
                s.contains("(completed)"),
                "Compressed entry should say completed: {}",
                s
            );
        }

        // Last 10 should remain as objects
        for entry in entries.iter().skip(5) {
            assert!(entry.is_object(), "Recent entries should remain as objects");
        }
    }
}
