use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::traits::{
    semantics_for_exact_read_actions, HandoffArtifact, StateStore, Task, TaskAttemptPatch,
    TaskHandoff, Tool, ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolRole,
};

/// Tool for task leads to manage tasks within their assigned goal.
pub struct ManageGoalTasksTool {
    goal_id: String,
    state: Arc<dyn StateStore>,
    goal_run_id: Option<String>,
}

impl ManageGoalTasksTool {
    pub fn new(goal_id: String, state: Arc<dyn StateStore>) -> Self {
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
        task.goal_id == self.goal_id && !Self::is_scheduled_root_task(task)
    }

    fn is_scheduled_root_task(task: &Task) -> bool {
        let description = task.description.trim_start().to_ascii_lowercase();
        description.starts_with("execute scheduled goal:")
            || description.starts_with("scheduled check:")
            || description.starts_with("manual scheduled run:")
            || description.starts_with("mandate review:")
    }

    async fn scoped_tasks(&self) -> anyhow::Result<Vec<Task>> {
        let tasks = if let Some(run_id) = self.goal_run_id.as_deref() {
            self.state.get_tasks_for_goal_run(run_id).await?
        } else {
            self.state.get_tasks_for_goal(&self.goal_id).await?
        };
        Ok(tasks
            .into_iter()
            .filter(|task| self.task_in_scope(task))
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
        if let Ok(Some(task)) = self.state.get_task(task_id).await {
            if self.task_in_scope(&task) && self.task_id_in_scope(task_id).await.unwrap_or(false) {
                return task_id.to_string();
            }
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

/// A task status is "terminal" when it will never advance to `completed` on its
/// own — either it already succeeded (`completed`/`skipped`), was explicitly
/// replaced (`superseded`), or it is dead
/// (`failed`/`blocked`/`interrupted`/`cancelled`/`abandoned`). Terminal tasks
/// are useful when deciding whether a scheduler can make further progress.
/// Terminal does not imply successful: strict finite/current-run completion
/// separately requires [`Task::satisfies_run_completion`].
pub(crate) fn is_terminal_task_status(status: &str) -> bool {
    matches!(
        status,
        "completed"
            | "skipped"
            | "superseded"
            | "failed"
            | "blocked"
            | "interrupted"
            | "cancelled"
            | "abandoned"
    )
}

pub(crate) fn tasks_satisfy_goal_completion(tasks: &[Task]) -> bool {
    !tasks.is_empty() && tasks.iter().all(Task::satisfies_run_completion)
}

pub(crate) fn goal_completion_summary_indicates_not_finished(summary: &str) -> bool {
    let lower = summary.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    [
        "i completed part of the request",
        "i'm blocked from safely finishing",
        "blocked from safely finishing",
        "haven't verified the final outcome",
        "have not verified the final outcome",
        "haven't verified yet",
        "have not verified yet",
        "not verified yet",
        "not yet verified",
        "verification pending",
        "need a final read-only check",
        "need a final read only check",
        "before i can claim success",
        "can't claim success",
        "cannot claim success",
        "still need to verify",
        "still need a final check",
        "partially completed",
        "partial completion",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
}

#[derive(Deserialize)]
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
    #[serde(default)]
    _mandate_id: Option<String>,
    #[serde(default)]
    _mandate_version: Option<i64>,
    #[serde(default)]
    _goal_run_id: Option<String>,
    #[serde(default)]
    _task_attempt_id: Option<String>,
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
            "description": "Manage tasks inside the current goal.",
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

    fn tool_role(&self) -> ToolRole {
        ToolRole::Management
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        semantics_for_exact_read_actions(arguments, &["list_tasks"], ToolMutationEffects::NONE)
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageGoalTasksArgs = serde_json::from_str(arguments)?;

        let mandate_context = args._mandate_id.is_some()
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
            other => Ok(format!("Unknown action: {}. Use: create_task, list_tasks, update_task, claim_task, retry_task, resolve_blocker, complete_goal, fail_goal", other)),
        }
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
}

/// Check if all dependencies for a task are completed. Returns Ok(()) if all are met,
/// or an error message string describing which dependencies are unmet.
async fn check_dependencies_met(state: &dyn StateStore, task: &Task) -> Result<(), String> {
    if let Some(ref deps_json) = task.depends_on {
        let dep_ids = serde_json::from_str::<Vec<String>>(deps_json)
            .map_err(|_| "dependency metadata is invalid".to_string())?;
        for dep_id in &dep_ids {
            match state.get_task(dep_id).await {
                Ok(Some(dep_task))
                    if matches!(
                        dep_task.status.as_str(),
                        "completed" | "skipped" | "superseded"
                    ) => {}
                Ok(Some(dep_task)) => {
                    return Err(format!(
                        "dependency {} is not completed or otherwise accepted (status: {})",
                        dep_id, dep_task.status
                    ));
                }
                Ok(None) => return Err(format!("dependency {} does not exist", dep_id)),
                Err(error) => return Err(format!("could not read dependency {dep_id}: {error}")),
            }
        }
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
/// Uses Kahn's algorithm (topological sort) on the task dependency graph.
fn validate_no_cycles(
    existing: &[Task],
    new_task_id: &str,
    new_deps: &[String],
) -> Result<(), String> {
    use std::collections::{HashMap, HashSet, VecDeque};

    // Build adjacency list: task_id -> set of tasks it depends on (owned strings)
    let mut deps_map: HashMap<String, HashSet<String>> = HashMap::new();
    let mut all_ids: HashSet<String> = HashSet::new();

    for task in existing {
        all_ids.insert(task.id.clone());
        if let Some(ref deps_json) = task.depends_on {
            if let Ok(dep_ids) = serde_json::from_str::<Vec<String>>(deps_json) {
                deps_map.insert(task.id.clone(), dep_ids.into_iter().collect());
            }
        }
    }

    // Add the new task
    all_ids.insert(new_task_id.to_string());
    let new_dep_set: HashSet<String> = new_deps.iter().cloned().collect();

    // Verify all dependencies reference existing tasks within this goal
    for dep in &new_dep_set {
        if !all_ids.contains(dep) {
            return Err(format!("Dependency {} does not exist in this goal", dep));
        }
    }

    deps_map.insert(new_task_id.to_string(), new_dep_set);

    // Kahn's algorithm: compute in-degree, then peel off zero-degree nodes
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    for id in &all_ids {
        in_degree.insert(id.clone(), 0);
    }

    // in-degree of task_id = number of deps it has
    for (task_id, deps) in &deps_map {
        *in_degree.entry(task_id.clone()).or_insert(0) += deps.len();
    }

    let mut queue: VecDeque<String> = VecDeque::new();
    for (id, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(id.clone());
        }
    }

    let mut processed = 0usize;
    while let Some(node) = queue.pop_front() {
        processed += 1;
        // Find all tasks that depend on this node and reduce their in-degree
        for (task_id, deps) in &deps_map {
            if deps.contains(&node) {
                if let Some(deg) = in_degree.get_mut(task_id) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(task_id.clone());
                    }
                }
            }
        }
    }

    if processed < all_ids.len() {
        Err("Dependency cycle detected — cannot create task".to_string())
    } else {
        Ok(())
    }
}

impl ManageGoalTasksTool {
    async fn create_task(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let persisted_mandate = self.state.get_mandate_for_goal(&self.goal_id).await?;
        let mandate_context = args._mandate_id.is_some() || persisted_mandate.is_some();
        let mandate_fence = if mandate_context {
            let mandate_id = args._mandate_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task creation is missing its dispatcher-owned mandate id")
            })?;
            let mandate_version = args._mandate_version.ok_or_else(|| {
                anyhow::anyhow!(
                    "mandate task creation is missing its dispatcher-owned policy version"
                )
            })?;
            let goal_run_id = args._goal_run_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task creation is missing its dispatcher-owned goal run")
            })?;
            let task_attempt_id = args._task_attempt_id.as_deref().ok_or_else(|| {
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
            anyhow::ensure!(
                args.workspace_policy.is_none(),
                "workspace_policy is unavailable for mandate tasks"
            );
            anyhow::ensure!(
                args.worker_profile.is_none(),
                "worker_profile assignment is unavailable for mandate tasks"
            );
            Some((mandate_id, mandate_version, goal_run_id, task_attempt_id))
        } else {
            None
        };

        let description = args
            .description
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("create_task requires 'description'"))?;
        if let Some(profile_id) = args.worker_profile.as_deref() {
            if self.state.get_worker_profile(profile_id).await?.is_none() {
                return Ok(format!(
                    "Cannot create task: worker profile {profile_id} not found"
                ));
            }
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
        if let Some(ref dep_ids) = args.depends_on {
            if !dep_ids.is_empty() {
                let existing = self.scoped_tasks().await?;
                if let Err(reason) = validate_no_cycles(&existing, &task_id, dep_ids) {
                    return Ok(format!("Cannot create task: {}", reason));
                }
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
            if let Some(profile_id) = args.worker_profile.as_deref() {
                if !self
                    .state
                    .assign_task_worker_profile(&task.id, profile_id, "task-lead", None)
                    .await?
                {
                    anyhow::bail!("worker profile '{}' is unavailable", profile_id);
                }
            }
            if let Some(policy) = args.workspace_policy.as_deref() {
                if !self
                    .state
                    .set_task_workspace_policy(&task.id, policy)
                    .await?
                {
                    anyhow::bail!("workspace policy could not be applied to the new task");
                }
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
            if let Some(ref deps) = task.depends_on {
                if let Ok(dep_ids) = serde_json::from_str::<Vec<String>>(deps) {
                    if !dep_ids.is_empty() {
                        let short_deps: Vec<String> = dep_ids
                            .iter()
                            .map(|d| d[..8.min(d.len())].to_string())
                            .collect();
                        details.push(format!("deps: [{}]", short_deps.join(", ")));
                    }
                }
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
            if let Ok(journal) = self.state.get_task_journal(&task.id, 12).await {
                if let Some(entry) = journal.iter().find(|entry| entry.actor_type == "human") {
                    let end = crate::utils::floor_char_boundary(&entry.body, 120);
                    details.push(format!(
                        "latest human {}: {}",
                        entry.entry_type,
                        &entry.body[..end]
                    ));
                }
            }

            let result_suffix = task
                .result
                .as_deref()
                .map(|r| {
                    let end = crate::utils::floor_char_boundary(r, 200);
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
        if let Some(ref new_status) = args.status {
            if new_status == "running" || new_status == "claimed" {
                if let Err(reason) = check_dependencies_met(self.state.as_ref(), &task).await {
                    return Ok(format!(
                        "Cannot set task {} to {}: {}",
                        task_id, new_status, reason
                    ));
                }
            }
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
        if let Some(error) = &args.error {
            task.error = (!error.trim().is_empty()).then(|| error.clone());
        }

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
                remaining_risk: args.remaining_risk.clone(),
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
        if task.status == "completed" && task.result.is_some() {
            if let Err(e) = self.accumulate_goal_context(&task).await {
                tracing::warn!(goal_id = %self.goal_id, "Failed to accumulate goal context: {}", e);
            }
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
        if args._mandate_id.is_some()
            || self
                .state
                .get_mandate_for_goal(&self.goal_id)
                .await?
                .is_some()
        {
            let mandate_id = args._mandate_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task claim is missing its dispatcher-owned mandate id")
            })?;
            let mandate_version = args._mandate_version.ok_or_else(|| {
                anyhow::anyhow!("mandate task claim is missing its dispatcher-owned policy version")
            })?;
            let goal_run_id = args._goal_run_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task claim is missing its dispatcher-owned goal run")
            })?;
            let root_attempt_id = args._task_attempt_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("mandate task claim is missing its dispatcher-owned root attempt")
            })?;
            anyhow::ensure!(
                self.goal_run_id.as_deref() == Some(goal_run_id),
                "mandate task claim run does not match the task-lead binding"
            );
            anyhow::ensure!(
                args.worker_profile.is_none(),
                "worker_profile assignment is unavailable for mandate tasks"
            );
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
            .claim_task_with_lease(
                task_id,
                agent_id,
                args.worker_profile.as_deref(),
                180,
            )
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
        if goal_completion_summary_indicates_not_finished(summary) {
            return Ok(
                "Blocked: do not call manage_goal_tasks(action=\"complete_goal\") when the summary says verification is still pending or only partial progress is done. Keep the goal active, finish the final read-only check, then complete it; or use fail_goal if the work cannot be finished."
                    .to_string(),
            );
        }

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
            if let Some(raw_context) = goal.context.as_deref() {
                if let Ok(mut context) = serde_json::from_str::<Value>(raw_context) {
                    if let Some(object) = context.as_object_mut() {
                        object.remove("failure_summary");
                        goal.context = if object.is_empty() {
                            None
                        } else {
                            Some(context.to_string())
                        };
                    }
                }
            }
        } else {
            goal.status = "completed".to_string();
            goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
        }
        goal.updated_at = chrono::Utc::now().to_rfc3339();

        self.state.update_goal(&goal).await?;
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
                let end = crate::utils::floor_char_boundary(r, 500);
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
        if serialized.len() > 32_000 {
            if let Some(arr) = ctx.get_mut("task_results").and_then(|v| v.as_array_mut()) {
                compress_old_entries(arr);
            }
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
            let goal_label = crate::tools::sanitize::short_goal_label(&goal.description);
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
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::store_prelude::*;
    use crate::traits::Goal;

    async fn setup_test_state() -> (Arc<dyn StateStore>, String) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );

        // Create a goal
        let goal = Goal::new_finite("Test goal", "test-session");
        state.create_goal(&goal).await.unwrap();

        // We need to keep db_file alive, but for tests we'll leak it
        std::mem::forget(db_file);
        (state as Arc<dyn StateStore>, goal.id)
    }

    fn test_task(goal_id: &str, id: &str, description: &str, created_at: &str) -> Task {
        Task {
            id: id.to_string(),
            goal_id: goal_id.to_string(),
            description: description.to_string(),
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
            max_retries: 0,
            created_at: created_at.to_string(),
            started_at: None,
            completed_at: None,
        }
    }

    #[tokio::test]
    async fn scheduled_task_tool_isolates_current_run_and_hides_root() {
        let (state, goal_id) = setup_test_state().await;
        let prior = test_task(
            &goal_id,
            "11111111-1111-1111-1111-111111111111",
            "Prior run task",
            "2026-07-27T12:00:00Z",
        );
        let root = test_task(
            &goal_id,
            "22222222-2222-2222-2222-222222222222",
            "Manual scheduled run: publish the blog",
            "2026-07-28T12:00:00Z",
        );
        let current = test_task(
            &goal_id,
            "33333333-3333-3333-3333-333333333333",
            "Current run task",
            "2026-07-28T12:00:01Z",
        );
        state.create_task(&prior).await.unwrap();
        let prior_run = state.get_current_goal_run(&goal_id).await.unwrap().unwrap();
        state
            .finish_goal_run(&prior_run.id, "completed", Some("prior run"))
            .await
            .unwrap();
        let current_run = state
            .start_goal_run(&goal_id, "manual", None, Some(&root.id))
            .await
            .unwrap();
        state.create_task(&root).await.unwrap();
        state.create_task(&current).await.unwrap();

        let tool =
            ManageGoalTasksTool::new(goal_id, state.clone()).with_goal_run_id(Some(current_run.id));
        let listed = tool
            .call(&json!({ "action": "list_tasks" }).to_string())
            .await
            .unwrap();
        assert!(listed.contains("Current run task"));
        assert!(!listed.contains("Prior run task"));
        assert!(!listed.contains("Manual scheduled run"));

        let update = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": prior.id,
                    "status": "completed"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(update.contains("earlier run"));
        assert_eq!(
            state.get_task(&prior.id).await.unwrap().unwrap().status,
            "pending"
        );
    }

    #[tokio::test]
    async fn fail_goal_keeps_continuous_goal_active() {
        let (state, _finite) = setup_test_state().await;
        let cg = Goal::new_continuous("Daily recurring goal", "test-session", None, None);
        assert_eq!(cg.status, "active");
        state.create_goal(&cg).await.unwrap();
        let tool = ManageGoalTasksTool::new(cg.id.clone(), state.clone());

        tool.call(&json!({ "action": "fail_goal", "summary": "deploy failed today" }).to_string())
            .await
            .unwrap();

        let g = state.get_goal(&cg.id).await.unwrap().unwrap();
        assert_eq!(
            g.status, "active",
            "a single run failure must NOT terminally fail a continuous goal"
        );
        assert!(
            g.dispatch_failures >= 1,
            "the run failure should be recorded for repeated-failure detection"
        );
    }

    #[tokio::test]
    async fn fail_goal_alerts_after_repeated_continuous_failures() {
        let (state, _finite) = setup_test_state().await;
        let cg = Goal::new_continuous("Daily recurring goal", "test-session", None, None);
        state.create_goal(&cg).await.unwrap();
        let tool = ManageGoalTasksTool::new(cg.id.clone(), state.clone());
        let body = json!({ "action": "fail_goal", "summary": "run failed" }).to_string();

        // First two failures: no alert yet.
        tool.call(&body).await.unwrap();
        tool.call(&body).await.unwrap();
        let count = |ns: Vec<crate::traits::NotificationEntry>, id: &str| {
            ns.into_iter()
                .filter(|n| n.notification_type == "evergreen_alert" && n.goal_id == id)
                .count()
        };
        assert_eq!(
            count(state.get_pending_notifications(20).await.unwrap(), &cg.id),
            0,
            "should not alert before the failure threshold"
        );

        // Third consecutive failure crosses the threshold: one alert.
        tool.call(&body).await.unwrap();
        assert_eq!(
            count(state.get_pending_notifications(20).await.unwrap(), &cg.id),
            1,
            "should alert once a continuous goal has failed 3 runs in a row"
        );
    }

    #[tokio::test]
    async fn complete_goal_keeps_continuous_active_and_clears_failures() {
        let (state, _finite) = setup_test_state().await;
        let cg = Goal::new_continuous("Daily recurring goal", "test-session", None, None);
        state.create_goal(&cg).await.unwrap();
        let tool = ManageGoalTasksTool::new(cg.id.clone(), state.clone());

        // Record a couple of failures.
        let fail = json!({ "action": "fail_goal", "summary": "x" }).to_string();
        tool.call(&fail).await.unwrap();
        tool.call(&fail).await.unwrap();

        // A successful run: one completed task, then complete_goal.
        tool.call(
            &json!({ "action": "create_task", "description": "daily check", "task_order": 1 })
                .to_string(),
        )
        .await
        .unwrap();
        let tasks = state.get_tasks_for_goal(&cg.id).await.unwrap();
        let mut t = tasks[0].clone();
        t.status = "completed".to_string();
        t.completed_at = Some(chrono::Utc::now().to_rfc3339());
        state.update_task(&t).await.unwrap();
        tool.call(&json!({ "action": "complete_goal", "summary": "daily check done" }).to_string())
            .await
            .unwrap();

        let g = state.get_goal(&cg.id).await.unwrap().unwrap();
        assert_eq!(
            g.status, "active",
            "a continuous goal stays active after a successful run"
        );
        assert_eq!(
            g.dispatch_failures, 0,
            "a successful run clears the failure streak"
        );
    }

    #[tokio::test]
    async fn complete_goal_not_blocked_by_dead_interrupted_task() {
        // Regression: the daily-tweet goal 9a744834 accumulated watchdog-
        // interrupted tasks from prior fires; because complete_goal treated any
        // non-completed/skipped status as "still incomplete", a dead interrupted
        // task permanently blocked every new run, which then looped and got
        // interrupted itself. An interrupted (dead terminal) task must NOT block.
        let (state, _finite) = setup_test_state().await;
        let cg = Goal::new_continuous("Daily recurring goal", "test-session", None, None);
        state.create_goal(&cg).await.unwrap();
        let tool = ManageGoalTasksTool::new(cg.id.clone(), state.clone());

        // This run's real work: post the tweet (completed).
        tool.call(&json!({ "action": "create_task", "description": "Post the tweet" }).to_string())
            .await
            .unwrap();
        // A dead task left over from a prior fire.
        tool.call(
            &json!({ "action": "create_task", "description": "Prior scheduled fire" }).to_string(),
        )
        .await
        .unwrap();
        let tasks = state.get_tasks_for_goal(&cg.id).await.unwrap();
        let mut done = tasks[0].clone();
        done.status = "completed".to_string();
        done.completed_at = Some(chrono::Utc::now().to_rfc3339());
        state.update_task(&done).await.unwrap();
        let mut dead = tasks[1].clone();
        dead.status = "interrupted".to_string();
        state.update_task(&dead).await.unwrap();

        let result = tool
            .call(&json!({ "action": "complete_goal", "summary": "tweet posted" }).to_string())
            .await
            .unwrap();

        assert!(
            !result.contains("Blocked"),
            "a dead interrupted task must not block completion: {result}"
        );
        assert!(result.contains("run completed"), "got: {result}");
        let g = state.get_goal(&cg.id).await.unwrap().unwrap();
        assert_eq!(
            g.status, "active",
            "continuous goal stays active after the run"
        );
    }

    #[tokio::test]
    async fn complete_goal_still_blocked_by_active_running_task() {
        // The guard's remaining purpose: genuinely in-flight work still blocks.
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());
        tool.call(&json!({ "action": "create_task", "description": "In-flight work" }).to_string())
            .await
            .unwrap();
        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let mut running = tasks[0].clone();
        running.status = "running".to_string();
        state.update_task(&running).await.unwrap();

        let result = tool
            .call(&json!({ "action": "complete_goal", "summary": "done" }).to_string())
            .await
            .unwrap();
        assert!(
            result.contains("Blocked") && result.contains("still running"),
            "an active running task must still block completion: {result}"
        );
    }

    #[tokio::test]
    async fn finite_goal_cannot_complete_with_failed_terminal_task() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Build and publish the site"
            })
            .to_string(),
        )
        .await
        .unwrap();
        let mut failed = state.get_tasks_for_goal(&goal_id).await.unwrap()[0].clone();
        failed.status = "failed".to_string();
        failed.error = Some("build never ran".to_string());
        failed.completed_at = Some(chrono::Utc::now().to_rfc3339());
        state.update_task(&failed).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "The source files exist"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(
            result.contains("requires successful task state"),
            "{result}"
        );
        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "active");
        assert!(goal.completed_at.is_none());
    }

    #[tokio::test]
    async fn finite_goal_cannot_complete_with_stale_blocker_on_completed_task() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Verify the deployment"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Reproduce a legacy inconsistent row without first moving through the
        // blocked state. The transition fence protects new blocked tasks; this
        // success predicate also protects already-corrupt persisted data.
        let mut inconsistent = state.get_tasks_for_goal(&goal_id).await.unwrap()[0].clone();
        inconsistent.status = "completed".to_string();
        inconsistent.blocker = Some("verification never ran".to_string());
        inconsistent.completed_at = Some(chrono::Utc::now().to_rfc3339());
        state.update_task(&inconsistent).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "The deployment is ready"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(
            result.contains("requires successful task state"),
            "{result}"
        );
        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "active");
        assert!(goal.completed_at.is_none());
    }

    #[tokio::test]
    async fn current_recurring_run_cannot_complete_with_failed_task() {
        let (state, _finite) = setup_test_state().await;
        let goal = Goal::new_continuous("Recurring publication", "test-session", None, None);
        state.create_goal(&goal).await.unwrap();
        let legacy_tool = ManageGoalTasksTool::new(goal.id.clone(), state.clone());
        legacy_tool
            .call(
                &json!({
                    "action": "create_task",
                    "description": "Publish this run"
                })
                .to_string(),
            )
            .await
            .unwrap();
        let run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        let mut failed = state.get_tasks_for_goal_run(&run.id).await.unwrap()[0].clone();
        failed.status = "failed".to_string();
        failed.error = Some("publication failed".to_string());
        state.update_task(&failed).await.unwrap();
        let scoped_tool =
            ManageGoalTasksTool::new(goal.id.clone(), state.clone()).with_goal_run_id(Some(run.id));

        let result = scoped_tool
            .call(&json!({"action": "complete_goal", "summary": "run done"}).to_string())
            .await
            .unwrap();
        assert!(
            result.contains("requires successful task state"),
            "{result}"
        );
    }

    #[tokio::test]
    async fn test_create_task_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_task",
                    "description": "Write the code",
                    "priority": "high",
                    "task_order": 1
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Created task"));
        assert!(result.contains("Write the code"));

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "Write the code");
        assert_eq!(tasks[0].priority, "high");
        assert_eq!(tasks[0].task_order, 1);
    }

    #[tokio::test]
    async fn test_list_tasks_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create two tasks
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task one",
                "task_order": 1
            })
            .to_string(),
        )
        .await
        .unwrap();

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task two",
                "task_order": 2
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(&json!({"action": "list_tasks"}).to_string())
            .await
            .unwrap();

        assert!(result.contains("2 total"));
        assert!(result.contains("Task one"));
        assert!(result.contains("Task two"));
    }

    #[tokio::test]
    async fn test_update_task_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Do something"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = &tasks[0].id;

        // Update it
        let result = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": task_id,
                    "status": "completed",
                    "result": "Done successfully"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("completed"));

        let updated = state.get_task(task_id).await.unwrap().unwrap();
        assert_eq!(updated.status, "completed");
        assert_eq!(updated.result.as_deref(), Some("Done successfully"));
        assert!(updated.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_complete_goal_requires_completed_tasks() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Do the work"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let task_id = state.get_tasks_for_goal(&goal_id).await.unwrap()[0]
            .id
            .clone();
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": task_id,
                "status": "completed",
                "result": "All tasks done"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "All tasks done"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("completed"));
        assert!(result.contains("All tasks done"));

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "completed");
        assert!(goal.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_complete_goal_includes_final_task_result_excerpt() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Find largest directories"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = &tasks[0].id;

        tool.call(
            &json!({
                "action": "update_task",
                "task_id": task_id,
                "status": "completed",
                "result": "1.2G /Users/me/projects/aidaemon/target"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "Finished disk usage audit"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Goal"));
        assert!(result.contains("Finished disk usage audit"));
        assert!(result.contains("Task result:"));
        assert!(result.contains("1.2G /Users/me/projects/aidaemon/target"));
    }

    #[tokio::test]
    async fn test_complete_goal_includes_multiple_recent_task_results() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        for i in 1..=4 {
            tool.call(
                &json!({
                    "action": "create_task",
                    "description": format!("Research step {}", i)
                })
                .to_string(),
            )
            .await
            .unwrap();
        }

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        for task in &tasks {
            tool.call(
                &json!({
                    "action": "update_task",
                    "task_id": task.id,
                    "status": "completed",
                    "result": format!("Result payload for {}", task.description)
                })
                .to_string(),
            )
            .await
            .unwrap();
        }

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "Compiled multi-step research outputs"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("4/4 tasks completed."));
        assert!(result.contains("Recent task results:"));
        assert!(result.contains("Research step 4"));
        assert!(result.contains("Research step 3"));
        assert!(result.contains("Research step 2"));
        assert!(result.contains("(+1 earlier completed task result omitted)"));
    }

    #[tokio::test]
    async fn test_complete_goal_blocks_verification_pending_summary() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "I completed part of the request, but I haven't verified the final outcome against /Users/davidloor/Library/Logs/aidaemon yet.\n\nI need a final read-only check before I can claim success."
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Blocked:"));

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "active");
        assert!(goal.completed_at.is_none());
    }

    #[tokio::test]
    async fn test_complete_goal_blocks_when_tasks_are_not_done() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Run final verification"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "complete_goal",
                    "summary": "Everything is finished"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Blocked:"));
        assert!(result.contains("Run final verification"));

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "active");
        assert!(goal.completed_at.is_none());
    }

    #[tokio::test]
    async fn test_fail_goal_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "fail_goal",
                    "summary": "Could not complete"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("failed"));
        assert!(result.contains("Could not complete"));

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert_eq!(goal.status, "failed");
        assert_eq!(
            goal.context
                .as_deref()
                .and_then(|ctx| serde_json::from_str::<Value>(ctx).ok())
                .and_then(|ctx| {
                    ctx.get("failure_summary")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned)
                }),
            Some("Could not complete".to_string())
        );
    }

    #[tokio::test]
    async fn test_claim_task_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Claimable task"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // First claim should succeed
        let result = tool
            .call(
                &json!({
                    "action": "claim_task",
                    "task_id": &task_id,
                    "agent_id": "executor-1"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Claimed task"));
        assert!(result.contains("executor-1"));

        // Second claim should fail
        let result2 = tool
            .call(
                &json!({
                    "action": "claim_task",
                    "task_id": &task_id,
                    "agent_id": "executor-2"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result2.contains("Failed to claim"));
    }

    #[tokio::test]
    async fn test_claim_task_dependency_check() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create task A
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task A",
                "task_order": 1
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_a_id = tasks[0].id.clone();

        // Create task B that depends on A
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task B",
                "task_order": 2,
                "depends_on": [&task_a_id]
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_b_id = tasks
            .iter()
            .find(|t| t.description == "Task B")
            .unwrap()
            .id
            .clone();

        // Claim B should fail — A not completed
        let result = tool
            .call(
                &json!({
                    "action": "claim_task",
                    "task_id": &task_b_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Cannot claim"));
        assert!(result.contains("not completed"));

        // Complete task A
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_a_id,
                "status": "completed",
                "result": "Done"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Now claim B should succeed
        let result2 = tool
            .call(
                &json!({
                    "action": "claim_task",
                    "task_id": &task_b_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result2.contains("Claimed task"));
    }

    #[tokio::test]
    async fn test_retry_task_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create an idempotent task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Retryable task",
                "idempotent": true
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // Fail the task
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "failed",
                "error": "Something went wrong"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Retry should succeed
        let result = tool
            .call(
                &json!({
                    "action": "retry_task",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("reset to pending"));
        assert!(result.contains("1/3"));

        // Verify the task was reset
        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "pending");
        assert_eq!(task.retry_count, 1);
        assert!(task.error.is_none());
        assert!(task.agent_id.is_none());
    }

    #[tokio::test]
    async fn test_retry_task_non_idempotent() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a non-idempotent task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Non-retryable task"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // Fail it
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "failed",
                "error": "Oops"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Retry should fail — not idempotent
        let result = tool
            .call(
                &json!({
                    "action": "retry_task",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("not marked as idempotent"));
    }

    #[tokio::test]
    async fn test_retry_task_max_retries_exceeded() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create an idempotent task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Exhaustible task",
                "idempotent": true
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // Exhaust all retries (max_retries = 3)
        for _ in 0..3 {
            // Fail the task
            tool.call(
                &json!({
                    "action": "update_task",
                    "task_id": &task_id,
                    "status": "failed",
                    "error": "Failed again"
                })
                .to_string(),
            )
            .await
            .unwrap();

            // Retry
            tool.call(
                &json!({
                    "action": "retry_task",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        }

        // Fail it one more time
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "failed",
                "error": "Failed again"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Next retry should fail — max retries reached
        let result = tool
            .call(
                &json!({
                    "action": "retry_task",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("max retries reached"));
    }

    #[tokio::test]
    async fn test_update_task_dependency_enforcement() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create task A
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task A"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_a_id = tasks[0].id.clone();

        // Create task B that depends on A
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task B",
                "depends_on": [&task_a_id]
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_b_id = tasks
            .iter()
            .find(|t| t.description == "Task B")
            .unwrap()
            .id
            .clone();

        // Try to set B to "running" — should fail
        let result = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": &task_b_id,
                    "status": "running"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Cannot set task"));
        assert!(result.contains("not completed"));

        // Complete A
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_a_id,
                "status": "completed"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Now setting B to "running" should succeed
        let result2 = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": &task_b_id,
                    "status": "running"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result2.contains("running"));
    }

    #[tokio::test]
    async fn test_list_tasks_shows_deps_and_groups() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create task A with parallel_group
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task A",
                "task_order": 1,
                "parallel_group": "phase-1"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_a_id = tasks[0].id.clone();

        // Create task B with depends_on and idempotent
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task B",
                "task_order": 2,
                "depends_on": [&task_a_id],
                "idempotent": true
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(&json!({"action": "list_tasks"}).to_string())
            .await
            .unwrap();

        assert!(result.contains("group: phase-1"));
        assert!(result.contains("deps: ["));
        assert!(result.contains("retries: 0/3"));
    }

    #[tokio::test]
    async fn test_context_accumulation_on_completion() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Build the frontend"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = &tasks[0].id;

        // Complete with a result
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": task_id,
                "status": "completed",
                "result": "Built React frontend with login page and dashboard"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Check goal context has the task result
        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        assert!(
            goal.context.is_some(),
            "Goal should have context after task completion"
        );
        let ctx: serde_json::Value =
            serde_json::from_str(goal.context.as_deref().unwrap()).unwrap();
        let results = ctx["task_results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["description"], "Build the frontend");
        assert!(results[0]["result_summary"]
            .as_str()
            .unwrap()
            .contains("React frontend"));
    }

    #[tokio::test]
    async fn test_context_accumulation_multiple_tasks() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create two tasks
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task Alpha"
            })
            .to_string(),
        )
        .await
        .unwrap();
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Task Beta"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();

        // Complete both
        for task in &tasks {
            tool.call(
                &json!({
                    "action": "update_task",
                    "task_id": task.id,
                    "status": "completed",
                    "result": format!("Completed {}", task.description)
                })
                .to_string(),
            )
            .await
            .unwrap();
        }

        let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
        let ctx: serde_json::Value =
            serde_json::from_str(goal.context.as_deref().unwrap()).unwrap();
        let results = ctx["task_results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_resolve_blocker_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        // Create a task
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Blocked task"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        // Set it to blocked
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "blocked"
            })
            .to_string(),
        )
        .await
        .unwrap();

        // Resolve the blocker
        let result = tool
            .call(
                &json!({
                    "action": "resolve_blocker",
                    "task_id": &task_id,
                    "result": "Found alternative approach"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Blocker resolved"));
        assert!(result.contains("pending"));

        // Verify task is pending with resolution context
        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "pending");
        assert!(task.blocker.is_none());
        let journal = state.get_task_journal(&task_id, 10).await.unwrap();
        assert!(journal.iter().any(|entry| {
            entry.entry_type == "unblocked" && entry.body.contains("Found alternative approach")
        }));
    }

    #[tokio::test]
    async fn blocked_task_cannot_be_overwritten_as_completed() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());
        tool.call(
            &json!({
                "action": "create_task",
                "description": "Inspect deployment tooling"
            })
            .to_string(),
        )
        .await
        .unwrap();
        let task_id = state.get_tasks_for_goal(&goal_id).await.unwrap()[0]
            .id
            .clone();
        tool.call(
            &json!({
                "action": "update_task",
                "task_id": &task_id,
                "status": "blocked",
                "result": "A preflight check did not run"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "update_task",
                    "task_id": &task_id,
                    "status": "completed",
                    "result": "Done"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Use resolve_blocker"));
        assert_eq!(
            state.get_task(&task_id).await.unwrap().unwrap().status,
            "blocked"
        );
    }

    #[tokio::test]
    async fn test_resolve_blocker_not_blocked() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

        tool.call(
            &json!({
                "action": "create_task",
                "description": "Normal task"
            })
            .to_string(),
        )
        .await
        .unwrap();

        let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
        let task_id = tasks[0].id.clone();

        let result = tool
            .call(
                &json!({
                    "action": "resolve_blocker",
                    "task_id": &task_id
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("not blocked"));
    }

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
        assert!(super::validate_no_cycles(
            &[task_a.clone(), task_b.clone()],
            "c",
            &["a".to_string()]
        )
        .is_ok());

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
