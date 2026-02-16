use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::traits::{StateStore, Task, Tool, ToolRole};

/// Tool for task leads to manage tasks within their assigned goal.
pub struct ManageGoalTasksTool {
    goal_id: String,
    state: Arc<dyn StateStore>,
}

impl ManageGoalTasksTool {
    pub fn new(goal_id: String, state: Arc<dyn StateStore>) -> Self {
        Self { goal_id, state }
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
            "description": "Manage tasks within your assigned goal. Use create_task to break work into steps, claim_task before spawning an executor, list_tasks to check progress, update_task to record results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create_task", "list_tasks", "update_task", "claim_task", "retry_task", "resolve_blocker", "complete_goal", "fail_goal"],
                        "description": "Action to perform"
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID (for update_task, claim_task, retry_task)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description (for create_task)"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Task priority"
                    },
                    "task_order": {
                        "type": "integer",
                        "description": "Execution order within the goal"
                    },
                    "parallel_group": {
                        "type": "string",
                        "description": "Group name for tasks that can run concurrently"
                    },
                    "depends_on": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Task IDs this task depends on"
                    },
                    "idempotent": {
                        "type": "boolean",
                        "description": "Whether this task is safe to retry"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "running", "completed", "failed", "blocked"],
                        "description": "New status (for update_task)"
                    },
                    "result": {
                        "type": "string",
                        "description": "Result text (for update_task with completed status)"
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message (for update_task with failed status)"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary (for complete_goal/fail_goal)"
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Agent identifier (for claim_task)"
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageGoalTasksArgs = serde_json::from_str(arguments)?;

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
}

/// Check if all dependencies for a task are completed. Returns Ok(()) if all are met,
/// or an error message string describing which dependencies are unmet.
async fn check_dependencies_met(state: &dyn StateStore, task: &Task) -> Result<(), String> {
    if let Some(ref deps_json) = task.depends_on {
        if let Ok(dep_ids) = serde_json::from_str::<Vec<String>>(deps_json) {
            for dep_id in &dep_ids {
                if let Ok(Some(dep_task)) = state.get_task(dep_id).await {
                    if dep_task.status != "completed" {
                        return Err(format!(
                            "dependency {} not completed (status: {})",
                            dep_id, dep_task.status
                        ));
                    }
                }
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
        let description = args
            .description
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("create_task requires 'description'"))?;

        let now = chrono::Utc::now().to_rfc3339();
        let task_id = uuid::Uuid::new_v4().to_string();

        // Validate dependencies don't create cycles
        if let Some(ref dep_ids) = args.depends_on {
            if !dep_ids.is_empty() {
                let existing = self.state.get_tasks_for_goal(&self.goal_id).await?;
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
            idempotent: args.idempotent.unwrap_or(false),
            retry_count: 0,
            max_retries: 3,
            created_at: now,
            started_at: None,
            completed_at: None,
        };

        self.state.create_task(&task).await?;
        info!(goal_id = %self.goal_id, task_id = %task.id, "Created task");

        Ok(format!(
            "Created task {} (order: {}, priority: {}): {}",
            task.id, task.task_order, task.priority, task.description
        ))
    }

    async fn list_tasks(&self) -> anyhow::Result<String> {
        let tasks = self.state.get_tasks_for_goal(&self.goal_id).await?;

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

            let result_suffix = task
                .result
                .as_deref()
                .map(|r| format!(" → {}", &r[..200.min(r.len())]))
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
        let task_id = args
            .task_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("update_task requires 'task_id'"))?;

        let Some(mut task) = self.state.get_task(task_id).await? else {
            return Ok(self.task_not_found_message(task_id).await);
        };

        if task.goal_id != self.goal_id {
            anyhow::bail!("Task {} does not belong to goal {}", task_id, self.goal_id);
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
            if status == "completed" || status == "failed" {
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
            task.error = Some(error.clone());
        }

        self.state.update_task(&task).await?;
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
        let task_id = args
            .task_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("claim_task requires 'task_id'"))?;

        let Some(task) = self.state.get_task(task_id).await? else {
            return Ok(self.task_not_found_message(task_id).await);
        };

        if task.goal_id != self.goal_id {
            anyhow::bail!("Task {} does not belong to goal {}", task_id, self.goal_id);
        }

        // Check dependency satisfaction
        if let Err(reason) = check_dependencies_met(self.state.as_ref(), &task).await {
            return Ok(format!("Cannot claim task {}: {}", task_id, reason));
        }

        let agent_id = args.agent_id.as_deref().unwrap_or("executor");
        let claimed = self.state.claim_task(task_id, agent_id).await?;
        if claimed {
            info!(goal_id = %self.goal_id, task_id, agent_id, "Claimed task");
            Ok(format!("Claimed task {} for agent {}", task_id, agent_id))
        } else {
            Ok(format!(
                "Failed to claim task {} — may already be claimed or not pending",
                task_id
            ))
        }
    }

    async fn retry_task(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let task_id = args
            .task_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("retry_task requires 'task_id'"))?;

        let Some(mut task) = self.state.get_task(task_id).await? else {
            return Ok(self.task_not_found_message(task_id).await);
        };

        if task.goal_id != self.goal_id {
            anyhow::bail!("Task {} does not belong to goal {}", task_id, self.goal_id);
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

        task.retry_count += 1;
        task.status = "pending".to_string();
        task.error = None;
        task.blocker = None;
        task.agent_id = None;
        task.started_at = None;
        task.completed_at = None;
        self.state.update_task(&task).await?;

        info!(
            goal_id = %self.goal_id,
            task_id,
            retry_count = task.retry_count,
            max_retries = task.max_retries,
            "Retried task"
        );

        Ok(format!(
            "Task {} reset to pending for retry ({}/{})",
            task_id, task.retry_count, task.max_retries
        ))
    }

    async fn complete_goal(&self, args: &ManageGoalTasksArgs) -> anyhow::Result<String> {
        let mut goal = self
            .state
            .get_goal(&self.goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Goal not found: {}", self.goal_id))?;

        goal.status = "completed".to_string();
        goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
        goal.updated_at = chrono::Utc::now().to_rfc3339();

        self.state.update_goal(&goal).await?;
        info!(goal_id = %self.goal_id, "Goal completed");

        let summary = args
            .summary
            .as_deref()
            .unwrap_or("Goal completed successfully");
        Ok(format!("Goal {} completed: {}", self.goal_id, summary))
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
            .map(|r| &r[..r.len().min(500)])
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
        let task_id = args
            .task_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("resolve_blocker requires 'task_id'"))?;

        let Some(mut task) = self.state.get_task(task_id).await? else {
            return Ok(self.task_not_found_message(task_id).await);
        };

        if task.goal_id != self.goal_id {
            anyhow::bail!("Task {} does not belong to goal {}", task_id, self.goal_id);
        }
        if task.status != "blocked" {
            return Ok(format!(
                "Task {} is not blocked (status: {})",
                task_id, task.status
            ));
        }

        task.status = "pending".to_string();
        task.blocker = None;

        // Append resolution context if provided
        if let Some(resolution) = &args.result {
            task.context = Some(format!(
                "{}\nBlocker resolution: {}",
                task.context.as_deref().unwrap_or(""),
                resolution
            ));
        }

        self.state.update_task(&task).await?;
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

        goal.status = "failed".to_string();
        goal.updated_at = chrono::Utc::now().to_rfc3339();

        self.state.update_goal(&goal).await?;
        info!(goal_id = %self.goal_id, "Goal failed");

        // Cancel remaining pending/claimed tasks so they don't get re-dispatched
        let tasks = self
            .state
            .get_tasks_for_goal(&self.goal_id)
            .await
            .unwrap_or_default();
        let mut cancelled = 0;
        for task in &tasks {
            if task.status == "pending" || task.status == "claimed" {
                let mut t = task.clone();
                t.status = "completed".to_string();
                t.error = Some("Cancelled: parent goal explicitly failed".to_string());
                t.completed_at = Some(chrono::Utc::now().to_rfc3339());
                let _ = self.state.update_task(&t).await;
                cancelled += 1;
            }
        }
        if cancelled > 0 {
            info!(goal_id = %self.goal_id, cancelled, "Cancelled pending tasks for failed goal");
        }

        let summary = args.summary.as_deref().unwrap_or("Goal failed");
        Ok(format!(
            "Goal {} failed (cancelled {} pending tasks): {}",
            self.goal_id, cancelled, summary
        ))
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
    async fn test_complete_goal_action() {
        let (state, goal_id) = setup_test_state().await;
        let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

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
        assert!(task.context.unwrap().contains("Found alternative approach"));
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
