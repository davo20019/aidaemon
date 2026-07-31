use serde::{Deserialize, Serialize};

pub const DEFAULT_PROJECT_ID: &str = "default";

/// Snapshot of a goal's token budget state.
#[derive(Debug, Clone)]
pub struct GoalTokenBudgetStatus {
    #[allow(dead_code)] // Reserved for future per-check budget enforcement.
    pub budget_per_check: Option<i64>,
    pub budget_daily: Option<i64>,
    pub tokens_used_today: i64,
}

/// Persisted runtime state for an active scheduled run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ScheduledRunHealth {
    pub evidence_gain_count: usize,
    pub total_successful_tool_calls: usize,
    pub stall_count: usize,
    pub consecutive_same_tool_count: usize,
    pub consecutive_same_tool_unique_args: usize,
    pub unrecovered_error_count: usize,
}

/// Persisted runtime state for an active scheduled run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduledRunState {
    pub goal_id: String,
    pub root_task_id: String,
    pub effective_budget_per_check: i64,
    pub tokens_used: i64,
    pub budget_extensions_count: usize,
    #[serde(default)]
    pub health: ScheduledRunHealth,
    pub created_at: String,
    pub updated_at: String,
}

// ==================== Goals + Tasks Data Model ====================

/// A goal — a tracked, potentially long-running objective.
///
/// Goals are stored in a single `goals` table with a `domain` that gates behavior:
/// - `orchestration`: can be scheduled/continuous, can have tasks, can be dispatched
/// - `personal`: tracked/injected/listed, never dispatched, usually no tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub description: String,
    /// "orchestration" (default) or "personal"
    pub domain: String,
    /// "finite" (one-shot) or "continuous" (monitoring/recurring)
    pub goal_type: String,
    /// "pending", "pending_confirmation", "active", "paused", "completed", "failed", "cancelled", "abandoned"
    pub status: String,
    /// "low", "medium", "high", "critical"
    pub priority: String,
    /// Success/completion conditions (human-readable)
    pub conditions: Option<String>,
    /// JSON context blob (original request, constraints, etc.)
    pub context: Option<String>,
    /// JSON array of resource references (files, URLs, etc.)
    pub resources: Option<String>,
    /// Max tokens per check (for continuous goals)
    pub budget_per_check: Option<i64>,
    /// Max tokens per day for this goal
    pub budget_daily: Option<i64>,
    /// Tokens used for the UTC day in `tokens_used_day` (reset daily).
    pub tokens_used_today: i64,
    /// UTC day anchor for `tokens_used_today` (YYYY-MM-DD).
    pub tokens_used_day: String,
    /// Timestamp of last meaningful action
    pub last_useful_action: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub completed_at: Option<String>,
    /// Parent goal ID for hierarchical decomposition
    pub parent_goal_id: Option<String>,
    /// Session where this goal was created
    pub session_id: String,
    /// Timestamp when user was notified of completion/failure (None = not yet notified)
    pub notified_at: Option<String>,
    /// Number of notification delivery attempts (gives up after 3)
    #[serde(default)]
    pub notification_attempts: i32,
    /// Consecutive dispatch cycles with no progress (circuit breaker: stalls at 3)
    #[serde(default)]
    pub dispatch_failures: i32,
    /// Personal-goal progress notes (append-only) stored as JSON array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_notes: Option<Vec<String>>,
    /// Optional episodic-memory provenance (personal goals).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_episode_id: Option<i64>,
    /// Optional legacy integer ID (for migrated pre-unification personal goals).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legacy_int_id: Option<i64>,
}

impl Goal {
    /// Create a new finite (one-shot) goal from a user request.
    pub fn new_finite(description: &str, session_id: &str) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let day = chrono::Utc::now().date_naive().to_string();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.to_string(),
            domain: "orchestration".to_string(),
            goal_type: "finite".to_string(),
            status: "active".to_string(),
            priority: "medium".to_string(),
            conditions: None,
            context: None,
            resources: None,
            // Safety defaults: generous enough for normal usage (including
            // sub-agent token overhead), but prevents runaway execution.
            budget_per_check: Some(100_000),
            budget_daily: Some(1_000_000),
            tokens_used_today: 0,
            tokens_used_day: day,
            last_useful_action: None,
            created_at: now.clone(),
            updated_at: now,
            completed_at: None,
            parent_goal_id: None,
            session_id: session_id.to_string(),
            notified_at: None,
            notification_attempts: 0,
            dispatch_failures: 0,
            progress_notes: None,
            source_episode_id: None,
            legacy_int_id: None,
        }
    }

    /// Create a new personal goal.
    ///
    /// Personal goals are tracked and injected (DM-only) but never dispatched
    /// as background work. Budgets are unset because they do not execute.
    pub fn new_personal(description: &str, session_id: &str) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let day = chrono::Utc::now().date_naive().to_string();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.to_string(),
            domain: "personal".to_string(),
            goal_type: "finite".to_string(),
            status: "active".to_string(),
            priority: "medium".to_string(),
            conditions: None,
            context: None,
            resources: None,
            budget_per_check: None,
            budget_daily: None,
            tokens_used_today: 0,
            tokens_used_day: day,
            last_useful_action: None,
            created_at: now.clone(),
            updated_at: now,
            completed_at: None,
            parent_goal_id: None,
            session_id: session_id.to_string(),
            notified_at: None,
            notification_attempts: 0,
            dispatch_failures: 0,
            progress_notes: Some(Vec::new()),
            source_episode_id: None,
            legacy_int_id: None,
        }
    }

    /// Create a deferred one-shot finite goal pending user confirmation.
    ///
    /// Scheduling is managed via `GoalSchedule` rows, not a goal column.
    pub fn new_deferred_finite(description: &str, session_id: &str) -> Self {
        let mut goal = Self::new_finite(description, session_id);
        goal.status = "pending_confirmation".to_string();
        goal
    }

    /// Create a new continuous (evergreen) goal.
    pub fn new_continuous(
        description: &str,
        session_id: &str,
        budget_per_check: Option<i64>,
        budget_daily: Option<i64>,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let day = chrono::Utc::now().date_naive().to_string();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.to_string(),
            domain: "orchestration".to_string(),
            goal_type: "continuous".to_string(),
            status: "active".to_string(),
            priority: "low".to_string(),
            conditions: None,
            context: None,
            resources: None,
            // Scheduled work carries task-lead and executor context overhead in
            // addition to the action itself. Observed write/build/deploy/verify
            // cycles exceed 300k even when healthy. Keep the budget bounded,
            // but leave headroom for one complete cycle and two daily attempts.
            budget_per_check: budget_per_check.or(Some(400_000)),
            budget_daily: budget_daily.or(Some(1_000_000)),
            tokens_used_today: 0,
            tokens_used_day: day,
            last_useful_action: None,
            created_at: now.clone(),
            updated_at: now,
            completed_at: None,
            parent_goal_id: None,
            session_id: session_id.to_string(),
            notified_at: None,
            notification_attempts: 0,
            dispatch_failures: 0,
            progress_notes: None,
            source_episode_id: None,
            legacy_int_id: None,
        }
    }

    /// Create a continuous goal pending user confirmation.
    pub fn new_continuous_pending(
        description: &str,
        session_id: &str,
        budget_per_check: Option<i64>,
        budget_daily: Option<i64>,
    ) -> Self {
        let mut goal =
            Self::new_continuous(description, session_id, budget_per_check, budget_daily);
        goal.status = "pending_confirmation".to_string();
        goal
    }
}

/// Goal schedule row — per-schedule state for a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalSchedule {
    pub id: String,
    pub goal_id: String,
    /// 5-field cron expression.
    pub cron_expr: String,
    /// Timezone label. Currently only `local` is supported.
    pub tz: String,
    /// User-provided schedule string (optional; for display/audit).
    pub original_schedule: Option<String>,
    /// "coalesce" (default) or "always_fire"
    pub fire_policy: String,
    pub is_one_shot: bool,
    pub is_paused: bool,
    pub last_run_at: Option<String>,
    pub next_run_at: String,
    pub created_at: String,
    pub updated_at: String,
}

/// A task — a discrete unit of work within a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used in Phase 2; StateStore methods and SQLite impl ready
pub struct Task {
    pub id: String,
    pub goal_id: String,
    pub description: String,
    /// "pending", "claimed", "running", "completed", "failed", "blocked",
    /// "skipped", or "superseded"
    pub status: String,
    /// "low", "medium", "high"
    pub priority: String,
    /// Execution order within the goal
    pub task_order: i32,
    /// Tasks in the same parallel group can run concurrently
    pub parallel_group: Option<String>,
    /// JSON array of task IDs this task depends on
    pub depends_on: Option<String>,
    /// Agent/executor ID that claimed this task
    pub agent_id: Option<String>,
    /// JSON context blob
    pub context: Option<String>,
    /// Result text on completion
    pub result: Option<String>,
    /// Error message on failure
    pub error: Option<String>,
    /// Blocker description if status is "blocked"
    pub blocker: Option<String>,
    /// Whether this task is safe to retry
    pub idempotent: bool,
    pub retry_count: i32,
    pub max_retries: i32,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

impl Task {
    /// SQLite contains legacy task rows whose optional error field is an empty
    /// string instead of NULL. Treat blank error text as absent everywhere task
    /// success is evaluated.
    pub fn has_error(&self) -> bool {
        self.error
            .as_deref()
            .is_some_and(|error| !error.trim().is_empty())
    }

    /// A blocker is unresolved work even if a stale coordinator has written a
    /// terminal-looking status. Treat legacy empty strings as absent, matching
    /// the normalization used for errors.
    pub fn has_blocker(&self) -> bool {
        self.blocker
            .as_deref()
            .is_some_and(|blocker| !blocker.trim().is_empty())
    }

    pub fn completed_successfully(&self) -> bool {
        self.status == "completed" && !self.has_error() && !self.has_blocker()
    }

    /// True when this task no longer represents required work for the current
    /// run. A superseded task is not itself a success, but a successful
    /// replacement means it must not poison the run's terminal outcome.
    pub fn satisfies_run_completion(&self) -> bool {
        self.completed_successfully() || matches!(self.status.as_str(), "skipped" | "superseded")
    }
}

/// A stable isolation boundary for goals, runs, worker policies, and channel views.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkProject {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// One execution of a goal. Finite goals normally have one run; each scheduled
/// firing gets a distinct run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalRun {
    pub id: String,
    pub project_id: String,
    pub goal_id: String,
    /// "finite", "scheduled", "manual", or "legacy"
    pub trigger_type: String,
    pub schedule_id: Option<String>,
    pub root_task_id: Option<String>,
    /// "pending", "running", "completed", "failed", "blocked", or "cancelled"
    pub status: String,
    pub outcome_summary: Option<String>,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl GoalRun {
    pub fn new(goal_id: &str, project_id: &str, trigger_type: &str) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            goal_id: goal_id.to_string(),
            trigger_type: trigger_type.to_string(),
            schedule_id: None,
            root_task_id: None,
            status: "running".to_string(),
            outcome_summary: None,
            started_at: now.clone(),
            completed_at: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }
}

/// Durable execution policy. The profile is stable configuration; a task
/// attempt and its worker instance remain distinct identities.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerProfile {
    pub id: String,
    pub project_id: Option<String>,
    pub name: String,
    pub specialist: String,
    pub model: Option<String>,
    pub tools_json: Option<String>,
    pub max_iterations: Option<i64>,
    pub tool_budget: Option<i64>,
    pub timeout_secs: Option<i64>,
    pub max_concurrency: i64,
    /// "shared", "isolated", or "worktree"
    pub workspace_policy: String,
    pub memory_scope: String,
    pub version: i64,
    pub enabled: bool,
    pub created_at: String,
    pub updated_at: String,
}

/// One fenced claim of a task by one worker instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskAttempt {
    pub id: String,
    pub task_id: String,
    pub goal_run_id: String,
    pub worker_profile_id: Option<String>,
    pub worker_instance_id: String,
    /// Local fencing token. It is never included in user-facing output.
    #[serde(skip_serializing)]
    pub lease_token: String,
    /// "claimed", "running", "completed", "failed", "blocked", "expired",
    /// "needs_verification", or "cancelled"
    pub status: String,
    pub lease_expires_at: String,
    pub last_heartbeat_at: String,
    pub workspace_id: Option<String>,
    pub started_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandoffArtifact {
    /// "path", "commit", "url", "message", or another explicit artifact kind.
    pub kind: String,
    pub reference: String,
    pub digest: Option<String>,
    pub metadata: Option<String>,
}

/// Low-volume, structured output from one attempt for the next worker or human.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskHandoff {
    pub id: String,
    pub task_id: String,
    pub attempt_id: String,
    pub summary: String,
    pub artifacts: Vec<HandoffArtifact>,
    pub verification: Vec<String>,
    pub remaining_risk: Option<String>,
    pub next_step: Option<String>,
    pub created_at: String,
}

/// Append-only collaboration and audit record. Tool/LLM telemetry remains in
/// `TaskActivity`; this journal is for durable decisions and human interaction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskJournalEntry {
    pub id: String,
    pub project_id: String,
    pub goal_id: String,
    pub goal_run_id: String,
    pub task_id: Option<String>,
    pub attempt_id: Option<String>,
    /// "comment", "blocked", "unblocked", "assigned", "handoff",
    /// "lease_lost", "workspace", or "transition"
    pub entry_type: String,
    /// "human", "agent", or "system"
    pub actor_type: String,
    pub actor_id: String,
    pub source_channel: Option<String>,
    pub body: String,
    pub payload: Option<String>,
    pub created_at: String,
}

impl TaskJournalEntry {
    pub fn new(
        project_id: &str,
        goal_id: &str,
        goal_run_id: &str,
        entry_type: &str,
        actor_type: &str,
        actor_id: &str,
        body: &str,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: project_id.to_string(),
            goal_id: goal_id.to_string(),
            goal_run_id: goal_run_id.to_string(),
            task_id: None,
            attempt_id: None,
            entry_type: entry_type.to_string(),
            actor_type: actor_type.to_string(),
            actor_id: actor_id.to_string(),
            source_channel: None,
            body: body.to_string(),
            payload: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn with_task(mut self, task_id: Option<&str>, attempt_id: Option<&str>) -> Self {
        self.task_id = task_id.map(ToOwned::to_owned);
        self.attempt_id = attempt_id.map(ToOwned::to_owned);
        self
    }

    pub fn with_source_channel(mut self, source_channel: Option<&str>) -> Self {
        self.source_channel = source_channel.map(ToOwned::to_owned);
        self
    }
}

/// Realized workspace for one attempt. Workspaces are preserved after execution
/// until explicitly released so downstream integration can consume them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskWorkspace {
    pub id: String,
    pub task_id: String,
    pub attempt_id: String,
    pub backend_id: String,
    pub policy: String,
    pub root_path: String,
    pub branch_name: Option<String>,
    pub base_ref: Option<String>,
    pub head_ref: Option<String>,
    /// "active", "preserved", "released", or "failed"
    pub status: String,
    pub created_at: String,
    pub released_at: Option<String>,
}

/// Attempt-scoped mutation applied only while the supplied lease is current.
#[derive(Debug, Clone, Default)]
pub struct TaskAttemptPatch {
    pub status: String,
    pub result: Option<String>,
    pub error: Option<String>,
    pub blocker: Option<String>,
    pub context: Option<String>,
    pub handoff: Option<TaskHandoff>,
}

/// Goal-level work projection used by chat and dashboard views.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkGoalSummary {
    pub project_id: String,
    pub goal_id: String,
    pub description: String,
    pub goal_status: String,
    pub run_id: Option<String>,
    pub run_status: Option<String>,
    pub waiting: i64,
    pub ready: i64,
    pub in_progress: i64,
    pub blocked: i64,
    pub needs_attention: i64,
    pub done: i64,
    pub updated_at: String,
}

/// Task-level work projection. `lane` is derived from scheduler state and is
/// never persisted as a second status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkTaskSummary {
    pub project_id: String,
    pub goal_id: String,
    pub goal_description: String,
    pub goal_run_id: String,
    pub task_id: String,
    pub description: String,
    pub status: String,
    pub lane: String,
    pub priority: String,
    pub worker_profile: Option<String>,
    pub worker_instance_id: Option<String>,
    pub lease_expires_at: Option<String>,
    pub blocker: Option<String>,
    pub updated_at: String,
}

/// A task activity log entry — records tool calls and results within a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used in Phase 2; StateStore methods and SQLite impl ready
pub struct TaskActivity {
    pub id: i64,
    pub task_id: String,
    /// "tool_call", "tool_result", "llm_call", "status_change"
    pub activity_type: String,
    pub tool_name: Option<String>,
    pub tool_args: Option<String>,
    pub result: Option<String>,
    pub success: Option<bool>,
    pub tokens_used: Option<i64>,
    pub created_at: String,
}

/// A queued notification awaiting delivery to the user.
///
/// Notifications are queued in SQLite when the originating channel is unavailable.
/// Retention depends on priority: status updates expire after 24 hours,
/// critical notifications persist indefinitely until delivered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationEntry {
    pub id: String,
    pub goal_id: String,
    pub session_id: String,
    /// "completed", "failed", "escalation", "progress", "stalled", "evergreen_alert", "token_alert"
    pub notification_type: String,
    /// "critical" (persist indefinitely) or "status_update" (expire after 24h)
    pub priority: String,
    pub message: String,
    pub created_at: String,
    pub delivered_at: Option<String>,
    pub attempts: i32,
    /// When this notification expires (None = never, for critical notifications)
    pub expires_at: Option<String>,
    /// Optional durable work item associated with this notification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Opaque token reserved for channel-native actions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_token: Option<String>,
}

impl NotificationEntry {
    /// Create a new notification entry.
    pub fn new(goal_id: &str, session_id: &str, notification_type: &str, message: &str) -> Self {
        let now = chrono::Utc::now();
        let priority = match notification_type {
            "completed" | "failed" | "escalation" | "evergreen_alert" | "token_alert" => "critical",
            _ => "status_update",
        };
        let expires_at = if priority == "status_update" {
            Some((now + chrono::Duration::hours(24)).to_rfc3339())
        } else {
            None // critical notifications never expire
        };
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            session_id: session_id.to_string(),
            notification_type: notification_type.to_string(),
            priority: priority.to_string(),
            message: message.to_string(),
            created_at: now.to_rfc3339(),
            delivered_at: None,
            attempts: 0,
            expires_at,
            task_id: None,
            action_token: None,
        }
    }

    pub fn with_task(mut self, task_id: &str) -> Self {
        self.task_id = Some(task_id.to_string());
        self.action_token = Some(uuid::Uuid::new_v4().to_string());
        self
    }
}
