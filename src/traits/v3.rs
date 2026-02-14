use serde::{Deserialize, Serialize};

/// Snapshot of a goal's token budget state.
#[derive(Debug, Clone)]
pub struct GoalTokenBudgetStatus {
    #[allow(dead_code)] // Reserved for future per-check budget enforcement.
    pub budget_per_check: Option<i64>,
    pub budget_daily: Option<i64>,
    pub tokens_used_today: i64,
}

// ==================== V3 Orchestration Data Model ====================

/// A V3 goal — a tracked, potentially long-running objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalV3 {
    pub id: String,
    pub description: String,
    /// "finite" (one-shot) or "continuous" (monitoring/recurring)
    pub goal_type: String,
    /// "pending", "pending_confirmation", "active", "paused", "completed", "failed", "cancelled"
    pub status: String,
    /// "low", "medium", "high", "critical"
    pub priority: String,
    /// Success/completion conditions (human-readable)
    pub conditions: Option<String>,
    /// Cron schedule for scheduled goals (continuous or deferred finite)
    pub schedule: Option<String>,
    /// JSON context blob (original request, constraints, etc.)
    pub context: Option<String>,
    /// JSON array of resource references (files, URLs, etc.)
    pub resources: Option<String>,
    /// Max tokens per check (for continuous goals)
    pub budget_per_check: Option<i64>,
    /// Max tokens per day for this goal
    pub budget_daily: Option<i64>,
    /// Tokens used today (reset daily)
    pub tokens_used_today: i64,
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
}

impl GoalV3 {
    /// Create a new finite (one-shot) goal from a user request.
    pub fn new_finite(description: &str, session_id: &str) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.to_string(),
            goal_type: "finite".to_string(),
            status: "active".to_string(),
            priority: "medium".to_string(),
            conditions: None,
            schedule: None,
            context: None,
            resources: None,
            // Safety defaults: generous enough for normal usage, but prevents
            // runaway autonomous execution from going unbounded.
            budget_per_check: Some(100_000),
            budget_daily: Some(500_000),
            tokens_used_today: 0,
            last_useful_action: None,
            created_at: now.clone(),
            updated_at: now,
            completed_at: None,
            parent_goal_id: None,
            session_id: session_id.to_string(),
            notified_at: None,
            notification_attempts: 0,
            dispatch_failures: 0,
        }
    }

    /// Create a deferred one-shot finite goal pending user confirmation.
    pub fn new_deferred_finite(description: &str, session_id: &str, schedule: &str) -> Self {
        let mut goal = Self::new_finite(description, session_id);
        goal.status = "pending_confirmation".to_string();
        goal.schedule = Some(schedule.to_string());
        goal
    }

    /// Create a new continuous (evergreen) goal with a cron schedule.
    pub fn new_continuous(
        description: &str,
        session_id: &str,
        schedule: &str,
        budget_per_check: Option<i64>,
        budget_daily: Option<i64>,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.to_string(),
            goal_type: "continuous".to_string(),
            status: "active".to_string(),
            priority: "low".to_string(),
            conditions: None,
            schedule: Some(schedule.to_string()),
            context: None,
            resources: None,
            // Apply defaults if caller omitted budgets.
            budget_per_check: budget_per_check.or(Some(50_000)),
            budget_daily: budget_daily.or(Some(200_000)),
            tokens_used_today: 0,
            last_useful_action: None,
            created_at: now.clone(),
            updated_at: now,
            completed_at: None,
            parent_goal_id: None,
            session_id: session_id.to_string(),
            notified_at: None,
            notification_attempts: 0,
            dispatch_failures: 0,
        }
    }

    /// Create a continuous goal pending user confirmation.
    pub fn new_continuous_pending(
        description: &str,
        session_id: &str,
        schedule: &str,
        budget_per_check: Option<i64>,
        budget_daily: Option<i64>,
    ) -> Self {
        let mut goal = Self::new_continuous(
            description,
            session_id,
            schedule,
            budget_per_check,
            budget_daily,
        );
        goal.status = "pending_confirmation".to_string();
        goal
    }
}

/// A V3 task — a discrete unit of work within a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used in Phase 2; StateStore methods and SQLite impl ready
pub struct TaskV3 {
    pub id: String,
    pub goal_id: String,
    pub description: String,
    /// "pending", "claimed", "running", "completed", "failed", "blocked"
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

/// A V3 task activity log entry — records tool calls and results within a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Used in Phase 2; StateStore methods and SQLite impl ready
pub struct TaskActivityV3 {
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
        }
    }
}

