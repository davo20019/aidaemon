use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use std::collections::HashMap;

use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::types::{ApprovalResponse, FactPrivacy, MediaMessage, StatusUpdate};

/// A message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub session_id: String,
    pub role: String, // "system", "user", "assistant", "tool"
    pub content: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub tool_calls_json: Option<String>, // serialized Vec<ToolCall>
    pub created_at: DateTime<Utc>,
    #[serde(default = "default_importance")]
    pub importance: f32,
    #[serde(skip)] // Don't serialize embedding to JSON (client doesn't need it)
    #[allow(dead_code)] // Reserved for semantic-memory paths that may be feature-gated.
    pub embedding: Option<Vec<f32>>,
}

fn default_importance() -> f32 {
    0.5
}

/// A single tool call as returned by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // JSON string
    /// Opaque extra fields from the provider (e.g. Gemini 3 thought signatures).
    /// Preserved and sent back verbatim in conversation history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<Value>,
}

/// A fact stored in Layer 2 memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: i64,
    pub category: String,
    pub key: String,
    pub value: String,
    pub source: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub recall_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<DateTime<Utc>>,
    /// Channel where this fact originated (e.g., "slack:C12345"). None for legacy/global facts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel_id: Option<String>,
    /// Privacy level controlling where this fact can be recalled.
    #[serde(default = "default_fact_privacy")]
    pub privacy: FactPrivacy,
}

fn default_fact_privacy() -> FactPrivacy {
    FactPrivacy::Global
}

/// An episode representing a session summary (episodic memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: i64,
    pub session_id: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topics: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotional_tone: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    pub importance: f32,
    pub recall_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<DateTime<Utc>>,
    pub message_count: i32,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    /// Channel where this episode occurred. None for legacy episodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel_id: Option<String>,
}

/// A goal being tracked over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: i64,
    pub description: String,
    pub status: String,   // "active", "completed", "abandoned"
    pub priority: String, // "low", "medium", "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_notes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_episode_id: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
}

/// User communication style preferences.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserProfile {
    pub id: i64,
    pub verbosity_preference: String, // "brief", "medium", "detailed"
    pub explanation_depth: String,    // "minimal", "moderate", "thorough"
    pub tone_preference: String,      // "casual", "neutral", "formal"
    pub emoji_preference: String,     // "none", "minimal", "frequent"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_session_length: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_hours: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub common_workflows: Option<Vec<String>>,
    pub asks_before_acting: bool,
    pub prefers_explanations: bool,
    pub likes_suggestions: bool,
    pub updated_at: DateTime<Utc>,
}

/// A detected behavior pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPattern {
    pub id: i64,
    pub pattern_type: String, // "sequence", "trigger", "habit"
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    pub confidence: f32,
    pub occurrence_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_seen_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// A learned procedure (procedural memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub id: i64,
    pub name: String,
    pub trigger_pattern: String,
    pub steps: Vec<String>,
    pub success_count: i32,
    pub failure_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Expertise level in a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expertise {
    pub id: i64,
    pub domain: String,
    pub tasks_attempted: i32,
    pub tasks_succeeded: i32,
    pub tasks_failed: i32,
    pub current_level: String, // "novice", "competent", "proficient", "expert"
    pub confidence_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub common_errors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_task_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A learned error-solution pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSolution {
    pub id: i64,
    pub error_pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    pub solution_summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solution_steps: Option<Vec<String>>,
    pub success_count: i32,
    pub failure_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// A conversation summary for a session, used by context window management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub session_id: String,
    pub summary: String,
    pub message_count: usize,
    pub last_message_id: String,
    pub updated_at: DateTime<Utc>,
}

/// An event emitted by triggers or channels.
#[derive(Debug, Clone)]
pub struct Event {
    pub source: String,
    pub session_id: String,
    pub content: String,
    /// Whether this event originates from an explicitly trusted source
    /// (e.g., a scheduled task marked `trusted = true` in config).
    pub trusted: bool,
}

/// A person in the owner's social circle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    pub id: i64,
    pub name: String,
    pub aliases: Vec<String>,
    pub relationship: Option<String>,
    pub platform_ids: HashMap<String, String>,
    pub notes: Option<String>,
    pub communication_style: Option<String>,
    pub language_preference: Option<String>,
    pub last_interaction_at: Option<DateTime<Utc>>,
    pub interaction_count: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A fact about a person (birthday, preference, interest, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonFact {
    pub id: i64,
    pub person_id: i64,
    pub category: String,
    pub key: String,
    pub value: String,
    pub source: String,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// V3 role assigned to an agent for role-based tool scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    /// Root agent — routes, classifies, full tool access (legacy behavior).
    Orchestrator,
    /// Plans & delegates — management tools only.
    TaskLead,
    /// Executes a single task — action tools + report_blocker.
    Executor,
}

/// Categorization of a tool for V3 role-based scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRole {
    /// Terminal, web_search, web_fetch, browser, etc.
    Action,
    /// ManageGoalTasksTool, ReportBlockerTool — task lead tools.
    Management,
    /// SystemInfoTool, RememberFactTool — available to all roles.
    Universal,
}

/// Safety and execution metadata for policy-driven tool selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub read_only: bool,
    pub external_side_effect: bool,
    pub needs_approval: bool,
    pub idempotent: bool,
    pub high_impact_write: bool,
}

impl Default for ToolCapabilities {
    fn default() -> Self {
        Self {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }
}

/// Tool trait — system tools, terminal, MCP-proxied tools.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// Returns the OpenAI-format function schema as a JSON Value.
    fn schema(&self) -> Value;
    /// Execute the tool with the given JSON arguments string, returns result text.
    async fn call(&self, arguments: &str) -> anyhow::Result<String>;

    /// Execute the tool with access to a status update channel for streaming feedback.
    /// Default implementation just calls `call()` - override for tools that emit progress.
    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        // Default: ignore status channel and just call the basic method
        let _ = status_tx;
        self.call(arguments).await
    }

    /// Categorize this tool for V3 role-based scoping.
    /// Default: Action (most tools are action tools).
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    /// Capability metadata used by the execution policy and risk gate.
    /// Defaults are intentionally conservative.
    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::default()
    }
}

/// Model provider — sends messages + tool defs to an LLM, gets back response.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
    ) -> anyhow::Result<ProviderResponse>;

    /// List available models from the provider. Returns model ID strings.
    async fn list_models(&self) -> anyhow::Result<Vec<String>>;
}

/// Token usage statistics from an LLM API response.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub model: String,
}

/// The LLM's response: either content text, tool calls, or both.
#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<TokenUsage>,
    /// Internal reasoning from thinking models (e.g. Gemini thought parts).
    /// Not shown to users directly but available as fallback when content is empty.
    pub thinking: Option<String>,
    /// Optional provider-specific note about why no useful output was returned
    /// (for example Gemini finishReason/safety blocking metadata).
    pub response_note: Option<String>,
}

/// A record of token usage from the database.
#[derive(Debug, Clone)]
pub struct TokenUsageRecord {
    pub model: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    #[allow(dead_code)] // Used for database queries
    pub created_at: String,
}

/// Snapshot of a goal's token budget state.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Goal budget enforcement wiring is staged.
pub struct GoalTokenBudgetStatus {
    pub budget_per_check: Option<i64>,
    pub budget_daily: Option<i64>,
    pub tokens_used_today: i64,
}

mod state_store;
pub use state_store::*;

/// Import this in modules that call store-trait methods on concrete types.
///
/// `StateStore` is a facade (supertrait) used for trait objects, but Rust still
/// requires the defining trait to be in scope for method-call syntax.
pub mod store_prelude {
    #![allow(unused_imports)]
    pub use super::{
        ConversationSummaryStore, DynamicBotStore, DynamicCliAgentStore, DynamicMcpServerStore,
        EpisodeStore, FactStore, HealthCheckStore, LearningStore, LegacyGoalStore, MessageStore,
        NotificationStore, OAuthStore, PeopleStore, SessionChannelStore, SettingsStore, SkillStore,
        StateStore, TokenUsageStore, V3Store,
    };
}

/// An OAuth connection to an external service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConnection {
    pub id: i64,
    pub service: String,
    pub auth_type: String,
    pub username: Option<String>,
    pub scopes: String,
    pub token_expires_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A dynamically added skill (stored in database).
/// Deprecated: kept for migration compatibility. Use filesystem skills instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicSkill {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub triggers_json: String, // JSON array of trigger strings
    pub body: String,
    pub source: String, // "url", "inline", "auto", "registry"
    pub source_url: Option<String>,
    pub enabled: bool,
    pub version: Option<String>,
    pub created_at: String,
    /// JSON-serialized Vec<ResourceEntry> for directory-based skills
    #[serde(default = "default_empty_json")]
    pub resources_json: String,
}

/// A skill draft created by auto-promotion, pending user review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDraft {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub triggers_json: String,
    pub body: String,
    pub source_procedure: String,
    pub status: String, // "pending", "approved", "dismissed"
    pub created_at: String,
}

fn default_empty_json() -> String {
    "[]".to_string()
}

/// A dynamically added bot configuration (stored in database).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBot {
    pub id: i64,
    pub channel_type: String, // "telegram", "discord", "slack"
    pub bot_token: String,
    pub app_token: Option<String>,     // Only for Slack
    pub allowed_user_ids: Vec<String>, // Stored as JSON
    pub extra_config: String,          // JSON for channel-specific settings
    pub created_at: String,
}

/// A dynamically added MCP server configuration (stored in database).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicMcpServer {
    pub id: i64,
    pub name: String,
    pub command: String,
    pub args_json: String,     // JSON array of argument strings
    pub env_keys_json: String, // JSON array of env var key names (values in keychain)
    pub triggers_json: String, // JSON array of trigger keywords
    pub enabled: bool,
    pub created_at: String,
}

/// A dynamically added CLI agent configuration (stored in database).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicCliAgent {
    pub id: i64,
    pub name: String,
    pub command: String,
    pub args_json: String, // JSON array of argument strings
    pub description: String,
    pub timeout_secs: Option<u64>,
    pub max_output_chars: Option<usize>,
    pub enabled: bool,
    pub created_at: String,
}

/// A record of a CLI agent invocation for logging/auditing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliAgentInvocation {
    pub id: i64,
    pub session_id: String,
    pub agent_name: String,
    pub prompt_summary: String,
    pub working_dir: Option<String>,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub exit_code: Option<i32>,
    pub output_summary: Option<String>,
    pub success: Option<bool>,
    pub duration_secs: Option<f64>,
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
            budget_per_check: None,
            budget_daily: None,
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
            budget_per_check,
            budget_daily,
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

/// Capabilities that vary by channel (Telegram, WhatsApp, SMS, Web, etc.).
///
/// Used by the agent and hub to adapt output format for each channel.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ChannelCapabilities {
    /// Whether the channel supports markdown/rich text formatting.
    pub markdown: bool,
    /// Whether the channel supports inline buttons (e.g., Telegram inline keyboards).
    pub inline_buttons: bool,
    /// Whether the channel supports sending media (photos, files).
    pub media: bool,
    /// Maximum message length in characters. Messages longer than this will be split.
    pub max_message_len: usize,
}

/// A communication channel (Telegram, WhatsApp, Web, SMS, etc.).
///
/// Each implementation handles transport-specific details for sending messages,
/// media, and approval requests. New channels (e.g., Discord, Slack, SMS) only
/// need to implement this trait to integrate with aidaemon.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Unique name for this channel (e.g., "telegram", "telegram:my_bot", "discord").
    /// For multi-bot setups, includes the bot username (e.g., "telegram:coding_bot").
    fn name(&self) -> String;

    /// Channel capabilities — used to adapt output format.
    fn capabilities(&self) -> ChannelCapabilities;

    /// Send a text message to a session.
    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()>;

    /// Send media (photo/file) to a session.
    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()>;

    /// Request user approval for a command. Blocks until the user responds.
    /// Channels without inline buttons should fall back to text-based approval
    /// (e.g., "Reply YES, ALWAYS, or NO").
    ///
    /// The `risk_level`, `warnings`, and `permission_mode` parameters provide
    /// context about why approval is being requested and which buttons to show.
    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
    ) -> anyhow::Result<ApprovalResponse>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_capabilities_default_is_conservative() {
        let caps = ToolCapabilities::default();
        assert!(!caps.read_only);
        assert!(!caps.external_side_effect);
        assert!(caps.needs_approval);
        assert!(!caps.idempotent);
        assert!(!caps.high_impact_write);
    }
}
