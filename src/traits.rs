use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use std::collections::HashMap;

use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::types::{ApprovalResponse, ChannelVisibility, FactPrivacy, MediaMessage, StatusUpdate};

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

/// Persistent state store (SQLite + working memory).
#[async_trait]
pub trait StateStore: Send + Sync {
    /// Append a message to the session history (both DB and working memory).
    async fn append_message(&self, msg: &Message) -> anyhow::Result<()>;
    /// Get recent messages for a session from working memory.
    async fn get_history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>>;
    /// Upsert a fact with channel provenance and privacy level.
    async fn upsert_fact(
        &self,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<()>;
    /// Get all facts, optionally filtered by category.
    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>>;
    /// Get facts semantically relevant to a query, falling back to get_facts on error.
    async fn get_relevant_facts(&self, _query: &str, max: usize) -> anyhow::Result<Vec<Fact>> {
        // Default: return all facts (capped). Implementations can override with semantic filtering.
        let mut facts = self.get_facts(None).await?;
        facts.truncate(max);
        Ok(facts)
    }
    /// Get facts for a specific channel context, respecting privacy levels.
    async fn get_relevant_facts_for_channel(
        &self,
        _query: &str,
        max: usize,
        _channel_id: Option<&str>,
        _visibility: ChannelVisibility,
    ) -> anyhow::Result<Vec<Fact>> {
        self.get_relevant_facts(_query, max).await
    }
    /// Get cross-channel hints: channel-scoped facts from OTHER channels relevant to the query.
    async fn get_cross_channel_hints(
        &self,
        _query: &str,
        _current_channel_id: &str,
        _max: usize,
    ) -> anyhow::Result<Vec<Fact>> {
        Ok(vec![])
    }
    /// Update a fact's privacy level (e.g., channel → global after approval).
    async fn update_fact_privacy(
        &self,
        _fact_id: i64,
        _privacy: FactPrivacy,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    /// Soft-delete a fact by superseding it.
    async fn delete_fact(&self, _fact_id: i64) -> anyhow::Result<()> {
        Ok(())
    }
    /// Get all active facts with provenance info for memory management display.
    async fn get_all_facts_with_provenance(&self) -> anyhow::Result<Vec<Fact>> {
        self.get_facts(None).await
    }
    /// Get episodes for a specific channel context.
    async fn get_relevant_episodes_for_channel(
        &self,
        _query: &str,
        _limit: usize,
        _channel_id: Option<&str>,
    ) -> anyhow::Result<Vec<Episode>> {
        Ok(vec![])
    }
    /// Get context using Tri-Hybrid retrieval (Recency + Vector + Salience).
    /// Default implementation just calls get_history.
    async fn get_context(
        &self,
        session_id: &str,
        _query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        self.get_history(session_id, limit).await
    }
    /// Clear conversation history for a session (working memory + DB messages).
    /// Facts are preserved.
    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()>;
    /// Record token usage from an LLM call.
    async fn record_token_usage(
        &self,
        _session_id: &str,
        _usage: &TokenUsage,
    ) -> anyhow::Result<()> {
        Ok(()) // default no-op
    }
    /// Get token usage records since a given datetime string (ISO 8601).
    async fn get_token_usage_since(&self, _since: &str) -> anyhow::Result<Vec<TokenUsageRecord>> {
        Ok(vec![]) // default no-op
    }

    // ==================== Extended Memory Methods ====================
    // These have default empty implementations for backwards compatibility.
    // SqliteStateStore overrides them with actual implementations.

    /// Get episodes relevant to a query.
    async fn get_relevant_episodes(
        &self,
        _query: &str,
        _limit: usize,
    ) -> anyhow::Result<Vec<Episode>> {
        Ok(vec![])
    }

    /// Get active goals.
    async fn get_active_goals(&self) -> anyhow::Result<Vec<Goal>> {
        Ok(vec![])
    }

    /// Update a goal's status and/or add a progress note.
    async fn update_goal(
        &self,
        _goal_id: i64,
        _status: Option<&str>,
        _progress_note: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get behavior patterns above a confidence threshold.
    async fn get_behavior_patterns(
        &self,
        _min_confidence: f32,
    ) -> anyhow::Result<Vec<BehaviorPattern>> {
        Ok(vec![])
    }

    /// Get procedures relevant to a query.
    async fn get_relevant_procedures(
        &self,
        _query: &str,
        _limit: usize,
    ) -> anyhow::Result<Vec<Procedure>> {
        Ok(vec![])
    }

    /// Get error solutions relevant to an error message.
    async fn get_relevant_error_solutions(
        &self,
        _error: &str,
        _limit: usize,
    ) -> anyhow::Result<Vec<ErrorSolution>> {
        Ok(vec![])
    }

    /// Get all expertise records.
    async fn get_all_expertise(&self) -> anyhow::Result<Vec<Expertise>> {
        Ok(vec![])
    }

    /// Get the user profile.
    async fn get_user_profile(&self) -> anyhow::Result<Option<UserProfile>> {
        Ok(None)
    }

    /// Get trusted command patterns for AI context.
    /// Returns patterns with 3+ approvals, ordered by approval count.
    async fn get_trusted_command_patterns(&self) -> anyhow::Result<Vec<(String, i32)>> {
        Ok(vec![])
    }

    // ==================== Write Methods for Learning System ====================
    // These have default no-op implementations for backwards compatibility.

    /// Increment expertise counters and update level for a domain.
    async fn increment_expertise(
        &self,
        _domain: &str,
        _success: bool,
        _error: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Insert or update a procedure.
    async fn upsert_procedure(&self, _procedure: &Procedure) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Update procedure outcome after execution.
    #[allow(dead_code)] // Reserved for procedure feedback loop
    async fn update_procedure_outcome(
        &self,
        _procedure_id: i64,
        _success: bool,
        _duration: Option<f32>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Insert a new error-solution pair.
    async fn insert_error_solution(&self, _solution: &ErrorSolution) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Update error solution outcome.
    #[allow(dead_code)] // Reserved for error solution feedback loop
    async fn update_error_solution_outcome(
        &self,
        _solution_id: i64,
        _success: bool,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    // ==================== Dynamic Bots Methods ====================
    // For runtime bot management via /connect command.

    /// Store a dynamically added bot configuration.
    async fn add_dynamic_bot(&self, _bot: &DynamicBot) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamically added bots.
    async fn get_dynamic_bots(&self) -> anyhow::Result<Vec<DynamicBot>> {
        Ok(vec![])
    }

    /// Delete a dynamic bot by ID.
    #[allow(dead_code)]
    async fn delete_dynamic_bot(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    // ==================== Dynamic Skills Methods ====================
    // For runtime skill management via manage_skills tool.

    /// Store a dynamically added skill.
    async fn add_dynamic_skill(&self, _skill: &DynamicSkill) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic skills.
    async fn get_dynamic_skills(&self) -> anyhow::Result<Vec<DynamicSkill>> {
        Ok(vec![])
    }

    /// Delete a dynamic skill by ID.
    async fn delete_dynamic_skill(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update the enabled flag of a dynamic skill.
    async fn update_dynamic_skill_enabled(&self, _id: i64, _enabled: bool) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get procedures eligible for skill promotion (success_count >= min_success, success rate >= min_rate).
    async fn get_promotable_procedures(
        &self,
        _min_success: i32,
        _min_rate: f32,
    ) -> anyhow::Result<Vec<Procedure>> {
        Ok(vec![])
    }

    // ==================== Dynamic MCP Servers Methods ====================
    // For runtime MCP server management via manage_mcp tool.

    /// Store a dynamically added MCP server.
    async fn save_dynamic_mcp_server(&self, _server: &DynamicMcpServer) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic MCP servers.
    async fn list_dynamic_mcp_servers(&self) -> anyhow::Result<Vec<DynamicMcpServer>> {
        Ok(vec![])
    }

    /// Delete a dynamic MCP server by ID.
    async fn delete_dynamic_mcp_server(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update a dynamic MCP server.
    async fn update_dynamic_mcp_server(&self, _server: &DynamicMcpServer) -> anyhow::Result<()> {
        Ok(())
    }

    // ==================== Settings Methods ====================
    // Generic key-value settings for runtime toggles.

    /// Get a setting value by key. Returns None if unset.
    async fn get_setting(&self, _key: &str) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    /// Set a setting value. Creates or updates the key.
    async fn set_setting(&self, _key: &str, _value: &str) -> anyhow::Result<()> {
        Ok(())
    }

    // ==================== People Methods ====================
    // For tracking the owner's social circle.

    /// Create or update a person record. Returns the person ID.
    async fn upsert_person(&self, _person: &Person) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get a person by their database ID.
    async fn get_person(&self, _id: i64) -> anyhow::Result<Option<Person>> {
        Ok(None)
    }

    /// Look up a person by a platform-qualified sender ID (e.g., "slack:U123").
    async fn get_person_by_platform_id(
        &self,
        _platform_id: &str,
    ) -> anyhow::Result<Option<Person>> {
        Ok(None)
    }

    /// Find a person by name or alias (case-insensitive).
    async fn find_person_by_name(&self, _name: &str) -> anyhow::Result<Option<Person>> {
        Ok(None)
    }

    /// Get all people.
    async fn get_all_people(&self) -> anyhow::Result<Vec<Person>> {
        Ok(vec![])
    }

    /// Delete a person and all their facts (cascade).
    async fn delete_person(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Link a platform identity to a person.
    async fn link_platform_id(
        &self,
        _person_id: i64,
        _platform_id: &str,
        _display_name: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update interaction tracking for a person.
    async fn touch_person_interaction(&self, _person_id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Create or update a fact about a person.
    async fn upsert_person_fact(
        &self,
        _person_id: i64,
        _category: &str,
        _key: &str,
        _value: &str,
        _source: &str,
        _confidence: f32,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get facts about a person, optionally filtered by category.
    async fn get_person_facts(
        &self,
        _person_id: i64,
        _category: Option<&str>,
    ) -> anyhow::Result<Vec<PersonFact>> {
        Ok(vec![])
    }

    /// Delete a person fact by ID.
    async fn delete_person_fact(&self, _fact_id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Confirm an auto-extracted person fact (set confidence to 1.0).
    async fn confirm_person_fact(&self, _fact_id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get people with upcoming dates (birthdays, important dates) within N days.
    async fn get_people_with_upcoming_dates(
        &self,
        _within_days: i32,
    ) -> anyhow::Result<Vec<(Person, PersonFact)>> {
        Ok(vec![])
    }

    /// Delete stale auto-extracted person facts older than N days with confidence < 1.0.
    async fn prune_stale_person_facts(&self, _retention_days: u32) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Get people who haven't interacted in more than N days.
    async fn get_people_needing_reconnect(
        &self,
        _inactive_days: u32,
    ) -> anyhow::Result<Vec<Person>> {
        Ok(vec![])
    }

    // ==================== OAuth Connection Methods ====================
    // For tracking OAuth-connected external services.

    /// Save an OAuth connection. Returns the connection ID.
    async fn save_oauth_connection(&self, _conn: &OAuthConnection) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get an OAuth connection by service name.
    async fn get_oauth_connection(
        &self,
        _service: &str,
    ) -> anyhow::Result<Option<OAuthConnection>> {
        Ok(None)
    }

    /// List all OAuth connections.
    async fn list_oauth_connections(&self) -> anyhow::Result<Vec<OAuthConnection>> {
        Ok(vec![])
    }

    /// Delete an OAuth connection by service name.
    async fn delete_oauth_connection(&self, _service: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update token expiry for an OAuth connection.
    async fn update_oauth_token_expiry(
        &self,
        _service: &str,
        _expires_at: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
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
