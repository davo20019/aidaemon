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

    /// Update the allowed_user_ids for a dynamic bot identified by its token.
    #[allow(dead_code)]
    async fn update_dynamic_bot_allowed_users(
        &self,
        _bot_token: &str,
        _allowed_user_ids: &[String],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete a dynamic bot by ID.
    #[allow(dead_code)]
    async fn delete_dynamic_bot(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    // ==================== Session Channel Mapping ====================

    /// Persist a session_id → channel_name mapping so it survives restarts.
    async fn save_session_channel(
        &self,
        _session_id: &str,
        _channel_name: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Load all persisted session → channel mappings (for populating session_map on startup).
    async fn load_session_channels(&self) -> anyhow::Result<Vec<(String, String)>> {
        Ok(vec![])
    }

    // ==================== Dynamic Skills Methods ====================
    // Deprecated: kept for migration compatibility. Use filesystem skills instead.

    /// Store a dynamically added skill.
    /// Deprecated: use `write_skill_to_file()` to persist skills to filesystem.
    #[allow(dead_code)]
    async fn add_dynamic_skill(&self, _skill: &DynamicSkill) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic skills.
    /// Deprecated: use `load_skills()` to read skills from filesystem.
    async fn get_dynamic_skills(&self) -> anyhow::Result<Vec<DynamicSkill>> {
        Ok(vec![])
    }

    /// Delete a dynamic skill by ID.
    /// Deprecated: use `remove_skill_file()` to remove skills from filesystem.
    #[allow(dead_code)]
    async fn delete_dynamic_skill(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update the enabled flag of a dynamic skill.
    /// Deprecated: file existence = active, no enable/disable needed.
    #[allow(dead_code)]
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

    // ==================== Skill Drafts Methods ====================
    // For auto-promoted skill drafts pending user review.

    /// Store a skill draft from auto-promotion. Returns the draft ID.
    async fn add_skill_draft(&self, _draft: &SkillDraft) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all pending skill drafts.
    async fn get_pending_skill_drafts(&self) -> anyhow::Result<Vec<SkillDraft>> {
        Ok(vec![])
    }

    /// Get a skill draft by ID.
    async fn get_skill_draft(&self, _id: i64) -> anyhow::Result<Option<SkillDraft>> {
        Ok(None)
    }

    /// Update a skill draft's status ("approved" or "dismissed").
    async fn update_skill_draft_status(&self, _id: i64, _status: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Check if a draft already exists for a given procedure name.
    async fn skill_draft_exists_for_procedure(
        &self,
        _procedure_name: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
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

    // ==================== Dynamic CLI Agents Methods ====================
    // For runtime CLI agent management via manage_cli_agents tool.

    /// Store a dynamically added CLI agent.
    async fn save_dynamic_cli_agent(&self, _agent: &DynamicCliAgent) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic CLI agents.
    async fn list_dynamic_cli_agents(&self) -> anyhow::Result<Vec<DynamicCliAgent>> {
        Ok(vec![])
    }

    /// Delete a dynamic CLI agent by ID.
    async fn delete_dynamic_cli_agent(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update a dynamic CLI agent.
    async fn update_dynamic_cli_agent(&self, _agent: &DynamicCliAgent) -> anyhow::Result<()> {
        Ok(())
    }

    /// Log the start of a CLI agent invocation. Returns the invocation ID.
    async fn log_cli_agent_start(
        &self,
        _session_id: &str,
        _agent_name: &str,
        _prompt_summary: &str,
        _working_dir: Option<&str>,
    ) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Log the completion of a CLI agent invocation.
    async fn log_cli_agent_complete(
        &self,
        _id: i64,
        _exit_code: Option<i32>,
        _output_summary: &str,
        _success: bool,
        _duration_secs: f64,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get recent CLI agent invocations (most recent first).
    async fn get_cli_agent_invocations(
        &self,
        _limit: usize,
    ) -> anyhow::Result<Vec<CliAgentInvocation>> {
        Ok(vec![])
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

    // ==================== V3 Orchestration Methods ====================
    // For the V3 goal/task tracking system.

    /// Create a new V3 goal.
    async fn create_goal_v3(&self, _goal: &GoalV3) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get a V3 goal by ID.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_goal_v3(&self, _id: &str) -> anyhow::Result<Option<GoalV3>> {
        Ok(None)
    }

    /// Update a V3 goal (full replacement).
    #[allow(dead_code)] // Used in Phase 2
    async fn update_goal_v3(&self, _goal: &GoalV3) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get all active V3 goals (status = "active" or "pending").
    #[allow(dead_code)] // Used in Phase 2
    async fn get_active_goals_v3(&self) -> anyhow::Result<Vec<GoalV3>> {
        Ok(vec![])
    }

    /// Get V3 goals for a specific session.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_goals_for_session_v3(&self, _session_id: &str) -> anyhow::Result<Vec<GoalV3>> {
        Ok(vec![])
    }

    /// Migrate legacy scheduler rows (`scheduled_tasks`) into V3 goals.
    /// Returns the number of migrated rows.
    async fn migrate_legacy_scheduled_tasks_to_v3(&self) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Get scheduled goals awaiting confirmation in a session.
    async fn get_pending_confirmation_goals(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Vec<GoalV3>> {
        Ok(vec![])
    }

    /// Activate a pending-confirmation goal.
    /// Returns true when the status transition was applied.
    async fn activate_goal_v3(&self, _goal_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Create a new V3 task within a goal.
    #[allow(dead_code)] // Used in Phase 2
    async fn create_task_v3(&self, _task: &TaskV3) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get a V3 task by ID.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_task_v3(&self, _id: &str) -> anyhow::Result<Option<TaskV3>> {
        Ok(None)
    }

    /// Update a V3 task (full replacement).
    #[allow(dead_code)] // Used in Phase 2
    async fn update_task_v3(&self, _task: &TaskV3) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get all V3 tasks for a goal.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_tasks_for_goal_v3(&self, _goal_id: &str) -> anyhow::Result<Vec<TaskV3>> {
        Ok(vec![])
    }

    /// Count completed/skipped tasks for a goal (used by progress-based circuit breaker).
    async fn count_completed_tasks_for_goal(&self, _goal_id: &str) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Atomically claim a pending task for an executor.
    #[allow(dead_code)] // Used in Phase 2
    async fn claim_task_v3(&self, _task_id: &str, _agent_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Log an activity entry for a V3 task.
    #[allow(dead_code)] // Used in Phase 2
    async fn log_task_activity_v3(&self, _activity: &TaskActivityV3) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get activity log for a V3 task.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_task_activities_v3(&self, _task_id: &str) -> anyhow::Result<Vec<TaskActivityV3>> {
        Ok(vec![])
    }

    /// Get continuous goals whose schedule is due (last_useful_action + interval < now).
    async fn get_due_evergreen_goals(&self) -> anyhow::Result<Vec<GoalV3>> {
        Ok(vec![])
    }

    /// Get deferred finite goals whose scheduled run is due.
    async fn get_due_scheduled_finite_goals(&self) -> anyhow::Result<Vec<GoalV3>> {
        Ok(vec![])
    }

    /// Cancel pending-confirmation goals older than max_age_secs.
    async fn cancel_stale_pending_confirmation_goals(
        &self,
        _max_age_secs: i64,
    ) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Get all V3 goals that are scheduled or awaiting confirmation.
    async fn get_scheduled_goals_v3(&self) -> anyhow::Result<Vec<GoalV3>> {
        Ok(vec![])
    }

    /// Reset tokens_used_today to 0 for all active continuous goals.
    async fn reset_daily_token_budgets(&self) -> anyhow::Result<u64> {
        Ok(0)
    }

    // ==================== Conversation Summary Methods ====================
    // For context window management — sliding window summarization.

    /// Get the conversation summary for a session.
    async fn get_conversation_summary(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Option<ConversationSummary>> {
        Ok(None)
    }

    /// Create or update a conversation summary for a session.
    async fn upsert_conversation_summary(
        &self,
        _summary: &ConversationSummary,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Database health check — verifies the connection is alive.
    async fn health_check(&self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get pending tasks ordered by priority, filtering out those with unmet dependencies.
    async fn get_pending_tasks_by_priority(&self, _limit: i64) -> anyhow::Result<Vec<TaskV3>> {
        Ok(vec![])
    }

    /// Get tasks stuck in running/claimed state longer than timeout_secs.
    async fn get_stuck_tasks(&self, _timeout_secs: i64) -> anyhow::Result<Vec<TaskV3>> {
        Ok(vec![])
    }

    /// Get tasks completed after a given timestamp.
    #[allow(dead_code)]
    async fn get_recently_completed_tasks(&self, _since: &str) -> anyhow::Result<Vec<TaskV3>> {
        Ok(vec![])
    }

    /// Mark a running/claimed task as interrupted (e.g., after crash or timeout).
    async fn mark_task_interrupted(&self, _task_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Count active evergreen (continuous) goals.
    async fn count_active_evergreen_goals(&self) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get goals that completed/failed but haven't been notified to the user yet.
    async fn get_goals_needing_notification(&self) -> anyhow::Result<Vec<GoalV3>> {
        Ok(vec![])
    }

    /// Mark a goal as notified (set notified_at timestamp).
    async fn mark_goal_notified(&self, _goal_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Mark stale active goals as abandoned/failed.
    ///
    /// - Legacy `goals` table: active goals with no update in `stale_hours` → abandoned
    /// - V3 finite goals: active goals with no update in `stale_hours` → failed
    /// - V3 evergreen goals: skipped (they have their own idle detection)
    ///
    /// Returns (legacy_count, v3_count) of goals cleaned up.
    async fn cleanup_stale_goals(&self, _stale_hours: i64) -> anyhow::Result<(u64, u64)> {
        Ok((0, 0))
    }

    // --- Notification Queue ---

    /// Enqueue a notification for delivery.
    async fn enqueue_notification(&self, _entry: &NotificationEntry) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get pending notifications ordered by priority (critical first), then creation time.
    async fn get_pending_notifications(
        &self,
        _limit: i64,
    ) -> anyhow::Result<Vec<NotificationEntry>> {
        Ok(vec![])
    }

    /// Mark a notification as delivered.
    async fn mark_notification_delivered(&self, _notification_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Increment the attempt counter for a notification.
    async fn increment_notification_attempt(&self, _notification_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete expired status_update notifications (past their expires_at).
    async fn cleanup_expired_notifications(&self) -> anyhow::Result<i64> {
        Ok(0)
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
    /// "completed", "failed", "escalation", "progress", "stalled", "evergreen_alert"
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
            "completed" | "failed" | "escalation" | "evergreen_alert" => "critical",
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
