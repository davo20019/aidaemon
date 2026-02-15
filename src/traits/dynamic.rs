use serde::{Deserialize, Serialize};

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
