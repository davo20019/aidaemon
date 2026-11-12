use serde::{Deserialize, Serialize};

pub use aidaemon_tools::services::{OAuthConnection, PendingOAuthFlow};

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

pub use aidaemon_mcp::registry::DynamicMcpServer;
pub use aidaemon_tools::services::{CliAgentInvocation, DynamicCliAgent};
