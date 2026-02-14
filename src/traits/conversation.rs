use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

/// A conversation summary for a session, used by context window management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub session_id: String,
    pub summary: String,
    pub message_count: usize,
    pub last_message_id: String,
    pub updated_at: DateTime<Utc>,
}

