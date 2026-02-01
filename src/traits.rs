use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub session_id: String,
    pub role: String,           // "system", "user", "assistant", "tool"
    pub content: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub tool_calls_json: Option<String>, // serialized Vec<ToolCall>
    pub created_at: DateTime<Utc>,
}

/// A single tool call as returned by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // JSON string
    /// Gemini 3+ thought signature — must be preserved and sent back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
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
}

/// An event emitted by triggers or channels.
#[derive(Debug, Clone)]
pub struct Event {
    pub source: String,
    pub session_id: String,
    pub content: String,
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

/// The LLM's response: either content text, tool calls, or both.
#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// Persistent state store (SQLite + working memory).
#[async_trait]
pub trait StateStore: Send + Sync {
    /// Append a message to the session history (both DB and working memory).
    async fn append_message(&self, msg: &Message) -> anyhow::Result<()>;
    /// Get recent messages for a session from working memory.
    async fn get_history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>>;
    /// Upsert a fact.
    async fn upsert_fact(&self, category: &str, key: &str, value: &str, source: &str) -> anyhow::Result<()>;
    /// Get all facts, optionally filtered by category.
    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>>;
}
