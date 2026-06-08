use async_trait::async_trait;
use serde_json::Value;

use super::ToolCall;

/// Structured response mode requested for a provider call.
#[derive(Debug, Clone, PartialEq, Default)]
#[allow(dead_code)]
pub enum ResponseMode {
    /// Default free-form text / tool-call behavior.
    #[default]
    Text,
    /// Provider should return a JSON object response.
    JsonObject,
    /// Provider should enforce a specific JSON schema.
    JsonSchema {
        /// Schema name identifier for provider telemetry/debugging.
        name: String,
        /// Draft-compatible JSON schema object.
        schema: Value,
        /// Strict schema enforcement if supported by provider.
        strict: bool,
    },
}

/// Tool-call behavior requested for a provider call.
#[derive(Debug, Clone, PartialEq, Default)]
#[allow(dead_code)]
pub enum ToolChoiceMode {
    /// Provider decides whether to call tools.
    #[default]
    Auto,
    /// Disable all tool calls.
    None,
    /// Require at least one tool call.
    Required,
    /// Require a specific tool name.
    Specific(String),
}

/// Per-call provider control surface used by the agent loop.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChatOptions {
    pub response_mode: ResponseMode,
    pub tool_choice: ToolChoiceMode,
    /// Override the provider's configured max_tokens for this call only.
    /// Used for billing-aware retry: when a 402 tells us how many tokens
    /// we can afford, we retry with a reduced cap instead of cascading.
    pub max_tokens_override: Option<u32>,
    /// Override the provider's configured reasoning_effort for this call only.
    /// Used for truncation recovery: when a thinking model spends all tokens on
    /// reasoning with no usable output, the retry reduces effort to "low".
    pub reasoning_effort_override: Option<String>,
    /// llama.cpp KV-cache slot pin (`id_slot`) for this call only.
    ///
    /// `None` (the default) means "use the provider's background slot" when slot
    /// routing is enabled, and "omit the field entirely" when it is disabled.
    /// The interactive generation loop sets `Some(interactive_slot)` to keep the
    /// conversation's KV cache warm against background-task eviction.
    pub id_slot: Option<u32>,
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

    /// Extended chat API with per-call behavior controls.
    ///
    /// Default implementation preserves backwards compatibility by delegating
    /// to `chat()` and ignoring options for providers that don't implement it.
    async fn chat_with_options(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        _options: &ChatOptions,
    ) -> anyhow::Result<ProviderResponse> {
        self.chat(model, messages, tools).await
    }

    /// List available models from the provider. Returns model ID strings.
    async fn list_models(&self) -> anyhow::Result<Vec<String>>;
}

/// Token usage statistics from an LLM API response.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Input tokens served from a provider-side cache, when reported.
    pub cached_input_tokens: Option<u32>,
    /// Input tokens newly written into a provider-side cache, when reported.
    pub cache_creation_input_tokens: Option<u32>,
    pub model: String,
}

impl TokenUsage {
    pub fn fresh_input_tokens(&self) -> Option<u32> {
        self.cached_input_tokens
            .map(|cached| self.input_tokens.saturating_sub(cached))
    }
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
    #[allow(dead_code)] // Used for diagnostics and future cost reports
    pub cached_input_tokens: Option<i64>,
    #[allow(dead_code)] // Used for diagnostics and future cost reports
    pub cache_creation_input_tokens: Option<i64>,
    #[allow(dead_code)] // Used by reconciliation/reporting paths that need correlation IDs.
    pub call_id: Option<String>,
    #[allow(dead_code)] // Used for database queries
    pub created_at: String,
}
