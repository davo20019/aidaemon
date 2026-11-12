//! # aidaemon-providers
//!
//! Real model backends for the [`aidaemon_core::Provider`] trait:
//!
//! - [`OpenAiCompatible`]: the Chat Completions API. Works with OpenAI,
//!   OpenRouter, Ollama, llama.cpp, vLLM and other compatible servers.
//! - [`Anthropic`]: the Messages API, with prompt caching of the stable
//!   prefix (system prompt and tool definitions).
//! - [`ChatGpt`]: models on a ChatGPT subscription through the Codex
//!   backend's Responses API, with credentials from a [`ChatGptAuth`].
//! - [`GoogleGenai`]: Gemini's `generateContent`, with thought signatures
//!   carried in `ToolCall::provider_data` and Google Search grounding.
//! - [`Xai`]: xAI's Responses API.
//!
//! Both stream over SSE: [`Provider::stream`](aidaemon_core::Provider::stream)
//! yields [`StreamEvent::TextDelta`](aidaemon_core::StreamEvent::TextDelta)
//! items as text arrives and then exactly one
//! [`StreamEvent::Done`](aidaemon_core::StreamEvent::Done) with the full
//! reply (text, tool calls with parsed JSON arguments, and usage).
//! `complete` is a fold over `stream`.
//!
//! HTTP failures map to [`ProviderError`](aidaemon_core::ProviderError):
//! 429 becomes `RateLimited` (with `retry-after`), 5xx, 408, timeouts,
//! connection failures and mid-stream error events become `Transient`,
//! 401/403 `Auth`, 402 `Billing` (with the affordable tokens when stated),
//! 404 `NotFound`, a context-length error `ContextOverflow` (with the
//! window when stated), and other 4xx `InvalidRequest`, each with a short
//! excerpt of the response body. Wrap a provider in
//! [`aidaemon_core::Retry`] or [`aidaemon_core::Failover`] to recover.
//!
//! ```no_run
//! use aidaemon_core::{Agent, Retry};
//! use aidaemon_providers::{Anthropic, OpenAiCompatible};
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! // Any OpenAI-compatible server; here a local Ollama.
//! let local = OpenAiCompatible::builder()
//!     .base_url("http://localhost:11434/v1")
//!     .build()?;
//!
//! let claude = Anthropic::builder()
//!     .api_key(std::env::var("ANTHROPIC_API_KEY")?)
//!     .build()?;
//!
//! let agent = Agent::builder("helper").allow_unguarded()
//!     .model("claude-sonnet-4-5")
//!     .instructions("You are a concise assistant.")
//!     .provider(Retry::new(claude))
//!     .build()?;
//! let outcome = agent.run("cli:synthetic-user-1", "Say hello").await?;
//! println!("{}", outcome.final_text.unwrap_or_default());
//! # let _ = local;
//! # Ok(())
//! # }
//! ```

mod anthropic;
mod chatgpt;
#[cfg(feature = "chatgpt-login")]
pub mod chatgpt_login;
mod errors;
mod google;
mod openai;
pub mod routing;
mod sse;
mod transport;
mod xai;

pub use anthropic::{
    Anthropic, AnthropicBuilder, ANTHROPIC_DEFAULT_BASE_URL, ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_VERSION, THINKING_BLOCKS_KEY,
};
pub use chatgpt::{
    ChatGpt, ChatGptAuth, ChatGptBuilder, ChatGptCredentials, StaticChatGptAuth,
    CHATGPT_DEFAULT_BASE_URL, CHATGPT_KNOWN_MODELS, CHATGPT_REASONING_EFFORTS,
};
pub use errors::{affordable_tokens, context_overflow, error_for_status};
pub use google::{
    GoogleGenai, GoogleGenaiBuilder, GOOGLE_DEFAULT_BASE_URL, GOOGLE_PART_FIELDS,
    TEXT_ONLY_RESPONSE_MARKER,
};
pub use openai::{OpenAiCompatible, OpenAiCompatibleBuilder, OPENAI_DEFAULT_BASE_URL};
pub use transport::ConfigError;
pub use xai::{Xai, XaiBuilder, XAI_DEFAULT_BASE_URL};
