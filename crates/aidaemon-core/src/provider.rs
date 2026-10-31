use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{Message, ToolCall};

/// What the model sees about a tool: the OpenAI-style function object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
}

/// How the model may use tools on this call.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Specific(String),
}

/// One provider call. Hooks may edit it in [`crate::Hook::before_model`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRequest {
    pub model: String,
    /// Instructions first (stable, cacheable), then the transcript.
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: ToolChoice,
    pub max_output_tokens: Option<u32>,
}

/// Token accounting for one call or a whole run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Input tokens served from the provider's prompt cache.
    pub cached_input_tokens: u64,
    pub cache_creation_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_tokens += other.cache_creation_tokens;
    }

    /// Tokens that count toward a budget: cached input is excluded.
    pub fn billable_tokens(&self) -> u64 {
        self.input_tokens.saturating_sub(self.cached_input_tokens) + self.output_tokens
    }
}

/// A complete (non-streamed) model reply.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
}

/// One item of a streamed reply. A well-formed stream yields any number of
/// deltas and ends with exactly one `Done` carrying the complete response.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Visible text as it is generated.
    TextDelta(String),
    /// The full reply, with tool calls and usage.
    Done(ModelResponse),
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ProviderError {
    #[error("rate limited")]
    RateLimited { retry_after: Option<Duration> },
    #[error("transient provider error: {0}")]
    Transient(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("authentication failed: {0}")]
    Auth(String),
    #[error("provider error: {0}")]
    Other(String),
}

impl ProviderError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RateLimited { .. } | Self::Transient(_))
    }
}

/// A model backend. Implementations translate [`ModelRequest`] to their wire
/// format and back. Retries and fallbacks are wrappers ([`Retry`]), not loop
/// logic.
///
/// The loop always calls [`Provider::stream`]. Providers without native
/// streaming implement only `complete`; the default `stream` wraps it in a
/// single `Done` event. Streaming providers implement both, usually with
/// `complete` as a fold over `stream` (see [`collect_stream`]).
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError>;

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        Box::pin(stream::once(async move {
            self.complete(request).await.map(StreamEvent::Done)
        }))
    }
}

/// Drain a stream into its final response. Useful for implementing
/// `complete` on a streaming provider.
pub async fn collect_stream(
    mut events: BoxStream<'_, Result<StreamEvent, ProviderError>>,
) -> Result<ModelResponse, ProviderError> {
    while let Some(event) = events.next().await {
        if let StreamEvent::Done(response) = event? {
            return Ok(response);
        }
    }
    Err(ProviderError::Transient(
        "stream ended without a final response".into(),
    ))
}

#[async_trait]
impl<P: Provider + ?Sized> Provider for std::sync::Arc<P> {
    fn name(&self) -> &str {
        (**self).name()
    }
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        (**self).complete(request).await
    }
    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        (**self).stream(request)
    }
}

/// Retry layer: exponential backoff on retryable errors.
pub struct Retry<P> {
    inner: P,
    max_attempts: u32,
    base_delay: Duration,
}

impl<P: Provider> Retry<P> {
    fn delay_for(&self, err: &ProviderError, attempt: u32) -> Duration {
        match err {
            ProviderError::RateLimited {
                retry_after: Some(after),
            } => *after,
            _ => self.base_delay * 2u32.saturating_pow(attempt.saturating_sub(1)),
        }
    }

    pub fn new(inner: P) -> Self {
        Self {
            inner,
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
        }
    }

    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts.max(1);
        self
    }

    pub fn base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }
}

#[async_trait]
impl<P: Provider> Provider for Retry<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.inner.complete(request).await {
                Ok(response) => return Ok(response),
                Err(err) if err.is_retryable() && attempt < self.max_attempts => {
                    tokio::time::sleep(self.delay_for(&err, attempt)).await;
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Retries only while nothing has been streamed yet: once a delta has
    /// reached the caller, a retry would duplicate visible output.
    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        let opened = async move {
            let mut attempt = 0;
            loop {
                attempt += 1;
                let mut events = self.inner.stream(request);
                match events.next().await {
                    Some(Err(err)) if err.is_retryable() && attempt < self.max_attempts => {
                        tokio::time::sleep(self.delay_for(&err, attempt)).await;
                    }
                    first => return (first, events),
                }
            }
        };
        Box::pin(stream::once(opened).flat_map(|(first, rest)| stream::iter(first).chain(rest)))
    }
}
