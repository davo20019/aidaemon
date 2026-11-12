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

/// The shape a reply must take.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    /// Any JSON object.
    JsonObject,
    /// JSON matching `schema`.
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

/// How a provider should answer, beyond the messages and tools. Providers
/// map what their API supports and ignore the rest.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RequestOptions {
    pub response_format: ResponseFormat,
    /// Provider-neutral reasoning effort: `"low"`, `"medium"` or `"high"`;
    /// `"off"` turns reasoning off. `None` keeps the provider's default.
    pub reasoning_effort: Option<String>,
    /// Exactly one attempt: providers skip their own internal retries, and
    /// [`Retry`] and [`crate::Failover`] give up at the first error. For a
    /// call that must not run twice.
    pub single_attempt: bool,
    /// With `single_attempt`, lets a [`crate::Failover`] chain move on to
    /// its next step after an error that proves the attempt used nothing
    /// ([`ProviderError::proves_nothing_spent`]) or after the policy
    /// refused the attempt before sending it. Any other error still ends
    /// the chain, and no step is retried. For a call charged against a
    /// fixed budget, where unknown spend must not be repeated.
    pub fallback_when_unspent: bool,
    /// Identifies the conversation for providers that route prompt-cache
    /// lookups by key, so its calls reach the same cache. Only a routing
    /// hint: providers that do not route by key ignore it.
    pub cache_key: Option<String>,
}

/// One provider call. Hooks may edit it in [`crate::Hook::before_model`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelRequest {
    pub model: String,
    /// Instructions first (stable, cacheable), then the transcript.
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: ToolChoice,
    pub max_output_tokens: Option<u32>,
    pub options: RequestOptions,
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
    /// The model's visible reasoning, when the provider returns it apart
    /// from the answer.
    pub reasoning: Option<String>,
    /// Why the reply may be incomplete (cut off at the token limit, held
    /// back by a safety filter), in the provider's words.
    pub note: Option<String>,
}

/// One item of a streamed reply. A well-formed stream yields any number of
/// deltas and ends with exactly one `Done` carrying the complete response.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Visible text as it is generated.
    TextDelta(String),
    /// The full reply, with tool calls and usage.
    Done(ModelResponse),
    /// A [`crate::Failover`] chain gave up on one provider and is trying
    /// the next. Comes before any text of the reply.
    Switched(crate::ProviderSwitch),
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
    /// The account cannot pay for the request. `affordable_tokens` is the
    /// output the provider says it could still pay for, when it says.
    #[error("billing: {message}")]
    Billing {
        message: String,
        affordable_tokens: Option<u64>,
    },
    /// A usage quota (a subscription plan's cap, a daily allowance) is
    /// used up until `resets_at`, when the provider says. Unlike
    /// [`ProviderError::RateLimited`], waiting seconds does not help.
    /// `per_model` is true when the quota covers only the requested model;
    /// otherwise it covers every model on the account. `in_stream` is true
    /// when the provider reported it inside a response it had already
    /// started, so the request was admitted and may have used tokens;
    /// false when the request was refused before any answer.
    #[error("quota exhausted: {message}")]
    QuotaExhausted {
        message: String,
        resets_at: Option<std::time::SystemTime>,
        per_model: bool,
        in_stream: bool,
    },
    /// The model or endpoint does not exist here.
    #[error("not found: {0}")]
    NotFound(String),
    /// The request does not fit the model's context window. `limit` is
    /// the window in tokens, when the provider states it.
    #[error("context window exceeded: {message}")]
    ContextOverflow { message: String, limit: Option<u64> },
    /// The reply could not be read. `parse` is true when it was not valid
    /// at all (often a cut-off stream, worth another attempt), false when it
    /// parsed but had the wrong shape.
    #[error("malformed reply: {message}")]
    Malformed { message: String, parse: bool },
    #[error("provider error: {0}")]
    Other(String),
}

impl ProviderError {
    /// The provider refused the request before doing any work, so the call
    /// used no tokens: bad credentials, an unknown model, no credit, or a
    /// used-up quota reported as a refusal. Anything else (a timeout, a
    /// server error, a cut-off or malformed reply, a quota reported inside
    /// an answer already started) may have been paid for.
    pub fn proves_nothing_spent(&self) -> bool {
        match self {
            Self::Auth(_) | Self::NotFound(_) | Self::Billing { .. } => true,
            Self::QuotaExhausted { in_stream, .. } => !in_stream,
            _ => false,
        }
    }

    /// Worth another attempt at the same provider.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. } | Self::Transient(_) | Self::Malformed { parse: true, .. }
        )
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

    /// When `model` can next be asked, if this provider already knows it
    /// cannot answer now (a used-up quota, a rejected credential, no
    /// credit; see [`crate::QuotaGuard`]). `None`: it may answer now.
    fn unavailable_until(&self, _model: &str) -> Option<std::time::SystemTime> {
        None
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
    // Forwarded like the calls: without it every `Arc<dyn Provider>` (the
    // daemon's model backends) reported a used-up model as available.
    fn unavailable_until(&self, model: &str) -> Option<std::time::SystemTime> {
        (**self).unavailable_until(model)
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
    fn unavailable_until(&self, model: &str) -> Option<std::time::SystemTime> {
        self.inner.unavailable_until(model)
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.inner.complete(request).await {
                Ok(response) => return Ok(response),
                Err(err)
                    if err.is_retryable()
                        && attempt < self.max_attempts
                        && !request.options.single_attempt =>
                {
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
                    Some(Err(err))
                        if err.is_retryable()
                            && attempt < self.max_attempts
                            && !request.options.single_attempt =>
                    {
                        tokio::time::sleep(self.delay_for(&err, attempt)).await;
                    }
                    first => return (first, events),
                }
            }
        };
        Box::pin(stream::once(opened).flat_map(|(first, rest)| stream::iter(first).chain(rest)))
    }
}
