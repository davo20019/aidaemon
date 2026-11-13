//! Shared HTTP plumbing: error mapping and the SSE streaming driver used by
//! every provider in this crate.

use std::collections::VecDeque;
use std::time::Duration;

use aidaemon_core::{ModelResponse, ProviderError, StreamEvent};
use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::sse::{SseEvent, SseParser};

/// Default whole-request timeout (connect, headers and full streamed body).
pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
/// Default TCP/TLS connect timeout.
pub(crate) const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

const BODY_EXCERPT_CHARS: usize = 300;

/// Misconfiguration caught when building a provider.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    #[error("missing API key")]
    MissingApiKey,
    #[error("missing {0}")]
    Missing(&'static str),
    #[error("invalid header `{0}`")]
    InvalidHeader(String),
    #[error("could not build HTTP client: {0}")]
    Client(String),
}

pub(crate) fn build_client(connect_timeout: Duration) -> Result<reqwest::Client, ConfigError> {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .build()
        .map_err(|err| ConfigError::Client(err.to_string()))
}

/// An API key whose `Debug` output is redacted.
#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct Secret(pub String);

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(***)")
    }
}

/// Validate user-supplied extra headers.
pub(crate) fn header_map(headers: &[(String, String)]) -> Result<HeaderMap, ConfigError> {
    let mut map = HeaderMap::new();
    for (name, value) in headers {
        let header_name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| ConfigError::InvalidHeader(name.clone()))?;
        let mut header_value = reqwest::header::HeaderValue::from_str(value)
            .map_err(|_| ConfigError::InvalidHeader(name.clone()))?;
        // Keep custom auth-style headers out of Debug output.
        header_value.set_sensitive(true);
        map.insert(header_name, header_value);
    }
    Ok(map)
}

/// Map a transport-level failure (connect, timeout, broken body).
pub(crate) fn map_reqwest_error(err: reqwest::Error) -> ProviderError {
    if err.is_builder() {
        return ProviderError::InvalidRequest(format!("could not build request: {err}"));
    }
    if err.is_timeout() {
        return ProviderError::Transient(format!("request timed out: {err}"));
    }
    if err.is_connect() {
        return ProviderError::Transient(format!("connection failed: {err}"));
    }
    // Body/decode/request errors mid-flight are network trouble, so retryable.
    ProviderError::Transient(format!("http error: {err}"))
}

/// Map a non-success HTTP status to a [`ProviderError`].
pub(crate) fn map_status(status: StatusCode, headers: &HeaderMap, body: &str) -> ProviderError {
    let excerpt = body_excerpt(body);
    let detail = format!("HTTP {}: {}", status.as_u16(), excerpt);
    // Some kinds (a rate limit) keep no message, so the body is only here.
    tracing::warn!(
        status = status.as_u16(),
        retry_after = ?retry_after(headers),
        body = %excerpt,
        "provider HTTP error"
    );
    match status.as_u16() {
        429 if !crate::errors::out_of_credit(body) => ProviderError::RateLimited {
            retry_after: retry_after(headers),
        },
        status => crate::errors::error_for_status(status, body, detail),
    }
}

/// Parse `retry-after-ms` or `retry-after` (seconds, possibly fractional).
/// HTTP-date values are ignored.
pub(crate) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let get = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
    };
    if let Some(ms) = get("retry-after-ms") {
        return Some(Duration::from_secs_f64(ms / 1000.0));
    }
    get("retry-after").map(Duration::from_secs_f64)
}

/// A short, human-readable excerpt of an error body. Prefers the standard
/// `{"error": {"message": ...}}` shape used by both OpenAI and Anthropic.
pub(crate) fn body_excerpt(body: &str) -> String {
    let message = serde_json::from_str::<Value>(body).ok().and_then(|v| {
        v.pointer("/error/message")
            .or_else(|| v.get("message"))
            .or_else(|| v.get("error").filter(|e| e.is_string()))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let text = message.unwrap_or_else(|| body.trim().to_string());
    truncate(&text, BODY_EXCERPT_CHARS)
}

fn truncate(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}…", &text[..idx]),
        None => text.to_string(),
    }
}

/// Parse a model-produced argument string. Empty means "no arguments";
/// invalid JSON is preserved as `{"_raw": "<string>"}` so nothing is lost
/// and the tool's validation can report it to the model.
pub(crate) fn parse_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Object(Default::default());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::json!({ "_raw": raw }))
}

/// Inverse of [`parse_arguments`] for replaying a transcript: a `{"_raw": s}`
/// placeholder goes back on the wire as the original string.
pub(crate) fn raw_arguments(arguments: &Value) -> Option<&str> {
    let obj = arguments.as_object()?;
    if obj.len() == 1 {
        obj.get("_raw").and_then(Value::as_str)
    } else {
        None
    }
}

/// Wire-format decoder plugged into [`drive`].
pub(crate) trait SseDecoder: Send + 'static {
    /// Handle one SSE event, pushing visible text deltas into `deltas`.
    /// Returns `Ok(true)` once the provider signalled the end of the reply.
    fn on_event(
        &mut self,
        event: &SseEvent,
        deltas: &mut Vec<String>,
    ) -> Result<bool, ProviderError>;

    /// Assemble the final response. `terminated` is false when the body
    /// ended before the provider's terminal event.
    fn finish(&mut self, terminated: bool) -> Result<ModelResponse, ProviderError>;
}

enum Phase {
    Start(Box<reqwest::RequestBuilder>),
    Body(BoxStream<'static, reqwest::Result<Bytes>>),
    Done,
}

struct Driver<D> {
    phase: Phase,
    parser: SseParser,
    decoder: D,
    queue: VecDeque<Result<StreamEvent, ProviderError>>,
}

impl<D: SseDecoder> Driver<D> {
    /// Run decoded events; returns true when the stream is over (terminal
    /// event or error), with the final item already queued.
    fn feed(&mut self, events: Vec<SseEvent>) -> bool {
        let mut deltas = Vec::new();
        for event in &events {
            match self.decoder.on_event(event, &mut deltas) {
                Ok(false) => {}
                Ok(true) => {
                    self.queue_deltas(deltas);
                    let done = self.decoder.finish(true).map(StreamEvent::Done);
                    self.queue.push_back(done);
                    return true;
                }
                Err(err) => {
                    self.queue_deltas(deltas);
                    self.queue.push_back(Err(err));
                    return true;
                }
            }
        }
        self.queue_deltas(deltas);
        false
    }

    fn queue_deltas(&mut self, deltas: Vec<String>) {
        self.queue.extend(
            deltas
                .into_iter()
                .filter(|d| !d.is_empty())
                .map(|d| Ok(StreamEvent::TextDelta(d))),
        );
    }
}

/// Send `request` and turn its SSE body into core stream events: text
/// deltas as they arrive, then exactly one `Done` (or one error).
pub(crate) fn drive<D: SseDecoder>(
    request: reqwest::RequestBuilder,
    decoder: D,
) -> BoxStream<'static, Result<StreamEvent, ProviderError>> {
    run(Phase::Start(Box::new(request)), decoder)
}

/// Like [`drive`], for a successful response the caller already received
/// (after its own authentication handshake, for example).
pub(crate) fn drive_response<D: SseDecoder>(
    response: reqwest::Response,
    decoder: D,
) -> BoxStream<'static, Result<StreamEvent, ProviderError>> {
    run(Phase::Body(response.bytes_stream().boxed()), decoder)
}

fn run<D: SseDecoder>(
    phase: Phase,
    decoder: D,
) -> BoxStream<'static, Result<StreamEvent, ProviderError>> {
    let driver = Driver {
        phase,
        parser: SseParser::default(),
        decoder,
        queue: VecDeque::new(),
    };
    Box::pin(stream::unfold(driver, |mut d| async move {
        loop {
            if let Some(item) = d.queue.pop_front() {
                return Some((item, d));
            }
            match std::mem::replace(&mut d.phase, Phase::Done) {
                Phase::Done => return None,
                Phase::Start(request) => match request.send().await {
                    Err(err) => d.queue.push_back(Err(map_reqwest_error(err))),
                    Ok(response) if !response.status().is_success() => {
                        let status = response.status();
                        let headers = response.headers().clone();
                        let body = response.text().await.unwrap_or_default();
                        d.queue.push_back(Err(map_status(status, &headers, &body)));
                    }
                    Ok(response) => d.phase = Phase::Body(response.bytes_stream().boxed()),
                },
                Phase::Body(mut body) => match body.next().await {
                    Some(Ok(chunk)) => {
                        let events = d.parser.push(&chunk);
                        if !d.feed(events) {
                            d.phase = Phase::Body(body);
                        }
                    }
                    Some(Err(err)) => d.queue.push_back(Err(map_reqwest_error(err))),
                    None => {
                        let events = d.parser.finish();
                        if !d.feed(events) {
                            let done = d.decoder.finish(false).map(StreamEvent::Done);
                            d.queue.push_back(done);
                        }
                    }
                },
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;
    use serde_json::json;

    #[test]
    fn status_mapping() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("7"));
        assert_eq!(
            map_status(StatusCode::TOO_MANY_REQUESTS, &headers, ""),
            ProviderError::RateLimited {
                retry_after: Some(Duration::from_secs(7))
            }
        );
        assert!(matches!(
            map_status(
                StatusCode::TOO_MANY_REQUESTS,
                &headers,
                r#"{"error":{"type":"insufficient_quota","code":"insufficient_quota"}}"#
            ),
            ProviderError::Billing { .. }
        ));
        assert!(matches!(
            map_status(StatusCode::UNAUTHORIZED, &HeaderMap::new(), "x"),
            ProviderError::Auth(_)
        ));
        assert!(matches!(
            map_status(StatusCode::BAD_GATEWAY, &HeaderMap::new(), "x"),
            ProviderError::Transient(_)
        ));
        match map_status(
            StatusCode::BAD_REQUEST,
            &HeaderMap::new(),
            r#"{"error":{"message":"bad field"}}"#,
        ) {
            ProviderError::InvalidRequest(msg) => assert_eq!(msg, "HTTP 400: bad field"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn excerpt_is_truncated() {
        let long = "x".repeat(1000);
        assert!(body_excerpt(&long).chars().count() <= BODY_EXCERPT_CHARS + 1);
    }

    #[test]
    fn argument_parsing_round_trip() {
        assert_eq!(parse_arguments(r#"{"a":1}"#), json!({"a": 1}));
        assert_eq!(parse_arguments(""), json!({}));
        let broken = parse_arguments(r#"{"a":"#);
        assert_eq!(broken, json!({"_raw": r#"{"a":"#}));
        assert_eq!(raw_arguments(&broken), Some(r#"{"a":"#));
        assert_eq!(raw_arguments(&json!({"a": 1})), None);
    }
}
