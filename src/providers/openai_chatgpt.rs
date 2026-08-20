//! ChatGPT subscription provider (Codex backend, Responses API).
//!
//! Talks to `chatgpt.com/backend-api/codex/responses` using OAuth credentials
//! obtained by [`crate::oauth::chatgpt_codex`], so model calls draw on the user's
//! ChatGPT Plus/Pro/Business plan instead of a metered API key.
//!
//! The agent loop speaks OpenAI *chat completions* shapes, so this module is
//! mostly two translations: chat `messages`/`tools` into Responses `input`
//! items, and the Responses event stream back into a [`ProviderResponse`].
//! Streaming is the transport only — the endpoint requires an SSE accept
//! header — and the stream is consumed to completion before returning, matching
//! the non-streaming contract every other provider here honors.

use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Map, Value};
use tracing::{debug, warn};

use super::error::{ProviderError, ProviderErrorKind};
use super::streaming::SseFramer;
use crate::oauth::chatgpt_codex::{self, ChatGptCredentialManager, ChatGptCredentials};
use crate::traits::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
    ToolChoiceMode,
};

/// Codex backend root. Overridable for tests and self-hosted proxies.
pub const DEFAULT_BASE_URL: &str = chatgpt_codex::CODEX_BACKEND_BASE_URL;

/// Identifies the calling tool to OpenAI. Third-party clients declare
/// themselves here rather than impersonating the Codex CLI.
const ORIGINATOR: &str = "aidaemon";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Models reachable with a ChatGPT subscription. The backend exposes no
/// `/models` listing, so this is a static catalog; unknown ids still pass
/// through to the API, which is the authority on what an account can use.
///
/// Tiers: `sol` is the flagship, `terra` balanced, `luna` fast/low-cost.
const KNOWN_MODELS: &[&str] = &[
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.6-pro",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
];

/// Reasoning-effort values the backend accepts, weakest to strongest.
///
/// Passed through verbatim from `reasoning_effort` in config (or a per-call
/// override) — not validated here, because each model advertises its own
/// supported subset and the API is the authority on which apply.
pub const REASONING_EFFORT_LEVELS: &[&str] = &[
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
];

pub struct OpenAiChatGptProvider {
    client: Client,
    base_url: String,
    reasoning_effort: Option<String>,
    /// Shared with other subscription-backed adapters (for example image
    /// generation) so refresh-token rotation is serialized process-wide.
    credentials: std::sync::Arc<ChatGptCredentialManager>,
}

impl OpenAiChatGptProvider {
    /// No `max_tokens` parameter: the Codex backend accepts no output-cap field
    /// (see [`build_request_body`]), so taking one would only imply a limit this
    /// provider cannot enforce.
    pub fn new(base_url: Option<&str>) -> Result<Self, String> {
        let client = super::build_http_client(DEFAULT_TIMEOUT)?;
        let base_url = base_url
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_BASE_URL)
            .trim_end_matches('/')
            .to_string();
        Ok(Self {
            client,
            base_url,
            reasoning_effort: None,
            credentials: chatgpt_codex::shared_credential_manager(),
        })
    }

    pub fn with_reasoning_effort(mut self, effort: Option<String>) -> Self {
        let effort = effort.filter(|e| !e.trim().is_empty());
        // Warn but still send it: a typo like "hgh" would otherwise come back as
        // an opaque 400, while a genuinely new level must keep working without a
        // client update.
        if let Some(level) = effort.as_deref() {
            // `off` is aidaemon's provider-neutral "disable thinking" sentinel.
            // The ChatGPT backend spells that level `none`; it is normalized
            // when the request body is built.
            if level == "off" {
                self.reasoning_effort = effort;
                return self;
            }
            if !REASONING_EFFORT_LEVELS.contains(&level) {
                warn!(
                    effort = level,
                    known = REASONING_EFFORT_LEVELS.join(", "),
                    "Unrecognized reasoning_effort for the ChatGPT provider; sending it anyway"
                );
            }
        }
        self.reasoning_effort = effort;
        self
    }

    /// Return usable credentials, refreshing under lock when near expiry.
    async fn credentials(&self) -> Result<ChatGptCredentials, ProviderError> {
        self.credentials
            .usable_credentials(&self.client)
            .await
            .map_err(|error| ProviderError::from_status(401, &error.to_string()))
    }

    async fn send(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> Result<ProviderResponse, ProviderError> {
        let mut creds = self.credentials().await?;
        let body = build_request_body(
            model,
            messages,
            tools,
            options,
            options
                .reasoning_effort_override
                .as_deref()
                .or(self.reasoning_effort.as_deref()),
        );

        let mut response = self.send_authenticated(&creds, &body).await?;

        // The resource server is authoritative about token validity. Access
        // tokens can be revoked before their JWT/local expiry, so recover once
        // through the shared refresh-token state machine and replay this
        // side-effect-free inference request with the rotated credential.
        if response.status().as_u16() == 401 {
            let rejected = creds.access_token.clone();
            // Consume the response before reusing the connection and ensure no
            // credential or large body survives into diagnostics.
            let _ = response.bytes().await;
            creds = self
                .credentials
                .refresh_after_rejection(&self.client, &rejected)
                .await
                .map_err(|error| ProviderError::from_status(401, &error.to_string()))?;
            response = self.send_authenticated(&creds, &body).await?;
            if response.status().as_u16() == 401 {
                let body = response.text().await.unwrap_or_default();
                let detail = annotate_error_body(401, &body);
                self.credentials
                    .mark_rejected(&creds.access_token, detail.clone())
                    .await;
                return Err(ProviderError::from_status(401, &detail));
            }
        }

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::from_status(
                status.as_u16(),
                &annotate_error_body(status.as_u16(), &body),
            ));
        }

        let mut framer = SseFramer::default();
        let mut collector = ResponseCollector::default();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ProviderError::network(&e))?;
            for payload in framer.feed(&chunk) {
                if payload == "[DONE]" {
                    continue;
                }
                match serde_json::from_str::<Value>(&payload) {
                    Ok(event) => collector.apply(&event)?,
                    // A single unparseable frame is not worth failing the turn;
                    // a stream that never completes is caught below.
                    Err(e) => debug!(error = %e, "Skipping unparseable Codex SSE frame"),
                }
            }
        }

        collector.finish()
    }

    async fn send_authenticated(
        &self,
        creds: &ChatGptCredentials,
        body: &Value,
    ) -> Result<reqwest::Response, ProviderError> {
        let url = format!("{}/responses", self.base_url);
        self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", creds.access_token))
            .header("chatgpt-account-id", &creds.account_id)
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", ORIGINATOR)
            .header("User-Agent", user_agent())
            .header("accept", "text/event-stream")
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|error| ProviderError::network(&error))
    }
}

fn user_agent() -> String {
    format!(
        "aidaemon/{} ({}; {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

/// Make usage-limit rejections legible; the backend's own message is terse and
/// the distinction (plan cap vs. bad token) drives what the user should do.
fn annotate_error_body(status: u16, body: &str) -> String {
    if status == 429 {
        format!(
            "ChatGPT subscription usage limit reached (or requests are being throttled). \
             Falling back to another configured model. Details: {body}"
        )
    } else if status == 401 {
        format!(
            "ChatGPT subscription auth rejected. Re-connect with `aidaemon auth login openai`. \
             Details: {body}"
        )
    } else {
        body.to_string()
    }
}

/// Accumulates Responses stream events into a single response.
#[derive(Default)]
struct ResponseCollector {
    /// Text assembled from deltas, used when no output item carries text.
    delta_text: String,
    /// Completed output items, gathered from `response.output_item.done`.
    ///
    /// These are the real payload on the Codex backend: it closes the stream
    /// with `response.completed` carrying `"output": []` (it echoes no
    /// assembled output when `store` is false), so reading only the terminal
    /// event yields an empty answer with no tool calls.
    items: Vec<Value>,
    /// The terminal `response.completed` payload — authoritative for usage and
    /// for `output` when the endpoint actually populates it.
    completed: Option<Value>,
    /// A typed failure captured from the terminal stream event. Keeping the
    /// provider classification here lets the shared recovery layer distinguish
    /// overload/rate-limit infrastructure failures from refusals or filters.
    failure: Option<ProviderError>,
}

fn stream_failure(event: &Value, message: &str) -> ProviderError {
    let code = event
        .get("code")
        .or_else(|| event.pointer("/error/code"))
        .or_else(|| event.pointer("/response/error/code"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let status = event
        .get("status")
        .or_else(|| event.pointer("/error/status"))
        .or_else(|| event.pointer("/response/error/status"))
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok());
    let lower_message = message.to_ascii_lowercase();

    let kind = match status {
        Some(429) => ProviderErrorKind::RateLimit,
        Some(500 | 502 | 503 | 504) => ProviderErrorKind::ServerError,
        Some(408) => ProviderErrorKind::Timeout,
        _ if code.contains("rate_limit") => ProviderErrorKind::RateLimit,
        _ if code.contains("overload")
            || code == "server_error"
            || code == "service_unavailable" =>
        {
            ProviderErrorKind::ServerError
        }
        // Some Codex SSE error events currently omit both status and code.
        // In that protocol shape, the provider's explicit overload marker is
        // the only available infrastructure signal.
        _ if lower_message.contains("overload") => ProviderErrorKind::ServerError,
        _ => ProviderErrorKind::Unknown,
    };

    ProviderError {
        kind,
        status,
        message: format!("Codex stream failed: {message}"),
        malformed_reason: None,
        retry_after_secs: None,
        affordable_tokens: None,
    }
}

impl ResponseCollector {
    fn apply(&mut self, event: &Value) -> Result<(), ProviderError> {
        let Some(kind) = event.get("type").and_then(Value::as_str) else {
            return Ok(());
        };
        match kind {
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.delta_text.push_str(delta);
                }
            }
            "response.output_item.done" => {
                if let Some(item) = event.get("item") {
                    self.items.push(item.clone());
                }
            }
            "response.completed" => {
                self.completed = event.get("response").cloned();
            }
            "response.failed" | "response.incomplete" => {
                let message = event
                    .pointer("/response/error/message")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        event
                            .pointer("/response/incomplete_details/reason")
                            .and_then(Value::as_str)
                    })
                    .unwrap_or("the model stopped before completing the response");
                self.failure = Some(stream_failure(event, message));
            }
            "error" => {
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unspecified stream error");
                self.failure = Some(stream_failure(event, message));
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<ProviderResponse, ProviderError> {
        let ResponseCollector {
            delta_text,
            items,
            completed,
            failure,
        } = self;

        if let Some(response) = completed {
            let mut parsed = parse_output_items(&terminal_output(&response, items));
            // Deltas are the last resort: some item shapes carry no text field
            // even though text streamed through.
            if parsed.content.is_none() && parsed.tool_calls.is_empty() && !delta_text.is_empty() {
                parsed.content = Some(delta_text);
            }
            parsed.usage = parse_usage(&response);
            return Ok(parsed);
        }
        if let Some(failure) = failure {
            return Err(failure);
        }
        // Stream died before the completion event. Hand back whatever arrived
        // rather than discarding a partial answer.
        let mut parsed = parse_output_items(&items);
        if parsed.content.is_none() && !delta_text.is_empty() {
            parsed.content = Some(delta_text);
        }
        if parsed.content.is_some() || !parsed.tool_calls.is_empty() {
            parsed.response_note = Some(
                "Codex stream ended without a completion event; response may be truncated."
                    .to_string(),
            );
            return Ok(parsed);
        }
        Err(ProviderError::malformed_shape(
            "Codex stream ended with no response events",
        ))
    }
}

/// Pick the output items to parse: the terminal event's own `output` when the
/// endpoint populated it, else the items gathered from the stream.
///
/// The public Responses API echoes the assembled output in `response.completed`;
/// the Codex backend sends `"output": []` and relies on the per-item events.
fn terminal_output(response: &Value, streamed: Vec<Value>) -> Vec<Value> {
    match response.get("output").and_then(Value::as_array) {
        Some(output) if !output.is_empty() => output.clone(),
        _ => streamed,
    }
}

/// Convert Responses output items into a [`ProviderResponse`] (usage excluded;
/// it lives on the terminal event, not the items — see [`parse_usage`]).
fn parse_output_items(items: &[Value]) -> ProviderResponse {
    let mut content = String::new();
    let mut thinking = String::new();
    let mut tool_calls = Vec::new();

    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                for part in item
                    .get("content")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        content.push_str(text);
                    }
                }
            }
            Some("function_call") => {
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if name.is_empty() {
                    continue;
                }
                tool_calls.push(ToolCall {
                    id: item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name,
                    arguments: item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}")
                        .to_string(),
                    extra_content: None,
                });
            }
            Some("reasoning") => {
                for part in item
                    .get("summary")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        thinking.push_str(text);
                    }
                }
            }
            _ => {}
        }
    }

    ProviderResponse {
        content: (!content.is_empty()).then_some(content),
        tool_calls,
        usage: None,
        thinking: (!thinking.is_empty()).then_some(thinking),
        response_note: None,
    }
}

/// Read token usage off a terminal `response.completed` payload.
fn parse_usage(response: &Value) -> Option<TokenUsage> {
    response.get("usage").map(|u| TokenUsage {
        input_tokens: u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
        output_tokens: u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
        cached_input_tokens: u
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        cache_creation_input_tokens: None,
        model: response
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        ..Default::default()
    })
}

/// Extract plain text from a chat message `content` field, which may be a
/// string or an array of typed parts.
fn message_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                p.get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| p.as_str().map(str::to_string))
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Build Responses `content` parts for a user message, preserving images.
fn user_content_parts(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::Array(parts)) => {
            let mut out = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("image_url") => {
                        if let Some(url) = part
                            .pointer("/image_url/url")
                            .and_then(Value::as_str)
                            .or_else(|| part.get("image_url").and_then(Value::as_str))
                        {
                            out.push(json!({"type": "input_image", "image_url": url}));
                        }
                    }
                    _ => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            out.push(json!({"type": "input_text", "text": text}));
                        }
                    }
                }
            }
            if out.is_empty() {
                out.push(json!({"type": "input_text", "text": ""}));
            }
            out
        }
        other => vec![json!({"type": "input_text", "text": message_text(other)})],
    }
}

/// Translate chat-completions messages into Responses input items.
///
/// Returns `(instructions, input_items)`. System and developer messages become
/// top-level `instructions` because the Responses API has no system role.
fn build_input(messages: &[Value]) -> (String, Vec<Value>) {
    let mut instructions: Vec<String> = Vec::new();
    let mut input: Vec<Value> = Vec::new();
    // Final wire-level guard: a function_call_output is only valid after a
    // matching function_call item. Upstream history fitting should preserve
    // this invariant, but enforcing it here keeps malformed stored or trimmed
    // history from reaching the Responses endpoint as a hard 400.
    let mut pending_function_call_ids = std::collections::HashSet::new();

    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message.get("content");

        match role {
            "system" | "developer" => {
                let text = message_text(content);
                if !text.is_empty() {
                    instructions.push(text);
                }
            }
            "tool" => {
                let call_id = message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !pending_function_call_ids.remove(call_id) {
                    warn!(
                        call_id,
                        "Dropping function call output without a preceding unmatched function call"
                    );
                    continue;
                }
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message_text(content),
                }));
            }
            "assistant" => {
                let text = message_text(content);
                if !text.is_empty() {
                    input.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }));
                }
                for call in message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                {
                    let function = call.get("function").unwrap_or(call);
                    let call_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": function.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "arguments": function
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}"),
                    }));
                    if !call_id.is_empty() {
                        pending_function_call_ids.insert(call_id.to_string());
                    }
                }
            }
            _ => {
                input.push(json!({
                    "type": "message",
                    "role": "user",
                    "content": user_content_parts(content),
                }));
            }
        }
    }

    (instructions.join("\n\n"), input)
}

/// Translate chat tool definitions into Responses tool definitions.
///
/// Accepts both the wrapped (`{"type":"function","function":{…}}`) and bare
/// (`{"name":…,"parameters":…}`) shapes, since both circulate in this codebase.
fn build_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|tool| {
            let spec = tool.get("function").unwrap_or(tool);
            let name = spec.get("name").and_then(Value::as_str)?;
            Some(json!({
                "type": "function",
                "name": name,
                "description": spec.get("description").and_then(Value::as_str).unwrap_or(""),
                "parameters": spec
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            }))
        })
        .collect()
}

fn build_tool_choice(mode: &ToolChoiceMode) -> Value {
    match mode {
        ToolChoiceMode::Auto => json!("auto"),
        ToolChoiceMode::None => json!("none"),
        ToolChoiceMode::Required => json!("required"),
        ToolChoiceMode::Specific(name) => json!({"type": "function", "name": name}),
    }
}

/// Assemble the full request body for `POST /responses`.
///
/// The Codex backend is a restricted Responses proxy, not the public API: it
/// allowlists parameters and answers anything else with
/// `400 {"detail":"Unsupported parameter: …"}`. `max_output_tokens` is one of
/// the rejected fields, so no output cap is sent here — configured
/// `provider.max_tokens` and per-call `max_tokens_override` cannot be honored
/// on this backend, and the model's own limits apply instead.
fn build_request_body(
    model: &str,
    messages: &[Value],
    tools: &[Value],
    options: &ChatOptions,
    reasoning_effort: Option<&str>,
) -> Value {
    let (instructions, input) = build_input(messages);

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("input".into(), json!(input));
    body.insert("stream".into(), json!(true));
    // Never let the backend retain conversation state; aidaemon owns history.
    body.insert("store".into(), json!(false));

    if !instructions.is_empty() {
        body.insert("instructions".into(), json!(instructions));
    }
    if !tools.is_empty() {
        let mapped = build_tools(tools);
        if !mapped.is_empty() {
            body.insert("tools".into(), json!(mapped));
            body.insert(
                "tool_choice".into(),
                build_tool_choice(&options.tool_choice),
            );
        }
    }
    if let Some(effort) = reasoning_effort {
        // Internal agent-loop recovery and computer-use paths use `off` for
        // providers such as llama.cpp/Gemma. The ChatGPT Responses backend
        // rejects `off` with HTTP 400 and uses `none` for the same semantics.
        let effort = if effort == "off" { "none" } else { effort };
        body.insert("reasoning".into(), json!({"effort": effort}));
    }

    match &options.response_mode {
        ResponseMode::Text => {}
        ResponseMode::JsonObject => {
            body.insert("text".into(), json!({"format": {"type": "json_object"}}));
        }
        ResponseMode::JsonSchema {
            name,
            schema,
            strict,
        } => {
            body.insert(
                "text".into(),
                json!({"format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": strict,
                }}),
            );
        }
    }

    Value::Object(body)
}

#[async_trait]
impl ModelProvider for OpenAiChatGptProvider {
    async fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
    ) -> anyhow::Result<ProviderResponse> {
        self.chat_with_options(model, messages, tools, &ChatOptions::default())
            .await
    }

    async fn chat_with_options(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> anyhow::Result<ProviderResponse> {
        self.send(model, messages, tools, options)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e.user_message()))
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        Ok(KNOWN_MODELS.iter().map(|m| m.to_string()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct AuthRecoveryFixture {
        response_calls: std::sync::atomic::AtomicUsize,
        refresh_calls: std::sync::atomic::AtomicUsize,
    }

    async fn auth_recovery_responses(
        axum::extract::State(state): axum::extract::State<std::sync::Arc<AuthRecoveryFixture>>,
        headers: axum::http::HeaderMap,
    ) -> axum::response::Response {
        use axum::response::IntoResponse;

        state
            .response_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            != Some("Bearer synthetic-rotated-access")
        {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                axum::Json(json!({"error": "synthetic rejected access"})),
            )
                .into_response();
        }
        let stream = concat!(
            "data: {\"type\":\"response.completed\",\"response\":{\"model\":\"synthetic-model\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"recovered\"}]}],\"usage\":{\"input_tokens\":4,\"output_tokens\":1}}}\n\n",
            "data: [DONE]\n\n"
        );
        (
            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
            stream,
        )
            .into_response()
    }

    async fn auth_recovery_token(
        axum::extract::State(state): axum::extract::State<std::sync::Arc<AuthRecoveryFixture>>,
    ) -> axum::Json<Value> {
        state
            .refresh_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        axum::Json(json!({
            "access_token": "synthetic-rotated-access",
            "refresh_token": "synthetic-rotated-refresh",
            "expires_in": 3600
        }))
    }

    fn options() -> ChatOptions {
        ChatOptions::default()
    }

    #[tokio::test]
    async fn remote_401_forces_one_refresh_and_replays_inference_once() {
        use axum::routing::post;

        let fixture = std::sync::Arc::new(AuthRecoveryFixture::default());
        let app = axum::Router::new()
            .route("/responses", post(auth_recovery_responses))
            .route("/token", post(auth_recovery_token))
            .with_state(fixture.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind synthetic provider");
        let address = listener.local_addr().expect("synthetic provider address");
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let credentials = ChatGptCredentialManager::with_test_token_url(
            ChatGptCredentials {
                access_token: "synthetic-rejected-access".into(),
                refresh_token: "synthetic-refresh".into(),
                account_id: "acct-synthetic-1".into(),
                expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
            },
            format!("http://{address}/token"),
        );
        let provider = OpenAiChatGptProvider {
            client: reqwest::Client::new(),
            base_url: format!("http://{address}"),
            reasoning_effort: None,
            credentials: std::sync::Arc::new(credentials),
        };
        let response = provider
            .chat(
                "synthetic-model",
                &[json!({"role": "user", "content": "synthetic request"})],
                &[],
            )
            .await
            .expect("provider should recover");
        server.abort();

        assert_eq!(response.content.as_deref(), Some("recovered"));
        assert_eq!(
            fixture
                .response_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            2
        );
        assert_eq!(
            fixture
                .refresh_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn system_messages_become_instructions() {
        let messages = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hi"}),
        ];
        let (instructions, input) = build_input(&messages);
        assert_eq!(instructions, "You are helpful.");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "Hi");
    }

    #[test]
    fn multiple_system_messages_are_joined() {
        let messages = vec![
            json!({"role": "system", "content": "First."}),
            json!({"role": "developer", "content": "Second."}),
        ];
        let (instructions, input) = build_input(&messages);
        assert_eq!(instructions, "First.\n\nSecond.");
        assert!(input.is_empty());
    }

    #[test]
    fn assistant_tool_calls_and_results_round_trip() {
        let messages = vec![
            json!({"role": "user", "content": "What time is it?"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "clock", "arguments": "{\"tz\":\"UTC\"}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call-1", "content": "12:00"}),
        ];
        let (_, input) = build_input(&messages);
        assert_eq!(input.len(), 3);
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call-1");
        assert_eq!(input[1]["name"], "clock");
        assert_eq!(input[1]["arguments"], "{\"tz\":\"UTC\"}");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call-1");
        assert_eq!(input[2]["output"], "12:00");
    }

    #[test]
    fn orphaned_function_call_output_is_not_sent() {
        let messages = vec![
            json!({"role": "user", "content": "continue"}),
            // Mirrors a context-window cut that retained the result but dropped
            // the assistant function_call that originally preceded it.
            json!({
                "role": "tool",
                "tool_call_id": "call_trimmed",
                "content": "written"
            }),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_live",
                    "function": {"name": "browser", "arguments": "{}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_live", "content": "ok"}),
        ];

        let (_, input) = build_input(&messages);

        assert!(input
            .iter()
            .all(|item| { item.get("call_id").and_then(Value::as_str) != Some("call_trimmed") }));
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_live");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_live");
    }

    #[test]
    fn assistant_text_and_tool_calls_both_survive() {
        let messages = vec![json!({
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [{"id": "call-1", "function": {"name": "t", "arguments": "{}"}}]
        })];
        let (_, input) = build_input(&messages);
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["type"], "function_call");
    }

    #[test]
    fn user_images_map_to_input_image() {
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]
        })];
        let (_, input) = build_input(&messages);
        let parts = input[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "input_text");
        assert_eq!(parts[1]["type"], "input_image");
        assert_eq!(parts[1]["image_url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn tools_accept_wrapped_and_bare_shapes() {
        let tools = vec![
            json!({"name": "bare", "description": "d", "parameters": {"type": "object"}}),
            json!({"type": "function", "function": {"name": "wrapped", "parameters": {}}}),
        ];
        let mapped = build_tools(&tools);
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0]["type"], "function");
        assert_eq!(mapped[0]["name"], "bare");
        assert_eq!(mapped[1]["name"], "wrapped");
        // Responses puts name at the top level, not nested under "function".
        assert!(mapped[0].get("function").is_none());
    }

    #[test]
    fn request_body_carries_stream_and_no_store() {
        let body = build_request_body(
            "gpt-5.6-terra",
            &[json!({"role": "user", "content": "hi"})],
            &[],
            &options(),
            Some("low"),
        );
        assert_eq!(body["model"], "gpt-5.6-terra");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["reasoning"]["effort"], "low");
        assert!(body.get("tools").is_none());
    }

    /// Regression: the Codex backend rejects `max_output_tokens` outright with
    /// `400 {"detail":"Unsupported parameter: max_output_tokens"}`, which broke
    /// every request whenever `provider.max_tokens` was configured.
    #[test]
    fn request_body_never_sends_an_output_cap() {
        let opts = ChatOptions {
            max_tokens_override: Some(1024),
            ..Default::default()
        };
        let body = build_request_body(
            "gpt-5.6-terra",
            &[json!({"role": "user", "content": "hi"})],
            &[],
            &opts,
            None,
        );
        assert!(
            body.get("max_output_tokens").is_none(),
            "output cap leaked into the Codex request body: {body}"
        );
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn request_body_maps_internal_off_reasoning_to_none() {
        let body = build_request_body(
            "gpt-5.6-terra",
            &[json!({"role": "user", "content": "continue"})],
            &[],
            &options(),
            Some("off"),
        );
        assert_eq!(body["reasoning"]["effort"], "none");
        assert_ne!(body["reasoning"]["effort"], "off");
    }

    #[test]
    fn tool_choice_modes_map() {
        assert_eq!(build_tool_choice(&ToolChoiceMode::Auto), json!("auto"));
        assert_eq!(build_tool_choice(&ToolChoiceMode::None), json!("none"));
        assert_eq!(
            build_tool_choice(&ToolChoiceMode::Required),
            json!("required")
        );
        assert_eq!(
            build_tool_choice(&ToolChoiceMode::Specific("t".into())),
            json!({"type": "function", "name": "t"})
        );
    }

    #[test]
    fn json_schema_mode_maps_to_text_format() {
        let opts = ChatOptions {
            response_mode: ResponseMode::JsonSchema {
                name: "answer".into(),
                schema: json!({"type": "object"}),
                strict: true,
            },
            ..Default::default()
        };
        let body = build_request_body("m", &[], &[], &opts, None);
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["name"], "answer");
        assert_eq!(body["text"]["format"]["strict"], true);
    }

    #[test]
    fn parses_text_and_usage() {
        let response = json!({
            "model": "gpt-5.6-terra",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello there."}]
            }],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 8}
            }
        });
        let parsed = parse_output_items(&terminal_output(&response, Vec::new()));
        assert_eq!(parsed.content.as_deref(), Some("Hello there."));
        let usage = parse_usage(&response).unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.cached_input_tokens, Some(8));
        assert_eq!(usage.model, "gpt-5.6-terra");
    }

    #[test]
    fn parses_multiple_tool_calls_and_reasoning() {
        let response = json!({
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
                {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{\"x\":1}"},
                {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"}
            ]
        });
        let parsed = parse_output_items(&terminal_output(&response, Vec::new()));
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].id, "c1");
        assert_eq!(parsed.tool_calls[0].name, "a");
        assert_eq!(parsed.tool_calls[0].arguments, "{\"x\":1}");
        assert_eq!(parsed.tool_calls[1].id, "c2");
        assert_eq!(parsed.thinking.as_deref(), Some("thinking"));
        assert!(parsed.content.is_none());
    }

    #[test]
    fn function_call_without_name_is_dropped() {
        let response = json!({
            "output": [{"type": "function_call", "call_id": "c1", "arguments": "{}"}]
        });
        let parsed = parse_output_items(&terminal_output(&response, Vec::new()));
        assert!(parsed.tool_calls.is_empty());
    }

    /// Regression: the Codex backend closes the stream with `"output": []` and
    /// delivers the real payload in `response.output_item.done`, so parsing only
    /// the terminal event returned empty answers and swallowed every tool call.
    #[test]
    fn completed_with_empty_output_falls_back_to_streamed_items() {
        let stream = concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Paris\"}]}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Quito\\\"}\"}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.6-terra\",\"output\":[],\"usage\":{\"input_tokens\":9,\"output_tokens\":5}}}\n\n",
        );
        let parsed = collect(stream).expect("stream should parse");
        assert_eq!(parsed.content.as_deref(), Some("Paris"));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(parsed.tool_calls[0].id, "call_1");
        // Usage still comes from the terminal event, which is where it lives.
        let usage = parsed.usage.expect("usage missing");
        assert_eq!(usage.input_tokens, 9);
        assert_eq!(usage.model, "gpt-5.6-terra");
        // A completed stream is not a truncated one.
        assert!(parsed.response_note.is_none());
    }

    /// Drive the collector the way the transport does: frame raw SSE bytes,
    /// then apply each payload.
    fn collect(stream: &str) -> Result<ProviderResponse, ProviderError> {
        let mut framer = SseFramer::default();
        let mut collector = ResponseCollector::default();
        for payload in framer.feed(stream.as_bytes()) {
            if payload == "[DONE]" {
                continue;
            }
            if let Ok(event) = serde_json::from_str::<Value>(&payload) {
                collector.apply(&event).unwrap();
            }
        }
        collector.finish()
    }

    #[test]
    fn completed_event_wins_over_deltas() {
        let stream = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"par\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"tial\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"final answer\"}]}]}}\n\n",
            "data: [DONE]\n\n",
        );
        let parsed = collect(stream).unwrap();
        assert_eq!(parsed.content.as_deref(), Some("final answer"));
        assert!(parsed.response_note.is_none());
    }

    #[test]
    fn truncated_stream_returns_partial_text_with_note() {
        let stream =
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"half a sent\"}\n\n";
        let parsed = collect(stream).unwrap();
        assert_eq!(parsed.content.as_deref(), Some("half a sent"));
        assert!(parsed.response_note.is_some());
    }

    #[test]
    fn stream_error_event_fails_the_turn() {
        let stream = "data: {\"type\":\"error\",\"message\":\"model overloaded\"}\n\n";
        let err = collect(stream).unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::ServerError);
        assert!(err.message.contains("model overloaded"));
        assert!(err.is_retryable());
    }

    #[test]
    fn stream_error_uses_structured_rate_limit_code() {
        let stream = "data: {\"type\":\"error\",\"code\":\"rate_limit_exceeded\",\"message\":\"capacity unavailable\"}\n\n";
        let err = collect(stream).unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::RateLimit);
        assert!(err.is_retryable());
    }

    #[test]
    fn failed_response_event_surfaces_reason() {
        let stream = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"content filtered\"}}}\n\n";
        let err = collect(stream).unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Unknown);
        assert!(err.user_message().contains("content filtered"));
    }

    #[test]
    fn empty_stream_is_an_error() {
        assert!(collect("").is_err());
    }

    #[test]
    fn rate_limit_body_is_annotated() {
        let annotated = annotate_error_body(429, "{\"detail\":\"limit\"}");
        assert!(annotated.contains("usage limit"));
        let auth = annotate_error_body(401, "bad token");
        assert!(auth.contains("aidaemon auth login openai"));
        assert_eq!(annotate_error_body(500, "boom"), "boom");
    }

    #[test]
    fn base_url_defaults_and_trims() {
        let provider = OpenAiChatGptProvider::new(None).unwrap();
        assert_eq!(provider.base_url, DEFAULT_BASE_URL);
        let custom = OpenAiChatGptProvider::new(Some("https://example.test/api/")).unwrap();
        assert_eq!(custom.base_url, "https://example.test/api");
    }

    // ── Live smoke tests ────────────────────────────────────────────────
    //
    // These hit the real Codex backend with your stored subscription
    // credentials. Everything above is fixture-level; these are what actually
    // prove the request mapping and SSE parsing match the live API.
    //
    //   aidaemon auth login openai
    //   AIDAEMON_LIVE_TEST=1 cargo test --lib openai_chatgpt::tests::live \
    //       -- --ignored --nocapture --test-threads=1
    //
    // Override the model with AIDAEMON_LIVE_MODEL=<id>.

    fn live_enabled() -> bool {
        std::env::var("AIDAEMON_LIVE_TEST").as_deref() == Ok("1")
    }

    fn live_model() -> String {
        std::env::var("AIDAEMON_LIVE_MODEL").unwrap_or_else(|_| "gpt-5.6-terra".to_string())
    }

    /// Skip loudly rather than passing silently, so a misconfigured run cannot
    /// look like a successful one.
    fn live_preconditions() -> bool {
        if !live_enabled() {
            eprintln!("SKIPPED: set AIDAEMON_LIVE_TEST=1 to run live tests");
            return false;
        }
        if !crate::oauth::chatgpt_codex::is_connected() {
            eprintln!("SKIPPED: no ChatGPT login found — run `aidaemon auth login openai`");
            return false;
        }
        true
    }

    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_plain_text_round_trip() {
        if !live_preconditions() {
            return;
        }
        let provider = OpenAiChatGptProvider::new(None).unwrap();
        let messages = vec![
            json!({"role": "system", "content": "Answer with a single word."}),
            json!({"role": "user", "content": "What is the capital of France?"}),
        ];

        let response = provider
            .chat(&live_model(), &messages, &[])
            .await
            .expect("live chat call failed");

        let content = response.content.unwrap_or_default();
        eprintln!("content: {content}");
        eprintln!("usage: {:?}", response.usage);
        assert!(!content.trim().is_empty(), "model returned no text");
        assert!(
            content.to_lowercase().contains("paris"),
            "unexpected answer: {content}"
        );
        let usage = response.usage.expect("no usage reported");
        assert!(usage.input_tokens > 0, "input tokens not reported");
        assert!(usage.output_tokens > 0, "output tokens not reported");
    }

    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_tool_call_round_trip() {
        if !live_preconditions() {
            return;
        }
        let provider = OpenAiChatGptProvider::new(None).unwrap();
        let tools = vec![json!({
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        })];
        let messages = vec![json!({
            "role": "user",
            "content": "Use the get_weather tool to check the weather in Quito."
        })];

        let response = provider
            .chat(&live_model(), &messages, &tools)
            .await
            .expect("live tool call failed");

        eprintln!("tool_calls: {:?}", response.tool_calls);
        let call = response
            .tool_calls
            .first()
            .expect("model returned no tool call — request mapping may be wrong");
        assert_eq!(call.name, "get_weather");
        assert!(!call.id.is_empty(), "tool call has no id to reply against");
        let args: Value =
            serde_json::from_str(&call.arguments).expect("tool arguments are not valid JSON");
        assert!(args.get("city").is_some(), "missing city argument: {args}");
    }

    /// The round trip that matters most: send a tool result back and confirm
    /// the model continues the conversation from it.
    #[tokio::test]
    #[ignore = "hits the real ChatGPT backend; run with --ignored"]
    async fn live_tool_result_continuation() {
        if !live_preconditions() {
            return;
        }
        let provider = OpenAiChatGptProvider::new(None).unwrap();
        let messages = vec![
            json!({"role": "user", "content": "What is the weather in Quito?"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_live_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Quito\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_live_1",
                "content": "18C and raining"
            }),
        ];
        let tools = vec![json!({
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        })];

        let response = provider
            .chat(&live_model(), &messages, &tools)
            .await
            .expect("live continuation failed");

        let content = response.content.unwrap_or_default();
        eprintln!("continuation: {content}");
        assert!(
            content.contains("18") || content.to_lowercase().contains("rain"),
            "model did not use the tool result: {content}"
        );
    }
}
