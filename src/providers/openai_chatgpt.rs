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

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Map, Value};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use super::error::{ProviderError, ProviderErrorKind};
use super::streaming::SseFramer;
use crate::oauth::chatgpt_codex::{self, ChatGptCredentials};
use crate::traits::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
    ToolChoiceMode,
};

/// Codex backend root. Overridable for tests and self-hosted proxies.
pub const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

/// Identifies the calling tool to OpenAI. Third-party clients declare
/// themselves here rather than impersonating the Codex CLI.
const ORIGINATOR: &str = "aidaemon";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Models reachable with a ChatGPT subscription. The backend exposes no
/// `/models` listing, so this is a static catalog; unknown ids still pass
/// through to the API, which is the authority on what an account can use.
const KNOWN_MODELS: &[&str] = &[
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1",
    "gpt-5",
    "codex-mini-latest",
];

pub struct OpenAiChatGptProvider {
    client: Client,
    base_url: String,
    max_tokens: Option<u32>,
    reasoning_effort: Option<String>,
    /// Cached credentials. The mutex also serializes refresh so concurrent
    /// agent turns cannot each burn a rotation of the refresh token.
    credentials: Arc<Mutex<Option<ChatGptCredentials>>>,
}

impl OpenAiChatGptProvider {
    pub fn new(base_url: Option<&str>, max_tokens: Option<u32>) -> Result<Self, String> {
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
            max_tokens,
            reasoning_effort: None,
            credentials: Arc::new(Mutex::new(None)),
        })
    }

    pub fn with_reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.reasoning_effort = effort.filter(|e| !e.trim().is_empty());
        self
    }

    /// Return usable credentials, refreshing under lock when near expiry.
    async fn credentials(&self) -> Result<ChatGptCredentials, ProviderError> {
        let mut guard = self.credentials.lock().await;

        if guard.is_none() {
            *guard = chatgpt_codex::load_credentials();
        }

        let current = guard.clone().ok_or_else(|| {
            ProviderError::from_status(
                401,
                "No ChatGPT subscription login found. Run `aidaemon auth login openai` to connect \
                 your ChatGPT account.",
            )
        })?;

        if !current.needs_refresh(chrono::Utc::now()) {
            return Ok(current);
        }

        debug!("Refreshing ChatGPT subscription access token");
        match chatgpt_codex::refresh_credentials(&self.client, &current).await {
            Ok(refreshed) => {
                // Persist immediately: OpenAI rotates refresh tokens and the old
                // one may already be dead.
                if let Err(e) = chatgpt_codex::store_credentials(&refreshed) {
                    warn!(error = %e, "Refreshed ChatGPT tokens could not be persisted");
                }
                *guard = Some(refreshed.clone());
                Ok(refreshed)
            }
            Err(e) => {
                // Drop the cached copy so the next call reloads from storage
                // rather than retrying a token we know is dead.
                *guard = None;
                Err(ProviderError::from_status(401, &e.to_string()))
            }
        }
    }

    async fn send(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> Result<ProviderResponse, ProviderError> {
        let creds = self.credentials().await?;
        let body = build_request_body(
            model,
            messages,
            tools,
            options,
            options.max_tokens_override.or(self.max_tokens),
            options
                .reasoning_effort_override
                .as_deref()
                .or(self.reasoning_effort.as_deref()),
        );

        let url = format!("{}/responses", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", creds.access_token))
            .header("chatgpt-account-id", &creds.account_id)
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", ORIGINATOR)
            .header("User-Agent", user_agent())
            .header("accept", "text/event-stream")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::network(&e))?;

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
    /// Text assembled from deltas, used when no terminal event arrives.
    delta_text: String,
    /// The authoritative payload from `response.completed`.
    completed: Option<Value>,
    failure: Option<String>,
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
            "response.completed" => {
                self.completed = event.get("response").cloned();
            }
            "response.failed" | "response.incomplete" => {
                self.failure = Some(
                    event
                        .pointer("/response/error/message")
                        .and_then(Value::as_str)
                        .or_else(|| {
                            event
                                .pointer("/response/incomplete_details/reason")
                                .and_then(Value::as_str)
                        })
                        .unwrap_or("the model stopped before completing the response")
                        .to_string(),
                );
            }
            "error" => {
                self.failure = Some(
                    event
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unspecified stream error")
                        .to_string(),
                );
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<ProviderResponse, ProviderError> {
        if let Some(response) = self.completed {
            return Ok(parse_response_payload(&response));
        }
        if let Some(failure) = self.failure {
            // Not a server outage: a failed/incomplete response is usually a
            // content filter, refusal, or truncation. `Unknown` keeps the real
            // reason in the user-facing message instead of replacing it with a
            // generic "provider is having issues, will retry".
            return Err(ProviderError {
                kind: ProviderErrorKind::Unknown,
                status: None,
                message: format!("Codex stream failed: {failure}"),
                malformed_reason: None,
                retry_after_secs: None,
                affordable_tokens: None,
            });
        }
        if !self.delta_text.is_empty() {
            // Stream died mid-response but text arrived. Hand back what we have
            // rather than discarding a partial answer.
            return Ok(ProviderResponse {
                content: Some(self.delta_text),
                tool_calls: Vec::new(),
                usage: None,
                thinking: None,
                response_note: Some(
                    "Codex stream ended without a completion event; response may be truncated."
                        .to_string(),
                ),
            });
        }
        Err(ProviderError::malformed_shape(
            "Codex stream ended with no response events",
        ))
    }
}

/// Convert a completed Responses payload into a [`ProviderResponse`].
fn parse_response_payload(response: &Value) -> ProviderResponse {
    let mut content = String::new();
    let mut thinking = String::new();
    let mut tool_calls = Vec::new();

    for item in response
        .get("output")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
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

    let usage = response.get("usage").map(|u| TokenUsage {
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
    });

    ProviderResponse {
        content: (!content.is_empty()).then_some(content),
        tool_calls,
        usage,
        thinking: (!thinking.is_empty()).then_some(thinking),
        response_note: None,
    }
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
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.get("id").and_then(Value::as_str).unwrap_or_default(),
                        "name": function.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "arguments": function
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}"),
                    }));
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
fn build_request_body(
    model: &str,
    messages: &[Value],
    tools: &[Value],
    options: &ChatOptions,
    max_tokens: Option<u32>,
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
    if let Some(limit) = max_tokens {
        body.insert("max_output_tokens".into(), json!(limit));
    }
    if let Some(effort) = reasoning_effort {
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

    fn options() -> ChatOptions {
        ChatOptions::default()
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
            "gpt-5.1-codex",
            &[json!({"role": "user", "content": "hi"})],
            &[],
            &options(),
            Some(1024),
            Some("low"),
        );
        assert_eq!(body["model"], "gpt-5.1-codex");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(body["max_output_tokens"], 1024);
        assert_eq!(body["reasoning"]["effort"], "low");
        assert!(body.get("tools").is_none());
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
        let body = build_request_body("m", &[], &[], &opts, None, None);
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["name"], "answer");
        assert_eq!(body["text"]["format"]["strict"], true);
    }

    #[test]
    fn parses_text_and_usage() {
        let response = json!({
            "model": "gpt-5.1-codex",
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
        let parsed = parse_response_payload(&response);
        assert_eq!(parsed.content.as_deref(), Some("Hello there."));
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.cached_input_tokens, Some(8));
        assert_eq!(usage.model, "gpt-5.1-codex");
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
        let parsed = parse_response_payload(&response);
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
        assert!(parse_response_payload(&response).tool_calls.is_empty());
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
        assert!(err.user_message().contains("model overloaded"));
    }

    #[test]
    fn failed_response_event_surfaces_reason() {
        let stream = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"content filtered\"}}}\n\n";
        let err = collect(stream).unwrap_err();
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
        let provider = OpenAiChatGptProvider::new(None, None).unwrap();
        assert_eq!(provider.base_url, DEFAULT_BASE_URL);
        let custom = OpenAiChatGptProvider::new(Some("https://example.test/api/"), None).unwrap();
        assert_eq!(custom.base_url, "https://example.test/api");
    }
}
