//! OpenAI-compatible Chat Completions provider.

use std::collections::BTreeMap;
use std::time::Duration;

use aidaemon_core::{
    collect_stream, Image, Message, ModelRequest, ModelResponse, Provider, ProviderError,
    ResponseFormat, Role, StreamEvent, ToolCall, ToolChoice, Usage,
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header::HeaderMap;
use serde_json::{json, Value};

use crate::sse::SseEvent;
use crate::transport::{
    body_excerpt, build_client, drive, header_map, parse_arguments, raw_arguments, ConfigError,
    Secret, SseDecoder, DEFAULT_CONNECT_TIMEOUT, DEFAULT_TIMEOUT,
};

/// Default base URL (OpenAI). Point it at OpenRouter, Ollama
/// (`http://localhost:11434/v1`), llama.cpp, vLLM, etc.
pub const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// A provider for any server that speaks the OpenAI Chat Completions API
/// with SSE streaming (`POST {base_url}/chat/completions`).
#[derive(Debug, Clone)]
pub struct OpenAiCompatible {
    name: String,
    client: reqwest::Client,
    endpoint: String,
    api_key: Option<Secret>,
    headers: HeaderMap,
    timeout: Duration,
    defaults: Defaults,
}

/// What a request that sets none of its own gets.
#[derive(Debug, Clone, Default)]
struct Defaults {
    max_tokens: Option<u32>,
    reasoning_effort: Option<String>,
}

impl OpenAiCompatible {
    pub fn builder() -> OpenAiCompatibleBuilder {
        OpenAiCompatibleBuilder::default()
    }

    /// The JSON body this provider would send for `request`.
    pub fn request_body(&self, request: &ModelRequest) -> Value {
        build_body(request, &self.defaults)
    }
}

/// Builder for [`OpenAiCompatible`].
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleBuilder {
    name: String,
    base_url: String,
    api_key: Option<Secret>,
    headers: Vec<(String, String)>,
    timeout: Duration,
    connect_timeout: Duration,
    defaults: Defaults,
}

impl Default for OpenAiCompatibleBuilder {
    fn default() -> Self {
        Self {
            name: "openai-compatible".into(),
            base_url: OPENAI_DEFAULT_BASE_URL.into(),
            api_key: None,
            headers: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            defaults: Defaults::default(),
        }
    }
}

impl OpenAiCompatibleBuilder {
    /// Name reported by [`Provider::name`] (default `openai-compatible`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// API root, without `/chat/completions` (default `https://api.openai.com/v1`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sent as `Authorization: Bearer <key>`. Optional for local servers.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(Secret(key.into()));
        self
    }

    /// Extra header on every request (e.g. OpenRouter's `HTTP-Referer`).
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Whole-request timeout, including the streamed body (default 10 min).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Connection timeout (default 30 s).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// The output cap for requests that set none.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.defaults.max_tokens = Some(tokens);
        self
    }

    /// The reasoning effort for requests that set none, sent as
    /// `reasoning: {effort}` (OpenRouter and compatible servers); `off`
    /// sends nothing.
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        let effort = effort.into();
        self.defaults.reasoning_effort = (!effort.trim().is_empty()).then_some(effort);
        self
    }

    pub fn build(self) -> Result<OpenAiCompatible, ConfigError> {
        Ok(OpenAiCompatible {
            name: self.name,
            client: build_client(self.connect_timeout)?,
            endpoint: format!("{}/chat/completions", self.base_url.trim_end_matches('/')),
            api_key: self.api_key.filter(|k| !k.0.is_empty()),
            headers: header_map(&self.headers)?,
            timeout: self.timeout,
            defaults: self.defaults,
        })
    }
}

#[async_trait]
impl Provider for OpenAiCompatible {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        collect_stream(self.stream(request)).await
    }

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        let mut http = self
            .client
            .post(&self.endpoint)
            .timeout(self.timeout)
            .headers(self.headers.clone())
            .header("accept", "text/event-stream")
            .json(&build_body(request, &self.defaults));
        if let Some(key) = &self.api_key {
            http = http.bearer_auth(&key.0);
        }
        drive(http, ChatDecoder::default())
    }
}

fn build_body(request: &ModelRequest, defaults: &Defaults) -> Value {
    let mut body = json!({
        "model": request.model,
        "messages": wire_messages(&request.messages),
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if !request.tools.is_empty() {
        body["tools"] = request
            .tools
            .iter()
            .map(|spec| json!({"type": "function", "function": spec}))
            .collect();
        body["tool_choice"] = match &request.tool_choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::None => json!("none"),
            ToolChoice::Required => json!("required"),
            ToolChoice::Specific(name) => {
                json!({"type": "function", "function": {"name": name}})
            }
        };
        body["parallel_tool_calls"] = json!(true);
    }
    // Audio models otherwise default to answering in audio.
    if request.messages.iter().any(|m| !m.audio.is_empty()) {
        body["modalities"] = json!(["text"]);
    }
    if let Some(max) = request.max_output_tokens.or(defaults.max_tokens) {
        body["max_tokens"] = json!(max);
    }
    let effort = request
        .options
        .reasoning_effort
        .as_deref()
        .or(defaults.reasoning_effort.as_deref());
    if let Some(effort) = effort.filter(|effort| *effort != "off") {
        body["reasoning"] = json!({"effort": effort});
    }
    match &request.options.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::JsonObject => {
            body["response_format"] = json!({"type": "json_object"});
        }
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => {
            body["response_format"] = json!({"type": "json_schema", "json_schema": {
                "name": name,
                "schema": schema,
                "strict": strict,
            }});
        }
    }
    body
}

/// Map the transcript to Chat Completions messages. Tool messages there are
/// text-only, so images from a run of consecutive tool results follow it in
/// one user message, labelled with the call each image came from.
fn wire_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    let mut tool_images: Vec<Value> = Vec::new();
    for message in messages {
        if message.role != Role::Tool && !tool_images.is_empty() {
            out.push(json!({"role": "user", "content": std::mem::take(&mut tool_images)}));
        }
        out.push(wire_message(message));
        if message.role == Role::Tool && !message.images.is_empty() {
            let call_id = message.tool_call_id.as_deref().unwrap_or_default();
            tool_images.push(json!({
                "type": "text",
                "text": format!("[Images returned by tool call {call_id}]"),
            }));
            tool_images.extend(message.images.iter().map(image_part));
        }
    }
    if !tool_images.is_empty() {
        out.push(json!({"role": "user", "content": tool_images}));
    }
    out
}

fn image_part(image: &Image) -> Value {
    json!({"type": "image_url", "image_url": {"url": image.to_url()}})
}

fn wire_message(message: &Message) -> Value {
    match message.role {
        Role::System => json!({"role": "system", "content": message.content}),
        Role::User if message.images.is_empty() && message.audio.is_empty() => {
            json!({"role": "user", "content": message.content})
        }
        Role::User => {
            let mut parts = Vec::with_capacity(message.images.len() + 1);
            if !message.content.trim().is_empty() {
                parts.push(json!({"type": "text", "text": message.content}));
            }
            parts.extend(message.images.iter().map(image_part));
            parts.extend(message.audio.iter().map(|audio| {
                json!({"type": "input_audio", "input_audio": {
                    "data": audio.data,
                    "format": audio.format,
                }})
            }));
            json!({"role": "user", "content": parts})
        }
        Role::Assistant => {
            let mut out = json!({"role": "assistant", "content": message.content});
            if !message.tool_calls.is_empty() {
                out["tool_calls"] = message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        let arguments = match raw_arguments(&call.arguments) {
                            Some(raw) => raw.to_string(),
                            None => call.arguments.to_string(),
                        };
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": arguments},
                        })
                    })
                    .collect();
            }
            out
        }
        Role::Tool => json!({
            "role": "tool",
            "tool_call_id": message.tool_call_id.clone().unwrap_or_default(),
            "content": message.content,
        }),
    }
}

#[derive(Debug, Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
struct ChatDecoder {
    text: String,
    reasoning: String,
    calls: BTreeMap<usize, PartialCall>,
    usage: Usage,
    finished: bool,
}

impl ChatDecoder {
    fn tool_delta(&mut self, position: usize, delta: &Value) {
        let id = delta.get("id").and_then(Value::as_str).unwrap_or_default();
        // Some servers omit `index`; fall back to "new id => new call".
        let index = match delta.get("index").and_then(Value::as_u64) {
            Some(index) => index as usize,
            None => {
                let last = self.calls.keys().next_back().copied();
                match last {
                    Some(last) if id.is_empty() || self.calls[&last].id == id => last,
                    Some(last) => last + 1,
                    None => position,
                }
            }
        };
        let call = self.calls.entry(index).or_default();
        if !id.is_empty() {
            call.id = id.to_string();
        }
        if let Some(function) = delta.get("function") {
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                // Normally sent once; tolerate servers that repeat it.
                if call.name.is_empty() {
                    call.name = name.to_string();
                } else if call.name != name {
                    call.name.push_str(name);
                }
            }
            match function.get("arguments") {
                Some(Value::String(fragment)) => call.arguments.push_str(fragment),
                // A few servers send arguments as an already-parsed object.
                Some(value @ Value::Object(_)) => call.arguments = value.to_string(),
                _ => {}
            }
        }
    }
}

fn map_usage(usage: &Value) -> Usage {
    let get = |v: Option<&Value>| v.and_then(Value::as_u64).unwrap_or(0);
    Usage {
        input_tokens: get(usage.get("prompt_tokens")),
        output_tokens: get(usage.get("completion_tokens")),
        cached_input_tokens: get(usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .or_else(|| usage.get("cached_tokens"))),
        cache_creation_tokens: 0,
    }
}

impl SseDecoder for ChatDecoder {
    fn on_event(
        &mut self,
        event: &SseEvent,
        deltas: &mut Vec<String>,
    ) -> Result<bool, ProviderError> {
        let data = event.data.trim();
        if data == "[DONE]" {
            return Ok(true);
        }
        if data.is_empty() {
            return Ok(false);
        }
        let chunk: Value = serde_json::from_str(data).map_err(|err| {
            ProviderError::Transient(format!(
                "malformed stream chunk ({err}): {}",
                body_excerpt(data)
            ))
        })?;
        if let Some(error) = chunk.get("error").filter(|e| !e.is_null()) {
            return Err(ProviderError::Transient(format!(
                "stream error: {}",
                body_excerpt(&json!({ "error": error }).to_string())
            )));
        }
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = map_usage(usage);
        }
        for choice in chunk
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            // Only the first choice is meaningful (we never request n > 1).
            if choice.get("index").and_then(Value::as_u64).unwrap_or(0) != 0 {
                continue;
            }
            let delta = choice.get("delta").or_else(|| choice.get("message"));
            if let Some(delta) = delta {
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    if !text.is_empty() {
                        self.text.push_str(text);
                        deltas.push(text.to_string());
                    }
                }
                // DeepSeek and Moonshot send `reasoning_content`, OpenRouter
                // sends `reasoning`.
                let thought = ["reasoning_content", "reasoning"]
                    .iter()
                    .find_map(|field| delta.get(*field).and_then(Value::as_str));
                if let Some(thought) = thought {
                    self.reasoning.push_str(thought);
                }
                if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for (position, call) in calls.iter().enumerate() {
                        self.tool_delta(position, call);
                    }
                }
            }
            if choice
                .get("finish_reason")
                .is_some_and(|reason| !reason.is_null())
            {
                self.finished = true;
            }
        }
        Ok(false)
    }

    fn finish(&mut self, terminated: bool) -> Result<ModelResponse, ProviderError> {
        if !terminated && !self.finished {
            return Err(ProviderError::Transient(
                "stream ended before the model finished".into(),
            ));
        }
        let tool_calls = std::mem::take(&mut self.calls)
            .into_values()
            .filter(|call| !call.name.is_empty())
            .map(|call| ToolCall::new(call.id, call.name, parse_arguments(&call.arguments)))
            .collect();
        Ok(ModelResponse {
            text: std::mem::take(&mut self.text),
            tool_calls,
            usage: self.usage,
            reasoning: (!self.reasoning.is_empty()).then(|| std::mem::take(&mut self.reasoning)),
            ..Default::default()
        })
    }
}
