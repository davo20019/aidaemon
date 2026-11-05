//! Calling an SDK [`sdk::Provider`] with the daemon's chat-completions JSON.
//!
//! Every backend lives in `aidaemon-providers` and every model call goes
//! through the SDK `Provider` trait. Many daemon call sites still assemble
//! chat-completions messages, so these helpers convert them to SDK messages
//! and the reply back, map SDK errors to the daemon's classified errors (for
//! its recovery decisions and telemetry), and report stream progress to the
//! model-call watchdog.

use aidaemon_core as sdk;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::error::{MalformedResponseReason, ProviderError, ProviderErrorKind};
use crate::traits::{
    ChatOptions, ProviderResponse, ResponseMode, TokenUsage, ToolCall, ToolChoiceMode,
};

/// How a provider's models are listed: the SDK has no listing call.
#[derive(Clone, Debug)]
pub(crate) enum ModelList {
    Fixed(Vec<String>),
    /// `GET {base}/models` with a bearer key (OpenAI-compatible servers).
    OpenAiModels {
        base_url: String,
        api_key: Option<String>,
    },
}

impl ModelList {
    pub(crate) async fn list(&self) -> anyhow::Result<Vec<String>> {
        match self {
            ModelList::Fixed(models) => Ok(models.clone()),
            ModelList::OpenAiModels { base_url, api_key } => {
                let client = super::build_http_client(std::time::Duration::from_secs(30))
                    .map_err(anyhow::Error::msg)?;
                let mut request = client.get(format!("{}/models", base_url.trim_end_matches('/')));
                if let Some(key) = api_key {
                    request = request.bearer_auth(key);
                }
                let body: Value = request.send().await?.error_for_status()?.json().await?;
                Ok(body["data"]
                    .as_array()
                    .map(|models| {
                        models
                            .iter()
                            .filter_map(|m| m["id"].as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default())
            }
        }
    }
}

/// Chat-completions calls on any SDK provider, for daemon code that builds
/// chat JSON. Bring it into scope to call `provider.chat(..)`.
#[async_trait]
pub(crate) trait ChatJson {
    async fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
    ) -> anyhow::Result<ProviderResponse>;

    async fn chat_with_options(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> anyhow::Result<ProviderResponse>;
}

#[async_trait]
impl<P: sdk::Provider + ?Sized> ChatJson for P {
    async fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
    ) -> anyhow::Result<ProviderResponse> {
        chat_with_options(self, model, messages, tools, &ChatOptions::default()).await
    }

    async fn chat_with_options(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> anyhow::Result<ProviderResponse> {
        chat_with_options(self, model, messages, tools, options).await
    }
}

/// One model call: `messages` and `tools` in chat-completions JSON. A
/// provider failure comes back as the daemon's classified [`ProviderError`].
pub(crate) async fn chat_with_options<P: sdk::Provider + ?Sized>(
    provider: &P,
    model: &str,
    messages: &[Value],
    tools: &[Value],
    options: &ChatOptions,
) -> anyhow::Result<ProviderResponse> {
    let request = to_sdk_request(model, messages, tools, options);
    send(provider, &request)
        .await
        .map(|response| to_provider_response(response, model))
}

/// Stream `request` to its end, noting progress for the watchdog.
pub(crate) async fn send<P: sdk::Provider + ?Sized>(
    provider: &P,
    request: &sdk::ModelRequest,
) -> anyhow::Result<sdk::ModelResponse> {
    let mut events = provider.stream(request);
    while let Some(event) = events.next().await {
        super::streaming::note_stream_progress();
        match event {
            Ok(sdk::StreamEvent::Done(response)) => return Ok(response),
            Ok(_) => {}
            Err(error) => return Err(to_daemon_error(error).into()),
        }
    }
    Err(ProviderError::malformed_shape("provider stream ended without a response").into())
}

pub(crate) fn to_sdk_request(
    model: &str,
    messages: &[Value],
    tools: &[Value],
    options: &ChatOptions,
) -> sdk::ModelRequest {
    sdk::ModelRequest {
        model: model.to_string(),
        messages: to_sdk_messages(messages),
        tools: tools.iter().filter_map(to_sdk_tool).collect(),
        tool_choice: match &options.tool_choice {
            ToolChoiceMode::Auto => sdk::ToolChoice::Auto,
            ToolChoiceMode::None => sdk::ToolChoice::None,
            ToolChoiceMode::Required => sdk::ToolChoice::Required,
            ToolChoiceMode::Specific(name) => sdk::ToolChoice::Specific(name.clone()),
        },
        max_output_tokens: options.max_tokens_override,
        options: sdk::RequestOptions {
            response_format: match &options.response_mode {
                ResponseMode::Text => sdk::ResponseFormat::Text,
                ResponseMode::JsonObject => sdk::ResponseFormat::JsonObject,
                ResponseMode::JsonSchema {
                    name,
                    schema,
                    strict,
                } => sdk::ResponseFormat::JsonSchema {
                    name: name.clone(),
                    schema: schema.clone(),
                    strict: *strict,
                },
            },
            reasoning_effort: options.reasoning_effort_override.clone(),
            single_attempt: options.single_attempt_fail_closed,
        },
    }
}

/// A tool definition in either the wrapped (`{"type":"function",
/// "function":{…}}`) or bare (`{"name":…}`) shape.
fn to_sdk_tool(tool: &Value) -> Option<sdk::ToolSpec> {
    let spec = tool.get("function").unwrap_or(tool);
    Some(sdk::ToolSpec {
        name: spec.get("name")?.as_str()?.to_string(),
        description: spec
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        parameters: spec
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    })
}

/// Text, images and audio of a chat `content` field (a string or parts).
fn split_content(content: Option<&Value>) -> (String, Vec<sdk::Image>, Vec<sdk::Audio>) {
    let mut text = Vec::new();
    let mut images = Vec::new();
    let mut audio = Vec::new();
    match content {
        Some(Value::String(s)) => text.push(s.clone()),
        Some(Value::Array(parts)) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("image_url") => {
                        let url = part
                            .pointer("/image_url/url")
                            .or_else(|| part.get("image_url"))
                            .and_then(Value::as_str);
                        if let Some(url) = url {
                            images.push(sdk::Image::from_url_or_data(url));
                        }
                    }
                    Some("input_audio") => {
                        let data = part.pointer("/input_audio/data").and_then(Value::as_str);
                        let format = part.pointer("/input_audio/format").and_then(Value::as_str);
                        if let (Some(data), Some(format)) = (data, format) {
                            audio.push(sdk::Audio::new(format, data));
                        }
                    }
                    _ => {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            text.push(t.to_string());
                        } else if let Some(t) = part.as_str() {
                            text.push(t.to_string());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    (text.join("\n"), images, audio)
}

/// Chat-completions messages as SDK messages, keeping what a provider
/// needs: images and audio, reasoning signatures on tool calls, and tool
/// arguments that are not valid JSON (as the SDK's `_raw` form, so the
/// call is replayed byte for byte).
pub(crate) fn to_sdk_messages(messages: &[Value]) -> Vec<sdk::Message> {
    messages
        .iter()
        .filter_map(|value| {
            let (text, images, audio) = split_content(value.get("content"));
            Some(match value.get("role").and_then(Value::as_str)? {
                "system" | "developer" => sdk::Message::system(text),
                "user" => sdk::Message::user(text)
                    .with_images(images)
                    .with_audio(audio),
                "assistant" => {
                    let calls = value
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .map(|calls| calls.iter().filter_map(to_sdk_call).collect())
                        .unwrap_or_default();
                    sdk::Message::assistant(text, calls)
                }
                "tool" => {
                    let call = value
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .or_else(|| value.get("name").and_then(Value::as_str))?;
                    sdk::Message::tool_result(call, text).with_images(images)
                }
                _ => return None,
            })
        })
        .collect()
}

fn to_sdk_call(call: &Value) -> Option<sdk::ToolCall> {
    let function = call.get("function").unwrap_or(call);
    let raw = function
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let arguments = match serde_json::from_str::<Value>(raw) {
        Ok(value) if value.is_object() => value,
        _ if raw.trim().is_empty() => json!({}),
        _ => json!({ "_raw": raw }),
    };
    let mut sdk_call = sdk::ToolCall::new(
        call.get("id").and_then(Value::as_str).unwrap_or_default(),
        function.get("name")?.as_str()?,
        arguments,
    );
    sdk_call.provider_data = call
        .get("extra_content")
        .filter(|data| !data.is_null())
        .cloned();
    Some(sdk_call)
}

pub(crate) fn to_provider_response(response: sdk::ModelResponse, model: &str) -> ProviderResponse {
    let usage = response.usage;
    ProviderResponse {
        content: (!response.text.is_empty()).then_some(response.text),
        tool_calls: response
            .tool_calls
            .into_iter()
            .map(|call| ToolCall {
                arguments: match call.arguments.get("_raw").and_then(Value::as_str) {
                    Some(raw) if call.arguments.as_object().is_some_and(|o| o.len() == 1) => {
                        raw.to_string()
                    }
                    _ => call.arguments.to_string(),
                },
                id: call.id,
                name: call.name,
                extra_content: call.provider_data,
            })
            .collect(),
        usage: Some(TokenUsage {
            input_tokens: usage.input_tokens.min(u32::MAX as u64) as u32,
            output_tokens: usage.output_tokens.min(u32::MAX as u64) as u32,
            cached_input_tokens: (usage.cached_input_tokens > 0)
                .then(|| usage.cached_input_tokens.min(u32::MAX as u64) as u32),
            cache_creation_input_tokens: (usage.cache_creation_tokens > 0)
                .then(|| usage.cache_creation_tokens.min(u32::MAX as u64) as u32),
            model: model.to_string(),
            ..Default::default()
        }),
        thinking: response.reasoning,
        response_note: response.note,
    }
}

/// The options a request carries, as daemon [`ChatOptions`] (for logging
/// and assertions in tests).
pub(crate) fn to_chat_options(request: &sdk::ModelRequest) -> ChatOptions {
    ChatOptions {
        response_mode: match &request.options.response_format {
            sdk::ResponseFormat::Text => ResponseMode::Text,
            sdk::ResponseFormat::JsonObject => ResponseMode::JsonObject,
            sdk::ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => ResponseMode::JsonSchema {
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        },
        tool_choice: match &request.tool_choice {
            sdk::ToolChoice::Auto => ToolChoiceMode::Auto,
            sdk::ToolChoice::None => ToolChoiceMode::None,
            sdk::ToolChoice::Required => ToolChoiceMode::Required,
            sdk::ToolChoice::Specific(name) => ToolChoiceMode::Specific(name.clone()),
        },
        max_tokens_override: request.max_output_tokens,
        reasoning_effort_override: request.options.reasoning_effort.clone(),
        single_attempt_fail_closed: request.options.single_attempt,
        ..Default::default()
    }
}

/// Tool definitions in the wrapped chat-completions shape.
pub(crate) fn to_chat_tools(specs: &[sdk::ToolSpec]) -> Vec<Value> {
    specs
        .iter()
        .map(|spec| {
            json!({"type": "function", "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }})
        })
        .collect()
}

fn image_block(image: &sdk::Image) -> Value {
    json!({"type": "image_url", "image_url": {"url": image.to_url()}})
}

/// Text, images and audio as chat content: a plain string with no media,
/// content parts with it.
fn chat_content(text: &str, images: &[sdk::Image], audio: &[sdk::Audio]) -> Value {
    if images.is_empty() && audio.is_empty() {
        return Value::String(text.to_string());
    }
    let mut blocks = Vec::with_capacity(images.len() + audio.len() + 1);
    if !text.trim().is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    blocks.extend(images.iter().map(image_block));
    blocks.extend(audio.iter().map(|audio| {
        json!({"type": "input_audio", "input_audio": {"data": audio.data, "format": audio.format}})
    }));
    Value::Array(blocks)
}

/// SDK messages as chat-completions messages. Images a tool observed cannot
/// ride on a `tool` message for every backend, so they follow the run of
/// tool results as a user message; injected user-role directives become
/// system messages.
pub(crate) fn to_chat_messages(messages: &[sdk::Message]) -> Vec<Value> {
    let mut names: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(messages.len());
    let mut pending_images: Vec<Value> = Vec::new();
    for message in messages {
        if message.role != sdk::Role::Tool && !pending_images.is_empty() {
            out.append(&mut pending_images);
        }
        match message.role {
            sdk::Role::System => out.push(json!({"role": "system", "content": message.content})),
            sdk::Role::User if message.is_injected() => out.push(json!({
                "role": "system",
                "content": message.content,
            })),
            sdk::Role::User => out.push(json!({
                "role": "user",
                "content": chat_content(&message.content, &message.images, &message.audio),
            })),
            sdk::Role::Assistant => {
                let mut value = json!({"role": "assistant", "content": message.content});
                if !message.tool_calls.is_empty() {
                    let calls: Vec<Value> = message
                        .tool_calls
                        .iter()
                        .map(|call| {
                            names.insert(call.id.as_str(), call.name.as_str());
                            let arguments = match call.arguments.get("_raw").and_then(Value::as_str)
                            {
                                Some(raw)
                                    if call.arguments.as_object().is_some_and(|o| o.len() == 1) =>
                                {
                                    raw.to_string()
                                }
                                _ => call.arguments.to_string(),
                            };
                            let mut value = json!({
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": arguments},
                            });
                            // Reasoning signatures and thinking blocks the
                            // provider needs back on replay.
                            if let Some(data) = &call.provider_data {
                                value["extra_content"] = data.clone();
                            }
                            value
                        })
                        .collect();
                    value["tool_calls"] = Value::Array(calls);
                }
                out.push(value);
            }
            sdk::Role::Tool => {
                let call_id = message.tool_call_id.as_deref().unwrap_or_default();
                let name = names.get(call_id).copied().unwrap_or("tool");
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": message.content,
                }));
                if !message.images.is_empty() {
                    let label = format!("[Image observed by `{name}`]");
                    pending_images.push(json!({
                        "role": "user",
                        "content": chat_content(&label, &message.images, &[]),
                    }));
                }
            }
        }
    }
    out.append(&mut pending_images);
    out
}

/// A daemon [`ChatOptions`]-free reply as an SDK response.
pub(crate) fn to_sdk_response(response: ProviderResponse) -> sdk::ModelResponse {
    let usage = response.usage.as_ref();
    sdk::ModelResponse {
        text: response.content.unwrap_or_default(),
        tool_calls: response
            .tool_calls
            .into_iter()
            .map(|call| {
                let arguments = serde_json::from_str::<Value>(&call.arguments)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| {
                        if call.arguments.trim().is_empty() {
                            json!({})
                        } else {
                            json!({ "_raw": call.arguments })
                        }
                    });
                let mut sdk_call = sdk::ToolCall::new(call.id, call.name, arguments);
                sdk_call.provider_data = call.extra_content.filter(|data| !data.is_null());
                sdk_call
            })
            .collect(),
        usage: sdk::Usage {
            input_tokens: usage.map_or(0, |u| u64::from(u.input_tokens)),
            output_tokens: usage.map_or(0, |u| u64::from(u.output_tokens)),
            cached_input_tokens: usage
                .and_then(|u| u.cached_input_tokens)
                .map_or(0, u64::from),
            cache_creation_tokens: usage
                .and_then(|u| u.cache_creation_input_tokens)
                .map_or(0, u64::from),
        },
        reasoning: response.thinking,
        note: response.response_note,
    }
}

/// A daemon classified error as the SDK's typed error (for test providers
/// that script daemon errors).
#[cfg(test)]
pub(crate) fn to_sdk_error(error: &ProviderError) -> sdk::ProviderError {
    let detail = match error.status {
        Some(status) => format!("HTTP {status}: {}", error.message),
        None => error.message.clone(),
    };
    match error.kind {
        ProviderErrorKind::Auth => sdk::ProviderError::Auth(detail),
        ProviderErrorKind::Billing => sdk::ProviderError::Billing {
            message: detail,
            affordable_tokens: error.affordable_tokens.map(u64::from),
        },
        ProviderErrorKind::RateLimit => sdk::ProviderError::RateLimited {
            retry_after: error.retry_after_secs.map(std::time::Duration::from_secs),
        },
        ProviderErrorKind::BadRequest => match super::context_overflow(&error.message) {
            Some(limit) => sdk::ProviderError::ContextOverflow {
                message: detail,
                limit: limit.map(|l| l as u64),
            },
            None => sdk::ProviderError::InvalidRequest(detail),
        },
        ProviderErrorKind::NotFound => sdk::ProviderError::NotFound(detail),
        ProviderErrorKind::Timeout => {
            sdk::ProviderError::Transient(format!("request timed out: {}", error.message))
        }
        ProviderErrorKind::Network => sdk::ProviderError::Transient(detail),
        ProviderErrorKind::ServerError => sdk::ProviderError::Transient(detail),
        ProviderErrorKind::MalformedResponse => sdk::ProviderError::Malformed {
            message: detail,
            parse: error.malformed_reason == Some(MalformedResponseReason::Parse),
        },
        ProviderErrorKind::Unknown => sdk::ProviderError::Other(detail),
    }
}

/// The HTTP status at the start of an SDK error detail (`HTTP 503: …`).
fn status_in(message: &str) -> Option<u16> {
    let rest = message.split("HTTP ").nth(1)?;
    rest.get(..3)?.parse().ok()
}

/// An SDK error as the daemon's classified error, so the recovery engine
/// picks the same route it always has.
pub(crate) fn to_daemon_error(error: sdk::ProviderError) -> ProviderError {
    let text = error.to_string();
    let status = status_in(&text);
    let mut out = ProviderError::from_status(status.unwrap_or(0), &text);
    out.status = status;
    out.message = text.clone();
    out.kind = match &error {
        sdk::ProviderError::RateLimited { retry_after } => {
            out.retry_after_secs = retry_after.map(|d| d.as_secs().max(1));
            ProviderErrorKind::RateLimit
        }
        sdk::ProviderError::Auth(_) => ProviderErrorKind::Auth,
        sdk::ProviderError::Billing {
            affordable_tokens, ..
        } => {
            out.affordable_tokens =
                affordable_tokens.map(|tokens| tokens.min(u32::MAX as u64) as u32);
            ProviderErrorKind::Billing
        }
        sdk::ProviderError::NotFound(_) => ProviderErrorKind::NotFound,
        sdk::ProviderError::InvalidRequest(_) | sdk::ProviderError::ContextOverflow { .. } => {
            ProviderErrorKind::BadRequest
        }
        sdk::ProviderError::Malformed { parse, .. } => {
            out.malformed_reason = Some(if *parse {
                MalformedResponseReason::Parse
            } else {
                MalformedResponseReason::Shape
            });
            ProviderErrorKind::MalformedResponse
        }
        sdk::ProviderError::Transient(message) => {
            if message.starts_with("request timed out") {
                ProviderErrorKind::Timeout
            } else if status.is_some_and(|s| s >= 500) {
                ProviderErrorKind::ServerError
            } else if status == Some(408) {
                ProviderErrorKind::Timeout
            } else if message.starts_with("Codex stream failed") || message.contains("overload") {
                ProviderErrorKind::ServerError
            } else {
                ProviderErrorKind::Network
            }
        }
        sdk::ProviderError::Other(_) => ProviderErrorKind::Unknown,
    };
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ProviderRecoveryRoute as Route;

    #[test]
    fn requests_carry_options_tools_images_audio_and_signatures() {
        let messages = vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": [
                {"type": "text", "text": "What is said here?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQID"}},
                {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "ogg"}}
            ]}),
            json!({"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "terminal", "arguments": "{\"command\":\"ls\"}"},
                "extra_content": {"thought_signature": "sig-synthetic"}
            }, {
                "id": "c2", "type": "function",
                "function": {"name": "terminal", "arguments": "{not json"}
            }]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "README.md"}),
        ];
        let tools = vec![json!({"type": "function", "function": {
            "name": "terminal", "description": "Run", "parameters": {"type": "object"}
        }})];
        let options = ChatOptions {
            tool_choice: ToolChoiceMode::Required,
            max_tokens_override: Some(900),
            reasoning_effort_override: Some("low".into()),
            single_attempt_fail_closed: true,
            ..Default::default()
        };
        let request = to_sdk_request("gpt-6-sol", &messages, &tools, &options);
        assert_eq!(request.messages.len(), 4);
        assert_eq!(request.messages[1].images.len(), 1);
        assert_eq!(request.messages[1].audio.len(), 1);
        let calls = &request.messages[2].tool_calls;
        assert_eq!(
            calls[0].provider_data,
            Some(json!({"thought_signature": "sig-synthetic"}))
        );
        assert_eq!(calls[1].arguments, json!({"_raw": "{not json"}));
        assert_eq!(request.tools[0].name, "terminal");
        assert_eq!(request.tool_choice, sdk::ToolChoice::Required);
        assert_eq!(request.max_output_tokens, Some(900));
        assert_eq!(request.options.reasoning_effort.as_deref(), Some("low"));
        assert!(request.options.single_attempt);
    }

    #[test]
    fn responses_keep_reasoning_notes_signatures_and_cache_usage() {
        let mut call = sdk::ToolCall::new("c1", "terminal", json!({"command": "ls"}));
        call.provider_data = Some(json!({"thought_signature": "sig-synthetic"}));
        let raw = sdk::ToolCall::new("c2", "terminal", json!({"_raw": "{not json"}));
        let response = to_provider_response(
            sdk::ModelResponse {
                text: "Listing.".into(),
                tool_calls: vec![call, raw],
                usage: sdk::Usage {
                    input_tokens: 100,
                    output_tokens: 7,
                    cached_input_tokens: 64,
                    cache_creation_tokens: 0,
                },
                reasoning: Some("Thought.".into()),
                note: Some("truncated".into()),
            },
            "gpt-6-sol",
        );
        assert_eq!(response.content.as_deref(), Some("Listing."));
        assert_eq!(
            response.tool_calls[0].extra_content,
            Some(json!({"thought_signature": "sig-synthetic"}))
        );
        assert_eq!(response.tool_calls[1].arguments, "{not json");
        let usage = response.usage.unwrap();
        assert_eq!(usage.cached_input_tokens, Some(64));
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(response.thinking.as_deref(), Some("Thought."));
        assert_eq!(response.response_note.as_deref(), Some("truncated"));
    }

    /// SDK errors land on the recovery routes the daemon's own backends
    /// produced.
    #[test]
    fn errors_keep_their_recovery_routes() {
        let route = |error| to_daemon_error(error).recovery_route(false);
        assert_eq!(
            route(sdk::ProviderError::RateLimited { retry_after: None }),
            Route::RetryRateLimitThenFallback
        );
        assert_eq!(
            route(sdk::ProviderError::Transient("HTTP 503: down".into())),
            Route::RetryTransientThenFallback
        );
        assert_eq!(
            route(sdk::ProviderError::Auth("HTTP 401: bad".into())),
            Route::Fail
        );
        assert_eq!(
            route(sdk::ProviderError::Billing {
                message: "HTTP 402".into(),
                affordable_tokens: Some(6917)
            }),
            Route::RetryReducedBudgetThenFallback
        );
        assert_eq!(
            route(sdk::ProviderError::NotFound("HTTP 404".into())),
            Route::Fallback
        );
        assert_eq!(
            route(sdk::ProviderError::InvalidRequest("HTTP 400: bad".into())),
            Route::AdaptRequestOnce
        );
        assert_eq!(
            route(sdk::ProviderError::Malformed {
                message: "cut".into(),
                parse: true
            }),
            Route::RetryMalformedThenFallback
        );
        let billing = to_daemon_error(sdk::ProviderError::Billing {
            message: "HTTP 402".into(),
            affordable_tokens: Some(6917),
        });
        assert_eq!(billing.affordable_tokens, Some(6917));
        let server = to_daemon_error(sdk::ProviderError::Transient("HTTP 503: down".into()));
        assert_eq!(server.kind, ProviderErrorKind::ServerError);
        assert_eq!(server.status, Some(503));
        let timeout = to_daemon_error(sdk::ProviderError::Transient(
            "request timed out: deadline".into(),
        ));
        assert_eq!(timeout.kind, ProviderErrorKind::Timeout);
    }
}
