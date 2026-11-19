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
use serde_json::{Value, json};

use super::error::{MalformedResponseReason, ProviderError, ProviderErrorKind};
use aidaemon_events::conversation::ToolCall;
use aidaemon_events::provider::{
    ChatOptions, ProviderResponse, ResponseMode, TokenUsage, ToolChoiceMode,
};

/// How a provider's models are listed: the SDK has no listing call.
#[derive(Clone, Debug)]
pub enum ModelList {
    Fixed(Vec<String>),
    /// `GET {base}/models` with a bearer key (OpenAI-compatible servers).
    OpenAiModels {
        base_url: String,
        api_key: Option<String>,
    },
    /// The ChatGPT subscription's own listing; `known` when it fails.
    ChatGpt {
        backend: std::sync::Arc<aidaemon_providers::ChatGpt>,
        known: Vec<String>,
    },
}

impl ModelList {
    pub async fn list(&self) -> anyhow::Result<Vec<String>> {
        match self {
            ModelList::Fixed(models) => Ok(models.clone()),
            ModelList::ChatGpt { backend, known } => match backend.list_models().await {
                Ok(listed) => {
                    let mut models: Vec<String> = listed
                        .into_iter()
                        .filter(|m| m.listed)
                        .map(|m| m.id)
                        .collect();
                    // The configured models stay offered even when the
                    // listing leaves them out.
                    for model in known {
                        if !models.contains(model) {
                            models.push(model.clone());
                        }
                    }
                    Ok(models)
                }
                Err(error) => {
                    tracing::warn!(%error, "ChatGPT model listing failed; offering the known models");
                    Ok(known.clone())
                }
            },
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
/// chat JSON. Bring it into scope to call `provider.chat_with_options(..)`.
#[async_trait]
pub trait ChatJson {
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
pub async fn chat_with_options<P: sdk::Provider + ?Sized>(
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
pub async fn send<P: sdk::Provider + ?Sized>(
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

pub fn to_sdk_request(
    model: &str,
    messages: &[Value],
    tools: &[Value],
    options: &ChatOptions,
) -> sdk::ModelRequest {
    let mut messages = to_sdk_messages(messages);
    drop_other_models_reasoning(&mut messages, model);
    sdk::ModelRequest {
        model: model.to_string(),
        messages,
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
            fallback_when_unspent: options.fallback_when_unspent,
            cache_key: options.cache_key.clone(),
        },
        tool_surfacing: None,
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
pub fn to_sdk_messages(messages: &[Value]) -> Vec<sdk::Message> {
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

/// The key a tool call's provider data carries the model that produced it
/// under. Adapters read only their own keys, so it rides along unseen.
pub const PRODUCED_BY_KEY: &str = "_produced_by";

/// Reasoning a provider hands back for replay (encrypted reasoning items,
/// thinking blocks, thought signatures) is only valid for the model that
/// produced it. When a conversation moves to another model it is dropped,
/// and the tool calls and text it came with stay.
pub fn drop_other_models_reasoning(messages: &mut [sdk::Message], model: &str) {
    for call in messages
        .iter_mut()
        .flat_map(|message| message.tool_calls.iter_mut())
    {
        let produced_by = call
            .provider_data
            .as_ref()
            .and_then(|data| data.get(PRODUCED_BY_KEY))
            .and_then(Value::as_str);
        if produced_by.is_some_and(|produced_by| produced_by != model) {
            call.provider_data = None;
        }
    }
}

/// `data` stamped with the model that produced it (an object only: other
/// shapes are left as they are).
fn stamped(data: Option<Value>, model: &str) -> Option<Value> {
    match data {
        Some(Value::Object(mut map)) => {
            map.insert(
                PRODUCED_BY_KEY.to_string(),
                Value::String(model.to_string()),
            );
            Some(Value::Object(map))
        }
        other => other,
    }
}

pub fn to_provider_response(response: sdk::ModelResponse, model: &str) -> ProviderResponse {
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
                extra_content: stamped(call.provider_data, model),
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
        cut_off: response.cut_off,
    }
}

/// The options a request carries, as daemon [`ChatOptions`] (for logging
/// and assertions in tests).
pub fn to_chat_options(request: &sdk::ModelRequest) -> ChatOptions {
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
        fallback_when_unspent: request.options.fallback_when_unspent,
        cache_key: request.options.cache_key.clone(),
        ..Default::default()
    }
}

/// Tool definitions in the wrapped chat-completions shape.
pub fn to_chat_tools(specs: &[sdk::ToolSpec]) -> Vec<Value> {
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
pub fn to_chat_messages(messages: &[sdk::Message]) -> Vec<Value> {
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
pub fn to_sdk_response(response: ProviderResponse) -> sdk::ModelResponse {
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
        cut_off: response.cut_off,
    }
}

/// A daemon classified error as the SDK's typed error (for test providers
/// that script daemon errors).
#[cfg(any(test, feature = "testing"))]
pub fn to_sdk_error(error: &ProviderError) -> sdk::ProviderError {
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
        ProviderErrorKind::BadRequest if error.rejected_tools => {
            sdk::ProviderError::InvalidTools(detail)
        }
        ProviderErrorKind::BadRequest => match error.context_overflow {
            Some(limit) => sdk::ProviderError::ContextOverflow {
                message: detail,
                limit,
            },
            None => sdk::ProviderError::InvalidRequest(detail),
        },
        ProviderErrorKind::NotFound => sdk::ProviderError::NotFound(detail),
        ProviderErrorKind::QuotaExhausted => sdk::ProviderError::QuotaExhausted {
            message: detail,
            resets_at: error.resets_at,
            per_model: false,
            in_stream: false,
        },
        ProviderErrorKind::Timeout => sdk::ProviderError::Transient(
            aidaemon_providers::TransportFailure::Timeout.message(&error.message),
        ),
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
pub fn to_daemon_error(error: sdk::ProviderError) -> ProviderError {
    let text = error.to_string();
    let status = status_in(&text);
    let mut out = ProviderError::from_status(status.unwrap_or(0), &text);
    out.status = status;
    out.message = text.clone();
    out.spent_nothing = error.proves_nothing_spent();
    // The SDK error's variant is the typed classification; only
    // `ContextOverflow` below carries an overflow.
    out.context_overflow = None;
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
        sdk::ProviderError::QuotaExhausted { resets_at, .. } => {
            out.resets_at = *resets_at;
            ProviderErrorKind::QuotaExhausted
        }
        sdk::ProviderError::InvalidRequest(_) => ProviderErrorKind::BadRequest,
        sdk::ProviderError::ContextOverflow { limit, .. } => {
            out.context_overflow = Some(*limit);
            ProviderErrorKind::BadRequest
        }
        sdk::ProviderError::InvalidTools(_) => {
            out.rejected_tools = true;
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
        // A transient failure is the network's when the transport raised it
        // (its typed marker), else the provider's own (a status, a stream
        // failure event such as an overload): never from the words.
        sdk::ProviderError::Transient(message) => {
            match (aidaemon_providers::TransportFailure::of(message), status) {
                (Some(aidaemon_providers::TransportFailure::Timeout), _) | (None, Some(408)) => {
                    ProviderErrorKind::Timeout
                }
                (Some(_), _) => ProviderErrorKind::Network,
                (None, _) => ProviderErrorKind::ServerError,
            }
        }
        sdk::ProviderError::Other(_) => ProviderErrorKind::Unknown,
    };
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_bridge::ProviderRecoveryRoute as Route;

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
                cut_off: false,
            },
            "gpt-6-sol",
        );
        assert_eq!(response.content.as_deref(), Some("Listing."));
        assert_eq!(
            response.tool_calls[0].extra_content,
            Some(json!({"thought_signature": "sig-synthetic", PRODUCED_BY_KEY: "gpt-6-sol"}))
        );
        assert_eq!(response.tool_calls[1].arguments, "{not json");
        let usage = response.usage.unwrap();
        assert_eq!(usage.cached_input_tokens, Some(64));
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(response.thinking.as_deref(), Some("Thought."));
        assert_eq!(response.response_note.as_deref(), Some("truncated"));
    }

    /// A conversation moved to another model sends its tool calls without
    /// the reasoning the earlier model produced; the same model, and data
    /// stored before stamping, keep it.
    #[test]
    fn reasoning_replays_only_to_the_model_that_produced_it() {
        let call = |id: &str, data: Value| {
            json!({
                "id": id, "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
                "extra_content": data,
            })
        };
        let messages = vec![json!({"role": "assistant", "content": "", "tool_calls": [
            call("c1", json!({"chatgpt_reasoning_items": ["sealed"], PRODUCED_BY_KEY: "model-a"})),
            call("c2", json!({"thought_signature": "sig", PRODUCED_BY_KEY: "model-b"})),
            call("c3", json!({"thought_signature": "older"})),
        ]})];
        let request = to_sdk_request("model-b", &messages, &[], &ChatOptions::default());
        let calls = &request.messages[0].tool_calls;
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].provider_data, None);
        assert_eq!(calls[0].name, "terminal");
        assert!(calls[1].provider_data.is_some());
        assert_eq!(
            calls[2].provider_data,
            Some(json!({"thought_signature": "older"}))
        );
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
            Route::Fallback
        );
        assert_eq!(
            route(sdk::ProviderError::QuotaExhausted {
                message: "HTTP 429: plan cap".into(),
                resets_at: None,
                per_model: false,
                in_stream: false,
            }),
            Route::Fallback
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
        let reset =
            std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_800_000_000);
        let capped = to_daemon_error(sdk::ProviderError::QuotaExhausted {
            message: "HTTP 429: plan cap".into(),
            resets_at: Some(reset),
            per_model: false,
            in_stream: false,
        });
        assert_eq!(capped.kind, ProviderErrorKind::QuotaExhausted);
        assert_eq!(capped.resets_at, Some(reset));
        let back = to_sdk_error(&capped);
        assert!(
            matches!(back, sdk::ProviderError::QuotaExhausted { resets_at: Some(at), .. } if at == reset)
        );
        let server = to_daemon_error(sdk::ProviderError::Transient("HTTP 503: down".into()));
        assert_eq!(server.kind, ProviderErrorKind::ServerError);
        assert_eq!(server.status, Some(503));
        let timeout = to_daemon_error(sdk::ProviderError::Transient(
            "request timed out: deadline".into(),
        ));
        assert_eq!(timeout.kind, ProviderErrorKind::Timeout);
        // A transport failure is the network's; a provider's stream failure
        // with no status is the provider's, whatever its words.
        let network = to_daemon_error(sdk::ProviderError::Transient(
            aidaemon_providers::TransportFailure::Connect.message("refused"),
        ));
        assert_eq!(network.kind, ProviderErrorKind::Network);
        let stream = to_daemon_error(sdk::ProviderError::Transient(
            "stream error (api_error): something broke".into(),
        ));
        assert_eq!(stream.kind, ProviderErrorKind::ServerError);
    }

    /// A context overflow stays typed through the daemon's error and back,
    /// however the provider worded it: nothing re-reads the message.
    #[test]
    fn a_context_overflow_is_carried_typed_not_reparsed() {
        let error = to_daemon_error(sdk::ProviderError::ContextOverflow {
            message: "HTTP 400: Your request is bigger than this model accepts.".into(),
            limit: Some(64_000),
        });
        assert_eq!(error.kind, ProviderErrorKind::BadRequest);
        assert_eq!(error.context_overflow, Some(Some(64_000)));
        assert!(matches!(
            to_sdk_error(&error),
            sdk::ProviderError::ContextOverflow {
                limit: Some(64_000),
                ..
            }
        ));
        // A plain rejection that quotes the old wording is not an overflow.
        let plain = to_daemon_error(sdk::ProviderError::InvalidRequest(
            "HTTP 400: the prompt is too long for this field".into(),
        ));
        assert_eq!(plain.context_overflow, None);
        assert!(matches!(
            to_sdk_error(&plain),
            sdk::ProviderError::InvalidRequest(_)
        ));
    }
}
