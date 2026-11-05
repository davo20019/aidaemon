//! The daemon's [`ModelProvider`] over an SDK [`sdk::Provider`].
//!
//! Every backend now lives in `aidaemon-providers`; this facade keeps the
//! daemon's call sites and its recovery engine working unchanged while they
//! move to the SDK interface. It converts chat-completions JSON to SDK
//! messages and back, maps SDK errors to the daemon's error kinds (so the
//! recovery routes stay the same), and reports stream progress to the
//! model-call watchdog.

use std::sync::Arc;

use aidaemon_core as sdk;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::error::{MalformedResponseReason, ProviderError, ProviderErrorKind};
use crate::traits::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
    ToolChoiceMode,
};

/// How a facade lists its models: the SDK has no listing call.
#[derive(Clone)]
pub(crate) enum ModelList {
    Fixed(Vec<String>),
    /// `GET {base}/models` with a bearer key (OpenAI-compatible servers).
    OpenAiModels {
        base_url: String,
        api_key: Option<String>,
    },
}

pub(crate) struct SdkBacked {
    provider: Arc<dyn sdk::Provider>,
    models: ModelList,
}

impl SdkBacked {
    pub(crate) fn new(provider: Arc<dyn sdk::Provider>, models: ModelList) -> Self {
        Self { provider, models }
    }
}

#[async_trait]
impl ModelProvider for SdkBacked {
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
        let request = to_sdk_request(model, messages, tools, options);
        let mut events = self.provider.stream(&request);
        while let Some(event) = events.next().await {
            super::streaming::note_stream_progress();
            match event {
                Ok(sdk::StreamEvent::Done(response)) => {
                    return Ok(to_provider_response(response, model))
                }
                Ok(_) => {}
                Err(error) => return Err(to_daemon_error(error).into()),
            }
        }
        Err(ProviderError::malformed_shape("provider stream ended without a response").into())
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        match &self.models {
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
