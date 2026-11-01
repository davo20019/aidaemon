//! The daemon's model providers as an SDK [`sdk::Provider`].
//!
//! The SDK run never talks to a backend itself. [`DaemonProvider`] hands each
//! request to the turn's dispatcher, which calls the daemon's provider stack
//! (router, billing-aware retries, fallback cascade, failover providers) on
//! the turn's own task. This module also translates between the SDK message
//! model and the OpenAI-style JSON the daemon providers speak.

use std::collections::HashMap;

use aidaemon_core as sdk;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use super::sdk_engine::DaemonCall;
use crate::traits::{ChatOptions, ProviderResponse, TokenUsage, ToolChoiceMode};

/// One model request handed from the SDK run to the daemon dispatcher.
pub(super) struct ModelCall {
    pub(super) request: sdk::ModelRequest,
    pub(super) reply: oneshot::Sender<Result<sdk::ModelResponse, sdk::ProviderError>>,
}

/// The daemon's provider stack as the SDK sees it. It only forwards the
/// request; the dispatcher makes the call. Retries and fallbacks already
/// happen there, so this must not be wrapped in [`sdk::Retry`].
pub(super) struct DaemonProvider {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Provider for DaemonProvider {
    fn name(&self) -> &str {
        "daemon"
    }

    async fn complete(
        &self,
        request: &sdk::ModelRequest,
    ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
        let (reply, answer) = oneshot::channel();
        self.calls
            .send(DaemonCall::Model(ModelCall {
                request: request.clone(),
                reply,
            }))
            .await
            .map_err(|_| sdk::ProviderError::Other("model dispatcher stopped".into()))?;
        answer
            .await
            .map_err(|_| sdk::ProviderError::Other("model dispatcher dropped the call".into()))?
    }
}

fn image_block(image: &sdk::Image) -> Value {
    json!({"type": "image_url", "image_url": {"url": image.to_url()}})
}

/// Text plus images as OpenAI-style content: a plain string without images,
/// content parts with them.
fn content_with_images(text: &str, images: &[sdk::Image]) -> Value {
    if images.is_empty() {
        return Value::String(text.to_string());
    }
    let mut blocks = Vec::with_capacity(images.len() + 1);
    if !text.trim().is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    blocks.extend(images.iter().map(image_block));
    Value::Array(blocks)
}

/// The request's transcript as the daemon providers expect it.
///
/// Images a tool observed cannot ride on a `tool` message for every backend,
/// so, as in the legacy loop, they follow the contiguous block of tool
/// results as a user message.
pub(super) fn to_daemon_messages(request: &sdk::ModelRequest) -> Vec<Value> {
    let mut names: HashMap<&str, &str> = HashMap::new();
    let mut out = Vec::with_capacity(request.messages.len());
    let mut pending_images: Vec<Value> = Vec::new();
    for message in &request.messages {
        if message.role != sdk::Role::Tool && !pending_images.is_empty() {
            out.append(&mut pending_images);
        }
        match message.role {
            sdk::Role::System => out.push(json!({"role": "system", "content": message.content})),
            sdk::Role::User => out.push(json!({
                "role": "user",
                "content": content_with_images(&message.content, &message.images),
            })),
            sdk::Role::Assistant => {
                let mut value = json!({"role": "assistant", "content": message.content});
                if !message.tool_calls.is_empty() {
                    let calls: Vec<Value> = message
                        .tool_calls
                        .iter()
                        .map(|call| {
                            names.insert(call.id.as_str(), call.name.as_str());
                            json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments.to_string(),
                                },
                            })
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
                        "content": content_with_images(&label, &message.images),
                    }));
                }
            }
        }
    }
    out.append(&mut pending_images);
    out
}

/// SDK tool specs as the daemon's OpenAI-style tool definitions.
pub(super) fn to_daemon_tools(specs: &[sdk::ToolSpec]) -> Vec<Value> {
    specs
        .iter()
        .map(|spec| {
            json!({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            })
        })
        .collect()
}

pub(super) fn to_chat_options(
    choice: &sdk::ToolChoice,
    max_output_tokens: Option<u32>,
) -> ChatOptions {
    ChatOptions {
        tool_choice: match choice {
            sdk::ToolChoice::Auto => ToolChoiceMode::Auto,
            sdk::ToolChoice::None => ToolChoiceMode::None,
            sdk::ToolChoice::Required => ToolChoiceMode::Required,
            sdk::ToolChoice::Specific(name) => ToolChoiceMode::Specific(name.clone()),
        },
        max_tokens_override: max_output_tokens,
        ..ChatOptions::default()
    }
}

pub(super) fn sdk_usage(usage: Option<&TokenUsage>) -> sdk::Usage {
    usage
        .map(|usage| sdk::Usage {
            input_tokens: usage.input_tokens.into(),
            output_tokens: usage.output_tokens.into(),
            cached_input_tokens: usage.cached_input_tokens.unwrap_or_default().into(),
            cache_creation_tokens: usage.cache_creation_input_tokens.unwrap_or_default().into(),
        })
        .unwrap_or_default()
}

/// A daemon provider reply as an SDK response. Arguments that are not valid
/// JSON become `{}`: the tool then reports the missing fields to the model.
pub(super) fn from_provider_response(response: ProviderResponse) -> sdk::ModelResponse {
    let usage = sdk_usage(response.usage.as_ref());
    sdk::ModelResponse {
        text: response.content.unwrap_or_default(),
        tool_calls: response
            .tool_calls
            .into_iter()
            .map(|call| {
                let arguments = serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}));
                sdk::ToolCall::new(call.id, call.name, arguments)
            })
            .collect(),
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn daemon_messages_round_trip_tool_pairs() {
        let request = sdk::ModelRequest {
            model: "mock-model".into(),
            messages: vec![
                sdk::Message::system("be brief"),
                sdk::Message::user("read a.txt"),
                sdk::Message::assistant(
                    "",
                    vec![sdk::ToolCall::new(
                        "call_1",
                        "read_file",
                        json!({"path": "a.txt"}),
                    )],
                ),
                sdk::Message::tool_result("call_1", "file body"),
            ],
            tools: Vec::new(),
            tool_choice: sdk::ToolChoice::Auto,
            max_output_tokens: None,
        };
        let messages = to_daemon_messages(&request);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be brief");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[2]["tool_calls"][0]["type"], "function");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a.txt\"}"
        );
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call_1");
        assert_eq!(messages[3]["name"], "read_file");
        assert_eq!(messages[3]["content"], "file body");
    }

    #[test]
    fn tool_images_follow_the_tool_result_block() {
        let request = sdk::ModelRequest {
            model: "m".into(),
            messages: vec![
                sdk::Message::user("look"),
                sdk::Message::assistant(
                    "",
                    vec![
                        sdk::ToolCall::new("a", "read_file", json!({})),
                        sdk::ToolCall::new("b", "read_file", json!({})),
                    ],
                ),
                sdk::Message::tool_result("a", "image file")
                    .with_images([sdk::Image::base64("image/png", "QUJD")]),
                sdk::Message::tool_result("b", "text file"),
            ],
            tools: Vec::new(),
            tool_choice: sdk::ToolChoice::Auto,
            max_output_tokens: None,
        };
        let messages = to_daemon_messages(&request);
        let roles: Vec<_> = messages.iter().map(|m| m["role"].clone()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "tool", "user"]);
        let blocks = messages[4]["content"].as_array().unwrap();
        assert_eq!(blocks[1]["type"], "image_url");
        assert_eq!(blocks[1]["image_url"]["url"], "data:image/png;base64,QUJD");
    }

    #[test]
    fn user_images_become_multimodal_content() {
        let request = sdk::ModelRequest {
            model: "m".into(),
            messages: vec![sdk::Message::user("what is this")
                .with_images([sdk::Image::url("https://example.com/a.png")])],
            tools: Vec::new(),
            tool_choice: sdk::ToolChoice::Auto,
            max_output_tokens: None,
        };
        let messages = to_daemon_messages(&request);
        let blocks = messages[0]["content"].as_array().unwrap();
        assert_eq!(blocks[0], json!({"type": "text", "text": "what is this"}));
        assert_eq!(blocks[1]["image_url"]["url"], "https://example.com/a.png");
    }

    #[test]
    fn daemon_tools_are_function_objects() {
        let tools = to_daemon_tools(&[sdk::ToolSpec {
            name: "read_file".into(),
            description: "Read a file".into(),
            parameters: json!({"type": "object"}),
        }]);
        assert_eq!(
            tools,
            vec![json!({"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object"}
            }})]
        );
    }

    #[test]
    fn provider_response_maps_usage_and_bad_arguments() {
        let response = from_provider_response(ProviderResponse {
            content: Some("hi".into()),
            tool_calls: vec![crate::traits::ToolCall {
                id: "c1".into(),
                name: "terminal".into(),
                arguments: "not json".into(),
                extra_content: None,
            }],
            usage: Some(TokenUsage {
                input_tokens: 100,
                output_tokens: 7,
                cached_input_tokens: Some(40),
                cache_creation_input_tokens: Some(3),
                model: "m".into(),
                ..Default::default()
            }),
            thinking: None,
            response_note: None,
        });
        assert_eq!(response.text, "hi");
        assert_eq!(response.tool_calls[0].arguments, json!({}));
        assert_eq!(response.tool_calls[0].name, "terminal");
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 7);
        assert_eq!(response.usage.cached_input_tokens, 40);
        assert_eq!(response.usage.cache_creation_tokens, 3);
    }

    #[test]
    fn tool_choice_maps_to_chat_options() {
        let options = to_chat_options(&sdk::ToolChoice::Required, Some(512));
        assert_eq!(options.tool_choice, ToolChoiceMode::Required);
        assert_eq!(options.max_tokens_override, Some(512));
        let options = to_chat_options(&sdk::ToolChoice::Specific("x".into()), None);
        assert_eq!(options.tool_choice, ToolChoiceMode::Specific("x".into()));
    }
}
