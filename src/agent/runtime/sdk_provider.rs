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
use crate::memory::context_window::{
    estimate_multimodal_message_tokens, fit_messages_with_source_quotas,
};
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
#[cfg(test)]
pub(super) fn to_daemon_messages(request: &sdk::ModelRequest) -> Vec<Value> {
    to_daemon_messages_from(&request.messages)
}

pub(super) fn to_daemon_messages_from(messages: &[sdk::Message]) -> Vec<Value> {
    let mut names: HashMap<&str, &str> = HashMap::new();
    let mut out = Vec::with_capacity(messages.len());
    let mut pending_images: Vec<Value> = Vec::new();
    for message in messages {
        if message.role != sdk::Role::Tool && !pending_images.is_empty() {
            out.append(&mut pending_images);
        }
        match message.role {
            sdk::Role::System => out.push(json!({"role": "system", "content": message.content})),
            // Harness and stop-hook messages are directives, which the
            // daemon sends as system messages.
            sdk::Role::User if message.is_injected() => out.push(json!({
                "role": "system",
                "content": message.content,
            })),
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

/// The daemon-format transcript for a model call: the projected `prefix`
/// unchanged (so the provider's prompt cache keeps hitting), then this run's
/// own messages. If the run outgrows the budget, only the run part is
/// trimmed, with the legacy role-quota fitter, and unpaired tool calls and
/// results are removed; the newest messages survive.
/// Split system notes a hook injected after the kernel's leading system
/// messages (for example the memory note) from the kernel transcript.
/// `prefix_system_lead` is how many leading system messages the starting
/// transcript had; any more in `messages` were added for this request.
pub(super) fn split_injected_notes(
    messages: &[sdk::Message],
    prefix_system_lead: usize,
) -> (Vec<String>, Vec<sdk::Message>) {
    let lead = messages
        .iter()
        .position(|message| message.role != sdk::Role::System)
        .unwrap_or(messages.len());
    if lead <= prefix_system_lead {
        return (Vec::new(), messages.to_vec());
    }
    let notes = messages[prefix_system_lead..lead]
        .iter()
        .map(|message| message.content.clone())
        .collect();
    let mut rest = messages[..prefix_system_lead].to_vec();
    rest.extend_from_slice(&messages[lead..]);
    (notes, rest)
}

/// The wire prefix with injected `notes` placed right after the task
/// context tail (so the session-static core and the archived history keep
/// their cached bytes), or after the leading system messages without a tail.
pub(super) fn prefix_with_notes(prefix: &[Value], notes: &[String]) -> Vec<Value> {
    let mut out = prefix.to_vec();
    if notes.is_empty() {
        return out;
    }
    let is_system = |message: &Value| message.get("role").and_then(Value::as_str) == Some("system");
    let at = out
        .iter()
        .position(|message| {
            is_system(message)
                && message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| {
                        content
                            .starts_with(crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER)
                    })
        })
        .map(|tail| tail + 1)
        .unwrap_or_else(|| out.iter().position(|m| !is_system(m)).unwrap_or(out.len()));
    for (offset, note) in notes.iter().enumerate() {
        out.insert(
            at + offset,
            serde_json::json!({"role": "system", "content": note}),
        );
    }
    out
}

pub(super) fn fit_run(prefix: &[Value], run: &[sdk::Message], budget_tokens: usize) -> Vec<Value> {
    let run_values = to_daemon_messages_from(run);
    let prefix_tokens = estimate_multimodal_message_tokens(prefix);
    let run_tokens = estimate_multimodal_message_tokens(&run_values);
    let mut out = prefix.to_vec();
    if prefix_tokens + run_tokens <= budget_tokens || run_values.len() <= 1 {
        out.extend(run_values);
        return out;
    }
    let (fitted, _) =
        fit_messages_with_source_quotas(run_values, budget_tokens.saturating_sub(prefix_tokens));
    out.extend(repair_tool_pairs(fitted));
    out
}

/// Daemon-format messages as SDK messages, for the kernel's own view of the
/// starting transcript. Text parts are joined and image parts kept; other
/// media stay only in the daemon-format prefix that is actually sent.
pub(super) fn values_to_sdk(values: &[Value]) -> Vec<sdk::Message> {
    values
        .iter()
        .filter_map(|value| {
            let (text, images) = split_content(value.get("content"));
            let message = match value.get("role").and_then(Value::as_str)? {
                "system" => sdk::Message::system(text),
                "user" => sdk::Message::user(text).with_images(images),
                "assistant" => {
                    let calls = value
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .map(|calls| {
                            calls
                                .iter()
                                .filter_map(|call| {
                                    let function = call.get("function")?;
                                    let arguments = function
                                        .get("arguments")
                                        .and_then(Value::as_str)
                                        .and_then(|raw| serde_json::from_str(raw).ok())
                                        .unwrap_or_else(|| json!({}));
                                    Some(sdk::ToolCall::new(
                                        call.get("id")?.as_str()?,
                                        function.get("name")?.as_str()?,
                                        arguments,
                                    ))
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    sdk::Message::assistant(text, calls)
                }
                "tool" => sdk::Message::tool_result(
                    value.get("tool_call_id").and_then(Value::as_str)?,
                    text,
                ),
                _ => return None,
            };
            Some(message)
        })
        .collect()
}

fn split_content(content: Option<&Value>) -> (String, Vec<sdk::Image>) {
    match content {
        Some(Value::String(text)) => (text.clone(), Vec::new()),
        Some(Value::Array(parts)) => {
            let mut text = Vec::new();
            let mut images = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(value) = part.get("text").and_then(Value::as_str) {
                            text.push(value.to_string());
                        }
                    }
                    Some("image_url") => {
                        if let Some(url) = part
                            .get("image_url")
                            .and_then(|image| image.get("url"))
                            .and_then(Value::as_str)
                        {
                            images.push(sdk::Image::url(url));
                        }
                    }
                    _ => {}
                }
            }
            (text.join("\n"), images)
        }
        _ => (String::new(), Vec::new()),
    }
}

/// Drop tool results whose call is not in the assistant message before
/// them, and tool calls whose result is not directly after them.
fn repair_tool_pairs(messages: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut index = 0;
    while index < messages.len() {
        let mut message = messages[index].clone();
        index += 1;
        if message["role"] == "tool" {
            // Only reached for results without a preceding call.
            continue;
        }
        let Some(calls) = message.get("tool_calls").and_then(Value::as_array).cloned() else {
            out.push(message);
            continue;
        };
        let mut results = Vec::new();
        while index < messages.len() && messages[index]["role"] == "tool" {
            results.push(messages[index].clone());
            index += 1;
        }
        let answered: Vec<Value> = calls
            .into_iter()
            .filter(|call| results.iter().any(|r| r["tool_call_id"] == call["id"]))
            .collect();
        results.retain(|r| answered.iter().any(|call| call["id"] == r["tool_call_id"]));
        if answered.is_empty() {
            message
                .as_object_mut()
                .map(|object| object.remove("tool_calls"));
            let empty = message["content"]
                .as_str()
                .is_none_or(|text| text.trim().is_empty());
            if !empty {
                out.push(message);
            }
            continue;
        }
        message["tool_calls"] = Value::Array(answered);
        out.push(message);
        out.extend(results);
    }
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

    fn long(text: &str) -> String {
        format!("{text} {}", "lorem ipsum dolor sit amet ".repeat(60))
    }

    fn assert_tool_pairs_valid(messages: &[Value]) {
        for (index, message) in messages.iter().enumerate() {
            if message["role"] != "tool" {
                continue;
            }
            let call_id = message["tool_call_id"].as_str().unwrap();
            let owner = messages[..index]
                .iter()
                .rev()
                .find(|m| m["role"] != "tool")
                .expect("tool result without a preceding message");
            let calls = owner["tool_calls"]
                .as_array()
                .expect("result after non-call");
            assert!(calls.iter().any(|call| call["id"] == call_id));
        }
        for (index, message) in messages.iter().enumerate() {
            let Some(calls) = message["tool_calls"].as_array() else {
                continue;
            };
            for call in calls {
                assert!(
                    messages[index + 1..]
                        .iter()
                        .take_while(|m| m["role"] == "tool")
                        .any(|m| m["tool_call_id"] == call["id"]),
                    "tool call {} has no result",
                    call["id"]
                );
            }
        }
    }

    fn prefix() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "core prompt"}),
            json!({"role": "user", "content": "an earlier question"}),
            json!({"role": "assistant", "content": "an earlier answer"}),
            json!({"role": "system", "content": "volatile tail"}),
            json!({"role": "user", "content": [
                {"type": "text", "text": "current question"},
                {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "ogg"}}
            ]}),
        ]
    }

    #[test]
    fn values_become_sdk_messages_with_paired_tool_calls() {
        let values = vec![
            json!({"role": "system", "content": "core"}),
            json!({"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
            ]}),
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}}
            ]}),
            json!({"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "body"}),
        ];
        let messages = values_to_sdk(&values);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, sdk::Role::System);
        assert_eq!(messages[1].content, "look");
        assert_eq!(messages[1].images.len(), 1);
        assert_eq!(messages[2].tool_calls[0].arguments, json!({"path": "a"}));
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(messages[3].content, "body");
    }

    #[test]
    fn the_projected_prefix_is_sent_byte_for_byte() {
        let run = vec![sdk::Message::assistant("done", Vec::new())];
        let fitted = fit_run(&prefix(), &run, 1_000_000);
        assert_eq!(fitted[..5], prefix()[..]);
        assert_eq!(fitted[5]["content"], "done");
    }

    #[test]
    fn an_oversized_run_is_trimmed_after_the_prefix_with_valid_pairs() {
        let mut run = Vec::new();
        for i in 0..20 {
            let id = format!("c{i}");
            run.push(sdk::Message::assistant(
                "",
                vec![sdk::ToolCall::new(id.clone(), "terminal", json!({"i": i}))],
            ));
            run.push(sdk::Message::tool_result(id, long(&format!("output {i}"))));
        }
        let fitted = fit_run(&prefix(), &run, 2_000);
        assert_eq!(fitted[..5], prefix()[..], "the prefix is never trimmed");
        assert_tool_pairs_valid(&fitted);
        assert!(fitted.len() < 5 + run.len());
        // The newest call survives.
        assert!(serde_json::to_string(&fitted)
            .unwrap()
            .contains("output 19"));
    }

    #[test]
    fn injected_directives_are_sent_as_system_messages() {
        let directive =
            sdk::Message::from_origin(sdk::MessageOrigin::StopHook, "[SYSTEM] answer now");
        let messages = to_daemon_messages_from(&[sdk::Message::user("hi"), directive]);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "system");
        assert_eq!(messages[1]["content"], "[SYSTEM] answer now");
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
