//! The host's model providers as an SDK [`sdk::Provider`].
//!
//! The SDK run never talks to a backend itself. [`HostProvider`] hands each
//! request to the turn's dispatcher, which calls the host's provider stack
//! (router, billing-aware retries, fallback cascade, failover providers) on
//! the turn's own task. This module also translates between the SDK message
//! model and the OpenAI-style JSON those providers speak.

use aidaemon_core as sdk;
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

use super::calls::{HostCall, ModelCall};
use crate::context_budget::estimate_multimodal_message_tokens;
use aidaemon_events::traits::{ChatOptions, ProviderResponse};

/// The daemon's provider stack as the SDK sees it. It only forwards the
/// request; the dispatcher makes the call. Retries and fallbacks already
/// happen there, so this must not be wrapped in [`sdk::Retry`].
pub struct HostProvider {
    pub calls: mpsc::Sender<HostCall>,
}

#[async_trait]
impl sdk::Provider for HostProvider {
    fn name(&self) -> &str {
        "daemon"
    }

    async fn complete(
        &self,
        request: &sdk::ModelRequest,
    ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
        let (reply, answer) = oneshot::channel();
        self.calls
            .send(HostCall::Model(ModelCall {
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

/// The request's transcript as the daemon providers expect it.
///
/// Images a tool observed cannot ride on a `tool` message for every backend,
/// so they follow the contiguous block of tool results as a user message.
pub fn to_daemon_messages(request: &sdk::ModelRequest) -> Vec<Value> {
    to_daemon_messages_from(&request.messages)
}

pub fn to_daemon_messages_from(messages: &[sdk::Message]) -> Vec<Value> {
    crate::chat_bridge::sdk_backed::to_chat_messages(messages)
}

/// Split the system notes hooks added to this request (for example the
/// memory note, marked [`sdk::MessageOrigin::HookNote`]) from the kernel
/// transcript.
pub fn split_injected_notes(messages: &[sdk::Message]) -> (Vec<String>, Vec<sdk::Message>) {
    let (notes, rest): (Vec<_>, Vec<_>) = messages
        .iter()
        .cloned()
        .partition(sdk::Message::is_hook_note);
    (
        notes.into_iter().map(|message| message.content).collect(),
        rest,
    )
}

/// The wire prefix with injected `notes` placed right after the task
/// context tail (so the session-static core and the archived history keep
/// their cached bytes), or after the leading system messages without a tail.
pub fn prefix_with_notes(prefix: &[Value], notes: &[String]) -> Vec<Value> {
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
                        content.starts_with(crate::prompt::fingerprint::TASK_CONTEXT_TAIL_MARKER)
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

/// One message of a model call's run: the kernel's own, or one the daemon
/// pinned at a run position (a directive, a user message taken mid-turn),
/// sent as it is.
#[derive(Debug, Clone)]
pub enum RunMessage {
    Kernel(sdk::Message),
    Pinned(Value),
}

/// A model call's wire transcript and its estimated size.
#[derive(Debug, Clone)]
pub struct FittedRequest {
    pub messages: Vec<Value>,
    /// The request's size in `window`'s raw estimate (characters / 4,
    /// the prefix, the tool definitions and the run as sent): what the
    /// provider's reported input tokens calibrate.
    pub estimated_tokens: usize,
}

/// The daemon-format transcript for a model call, budgeted by the policies
/// crate's [`aidaemon_policies::ContextWindow`]: the projected `prefix`
/// unchanged (so the provider's prompt cache keeps hitting) and the tool
/// definitions are counted against the budget, then this run's messages
/// follow, with old tool results stubbed and the oldest rounds dropped in
/// coarse, stable steps when the request would pass the budget. Pinned
/// messages are never trimmed; a tool call keeps its results.
pub fn fit_request(
    prefix: &[Value],
    run: Vec<RunMessage>,
    tools: &[sdk::ToolSpec],
    window: &aidaemon_policies::ContextWindow,
) -> FittedRequest {
    let overhead =
        estimate_multimodal_message_tokens(prefix) * 4 + aidaemon_policies::tool_spec_chars(tools);
    let mut view: Vec<sdk::Message> = run
        .iter()
        .map(|message| match message {
            RunMessage::Kernel(message) => message.clone(),
            RunMessage::Pinned(value) => pinned_stand_in(value),
        })
        .collect();
    let kept = window.compact_with_overhead(&mut view, overhead);
    let estimated_tokens = aidaemon_policies::estimate_tokens(&view) + overhead.div_ceil(4);
    let mut messages = prefix.to_vec();
    // Kernel messages convert together (a tool's images follow its block of
    // results); pinned ones sit between rounds, never inside one.
    let mut segment = Vec::new();
    let mut shown = view.into_iter();
    for (message, kept) in run.into_iter().zip(kept) {
        if !kept {
            continue;
        }
        let Some(sent) = shown.next() else { break };
        match message {
            RunMessage::Kernel(_) => segment.push(sent),
            RunMessage::Pinned(value) => {
                messages.extend(to_daemon_messages_from(&std::mem::take(&mut segment)));
                messages.push(value);
            }
        }
    }
    messages.extend(to_daemon_messages_from(&segment));
    FittedRequest {
        messages,
        estimated_tokens,
    }
}

/// What a pinned message counts for in the budget. A system message or an
/// injected user message is never elided or dropped.
fn pinned_stand_in(value: &Value) -> sdk::Message {
    let (text, images) = split_content(value.get("content"));
    match value.get("role").and_then(Value::as_str) {
        Some("user") => {
            sdk::Message::from_origin(sdk::MessageOrigin::Harness, text).with_images(images)
        }
        _ => sdk::Message::system(text),
    }
}

/// Daemon-format messages as SDK messages, for the kernel's own view of the
/// starting transcript. Text parts are joined and image parts kept; other
/// media stay only in the daemon-format prefix that is actually sent.
pub fn values_to_sdk(values: &[Value]) -> Vec<sdk::Message> {
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
                                    let mut sdk_call = sdk::ToolCall::new(
                                        call.get("id")?.as_str()?,
                                        function.get("name")?.as_str()?,
                                        arguments,
                                    );
                                    sdk_call.provider_data = call
                                        .get("extra_content")
                                        .filter(|data| !data.is_null())
                                        .cloned();
                                    Some(sdk_call)
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

/// A wire message's text, image parts left out.
pub fn message_text(message: &Value) -> String {
    split_content(message.get("content")).0
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

/// SDK tool specs as the daemon's OpenAI-style tool definitions.
pub fn to_daemon_tools(specs: &[sdk::ToolSpec]) -> Vec<Value> {
    crate::chat_bridge::sdk_backed::to_chat_tools(specs)
}

pub fn to_chat_options(request: &sdk::ModelRequest) -> ChatOptions {
    crate::chat_bridge::sdk_backed::to_chat_options(request)
}

/// A daemon provider reply as an SDK response.
pub fn from_provider_response(response: ProviderResponse) -> sdk::ModelResponse {
    crate::chat_bridge::sdk_backed::to_sdk_response(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_events::traits::{TokenUsage, ToolChoiceMode};
    use serde_json::json;

    /// Hook notes are found by their origin, wherever they sit; a plain
    /// system message stays in the transcript.
    #[test]
    fn split_injected_notes_follows_the_hook_note_origin() {
        let messages = vec![
            sdk::Message::system("core"),
            sdk::Message::hook_note("recalled memory"),
            sdk::Message::system("task tail"),
            sdk::Message::user("hi"),
            sdk::Message::hook_note("late note"),
        ];
        let (notes, rest) = split_injected_notes(&messages);
        assert_eq!(notes, vec!["recalled memory", "late note"]);
        let contents: Vec<&str> = rest.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(contents, vec!["core", "task tail", "hi"]);
    }

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
            ..Default::default()
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
            ..Default::default()
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
            messages: vec![
                sdk::Message::user("what is this")
                    .with_images([sdk::Image::url("https://example.com/a.png")]),
            ],
            tools: Vec::new(),
            tool_choice: sdk::ToolChoice::Auto,
            max_output_tokens: None,
            ..Default::default()
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

    /// A reasoning signature the provider returned on a tool call is sent
    /// back with that call on the run's next request, and the reply's
    /// reasoning and note reach the SDK response.
    #[test]
    fn provider_data_round_trips_through_the_run_transcript() {
        let signature = json!({"thought_signature": "sig-synthetic"});
        let response = from_provider_response(ProviderResponse {
            content: Some(String::new()),
            tool_calls: vec![aidaemon_events::traits::ToolCall {
                id: "c1".into(),
                name: "terminal".into(),
                arguments: r#"{"command":"ls"}"#.into(),
                extra_content: Some(signature.clone()),
            }],
            usage: None,
            thinking: Some("considered the listing".into()),
            response_note: Some("truncated".into()),
        });
        assert_eq!(
            response.reasoning.as_deref(),
            Some("considered the listing")
        );
        assert_eq!(response.note.as_deref(), Some("truncated"));
        assert_eq!(
            response.tool_calls[0].provider_data,
            Some(signature.clone())
        );

        let replayed =
            to_daemon_messages_from(&[sdk::Message::assistant("", response.tool_calls.clone())]);
        assert_eq!(replayed[0]["tool_calls"][0]["extra_content"], signature);
        let back = values_to_sdk(&replayed);
        assert_eq!(back[0].tool_calls[0].provider_data, Some(signature));
    }

    #[test]
    fn provider_response_maps_usage_and_bad_arguments() {
        let response = from_provider_response(ProviderResponse {
            content: Some("hi".into()),
            tool_calls: vec![aidaemon_events::traits::ToolCall {
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
        // Kept verbatim in the SDK's `_raw` form: the tool reports the bad
        // arguments, and the call replays byte for byte.
        assert_eq!(
            response.tool_calls[0].arguments,
            json!({"_raw": "not json"})
        );
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

    fn kernel(messages: Vec<sdk::Message>) -> Vec<RunMessage> {
        messages.into_iter().map(RunMessage::Kernel).collect()
    }

    fn window(tokens: usize) -> aidaemon_policies::ContextWindow {
        aidaemon_policies::ContextWindow::new(tokens)
    }

    #[test]
    fn the_projected_prefix_is_sent_byte_for_byte() {
        let run = kernel(vec![sdk::Message::assistant("done", Vec::new())]);
        let fitted = fit_request(&prefix(), run, &[], &window(1_000_000));
        assert_eq!(fitted.messages[..5], prefix()[..]);
        assert_eq!(fitted.messages[5]["content"], "done");
        assert!(fitted.estimated_tokens > 0);
    }

    fn rounds(n: usize) -> Vec<sdk::Message> {
        let mut run = Vec::new();
        for i in 0..n {
            let id = format!("c{i}");
            run.push(sdk::Message::assistant(
                "",
                vec![sdk::ToolCall::new(id.clone(), "terminal", json!({"i": i}))],
            ));
            run.push(sdk::Message::tool_result(id, long(&format!("output {i}"))));
        }
        run
    }

    /// The prefix, the tool definitions and the run are all counted: the
    /// request stays under the budget, the prefix untouched, every tool call
    /// with its result, the newest round whole and older ones stubbed.
    #[test]
    fn an_oversized_run_is_fitted_after_the_prefix_with_valid_pairs() {
        let tools = vec![sdk::ToolSpec {
            name: "terminal".into(),
            description: "d".repeat(2_000),
            parameters: json!({"type": "object"}),
        }];
        let fitted = fit_request(&prefix(), kernel(rounds(20)), &tools, &window(3_000));
        assert_eq!(
            fitted.messages[..5],
            prefix()[..],
            "the prefix is never trimmed"
        );
        assert_tool_pairs_valid(&fitted.messages);
        assert!(
            fitted.estimated_tokens <= 3_000,
            "{}",
            fitted.estimated_tokens
        );
        let sent = serde_json::to_string(&fitted.messages).unwrap();
        assert!(sent.contains("output 19"), "the newest result is whole");
        assert!(
            sent.contains("[earlier tool result elided"),
            "older ones stubbed"
        );
    }

    /// Directives and mid-turn user messages are sent as they were pinned,
    /// at their run position, however the rounds around them are fitted.
    #[test]
    fn pinned_messages_survive_fitting_in_place() {
        let mut run = kernel(rounds(10));
        run.insert(
            10,
            RunMessage::Pinned(json!({"role": "system", "content": "[SYSTEM] directive"})),
        );
        run.push(RunMessage::Pinned(json!({"role": "user", "content": [
            {"type": "text", "text": "also check b"},
            {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "ogg"}}
        ]})));
        let fitted = fit_request(&prefix(), run, &[], &window(2_500));
        let messages = &fitted.messages;
        assert_tool_pairs_valid(messages);
        let directive = messages
            .iter()
            .position(|m| m["content"] == "[SYSTEM] directive")
            .expect("directive kept");
        // Still right after round 4's result.
        assert_eq!(messages[directive - 1]["tool_call_id"], "c4");
        let last = messages.last().unwrap();
        assert_eq!(last["role"], "user");
        assert_eq!(last["content"][1]["type"], "input_audio", "sent as pinned");
    }

    /// The 2c753f97 context_tally failure: one turn reads ten ~1.9k-token
    /// shards at once into a 10k window. The old quota fitter never cut
    /// below seven messages and sent ~11.6k tokens; the request now stays
    /// under the window, and each result it leaves out says it was not
    /// shown and how to get it.
    #[test]
    fn a_turn_of_reads_bigger_than_the_window_is_sent_under_it() {
        let mut config = crate::settings::ContextWindowConfig::default();
        let model = "synthetic-parallel-reads-model";
        config.model_budgets.insert(model.into(), 10_000);
        let window = crate::context_budget::request_window(model, &config);
        let calls = (0..10)
            .map(|i| sdk::ToolCall::new(format!("s{i}"), "read_shard", json!({"path": i})))
            .collect();
        let mut run = vec![sdk::Message::assistant("", calls)];
        for i in 0..10 {
            run.push(sdk::Message::tool_result(
                format!("s{i}"),
                format!(
                    "shard {i}\n{}\nfailed_requests=9\nEnd of shard {i}",
                    "x".repeat(7_600)
                ),
            ));
        }
        let prefix = vec![
            json!({"role": "system", "content": "core prompt"}),
            json!({"role": "user", "content": "Sum the failures in the ten shards."}),
        ];
        let fitted = fit_request(&prefix, kernel(run), &[], &window);
        assert!(fitted.estimated_tokens <= window.budget_tokens());
        let sent = values_to_sdk(&fitted.messages);
        assert!(aidaemon_policies::estimate_tokens(&sent) <= 10_000);
        assert_tool_pairs_valid(&fitted.messages);
        let unshown = sent
            .iter()
            .filter(|m| m.content.starts_with("[tool result not shown"))
            .count();
        assert!(unshown > 0 && unshown < 10, "{unshown}");
        assert!(
            sent.iter()
                .all(|m| !m.content.starts_with("[earlier tool result elided"))
        );
    }

    /// A tool's images still follow its block of results.
    #[test]
    fn tool_images_follow_their_results_after_fitting() {
        let run = kernel(vec![
            sdk::Message::assistant("", vec![sdk::ToolCall::new("a", "read_file", json!({}))]),
            sdk::Message::tool_result("a", "image file")
                .with_images([sdk::Image::base64("image/png", "QUJD")]),
        ]);
        let fitted = fit_request(&[], run, &[], &window(100_000));
        let roles: Vec<_> = fitted.messages.iter().map(|m| m["role"].clone()).collect();
        assert_eq!(roles, ["assistant", "tool", "user"]);
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

    /// Everything a kernel request asks for reaches the daemon call path:
    /// tool choice, output cap, response format, effort, single attempt.
    #[test]
    fn request_options_map_to_chat_options() {
        let request = sdk::ModelRequest {
            tool_choice: sdk::ToolChoice::Required,
            max_output_tokens: Some(512),
            options: sdk::RequestOptions {
                response_format: sdk::ResponseFormat::JsonObject,
                reasoning_effort: Some("low".into()),
                single_attempt: true,
                fallback_when_unspent: false,
                cache_key: None,
            },
            ..Default::default()
        };
        let options = to_chat_options(&request);
        assert_eq!(options.tool_choice, ToolChoiceMode::Required);
        assert_eq!(options.max_tokens_override, Some(512));
        assert_eq!(
            options.response_mode,
            aidaemon_events::traits::ResponseMode::JsonObject
        );
        assert_eq!(options.reasoning_effort_override.as_deref(), Some("low"));
        assert!(options.single_attempt_fail_closed);
        let request = sdk::ModelRequest {
            tool_choice: sdk::ToolChoice::Specific("x".into()),
            ..Default::default()
        };
        assert_eq!(
            to_chat_options(&request).tool_choice,
            ToolChoiceMode::Specific("x".into())
        );
    }
}
