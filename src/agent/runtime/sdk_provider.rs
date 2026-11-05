//! The daemon's model providers as an SDK [`sdk::Provider`].
//!
//! The SDK run never talks to a backend itself. [`DaemonProvider`] hands each
//! request to the turn's dispatcher, which calls the daemon's provider stack
//! (router, billing-aware retries, fallback cascade, failover providers) on
//! the turn's own task. This module also translates between the SDK message
//! model and the OpenAI-style JSON the daemon providers speak.

use aidaemon_core as sdk;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use super::sdk_engine::DaemonCall;
use crate::memory::context_window::{
    estimate_multimodal_message_tokens, fit_messages_with_source_quotas,
};
use crate::traits::{ChatOptions, ProviderResponse};

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

/// The daemon's live provider stack as a standalone SDK provider, for SDK
/// components that call a model outside an agent run (memory extraction).
///
/// It recovers like a turn's model call, on a smaller budget: retryable
/// failures (rate limit, transient, malformed parse) are retried with
/// backoff, and a failure that routes to fallback, or exhausts its retries,
/// moves on to the primary model. Every successful call is accounted.
#[cfg_attr(not(feature = "sdk-memory"), allow(dead_code))]
pub(crate) struct RuntimeProvider {
    pub(crate) runtime: crate::llm_runtime::SharedLlmRuntime,
    pub(crate) accounting: Option<ModelCallAccounting>,
}

/// Where a [`RuntimeProvider`] records each call's tokens and `llm_call`
/// event, and the purpose they are filed under.
#[cfg_attr(not(feature = "sdk-memory"), allow(dead_code))]
pub(crate) struct ModelCallAccounting {
    pub(crate) state: std::sync::Arc<dyn crate::traits::StateStore>,
    pub(crate) event_store: std::sync::Arc<crate::events::EventStore>,
    /// The call purpose; the session is `background:<purpose>`.
    pub(crate) purpose: &'static str,
}

/// Attempts per model before moving to the next candidate.
#[cfg_attr(not(feature = "sdk-memory"), allow(dead_code))]
const RUNTIME_ATTEMPTS_PER_MODEL: u32 = 3;
/// Longest wait between attempts, whatever a rate limit asks for.
#[cfg_attr(not(feature = "sdk-memory"), allow(dead_code))]
const RUNTIME_MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(30);
/// How long one background attempt may take to start answering.
const RUNTIME_ATTEMPT_DEADLINE: std::time::Duration = std::time::Duration::from_secs(360);

/// Recovery for background calls: rate limits, transient errors and parse
/// failures retry with backoff, billing and not-found move to the next
/// model, anything else fails. Records the models it attempted.
#[derive(Default)]
struct BackgroundRecovery {
    attempted: std::sync::Mutex<Vec<String>>,
}

impl sdk::FailoverPolicy for BackgroundRecovery {
    fn before_attempt(&self, request: &sdk::ModelRequest) -> Result<(), sdk::ProviderError> {
        if let Ok(mut attempted) = self.attempted.lock() {
            attempted.push(request.model.clone());
        }
        Ok(())
    }

    fn after_error(
        &self,
        request: &sdk::ModelRequest,
        error: &sdk::ProviderError,
        attempt: u32,
        attempts: u32,
    ) -> sdk::FailoverAction {
        use crate::providers::ProviderRecoveryRoute as Route;
        let daemon_error = crate::providers::sdk_backed::to_daemon_error(error.clone());
        let action = match daemon_error.recovery_route(false) {
            Route::Fail | Route::AdaptRequestOnce | Route::RetryMalformedThenFail => {
                sdk::FailoverAction::Stop
            }
            Route::Fallback | Route::RetryReducedBudgetThenFallback => sdk::FailoverAction::Next,
            Route::RetryRateLimitThenFallback
            | Route::RetryTransientThenFallback
            | Route::RetryMalformedThenFallback => {
                if attempt >= attempts {
                    sdk::FailoverAction::Next
                } else {
                    let backoff = daemon_error
                        .retry_after_secs
                        .map(std::time::Duration::from_secs)
                        .unwrap_or_else(|| std::time::Duration::from_secs(1 << (attempt - 1)));
                    sdk::FailoverAction::Retry(backoff.min(RUNTIME_MAX_BACKOFF))
                }
            }
        };
        tracing::warn!(
            model = %request.model,
            attempt,
            %error,
            next = ?action,
            "background model call failed"
        );
        action
    }
}

/// Shares one [`BackgroundRecovery`] with the caller.
struct SharedRecovery(std::sync::Arc<BackgroundRecovery>);

impl sdk::FailoverPolicy for SharedRecovery {
    fn before_attempt(&self, request: &sdk::ModelRequest) -> Result<(), sdk::ProviderError> {
        self.0.before_attempt(request)
    }

    fn after_error(
        &self,
        request: &sdk::ModelRequest,
        error: &sdk::ProviderError,
        attempt: u32,
        attempts: u32,
    ) -> sdk::FailoverAction {
        self.0.after_error(request, error, attempt, attempts)
    }
}

#[async_trait]
impl sdk::Provider for RuntimeProvider {
    fn name(&self) -> &str {
        "daemon-runtime"
    }

    /// The requested model, then the primary model, on the runtime's
    /// provider, each with retries; every success is accounted.
    async fn complete(
        &self,
        request: &sdk::ModelRequest,
    ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
        let snapshot = self.runtime.snapshot();
        let provider = snapshot.provider();
        let primary = snapshot.primary_model();
        let step = |model: &str| {
            sdk::FailoverStep::from_arc(provider.clone())
                .model(model)
                .deadline(RUNTIME_ATTEMPT_DEADLINE)
                .attempts(RUNTIME_ATTEMPTS_PER_MODEL)
        };
        let mut steps = vec![step(&request.model)];
        if primary != request.model {
            steps.push(step(&primary));
        }
        let recovery = std::sync::Arc::new(BackgroundRecovery::default());
        let chain = sdk::Failover::new(steps).policy(SharedRecovery(recovery.clone()));
        let started = std::time::Instant::now();
        let response = chain.complete(request).await?;
        if let Some(accounting) = &self.accounting {
            let model = recovery
                .attempted
                .lock()
                .ok()
                .and_then(|models| models.last().cloned())
                .unwrap_or_else(|| request.model.clone());
            crate::events::record_background_model_call_telemetry(
                accounting.event_store.clone(),
                accounting.state.as_ref(),
                &format!("background:{}", accounting.purpose),
                accounting.purpose,
                &model,
                &crate::providers::sdk_backed::to_provider_response(response.clone(), &model),
                started.elapsed(),
            )
            .await;
        }
        Ok(response)
    }
}

/// The request's transcript as the daemon providers expect it.
///
/// Images a tool observed cannot ride on a `tool` message for every backend,
/// so they follow the contiguous block of tool results as a user message.
#[cfg(test)]
pub(super) fn to_daemon_messages(request: &sdk::ModelRequest) -> Vec<Value> {
    to_daemon_messages_from(&request.messages)
}

pub(super) fn to_daemon_messages_from(messages: &[sdk::Message]) -> Vec<Value> {
    crate::providers::sdk_backed::to_chat_messages(messages)
}

/// The daemon-format transcript for a model call: the projected `prefix`
/// unchanged (so the provider's prompt cache keeps hitting), then this run's
/// own messages. If the run outgrows the budget, only the run part is
/// trimmed, with the daemon's role-quota fitter, and unpaired tool calls and
/// results are removed; the newest messages survive.
/// Split the system notes hooks added to this request (for example the
/// memory note, marked [`sdk::MessageOrigin::HookNote`]) from the kernel
/// transcript.
pub(super) fn split_injected_notes(messages: &[sdk::Message]) -> (Vec<String>, Vec<sdk::Message>) {
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

#[cfg(test)]
pub(super) fn fit_run(prefix: &[Value], run: &[sdk::Message], budget_tokens: usize) -> Vec<Value> {
    fit_run_values(prefix, to_daemon_messages_from(run), budget_tokens)
}

/// [`fit_run`] over a run already in the daemon wire format (after
/// compaction).
pub(super) fn fit_run_values(
    prefix: &[Value],
    run_values: Vec<Value>,
    budget_tokens: usize,
) -> Vec<Value> {
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
pub(super) fn message_text(message: &Value) -> String {
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
    crate::providers::sdk_backed::to_chat_tools(specs)
}

pub(super) fn to_chat_options(request: &sdk::ModelRequest) -> ChatOptions {
    crate::providers::sdk_backed::to_chat_options(request)
}

/// A daemon provider reply as an SDK response.
pub(super) fn from_provider_response(response: ProviderResponse) -> sdk::ModelResponse {
    crate::providers::sdk_backed::to_sdk_response(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{TokenUsage, ToolChoiceMode};
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
    fn runtime_provider_recovery_follows_the_typed_route() {
        use sdk::{FailoverAction, FailoverPolicy, ProviderError};
        use std::time::Duration;

        let policy = BackgroundRecovery::default();
        let request = sdk::ModelRequest {
            model: "fast-model".into(),
            ..Default::default()
        };
        let after = |error: ProviderError, attempt| {
            policy.after_error(&request, &error, attempt, RUNTIME_ATTEMPTS_PER_MODEL)
        };
        assert_eq!(
            after(ProviderError::Transient("request timed out: x".into()), 2),
            FailoverAction::Retry(Duration::from_secs(2))
        );
        assert_eq!(
            after(
                ProviderError::RateLimited {
                    retry_after: Some(Duration::from_secs(600))
                },
                1
            ),
            FailoverAction::Retry(Duration::from_secs(30))
        );
        assert_eq!(
            after(
                ProviderError::Transient("HTTP 503: down".into()),
                RUNTIME_ATTEMPTS_PER_MODEL
            ),
            FailoverAction::Next
        );
        assert_eq!(
            after(ProviderError::NotFound("HTTP 404".into()), 1),
            FailoverAction::Next
        );
        assert_eq!(
            after(ProviderError::Auth("HTTP 401".into()), 1),
            FailoverAction::Stop
        );
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
            messages: vec![sdk::Message::user("what is this")
                .with_images([sdk::Image::url("https://example.com/a.png")])],
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
            tool_calls: vec![crate::traits::ToolCall {
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
            },
            ..Default::default()
        };
        let options = to_chat_options(&request);
        assert_eq!(options.tool_choice, ToolChoiceMode::Required);
        assert_eq!(options.max_tokens_override, Some(512));
        assert_eq!(
            options.response_mode,
            crate::traits::ResponseMode::JsonObject
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
