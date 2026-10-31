//! The core provider conformance kit against the scripted test providers and
//! both HTTP providers (served by the local mock server; no network).

mod common;

use std::time::Duration;

use aidaemon_core::conformance::{check_provider, PROVIDER_CALLS_PER_REQUEST};
use aidaemon_core::testing::{call, text, tool_calls, ScriptedProvider, StreamingProvider};
use aidaemon_core::ModelResponse;
use aidaemon_providers::{Anthropic, OpenAiCompatible};
use common::{data_events, named_events, MockResponse, MockServer};
use serde_json::json;

/// The kit sends `plain` then `with_tool`, each [`PROVIDER_CALLS_PER_REQUEST`]
/// times; queue each reply that many times.
fn queue(plain: String, with_tool: String) -> Vec<MockResponse> {
    let mut responses = Vec::new();
    for body in [plain, with_tool] {
        for _ in 0..PROVIDER_CALLS_PER_REQUEST {
            responses.push(MockResponse::sse(body.clone(), 11));
        }
    }
    responses
}

/// A provider that answers the plain probe with text and the tool probe
/// with a tool call, deterministically.
struct ByRequest(ModelResponse, ModelResponse);

#[async_trait::async_trait]
impl aidaemon_core::Provider for ByRequest {
    fn name(&self) -> &str {
        "by-request"
    }
    async fn complete(
        &self,
        request: &aidaemon_core::ModelRequest,
    ) -> Result<ModelResponse, aidaemon_core::ProviderError> {
        Ok(if request.tools.is_empty() {
            self.0.clone()
        } else {
            self.1.clone()
        })
    }
}

fn scripted_pair() -> ByRequest {
    let mut with_tool = tool_calls([call("call_1", "get_time", json!({"city": "Springfield"}))]);
    with_tool.text = "Checking the time. ".into();
    ByRequest(text("Hello from the probe."), with_tool)
}

#[tokio::test]
async fn scripted_providers_pass_the_provider_kit() {
    check_provider(&ScriptedProvider::new([]).then_repeat(text("Hi there.")))
        .await
        .assert_all_passed();
    check_provider(&scripted_pair()).await.assert_all_passed();
    let report = check_provider(&StreamingProvider(scripted_pair())).await;
    report.assert_all_passed();
    // Tool calls were actually exercised on the tool probe.
    assert!(
        !report.notes.iter().any(|n| n.starts_with("with_tool:")),
        "{report}"
    );
}

fn anthropic_text() -> String {
    named_events(&[
        (
            "message_start",
            json!({"type": "message_start", "message": {
                "id": "msg_synthetic", "type": "message", "role": "assistant", "content": [],
                "model": "claude-test",
                "usage": {"input_tokens": 12, "output_tokens": 1, "cache_read_input_tokens": 40}
            }}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "there."}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ])
}

fn anthropic_tool() -> String {
    named_events(&[
        (
            "message_start",
            json!({"type": "message_start", "message": {
                "id": "msg_synthetic", "type": "message", "role": "assistant", "content": [],
                "model": "claude-test", "usage": {"input_tokens": 30, "output_tokens": 1}
            }}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me check."}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {}}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": "}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Springfield\"}"}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 1}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ])
}

#[tokio::test]
async fn anthropic_passes_the_provider_kit() {
    let server = MockServer::start(queue(anthropic_text(), anthropic_tool())).await;
    let provider = Anthropic::builder()
        .api_key("sk-ant-synthetic")
        .base_url(&server.base)
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let report = check_provider(&provider).await;
    report.assert_all_passed();
    assert!(
        !report.notes.iter().any(|n| n.starts_with("with_tool:")),
        "{report}"
    );
    assert_eq!(server.requests().len(), 2 * PROVIDER_CALLS_PER_REQUEST);
}

fn openai_text() -> String {
    let events = vec![
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello "}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {"content": "there."}, "finish_reason": null}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 20, "completion_tokens": 3,
                         "prompt_tokens_details": {"cached_tokens": 8}}}),
    ];
    format!("{}data: [DONE]\n\n", data_events(&events))
}

fn openai_tool() -> String {
    let events = vec![
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": null, "tool_calls": [
            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_time", "arguments": ""}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"city\": \"Spring"}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "field\"}"}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    format!("{}data: [DONE]\n\n", data_events(&events))
}

#[tokio::test]
async fn openai_compatible_passes_the_provider_kit() {
    let server = MockServer::start(queue(openai_text(), openai_tool())).await;
    let provider = OpenAiCompatible::builder()
        .base_url(format!("{}/v1/", server.base))
        .api_key("sk-synthetic-test")
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let report = check_provider(&provider).await;
    report.assert_all_passed();
    assert!(
        !report.notes.iter().any(|n| n.starts_with("with_tool:")),
        "{report}"
    );
    assert_eq!(server.requests().len(), 2 * PROVIDER_CALLS_PER_REQUEST);
}

#[tokio::test]
async fn provider_errors_are_reported_not_panicked() {
    // An unscripted mock answers 500 to everything.
    let server = MockServer::start(Vec::new()).await;
    let provider = OpenAiCompatible::builder()
        .base_url(format!("{}/v1/", server.base))
        .api_key("sk-synthetic-test")
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let report = check_provider(&provider).await;
    assert!(!report.is_ok());
    assert!(report
        .failure("provider.stream_well_formed/plain")
        .is_some_and(|r| r.contains("error")));
}
