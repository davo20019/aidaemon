mod common;

use std::time::Duration;

use aidaemon_core::{
    Message, ModelRequest, Provider, ProviderError, ToolCall, ToolChoice, ToolSpec, Usage,
};
use aidaemon_providers::{Anthropic, ConfigError};
use common::{drain, named_events, MockResponse, MockServer};
use serde_json::{json, Value};

fn provider(server: &MockServer) -> Anthropic {
    Anthropic::builder()
        .api_key("sk-ant-synthetic")
        .base_url(&server.base)
        .timeout(Duration::from_secs(5))
        .build()
        .expect("build")
}

fn tool(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: format!("{name} tool"),
        parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
    }
}

fn request(messages: Vec<Message>) -> ModelRequest {
    ModelRequest {
        model: "claude-test".into(),
        messages,
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_output_tokens: None,
    }
}

fn message_start(usage: Value) -> (&'static str, Value) {
    (
        "message_start",
        json!({"type": "message_start", "message": {
            "id": "msg_synthetic", "type": "message", "role": "assistant",
            "content": [], "model": "claude-test", "usage": usage
        }}),
    )
}

fn text_reply(parts: &[&str]) -> String {
    let mut events = vec![
        message_start(json!({"input_tokens": 10, "output_tokens": 1})),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        ("ping", json!({"type": "ping"})),
    ];
    for part in parts {
        events.push((
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": part}}),
        ));
    }
    events.extend([
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ]);
    named_events(&events)
}

#[tokio::test]
async fn request_body_shape_and_cache_markers() {
    let server = MockServer::start(vec![MockResponse::sse(text_reply(&["ok"]), 64)]).await;
    let mut req = request(vec![
        Message::system("Stable instructions."),
        Message::system("More stable context."),
        Message::user("Read both files."),
        Message::assistant(
            "Reading.",
            vec![
                ToolCall::new("toolu_1", "read", json!({"path": "a.txt"})),
                ToolCall::new("toolu_2", "read", json!({"path": "b.txt"})),
            ],
        ),
        Message::tool_result("toolu_1", "alpha"),
        Message::tool_result("toolu_2", "beta"),
        Message::user("Now summarize."),
    ]);
    req.tools = vec![tool("read"), tool("write")];
    req.tool_choice = ToolChoice::Required;

    provider(&server).complete(&req).await.expect("complete");
    let recorded = &server.requests()[0];
    assert_eq!(recorded.path, "/v1/messages");
    assert_eq!(recorded.headers["x-api-key"], "sk-ant-synthetic");
    assert_eq!(recorded.headers["anthropic-version"], "2023-06-01");
    let body = &recorded.body;
    assert_eq!(body["model"], "claude-test");
    assert_eq!(body["max_tokens"], 4096);
    assert_eq!(body["stream"], true);
    assert_eq!(
        body["system"],
        json!([
            {"type": "text", "text": "Stable instructions."},
            {"type": "text", "text": "More stable context.", "cache_control": {"type": "ephemeral"}},
        ])
    );
    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools[0]["name"], "read");
    assert_eq!(tools[0]["input_schema"], tool("read").parameters);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"], json!({"type": "ephemeral"}));
    assert_eq!(body["tool_choice"], json!({"type": "any"}));

    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3, "{messages:#?}");
    assert_eq!(
        messages[0],
        json!({"role": "user", "content": [{"type": "text", "text": "Read both files."}]})
    );
    assert_eq!(
        messages[1],
        json!({"role": "assistant", "content": [
            {"type": "text", "text": "Reading."},
            {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.txt"}},
            {"type": "tool_use", "id": "toolu_2", "name": "read", "input": {"path": "b.txt"}},
        ]})
    );
    // Both tool results merged into one user turn, ahead of the follow-up text.
    assert_eq!(
        messages[2],
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "alpha"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "beta"},
            {"type": "text", "text": "Now summarize."},
        ]})
    );
}

#[test]
fn tool_choice_mapping_and_options() {
    let p = Anthropic::builder()
        .api_key("k")
        .max_tokens(1000)
        .prompt_caching(false)
        .build()
        .unwrap();
    let mut req = request(vec![Message::system("s"), Message::user("hi")]);
    let body = p.request_body(&req);
    assert!(body.get("tools").is_none() && body.get("tool_choice").is_none());
    assert_eq!(body["max_tokens"], 1000);
    assert!(body["system"][0].get("cache_control").is_none());

    req.tools = vec![tool("read")];
    req.max_output_tokens = Some(77);
    for (choice, expected) in [
        (ToolChoice::Auto, json!({"type": "auto"})),
        (ToolChoice::None, json!({"type": "none"})),
        (
            ToolChoice::Specific("read".into()),
            json!({"type": "tool", "name": "read"}),
        ),
    ] {
        req.tool_choice = choice;
        let body = p.request_body(&req);
        assert_eq!(body["tool_choice"], expected);
        assert_eq!(body["max_tokens"], 77);
    }
}

#[test]
fn missing_key_and_redacted_debug() {
    assert!(matches!(
        Anthropic::builder().build(),
        Err(ConfigError::MissingApiKey)
    ));
    let p = Anthropic::builder()
        .api_key("sk-ant-very-secret")
        .build()
        .unwrap();
    assert!(!format!("{p:?}").contains("very-secret"));
    let b = Anthropic::builder().api_key("sk-ant-very-secret");
    assert!(!format!("{b:?}").contains("very-secret"));
}

#[tokio::test]
async fn streamed_text_and_tool_use_with_usage() {
    let events = vec![
        message_start(json!({
            "input_tokens": 50, "output_tokens": 1,
            "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 300
        })),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "look."}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "read", "input": {}}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"pa"}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "th\": \"notes/a.txt\"}"}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 1}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_10", "name": "list", "input": {}}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 2}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 42}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ];
    let server = MockServer::start(vec![MockResponse::sse(named_events(&events), 9)]).await;
    let p = provider(&server);
    let req = request(vec![Message::user("read a")]);
    let (deltas, done, error) = drain(p.stream(&req)).await;
    assert!(error.is_none(), "{error:?}");
    assert_eq!(deltas, ["Let me ", "look."]);
    assert_eq!(done.len(), 1);
    let response = &done[0];
    assert_eq!(response.text, "Let me look.");
    assert_eq!(
        response.tool_calls,
        vec![
            ToolCall::new("toolu_9", "read", json!({"path": "notes/a.txt"})),
            ToolCall::new("toolu_10", "list", json!({})),
        ]
    );
    assert_eq!(
        response.usage,
        Usage {
            input_tokens: 2350,
            output_tokens: 42,
            cached_input_tokens: 2000,
            cache_creation_tokens: 300,
        }
    );
    assert_eq!(response.usage.billable_tokens(), 350 + 42);
}

async fn error_for(response: MockResponse) -> ProviderError {
    let server = MockServer::start(vec![response]).await;
    provider(&server)
        .complete(&request(vec![Message::user("x")]))
        .await
        .expect_err("should fail")
}

#[tokio::test]
async fn http_error_mapping() {
    assert_eq!(
        error_for(MockResponse::status(429, "{}").header("retry-after", "12")).await,
        ProviderError::RateLimited {
            retry_after: Some(Duration::from_secs(12))
        }
    );
    // 529 overloaded is in the 5xx range.
    assert!(matches!(
        error_for(MockResponse::status(
            529,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
        ))
        .await,
        ProviderError::Transient(m) if m.contains("Overloaded")
    ));
    assert!(matches!(
        error_for(MockResponse::status(401, r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#)).await,
        ProviderError::Auth(m) if m.contains("invalid x-api-key")
    ));
    assert!(matches!(
        error_for(MockResponse::status(404, r#"{"type":"error","error":{"type":"not_found_error","message":"model: nope"}}"#)).await,
        ProviderError::InvalidRequest(m) if m.contains("model: nope")
    ));
}

#[tokio::test]
async fn mid_stream_error_event_is_transient() {
    let body = named_events(&[
        message_start(json!({"input_tokens": 1})),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        ),
        (
            "error",
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
        ),
    ]);
    let server = MockServer::start(vec![MockResponse::sse(body, 11)]).await;
    let p = provider(&server);
    let req = request(vec![Message::user("x")]);
    let (deltas, done, error) = drain(p.stream(&req)).await;
    assert_eq!(deltas, ["Hi"]);
    assert!(done.is_empty());
    assert!(
        matches!(error, Some(ProviderError::Transient(ref m)) if m.contains("overloaded_error")),
        "{error:?}"
    );
}

#[tokio::test]
async fn stream_without_message_stop_is_transient() {
    let body = text_reply(&["cut"]).replace("event: message_stop", "event: ignored");
    let body = body.replace("{\"type\":\"message_stop\"}", "{\"type\":\"ignored\"}");
    let err = error_for(MockResponse::sse(body, 32)).await;
    assert!(matches!(err, ProviderError::Transient(_)), "{err:?}");
}
