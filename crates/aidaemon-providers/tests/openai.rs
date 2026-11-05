mod common;

use std::time::Duration;

use aidaemon_core::{
    Image, Message, ModelRequest, Provider, ProviderError, ToolCall, ToolChoice, ToolSpec, Usage,
};
use aidaemon_providers::OpenAiCompatible;
use common::{data_events, drain, MockResponse, MockServer};
use serde_json::json;

fn provider(server: &MockServer) -> OpenAiCompatible {
    OpenAiCompatible::builder()
        .base_url(format!("{}/v1/", server.base))
        .api_key("sk-synthetic-test")
        .header("x-synthetic-header", "on")
        .timeout(Duration::from_secs(5))
        .build()
        .expect("build")
}

fn weather_tool() -> ToolSpec {
    ToolSpec {
        name: "get_weather".into(),
        description: "Weather for a city".into(),
        parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    }
}

fn request(messages: Vec<Message>) -> ModelRequest {
    ModelRequest {
        model: "test-model".into(),
        messages,
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        max_output_tokens: None,
        ..Default::default()
    }
}

fn text_stream(parts: &[&str]) -> String {
    let mut events: Vec<_> = parts
        .iter()
        .map(|p| json!({"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": null}]}))
        .collect();
    events.push(json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}));
    format!("{}data: [DONE]\n\n", data_events(&events))
}

#[tokio::test]
async fn request_body_shape() {
    let server = MockServer::start(vec![MockResponse::sse(text_stream(&["ok"]), 64)]).await;
    let mut req = request(vec![
        Message::system("You are terse."),
        Message::user("Weather in Springfield?"),
        Message::assistant(
            "Checking.",
            vec![
                ToolCall::new("call_1", "get_weather", json!({"city": "Springfield"})),
                ToolCall::new("call_2", "get_weather", json!({"_raw": "{broken"})),
            ],
        ),
        Message::tool_result("call_1", "sunny"),
        Message::tool_result("call_2", "error: bad arguments"),
    ]);
    req.tools = vec![weather_tool()];
    req.tool_choice = ToolChoice::Specific("get_weather".into());
    req.max_output_tokens = Some(256);

    provider(&server).complete(&req).await.expect("complete");

    let recorded = &server.requests()[0];
    assert_eq!(recorded.method, "POST");
    assert_eq!(recorded.path, "/v1/chat/completions");
    assert_eq!(
        recorded.headers["authorization"],
        "Bearer sk-synthetic-test"
    );
    assert_eq!(recorded.headers["x-synthetic-header"], "on");
    let body = &recorded.body;
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
    assert_eq!(body["max_tokens"], 256);
    assert_eq!(
        body["tools"],
        json!([{"type": "function", "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": weather_tool().parameters,
        }}])
    );
    assert_eq!(
        body["tool_choice"],
        json!({"type": "function", "function": {"name": "get_weather"}})
    );
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(
        messages[0],
        json!({"role": "system", "content": "You are terse."})
    );
    assert_eq!(
        messages[1],
        json!({"role": "user", "content": "Weather in Springfield?"})
    );
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["content"], "Checking.");
    let calls = messages[2]["tool_calls"].as_array().unwrap();
    assert_eq!(calls[0]["id"], "call_1");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    // Arguments travel as a JSON *string*.
    let args = calls[0]["function"]["arguments"].as_str().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(args).unwrap(),
        json!({"city": "Springfield"})
    );
    // Unparseable arguments replay verbatim.
    assert_eq!(calls[1]["function"]["arguments"], "{broken");
    assert_eq!(
        messages[3],
        json!({"role": "tool", "tool_call_id": "call_1", "content": "sunny"})
    );
    assert_eq!(messages[4]["tool_call_id"], "call_2");
}

#[tokio::test]
async fn images_become_content_parts_and_follow_tool_results() {
    let server = MockServer::start(vec![]).await;
    let p = provider(&server);
    let req = request(vec![
        Message::user("What is this?").with_images([Image::base64("image/png", "AAAA")]),
        Message::assistant(
            "",
            vec![
                ToolCall::new("call_1", "screenshot", json!({})),
                ToolCall::new("call_2", "get_weather", json!({"city": "Paris"})),
            ],
        ),
        Message::tool_result("call_1", "captured")
            .with_images([Image::url("https://example.com/s.png")]),
        Message::tool_result("call_2", "sunny"),
        Message::assistant("Here it is.", vec![]),
    ]);
    let messages = p.request_body(&req)["messages"].clone();
    assert_eq!(
        messages[0]["content"],
        json!([
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ])
    );
    // Both tool results stay contiguous after the call; the image follows them.
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["content"], "captured");
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(
        messages[4],
        json!({"role": "user", "content": [
            {"type": "text", "text": "[Images returned by tool call call_1]"},
            {"type": "image_url", "image_url": {"url": "https://example.com/s.png"}},
        ]})
    );
    assert_eq!(messages[5]["role"], "assistant");
    assert_eq!(messages.as_array().unwrap().len(), 6);
}

#[tokio::test]
async fn tool_choice_mapping_and_omission() {
    let server = MockServer::start(vec![]).await;
    let p = provider(&server);
    let mut req = request(vec![Message::user("hi")]);
    let body = p.request_body(&req);
    assert!(body.get("tools").is_none());
    assert!(body.get("tool_choice").is_none());
    assert!(body.get("max_tokens").is_none());

    req.tools = vec![weather_tool()];
    for (choice, expected) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::None, json!("none")),
        (ToolChoice::Required, json!("required")),
    ] {
        req.tool_choice = choice;
        assert_eq!(p.request_body(&req)["tool_choice"], expected);
    }
}

#[tokio::test]
async fn streamed_text_deltas_reassemble() {
    let parts = ["Hel", "lo, ", "wör", "ld", "!"];
    // 5-byte writes split events, JSON and multi-byte characters.
    let server = MockServer::start(vec![MockResponse::sse(text_stream(&parts), 5)]).await;
    let p = provider(&server);
    let req = request(vec![Message::user("greet")]);
    let (deltas, done, error) = drain(p.stream(&req)).await;
    assert!(error.is_none(), "{error:?}");
    assert_eq!(deltas, parts);
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].text, "Hello, wörld!");
    assert!(done[0].tool_calls.is_empty());
}

#[tokio::test]
async fn streamed_tool_calls_reassemble_across_fragments() {
    let events = vec![
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": null, "tool_calls": [
            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"ci"}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Shelby"}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "ty\": \"Springfield\"}"}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 1, "function": {"arguments": "ville\""}}
        ]}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let body = format!("{}data: [DONE]\n\n", data_events(&events));
    let server = MockServer::start(vec![MockResponse::sse(body, 7)]).await;
    let response = provider(&server)
        .complete(&request(vec![Message::user("weather x2")]))
        .await
        .expect("complete");
    assert_eq!(response.text, "");
    assert_eq!(
        response.tool_calls,
        vec![
            ToolCall::new("call_a", "get_weather", json!({"city": "Springfield"})),
            // Truncated JSON is preserved, not dropped.
            ToolCall::new(
                "call_b",
                "get_weather",
                json!({"_raw": "{\"city\": \"Shelbyville\""})
            ),
        ]
    );
}

#[tokio::test]
async fn usage_includes_cached_tokens() {
    let mut body = text_stream(&["done"]);
    body = body.replace(
        "data: [DONE]",
        &format!(
            "data: {}\n\ndata: [DONE]",
            json!({"choices": [], "usage": {
                "prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240,
                "prompt_tokens_details": {"cached_tokens": 1024}
            }})
        ),
    );
    let server = MockServer::start(vec![MockResponse::sse(body, 32)]).await;
    let response = provider(&server)
        .complete(&request(vec![Message::user("x")]))
        .await
        .expect("complete");
    assert_eq!(
        response.usage,
        Usage {
            input_tokens: 1200,
            output_tokens: 40,
            cached_input_tokens: 1024,
            cache_creation_tokens: 0,
        }
    );
    assert_eq!(response.usage.billable_tokens(), 216);
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
        error_for(MockResponse::status(429, "{}").header("retry-after", "3")).await,
        ProviderError::RateLimited {
            retry_after: Some(Duration::from_secs(3))
        }
    );
    assert_eq!(
        error_for(MockResponse::status(429, "{}")).await,
        ProviderError::RateLimited { retry_after: None }
    );
    assert!(matches!(
        error_for(MockResponse::status(503, "upstream down")).await,
        ProviderError::Transient(_)
    ));
    assert!(matches!(
        error_for(MockResponse::status(401, r#"{"error":{"message":"bad key"}}"#)).await,
        ProviderError::Auth(msg) if msg.contains("bad key")
    ));
    assert!(matches!(
        error_for(MockResponse::status(403, "forbidden")).await,
        ProviderError::Auth(_)
    ));
    match error_for(MockResponse::status(
        400,
        r#"{"error":{"message":"Unknown parameter: 'foo'","type":"invalid_request_error"}}"#,
    ))
    .await
    {
        ProviderError::InvalidRequest(msg) => {
            assert!(
                msg.contains("400") && msg.contains("Unknown parameter"),
                "{msg}"
            )
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn mid_stream_error_is_transient_after_deltas() {
    let body = format!(
        "{}data: {}\n\n",
        data_events(&[json!({"choices": [{"index": 0, "delta": {"content": "partial"}}]})]),
        json!({"error": {"message": "overloaded", "code": 503}})
    );
    let server = MockServer::start(vec![MockResponse::sse(body, 16)]).await;
    let p = provider(&server);
    let req = request(vec![Message::user("x")]);
    let (deltas, done, error) = drain(p.stream(&req)).await;
    assert_eq!(deltas, ["partial"]);
    assert!(done.is_empty());
    assert!(matches!(error, Some(ProviderError::Transient(m)) if m.contains("overloaded")));
}

#[tokio::test]
async fn truncated_stream_is_transient() {
    let body = data_events(&[json!({"choices": [{"index": 0, "delta": {"content": "par"}}]})]);
    let err = error_for(MockResponse::sse(body, 8)).await;
    assert!(matches!(err, ProviderError::Transient(_)), "{err:?}");
}

#[tokio::test]
async fn timeout_is_transient() {
    let server = MockServer::start(vec![
        MockResponse::sse(text_stream(&["late"]), 64).delay(Duration::from_secs(3))
    ])
    .await;
    let p = OpenAiCompatible::builder()
        .base_url(&server.base)
        .timeout(Duration::from_millis(200))
        .build()
        .unwrap();
    let err = p
        .complete(&request(vec![Message::user("x")]))
        .await
        .unwrap_err();
    assert!(
        matches!(err, ProviderError::Transient(ref m) if m.contains("timed out")),
        "{err:?}"
    );
}

#[tokio::test]
async fn connection_refused_is_transient() {
    // Bind then drop to get a port with nothing listening.
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let p = OpenAiCompatible::builder()
        .base_url(format!("http://127.0.0.1:{port}/v1"))
        .build()
        .unwrap();
    let err = p
        .complete(&request(vec![Message::user("x")]))
        .await
        .unwrap_err();
    assert!(err.is_retryable(), "{err:?}");
}

#[tokio::test]
async fn no_auth_header_without_key() {
    let server = MockServer::start(vec![MockResponse::sse(text_stream(&["ok"]), 64)]).await;
    let p = OpenAiCompatible::builder()
        .base_url(&server.base)
        .build()
        .unwrap();
    p.complete(&request(vec![Message::user("x")]))
        .await
        .unwrap();
    assert!(!server.requests()[0].headers.contains_key("authorization"));
    assert_eq!(server.requests()[0].path, "/chat/completions");
}

#[test]
fn json_formats_reasoning_and_defaults_reach_the_body() {
    let provider = OpenAiCompatible::builder()
        .api_key("k")
        .max_tokens(700)
        .reasoning_effort("high")
        .build()
        .unwrap();
    let mut req = request(vec![Message::user("Give JSON.")]);
    req.options.response_format = aidaemon_core::ResponseFormat::JsonSchema {
        name: "answer".into(),
        schema: json!({"type": "object"}),
        strict: true,
    };
    let body = provider.request_body(&req);
    assert_eq!(body["max_tokens"], 700);
    assert_eq!(body["reasoning"], json!({"effort": "high"}));
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "answer");

    req.options.reasoning_effort = Some("off".into());
    req.options.response_format = aidaemon_core::ResponseFormat::JsonObject;
    req.max_output_tokens = Some(50);
    let body = provider.request_body(&req);
    assert!(body.get("reasoning").is_none());
    assert_eq!(body["response_format"], json!({"type": "json_object"}));
    assert_eq!(body["max_tokens"], 50);
}
