//! A real `aidaemon_core::Agent` driving each provider against the mock
//! server through a full tool round trip.

mod common;

use std::time::Duration;

use aidaemon_core::{Agent, FnTool, Retry, ToolAnnotations, ToolOutput, ToolSpec};
use aidaemon_providers::{Anthropic, OpenAiCompatible};
use common::{data_events, named_events, MockResponse, MockServer};
use serde_json::{json, Value};

fn lookup_tool() -> impl aidaemon_core::Tool {
    let spec = ToolSpec {
        name: "lookup_order".into(),
        description: "Look up an order by id".into(),
        parameters: json!({"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}),
    };
    FnTool::new(spec, ToolAnnotations::READ_ONLY, |args: Value| async move {
        let id = args["order_id"].as_str().unwrap_or_default().to_string();
        Ok(ToolOutput::text(format!("order {id}: shipped")))
    })
}

#[tokio::test]
async fn openai_agent_tool_round_trip() {
    let first = format!(
        "{}data: [DONE]\n\n",
        data_events(&[
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_77", "type": "function", "function": {"name": "lookup_order", "arguments": "{\"order_"}}
            ]}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "id\": \"A-100\"}"}}
            ]}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}),
        ])
    );
    let second = format!(
        "{}data: [DONE]\n\n",
        data_events(&[
            json!({"choices": [{"index": 0, "delta": {"content": "Order A-100 "}}]}),
            json!({"choices": [{"index": 0, "delta": {"content": "has shipped."}, "finish_reason": "stop"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 150, "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 100}}}),
        ])
    );
    let server = MockServer::start(vec![
        // A transient failure first, absorbed by the core Retry wrapper.
        MockResponse::status(503, "warming up"),
        MockResponse::sse(first, 13),
        MockResponse::sse(second, 13),
    ])
    .await;
    let provider = OpenAiCompatible::builder()
        .base_url(&server.base)
        .api_key("sk-synthetic")
        .build()
        .unwrap();
    let agent = Agent::builder("order-bot")
        .model("test-model")
        .instructions("You answer order questions.")
        .provider(Retry::new(provider).base_delay(Duration::from_millis(5)))
        .tool(lookup_tool())
        .build()
        .unwrap();

    let outcome = agent
        .run("test:synthetic-user-1", "Where is order A-100?")
        .await
        .unwrap();
    assert!(outcome.is_completed(), "{:?}", outcome.status);
    assert_eq!(
        outcome.final_text.as_deref(),
        Some("Order A-100 has shipped.")
    );
    assert_eq!(outcome.receipts().len(), 1);
    assert_eq!(outcome.receipts()[0].tool, "lookup_order");
    assert_eq!(outcome.usage().input_tokens, 250);
    assert_eq!(outcome.usage().cached_input_tokens, 100);

    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    let messages = requests[2].body["messages"].as_array().unwrap().clone();
    assert_eq!(messages[0]["role"], "system");
    let assistant = messages.iter().find(|m| m["role"] == "assistant").unwrap();
    assert_eq!(assistant["tool_calls"][0]["id"], "call_77");
    let tool_msg = messages.iter().find(|m| m["role"] == "tool").unwrap();
    assert_eq!(tool_msg["tool_call_id"], "call_77");
    assert!(tool_msg["content"]
        .as_str()
        .unwrap()
        .contains("order A-100: shipped"));
    assert_eq!(
        requests[2].body["tools"][0]["function"]["name"],
        "lookup_order"
    );
}

#[tokio::test]
async fn anthropic_agent_tool_round_trip() {
    let start = |usage: Value| {
        (
            "message_start",
            json!({"type": "message_start", "message": {"id": "msg_s", "role": "assistant", "content": [], "usage": usage}}),
        )
    };
    let first = named_events(&[
        start(json!({"input_tokens": 20, "cache_creation_input_tokens": 500})),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup_order", "input": {}}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"order_id\": "}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"B-7\"}"}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 15}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ]);
    let second = named_events(&[
        start(json!({"input_tokens": 40, "cache_read_input_tokens": 500})),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "B-7 shipped."}}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ]);
    let server = MockServer::start(vec![
        MockResponse::sse(first, 17),
        MockResponse::sse(second, 17),
    ])
    .await;
    let provider = Anthropic::builder()
        .api_key("sk-ant-synthetic")
        .base_url(&server.base)
        .build()
        .unwrap();
    let agent = Agent::builder("order-bot")
        .model("claude-test")
        .instructions("You answer order questions.")
        .provider(provider)
        .tool(lookup_tool())
        .build()
        .unwrap();

    let outcome = agent
        .run("test:synthetic-user-1", "Status of B-7?")
        .await
        .unwrap();
    assert!(outcome.is_completed(), "{:?}", outcome.status);
    assert_eq!(outcome.final_text.as_deref(), Some("B-7 shipped."));
    assert_eq!(outcome.usage().cached_input_tokens, 500);
    assert_eq!(outcome.usage().cache_creation_tokens, 500);

    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    let body = &requests[1].body;
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["content"][0]["type"], "tool_use");
    assert_eq!(
        messages[1]["content"][0]["input"],
        json!({"order_id": "B-7"})
    );
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"][0]["type"], "tool_result");
    assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_1");
    assert!(messages[2]["content"][0]["content"]
        .as_str()
        .unwrap()
        .contains("order B-7: shipped"));
}
