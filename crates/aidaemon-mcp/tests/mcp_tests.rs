use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::testing::{call, text, tool_calls, ScriptedProvider};
use aidaemon_core::{Agent, ReceiptStatus, Role, Tool, ToolAnnotations, ToolContext};
use aidaemon_mcp::{McpError, McpOptions, McpServer};
use serde_json::{json, Value};

const FAKE: &str = env!("CARGO_BIN_EXE_fake_mcp_server");

fn no_env() -> Vec<(String, String)> {
    Vec::new()
}

async fn spawn() -> McpServer {
    McpServer::spawn("fake", FAKE, Vec::<String>::new(), no_env())
        .await
        .expect("fake server starts")
}

async fn spawn_with(options: McpOptions) -> McpServer {
    McpServer::spawn_with("fake", FAKE, Vec::<String>::new(), no_env(), options)
        .await
        .expect("fake server starts")
}

fn ctx() -> ToolContext {
    ToolContext::new("s", "c")
}

fn find(server: &McpServer, name: &str) -> Arc<dyn Tool> {
    server
        .tools()
        .into_iter()
        .find(|t| t.spec().name == name)
        .unwrap_or_else(|| panic!("no tool {name}"))
}

#[tokio::test]
async fn handshake_reports_server_info_and_captures_stderr() {
    let server = spawn().await;
    let info = server.info();
    assert_eq!(info.protocol_version, "2025-06-18");
    assert_eq!(info.server_info["name"], "fake-mcp");
    assert_eq!(info.instructions.as_deref(), Some("synthetic test server"));
    assert!(server.is_alive());
    assert_eq!(server.request("ping", json!({})).await.unwrap(), json!({}));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(server.stderr_tail().contains("fake-mcp: started"));
}

#[tokio::test]
async fn unsupported_protocol_version_is_refused() {
    let err = McpServer::spawn("fake", FAKE, ["--protocol", "1999-01-01"], no_env())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, McpError::Protocol { message, .. } if message.contains("1999-01-01")),
        "{err}"
    );
}

#[tokio::test]
async fn handshake_times_out() {
    let options = McpOptions {
        init_timeout: Duration::from_millis(200),
        ..McpOptions::default()
    };
    let err = McpServer::spawn_with("fake", FAKE, ["--hang-init"], no_env(), options)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, McpError::Timeout { method, .. } if method == "initialize"),
        "{err}"
    );
}

#[tokio::test]
async fn spawn_failure_is_an_error() {
    let err = McpServer::spawn(
        "x",
        "/nonexistent/mcp-server-binary",
        Vec::<String>::new(),
        no_env(),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, McpError::Spawn { .. }), "{err}");
}

#[tokio::test]
async fn pagination_collects_every_page_and_names_are_sanitized() {
    let server = spawn().await;
    let names: Vec<String> = server.tools().iter().map(|t| t.spec().name).collect();
    assert_eq!(names.len(), 10, "{names:?}");
    for expected in [
        "fake__echo",
        "fake__write_file",
        "fake__no_annotations",
        "fake__fail",
        "fake__env_probe",
    ] {
        assert!(names.contains(&expected.to_string()), "{names:?}");
    }
    let long = names.iter().find(|n| n.starts_with("fake__aaa")).unwrap();
    assert_eq!(long.len(), 64);
    for name in &names {
        assert!(name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'));
    }

    let mcp = server.mcp_tools();
    let write = mcp
        .iter()
        .find(|t| t.remote_name() == "write.file")
        .unwrap();
    assert_eq!(write.server_name(), "fake");

    let echo = find(&server, "fake__echo");
    let spec = echo.spec();
    assert_eq!(spec.description, "Echo the text back");
    assert_eq!(spec.parameters["required"], json!(["text"]));
    assert!(echo.validate(&json!({})).is_err(), "schema is enforced");

    // Re-listing works on a live connection.
    assert_eq!(server.list_tools().await.unwrap().len(), 10);
}

#[tokio::test]
async fn annotations_map_including_missing_ones() {
    let server = spawn().await;
    assert_eq!(
        find(&server, "fake__echo").annotations(),
        ToolAnnotations::READ_ONLY
    );
    assert_eq!(
        find(&server, "fake__write_file").annotations(),
        ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: true,
            egress: true
        }
    );
    assert_eq!(
        find(&server, "fake__no_annotations").annotations(),
        ToolAnnotations::DESTRUCTIVE
    );
    assert_eq!(
        find(&server, "fake__crash").annotations(),
        ToolAnnotations::DESTRUCTIVE
    );
}

#[tokio::test]
async fn successful_call_returns_text() {
    let server = spawn().await;
    let out = find(&server, "fake__echo")
        .call(json!({"text": "hello there"}), &ctx())
        .await
        .unwrap();
    assert!(!out.is_error);
    assert_eq!(out.content, "hello there");
    assert_eq!(out.details["server"], "fake");
}

#[tokio::test]
async fn rich_content_is_summarized_with_structured_details() {
    let server = spawn().await;
    let out = find(&server, "fake__rich")
        .call(json!({}), &ctx())
        .await
        .unwrap();
    assert!(out
        .content
        .starts_with("summary\n[image content: image/png"));
    assert_eq!(out.details["structured_content"], json!({"rows": 3}));
}

#[tokio::test]
async fn is_error_result_becomes_error_output() {
    let server = spawn().await;
    let out = find(&server, "fake__fail")
        .call(json!({}), &ctx())
        .await
        .unwrap();
    assert!(out.is_error);
    assert_eq!(out.content, "disk is full");
}

#[tokio::test]
async fn server_requests_and_noise_are_handled() {
    let server = spawn().await;
    let out = find(&server, "fake__ping_client")
        .call(json!({}), &ctx())
        .await
        .unwrap();
    assert_eq!(out.content, "pong received");
}

#[tokio::test]
async fn environment_is_scrubbed_but_explicit_env_passes() {
    let server = McpServer::spawn(
        "fake",
        FAKE,
        Vec::<String>::new(),
        [("FAKE_MCP_TOKEN", "synthetic-token-123")],
    )
    .await
    .unwrap();
    let out = find(&server, "fake__env_probe")
        .call(json!({}), &ctx())
        .await
        .unwrap();
    let probe: &Value = &out.details["structured_content"];
    assert_eq!(probe["token"], "synthetic-token-123");
    assert_eq!(probe["has_cargo_manifest_dir"], false);
}

#[tokio::test]
async fn crash_turns_into_tool_error_not_panic() {
    let server = spawn().await;
    let err = find(&server, "fake__crash")
        .call(json!({}), &ctx())
        .await
        .unwrap_err();
    assert!(err.0.contains("not running"), "{err}");
    assert!(
        err.0.contains("crashing on purpose"),
        "stderr tail included: {err}"
    );
    assert!(!server.is_alive());

    // Later calls fail fast too.
    let again = find(&server, "fake__echo")
        .call(json!({"text": "x"}), &ctx())
        .await
        .unwrap_err();
    assert!(again.0.contains("not running"), "{again}");
}

#[tokio::test]
async fn call_times_out_and_connection_stays_usable() {
    let server = spawn_with(McpOptions {
        request_timeout: Duration::from_millis(200),
        ..McpOptions::default()
    })
    .await;
    let err = find(&server, "fake__sleep")
        .call(json!({}), &ctx())
        .await
        .unwrap_err();
    assert!(err.0.contains("did not answer"), "{err}");
    let ok = find(&server, "fake__echo")
        .call(json!({"text": "still here"}), &ctx())
        .await
        .unwrap();
    assert_eq!(ok.content, "still here");
}

#[tokio::test]
async fn call_honours_cancellation() {
    let server = spawn().await;
    let context = ctx();
    let cancel = context.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();
    });
    let err = find(&server, "fake__sleep")
        .call(json!({}), &context)
        .await
        .unwrap_err();
    assert!(err.0.contains("cancelled"), "{err}");
}

#[tokio::test]
async fn rpc_errors_are_typed() {
    let server = spawn().await;
    let err = server
        .request("tools/call", json!({"name": "missing", "arguments": {}}))
        .await
        .unwrap_err();
    assert!(matches!(err, McpError::Rpc { code: -32602, .. }), "{err}");
}

#[tokio::test]
async fn shutdown_stops_the_server() {
    let server = spawn().await;
    let echo = find(&server, "fake__echo");
    server.shutdown().await;
    assert!(!server.is_alive());
    assert!(echo.call(json!({"text": "x"}), &ctx()).await.is_err());
}

#[tokio::test]
async fn agent_calls_mcp_tools_end_to_end() {
    let server = spawn().await;
    let mut builder = Agent::builder("assistant")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([
            tool_calls([
                call("c1", "fake__echo", json!({"text": "from the model"})),
                call("c2", "fake__fail", json!({})),
            ]),
            tool_calls([call("c3", "fake__crash", json!({}))]),
            text("all done"),
        ]));
    for tool in server.tools() {
        builder = builder.tool_arc(tool);
    }
    let agent = builder.build().unwrap();

    let out = agent.run("session-1", "use the tools").await.unwrap();
    assert!(out.is_completed());
    assert_eq!(out.final_text.as_deref(), Some("all done"));
    let statuses: Vec<ReceiptStatus> = out.receipts().iter().map(|r| r.status).collect();
    assert_eq!(
        statuses,
        vec![
            ReceiptStatus::Succeeded,
            ReceiptStatus::Failed,
            ReceiptStatus::Failed
        ]
    );
    assert!(out.receipts()[0].read_only);
    let results: Vec<&str> = out
        .state
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| m.content.as_str())
        .collect();
    assert!(results[0].contains("from the model"), "{results:?}");
    assert!(results[1].contains("disk is full"), "{results:?}");
    assert!(results[2].contains("not running"), "{results:?}");
}
