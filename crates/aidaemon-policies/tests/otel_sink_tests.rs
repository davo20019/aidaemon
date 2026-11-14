//! A run exported as OpenTelemetry spans, checked against an in-memory exporter.
#![cfg(feature = "otel")]

use std::sync::Arc;

use aidaemon_core::testing::{ScriptedProvider, call, spec, text, tool_calls};
use aidaemon_core::{Agent, FnTool, ToolAnnotations, ToolOutput};
use aidaemon_policies::OtelSink;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use serde_json::{Value, json};

fn attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.to_string())
}

async fn exported(sink: impl FnOnce(&SdkTracerProvider) -> OtelSink) -> Vec<SpanData> {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let tool = FnTool::new(
        spec("lookup", "Looks things up", &[]),
        ToolAnnotations::READ_ONLY,
        |_: Value| async { Ok(ToolOutput::text("found the secret-value")) },
    );
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("test-model")
        .provider(Arc::new(ScriptedProvider::new([
            tool_calls([call("c1", "lookup", json!({}))]),
            text("my answer has secret-value"),
        ])))
        .tool(tool)
        .event_sink(sink(&provider))
        .build()
        .unwrap();

    agent
        .run("telegram:synthetic-user-1", "my prompt")
        .await
        .unwrap();

    provider.force_flush().unwrap();
    exporter.get_finished_spans().unwrap()
}

fn named<'a>(spans: &'a [SpanData], name: &str) -> Vec<&'a SpanData> {
    spans.iter().filter(|s| s.name == name).collect()
}

#[tokio::test]
async fn a_run_exports_one_root_with_model_and_tool_children() {
    let spans = exported(OtelSink::new).await;

    let root = named(&spans, "invoke_agent")[0];
    let models = named(&spans, "chat test-model");
    let tools = named(&spans, "execute_tool lookup");
    assert_eq!(models.len(), 2);
    assert_eq!(tools.len(), 1);
    assert_eq!(models[0].parent_span_id, root.span_context.span_id());
    assert_eq!(tools[0].parent_span_id, root.span_context.span_id());
    assert_eq!(
        attr(models[0], "gen_ai.request.model").as_deref(),
        Some("test-model")
    );
    assert_eq!(
        attr(models[0], "gen_ai.usage.input_tokens").as_deref(),
        Some("100")
    );
    assert_eq!(
        attr(tools[0], "gen_ai.tool.name").as_deref(),
        Some("lookup")
    );
    assert_eq!(attr(tools[0], "gen_ai.tool.call.id").as_deref(), Some("c1"));
}

#[tokio::test]
async fn nothing_the_user_wrote_leaves_by_default() {
    let spans = exported(OtelSink::new).await;

    let everything = format!("{spans:?}");
    assert!(!everything.contains("my prompt"));
    assert!(!everything.contains("my answer"));
    assert!(
        !everything.contains("synthetic-user-1"),
        "the session id is hashed"
    );
}

#[tokio::test]
async fn captured_content_goes_through_the_redactor() {
    let spans = exported(|provider| {
        OtelSink::new(provider).with_content(|text: &str| text.replace("secret-value", "[gone]"))
    })
    .await;

    let model = named(&spans, "chat test-model")[1];
    let input = attr(model, "gen_ai.input.messages").unwrap();
    let output = attr(model, "gen_ai.output.messages").unwrap();
    assert!(input.contains("my prompt"));
    assert!(input.contains("[gone]") && !input.contains("secret-value"));
    assert!(output.contains("[gone]") && !output.contains("secret-value"));
}

#[tokio::test]
async fn a_finished_run_leaves_no_open_spans_behind() {
    let spans = exported(OtelSink::new).await;

    assert_eq!(spans.len(), 4);
    assert_eq!(named(&spans, "invoke_agent").len(), 1);
}

#[tokio::test]
async fn the_prompt_actually_sent_replaces_the_kernels_view() {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let sink = Arc::new(
        OtelSink::new(&provider)
            .with_content(|text: &str| text.to_string())
            .with_wire_requests(),
    );
    let run = aidaemon_core::RunId::new();
    use aidaemon_core::{Event, EventSink, Message, ModelRequest};
    sink.emit(
        run,
        &Event::RunStarted {
            session_id: "s".into(),
        },
    );
    sink.emit(
        run,
        &Event::ModelRequested {
            turn: 1,
            model: "m".into(),
            messages: 1,
            tools: 0,
        },
    );
    sink.observe_request(
        run,
        &ModelRequest {
            messages: vec![Message::user("the kernel's view")],
            ..Default::default()
        },
    );
    sink.observe_wire(run, &json!([{"role": "user", "content": "what was sent"}]));
    sink.emit(
        run,
        &Event::ModelResponded {
            turn: 1,
            usage: Default::default(),
            text: "ok".into(),
            tool_calls: 0,
            note: None,
        },
    );

    provider.force_flush().unwrap();
    let spans = exporter.get_finished_spans().unwrap();
    let input = attr(named(&spans, "chat m")[0], "gen_ai.input.messages").unwrap();
    assert!(input.contains("what was sent"));
    assert!(!input.contains("the kernel's view"));
}
