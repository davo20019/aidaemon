//! Exports runs as OpenTelemetry spans, following the GenAI conventions, for
//! any OTLP backend (LangSmith, Langfuse, a collector).
//!
//! Each run segment is one short trace: a root span from `RunStarted` or
//! `RunResumed` to `Finished`, with a child per model call and tool call. A
//! pause for an owner ends the segment, so no span stays open for hours and
//! a resume does not depend on a live span. Segments of a run share
//! `aidaemon.run_id`.
//!
//! Exporting is egress of whatever a span holds, so by default nothing the
//! user or the model wrote leaves: spans carry names, counts and timings,
//! and the session id is hashed. [`OtelSink::with_content`] opts in to
//! prompts and replies and requires a redactor.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use aidaemon_core::{Event, EventSink, ModelRequest, RunId, RunStatus};
use opentelemetry::trace::{
    Span as _, SpanKind, Status, TraceContextExt, Tracer as _, TracerProvider as _,
};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, Span};
use opentelemetry_sdk::Resource;

/// Longest text put on a span when content capture is on.
const MAX_CONTENT_CHARS: usize = 32_000;

type Redact = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// A provider exporting to the OTLP/HTTP endpoint at `endpoint`, in batches
/// on its own thread. `headers` usually hold the backend's API key.
pub fn otlp_provider(
    endpoint: &str,
    headers: HashMap<String, String>,
    service_name: &str,
) -> Result<SdkTracerProvider, String> {
    let endpoint = format!("{}/v1/traces", endpoint.trim_end_matches('/'));
    let service_name = service_name.to_string();
    // The exporter's blocking HTTP client cannot be built on a Tokio worker.
    std::thread::spawn(move || {
        let exporter = SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .with_protocol(Protocol::HttpBinary)
            .with_headers(headers)
            .build()
            .map_err(|error| format!("could not build the OTLP exporter: {error}"))?;
        Ok(SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(Resource::builder().with_service_name(service_name).build())
            .build())
    })
    .join()
    .map_err(|_| "the OTLP exporter thread panicked".to_string())?
}

#[derive(Default)]
struct OpenRun {
    root: Option<Context>,
    model: Option<Span>,
    tools: HashMap<String, Span>,
}

/// An [`EventSink`] that turns the run's events into spans.
pub struct OtelSink {
    tracer: SdkTracer,
    runs: Mutex<HashMap<RunId, OpenRun>>,
    content: Option<Redact>,
    wire_requests: bool,
}

impl OtelSink {
    pub fn new(provider: &SdkTracerProvider) -> Self {
        Self {
            tracer: provider.tracer("aidaemon"),
            runs: Mutex::default(),
            content: None,
            wire_requests: false,
        }
    }

    /// The application reshapes each request after the kernel's hooks, so
    /// the prompt to export comes from [`OtelSink::observe_wire`] instead.
    pub fn with_wire_requests(mut self) -> Self {
        self.wire_requests = true;
        self
    }

    /// Record the messages actually sent for the run's current model call.
    pub fn observe_wire<T: serde::Serialize + ?Sized>(&self, run_id: RunId, messages: &T) {
        if self.content.is_some() {
            self.record_input(run_id, serde_json::to_string(messages).unwrap_or_default());
        }
    }

    fn record_input(&self, run_id: RunId, messages: String) {
        let Ok(mut runs) = self.runs.lock() else {
            return;
        };
        let Some(span) = runs.get_mut(&run_id).and_then(|run| run.model.as_mut()) else {
            return;
        };
        if let Some(messages) = self.captured(&messages) {
            span.set_attribute(KeyValue::new("gen_ai.input.messages", messages));
        }
    }

    /// Also export prompts, replies and tool arguments, each passed through
    /// `redact` first.
    pub fn with_content(mut self, redact: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        self.content = Some(Arc::new(redact));
        self
    }

    fn captured(&self, text: &str) -> Option<String> {
        let redact = self.content.as_ref()?;
        Some(redact(text).chars().take(MAX_CONTENT_CHARS).collect())
    }

    fn session_key(&self, session_id: &str) -> String {
        if self.content.is_some() {
            return session_id.to_string();
        }
        let mut hasher = DefaultHasher::new();
        session_id.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn open_root(&self, run_id: RunId, attributes: Vec<KeyValue>) -> Context {
        let mut attributes = attributes;
        attributes.push(KeyValue::new("gen_ai.operation.name", "invoke_agent"));
        attributes.push(KeyValue::new("aidaemon.run_id", run_id.to_string()));
        let span = self
            .tracer
            .span_builder("invoke_agent")
            .with_kind(SpanKind::Internal)
            .with_attributes(attributes)
            .start_with_context(&self.tracer, &Context::new());
        Context::new().with_span(span)
    }

    fn child(&self, root: &Context, name: String, attributes: Vec<KeyValue>) -> Span {
        self.tracer
            .span_builder(name)
            .with_kind(SpanKind::Client)
            .with_attributes(attributes)
            .start_with_context(&self.tracer, root)
    }
}

fn root_of<'a>(sink: &OtelSink, run_id: RunId, run: &'a mut OpenRun) -> &'a Context {
    // A run recovered after a crash journals neither a start nor a resume.
    run.root
        .get_or_insert_with(|| sink.open_root(run_id, Vec::new()))
}

impl EventSink for OtelSink {
    fn emit(&self, run_id: RunId, event: &Event) {
        let Ok(mut runs) = self.runs.lock() else {
            return;
        };
        let run = runs.entry(run_id).or_default();
        match event {
            Event::RunStarted { session_id } => {
                let key = self.session_key(session_id);
                let root = self.open_root(
                    run_id,
                    vec![
                        KeyValue::new("gen_ai.conversation.id", key.clone()),
                        KeyValue::new("langsmith.trace.session_id", key.clone()),
                        KeyValue::new("langfuse.session.id", key),
                    ],
                );
                run.root = Some(root);
            }
            Event::RunResumed => {
                run.root = Some(self.open_root(run_id, Vec::new()));
            }
            Event::RunLinked { parent_run } => {
                root_of(self, run_id, run)
                    .span()
                    .set_attribute(KeyValue::new(
                        "aidaemon.parent_run_id",
                        parent_run.to_string(),
                    ));
            }
            Event::ModelRequested {
                turn,
                model,
                messages,
                tools,
            } => {
                let root = root_of(self, run_id, run).clone();
                run.model = Some(self.child(
                    &root,
                    format!("chat {model}"),
                    vec![
                        KeyValue::new("gen_ai.operation.name", "chat"),
                        KeyValue::new("gen_ai.request.model", model.clone()),
                        KeyValue::new("aidaemon.turn", i64::from(*turn)),
                        KeyValue::new("aidaemon.messages", *messages as i64),
                        KeyValue::new("aidaemon.tools", *tools as i64),
                    ],
                ));
            }
            Event::ModelResponded {
                usage, text, note, ..
            } => {
                if let Some(mut span) = run.model.take() {
                    span.set_attribute(KeyValue::new(
                        "gen_ai.usage.input_tokens",
                        usage.input_tokens as i64,
                    ));
                    span.set_attribute(KeyValue::new(
                        "gen_ai.usage.output_tokens",
                        usage.output_tokens as i64,
                    ));
                    if let Some(note) = note {
                        span.set_attribute(KeyValue::new("aidaemon.note", note.clone()));
                    }
                    if let Some(text) = self.captured(text) {
                        span.set_attribute(KeyValue::new("gen_ai.output.messages", text));
                    }
                    span.end();
                }
            }
            Event::ModelRetried { error, .. } => {
                if let Some(mut span) = run.model.take() {
                    span.set_status(Status::error(error.clone()));
                    span.end();
                }
            }
            Event::ToolStarted { call_id, tool } => {
                let root = root_of(self, run_id, run).clone();
                let span = self.child(
                    &root,
                    format!("execute_tool {tool}"),
                    vec![
                        KeyValue::new("gen_ai.operation.name", "execute_tool"),
                        KeyValue::new("gen_ai.tool.name", tool.clone()),
                        KeyValue::new("gen_ai.tool.call.id", call_id.clone()),
                    ],
                );
                run.tools.insert(call_id.clone(), span);
            }
            Event::ToolFinished { receipt } => {
                if let Some(mut span) = run.tools.remove(&receipt.call_id) {
                    span.set_attribute(KeyValue::new(
                        "aidaemon.tool.status",
                        format!("{:?}", receipt.status),
                    ));
                    span.set_attribute(KeyValue::new(
                        "aidaemon.tool.duration_ms",
                        receipt.duration_ms as i64,
                    ));
                    if let Some(arguments) = self.captured(&receipt.arguments.to_string()) {
                        span.set_attribute(KeyValue::new("gen_ai.tool.call.arguments", arguments));
                    }
                    span.end();
                }
            }
            Event::HookDecided {
                point,
                hook,
                decision,
                ..
            } => {
                root_of(self, run_id, run).span().add_event(
                    "hook_decided",
                    vec![
                        KeyValue::new("aidaemon.hook.point", format!("{point:?}")),
                        KeyValue::new("aidaemon.hook.name", hook.clone()),
                        KeyValue::new("aidaemon.hook.decision", decision.clone()),
                    ],
                );
            }
            Event::HookFailed {
                point,
                hook,
                failure,
            } => {
                root_of(self, run_id, run).span().add_event(
                    "hook_failed",
                    vec![
                        KeyValue::new("aidaemon.hook.point", format!("{point:?}")),
                        KeyValue::new("aidaemon.hook.name", hook.clone()),
                        KeyValue::new("aidaemon.hook.failure", format!("{failure:?}")),
                    ],
                );
            }
            Event::LimitExtended { limit, amount } => {
                root_of(self, run_id, run).span().add_event(
                    "limit_extended",
                    vec![
                        KeyValue::new("aidaemon.limit", format!("{limit:?}")),
                        KeyValue::new("aidaemon.amount", *amount as i64),
                    ],
                );
            }
            Event::TranscriptEdited { before, after, .. } => {
                root_of(self, run_id, run).span().add_event(
                    "transcript_edited",
                    vec![
                        KeyValue::new("aidaemon.messages.before", *before as i64),
                        KeyValue::new("aidaemon.messages.after", *after as i64),
                    ],
                );
            }
            Event::Finished {
                status,
                usage,
                turns,
                ..
            } => {
                let mut run = runs.remove(&run_id).unwrap_or_default();
                let failed = !matches!(
                    status,
                    RunStatus::Completed
                        | RunStatus::Finished { .. }
                        | RunStatus::Interrupted { .. }
                );
                if let Some(mut span) = run.model.take() {
                    if failed {
                        span.set_status(Status::error("the run ended during this call"));
                    }
                    span.end();
                }
                for (_, mut span) in run.tools.drain() {
                    span.end();
                }
                let root = root_of(self, run_id, &mut run).span();
                root.set_attribute(KeyValue::new("aidaemon.turns", i64::from(*turns)));
                root.set_attribute(KeyValue::new(
                    "gen_ai.usage.input_tokens",
                    usage.input_tokens as i64,
                ));
                root.set_attribute(KeyValue::new(
                    "gen_ai.usage.output_tokens",
                    usage.output_tokens as i64,
                ));
                if failed {
                    root.set_status(Status::error(format!("{status:?}")));
                }
                root.end();
            }
            _ => {}
        }
    }

    fn observe_request(&self, run_id: RunId, request: &ModelRequest) {
        if self.content.is_some() && !self.wire_requests {
            self.record_input(
                run_id,
                serde_json::to_string(&request.messages).unwrap_or_default(),
            );
        }
    }
}
