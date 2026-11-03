//! Streaming, usage roll-up, access manifests and crash recovery.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider, StreamingProvider};
use aidaemon_core::{
    AccessManifest, Agent, Event, Hook, HookContext, MemoryStore, ProviderError, ReceiptStatus,
    Retry, Role, RunId, RunStatus, RunStore, Tool, ToolAnnotations, ToolCall, ToolContext,
    ToolDecision, ToolInfo, ToolOutput, ToolResult, ToolSpec, Usage,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn streamed_text_arrives_as_deltas_before_the_response() {
    let events: Arc<Mutex<Vec<Event>>> = Arc::default();
    let sink = events.clone();
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(StreamingProvider(ScriptedProvider::new([text(
            "hello brave world",
        )])))
        .event_sink(move |_, e: &Event| sink.lock().unwrap().push(e.clone()))
        .build()
        .unwrap();

    let out = agent.run("s", "hi").await.unwrap();
    assert_eq!(out.final_text.as_deref(), Some("hello brave world"));
    let deltas: String = events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| match e {
            Event::TextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, "hello brave world");
}

#[tokio::test]
async fn retry_layer_retries_a_stream_that_fails_before_any_output() {
    let flaky = ScriptedProvider::with_results([
        Err(ProviderError::Transient("503".into())),
        Ok(text("recovered")),
    ]);
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(Retry::new(StreamingProvider(flaky)).base_delay(Duration::from_millis(1)))
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(out.final_text.as_deref(), Some("recovered"));
}

/// A tool that spends model tokens (like a sub-agent) and declares access.
struct Delegate;
#[async_trait]
impl Tool for Delegate {
    fn spec(&self) -> ToolSpec {
        spec("delegate", "spends tokens", &["path"])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }
    fn access(&self, args: &Value) -> AccessManifest {
        AccessManifest {
            reads: vec![args["path"].as_str().unwrap_or_default().to_string()],
            ..AccessManifest::default()
        }
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
        Ok(ToolOutput::text("child done").with_usage(Usage {
            input_tokens: 1_000,
            output_tokens: 50,
            ..Usage::default()
        }))
    }
}

struct ScopeToDocs(Arc<Mutex<Vec<AccessManifest>>>);
#[async_trait]
impl Hook for ScopeToDocs {
    async fn before_tool(
        &self,
        _: &HookContext<'_>,
        _: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        self.0.lock().unwrap().push(tool.access.clone());
        if tool.access.reads.iter().all(|p| p.starts_with("docs/")) {
            ToolDecision::Allow
        } else {
            ToolDecision::Deny {
                reason: "outside docs/".into(),
            }
        }
    }
}

#[tokio::test]
async fn tool_usage_counts_toward_the_run_and_hooks_see_access() {
    let seen: Arc<Mutex<Vec<AccessManifest>>> = Arc::default();
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([
            tool_calls([
                call("c1", "delegate", json!({"path": "docs/a.md"})),
                call("c2", "delegate", json!({"path": "/etc/passwd"})),
            ]),
            text("ok"),
        ]))
        .tool(Delegate)
        .hook(ScopeToDocs(seen.clone()))
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    // two model turns (100 in / 20 out each) + one child call (1000 / 50)
    assert_eq!(out.usage().input_tokens, 1_200);
    assert_eq!(out.usage().output_tokens, 90);
    assert_eq!(out.receipts()[1].status, ReceiptStatus::Denied);
    assert_eq!(
        seen.lock().unwrap()[1].reads,
        vec!["/etc/passwd".to_string()]
    );
}

// ---------- crash recovery ----------

/// Blocks forever on its first call (simulating a crash mid-call), then
/// succeeds on later calls.
struct HangsOnce {
    name: &'static str,
    idempotent: bool,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for HangsOnce {
    fn spec(&self) -> ToolSpec {
        spec(self.name, "hangs once", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: self.idempotent,
        }
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            std::future::pending::<()>().await;
        }
        Ok(ToolOutput::text(format!("{} ran", self.name)))
    }
}

async fn crash_then_recover(idempotent: bool) -> (aidaemon_core::RunOutcome, usize, usize) {
    let store = Arc::new(MemoryStore::new());
    let hang_calls = Arc::new(AtomicUsize::new(0));
    let after_calls = Arc::new(AtomicUsize::new(1)); // never hangs
    let run_id: Arc<Mutex<Option<RunId>>> = Arc::default();
    let rid = run_id.clone();
    let build = |provider: ScriptedProvider| {
        Agent::builder("worker")
            .allow_unguarded()
            .model("m")
            .provider(provider)
            .tool(HangsOnce {
                name: "write_report",
                idempotent,
                calls: hang_calls.clone(),
            })
            .tool(HangsOnce {
                name: "notify",
                idempotent: false,
                calls: after_calls.clone(),
            })
            .store(store.clone())
    };

    // First process: dies while `write_report` is running.
    let first = build(ScriptedProvider::new([tool_calls([
        call("c1", "write_report", json!({})),
        call("c2", "notify", json!({})),
    ])]))
    .event_sink(move |id, _e: &Event| {
        rid.lock().unwrap().get_or_insert(id);
    })
    .build()
    .unwrap();
    let crashed = tokio::time::timeout(Duration::from_millis(200), first.run("s", "report")).await;
    assert!(crashed.is_err(), "the first run must not finish");

    // Second process: load the checkpoint and continue.
    let id = run_id.lock().unwrap().unwrap();
    let saved = store
        .load_state(id)
        .await
        .unwrap()
        .expect("checkpoint saved");
    assert!(saved.in_flight.is_some());
    let second = build(ScriptedProvider::new([text("report handled")]))
        .build()
        .unwrap();
    let out = second
        .run_from(saved, CancellationToken::new())
        .await
        .unwrap();
    (
        out,
        hang_calls.load(Ordering::SeqCst),
        after_calls.load(Ordering::SeqCst) - 1,
    )
}

#[tokio::test]
async fn non_idempotent_call_in_flight_is_marked_indeterminate_not_rerun() {
    let (out, report_calls, notify_calls) = crash_then_recover(false).await;
    assert_eq!(out.status, RunStatus::Completed);
    assert_eq!(report_calls, 1, "not re-run after the crash");
    assert_eq!(notify_calls, 1, "the call that never started still runs");
    let r: Vec<ReceiptStatus> = out.receipts().iter().map(|r| r.status).collect();
    assert_eq!(
        r,
        vec![ReceiptStatus::Indeterminate, ReceiptStatus::Succeeded]
    );
    let tool_msgs = out
        .state
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .count();
    assert_eq!(tool_msgs, 2, "every call has exactly one result");
}

#[tokio::test]
async fn idempotent_call_in_flight_is_rerun() {
    let (out, report_calls, notify_calls) = crash_then_recover(true).await;
    assert_eq!(out.status, RunStatus::Completed);
    assert_eq!(report_calls, 2, "re-run once after the crash");
    assert_eq!(notify_calls, 1);
    assert!(out.receipts().iter().all(|r| r.succeeded()));
}

// ---------- typed message origins, error flags, timeout hints, with_limits ----------

#[tokio::test]
async fn injected_messages_carry_a_typed_origin_and_errors_are_flagged() {
    use aidaemon_core::{FnTool, Limits, MessageOrigin, StopDecision};
    struct RejectOnce(AtomicUsize);
    #[async_trait]
    impl Hook for RejectOnce {
        async fn on_stop(&self, _: &HookContext<'_>, _: &str) -> StopDecision {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                StopDecision::Continue {
                    message: "verify first".into(),
                }
            } else {
                StopDecision::Accept
            }
        }
    }
    let failing = FnTool::new(
        spec("grep", "search", &[]),
        ToolAnnotations::READ_ONLY,
        |_args: Value| async move { Ok(ToolOutput::error("no matches for pattern")) },
    );
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "grep", json!({}))]),
            text("done"),
            text("done, verified"),
        ]))
        .tool(failing)
        .hook(RejectOnce(AtomicUsize::new(0)))
        .build()
        .unwrap();
    let out = agent.run("s", "find it").await.unwrap();

    let msgs = &out.state.messages;
    assert_eq!(msgs[0].origin, Some(MessageOrigin::User));
    let tool_msg = msgs.iter().find(|m| m.role == Role::Tool).unwrap();
    assert!(tool_msg.is_error);
    assert_eq!(
        out.receipts()[0].reason.as_deref(),
        Some("no matches for pattern")
    );
    let injected = msgs.iter().find(|m| m.content == "verify first").unwrap();
    assert_eq!(injected.origin, Some(MessageOrigin::StopHook));
    assert!(injected.is_injected());

    // with_limits keeps identity and tools, changes only the budget.
    let child = agent.with_limits(Limits {
        max_turns: 2,
        ..Limits::default()
    });
    assert_eq!(child.id(), agent.id());
    assert_eq!(child.limits().max_turns, 2);
    assert_eq!(child.tool_specs().len(), 1);
}

#[tokio::test]
async fn a_tool_timeout_hint_overrides_the_default() {
    use aidaemon_core::Limits;
    struct Patient;
    #[async_trait]
    impl Tool for Patient {
        fn spec(&self) -> ToolSpec {
            spec("patient", "takes 150ms", &[])
        }
        fn annotations(&self) -> ToolAnnotations {
            ToolAnnotations::READ_ONLY
        }
        fn timeout(&self) -> Option<Duration> {
            Some(Duration::from_secs(5))
        }
        async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
            tokio::time::sleep(Duration::from_millis(150)).await;
            Ok(ToolOutput::text("finished"))
        }
    }
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "patient", json!({}))]),
            text("ok"),
        ]))
        .tool(Patient)
        .limits(Limits {
            tool_timeout: Duration::from_millis(20),
            ..Limits::default()
        })
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Succeeded);
}
