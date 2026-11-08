//! A hook that panics, hangs, or belongs to a changed chain must not take
//! the run, or the safety of a resumed call, with it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, AgentBuilder, Event, FnTool, Hook, HookContext, HookFailure, HookPoint, InterruptKind,
    Limits, ModelDecision, ModelResponse, ReceiptStatus, Resolution, RunOutcome, RunStatus,
    StopDecision, Tool, ToolAnnotations, ToolCall, ToolDecision, ToolInfo, ToolOutput,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

fn write_tool(calls: Arc<AtomicUsize>) -> impl Tool {
    FnTool::new(
        spec("write", "A mutating tool that counts its calls", &[]),
        ToolAnnotations::default(),
        move |_: Value| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutput::text("the private result"))
            }
        },
    )
}

fn call_then_done() -> Arc<ScriptedProvider> {
    Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write", json!({}))]),
        text("done"),
    ]))
}

type Events = Arc<Mutex<Vec<Event>>>;

fn builder(provider: Arc<ScriptedProvider>, calls: Arc<AtomicUsize>) -> (AgentBuilder, Events) {
    let events: Events = Arc::default();
    let sink = events.clone();
    let builder = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(provider)
        .tool(write_tool(calls))
        .event_sink(move |_, e: &Event| sink.lock().unwrap().push(e.clone()));
    (builder, events)
}

fn failure(events: &Events, wanted: HookPoint) -> Option<(String, HookFailure)> {
    events.lock().unwrap().iter().find_map(|e| match e {
        Event::HookFailed {
            point,
            hook,
            failure,
        } if *point == wanted => Some((hook.clone(), failure.clone())),
        _ => None,
    })
}

async fn hang() {
    std::future::pending::<()>().await;
}

// ---------- a hook that panics ----------

struct PanicsBeforeTool;
#[async_trait]
impl Hook for PanicsBeforeTool {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _info: &ToolInfo,
    ) -> ToolDecision {
        panic!("gate broke");
    }
}

#[tokio::test]
async fn a_tool_check_that_panics_refuses_the_call_and_the_run_goes_on() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (agent, events) = builder(call_then_done(), calls.clone());
    let agent = agent.hook(PanicsBeforeTool).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(out.state.receipts[0].status, ReceiptStatus::Denied);
    assert_eq!(out.final_text.as_deref(), Some("done"));
    let (hook, failed) = failure(&events, HookPoint::BeforeTool).unwrap();
    assert!(hook.contains("PanicsBeforeTool"), "{hook}");
    assert!(matches!(failed, HookFailure::Panicked { message } if message.contains("gate broke")));
}

struct PanicsAfterTool;
#[async_trait]
impl Hook for PanicsAfterTool {
    async fn after_tool(
        &self,
        _ctx: &HookContext<'_>,
        _receipt: &aidaemon_core::Receipt,
        _output: &mut ToolOutput,
    ) {
        panic!("redactor broke");
    }
}

#[tokio::test]
async fn a_result_check_that_panics_withholds_the_result_from_the_model() {
    let provider = call_then_done();
    let (agent, _) = builder(provider.clone(), Arc::default());
    let agent = agent.hook(PanicsAfterTool).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.state.receipts[0].status, ReceiptStatus::Succeeded);
    let second = &provider.requests()[1].messages;
    assert!(second.iter().all(|m| !m.content.contains("private result")));
}

struct PanicsOnStop;
#[async_trait]
impl Hook for PanicsOnStop {
    async fn on_stop(&self, _ctx: &HookContext<'_>, _text: &str) -> StopDecision {
        panic!("checker broke");
    }
}

#[tokio::test]
async fn a_completion_check_that_panics_ends_the_run_unverified() {
    let (agent, _) = builder(
        Arc::new(ScriptedProvider::new([text("done")])),
        Arc::default(),
    );
    let agent = agent.hook(PanicsOnStop).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.status, RunStatus::Completed);
    assert!(!out.is_succeeded());
    assert_eq!(out.state.unmet.len(), 1);
}

struct PanicsAfterModel;
#[async_trait]
impl Hook for PanicsAfterModel {
    async fn after_model(
        &self,
        _ctx: &HookContext<'_>,
        _response: &mut ModelResponse,
    ) -> ModelDecision {
        panic!("filter broke");
    }
}

#[tokio::test]
async fn a_response_check_that_panics_stops_the_run_without_acting_on_it() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (agent, _) = builder(call_then_done(), calls.clone());
    let agent = agent.hook(PanicsAfterModel).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert!(matches!(out.status, RunStatus::Stopped { .. }));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

struct PanicsBeforeModel;
#[async_trait]
impl Hook for PanicsBeforeModel {
    async fn before_model(
        &self,
        _ctx: &HookContext<'_>,
        _request: &mut aidaemon_core::ModelRequest,
    ) {
        panic!("memory broke");
    }
}

#[tokio::test]
async fn a_request_hook_that_panics_is_recorded_and_the_turn_goes_on() {
    let (agent, events) = builder(
        Arc::new(ScriptedProvider::new([text("done")])),
        Arc::default(),
    );
    let agent = agent.hook(PanicsBeforeModel).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("done"));
    assert!(failure(&events, HookPoint::BeforeModel).is_some());
}

struct PanicsAtTheEnd;
#[async_trait]
impl Hook for PanicsAtTheEnd {
    async fn on_run_end(&self, _ctx: &HookContext<'_>, _outcome: &RunOutcome) {
        panic!("writer broke");
    }
}

#[tokio::test]
async fn a_run_end_hook_that_panics_does_not_lose_the_outcome() {
    let (agent, events) = builder(
        Arc::new(ScriptedProvider::new([text("done")])),
        Arc::default(),
    );
    let agent = agent.hook(PanicsAtTheEnd).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("done"));
    assert!(failure(&events, HookPoint::OnRunEnd).is_some());
}

// ---------- a hook that hangs ----------

struct HangsBeforeTool;
#[async_trait]
impl Hook for HangsBeforeTool {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _info: &ToolInfo,
    ) -> ToolDecision {
        hang().await;
        ToolDecision::Allow
    }
}

#[tokio::test]
async fn a_hook_that_hangs_times_out_and_refuses_the_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (agent, events) = builder(call_then_done(), calls.clone());
    let agent = agent
        .hook(HangsBeforeTool)
        .limits(Limits {
            hook_timeout: Duration::from_millis(20),
            ..Limits::default()
        })
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(out.state.receipts[0].status, ReceiptStatus::Denied);
    assert!(matches!(
        failure(&events, HookPoint::BeforeTool),
        Some((_, HookFailure::TimedOut { .. }))
    ));
}

struct PatientHook(Arc<AtomicUsize>);
#[async_trait]
impl Hook for PatientHook {
    fn timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(5))
    }
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _info: &ToolInfo,
    ) -> ToolDecision {
        tokio::time::sleep(Duration::from_millis(60)).await;
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolDecision::Allow
    }
}

#[tokio::test]
async fn a_hook_can_ask_for_more_time_than_the_default() {
    let calls = Arc::new(AtomicUsize::new(0));
    let waited = Arc::new(AtomicUsize::new(0));
    let (agent, _) = builder(call_then_done(), calls.clone());
    let agent = agent
        .hook(PatientHook(waited.clone()))
        .limits(Limits {
            hook_timeout: Duration::from_millis(10),
            ..Limits::default()
        })
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(waited.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancel_interrupts_a_hook_that_hangs() {
    let (agent, _) = builder(call_then_done(), Arc::default());
    let agent = agent.hook(HangsBeforeTool).build().unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        trigger.cancel();
    });

    let started = Instant::now();
    let out = agent.run_with_cancel("s", "go", cancel).await.unwrap();

    assert_eq!(out.status, RunStatus::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn the_wall_clock_budget_bounds_a_hook_that_hangs() {
    let (agent, _) = builder(call_then_done(), Arc::default());
    let agent = agent
        .hook(HangsBeforeTool)
        .limits(Limits {
            timeout: Some(Duration::from_millis(30)),
            ..Limits::default()
        })
        .build()
        .unwrap();

    let started = Instant::now();
    agent.run("s", "go").await.unwrap();

    assert!(started.elapsed() < Duration::from_secs(5));
}

struct HangsOnStop;
#[async_trait]
impl Hook for HangsOnStop {
    async fn on_stop(&self, _ctx: &HookContext<'_>, _text: &str) -> StopDecision {
        hang().await;
        StopDecision::Accept
    }
}

#[tokio::test]
async fn cancel_during_a_completion_check_ends_the_run_cancelled() {
    let (agent, _) = builder(
        Arc::new(ScriptedProvider::new([text("done")])),
        Arc::default(),
    );
    let agent = agent.hook(HangsOnStop).build().unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        trigger.cancel();
    });

    let out = agent.run_with_cancel("s", "go", cancel).await.unwrap();

    assert_eq!(out.status, RunStatus::Cancelled);
}

// ---------- an ended run is not resumable ----------

/// Cancels the run as its first call executes, so the next call never starts.
fn cancelling_tool(cancel: CancellationToken) -> impl Tool {
    FnTool::new(
        spec("stop_here", "Cancels the run", &[]),
        ToolAnnotations::default(),
        move |_: Value| {
            let cancel = cancel.clone();
            async move {
                cancel.cancel();
                Ok(ToolOutput::text("cancelled from inside"))
            }
        },
    )
}

#[tokio::test]
async fn a_run_cancelled_mid_batch_leaves_nothing_to_recover() {
    let cancel = CancellationToken::new();
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(Arc::new(ScriptedProvider::new([tool_calls([
            call("c1", "stop_here", json!({})),
            call("c2", "stop_here", json!({})),
        ])])))
        .tool(cancelling_tool(cancel.clone()))
        .build()
        .unwrap();

    let out = agent.run_with_cancel("s", "go", cancel).await.unwrap();

    assert_eq!(out.status, RunStatus::Cancelled);
    assert!(out.state.in_flight.is_none());
}

#[tokio::test]
async fn a_run_stopped_by_a_limit_mid_batch_leaves_nothing_to_recover() {
    let (agent, _) = builder(
        Arc::new(ScriptedProvider::new([tool_calls([
            call("c1", "write", json!({})),
            call("c2", "write", json!({})),
        ])])),
        Arc::default(),
    );
    let agent = agent
        .limits(Limits {
            max_tool_calls: Some(1),
            ..Limits::default()
        })
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert!(matches!(out.status, RunStatus::LimitReached { .. }));
    assert!(out.state.in_flight.is_none());
}

// ---------- an approval belongs to the hooks that saw it ----------

struct Filler;
#[async_trait]
impl Hook for Filler {}

struct Asker;
#[async_trait]
impl Hook for Asker {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _info: &ToolInfo,
    ) -> ToolDecision {
        ToolDecision::Interrupt {
            reason: "ask".into(),
            kind: InterruptKind::default(),
        }
    }
}

struct Gate;
#[async_trait]
impl Hook for Gate {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _info: &ToolInfo,
    ) -> ToolDecision {
        ToolDecision::Deny {
            reason: "the gate refuses".into(),
        }
    }
}

#[tokio::test]
async fn an_approval_is_judged_again_when_the_hook_chain_changed() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (before, _) = builder(call_then_done(), calls.clone());
    let before = before.hook(Filler).hook(Asker).hook(Gate).build().unwrap();
    let paused = before.run("s", "go").await.unwrap();
    assert!(matches!(paused.status, RunStatus::Interrupted { .. }));

    // The chain shrank: the saved position now points past the gate.
    let (after, _) = builder(call_then_done(), calls.clone());
    let after = after.hook(Asker).hook(Gate).build().unwrap();
    let resumed = after
        .resume(paused.state, Resolution::Approve)
        .await
        .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(matches!(resumed.status, RunStatus::Interrupted { .. }));
}

#[tokio::test]
async fn an_approval_settles_the_asker_when_the_chain_is_unchanged() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (agent, _) = builder(call_then_done(), calls.clone());
    let agent = agent.hook(Filler).hook(Asker).build().unwrap();

    let paused = agent.run("s", "go").await.unwrap();
    let done = agent
        .resume(paused.state, Resolution::Approve)
        .await
        .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(done.final_text.as_deref(), Some("done"));
}
