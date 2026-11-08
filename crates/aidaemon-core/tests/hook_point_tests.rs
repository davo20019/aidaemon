//! The moments of the loop a hook can act on beyond tools and stopping: run
//! start, the model's response, a failed model call, a reached limit, and a
//! durable transcript edit.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, Event, FnTool, Hook, HookContext, InterruptKind, LimitDecision, LimitGrants, LimitKind,
    Limits, Message, ModelDecision, ModelErrorDecision, ModelResponse, ProviderError,
    ReceiptStatus, Resolution, Role, RunState, RunStatus, StopDecision, Tool, ToolAnnotations,
    ToolCall, ToolDecision, ToolInfo, ToolOutput, HARD_TURN_CAP,
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
                Ok(ToolOutput::text("written"))
            }
        },
    )
}

fn agent(provider: Arc<ScriptedProvider>) -> aidaemon_core::AgentBuilder {
    Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(provider)
}

type Events = Arc<Mutex<Vec<Event>>>;

fn recording() -> (Events, impl Fn(aidaemon_core::RunId, &Event) + Send + Sync) {
    let events: Events = Arc::default();
    let sink = events.clone();
    (events, move |_, e: &Event| {
        sink.lock().unwrap().push(e.clone())
    })
}

// ---------- on_run_start ----------

#[derive(Default)]
struct CountStarts {
    starts: AtomicUsize,
    turn_at_start: Mutex<Option<u32>>,
}
#[async_trait]
impl Hook for CountStarts {
    async fn on_run_start(&self, ctx: &HookContext<'_>) {
        self.starts.fetch_add(1, Ordering::SeqCst);
        *self.turn_at_start.lock().unwrap() = Some(ctx.turn);
    }
}

#[tokio::test]
async fn run_start_fires_once_before_the_first_turn() {
    let hook = Arc::new(CountStarts::default());
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write", json!({}))]),
        text("done"),
    ]));
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .hook_arc(hook.clone())
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert_eq!(hook.starts.load(Ordering::SeqCst), 1);
    assert_eq!(*hook.turn_at_start.lock().unwrap(), Some(0));
}

struct AskOnce(AtomicUsize);
#[async_trait]
impl Hook for AskOnce {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _info: &ToolInfo,
    ) -> ToolDecision {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            ToolDecision::Interrupt {
                reason: "ask".into(),
                kind: InterruptKind::default(),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

#[tokio::test]
async fn resuming_a_paused_run_is_not_a_run_start() {
    let hook = Arc::new(CountStarts::default());
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write", json!({}))]),
        text("done"),
    ]));
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .hook(AskOnce(AtomicUsize::new(0)))
        .hook_arc(hook.clone())
        .build()
        .unwrap();

    let paused = agent.run("s", "go").await.unwrap();
    assert!(matches!(paused.status, RunStatus::Interrupted { .. }));
    agent
        .resume(paused.state, Resolution::Approve)
        .await
        .unwrap();

    assert_eq!(hook.starts.load(Ordering::SeqCst), 1);
}

// ---------- after_model ----------

struct RewriteText;
#[async_trait]
impl Hook for RewriteText {
    async fn after_model(
        &self,
        _ctx: &HookContext<'_>,
        response: &mut ModelResponse,
    ) -> ModelDecision {
        response.text = response.text.replace("secret", "[redacted]");
        ModelDecision::Accept
    }
}

#[tokio::test]
async fn after_model_rewrites_what_is_recorded() {
    let provider = Arc::new(ScriptedProvider::new([text("the secret is out")]));
    let agent = agent(provider).hook(RewriteText).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("the [redacted] is out"));
    assert_eq!(
        out.state.messages.last().unwrap().content,
        "the [redacted] is out"
    );
}

struct InjectCall;
#[async_trait]
impl Hook for InjectCall {
    async fn after_model(
        &self,
        ctx: &HookContext<'_>,
        response: &mut ModelResponse,
    ) -> ModelDecision {
        if ctx.turn == 1 {
            response.tool_calls.push(call("", "write", json!({})));
        }
        ModelDecision::Accept
    }
}

struct DenyWrite;
#[async_trait]
impl Hook for DenyWrite {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _info: &ToolInfo,
    ) -> ToolDecision {
        ToolDecision::Deny {
            reason: "policy".into(),
        }
    }
}

#[tokio::test]
async fn a_call_added_by_after_model_still_faces_every_tool_check() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(ScriptedProvider::new([text("thinking"), text("done")]));
    let agent = agent(provider)
        .tool(write_tool(calls.clone()))
        .hook(InjectCall)
        .hook(DenyWrite)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let receipt = &out.state.receipts[0];
    assert_eq!(receipt.status, ReceiptStatus::Denied);
    assert_eq!(
        receipt.call_id, "call_1_0",
        "the kernel fills in a blank id"
    );
}

struct RejectFirst(AtomicUsize);
#[async_trait]
impl Hook for RejectFirst {
    async fn after_model(
        &self,
        _ctx: &HookContext<'_>,
        _response: &mut ModelResponse,
    ) -> ModelDecision {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            ModelDecision::Reject {
                message: "cite a source".into(),
            }
        } else {
            ModelDecision::Accept
        }
    }
}

#[tokio::test]
async fn a_rejected_response_runs_no_tools_and_the_model_sees_why() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut draft = tool_calls([call("c1", "write", json!({}))]);
    draft.text = "draft".into();
    let provider = Arc::new(ScriptedProvider::new([draft, text("done")]));
    let (events, sink) = recording();
    let agent = agent(provider.clone())
        .tool(write_tool(calls.clone()))
        .hook(RejectFirst(AtomicUsize::new(0)))
        .event_sink(sink)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("done"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(out.state.receipts.is_empty());
    // The model sees its own attempt and the feedback; the unanswered calls
    // are gone, because a provider refuses a call with no result.
    let second = &provider.requests()[1].messages;
    assert!(second.iter().any(|m| m.content == "draft"));
    assert!(second.iter().any(|m| m.content == "cite a source"));
    assert!(second.iter().all(|m| m.tool_calls.is_empty()));
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|e| matches!(e, Event::ModelRejected { message, .. } if message == "cite a source")));
}

#[tokio::test]
async fn a_rejected_response_still_counts_against_the_budget() {
    let provider = Arc::new(ScriptedProvider::new([text("draft"), text("done")]));
    let agent = agent(provider)
        .hook(RejectFirst(AtomicUsize::new(0)))
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.state.turn, 2);
    assert_eq!(out.state.run_usage.output_tokens, 40);
}

// ---------- on_model_error ----------

#[derive(Default)]
struct RetryRetryable {
    seen: Mutex<Vec<ProviderError>>,
    models_asked: AtomicUsize,
}
#[async_trait]
impl Hook for RetryRetryable {
    async fn before_model(
        &self,
        _ctx: &HookContext<'_>,
        _request: &mut aidaemon_core::ModelRequest,
    ) {
        self.models_asked.fetch_add(1, Ordering::SeqCst);
    }
    async fn on_model_error(
        &self,
        _ctx: &HookContext<'_>,
        error: &ProviderError,
    ) -> ModelErrorDecision {
        self.seen.lock().unwrap().push(error.clone());
        if error.is_retryable() {
            ModelErrorDecision::Retry {
                after: Duration::ZERO,
            }
        } else {
            ModelErrorDecision::Fail
        }
    }
}

#[tokio::test]
async fn a_hook_can_retry_a_failed_model_call_with_the_typed_error() {
    let hook = Arc::new(RetryRetryable::default());
    let provider = Arc::new(ScriptedProvider::with_results([
        Err(ProviderError::Transient("503".into())),
        Ok(text("recovered")),
    ]));
    let (events, sink) = recording();
    let agent = agent(provider)
        .hook_arc(hook.clone())
        .event_sink(sink)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("recovered"));
    assert_eq!(
        *hook.seen.lock().unwrap(),
        vec![ProviderError::Transient("503".into())]
    );
    // A retry goes back through `before_model`, so a hook can shrink the
    // request or change the model for the next attempt.
    assert_eq!(hook.models_asked.load(Ordering::SeqCst), 2);
    assert_eq!(out.state.turn, 2);
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|e| matches!(e, Event::ModelRetried { .. })));
}

#[tokio::test]
async fn a_model_failure_ends_the_run_when_no_hook_retries() {
    let hook = Arc::new(RetryRetryable::default());
    let provider = Arc::new(ScriptedProvider::with_results([Err(ProviderError::Auth(
        "bad key".into(),
    ))]));
    let agent = agent(provider).hook_arc(hook).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert!(matches!(out.status, RunStatus::ProviderFailed { .. }));
}

struct RetryLater;
#[async_trait]
impl Hook for RetryLater {
    async fn on_model_error(
        &self,
        _ctx: &HookContext<'_>,
        _error: &ProviderError,
    ) -> ModelErrorDecision {
        ModelErrorDecision::Retry {
            after: Duration::from_secs(3600),
        }
    }
}

#[tokio::test]
async fn the_wait_before_a_retry_stops_on_cancel() {
    let provider = Arc::new(ScriptedProvider::with_results([Err(
        ProviderError::Transient("503".into()),
    )]));
    let agent = agent(provider).hook(RetryLater).build().unwrap();
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
async fn the_wait_before_a_retry_stays_inside_the_wall_clock_budget() {
    let provider = Arc::new(ScriptedProvider::with_results([Err(
        ProviderError::Transient("503".into()),
    )]));
    let agent = agent(provider)
        .hook(RetryLater)
        .limits(Limits {
            timeout: Some(Duration::from_millis(30)),
            ..Limits::default()
        })
        .build()
        .unwrap();

    let started = Instant::now();
    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(
        out.status,
        RunStatus::LimitReached {
            limit: LimitKind::WallClock
        }
    );
    assert!(started.elapsed() < Duration::from_secs(5));
}

// ---------- on_limit ----------

#[derive(Default)]
struct Extend {
    amount: u64,
    asked: Mutex<Vec<LimitKind>>,
}
#[async_trait]
impl Hook for Extend {
    async fn on_limit(&self, _ctx: &HookContext<'_>, limit: LimitKind) -> LimitDecision {
        self.asked.lock().unwrap().push(limit);
        LimitDecision::Extend {
            amount: self.amount,
        }
    }
}

fn extend(amount: u64) -> Arc<Extend> {
    Arc::new(Extend {
        amount,
        ..Extend::default()
    })
}

fn max_turns(turns: u32) -> Limits {
    Limits {
        max_turns: turns,
        ..Limits::default()
    }
}

#[tokio::test]
async fn a_limit_ends_the_run_when_no_hook_extends_it() {
    let provider = Arc::new(ScriptedProvider::new([tool_calls([call(
        "c1",
        "write",
        json!({}),
    )])]));
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .limits(max_turns(1))
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(
        out.status,
        RunStatus::LimitReached {
            limit: LimitKind::Turns
        }
    );
}

#[tokio::test]
async fn a_hook_can_extend_a_limit_and_the_grant_is_on_the_record() {
    let hook = extend(1);
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write", json!({}))]),
        text("done"),
    ]));
    let (events, sink) = recording();
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .limits(max_turns(1))
        .hook_arc(hook.clone())
        .event_sink(sink)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("done"));
    assert_eq!(*hook.asked.lock().unwrap(), vec![LimitKind::Turns]);
    assert_eq!(out.state.limit_grants.turns, 1);
    assert!(events.lock().unwrap().iter().any(|e| matches!(
        e,
        Event::LimitExtended {
            limit: LimitKind::Turns,
            amount: 1,
        }
    )));
}

#[tokio::test]
async fn a_grant_of_nothing_ends_the_run() {
    let hook = extend(0);
    let provider = Arc::new(ScriptedProvider::new([tool_calls([call(
        "c1",
        "write",
        json!({}),
    )])]));
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .limits(max_turns(1))
        .hook_arc(hook.clone())
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert!(matches!(out.status, RunStatus::LimitReached { .. }));
    assert_eq!(hook.asked.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_limit_inside_a_batch_can_be_extended() {
    let calls = Arc::new(AtomicUsize::new(0));
    let hook = extend(1);
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([
            call("c1", "write", json!({})),
            call("c2", "write", json!({})),
        ]),
        text("done"),
    ]));
    let agent = agent(provider)
        .tool(write_tool(calls.clone()))
        .limits(Limits {
            max_tool_calls: Some(1),
            ..Limits::default()
        })
        .hook_arc(hook.clone())
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("done"));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(hook.asked.lock().unwrap().contains(&LimitKind::ToolCalls));
}

struct AlwaysContinue;
#[async_trait]
impl Hook for AlwaysContinue {
    async fn on_stop(&self, _ctx: &HookContext<'_>, _text: &str) -> StopDecision {
        StopDecision::Continue {
            message: "more".into(),
        }
    }
}

#[tokio::test]
async fn no_hook_can_pass_the_hard_turn_cap() {
    let hook = extend(1_000);
    let provider = Arc::new(ScriptedProvider::default().then_repeat(text("again")));
    let agent = agent(provider)
        .limits(Limits {
            max_turns: HARD_TURN_CAP,
            max_idle_continuations: u32::MAX,
            ..Limits::default()
        })
        .hook(AlwaysContinue)
        .hook_arc(hook.clone())
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.state.turn, HARD_TURN_CAP);
    assert!(matches!(out.status, RunStatus::LimitReached { .. }));
    assert!(hook.asked.lock().unwrap().is_empty());
}

#[tokio::test]
async fn grants_survive_a_save_and_a_new_run_starts_without_them() {
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write", json!({}))]),
        text("done"),
    ]));
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .limits(max_turns(1))
        .hook_arc(extend(4))
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();

    let saved = serde_json::to_string(&out.state).unwrap();
    let restored: RunState = serde_json::from_str(&saved).unwrap();
    assert_eq!(restored.limit_grants.turns, 4);

    let next = restored.with_user_message("again");
    assert_eq!(next.limit_grants, LimitGrants::default());
}

// ---------- edit_transcript ----------

struct EditAtTurn {
    turn: u32,
    replacement: Vec<Message>,
}
#[async_trait]
impl Hook for EditAtTurn {
    async fn edit_transcript(&self, ctx: &HookContext<'_>) -> Option<Vec<Message>> {
        (ctx.turn == self.turn).then(|| self.replacement.clone())
    }
}

fn two_turns() -> Arc<ScriptedProvider> {
    Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write", json!({}))]),
        text("done"),
    ]))
}

#[tokio::test]
async fn a_transcript_edit_is_durable_and_leaves_the_receipts_alone() {
    let provider = two_turns();
    let (events, sink) = recording();
    let agent = agent(provider.clone())
        .tool(write_tool(Arc::default()))
        .hook(EditAtTurn {
            turn: 2,
            replacement: vec![Message::user("summary of the work so far")],
        })
        .event_sink(sink)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    let request = &provider.requests()[1].messages;
    assert!(request
        .iter()
        .any(|m| m.content == "summary of the work so far"));
    assert!(request.iter().all(|m| m.role != Role::Tool));
    assert_eq!(out.state.messages[0].content, "summary of the work so far");
    assert_eq!(
        out.state.receipts.len(),
        1,
        "evidence outlives the transcript"
    );
    assert!(events.lock().unwrap().iter().any(|e| matches!(
        e,
        Event::TranscriptEdited {
            before: 3,
            after: 1,
            ..
        }
    )));
}

async fn assert_refused(replacement: Vec<Message>) {
    let provider = two_turns();
    let (events, sink) = recording();
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .hook(EditAtTurn {
            turn: 2,
            replacement,
        })
        .event_sink(sink)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(
        out.state.messages[0].content, "go",
        "the edit was not applied"
    );
    assert_eq!(out.final_text.as_deref(), Some("done"));
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|e| matches!(e, Event::TranscriptEditRefused { .. })));
}

#[tokio::test]
async fn an_edit_that_leaves_a_call_unanswered_is_refused() {
    assert_refused(vec![
        Message::user("summary"),
        Message::assistant("", vec![call("c1", "write", json!({}))]),
    ])
    .await;
}

#[tokio::test]
async fn an_edit_that_leaves_a_result_without_its_call_is_refused() {
    assert_refused(vec![
        Message::user("summary"),
        Message::tool_result("c1", "written"),
    ])
    .await;
}

#[tokio::test]
async fn an_edit_with_no_user_message_is_refused() {
    assert_refused(vec![Message::assistant("only me", Vec::new())]).await;
}

#[tokio::test]
async fn an_empty_edit_is_refused() {
    assert_refused(Vec::new()).await;
}

// ---------- a restrictive answer beats a permissive one ----------

struct Forbid;
#[async_trait]
impl Hook for Forbid {
    async fn on_limit(&self, _ctx: &HookContext<'_>, _limit: LimitKind) -> LimitDecision {
        LimitDecision::Forbid
    }
    async fn on_model_error(
        &self,
        _ctx: &HookContext<'_>,
        _error: &ProviderError,
    ) -> ModelErrorDecision {
        ModelErrorDecision::Abort
    }
    async fn on_stop(&self, _ctx: &HookContext<'_>, _text: &str) -> StopDecision {
        StopDecision::Halt {
            reason: "the owner stopped this work".into(),
        }
    }
}

#[tokio::test]
async fn a_later_hook_can_forbid_an_extension() {
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write", json!({}))]),
        text("done"),
    ]));
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .limits(max_turns(1))
        .hook_arc(extend(5))
        .hook(Forbid)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert!(matches!(out.status, RunStatus::LimitReached { .. }));
    assert_eq!(out.state.limit_grants, LimitGrants::default());
}

#[tokio::test]
async fn a_later_hook_can_abort_a_retry() {
    let provider = Arc::new(ScriptedProvider::with_results([
        Err(ProviderError::Transient("503".into())),
        Ok(text("never")),
    ]));
    let agent = agent(provider)
        .hook_arc(Arc::new(RetryRetryable::default()))
        .hook(Forbid)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert!(matches!(out.status, RunStatus::ProviderFailed { .. }));
}

#[tokio::test]
async fn a_later_hook_can_halt_a_continuation() {
    let provider = Arc::new(ScriptedProvider::new([text("done")]));
    let agent = agent(provider.clone())
        .hook(AlwaysContinue)
        .hook(Forbid)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.status, RunStatus::Completed);
    assert!(!out.is_succeeded());
    assert_eq!(out.state.unmet, vec!["the owner stopped this work"]);
    assert_eq!(provider.requests().len(), 1);
}

// ---------- continuations need progress ----------

fn idle_cap(cap: u32) -> Limits {
    Limits {
        max_idle_continuations: cap,
        ..Limits::default()
    }
}

#[tokio::test]
async fn continuing_without_progress_ends_the_run_unmet() {
    let provider = Arc::new(ScriptedProvider::default().then_repeat(text("still nothing")));
    let agent = agent(provider.clone())
        .limits(idle_cap(2))
        .hook(AlwaysContinue)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.status, RunStatus::Completed);
    assert!(!out.is_succeeded());
    assert_eq!(provider.requests().len(), 3);
}

struct UntilThreeReceipts;
#[async_trait]
impl Hook for UntilThreeReceipts {
    async fn on_stop(&self, ctx: &HookContext<'_>, _text: &str) -> StopDecision {
        if ctx.run_receipts().count() < 3 {
            StopDecision::Continue {
                message: "keep going".into(),
            }
        } else {
            StopDecision::Accept
        }
    }
}

#[tokio::test]
async fn new_successful_work_resets_the_continuation_count() {
    let write = |id: &str| tool_calls([call(id, "write", json!({}))]);
    let provider = Arc::new(ScriptedProvider::new([
        text("a"),
        write("c1"),
        text("b"),
        write("c2"),
        text("c"),
        write("c3"),
        text("finished"),
    ]));
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .limits(idle_cap(1))
        .hook(UntilThreeReceipts)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("finished"));
    assert!(out.is_succeeded());
}

// ---------- answering in the model's place ----------

struct Cached;
#[async_trait]
impl Hook for Cached {
    async fn answer_model(
        &self,
        _ctx: &HookContext<'_>,
        _request: &aidaemon_core::ModelRequest,
    ) -> Option<ModelResponse> {
        Some(text("from the cache"))
    }
}

#[tokio::test]
async fn a_hook_can_answer_in_the_models_place() {
    let provider = Arc::new(ScriptedProvider::new([text("from the model")]));
    let agent = agent(provider.clone()).hook(Cached).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("from the cache"));
    assert!(provider.requests().is_empty());
}

struct RecoverWithFallback;
#[async_trait]
impl Hook for RecoverWithFallback {
    async fn on_model_error(
        &self,
        _ctx: &HookContext<'_>,
        _error: &ProviderError,
    ) -> ModelErrorDecision {
        ModelErrorDecision::Recover {
            response: text("from the fallback"),
        }
    }
}

#[tokio::test]
async fn a_hook_can_supply_the_response_of_a_failed_call() {
    let provider = Arc::new(ScriptedProvider::with_results([Err(ProviderError::Auth(
        "bad key".into(),
    ))]));
    let agent = agent(provider).hook(RecoverWithFallback).build().unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.final_text.as_deref(), Some("from the fallback"));
}

// ---------- what an exporter can see ----------

#[derive(Default)]
struct Observer {
    events: Mutex<Vec<Event>>,
    requests: Mutex<Vec<aidaemon_core::ModelRequest>>,
}
struct ArcObserver(Arc<Observer>);
impl aidaemon_core::EventSink for ArcObserver {
    fn emit(&self, _run: aidaemon_core::RunId, event: &Event) {
        self.0.events.lock().unwrap().push(event.clone());
    }
    fn observe_request(&self, _run: aidaemon_core::RunId, request: &aidaemon_core::ModelRequest) {
        self.0.requests.lock().unwrap().push(request.clone());
    }
}

struct AddsNote;
#[async_trait]
impl Hook for AddsNote {
    async fn before_model(
        &self,
        _ctx: &HookContext<'_>,
        request: &mut aidaemon_core::ModelRequest,
    ) {
        request.messages.push(Message::hook_note("added by a hook"));
    }
}

#[tokio::test]
async fn a_sink_sees_the_final_request_and_each_hook_decision() {
    let observer = Arc::new(Observer::default());
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write", json!({}))]),
        text("done"),
    ]));
    let agent = agent(provider)
        .tool(write_tool(Arc::default()))
        .hook(AddsNote)
        .hook(DenyWrite)
        .event_sink(ArcObserver(observer.clone()))
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    let requests = observer.requests.lock().unwrap();
    assert!(requests[0]
        .messages
        .iter()
        .any(|m| m.content == "added by a hook"));
    let events = observer.events.lock().unwrap();
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::ModelRequested { model, .. } if model == "m")));
    assert!(events.iter().any(|e| matches!(
        e,
        Event::HookDecided { hook, decision, reason, .. }
            if hook.contains("DenyWrite") && decision == "deny" && reason.as_deref() == Some("policy")
    )));
}

#[tokio::test]
async fn several_sinks_all_receive_the_events() {
    let (first, sink_one) = recording();
    let (second, sink_two) = recording();
    let provider = Arc::new(ScriptedProvider::new([text("done")]));
    let agent = agent(provider)
        .event_sink(sink_one)
        .event_sink(sink_two)
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert!(!first.lock().unwrap().is_empty());
    assert_eq!(first.lock().unwrap().len(), second.lock().unwrap().len());
}

#[tokio::test]
async fn a_child_run_names_its_parent() {
    let (events, sink) = recording();
    let provider = Arc::new(ScriptedProvider::new([text("done")]));
    let agent = agent(provider).event_sink(sink).build().unwrap();
    let parent = aidaemon_core::RunId::new();
    let state = RunState::new("a", "s", "go").with_parent(parent);

    agent
        .run_from(state, CancellationToken::new())
        .await
        .unwrap();

    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|e| matches!(e, Event::RunLinked { parent_run } if *parent_run == parent)));
}

// ---------- ending a run from a response ----------

struct StopOnResponse;
#[async_trait]
impl Hook for StopOnResponse {
    async fn after_model(
        &self,
        _ctx: &HookContext<'_>,
        _response: &mut ModelResponse,
    ) -> ModelDecision {
        ModelDecision::Stop {
            reason: "the budget is spent".into(),
        }
    }
}

#[tokio::test]
async fn a_response_hook_can_end_the_run_before_anything_acts_on_it() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(ScriptedProvider::new([tool_calls([call(
        "c1",
        "write",
        json!({}),
    )])]));
    let agent = agent(provider)
        .tool(write_tool(calls.clone()))
        .hook(StopOnResponse)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(
        out.status,
        RunStatus::Stopped {
            reason: "the budget is spent".into()
        }
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(out.state.messages.iter().all(|m| m.tool_calls.is_empty()));
}
