//! A tool can end the run: `ToolOutput::finish`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::testing::{ScriptedProvider, call, spec, text, tool_calls};
use aidaemon_core::{
    Agent, Finish, FinishOutcome, FnTool, Hook, HookContext, InterruptKind, InterruptResolver,
    Interruption, Limits, Receipt, ReceiptStatus, Resolution, RunOutcome, RunState, RunStatus,
    StopDecision, Tool, ToolAnnotations, ToolCall, ToolDecision, ToolInfo, ToolOutput,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

fn base(provider: ScriptedProvider) -> aidaemon_core::AgentBuilder {
    Agent::builder("test-agent")
        .allow_unguarded()
        .model("scripted")
        .provider(provider)
}

/// A tool that finishes the run with `outcome` and `reply`, after `delay`.
fn finisher(
    name: &'static str,
    annotations: ToolAnnotations,
    outcome: FinishOutcome,
    reply: &'static str,
    delay: Duration,
) -> impl Tool {
    FnTool::new(
        spec(name, "finishes the run", &[]),
        annotations,
        move |_: Value| {
            let outcome = outcome.clone();
            async move {
                tokio::time::sleep(delay).await;
                Ok(ToolOutput::text(format!("{name} done"))
                    .finish(Finish::new(outcome, Some(reply.to_string()))))
            }
        },
    )
}

fn counter(name: &'static str, calls: Arc<AtomicUsize>) -> impl Tool {
    FnTool::new(
        spec(name, "counts calls", &[]),
        ToolAnnotations::WRITE,
        move |_: Value| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutput::text("counted"))
            }
        },
    )
}

#[tokio::test]
async fn finish_ends_run_after_batch() {
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([
            call("c1", "finisher", json!({})),
            call("c2", "counter", json!({})),
        ]),
        text("the model must not be asked again"),
    ]));
    let counted = Arc::new(AtomicUsize::new(0));
    let agent = Agent::builder("test-agent")
        .allow_unguarded()
        .model("scripted")
        .provider(provider.clone())
        .tool(finisher(
            "finisher",
            ToolAnnotations::WRITE,
            FinishOutcome::Succeeded,
            "all done",
            Duration::ZERO,
        ))
        .tool(counter("counter", counted.clone()))
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(
        out.status,
        RunStatus::Finished {
            outcome: FinishOutcome::Succeeded,
            tool: "finisher".to_string(),
        }
    );
    assert_eq!(out.final_text.as_deref(), Some("all done"));
    // The whole batch ran: the call after the finishing one too.
    assert_eq!(counted.load(Ordering::SeqCst), 1);
    let receipts: Vec<(&str, ReceiptStatus)> = out
        .receipts()
        .iter()
        .map(|r| (r.call_id.as_str(), r.status))
        .collect();
    assert_eq!(
        receipts,
        vec![
            ("c1", ReceiptStatus::Succeeded),
            ("c2", ReceiptStatus::Succeeded)
        ]
    );
    // No second model call.
    assert_eq!(provider.requests().len(), 1);
    assert_eq!(provider.remaining(), 1);
}

#[tokio::test]
async fn first_finish_wins() {
    // Both read-only, so they run concurrently; the first in call order is
    // slower, yet it still wins, because results are judged in call order.
    let agent = base(ScriptedProvider::new([
        tool_calls([
            call("c1", "slow_finish", json!({})),
            call("c2", "fast_finish", json!({})),
        ]),
        text("never"),
    ]))
    .tool(finisher(
        "slow_finish",
        ToolAnnotations::READ_ONLY,
        FinishOutcome::Partial,
        "first",
        Duration::from_millis(30),
    ))
    .tool(finisher(
        "fast_finish",
        ToolAnnotations::READ_ONLY,
        FinishOutcome::Blocked,
        "second",
        Duration::ZERO,
    ))
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(
        out.status,
        RunStatus::Finished {
            outcome: FinishOutcome::Partial,
            tool: "slow_finish".to_string(),
        }
    );
    assert_eq!(out.final_text.as_deref(), Some("first"));
    assert_eq!(out.receipts().len(), 2);
}

/// Records run ends and counts `on_stop` calls.
#[derive(Default)]
struct Log {
    run_ends: Mutex<Vec<(RunStatus, Option<String>)>>,
    stops: AtomicUsize,
}
struct LogHook(Arc<Log>);

#[async_trait]
impl Hook for LogHook {
    async fn on_stop(&self, _ctx: &HookContext<'_>, _text: &str) -> StopDecision {
        self.0.stops.fetch_add(1, Ordering::SeqCst);
        StopDecision::Continue {
            message: "not done yet".into(),
        }
    }

    async fn on_run_end(&self, _ctx: &HookContext<'_>, outcome: &RunOutcome) {
        self.0
            .run_ends
            .lock()
            .unwrap()
            .push((outcome.status.clone(), outcome.final_text.clone()));
    }
}

#[tokio::test]
async fn finished_run_reports_outcome_to_on_run_end() {
    let log = Arc::new(Log::default());
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "blocker", json!({}))]),
        text("never"),
    ]))
    .tool(finisher(
        "blocker",
        ToolAnnotations::WRITE,
        FinishOutcome::Blocked,
        "waiting on the owner",
        Duration::ZERO,
    ))
    .hook(LogHook(log.clone()))
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    let expected = RunStatus::Finished {
        outcome: FinishOutcome::Blocked,
        tool: "blocker".to_string(),
    };
    assert_eq!(out.status, expected);
    // A stop hook that would reject every answer is not consulted.
    assert_eq!(log.stops.load(Ordering::SeqCst), 0);
    assert_eq!(
        *log.run_ends.lock().unwrap(),
        vec![(expected, Some("waiting on the owner".to_string()))]
    );
}

#[test]
fn finish_constructors_set_outcome_and_reply() {
    let done = Finish::succeeded("done");
    assert_eq!(done.outcome, FinishOutcome::Succeeded);
    assert_eq!(done.reply.as_deref(), Some("done"));
    assert_eq!(Finish::partial("half").outcome, FinishOutcome::Partial);
    assert_eq!(Finish::blocked("stuck").outcome, FinishOutcome::Blocked);
    assert_eq!(Finish::new(FinishOutcome::Succeeded, None).reply, None);
}

/// Interrupts every call to `send`.
struct AskBeforeSend;

#[async_trait]
impl Hook for AskBeforeSend {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        _info: &ToolInfo,
    ) -> ToolDecision {
        if call.name == "send" {
            ToolDecision::Interrupt {
                reason: "send needs approval".into(),
                kind: InterruptKind::Untrusted,
            }
        } else {
            ToolDecision::Allow
        }
    }
}

struct ApproveAll;

#[async_trait]
impl InterruptResolver for ApproveAll {
    async fn resolve(&self, _: &RunState, _: &Interruption) -> Option<Resolution> {
        Some(Resolution::Approve)
    }
}

/// A batch whose finishing call runs before a call that needs approval.
fn finish_then_send(provider: Arc<ScriptedProvider>, sent: Arc<AtomicUsize>) -> Agent {
    Agent::builder("test-agent")
        .allow_unguarded()
        .model("scripted")
        .provider(provider)
        .tool(finisher(
            "blocker",
            ToolAnnotations::WRITE,
            FinishOutcome::Blocked,
            "waiting on the owner",
            Duration::ZERO,
        ))
        .tool(counter("send", sent))
        .hook(AskBeforeSend)
        .build()
        .unwrap()
}

fn finish_then_send_script() -> Arc<ScriptedProvider> {
    Arc::new(ScriptedProvider::new([
        tool_calls([
            call("c1", "blocker", json!({})),
            call("c2", "send", json!({})),
        ]),
        text("the model must not be asked again"),
    ]))
}

fn blocked_by(tool: &str) -> RunStatus {
    RunStatus::Finished {
        outcome: FinishOutcome::Blocked,
        tool: tool.to_string(),
    }
}

#[tokio::test]
async fn finish_survives_approval_interrupt() {
    let provider = finish_then_send_script();
    let sent = Arc::new(AtomicUsize::new(0));
    let agent = finish_then_send(provider.clone(), sent.clone());

    let out = agent
        .run_resolving(
            RunState::new("test-agent", "s", "go"),
            CancellationToken::new(),
            &ApproveAll,
        )
        .await
        .unwrap();

    assert_eq!(out.status, blocked_by("blocker"));
    assert_eq!(out.final_text.as_deref(), Some("waiting on the owner"));
    assert_eq!(sent.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn finish_survives_resume() {
    let provider = finish_then_send_script();
    let sent = Arc::new(AtomicUsize::new(0));
    let agent = finish_then_send(provider.clone(), sent.clone());

    let paused = agent.run("s", "go").await.unwrap();
    assert!(matches!(paused.status, RunStatus::Interrupted { .. }));
    let json = serde_json::to_string(&paused.state).unwrap();
    let restored: RunState = serde_json::from_str(&json).unwrap();

    let out = agent.resume(restored, Resolution::Approve).await.unwrap();

    assert_eq!(out.status, blocked_by("blocker"));
    assert_eq!(out.final_text.as_deref(), Some("waiting on the owner"));
    assert_eq!(sent.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn recorded_finish_wins_over_mid_batch_limit() {
    let counted = Arc::new(AtomicUsize::new(0));
    let agent = base(ScriptedProvider::new([
        tool_calls([
            call("c1", "blocker", json!({})),
            call("c2", "counter", json!({})),
        ]),
        text("never"),
    ]))
    .tool(finisher(
        "blocker",
        ToolAnnotations::WRITE,
        FinishOutcome::Blocked,
        "waiting on the owner",
        Duration::ZERO,
    ))
    .tool(counter("counter", counted.clone()))
    .limits(Limits {
        max_tool_calls: Some(1),
        ..Limits::default()
    })
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.status, blocked_by("blocker"));
    assert_eq!(out.final_text.as_deref(), Some("waiting on the owner"));
    // The limit still stopped the rest of the batch.
    assert_eq!(counted.load(Ordering::SeqCst), 0);
    let statuses: Vec<ReceiptStatus> = out.receipts().iter().map(|r| r.status).collect();
    assert_eq!(
        statuses,
        vec![ReceiptStatus::Succeeded, ReceiptStatus::Skipped]
    );
}

/// Removes any finish from a tool's output.
struct StripFinish;

#[async_trait]
impl Hook for StripFinish {
    async fn after_tool(
        &self,
        _ctx: &HookContext<'_>,
        _receipt: &Receipt,
        output: &mut ToolOutput,
    ) {
        output.finish = None;
    }
}

#[tokio::test]
async fn after_tool_hook_can_strip_finish() {
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "finisher", json!({}))]),
        text("kept going"),
    ]));
    let agent = Agent::builder("test-agent")
        .allow_unguarded()
        .model("scripted")
        .provider(provider.clone())
        .tool(finisher(
            "finisher",
            ToolAnnotations::WRITE,
            FinishOutcome::Succeeded,
            "all done",
            Duration::ZERO,
        ))
        .hook(StripFinish)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.status, RunStatus::Completed);
    assert_eq!(out.final_text.as_deref(), Some("kept going"));
    assert_eq!(provider.requests().len(), 2);
}

/// A run in which the owner declined something never reports success: a
/// tool's `Succeeded` finish is reported as `Partial`, so no host can say
/// "Done" over the decline.
#[tokio::test]
async fn declined_run_finish_succeeded_reports_partial() {
    let agent = |provider: ScriptedProvider| {
        base(provider)
            .tool(finisher(
                "finisher",
                ToolAnnotations::WRITE,
                FinishOutcome::Succeeded,
                "all done",
                Duration::ZERO,
            ))
            .build()
            .unwrap()
    };
    let script = || ScriptedProvider::new([tool_calls([call("c1", "finisher", json!({}))])]);

    // Control: without a decline the finish stays `Succeeded`.
    let out = agent(script()).run("s", "go").await.unwrap();
    assert_eq!(
        out.status,
        RunStatus::Finished {
            outcome: FinishOutcome::Succeeded,
            tool: "finisher".to_string(),
        }
    );

    let mut state = RunState::new("test-agent", "s", "go");
    state.declined.push(aidaemon_core::EffectKey {
        class: aidaemon_core::EffectClass::Egress,
        target: "h.example.invalid:443".to_string(),
    });
    let out = agent(script())
        .run_from(state, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        out.status,
        RunStatus::Finished {
            outcome: FinishOutcome::Partial,
            tool: "finisher".to_string(),
        }
    );
    assert_eq!(out.final_text.as_deref(), Some("all done"));
}
