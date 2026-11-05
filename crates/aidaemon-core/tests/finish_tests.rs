//! A tool can end the run: `ToolOutput::finish`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, Finish, FinishOutcome, FnTool, Hook, HookContext, ReceiptStatus, RunOutcome, RunStatus,
    StopDecision, Tool, ToolAnnotations, ToolOutput,
};
use async_trait::async_trait;
use serde_json::{json, Value};

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
                Ok(ToolOutput::text(format!("{name} done")).finish(Finish {
                    reply: Some(reply.to_string()),
                    outcome,
                }))
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
