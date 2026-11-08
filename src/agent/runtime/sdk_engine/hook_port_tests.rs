//! Forced text replies, budget stops and stuck notices through the kernel's
//! hooks, run against the real loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use aidaemon_core as sdk;
use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::budgets::GOAL_BUDGET_EXHAUSTED;
use super::hooks::{BudgetGate, StuckNotice, TextOnlyNext};
use super::{DaemonCall, KernelNotice, TaskOutcome, TaskStatus, TurnVerdict};

fn builder(provider: Arc<ScriptedProvider>) -> sdk::AgentBuilder {
    sdk::Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(provider)
}

fn echo() -> impl sdk::Tool {
    sdk::FnTool::new(
        spec("echo", "Echo", &[]),
        sdk::ToolAnnotations::READ_ONLY,
        |_: Value| async { Ok(sdk::ToolOutput::text("same")) },
    )
}

#[tokio::test]
async fn a_forced_text_reply_applies_to_one_request() {
    let flag = Arc::new(AtomicBool::new(true));
    let provider = Arc::new(ScriptedProvider::new([text("done")]));
    let agent = builder(provider.clone())
        .hook(TextOnlyNext { flag: flag.clone() })
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert_eq!(provider.requests()[0].tool_choice, sdk::ToolChoice::None);
    assert!(!flag.load(Ordering::SeqCst), "it applies to one request");
}

#[tokio::test]
async fn requests_keep_their_tool_choice_unless_a_text_reply_is_forced() {
    let provider = Arc::new(ScriptedProvider::new([text("done")]));
    let agent = builder(provider.clone())
        .hook(TextOnlyNext {
            flag: Arc::new(AtomicBool::new(false)),
        })
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert_eq!(provider.requests()[0].tool_choice, sdk::ToolChoice::Auto);
}

/// Answers every budget check with `decision`.
fn budget_server(decision: sdk::ModelDecision) -> mpsc::Sender<DaemonCall> {
    let (tx, mut rx) = mpsc::channel(4);
    tokio::spawn(async move {
        while let Some(call) = rx.recv().await {
            if let DaemonCall::Budget(check) = call {
                let _ = check.reply.send(decision.clone());
            }
        }
    });
    tx
}

#[tokio::test]
async fn a_used_up_budget_stops_the_run_and_shows_the_budget_reply() {
    let provider = Arc::new(ScriptedProvider::new([tool_calls([call(
        "c1",
        "echo",
        json!({}),
    )])]));
    let agent = builder(provider)
        .tool(echo())
        .hook(BudgetGate {
            calls: budget_server(sdk::ModelDecision::Stop {
                reason: GOAL_BUDGET_EXHAUSTED.into(),
            }),
        })
        .build()
        .unwrap();

    let outcome = agent.run("s", "go").await.unwrap();

    assert!(outcome.state.receipts.is_empty(), "nothing acted on it");
    let verdict = TurnVerdict::default();
    verdict.record_budget_stop("Out of budget.".into(), TaskOutcome::Failed);
    let (reply, status, task_outcome) = verdict.resolve(&outcome);
    assert_eq!(reply, "Out of budget.");
    assert_eq!(status, TaskStatus::Completed);
    assert_eq!(task_outcome, TaskOutcome::Failed);
}

#[tokio::test]
async fn a_run_within_budget_carries_on() {
    let provider = Arc::new(ScriptedProvider::new([text("done")]));
    let agent = builder(provider)
        .hook(BudgetGate {
            calls: budget_server(sdk::ModelDecision::Accept),
        })
        .build()
        .unwrap();

    let outcome = agent.run("s", "go").await.unwrap();

    assert_eq!(outcome.final_text.as_deref(), Some("done"));
}

#[tokio::test]
async fn being_stuck_is_recorded_and_the_kernel_still_decides() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let provider = Arc::new(ScriptedProvider::default().then_repeat(tool_calls([call(
        "c1",
        "echo",
        json!({}),
    )])));
    let agent = builder(provider)
        .tool(echo())
        .hook(StuckNotice { notices: tx })
        .build()
        .unwrap();

    let outcome = agent.run("s", "go").await.unwrap();

    assert!(matches!(outcome.status, sdk::RunStatus::Stuck { .. }));
    assert!(matches!(rx.try_recv(), Ok(KernelNotice::Stuck)));
}
