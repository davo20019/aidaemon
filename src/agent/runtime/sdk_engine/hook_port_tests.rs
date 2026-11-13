//! Forced text replies, budget stops and stuck notices through the kernel's
//! hooks, run against the real loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use aidaemon_core as sdk;
use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::budgets::GOAL_BUDGET_EXHAUSTED;
use super::hooks::{BudgetGate, CutoffAdjust, ModelPin, ModelRetry, StuckNotice, TextOnlyNext};
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

/// Answers every model-failure check with `decision`.
fn model_error_server(decision: sdk::ModelErrorDecision) -> mpsc::Sender<DaemonCall> {
    let (tx, mut rx) = mpsc::channel(4);
    tokio::spawn(async move {
        while let Some(call) = rx.recv().await {
            if let DaemonCall::ModelError(check) = call {
                let _ = check.reply.send(decision.clone());
            }
        }
    });
    tx
}

fn failing_once() -> Arc<ScriptedProvider> {
    Arc::new(ScriptedProvider::with_results([
        Err(sdk::ProviderError::Transient("timed out".into())),
        Ok(text("recovered")),
    ]))
}

#[tokio::test]
async fn a_failed_model_call_is_retried_when_the_daemon_says_so() {
    let agent = builder(failing_once())
        .hook(ModelRetry {
            calls: model_error_server(sdk::ModelErrorDecision::Retry {
                after: std::time::Duration::ZERO,
            }),
        })
        .build()
        .unwrap();

    let outcome = agent.run("s", "go").await.unwrap();

    assert_eq!(outcome.final_text.as_deref(), Some("recovered"));
}

#[tokio::test]
async fn a_failed_model_call_ends_the_run_when_the_daemon_gives_up() {
    let agent = builder(failing_once())
        .hook(ModelRetry {
            calls: model_error_server(sdk::ModelErrorDecision::Fail),
        })
        .build()
        .unwrap();

    let outcome = agent.run("s", "go").await.unwrap();

    assert!(matches!(
        outcome.status,
        sdk::RunStatus::ProviderFailed { .. }
    ));
}

fn cut_off(output_tokens: u64) -> sdk::ModelResponse {
    sdk::ModelResponse {
        usage: sdk::Usage {
            output_tokens,
            ..Default::default()
        },
        note: Some("response truncated at the token limit".into()),
        ..Default::default()
    }
}

/// Sends an empty reply back, as the daemon's reply repair does.
struct RetryEmpty;
#[async_trait::async_trait]
impl sdk::Hook for RetryEmpty {
    async fn on_stop(&self, _ctx: &sdk::HookContext<'_>, final_text: &str) -> sdk::StopDecision {
        if final_text.is_empty() {
            sdk::StopDecision::Continue {
                message: "answer".into(),
            }
        } else {
            sdk::StopDecision::Accept
        }
    }
}

#[tokio::test]
async fn a_cutoff_raises_output_room_then_lowers_reasoning() {
    let provider = Arc::new(ScriptedProvider::new([
        cut_off(1_000),
        cut_off(2_000),
        text("done"),
        text("after"),
    ]));
    let agent = builder(provider.clone())
        .hook(RetryEmpty)
        .hook(CutoffAdjust::default())
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    let requests = provider.requests();
    assert_eq!(requests[0].max_output_tokens, None);
    assert_eq!(requests[1].max_output_tokens, Some(2_000));
    assert_eq!(requests[1].options.reasoning_effort, None);
    assert_eq!(requests[2].max_output_tokens, Some(4_000));
    assert_eq!(requests[2].options.reasoning_effort.as_deref(), Some("low"));
}

#[tokio::test]
async fn output_clears_the_cutoff_count() {
    let hook = Arc::new(CutoffAdjust::default());
    let provider = Arc::new(ScriptedProvider::new([cut_off(1_000), text("done")]));
    let agent = builder(provider)
        .hook(RetryEmpty)
        .hook_arc(hook.clone())
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert_eq!(hook.cutoffs(), 0);
}

#[tokio::test]
async fn a_task_without_a_pinned_model_keeps_its_model() {
    let provider = Arc::new(ScriptedProvider::new([text("done")]));
    let agent = builder(provider.clone())
        .hook(ModelPin {
            task_id: "synthetic-task".into(),
            computer_use: None,
        })
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert_eq!(provider.requests()[0].model, "m");
}

#[cfg(feature = "computer_use")]
#[tokio::test]
async fn a_pinned_task_uses_its_pinned_model() {
    use crate::tools::computer_use::pin_registry::ComputerUsePinRegistry;
    let task = "synthetic-pinned-task";
    ComputerUsePinRegistry::shared()
        .pin(task, "vision-model")
        .await;
    let provider = Arc::new(ScriptedProvider::new([text("done")]));
    let agent = builder(provider.clone())
        .hook(ModelPin {
            task_id: task.into(),
            computer_use: Some(Arc::new(crate::agent::computer_use::ComputerUsePins)),
        })
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert_eq!(provider.requests()[0].model, "vision-model");
    ComputerUsePinRegistry::shared().clear_task(task).await;
}
