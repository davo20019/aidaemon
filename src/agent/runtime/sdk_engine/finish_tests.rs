//! Run-ending tool results on SDK turns: what each path ships, with which
//! task status and outcome, and what the ledger closeout records.

use std::sync::Arc;

use crate::events::{TaskEndData, TaskOutcome, TaskStatus};
use crate::testing::{MockProvider, MockTool, TestHarness};
use crate::traits::{Tool, ToolCall, ToolCallMetadata};
use crate::types::{ChannelContext, UserRole};

/// The latest task end recorded for `session`.
async fn task_end(harness: &TestHarness, session: &str) -> TaskEndData {
    harness
        .agent
        .event_store()
        .query_events_by_types(session, &[crate::events::EventType::TaskEnd], 5)
        .await
        .unwrap()
        .into_iter()
        .find_map(|event| event.parse_data::<TaskEndData>().ok())
        .expect("task end")
}

/// One model response calling every `(tool, arguments)` pair.
fn batch(calls: &[(&str, &str)]) -> crate::traits::ProviderResponse {
    let mut response = MockProvider::tool_call_response(calls[0].0, calls[0].1);
    response.tool_calls = calls
        .iter()
        .enumerate()
        .map(|(index, (name, arguments))| ToolCall {
            id: format!("call_synthetic_{index}"),
            name: name.to_string(),
            arguments: arguments.to_string(),
            extra_content: None,
        })
        .collect();
    response
}

async fn root(provider: MockProvider, tools: Vec<Arc<dyn Tool>>) -> TestHarness {
    crate::testing::setup_test_agent_root_with_extra_tools_and_llm_timeout(provider, tools, None)
        .await
        .unwrap()
}

async fn ask(harness: &TestHarness, session: &str, text: &str) -> String {
    harness
        .agent
        .handle_message(
            session,
            text,
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap()
}

/// A launch the tool moved to background tracking, as `terminal` and
/// `cli_agent` report it.
fn launcher(notifications: bool) -> MockTool {
    MockTool::new(
        "synthetic_launcher",
        "Starts a synthetic long job",
        "Started synthetic job 7; it keeps running.\nMore detail on the second line.",
    )
    .with_metadata(ToolCallMetadata {
        outcome_status: Some(crate::traits::ToolOutcomeStatus::Backgrounded),
        background_started: true,
        timed_out: true,
        completion_notifications_enabled: notifications,
        ..ToolCallMetadata::default()
    })
}

/// A background launch without completion notifications ends the run with
/// the launch's own first line, as partial work, without another model call.
#[tokio::test]
async fn background_launch_without_notifications_ends_with_its_first_line() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_launcher", "{}"),
        MockProvider::text_response("UNREACHED model narration"),
    ]);
    let harness = root(provider, vec![Arc::new(launcher(false))]).await;
    let reply = ask(&harness, "finish-launch-quiet", "Start the synthetic job").await;

    assert_eq!(reply, "Started synthetic job 7; it keeps running.");
    assert_eq!(harness.provider.call_count().await, 1);
    let end = task_end(&harness, "finish-launch-quiet").await;
    assert_eq!(end.status, TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), TaskOutcome::Partial);
}

/// With completion notifications on, the reply is the friendly handoff and
/// the assistant message is recorded as a background handoff.
#[tokio::test]
async fn background_launch_with_notifications_ends_with_the_friendly_handoff() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_launcher", "{}"),
        MockProvider::text_response("UNREACHED model narration"),
    ]);
    let harness = root(provider, vec![Arc::new(launcher(true))]).await;
    let reply = ask(&harness, "finish-launch-notify", "Start the synthetic job").await;

    assert_eq!(
        reply,
        crate::agent::post_task::build_friendly_background_handoff("Start the synthetic job")
    );
    assert_eq!(harness.provider.call_count().await, 1);
    let end = task_end(&harness, "finish-launch-notify").await;
    assert_eq!(end.status, TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), TaskOutcome::Partial);
}

/// A tool's direct response answers a root turn's lone call: it is the
/// reply, the task succeeds, and no model call follows.
#[tokio::test]
async fn lone_direct_response_is_the_reply() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_connector", "{}"),
        MockProvider::text_response("UNREACHED model narration"),
    ]);
    let harness = root(provider, vec![Arc::new(connector())]).await;
    let reply = ask(&harness, "finish-direct", "Connect the synthetic service").await;

    assert_eq!(reply, CONNECTOR_REPLY);
    assert_eq!(harness.provider.call_count().await, 1);
    let end = task_end(&harness, "finish-direct").await;
    assert_eq!(end.status, TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), TaskOutcome::Succeeded);
}

/// A direct response that shares its model response with another call is not
/// the answer: the model replies after the batch.
#[tokio::test]
async fn direct_response_in_a_batch_lets_the_model_answer() {
    let provider = MockProvider::with_responses(vec![
        batch(&[("synthetic_connector", "{}"), ("synthetic_probe", "{}")]),
        MockProvider::text_response("Connected, and the probe is fine."),
    ]);
    let harness = root(
        provider,
        vec![
            Arc::new(connector()),
            Arc::new(MockTool::new("synthetic_probe", "Probes", "probe ok")),
        ],
    )
    .await;
    let reply = ask(&harness, "finish-direct-batch", "Connect and probe").await;

    assert_eq!(reply, "Connected, and the probe is fine.");
    assert_eq!(harness.provider.call_count().await, 2);
}

const CONNECTOR_REPLY: &str = "Synthetic service connected. You can use it now.";

/// A connector whose success message answers the user directly, as
/// `manage_oauth connect` does.
struct Connector;

fn connector() -> Connector {
    Connector
}

#[async_trait::async_trait]
impl Tool for Connector {
    fn name(&self) -> &str {
        "synthetic_connector"
    }

    fn description(&self) -> &str {
        "Connects a synthetic service"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "synthetic_connector",
            "description": "Connects a synthetic service",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
        })
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        Ok(CONNECTOR_REPLY.to_string())
    }

    async fn call_with_execution_context(
        &self,
        _arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<crate::types::StatusUpdate>>,
        _facts: crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        Ok(crate::traits::ToolCallOutcome {
            output: CONNECTOR_REPLY.to_string(),
            metadata: ToolCallMetadata {
                outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
                direct_response: Some(CONNECTOR_REPLY.to_string()),
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }
}

impl crate::traits::ToolSemantics for Connector {}

/// A run a tool ended still closes against the replayed ledger: the task
/// end is reconciled against durable receipt state.
#[tokio::test]
async fn ledger_closeout_runs_on_finish() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_connector", "{}"),
        MockProvider::text_response("UNREACHED model narration"),
    ]);
    let harness = root(provider, vec![Arc::new(connector())]).await;
    ask(&harness, "finish-ledger", "Connect the synthetic service").await;

    let end = task_end(&harness, "finish-ledger").await;
    let points = harness
        .agent
        .event_store()
        .query_decision_points("finish-ledger", &end.task_id)
        .await
        .unwrap();
    let reconciled = points
        .iter()
        .filter_map(|event| event.parse_data::<crate::events::DecisionPointData>().ok())
        .find(|point| point.metadata["condition"] == "task_finalization_reconciled")
        .expect("the ledger closeout ran");
    assert_eq!(reconciled.metadata["requested_outcome"], "succeeded");
    assert_eq!(reconciled.metadata["requested_status"], "completed");
    assert_eq!(reconciled.metadata["tool_calls_count"], 1);
}

/// A run in which the owner declined something never ends `Succeeded`, even
/// when a tool finishes it as succeeded.
#[test]
fn declined_run_never_finishes_succeeded() {
    use super::TurnVerdict;
    use aidaemon_core as sdk;
    let mut state = sdk::RunState::new("aidaemon", "s", "hi");
    state.declined =
        vec![sdk::EffectKey::path(sdk::EffectClass::Delete, "/tmp/synthetic").unwrap()];
    let outcome = sdk::RunOutcome {
        status: sdk::RunStatus::Finished {
            outcome: sdk::FinishOutcome::Succeeded,
            tool: "synthetic_connector".into(),
        },
        final_text: Some("All connected.".into()),
        state,
    };
    let (reply, status, task_outcome) = TurnVerdict::default().resolve(&outcome);
    assert_eq!(reply, "All connected.");
    assert_eq!(status, TaskStatus::Completed);
    assert_ne!(task_outcome, TaskOutcome::Succeeded);
}

/// A finish without a reply never ships an empty answer: the finishing
/// call's own output is the reply.
#[tokio::test]
async fn finish_without_a_reply_ships_the_tool_output() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_finisher", "{}"),
        MockProvider::text_response("UNREACHED model narration"),
    ]);
    let finisher = MockTool::new(
        "synthetic_finisher",
        "Finishes the synthetic task",
        "Synthetic task filed for review.",
    )
    .with_finish(aidaemon_core::Finish::new(
        aidaemon_core::FinishOutcome::Partial,
        None,
    ));
    let harness = root(provider, vec![Arc::new(finisher)]).await;
    let reply = ask(&harness, "finish-no-reply", "File the synthetic task").await;

    assert_eq!(reply, "Synthetic task filed for review.");
    assert_eq!(harness.provider.call_count().await, 1);
    let end = task_end(&harness, "finish-no-reply").await;
    assert_eq!(end.status, TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), TaskOutcome::Partial);
}

/// The kernel-side fallback for a finish with no reply (a tool outside the
/// daemon dispatcher): the finishing call's output, never an empty reply.
#[test]
fn finished_run_without_text_falls_back_to_the_finishing_output() {
    use aidaemon_core as sdk;
    let mut state = sdk::RunState::new("aidaemon", "s", "hi");
    state.messages.push(sdk::Message::tool_result(
        "c1",
        "Remembered: the synthetic preference.",
    ));
    state.receipts.push(sdk::Receipt {
        run_id: state.run_id,
        call_id: "c1".into(),
        tool: "remember".into(),
        arguments: serde_json::json!({}),
        turn: 1,
        stage: sdk::InvocationStage::Dispatched,
        status: sdk::ReceiptStatus::Succeeded,
        read_only: false,
        duration_ms: 1,
        reason: None,
        details: serde_json::Value::Null,
        source: sdk::SourceTrust::Trusted,
        after_untrusted: false,
    });
    let outcome = sdk::RunOutcome {
        status: sdk::RunStatus::Finished {
            outcome: sdk::FinishOutcome::Succeeded,
            tool: "remember".into(),
        },
        final_text: None,
        state,
    };
    let (reply, status, task_outcome) = super::TurnVerdict::default().resolve(&outcome);
    assert_eq!(reply, "Remembered: the synthetic preference.");
    assert_eq!(status, TaskStatus::Completed);
    assert_eq!(task_outcome, TaskOutcome::Succeeded);
}
