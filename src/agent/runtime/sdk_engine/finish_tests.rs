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
        background_handoff: true,
        timed_out: true,
        completion_notifications_enabled: notifications,
        ..ToolCallMetadata::default()
    })
    .with_finish(handoff())
}

/// The finish a tool hands the turn to the background with
/// (`ToolCallOutcome::handed_to_background`).
fn handoff() -> aidaemon_core::Finish {
    crate::traits::ToolCallOutcome::from_output(String::new())
        .handed_to_background()
        .finish
        .unwrap()
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
async fn oauth_direct_response_is_final_text() {
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

    async fn execute(
        &self,
        _arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let facts = facts.clone();

        let outcome = crate::traits::ToolCallOutcome {
            output: CONNECTOR_REPLY.to_string(),
            metadata: ToolCallMetadata {
                outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
                ..ToolCallMetadata::default()
            },
            finish: None,
        };
        // As `manage_oauth connect`: the answer when it is the lone call of
        // a root turn.
        Ok(if facts.lone_root_call {
            outcome.with_finish(aidaemon_core::Finish::succeeded(CONNECTOR_REPLY))
        } else {
            outcome
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

/// A tool that hands the turn to a background job ends the run from its
/// typed outcome, whatever its output says: no "moved to background" wording
/// is needed. As `spawn_agent` does, it sets no process-launch receipt.
#[tokio::test]
async fn background_detach_finishes_without_text_match() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_delegate", "{}"),
        MockProvider::text_response("UNREACHED model narration"),
    ]);
    let delegate = MockTool::new(
        "synthetic_delegate",
        "Hands a synthetic report to a helper",
        "A helper is preparing the synthetic report; it will arrive here when ready.",
    )
    .with_metadata(ToolCallMetadata {
        background_handoff: true,
        ..ToolCallMetadata::default()
    })
    .with_finish(handoff());
    let harness = root(provider, vec![Arc::new(delegate)]).await;
    let reply = ask(&harness, "finish-delegate", "Prepare the synthetic report").await;

    assert_eq!(
        reply,
        "A helper is preparing the synthetic report; it will arrive here when ready."
    );
    assert_eq!(harness.provider.call_count().await, 1);
    let end = task_end(&harness, "finish-delegate").await;
    assert_eq!(end.status, TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), TaskOutcome::Partial);
}

/// Work the model declared beyond the background job keeps the run going:
/// the handoff does not end it while a second checklist item is open.
#[tokio::test]
async fn background_handoff_with_more_declared_work_lets_the_model_continue() {
    let checklist = r#"{"items":[
        {"text":"Start the job","status":"in_progress"},
        {"text":"Tidy up","status":"pending"}
    ]}"#;
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", checklist),
        MockProvider::tool_call_response("synthetic_launcher", "{}"),
        MockProvider::text_response("The job is running and the rest is done."),
    ]);
    let harness = root(provider, vec![Arc::new(launcher(false))]).await;
    let reply = ask(&harness, "finish-inline", "Start the job, then tidy up").await;

    assert_eq!(reply, "The job is running and the rest is done.");
    assert_eq!(harness.provider.call_count().await, 3);
}

/// The job as the only open item is a plain handoff: nothing is left to do
/// inline, so the run ends on the tool's finish.
#[tokio::test]
async fn background_handoff_as_the_only_open_item_ends_the_run() {
    let checklist = r#"{"items":[
        {"text":"Start the job","status":"in_progress","mutation_effects":["process_state"]}
    ]}"#;
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", checklist),
        MockProvider::tool_call_response("synthetic_launcher", "{}"),
    ]);
    let harness = root(provider, vec![Arc::new(launcher(false))]).await;
    ask(&harness, "finish-one-item", "Start the job").await;

    assert_eq!(harness.provider.call_count().await, 2);
}

/// Only a tool's typed finish ends a run: output that merely reads like a
/// background handoff (a page, a log line) is data, and the model answers.
#[tokio::test]
async fn background_phrase_in_output_does_not_end_the_run() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_reader", "{}"),
        MockProvider::text_response("The log says a job moved to the background earlier."),
    ]);
    let reader = MockTool::new(
        "synthetic_reader",
        "Reads a synthetic log",
        "Moved to background (pid=41). Worker started in background; helper spawned in background.",
    );
    let harness = root(provider, vec![Arc::new(reader)]).await;
    let reply = ask(&harness, "finish-phrase", "What does the log say?").await;

    assert_eq!(reply, "The log says a job moved to the background earlier.");
    assert_eq!(harness.provider.call_count().await, 2);
}

/// A handoff stands even when the handed-off work's description reads like a
/// failure (a sub-agent sent to fix an error): the work is running.
#[tokio::test]
async fn handoff_whose_output_reads_like_a_failure_still_ends_the_run() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_delegate", "{}"),
        MockProvider::text_response("UNREACHED model narration"),
    ]);
    let delegate = MockTool::new(
        "synthetic_delegate",
        "Hands a fix to a helper",
        "Error: build failed. A helper is fixing it and will report when done.",
    )
    .with_metadata(ToolCallMetadata {
        background_handoff: true,
        ..ToolCallMetadata::default()
    })
    .with_finish(handoff());
    let harness = root(provider, vec![Arc::new(delegate)]).await;
    let reply = ask(&harness, "finish-delegate-error", "Fix the synthetic build").await;

    assert_eq!(
        reply,
        "Error: build failed. A helper is fixing it and will report when done."
    );
    assert_eq!(harness.provider.call_count().await, 1);
}

/// Text is not the judge, the tool's type is: a handoff the tool itself
/// typed as a permanent failure hands nothing off, and the model goes on.
#[tokio::test]
async fn handoff_typed_as_a_permanent_failure_does_not_end_the_run() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_delegate", "{}"),
        MockProvider::text_response("The helper could not start; here is what I found."),
    ]);
    let delegate = MockTool::new(
        "synthetic_delegate",
        "Hands a fix to a helper",
        "The helper was refused before it started.",
    )
    .with_metadata(ToolCallMetadata {
        outcome_status: Some(crate::traits::ToolOutcomeStatus::FailedPermanent),
        background_handoff: true,
        ..ToolCallMetadata::default()
    })
    .with_finish(handoff());
    let harness = root(provider, vec![Arc::new(delegate)]).await;
    let reply = ask(
        &harness,
        "finish-delegate-typed-failure",
        "Fix the synthetic build",
    )
    .await;

    // The run did not end on the tool's finish: the model was asked again.
    assert_ne!(reply, "The helper was refused before it started.");
    assert!(harness.provider.call_count().await >= 2);
}

/// A send whose first attempt reports `first`, then succeeds.
struct FlakySend {
    executions: Arc<std::sync::atomic::AtomicUsize>,
    first: ToolCallMetadata,
}

impl crate::traits::ToolSemantics for FlakySend {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> crate::traits::ToolCallSemantics {
        crate::traits::ToolCallSemantics::mutation_with(
            crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY,
        )
    }

    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Trusted
    }
}

#[async_trait::async_trait]
impl Tool for FlakySend {
    fn name(&self) -> &str {
        "flaky_send"
    }

    fn description(&self) -> &str {
        "Send a message"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "flaky_send",
            "description": "Send a message",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
        })
    }

    async fn execute(
        &self,
        _arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        use std::sync::atomic::Ordering;
        let first = self.executions.fetch_add(1, Ordering::SeqCst) == 0;
        Ok(crate::traits::ToolCallOutcome {
            output: if first { "not delivered" } else { "delivered" }.to_string(),
            metadata: if first {
                self.first.clone()
            } else {
                ToolCallMetadata {
                    outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
                    ..ToolCallMetadata::default()
                }
            },
            finish: None,
        })
    }
}

async fn executions_after_three_identical_sends(session: &str, first: ToolCallMetadata) -> usize {
    let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("flaky_send", "{}"),
        MockProvider::tool_call_response("flaky_send", "{}"),
        MockProvider::tool_call_response("flaky_send", "{}"),
        MockProvider::text_response("Done."),
    ]);
    let tool = FlakySend {
        executions: executions.clone(),
        first,
    };
    let harness = root(provider, vec![Arc::new(tool)]).await;
    ask(&harness, session, "Send it").await;
    executions.load(std::sync::atomic::Ordering::SeqCst)
}

/// A refused send runs again, and once it has succeeded an identical call
/// replays the success instead of sending twice.
#[tokio::test]
async fn refused_send_runs_again_and_its_success_is_never_repeated() {
    let refused = ToolCallMetadata {
        outcome_status: Some(crate::traits::ToolOutcomeStatus::FailedRetryable),
        ..ToolCallMetadata::default()
    };
    assert_eq!(
        executions_after_three_identical_sends("retry-refused", refused).await,
        2
    );
}

/// A timed-out send may have gone through, so it is never sent again.
#[tokio::test]
async fn timed_out_send_is_not_sent_again() {
    let timed_out = ToolCallMetadata {
        outcome_status: Some(crate::traits::ToolOutcomeStatus::FailedRetryable),
        timed_out: true,
        ..ToolCallMetadata::default()
    };
    assert_eq!(
        executions_after_three_identical_sends("retry-timeout", timed_out).await,
        1
    );
}
