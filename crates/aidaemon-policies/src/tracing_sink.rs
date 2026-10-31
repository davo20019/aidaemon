//! An `EventSink` that turns the run's event stream into `tracing` spans and
//! events, ready for any subscriber (fmt, JSON, OpenTelemetry layer).

use std::collections::HashMap;
use std::sync::Mutex;

use aidaemon_core::{Event, EventSink, RunId, RunStatus};
use tracing::{field, Level, Span};

/// Emits one `tracing` event per kernel [`Event`], inside spans:
///
/// - an `agent_run` span per run (fields `run_id`, `session_id`, and on
///   finish `status`, `turns`, `input_tokens`, `output_tokens`), opened on
///   `RunStarted`/`RunResumed` and closed on `Finished`;
/// - a `tool_call` span per call (fields `run_id`, `call_id`, `tool`, and
///   on finish `status`, `duration_ms`), a child of the run span.
///
/// Events use target `aidaemon::run` and carry structured fields
/// (`run_id`, `turn`, `tool`, `status`, token counts), so an OpenTelemetry
/// layer maps them to span events and attributes without parsing messages.
/// `TextDelta` is emitted at `TRACE` level only.
#[derive(Default)]
pub struct TracingSink {
    runs: Mutex<HashMap<RunId, Span>>,
    tools: Mutex<HashMap<(RunId, String), Span>>,
}

impl TracingSink {
    pub fn new() -> Self {
        Self::default()
    }

    fn run_span(&self, run_id: RunId, session_id: Option<&str>) -> Span {
        let mut runs = self.runs.lock().expect("tracing sink poisoned");
        runs.entry(run_id)
            .or_insert_with(|| {
                tracing::span!(
                    Level::INFO,
                    "agent_run",
                    run_id = %run_id,
                    session_id = session_id.unwrap_or(""),
                    status = field::Empty,
                    turns = field::Empty,
                    input_tokens = field::Empty,
                    output_tokens = field::Empty,
                )
            })
            .clone()
    }
}

fn status_name(status: &RunStatus) -> String {
    serde_json::to_value(status)
        .ok()
        .and_then(|v| v.get("status").and_then(|s| s.as_str()).map(str::to_string))
        .unwrap_or_else(|| "unknown".into())
}

fn json_name<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".into())
}

impl EventSink for TracingSink {
    fn emit(&self, run_id: RunId, event: &Event) {
        let session = match event {
            Event::RunStarted { session_id } => Some(session_id.as_str()),
            _ => None,
        };
        let run = self.run_span(run_id, session);
        let _entered = run.enter();
        match event {
            Event::RunStarted { session_id } => {
                tracing::info!(target: "aidaemon::run", run_id = %run_id, session_id = %session_id, "run started");
            }
            Event::RunResumed => {
                tracing::info!(target: "aidaemon::run", run_id = %run_id, "run resumed");
            }
            Event::TurnStarted { turn } => {
                tracing::debug!(target: "aidaemon::run", run_id = %run_id, turn = *turn, "turn started");
            }
            Event::TextDelta { turn, text } => {
                tracing::trace!(target: "aidaemon::run", run_id = %run_id, turn = *turn, chars = text.len(), "text delta");
            }
            Event::Recovered {
                reran,
                indeterminate,
            } => {
                tracing::warn!(
                    target: "aidaemon::run",
                    run_id = %run_id,
                    reran = reran.len(),
                    indeterminate = indeterminate.len(),
                    "run recovered after restart"
                );
            }
            Event::ModelResponded {
                turn,
                usage,
                tool_calls,
                ..
            } => {
                tracing::info!(
                    target: "aidaemon::run",
                    run_id = %run_id,
                    turn = *turn,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    cached_input_tokens = usage.cached_input_tokens,
                    tool_calls = *tool_calls as u64,
                    "model responded"
                );
            }
            Event::ToolStarted { call_id, tool } => {
                let span = tracing::span!(
                    parent: &run,
                    Level::INFO,
                    "tool_call",
                    run_id = %run_id,
                    call_id = %call_id,
                    tool = %tool,
                    status = field::Empty,
                    duration_ms = field::Empty,
                );
                tracing::debug!(target: "aidaemon::run", parent: &span, run_id = %run_id, call_id = %call_id, tool = %tool, "tool started");
                self.tools
                    .lock()
                    .expect("tracing sink poisoned")
                    .insert((run_id, call_id.clone()), span);
            }
            Event::ToolFinished { receipt } => {
                let status = json_name(&receipt.status);
                let span = self
                    .tools
                    .lock()
                    .expect("tracing sink poisoned")
                    .remove(&(run_id, receipt.call_id.clone()));
                let parent = span.clone().unwrap_or_else(|| run.clone());
                if let Some(span) = &span {
                    span.record("status", status.as_str());
                    span.record("duration_ms", receipt.duration_ms);
                }
                let stage = json_name(&receipt.stage);
                if receipt.succeeded() {
                    tracing::info!(
                        target: "aidaemon::run",
                        parent: &parent,
                        run_id = %run_id,
                        turn = receipt.turn,
                        call_id = %receipt.call_id,
                        tool = %receipt.tool,
                        status = %status,
                        stage = %stage,
                        duration_ms = receipt.duration_ms,
                        "tool finished"
                    );
                } else {
                    tracing::warn!(
                        target: "aidaemon::run",
                        parent: &parent,
                        run_id = %run_id,
                        turn = receipt.turn,
                        call_id = %receipt.call_id,
                        tool = %receipt.tool,
                        status = %status,
                        stage = %stage,
                        duration_ms = receipt.duration_ms,
                        reason = receipt.reason.as_deref().unwrap_or(""),
                        "tool finished"
                    );
                }
            }
            Event::Stuck { signal } => {
                tracing::warn!(target: "aidaemon::run", run_id = %run_id, signal = ?signal, "stuck detected");
            }
            Event::StopRejected { message } => {
                tracing::info!(target: "aidaemon::run", run_id = %run_id, chars = message.len(), "stop rejected by a hook");
            }
            Event::Interrupted { interruption } => {
                tracing::info!(
                    target: "aidaemon::run",
                    run_id = %run_id,
                    tool = %interruption.call.name,
                    call_id = %interruption.call.id,
                    reason = %interruption.reason,
                    "run interrupted"
                );
            }
            Event::Finished {
                status,
                usage,
                turns,
            } => {
                let status = status_name(status);
                run.record("status", status.as_str());
                run.record("turns", *turns);
                run.record("input_tokens", usage.input_tokens);
                run.record("output_tokens", usage.output_tokens);
                tracing::info!(
                    target: "aidaemon::run",
                    run_id = %run_id,
                    status = %status,
                    turns = *turns,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    billable_tokens = usage.billable_tokens(),
                    "run finished"
                );
                drop(_entered);
                self.runs
                    .lock()
                    .expect("tracing sink poisoned")
                    .remove(&run_id);
                self.tools
                    .lock()
                    .expect("tracing sink poisoned")
                    .retain(|(id, _), _| *id != run_id);
            }
            other => {
                tracing::debug!(target: "aidaemon::run", run_id = %run_id, event = ?other, "event");
            }
        }
    }
}
