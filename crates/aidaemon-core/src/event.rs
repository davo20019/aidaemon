use serde::{Deserialize, Serialize};

use crate::hook::{HookFailure, HookPoint};
use crate::limits::LimitKind;
use crate::outcome::RunStatus;
use crate::provider::{ModelRequest, Usage};
use crate::receipt::Receipt;
use crate::state::{Interruption, RunId};
use crate::stuck::StuckSignal;

/// Everything observable about a run, in order. One stream feeds UIs,
/// tracing exporters, stores and eval assertions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[non_exhaustive]
pub enum Event {
    RunStarted {
        session_id: String,
    },
    RunResumed,
    /// This run was started by a tool call of `parent_run`.
    RunLinked {
        parent_run: RunId,
    },
    TurnStarted {
        turn: u32,
    },
    /// Streamed model text, before the turn's `ModelResponded`.
    TextDelta {
        turn: u32,
        text: String,
    },
    /// The run continued after a crash or restart.
    Recovered {
        reran: Vec<String>,
        indeterminate: Vec<String>,
    },
    /// A [`crate::Failover`] chain moved to its next provider during this
    /// turn's model call.
    ProviderSwitched {
        turn: u32,
        switch: crate::ProviderSwitch,
    },
    /// The model is about to be called; pairs with `ModelResponded`, or
    /// with `ModelRetried` or `Finished` if the call fails. Content is not
    /// journaled: see [`EventSink::observe_request`].
    ModelRequested {
        turn: u32,
        model: String,
        messages: usize,
        tools: usize,
    },
    ModelResponded {
        turn: u32,
        usage: Usage,
        text: String,
        tool_calls: usize,
        /// Why the reply may be incomplete, in the provider's words.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    ToolStarted {
        call_id: String,
        tool: String,
    },
    ToolFinished {
        receipt: Receipt,
    },
    Stuck {
        signal: StuckSignal,
    },
    StopRejected {
        message: String,
    },
    /// A hook made a decision other than letting things proceed. Together
    /// with `HookFailed` this says which hook did what, and why.
    HookDecided {
        point: HookPoint,
        hook: String,
        decision: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// A hook panicked, timed out or was cancelled, and gave no answer.
    /// See [`crate::Hook`] for what the run does instead.
    HookFailed {
        point: HookPoint,
        hook: String,
        failure: HookFailure,
    },
    /// A hook answered in the model's place, or supplied the response of a
    /// failed call; the model's own reply was not used.
    ModelAnswered {
        turn: u32,
        hook: String,
    },
    /// An `after_model` hook rejected this turn's response.
    ModelRejected {
        turn: u32,
        message: String,
    },
    /// The model call failed and an `on_model_error` hook asked for a retry.
    ModelRetried {
        turn: u32,
        error: String,
        after_ms: u64,
    },
    /// An `on_limit` hook raised a limit; `amount` is in the limit's unit.
    LimitExtended {
        limit: LimitKind,
        amount: u64,
    },
    /// An `edit_transcript` hook replaced the stored transcript.
    TranscriptEdited {
        turn: u32,
        before: usize,
        after: usize,
    },
    /// An `edit_transcript` hook's edit was malformed and was not applied.
    TranscriptEditRefused {
        turn: u32,
        reason: String,
    },
    Interrupted {
        interruption: Interruption,
    },
    Finished {
        status: RunStatus,
        usage: Usage,
        turns: u32,
        /// How long the run took, summed across resumes, on the runtime's
        /// clock (see [`crate::RunOutcome::elapsed`]).
        #[serde(default)]
        elapsed_ms: u64,
    },
}

/// Receives events synchronously as they happen. Keep it cheap; forward to a
/// channel if the consumer is slow. Events carry no timestamps: a sink is
/// called the moment each happens, so it stamps them with whatever clock it
/// keeps (receipts carry each call's `duration_ms`, and
/// [`Event::Finished`] the run's `elapsed_ms`).
pub trait EventSink: Send + Sync {
    fn emit(&self, run_id: RunId, event: &Event);

    /// The exact request about to go to the model, after every hook has
    /// shaped it. Prompts are not events, so they are never journaled; a
    /// sink that exports them takes on deciding what may leave.
    fn observe_request(&self, _run_id: RunId, _request: &ModelRequest) {}
}

impl<F> EventSink for F
where
    F: Fn(RunId, &Event) + Send + Sync,
{
    fn emit(&self, run_id: RunId, event: &Event) {
        self(run_id, event)
    }
}
