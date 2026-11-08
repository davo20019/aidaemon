use serde::{Deserialize, Serialize};

use crate::limits::LimitKind;
use crate::outcome::RunStatus;
use crate::provider::Usage;
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
    ModelResponded {
        turn: u32,
        usage: Usage,
        text: String,
        tool_calls: usize,
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
    },
}

/// Receives events synchronously as they happen. Keep it cheap; forward to a
/// channel if the consumer is slow.
pub trait EventSink: Send + Sync {
    fn emit(&self, run_id: RunId, event: &Event);
}

impl<F> EventSink for F
where
    F: Fn(RunId, &Event) + Send + Sync,
{
    fn emit(&self, run_id: RunId, event: &Event) {
        self(run_id, event)
    }
}
