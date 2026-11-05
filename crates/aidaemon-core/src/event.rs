use serde::{Deserialize, Serialize};

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
