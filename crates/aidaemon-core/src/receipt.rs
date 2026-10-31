use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How far a call got before its outcome was decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvocationStage {
    /// Refused before the tool ran (unknown tool, bad arguments, policy).
    RejectedBeforeDispatch,
    /// The tool's code was invoked.
    Dispatched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptStatus {
    Succeeded,
    /// Ran and reported failure, or returned a [`crate::ToolError`].
    Failed,
    /// A hook or the user refused it.
    Denied,
    TimedOut,
    Cancelled,
    UnknownTool,
    InvalidArguments,
    /// The call may or may not have completed: the process stopped while it
    /// was running and the tool is not idempotent, so it was not re-run.
    Indeterminate,
}

/// The durable record of one tool call. Every call produces exactly one,
/// refusals included, so "what actually happened" never depends on what the
/// model says happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    pub call_id: String,
    pub tool: String,
    pub arguments: Value,
    pub turn: u32,
    pub stage: InvocationStage,
    pub status: ReceiptStatus,
    pub read_only: bool,
    pub duration_ms: u64,
    /// Why it was denied or failed, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Structured evidence from [`crate::ToolOutput::details`].
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub details: Value,
}

impl Receipt {
    pub fn succeeded(&self) -> bool {
        self.status == ReceiptStatus::Succeeded
    }
}
