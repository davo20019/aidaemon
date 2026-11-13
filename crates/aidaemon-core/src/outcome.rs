use serde::{Deserialize, Serialize};

use crate::limits::LimitKind;
use crate::provider::Usage;
use crate::receipt::Receipt;
use crate::state::{Interruption, RunState};
use crate::stuck::StuckSignal;

/// Why a run ended. Every exit path maps to exactly one variant, so callers
/// never have to parse prose to learn what happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
#[non_exhaustive]
pub enum RunStatus {
    /// The model gave a final answer and every stop hook accepted it.
    Completed,
    /// Paused before a tool call; resume with [`crate::Agent::resume`].
    Interrupted {
        interruption: Interruption,
    },
    LimitReached {
        limit: LimitKind,
    },
    /// A custom [`crate::StopCondition`] ended the run.
    Stopped {
        reason: String,
    },
    Stuck {
        signal: StuckSignal,
    },
    Cancelled,
    /// The provider failed in a way the provider layer could not recover.
    ProviderFailed {
        error: String,
    },
    /// A tool ended the run with [`crate::ToolOutput::finish`]; `tool` is
    /// the name of the tool whose finish won. A `Succeeded` finish in a run
    /// with declined effects ([`crate::RunState::declined`]) is reported as
    /// `Partial`.
    Finished {
        outcome: FinishOutcome,
        tool: String,
    },
}

/// How a tool-driven finish judged the work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishOutcome {
    Succeeded,
    Partial,
    Blocked,
}

/// The typed result of [`crate::Agent::run`] or [`crate::Agent::resume`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunOutcome {
    pub status: RunStatus,
    /// Text of the last assistant message, when the run completed.
    pub final_text: Option<String>,
    /// Full state, ready to persist, resume, or continue with
    /// [`RunState::with_user_message`].
    pub state: RunState,
}

impl RunOutcome {
    /// The run stopped because the model answered and every stop hook let
    /// it stop. That is not the same as the work being done: see
    /// [`RunOutcome::is_succeeded`].
    pub fn is_completed(&self) -> bool {
        matches!(self.status, RunStatus::Completed)
    }

    /// The run's objective was met, as far as the kernel and its hooks can
    /// tell: it completed with no unmet requirement and nothing the owner
    /// declined, or a tool finished it as `Succeeded` (the kernel already
    /// reports a finish in a run with declines as `Partial`).
    pub fn is_succeeded(&self) -> bool {
        match &self.status {
            RunStatus::Completed => self.state.unmet.is_empty() && self.state.declined.is_empty(),
            RunStatus::Finished { outcome, .. } => *outcome == FinishOutcome::Succeeded,
            _ => false,
        }
    }

    /// Requirements a stop hook reported as unmet when the run stopped.
    pub fn unmet(&self) -> &[String] {
        &self.state.unmet
    }

    /// Usage of the whole session.
    pub fn usage(&self) -> Usage {
        self.state.usage
    }

    /// Usage of this run only.
    pub fn run_usage(&self) -> Usage {
        self.state.run_usage
    }

    pub fn receipts(&self) -> &[Receipt] {
        &self.state.receipts
    }

    pub fn turns(&self) -> u32 {
        self.state.turn
    }

    /// How long the run took, summed across resumes, on the runtime's
    /// clock (Tokio's, so a test with paused time measures paused time).
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.state.elapsed_ms)
    }

    /// Each tool call of this run with the result the model was given, in
    /// call order: the receipt (tool, arguments, status, details) and the
    /// text of the tool message that answered it. A sub-agent's answer
    /// reaches its parent this way. `text` is `None` when no message
    /// answers the call any more (a hook edited the transcript).
    pub fn tool_results(&self) -> Vec<CallResult<'_>> {
        self.state
            .run_receipts()
            .map(|receipt| {
                let message = self.state.messages.iter().rev().find(|m| {
                    m.role == crate::Role::Tool
                        && m.tool_call_id.as_deref() == Some(receipt.call_id.as_str())
                });
                CallResult {
                    receipt,
                    text: message.map(|m| m.content.as_str()),
                }
            })
            .collect()
    }
}

/// One tool call of a run and what the model was told about it (see
/// [`RunOutcome::tool_results`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallResult<'a> {
    pub receipt: &'a Receipt,
    /// The tool message the model saw for this call.
    pub text: Option<&'a str>,
}

impl CallResult<'_> {
    /// The call ran and succeeded.
    pub fn succeeded(&self) -> bool {
        self.receipt.succeeded()
    }
}
