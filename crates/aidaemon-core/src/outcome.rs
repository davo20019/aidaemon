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
    pub fn is_completed(&self) -> bool {
        matches!(self.status, RunStatus::Completed)
    }

    pub fn usage(&self) -> Usage {
        self.state.usage
    }

    pub fn receipts(&self) -> &[Receipt] {
        &self.state.receipts
    }

    pub fn turns(&self) -> u32 {
        self.state.turn
    }
}
