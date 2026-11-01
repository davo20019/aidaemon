use async_trait::async_trait;
use serde_json::Value;

use crate::message::{Message, ToolCall};
use crate::outcome::RunOutcome;
use crate::provider::{ModelRequest, Usage};
use crate::receipt::Receipt;
use crate::state::RunId;
use crate::stuck::StuckSignal;
use crate::tool::{ToolInfo, ToolOutput};

/// Read-only view of the run handed to every hook.
#[derive(Debug, Clone, Copy)]
pub struct HookContext<'a> {
    pub run_id: RunId,
    pub agent_id: &'a str,
    pub session_id: &'a str,
    pub turn: u32,
    pub messages: &'a [Message],
    /// Receipts of the whole session; see [`HookContext::run_receipts`].
    pub receipts: &'a [Receipt],
    /// Usage of the whole session.
    pub usage: Usage,
    /// Usage of the current run.
    pub run_usage: Usage,
}

impl HookContext<'_> {
    /// Receipts produced by the current run only. Completion checks should
    /// use this so work from an earlier run in the session is not credited.
    pub fn run_receipts(&self) -> impl Iterator<Item = &Receipt> {
        let run_id = self.run_id;
        self.receipts.iter().filter(move |r| r.run_id == run_id)
    }
}

/// What a `before_tool` hook decides about one call.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolDecision {
    Allow,
    /// Refuse. The reason goes back to the model as the tool result, and a
    /// `Denied` receipt is recorded.
    Deny {
        reason: String,
    },
    /// Replace the arguments, then keep asking the remaining hooks.
    Rewrite {
        arguments: Value,
    },
    /// Pause the run before this call (for example, to ask a human). The run
    /// returns [`crate::RunStatus::Interrupted`] with serializable state that
    /// [`crate::Agent::resume`] continues later, in this or another process.
    Interrupt {
        reason: String,
    },
}

/// What an `on_stop` hook decides when the model replies without tool calls.
#[derive(Debug, Clone, PartialEq)]
pub enum StopDecision {
    Accept,
    /// Do not finish yet; append this message and run another turn.
    Continue {
        message: String,
    },
}

/// What to do when the stuck detector fires.
#[derive(Debug, Clone, PartialEq)]
pub enum StuckDecision {
    /// Tell the model it is repeating itself and keep going.
    Nudge { message: String },
    /// End the run with [`crate::RunStatus::Stuck`].
    Stop,
    /// Not stuck (for example, deliberate polling). Keep going silently.
    Ignore,
}

/// Extension point for every policy. All methods have no-op defaults, so a
/// hook implements only what it needs. Hooks run in registration order and
/// outside the model's context window: they cost no tokens and the model
/// cannot argue its way past them.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Edit the request before it is sent: trim context, inject memory,
    /// change the model or tool choice.
    async fn before_model(&self, _ctx: &HookContext<'_>, _request: &mut ModelRequest) {}

    /// Decide whether a call may run. The first `Deny` or `Interrupt` wins;
    /// `Rewrite` changes the arguments seen by later hooks and the tool.
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _tool: &ToolInfo,
    ) -> ToolDecision {
        ToolDecision::Allow
    }

    /// Inspect or reshape a result before the model sees it (redaction,
    /// injection screening). The receipt is already final.
    async fn after_tool(
        &self,
        _ctx: &HookContext<'_>,
        _receipt: &Receipt,
        _output: &mut ToolOutput,
    ) {
    }

    /// The model produced a final answer. Check it against receipts, a
    /// checklist, tests, anything. The first `Continue` wins.
    async fn on_stop(&self, _ctx: &HookContext<'_>, _final_text: &str) -> StopDecision {
        StopDecision::Accept
    }

    /// The run ended, with any status. Fires exactly once per run, after the
    /// outcome is final and every `on_stop` check has had its say, so it is
    /// the place for after-the-fact work such as writing memory. It cannot
    /// change the outcome.
    async fn on_run_end(&self, _ctx: &HookContext<'_>, _outcome: &RunOutcome) {}

    /// The stuck detector fired. `None` means "no opinion"; if every hook
    /// abstains, the default policy nudges and then stops.
    async fn on_stuck(
        &self,
        _ctx: &HookContext<'_>,
        _signal: &StuckSignal,
    ) -> Option<StuckDecision> {
        None
    }
}
