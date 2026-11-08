use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::approval::{EffectKey, InterruptKind};
use crate::limits::LimitKind;
use crate::message::{Message, ToolCall};
use crate::outcome::RunOutcome;
use crate::provider::{ModelRequest, ModelResponse, ProviderError, Usage};
use crate::receipt::Receipt;
use crate::safeguard::Safeguard;
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
    /// The run has read untrusted content so far (see
    /// [`crate::RunState::untrusted_seen`]).
    pub untrusted_seen: bool,
    /// Effects the owner declined earlier in this run (see
    /// [`crate::RunState::declined`]).
    pub declined: &'a [EffectKey],
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
    /// Replace the arguments and restart all hooks so earlier judgments
    /// cannot authorize different effects. Identical arguments are a no-op;
    /// more than 16 changes refuses the call. Rewriters should be idempotent.
    Rewrite {
        arguments: Value,
    },
    /// Pause the run before this call (for example, to ask a human). The run
    /// returns [`crate::RunStatus::Interrupted`] with serializable state that
    /// [`crate::Agent::resume`] continues later, in this or another process.
    Interrupt {
        reason: String,
        /// Who the interruption waits for; resolvers settle the kinds they
        /// own.
        kind: InterruptKind,
    },
    /// Answer in the tool's place: the tool does not run and the model sees
    /// `output`. For results the host already holds (a replayed receipt, a
    /// suppressed duplicate) or a call deferred on purpose. The receipt is
    /// [`crate::ReceiptStatus::Answered`]; an answer never ends the run (its
    /// `finish` is dropped) and never counts as work done.
    Answer {
        output: ToolOutput,
    },
}

/// What an `on_stop` hook decides when the model replies without tool calls.
///
/// Stopping and succeeding are separate decisions. A hook can let the run
/// stop (more turns would not help: the work is blocked, deferred or its
/// budget for asking is spent) while recording that the objective was not
/// met. Such a run ends [`crate::RunStatus::Completed`] with
/// [`crate::RunState::unmet`] filled, and
/// [`crate::RunOutcome::is_succeeded`] is false.
#[derive(Debug, Clone, PartialEq)]
pub enum StopDecision {
    /// Stop, and this hook has nothing against calling the run a success.
    Accept,
    /// Do not finish yet; append this message and run another turn.
    Continue { message: String },
    /// Stop, but the objective is not met. Each line names one unmet
    /// requirement in words a person can read. The run ends `Completed`
    /// (the model may not keep going) and never counts as succeeded.
    AcceptUnmet { unmet: Vec<String> },
}

impl StopDecision {
    /// Whether the run may stop (anything but [`StopDecision::Continue`]).
    pub fn permits_stop(&self) -> bool {
        !matches!(self, Self::Continue { .. })
    }
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

/// What an `after_model` hook decides about a response.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelDecision {
    /// Carry on with the response as the hooks left it.
    Accept,
    /// Do not act on this response: none of its tool calls run and it is
    /// not a final answer. Its text stays in the transcript, so the model
    /// can see what it is correcting, followed by `message`, and another
    /// turn runs. To keep the text from the model, clear it first.
    Reject { message: String },
}

/// What an `on_model_error` hook decides about a failed model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelErrorDecision {
    /// End the run with [`crate::RunStatus::ProviderFailed`].
    Fail,
    /// Wait, then run another turn. The transcript is untouched and
    /// `before_model` runs again, so a hook can shrink the request or change
    /// the model first. The wait stops on cancel and at the wall-clock
    /// budget, and the retry counts as a turn: budgets bound the retries.
    Retry { after: Duration },
}

/// What an `on_limit` hook decides about a limit the run has reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitDecision {
    /// End the run with [`crate::RunStatus::LimitReached`].
    Stop,
    /// Raise this limit for the rest of the run. `amount` is in the limit's
    /// own unit: turns, billable tokens, tool calls, or milliseconds. An
    /// amount of zero is a [`LimitDecision::Stop`].
    Extend { amount: u64 },
}

/// Extension point for every policy. All methods have no-op defaults, so a
/// hook implements only what it needs. Hooks run in registration order and
/// outside the model's context window: they cost no tokens and the model
/// cannot argue its way past them.
#[async_trait]
pub trait Hook: Send + Sync {
    /// The safeguards this hook enforces. [`crate::AgentBuilder::build`]
    /// requires every [`Safeguard::REQUIRED`] one to be covered by a hook
    /// or an attestation.
    fn safeguards(&self) -> &'static [Safeguard] {
        &[]
    }

    /// A new run is about to take its first turn: load what the other hooks
    /// will need. Resuming a paused run is not a start. A process that dies
    /// before the first turn starts the run again, so make this repeatable.
    async fn on_run_start(&self, _ctx: &HookContext<'_>) {}

    /// Replace the stored transcript at the start of a turn, for example
    /// with a summary. Prefer a `before_model` edit, which changes one
    /// request and loses nothing; use this when the shortened history should
    /// be what later turns and resumes start from.
    ///
    /// Hooks run in order and each sees the last one's edit. The kernel
    /// refuses an empty edit, one with no user message, and one that splits
    /// a tool call from its result (providers reject those requests), and
    /// keeps the transcript it had. Receipts and the run's trust state do
    /// not change: they are the record of what happened, whatever the
    /// transcript says now.
    async fn edit_transcript(&self, _ctx: &HookContext<'_>) -> Option<Vec<Message>> {
        None
    }

    /// Edit the request before it is sent: trim context, inject memory,
    /// change the model or tool choice.
    async fn before_model(&self, _ctx: &HookContext<'_>, _request: &mut ModelRequest) {}

    /// The model replied. Check or rewrite the response before it is
    /// recorded or acted on; the first `Reject` wins. Calls a hook adds or
    /// changes face the same validation and `before_tool` hooks as any
    /// other. Streamed text has already gone out as events, and the tokens
    /// are already counted.
    async fn after_model(
        &self,
        _ctx: &HookContext<'_>,
        _response: &mut ModelResponse,
    ) -> ModelDecision {
        ModelDecision::Accept
    }

    /// A model call failed after whatever retries its provider does itself.
    /// The first `Retry` wins; with none, the run ends.
    async fn on_model_error(
        &self,
        _ctx: &HookContext<'_>,
        _error: &ProviderError,
    ) -> ModelErrorDecision {
        ModelErrorDecision::Fail
    }

    /// The run reached a limit in [`crate::Limits`]. The first `Extend`
    /// wins, is recorded in [`crate::RunState::limit_grants`], and lasts for
    /// this run. [`crate::HARD_TURN_CAP`] cannot be extended and is never
    /// put to a hook. A model call cut off by the wall clock is lost and
    /// runs again if the limit is extended.
    async fn on_limit(&self, _ctx: &HookContext<'_>, _limit: LimitKind) -> LimitDecision {
        LimitDecision::Stop
    }

    /// Decide whether a call may run. The first `Deny`, `Interrupt` or
    /// `Answer` wins;
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
    /// change the outcome. A `run_resolving` call is one run: the
    /// interruptions it settles are not run ends.
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
