use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::approval::{EffectKey, InterruptKind};
use crate::authority::Authority;
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
    /// Stop now, whatever any other hook asked for, and record `reason` as
    /// unmet: for a hook that knows more turns must not happen.
    Halt { reason: String },
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
    /// End the run now with [`crate::RunStatus::Stopped`]: none of the
    /// response's tool calls run and it is not recorded. For a hook that
    /// knows nothing more may happen, such as a spent budget.
    Stop { reason: String },
}

/// What an `on_model_error` hook decides about a failed model call.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelErrorDecision {
    /// End the run with [`crate::RunStatus::ProviderFailed`].
    Fail,
    /// Wait, then run another turn. The transcript is untouched and
    /// `before_model` runs again, so a hook can shrink the request or change
    /// the model first. The wait stops on cancel and at the wall-clock
    /// budget, and the retry counts as a turn: budgets bound the retries.
    Retry { after: Duration },
    /// Carry on with this response as if the model had given it.
    Recover { response: ModelResponse },
    /// End the run, whatever any other hook asked for.
    Abort,
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
    /// End the run, whatever any other hook granted.
    Forbid,
}

/// A moment of the loop at which hooks run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookPoint {
    OnRunStart,
    EditTranscript,
    BeforeModel,
    AfterModel,
    AnswerModel,
    OnModelError,
    OnLimit,
    BeforeTool,
    AfterTool,
    OnStop,
    OnStuck,
    OnRunEnd,
}

/// How a hook failed to give an answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum HookFailure {
    Panicked {
        message: String,
    },
    TimedOut {
        after_ms: u64,
    },
    /// The run was cancelled while the hook was working.
    Cancelled,
}

/// Extension point for every policy. All methods have no-op defaults, so a
/// hook implements only what it needs. Hooks run in registration order and
/// outside the model's context window: they cost no tokens and the model
/// cannot argue its way past them.
///
/// The kernel contains every call: a hook that panics, outlasts its time or
/// is overtaken by a cancel has given no answer, an
/// [`crate::Event::HookFailed`] is recorded, and the run goes on as that
/// moment requires. A check that guards something fails closed; anything
/// else fails open, because the run can go on without it:
///
/// | Point | With no answer |
/// |---|---|
/// | `before_tool` | the call is refused |
/// | `after_tool` | the model does not see the result |
/// | `after_model` | the response is not acted on and the run stops |
/// | `on_stop` | the run ends with an unmet requirement, never as a success |
/// | `on_limit`, `on_model_error` | the limit stands, the failure stands |
/// | `edit_transcript` | the transcript stays as it was |
/// | `on_run_start`, `before_model`, `on_stuck`, `on_run_end` | the run goes on |
///
/// `on_run_start`, `edit_transcript`, `answer_model`, `on_model_error` and
/// `on_limit` are unstable: tests cover them
/// but no shipped policy depends on them yet, so they may still change.
///
/// Timeouts and cancels stop a hook at its next `.await`, so a hook that
/// blocks a thread is not interrupted: keep blocking work off the executor.
#[async_trait]
pub trait Hook: Send + Sync {
    /// The safeguards this hook enforces. [`crate::AgentBuilder::build`]
    /// requires every [`Safeguard::REQUIRED`] one to be covered by a hook
    /// or an attestation.
    fn safeguards(&self) -> &'static [Safeguard] {
        &[]
    }

    /// How this hook is named in events and in a saved interruption. Stable
    /// across restarts of the same build; override it to keep a hook's
    /// identity across a rename.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// How long one call of this hook may take, in place of
    /// [`crate::Limits::hook_timeout`]. The wall-clock budget still applies.
    fn timeout(&self) -> Option<Duration> {
        None
    }

    /// The authority this hook enforces, if it enforces one. Passed to
    /// tools so a tool that starts another run can hold it to the same.
    fn authority(&self) -> Option<Authority> {
        None
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

    /// Answer in the model's place, for example from a cache. It runs after
    /// every `before_model` hook, so it sees the final request. The first
    /// answer wins; the response is treated as the model's, `after_model`
    /// included, and the turn counts.
    async fn answer_model(
        &self,
        _ctx: &HookContext<'_>,
        _request: &ModelRequest,
    ) -> Option<ModelResponse> {
        None
    }

    /// The model replied. Check or rewrite the response before it is
    /// recorded or acted on; the first `Reject` or `Stop` wins. Calls a hook adds or
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
    /// Every hook is asked: an `Abort` beats everything, then the first
    /// `Recover` or `Retry` wins. With none, the run ends.
    async fn on_model_error(
        &self,
        _ctx: &HookContext<'_>,
        _error: &ProviderError,
    ) -> ModelErrorDecision {
        ModelErrorDecision::Fail
    }

    /// The run reached a limit in [`crate::Limits`]. Every hook is asked: a
    /// `Forbid` beats any grant, then the first `Extend` wins, is recorded in [`crate::RunState::limit_grants`], and lasts for
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
    /// checklist, tests, anything. Every hook is asked: a `Halt` beats
    /// everything, then the first `Continue` wins. More than
    /// [`crate::Limits::max_idle_continuations`] in a row with no new
    /// successful call ends the run unmet.
    async fn on_stop(&self, _ctx: &HookContext<'_>, _final_text: &str) -> StopDecision {
        StopDecision::Accept
    }

    /// The run ended, with any status: fires once per `run` or `resume` that
    /// returns (a pause returns too), after the
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
