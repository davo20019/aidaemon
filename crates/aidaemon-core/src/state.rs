use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::approval::{EffectKey, InterruptKind, SharedDeclines};
use crate::limits::LimitGrants;
use crate::message::{Message, ToolCall};
use crate::provider::Usage;
use crate::receipt::Receipt;
use crate::stuck::StuckState;
use crate::tool::Finish;

/// Version of the [`RunState`] serialization format this kernel writes and
/// can resume. State saved before the field existed reads as version 1.
/// Bump it when a change would make older saved state resume wrongly, and
/// teach `resume` to migrate or refuse the old version.
///
/// Version 2 added [`Pending::child`]. A version 1 snapshot has no child and
/// means the same thing under version 2, so it loads as is. A kernel that
/// knows only version 1 must refuse version 2: it would drop the child and
/// take the child's interruption for the parent's own.
pub const RUN_STATE_SCHEMA: u32 = 2;

fn first_schema() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(Uuid);

impl RunId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Why the run paused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interruption {
    /// The call waiting for a decision. It has not run. In a parent paused
    /// for a child ([`Pending::child`]) this is the child's call, at whatever
    /// depth it is waiting, so the approver is shown what would really run.
    pub call: ToolCall,
    pub reason: String,
    /// Who the interruption waits for. State saved before kinds existed
    /// reads as [`InterruptKind::Untrusted`].
    #[serde(default)]
    pub kind: InterruptKind,
}

/// The unfinished tool batch of an interrupted turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pending {
    #[doc(hidden)]
    pub calls: Vec<ToolCall>,
    /// Index of the interrupted call; earlier calls already have results.
    #[doc(hidden)]
    pub next: usize,
    pub interruption: Interruption,
    /// Index of the hook that interrupted. An approval settles that hook
    /// only: hooks registered after it still judge the call on resume.
    #[serde(default)]
    #[doc(hidden)]
    pub hook_index: Option<usize>,
    /// The hooks that were installed when the run paused. `hook_index` means
    /// nothing under a different chain, so an approval is judged again from
    /// the first hook when this does not match.
    #[serde(default)]
    #[doc(hidden)]
    pub hook_chain: Vec<String>,
    /// The paused run of a child this call started, when the interrupted
    /// call is a tool that delegates ([`crate::Tool::suspends`]). The call at
    /// `next` is already inside that tool, so an answer goes to the child (at
    /// any depth) and the call's result is what the child ends with.
    /// [`Pending::interruption`] is then a copy of the deepest run's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child: Option<Box<RunState>>,
}

/// A tool batch that was executing when the state was last saved. Present
/// only in checkpoints taken mid-turn; used for crash recovery.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InFlight {
    pub calls: Vec<ToolCall>,
    /// Calls that were handed to their tool.
    #[serde(default)]
    pub started: Vec<String>,
    /// Calls whose result is already in the transcript.
    #[serde(default)]
    pub done: Vec<String>,
}

/// A finish a dispatched call returned in the current batch, kept until the
/// batch ends (across an interruption or a restart) and then applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingFinish {
    pub finish: Finish,
    /// The name of the tool whose call returned it.
    pub tool: String,
}

/// Everything needed to continue a run, as plain serializable data.
///
/// Persist it anywhere (a database row, a file, a queue message) and hand it
/// to [`crate::Agent::resume`] later. Agents are matched by their stable
/// `agent_id`, never by position or display name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    /// See [`RUN_STATE_SCHEMA`]; read it with [`RunState::schema`].
    #[serde(default = "first_schema")]
    pub(crate) schema: u32,
    pub run_id: RunId,
    pub agent_id: String,
    pub session_id: String,
    /// Transcript without the agent's instructions (those are re-sent on
    /// every request).
    pub messages: Vec<Message>,
    /// Receipts of every run in this session. Each carries its `run_id`;
    /// use [`RunState::run_receipts`] for the current run only.
    pub receipts: Vec<Receipt>,
    /// Token usage of every run in this session.
    pub usage: Usage,
    /// Token usage of the current run only. Budgets in [`crate::Limits`]
    /// are checked against this, like `turn` and `tool_calls`.
    #[serde(default)]
    pub run_usage: Usage,
    pub turn: u32,
    pub tool_calls: u32,
    /// Wall-clock time already spent, summed across resumes.
    pub elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending: Option<Pending>,
    /// Kernel bookkeeping for crash recovery (see `aidaemon-store-sqlite`'s
    /// `in_flight_runs`), not part of the stable API.
    #[doc(hidden)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_flight: Option<InFlight>,
    #[doc(hidden)]
    #[serde(default)]
    pub nudges: u32,
    /// The run has seen untrusted content (a dispatched call whose result
    /// was authored by someone else, whatever its status), or was seeded
    /// that way. Never cleared within a run.
    #[serde(default)]
    pub untrusted_seen: bool,
    #[serde(default)]
    pub(crate) stuck: StuckState,
    /// Effects the owner declined in this run (deduplicated). The kernel
    /// records a denied [`InterruptKind::OwnerApproval`] request's effects
    /// here; hooks read them through [`crate::HookContext::declined`]. A new
    /// run ([`RunState::with_user_message`]) starts with none.
    #[serde(default)]
    pub declined: Vec<EffectKey>,
    /// Declines shared with related runs (a parent and its children), set
    /// with [`RunState::share_declines`]. A runtime handle, never
    /// serialized: `declined` is the run's record, and a resumed run shares
    /// again once the application reattaches a set.
    #[serde(skip)]
    pub(crate) shared_declines: Option<SharedDeclines>,
    /// Requirements stop hooks reported as unmet when the run stopped
    /// ([`crate::StopDecision::AcceptUnmet`]), deduplicated, in hook order.
    /// Set only when the run ends `Completed`; a new run
    /// ([`RunState::with_user_message`]) starts with none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unmet: Vec<String>,
    /// The first finish a call returned in the unfinished batch; the run
    /// ends with it once the batch ends (see [`crate::ToolOutput::finish`]).
    #[doc(hidden)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finishing: Option<PendingFinish>,
    /// How far `on_limit` hooks raised this run's limits. A new run
    /// ([`RunState::with_user_message`]) starts with none.
    #[doc(hidden)]
    #[serde(default)]
    pub limit_grants: LimitGrants,
    /// The run whose tool call started this one, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run: Option<RunId>,
    /// `on_stop` continuations since the last new successful call.
    #[serde(default)]
    pub(crate) idle_continuations: u32,
    /// Successful calls of this run when `idle_continuations` last changed.
    #[serde(default)]
    pub(crate) continued_at_successes: usize,
}

impl RunState {
    /// A fresh state for a new run, so callers know the run id before it
    /// starts. Run it with [`crate::Agent::run_from`].
    /// `input` is text or a user [`Message`] (for example one with images).
    pub fn new(agent_id: &str, session_id: &str, input: impl Into<Message>) -> Self {
        Self {
            schema: RUN_STATE_SCHEMA,
            run_id: RunId::new(),
            agent_id: agent_id.to_string(),
            session_id: session_id.to_string(),
            messages: vec![input.into()],
            receipts: Vec::new(),
            usage: Usage::default(),
            run_usage: Usage::default(),
            turn: 0,
            tool_calls: 0,
            elapsed_ms: 0,
            pending: None,
            in_flight: None,
            nudges: 0,
            untrusted_seen: false,
            stuck: StuckState::default(),
            declined: Vec::new(),
            shared_declines: None,
            unmet: Vec::new(),
            finishing: None,
            limit_grants: LimitGrants::default(),
            parent_run: None,
            idle_continuations: 0,
            continued_at_successes: 0,
        }
    }

    /// Record the run whose tool call started this one.
    pub fn with_parent(mut self, parent: RunId) -> Self {
        self.parent_run = Some(parent);
        self
    }

    /// Start the run as if it had already read untrusted content (the
    /// caller knows the input itself is someone else's).
    pub fn with_untrusted_context(mut self) -> Self {
        self.untrusted_seen = true;
        self
    }

    /// Continue an earlier, finished run with a new user message (same
    /// session, new run). Session receipts and `usage` keep accumulating;
    /// per-run counters and budgets (`turn`, `tool_calls`, `run_usage`,
    /// elapsed time) start again from zero, and the run's
    /// [`RunState::declined`] effects and [`RunState::unmet`] requirements
    /// are forgotten, as is any [`SharedDeclines`] set: the new run shares
    /// only a set attached to it.
    pub fn with_user_message(mut self, input: impl Into<Message>) -> Self {
        self.messages.push(input.into());
        self.run_id = RunId::new();
        self.run_usage = Usage::default();
        self.turn = 0;
        self.tool_calls = 0;
        self.elapsed_ms = 0;
        self.nudges = 0;
        self.in_flight = None;
        self.stuck.reset();
        self.declined.clear();
        self.shared_declines = None;
        self.unmet.clear();
        self.finishing = None;
        self.limit_grants = LimitGrants::default();
        self.idle_continuations = 0;
        self.continued_at_successes = 0;
        self
    }

    /// Share this run's declines with the other runs holding `declines`
    /// (see [`SharedDeclines`]): each is refused the others' declined
    /// effects from its next call on. The run's own declines are published
    /// at once and the set's are taken in. Attach it again after resuming
    /// saved state: the handle is not serialized.
    pub fn share_declines(mut self, declines: SharedDeclines) -> Self {
        declines.exchange(&mut self.declined);
        self.shared_declines = Some(declines);
        self
    }

    /// The decline set this run shares, if any.
    pub fn shared_declines(&self) -> Option<&SharedDeclines> {
        self.shared_declines.as_ref()
    }

    /// Take in what the runs sharing this run's set declined and publish
    /// this run's own. Returns whether this run's declines grew.
    pub(crate) fn sync_declines(&mut self) -> bool {
        match self.shared_declines.as_ref() {
            Some(shared) => shared.exchange(&mut self.declined),
            None => false,
        }
    }

    /// The run actually waiting for an answer: this one, or the child it is
    /// paused for, at any depth. `None` when this run is not interrupted.
    pub fn waiting_run(&self) -> Option<&RunState> {
        let mut run = self;
        run.pending.as_ref()?;
        while let Some(child) = run.pending.as_ref().and_then(|p| p.child.as_deref()) {
            run = child;
        }
        Some(run)
    }

    /// The serialization format version this state was written with.
    pub fn schema(&self) -> u32 {
        self.schema
    }

    /// Receipts produced by the current run (not earlier runs in the session).
    pub fn run_receipts(&self) -> impl Iterator<Item = &Receipt> {
        let run_id = self.run_id;
        self.receipts.iter().filter(move |r| r.run_id == run_id)
    }
}
