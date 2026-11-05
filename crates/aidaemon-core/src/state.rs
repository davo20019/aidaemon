use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::approval::{EffectKey, InterruptKind};
use crate::message::{Message, ToolCall};
use crate::provider::Usage;
use crate::receipt::Receipt;
use crate::stuck::StuckState;
use crate::tool::Finish;

/// Version of the [`RunState`] serialization format this kernel writes and
/// can resume. State saved before the field existed reads as version 1.
/// Bump it when a change would make older saved state resume wrongly, and
/// teach `resume` to migrate or refuse the old version.
pub const RUN_STATE_SCHEMA: u32 = 1;

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
    /// The call waiting for a decision. It has not run.
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
    pub calls: Vec<ToolCall>,
    /// Index of the interrupted call; earlier calls already have results.
    pub next: usize,
    pub interruption: Interruption,
    /// Index of the hook that interrupted. An approval settles that hook
    /// only: hooks registered after it still judge the call on resume.
    #[serde(default)]
    pub hook_index: Option<usize>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_flight: Option<InFlight>,
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
    /// The first finish a call returned in the unfinished batch; the run
    /// ends with it once the batch ends (see [`crate::ToolOutput::finish`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finishing: Option<PendingFinish>,
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
            finishing: None,
        }
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
    /// [`RunState::declined`] effects are forgotten.
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
        self.finishing = None;
        self
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
