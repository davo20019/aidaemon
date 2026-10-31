use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::message::{Message, ToolCall};
use crate::provider::Usage;
use crate::receipt::Receipt;
use crate::stuck::StuckState;

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
}

/// The unfinished tool batch of an interrupted turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pending {
    pub calls: Vec<ToolCall>,
    /// Index of the interrupted call; earlier calls already have results.
    pub next: usize,
    pub interruption: Interruption,
}

/// Everything needed to continue a run, as plain serializable data.
///
/// Persist it anywhere (a database row, a file, a queue message) and hand it
/// to [`crate::Agent::resume`] later. Agents are matched by their stable
/// `agent_id`, never by position or display name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunState {
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
    #[serde(default)]
    pub nudges: u32,
    #[serde(default)]
    pub(crate) stuck: StuckState,
}

impl RunState {
    pub(crate) fn new(agent_id: &str, session_id: &str, input: &str) -> Self {
        Self {
            run_id: RunId::new(),
            agent_id: agent_id.to_string(),
            session_id: session_id.to_string(),
            messages: vec![Message::user(input)],
            receipts: Vec::new(),
            usage: Usage::default(),
            run_usage: Usage::default(),
            turn: 0,
            tool_calls: 0,
            elapsed_ms: 0,
            pending: None,
            nudges: 0,
            stuck: StuckState::default(),
        }
    }

    /// Continue an earlier, finished run with a new user message (same
    /// session, new run). Session receipts and `usage` keep accumulating;
    /// per-run counters and budgets (`turn`, `tool_calls`, `run_usage`,
    /// elapsed time) start again from zero.
    pub fn with_user_message(mut self, input: impl Into<String>) -> Self {
        self.messages.push(Message::user(input));
        self.run_id = RunId::new();
        self.run_usage = Usage::default();
        self.turn = 0;
        self.tool_calls = 0;
        self.elapsed_ms = 0;
        self.nudges = 0;
        self.stuck.reset();
        self
    }

    /// Receipts produced by the current run (not earlier runs in the session).
    pub fn run_receipts(&self) -> impl Iterator<Item = &Receipt> {
        let run_id = self.run_id;
        self.receipts.iter().filter(move |r| r.run_id == run_id)
    }
}
