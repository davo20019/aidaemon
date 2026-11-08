use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::provider::Usage;

/// Turn cap that applies even if `Limits::max_turns` is set higher. A run
/// that needs more than this is a bug, not a task.
pub const HARD_TURN_CAP: u32 = 1_000;

/// Hard budgets for one run. Turns, tokens and tool calls are checked at the
/// top of every turn; tool calls and wall-clock time are also checked before
/// every call in a batch, and the model call and each tool are bounded by the
/// remaining wall-clock budget. `max_turns` cannot be switched off: it is
/// clamped to [`HARD_TURN_CAP`].
#[derive(Debug, Clone, PartialEq)]
pub struct Limits {
    pub max_turns: u32,
    /// Billable tokens for this run (cached input excluded).
    pub max_tokens: Option<u64>,
    pub max_tool_calls: Option<u32>,
    /// Wall-clock budget for the whole run, summed across resumes.
    pub timeout: Option<Duration>,
    /// Per-call timeout for tools.
    pub tool_timeout: Duration,
    /// Tool output beyond this many characters is cut to head + tail.
    pub max_tool_output_chars: usize,
    /// How long one hook call may take (see [`crate::Hook::timeout`]).
    pub hook_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_turns: 50,
            max_tokens: None,
            max_tool_calls: None,
            timeout: None,
            tool_timeout: Duration::from_secs(120),
            max_tool_output_chars: 30_000,
            hook_timeout: Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    Turns,
    Tokens,
    ToolCalls,
    WallClock,
}

/// How far hooks have raised each limit for this run (see
/// [`crate::Hook::on_limit`]), on top of the agent's [`Limits`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LimitGrants {
    pub turns: u32,
    pub tokens: u64,
    pub tool_calls: u32,
    pub wall_clock_ms: u64,
}

impl LimitGrants {
    pub(crate) fn add(&mut self, limit: LimitKind, amount: u64) {
        let small = u32::try_from(amount).unwrap_or(u32::MAX);
        match limit {
            LimitKind::Turns => self.turns = self.turns.saturating_add(small),
            LimitKind::Tokens => self.tokens = self.tokens.saturating_add(amount),
            LimitKind::ToolCalls => self.tool_calls = self.tool_calls.saturating_add(small),
            LimitKind::WallClock => self.wall_clock_ms = self.wall_clock_ms.saturating_add(amount),
        }
    }
}

/// What a [`StopCondition`] can see.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunSnapshot {
    pub turn: u32,
    pub tool_calls: u32,
    /// Usage of the current run.
    pub usage: Usage,
    pub elapsed: Duration,
}

/// A custom stop rule, checked with the built-in limits at the top of each
/// turn. Return `Some(reason)` to end the run. Closures implement it.
pub trait StopCondition: Send + Sync {
    fn check(&self, snapshot: &RunSnapshot) -> Option<String>;
}

impl<F> StopCondition for F
where
    F: Fn(&RunSnapshot) -> Option<String> + Send + Sync,
{
    fn check(&self, snapshot: &RunSnapshot) -> Option<String> {
        self(snapshot)
    }
}

impl Limits {
    /// These limits, raised by a run's grants.
    pub(crate) fn with_grants(&self, grants: &LimitGrants) -> Limits {
        Limits {
            max_turns: self.max_turns.saturating_add(grants.turns),
            max_tokens: self.max_tokens.map(|m| m.saturating_add(grants.tokens)),
            max_tool_calls: self
                .max_tool_calls
                .map(|m| m.saturating_add(grants.tool_calls)),
            timeout: self
                .timeout
                .map(|t| t.saturating_add(Duration::from_millis(grants.wall_clock_ms))),
            ..self.clone()
        }
    }

    pub(crate) fn exceeded(&self, snapshot: &RunSnapshot) -> Option<LimitKind> {
        if snapshot.turn >= self.max_turns.min(HARD_TURN_CAP) {
            return Some(LimitKind::Turns);
        }
        if let Some(max) = self.max_tokens {
            if snapshot.usage.billable_tokens() >= max {
                return Some(LimitKind::Tokens);
            }
        }
        if let Some(max) = self.max_tool_calls {
            if snapshot.tool_calls >= max {
                return Some(LimitKind::ToolCalls);
            }
        }
        if let Some(max) = self.timeout {
            if snapshot.elapsed >= max {
                return Some(LimitKind::WallClock);
            }
        }
        None
    }
}
