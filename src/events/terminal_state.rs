//! Pillar B: per-turn terminal state for archived rendering.
//! Spec: 2026-06-06-cross-turn-prefix-stability-design.md §Rendering/§prerequisite.
use crate::events::TaskStatus;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalState {
    Completed,
    Failed,
    Cancelled,
    Interrupted,
}

impl TerminalState {
    /// `None` = no TaskEnd record exists for the turn (crash) → Interrupted.
    pub fn from_task_status(status: Option<TaskStatus>) -> Self {
        match status {
            Some(TaskStatus::Completed) => Self::Completed,
            Some(TaskStatus::Failed) => Self::Failed,
            Some(TaskStatus::Cancelled) => Self::Cancelled,
            None => Self::Interrupted,
        }
    }
    /// Fixed deterministic placeholder for a no-text-reply turn (spec §Rendering).
    pub fn placeholder(self, tool_steps: usize) -> String {
        match self {
            Self::Completed => format!("[completed: {tool_steps} tool steps, no text reply]"),
            Self::Failed => format!("[failed: {tool_steps} tool steps, no text reply]"),
            Self::Cancelled => format!("[cancelled: {tool_steps} tool steps, no text reply]"),
            Self::Interrupted => "[task interrupted]".to_string(),
        }
    }
    /// Stable string tag for fingerprinting/cache keys — decoupled from `Debug`
    /// so variant renames/reorders never silently invalidate `content_fp`.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_state_from_task_status() {
        assert_eq!(
            TerminalState::from_task_status(Some(TaskStatus::Completed)),
            TerminalState::Completed
        );
        assert_eq!(
            TerminalState::from_task_status(Some(TaskStatus::Failed)),
            TerminalState::Failed
        );
        assert_eq!(
            TerminalState::from_task_status(Some(TaskStatus::Cancelled)),
            TerminalState::Cancelled
        );
        // No TaskEnd record for the turn → crash/interrupted.
        assert_eq!(
            TerminalState::from_task_status(None),
            TerminalState::Interrupted
        );
    }

    #[test]
    fn terminal_state_placeholder_strings_are_fixed() {
        assert_eq!(
            TerminalState::Completed.placeholder(3),
            "[completed: 3 tool steps, no text reply]"
        );
        assert_eq!(
            TerminalState::Failed.placeholder(2),
            "[failed: 2 tool steps, no text reply]"
        );
        assert_eq!(
            TerminalState::Cancelled.placeholder(1),
            "[cancelled: 1 tool steps, no text reply]"
        );
        assert_eq!(
            TerminalState::Interrupted.placeholder(0),
            "[task interrupted]"
        );
    }
}
