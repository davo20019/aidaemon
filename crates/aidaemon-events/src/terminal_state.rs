//! Pillar B: per-turn terminal state for archived rendering.
//! Spec: 2026-06-06-cross-turn-prefix-stability-design.md §Rendering/§prerequisite.
use crate::TaskStatus;

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
            Some(TaskStatus::Interrupted) | None => Self::Interrupted,
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
    /// The note an earlier turn carries in the model's history when it
    /// stopped before it finished: stopped by the owner (`Cancelled`), or cut
    /// off with no finish of its own (`Interrupted`: the daemon restarted, or
    /// no `task_end` was ever written). Without it the request reads as still
    /// open and the next turn takes it up again instead of answering the new
    /// message. `None` for turns that finished (or failed) on their own.
    pub fn cut_off_marker(self) -> Option<&'static str> {
        match self {
            Self::Cancelled => Some(
                "[Request stopped by the owner before it finished. It is not waiting for a \
                 reply: answer the owner's newest message on its own, and pick this request \
                 up again only if the owner asks to continue it.]",
            ),
            Self::Interrupted => Some(
                "[Request interrupted before it finished (the daemon restarted or the run was \
                 cut off). It is not waiting for a reply: answer the owner's newest message on \
                 its own, and pick this request up again only if the owner asks to continue it.]",
            ),
            Self::Completed | Self::Failed => None,
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
        assert_eq!(
            TerminalState::from_task_status(Some(TaskStatus::Interrupted)),
            TerminalState::Interrupted
        );
        // No TaskEnd record for the turn → crash/interrupted.
        assert_eq!(
            TerminalState::from_task_status(None),
            TerminalState::Interrupted
        );
    }

    #[test]
    fn cut_off_marker_only_for_turns_that_did_not_finish() {
        assert!(TerminalState::Completed.cut_off_marker().is_none());
        assert!(TerminalState::Failed.cut_off_marker().is_none());
        let stopped = TerminalState::Cancelled.cut_off_marker().unwrap();
        let cut_off = TerminalState::Interrupted.cut_off_marker().unwrap();
        assert!(stopped.contains("stopped by the owner"));
        assert!(cut_off.contains("daemon restarted"));
        // A turn with no recorded end is one the daemon never finished.
        assert_eq!(
            TerminalState::from_task_status(None).cut_off_marker(),
            Some(cut_off)
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
