//! Tool-loop state extraction scaffolding.
//!
//! This captures iteration-level bookkeeping currently interleaved in
//! `main_loop.rs` so the loop can be reduced to orchestration code.
#![allow(dead_code)]

/// Per-iteration execution result needed for loop-state updates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IterationProgress {
    pub successful_tool_calls: usize,
    pub iteration_had_tool_failures: bool,
}

/// Signals produced after applying one iteration's progress.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IterationSignals {
    pub no_progress: bool,
    pub should_attempt_fallback_expansion: bool,
}

/// Mutable loop counters used across iterations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolLoopState {
    pub stall_count: usize,
    pub total_successful_tool_calls: usize,
    pub consecutive_clean_iterations: usize,
    pub fallback_expanded_once: bool,
}

impl ToolLoopState {
    /// Apply one iteration's progress and return derived signals.
    ///
    /// This mirrors the current stall/no-progress behavior:
    /// - no successful calls => stall_count++, no-progress signal
    /// - exactly on stall_count == 2 and not yet expanded => trigger fallback signal
    /// - any success => stall reset + total_successful increment
    pub fn apply_iteration(&mut self, progress: IterationProgress) -> IterationSignals {
        if progress.successful_tool_calls == 0 {
            self.stall_count = self.stall_count.saturating_add(1);
            self.consecutive_clean_iterations = 0;

            let should_attempt_fallback_expansion =
                self.stall_count == 2 && !self.fallback_expanded_once;
            if should_attempt_fallback_expansion {
                self.fallback_expanded_once = true;
            }

            return IterationSignals {
                no_progress: true,
                should_attempt_fallback_expansion,
            };
        }

        self.stall_count = 0;
        self.total_successful_tool_calls = self
            .total_successful_tool_calls
            .saturating_add(progress.successful_tool_calls);
        if progress.iteration_had_tool_failures {
            self.consecutive_clean_iterations = 0;
        } else {
            self.consecutive_clean_iterations = self.consecutive_clean_iterations.saturating_add(1);
        }

        IterationSignals {
            no_progress: false,
            should_attempt_fallback_expansion: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_no_progress_iteration_marks_no_progress_without_expansion() {
        let mut state = ToolLoopState::default();
        let signals = state.apply_iteration(IterationProgress {
            successful_tool_calls: 0,
            iteration_had_tool_failures: false,
        });
        assert!(signals.no_progress);
        assert!(!signals.should_attempt_fallback_expansion);
        assert_eq!(state.stall_count, 1);
    }

    #[test]
    fn second_no_progress_iteration_requests_fallback_expansion_once() {
        let mut state = ToolLoopState::default();
        let _ = state.apply_iteration(IterationProgress {
            successful_tool_calls: 0,
            iteration_had_tool_failures: false,
        });
        let signals = state.apply_iteration(IterationProgress {
            successful_tool_calls: 0,
            iteration_had_tool_failures: false,
        });
        assert!(signals.no_progress);
        assert!(signals.should_attempt_fallback_expansion);
        assert!(state.fallback_expanded_once);

        let signals_after = state.apply_iteration(IterationProgress {
            successful_tool_calls: 0,
            iteration_had_tool_failures: false,
        });
        assert!(signals_after.no_progress);
        assert!(!signals_after.should_attempt_fallback_expansion);
    }

    #[test]
    fn successful_iteration_resets_stall_and_tracks_clean_streak() {
        let mut state = ToolLoopState {
            stall_count: 2,
            total_successful_tool_calls: 3,
            consecutive_clean_iterations: 0,
            fallback_expanded_once: true,
        };
        let signals = state.apply_iteration(IterationProgress {
            successful_tool_calls: 2,
            iteration_had_tool_failures: false,
        });
        assert!(!signals.no_progress);
        assert_eq!(state.stall_count, 0);
        assert_eq!(state.total_successful_tool_calls, 5);
        assert_eq!(state.consecutive_clean_iterations, 1);
    }
}
