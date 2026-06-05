use std::collections::{HashSet, VecDeque};

#[derive(Debug, Default)]
pub(in crate::agent) struct StallTracker {
    stall_count: usize,
    consecutive_same_tool: (String, usize),
    consecutive_same_tool_arg_hashes: HashSet<u64>,
    recent_tool_calls: VecDeque<u64>,
    recent_tool_names: VecDeque<String>,
    consecutive_clean_iterations: usize,
    last_escalation_iteration: Option<usize>,
}

pub(in crate::agent) struct StoppingStallState<'a> {
    pub stall_count: usize,
    pub consecutive_same_tool: &'a (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a HashSet<u64>,
    pub last_escalation_iteration: &'a mut Option<usize>,
    pub consecutive_clean_iterations: &'a mut usize,
}

pub(in crate::agent) struct LlmStallState<'a> {
    pub stall_count: &'a mut usize,
    pub consecutive_same_tool: &'a (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a HashSet<u64>,
}

pub(in crate::agent) struct ResponseStallState<'a> {
    pub stall_count: &'a mut usize,
    pub consecutive_clean_iterations: &'a mut usize,
}

pub(in crate::agent) struct ToolExecutionStallState<'a> {
    pub recent_tool_calls: &'a mut VecDeque<u64>,
    pub consecutive_same_tool: &'a mut (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a mut HashSet<u64>,
    pub recent_tool_names: &'a mut VecDeque<String>,
    pub stall_count: &'a mut usize,
    pub consecutive_clean_iterations: &'a mut usize,
}

impl StallTracker {
    pub(in crate::agent) fn with_recent_capacity(capacity: usize) -> Self {
        Self {
            recent_tool_calls: VecDeque::with_capacity(capacity),
            ..Self::default()
        }
    }

    pub(in crate::agent) fn for_stopping_phase(&mut self) -> StoppingStallState<'_> {
        StoppingStallState {
            stall_count: self.stall_count,
            consecutive_same_tool: &self.consecutive_same_tool,
            consecutive_same_tool_arg_hashes: &self.consecutive_same_tool_arg_hashes,
            last_escalation_iteration: &mut self.last_escalation_iteration,
            consecutive_clean_iterations: &mut self.consecutive_clean_iterations,
        }
    }

    pub(in crate::agent) fn for_llm_phase(&mut self) -> LlmStallState<'_> {
        LlmStallState {
            stall_count: &mut self.stall_count,
            consecutive_same_tool: &self.consecutive_same_tool,
            consecutive_same_tool_arg_hashes: &self.consecutive_same_tool_arg_hashes,
        }
    }

    pub(in crate::agent) fn for_response_phase(&mut self) -> ResponseStallState<'_> {
        ResponseStallState {
            stall_count: &mut self.stall_count,
            consecutive_clean_iterations: &mut self.consecutive_clean_iterations,
        }
    }

    pub(in crate::agent) fn for_tool_execution_phase(&mut self) -> ToolExecutionStallState<'_> {
        ToolExecutionStallState {
            recent_tool_calls: &mut self.recent_tool_calls,
            consecutive_same_tool: &mut self.consecutive_same_tool,
            consecutive_same_tool_arg_hashes: &mut self.consecutive_same_tool_arg_hashes,
            recent_tool_names: &mut self.recent_tool_names,
            stall_count: &mut self.stall_count,
            consecutive_clean_iterations: &mut self.consecutive_clean_iterations,
        }
    }

    pub(in crate::agent) fn stall_count(&self) -> usize {
        self.stall_count
    }

    pub(in crate::agent) fn stall_count_mut(&mut self) -> &mut usize {
        &mut self.stall_count
    }

    pub(in crate::agent) fn consecutive_same_tool(&self) -> (&str, usize) {
        (&self.consecutive_same_tool.0, self.consecutive_same_tool.1)
    }

    pub(in crate::agent) fn consecutive_same_tool_ref(&self) -> &(String, usize) {
        &self.consecutive_same_tool
    }

    pub(in crate::agent) fn consecutive_same_tool_mut(&mut self) -> &mut (String, usize) {
        &mut self.consecutive_same_tool
    }

    pub(in crate::agent) fn consecutive_same_tool_arg_hash_count(&self) -> usize {
        self.consecutive_same_tool_arg_hashes.len()
    }

    pub(in crate::agent) fn consecutive_same_tool_arg_hashes_ref(&self) -> &HashSet<u64> {
        &self.consecutive_same_tool_arg_hashes
    }

    pub(in crate::agent) fn consecutive_same_tool_arg_hashes_mut(&mut self) -> &mut HashSet<u64> {
        &mut self.consecutive_same_tool_arg_hashes
    }

    pub(in crate::agent) fn recent_tool_call_count(&self) -> usize {
        self.recent_tool_calls.len()
    }

    pub(in crate::agent) fn recent_tool_calls_mut(&mut self) -> &mut VecDeque<u64> {
        &mut self.recent_tool_calls
    }

    pub(in crate::agent) fn recent_tool_name_count(&self) -> usize {
        self.recent_tool_names.len()
    }

    pub(in crate::agent) fn recent_tool_names_mut(&mut self) -> &mut VecDeque<String> {
        &mut self.recent_tool_names
    }

    pub(in crate::agent) fn consecutive_clean_iterations(&self) -> usize {
        self.consecutive_clean_iterations
    }

    pub(in crate::agent) fn consecutive_clean_iterations_mut(&mut self) -> &mut usize {
        &mut self.consecutive_clean_iterations
    }

    pub(in crate::agent) fn last_escalation_iteration(&self) -> Option<usize> {
        self.last_escalation_iteration
    }

    pub(in crate::agent) fn last_escalation_iteration_mut(&mut self) -> &mut Option<usize> {
        &mut self.last_escalation_iteration
    }

    pub(in crate::agent) fn record_tool_call(&mut self, tool_name: &str, arg_hash: u64) {
        if self.consecutive_same_tool.0 == tool_name {
            self.consecutive_same_tool.1 = self.consecutive_same_tool.1.saturating_add(1);
        } else {
            self.consecutive_same_tool = (tool_name.to_string(), 1);
            self.consecutive_same_tool_arg_hashes.clear();
        }

        self.consecutive_same_tool_arg_hashes.insert(arg_hash);
        self.recent_tool_calls.push_back(arg_hash);
        self.recent_tool_names.push_back(tool_name.to_string());
    }

    pub(in crate::agent) fn record_clean_iteration(&mut self) {
        self.consecutive_clean_iterations = self.consecutive_clean_iterations.saturating_add(1);
    }

    pub(in crate::agent) fn increment_stall(&mut self) {
        self.stall_count = self.stall_count.saturating_add(1);
        self.consecutive_clean_iterations = 0;
    }

    pub(in crate::agent) fn reset_stall_after_progress(&mut self) {
        self.stall_count = 0;
        self.record_clean_iteration();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_has_no_stall_or_tool_history() {
        let tracker = StallTracker::default();

        assert_eq!(tracker.stall_count(), 0);
        assert_eq!(tracker.consecutive_same_tool(), ("", 0));
        assert_eq!(tracker.consecutive_same_tool_arg_hash_count(), 0);
        assert_eq!(tracker.recent_tool_call_count(), 0);
        assert_eq!(tracker.recent_tool_name_count(), 0);
        assert_eq!(tracker.consecutive_clean_iterations(), 0);
        assert_eq!(tracker.last_escalation_iteration(), None);
    }

    #[test]
    fn repeated_tool_call_increments_streak_and_tracks_unique_arg_hashes() {
        let mut tracker = StallTracker::default();

        tracker.record_tool_call("terminal", 11);
        tracker.record_tool_call("terminal", 11);
        tracker.record_tool_call("terminal", 22);

        assert_eq!(tracker.consecutive_same_tool(), ("terminal", 3));
        assert_eq!(tracker.consecutive_same_tool_arg_hash_count(), 2);
        assert_eq!(tracker.recent_tool_call_count(), 3);
        assert_eq!(tracker.recent_tool_name_count(), 3);
    }

    #[test]
    fn different_tool_resets_same_tool_streak_and_arg_hashes() {
        let mut tracker = StallTracker::default();

        tracker.record_tool_call("terminal", 11);
        tracker.record_tool_call("terminal", 22);
        tracker.record_tool_call("read_file", 33);

        assert_eq!(tracker.consecutive_same_tool(), ("read_file", 1));
        assert_eq!(tracker.consecutive_same_tool_arg_hash_count(), 1);
        assert_eq!(tracker.recent_tool_call_count(), 3);
        assert_eq!(tracker.recent_tool_name_count(), 3);
    }

    #[test]
    fn clean_iteration_tracking_resets_after_stall_increment() {
        let mut tracker = StallTracker::default();

        tracker.record_clean_iteration();
        tracker.record_clean_iteration();
        assert_eq!(tracker.consecutive_clean_iterations(), 2);

        tracker.increment_stall();
        assert_eq!(tracker.stall_count(), 1);
        assert_eq!(tracker.consecutive_clean_iterations(), 0);

        tracker.reset_stall_after_progress();
        assert_eq!(tracker.stall_count(), 0);
        assert_eq!(tracker.consecutive_clean_iterations(), 1);
    }
}
