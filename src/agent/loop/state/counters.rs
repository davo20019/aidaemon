use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Default)]
pub(in crate::agent) struct LoopCounters {
    iteration: usize,
    total_successful_tool_calls: usize,
    total_tool_calls_attempted: usize,
    tool_call_count: HashMap<String, usize>,
    tool_result_cache: HashMap<u64, String>,
    tool_result_cache_order: VecDeque<u64>,
    successful_send_file_keys: HashSet<String>,
    deferred_no_tool_streak: usize,
    deferred_no_tool_model_switches: usize,
    personal_memory_tool_calls: usize,
    needs_tools_for_turn: bool,
}

pub(in crate::agent) struct StoppingCountersState<'a> {
    pub deferred_no_tool_streak: usize,
    pub total_successful_tool_calls: usize,
    pub successful_send_file_keys: &'a HashSet<String>,
}

pub(in crate::agent) struct LlmCountersState {
    pub total_successful_tool_calls: usize,
    pub deferred_no_tool_streak: usize,
    pub tools_required_for_turn: bool,
}

pub(in crate::agent) struct ResponseCountersState<'a> {
    pub total_successful_tool_calls: usize,
    pub deferred_no_tool_streak: &'a mut usize,
    pub deferred_no_tool_model_switches: &'a mut usize,
    pub needs_tools_for_turn: &'a mut bool,
}

pub(in crate::agent) struct ToolExecutionCountersState<'a> {
    pub total_tool_calls_attempted: &'a mut usize,
    pub total_successful_tool_calls: &'a mut usize,
    pub tool_call_count: &'a mut HashMap<String, usize>,
    pub personal_memory_tool_calls: &'a mut usize,
    pub successful_send_file_keys: &'a mut HashSet<String>,
    pub deferred_no_tool_streak: &'a mut usize,
    pub tool_result_cache: &'a mut HashMap<u64, String>,
}

impl LoopCounters {
    pub(in crate::agent) fn new(needs_tools_for_turn: bool) -> Self {
        Self {
            needs_tools_for_turn,
            ..Self::default()
        }
    }

    pub(in crate::agent) fn for_stopping_phase(&mut self) -> StoppingCountersState<'_> {
        StoppingCountersState {
            deferred_no_tool_streak: self.deferred_no_tool_streak,
            total_successful_tool_calls: self.total_successful_tool_calls,
            successful_send_file_keys: &self.successful_send_file_keys,
        }
    }

    pub(in crate::agent) fn for_llm_phase(&self) -> LlmCountersState {
        LlmCountersState {
            total_successful_tool_calls: self.total_successful_tool_calls,
            deferred_no_tool_streak: self.deferred_no_tool_streak,
            tools_required_for_turn: self.needs_tools_for_turn,
        }
    }

    pub(in crate::agent) fn for_response_phase(&mut self) -> ResponseCountersState<'_> {
        ResponseCountersState {
            total_successful_tool_calls: self.total_successful_tool_calls,
            deferred_no_tool_streak: &mut self.deferred_no_tool_streak,
            deferred_no_tool_model_switches: &mut self.deferred_no_tool_model_switches,
            needs_tools_for_turn: &mut self.needs_tools_for_turn,
        }
    }

    pub(in crate::agent) fn for_tool_execution_phase(&mut self) -> ToolExecutionCountersState<'_> {
        ToolExecutionCountersState {
            total_tool_calls_attempted: &mut self.total_tool_calls_attempted,
            total_successful_tool_calls: &mut self.total_successful_tool_calls,
            tool_call_count: &mut self.tool_call_count,
            personal_memory_tool_calls: &mut self.personal_memory_tool_calls,
            successful_send_file_keys: &mut self.successful_send_file_keys,
            deferred_no_tool_streak: &mut self.deferred_no_tool_streak,
            tool_result_cache: &mut self.tool_result_cache,
        }
    }

    pub(in crate::agent) fn advance_iteration(&mut self) -> usize {
        self.iteration = self.iteration.saturating_add(1);
        self.iteration
    }

    pub(in crate::agent) fn iteration(&self) -> usize {
        self.iteration
    }

    pub(in crate::agent) fn record_tool_attempt(&mut self) {
        self.total_tool_calls_attempted = self.total_tool_calls_attempted.saturating_add(1);
    }

    pub(in crate::agent) fn total_tool_calls_attempted(&self) -> usize {
        self.total_tool_calls_attempted
    }

    pub(in crate::agent) fn record_successful_tool_call(&mut self, tool_name: &str) {
        self.total_successful_tool_calls = self.total_successful_tool_calls.saturating_add(1);
        *self
            .tool_call_count
            .entry(tool_name.to_string())
            .or_insert(0) += 1;
    }

    pub(in crate::agent) fn total_successful_tool_calls(&self) -> usize {
        self.total_successful_tool_calls
    }

    pub(in crate::agent) fn tool_call_count(&self, tool_name: &str) -> usize {
        self.tool_call_count.get(tool_name).copied().unwrap_or(0)
    }

    pub(in crate::agent) fn record_successful_send_file_key(&mut self, key: impl Into<String>) {
        self.successful_send_file_keys.insert(key.into());
    }

    pub(in crate::agent) fn has_successful_send_file_key(&self, key: &str) -> bool {
        self.successful_send_file_keys.contains(key)
    }

    pub(in crate::agent) fn successful_send_file_key_count(&self) -> usize {
        self.successful_send_file_keys.len()
    }

    pub(in crate::agent) fn record_personal_memory_tool_call(&mut self) {
        self.personal_memory_tool_calls = self.personal_memory_tool_calls.saturating_add(1);
    }

    pub(in crate::agent) fn personal_memory_tool_calls(&self) -> usize {
        self.personal_memory_tool_calls
    }

    pub(in crate::agent) fn increment_deferred_no_tool_streak(&mut self) {
        self.deferred_no_tool_streak = self.deferred_no_tool_streak.saturating_add(1);
    }

    pub(in crate::agent) fn reset_deferred_no_tool_streak(&mut self) {
        self.deferred_no_tool_streak = 0;
    }

    pub(in crate::agent) fn deferred_no_tool_streak(&self) -> usize {
        self.deferred_no_tool_streak
    }

    pub(in crate::agent) fn increment_deferred_no_tool_model_switches(&mut self) {
        self.deferred_no_tool_model_switches =
            self.deferred_no_tool_model_switches.saturating_add(1);
    }

    pub(in crate::agent) fn deferred_no_tool_model_switches(&self) -> usize {
        self.deferred_no_tool_model_switches
    }

    pub(in crate::agent) fn needs_tools_for_turn(&self) -> bool {
        self.needs_tools_for_turn
    }

    pub(in crate::agent) fn set_needs_tools_for_turn(&mut self, needs_tools: bool) {
        self.needs_tools_for_turn = needs_tools;
    }

    pub(in crate::agent) fn insert_tool_result_cache(
        &mut self,
        key: u64,
        value: String,
        max_entries: usize,
    ) {
        if !self.tool_result_cache.contains_key(&key) {
            self.tool_result_cache_order.push_back(key);
        }
        self.tool_result_cache.insert(key, value);
        while self.tool_result_cache.len() > max_entries {
            if let Some(oldest) = self.tool_result_cache_order.pop_front() {
                self.tool_result_cache.remove(&oldest);
            } else if let Some(key) = self.tool_result_cache.keys().next().copied() {
                self.tool_result_cache.remove(&key);
            } else {
                break;
            }
        }
    }

    pub(in crate::agent) fn take_tool_result_cache_entry(&mut self, key: u64) -> Option<String> {
        let value = self.tool_result_cache.remove(&key);
        if value.is_some() {
            self.tool_result_cache_order
                .retain(|existing| *existing != key);
        }
        value
    }

    pub(in crate::agent) fn tool_result_cache_len(&self) -> usize {
        self.tool_result_cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_iteration_increment() {
        let mut counters = LoopCounters::new(false);

        assert_eq!(counters.iteration(), 0);
        assert_eq!(counters.advance_iteration(), 1);
        assert_eq!(counters.advance_iteration(), 2);
        assert_eq!(counters.iteration(), 2);
    }

    #[test]
    fn accumulates_tool_call_counts() {
        let mut counters = LoopCounters::new(false);

        counters.record_tool_attempt();
        counters.record_tool_attempt();
        counters.record_successful_tool_call("read_file");
        counters.record_successful_tool_call("read_file");
        counters.record_successful_tool_call("terminal");

        assert_eq!(counters.total_tool_calls_attempted(), 2);
        assert_eq!(counters.total_successful_tool_calls(), 3);
        assert_eq!(counters.tool_call_count("read_file"), 2);
        assert_eq!(counters.tool_call_count("terminal"), 1);
        assert_eq!(counters.tool_call_count("missing"), 0);
    }

    #[test]
    fn tracks_send_file_keys() {
        let mut counters = LoopCounters::new(false);

        assert!(!counters.has_successful_send_file_key("telegram:file.txt"));
        counters.record_successful_send_file_key("telegram:file.txt");

        assert!(counters.has_successful_send_file_key("telegram:file.txt"));
        assert_eq!(counters.successful_send_file_key_count(), 1);
    }

    #[test]
    fn counts_personal_memory_tool_calls() {
        let mut counters = LoopCounters::new(false);

        counters.record_personal_memory_tool_call();
        counters.record_personal_memory_tool_call();

        assert_eq!(counters.personal_memory_tool_calls(), 2);
    }

    #[test]
    fn tracks_deferred_no_tool_state_and_tool_need() {
        let mut counters = LoopCounters::new(true);

        assert!(counters.needs_tools_for_turn());
        assert_eq!(counters.deferred_no_tool_streak(), 0);
        counters.increment_deferred_no_tool_streak();
        counters.increment_deferred_no_tool_streak();
        counters.increment_deferred_no_tool_model_switches();

        assert_eq!(counters.deferred_no_tool_streak(), 2);
        assert_eq!(counters.deferred_no_tool_model_switches(), 1);

        counters.reset_deferred_no_tool_streak();
        counters.set_needs_tools_for_turn(false);

        assert_eq!(counters.deferred_no_tool_streak(), 0);
        assert!(!counters.needs_tools_for_turn());
    }

    #[test]
    fn inserts_and_takes_tool_result_cache_entries() {
        let mut counters = LoopCounters::new(false);

        counters.insert_tool_result_cache(10, "first".to_string(), 2);
        counters.insert_tool_result_cache(20, "second".to_string(), 2);
        counters.insert_tool_result_cache(30, "third".to_string(), 2);

        assert_eq!(counters.tool_result_cache_len(), 2);
        assert_eq!(counters.take_tool_result_cache_entry(10), None);
        assert_eq!(
            counters.take_tool_result_cache_entry(20),
            Some("second".to_string())
        );
        assert_eq!(counters.tool_result_cache_len(), 1);
    }
}
