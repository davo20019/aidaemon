#[derive(Debug, Default)]
pub(in crate::agent) struct RecoveryState {
    force_text_response: bool,
    force_text_iterations: usize,
    empty_response_retry_used: bool,
    empty_response_retry_pending: bool,
    empty_response_retry_note: Option<String>,
    truncated_text_prefix: Option<String>,
    thinking_truncation_count: u8,
    fallback_expanded_once: bool,
}

pub(in crate::agent) struct StoppingRecoveryState<'a> {
    pub force_text_response: &'a mut bool,
}

pub(in crate::agent) struct MessageBuildRecoveryState {
    pub empty_response_retry_pending: bool,
}

pub(in crate::agent) struct LlmRecoveryState<'a> {
    pub force_text_response: bool,
    pub empty_response_retry_pending: &'a mut bool,
    pub empty_response_retry_note: &'a mut Option<String>,
    pub truncated_text_prefix: &'a mut Option<String>,
    pub thinking_truncation_count: &'a mut u8,
}

pub(in crate::agent) struct ResponseRecoveryState<'a> {
    pub fallback_expanded_once: &'a mut bool,
    pub empty_response_retry_used: &'a mut bool,
    pub empty_response_retry_pending: &'a mut bool,
    pub empty_response_retry_note: &'a mut Option<String>,
    pub force_text_response: &'a mut bool,
}

pub(in crate::agent) struct ToolPreludeRecoveryState<'a> {
    pub force_text_response: &'a mut bool,
}

pub(in crate::agent) struct ToolExecutionRecoveryState<'a> {
    pub force_text_response: &'a mut bool,
    pub fallback_expanded_once: &'a mut bool,
}

impl RecoveryState {
    pub(in crate::agent) fn for_stopping_phase(&mut self) -> StoppingRecoveryState<'_> {
        StoppingRecoveryState {
            force_text_response: &mut self.force_text_response,
        }
    }

    pub(in crate::agent) fn for_message_build_phase(&self) -> MessageBuildRecoveryState {
        MessageBuildRecoveryState {
            empty_response_retry_pending: self.empty_response_retry_pending,
        }
    }

    pub(in crate::agent) fn for_llm_phase(&mut self) -> LlmRecoveryState<'_> {
        LlmRecoveryState {
            force_text_response: self.force_text_response,
            empty_response_retry_pending: &mut self.empty_response_retry_pending,
            empty_response_retry_note: &mut self.empty_response_retry_note,
            truncated_text_prefix: &mut self.truncated_text_prefix,
            thinking_truncation_count: &mut self.thinking_truncation_count,
        }
    }

    pub(in crate::agent) fn for_response_phase(&mut self) -> ResponseRecoveryState<'_> {
        ResponseRecoveryState {
            fallback_expanded_once: &mut self.fallback_expanded_once,
            empty_response_retry_used: &mut self.empty_response_retry_used,
            empty_response_retry_pending: &mut self.empty_response_retry_pending,
            empty_response_retry_note: &mut self.empty_response_retry_note,
            force_text_response: &mut self.force_text_response,
        }
    }

    pub(in crate::agent) fn for_tool_prelude_phase(&mut self) -> ToolPreludeRecoveryState<'_> {
        ToolPreludeRecoveryState {
            force_text_response: &mut self.force_text_response,
        }
    }

    pub(in crate::agent) fn for_tool_execution_phase(&mut self) -> ToolExecutionRecoveryState<'_> {
        ToolExecutionRecoveryState {
            force_text_response: &mut self.force_text_response,
            fallback_expanded_once: &mut self.fallback_expanded_once,
        }
    }

    pub(in crate::agent) fn force_text_response(&self) -> bool {
        self.force_text_response
    }

    pub(in crate::agent) fn set_force_text_response(&mut self, value: bool) {
        self.force_text_response = value;
    }

    pub(in crate::agent) fn force_text_iterations(&self) -> usize {
        self.force_text_iterations
    }

    pub(in crate::agent) fn record_force_text_iteration(&mut self) -> usize {
        self.force_text_iterations = self.force_text_iterations.saturating_add(1);
        self.force_text_iterations
    }

    pub(in crate::agent) fn reset_force_text_iterations(&mut self) {
        self.force_text_iterations = 0;
    }

    pub(in crate::agent) fn empty_response_retry_used(&self) -> bool {
        self.empty_response_retry_used
    }

    pub(in crate::agent) fn empty_response_retry_pending(&self) -> bool {
        self.empty_response_retry_pending
    }

    pub(in crate::agent) fn empty_response_retry_note(&self) -> Option<&str> {
        self.empty_response_retry_note.as_deref()
    }

    pub(in crate::agent) fn schedule_empty_response_retry(&mut self, note: String) {
        self.empty_response_retry_used = true;
        self.empty_response_retry_pending = true;
        self.empty_response_retry_note = Some(note);
    }

    pub(in crate::agent) fn clear_empty_response_retry_pending(&mut self) {
        self.empty_response_retry_pending = false;
        self.empty_response_retry_note = None;
    }

    pub(in crate::agent) fn truncated_text_prefix(&self) -> Option<&str> {
        self.truncated_text_prefix.as_deref()
    }

    pub(in crate::agent) fn set_truncated_text_prefix(&mut self, prefix: String) {
        self.truncated_text_prefix = Some(prefix);
    }

    pub(in crate::agent) fn take_truncated_text_prefix(&mut self) -> Option<String> {
        self.truncated_text_prefix.take()
    }

    pub(in crate::agent) fn thinking_truncation_count(&self) -> u8 {
        self.thinking_truncation_count
    }

    pub(in crate::agent) fn increment_thinking_truncation_count(&mut self) -> u8 {
        self.thinking_truncation_count = self.thinking_truncation_count.saturating_add(1);
        self.thinking_truncation_count
    }

    pub(in crate::agent) fn reset_thinking_truncation_count(&mut self) {
        self.thinking_truncation_count = 0;
    }

    pub(in crate::agent) fn fallback_expanded_once(&self) -> bool {
        self.fallback_expanded_once
    }

    pub(in crate::agent) fn mark_fallback_expanded(&mut self) {
        self.fallback_expanded_once = true;
    }
}

#[cfg(test)]
mod tests {
    use super::RecoveryState;

    #[test]
    fn default_state_has_no_active_recovery() {
        let state = RecoveryState::default();

        assert!(!state.force_text_response());
        assert_eq!(state.force_text_iterations(), 0);
        assert!(!state.empty_response_retry_used());
        assert!(!state.empty_response_retry_pending());
        assert_eq!(state.empty_response_retry_note(), None);
        assert_eq!(state.truncated_text_prefix(), None);
        assert_eq!(state.thinking_truncation_count(), 0);
        assert!(!state.fallback_expanded_once());
    }

    #[test]
    fn tracks_force_text_activation_and_iteration_reset() {
        let mut state = RecoveryState::default();

        state.set_force_text_response(true);
        assert!(state.force_text_response());
        assert_eq!(state.record_force_text_iteration(), 1);
        assert_eq!(state.record_force_text_iteration(), 2);

        state.set_force_text_response(false);
        state.reset_force_text_iterations();
        assert!(!state.force_text_response());
        assert_eq!(state.force_text_iterations(), 0);
    }

    #[test]
    fn tracks_empty_response_retry_flags_and_note() {
        let mut state = RecoveryState::default();

        state.schedule_empty_response_retry("recover with concise answer".to_string());
        assert!(state.empty_response_retry_used());
        assert!(state.empty_response_retry_pending());
        assert_eq!(
            state.empty_response_retry_note(),
            Some("recover with concise answer")
        );

        state.clear_empty_response_retry_pending();
        assert!(!state.empty_response_retry_pending());
        assert_eq!(state.empty_response_retry_note(), None);
        assert!(state.empty_response_retry_used());
    }

    #[test]
    fn stores_and_takes_truncated_text_prefix() {
        let mut state = RecoveryState::default();

        state.set_truncated_text_prefix("partial".to_string());
        assert_eq!(state.truncated_text_prefix(), Some("partial"));
        assert_eq!(
            state.take_truncated_text_prefix(),
            Some("partial".to_string())
        );
        assert_eq!(state.truncated_text_prefix(), None);
    }

    #[test]
    fn tracks_thinking_truncation_count_and_reset() {
        let mut state = RecoveryState::default();

        assert_eq!(state.increment_thinking_truncation_count(), 1);
        assert_eq!(state.increment_thinking_truncation_count(), 2);
        state.reset_thinking_truncation_count();

        assert_eq!(state.thinking_truncation_count(), 0);
    }

    #[test]
    fn tracks_fallback_expansion_flag_once() {
        let mut state = RecoveryState::default();

        assert!(!state.fallback_expanded_once());
        state.mark_fallback_expanded();
        assert!(state.fallback_expanded_once());
        state.mark_fallback_expanded();
        assert!(state.fallback_expanded_once());
    }
}
