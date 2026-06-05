use crate::agent::tool_execution_phase::PendingReflectionRecovery;

#[derive(Debug, Default)]
pub(in crate::agent) struct ReflectionState {
    reflection_completed: std::collections::HashSet<(String, String)>,
    pending_reflection_recoveries: std::collections::HashMap<String, PendingReflectionRecovery>,
    in_session_learned: std::collections::HashSet<(String, String)>,
    pending_error_solution_ids: Vec<i64>,
}

pub(in crate::agent) struct ToolExecutionReflectionState<'a> {
    pub pending_error_solution_ids: &'a mut Vec<i64>,
    pub reflection_completed: &'a mut std::collections::HashSet<(String, String)>,
    pub pending_reflection_recoveries:
        &'a mut std::collections::HashMap<String, PendingReflectionRecovery>,
    pub in_session_learned: &'a mut std::collections::HashSet<(String, String)>,
}

impl ReflectionState {
    pub(in crate::agent) fn for_tool_execution_phase(
        &mut self,
    ) -> ToolExecutionReflectionState<'_> {
        ToolExecutionReflectionState {
            pending_error_solution_ids: &mut self.pending_error_solution_ids,
            reflection_completed: &mut self.reflection_completed,
            pending_reflection_recoveries: &mut self.pending_reflection_recoveries,
            in_session_learned: &mut self.in_session_learned,
        }
    }

    pub(in crate::agent) fn push_pending_error_solution_id(&mut self, solution_id: i64) {
        self.pending_error_solution_ids.push(solution_id);
    }

    pub(in crate::agent) fn pending_error_solution_ids(&self) -> &[i64] {
        &self.pending_error_solution_ids
    }

    pub(in crate::agent) fn take_pending_error_solution_ids(&mut self) -> Vec<i64> {
        std::mem::take(&mut self.pending_error_solution_ids)
    }

    pub(in crate::agent) fn mark_reflection_completed(
        &mut self,
        tool_name: impl Into<String>,
        signature: impl Into<String>,
    ) -> bool {
        self.reflection_completed
            .insert((tool_name.into(), signature.into()))
    }

    pub(in crate::agent) fn reflection_completed_count(&self) -> usize {
        self.reflection_completed.len()
    }

    pub(in crate::agent) fn insert_pending_reflection_recovery(
        &mut self,
        tool_name: impl Into<String>,
        recovery: PendingReflectionRecovery,
    ) -> Option<PendingReflectionRecovery> {
        self.pending_reflection_recoveries
            .insert(tool_name.into(), recovery)
    }

    pub(in crate::agent) fn take_pending_reflection_recovery(
        &mut self,
        tool_name: &str,
    ) -> Option<PendingReflectionRecovery> {
        self.pending_reflection_recoveries.remove(tool_name)
    }

    pub(in crate::agent) fn pending_reflection_recovery_count(&self) -> usize {
        self.pending_reflection_recoveries.len()
    }

    pub(in crate::agent) fn mark_in_session_learned(
        &mut self,
        tool_name: impl Into<String>,
        signature: impl Into<String>,
    ) -> bool {
        self.in_session_learned
            .insert((tool_name.into(), signature.into()))
    }

    pub(in crate::agent) fn in_session_learned_count(&self) -> usize {
        self.in_session_learned.len()
    }
}

#[cfg(test)]
mod tests {
    use super::ReflectionState;
    use crate::agent::tool_execution_phase::PendingReflectionRecovery;

    fn recovery(id: i64, verify_on_iteration: usize) -> PendingReflectionRecovery {
        PendingReflectionRecovery {
            signature: format!("sig-{id}"),
            solution_ids: vec![id],
            verify_on_iteration,
        }
    }

    #[test]
    fn default_state_has_no_reflection_entries() {
        let state = ReflectionState::default();

        assert!(state.pending_error_solution_ids().is_empty());
        assert_eq!(state.reflection_completed_count(), 0);
        assert_eq!(state.pending_reflection_recovery_count(), 0);
        assert_eq!(state.in_session_learned_count(), 0);
    }

    #[test]
    fn stores_and_drains_pending_error_solution_ids() {
        let mut state = ReflectionState::default();

        state.push_pending_error_solution_id(10);
        state.push_pending_error_solution_id(11);

        assert_eq!(state.pending_error_solution_ids(), &[10, 11]);
        assert_eq!(state.take_pending_error_solution_ids(), vec![10, 11]);
        assert!(state.pending_error_solution_ids().is_empty());
    }

    #[test]
    fn stores_reflection_completion_keys_without_duplicates() {
        let mut state = ReflectionState::default();

        assert!(state.mark_reflection_completed("read_file", "missing-path"));
        assert!(!state.mark_reflection_completed("read_file", "missing-path"));
        assert!(state.mark_reflection_completed("search_files", "missing-path"));

        assert_eq!(state.reflection_completed_count(), 2);
    }

    #[test]
    fn stores_and_drains_pending_reflection_recoveries() {
        let mut state = ReflectionState::default();

        state.insert_pending_reflection_recovery("terminal", recovery(1, 3));
        state.insert_pending_reflection_recovery("read_file", recovery(2, 4));

        assert_eq!(state.pending_reflection_recovery_count(), 2);
        assert_eq!(
            state
                .take_pending_reflection_recovery("terminal")
                .expect("terminal recovery")
                .solution_ids,
            vec![1]
        );
        assert_eq!(state.pending_reflection_recovery_count(), 1);
    }

    #[test]
    fn stores_in_session_learned_keys_without_duplicates() {
        let mut state = ReflectionState::default();

        assert!(state.mark_in_session_learned("terminal", "permission-denied"));
        assert!(!state.mark_in_session_learned("terminal", "permission-denied"));

        assert_eq!(state.in_session_learned_count(), 1);
    }
}
