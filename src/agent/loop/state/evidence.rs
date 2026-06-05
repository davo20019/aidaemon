#[derive(Debug, Default)]
pub(in crate::agent) struct EvidenceLedger {
    evidence_state: crate::agent::EvidenceState,
    validation_state: crate::agent::ValidationState,
    no_evidence_result_streak: usize,
    evidence_gain_count: usize,
    no_evidence_tools_seen: std::collections::HashSet<String>,
    known_project_dir: Option<String>,
    dirs_with_project_inspect_file_evidence: std::collections::HashSet<String>,
    dirs_with_search_no_matches: std::collections::HashSet<String>,
    require_file_recheck_before_answer: bool,
}

pub(in crate::agent) struct StoppingEvidenceState<'a> {
    pub evidence_gain_count: usize,
    pub validation_state: &'a mut crate::agent::ValidationState,
}

pub(in crate::agent) struct LlmEvidenceState {
    pub evidence_gain_count: usize,
}

pub(in crate::agent) struct ResponseEvidenceState<'a> {
    pub require_file_recheck_before_answer: &'a mut bool,
    pub validation_state: &'a mut crate::agent::ValidationState,
}

pub(in crate::agent) struct ToolPreludeEvidenceState<'a> {
    pub evidence_state: &'a crate::agent::EvidenceState,
    pub validation_state: &'a mut crate::agent::ValidationState,
}

pub(in crate::agent) struct ToolExecutionEvidenceState<'a> {
    pub no_evidence_result_streak: &'a mut usize,
    pub no_evidence_tools_seen: &'a mut std::collections::HashSet<String>,
    pub evidence_gain_count: &'a mut usize,
    pub evidence_state: &'a mut crate::agent::EvidenceState,
    pub known_project_dir: &'a mut Option<String>,
    pub dirs_with_project_inspect_file_evidence: &'a mut std::collections::HashSet<String>,
    pub dirs_with_search_no_matches: &'a mut std::collections::HashSet<String>,
    pub require_file_recheck_before_answer: &'a mut bool,
    pub validation_state: &'a mut crate::agent::ValidationState,
}

impl EvidenceLedger {
    pub(in crate::agent) fn with_known_project_dir(known_project_dir: Option<String>) -> Self {
        Self {
            known_project_dir,
            ..Self::default()
        }
    }

    pub(in crate::agent) fn for_stopping_phase(&mut self) -> StoppingEvidenceState<'_> {
        StoppingEvidenceState {
            evidence_gain_count: self.evidence_gain_count,
            validation_state: &mut self.validation_state,
        }
    }

    pub(in crate::agent) fn for_llm_phase(&self) -> LlmEvidenceState {
        LlmEvidenceState {
            evidence_gain_count: self.evidence_gain_count,
        }
    }

    pub(in crate::agent) fn for_response_phase(&mut self) -> ResponseEvidenceState<'_> {
        ResponseEvidenceState {
            require_file_recheck_before_answer: &mut self.require_file_recheck_before_answer,
            validation_state: &mut self.validation_state,
        }
    }

    pub(in crate::agent) fn for_tool_prelude_phase(&mut self) -> ToolPreludeEvidenceState<'_> {
        ToolPreludeEvidenceState {
            evidence_state: &self.evidence_state,
            validation_state: &mut self.validation_state,
        }
    }

    pub(in crate::agent) fn for_tool_execution_phase(&mut self) -> ToolExecutionEvidenceState<'_> {
        ToolExecutionEvidenceState {
            no_evidence_result_streak: &mut self.no_evidence_result_streak,
            no_evidence_tools_seen: &mut self.no_evidence_tools_seen,
            evidence_gain_count: &mut self.evidence_gain_count,
            evidence_state: &mut self.evidence_state,
            known_project_dir: &mut self.known_project_dir,
            dirs_with_project_inspect_file_evidence: &mut self
                .dirs_with_project_inspect_file_evidence,
            dirs_with_search_no_matches: &mut self.dirs_with_search_no_matches,
            require_file_recheck_before_answer: &mut self.require_file_recheck_before_answer,
            validation_state: &mut self.validation_state,
        }
    }

    pub(in crate::agent) fn validation_state_mut(&mut self) -> &mut crate::agent::ValidationState {
        &mut self.validation_state
    }

    pub(in crate::agent) fn evidence_gain_count(&self) -> usize {
        self.evidence_gain_count
    }

    pub(in crate::agent) fn increment_evidence_gain_count(&mut self) -> usize {
        self.evidence_gain_count = self.evidence_gain_count.saturating_add(1);
        self.evidence_gain_count
    }

    pub(in crate::agent) fn no_evidence_result_streak(&self) -> usize {
        self.no_evidence_result_streak
    }

    pub(in crate::agent) fn no_evidence_tools_seen_count(&self) -> usize {
        self.no_evidence_tools_seen.len()
    }

    pub(in crate::agent) fn record_no_evidence_result(&mut self, tool_name: impl Into<String>) {
        self.no_evidence_result_streak = self.no_evidence_result_streak.saturating_add(1);
        self.no_evidence_tools_seen.insert(tool_name.into());
    }

    pub(in crate::agent) fn record_evidence_result(&mut self) {
        self.no_evidence_result_streak = 0;
        self.no_evidence_tools_seen.clear();
        self.increment_evidence_gain_count();
    }

    pub(in crate::agent) fn known_project_dir(&self) -> Option<&str> {
        self.known_project_dir.as_deref()
    }

    pub(in crate::agent) fn set_known_project_dir(&mut self, dir: impl Into<String>) {
        self.known_project_dir = Some(dir.into());
    }

    pub(in crate::agent) fn record_project_inspect_file_evidence(
        &mut self,
        dir: impl Into<String>,
    ) {
        self.dirs_with_project_inspect_file_evidence
            .insert(dir.into());
    }

    pub(in crate::agent) fn record_search_no_matches(&mut self, dir: impl Into<String>) {
        self.dirs_with_search_no_matches.insert(dir.into());
    }

    pub(in crate::agent) fn has_project_inspect_file_evidence(&self, dir: &str) -> bool {
        self.dirs_with_project_inspect_file_evidence.contains(dir)
    }

    pub(in crate::agent) fn has_search_no_matches(&self, dir: &str) -> bool {
        self.dirs_with_search_no_matches.contains(dir)
    }

    pub(in crate::agent) fn require_file_recheck_before_answer(&self) -> bool {
        self.require_file_recheck_before_answer
    }

    pub(in crate::agent) fn set_require_file_recheck_before_answer(&mut self, required: bool) {
        self.require_file_recheck_before_answer = required;
    }
}

#[cfg(test)]
mod tests {
    use super::EvidenceLedger;

    #[test]
    fn default_ledger_has_no_evidence_or_recheck_requirement() {
        let ledger = EvidenceLedger::default();

        assert_eq!(ledger.evidence_gain_count(), 0);
        assert_eq!(ledger.no_evidence_result_streak(), 0);
        assert_eq!(ledger.no_evidence_tools_seen_count(), 0);
        assert_eq!(ledger.known_project_dir(), None);
        assert!(!ledger.require_file_recheck_before_answer());
    }

    #[test]
    fn records_evidence_gain_and_resets_no_evidence_streak() {
        let mut ledger = EvidenceLedger::default();

        ledger.record_no_evidence_result("search_files");
        ledger.record_no_evidence_result("read_file");
        assert_eq!(ledger.no_evidence_result_streak(), 2);
        assert_eq!(ledger.no_evidence_tools_seen_count(), 2);

        ledger.record_evidence_result();
        assert_eq!(ledger.evidence_gain_count(), 1);
        assert_eq!(ledger.no_evidence_result_streak(), 0);
        assert_eq!(ledger.no_evidence_tools_seen_count(), 0);
    }

    #[test]
    fn stores_known_project_dir() {
        let mut ledger = EvidenceLedger::with_known_project_dir(Some("/repo".to_string()));
        assert_eq!(ledger.known_project_dir(), Some("/repo"));

        ledger.set_known_project_dir("/repo/sub");
        assert_eq!(ledger.known_project_dir(), Some("/repo/sub"));
    }

    #[test]
    fn tracks_contradictory_directory_evidence_sets() {
        let mut ledger = EvidenceLedger::default();

        ledger.record_project_inspect_file_evidence("/repo");
        ledger.record_search_no_matches("/repo");

        assert!(ledger.has_project_inspect_file_evidence("/repo"));
        assert!(ledger.has_search_no_matches("/repo"));
    }

    #[test]
    fn tracks_recheck_required_flag() {
        let mut ledger = EvidenceLedger::default();

        ledger.set_require_file_recheck_before_answer(true);
        assert!(ledger.require_file_recheck_before_answer());

        ledger.set_require_file_recheck_before_answer(false);
        assert!(!ledger.require_file_recheck_before_answer());
    }
}
