use crate::agent::loop_utils::ToolFailureClass;
use crate::agent::tool_execution_phase::ToolErrorEntry;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default)]
pub(in crate::agent) struct FailureLedger {
    tool_failure_count: HashMap<String, usize>,
    tool_failure_signatures: HashMap<(String, String), usize>,
    tool_failure_patterns: HashMap<(String, String), usize>,
    last_tool_failure: Option<(String, String)>,
    tool_error_history: HashMap<(String, String), Vec<ToolErrorEntry>>,
    tool_transient_failure_count: HashMap<String, usize>,
    tool_cooldown_until_iteration: HashMap<String, usize>,
    unknown_tools: HashSet<String>,
    last_failure_class: Option<ToolFailureClass>,
}

pub(in crate::agent) struct StoppingFailureState<'a> {
    pub tool_failure_count: &'a HashMap<String, usize>,
    pub last_failure_class: Option<ToolFailureClass>,
}

pub(in crate::agent) struct ToolExecutionFailureState<'a> {
    pub tool_failure_count: &'a mut HashMap<String, usize>,
    pub tool_failure_signatures: &'a mut HashMap<(String, String), usize>,
    pub tool_transient_failure_count: &'a mut HashMap<String, usize>,
    pub tool_cooldown_until_iteration: &'a mut HashMap<String, usize>,
    pub tool_error_history: &'a mut HashMap<(String, String), Vec<ToolErrorEntry>>,
    pub tool_failure_patterns: &'a mut HashMap<(String, String), usize>,
    pub last_tool_failure: &'a mut Option<(String, String)>,
    pub unknown_tools: &'a mut HashSet<String>,
    pub last_failure_class: &'a mut Option<ToolFailureClass>,
}

impl FailureLedger {
    /// Clear retry gates that belong to the abandoned approach. Durable error
    /// history and the learning ledger still describe what failed, while a
    /// genuinely new approach gets a fresh chance to use the same tool.
    pub(in crate::agent) fn reset_for_pivot(&mut self) {
        self.tool_failure_count.clear();
        self.tool_failure_signatures.clear();
        self.tool_failure_patterns.clear();
        self.last_tool_failure = None;
        self.tool_transient_failure_count.clear();
        self.tool_cooldown_until_iteration.clear();
        self.last_failure_class = None;
    }

    pub(in crate::agent) fn for_stopping_phase(&self) -> StoppingFailureState<'_> {
        StoppingFailureState {
            tool_failure_count: &self.tool_failure_count,
            last_failure_class: self.last_failure_class,
        }
    }

    pub(in crate::agent) fn for_tool_execution_phase(&mut self) -> ToolExecutionFailureState<'_> {
        ToolExecutionFailureState {
            tool_failure_count: &mut self.tool_failure_count,
            tool_failure_signatures: &mut self.tool_failure_signatures,
            tool_transient_failure_count: &mut self.tool_transient_failure_count,
            tool_cooldown_until_iteration: &mut self.tool_cooldown_until_iteration,
            tool_error_history: &mut self.tool_error_history,
            tool_failure_patterns: &mut self.tool_failure_patterns,
            last_tool_failure: &mut self.last_tool_failure,
            unknown_tools: &mut self.unknown_tools,
            last_failure_class: &mut self.last_failure_class,
        }
    }

    pub(in crate::agent) fn failure_count(&self, tool_name: &str) -> usize {
        self.tool_failure_count.get(tool_name).copied().unwrap_or(0)
    }

    pub(in crate::agent) fn signature_count(&self, tool_name: &str, signature: &str) -> usize {
        self.tool_failure_signatures
            .get(&(tool_name.to_string(), signature.to_string()))
            .copied()
            .unwrap_or(0)
    }

    pub(in crate::agent) fn pattern_count(&self, tool_name: &str, pattern: &str) -> usize {
        self.tool_failure_patterns
            .get(&(tool_name.to_string(), pattern.to_string()))
            .copied()
            .unwrap_or(0)
    }

    pub(in crate::agent) fn transient_failure_count(&self, tool_name: &str) -> usize {
        self.tool_transient_failure_count
            .get(tool_name)
            .copied()
            .unwrap_or(0)
    }

    pub(in crate::agent) fn cooldown_until(&self, tool_name: &str) -> Option<usize> {
        self.tool_cooldown_until_iteration.get(tool_name).copied()
    }

    pub(in crate::agent) fn last_tool_failure(&self) -> Option<(&str, &str)> {
        self.last_tool_failure
            .as_ref()
            .map(|(tool, signature)| (tool.as_str(), signature.as_str()))
    }

    pub(in crate::agent) fn is_unknown_tool(&self, tool_name: &str) -> bool {
        self.unknown_tools.contains(tool_name)
    }

    pub(in crate::agent) fn error_history(
        &self,
        tool_name: &str,
        signature: &str,
    ) -> Option<&[ToolErrorEntry]> {
        self.tool_error_history
            .get(&(tool_name.to_string(), signature.to_string()))
            .map(Vec::as_slice)
    }

    pub(in crate::agent) fn increment_failure(&mut self, tool_name: &str) {
        *self
            .tool_failure_count
            .entry(tool_name.to_string())
            .or_insert(0) += 1;
    }

    pub(in crate::agent) fn increment_signature(&mut self, tool_name: &str, signature: &str) {
        *self
            .tool_failure_signatures
            .entry((tool_name.to_string(), signature.to_string()))
            .or_insert(0) += 1;
    }

    pub(in crate::agent) fn increment_pattern(&mut self, tool_name: &str, pattern: &str) {
        *self
            .tool_failure_patterns
            .entry((tool_name.to_string(), pattern.to_string()))
            .or_insert(0) += 1;
    }

    pub(in crate::agent) fn set_last_tool_failure(&mut self, tool_name: &str, signature: &str) {
        self.last_tool_failure = Some((tool_name.to_string(), signature.to_string()));
    }

    pub(in crate::agent) fn increment_transient_failure(&mut self, tool_name: &str) {
        *self
            .tool_transient_failure_count
            .entry(tool_name.to_string())
            .or_insert(0) += 1;
    }

    pub(in crate::agent) fn set_cooldown_until(&mut self, tool_name: &str, iteration: usize) {
        self.tool_cooldown_until_iteration
            .insert(tool_name.to_string(), iteration);
    }

    pub(in crate::agent) fn record_unknown_tool(&mut self, tool_name: &str) {
        self.unknown_tools.insert(tool_name.to_string());
    }

    pub(in crate::agent) fn push_error_history(
        &mut self,
        tool_name: &str,
        signature: &str,
        entry: ToolErrorEntry,
    ) {
        self.tool_error_history
            .entry((tool_name.to_string(), signature.to_string()))
            .or_default()
            .push(entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ledger_has_no_failures_or_unknown_tools() {
        let ledger = FailureLedger::default();

        assert_eq!(ledger.failure_count("terminal"), 0);
        assert_eq!(ledger.signature_count("terminal", "denied"), 0);
        assert_eq!(ledger.pattern_count("terminal", "permission"), 0);
        assert_eq!(ledger.transient_failure_count("terminal"), 0);
        assert_eq!(ledger.cooldown_until("terminal"), None);
        assert_eq!(ledger.last_tool_failure(), None);
        assert!(!ledger.is_unknown_tool("missing_tool"));
        assert!(ledger.error_history("terminal", "denied").is_none());
    }

    #[test]
    fn records_failure_counts_signatures_patterns_and_last_failure() {
        let mut ledger = FailureLedger::default();

        ledger.increment_failure("terminal");
        ledger.increment_failure("terminal");
        ledger.increment_signature("terminal", "permission denied");
        ledger.increment_pattern("terminal", "permission");
        ledger.set_last_tool_failure("terminal", "permission denied");

        assert_eq!(ledger.failure_count("terminal"), 2);
        assert_eq!(ledger.signature_count("terminal", "permission denied"), 1);
        assert_eq!(ledger.pattern_count("terminal", "permission"), 1);
        assert_eq!(
            ledger.last_tool_failure(),
            Some(("terminal", "permission denied"))
        );
    }

    #[test]
    fn records_transient_cooldown_and_unknown_tools() {
        let mut ledger = FailureLedger::default();

        ledger.increment_transient_failure("web_fetch");
        ledger.set_cooldown_until("web_fetch", 7);
        ledger.record_unknown_tool("missing_tool");

        assert_eq!(ledger.transient_failure_count("web_fetch"), 1);
        assert_eq!(ledger.cooldown_until("web_fetch"), Some(7));
        assert!(ledger.is_unknown_tool("missing_tool"));
    }

    #[test]
    fn pivot_clears_retry_gates_but_preserves_unknown_tool_facts() {
        let mut ledger = FailureLedger::default();
        ledger.increment_failure("web_fetch");
        ledger.increment_signature("web_fetch", "timeout");
        ledger.increment_transient_failure("web_fetch");
        ledger.set_cooldown_until("web_fetch", 7);
        ledger.record_unknown_tool("missing_tool");
        ledger.last_failure_class = Some(ToolFailureClass::Transient);

        ledger.reset_for_pivot();

        assert_eq!(ledger.failure_count("web_fetch"), 0);
        assert_eq!(ledger.signature_count("web_fetch", "timeout"), 0);
        assert_eq!(ledger.transient_failure_count("web_fetch"), 0);
        assert_eq!(ledger.cooldown_until("web_fetch"), None);
        assert_eq!(ledger.last_failure_class, None);
        assert!(ledger.is_unknown_tool("missing_tool"));
    }

    #[test]
    fn records_error_history_entries_by_tool_and_signature() {
        let mut ledger = FailureLedger::default();
        let entry = ToolErrorEntry {
            iteration: 3,
            arguments_summary: "{\"cmd\":\"npm test\"}".to_string(),
            error_text: "command failed".to_string(),
        };

        ledger.push_error_history("terminal", "command failed", entry.clone());

        assert_eq!(
            ledger.error_history("terminal", "command failed"),
            Some(&[entry][..])
        );
    }
}
