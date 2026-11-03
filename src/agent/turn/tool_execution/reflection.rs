#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::agent) struct PendingReflectionRecovery {
    pub signature: String,
    pub solution_ids: Vec<i64>,
    /// The first assistant turn after reflection where the tool retry can
    /// legitimately confirm this learning.
    pub verify_on_iteration: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::agent) struct ToolErrorEntry {
    pub iteration: usize,
    pub arguments_summary: String,
    pub error_text: String,
}
