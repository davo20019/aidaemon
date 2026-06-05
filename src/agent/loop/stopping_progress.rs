use super::{CompletionProgress, TurnContext};

pub(super) fn turn_contract_is_text_only(turn_context: &TurnContext) -> bool {
    !turn_context.completion_contract.expects_mutation
        && !turn_context.completion_contract.requires_observation
}

pub(super) fn has_task_relevant_progress(
    turn_context: &TurnContext,
    completion_progress: &CompletionProgress,
) -> bool {
    (turn_context.completion_contract.expects_mutation && completion_progress.mutation_count > 0)
        || completion_progress.observation_count > 0
        || completion_progress.verification_count > 0
}

pub(super) fn has_any_concrete_execution(
    turn_context: &TurnContext,
    completion_progress: &CompletionProgress,
    recoverable_tool_snapshot_present: bool,
    total_successful_tool_calls: usize,
) -> bool {
    has_task_relevant_progress(turn_context, completion_progress)
        || recoverable_tool_snapshot_present
        // Any successfully completed tool call counts as concrete work,
        // even if its semantics did not classify as observation/mutation
        // (common for MCP tools with Unknown/Administrative effects).
        // This prevents the harsh "abandon" path when tools DID execute.
        || total_successful_tool_calls > 0
}

pub(super) fn only_final_response_remains(
    turn_context: &TurnContext,
    completion_progress: &CompletionProgress,
    recoverable_tool_snapshot_present: bool,
    total_successful_tool_calls: usize,
) -> bool {
    has_any_concrete_execution(
        turn_context,
        completion_progress,
        recoverable_tool_snapshot_present,
        total_successful_tool_calls,
    ) && !completion_progress.verification_pending
}
