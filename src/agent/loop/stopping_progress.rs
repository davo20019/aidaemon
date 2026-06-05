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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CompletionContract;

    #[test]
    fn recoverable_tool_snapshot_counts_as_concrete_progress() {
        let turn_context = TurnContext::default();
        let completion_progress = CompletionProgress::default();

        assert!(has_any_concrete_execution(
            &turn_context,
            &completion_progress,
            true,
            0,
        ));
        assert!(only_final_response_remains(
            &turn_context,
            &completion_progress,
            true,
            0,
        ));
    }

    #[test]
    fn successful_tool_calls_count_as_concrete_work() {
        let turn_context = TurnContext::default();
        let completion_progress = CompletionProgress::default();

        // No snapshot, no progress, but successful tool calls → concrete work
        assert!(has_any_concrete_execution(
            &turn_context,
            &completion_progress,
            false,
            1,
        ));
        // Zero successful tool calls and no other progress → not concrete
        assert!(!has_any_concrete_execution(
            &turn_context,
            &completion_progress,
            false,
            0,
        ));
    }

    #[test]
    fn verification_pending_prevents_final_response_closeout_even_with_snapshot() {
        let turn_context = TurnContext {
            completion_contract: CompletionContract {
                requires_observation: true,
                ..CompletionContract::default()
            },
            ..TurnContext::default()
        };
        let completion_progress = CompletionProgress {
            verification_pending: true,
            ..CompletionProgress::default()
        };

        assert!(has_any_concrete_execution(
            &turn_context,
            &completion_progress,
            true,
            0,
        ));
        assert!(!only_final_response_remains(
            &turn_context,
            &completion_progress,
            true,
            0,
        ));
    }
}
