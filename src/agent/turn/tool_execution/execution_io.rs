use crate::agent::*;

fn should_replay_durable_result(
    result: &crate::events::ToolResultData,
    tool_is_idempotent: bool,
) -> bool {
    // A retryable failure is the tool's own statement that the operation did
    // not take effect, so it may run again. A timeout carries that status too
    // but proves nothing about the effect: it runs again only when repeating
    // is safe. Every other durable outcome, including backgrounded and
    // blocked, is replayed to avoid duplicating an indeterminate or
    // already-completed effect.
    let may_run_again = result.receipt.as_ref().is_some_and(|receipt| {
        receipt.outcome_status == crate::traits::ToolOutcomeStatus::FailedRetryable
            && (tool_is_idempotent || !receipt.timed_out)
    });
    !may_run_again
}

fn durable_result_identity(result: &crate::events::ToolResultData) -> String {
    result
        .receipt
        .as_ref()
        .and_then(|receipt| receipt.result_provenance.result_id.clone())
        .unwrap_or_else(|| format!("receipt:{}", result.tool_call_id))
}

/// What the durable ledger says about an operation before it runs again.
pub(in crate::agent) struct DurableReplayPreflight {
    /// A completed earlier result to reuse instead of repeating the I/O.
    pub replayed_result: Option<Box<crate::events::ToolResultData>>,
    /// The earlier effect no longer holds: (operation id, result id, reason).
    pub replay_invalidation: Option<(String, String, String)>,
    /// The operation must not run (indeterminate earlier attempt, ledger error).
    pub block_reason: Option<String>,
}

/// Claim/replay preflight for an operation with an idempotency key. A
/// completed key reuses its durable receipt; a claimed key with no result
/// is indeterminate after a crash and is blocked for reconciliation.
pub(in crate::agent) async fn durable_replay_preflight(
    agent: &Agent,
    session_id: &str,
    tool_name: &str,
    effective_arguments: &str,
    idempotency_key: Option<&str>,
) -> DurableReplayPreflight {
    // A completed key reuses its durable receipt; a claimed key with no result
    // is indeterminate after a crash and is blocked for reconciliation.
    let tool_is_idempotent = agent
        .tools
        .iter()
        .find(|tool| tool.name() == tool_name && tool.is_available())
        .is_some_and(|tool| tool.capabilities().idempotent);
    let mut replayed_result: Option<Box<crate::events::ToolResultData>> = None;
    let mut replay_invalidation: Option<(String, String, String)> = None;
    let mut idempotency_block_reason: Option<String> = None;
    if let Some(idempotency_key) = idempotency_key {
        match agent
            .event_store
            .get_tool_result_by_idempotency_key(session_id, idempotency_key)
            .await
        {
            Ok(Some(result)) => {
                if should_replay_durable_result(&result, tool_is_idempotent) {
                    let receipt_succeeded = result.receipt.as_ref().is_some_and(|receipt| {
                        receipt.outcome_status == crate::traits::ToolOutcomeStatus::Succeeded
                    });
                    let replay_decision = if receipt_succeeded {
                        match agent
                            .tools
                            .iter()
                            .find(|tool| tool.name() == tool_name && tool.is_available())
                        {
                            Some(tool) => tool.durable_replay_decision(effective_arguments).await,
                            None => crate::traits::DurableReplayDecision::Replay,
                        }
                    } else {
                        crate::traits::DurableReplayDecision::Replay
                    };
                    match replay_decision {
                        crate::traits::DurableReplayDecision::Replay => {
                            replayed_result = Some(Box::new(result));
                        }
                        crate::traits::DurableReplayDecision::Reexecute { reason } => {
                            replay_invalidation = Some((
                                result.tool_call_id.clone(),
                                durable_result_identity(&result),
                                reason,
                            ));
                        }
                        crate::traits::DurableReplayDecision::Block { reason } => {
                            idempotency_block_reason = Some(reason);
                        }
                    }
                }
            }
            Ok(None) => match agent
                .event_store
                .has_unresolved_tool_call_for_idempotency_key(session_id, idempotency_key)
                .await
            {
                Ok(true) => {
                    idempotency_block_reason = Some(format!(
                        "A prior `{}` invocation with idempotency key `{}` has no durable result. Its side effect is indeterminate after interruption; inspect the target before issuing a different operation.",
                        tool_name, idempotency_key
                    ));
                }
                Ok(false) => {}
                Err(error) => {
                    idempotency_block_reason = Some(format!(
                        "Could not verify whether `{}` was already executed ({error}); refusing a potentially duplicate side effect.",
                        tool_name
                    ));
                }
            },
            Err(error) => {
                idempotency_block_reason = Some(format!(
                    "Could not read the durable receipt for `{}` ({error}); refusing a potentially duplicate side effect.",
                    tool_name
                ));
            }
        }
    }
    DurableReplayPreflight {
        replayed_result,
        replay_invalidation,
        block_reason: idempotency_block_reason,
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;

    fn result_with_status(
        status: crate::traits::ToolOutcomeStatus,
    ) -> crate::events::ToolResultData {
        let metadata = crate::traits::ToolCallMetadata {
            outcome_status: Some(status),
            semantics: crate::traits::ToolCallSemantics::mutation(),
            ..crate::traits::ToolCallMetadata::default()
        };
        crate::events::ToolResultData {
            message_id: None,
            tool_call_id: "call-1".to_string(),
            name: "probe".to_string(),
            result: "result".to_string(),
            success: status == crate::traits::ToolOutcomeStatus::Succeeded,
            duration_ms: 1,
            error: None,
            task_id: None,
            annotations: Vec::new(),
            turn_id: None,
            attachments: Vec::new(),
            receipt: Some(crate::events::ToolReceiptV1::from_metadata(
                &metadata,
                status,
                crate::events::ToolOutcomeEvidenceSource::ToolReported,
                Some("exec:e1:probe:abc".to_string()),
            )),
        }
    }

    #[test]
    fn durable_replay_reexecutes_only_failures_that_took_no_effect() {
        let retryable = result_with_status(crate::traits::ToolOutcomeStatus::FailedRetryable);
        assert!(!should_replay_durable_result(&retryable, true));
        assert!(
            !should_replay_durable_result(&retryable, false),
            "a retryable failure took no effect, so a write may run again"
        );

        let mut timed_out = result_with_status(crate::traits::ToolOutcomeStatus::FailedRetryable);
        timed_out.receipt.as_mut().unwrap().timed_out = true;
        assert!(
            should_replay_durable_result(&timed_out, false),
            "a timed-out write may have taken effect and must not run again"
        );
        assert!(!should_replay_durable_result(&timed_out, true));

        for status in [
            crate::traits::ToolOutcomeStatus::Succeeded,
            crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult,
            crate::traits::ToolOutcomeStatus::FailedPermanent,
            crate::traits::ToolOutcomeStatus::Blocked,
            crate::traits::ToolOutcomeStatus::Backgrounded,
        ] {
            assert!(should_replay_durable_result(
                &result_with_status(status),
                true
            ));
        }
    }
}
