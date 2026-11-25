//! The checks a tool call passes before it runs: a mandate worker's call
//! through its mandate's authority, and an idempotent operation's durable
//! replay preflight.

use aidaemon_tools::semantics::ToolCallSemantics;

use crate::engine::work_scope::WorkScope;
use crate::host::TurnHost;

// ── Mandate gate ─────────────────────────────────────────────────────────────

pub enum MandateGateDecision {
    NotApplicable,
    Allow {
        grant: Option<aidaemon_work::traits::MandateAuthorityGrant>,
    },
    Deny {
        reason: String,
    },
}

pub async fn mandate_gate_for_tool_call<H: TurnHost + ?Sized>(
    agent: &H,
    goal_id: &str,
    tool_call_id: &str,
    tool_name: &str,
    effective_arguments: &str,
    semantics: &ToolCallSemantics,
) -> anyhow::Result<MandateGateDecision> {
    let Some(mandate) = agent.mandates().get_mandate_for_goal(goal_id).await? else {
        return Ok(if agent.mandate_execution().is_some() {
            MandateGateDecision::Deny {
                reason: "mandate_record_missing".to_string(),
            }
        } else {
            MandateGateDecision::NotApplicable
        });
    };
    if !mandate.is_active() {
        return Ok(MandateGateDecision::Deny {
            reason: "mandate_inactive".to_string(),
        });
    }
    if WorkScope::require_live_mandate_execution(agent, goal_id)
        .await
        .is_err()
    {
        return Ok(MandateGateDecision::Deny {
            reason: "mandate_execution_fence_lost".to_string(),
        });
    }
    let Some(goal_run) = agent.work().get_current_goal_run(goal_id).await? else {
        return Ok(MandateGateDecision::Deny {
            reason: "mandate_run_missing".to_string(),
        });
    };
    if goal_run.trigger_type != "mandate" || goal_run.status != "running" {
        return Ok(MandateGateDecision::Deny {
            reason: "invalid_mandate_run".to_string(),
        });
    }

    let class = aidaemon_work::mandates::policy::classify_mandate_call(
        tool_name,
        effective_arguments,
        semantics,
    );
    let Some(fence) = agent.mandate_execution() else {
        return Ok(MandateGateDecision::Deny {
            reason: "mandate_role_fence_missing".to_string(),
        });
    };
    if agent.task_id() != Some(fence.worker_task_id.as_str()) {
        return Ok(MandateGateDecision::Deny {
            reason: "mandate_worker_identity_mismatch".to_string(),
        });
    }
    let is_task_lead = fence.worker_task_id == fence.root_task_id;
    if !aidaemon_work::mandates::policy::role_allows_mandate_call(class, tool_name, is_task_lead) {
        return Ok(MandateGateDecision::Deny {
            reason: if is_task_lead
                && class == aidaemon_work::mandates::policy::MandateCallClass::GovernedMutation
            {
                // This check deliberately precedes authorization and durable
                // reservation so a task-lead mistake cannot burn quota.
                "mandate_task_lead_mutation_forbidden"
            } else {
                "mandate_role_forbidden"
            }
            .to_string(),
        });
    }
    if class == aidaemon_work::mandates::policy::MandateCallClass::ProtocolObservation {
        return Ok(MandateGateDecision::Allow { grant: None });
    }
    if matches!(
        class,
        aidaemon_work::mandates::policy::MandateCallClass::RecordDecision
            | aidaemon_work::mandates::policy::MandateCallClass::RecordMeasurement
    ) {
        return Ok(MandateGateDecision::Allow { grant: None });
    }
    if class == aidaemon_work::mandates::policy::MandateCallClass::Observation {
        return Ok(
            match aidaemon_work::authority::authorize_mandate_observation(
                &mandate,
                tool_name,
                effective_arguments,
                semantics,
                &chrono::Utc::now(),
            ) {
                Ok(()) => MandateGateDecision::Allow { grant: None },
                Err(reason) => MandateGateDecision::Deny {
                    reason: reason.as_str().to_string(),
                },
            },
        );
    }

    let decision = agent
        .mandates()
        .get_mandate_decision_for_run(&goal_run.id)
        .await?;
    let Some(decision) = decision else {
        return Ok(MandateGateDecision::Deny {
            reason: "act_decision_required".to_string(),
        });
    };
    if decision.goal_run_id != goal_run.id
        || decision.mandate_id != mandate.id
        || decision.mandate_version != mandate.version
        || decision.outcome != aidaemon_work::mandates::MandateDecisionOutcome::Act
    {
        return Ok(MandateGateDecision::Deny {
            reason: "current_act_decision_required".to_string(),
        });
    }

    match class {
        aidaemon_work::mandates::policy::MandateCallClass::ActControl => {
            Ok(MandateGateDecision::Allow { grant: None })
        }
        aidaemon_work::mandates::policy::MandateCallClass::GovernedMutation => {
            match aidaemon_work::authority::authorize_mandate_action(
                &mandate,
                &decision,
                tool_name,
                effective_arguments,
                semantics,
                &chrono::Utc::now(),
            ) {
                aidaemon_work::authority::MandateAuthorityDecision::Allow(mut grant) => {
                    if !grant.counts_toward_cycle_budget {
                        return Ok(MandateGateDecision::Deny {
                            reason: "mutation_grant_not_metered".to_string(),
                        });
                    }
                    grant.tool_call_id = Some(tool_call_id.to_string());
                    // This is an uncommitted capability candidate. Durable
                    // quota reservation happens only at the final common
                    // dispatcher, after every local no-I/O guard has passed.
                    Ok(MandateGateDecision::Allow { grant: Some(grant) })
                }
                aidaemon_work::authority::MandateAuthorityDecision::Deny(reason) => {
                    Ok(MandateGateDecision::Deny {
                        reason: reason.as_str().to_string(),
                    })
                }
            }
        }
        aidaemon_work::mandates::policy::MandateCallClass::Deny => Ok(MandateGateDecision::Deny {
            reason: "unsupported_mandate_call".to_string(),
        }),
        aidaemon_work::mandates::policy::MandateCallClass::ProtocolObservation
        | aidaemon_work::mandates::policy::MandateCallClass::Observation
        | aidaemon_work::mandates::policy::MandateCallClass::RecordDecision
        | aidaemon_work::mandates::policy::MandateCallClass::RecordMeasurement => unreachable!(),
    }
}

fn should_replay_durable_result(
    result: &aidaemon_events::ToolResultData,
    tool_is_idempotent: bool,
) -> bool {
    // A failure of a tool that is safe to repeat may always run again. On any
    // other tool, a retryable failure is the tool's own statement that the
    // operation did not take effect, except a timeout, which carries that
    // status but proves nothing about the effect. Every other durable outcome,
    // including backgrounded and blocked, is replayed to avoid duplicating an
    // indeterminate or already-completed effect.
    let may_run_again = result.receipt.as_ref().is_some_and(|receipt| {
        use aidaemon_tools::semantics::ToolOutcomeStatus::{FailedPermanent, FailedRetryable};
        match receipt.outcome_status {
            FailedPermanent => tool_is_idempotent,
            FailedRetryable => tool_is_idempotent || !receipt.timed_out,
            _ => false,
        }
    });
    !may_run_again
}

fn durable_result_identity(result: &aidaemon_events::ToolResultData) -> String {
    result
        .receipt
        .as_ref()
        .and_then(|receipt| receipt.result_provenance.result_id.clone())
        .unwrap_or_else(|| format!("receipt:{}", result.tool_call_id))
}

/// What the durable ledger says about an operation before it runs again.
pub struct DurableReplayPreflight {
    /// A completed earlier result to reuse instead of repeating the I/O.
    pub replayed_result: Option<Box<aidaemon_events::ToolResultData>>,
    /// The earlier effect no longer holds: (operation id, result id, reason).
    pub replay_invalidation: Option<(String, String, String)>,
    /// The operation must not run (indeterminate earlier attempt, ledger error).
    pub block_reason: Option<String>,
}

/// Claim/replay preflight for an operation with an idempotency key. A
/// completed key reuses its durable receipt; a claimed key with no result
/// is indeterminate after a crash and is blocked for reconciliation.
pub async fn durable_replay_preflight<H: TurnHost + ?Sized>(
    agent: &H,
    session_id: &str,
    tool_name: &str,
    effective_arguments: &str,
    idempotency_key: Option<&str>,
) -> DurableReplayPreflight {
    // A completed key reuses its durable receipt; a claimed key with no result
    // is indeterminate after a crash and is blocked for reconciliation.
    let tool_is_idempotent = agent
        .tools()
        .iter()
        .find(|tool| tool.name() == tool_name && tool.is_available())
        .is_some_and(|tool| tool.capabilities().idempotent);
    let mut replayed_result: Option<Box<aidaemon_events::ToolResultData>> = None;
    let mut replay_invalidation: Option<(String, String, String)> = None;
    let mut idempotency_block_reason: Option<String> = None;
    if let Some(idempotency_key) = idempotency_key {
        match agent
            .events()
            .get_tool_result_by_idempotency_key(session_id, idempotency_key)
            .await
        {
            Ok(Some(result)) => {
                if should_replay_durable_result(&result, tool_is_idempotent) {
                    let receipt_succeeded = result.receipt.as_ref().is_some_and(|receipt| {
                        receipt.outcome_status
                            == aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded
                    });
                    let replay_decision = if receipt_succeeded {
                        match agent
                            .tools()
                            .iter()
                            .find(|tool| tool.name() == tool_name && tool.is_available())
                        {
                            Some(tool) => tool.durable_replay_decision(effective_arguments).await,
                            None => aidaemon_tools::semantics::DurableReplayDecision::Replay,
                        }
                    } else {
                        aidaemon_tools::semantics::DurableReplayDecision::Replay
                    };
                    match replay_decision {
                        aidaemon_tools::semantics::DurableReplayDecision::Replay => {
                            replayed_result = Some(Box::new(result));
                        }
                        aidaemon_tools::semantics::DurableReplayDecision::Reexecute { reason } => {
                            replay_invalidation = Some((
                                result.tool_call_id.clone(),
                                durable_result_identity(&result),
                                reason,
                            ));
                        }
                        aidaemon_tools::semantics::DurableReplayDecision::Block { reason } => {
                            idempotency_block_reason = Some(reason);
                        }
                    }
                }
            }
            Ok(None) => match agent
                .events()
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
        status: aidaemon_tools::semantics::ToolOutcomeStatus,
    ) -> aidaemon_events::ToolResultData {
        let metadata = aidaemon_tools::semantics::ToolCallMetadata {
            outcome_status: Some(status),
            semantics: aidaemon_tools::semantics::ToolCallSemantics::mutation(),
            ..aidaemon_tools::semantics::ToolCallMetadata::default()
        };
        aidaemon_events::ToolResultData {
            message_id: None,
            tool_call_id: "call-1".to_string(),
            name: "probe".to_string(),
            result: "result".to_string(),
            success: status == aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded,
            duration_ms: 1,
            error: None,
            task_id: None,
            annotations: Vec::new(),
            turn_id: None,
            attachments: Vec::new(),
            receipt: Some(aidaemon_events::ToolReceiptV1::from_metadata(
                &metadata,
                status,
                aidaemon_events::ToolOutcomeEvidenceSource::ToolReported,
                Some("exec:e1:probe:abc".to_string()),
            )),
        }
    }

    #[test]
    fn durable_replay_reexecutes_only_failures_that_took_no_effect() {
        let retryable =
            result_with_status(aidaemon_tools::semantics::ToolOutcomeStatus::FailedRetryable);
        assert!(!should_replay_durable_result(&retryable, true));
        assert!(
            !should_replay_durable_result(&retryable, false),
            "a retryable failure took no effect, so a write may run again"
        );

        let mut timed_out =
            result_with_status(aidaemon_tools::semantics::ToolOutcomeStatus::FailedRetryable);
        timed_out.receipt.as_mut().unwrap().timed_out = true;
        assert!(
            should_replay_durable_result(&timed_out, false),
            "a timed-out write may have taken effect and must not run again"
        );
        assert!(!should_replay_durable_result(&timed_out, true));

        let permanent =
            result_with_status(aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent);
        assert!(
            !should_replay_durable_result(&permanent, true),
            "a tool that is safe to repeat may always try again"
        );
        assert!(
            should_replay_durable_result(&permanent, false),
            "a generic failure leaves a write's effect unknown"
        );

        for status in [
            aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded,
            aidaemon_tools::semantics::ToolOutcomeStatus::CompletedWithNegativeResult,
            aidaemon_tools::semantics::ToolOutcomeStatus::Blocked,
            aidaemon_tools::semantics::ToolOutcomeStatus::Backgrounded,
        ] {
            assert!(should_replay_durable_result(
                &result_with_status(status),
                true
            ));
        }
    }
}
