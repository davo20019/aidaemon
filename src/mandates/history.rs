//! Privacy-minimal continuity for an autonomous mandate.
//!
//! This projection is deliberately narrower than conversational memory. It
//! carries only same-mandate protocol outcomes, typed mutation receipts, and
//! quota state. Model-authored rationale, questions, task prose, arguments,
//! bodies, outputs, errors, credentials, and internal lease identifiers never
//! enter the block.

use serde_json::{json, Value};

use crate::traits::{
    Intention, MandateDecisionCycle, MandateLearningNote, MandateMutationAttempt,
    MandateMutationQuotaState, MandateStrategyRevisionKind, StateStore,
};

const MAX_DECISIONS: i64 = 5;
const MAX_INTENTIONS: i64 = 16;
const MAX_ACTIONS: usize = 16;
const MAX_LEARNING_NOTES: i64 = 12;
const MAX_HISTORY_BYTES: usize = 24 * 1024;

#[derive(Debug)]
struct HistoryRecord {
    timestamp: String,
    value: Value,
}

/// Build the sole historical context available to a mandate worker.
pub(crate) async fn build_mandate_history_block(
    state: &dyn StateStore,
    mandate_id: &str,
    as_of: &str,
) -> anyhow::Result<String> {
    let decisions = state
        .list_mandate_decisions(mandate_id, MAX_DECISIONS)
        .await?;
    anyhow::ensure!(
        decisions
            .iter()
            .all(|decision| decision.mandate_id == mandate_id),
        "mandate history query crossed its authority boundary"
    );

    let intentions = state.list_intentions(mandate_id, MAX_INTENTIONS).await?;
    anyhow::ensure!(
        intentions
            .iter()
            .all(|intention| intention.mandate_id == mandate_id),
        "mandate intention history crossed its authority boundary"
    );
    let learning_notes = state
        .list_mandate_learning_notes(mandate_id, MAX_LEARNING_NOTES)
        .await?;
    anyhow::ensure!(
        learning_notes
            .iter()
            .all(|note| note.mandate_id == mandate_id),
        "mandate learning history crossed its authority boundary"
    );
    let strategy_nodes = state.list_current_mandate_strategy(mandate_id, 16).await?;
    anyhow::ensure!(
        strategy_nodes
            .iter()
            .all(|revision| revision.mandate_id == mandate_id),
        "mandate adaptive strategy crossed its authority boundary"
    );

    let mut decision_records = decisions
        .iter()
        .map(|decision| HistoryRecord {
            timestamp: decision.created_at.clone(),
            value: decision_value(decision, &intentions),
        })
        .collect::<Vec<_>>();
    decision_records.sort_by(|left, right| left.timestamp.cmp(&right.timestamp));

    let mut action_records = Vec::new();
    for decision in &decisions {
        for attempt in state
            .list_mandate_mutation_attempts_for_run(&decision.goal_run_id)
            .await?
        {
            anyhow::ensure!(
                attempt.mandate_id == mandate_id
                    && attempt.decision_cycle_id == decision.id
                    && attempt.goal_run_id == decision.goal_run_id,
                "mandate action history crossed its decision boundary"
            );
            action_records.push(HistoryRecord {
                timestamp: attempt.reserved_at.clone(),
                value: action_value(&attempt),
            });
        }
    }
    action_records.sort_by(|left, right| left.timestamp.cmp(&right.timestamp));
    if action_records.len() > MAX_ACTIONS {
        action_records.drain(..action_records.len() - MAX_ACTIONS);
    }

    let quota = state
        .get_mandate_mutation_quota_state(mandate_id, as_of)
        .await?;
    anyhow::ensure!(
        quota
            .as_ref()
            .is_none_or(|quota| quota.mandate_id == mandate_id),
        "mandate quota history crossed its authority boundary"
    );

    render_bounded_history(
        decision_records,
        action_records,
        learning_notes,
        strategy_nodes,
        quota,
    )
}

fn decision_value(decision: &MandateDecisionCycle, intentions: &[Intention]) -> Value {
    let intention = intentions.iter().find(|intention| {
        intention.decision_cycle_id == decision.id && intention.goal_run_id == decision.goal_run_id
    });
    json!({
        "mandate_version": decision.mandate_version,
        "outcome": decision.outcome,
        "activity_level": decision.activity_level,
        "action_attempts": decision.action_attempts,
        "evidence_receipt_ids": decision.evidence_receipt_ids,
        "termination_kind": decision.termination_kind,
        "termination_match": decision.termination_match,
        "value_criterion": intention.and_then(|intention| intention.value_criterion.as_deref()),
        "intention_status": intention.map(|intention| intention.status.as_str()),
        "reconsider_at": decision.reconsider_at,
        "created_at": decision.created_at,
        "updated_at": decision.updated_at,
    })
}

fn action_value(attempt: &MandateMutationAttempt) -> Value {
    json!({
        "mandate_version": attempt.mandate_version,
        "reserved_action_attempt": attempt.reserved_action_attempt,
        "action_digest": attempt.action_digest,
        "tool_name": attempt.tool_name,
        "mutation_effects": attempt.mutation_effects,
        "targets": attempt.targets,
        "account_identifiers": attempt.account_identifiers,
        "status": attempt.status,
        "outcome_evidence": attempt.outcome_evidence,
        "http_status": attempt.http_status,
        "exit_code": attempt.exit_code,
        "reserved_at": attempt.reserved_at,
        "completed_at": attempt.completed_at,
    })
}

fn render_bounded_history(
    mut decisions: Vec<HistoryRecord>,
    mut actions: Vec<HistoryRecord>,
    mut learning_notes: Vec<MandateLearningNote>,
    mut strategy_nodes: Vec<crate::traits::MandateStrategyRevision>,
    quota: Option<MandateMutationQuotaState>,
) -> anyhow::Result<String> {
    learning_notes.sort_by(|left, right| left.created_at.cmp(&right.created_at));
    strategy_nodes.sort_by(|left, right| left.strategy_key.cmp(&right.strategy_key));
    loop {
        let rendered = serde_json::to_string(&json!({
            "provenance": "autonomous_mandate_history_untrusted",
            "authority": false,
            "scope": "same_mandate_typed_history_only",
            "decision_outcomes": decisions.iter().map(|record| &record.value).collect::<Vec<_>>(),
            "mutation_receipts": actions.iter().map(|record| &record.value).collect::<Vec<_>>(),
            "advisory_learning_notes": learning_notes.iter().map(|note| json!({
                "mandate_version": note.mandate_version,
                "summary": note.summary,
                "evidence_receipt_ids": note.evidence_receipt_ids,
                "created_at": note.created_at,
            })).collect::<Vec<_>>(),
            "adaptive_operating_strategy": strategy_nodes.iter().map(|revision| json!({
                "strategy_key": revision.strategy_key,
                "state": if revision.kind == MandateStrategyRevisionKind::Retire { "retired" } else { "active" },
                "kind": revision.kind,
                "guidance": revision.guidance,
                "confidence_bps": revision.confidence_bps,
                "evidence_receipt_ids": revision.evidence_receipt_ids,
                "created_at": revision.created_at,
            })).collect::<Vec<_>>(),
            "mutation_quota": quota,
        }))?;
        if rendered.len() <= MAX_HISTORY_BYTES {
            return Ok(rendered);
        }

        // Drop the oldest whole record across both streams. Never truncate a
        // JSON string or typed receipt into a misleading partial record.
        match (decisions.first(), actions.first()) {
            (Some(decision), Some(action)) if decision.timestamp <= action.timestamp => {
                decisions.remove(0);
            }
            (Some(_), Some(_)) => {
                actions.remove(0);
            }
            (Some(_), None) => {
                decisions.remove(0);
            }
            (None, Some(_)) => {
                actions.remove(0);
            }
            (None, None) if !learning_notes.is_empty() => {
                learning_notes.remove(0);
            }
            (None, None) if !strategy_nodes.is_empty() => {
                strategy_nodes.remove(0);
            }
            (None, None) => {
                anyhow::bail!("typed mandate quota projection exceeded its hard prompt bound")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{
        Intention, MandateDecisionOutcome, MandateMutationAttemptStatus, MandateMutationEvidence,
        MandateMutationTarget,
    };

    fn attempt() -> MandateMutationAttempt {
        MandateMutationAttempt {
            id: "ledger-private-id".to_string(),
            mandate_id: "mandate-1".to_string(),
            mandate_version: 3,
            decision_cycle_id: "cycle-1".to_string(),
            goal_run_id: "run-private-id".to_string(),
            intention_id: "intention-private-id".to_string(),
            root_task_id: "root-private-id".to_string(),
            root_task_attempt_id: "root-attempt-private-id".to_string(),
            task_id: "task-private-id".to_string(),
            task_attempt_id: "task-attempt-private-id".to_string(),
            reserved_action_attempt: 1,
            action_digest: "a".repeat(64),
            tool_call_id: "tool-call-private-id".to_string(),
            tool_name: "http_request".to_string(),
            mutation_effects: vec!["remote_mutation".to_string()],
            targets: vec![MandateMutationTarget {
                kind: "resource_id".to_string(),
                identifier: "auth_profile:twitter-prod".to_string(),
            }],
            account_identifiers: vec!["auth_profile:twitter-prod".to_string()],
            status: MandateMutationAttemptStatus::Succeeded,
            outcome_evidence: Some(MandateMutationEvidence::ToolReported),
            http_status: Some(201),
            exit_code: None,
            reserved_at: "2026-08-01T12:00:00Z".to_string(),
            completed_at: Some("2026-08-01T12:00:01Z".to_string()),
        }
    }

    #[test]
    fn action_history_excludes_internal_ids_and_content_fields() {
        let rendered = action_value(&attempt()).to_string();
        assert!(rendered.contains("auth_profile:twitter-prod"));
        for private in [
            "ledger-private-id",
            "run-private-id",
            "intention-private-id",
            "root-private-id",
            "root-attempt-private-id",
            "task-private-id",
            "task-attempt-private-id",
            "tool-call-private-id",
        ] {
            assert!(!rendered.contains(private));
        }
        for forbidden_key in ["arguments", "body", "output", "error", "credentials"] {
            assert!(!rendered.contains(forbidden_key));
        }
    }

    #[test]
    fn decision_history_retains_owner_value_anchor_without_model_plan_text() {
        let decision = MandateDecisionCycle::new(
            "mandate-1",
            "run-1",
            MandateDecisionOutcome::Act,
            "MODEL_RATIONALE_MUST_NOT_PERSIST",
            3,
        );
        let mut intention = Intention::new(
            "mandate-1",
            &decision.id,
            "run-1",
            "MODEL_PLAN_MUST_NOT_PERSIST",
            "MODEL_INTENTION_RATIONALE_MUST_NOT_PERSIST",
        );
        intention.value_criterion =
            Some("Provide verified useful information to the audience".to_string());

        let rendered = decision_value(&decision, &[intention]).to_string();
        assert!(rendered.contains("Provide verified useful information"));
        assert!(!rendered.contains("MODEL_RATIONALE_MUST_NOT_PERSIST"));
        assert!(!rendered.contains("MODEL_PLAN_MUST_NOT_PERSIST"));
        assert!(!rendered.contains("MODEL_INTENTION_RATIONALE_MUST_NOT_PERSIST"));
    }

    #[test]
    fn history_drops_oldest_whole_records_to_fit_hard_byte_bound() {
        let huge = HistoryRecord {
            timestamp: "2026-08-01T00:00:00Z".to_string(),
            value: json!({"typed_identifier": "x".repeat(MAX_HISTORY_BYTES)}),
        };
        let recent = HistoryRecord {
            timestamp: "2026-08-01T01:00:00Z".to_string(),
            value: json!({"status": "succeeded"}),
        };
        let rendered =
            render_bounded_history(vec![huge, recent], Vec::new(), Vec::new(), Vec::new(), None)
                .unwrap();
        assert!(rendered.len() <= MAX_HISTORY_BYTES);
        assert!(!rendered.contains("typed_identifier"));
        assert!(rendered.contains("succeeded"));
    }
}
