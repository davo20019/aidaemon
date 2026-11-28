//! Recording a review's decision cycle and intentions, and reading them back.

use super::*;

pub(super) fn clamped_next_review_at(
    mandate: &Mandate,
    requested: Option<&str>,
    now: chrono::DateTime<chrono::Utc>,
) -> anyhow::Result<String> {
    let candidate = match requested {
        Some(value) => chrono::DateTime::parse_from_rfc3339(value)
            .map_err(|error| anyhow::anyhow!("invalid reconsider_at: {error}"))?
            .with_timezone(&chrono::Utc),
        None => now + chrono::Duration::seconds(mandate.default_review_secs),
    };
    let earliest = now + chrono::Duration::seconds(mandate.min_review_secs);
    let latest = now + chrono::Duration::seconds(mandate.max_review_secs);
    let candidate = candidate.clamp(earliest, latest);
    Ok(mandate
        .expires_at
        .as_deref()
        .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
        .map(|expires| expires.with_timezone(&chrono::Utc))
        .filter(|expires| *expires < candidate)
        .unwrap_or(candidate)
        .to_rfc3339())
}

pub(super) async fn record_mandate_decision<S: WorkDatabase>(
    store: &S,
    decision: &MandateDecisionCycle,
    intention: Option<&Intention>,
    task_attempt_id: Option<&str>,
) -> anyhow::Result<()> {
    store
        .record_mandate_decision_with_updates(decision, intention, None, task_attempt_id)
        .await
}

pub(super) async fn record_mandate_decision_with_updates<S: WorkDatabase>(
    store: &S,
    decision: &MandateDecisionCycle,
    intention: Option<&Intention>,
    operating_updates: Option<&MandateOperatingUpdates>,
    task_attempt_id: Option<&str>,
) -> anyhow::Result<()> {
    decision
        .validate_content_bounds()
        .map_err(anyhow::Error::msg)?;
    if let Some(updates) = operating_updates
        && let Some(note) = updates.learning_note.as_ref()
    {
        note.validate().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            note.mandate_id == decision.mandate_id
                && note.mandate_version == decision.mandate_version
                && note.learned_in_decision_cycle_id == decision.id,
            "learning note does not belong to this decision"
        );
        anyhow::ensure!(
            note.evidence_receipt_ids
                .iter()
                .all(|id| decision.evidence_receipt_ids.contains(id)),
            "learning evidence must be a subset of current decision evidence"
        );
    }
    if let Some(updates) = operating_updates {
        validate_action_outcomes(decision, &updates.action_outcomes)?;
    }
    anyhow::ensure!(
        decision.action_attempts == 0,
        "a new decision cannot pre-spend actions"
    );
    anyhow::ensure!(
        (decision.outcome == MandateDecisionOutcome::Act) == intention.is_some(),
        "ACT requires one intention and non-ACT outcomes must not create one"
    );
    // Reject malformed or policy-unrelated value judgments before taking
    // the SQLite writer lock. The same checks are repeated below against
    // the transaction-fenced mandate snapshot, so a racing policy update
    // still fails closed.
    if let Some(intention) = intention
        && let Some(mandate) = store.get_mandate(&decision.mandate_id).await?
    {
        intention
            .validate_value_against(&mandate)
            .map_err(anyhow::Error::msg)?;
    }
    if decision.outcome == MandateDecisionOutcome::Ask {
        anyhow::ensure!(
            decision
                .question
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty()),
            "ASK requires a concrete question"
        );
    }
    if let Some(alert) = decision.owner_alert.as_deref() {
        anyhow::ensure!(
            decision.outcome == MandateDecisionOutcome::Wait && !alert.trim().is_empty(),
            "owner_alert is a non-empty finding carried by a WAIT"
        );
    }
    anyhow::ensure!(
        decision.outcome == MandateDecisionOutcome::Stop
            || (decision.termination_kind.is_none() && decision.termination_match.is_none()),
        "termination fields are valid only for STOP"
    );

    let now = chrono::Utc::now();
    let now_string = now.to_rfc3339();
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
    if let Some(task_attempt_id) = task_attempt_id {
        anyhow::ensure!(
            !task_attempt_id.trim().is_empty() && task_attempt_id.trim() == task_attempt_id,
            "task attempt id must be canonical"
        );
        // This is deliberately a write-CAS rather than a preceding read.
        // It acquires the SQLite writer lock while proving that the exact
        // dispatcher-owned root attempt is still current, live, and bound
        // to the pinned mandate run. Attempt replacement cannot race the
        // decision insert after this statement within the transaction.
        let fenced = sqlx::query(
            "UPDATE task_attempts
                 SET last_heartbeat_at = datetime('now')
                 WHERE id = ? AND goal_run_id = ?
                   AND status IN ('claimed', 'running')
                   AND datetime(lease_expires_at) > datetime('now')
                   AND EXISTS (
                       SELECT 1
                       FROM tasks t
                       JOIN goal_runs gr ON gr.id = task_attempts.goal_run_id
                       WHERE t.id = task_attempts.task_id
                         AND t.current_attempt_id = task_attempts.id
                         AND t.status IN ('claimed', 'running')
                         AND t.error IS NULL AND t.blocker IS NULL
                         AND gr.id = ? AND gr.root_task_id = t.id
                         AND gr.trigger_type = 'mandate' AND gr.status = 'running'
                   )",
        )
        .bind(task_attempt_id)
        .bind(&decision.goal_run_id)
        .bind(&decision.goal_run_id)
        .execute(&mut *tx)
        .await?;
        anyhow::ensure!(
            fenced.rows_affected() == 1,
            "mandate task-lead attempt is no longer current and executable"
        );
    }
    let query = format!(
        "SELECT {}
             FROM mandates m
             JOIN goals g ON g.id = m.goal_id
             JOIN goal_runs gr ON gr.goal_id = m.goal_id
             WHERE m.id = ? AND gr.id = ?
               AND m.status = 'active'
               AND m.confirmed_at IS NOT NULL
               AND m.review_lease_token IS NOT NULL
               AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
               AND g.domain = 'orchestration' AND g.goal_type = 'continuous'
               AND g.status = 'active'
               AND gr.trigger_type = 'mandate'
               AND gr.status = 'running'",
        qualified_columns("m", MANDATE_COLUMNS)
    );
    let row = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(&decision.mandate_id)
        .bind(&decision.goal_run_id)
        .bind(&now_string)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or_else(|| anyhow::anyhow!("mandate is not actively leased for this goal run"))?;
    let mandate = mandate_from_row(&row)?;
    anyhow::ensure!(
        decision.mandate_version == mandate.version,
        "mandate version changed before the decision was committed"
    );
    anyhow::ensure!(
        decision.owner_alert.is_none() || mandate.authority.owner_alerts,
        "this mandate's confirmed policy does not include telling the owner what it finds; record WAIT without owner_alert, or ASK the owner to confirm a policy update that adds it"
    );
    validate_current_run_receipt_refs(
        &mut tx,
        &decision.goal_run_id,
        &decision.evidence_receipt_ids,
    )
    .await?;
    // A runtime-authored fallback is a typed review-failure marker, not a
    // deliberation: it carries no evidence, no intention, and no
    // authority, so the semantic gates below (measurement) cannot apply
    // to it. Structural gates (lease, version, one decision
    // per run) still apply.
    let runtime_fallback = decision.is_runtime_fallback();
    if runtime_fallback {
        anyhow::ensure!(
            decision.outcome == MandateDecisionOutcome::Wait
                && decision.evidence_receipt_ids.is_empty()
                && decision.action_attempts == 0
                && intention.is_none()
                && operating_updates.is_none(),
            "runtime fallback decisions must be bare WAIT markers"
        );
    }
    // Whether ACT is admissible now is decided here, in the transaction
    // that commits the decision, by the same rules as every other
    // outcome (`admissibility`), so no caller keeps its own copy.
    if decision.outcome == MandateDecisionOutcome::Act {
        let quota = mutation_quota_state_on_connection(&mut tx, &mandate.id, &now_string).await?;
        let block = match &quota {
            Some(quota) => quota.block_reason,
            None => Some(MandateMutationQuotaBlockReason::MutationDisabled),
        };
        let admissible =
            crate::admissibility::admissible_outcomes(&crate::admissibility::OutcomeState {
                quota_block: block,
            });
        if !admissible.act {
            if block == Some(MandateMutationQuotaBlockReason::MutationDisabled) {
                anyhow::bail!(
                    "ACT requires a positive governed mutation budget; choose WAIT, ASK, or STOP for an observation-only cycle"
                );
            }
            let earliest = quota
                .as_ref()
                .and_then(|quota| quota.earliest_next_slot_at.as_deref())
                .unwrap_or("unavailable");
            anyhow::bail!(
                "mandate_mutation_quota_unavailable: choose WAIT; earliest_next_slot_at={earliest}"
            );
        }
    }
    let current_measurement = if mandate.objective_control.is_some()
        && !runtime_fallback
        && matches!(
            decision.outcome,
            MandateDecisionOutcome::Act
                | MandateDecisionOutcome::Wait
                | MandateDecisionOutcome::Stop
        ) {
        sqlx::query_as::<_, (i64, i64)>(
            "SELECT value_micros, confidence_bps FROM mandate_objective_measurements
                 WHERE mandate_id = ? AND mandate_version = ? AND goal_run_id = ?
                 ORDER BY julianday(observed_at) DESC, id DESC LIMIT 1",
        )
        .bind(&mandate.id)
        .bind(mandate.version)
        .bind(&decision.goal_run_id)
        .fetch_optional(&mut *tx)
        .await?
    } else {
        None
    };
    // The measured objective belongs to autopilot. A bounded mandate's
    // objective control is advisory: recorded measurements are kept, but
    // a decision never waits on one.
    let control_loop = mandate.autonomy_mode.is_autopilot();
    if control_loop
        && mandate.objective_control.is_some()
        && !runtime_fallback
        && matches!(
            decision.outcome,
            MandateDecisionOutcome::Act | MandateDecisionOutcome::Wait
        )
    {
        anyhow::ensure!(
            current_measurement.is_some(),
            "controlled mandate decisions require a receipt-backed metric measurement from the current run"
        );
    }
    if decision.outcome == MandateDecisionOutcome::Stop {
        let termination_kind = decision
            .termination_kind
            .ok_or_else(|| anyhow::anyhow!("STOP requires a typed termination_kind"))?;
        match termination_kind {
            MandateTerminationKind::SuccessCriteriaSatisfied => {
                if let Some(control) = mandate.objective_control.as_ref().filter(|_| control_loop) {
                    anyhow::ensure!(
                        current_measurement.is_some_and(|(value, confidence)| {
                            confidence > 0 && control.target_reached(value)
                        }),
                        "success termination requires a credible (confidence > 0) current receipt-backed metric that reaches the owner-confirmed target"
                    );
                }
                let matched = decision.termination_match.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("success termination requires termination_match")
                })?;
                anyhow::ensure!(
                    mandate
                        .success_criteria
                        .iter()
                        .any(|entry| entry == matched),
                    "termination_match is not an owner-authored success criterion"
                );
            }
            MandateTerminationKind::StopConditionMet => {
                let matched = decision.termination_match.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("stop-condition termination requires termination_match")
                })?;
                anyhow::ensure!(
                    mandate.stop_conditions.iter().any(|entry| entry == matched),
                    "termination_match is not an owner-authored stop condition"
                );
            }
            MandateTerminationKind::SafetyTermination => {}
        }
        anyhow::ensure!(
            !termination_kind.requires_receipt_evidence()
                || !decision.evidence_receipt_ids.is_empty(),
            "evidence-dependent STOP requires at least one current-run receipt"
        );
    }

    let mut next_review_at =
        clamped_next_review_at(&mandate, decision.reconsider_at.as_deref(), now)?;
    if decision.outcome != MandateDecisionOutcome::Stop
        && let Some(control) = mandate.objective_control.as_ref()
    {
        let measurement_due = now + chrono::Duration::seconds(control.measurement_cadence_secs);
        let selected =
            chrono::DateTime::parse_from_rfc3339(&next_review_at)?.with_timezone(&chrono::Utc);
        if selected > measurement_due {
            next_review_at = measurement_due.to_rfc3339();
        }
    }
    let persisted_reconsider_at = if decision.outcome == MandateDecisionOutcome::Stop {
        None
    } else {
        Some(next_review_at.as_str())
    };
    sqlx::query(
        "INSERT INTO mandate_decision_cycles (
                id, mandate_id, goal_run_id, mandate_version, outcome, origin, activity_level,
                rationale, belief_snapshot, evidence_receipt_ids_json, question, owner_alert,
                termination_kind, termination_match, reconsider_at, action_attempts, created_at,
                updated_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&decision.id)
    .bind(&decision.mandate_id)
    .bind(&decision.goal_run_id)
    .bind(decision.mandate_version)
    .bind(decision.outcome.as_str())
    .bind(decision.origin.as_str())
    .bind(decision.activity_level.as_str())
    .bind(&decision.rationale)
    .bind(&decision.belief_snapshot)
    .bind(serde_json::to_string(&decision.evidence_receipt_ids)?)
    .bind(&decision.question)
    .bind(&decision.owner_alert)
    .bind(
        decision
            .termination_kind
            .map(MandateTerminationKind::as_str),
    )
    .bind(&decision.termination_match)
    .bind(persisted_reconsider_at)
    .bind(decision.action_attempts)
    .bind(&decision.created_at)
    .bind(&decision.updated_at)
    .execute(&mut *tx)
    .await?;

    if let Some(intention) = intention {
        anyhow::ensure!(
            intention.mandate_id == decision.mandate_id
                && intention.decision_cycle_id == decision.id
                && intention.goal_run_id == decision.goal_run_id,
            "intention does not belong to this decision cycle"
        );
        intention
            .validate_content_bounds()
            .map_err(anyhow::Error::msg)?;
        intention
            .validate_value_against(&mandate)
            .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            intention.status == IntentionStatus::Committed && intention.completed_at.is_none(),
            "a new intention must be committed and incomplete"
        );
        sqlx::query(
            "INSERT INTO intentions (
                    id, mandate_id, decision_cycle_id, goal_run_id, description,
                    rationale, value_criterion, expected_benefit, risk, invalidation_criteria,
                    format_key, status, created_at, updated_at, completed_at
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&intention.id)
        .bind(&intention.mandate_id)
        .bind(&intention.decision_cycle_id)
        .bind(&intention.goal_run_id)
        .bind(&intention.description)
        .bind(&intention.rationale)
        .bind(&intention.value_criterion)
        .bind(&intention.expected_benefit)
        .bind(&intention.risk)
        .bind(&intention.invalidation_criteria)
        .bind(&intention.format_key)
        .bind(intention.status.as_str())
        .bind(&intention.created_at)
        .bind(&intention.updated_at)
        .bind(&intention.completed_at)
        .execute(&mut *tx)
        .await?;
    }

    if let Some(updates) = operating_updates
        && let Some(note) = updates.learning_note.as_ref()
    {
        sqlx::query(
            "INSERT INTO mandate_learning_notes (
                        id, mandate_id, mandate_version, learned_in_decision_cycle_id,
                        summary, evidence_receipt_ids_json, created_at
                     ) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&note.id)
        .bind(&note.mandate_id)
        .bind(note.mandate_version)
        .bind(&note.learned_in_decision_cycle_id)
        .bind(&note.summary)
        .bind(serde_json::to_string(&note.evidence_receipt_ids)?)
        .bind(&note.created_at)
        .execute(&mut *tx)
        .await?;
    }

    if let Some(updates) = operating_updates {
        for outcome in &updates.action_outcomes {
            // The resource must be one this mandate created with a strictly
            // successful mutation; its format comes from that mutation's
            // intention, never from the caller.
            let created = sqlx::query_as::<_, (Option<String>,)>(
                "SELECT i.format_key
                     FROM mandate_mutation_attempts a
                     JOIN intentions i ON i.id = a.intention_id
                     WHERE a.mandate_id = ? AND a.external_resource_id = ?
                       AND a.status = 'succeeded'
                     ORDER BY julianday(a.reserved_at) DESC, a.id DESC LIMIT 1",
            )
            .bind(&decision.mandate_id)
            .bind(&outcome.external_resource_id)
            .fetch_optional(&mut *tx)
            .await?;
            let Some((format_key,)) = created else {
                anyhow::bail!(
                    "action_outcomes external_resource_id `{}` is not a resource this mandate created with a succeeded action",
                    outcome.external_resource_id
                );
            };
            let format_key = format_key.filter(|key| crate::traits::is_canonical_format_key(key));
            sqlx::query(
                "INSERT INTO mandate_action_outcomes (
                        id, mandate_id, mandate_version, decision_cycle_id,
                        external_resource_id, format_key, engagements_micros, impressions,
                        evidence_receipt_ids_json, observed_at, created_at
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&outcome.id)
            .bind(&outcome.mandate_id)
            .bind(outcome.mandate_version)
            .bind(&outcome.decision_cycle_id)
            .bind(&outcome.external_resource_id)
            .bind(&format_key)
            .bind(outcome.engagements_micros)
            .bind(outcome.impressions)
            .bind(serde_json::to_string(&outcome.evidence_receipt_ids)?)
            .bind(&outcome.observed_at)
            .bind(&outcome.created_at)
            .execute(&mut *tx)
            .await?;
        }
    }

    // The durable decision remains provisional until the root task itself
    // is durably successful and the proof finalizer closes the run. This
    // keeps every post-decision crash inside active orphan reconciliation.
    let transitioned = sqlx::query(
        "UPDATE mandates
             SET next_review_at = ?, review_lease_token = NULL,
                 review_lease_expires_at = NULL, updated_at = ?
             WHERE id = ? AND status = 'active' AND version = ?
               AND confirmed_at IS NOT NULL
               AND review_lease_token IS NOT NULL",
    )
    .bind(&next_review_at)
    .bind(&now_string)
    .bind(&decision.mandate_id)
    .bind(decision.mandate_version)
    .execute(&mut *tx)
    .await?;
    anyhow::ensure!(
        transitioned.rows_affected() == 1,
        "mandate changed before the decision was committed"
    );
    tx.commit().await?;
    Ok(())
}

/// Checks the outcomes a decision records before any write: each is valid,
/// belongs to this decision, cites only the decision's own evidence, and
/// names a distinct resource.
fn validate_action_outcomes(
    decision: &MandateDecisionCycle,
    outcomes: &[MandateActionOutcome],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        outcomes.len() <= crate::traits::MAX_MANDATE_ACTION_OUTCOMES,
        "a decision records at most {} action outcomes",
        crate::traits::MAX_MANDATE_ACTION_OUTCOMES
    );
    let mut seen = std::collections::HashSet::new();
    for outcome in outcomes {
        outcome.validate().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            outcome.mandate_id == decision.mandate_id
                && outcome.mandate_version == decision.mandate_version
                && outcome.decision_cycle_id == decision.id,
            "action outcome does not belong to this decision"
        );
        anyhow::ensure!(
            outcome
                .evidence_receipt_ids
                .iter()
                .all(|id| decision.evidence_receipt_ids.contains(id)),
            "action outcome evidence must be a subset of current decision evidence"
        );
        anyhow::ensure!(
            seen.insert(outcome.external_resource_id.as_str()),
            "action outcome `{}` is recorded twice",
            outcome.external_resource_id
        );
    }
    Ok(())
}

pub(super) async fn list_mandate_action_outcomes<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    limit: i64,
) -> anyhow::Result<Vec<MandateActionOutcome>> {
    let query = format!(
        "SELECT {ACTION_OUTCOME_COLUMNS} FROM mandate_action_outcomes
             WHERE mandate_id = ? ORDER BY julianday(observed_at) DESC, id DESC LIMIT ?"
    );
    let rows = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(mandate_id)
        .bind(limit.clamp(1, 200))
        .fetch_all(store.work_pool())
        .await?;
    rows.iter().map(action_outcome_from_row).collect()
}

pub(super) async fn get_mandate_decision_for_run<S: WorkDatabase>(
    store: &S,
    goal_run_id: &str,
) -> anyhow::Result<Option<MandateDecisionCycle>> {
    let query =
        format!("SELECT {DECISION_COLUMNS} FROM mandate_decision_cycles WHERE goal_run_id = ?");
    let row = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(goal_run_id)
        .fetch_optional(store.work_pool())
        .await?;
    row.as_ref().map(decision_from_row).transpose()
}

pub(super) async fn current_run_evidence_receipt_ids<S: WorkDatabase>(
    store: &S,
    goal_run_id: &str,
    limit: usize,
) -> anyhow::Result<Vec<String>> {
    // The predicate of `validate_current_run_receipt_refs`, so every id
    // returned here passes that check.
    let excluded = crate::MANDATE_PROTOCOL_TOOLS
        .iter()
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(", ");
    let query = format!(
        "SELECT json_extract(e.data, '$.tool_call_id')
             FROM events e
             JOIN tasks t ON t.id = json_extract(e.data, '$.task_id')
             WHERE e.event_type = 'tool_result'
               AND t.goal_run_id = ?
               AND json_extract(e.data, '$.receipt.schema_version') = ?
               AND json_extract(e.data, '$.receipt.outcome_status') = 'succeeded'
               AND json_extract(e.data, '$.receipt.outcome_evidence')
                   IN ('tool_reported', 'structured_metadata')
               AND json_extract(e.data, '$.tool_call_id') IS NOT NULL
               AND json_extract(e.data, '$.name') NOT IN ({excluded})
             ORDER BY e.id DESC
             LIMIT ?"
    );
    let mut statement = sqlx::query_scalar::<_, String>(sqlx::AssertSqlSafe(&*query))
        .bind(goal_run_id)
        .bind(i64::from(aidaemon_events::ToolReceiptV1::SCHEMA_VERSION));
    for tool in crate::MANDATE_PROTOCOL_TOOLS {
        statement = statement.bind(*tool);
    }
    Ok(statement
        .bind(i64::try_from(limit).unwrap_or(i64::MAX))
        .fetch_all(store.work_pool())
        .await?)
}

pub(super) async fn list_mandate_decisions<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    limit: i64,
) -> anyhow::Result<Vec<MandateDecisionCycle>> {
    let query = format!(
        "SELECT {DECISION_COLUMNS} FROM mandate_decision_cycles
             WHERE mandate_id = ? ORDER BY julianday(created_at) DESC, id DESC LIMIT ?"
    );
    let rows = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(mandate_id)
        .bind(limit.clamp(1, 200))
        .fetch_all(store.work_pool())
        .await?;
    rows.iter().map(decision_from_row).collect()
}

pub(super) async fn list_intentions<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    limit: i64,
) -> anyhow::Result<Vec<Intention>> {
    let query = format!(
        "SELECT {INTENTION_COLUMNS} FROM intentions
             WHERE mandate_id = ? ORDER BY julianday(created_at) DESC, id DESC LIMIT ?"
    );
    let rows = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(mandate_id)
        .bind(limit.clamp(1, 200))
        .fetch_all(store.work_pool())
        .await?;
    rows.iter().map(intention_from_row).collect()
}
