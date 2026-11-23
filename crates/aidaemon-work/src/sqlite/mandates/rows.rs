//! Column lists and row decoding for the mandate tables.

use super::*;

pub(super) const MANDATE_COLUMNS: &str = "id, goal_id, source_goal_id, name, objective, status, autonomy_mode, authority_json, strategy_json, \
     objective_control_json, suspension_json, constraints_json, \
     success_criteria_json, stop_conditions_json, min_review_secs, max_review_secs, \
     default_review_secs, review_effort, next_review_at, review_lease_token, review_lease_expires_at, \
     expires_at, confirmed_at, version, owner_principal_id, created_by_session, created_at, updated_at";

pub(super) const DECISION_COLUMNS: &str = "id, mandate_id, goal_run_id, mandate_version, outcome, origin, activity_level, rationale, belief_snapshot, \
     evidence_receipt_ids_json, question, owner_alert, termination_kind, termination_match, \
     reconsider_at, action_attempts, created_at, updated_at";

pub(super) const INTENTION_COLUMNS: &str = "id, mandate_id, decision_cycle_id, goal_run_id, description, rationale, \
     value_criterion, expected_benefit, risk, invalidation_criteria, status, created_at, updated_at, completed_at";

pub(super) const LEARNING_NOTE_COLUMNS: &str = "id, mandate_id, mandate_version, learned_in_decision_cycle_id, summary, \
     evidence_receipt_ids_json, created_at";

pub(super) const MUTATION_ATTEMPT_COLUMNS: &str = "id, mandate_id, mandate_version, decision_cycle_id, goal_run_id, intention_id, \
     root_task_id, root_task_attempt_id, task_id, task_attempt_id, reserved_action_attempt, \
     action_digest, tool_call_id, tool_name, mutation_effects_json, targets_json, \
     account_identifiers_json, status, outcome_evidence, http_status, exit_code, \
     reserved_at, completed_at";

pub(super) fn qualified_columns(alias: &str, columns: &str) -> String {
    columns
        .split(", ")
        .map(|column| format!("{alias}.{column}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn mandate_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Mandate> {
    let status_raw: String = row.get("status");
    let status = MandateStatus::parse(&status_raw)
        .ok_or_else(|| anyhow::anyhow!("invalid mandate status `{status_raw}`"))?;
    Ok(Mandate {
        id: row.get("id"),
        goal_id: row.get("goal_id"),
        source_goal_id: row.get("source_goal_id"),
        name: row.get("name"),
        objective: row.get("objective"),
        status,
        autonomy_mode: MandateAutonomyMode::parse(&row.get::<String, _>("autonomy_mode"))
            .ok_or_else(|| anyhow::anyhow!("invalid mandate autonomy mode"))?,
        authority: serde_json::from_str::<MandateAuthority>(
            &row.get::<String, _>("authority_json"),
        )?,
        strategy: row
            .get::<Option<String>, _>("strategy_json")
            .map(|value| serde_json::from_str(&value))
            .transpose()?,
        objective_control: row
            .get::<Option<String>, _>("objective_control_json")
            .map(|value| serde_json::from_str(&value))
            .transpose()?,
        suspension: row
            .get::<Option<String>, _>("suspension_json")
            .map(|value| serde_json::from_str(&value))
            .transpose()?,
        constraints: serde_json::from_str(&row.get::<String, _>("constraints_json"))?,
        success_criteria: serde_json::from_str(&row.get::<String, _>("success_criteria_json"))?,
        stop_conditions: serde_json::from_str(&row.get::<String, _>("stop_conditions_json"))?,
        min_review_secs: row.get("min_review_secs"),
        max_review_secs: row.get("max_review_secs"),
        default_review_secs: row.get("default_review_secs"),
        review_effort: row.get("review_effort"),
        next_review_at: row.get("next_review_at"),
        review_lease_token: row.get("review_lease_token"),
        review_lease_expires_at: row.get("review_lease_expires_at"),
        expires_at: row.get("expires_at"),
        confirmed_at: row.get("confirmed_at"),
        version: row.get("version"),
        owner_principal_id: row.get("owner_principal_id"),
        created_by_session: row.get("created_by_session"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

pub(super) fn decision_from_row(
    row: &sqlx::sqlite::SqliteRow,
) -> anyhow::Result<MandateDecisionCycle> {
    let outcome_raw: String = row.get("outcome");
    let outcome = MandateDecisionOutcome::parse(&outcome_raw)
        .ok_or_else(|| anyhow::anyhow!("invalid mandate decision outcome `{outcome_raw}`"))?;
    Ok(MandateDecisionCycle {
        id: row.get("id"),
        mandate_id: row.get("mandate_id"),
        goal_run_id: row.get("goal_run_id"),
        mandate_version: row.get("mandate_version"),
        outcome,
        origin: MandateDecisionOrigin::parse(&row.get::<String, _>("origin"))
            .ok_or_else(|| anyhow::anyhow!("invalid mandate decision origin"))?,
        activity_level: MandateActivityLevel::parse(&row.get::<String, _>("activity_level"))
            .ok_or_else(|| anyhow::anyhow!("invalid mandate activity level"))?,
        rationale: row.get("rationale"),
        belief_snapshot: row.get("belief_snapshot"),
        evidence_receipt_ids: serde_json::from_str(
            &row.get::<String, _>("evidence_receipt_ids_json"),
        )?,
        question: row.get("question"),
        owner_alert: row.get("owner_alert"),
        termination_kind: row
            .get::<Option<String>, _>("termination_kind")
            .map(|value| {
                MandateTerminationKind::parse(&value)
                    .ok_or_else(|| anyhow::anyhow!("invalid mandate termination kind `{value}`"))
            })
            .transpose()?,
        termination_match: row.get("termination_match"),
        reconsider_at: row.get("reconsider_at"),
        action_attempts: row.get("action_attempts"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

pub(super) fn objective_measurement_from_row(
    row: &sqlx::sqlite::SqliteRow,
) -> anyhow::Result<MandateObjectiveMeasurement> {
    Ok(MandateObjectiveMeasurement {
        id: row.get("id"),
        mandate_id: row.get("mandate_id"),
        mandate_version: row.get("mandate_version"),
        goal_run_id: row.get("goal_run_id"),
        value_micros: row.get("value_micros"),
        confidence_bps: u16::try_from(row.get::<i64, _>("confidence_bps"))?,
        evidence_receipt_ids: serde_json::from_str(
            &row.get::<String, _>("evidence_receipt_ids_json"),
        )?,
        observed_at: row.get("observed_at"),
        created_at: row.get("created_at"),
    })
}

pub(super) fn intention_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Intention> {
    let status_raw: String = row.get("status");
    let status = IntentionStatus::parse(&status_raw)
        .ok_or_else(|| anyhow::anyhow!("invalid intention status `{status_raw}`"))?;
    Ok(Intention {
        id: row.get("id"),
        mandate_id: row.get("mandate_id"),
        decision_cycle_id: row.get("decision_cycle_id"),
        goal_run_id: row.get("goal_run_id"),
        description: row.get("description"),
        rationale: row.get("rationale"),
        value_criterion: row.get("value_criterion"),
        expected_benefit: row.get("expected_benefit"),
        risk: row.get("risk"),
        invalidation_criteria: row.get("invalidation_criteria"),
        status,
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        completed_at: row.get("completed_at"),
    })
}

pub(super) fn learning_note_from_row(
    row: &sqlx::sqlite::SqliteRow,
) -> anyhow::Result<MandateLearningNote> {
    Ok(MandateLearningNote {
        id: row.get("id"),
        mandate_id: row.get("mandate_id"),
        mandate_version: row.get("mandate_version"),
        learned_in_decision_cycle_id: row.get("learned_in_decision_cycle_id"),
        summary: row.get("summary"),
        evidence_receipt_ids: serde_json::from_str(
            &row.get::<String, _>("evidence_receipt_ids_json"),
        )?,
        created_at: row.get("created_at"),
    })
}

pub(super) fn mutation_attempt_from_row(
    row: &sqlx::sqlite::SqliteRow,
) -> anyhow::Result<MandateMutationAttempt> {
    let status_raw: String = row.get("status");
    let status = MandateMutationAttemptStatus::parse(&status_raw)
        .ok_or_else(|| anyhow::anyhow!("invalid mandate mutation status `{status_raw}`"))?;
    let evidence_raw: Option<String> = row.get("outcome_evidence");
    let outcome_evidence = evidence_raw
        .as_deref()
        .map(|value| {
            MandateMutationEvidence::parse(value)
                .ok_or_else(|| anyhow::anyhow!("invalid mandate mutation evidence `{value}`"))
        })
        .transpose()?;
    let http_status = row
        .get::<Option<i64>, _>("http_status")
        .map(u16::try_from)
        .transpose()
        .map_err(|_| anyhow::anyhow!("invalid mandate mutation HTTP status"))?;
    Ok(MandateMutationAttempt {
        id: row.get("id"),
        mandate_id: row.get("mandate_id"),
        mandate_version: row.get("mandate_version"),
        decision_cycle_id: row.get("decision_cycle_id"),
        goal_run_id: row.get("goal_run_id"),
        intention_id: row.get("intention_id"),
        root_task_id: row.get("root_task_id"),
        root_task_attempt_id: row.get("root_task_attempt_id"),
        task_id: row.get("task_id"),
        task_attempt_id: row.get("task_attempt_id"),
        reserved_action_attempt: row.get("reserved_action_attempt"),
        action_digest: row.get("action_digest"),
        tool_call_id: row.get("tool_call_id"),
        tool_name: row.get("tool_name"),
        mutation_effects: serde_json::from_str(&row.get::<String, _>("mutation_effects_json"))?,
        targets: serde_json::from_str(&row.get::<String, _>("targets_json"))?,
        account_identifiers: serde_json::from_str(
            &row.get::<String, _>("account_identifiers_json"),
        )?,
        status,
        outcome_evidence,
        http_status,
        exit_code: row.get("exit_code"),
        reserved_at: row.get("reserved_at"),
        completed_at: row.get("completed_at"),
    })
}
