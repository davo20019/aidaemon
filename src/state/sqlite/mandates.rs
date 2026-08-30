use super::*;
use crate::traits::{
    GoalRun, Intention, IntentionStatus, Mandate, MandateActivityLevel, MandateAuthority,
    MandateAutonomyMode, MandateDecisionCycle, MandateDecisionOutcome,
    MandateFinalizationRejectReason, MandateFinalizationStaleReason, MandateLearningNote,
    MandateMutationAttempt, MandateMutationAttemptStatus, MandateMutationDispatchClaim,
    MandateMutationEvidence, MandateMutationOutcomeProjection, MandateMutationQuotaBlockReason,
    MandateMutationQuotaState, MandateMutationReservation, MandateObjectiveMeasurement,
    MandateOperatingUpdates, MandateReconciliationReason, MandateReconciliationResolution,
    MandateRunFinalizationRequest, MandateRunFinalizationResult, MandateRunProofCounts,
    MandateStatus, MandateStore, MandateStrategyRevision, MandateStrategyRevisionKind,
    MandateSuspension, MandateSuspensionKind, MandateTerminationKind, MandateWakeSignal, Task,
};
use sha2::{Digest, Sha256};

const MANDATE_COLUMNS: &str =
    "id, goal_id, source_goal_id, objective, status, autonomy_mode, authority_json, strategy_json, \
     objective_control_json, suspension_json, constraints_json, \
     success_criteria_json, stop_conditions_json, min_review_secs, max_review_secs, \
     default_review_secs, review_effort, next_review_at, review_lease_token, review_lease_expires_at, \
     expires_at, confirmed_at, version, owner_principal_id, created_by_session, created_at, updated_at";

const DECISION_COLUMNS: &str =
    "id, mandate_id, goal_run_id, mandate_version, outcome, activity_level, rationale, belief_snapshot, \
     evidence_receipt_ids_json, question, termination_kind, termination_match, reconsider_at, \
     action_attempts, created_at, updated_at";

const INTENTION_COLUMNS: &str =
    "id, mandate_id, decision_cycle_id, goal_run_id, description, rationale, \
     value_criterion, expected_benefit, risk, invalidation_criteria, status, created_at, updated_at, completed_at";

const LEARNING_NOTE_COLUMNS: &str =
    "id, mandate_id, mandate_version, learned_in_decision_cycle_id, summary, \
     evidence_receipt_ids_json, created_at";

const STRATEGY_REVISION_COLUMNS: &str =
    "id, mandate_id, mandate_version, decision_cycle_id, strategy_key, kind, guidance, \
     confidence_bps, evidence_receipt_ids_json, created_at";

const MUTATION_ATTEMPT_COLUMNS: &str =
    "id, mandate_id, mandate_version, decision_cycle_id, goal_run_id, intention_id, \
     root_task_id, root_task_attempt_id, task_id, task_attempt_id, reserved_action_attempt, \
     action_digest, tool_call_id, tool_name, mutation_effects_json, targets_json, \
     account_identifiers_json, status, outcome_evidence, http_status, exit_code, \
     reserved_at, completed_at";

fn qualified_columns(alias: &str, columns: &str) -> String {
    columns
        .split(", ")
        .map(|column| format!("{alias}.{column}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn mandate_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Mandate> {
    let status_raw: String = row.get("status");
    let status = MandateStatus::parse(&status_raw)
        .ok_or_else(|| anyhow::anyhow!("invalid mandate status `{status_raw}`"))?;
    Ok(Mandate {
        id: row.get("id"),
        goal_id: row.get("goal_id"),
        source_goal_id: row.get("source_goal_id"),
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

fn decision_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<MandateDecisionCycle> {
    let outcome_raw: String = row.get("outcome");
    let outcome = MandateDecisionOutcome::parse(&outcome_raw)
        .ok_or_else(|| anyhow::anyhow!("invalid mandate decision outcome `{outcome_raw}`"))?;
    Ok(MandateDecisionCycle {
        id: row.get("id"),
        mandate_id: row.get("mandate_id"),
        goal_run_id: row.get("goal_run_id"),
        mandate_version: row.get("mandate_version"),
        outcome,
        activity_level: MandateActivityLevel::parse(&row.get::<String, _>("activity_level"))
            .ok_or_else(|| anyhow::anyhow!("invalid mandate activity level"))?,
        rationale: row.get("rationale"),
        belief_snapshot: row.get("belief_snapshot"),
        evidence_receipt_ids: serde_json::from_str(
            &row.get::<String, _>("evidence_receipt_ids_json"),
        )?,
        question: row.get("question"),
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

fn objective_measurement_from_row(
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
        attributed_intention_ids: serde_json::from_str(
            &row.get::<String, _>("attributed_intention_ids_json"),
        )?,
        observed_at: row.get("observed_at"),
        created_at: row.get("created_at"),
    })
}

fn intention_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Intention> {
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

fn learning_note_from_row(row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<MandateLearningNote> {
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

fn strategy_revision_from_row(
    row: &sqlx::sqlite::SqliteRow,
) -> anyhow::Result<MandateStrategyRevision> {
    let kind_raw: String = row.get("kind");
    Ok(MandateStrategyRevision {
        id: row.get("id"),
        mandate_id: row.get("mandate_id"),
        mandate_version: row.get("mandate_version"),
        decision_cycle_id: row.get("decision_cycle_id"),
        strategy_key: row.get("strategy_key"),
        kind: MandateStrategyRevisionKind::parse(&kind_raw).ok_or_else(|| {
            anyhow::anyhow!("invalid mandate strategy revision kind `{kind_raw}`")
        })?,
        guidance: row.get("guidance"),
        confidence_bps: row.get::<i64, _>("confidence_bps").try_into()?,
        evidence_receipt_ids: serde_json::from_str(
            &row.get::<String, _>("evidence_receipt_ids_json"),
        )?,
        created_at: row.get("created_at"),
    })
}

fn mutation_attempt_from_row(
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

fn validate_timestamp(label: &str, value: &str) -> anyhow::Result<()> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("invalid {label}: {error}"))
}

fn validate_mandate(mandate: &Mandate) -> anyhow::Result<()> {
    anyhow::ensure!(!mandate.id.trim().is_empty(), "mandate id is required");
    mandate
        .validate_content_bounds()
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        !mandate.created_by_session.trim().is_empty(),
        "mandate owner session is required"
    );
    anyhow::ensure!(
        mandate.owner_principal_id.starts_with("principal:")
            && mandate.owner_principal_id.len() <= 256
            && !mandate.owner_principal_id.chars().any(char::is_control),
        "mandate owner principal is invalid"
    );
    mandate.authority.validate().map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        !mandate.autonomy_mode.is_autopilot() || mandate.objective_control.is_some(),
        "autopilot mandates require a validated objective control"
    );
    if let Some(control) = mandate.objective_control.as_ref() {
        anyhow::ensure!(
            control.measurement_cadence_secs >= mandate.min_review_secs
                && control.measurement_cadence_secs <= mandate.max_review_secs,
            "objective measurement cadence must fit inside the mandate review bounds"
        );
    }
    anyhow::ensure!(
        mandate.min_review_secs > 0
            && mandate.min_review_secs <= mandate.default_review_secs
            && mandate.default_review_secs <= mandate.max_review_secs,
        "mandate review bounds must satisfy 0 < min <= default <= max"
    );
    anyhow::ensure!(mandate.version > 0, "mandate version must be positive");
    validate_timestamp("next_review_at", &mandate.next_review_at)?;
    if let Some(expires_at) = mandate.expires_at.as_deref() {
        validate_timestamp("expires_at", expires_at)?;
    }
    if let Some(confirmed_at) = mandate.confirmed_at.as_deref() {
        validate_timestamp("confirmed_at", confirmed_at)?;
    }
    anyhow::ensure!(
        mandate.status != MandateStatus::Active || mandate.confirmed_at.is_some(),
        "an active mandate requires durable owner confirmation"
    );
    anyhow::ensure!(
        match mandate.status {
            MandateStatus::Active | MandateStatus::Completed | MandateStatus::Cancelled => {
                mandate.suspension.is_none()
            }
            MandateStatus::Paused if mandate.confirmed_at.is_none() => mandate.suspension.is_none(),
            MandateStatus::Paused => mandate
                .suspension
                .as_ref()
                .is_some_and(|value| value.kind == MandateSuspensionKind::OwnerPaused),
            MandateStatus::AwaitingInput => mandate.suspension.is_some(),
        },
        "mandate lifecycle status and typed suspension disagree"
    );
    match (
        mandate.review_lease_token.as_deref(),
        mandate.review_lease_expires_at.as_deref(),
    ) {
        (None, None) => {}
        (Some(token), Some(expires_at)) => {
            anyhow::ensure!(!token.trim().is_empty(), "review lease token is empty");
            validate_timestamp("review_lease_expires_at", expires_at)?;
        }
        _ => {
            anyhow::bail!("review lease token and expiry must either both be set or both be empty")
        }
    }
    Ok(())
}

fn validate_new_mandate(mandate: &Mandate) -> anyhow::Result<()> {
    validate_mandate(mandate)?;
    anyhow::ensure!(
        mandate.version == 1,
        "a new mandate must start at version 1"
    );
    anyhow::ensure!(
        mandate.review_lease_token.is_none() && mandate.review_lease_expires_at.is_none(),
        "a new mandate cannot start with a review lease"
    );
    Ok(())
}

fn controller_goal_status(mandate_status: MandateStatus) -> &'static str {
    match mandate_status {
        MandateStatus::Active => "active",
        MandateStatus::Paused | MandateStatus::AwaitingInput => "paused",
        MandateStatus::Completed => "completed",
        MandateStatus::Cancelled => "cancelled",
    }
}

fn intention_transition(
    from: IntentionStatus,
    to: IntentionStatus,
) -> anyhow::Result<(&'static str, &'static str)> {
    anyhow::ensure!(
        from.can_transition_to(to),
        "invalid intention status transition {from} -> {to}"
    );
    Ok((from.as_str(), to.as_str()))
}

fn clamped_next_review_at(
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

async fn validate_current_run_receipt_refs(
    connection: &mut sqlx::SqliteConnection,
    goal_run_id: &str,
    refs: &[String],
) -> anyhow::Result<()> {
    let mut unique = std::collections::HashSet::new();
    for receipt_id in refs {
        anyhow::ensure!(
            unique.insert(receipt_id),
            "duplicate mandate evidence receipt ID"
        );
        // A durable receipt has two canonical identities that the model can
        // legitimately observe: the tool_call_id it issued and the result_id
        // stamped on the receipt footer it read back. Either one names the
        // same current-run structured success.
        let found = sqlx::query_scalar::<_, i64>(
            "SELECT 1
             FROM events e
             JOIN tasks t ON t.id = json_extract(e.data, '$.task_id')
             WHERE e.event_type = 'tool_result'
               AND t.goal_run_id = ?
               AND (
                   json_extract(e.data, '$.tool_call_id') = ?
                   OR json_extract(e.data, '$.receipt.result_provenance.result_id') = ?
               )
               AND json_extract(e.data, '$.receipt.schema_version') = ?
               AND json_extract(e.data, '$.receipt.outcome_status') = 'succeeded'
               AND json_extract(e.data, '$.receipt.outcome_evidence')
                   IN ('tool_reported', 'structured_metadata')
             LIMIT 1",
        )
        .bind(goal_run_id)
        .bind(receipt_id)
        .bind(receipt_id)
        .bind(i64::from(crate::events::ToolReceiptV1::SCHEMA_VERSION))
        .fetch_optional(&mut *connection)
        .await?;
        anyhow::ensure!(
            found.is_some(),
            "mandate evidence receipt `{receipt_id}` is not a current-run structured success"
        );
    }
    Ok(())
}

fn validate_canonical_identifier(label: &str, value: &str, max_chars: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty()
            && value.trim() == value
            && value.chars().count() <= max_chars
            && !value.chars().any(char::is_control),
        "invalid mandate mutation {label}"
    );
    Ok(())
}

fn validate_mutation_reservation(
    reservation: &MandateMutationReservation,
) -> anyhow::Result<chrono::DateTime<chrono::Utc>> {
    let grant = &reservation.grant;
    validate_canonical_identifier("mandate id", &grant.mandate_id, 256)?;
    validate_canonical_identifier("owner principal id", &grant.owner_principal_id, 256)?;
    anyhow::ensure!(
        grant.owner_principal_id.starts_with("principal:"),
        "invalid mandate owner principal"
    );
    validate_canonical_identifier("decision cycle id", &grant.decision_cycle_id, 256)?;
    anyhow::ensure!(grant.mandate_version > 0, "invalid mandate version");
    anyhow::ensure!(
        grant.counts_toward_cycle_budget && grant.reserved_action_attempt > 0,
        "mutation reservation requires a positive metered grant"
    );
    anyhow::ensure!(
        grant.action_digest.len() == 64
            && grant
                .action_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "invalid mandate mutation action digest"
    );
    anyhow::ensure!(
        grant.tool_call_id.as_deref() == Some(reservation.tool_call_id.as_str()),
        "mutation reservation tool call does not match its bound grant"
    );
    for (label, value) in [
        ("goal run id", reservation.goal_run_id.as_str()),
        ("root task id", reservation.root_task_id.as_str()),
        (
            "root task attempt id",
            reservation.root_task_attempt_id.as_str(),
        ),
        ("task id", reservation.task_id.as_str()),
        ("task attempt id", reservation.task_attempt_id.as_str()),
        ("tool call id", reservation.tool_call_id.as_str()),
        ("tool name", reservation.tool_name.as_str()),
    ] {
        validate_canonical_identifier(label, value, 256)?;
    }
    anyhow::ensure!(
        reservation.task_id != reservation.root_task_id,
        "the mandate deliberator cannot reserve mutations"
    );
    anyhow::ensure!(
        !reservation.mutation_effects.is_empty()
            && reservation.mutation_effects.len() <= MandateAuthority::EFFECT_NAMES.len(),
        "invalid mandate mutation effects"
    );
    let mut seen_effects = std::collections::HashSet::new();
    for effect in &reservation.mutation_effects {
        anyhow::ensure!(
            MandateAuthority::EFFECT_NAMES.contains(&effect.as_str())
                && seen_effects.insert(effect.as_str()),
            "invalid or duplicate mandate mutation effect"
        );
    }
    anyhow::ensure!(
        !reservation.targets.is_empty() && reservation.targets.len() <= 16,
        "mandate mutations require bounded typed targets"
    );
    let mut seen_targets = std::collections::HashSet::new();
    for target in &reservation.targets {
        anyhow::ensure!(
            matches!(target.kind.as_str(), "url" | "resource_id"),
            "unsupported mandate mutation target kind"
        );
        validate_canonical_identifier("target", &target.identifier, 2_048)?;
        anyhow::ensure!(
            seen_targets.insert((target.kind.as_str(), target.identifier.as_str())),
            "duplicate mandate mutation target"
        );
        if target.kind == "url" {
            let parsed = reqwest::Url::parse(&target.identifier)
                .map_err(|_| anyhow::anyhow!("invalid mandate mutation URL target"))?;
            anyhow::ensure!(
                parsed.query().is_none()
                    && parsed.fragment().is_none()
                    && parsed.username().is_empty()
                    && parsed.password().is_none(),
                "mandate mutation audit URLs cannot contain query, fragment, or userinfo"
            );
        }
    }
    anyhow::ensure!(
        reservation.account_identifiers.len() <= 8,
        "too many mandate mutation account identifiers"
    );
    let resource_targets = reservation
        .targets
        .iter()
        .filter(|target| target.kind == "resource_id")
        .map(|target| target.identifier.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut seen_accounts = std::collections::HashSet::new();
    for account in &reservation.account_identifiers {
        validate_canonical_identifier("account identifier", account, 256)?;
        anyhow::ensure!(
            (account.starts_with("auth_profile:") || account.starts_with("account:"))
                && resource_targets.contains(account.as_str())
                && seen_accounts.insert(account.as_str()),
            "invalid mandate mutation account identifier"
        );
    }
    let reserved_at = chrono::DateTime::parse_from_rfc3339(&reservation.reserved_at)
        .map_err(|_| anyhow::anyhow!("invalid mandate mutation reservation timestamp"))?
        .with_timezone(&chrono::Utc);
    Ok(reserved_at)
}

async fn mutation_quota_state_on_connection(
    connection: &mut sqlx::SqliteConnection,
    mandate_id: &str,
    as_of_raw: &str,
) -> anyhow::Result<Option<MandateMutationQuotaState>> {
    let as_of = chrono::DateTime::parse_from_rfc3339(as_of_raw)
        .map_err(|_| anyhow::anyhow!("invalid mutation quota timestamp"))?
        .with_timezone(&chrono::Utc);
    let authority_json =
        sqlx::query_scalar::<_, String>("SELECT authority_json FROM mandates WHERE id = ?")
            .bind(mandate_id)
            .fetch_optional(&mut *connection)
            .await?;
    let Some(authority_json) = authority_json else {
        return Ok(None);
    };
    let authority: MandateAuthority = serde_json::from_str(&authority_json)?;
    authority.validate().map_err(anyhow::Error::msg)?;
    let window_start = as_of - chrono::Duration::hours(24);
    let rows = sqlx::query_scalar::<_, String>(
        "SELECT reserved_at FROM mandate_mutation_attempts
         WHERE mandate_id = ? AND julianday(reserved_at) > julianday(?)
           AND status != 'never_dispatched'
         ORDER BY julianday(reserved_at), id",
    )
    .bind(mandate_id)
    .bind(window_start.to_rfc3339())
    .fetch_all(&mut *connection)
    .await?;
    let mut reservations = Vec::with_capacity(rows.len());
    for row in rows {
        reservations.push(
            chrono::DateTime::parse_from_rfc3339(&row)
                .map_err(|_| anyhow::anyhow!("invalid persisted mutation reservation timestamp"))?
                .with_timezone(&chrono::Utc),
        );
    }
    let used = u32::try_from(reservations.len()).unwrap_or(u32::MAX);
    let max = authority.max_mutating_actions_per_rolling_24h;
    let remaining = max.saturating_sub(used);
    let last = reservations.last().copied();
    if authority.max_mutating_actions_per_cycle == 0 || max == 0 {
        return Ok(Some(MandateMutationQuotaState {
            mandate_id: mandate_id.to_string(),
            as_of: as_of.to_rfc3339(),
            max_mutating_actions_per_rolling_24h: max,
            min_seconds_between_mutations: authority.min_seconds_between_mutations,
            reserved_in_rolling_24h: used,
            remaining_in_rolling_24h: remaining,
            last_reserved_at: last.map(|value| value.to_rfc3339()),
            available_now: false,
            block_reason: Some(MandateMutationQuotaBlockReason::MutationDisabled),
            earliest_next_slot_at: None,
        }));
    }

    let quota_slot = (used >= max)
        .then(|| reservations.first().copied())
        .flatten()
        .map(|oldest| oldest + chrono::Duration::hours(24));
    let cooldown_slot = last.map(|last| {
        last + chrono::Duration::seconds(i64::from(authority.min_seconds_between_mutations))
    });
    let earliest = match (quota_slot, cooldown_slot) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    };
    let available_now = used < max && earliest.is_none_or(|slot| slot <= as_of);
    let block_reason = if used >= max {
        Some(MandateMutationQuotaBlockReason::Rolling24hExhausted)
    } else if !available_now {
        Some(MandateMutationQuotaBlockReason::Cooldown)
    } else {
        None
    };
    Ok(Some(MandateMutationQuotaState {
        mandate_id: mandate_id.to_string(),
        as_of: as_of.to_rfc3339(),
        max_mutating_actions_per_rolling_24h: max,
        min_seconds_between_mutations: authority.min_seconds_between_mutations,
        reserved_in_rolling_24h: used,
        remaining_in_rolling_24h: remaining,
        last_reserved_at: last.map(|value| value.to_rfc3339()),
        available_now,
        block_reason,
        earliest_next_slot_at: (!available_now)
            .then(|| earliest.map(|value| value.to_rfc3339()))
            .flatten(),
    }))
}

async fn insert_mandate_row(
    connection: &mut sqlx::SqliteConnection,
    mandate: &Mandate,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT INTO mandates (
            id, goal_id, source_goal_id, objective, status, autonomy_mode, authority_json, strategy_json,
            objective_control_json, suspension_json,
            constraints_json, success_criteria_json, stop_conditions_json,
            min_review_secs, max_review_secs, default_review_secs, review_effort, next_review_at,
            review_lease_token, review_lease_expires_at, expires_at, confirmed_at,
            version, owner_principal_id, created_by_session, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&mandate.id)
    .bind(&mandate.goal_id)
    .bind(&mandate.source_goal_id)
    .bind(&mandate.objective)
    .bind(mandate.status.as_str())
    .bind(mandate.autonomy_mode.as_str())
    .bind(serde_json::to_string(&mandate.authority)?)
    .bind(
        mandate
            .strategy
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
    )
    .bind(
        mandate
            .objective_control
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
    )
    .bind(
        mandate
            .suspension
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
    )
    .bind(serde_json::to_string(&mandate.constraints)?)
    .bind(serde_json::to_string(&mandate.success_criteria)?)
    .bind(serde_json::to_string(&mandate.stop_conditions)?)
    .bind(mandate.min_review_secs)
    .bind(mandate.max_review_secs)
    .bind(mandate.default_review_secs)
    .bind(&mandate.review_effort)
    .bind(&mandate.next_review_at)
    .bind(&mandate.review_lease_token)
    .bind(&mandate.review_lease_expires_at)
    .bind(&mandate.expires_at)
    .bind(&mandate.confirmed_at)
    .bind(mandate.version)
    .bind(&mandate.owner_principal_id)
    .bind(&mandate.created_by_session)
    .bind(&mandate.created_at)
    .bind(&mandate.updated_at)
    .execute(connection)
    .await?;
    Ok(())
}

async fn validate_source_goal(
    connection: &mut sqlx::SqliteConnection,
    source_goal_id: Option<&str>,
    owner_session_id: &str,
) -> anyhow::Result<()> {
    let Some(source_goal_id) = source_goal_id else {
        return Ok(());
    };
    let source = sqlx::query("SELECT domain, session_id FROM goals WHERE id = ?")
        .bind(source_goal_id)
        .fetch_optional(connection)
        .await?
        .ok_or_else(|| anyhow::anyhow!("source personal goal not found"))?;
    anyhow::ensure!(
        source.get::<String, _>("domain") == "personal",
        "source_goal_id must reference a personal goal"
    );
    anyhow::ensure!(
        source.get::<String, _>("session_id") == owner_session_id,
        "source personal goal belongs to a different owner session; choose a goal marked mandate-source compatible or omit source_goal_id"
    );
    Ok(())
}

async fn update_controller_status(
    connection: &mut sqlx::SqliteConnection,
    goal_id: &str,
    mandate_status: MandateStatus,
    now: &str,
) -> anyhow::Result<()> {
    let goal_status = controller_goal_status(mandate_status);
    let result = sqlx::query(
        "UPDATE goals
         SET status = ?, updated_at = ?,
             completed_at = CASE WHEN ? IN ('completed', 'cancelled') THEN ? ELSE NULL END
         WHERE id = ? AND domain = 'orchestration' AND goal_type = 'continuous'",
    )
    .bind(goal_status)
    .bind(now)
    .bind(goal_status)
    .bind(now)
    .bind(goal_id)
    .execute(connection)
    .await?;
    anyhow::ensure!(
        result.rows_affected() == 1,
        "mandate controller goal not found"
    );
    Ok(())
}

async fn invalidate_open_mandate_runs(
    connection: &mut sqlx::SqliteConnection,
    goal_id: &str,
    now: &str,
) -> anyhow::Result<i64> {
    let claimed_unresolved: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)
         FROM mandate_mutation_attempts ma
         JOIN goal_runs gr ON gr.id = ma.goal_run_id
         WHERE gr.goal_id = ? AND gr.trigger_type = 'mandate'
           AND gr.status IN ('pending', 'running', 'blocked')
           AND ma.status = 'reserved' AND ma.dispatch_claimed_at IS NOT NULL",
    )
    .bind(goal_id)
    .fetch_one(&mut *connection)
    .await?;

    // Linearize authority revocation with the mutation ledger before leases
    // and runs are closed. A reservation that never reached the final
    // dispatcher is explicitly non-quota/non-ambiguous; once the one-use
    // dispatch claim exists, absence of a strict receipt is externally
    // indeterminate and must remain reconciliation-required.
    sqlx::query(
        "UPDATE mandate_mutation_attempts
         SET status = CASE
                 WHEN dispatch_claimed_at IS NULL THEN 'never_dispatched'
                 ELSE 'ambiguous'
             END,
             completed_at = COALESCE(completed_at, ?)
         WHERE status = 'reserved'
           AND goal_run_id IN (
               SELECT id FROM goal_runs
               WHERE goal_id = ? AND trigger_type = 'mandate'
                 AND status IN ('pending', 'running', 'blocked')
           )",
    )
    .bind(now)
    .bind(goal_id)
    .execute(&mut *connection)
    .await?;

    if claimed_unresolved > 0 {
        let rows = sqlx::query(
            "SELECT m.id AS mandate_id, m.version AS mandate_version,
                    gr.goal_id, gr.id AS goal_run_id, g.session_id,
                    (SELECT COUNT(*) FROM tasks t
                     WHERE t.goal_run_id = gr.id AND t.id != COALESCE(gr.root_task_id, ''))
                        AS non_root_tasks,
                    (SELECT COUNT(*) FROM tasks t
                     WHERE t.goal_run_id = gr.id AND t.id != COALESCE(gr.root_task_id, '')
                       AND t.status = 'completed'
                       AND COALESCE(NULLIF(trim(t.error), ''), '') = ''
                       AND COALESCE(NULLIF(trim(t.blocker), ''), '') = '')
                        AS completed_tasks,
                    (SELECT COUNT(*) FROM tasks t
                     WHERE t.goal_run_id = gr.id AND t.id != COALESCE(gr.root_task_id, '')
                       AND (t.status IN ('failed', 'blocked', 'interrupted')
                         OR COALESCE(NULLIF(trim(t.error), ''), '') != ''
                         OR COALESCE(NULLIF(trim(t.blocker), ''), '') != ''))
                        AS failed_or_blocked_tasks,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id) AS mutation_reservations,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id AND x.status = 'succeeded')
                        AS succeeded_mutations,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id AND x.status = 'failed')
                        AS failed_mutations,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id AND x.status = 'never_dispatched')
                        AS never_dispatched_mutations,
                    (SELECT COUNT(*) FROM mandate_mutation_attempts x
                     WHERE x.goal_run_id = gr.id AND x.status IN ('reserved', 'ambiguous'))
                        AS ambiguous_mutations
             FROM goal_runs gr
             JOIN mandates m ON m.goal_id = gr.goal_id
             JOIN goals g ON g.id = gr.goal_id
             WHERE gr.goal_id = ? AND gr.trigger_type = 'mandate'
               AND gr.status IN ('pending', 'running', 'blocked')
               AND EXISTS (
                   SELECT 1 FROM mandate_mutation_attempts ma
                   WHERE ma.goal_run_id = gr.id AND ma.status = 'ambiguous'
               )",
        )
        .bind(goal_id)
        .fetch_all(&mut *connection)
        .await?;
        let count = |row: &sqlx::sqlite::SqliteRow, name: &str| {
            u32::try_from(row.get::<i64, _>(name).max(0)).unwrap_or(u32::MAX)
        };
        for row in rows {
            let non_root_tasks = count(&row, "non_root_tasks");
            let completed_tasks = count(&row, "completed_tasks");
            let notice = crate::traits::MandateRunNotification::new(
                &row.get::<String, _>("mandate_id"),
                row.get("mandate_version"),
                &row.get::<String, _>("goal_id"),
                &row.get::<String, _>("goal_run_id"),
                &row.get::<String, _>("session_id"),
                crate::traits::MandateRunNotificationKind::AuthorityRevokedWithUnresolvedMutation,
                crate::traits::MandateRunProofCounts {
                    non_root_tasks,
                    completed_tasks,
                    incomplete_tasks: non_root_tasks.saturating_sub(completed_tasks),
                    failed_or_blocked_tasks: count(&row, "failed_or_blocked_tasks"),
                    mutation_reservations: count(&row, "mutation_reservations"),
                    succeeded_mutations: count(&row, "succeeded_mutations"),
                    failed_mutations: count(&row, "failed_mutations"),
                    never_dispatched_mutations: count(&row, "never_dispatched_mutations"),
                    ambiguous_or_reserved_mutations: count(&row, "ambiguous_mutations"),
                },
                now,
            );
            super::notifications::enqueue_mandate_run_notification_on_connection(
                &mut *connection,
                &notice,
            )
            .await?;
        }
    }
    sqlx::query(
        "UPDATE task_attempts
         SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
         WHERE goal_run_id IN (
             SELECT id FROM goal_runs
             WHERE goal_id = ? AND trigger_type = 'mandate'
               AND status IN ('pending', 'running', 'blocked')
         ) AND status IN ('claimed', 'running')",
    )
    .bind(now)
    .bind(goal_id)
    .execute(&mut *connection)
    .await?;
    sqlx::query(
        "UPDATE tasks
         SET status = 'cancelled', current_attempt_id = NULL,
             completed_at = COALESCE(completed_at, ?), updated_at = ?, version = version + 1
         WHERE goal_run_id IN (
             SELECT id FROM goal_runs
             WHERE goal_id = ? AND trigger_type = 'mandate'
               AND status IN ('pending', 'running', 'blocked')
         ) AND status IN ('pending', 'claimed', 'running', 'blocked')",
    )
    .bind(now)
    .bind(now)
    .bind(goal_id)
    .execute(&mut *connection)
    .await?;
    let (intention_from, intention_to) =
        intention_transition(IntentionStatus::Committed, IntentionStatus::Suspended)?;
    sqlx::query(
        "UPDATE intentions
         SET status = ?, completed_at = COALESCE(completed_at, ?), updated_at = ?
         WHERE goal_run_id IN (
             SELECT id FROM goal_runs
             WHERE goal_id = ? AND trigger_type = 'mandate'
               AND status IN ('pending', 'running', 'blocked')
         ) AND status = ?",
    )
    .bind(intention_to)
    .bind(now)
    .bind(now)
    .bind(goal_id)
    .bind(intention_from)
    .execute(&mut *connection)
    .await?;
    sqlx::query(
        "UPDATE goal_runs
         SET status = 'cancelled',
             outcome_summary = CASE WHEN EXISTS (
                 SELECT 1 FROM mandate_mutation_attempts ma
                 WHERE ma.goal_run_id = goal_runs.id AND ma.status = 'ambiguous'
             ) THEN 'mandate_reconciliation_required:lifecycle_invalidated_after_dispatch_claim'
             ELSE 'Mandate lifecycle changed; a fresh decision cycle is required.' END,
             completed_at = COALESCE(completed_at, ?), updated_at = ?
         WHERE goal_id = ? AND trigger_type = 'mandate'
           AND status IN ('pending', 'running', 'blocked')",
    )
    .bind(now)
    .bind(now)
    .bind(goal_id)
    .execute(&mut *connection)
    .await?;
    Ok(claimed_unresolved)
}

async fn mandate_run_proof_counts_on_connection(
    connection: &mut sqlx::SqliteConnection,
    goal_run_id: &str,
    root_task_id: Option<&str>,
) -> anyhow::Result<MandateRunProofCounts> {
    let task_counts = sqlx::query(
        "SELECT COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN status = 'completed'
                    AND COALESCE(NULLIF(trim(error), ''), '') = ''
                    AND COALESCE(NULLIF(trim(blocker), ''), '') = ''
                    THEN 1 ELSE 0 END), 0) AS completed,
                COALESCE(SUM(CASE WHEN status IN ('failed', 'blocked', 'interrupted')
                    OR COALESCE(NULLIF(trim(error), ''), '') != ''
                    OR COALESCE(NULLIF(trim(blocker), ''), '') != ''
                    THEN 1 ELSE 0 END), 0) AS failed_or_blocked
         FROM tasks
         WHERE goal_run_id = ? AND id != COALESCE(?, '')",
    )
    .bind(goal_run_id)
    .bind(root_task_id)
    .fetch_one(&mut *connection)
    .await?;
    let non_root_tasks: i64 = task_counts.get("total");
    let completed_tasks: i64 = task_counts.get("completed");
    let failed_or_blocked_tasks: i64 = task_counts.get("failed_or_blocked");

    let mutation_counts = sqlx::query(
        "SELECT COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END), 0)
                    AS succeeded,
                COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed,
                COALESCE(SUM(CASE WHEN status = 'never_dispatched' THEN 1 ELSE 0 END), 0)
                    AS never_dispatched,
                COALESCE(SUM(CASE WHEN status IN ('reserved', 'ambiguous') THEN 1 ELSE 0 END), 0)
                    AS ambiguous_or_reserved
         FROM mandate_mutation_attempts
         WHERE goal_run_id = ?",
    )
    .bind(goal_run_id)
    .fetch_one(&mut *connection)
    .await?;
    let mutation_total: i64 = mutation_counts.get("total");
    let mutation_succeeded: i64 = mutation_counts.get("succeeded");
    let mutation_failed: i64 = mutation_counts.get("failed");
    let mutation_never_dispatched: i64 = mutation_counts.get("never_dispatched");
    let mutation_ambiguous: i64 = mutation_counts.get("ambiguous_or_reserved");
    let count = |value: i64| u32::try_from(value.max(0)).unwrap_or(u32::MAX);

    Ok(MandateRunProofCounts {
        non_root_tasks: count(non_root_tasks),
        completed_tasks: count(completed_tasks),
        incomplete_tasks: count(non_root_tasks.saturating_sub(completed_tasks)),
        failed_or_blocked_tasks: count(failed_or_blocked_tasks),
        mutation_reservations: count(mutation_total),
        succeeded_mutations: count(mutation_succeeded),
        failed_mutations: count(mutation_failed),
        never_dispatched_mutations: count(mutation_never_dispatched),
        ambiguous_or_reserved_mutations: count(mutation_ambiguous),
    })
}

#[allow(clippy::too_many_arguments)]
async fn retry_orphaned_mandate_review_without_dispatch(
    connection: &mut sqlx::SqliteConnection,
    mandate_id: &str,
    goal_id: &str,
    mandate_version: i64,
    owner_session_id: &str,
    goal_run_id: &str,
    min_review_secs: i64,
    max_review_secs: i64,
    review_failures: i32,
    now: &str,
) -> anyhow::Result<()> {
    let reason = MandateFinalizationRejectReason::DecisionMissing;
    sqlx::query(
        "UPDATE task_attempts
         SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
         WHERE goal_run_id = ? AND status IN ('claimed', 'running')",
    )
    .bind(now)
    .bind(goal_run_id)
    .execute(&mut *connection)
    .await?;
    sqlx::query(
        "UPDATE tasks
         SET status = 'cancelled', current_attempt_id = NULL,
             completed_at = COALESCE(completed_at, ?), updated_at = ?, version = version + 1
         WHERE goal_run_id = ?
           AND status IN ('pending', 'claimed', 'running', 'blocked')",
    )
    .bind(now)
    .bind(now)
    .bind(goal_run_id)
    .execute(&mut *connection)
    .await?;
    let run_updated = sqlx::query(
        "UPDATE goal_runs
         SET status = 'failed', outcome_summary = ?, completed_at = ?, updated_at = ?
         WHERE id = ? AND status IN ('pending', 'running', 'blocked')
           AND trigger_type = 'mandate'",
    )
    .bind(format!("mandate_review_failed:{}", reason.as_str()))
    .bind(now)
    .bind(now)
    .bind(goal_run_id)
    .execute(&mut *connection)
    .await?;

    let finalized_at = chrono::DateTime::parse_from_rfc3339(now)
        .map_err(|error| anyhow::anyhow!("invalid orphaned review timestamp: {error}"))?
        .with_timezone(&chrono::Utc);
    let next_review_failures = review_failures.saturating_add(1);
    let backoff_shift = u32::try_from(next_review_failures.saturating_sub(1))
        .unwrap_or(0)
        .min(6);
    let retry_delay_secs = min_review_secs
        .saturating_mul(1_i64 << backoff_shift)
        .min(max_review_secs);
    let retry_at = (finalized_at + chrono::Duration::seconds(retry_delay_secs)).to_rfc3339();
    let mandate_updated = sqlx::query(
        "UPDATE mandates
         SET next_review_at = ?, review_lease_token = NULL,
             review_lease_expires_at = NULL, updated_at = ?
         WHERE id = ? AND goal_id = ? AND version = ? AND status = 'active'",
    )
    .bind(&retry_at)
    .bind(now)
    .bind(mandate_id)
    .bind(goal_id)
    .bind(mandate_version)
    .execute(&mut *connection)
    .await?;
    anyhow::ensure!(
        run_updated.rows_affected() == 1 && mandate_updated.rows_affected() == 1,
        "orphaned mandate review changed during no-dispatch recovery"
    );
    update_controller_status(connection, goal_id, MandateStatus::Active, now).await?;
    sqlx::query("UPDATE goals SET dispatch_failures = ?, updated_at = ? WHERE id = ?")
        .bind(next_review_failures)
        .bind(now)
        .bind(goal_id)
        .execute(&mut *connection)
        .await?;

    let notice = crate::traits::MandateRunNotification::new(
        mandate_id,
        mandate_version,
        goal_id,
        goal_run_id,
        owner_session_id,
        crate::traits::MandateRunNotificationKind::ReviewFailed { reason },
        MandateRunProofCounts::default(),
        now,
    );
    super::notifications::enqueue_mandate_run_notification_on_connection(&mut *connection, &notice)
        .await?;
    Ok(())
}

async fn reconcile_orphaned_mandate_runs(
    connection: &mut sqlx::SqliteConnection,
    now: &str,
) -> anyhow::Result<()> {
    let rows = sqlx::query(
        "SELECT m.id AS mandate_id, m.goal_id, m.version, g.session_id,
                g.dispatch_failures AS review_failures,
                m.min_review_secs, m.max_review_secs,
                gr.id AS goal_run_id, gr.root_task_id,
                (SELECT COUNT(*) FROM mandate_decision_cycles dc
                 WHERE dc.mandate_id = m.id AND dc.goal_run_id = gr.id)
                    AS decision_count,
                (SELECT COUNT(*) FROM mandate_mutation_attempts ma
                 WHERE ma.mandate_id = m.id AND ma.goal_run_id = gr.id)
                    AS mutation_reservation_count,
                (SELECT COUNT(*) FROM mandate_mutation_attempts ma
                 WHERE ma.mandate_id = m.id AND ma.goal_run_id = gr.id
                   AND ma.dispatch_claimed_at IS NOT NULL)
                    AS dispatch_claim_count
         FROM mandates m
         JOIN goals g ON g.id = m.goal_id
         JOIN goal_runs gr ON gr.goal_id = m.goal_id AND gr.trigger_type = 'mandate'
         JOIN tasks root ON root.id = gr.root_task_id
         WHERE m.status = 'active' AND m.confirmed_at IS NOT NULL
           AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
           AND g.status = 'active'
           AND gr.status IN ('pending', 'running', 'blocked')
           AND (
               root.status IN ('claimed', 'running', 'blocked', 'interrupted', 'failed', 'abandoned')
               OR (
                   root.status = 'completed'
                   AND root.completed_at IS NOT NULL
                   AND julianday(root.completed_at) <= julianday(?) - (60.0 / 86400.0)
               )
           )
           AND NOT EXISTS (
               SELECT 1 FROM task_attempts a
               WHERE a.id = root.current_attempt_id
                 AND a.task_id = root.id AND a.goal_run_id = gr.id
                 AND a.status IN ('claimed', 'running')
                 AND datetime(a.lease_expires_at) > datetime('now')
           )",
    )
    .bind(now)
    .bind(now)
    .fetch_all(&mut *connection)
    .await?;

    for row in rows {
        let mandate_id: String = row.get("mandate_id");
        let goal_id: String = row.get("goal_id");
        let version: i64 = row.get("version");
        let session_id: String = row.get("session_id");
        let review_failures: i32 = row.get("review_failures");
        let min_review_secs: i64 = row.get("min_review_secs");
        let max_review_secs: i64 = row.get("max_review_secs");
        let goal_run_id: String = row.get("goal_run_id");
        let root_task_id: Option<String> = row.get("root_task_id");
        let decision_count: i64 = row.get("decision_count");
        let mutation_reservation_count: i64 = row.get("mutation_reservation_count");
        let dispatch_claim_count: i64 = row.get("dispatch_claim_count");

        // A mandate mutation can only cross the external I/O boundary after
        // both a durable decision and a one-use dispatch claim exist. If the
        // worker died before any of those proofs were written, there is no
        // external effect to reconcile and pausing the mandate would turn a
        // recoverable review interruption into a false safety incident.
        if decision_count == 0 && mutation_reservation_count == 0 && dispatch_claim_count == 0 {
            retry_orphaned_mandate_review_without_dispatch(
                connection,
                &mandate_id,
                &goal_id,
                version,
                &session_id,
                &goal_run_id,
                min_review_secs,
                max_review_secs,
                review_failures,
                now,
            )
            .await?;
            continue;
        }

        let mut suspension = MandateSuspension::new(
            MandateSuspensionKind::ExecutionLeaseLost,
            Some("orphaned_mandate_run".to_string()),
        );
        suspension.goal_run_id = Some(goal_run_id.clone());
        let transitioned = sqlx::query(
            "UPDATE mandates
             SET status = 'awaiting_input', review_lease_token = NULL,
                 review_lease_expires_at = NULL, suspension_json = ?,
                 version = version + 1, updated_at = ?
             WHERE id = ? AND goal_id = ? AND status = 'active' AND version = ?",
        )
        .bind(serde_json::to_string(&suspension)?)
        .bind(now)
        .bind(&mandate_id)
        .bind(&goal_id)
        .bind(version)
        .execute(&mut *connection)
        .await?;
        if transitioned.rows_affected() != 1 {
            continue;
        }

        let claimed_unresolved = invalidate_open_mandate_runs(connection, &goal_id, now).await?;
        update_controller_status(connection, &goal_id, MandateStatus::AwaitingInput, now).await?;
        if claimed_unresolved == 0 {
            let counts = mandate_run_proof_counts_on_connection(
                connection,
                &goal_run_id,
                root_task_id.as_deref(),
            )
            .await?;
            let notice = crate::traits::MandateRunNotification::new(
                &mandate_id,
                version + 1,
                &goal_id,
                &goal_run_id,
                &session_id,
                crate::traits::MandateRunNotificationKind::ExecutionLeaseLost,
                counts,
                now,
            );
            super::notifications::enqueue_mandate_run_notification_on_connection(
                &mut *connection,
                &notice,
            )
            .await?;
        }
    }
    Ok(())
}

/// Quarantine active autopilot rows created before objective controls became a
/// mandatory policy invariant. We cannot invent a baseline, target, metric
/// source, or failure budget on the owner's behalf, so legacy rows fail closed
/// into an explicit recovery state before any review can be leased.
async fn quarantine_uncontrolled_autopilot_mandates(
    connection: &mut sqlx::SqliteConnection,
    now: &str,
) -> anyhow::Result<()> {
    let rows = sqlx::query(
        "SELECT m.id, m.goal_id, m.version, g.session_id
         FROM mandates m
         JOIN goals g ON g.id = m.goal_id
         WHERE m.status = 'active'
           AND m.autonomy_mode = 'autopilot'
           AND m.objective_control_json IS NULL",
    )
    .fetch_all(&mut *connection)
    .await?;

    for row in rows {
        let mandate_id: String = row.get("id");
        let goal_id: String = row.get("goal_id");
        let version: i64 = row.get("version");
        let session_id: String = row.get("session_id");
        let unresolved_mutations = invalidate_open_mandate_runs(connection, &goal_id, now).await?;
        let (kind, reason_code, message) = if unresolved_mutations > 0 {
            (
                MandateSuspensionKind::AuthorityRevokedWithUnresolvedMutation,
                "legacy_autopilot_missing_objective_control_with_unresolved_mutation",
                format!(
                    "Autopilot mandate {} was paused because its legacy policy lacks objective control and an earlier mutation has no terminal receipt. Reconcile the external target first, then configure an owner-approved baseline, target, measurement source/cadence, experiment window, and failure budget before resuming.",
                    mandate_id.chars().take(8).collect::<String>()
                ),
            )
        } else {
            (
                MandateSuspensionKind::ObjectiveControlRequired,
                "legacy_autopilot_missing_objective_control",
                format!(
                    "Autopilot mandate {} was paused because its legacy policy has no objective control. Configure an owner-approved baseline, target, measurement source/cadence, experiment window, and failure budget before resuming; no values were inferred automatically.",
                    mandate_id.chars().take(8).collect::<String>()
                ),
            )
        };
        let suspension = MandateSuspension::new(kind, Some(reason_code.to_string()));
        let transitioned = sqlx::query(
            "UPDATE mandates
             SET status = 'awaiting_input', suspension_json = ?,
                 review_lease_token = NULL, review_lease_expires_at = NULL,
                 version = version + 1, updated_at = ?
             WHERE id = ? AND goal_id = ? AND status = 'active'
               AND autonomy_mode = 'autopilot'
               AND objective_control_json IS NULL AND version = ?",
        )
        .bind(serde_json::to_string(&suspension)?)
        .bind(now)
        .bind(&mandate_id)
        .bind(&goal_id)
        .bind(version)
        .execute(&mut *connection)
        .await?;
        if transitioned.rows_affected() != 1 {
            continue;
        }
        update_controller_status(connection, &goal_id, MandateStatus::AwaitingInput, now).await?;
        sqlx::query(
            "INSERT OR IGNORE INTO notification_queue
                (id, goal_id, session_id, notification_type, priority, message,
                 created_at, delivered_at, attempts, expires_at, task_id, action_token)
             VALUES (?, ?, ?, 'mandate_objective_control_required', 'critical', ?, ?,
                     NULL, 0, NULL, NULL, NULL)",
        )
        .bind(format!(
            "mandate-objective-control-required:{}:{}",
            mandate_id,
            version + 1
        ))
        .bind(&goal_id)
        .bind(&session_id)
        .bind(message)
        .bind(now)
        .execute(&mut *connection)
        .await?;
    }
    Ok(())
}

/// Reconcile persisted authority invariants before any background dispatcher or
/// management surface can observe legacy rows. Waiting until the next due
/// lease leaves an unmeasurable autopilot mandate visibly `active` after a
/// restart, even though no safe controller run can execute it.
pub(super) async fn enforce_mandate_invariants_on_startup(
    pool: &sqlx::SqlitePool,
) -> anyhow::Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let mut transaction = pool.begin().await?;
    quarantine_uncontrolled_autopilot_mandates(&mut transaction, &now).await?;
    transaction.commit().await?;
    Ok(())
}

#[async_trait]
impl MandateStore for SqliteStateStore {
    async fn create_mandate_controller(
        &self,
        goal: &Goal,
        mandate: &Mandate,
    ) -> anyhow::Result<()> {
        validate_new_mandate(mandate)?;
        anyhow::ensure!(
            goal.id == mandate.goal_id,
            "mandate goal_id does not match controller"
        );
        anyhow::ensure!(
            goal.domain == "orchestration" && goal.goal_type == "continuous",
            "mandates require a continuous orchestration goal"
        );
        anyhow::ensure!(
            matches!(
                (
                    goal.status.as_str(),
                    mandate.status,
                    mandate.confirmed_at.is_some()
                ),
                ("active", MandateStatus::Active, true)
                    | ("pending_confirmation", MandateStatus::Paused, false)
                    | ("paused", MandateStatus::Paused, true)
            ),
            "a new mandate controller must carry confirmation exactly when its lifecycle requires it"
        );
        anyhow::ensure!(
            goal.session_id == mandate.created_by_session,
            "controller session must match the mandate owner session"
        );

        let progress_notes_json = goal
            .progress_notes
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let mut tx = self.pool.begin().await?;
        validate_source_goal(
            &mut tx,
            mandate.source_goal_id.as_deref(),
            &mandate.created_by_session,
        )
        .await?;

        let active_evergreen_count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM goals
             WHERE domain = 'orchestration' AND goal_type = 'continuous' AND status = 'active'",
        )
        .fetch_one(&mut *tx)
        .await?;
        anyhow::ensure!(
            active_evergreen_count < 10,
            "Cannot create mandate controller: hard cap of 10 active evergreen goals reached"
        );

        sqlx::query(
            "INSERT INTO goals (
                id, description, domain, goal_type, status, priority, conditions, context, resources,
                budget_per_check, budget_daily, tokens_used_today, tokens_used_day, last_useful_action,
                created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at,
                notification_attempts, dispatch_failures, progress_notes, source_episode_id,
                legacy_int_id, project_id
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       COALESCE((SELECT project_id FROM session_work_projects WHERE session_id = ?), 'default'))",
        )
        .bind(&goal.id)
        .bind(&goal.description)
        .bind(&goal.domain)
        .bind(&goal.goal_type)
        .bind(&goal.status)
        .bind(&goal.priority)
        .bind(&goal.conditions)
        .bind(&goal.context)
        .bind(&goal.resources)
        .bind(goal.budget_per_check)
        .bind(goal.budget_daily)
        .bind(goal.tokens_used_today)
        .bind(&goal.tokens_used_day)
        .bind(&goal.last_useful_action)
        .bind(&goal.created_at)
        .bind(&goal.updated_at)
        .bind(&goal.completed_at)
        .bind(&goal.parent_goal_id)
        .bind(&goal.session_id)
        .bind(&goal.notified_at)
        .bind(goal.notification_attempts)
        .bind(goal.dispatch_failures)
        .bind(&progress_notes_json)
        .bind(goal.source_episode_id)
        .bind(goal.legacy_int_id)
        .bind(&goal.session_id)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO work_channel_links
                (goal_id, channel_session_id, created_at, updated_at)
             VALUES (?, ?, datetime('now'), datetime('now'))",
        )
        .bind(&goal.id)
        .bind(&goal.session_id)
        .execute(&mut *tx)
        .await?;

        insert_mandate_row(&mut tx, mandate).await?;
        sqlx::query(
            "INSERT INTO mandate_principal_sessions
                (principal_id, session_id, linked_at, linked_by_session)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(principal_id, session_id) DO NOTHING",
        )
        .bind(&mandate.owner_principal_id)
        .bind(&mandate.created_by_session)
        .bind(&mandate.created_at)
        .bind(&mandate.created_by_session)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn get_mandate(&self, id: &str) -> anyhow::Result<Option<Mandate>> {
        let query = format!("SELECT {MANDATE_COLUMNS} FROM mandates WHERE id = ?");
        let row = sqlx::query(&query)
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(mandate_from_row).transpose()
    }

    async fn get_mandate_for_goal(&self, goal_id: &str) -> anyhow::Result<Option<Mandate>> {
        let query = format!("SELECT {MANDATE_COLUMNS} FROM mandates WHERE goal_id = ?");
        let row = sqlx::query(&query)
            .bind(goal_id)
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(mandate_from_row).transpose()
    }

    async fn list_mandates(
        &self,
        session_id: Option<&str>,
        include_terminal: bool,
    ) -> anyhow::Result<Vec<Mandate>> {
        let mut query = format!(
            "SELECT {} FROM mandates m JOIN goals g ON g.id = m.goal_id WHERE 1 = 1",
            qualified_columns("m", MANDATE_COLUMNS)
        );
        if session_id.is_some() {
            query.push_str(
                " AND (EXISTS (
                    SELECT 1 FROM mandate_principal_sessions ps
                    WHERE ps.principal_id = m.owner_principal_id AND ps.session_id = ?
                ) OR m.owner_principal_id = ?)",
            );
        }
        if !include_terminal {
            query.push_str(" AND m.status NOT IN ('completed', 'cancelled')");
        }
        query.push_str(" ORDER BY julianday(m.updated_at) DESC, m.id DESC");
        let rows = if let Some(session_id) = session_id {
            let stable_principal =
                crate::session::stable_private_owner_principal_id(session_id).unwrap_or_default();
            sqlx::query(&query)
                .bind(session_id)
                .bind(stable_principal)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query(&query).fetch_all(&self.pool).await?
        };
        rows.iter().map(mandate_from_row).collect()
    }

    async fn is_mandate_session_authorized(
        &self,
        mandate_id: &str,
        session_id: &str,
    ) -> anyhow::Result<bool> {
        let stable_principal =
            crate::session::stable_private_owner_principal_id(session_id).unwrap_or_default();
        Ok(sqlx::query_scalar::<_, i64>(
            "SELECT EXISTS (
                SELECT 1 FROM mandates m
                WHERE m.id = ? AND (
                    EXISTS (
                        SELECT 1 FROM mandate_principal_sessions ps
                        WHERE ps.principal_id = m.owner_principal_id
                          AND ps.session_id = ?
                    ) OR m.owner_principal_id = ?
                )
            )",
        )
        .bind(mandate_id)
        .bind(session_id)
        .bind(stable_principal)
        .fetch_one(&self.pool)
        .await?
            != 0)
    }

    async fn transfer_mandate_ownership(
        &self,
        mandate_id: &str,
        expected_version: i64,
        from_session_id: &str,
        to_session_id: &str,
    ) -> anyhow::Result<bool> {
        for (label, value) in [
            ("mandate id", mandate_id),
            ("source session", from_session_id),
            ("target session", to_session_id),
        ] {
            anyhow::ensure!(
                !value.trim().is_empty()
                    && value.trim() == value
                    && value.len() <= 256
                    && !value.chars().any(char::is_control),
                "invalid ownership transfer {label}"
            );
        }
        anyhow::ensure!(
            from_session_id != to_session_id,
            "ownership transfer target must differ from the source session"
        );
        anyhow::ensure!(expected_version > 0, "mandate version must be positive");
        let now = chrono::Utc::now().to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query(
            "SELECT m.goal_id, m.owner_principal_id
             FROM mandates m
             JOIN mandate_principal_sessions ps
               ON ps.principal_id = m.owner_principal_id
             WHERE m.id = ? AND m.version = ? AND ps.session_id = ?
               AND m.status NOT IN ('completed', 'cancelled')",
        )
        .bind(mandate_id)
        .bind(expected_version)
        .bind(from_session_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(row) = row else {
            tx.rollback().await?;
            return Ok(false);
        };
        let goal_id: String = row.get("goal_id");
        let principal_id: String = row.get("owner_principal_id");
        let claimed_unresolved = invalidate_open_mandate_runs(&mut tx, &goal_id, &now).await?;
        if claimed_unresolved > 0 {
            tx.rollback().await?;
            anyhow::bail!(
                "ownership cannot move while a dispatched mutation awaits reconciliation"
            );
        }
        sqlx::query(
            "INSERT INTO mandate_principal_sessions
                (principal_id, session_id, linked_at, linked_by_session)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(principal_id, session_id) DO NOTHING",
        )
        .bind(&principal_id)
        .bind(to_session_id)
        .bind(&now)
        .bind(from_session_id)
        .execute(&mut *tx)
        .await?;
        let updated = sqlx::query(
            "UPDATE mandates
             SET created_by_session = ?, version = version + 1,
                 review_lease_token = NULL, review_lease_expires_at = NULL,
                 updated_at = ?
             WHERE id = ? AND version = ?",
        )
        .bind(to_session_id)
        .bind(&now)
        .bind(mandate_id)
        .bind(expected_version)
        .execute(&mut *tx)
        .await?;
        anyhow::ensure!(
            updated.rows_affected() == 1,
            "mandate version changed during transfer"
        );
        sqlx::query("UPDATE goals SET session_id = ?, updated_at = ? WHERE id = ?")
            .bind(to_session_id)
            .bind(&now)
            .bind(&goal_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "INSERT INTO work_channel_links (goal_id, channel_session_id, created_at, updated_at)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(goal_id, channel_session_id) DO UPDATE SET updated_at = excluded.updated_at",
        )
        .bind(&goal_id)
        .bind(to_session_id)
        .bind(&now)
        .bind(&now)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE notification_queue SET session_id = ?
             WHERE goal_id = ? AND delivered_at IS NULL",
        )
        .bind(to_session_id)
        .bind(&goal_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "INSERT INTO mandate_ownership_transfers
                (id, mandate_id, principal_id, from_session_id, to_session_id,
                 from_version, to_version, transferred_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(mandate_id)
        .bind(&principal_id)
        .bind(from_session_id)
        .bind(to_session_id)
        .bind(expected_version)
        .bind(expected_version + 1)
        .bind(&now)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn record_mandate_objective_measurement(
        &self,
        measurement: &MandateObjectiveMeasurement,
    ) -> anyhow::Result<()> {
        measurement.validate().map_err(anyhow::Error::msg)?;
        let observed_at = chrono::DateTime::parse_from_rfc3339(&measurement.observed_at)?
            .with_timezone(&chrono::Utc);
        anyhow::ensure!(
            observed_at <= chrono::Utc::now() + chrono::Duration::minutes(5),
            "objective measurement cannot be future-dated"
        );
        let mut tx = self.pool.begin().await?;
        let run_started_at = sqlx::query_scalar::<_, Option<String>>(
            "SELECT gr.started_at
             FROM mandates m
             JOIN goal_runs gr ON gr.goal_id = m.goal_id
             WHERE m.id = ? AND m.version = ? AND m.objective_control_json IS NOT NULL
               AND m.status = 'active' AND m.confirmed_at IS NOT NULL
               AND gr.id = ? AND gr.trigger_type = 'mandate' AND gr.status = 'running'",
        )
        .bind(&measurement.mandate_id)
        .bind(measurement.mandate_version)
        .bind(&measurement.goal_run_id)
        .fetch_optional(&mut *tx)
        .await?
        .flatten()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "objective measurement is not bound to a current controlled mandate run"
            )
        })?;
        let run_started_at = chrono::DateTime::parse_from_rfc3339(&run_started_at)
            .or_else(|_| chrono::DateTime::parse_from_str(&run_started_at, "%Y-%m-%d %H:%M:%S%#z"))?
            .with_timezone(&chrono::Utc);
        anyhow::ensure!(
            observed_at >= run_started_at - chrono::Duration::minutes(5),
            "objective measurement predates the current mandate run"
        );
        validate_current_run_receipt_refs(
            &mut tx,
            &measurement.goal_run_id,
            &measurement.evidence_receipt_ids,
        )
        .await?;
        for intention_id in &measurement.attributed_intention_ids {
            let exists = sqlx::query_scalar::<_, i64>(
                "SELECT 1 FROM intentions
                 WHERE id = ? AND mandate_id = ? AND goal_run_id <> ? LIMIT 1",
            )
            .bind(intention_id)
            .bind(&measurement.mandate_id)
            .bind(&measurement.goal_run_id)
            .fetch_optional(&mut *tx)
            .await?;
            anyhow::ensure!(
                exists.is_some(),
                "objective measurement attribution must reference a prior intention from the same mandate"
            );
        }
        sqlx::query(
            "INSERT INTO mandate_objective_measurements
                (id, mandate_id, mandate_version, goal_run_id, value_micros,
                 confidence_bps, evidence_receipt_ids_json, attributed_intention_ids_json,
                 observed_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&measurement.id)
        .bind(&measurement.mandate_id)
        .bind(measurement.mandate_version)
        .bind(&measurement.goal_run_id)
        .bind(measurement.value_micros)
        .bind(i64::from(measurement.confidence_bps))
        .bind(serde_json::to_string(&measurement.evidence_receipt_ids)?)
        .bind(serde_json::to_string(
            &measurement.attributed_intention_ids,
        )?)
        .bind(&measurement.observed_at)
        .bind(&measurement.created_at)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn list_mandate_objective_measurements(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<MandateObjectiveMeasurement>> {
        let rows = sqlx::query(
            "SELECT id, mandate_id, mandate_version, goal_run_id, value_micros,
                    confidence_bps, evidence_receipt_ids_json,
                    attributed_intention_ids_json, observed_at, created_at
             FROM mandate_objective_measurements
             WHERE mandate_id = ?
             ORDER BY julianday(observed_at) DESC, id DESC LIMIT ?",
        )
        .bind(mandate_id)
        .bind(limit.clamp(1, 200))
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(objective_measurement_from_row).collect()
    }

    async fn update_mandate(&self, mandate: &Mandate) -> anyhow::Result<()> {
        validate_mandate(mandate)?;
        let expected_version = mandate
            .version
            .checked_sub(1)
            .filter(|version| *version > 0)
            .ok_or_else(|| anyhow::anyhow!("updated mandate version must be at least 2"))?;
        let now = chrono::Utc::now().to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let result = sqlx::query(
            "UPDATE mandates
             SET objective = ?, autonomy_mode = ?, authority_json = ?, strategy_json = ?, objective_control_json = ?, constraints_json = ?,
                 success_criteria_json = ?, stop_conditions_json = ?, min_review_secs = ?,
                 max_review_secs = ?, default_review_secs = ?, review_effort = ?, next_review_at = ?, expires_at = ?,
                 version = ?, review_lease_token = NULL, review_lease_expires_at = NULL,
                 updated_at = ?
             WHERE id = ? AND goal_id = ? AND version = ? AND status = ?",
        )
        .bind(&mandate.objective)
        .bind(mandate.autonomy_mode.as_str())
        .bind(serde_json::to_string(&mandate.authority)?)
        .bind(
            mandate
                .strategy
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?,
        )
        .bind(
            mandate
                .objective_control
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?,
        )
        .bind(serde_json::to_string(&mandate.constraints)?)
        .bind(serde_json::to_string(&mandate.success_criteria)?)
        .bind(serde_json::to_string(&mandate.stop_conditions)?)
        .bind(mandate.min_review_secs)
        .bind(mandate.max_review_secs)
        .bind(mandate.default_review_secs)
        .bind(&mandate.review_effort)
        .bind(&mandate.next_review_at)
        .bind(&mandate.expires_at)
        .bind(mandate.version)
        .bind(&now)
        .bind(&mandate.id)
        .bind(&mandate.goal_id)
        .bind(expected_version)
        .bind(mandate.status.as_str())
        .execute(&mut *tx)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "mandate version conflict or mandate not found"
        );
        // Updating policy is an authority-epoch change. Invalidate every run,
        // task, and attempt created under the previous immutable version in
        // the same transaction as the version CAS.
        let claimed_unresolved =
            invalidate_open_mandate_runs(&mut tx, &mandate.goal_id, &now).await?;
        if claimed_unresolved > 0 {
            // A policy edit cannot make a claimed-but-unproved external effect
            // safe. Land the new policy, but pause its controller until the
            // owner reconciles the prior target; otherwise the updated active
            // mandate could immediately begin a duplicate cycle.
            let suspension = MandateSuspension::new(
                MandateSuspensionKind::AuthorityRevokedWithUnresolvedMutation,
                Some("policy_update_invalidated_dispatched_mutation".to_string()),
            );
            let paused = sqlx::query(
                "UPDATE mandates
                 SET status = 'awaiting_input', review_lease_token = NULL,
                     review_lease_expires_at = NULL, suspension_json = ?, updated_at = ?
                 WHERE id = ? AND version = ? AND status = 'active'",
            )
            .bind(serde_json::to_string(&suspension)?)
            .bind(&now)
            .bind(&mandate.id)
            .bind(mandate.version)
            .execute(&mut *tx)
            .await?;
            anyhow::ensure!(
                paused.rows_affected() == 1,
                "claimed mutation invalidation requires an awaiting-input mandate"
            );
            update_controller_status(
                &mut tx,
                &mandate.goal_id,
                MandateStatus::AwaitingInput,
                &now,
            )
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn confirm_mandate(
        &self,
        mandate_id: &str,
        expected_version: i64,
        activation_duration_secs: Option<i64>,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(!mandate_id.trim().is_empty(), "mandate id is required");
        anyhow::ensure!(expected_version > 0, "mandate version must be positive");
        if let Some(duration_secs) = activation_duration_secs {
            anyhow::ensure!(
                duration_secs > 0,
                "mandate activation duration must be greater than zero"
            );
        }
        let activated_at = chrono::Utc::now();
        let expires_at = activation_duration_secs
            .map(|duration_secs| {
                activated_at
                    .checked_add_signed(chrono::Duration::seconds(duration_secs))
                    .ok_or_else(|| anyhow::anyhow!("mandate activation duration is too large"))
                    .map(|value| value.to_rfc3339())
            })
            .transpose()?;
        let now = activated_at.to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let goal_id = sqlx::query_scalar::<_, String>(
            "UPDATE mandates
             SET status = 'active', confirmed_at = ?, next_review_at = ?,
                 expires_at = COALESCE(?, expires_at),
                 review_lease_token = NULL, review_lease_expires_at = NULL, suspension_json = NULL,
                 version = version + 1, updated_at = ?
             WHERE id = ? AND version = ? AND status = 'paused' AND confirmed_at IS NULL
               AND EXISTS (
                   SELECT 1 FROM goals g
                   WHERE g.id = mandates.goal_id
                     AND g.domain = 'orchestration'
                     AND g.goal_type = 'continuous'
                     AND g.status = 'pending_confirmation'
               )
             RETURNING goal_id",
        )
        .bind(&now)
        .bind(&now)
        .bind(expires_at.as_deref())
        .bind(&now)
        .bind(mandate_id)
        .bind(expected_version)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(goal_id) = goal_id else {
            tx.rollback().await?;
            return Ok(false);
        };
        update_controller_status(&mut tx, &goal_id, MandateStatus::Active, &now).await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn transition_mandate_status(
        &self,
        mandate_id: &str,
        from_status: MandateStatus,
        to_status: MandateStatus,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            from_status != to_status,
            "mandate status transition is a no-op"
        );
        // Keep lifecycle evolution explicit. Terminal mandates are immutable,
        // and only owner-mediated resume paths may leave paused/ASK states.
        // Invalid-but-known transitions report a failed CAS rather than an
        // error so callers cannot distinguish them from a concurrent change.
        if !from_status.can_transition_to(to_status) {
            return Ok(false);
        }
        let now = chrono::Utc::now().to_rfc3339();
        let suspension = match to_status {
            MandateStatus::Paused => Some(MandateSuspension::new(
                MandateSuspensionKind::OwnerPaused,
                None,
            )),
            MandateStatus::AwaitingInput => Some(MandateSuspension::new(
                MandateSuspensionKind::ReviewFailed,
                Some("lifecycle_transition".to_string()),
            )),
            MandateStatus::Active | MandateStatus::Completed | MandateStatus::Cancelled => None,
        };
        let mut tx = self.pool.begin().await?;
        let goal_id = sqlx::query_scalar::<_, String>(
            "UPDATE mandates
             SET status = ?,
                 next_review_at = CASE WHEN ? = 'active' THEN ? ELSE next_review_at END,
                 review_lease_token = NULL, review_lease_expires_at = NULL,
                 suspension_json = ?,
                 version = version + 1, updated_at = ?
             WHERE id = ? AND status = ?
               AND (? != 'active' OR confirmed_at IS NOT NULL)
             RETURNING goal_id",
        )
        .bind(to_status.as_str())
        .bind(to_status.as_str())
        .bind(&now)
        .bind(suspension.as_ref().map(serde_json::to_string).transpose()?)
        .bind(&now)
        .bind(mandate_id)
        .bind(from_status.as_str())
        .bind(to_status.as_str())
        .fetch_optional(&mut *tx)
        .await?;
        let Some(goal_id) = goal_id else {
            tx.rollback().await?;
            return Ok(false);
        };

        // A lifecycle transition is an authority-epoch change. Close every
        // open mandate run and its work in the same transaction so pausing and
        // later resuming cannot resurrect a pre-pause ACT or attempt.
        let claimed_unresolved = invalidate_open_mandate_runs(&mut tx, &goal_id, &now).await?;
        let effective_status = if to_status == MandateStatus::Paused && claimed_unresolved > 0 {
            let suspension = MandateSuspension::new(
                MandateSuspensionKind::ReconciliationRequired,
                Some("lifecycle_invalidated_after_dispatch_claim".to_string()),
            );
            let changed = sqlx::query(
                "UPDATE mandates SET status = 'awaiting_input', suspension_json = ?, updated_at = ?
                 WHERE id = ? AND status = 'paused'",
            )
            .bind(serde_json::to_string(&suspension)?)
            .bind(&now)
            .bind(mandate_id)
            .execute(&mut *tx)
            .await?;
            anyhow::ensure!(
                changed.rows_affected() == 1,
                "claimed mutation requires typed reconciliation"
            );
            MandateStatus::AwaitingInput
        } else {
            to_status
        };
        update_controller_status(&mut tx, &goal_id, effective_status, &now).await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn resume_mandate_with_context(
        &self,
        mandate_id: &str,
        from_status: MandateStatus,
        expected_version: i64,
        controller_context: Option<&str>,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            from_status == MandateStatus::Paused
                && from_status.can_transition_to(MandateStatus::Active),
            "ordinary resume is only valid for owner-paused mandates"
        );
        anyhow::ensure!(expected_version > 0, "mandate version must be positive");
        if let Some(context) = controller_context {
            anyhow::ensure!(
                context.chars().count() <= 64_000,
                "mandate controller context is too large"
            );
            serde_json::from_str::<serde_json::Value>(context)?;
        }
        let now = chrono::Utc::now().to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let goal_id = sqlx::query_scalar::<_, String>(
            "UPDATE mandates
             SET status = 'active', next_review_at = ?,
                 review_lease_token = NULL, review_lease_expires_at = NULL,
                 suspension_json = NULL,
                 version = version + 1, updated_at = ?
             WHERE id = ? AND status = ? AND version = ?
               AND confirmed_at IS NOT NULL
               AND json_extract(suspension_json, '$.kind') = 'owner_paused'
             RETURNING goal_id",
        )
        .bind(&now)
        .bind(&now)
        .bind(mandate_id)
        .bind(from_status.as_str())
        .bind(expected_version)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(goal_id) = goal_id else {
            tx.rollback().await?;
            return Ok(false);
        };
        invalidate_open_mandate_runs(&mut tx, &goal_id, &now).await?;
        let controller = sqlx::query(
            "UPDATE goals
             SET status = 'active', context = COALESCE(?, context),
                 completed_at = NULL, dispatch_failures = 0, updated_at = ?
             WHERE id = ? AND domain = 'orchestration' AND goal_type = 'continuous'",
        )
        .bind(controller_context)
        .bind(&now)
        .bind(&goal_id)
        .execute(&mut *tx)
        .await?;
        anyhow::ensure!(
            controller.rows_affected() == 1,
            "mandate controller goal not found"
        );
        tx.commit().await?;
        Ok(true)
    }

    async fn create_mandate_review_run(
        &self,
        mandate_id: &str,
        review_lease_token: &str,
        goal_run_id: &str,
        root_task: &Task,
    ) -> anyhow::Result<GoalRun> {
        anyhow::ensure!(!mandate_id.trim().is_empty(), "mandate id is required");
        anyhow::ensure!(
            !review_lease_token.trim().is_empty(),
            "review lease token is required"
        );
        anyhow::ensure!(!goal_run_id.trim().is_empty(), "goal run id is required");
        anyhow::ensure!(!root_task.id.trim().is_empty(), "root task id is required");
        anyhow::ensure!(
            !root_task.description.trim().is_empty(),
            "root task description is required"
        );
        anyhow::ensure!(
            root_task.status == "pending"
                && root_task.started_at.is_none()
                && root_task.completed_at.is_none(),
            "a mandate review root task must start pending and unstarted"
        );

        let now = chrono::Utc::now();
        let now_string = now.to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query(
            "SELECT m.goal_id, g.project_id
             FROM mandates m
             JOIN goals g ON g.id = m.goal_id
             WHERE m.id = ?
               AND m.status = 'active'
               AND m.confirmed_at IS NOT NULL
               AND m.review_lease_token = ?
               AND m.review_lease_expires_at IS NOT NULL
               AND julianday(m.review_lease_expires_at) > julianday(?)
               AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
               AND g.domain = 'orchestration'
               AND g.goal_type = 'continuous'
               AND g.status = 'active'",
        )
        .bind(mandate_id)
        .bind(review_lease_token)
        .bind(&now_string)
        .bind(&now_string)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!("mandate is not confirmed, active, or leased by this dispatcher")
        })?;
        let goal_id: String = row.get("goal_id");
        let project_id: String = row.get("project_id");
        anyhow::ensure!(
            root_task.goal_id == goal_id,
            "root task does not belong to the mandate controller"
        );

        let open_run_count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM goal_runs
             WHERE goal_id = ? AND status IN ('pending', 'running', 'blocked')",
        )
        .bind(&goal_id)
        .fetch_one(&mut *tx)
        .await?;
        anyhow::ensure!(
            open_run_count == 0,
            "mandate controller already has an open goal run"
        );

        let mut run = GoalRun::new(&goal_id, &project_id, "mandate");
        run.id = goal_run_id.to_string();
        run.root_task_id = Some(root_task.id.clone());
        sqlx::query(
            "INSERT INTO goal_runs
                (id, project_id, goal_id, trigger_type, schedule_id, root_task_id,
                 status, outcome_summary, started_at, completed_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&run.id)
        .bind(&run.project_id)
        .bind(&run.goal_id)
        .bind(&run.trigger_type)
        .bind(&run.schedule_id)
        .bind(&run.root_task_id)
        .bind(&run.status)
        .bind(&run.outcome_summary)
        .bind(&run.started_at)
        .bind(&run.completed_at)
        .bind(&run.created_at)
        .bind(&run.updated_at)
        .execute(&mut *tx)
        .await?;

        let result = root_task
            .result
            .as_deref()
            .filter(|value| !value.trim().is_empty());
        let error = root_task
            .error
            .as_deref()
            .filter(|value| !value.trim().is_empty());
        let blocker = root_task
            .blocker
            .as_deref()
            .filter(|value| !value.trim().is_empty());
        let root_insert = sqlx::query(
            "INSERT INTO tasks (
                id, goal_id, goal_run_id, description, status, priority, task_order,
                parallel_group, depends_on, agent_id, context, result, error, blocker,
                idempotent, retry_count, max_retries, created_at, started_at, completed_at,
                updated_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&root_task.id)
        .bind(&root_task.goal_id)
        .bind(&run.id)
        .bind(&root_task.description)
        .bind(&root_task.status)
        .bind(&root_task.priority)
        .bind(root_task.task_order)
        .bind(&root_task.parallel_group)
        .bind(&root_task.depends_on)
        .bind(&root_task.agent_id)
        .bind(&root_task.context)
        .bind(result)
        .bind(error)
        .bind(blocker)
        .bind(root_task.idempotent as i32)
        .bind(root_task.retry_count)
        .bind(root_task.max_retries)
        .bind(&root_task.created_at)
        .bind(&root_task.started_at)
        .bind(&root_task.completed_at)
        .bind(&root_task.created_at)
        .execute(&mut *tx)
        .await;
        if let Err(insert_error) = root_insert {
            // A failed statement leaves SQLite's write transaction active until
            // rollback completes. Await it here so an immediate recovery query
            // cannot race the drop-triggered asynchronous rollback and observe
            // SQLITE_BUSY under load.
            if let Err(rollback_error) = tx.rollback().await {
                return Err(anyhow::anyhow!(
                    "mandate review root insert failed ({insert_error}); rollback failed ({rollback_error})"
                ));
            }
            return Err(insert_error.into());
        }

        tx.commit().await?;
        Ok(run)
    }

    async fn keep_mandate_controller_active(
        &self,
        mandate_id: &str,
        expected_version: i64,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(expected_version > 0, "mandate version must be positive");
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE goals
             SET status = 'active', completed_at = NULL, dispatch_failures = 0, updated_at = ?
             WHERE id = (
                 SELECT goal_id FROM mandates
                 WHERE id = ? AND status = 'active' AND confirmed_at IS NOT NULL
                   AND version = ?
             )
               AND domain = 'orchestration' AND goal_type = 'continuous'",
        )
        .bind(&now)
        .bind(mandate_id)
        .bind(expected_version)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    async fn claim_due_mandates(
        &self,
        limit: i64,
        lease_owner: &str,
        lease_secs: i64,
    ) -> anyhow::Result<Vec<Mandate>> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        anyhow::ensure!(!lease_owner.trim().is_empty(), "lease owner is required");
        anyhow::ensure!(
            (1..=86_400).contains(&lease_secs),
            "review lease duration must be between 1 second and 24 hours"
        );
        let now = chrono::Utc::now();
        let now_string = now.to_rfc3339();
        let lease_expires_at = (now + chrono::Duration::seconds(lease_secs)).to_rfc3339();
        let lease_token = format!("{}:{}", lease_owner.trim(), uuid::Uuid::new_v4());
        let mut tx = self.pool.begin().await?;

        // Recover the only crash state produced by the pre-atomic dispatcher:
        // an open mandate run whose root task was never inserted. No work can
        // have executed in such a run, so close it and make the mandate due in
        // the same transaction before selecting claims.
        sqlx::query(
            "UPDATE mandates
             SET next_review_at = ?, review_lease_token = NULL,
                 review_lease_expires_at = NULL, updated_at = ?
             WHERE status = 'active' AND confirmed_at IS NOT NULL
               AND EXISTS (
                   SELECT 1 FROM goal_runs gr
                   WHERE gr.goal_id = mandates.goal_id
                     AND gr.trigger_type = 'mandate'
                     AND gr.status IN ('pending', 'running', 'blocked')
                     AND NOT EXISTS (
                         SELECT 1 FROM tasks t WHERE t.goal_run_id = gr.id
                     )
               )",
        )
        .bind(&now_string)
        .bind(&now_string)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE goal_runs
             SET status = 'failed',
                 outcome_summary = 'Recovered an interrupted mandate dispatch before its root task was created.',
                 completed_at = COALESCE(completed_at, ?), updated_at = ?
             WHERE trigger_type = 'mandate'
               AND status IN ('pending', 'running', 'blocked')
               AND NOT EXISTS (
                   SELECT 1 FROM tasks t WHERE t.goal_run_id = goal_runs.id
               )",
        )
        .bind(&now_string)
        .bind(&now_string)
        .execute(&mut *tx)
        .await?;

        // A root whose execution lease vanished may have performed an
        // externally visible effect before crashing. Never auto-retry that
        // ambiguous cycle: atomically pause it for owner reconciliation.
        reconcile_orphaned_mandate_runs(&mut tx, &now_string).await?;

        // Older rows can predate the objective-control invariant enforced on
        // every new/update path. Reconcile them at the authoritative lease
        // boundary so a restart cannot silently keep an unmeasurable autopilot
        // objective active.
        quarantine_uncontrolled_autopilot_mandates(&mut tx, &now_string).await?;

        // Expiry is a hard authority boundary, not merely a query filter. Keep
        // the visible lifecycle coherent so an expired mandate cannot remain
        // labelled active indefinitely with an active backing controller. It
        // revokes attempts/runs/intentions in the same transaction, exactly as
        // an explicit lifecycle transition does.
        let expired_goal_ids = sqlx::query_scalar::<_, String>(
            "SELECT goal_id FROM mandates
             WHERE status = 'active' AND expires_at IS NOT NULL
               AND julianday(expires_at) <= julianday(?)",
        )
        .bind(&now_string)
        .fetch_all(&mut *tx)
        .await?;
        for goal_id in &expired_goal_ids {
            invalidate_open_mandate_runs(&mut tx, goal_id, &now_string).await?;
        }
        sqlx::query(
            "UPDATE goals
             SET status = 'completed', completed_at = COALESCE(completed_at, ?), updated_at = ?
             WHERE id IN (
                 SELECT goal_id FROM mandates
                 WHERE status = 'active' AND expires_at IS NOT NULL
                   AND julianday(expires_at) <= julianday(?)
             )",
        )
        .bind(&now_string)
        .bind(&now_string)
        .bind(&now_string)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE mandates
             SET status = 'completed', review_lease_token = NULL,
                 review_lease_expires_at = NULL, version = version + 1, updated_at = ?
             WHERE status = 'active' AND expires_at IS NOT NULL
               AND julianday(expires_at) <= julianday(?)",
        )
        .bind(&now_string)
        .bind(&now_string)
        .execute(&mut *tx)
        .await?;
        let query = format!(
            "UPDATE mandates
             SET review_lease_token = ?, review_lease_expires_at = ?, updated_at = ?
             WHERE id IN (
                 SELECT m.id
                 FROM mandates m
                 JOIN goals g ON g.id = m.goal_id
                 WHERE m.status = 'active'
                   AND m.confirmed_at IS NOT NULL
                   AND g.domain = 'orchestration'
                   AND g.goal_type = 'continuous'
                   AND g.status = 'active'
                   AND julianday(m.next_review_at) <= julianday(?)
                   AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
                   AND (
                       m.review_lease_token IS NULL
                       OR m.review_lease_expires_at IS NULL
                       OR julianday(m.review_lease_expires_at) <= julianday(?)
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM goal_runs gr
                       WHERE gr.goal_id = m.goal_id
                         AND gr.status IN ('pending', 'running', 'blocked')
                   )
                 ORDER BY julianday(m.next_review_at), m.id
                 LIMIT ?
             )
               AND status = 'active'
               AND confirmed_at IS NOT NULL
               AND (
                   review_lease_token IS NULL
                   OR review_lease_expires_at IS NULL
                   OR julianday(review_lease_expires_at) <= julianday(?)
               )
             RETURNING {MANDATE_COLUMNS}"
        );
        let rows = sqlx::query(&query)
            .bind(&lease_token)
            .bind(&lease_expires_at)
            .bind(&now_string)
            .bind(&now_string)
            .bind(&now_string)
            .bind(&now_string)
            .bind(limit.min(100))
            .bind(&now_string)
            .fetch_all(&mut *tx)
            .await?;
        tx.commit().await?;
        let mut mandates = rows
            .iter()
            .map(mandate_from_row)
            .collect::<anyhow::Result<Vec<_>>>()?;
        mandates.sort_by(|left, right| {
            left.next_review_at
                .cmp(&right.next_review_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(mandates)
    }

    async fn release_mandate_review_lease(
        &self,
        mandate_id: &str,
        lease_token: &str,
        retry_at: &str,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            !lease_token.trim().is_empty(),
            "review lease token is required"
        );
        validate_timestamp("retry_at", retry_at)?;
        let result = sqlx::query(
            "UPDATE mandates
             SET next_review_at = ?, review_lease_token = NULL,
                 review_lease_expires_at = NULL, updated_at = ?
             WHERE id = ? AND review_lease_token = ?",
        )
        .bind(retry_at)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(mandate_id)
        .bind(lease_token)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    async fn create_mandate_task_from_attempt(
        &self,
        task: &Task,
        mandate_id: &str,
        mandate_version: i64,
        goal_run_id: &str,
        root_task_attempt_id: &str,
        max_non_root_tasks: i64,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(mandate_version > 0, "mandate version must be positive");
        anyhow::ensure!(
            (1..=64).contains(&max_non_root_tasks),
            "mandate task cap must be between 1 and 64"
        );
        anyhow::ensure!(task.status == "pending", "mandate tasks must start pending");
        anyhow::ensure!(
            !task.idempotent && task.retry_count == 0 && task.max_retries == 0,
            "mandate tasks cannot be replayable"
        );
        anyhow::ensure!(
            !task.id.trim().is_empty()
                && !task.goal_id.trim().is_empty()
                && !task.description.trim().is_empty()
                && task.description.trim() == task.description
                && task.description.chars().count() <= 8_000,
            "mandate task identity and description must be canonical"
        );
        anyhow::ensure!(
            task.agent_id.is_none()
                && task.result.is_none()
                && task.error.is_none()
                && task.blocker.is_none()
                && task.started_at.is_none()
                && task.completed_at.is_none(),
            "new mandate tasks cannot carry execution state"
        );

        let now = chrono::Utc::now().to_rfc3339();
        let mut tx = self.pool.begin().await?;
        // First write: acquire the writer lock and prove the complete authority
        // chain in one CAS. A pause, policy revision, run replacement, or root
        // attempt replacement either linearizes before this statement (and
        // rejects it) or after this transaction commits.
        let fenced = sqlx::query(
            "UPDATE task_attempts
             SET last_heartbeat_at = datetime('now')
             WHERE id = ? AND goal_run_id = ?
               AND status IN ('claimed', 'running')
               AND datetime(lease_expires_at) > datetime('now')
               AND EXISTS (
                   SELECT 1
                   FROM tasks root
                   JOIN goal_runs gr ON gr.id = task_attempts.goal_run_id
                   JOIN mandates m ON m.goal_id = gr.goal_id
                   JOIN goals g ON g.id = m.goal_id
                   JOIN mandate_decision_cycles dc ON dc.goal_run_id = gr.id
                   JOIN intentions i ON i.decision_cycle_id = dc.id
                   WHERE root.id = task_attempts.task_id
                     AND root.current_attempt_id = task_attempts.id
                     AND root.status IN ('claimed', 'running')
                     AND root.error IS NULL AND root.blocker IS NULL
                     AND gr.id = ? AND gr.root_task_id = root.id
                     AND gr.trigger_type = 'mandate' AND gr.status = 'running'
                     AND m.id = ? AND m.goal_id = ? AND m.version = ?
                     AND m.status = 'active' AND m.confirmed_at IS NOT NULL
                     AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
                     AND g.status = 'active' AND g.domain = 'orchestration'
                     AND g.goal_type = 'continuous'
                     AND dc.mandate_id = m.id AND dc.mandate_version = m.version
                     AND dc.outcome = 'act'
                     AND i.mandate_id = m.id AND i.goal_run_id = gr.id
                     AND i.status = 'committed'
               )",
        )
        .bind(root_task_attempt_id)
        .bind(goal_run_id)
        .bind(goal_run_id)
        .bind(mandate_id)
        .bind(&task.goal_id)
        .bind(mandate_version)
        .bind(&now)
        .execute(&mut *tx)
        .await?;
        if fenced.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(false);
        }

        let task_count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM tasks
             WHERE goal_run_id = ?
               AND id != (SELECT root_task_id FROM goal_runs WHERE id = ?)",
        )
        .bind(goal_run_id)
        .bind(goal_run_id)
        .fetch_one(&mut *tx)
        .await?;
        if task_count >= max_non_root_tasks {
            tx.rollback().await?;
            return Ok(false);
        }

        if let Some(raw_dependencies) = task.depends_on.as_deref() {
            let dependencies = serde_json::from_str::<Vec<String>>(raw_dependencies)?;
            let unique = dependencies
                .iter()
                .collect::<std::collections::HashSet<_>>();
            anyhow::ensure!(
                unique.len() == dependencies.len(),
                "mandate task dependencies must be unique"
            );
            for dependency in dependencies {
                let in_run = sqlx::query_scalar::<_, i64>(
                    "SELECT COUNT(*) FROM tasks WHERE id = ? AND goal_run_id = ?",
                )
                .bind(&dependency)
                .bind(goal_run_id)
                .fetch_one(&mut *tx)
                .await?;
                anyhow::ensure!(in_run == 1, "mandate task dependency is outside this run");
            }
        }

        let inserted = sqlx::query(
            "INSERT INTO tasks (
                id, goal_id, goal_run_id, description, status, priority, task_order,
                parallel_group, depends_on, agent_id, context, result, error, blocker,
                idempotent, retry_count, max_retries, created_at, started_at, completed_at,
                updated_at
             ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL,
                       0, 0, 0, ?, NULL, NULL, ?)",
        )
        .bind(&task.id)
        .bind(&task.goal_id)
        .bind(goal_run_id)
        .bind(&task.description)
        .bind(&task.priority)
        .bind(task.task_order)
        .bind(&task.parallel_group)
        .bind(&task.depends_on)
        .bind(&task.created_at)
        .bind(&task.created_at)
        .execute(&mut *tx)
        .await?;
        anyhow::ensure!(
            inserted.rows_affected() == 1,
            "mandate task was not inserted"
        );
        tx.commit().await?;
        Ok(true)
    }

    async fn claim_mandate_task_from_attempt(
        &self,
        task_id: &str,
        worker_instance_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        goal_run_id: &str,
        root_task_attempt_id: &str,
        lease_secs: i64,
    ) -> anyhow::Result<Option<crate::traits::TaskAttempt>> {
        anyhow::ensure!(mandate_version > 0, "mandate version must be positive");
        anyhow::ensure!(
            !task_id.trim().is_empty()
                && !worker_instance_id.trim().is_empty()
                && !root_task_attempt_id.trim().is_empty(),
            "mandate claim identity is required"
        );
        let lease_secs = lease_secs.clamp(1, 86_400);
        let now = chrono::Utc::now();
        let now_string = now.to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let fenced = sqlx::query(
            "UPDATE task_attempts
             SET last_heartbeat_at = datetime('now')
             WHERE id = ? AND goal_run_id = ?
               AND status IN ('claimed', 'running')
               AND datetime(lease_expires_at) > datetime('now')
               AND EXISTS (
                   SELECT 1
                   FROM tasks root
                   JOIN goal_runs gr ON gr.id = task_attempts.goal_run_id
                   JOIN mandates m ON m.goal_id = gr.goal_id
                   JOIN goals g ON g.id = m.goal_id
                   JOIN mandate_decision_cycles dc ON dc.goal_run_id = gr.id
                   JOIN intentions i ON i.decision_cycle_id = dc.id
                   WHERE root.id = task_attempts.task_id
                     AND root.current_attempt_id = task_attempts.id
                     AND root.status IN ('claimed', 'running')
                     AND root.error IS NULL AND root.blocker IS NULL
                     AND gr.id = ? AND gr.root_task_id = root.id
                     AND gr.trigger_type = 'mandate' AND gr.status = 'running'
                     AND m.id = ?
                     AND m.goal_id = (SELECT goal_id FROM tasks WHERE id = ?)
                     AND m.version = ?
                     AND m.status = 'active' AND m.confirmed_at IS NOT NULL
                     AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
                     AND g.status = 'active' AND g.domain = 'orchestration'
                     AND g.goal_type = 'continuous'
                     AND dc.mandate_id = m.id AND dc.mandate_version = m.version
                     AND dc.outcome = 'act'
                     AND i.mandate_id = m.id AND i.goal_run_id = gr.id
                     AND i.status = 'committed'
               )",
        )
        .bind(root_task_attempt_id)
        .bind(goal_run_id)
        .bind(goal_run_id)
        .bind(mandate_id)
        .bind(task_id)
        .bind(mandate_version)
        .bind(&now_string)
        .execute(&mut *tx)
        .await?;
        if fenced.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }

        let claimable = sqlx::query_scalar::<_, String>(
            "SELECT t.goal_id
             FROM tasks t JOIN goal_runs gr ON gr.id = t.goal_run_id
             WHERE t.id = ? AND t.goal_run_id = ? AND t.id != gr.root_task_id
               AND t.status = 'pending' AND t.current_attempt_id IS NULL
               AND t.worker_profile_id IS NULL
               AND COALESCE(t.workspace_policy_explicit, 0) = 0
               AND NOT EXISTS (SELECT 1 FROM task_workspaces w WHERE w.task_id = t.id)",
        )
        .bind(task_id)
        .bind(goal_run_id)
        .fetch_optional(&mut *tx)
        .await?;
        if claimable.is_none() {
            tx.rollback().await?;
            return Ok(None);
        }
        let unmet = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*)
             FROM json_each(COALESCE((SELECT depends_on FROM tasks WHERE id = ?), '[]')) dep
             WHERE NOT EXISTS (
                SELECT 1 FROM tasks d
                WHERE d.id = CAST(dep.value AS TEXT)
                  AND d.goal_run_id = ?
                  AND d.status IN ('completed', 'skipped', 'superseded')
             )",
        )
        .bind(task_id)
        .bind(goal_run_id)
        .fetch_one(&mut *tx)
        .await?;
        if unmet > 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        let attempt = crate::traits::TaskAttempt {
            id: uuid::Uuid::new_v4().to_string(),
            task_id: task_id.to_string(),
            goal_run_id: goal_run_id.to_string(),
            worker_profile_id: None,
            worker_instance_id: worker_instance_id.to_string(),
            lease_token: uuid::Uuid::new_v4().to_string(),
            status: "claimed".to_string(),
            lease_expires_at: (now + chrono::Duration::seconds(lease_secs)).to_rfc3339(),
            last_heartbeat_at: now_string.clone(),
            workspace_id: None,
            started_at: now_string,
            completed_at: None,
        };
        sqlx::query(
            "INSERT INTO task_attempts
                (id, task_id, goal_run_id, worker_profile_id, worker_instance_id,
                 lease_token, status, lease_expires_at, last_heartbeat_at,
                 workspace_id, started_at, completed_at)
             VALUES (?, ?, ?, NULL, ?, ?, 'claimed', ?, ?, NULL, ?, NULL)",
        )
        .bind(&attempt.id)
        .bind(&attempt.task_id)
        .bind(&attempt.goal_run_id)
        .bind(&attempt.worker_instance_id)
        .bind(&attempt.lease_token)
        .bind(&attempt.lease_expires_at)
        .bind(&attempt.last_heartbeat_at)
        .bind(&attempt.started_at)
        .execute(&mut *tx)
        .await?;
        let updated = sqlx::query(
            "UPDATE tasks
             SET status = 'claimed', agent_id = ?, current_attempt_id = ?,
                 worker_profile_id = NULL, started_at = COALESCE(started_at, datetime('now')),
                 result = NULL, error = NULL, blocker = NULL, completed_at = NULL,
                 updated_at = datetime('now'), version = version + 1
             WHERE id = ? AND goal_run_id = ? AND status = 'pending'
               AND current_attempt_id IS NULL",
        )
        .bind(worker_instance_id)
        .bind(&attempt.id)
        .bind(task_id)
        .bind(goal_run_id)
        .execute(&mut *tx)
        .await?;
        if updated.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }
        tx.commit().await?;
        Ok(Some(attempt))
    }

    async fn record_mandate_decision(
        &self,
        decision: &MandateDecisionCycle,
        intention: Option<&Intention>,
        task_attempt_id: Option<&str>,
    ) -> anyhow::Result<()> {
        self.record_mandate_decision_with_updates(decision, intention, None, task_attempt_id)
            .await
    }

    async fn record_mandate_decision_with_updates(
        &self,
        decision: &MandateDecisionCycle,
        intention: Option<&Intention>,
        operating_updates: Option<&MandateOperatingUpdates>,
        task_attempt_id: Option<&str>,
    ) -> anyhow::Result<()> {
        decision
            .validate_content_bounds()
            .map_err(anyhow::Error::msg)?;
        if let Some(updates) = operating_updates {
            anyhow::ensure!(
                updates.strategy_revisions.len() <= 4,
                "one decision may revise at most four adaptive strategy nodes"
            );
            if let Some(note) = updates.learning_note.as_ref() {
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
            let mut strategy_keys = std::collections::HashSet::new();
            for revision in &updates.strategy_revisions {
                revision.validate().map_err(anyhow::Error::msg)?;
                anyhow::ensure!(
                    revision.mandate_id == decision.mandate_id
                        && revision.mandate_version == decision.mandate_version
                        && revision.decision_cycle_id == decision.id,
                    "strategy revision does not belong to this decision"
                );
                anyhow::ensure!(
                    strategy_keys.insert(revision.strategy_key.as_str()),
                    "one decision cannot revise the same strategy key twice"
                );
                anyhow::ensure!(
                    revision
                        .evidence_receipt_ids
                        .iter()
                        .all(|id| decision.evidence_receipt_ids.contains(id)),
                    "strategy evidence must be a subset of current decision evidence"
                );
            }
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
        if let Some(intention) = intention {
            if let Some(mandate) = self.get_mandate(&decision.mandate_id).await? {
                if !mandate.success_criteria.is_empty() {
                    intention
                        .validate_value_contract()
                        .map_err(anyhow::Error::msg)?;
                    let value_criterion = intention
                        .value_criterion
                        .as_deref()
                        .expect("validated value contract has a criterion");
                    anyhow::ensure!(
                        mandate
                            .success_criteria
                            .iter()
                            .any(|criterion| criterion == value_criterion),
                        "ACT value_criterion is not an exact owner-authored success criterion"
                    );
                }
            }
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
        anyhow::ensure!(
            decision.outcome == MandateDecisionOutcome::Stop
                || (decision.termination_kind.is_none() && decision.termination_match.is_none()),
            "termination fields are valid only for STOP"
        );

        let now = chrono::Utc::now();
        let now_string = now.to_rfc3339();
        let mut tx = self.pool.begin().await?;
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
        let row = sqlx::query(&query)
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
        validate_current_run_receipt_refs(
            &mut tx,
            &decision.goal_run_id,
            &decision.evidence_receipt_ids,
        )
        .await?;
        // A runtime-authored fallback is a typed review-failure marker, not a
        // deliberation: it carries no evidence, no intention, and no
        // authority, so the semantic gates below (measurement, stagnation)
        // cannot apply to it. Structural gates (lease, version, one decision
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
        let current_measurement = if mandate.objective_control.is_some()
            && !runtime_fallback
            && matches!(
                decision.outcome,
                MandateDecisionOutcome::Act
                    | MandateDecisionOutcome::Wait
                    | MandateDecisionOutcome::Stop
            ) {
            sqlx::query_scalar::<_, i64>(
                "SELECT value_micros FROM mandate_objective_measurements
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
        if mandate.objective_control.is_some()
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
        if decision.outcome == MandateDecisionOutcome::Wait && !runtime_fallback {
            if let Some(control) = mandate.objective_control.as_ref() {
                let limit = i64::from(control.max_stagnant_measurements) + 1;
                let values = sqlx::query_scalar::<_, i64>(
                    "SELECT value_micros FROM (
                         SELECT value_micros, observed_at, id,
                                ROW_NUMBER() OVER (
                                    PARTITION BY goal_run_id
                                    ORDER BY julianday(observed_at) DESC, id DESC
                                ) AS run_rank
                         FROM mandate_objective_measurements
                         WHERE mandate_id = ?
                     )
                     WHERE run_rank = 1
                     ORDER BY julianday(observed_at) DESC, id DESC LIMIT ?",
                )
                .bind(&mandate.id)
                .bind(limit)
                .fetch_all(&mut *tx)
                .await?;
                if values.len() >= limit as usize {
                    let newest = values[0];
                    let oldest = *values.last().unwrap_or(&newest);
                    let improvement = match control.direction {
                        crate::traits::ObjectiveMetricDirection::AtLeast => {
                            newest.saturating_sub(oldest)
                        }
                        crate::traits::ObjectiveMetricDirection::AtMost => {
                            oldest.saturating_sub(newest)
                        }
                    };
                    anyhow::ensure!(
                        improvement >= control.minimum_effect_micros,
                        "objective_control_stagnant: WAIT is not permitted after the configured no-progress window; choose a bounded ACT, ASK, or STOP decision"
                    );
                }
            }
        }
        if decision.outcome == MandateDecisionOutcome::Stop {
            let termination_kind = decision
                .termination_kind
                .ok_or_else(|| anyhow::anyhow!("STOP requires a typed termination_kind"))?;
            match termination_kind {
                MandateTerminationKind::SuccessCriteriaSatisfied => {
                    if let Some(control) = mandate.objective_control.as_ref() {
                        anyhow::ensure!(
                            current_measurement
                                .is_some_and(|value| control.target_reached(value)),
                            "success termination requires the current receipt-backed metric to reach the owner-confirmed target"
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
        if decision.outcome != MandateDecisionOutcome::Stop {
            if let Some(control) = mandate.objective_control.as_ref() {
                let measurement_due =
                    now + chrono::Duration::seconds(control.measurement_cadence_secs);
                let selected = chrono::DateTime::parse_from_rfc3339(&next_review_at)?
                    .with_timezone(&chrono::Utc);
                if selected > measurement_due {
                    next_review_at = measurement_due.to_rfc3339();
                }
            }
        }
        let persisted_reconsider_at = if decision.outcome == MandateDecisionOutcome::Stop {
            None
        } else {
            Some(next_review_at.as_str())
        };
        sqlx::query(
            "INSERT INTO mandate_decision_cycles (
                id, mandate_id, goal_run_id, mandate_version, outcome, activity_level, rationale,
                belief_snapshot, evidence_receipt_ids_json, question, termination_kind,
                termination_match, reconsider_at, action_attempts, created_at, updated_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&decision.id)
        .bind(&decision.mandate_id)
        .bind(&decision.goal_run_id)
        .bind(decision.mandate_version)
        .bind(decision.outcome.as_str())
        .bind(decision.activity_level.as_str())
        .bind(&decision.rationale)
        .bind(&decision.belief_snapshot)
        .bind(serde_json::to_string(&decision.evidence_receipt_ids)?)
        .bind(&decision.question)
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
            if !mandate.success_criteria.is_empty() {
                intention
                    .validate_value_contract()
                    .map_err(anyhow::Error::msg)?;
                let value_criterion = intention
                    .value_criterion
                    .as_deref()
                    .expect("validated value contract has a criterion");
                anyhow::ensure!(
                    mandate
                        .success_criteria
                        .iter()
                        .any(|criterion| criterion == value_criterion),
                    "ACT value_criterion is not an exact owner-authored success criterion"
                );
            }
            anyhow::ensure!(
                intention.status == IntentionStatus::Committed && intention.completed_at.is_none(),
                "a new intention must be committed and incomplete"
            );
            sqlx::query(
                "INSERT INTO intentions (
                    id, mandate_id, decision_cycle_id, goal_run_id, description,
                    rationale, value_criterion, expected_benefit, risk, invalidation_criteria,
                    status, created_at, updated_at, completed_at
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            .bind(intention.status.as_str())
            .bind(&intention.created_at)
            .bind(&intention.updated_at)
            .bind(&intention.completed_at)
            .execute(&mut *tx)
            .await?;
        }

        if let Some(updates) = operating_updates {
            if let Some(note) = updates.learning_note.as_ref() {
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
            for revision in &updates.strategy_revisions {
                sqlx::query(
                    "INSERT INTO mandate_strategy_revisions (
                        id, mandate_id, mandate_version, decision_cycle_id, strategy_key,
                        kind, guidance, confidence_bps, evidence_receipt_ids_json, created_at
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(&revision.id)
                .bind(&revision.mandate_id)
                .bind(revision.mandate_version)
                .bind(&revision.decision_cycle_id)
                .bind(&revision.strategy_key)
                .bind(revision.kind.as_str())
                .bind(&revision.guidance)
                .bind(i64::from(revision.confidence_bps))
                .bind(serde_json::to_string(&revision.evidence_receipt_ids)?)
                .bind(&revision.created_at)
                .execute(&mut *tx)
                .await?;
            }
            let active_strategy_nodes: i64 = sqlx::query_scalar(
                "SELECT COUNT(*)
                 FROM mandate_strategy_revisions r
                 WHERE r.mandate_id = ? AND r.kind != 'retire'
                   AND NOT EXISTS (
                       SELECT 1 FROM mandate_strategy_revisions newer
                       WHERE newer.mandate_id = r.mandate_id
                         AND newer.strategy_key = r.strategy_key
                         AND (julianday(newer.created_at) > julianday(r.created_at)
                              OR (newer.created_at = r.created_at AND newer.id > r.id))
                   )",
            )
            .bind(&decision.mandate_id)
            .fetch_one(&mut *tx)
            .await?;
            anyhow::ensure!(
                active_strategy_nodes <= 16,
                "adaptive strategy cannot exceed sixteen active nodes"
            );
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

    async fn get_mandate_decision_for_run(
        &self,
        goal_run_id: &str,
    ) -> anyhow::Result<Option<MandateDecisionCycle>> {
        let query =
            format!("SELECT {DECISION_COLUMNS} FROM mandate_decision_cycles WHERE goal_run_id = ?");
        let row = sqlx::query(&query)
            .bind(goal_run_id)
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(decision_from_row).transpose()
    }

    async fn list_mandate_decisions(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<MandateDecisionCycle>> {
        let query = format!(
            "SELECT {DECISION_COLUMNS} FROM mandate_decision_cycles
             WHERE mandate_id = ? ORDER BY julianday(created_at) DESC, id DESC LIMIT ?"
        );
        let rows = sqlx::query(&query)
            .bind(mandate_id)
            .bind(limit.clamp(1, 200))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(decision_from_row).collect()
    }

    async fn list_intentions(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<Intention>> {
        let query = format!(
            "SELECT {INTENTION_COLUMNS} FROM intentions
             WHERE mandate_id = ? ORDER BY julianday(created_at) DESC, id DESC LIMIT ?"
        );
        let rows = sqlx::query(&query)
            .bind(mandate_id)
            .bind(limit.clamp(1, 200))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(intention_from_row).collect()
    }

    async fn record_mandate_learning_note(&self, note: &MandateLearningNote) -> anyhow::Result<()> {
        note.validate().map_err(anyhow::Error::msg)?;
        let mut tx = self.pool.begin().await?;
        let decision_exists = sqlx::query_scalar::<_, i64>(
            "SELECT 1
             FROM mandate_decision_cycles dc
             JOIN mandates m ON m.id = dc.mandate_id
             WHERE dc.id = ? AND dc.mandate_id = ? AND dc.mandate_version = ?
               AND m.version >= dc.mandate_version
             LIMIT 1",
        )
        .bind(&note.learned_in_decision_cycle_id)
        .bind(&note.mandate_id)
        .bind(note.mandate_version)
        .fetch_optional(&mut *tx)
        .await?;
        anyhow::ensure!(
            decision_exists.is_some(),
            "mandate learning note does not belong to its decision epoch"
        );

        let mut unique = std::collections::HashSet::new();
        for receipt_id in &note.evidence_receipt_ids {
            anyhow::ensure!(
                unique.insert(receipt_id),
                "duplicate learning evidence receipt ID"
            );
            let found = sqlx::query_scalar::<_, i64>(
                "SELECT 1
                 FROM events e
                 JOIN tasks t ON t.id = json_extract(e.data, '$.task_id')
                 JOIN goal_runs gr ON gr.id = t.goal_run_id
                 JOIN mandates m ON m.goal_id = gr.goal_id
                 WHERE m.id = ? AND gr.trigger_type = 'mandate'
                   AND e.event_type = 'tool_result'
                   AND json_extract(e.data, '$.tool_call_id') = ?
                   AND json_extract(e.data, '$.receipt.schema_version') = ?
                   AND json_extract(e.data, '$.receipt.outcome_status') = 'succeeded'
                   AND json_extract(e.data, '$.receipt.outcome_evidence')
                       IN ('tool_reported', 'structured_metadata')
                 LIMIT 1",
            )
            .bind(&note.mandate_id)
            .bind(receipt_id)
            .bind(i64::from(crate::events::ToolReceiptV1::SCHEMA_VERSION))
            .fetch_optional(&mut *tx)
            .await?;
            anyhow::ensure!(
                found.is_some(),
                "learning evidence receipt `{receipt_id}` is not a same-mandate structured success"
            );
        }

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
        tx.commit().await?;
        Ok(())
    }

    async fn list_mandate_learning_notes(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<MandateLearningNote>> {
        let query = format!(
            "SELECT {LEARNING_NOTE_COLUMNS} FROM mandate_learning_notes
             WHERE mandate_id = ? ORDER BY julianday(created_at) DESC, id DESC LIMIT ?"
        );
        let rows = sqlx::query(&query)
            .bind(mandate_id)
            .bind(limit.clamp(1, 100))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(learning_note_from_row).collect()
    }

    async fn list_current_mandate_strategy(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<MandateStrategyRevision>> {
        let query = format!(
            "SELECT {STRATEGY_REVISION_COLUMNS}
             FROM mandate_strategy_revisions r
             WHERE r.mandate_id = ?
               AND NOT EXISTS (
                   SELECT 1 FROM mandate_strategy_revisions newer
                   WHERE newer.mandate_id = r.mandate_id
                     AND newer.strategy_key = r.strategy_key
                     AND (julianday(newer.created_at) > julianday(r.created_at)
                          OR (newer.created_at = r.created_at AND newer.id > r.id))
               )
             ORDER BY julianday(r.created_at) DESC, r.id DESC LIMIT ?"
        );
        let rows = sqlx::query(&query)
            .bind(mandate_id)
            .bind(limit.clamp(1, 100))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(strategy_revision_from_row).collect()
    }

    async fn wake_mandates_for_signal(
        &self,
        signal: &MandateWakeSignal,
    ) -> anyhow::Result<Vec<String>> {
        signal.validate().map_err(anyhow::Error::msg)?;
        let mandates = self.list_mandates(None, false).await?;
        let mut target = reqwest::Url::parse(&signal.target_url)?;
        target.set_query(None);
        target.set_fragment(None);
        let target_url = target.to_string();
        let signal_digest = format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "aidaemon.mandate.wake.v1\0{}\0{}\0{}\0{}\0{}",
                    signal.kind.as_str(),
                    signal.source,
                    signal.target_url,
                    signal.account_id.as_deref().unwrap_or(""),
                    signal.dedupe_key
                )
                .as_bytes()
            )
        );
        let received_at = chrono::Utc::now().to_rfc3339();
        let mut awakened = Vec::new();
        let mut tx = self.pool.begin().await?;
        for mandate in mandates.into_iter().filter(|mandate| {
            crate::mandates::authority::mandate_accepts_wake_signal(mandate, signal)
        }) {
            let inserted = sqlx::query(
                "INSERT OR IGNORE INTO mandate_wake_signals (
                    mandate_id, mandate_version, signal_digest, kind, source,
                    target_url, account_id, occurred_at, received_at
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&mandate.id)
            .bind(mandate.version)
            .bind(&signal_digest)
            .bind(signal.kind.as_str())
            .bind(&signal.source)
            .bind(&target_url)
            .bind(&signal.account_id)
            .bind(&signal.occurred_at)
            .bind(&received_at)
            .execute(&mut *tx)
            .await?;
            if inserted.rows_affected() == 0 {
                continue;
            }
            let updated = sqlx::query(
                "UPDATE mandates
                 SET next_review_at = CASE
                        WHEN julianday(next_review_at) > julianday(?) THEN ?
                        ELSE next_review_at
                     END
                 WHERE id = ? AND version = ? AND status = 'active'
                   AND autonomy_mode = 'autopilot' AND confirmed_at IS NOT NULL",
            )
            .bind(&received_at)
            .bind(&received_at)
            .bind(&mandate.id)
            .bind(mandate.version)
            .execute(&mut *tx)
            .await?;
            if updated.rows_affected() == 1 {
                awakened.push(mandate.id);
            }
        }
        tx.commit().await?;
        Ok(awakened)
    }

    async fn resolve_mandate_suspension(
        &self,
        mandate_id: &str,
        expected_version: i64,
        expected_kind: MandateSuspensionKind,
        controller_context: Option<&str>,
        reconciliation_resolution: Option<MandateReconciliationResolution>,
        owner_guidance: &str,
        owner_session: &str,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            !owner_guidance.trim().is_empty() && owner_guidance.trim() == owner_guidance,
            "owner guidance is required and must be canonical"
        );
        anyhow::ensure!(
            owner_guidance.chars().count() <= 1024 && owner_guidance.len() <= 1024,
            "owner guidance exceeds its 1 KiB bound"
        );
        anyhow::ensure!(
            !owner_session.trim().is_empty(),
            "owner session is required"
        );
        if let Some(context) = controller_context {
            anyhow::ensure!(
                context.chars().count() <= 64_000,
                "mandate controller context is too large"
            );
            serde_json::from_str::<serde_json::Value>(context)?;
        }
        match expected_kind {
            MandateSuspensionKind::AwaitingAnswer
            | MandateSuspensionKind::ObjectiveControlRequired => anyhow::ensure!(
                reconciliation_resolution.is_none(),
                "non-reconciliation suspensions cannot carry a reconciliation resolution"
            ),
            MandateSuspensionKind::ReconciliationRequired
            | MandateSuspensionKind::ExecutionLeaseLost
            | MandateSuspensionKind::ReviewFailed
            | MandateSuspensionKind::AuthorityRevokedWithUnresolvedMutation => anyhow::ensure!(
                reconciliation_resolution.is_some(),
                "safety suspension requires a typed reconciliation resolution"
            ),
            MandateSuspensionKind::OwnerPaused => {
                anyhow::bail!("owner-paused mandates must use ordinary resume")
            }
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let current = sqlx::query(
            "SELECT goal_id, created_by_session, suspension_json, objective_control_json
             FROM mandates
             WHERE id = ? AND version = ? AND status = 'awaiting_input'
               AND confirmed_at IS NOT NULL",
        )
        .bind(mandate_id)
        .bind(expected_version)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(current) = current else {
            tx.rollback().await?;
            return Ok(false);
        };
        anyhow::ensure!(
            current.get::<String, _>("created_by_session") == owner_session,
            "mandate does not belong to this owner session"
        );
        let suspension: MandateSuspension = serde_json::from_str(
            &current
                .get::<Option<String>, _>("suspension_json")
                .ok_or_else(|| anyhow::anyhow!("awaiting-input mandate has no typed suspension"))?,
        )?;
        anyhow::ensure!(
            suspension.kind == expected_kind,
            "mandate suspension changed before it was resolved"
        );
        if expected_kind == MandateSuspensionKind::ObjectiveControlRequired {
            anyhow::ensure!(
                current
                    .get::<Option<String>, _>("objective_control_json")
                    .is_some(),
                "objective control must be configured before this mandate can resume"
            );
        }
        let goal_id: String = current.get("goal_id");

        if let Some(resolution) = reconciliation_resolution {
            sqlx::query(
                "INSERT INTO mandate_reconciliations (
                    id, mandate_id, suspended_version, suspension_kind, resolution,
                    owner_guidance, resolved_by_session, resolved_at
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(uuid::Uuid::new_v4().to_string())
            .bind(mandate_id)
            .bind(expected_version)
            .bind(expected_kind.as_str())
            .bind(resolution.as_str())
            .bind(owner_guidance)
            .bind(owner_session)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }

        let updated = sqlx::query(
            "UPDATE mandates
             SET status = 'active', suspension_json = NULL, next_review_at = ?,
                 review_lease_token = NULL, review_lease_expires_at = NULL,
                 version = version + 1, updated_at = ?
             WHERE id = ? AND version = ? AND status = 'awaiting_input'",
        )
        .bind(&now)
        .bind(&now)
        .bind(mandate_id)
        .bind(expected_version)
        .execute(&mut *tx)
        .await?;
        if updated.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(false);
        }
        if let Some(context) = controller_context {
            sqlx::query("UPDATE goals SET context = ?, updated_at = ? WHERE id = ?")
                .bind(context)
                .bind(&now)
                .bind(&goal_id)
                .execute(&mut *tx)
                .await?;
        }
        update_controller_status(&mut tx, &goal_id, MandateStatus::Active, &now).await?;
        tx.commit().await?;
        Ok(true)
    }

    async fn reserve_mandate_action_attempt(
        &self,
        reservation: &MandateMutationReservation,
    ) -> anyhow::Result<Option<MandateMutationAttempt>> {
        let reserved_at = validate_mutation_reservation(reservation)?;
        let expected_attempts = reservation.grant.reserved_action_attempt - 1;
        let reserved_at_string = reserved_at.to_rfc3339();
        let mutation_effects_json = serde_json::to_string(&reservation.mutation_effects)?;
        let targets_json = serde_json::to_string(&reservation.targets)?;
        let account_identifiers_json = serde_json::to_string(&reservation.account_identifiers)?;
        let ledger_id = uuid::Uuid::new_v4().to_string();
        let mut tx = self.pool.begin().await?;

        // This UPDATE is deliberately the transaction's first database access.
        // SQLite therefore acquires the writer lock before evaluating the
        // cross-cycle counts, making the last quota slot race-free.
        let result = sqlx::query(
            "UPDATE mandate_decision_cycles AS dc
             SET action_attempts = action_attempts + 1, updated_at = ?
             WHERE dc.id = ? AND dc.outcome = 'act'
               AND dc.mandate_id = ? AND dc.mandate_version = ?
               AND dc.goal_run_id = ?
               AND dc.action_attempts = ?
               AND ? = dc.action_attempts + 1
               AND EXISTS (
                   SELECT 1
                   FROM mandates m
                   JOIN goals g ON g.id = m.goal_id
                   JOIN goal_runs gr ON gr.id = dc.goal_run_id AND gr.goal_id = m.goal_id
                   JOIN intentions i ON i.decision_cycle_id = dc.id
                   JOIN tasks root ON root.id = gr.root_task_id AND root.goal_run_id = gr.id
                   JOIN task_attempts root_attempt
                     ON root_attempt.id = root.current_attempt_id
                    AND root_attempt.task_id = root.id
                    AND root_attempt.goal_run_id = gr.id
                   JOIN tasks worker ON worker.id = ? AND worker.goal_run_id = gr.id
                   JOIN task_attempts worker_attempt
                     ON worker_attempt.id = worker.current_attempt_id
                    AND worker_attempt.task_id = worker.id
                    AND worker_attempt.goal_run_id = gr.id
                   WHERE m.id = dc.mandate_id
                     AND m.owner_principal_id = ?
                     AND m.status = 'active' AND g.status = 'active'
                     AND m.confirmed_at IS NOT NULL
                     AND m.version = dc.mandate_version
                     AND gr.trigger_type = 'mandate'
                     AND gr.status = 'running'
                     AND gr.root_task_id = ?
                     AND root.id != worker.id
                     AND root.status IN ('claimed', 'running')
                     AND root.current_attempt_id = ?
                     AND root_attempt.id = ?
                     AND root_attempt.status IN ('claimed', 'running')
                     AND julianday(root_attempt.lease_expires_at) > julianday(?)
                     AND worker.status IN ('claimed', 'running')
                     AND worker.current_attempt_id = ?
                     AND worker_attempt.id = ?
                     AND worker_attempt.status IN ('claimed', 'running')
                     AND julianday(worker_attempt.lease_expires_at) > julianday(?)
                     AND i.mandate_id = m.id AND i.goal_run_id = gr.id
                     AND i.status = 'committed' AND i.completed_at IS NULL
                     AND dc.action_attempts < CAST(
                         COALESCE(json_extract(m.authority_json,
                             '$.max_mutating_actions_per_cycle'), 0) AS INTEGER
                     )
                     AND CAST(COALESCE(json_extract(m.authority_json,
                             '$.max_mutating_actions_per_rolling_24h'), 0) AS INTEGER) > 0
                     AND (
                         SELECT COUNT(*) FROM mandate_mutation_attempts prior
                         WHERE prior.mandate_id = m.id
                           AND prior.status != 'never_dispatched'
                           AND julianday(prior.reserved_at) > julianday(?) - 1.0
                     ) < CAST(COALESCE(json_extract(m.authority_json,
                             '$.max_mutating_actions_per_rolling_24h'), 0) AS INTEGER)
                     AND NOT EXISTS (
                         SELECT 1 FROM mandate_mutation_attempts recent
                         WHERE recent.mandate_id = m.id
                           AND recent.status != 'never_dispatched'
                           AND julianday(recent.reserved_at) > julianday(?) -
                               (CAST(COALESCE(json_extract(m.authority_json,
                                   '$.min_seconds_between_mutations'), 0) AS REAL) / 86400.0)
                     )
                     AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
               )",
        )
        .bind(&reserved_at_string)
        .bind(&reservation.grant.decision_cycle_id)
        .bind(&reservation.grant.mandate_id)
        .bind(reservation.grant.mandate_version)
        .bind(&reservation.goal_run_id)
        .bind(expected_attempts)
        .bind(reservation.grant.reserved_action_attempt)
        .bind(&reservation.task_id)
        .bind(&reservation.grant.owner_principal_id)
        .bind(&reservation.root_task_id)
        .bind(&reservation.root_task_attempt_id)
        .bind(&reservation.root_task_attempt_id)
        .bind(&reserved_at_string)
        .bind(&reservation.task_attempt_id)
        .bind(&reservation.task_attempt_id)
        .bind(&reserved_at_string)
        .bind(&reserved_at_string)
        .bind(&reserved_at_string)
        .bind(&reserved_at_string)
        .execute(&mut *tx)
        .await?;
        if result.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }

        let intention_id = sqlx::query_scalar::<_, String>(
            "SELECT id FROM intentions
             WHERE decision_cycle_id = ? AND mandate_id = ? AND goal_run_id = ?
               AND status = 'committed' AND completed_at IS NULL",
        )
        .bind(&reservation.grant.decision_cycle_id)
        .bind(&reservation.grant.mandate_id)
        .bind(&reservation.goal_run_id)
        .fetch_one(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO mandate_mutation_attempts (
                id, mandate_id, mandate_version, decision_cycle_id, goal_run_id, intention_id,
                root_task_id, root_task_attempt_id, task_id, task_attempt_id,
                reserved_action_attempt, action_digest, tool_call_id, tool_name,
                mutation_effects_json, targets_json, account_identifiers_json,
                status, reserved_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?)",
        )
        .bind(&ledger_id)
        .bind(&reservation.grant.mandate_id)
        .bind(reservation.grant.mandate_version)
        .bind(&reservation.grant.decision_cycle_id)
        .bind(&reservation.goal_run_id)
        .bind(&intention_id)
        .bind(&reservation.root_task_id)
        .bind(&reservation.root_task_attempt_id)
        .bind(&reservation.task_id)
        .bind(&reservation.task_attempt_id)
        .bind(reservation.grant.reserved_action_attempt)
        .bind(&reservation.grant.action_digest)
        .bind(&reservation.tool_call_id)
        .bind(&reservation.tool_name)
        .bind(&mutation_effects_json)
        .bind(&targets_json)
        .bind(&account_identifiers_json)
        .bind(&reserved_at_string)
        .execute(&mut *tx)
        .await?;

        let query = format!(
            "SELECT {MUTATION_ATTEMPT_COLUMNS} FROM mandate_mutation_attempts WHERE id = ?"
        );
        let row = sqlx::query(&query)
            .bind(&ledger_id)
            .fetch_one(&mut *tx)
            .await?;
        let attempt = mutation_attempt_from_row(&row)?;
        tx.commit().await?;
        Ok(Some(attempt))
    }

    async fn claim_mandate_mutation_dispatch(
        &self,
        claim: &MandateMutationDispatchClaim,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            claim.grant.counts_toward_cycle_budget
                && claim.grant.mandate_version > 0
                && claim.grant.reserved_action_attempt > 0,
            "mutation dispatch requires a positive metered grant"
        );
        validate_canonical_identifier("owner principal id", &claim.grant.owner_principal_id, 256)?;
        anyhow::ensure!(
            claim.grant.action_digest.len() == 64
                && claim
                    .grant
                    .action_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "invalid mutation dispatch action digest"
        );
        anyhow::ensure!(
            claim.grant.tool_call_id.as_deref() == Some(claim.tool_call_id.as_str()),
            "mutation dispatch tool call does not match its bound grant"
        );
        for (label, value) in [
            ("mandate id", claim.grant.mandate_id.as_str()),
            ("decision cycle id", claim.grant.decision_cycle_id.as_str()),
            ("goal run id", claim.goal_run_id.as_str()),
            ("root task id", claim.root_task_id.as_str()),
            ("root task attempt id", claim.root_task_attempt_id.as_str()),
            ("task id", claim.task_id.as_str()),
            ("task attempt id", claim.task_attempt_id.as_str()),
            ("tool call id", claim.tool_call_id.as_str()),
            ("tool name", claim.tool_name.as_str()),
        ] {
            validate_canonical_identifier(label, value, 256)?;
        }
        anyhow::ensure!(
            claim.root_task_id != claim.task_id,
            "the mandate deliberator cannot dispatch mutations"
        );
        validate_timestamp("mutation dispatch claimed_at", &claim.claimed_at)?;

        // This single conditional write is the one-use point. Concurrent or
        // replayed dispatches cannot both move dispatch_claimed_at from NULL,
        // and every mutable lifecycle/lease fact is checked in the same
        // writer transaction before adapter I/O begins.
        let result = sqlx::query(
            "UPDATE mandate_mutation_attempts AS ma
             SET dispatch_claimed_at = ?
             WHERE ma.mandate_id = ? AND ma.mandate_version = ?
               AND ma.decision_cycle_id = ? AND ma.goal_run_id = ?
               AND ma.root_task_id = ? AND ma.root_task_attempt_id = ?
               AND ma.task_id = ? AND ma.task_attempt_id = ?
               AND ma.tool_call_id = ? AND ma.tool_name = ?
               AND ma.reserved_action_attempt = ? AND ma.action_digest = ?
               AND ma.status = 'reserved' AND ma.dispatch_claimed_at IS NULL
               AND EXISTS (
                   SELECT 1
                   FROM mandates m
                   JOIN goals g ON g.id = m.goal_id
                   JOIN goal_runs gr ON gr.id = ma.goal_run_id AND gr.goal_id = m.goal_id
                   JOIN mandate_decision_cycles dc
                     ON dc.id = ma.decision_cycle_id
                    AND dc.goal_run_id = gr.id
                    AND dc.mandate_id = m.id
                   JOIN intentions i
                     ON i.id = ma.intention_id
                    AND i.decision_cycle_id = dc.id
                    AND i.goal_run_id = gr.id
                    AND i.mandate_id = m.id
                   JOIN tasks root
                     ON root.id = ma.root_task_id AND root.goal_run_id = gr.id
                   JOIN task_attempts root_attempt
                     ON root_attempt.id = ma.root_task_attempt_id
                    AND root_attempt.task_id = root.id
                    AND root_attempt.goal_run_id = gr.id
                   JOIN tasks worker
                     ON worker.id = ma.task_id AND worker.goal_run_id = gr.id
                   JOIN task_attempts worker_attempt
                     ON worker_attempt.id = ma.task_attempt_id
                    AND worker_attempt.task_id = worker.id
                    AND worker_attempt.goal_run_id = gr.id
                   WHERE m.id = ma.mandate_id
                     AND m.owner_principal_id = ?
                     AND m.version = ma.mandate_version
                     AND m.status = 'active' AND m.confirmed_at IS NOT NULL
                     AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday(?))
                     AND g.status = 'active'
                     AND gr.trigger_type = 'mandate' AND gr.status = 'running'
                     AND gr.root_task_id = root.id AND root.id != worker.id
                     AND dc.mandate_version = m.version AND dc.outcome = 'act'
                     AND dc.action_attempts = ma.reserved_action_attempt
                     AND i.status = 'committed' AND i.completed_at IS NULL
                     AND root.status IN ('claimed', 'running')
                     AND root.current_attempt_id = root_attempt.id
                     AND root.error IS NULL AND root.blocker IS NULL
                     AND root_attempt.status IN ('claimed', 'running')
                     AND julianday(root_attempt.lease_expires_at) > julianday(?)
                     AND worker.status IN ('claimed', 'running')
                     AND worker.current_attempt_id = worker_attempt.id
                     AND worker.error IS NULL AND worker.blocker IS NULL
                     AND worker_attempt.status IN ('claimed', 'running')
                     AND julianday(worker_attempt.lease_expires_at) > julianday(?)
               )",
        )
        .bind(&claim.claimed_at)
        .bind(&claim.grant.mandate_id)
        .bind(claim.grant.mandate_version)
        .bind(&claim.grant.decision_cycle_id)
        .bind(&claim.goal_run_id)
        .bind(&claim.root_task_id)
        .bind(&claim.root_task_attempt_id)
        .bind(&claim.task_id)
        .bind(&claim.task_attempt_id)
        .bind(&claim.tool_call_id)
        .bind(&claim.tool_name)
        .bind(claim.grant.reserved_action_attempt)
        .bind(&claim.grant.action_digest)
        .bind(&claim.grant.owner_principal_id)
        .bind(&claim.claimed_at)
        .bind(&claim.claimed_at)
        .bind(&claim.claimed_at)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    async fn get_mandate_mutation_quota_state(
        &self,
        mandate_id: &str,
        as_of: &str,
    ) -> anyhow::Result<Option<MandateMutationQuotaState>> {
        validate_canonical_identifier("mandate id", mandate_id, 256)?;
        let mut connection = self.pool.acquire().await?;
        mutation_quota_state_on_connection(&mut connection, mandate_id, as_of).await
    }

    async fn project_mandate_mutation_outcome(
        &self,
        projection: &MandateMutationOutcomeProjection,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            projection.grant.counts_toward_cycle_budget
                && projection.grant.reserved_action_attempt > 0,
            "mutation outcome requires an exact metered grant"
        );
        anyhow::ensure!(
            projection.grant.tool_call_id.as_deref() == Some(projection.tool_call_id.as_str()),
            "mutation outcome tool call does not match its bound grant"
        );
        anyhow::ensure!(
            matches!(
                projection.status,
                MandateMutationAttemptStatus::Succeeded
                    | MandateMutationAttemptStatus::Failed
                    | MandateMutationAttemptStatus::Ambiguous
            ),
            "a receipt projection requires a post-dispatch outcome status"
        );
        for (label, value) in [
            ("goal run id", projection.goal_run_id.as_str()),
            ("task id", projection.task_id.as_str()),
            ("task attempt id", projection.task_attempt_id.as_str()),
            ("tool call id", projection.tool_call_id.as_str()),
        ] {
            validate_canonical_identifier(label, value, 256)?;
        }
        validate_timestamp("mutation outcome completed_at", &projection.completed_at)?;
        if projection.status == MandateMutationAttemptStatus::Succeeded {
            anyhow::ensure!(
                projection.receipt_schema_version == crate::events::ToolReceiptV1::SCHEMA_VERSION
                    && projection.outcome_evidence.is_some()
                    && !projection.timed_out
                    && !projection.background_started
                    && !projection.detached
                    && !projection.completion_notifications_enabled
                    && !projection.transport_error_present
                    && projection.semantics_match
                    && projection.exit_code.is_none_or(|code| code == 0),
                "mutation success lacks strict receipt proof"
            );
        }

        let result = sqlx::query(
            "UPDATE mandate_mutation_attempts
             SET status = ?, outcome_evidence = ?, http_status = ?, exit_code = ?,
                 completed_at = ?
             WHERE mandate_id = ? AND mandate_version = ?
               AND decision_cycle_id = ? AND reserved_action_attempt = ?
               AND action_digest = ? AND goal_run_id = ?
               AND task_id = ? AND task_attempt_id = ? AND tool_call_id = ?
               AND status = 'reserved' AND dispatch_claimed_at IS NOT NULL
               AND (
                   ? != 'succeeded'
                   OR tool_name != 'http_request'
                   OR ((? BETWEEN 200 AND 299) AND ? != 202)
               )",
        )
        .bind(projection.status.as_str())
        .bind(projection.outcome_evidence.map(|value| value.as_str()))
        .bind(projection.http_status.map(i64::from))
        .bind(projection.exit_code)
        .bind(&projection.completed_at)
        .bind(&projection.grant.mandate_id)
        .bind(projection.grant.mandate_version)
        .bind(&projection.grant.decision_cycle_id)
        .bind(projection.grant.reserved_action_attempt)
        .bind(&projection.grant.action_digest)
        .bind(&projection.goal_run_id)
        .bind(&projection.task_id)
        .bind(&projection.task_attempt_id)
        .bind(&projection.tool_call_id)
        .bind(projection.status.as_str())
        .bind(projection.http_status.map(i64::from))
        .bind(projection.http_status.map(i64::from))
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 1 {
            return Ok(true);
        }
        // The canonical ToolResult append projects the same receipt and ledger
        // transition atomically. The execution loop repeats this call as a
        // defensive compatibility check, so accept an exact terminal match.
        let already_projected = sqlx::query_scalar::<_, i64>(
            "SELECT 1 FROM mandate_mutation_attempts
             WHERE mandate_id = ? AND mandate_version = ? AND decision_cycle_id = ?
               AND reserved_action_attempt = ? AND action_digest = ?
               AND goal_run_id = ? AND task_id = ? AND task_attempt_id = ?
               AND tool_call_id = ? AND status = ?
               AND COALESCE(http_status, -1) = COALESCE(?, -1)
               AND COALESCE(exit_code, -2147483648) = COALESCE(?, -2147483648)
             LIMIT 1",
        )
        .bind(&projection.grant.mandate_id)
        .bind(projection.grant.mandate_version)
        .bind(&projection.grant.decision_cycle_id)
        .bind(projection.grant.reserved_action_attempt)
        .bind(&projection.grant.action_digest)
        .bind(&projection.goal_run_id)
        .bind(&projection.task_id)
        .bind(&projection.task_attempt_id)
        .bind(&projection.tool_call_id)
        .bind(projection.status.as_str())
        .bind(projection.http_status.map(i64::from))
        .bind(projection.exit_code)
        .fetch_optional(&self.pool)
        .await?;
        Ok(already_projected.is_some())
    }

    async fn list_mandate_mutation_attempts_for_run(
        &self,
        goal_run_id: &str,
    ) -> anyhow::Result<Vec<MandateMutationAttempt>> {
        validate_canonical_identifier("goal run id", goal_run_id, 256)?;
        let query = format!(
            "SELECT {MUTATION_ATTEMPT_COLUMNS} FROM mandate_mutation_attempts
             WHERE goal_run_id = ? ORDER BY julianday(reserved_at), id"
        );
        let rows = sqlx::query(&query)
            .bind(goal_run_id)
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(mutation_attempt_from_row).collect()
    }

    async fn finalize_mandate_run_from_proof(
        &self,
        request: &MandateRunFinalizationRequest,
    ) -> anyhow::Result<MandateRunFinalizationResult> {
        if validate_canonical_identifier("mandate id", &request.mandate_id, 256).is_err()
            || validate_canonical_identifier("goal run id", &request.goal_run_id, 256).is_err()
            || request.expected_mandate_version <= 0
            || validate_timestamp("mandate finalization timestamp", &request.finalized_at).is_err()
        {
            return Ok(MandateRunFinalizationResult::Rejected {
                reason: MandateFinalizationRejectReason::InvalidRequest,
            });
        }

        let mut tx = self.pool.begin().await?;
        // Acquire SQLite's writer lock before reading the proof snapshot. All
        // subsequent task, ledger, policy, and lifecycle checks therefore
        // linearize with owner updates and worker finalization writes.
        let fenced = sqlx::query(
            "UPDATE goal_runs SET updated_at = updated_at
             WHERE id = ? AND trigger_type = 'mandate' AND status = 'running'",
        )
        .bind(&request.goal_run_id)
        .execute(&mut *tx)
        .await?;
        if fenced.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(MandateRunFinalizationResult::Stale {
                reason: MandateFinalizationStaleReason::RunNotCurrent,
            });
        }

        let row = sqlx::query(
            "SELECT m.version AS mandate_version, m.status AS mandate_status,
                    m.min_review_secs AS min_review_secs,
                    m.max_review_secs AS max_review_secs,
                    g.session_id AS owner_session_id,
                    g.dispatch_failures AS review_failures,
                    gr.goal_id AS goal_id, gr.root_task_id AS root_task_id,
                    dc.id AS decision_cycle_id, dc.mandate_version AS decision_version,
                    dc.outcome AS decision_outcome, dc.rationale AS decision_rationale,
                    dc.action_attempts AS decision_action_attempts
             FROM goal_runs gr
             JOIN mandates m ON m.goal_id = gr.goal_id
             JOIN goals g ON g.id = gr.goal_id
             LEFT JOIN mandate_decision_cycles dc ON dc.goal_run_id = gr.id
             WHERE gr.id = ? AND m.id = ?",
        )
        .bind(&request.goal_run_id)
        .bind(&request.mandate_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(row) = row else {
            tx.rollback().await?;
            return Ok(MandateRunFinalizationResult::Stale {
                reason: MandateFinalizationStaleReason::MandateMissingOrVersionChanged,
            });
        };
        let mandate_version: i64 = row.get("mandate_version");
        let min_review_secs: i64 = row.get("min_review_secs");
        let max_review_secs: i64 = row.get("max_review_secs");
        let review_failures: i32 = row.get("review_failures");
        let goal_id: String = row.get("goal_id");
        let owner_session_id: String = row.get("owner_session_id");
        if mandate_version != request.expected_mandate_version {
            tx.rollback().await?;
            return Ok(MandateRunFinalizationResult::Stale {
                reason: MandateFinalizationStaleReason::MandateMissingOrVersionChanged,
            });
        }
        let open_run_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM goal_runs
             WHERE goal_id = ? AND status IN ('pending', 'running', 'blocked')",
        )
        .bind(&goal_id)
        .fetch_one(&mut *tx)
        .await?;
        if open_run_count != 1 {
            tx.rollback().await?;
            return Ok(MandateRunFinalizationResult::Stale {
                reason: MandateFinalizationStaleReason::RunNotCurrent,
            });
        }

        macro_rules! close_review_failed_retry {
            ($reason:expr) => {{
                let reason = $reason;
                sqlx::query(
                    "UPDATE task_attempts
                     SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
                     WHERE goal_run_id = ? AND status IN ('claimed', 'running')",
                )
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE tasks
                     SET status = 'cancelled', current_attempt_id = NULL,
                         completed_at = COALESCE(completed_at, ?), updated_at = ?,
                         version = version + 1
                     WHERE goal_run_id = ?
                       AND status IN ('pending', 'claimed', 'running', 'blocked')",
                )
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let run_updated = sqlx::query(
                    "UPDATE goal_runs
                     SET status = 'failed', outcome_summary = ?, completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
                )
                .bind(format!("mandate_review_failed:{}", reason.as_str()))
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let finalized_at = chrono::DateTime::parse_from_rfc3339(&request.finalized_at)
                    .expect("validated mandate finalization timestamp")
                    .with_timezone(&chrono::Utc);
                let next_review_failures = review_failures.saturating_add(1);
                let backoff_shift = u32::try_from(next_review_failures.saturating_sub(1))
                    .unwrap_or(0)
                    .min(6);
                let retry_delay_secs = min_review_secs
                    .saturating_mul(1_i64 << backoff_shift)
                    .min(max_review_secs);
                let retry_at =
                    (finalized_at + chrono::Duration::seconds(retry_delay_secs)).to_rfc3339();
                let mandate_updated = sqlx::query(
                    "UPDATE mandates
                     SET next_review_at = ?, review_lease_token = NULL,
                         review_lease_expires_at = NULL, updated_at = ?
                     WHERE id = ? AND version = ? AND status = 'active'",
                )
                .bind(&retry_at)
                .bind(&request.finalized_at)
                .bind(&request.mandate_id)
                .bind(request.expected_mandate_version)
                .execute(&mut *tx)
                .await?;
                anyhow::ensure!(
                    run_updated.rows_affected() == 1 && mandate_updated.rows_affected() == 1,
                    "mandate review-failure retry state changed during finalization"
                );
                update_controller_status(
                    &mut tx,
                    &goal_id,
                    MandateStatus::Active,
                    &request.finalized_at,
                )
                .await?;
                sqlx::query(
                    "UPDATE goals SET dispatch_failures = ?, updated_at = ? WHERE id = ?",
                )
                .bind(next_review_failures)
                .bind(&request.finalized_at)
                .bind(&goal_id)
                .execute(&mut *tx)
                .await?;
                let notice = crate::traits::MandateRunNotification::new(
                    &request.mandate_id,
                    request.expected_mandate_version,
                    &goal_id,
                    &request.goal_run_id,
                    &owner_session_id,
                    crate::traits::MandateRunNotificationKind::ReviewFailed { reason },
                    crate::traits::MandateRunProofCounts::default(),
                    &request.finalized_at,
                );
                super::notifications::enqueue_mandate_run_notification_on_connection(
                    &mut *tx,
                    &notice,
                )
                .await?;
                tx.commit().await?;
                return Ok(MandateRunFinalizationResult::Rejected { reason });
            }};
        }

        macro_rules! close_invalid_decision_state {
            () => {{
                let reason = MandateFinalizationRejectReason::InvalidDecisionState;
                sqlx::query(
                    "UPDATE mandate_mutation_attempts
                     SET status = CASE
                             WHEN dispatch_claimed_at IS NULL THEN 'never_dispatched'
                             ELSE 'ambiguous'
                         END,
                         completed_at = COALESCE(completed_at, ?)
                     WHERE goal_run_id = ? AND mandate_id = ? AND status = 'reserved'",
                )
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .bind(&request.mandate_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE task_attempts
                     SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
                     WHERE goal_run_id = ? AND status IN ('claimed', 'running')",
                )
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE tasks
                     SET status = 'cancelled', current_attempt_id = NULL,
                         completed_at = COALESCE(completed_at, ?), updated_at = ?,
                         version = version + 1
                     WHERE goal_run_id = ?
                       AND status IN ('pending', 'claimed', 'running', 'blocked')",
                )
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let (intention_from, intention_to) = intention_transition(
                    IntentionStatus::Committed,
                    IntentionStatus::Suspended,
                )?;
                sqlx::query(
                    "UPDATE intentions
                     SET status = ?, updated_at = ?, completed_at = ?
                     WHERE goal_run_id = ? AND status = ?",
                )
                .bind(intention_to)
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .bind(intention_from)
                .execute(&mut *tx)
                .await?;
                let run_updated = sqlx::query(
                    "UPDATE goal_runs
                     SET status = 'failed', outcome_summary = ?, completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
                )
                .bind(format!("mandate_review_failed:{}", reason.as_str()))
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let mut suspension = MandateSuspension::new(
                    MandateSuspensionKind::ReviewFailed,
                    Some(reason.as_str().to_string()),
                );
                suspension.goal_run_id = Some(request.goal_run_id.clone());
                let mandate_updated = sqlx::query(
                    "UPDATE mandates
                     SET status = 'awaiting_input', review_lease_token = NULL,
                         review_lease_expires_at = NULL, suspension_json = ?, updated_at = ?
                     WHERE id = ? AND version = ? AND status = 'active'",
                )
                .bind(serde_json::to_string(&suspension)?)
                .bind(&request.finalized_at)
                .bind(&request.mandate_id)
                .bind(request.expected_mandate_version)
                .execute(&mut *tx)
                .await?;
                anyhow::ensure!(
                    run_updated.rows_affected() == 1 && mandate_updated.rows_affected() == 1,
                    "invalid mandate decision state changed during finalization"
                );
                update_controller_status(
                    &mut tx,
                    &goal_id,
                    MandateStatus::AwaitingInput,
                    &request.finalized_at,
                )
                .await?;
                let notice = crate::traits::MandateRunNotification::new(
                    &request.mandate_id,
                    request.expected_mandate_version,
                    &goal_id,
                    &request.goal_run_id,
                    &owner_session_id,
                    crate::traits::MandateRunNotificationKind::ReviewFailed { reason },
                    crate::traits::MandateRunProofCounts::default(),
                    &request.finalized_at,
                );
                super::notifications::enqueue_mandate_run_notification_on_connection(
                    &mut *tx,
                    &notice,
                )
                .await?;
                tx.commit().await?;
                return Ok(MandateRunFinalizationResult::Rejected { reason });
            }};
        }
        let decision_cycle_id: Option<String> = row.get("decision_cycle_id");
        let decision_version: Option<i64> = row.get("decision_version");
        let decision_outcome_raw: Option<String> = row.get("decision_outcome");
        let decision_rationale: Option<String> = row.get("decision_rationale");
        let decision_action_attempts: Option<i64> = row.get("decision_action_attempts");
        let (Some(decision_cycle_id), Some(decision_version), Some(decision_outcome_raw)) =
            (decision_cycle_id, decision_version, decision_outcome_raw)
        else {
            close_review_failed_retry!(MandateFinalizationRejectReason::DecisionMissing);
        };
        if decision_version != mandate_version {
            tx.rollback().await?;
            return Ok(MandateRunFinalizationResult::Stale {
                reason: MandateFinalizationStaleReason::DecisionVersionChanged,
            });
        }
        let Some(decision_outcome) = MandateDecisionOutcome::parse(&decision_outcome_raw) else {
            close_invalid_decision_state!();
        };
        if decision_outcome == MandateDecisionOutcome::Wait
            && decision_rationale
                .as_deref()
                .is_some_and(crate::traits::is_runtime_fallback_rationale)
        {
            close_review_failed_retry!(MandateFinalizationRejectReason::DeliberatorFailed);
        }
        let Some(decision_action_attempts) = decision_action_attempts else {
            close_invalid_decision_state!();
        };
        let mandate_status_raw: String = row.get("mandate_status");
        let Some(mandate_status) = MandateStatus::parse(&mandate_status_raw) else {
            close_invalid_decision_state!();
        };
        let expected_status = MandateStatus::Active;
        if mandate_status != expected_status {
            tx.rollback().await?;
            return Ok(MandateRunFinalizationResult::Stale {
                reason: MandateFinalizationStaleReason::MandateMissingOrVersionChanged,
            });
        }

        let root_task_id: Option<String> = row.get("root_task_id");
        let root_success = if let Some(root_task_id) = root_task_id.as_deref() {
            sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM tasks
                 WHERE id = ? AND goal_run_id = ? AND status = 'completed'
                   AND COALESCE(NULLIF(trim(error), ''), '') = ''
                   AND COALESCE(NULLIF(trim(blocker), ''), '') = ''",
            )
            .bind(root_task_id)
            .bind(&request.goal_run_id)
            .fetch_one(&mut *tx)
            .await?
                == 1
        } else {
            false
        };

        let task_counts = sqlx::query(
            "SELECT COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN status = 'completed'
                        AND COALESCE(NULLIF(trim(error), ''), '') = ''
                        AND COALESCE(NULLIF(trim(blocker), ''), '') = ''
                        THEN 1 ELSE 0 END), 0) AS completed,
                    COALESCE(SUM(CASE WHEN status IN ('failed', 'blocked', 'interrupted')
                        OR COALESCE(NULLIF(trim(error), ''), '') != ''
                        OR COALESCE(NULLIF(trim(blocker), ''), '') != ''
                        THEN 1 ELSE 0 END), 0) AS failed_or_blocked
             FROM tasks WHERE goal_run_id = ? AND id != ?",
        )
        .bind(&request.goal_run_id)
        .bind(root_task_id.as_deref().unwrap_or(""))
        .fetch_one(&mut *tx)
        .await?;
        let non_root_tasks: i64 = task_counts.get("total");
        let completed_tasks: i64 = task_counts.get("completed");
        let failed_or_blocked_tasks: i64 = task_counts.get("failed_or_blocked");

        // Close any crash window between durable reservation, the one-use
        // final-dispatch claim, and strict receipt projection before proving
        // the run. Unclaimed rows prove no adapter I/O was authorized; claimed
        // rows without a receipt are necessarily ambiguous.
        sqlx::query(
            "UPDATE mandate_mutation_attempts
             SET status = CASE
                     WHEN dispatch_claimed_at IS NULL THEN 'never_dispatched'
                     ELSE 'ambiguous'
                 END,
                 completed_at = COALESCE(completed_at, ?)
             WHERE goal_run_id = ? AND mandate_id = ? AND status = 'reserved'",
        )
        .bind(&request.finalized_at)
        .bind(&request.goal_run_id)
        .bind(&request.mandate_id)
        .execute(&mut *tx)
        .await?;

        let mutation_counts = sqlx::query(
            "SELECT COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END), 0) AS succeeded,
                    COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed,
                    COALESCE(SUM(CASE WHEN status = 'never_dispatched' THEN 1 ELSE 0 END), 0)
                        AS never_dispatched,
                    COALESCE(SUM(CASE WHEN status = 'ambiguous' THEN 1 ELSE 0 END), 0)
                        AS ambiguous,
                    COALESCE(SUM(CASE WHEN mandate_version != ? OR decision_cycle_id != ?
                        THEN 1 ELSE 0 END), 0) AS mismatched
             FROM mandate_mutation_attempts
             WHERE goal_run_id = ? AND mandate_id = ?",
        )
        .bind(mandate_version)
        .bind(&decision_cycle_id)
        .bind(&request.goal_run_id)
        .bind(&request.mandate_id)
        .fetch_one(&mut *tx)
        .await?;
        let mutation_total: i64 = mutation_counts.get("total");
        let mutation_succeeded: i64 = mutation_counts.get("succeeded");
        let mutation_failed: i64 = mutation_counts.get("failed");
        let mutation_never_dispatched: i64 = mutation_counts.get("never_dispatched");
        let mutation_ambiguous: i64 = mutation_counts.get("ambiguous");
        let mutation_mismatched: i64 = mutation_counts.get("mismatched");
        let count = |value: i64| u32::try_from(value.max(0)).unwrap_or(u32::MAX);
        let counts = MandateRunProofCounts {
            non_root_tasks: count(non_root_tasks),
            completed_tasks: count(completed_tasks),
            incomplete_tasks: count(non_root_tasks.saturating_sub(completed_tasks)),
            failed_or_blocked_tasks: count(failed_or_blocked_tasks),
            mutation_reservations: count(mutation_total),
            succeeded_mutations: count(mutation_succeeded),
            failed_mutations: count(mutation_failed),
            never_dispatched_mutations: count(mutation_never_dispatched),
            ambiguous_or_reserved_mutations: count(mutation_ambiguous),
        };
        macro_rules! reconcile_fail_closed {
            ($reason:expr) => {{
                let reason = $reason;
                sqlx::query(
                    "UPDATE task_attempts
                     SET status = 'cancelled', completed_at = COALESCE(completed_at, ?)
                     WHERE goal_run_id = ? AND status IN ('claimed', 'running')",
                )
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE tasks
                     SET status = 'cancelled', current_attempt_id = NULL,
                         completed_at = COALESCE(completed_at, ?), updated_at = ?, version = version + 1
                     WHERE goal_run_id = ? AND status IN ('pending', 'claimed', 'running')",
                )
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let (intention_from, intention_to) = intention_transition(
                    IntentionStatus::Committed,
                    IntentionStatus::Suspended,
                )?;
                sqlx::query(
                    "UPDATE intentions
                     SET status = ?, updated_at = ?, completed_at = ?
                     WHERE goal_run_id = ? AND status = ?",
                )
                .bind(intention_to)
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .bind(intention_from)
                .execute(&mut *tx)
                .await?;
                let run_updated = sqlx::query(
                    "UPDATE goal_runs
                     SET status = 'failed', outcome_summary = ?, completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
                )
                .bind(format!("mandate_reconciliation_required:{}", reason.as_str()))
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                let mut suspension = MandateSuspension::new(
                    MandateSuspensionKind::ReconciliationRequired,
                    Some(reason.as_str().to_string()),
                );
                suspension.decision_cycle_id = Some(decision_cycle_id.clone());
                suspension.goal_run_id = Some(request.goal_run_id.clone());
                let mandate_updated = sqlx::query(
                    "UPDATE mandates
                     SET status = 'awaiting_input', review_lease_token = NULL,
                         review_lease_expires_at = NULL, suspension_json = ?, updated_at = ?
                     WHERE id = ? AND version = ? AND status = ?",
                )
                .bind(serde_json::to_string(&suspension)?)
                .bind(&request.finalized_at)
                .bind(&request.mandate_id)
                .bind(request.expected_mandate_version)
                .bind(expected_status.as_str())
                .execute(&mut *tx)
                .await?;
                if run_updated.rows_affected() != 1 || mandate_updated.rows_affected() != 1 {
                    tx.rollback().await?;
                    return Ok(MandateRunFinalizationResult::Stale {
                        reason: MandateFinalizationStaleReason::RunNotCurrent,
                    });
                }
                update_controller_status(
                    &mut tx,
                    &goal_id,
                    MandateStatus::AwaitingInput,
                    &request.finalized_at,
                )
                .await?;
                let notice = crate::traits::MandateRunNotification::new(
                    &request.mandate_id,
                    request.expected_mandate_version,
                    &goal_id,
                    &request.goal_run_id,
                    &owner_session_id,
                    crate::traits::MandateRunNotificationKind::ReconciliationRequired { reason },
                    counts.clone(),
                    &request.finalized_at,
                );
                super::notifications::enqueue_mandate_run_notification_on_connection(
                    &mut *tx,
                    &notice,
                )
                .await?;
                tx.commit().await?;
                return Ok(MandateRunFinalizationResult::ReconciliationRequired {
                    reason,
                    counts: counts.clone(),
                });
            }};
        }
        if !root_success {
            reconcile_fail_closed!(MandateReconciliationReason::RootTaskNotSuccessful);
        }

        match decision_outcome {
            MandateDecisionOutcome::Act => {
                let intention_id = sqlx::query_scalar::<_, String>(
                    "SELECT id FROM intentions
                     WHERE mandate_id = ? AND decision_cycle_id = ? AND goal_run_id = ?
                       AND status = 'committed' AND completed_at IS NULL",
                )
                .bind(&request.mandate_id)
                .bind(&decision_cycle_id)
                .bind(&request.goal_run_id)
                .fetch_optional(&mut *tx)
                .await?;
                let Some(intention_id) = intention_id else {
                    reconcile_fail_closed!(MandateReconciliationReason::ActMissingIntention);
                };
                if non_root_tasks == 0 {
                    reconcile_fail_closed!(MandateReconciliationReason::ActMissingWorkTask);
                }
                if completed_tasks != non_root_tasks || failed_or_blocked_tasks > 0 {
                    reconcile_fail_closed!(MandateReconciliationReason::WorkTasksIncomplete);
                }
                if mutation_mismatched > 0 || decision_action_attempts != mutation_total {
                    reconcile_fail_closed!(MandateReconciliationReason::ActionLedgerMismatch);
                }
                if mutation_failed > 0 {
                    reconcile_fail_closed!(MandateReconciliationReason::MutationOutcomeFailed);
                }
                if mutation_never_dispatched > 0 {
                    reconcile_fail_closed!(MandateReconciliationReason::MutationOutcomeFailed);
                }
                if mutation_ambiguous > 0 {
                    reconcile_fail_closed!(MandateReconciliationReason::MutationOutcomeAmbiguous);
                }
                if mutation_succeeded == 0 {
                    reconcile_fail_closed!(MandateReconciliationReason::ActMissingVerifiedMutation);
                }
                let (intention_from, intention_to) =
                    intention_transition(IntentionStatus::Committed, IntentionStatus::Satisfied)?;
                let intention_updated = sqlx::query(
                    "UPDATE intentions SET status = ?, updated_at = ?, completed_at = ?
                     WHERE id = ? AND status = ? AND completed_at IS NULL",
                )
                .bind(intention_to)
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&intention_id)
                .bind(intention_from)
                .execute(&mut *tx)
                .await?;
                if intention_updated.rows_affected() != 1 {
                    tx.rollback().await?;
                    return Ok(MandateRunFinalizationResult::Stale {
                        reason: MandateFinalizationStaleReason::RunNotCurrent,
                    });
                }
                let run_updated = sqlx::query(
                    "UPDATE goal_runs
                     SET status = 'completed', outcome_summary = 'mandate_act_satisfied',
                         completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
                )
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                if run_updated.rows_affected() != 1 {
                    tx.rollback().await?;
                    return Ok(MandateRunFinalizationResult::Stale {
                        reason: MandateFinalizationStaleReason::RunNotCurrent,
                    });
                }
                let notice = crate::traits::MandateRunNotification::new(
                    &request.mandate_id,
                    request.expected_mandate_version,
                    &goal_id,
                    &request.goal_run_id,
                    &owner_session_id,
                    crate::traits::MandateRunNotificationKind::ActSatisfied,
                    counts.clone(),
                    &request.finalized_at,
                );
                super::notifications::enqueue_mandate_run_notification_on_connection(
                    &mut tx, &notice,
                )
                .await?;
                sqlx::query("UPDATE goals SET dispatch_failures = 0, updated_at = ? WHERE id = ?")
                    .bind(&request.finalized_at)
                    .bind(&goal_id)
                    .execute(&mut *tx)
                    .await?;
                tx.commit().await?;
                Ok(MandateRunFinalizationResult::ActSatisfied { counts })
            }
            MandateDecisionOutcome::Wait
            | MandateDecisionOutcome::Ask
            | MandateDecisionOutcome::Stop => {
                if non_root_tasks != 0 {
                    reconcile_fail_closed!(MandateReconciliationReason::NonActionCreatedWork);
                }
                if mutation_total != 0 || decision_action_attempts != 0 {
                    reconcile_fail_closed!(MandateReconciliationReason::NonActionReservedMutation);
                }
                let intention_count: i64 = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM intentions
                     WHERE mandate_id = ? AND decision_cycle_id = ? AND goal_run_id = ?",
                )
                .bind(&request.mandate_id)
                .bind(&decision_cycle_id)
                .bind(&request.goal_run_id)
                .fetch_one(&mut *tx)
                .await?;
                if intention_count != 0 {
                    close_invalid_decision_state!();
                }
                let final_mandate_status = match decision_outcome {
                    MandateDecisionOutcome::Wait => MandateStatus::Active,
                    MandateDecisionOutcome::Ask => MandateStatus::AwaitingInput,
                    MandateDecisionOutcome::Stop => MandateStatus::Completed,
                    MandateDecisionOutcome::Act => unreachable!(),
                };
                let suspension = (decision_outcome == MandateDecisionOutcome::Ask).then(|| {
                    let mut value = MandateSuspension::new(
                        MandateSuspensionKind::AwaitingAnswer,
                        Some("agent_question".to_string()),
                    );
                    value.decision_cycle_id = Some(decision_cycle_id.clone());
                    value.goal_run_id = Some(request.goal_run_id.clone());
                    value
                });
                let mandate_updated = sqlx::query(
                    "UPDATE mandates
                     SET status = ?, review_lease_token = NULL,
                         review_lease_expires_at = NULL, suspension_json = ?, updated_at = ?
                     WHERE id = ? AND version = ? AND status = 'active'",
                )
                .bind(final_mandate_status.as_str())
                .bind(suspension.as_ref().map(serde_json::to_string).transpose()?)
                .bind(&request.finalized_at)
                .bind(&request.mandate_id)
                .bind(request.expected_mandate_version)
                .execute(&mut *tx)
                .await?;
                if mandate_updated.rows_affected() != 1 {
                    tx.rollback().await?;
                    return Ok(MandateRunFinalizationResult::Stale {
                        reason: MandateFinalizationStaleReason::MandateMissingOrVersionChanged,
                    });
                }
                update_controller_status(
                    &mut tx,
                    &goal_id,
                    final_mandate_status,
                    &request.finalized_at,
                )
                .await?;
                sqlx::query("UPDATE goals SET dispatch_failures = 0, updated_at = ? WHERE id = ?")
                    .bind(&request.finalized_at)
                    .bind(&goal_id)
                    .execute(&mut *tx)
                    .await?;
                let run_updated = sqlx::query(
                    "UPDATE goal_runs
                     SET status = 'completed', outcome_summary = 'mandate_non_action_satisfied',
                         completed_at = ?, updated_at = ?
                     WHERE id = ? AND status = 'running' AND trigger_type = 'mandate'",
                )
                .bind(&request.finalized_at)
                .bind(&request.finalized_at)
                .bind(&request.goal_run_id)
                .execute(&mut *tx)
                .await?;
                if run_updated.rows_affected() != 1 {
                    tx.rollback().await?;
                    return Ok(MandateRunFinalizationResult::Stale {
                        reason: MandateFinalizationStaleReason::RunNotCurrent,
                    });
                }
                if let Some(kind) = match decision_outcome {
                    MandateDecisionOutcome::Wait => None,
                    MandateDecisionOutcome::Ask => {
                        Some(crate::traits::MandateRunNotificationKind::Ask)
                    }
                    MandateDecisionOutcome::Stop => {
                        Some(crate::traits::MandateRunNotificationKind::Stopped)
                    }
                    MandateDecisionOutcome::Act => unreachable!(),
                } {
                    let notice = crate::traits::MandateRunNotification::new(
                        &request.mandate_id,
                        request.expected_mandate_version,
                        &goal_id,
                        &request.goal_run_id,
                        &owner_session_id,
                        kind,
                        counts.clone(),
                        &request.finalized_at,
                    );
                    super::notifications::enqueue_mandate_run_notification_on_connection(
                        &mut tx, &notice,
                    )
                    .await?;
                }
                tx.commit().await?;
                Ok(MandateRunFinalizationResult::NonActionSatisfied {
                    outcome: decision_outcome,
                    counts,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::traits::store_prelude::*;
    use crate::traits::{
        Goal, Intention, Mandate, MandateAuthority, MandateDecisionCycle, MandateMutationTarget,
        MandateObjectiveControl, MandateOperationKind, MandateOperationScope,
        MandateWakeSignalKind, ObjectiveMetricDirection, Task, ToolCallOperation,
    };
    use std::sync::Arc;

    async fn test_store() -> (SqliteStateStore, tempfile::NamedTempFile) {
        let database = tempfile::NamedTempFile::new().unwrap();
        let store = SqliteStateStore::new(
            database.path().to_str().unwrap(),
            100,
            None,
            Arc::new(EmbeddingService::new().unwrap()),
        )
        .await
        .unwrap();
        (store, database)
    }

    fn authority(max_actions: u32) -> MandateAuthority {
        MandateAuthority {
            allowed_tools: vec!["http_request".to_string()],
            allowed_mutation_effects: vec!["external_delivery".to_string()],
            allowed_target_prefixes: vec!["https://api.x.com/2/".to_string()],
            max_mutating_actions_per_cycle: max_actions,
            max_mutating_actions_per_rolling_24h: if max_actions == 0 { 0 } else { 24 },
            min_seconds_between_mutations: if max_actions == 0 { 0 } else { 900 },
            ..MandateAuthority::default()
        }
    }

    fn controller(session_id: &str, max_actions: u32) -> (Goal, Mandate) {
        let goal = Goal::new_continuous(
            "Steward the account",
            session_id,
            Some(10_000),
            Some(50_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Maintain a useful, authentic account presence",
            session_id,
            authority(max_actions),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
        (goal, mandate)
    }

    fn objective_control() -> MandateObjectiveControl {
        MandateObjectiveControl {
            schema_version: MandateObjectiveControl::SCHEMA_VERSION,
            metric_name: "synthetic useful interactions".to_string(),
            unit: "count".to_string(),
            baseline_micros: 10_000_000,
            target_micros: 20_000_000,
            direction: ObjectiveMetricDirection::AtLeast,
            measurement_source: "metric_source:synthetic-analytics".to_string(),
            measurement_cadence_secs: 3_600,
            experiment_cohort: "synthetic-cohort-a".to_string(),
            experiment_window_secs: 86_400,
            minimum_effect_micros: 1_000_000,
            max_stagnant_measurements: 3,
            run_failure_budget: 3,
            baseline_observed_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    #[tokio::test]
    async fn startup_quarantines_legacy_autopilot_before_any_lease_claim() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 0);
        mandate.autonomy_mode = MandateAutonomyMode::Autopilot;
        mandate.objective_control = Some(objective_control());
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        // Reproduce a legacy active row without the now-mandatory controller.
        sqlx::query("UPDATE mandates SET objective_control_json = NULL WHERE id = ?")
            .bind(&mandate.id)
            .execute(&store.pool)
            .await
            .unwrap();

        enforce_mandate_invariants_on_startup(&store.pool)
            .await
            .unwrap();

        let quarantined = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(quarantined.status, MandateStatus::AwaitingInput);
        assert!(quarantined.objective_control.is_none());
        assert_eq!(
            quarantined.suspension.as_ref().map(|value| value.kind),
            Some(MandateSuspensionKind::ObjectiveControlRequired)
        );
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "paused"
        );
    }

    #[tokio::test]
    async fn due_claim_quarantines_legacy_autopilot_without_inventing_objective_control() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 0);
        mandate.autonomy_mode = MandateAutonomyMode::Autopilot;
        mandate.objective_control = Some(objective_control());
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        // Simulate a row created before objective_control became mandatory.
        sqlx::query("UPDATE mandates SET objective_control_json = NULL WHERE id = ?")
            .bind(&mandate.id)
            .execute(&store.pool)
            .await
            .unwrap();

        assert!(store
            .claim_due_mandates(10, "synthetic-heartbeat", 300)
            .await
            .unwrap()
            .is_empty());
        let mut quarantined = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(quarantined.status, MandateStatus::AwaitingInput);
        assert!(quarantined.objective_control.is_none());
        assert_eq!(
            quarantined.suspension.as_ref().map(|value| value.kind),
            Some(MandateSuspensionKind::ObjectiveControlRequired)
        );
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "paused"
        );
        let notices = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM notification_queue
             WHERE goal_id = ? AND notification_type = 'mandate_objective_control_required'",
        )
        .bind(&goal.id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(notices, 1);

        quarantined.objective_control = Some(objective_control());
        quarantined.version += 1;
        store.update_mandate(&quarantined).await.unwrap();
        assert!(store
            .resolve_mandate_suspension(
                &mandate.id,
                quarantined.version,
                MandateSuspensionKind::ObjectiveControlRequired,
                None,
                None,
                "Owner configured the validated objective control required for autopilot.",
                "owner-session",
            )
            .await
            .unwrap());
        let resumed = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(resumed.status, MandateStatus::Active);
        assert!(resumed.objective_control.is_some());
        assert!(resumed.suspension.is_none());
    }

    #[tokio::test]
    async fn structured_signal_wakes_only_matching_autopilot_once() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 0);
        mandate.autonomy_mode = MandateAutonomyMode::Autopilot;
        mandate.objective_control = Some(objective_control());
        mandate.authority = MandateAuthority::from_operation_scopes(
            true,
            vec![MandateOperationScope {
                tool: "http_request".to_string(),
                operation: ToolCallOperation::Get,
                kind: MandateOperationKind::Observation,
                target_prefixes: vec![
                    "https://api.example.test/v1/mentions".to_string(),
                    "auth_profile:synthetic-social".to_string(),
                    "account:synthetic-1".to_string(),
                ],
                allowed_query_params: vec!["since_id".to_string()],
                mutation_effects: Vec::new(),
            }],
            0,
            0,
            0,
        );
        mandate.next_review_at = (chrono::Utc::now() + chrono::Duration::hours(3)).to_rfc3339();
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let signal = MandateWakeSignal {
            kind: MandateWakeSignalKind::Mention,
            source: "synthetic_social".to_string(),
            target_url: "https://api.example.test/v1/mentions?since_id=42".to_string(),
            account_id: Some("account:synthetic-1".to_string()),
            dedupe_key: "mention:42".to_string(),
            occurred_at: chrono::Utc::now().to_rfc3339(),
        };

        assert_eq!(
            store.wake_mandates_for_signal(&signal).await.unwrap(),
            vec![mandate.id.clone()]
        );
        let awakened = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert!(
            chrono::DateTime::parse_from_rfc3339(&awakened.next_review_at).unwrap()
                <= chrono::Utc::now() + chrono::Duration::seconds(1)
        );
        assert!(store
            .wake_mandates_for_signal(&signal)
            .await
            .unwrap()
            .is_empty());

        let mut wrong_account = signal;
        wrong_account.dedupe_key = "mention:43".to_string();
        wrong_account.account_id = Some("account:synthetic-2".to_string());
        assert!(store
            .wake_mandates_for_signal(&wrong_account)
            .await
            .unwrap()
            .is_empty());
    }

    fn review_root(goal: &Goal, run_id: &str, task_id: &str) -> Task {
        Task {
            id: task_id.to_string(),
            goal_id: goal.id.clone(),
            description: format!("Mandate review: test controller for goal run {run_id}"),
            status: "pending".to_string(),
            priority: goal.priority.clone(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        }
    }

    async fn claim_and_start_run(
        store: &SqliteStateStore,
        goal: &Goal,
        mandate: &Mandate,
    ) -> crate::traits::GoalRun {
        store
            .create_mandate_controller(goal, mandate)
            .await
            .unwrap();
        let claimed = store
            .claim_due_mandates(1, "test-heartbeat", 300)
            .await
            .unwrap();
        assert_eq!(claimed.len(), 1);
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let run = store
            .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
            .await
            .unwrap();
        store
            .create_task(&review_root(goal, &run.id, &root_task_id))
            .await
            .unwrap();
        // Goal runs now remain pending until a worker claim transitions them.
        // These storage tests exercise post-claim mandate invariants without a
        // task worker, so model that already-claimed state explicitly.
        sqlx::query("UPDATE goal_runs SET status = 'running' WHERE id = ?")
            .bind(&run.id)
            .execute(&store.pool)
            .await
            .unwrap();
        run
    }

    async fn start_existing_claimed_run(
        store: &SqliteStateStore,
        goal: &Goal,
        mandate: &Mandate,
    ) -> GoalRun {
        let claimed = store
            .claim_due_mandates(1, "test-heartbeat-next", 300)
            .await
            .unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].id, mandate.id);
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let run = store
            .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
            .await
            .unwrap();
        store
            .create_task(&review_root(goal, &run.id, &root_task_id))
            .await
            .unwrap();
        sqlx::query("UPDATE goal_runs SET status = 'running' WHERE id = ?")
            .bind(&run.id)
            .execute(&store.pool)
            .await
            .unwrap();
        run
    }

    struct MutationFence {
        root_attempt: crate::traits::TaskAttempt,
        worker_task: Task,
        worker_attempt: crate::traits::TaskAttempt,
    }

    async fn claim_mutation_fence(
        store: &SqliteStateStore,
        goal: &Goal,
        mandate: &Mandate,
        run: &GoalRun,
    ) -> MutationFence {
        let root_task_id = run.root_task_id.as_deref().unwrap();
        let root_attempt = store
            .claim_task_with_lease(
                root_task_id,
                "root-worker",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        let worker_task_id = uuid::Uuid::new_v4().to_string();
        let mut worker_task = review_root(goal, &run.id, &worker_task_id);
        worker_task.description = "Perform one governed HTTP mutation".to_string();
        worker_task.task_order = 1;
        assert!(store
            .create_mandate_task_from_attempt(
                &worker_task,
                &mandate.id,
                mandate.version,
                &run.id,
                &root_attempt.id,
                16,
            )
            .await
            .unwrap());
        let worker_attempt = store
            .claim_mandate_task_from_attempt(
                &worker_task_id,
                "executor-worker",
                &mandate.id,
                mandate.version,
                &run.id,
                &root_attempt.id,
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        MutationFence {
            root_attempt,
            worker_task,
            worker_attempt,
        }
    }

    fn reservation(
        mandate: &Mandate,
        decision: &MandateDecisionCycle,
        run: &GoalRun,
        fence: &MutationFence,
        reserved_action_attempt: i64,
        sequence: u64,
        reserved_at: chrono::DateTime<chrono::Utc>,
    ) -> MandateMutationReservation {
        MandateMutationReservation {
            grant: crate::traits::MandateAuthorityGrant {
                mandate_id: mandate.id.clone(),
                mandate_version: mandate.version,
                owner_principal_id: mandate.owner_principal_id.clone(),
                decision_cycle_id: decision.id.clone(),
                action_digest: format!("{sequence:064x}"),
                counts_toward_cycle_budget: true,
                reserved_action_attempt,
                tool_call_id: Some(format!("tool-call-{sequence}")),
            },
            goal_run_id: run.id.clone(),
            root_task_id: run.root_task_id.clone().unwrap(),
            root_task_attempt_id: fence.root_attempt.id.clone(),
            task_id: fence.worker_task.id.clone(),
            task_attempt_id: fence.worker_attempt.id.clone(),
            tool_call_id: format!("tool-call-{sequence}"),
            tool_name: "http_request".to_string(),
            mutation_effects: vec!["external_delivery".to_string()],
            targets: vec![MandateMutationTarget {
                kind: "url".to_string(),
                identifier: "https://api.x.com/2/tweets".to_string(),
            }],
            account_identifiers: Vec::new(),
            reserved_at: reserved_at.to_rfc3339(),
        }
    }

    fn outcome_projection(
        reservation: &MandateMutationReservation,
        status: MandateMutationAttemptStatus,
        http_status: Option<u16>,
        completed_at: chrono::DateTime<chrono::Utc>,
    ) -> MandateMutationOutcomeProjection {
        MandateMutationOutcomeProjection {
            grant: reservation.grant.clone(),
            goal_run_id: reservation.goal_run_id.clone(),
            task_id: reservation.task_id.clone(),
            task_attempt_id: reservation.task_attempt_id.clone(),
            tool_call_id: reservation.tool_call_id.clone(),
            status,
            receipt_schema_version: crate::events::ToolReceiptV1::SCHEMA_VERSION,
            outcome_evidence: Some(MandateMutationEvidence::StructuredMetadata),
            timed_out: false,
            background_started: false,
            detached: false,
            completion_notifications_enabled: false,
            transport_error_present: false,
            semantics_match: true,
            http_status,
            exit_code: None,
            completed_at: completed_at.to_rfc3339(),
        }
    }

    fn dispatch_claim(
        reservation: &MandateMutationReservation,
        claimed_at: chrono::DateTime<chrono::Utc>,
    ) -> MandateMutationDispatchClaim {
        MandateMutationDispatchClaim {
            grant: reservation.grant.clone(),
            goal_run_id: reservation.goal_run_id.clone(),
            root_task_id: reservation.root_task_id.clone(),
            root_task_attempt_id: reservation.root_task_attempt_id.clone(),
            task_id: reservation.task_id.clone(),
            task_attempt_id: reservation.task_attempt_id.clone(),
            tool_call_id: reservation.tool_call_id.clone(),
            tool_name: reservation.tool_name.clone(),
            claimed_at: claimed_at.to_rfc3339(),
        }
    }

    async fn complete_mutation_fence(store: &SqliteStateStore, fence: &MutationFence) {
        let complete = crate::traits::TaskAttemptPatch {
            status: "completed".to_string(),
            result: Some("verified".to_string()),
            ..Default::default()
        };
        assert!(store
            .patch_task_from_attempt(
                &fence.worker_attempt.id,
                &fence.worker_attempt.lease_token,
                &complete,
            )
            .await
            .unwrap());
        assert!(store
            .patch_task_from_attempt(
                &fence.root_attempt.id,
                &fence.root_attempt.lease_token,
                &complete,
            )
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn controller_and_mandate_are_created_atomically() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 2);
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let loaded = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(loaded.goal_id, goal.id);
        assert_eq!(loaded.version, 1);
        assert!(loaded.review_lease_token.is_none());
        assert!(store.get_goal(&goal.id).await.unwrap().is_some());
        let channel_links: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM work_channel_links WHERE goal_id = ? AND channel_session_id = ?",
        )
        .bind(&goal.id)
        .bind(&goal.session_id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(channel_links, 1);

        let pending_goal = Goal::new_continuous_pending(
            "Mandate awaiting confirmation",
            "owner-session",
            None,
            None,
        );
        let mut pending_mandate = Mandate::new(
            &pending_goal.id,
            None,
            "Await explicit owner confirmation",
            "owner-session",
            authority(0),
            60,
            3_600,
            300,
        );
        pending_mandate.status = MandateStatus::Paused;
        pending_mandate.confirmed_at = None;
        store
            .create_mandate_controller(&pending_goal, &pending_mandate)
            .await
            .unwrap();
        assert_eq!(
            store
                .get_goal(&pending_goal.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            "pending_confirmation"
        );
        assert_eq!(
            store
                .get_mandate(&pending_mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::Paused
        );

        let (bad_goal, mut bad_mandate) = controller("owner-session", 2);
        bad_mandate.source_goal_id = Some("missing-personal-goal".to_string());
        assert!(store
            .create_mandate_controller(&bad_goal, &bad_mandate)
            .await
            .is_err());
        assert!(store.get_goal(&bad_goal.id).await.unwrap().is_none());

        let same_owner_source = Goal::new_personal("Grow an audience", "owner-session");
        store.create_goal(&same_owner_source).await.unwrap();
        let (linked_goal, mut linked_mandate) = controller("owner-session", 2);
        linked_mandate.source_goal_id = Some(same_owner_source.id);
        store
            .create_mandate_controller(&linked_goal, &linked_mandate)
            .await
            .unwrap();
        assert!(store.get_goal(&linked_goal.id).await.unwrap().is_some());

        let foreign_source = Goal::new_personal("Grow an audience", "different-owner-session");
        store.create_goal(&foreign_source).await.unwrap();
        let (foreign_goal, mut foreign_mandate) = controller("owner-session", 2);
        foreign_mandate.source_goal_id = Some(foreign_source.id);
        assert!(store
            .create_mandate_controller(&foreign_goal, &foreign_mandate)
            .await
            .is_err());
        assert!(store.get_goal(&foreign_goal.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn verified_private_owner_is_authorized_across_bot_routes() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("alpha_bot:12345", 1);
        assert_eq!(mandate.owner_principal_id, "principal:telegram:12345");
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        assert!(store
            .is_mandate_session_authorized(&mandate.id, "beta_bot:12345")
            .await
            .unwrap());
        assert_eq!(
            store
                .list_mandates(Some("beta_bot:12345"), false)
                .await
                .unwrap()
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            [mandate.id.as_str()]
        );
        assert!(!store
            .is_mandate_session_authorized(&mandate.id, "beta_bot:67890")
            .await
            .unwrap());

        let (group_goal, group_mandate) = controller("alpha_bot:-10012345", 1);
        assert_ne!(
            group_mandate.owner_principal_id,
            "principal:telegram:-10012345"
        );
        store
            .create_mandate_controller(&group_goal, &group_mandate)
            .await
            .unwrap();
        assert!(!store
            .is_mandate_session_authorized(&group_mandate.id, "beta_bot:-10012345")
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn ownership_transfer_preserves_principal_and_moves_runtime_route_atomically() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("telegram:synthetic-owner-a", 1);
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        assert!(store
            .transfer_mandate_ownership(
                &mandate.id,
                mandate.version,
                "telegram:synthetic-owner-a",
                "telegram:synthetic-owner-b",
            )
            .await
            .unwrap());
        let moved = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(moved.owner_principal_id, mandate.owner_principal_id);
        assert_eq!(moved.created_by_session, "telegram:synthetic-owner-b");
        assert_eq!(moved.version, mandate.version + 1);
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().session_id,
            "telegram:synthetic-owner-b"
        );
        assert!(store
            .is_mandate_session_authorized(&mandate.id, "telegram:synthetic-owner-b")
            .await
            .unwrap());
        assert!(!store
            .transfer_mandate_ownership(
                &mandate.id,
                mandate.version,
                "telegram:synthetic-owner-a",
                "telegram:synthetic-owner-c",
            )
            .await
            .unwrap());
        let audit_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM mandate_ownership_transfers WHERE mandate_id = ?",
        )
        .bind(&mandate.id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(audit_count, 1);
    }

    #[tokio::test]
    async fn objective_measurements_require_current_run_structured_receipts() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 1);
        mandate.objective_control = Some(objective_control());
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let receipt_id = "synthetic-metric-receipt";
        let task_id = run.root_task_id.as_deref().unwrap();
        sqlx::query(
            "INSERT INTO events
                (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES (?, 'tool_result', ?, ?, ?, 'http_request')",
        )
        .bind("owner-session")
        .bind(
            serde_json::json!({
                "task_id": task_id,
                "tool_call_id": receipt_id,
                "name": "http_request",
                "result": "synthetic metric value",
                "success": true,
                "duration_ms": 1,
                "receipt": {
                    "schema_version": crate::events::ToolReceiptV1::SCHEMA_VERSION,
                    "outcome_status": "succeeded",
                    "outcome_evidence": "structured_metadata"
                }
            })
            .to_string(),
        )
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(task_id)
        .execute(&store.pool)
        .await
        .unwrap();

        let measurement = MandateObjectiveMeasurement::new(
            &mandate.id,
            mandate.version,
            &run.id,
            12_000_000,
            9_500,
            vec![receipt_id.to_string()],
            &chrono::Utc::now().to_rfc3339(),
        );
        store
            .record_mandate_objective_measurement(&measurement)
            .await
            .unwrap();
        assert_eq!(
            store
                .list_mandate_objective_measurements(&mandate.id, 10)
                .await
                .unwrap(),
            vec![measurement]
        );

        let unproven = MandateObjectiveMeasurement::new(
            &mandate.id,
            mandate.version,
            &run.id,
            13_000_000,
            9_500,
            vec!["receipt-from-another-run".to_string()],
            &chrono::Utc::now().to_rfc3339(),
        );
        assert!(store
            .record_mandate_objective_measurement(&unproven)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn persistence_rejects_text_bound_bypasses_before_writing() {
        let (store, _database) = test_store().await;
        let (bad_goal, mut bad_mandate) = controller("owner-session", 2);
        bad_mandate.objective = "é".repeat(1_025);
        assert!(store
            .create_mandate_controller(&bad_goal, &bad_mandate)
            .await
            .is_err());
        assert!(store.get_goal(&bad_goal.id).await.unwrap().is_none());

        let (goal, mandate) = controller("owner-session", 2);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let oversized_decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Wait,
            &"é".repeat(1_025),
            mandate.version,
        );
        assert!(store
            .record_mandate_decision(&oversized_decision, None, None)
            .await
            .is_err());

        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "A bounded governed action is useful",
            mandate.version,
        );
        let oversized_intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            &"é".repeat(513),
            "A bounded governed action is useful",
        );
        assert!(store
            .record_mandate_decision(&act, Some(&oversized_intention), None)
            .await
            .is_err());
        assert!(store
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn owner_update_is_cas_and_clears_a_live_lease() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 2);
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        assert_eq!(
            store
                .claim_due_mandates(1, "heartbeat", 300)
                .await
                .unwrap()
                .len(),
            1
        );

        let mut update = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        update.version += 1;
        update.objective = "Updated owner objective".to_string();
        store.update_mandate(&update).await.unwrap();

        let loaded = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(loaded.version, 2);
        assert_eq!(loaded.objective, "Updated owner objective");
        assert!(loaded.review_lease_token.is_none());
        assert!(store.update_mandate(&update).await.is_err());

        let mut attempted_pause = loaded;
        attempted_pause.version += 1;
        attempted_pause.status = MandateStatus::Paused;
        assert!(store.update_mandate(&attempted_pause).await.is_err());
        assert!(store
            .transition_mandate_status(&mandate.id, MandateStatus::Active, MandateStatus::Paused,)
            .await
            .unwrap());
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "paused"
        );
    }

    #[tokio::test]
    async fn confirmation_is_durable_atomic_and_required_for_activation() {
        let (store, _database) = test_store().await;
        let goal = Goal::new_continuous_pending(
            "Mandate awaiting owner confirmation",
            "owner-session",
            None,
            None,
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Await explicit owner confirmation",
            "owner-session",
            authority(0),
            60,
            3_600,
            300,
        );
        mandate.status = MandateStatus::Paused;
        mandate.confirmed_at = None;
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        assert!(!store
            .transition_mandate_status(&mandate.id, MandateStatus::Paused, MandateStatus::Active,)
            .await
            .unwrap());
        // Confirmation is a CAS against the exact policy version displayed to
        // the owner; a stale or speculative callback cannot activate it.
        assert!(!store
            .confirm_mandate(&mandate.id, mandate.version + 1, None)
            .await
            .unwrap());
        assert_eq!(
            store
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::Paused
        );
        assert!(store
            .confirm_mandate(&mandate.id, mandate.version, None)
            .await
            .unwrap());
        assert!(!store
            .confirm_mandate(&mandate.id, mandate.version, None)
            .await
            .unwrap());

        let confirmed = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(confirmed.status, MandateStatus::Active);
        assert_eq!(confirmed.version, 2);
        assert!(confirmed.confirmed_at.is_some());
        assert!(confirmed.is_active());
        let controller = store.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(controller.status, "active");
        assert_eq!(controller.dispatch_failures, 0);
    }

    #[tokio::test]
    async fn stale_confirmation_cleanup_cancels_linked_mandate_and_cannot_activate_it() {
        let (store, _database) = test_store().await;
        let goal =
            Goal::new_continuous_pending("Stale mandate confirmation", "owner-session", None, None);
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Never activate without confirmation",
            "owner-session",
            authority(0),
            60,
            3_600,
            300,
        );
        mandate.status = MandateStatus::Paused;
        mandate.confirmed_at = None;
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        sqlx::query("UPDATE goals SET created_at = ? WHERE id = ?")
            .bind((chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339())
            .bind(&goal.id)
            .execute(&store.pool)
            .await
            .unwrap();

        assert_eq!(
            store
                .cancel_stale_pending_confirmation_goals(3_600)
                .await
                .unwrap(),
            1
        );
        let cancelled = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(cancelled.status, MandateStatus::Cancelled);
        assert!(cancelled.confirmed_at.is_none());
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "cancelled"
        );
        assert!(!store
            .confirm_mandate(&mandate.id, mandate.version, None)
            .await
            .unwrap());
        assert!(
            !store
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Cancelled,
                    MandateStatus::Active,
                )
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn atomic_review_creation_rolls_back_run_when_root_insert_fails() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let claimed = store
            .claim_due_mandates(1, "test-heartbeat", 300)
            .await
            .unwrap();
        let lease = claimed[0].review_lease_token.as_deref().unwrap();

        let other_goal = Goal::new_finite("Existing task owner", "owner-session");
        store.create_goal(&other_goal).await.unwrap();
        let existing_task = review_root(&other_goal, "other-run", "duplicate-root-id");
        store.create_task(&existing_task).await.unwrap();

        let attempted_run_id = uuid::Uuid::new_v4().to_string();
        let duplicate_root = review_root(&goal, &attempted_run_id, "duplicate-root-id");
        assert!(store
            .create_mandate_review_run(&mandate.id, lease, &attempted_run_id, &duplicate_root,)
            .await
            .is_err());
        let orphan_count =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM goal_runs WHERE id = ?")
                .bind(&attempted_run_id)
                .fetch_one(&store.pool)
                .await
                .unwrap();
        assert_eq!(orphan_count, 0);

        let successful_run_id = uuid::Uuid::new_v4().to_string();
        let fresh_root = review_root(&goal, &successful_run_id, &uuid::Uuid::new_v4().to_string());
        let run = store
            .create_mandate_review_run(&mandate.id, lease, &successful_run_id, &fresh_root)
            .await
            .unwrap();
        assert_eq!(run.root_task_id.as_deref(), Some(fresh_root.id.as_str()));
        assert_eq!(
            store.get_tasks_for_goal_run(&run.id).await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn claiming_recovers_pre_atomic_run_created_without_a_root_task() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let first_claim = store
            .claim_due_mandates(1, "old-heartbeat", 300)
            .await
            .unwrap();
        assert_eq!(first_claim.len(), 1);
        let interrupted = store
            .start_goal_run(&goal.id, "mandate", None, Some("missing-root"))
            .await
            .unwrap();

        let recovered = store
            .claim_due_mandates(1, "new-heartbeat", 300)
            .await
            .unwrap();
        assert_eq!(recovered.len(), 1);
        let old_run = store
            .get_goal_runs(&goal.id)
            .await
            .unwrap()
            .into_iter()
            .find(|run| run.id == interrupted.id)
            .unwrap();
        assert_eq!(old_run.status, "failed");
        assert!(old_run
            .outcome_summary
            .as_deref()
            .unwrap()
            .contains("interrupted mandate dispatch"));

        let run_id = uuid::Uuid::new_v4().to_string();
        let root = review_root(&goal, &run_id, &uuid::Uuid::new_v4().to_string());
        let new_run = store
            .create_mandate_review_run(
                &mandate.id,
                recovered[0].review_lease_token.as_deref().unwrap(),
                &run_id,
                &root,
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .get_tasks_for_goal_run(&new_run.id)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn completed_root_crash_reconciles_after_grace_and_cancels_pending_child() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let root_task_id = run.root_task_id.as_deref().unwrap();
        let root_attempt = store
            .claim_task_with_lease(
                root_task_id,
                "crashing-root",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "One bounded child should perform the action",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Perform the bounded child action",
            "It advances the mandate",
        );
        store
            .record_mandate_decision(&act, Some(&intention), Some(&root_attempt.id))
            .await
            .unwrap();
        let child_id = uuid::Uuid::new_v4().to_string();
        let mut child = review_root(&goal, &run.id, &child_id);
        child.task_order = 1;
        child.description = "Pending governed child".to_string();
        assert!(store
            .create_mandate_task_from_attempt(
                &child,
                &mandate.id,
                mandate.version,
                &run.id,
                &root_attempt.id,
                16,
            )
            .await
            .unwrap());
        assert!(store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("decision recorded; child queued".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());

        // The immediate finalizer gets a short uncontended window.
        assert!(store
            .claim_due_mandates(1, "grace-probe", 300)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            store
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::Active
        );
        assert_eq!(
            store.get_task(&child_id).await.unwrap().unwrap().status,
            "pending"
        );

        let stale_completion = (chrono::Utc::now() - chrono::Duration::seconds(61)).to_rfc3339();
        sqlx::query("UPDATE tasks SET completed_at = ? WHERE id = ?")
            .bind(&stale_completion)
            .bind(root_task_id)
            .execute(&store.pool)
            .await
            .unwrap();
        assert!(store
            .claim_due_mandates(1, "reconciliation-probe", 300)
            .await
            .unwrap()
            .is_empty());

        assert_eq!(
            store
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::AwaitingInput
        );
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "paused"
        );
        assert_eq!(
            store.get_task(&child_id).await.unwrap().unwrap().status,
            "cancelled"
        );
        assert_eq!(
            store.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "cancelled"
        );
        let notification_message: String = sqlx::query_scalar(
            "SELECT message FROM notification_queue
             WHERE goal_id = ? AND notification_type = 'mandate_reconciliation_required'
               AND priority = 'critical' AND expires_at IS NULL
             LIMIT 1",
        )
        .bind(&goal.id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert!(notification_message.contains("work_tasks=1"));
        assert!(notification_message.contains("mutation_reservations=0"));
    }

    #[tokio::test]
    async fn orphaned_review_without_decision_retries_without_reconciliation_pause() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let root_attempt = store
            .claim_task_with_lease(
                run.root_task_id.as_deref().unwrap(),
                "crashing-reviewer",
                Some("profile-task-lead"),
                180,
            )
            .await
            .unwrap()
            .unwrap();
        sqlx::query("UPDATE task_attempts SET lease_expires_at = ? WHERE id = ?")
            .bind((chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339())
            .bind(&root_attempt.id)
            .execute(&store.pool)
            .await
            .unwrap();

        assert!(store
            .claim_due_mandates(1, "orphan-recovery", 300)
            .await
            .unwrap()
            .is_empty());

        let recovered_mandate = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(recovered_mandate.status, MandateStatus::Active);
        assert!(recovered_mandate.next_review_at > chrono::Utc::now().to_rfc3339());
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
        let recovered_run = store
            .get_goal_runs(&goal.id)
            .await
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.id == run.id)
            .unwrap();
        assert_eq!(recovered_run.status, "failed");
        assert_eq!(
            recovered_run.outcome_summary.as_deref(),
            Some("mandate_review_failed:decision_missing")
        );

        let root = store
            .get_task(run.root_task_id.as_deref().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(root.status, "cancelled");
        let notification: (String, String) = sqlx::query_as(
            "SELECT notification_type, message FROM notification_queue
             WHERE id = ?",
        )
        .bind(format!("mandate-run-notice:{}", run.id))
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(notification.0, "mandate_review_failed");
        assert!(notification.1.contains("reason=decision_missing"));
        assert!(notification
            .1
            .contains("No action was authorized or executed"));
        assert!(!notification.1.contains("Inspect the external target"));
    }

    #[tokio::test]
    async fn stale_controller_repair_cannot_revive_an_owner_cancellation() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        assert!(
            store
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Active,
                    MandateStatus::Cancelled,
                )
                .await
                .unwrap()
        );

        assert!(!store
            .keep_mandate_controller_active(&mandate.id, mandate.version)
            .await
            .unwrap());
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "cancelled"
        );
    }

    #[tokio::test]
    async fn lifecycle_transition_revokes_open_cycle_and_stale_owner_snapshot() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 2);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "A bounded action was justified before the pause",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Perform one bounded action",
            "It advances the mandate",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;

        let mut stale_update = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        stale_update.version += 1;
        stale_update.objective = "A stale owner snapshot".to_string();

        assert!(store
            .transition_mandate_status(&mandate.id, MandateStatus::Active, MandateStatus::Paused,)
            .await
            .unwrap());
        let paused = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(paused.status, MandateStatus::Paused);
        assert_eq!(paused.version, mandate.version + 1);
        assert!(store
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            store.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "cancelled"
        );
        assert_eq!(
            store.list_intentions(&mandate.id, 1).await.unwrap()[0].status,
            IntentionStatus::Suspended
        );
        assert!(store
            .reserve_mandate_action_attempt(&reservation(
                &mandate,
                &act,
                &run,
                &fence,
                1,
                1,
                chrono::Utc::now(),
            ))
            .await
            .unwrap()
            .is_none());

        assert!(store
            .transition_mandate_status(&mandate.id, MandateStatus::Paused, MandateStatus::Active,)
            .await
            .unwrap());
        let resumed = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(resumed.version, mandate.version + 2);
        assert!(store
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .is_none());

        // The update was prepared before the lifecycle epoch changed, so its
        // CAS must fail instead of restoring its stale active status/policy.
        assert!(store.update_mandate(&stale_update).await.is_err());

        let mut stale_cancel_racer = resumed;
        stale_cancel_racer.version += 1;
        stale_cancel_racer.objective = "Must not resurrect after cancel".to_string();
        assert!(
            store
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Active,
                    MandateStatus::Cancelled,
                )
                .await
                .unwrap()
        );
        assert!(store.update_mandate(&stale_cancel_racer).await.is_err());
        let cancelled = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(cancelled.status, MandateStatus::Cancelled);
        assert_eq!(cancelled.version, mandate.version + 3);
    }

    #[tokio::test]
    async fn policy_update_marks_unclaimed_reservation_never_dispatched_and_releases_rolling_quota()
    {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 2);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "A bounded action was justified before the policy update",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Perform one bounded action",
            "It tests pre-dispatch invalidation",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let now = chrono::Utc::now();
        let unclaimed = reservation(&mandate, &act, &run, &fence, 1, 41, now);
        assert!(store
            .reserve_mandate_action_attempt(&unclaimed)
            .await
            .unwrap()
            .is_some());

        let mut updated = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        updated.version += 1;
        updated
            .constraints
            .push("Owner narrowed the policy".to_string());
        store.update_mandate(&updated).await.unwrap();

        let attempts = store
            .list_mandate_mutation_attempts_for_run(&run.id)
            .await
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(
            attempts[0].status,
            MandateMutationAttemptStatus::NeverDispatched
        );
        assert!(attempts[0].completed_at.is_some());
        let quota = store
            .get_mandate_mutation_quota_state(
                &mandate.id,
                &(now + chrono::Duration::seconds(1)).to_rfc3339(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(quota.reserved_in_rolling_24h, 0);
        assert_eq!(quota.remaining_in_rolling_24h, 24);
    }

    #[tokio::test]
    async fn pause_marks_claimed_unresolved_reservation_ambiguous_and_run_reconciliation_required()
    {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "One mutation was authorized before the pause",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Perform one bounded action",
            "It tests claimed-dispatch invalidation",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let now = chrono::Utc::now();
        let claimed = reservation(&mandate, &act, &run, &fence, 1, 42, now);
        assert!(store
            .reserve_mandate_action_attempt(&claimed)
            .await
            .unwrap()
            .is_some());
        assert!(store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &claimed,
                now + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap());

        assert!(store
            .transition_mandate_status(&mandate.id, MandateStatus::Active, MandateStatus::Paused,)
            .await
            .unwrap());
        let attempts = store
            .list_mandate_mutation_attempts_for_run(&run.id)
            .await
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].status, MandateMutationAttemptStatus::Ambiguous);
        assert!(attempts[0].completed_at.is_some());
        let closed_run = store
            .get_goal_runs(&goal.id)
            .await
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.id == run.id)
            .unwrap();
        assert_eq!(closed_run.status, "cancelled");
        assert_eq!(
            closed_run.outcome_summary.as_deref(),
            Some("mandate_reconciliation_required:lifecycle_invalidated_after_dispatch_claim")
        );
        let notice_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM notification_queue
             WHERE id = ? AND notification_type = 'mandate_reconciliation_required'
               AND priority = 'critical' AND expires_at IS NULL",
        )
        .bind(format!("mandate-run-notice:{}", run.id))
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(notice_count, 1);
        let suspended = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(suspended.status, MandateStatus::AwaitingInput);
        assert_eq!(
            suspended.suspension.as_ref().map(|value| value.kind),
            Some(MandateSuspensionKind::ReconciliationRequired)
        );
        assert!(store
            .resolve_mandate_suspension(
                &mandate.id,
                suspended.version,
                MandateSuspensionKind::ReconciliationRequired,
                None,
                Some(MandateReconciliationResolution::ConfirmedNoEffect),
                "I inspected the exact target and confirmed no effect occurred.",
                "owner-session",
            )
            .await
            .unwrap());
        let resumed = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(resumed.status, MandateStatus::Active);
        assert!(resumed.suspension.is_none());
        let reconciliations: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM mandate_reconciliations
             WHERE mandate_id = ? AND resolution = 'confirmed_no_effect'",
        )
        .bind(&mandate.id)
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(reconciliations, 1);
    }

    #[tokio::test]
    async fn policy_update_with_claimed_unresolved_mutation_lands_awaiting_input() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "One action was authorized under the old policy",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Perform one bounded action",
            "It tests policy-race reconciliation",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let now = chrono::Utc::now();
        let claimed = reservation(&mandate, &act, &run, &fence, 1, 43, now);
        assert!(store
            .reserve_mandate_action_attempt(&claimed)
            .await
            .unwrap()
            .is_some());
        assert!(store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &claimed,
                now + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap());

        let mut updated = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        updated.version += 1;
        updated
            .constraints
            .push("Owner narrowed the policy after dispatch".to_string());
        store.update_mandate(&updated).await.unwrap();

        let stored = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(stored.version, updated.version);
        assert_eq!(stored.status, MandateStatus::AwaitingInput);
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "paused"
        );
        assert_eq!(
            store
                .list_mandate_mutation_attempts_for_run(&run.id)
                .await
                .unwrap()[0]
                .status,
            MandateMutationAttemptStatus::Ambiguous
        );
        let notices: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM notification_queue WHERE id = ?")
                .bind(format!("mandate-run-notice:{}", run.id))
                .fetch_one(&store.pool)
                .await
                .unwrap();
        assert_eq!(notices, 1);
    }

    #[tokio::test]
    async fn due_review_lease_has_one_winner_and_requires_its_token_to_release() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 2);
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let (left, right) = tokio::join!(
            store.claim_due_mandates(1, "heartbeat-a", 300),
            store.claim_due_mandates(1, "heartbeat-b", 300)
        );
        let mut winners = left.unwrap();
        winners.extend(right.unwrap());
        assert_eq!(winners.len(), 1);
        let lease_token = winners[0].review_lease_token.clone().unwrap();
        assert!(!store
            .release_mandate_review_lease(
                &mandate.id,
                "wrong-token",
                &(chrono::Utc::now() + chrono::Duration::minutes(5)).to_rfc3339(),
            )
            .await
            .unwrap());

        let retry_at = (chrono::Utc::now() + chrono::Duration::minutes(5)).to_rfc3339();
        assert!(store
            .release_mandate_review_lease(&mandate.id, &lease_token, &retry_at)
            .await
            .unwrap());
        let released = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(released.next_review_at, retry_at);
        assert!(released.review_lease_token.is_none());
        assert!(store
            .claim_due_mandates(1, "heartbeat-c", 300)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn claiming_reviews_atomically_closes_expired_mandates_and_controllers() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 2);
        mandate.expires_at = Some((chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339());
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        assert!(store
            .claim_due_mandates(1, "heartbeat", 300)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            store
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::Completed
        );
        let controller = store.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(controller.status, "completed");
        assert!(controller.completed_at.is_some());
    }

    #[tokio::test]
    async fn decision_validates_policy_version_and_commits_transition_atomically() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 2);
        let run = claim_and_start_run(&store, &goal, &mandate).await;

        let stale = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Wait,
            "Nothing useful yet",
            mandate.version + 1,
        );
        assert!(store
            .record_mandate_decision(&stale, None, None)
            .await
            .is_err());
        assert!(store
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .is_none());
        assert!(store
            .get_mandate(&mandate.id)
            .await
            .unwrap()
            .unwrap()
            .review_lease_token
            .is_some());

        let mut wait = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Wait,
            "No action clears the quality threshold",
            mandate.version,
        );
        wait.reconsider_at = Some((chrono::Utc::now() + chrono::Duration::seconds(1)).to_rfc3339());
        let before = chrono::Utc::now();
        store
            .record_mandate_decision(&wait, None, None)
            .await
            .unwrap();

        let loaded = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(loaded.status, MandateStatus::Active);
        assert!(loaded.review_lease_token.is_none());
        let next = chrono::DateTime::parse_from_rfc3339(&loaded.next_review_at)
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(next >= before + chrono::Duration::seconds(mandate.min_review_secs - 1));
        let stored_decision = store
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored_decision.mandate_version, mandate.version);
        assert_eq!(stored_decision.reconsider_at, Some(loaded.next_review_at));
        let history = store.list_mandate_decisions(&mandate.id, 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].id, stored_decision.id);
        assert_eq!(history[0].outcome, MandateDecisionOutcome::Wait);
    }

    #[tokio::test]
    async fn act_for_value_contract_requires_exact_owner_criterion_and_full_judgment() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 1);
        mandate.success_criteria = vec![
            "Each intervention provides verified useful information to the audience".to_string(),
        ];
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "Current evidence supports one bounded useful intervention",
            mandate.version,
        );
        let mut intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Provide one verified useful update",
            "The intervention advances the confirmed value criterion",
        );
        intention.value_criterion = Some("Publish something on every review".to_string());
        intention.expected_benefit =
            Some("Give the audience current, grounded information".to_string());
        intention.risk = Some("Low volume limits attention and reputation costs".to_string());
        intention.invalidation_criteria =
            Some("The information is stale, duplicate, or unsupported".to_string());

        let error = store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("not an exact owner-authored success criterion"));

        intention.value_criterion = mandate.success_criteria.first().cloned();
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        assert_eq!(
            store.list_intentions(&mandate.id, 1).await.unwrap()[0],
            intention
        );
    }

    #[tokio::test]
    async fn action_reservations_require_a_unique_live_proof_and_respect_the_cycle_cap() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 2);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "A sourced reply is timely and useful",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Post one sourced reply",
            "It contributes concrete information",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        assert_eq!(
            store.list_intentions(&mandate.id, 10).await.unwrap(),
            vec![intention]
        );
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let now = chrono::Utc::now();
        let first = reservation(&mandate, &act, &run, &fence, 1, 10, now);

        assert!(store
            .reserve_mandate_action_attempt(&first)
            .await
            .unwrap()
            .is_some());
        assert!(store
            .reserve_mandate_action_attempt(&first)
            .await
            .unwrap()
            .is_none());
        assert!(store
            .reserve_mandate_action_attempt(&reservation(
                &mandate,
                &act,
                &run,
                &fence,
                2,
                11,
                now + chrono::Duration::seconds(901),
            ))
            .await
            .unwrap()
            .is_some());
        assert!(store
            .reserve_mandate_action_attempt(&reservation(
                &mandate,
                &act,
                &run,
                &fence,
                3,
                12,
                now + chrono::Duration::seconds(1_802),
            ))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn owner_policy_update_invalidates_an_existing_action_budget() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 3);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "One action currently fits the policy",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Perform the authorized action",
            "It advances the mandate",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let now = chrono::Utc::now();
        assert!(store
            .reserve_mandate_action_attempt(&reservation(&mandate, &act, &run, &fence, 1, 20, now,))
            .await
            .unwrap()
            .is_some());

        let mut owner_update = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        owner_update.version += 1;
        owner_update
            .constraints
            .push("New owner constraint".to_string());
        store.update_mandate(&owner_update).await.unwrap();

        assert!(store
            .reserve_mandate_action_attempt(&reservation(
                &mandate,
                &act,
                &run,
                &fence,
                2,
                21,
                now + chrono::Duration::seconds(901),
            ))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn concurrent_reservations_have_one_last_slot_winner() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 1);
        mandate.authority.max_mutating_actions_per_rolling_24h = 1;
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "One exact action is justified",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Perform exactly one action",
            "It tests the atomic quota fence",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let now = chrono::Utc::now();
        let left = reservation(&mandate, &act, &run, &fence, 1, 101, now);
        let right = reservation(&mandate, &act, &run, &fence, 1, 102, now);
        let (left, right) = tokio::join!(
            store.reserve_mandate_action_attempt(&left),
            store.reserve_mandate_action_attempt(&right),
        );
        let winners = [left.unwrap(), right.unwrap()]
            .into_iter()
            .filter(Option::is_some)
            .count();
        assert_eq!(winners, 1);
        let attempts = store
            .list_mandate_mutation_attempts_for_run(&run.id)
            .await
            .unwrap();
        assert_eq!(attempts.len(), 1);
    }

    #[tokio::test]
    async fn cooldown_failure_and_rolling_expiry_are_cross_cycle_safety_state() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 2);
        mandate.authority.max_mutating_actions_per_rolling_24h = 2;
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let act = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "Two spaced actions fit the envelope",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &act.id,
            &run.id,
            "Perform two governed actions",
            "It tests rolling quota semantics",
        );
        store
            .record_mandate_decision(&act, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let base = chrono::Utc::now();
        let first = reservation(&mandate, &act, &run, &fence, 1, 201, base);
        assert!(store
            .reserve_mandate_action_attempt(&first)
            .await
            .unwrap()
            .is_some());
        assert!(store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &first,
                base + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap());
        assert!(!store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &first,
                base + chrono::Duration::milliseconds(2),
            ))
            .await
            .unwrap());
        assert!(store
            .project_mandate_mutation_outcome(&outcome_projection(
                &first,
                MandateMutationAttemptStatus::Failed,
                Some(400),
                base + chrono::Duration::seconds(1),
            ))
            .await
            .unwrap());
        let too_soon = reservation(
            &mandate,
            &act,
            &run,
            &fence,
            2,
            202,
            base + chrono::Duration::seconds(899),
        );
        assert!(store
            .reserve_mandate_action_attempt(&too_soon)
            .await
            .unwrap()
            .is_none());
        let cooldown = store
            .get_mandate_mutation_quota_state(
                &mandate.id,
                &(base + chrono::Duration::seconds(899)).to_rfc3339(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            cooldown.block_reason,
            Some(MandateMutationQuotaBlockReason::Cooldown)
        );
        assert_eq!(cooldown.reserved_in_rolling_24h, 1);

        let second = reservation(
            &mandate,
            &act,
            &run,
            &fence,
            2,
            203,
            base + chrono::Duration::seconds(900),
        );
        assert!(store
            .reserve_mandate_action_attempt(&second)
            .await
            .unwrap()
            .is_some());
        let full = store
            .get_mandate_mutation_quota_state(
                &mandate.id,
                &(base + chrono::Duration::seconds(901)).to_rfc3339(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            full.block_reason,
            Some(MandateMutationQuotaBlockReason::Rolling24hExhausted)
        );
        assert_eq!(full.remaining_in_rolling_24h, 0);

        let observation = crate::traits::ToolCallSemantics::observation().with_target_hint(
            crate::traits::ToolTargetHintKind::Url,
            "https://api.x.com/2/tweets",
        );
        assert!(crate::mandates::authority::authorize_mandate_observation(
            &mandate,
            "http_request",
            r#"{"method":"GET","url":"https://api.x.com/2/tweets"}"#,
            &observation,
            &(base + chrono::Duration::seconds(901)),
        )
        .is_ok());

        let expired = store
            .get_mandate_mutation_quota_state(
                &mandate.id,
                &(base + chrono::Duration::hours(24) + chrono::Duration::seconds(1)).to_rfc3339(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(expired.reserved_in_rolling_24h, 1);
        assert_eq!(expired.remaining_in_rolling_24h, 1);
        assert!(expired.available_now);
    }

    #[tokio::test]
    async fn rolling_quota_survives_policy_versions_pause_resume_and_new_cycles() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 1);
        mandate.authority.max_mutating_actions_per_rolling_24h = 1;
        let first_run = claim_and_start_run(&store, &goal, &mandate).await;
        let first_act = MandateDecisionCycle::new(
            &mandate.id,
            &first_run.id,
            MandateDecisionOutcome::Act,
            "Use the sole daily slot",
            mandate.version,
        );
        let first_intention = Intention::new(
            &mandate.id,
            &first_act.id,
            &first_run.id,
            "Use one daily action",
            "It tests cross-cycle accounting",
        );
        store
            .record_mandate_decision(&first_act, Some(&first_intention), None)
            .await
            .unwrap();
        let first_fence = claim_mutation_fence(&store, &goal, &mandate, &first_run).await;
        let base = chrono::Utc::now();
        let first = reservation(&mandate, &first_act, &first_run, &first_fence, 1, 301, base);
        assert!(store
            .reserve_mandate_action_attempt(&first)
            .await
            .unwrap()
            .is_some());
        assert!(store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &first,
                base + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap());
        assert!(store
            .project_mandate_mutation_outcome(&outcome_projection(
                &first,
                MandateMutationAttemptStatus::Failed,
                Some(400),
                base + chrono::Duration::seconds(1),
            ))
            .await
            .unwrap());
        assert!(store
            .transition_mandate_status(&mandate.id, MandateStatus::Active, MandateStatus::Paused,)
            .await
            .unwrap());
        assert!(store
            .transition_mandate_status(&mandate.id, MandateStatus::Paused, MandateStatus::Active,)
            .await
            .unwrap());
        let current = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert!(current.version > mandate.version);
        let second_run = start_existing_claimed_run(&store, &goal, &current).await;
        let second_act = MandateDecisionCycle::new(
            &current.id,
            &second_run.id,
            MandateDecisionOutcome::Act,
            "A new cycle still sees the old reservation",
            current.version,
        );
        let second_intention = Intention::new(
            &current.id,
            &second_act.id,
            &second_run.id,
            "Try another action",
            "The daily ledger must reject it",
        );
        store
            .record_mandate_decision(&second_act, Some(&second_intention), None)
            .await
            .unwrap();
        let second_fence = claim_mutation_fence(&store, &goal, &current, &second_run).await;
        let second = reservation(
            &current,
            &second_act,
            &second_run,
            &second_fence,
            1,
            302,
            base + chrono::Duration::seconds(901),
        );
        assert!(store
            .reserve_mandate_action_attempt(&second)
            .await
            .unwrap()
            .is_none());
        let quota = store
            .get_mandate_mutation_quota_state(
                &current.id,
                &(base + chrono::Duration::seconds(901)).to_rfc3339(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(quota.reserved_in_rolling_24h, 1);
        assert_eq!(
            quota.block_reason,
            Some(MandateMutationQuotaBlockReason::Rolling24hExhausted)
        );
    }

    #[tokio::test]
    async fn canonical_tool_result_atomically_projects_the_mandate_ledger() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "Exercise atomic receipt projection",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &decision.id,
            &run.id,
            "Issue one governed request",
            "The event and ledger must commit together",
        );
        store
            .record_mandate_decision(&decision, Some(&intention), None)
            .await
            .unwrap();
        let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
        let now = chrono::Utc::now();
        let reservation = reservation(&mandate, &decision, &run, &fence, 1, 777, now);
        store
            .reserve_mandate_action_attempt(&reservation)
            .await
            .unwrap()
            .expect("reserved mutation");
        assert!(store
            .claim_mandate_mutation_dispatch(&dispatch_claim(
                &reservation,
                now + chrono::Duration::milliseconds(1),
            ))
            .await
            .unwrap());

        let semantics = crate::traits::ToolCallSemantics::mutation_with(
            crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY,
        );
        let mut receipt = crate::events::ToolReceiptV1::from_metadata(
            &crate::traits::ToolCallMetadata {
                outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
                http_status: Some(201),
                semantics,
                ..Default::default()
            },
            crate::traits::ToolOutcomeStatus::Succeeded,
            crate::events::ToolOutcomeEvidenceSource::StructuredMetadata,
            Some("mandate-atomic-receipt".to_string()),
        );
        receipt.mandate_authority = Some(reservation.grant.clone());
        let result = crate::events::ToolResultData {
            message_id: None,
            tool_call_id: reservation.tool_call_id.clone(),
            name: "http_request".to_string(),
            result: "created".to_string(),
            success: true,
            duration_ms: 1,
            error: None,
            task_id: Some(fence.worker_task.id.clone()),
            annotations: vec![],
            turn_id: None,
            attachments: vec![],
            receipt: Some(receipt),
        };
        let event_store = crate::events::EventStore::new(store.pool()).await.unwrap();
        event_store
            .append(crate::events::Event::new(
                "specialist:mandate",
                crate::events::EventType::ToolResult,
                serde_json::to_value(result).unwrap(),
            ))
            .await
            .unwrap();

        let attempts = store
            .list_mandate_mutation_attempts_for_run(&run.id)
            .await
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].status, MandateMutationAttemptStatus::Succeeded);
        assert!(store
            .project_mandate_mutation_outcome(&outcome_projection(
                &reservation,
                MandateMutationAttemptStatus::Succeeded,
                Some(201),
                now + chrono::Duration::seconds(1),
            ))
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn strict_receipts_finalize_act_and_mixed_failure_fails_closed() {
        for include_failure in [false, true] {
            let (store, _database) = test_store().await;
            let cycle_cap = if include_failure { 2 } else { 1 };
            let (goal, mut mandate) = controller("owner-session", cycle_cap);
            mandate.authority.max_mutating_actions_per_rolling_24h = cycle_cap;
            let run = claim_and_start_run(&store, &goal, &mandate).await;
            let act = MandateDecisionCycle::new(
                &mandate.id,
                &run.id,
                MandateDecisionOutcome::Act,
                "Execute and prove the governed action",
                mandate.version,
            );
            let intention = Intention::new(
                &mandate.id,
                &act.id,
                &run.id,
                "Execute governed mutation",
                "Strict receipts determine completion",
            );
            store
                .record_mandate_decision(&act, Some(&intention), None)
                .await
                .unwrap();
            let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
            let base = chrono::Utc::now();
            let first = reservation(&mandate, &act, &run, &fence, 1, 401, base);
            assert!(store
                .reserve_mandate_action_attempt(&first)
                .await
                .unwrap()
                .is_some());
            assert!(store
                .claim_mandate_mutation_dispatch(&dispatch_claim(
                    &first,
                    base + chrono::Duration::milliseconds(1),
                ))
                .await
                .unwrap());
            assert!(!store
                .project_mandate_mutation_outcome(&outcome_projection(
                    &first,
                    MandateMutationAttemptStatus::Succeeded,
                    Some(202),
                    base + chrono::Duration::milliseconds(500),
                ))
                .await
                .unwrap());
            assert!(store
                .project_mandate_mutation_outcome(&outcome_projection(
                    &first,
                    MandateMutationAttemptStatus::Succeeded,
                    Some(200),
                    base + chrono::Duration::seconds(1),
                ))
                .await
                .unwrap());
            if include_failure {
                let second = reservation(
                    &mandate,
                    &act,
                    &run,
                    &fence,
                    2,
                    402,
                    base + chrono::Duration::seconds(900),
                );
                assert!(store
                    .reserve_mandate_action_attempt(&second)
                    .await
                    .unwrap()
                    .is_some());
                assert!(store
                    .claim_mandate_mutation_dispatch(&dispatch_claim(
                        &second,
                        base + chrono::Duration::milliseconds(900_001),
                    ))
                    .await
                    .unwrap());
                assert!(store
                    .project_mandate_mutation_outcome(&outcome_projection(
                        &second,
                        MandateMutationAttemptStatus::Failed,
                        Some(400),
                        base + chrono::Duration::seconds(901),
                    ))
                    .await
                    .unwrap());
            }
            complete_mutation_fence(&store, &fence).await;
            let result = store
                .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                    mandate_id: mandate.id.clone(),
                    expected_mandate_version: mandate.version,
                    goal_run_id: run.id.clone(),
                    finalized_at: (base + chrono::Duration::seconds(1_000)).to_rfc3339(),
                })
                .await
                .unwrap();
            if include_failure {
                assert!(matches!(
                    result,
                    MandateRunFinalizationResult::ReconciliationRequired {
                        reason: MandateReconciliationReason::MutationOutcomeFailed,
                        ..
                    }
                ));
                assert_eq!(
                    store.get_goal_runs(&goal.id).await.unwrap()[0].status,
                    "failed"
                );
                assert!(store
                    .get_current_goal_run(&goal.id)
                    .await
                    .unwrap()
                    .is_none());
                assert_eq!(
                    store.list_intentions(&mandate.id, 1).await.unwrap()[0].status,
                    IntentionStatus::Suspended
                );
                assert_eq!(
                    store
                        .get_mandate(&mandate.id)
                        .await
                        .unwrap()
                        .unwrap()
                        .status,
                    MandateStatus::AwaitingInput
                );
                assert_eq!(
                    store.get_goal(&goal.id).await.unwrap().unwrap().status,
                    "paused"
                );
            } else {
                assert!(matches!(
                    result,
                    MandateRunFinalizationResult::ActSatisfied { .. }
                ));
                assert_eq!(
                    store.get_goal_runs(&goal.id).await.unwrap()[0].status,
                    "completed"
                );
                assert_eq!(
                    store.list_intentions(&mandate.id, 1).await.unwrap()[0].status,
                    IntentionStatus::Satisfied
                );
            }
        }
    }

    #[tokio::test]
    async fn proof_distinguishes_never_dispatched_from_claimed_ambiguity() {
        for claim_before_finalization in [false, true] {
            let (store, _database) = test_store().await;
            let (goal, mandate) = controller("owner-session", 1);
            let run = claim_and_start_run(&store, &goal, &mandate).await;
            let act = MandateDecisionCycle::new(
                &mandate.id,
                &run.id,
                MandateDecisionOutcome::Act,
                "Exercise the finalizer crash-window classification",
                mandate.version,
            );
            let intention = Intention::new(
                &mandate.id,
                &act.id,
                &run.id,
                "Attempt one governed mutation",
                "The proof must classify its exact dispatch state",
            );
            store
                .record_mandate_decision(&act, Some(&intention), None)
                .await
                .unwrap();
            let fence = claim_mutation_fence(&store, &goal, &mandate, &run).await;
            let now = chrono::Utc::now();
            let reserved = reservation(&mandate, &act, &run, &fence, 1, 451, now);
            assert!(store
                .reserve_mandate_action_attempt(&reserved)
                .await
                .unwrap()
                .is_some());
            if claim_before_finalization {
                assert!(store
                    .claim_mandate_mutation_dispatch(&dispatch_claim(
                        &reserved,
                        now + chrono::Duration::milliseconds(1),
                    ))
                    .await
                    .unwrap());
            }
            complete_mutation_fence(&store, &fence).await;

            let result = store
                .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                    mandate_id: mandate.id.clone(),
                    expected_mandate_version: mandate.version,
                    goal_run_id: run.id.clone(),
                    finalized_at: (now + chrono::Duration::seconds(1)).to_rfc3339(),
                })
                .await
                .unwrap();
            let MandateRunFinalizationResult::ReconciliationRequired { reason, counts } = result
            else {
                panic!("unresolved reservation must fail closed")
            };
            if claim_before_finalization {
                assert_eq!(
                    reason,
                    MandateReconciliationReason::MutationOutcomeAmbiguous
                );
                assert_eq!(counts.ambiguous_or_reserved_mutations, 1);
                assert_eq!(counts.never_dispatched_mutations, 0);
            } else {
                assert_eq!(reason, MandateReconciliationReason::MutationOutcomeFailed);
                assert_eq!(counts.ambiguous_or_reserved_mutations, 0);
                assert_eq!(counts.never_dispatched_mutations, 1);
            }
            let attempts = store
                .list_mandate_mutation_attempts_for_run(&run.id)
                .await
                .unwrap();
            assert_eq!(attempts.len(), 1);
            assert_eq!(
                attempts[0].status,
                if claim_before_finalization {
                    MandateMutationAttemptStatus::Ambiguous
                } else {
                    MandateMutationAttemptStatus::NeverDispatched
                }
            );
        }
    }

    #[tokio::test]
    async fn evidence_receipts_accept_either_tool_call_id_or_receipt_result_id() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 1);
        mandate.objective_control = Some(objective_control());
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let task_id = run.root_task_id.as_deref().unwrap();
        sqlx::query(
            "INSERT INTO events
                (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES (?, 'tool_result', ?, ?, ?, 'http_request')",
        )
        .bind("owner-session")
        .bind(
            serde_json::json!({
                "task_id": task_id,
                "tool_call_id": "call_synthetic_metric",
                "name": "http_request",
                "result": "synthetic metric value",
                "success": true,
                "duration_ms": 1,
                "receipt": {
                    "schema_version": crate::events::ToolReceiptV1::SCHEMA_VERSION,
                    "outcome_status": "succeeded",
                    "outcome_evidence": "structured_metadata",
                    "result_provenance": { "result_id": "sha256:synthetic-metric-digest" }
                }
            })
            .to_string(),
        )
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(task_id)
        .execute(&store.pool)
        .await
        .unwrap();

        for receipt_id in ["call_synthetic_metric", "sha256:synthetic-metric-digest"] {
            let measurement = MandateObjectiveMeasurement::new(
                &mandate.id,
                mandate.version,
                &run.id,
                12_000_000,
                9_500,
                vec![receipt_id.to_string()],
                &chrono::Utc::now().to_rfc3339(),
            );
            store
                .record_mandate_objective_measurement(&measurement)
                .await
                .unwrap_or_else(|error| panic!("{receipt_id} should be accepted: {error}"));
        }
        let invented = MandateObjectiveMeasurement::new(
            &mandate.id,
            mandate.version,
            &run.id,
            12_000_000,
            9_500,
            vec!["sha256:not-a-receipt".to_string()],
            &chrono::Utc::now().to_rfc3339(),
        );
        assert!(store
            .record_mandate_objective_measurement(&invented)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn runtime_fallback_wait_bypasses_semantic_gates_and_finalizes_as_deliberator_failure() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 1);
        mandate.objective_control = Some(objective_control());
        let run = claim_and_start_run(&store, &goal, &mandate).await;

        // A model-authored WAIT on a controlled mandate still needs a
        // current-run measurement.
        let model_wait = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Wait,
            "model decided to wait without measuring",
            mandate.version,
        );
        let error = store
            .record_mandate_decision(&model_wait, None, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("metric measurement"), "{error}");

        // The runtime fallback carries no authority and must always persist,
        // regardless of objective control state.
        let fallback =
            MandateDecisionCycle::runtime_fallback_wait(&mandate, &run.id, chrono::Utc::now());
        assert!(fallback.is_runtime_fallback());
        store
            .record_mandate_decision(&fallback, None, None)
            .await
            .unwrap();
        let mut spoofed = fallback.clone();
        spoofed.id = uuid::Uuid::new_v4().to_string();
        spoofed.evidence_receipt_ids = vec!["call_anything".to_string()];
        assert!(store
            .record_mandate_decision(&spoofed, None, None)
            .await
            .is_err());

        let root_attempt = store
            .claim_task_with_lease(
                run.root_task_id.as_deref().unwrap(),
                "fallback-root",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        assert!(store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("task lead returned without a decision".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        let result = store
            .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                mandate_id: mandate.id.clone(),
                expected_mandate_version: mandate.version,
                goal_run_id: run.id.clone(),
                finalized_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();
        assert_eq!(
            result,
            MandateRunFinalizationResult::Rejected {
                reason: MandateFinalizationRejectReason::DeliberatorFailed
            }
        );
        let current = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(current.status, MandateStatus::Active);
    }

    #[tokio::test]
    async fn missing_decision_closes_run_schedules_retry_and_commits_review_failed_notice() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let root_attempt = store
            .claim_task_with_lease(
                run.root_task_id.as_deref().unwrap(),
                "decision-missing-root",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        assert!(store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("task lead returned without a decision".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        let finalized_at = chrono::Utc::now();

        let result = store
            .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                mandate_id: mandate.id.clone(),
                expected_mandate_version: mandate.version,
                goal_run_id: run.id.clone(),
                finalized_at: finalized_at.to_rfc3339(),
            })
            .await
            .unwrap();
        assert_eq!(
            result,
            MandateRunFinalizationResult::Rejected {
                reason: MandateFinalizationRejectReason::DecisionMissing
            }
        );
        assert_eq!(
            store.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "failed"
        );
        let current = store.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(current.status, MandateStatus::Active);
        assert!(
            chrono::DateTime::parse_from_rfc3339(&current.next_review_at)
                .unwrap()
                .with_timezone(&chrono::Utc)
                >= finalized_at + chrono::Duration::seconds(mandate.min_review_secs)
        );
        let controller = store.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(controller.status, "active");
        assert_eq!(controller.dispatch_failures, 1);
        let queued: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM notification_queue
             WHERE id = ? AND notification_type = 'mandate_review_failed'
               AND priority = 'critical' AND expires_at IS NULL",
        )
        .bind(format!("mandate-run-notice:{}", run.id))
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(queued, 1);
    }

    #[tokio::test]
    async fn invalid_decision_state_pauses_controller_and_commits_review_failed_notice() {
        let (store, _database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let root_attempt = store
            .claim_task_with_lease(
                run.root_task_id.as_deref().unwrap(),
                "invalid-decision-root",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        let decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "Persist a valid row before simulating corruption",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &decision.id,
            &run.id,
            "A committed intention that cannot belong to WAIT",
            "It creates a valid but inconsistent persisted decision state",
        );
        store
            .record_mandate_decision(&decision, Some(&intention), Some(&root_attempt.id))
            .await
            .unwrap();
        sqlx::query("UPDATE mandate_decision_cycles SET outcome = 'wait' WHERE id = ?")
            .bind(&decision.id)
            .execute(&store.pool)
            .await
            .unwrap();
        assert!(store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("invalid decision state simulated".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());

        let result = store
            .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                mandate_id: mandate.id.clone(),
                expected_mandate_version: mandate.version,
                goal_run_id: run.id.clone(),
                finalized_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();
        assert_eq!(
            result,
            MandateRunFinalizationResult::Rejected {
                reason: MandateFinalizationRejectReason::InvalidDecisionState
            }
        );
        assert_eq!(
            store
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::AwaitingInput
        );
        assert_eq!(
            store.get_goal(&goal.id).await.unwrap().unwrap().status,
            "paused"
        );
        assert_eq!(
            store.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "failed"
        );
        let queued: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM notification_queue
             WHERE id = ? AND notification_type = 'mandate_review_failed'
               AND priority = 'critical' AND expires_at IS NULL",
        )
        .bind(format!("mandate-run-notice:{}", run.id))
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(queued, 1);
    }

    #[tokio::test]
    async fn ask_and_stop_atomically_change_mandate_and_controller_status() {
        for (outcome, expected_mandate, expected_goal) in [
            (
                MandateDecisionOutcome::Ask,
                MandateStatus::AwaitingInput,
                "paused",
            ),
            (
                MandateDecisionOutcome::Stop,
                MandateStatus::Completed,
                "completed",
            ),
        ] {
            let (store, _database) = test_store().await;
            let (goal, mandate) = controller("owner-session", 1);
            let run = claim_and_start_run(&store, &goal, &mandate).await;
            let root_attempt = store
                .claim_task_with_lease(
                    run.root_task_id.as_deref().unwrap(),
                    "non-action-root",
                    Some("profile-task-lead"),
                    7_200,
                )
                .await
                .unwrap()
                .unwrap();
            let mut decision = MandateDecisionCycle::new(
                &mandate.id,
                &run.id,
                outcome,
                "The mandate cannot safely proceed",
                mandate.version,
            );
            if outcome == MandateDecisionOutcome::Ask {
                decision.question = Some("May I broaden the allowed target?".to_string());
            } else {
                decision.termination_kind = Some(MandateTerminationKind::SafetyTermination);
            }
            store
                .record_mandate_decision(&decision, None, Some(&root_attempt.id))
                .await
                .unwrap();
            assert_eq!(
                store
                    .get_mandate(&mandate.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                MandateStatus::Active,
                "the lifecycle transition remains provisional until proof finalization"
            );
            assert_eq!(
                store.get_goal(&goal.id).await.unwrap().unwrap().status,
                "active"
            );
            assert!(store
                .patch_task_from_attempt(
                    &root_attempt.id,
                    &root_attempt.lease_token,
                    &crate::traits::TaskAttemptPatch {
                        status: "completed".to_string(),
                        result: Some("non-action decision durably recorded".to_string()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap());
            let result = store
                .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                    mandate_id: mandate.id.clone(),
                    expected_mandate_version: mandate.version,
                    goal_run_id: run.id.clone(),
                    finalized_at: chrono::Utc::now().to_rfc3339(),
                })
                .await
                .unwrap();
            assert!(matches!(
                result,
                MandateRunFinalizationResult::NonActionSatisfied {
                    outcome: finalized_outcome,
                    ..
                } if finalized_outcome == outcome
            ));
            assert_eq!(
                store
                    .get_mandate(&mandate.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                expected_mandate
            );
            assert_eq!(
                store.get_goal(&goal.id).await.unwrap().unwrap().status,
                expected_goal
            );
        }
    }

    #[tokio::test]
    async fn receipt_backed_stop_and_learning_are_same_mandate_only() {
        let (store, _database) = test_store().await;
        let (goal, mut mandate) = controller("owner-session", 0);
        mandate.success_criteria = vec!["Account stewardship is no longer needed".to_string()];
        let run = claim_and_start_run(&store, &goal, &mandate).await;
        let root_task_id = run.root_task_id.as_deref().unwrap();
        let root_attempt = store
            .claim_task_with_lease(
                root_task_id,
                "receipt-backed-stop",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();

        let receipt = crate::events::ToolReceiptV1::from_metadata(
            &crate::traits::ToolCallMetadata {
                outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
                semantics: crate::traits::ToolCallSemantics::observation(),
                ..Default::default()
            },
            crate::traits::ToolOutcomeStatus::Succeeded,
            crate::events::ToolOutcomeEvidenceSource::StructuredMetadata,
            None,
        );
        let result = crate::events::ToolResultData {
            message_id: None,
            tool_call_id: "observation-receipt-1".to_string(),
            name: "web_fetch".to_string(),
            result: "bounded observation".to_string(),
            success: true,
            duration_ms: 1,
            error: None,
            task_id: Some(root_task_id.to_string()),
            annotations: vec![],
            turn_id: None,
            attachments: vec![],
            receipt: Some(receipt),
        };
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES (?, 'tool_result', ?, ?, ?, 'web_fetch')",
        )
        .bind("specialist:mandate")
        .bind(serde_json::to_string(&result).unwrap())
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(root_task_id)
        .execute(&store.pool)
        .await
        .unwrap();

        let mut decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Stop,
            "A current structured observation proves the owner criterion",
            mandate.version,
        );
        decision.termination_kind = Some(MandateTerminationKind::SuccessCriteriaSatisfied);
        decision.activity_level = MandateActivityLevel::Urgent;
        decision.termination_match = mandate.success_criteria.first().cloned();
        decision.evidence_receipt_ids = vec!["observation-receipt-1".to_string()];
        let note = MandateLearningNote::new(
            &mandate.id,
            mandate.version,
            &decision.id,
            "Explicit verification is more reliable than inferred completion",
            vec!["observation-receipt-1".to_string()],
        );
        let revision = MandateStrategyRevision::new(
            &mandate.id,
            mandate.version,
            &decision.id,
            "verify_before_completion",
            MandateStrategyRevisionKind::Reinforce,
            "Prefer explicit verification before declaring completion",
            9_000,
            vec!["observation-receipt-1".to_string()],
        );
        store
            .record_mandate_decision_with_updates(
                &decision,
                None,
                Some(&MandateOperatingUpdates {
                    learning_note: Some(note.clone()),
                    strategy_revisions: vec![revision.clone()],
                }),
                Some(&root_attempt.id),
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .list_mandate_learning_notes(&mandate.id, 10)
                .await
                .unwrap(),
            vec![note]
        );
        assert_eq!(
            store
                .list_current_mandate_strategy(&mandate.id, 16)
                .await
                .unwrap(),
            vec![revision]
        );
        assert_eq!(
            store
                .get_mandate_decision_for_run(&run.id)
                .await
                .unwrap()
                .unwrap()
                .activity_level,
            MandateActivityLevel::Urgent
        );

        assert!(store
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("receipt-backed STOP recorded".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        let finalized = store
            .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                mandate_id: mandate.id.clone(),
                expected_mandate_version: mandate.version,
                goal_run_id: run.id,
                finalized_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();
        assert!(matches!(
            finalized,
            MandateRunFinalizationResult::NonActionSatisfied {
                outcome: MandateDecisionOutcome::Stop,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn ask_and_stop_crashes_remain_reconcilable_until_proof_finalization() {
        for outcome in [MandateDecisionOutcome::Ask, MandateDecisionOutcome::Stop] {
            let (store, _database) = test_store().await;
            let (goal, mandate) = controller("owner-session", 1);
            let run = claim_and_start_run(&store, &goal, &mandate).await;
            let root_attempt = store
                .claim_task_with_lease(
                    run.root_task_id.as_deref().unwrap(),
                    "crashing-non-action-root",
                    Some("profile-task-lead"),
                    7_200,
                )
                .await
                .unwrap()
                .unwrap();
            let mut decision = MandateDecisionCycle::new(
                &mandate.id,
                &run.id,
                outcome,
                "Record the non-action decision before the simulated crash",
                mandate.version,
            );
            if outcome == MandateDecisionOutcome::Ask {
                decision.question = Some("May I broaden the exact target?".to_string());
            } else {
                decision.termination_kind = Some(MandateTerminationKind::SafetyTermination);
            }
            store
                .record_mandate_decision(&decision, None, Some(&root_attempt.id))
                .await
                .unwrap();
            assert_eq!(
                store
                    .get_mandate(&mandate.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                MandateStatus::Active
            );
            sqlx::query("UPDATE task_attempts SET lease_expires_at = ? WHERE id = ?")
                .bind((chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339())
                .bind(&root_attempt.id)
                .execute(&store.pool)
                .await
                .unwrap();

            assert!(store
                .claim_due_mandates(1, "post-crash-reconciler", 300)
                .await
                .unwrap()
                .is_empty());
            assert_eq!(
                store
                    .get_mandate(&mandate.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                MandateStatus::AwaitingInput
            );
            assert_eq!(
                store.get_goal(&goal.id).await.unwrap().unwrap().status,
                "paused"
            );
            assert_eq!(
                store.get_goal_runs(&goal.id).await.unwrap()[0].status,
                "cancelled"
            );
        }
    }

    #[tokio::test]
    async fn mandate_schema_and_rows_survive_an_idempotent_reopen() {
        let (store, database) = test_store().await;
        let (goal, mandate) = controller("owner-session", 1);
        store
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        store.pool.close().await;
        drop(store);

        let reopened = SqliteStateStore::new(
            database.path().to_str().unwrap(),
            100,
            None,
            Arc::new(EmbeddingService::new().unwrap()),
        )
        .await
        .unwrap();
        let loaded = reopened.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(loaded.goal_id, goal.id);
        assert_eq!(loaded.version, mandate.version);
        assert_eq!(loaded.next_review_at, mandate.next_review_at);
    }
}
