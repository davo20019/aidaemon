//! Validation of mandates, timestamps, identifiers and receipt references before they are written.

use super::*;

pub(super) fn validate_timestamp(label: &str, value: &str) -> anyhow::Result<()> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("invalid {label}: {error}"))
}

pub(super) fn validate_mandate(mandate: &Mandate) -> anyhow::Result<()> {
    anyhow::ensure!(!mandate.id.trim().is_empty(), "mandate id is required");
    mandate
        .validate_content_bounds()
        .map_err(anyhow::Error::msg)?;
    crate::authority::validate_objective_measurement_source(mandate).map_err(anyhow::Error::msg)?;
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

pub(super) fn validate_new_mandate(mandate: &Mandate) -> anyhow::Result<()> {
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

pub(super) async fn validate_current_run_receipt_refs(
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
        .bind(i64::from(aidaemon_events::ToolReceiptV1::SCHEMA_VERSION))
        .fetch_optional(&mut *connection)
        .await?;
        anyhow::ensure!(
            found.is_some(),
            "mandate evidence receipt `{receipt_id}` is not a current-run structured success"
        );
    }
    Ok(())
}

pub(super) fn validate_canonical_identifier(
    label: &str,
    value: &str,
    max_chars: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty()
            && value.trim() == value
            && value.chars().count() <= max_chars
            && !value.chars().any(char::is_control),
        "invalid mandate mutation {label}"
    );
    Ok(())
}
