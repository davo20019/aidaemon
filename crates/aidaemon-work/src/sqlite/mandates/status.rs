//! Mandate lifecycle: status transitions, resuming, owner guidance, wake signals and suspensions.

use super::*;

pub(super) async fn transition_mandate_status<S: WorkDatabase>(
    store: &S,
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
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
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

pub(super) async fn resume_mandate_with_context<S: WorkDatabase>(
    store: &S,
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
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
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

pub(super) async fn record_active_mandate_owner_guidance<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    expected_version: i64,
    owner_session: &str,
    controller_context: &str,
) -> anyhow::Result<bool> {
    anyhow::ensure!(
        controller_context.chars().count() <= 64_000,
        "mandate controller context is too large"
    );
    serde_json::from_str::<serde_json::Value>(controller_context)?;
    let now = chrono::Utc::now().to_rfc3339();
    let updated = sqlx::query(
        "UPDATE goals SET context = ?, updated_at = ?
             WHERE id = (
                 SELECT goal_id FROM mandates
                 WHERE id = ? AND version = ? AND status IN ('active', 'paused')
                   AND confirmed_at IS NOT NULL AND created_by_session = ?
             )
               AND domain = 'orchestration' AND goal_type = 'continuous'
               AND status = (SELECT status FROM mandates WHERE id = ?)",
    )
    .bind(controller_context)
    .bind(&now)
    .bind(mandate_id)
    .bind(expected_version)
    .bind(owner_session)
    .bind(mandate_id)
    .execute(store.work_pool())
    .await?;
    Ok(updated.rows_affected() == 1)
}

pub(super) async fn keep_mandate_controller_active<S: WorkDatabase>(
    store: &S,
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
    .execute(store.work_pool())
    .await?;
    Ok(result.rows_affected() == 1)
}

pub(super) async fn wake_mandates_for_signal<S: WorkDatabase>(
    store: &S,
    signal: &MandateWakeSignal,
) -> anyhow::Result<Vec<String>> {
    signal.validate().map_err(anyhow::Error::msg)?;
    let mandates = store.list_mandates(None, false).await?;
    let mut target = reqwest::Url::parse(&signal.target_url)?;
    target.set_query(None);
    target.set_fragment(None);
    let target_url = target.to_string();
    let signal_digest = hex::encode(Sha256::digest(
        format!(
            "aidaemon.mandate.wake.v1\0{}\0{}\0{}\0{}\0{}",
            signal.kind.as_str(),
            signal.source,
            signal.target_url,
            signal.account_id.as_deref().unwrap_or(""),
            signal.dedupe_key
        )
        .as_bytes(),
    ));
    let received_at = chrono::Utc::now().to_rfc3339();
    let mut awakened = Vec::new();
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
    for mandate in mandates
        .into_iter()
        .filter(|mandate| crate::authority::mandate_accepts_wake_signal(mandate, signal))
    {
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

// The trait method's parameters, passed through.
#[allow(clippy::too_many_arguments)]
pub(super) async fn resolve_mandate_suspension<S: WorkDatabase>(
    store: &S,
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
        MandateSuspensionKind::AwaitingAnswer => anyhow::ensure!(
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
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
    let current = sqlx::query(
        "SELECT goal_id, created_by_session, suspension_json
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
