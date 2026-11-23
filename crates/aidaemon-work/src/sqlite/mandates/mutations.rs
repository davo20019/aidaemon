//! Mutation attempts: reservations, dispatch claims and the rolling quota.

use super::*;

pub(super) fn validate_mutation_reservation(
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

pub(super) async fn mutation_quota_state_on_connection(
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

pub(super) async fn reserve_mandate_action_attempt<S: WorkDatabase>(
    store: &S,
    reservation: &MandateMutationReservation,
) -> anyhow::Result<Option<MandateMutationAttempt>> {
    let reserved_at = validate_mutation_reservation(reservation)?;
    let expected_attempts = reservation.grant.reserved_action_attempt - 1;
    let reserved_at_string = reserved_at.to_rfc3339();
    let mutation_effects_json = serde_json::to_string(&reservation.mutation_effects)?;
    let targets_json = serde_json::to_string(&reservation.targets)?;
    let account_identifiers_json = serde_json::to_string(&reservation.account_identifiers)?;
    let ledger_id = uuid::Uuid::new_v4().to_string();
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;

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

    let query =
        format!("SELECT {MUTATION_ATTEMPT_COLUMNS} FROM mandate_mutation_attempts WHERE id = ?");
    let row = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(&ledger_id)
        .fetch_one(&mut *tx)
        .await?;
    let attempt = mutation_attempt_from_row(&row)?;
    tx.commit().await?;
    Ok(Some(attempt))
}

pub(super) async fn claim_mandate_mutation_dispatch<S: WorkDatabase>(
    store: &S,
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
    .execute(store.work_pool())
    .await?;
    Ok(result.rows_affected() == 1)
}

pub(super) async fn get_mandate_mutation_quota_state<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    as_of: &str,
) -> anyhow::Result<Option<MandateMutationQuotaState>> {
    validate_canonical_identifier("mandate id", mandate_id, 256)?;
    let mut connection = store.work_pool().acquire().await?;
    mutation_quota_state_on_connection(&mut connection, mandate_id, as_of).await
}

pub(super) async fn list_mandate_mutation_attempts_for_run<S: WorkDatabase>(
    store: &S,
    goal_run_id: &str,
) -> anyhow::Result<Vec<MandateMutationAttempt>> {
    validate_canonical_identifier("goal run id", goal_run_id, 256)?;
    let query = format!(
        "SELECT {MUTATION_ATTEMPT_COLUMNS} FROM mandate_mutation_attempts
             WHERE goal_run_id = ? ORDER BY julianday(reserved_at), id"
    );
    let rows = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(goal_run_id)
        .fetch_all(store.work_pool())
        .await?;
    rows.iter().map(mutation_attempt_from_row).collect()
}
