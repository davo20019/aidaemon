//! Creating, reading, listing, updating and confirming mandates and their controller goals.

use super::*;

pub(super) async fn insert_mandate_row(
    connection: &mut sqlx::SqliteConnection,
    mandate: &Mandate,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT INTO mandates (
            id, goal_id, source_goal_id, name, objective, status, autonomy_mode, authority_json, strategy_json,
            objective_control_json, suspension_json,
            constraints_json, success_criteria_json, stop_conditions_json,
            min_review_secs, max_review_secs, default_review_secs, review_effort, next_review_at,
            review_lease_token, review_lease_expires_at, expires_at, confirmed_at,
            version, owner_principal_id, created_by_session, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&mandate.id)
    .bind(&mandate.goal_id)
    .bind(&mandate.source_goal_id)
    .bind(&mandate.name)
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

pub(super) async fn validate_source_goal(
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

pub(super) async fn create_mandate_controller<S: WorkDatabase>(
    store: &S,
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
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
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

pub(super) async fn get_mandate<S: WorkDatabase>(
    store: &S,
    id: &str,
) -> anyhow::Result<Option<Mandate>> {
    let query = format!("SELECT {MANDATE_COLUMNS} FROM mandates WHERE id = ?");
    let row = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(id)
        .fetch_optional(store.work_pool())
        .await?;
    row.as_ref().map(mandate_from_row).transpose()
}

pub(super) async fn get_mandate_for_goal<S: WorkDatabase>(
    store: &S,
    goal_id: &str,
) -> anyhow::Result<Option<Mandate>> {
    let query = format!("SELECT {MANDATE_COLUMNS} FROM mandates WHERE goal_id = ?");
    let row = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(goal_id)
        .fetch_optional(store.work_pool())
        .await?;
    row.as_ref().map(mandate_from_row).transpose()
}

pub(super) async fn list_mandates<S: WorkDatabase>(
    store: &S,
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
            aidaemon_events::session::stable_private_owner_principal_id(session_id)
                .unwrap_or_default();
        sqlx::query(sqlx::AssertSqlSafe(&*query))
            .bind(session_id)
            .bind(stable_principal)
            .fetch_all(store.work_pool())
            .await?
    } else {
        sqlx::query(sqlx::AssertSqlSafe(&*query))
            .fetch_all(store.work_pool())
            .await?
    };
    rows.iter().map(mandate_from_row).collect()
}

pub(super) async fn is_mandate_session_authorized<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    session_id: &str,
) -> anyhow::Result<bool> {
    let stable_principal =
        aidaemon_events::session::stable_private_owner_principal_id(session_id).unwrap_or_default();
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
    .fetch_one(store.work_pool())
    .await?
        != 0)
}

pub(super) async fn update_mandate<S: WorkDatabase>(
    store: &S,
    mandate: &Mandate,
) -> anyhow::Result<()> {
    validate_mandate(mandate)?;
    let expected_version = mandate
        .version
        .checked_sub(1)
        .filter(|version| *version > 0)
        .ok_or_else(|| anyhow::anyhow!("updated mandate version must be at least 2"))?;
    let now = chrono::Utc::now().to_rfc3339();
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
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
    let claimed_unresolved = invalidate_open_mandate_runs(&mut tx, &mandate.goal_id, &now).await?;
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

pub(super) async fn confirm_mandate<S: WorkDatabase>(
    store: &S,
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
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
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
