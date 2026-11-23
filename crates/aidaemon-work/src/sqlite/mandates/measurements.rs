//! Objective measurements and learning notes recorded by mandate reviews.

use super::*;

pub(super) async fn record_mandate_objective_measurement<S: WorkDatabase>(
    store: &S,
    measurement: &MandateObjectiveMeasurement,
) -> anyhow::Result<()> {
    measurement.validate().map_err(anyhow::Error::msg)?;
    let observed_at =
        chrono::DateTime::parse_from_rfc3339(&measurement.observed_at)?.with_timezone(&chrono::Utc);
    anyhow::ensure!(
        observed_at <= chrono::Utc::now() + chrono::Duration::minutes(5),
        "objective measurement cannot be future-dated"
    );
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
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
        anyhow::anyhow!("objective measurement is not bound to a current controlled mandate run")
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
    sqlx::query(
        "INSERT INTO mandate_objective_measurements
                (id, mandate_id, mandate_version, goal_run_id, value_micros,
                 confidence_bps, evidence_receipt_ids_json, observed_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&measurement.id)
    .bind(&measurement.mandate_id)
    .bind(measurement.mandate_version)
    .bind(&measurement.goal_run_id)
    .bind(measurement.value_micros)
    .bind(i64::from(measurement.confidence_bps))
    .bind(serde_json::to_string(&measurement.evidence_receipt_ids)?)
    .bind(&measurement.observed_at)
    .bind(&measurement.created_at)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(())
}

pub(super) async fn list_mandate_objective_measurements<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    limit: i64,
) -> anyhow::Result<Vec<MandateObjectiveMeasurement>> {
    let rows = sqlx::query(
        "SELECT id, mandate_id, mandate_version, goal_run_id, value_micros,
                    confidence_bps, evidence_receipt_ids_json, observed_at, created_at
             FROM mandate_objective_measurements
             WHERE mandate_id = ?
             ORDER BY julianday(observed_at) DESC, id DESC LIMIT ?",
    )
    .bind(mandate_id)
    .bind(limit.clamp(1, 200))
    .fetch_all(store.work_pool())
    .await?;
    rows.iter().map(objective_measurement_from_row).collect()
}

pub(super) async fn record_mandate_learning_note<S: WorkDatabase>(
    store: &S,
    note: &MandateLearningNote,
) -> anyhow::Result<()> {
    note.validate().map_err(anyhow::Error::msg)?;
    let mut tx = store.work_pool().begin_with("BEGIN IMMEDIATE").await?;
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
        .bind(i64::from(aidaemon_events::ToolReceiptV1::SCHEMA_VERSION))
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

pub(super) async fn list_mandate_learning_notes<S: WorkDatabase>(
    store: &S,
    mandate_id: &str,
    limit: i64,
) -> anyhow::Result<Vec<MandateLearningNote>> {
    let query = format!(
        "SELECT {LEARNING_NOTE_COLUMNS} FROM mandate_learning_notes
             WHERE mandate_id = ? ORDER BY julianday(created_at) DESC, id DESC LIMIT ?"
    );
    let rows = sqlx::query(sqlx::AssertSqlSafe(&*query))
        .bind(mandate_id)
        .bind(limit.clamp(1, 100))
        .fetch_all(store.work_pool())
        .await?;
    rows.iter().map(learning_note_from_row).collect()
}
