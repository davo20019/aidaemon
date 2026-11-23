//! Tests of objective measurements and the stops they back.

use super::*;

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
                "schema_version": aidaemon_events::ToolReceiptV1::SCHEMA_VERSION,
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
    assert!(
        store
            .record_mandate_objective_measurement(&unproven)
            .await
            .is_err()
    );
}

/// Seed one objective reading on a closed prior review run, as the
/// history of an earlier cycle would leave it.
async fn seed_prior_measurement(
    store: &SqliteWorkStore,
    mandate: &Mandate,
    current_run: &GoalRun,
    value_micros: i64,
    confidence_bps: i64,
    hours_ago: i64,
) {
    let run_id = uuid::Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO goal_runs (id, project_id, goal_id, trigger_type, status)
             SELECT ?, project_id, goal_id, 'mandate', 'completed' FROM goal_runs WHERE id = ?",
    )
    .bind(&run_id)
    .bind(&current_run.id)
    .execute(&store.pool)
    .await
    .unwrap();
    seed_measurement_on_run(
        store,
        mandate,
        &run_id,
        value_micros,
        confidence_bps,
        hours_ago,
    )
    .await;
}

async fn seed_measurement_on_run(
    store: &SqliteWorkStore,
    mandate: &Mandate,
    run_id: &str,
    value_micros: i64,
    confidence_bps: i64,
    hours_ago: i64,
) {
    let observed_at = (chrono::Utc::now() - chrono::Duration::hours(hours_ago)).to_rfc3339();
    sqlx::query(
        "INSERT INTO mandate_objective_measurements
                (id, mandate_id, mandate_version, goal_run_id, value_micros, confidence_bps,
                 evidence_receipt_ids_json, attributed_intention_ids_json, observed_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, '[\"call_synthetic_seed\"]', '[]', ?, ?)",
    )
    .bind(uuid::Uuid::new_v4().to_string())
    .bind(&mandate.id)
    .bind(mandate.version)
    .bind(run_id)
    .bind(value_micros)
    .bind(confidence_bps)
    .bind(&observed_at)
    .bind(&observed_at)
    .execute(&store.pool)
    .await
    .unwrap();
}

/// Measurements are read, not judged: flat (or unreadable) readings
/// never refuse a WAIT; whether to change course is the reviewer's call.
#[tokio::test]
async fn flat_measurements_never_refuse_a_wait() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 1);
    mandate.autonomy_mode = MandateAutonomyMode::Autopilot;
    mandate.objective_control = Some(objective_control());
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    for hours_ago in 2..=5 {
        seed_prior_measurement(&store, &mandate, &run, 10_000_000, 9_000, hours_ago).await;
    }
    seed_prior_measurement(&store, &mandate, &run, 0, 0, 1).await;
    seed_measurement_on_run(&store, &mandate, &run.id, 10_000_000, 9_000, 0).await;

    let wait = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Wait,
        "no change observed",
        mandate.version,
    );
    store
        .record_mandate_decision(&wait, None, None)
        .await
        .expect("a flat metric must not force the mandate out of WAIT");
}

#[tokio::test]
async fn success_stop_requires_a_credible_current_measurement() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 1);
    // STOP on target is part of autopilot's measured control loop.
    mandate.autonomy_mode = MandateAutonomyMode::Autopilot;
    let control = objective_control();
    let target = control.target_micros;
    mandate.objective_control = Some(control);
    mandate.success_criteria = vec!["Synthetic interactions reach the target".to_string()];
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    seed_measurement_on_run(&store, &mandate, &run.id, target, 0, 0).await;

    let mut stop = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Stop,
        "target reached",
        mandate.version,
    );
    stop.termination_kind = Some(MandateTerminationKind::SuccessCriteriaSatisfied);
    stop.termination_match = Some("Synthetic interactions reach the target".to_string());
    stop.evidence_receipt_ids = vec![];
    let error = store
        .record_mandate_decision(&stop, None, None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("credible"), "{error}");
}

/// A bounded mandate's advisory control does not gate a success STOP on
/// a target-reaching measurement; the owner criterion match still does.
#[tokio::test]
async fn bounded_success_stop_does_not_require_a_target_measurement() {
    let (store, _database) = test_store().await;
    let (goal, mut mandate) = controller("owner-session", 1);
    mandate.objective_control = Some(objective_control());
    mandate.success_criteria = vec!["One grounded note is posted".to_string()];
    let run = claim_and_start_run(&store, &goal, &mandate).await;
    let mut stop = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Stop,
        "the note is posted",
        mandate.version,
    );
    stop.termination_kind = Some(MandateTerminationKind::SuccessCriteriaSatisfied);
    stop.termination_match = Some("One grounded note is posted".to_string());
    let root_task_id = run.root_task_id.clone().expect("review root task");
    let receipt = aidaemon_events::ToolReceiptV1::from_metadata(
        &crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
            semantics: crate::traits::ToolCallSemantics::observation(),
            ..Default::default()
        },
        crate::traits::ToolOutcomeStatus::Succeeded,
        aidaemon_events::ToolOutcomeEvidenceSource::StructuredMetadata,
        None,
    );
    let result = aidaemon_events::ToolResultData {
        message_id: None,
        tool_call_id: "call_read_notes".to_string(),
        name: "http_request".to_string(),
        result: "one note posted".to_string(),
        success: true,
        duration_ms: 1,
        error: None,
        task_id: Some(root_task_id.clone()),
        annotations: vec![],
        turn_id: None,
        attachments: vec![],
        receipt: Some(receipt),
    };
    sqlx::query(
        "INSERT INTO events (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES ('specialist:mandate', 'tool_result', ?, ?, ?, 'http_request')",
    )
    .bind(serde_json::to_string(&result).unwrap())
    .bind(chrono::Utc::now().to_rfc3339())
    .bind(&root_task_id)
    .execute(&store.pool)
    .await
    .unwrap();
    stop.evidence_receipt_ids = vec!["call_read_notes".to_string()];
    store
        .record_mandate_decision(&stop, None, None)
        .await
        .expect("a bounded mandate stops on its criterion");
}
