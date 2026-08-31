//! One authoritative lifecycle readout for a goal-backed objective.
//!
//! An objective's health lives in typed durable state (schedules, goal runs,
//! recovery disposition), but it used to be visible only through
//! `scheduled_goal_runs`. An owner audit that happened to consult a different
//! surface (for example the mandate listing) honestly reported the objective
//! as "unknown" even while the durable state said healthy. Every owner-facing
//! surface that lists an objective must project THIS shared summary, so the
//! answer does not depend on which tool the model consults.

/// Typed lifecycle snapshot for one goal-backed objective.
pub(crate) struct ObjectiveStatus {
    /// "active", "paused", or "missing".
    pub schedule_state: &'static str,
    /// Earliest next occurrence among active schedules.
    pub next_run_at: Option<String>,
    /// Latest goal run: (status, finished-or-started timestamp, summary).
    pub latest_run: Option<(String, String, Option<String>)>,
    /// Recovery ledger: (disposition, consecutive_failures, failure_budget).
    pub recovery: Option<(&'static str, u16, u16)>,
    /// Terminal run history counts (bounded by the store's returned window).
    pub runs_completed: usize,
    pub runs_failed: usize,
    /// Delegated control-loop facet; `None` means no mandate governs this
    /// goal, which every surface must render as an explicit "control absent"
    /// rather than leaving the question unanswerable.
    pub control: Option<ObjectiveControlFacet>,
}

/// Typed delegation/control readout for one goal-backed objective, derived
/// from the goal's mandate row and its recorded objective measurements.
pub(crate) struct ObjectiveControlFacet {
    pub mandate_status: String,
    pub autonomy_mode: String,
    /// Owner-confirmed control-loop metric, when the mandate has one.
    pub metric_name: Option<String>,
    pub measurement_count: usize,
    /// Distinct `account:` / `auth_profile:` identities from the mandate's
    /// authority scopes: the typed answer to "which owned account and
    /// credentials is this objective delegated to?".
    pub delegated_identities: Vec<String>,
}

/// Project the goal's mandate row into the control facet. `None` in means no
/// mandate exists for the goal; measurement rows without a mandate cannot
/// exist, so the absent case always carries zero measurements.
pub(crate) fn objective_control_facet(
    mandate: Option<&crate::traits::Mandate>,
    measurement_count: usize,
) -> Option<ObjectiveControlFacet> {
    mandate.map(|mandate| {
        let mut delegated_identities = Vec::new();
        for scope in &mandate.authority.operation_scopes {
            for prefix in &scope.target_prefixes {
                if (prefix.starts_with("account:") || prefix.starts_with("auth_profile:"))
                    && !delegated_identities.contains(prefix)
                {
                    delegated_identities.push(prefix.clone());
                }
            }
        }
        ObjectiveControlFacet {
            mandate_status: mandate.status.to_string(),
            autonomy_mode: mandate.autonomy_mode.to_string(),
            metric_name: mandate
                .objective_control
                .as_ref()
                .map(|control| control.metric_name.clone()),
            measurement_count,
            delegated_identities,
        }
    })
}

/// The control facet as a typed JSON object for structured surfaces. The
/// absent case is an explicit value, never a missing key, so an audit can
/// distinguish "no mandate" from "not inspected".
pub(crate) fn objective_control_json(control: Option<&ObjectiveControlFacet>) -> serde_json::Value {
    match control {
        Some(facet) => serde_json::json!({
            "mandate_status": facet.mandate_status,
            "autonomy_mode": facet.autonomy_mode,
            "metric": facet.metric_name,
            "measurement_count": facet.measurement_count,
            "delegated_identities": facet.delegated_identities,
        }),
        None => serde_json::json!({
            "mandate_status": "absent",
            "metric": serde_json::Value::Null,
            "measurement_count": 0,
            "delegated_identities": [],
        }),
    }
}

/// Aggregate the typed rows into the snapshot. Pure so every surface —
/// whatever store facade it holds — fetches its own rows and projects the
/// identical summary.
pub(crate) fn objective_status(
    schedules: &[crate::traits::GoalSchedule],
    runs: &[crate::traits::GoalRun],
    recovery: Option<&crate::traits::ScheduledRecoveryState>,
    mandate: Option<&crate::traits::Mandate>,
    measurement_count: usize,
) -> ObjectiveStatus {
    let active_count = schedules
        .iter()
        .filter(|schedule| !schedule.is_paused)
        .count();
    let schedule_state = if schedules.is_empty() {
        "missing"
    } else if active_count > 0 {
        "active"
    } else {
        "paused"
    };
    let next_run_at = schedules
        .iter()
        .filter(|schedule| !schedule.is_paused)
        .map(|schedule| schedule.next_run_at.clone())
        .min();
    let latest_run = runs.first().map(|run| {
        (
            run.status.clone(),
            run.completed_at
                .clone()
                .unwrap_or_else(|| run.started_at.clone()),
            run.outcome_summary.clone(),
        )
    });
    let runs_completed = runs.iter().filter(|run| run.status == "completed").count();
    let runs_failed = runs
        .iter()
        .filter(|run| matches!(run.status.as_str(), "failed" | "blocked" | "cancelled"))
        .count();
    let recovery = recovery.map(|recovery| {
        (
            recovery.disposition.as_str(),
            recovery.consecutive_failures,
            recovery.failure_budget,
        )
    });
    ObjectiveStatus {
        schedule_state,
        next_run_at,
        latest_run,
        recovery,
        runs_completed,
        runs_failed,
        control: objective_control_facet(mandate, measurement_count),
    }
}

/// Render the snapshot as a single plain-text line for list surfaces.
pub(crate) fn render_objective_status_line(status: &ObjectiveStatus) -> String {
    let mut line = format!("schedule {}", status.schedule_state);
    if let Some(next) = &status.next_run_at {
        line.push_str(&format!(" (next {next})"));
    }
    match &status.latest_run {
        Some((run_status, at, summary)) => {
            line.push_str(&format!("; last run {run_status} at {at}"));
            if let Some(summary) = summary {
                let summary = summary.trim();
                if !summary.is_empty() {
                    line.push_str(" — ");
                    line.push_str(&crate::utils::truncate_str(summary, 140));
                }
            }
        }
        None => line.push_str("; no runs recorded"),
    }
    if let Some((disposition, failures, budget)) = &status.recovery {
        line.push_str(&format!(
            "; recovery {disposition} ({failures}/{budget} failures)"
        ));
    }
    line.push_str(&format!(
        "; run history {} completed / {} failed",
        status.runs_completed, status.runs_failed
    ));
    match &status.control {
        Some(control) => {
            let metric = control
                .metric_name
                .as_deref()
                .map(|name| format!("metric {name}"))
                .unwrap_or_else(|| "no metric loop".to_string());
            let identities = if control.delegated_identities.is_empty() {
                "no delegated account".to_string()
            } else {
                format!("delegated to {}", control.delegated_identities.join(", "))
            };
            line.push_str(&format!(
                "; control {} ({}; {}; {}; {} measurements)",
                control.mandate_status,
                control.autonomy_mode,
                metric,
                identities,
                control.measurement_count
            ));
        }
        None => line.push_str(
            "; control absent (no mandate; no delegated account or credentials; 0 measurements)",
        ),
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::store_prelude::*;
    use crate::traits::{Goal, GoalSchedule};

    #[tokio::test]
    async fn objective_status_projects_schedule_run_and_recovery_from_typed_state() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("publish synthetic report", "session-a", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: "2026-09-01T10:00:00+00:00".to_string(),
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();
        let run = state
            .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
            .await
            .unwrap();
        assert!(state
            .finish_goal_run(
                &run.id,
                "completed",
                Some("Published and verified the daily post.")
            )
            .await
            .unwrap());

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        let runs = state.get_goal_runs(&goal.id).await.unwrap();
        let recovery = state.get_scheduled_recovery_state(&goal.id).await.unwrap();
        let status = objective_status(&schedules, &runs, recovery.as_ref(), None, 0);
        assert_eq!(status.schedule_state, "active");
        assert_eq!(
            status.next_run_at.as_deref(),
            Some("2026-09-01T10:00:00+00:00")
        );
        let (run_status, _, summary) = status.latest_run.as_ref().unwrap();
        assert_eq!(run_status, "completed");
        assert_eq!(
            summary.as_deref(),
            Some("Published and verified the daily post.")
        );
        assert_eq!(status.runs_completed, 1);
        assert_eq!(status.runs_failed, 0);

        let line = render_objective_status_line(&status);
        assert!(line.contains("schedule active (next 2026-09-01T10:00:00+00:00)"));
        assert!(line.contains("last run completed"));
        assert!(line.contains("Published and verified the daily post."));
        assert!(line.contains("run history 1 completed / 0 failed"));
        // Without a mandate the control question still has a typed answer:
        // explicitly absent with zero measurements, never silently missing.
        assert!(status.control.is_none());
        assert!(line.contains(
            "control absent (no mandate; no delegated account or credentials; 0 measurements)"
        ));
        let control_json = objective_control_json(None);
        assert_eq!(control_json["mandate_status"], "absent");
        assert_eq!(control_json["measurement_count"], 0);
        assert_eq!(control_json["delegated_identities"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn objective_status_reports_paused_schedule_and_recovery_disposition() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("publish synthetic report", "session-a", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: now.clone(),
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();
        // Three failed runs escalate and pause through the recovery ledger.
        for _ in 0..3 {
            let run = state
                .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
                .await
                .unwrap();
            assert!(state
                .finish_goal_run(&run.id, "failed", Some("synthetic failure"))
                .await
                .unwrap());
        }
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        let runs = state.get_goal_runs(&goal.id).await.unwrap();
        let recovery = state.get_scheduled_recovery_state(&goal.id).await.unwrap();
        let status = objective_status(&schedules, &runs, recovery.as_ref(), None, 0);
        assert_eq!(status.schedule_state, "paused");
        assert_eq!(status.next_run_at, None);
        let (disposition, failures, _) = status.recovery.unwrap();
        assert_eq!(disposition, "escalated");
        assert_eq!(failures, 3);
        assert_eq!(status.runs_failed, 3);
        let line = render_objective_status_line(&status);
        assert!(line.contains("schedule paused"));
        assert!(line.contains("recovery escalated (3/3 failures)"));
    }

    #[tokio::test]
    async fn objective_status_projects_mandate_control_loop_and_measurements() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("publish synthetic report", "session-a", None, None);
        state.create_goal(&goal).await.unwrap();
        let authority = crate::traits::MandateAuthority::from_operation_scopes(
            true,
            serde_json::from_value(serde_json::json!([
                {
                    "tool": "http_request",
                    "operation": "POST",
                    "kind": "mutation",
                    "target_prefixes": [
                        "https://api.x.com/2/tweets",
                        "auth_profile:twitter",
                        "account:12345"
                    ],
                    "mutation_effects": ["remote_mutation", "external_delivery"]
                }
            ]))
            .unwrap(),
            1,
            4,
            900,
        );
        let mut mandate = crate::traits::Mandate::new(
            &goal.id,
            None,
            "publish synthetic report",
            "session-a",
            authority,
            3_600,
            21_600,
            10_800,
        );
        mandate.objective_control = Some(crate::traits::MandateObjectiveControl {
            schema_version: crate::traits::MandateObjectiveControl::SCHEMA_VERSION,
            metric_name: "daily_visits".to_string(),
            unit: "visits".to_string(),
            baseline_micros: 0,
            target_micros: 1_000_000,
            direction: crate::traits::ObjectiveMetricDirection::AtLeast,
            measurement_source: "https://analytics.example.com/api".to_string(),
            measurement_cadence_secs: 7_200,
            experiment_cohort: "cohort-a".to_string(),
            experiment_window_secs: 604_800,
            minimum_effect_micros: 1,
            max_stagnant_measurements: 5,
            run_failure_budget: 3,
            baseline_observed_at: chrono::Utc::now().to_rfc3339(),
        });

        let status = objective_status(&[], &[], None, Some(&mandate), 3);
        let control = status.control.as_ref().unwrap();
        assert_eq!(control.mandate_status, "active");
        assert_eq!(control.autonomy_mode, "bounded");
        assert_eq!(control.metric_name.as_deref(), Some("daily_visits"));
        assert_eq!(control.measurement_count, 3);
        assert_eq!(
            control.delegated_identities,
            vec![
                "auth_profile:twitter".to_string(),
                "account:12345".to_string()
            ]
        );

        let line = render_objective_status_line(&status);
        assert!(
            line.contains(
                "control active (bounded; metric daily_visits; delegated to auth_profile:twitter, account:12345; 3 measurements)"
            ),
            "{line}"
        );
        let json = objective_control_json(status.control.as_ref());
        assert_eq!(json["mandate_status"], "active");
        assert_eq!(json["metric"], "daily_visits");
        assert_eq!(json["measurement_count"], 3);
        assert_eq!(
            json["delegated_identities"],
            serde_json::json!(["auth_profile:twitter", "account:12345"])
        );

        // A mandate without an owner-confirmed metric loop still projects a
        // typed answer rather than an unknown.
        mandate.objective_control = None;
        mandate.authority.operation_scopes.clear();
        let status = objective_status(&[], &[], None, Some(&mandate), 0);
        let line = render_objective_status_line(&status);
        assert!(
            line.contains(
                "control active (bounded; no metric loop; no delegated account; 0 measurements)"
            ),
            "{line}"
        );
    }
}
