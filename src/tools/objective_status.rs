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
}

/// Aggregate the typed rows into the snapshot. Pure so every surface —
/// whatever store facade it holds — fetches its own rows and projects the
/// identical summary.
pub(crate) fn objective_status(
    schedules: &[crate::traits::GoalSchedule],
    runs: &[crate::traits::GoalRun],
    recovery: Option<&crate::traits::ScheduledRecoveryState>,
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
        let status = objective_status(&schedules, &runs, recovery.as_ref());
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
        let status = objective_status(&schedules, &runs, recovery.as_ref());
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
}
