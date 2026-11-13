//! The objective lifecycle readout lives in
//! `aidaemon_work::tools::objective_status`; the tests here run it against the
//! daemon's assembled state store.

#[cfg(test)]
pub(crate) use aidaemon_work::tools::objective_status::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::store_prelude::*;
    use crate::traits::ToolCalls as _;
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
        let status = objective_portfolio_row(
            &goal.id,
            ObjectiveCollection::ScheduledGoals,
            &schedules,
            &runs,
            recovery.as_ref(),
            None,
            0,
        )
        .unwrap()
        .status;
        assert_eq!(status.wake.state(), "active");
        assert_eq!(status.wake.next_at(), Some("2026-09-01T10:00:00+00:00"));
        let (run_status, _, summary) = status.latest_run.as_ref().unwrap();
        assert_eq!(run_status, "completed");
        assert_eq!(
            summary.as_deref(),
            Some("Published and verified the daily post.")
        );
        assert_eq!(status.runs_completed, 1);
        assert_eq!(status.runs_failed, 0);

        let line = render_objective_status_line(&status);
        // The owner reads the time in their own zone and format.
        let shown = crate::cron_utils::owner_time_rfc3339("2026-09-01T10:00:00+00:00").unwrap();
        assert!(
            line.contains(&format!("schedule active (next {shown})")),
            "{line}"
        );
        assert!(!line.contains("2026-09-01T10:00:00"), "{line}");
        assert!(line.contains("last run completed"));
        assert!(line.contains("Published and verified the daily post."));
        assert!(line.contains("run history 1 completed / 0 failed"));
        // Without a mandate the control question still has a typed answer:
        // explicitly absent with zero measurements, never silently missing.
        assert!(status.control.is_none());
        assert!(line.contains(
            "control absent (no mandate; no delegated account or credentials; 0 measurements)"
        ));
        assert_eq!(objective_control_json(None), serde_json::json!("absent"));
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
        let status = objective_portfolio_row(
            &goal.id,
            ObjectiveCollection::ScheduledGoals,
            &schedules,
            &runs,
            recovery.as_ref(),
            None,
            0,
        )
        .unwrap()
        .status;
        assert_eq!(status.wake.state(), "paused");
        assert_eq!(status.wake.next_at(), None);
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
            measurement_source: "metric_source:synthetic-analytics".to_string(),
            measurement_cadence_secs: 7_200,
            baseline_observed_at: chrono::Utc::now().to_rfc3339(),
        });

        let status = objective_portfolio_row(
            &goal.id,
            ObjectiveCollection::MandateControllers,
            &[],
            &[],
            None,
            Some(&mandate),
            3,
        )
        .unwrap()
        .status;
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
        let status = objective_portfolio_row(
            &goal.id,
            ObjectiveCollection::MandateControllers,
            &[],
            &[],
            None,
            Some(&mandate),
            0,
        )
        .unwrap()
        .status;
        let line = render_objective_status_line(&status);
        assert!(
            line.contains(
                "control active (bounded; no metric loop; no delegated account; 0 measurements)"
            ),
            "{line}"
        );
    }

    /// A mandate controller has no goal schedule by design: it wakes from its
    /// own clock while active, and from nothing once it is not.
    #[test]
    fn a_mandate_wakes_from_its_own_clock_or_not_at_all() {
        let mut mandate = crate::traits::Mandate::new(
            "goal-controller",
            None,
            "Keep the Acme Corp account useful",
            "telegram:synthetic-user-1",
            crate::traits::MandateAuthority::default(),
            900,
            3600,
            1800,
        );
        mandate.confirmed_at = Some(chrono::Utc::now().to_rfc3339());
        let status = |mandate: &crate::traits::Mandate| {
            objective_portfolio_row(
                "goal-controller",
                ObjectiveCollection::MandateControllers,
                &[],
                &[],
                None,
                Some(mandate),
                0,
            )
            .unwrap()
            .status
        };

        let active = status(&mandate);
        assert_eq!(active.wake.source(), "mandate_clock");
        assert_eq!(active.wake.next_at(), Some(mandate.next_review_at.as_str()));
        let line = render_objective_status_line(&active);
        assert!(line.starts_with("mandate clock (next review"), "{line}");
        assert!(!line.contains("missing"), "{line}");

        mandate.status = crate::traits::MandateStatus::AwaitingInput;
        mandate.suspension = Some(crate::traits::MandateSuspension::new(
            crate::traits::MandateSuspensionKind::ReconciliationRequired,
            Some("act_blocked_before_dispatch:untrusted_content_check".to_string()),
        ));
        let suspended = status(&mandate);
        assert_eq!(suspended.wake.source(), "none");
        assert_eq!(suspended.wake.next_at(), None);
        assert_eq!(mandate.scheduled_next_review(), None);
        let line = render_objective_status_line(&suspended);
        assert!(
            line.starts_with(
                "suspended (act_blocked_before_dispatch:untrusted_content_check), no review scheduled"
            ),
            "{line}"
        );
    }

    #[test]
    fn portfolio_keeps_unrelated_collection_subjects_distinct_and_scopes_absence() {
        let mut scheduled_goal =
            Goal::new_continuous("Publish the Acme digest", "owner-session", None, None);
        scheduled_goal.id = "goal-scheduled-a".to_string();
        let mut controller_goal =
            Goal::new_continuous("Review Acme engagement", "owner-session", None, None);
        controller_goal.id = "goal-controller-b".to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: "schedule-a".to_string(),
            goal_id: scheduled_goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: "2026-09-02T10:00:00+00:00".to_string(),
            created_at: now.clone(),
            updated_at: now,
        };
        let mandate = crate::traits::Mandate::new(
            &controller_goal.id,
            None,
            "Review Acme engagement",
            "owner-session",
            crate::traits::MandateAuthority::default(),
            3_600,
            21_600,
            10_800,
        );

        let scheduled_row = objective_portfolio_row(
            &scheduled_goal.id,
            ObjectiveCollection::ScheduledGoals,
            std::slice::from_ref(&schedule),
            &[],
            None,
            None,
            0,
        )
        .unwrap();
        let controller_row = objective_portfolio_row(
            &controller_goal.id,
            ObjectiveCollection::MandateControllers,
            &[],
            &[],
            None,
            Some(&mandate),
            1,
        )
        .unwrap();

        // Insert in the reverse of the collection/subject order. Identity,
        // never position, controls the merge.
        let mut portfolio = ObjectivePortfolio::default();
        portfolio
            .record_collection(
                ObjectiveCollectionCoverage::new(ObjectiveCollection::ScheduledGoals, 1, 1)
                    .unwrap(),
            )
            .unwrap();
        portfolio
            .record_collection(
                ObjectiveCollectionCoverage::new(ObjectiveCollection::MandateControllers, 1, 1)
                    .unwrap(),
            )
            .unwrap();
        portfolio.insert(controller_row.clone()).unwrap();
        portfolio.insert(scheduled_row.clone()).unwrap();

        assert_eq!(portfolio.subjects().count(), 2);
        let scheduled = portfolio.subject(&scheduled_goal.id).unwrap();
        assert_eq!(scheduled.status.wake.state(), "active");
        assert!(scheduled.status.control.is_none());
        let controller = portfolio.subject(&controller_goal.id).unwrap();
        assert_eq!(controller.status.wake.state(), "mandate_clock");
        assert_eq!(
            controller
                .status
                .control
                .as_ref()
                .map(|control| control.measurement_count),
            Some(1)
        );
        assert_eq!(
            portfolio.membership(&scheduled_goal.id, ObjectiveCollection::ScheduledGoals),
            ObjectiveCollectionMembership::Present
        );
        assert_eq!(
            portfolio.membership(&scheduled_goal.id, ObjectiveCollection::MandateControllers),
            ObjectiveCollectionMembership::Absent
        );
        assert_eq!(
            portfolio.membership(&controller_goal.id, ObjectiveCollection::ScheduledGoals),
            ObjectiveCollectionMembership::Absent
        );
        assert_eq!(
            portfolio.membership(&controller_goal.id, ObjectiveCollection::MandateControllers),
            ObjectiveCollectionMembership::Present
        );

        // A limit makes non-membership unknown, and never global absence.
        let mut limited = ObjectivePortfolio::default();
        limited
            .record_collection(
                ObjectiveCollectionCoverage::new(ObjectiveCollection::ScheduledGoals, 2, 1)
                    .unwrap(),
            )
            .unwrap();
        limited.insert(scheduled_row).unwrap();
        assert_eq!(
            limited.membership(&controller_goal.id, ObjectiveCollection::ScheduledGoals),
            ObjectiveCollectionMembership::Unknown
        );
        let scope = limited
            .collection_scope_json(ObjectiveCollection::ScheduledGoals)
            .unwrap();
        assert_eq!(scope["complete"], false);
        assert_eq!(scope["absence_semantics"], "unknown_outside_returned_rows");
        assert_eq!(
            scope["not_enumerated"],
            serde_json::json!(["mandate_controllers"])
        );

        // Passing one subject's durable row while naming another fails before
        // any lifecycle or control field can be projected.
        let error = objective_portfolio_row(
            &controller_goal.id,
            ObjectiveCollection::MandateControllers,
            &[schedule],
            &[],
            None,
            Some(&mandate),
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not objective subject"));
    }

    fn scheduled_collection_of(
        metadata: &crate::traits::ToolCallMetadata,
    ) -> crate::traits::ToolCollectionObservation {
        metadata
            .collection_observations
            .iter()
            .find(|collection| {
                collection.collection.value == ObjectiveCollection::ScheduledGoals.resource_id()
            })
            .cloned()
            .expect("scheduled-goal collection coverage")
    }

    #[tokio::test]
    async fn every_owner_surface_enumerates_the_same_scheduled_goal_collection() {
        // R53: an owner audit consulted only `manage_mandates list`, which
        // reported the scheduled-goal collection as `0/0 (complete)` because
        // the one scheduled objective had been created through a sibling
        // channel session of the same owner. `scheduled_goal_runs overview`
        // saw it. One collection ID must have one membership everywhere.
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let now = chrono::Utc::now().to_rfc3339();
        let mut created = Vec::new();
        for (description, session) in [
            ("Publish the synthetic daily digest", "owner-bot-a:1001"),
            ("Refresh the synthetic status page", "owner-bot-b:1001"),
            ("Roll the synthetic weekly summary", "owner-slack-dm:1001"),
        ] {
            let goal = Goal::new_continuous(description, session, None, None);
            state.create_goal(&goal).await.unwrap();
            state
                .create_goal_schedule(&GoalSchedule {
                    id: uuid::Uuid::new_v4().to_string(),
                    goal_id: goal.id.clone(),
                    cron_expr: "0 6 * * *".to_string(),
                    tz: "local".to_string(),
                    original_schedule: Some("daily".to_string()),
                    fire_policy: "coalesce".to_string(),
                    is_one_shot: false,
                    is_paused: false,
                    last_run_at: None,
                    next_run_at: "2026-09-03T10:00:00+00:00".to_string(),
                    created_at: now.clone(),
                    updated_at: now.clone(),
                })
                .await
                .unwrap();
            created.push(objective_resource_id(&goal.id));
        }

        let mandates = crate::tools::manage_mandates::ManageMandatesTool::new(state.clone());
        let runs = crate::tools::scheduled_goal_runs::ScheduledGoalRunsTool::new(state.clone());

        let mandate_view = mandates
            .call_with_execution_context(
                &serde_json::json!({ "action": "list" }).to_string(),
                None,
                crate::traits::ToolCallContext {
                    session_id: "owner-bot-a:1001".into(),
                    user_role: Some(crate::types::UserRole::Owner),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let runs_view = runs
            .call_with_status_outcome(r#"{"action":"overview"}"#, None)
            .await
            .unwrap();

        let from_mandates = scheduled_collection_of(&mandate_view.metadata);
        let from_runs = scheduled_collection_of(&runs_view.metadata);
        assert_eq!(from_mandates.total_count, Some(3));
        assert_eq!(from_mandates.returned_count, 3);
        assert_eq!(
            from_mandates.completeness,
            crate::traits::ToolCollectionCompleteness::Complete
        );
        let mut mandate_members = from_mandates
            .members
            .iter()
            .map(|member| member.value.clone())
            .collect::<Vec<_>>();
        let mut runs_members = from_runs
            .members
            .iter()
            .map(|member| member.value.clone())
            .collect::<Vec<_>>();
        mandate_members.sort();
        runs_members.sort();
        created.sort();
        assert_eq!(mandate_members, created);
        assert_eq!(runs_members, created);
        assert_eq!(from_runs.total_count, from_mandates.total_count);
        assert_eq!(from_runs.completeness, from_mandates.completeness);

        // The rendered audit text names every objective, not just the ones
        // created through the calling session.
        for member in &created {
            assert!(
                mandate_view.output.contains(member),
                "{}",
                mandate_view.output
            );
        }
    }

    #[tokio::test]
    async fn scheduled_goal_collection_reports_truncation_as_partial_coverage() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let now = chrono::Utc::now().to_rfc3339();
        for index in 0..3 {
            let goal = Goal::new_continuous(
                &format!("Synthetic scheduled objective {index}"),
                "owner-bot-a:1001",
                None,
                None,
            );
            state.create_goal(&goal).await.unwrap();
            state
                .create_goal_schedule(&GoalSchedule {
                    id: uuid::Uuid::new_v4().to_string(),
                    goal_id: goal.id.clone(),
                    cron_expr: "0 6 * * *".to_string(),
                    tz: "local".to_string(),
                    original_schedule: Some("daily".to_string()),
                    fire_policy: "coalesce".to_string(),
                    is_one_shot: false,
                    is_paused: false,
                    last_run_at: None,
                    next_run_at: "2026-09-03T10:00:00+00:00".to_string(),
                    created_at: now.clone(),
                    updated_at: now.clone(),
                })
                .await
                .unwrap();
        }

        let (goals, coverage) = load_scheduled_goal_collection(state.as_ref(), 2)
            .await
            .unwrap();
        assert_eq!(goals.len(), 2);
        assert_eq!(coverage.total, 3);
        assert_eq!(coverage.returned, 2);
        assert!(!coverage.is_complete());
        let (goals, coverage) = load_scheduled_goal_collection(state.as_ref(), 50)
            .await
            .unwrap();
        assert_eq!(goals.len(), 3);
        assert!(coverage.is_complete());
    }
}
