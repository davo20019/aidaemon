//! `manage_automations` lives in `aidaemon_runtime::tools::manage_automations`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_runtime::tools::manage_automations::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::CallWithFacts;
    use crate::testing::{MockProvider, setup_test_agent};
    use crate::tools::ManageMandatesTool;
    use crate::traits::StateStore;
    use crate::traits::ToolCallContext;
    use crate::traits::ToolCalls as _;
    use crate::types::ChannelVisibility;
    use crate::types::UserRole;
    use aidaemon_work::automations;
    use serde_json::Value;
    use serde_json::json;
    use std::sync::Arc;

    fn owner() -> ToolCallContext {
        ToolCallContext {
            session_id: "owner-session".into(),
            user_role: Some(UserRole::Owner),
            channel_visibility: Some(ChannelVisibility::Private),
            ..Default::default()
        }
    }

    fn approved_owner() -> ToolCallContext {
        ToolCallContext {
            kernel_approved: Some(crate::traits::ApprovalScope::Call),
            ..owner()
        }
    }

    struct Fixture {
        state: Arc<dyn StateStore>,
        tool: ManageAutomationsTool,
        /// Scheduled automations by name, recorded when created.
        scheduled: std::sync::Mutex<std::collections::HashMap<String, String>>,
        /// Keeps the test database alive.
        _harness: crate::testing::TestHarness,
    }

    async fn fixture() -> Fixture {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state: Arc<dyn StateStore> = harness.state.clone();
        let tool = ManageAutomationsTool::new(state.clone());
        Fixture {
            state,
            tool,
            scheduled: Default::default(),
            _harness: harness,
        }
    }

    impl Fixture {
        async fn reminder(&self, name: &str, when: &str) -> String {
            let output = crate::tools::SetReminderTool::new(self.state.clone())
                .call_with(
                    &json!({"message": "Check the oven.", "when": when, "name": name}).to_string(),
                    owner(),
                )
                .await
                .unwrap();
            self.record(name).await;
            output
        }

        /// Remember the goal now carrying `name`, if one does.
        async fn record(&self, name: &str) {
            let found = self
                .state
                .get_scheduled_goals()
                .await
                .unwrap()
                .into_iter()
                .find(|goal| {
                    automations::name_from_context(goal.context.as_deref()).as_deref() == Some(name)
                });
            if let Some(goal) = found {
                self.scheduled
                    .lock()
                    .unwrap()
                    .insert(name.to_string(), goal.id);
            }
        }

        async fn scheduled_work(&self, name: &str, schedule: &str) -> String {
            let output = crate::tools::ScheduleTaskTool::new(self.state.clone())
                .call_with(
                    &json!({
                        "task": "Summarize the synthetic inbox",
                        "when": schedule,
                        "name": name,
                    })
                    .to_string(),
                    approved_owner(),
                )
                .await
                .unwrap();
            self.record(name).await;
            output
        }

        async fn watching(&self, name: &str) -> String {
            ManageMandatesTool::new(self.state.clone())
                .call_with(
                    &json!({
                        "action": "create",
                        "name": name,
                        "objective": "Watch the synthetic status page",
                        "max_mutating_actions_per_cycle": 0,
                        "min_review_minutes": 60,
                        "max_review_minutes": 60,
                        "default_review_minutes": 60,
                        "success_criteria": ["The status page is reviewed without unauthorized action"],
                    })
                    .to_string(),
                    approved_owner(),
                )
                .await
                .unwrap()
        }

        async fn call(&self, args: Value) -> String {
            self.tool
                .call_with(&args.to_string(), owner())
                .await
                .unwrap()
        }

        async fn scheduled_status(&self, name: &str) -> String {
            let id = self.scheduled.lock().unwrap().get(name).cloned();
            let id = id.unwrap_or_else(|| panic!("no scheduled automation named {name}"));
            self.state.get_goal(&id).await.unwrap().unwrap().status
        }

        async fn mandate_status(&self, name: &str) -> String {
            self.state
                .list_mandates(Some("owner-session"), true)
                .await
                .unwrap()
                .into_iter()
                .find(|mandate| mandate.has_name(name))
                .map(|mandate| mandate.status.as_str().to_string())
                .unwrap_or_default()
        }
    }

    #[tokio::test]
    async fn one_list_shows_both_kinds_with_cadence_state_and_next_run() {
        let f = fixture().await;
        assert!(
            f.reminder("Oven", "every weekday at 8am")
                .await
                .starts_with("Reminder set")
        );
        let work = f.scheduled_work("Inbox digest", "every day at 9am").await;
        assert!(work.starts_with("Scheduled \"Inbox digest\""), "{work}");
        let watch = f.watching("Status watch").await;
        assert!(watch.contains("Activated mandate"), "{watch}");

        let list = f.call(json!({"action": "list"})).await;
        assert!(list.starts_with("Your automations (3):"), "{list}");
        for row in [
            "\"Oven\" · Scheduled · Every weekday at 8:00 AM",
            "\"Inbox digest\" · Scheduled · Every day at 9:00 AM",
            "\"Status watch\" · Watching · Every hour · On · next ",
        ] {
            assert!(list.contains(row), "missing {row:?} in {list}");
        }
        assert!(
            !list.contains("mandate") && !list.contains("goal"),
            "{list}"
        );

        // The typed kind narrows the list.
        let watching = f.call(json!({"action": "list", "kind": "watching"})).await;
        assert!(
            watching.starts_with("Your watching automations (1):"),
            "{watching}"
        );
        assert!(!watching.contains("Oven"), "{watching}");
    }

    #[tokio::test]
    async fn show_gives_the_same_five_line_card_for_both_kinds() {
        let f = fixture().await;
        f.scheduled_work("Inbox digest", "every day at 9am").await;
        f.watching("Status watch").await;

        let scheduled = f
            .call(json!({"action": "show", "name": "inbox digest"}))
            .await;
        for line in [
            "\"Inbox digest\" (Scheduled)",
            "What: Summarize the synthetic inbox",
            "How often: Every day at 9:00 AM",
            "Can do: Anything you can do – it runs with your full access",
            "Tells you when: a run finishes, or it hits a problem",
            "Ends: Only when you stop it",
            "State: On · next ",
        ] {
            assert!(scheduled.contains(line), "missing {line:?} in {scheduled}");
        }
        let watching = f
            .call(json!({"action": "show", "name": "Status watch"}))
            .await;
        for line in [
            "\"Status watch\" (Watching)",
            "What: Watch the synthetic status page",
            "How often: Every hour",
            "Can do: Read only – makes no changes",
            "Tells you when: ",
            "Ends: Only when you stop it",
        ] {
            assert!(watching.contains(line), "missing {line:?} in {watching}");
        }
    }

    #[tokio::test]
    async fn cancel_by_name_works_for_each_kind() {
        let f = fixture().await;
        f.reminder("Oven", "every weekday at 8am").await;
        f.watching("Status watch").await;

        let cancelled = f.call(json!({"action": "cancel", "name": "oven"})).await;
        assert_eq!(
            cancelled,
            "Cancelled \"Oven\" (scheduled). It won't run again."
        );
        assert_eq!(f.scheduled_status("Oven").await, "cancelled");

        let cancelled = f
            .call(json!({"action": "cancel", "name": "Status watch"}))
            .await;
        assert_eq!(
            cancelled,
            "Cancelled \"Status watch\" (watching). It won't run again."
        );
        assert_eq!(f.mandate_status("Status watch").await, "cancelled");

        let list = f.call(json!({"action": "list"})).await;
        assert_eq!(list, "Your automations: none.");
        let missing = f.call(json!({"action": "pause", "name": "Oven"})).await;
        assert!(
            missing.starts_with("No automation is named \"Oven\""),
            "{missing}"
        );
    }

    #[tokio::test]
    async fn pause_and_resume_by_name_round_trip_for_each_kind() {
        let f = fixture().await;
        f.scheduled_work("Inbox digest", "every day at 9am").await;
        f.watching("Status watch").await;
        for (name, kind) in [("Inbox digest", "scheduled"), ("Status watch", "watching")] {
            let paused = f.call(json!({"action": "pause", "name": name})).await;
            assert_eq!(
                paused,
                format!("Paused \"{name}\" ({kind}). It won't run until you resume it.")
            );
            let again = f.call(json!({"action": "pause", "name": name})).await;
            assert_eq!(
                again,
                format!("Couldn't pause \"{name}\" ({kind}): it is Paused.")
            );
            let resumed = f.call(json!({"action": "resume", "name": name})).await;
            assert!(
                resumed.starts_with(&format!("Resumed \"{name}\" ({kind}).")),
                "{resumed}"
            );
        }
        assert_eq!(f.scheduled_status("Inbox digest").await, "active");
        assert_eq!(f.mandate_status("Status watch").await, "active");
    }

    #[tokio::test]
    async fn the_typed_kind_picks_between_two_automations_with_one_name() {
        let f = fixture().await;
        f.reminder("Prices", "every weekday at 8am").await;
        f.watching("Prices").await;

        let ambiguous = f.call(json!({"action": "pause", "name": "Prices"})).await;
        assert!(
            ambiguous.starts_with("More than one automation matches"),
            "{ambiguous}"
        );
        assert_eq!(f.scheduled_status("Prices").await, "active");
        assert_eq!(f.mandate_status("Prices").await, "active");

        let paused = f
            .call(json!({"action": "pause", "name": "Prices", "kind": "watching"}))
            .await;
        assert!(
            paused.starts_with("Paused \"Prices\" (watching)"),
            "{paused}"
        );
        assert_eq!(f.mandate_status("Prices").await, "paused");
        assert_eq!(f.scheduled_status("Prices").await, "active");

        let cancelled = f
            .call(json!({"action": "cancel", "name": "Prices", "kind": "scheduled"}))
            .await;
        assert!(
            cancelled.starts_with("Cancelled \"Prices\" (scheduled)"),
            "{cancelled}"
        );
        assert_eq!(f.mandate_status("Prices").await, "paused");
    }

    #[tokio::test]
    async fn a_name_already_in_use_is_refused_and_unnamed_ones_go_by_ref() {
        let f = fixture().await;
        f.watching("Oven").await;
        let refused = f.reminder("oven", "in 2h").await;
        assert!(
            refused.starts_with("An automation is already named \"oven\" (a watching one)"),
            "{refused}"
        );

        let set = crate::tools::SetReminderTool::new(f.state.clone())
            .call_with(
                &json!({"message": "Stretch.", "when": "every day at 3pm"}).to_string(),
                owner(),
            )
            .await
            .unwrap();
        assert!(set.starts_with("Reminder set"), "{set}");
        let list = f.call(json!({"action": "list", "kind": "scheduled"})).await;
        let row = list
            .lines()
            .find(|line| line.contains("Reminds you: Stretch."))
            .unwrap_or_else(|| panic!("{list}"));
        let reference = row
            .rsplit_once("[ref ")
            .and_then(|(_, rest)| rest.strip_suffix(']'))
            .unwrap();
        let cancelled = f.call(json!({"action": "cancel", "id": reference})).await;
        assert!(
            cancelled.starts_with("Cancelled Reminds you: Stretch."),
            "{cancelled}"
        );
    }

    #[tokio::test]
    async fn only_the_owner_in_a_private_chat_sees_automations() {
        let f = fixture().await;
        for facts in [
            ToolCallContext {
                user_role: Some(UserRole::Guest),
                ..owner()
            },
            ToolCallContext {
                channel_visibility: Some(ChannelVisibility::Public),
                ..owner()
            },
            ToolCallContext {
                channel_visibility: Some(ChannelVisibility::Internal),
                ..owner()
            },
        ] {
            let answer = f
                .tool
                .call_with(&json!({"action": "list"}).to_string(), facts)
                .await
                .unwrap();
            assert_eq!(answer, OWNER_ONLY);
        }
    }

    #[tokio::test]
    async fn edit_reschedules_renames_and_snoozes_one_scheduled_automation() {
        let f = fixture().await;
        f.scheduled_work("Inbox digest", "every day at 9am").await;

        let edited = f
            .call(json!({"action": "edit", "name": "Inbox digest", "when": "every weekday at 7am", "new_name": "Morning digest"}))
            .await;
        assert!(
            edited.starts_with("Edited \"Inbox digest\": renamed it \"Morning digest\"; its schedule is now Every weekday at 7:00 AM"),
            "{edited}"
        );
        let id = f
            .scheduled
            .lock()
            .unwrap()
            .get("Inbox digest")
            .cloned()
            .unwrap();
        let schedules = f.state.get_schedules_for_goal(&id).await.unwrap();
        assert_eq!(schedules.len(), 1);
        assert_eq!(schedules[0].cron_expr, "0 7 * * 1-5");
        let shown = f
            .call(json!({"action": "show", "name": "morning digest"}))
            .await;
        assert!(
            shown.contains("How often: Every weekday at 7:00 AM"),
            "{shown}"
        );

        let snoozed = f
            .call(json!({"action": "edit", "name": "Morning digest", "snooze_until": "in 3 days"}))
            .await;
        assert!(snoozed.contains("its next run moved to"), "{snoozed}");
        let schedule = f.state.get_schedules_for_goal(&id).await.unwrap().remove(0);
        // The cadence stays; only the next run moved.
        assert_eq!(schedule.cron_expr, "0 7 * * 1-5");
        let next = chrono::DateTime::parse_from_rfc3339(&schedule.next_run_at).unwrap();
        assert!(
            next > chrono::Utc::now() + chrono::Duration::days(2),
            "{next}"
        );

        // A recurring snooze, a one-time reschedule of recurring work and a
        // name already taken are refused, and change nothing.
        for (args, refusal) in [
            (
                json!({"action": "edit", "name": "Morning digest", "snooze_until": "every day at 8am"}),
                "`snooze_until` is one time",
            ),
            (
                json!({"action": "edit", "name": "Morning digest", "when": "tomorrow at 9am"}),
                "recurs; to make it a one-time one",
            ),
        ] {
            let answer = f.call(args).await;
            assert!(answer.contains(refusal), "{answer}");
        }
        f.watching("Status watch").await;
        let taken = f
            .call(json!({"action": "edit", "name": "Morning digest", "new_name": "Status watch"}))
            .await;
        assert!(
            taken.starts_with("An automation is already named"),
            "{taken}"
        );
        assert_eq!(
            f.state.get_schedules_for_goal(&id).await.unwrap()[0].cron_expr,
            "0 7 * * 1-5"
        );
    }

    #[tokio::test]
    async fn several_refs_cancel_together_and_nothing_cancels_by_matching_words() {
        let f = fixture().await;
        f.reminder("Oven", "every weekday at 8am").await;
        f.reminder("Stretch", "every day at 3pm").await;
        f.reminder("Water", "every day at 4pm").await;
        let refs: Vec<String> = ["Oven", "Stretch"]
            .iter()
            .map(|name| f.scheduled.lock().unwrap().get(*name).cloned().unwrap()[..8].to_string())
            .collect();
        let cancelled = f.call(json!({"action": "cancel", "ids": refs})).await;
        assert_eq!(cancelled.lines().count(), 2, "{cancelled}");
        assert_eq!(f.scheduled_status("Oven").await, "cancelled");
        assert_eq!(f.scheduled_status("Stretch").await, "cancelled");
        assert_eq!(f.scheduled_status("Water").await, "active");
        // A word from a name is not a ref.
        let outcome = f
            .tool
            .call_with_execution_context(
                &json!({"action": "cancel", "ids": ["Water"]}).to_string(),
                None,
                owner(),
            )
            .await
            .unwrap();
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult)
        );
        assert_eq!(f.scheduled_status("Water").await, "active");
    }

    #[tokio::test]
    async fn run_now_starts_a_run_of_named_scheduled_work() {
        let f = fixture().await;
        f.scheduled_work("Inbox digest", "every day at 9am").await;
        let outcome = f
            .tool
            .call_with_execution_context(
                &json!({"action": "run_now", "name": "Inbox digest"}).to_string(),
                None,
                owner(),
            )
            .await
            .unwrap();
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::Succeeded),
            "{}",
            outcome.output
        );
        let id = f
            .scheduled
            .lock()
            .unwrap()
            .get("Inbox digest")
            .cloned()
            .unwrap();
        assert_eq!(f.state.get_tasks_for_goal(&id).await.unwrap().len(), 1);
        // A paused one does not run, and that is no success.
        f.call(json!({"action": "pause", "name": "Inbox digest"}))
            .await;
        let refused = f
            .tool
            .call_with_execution_context(
                &json!({"action": "run_now", "id": id}).to_string(),
                None,
                owner(),
            )
            .await
            .unwrap();
        assert_ne!(
            refused.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::Succeeded),
            "{}",
            refused.output
        );
    }

    #[tokio::test]
    async fn health_checks_are_listed_paused_and_cancelled_here() {
        let f = fixture().await;
        let store = Arc::new(
            crate::health::HealthProbeStore::new(f._harness.state.pool())
                .await
                .unwrap(),
        );
        let tool =
            ManageAutomationsTool::new(f.state.clone()).with_health_probes(Some(store.clone()));
        let mut probe = crate::health::HealthProbe::new(
            "Synthetic site".to_string(),
            crate::health::ProbeType::Http,
            "https://status.example.test/".to_string(),
            "*/5 * * * *".to_string(),
            "tool".to_string(),
        );
        probe.next_run_at = chrono::Utc::now() + chrono::Duration::minutes(5);
        store.upsert_probe(&probe).await.unwrap();
        let call = |args: Value| {
            let tool = &tool;
            async move { tool.call_with(&args.to_string(), owner()).await.unwrap() }
        };
        let list = call(json!({"action": "list", "kind": "probe"})).await;
        assert!(
            list.contains("\"Synthetic site\" · Health check · Every 5 minutes · On"),
            "{list}"
        );
        let paused = call(json!({"action": "pause", "name": "synthetic site"})).await;
        assert!(
            paused.starts_with("Paused \"Synthetic site\" (probe)"),
            "{paused}"
        );
        assert!(store.get_probe(&probe.id).await.unwrap().unwrap().is_paused);
        let cancelled = call(json!({"action": "cancel", "name": "Synthetic site"})).await;
        assert!(
            cancelled.starts_with("Cancelled \"Synthetic site\" (probe)"),
            "{cancelled}"
        );
        assert!(store.get_probe(&probe.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_moved_schedule_action_names_its_new_home_and_runs_nothing() {
        let f = fixture().await;
        f.reminder("Oven", "every weekday at 8am").await;
        let id = f.scheduled.lock().unwrap().get("Oven").cloned().unwrap();
        let memories = crate::tools::ManageMemoriesTool::new(
            f.state.clone(),
            f._harness.state.memory_stores(),
        );
        let outcome = memories
            .call_with_execution_context(
                &json!({"action": "cancel_scheduled", "goal_id": id}).to_string(),
                None,
                owner(),
            )
            .await
            .unwrap();
        assert_eq!(
            outcome.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::FailedPermanent)
        );
        assert!(
            outcome.output.contains("Use manage_automations cancel"),
            "{}",
            outcome.output
        );
        assert_eq!(f.scheduled_status("Oven").await, "active");
    }
}
