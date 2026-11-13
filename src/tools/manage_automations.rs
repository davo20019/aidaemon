//! `manage_automations` lives in `aidaemon_runtime::tools::manage_automations`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_runtime::tools::manage_automations::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::CallWithFacts;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::tools::ManageMandatesTool;
    use crate::tools::ManageMemoriesTool;
    use crate::traits::StateStore;
    use crate::traits::ToolCallContext;
    use crate::types::ChannelVisibility;
    use crate::types::UserRole;
    use aidaemon_work::automations;
    use serde_json::json;
    use serde_json::Value;
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
        memory: Arc<dyn aidaemon_memory::stores::MemoryStores>,
        tool: ManageAutomationsTool,
        /// Scheduled automations by name, recorded when created.
        scheduled: std::sync::Mutex<std::collections::HashMap<String, String>>,
        /// Keeps the test database alive.
        _harness: crate::testing::TestHarness,
    }

    async fn fixture() -> Fixture {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state: Arc<dyn StateStore> = harness.state.clone();
        let memory = harness.state.memory_stores();
        let tool = ManageAutomationsTool::new(state.clone(), memory.clone());
        Fixture {
            state,
            memory,
            tool,
            scheduled: Default::default(),
            _harness: harness,
        }
    }

    impl Fixture {
        async fn reminder(&self, name: &str, when: &str) -> String {
            let output = crate::tools::SetReminderTool::new(self.state.clone(), self.memory())
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
            let output = ManageMemoriesTool::new(self.state.clone(), self.memory())
                .call_with(
                    &json!({
                        "action": "create_scheduled_goal",
                        "goal": "Summarize the synthetic inbox",
                        "schedule": schedule,
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

        fn memory(&self) -> Arc<dyn aidaemon_memory::stores::MemoryStores> {
            self.memory.clone()
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
        assert!(f
            .reminder("Oven", "every weekday at 8am")
            .await
            .starts_with("Reminder set"));
        let work = f.scheduled_work("Inbox digest", "every day at 9am").await;
        assert!(work.contains("Confirmed and activated"), "{work}");
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

        let set = crate::tools::SetReminderTool::new(f.state.clone(), f.memory())
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
}
