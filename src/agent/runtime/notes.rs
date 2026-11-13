//! Session notes live in `aidaemon-runtime` ([`aidaemon_runtime::turn::notes`]).

#[cfg(test)]
mod tests {
    use crate::agent::TurnNotes;

    #[tokio::test]
    async fn delivered_mandate_ask_persists_typed_owner_input_context() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::{DialogueStateStore, QuestionKind};

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup harness");
        let session_id = "telegram:test_bot:synthetic-owner";
        let mandate_id = "08012d3d-synthetic";

        harness
            .agent
            .record_mandate_owner_input_context(
                session_id,
                mandate_id,
                2,
                "mandate-run-notice:review-synthetic",
            )
            .await
            .expect("record mandate owner input");

        let state = harness
            .state
            .get_dialogue_state(session_id)
            .await
            .expect("load dialogue state")
            .expect("dialogue state persisted");
        let question = state
            .open_question
            .expect("mandate ASK should remain an open owner question");
        assert_eq!(question.kind, QuestionKind::MandateInput);
        assert_eq!(question.mandate_id.as_deref(), Some(mandate_id));
        assert!(question.awaiting_user_reply);
        assert!(!question.text.contains("generated question"));

        harness
            .agent
            .handle_message(
                session_id,
                "Walk me through the reason.",
                None,
                crate::types::UserRole::Owner,
                crate::types::ChannelContext::private("telegram"),
                None,
            )
            .await
            .expect("handle bound owner follow-up");

        let calls = harness.provider.call_log.lock().await;
        let prompt = serde_json::to_string(&calls[0].messages).expect("serialize prompt");
        assert!(prompt.contains("structurally bound to unresolved mandate"));
        assert!(prompt.contains("manage_mandates"));
        assert!(prompt.contains(mandate_id));
    }

    #[tokio::test]
    async fn parent_text_result_delivery_records_parent_visible_text() {
        use super::super::parent_delivery::ParentDeliveryKind;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup harness");
        let session_id = "telegram:test_bot:301753035";
        let delivered = "Goal completed:\n\nCreated ~/morning_ai_job_preparation_tips_report.md";

        let outcome = harness
            .agent
            .deliver_parent_text_result(
                None,
                session_id,
                delivered,
                ParentDeliveryKind::GoalNotification,
            )
            .await
            .expect("parent delivery result");

        assert!(outcome.recorded);
        assert!(!outcome.sent);

        let history = harness
            .state
            .get_history(session_id, 10)
            .await
            .expect("load history");

        assert!(
            history.iter().any(|msg| {
                msg.role == "assistant"
                    && msg
                        .content
                        .as_deref()
                        .is_some_and(|content| content.contains(delivered))
            }),
            "parent delivery should preserve visible text in parent history: {:?}",
            history
        );
    }
    /// A notification delivered while no turn is running (a scheduled goal at
    /// night) must still be in the conversation the owner's next message
    /// sees. The history fetch is turn-anchored, so a note stamped with no
    /// turn would never reach the model, and "tell me more" would be answered
    /// from whatever came before it.
    #[tokio::test]
    async fn idle_goal_notification_reaches_the_next_turn() {
        use super::super::parent_delivery::ParentDeliveryKind;
        use crate::testing::MockProvider;

        // Top-level chat agent (depth 0): the default harness is a delegated
        // worker, whose transcript keeps no earlier dialogue at all.
        let harness = crate::testing::setup_test_agent_with_models(
            MockProvider::new(),
            "mock-model",
            "mock-model",
        )
        .await
        .expect("setup harness");
        let session_id = "telegram:test_bot:synthetic-owner-2";

        harness
            .agent
            .handle_message(
                session_id,
                "Count the routes in the demo module.",
                None,
                crate::types::UserRole::Owner,
                crate::types::ChannelContext::private("telegram"),
                None,
            )
            .await
            .expect("earlier unrelated turn");
        assert!(
            harness
                .agent
                .runtime
                .current_turn_ids
                .read()
                .await
                .get(session_id)
                .is_none(),
            "no turn may be running when the notification lands"
        );

        let notice = "Your automation \"Grow the synthetic newsletter\": a review failed \
                      before it finished (usage limit reached).";
        let outcome = harness
            .agent
            .deliver_parent_text_result(
                None,
                session_id,
                notice,
                ParentDeliveryKind::GoalNotification,
            )
            .await
            .expect("record notification");
        assert!(outcome.recorded);

        let before = harness.provider.call_log.lock().await.len();
        harness
            .agent
            .handle_message(
                session_id,
                "What happened there?",
                None,
                crate::types::UserRole::Owner,
                crate::types::ChannelContext::private("telegram"),
                None,
            )
            .await
            .expect("follow-up turn");

        let calls = harness.provider.call_log.lock().await;
        let prompt = serde_json::to_string(&calls[before].messages).expect("serialize prompt");
        assert!(
            prompt.contains("Grow the synthetic newsletter"),
            "the idle notification must be in the next turn's conversation: {prompt}"
        );
    }

    #[tokio::test]
    async fn parent_text_result_does_not_record_on_dropped_hub() {
        use super::super::parent_delivery::ParentDeliveryKind;
        use crate::channels::{ChannelHub, SessionMap};
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;
        use std::collections::HashMap;
        use std::sync::{Arc, Weak};

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup harness");
        let session_id = "telegram:test_bot:301753036";
        let delivered = "Result that should not be recorded when hub is dropped";

        // Build a Weak that cannot upgrade by dropping the strong Arc first.
        let weak_hub: Weak<dyn crate::runtime_ports::OutboundRouter> = {
            let session_map: SessionMap = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
            let hub = Arc::new(ChannelHub::new(Vec::new(), session_map));
            let outbound: Arc<dyn crate::runtime_ports::OutboundRouter> = hub;
            let weak = Arc::downgrade(&outbound);
            drop(outbound);
            weak
        };
        assert!(weak_hub.upgrade().is_none(), "weak hub must be dead");

        let outcome = harness
            .agent
            .deliver_parent_text_result(
                Some(&weak_hub),
                session_id,
                delivered,
                ParentDeliveryKind::ExecutorResult,
            )
            .await
            .expect("parent delivery result");

        assert!(!outcome.sent);
        assert!(
            !outcome.recorded,
            "must not record when hub upgrade fails — queue retry owns delivery"
        );

        let history = harness
            .state
            .get_history(session_id, 10)
            .await
            .expect("load history");

        assert!(
            !history.iter().any(|msg| {
                msg.role == "assistant"
                    && msg
                        .content
                        .as_deref()
                        .is_some_and(|content| content.contains(delivered))
            }),
            "dropped-hub delivery must not leave a parent-visible note: {:?}",
            history
        );
    }
}
