//! Auxiliary note recording: persisting assistant/parent-visible notes into
//! session history.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged.

use super::*;

// impl-Agent justification: cross-subsystem note API — called from channels/hub.rs and telegram.rs via Arc<Agent>.
impl Agent {
    pub(crate) async fn record_auxiliary_assistant_note(
        &self,
        session_id: &str,
        content: &str,
    ) -> anyhow::Result<()> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let emitter = crate::events::EventEmitter::new(self.event_store.clone(), session_id);
        let msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(trimmed.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.2,
            ..Message::runtime_defaults()
        };
        self.append_assistant_message_with_event(&emitter, &msg, "system", None, None)
            .await
    }

    pub(crate) async fn record_parent_visible_result_note(
        &self,
        session_id: &str,
        prefix: &str,
        text: &str,
    ) -> anyhow::Result<()> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let summary = format!("{}:\n\n{}", prefix, trimmed);
        self.record_auxiliary_assistant_note(session_id, &summary)
            .await
    }
}

#[cfg(test)]
mod tests {
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
        let weak_hub: Weak<ChannelHub> = {
            let session_map: SessionMap = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
            let hub = Arc::new(ChannelHub::new(Vec::new(), session_map));
            let weak = Arc::downgrade(&hub);
            drop(hub);
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
