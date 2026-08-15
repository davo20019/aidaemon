//! Auxiliary note recording: persisting assistant/parent-visible notes into
//! session history.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged.

use super::*;

// impl-Agent justification: this is the concrete AssistantNoteSink port used by
// channel delivery without exposing the rest of Agent.
impl Agent {
    /// Bind the next related owner turn to a delivered, content-free mandate
    /// ASK notice. The actual generated question remains in mandate-local
    /// storage and must be inspected with the management tool.
    pub(crate) async fn record_mandate_owner_input_context(
        &self,
        session_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        notification_id: &str,
    ) -> anyhow::Result<()> {
        super::dialogue_state::record_mandate_owner_input(
            self,
            session_id,
            mandate_id,
            mandate_version,
            notification_id,
        )
        .await
    }

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
