//! Parent-visible deliveries live in `aidaemon-runtime`
//! ([`aidaemon_runtime::turn::notes`]).

#[cfg(test)]
pub(crate) use aidaemon_runtime::turn::notes::ParentDeliveryKind;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::TurnNotes;
    use crate::channels::ChannelHub;
    use crate::runtime_ports::OutboundRouter;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::tools::command_risk::{PermissionMode, RiskLevel};
    use crate::traits::{ChannelCapabilities, HubChannel};
    use crate::types::{ApprovalResponse, MediaMessage};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::{Mutex, RwLock};

    struct EditingChannel {
        sends: Mutex<Vec<String>>,
        edits: Mutex<Vec<(String, String)>>,
    }

    #[async_trait]
    impl HubChannel for EditingChannel {
        fn name(&self) -> String {
            "editing".to_string()
        }

        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities {
                markdown: true,
                inline_buttons: false,
                media: false,
                max_message_len: 4096,
            }
        }

        async fn send_text(&self, _session_id: &str, text: &str) -> anyhow::Result<()> {
            self.sends.lock().await.push(text.to_string());
            Ok(())
        }

        async fn edit_text(
            &self,
            _session_id: &str,
            message_id: &str,
            text: &str,
        ) -> anyhow::Result<bool> {
            self.edits
                .lock()
                .await
                .push((message_id.to_string(), text.to_string()));
            Ok(true)
        }

        async fn send_media(&self, _session_id: &str, _media: &MediaMessage) -> anyhow::Result<()> {
            Ok(())
        }

        async fn request_approval(
            &self,
            _session_id: &str,
            _command: &str,
            _risk_level: RiskLevel,
            _warnings: &[String],
            _permission_mode: PermissionMode,
            _one_time_only: bool,
        ) -> anyhow::Result<ApprovalResponse> {
            Ok(ApprovalResponse::Deny)
        }
    }

    #[tokio::test]
    async fn terminal_result_replaces_existing_progress_surface() {
        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup harness");
        let session_id = "telegram:test_bot:synthetic-owner";
        let channel = Arc::new(EditingChannel {
            sends: Mutex::new(Vec::new()),
            edits: Mutex::new(Vec::new()),
        });
        let session_map = Arc::new(RwLock::new(HashMap::from([(
            session_id.to_string(),
            "editing".to_string(),
        )])));
        let hub = Arc::new(ChannelHub::new(
            vec![channel.clone() as Arc<dyn HubChannel>],
            session_map,
        ));
        let outbound: Arc<dyn OutboundRouter> = hub;

        let outcome = harness
            .agent
            .deliver_parent_text_result_to_surface(
                Some(&Arc::downgrade(&outbound)),
                session_id,
                Some("42"),
                "✅ **Scheduled run complete**\n\nPublished.",
                ParentDeliveryKind::GoalNotification,
            )
            .await
            .expect("deliver terminal result");

        assert!(outcome.sent);
        assert!(outcome.recorded);
        assert!(channel.sends.lock().await.is_empty());
        assert_eq!(
            channel.edits.lock().await.as_slice(),
            &[(
                "42".to_string(),
                "✅ **Scheduled run complete**\n\nPublished.".to_string()
            )]
        );
    }

    /// The recorded note's turn id and the session's tainted turns after a
    /// background child's result is delivered with `trust`.
    async fn delivered_note_turn(
        trust: aidaemon_core::SourceTrust,
    ) -> (Option<String>, Vec<String>) {
        use crate::traits::store_prelude::*;
        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup harness");
        let session_id = "telegram:test_bot:synthetic-owner";
        let outcome = harness
            .agent
            .deliver_background_spawn_result(
                None,
                session_id,
                "Background task complete\nMission: Summarize\n\nA synthetic summary.",
                trust,
            )
            .await
            .expect("deliver background result");
        assert!(outcome.recorded);
        let history = harness
            .state
            .get_history(session_id, 10)
            .await
            .expect("history");
        let note = history
            .iter()
            .find(|message| {
                message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("A synthetic summary."))
            })
            .expect("the note is recorded");
        let tainted = crate::agent::untrusted::tainted_turns(harness.state.as_ref(), session_id)
            .await
            .expect("tainted turns");
        (note.turn_id.clone(), tainted)
    }

    /// A tainted background child's result enters the parent session as a
    /// turn recorded tainted, so the next turn that sees it starts tainted.
    #[tokio::test]
    async fn a_tainted_background_childs_result_is_a_tainted_turn() {
        let (turn, tainted) = delivered_note_turn(aidaemon_core::SourceTrust::Untrusted).await;
        let turn = turn.expect("the note carries its own turn");
        assert_eq!(tainted, vec![turn]);
    }

    #[tokio::test]
    async fn a_clean_background_childs_result_taints_nothing() {
        let (_, tainted) = delivered_note_turn(aidaemon_core::SourceTrust::Trusted).await;
        assert!(tainted.is_empty(), "{tainted:?}");
    }
}
