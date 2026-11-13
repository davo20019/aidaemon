use super::Agent;
use crate::runtime_ports::OutboundRouter;
use std::sync::Weak;
use tracing::warn;

/// Classifies the call site driving a parent-mediated delivery so the
/// recorded note can be labeled appropriately. The parent LLM will see the
/// resulting note in its session history; the label clarifies how the text
/// reached the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParentDeliveryKind {
    /// Result text returned by an executor specialist for a single task.
    ExecutorResult,
    /// Terminal goal notification (completion or failure summary).
    GoalNotification,
    /// Reply for a pure-wait short-circuit ("Waited for N seconds.").
    WaitResult,
    /// Completion/failure message from a backgrounded `spawn_agent`.
    BackgroundSpawnResult,
}

impl ParentDeliveryKind {
    fn note_prefix(self) -> &'static str {
        match self {
            ParentDeliveryKind::ExecutorResult => "Parent-visible result (executor)",
            ParentDeliveryKind::GoalNotification => "Parent-visible result (goal notification)",
            ParentDeliveryKind::WaitResult => "Parent-visible result (wait)",
            ParentDeliveryKind::BackgroundSpawnResult => "Parent-visible result (background spawn)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParentDeliveryOutcome {
    /// True when the text was actually emitted to the user via the channel hub.
    pub sent: bool,
    /// True when the text was recorded in parent session history. We only
    /// record on a delivery success path so the parent LLM never sees a note
    /// claiming the user saw something they did not. The persistence-only
    /// mode (hub = None) also records, since the caller explicitly opted out
    /// of live delivery.
    pub recorded: bool,
}

// impl-Agent justification: parent-result delivery via hub — called cross-subsystem on task completion.
impl Agent {
    /// Deliver parent-visible text by replacing an existing progress surface
    /// when possible. This keeps a background run's lifecycle in one message:
    /// queued/running/progress becomes the terminal result instead of leaving a
    /// stale status bubble above a second completion notification.
    pub(crate) async fn deliver_parent_text_result_to_surface(
        &self,
        hub: Option<&Weak<dyn OutboundRouter>>,
        parent_session_id: &str,
        surface_id: Option<&str>,
        text: &str,
        kind: ParentDeliveryKind,
    ) -> anyhow::Result<ParentDeliveryOutcome> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(ParentDeliveryOutcome {
                sent: false,
                recorded: false,
            });
        }

        if let (Some(hub_weak), Some(surface_id)) = (hub, surface_id) {
            if let Some(hub_arc) = hub_weak.upgrade() {
                match hub_arc
                    .edit_text(parent_session_id, surface_id, trimmed)
                    .await
                {
                    Ok(true) => {
                        self.record_parent_visible_result_note(
                            parent_session_id,
                            kind.note_prefix(),
                            trimmed,
                        )
                        .await?;
                        return Ok(ParentDeliveryOutcome {
                            sent: true,
                            recorded: true,
                        });
                    }
                    Ok(false) => {}
                    Err(error) => warn!(
                        session_id = %parent_session_id,
                        message_id = %surface_id,
                        %error,
                        "Failed to replace parent progress surface; sending a new message"
                    ),
                }
            }
        }

        self.deliver_parent_text_result(hub, parent_session_id, trimmed, kind)
            .await
    }

    /// Deliver a backgrounded child's result. A result from a child that
    /// read untrusted content lands in a turn recorded tainted before the
    /// note exists (`tainted_note_turn`), so every later turn that still
    /// sees it starts tainted.
    pub(crate) async fn deliver_background_spawn_result(
        &self,
        hub: Option<&Weak<dyn OutboundRouter>>,
        parent_session_id: &str,
        text: &str,
        trust: aidaemon_core::SourceTrust,
    ) -> anyhow::Result<ParentDeliveryOutcome> {
        let turn_id = match trust {
            aidaemon_core::SourceTrust::Trusted => None,
            aidaemon_core::SourceTrust::Untrusted => {
                Some(self.tainted_note_turn(parent_session_id).await?)
            }
        };
        self.deliver_parent_text_result_in_turn(
            hub,
            parent_session_id,
            text,
            ParentDeliveryKind::BackgroundSpawnResult,
            turn_id.as_deref(),
        )
        .await
    }

    /// The turn an untrusted note joins, recorded tainted before the note
    /// is written: the session's running turn if there is one, else a turn
    /// of its own. Fails if the taint cannot be recorded; callers then do
    /// not write the note.
    pub(crate) async fn tainted_note_turn(&self, session_id: &str) -> anyhow::Result<String> {
        let turn_id = self
            .runtime
            .current_turn_ids
            .read()
            .await
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        crate::agent::untrusted::record_tainted_turn(self.state.as_ref(), session_id, &turn_id)
            .await?;
        Ok(turn_id)
    }

    pub(crate) async fn deliver_parent_text_result(
        &self,
        hub: Option<&Weak<dyn OutboundRouter>>,
        parent_session_id: &str,
        text: &str,
        kind: ParentDeliveryKind,
    ) -> anyhow::Result<ParentDeliveryOutcome> {
        self.deliver_parent_text_result_in_turn(hub, parent_session_id, text, kind, None)
            .await
    }

    /// `turn_id` stamps the recorded note; `None` joins the session's
    /// current turn, if any.
    async fn deliver_parent_text_result_in_turn(
        &self,
        hub: Option<&Weak<dyn OutboundRouter>>,
        parent_session_id: &str,
        text: &str,
        kind: ParentDeliveryKind,
        turn_id: Option<&str>,
    ) -> anyhow::Result<ParentDeliveryOutcome> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(ParentDeliveryOutcome {
                sent: false,
                recorded: false,
            });
        }

        // Persistence-only mode: caller passed no hub. Record so the parent
        // LLM can reason about the produced text (tests, headless flows).
        let Some(hub_weak) = hub else {
            self.record_parent_visible_result_note_in_turn(
                parent_session_id,
                kind.note_prefix(),
                trimmed,
                turn_id,
            )
            .await?;
            return Ok(ParentDeliveryOutcome {
                sent: false,
                recorded: true,
            });
        };

        // Hub dropped between scheduling and delivery: skip recording. The
        // outbound queue / heartbeat is responsible for retry; recording a
        // "parent-visible result" note here would lie to the parent LLM.
        let Some(hub_arc) = hub_weak.upgrade() else {
            return Ok(ParentDeliveryOutcome {
                sent: false,
                recorded: false,
            });
        };

        match hub_arc.send_text(parent_session_id, trimmed).await {
            Ok(()) => {
                self.record_parent_visible_result_note_in_turn(
                    parent_session_id,
                    kind.note_prefix(),
                    trimmed,
                    turn_id,
                )
                .await?;
                Ok(ParentDeliveryOutcome {
                    sent: true,
                    recorded: true,
                })
            }
            Err(err) => {
                warn!(
                    session_id = %parent_session_id,
                    error = %err,
                    "Failed to send parent-mediated result"
                );
                Ok(ParentDeliveryOutcome {
                    sent: false,
                    recorded: false,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::ChannelHub;
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
