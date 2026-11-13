//! Notes a turn's host keeps in a session's history: auxiliary assistant
//! notes, parent-visible results of children, goals and background work and
//! their delivery through the hub, and a mandate run's origin for the
//! untrusted-content rule ([`TurnNotes`], on any [`TurnHost`]).

use aidaemon_events::conversation::Message;
use chrono::Utc;
use uuid::Uuid;

use crate::host::TurnHost;
use crate::untrusted::MandateOrigin;

use aidaemon_channels::ports::OutboundRouter;
use std::sync::Weak;
use tracing::warn;

/// Classifies the call site driving a parent-mediated delivery so the
/// recorded note can be labeled appropriately. The parent LLM will see the
/// resulting note in its session history; the label clarifies how the text
/// reached the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParentDeliveryKind {
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
    pub fn note_prefix(self) -> &'static str {
        match self {
            ParentDeliveryKind::ExecutorResult => "Parent-visible result (executor)",
            ParentDeliveryKind::GoalNotification => "Parent-visible result (goal notification)",
            ParentDeliveryKind::WaitResult => "Parent-visible result (wait)",
            ParentDeliveryKind::BackgroundSpawnResult => "Parent-visible result (background spawn)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParentDeliveryOutcome {
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

/// Notes and parent-visible deliveries, for any [`TurnHost`].
#[async_trait::async_trait]
pub trait TurnNotes: TurnHost + crate::turn::turn_context::TurnHistory {
    /// Bind the next related owner turn to a delivered, content-free mandate
    /// ASK notice. The actual generated question remains in mandate-local
    /// storage and must be inspected with the management tool.
    async fn record_mandate_owner_input_context(
        &self,
        session_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        notification_id: &str,
    ) -> anyhow::Result<()> {
        crate::turn::dialogue_state::record_mandate_owner_input(
            self,
            session_id,
            mandate_id,
            mandate_version,
            notification_id,
        )
        .await
    }

    async fn record_auxiliary_assistant_note(
        &self,
        session_id: &str,
        content: &str,
    ) -> anyhow::Result<()> {
        self.record_auxiliary_assistant_note_in_turn(session_id, content, None)
            .await
    }

    /// `turn_id` stamps the note; `None` joins the session's current turn, or
    /// opens a turn of its own when none is running. History is fetched by
    /// turn, so a note with no turn would never reach the model: a goal
    /// notification delivered overnight must still be what "tell me more"
    /// refers to the next morning.
    async fn record_auxiliary_assistant_note_in_turn(
        &self,
        session_id: &str,
        content: &str,
        turn_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let turn_id = match turn_id {
            Some(turn_id) => turn_id.to_string(),
            None => self
                .turn_runtime()
                .current_turn_ids
                .read()
                .await
                .get(session_id)
                .cloned()
                .unwrap_or_else(|| Uuid::new_v4().to_string()),
        };

        let emitter = aidaemon_events::EventEmitter::new(self.event_log(), session_id);
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
            turn_id: Some(turn_id),
            ..Message::runtime_defaults()
        };
        self.append_assistant_message_with_event(&emitter, &msg, "system", None, None)
            .await
    }

    async fn record_parent_visible_result_note(
        &self,
        session_id: &str,
        prefix: &str,
        text: &str,
    ) -> anyhow::Result<()> {
        self.record_parent_visible_result_note_in_turn(session_id, prefix, text, None)
            .await
    }

    async fn record_parent_visible_result_note_in_turn(
        &self,
        session_id: &str,
        prefix: &str,
        text: &str,
        turn_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let summary = format!("{}:\n\n{}", prefix, trimmed);
        self.record_auxiliary_assistant_note_in_turn(session_id, &summary, turn_id)
            .await
    }

    /// Deliver parent-visible text by replacing an existing progress surface
    /// when possible. This keeps a background run's lifecycle in one message:
    /// queued/running/progress becomes the terminal result instead of leaving a
    /// stale status bubble above a second completion notification.
    async fn deliver_parent_text_result_to_surface(
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
    async fn deliver_background_spawn_result(
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
    async fn tainted_note_turn(&self, session_id: &str) -> anyhow::Result<String> {
        let turn_id = self
            .turn_runtime()
            .current_turn_ids
            .read()
            .await
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        crate::untrusted::record_tainted_turn(self.settings(), session_id, &turn_id).await?;
        Ok(turn_id)
    }

    async fn deliver_parent_text_result(
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

    /// The confirmed mandate as this run's owner request; `None` outside a
    /// mandate run. A run whose fence no longer matches an owner-confirmed
    /// policy has no owner request at all, so its egress is refused.
    async fn mandate_origin(&self) -> Option<MandateOrigin> {
        let fence = self.mandate_execution()?;
        let mandate = match self.mandates().get_mandate(&fence.mandate_id).await {
            Ok(mandate) => mandate,
            Err(error) => {
                warn!(mandate_id = %fence.mandate_id, %error, "Mandate unreadable for the egress check");
                None
            }
        };
        Some(
            match mandate.filter(|mandate| {
                mandate.version == fence.mandate_version
                    && mandate.authority == fence.authority
                    && mandate.confirmed_at.is_some()
            }) {
                Some(mandate) => MandateOrigin {
                    request: aidaemon_work::mandates::policy::verifier_request(&mandate),
                    trusted: true,
                },
                None => MandateOrigin {
                    request: format!("Stale or unconfirmed mandate {}.", fence.mandate_id),
                    trusted: false,
                },
            },
        )
    }
}

impl<T: TurnHost + ?Sized> TurnNotes for T {}
