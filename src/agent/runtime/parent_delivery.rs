use super::Agent;
use crate::channels::ChannelHub;
use std::sync::Weak;
use tracing::warn;

/// Classifies the call site driving a parent-mediated delivery so the
/// recorded note can be labeled appropriately. The parent LLM will see the
/// resulting note in its session history; the label clarifies how the text
/// reached the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParentDeliveryKind {
    /// Result text returned by a task-lead specialist for a still-active goal.
    TaskLeadResult,
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
            ParentDeliveryKind::TaskLeadResult => "Parent-visible result (task lead)",
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

impl Agent {
    pub(crate) async fn deliver_parent_text_result(
        &self,
        hub: Option<&Weak<ChannelHub>>,
        parent_session_id: &str,
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

        // Persistence-only mode: caller passed no hub. Record so the parent
        // LLM can reason about the produced text (tests, headless flows).
        let Some(hub_weak) = hub else {
            self.record_parent_visible_result_note(parent_session_id, kind.note_prefix(), trimmed)
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
                self.record_parent_visible_result_note(
                    parent_session_id,
                    kind.note_prefix(),
                    trimmed,
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
