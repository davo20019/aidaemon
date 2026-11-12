//! The event store as the checkpoint manager's audit log.

use aidaemon_tools::checkpoints::{CheckpointAudit, CheckpointEvent};
use serde_json::Value;

use crate::store::EventStore;
use crate::{Event, EventType};

/// The event recorded for a checkpoint lifecycle step.
pub fn checkpoint_event_type(event: CheckpointEvent) -> EventType {
    match event {
        CheckpointEvent::Created => EventType::CheckpointCreated,
        CheckpointEvent::Finalized => EventType::CheckpointFinalized,
        CheckpointEvent::Skipped => EventType::CheckpointSkipped,
        CheckpointEvent::RollbackStarted => EventType::RollbackStarted,
        CheckpointEvent::RollbackCompleted => EventType::RollbackCompleted,
        CheckpointEvent::RollbackFailed => EventType::RollbackFailed,
    }
}

#[async_trait::async_trait]
impl CheckpointAudit for EventStore {
    async fn record(
        &self,
        session_id: &str,
        event: CheckpointEvent,
        data: Value,
    ) -> anyhow::Result<()> {
        self.append(Event::new(session_id, checkpoint_event_type(event), data))
            .await
            .map(|_| ())
    }
}
