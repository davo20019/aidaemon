//! The daemon's agent as the turn runtime's host.

use aidaemon_events::stores::DialogueStateStore;
use aidaemon_runtime::host::{DecisionRecorder, TurnStores, TurnWorkspace};

use super::Agent;
use crate::events::EventStore;
use crate::execution::ExecutionBackend;

impl TurnStores for Agent {
    fn dialogue_states(&self) -> &dyn DialogueStateStore {
        &*self.state
    }

    fn events(&self) -> &EventStore {
        &self.event_store
    }
}

impl TurnWorkspace for Agent {
    fn execution_backend(&self) -> &dyn ExecutionBackend {
        &*self.execution_backend
    }

    fn project_roots(&self) -> &[String] {
        &self.path_aliases.projects
    }
}

#[async_trait::async_trait]
impl DecisionRecorder for Agent {
    async fn record_decision_point(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: crate::events::DecisionType,
        summary: String,
        metadata: serde_json::Value,
    ) {
        self.emit_decision_point(
            emitter,
            task_id,
            iteration,
            decision_type,
            summary,
            metadata,
        )
        .await;
    }
}
