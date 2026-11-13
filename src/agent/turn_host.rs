//! The daemon's agent as the turn runtime's host.

use aidaemon_events::stores::{DialogueStateStore, MessageStore};
use aidaemon_runtime::host::{DecisionRecorder, TurnHost, TurnStores, TurnWorkspace};
use aidaemon_runtime::turn_runtime::TurnRuntime;

use super::Agent;
use crate::events::EventStore;
use crate::execution::ExecutionBackend;

impl TurnStores for Agent {
    fn dialogue_states(&self) -> &dyn DialogueStateStore {
        &*self.state
    }

    fn messages(&self) -> &dyn MessageStore {
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

impl TurnHost for Agent {
    fn turn_runtime(&self) -> &TurnRuntime {
        &self.runtime
    }

    fn is_worker(&self) -> bool {
        self.depth > 0 || self.mandate_execution.is_some()
    }

    fn inherited_authority(&self) -> Option<&aidaemon_runtime::authority::InheritedAuthority> {
        self.inherited_authority.as_ref()
    }

    fn tools(&self) -> &[std::sync::Arc<dyn crate::traits::Tool>] {
        &self.tools
    }
}
