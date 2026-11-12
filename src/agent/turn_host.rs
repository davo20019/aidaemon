//! The daemon's agent as the turn runtime's host.

use aidaemon_events::stores::DialogueStateStore;
use aidaemon_runtime::host::{TurnStores, TurnWorkspace};

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
