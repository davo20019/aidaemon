use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::event::Event;
use crate::state::{RunId, RunState};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("run store error: {0}")]
pub struct StoreError(pub String);

/// Durable home for runs: an append-only event journal plus the latest state
/// snapshot. The loop saves state after every turn and at every exit, so a
/// crash loses at most the turn in flight.
///
/// This is the whole persistence contract the kernel needs. Domain data
/// (memory, goals, people) belongs to the plugin crates that own it.
#[async_trait]
pub trait RunStore: Send + Sync {
    async fn append_event(&self, run_id: RunId, event: &Event) -> Result<(), StoreError>;
    async fn save_state(&self, state: &RunState) -> Result<(), StoreError>;
    async fn load_state(&self, run_id: RunId) -> Result<Option<RunState>, StoreError>;
}

/// In-process store for tests and short-lived agents.
#[derive(Default)]
pub struct MemoryStore {
    events: Mutex<HashMap<RunId, Vec<Event>>>,
    states: Mutex<HashMap<RunId, RunState>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self, run_id: RunId) -> Vec<Event> {
        self.events
            .lock()
            .expect("memory store poisoned")
            .get(&run_id)
            .cloned()
            .unwrap_or_default()
    }
}

#[async_trait]
impl RunStore for MemoryStore {
    async fn append_event(&self, run_id: RunId, event: &Event) -> Result<(), StoreError> {
        self.events
            .lock()
            .map_err(|e| StoreError(e.to_string()))?
            .entry(run_id)
            .or_default()
            .push(event.clone());
        Ok(())
    }

    async fn save_state(&self, state: &RunState) -> Result<(), StoreError> {
        self.states
            .lock()
            .map_err(|e| StoreError(e.to_string()))?
            .insert(state.run_id, state.clone());
        Ok(())
    }

    async fn load_state(&self, run_id: RunId) -> Result<Option<RunState>, StoreError> {
        Ok(self
            .states
            .lock()
            .map_err(|e| StoreError(e.to_string()))?
            .get(&run_id)
            .cloned())
    }
}
