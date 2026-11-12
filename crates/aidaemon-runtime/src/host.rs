//! What the turn runtime asks of its host: a few cohesive traits the daemon's
//! agent implements, so the runtime's handlers run on any host that supplies
//! the same parts.
//!
//! - [`TurnStores`]: the conversation's durable state (dialogue state and the
//!   event log).
//! - [`TurnWorkspace`]: where the turn's file work happens (the execution
//!   backend and the configured project roots).

use aidaemon_events::stores::DialogueStateStore;
use aidaemon_events::EventStore;
use aidaemon_tools::ExecutionBackend;

/// The conversation's durable state: per-session dialogue state and the
/// event log the turn records into and rebuilds from.
pub trait TurnStores: Send + Sync {
    /// Per-session dialogue state (open request, open question, last turns).
    fn dialogue_states(&self) -> &dyn DialogueStateStore;
    /// The event log.
    fn events(&self) -> &EventStore;
}

/// Where the turn's file work happens.
pub trait TurnWorkspace: Send + Sync {
    /// The execution backend file and command work runs on.
    fn execution_backend(&self) -> &dyn ExecutionBackend;
    /// Configured project roots path references resolve against.
    fn project_roots(&self) -> &[String];
}
