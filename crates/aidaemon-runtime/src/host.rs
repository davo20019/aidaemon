//! What the turn runtime asks of its host: a few cohesive traits the daemon's
//! agent implements, so the runtime's handlers run on any host that supplies
//! the same parts.
//!
//! - [`TurnStores`]: the conversation's durable state (dialogue state and the
//!   event log).
//! - [`TurnWorkspace`]: where the turn's file work happens (the execution
//!   backend and the configured project roots).
//! - [`DecisionRecorder`]: where the turn's decision points go.
//! - [`TurnHost`]: all of the above plus the agent's turn runtime and who
//!   the agent is; the turn's handlers in this crate run on it.

use aidaemon_events::stores::{DialogueStateStore, MessageStore};
use aidaemon_events::EventStore;
use aidaemon_tools::ExecutionBackend;

/// The conversation's durable state: per-session dialogue state and the
/// event log the turn records into and rebuilds from.
pub trait TurnStores: Send + Sync {
    /// Per-session dialogue state (open request, open question, last turns).
    fn dialogue_states(&self) -> &dyn DialogueStateStore;
    /// The conversation's messages and its durable context boundary.
    fn messages(&self) -> &dyn MessageStore;
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

/// Where a turn's decision points go: the record of why the runtime chose
/// what it did, kept with the turn's events when the host records them.
#[async_trait::async_trait]
pub trait DecisionRecorder: Send + Sync {
    /// Record one decision point of `task_id` at loop `iteration`.
    async fn record_decision_point(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: aidaemon_events::DecisionType,
        summary: String,
        metadata: serde_json::Value,
    );
}

/// The host a turn runs on: its stores, workspace and decision record, the
/// agent's [`TurnRuntime`] and where the agent stands in its tree.
pub trait TurnHost: TurnStores + TurnWorkspace + DecisionRecorder {
    /// The agent's turn settings and caches.
    fn turn_runtime(&self) -> &crate::turn_runtime::TurnRuntime;
    /// Whether this agent works for another (a spawned child, or a mandate's
    /// worker): its transcript is an authority envelope, not a conversation
    /// with the user.
    fn is_worker(&self) -> bool;
    /// What a spawned child inherited from the turn that spawned it; `None`
    /// for a root agent.
    fn inherited_authority(&self) -> Option<&crate::authority::InheritedAuthority>;
    /// The agent's tools (their task-end hooks run when a task ends).
    fn tools(&self) -> &[std::sync::Arc<dyn aidaemon_tools::OutcomeTool>];
}
