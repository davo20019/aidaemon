//! Event-sourced architecture for agent activity tracking.
//!
//! The store, its payloads and the run aggregate live in `aidaemon-events`;
//! consolidation into long-term memory and the event store's seams in
//! `aidaemon-runtime` (`consolidation`, `history`). This module keeps their
//! names stable for the daemon.

#[cfg(test)]
#[path = "consolidation_golden_tests.rs"]
mod consolidation_golden_tests;
#[cfg(test)]
mod consolidation_tests;
#[cfg(test)]
mod run_aggregate_checklist_tests;
#[cfg(test)]
mod store_host_tests;

pub use aidaemon_runtime::consolidation::{Consolidator, Pruner};

#[allow(unused_imports)]
pub use aidaemon_events::conversation_turn::{
    ConversationTurn, ConversationTurnRole, FetchedTurn, turn_from_event,
};
pub use aidaemon_events::payloads::*;
#[cfg(test)]
pub(crate) use aidaemon_events::run_aggregate::RunAggregate;
#[cfg(test)]
pub(crate) use aidaemon_events::run_aggregate::{RunObligationState, RunTerminalDecision};
#[allow(unused_imports)]
pub use aidaemon_events::store::{
    ContinuationToolEvidence, EventEmitter, EventHost, EventStore, LlmStats,
    PolicyGraduationReport, ProviderReadiness, SessionWriteDrift, TaskLlmSummary,
    TaskReceiptClosure, TaskWindowStats, ToolStats, WriteConsistencyGateStatus,
    WriteConsistencyReport, WriteConsistencyThresholds,
};
#[allow(unused_imports)]
pub use aidaemon_events::terminal_state::TerminalState;
#[allow(unused_imports)]
pub use aidaemon_events::{
    Event, EventType, TaskOutcome, TaskStatus, is_memory_isolated_worker_session,
    session_has_mandate_execution_provenance,
};

/// The daemon's end of the event store's seams.
pub(crate) use aidaemon_runtime::history::HistoryEventHost as DaemonEventHost;

/// `EventStore::new` for the daemon's tests: its production seams, on the
/// paths of the test process.
#[cfg(test)]
pub(crate) trait EventStoreForTests: Sized {
    async fn new(pool: sqlx::SqlitePool) -> anyhow::Result<Self>;
}

#[cfg(test)]
impl EventStoreForTests for EventStore {
    async fn new(pool: sqlx::SqlitePool) -> anyhow::Result<Self> {
        EventStore::on_paths(
            pool,
            aidaemon_tools::LexicalPaths::daemon_host(),
            std::sync::Arc::new(DaemonEventHost),
        )
        .await
    }
}
