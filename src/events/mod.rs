//! Event-sourced architecture for agent activity tracking.
//!
//! The store, its payloads and the run aggregate live in `aidaemon-events`.
//! This module keeps their names stable for the daemon and holds what is
//! the daemon's own: consolidation into long-term memory, and the event
//! store's seams.

mod consolidation;
#[cfg(test)]
mod run_aggregate_checklist_tests;
#[cfg(test)]
mod store_host_tests;

pub use consolidation::{Consolidator, Pruner};

pub use aidaemon_events::context::SessionContextCompiler;
#[allow(unused_imports)]
pub use aidaemon_events::conversation_turn::{
    turn_from_event, ConversationTurn, ConversationTurnRole, FetchedTurn,
};
pub use aidaemon_events::model_call_telemetry::{
    record_model_call_telemetry, ModelCallTelemetryInput,
};
pub use aidaemon_events::payloads::*;
#[cfg(test)]
pub(crate) use aidaemon_events::run_aggregate::RunAggregate;
pub(crate) use aidaemon_events::run_aggregate::{
    CloseoutDecision, RunObligationClass, RunObligationState, RunTerminalDecision,
    OWNER_APPROVAL_DECLINED,
};
#[cfg(test)]
pub(crate) use aidaemon_events::store::GeneratedResponseRef;
pub(crate) use aidaemon_events::store::{capture_generated_responses, CapturedGeneratedResponse};
#[allow(unused_imports)]
pub use aidaemon_events::store::{
    ContinuationToolEvidence, EventEmitter, EventHost, EventStore, LlmStats,
    PolicyGraduationReport, ProviderReadiness, SessionWriteDrift, TaskLlmSummary,
    TaskReceiptClosure, TaskWindowStats, ToolStats, WriteConsistencyGateStatus,
    WriteConsistencyReport, WriteConsistencyThresholds,
};
#[allow(unused_imports)]
pub use aidaemon_events::terminal_state::TerminalState;
pub use aidaemon_events::{
    is_memory_isolated_worker_session, session_has_mandate_execution_provenance, Event, EventType,
    TaskOutcome, TaskStatus,
};

/// The daemon's end of the event store's seams.
pub(crate) struct DaemonEventHost;

#[async_trait::async_trait]
impl EventHost for DaemonEventHost {
    async fn project_history(
        &self,
        pool: &sqlx::SqlitePool,
        event_id: i64,
    ) -> anyhow::Result<bool> {
        crate::state::sqlite::history_search::project_event(pool, event_id).await
    }

    async fn project_memory_span(
        &self,
        pool: &sqlx::SqlitePool,
        event_id: i64,
    ) -> anyhow::Result<()> {
        crate::memory::project_event_span(pool, event_id).await
    }

    fn error_pattern(&self, error: &str) -> String {
        crate::memory::procedures::extract_error_pattern(error)
    }
}

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
