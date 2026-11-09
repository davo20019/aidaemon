//! Event-sourced architecture for agent activity tracking.
//!
//! This module provides a pure event-sourcing system where all agent activity
//! is captured as immutable events. Events serve multiple purposes:
//! - **Working context**: Answer "what are you doing?" and "what was the error?"
//! - **Conversation history**: Canonical chat/task timeline
//! - **Learning input**: Feed the consolidation system for long-term memory
//! - **Audit trail**: Full debugging and reconstruction capability

mod consolidation;
mod context;
mod conversation_turn;
mod model_call_telemetry;
mod store;
#[cfg(test)]
mod run_aggregate_checklist_tests;
pub mod terminal_state;

pub use consolidation::{Consolidator, Pruner};
pub use context::SessionContextCompiler;
#[allow(unused_imports)]
pub use conversation_turn::{turn_from_event, ConversationTurn, ConversationTurnRole, FetchedTurn};
pub use model_call_telemetry::{
    record_background_model_call_failure, record_background_model_call_telemetry,
    record_model_call_telemetry, ModelCallTelemetryInput,
};
pub use aidaemon_events::payloads::*;
pub use aidaemon_events::{
    is_memory_isolated_worker_session, session_has_mandate_execution_provenance, Event, EventType,
    TaskOutcome, TaskStatus,
};
pub(crate) use aidaemon_events::run_aggregate::{
    CloseoutDecision, RunAggregate, RunObligation, RunObligationClass, RunObligationState,
    RunTerminalDecision, TaskKernelAdmission, OWNER_APPROVAL_DECLINED,
};
#[cfg(test)]
pub(crate) use store::GeneratedResponseRef;
pub(crate) use store::{capture_generated_responses, CapturedGeneratedResponse};
#[allow(unused_imports)]
pub use store::{
    ContinuationToolEvidence, EventEmitter, EventStore, LlmStats, PolicyGraduationReport,
    ProviderReadiness, SessionWriteDrift, TaskLlmSummary, TaskReceiptClosure, TaskWindowStats,
    ToolStats, WriteConsistencyGateStatus, WriteConsistencyReport, WriteConsistencyThresholds,
};
#[allow(unused_imports)]
pub use terminal_state::TerminalState;
