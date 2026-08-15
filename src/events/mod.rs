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
mod payloads;
mod store;
pub mod terminal_state;

pub use consolidation::{Consolidator, Pruner};
pub use context::SessionContextCompiler;
#[allow(unused_imports)]
pub use conversation_turn::{turn_from_event, ConversationTurn, ConversationTurnRole, FetchedTurn};
pub use model_call_telemetry::{
    record_background_model_call_telemetry, record_model_call_telemetry, ModelCallTelemetryInput,
};
pub use payloads::*;
#[allow(unused_imports)]
pub use store::{
    EventEmitter, EventStore, LlmStats, PolicyGraduationReport, SessionWriteDrift, TaskLlmSummary,
    TaskWindowStats, ToolStats, WriteConsistencyGateStatus, WriteConsistencyReport,
    WriteConsistencyThresholds,
};
#[allow(unused_imports)]
pub use terminal_state::TerminalState;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Returns true when a session is an implementation-internal worker whose
/// transcript must not feed owner/global memory. Keep this as a legacy
/// defense: mandate execution also carries explicit durable provenance below.
pub(crate) fn is_memory_isolated_worker_session(session_id: &str) -> bool {
    session_id.starts_with("specialist:")
        || session_id.starts_with("sub-")
        || session_id.starts_with("background:")
}

/// Resolve durable mandate-execution provenance for a session.
///
/// New mandate workers stamp `execution_origin = "mandate"` on their canonical
/// opening user-message event. The task/run join is a second durable signal for
/// run-bound workers and protects sessions recorded during a partially upgraded
/// process. Both paths use existing indexed event/task identities; no session
/// naming convention or claimed speaker role grants the classification.
pub(crate) async fn session_has_mandate_execution_provenance(
    pool: &sqlx::SqlitePool,
    session_id: &str,
) -> anyhow::Result<bool> {
    // A malformed canonical payload cannot safely prove ordinary provenance.
    // Treat its session as isolated, and guard json_extract so legacy bad JSON
    // never aborts retention/consolidation.
    let explicit_or_unreadable: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS (
            SELECT 1
            FROM events e
            WHERE e.session_id = ?
              AND CASE
                    WHEN NOT json_valid(e.data) THEN 1
                    WHEN e.event_type = 'user_message'
                    THEN COALESCE(
                        json_extract(e.data, '$.execution_origin') = 'mandate',
                        0
                    )
                    ELSE 0
                  END
            LIMIT 1
        )
        "#,
    )
    .bind(session_id)
    .fetch_one(pool)
    .await?;
    if explicit_or_unreadable != 0 {
        return Ok(true);
    }

    // Some narrow EventStore-only fixtures intentionally omit the work model.
    // The explicit marker above remains authoritative there; only attempt the
    // compatibility task/run join when both durable tables exist.
    let work_tables: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master
         WHERE type = 'table' AND name IN ('tasks', 'goal_runs')",
    )
    .fetch_one(pool)
    .await?;
    if work_tables != 2 {
        return Ok(false);
    }

    let run_bound: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS (
            SELECT 1
            FROM events e
            JOIN tasks t ON t.id = e.task_id
            JOIN goal_runs gr ON gr.id = t.goal_run_id
            WHERE e.session_id = ?
              AND gr.trigger_type = 'mandate'
            LIMIT 1
        )
        "#,
    )
    .bind(session_id)
    .fetch_one(pool)
    .await?;
    Ok(run_bound != 0)
}

/// A single immutable event in the event store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: i64,
    pub session_id: String,
    pub event_type: EventType,
    pub data: JsonValue,
    pub created_at: DateTime<Utc>,
    /// When this event was processed by consolidation (None = not yet consolidated)
    pub consolidated_at: Option<DateTime<Utc>>,
    /// Optional task ID for indexing (extracted from data)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Optional tool name for indexing (extracted from data)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Optional turn ID for turn-anchored history (extracted from data).
    /// Globally-unique UUID = the opening user-message id of the turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

impl Event {
    /// Create a new event (id will be assigned by the database)
    pub fn new(session_id: impl Into<String>, event_type: EventType, data: JsonValue) -> Self {
        let session_id = session_id.into();

        // Extract task_id and tool_name from data for indexing
        let task_id = data
            .get("task_id")
            .and_then(|v| v.as_str())
            .map(String::from);
        let tool_name = data.get("name").and_then(|v| v.as_str()).map(String::from);
        let turn_id = data
            .get("turn_id")
            .and_then(|v| v.as_str())
            .map(String::from);

        Self {
            id: 0, // Will be set by database
            session_id,
            event_type,
            data,
            created_at: Utc::now(),
            consolidated_at: None,
            task_id,
            tool_name,
            turn_id,
        }
    }

    /// Parse the event data into a typed payload
    pub fn parse_data<T: for<'de> Deserialize<'de>>(&self) -> anyhow::Result<T> {
        Ok(serde_json::from_value(self.data.clone())?)
    }
}

/// Types of events that can be stored.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::IntoStaticStr,
    strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum EventType {
    // === Session Lifecycle ===
    /// A new session started
    SessionStart,
    /// Session ended (explicit or timeout)
    SessionEnd,

    // === Conversation ===
    /// User sent a message
    UserMessage,
    /// Assistant generated a response
    AssistantResponse,
    /// An assistant response advanced through a channel transport lifecycle.
    ResponseDelivery,

    // === Tool Lifecycle ===
    /// Tool execution started
    ToolCall,
    /// Tool execution completed
    ToolResult,

    // === Resource Lifecycle ===
    /// A stable resource handle was registered from an attachment or tool.
    ResourceRegistered,
    /// A previously registered resource is no longer safe/current to use.
    ResourceInvalidated,

    // === Human Interaction Lifecycle ===
    /// A run paused while waiting for a human decision.
    InteractionRequested,
    /// A pending human decision was resolved or became unavailable.
    InteractionResolved,

    // === Filesystem Checkpoints ===
    /// A pre-mutation workspace checkpoint was created.
    CheckpointCreated,
    /// A checkpoint's post-task tree was captured.
    CheckpointFinalized,
    /// Checkpointing was skipped or failed before a mutation.
    CheckpointSkipped,
    /// A confirmed rollback started.
    RollbackStarted,
    /// A rollback completed, possibly with preserved conflicts.
    RollbackCompleted,
    /// A rollback failed before completion.
    RollbackFailed,

    // === LLM Lifecycle ===
    /// A single LLM provider call completed (with latency + token telemetry)
    LlmCall,

    // === Agent Thinking ===
    /// Agent started a new thinking iteration
    ThinkingStart,
    /// Policy routing shadow/enforcement decision emitted at task start.
    PolicyDecision,
    /// Structured decision-point emission for self-diagnosis flight recorder.
    DecisionPoint,
    /// Cumulative policy counters checkpointed with a process boot identity.
    PolicyMetricsSnapshot,
    /// A current-request capability prohibition was attempted. The payload
    /// records whether enforcement prevented any side effect.
    UserConstraintViolation,

    // === Task Lifecycle ===
    /// A task (user request) started processing
    TaskStart,
    /// A task completed (success, failure, or cancellation)
    TaskEnd,

    // === Errors ===
    /// An error occurred during processing
    Error,

    // === Sub-Agents ===
    /// A sub-agent was spawned
    SubAgentSpawn,
    /// A sub-agent completed its work
    SubAgentComplete,

    // === Approvals ===
    /// Approval was requested from user
    ApprovalRequested,
    /// User responded to approval request
    ApprovalGranted,
    /// User denied approval request
    ApprovalDenied,
}

impl EventType {
    /// Returns the string representation for database storage
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }

    /// Parse from string (database storage)
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }

    /// Event types that represent conversation messages (for history retrieval)
    pub fn is_conversation_event(&self) -> bool {
        matches!(
            self,
            EventType::UserMessage | EventType::AssistantResponse | EventType::ToolResult
        )
    }

    /// Event types that should trigger consolidation learning
    pub fn is_learnable(&self) -> bool {
        matches!(
            self,
            EventType::TaskEnd | EventType::Error | EventType::ToolResult
        )
    }
}

/// Task completion status
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    strum::IntoStaticStr,
    strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TaskStatus {
    Completed,
    Cancelled,
    Failed,
    /// Execution stopped without reaching a semantic terminal result and may
    /// continue from durable state. This is not equivalent to failure.
    Interrupted,
}

/// Semantic outcome delivered by a task execution.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    strum::IntoStaticStr,
    strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TaskOutcome {
    Succeeded,
    Partial,
    Failed,
}

impl TaskOutcome {
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }

    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }

    /// Maps semantic outcome to learning success signal.
    pub fn task_success(&self) -> bool {
        matches!(self, TaskOutcome::Succeeded)
    }
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }

    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_roundtrip() {
        for event_type in [
            EventType::SessionStart,
            EventType::UserMessage,
            EventType::ToolCall,
            EventType::LlmCall,
            EventType::PolicyDecision,
            EventType::DecisionPoint,
            EventType::TaskEnd,
            EventType::Error,
        ] {
            let s = event_type.as_str();
            let parsed = EventType::from_str(s).expect("should parse");
            assert_eq!(event_type, parsed);
        }
    }

    #[test]
    fn test_event_creation() {
        let event = Event::new(
            "session_123",
            EventType::TaskStart,
            serde_json::json!({
                "task_id": "task_456",
                "description": "Test task"
            }),
        );

        assert_eq!(event.session_id, "session_123");
        assert_eq!(event.event_type, EventType::TaskStart);
        assert_eq!(event.task_id, Some("task_456".to_string()));
    }
}
