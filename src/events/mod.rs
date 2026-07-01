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

    // === Tool Lifecycle ===
    /// Tool execution started
    ToolCall,
    /// Tool execution completed
    ToolResult,

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
