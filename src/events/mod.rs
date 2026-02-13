//! Event-sourced architecture for agent activity tracking.
//!
//! This module provides a pure event-sourcing system where all agent activity
//! is captured as immutable events. Events serve multiple purposes:
//! - **Working context**: Answer "what are you doing?" and "what was the error?"
//! - **Conversation history**: Replace the messages table
//! - **Learning input**: Feed the consolidation system for long-term memory
//! - **Audit trail**: Full debugging and reconstruction capability

mod consolidation;
mod context;
mod payloads;
mod store;

pub use consolidation::{Consolidator, Pruner};
pub use context::SessionContextCompiler;
pub use payloads::*;
#[allow(unused_imports)]
pub use store::{
    EventEmitter, EventStore, PolicyGraduationReport, SessionWriteDrift, TaskWindowStats,
    WriteConsistencyGateStatus, WriteConsistencyReport, WriteConsistencyThresholds,
};

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

        Self {
            id: 0, // Will be set by database
            session_id,
            event_type,
            data,
            created_at: Utc::now(),
            consolidated_at: None,
            task_id,
            tool_name,
        }
    }

    /// Parse the event data into a typed payload
    pub fn parse_data<T: for<'de> Deserialize<'de>>(&self) -> anyhow::Result<T> {
        Ok(serde_json::from_value(self.data.clone())?)
    }
}

/// Types of events that can be stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    // === Session Lifecycle ===
    /// A new session started
    SessionStart,
    /// Session ended (explicit or timeout)
    SessionEnd,

    // === Conversation (replaces messages table) ===
    /// User sent a message
    UserMessage,
    /// Assistant generated a response
    AssistantResponse,

    // === Tool Lifecycle ===
    /// Tool execution started
    ToolCall,
    /// Tool execution completed
    ToolResult,

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
        match self {
            EventType::SessionStart => "session_start",
            EventType::SessionEnd => "session_end",
            EventType::UserMessage => "user_message",
            EventType::AssistantResponse => "assistant_response",
            EventType::ToolCall => "tool_call",
            EventType::ToolResult => "tool_result",
            EventType::ThinkingStart => "thinking_start",
            EventType::PolicyDecision => "policy_decision",
            EventType::DecisionPoint => "decision_point",
            EventType::TaskStart => "task_start",
            EventType::TaskEnd => "task_end",
            EventType::Error => "error",
            EventType::SubAgentSpawn => "sub_agent_spawn",
            EventType::SubAgentComplete => "sub_agent_complete",
            EventType::ApprovalRequested => "approval_requested",
            EventType::ApprovalGranted => "approval_granted",
            EventType::ApprovalDenied => "approval_denied",
        }
    }

    /// Parse from string (database storage)
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "session_start" => Some(EventType::SessionStart),
            "session_end" => Some(EventType::SessionEnd),
            "user_message" => Some(EventType::UserMessage),
            "assistant_response" => Some(EventType::AssistantResponse),
            "tool_call" => Some(EventType::ToolCall),
            "tool_result" => Some(EventType::ToolResult),
            "thinking_start" => Some(EventType::ThinkingStart),
            "policy_decision" => Some(EventType::PolicyDecision),
            "decision_point" => Some(EventType::DecisionPoint),
            "task_start" => Some(EventType::TaskStart),
            "task_end" => Some(EventType::TaskEnd),
            "error" => Some(EventType::Error),
            "sub_agent_spawn" => Some(EventType::SubAgentSpawn),
            "sub_agent_complete" => Some(EventType::SubAgentComplete),
            "approval_requested" => Some(EventType::ApprovalRequested),
            "approval_granted" => Some(EventType::ApprovalGranted),
            "approval_denied" => Some(EventType::ApprovalDenied),
            _ => None,
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Completed,
    Cancelled,
    Failed,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Completed => "completed",
            TaskStatus::Cancelled => "cancelled",
            TaskStatus::Failed => "failed",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "completed" => Some(TaskStatus::Completed),
            "cancelled" => Some(TaskStatus::Cancelled),
            "failed" => Some(TaskStatus::Failed),
            _ => None,
        }
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
