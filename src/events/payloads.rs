//! Event payload data structures.
//!
//! Each event type has a corresponding payload struct that contains
//! the event-specific data serialized as JSON.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use super::TaskStatus;

// =============================================================================
// Session Events
// =============================================================================

/// Data for SessionStart event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStartData {
    /// Channel name (e.g., "telegram", "discord")
    pub channel: String,
    /// Platform-specific user identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Data for SessionEnd event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEndData {
    /// Reason for session end
    pub reason: SessionEndReason,
    /// Total duration in seconds
    pub duration_secs: u64,
    /// Number of events in this session
    pub event_count: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEndReason {
    /// User explicitly ended the session
    UserEnded,
    /// Session timed out due to inactivity
    Timeout,
    /// Process is shutting down
    Shutdown,
    /// Error caused session to end
    Error,
}

// =============================================================================
// Conversation Events (replaces messages table)
// =============================================================================

/// Data for UserMessage event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessageData {
    /// The message content
    pub content: String,
    /// Platform-specific message ID (for reference)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Whether this message has attachments
    #[serde(default)]
    pub has_attachments: bool,
}

/// Data for AssistantResponse event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantResponseData {
    /// The response text content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls included in this response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
    /// Model used for this response
    pub model: String,
    /// Input tokens used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    /// Output tokens used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
}

/// Tool call information (subset of ToolCall for storage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    /// Tool call ID from the provider
    pub id: String,
    /// Tool name
    pub name: String,
    /// Arguments as JSON value (not string, for better querying)
    pub arguments: JsonValue,
}

// =============================================================================
// Tool Events
// =============================================================================

/// Data for ToolCall event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallData {
    /// Tool call ID (for matching with result)
    pub tool_call_id: String,
    /// Tool name
    pub name: String,
    /// Arguments passed to the tool
    pub arguments: JsonValue,
    /// Brief summary for display (e.g., "ls -la /home")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Associated task ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

/// Data for ToolResult event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultData {
    /// Tool call ID (matches ToolCall event)
    pub tool_call_id: String,
    /// Tool name
    pub name: String,
    /// Result content (may be truncated for large outputs)
    pub result: String,
    /// Whether the tool succeeded
    pub success: bool,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Associated task ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

// =============================================================================
// Agent Thinking Events
// =============================================================================

/// Data for ThinkingStart event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingStartData {
    /// Current iteration number (1-based)
    pub iteration: u32,
    /// Associated task ID
    pub task_id: String,
    /// Total tool calls so far in this task
    #[serde(default)]
    pub total_tool_calls: u32,
}

// =============================================================================
// Task Events
// =============================================================================

/// Data for TaskStart event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStartData {
    /// Unique task ID
    pub task_id: String,
    /// Brief description of the task (from user message)
    pub description: String,
    /// Parent task ID if this is a sub-task
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<String>,
    /// The full user message that triggered this task
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_message: Option<String>,
}

/// Data for TaskEnd event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEndData {
    /// Task ID (matches TaskStart)
    pub task_id: String,
    /// How the task ended
    pub status: TaskStatus,
    /// Total duration in seconds
    pub duration_secs: u64,
    /// Number of thinking iterations
    pub iterations: u32,
    /// Number of tool calls made
    pub tool_calls_count: u32,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Brief summary of what was accomplished
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

// =============================================================================
// Error Events
// =============================================================================

/// Data for Error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorData {
    /// Error message
    pub message: String,
    /// Error type/category
    pub error_type: ErrorType,
    /// Additional context about what was happening
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Whether the error was recovered from
    #[serde(default)]
    pub recovered: bool,
    /// Associated task ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Associated tool name (if tool error)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    /// Error from tool execution
    ToolError,
    /// Error from LLM provider
    LlmError,
    /// Timeout during operation
    Timeout,
    /// Rate limit hit
    RateLimit,
    /// Permission/approval denied
    PermissionDenied,
    /// Internal/unexpected error
    Internal,
    /// User cancelled the operation
    Cancelled,
}

// =============================================================================
// Sub-Agent Events
// =============================================================================

/// Data for SubAgentSpawn event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentSpawnData {
    /// Session ID of the child agent
    pub child_session_id: String,
    /// Mission description for the sub-agent
    pub mission: String,
    /// Specific task assigned
    pub task: String,
    /// Depth in the agent hierarchy (1 = first sub-agent)
    pub depth: u32,
    /// Parent task ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<String>,
}

/// Data for SubAgentComplete event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentCompleteData {
    /// Session ID of the child agent
    pub child_session_id: String,
    /// Whether the sub-agent succeeded
    pub success: bool,
    /// Brief summary of the result
    pub result_summary: String,
    /// Duration in seconds
    pub duration_secs: u64,
    /// Parent task ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<String>,
}

// =============================================================================
// Approval Events
// =============================================================================

/// Data for ApprovalRequested event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequestedData {
    /// The command or action requiring approval
    pub command: String,
    /// Risk level assessed
    pub risk_level: String,
    /// Warning messages shown to user
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Associated task ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

/// Data for ApprovalGranted event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalGrantedData {
    /// The command that was approved
    pub command: String,
    /// Type of approval (once, session, always)
    pub approval_type: String,
    /// Associated task ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

/// Data for ApprovalDenied event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDeniedData {
    /// The command that was denied
    pub command: String,
    /// Associated task ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

// =============================================================================
// Helper Implementations
// =============================================================================

impl ToolCallData {
    /// Create from a tool call, generating a summary
    pub fn from_tool_call(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: JsonValue,
        task_id: Option<String>,
    ) -> Self {
        let name = name.into();
        let summary = Self::generate_summary(&name, &arguments);

        Self {
            tool_call_id: tool_call_id.into(),
            name,
            arguments,
            summary: Some(summary),
            task_id,
        }
    }

    fn generate_summary(name: &str, arguments: &JsonValue) -> String {
        // Generate a brief human-readable summary of the tool call
        match name {
            "terminal" => {
                if let Some(cmd) = arguments.get("command").and_then(|v| v.as_str()) {
                    let truncated = if cmd.len() > 50 {
                        format!("{}...", &cmd[..47])
                    } else {
                        cmd.to_string()
                    };
                    format!("`{}`", truncated)
                } else {
                    "terminal command".to_string()
                }
            }
            "web_search" => {
                if let Some(query) = arguments.get("query").and_then(|v| v.as_str()) {
                    format!("\"{}\"", query)
                } else {
                    "web search".to_string()
                }
            }
            "web_fetch" => {
                if let Some(url) = arguments.get("url").and_then(|v| v.as_str()) {
                    let truncated = if url.len() > 40 {
                        format!("{}...", &url[..37])
                    } else {
                        url.to_string()
                    };
                    truncated
                } else {
                    "fetch URL".to_string()
                }
            }
            _ => {
                // Generic: show first argument value if simple
                if let Some(obj) = arguments.as_object() {
                    if let Some((_, first_val)) = obj.iter().next() {
                        if let Some(s) = first_val.as_str() {
                            let truncated = if s.len() > 30 {
                                format!("{}...", &s[..27])
                            } else {
                                s.to_string()
                            };
                            return truncated;
                        }
                    }
                }
                name.to_string()
            }
        }
    }
}

impl ErrorData {
    /// Create a tool error
    pub fn tool_error(
        tool_name: impl Into<String>,
        message: impl Into<String>,
        task_id: Option<String>,
    ) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::ToolError,
            context: None,
            recovered: false,
            task_id,
            tool_name: Some(tool_name.into()),
        }
    }

    /// Create an LLM error
    pub fn llm_error(message: impl Into<String>, task_id: Option<String>) -> Self {
        Self {
            message: message.into(),
            error_type: ErrorType::LlmError,
            context: None,
            recovered: false,
            task_id,
            tool_name: None,
        }
    }

    /// Mark as recovered
    pub fn with_recovered(mut self) -> Self {
        self.recovered = true;
        self
    }

    /// Add context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}
