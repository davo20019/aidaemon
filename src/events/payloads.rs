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
// Conversation Events (canonical event stream)
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
    /// Canonical conversation message ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
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
    /// Provider-specific metadata (e.g., Gemini thought_signature).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<JsonValue>,
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
    /// Optional idempotency key for replay-safe execution tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    /// Optional policy revision used when this tool call was emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_rev: Option<u32>,
    /// Optional risk score (0.0-1.0) observed at tool-call time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk_score: Option<f32>,
}

/// Data for ToolResult event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultData {
    /// Canonical conversation message ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
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
// Policy / Routing Events
// =============================================================================

/// Data for PolicyDecision event (shadow vs thin-router decisions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecisionData {
    /// Associated task ID.
    pub task_id: String,
    /// Baseline model decision from prior routing path.
    pub old_model: String,
    /// Policy profile-based model decision.
    pub new_model: String,
    /// Baseline tier ("fast" | "primary" | "smart").
    pub old_tier: String,
    /// Policy profile ("cheap" | "balanced" | "strong").
    pub new_profile: String,
    /// Whether old/new decisions differed.
    pub diverged: bool,
    /// Whether policy routing is enforced.
    pub policy_enforce: bool,
    /// Risk score at routing time.
    pub risk_score: f32,
    /// Uncertainty score at routing time.
    pub uncertainty_score: f32,
}

/// Runtime policy metrics snapshot exposed by the dashboard API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMetricsData {
    pub tool_exposure_samples: u64,
    pub tool_exposure_before_sum: u64,
    pub tool_exposure_after_sum: u64,
    pub tool_schema_contract_rejections_total: u64,
    pub ambiguity_detected_total: u64,
    pub uncertainty_clarify_total: u64,
    pub context_refresh_total: u64,
    pub escalation_total: u64,
    pub fallback_expansion_total: u64,
    pub consultant_direct_return_total: u64,
    pub consultant_fallthrough_total: u64,
    pub consultant_route_clarification_required_total: u64,
    pub consultant_route_tools_required_total: u64,
    pub consultant_route_short_correction_direct_reply_total: u64,
    pub consultant_route_acknowledgment_direct_reply_total: u64,
    pub consultant_route_default_continue_total: u64,
    pub context_bleed_prevented_total: u64,
    pub context_mismatch_preflight_drop_total: u64,
    pub followup_mode_overrides_total: u64,
    pub cross_scope_blocked_total: u64,
    pub route_drift_alert_total: u64,
    pub route_drift_failsafe_activation_total: u64,
    pub route_failsafe_active_turn_total: u64,
    pub tokens_failed_tasks_total: u64,
    pub no_progress_iterations_total: u64,
}

/// Data for DecisionPoint event (flight recorder).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPointData {
    /// Decision type emitted at this point in the task.
    pub decision_type: DecisionType,
    /// Associated task ID.
    pub task_id: String,
    /// Iteration where this decision occurred (0 when outside loop).
    pub iteration: u32,
    /// Flexible structured data for this decision.
    pub metadata: JsonValue,
    /// Human-readable summary of the decision.
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionType {
    SkillMatch,
    MemoryRetrieval,
    IntentGate,
    RepetitiveCallDetection,
    ConsecutiveSameToolDetection,
    AlternatingPatternDetection,
    ToolBudgetBlock,
    RouteDriftAlert,
    StoppingCondition,
    InstructionsSnapshot,
    BudgetAutoExtension,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    BadAssumption,
    MissingContext,
    ToolFailure,
    SandboxPermissionBlock,
    InvalidEditPatch,
    DependencyRuntimeMismatch,
    PartialCompletionRegression,
    AgentLoop,
    ProviderError,
    IncorrectResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub event_id: i64,
    pub event_type: String,
    pub timestamp: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCauseCandidate {
    pub category: FailureCategory,
    pub confidence: f32,
    pub description: String,
    pub evidence: Vec<EvidenceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why_previous_step_looked_valid: Option<String>,
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
            idempotency_key: None,
            policy_rev: None,
            risk_score: None,
        }
    }

    pub fn with_policy_metadata(
        mut self,
        idempotency_key: Option<String>,
        policy_rev: Option<u32>,
        risk_score: Option<f32>,
    ) -> Self {
        self.idempotency_key = idempotency_key;
        self.policy_rev = policy_rev;
        self.risk_score = risk_score;
        self
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_call_data_defaults_policy_metadata_to_none() {
        let data: ToolCallData = serde_json::from_value(json!({
            "tool_call_id": "c1",
            "name": "read_file",
            "arguments": {"path":"README.md"}
        }))
        .expect("deserialize ToolCallData");
        assert!(data.idempotency_key.is_none());
        assert!(data.policy_rev.is_none());
        assert!(data.risk_score.is_none());
    }

    #[test]
    fn tool_call_data_with_policy_metadata_sets_optional_fields() {
        let data = ToolCallData::from_tool_call(
            "c1",
            "write_file",
            json!({"path":"notes.txt"}),
            Some("task-1".to_string()),
        )
        .with_policy_metadata(Some("idem-task-1-c1".to_string()), Some(3), Some(0.72));

        assert_eq!(data.idempotency_key.as_deref(), Some("idem-task-1-c1"));
        assert_eq!(data.policy_rev, Some(3));
        assert_eq!(data.risk_score, Some(0.72));
    }

    #[test]
    fn decision_point_data_serde_roundtrip() {
        let data = DecisionPointData {
            decision_type: DecisionType::IntentGate,
            task_id: "task-123".to_string(),
            iteration: 1,
            metadata: json!({"needs_tools": true, "can_answer_now": false}),
            summary: "Intent gate requested tool mode".to_string(),
        };
        let serialized = serde_json::to_string(&data).expect("serialize");
        let parsed: DecisionPointData = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(parsed.task_id, "task-123");
        assert_eq!(parsed.iteration, 1);
        assert_eq!(parsed.decision_type, DecisionType::IntentGate);
    }

    #[test]
    fn failure_category_serde_roundtrip() {
        let cat = FailureCategory::SandboxPermissionBlock;
        let serialized = serde_json::to_string(&cat).expect("serialize");
        let parsed: FailureCategory = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(parsed, FailureCategory::SandboxPermissionBlock);
    }
}
