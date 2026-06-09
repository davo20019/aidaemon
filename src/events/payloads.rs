//! Event payload data structures.
//!
//! Each event type has a corresponding payload struct that contains
//! the event-specific data serialized as JSON.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use super::{TaskOutcome, TaskStatus};
use crate::traits::{MessageAnnotation, MessageAttachment};

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
    /// Structured annotations attached to the rendered content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<MessageAnnotation>,
    /// Role of the speaker when the message was received.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_role: Option<String>,
    /// Turn ID (globally-unique UUID = this turn's opening user-message id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Structured file attachments (images, documents, etc.) saved to the inbox.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<MessageAttachment>,
}
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
    /// Structured annotations attached to the rendered content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<MessageAnnotation>,
    /// Turn ID (globally-unique UUID = this turn's opening user-message id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
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
    /// Turn ID (globally-unique UUID = this turn's opening user-message id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
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
    /// Structured annotations attached to the rendered content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<MessageAnnotation>,
    /// Turn ID (globally-unique UUID = this turn's opening user-message id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Files saved for agent vision (e.g. browser screenshots).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<MessageAttachment>,
}

// =============================================================================
// LLM Events
// =============================================================================

/// Data for LlmCall event — per-call latency and token telemetry.
///
/// Emitted once per LLM provider call in the agent loop (including the root
/// orchestrator turns that `task_activity` rows do not cover). Captures the
/// missing "how long did the model take?" signal plus fallback metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallData {
    /// Unique identifier correlating this event with a token_usage row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// Background or auxiliary call purpose (e.g. summarization, intent_classifier).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_purpose: Option<String>,
    /// Associated task ID.
    pub task_id: String,
    /// Iteration where this call occurred (1-based).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration: Option<u32>,
    /// Model requested for this call.
    pub model: String,
    /// Model that actually produced the response (differs from `model` on fallback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_model: Option<String>,
    /// Whether the call fell back to a different model/provider.
    #[serde(default)]
    pub fell_back: bool,
    /// Number of provider attempts made (1 = no retry/fallback).
    #[serde(default)]
    pub attempts: u32,
    /// End-to-end latency including any recovery/fallback, in milliseconds.
    pub latency_ms: u64,
    /// Actual input (prompt) tokens reported by the provider.
    #[serde(default)]
    pub input_tokens: u32,
    /// Actual output (completion) tokens reported by the provider.
    #[serde(default)]
    pub output_tokens: u32,
    /// Input tokens served from a provider-side cache, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u32>,
    /// Input tokens newly written into a provider-side cache, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Input tokens that were not provider-cache hits (`input_tokens - cached_input_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fresh_input_tokens: Option<u32>,
    /// Estimated input tokens computed at message-build time (for est-vs-actual drift).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub est_input_tokens: Option<u32>,
    /// Number of tool calls requested in the response.
    #[serde(default)]
    pub tool_calls_count: u32,
    /// Message-build phase duration for this iteration, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_ms: Option<u64>,
    /// Hash of message zero (system prompt) for prefix-cache attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_hash_system: Option<String>,
    /// Hash of the pre-current-turn history region for prefix-cache attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_hash_pre_boundary: Option<String>,
    /// Hash of effective tool definitions sent to the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_defs_hash: Option<String>,
    /// Hash of the injected session summary message, if present.
    /// Retired by Pillar A; always written as `None`/empty. Kept for
    /// parser/back-compat so older event rows can still be deserialized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_summary_hash: Option<String>,
    /// Hash of the task-context tail message (the `role == "system"` message
    /// whose content starts with `[Task Context]`) within the `[1..boundary)`
    /// region. `None`/empty when no tail is present.
    ///
    /// A tail-only flip (this field changes while `prefix_hash_archived` is
    /// stable) is EXPECTED per task: the tail carries per-task volatile context
    /// (timestamp, session context, memories) that changes every turn. It is
    /// not a bug signal. Only a `prefix_hash_archived` flip without a Window
    /// decision / Prefix mutation / fp_mismatch indicates a problem.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_hash: Option<String>,
    /// Hash of the pre-boundary archived region `[1..boundary)` EXCLUDING the
    /// tail message. Equals `prefix_hash_pre_boundary` when no tail is
    /// present.
    ///
    /// Diagnosis rule: `prefix_hash_archived` flipping WITHOUT a Window
    /// decision / Prefix mutation / fp_mismatch = bug; tail-only flip =
    /// expected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_hash_archived: Option<String>,
    /// Index of the current-turn boundary in the final provider payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_pos: Option<usize>,
    /// Number of messages in the final provider payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_count: Option<usize>,
    /// Whether this call forced text-only mode and omitted tool definitions.
    #[serde(default)]
    pub force_text: bool,
    /// Whether provider token usage was available for this call.
    #[serde(default)]
    pub token_usage_present: bool,
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
    pub response_direct_return_total: u64,
    pub response_fallthrough_total: u64,
    pub orchestration_route_clarification_required_total: u64,
    pub orchestration_route_tools_required_total: u64,
    pub orchestration_route_short_correction_direct_reply_total: u64,
    pub orchestration_route_acknowledgment_direct_reply_total: u64,
    pub orchestration_route_default_continue_total: u64,
    pub context_bleed_prevented_total: u64,
    pub context_mismatch_preflight_drop_total: u64,
    pub followup_mode_overrides_total: u64,
    pub cross_scope_blocked_total: u64,
    pub route_drift_alert_total: u64,
    pub route_drift_failsafe_activation_total: u64,
    pub route_failsafe_active_turn_total: u64,
    pub tokens_failed_tasks_total: u64,
    pub est_input_token_samples: u64,
    pub est_input_tokens_total: u64,
    pub est_msg_tokens_total: u64,
    pub est_tool_tokens_total: u64,
    pub est_tool_tokens_high_share_total: u64,
    pub est_tool_tokens_high_abs_total: u64,
    pub no_progress_iterations_total: u64,
    pub deferred_no_tool_forced_required_total: u64,
    pub deferred_no_tool_deferral_detected_total: u64,
    pub deferred_no_tool_model_switch_total: u64,
    pub deferred_no_tool_error_marker_total: u64,
    pub llm_payload_invalid_total: u64,
    pub llm_payload_invalid_breakdown: Vec<LlmPayloadInvalidMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmPayloadInvalidMetric {
    pub provider: String,
    pub model: String,
    pub reason: String,
    pub count: u64,
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
    /// Severity of the persisted decision signal.
    #[serde(default)]
    pub severity: DiagnosticSeverity,
    /// Stable machine-readable code for analytics/grouping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Flexible structured data for this decision.
    pub metadata: JsonValue,
    /// Human-readable summary of the decision.
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    #[default]
    Info,
    Warning,
    Error,
}

impl DiagnosticSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            DiagnosticSeverity::Info => "info",
            DiagnosticSeverity::Warning => "warning",
            DiagnosticSeverity::Error => "error",
        }
    }

    pub fn is_warning_or_higher(self) -> bool {
        matches!(
            self,
            DiagnosticSeverity::Warning | DiagnosticSeverity::Error
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionType {
    SkillMatch,
    MemoryRetrieval,
    IntentGate,
    ExecutionPlanningGate,
    ExecutionCritiquePass,
    ExecutionBudgetSelection,
    ExecutionStateSnapshot,
    EvidenceGate,
    ExecutionFailureClassification,
    PostExecutionValidation,
    RepetitiveCallDetection,
    SemanticReadDecision,
    ConsecutiveSameToolDetection,
    AlternatingPatternDetection,
    ToolBudgetBlock,
    RouteDriftAlert,
    StoppingCondition,
    InstructionsSnapshot,
    BudgetAutoExtension,
    LlmEfficiencyAlert,
}

impl DecisionType {
    pub fn as_str(self) -> &'static str {
        match self {
            DecisionType::SkillMatch => "skill_match",
            DecisionType::MemoryRetrieval => "memory_retrieval",
            DecisionType::IntentGate => "intent_gate",
            DecisionType::ExecutionPlanningGate => "execution_planning_gate",
            DecisionType::ExecutionCritiquePass => "execution_critique_pass",
            DecisionType::ExecutionBudgetSelection => "execution_budget_selection",
            DecisionType::ExecutionStateSnapshot => "execution_state_snapshot",
            DecisionType::EvidenceGate => "evidence_gate",
            DecisionType::ExecutionFailureClassification => "execution_failure_classification",
            DecisionType::PostExecutionValidation => "post_execution_validation",
            DecisionType::RepetitiveCallDetection => "repetitive_call_detection",
            DecisionType::SemanticReadDecision => "semantic_read_decision",
            DecisionType::ConsecutiveSameToolDetection => "consecutive_same_tool_detection",
            DecisionType::AlternatingPatternDetection => "alternating_pattern_detection",
            DecisionType::ToolBudgetBlock => "tool_budget_block",
            DecisionType::RouteDriftAlert => "route_drift_alert",
            DecisionType::StoppingCondition => "stopping_condition",
            DecisionType::InstructionsSnapshot => "instructions_snapshot",
            DecisionType::BudgetAutoExtension => "budget_auto_extension",
            DecisionType::LlmEfficiencyAlert => "llm_efficiency_alert",
        }
    }
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
    /// Turn ID (globally-unique UUID = this turn's opening user-message id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

/// Data for TaskEnd event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEndData {
    /// Task ID (matches TaskStart)
    pub task_id: String,
    /// How the task ended
    pub status: TaskStatus,
    /// Semantic result delivered by this execution.
    #[serde(default)]
    pub outcome: Option<TaskOutcome>,
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
    /// Per-task efficiency rollup captured when the task ends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub efficiency: Option<TaskEfficiencyData>,
    /// Turn ID (globally-unique UUID = this turn's opening user-message id).
    /// For recovery TaskEnd this is the interrupted task's ORIGINAL turn_id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Per-task harness effectiveness snapshot (routing, progress, quality, cost).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness_eval: Option<HarnessEvalSnapshot>,
}

impl TaskEndData {
    /// Single resolver for semantic outcome, including legacy fallback from status.
    pub fn effective_outcome(&self) -> TaskOutcome {
        if let Some(outcome) = self.outcome {
            return outcome;
        }
        match self.status {
            TaskStatus::Completed => TaskOutcome::Succeeded,
            TaskStatus::Cancelled | TaskStatus::Failed => TaskOutcome::Failed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEfficiencyData {
    pub llm_calls: u64,
    pub attempts: u64,
    pub fell_back_count: u64,
    pub p95_latency_ms: u64,
    pub max_latency_ms: u64,
    pub max_latency_iteration: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fresh_input_tokens: Option<u64>,
    pub est_input_drift: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
}

// =============================================================================
// Harness evaluation (effectiveness snapshot on TaskEnd)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessEvalSnapshot {
    pub task_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub depth: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<String>,
    pub completion_task_kind: String,
    pub orchestration_route: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub followup_mode: Option<String>,
    pub routing: RoutingEvalPayload,
    pub progress: ProgressEvalPayload,
    pub quality: QualityEvalPayload,
    pub cost: CostEvalPayload,
    pub scores: HarnessScoresPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessScoresPayload {
    pub routing_accuracy: f32,
    pub progress_yield: f32,
    pub contract_fulfillment: f32,
    pub cost_efficiency: f32,
    pub overall: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingEvalPayload {
    pub orchestration_route: String,
    pub tools_required_predicted: bool,
    pub tools_actually_used: bool,
    pub direct_return_attempted: bool,
    pub route_drift_failsafe: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills_activated: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_profile: Option<String>,
    #[serde(default)]
    pub model_escalated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEvalPayload {
    pub iterations: u32,
    pub tool_calls_attempted: u32,
    pub tool_calls_succeeded: u32,
    pub evidence_gain_total: u32,
    pub no_progress_iterations: u32,
    pub stall_guard_fires: u32,
    pub repetition_guard_fires: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_steps_completed: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_steps_total: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEvalPayload {
    pub outcome: String,
    pub stop_reason: String,
    pub contract: ContractFulfillmentPayload,
    pub post_exec_validation_failures: u32,
    pub unrecovered_errors: u32,
    #[serde(default)]
    pub approval_denied: bool,
    #[serde(default)]
    pub contract_fulfilled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractFulfillmentPayload {
    pub expects_mutation: bool,
    pub mutation_count: u32,
    pub requires_observation: bool,
    pub observation_count: u32,
    pub verification_required: bool,
    pub verification_count: u32,
    pub verification_blocks: u32,
    #[serde(default)]
    pub fulfilled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEvalPayload {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub weighted_tokens: u64,
    pub llm_calls: u32,
    pub fell_back_count: u32,
    #[serde(default)]
    pub sub_agent_weighted_tokens: u64,
    #[serde(default)]
    pub sub_agent_spawn_count: u32,
    #[serde(default)]
    pub sub_agent_failures: u32,
    #[serde(default)]
    pub tokens_failed_waste: bool,
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
    /// Stable specialist profile assigned to the child agent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specialist_kind: Option<String>,
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
    /// Stable specialist profile assigned to the child agent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specialist_kind: Option<String>,
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
            turn_id: None,
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
        use crate::utils::truncate_str;
        // Generate a brief human-readable summary of the tool call
        match name {
            "terminal" => {
                if let Some(cmd) = arguments.get("command").and_then(|v| v.as_str()) {
                    format!("`{}`", truncate_str(cmd, 50))
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
                    truncate_str(url, 40)
                } else {
                    "fetch URL".to_string()
                }
            }
            _ => {
                // Generic: show first argument value if simple
                if let Some(obj) = arguments.as_object() {
                    if let Some((_, first_val)) = obj.iter().next() {
                        if let Some(s) = first_val.as_str() {
                            return truncate_str(s, 30);
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
    fn llm_call_data_serde_roundtrip() {
        let data = LlmCallData {
            call_id: Some("call-42".to_string()),
            call_purpose: Some("agent_loop".to_string()),
            task_id: "task-42".to_string(),
            iteration: Some(2),
            model: "primary".to_string(),
            final_model: Some("fallback".to_string()),
            fell_back: true,
            attempts: 3,
            latency_ms: 1234,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: Some(60),
            cache_creation_input_tokens: Some(10),
            fresh_input_tokens: Some(40),
            est_input_tokens: Some(120),
            tool_calls_count: 1,
            build_ms: Some(7),
            prefix_hash_system: Some("sys-hash".to_string()),
            prefix_hash_pre_boundary: Some("pre-hash".to_string()),
            tool_defs_hash: Some("tools-hash".to_string()),
            // session_summary_hash: retired by Pillar A; always None at write time.
            session_summary_hash: None,
            tail_hash: Some("tail-hash".to_string()),
            prefix_hash_archived: Some("archived-hash".to_string()),
            boundary_pos: Some(12),
            message_count: Some(18),
            force_text: true,
            token_usage_present: true,
        };
        let json = serde_json::to_value(&data).expect("serialize");
        let back: LlmCallData = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back.task_id, "task-42");
        assert_eq!(back.iteration, Some(2));
        assert_eq!(back.final_model.as_deref(), Some("fallback"));
        assert!(back.fell_back);
        assert_eq!(back.attempts, 3);
        assert_eq!(back.latency_ms, 1234);
        assert_eq!(back.input_tokens, 100);
        assert_eq!(back.output_tokens, 50);
        assert_eq!(back.cached_input_tokens, Some(60));
        assert_eq!(back.cache_creation_input_tokens, Some(10));
        assert_eq!(back.fresh_input_tokens, Some(40));
        assert_eq!(back.est_input_tokens, Some(120));
        assert_eq!(back.tool_calls_count, 1);
        assert_eq!(back.build_ms, Some(7));
        assert_eq!(back.prefix_hash_system.as_deref(), Some("sys-hash"));
        assert_eq!(back.prefix_hash_pre_boundary.as_deref(), Some("pre-hash"));
        assert_eq!(back.tool_defs_hash.as_deref(), Some("tools-hash"));
        // session_summary_hash is retired; None skips serialization, deserializes as None.
        assert!(back.session_summary_hash.is_none());
        assert_eq!(back.tail_hash.as_deref(), Some("tail-hash"));
        assert_eq!(back.prefix_hash_archived.as_deref(), Some("archived-hash"));
        assert_eq!(back.boundary_pos, Some(12));
        assert_eq!(back.message_count, Some(18));
        assert!(back.force_text);
    }

    #[test]
    fn llm_call_data_minimal_defaults() {
        // Only required fields present; optionals/defaults fill in.
        let data: LlmCallData = serde_json::from_value(json!({
            "task_id": "t1",
            "iteration": 1,
            "model": "m",
            "latency_ms": 0
        }))
        .expect("deserialize minimal LlmCallData");
        assert!(!data.fell_back);
        assert_eq!(data.attempts, 0);
        assert_eq!(data.input_tokens, 0);
        assert!(data.cached_input_tokens.is_none());
        assert!(data.cache_creation_input_tokens.is_none());
        assert!(data.fresh_input_tokens.is_none());
        assert!(data.final_model.is_none());
        assert!(data.est_input_tokens.is_none());
        assert!(data.build_ms.is_none());
        assert!(data.prefix_hash_system.is_none());
        assert!(data.prefix_hash_pre_boundary.is_none());
        assert!(data.tool_defs_hash.is_none());
        assert!(data.session_summary_hash.is_none());
        assert!(data.tail_hash.is_none());
        assert!(data.prefix_hash_archived.is_none());
        assert!(data.boundary_pos.is_none());
        assert!(data.message_count.is_none());
        assert!(!data.force_text);
    }

    #[test]
    fn task_end_data_defaults_efficiency_to_none() {
        let data: TaskEndData = serde_json::from_value(json!({
            "task_id": "t1",
            "status": "completed",
            "duration_secs": 5,
            "iterations": 2,
            "tool_calls_count": 1
        }))
        .expect("deserialize minimal TaskEndData");

        assert!(data.efficiency.is_none());
        assert!(data.harness_eval.is_none());
        assert!(data.outcome.is_none());
        assert_eq!(data.effective_outcome(), TaskOutcome::Succeeded);
    }

    #[test]
    fn task_end_legacy_failed_status_defaults_outcome_to_failed() {
        let data: TaskEndData = serde_json::from_value(json!({
            "task_id": "t1",
            "status": "failed",
            "duration_secs": 1,
            "iterations": 1,
            "tool_calls_count": 0
        }))
        .expect("deserialize legacy failed task end");
        assert_eq!(data.effective_outcome(), TaskOutcome::Failed);
    }

    #[test]
    fn task_end_outcome_precedence_over_status() {
        let data: TaskEndData = serde_json::from_value(json!({
            "task_id": "t1",
            "status": "completed",
            "outcome": "partial",
            "duration_secs": 1,
            "iterations": 1,
            "tool_calls_count": 0
        }))
        .expect("deserialize partial completed task end");
        assert_eq!(data.effective_outcome(), TaskOutcome::Partial);
    }

    #[test]
    fn task_end_all_status_outcome_combinations_serialize() {
        for status in ["completed", "failed", "cancelled"] {
            for outcome in ["succeeded", "partial", "failed"] {
                let value = json!({
                    "task_id": "t1",
                    "status": status,
                    "outcome": outcome,
                    "duration_secs": 1,
                    "iterations": 1,
                    "tool_calls_count": 0
                });
                let data: TaskEndData =
                    serde_json::from_value(value).expect("deserialize task end combo");
                assert_eq!(data.outcome.map(|o| o.as_str()), Some(outcome));
            }
        }
    }

    #[test]
    fn task_end_data_roundtrips_efficiency_summary() {
        let data = TaskEndData {
            task_id: "t2".to_string(),
            status: TaskStatus::Completed,
            outcome: Some(TaskOutcome::Succeeded),
            duration_secs: 10,
            iterations: 3,
            tool_calls_count: 2,
            error: None,
            summary: Some("done".to_string()),
            efficiency: Some(TaskEfficiencyData {
                llm_calls: 3,
                attempts: 4,
                fell_back_count: 1,
                p95_latency_ms: 900,
                max_latency_ms: 1200,
                max_latency_iteration: 2,
                input_tokens: 42_000,
                output_tokens: 700,
                cached_input_tokens: Some(32_000),
                cache_creation_input_tokens: Some(1_000),
                fresh_input_tokens: Some(10_000),
                est_input_drift: -300,
                final_model: Some("fallback".to_string()),
                reasons: vec!["1 fallback(s)".to_string()],
            }),
            turn_id: None,
            harness_eval: None,
        };

        let json = serde_json::to_value(&data).expect("serialize");
        let back: TaskEndData = serde_json::from_value(json).expect("deserialize");
        let efficiency = back.efficiency.expect("efficiency summary");
        assert_eq!(efficiency.llm_calls, 3);
        assert_eq!(efficiency.attempts, 4);
        assert_eq!(efficiency.fell_back_count, 1);
        assert_eq!(efficiency.p95_latency_ms, 900);
        assert_eq!(efficiency.max_latency_ms, 1200);
        assert_eq!(efficiency.max_latency_iteration, 2);
        assert_eq!(efficiency.input_tokens, 42_000);
        assert_eq!(efficiency.output_tokens, 700);
        assert_eq!(efficiency.cached_input_tokens, Some(32_000));
        assert_eq!(efficiency.cache_creation_input_tokens, Some(1_000));
        assert_eq!(efficiency.fresh_input_tokens, Some(10_000));
        assert_eq!(efficiency.est_input_drift, -300);
        assert_eq!(efficiency.final_model.as_deref(), Some("fallback"));
        assert_eq!(efficiency.reasons, vec!["1 fallback(s)"]);
    }

    #[test]
    fn conversation_payloads_turn_id_survives_roundtrip() {
        // UserMessageData
        let um = UserMessageData {
            content: "hi".to_string(),
            message_id: None,
            has_attachments: false,
            annotations: vec![],
            user_role: None,
            turn_id: Some("t1".to_string()),
            attachments: vec![],
        };
        let back: UserMessageData =
            serde_json::from_value(serde_json::to_value(&um).unwrap()).unwrap();
        assert_eq!(back.turn_id.as_deref(), Some("t1"));

        // AssistantResponseData
        let ar = AssistantResponseData {
            message_id: None,
            content: Some("ok".to_string()),
            tool_calls: None,
            model: "m".to_string(),
            input_tokens: None,
            output_tokens: None,
            annotations: vec![],
            turn_id: Some("t1".to_string()),
        };
        let back: AssistantResponseData =
            serde_json::from_value(serde_json::to_value(&ar).unwrap()).unwrap();
        assert_eq!(back.turn_id.as_deref(), Some("t1"));

        // ToolCallData
        let tc = ToolCallData {
            tool_call_id: "c1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "README.md"}),
            summary: None,
            task_id: None,
            idempotency_key: None,
            policy_rev: None,
            risk_score: None,
            turn_id: Some("t1".to_string()),
        };
        let back: ToolCallData =
            serde_json::from_value(serde_json::to_value(&tc).unwrap()).unwrap();
        assert_eq!(back.turn_id.as_deref(), Some("t1"));

        // ToolResultData
        let tr = ToolResultData {
            message_id: None,
            tool_call_id: "c1".to_string(),
            name: "read_file".to_string(),
            result: "data".to_string(),
            success: true,
            duration_ms: 1,
            error: None,
            task_id: None,
            annotations: vec![],
            turn_id: Some("t1".to_string()),
            attachments: vec![],
        };
        let back: ToolResultData =
            serde_json::from_value(serde_json::to_value(&tr).unwrap()).unwrap();
        assert_eq!(back.turn_id.as_deref(), Some("t1"));

        // TaskStartData
        let ts = TaskStartData {
            task_id: "task-1".to_string(),
            description: "do x".to_string(),
            parent_task_id: None,
            user_message: None,
            turn_id: Some("t1".to_string()),
        };
        let back: TaskStartData =
            serde_json::from_value(serde_json::to_value(&ts).unwrap()).unwrap();
        assert_eq!(back.turn_id.as_deref(), Some("t1"));

        // TaskEndData
        let te = TaskEndData {
            task_id: "task-1".to_string(),
            status: TaskStatus::Completed,
            outcome: Some(TaskOutcome::Succeeded),
            duration_secs: 1,
            iterations: 1,
            tool_calls_count: 0,
            error: None,
            summary: None,
            efficiency: None,
            turn_id: Some("t1".to_string()),
            harness_eval: None,
        };
        let back: TaskEndData = serde_json::from_value(serde_json::to_value(&te).unwrap()).unwrap();
        assert_eq!(back.turn_id.as_deref(), Some("t1"));
    }

    #[test]
    fn conversation_payloads_turn_id_absent_defaults_to_none() {
        // Minimal JSON without a turn_id key deserializes to None (back-compat).
        let um: UserMessageData = serde_json::from_value(json!({"content": "hi"})).unwrap();
        assert!(um.turn_id.is_none());

        let ar: AssistantResponseData = serde_json::from_value(json!({"model": "m"})).unwrap();
        assert!(ar.turn_id.is_none());

        let tc: ToolCallData = serde_json::from_value(json!({
            "tool_call_id": "c1", "name": "read_file", "arguments": {}
        }))
        .unwrap();
        assert!(tc.turn_id.is_none());

        let tr: ToolResultData = serde_json::from_value(json!({
            "tool_call_id": "c1", "name": "read_file", "result": "x",
            "success": true, "duration_ms": 1
        }))
        .unwrap();
        assert!(tr.turn_id.is_none());

        let ts: TaskStartData = serde_json::from_value(json!({
            "task_id": "task-1", "description": "do x"
        }))
        .unwrap();
        assert!(ts.turn_id.is_none());

        let te: TaskEndData = serde_json::from_value(json!({
            "task_id": "task-1", "status": "completed",
            "duration_secs": 1, "iterations": 1, "tool_calls_count": 0
        }))
        .unwrap();
        assert!(te.turn_id.is_none());
    }

    #[test]
    fn decision_point_data_serde_roundtrip() {
        let data = DecisionPointData {
            decision_type: DecisionType::IntentGate,
            task_id: "task-123".to_string(),
            iteration: 1,
            severity: DiagnosticSeverity::Warning,
            code: Some("intent_gate".to_string()),
            metadata: json!({"needs_tools": true, "can_answer_now": false}),
            summary: "Intent gate requested tool mode".to_string(),
        };
        let serialized = serde_json::to_string(&data).expect("serialize");
        let parsed: DecisionPointData = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(parsed.task_id, "task-123");
        assert_eq!(parsed.iteration, 1);
        assert_eq!(parsed.decision_type, DecisionType::IntentGate);
        assert_eq!(parsed.severity, DiagnosticSeverity::Warning);
        assert_eq!(parsed.code.as_deref(), Some("intent_gate"));
    }

    #[test]
    fn decision_point_data_defaults_new_fields_for_older_rows() {
        let data: DecisionPointData = serde_json::from_value(json!({
            "decision_type": "intent_gate",
            "task_id": "task-123",
            "iteration": 1,
            "metadata": {"needs_tools": true},
            "summary": "Intent gate requested tool mode"
        }))
        .expect("deserialize DecisionPointData");

        assert_eq!(data.severity, DiagnosticSeverity::Info);
        assert!(data.code.is_none());
    }

    #[test]
    fn failure_category_serde_roundtrip() {
        let cat = FailureCategory::SandboxPermissionBlock;
        let serialized = serde_json::to_string(&cat).expect("serialize");
        let parsed: FailureCategory = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(parsed, FailureCategory::SandboxPermissionBlock);
    }
}
