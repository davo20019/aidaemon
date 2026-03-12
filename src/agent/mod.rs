use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::Utc;
use croner::Cron;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

use crate::channels::ChannelHub;
use crate::config::{IterationLimitConfig, PathAliasConfig, PolicyConfig};
use crate::events::{
    AssistantResponseData, DecisionPointData, DecisionType, ErrorData, EventStore, EventType,
    LlmPayloadInvalidMetric, PolicyMetricsData, SubAgentCompleteData, SubAgentSpawnData,
    TaskEndData, TaskStartData, TaskStatus, ThinkingStartData, ToolCallData, ToolCallInfo,
    ToolResultData,
};
use crate::execution_policy::{ApprovalMode, ExecutionPolicy, ModelProfile};
use crate::goal_tokens::{GoalRunBudgetStatus, GoalTokenRegistry};
use crate::llm_runtime::SharedLlmRuntime;
use crate::mcp::McpRegistry;
use crate::providers::{ProviderError, ProviderErrorKind};
use crate::router::{self, Router};
use crate::skills::{self, MemoryContext};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::goal_completion_summary_indicates_not_finished;
use crate::tools::VerificationTracker;
use crate::traits::{
    AgentRole, ChatOptions, Goal, Message, ModelProvider, ScheduledRunState, StateStore, Task,
    TaskActivity, Tool, ToolCall, ToolCapabilities, ToolChoiceMode, ToolRole,
};
use crate::types::{ApprovalResponse, ChannelContext, ChannelVisibility, UserRole};
// Re-export StatusUpdate from types for backwards compatibility
pub use crate::types::StatusUpdate;

/// Constants for stall and repetitive behavior detection
const MAX_STALL_ITERATIONS: usize = 5;
const DEFERRED_NO_TOOL_SWITCH_THRESHOLD: usize = 2;
const MAX_DEFERRED_NO_TOOL_MODEL_SWITCHES: usize = 1;
const DEFERRED_NO_TOOL_ERROR_MARKER: &str = "deferred-action no-tool loop";
/// After this many deferred-no-tool retries, accept substantive text-only responses
/// instead of continuing to force tool use. This prevents stalls on simple
/// conversational queries (greetings, capability questions, jokes) that don't need tools.
const DEFERRED_NO_TOOL_ACCEPT_THRESHOLD: usize = 2;
const MAX_REPETITIVE_CALLS: usize = 8;
const RECENT_CALLS_WINDOW: usize = 12;
/// After this many identical calls (same tool+args hash), skip execution and
/// inject a coaching message so the LLM adapts before the hard stall fires.
const REPETITIVE_REDIRECT_THRESHOLD: usize = 3;
/// If the same tool NAME is called this many consecutive iterations (even with
/// different arguments), treat it as a loop.  This catches the case where the
/// LLM keeps calling e.g. `terminal` with varied commands without progress.
/// Set high enough to allow complex multi-step investigations from mobile,
/// and to leave room for follow-up work after cli_agent returns.
const MAX_CONSECUTIVE_SAME_TOOL: usize = 8;
/// Hard iteration cap even in "unlimited" mode — prevents runaway resource
/// consumption if stall detection is bypassed (e.g. alternating tool names).
const HARD_ITERATION_CAP: usize = 200;
/// Maximum character length for old-interaction assistant messages in history.
/// Longer content is truncated with a "[prior turn, truncated]" marker to
/// prevent stale context from polluting subsequent interactions.
const MAX_OLD_ASSISTANT_CONTENT_CHARS: usize = 200;
/// Window size for detecting alternating tool patterns (A-B-A-B cycles).
const ALTERNATING_PATTERN_WINDOW: usize = 10;
const PROGRESS_SUMMARY_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes
/// Legacy fallback schedule text heuristics used as a guardrail when the model
/// omits schedule fields. These heuristics explicitly ignore "tell me about this
/// scheduled goal" meta-queries to avoid accidental schedule creation.
const ENABLE_SCHEDULE_HEURISTICS: bool = true;

#[cfg(test)]
#[path = "intent/intent_gate.rs"]
mod intent_gate;
#[cfg(test)]
use intent_gate::extract_intent_gate;
#[cfg(test)]
use intent_gate::parse_intent_gate_json;
#[path = "response_analysis.rs"]
mod response_analysis;
#[cfg(test)]
use response_analysis::has_action_promise;
#[cfg(test)]
use response_analysis::sanitize_response_analysis;
use response_analysis::{is_substantive_text_response, looks_like_deferred_action_response};
#[path = "intent/intent_routing.rs"]
mod intent_routing;
use intent_routing::{
    classify_intent_complexity, contains_keyword_as_words, infer_intent_gate,
    is_internal_maintenance_intent, IntentComplexity,
};
#[cfg(test)]
use intent_routing::{detect_schedule_heuristic, looks_like_recurring_intent_without_timing};
#[path = "policy/policy_signals.rs"]
mod policy_signals;
#[cfg(test)]
use policy_signals::is_short_user_correction;
use policy_signals::{
    build_policy_bundle, default_clarifying_question, detect_explicit_outcome_signal,
    tool_is_side_effecting,
};
#[path = "loop/evidence_state.rs"]
mod evidence_state;
pub(in crate::agent) use evidence_state::{
    assess_pre_execution_evidence_gate, has_completed_side_effecting_tool_call,
    record_successful_tool_evidence, EvidenceState,
};
#[path = "loop/validation_state.rs"]
mod validation_state;
pub(in crate::agent) use validation_state::{
    build_abandon_request, build_partial_done_blocked_request, build_reduce_scope_request,
    ApprovalState, LoopRepetitionReason, ValidationFailure, ValidationOutcome,
};
pub(crate) use validation_state::{
    build_needs_approval_request, derive_executor_step_result, persist_executor_handoff_context,
    persist_executor_result_context, ExecutorHandoff, ExecutorStepResult, PartialResult,
    StepValidationOutcome, TaskValidationOutcome, ValidationState,
};
#[path = "loop/execution_state.rs"]
mod execution_state;
#[cfg(test)]
pub(crate) use execution_state::ExecutionBudget;
pub(crate) use execution_state::TargetScope;
pub(in crate::agent) use execution_state::{
    classify_step_execution_outcome, compile_step_execution_plan, select_initial_execution_budget,
    ApprovalRequirement, ExecutionBudgetLimit, ExecutionPersistence, ExecutionState,
    StepExecutionOutcome,
};
#[path = "loop/loop_utils.rs"]
mod loop_utils;
#[path = "policy/recall_guardrails.rs"]
mod recall_guardrails;
use loop_utils::{
    build_task_boundary_hint, classify_execution_failure_kind,
    classify_tool_result_failure_with_context, extract_command_from_args,
    extract_file_path_from_args, extract_key_error_line, extract_send_file_dedupe_key_from_args,
    fixup_message_ordering, hash_tool_call, is_trigger_session, semantic_failure_limit,
    strip_appended_diagnostics, ExecutionFailureKind, ToolFailureClass,
};
#[path = "runtime/post_task.rs"]
mod post_task;
use post_task::LearningContext;
pub(in crate::agent) use post_task::ReplayNoteCategory;
#[path = "loop/stopping_conditions.rs"]
mod stopping_conditions;
#[path = "loop/tool_loop_state.rs"]
mod tool_loop_state;

#[path = "loop/bootstrap_phase.rs"]
mod bootstrap_phase;
#[path = "loop/completion_phase.rs"]
mod completion_phase;
#[path = "loop/direct_return.rs"]
mod direct_return;
#[path = "loop/fallthrough.rs"]
mod fallthrough;
#[path = "runtime/graceful.rs"]
mod graceful;
#[path = "runtime/history.rs"]
mod history;
#[path = "loop/orchestration_phase.rs"]
mod orchestration_phase;
#[path = "loop/response_phase.rs"]
mod response_phase;
pub(in crate::agent) use history::CompletionContract;
pub(in crate::agent) use history::CompletionProgress;
pub(in crate::agent) use history::CompletionTaskKind;
pub(in crate::agent) use history::FollowupMode;
pub(in crate::agent) use history::TurnContext;
pub(in crate::agent) use history::VerificationTarget;
pub(in crate::agent) use history::VerificationTargetKind;
#[path = "runtime/llm.rs"]
mod llm;
#[path = "loop/llm_phase.rs"]
mod llm_phase;
#[path = "loop/main_loop.rs"]
mod main_loop;
#[path = "loop/message_build_phase.rs"]
mod message_build_phase;
#[path = "runtime/models.rs"]
mod models;
#[path = "runtime/resume.rs"]
mod resume;
#[path = "runtime/spawn.rs"]
mod spawn;
#[path = "loop/stopping_phase.rs"]
mod stopping_phase;
#[path = "loop/system_directives.rs"]
mod system_directives;
#[path = "runtime/system_prompt.rs"]
mod system_prompt;
#[path = "tools/tool_defs.rs"]
mod tool_defs;
#[path = "tools/tool_exec.rs"]
mod tool_exec;
#[path = "loop/tool_execution_phase.rs"]
mod tool_execution_phase;
#[path = "loop/tool_prelude_phase.rs"]
mod tool_prelude_phase;
#[path = "loop/tool_result_notices.rs"]
mod tool_result_notices;

pub(in crate::agent) use system_directives::{EarlyStopSeverity, SystemDirective};
use system_prompt::{build_tool_loop_system_prompt, format_goal_context, ToolLoopPromptStyle};
pub(in crate::agent) use tool_result_notices::ToolResultNotice;

#[cfg(test)]
use system_prompt::strip_markdown_section;

struct PolicyRuntimeMetrics {
    tool_exposure_samples: AtomicU64,
    tool_exposure_before_sum: AtomicU64,
    tool_exposure_after_sum: AtomicU64,
    tool_schema_contract_rejections_total: AtomicU64,
    ambiguity_detected_total: AtomicU64,
    uncertainty_clarify_total: AtomicU64,
    context_refresh_total: AtomicU64,
    escalation_total: AtomicU64,
    fallback_expansion_total: AtomicU64,
    response_direct_return_total: AtomicU64,
    response_fallthrough_total: AtomicU64,
    orchestration_route_clarification_required_total: AtomicU64,
    orchestration_route_tools_required_total: AtomicU64,
    orchestration_route_short_correction_direct_reply_total: AtomicU64,
    orchestration_route_acknowledgment_direct_reply_total: AtomicU64,
    orchestration_route_default_continue_total: AtomicU64,
    context_bleed_prevented_total: AtomicU64,
    context_mismatch_preflight_drop_total: AtomicU64,
    followup_mode_overrides_total: AtomicU64,
    cross_scope_blocked_total: AtomicU64,
    route_drift_alert_total: AtomicU64,
    route_drift_failsafe_activation_total: AtomicU64,
    route_failsafe_active_turn_total: AtomicU64,
    tokens_failed_tasks_total: AtomicU64,
    est_input_token_samples: AtomicU64,
    est_input_tokens_total: AtomicU64,
    est_msg_tokens_total: AtomicU64,
    est_tool_tokens_total: AtomicU64,
    est_tool_tokens_high_share_total: AtomicU64,
    est_tool_tokens_high_abs_total: AtomicU64,
    no_progress_iterations_total: AtomicU64,
    deferred_no_tool_forced_required_total: AtomicU64,
    deferred_no_tool_deferral_detected_total: AtomicU64,
    deferred_no_tool_model_switch_total: AtomicU64,
    deferred_no_tool_error_marker_total: AtomicU64,
    llm_payload_invalid_total: AtomicU64,
}

impl PolicyRuntimeMetrics {
    const fn new() -> Self {
        Self {
            tool_exposure_samples: AtomicU64::new(0),
            tool_exposure_before_sum: AtomicU64::new(0),
            tool_exposure_after_sum: AtomicU64::new(0),
            tool_schema_contract_rejections_total: AtomicU64::new(0),
            ambiguity_detected_total: AtomicU64::new(0),
            uncertainty_clarify_total: AtomicU64::new(0),
            context_refresh_total: AtomicU64::new(0),
            escalation_total: AtomicU64::new(0),
            fallback_expansion_total: AtomicU64::new(0),
            response_direct_return_total: AtomicU64::new(0),
            response_fallthrough_total: AtomicU64::new(0),
            orchestration_route_clarification_required_total: AtomicU64::new(0),
            orchestration_route_tools_required_total: AtomicU64::new(0),
            orchestration_route_short_correction_direct_reply_total: AtomicU64::new(0),
            orchestration_route_acknowledgment_direct_reply_total: AtomicU64::new(0),
            orchestration_route_default_continue_total: AtomicU64::new(0),
            context_bleed_prevented_total: AtomicU64::new(0),
            context_mismatch_preflight_drop_total: AtomicU64::new(0),
            followup_mode_overrides_total: AtomicU64::new(0),
            cross_scope_blocked_total: AtomicU64::new(0),
            route_drift_alert_total: AtomicU64::new(0),
            route_drift_failsafe_activation_total: AtomicU64::new(0),
            route_failsafe_active_turn_total: AtomicU64::new(0),
            tokens_failed_tasks_total: AtomicU64::new(0),
            est_input_token_samples: AtomicU64::new(0),
            est_input_tokens_total: AtomicU64::new(0),
            est_msg_tokens_total: AtomicU64::new(0),
            est_tool_tokens_total: AtomicU64::new(0),
            est_tool_tokens_high_share_total: AtomicU64::new(0),
            est_tool_tokens_high_abs_total: AtomicU64::new(0),
            no_progress_iterations_total: AtomicU64::new(0),
            deferred_no_tool_forced_required_total: AtomicU64::new(0),
            deferred_no_tool_deferral_detected_total: AtomicU64::new(0),
            deferred_no_tool_model_switch_total: AtomicU64::new(0),
            deferred_no_tool_error_marker_total: AtomicU64::new(0),
            llm_payload_invalid_total: AtomicU64::new(0),
        }
    }
}

static POLICY_METRICS: Lazy<PolicyRuntimeMetrics> = Lazy::new(PolicyRuntimeMetrics::new);
const MAX_LLM_PAYLOAD_INVALID_METRIC_KEYS: usize = 512;
const LLM_PAYLOAD_INVALID_OVERFLOW_PROVIDER: &str = "__other__";
const LLM_PAYLOAD_INVALID_OVERFLOW_MODEL: &str = "__other__";
const LLM_PAYLOAD_INVALID_OVERFLOW_REASON: &str = "__other__";
type PayloadInvalidMetricKey = (String, String, String);
type PayloadInvalidMetricMap = HashMap<PayloadInvalidMetricKey, u64>;
static LLM_PAYLOAD_INVALID_BREAKDOWN: Lazy<std::sync::Mutex<PayloadInvalidMetricMap>> =
    Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

pub(in crate::agent) fn provider_kind_metric_label(
    kind: crate::config::ProviderKind,
) -> &'static str {
    match kind {
        crate::config::ProviderKind::OpenaiCompatible => "openai_compatible",
        crate::config::ProviderKind::XaiNative => "xai_native",
        crate::config::ProviderKind::GoogleGenai => "google_genai",
        crate::config::ProviderKind::Anthropic => "anthropic",
    }
}

pub(in crate::agent) fn record_llm_payload_invalid_metric(
    provider: &str,
    model: &str,
    reason: &str,
) {
    POLICY_METRICS
        .llm_payload_invalid_total
        .fetch_add(1, Ordering::Relaxed);

    let Ok(mut breakdown) = LLM_PAYLOAD_INVALID_BREAKDOWN.lock() else {
        return;
    };
    let key = (provider.to_string(), model.to_string(), reason.to_string());
    if let Some(count) = breakdown.get_mut(&key) {
        *count = count.saturating_add(1);
        return;
    }

    if breakdown.len() >= MAX_LLM_PAYLOAD_INVALID_METRIC_KEYS {
        let overflow_key = (
            LLM_PAYLOAD_INVALID_OVERFLOW_PROVIDER.to_string(),
            LLM_PAYLOAD_INVALID_OVERFLOW_MODEL.to_string(),
            LLM_PAYLOAD_INVALID_OVERFLOW_REASON.to_string(),
        );
        let count = breakdown.entry(overflow_key).or_insert(0);
        *count = count.saturating_add(1);
        return;
    }

    breakdown.insert(key, 1);
}

fn llm_payload_invalid_breakdown_snapshot() -> Vec<LlmPayloadInvalidMetric> {
    let Ok(breakdown) = LLM_PAYLOAD_INVALID_BREAKDOWN.lock() else {
        return Vec::new();
    };
    let mut rows: Vec<LlmPayloadInvalidMetric> = breakdown
        .iter()
        .map(
            |((provider, model, reason), count)| LlmPayloadInvalidMetric {
                provider: provider.clone(),
                model: model.clone(),
                reason: reason.clone(),
                count: *count,
            },
        )
        .collect();
    rows.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.provider.cmp(&b.provider))
            .then_with(|| a.model.cmp(&b.model))
            .then_with(|| a.reason.cmp(&b.reason))
    });
    rows
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct RouteDriftSample {
    reason: RouteDriftReason,
    action: RouteDriftAction,
    reply_len: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum RouteDriftReason {
    ClarificationRequired,
    ToolsRequired,
    ShortCorrectionDirectReply,
    AcknowledgmentDirectReply,
    DefaultContinue,
    Unknown,
}

#[allow(dead_code)]
impl RouteDriftReason {
    fn from_str(reason: &str) -> Self {
        match reason {
            "clarification_required" => Self::ClarificationRequired,
            "tools_required" => Self::ToolsRequired,
            "short_correction_direct_reply" => Self::ShortCorrectionDirectReply,
            "acknowledgment_direct_reply" => Self::AcknowledgmentDirectReply,
            "default_continue" => Self::DefaultContinue,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum RouteDriftAction {
    Return,
    Continue,
    Unknown,
}

#[allow(dead_code)]
impl RouteDriftAction {
    fn from_str(action: &str) -> Self {
        match action {
            "return" => Self::Return,
            "continue" => Self::Continue,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct RouteDriftSessionState {
    samples: VecDeque<RouteDriftSample>,
    last_seen_epoch_secs: u64,
    last_alert_epoch_secs: u64,
    consecutive_anomaly_windows: u32,
    failsafe_until_epoch_secs: u64,
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct RouteDriftMonitor {
    sessions: HashMap<String, RouteDriftSessionState>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(in crate::agent) struct RouteDriftSignal {
    pub summary: String,
    pub failsafe_activated: bool,
}

#[allow(dead_code)]
const ROUTE_DRIFT_WINDOW_SIZE: usize = 24;
#[allow(dead_code)]
const ROUTE_DRIFT_MIN_WINDOW: usize = 12;
#[allow(dead_code)]
const ROUTE_DRIFT_ALERT_COOLDOWN_SECS: u64 = 300;
#[allow(dead_code)]
const ROUTE_DRIFT_FAILSAFE_DURATION_SECS: u64 = 900;
#[allow(dead_code)]
const ROUTE_DRIFT_FAILSAFE_STREAK: u32 = 2;
const ROUTE_DRIFT_MAX_TRACKED_SESSIONS: usize = 256;
const ROUTE_DRIFT_STALE_SESSION_SECS: u64 = 7200;

#[allow(dead_code)]
static ROUTE_DRIFT_MONITOR: Lazy<std::sync::Mutex<RouteDriftMonitor>> =
    Lazy::new(|| std::sync::Mutex::new(RouteDriftMonitor::default()));

#[allow(dead_code)]
fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[allow(dead_code)]
fn prune_route_drift_sessions(monitor: &mut RouteDriftMonitor, now: u64) {
    monitor.sessions.retain(|_, state| {
        now.saturating_sub(state.last_seen_epoch_secs) <= ROUTE_DRIFT_STALE_SESSION_SECS
            || state.failsafe_until_epoch_secs > now
    });

    if monitor.sessions.len() <= ROUTE_DRIFT_MAX_TRACKED_SESSIONS {
        return;
    }

    let mut oldest: Vec<(String, u64)> = monitor
        .sessions
        .iter()
        .map(|(session_id, state)| (session_id.clone(), state.last_seen_epoch_secs))
        .collect();
    oldest.sort_by_key(|(_, ts)| *ts);
    let remove_count = monitor
        .sessions
        .len()
        .saturating_sub(ROUTE_DRIFT_MAX_TRACKED_SESSIONS);
    for (session_id, _) in oldest.into_iter().take(remove_count) {
        monitor.sessions.remove(&session_id);
    }
}

#[allow(dead_code)]
pub(in crate::agent) fn observe_route_reason_for_drift(
    session_id: &str,
    route_reason: &str,
    route_action: &str,
    route_reply_len: Option<usize>,
) -> Option<RouteDriftSignal> {
    let now = now_epoch_secs();
    let Ok(mut monitor) = ROUTE_DRIFT_MONITOR.lock() else {
        return None;
    };
    let state = monitor
        .sessions
        .entry(session_id.to_string())
        .or_insert_with(RouteDriftSessionState::default);
    state.last_seen_epoch_secs = now;
    state.samples.push_back(RouteDriftSample {
        reason: RouteDriftReason::from_str(route_reason),
        action: RouteDriftAction::from_str(route_action),
        reply_len: route_reply_len,
    });
    while state.samples.len() > ROUTE_DRIFT_WINDOW_SIZE {
        state.samples.pop_front();
    }

    let sample_count = state.samples.len();
    if sample_count < ROUTE_DRIFT_MIN_WINDOW {
        prune_route_drift_sessions(&mut monitor, now);
        return None;
    }

    let tools_required = state
        .samples
        .iter()
        .filter(|sample| sample.reason == RouteDriftReason::ToolsRequired)
        .count();
    let default_continue = state
        .samples
        .iter()
        .filter(|sample| sample.reason == RouteDriftReason::DefaultContinue)
        .count();
    let clarification_required = state
        .samples
        .iter()
        .filter(|sample| sample.reason == RouteDriftReason::ClarificationRequired)
        .count();
    let empty_direct_replies = state
        .samples
        .iter()
        .filter(|sample| sample.action == RouteDriftAction::Return && sample.reply_len == Some(0))
        .count();

    let total = sample_count as f64;
    let tools_rate = tools_required as f64 / total;
    let default_rate = default_continue as f64 / total;
    let clarification_rate = clarification_required as f64 / total;

    let mut anomaly_reasons: Vec<String> = Vec::new();
    if empty_direct_replies > 0 {
        anomaly_reasons.push(format!("empty_direct_replies={}", empty_direct_replies));
    }
    if tools_rate <= 0.05 && default_rate >= 0.85 {
        anomaly_reasons.push(format!(
            "tools_required_rate={:.0}% default_continue_rate={:.0}%",
            tools_rate * 100.0,
            default_rate * 100.0
        ));
    }
    if clarification_rate >= 0.75 {
        anomaly_reasons.push(format!(
            "clarification_required_rate={:.0}%",
            clarification_rate * 100.0
        ));
    }

    let mut signal: Option<RouteDriftSignal> = None;
    if anomaly_reasons.is_empty() {
        state.consecutive_anomaly_windows = 0;
    } else {
        state.consecutive_anomaly_windows = state.consecutive_anomaly_windows.saturating_add(1);
        let cooldown_elapsed =
            now.saturating_sub(state.last_alert_epoch_secs) >= ROUTE_DRIFT_ALERT_COOLDOWN_SECS;
        let mut failsafe_activated = false;
        if state.consecutive_anomaly_windows >= ROUTE_DRIFT_FAILSAFE_STREAK
            && state.failsafe_until_epoch_secs <= now
        {
            state.failsafe_until_epoch_secs = now + ROUTE_DRIFT_FAILSAFE_DURATION_SECS;
            POLICY_METRICS
                .route_drift_failsafe_activation_total
                .fetch_add(1, Ordering::Relaxed);
            failsafe_activated = true;
        }
        if cooldown_elapsed || failsafe_activated {
            state.last_alert_epoch_secs = now;
            POLICY_METRICS
                .route_drift_alert_total
                .fetch_add(1, Ordering::Relaxed);
            signal = Some(RouteDriftSignal {
                summary: format!(
                    "route drift anomaly: {} (window={} turns)",
                    anomaly_reasons.join(", "),
                    sample_count
                ),
                failsafe_activated,
            });
        }
    }

    prune_route_drift_sessions(&mut monitor, now);
    signal
}

pub(in crate::agent) fn route_failsafe_active_for_session(session_id: &str) -> bool {
    let now = now_epoch_secs();
    let Ok(mut monitor) = ROUTE_DRIFT_MONITOR.lock() else {
        return false;
    };
    let active = monitor
        .sessions
        .get(session_id)
        .is_some_and(|state| state.failsafe_until_epoch_secs > now);
    if active {
        POLICY_METRICS
            .route_failsafe_active_turn_total
            .fetch_add(1, Ordering::Relaxed);
    }
    prune_route_drift_sessions(&mut monitor, now);
    active
}

#[cfg(test)]
pub(crate) fn set_route_failsafe_for_session_for_test(session_id: &str, active: bool) {
    let now = now_epoch_secs();
    let Ok(mut monitor) = ROUTE_DRIFT_MONITOR.lock() else {
        return;
    };
    if active {
        let state = monitor
            .sessions
            .entry(session_id.to_string())
            .or_insert_with(RouteDriftSessionState::default);
        state.last_seen_epoch_secs = now;
        state.failsafe_until_epoch_secs = now + ROUTE_DRIFT_FAILSAFE_DURATION_SECS;
    } else {
        monitor.sessions.remove(session_id);
    }
}

struct PolicyRuntimeTunables {
    initialized: AtomicBool,
    // Stored as basis points (e.g. 0.55 => 5500) for lock-free updates.
    uncertainty_threshold_bp: AtomicU64,
}

impl PolicyRuntimeTunables {
    const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            uncertainty_threshold_bp: AtomicU64::new(5500),
        }
    }
}

static POLICY_TUNABLES: Lazy<PolicyRuntimeTunables> = Lazy::new(PolicyRuntimeTunables::new);

pub fn policy_metrics_snapshot() -> PolicyMetricsData {
    PolicyMetricsData {
        tool_exposure_samples: POLICY_METRICS.tool_exposure_samples.load(Ordering::Relaxed),
        tool_exposure_before_sum: POLICY_METRICS
            .tool_exposure_before_sum
            .load(Ordering::Relaxed),
        tool_exposure_after_sum: POLICY_METRICS
            .tool_exposure_after_sum
            .load(Ordering::Relaxed),
        tool_schema_contract_rejections_total: POLICY_METRICS
            .tool_schema_contract_rejections_total
            .load(Ordering::Relaxed),
        ambiguity_detected_total: POLICY_METRICS
            .ambiguity_detected_total
            .load(Ordering::Relaxed),
        uncertainty_clarify_total: POLICY_METRICS
            .uncertainty_clarify_total
            .load(Ordering::Relaxed),
        context_refresh_total: POLICY_METRICS.context_refresh_total.load(Ordering::Relaxed),
        escalation_total: POLICY_METRICS.escalation_total.load(Ordering::Relaxed),
        fallback_expansion_total: POLICY_METRICS
            .fallback_expansion_total
            .load(Ordering::Relaxed),
        response_direct_return_total: POLICY_METRICS
            .response_direct_return_total
            .load(Ordering::Relaxed),
        response_fallthrough_total: POLICY_METRICS
            .response_fallthrough_total
            .load(Ordering::Relaxed),
        orchestration_route_clarification_required_total: POLICY_METRICS
            .orchestration_route_clarification_required_total
            .load(Ordering::Relaxed),
        orchestration_route_tools_required_total: POLICY_METRICS
            .orchestration_route_tools_required_total
            .load(Ordering::Relaxed),
        orchestration_route_short_correction_direct_reply_total: POLICY_METRICS
            .orchestration_route_short_correction_direct_reply_total
            .load(Ordering::Relaxed),
        orchestration_route_acknowledgment_direct_reply_total: POLICY_METRICS
            .orchestration_route_acknowledgment_direct_reply_total
            .load(Ordering::Relaxed),
        orchestration_route_default_continue_total: POLICY_METRICS
            .orchestration_route_default_continue_total
            .load(Ordering::Relaxed),
        context_bleed_prevented_total: POLICY_METRICS
            .context_bleed_prevented_total
            .load(Ordering::Relaxed),
        context_mismatch_preflight_drop_total: POLICY_METRICS
            .context_mismatch_preflight_drop_total
            .load(Ordering::Relaxed),
        followup_mode_overrides_total: POLICY_METRICS
            .followup_mode_overrides_total
            .load(Ordering::Relaxed),
        cross_scope_blocked_total: POLICY_METRICS
            .cross_scope_blocked_total
            .load(Ordering::Relaxed),
        route_drift_alert_total: POLICY_METRICS
            .route_drift_alert_total
            .load(Ordering::Relaxed),
        route_drift_failsafe_activation_total: POLICY_METRICS
            .route_drift_failsafe_activation_total
            .load(Ordering::Relaxed),
        route_failsafe_active_turn_total: POLICY_METRICS
            .route_failsafe_active_turn_total
            .load(Ordering::Relaxed),
        tokens_failed_tasks_total: POLICY_METRICS
            .tokens_failed_tasks_total
            .load(Ordering::Relaxed),
        est_input_token_samples: POLICY_METRICS
            .est_input_token_samples
            .load(Ordering::Relaxed),
        est_input_tokens_total: POLICY_METRICS
            .est_input_tokens_total
            .load(Ordering::Relaxed),
        est_msg_tokens_total: POLICY_METRICS.est_msg_tokens_total.load(Ordering::Relaxed),
        est_tool_tokens_total: POLICY_METRICS.est_tool_tokens_total.load(Ordering::Relaxed),
        est_tool_tokens_high_share_total: POLICY_METRICS
            .est_tool_tokens_high_share_total
            .load(Ordering::Relaxed),
        est_tool_tokens_high_abs_total: POLICY_METRICS
            .est_tool_tokens_high_abs_total
            .load(Ordering::Relaxed),
        no_progress_iterations_total: POLICY_METRICS
            .no_progress_iterations_total
            .load(Ordering::Relaxed),
        deferred_no_tool_forced_required_total: POLICY_METRICS
            .deferred_no_tool_forced_required_total
            .load(Ordering::Relaxed),
        deferred_no_tool_deferral_detected_total: POLICY_METRICS
            .deferred_no_tool_deferral_detected_total
            .load(Ordering::Relaxed),
        deferred_no_tool_model_switch_total: POLICY_METRICS
            .deferred_no_tool_model_switch_total
            .load(Ordering::Relaxed),
        deferred_no_tool_error_marker_total: POLICY_METRICS
            .deferred_no_tool_error_marker_total
            .load(Ordering::Relaxed),
        llm_payload_invalid_total: POLICY_METRICS
            .llm_payload_invalid_total
            .load(Ordering::Relaxed),
        llm_payload_invalid_breakdown: llm_payload_invalid_breakdown_snapshot(),
    }
}

pub(super) fn record_failed_task_tokens(tokens_used: u64) {
    POLICY_METRICS
        .tokens_failed_tasks_total
        .fetch_add(tokens_used, Ordering::Relaxed);
}

pub fn init_policy_tunables_once(base_uncertainty_threshold: f32) {
    if POLICY_TUNABLES
        .initialized
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        let bp = (base_uncertainty_threshold.clamp(0.0, 1.0) * 10_000.0) as u64;
        POLICY_TUNABLES
            .uncertainty_threshold_bp
            .store(bp, Ordering::SeqCst);
    }
}

fn current_uncertainty_threshold(default_threshold: f32) -> f32 {
    if POLICY_TUNABLES.initialized.load(Ordering::SeqCst) {
        let bp = POLICY_TUNABLES
            .uncertainty_threshold_bp
            .load(Ordering::SeqCst);
        (bp as f32 / 10_000.0).clamp(0.0, 1.0)
    } else {
        default_threshold
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PolicyAutotuneSnapshot {
    pub uncertainty_threshold: f32,
}

pub fn policy_autotune_snapshot(default_threshold: f32) -> PolicyAutotuneSnapshot {
    PolicyAutotuneSnapshot {
        uncertainty_threshold: current_uncertainty_threshold(default_threshold),
    }
}

/// Bounded auto-tuning: adjust uncertainty threshold within safe bounds.
/// Returns (old, new) when a change is applied.
pub fn apply_bounded_autotune_from_failure_ratio(
    failure_ratio: f64,
    enforce: bool,
) -> Option<(f32, f32)> {
    if !enforce {
        return None;
    }
    let old_bp = POLICY_TUNABLES
        .uncertainty_threshold_bp
        .load(Ordering::SeqCst);
    let old = old_bp as f32 / 10_000.0;
    let mut next = old;
    // High failure ratio -> tighten policy (ask clarification earlier).
    if failure_ratio >= 0.25 {
        next = (next - 0.02).max(0.45);
    // Low failure ratio -> relax slightly.
    } else if failure_ratio <= 0.05 {
        next = (next + 0.01).min(0.75);
    }
    if (next - old).abs() < f32::EPSILON {
        return None;
    }
    let next_bp = (next * 10_000.0) as u64;
    POLICY_TUNABLES
        .uncertainty_threshold_bp
        .store(next_bp, Ordering::SeqCst);
    Some((old, next))
}

/// Best-effort send — never blocks the agent loop if the receiver is slow/full.
pub fn send_status(tx: &Option<mpsc::Sender<StatusUpdate>>, update: StatusUpdate) {
    if let Some(ref tx) = tx {
        let _ = tx.try_send(update);
    }
}

/// Update the heartbeat timestamp to signal the agent is alive.
/// No-op when heartbeat is None (sub-agents, triggers, tests).
pub fn touch_heartbeat(hb: &Option<Arc<AtomicU64>>) {
    if let Some(ref hb) = hb {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        hb.store(now, Ordering::Relaxed);
    }
}

/// Extract a brief human-readable summary from tool arguments JSON.
/// Helper to truncate a string and append "..." if it exceeds `max` chars.
fn truncate_summary(s: &str, max: usize) -> String {
    let truncated: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

/// Helper to extract the last path component (file/dir name) for compact display.
fn short_path(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn summarize_tool_args(name: &str, arguments: &str) -> String {
    let val: Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };

    // Helper closure to get a string field from the JSON args.
    let get_str = |key: &str| val.get(key).and_then(|v| v.as_str());

    match name {
        // --- Command execution ---
        "terminal" | "run_command" => get_str("command")
            .map(|cmd| format!("`{}`", truncate_summary(cmd, 60)))
            .unwrap_or_default(),

        // --- File operations ---
        "read_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "write_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "edit_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "search_files" => {
            let pattern = get_str("pattern").or_else(|| get_str("glob")).unwrap_or("");
            if pattern.is_empty() {
                String::new()
            } else {
                truncate_summary(pattern, 40)
            }
        }
        "list_files" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),

        // --- Web & network ---
        "web_search" => get_str("query")
            .map(|q| truncate_summary(q, 50))
            .unwrap_or_default(),
        "web_fetch" => get_str("url")
            .map(|u| truncate_summary(u, 60))
            .unwrap_or_default(),
        "http_request" => {
            let method = get_str("method").unwrap_or("GET");
            let url = get_str("url").unwrap_or("");
            if url.is_empty() {
                method.to_string()
            } else {
                format!("{} {}", method, truncate_summary(url, 50))
            }
        }

        // --- Browser ---
        "browser" => {
            let action = get_str("action").unwrap_or("");
            let url = get_str("url").unwrap_or("");
            if !url.is_empty() {
                format!("{} {}", action, truncate_summary(url, 50))
            } else {
                action.to_string()
            }
        }

        // --- Git ---
        "git_info" => {
            let include = val.get("include").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            });
            include.unwrap_or_default()
        }
        "git_commit" => get_str("message")
            .map(|m| truncate_summary(m, 40))
            .unwrap_or_default(),

        // --- Memory ---
        "remember_fact" => {
            let fact = get_str("fact").or_else(|| get_str("value")).unwrap_or("");
            if fact.is_empty() {
                "saving to memory".to_string()
            } else {
                truncate_summary(fact, 40)
            }
        }
        "manage_memories" => get_str("action").unwrap_or("").to_string(),

        // --- Skills ---
        "use_skill" => get_str("skill_name").unwrap_or("").to_string(),
        "manage_skills" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }

        // --- People ---
        "manage_people" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }

        // --- Agents ---
        "spawn_agent" => get_str("mission")
            .map(|m| truncate_summary(m, 50))
            .unwrap_or_default(),
        "cli_agent" => {
            let action = get_str("action").unwrap_or("run");
            if action != "run" {
                return format!("action={}", action);
            }
            let tool = get_str("tool").unwrap_or("auto");
            let prompt = get_str("prompt")
                .or_else(|| get_str("task"))
                .or_else(|| get_str("mission"))
                .or_else(|| get_str("description"))
                .or_else(|| get_str("command"))
                .unwrap_or("");
            let task_desc = truncate_summary(prompt, 50);
            if task_desc.is_empty() {
                format!("→ {}", tool)
            } else {
                format!("→ {}: {}", tool, task_desc)
            }
        }
        "manage_cli_agents" => get_str("action").unwrap_or("").to_string(),

        // --- Config / diagnostic ---
        "manage_config" => get_str("action").unwrap_or("").to_string(),
        "manage_mcp" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }
        "project_inspect" => {
            if let Some(path) = get_str("path") {
                short_path(path).to_string()
            } else if let Some(paths) = val.get("paths").and_then(|v| v.as_array()) {
                let mut summarized: Vec<String> = paths
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(short_path)
                    .map(str::to_string)
                    .take(3)
                    .collect();
                if summarized.is_empty() {
                    String::new()
                } else {
                    let total = paths.iter().filter_map(|v| v.as_str()).count();
                    if total > summarized.len() {
                        summarized.push(format!("+{} more", total - summarized.len()));
                    }
                    summarized.join(", ")
                }
            } else {
                String::new()
            }
        }

        // --- Channel operations ---
        "read_channel_history" => {
            let channel = get_str("channel_id").unwrap_or("");
            if channel.is_empty() {
                String::new()
            } else {
                truncate_summary(channel, 30)
            }
        }
        "send_file" => get_str("file_path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),

        // --- MCP tools: extract a human-readable name from the prefix ---
        _ if name.starts_with("mcp__") => {
            // mcp__chrome-devtools__take_screenshot → chrome-devtools: take_screenshot
            let without_prefix = &name[5..]; // skip "mcp__"
            if let Some(idx) = without_prefix.find("__") {
                let server = &without_prefix[..idx];
                let tool = &without_prefix[idx + 2..];
                // For common tools, add key arg info
                let arg_info = match tool {
                    "navigate_page" => get_str("url")
                        .map(|u| format!(" {}", truncate_summary(u, 40)))
                        .unwrap_or_default(),
                    "click" | "hover" | "fill" => get_str("uid")
                        .map(|u| format!(" #{}", u))
                        .unwrap_or_default(),
                    "evaluate_script" => get_str("function")
                        .map(|f| format!(" {}", truncate_summary(f, 30)))
                        .unwrap_or_default(),
                    _ => String::new(),
                };
                format!("{}: {}{}", server, tool.replace('_', " "), arg_info)
            } else {
                without_prefix.replace('_', " ")
            }
        }

        _ => String::new(),
    }
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct IntentGateDecision {
    can_answer_now: Option<bool>,
    needs_tools: Option<bool>,
    needs_clarification: Option<bool>,
    clarifying_question: Option<String>,
    missing_info: Vec<String>,
    complexity: Option<String>,
    /// LLM-classified: true when user explicitly asks to cancel active work.
    cancel_intent: Option<bool>,
    /// LLM-classified cancel scope: "generic" (broad cancel) or
    /// "targeted" (specific goal/task).
    cancel_scope: Option<String>,
    /// LLM-classified: true when the user's message is a pure conversational
    /// acknowledgment with no embedded request (works across all languages).
    is_acknowledgment: Option<bool>,
    schedule: Option<String>,
    schedule_type: Option<String>,
    schedule_cron: Option<String>,
    domains: Vec<String>,
}

#[derive(Debug, Clone)]
struct ResumeCheckpoint {
    task_id: String,
    description: String,
    original_user_message: Option<String>,
    elapsed_secs: u64,
    last_iteration: u32,
    tool_results_count: u32,
    pending_tool_call_ids: Vec<String>,
    last_assistant_summary: Option<String>,
    last_tool_summary: Option<String>,
    last_error: Option<String>,
    execution_snapshot: Option<ResumeExecutionSnapshot>,
}

#[derive(Debug, Clone)]
struct ResumeExecutionSnapshot {
    execution_id: String,
    current_step_id: Option<String>,
    current_tool: Option<String>,
    current_target: Option<String>,
    last_outcome: Option<StepExecutionOutcome>,
    background_handoff_active: bool,
    idempotency_key: Option<String>,
}

impl ResumeCheckpoint {
    fn render_prompt_section(&self) -> String {
        let mut lines = vec![
            "## Resume Checkpoint".to_string(),
            "The user explicitly asked to continue prior in-progress work. Resume from this checkpoint and avoid restarting completed steps."
                .to_string(),
            format!("- Previous task_id: {}", self.task_id),
            format!("- Original task: {}", self.description),
            format!("- Elapsed before interruption: {}s", self.elapsed_secs),
            format!("- Last completed iteration: {}", self.last_iteration),
            format!("- Completed tool results: {}", self.tool_results_count),
            format!(
                "- Pending unresolved tool calls: {}",
                self.pending_tool_call_ids.len()
            ),
        ];

        if !self.pending_tool_call_ids.is_empty() {
            let pending = self
                .pending_tool_call_ids
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("- Pending tool call IDs: {}", pending));
        }
        if let Some(msg) = &self.original_user_message {
            lines.push(format!(
                "- Original user request: {}",
                truncate_for_resume(msg, 180)
            ));
        }
        if let Some(summary) = &self.last_assistant_summary {
            lines.push(format!("- Last assistant output: {}", summary));
        }
        if let Some(summary) = &self.last_tool_summary {
            lines.push(format!("- Last tool result: {}", summary));
        }
        if let Some(err) = &self.last_error {
            lines.push(format!("- Last error: {}", err));
        }
        if let Some(snapshot) = &self.execution_snapshot {
            lines.push(format!("- Execution id: {}", snapshot.execution_id));
            if let Some(step_id) = &snapshot.current_step_id {
                lines.push(format!("- Last execution step: {}", step_id));
            }
            if let Some(tool) = &snapshot.current_tool {
                lines.push(format!("- Last execution tool: {}", tool));
            }
            if let Some(target) = &snapshot.current_target {
                lines.push(format!("- Last execution target: {}", target));
            }
            if let Some(outcome) = snapshot.last_outcome {
                lines.push(format!("- Last execution outcome: {:?}", outcome));
            }
            if snapshot.background_handoff_active {
                lines.push("- Background execution was active before interruption.".to_string());
            }
            if let Some(key) = &snapshot.idempotency_key {
                lines.push(format!(
                    "- Replay/idempotency key: {}",
                    truncate_for_resume(key, 120)
                ));
            }
        }
        lines.push(
            "Resume from the next concrete step immediately. Re-run tools only if needed to verify or recover."
                .to_string(),
        );
        lines.join("\n")
    }
}

fn truncate_for_resume(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut out = String::new();
    for (count, ch) in text.chars().enumerate() {
        if count >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

fn build_empty_response_fallback(response_note: Option<&str>) -> String {
    let base = "I wasn't able to process that request.";
    let generic = format!("{base} Could you try rephrasing?");
    let Some(note) = response_note.map(str::trim).filter(|s| !s.is_empty()) else {
        return generic;
    };

    let flattened = note.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = flattened.trim_matches(|c: char| c == '"' || c == '\'');
    let trimmed = trimmed.trim_end_matches(['.', '!', '?']);
    if trimmed.is_empty() {
        return generic;
    }

    let note_preview = truncate_for_resume(trimmed, 180);
    format!(
        "{base} The model returned no usable output ({note_preview}). Could you try rephrasing?"
    )
}

fn normalize_for_resume_intent(text: &str) -> String {
    text.split_whitespace()
        .map(|part| part.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_resume_request(text: &str) -> bool {
    let normalized = normalize_for_resume_intent(text);
    if normalized.is_empty() {
        return false;
    }

    const EXACT: &[&str] = &[
        "continue",
        "resume",
        "keep going",
        "go on",
        "carry on",
        "next phase",
        "next step",
    ];
    if EXACT.contains(&normalized.as_str()) {
        return true;
    }

    normalized.starts_with("continue ")
        || normalized.starts_with("resume ")
        || normalized.starts_with("keep going ")
        || normalized.starts_with("go on ")
        || normalized.starts_with("carry on ")
        || normalized.starts_with("next phase ")
        || normalized.starts_with("next step ")
}

fn user_text_references_filesystem_path(user_text: &str) -> bool {
    // Conservative: only treat as a filesystem reference when there's strong evidence the user is
    // pointing at a local path or a concrete filename.
    //
    // This intentionally avoids broad `text.contains('/')` checks which misfire on fractions/dates
    // (e.g. "3/4", "2/14") and common shorthand like "yes/no" or "w/o".
    if user_text.trim().is_empty() {
        return false;
    }

    const NON_PATH_SLASH_PHRASES: &[&str] = &["yes/no", "no/yes", "and/or", "w/o", "on/off"];
    const FILE_EXTS: &[&str] = &[
        "rs", "py", "js", "ts", "tsx", "json", "toml", "yaml", "yml", "md", "txt", "log", "env",
        "sql", "csv", "go", "java", "c", "cc", "cpp", "h", "hpp", "sh", "zsh", "bash",
    ];
    const COMMON_RELATIVE_DIRS: &[&str] = &[
        "src", "tests", "test", "target", "crates", "apps", "packages", "scripts", "bin", "lib",
        "include", "cmd", "internal", "docs",
    ];

    for raw in user_text.split_whitespace() {
        let token = raw.trim_matches(|c: char| c.is_ascii_punctuation());
        if token.is_empty() {
            continue;
        }
        let lower = token.to_ascii_lowercase();

        // Obvious URLs are not filesystem paths.
        if lower.contains("://") {
            continue;
        }

        // Windows / UNC
        if lower.starts_with("\\\\") {
            return true;
        }
        if lower.len() >= 3 {
            let bytes = lower.as_bytes();
            let drive = bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
            let sep = bytes[2] == b'\\' || bytes[2] == b'/';
            if drive && sep {
                return true;
            }
        }
        if lower.contains('\\') {
            return true;
        }

        // Unix-ish absolute / relative anchors
        if lower.starts_with("~/") || lower.starts_with("./") || lower.starts_with("../") {
            return true;
        }
        if lower.starts_with('/') {
            return true;
        }

        // Concrete filenames (no slashes required)
        if let Some((_, ext)) = lower.rsplit_once('.') {
            if FILE_EXTS.contains(&ext) {
                return true;
            }
        }

        if !lower.contains('/') {
            continue;
        }

        // Avoid false positives: fractions/dates and a few common slash phrases.
        if NON_PATH_SLASH_PHRASES.contains(&lower.as_str()) {
            continue;
        }
        let is_simple_fraction_or_date = {
            let parts: Vec<&str> = lower.split('/').collect();
            (parts.len() == 2 || parts.len() == 3)
                && parts
                    .iter()
                    .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
        };
        if is_simple_fraction_or_date {
            continue;
        }

        // Multi-segment paths are strong evidence.
        if lower.matches('/').count() >= 2 {
            return true;
        }

        // One slash: treat as a path only for common repo directories, or when the token contains a dot.
        if lower.contains('.') {
            return true;
        }
        if let Some((first, _rest)) = lower.split_once('/') {
            if COMMON_RELATIVE_DIRS.contains(&first) {
                return true;
            }
        }
    }

    false
}

fn text_contains_any_phrase_as_words(text: &str, phrases: &[&str]) -> bool {
    phrases
        .iter()
        .any(|phrase| contains_keyword_as_words(text, phrase))
}

fn text_has_explicit_project_scope_cues(text: &str) -> bool {
    text_contains_any_phrase_as_words(
        text,
        &[
            "project",
            "repo",
            "repository",
            "workspace",
            "directory",
            "folder",
            "codebase",
            "code base",
            "work in",
            "inside",
            "under",
        ],
    )
}

fn text_has_local_project_command_cues(text: &str, token: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() || lower.ends_with('?') {
        return false;
    }

    let words: Vec<&str> = lower.split_whitespace().collect();
    let strong_local_verbs = [
        "run", "build", "deploy", "publish", "restart", "reload", "commit", "push", "lint",
        "format", "fmt", "compile", "test", "debug", "fix", "refactor", "edit",
    ];
    // Allow short adverbial prefixes ("now", "also", "please", "just", "quickly") before the verb
    // so that "Now deploy blog.aidaemon.ai" is treated the same as "Deploy blog.aidaemon.ai".
    const COMMAND_PREFIXES: &[&str] = &["now", "also", "please", "just", "quickly", "go"];
    let starts_like_local_command = words
        .iter()
        .take(2)
        .map(|word| word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_'))
        .enumerate()
        .any(|(i, word)| {
            strong_local_verbs
                .iter()
                .any(|verb| word.eq_ignore_ascii_case(verb))
                && (i == 0 || words.first().is_some_and(|w| COMMAND_PREFIXES.contains(w)))
        });
    if !starts_like_local_command {
        return false;
    }

    let normalized_token = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.');
    !normalized_token.is_empty() && contains_keyword_as_words(&lower, normalized_token)
}

fn should_allow_contextual_project_nickname_scope(text: &str, token: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    user_text_references_filesystem_path(text)
        || text_has_explicit_project_scope_cues(&lower)
        || text_has_local_project_command_cues(text, token)
}

fn user_explicitly_requests_local_file_inspection(user_text: &str) -> bool {
    if user_text_references_filesystem_path(user_text) {
        return true;
    }

    let lower = user_text.to_ascii_lowercase();
    let mentions_local_subject = [
        "file",
        "files",
        "repo",
        "repository",
        "codebase",
        "directory",
        "folder",
        "workspace",
        "local file",
        "local files",
        "current repo",
        "this repo",
        "the repo",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));
    let mentions_inspection_verb = [
        "read", "open", "inspect", "look in", "look at", "search", "scan", "check", "review",
        "show", "list", "find", "grep", "compare",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    mentions_local_subject && mentions_inspection_verb
}

fn matched_untrusted_external_reference_skill_names(
    skills_snapshot: &[skills::Skill],
    user_text: &str,
    user_role: UserRole,
    visibility: ChannelVisibility,
) -> Vec<String> {
    skills::match_skills(skills_snapshot, user_text, user_role, visibility)
        .skills
        .into_iter()
        .filter(|skill| skills::is_untrusted_external_reference_skill(skill))
        .map(|skill| skill.name.clone())
        .collect()
}

fn is_untrusted_external_reference_blocked_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file"
            | "search_files"
            | "project_inspect"
            | "check_environment"
            | "web_fetch"
            | "web_search"
            | "browser"
            | "send_file"
            | "skill_resources"
    )
}

fn filter_tool_defs_for_untrusted_external_reference(defs: &[Value]) -> Vec<Value> {
    defs.iter()
        .filter(|def| {
            let name = def
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str());
            !name.is_some_and(is_untrusted_external_reference_blocked_tool)
        })
        .cloned()
        .collect()
}

#[allow(dead_code)]
fn merge_intent_gate_decision(
    model_decision: Option<IntentGateDecision>,
    inferred: IntentGateDecision,
) -> IntentGateDecision {
    let Some(model) = model_decision else {
        return inferred;
    };
    IntentGateDecision {
        can_answer_now: model.can_answer_now.or(inferred.can_answer_now),
        needs_tools: model.needs_tools.or(inferred.needs_tools),
        needs_clarification: model.needs_clarification.or(inferred.needs_clarification),
        clarifying_question: model.clarifying_question.or(inferred.clarifying_question),
        missing_info: if model.missing_info.is_empty() {
            inferred.missing_info
        } else {
            model.missing_info
        },
        complexity: model.complexity.or(inferred.complexity),
        cancel_intent: model.cancel_intent.or(inferred.cancel_intent),
        cancel_scope: model.cancel_scope.or(inferred.cancel_scope),
        is_acknowledgment: model.is_acknowledgment.or(inferred.is_acknowledgment),
        schedule: model.schedule.or(inferred.schedule),
        schedule_type: model.schedule_type.or(inferred.schedule_type),
        schedule_cron: model.schedule_cron.or(inferred.schedule_cron),
        domains: if model.domains.is_empty() {
            inferred.domains
        } else {
            model.domains
        },
    }
}

pub struct Agent {
    llm_runtime: SharedLlmRuntime,
    state: Arc<dyn StateStore>,
    event_store: Arc<EventStore>,
    tools: Vec<Arc<dyn Tool>>,
    model: RwLock<String>,
    fallback_model: RwLock<String>,
    system_prompt: String,
    config_path: PathBuf,
    skills_dir: PathBuf,
    skill_cache: skills::SkillCache,
    /// Current recursion depth (0 = root agent).
    depth: usize,
    /// Maximum allowed recursion depth for sub-agent spawning.
    max_depth: usize,
    /// Iteration limit configuration (unlimited, soft, or hard limits).
    iteration_config: IterationLimitConfig,
    /// Legacy: Maximum agentic loop iterations per invocation (for backward compat).
    #[allow(dead_code)]
    max_iterations: usize,
    /// Legacy: Hard cap on iterations (for backward compat).
    #[allow(dead_code)]
    max_iterations_cap: usize,
    /// Max chars for sub-agent response truncation.
    max_response_chars: usize,
    /// Timeout in seconds for sub-agent execution.
    timeout_secs: u64,
    /// Maximum number of facts to inject into the system prompt.
    max_facts: usize,
    /// When true, the user has manually set a model via /model — skip auto-routing.
    model_override: RwLock<bool>,
    /// Optional daily token budget — rejects LLM calls when exceeded.
    daily_token_budget: Option<u64>,
    /// Per-LLM-call timeout (watchdog). None disables the timeout.
    llm_call_timeout: Option<Duration>,
    /// Optional task timeout - maximum time per task.
    task_timeout: Option<Duration>,
    /// Optional token budget per task.
    task_token_budget: Option<u64>,
    /// Path verification tracker — gates file-modifying commands on unverified paths.
    /// None for sub-agents (they inherit parent context).
    verification_tracker: Option<Arc<VerificationTracker>>,
    /// Optional MCP server registry for dynamic, context-aware MCP tool injection.
    mcp_registry: Option<McpRegistry>,
    /// Role for this agent instance.
    role: AgentRole,
    /// Task ID for executor agents — enables activity logging.
    task_id: Option<String>,
    /// Goal ID for task lead agents — enables context injection into spawn calls.
    goal_id: Option<String>,
    /// Cancellation token — checked each iteration; cancelled by parent or user.
    cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// Goal cancellation token registry — shared across agent hierarchy.
    goal_token_registry: Option<GoalTokenRegistry>,
    /// Weak reference to the ChannelHub for background notifications.
    /// Uses RwLock because hub is created after Agent (core.rs ordering).
    hub: RwLock<Option<Weak<ChannelHub>>>,
    /// Session IDs that have granted schedule confirmation for this process lifetime.
    /// Allows schedule creation to auto-confirm after an explicit AllowSession/AllowAlways.
    schedule_approved_sessions: Arc<tokio::sync::RwLock<HashSet<String>>>,
    /// Weak self-reference for background task spawning.
    /// Set after Arc creation via `set_self_ref()`.
    self_ref: RwLock<Option<Weak<Agent>>>,
    /// Context window management configuration.
    context_window_config: crate::config::ContextWindowConfig,
    /// Policy rollout and enforcement configuration.
    policy_config: PolicyConfig,
    /// Configured path alias roots (for example, `projects/...`).
    path_aliases: PathAliasConfig,
    /// Parent scope carried into spawned child agents.
    inherited_project_scope: Option<String>,
    /// Full tool list from the root agent — used by TaskLead when spawning
    /// Executor children so they can access Action tools that were filtered
    /// out of the TaskLead's own `tools` vec.
    root_tools: Option<Vec<Arc<dyn Tool>>>,
    /// Emit structured decision points into the event store for self-diagnostics.
    record_decision_points: bool,
    /// Test-only override for the execution budget selected at the start of
    /// the agent loop. When `Some`, `select_initial_execution_budget` is
    /// bypassed and this budget is used instead.
    #[cfg(test)]
    execution_budget_override: Option<ExecutionBudget>,
}

/// Blocked path patterns for auto-sent files (mirrors SendFileTool).
const AUTO_SEND_BLOCKED_PATTERNS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".env",
    "credentials",
    ".key",
    ".pem",
    ".aws/credentials",
    ".netrc",
    ".docker/config.json",
    "config.toml",
];

/// Extract absolute file paths from text (e.g. goal completion messages).
/// Only returns paths that exist on disk and aren't security-sensitive.
pub(crate) fn extract_file_paths_from_text(text: &str) -> Vec<String> {
    let re = regex::Regex::new(r"(/[\w./-]+\.\w{1,10})").unwrap();
    let mut paths = Vec::new();
    for cap in re.captures_iter(text) {
        let path_str = &cap[1];
        let path = std::path::Path::new(path_str);

        // Must exist and be a regular file
        if !path.exists() || !path.is_file() {
            continue;
        }

        // Check against blocked patterns
        let path_display = path.to_string_lossy();
        let blocked = AUTO_SEND_BLOCKED_PATTERNS.iter().any(|pattern| {
            if pattern.starts_with('.') || pattern.starts_with('/') {
                path_display.contains(&format!("/{}", pattern))
                    || path_display.contains(&format!("/{}/", pattern))
            } else {
                path.file_name()
                    .map(|n| n.to_string_lossy() == *pattern)
                    .unwrap_or(false)
                    || path_display.contains(&format!("/{}", pattern))
                    || path_display.contains(&format!("/{}/", pattern))
            }
        });
        if blocked {
            continue;
        }

        // Also block .key and .pem extensions
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy();
            if ext == "key" || ext == "pem" {
                continue;
            }
        }

        paths.push(path_str.to_string());
    }
    paths
}

/// Parse simple wait instructions like "Wait for 5 minutes." into seconds.
/// Returns None when the task is not a plain wait command.
fn parse_wait_task_seconds(task_description: &str) -> Option<u64> {
    static WAIT_TASK_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)^\s*wait\s+for\s+(\d+)\s*(seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|h)\b",
        )
        .expect("wait task regex should compile")
    });

    let caps = WAIT_TASK_RE.captures(task_description.trim())?;
    let value: u64 = caps.get(1)?.as_str().parse().ok()?;
    let unit = caps.get(2)?.as_str().to_ascii_lowercase();

    match unit.as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(value),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(value.saturating_mul(60)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Some(value.saturating_mul(3600)),
        _ => None,
    }
}

/// Parse a leading wait/delay prefix from a goal description.
/// Handles patterns like:
///   - "wait for 2 minutes then check disk space"
///   - "in 5 minutes check disk space"
///   - "after 30 seconds run df -h"
///   - "wait 2 minutes" (pure wait, no remainder)
///
/// Returns the wait duration in seconds, or None if no wait prefix is found.
fn parse_goal_leading_wait(description: &str) -> Option<u64> {
    static LEADING_WAIT_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)^\s*(?:wait\s+(?:for\s+)?|in\s+|after\s+)(\d+)\s*(seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|h)\b",
        )
        .expect("leading wait regex should compile")
    });

    let caps = LEADING_WAIT_RE.captures(description.trim())?;
    let value: u64 = caps.get(1)?.as_str().parse().ok()?;
    let unit = caps.get(2)?.as_str().to_ascii_lowercase();

    match unit.as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(value),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(value.saturating_mul(60)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Some(value.saturating_mul(3600)),
        _ => None,
    }
}

/// Strip the leading wait/delay prefix from a goal description, returning
/// the remainder (the actual work to do after the wait).
/// Returns empty string if there's nothing meaningful after the wait.
fn strip_leading_wait(description: &str) -> String {
    static STRIP_WAIT_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)^\s*(?:wait\s+(?:for\s+)?|in\s+|after\s+)\d+\s*(?:seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|h)\s*[,;]?\s*(?:then\s+|and\s+|,\s*)?",
        )
        .expect("strip wait regex should compile")
    });

    let remainder = STRIP_WAIT_RE.replace(description.trim(), "").to_string();
    let trimmed = remainder.trim().to_string();
    // If what's left is too short to be a real instruction, treat as pure wait
    if trimmed.len() < 3 {
        String::new()
    } else {
        trimmed
    }
}

/// Check whether a session ID corresponds to a group/public channel (not a DM).
/// Used to suppress noisy progress updates in shared channels.
pub fn is_group_session(session_id: &str) -> bool {
    crate::session::is_group_session(session_id)
}

fn is_scheduled_task_description(text: &str) -> bool {
    let trimmed = text.trim_start().to_ascii_lowercase();
    trimmed.starts_with("execute scheduled goal:")
        || trimmed.starts_with("scheduled check:")
        || trimmed.starts_with("manual scheduled run:")
}

fn user_facing_task_description(description: &str) -> String {
    static SCHEDULED_TASK_PREFIX_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?i)^\s*(?:execute scheduled goal:|scheduled check:|manual scheduled run:)\s*")
            .expect("scheduled task prefix regex should compile")
    });
    static SCHEDULED_SYSTEM_SUFFIX_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?i)\s*\[system:[^\]]*\]\s*$")
            .expect("scheduled task suffix regex should compile")
    });

    let mut cleaned = description.trim().to_string();
    if is_scheduled_task_description(&cleaned) {
        cleaned = SCHEDULED_SYSTEM_SUFFIX_RE
            .replace(&cleaned, "")
            .trim()
            .to_string();
        cleaned = SCHEDULED_TASK_PREFIX_RE
            .replace(&cleaned, "")
            .trim()
            .to_string();
    }

    let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(&cleaned);
    let collapsed = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        "current task".to_string()
    } else {
        collapsed
    }
}

async fn task_has_scheduled_provenance(state: &Arc<dyn StateStore>, task_id: Option<&str>) -> bool {
    if let Some(tid) = task_id {
        if let Ok(Some(task)) = state.get_task(tid).await {
            return is_scheduled_task_description(&task.description);
        }
    }

    false
}

async fn active_scheduled_root_task_id(
    state: &Arc<dyn StateStore>,
    goal_id: &str,
) -> Option<String> {
    let tasks = state.get_tasks_for_goal(goal_id).await.ok()?;
    tasks
        .into_iter()
        .filter(|task| is_scheduled_task_description(&task.description))
        .filter(|task| {
            !matches!(
                task.status.as_str(),
                "completed" | "failed" | "cancelled" | "skipped"
            )
        })
        .max_by(|a, b| a.created_at.cmp(&b.created_at))
        .map(|task| task.id)
}

async fn goal_has_scheduled_provenance(
    state: &Arc<dyn StateStore>,
    goal_id: &str,
    task_id: Option<&str>,
) -> bool {
    if task_has_scheduled_provenance(state, task_id).await {
        return true;
    }

    if let Ok(schedules) = state.get_schedules_for_goal(goal_id).await {
        if !schedules.is_empty() {
            return true;
        }
    }

    if let Ok(tasks) = state.get_tasks_for_goal(goal_id).await {
        if tasks
            .iter()
            .any(|task| is_scheduled_task_description(&task.description))
        {
            return true;
        }
    }

    false
}

async fn persist_scheduled_run_state(
    state: &Arc<dyn StateStore>,
    goal_id: &str,
    root_task_id_hint: Option<&str>,
    status: &GoalRunBudgetStatus,
) {
    let existing = state.get_scheduled_run_state(goal_id).await.ok().flatten();
    let existing_created_at = existing.as_ref().map(|record| record.created_at.clone());
    let root_task_id = if let Some(record) = existing.as_ref() {
        Some(record.root_task_id.clone())
    } else if let Some(root_task_id) = root_task_id_hint {
        Some(root_task_id.to_string())
    } else {
        active_scheduled_root_task_id(state, goal_id).await
    };

    let Some(root_task_id) = root_task_id else {
        return;
    };

    let now = chrono::Utc::now().to_rfc3339();
    let record = ScheduledRunState {
        goal_id: goal_id.to_string(),
        root_task_id,
        effective_budget_per_check: status.effective_budget_per_check,
        tokens_used: status.tokens_used,
        budget_extensions_count: status.budget_extensions_count,
        health: status.health.clone(),
        created_at: existing_created_at.unwrap_or_else(|| now.clone()),
        updated_at: now,
    };
    let _ = state.upsert_scheduled_run_state(&record).await;
}

async fn clear_scheduled_run_state(state: &Arc<dyn StateStore>, goal_id: &str) {
    let _ = state.delete_scheduled_run_state(goal_id).await;
}

fn auto_dispatch_scheduled_run_extension_budget(
    status: &GoalRunBudgetStatus,
    max_budget_extensions: usize,
    hard_token_cap: i64,
) -> Option<i64> {
    let old_budget = status.effective_budget_per_check;
    let new_budget = old_budget
        .saturating_mul(2)
        .max(status.tokens_used.saturating_add(old_budget / 2))
        .min(hard_token_cap);

    let has_meaningful_progress = Agent::has_meaningful_budget_progress(
        status.health.evidence_gain_count,
        status.health.total_successful_tool_calls,
    );
    let clearly_unproductive =
        Agent::scheduled_run_metrics_are_clearly_unproductive(&status.health);

    if status.budget_extensions_count < max_budget_extensions
        && old_budget < hard_token_cap
        && new_budget > status.tokens_used
        && has_meaningful_progress
        && !clearly_unproductive
    {
        Some(new_budget)
    } else {
        None
    }
}

async fn effective_goal_daily_budget(
    goal: &Goal,
    registry: Option<&GoalTokenRegistry>,
) -> Option<i64> {
    let shared = if let Some(registry) = registry {
        registry.get_effective_daily_budget(&goal.id).await
    } else {
        None
    };
    shared.or(goal.budget_daily)
}

/// Detect low-signal task-lead replies that should not be sent as the
/// primary user-facing result when richer goal/task outputs are available.
fn is_low_signal_task_lead_reply(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    if trimmed == "Done." || trimmed.eq_ignore_ascii_case("goal completed successfully") {
        return true;
    }

    if trimmed.starts_with("Done — ") && !trimmed.contains('\n') {
        return true;
    }

    if trimmed.starts_with("Goal ")
        && trimmed.contains(" completed:")
        && !trimmed.contains('\n')
        && trimmed.len() <= 220
    {
        return true;
    }

    false
}

fn looks_like_incomplete_live_work_summary(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let has_attempt_structure = lower.contains("what i tried:")
        || lower.contains("current status:")
        || (contains_keyword_as_words(&lower, "i attempted to")
            && contains_keyword_as_words(&lower, "current status"));

    let has_blocked_outcome = contains_keyword_as_words(&lower, "no results retrieved yet")
        || contains_keyword_as_words(&lower, "no results found yet")
        || contains_keyword_as_words(&lower, "could not retrieve results")
        || contains_keyword_as_words(&lower, "encountered api errors")
        || contains_keyword_as_words(&lower, "bad request")
        || contains_keyword_as_words(&lower, "request is malformed")
        || contains_keyword_as_words(&lower, "request was malformed")
        || contains_keyword_as_words(&lower, "api is rejecting");

    has_attempt_structure && has_blocked_outcome
}

fn looks_like_false_capability_denial_after_tool_success(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    const DIRECT_DENIALS: &[&str] = &[
        "can't browse",
        "cannot browse",
        "can't access",
        "cannot access",
        "don't have access",
        "do not have access",
        "can't perform a live search",
        "cannot perform a live search",
        "unable to perform a live search",
        "can't search the web",
        "cannot search the web",
        "don't have real time access",
        "do not have real time access",
        "don't have real-time access",
        "do not have real-time access",
        "can't access real time information",
        "cannot access real time information",
        "can't access real-time information",
        "cannot access real-time information",
        "from my training data",
        "based on my training data",
        "from training data",
        "based on training data",
    ];

    if DIRECT_DENIALS.iter().any(|phrase| lower.contains(phrase)) {
        return true;
    }

    let guide_only = lower.contains("i can guide you on how to find")
        || lower.contains("i can guide you on how to")
        || lower.contains("here's how to find");
    let live_data_context = lower.contains("live search")
        || lower.contains("current databases")
        || lower.contains("current database")
        || lower.contains("real time information")
        || lower.contains("real-time information");

    guide_only && live_data_context
}

fn looks_like_evidence_grounding_challenge(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let direct_grounding_challenges = [
        "made them up",
        "make them up",
        "made that up",
        "make that up",
        "made this up",
        "make this up",
        "fabricated",
        "invented",
        "hallucinated",
    ];
    if direct_grounding_challenges
        .iter()
        .any(|phrase| contains_keyword_as_words(&lower, phrase))
    {
        return true;
    }

    let blocker_terms = [
        "disabled",
        "blocked",
        "stopped",
        "stop",
        "failed",
        "failure",
        "error",
        "errors",
        "text-only",
        "plain text",
        "tool mode",
        "couldn't",
        "could not",
        "unable",
    ];
    if contains_keyword_as_words(&lower, "why")
        && blocker_terms
            .iter()
            .any(|term| contains_keyword_as_words(&lower, term))
    {
        return true;
    }

    let grounding_focus = [
        "real", "really", "actually", "exact", "exactly", "quote", "quoted",
    ];
    let evidence_terms = [
        "error", "errors", "result", "results", "output", "message", "messages", "line", "lines",
        "status", "statuses", "id", "ids", "value", "values", "count", "counts", "failure",
        "failures", "file", "files", "test", "tests", "api",
    ];
    let challenge_phrases = [
        "where did you get that",
        "show the exact output",
        "show the exact result",
        "what did it actually say",
        "what did that actually say",
        "what did the tool actually say",
        "did it actually say",
        "did that actually say",
        "did it really say",
        "did that really say",
        "did it really return",
        "did that really return",
        "did it actually return",
        "did that actually return",
        "did it actually fail",
        "did that actually fail",
        "was that real",
        "were those real",
        "is that real",
        "are those real",
    ];

    challenge_phrases
        .iter()
        .any(|phrase| contains_keyword_as_words(&lower, phrase))
        || (grounding_focus
            .iter()
            .any(|word| contains_keyword_as_words(&lower, word))
            && evidence_terms
                .iter()
                .any(|term| contains_keyword_as_words(&lower, term)))
}

pub(crate) fn goal_completion_response_indicates_incomplete_work(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    goal_completion_summary_indicates_not_finished(trimmed)
        || is_low_signal_task_lead_reply(trimmed)
        || looks_like_incomplete_live_work_summary(trimmed)
        || (looks_like_deferred_action_response(trimmed)
            && !is_substantive_text_response(trimmed, 200))
}

fn truncate_goal_result_text(text: &str, max_chars: usize) -> String {
    let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(text);
    let trimmed = sanitized.trim();
    let truncated: String = trimmed.chars().take(max_chars).collect();
    if trimmed.chars().count() > max_chars {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn goal_failure_summary_from_context(goal: &Goal) -> Option<String> {
    goal.context
        .as_deref()
        .and_then(|ctx| serde_json::from_str::<serde_json::Value>(ctx).ok())
        .and_then(|ctx| {
            ctx.get("failure_summary")
                .and_then(|v| v.as_str())
                .map(ToOwned::to_owned)
        })
        .map(|summary| summary.trim().to_string())
        .filter(|summary| !summary.is_empty())
}

fn latest_problem_task_summary(tasks: &[Task]) -> Option<String> {
    tasks
        .iter()
        .filter(|task| matches!(task.status.as_str(), "failed" | "blocked"))
        .max_by(|a, b| {
            let a_key = a
                .completed_at
                .as_deref()
                .or(a.started_at.as_deref())
                .unwrap_or(a.created_at.as_str());
            let b_key = b
                .completed_at
                .as_deref()
                .or(b.started_at.as_deref())
                .unwrap_or(b.created_at.as_str());
            a_key
                .cmp(b_key)
                .then_with(|| a.task_order.cmp(&b.task_order))
                .then_with(|| a.id.cmp(&b.id))
        })
        .and_then(|task| {
            let detail = task
                .result
                .as_deref()
                .or(task.error.as_deref())
                .or(task.blocker.as_deref())
                .map(str::trim)
                .filter(|detail| !detail.is_empty())?;
            Some(format!(
                "{}: {}",
                task.description,
                truncate_goal_result_text(detail, 1000)
            ))
        })
}

pub(crate) fn build_goal_failure_summary(
    goal: Option<&Goal>,
    tasks: &[Task],
    task_lead_response: Option<&str>,
    task_lead_error: Option<&str>,
) -> String {
    let mut summary = goal
        .and_then(goal_failure_summary_from_context)
        .or_else(|| {
            task_lead_response
                .map(str::trim)
                .filter(|reply| !is_low_signal_task_lead_reply(reply))
                .filter(|reply| !reply.is_empty())
                .map(ToOwned::to_owned)
        })
        .or_else(|| latest_problem_task_summary(tasks))
        .or_else(|| {
            task_lead_error
                .map(str::trim)
                .filter(|err| !err.is_empty())
                .map(ToOwned::to_owned)
        })
        .or_else(|| goal.map(|g| g.description.clone()))
        .unwrap_or_else(|| "task lead exited without completing all tasks".to_string());

    if summary.to_ascii_lowercase().starts_with("goal failed:") {
        summary = summary["Goal failed:".len()..].trim().to_string();
    }

    truncate_goal_result_text(&summary, 3500)
}

/// Build a user-facing summary from successful task results.
/// Includes recent completed tasks (bounded) instead of only the last one.
pub(crate) fn build_goal_task_results_summary(tasks: &[Task], fallback: &str) -> String {
    const MAX_INCLUDED_TASK_RESULTS: usize = 3;
    const MAX_CHARS_PER_TASK_RESULT: usize = 800;

    let mut successful: Vec<&Task> = tasks
        .iter()
        .filter(|t| t.status == "completed" && t.error.is_none())
        .filter(|t| t.result.as_deref().is_some_and(|r| !r.trim().is_empty()))
        .collect();

    if successful.is_empty() {
        return truncate_goal_result_text(fallback, 4000);
    }

    successful.sort_by(|a, b| {
        let a_key = a.completed_at.as_deref().unwrap_or(a.created_at.as_str());
        let b_key = b.completed_at.as_deref().unwrap_or(b.created_at.as_str());
        a_key
            .cmp(b_key)
            .then_with(|| a.task_order.cmp(&b.task_order))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut selected: Vec<&Task> = successful
        .iter()
        .rev()
        .take(MAX_INCLUDED_TASK_RESULTS)
        .copied()
        .collect();
    selected.reverse();

    let sections: Vec<String> = selected
        .iter()
        .map(|t| {
            let result = t.result.as_deref().unwrap_or("");
            format!(
                "**{}**\n{}",
                t.description,
                truncate_goal_result_text(result, MAX_CHARS_PER_TASK_RESULT)
            )
        })
        .collect();

    if sections.is_empty() {
        return truncate_goal_result_text(fallback, 4000);
    }

    let omitted = successful.len().saturating_sub(selected.len());
    if sections.len() == 1 && omitted == 0 {
        return sections[0].clone();
    }

    let mut summary = format!(
        "{}/{} tasks completed.\n\n{}",
        successful.len(),
        tasks.len(),
        sections.join("\n\n")
    );
    if omitted > 0 {
        let suffix = if omitted == 1 { "" } else { "s" };
        summary.push_str(&format!(
            "\n\n(+{} earlier completed task result{} omitted)",
            omitted, suffix
        ));
    }
    summary
}

/// Spawn a task lead in the background (free function to satisfy Send requirements).
/// This runs `spawn_child` on the given agent with TaskLead role, then updates
/// the goal and notifies the user when complete.
#[allow(clippy::too_many_arguments)]
pub fn spawn_background_task_lead(
    agent: Arc<Agent>,
    goal: crate::traits::Goal,
    user_text: String,
    session_id: String,
    channel_ctx: ChannelContext,
    user_role: UserRole,
    state: Arc<dyn crate::traits::StateStore>,
    hub: Option<Weak<crate::channels::ChannelHub>>,
    goal_token_registry: Option<crate::goal_tokens::GoalTokenRegistry>,
    dispatch_trigger_task_id: Option<String>,
) {
    tokio::spawn(async move {
        let goal_id = goal.id.clone();
        let mission = goal.description.clone();
        // Clone channel_ctx and user_role for potential direct fallback and auto-dispatch
        let fallback_channel_ctx = channel_ctx.clone();
        let dispatch_channel_ctx = channel_ctx.clone();
        let fallback_user_role = user_role;

        // Heartbeat dispatch may claim a temporary "trigger" task before spawning this
        // background lead. Release that temporary claim back to `pending` so the real
        // unit of work is not skipped.
        if let Some(trigger_task_id) = dispatch_trigger_task_id {
            match state.get_task(&trigger_task_id).await {
                Ok(Some(task))
                    if (task.status == "claimed" || task.status == "running")
                        && task
                            .agent_id
                            .as_deref()
                            .is_some_and(|aid| aid.starts_with("heartbeat-dispatch-")) =>
                {
                    let mut updated = task.clone();
                    updated.status = "pending".to_string();
                    updated.agent_id = None;
                    updated.started_at = None;
                    updated.completed_at = None;
                    if let Err(e) = state.update_task(&updated).await {
                        warn!(
                            task_id = %trigger_task_id,
                            goal_id = %goal_id,
                            error = %e,
                            "Failed to release dispatch trigger task claim"
                        );
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    warn!(
                        task_id = %trigger_task_id,
                        goal_id = %goal_id,
                        error = %e,
                        "Failed to load dispatch trigger task"
                    );
                }
            }
        }

        // Prevent duplicate concurrent task leads (and duplicate heartbeats) for the same goal.
        // Multiple codepaths can attempt to dispatch work for a goal (initial spawn, heartbeat
        // orphan recovery, auto-dispatch). This in-memory guard keeps progress messages sane
        // and avoids overlapping TaskLead runs.
        let _run_guard = if let Some(ref registry) = goal_token_registry {
            match registry.try_acquire_run(&goal_id) {
                Some(g) => Some(g),
                None => {
                    info!(
                        goal_id = %goal_id,
                        session_id = %session_id,
                        "Goal already has an active task lead; skipping duplicate background spawn"
                    );
                    return;
                }
            }
        } else {
            None
        };

        // Progress heartbeat: send periodic status updates while the task lead works.
        // This prevents the "goal appears abandoned" UX problem where the user sees
        // nothing between "On it." and the final notification.
        // Only send progress updates to DM sessions — group channels already have the
        // "Running scheduled task" notification and the final result. Progress updates
        // every 30s are too noisy for shared channels.
        let is_group_channel = is_group_session(&session_id);
        let heartbeat_hub = hub.clone();
        let heartbeat_session = session_id.clone();
        let heartbeat_state = state.clone();
        let heartbeat_goal_id = goal_id.clone();
        let (heartbeat_cancel_tx, mut heartbeat_cancel_rx) = tokio::sync::oneshot::channel::<()>();
        let heartbeat_handle = tokio::spawn(async move {
            if is_group_channel {
                // In group channels, just wait for cancellation — no progress spam
                let _ = heartbeat_cancel_rx.await;
                return;
            }
            let mut interval_count = 0u32;
            let mut last_progress_key: Option<String> = None;
            let mut repeated_progress = 0u32;
            let mut planning_msg_count = 0u32;
            let mut total_progress_emitted = 0u32;
            const MAX_PROGRESS_MESSAGES: u32 = 4;
            loop {
                // First update after 15s, then every 30s
                let wait_secs = if interval_count == 0 { 15 } else { 30 };
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(wait_secs)) => {},
                    _ = &mut heartbeat_cancel_rx => break,
                }
                interval_count += 1;

                // Build progress message from task statuses
                let tasks = heartbeat_state
                    .get_tasks_for_goal(&heartbeat_goal_id)
                    .await
                    .unwrap_or_default();
                if tasks.is_empty() {
                    // Tasks not yet created — send one planning message only.
                    // Previously sent at count 1 and 5, causing repeated spam.
                    // Now: send only on the first empty-tasks heartbeat, then
                    // stay silent until tasks are actually created.
                    planning_msg_count += 1;
                    if planning_msg_count == 1 && total_progress_emitted < MAX_PROGRESS_MESSAGES {
                        total_progress_emitted += 1;
                        if let Some(hub_weak) = &heartbeat_hub {
                            if let Some(hub_arc) = hub_weak.upgrade() {
                                let _ = hub_arc
                                    .send_text(
                                        &heartbeat_session,
                                        "⏳ Still working on your request — planning the steps...",
                                    )
                                    .await;
                            }
                        }
                    }
                } else {
                    // Count genuinely completed tasks (exclude cancelled ones with errors)
                    let completed = tasks
                        .iter()
                        .filter(|t| t.status == "completed" && t.error.is_none())
                        .count();
                    let started = tasks.iter().filter(|t| t.status != "pending").count();
                    let active_count = tasks
                        .iter()
                        .filter(|t| t.status == "claimed" || t.status == "running")
                        .count();
                    let total = tasks.len();
                    let in_progress: Vec<String> = tasks
                        .iter()
                        .filter(|t| t.status == "claimed" || t.status == "running")
                        .take(2)
                        .map(|t| user_facing_task_description(&t.description))
                        .collect();
                    let progress_msg = if in_progress.is_empty() && completed == total {
                        format!("⏳ Progress: {}/{} steps completed", completed, total)
                    } else if active_count > 0 {
                        format!(
                            "⏳ Progress: {}/{} steps completed ({} in progress, {} started). Working on: {}",
                            completed,
                            total,
                            active_count,
                            started,
                            in_progress.join(", ")
                        )
                    } else if started > completed {
                        format!(
                            "⏳ Progress: {}/{} steps completed ({} started).",
                            completed, total, started
                        )
                    } else if in_progress.is_empty() {
                        "⏳ Still working on your request...".to_string()
                    } else {
                        format!(
                            "⏳ Progress: {}/{} steps completed. Working on: {}",
                            completed,
                            total,
                            in_progress.join(", ")
                        )
                    };

                    // Dedup key uses only completed|total so we don't spam when
                    // sub-tasks change status without any step actually completing.
                    let progress_key = format!("{}|{}", completed, total);
                    let should_emit = if last_progress_key.as_deref() == Some(progress_key.as_str())
                    {
                        repeated_progress = repeated_progress.saturating_add(1);
                        // Reduce spam for long-running tasks with unchanged state:
                        // emit every 4th repeat (i.e. roughly every 2 minutes).
                        repeated_progress.is_multiple_of(4)
                    } else {
                        last_progress_key = Some(progress_key);
                        repeated_progress = 0;
                        true
                    };
                    if !should_emit || total_progress_emitted >= MAX_PROGRESS_MESSAGES {
                        continue;
                    }
                    total_progress_emitted += 1;
                    if let Some(hub_weak) = &heartbeat_hub {
                        if let Some(hub_arc) = hub_weak.upgrade() {
                            let _ = hub_arc.send_text(&heartbeat_session, &progress_msg).await;
                        }
                    }
                }
            }
        });

        // Intercept pure wait/sleep goals to avoid spawning a full LLM task lead
        // just to orchestrate a timer.  For compound goals ("wait 2 minutes then
        // check disk space") we sleep first and then let the task lead handle the
        // remainder — but only if there actually IS a remainder after the wait.
        let effective_mission;
        let effective_user_text;
        if let Some(wait_secs) = parse_goal_leading_wait(&mission) {
            let remainder = strip_leading_wait(&mission);
            info!(
                goal_id = %goal_id,
                wait_secs,
                has_remainder = !remainder.is_empty(),
                "Intercepted wait prefix in goal — sleeping locally"
            );
            tokio::time::sleep(Duration::from_secs(wait_secs)).await;

            if remainder.is_empty() {
                // Pure wait goal with nothing after — mark complete, skip LLM entirely.
                let _ = heartbeat_cancel_tx.send(());
                let _ = heartbeat_handle.await;
                let now = chrono::Utc::now().to_rfc3339();
                let msg = format!("Waited for {} second(s).", wait_secs);
                if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                    if g.status == "active" || g.status == "pending" {
                        g.status = "completed".to_string();
                        g.completed_at = Some(now.clone());
                        g.updated_at = now.clone();
                        let _ = state.update_goal(&g).await;
                    }
                }

                // Finalize any non-terminal tasks so we don't leave pending rows behind
                // after a local pure-wait short-circuit.
                if let Ok(tasks) = state.get_tasks_for_goal(&goal_id).await {
                    for task in tasks {
                        if task.status != "completed"
                            && task.status != "failed"
                            && task.status != "cancelled"
                        {
                            let mut updated = task.clone();
                            updated.status = "completed".to_string();
                            updated.error = None;
                            updated.result = Some(msg.clone());
                            updated.completed_at = Some(now.clone());
                            let _ = state.update_task(&updated).await;
                        }
                    }
                }

                if let Some(hub_weak) = &hub {
                    if let Some(hub_arc) = hub_weak.upgrade() {
                        let _ = hub_arc.send_text(&session_id, &msg).await;
                    }
                }
                return;
            }
            effective_mission = remainder.clone();
            effective_user_text = remainder;
        } else {
            effective_mission = mission.clone();
            effective_user_text = user_text.clone();
        }

        let result = agent
            .spawn_child(
                &effective_mission,
                &effective_user_text,
                None,
                channel_ctx,
                fallback_user_role,
                Some(AgentRole::TaskLead),
                Some(goal_id.as_str()),
                None,
                None,
            )
            .await;

        // Keep the task-lead textual response, but defer relay until we know
        // whether the goal is terminal. For terminal goals, we prefer the
        // canonical completion summary built from task results.
        let task_lead_response = result
            .as_ref()
            .ok()
            .map(|response| response.trim().to_string())
            .filter(|response| !response.is_empty());

        // Track whether any executor results were already sent inline to the user.
        // Used to avoid duplicate content in the completion notification.
        let mut any_executor_results_sent = false;

        // Auto-dispatch: dispatch remaining pending tasks after task lead returns.
        // This handles both cases: LLMs that create tasks but don't spawn executors,
        // AND task leads that completed some tasks but left others pending.
        // Uses a loop to re-evaluate after each batch — completing a task may
        // unblock dependent tasks that weren't dispatchable in the previous pass.
        {
            let max_dispatch_rounds = 4; // safety limit — keep low to bound token usage
            const AUTO_DISPATCH_MAX_BUDGET_EXTENSIONS: usize = 12;
            const AUTO_DISPATCH_HARD_TOKEN_CAP: i64 = 20_000_000;
            let mut budget_exhausted = false;
            for _round in 0..max_dispatch_rounds {
                let all_tasks: Vec<crate::traits::Task> =
                    state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();

                // Build set of completed task IDs for dependency checking
                let completed_ids: std::collections::HashSet<String> = all_tasks
                    .iter()
                    .filter(|t| t.status == "completed" || t.status == "skipped")
                    .map(|t| t.id.clone())
                    .collect();

                // Filter to pending tasks whose dependencies are all met
                let dispatchable: Vec<crate::traits::Task> = all_tasks
                    .iter()
                    .filter(|t| t.status == "pending")
                    .filter(|t| match &t.depends_on {
                        None => true,
                        Some(deps_json) => serde_json::from_str::<Vec<String>>(deps_json)
                            .unwrap_or_default()
                            .iter()
                            .all(|dep_id| completed_ids.contains(dep_id)),
                    })
                    .cloned()
                    .collect();

                if dispatchable.is_empty() {
                    break; // No more tasks to dispatch
                }

                // Conservative fallback behavior: only dispatch the earliest
                // task_order in each round. This preserves intended sequencing
                // when a task lead created ordered tasks but omitted depends_on.
                let min_task_order = dispatchable.iter().map(|t| t.task_order).min().unwrap_or(0);
                let dispatch_batch: Vec<crate::traits::Task> = dispatchable
                    .into_iter()
                    .filter(|t| t.task_order == min_task_order)
                    .collect();

                info!(
                    goal_id = %goal_id,
                    count = dispatch_batch.len(),
                    task_order = min_task_order,
                    round = _round,
                    "Auto-dispatching pending tasks after task lead"
                );

                for task in &dispatch_batch {
                    // Stop dispatching when the active run has exhausted its
                    // shared per-run budget, or when a non-scheduled goal hits
                    // its daily budget.
                    if let Ok(Some(g)) = state.get_goal(&goal_id).await {
                        let is_scheduled =
                            goal_has_scheduled_provenance(&state, &goal_id, Some(&task.id)).await;
                        if is_scheduled {
                            let run_budget = if let Some(registry) = goal_token_registry.as_ref() {
                                registry.get_run_budget(&goal_id).await
                            } else {
                                None
                            };
                            if let Some(run_budget) = run_budget {
                                if run_budget.tokens_used >= run_budget.effective_budget_per_check {
                                    let old_budget = run_budget.effective_budget_per_check;
                                    if let Some(new_budget) =
                                        auto_dispatch_scheduled_run_extension_budget(
                                            &run_budget,
                                            AUTO_DISPATCH_MAX_BUDGET_EXTENSIONS,
                                            AUTO_DISPATCH_HARD_TOKEN_CAP,
                                        )
                                    {
                                        if let Some(registry) = goal_token_registry.as_ref() {
                                            if let Some(updated) = registry
                                                .auto_extend_run_budget(&goal_id, new_budget)
                                                .await
                                            {
                                                persist_scheduled_run_state(
                                                    &state, &goal_id, None, &updated,
                                                )
                                                .await;
                                                info!(
                                                    goal_id = %goal_id,
                                                    tokens_used = updated.tokens_used,
                                                    old_budget,
                                                    new_budget,
                                                    extension = updated.budget_extensions_count,
                                                    "Auto-extended scheduled run budget during auto-dispatch"
                                                );
                                            } else {
                                                budget_exhausted = true;
                                                info!(
                                                    goal_id = %goal_id,
                                                    tokens_used = run_budget.tokens_used,
                                                    budget = run_budget.effective_budget_per_check,
                                                    "Stopping auto-dispatch — scheduled run budget exhausted"
                                                );
                                                break;
                                            }
                                        }
                                    } else {
                                        budget_exhausted = true;
                                        info!(
                                            goal_id = %goal_id,
                                            tokens_used = run_budget.tokens_used,
                                            budget = run_budget.effective_budget_per_check,
                                            "Stopping auto-dispatch — scheduled run budget exhausted"
                                        );
                                        break;
                                    }
                                }
                            }
                        } else if let Some(budget_daily) =
                            effective_goal_daily_budget(&g, goal_token_registry.as_ref()).await
                        {
                            if g.tokens_used_today >= budget_daily {
                                budget_exhausted = true;
                                info!(
                                    goal_id = %goal_id,
                                    tokens_used = g.tokens_used_today,
                                    budget = budget_daily,
                                    "Stopping auto-dispatch — goal daily budget exhausted"
                                );
                                break;
                            }
                        }
                    }

                    // Claim the task
                    let claimed = match state
                        .claim_task(&task.id, &format!("auto-dispatch-{}", goal_id))
                        .await
                    {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    if !claimed {
                        continue;
                    }

                    // Execute pure wait tasks locally to avoid unnecessary LLM
                    // calls and provider rate-limit churn.
                    if let Some(wait_secs) = parse_wait_task_seconds(&task.description) {
                        info!(
                            goal_id = %goal_id,
                            task_id = %task.id,
                            wait_secs,
                            "Executing wait task locally"
                        );

                        // Keep the claimed task fresh so heartbeat stuck-task
                        // detection does not interrupt legitimate waits.
                        let mut remaining = wait_secs;
                        while remaining > 0 {
                            let step = remaining.min(60);
                            tokio::time::sleep(Duration::from_secs(step)).await;
                            remaining = remaining.saturating_sub(step);
                            if remaining > 0 {
                                if let Ok(Some(mut claimed_task)) = state.get_task(&task.id).await {
                                    claimed_task.started_at = Some(chrono::Utc::now().to_rfc3339());
                                    claimed_task.status = "claimed".to_string();
                                    let _ = state.update_task(&claimed_task).await;
                                }
                            }
                        }

                        if let Ok(Some(mut completed_task)) = state.get_task(&task.id).await {
                            completed_task.status = "completed".to_string();
                            completed_task.result =
                                Some(format!("Waited for {} second(s).", wait_secs));
                            completed_task.error = None;
                            completed_task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            let _ = state.update_task(&completed_task).await;
                        }
                        continue;
                    }

                    // Spawn executor
                    let exec_result = agent
                        .spawn_child(
                            &task.description,
                            &task.description,
                            None,
                            dispatch_channel_ctx.clone(),
                            fallback_user_role,
                            Some(AgentRole::Executor),
                            Some(goal_id.as_str()),
                            Some(task.id.as_str()),
                            None,
                        )
                        .await;

                    let mut latest_task = state.get_task(&task.id).await.ok().flatten();
                    match exec_result {
                        Ok(response) => {
                            let delivery_text = if !response.trim().is_empty() {
                                response.clone()
                            } else {
                                latest_task
                                    .as_ref()
                                    .and_then(|task| {
                                        task.result
                                            .clone()
                                            .filter(|result| !result.trim().is_empty())
                                            .or_else(|| {
                                                task.blocker
                                                    .clone()
                                                    .filter(|blocker| !blocker.trim().is_empty())
                                            })
                                    })
                                    .unwrap_or_default()
                            };

                            if !delivery_text.trim().is_empty() {
                                if let Some(hub_weak) = &hub {
                                    if let Some(hub_arc) = hub_weak.upgrade() {
                                        let _ =
                                            hub_arc.send_text(&session_id, &delivery_text).await;
                                        any_executor_results_sent = true;
                                    }
                                }
                            }

                            if let Some(ref mut current_task) = latest_task {
                                if current_task
                                    .result
                                    .as_deref()
                                    .is_none_or(|result| result.trim().is_empty())
                                    && !response.trim().is_empty()
                                {
                                    current_task.result = Some(response);
                                    current_task.completed_at =
                                        Some(chrono::Utc::now().to_rfc3339());
                                    if !matches!(
                                        current_task.status.as_str(),
                                        "completed" | "blocked" | "failed"
                                    ) {
                                        current_task.status = "completed".to_string();
                                        current_task.blocker = None;
                                    }
                                    let _ = state.update_task(current_task).await;
                                }
                            }
                        }
                        Err(e) => {
                            if let Some(ref mut current_task) = latest_task {
                                if !matches!(
                                    current_task.status.as_str(),
                                    "completed" | "blocked" | "failed"
                                ) {
                                    current_task.status = "failed".to_string();
                                    current_task.error = Some(e.to_string());
                                    current_task.completed_at =
                                        Some(chrono::Utc::now().to_rfc3339());
                                    let _ = state.update_task(current_task).await;
                                }
                            } else {
                                let mut updated = task.clone();
                                updated.status = "failed".to_string();
                                updated.error = Some(e.to_string());
                                let _ = state.update_task(&updated).await;
                            }
                        }
                    }
                }

                if budget_exhausted {
                    break;
                }
            }
        }

        // Stop the heartbeat
        let _ = heartbeat_cancel_tx.send(());
        let _ = heartbeat_handle.await;

        // Check the actual goal status from DB — the task lead may have already
        // set it via complete_goal/fail_goal. Only update if still "active".
        let current_goal = state.get_goal(&goal.id).await;
        let needs_status_update = match &current_goal {
            Ok(Some(g)) => g.status == "active" || g.status == "pending",
            _ => true, // fallback: update if we can't read
        };

        if needs_status_update {
            // Task lead returned without explicitly completing/failing the goal.
            // Use progress-based circuit breaker: compare completed task count
            // before vs after to detect whether the dispatch made progress.
            let completed_after = state
                .count_completed_tasks_for_goal(&goal_id)
                .await
                .unwrap_or(0);

            let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();
            let all_done = !tasks.is_empty()
                && tasks
                    .iter()
                    .all(|t| t.status == "completed" || t.status == "skipped");

            let mut updated_goal = match state.get_goal(&goal_id).await {
                Ok(Some(g)) => g,
                _ => goal,
            };

            let scheduled_goal_active = goal_has_scheduled_provenance(&state, &goal_id, None).await;
            let scheduled_run_budget_exhausted = if scheduled_goal_active {
                if let Some(registry) = goal_token_registry.as_ref() {
                    registry
                        .get_run_budget(&goal_id)
                        .await
                        .is_some_and(|status| {
                            status.tokens_used >= status.effective_budget_per_check
                        })
                } else {
                    false
                }
            } else {
                false
            };
            let effective_goal_budget =
                effective_goal_daily_budget(&updated_goal, goal_token_registry.as_ref()).await;
            let goal_budget_exhausted = !scheduled_goal_active
                && effective_goal_budget.is_some_and(|b| updated_goal.tokens_used_today >= b);

            // For finite goals: detect when no tasks were completed after
            // the task lead finished — fail immediately since there's no
            // re-dispatch mechanism for finite goals.
            let is_finite = updated_goal.goal_type == "finite";
            let any_completed = tasks.iter().any(|t| t.status == "completed");
            let no_tasks_completed_finite = is_finite && !tasks.is_empty() && !any_completed;

            if all_done {
                // All tasks finished — goal is complete
                updated_goal.status = "completed".to_string();
                updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                updated_goal.dispatch_failures = 0;
            } else if scheduled_run_budget_exhausted {
                updated_goal.dispatch_failures = 0;
                info!(
                    goal_id = %goal_id,
                    "Goal dispatch paused: scheduled run budget exhausted"
                );
            } else if goal_budget_exhausted {
                // Budget exhausted is a safety stop, not "no progress". Keep the goal active
                // and avoid stalling it; it can resume after budgets reset.
                updated_goal.dispatch_failures = 0;
                info!(
                    goal_id = %goal_id,
                    tokens_used = updated_goal.tokens_used_today,
                    budget = effective_goal_budget.unwrap_or(0),
                    "Goal dispatch paused: daily token budget exhausted"
                );
            } else if no_tasks_completed_finite {
                // Finite goal with zero completed tasks — fail fast.
                // This covers tasks stuck in any non-completed status:
                // pending, claimed, blocked, or failed. Since finite goals
                // have no re-dispatch loop, waiting is pointless.
                updated_goal.status = "failed".to_string();
                updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                let pending = tasks
                    .iter()
                    .filter(|t| t.status == "pending" || t.status == "claimed")
                    .count();
                let blocked = tasks.iter().filter(|t| t.status == "blocked").count();
                let failed = tasks.iter().filter(|t| t.status == "failed").count();
                info!(
                    goal_id = %goal_id,
                    pending,
                    blocked,
                    failed,
                    "Finite goal failed: no tasks completed after dispatch"
                );
            } else if result.is_err() {
                // Task lead crashed — count as no progress
                updated_goal.dispatch_failures += 1;
                info!(
                    goal_id = %goal_id,
                    dispatch_failures = updated_goal.dispatch_failures,
                    "Task lead errored, incrementing dispatch_failures"
                );
            } else if is_finite {
                // Finite goal with some tasks completed but others remain.
                // Since finite goals have no re-dispatch, mark as completed
                // (partial success) rather than leaving it stuck.
                let completed_count = tasks
                    .iter()
                    .filter(|t| t.status == "completed" && t.error.is_none())
                    .count();
                let failed_count = tasks.iter().filter(|t| t.status == "failed").count();
                let blocked_count = tasks.iter().filter(|t| t.status == "blocked").count();
                let remaining = tasks
                    .iter()
                    .filter(|t| t.status != "completed" && t.status != "skipped")
                    .count();
                updated_goal.status = "completed".to_string();
                updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());

                // Store completion summary in context for notification enrichment
                if failed_count > 0 || blocked_count > 0 {
                    let summary = serde_json::json!({
                        "partial_success": true,
                        "completed": completed_count,
                        "failed": failed_count,
                        "blocked": blocked_count,
                        "total": tasks.len(),
                    });
                    updated_goal.context = Some(summary.to_string());
                }
                info!(
                    goal_id = %goal_id,
                    completed_count,
                    failed_count,
                    blocked_count,
                    remaining,
                    "Finite goal partially completed after dispatch"
                );
            } else {
                // Continuous goal: task lead returned Ok but tasks remain.
                // Check if any tasks were completed recently during this dispatch.
                let recently_completed = tasks.iter().any(|t| {
                    t.status == "completed"
                        && t.completed_at.as_ref().is_some_and(|ca| {
                            chrono::DateTime::parse_from_rfc3339(ca)
                                .map(|dt| {
                                    let age = chrono::Utc::now() - dt.with_timezone(&chrono::Utc);
                                    age.num_minutes() < 30
                                })
                                .unwrap_or(false)
                        })
                });

                // Check if all remaining non-completed tasks are blocked
                // (waiting on external input/dependencies). Blocked tasks are
                // waiting, not failing — don't count as "no progress".
                let all_remaining_blocked = tasks
                    .iter()
                    .filter(|t| t.status != "completed" && t.status != "skipped")
                    .all(|t| t.status == "blocked");

                if recently_completed {
                    // Progress was made — reset failures
                    updated_goal.dispatch_failures = 0;
                } else if all_remaining_blocked && !tasks.is_empty() {
                    // All remaining tasks are blocked — don't increment failures
                    info!(
                        goal_id = %goal_id,
                        blocked_tasks = tasks.iter().filter(|t| t.status == "blocked").count(),
                        "All remaining tasks are blocked — not incrementing dispatch_failures"
                    );
                } else {
                    // No progress this cycle
                    updated_goal.dispatch_failures += 1;
                    info!(
                        goal_id = %goal_id,
                        dispatch_failures = updated_goal.dispatch_failures,
                        completed_tasks = completed_after,
                        remaining_tasks = tasks.iter().filter(|t| t.status == "pending" || t.status == "claimed").count(),
                        "No progress this dispatch cycle"
                    );
                }
            }

            // Circuit breaker: stall after 3 consecutive failures
            const MAX_DISPATCH_FAILURES: i32 = 3;
            if updated_goal.dispatch_failures >= MAX_DISPATCH_FAILURES
                && updated_goal.status != "completed"
                && updated_goal.status != "failed"
            {
                updated_goal.status = "stalled".to_string();
                info!(
                    goal_id = %goal_id,
                    dispatch_failures = updated_goal.dispatch_failures,
                    "Goal stalled: {} consecutive dispatch cycles with no progress",
                    updated_goal.dispatch_failures
                );
            }

            updated_goal.updated_at = chrono::Utc::now().to_rfc3339();
            let _ = state.update_goal(&updated_goal).await;

            // If goal is stalled or failed, cancel remaining pending tasks
            if updated_goal.status == "stalled" || updated_goal.status == "failed" {
                let mut cancelled = 0;
                for task in &tasks {
                    if task.status == "pending" || task.status == "claimed" {
                        let mut t = task.clone();
                        t.status = "completed".to_string();
                        t.error = Some(
                            "Cancelled: goal stalled (no progress after 3 dispatch cycles)"
                                .to_string(),
                        );
                        t.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        let _ = state.update_task(&t).await;
                        cancelled += 1;
                    }
                }
                if cancelled > 0 {
                    info!(goal_id = %goal_id, cancelled, "Cancelled orphaned tasks for stalled goal");
                }
            }
        }

        // Enqueue notification for delivery (persisted in SQLite).
        // Then attempt immediate delivery via hub if available.
        let final_goal = state.get_goal(&goal_id).await;
        let status = final_goal
            .as_ref()
            .ok()
            .and_then(|g| g.as_ref())
            .map(|g| g.status.as_str())
            .unwrap_or("unknown");
        // Only notify for terminal states — "active" means it's still in progress
        if status == "active" || status == "pending" {
            // Goal still in progress: optionally relay substantive task-lead text,
            // but only if executor results haven't already been sent inline
            // (which would cover the same content).
            if !any_executor_results_sent {
                if let Some(response) = task_lead_response.as_ref() {
                    if !is_low_signal_task_lead_reply(response) {
                        if let Some(hub_weak) = &hub {
                            if let Some(hub_arc) = hub_weak.upgrade() {
                                let _ = hub_arc.send_text(&session_id, response).await;
                            }
                        }
                    }
                }
            }

            // Goal is still active, no notification needed.
            // Clean up cancellation token and return.
            if let Some(ref registry) = goal_token_registry {
                registry.remove(&goal_id).await;
            }
            return;
        }

        // For failed/stalled finite goals: attempt direct fallback before giving up.
        // The goal system decomposed the request into subtasks but they weren't
        // completed. Instead of sending a cryptic failure message, try handling
        // the request directly through the agent's main capabilities.
        //
        // Skip fallback if the goal was already notified — this means another
        // task lead (e.g., spawned by the heartbeat) already handled the failure.
        let goal_already_notified = final_goal
            .as_ref()
            .ok()
            .and_then(|g| g.as_ref())
            .map(|g| g.notified_at.is_some())
            .unwrap_or(false);
        let (notification_type, msg) = if (status == "failed" || status == "stalled")
            && !goal_already_notified
            && final_goal
                .as_ref()
                .ok()
                .and_then(|g| g.as_ref())
                .map(|g| g.goal_type == "finite")
                .unwrap_or(false)
        {
            info!(goal_id = %goal_id, "Finite goal failed — attempting direct fallback");

            // Mark as notified immediately to prevent the heartbeat from
            // sending a duplicate "Goal failed" notification while the
            // fallback is in progress.
            let _ = state.mark_goal_notified(&goal_id).await;

            // Notify user we're retrying with a different approach
            if let Some(hub_weak) = &hub {
                if let Some(hub_arc) = hub_weak.upgrade() {
                    let _ = hub_arc
                        .send_text(
                            &session_id,
                            "The task planner couldn't complete this. Let me try handling it directly...",
                        )
                        .await;
                }
            }

            // Spawn a direct executor to handle the original request
            // without goal/task decomposition
            let fallback_result = agent
                .spawn_child(
                    &user_text,
                    &user_text,
                    None,
                    fallback_channel_ctx,
                    fallback_user_role,
                    None, // no specific role — gets full tool access
                    None, // no goal_id — prevents goal re-entry
                    None,
                    None,
                )
                .await;

            match fallback_result {
                Ok(response)
                    if !response.trim().is_empty()
                        && !goal_completion_response_indicates_incomplete_work(&response) =>
                {
                    // Direct handling succeeded — update goal to completed
                    if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                        g.status = "completed".to_string();
                        g.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        g.updated_at = chrono::Utc::now().to_rfc3339();
                        let _ = state.update_goal(&g).await;
                    }
                    info!(goal_id = %goal_id, "Direct fallback succeeded");
                    (
                        "completed",
                        format!(
                            "Goal completed: {}",
                            truncate_goal_result_text(&response, 4000)
                        ),
                    )
                }
                Ok(response) if !response.trim().is_empty() => {
                    info!(
                        goal_id = %goal_id,
                        "Direct fallback returned an incomplete/unverified response"
                    );
                    (
                        "failed",
                        format!(
                            "I made some progress, but I couldn't verify the final outcome:\n\n{}",
                            truncate_goal_result_text(&response, 3500)
                        ),
                    )
                }
                _ => {
                    // Direct handling also failed — give detailed info
                    let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();
                    let task_summary: String = tasks
                        .iter()
                        .take(5)
                        .map(|t| {
                            let err = t.error.as_deref().unwrap_or("no details");
                            format!("• {} ({})", t.description, err)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    info!(goal_id = %goal_id, "Direct fallback also failed");
                    (
                        "failed",
                        format!(
                            "I wasn't able to complete your request. Here's what I tried:\n{}\n\nYou could try rephrasing or breaking it into smaller steps.",
                            if task_summary.is_empty() {
                                "(no task details available)".to_string()
                            } else {
                                task_summary
                            }
                        ),
                    )
                }
            }
        } else {
            let completed_tasks = state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();
            let task_lead_error = result.as_ref().err().map(|e| e.to_string());
            match status {
                "completed" => {
                    if any_executor_results_sent {
                        // Executor results were already sent inline — don't repeat them.
                        // Send a brief completion signal instead.
                        let desc_preview: String = final_goal
                            .as_ref()
                            .ok()
                            .and_then(|g| g.as_ref())
                            .map(|g| g.description.chars().take(100).collect::<String>())
                            .unwrap_or_default();
                        ("completed", format!("Goal completed: {}", desc_preview))
                    } else {
                        // No inline results sent — include full task results in notification.
                        let fallback_summary = match &result {
                            Ok(r) => r.as_str(),
                            Err(_) => "All tasks completed.",
                        };
                        let task_results_summary =
                            build_goal_task_results_summary(&completed_tasks, fallback_summary);

                        // Check for partial success metadata in the goal context
                        let partial_info = final_goal
                            .as_ref()
                            .ok()
                            .and_then(|g| g.as_ref())
                            .and_then(|g| g.context.as_deref())
                            .and_then(|ctx| serde_json::from_str::<serde_json::Value>(ctx).ok())
                            .filter(|v| {
                                v.get("partial_success")
                                    .and_then(|p| p.as_bool())
                                    .unwrap_or(false)
                            });

                        if let Some(summary) = partial_info {
                            let completed = summary
                                .get("completed")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let failed =
                                summary.get("failed").and_then(|v| v.as_u64()).unwrap_or(0);
                            let blocked =
                                summary.get("blocked").and_then(|v| v.as_u64()).unwrap_or(0);
                            let total = summary.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
                            (
                                "completed",
                                format!(
                                    "Goal partially completed ({}/{} tasks succeeded, {} failed, {} blocked):\n\n{}",
                                    completed,
                                    total,
                                    failed,
                                    blocked,
                                    task_results_summary.chars().take(3500).collect::<String>()
                                ),
                            )
                        } else {
                            (
                                "completed",
                                format!(
                                    "Goal completed:\n\n{}",
                                    task_results_summary.chars().take(4000).collect::<String>()
                                ),
                            )
                        }
                    }
                }
                "failed" => (
                    "failed",
                    format!(
                        "Goal failed: {}",
                        build_goal_failure_summary(
                            final_goal.as_ref().ok().and_then(|g| g.as_ref()),
                            &completed_tasks,
                            task_lead_response.as_deref(),
                            task_lead_error.as_deref(),
                        )
                    ),
                ),
                "cancelled" => ("completed", "Goal was cancelled.".to_string()),
                "stalled" => (
                    "failed",
                    format!(
                        "Goal stalled (no progress after 3 dispatch cycles): {}",
                        goal_id
                    ),
                ),
                _ => (
                    "failed",
                    format!(
                        "Goal failed: {}",
                        build_goal_failure_summary(
                            final_goal.as_ref().ok().and_then(|g| g.as_ref()),
                            &completed_tasks,
                            task_lead_response.as_deref(),
                            task_lead_error.as_deref(),
                        )
                    ),
                ),
            }
        };

        let entry =
            crate::traits::NotificationEntry::new(&goal_id, &session_id, notification_type, &msg);
        let notification_id = entry.id.clone();
        let _ = state.enqueue_notification(&entry).await;

        // Mark goal as notified so heartbeat doesn't double-enqueue
        let _ = state.mark_goal_notified(&goal_id).await;

        // Attempt immediate delivery — if it fails, heartbeat will retry from queue
        if let Some(hub_weak) = &hub {
            if let Some(hub_arc) = hub_weak.upgrade() {
                if hub_arc.send_text(&session_id, &msg).await.is_ok() {
                    let _ = state.mark_notification_delivered(&notification_id).await;

                    // Auto-send any files referenced in the completion message
                    let file_paths = extract_file_paths_from_text(&msg);
                    for path in file_paths {
                        let filename = std::path::Path::new(&path)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "file".to_string());
                        let media = crate::types::MediaMessage {
                            session_id: session_id.clone(),
                            caption: filename.clone(),
                            kind: crate::types::MediaKind::Document {
                                file_path: path.clone(),
                                filename,
                            },
                        };
                        if let Err(e) = hub_arc.send_media(&session_id, &media).await {
                            warn!("Failed to auto-send goal file {}: {}", path, e);
                        }
                    }
                }
            }
        }

        // Clean up cancellation token
        if let Some(ref registry) = goal_token_registry {
            registry.remove(&goal_id).await;
        }
    });
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm_runtime: SharedLlmRuntime,
        state: Arc<dyn StateStore>,
        event_store: Arc<EventStore>,
        tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        config_path: PathBuf,
        skills_dir: PathBuf,
        max_depth: usize,
        max_iterations: usize,
        max_iterations_cap: usize,
        max_response_chars: usize,
        timeout_secs: u64,
        max_facts: usize,
        daily_token_budget: Option<u64>,
        iteration_config: IterationLimitConfig,
        task_timeout_secs: Option<u64>,
        task_token_budget: Option<u64>,
        llm_call_timeout_secs: Option<u64>,
        mcp_registry: Option<McpRegistry>,
        goal_token_registry: Option<GoalTokenRegistry>,
        hub: Option<Weak<ChannelHub>>,
        record_decision_points: bool,
        context_window_config: crate::config::ContextWindowConfig,
        policy_config: PolicyConfig,
        path_aliases: PathAliasConfig,
        inherited_project_scope: Option<String>,
    ) -> Self {
        init_policy_tunables_once(policy_config.uncertainty_clarify_threshold);
        let fallback = if let Some(router) = llm_runtime.router() {
            info!(
                default_model = router.default_model(),
                fallbacks = ?router.fallback_models(),
                "Model router enabled"
            );
            router
                .first_fallback()
                .map(str::to_string)
                .unwrap_or_else(|| model.clone())
        } else {
            info!("No distinct fallback models configured; fallback cascade limited");
            model.clone()
        };

        // Log iteration config
        match &iteration_config {
            IterationLimitConfig::Unlimited => {
                info!("Iteration limit: Unlimited (natural completion)");
            }
            IterationLimitConfig::Soft { threshold, warn_at } => {
                info!(threshold, warn_at, "Iteration limit: Soft");
            }
            IterationLimitConfig::Hard { initial, cap } => {
                info!(initial, cap, "Iteration limit: Hard (legacy)");
            }
        }

        if let Some(secs) = llm_call_timeout_secs {
            info!(timeout_secs = secs, "LLM call watchdog timeout enabled");
        }

        Self {
            llm_runtime,
            state,
            event_store,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skill_cache: skills::SkillCache::new(skills_dir.clone()),
            skills_dir,
            depth: 0,
            max_depth,
            iteration_config,
            max_iterations,
            max_iterations_cap,
            max_response_chars,
            timeout_secs,
            max_facts,
            model_override: RwLock::new(false),
            daily_token_budget,
            llm_call_timeout: llm_call_timeout_secs.map(Duration::from_secs),
            task_timeout: task_timeout_secs.map(Duration::from_secs),
            task_token_budget,
            verification_tracker: Some(Arc::new(VerificationTracker::new())),
            mcp_registry,
            role: AgentRole::Orchestrator,
            task_id: None,
            goal_id: None,
            cancel_token: None,
            goal_token_registry,
            hub: RwLock::new(hub),
            schedule_approved_sessions: Arc::new(tokio::sync::RwLock::new(HashSet::new())),
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
            path_aliases,
            inherited_project_scope,
            root_tools: None, // Root agent — its own tools ARE the root tools
            record_decision_points,
            #[cfg(test)]
            execution_budget_override: None,
        }
    }

    /// Override agent to executor mode (depth=1) for integration tests.
    /// This bypasses orchestrator routing so tests exercise the execution loop directly.
    #[cfg(test)]
    pub fn set_test_executor_mode(&mut self) {
        self.depth = 1;
        self.role = AgentRole::Executor;
    }

    /// Reset agent to orchestrator mode (depth=0) for integration tests.
    /// Use this when testing depth-0-only code paths (e.g. "Done" synthesis).
    #[cfg(test)]
    pub fn set_test_orchestrator_mode(&mut self) {
        self.depth = 0;
        self.role = AgentRole::Orchestrator;
    }

    #[cfg(test)]
    pub fn set_test_task_lead_mode(&mut self) {
        self.depth = 1;
        self.role = AgentRole::TaskLead;
    }

    #[cfg(test)]
    pub fn set_test_task_token_budget(&mut self, budget: Option<u64>) {
        self.task_token_budget = budget;
    }

    #[cfg(test)]
    pub fn set_test_execution_budget_override(&mut self, budget: Option<ExecutionBudget>) {
        self.execution_budget_override = budget;
    }

    #[cfg(test)]
    pub fn set_test_daily_token_budget(&mut self, budget: Option<u64>) {
        self.daily_token_budget = budget;
    }

    #[cfg(test)]
    pub fn set_test_iteration_config(&mut self, config: IterationLimitConfig) {
        self.iteration_config = config;
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub fn set_test_task_timeout(&mut self, timeout: Option<Duration>) {
        self.task_timeout = timeout;
    }

    #[cfg(test)]
    pub fn set_test_goal_id(&mut self, goal_id: Option<String>) {
        self.goal_id = goal_id;
    }

    #[cfg(test)]
    pub fn set_test_task_id(&mut self, task_id: Option<String>) {
        self.task_id = task_id;
    }

    #[cfg(test)]
    pub async fn set_test_schedule_approval_for_session(&self, session_id: &str, approved: bool) {
        let mut sessions = self.schedule_approved_sessions.write().await;
        if approved {
            sessions.insert(session_id.to_string());
        } else {
            sessions.remove(session_id);
        }
    }

    /// Create an Agent with explicit depth/max_depth (used internally for sub-agents).
    /// Sub-agents don't auto-route — they use whatever model was selected by the parent.
    #[allow(clippy::too_many_arguments)]
    fn with_depth(
        llm_runtime: SharedLlmRuntime,
        state: Arc<dyn StateStore>,
        event_store: Arc<EventStore>,
        tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        config_path: PathBuf,
        skills_dir: PathBuf,
        depth: usize,
        max_depth: usize,
        iteration_config: IterationLimitConfig,
        max_iterations: usize,
        max_iterations_cap: usize,
        max_response_chars: usize,
        timeout_secs: u64,
        max_facts: usize,
        task_timeout: Option<Duration>,
        task_token_budget: Option<u64>,
        llm_call_timeout: Option<Duration>,
        mcp_registry: Option<McpRegistry>,
        verification_tracker: Option<Arc<VerificationTracker>>,
        role: AgentRole,
        task_id: Option<String>,
        goal_id: Option<String>,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
        goal_token_registry: Option<GoalTokenRegistry>,
        hub: Option<Weak<ChannelHub>>,
        schedule_approved_sessions: Arc<tokio::sync::RwLock<HashSet<String>>>,
        record_decision_points: bool,
        context_window_config: crate::config::ContextWindowConfig,
        policy_config: PolicyConfig,
        path_aliases: PathAliasConfig,
        inherited_project_scope: Option<String>,
        root_tools: Option<Vec<Arc<dyn Tool>>>,
    ) -> Self {
        let fallback = llm_runtime
            .router()
            .and_then(|router| router.first_fallback().map(str::to_string))
            .unwrap_or_else(|| model.clone());
        Self {
            llm_runtime,
            state,
            event_store,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skill_cache: skills::SkillCache::new(skills_dir.clone()),
            skills_dir,
            depth,
            max_depth,
            iteration_config,
            max_iterations,
            max_iterations_cap,
            max_response_chars,
            timeout_secs,
            max_facts,
            model_override: RwLock::new(false),
            daily_token_budget: None,
            llm_call_timeout,
            task_timeout,
            task_token_budget,
            verification_tracker,
            mcp_registry,
            role,
            task_id,
            goal_id,
            cancel_token,
            goal_token_registry,
            hub: RwLock::new(hub),
            schedule_approved_sessions,
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
            path_aliases,
            inherited_project_scope,
            root_tools,
            record_decision_points,
            #[cfg(test)]
            execution_budget_override: None,
        }
    }

    /// Set the ChannelHub reference (called after hub creation in core.rs).
    pub async fn set_hub(&self, hub: Weak<ChannelHub>) {
        *self.hub.write().await = Some(hub);
    }

    /// Set a weak self-reference for background task spawning.
    /// Must be called after wrapping the Agent in Arc.
    pub async fn set_self_ref(&self, weak: Weak<Agent>) {
        *self.self_ref.write().await = Some(weak);
    }

    /// Current recursion depth of this agent.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Maximum recursion depth allowed.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Role for this agent instance.
    pub fn role(&self) -> AgentRole {
        self.role
    }

    /// Validate that an executor spawn targets a valid, pre-claimed task.
    ///
    /// This prevents duplicate/invalid execution when task leads attempt to
    /// spawn executors without claiming, with stale IDs, or against finished tasks.
    pub async fn validate_executor_task_for_spawn(
        &self,
        task_id: &str,
        expected_goal_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let Some(task) = self.state.get_task(task_id).await? else {
            anyhow::bail!(
                "Task '{}' was not found. Use manage_goal_tasks(list_tasks) and pass a valid task_id.",
                task_id
            );
        };

        if let Some(goal_id) = expected_goal_id {
            if task.goal_id != goal_id {
                anyhow::bail!(
                    "Task '{}' belongs to goal '{}', not '{}'.",
                    task_id,
                    task.goal_id,
                    goal_id
                );
            }
        }

        match task.status.as_str() {
            "claimed" => Ok(()),
            "pending" => anyhow::bail!(
                "Task '{}' is still pending. Claim it first with manage_goal_tasks(action='claim_task').",
                task_id
            ),
            "running" => anyhow::bail!(
                "Task '{}' is already running. Do not spawn another executor for the same task.",
                task_id
            ),
            "completed" | "failed" | "blocked" | "cancelled" => anyhow::bail!(
                "Task '{}' is '{}' and should not be executed again without an explicit retry/reset.",
                task_id,
                task.status
            ),
            other => anyhow::bail!(
                "Task '{}' has unsupported status '{}' for executor spawn (expected 'claimed').",
                task_id,
                other
            ),
        }
    }

    /// Maximum agentic loop iterations per invocation.
    #[allow(dead_code)]
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Maximum number of retries for transient LLM errors.
    const MAX_LLM_RETRIES: u32 = 3;
    /// Base delay for exponential backoff on transient errors (seconds).
    const RETRY_BASE_DELAY_SECS: u64 = 2;
    /// Single retry budget for malformed payloads that are likely deterministic
    /// (shape/unknown). Parse errors use transient retry + fallback recovery.
    const MAX_MALFORMED_PAYLOAD_RETRIES: u32 = 1;
    /// Small delay before malformed-payload retry to smooth transient gateway glitches.
    const MALFORMED_PAYLOAD_RETRY_DELAY_SECS: u64 = 1;

    // ==================== Orchestration Methods ====================

    /// Run the agentic loop for a user message in the given session.
    /// Returns the final assistant text response.
    /// `heartbeat` is an optional atomic timestamp updated on each activity point.
    /// Channels pass `Some(heartbeat)` so the typing indicator can detect stalls;
    /// sub-agents, triggers, and tests pass `None`.
    fn sanitize_final_reply_markers(reply: &str) -> String {
        crate::tools::sanitize::sanitize_user_facing_reply(reply)
    }

    pub async fn handle_message(
        &self,
        session_id: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String> {
        let scheduled_goal_to_clear = if let Some(goal_id) = self.goal_id.as_deref() {
            let is_scheduled_goal =
                goal_has_scheduled_provenance(&self.state, goal_id, self.task_id.as_deref()).await;
            let is_root_scheduled_run = if self.task_id.is_none() {
                is_scheduled_goal
            } else {
                task_has_scheduled_provenance(&self.state, self.task_id.as_deref()).await
            };
            if is_root_scheduled_run {
                Some(goal_id.to_string())
            } else {
                None
            }
        } else {
            None
        };

        let reply = self
            .handle_message_impl(
                session_id,
                user_text,
                status_tx,
                user_role,
                channel_ctx,
                heartbeat,
            )
            .await;

        if let Some(goal_id) = scheduled_goal_to_clear.as_deref() {
            if let Some(registry) = self.goal_token_registry.as_ref() {
                registry.clear_run_budget(goal_id).await;
            }
            clear_scheduled_run_state(&self.state, goal_id).await;
        }

        let reply = reply?;

        // Strip control markers that may have leaked through model echoing.
        let reply = Self::sanitize_final_reply_markers(&reply);

        Ok(reply)
    }

    /// Cancel all active/pending goals for a session.
    ///
    /// This is used by channels to implement fast-path `cancel`/`stop` handling
    /// without needing an LLM call. It cancels the goal token (cascading to task
    /// leads/executors), updates goal/task DB state, and removes any schedules.
    pub async fn cancel_active_goals_for_session(&self, session_id: &str) -> Vec<String> {
        let goals = self
            .state
            .get_goals_for_session(session_id)
            .await
            .unwrap_or_default();
        let active: Vec<&crate::traits::Goal> = goals
            .iter()
            .filter(|g| {
                matches!(
                    g.status.as_str(),
                    "active" | "pending" | "pending_confirmation"
                )
            })
            .collect();
        if active.is_empty() {
            return Vec::new();
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut cancelled = Vec::new();
        for goal in active {
            if let Some(ref registry) = self.goal_token_registry {
                registry.cancel(&goal.id).await;
                registry.clear_run_budget(&goal.id).await;
            }
            clear_scheduled_run_state(&self.state, &goal.id).await;

            let mut updated = goal.clone();
            updated.status = "cancelled".to_string();
            updated.updated_at = now.clone();
            updated.completed_at = Some(now.clone());
            let _ = self.state.update_goal(&updated).await;

            // Best-effort cleanup: cancelled goals should not retain schedules.
            if let Ok(schedules) = self.state.get_schedules_for_goal(&updated.id).await {
                for s in &schedules {
                    let _ = self.state.delete_goal_schedule(&s.id).await;
                }
            }

            // Cancel all remaining tasks for this goal.
            if let Ok(tasks) = self.state.get_tasks_for_goal(&updated.id).await {
                for task in tasks {
                    if task.status != "completed"
                        && task.status != "failed"
                        && task.status != "cancelled"
                    {
                        let mut t = task.clone();
                        t.status = "cancelled".to_string();
                        t.completed_at = Some(now.clone());
                        let _ = self.state.update_task(&t).await;
                    }
                }
            }

            cancelled.push(updated.description.chars().take(100).collect());
        }

        cancelled
    }
}

#[cfg(test)]
mod final_reply_marker_tests {
    use std::collections::HashMap;

    use chrono::Utc;

    use super::{post_task, user_facing_task_description, Agent, LearningContext};

    #[test]
    fn strips_control_markers_from_final_reply() {
        let reply = "Done.\n\n[SYSTEM] internal note\n[DIAGNOSTIC] trace\n[UNTRUSTED EXTERNAL DATA from 'web_fetch' — test]\npayload\n[END UNTRUSTED EXTERNAL DATA]";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(!sanitized.contains("[SYSTEM]"));
        assert!(!sanitized.contains("[DIAGNOSTIC]"));
        assert!(
            !sanitized.contains("internal note"),
            "SYSTEM content leaked: {sanitized}"
        );
        assert!(!sanitized.contains("UNTRUSTED EXTERNAL DATA"));
        assert!(sanitized.contains("Done."));
    }

    #[test]
    fn strips_diagnostic_blocks_with_content_from_final_reply() {
        let reply = "I encountered an error with the search.\n\n\
            [DIAGNOSTIC] Similar errors resolved before:\n\
            - Used terminal to resolve the issue\n\
              Steps: run cargo build -> fix errors\n\n\
            [TOOL STATS] search_files (24h): 8 calls, 0 failed (0%), avg 296ms\n\
              - 2x: pattern not found\n\n\
            [SYSTEM] This tool has errored 2 semantic times. Do NOT retry it.\n\n\
            I will try a different approach.";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(
            !sanitized.contains("[DIAGNOSTIC]"),
            "DIAGNOSTIC tag leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("Similar errors resolved before"),
            "diagnostic content leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("Used terminal"),
            "solution leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("[TOOL STATS]"),
            "TOOL STATS tag leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("8 calls"),
            "stats content leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("296ms"),
            "stats duration leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("[SYSTEM]"),
            "SYSTEM tag leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("errored 2 semantic times"),
            "system content leaked: {sanitized}"
        );
        assert!(sanitized.contains("I encountered an error with the search."));
        assert!(sanitized.contains("I will try a different approach."));
    }

    #[test]
    fn strips_prior_turn_markers_from_final_reply() {
        let reply = "Summary [prior turn, truncated]\nNext [prior turn]";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(!sanitized.contains("[prior turn"));
        assert_eq!(sanitized, "Summary\nNext");
    }

    #[test]
    fn strips_model_identity_leaks_from_final_reply() {
        let reply = "I am a large language model, trained by Google. How can I help?";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(!sanitized.contains("trained by Google"));
        assert!(sanitized.contains("aidaemon"));
    }

    #[test]
    fn strips_leaked_tool_protocol_tokens_after_graceful_summary() {
        let learning_ctx = LearningContext {
            user_text: "debug this failure".to_string(),
            intent_domains: vec![],
            tool_calls: vec!["terminal(`vendor/bin/drush status`)".to_string()],
            errors: vec![],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
            replay_notes: Vec::new(),
        };
        let mut tool_failure_count = HashMap::new();
        tool_failure_count.insert(
            "terminal".to_string(),
            super::semantic_failure_limit("terminal"),
        );
        let graceful = post_task::graceful_stall_response(
            &learning_ctx,
            false,
            "deferred-no-tool",
            &tool_failure_count,
        );
        assert!(graceful.contains("command execution"));

        let leaked = format!(
            "{}\n<|tool_calls_section_begin|>\nfunctions.terminal:0 {{\"command\":\"pwd\"}}",
            graceful
        );
        let sanitized = Agent::sanitize_final_reply_markers(&leaked);
        assert!(!sanitized.contains("<|tool_calls_section_begin|>"));
        assert!(!sanitized.contains("functions.terminal:0"));
        assert!(sanitized.contains("command execution"));
    }

    #[test]
    fn strips_xml_function_call_blocks_from_final_reply() {
        let reply = "I'll read the most recent 300 lines from that log file.\n\n<function_calls>\n<invoke name=\"terminal\">\n<parameter name=\"command\">tail -n 300 ~/Library/Logs/aidaemon/stdout.log</parameter>\n</invoke>\n</function_calls>\n\nHere's what I found.";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(!sanitized.contains("<function_calls>"));
        assert!(!sanitized.contains("<invoke"));
        assert!(!sanitized.contains("<parameter"));
        assert!(!sanitized.contains("tail -n 300"));
        assert!(sanitized.contains("I'll read the most recent 300 lines"));
        assert!(sanitized.contains("Here's what I found."));
    }

    #[test]
    fn strips_internal_scheduler_annotations_from_progress_descriptions() {
        let cleaned = user_facing_task_description(
            "Scheduled check: Post evening tweet about aidaemon features [SYSTEM: already scheduled and firing now; do not reschedule.]",
        );
        assert_eq!(cleaned, "Post evening tweet about aidaemon features");
    }
}

#[cfg(test)]
mod tests;
