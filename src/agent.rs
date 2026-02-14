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
use crate::config::{IterationLimitConfig, ModelsConfig, PolicyConfig};
use crate::events::{
    AssistantResponseData, DecisionPointData, DecisionType, ErrorData, EventStore, EventType,
    PolicyDecisionData, SubAgentCompleteData, SubAgentSpawnData, TaskEndData, TaskStartData,
    TaskStatus, ThinkingStartData, ToolCallData, ToolCallInfo, ToolResultData, UserMessageData,
};
use crate::execution_policy::{ApprovalMode, ExecutionPolicy, ModelProfile};
use crate::goal_tokens::GoalTokenRegistry;
use crate::mcp::McpRegistry;
use crate::providers::{ProviderError, ProviderErrorKind};
use crate::router::{self, Router};
use crate::skills::{self, MemoryContext};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::VerificationTracker;
use crate::traits::{
    AgentRole, GoalV3, Message, ModelProvider, StateStore, TaskActivityV3, Tool, ToolCall,
    ToolCapabilities, ToolRole,
};
use crate::types::{ApprovalResponse, ChannelContext, ChannelVisibility, UserRole};
// Re-export StatusUpdate from types for backwards compatibility
pub use crate::types::StatusUpdate;

/// Constants for stall and repetitive behavior detection
const MAX_STALL_ITERATIONS: usize = 3;
const DEFERRED_NO_TOOL_SWITCH_THRESHOLD: usize = 2;
const MAX_DEFERRED_NO_TOOL_MODEL_SWITCHES: usize = 1;
const DEFERRED_NO_TOOL_ERROR_MARKER: &str = "deferred-action no-tool loop";
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
const MAX_CONSECUTIVE_SAME_TOOL: usize = 16;
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
/// Marker for consultant mode so providers can enforce text-only behavior.
const CONSULTANT_TEXT_ONLY_MARKER: &str = "[CONSULTANT_TEXT_ONLY_MODE]";
/// Machine-readable intent decision line emitted by the consultant pass.
const INTENT_GATE_MARKER: &str = "[INTENT_GATE]";
/// Legacy fallback schedule text heuristics are disabled by default because they
/// can misclassify "tell me about this scheduled goal" queries as new schedules.
const ENABLE_SCHEDULE_HEURISTICS: bool = false;

mod intent_gate;
use intent_gate::extract_intent_gate;
#[cfg(test)]
use intent_gate::parse_intent_gate_json;
mod consultant_analysis;
#[cfg(test)]
use consultant_analysis::has_action_promise;
use consultant_analysis::{looks_like_deferred_action_response, sanitize_consultant_analysis};
mod intent_routing;
use intent_routing::{
    classify_intent_complexity, contains_keyword_as_words, infer_intent_gate,
    is_internal_maintenance_intent, IntentComplexity,
};
#[cfg(test)]
use intent_routing::{detect_schedule_heuristic, looks_like_recurring_intent_without_timing};
mod policy_signals;
use policy_signals::{
    build_policy_bundle_v1, default_clarifying_question, detect_explicit_outcome_signal,
    is_short_user_correction, tool_is_side_effecting,
};
mod loop_utils;
#[cfg(test)]
use loop_utils::merge_consecutive_messages;
use loop_utils::{
    extract_command_from_args, extract_file_path_from_args, extract_send_file_dedupe_key_from_args,
    fixup_message_ordering, hash_tool_call, is_trigger_session,
};
mod post_task;
use post_task::LearningContext;

mod graceful;
mod history;
mod llm;
mod main_loop;
mod models;
mod resume;
mod spawn;
mod system_prompt;
mod tool_defs;
mod tool_exec;

use system_prompt::{build_consultant_system_prompt, format_goal_context};

#[cfg(test)]
use system_prompt::strip_markdown_section;

struct PolicyRuntimeMetrics {
    router_shadow_total: AtomicU64,
    router_shadow_diverged: AtomicU64,
    tool_exposure_samples: AtomicU64,
    tool_exposure_before_sum: AtomicU64,
    tool_exposure_after_sum: AtomicU64,
    ambiguity_detected_total: AtomicU64,
    uncertainty_clarify_total: AtomicU64,
    context_refresh_total: AtomicU64,
    escalation_total: AtomicU64,
    fallback_expansion_total: AtomicU64,
}

impl PolicyRuntimeMetrics {
    const fn new() -> Self {
        Self {
            router_shadow_total: AtomicU64::new(0),
            router_shadow_diverged: AtomicU64::new(0),
            tool_exposure_samples: AtomicU64::new(0),
            tool_exposure_before_sum: AtomicU64::new(0),
            tool_exposure_after_sum: AtomicU64::new(0),
            ambiguity_detected_total: AtomicU64::new(0),
            uncertainty_clarify_total: AtomicU64::new(0),
            context_refresh_total: AtomicU64::new(0),
            escalation_total: AtomicU64::new(0),
            fallback_expansion_total: AtomicU64::new(0),
        }
    }
}

static POLICY_METRICS: Lazy<PolicyRuntimeMetrics> = Lazy::new(PolicyRuntimeMetrics::new);

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

#[derive(Debug, Clone, serde::Serialize)]
pub struct PolicyMetricsSnapshot {
    pub router_shadow_total: u64,
    pub router_shadow_diverged: u64,
    pub tool_exposure_samples: u64,
    pub tool_exposure_before_sum: u64,
    pub tool_exposure_after_sum: u64,
    pub ambiguity_detected_total: u64,
    pub uncertainty_clarify_total: u64,
    pub context_refresh_total: u64,
    pub escalation_total: u64,
    pub fallback_expansion_total: u64,
}

pub fn policy_metrics_snapshot() -> PolicyMetricsSnapshot {
    PolicyMetricsSnapshot {
        router_shadow_total: POLICY_METRICS.router_shadow_total.load(Ordering::Relaxed),
        router_shadow_diverged: POLICY_METRICS
            .router_shadow_diverged
            .load(Ordering::Relaxed),
        tool_exposure_samples: POLICY_METRICS.tool_exposure_samples.load(Ordering::Relaxed),
        tool_exposure_before_sum: POLICY_METRICS
            .tool_exposure_before_sum
            .load(Ordering::Relaxed),
        tool_exposure_after_sum: POLICY_METRICS
            .tool_exposure_after_sum
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
    }
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
fn summarize_tool_args(name: &str, arguments: &str) -> String {
    let val: Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };

    match name {
        "terminal" => val
            .get("command")
            .and_then(|v| v.as_str())
            .map(|cmd| {
                let truncated: String = cmd.chars().take(60).collect();
                if cmd.len() > 60 {
                    format!("`{}...`", truncated)
                } else {
                    format!("`{}`", truncated)
                }
            })
            .unwrap_or_default(),
        "browser" => {
            let action = val.get("action").and_then(|v| v.as_str()).unwrap_or("");
            let url = val.get("url").and_then(|v| v.as_str()).unwrap_or("");
            if !url.is_empty() {
                format!("{} {}", action, url)
            } else {
                action.to_string()
            }
        }
        "spawn_agent" => val
            .get("mission")
            .and_then(|v| v.as_str())
            .map(|m| {
                let truncated: String = m.chars().take(50).collect();
                if m.len() > 50 {
                    format!("{}...", truncated)
                } else {
                    truncated
                }
            })
            .unwrap_or_default(),
        "remember_fact" => val
            .get("fact")
            .and_then(|v| v.as_str())
            .map(|f| {
                let truncated: String = f.chars().take(40).collect();
                if f.len() > 40 {
                    format!("{}...", truncated)
                } else {
                    truncated
                }
            })
            .unwrap_or_else(|| "saving to memory".to_string()),
        _ => String::new(),
    }
}

#[derive(Debug, Clone, Default)]
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
    provider: Arc<dyn ModelProvider>,
    state: Arc<dyn StateStore>,
    event_store: Arc<EventStore>,
    tools: Vec<Arc<dyn Tool>>,
    model: RwLock<String>,
    fallback_model: RwLock<String>,
    system_prompt: String,
    config_path: PathBuf,
    skills_dir: PathBuf,
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
    /// Smart router for automatic model tier selection. None for sub-agents
    /// or when all tiers resolve to the same model.
    router: Option<Router>,
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
    /// V3 role for this agent instance.
    role: AgentRole,
    /// V3 task ID for executor agents — enables activity logging.
    v3_task_id: Option<String>,
    /// V3 goal ID for task lead agents — enables context injection into spawn calls.
    v3_goal_id: Option<String>,
    /// Cancellation token — checked each iteration; cancelled by parent or user.
    cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// Goal cancellation token registry — shared across agent hierarchy.
    goal_token_registry: Option<GoalTokenRegistry>,
    /// Weak reference to the ChannelHub for background notifications.
    /// Uses RwLock because hub is created after Agent (core.rs ordering).
    hub: RwLock<Option<Weak<ChannelHub>>>,
    /// Weak self-reference for background task spawning.
    /// Set after Arc creation via `set_self_ref()`.
    self_ref: RwLock<Option<Weak<Agent>>>,
    /// Context window management configuration.
    context_window_config: crate::config::ContextWindowConfig,
    /// Policy rollout and enforcement configuration.
    policy_config: PolicyConfig,
    /// Runtime state: whether legacy classify_query() routing has graduated/retired.
    classify_query_retired: AtomicBool,
    /// Last graduation check epoch seconds (throttles DB checks).
    last_graduation_check_epoch: AtomicU64,
    /// Full tool list from the root agent — used by TaskLead when spawning
    /// Executor children so they can access Action tools that were filtered
    /// out of the TaskLead's own `tools` vec.
    root_tools: Option<Vec<Arc<dyn Tool>>>,
    /// Emit structured decision points into the event store for self-diagnostics.
    record_decision_points: bool,
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

/// Check whether a session ID corresponds to a group/public channel (not a DM).
/// Used to suppress noisy progress updates in shared channels.
pub fn is_group_session(session_id: &str) -> bool {
    // Discord guild channels: "discord:ch:{channel_id}" or "{bot}:discord:ch:{channel_id}"
    if session_id.contains(":ch:") {
        return true;
    }
    // Slack public/private channels: IDs start with C (public) or G (group/private).
    // DMs start with D. Formats: "slack:C123", "slack:G123", "{bot}:slack:C123",
    // "slack:C123:thread_ts"
    if let Some(idx) = session_id.rfind("slack:") {
        let after_slack = &session_id[idx + 6..];
        if after_slack.starts_with('C') || after_slack.starts_with('G') {
            return true;
        }
    }
    false
}

/// Spawn a V3 task lead in the background (free function to satisfy Send requirements).
/// This runs `spawn_child` on the given agent with TaskLead role, then updates
/// the goal and notifies the user when complete.
#[allow(clippy::too_many_arguments)]
pub fn spawn_background_task_lead(
    agent: Arc<Agent>,
    goal: crate::traits::GoalV3,
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

        // Heartbeat dispatch claims a "trigger" task before spawning this background
        // lead. Mark that trigger task complete immediately so it does not remain
        // stuck in claimed/interrupted state while real subtasks execute.
        if let Some(trigger_task_id) = dispatch_trigger_task_id {
            match state.get_task_v3(&trigger_task_id).await {
                Ok(Some(task)) if task.status == "claimed" || task.status == "running" => {
                    let mut updated = task.clone();
                    updated.status = "completed".to_string();
                    updated.result =
                        Some("Task lead dispatched for deferred goal execution.".to_string());
                    updated.error = None;
                    updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    if let Err(e) = state.update_task_v3(&updated).await {
                        warn!(
                            task_id = %trigger_task_id,
                            goal_id = %goal_id,
                            error = %e,
                            "Failed to mark dispatch trigger task completed"
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
                    .get_tasks_for_goal_v3(&heartbeat_goal_id)
                    .await
                    .unwrap_or_default();
                if tasks.is_empty() {
                    // Tasks not yet created — generic message
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
                } else {
                    // Count genuinely completed tasks (exclude cancelled ones with errors)
                    let completed = tasks
                        .iter()
                        .filter(|t| t.status == "completed" && t.error.is_none())
                        .count();
                    let total = tasks.len();
                    let in_progress: Vec<&str> = tasks
                        .iter()
                        .filter(|t| t.status == "claimed" || t.status == "pending")
                        .take(2)
                        .map(|t| t.description.as_str())
                        .collect();
                    let progress_msg = if in_progress.is_empty() && completed == total {
                        format!("⏳ Progress: {}/{} steps completed", completed, total)
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
                    if let Some(hub_weak) = &heartbeat_hub {
                        if let Some(hub_arc) = hub_weak.upgrade() {
                            let _ = hub_arc.send_text(&heartbeat_session, &progress_msg).await;
                        }
                    }
                }
            }
        });

        let result = agent
            .spawn_child(
                &mission,
                &user_text,
                None,
                channel_ctx,
                fallback_user_role,
                Some(AgentRole::TaskLead),
                Some(goal_id.as_str()),
                None,
            )
            .await;

        // Deliver the task lead result to the originating channel.
        // Without this, scheduled/background task results are stored in DB
        // but never sent to the user (notifications only fire on goal completion).
        let mut delivered_directly = false;
        if let Ok(ref response) = result {
            if !response.trim().is_empty() {
                if let Some(hub_weak) = &hub {
                    if let Some(hub_arc) = hub_weak.upgrade() {
                        if hub_arc.send_text(&session_id, response).await.is_ok() {
                            delivered_directly = true;
                        }
                    }
                }
            }
        }

        // Auto-dispatch: dispatch remaining pending tasks after task lead returns.
        // This handles both cases: LLMs that create tasks but don't spawn executors,
        // AND task leads that completed some tasks but left others pending.
        // Uses a loop to re-evaluate after each batch — completing a task may
        // unblock dependent tasks that weren't dispatchable in the previous pass.
        {
            let max_dispatch_rounds = 10; // safety limit
            for _round in 0..max_dispatch_rounds {
                let all_tasks: Vec<crate::traits::TaskV3> = state
                    .get_tasks_for_goal_v3(&goal_id)
                    .await
                    .unwrap_or_default();

                // Build set of completed task IDs for dependency checking
                let completed_ids: std::collections::HashSet<String> = all_tasks
                    .iter()
                    .filter(|t| t.status == "completed" || t.status == "skipped")
                    .map(|t| t.id.clone())
                    .collect();

                // Filter to pending tasks whose dependencies are all met
                let dispatchable: Vec<crate::traits::TaskV3> = all_tasks
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
                let dispatch_batch: Vec<crate::traits::TaskV3> = dispatchable
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
                    // Claim the task
                    let claimed = match state
                        .claim_task_v3(&task.id, &format!("auto-dispatch-{}", goal_id))
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
                                if let Ok(Some(mut claimed_task)) =
                                    state.get_task_v3(&task.id).await
                                {
                                    claimed_task.started_at = Some(chrono::Utc::now().to_rfc3339());
                                    claimed_task.status = "claimed".to_string();
                                    let _ = state.update_task_v3(&claimed_task).await;
                                }
                            }
                        }

                        if let Ok(Some(mut completed_task)) = state.get_task_v3(&task.id).await {
                            completed_task.status = "completed".to_string();
                            completed_task.result =
                                Some(format!("Waited for {} second(s).", wait_secs));
                            completed_task.error = None;
                            completed_task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            let _ = state.update_task_v3(&completed_task).await;
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
                        )
                        .await;

                    // Update task with result and deliver to channel
                    let mut updated = task.clone();
                    match exec_result {
                        Ok(response) => {
                            // Deliver executor result to the originating channel
                            if !response.trim().is_empty() {
                                if let Some(hub_weak) = &hub {
                                    if let Some(hub_arc) = hub_weak.upgrade() {
                                        let _ = hub_arc.send_text(&session_id, &response).await;
                                    }
                                }
                            }
                            updated.status = "completed".to_string();
                            updated.result = Some(response);
                            updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        }
                        Err(e) => {
                            updated.status = "failed".to_string();
                            updated.error = Some(e.to_string());
                        }
                    }
                    let _ = state.update_task_v3(&updated).await;
                }
            }
        }

        // Stop the heartbeat
        let _ = heartbeat_cancel_tx.send(());
        let _ = heartbeat_handle.await;

        // Check the actual goal status from DB — the task lead may have already
        // set it via complete_goal/fail_goal. Only update if still "active".
        let current_goal = state.get_goal_v3(&goal.id).await;
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

            let tasks = state
                .get_tasks_for_goal_v3(&goal_id)
                .await
                .unwrap_or_default();
            let all_done = !tasks.is_empty()
                && tasks
                    .iter()
                    .all(|t| t.status == "completed" || t.status == "skipped");

            let mut updated_goal = match state.get_goal_v3(&goal_id).await {
                Ok(Some(g)) => g,
                _ => goal,
            };

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
            let _ = state.update_goal_v3(&updated_goal).await;

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
                        let _ = state.update_task_v3(&t).await;
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
        let final_goal = state.get_goal_v3(&goal_id).await;
        let status = final_goal
            .as_ref()
            .ok()
            .and_then(|g| g.as_ref())
            .map(|g| g.status.as_str())
            .unwrap_or("unknown");
        // Only notify for terminal states — "active" means it's still in progress
        if status == "active" || status == "pending" {
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
                )
                .await;

            match fallback_result {
                Ok(response) if !response.trim().is_empty() => {
                    // Direct handling succeeded — update goal to completed
                    if let Ok(Some(mut g)) = state.get_goal_v3(&goal_id).await {
                        g.status = "completed".to_string();
                        g.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        g.updated_at = chrono::Utc::now().to_rfc3339();
                        let _ = state.update_goal_v3(&g).await;
                    }
                    info!(goal_id = %goal_id, "Direct fallback succeeded");
                    (
                        "completed",
                        format!(
                            "Goal completed: {}",
                            response.chars().take(4000).collect::<String>()
                        ),
                    )
                }
                _ => {
                    // Direct handling also failed — give detailed info
                    let tasks = state
                        .get_tasks_for_goal_v3(&goal_id)
                        .await
                        .unwrap_or_default();
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
            match status {
                "completed" => {
                    // Build notification from actual task results, not the task lead's
                    // planning message. The task lead response is just a plan outline;
                    // the real outputs come from the executor tasks.
                    let completed_tasks = state
                        .get_tasks_for_goal_v3(&goal_id)
                        .await
                        .unwrap_or_default();

                    // Build a result summary from completed tasks (skip the echo/setup tasks
                    // and focus on tasks that produced meaningful output)
                    let task_results: Vec<String> = completed_tasks
                        .iter()
                        .filter(|t| t.status == "completed" && t.error.is_none())
                        .filter_map(|t| {
                            t.result.as_ref().map(|r| {
                                let truncated: String = r.chars().take(800).collect();
                                format!("**{}**\n{}", t.description, truncated)
                            })
                        })
                        .collect();

                    let task_results_summary = if task_results.is_empty() {
                        // Fall back to task lead response if no task results exist
                        result
                            .as_ref()
                            .map(|r| r.chars().take(4000).collect::<String>())
                            .unwrap_or_else(|_| "All tasks completed.".to_string())
                    } else {
                        // Use the last task's result as primary output (usually the final
                        // deliverable like a report or summary), with a brief header
                        let last_result = task_results.last().unwrap();
                        if task_results.len() == 1 {
                            last_result.clone()
                        } else {
                            // Show count and the final result
                            format!(
                                "{}/{} tasks completed.\n\n{}",
                                completed_tasks
                                    .iter()
                                    .filter(|t| t.status == "completed" && t.error.is_none())
                                    .count(),
                                completed_tasks.len(),
                                last_result
                            )
                        }
                    };

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
                        let failed = summary.get("failed").and_then(|v| v.as_u64()).unwrap_or(0);
                        let blocked = summary.get("blocked").and_then(|v| v.as_u64()).unwrap_or(0);
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
                        result
                            .as_ref()
                            .err()
                            .map(|e| e.to_string())
                            .unwrap_or_else(|| {
                                "task lead exited without completing all tasks".to_string()
                            })
                    ),
                ),
            }
        };

        // Skip completion notification if we already delivered the result directly
        // to avoid sending the same content twice. Still notify for failures/stalls
        // since those carry different information.
        if delivered_directly && notification_type == "completed" {
            let _ = state.mark_goal_notified(&goal_id).await;
            if let Some(ref registry) = goal_token_registry {
                registry.remove(&goal_id).await;
            }
            return;
        }

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
        provider: Arc<dyn ModelProvider>,
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
        models_config: ModelsConfig,
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
    ) -> Self {
        init_policy_tunables_once(policy_config.uncertainty_clarify_threshold);
        let fallback = model.clone();
        let router = Router::new(models_config);
        let router = if router.is_uniform() {
            info!("All model tiers identical, auto-routing disabled");
            None
        } else {
            info!(
                fast = router.select(crate::router::Tier::Fast),
                primary = router.select(crate::router::Tier::Primary),
                smart = router.select(crate::router::Tier::Smart),
                "Smart router enabled"
            );
            Some(router)
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
            provider,
            state,
            event_store,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skills_dir,
            depth: 0,
            max_depth,
            iteration_config,
            max_iterations,
            max_iterations_cap,
            max_response_chars,
            timeout_secs,
            max_facts,
            router,
            model_override: RwLock::new(false),
            daily_token_budget,
            llm_call_timeout: llm_call_timeout_secs.map(Duration::from_secs),
            task_timeout: task_timeout_secs.map(Duration::from_secs),
            task_token_budget,
            verification_tracker: Some(Arc::new(VerificationTracker::new())),
            mcp_registry,
            role: AgentRole::Orchestrator,
            v3_task_id: None,
            v3_goal_id: None,
            cancel_token: None,
            goal_token_registry,
            hub: RwLock::new(hub),
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
            classify_query_retired: AtomicBool::new(false),
            last_graduation_check_epoch: AtomicU64::new(0),
            root_tools: None, // Root agent — its own tools ARE the root tools
            record_decision_points,
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

    /// Create an Agent with explicit depth/max_depth (used internally for sub-agents).
    /// Sub-agents don't auto-route — they use whatever model was selected by the parent.
    #[allow(clippy::too_many_arguments)]
    fn with_depth(
        provider: Arc<dyn ModelProvider>,
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
        v3_task_id: Option<String>,
        v3_goal_id: Option<String>,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
        goal_token_registry: Option<GoalTokenRegistry>,
        hub: Option<Weak<ChannelHub>>,
        record_decision_points: bool,
        context_window_config: crate::config::ContextWindowConfig,
        policy_config: PolicyConfig,
        root_tools: Option<Vec<Arc<dyn Tool>>>,
    ) -> Self {
        let fallback = model.clone();
        Self {
            provider,
            state,
            event_store,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skills_dir,
            depth,
            max_depth,
            iteration_config,
            max_iterations,
            max_iterations_cap,
            max_response_chars,
            timeout_secs,
            max_facts,
            router: None,
            model_override: RwLock::new(false),
            daily_token_budget: None,
            llm_call_timeout,
            task_timeout,
            task_token_budget,
            verification_tracker,
            mcp_registry,
            role,
            v3_task_id,
            v3_goal_id,
            cancel_token,
            goal_token_registry,
            hub: RwLock::new(hub),
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
            classify_query_retired: AtomicBool::new(false),
            last_graduation_check_epoch: AtomicU64::new(0),
            root_tools,
            record_decision_points,
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

    /// V3 role for this agent instance.
    pub fn role(&self) -> AgentRole {
        self.role
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

    // ==================== V3 Orchestration Methods ====================

    /// Run the agentic loop for a user message in the given session.
    /// Returns the final assistant text response.
    /// `heartbeat` is an optional atomic timestamp updated on each activity point.
    /// Channels pass `Some(heartbeat)` so the typing indicator can detect stalls;
    /// sub-agents, triggers, and tests pass `None`.
    pub async fn handle_message(
        &self,
        session_id: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String> {
        self.handle_message_impl(
            session_id,
            user_text,
            status_tx,
            user_role,
            channel_ctx,
            heartbeat,
        )
        .await
    }
}

#[cfg(test)]
mod message_ordering_tests;

#[cfg(test)]
mod heartbeat_tests;

#[cfg(test)]
mod tool_watchdog_tests;

#[cfg(test)]
mod group_session_tests;

#[cfg(test)]
mod consultant_prompt_tests;

#[cfg(test)]
mod resume_checkpoint_tests;

#[cfg(test)]
mod v3_intent_tests;

#[cfg(test)]
mod v3_tool_scoping_tests;

#[cfg(test)]
mod file_path_extraction_tests;

#[cfg(test)]
mod policy_signal_tests;
