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
use crate::config::{IterationLimitConfig, PolicyConfig};
use crate::events::{
    AssistantResponseData, DecisionPointData, DecisionType, ErrorData, EventStore, EventType,
    PolicyMetricsData, SubAgentCompleteData, SubAgentSpawnData, TaskEndData, TaskStartData,
    TaskStatus, ThinkingStartData, ToolCallData, ToolCallInfo, ToolResultData,
};
use crate::execution_policy::{ApprovalMode, ExecutionPolicy, ModelProfile};
use crate::goal_tokens::GoalTokenRegistry;
use crate::llm_markers::{CONSULTANT_TEXT_ONLY_MARKER, INTENT_GATE_MARKER};
use crate::llm_runtime::SharedLlmRuntime;
use crate::mcp::McpRegistry;
use crate::providers::{ProviderError, ProviderErrorKind};
use crate::router::{self, Router};
use crate::skills::{self, MemoryContext};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::VerificationTracker;
use crate::traits::{
    AgentRole, Goal, Message, ModelProvider, StateStore, TaskActivity, Tool, ToolCall,
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
/// Legacy fallback schedule text heuristics used as a guardrail when the model
/// omits schedule fields. These heuristics explicitly ignore "tell me about this
/// scheduled goal" meta-queries to avoid accidental schedule creation.
const ENABLE_SCHEDULE_HEURISTICS: bool = true;

#[path = "intent/intent_gate.rs"]
mod intent_gate;
use intent_gate::extract_intent_gate;
#[cfg(test)]
use intent_gate::parse_intent_gate_json;
#[path = "consultant/consultant_analysis.rs"]
mod consultant_analysis;
#[path = "consultant/consultant_pass.rs"]
mod consultant_pass;
#[cfg(test)]
use consultant_analysis::has_action_promise;
use consultant_analysis::{looks_like_deferred_action_response, sanitize_consultant_analysis};
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
use policy_signals::{
    build_policy_bundle, default_clarifying_question, detect_explicit_outcome_signal,
    is_short_user_correction, tool_is_side_effecting,
};
#[path = "loop/loop_utils.rs"]
mod loop_utils;
#[path = "policy/recall_guardrails.rs"]
mod recall_guardrails;
use loop_utils::{
    build_task_boundary_hint, extract_command_from_args, extract_file_path_from_args,
    extract_send_file_dedupe_key_from_args, fixup_message_ordering, hash_tool_call,
    is_trigger_session, strip_appended_diagnostics,
};
#[path = "runtime/post_task.rs"]
mod post_task;
use post_task::LearningContext;
#[path = "loop/stopping_conditions.rs"]
mod stopping_conditions;
#[path = "loop/tool_loop_state.rs"]
mod tool_loop_state;

#[path = "loop/bootstrap_phase.rs"]
mod bootstrap_phase;
#[path = "loop/consultant_completion_phase.rs"]
mod consultant_completion_phase;
#[path = "loop/consultant_decision_phase.rs"]
mod consultant_decision_phase;
#[path = "loop/consultant_direct_return.rs"]
mod consultant_direct_return;
#[path = "loop/consultant_fallthrough.rs"]
mod consultant_fallthrough;
#[path = "loop/consultant_intent_gate_phase.rs"]
mod consultant_intent_gate_phase;
#[path = "loop/consultant_orchestration_phase.rs"]
mod consultant_orchestration_phase;
#[path = "loop/consultant_phase.rs"]
mod consultant_phase;
#[path = "runtime/graceful.rs"]
mod graceful;
#[path = "runtime/history.rs"]
mod history;
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

use system_prompt::{build_consultant_system_prompt, format_goal_context, ConsultantPromptStyle};

#[cfg(test)]
use system_prompt::strip_markdown_section;

struct PolicyRuntimeMetrics {
    tool_exposure_samples: AtomicU64,
    tool_exposure_before_sum: AtomicU64,
    tool_exposure_after_sum: AtomicU64,
    ambiguity_detected_total: AtomicU64,
    uncertainty_clarify_total: AtomicU64,
    context_refresh_total: AtomicU64,
    escalation_total: AtomicU64,
    fallback_expansion_total: AtomicU64,
    consultant_direct_return_total: AtomicU64,
    consultant_fallthrough_total: AtomicU64,
    consultant_route_clarification_required_total: AtomicU64,
    consultant_route_tools_required_total: AtomicU64,
    consultant_route_short_correction_direct_reply_total: AtomicU64,
    consultant_route_acknowledgment_direct_reply_total: AtomicU64,
    consultant_route_default_continue_total: AtomicU64,
    tokens_failed_tasks_total: AtomicU64,
    no_progress_iterations_total: AtomicU64,
}

impl PolicyRuntimeMetrics {
    const fn new() -> Self {
        Self {
            tool_exposure_samples: AtomicU64::new(0),
            tool_exposure_before_sum: AtomicU64::new(0),
            tool_exposure_after_sum: AtomicU64::new(0),
            ambiguity_detected_total: AtomicU64::new(0),
            uncertainty_clarify_total: AtomicU64::new(0),
            context_refresh_total: AtomicU64::new(0),
            escalation_total: AtomicU64::new(0),
            fallback_expansion_total: AtomicU64::new(0),
            consultant_direct_return_total: AtomicU64::new(0),
            consultant_fallthrough_total: AtomicU64::new(0),
            consultant_route_clarification_required_total: AtomicU64::new(0),
            consultant_route_tools_required_total: AtomicU64::new(0),
            consultant_route_short_correction_direct_reply_total: AtomicU64::new(0),
            consultant_route_acknowledgment_direct_reply_total: AtomicU64::new(0),
            consultant_route_default_continue_total: AtomicU64::new(0),
            tokens_failed_tasks_total: AtomicU64::new(0),
            no_progress_iterations_total: AtomicU64::new(0),
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

pub fn policy_metrics_snapshot() -> PolicyMetricsData {
    PolicyMetricsData {
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
        consultant_direct_return_total: POLICY_METRICS
            .consultant_direct_return_total
            .load(Ordering::Relaxed),
        consultant_fallthrough_total: POLICY_METRICS
            .consultant_fallthrough_total
            .load(Ordering::Relaxed),
        consultant_route_clarification_required_total: POLICY_METRICS
            .consultant_route_clarification_required_total
            .load(Ordering::Relaxed),
        consultant_route_tools_required_total: POLICY_METRICS
            .consultant_route_tools_required_total
            .load(Ordering::Relaxed),
        consultant_route_short_correction_direct_reply_total: POLICY_METRICS
            .consultant_route_short_correction_direct_reply_total
            .load(Ordering::Relaxed),
        consultant_route_acknowledgment_direct_reply_total: POLICY_METRICS
            .consultant_route_acknowledgment_direct_reply_total
            .load(Ordering::Relaxed),
        consultant_route_default_continue_total: POLICY_METRICS
            .consultant_route_default_continue_total
            .load(Ordering::Relaxed),
        tokens_failed_tasks_total: POLICY_METRICS
            .tokens_failed_tasks_total
            .load(Ordering::Relaxed),
        no_progress_iterations_total: POLICY_METRICS
            .no_progress_iterations_total
            .load(Ordering::Relaxed),
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
            let prompt = get_str("prompt").unwrap_or("");
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
        "send_file" => get_str("path")
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
    /// Weak self-reference for background task spawning.
    /// Set after Arc creation via `set_self_ref()`.
    self_ref: RwLock<Option<Weak<Agent>>>,
    /// Context window management configuration.
    context_window_config: crate::config::ContextWindowConfig,
    /// Policy rollout and enforcement configuration.
    policy_config: PolicyConfig,
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
            let max_dispatch_rounds = 4; // safety limit — keep low to bound token usage
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
                    // Stop dispatching as soon as the goal hits its daily token budget.
                    if let Ok(Some(g)) = state.get_goal(&goal_id).await {
                        if let Some(budget_daily) = g.budget_daily {
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
                    let _ = state.update_task(&updated).await;
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

            let goal_budget_exhausted = updated_goal
                .budget_daily
                .is_some_and(|b| updated_goal.tokens_used_today >= b);

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
            } else if goal_budget_exhausted {
                // Budget exhausted is a safety stop, not "no progress". Keep the goal active
                // and avoid stalling it; it can resume after budgets reset.
                updated_goal.dispatch_failures = 0;
                info!(
                    goal_id = %goal_id,
                    tokens_used = updated_goal.tokens_used_today,
                    budget = updated_goal.budget_daily.unwrap_or(0),
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
                            response.chars().take(4000).collect::<String>()
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
            match status {
                "completed" => {
                    // Build notification from actual task results, not the task lead's
                    // planning message. The task lead response is just a plan outline;
                    // the real outputs come from the executor tasks.
                    let completed_tasks =
                        state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();

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
    ) -> Self {
        init_policy_tunables_once(policy_config.uncertainty_clarify_threshold);
        let fallback = model.clone();
        if let Some(router) = llm_runtime.router() {
            info!(
                fast = router.select(crate::router::Tier::Fast),
                primary = router.select(crate::router::Tier::Primary),
                smart = router.select(crate::router::Tier::Smart),
                "Smart router enabled"
            );
        } else {
            info!("All model tiers identical, auto-routing disabled");
        }

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
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
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

    #[cfg(test)]
    pub fn set_test_task_token_budget(&mut self, budget: Option<u64>) {
        self.task_token_budget = budget;
    }

    #[cfg(test)]
    pub fn set_test_goal_id(&mut self, goal_id: Option<String>) {
        self.goal_id = goal_id;
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
        record_decision_points: bool,
        context_window_config: crate::config::ContextWindowConfig,
        policy_config: PolicyConfig,
        root_tools: Option<Vec<Arc<dyn Tool>>>,
    ) -> Self {
        let fallback = model.clone();
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
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
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

    // ==================== Orchestration Methods ====================

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
        let reply = self
            .handle_message_impl(
                session_id,
                user_text,
                status_tx,
                user_role,
                channel_ctx,
                heartbeat,
            )
            .await?;

        // Strip internal context markers that the LLM may echo back.
        // These markers are injected into old assistant messages to help the
        // model distinguish prior-turn context, but must never leak to users.
        let reply = reply
            .replace(" [prior turn, truncated]", "")
            .replace(" [prior turn]", "")
            .replace("[prior turn, truncated]", "")
            .replace("[prior turn]", "");

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
            }

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
mod tests;
