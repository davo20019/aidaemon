use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::Utc;
use croner::Cron;
use serde_json::{json, Value};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

use crate::channels::ChannelHub;
use crate::config::{IterationLimitConfig, PathAliasConfig, PolicyConfig};
use crate::events::{
    AssistantResponseData, DecisionPointData, DecisionType, ErrorData, EventStore, EventType,
    LlmCallData, SubAgentCompleteData, SubAgentSpawnData, TaskEndData, TaskStartData, TaskStatus,
    ThinkingStartData, ToolCallData, ToolCallInfo, ToolResultData,
};
use crate::execution_policy::{ApprovalMode, ExecutionPolicy, ModelProfile};
use crate::goal_tokens::GoalTokenRegistry;
use crate::llm_runtime::SharedLlmRuntime;
use crate::mcp::McpRegistry;
use crate::providers::{ProviderError, ProviderErrorKind};
use crate::router::{self, Router};
use crate::skills::{self, MemoryContext};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::VerificationTracker;
use crate::traits::{
    AgentRole, ChatOptions, Goal, Message, ModelProvider, StateStore, TaskActivity, Tool, ToolCall,
    ToolCapabilities, ToolChoiceMode, ToolRole,
};
use crate::types::{ApprovalResponse, ChannelContext, ChannelVisibility, UserRole};
// Re-export StatusUpdate from types for backwards compatibility
pub use crate::types::StatusUpdate;
// `Task` is consumed only by in-tree `#[cfg(test)]` modules that import the
// agent namespace via `use super::*` (e.g. `runtime/goal_failure_tests.rs`).
#[cfg(test)]
use crate::traits::Task;

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
use response_analysis::{
    claims_completed_side_effect, claims_delegation_started, is_substantive_text_response,
    looks_like_deferred_action_response, looks_like_multi_part_request,
};
#[path = "intent/keywords.rs"]
mod intent_keywords;
#[path = "intent/intent_routing.rs"]
mod intent_routing;
#[path = "intent/llm_classifier.rs"]
pub mod llm_classifier;
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
#[allow(unused_imports)]
pub(in crate::agent) use validation_state::{
    build_abandon_request, build_partial_done_blocked_request,
    build_partial_done_blocked_request_with_plan, build_reduce_scope_request,
    build_reduce_scope_request_with_plan, ApprovalState, LoopRepetitionReason, ValidationFailure,
    ValidationOutcome,
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
#[path = "runtime/task_outcome.rs"]
mod task_outcome;
use post_task::LearningContext;
pub(in crate::agent) use post_task::ReplayNoteCategory;
pub(in crate::agent) use task_outcome::{
    response_has_user_value, response_looks_like_plain_text_tool_call, TaskOutcomeDerivation,
};
#[allow(dead_code, unused_imports)]
#[path = "loop/state/mod.rs"]
mod loop_state;
#[path = "loop/stopping_conditions.rs"]
mod stopping_conditions;
#[path = "loop/tool_loop_state.rs"]
mod tool_loop_state;

#[path = "loop/bootstrap_phase.rs"]
mod bootstrap_phase;
#[path = "loop/completion_checks.rs"]
mod completion_checks;
#[path = "runtime/completion_contract.rs"]
mod completion_contract;
#[path = "loop/completion_phase.rs"]
mod completion_phase;
#[path = "runtime/core_prompt.rs"]
mod core_prompt;
#[path = "runtime/dialogue_state.rs"]
mod dialogue_state;
#[path = "loop/direct_return.rs"]
mod direct_return;
#[path = "loop/fallthrough.rs"]
mod fallthrough;
#[path = "runtime/followup.rs"]
mod followup;
#[path = "runtime/graceful.rs"]
mod graceful;
#[path = "runtime/history.rs"]
mod history;
#[path = "runtime/notes.rs"]
mod notes;
#[path = "loop/orchestration_phase.rs"]
mod orchestration_phase;
#[path = "runtime/parent_delivery.rs"]
mod parent_delivery;
#[path = "runtime/project_scope.rs"]
mod project_scope;
pub(crate) mod specialists;
#[path = "runtime/turn_context.rs"]
mod turn_context;
pub(crate) use parent_delivery::ParentDeliveryKind;
#[path = "loop/response_phase.rs"]
mod response_phase;
#[path = "loop/services.rs"]
mod services;
pub(in crate::agent) use history::CompletionContract;
pub(in crate::agent) use history::CompletionProgress;
pub(in crate::agent) use history::CompletionTaskKind;
pub(in crate::agent) use history::FollowupMode;
pub(in crate::agent) use history::TurnContext;
pub(in crate::agent) use history::VerificationTarget;
pub(in crate::agent) use history::VerificationTargetKind;
#[path = "loop/compaction.rs"]
mod compaction;
#[path = "runtime/llm.rs"]
mod llm;
pub(in crate::agent) use llm::LlmCallTelemetry;
#[path = "loop/llm_phase.rs"]
mod llm_phase;
#[path = "loop/main_loop.rs"]
mod main_loop;
#[path = "loop/message_build_phase.rs"]
mod message_build_phase;
#[path = "runtime/models.rs"]
mod models;
#[path = "loop/prefix_fingerprint.rs"]
mod prefix_fingerprint;
#[path = "loop/request_dump.rs"]
mod request_dump;
#[path = "runtime/resume.rs"]
mod resume;
#[path = "loop/sliding_window.rs"]
mod sliding_window;
#[path = "runtime/spawn.rs"]
mod spawn;
#[path = "loop/stopping_helpers.rs"]
mod stopping_helpers;
#[path = "loop/stopping_phase.rs"]
mod stopping_phase;
#[path = "loop/stopping_progress.rs"]
mod stopping_progress;
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
#[path = "loop/turn_eviction.rs"]
pub(crate) mod turn_eviction;
#[path = "loop/turn_render.rs"]
pub(crate) mod turn_render;
#[path = "loop/turn_render_cache.rs"]
pub(crate) mod turn_render_cache;

pub(in crate::agent) use system_directives::{EarlyStopSeverity, SystemDirective};
use system_prompt::format_goal_context;
pub(in crate::agent) use tool_result_notices::ToolResultNotice;

// Policy runtime metrics, route-drift monitor, and bounded autotuning.
// Implementation lives in `policy_metrics.rs` (Phase 5 decoupling).
#[path = "policy_metrics.rs"]
mod policy_metrics;
pub use policy_metrics::{
    apply_bounded_autotune_from_failure_ratio, init_policy_tunables_once, policy_autotune_snapshot,
    policy_metrics_snapshot,
};
// `PolicyAutotuneSnapshot` is part of the public surface but currently has no
// in-tree consumer (the legacy definition carried `#[allow(dead_code)]`).
pub(in crate::agent) use policy_metrics::record_failed_task_tokens;
#[cfg(test)]
pub(crate) use policy_metrics::set_route_failsafe_for_session_for_test;
#[allow(unused_imports)]
pub use policy_metrics::PolicyAutotuneSnapshot;
pub(in crate::agent) use policy_metrics::{
    current_uncertainty_threshold, provider_kind_metric_label, record_llm_payload_invalid_metric,
    route_failsafe_active_for_session, POLICY_METRICS,
};
// `observe_route_reason_for_drift` / `RouteDriftSignal` are wired but currently
// unreferenced in-tree (the legacy definitions carried `#[allow(dead_code)]`).
#[allow(unused_imports)]
pub(in crate::agent) use policy_metrics::{observe_route_reason_for_drift, RouteDriftSignal};

// Free-function helpers (status, resume, project-scope cues, untrusted-reference
// filtering, intent-gate merge). Implementation lives in `agent_helpers.rs`.
#[path = "agent_helpers.rs"]
mod agent_helpers;
pub(in crate::agent) use agent_helpers::{
    build_empty_response_fallback, filter_tool_defs_for_untrusted_external_reference,
    is_resume_request, is_untrusted_external_reference_blocked_tool,
    matched_untrusted_external_reference_skill_names,
    should_allow_contextual_project_nickname_scope, summarize_tool_args,
    text_has_explicit_project_scope_cues, truncate_for_resume,
    user_explicitly_requests_local_file_inspection, user_text_references_filesystem_path,
    IntentGateDecision, ResumeCheckpoint, ResumeExecutionSnapshot,
};
pub use agent_helpers::{send_status, touch_heartbeat};

/// Phase 0 per-session window-boundary memory: `session_id` →
/// (last `keep_from` index, last oldest-kept persisted message id). Used by
/// `message_build_phase` to emit an explicit boundary-movement event.
type WindowBoundaryMemory = Arc<tokio::sync::RwLock<HashMap<String, (usize, Option<String>)>>>;

/// Pillar A per-session core-prompt cache: `session_id` → last rendered
/// [`core_prompt::CachedCore`]. On a HIT (aggregate hash unchanged) the cached
/// bytes are reused verbatim with no re-render; on a MISS the changed
/// component(s) are logged (`Core prompt invalidated component=...`) and the
/// entry is replaced. In-memory, lost on restart.
type CorePromptCache = Arc<tokio::sync::RwLock<HashMap<String, core_prompt::CachedCore>>>;

/// Pillar B per-session/per-turn render cache: `session_id` → (`turn_id` →
/// [`turn_render_cache::CachedRender`]). Reuses rendered turn bytes verbatim
/// when `content_fp` + `renderer_version` + `mode` all match; on a miss the
/// turn is re-rendered and the entry replaced. In-memory, lost on restart.
type TurnRenderCache =
    Arc<tokio::sync::RwLock<HashMap<String, HashMap<String, turn_render_cache::CachedRender>>>>;

/// Pillar B per-session in-memory anchor: `session_id` → anchor `turn_seq`. The
/// archived region carried into each payload starts at this turn; eviction
/// advances it. In-memory, lost on restart (one re-prefill per restart is
/// accepted). Consumed by the message-build integration in Task 7; unused until
/// then.
type TurnAnchorMemory = Arc<tokio::sync::RwLock<HashMap<String, i64>>>;

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
    /// Static limits, budgets, and timeout settings.
    limits: AgentLimits,
    /// When true, the user has manually set a model via /model — skip auto-routing.
    model_override: RwLock<bool>,
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
    /// Models that recently returned 402 billing errors. Maps model name → failure time.
    /// Shared across parent/child agents. Entries expire after BILLING_FAIL_CACHE_TTL.
    billing_failed_models: Arc<tokio::sync::RwLock<HashMap<String, Instant>>>,
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
    /// Per-session conversation turn IDs. Populated at the start of each
    /// `handle_message_impl` and read by `append_message_canonical` so every
    /// message written during a turn is stamped with the same `turn_id`.
    /// `message_build_phase` uses this stamp to find the current-task
    /// boundary without inferring from message content.
    current_turn_ids: Arc<tokio::sync::RwLock<HashMap<String, String>>>,
    /// Phase 0 observability — per-session memory of the last sliding-window
    /// trim decision (`keep_from` index + oldest-kept persisted message id).
    /// Lets `message_build_phase` emit an explicit movement event when the
    /// window boundary shifts between builds, which is the prime suspect for
    /// llama.cpp prefix-cache breaks. In-memory, lost on restart (the
    /// server-side KV cache is stale then anyway).
    window_keep_from_tracker: WindowBoundaryMemory,
    /// Test-only override for the execution budget selected at the start of
    /// the agent loop. When `Some`, `select_initial_execution_budget` is
    /// bypassed and this budget is used instead.
    #[cfg(test)]
    execution_budget_override: Option<ExecutionBudget>,
    /// Per-process specialist registry (bundled + user-override `.md` files).
    /// Loaded once at agent construction and shared with every child agent
    /// spawned from this hierarchy.
    pub(crate) specialists: Arc<crate::agent::specialists::SpecialistRegistry>,
    /// Pillar A per-session core-prompt cache (Task 7). Keyed by `session_id`;
    /// reuses rendered core bytes verbatim across tasks when the session-static
    /// inputs are unchanged, and logs the changed component on invalidation.
    core_prompts: CorePromptCache,
    /// Pillar B per-session/per-turn render cache (Task 5). Keyed by
    /// `session_id` then `turn_id`; reuses rendered turn bytes verbatim across
    /// builds when `content_fp` + `renderer_version` + `mode` are unchanged.
    /// Consumed by the message-build integration in Task 7 (Pillar B); unused
    /// until then.
    #[allow(dead_code)]
    turn_renders: TurnRenderCache,
    /// Pillar B per-session anchor tracker (Task 6). Keyed by `session_id`;
    /// maps to the anchor `turn_seq` of the oldest archived turn carried into
    /// the payload. Eviction advances it. Consumed by the message-build
    /// integration in Task 7 (Pillar B); unused until then.
    #[allow(dead_code)]
    turn_anchors: TurnAnchorMemory,
}

struct AgentLimits {
    max_depth: usize,
    iteration_config: IterationLimitConfig,
    #[allow(dead_code)]
    max_iterations: usize,
    #[allow(dead_code)]
    max_iterations_cap: usize,
    max_response_chars: usize,
    timeout_secs: u64,
    max_facts: usize,
    daily_token_budget: Option<u64>,
    llm_call_timeout: Option<Duration>,
    task_timeout: Option<Duration>,
    task_token_budget: Option<u64>,
}

impl AgentLimits {
    /// Absolute timeout ceiling (seconds) a spawned specialist may request.
    ///
    /// `timeout_secs` defaults to 300, but a configured `0` means "no parent
    /// timeout"; in that case we still impose a 1-hour implicit cap so a
    /// specialist cannot request an unbounded timeout. Encapsulating this here
    /// keeps the `> 0` guard in one place instead of inline at each call site.
    fn timeout_cap(&self) -> u64 {
        if self.timeout_secs > 0 {
            self.timeout_secs
        } else {
            3600
        }
    }
}

// Goal/task dispatch helpers. Implementation lives in `goal_dispatch.rs`.
#[path = "goal_dispatch.rs"]
mod goal_dispatch;
pub use goal_dispatch::is_group_session;
pub(in crate::agent) use goal_dispatch::{
    active_scheduled_root_task_id, auto_dispatch_scheduled_run_extension_budget,
    clear_scheduled_run_state, effective_goal_daily_budget, goal_has_scheduled_provenance,
    is_low_signal_task_lead_reply, looks_like_evidence_grounding_challenge,
    looks_like_false_capability_denial_after_tool_success, looks_like_incomplete_live_work_summary,
    parse_goal_leading_wait, parse_wait_task_seconds, persist_scheduled_run_state,
    salvageable_task_lead_result, strip_leading_wait, task_has_scheduled_provenance,
    truncate_goal_result_text, user_facing_task_description,
};
pub(crate) use goal_dispatch::{
    build_goal_failure_summary, build_goal_task_results_summary, extract_file_paths_from_text,
    goal_completion_response_indicates_incomplete_work,
};

// Background task-lead spawner. Implementation lives in `background_task_lead.rs`.
#[path = "background_task_lead.rs"]
mod background_task_lead;
pub use background_task_lead::spawn_background_task_lead;

// `Agent` constructors (new / with_depth / set_test_*) live in `construct.rs`.
#[path = "construct.rs"]
mod construct;

// impl-Agent justification: public entry surface (handle_message, hub/self-ref wiring, role/depth accessors, goal cancellation) — Agent's API to channels and core.
impl Agent {
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
        self.limits.max_depth
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
        self.limits.max_iterations
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
        // NOTE: The primary sanitization pass already runs in completion_phase.rs.
        // This second pass is a lightweight safety net that only strips control
        // markers that the model may echo verbatim.  We intentionally do NOT
        // call the full `sanitize_user_facing_reply` again — double sanitization
        // can reduce a valid reply to empty when the first pass already transformed
        // tool-name references into generic prose that the second pass then matches
        // as a different pattern.
        crate::tools::sanitize::strip_leaked_control_markers(reply)
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
            task_outcome: None,
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
