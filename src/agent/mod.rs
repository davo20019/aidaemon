use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::Utc;
use serde_json::{json, Value};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

use crate::channels::ChannelHub;
use crate::config::{
    AudioConfig, IterationLimitConfig, PathAliasConfig, PolicyConfig, SttConfig, VisionConfig,
};
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
    AgentRole, ChatOptions, Message, ModelProvider, StateStore, TaskActivity, Tool, ToolCall,
    ToolCapabilities, ToolChoiceMode, ToolRole,
};
use crate::types::{ApprovalResponse, ChannelContext, ChannelVisibility, UserRole};
// Re-export StatusUpdate from types for backwards compatibility
pub use crate::types::StatusUpdate;
// `Task` is consumed only by in-tree `#[cfg(test)]` modules that import the
// agent namespace via `use super::*` (e.g. `runtime/goal_failure_tests.rs`).
#[cfg(test)]
use crate::traits::Task;

type CorrectionContext = Arc<crate::agent::correction_execution::CorrectionExecutionContext>;

#[derive(Default)]
struct CorrectionContextRegistry {
    contexts: HashMap<String, CorrectionContext>,
    insertion_order: VecDeque<String>,
}

impl CorrectionContextRegistry {
    fn insert(&mut self, goal_id: String, ctx: CorrectionContext) {
        if !self.contexts.contains_key(&goal_id) {
            self.insertion_order.push_back(goal_id.clone());
        }
        self.contexts.insert(goal_id, ctx);
        self.evict_oldest_until_bounded();
    }

    fn get(&self, goal_id: &str) -> Option<CorrectionContext> {
        self.contexts.get(goal_id).cloned()
    }

    fn remove(&mut self, goal_id: &str) {
        self.contexts.remove(goal_id);
        self.insertion_order.retain(|existing| existing != goal_id);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.contexts.len()
    }

    fn evict_oldest_until_bounded(&mut self) {
        while self.contexts.len() > MAX_CORRECTION_CONTEXTS {
            if let Some(stale) = self.insertion_order.pop_front() {
                self.contexts.remove(&stale);
            } else {
                break;
            }
        }
    }
}

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
/// The adjacent prior assistant answer is recent transcript, not generic
/// archive. Keep ordinary answers verbatim up to roughly 3k tokens; only
/// genuinely oversized answers use deterministic head/tail compaction.
const MAX_ADJACENT_ASSISTANT_CONTENT_CHARS: usize = 12_000;
/// Higher cap for archived assistant messages that end in a clarifying
/// question/menu: the options ARE the actionable state the next user turn
/// refers to ("Yes do 1, 2"), so they must survive archival. Content over the
/// cap keeps the question head and the option tail and elides the middle.
const MAX_CLARIFYING_ASSISTANT_CONTENT_CHARS: usize = 2000;
/// Window size for detecting alternating tool patterns (A-B-A-B cycles).
const ALTERNATING_PATTERN_WINDOW: usize = 10;
const PROGRESS_SUMMARY_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes
pub mod activity_gate;
#[path = "response_analysis.rs"]
mod response_analysis;
#[cfg(test)]
use response_analysis::has_action_promise;
#[cfg(test)]
use response_analysis::sanitize_response_analysis;
use response_analysis::{
    claims_completed_side_effect, claims_delegation_started, is_substantive_text_response,
    looks_like_deferred_action_response, looks_like_incomplete_retry_plan,
    looks_like_multi_part_request, reply_is_pasted_file_page,
};
#[cfg(test)]
use response_analysis::{reply_defers_file_access, user_text_references_file};
#[path = "intent/keywords.rs"]
mod intent_keywords;
#[path = "intent/intent_routing.rs"]
mod intent_routing;
#[path = "intent/llm_classifier.rs"]
pub mod llm_classifier;
#[path = "intent/relational_prefilter.rs"]
pub mod relational_prefilter;
use intent_routing::contains_keyword_as_words;
#[cfg(test)]
use intent_routing::infer_intent_gate;
// Re-export for use outside the `agent` subtree (e.g. `tools/browser/policy`).
pub(crate) use intent_routing::contains_keyword_as_words as keyword_match;
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
#[path = "loop/approach_pivot.rs"]
mod approach_pivot;
#[path = "policy/heuristic_telemetry.rs"]
pub(crate) mod heuristic_telemetry;
#[path = "loop/loop_utils.rs"]
mod loop_utils;
#[path = "policy/recall_guardrails.rs"]
mod recall_guardrails;
#[path = "policy/trust_tier.rs"]
pub(crate) mod trust_tier;
use loop_utils::{
    build_task_boundary_hint, classify_execution_failure_kind, classify_tool_outcome_status,
    extract_command_from_args, extract_file_path_from_args, extract_key_error_line,
    extract_send_file_dedupe_key_from_args, failure_class_from_outcome_status,
    fixup_message_ordering, hash_tool_call, is_trigger_session, semantic_failure_limit,
    strip_appended_diagnostics, tool_outcome_evidence_source, ExecutionFailureKind,
    ToolFailureClass,
};
#[path = "runtime/post_task.rs"]
mod post_task;
#[path = "runtime/task_outcome.rs"]
mod task_outcome;
pub(crate) use post_task::is_friendly_background_handoff;
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

#[cfg(feature = "computer_use")]
mod computer_use;

#[path = "loop/answer_grounding.rs"]
mod answer_grounding;
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
#[path = "runtime/followup.rs"]
mod followup;
#[path = "runtime/graceful.rs"]
mod graceful;
#[path = "runtime/history.rs"]
mod history;
#[path = "runtime/notes.rs"]
mod notes;
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
pub(in crate::agent) use history::ExecutionRequirement;
pub(in crate::agent) use history::FollowupMode;
pub(in crate::agent) use history::ForbiddenMutationAction;
pub(in crate::agent) use history::TurnContext;
pub(in crate::agent) use history::VerificationTarget;
pub(in crate::agent) use history::VerificationTargetKind;
pub(in crate::agent) use history::{
    apply_planned_contract_signals, apply_planned_mutation_constraints,
    apply_planned_required_mutation_effects, parse_planned_forbidden_action,
    parse_planned_mutation_effects, parse_planned_task_kind,
};
pub(in crate::agent) use history::{
    authored_artifact_still_needs_delivery_recovery, completion_contract_allows_force_text,
    mutation_contract_fulfilled,
};
#[path = "runtime/llm.rs"]
mod llm;
pub(in crate::agent) use llm::LlmCallTelemetry;
pub(crate) mod attachment_content;
pub(crate) mod audio;
pub(in crate::agent) mod eval;
#[path = "loop/hand_holding_telemetry.rs"]
mod hand_holding_telemetry;
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
pub(crate) mod stt;
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
#[path = "loop/turn_transition.rs"]
mod turn_transition;
pub(crate) mod vision;

pub(in crate::agent) use eval::{
    HarnessEvalAccumulator, HarnessEvalConfig, HarnessEvalSeed, StopReason,
};
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
#[cfg(test)]
pub(in crate::agent) use agent_helpers::IntentGateDecision;
pub(in crate::agent) use agent_helpers::{
    build_empty_response_fallback, filter_tool_defs_for_untrusted_external_reference,
    is_resume_request, is_untrusted_external_reference_blocked_tool,
    matched_untrusted_external_reference_skill_names,
    should_allow_contextual_project_nickname_scope, summarize_tool_args,
    text_has_explicit_project_scope_cues, truncate_for_resume,
    user_explicitly_requests_local_file_inspection, user_text_references_filesystem_path,
    ResumeCheckpoint, ResumeExecutionSnapshot,
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

/// Immutable execution identity for one worker inside one autonomous mandate
/// decision cycle. The lease token is Rust-side control state and must never be
/// serialized, logged, or exposed to the model.
#[derive(Clone)]
pub(crate) struct MandateExecutionFence {
    pub(crate) mandate_id: String,
    pub(crate) mandate_version: i64,
    pub(crate) authority: crate::traits::MandateAuthority,
    pub(crate) goal_id: String,
    pub(crate) goal_run_id: String,
    pub(crate) root_task_id: String,
    pub(crate) root_task_attempt_id: String,
    pub(crate) worker_task_id: String,
    pub(crate) attempt_id: String,
    pub(crate) lease_token: String,
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
    /// Durable task ID for executor agents and run-bound mandate task leads.
    task_id: Option<String>,
    /// Goal ID for task lead agents — enables context injection into spawn calls.
    goal_id: Option<String>,
    /// Exact mandate run/task-attempt identity inherited at child creation.
    /// Unlike `goal_id`, this can never be rebound to a later decision cycle.
    mandate_execution: Option<MandateExecutionFence>,
    /// Cancellation token — checked each iteration; cancelled by parent or user.
    cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// Goal cancellation token registry — shared across agent hierarchy.
    goal_token_registry: Option<GoalTokenRegistry>,
    /// Weak reference to the ChannelHub for background notifications.
    /// Uses RwLock because hub is created after Agent (core.rs ordering).
    hub: RwLock<Option<Weak<ChannelHub>>>,
    /// Plan store for the requirement-checklist feature. Set after construction
    /// (core.rs ordering) via `set_plan_store`; `None` on subagents and in tests,
    /// in which case checklist verification/recap degrade to current behavior.
    plan_store: RwLock<Option<Arc<crate::plans::PlanStore>>>,
    /// Once-per-task guards for the requirement-checklist soft verification.
    /// Keys: "{task_id}:nudged" (nudged once) and "{task_id}:recapped" (recap
    /// posted once). Keeps the nudge/recap from firing on every iteration.
    checklist_turn_flags: Arc<tokio::sync::RwLock<HashSet<String>>>,
    /// Session IDs that have granted schedule confirmation for this process lifetime.
    /// Allows schedule creation to auto-confirm after an explicit AllowSession/AllowAlways.
    schedule_approved_sessions: Arc<tokio::sync::RwLock<HashSet<String>>>,
    /// Models that recently returned 402 billing errors. Maps model name → failure time.
    /// Shared across parent/child agents. Entries expire after BILLING_FAIL_CACHE_TTL.
    billing_failed_models: Arc<tokio::sync::RwLock<HashMap<String, Instant>>>,
    /// Models that ignored a forced `tool_choice=required` call (returned text
    /// with zero tool calls). Some serving stacks (llama.cpp + Gemma) don't
    /// enforce `required` and can degenerate into a repetition loop until the
    /// token limit when it is requested. Once flagged, the deferred/no-tool
    /// recovery stops forcing `required` for that model and relies on the
    /// substantive-text acceptance path instead. Shared across parent/child
    /// agents; in-memory only (re-learned after restart).
    required_tool_choice_ignored_models: Arc<tokio::sync::RwLock<HashSet<String>>>,
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
    /// Human-facing session that owns approvals for this internal child.
    /// Root agents leave this unset; every spawned descendant inherits the
    /// originating conversation while retaining its own isolated runtime ID.
    approval_session_id: Option<String>,
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
    /// DEPRECATED (Pillar B Task 7): retired. This tracked the legacy
    /// sliding-window `keep_from` boundary; the boundary signal is now the
    /// turn-anchor (`turn_anchors`). The age-ladder/sliding-window block that
    /// wrote here was deleted in Task 7, so the field is now unused (left in
    /// place to avoid churning the `Agent` constructor; remove in a later
    /// cleanup pass). No code reads or writes it.
    #[allow(dead_code)]
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
    /// Consumed by the message-build integration (Pillar B Task 7).
    turn_renders: TurnRenderCache,
    /// Pillar B per-session anchor tracker (Task 6). Keyed by `session_id`;
    /// maps to the anchor `turn_seq` of the oldest archived turn carried into
    /// the payload. Eviction advances it. Consumed by the message-build
    /// integration (Pillar B Task 7).
    turn_anchors: TurnAnchorMemory,
    /// llama.cpp interactive KV-cache slot for the main generation loop.
    /// `Some(interactive_slot)` only when slot routing is enabled on the primary
    /// provider; `None` otherwise (and for all sub-agents). When set, it is
    /// applied to the main-loop `ChatOptions.id_slot` so the interactive
    /// conversation pins a dedicated slot that background tasks cannot evict.
    interactive_slot: Option<u32>,
    /// Caches the salient fact IDs and person IDs for the core profile per session.
    /// Locks the selected facts on the first owner turn to prevent the profile from
    /// shifting within a session due to recall_count bumps.
    session_core_profile_ids: Arc<tokio::sync::RwLock<HashMap<String, Vec<String>>>>,
    /// Vision/image encoding settings for multimodal LLM requests.
    vision_config: VisionConfig,
    /// Native audio input settings for multimodal LLM requests.
    audio_config: AudioConfig,
    /// Local Whisper STT fallback when native audio is skipped.
    stt_config: SttConfig,
    /// Per-task harness effectiveness accumulator (cleared after TaskEnd).
    harness_eval: Arc<RwLock<Option<HarnessEvalAccumulator>>>,
    /// Harness eval scoring configuration.
    harness_eval_config: HarnessEvalConfig,
    /// Per-remediation correction-execution contexts, keyed by the synthetic
    /// remediation **goal id** (a fresh UUID minted in
    /// `correction_dispatch::dispatch_correction_remediation`). Shared across the
    /// whole agent hierarchy (parent → child task-lead → executors) via `Arc`,
    /// so every loop running under a given remediation goal id can read the same
    /// `Arc<CorrectionExecutionContext>` and thread it into its
    /// `ToolExecutionCtx.correction`.
    ///
    /// Safety / non-stickiness:
    /// - The key is a globally-unique UUID associated with exactly ONE
    ///   deliberately-dispatched remediation goal. No user turn, no other goal,
    ///   and no other session ever carries that `goal_id`, so the per-call
    ///   correction gate can only ever fire for the remediation task it was
    ///   dispatched for — every other turn reads `None`.
    /// - Reads CLONE the `Arc` (they do not remove it) so the gate fires on the
    ///   remediation task-lead AND all of its executors (which inherit the same
    ///   `goal_id`). This is NOT a session-global or sticky flag: it is scoped
    ///   to a single unique goal id.
    /// - The map is bounded (oldest entries evicted on insert) so a fire-and-forget
    ///   background spawn that never reports completion cannot leak unboundedly.
    correction_contexts: Arc<tokio::sync::RwLock<CorrectionContextRegistry>>,
}

/// Maximum number of in-flight remediation correction contexts retained in
/// `Agent::correction_contexts`. Remediations are rare and short-lived; this cap
/// only guards against an unbounded leak if a background spawn never tears its
/// entry down. When exceeded, the oldest-inserted entry is evicted.
///
/// Consumed by `register_correction_context`, which is exercised by the dispatch
/// path / tests; tolerated as dead in plain (non-test) lib builds until the live
/// reaper caller lands in a later 3c task.
#[allow(dead_code)]
const MAX_CORRECTION_CONTEXTS: usize = 64;

pub(in crate::agent) use system_directives::{EarlyStopSeverity, SystemDirective};

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
    is_goal_run_root_task_description, is_low_signal_task_lead_reply,
    looks_like_evidence_grounding_challenge, looks_like_false_capability_denial_after_tool_success,
    looks_like_incomplete_live_work_summary, parse_goal_leading_wait, parse_wait_task_seconds,
    persist_scheduled_run_state, strip_leading_wait, task_has_scheduled_provenance,
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

pub mod correction_dispatch;
pub mod correction_execution;
pub mod correction_intent;
pub mod correction_sandbox;
pub mod self_correction;

// `Agent` constructors (new / with_depth / set_test_*) live in `construct.rs`.
#[path = "construct.rs"]
mod construct;

// impl-Agent justification: public entry surface (handle_message, hub/self-ref wiring, role/depth accessors, goal cancellation) — Agent's API to channels and core.
impl Agent {
    /// Revalidate the immutable mandate run and exact task-attempt lease. This
    /// runs before grant issuance and again at the final adapter boundary.
    pub(crate) async fn require_live_mandate_execution(
        &self,
        expected_goal_id: &str,
    ) -> anyhow::Result<()> {
        const FENCE_RENEWAL_SECS: i64 = 180;

        let fence = self.mandate_execution.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Mandate execution is missing its immutable run/task fence.")
        })?;
        anyhow::ensure!(
            fence.goal_id == expected_goal_id
                && self.goal_id.as_deref() == Some(expected_goal_id)
                && self.task_id.as_deref() == Some(fence.worker_task_id.as_str()),
            "Mandate execution identity does not match this child agent."
        );
        let mandate = self
            .state
            .get_mandate_for_goal(expected_goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate authority record disappeared."))?;
        anyhow::ensure!(
            mandate.id == fence.mandate_id
                && mandate.version == fence.mandate_version
                && mandate.authority == fence.authority
                && mandate.is_active(),
            "Mandate authority changed or is no longer active."
        );

        let task = self
            .state
            .get_task(&fence.worker_task_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate worker task no longer exists."))?;
        anyhow::ensure!(
            task.goal_id == expected_goal_id
                && matches!(task.status.as_str(), "claimed" | "running")
                && !task.has_error()
                && !task.has_blocker(),
            "Mandate worker task is no longer executable."
        );

        let attempt = self
            .state
            .get_current_task_attempt(&fence.worker_task_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate task-attempt lease is no longer active."))?;
        anyhow::ensure!(
            attempt.id == fence.attempt_id
                && attempt.lease_token == fence.lease_token
                && attempt.task_id == fence.worker_task_id
                && attempt.goal_run_id == fence.goal_run_id
                && matches!(attempt.status.as_str(), "claimed" | "running"),
            "Mandate task-attempt lease was replaced or lost."
        );
        anyhow::ensure!(
            self.state
                .heartbeat_task_attempt(&fence.attempt_id, &fence.lease_token, FENCE_RENEWAL_SECS,)
                .await?,
            "Mandate task-attempt lease expired or was revoked."
        );

        // Reload after the atomic lease check so a closed cycle cannot be
        // rebound by goal ID to a later run.
        let run = self
            .state
            .get_current_goal_run(expected_goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate decision cycle is no longer active."))?;
        anyhow::ensure!(
            run.id == fence.goal_run_id
                && run.goal_id == expected_goal_id
                && run.trigger_type == "mandate"
                && run.status == "running"
                && run.root_task_id.as_deref() == Some(fence.root_task_id.as_str()),
            "Mandate execution is stale, blocked, or bound to a different decision cycle."
        );
        Ok(())
    }

    /// Set the ChannelHub reference (called after hub creation in core.rs).
    pub async fn set_hub(&self, hub: Weak<ChannelHub>) {
        *self.hub.write().await = Some(hub);
    }

    /// Set the plan store (called after construction in core.rs). Enables the
    /// completion phase's checklist soft-verification + recap. Absent on
    /// subagents/tests, where checklist handling degrades to current behavior.
    pub async fn set_plan_store(&self, plan_store: Arc<crate::plans::PlanStore>) {
        *self.plan_store.write().await = Some(plan_store);
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

    /// Read-only context-window diagnostics for the owner-facing `/context`
    /// command. No prompt content or private memory values are exposed.
    pub(crate) async fn context_debug_settings(
        &self,
        session_id: &str,
        model: &str,
    ) -> (bool, usize, usize, usize, Option<i64>) {
        (
            self.context_window_config.enabled,
            crate::memory::context_window::model_context_budget(model, &self.context_window_config),
            self.context_window_config
                .summarize_token_threshold_for(model),
            self.context_window_config.summary_recent_tokens_for(model),
            self.turn_anchors.read().await.get(session_id).copied(),
        )
    }

    /// Lower the token-pressure watermark in focused tests without weakening
    /// production defaults or making tests manufacture multi-megabyte prompts.
    #[cfg(test)]
    pub(crate) fn set_context_compaction_tokens_for_test(
        &mut self,
        token_threshold: usize,
        recent_tokens: usize,
    ) {
        self.context_window_config.summarize_token_threshold = token_threshold;
        self.context_window_config.summary_recent_tokens = recent_tokens;
    }

    /// Register a correction-execution context for a remediation goal id.
    ///
    /// Called by `correction_dispatch::dispatch_correction_remediation` BEFORE
    /// spawning the remediation task lead, so that the spawned hierarchy's agent
    /// loops can thread `Some(ctx)` into their `ToolExecutionCtx.correction` (see
    /// `correction_contexts` field docs for the safety/non-stickiness invariant).
    ///
    /// The map is bounded by [`MAX_CORRECTION_CONTEXTS`]; if inserting would
    /// exceed the cap, the oldest-inserted entry is evicted first (FIFO) so a
    /// fire-and-forget background spawn that never tears its entry down cannot
    /// leak unboundedly.
    ///
    /// Exercised by the dispatch path and its tests; tolerated as dead in plain
    /// (non-test) lib builds until the live caller lands in a later 3c task.
    #[allow(dead_code)]
    pub(crate) async fn register_correction_context(
        &self,
        goal_id: &str,
        ctx: Arc<crate::agent::correction_execution::CorrectionExecutionContext>,
    ) {
        self.correction_contexts
            .write()
            .await
            .insert(goal_id.to_string(), ctx);
    }

    /// Read (clone) the correction-execution context for this agent's current
    /// goal id, if one was registered.
    ///
    /// This is a PEEK (it does not remove the entry) so the same context is
    /// visible to the remediation task-lead AND every executor it spawns (they
    /// all inherit the same `goal_id`). Returns `None` for `goal_id == None` and
    /// for any goal id with no registered context — i.e. every normal,
    /// user-initiated turn. Used at the `ToolExecutionCtx` construction site in
    /// `main_loop.rs`.
    pub(crate) async fn correction_context_for_current_goal(
        &self,
    ) -> Option<Arc<crate::agent::correction_execution::CorrectionExecutionContext>> {
        let goal_id = self.goal_id.as_deref()?;
        self.correction_contexts.read().await.get(goal_id)
    }

    /// Remove the correction-execution context for a remediation goal id once the
    /// remediation task has finished. Idempotent. Called by the dispatch's
    /// teardown so completed remediations do not linger in the bounded map.
    ///
    /// Tolerated as dead until the live teardown caller lands in a later 3c task.
    #[allow(dead_code)]
    pub(crate) async fn clear_correction_context(&self, goal_id: &str) {
        self.correction_contexts.write().await.remove(goal_id);
    }

    /// Test-only: number of registered correction contexts.
    #[cfg(test)]
    pub(crate) async fn correction_context_count(&self) -> usize {
        self.correction_contexts.read().await.len()
    }

    /// Shared `EventStore` handle. Used by the idle-reaper → correction bridge
    /// (`TerminalTool`) to fetch recent conversation history for subject
    /// reconstruction. Returns a borrow of the `Arc`; clone it if an owned
    /// handle is needed.
    pub(crate) fn event_store(&self) -> &Arc<EventStore> {
        &self.event_store
    }

    /// Shared `StateStore` handle. Used by the idle-reaper → correction bridge
    /// to hand the dispatch path an owned `Arc<dyn StateStore>`.
    pub(crate) fn state_arc(&self) -> Arc<dyn StateStore> {
        self.state.clone()
    }

    pub(crate) fn render_options(&self, model: &str) -> turn_render::RenderOptions {
        turn_render::RenderOptions {
            vision: self.vision_config.clone(),
            audio: self.audio_config.clone(),
            stt: self.stt_config.clone(),
            model: model.to_string(),
        }
    }

    pub(in crate::agent) fn harness_eval_handle(&self) -> crate::agent::eval::HarnessEvalHandle {
        Arc::clone(&self.harness_eval)
    }

    pub(in crate::agent) async fn install_harness_eval(&self, accumulator: HarnessEvalAccumulator) {
        *self.harness_eval.write().await = Some(accumulator);
    }

    pub(in crate::agent) async fn with_harness_eval<R>(
        &self,
        f: impl FnOnce(&mut HarnessEvalAccumulator) -> R,
    ) -> Option<R> {
        self.harness_eval.write().await.as_mut().map(f)
    }

    pub(in crate::agent) fn harness_eval_enabled(&self) -> bool {
        self.harness_eval_config.enabled
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

    /// One-shot, TOOL-LESS interpretation of a finished background command's
    /// result. The terminal notifier uses this to turn a bare value (a `wc -l`
    /// count, a path, a status line) into one short plain-language sentence
    /// WITHOUT re-entering the full agent loop. The provider is called with an
    /// empty tools slice, so the model can only reply in text — it physically
    /// cannot re-run the command. (Re-running is what caused the small-model
    /// background-detach churn: the loop re-ran a slow `find`, re-detached, and
    /// re-engaged repeatedly, emitting duplicate "finished" pings.)
    ///
    /// Returns `None` on any failure or empty reply so the caller can fall back
    /// to delivering the raw result — the answer is never lost.
    pub async fn interpret_background_result(&self, command: &str, output: &str) -> Option<String> {
        let snapshot = self.llm_runtime.snapshot();
        let provider = snapshot.provider();
        let model = snapshot.primary_model();

        let system = "You translate the result of a shell command into ONE short, \
            plain-language sentence for a non-technical user. Say what the result \
            represents and flag any obvious caveat (for example, a raw text-match \
            count is not the same as a count of files). Never suggest running \
            anything, never use code formatting, and reply with the sentence only.";
        let user = format!(
            "Command that was run:\n{command}\n\nIts output:\n{output}\n\n\
             In one sentence, tell the user what this result means."
        );
        let messages = vec![
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user}),
        ];

        // Empty tools slice — the model has nothing to call, so no re-run / churn.
        match provider.chat(&model, &messages, &[]).await {
            Ok(resp) => {
                let text = resp.content.unwrap_or_default();
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
            Err(e) => {
                warn!(error = %e, "Background-result interpretation LLM call failed");
                None
            }
        }
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
        self.handle_message_with_attachments(
            session_id,
            user_text,
            &[],
            status_tx,
            user_role,
            channel_ctx,
            heartbeat,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn handle_message_with_attachments(
        &self,
        session_id: &str,
        user_text: &str,
        attachments: &[crate::traits::MessageAttachment],
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String> {
        // Marks an agent task in flight for the entire turn (all recursion
        // depths route through here: interactive turns, goal-run task leads,
        // spawned specialists). The background memory pipeline defers its
        // LLM work while this is held — see activity_gate.
        let _activity = activity_gate::AgentActivityGuard::acquire();
        let scheduled_goal_to_clear = if let Some(goal_id) = self.goal_id.as_deref() {
            let is_scheduled_goal =
                goal_has_scheduled_provenance(&self.state, goal_id, self.task_id.as_deref()).await;
            let is_root_scheduled_run = if self.task_id.is_none() {
                is_scheduled_goal
            } else {
                task_has_scheduled_provenance(&self.state, self.task_id.as_deref()).await
            };
            // Background scheduled cycles are owned by background_task_lead,
            // which must keep one shared budget alive across planning and all
            // executor siblings. Direct/resume runs without a background task
            // identity still own their cleanup here.
            if agent_return_owns_scheduled_run_cleanup(
                self.depth,
                self.task_id.is_some(),
                is_root_scheduled_run,
            ) {
                Some(goal_id.to_string())
            } else {
                None
            }
        } else {
            None
        };

        let backend = crate::execution::active_execution_backend();
        let mut execution_user_text = user_text.to_string();
        if backend.kind() != crate::execution::BackendKind::Local {
            for attachment in attachments {
                let safe_name: String = attachment
                    .filename
                    .chars()
                    .map(|character| {
                        if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_')
                        {
                            character
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let destination = backend
                    .workspace_root()
                    .join(".aidaemon")
                    .join("inbox")
                    .join(format!("{}-{}", uuid::Uuid::new_v4(), safe_name));
                let imported = backend
                    .import_local_file(std::path::Path::new(&attachment.local_path), &destination)
                    .await
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "Failed to import attachment {:?} into the {} execution workspace: {}",
                            attachment.filename,
                            backend.kind().as_str(),
                            error
                        )
                    })?;
                execution_user_text =
                    execution_user_text.replace(&attachment.local_path, imported.as_str());
            }
        }

        let reply = self
            .handle_message_impl(
                session_id,
                &execution_user_text,
                attachments,
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
        self.cancel_active_goals_for_session_scoped(session_id, SessionGoalCancellationScope::All)
            .await
    }

    /// Cancel only finite orchestration work for a session.
    ///
    /// Mid-task pivots use this narrower scope so a conversational correction
    /// cannot cancel durable recurring goals or tracked personal goals that
    /// happen to share the same channel session.
    pub(crate) async fn cancel_active_finite_work_for_session(
        &self,
        session_id: &str,
    ) -> Vec<String> {
        self.cancel_active_goals_for_session_scoped(
            session_id,
            SessionGoalCancellationScope::FiniteOrchestration,
        )
        .await
    }

    async fn cancel_active_goals_for_session_scoped(
        &self,
        session_id: &str,
        scope: SessionGoalCancellationScope,
    ) -> Vec<String> {
        let goals = self
            .state
            .get_goals_for_session(session_id)
            .await
            .unwrap_or_default();
        let active: Vec<&crate::traits::Goal> = goals
            .iter()
            .filter(|goal| session_goal_matches_cancellation_scope(goal, scope))
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

#[derive(Clone, Copy)]
enum SessionGoalCancellationScope {
    All,
    FiniteOrchestration,
}

fn session_goal_matches_cancellation_scope(
    goal: &crate::traits::Goal,
    scope: SessionGoalCancellationScope,
) -> bool {
    let is_active = matches!(
        goal.status.as_str(),
        "active" | "pending" | "pending_confirmation"
    );
    if !is_active {
        return false;
    }

    match scope {
        SessionGoalCancellationScope::All => true,
        SessionGoalCancellationScope::FiniteOrchestration => {
            goal.domain == "orchestration" && goal.goal_type == "finite"
        }
    }
}

fn agent_return_owns_scheduled_run_cleanup(
    depth: usize,
    has_task_id: bool,
    is_root_scheduled_run: bool,
) -> bool {
    is_root_scheduled_run && (depth == 0 || !has_task_id)
}

#[cfg(test)]
mod session_goal_cancellation_tests {
    use crate::traits::Goal;

    use super::{session_goal_matches_cancellation_scope, SessionGoalCancellationScope};

    #[test]
    fn pivot_scope_preserves_continuous_and_personal_goals() {
        let finite = Goal::new_finite("Answer the current question", "session-1");
        let recurring = Goal::new_continuous("Publish a daily blog post", "session-1", None, None);
        let personal = Goal::new_personal("Keep writing", "session-1");

        assert!(session_goal_matches_cancellation_scope(
            &finite,
            SessionGoalCancellationScope::FiniteOrchestration
        ));
        assert!(!session_goal_matches_cancellation_scope(
            &recurring,
            SessionGoalCancellationScope::FiniteOrchestration
        ));
        assert!(!session_goal_matches_cancellation_scope(
            &personal,
            SessionGoalCancellationScope::FiniteOrchestration
        ));
    }

    #[test]
    fn explicit_stop_scope_still_includes_continuous_goals() {
        let recurring = Goal::new_continuous("Publish a daily blog post", "session-1", None, None);

        assert!(session_goal_matches_cancellation_scope(
            &recurring,
            SessionGoalCancellationScope::All
        ));
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
        assert!(sanitized.contains("your personal AI assistant"));
        assert!(!sanitized.contains("ChatGPT"));
        assert!(!sanitized.contains("Gemini"));
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
