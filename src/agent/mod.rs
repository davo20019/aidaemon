use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

#[cfg(test)]
use chrono::Utc;
#[cfg(test)]
use serde_json::json;
#[cfg(test)]
use serde_json::Value;
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};

use crate::config::{IterationLimitConfig, PathAliasConfig};
use crate::events::EventStore;
#[cfg(test)]
use crate::events::TaskStatus;
#[cfg(test)]
use crate::events::{
    AssistantResponseData, TaskEndData, TaskStartData, ThinkingStartData, ToolResultData,
};
#[cfg(test)]
use crate::events::{DecisionPointData, DecisionType, EventType};
#[cfg(test)]
use crate::execution_policy::{ExecutionPolicy, ModelProfile};
use crate::goal_tokens::GoalTokenRegistry;
use crate::llm_runtime::SharedLlmRuntime;
use crate::mcp::McpRegistry;
use crate::runtime_ports::OutboundRouter;
use crate::skills;
#[cfg(test)]
use crate::traits::Message;
#[cfg(test)]
use crate::traits::ToolCapabilities;
use crate::traits::{AgentRole, StateStore, Tool};
#[cfg(test)]
use crate::types::ChannelVisibility;
use crate::types::{ChannelContext, UserRole};
// Re-export StatusUpdate from types for backwards compatibility
#[cfg(test)]
use crate::events::ToolCallInfo;
#[cfg(test)]
use crate::traits::ToolCall;
pub use crate::types::StatusUpdate;
// `Task` is consumed only by in-tree `#[cfg(test)]` modules that import the
// agent namespace via `use super::*` (e.g. `runtime/goal_failure_tests.rs`).
#[cfg(test)]
use crate::traits::Task;

#[cfg(test)]
const DEFERRED_NO_TOOL_ERROR_MARKER: &str = "deferred-action no-tool loop";
#[path = "response_analysis.rs"]
mod response_analysis;
#[cfg(test)]
use response_analysis::looks_like_incomplete_retry_plan;
#[cfg(test)]
use response_analysis::sanitize_response_analysis;
#[cfg(test)]
use response_analysis::{reply_defers_file_access, user_text_references_file};
#[path = "intent/keywords.rs"]
mod intent_keywords;
#[path = "intent/intent_routing.rs"]
mod intent_routing;
#[path = "intent/relational_prefilter.rs"]
#[cfg(test)]
pub mod relational_prefilter;
#[allow(unused_imports)]
use intent_routing::contains_keyword_as_words;
#[cfg(test)]
use intent_routing::infer_intent_gate;
// Re-export for use outside the `agent` subtree (e.g. `tools/browser/policy`).
#[allow(unused_imports)]
pub(crate) use intent_routing::contains_keyword_as_words as keyword_match;
/// Policy signals live in `aidaemon-runtime`.
#[cfg(test)]
mod policy_signals {
    pub(super) use aidaemon_runtime::policy::signals::*;
}
#[cfg(test)]
use policy_signals::is_short_user_correction;
/// Validation state lives in `aidaemon-runtime`.
mod validation_state {
    pub(crate) use aidaemon_runtime::turn::validation_state::*;
}
pub(crate) use validation_state::{
    build_needs_approval_request, derive_executor_step_result, extract_executor_handoff_context,
    persist_executor_handoff_context, persist_executor_result_context, ExecutorHandoff,
    ExecutorStepResult, PartialResult, StepValidationOutcome, TaskValidationOutcome,
};
#[path = "turn/execution_state.rs"]
mod execution_state;
#[cfg(test)]
pub(crate) use execution_state::ExecutionBudget;
pub(crate) use execution_state::TargetScope;
/// The post-run learning context lives in `aidaemon-runtime`.
mod post_task {
    pub(crate) use aidaemon_runtime::turn::post_task::*;
}
#[cfg(test)]
use post_task::LearningContext;

#[cfg(feature = "computer_use")]
mod computer_use;

#[path = "runtime/approvals.rs"]
pub(crate) mod approvals;
#[path = "runtime/authority.rs"]
pub(crate) mod authority;
#[path = "turn/bootstrap/mod.rs"]
mod bootstrap;
/// Completion contracts live in `aidaemon-runtime`.
#[cfg(test)]
mod completion_contract {
    pub(in crate::agent) use aidaemon_runtime::turn::completion_contract::*;
}
pub(crate) mod conversation_compaction;
#[path = "runtime/coordination.rs"]
pub(crate) mod coordination;
/// The cacheable core prompt lives in `aidaemon-runtime`.
#[cfg(test)]
mod core_prompt {
    pub(crate) use aidaemon_runtime::prompt::core::*;
}
/// Dialogue state lives in `aidaemon-runtime`.
mod dialogue_state {
    #[cfg(test)]
    pub(in crate::agent) use aidaemon_runtime::turn::dialogue_state::*;
}
#[cfg(test)]
#[path = "runtime/dialogue_state_tests.rs"]
mod dialogue_state_tests;
/// Follow-up classification lives in `aidaemon-runtime`.
mod followup {
    #[cfg(test)]
    pub(in crate::agent) use aidaemon_runtime::turn::followup::*;
}
#[path = "runtime/graceful.rs"]
mod graceful;
#[cfg(test)]
#[path = "runtime/history.rs"]
mod history;
pub(crate) mod memory_plugin;
#[path = "runtime/notes.rs"]
mod notes;
#[path = "runtime/parent_delivery.rs"]
mod parent_delivery;
#[path = "turn/post_turn.rs"]
mod post_turn;
#[path = "runtime/sdk_engine.rs"]
pub(crate) mod sdk_engine;
#[path = "runtime/sdk_provider.rs"]
mod sdk_provider;
#[cfg(test)]
#[path = "turn/services.rs"]
mod services;
pub(crate) mod specialists;
#[path = "runtime/tool_result_fit.rs"]
mod tool_result_fit;
#[path = "runtime/turn_context.rs"]
mod turn_context;
#[path = "runtime/untrusted.rs"]
pub(crate) mod untrusted;
#[cfg(test)]
pub(in crate::agent) use completion_contract::CompletionTaskKind;
#[cfg(test)]
pub(in crate::agent) use followup::FollowupMode;
#[cfg(test)]
pub(in crate::agent) use history::CompletionContract;
#[cfg(test)]
pub(in crate::agent) use history::CompletionProgress;
#[cfg(test)]
pub(in crate::agent) use history::TurnContext;
pub(in crate::agent) mod eval;
#[path = "runtime/llm.rs"]
mod llm;
#[path = "turn/main_loop.rs"]
mod main_loop;
#[path = "turn/mandate_run.rs"]
mod mandate_run;
#[path = "turn/message_build.rs"]
mod message_build;
#[path = "runtime/models.rs"]
mod models;
/// Prefix fingerprints live in `aidaemon-runtime`.
mod prefix_fingerprint {
    #[cfg(test)]
    pub(crate) use aidaemon_runtime::prompt::fingerprint::*;
}
#[path = "runtime/resume.rs"]
mod resume;
mod runtime_ports;
#[path = "runtime/spawn.rs"]
mod spawn;
mod turn_host;
/// Session notes and parent-visible deliveries, on any TurnHost.
pub(crate) use aidaemon_runtime::turn::notes::TurnNotes;
/// The speech-to-text fallback lives in `aidaemon-runtime`.
#[cfg(test)]
pub(crate) mod stt {
    pub(crate) use aidaemon_runtime::media::stt::*;
}
/// System directives live in `aidaemon-runtime`.
#[cfg(test)]
mod system_directives {
    pub(in crate::agent) use aidaemon_runtime::turn::system_directives::*;
}
#[path = "runtime/system_prompt.rs"]
mod system_prompt;
/// Tool dispatch and the tool roster live in `aidaemon-runtime`
/// (`engine::dispatcher`, `engine::tool_roster`); their tests run here, on
/// the daemon's harness.
#[cfg(test)]
#[path = "tools/tool_defs_tests.rs"]
mod tool_defs_tests;
#[cfg(test)]
#[path = "tools/tool_exec_tests.rs"]
mod tool_exec;
#[path = "turn/tool_execution/mod.rs"]
mod tool_execution;

pub(in crate::agent) use eval::HarnessEvalConfig;

/// Policy runtime metrics and bounded autotuning live in `aidaemon-runtime`.
mod policy_metrics {
    pub use aidaemon_runtime::policy::metrics::*;
}
pub use policy_metrics::{
    apply_bounded_autotune_from_failure_ratio, init_policy_tunables_once, policy_autotune_snapshot,
    policy_metrics_boot_id, policy_metrics_snapshot,
};
// `PolicyAutotuneSnapshot` is part of the public surface but currently has no
// in-tree consumer (the legacy definition carried `#[allow(dead_code)]`).
#[allow(unused_imports)]
pub use policy_metrics::PolicyAutotuneSnapshot;

// Free-function helpers (status, resume, project-scope cues, untrusted-reference
// filtering, intent-gate merge). Implementation lives in `agent_helpers.rs`.
#[path = "agent_helpers.rs"]
mod agent_helpers;
#[cfg(test)]
pub use agent_helpers::touch_heartbeat;
#[cfg(test)]
pub(in crate::agent) use agent_helpers::IntentGateDecision;
#[cfg(test)]
pub(in crate::agent) use agent_helpers::ResumeCheckpoint;
#[cfg(test)]
pub(in crate::agent) use agent_helpers::{
    user_explicitly_requests_local_file_inspection, user_text_references_filesystem_path,
};

/// Phase 0 per-session window-boundary memory: `session_id` →
/// (last `keep_from` index, last oldest-kept persisted message id). Used by
/// `message_build` to emit an explicit boundary-movement event.
type WindowBoundaryMemory = Arc<tokio::sync::RwLock<HashMap<String, (usize, Option<String>)>>>;

/// Immutable execution identity for one worker inside one autonomous mandate
/// decision cycle. The lease token is Rust-side control state and must never be
/// serialized, logged, or exposed to the model.
pub(crate) use aidaemon_runtime::turn_runtime::MandateExecutionFence;

pub struct Agent {
    llm_runtime: SharedLlmRuntime,
    /// The turn runtime's settings and caches (context window, policy,
    /// media, prompt and render caches, turn ids).
    runtime: aidaemon_runtime::turn_runtime::TurnRuntime,
    /// The gateway for model calls made outside the agent loop.
    side_calls: crate::llm_runtime::gateway::SideCallGateway,
    state: Arc<dyn StateStore>,
    memory_store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    event_store: Arc<EventStore>,
    tools: Vec<Arc<dyn Tool>>,
    model: RwLock<String>,
    fallback_model: RwLock<String>,
    system_prompt: String,
    config_path: PathBuf,
    /// How paths are classified for this daemon (its own files included).
    daemon_paths: crate::tools::DaemonPaths,
    /// The process's one execution environment, built once at startup.
    execution_backend: crate::execution::SharedExecutionBackend,
    skills_dir: PathBuf,
    skill_cache: skills::SkillCache,
    /// Current recursion depth (0 = root agent).
    depth: usize,
    /// Static limits, budgets, and timeout settings.
    limits: AgentLimits,
    /// When true, the user has manually set a model via /model — skip auto-routing.
    model_override: RwLock<bool>,
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
    /// This child's progress clock: its model and tool calls and kernel
    /// events keep it from being stopped as idle (`subagents.timeout_secs`).
    child_progress: Option<crate::tools::ProgressClock>,
    /// The sub-agents running now, by the chat they report to; shared with
    /// every child so `/status` and `/tasks` see the whole tree.
    running_children: Arc<spawn::RunningChildren>,
    /// SDK kernel settings for this agent's turns; inherited by spawned children.
    sdk_engine: Arc<sdk_engine::SdkEngineSettings>,
    /// The kernel's durable run store: every SDK run journals its events and
    /// checkpoints its state here. Inherited by spawned children.
    run_store: Option<Arc<aidaemon_store_sqlite::SqliteStore>>,
    /// The owner's approval cards, grants and waiting cards; shared with
    /// every child, so a grant in a chat covers its children's calls.
    owner_approvals: Arc<approvals::OwnerApprovals>,
    /// Long-term memory for SDK turns (hooks, tools, owned daemon tools).
    memory: Arc<dyn memory_plugin::MemoryPlugin>,
    /// The daemon store's summarization and learning, run on every turn
    /// whatever the memory plugin.
    transcript_writes: Arc<dyn memory_plugin::MemoryWrites>,
    /// Goal cancellation token registry — shared across agent hierarchy.
    goal_token_registry: Option<GoalTokenRegistry>,
    /// Weak reference to the ChannelHub for background notifications.
    /// Uses RwLock because hub is created after Agent (core.rs ordering).
    hub: RwLock<Option<Weak<dyn OutboundRouter>>>,
    /// Plan store for the requirement-checklist feature. Set after construction
    /// (core.rs ordering) via `set_plan_store`; `None` on subagents and in tests,
    /// in which case checklist verification/recap degrade to current behavior.
    plan_store: RwLock<Option<Arc<crate::plans::PlanStore>>>,
    /// Session IDs that have granted schedule confirmation for this process lifetime.
    /// Allows schedule creation to auto-confirm after an explicit AllowSession/AllowAlways.
    schedule_approved_sessions: Arc<tokio::sync::RwLock<HashSet<String>>>,
    /// Models that recently returned 402 billing errors. Maps model name → failure time.
    /// Shared across parent/child agents. Entries expire after BILLING_FAIL_CACHE_TTL.
    billing_failed_models: Arc<tokio::sync::RwLock<HashMap<String, Instant>>>,
    /// Weak self-reference for background task spawning.
    /// Set after Arc creation via `set_self_ref()`.
    self_ref: RwLock<Option<Weak<Agent>>>,
    /// The channels' per-session message queue. A root turn takes messages
    /// the same sender sends while it runs into the run (mid-turn steering).
    /// Set after channel startup via `set_task_registry()`.
    task_registry: std::sync::OnceLock<Arc<crate::tasks::TaskRegistry>>,
    /// Configured path alias roots (for example, `projects/...`).
    path_aliases: PathAliasConfig,
    /// Where received attachments are saved: daemon storage, never a project.
    inbox_dir: Option<std::path::PathBuf>,
    /// Parent scope carried into spawned child agents.
    inherited_project_scope: Option<String>,
    /// Human-facing session that owns approvals for this internal child.
    /// Root agents leave this unset; every spawned descendant inherits the
    /// originating conversation while retaining its own isolated runtime ID.
    approval_session_id: Option<String>,
    /// What a spawned child inherited from the turn that spawned it: its
    /// authority, owner level and workspace narrow the child's own
    /// resolution (a child is never wider than its parent), its mode session
    /// is the chat whose `/readonly` narrows the child, and its taint and root
    /// request carry the untrusted-content rule across. Root agents leave it
    /// unset.
    inherited_authority: Option<crate::agent::authority::InheritedAuthority>,
    /// Full tool list from the root agent — used by TaskLead when spawning
    /// Executor children so they can access Action tools that were filtered
    /// out of the TaskLead's own `tools` vec.
    root_tools: Option<Vec<Arc<dyn Tool>>>,
    /// Some turn of this agent read (or started with) untrusted content.
    /// Kept in memory so a taint that could not be recorded still counts
    /// when a parent reads this child's result.
    read_untrusted: Arc<std::sync::atomic::AtomicBool>,
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
    /// llama.cpp interactive KV-cache slot for the main generation loop.
    /// `Some(interactive_slot)` only when slot routing is enabled on the primary
    /// provider; `None` otherwise (and for all sub-agents). When set, it is
    /// applied to the main-loop `ChatOptions.id_slot` so the interactive
    /// conversation pins a dedicated slot that background tasks cannot evict.
    interactive_slot: Option<u32>,
}

#[cfg(test)]
pub(in crate::agent) use system_directives::SystemDirective;

struct AgentLimits {
    max_depth: usize,
    iteration_config: IterationLimitConfig,
    max_iterations: usize,
    max_iterations_cap: usize,
    max_response_chars: usize,
    timeout_secs: u64,
    max_facts: usize,
    llm_call_timeout: Option<Duration>,
    task_token_budget: Option<u64>,
    spawn_family: authority::SpawnFamilyLimits,
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

// Goal/task dispatch helpers live in `aidaemon_runtime::goal_dispatch`.
#[path = "goal_dispatch.rs"]
mod goal_dispatch;
#[cfg(test)]
pub(crate) use goal_dispatch::{build_goal_failure_summary, extract_file_paths_from_text};
pub(crate) use goal_dispatch::{
    build_goal_task_results_summary, SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS,
    SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
};
#[cfg(test)]
pub(in crate::agent) use goal_dispatch::{parse_wait_task_seconds, user_facing_task_description};

// A goal's background task lead lives in `aidaemon_runtime::background_task_lead`.
#[path = "background_task_lead.rs"]
mod background_task_lead;
pub use background_task_lead::spawn_background_task_lead;

// `Agent` constructors (new / with_depth / set_test_*) live in `construct.rs`.
#[path = "construct.rs"]
mod construct;
pub(crate) use construct::{AgentConstruction, AgentRuntimeDependencies};

// impl-Agent justification: public entry surface (handle_message, hub/self-ref wiring, role/depth accessors, goal cancellation) — Agent's API to channels and core.
impl Agent {
    /// When some configured model can next answer, if none can now.
    pub(crate) fn next_model_answer_at(&self) -> Option<std::time::SystemTime> {
        self.llm_runtime.snapshot().next_answer_at()
    }

    /// Set the ChannelHub reference (called after hub creation in core.rs).
    pub(crate) async fn set_hub(&self, hub: Weak<dyn OutboundRouter>) {
        *self.hub.write().await = Some(hub);
    }

    pub(crate) async fn runtime_wiring_ready(&self) -> bool {
        self.self_ref
            .read()
            .await
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some()
            && self
                .hub
                .read()
                .await
                .as_ref()
                .and_then(Weak::upgrade)
                .is_some()
    }

    pub(crate) async fn self_reference_ready(&self) -> bool {
        self.self_ref
            .read()
            .await
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some()
    }

    /// Set the plan store (called after construction in core.rs). Enables the
    /// completion phase's checklist soft-verification + recap. Absent on
    /// subagents/tests, where checklist handling degrades to current behavior.
    pub async fn set_plan_store(&self, plan_store: Arc<crate::plans::PlanStore>) {
        *self.plan_store.write().await = Some(plan_store);
    }

    /// Set a weak self-reference for background task spawning.
    /// Must be called after wrapping the Agent in Arc.
    /// Share the channels' task registry so a running root turn can take
    /// messages the same sender queues behind it (mid-turn steering).
    pub(crate) fn set_task_registry(&self, registry: Arc<crate::tasks::TaskRegistry>) {
        let _ = self.task_registry.set(registry);
    }

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
            self.runtime.context_window_config.enabled,
            crate::context_budget::model_context_budget(model, &self.runtime.context_window_config),
            self.runtime
                .context_window_config
                .summarize_token_threshold_for(model),
            self.runtime
                .context_window_config
                .summary_recent_tokens_for(model),
            self.runtime
                .turn_anchors
                .read()
                .await
                .get(session_id)
                .copied(),
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
        self.runtime.context_window_config.summarize_token_threshold = token_threshold;
        self.runtime.context_window_config.summary_recent_tokens = recent_tokens;
        // The turn's post-reply compaction reads the writer's own copy.
        self.transcript_writes = memory_plugin::DaemonMemory::new(
            self.state.clone(),
            self.memory_store.clone(),
            self.event_store.clone(),
            self.llm_runtime.clone(),
            self.side_calls.clone(),
            self.runtime.context_window_config.clone(),
            0,
            false,
            "",
        )
        .writes();
    }

    /// Shared `EventStore` handle. Returns a borrow of the `Arc`; clone it if
    /// an owned handle is needed.
    #[cfg(test)]
    pub(crate) fn event_store(&self) -> &Arc<EventStore> {
        &self.event_store
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
        aidaemon_runtime::engine::work_scope::WorkScope::validate_executor_task_for_spawn(
            self,
            task_id,
            expected_goal_id,
        )
        .await
    }

    /// Maximum agentic loop iterations per invocation.
    pub fn max_iterations(&self) -> usize {
        self.limits.max_iterations
    }

    // ==================== Orchestration Methods ====================

    /// Run the agentic loop for a user message in the given session.
    /// Returns the final assistant text response.
    /// `heartbeat` is an optional atomic timestamp updated on each activity point.
    /// Channels pass `Some(heartbeat)` so the typing indicator can detect stalls;
    /// sub-agents, triggers, and tests pass `None`.
    #[cfg(test)]
    fn sanitize_final_reply_markers(reply: &str) -> String {
        // NOTE: The primary sanitization pass already runs in post_turn.rs.
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
    /// WITHOUT re-entering the full agent loop. The side-call gateway sends no
    /// tools, so the model can only reply in text — it physically cannot
    /// re-run the command. (Re-running is what caused the small-model
    /// background-detach churn: the loop re-ran a slow `find`, re-detached, and
    /// re-engaged repeatedly, emitting duplicate "finished" pings.)
    ///
    /// Returns `None` on any failure or empty reply so the caller can fall back
    /// to delivering the raw result — the answer is never lost.
    pub async fn interpret_background_result(&self, command: &str, output: &str) -> Option<String> {
        self.engine()
            .interpret_background_result(command, output)
            .await
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
        self.handle_message_with_attachments_and_ingress(
            session_id,
            user_text,
            attachments,
            status_tx,
            user_role,
            channel_ctx,
            heartbeat,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn handle_message_with_attachments_and_ingress(
        &self,
        session_id: &str,
        user_text: &str,
        attachments: &[crate::traits::MessageAttachment],
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
        ingress_timing: Option<crate::runtime_ports::InboundMessageTiming>,
    ) -> anyhow::Result<String> {
        self.engine()
            .handle_inbound(
                session_id,
                user_text,
                attachments,
                status_tx,
                user_role,
                channel_ctx,
                heartbeat,
                ingress_timing,
            )
            .await
    }

    /// Re-enter an unfinished conversation with runtime-produced evidence.
    /// Unlike a channel ingress turn, this must not be interpreted as new
    /// owner intent or let worker output broaden the request's target scope.
    pub(crate) async fn handle_internal_continuation(
        &self,
        request: &crate::runtime_ports::ConversationRequest,
    ) -> anyhow::Result<String> {
        self.engine()
            .handle_message_impl(
                &request.session_id,
                &request.user_text,
                &[],
                request.status_tx.clone(),
                request.user_role,
                request.channel_ctx.clone(),
                request.heartbeat.clone(),
                None,
                true,
                request.parent_task_id.as_deref(),
                request.parent_tool_call_id.as_deref(),
                request.parent_result_id.as_deref(),
            )
            .await
    }
}

#[cfg(test)]
mod final_reply_marker_tests {

    use super::{user_facing_task_description, Agent};

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
    fn scheduler_prose_has_no_lifecycle_meaning_in_progress_descriptions() {
        let cleaned = user_facing_task_description(
            "Scheduled check: Post evening tweet about aidaemon features [SYSTEM: already scheduled and firing now; do not reschedule.]",
        );
        assert_eq!(
            cleaned,
            "Scheduled check: Post evening tweet about aidaemon features"
        );
    }
}

#[cfg(test)]
mod tests;
