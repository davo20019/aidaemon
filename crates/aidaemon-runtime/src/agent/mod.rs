//! The agent: the turn engine's host.
//!
//! [`Agent`] holds what one agent instance of a tree is (its role, depth,
//! tools, model and the stores and services it shares with the tree) and is
//! the [`TurnHost`](crate::host::TurnHost) the turn engine runs on. The
//! daemon builds one from its config ([`AgentParts`] plus the host's own
//! [`HostPorts`]); every child is built from its parent
//! ([`TurnHost::create_child`](crate::host::TurnHost::create_child)).
//!
//! What only the host can supply comes in through [`HostPorts`]: its memory,
//! its file policy, its trace export and computer-use state. A provider
//! reload is split the same way: the host builds the new backends from its
//! config and [`Agent::swap_provider`] puts them in place.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use aidaemon_channels::context::ChannelContext;
use aidaemon_channels::ports::OutboundRouter;
use aidaemon_events::EventStore;
use aidaemon_events::traits::AgentRole;
use aidaemon_mcp::registry::McpRegistry;
use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::status::StatusUpdate;
use aidaemon_work::goal_tokens::GoalTokenRegistry;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn};

use crate::eval::HarnessEvalConfig;
use crate::llm::SharedLlmRuntime;
use crate::llm::gateway::SideCallGateway;
use crate::settings::{
    AudioConfig, ContextWindowConfig, IterationLimitConfig, PathAliasConfig, PolicyConfig,
    SttConfig, VisionConfig,
};
use crate::stores::StateStore;
use crate::turn::memory::MemoryPlugin;
use crate::turn::memory_writer::MemoryWrites;
use crate::turn_runtime::{AgentLimits, MandateExecutionFence};

mod child;
mod host;
mod ports;
#[cfg(feature = "testing")]
pub mod testing;

/// Trace export the host runs (OpenTelemetry in the daemon): read at call
/// time, so an exporter started after the agent is built still applies.
pub trait TraceExport: Send + Sync {
    /// The sink a run's kernel events are copied to, if export is on.
    fn sink(&self) -> Option<Arc<dyn aidaemon_core::EventSink>>;
    /// A model call's messages, as sent.
    fn observe_sent(&self, run: aidaemon_core::RunId, messages: &[serde_json::Value]);
}

/// What only the host supplies to an agent; every child shares them.
pub struct HostPorts {
    /// Long-term memory for the agent's turns.
    pub memory: Arc<dyn MemoryPlugin>,
    /// The store's summarization and learning, run on every turn whatever
    /// the memory plugin.
    pub transcript_writes: Arc<dyn MemoryWrites>,
    /// How the file tools treat paths on this host.
    pub file_policy: Arc<dyn aidaemon_tools::FilePolicy>,
    /// Trace export, if the host has one.
    pub trace: Option<Arc<dyn TraceExport>>,
    /// Computer-use pins, when the host runs the `computer_use` tool.
    pub computer_use: Option<Arc<dyn crate::host::ComputerUseState>>,
}

/// The services every agent of a tree shares.
pub struct AgentDependencies {
    pub llm_runtime: SharedLlmRuntime,
    pub side_calls: SideCallGateway,
    pub state: Arc<dyn StateStore>,
    pub memory_store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    pub event_store: Arc<EventStore>,
    pub tools: Vec<Arc<dyn Tool>>,
}

/// What a root agent is built from: its services, settings and limits.
pub struct AgentParts {
    pub dependencies: AgentDependencies,
    pub model: String,
    pub system_prompt: String,
    pub config_path: PathBuf,
    pub daemon_paths: aidaemon_tools::path_class::DaemonPaths,
    pub execution_backend: aidaemon_tools::backend::SharedExecutionBackend,
    pub skills_dir: PathBuf,
    pub max_depth: usize,
    pub max_iterations: usize,
    pub max_iterations_cap: usize,
    pub max_response_chars: usize,
    pub timeout_secs: u64,
    pub max_facts: usize,
    pub iteration_config: IterationLimitConfig,
    pub task_token_budget: Option<u64>,
    pub spawn_family: crate::authority::SpawnFamilyLimits,
    pub llm_call_timeout_secs: Option<u64>,
    pub mcp_registry: Option<McpRegistry>,
    pub goal_token_registry: Option<GoalTokenRegistry>,
    pub hub: Option<Weak<dyn OutboundRouter>>,
    pub record_decision_points: bool,
    pub context_window_config: ContextWindowConfig,
    pub policy_config: PolicyConfig,
    pub path_aliases: PathAliasConfig,
    pub inbox_dir: Option<PathBuf>,
    pub inherited_project_scope: Option<String>,
    pub specialists: Arc<crate::specialists::SpecialistRegistry>,
    pub interactive_slot: Option<u32>,
    pub vision_config: VisionConfig,
    pub audio_config: AudioConfig,
    pub stt_config: SttConfig,
    pub harness_eval_config: HarnessEvalConfig,
}

/// A provider reload's new backends, built by the host from its config.
pub struct ProviderSwap {
    pub provider: crate::llm::ModelBackend,
    pub models: crate::chat_bridge::sdk_backed::ModelList,
    pub primary_model: String,
    pub router: Option<aidaemon_providers::routing::Router>,
    pub provider_kind: aidaemon_providers::routing::ProviderKind,
    pub failover_targets: Vec<crate::llm::ProviderRuntimeTarget>,
    /// The model a failed call falls back to.
    pub fallback_model: String,
    /// The side calls' decision layer under the new config.
    pub decisions: Option<crate::llm::decide::DecisionLayer>,
}

pub struct Agent {
    llm_runtime: SharedLlmRuntime,
    /// The turn runtime's settings and caches (context window, policy,
    /// media, prompt and render caches, turn ids).
    runtime: crate::turn_runtime::TurnRuntime,
    /// The gateway for model calls made outside the agent loop.
    side_calls: SideCallGateway,
    state: Arc<dyn StateStore>,
    memory_store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    event_store: Arc<EventStore>,
    tools: Vec<Arc<dyn Tool>>,
    model: RwLock<String>,
    fallback_model: RwLock<String>,
    system_prompt: String,
    config_path: PathBuf,
    /// How paths are classified for this daemon (its own files included).
    daemon_paths: aidaemon_tools::path_class::DaemonPaths,
    /// The process's one execution environment, built once at startup.
    execution_backend: aidaemon_tools::backend::SharedExecutionBackend,
    skills_dir: PathBuf,
    skill_cache: aidaemon_skills::SkillCache,
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
    child_progress: Option<aidaemon_channels::owner_wait::ProgressClock>,
    /// The sub-agents running now, by the chat they report to; shared with
    /// every child so `/status` and `/tasks` see the whole tree.
    running_children: Arc<crate::spawn::RunningChildren>,
    /// SDK kernel settings for this agent's turns; inherited by spawned children.
    sdk_engine: Arc<crate::engine::settings::SdkEngineSettings>,
    /// The kernel's durable run store: every SDK run journals its events and
    /// checkpoints its state here. Inherited by spawned children.
    run_store: Option<Arc<aidaemon_store_sqlite::SqliteStore>>,
    /// The owner's approval cards, grants and waiting cards; shared with
    /// every child, so a grant in a chat covers its children's calls.
    owner_approvals: Arc<crate::approvals::OwnerApprovals>,
    /// Long-term memory for SDK turns (hooks, tools, owned daemon tools).
    memory: Arc<dyn MemoryPlugin>,
    /// The daemon store's summarization and learning, run on every turn
    /// whatever the memory plugin.
    transcript_writes: Arc<dyn MemoryWrites>,
    /// How the file tools treat paths on this host.
    file_policy: Arc<dyn aidaemon_tools::FilePolicy>,
    /// The host's trace export, if any.
    trace: Option<Arc<dyn TraceExport>>,
    /// The host's computer-use pins, if it runs that tool.
    computer_use: Option<Arc<dyn crate::host::ComputerUseState>>,
    /// Goal cancellation token registry — shared across agent hierarchy.
    goal_token_registry: Option<GoalTokenRegistry>,
    /// Weak reference to the ChannelHub for background notifications.
    /// Uses RwLock because hub is created after Agent (core.rs ordering).
    hub: RwLock<Option<Weak<dyn OutboundRouter>>>,
    /// Plan store for the requirement-checklist feature. Set after construction
    /// (core.rs ordering) via `set_plan_store`; `None` on subagents and in tests,
    /// in which case checklist verification/recap degrade to current behavior.
    plan_store: RwLock<Option<Arc<aidaemon_work::plans::PlanStore>>>,
    /// Session IDs that have granted schedule confirmation for this process lifetime.
    /// Allows schedule creation to auto-confirm after an explicit AllowSession/AllowAlways.
    schedule_approved_sessions: Arc<RwLock<HashSet<String>>>,
    /// Models that recently returned 402 billing errors. Maps model name → failure time.
    /// Shared across parent/child agents. Entries expire after BILLING_FAIL_CACHE_TTL.
    billing_failed_models: Arc<RwLock<HashMap<String, Instant>>>,
    /// Weak self-reference for background task spawning.
    /// Set after Arc creation via `set_self_ref()`.
    self_ref: RwLock<Option<Weak<Agent>>>,
    /// The channels' per-session message queue. A root turn takes messages
    /// the same sender sends while it runs into the run (mid-turn steering).
    /// Set after channel startup via `set_task_registry()`.
    task_registry: std::sync::OnceLock<Arc<aidaemon_channels::tasks::TaskRegistry>>,
    /// Configured path alias roots (for example, `projects/...`).
    path_aliases: PathAliasConfig,
    /// Where received attachments are saved: daemon storage, never a project.
    inbox_dir: Option<PathBuf>,
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
    inherited_authority: Option<crate::authority::InheritedAuthority>,
    /// Full tool list from the root agent — used by TaskLead when spawning
    /// Executor children so they can access Action tools that were filtered
    /// out of the TaskLead's own `tools` vec.
    root_tools: Option<Vec<Arc<dyn Tool>>>,
    /// Some turn of this agent read (or started with) untrusted content.
    /// Kept in memory so a taint that could not be recorded still counts
    /// when a parent reads this child's result.
    read_untrusted: Arc<std::sync::atomic::AtomicBool>,
    /// An execution budget pinned for this agent's turns (tests pin one;
    /// the daemon never does). When `Some`, `select_initial_execution_budget`
    /// is bypassed and this budget is used instead.
    execution_budget_override: Option<crate::turn::execution_state::ExecutionBudget>,
    /// Per-process specialist registry (bundled + user-override `.md` files).
    /// Loaded once at agent construction and shared with every child agent
    /// spawned from this hierarchy.
    specialists: Arc<crate::specialists::SpecialistRegistry>,
    /// llama.cpp interactive KV-cache slot for the main generation loop.
    /// `Some(interactive_slot)` only when slot routing is enabled on the primary
    /// provider; `None` otherwise (and for all sub-agents). When set, it is
    /// applied to the main-loop `ChatOptions.id_slot` so the interactive
    /// conversation pins a dedicated slot that background tasks cannot evict.
    interactive_slot: Option<u32>,
}

impl Agent {
    /// A root agent from its parts and the host's ports.
    pub fn new(parts: AgentParts, ports: HostPorts) -> Self {
        let AgentParts {
            dependencies:
                AgentDependencies {
                    llm_runtime,
                    side_calls,
                    state,
                    memory_store,
                    event_store,
                    tools,
                },
            model,
            system_prompt,
            config_path,
            daemon_paths,
            execution_backend,
            skills_dir,
            max_depth,
            max_iterations,
            max_iterations_cap,
            max_response_chars,
            timeout_secs,
            max_facts,
            iteration_config,
            task_token_budget,
            spawn_family,
            llm_call_timeout_secs,
            mcp_registry,
            goal_token_registry,
            hub,
            record_decision_points,
            context_window_config,
            policy_config,
            path_aliases,
            inbox_dir,
            inherited_project_scope,
            specialists,
            interactive_slot,
            vision_config,
            audio_config,
            stt_config,
            harness_eval_config,
        } = parts;
        let HostPorts {
            memory,
            transcript_writes,
            file_policy,
            trace,
            computer_use,
        } = ports;
        crate::policy::metrics::init_policy_tunables_once(
            policy_config.uncertainty_clarify_threshold,
        );
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
            runtime: crate::turn_runtime::TurnRuntime::new(
                context_window_config,
                policy_config,
                vision_config,
                audio_config,
                stt_config,
                record_decision_points,
                harness_eval_config,
            ),
            llm_runtime,
            side_calls,
            state,
            memory_store,
            event_store,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            daemon_paths,
            execution_backend,
            skill_cache: aidaemon_skills::SkillCache::new(skills_dir.clone()),
            skills_dir,
            depth: 0,
            limits: AgentLimits {
                max_depth,
                iteration_config,
                max_iterations,
                max_iterations_cap,
                max_response_chars,
                timeout_secs,
                max_facts,
                llm_call_timeout: llm_call_timeout_secs.map(Duration::from_secs),
                task_token_budget,
                spawn_family,
            },
            model_override: RwLock::new(false),
            mcp_registry,
            role: AgentRole::Orchestrator,
            task_id: None,
            goal_id: None,
            mandate_execution: None,
            cancel_token: None,
            child_progress: None,
            running_children: Arc::default(),
            sdk_engine: Arc::default(),
            run_store: None,
            owner_approvals: Arc::default(),
            memory,
            transcript_writes,
            file_policy,
            trace,
            computer_use,
            goal_token_registry,
            hub: RwLock::new(hub),
            plan_store: RwLock::new(None),
            schedule_approved_sessions: Arc::new(RwLock::new(HashSet::new())),
            billing_failed_models: Arc::new(RwLock::new(HashMap::new())),
            self_ref: RwLock::new(None),
            task_registry: std::sync::OnceLock::new(),
            path_aliases,
            inbox_dir,
            inherited_project_scope,
            approval_session_id: None,
            inherited_authority: None,
            root_tools: None, // Root agent — its own tools ARE the root tools
            read_untrusted: Arc::default(),
            execution_budget_override: None,
            specialists,
            interactive_slot,
        }
    }

    /// The turn engine on this agent.
    pub fn engine(&self) -> crate::engine::turn_engine::TurnEngine<'_> {
        crate::engine::turn_engine::TurnEngine::new(self)
    }

    /// SDK kernel settings for this agent's turns (and its children's).
    pub fn set_sdk_engine(&mut self, settings: crate::engine::settings::SdkEngineSettings) {
        self.sdk_engine = Arc::new(settings);
    }

    /// The kernel run store for this agent's turns (and its children's).
    pub fn set_run_store(&mut self, store: Arc<aidaemon_store_sqlite::SqliteStore>) {
        self.run_store = Some(store);
    }

    /// The owner's approvals for this agent's turns (and its children's).
    pub fn owner_approvals(&self) -> Arc<crate::approvals::OwnerApprovals> {
        self.owner_approvals.clone()
    }

    /// The memory backend for this agent's turns (and its children's).
    pub fn set_memory_plugin(&mut self, memory: Arc<dyn MemoryPlugin>) {
        self.memory = memory;
    }

    /// Set the ChannelHub reference (called after hub creation).
    pub async fn set_hub(&self, hub: Weak<dyn OutboundRouter>) {
        *self.hub.write().await = Some(hub);
    }

    pub async fn runtime_wiring_ready(&self) -> bool {
        self.self_reference_ready().await
            && self
                .hub
                .read()
                .await
                .as_ref()
                .and_then(Weak::upgrade)
                .is_some()
    }

    pub async fn self_reference_ready(&self) -> bool {
        self.self_ref
            .read()
            .await
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some()
    }

    /// Set the plan store (called after construction). Enables the
    /// completion phase's checklist soft-verification + recap. Absent on
    /// subagents/tests, where checklist handling degrades to current behavior.
    pub async fn set_plan_store(&self, plan_store: Arc<aidaemon_work::plans::PlanStore>) {
        *self.plan_store.write().await = Some(plan_store);
    }

    /// Share the channels' task registry so a running root turn can take
    /// messages the same sender queues behind it (mid-turn steering).
    pub fn set_task_registry(&self, registry: Arc<aidaemon_channels::tasks::TaskRegistry>) {
        let _ = self.task_registry.set(registry);
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

    /// Maximum agentic loop iterations per invocation.
    pub fn max_iterations(&self) -> usize {
        self.limits.max_iterations
    }

    /// Role for this agent instance.
    pub fn role(&self) -> AgentRole {
        self.role
    }

    /// Read-only context-window diagnostics for the owner-facing `/context`
    /// command. No prompt content or private memory values are exposed.
    pub async fn context_debug_settings(
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

    /// Validate that an executor spawn targets a valid, pre-claimed task.
    ///
    /// This prevents duplicate/invalid execution when task leads attempt to
    /// spawn executors without claiming, with stale IDs, or against finished tasks.
    pub async fn validate_executor_task_for_spawn(
        &self,
        task_id: &str,
        expected_goal_id: Option<&str>,
    ) -> anyhow::Result<()> {
        crate::engine::work_scope::WorkScope::validate_executor_task_for_spawn(
            self,
            task_id,
            expected_goal_id,
        )
        .await
    }

    /// One-shot, TOOL-LESS interpretation of a finished background command's
    /// result (see
    /// [`TurnEngine::interpret_background_result`](crate::engine::turn_engine::TurnEngine::interpret_background_result)).
    /// Returns `None` on any failure or empty reply so the caller can fall
    /// back to delivering the raw result.
    pub async fn interpret_background_result(&self, command: &str, output: &str) -> Option<String> {
        self.engine()
            .interpret_background_result(command, output)
            .await
    }

    /// Run the agentic loop for a user message in the given session and
    /// return the final assistant text. `heartbeat` is an optional atomic
    /// timestamp updated on each activity point (channels pass one so the
    /// typing indicator can detect stalls).
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
        attachments: &[aidaemon_events::conversation::MessageAttachment],
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
    pub async fn handle_message_with_attachments_and_ingress(
        &self,
        session_id: &str,
        user_text: &str,
        attachments: &[aidaemon_events::conversation::MessageAttachment],
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
        ingress_timing: Option<aidaemon_channels::ports::InboundMessageTiming>,
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
    pub async fn handle_internal_continuation(
        &self,
        request: &crate::spawn::ConversationRequest,
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

    /// Put a reloaded provider in place without a restart: the model
    /// backends and router, the primary and fallback models (a manual
    /// `/model` pin is cleared) and the side calls' decision layer. Returns
    /// the owner-facing summary.
    pub async fn swap_provider(&self, swap: ProviderSwap) -> anyhow::Result<String> {
        let ProviderSwap {
            provider,
            models,
            primary_model: new_primary,
            router,
            provider_kind,
            failover_targets,
            fallback_model: new_fallback,
            decisions,
        } = swap;
        let old_model = match tokio::time::timeout(Duration::from_secs(2), self.model.read()).await
        {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring model lock during provider reload");
                self.llm_runtime.snapshot().primary_model()
            }
        };

        let old_runtime = self.llm_runtime.swap(
            provider,
            router,
            provider_kind,
            new_primary.clone(),
            failover_targets,
            models,
        );

        {
            let mut model = tokio::time::timeout(Duration::from_secs(2), self.model.write())
                .await
                .map_err(|_| anyhow::anyhow!("timed out acquiring model write lock"))?;
            let mut fallback =
                tokio::time::timeout(Duration::from_secs(2), self.fallback_model.write())
                    .await
                    .map_err(|_| anyhow::anyhow!("timed out acquiring fallback write lock"))?;
            *model = new_primary.clone();
            *fallback = new_fallback.clone();
        }
        let manual_model_cleared;
        {
            let mut override_guard =
                tokio::time::timeout(Duration::from_secs(2), self.model_override.write())
                    .await
                    .map_err(|_| anyhow::anyhow!("timed out acquiring model_override lock"))?;
            manual_model_cleared = *override_guard;
            *override_guard = false;
        }

        // The decision layer follows the reloaded config, so a Gemini key
        // set from chat (`/setkey`) takes effect without a restart.
        self.side_calls.set_decisions(decisions);

        info!(
            old_provider = ?old_runtime.provider_kind(),
            new_provider = ?provider_kind,
            old_model = %old_model,
            new_model = %new_primary,
            new_fallback = %new_fallback,
            "Provider runtime reloaded"
        );

        Ok(aidaemon_providers::routing::ProviderReload {
            old_provider: old_runtime.provider_kind(),
            new_provider: provider_kind,
            old_model,
            new_model: new_primary,
            manual_model_cleared,
        }
        .summary())
    }

    /// HARD-clear a session (destructive: deletes events). Backs `/wipe`.
    pub async fn clear_session(&self, session_id: &str) -> anyhow::Result<()> {
        self.state.clear_session(session_id).await
    }

    /// Reset a session's conversation context non-destructively (events remain
    /// for memory + audit). Backs `/clear`. Also drops the in-memory turn
    /// anchor so the next context build re-initializes past the new boundary,
    /// and the deferred tools the session loaded, so it starts from the core
    /// tools again.
    pub async fn clear_session_context(&self, session_id: &str) -> anyhow::Result<()> {
        self.state.clear_session_context(session_id).await?;
        self.runtime.turn_anchors.write().await.remove(session_id);
        crate::engine::loaded_tools::clear(session_id);
        Ok(())
    }

    /// List available models from the provider.
    pub async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        self.llm_runtime.snapshot().models().list().await
    }

    /// Stamp the current config as "last known good" — called after a
    /// successful LLM response proves the config actually works.
    async fn stamp_lastgood_config(&self) {
        let lastgood = self.config_path.with_extension("toml.lastgood");
        if let Err(e) = tokio::fs::copy(&self.config_path, &lastgood).await {
            warn!(error = %e, "Failed to stamp lastgood config");
        }
    }
}
