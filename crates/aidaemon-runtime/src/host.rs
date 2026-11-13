//! What the turn runtime asks of its host: a few cohesive traits the daemon's
//! agent implements, so the runtime's handlers run on any host that supplies
//! the same parts.
//!
//! - [`TurnStores`]: the conversation's durable state (dialogue state and the
//!   event log).
//! - [`TurnWorkspace`]: where the turn's file work happens (the execution
//!   backend and the configured project roots).
//! - [`DecisionRecorder`]: where the turn's decision points go.
//! - [`EngineContext`]: what the turn engine is handed with the agent (the
//!   daemon's paths, the engine settings, the run store, the approval chat
//!   and a child's links to its parent).
//! - [`TurnServices`]: the registries and per-run handles a turn uses (the
//!   task registry, plans, MCP tools, transcript writes, cancellation, the
//!   interactive slot and the turn's limits).
//! - [`EngineHostOps`]: what only the host can do for the engine (queue a
//!   request for provider recovery).
//! - [`TurnHost`]: all of the above plus the agent's turn runtime and who
//!   the agent is; the turn's handlers in this crate run on it.

use aidaemon_events::stores::{DialogueStateStore, MessageStore};
use aidaemon_events::EventStore;
use aidaemon_tools::ExecutionBackend;

/// The conversation's durable state: per-session dialogue state and the
/// event log the turn records into and rebuilds from.
pub trait TurnStores: Send + Sync {
    /// Per-session dialogue state (open request, open question, last turns).
    fn dialogue_states(&self) -> &dyn DialogueStateStore;
    /// The conversation's messages and its durable context boundary.
    fn messages(&self) -> &dyn MessageStore;
    /// The rendered core prompts, by hash, for replaying past model calls.
    fn prompt_snapshots(&self) -> &dyn aidaemon_events::stores::PromptSnapshotStore;
    /// Tasks (a mandate executor's prompt reads the task it was claimed for).
    fn tasks(&self) -> &dyn aidaemon_work::stores::TaskStore;
    /// Mandates (a mandate worker's prompt reads its authority and history).
    fn mandates(&self) -> &dyn aidaemon_work::stores::MandateStore;
    /// The conversation's cumulative summaries.
    fn summaries(&self) -> &dyn aidaemon_events::stores::ConversationSummaryStore;
    /// Goals (a goal's bound workspace scopes its tasks' turns).
    fn goals(&self) -> &dyn aidaemon_work::stores::GoalStore;
    /// The work stores beyond the conversation: tasks and their activity
    /// log, goals' schedules and runs, a scheduled run's persisted budget
    /// state, and the notices delivered to the owner.
    fn work(&self) -> &dyn aidaemon_work::scheduler::SchedulerStore;
    /// Token usage, and the mandate runs' token leases.
    fn token_usage(&self) -> &dyn aidaemon_events::provider::TokenUsageStore;
    /// The bots added at runtime.
    fn dynamic_bots(&self) -> &dyn aidaemon_channels::hub::DynamicBotStore;
    /// The event log.
    fn events(&self) -> &EventStore;
    /// A shared handle to the event log, for emitters and spawned work.
    fn event_log(&self) -> std::sync::Arc<EventStore>;
}

/// Where the turn's file work happens.
pub trait TurnWorkspace: Send + Sync {
    /// The execution backend file and command work runs on.
    fn execution_backend(&self) -> &dyn ExecutionBackend;
    /// Configured project roots path references resolve against.
    fn project_roots(&self) -> &[String];
}

/// Where a turn's decision points go: the record of why the runtime chose
/// what it did, kept with the turn's events when the host records them.
#[async_trait::async_trait]
pub trait DecisionRecorder: Send + Sync {
    /// Record one decision point of `task_id` at loop `iteration`.
    async fn record_decision_point(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: aidaemon_events::DecisionType,
        summary: String,
        metadata: serde_json::Value,
    );
}

/// What the turn engine is handed with the agent.
pub trait EngineContext: Send + Sync {
    /// The daemon's own paths (config, data, logs), which file policy and
    /// path classification read.
    fn daemon_paths(&self) -> &aidaemon_tools::path_class::DaemonPaths;
    /// How turns run on the SDK kernel; shared by every agent in the tree.
    fn engine_settings(&self) -> &std::sync::Arc<crate::engine::settings::SdkEngineSettings>;
    /// The durable kernel run store, when runs are recorded.
    fn run_store(&self) -> Option<&std::sync::Arc<aidaemon_store_sqlite::SqliteStore>>;
    /// The chat whose approval grants a child's turns share; `None` for a
    /// root agent (its own session's grants apply).
    fn approval_session_id(&self) -> Option<&str>;
    /// The owner's approvals: pending cards and session grants.
    fn owner_approvals(&self) -> &std::sync::Arc<crate::approvals::OwnerApprovals>;
    /// A spawned child's progress clock, read by its parent's watchdog.
    fn child_progress(&self) -> Option<&aidaemon_channels::owner_wait::ProgressClock>;
    /// Set once this agent's turn read untrusted content; its parent reads it.
    fn untrusted_read_flag(&self) -> &std::sync::Arc<std::sync::atomic::AtomicBool>;
    /// The sub-agents running for the requests of this agent's tree.
    fn running_children(&self) -> &std::sync::Arc<crate::spawn::RunningChildren>;
}

/// The limits one turn runs under.
#[derive(Debug, Clone, Copy, Default)]
pub struct TurnLimits {
    /// The configured per-call model timeout (`None`: the engine default).
    pub llm_call_timeout: Option<std::time::Duration>,
    /// The token budget of one task.
    pub task_token_budget: Option<u64>,
    /// How many sub-agents one request's tree may run.
    pub spawn_family: crate::authority::SpawnFamilyLimits,
}

/// The registries and per-run handles a turn uses.
#[async_trait::async_trait]
pub trait TurnServices: Send + Sync {
    /// The running tasks of every chat (`/tasks`, cancel).
    fn task_registry(&self) -> Option<&std::sync::Arc<aidaemon_channels::tasks::TaskRegistry>>;
    /// The task plans, when plans are enabled.
    async fn plan_store(&self) -> Option<std::sync::Arc<aidaemon_work::plans::PlanStore>>;
    /// The available MCP tool named `name`, if a connected server offers it.
    async fn mcp_tool(&self, name: &str)
        -> Option<std::sync::Arc<dyn aidaemon_tools::OutcomeTool>>;
    /// Where a turn's memory side effects go (extraction, summaries,
    /// learning).
    fn transcript_writes(&self) -> std::sync::Arc<dyn crate::turn::memory_writer::MemoryWrites>;
    /// Cancels this agent's run (a child or a background task).
    fn cancel_token(&self) -> Option<&tokio_util::sync::CancellationToken>;
    /// The local model slot an interactive turn pins to keep its cache warm.
    fn interactive_slot(&self) -> Option<u32>;
    /// The limits this agent's turns run under.
    fn turn_limits(&self) -> TurnLimits;
}

/// What only the host can do for the turn engine.
#[async_trait::async_trait]
pub trait EngineHostOps: Send + Sync {
    /// Queue (or re-queue) a request whose provider failed before any tool
    /// work committed, to run again once the provider recovers. Returns the
    /// attempt number that will run next, or `None` when the request is not
    /// deferrable or has exhausted its cap.
    async fn defer_request_for_provider_recovery(
        &self,
        session_id: &str,
        user_text: &str,
        user_role: aidaemon_tools::semantics::UserRole,
        channel_ctx: &aidaemon_channels::context::ChannelContext,
        error: &anyhow::Error,
    ) -> Option<u16>;
    /// Where kernel runs are traced (an OTLP export), if anywhere.
    fn trace_sink(&self) -> Option<std::sync::Arc<dyn aidaemon_core::EventSink>>;
    /// Report the messages actually sent for a model call of `run` to the
    /// trace export: only these show what the model received.
    fn observe_sent(&self, run: aidaemon_core::RunId, messages: &[serde_json::Value]);
    /// Computer use's per-task state, when the host runs computer use.
    fn computer_use(&self) -> Option<std::sync::Arc<dyn ComputerUseState>>;
}

/// What a computer-use GUI loop records per task, as the turn engine reads
/// it: the vision model the task is pinned to, and whether its last
/// coordinate click is still unconfirmed.
#[async_trait::async_trait]
pub trait ComputerUseState: Send + Sync {
    /// The model the task is pinned to, if a GUI loop pinned one.
    async fn pinned_model(&self, task_id: &str) -> Option<String>;
    /// A coordinate click of the task landed and no follow-up observation
    /// has confirmed it yet.
    async fn has_unverified_coordinate_click(&self, task_id: &str) -> bool;
}

/// The host a turn runs on: its stores, workspace and decision record, the
/// agent's [`TurnRuntime`] and where the agent stands in its tree.
#[async_trait::async_trait]
pub trait TurnHost:
    TurnStores + TurnWorkspace + DecisionRecorder + EngineContext + TurnServices + EngineHostOps
{
    /// The agent's turn settings and caches.
    fn turn_runtime(&self) -> &crate::turn_runtime::TurnRuntime;
    /// Whether this agent works for another (a spawned child, or a mandate's
    /// worker): its transcript is an authority envelope, not a conversation
    /// with the user.
    fn is_worker(&self) -> bool;
    /// The mandate run this agent works in, if it is a mandate's worker.
    fn mandate_execution(&self) -> Option<&crate::turn_runtime::MandateExecutionFence>;
    /// Whether this agent is a mandate's worker (bound to one decision
    /// cycle's run).
    fn works_for_mandate(&self) -> bool {
        self.mandate_execution().is_some()
    }
    /// The configured persona: the base of the system prompt.
    fn persona(&self) -> &str;
    /// The session-static tool roster the core prompt names: `(name,
    /// serialized schema)`, sorted by name.
    fn session_static_tool_roster(
        &self,
        user_role: aidaemon_tools::semantics::UserRole,
        channel_ctx: &aidaemon_channels::context::ChannelContext,
    ) -> Vec<(String, String)>;
    /// Whether a tool named `name` is registered.
    fn has_registered_tool(&self, name: &str) -> bool;
    /// The specialists a parent may name, with their descriptions.
    fn specialist_kinds(&self) -> Vec<(String, String)>;
    /// The bundled specialists only (no user overrides), for a mandate
    /// worker's prompt.
    fn bundled_specialist_kinds(&self) -> Vec<(String, String)>;
    /// The memory system serving the agent's turns.
    fn memory_plugin(&self) -> &dyn crate::turn::memory::MemoryPlugin;
    /// Whether the memory system is the daemon's own.
    fn memory_is_daemons_own(&self) -> bool {
        self.memory_plugin().is_daemons_own()
    }
    /// Whether the memory system serves the daemon tool `name` itself.
    fn memory_owns_daemon_tool(&self, name: &str) -> bool {
        self.memory_plugin().owned_daemon_tools().contains(&name)
    }
    /// A shared handle to the execution backend.
    fn execution_backend_handle(&self) -> aidaemon_tools::backend::SharedExecutionBackend;
    /// What a spawned child inherited from the turn that spawned it; `None`
    /// for a root agent.
    fn inherited_authority(&self) -> Option<&crate::authority::InheritedAuthority>;
    /// The agent's role in its tree.
    fn role(&self) -> aidaemon_events::traits::AgentRole;
    /// The goal a task lead works for.
    fn goal_id(&self) -> Option<&str>;
    /// The project scope a spawned child was handed.
    fn inherited_project_scope(&self) -> Option<&str>;
    /// Where received attachments are saved: daemon storage, never a project.
    fn inbox_dir(&self) -> Option<&std::path::Path>;
    /// How deep this agent is in its tree (0 is the root agent).
    fn depth(&self) -> usize;
    /// Settings: default alert sessions, goal budget overrides.
    fn settings(&self) -> &dyn aidaemon_tools::services::SettingsStore;
    /// Goals' daily token counters.
    fn goal_budgets(&self) -> &dyn aidaemon_work::stores::GoalBudgetStore;
    /// The notification queue token alerts go through.
    fn notifications(&self) -> &dyn aidaemon_work::stores::NotificationStore;
    /// The goals' shared token budgets, when the agent tracks them.
    fn goal_tokens(&self) -> Option<&aidaemon_work::goal_tokens::GoalTokenRegistry>;
    /// The chat hub: notices to the owner and inline approval cards. Set
    /// after the agent is built, so it may be empty.
    fn hub(
        &self,
    ) -> &tokio::sync::RwLock<Option<std::sync::Weak<dyn aidaemon_channels::ports::OutboundRouter>>>;
    /// Record a scheduled run's budget state after an extension.
    async fn persist_scheduled_run_state(
        &self,
        goal_id: &str,
        status: &aidaemon_work::goal_tokens::GoalRunBudgetStatus,
    );
    /// The durable task this agent was started for (executors and run-bound
    /// task leads).
    fn task_id(&self) -> Option<&str>;
    /// The model runtime (primary, router, failover chain).
    fn llm_runtime(&self) -> &crate::llm::SharedLlmRuntime;
    /// The door for model calls made outside the agent loop.
    fn side_calls(&self) -> &crate::llm::gateway::SideCallGateway;
    /// The model this agent runs on.
    fn model_lock(&self) -> &tokio::sync::RwLock<String>;
    /// Set when the owner pinned the model (`/model`), so routing leaves it.
    fn model_override_lock(&self) -> &tokio::sync::RwLock<bool>;
    /// The stored fallback model, tried first when a call fails.
    fn fallback_model_lock(&self) -> &tokio::sync::RwLock<String>;
    /// Models that recently answered with a billing error, and when; shared
    /// across the agent tree.
    fn billing_failed_models(
        &self,
    ) -> &std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, std::time::Instant>>>;
    /// Record the configuration as last known good (a model call succeeded).
    async fn stamp_lastgood(&self);
    /// A shared handle to the message store, for spawned work.
    fn message_log(&self) -> std::sync::Arc<dyn MessageStore>;
    /// A shared handle to what conversation compaction reads and writes.
    fn compaction_store(&self) -> std::sync::Arc<dyn crate::turn::compaction::CompactionStore>;
    /// The skill library.
    fn skills(&self) -> &aidaemon_skills::SkillCache;
    /// Every registered tool's definition and capabilities.
    fn registered_tool_definitions_with_capabilities(
        &self,
    ) -> (
        Vec<serde_json::Value>,
        std::collections::HashMap<String, aidaemon_tools::semantics::ToolCapabilities>,
    );
    /// Drop from `defs` and `capabilities` the tools not offered on a turn
    /// of `session_id` (each tool's `ToolSemantics::offered_in_session`).
    /// The default keeps everything.
    fn retain_offered_in_session(
        &self,
        _defs: &mut Vec<serde_json::Value>,
        _capabilities: &mut std::collections::HashMap<
            String,
            aidaemon_tools::semantics::ToolCapabilities,
        >,
        _session_id: &str,
    ) {
    }
    /// The tools offered for `user_message`: the registered ones plus the MCP
    /// tools it calls for.
    async fn tool_definitions_with_capabilities(
        &self,
        user_message: &str,
    ) -> (
        Vec<serde_json::Value>,
        std::collections::HashMap<String, aidaemon_tools::semantics::ToolCapabilities>,
    );
    /// `defs` narrowed to what `policy` allows at `risk_score`.
    fn filter_tool_definitions_for_policy(
        &self,
        defs: &[serde_json::Value],
        capabilities: &std::collections::HashMap<
            String,
            aidaemon_tools::semantics::ToolCapabilities,
        >,
        policy: &crate::policy::execution::ExecutionPolicy,
        risk_score: f32,
        widen: bool,
    ) -> Vec<serde_json::Value>;
    /// Run one tool call with the host's tool execution (its watchdog,
    /// mandate validation and workspace scoping).
    async fn execute_tool_with_watchdog_outcome(
        &self,
        name: &str,
        arguments: &str,
        ctx: &crate::engine::tool_exec::ToolExecCtx<'_>,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome>;
    /// Cancel the finite work the session started (background task leads
    /// and executors); what was cancelled.
    async fn cancel_active_finite_work_for_session(&self, session_id: &str) -> Vec<String>;
    /// Build a child of this agent: a fresh agent sharing this one's model
    /// runtime, stores, hub, approvals and sub-agent registry.
    async fn create_child(
        &self,
        spec: crate::spawn::ChildSpec,
    ) -> std::sync::Arc<dyn crate::spawn::ChildAgent>;
    /// The agent's tools (their task-end hooks run when a task ends).
    fn tools(&self) -> &[std::sync::Arc<dyn aidaemon_tools::OutcomeTool>];
}
