//! What the turn runtime asks of its host: a few cohesive traits the daemon's
//! agent implements, so the runtime's handlers run on any host that supplies
//! the same parts.
//!
//! - [`TurnStores`]: the conversation's durable state (dialogue state and the
//!   event log).
//! - [`TurnWorkspace`]: where the turn's file work happens (the execution
//!   backend and the configured project roots).
//! - [`DecisionRecorder`]: where the turn's decision points go.
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

/// The host a turn runs on: its stores, workspace and decision record, the
/// agent's [`TurnRuntime`] and where the agent stands in its tree.
#[async_trait::async_trait]
pub trait TurnHost: TurnStores + TurnWorkspace + DecisionRecorder {
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
    /// Whether the memory system is the daemon's own.
    fn memory_is_daemons_own(&self) -> bool;
    /// Whether the memory system serves the daemon tool `name` itself.
    fn memory_owns_daemon_tool(&self, name: &str) -> bool;
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
    /// Stop the CLI agents running for the session (a stop command); the
    /// tool's reply.
    async fn cancel_cli_agents(
        &self,
        session_id: &str,
        task_id: &str,
        status_tx: Option<tokio::sync::mpsc::Sender<aidaemon_tools::status::StatusUpdate>>,
        channel_ctx: &aidaemon_channels::context::ChannelContext,
        user_role: aidaemon_tools::semantics::UserRole,
    ) -> anyhow::Result<String>;
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
