//! The agent as the turn engine's host.

use aidaemon_events::EventStore;
use aidaemon_events::stores::{DialogueStateStore, MessageStore};
use aidaemon_tools::ExecutionBackend;

use super::Agent;
use crate::host::{
    DecisionRecorder, EngineContext, EngineHostOps, TurnHost, TurnLimits, TurnServices, TurnStores,
    TurnWorkspace,
};
use crate::turn_runtime::TurnRuntime;

impl TurnStores for Agent {
    fn dialogue_states(&self) -> &dyn DialogueStateStore {
        &*self.state
    }

    fn messages(&self) -> &dyn MessageStore {
        &*self.state
    }

    fn prompt_snapshots(&self) -> &dyn aidaemon_events::stores::PromptSnapshotStore {
        &*self.state
    }

    fn tasks(&self) -> &dyn aidaemon_work::stores::TaskStore {
        &*self.state
    }

    fn mandates(&self) -> &dyn aidaemon_work::stores::MandateStore {
        &*self.state
    }

    fn summaries(&self) -> &dyn aidaemon_events::stores::ConversationSummaryStore {
        &*self.state
    }

    fn goals(&self) -> &dyn aidaemon_work::stores::GoalStore {
        &*self.state
    }

    fn work(&self) -> &dyn aidaemon_work::scheduler::SchedulerStore {
        &*self.state
    }

    fn token_usage(&self) -> &dyn aidaemon_events::provider::TokenUsageStore {
        &*self.state
    }

    fn dynamic_bots(&self) -> &dyn aidaemon_channels::hub::DynamicBotStore {
        &*self.state
    }

    fn events(&self) -> &EventStore {
        &self.event_store
    }

    fn event_log(&self) -> std::sync::Arc<EventStore> {
        self.event_store.clone()
    }
}

impl TurnWorkspace for Agent {
    fn execution_backend(&self) -> &dyn ExecutionBackend {
        &*self.execution_backend
    }

    fn project_roots(&self) -> &[String] {
        &self.path_aliases.projects
    }
}

#[async_trait::async_trait]
impl DecisionRecorder for Agent {
    async fn record_decision_point(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: aidaemon_events::DecisionType,
        summary: String,
        metadata: serde_json::Value,
    ) {
        self.engine()
            .emit_decision_point(
                emitter,
                task_id,
                iteration,
                decision_type,
                summary,
                metadata,
            )
            .await;
    }
}

impl EngineContext for Agent {
    fn daemon_paths(&self) -> &aidaemon_tools::path_class::DaemonPaths {
        &self.daemon_paths
    }

    fn engine_settings(&self) -> &std::sync::Arc<crate::engine::settings::SdkEngineSettings> {
        &self.sdk_engine
    }

    fn run_store(&self) -> Option<&std::sync::Arc<aidaemon_store_sqlite::SqliteStore>> {
        self.run_store.as_ref()
    }

    fn approval_session_id(&self) -> Option<&str> {
        self.approval_session_id.as_deref()
    }

    fn owner_approvals(&self) -> &std::sync::Arc<crate::approvals::OwnerApprovals> {
        &self.owner_approvals
    }

    fn child_progress(&self) -> Option<&aidaemon_channels::owner_wait::ProgressClock> {
        self.child_progress.as_ref()
    }

    fn untrusted_read_flag(&self) -> &std::sync::Arc<std::sync::atomic::AtomicBool> {
        &self.read_untrusted
    }

    fn running_children(&self) -> &std::sync::Arc<crate::spawn::RunningChildren> {
        &self.running_children
    }

    fn file_policy(&self) -> std::sync::Arc<dyn aidaemon_tools::FilePolicy> {
        self.file_policy.clone()
    }
}

#[async_trait::async_trait]
impl TurnServices for Agent {
    fn task_registry(&self) -> Option<&std::sync::Arc<aidaemon_channels::tasks::TaskRegistry>> {
        self.task_registry.get()
    }

    async fn plan_store(&self) -> Option<std::sync::Arc<aidaemon_work::plans::PlanStore>> {
        self.plan_store.read().await.clone()
    }

    async fn mcp_tool(
        &self,
        name: &str,
    ) -> Option<std::sync::Arc<dyn aidaemon_tools::OutcomeTool>> {
        self.mcp_registry.as_ref()?.find_tool(name).await
    }

    async fn mcp_tools_matching(
        &self,
        user_message: &str,
    ) -> Vec<std::sync::Arc<dyn aidaemon_tools::OutcomeTool>> {
        match self.mcp_registry.as_ref() {
            Some(registry) => registry.match_tools(user_message).await,
            None => Vec::new(),
        }
    }

    async fn mcp_tool_names(&self) -> Vec<String> {
        let Some(registry) = self.mcp_registry.as_ref() else {
            return Vec::new();
        };
        registry
            .list_servers()
            .await
            .into_iter()
            .flat_map(|info| info.tool_names)
            .collect()
    }

    fn transcript_writes(&self) -> std::sync::Arc<dyn crate::turn::memory_writer::MemoryWrites> {
        self.transcript_writes.clone()
    }

    fn cancel_token(&self) -> Option<&tokio_util::sync::CancellationToken> {
        self.cancel_token.as_ref()
    }

    fn interactive_slot(&self) -> Option<u32> {
        self.interactive_slot
    }

    fn turn_limits(&self) -> TurnLimits {
        TurnLimits {
            llm_call_timeout: self.limits.llm_call_timeout,
            task_token_budget: self.limits.task_token_budget,
            spawn_family: self.limits.spawn_family,
            max_depth: self.limits.max_depth,
            max_iterations_cap: self.limits.max_iterations_cap,
            timeout_cap_secs: self.limits.timeout_cap(),
        }
    }
}

#[async_trait::async_trait]
impl EngineHostOps for Agent {
    fn trace_sink(&self) -> Option<std::sync::Arc<dyn aidaemon_core::EventSink>> {
        self.trace.as_ref()?.sink()
    }

    fn observe_sent(&self, run: aidaemon_core::RunId, messages: &[serde_json::Value]) {
        if let Some(trace) = self.trace.as_ref() {
            trace.observe_sent(run, messages);
        }
    }

    fn computer_use(&self) -> Option<std::sync::Arc<dyn crate::host::ComputerUseState>> {
        self.computer_use.clone()
    }
}

#[async_trait::async_trait]
impl TurnHost for Agent {
    fn turn_runtime(&self) -> &TurnRuntime {
        &self.runtime
    }

    fn is_worker(&self) -> bool {
        self.depth > 0 || self.mandate_execution.is_some()
    }

    fn mandate_execution(&self) -> Option<&crate::turn_runtime::MandateExecutionFence> {
        self.mandate_execution.as_ref()
    }

    fn persona(&self) -> &str {
        &self.system_prompt
    }

    fn specialist_kinds(&self) -> Vec<(String, String)> {
        self.specialists
            .llm_visible_kinds()
            .into_iter()
            .map(|(name, desc)| (name.to_string(), desc))
            .collect()
    }

    fn bundled_specialist_kinds(&self) -> Vec<(String, String)> {
        crate::specialists::SpecialistRegistry::load(None)
            .llm_visible_kinds()
            .into_iter()
            .map(|(name, description)| (name.to_string(), description))
            .collect()
    }

    fn memory_plugin(&self) -> &dyn crate::turn::memory::MemoryPlugin {
        &*self.memory
    }

    fn execution_backend_handle(&self) -> aidaemon_tools::backend::SharedExecutionBackend {
        self.execution_backend.clone()
    }

    fn role(&self) -> aidaemon_events::traits::AgentRole {
        Agent::role(self)
    }

    fn goal_id(&self) -> Option<&str> {
        self.goal_id.as_deref()
    }

    fn inherited_project_scope(&self) -> Option<&str> {
        self.inherited_project_scope.as_deref()
    }

    fn inbox_dir(&self) -> Option<&std::path::Path> {
        self.inbox_dir.as_deref()
    }

    fn inherited_authority(&self) -> Option<&crate::authority::InheritedAuthority> {
        self.inherited_authority.as_ref()
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn settings(&self) -> &dyn aidaemon_tools::services::SettingsStore {
        &*self.state
    }

    fn settings_store(&self) -> std::sync::Arc<dyn aidaemon_tools::services::SettingsStore> {
        self.state.clone()
    }

    fn goal_budgets(&self) -> &dyn aidaemon_work::stores::GoalBudgetStore {
        &*self.state
    }

    fn notifications(&self) -> &dyn aidaemon_work::stores::NotificationStore {
        &*self.state
    }

    fn goal_tokens(&self) -> Option<&aidaemon_work::goal_tokens::GoalTokenRegistry> {
        self.goal_token_registry.as_ref()
    }

    fn hub(
        &self,
    ) -> &tokio::sync::RwLock<Option<std::sync::Weak<dyn aidaemon_channels::ports::OutboundRouter>>>
    {
        &self.hub
    }

    async fn persist_scheduled_run_state(
        &self,
        goal_id: &str,
        status: &aidaemon_work::goal_tokens::GoalRunBudgetStatus,
    ) {
        crate::goal_dispatch::persist_scheduled_run_state(&self.state, goal_id, None, status).await;
    }

    fn task_id(&self) -> Option<&str> {
        self.task_id.as_deref()
    }

    fn llm_runtime(&self) -> &crate::llm::SharedLlmRuntime {
        &self.llm_runtime
    }

    fn side_calls(&self) -> &crate::llm::gateway::SideCallGateway {
        &self.side_calls
    }

    fn model_lock(&self) -> &tokio::sync::RwLock<String> {
        &self.model
    }

    fn model_override_lock(&self) -> &tokio::sync::RwLock<bool> {
        &self.model_override
    }

    fn fallback_model_lock(&self) -> &tokio::sync::RwLock<String> {
        &self.fallback_model
    }

    fn billing_failed_models(
        &self,
    ) -> &std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, std::time::Instant>>>
    {
        &self.billing_failed_models
    }

    async fn stamp_lastgood(&self) {
        self.stamp_lastgood_config().await;
    }

    fn message_log(&self) -> std::sync::Arc<dyn MessageStore> {
        self.state.clone()
    }

    fn compaction_store(&self) -> std::sync::Arc<dyn crate::turn::compaction::CompactionStore> {
        self.state.clone()
    }

    fn skills(&self) -> &aidaemon_skills::SkillCache {
        &self.skill_cache
    }

    async fn create_child(
        &self,
        spec: crate::spawn::ChildSpec,
    ) -> std::sync::Arc<dyn crate::spawn::ChildAgent> {
        self.create_child_agent(spec).await
    }

    fn root_tools(&self) -> Option<&[std::sync::Arc<dyn aidaemon_tools::OutcomeTool>]> {
        self.root_tools.as_deref()
    }

    fn specialists(&self) -> &std::sync::Arc<crate::specialists::SpecialistRegistry> {
        &self.specialists
    }

    fn goal_tasks_tool(
        &self,
        goal_id: &str,
        goal_run_id: Option<String>,
    ) -> std::sync::Arc<dyn aidaemon_tools::OutcomeTool> {
        std::sync::Arc::new(
            aidaemon_work::tools::manage_goal_tasks::ManageGoalTasksTool::new(
                goal_id.to_string(),
                self.state.clone(),
            )
            .with_goal_run_id(goal_run_id),
        )
    }

    fn report_blocker_tool(
        &self,
        task_id: &str,
        attempt: aidaemon_work::traits::TaskAttempt,
        mandate: bool,
    ) -> std::sync::Arc<dyn aidaemon_tools::OutcomeTool> {
        std::sync::Arc::new(
            crate::tools::report_blocker::ReportBlockerTool::for_attempt(
                task_id.to_string(),
                self.state.clone(),
                attempt,
                mandate,
            ),
        )
    }

    fn tools(&self) -> &[std::sync::Arc<dyn aidaemon_tools::OutcomeTool>] {
        &self.tools
    }

    fn execution_budget_override(&self) -> Option<crate::turn::execution_state::ExecutionBudget> {
        self.execution_budget_override.clone()
    }
}

#[async_trait::async_trait]
impl crate::spawn::ChildAgent for Agent {
    async fn handle_message(
        &self,
        session_id: &str,
        user_text: &str,
        status_tx: Option<tokio::sync::mpsc::Sender<aidaemon_tools::status::StatusUpdate>>,
        user_role: aidaemon_tools::semantics::UserRole,
        channel_ctx: aidaemon_channels::context::ChannelContext,
        heartbeat: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
    ) -> anyhow::Result<String> {
        Agent::handle_message(
            self,
            session_id,
            user_text,
            status_tx,
            user_role,
            channel_ctx,
            heartbeat,
        )
        .await
    }

    fn read_untrusted(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.read_untrusted.clone()
    }
}
