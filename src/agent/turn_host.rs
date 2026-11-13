//! The daemon's agent as the turn runtime's host.

use aidaemon_events::stores::{DialogueStateStore, MessageStore};
use aidaemon_runtime::host::{DecisionRecorder, TurnHost, TurnStores, TurnWorkspace};
use aidaemon_runtime::turn_runtime::TurnRuntime;

use super::Agent;
use crate::events::EventStore;
use crate::execution::ExecutionBackend;

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
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: crate::events::DecisionType,
        summary: String,
        metadata: serde_json::Value,
    ) {
        self.emit_decision_point(
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

#[async_trait::async_trait]
impl TurnHost for Agent {
    fn turn_runtime(&self) -> &TurnRuntime {
        &self.runtime
    }

    fn is_worker(&self) -> bool {
        self.depth > 0 || self.mandate_execution.is_some()
    }

    fn mandate_execution(&self) -> Option<&aidaemon_runtime::turn_runtime::MandateExecutionFence> {
        self.mandate_execution.as_ref()
    }

    fn persona(&self) -> &str {
        &self.system_prompt
    }

    fn session_static_tool_roster(
        &self,
        user_role: crate::types::UserRole,
        channel_ctx: &crate::types::ChannelContext,
    ) -> Vec<(String, String)> {
        Agent::session_static_tool_roster(self, user_role, channel_ctx)
    }

    fn has_registered_tool(&self, name: &str) -> bool {
        Agent::has_registered_tool(self, name)
    }

    fn specialist_kinds(&self) -> Vec<(String, String)> {
        self.specialists
            .llm_visible_kinds()
            .into_iter()
            .map(|(name, desc)| (name.to_string(), desc))
            .collect()
    }

    fn bundled_specialist_kinds(&self) -> Vec<(String, String)> {
        super::specialists::SpecialistRegistry::load(None)
            .llm_visible_kinds()
            .into_iter()
            .map(|(name, description)| (name.to_string(), description))
            .collect()
    }

    fn memory_is_daemons_own(&self) -> bool {
        self.memory.is_daemons_own()
    }

    fn memory_owns_daemon_tool(&self, name: &str) -> bool {
        self.memory.owned_daemon_tools().contains(&name)
    }

    fn execution_backend_handle(&self) -> crate::execution::SharedExecutionBackend {
        self.execution_backend.clone()
    }

    fn role(&self) -> crate::traits::AgentRole {
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

    fn inherited_authority(&self) -> Option<&aidaemon_runtime::authority::InheritedAuthority> {
        self.inherited_authority.as_ref()
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn settings(&self) -> &dyn aidaemon_tools::services::SettingsStore {
        &*self.state
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
        super::goal_dispatch::persist_scheduled_run_state(&self.state, goal_id, None, status).await;
    }

    fn task_id(&self) -> Option<&str> {
        self.task_id.as_deref()
    }

    fn llm_runtime(&self) -> &crate::llm_runtime::SharedLlmRuntime {
        &self.llm_runtime
    }

    fn side_calls(&self) -> &crate::llm_runtime::gateway::SideCallGateway {
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
        Agent::stamp_lastgood(self).await;
    }

    fn message_log(&self) -> std::sync::Arc<dyn MessageStore> {
        self.state.clone()
    }

    fn compaction_store(
        &self,
    ) -> std::sync::Arc<dyn aidaemon_runtime::turn::compaction::CompactionStore> {
        self.state.clone()
    }

    fn skills(&self) -> &crate::skills::SkillCache {
        &self.skill_cache
    }

    fn registered_tool_definitions_with_capabilities(
        &self,
    ) -> (
        Vec<serde_json::Value>,
        std::collections::HashMap<String, crate::traits::ToolCapabilities>,
    ) {
        Agent::registered_tool_definitions_with_capabilities(self)
    }

    async fn tool_definitions_with_capabilities(
        &self,
        user_message: &str,
    ) -> (
        Vec<serde_json::Value>,
        std::collections::HashMap<String, crate::traits::ToolCapabilities>,
    ) {
        Agent::tool_definitions_with_capabilities(self, user_message).await
    }

    fn filter_tool_definitions_for_policy(
        &self,
        defs: &[serde_json::Value],
        capabilities: &std::collections::HashMap<String, crate::traits::ToolCapabilities>,
        policy: &crate::execution_policy::ExecutionPolicy,
        risk_score: f32,
        widen: bool,
    ) -> Vec<serde_json::Value> {
        Agent::filter_tool_definitions_for_policy(
            self,
            defs,
            capabilities,
            policy,
            risk_score,
            widen,
        )
    }

    async fn execute_tool_with_watchdog_outcome(
        &self,
        name: &str,
        arguments: &str,
        ctx: &aidaemon_runtime::engine::tool_exec::ToolExecCtx<'_>,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        Agent::execute_tool_with_watchdog_outcome(self, name, arguments, ctx).await
    }

    async fn cancel_active_finite_work_for_session(&self, session_id: &str) -> Vec<String> {
        Agent::cancel_active_finite_work_for_session(self, session_id).await
    }

    async fn create_child(
        &self,
        spec: aidaemon_runtime::spawn::ChildSpec,
    ) -> std::sync::Arc<dyn aidaemon_runtime::spawn::ChildAgent> {
        self.create_child_agent(spec).await
    }

    fn tools(&self) -> &[std::sync::Arc<dyn crate::traits::Tool>] {
        &self.tools
    }
}

#[async_trait::async_trait]
impl aidaemon_runtime::spawn::ChildAgent for Agent {
    async fn handle_message(
        &self,
        session_id: &str,
        user_text: &str,
        status_tx: Option<tokio::sync::mpsc::Sender<crate::types::StatusUpdate>>,
        user_role: crate::types::UserRole,
        channel_ctx: crate::types::ChannelContext,
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
