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

    fn works_for_mandate(&self) -> bool {
        self.mandate_execution.is_some()
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

    fn tools(&self) -> &[std::sync::Arc<dyn crate::traits::Tool>] {
        &self.tools
    }
}
