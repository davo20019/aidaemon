//! The agent's test surface (`testing` feature): what a host's tests set or
//! read on an [`Agent`] beyond its public API. Nothing here is used by a
//! running agent.

use std::sync::Arc;

use aidaemon_tools::OutcomeTool as Tool;

use super::Agent;
use crate::settings::{AudioConfig, IterationLimitConfig, SttConfig, VisionConfig};

/// Every `InheritedAuthority` a child agent was created with (the real
/// spawn path, captured for tests).
pub static CREATED_CHILD_INHERITANCE: std::sync::Mutex<Vec<crate::authority::InheritedAuthority>> =
    std::sync::Mutex::new(Vec::new());

#[doc(hidden)]
impl Agent {
    /// Replace the owner's approvals (tests use a short card timeout).
    pub fn set_owner_approvals(&mut self, approvals: Arc<crate::approvals::OwnerApprovals>) {
        self.owner_approvals = approvals;
    }

    pub fn set_test_transcript_writes(
        &mut self,
        writes: Arc<dyn crate::turn::memory_writer::MemoryWrites>,
    ) {
        self.transcript_writes = writes;
    }

    /// Override agent to executor mode (depth=1) so tests exercise the
    /// execution loop directly.
    pub fn set_test_executor_mode(&mut self) {
        self.depth = 1;
        self.role = aidaemon_events::traits::AgentRole::Executor;
    }

    /// Reset agent to orchestrator mode (depth=0).
    pub fn set_test_orchestrator_mode(&mut self) {
        self.depth = 0;
        self.role = aidaemon_events::traits::AgentRole::Orchestrator;
    }

    pub fn set_test_task_lead_mode(&mut self) {
        self.depth = 1;
        self.role = aidaemon_events::traits::AgentRole::TaskLead;
    }

    pub fn set_test_vision_config(&mut self, vision: VisionConfig) {
        self.runtime.vision_config = vision;
    }

    pub fn set_test_audio_config(&mut self, audio: AudioConfig) {
        self.runtime.audio_config = audio;
    }

    pub fn set_test_stt_config(&mut self, stt: SttConfig) {
        self.runtime.stt_config = stt;
    }

    pub async fn set_test_model(&self, model: impl Into<String>) {
        *self.model.write().await = model.into();
    }

    pub fn set_test_task_token_budget(&mut self, budget: Option<u64>) {
        self.limits.task_token_budget = budget;
    }

    pub fn set_test_execution_budget_override(
        &mut self,
        budget: Option<crate::turn::execution_state::ExecutionBudget>,
    ) {
        self.execution_budget_override = budget;
    }

    pub fn set_test_iteration_config(&mut self, config: IterationLimitConfig) {
        self.limits.iteration_config = config;
    }

    /// Offer and run the tools of `registry`'s MCP servers.
    pub fn set_test_mcp_registry(&mut self, registry: aidaemon_mcp::registry::McpRegistry) {
        self.mcp_registry = Some(registry);
    }

    /// Offer every authorized tool, whatever the policy profile would prune.
    pub fn set_test_tool_filter_enforce(&mut self, enforce: bool) {
        self.runtime.policy_config.tool_filter_enforce = enforce;
    }

    pub fn set_test_inherited_authority(
        &mut self,
        authority: crate::authority::InheritedAuthority,
    ) {
        self.inherited_authority = Some(authority);
    }

    pub fn set_test_approval_session_id(&mut self, session: Option<String>) {
        self.approval_session_id = session;
    }

    pub fn set_test_goal_id(&mut self, goal_id: Option<String>) {
        self.goal_id = goal_id;
    }

    pub fn set_test_task_id(&mut self, task_id: Option<String>) {
        self.task_id = task_id;
    }

    pub fn push_test_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_test_mandate_execution(
        &mut self,
        mandate_id: &str,
        mandate_version: i64,
        authority: aidaemon_work::traits::MandateAuthority,
        goal_id: &str,
        root_task_id: &str,
        root_task_attempt_id: &str,
        attempt: &aidaemon_work::traits::TaskAttempt,
    ) {
        self.task_id = Some(attempt.task_id.clone());
        self.goal_id = Some(goal_id.to_string());
        self.mandate_execution = Some(crate::turn_runtime::MandateExecutionFence {
            mandate_id: mandate_id.to_string(),
            mandate_version,
            authority,
            goal_id: goal_id.to_string(),
            goal_run_id: attempt.goal_run_id.clone(),
            root_task_id: root_task_id.to_string(),
            root_task_attempt_id: root_task_attempt_id.to_string(),
            worker_task_id: attempt.task_id.clone(),
            attempt_id: attempt.id.clone(),
            lease_token: attempt.lease_token.clone(),
        });
    }

    /// Lower the token-pressure watermark without weakening production
    /// defaults. The turn's post-reply compaction reads the transcript
    /// writer's own copy, so the host rebuilds that with
    /// [`set_test_transcript_writes`](Self::set_test_transcript_writes).
    pub fn set_test_context_compaction_tokens(
        &mut self,
        token_threshold: usize,
        recent_tokens: usize,
    ) {
        self.runtime.context_window_config.summarize_token_threshold = token_threshold;
        self.runtime.context_window_config.summary_recent_tokens = recent_tokens;
    }

    /// The shared event log handle.
    pub fn event_store(&self) -> &Arc<aidaemon_events::EventStore> {
        &self.event_store
    }

    /// The daemon state store this agent was built with.
    pub fn state_store(&self) -> &Arc<dyn crate::stores::StateStore> {
        &self.state
    }

    /// The memory stores this agent was built with.
    pub fn memory_stores(&self) -> &Arc<dyn aidaemon_memory::stores::MemoryStores> {
        &self.memory_store
    }

    /// The turn runtime's settings and caches, to change in a test.
    pub fn test_runtime_mut(&mut self) -> &mut crate::turn_runtime::TurnRuntime {
        &mut self.runtime
    }

    /// The agent's tool list, to change in a test.
    pub fn test_tools_mut(&mut self) -> &mut Vec<Arc<dyn Tool>> {
        &mut self.tools
    }

    pub fn set_test_side_calls(&mut self, side_calls: crate::llm::gateway::SideCallGateway) {
        self.side_calls = side_calls;
    }

    pub fn set_test_inherited_project_scope(&mut self, scope: Option<String>) {
        self.inherited_project_scope = scope;
    }

    pub fn set_test_cancel_token(&mut self, token: Option<tokio_util::sync::CancellationToken>) {
        self.cancel_token = token;
    }

    pub fn set_test_llm_call_timeout(&mut self, timeout: Option<std::time::Duration>) {
        self.limits.llm_call_timeout = timeout;
    }

    pub fn set_test_project_roots(&mut self, roots: Vec<String>) {
        self.path_aliases.projects = roots;
    }

    pub fn set_test_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Read skills from `dir` (the skill cache follows).
    pub fn set_test_skills_dir(&mut self, dir: std::path::PathBuf) {
        self.skill_cache = aidaemon_skills::SkillCache::new(dir.clone());
        self.skills_dir = dir;
    }

    pub fn set_test_specialists(
        &mut self,
        specialists: Arc<crate::specialists::SpecialistRegistry>,
    ) {
        self.specialists = specialists;
    }
}
