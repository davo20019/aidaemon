//! The host side of making a child: its own tools, MCP registry and agent.
//! Deciding and running the child is the engine's
//! (`crate::engine::turn_engine::spawn`).

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::warn;

use super::Agent;
use crate::spawn::ChildSpec;

impl Agent {
    /// Build a child of this agent from `spec` (the host side of
    /// [`TurnHost::create_child`](crate::host::TurnHost::create_child)).
    pub(super) async fn create_child_agent(&self, spec: ChildSpec) -> Arc<Agent> {
        let ChildSpec {
            mut tools,
            model,
            system_prompt,
            depth,
            role,
            task_id,
            goal_id,
            mandate_execution,
            cancel_token,
            root_tools,
            add_spawn_tool,
            inherited_project_scope,
            approval_session_id,
            max_iterations_override,
            timeout_secs_override,
            inherited_authority,
            child_progress,
        } = spec;
        let hub = match tokio::time::timeout(Duration::from_secs(2), self.hub.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring hub lock while spawning child agent");
                None
            }
        };

        let spawn_tool = add_spawn_tool.then(|| {
            Arc::new(crate::spawn_tool::SpawnAgentTool::for_child(
                self.limits.max_response_chars,
                self.limits.timeout_secs,
                self.state.clone(),
                hub.clone(),
            ))
        });
        if let Some(ref spawn_tool) = spawn_tool {
            tools.push(spawn_tool.clone());
        }
        // A plain child in a tree can hand the coordinator a decision.
        if inherited_authority
            .spawn_family
            .as_ref()
            .is_some_and(|family| family.coordinates())
        {
            tools.push(Arc::new(crate::tools::escalate::EscalateToParentTool));
        }

        let mcp_registry = match mandate_execution.as_ref() {
            Some(_) => self
                .mcp_registry
                .as_ref()
                .map(|registry| registry.scoped_to_mandate()),
            None => self.mcp_registry.clone(),
        };
        let specialists = if mandate_execution.is_some() {
            Arc::new(crate::specialists::SpecialistRegistry::load(None))
        } else {
            self.specialists.clone()
        };
        // Sub-agents don't auto-route: they use the model the parent chose.
        let fallback = self
            .llm_runtime
            .router()
            .and_then(|router| router.first_fallback().map(str::to_string))
            .unwrap_or_else(|| model.clone());

        #[cfg(feature = "testing")]
        super::testing::CREATED_CHILD_INHERITANCE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(inherited_authority.clone());

        let child = Arc::new(Agent {
            runtime: crate::turn_runtime::TurnRuntime::new(
                self.runtime.context_window_config.clone(),
                self.runtime.policy_config.clone(),
                self.runtime.vision_config.clone(),
                self.runtime.audio_config.clone(),
                self.runtime.stt_config.clone(),
                self.runtime.record_decision_points,
                self.runtime.harness_eval_config.clone(),
            ),
            llm_runtime: self.llm_runtime.clone(),
            side_calls: self.side_calls.clone(),
            state: self.state.clone(),
            memory_store: self.memory_store.clone(),
            event_store: self.event_store.clone(),
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path: self.config_path.clone(),
            daemon_paths: self.daemon_paths.clone(),
            execution_backend: self.execution_backend.clone(),
            skill_cache: aidaemon_skills::SkillCache::new(self.skills_dir.clone()),
            skills_dir: self.skills_dir.clone(),
            depth,
            limits: crate::turn_runtime::AgentLimits {
                max_depth: self.limits.max_depth,
                iteration_config: self.limits.iteration_config.clone(),
                max_iterations: max_iterations_override.unwrap_or(self.limits.max_iterations),
                max_iterations_cap: self.limits.max_iterations_cap,
                max_response_chars: self.limits.max_response_chars,
                timeout_secs: timeout_secs_override.unwrap_or(self.limits.timeout_secs),
                max_facts: self.limits.max_facts,
                llm_call_timeout: self.limits.llm_call_timeout,
                task_token_budget: self.limits.task_token_budget,
                spawn_family: Default::default(),
            },
            model_override: RwLock::new(false),
            mcp_registry,
            role,
            task_id,
            goal_id,
            mandate_execution,
            cancel_token,
            child_progress,
            running_children: self.running_children.clone(),
            sdk_engine: self.sdk_engine.clone(),
            run_store: self.run_store.clone(),
            owner_approvals: self.owner_approvals.clone(),
            memory: self.memory.clone(),
            transcript_writes: self.transcript_writes.clone(),
            file_policy: self.file_policy.clone(),
            trace: self.trace.clone(),
            computer_use: self.computer_use.clone(),
            goal_token_registry: self.goal_token_registry.clone(),
            hub: RwLock::new(hub),
            plan_store: RwLock::new(None),
            schedule_approved_sessions: self.schedule_approved_sessions.clone(),
            billing_failed_models: self.billing_failed_models.clone(),
            self_ref: RwLock::new(None),
            task_registry: std::sync::OnceLock::new(),
            path_aliases: self.path_aliases.clone(),
            inbox_dir: self.inbox_dir.clone(),
            inherited_project_scope,
            approval_session_id,
            inherited_authority: Some(inherited_authority),
            root_tools,
            read_untrusted: Arc::default(),
            execution_budget_override: None,
            specialists,
            // Sub-agents never pin the interactive slot — only the root agent's
            // main generation loop does. They default to the background slot.
            interactive_slot: None,
        });

        if let Some(spawn_tool) = spawn_tool {
            let runtime: Arc<dyn crate::spawn::ChildAgentRuntime> = child.clone();
            spawn_tool.set_agent(Arc::downgrade(&runtime));
        }

        child
    }
}
