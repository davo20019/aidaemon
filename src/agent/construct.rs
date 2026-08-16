//! `Agent` constructors extracted from `agent/mod.rs` (Phase 5 decoupling).
//!
//! Pure relocation — no logic changes. Houses `Agent::new`, the `#[cfg(test)]`
//! `set_test_*` overrides, and the internal `with_depth` sub-agent constructor.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::info;

use crate::config::{
    AudioConfig, IterationLimitConfig, PathAliasConfig, PolicyConfig, SttConfig, VisionConfig,
};
use crate::events::EventStore;
use crate::goal_tokens::GoalTokenRegistry;
use crate::llm_runtime::SharedLlmRuntime;
use crate::mcp::McpRegistry;
use crate::runtime_ports::OutboundRouter;
use crate::skills;
use crate::traits::{AgentRole, StateStore, Tool};

#[cfg(test)]
use super::execution_state::ExecutionBudget;
use super::{init_policy_tunables_once, Agent, AgentLimits, HarnessEvalConfig};

/// Runtime services shared by every agent instance.
///
/// Keeping these infrastructure dependencies together prevents the startup
/// layer from having to know the internal shape of the agent's construction
/// contract. Agent policy and per-instance configuration remain separate data.
pub(crate) struct AgentRuntimeDependencies {
    pub llm_runtime: SharedLlmRuntime,
    pub state: Arc<dyn StateStore>,
    pub event_store: Arc<EventStore>,
    pub tools: Vec<Arc<dyn Tool>>,
}

/// Named root-agent construction contract. Keeping startup inputs as data
/// prevents positional argument drift as dependencies evolve.
pub(crate) struct AgentConstruction {
    pub dependencies: AgentRuntimeDependencies,
    pub model: String,
    pub system_prompt: String,
    pub config_path: PathBuf,
    pub skills_dir: PathBuf,
    pub max_depth: usize,
    pub max_iterations: usize,
    pub max_iterations_cap: usize,
    pub max_response_chars: usize,
    pub timeout_secs: u64,
    pub max_facts: usize,
    pub daily_token_budget: Option<u64>,
    pub iteration_config: IterationLimitConfig,
    pub task_timeout_secs: Option<u64>,
    pub task_token_budget: Option<u64>,
    pub llm_call_timeout_secs: Option<u64>,
    pub mcp_registry: Option<McpRegistry>,
    pub goal_token_registry: Option<GoalTokenRegistry>,
    pub hub: Option<Weak<dyn OutboundRouter>>,
    pub record_decision_points: bool,
    pub context_window_config: crate::config::ContextWindowConfig,
    pub policy_config: PolicyConfig,
    pub path_aliases: PathAliasConfig,
    pub inherited_project_scope: Option<String>,
    pub specialists: Arc<crate::agent::specialists::SpecialistRegistry>,
    pub interactive_slot: Option<u32>,
    pub vision_config: VisionConfig,
    pub audio_config: AudioConfig,
    pub stt_config: SttConfig,
    pub harness_eval_config: HarnessEvalConfig,
}

// impl-Agent justification: constructor, with_depth, and test setters — the only place Agent fields are wired.
impl Agent {
    pub(crate) fn new(input: AgentConstruction) -> Self {
        let AgentConstruction {
            dependencies:
                AgentRuntimeDependencies {
                    llm_runtime,
                    state,
                    event_store,
                    tools,
                },
            model,
            system_prompt,
            config_path,
            skills_dir,
            max_depth,
            max_iterations,
            max_iterations_cap,
            max_response_chars,
            timeout_secs,
            max_facts,
            daily_token_budget,
            iteration_config,
            task_timeout_secs,
            task_token_budget,
            llm_call_timeout_secs,
            mcp_registry,
            goal_token_registry,
            hub,
            record_decision_points,
            context_window_config,
            policy_config,
            path_aliases,
            inherited_project_scope,
            specialists,
            interactive_slot,
            vision_config,
            audio_config,
            stt_config,
            harness_eval_config,
        } = input;
        init_policy_tunables_once(policy_config.uncertainty_clarify_threshold);
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

        // Log iteration config
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
            llm_runtime,
            state,
            event_store,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skill_cache: skills::SkillCache::new(skills_dir.clone()),
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
                daily_token_budget,
                llm_call_timeout: llm_call_timeout_secs.map(Duration::from_secs),
                task_timeout: task_timeout_secs.map(Duration::from_secs),
                task_token_budget,
            },
            model_override: RwLock::new(false),
            mcp_registry,
            role: AgentRole::Orchestrator,
            task_id: None,
            goal_id: None,
            mandate_execution: None,
            cancel_token: None,
            goal_token_registry,
            hub: RwLock::new(hub),
            plan_store: RwLock::new(None),
            checklist_turn_flags: Arc::new(tokio::sync::RwLock::new(HashSet::new())),
            schedule_approved_sessions: Arc::new(tokio::sync::RwLock::new(HashSet::new())),
            billing_failed_models: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            // Seed the in-memory ignore-set from config so a fresh start never
            // forces `tool_choice=required` on a known-bad model. Persisted
            // runtime-learned entries are merged in later via
            // `load_required_tool_choice_ignored()` (async, post-construction).
            required_tool_choice_ignored_models: Arc::new(tokio::sync::RwLock::new(
                policy_config
                    .required_tool_choice_ignored_models
                    .iter()
                    .cloned()
                    .collect(),
            )),
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
            path_aliases,
            inherited_project_scope,
            approval_session_id: None,
            root_tools: None, // Root agent — its own tools ARE the root tools
            record_decision_points,
            current_turn_ids: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            window_keep_from_tracker: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            #[cfg(test)]
            execution_budget_override: None,
            specialists,
            core_prompts: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            turn_renders: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            turn_anchors: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            interactive_slot,
            session_core_profile_ids: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            vision_config,
            audio_config,
            stt_config,
            harness_eval: Arc::new(RwLock::new(None)),
            harness_eval_config,
            correction_contexts: Arc::new(tokio::sync::RwLock::new(
                super::CorrectionContextRegistry::default(),
            )),
        }
    }

    /// Override agent to executor mode (depth=1) for integration tests.
    /// This bypasses orchestrator routing so tests exercise the execution loop directly.
    #[cfg(test)]
    pub fn set_test_executor_mode(&mut self) {
        self.depth = 1;
        self.role = AgentRole::Executor;
    }

    #[cfg(test)]
    pub fn set_test_vision_config(&mut self, vision: VisionConfig) {
        self.vision_config = vision;
    }

    #[cfg(test)]
    pub fn set_test_audio_config(&mut self, audio: AudioConfig) {
        self.audio_config = audio;
    }

    #[cfg(test)]
    pub fn set_test_stt_config(&mut self, stt: SttConfig) {
        self.stt_config = stt;
    }

    #[cfg(test)]
    pub async fn set_test_model(&self, model: impl Into<String>) {
        *self.model.write().await = model.into();
    }

    /// Reset agent to orchestrator mode (depth=0) for integration tests.
    /// Use this when testing depth-0-only code paths (e.g. "Done" synthesis).
    #[cfg(test)]
    pub fn set_test_orchestrator_mode(&mut self) {
        self.depth = 0;
        self.role = AgentRole::Orchestrator;
    }

    #[cfg(test)]
    pub fn set_test_task_lead_mode(&mut self) {
        self.depth = 1;
        self.role = AgentRole::TaskLead;
    }

    #[cfg(test)]
    pub fn set_test_task_token_budget(&mut self, budget: Option<u64>) {
        self.limits.task_token_budget = budget;
    }

    #[cfg(test)]
    pub fn set_test_execution_budget_override(&mut self, budget: Option<ExecutionBudget>) {
        self.execution_budget_override = budget;
    }

    #[cfg(test)]
    pub fn set_test_daily_token_budget(&mut self, budget: Option<u64>) {
        self.limits.daily_token_budget = budget;
    }

    #[cfg(test)]
    pub fn set_test_iteration_config(&mut self, config: IterationLimitConfig) {
        self.limits.iteration_config = config;
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub fn set_test_task_timeout(&mut self, timeout: Option<Duration>) {
        self.limits.task_timeout = timeout;
    }

    #[cfg(test)]
    pub fn set_test_goal_id(&mut self, goal_id: Option<String>) {
        self.goal_id = goal_id;
    }

    #[cfg(test)]
    pub fn set_test_task_id(&mut self, task_id: Option<String>) {
        self.task_id = task_id;
    }

    #[cfg(test)]
    pub(crate) fn push_test_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_test_mandate_execution(
        &mut self,
        mandate_id: &str,
        mandate_version: i64,
        authority: crate::traits::MandateAuthority,
        goal_id: &str,
        root_task_id: &str,
        root_task_attempt_id: &str,
        attempt: &crate::traits::TaskAttempt,
    ) {
        self.task_id = Some(attempt.task_id.clone());
        self.goal_id = Some(goal_id.to_string());
        self.mandate_execution = Some(super::MandateExecutionFence {
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

    /// Create an Agent with explicit depth/max_depth (used internally for sub-agents).
    /// Sub-agents don't auto-route — they use whatever model was selected by the parent.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent) fn with_depth(
        llm_runtime: SharedLlmRuntime,
        state: Arc<dyn StateStore>,
        event_store: Arc<EventStore>,
        tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        config_path: PathBuf,
        skills_dir: PathBuf,
        depth: usize,
        max_depth: usize,
        iteration_config: IterationLimitConfig,
        max_iterations: usize,
        max_iterations_cap: usize,
        max_response_chars: usize,
        timeout_secs: u64,
        max_facts: usize,
        task_timeout: Option<Duration>,
        task_token_budget: Option<u64>,
        llm_call_timeout: Option<Duration>,
        mcp_registry: Option<McpRegistry>,
        role: AgentRole,
        task_id: Option<String>,
        goal_id: Option<String>,
        mandate_execution: Option<super::MandateExecutionFence>,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
        goal_token_registry: Option<GoalTokenRegistry>,
        hub: Option<Weak<dyn OutboundRouter>>,
        schedule_approved_sessions: Arc<tokio::sync::RwLock<HashSet<String>>>,
        billing_failed_models: Arc<tokio::sync::RwLock<HashMap<String, Instant>>>,
        required_tool_choice_ignored_models: Arc<tokio::sync::RwLock<HashSet<String>>>,
        record_decision_points: bool,
        context_window_config: crate::config::ContextWindowConfig,
        policy_config: PolicyConfig,
        path_aliases: PathAliasConfig,
        inherited_project_scope: Option<String>,
        approval_session_id: Option<String>,
        root_tools: Option<Vec<Arc<dyn Tool>>>,
        specialists: Arc<crate::agent::specialists::SpecialistRegistry>,
        vision_config: VisionConfig,
        audio_config: AudioConfig,
        stt_config: SttConfig,
        harness_eval_config: HarnessEvalConfig,
        correction_contexts: Arc<tokio::sync::RwLock<super::CorrectionContextRegistry>>,
    ) -> Self {
        let fallback = llm_runtime
            .router()
            .and_then(|router| router.first_fallback().map(str::to_string))
            .unwrap_or_else(|| model.clone());
        Self {
            llm_runtime,
            state,
            event_store,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skill_cache: skills::SkillCache::new(skills_dir.clone()),
            skills_dir,
            depth,
            limits: AgentLimits {
                max_depth,
                iteration_config,
                max_iterations,
                max_iterations_cap,
                max_response_chars,
                timeout_secs,
                max_facts,
                daily_token_budget: None,
                llm_call_timeout,
                task_timeout,
                task_token_budget,
            },
            model_override: RwLock::new(false),
            mcp_registry,
            role,
            task_id,
            goal_id,
            mandate_execution,
            cancel_token,
            goal_token_registry,
            hub: RwLock::new(hub),
            plan_store: RwLock::new(None),
            checklist_turn_flags: Arc::new(tokio::sync::RwLock::new(HashSet::new())),
            schedule_approved_sessions,
            billing_failed_models,
            required_tool_choice_ignored_models,
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
            path_aliases,
            inherited_project_scope,
            approval_session_id,
            root_tools,
            record_decision_points,
            current_turn_ids: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            window_keep_from_tracker: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            #[cfg(test)]
            execution_budget_override: None,
            specialists,
            core_prompts: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            turn_renders: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            turn_anchors: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            // Sub-agents never pin the interactive slot — only the root agent's
            // main generation loop does. They default to the background slot.
            interactive_slot: None,
            session_core_profile_ids: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            vision_config,
            audio_config,
            stt_config,
            harness_eval: Arc::new(RwLock::new(None)),
            harness_eval_config,
            // Shared with the spawning parent so the remediation hierarchy
            // (task-lead + executors) all see the same registered context.
            correction_contexts,
        }
    }
}
