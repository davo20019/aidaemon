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

use crate::channels::ChannelHub;
use crate::config::{IterationLimitConfig, PathAliasConfig, PolicyConfig};
use crate::events::EventStore;
use crate::goal_tokens::GoalTokenRegistry;
use crate::llm_runtime::SharedLlmRuntime;
use crate::mcp::McpRegistry;
use crate::skills;
use crate::tools::VerificationTracker;
use crate::traits::{AgentRole, StateStore, Tool};

#[cfg(test)]
use super::execution_state::ExecutionBudget;
use super::{init_policy_tunables_once, Agent, AgentLimits};

// impl-Agent justification: constructor, with_depth, and test setters — the only place Agent fields are wired.
impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm_runtime: SharedLlmRuntime,
        state: Arc<dyn StateStore>,
        event_store: Arc<EventStore>,
        tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        config_path: PathBuf,
        skills_dir: PathBuf,
        max_depth: usize,
        max_iterations: usize,
        max_iterations_cap: usize,
        max_response_chars: usize,
        timeout_secs: u64,
        max_facts: usize,
        daily_token_budget: Option<u64>,
        iteration_config: IterationLimitConfig,
        task_timeout_secs: Option<u64>,
        task_token_budget: Option<u64>,
        llm_call_timeout_secs: Option<u64>,
        mcp_registry: Option<McpRegistry>,
        goal_token_registry: Option<GoalTokenRegistry>,
        hub: Option<Weak<ChannelHub>>,
        record_decision_points: bool,
        context_window_config: crate::config::ContextWindowConfig,
        policy_config: PolicyConfig,
        path_aliases: PathAliasConfig,
        inherited_project_scope: Option<String>,
        specialists: Arc<crate::agent::specialists::SpecialistRegistry>,
    ) -> Self {
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
            verification_tracker: Some(Arc::new(VerificationTracker::new())),
            mcp_registry,
            role: AgentRole::Orchestrator,
            task_id: None,
            goal_id: None,
            cancel_token: None,
            goal_token_registry,
            hub: RwLock::new(hub),
            schedule_approved_sessions: Arc::new(tokio::sync::RwLock::new(HashSet::new())),
            billing_failed_models: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
            path_aliases,
            inherited_project_scope,
            root_tools: None, // Root agent — its own tools ARE the root tools
            record_decision_points,
            current_turn_ids: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            #[cfg(test)]
            execution_budget_override: None,
            specialists,
        }
    }

    /// Override agent to executor mode (depth=1) for integration tests.
    /// This bypasses orchestrator routing so tests exercise the execution loop directly.
    #[cfg(test)]
    pub fn set_test_executor_mode(&mut self) {
        self.depth = 1;
        self.role = AgentRole::Executor;
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
    pub async fn set_test_schedule_approval_for_session(&self, session_id: &str, approved: bool) {
        let mut sessions = self.schedule_approved_sessions.write().await;
        if approved {
            sessions.insert(session_id.to_string());
        } else {
            sessions.remove(session_id);
        }
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
        verification_tracker: Option<Arc<VerificationTracker>>,
        role: AgentRole,
        task_id: Option<String>,
        goal_id: Option<String>,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
        goal_token_registry: Option<GoalTokenRegistry>,
        hub: Option<Weak<ChannelHub>>,
        schedule_approved_sessions: Arc<tokio::sync::RwLock<HashSet<String>>>,
        billing_failed_models: Arc<tokio::sync::RwLock<HashMap<String, Instant>>>,
        record_decision_points: bool,
        context_window_config: crate::config::ContextWindowConfig,
        policy_config: PolicyConfig,
        path_aliases: PathAliasConfig,
        inherited_project_scope: Option<String>,
        root_tools: Option<Vec<Arc<dyn Tool>>>,
        specialists: Arc<crate::agent::specialists::SpecialistRegistry>,
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
            verification_tracker,
            mcp_registry,
            role,
            task_id,
            goal_id,
            cancel_token,
            goal_token_registry,
            hub: RwLock::new(hub),
            schedule_approved_sessions,
            billing_failed_models,
            self_ref: RwLock::new(None),
            context_window_config,
            policy_config,
            path_aliases,
            inherited_project_scope,
            root_tools,
            record_decision_points,
            current_turn_ids: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            #[cfg(test)]
            execution_budget_override: None,
            specialists,
        }
    }
}
