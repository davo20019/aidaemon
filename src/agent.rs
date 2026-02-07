use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde_json::{json, Value};
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

use crate::config::{IterationLimitConfig, ModelsConfig};
use crate::types::{ChannelContext, ChannelVisibility, UserRole};
use crate::events::{
    EventStore, EventType, TaskStatus,
    TaskStartData, TaskEndData, ThinkingStartData, ToolCallData, ToolResultData,
    UserMessageData, AssistantResponseData, ToolCallInfo, ErrorData,
    SubAgentSpawnData, SubAgentCompleteData,
};
use crate::plans::{PlanStore, StepTracker};
use crate::providers::{ProviderError, ProviderErrorKind};
use crate::router::{self, Router};
use crate::skills::{self, MemoryContext, SharedSkillRegistry};
use crate::tools::VerificationTracker;
use crate::traits::{Message, ModelProvider, StateStore, Tool, ToolCall};
// Re-export StatusUpdate from types for backwards compatibility
pub use crate::types::StatusUpdate;

/// Constants for stall and repetitive behavior detection
const MAX_STALL_ITERATIONS: usize = 3;
const MAX_REPETITIVE_CALLS: usize = 4;
const RECENT_CALLS_WINDOW: usize = 5;
const PROGRESS_SUMMARY_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes

/// Context accumulated during handle_message for post-task learning.
struct LearningContext {
    user_text: String,
    tool_calls: Vec<String>,           // "tool_name(summary)"
    errors: Vec<(String, bool)>,       // (error_text, was_recovered)
    first_error: Option<String>,
    recovery_actions: Vec<String>,
    #[allow(dead_code)] // Reserved for duration-based learning
    start_time: chrono::DateTime<Utc>,
    completed_naturally: bool,
}

/// Best-effort send — never blocks the agent loop if the receiver is slow/full.
pub fn send_status(tx: &Option<mpsc::Sender<StatusUpdate>>, update: StatusUpdate) {
    if let Some(ref tx) = tx {
        let _ = tx.try_send(update);
    }
}

/// Extract a brief human-readable summary from tool arguments JSON.
fn summarize_tool_args(name: &str, arguments: &str) -> String {
    let val: Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };

    match name {
        "terminal" => val
            .get("command")
            .and_then(|v| v.as_str())
            .map(|cmd| {
                let truncated: String = cmd.chars().take(60).collect();
                if cmd.len() > 60 {
                    format!("`{}...`", truncated)
                } else {
                    format!("`{}`", truncated)
                }
            })
            .unwrap_or_default(),
        "browser" => {
            let action = val.get("action").and_then(|v| v.as_str()).unwrap_or("");
            let url = val.get("url").and_then(|v| v.as_str()).unwrap_or("");
            if !url.is_empty() {
                format!("{} {}", action, url)
            } else {
                action.to_string()
            }
        }
        "spawn_agent" => val
            .get("mission")
            .and_then(|v| v.as_str())
            .map(|m| {
                let truncated: String = m.chars().take(50).collect();
                if m.len() > 50 {
                    format!("{}...", truncated)
                } else {
                    truncated
                }
            })
            .unwrap_or_default(),
        "remember_fact" => val
            .get("fact")
            .and_then(|v| v.as_str())
            .map(|f| {
                let truncated: String = f.chars().take(40).collect();
                if f.len() > 40 {
                    format!("{}...", truncated)
                } else {
                    truncated
                }
            })
            .unwrap_or_else(|| "saving to memory".to_string()),
        _ => String::new(),
    }
}

pub struct Agent {
    provider: Arc<dyn ModelProvider>,
    state: Arc<dyn StateStore>,
    event_store: Arc<EventStore>,
    plan_store: Option<Arc<PlanStore>>,
    step_tracker: Option<Arc<StepTracker>>,
    tools: Vec<Arc<dyn Tool>>,
    model: RwLock<String>,
    fallback_model: RwLock<String>,
    system_prompt: String,
    config_path: PathBuf,
    skills: SharedSkillRegistry,
    /// Current recursion depth (0 = root agent).
    depth: usize,
    /// Maximum allowed recursion depth for sub-agent spawning.
    max_depth: usize,
    /// Iteration limit configuration (unlimited, soft, or hard limits).
    iteration_config: IterationLimitConfig,
    /// Legacy: Maximum agentic loop iterations per invocation (for backward compat).
    #[allow(dead_code)]
    max_iterations: usize,
    /// Legacy: Hard cap on iterations (for backward compat).
    #[allow(dead_code)]
    max_iterations_cap: usize,
    /// Max chars for sub-agent response truncation.
    max_response_chars: usize,
    /// Timeout in seconds for sub-agent execution.
    timeout_secs: u64,
    /// Maximum number of facts to inject into the system prompt.
    max_facts: usize,
    /// Smart router for automatic model tier selection. None for sub-agents
    /// or when all tiers resolve to the same model.
    router: Option<Router>,
    /// When true, the user has manually set a model via /model — skip auto-routing.
    model_override: RwLock<bool>,
    /// Optional daily token budget — rejects LLM calls when exceeded.
    daily_token_budget: Option<u64>,
    /// Optional task timeout - maximum time per task.
    task_timeout: Option<Duration>,
    /// Optional token budget per task.
    task_token_budget: Option<u64>,
    /// Path verification tracker — gates file-modifying commands on unverified paths.
    /// None for sub-agents (they inherit parent context).
    verification_tracker: Option<Arc<VerificationTracker>>,
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        state: Arc<dyn StateStore>,
        event_store: Arc<EventStore>,
        plan_store: Arc<PlanStore>,
        step_tracker: Arc<StepTracker>,
        tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        config_path: PathBuf,
        skills: SharedSkillRegistry,
        max_depth: usize,
        max_iterations: usize,
        max_iterations_cap: usize,
        max_response_chars: usize,
        timeout_secs: u64,
        models_config: ModelsConfig,
        max_facts: usize,
        daily_token_budget: Option<u64>,
        iteration_config: IterationLimitConfig,
        task_timeout_secs: Option<u64>,
        task_token_budget: Option<u64>,
    ) -> Self {
        let fallback = model.clone();
        let router = Router::new(models_config);
        let router = if router.is_uniform() {
            info!("All model tiers identical, auto-routing disabled");
            None
        } else {
            info!(
                fast = router.select(crate::router::Tier::Fast),
                primary = router.select(crate::router::Tier::Primary),
                smart = router.select(crate::router::Tier::Smart),
                "Smart router enabled"
            );
            Some(router)
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

        Self {
            provider,
            state,
            event_store,
            plan_store: Some(plan_store),
            step_tracker: Some(step_tracker),
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skills,
            depth: 0,
            max_depth,
            iteration_config,
            max_iterations,
            max_iterations_cap,
            max_response_chars,
            timeout_secs,
            max_facts,
            router,
            model_override: RwLock::new(false),
            daily_token_budget,
            task_timeout: task_timeout_secs.map(Duration::from_secs),
            task_token_budget,
            verification_tracker: Some(Arc::new(VerificationTracker::new())),
        }
    }

    /// Create an Agent with explicit depth/max_depth (used internally for sub-agents).
    /// Sub-agents don't auto-route — they use whatever model was selected by the parent.
    #[allow(clippy::too_many_arguments)]
    fn with_depth(
        provider: Arc<dyn ModelProvider>,
        state: Arc<dyn StateStore>,
        event_store: Arc<EventStore>,
        tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        config_path: PathBuf,
        skills: SharedSkillRegistry,
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
    ) -> Self {
        let fallback = model.clone();
        Self {
            provider,
            state,
            event_store,
            plan_store: None, // Sub-agents don't manage plans
            step_tracker: None, // Sub-agents don't manage plans
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skills,
            depth,
            max_depth,
            iteration_config,
            max_iterations,
            max_iterations_cap,
            max_response_chars,
            timeout_secs,
            max_facts,
            router: None,
            model_override: RwLock::new(false),
            daily_token_budget: None,
            task_timeout,
            task_token_budget,
            verification_tracker: None, // Sub-agents skip path verification
        }
    }

    /// Current recursion depth of this agent.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Maximum recursion depth allowed.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Maximum agentic loop iterations per invocation.
    #[allow(dead_code)]
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Spawn a child agent with an incremented depth and a focused mission.
    ///
    /// The child runs its own agentic loop in a fresh session and returns the
    /// final text response. It inherits the parent's provider, state, model,
    /// and non-spawn tools. If the child hasn't reached max_depth it also gets
    /// its own `spawn_agent` tool so it can recurse further.
    pub async fn spawn_child(
        self: &Arc<Self>,
        mission: &str,
        task: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_ctx: ChannelContext,
    ) -> anyhow::Result<String> {
        if self.depth >= self.max_depth {
            anyhow::bail!(
                "Cannot spawn sub-agent: max recursion depth ({}) reached",
                self.max_depth
            );
        }

        let child_depth = self.depth + 1;
        let model = self.model.read().await.clone();

        // Collect parent's non-spawn tools for the child.
        let base_tools: Vec<Arc<dyn Tool>> = self
            .tools
            .iter()
            .filter(|t| t.name() != "spawn_agent")
            .cloned()
            .collect();

        // Build the child's system prompt with the mission context.
        let at_max_depth = child_depth >= self.max_depth;
        let depth_note = if at_max_depth {
            "\nYou are at the maximum sub-agent depth. You CANNOT spawn further sub-agents; \
            the `spawn_agent` tool is not available to you. Complete the task directly."
        } else {
            ""
        };
        let child_system_prompt = format!(
            "{}\n\n## Sub-Agent Context\n\
            You are a sub-agent (depth {}/{}) spawned to accomplish a specific mission.\n\
            **Mission:** {}\n\n\
            Focus exclusively on this mission. Be concise. Return your findings/results \
            directly — they will be consumed by the parent agent.{}",
            self.system_prompt, child_depth, self.max_depth, mission, depth_note
        );

        let child_session = format!("sub-{}-{}", child_depth, Uuid::new_v4());

        info!(
            parent_depth = self.depth,
            child_depth,
            child_session = %child_session,
            mission,
            "Spawning sub-agent"
        );

        // Emit SubAgentSpawn event
        {
            let emitter = crate::events::EventEmitter::new(
                self.event_store.clone(),
                child_session.clone(),
            );
            let _ = emitter
                .emit(
                    EventType::SubAgentSpawn,
                    SubAgentSpawnData {
                        child_session_id: child_session.clone(),
                        mission: mission.to_string(),
                        task: task.chars().take(500).collect(),
                        depth: child_depth as u32,
                        parent_task_id: None,
                    },
                )
                .await;
        }

        let start = std::time::Instant::now();

        // If the child can still recurse, give it a spawn tool with a deferred
        // agent reference (set after wrapping in Arc).
        let result = if child_depth < self.max_depth {
            let spawn_tool = Arc::new(crate::tools::spawn::SpawnAgentTool::new_deferred(
                self.max_response_chars,
                self.timeout_secs,
            ));

            let mut child_tools = base_tools;
            child_tools.push(spawn_tool.clone());

            let child = Arc::new(Agent::with_depth(
                self.provider.clone(),
                self.state.clone(),
                self.event_store.clone(),
                child_tools,
                model,
                child_system_prompt,
                self.config_path.clone(),
                self.skills.clone(),
                child_depth,
                self.max_depth,
                self.iteration_config.clone(),
                self.max_iterations,
                self.max_iterations_cap,
                self.max_response_chars,
                self.timeout_secs,
                self.max_facts,
                self.task_timeout,
                self.task_token_budget,
            ));

            // Close the loop: give the spawn tool a weak ref to the child.
            spawn_tool.set_agent(Arc::downgrade(&child));

            child
                .handle_message(&child_session, task, status_tx, UserRole::Owner, channel_ctx)
                .await
        } else {
            // At max depth — no spawn tool, no recursion.
            let child = Arc::new(Agent::with_depth(
                self.provider.clone(),
                self.state.clone(),
                self.event_store.clone(),
                base_tools,
                model,
                child_system_prompt,
                self.config_path.clone(),
                self.skills.clone(),
                child_depth,
                self.max_depth,
                self.iteration_config.clone(),
                self.max_iterations,
                self.max_iterations_cap,
                self.max_response_chars,
                self.timeout_secs,
                self.max_facts,
                self.task_timeout,
                self.task_token_budget,
            ));

            child
                .handle_message(&child_session, task, status_tx, UserRole::Owner, channel_ctx)
                .await
        };

        let duration = start.elapsed();

        // Emit SubAgentComplete event
        {
            let emitter = crate::events::EventEmitter::new(
                self.event_store.clone(),
                child_session.clone(),
            );
            let (success, summary) = match &result {
                Ok(response) => (true, response.chars().take(200).collect()),
                Err(e) => (false, format!("{}", e)),
            };
            let _ = emitter
                .emit(
                    EventType::SubAgentComplete,
                    SubAgentCompleteData {
                        child_session_id: child_session,
                        success,
                        result_summary: summary,
                        duration_secs: duration.as_secs(),
                        parent_task_id: None,
                    },
                )
                .await;
        }

        result
    }

    /// Get the current model name.
    pub async fn current_model(&self) -> String {
        self.model.read().await.clone()
    }

    /// Switch the active model at runtime. Keeps the old model as fallback.
    /// Also disables auto-routing until `clear_model_override()` is called.
    pub async fn set_model(&self, model: String) {
        let mut m = self.model.write().await;
        let mut fb = self.fallback_model.write().await;
        info!(old = %*m, new = %model, "Model switched");
        *fb = m.clone();
        *m = model;
        *self.model_override.write().await = true;
    }

    /// Re-enable auto-routing after a manual model override.
    pub async fn clear_model_override(&self) {
        *self.model_override.write().await = false;
        info!("Model override cleared, auto-routing re-enabled");
    }

    /// Clear conversation history for a session, preserving facts.
    pub async fn clear_session(&self, session_id: &str) -> anyhow::Result<()> {
        self.state.clear_session(session_id).await
    }

    /// List available models from the provider.
    pub async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        self.provider.list_models().await
    }

    /// Stamp the current config as "last known good" — called after a
    /// successful LLM response proves the config actually works.
    async fn stamp_lastgood(&self) {
        let lastgood = self.config_path.with_extension("toml.lastgood");
        if let Err(e) = tokio::fs::copy(&self.config_path, &lastgood).await {
            warn!(error = %e, "Failed to stamp lastgood config");
        }
    }

    /// Build the OpenAI-format tool definitions.
    fn tool_definitions(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": t.schema()
                })
            })
            .collect()
    }

    /// Pick a fallback model: stored fallback if different from `failed_model`,
    /// otherwise ask the router for any tier model that differs.
    async fn pick_fallback(&self, failed_model: &str) -> Option<String> {
        let stored = self.fallback_model.read().await.clone();
        if stored != failed_model {
            return Some(stored);
        }
        // Stored fallback is the same model — try the router tiers
        if let Some(ref router) = self.router {
            for tier in &[
                crate::router::Tier::Primary,
                crate::router::Tier::Smart,
                crate::router::Tier::Fast,
            ] {
                let candidate = router.select(*tier).to_string();
                if candidate != failed_model {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Attempt an LLM call with error-classified recovery:
    /// - RateLimit → wait retry_after, retry once, then fallback
    /// - Timeout/Network/ServerError → retry once, then fallback
    /// - NotFound → fallback to previous model
    /// - Auth/Billing → return user-facing error immediately
    async fn call_llm_with_recovery(
        &self,
        model: &str,
        messages: &[Value],
        tool_defs: &[Value],
    ) -> anyhow::Result<crate::traits::ProviderResponse> {
        match self.provider.chat(model, messages, tool_defs).await {
            Ok(resp) => {
                // Config works — stamp as last known good (best-effort, non-blocking)
                self.stamp_lastgood().await;
                Ok(resp)
            }
            Err(e) => {
                // Try to downcast to our classified ProviderError
                let provider_err = match e.downcast::<ProviderError>() {
                    Ok(pe) => pe,
                    Err(other) => return Err(other), // not a provider error, propagate
                };

                warn!(
                    kind = ?provider_err.kind,
                    status = ?provider_err.status,
                    "LLM call failed: {}",
                    provider_err
                );

                match provider_err.kind {
                    // --- Non-retryable: tell the user, stop ---
                    ProviderErrorKind::Auth | ProviderErrorKind::Billing => {
                        Err(anyhow::anyhow!("{}", provider_err.user_message()))
                    }

                    // --- Rate limit: wait, retry once, then fallback ---
                    ProviderErrorKind::RateLimit => {
                        let wait = provider_err.retry_after_secs.unwrap_or(5);
                        let wait = wait.min(60); // cap at 60s
                        info!(wait_secs = wait, "Rate limited, waiting before retry");
                        tokio::time::sleep(Duration::from_secs(wait)).await;
                        match self.provider.chat(model, messages, tool_defs).await {
                            Ok(resp) => {
                                self.stamp_lastgood().await;
                                Ok(resp)
                            }
                            Err(_retry_err) => {
                                // Still rate-limited — try a fallback model
                                if let Some(fallback) = self.pick_fallback(model).await {
                                    warn!(fallback = %fallback, "Rate limit retry failed, trying fallback model");
                                    let resp = self
                                        .provider
                                        .chat(&fallback, messages, tool_defs)
                                        .await?;
                                    *self.model.write().await = fallback;
                                    Ok(resp)
                                } else {
                                    Err(anyhow::anyhow!("{}", provider_err.user_message()))
                                }
                            }
                        }
                    }

                    // --- Timeout / Network / Server: retry once ---
                    ProviderErrorKind::Timeout
                    | ProviderErrorKind::Network
                    | ProviderErrorKind::ServerError => {
                        info!("Retrying after transient error");
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        match self.provider.chat(model, messages, tool_defs).await {
                            Ok(resp) => {
                                self.stamp_lastgood().await;
                                Ok(resp)
                            }
                            Err(_retry_err) => {
                                // Second failure — try a fallback model
                                if let Some(fallback) = self.pick_fallback(model).await {
                                    warn!(fallback = %fallback, "Retry failed, trying fallback model");
                                    let resp = self
                                        .provider
                                        .chat(&fallback, messages, tool_defs)
                                        .await?;
                                    *self.model.write().await = fallback;
                                    Ok(resp)
                                } else {
                                    Err(anyhow::anyhow!("{}", provider_err.user_message()))
                                }
                            }
                        }
                    }

                    // --- NotFound (bad model name): fallback immediately ---
                    ProviderErrorKind::NotFound => {
                        if let Some(fallback) = self.pick_fallback(model).await {
                            warn!(
                                bad_model = model,
                                fallback = %fallback,
                                "Model not found, reverting to fallback"
                            );
                            *self.model.write().await = fallback.clone();
                            let resp = self
                                .provider
                                .chat(&fallback, messages, tool_defs)
                                .await?;
                            self.stamp_lastgood().await;
                            Ok(resp)
                        } else {
                            Err(anyhow::anyhow!("{}", provider_err.user_message()))
                        }
                    }

                    // --- Unknown: propagate ---
                    ProviderErrorKind::Unknown => {
                        Err(anyhow::anyhow!("{}", provider_err.user_message()))
                    }
                }
            }
        }
    }

    /// Run the agentic loop for a user message in the given session.
    /// Returns the final assistant text response.
    pub async fn handle_message(
        &self,
        session_id: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
    ) -> anyhow::Result<String> {
        // Generate task ID for this request
        let task_id = Uuid::new_v4().to_string();

        // Create event emitter for this session/task
        let emitter = crate::events::EventEmitter::new(
            self.event_store.clone(),
            session_id.to_string(),
        ).with_task_id(task_id.clone());

        // Emit TaskStart event
        let _ = emitter.emit(
            EventType::TaskStart,
            TaskStartData {
                task_id: task_id.clone(),
                description: user_text.chars().take(200).collect(),
                parent_task_id: None,
                user_message: Some(user_text.to_string()),
            },
        ).await;

        // Link any existing incomplete plan to this task
        if let Some(ref ps) = self.plan_store {
            if let Ok(Some(plan)) = ps.get_incomplete_for_session(session_id).await {
                let _ = ps.set_task_id(&plan.id, &task_id).await;
            }
        }

        // 1. Persist the user message
        let user_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "user".to_string(),
            content: Some(user_text.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5, // Will be updated by score_message below
            embedding: None,
        };
        // Calculate heuristic score immediately
        let score = crate::memory::scoring::score_message(&user_msg);
        let mut user_msg = user_msg;
        user_msg.importance = score;

        self.state.append_message(&user_msg).await?;

        // Emit UserMessage event
        let _ = emitter.emit(
            EventType::UserMessage,
            UserMessageData {
                content: user_text.to_string(),
                message_id: None,
                has_attachments: false,
            },
        ).await;

        // Detect stop/cancel commands and automatically cancel running cli_agents
        let lower = user_text.to_lowercase();
        let is_stop_command = lower == "stop" || lower == "cancel" || lower == "abort"
            || lower.starts_with("stop ") || lower.starts_with("cancel ");
        if is_stop_command {
            // Cancel all running cli_agents for this session
            let cancel_result = self.execute_tool(
                "cli_agent",
                &format!(r#"{{"action": "cancel_all"}}"#),
                session_id,
                status_tx.clone(),
                channel_ctx.visibility,
            ).await;
            if let Ok(msg) = cancel_result {
                if !msg.contains("No running CLI agents") {
                    info!(session_id, "Auto-cancelled cli_agents on stop command: {}", msg);
                }
            }
        }

        // Initialize learning context for post-task learning
        let mut learning_ctx = LearningContext {
            user_text: user_text.to_string(),
            tool_calls: Vec::new(),
            errors: Vec::new(),
            first_error: None,
            recovery_actions: Vec::new(),
            start_time: Utc::now(),
            completed_naturally: false,
        };

        let tool_defs = if user_role == UserRole::Public {
            vec![]
        } else {
            self.tool_definitions()
        };
        let model = {
            let is_override = *self.model_override.read().await;
            if !is_override {
                if let Some(ref router) = self.router {
                    let result = router::classify_query(user_text);
                    let routed_model = router.select(result.tier).to_string();
                    info!(
                        tier = %result.tier,
                        reason = %result.reason,
                        model = %routed_model,
                        "Smart router selected model"
                    );
                    routed_model
                } else {
                    self.model.read().await.clone()
                }
            } else {
                self.model.read().await.clone()
            }
        };

        // 2. Build system prompt ONCE before the loop: match skills + inject facts + memory
        let skills_snapshot = self.skills.snapshot().await;
        let mut active_skills = skills::match_skills(&skills_snapshot, user_text);
        if !active_skills.is_empty() {
            let names: Vec<&str> = active_skills.iter().map(|s| s.name.as_str()).collect();
            info!(session_id, skills = ?names, "Matched skills for message");

            // LLM confirmation: only when a distinct fast model is available via the router
            if let Some(ref router) = self.router {
                let fast_model = router.select(router::Tier::Fast);
                match skills::confirm_skills(&*self.provider, fast_model, active_skills.clone(), user_text).await {
                    Ok(confirmed) => {
                        let confirmed_names: Vec<&str> = confirmed.iter().map(|s| s.name.as_str()).collect();
                        info!(session_id, confirmed = ?confirmed_names, "LLM-confirmed skills");
                        active_skills = confirmed;
                    }
                    Err(e) => {
                        warn!("Skill confirmation failed, using keyword matches: {}", e);
                    }
                }
            }
        }

        // Fetch memory components — skip personal memory in public/group channels
        let inject_personal = channel_ctx.should_inject_personal_memory();
        let facts = if inject_personal {
            self.state.get_relevant_facts(user_text, self.max_facts).await?
        } else {
            vec![]
        };
        let episodes = if inject_personal {
            self.state.get_relevant_episodes(user_text, 3).await.unwrap_or_default()
        } else {
            vec![]
        };
        let goals = if inject_personal {
            self.state.get_active_goals().await.unwrap_or_default()
        } else {
            vec![]
        };
        let patterns = if inject_personal {
            self.state.get_behavior_patterns(0.5).await.unwrap_or_default()
        } else {
            vec![]
        };
        // Procedures, error solutions, and expertise are operational — always load
        let procedures = self.state.get_relevant_procedures(user_text, 5).await.unwrap_or_default();
        let error_solutions = self.state.get_relevant_error_solutions(user_text, 5).await.unwrap_or_default();
        let expertise = self.state.get_all_expertise().await.unwrap_or_default();
        let profile = if inject_personal {
            self.state.get_user_profile().await.ok().flatten()
        } else {
            None
        };

        // Get trusted command patterns for AI context (skip in public channels)
        let trusted_patterns = if inject_personal {
            self.state.get_trusted_command_patterns().await.unwrap_or_default()
        } else {
            vec![]
        };

        // Build extended system prompt with all memory components
        let memory_context = MemoryContext {
            facts: &facts,
            episodes: &episodes,
            goals: &goals,
            patterns: &patterns,
            procedures: &procedures,
            error_solutions: &error_solutions,
            expertise: &expertise,
            profile: profile.as_ref(),
            trusted_command_patterns: &trusted_patterns,
        };

        // Generate proactive suggestions if user likes them
        let suggestions = if profile.as_ref().is_some_and(|p| p.likes_suggestions) {
            let engine = crate::memory::proactive::ProactiveEngine::new(
                patterns.clone(),
                goals.clone(),
                procedures.clone(),
                episodes.clone(),
                profile.clone().unwrap_or_default(),
            );
            let ctx = crate::memory::proactive::SuggestionContext {
                last_action: None,
                current_topic: episodes.first().and_then(|e| e.topics.as_ref()?.first().cloned()),
                session_duration_mins: 0,
                tool_call_count: 0,
                has_errors: false,
                user_message: user_text.to_string(),
            };
            engine.get_suggestions(&ctx)
        } else {
            vec![]
        };

        // Compile session context from recent events (for "what are you doing?" awareness)
        let context_compiler = crate::events::SessionContextCompiler::new(self.event_store.clone());
        let session_context = context_compiler
            .compile(session_id, chrono::Duration::hours(1))
            .await
            .unwrap_or_default();
        let session_context_str = session_context.format_for_prompt();

        let mut system_prompt = skills::build_system_prompt_with_memory(
            &self.system_prompt,
            &skills_snapshot,
            &active_skills,
            &memory_context,
            self.max_facts,
            if suggestions.is_empty() { None } else { Some(&suggestions) },
        );

        // Inject user role context
        system_prompt = format!(
            "{}\n\n[User Role: {}]{}",
            system_prompt,
            user_role,
            match user_role {
                UserRole::Guest => {
                    " The current user is a guest. Be cautious with destructive actions, \
                     sensitive data, and system configuration changes."
                }
                UserRole::Public => {
                    " You have NO tools available. Respond conversationally only. \
                     If the user asks you to perform actions that would require tools \
                     (running commands, reading files, browsing the web, etc.), politely \
                     explain that tool-based actions are not available for public users."
                }
                _ => "",
            }
        );

        // Inject channel context for non-private channels
        match channel_ctx.visibility {
            ChannelVisibility::Public => {
                let ch_label = channel_ctx
                    .channel_name
                    .as_deref()
                    .map(|n| format!(" \"{}\"", n))
                    .unwrap_or_default();
                system_prompt = format!(
                    "{}\n\n[Channel Context: PUBLIC {} channel{}]\n\
                     You are responding in a public channel visible to many people. Rules:\n\
                     - Do NOT reference private information, personal facts, goals, or memory from DMs.\n\
                     - If the user asks about something that requires private context, suggest continuing in a DM.\n\
                     - Be professional and concise. Assume others are reading.\n\
                     - Do not mention the user's personal projects, habits, or preferences.",
                    system_prompt, channel_ctx.platform, ch_label
                );
            }
            ChannelVisibility::PrivateGroup => {
                let ch_label = channel_ctx
                    .channel_name
                    .as_deref()
                    .map(|n| format!(" \"{}\"", n))
                    .unwrap_or_default();
                system_prompt = format!(
                    "{}\n\n[Channel Context: PRIVATE GROUP on {}{}]\n\
                     You are in a private group chat. Be cautious about sharing highly \
                     sensitive personal information. If asked about something very private, \
                     suggest continuing in a direct message.",
                    system_prompt, channel_ctx.platform, ch_label
                );
            }
            // Private and Internal: no additional injection (current behavior)
            _ => {}
        }

        // Inject session context if present
        if !session_context_str.is_empty() {
            system_prompt = format!("{}\n\n{}", system_prompt, session_context_str);
        }

        // Check for incomplete task plans and inject context
        let incomplete_plan = if let Some(ref ps) = self.plan_store {
            ps.get_incomplete_for_session(session_id).await.ok().flatten()
        } else {
            None
        };
        let has_incomplete_plan = incomplete_plan.is_some();
        if let Some(ref plan) = incomplete_plan {
            system_prompt = format!("{}\n\n{}", system_prompt, plan.format_for_prompt());
        }

        // If no incomplete plan, check if this task should have one
        if incomplete_plan.is_none() {
            let plan_trigger = crate::plans::should_create_plan(user_text);
            if plan_trigger.is_auto_create() {
                if let Some(ref ps) = self.plan_store {
                    let reason = plan_trigger.reason().unwrap_or("high-stakes operation");
                    info!(session_id, reason, "Auto-creating plan for high-stakes operation");
                    match crate::plans::generate_plan_steps(&*self.provider, &model, user_text).await {
                        Ok(steps) if !steps.is_empty() => {
                            let plan = crate::plans::TaskPlan::new(
                                session_id, user_text,
                                &user_text.chars().take(200).collect::<String>(),
                                steps, format!("auto_create:{}", reason),
                            );
                            if let Err(e) = ps.create(&plan).await {
                                warn!(error = %e, "Failed to auto-create plan");
                            } else {
                                let _ = ps.set_task_id(&plan.id, &task_id).await;
                                system_prompt = format!("{}\n\n{}", system_prompt, plan.format_for_prompt());
                                info!(plan_id = %plan.id, steps = plan.steps.len(), "Auto-created plan");
                            }
                        }
                        Ok(_) => warn!("Auto-create returned empty steps"),
                        Err(e) => warn!(error = %e, "Failed to generate auto-create plan"),
                    }
                }
            } else if let Some(hint) = crate::plans::get_plan_suggestion_prompt(&plan_trigger) {
                system_prompt = format!("{}{}", system_prompt, hint);
            }
        }

        info!(
            session_id,
            facts = facts.len(),
            episodes = episodes.len(),
            goals = goals.len(),
            patterns = patterns.len(),
            procedures = procedures.len(),
            expertise = expertise.len(),
            has_session_context = !session_context_str.is_empty(),
            has_incomplete_plan,
            "Memory context loaded"
        );

        // 2b. Retrieve Context ONCE (Optimization)
        // Use Tri-Hybrid Retrieval to get relevant history
        let mut initial_history = self.state.get_context(session_id, user_text, 50).await?;

        // Fallback: if messages table is empty, try event store for conversation history
        if initial_history.is_empty() {
            info!(session_id, "Messages table empty, falling back to event store history");
            initial_history = self.event_store.get_conversation_history(session_id, 50).await?;
        }

        // Optimize: Identify "Pinned" memories (Relevant/Salient but old) to avoid re-fetching
        let recency_window = 20;
        let recent_ids: std::collections::HashSet<String> = initial_history.iter()
            .rev()
            .take(recency_window)
            .map(|m| m.id.clone())
            .collect();
            
        let pinned_memories: Vec<Message> = initial_history
            .drain(..)
            .filter(|m| !recent_ids.contains(&m.id))
            .collect();

        info!(
            session_id,
            total_context = initial_history.len(),
            pinned_old_memories = pinned_memories.len(),
            depth = self.depth,
            "Context prepared"
        );

        // 3. Agentic loop — runs until natural completion or safety limits
        let task_start = Instant::now();
        let mut last_progress_summary = Instant::now();
        let mut iteration: usize = 0;
        let mut stall_count: usize = 0;
        let mut task_tokens_used: u64 = 0;
        let mut tool_failure_count: HashMap<String, usize> = HashMap::new();
        let mut tool_call_count: HashMap<String, usize> = HashMap::new();
        let mut recent_tool_calls: VecDeque<u64> = VecDeque::with_capacity(RECENT_CALLS_WINDOW);
        let mut soft_limit_warned = false;

        // Determine iteration limit behavior
        let (hard_cap, soft_threshold, soft_warn_at) = match &self.iteration_config {
            IterationLimitConfig::Unlimited => (None, None, None),
            IterationLimitConfig::Soft { threshold, warn_at } => (None, Some(*threshold), Some(*warn_at)),
            IterationLimitConfig::Hard { initial: _, cap } => (Some(*cap), None, None),
        };

        loop {
            iteration += 1;
            info!(iteration, session_id, model = %model, depth = self.depth, "Agent loop iteration");

            // Emit ThinkingStart event
            let _ = emitter.emit(
                EventType::ThinkingStart,
                ThinkingStartData {
                    iteration: iteration as u32,
                    task_id: task_id.clone(),
                    total_tool_calls: learning_ctx.tool_calls.len() as u32,
                },
            ).await;

            // === STOPPING CONDITIONS ===

            // 1. Hard iteration cap (legacy mode)
            if let Some(cap) = hard_cap {
                if iteration > cap {
                    warn!(session_id, iteration, cap, "Hard iteration cap reached");
                    let result = self.graceful_cap_response(session_id, &learning_ctx, iteration).await;
                    let (status, error, summary) = match &result {
                        Ok(reply) => (TaskStatus::Completed, None, Some(reply.chars().take(200).collect())),
                        Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                    };
                    self.emit_task_end(&emitter, &task_id, status, task_start, iteration, learning_ctx.tool_calls.len(), error, summary).await;
                    return result;
                }
            }

            // 2. Task timeout (if configured)
            if let Some(timeout) = self.task_timeout {
                if task_start.elapsed() > timeout {
                    warn!(session_id, elapsed_secs = task_start.elapsed().as_secs(), "Task timeout reached");
                    let result = self.graceful_timeout_response(session_id, &learning_ctx, task_start.elapsed()).await;
                    let (status, error, summary) = match &result {
                        Ok(reply) => (TaskStatus::Completed, None, Some(reply.chars().take(200).collect())),
                        Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                    };
                    self.emit_task_end(&emitter, &task_id, status, task_start, iteration, learning_ctx.tool_calls.len(), error, summary).await;
                    return result;
                }
            }

            // 3. Task token budget (if configured)
            if let Some(budget) = self.task_token_budget {
                if task_tokens_used >= budget {
                    warn!(session_id, tokens_used = task_tokens_used, budget, "Task token budget exhausted");
                    let result = self.graceful_budget_response(session_id, &learning_ctx, task_tokens_used).await;
                    let (status, error, summary) = match &result {
                        Ok(reply) => (TaskStatus::Completed, None, Some(reply.chars().take(200).collect())),
                        Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                    };
                    self.emit_task_end(&emitter, &task_id, status, task_start, iteration, learning_ctx.tool_calls.len(), error, summary).await;
                    return result;
                }
            }

            // 4. Daily token budget (existing global limit)
            if let Some(daily_budget) = self.daily_token_budget {
                let today_start = Utc::now().format("%Y-%m-%d 00:00:00").to_string();
                if let Ok(records) = self.state.get_token_usage_since(&today_start).await {
                    let total: u64 = records.iter().map(|r| (r.input_tokens + r.output_tokens) as u64).sum();
                    if total >= daily_budget {
                        let error_msg = format!(
                            "Daily token budget of {} exceeded (used: {}). Resets at midnight UTC.",
                            daily_budget, total
                        );
                        self.emit_task_end(&emitter, &task_id, TaskStatus::Failed, task_start, iteration, learning_ctx.tool_calls.len(), Some(error_msg.clone()), None).await;
                        return Err(anyhow::anyhow!(error_msg));
                    }
                }
            }

            // 5. Stall detection — agent spinning without progress
            if stall_count >= MAX_STALL_ITERATIONS {
                warn!(session_id, stall_count, "Agent stalled - no progress detected");
                let result = self.graceful_stall_response(session_id, &learning_ctx).await;
                let (status, error, summary) = match &result {
                    Ok(reply) => (TaskStatus::Failed, Some("Agent stalled".to_string()), Some(reply.chars().take(200).collect())),
                    Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                };
                self.emit_task_end(&emitter, &task_id, status, task_start, iteration, learning_ctx.tool_calls.len(), error, summary).await;
                return result;
            }

            // 6. Soft limit warning (warnings only, no forced stop)
            if let (Some(threshold), Some(warn_at)) = (soft_threshold, soft_warn_at) {
                if iteration >= warn_at && !soft_limit_warned {
                    soft_limit_warned = true;
                    send_status(&status_tx, StatusUpdate::IterationWarning {
                        current: iteration,
                        threshold,
                    });
                    info!(session_id, iteration, threshold, "Soft iteration limit warning");
                }
            }

            // 7. Progress summary for long-running tasks (every 5 minutes)
            if last_progress_summary.elapsed() >= PROGRESS_SUMMARY_INTERVAL {
                let elapsed_mins = task_start.elapsed().as_secs() / 60;
                let summary = format!(
                    "Working... {} iterations, {} tool calls, {} mins elapsed",
                    iteration,
                    learning_ctx.tool_calls.len(),
                    elapsed_mins
                );
                send_status(&status_tx, StatusUpdate::ProgressSummary {
                    elapsed_mins,
                    summary,
                });
                last_progress_summary = Instant::now();
            }

            // === BUILD MESSAGES ===

            // Fetch only recent history inside the loop
            let recent_history = self.state.get_history(session_id, 20).await?;

            // Merge Pinned + Recent using iterators to avoid cloning the Message structs
            let mut seen_ids: std::collections::HashSet<&String> = std::collections::HashSet::new();

            // Deduplicated, ordered message list
            let deduped_msgs: Vec<&Message> = pinned_memories.iter()
                .chain(recent_history.iter())
                .filter(|m| seen_ids.insert(&m.id))
                .collect();

            // Collect tool_call_ids that have valid tool responses (role=tool with a name)
            let valid_tool_call_ids: std::collections::HashSet<&str> = deduped_msgs.iter()
                .filter(|m| m.role == "tool" && m.tool_name.as_ref().is_some_and(|n| !n.is_empty()))
                .filter_map(|m| m.tool_call_id.as_deref())
                .collect();

            let mut messages: Vec<Value> = deduped_msgs.iter()
                // Skip tool results with empty/missing tool_name
                .filter(|m| !(m.role == "tool" && m.tool_name.as_ref().is_none_or(|n| n.is_empty())))
                // Skip tool results whose tool_call_id has no matching tool_call in an assistant message
                .filter(|m| {
                    if m.role == "tool" {
                        m.tool_call_id.as_ref().is_some_and(|id| valid_tool_call_ids.contains(id.as_str()))
                    } else {
                        true
                    }
                })
                .filter_map(|m| {
                    let mut obj = json!({
                        "role": m.role,
                        "content": m.content,
                    });
                    // For assistant messages with tool_calls, convert from ToolCall struct format
                    // to OpenAI wire format and strip any that lack a matching tool result
                    if let Some(tc_json) = &m.tool_calls_json {
                        if let Ok(tcs) = serde_json::from_str::<Vec<ToolCall>>(tc_json) {
                            let filtered: Vec<Value> = tcs.iter()
                                .filter(|tc| valid_tool_call_ids.contains(tc.id.as_str()))
                                .map(|tc| {
                                    let mut val = json!({
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.name,
                                            "arguments": tc.arguments
                                        }
                                    });
                                    if let Some(ref extra) = tc.extra_content {
                                        val["extra_content"] = extra.clone();
                                    }
                                    val
                                })
                                .collect();
                            if !filtered.is_empty() {
                                obj["tool_calls"] = json!(filtered);
                                if m.content.is_none() {
                                    obj["content"] = Value::Null;
                                }
                            } else if m.content.is_none() {
                                // Assistant message had tool_calls but all were orphaned,
                                // and no text content — drop it entirely
                                return None;
                            }
                        }
                    }
                    if let Some(name) = &m.tool_name {
                        if !name.is_empty() {
                            obj["name"] = json!(name);
                        }
                    }
                    if let Some(tcid) = &m.tool_call_id {
                        obj["tool_call_id"] = json!(tcid);
                    }
                    Some(obj)
                })
                .collect();

            // Final safety: drop any tool-role messages that still lack a "name" field
            messages.retain(|m| {
                if m.get("role").and_then(|r| r.as_str()) == Some("tool") {
                    let has_name = m.get("name").and_then(|n| n.as_str()).is_some_and(|n| !n.is_empty());
                    if !has_name {
                        warn!("Dropping tool message with missing/empty name: tool_call_id={:?}",
                            m.get("tool_call_id"));
                    }
                    has_name
                } else {
                    true
                }
            });

            // Three-pass fixup: merge → drop orphans → merge again.
            fixup_message_ordering(&mut messages);

            // Ensure the current user message is in the context (fixes race condition with DB)
            // Only on first iteration - subsequent iterations already have the user message
            if iteration == 1 {
                let has_current_user_msg = messages.last()
                    .and_then(|m| m.get("role"))
                    .and_then(|r| r.as_str()) == Some("user")
                    && messages.last()
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str()) == Some(user_text);

                if !has_current_user_msg {
                    // User message might not be in history yet - add it explicitly
                    messages.push(json!({
                        "role": "user",
                        "content": user_text,
                    }));
                }
            }

            messages.insert(0, json!({
                "role": "system",
                "content": system_prompt,
            }));

            // Emit "Thinking" status for iterations after the first
            if iteration > 1 {
                send_status(&status_tx, StatusUpdate::Thinking(iteration));
            }

            // Debug: log message structure and estimated token count
            {
                let summary: Vec<String> = messages.iter().map(|m| {
                    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                    let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let tc_id = m.get("tool_call_id").and_then(|id| id.as_str()).unwrap_or("");
                    let tc_count = m.get("tool_calls").and_then(|v| v.as_array()).map_or(0, |a| a.len());
                    if role == "tool" {
                        format!("tool({},tc_id={})", name, &tc_id[..tc_id.len().min(12)])
                    } else if tc_count > 0 {
                        format!("{}(tc={})", role, tc_count)
                    } else {
                        role.to_string()
                    }
                }).collect();

                // Estimate tokens: ~4 chars per token for English text
                let messages_json = serde_json::to_string(&messages).unwrap_or_default();
                let tools_json = serde_json::to_string(&tool_defs).unwrap_or_default();
                let est_msg_tokens = messages_json.len() / 4;
                let est_tool_tokens = tools_json.len() / 4;
                let est_total_tokens = est_msg_tokens + est_tool_tokens;

                info!(
                    session_id,
                    iteration,
                    est_input_tokens = est_total_tokens,
                    est_msg_tokens,
                    est_tool_tokens,
                    msg_count = messages.len(),
                    msgs = ?summary,
                    "Context before LLM call"
                );
            }

            // === CALL LLM ===

            let resp = self
                .call_llm_with_recovery(&model, &messages, &tool_defs)
                .await?;

            // Record token usage (both for task budget and daily budget)
            if let Some(ref usage) = resp.usage {
                task_tokens_used += (usage.input_tokens + usage.output_tokens) as u64;
                info!(
                    session_id,
                    iteration,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    total_tokens = usage.input_tokens + usage.output_tokens,
                    task_tokens_used,
                    "LLM token usage"
                );
                if let Err(e) = self.state.record_token_usage(session_id, usage).await {
                    warn!(session_id, error = %e, "Failed to record token usage");
                }
            }

            // Log tool call names for debugging
            let tc_names: Vec<&str> = resp.tool_calls.iter().map(|tc| tc.name.as_str()).collect();
            info!(
                session_id,
                has_content = resp.content.is_some(),
                tool_calls = resp.tool_calls.len(),
                tool_names = ?tc_names,
                "LLM response received"
            );

            // === NATURAL COMPLETION: No tool calls ===
            if resp.tool_calls.is_empty() {
                let reply = resp.content.filter(|s| !s.is_empty()).unwrap_or_default();
                if reply.is_empty() {
                    // If the agent actually did work (iteration > 1 means tool calls happened)
                    // and this is the top-level agent (depth 0), send a brief completion note
                    // so the user knows the task finished. Without this, the user gets silence
                    // because the LLM decided the tool output already communicated the answer.
                    if iteration > 1 && self.depth == 0 {
                        let task_hint: String = learning_ctx.user_text.chars().take(80).collect();
                        let task_hint = task_hint.trim();
                        let reply = if task_hint.is_empty() {
                            "Done.".to_string()
                        } else if learning_ctx.user_text.len() > 80 {
                            format!("Done — {}...", task_hint)
                        } else {
                            format!("Done — {}", task_hint)
                        };
                        info!(session_id, iteration, "Agent completed with synthesized completion message");
                        return Ok(reply);
                    }
                    // No work was done (first iteration) or sub-agent — stay silent
                    info!(session_id, iteration, "Agent completed with empty response");
                    return Ok(String::new());
                }

                // Emit AssistantResponse event
                let _ = emitter.emit(
                    EventType::AssistantResponse,
                    AssistantResponseData {
                        content: Some(reply.clone()),
                        model: model.clone(),
                        tool_calls: None,
                        input_tokens: resp.usage.as_ref().map(|u| u.input_tokens as u32),
                        output_tokens: resp.usage.as_ref().map(|u| u.output_tokens as u32),
                    },
                ).await;

                let assistant_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "assistant".to_string(),
                    content: Some(reply.clone()),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.5,
                    embedding: None,
                };
                self.state.append_message(&assistant_msg).await?;

                // Emit TaskEnd event
                self.emit_task_end(&emitter, &task_id, TaskStatus::Completed, task_start, iteration, learning_ctx.tool_calls.len(), None, Some(reply.chars().take(200).collect())).await;

                // Process learning in background
                learning_ctx.completed_naturally = true;
                let state = self.state.clone();
                tokio::spawn(async move {
                    if let Err(e) = process_learning(&state, learning_ctx).await {
                        warn!("Learning failed: {}", e);
                    }
                });

                info!(session_id, iteration, "Agent completed naturally");
                return Ok(reply);
            }

            // === EXECUTE TOOL CALLS ===

            // Persist assistant message with tool calls
            let assistant_msg = Message {
                id: Uuid::new_v4().to_string(),
                session_id: session_id.to_string(),
                role: "assistant".to_string(),
                content: resp.content.clone(),
                tool_call_id: None,
                tool_name: None,
                tool_calls_json: Some(serde_json::to_string(&resp.tool_calls)?),
                created_at: Utc::now(),
                importance: 0.5,
                embedding: None,
            };
            self.state.append_message(&assistant_msg).await?;

            // Emit AssistantResponse event with tool calls
            let tool_call_infos: Vec<ToolCallInfo> = resp.tool_calls.iter().map(|tc| ToolCallInfo {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({})),
            }).collect();
            let _ = emitter.emit(
                EventType::AssistantResponse,
                AssistantResponseData {
                    content: resp.content.clone(),
                    model: model.clone(),
                    tool_calls: Some(tool_call_infos),
                    input_tokens: resp.usage.as_ref().map(|u| u.input_tokens as u32),
                    output_tokens: resp.usage.as_ref().map(|u| u.output_tokens as u32),
                },
            ).await;

            // Intent gate: on first iteration, require narration before tool calls.
            // Forces the agent to "show its work" so the user can catch misunderstandings.
            if iteration == 1
                && self.depth == 0
                && !resp.tool_calls.is_empty()
                && resp.content.as_ref().map_or(true, |c| c.trim().len() < 20)
            {
                info!(session_id, "Intent gate: requiring narration before tool execution");
                for tc in &resp.tool_calls {
                    let result_text = "[SYSTEM] Before executing tools, briefly state what you \
                        understand the user is asking and what you plan to do. \
                        Then re-issue the tool calls."
                        .to_string();
                    let tool_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "tool".to_string(),
                        content: Some(result_text),
                        tool_call_id: Some(tc.id.clone()),
                        tool_name: Some(tc.name.clone()),
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.3,
                        embedding: None,
                    };
                    self.state.append_message(&tool_msg).await?;
                }
                continue; // Skip to next iteration — agent will narrate then retry
            }

            let mut successful_tool_calls = 0;

            for tc in &resp.tool_calls {
                // Check for repetitive behavior (same tool call hash appearing too often)
                let call_hash = hash_tool_call(&tc.name, &tc.arguments);
                recent_tool_calls.push_back(call_hash);
                if recent_tool_calls.len() > RECENT_CALLS_WINDOW {
                    recent_tool_calls.pop_front();
                }

                // Count how many of the recent calls match this one
                let repetitive_count = recent_tool_calls.iter().filter(|&&h| h == call_hash).count();
                if repetitive_count >= MAX_REPETITIVE_CALLS {
                    warn!(
                        session_id,
                        tool = %tc.name,
                        repetitive_count,
                        "Repetitive tool call detected - agent may be stuck"
                    );
                    let result = self.graceful_repetitive_response(session_id, &learning_ctx, &tc.name).await;
                    let (status, error, summary) = match &result {
                        Ok(reply) => (TaskStatus::Failed, Some("Repetitive tool calls".to_string()), Some(reply.chars().take(200).collect())),
                        Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                    };
                    self.emit_task_end(&emitter, &task_id, status, task_start, iteration, learning_ctx.tool_calls.len(), error, summary).await;
                    return result;
                }

                // Check if this tool has been called too many times or failed too often
                let prior_failures = tool_failure_count.get(&tc.name).copied().unwrap_or(0);
                let prior_calls = tool_call_count.get(&tc.name).copied().unwrap_or(0);
                let blocked = if prior_failures >= 3 {
                    Some(format!(
                        "[SYSTEM] Tool '{}' has encountered {} errors. \
                         Do not call it again. Use a different approach or \
                         answer the user with what you have.",
                        tc.name, prior_failures
                    ))
                } else if prior_calls >= 3 && !matches!(tc.name.as_str(), "terminal" | "plan_manager" | "cli_agent" | "remember_fact" | "web_fetch") {
                    if tc.name == "web_search" && prior_failures == 0 {
                        Some(format!(
                            "[SYSTEM] web_search returned no useful results {} times. \
                             The DuckDuckGo backend is likely blocked.\n\n\
                             Tell the user web search is not working and suggest they set up Brave Search:\n\
                             1. Get a free API key at https://brave.com/search/api/ (free tier = 2000 queries/month)\n\
                             2. Paste the API key in this chat\n\n\
                             When the user provides a Brave API key, use manage_config to:\n\
                             - set search.backend to '\"brave\"'\n\
                             - set search.api_key to '\"THEIR_KEY\"'\n\
                             Then tell them to type /reload to apply the changes.",
                            prior_calls
                        ))
                    } else {
                        // terminal is expected to be called many times; others are suspicious
                        Some(format!(
                            "[SYSTEM] You have already called '{}' {} times this turn. \
                             Do not call it again. Use the results you already have to \
                             answer the user's question now.",
                            tc.name, prior_calls
                        ))
                    }
                } else {
                    None
                };
                if let Some(result_text) = blocked {
                    warn!(
                        tool = %tc.name,
                        failures = prior_failures,
                        calls = prior_calls,
                        "Blocking repeated tool call"
                    );
                    let tool_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "tool".to_string(),
                        content: Some(result_text),
                        tool_call_id: Some(tc.id.clone()),
                        tool_name: Some(tc.name.clone()),
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.1,
                        embedding: None,
                    };
                    self.state.append_message(&tool_msg).await?;
                    continue;
                }

                send_status(
                    &status_tx,
                    StatusUpdate::ToolStart {
                        name: tc.name.clone(),
                        summary: summarize_tool_args(&tc.name, &tc.arguments),
                    },
                );

                // Emit ToolCall event
                let _ = emitter.emit(
                    EventType::ToolCall,
                    ToolCallData::from_tool_call(
                        tc.id.clone(),
                        tc.name.clone(),
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({})),
                        Some(task_id.clone()),
                    ),
                ).await;

                // Track tool call in step tracker for plan progress
                if let Some(ref st) = self.step_tracker {
                    let _ = st.record_tool_call(session_id, &tc.name, &tc.id).await;
                }

                let result = self.execute_tool(&tc.name, &tc.arguments, session_id, status_tx.clone(), channel_ctx.visibility).await;
                let mut result_text = match result {
                    Ok(text) => text,
                    Err(e) => format!("Error: {}", e),
                };

                // Track total calls per tool
                *tool_call_count.entry(tc.name.clone()).or_insert(0) += 1;

                // Track tool call for learning
                let tool_summary = format!("{}({})", tc.name, summarize_tool_args(&tc.name, &tc.arguments));
                learning_ctx.tool_calls.push(tool_summary.clone());

                // Track tool failures across iterations (actual errors only)
                let is_error = result_text.starts_with("ERROR:")
                    || result_text.starts_with("Error:")
                    || result_text.starts_with("Failed to ");
                if is_error {
                    let count = tool_failure_count.entry(tc.name.clone()).or_insert(0);
                    *count += 1;

                    // DIAGNOSTIC LOOP: On first failure, query memory for similar errors
                    if *count == 1 {
                        if let Ok(solutions) = self.state.get_relevant_error_solutions(&result_text, 3).await {
                            if !solutions.is_empty() {
                                let diagnostic_hints: Vec<String> = solutions
                                    .iter()
                                    .map(|s| {
                                        if let Some(ref steps) = s.solution_steps {
                                            format!("- {}\n  Steps: {}", s.solution_summary, steps.join(" -> "))
                                        } else {
                                            format!("- {}", s.solution_summary)
                                        }
                                    })
                                    .collect();
                                result_text = format!(
                                    "{}\n\n[DIAGNOSTIC] Similar errors resolved before:\n{}",
                                    result_text,
                                    diagnostic_hints.join("\n")
                                );
                                info!(
                                    tool = %tc.name,
                                    solutions_found = solutions.len(),
                                    "Diagnostic loop: injected error solutions"
                                );
                            }
                        }
                    }

                    if *count >= 2 {
                        result_text = format!(
                            "{}\n\n[SYSTEM] This tool has errored {} times. Do NOT retry it. \
                             Use a different approach or respond with what you have.",
                            result_text, count
                        );
                    }

                    // Track error for learning
                    if learning_ctx.first_error.is_none() {
                        learning_ctx.first_error = Some(result_text.clone());
                    }
                    learning_ctx.errors.push((result_text.clone(), false));
                } else {
                    successful_tool_calls += 1;

                    if !learning_ctx.errors.is_empty() {
                        // Successful action after an error - this is recovery
                        learning_ctx.recovery_actions.push(tool_summary);
                        // Mark the last error as recovered
                        if let Some((_, recovered)) = learning_ctx.errors.last_mut() {
                            *recovered = true;
                        }
                    }
                }

                // Emit ToolResult event
                let _ = emitter.emit(
                    EventType::ToolResult,
                    ToolResultData {
                        tool_call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        result: result_text.clone(),
                        success: !is_error,
                        duration_ms: 0, // Could add timing if needed
                        error: if is_error { Some(result_text.clone()) } else { None },
                        task_id: Some(task_id.clone()),
                    },
                ).await;

                // Emit Error event if tool failed
                if is_error {
                    let _ = emitter.emit(
                        EventType::Error,
                        ErrorData::tool_error(
                            tc.name.clone(),
                            result_text.clone(),
                            Some(task_id.clone()),
                        ),
                    ).await;
                }

                // Track tool result in step tracker for plan progress
                if tc.name != "plan_manager" {
                    if let Some(ref st) = self.step_tracker {
                        let summary = result_text.chars().take(200).collect::<String>();
                        if let Ok(Some((_, step_completed))) = st.on_tool_result(session_id, &tc.name, !is_error, &summary).await {
                            if step_completed {
                                info!(tool = %tc.name, "Plan step auto-completed by tool result");
                            }
                        }
                    }
                }

                // Emit plan status updates for plan_manager tool calls
                if tc.name == "plan_manager" && !is_error {
                    if let Some(ref ps) = self.plan_store {
                        self.emit_plan_status_update(
                            ps,
                            session_id,
                            &tc.arguments,
                            &result_text,
                            &status_tx,
                        ).await;
                    }
                }

                let tool_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "tool".to_string(),
                    content: Some(result_text),
                    tool_call_id: Some(tc.id.clone()),
                    tool_name: Some(tc.name.clone()),
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.3, // Tool outputs default to lower importance
                    embedding: None,
                };
                self.state.append_message(&tool_msg).await?;
            }

            // Update stall detection
            if successful_tool_calls == 0 {
                stall_count += 1;
            } else {
                stall_count = 0; // Reset on any successful progress
            }
        }
    }

    /// Graceful response when task timeout is reached.
    async fn graceful_timeout_response(
        &self,
        session_id: &str,
        learning_ctx: &LearningContext,
        elapsed: Duration,
    ) -> anyhow::Result<String> {
        self.pause_active_plan(session_id, "timeout").await;
        let summary = format!(
            "I've been working on this task for {} minutes and reached the time limit. \
            Here's what I accomplished:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n\
            The task may be incomplete. You can continue where I left off or try breaking it into smaller parts.",
            elapsed.as_secs() / 60,
            learning_ctx.tool_calls.len(),
            learning_ctx.errors.len()
        );

        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(summary.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        self.state.append_message(&assistant_msg).await?;
        Ok(summary)
    }

    /// Graceful response when task token budget is exhausted.
    async fn graceful_budget_response(
        &self,
        session_id: &str,
        learning_ctx: &LearningContext,
        tokens_used: u64,
    ) -> anyhow::Result<String> {
        self.pause_active_plan(session_id, "token_budget").await;
        let summary = format!(
            "I've used {} tokens on this task and reached the budget limit. \
            Here's what I accomplished:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n\
            The task may be incomplete. You can continue where I left off.",
            tokens_used,
            learning_ctx.tool_calls.len(),
            learning_ctx.errors.len()
        );

        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(summary.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        self.state.append_message(&assistant_msg).await?;
        Ok(summary)
    }

    /// Graceful response when agent is stalled (no progress).
    async fn graceful_stall_response(
        &self,
        session_id: &str,
        learning_ctx: &LearningContext,
    ) -> anyhow::Result<String> {
        self.pause_active_plan(session_id, "stall").await;
        let summary = format!(
            "I seem to be stuck and not making progress. \
            Here's what I tried:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n\
            Recent errors:\n{}\n\n\
            Please try rephrasing your request or providing more specific guidance.",
            learning_ctx.tool_calls.len(),
            learning_ctx.errors.len(),
            learning_ctx.errors.iter()
                .rev()
                .take(3)
                .map(|(e, _)| format!("- {}", e.chars().take(100).collect::<String>()))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(summary.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        self.state.append_message(&assistant_msg).await?;
        Ok(summary)
    }

    /// Graceful response when repetitive tool calls are detected.
    async fn graceful_repetitive_response(
        &self,
        session_id: &str,
        learning_ctx: &LearningContext,
        tool_name: &str,
    ) -> anyhow::Result<String> {
        self.pause_active_plan(session_id, "repetitive").await;
        let summary = format!(
            "I noticed I'm calling `{}` repeatedly with similar parameters, which suggests I'm stuck in a loop. \
            Here's what I've done so far:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n\
            Please try a different approach or provide more specific instructions.",
            tool_name,
            learning_ctx.tool_calls.len(),
            learning_ctx.errors.len()
        );

        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(summary.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        self.state.append_message(&assistant_msg).await?;
        Ok(summary)
    }

    /// Graceful response when hard iteration cap is reached (legacy mode).
    async fn graceful_cap_response(
        &self,
        session_id: &str,
        learning_ctx: &LearningContext,
        iterations: usize,
    ) -> anyhow::Result<String> {
        self.pause_active_plan(session_id, "iteration_cap").await;
        let summary = format!(
            "I've reached the maximum iteration limit ({} iterations). \
            Here's what I accomplished:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n\
            The task may be incomplete. Consider increasing the iteration limit in config or using unlimited mode.",
            iterations,
            learning_ctx.tool_calls.len(),
            learning_ctx.errors.len()
        );

        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(summary.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        self.state.append_message(&assistant_msg).await?;
        Ok(summary)
    }

    /// Pause any active plan for the session when hitting a safety limit.
    async fn pause_active_plan(&self, session_id: &str, reason: &str) {
        if let Some(ref st) = self.step_tracker {
            match st.pause_plan(session_id).await {
                Ok(Some(plan)) => info!(plan_id = %plan.id, reason, "Paused plan due to safety limit"),
                Ok(None) => {}
                Err(e) => warn!(error = %e, "Failed to pause plan on safety exit"),
            }
        }
    }

    /// Emit a TaskEnd event. Called from every exit path in the agent loop.
    #[allow(clippy::too_many_arguments)]
    async fn emit_task_end(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        status: TaskStatus,
        task_start: Instant,
        iteration: usize,
        tool_calls_count: usize,
        error: Option<String>,
        summary: Option<String>,
    ) {
        let _ = emitter
            .emit(
                EventType::TaskEnd,
                TaskEndData {
                    task_id: task_id.to_string(),
                    status,
                    duration_secs: task_start.elapsed().as_secs(),
                    iterations: iteration as u32,
                    tool_calls_count: tool_calls_count as u32,
                    error,
                    summary,
                },
            )
            .await;
    }

    async fn execute_tool(
        &self,
        name: &str,
        arguments: &str,
        session_id: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        channel_visibility: ChannelVisibility,
    ) -> anyhow::Result<String> {
        let enriched_args = match serde_json::from_str::<Value>(arguments) {
            Ok(Value::Object(mut map)) => {
                map.insert("_session_id".to_string(), json!(session_id));
                map.insert(
                    "_channel_visibility".to_string(),
                    json!(channel_visibility.to_string()),
                );
                // Mark as untrusted if this session originated from an automated
                // trigger (e.g., email) rather than direct user interaction.
                // This forces tools like terminal to require explicit approval.
                if is_trigger_session(session_id) {
                    map.insert("_untrusted_source".to_string(), json!(true));
                }
                serde_json::to_string(&map)?
            }
            _ => arguments.to_string(),
        };

        // Path verification pre-check: gate file-modifying terminal commands
        if name == "terminal" {
            if let Some(ref tracker) = self.verification_tracker {
                if let Some(cmd) = extract_command_from_args(&enriched_args) {
                    if let Some(warning) = tracker.check_modifying_command(session_id, &cmd).await {
                        return Ok(format!(
                            "[VERIFICATION WARNING] {}\nUnverified paths: {}\n\
                             Verify targets exist using 'ls' or 'stat' first, then retry.",
                            warning.message,
                            warning.unverified_paths.join(", ")
                        ));
                    }
                }
            }
        }

        for tool in &self.tools {
            if tool.name() == name {
                let result = tool.call_with_status(&enriched_args, status_tx).await;

                // Post-execution: record seen paths from successful commands
                if result.is_ok() {
                    if let Some(ref tracker) = self.verification_tracker {
                        match name {
                            "terminal" => {
                                if let Some(cmd) = extract_command_from_args(&enriched_args) {
                                    tracker.record_from_command(session_id, &cmd).await;
                                }
                            }
                            "send_file" => {
                                if let Some(path) = extract_file_path_from_args(&enriched_args) {
                                    tracker.record_seen_path(session_id, &path).await;
                                }
                            }
                            _ => {}
                        }
                    }
                }

                return result;
            }
        }
        anyhow::bail!("Unknown tool: {}", name)
    }

    /// Emit status updates for plan_manager tool calls.
    async fn emit_plan_status_update(
        &self,
        plan_store: &Arc<PlanStore>,
        session_id: &str,
        arguments: &str,
        result_text: &str,
        status_tx: &Option<mpsc::Sender<StatusUpdate>>,
    ) {
        // Parse action from arguments
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|v| v["action"].as_str().map(String::from))
            .unwrap_or_default();

        // Get current plan state for accurate status info
        let plan = plan_store
            .get_incomplete_for_session(session_id)
            .await
            .ok()
            .flatten();

        match action.as_str() {
            "create" => {
                // Parse plan info from result: "Created plan 'X' with N steps..."
                if result_text.contains("Created plan") {
                    if let Some(plan) = plan {
                        send_status(
                            status_tx,
                            StatusUpdate::PlanCreated {
                                plan_id: plan.id.clone(),
                                description: plan.description.clone(),
                                total_steps: plan.steps.len(),
                            },
                        );
                        // Also emit step start for first step
                        if let Some(step) = plan.steps.first() {
                            send_status(
                                status_tx,
                                StatusUpdate::PlanStepStart {
                                    plan_id: plan.id.clone(),
                                    step_index: 0,
                                    total_steps: plan.steps.len(),
                                    description: step.description.clone(),
                                },
                            );
                        }
                    }
                }
            }
            "complete_step" => {
                // Check if plan completed or moved to next step
                if result_text.contains("All steps done") {
                    // Plan completed - need to get the now-completed plan
                    if let Ok(Some(completed_plan)) = plan_store
                        .get_recent_for_session(session_id, 1)
                        .await
                        .map(|v| v.into_iter().next())
                    {
                        send_status(
                            status_tx,
                            StatusUpdate::PlanComplete {
                                plan_id: completed_plan.id.clone(),
                                description: completed_plan.description.clone(),
                                total_steps: completed_plan.steps.len(),
                                duration_secs: completed_plan.duration_secs(),
                            },
                        );
                    }
                } else if let Some(plan) = plan {
                    // Step completed, now on next step
                    let prev_step = plan.current_step.saturating_sub(1);
                    if let Some(completed_step) = plan.steps.get(prev_step) {
                        send_status(
                            status_tx,
                            StatusUpdate::PlanStepComplete {
                                plan_id: plan.id.clone(),
                                step_index: prev_step,
                                total_steps: plan.steps.len(),
                                description: completed_step.description.clone(),
                                summary: completed_step.result_summary.clone(),
                            },
                        );
                    }
                    // Emit start for current step
                    if let Some(current_step) = plan.steps.get(plan.current_step) {
                        send_status(
                            status_tx,
                            StatusUpdate::PlanStepStart {
                                plan_id: plan.id.clone(),
                                step_index: plan.current_step,
                                total_steps: plan.steps.len(),
                                description: current_step.description.clone(),
                            },
                        );
                    }
                }
            }
            "fail_step" => {
                if let Some(plan) = plan {
                    if let Some(step) = plan.steps.get(plan.current_step) {
                        send_status(
                            status_tx,
                            StatusUpdate::PlanStepFailed {
                                plan_id: plan.id.clone(),
                                step_index: plan.current_step,
                                description: step.description.clone(),
                                error: step.error.clone().unwrap_or_else(|| "Unknown error".to_string()),
                            },
                        );
                    }
                }
            }
            "skip_step" => {
                // Similar to complete_step but the step was skipped
                if result_text.contains("All steps done") {
                    if let Ok(Some(completed_plan)) = plan_store
                        .get_recent_for_session(session_id, 1)
                        .await
                        .map(|v| v.into_iter().next())
                    {
                        send_status(
                            status_tx,
                            StatusUpdate::PlanComplete {
                                plan_id: completed_plan.id.clone(),
                                description: completed_plan.description.clone(),
                                total_steps: completed_plan.steps.len(),
                                duration_secs: completed_plan.duration_secs(),
                            },
                        );
                    }
                } else if let Some(plan) = plan {
                    // Now on next step after skip
                    if let Some(current_step) = plan.steps.get(plan.current_step) {
                        send_status(
                            status_tx,
                            StatusUpdate::PlanStepStart {
                                plan_id: plan.id.clone(),
                                step_index: plan.current_step,
                                total_steps: plan.steps.len(),
                                description: current_step.description.clone(),
                            },
                        );
                    }
                }
            }
            "abandon" => {
                // Plan was abandoned - result contains the description
                if result_text.contains("Abandoned plan") {
                    // Extract description from result text
                    let description = result_text
                        .strip_prefix("Abandoned plan '")
                        .and_then(|s| s.split('\'').next())
                        .unwrap_or("Unknown")
                        .to_string();

                    send_status(
                        status_tx,
                        StatusUpdate::PlanAbandoned {
                            plan_id: String::new(), // Plan is now abandoned so we can't get ID
                            description,
                        },
                    );
                }
            }
            "resume" => {
                // Plan resumed - emit step start for current step
                if let Some(plan) = plan {
                    if let Some(step) = plan.steps.get(plan.current_step) {
                        send_status(
                            status_tx,
                            StatusUpdate::PlanStepStart {
                                plan_id: plan.id.clone(),
                                step_index: plan.current_step,
                                total_steps: plan.steps.len(),
                                description: step.description.clone(),
                            },
                        );
                    }
                }
            }
            "revise" => {
                // Plan was revised - emit revision status and current step
                if result_text.contains("Revised plan") {
                    if let Some(plan) = plan {
                        // Extract reason from arguments
                        let reason = serde_json::from_str::<Value>(arguments)
                            .ok()
                            .and_then(|v| v["revision_reason"].as_str().map(String::from))
                            .unwrap_or_else(|| "User requested changes".to_string());

                        send_status(
                            status_tx,
                            StatusUpdate::PlanRevised {
                                plan_id: plan.id.clone(),
                                description: plan.description.clone(),
                                reason,
                                new_total_steps: plan.steps.len(),
                            },
                        );

                        // Also emit step start for current step
                        if let Some(step) = plan.steps.get(plan.current_step) {
                            send_status(
                                status_tx,
                                StatusUpdate::PlanStepStart {
                                    plan_id: plan.id.clone(),
                                    step_index: plan.current_step,
                                    total_steps: plan.steps.len(),
                                    description: step.description.clone(),
                                },
                            );
                        }
                    }
                }
            }
            _ => {} // get, checkpoint, retry_step don't need status updates
        }
    }
}

/// Process learning from a completed task - runs in background.
async fn process_learning(state: &Arc<dyn StateStore>, ctx: LearningContext) -> anyhow::Result<()> {
    use crate::memory::{expertise, procedures};

    // Determine if task was successful
    let unrecovered_errors = ctx.errors.iter().filter(|(_, recovered)| !recovered).count();
    let task_success = ctx.completed_naturally && unrecovered_errors == 0;

    // 1. Update expertise for detected domains
    let domains = expertise::detect_domains(&ctx.user_text);
    for domain in &domains {
        let error = if !task_success {
            ctx.errors.first().map(|(e, _)| e.as_str())
        } else {
            None
        };
        if let Err(e) = state.increment_expertise(domain, task_success, error).await {
            warn!(domain = %domain, error = %e, "Failed to update expertise");
        }
    }

    // 2. Save procedure if successful with 2+ actions
    if task_success && ctx.tool_calls.len() >= 2 {
        let generalized = procedures::generalize_procedure(&ctx.tool_calls);
        let procedure = procedures::create_procedure(
            procedures::generate_procedure_name(&ctx.user_text),
            procedures::extract_trigger_pattern(&ctx.user_text),
            generalized,
        );
        if let Err(e) = state.upsert_procedure(&procedure).await {
            warn!(procedure = %procedure.name, error = %e, "Failed to save procedure");
        }
    }

    // 3. Learn error-solution if error was recovered
    if let Some(error) = ctx.first_error {
        if !ctx.recovery_actions.is_empty() {
            let solution = procedures::create_error_solution(
                procedures::extract_error_pattern(&error),
                domains.into_iter().next(),
                procedures::summarize_solution(&ctx.recovery_actions),
                Some(ctx.recovery_actions),
            );
            if let Err(e) = state.insert_error_solution(&solution).await {
                warn!(error_pattern = %solution.error_pattern, error = %e, "Failed to save error solution");
            }
        }
    }

    Ok(())
}

/// Merge consecutive same-role messages (assistant or user) to satisfy
/// provider ordering constraints (Gemini, Anthropic). Combines content
/// text and tool_calls arrays.
fn merge_consecutive_messages(messages: &mut Vec<Value>) {
    let mut i = 1;
    while i < messages.len() {
        let prev_role = messages[i - 1]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        let curr_role = messages[i]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        if prev_role == curr_role && (curr_role == "assistant" || curr_role == "user") {
            // Merge content text
            let curr_content = messages[i]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let prev_content = messages[i - 1]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            if !curr_content.is_empty() {
                let merged = if prev_content.is_empty() {
                    curr_content
                } else {
                    format!("{}\n{}", prev_content, curr_content)
                };
                messages[i - 1]["content"] = json!(merged);
            }
            // Combine tool_calls arrays (not replace)
            let curr_tcs = messages[i]
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .cloned();
            if let Some(curr_arr) = curr_tcs {
                if let Some(prev_arr) = messages[i - 1]
                    .get_mut("tool_calls")
                    .and_then(|v| v.as_array_mut())
                {
                    prev_arr.extend(curr_arr);
                } else {
                    messages[i - 1]["tool_calls"] = json!(curr_arr);
                }
            }
            messages.remove(i);
        } else {
            i += 1;
        }
    }
}

/// Three-pass fixup for provider message ordering constraints.
///
/// 1. Merge consecutive same-role messages
/// 2. Drop orphaned tool results and strip orphaned tool_calls
/// 3. Merge again (dropping orphans can create new consecutive messages)
///
/// Returns the number of messages before the first non-system message
/// that is a tool-role message (should always be 0 after fixup).
fn fixup_message_ordering(messages: &mut Vec<Value>) {
    // Pass 1
    merge_consecutive_messages(messages);

    // Pass 2: drop orphans in both directions
    {
        let assistant_tc_ids: std::collections::HashSet<String> = messages
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
            .filter_map(|m| m.get("tool_calls"))
            .filter_map(|tcs| tcs.as_array())
            .flat_map(|arr| arr.iter())
            .filter_map(|tc| tc.get("id").and_then(|id| id.as_str()))
            .map(|s| s.to_string())
            .collect();

        let tool_result_ids: std::collections::HashSet<String> = messages
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
            .filter_map(|m| m.get("tool_call_id").and_then(|id| id.as_str()))
            .map(|s| s.to_string())
            .collect();

        // Drop orphaned tool results
        messages.retain(|m| {
            if m.get("role").and_then(|r| r.as_str()) == Some("tool") {
                let tc_id = m.get("tool_call_id").and_then(|id| id.as_str()).unwrap_or("");
                assistant_tc_ids.contains(tc_id)
            } else {
                true
            }
        });

        // Strip orphaned tool_calls from assistant messages
        for m in messages.iter_mut() {
            if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
                continue;
            }
            if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()).cloned() {
                let filtered: Vec<Value> = tcs
                    .into_iter()
                    .filter(|tc| {
                        tc.get("id")
                            .and_then(|id| id.as_str())
                            .is_some_and(|id| tool_result_ids.contains(id))
                    })
                    .collect();
                if filtered.is_empty() {
                    m.as_object_mut().map(|o| o.remove("tool_calls"));
                } else {
                    m["tool_calls"] = json!(filtered);
                }
            }
        }

        // Drop assistant messages that became empty
        messages.retain(|m| {
            if m.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                let has_content = m
                    .get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| !s.is_empty());
                let has_tool_calls = m
                    .get("tool_calls")
                    .and_then(|tc| tc.as_array())
                    .is_some_and(|a| !a.is_empty());
                has_content || has_tool_calls
            } else {
                true
            }
        });
    }

    // Pass 3
    merge_consecutive_messages(messages);

    // Pass 4: Ensure the first non-system message is a user message.
    // Gemini requires that assistant/tool messages come after a user turn.
    // If history eviction dropped the original user message but kept later
    // assistant+tool pairs, we need to drop those leading non-user messages.
    if let Some(first_non_system) = messages.iter().position(|m| {
        m.get("role").and_then(|r| r.as_str()) != Some("system")
    }) {
        let first_role = messages[first_non_system]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        if first_role != "user" {
            if let Some(first_user_rel) = messages[first_non_system..].iter().position(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
            }) {
                let abs_end = first_non_system + first_user_rel;
                warn!(
                    dropped = abs_end - first_non_system,
                    "Dropping leading non-user messages to satisfy provider ordering"
                );
                messages.drain(first_non_system..abs_end);
            }
        }
    }

    // Pass 5: Gemini-specific - ensure assistant messages with tool_calls only follow
    // user or tool messages (not other assistant messages).
    // If we find assistant→assistant where the second has tool_calls, merge them.
    let mut i = 1;
    while i < messages.len() {
        let prev_role = messages[i - 1].get("role").and_then(|r| r.as_str()).unwrap_or("");
        let curr_role = messages[i].get("role").and_then(|r| r.as_str()).unwrap_or("");
        let curr_has_tc = messages[i]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty());

        // If current is assistant with tool_calls, and previous is also assistant
        // (which shouldn't happen after Pass 3, but just in case), merge them
        if prev_role == "assistant" && curr_role == "assistant" && curr_has_tc {
            warn!(
                "Pass 5: Found consecutive assistant messages, merging to satisfy Gemini constraint"
            );
            // Merge content
            let curr_content = messages[i]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let prev_content = messages[i - 1]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            if !curr_content.is_empty() {
                let merged = if prev_content.is_empty() {
                    curr_content
                } else {
                    format!("{}\n{}", prev_content, curr_content)
                };
                messages[i - 1]["content"] = json!(merged);
            }
            // Merge tool_calls
            if let Some(curr_tcs) = messages[i].get("tool_calls").and_then(|v| v.as_array()).cloned() {
                if let Some(prev_tcs) = messages[i - 1].get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                    prev_tcs.extend(curr_tcs);
                } else {
                    messages[i - 1]["tool_calls"] = json!(curr_tcs);
                }
            }
            messages.remove(i);
        } else {
            i += 1;
        }
    }

    // Pass 6: Gemini-specific - ensure assistant messages with tool_calls only follow
    // user or tool messages. If an assistant(tc) follows another assistant(tc) or
    // a plain assistant, strip the tool_calls from the second one (keeping any content).
    // This handles edge cases not caught by Pass 5.
    let mut i = 1;
    while i < messages.len() {
        let curr_role = messages[i].get("role").and_then(|r| r.as_str()).unwrap_or("");
        let curr_has_tc = messages[i]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty());

        if curr_role == "assistant" && curr_has_tc {
            let prev_role = messages[i - 1].get("role").and_then(|r| r.as_str()).unwrap_or("");
            // Valid predecessors for assistant with tool_calls: "user" or "tool"
            if prev_role != "user" && prev_role != "tool" {
                warn!(
                    prev_role,
                    "Pass 6: Stripping tool_calls from assistant that doesn't follow user/tool"
                );
                // Remove tool_calls but keep content
                messages[i].as_object_mut().map(|o| o.remove("tool_calls"));
                // If this leaves an empty assistant, it will be caught by the retain logic
                // but since we're past that, let's check now
                let has_content = messages[i]
                    .get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| !s.is_empty());
                if !has_content {
                    messages.remove(i);
                    continue;
                }
            }
        }
        i += 1;
    }
}

/// Extract the "command" field from tool arguments JSON (for terminal tool).
fn extract_command_from_args(args_json: &str) -> Option<String> {
    serde_json::from_str::<Value>(args_json)
        .ok()
        .and_then(|v| v.get("command")?.as_str().map(String::from))
}

/// Extract the "file_path" field from tool arguments JSON (for send_file tool).
fn extract_file_path_from_args(args_json: &str) -> Option<String> {
    serde_json::from_str::<Value>(args_json)
        .ok()
        .and_then(|v| v.get("file_path")?.as_str().map(String::from))
}

/// Check if a session ID indicates it was triggered by an automated source
/// (e.g., email trigger) rather than direct user interaction via Telegram.
fn is_trigger_session(session_id: &str) -> bool {
    session_id.contains("trigger") || session_id.starts_with("event_")
}

/// Hash a tool call (name + arguments) for repetitive behavior detection.
fn hash_tool_call(name: &str, arguments: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    arguments.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod message_ordering_tests {
    use super::*;
    use serde_json::json;

    /// Helper: assert no tool message appears without a matching assistant tool_call.
    fn assert_no_orphaned_tools(messages: &[Value]) {
        let assistant_tc_ids: std::collections::HashSet<String> = messages
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
            .filter_map(|m| m.get("tool_calls"))
            .filter_map(|tcs| tcs.as_array())
            .flat_map(|arr| arr.iter())
            .filter_map(|tc| tc.get("id").and_then(|id| id.as_str()))
            .map(|s| s.to_string())
            .collect();

        for m in messages {
            if m.get("role").and_then(|r| r.as_str()) == Some("tool") {
                let tc_id = m.get("tool_call_id").and_then(|id| id.as_str()).unwrap_or("");
                assert!(
                    assistant_tc_ids.contains(tc_id),
                    "Orphaned tool message: tool_call_id={} has no matching assistant tool_call",
                    tc_id
                );
            }
        }
    }

    /// Helper: assert no assistant tool_call exists without a matching tool result.
    fn assert_no_orphaned_tool_calls(messages: &[Value]) {
        let tool_result_ids: std::collections::HashSet<String> = messages
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
            .filter_map(|m| m.get("tool_call_id").and_then(|id| id.as_str()))
            .map(|s| s.to_string())
            .collect();

        for m in messages {
            if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
                continue;
            }
            if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    let id = tc.get("id").and_then(|id| id.as_str()).unwrap_or("");
                    assert!(
                        tool_result_ids.contains(id),
                        "Orphaned tool_call: id={} has no matching tool result",
                        id
                    );
                }
            }
        }
    }

    /// Helper: assert no consecutive same-role messages.
    fn assert_no_consecutive_same_role(messages: &[Value]) {
        for i in 1..messages.len() {
            let prev = messages[i - 1].get("role").and_then(|r| r.as_str()).unwrap_or("");
            let curr = messages[i].get("role").and_then(|r| r.as_str()).unwrap_or("");
            if (curr == "assistant" || curr == "user") && prev == curr {
                panic!(
                    "Consecutive same-role messages at index {}-{}: role={}",
                    i - 1, i, curr
                );
            }
        }
    }

    /// Helper: assert the first non-system message is NOT a tool message.
    fn assert_no_leading_tool(messages: &[Value]) {
        for m in messages {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role == "system" {
                continue;
            }
            assert_ne!(
                role, "tool",
                "First non-system message is a tool message (orphaned function_response)"
            );
            break;
        }
    }

    fn assert_all_invariants(messages: &[Value]) {
        assert_no_orphaned_tools(messages);
        assert_no_orphaned_tool_calls(messages);
        assert_no_consecutive_same_role(messages);
        assert_no_leading_tool(messages);
    }

    fn tc(id: &str, name: &str) -> Value {
        json!({"id": id, "type": "function", "function": {"name": name, "arguments": "{}"}})
    }

    #[test]
    fn test_clean_conversation_unchanged() {
        let mut msgs = vec![
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": "I'll check", "tool_calls": [tc("c1", "terminal")]}),
            json!({"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "ok"}),
            json!({"role": "assistant", "content": "Done"}),
        ];
        fixup_message_ordering(&mut msgs);
        assert_eq!(msgs.len(), 4);
        assert_all_invariants(&msgs);
    }

    #[test]
    fn test_orphaned_tool_at_start_of_window() {
        // Context window starts with tool result whose assistant is outside window.
        let mut msgs = vec![
            json!({"role": "tool", "tool_call_id": "c0", "name": "terminal", "content": "old result"}),
            json!({"role": "assistant", "content": "noted"}),
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": "hi"}),
        ];
        fixup_message_ordering(&mut msgs);
        assert_all_invariants(&msgs);
        // The orphaned tool should be gone
        assert!(msgs.iter().all(|m| m.get("role").and_then(|r| r.as_str()) != Some("tool")));
    }

    #[test]
    fn test_two_orphaned_tools_at_start() {
        let mut msgs = vec![
            json!({"role": "tool", "tool_call_id": "c0", "name": "terminal", "content": "r0"}),
            json!({"role": "tool", "tool_call_id": "c1", "name": "browser", "content": "r1"}),
            json!({"role": "assistant", "content": "summary of prev"}),
            json!({"role": "user", "content": "next question"}),
            json!({"role": "assistant", "content": "answer"}),
        ];
        fixup_message_ordering(&mut msgs);
        assert_all_invariants(&msgs);
    }

    #[test]
    fn test_orphan_drop_creates_consecutive_assistants() {
        // assistant A → tool(orphaned) → assistant B → user
        // After dropping tool, assistant A and B are consecutive → must merge.
        let mut msgs = vec![
            json!({"role": "assistant", "content": "step 1", "tool_calls": [tc("c1", "terminal")]}),
            json!({"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "result"}),
            json!({"role": "assistant", "content": "step 2", "tool_calls": [tc("c2", "browser")]}),
            // c2 tool result is missing (outside window)
            json!({"role": "user", "content": "ok"}),
            json!({"role": "assistant", "content": "done"}),
        ];
        fixup_message_ordering(&mut msgs);
        assert_all_invariants(&msgs);
    }

    #[test]
    fn test_multiple_tool_calls_partial_orphan() {
        // Assistant has 2 tool_calls, only 1 has a result in context.
        let mut msgs = vec![
            json!({"role": "user", "content": "do stuff"}),
            json!({"role": "assistant", "content": "ok", "tool_calls": [tc("c1", "terminal"), tc("c2", "browser")]}),
            json!({"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "result1"}),
            // c2 result missing
            json!({"role": "assistant", "content": "done"}),
        ];
        fixup_message_ordering(&mut msgs);
        assert_all_invariants(&msgs);
        // c2 should be stripped from tool_calls but c1 kept
        let assistant_tc = &msgs[1];
        let tcs = assistant_tc.get("tool_calls").unwrap().as_array().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["id"], "c1");
    }

    #[test]
    fn test_long_agentic_loop_context_window() {
        // Simulates 10 iterations with a 20-message window.
        // First few iterations' messages are outside the window.
        let mut msgs = vec![];
        // Messages 0-19 from a long conversation — window starts mid-conversation.
        // Old orphaned tool:
        msgs.push(json!({"role": "tool", "tool_call_id": "old_c1", "name": "terminal", "content": "old"}));
        // Old assistant final response:
        msgs.push(json!({"role": "assistant", "content": "done with prev task"}));
        // New user message:
        msgs.push(json!({"role": "user", "content": "new task"}));
        // 5 iterations of assistant→tool pairs:
        for i in 0..5 {
            let cid = format!("iter_{}", i);
            msgs.push(json!({"role": "assistant", "content": format!("step {}", i), "tool_calls": [tc(&cid, "terminal")]}));
            msgs.push(json!({"role": "tool", "tool_call_id": cid, "name": "terminal", "content": format!("result {}", i)}));
        }

        fixup_message_ordering(&mut msgs);
        assert_all_invariants(&msgs);
    }

    #[test]
    fn test_assistant_with_null_content_and_tool_calls() {
        let mut msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": null, "tool_calls": [tc("c1", "write_file")]}),
            json!({"role": "tool", "tool_call_id": "c1", "name": "write_file", "content": "ok"}),
            json!({"role": "assistant", "content": "done"}),
        ];
        fixup_message_ordering(&mut msgs);
        assert_all_invariants(&msgs);
        assert_eq!(msgs.len(), 4);
    }

    #[test]
    fn test_merge_combines_tool_calls() {
        // Two consecutive assistants with different tool_calls → merge should combine.
        let mut msgs = vec![
            json!({"role": "assistant", "content": "a", "tool_calls": [tc("c1", "t1")]}),
            json!({"role": "assistant", "content": "b", "tool_calls": [tc("c2", "t2")]}),
        ];
        merge_consecutive_messages(&mut msgs);
        assert_eq!(msgs.len(), 1);
        let tcs = msgs[0].get("tool_calls").unwrap().as_array().unwrap();
        assert_eq!(tcs.len(), 2);
    }
}
