use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use serde_json::{json, Value};
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::Uuid;

use crate::providers::{ProviderError, ProviderErrorKind};
use crate::skills::{self, Skill};
use crate::traits::{Message, ModelProvider, StateStore, Tool};

const MAX_ITERATIONS: usize = 10;
const DEFAULT_MAX_DEPTH: usize = 3;

pub struct Agent {
    provider: Arc<dyn ModelProvider>,
    state: Arc<dyn StateStore>,
    tools: Vec<Arc<dyn Tool>>,
    model: RwLock<String>,
    fallback_model: RwLock<String>,
    system_prompt: String,
    config_path: PathBuf,
    skills: Vec<Skill>,
    /// Current recursion depth (0 = root agent).
    depth: usize,
    /// Maximum allowed recursion depth for sub-agent spawning.
    max_depth: usize,
}

impl Agent {
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        state: Arc<dyn StateStore>,
        tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        config_path: PathBuf,
        skills: Vec<Skill>,
    ) -> Self {
        let fallback = model.clone();
        Self {
            provider,
            state,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skills,
            depth: 0,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Create an Agent with explicit depth/max_depth (used internally for sub-agents).
    fn with_depth(
        provider: Arc<dyn ModelProvider>,
        state: Arc<dyn StateStore>,
        tools: Vec<Arc<dyn Tool>>,
        model: String,
        system_prompt: String,
        config_path: PathBuf,
        skills: Vec<Skill>,
        depth: usize,
        max_depth: usize,
    ) -> Self {
        let fallback = model.clone();
        Self {
            provider,
            state,
            tools,
            model: RwLock::new(model),
            fallback_model: RwLock::new(fallback),
            system_prompt,
            config_path,
            skills,
            depth,
            max_depth,
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
        let child_system_prompt = format!(
            "{}\n\n## Sub-Agent Context\n\
            You are a sub-agent (depth {}/{}) spawned to accomplish a specific mission.\n\
            **Mission:** {}\n\n\
            Focus exclusively on this mission. Be concise. Return your findings/results \
            directly — they will be consumed by the parent agent.",
            self.system_prompt, child_depth, self.max_depth, mission
        );

        let child_session = format!("sub-{}-{}", child_depth, Uuid::new_v4());

        info!(
            parent_depth = self.depth,
            child_depth,
            child_session = %child_session,
            mission,
            "Spawning sub-agent"
        );

        // If the child can still recurse, give it a spawn tool with a deferred
        // agent reference (set after wrapping in Arc).
        if child_depth < self.max_depth {
            let spawn_tool = Arc::new(crate::tools::spawn::SpawnAgentTool::new_deferred());

            let mut child_tools = base_tools;
            child_tools.push(spawn_tool.clone());

            let child = Arc::new(Agent::with_depth(
                self.provider.clone(),
                self.state.clone(),
                child_tools,
                model,
                child_system_prompt,
                self.config_path.clone(),
                self.skills.clone(),
                child_depth,
                self.max_depth,
            ));

            // Close the loop: give the spawn tool a weak ref to the child.
            spawn_tool.set_agent(Arc::downgrade(&child));

            child.handle_message(&child_session, task).await
        } else {
            // At max depth — no spawn tool, no recursion.
            let child = Arc::new(Agent::with_depth(
                self.provider.clone(),
                self.state.clone(),
                base_tools,
                model,
                child_system_prompt,
                self.config_path.clone(),
                self.skills.clone(),
                child_depth,
                self.max_depth,
            ));

            child.handle_message(&child_session, task).await
        }
    }

    /// Get the current model name.
    pub async fn current_model(&self) -> String {
        self.model.read().await.clone()
    }

    /// Switch the active model at runtime. Keeps the old model as fallback.
    pub async fn set_model(&self, model: String) {
        let mut m = self.model.write().await;
        let mut fb = self.fallback_model.write().await;
        info!(old = %*m, new = %model, "Model switched");
        *fb = m.clone();
        *m = model;
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

    /// Attempt an LLM call with error-classified recovery:
    /// - RateLimit → wait retry_after, retry once
    /// - Timeout/Network/ServerError → retry once
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

                    // --- Rate limit: wait and retry once ---
                    ProviderErrorKind::RateLimit => {
                        let wait = provider_err.retry_after_secs.unwrap_or(5);
                        let wait = wait.min(60); // cap at 60s
                        info!(wait_secs = wait, "Rate limited, waiting before retry");
                        tokio::time::sleep(Duration::from_secs(wait)).await;
                        self.provider.chat(model, messages, tool_defs).await
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
                                // Second failure — try the fallback model
                                let fallback = self.fallback_model.read().await.clone();
                                if fallback != model {
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
                        let fallback = self.fallback_model.read().await.clone();
                        if fallback != model {
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
    ) -> anyhow::Result<String> {
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

        let tool_defs = self.tool_definitions();
        let model = self.model.read().await.clone();

        // 2. Build system prompt ONCE before the loop: match skills + inject facts
        let active_skills = skills::match_skills(&self.skills, user_text);
        if !active_skills.is_empty() {
            let names: Vec<&str> = active_skills.iter().map(|s| s.name.as_str()).collect();
            info!(session_id, skills = ?names, "Matched skills for message");
        }
        let facts = self.state.get_facts(None).await?;
        let system_prompt = skills::build_system_prompt(
            &self.system_prompt,
            &self.skills,
            &active_skills,
            &facts,
        );

        // 2b. Retrieve Context ONCE (Optimization)
        // Use Tri-Hybrid Retrieval to get relevant history
        let mut initial_history = self.state.get_context(session_id, user_text, 50).await?;

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

        // 3. Agentic loop
        // 3. Agentic loop
        for iteration in 0..MAX_ITERATIONS {
            info!(iteration, session_id, model = %model, depth = self.depth, "Agent loop iteration");

            // Fetch only recent history inside the loop
            let recent_history = self.state.get_history(session_id, 20).await?;
            
            // Merge Pinned + Recent using iterators to avoid cloning the Message structs
            // We expect no overlap because pinned are strictly older, but we dedup just in case
            let mut seen_ids: std::collections::HashSet<&String> = std::collections::HashSet::new();
            
            let mut messages: Vec<Value> = pinned_memories.iter()
                .chain(recent_history.iter())
                .filter(|m| seen_ids.insert(&m.id)) // Dedup
                .map(|m| {
                    let mut obj = json!({
                        "role": m.role,
                        "content": m.content,
                    });
                    if let Some(tc_json) = &m.tool_calls_json {
                        obj["tool_calls"] = serde_json::from_str::<Value>(tc_json).unwrap_or(json!([]));
                    }
                    if let Some(name) = &m.tool_name {
                        obj["name"] = json!(name);
                    }
                    if let Some(tcid) = &m.tool_call_id {
                        obj["tool_call_id"] = json!(tcid);
                    }
                    obj
                })
                .collect();
            
            messages.insert(0, json!({
                "role": "system",
                "content": system_prompt,
            }));

            // 4. Call the LLM with error-classified recovery
            // On the last iteration, omit tools to force a text response
            let effective_tools = if iteration >= MAX_ITERATIONS - 1 {
                info!(session_id, "Last iteration — calling LLM without tools to force summary");
                &[][..]
            } else {
                &tool_defs[..]
            };
            let resp = self
                .call_llm_with_recovery(&model, &messages, effective_tools)
                .await?;

            // Log tool call names for debugging
            let tc_names: Vec<&str> = resp.tool_calls.iter().map(|tc| tc.name.as_str()).collect();
            info!(
                session_id,
                has_content = resp.content.is_some(),
                tool_calls = resp.tool_calls.len(),
                tool_names = ?tc_names,
                "LLM response received"
            );

            // 5. If there are tool calls, execute them
            // On the last iteration, force a text response even if the model returns tool calls
            if !resp.tool_calls.is_empty() && iteration < MAX_ITERATIONS - 1 {
                // Nudge the model to wrap up when approaching the limit
                if iteration >= MAX_ITERATIONS - 2 {
                    let nudge = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "user".to_string(),
                        content: Some(
                            "[SYSTEM: You are running low on tool call iterations. \
                            Summarize your findings and respond to the user NOW. \
                            Do not make any more tool calls.]"
                                .to_string(),
                        ),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.1,
                        embedding: None,
                    };
                    self.state.append_message(&nudge).await?;
                }
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

                // Execute each tool call
                for tc in &resp.tool_calls {
                    let result = self.execute_tool(&tc.name, &tc.arguments, session_id).await;
                    let result_text = match result {
                        Ok(text) => text,
                        Err(e) => format!("Error: {}", e),
                    };

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

                // Continue the loop to let LLM process tool results
                continue;
            }

            // 6. No tool calls (or last iteration) — this is the final response
            let reply = resp.content.filter(|s| !s.is_empty()).unwrap_or_else(|| {
                if iteration >= MAX_ITERATIONS - 1 {
                    "I used all available tool iterations but couldn't finish the task. \
                    Please try a simpler or more specific request."
                        .to_string()
                } else {
                    String::new()
                }
            });
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

            return Ok(reply);
        }

        warn!(session_id, "Agent loop hit max iterations");
        Ok("I've reached the maximum number of steps for this request. Please try again with a simpler query.".to_string())
    }

    async fn execute_tool(&self, name: &str, arguments: &str, session_id: &str) -> anyhow::Result<String> {
        let enriched_args = match serde_json::from_str::<Value>(arguments) {
            Ok(Value::Object(mut map)) => {
                map.insert("_session_id".to_string(), json!(session_id));
                serde_json::to_string(&map)?
            }
            _ => arguments.to_string(),
        };
        for tool in &self.tools {
            if tool.name() == name {
                return tool.call(&enriched_args).await;
            }
        }
        anyhow::bail!("Unknown tool: {}", name)
    }
}
