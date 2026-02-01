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
use crate::traits::{Message, ModelProvider, StateStore, Tool, ToolCall};

const MAX_ITERATIONS: usize = 10;

pub struct Agent {
    provider: Arc<dyn ModelProvider>,
    state: Arc<dyn StateStore>,
    tools: Vec<Arc<dyn Tool>>,
    model: RwLock<String>,
    fallback_model: RwLock<String>,
    system_prompt: String,
    config_path: PathBuf,
    skills: Vec<Skill>,
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
        let initial_history = self.state.get_context(session_id, user_text, 50).await?;

        // 3. Agentic loop
        for iteration in 0..MAX_ITERATIONS {
            info!(iteration, session_id, model = %model, "Agent loop iteration");

            // Build messages array
            let mut messages = Vec::new();

            messages.push(json!({"role": "system", "content": system_prompt}));

            // Use the retrieved history (which includes recent + relevant messages)
            // We append the new messages generated within this loop (tool calls etc)
            // BUT: get_context only returns past messages.
            // We need to verify if we should refresh history on each loop.
            // Actually, get_context returns persisted messages.
            // The tool calls/tool results generated IN THIS LOOP are persisted to state
            // but might NOT be in `initial_history`.
            // So we really need: Initial Context + Dynamic New Messages.
            
            // Re-fetching history is safer for tools, but expensive for vectors.
            // Compromise: Use get_history for the loop (recency) but prepend the Relevant/Salient messages found initially?
            // Or just stick to the user's request: Move it out.
            // If we move it out, we miss the tool calls generated in step 1 when we are in step 2.
            // FIX: We need to merge `initial_history` (past) with `current_session_new_messages`.
            // EASIER FIX: `get_context` is heavy. `get_history` is cheap.
            // We should use `get_context` ONCE to find old relevant stuff.
            // And use `get_history` inside the loop to get the latest state.
            // And merge them.
            
            // For now, let's just use `get_history` inside the loop, but PREPEND the "Relevant/Salient" messages found by `get_context`?
            // Simpler: Just call `get_context` outside, and in the loop we manually fetch *just* the new messages?
            // No, that's complex logic.
            //
            // Correct approach for V1:
            // 1. Fetch `relevant_context` (Search results) OUTSIDE loop.
            // 2. Inside loop, fetch `recent_history` (Recency).
            // 3. Deduplicate and merge.
            
            // Let's implement this "Merge" logic here.
            
            let recent_history = self.state.get_history(session_id, 20).await?; // Fetch latest 20
            
            // Combine initial_history (which has high relevance) + recent_history
            // Deduplicate by ID
            let mut final_history = initial_history.clone();
            for msg in recent_history {
                if !final_history.iter().any(|m| m.id == msg.id) {
                    final_history.push(msg);
                }
            }
            // Sort by created_at
            final_history.sort_by_key(|m| m.created_at);

            // Conversation history
            for msg in &final_history {
                let mut m = json!({"role": msg.role});
                if let Some(ref c) = msg.content {
                    m["content"] = json!(c);
                }
                if let Some(ref tc_json) = msg.tool_calls_json {
                    if let Ok(tcs) = serde_json::from_str::<Vec<ToolCall>>(tc_json) {
                        let tc_val: Vec<Value> = tcs
                            .iter()
                            .map(|tc| {
                                let mut val = json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments
                                    }
                                });
                                // Preserve extra_content (Gemini 3 thought signatures, etc.)
                                if let Some(ref extra) = tc.extra_content {
                                    val["extra_content"] = extra.clone();
                                }
                                val
                            })
                            .collect();
                        m["tool_calls"] = json!(tc_val);
                        // OpenAI requires content to be null if tool_calls present
                        if msg.content.is_none() {
                            m["content"] = Value::Null;
                        }
                    }
                }
                if let Some(ref tid) = msg.tool_call_id {
                    m["tool_call_id"] = json!(tid);
                }
                if let Some(ref tname) = msg.tool_name {
                    m["name"] = json!(tname);
                }
                messages.push(m);
            }

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
