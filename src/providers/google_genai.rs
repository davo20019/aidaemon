use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info};

use crate::providers::ProviderError;
use crate::traits::{ModelProvider, ProviderResponse, ToolCall};

pub struct GoogleGenAiProvider {
    client: Client,
    base_url: String,
    api_key: String,
}

impl GoogleGenAiProvider {
    pub fn new(api_key: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");
        Self {
            client,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            api_key: api_key.to_string(),
        }
    }

    /// Convert OpenAI-format messages to Gemini "contents" + "system_instruction"
    fn convert_messages(&self, messages: &[Value]) -> (Option<Value>, Vec<Value>) {
        let mut system_instruction: Option<Value> = None;
        let mut contents = Vec::new();

        for msg in messages {
            let role = msg["role"].as_str().unwrap_or("user");
            
            match role {
                "system" => {
                    // Gemini supports top-level system_instruction (for 1.5+ models)
                    // If multiple system messages exist, we concatenate them (or just take the last one, but concatenation is safer)
                    let text = msg["content"].as_str().unwrap_or("");
                    if let Some(existing) = &mut system_instruction {
                        // Append to existing parts
                        if let Some(parts) = existing["parts"].as_array_mut() {
                            parts.push(json!({"text": text}));
                        }
                    } else {
                        system_instruction = Some(json!({
                            "parts": [{ "text": text }]
                        }));
                    }
                }
                "user" => {
                    let text = msg["content"].as_str().unwrap_or("");
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "text": text }]
                    }));
                }
                "assistant" => {
                    let mut parts = Vec::new();
                    // Text content
                    if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            parts.push(json!({"text": text}));
                        }
                    }
                    // Tool calls
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                        for tc in tool_calls {
                            let function_call = json!({
                                "name": tc["function"]["name"],
                                "args": serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}))
                            });
                            parts.push(json!({ "functionCall": function_call }));
                        }
                    }
                    contents.push(json!({
                        "role": "model",
                        "parts": parts
                    }));
                }
                "tool" => {
                    // Tool response
                    let tool_call_id = msg["tool_call_id"].as_str().unwrap_or("");
                    let tool_name = msg["name"].as_str().unwrap_or(""); // OpenAI puts name here? Or in tool_call?
                    // In our trait Message, we have tool_name. usage in openai_compatible.rs seems to verify this.
                    // But actually OpenAI format for tool response is: { role: tool, tool_call_id: ..., content: ... }
                    // Gemini needs:
                    // { role: "function", parts: [{ "functionResponse": { "name": ..., "response": ... } }] }
                    // Note: Gemini API requires previous message to be "model" with "functionCall".
                    
                    let content_str = msg["content"].as_str().unwrap_or("");
                    let response_json = serde_json::from_str::<Value>(content_str).unwrap_or(json!({ "result": content_str }));

                    // Hack: We need the tool name. OpenAI format msg has it sometimes?
                    // Our internal Message struct has it. The `messages` passed here are straight JSON from agent.rs.
                    // Checking agent.rs:
                    // if let Some(ref tname) = msg.tool_name { m["name"] = json!(tname); }
                    // So yes, "name" should be present.

                    let name = msg.get("name").and_then(|n| n.as_str()).unwrap_or("unknown_tool");

                    contents.push(json!({
                        "role": "function", // API expects "function" role or just parts with functionResponse?
                        // Actually documentation says role should be 'function' (v1beta).
                        // wait, checking recent docs... v1beta 'role' can be 'user' or 'model'. Function responses are usually 'function'? 
                        // Actually, for generateContent, it's:
                        // role: "function" is NOT standard. It's usually "user" or "model" or implicitly handled?
                        // "role": "function" was used in PaLM. Gemini uses "function_response" part in a "user" (or separate) role?
                        // Docs: "The role of the author... 'user' or 'model'."
                        // Checks: https://ai.google.dev/gemini-api/docs/function-calling
                        // "Send a message... with the function response... role='function'"
                        // So 'function' IS a valid role.
                        "parts": [{
                            "functionResponse": {
                                "name": name,
                                "response": response_json
                            }
                        }]
                    }));
                }
                _ => {}
            }
        }
        (system_instruction, contents)
    }

    /// Convert OpenAI tool definitions to Gemini "tools"
    fn convert_tools(&self, tools: &[Value]) -> Option<Vec<Value>> {
        if tools.is_empty() {
            return None;
        }
        let mut function_declarations = Vec::new();
        for tool in tools {
            if let Some(func) = tool.get("function") {
                function_declarations.push(json!({
                    "name": func["name"],
                    "description": func.get("description").unwrap_or(&json!("")),
                    "parameters": func["parameters"]
                }));
            }
        }
        Some(vec![json!({ "function_declarations": function_declarations })])
    }
}

#[async_trait]
impl ModelProvider for GoogleGenAiProvider {
    async fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
    ) -> anyhow::Result<ProviderResponse> {
        let (system_instruction, contents) = self.convert_messages(messages);
        let gemini_tools = self.convert_tools(tools);

        let mut body = json!({
            "contents": contents,
        });

        if let Some(sys) = system_instruction {
            body["system_instruction"] = sys;
        }
        if let Some(gt) = gemini_tools {
            body["tools"] = json!(gt);
        }

        // Endpoint: models/{model}:generateContent
        // Model name might be "gemini-1.5-flash". 
        // If user passes full "google/gemini..." we might need to strip prefix, but AgentZero seems to pass raw names from config.
        let url = format!("{}/models/{}:generateContent?key={}", self.base_url, model, self.api_key);
        
        info!(model, url_prefix = %self.base_url, "Calling Google GenAI");

        let resp = match self.client.post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                error!("Google GenAI HTTP request failed: {}", e);
                return Err(ProviderError::network(&e).into());
            }
        };

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            error!(status = %status, "Google GenAI API error: {}", text);
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        let data: Value = serde_json::from_str(&text)?;
        
        // Parse response
        // { "candidates": [ { "content": { "parts": [ ... ], "role": "model" }, ... } ] }
        let candidate = data["candidates"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("No candidates in Google GenAI response: {}", text))?;
        
        let content_parts = candidate["content"]["parts"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("No content parts"))?;

        let mut final_text = String::new();
        let mut tool_calls = Vec::new();

        for part in content_parts {
            if let Some(text) = part.get("text").and_then(|s| s.as_str()) {
                final_text.push_str(text);
            }
            if let Some(fc) = part.get("functionCall") {
                let name = fc["name"].as_str().unwrap_or("").to_string();
                let args = fc["args"].clone(); // already JSON object
                let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                
                tool_calls.push(ToolCall {
                    id: format!("call_{}", uuid::Uuid::new_v4()), // Gemini doesn't give tool call IDs? Re-check... usually implicit order.
                    name,
                    arguments: args_str,
                    extra_content: None,
                });
            }
        }

        Ok(ProviderResponse {
            content: if final_text.is_empty() { None } else { Some(final_text) },
            tool_calls,
        })
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let url = format!("{}/models?key={}&page_size=50", self.base_url, self.api_key);
        let resp = self.client.get(&url).send().await?;
        
        if !resp.status().is_success() {
            anyhow::bail!("Failed to list models: {}", resp.status());
        }
        
        let data: Value = resp.json().await?;
        let models = data["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        m["name"]
                            .as_str()
                            .map(|s| s.trim_start_matches("models/").to_string())
                    })
                    .collect()
            })
            .unwrap_or_default();
            
        Ok(models)
    }
}
