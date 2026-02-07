use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{error, info};

use crate::providers::ProviderError;
use crate::traits::{ModelProvider, ProviderResponse, TokenUsage, ToolCall};

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
                            // Merge extra fields (thought_signature etc.) as siblings of functionCall in the part
                            let mut part_obj = json!({ "functionCall": function_call });
                            if let Some(extra) = tc.get("extra_content").and_then(|e| e.as_object()) {
                                if let Some(part_map) = part_obj.as_object_mut() {
                                    for (k, v) in extra {
                                        part_map.insert(k.clone(), v.clone());
                                    }
                                }

                            }
                            parts.push(part_obj);
                        }
                    }
                    contents.push(json!({
                        "role": "model",
                        "parts": parts
                    }));
                }
                "tool" => {
                    // Tool response
                    let _tool_call_id = msg["tool_call_id"].as_str().unwrap_or("");
                    let _tool_name = msg["name"].as_str().unwrap_or(""); // OpenAI puts name here? Or in tool_call?
                    // In our trait Message, we have tool_name. usage in openai_compatible.rs seems to verify this.
                    // But actually OpenAI format for tool response is: { role: tool, tool_call_id: ..., content: ... }
                    // Gemini needs:
                    // { role: "function", parts: [{ "functionResponse": { "name": ..., "response": ... } }] }
                    // Note: Gemini API requires previous message to be "model" with "functionCall".
                    
                    let content_str = msg["content"].as_str().unwrap_or("");
                    // Gemini's functionResponse.response maps to protobuf Struct,
                    // which must be a JSON object — never an array or primitive.
                    let response_json = match serde_json::from_str::<Value>(content_str) {
                        Ok(Value::Object(obj)) => Value::Object(obj),
                        Ok(other) => json!({ "result": other }),
                        Err(_) => json!({ "result": content_str }),
                    };

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

    /// Convert OpenAI tool definitions to Gemini "tools".
    /// When `include_grounding` is true, appends Google Search grounding.
    /// Expose convert_tools for integration tests.
    #[cfg(test)]
    pub fn convert_tools_for_test(&self, tools: &[Value], include_grounding: bool) -> Option<Vec<Value>> {
        self.convert_tools(tools, include_grounding)
    }

    fn convert_tools(&self, tools: &[Value], include_grounding: bool) -> Option<Vec<Value>> {
        let mut gemini_tools = Vec::new();

        if !tools.is_empty() {
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
            gemini_tools.push(json!({ "function_declarations": function_declarations }));
        }

        if include_grounding {
            gemini_tools.push(json!({ "google_search": {} }));
        }

        if gemini_tools.is_empty() {
            None
        } else {
            Some(gemini_tools)
        }
    }

    /// Parse a Gemini generateContent response into a ProviderResponse.
    fn parse_response(&self, data: &Value, model: &str) -> anyhow::Result<ProviderResponse> {
        let candidate = data["candidates"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("No candidates in Google GenAI response: {}", data))?;

        let empty_parts = vec![];
        let content_parts = candidate["content"]["parts"]
            .as_array()
            .unwrap_or(&empty_parts);

        let mut final_text = String::new();
        let mut tool_calls = Vec::new();

        for part in content_parts {
            if let Some(text) = part.get("text").and_then(|s| s.as_str()) {
                final_text.push_str(text);
            }
            if let Some(fc) = part.get("functionCall") {
                let name = fc["name"].as_str().unwrap_or("").to_string();
                let args = fc["args"].clone();
                let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());

                let mut extra = serde_json::Map::new();
                if let Some(obj) = part.as_object() {
                    for (k, v) in obj {
                        if k != "functionCall" {
                            extra.insert(k.clone(), v.clone());
                        }
                    }
                }

                let extra_content = if extra.is_empty() { None } else { Some(Value::Object(extra)) };

                tool_calls.push(ToolCall {
                    id: format!("call_{}", uuid::Uuid::new_v4()),
                    name,
                    arguments: args_str,
                    extra_content,
                });
            }
        }

        // Extract grounding sources from Google Search
        if let Some(metadata) = candidate.get("groundingMetadata") {
            if let Some(chunks) = metadata.get("groundingChunks").and_then(|c| c.as_array()) {
                let sources: Vec<String> = chunks
                    .iter()
                    .filter_map(|chunk| {
                        let web = chunk.get("web")?;
                        let uri = web.get("uri")?.as_str()?;
                        let title = web.get("title").and_then(|t| t.as_str()).unwrap_or(uri);
                        Some(format!("- [{}]({})", title, uri))
                    })
                    .collect();
                if !sources.is_empty() {
                    final_text.push_str("\n\nSources:\n");
                    final_text.push_str(&sources.join("\n"));
                }
            }
        }

        let usage = data.get("usageMetadata").and_then(|u| {
            Some(TokenUsage {
                input_tokens: u.get("promptTokenCount")?.as_u64()? as u32,
                output_tokens: u.get("candidatesTokenCount")?.as_u64()? as u32,
                model: model.to_string(),
            })
        });

        Ok(ProviderResponse {
            content: if final_text.is_empty() { None } else { Some(final_text) },
            tool_calls,
            usage,
        })
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

        // Include google_search grounding ONLY when no function-calling tools are
        // present. When both are sent together the model may route through grounding
        // instead of calling the user's function tools, producing hallucinated text
        // responses instead of tool calls.
        let has_function_tools = !tools.is_empty();
        let include_grounding = !has_function_tools;
        let gemini_tools = self.convert_tools(tools, include_grounding);

        let mut body = json!({
            "contents": contents,
        });

        if let Some(ref sys) = system_instruction {
            body["system_instruction"] = sys.clone();
        }
        if let Some(ref gt) = gemini_tools {
            body["tools"] = json!(gt);
        }

        // Use header-based authentication instead of URL query parameter
        // to avoid API key exposure in logs, proxies, and error messages
        let url = format!("{}/models/{}:generateContent", self.base_url, model);

        info!(model, url_prefix = %self.base_url, "Calling Google GenAI");

        let resp = match self.client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
        {
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
        self.parse_response(&data, model)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        // Use header-based authentication instead of URL query parameter
        let url = format!("{}/models?page_size=50", self.base_url);
        let resp = self.client
            .get(&url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        
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
