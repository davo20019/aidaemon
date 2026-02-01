use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info};

use crate::providers::ProviderError;
use crate::traits::{ModelProvider, ProviderResponse, ToolCall};

pub struct AnthropicNativeProvider {
    client: Client,
    base_url: String,
    api_key: String,
}

impl AnthropicNativeProvider {
    pub fn new(api_key: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");
        Self {
            client,
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_key: api_key.to_string(),
        }
    }

    /// Convert OpenAI-format messages to Anthropic format
    fn convert_messages(&self, messages: &[Value]) -> (Option<String>, Vec<Value>) {
        let mut system_prompt: Option<String> = None;
        let mut anthropic_msgs = Vec::new();

        for msg in messages {
            let role = msg["role"].as_str().unwrap_or("user");
            match role {
                "system" => {
                    let text = msg["content"].as_str().unwrap_or("").to_string();
                    if let Some(ref mut existing) = system_prompt {
                        existing.push_str("\n\n");
                        existing.push_str(&text);
                    } else {
                        system_prompt = Some(text);
                    }
                }
                "user" => {
                    let content_str = msg["content"].as_str().unwrap_or("");
                    // Anthropic user message content is string or array of blocks
                    anthropic_msgs.push(json!({
                        "role": "user",
                        "content": content_str
                    }));
                }
                "assistant" => {
                    // Split into text content and tool_use blocks
                    let mut content_blocks = Vec::new();
                    
                    if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            content_blocks.push(json!({
                                "type": "text",
                                "text": text
                            }));
                        }
                    }
                    
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                        for tc in tool_calls {
                            let name = tc["function"]["name"].as_str().unwrap_or("");
                            let id = tc["id"].as_str().unwrap_or("");
                            let input: Value = serde_json::from_str(
                                tc["function"]["arguments"].as_str().unwrap_or("{}")
                            ).unwrap_or(json!({}));
                            
                            content_blocks.push(json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input
                            }));
                        }
                    }

                    if !content_blocks.is_empty() {
                        anthropic_msgs.push(json!({
                            "role": "assistant",
                            "content": content_blocks
                        }));
                    }
                }
                "tool" => {
                    // Tool result
                    let tool_use_id = msg["tool_call_id"].as_str().unwrap_or("");
                    let content_str = msg["content"].as_str().unwrap_or("");
                    
                    anthropic_msgs.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": content_str
                        }]
                    }));
                }
                _ => {}
            }
        }
        
        // Anthropic requires messages to alternate User/Assistant. 
        // Our simple conversion might produce Use -> Tool (User) -> Tool (User). 
        // We might need to merge consecutive user messages if they include tool results?
        // Actually, valid tool use flow: 
        // User -> Assistant (ToolUse) -> User (ToolResult) -> Assistant...
        // AgentZero loop creates distinct messages.
        // But if multiple tool results come back, they are separate messages in AgentZero state.
        // Anthropic expects: User (ToolResult 1), User (ToolResult 2)... 
        // Wait, Anthropic usually allows multiple tool results in one message block or sequential user messages?
        // Actually, "The messages API... requires user and assistant roles to alternate."
        // So User (ToolResult 1) followed by User (ToolResult 2) is INVALID.
        // We MUST merge adjacent messages of the same role.
        
        let merged_msgs = self.merge_adjacent_roles(anthropic_msgs);
        (system_prompt, merged_msgs)
    }

    fn merge_adjacent_roles(&self, msgs: Vec<Value>) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();
        
        for msg in msgs {
            if let Some(last) = result.last_mut() {
                if last["role"] == msg["role"] {
                    // Merge content
                    // Ensure both are arrays of blocks (normalize string content to block first)
                    Self::normalize_content_to_blocks(last);
                    let mut new_blocks = Self::msg_content_to_blocks(&msg);
                    
                    last["content"].as_array_mut().unwrap().append(&mut new_blocks);
                    continue;
                }
            }
            // Push new, but normalize to blocks if it's a tool result usage (consistent structure)
            // primarily for cleaner merging later if needed.
            let mut new_msg = msg.clone();
            // If it's a tool result, it's already blocks. If it's simple text, leave as string? 
            // Better to normalize everything to blocks if we are doing merging.
            Self::normalize_content_to_blocks(&mut new_msg);
            result.push(new_msg);
        }
        result
    }

    fn normalize_content_to_blocks(msg: &mut Value) {
        if let Some(content_str) = msg["content"].as_str() {
            let text = content_str.to_string();
            msg["content"] = json!([{ "type": "text", "text": text }]);
        }
        // If it's already array, do nothing
    }

    fn msg_content_to_blocks(msg: &Value) -> Vec<Value> {
        if let Some(content_str) = msg["content"].as_str() {
            vec![json!({ "type": "text", "text": content_str })]
        } else if let Some(arr) = msg["content"].as_array() {
            arr.clone()
        } else {
            vec![]
        }
    }

    /// Convert OpenAI tool definitions to Anthropic tools
    fn convert_tools(&self, tools: &[Value]) -> Option<Vec<Value>> {
        if tools.is_empty() {
             return None;
        }
        let mut anthropic_tools = Vec::new();
        for tool in tools {
            if let Some(func) = tool.get("function") {
                anthropic_tools.push(json!({
                    "name": func["name"],
                    "description": func.get("description").unwrap_or(&json!("")),
                    "input_schema": func["parameters"]
                }));
            }
        }
        Some(anthropic_tools)
    }
}

#[async_trait]
impl ModelProvider for AnthropicNativeProvider {
    async fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
    ) -> anyhow::Result<ProviderResponse> {
        let (system, converted_msgs) = self.convert_messages(messages);
        let anthropic_tools = self.convert_tools(tools);

        let mut body = json!({
            "model": model,
            "max_tokens": 4096,
            "messages": converted_msgs,
        });

        if let Some(sys) = system {
            body["system"] = json!(sys);
        }
        if let Some(at) = anthropic_tools {
            body["tools"] = json!(at);
        }

        info!(model, url = %self.base_url, "Calling Anthropic Native");

        let resp = match self.client.post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await 
        {
            Ok(r) => r,
            Err(e) => {
                error!("Anthropic HTTP request failed: {}", e);
                return Err(ProviderError::network(&e).into());
            }
        };

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            error!(status = %status, "Anthropic API error: {}", text);
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        let data: Value = serde_json::from_str(&text)?;
        
        let mut final_text = String::new();
        let mut tool_calls = Vec::new();

        if let Some(content_arr) = data["content"].as_array() {
            for block in content_arr {
                let btype = block["type"].as_str().unwrap_or("");
                if btype == "text" {
                    if let Some(t) = block["text"].as_str() {
                        final_text.push_str(t);
                    }
                } else if btype == "tool_use" {
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    let id = block["id"].as_str().unwrap_or("").to_string();
                    let input = &block["input"];
                    let args = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                    
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: args,
                        extra_content: None,
                    });
                }
            }
        }

        Ok(ProviderResponse {
            content: if final_text.is_empty() { None } else { Some(final_text) },
            tool_calls,
        })
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        // Anthropic doesn't have a public list_models endpoint yet (AFAIK).
        // return a hardcoded list of known models
        Ok(vec![
            "claude-3-5-sonnet-20240620".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ])
    }
}
