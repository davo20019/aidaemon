use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{error, info, warn};
use zeroize::Zeroize;

use crate::providers::ProviderError;
use crate::traits::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
    ToolChoiceMode,
};

const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 16_384;

pub struct AnthropicNativeProvider {
    client: Client,
    base_url: String,
    api_key: String,
    max_tokens: u32,
    extra_headers: HashMap<String, String>,
}

fn normalize_tool_name(name: &str) -> String {
    name.trim().to_string()
}

impl Drop for AnthropicNativeProvider {
    fn drop(&mut self) {
        self.api_key.zeroize();
    }
}

impl AnthropicNativeProvider {
    pub fn new_with_options(
        api_key: &str,
        base_url: Option<&str>,
        max_tokens: Option<u32>,
        extra_headers: Option<HashMap<String, String>>,
    ) -> Self {
        let client = crate::providers::build_http_client(Duration::from_secs(120))
            .unwrap_or_else(|e| panic!("failed to build HTTP client: {e}"));
        let normalized_base_url = base_url
            .unwrap_or("https://api.anthropic.com/v1")
            .trim_end_matches('/')
            .to_string();
        Self {
            client,
            base_url: normalized_base_url,
            api_key: api_key.to_string(),
            max_tokens: max_tokens
                .filter(|v| *v > 0)
                .unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS),
            extra_headers: extra_headers.unwrap_or_default(),
        }
    }

    fn with_extra_headers(&self, mut request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (k, v) in &self.extra_headers {
            request = request.header(k, v);
        }
        request
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
                                tc["function"]["arguments"].as_str().unwrap_or("{}"),
                            )
                            .unwrap_or(json!({}));

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

                    last["content"]
                        .as_array_mut()
                        .unwrap()
                        .append(&mut new_blocks);
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

    fn build_request_body(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> Value {
        let (system, converted_msgs) = self.convert_messages(messages);
        let effective_tools: &[Value] = if matches!(options.tool_choice, ToolChoiceMode::None) {
            &[]
        } else {
            tools
        };
        let anthropic_tools = self.convert_tools(effective_tools);

        let mut body = json!({
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": converted_msgs,
        });

        if let Some(sys) = system {
            body["system"] = json!(sys);
        }
        if let Some(at) = anthropic_tools {
            body["tools"] = json!(at);
        }

        if !effective_tools.is_empty() {
            match &options.tool_choice {
                ToolChoiceMode::Auto | ToolChoiceMode::None => {}
                ToolChoiceMode::Required => {
                    body["tool_choice"] = json!({ "type": "any" });
                }
                ToolChoiceMode::Specific(name) => {
                    body["tool_choice"] = json!({
                        "type": "tool",
                        "name": name
                    });
                }
            }
        } else if matches!(
            options.tool_choice,
            ToolChoiceMode::Required | ToolChoiceMode::Specific(_)
        ) {
            warn!(
                tool_choice = ?options.tool_choice,
                "Ignoring required/specific tool_choice because no tools were provided"
            );
        }

        body
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
        self.chat_with_options(model, messages, tools, &ChatOptions::default())
            .await
    }

    async fn chat_with_options(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> anyhow::Result<ProviderResponse> {
        let body = self.build_request_body(model, messages, tools, options);

        if !matches!(options.response_mode, ResponseMode::Text) {
            warn!(
                response_mode = ?options.response_mode,
                "Anthropic native provider does not enforce response_mode; relying on prompt contract"
            );
        }

        info!(
            model,
            url = %self.base_url,
            response_mode = ?options.response_mode,
            tool_choice = ?options.tool_choice,
            "Calling Anthropic Native"
        );

        let request = self
            .with_extra_headers(
                self.client
                    .post(format!("{}/messages", self.base_url))
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json"),
            )
            .json(&body);
        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                error!("Anthropic HTTP request failed: {}", e);
                return Err(ProviderError::network(&e).into());
            }
        };

        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            error!("Failed to read response body: {}", e);
            ProviderError::network(&e)
        })?;

        if !status.is_success() {
            error!(status = %status, "Anthropic API error: {}", text);
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        let data: Value = serde_json::from_str(&text).map_err(|e| {
            error!("Failed to parse Anthropic response JSON: {}", e);
            ProviderError::malformed_parse(format!(
                "Malformed response from LLM provider (JSON parse error: {})",
                e
            ))
        })?;

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
                    let name = normalize_tool_name(block["name"].as_str().unwrap_or(""));
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

        let usage = data.get("usage").and_then(|u| {
            Some(TokenUsage {
                input_tokens: u.get("input_tokens")?.as_u64()? as u32,
                output_tokens: u.get("output_tokens")?.as_u64()? as u32,
                model: model.to_string(),
            })
        });

        Ok(ProviderResponse {
            content: if final_text.is_empty() {
                None
            } else {
                Some(final_text)
            },
            tool_calls,
            usage,
            thinking: None,
            response_note: None,
        })
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let known_models = vec![
            "claude-sonnet-4-5-20250514".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-haiku-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ];

        let url = format!("{}/models", self.base_url);
        let resp = match self
            .with_extra_headers(
                self.client
                    .get(&url)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", "2023-06-01"),
            )
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to fetch Anthropic model list, using known models: {e}");
                return Ok(known_models);
            }
        };

        if !resp.status().is_success() {
            warn!(
                "Anthropic /models returned {}, using known models",
                resp.status()
            );
            return Ok(known_models);
        }

        let data: Value = resp.json().await?;
        let models: Vec<String> = data["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        if models.is_empty() {
            return Ok(known_models);
        }

        Ok(models)
    }
}

#[cfg(test)]
impl AnthropicNativeProvider {
    pub fn new(api_key: &str) -> Self {
        Self::new_with_options(api_key, None, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn provider() -> AnthropicNativeProvider {
        AnthropicNativeProvider::new("test-key")
    }

    #[test]
    fn test_system_message_extracted() {
        let p = provider();
        let messages = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hello"}),
        ];

        let (system, msgs) = p.convert_messages(&messages);
        assert_eq!(system, Some("You are helpful.".to_string()));
        // System message should not appear in the converted messages
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn test_multiple_system_messages_merged() {
        let p = provider();
        let messages = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "system", "content": "Be concise."}),
            json!({"role": "user", "content": "Hello"}),
        ];

        let (system, _msgs) = p.convert_messages(&messages);
        let system_text = system.unwrap();
        assert!(
            system_text.contains("You are helpful."),
            "System prompt should contain first system message"
        );
        assert!(
            system_text.contains("Be concise."),
            "System prompt should contain second system message"
        );
        assert!(
            system_text.contains("\n\n"),
            "System messages should be joined with double newline"
        );
        assert_eq!(system_text, "You are helpful.\n\nBe concise.");
    }

    #[test]
    fn test_assistant_tool_calls_converted() {
        let p = provider();
        let messages = vec![
            json!({"role": "user", "content": "What time is it?"}),
            json!({
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "get_time",
                        "arguments": "{\"timezone\": \"UTC\"}"
                    }
                }]
            }),
        ];

        let (_system, msgs) = p.convert_messages(&messages);
        assert_eq!(msgs.len(), 2);

        // The assistant message should have content blocks
        let assistant_msg = &msgs[1];
        assert_eq!(assistant_msg["role"], "assistant");
        let content = assistant_msg["content"].as_array().unwrap();

        // Should have a text block and a tool_use block
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Let me check.");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["id"], "call_123");
        assert_eq!(content[1]["name"], "get_time");
        assert_eq!(content[1]["input"]["timezone"], "UTC");
    }

    #[test]
    fn test_tool_result_as_user_message() {
        let p = provider();
        let messages = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({
                "role": "assistant",
                "content": "Calling tool.",
                "tool_calls": [{
                    "id": "call_abc",
                    "function": {
                        "name": "my_tool",
                        "arguments": "{}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "tool output here"
            }),
        ];

        let (_system, msgs) = p.convert_messages(&messages);
        assert_eq!(msgs.len(), 3);

        // Tool result should become a user message with tool_result content block
        let tool_msg = &msgs[2];
        assert_eq!(tool_msg["role"], "user");
        let content = tool_msg["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "call_abc");
        assert_eq!(content[0]["content"], "tool output here");
    }

    #[test]
    fn test_consecutive_same_role_merged() {
        let p = provider();
        // Two tool results in a row: both become "user" role messages
        // which must be merged to satisfy Anthropic's alternating role requirement
        let messages = vec![
            json!({"role": "user", "content": "Do two things"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "tool_a", "arguments": "{}"}
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "tool_b", "arguments": "{}"}
                    }
                ]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "result A"
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "result B"
            }),
        ];

        let (_system, msgs) = p.convert_messages(&messages);

        // The two tool results (both user role) should be merged into one user message
        // Expected: user, assistant, user (merged tool results)
        assert_eq!(
            msgs.len(),
            3,
            "Two tool results should be merged into one user message"
        );

        let merged_user = &msgs[2];
        assert_eq!(merged_user["role"], "user");
        let content = merged_user["content"].as_array().unwrap();
        assert_eq!(
            content.len(),
            2,
            "Merged message should have 2 content blocks"
        );
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "call_1");
        assert_eq!(content[1]["type"], "tool_result");
        assert_eq!(content[1]["tool_use_id"], "call_2");
    }

    #[test]
    fn test_empty_content_skipped() {
        let p = provider();
        // Assistant message with empty content and no tool_calls should produce no message
        let messages = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({
                "role": "assistant",
                "content": "",
            }),
        ];

        let (_system, msgs) = p.convert_messages(&messages);

        // The empty assistant message should be skipped (no content blocks)
        assert_eq!(
            msgs.len(),
            1,
            "Empty assistant message should be skipped, got {} messages",
            msgs.len()
        );
        assert_eq!(msgs[0]["role"], "user");
    }

    fn openai_tool(name: &str) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    }
                }
            }
        })
    }

    #[test]
    fn test_build_request_body_required_tool_choice_sets_any() {
        let p = provider();
        let messages = vec![json!({"role": "user", "content": "run a tool"})];
        let tools = vec![openai_tool("search_files")];
        let options = ChatOptions {
            response_mode: ResponseMode::Text,
            tool_choice: ToolChoiceMode::Required,
        };

        let body = p.build_request_body("claude-3-5-sonnet-20241022", &messages, &tools, &options);

        assert_eq!(body["tool_choice"]["type"], "any");
        assert!(body.get("tools").is_some(), "tools should be present");
    }

    #[test]
    fn test_build_request_body_specific_tool_choice_sets_named_tool() {
        let p = provider();
        let messages = vec![json!({"role": "user", "content": "run search_files"})];
        let tools = vec![openai_tool("search_files")];
        let options = ChatOptions {
            response_mode: ResponseMode::Text,
            tool_choice: ToolChoiceMode::Specific("search_files".to_string()),
        };

        let body = p.build_request_body("claude-3-5-sonnet-20241022", &messages, &tools, &options);

        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "search_files");
    }

    #[test]
    fn test_build_request_body_none_tool_choice_strips_tools() {
        let p = provider();
        let messages = vec![json!({"role": "user", "content": "just answer text"})];
        let tools = vec![openai_tool("search_files")];
        let options = ChatOptions {
            response_mode: ResponseMode::Text,
            tool_choice: ToolChoiceMode::None,
        };

        let body = p.build_request_body("claude-3-5-sonnet-20241022", &messages, &tools, &options);

        assert!(body.get("tools").is_none(), "tools should be stripped");
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice should be omitted when tools are stripped"
        );
    }

    #[test]
    fn test_build_request_body_consultant_style_none_with_empty_tools_is_safe() {
        let p = provider();
        let messages = vec![json!({"role": "user", "content": "classify intent"})];
        let tools: Vec<Value> = vec![];
        let options = ChatOptions {
            response_mode: ResponseMode::Text,
            tool_choice: ToolChoiceMode::None,
        };

        let body = p.build_request_body("claude-3-5-sonnet-20241022", &messages, &tools, &options);

        assert!(body.get("tools").is_none(), "tools should be omitted");
        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice should stay omitted when no tools are provided"
        );
    }

    #[test]
    fn test_build_request_body_uses_provider_max_tokens_default() {
        let p = provider();
        let messages = vec![json!({"role": "user", "content": "hello"})];
        let body = p.build_request_body(
            "claude-3-5-sonnet-20241022",
            &messages,
            &[],
            &ChatOptions::default(),
        );
        assert_eq!(body["max_tokens"], DEFAULT_ANTHROPIC_MAX_TOKENS);
    }

    #[test]
    fn test_build_request_body_uses_configured_max_tokens() {
        let p = AnthropicNativeProvider::new_with_options("test-key", None, Some(32768), None);
        let messages = vec![json!({"role": "user", "content": "hello"})];
        let body = p.build_request_body(
            "claude-3-5-sonnet-20241022",
            &messages,
            &[],
            &ChatOptions::default(),
        );
        assert_eq!(body["max_tokens"], 32768);
    }

    #[test]
    fn test_new_with_options_applies_base_url_and_headers() {
        let p = AnthropicNativeProvider::new_with_options(
            "test-key",
            Some("https://example.com/v1/"),
            Some(2048),
            Some(HashMap::from([("x-test".to_string(), "1".to_string())])),
        );
        assert_eq!(p.base_url, "https://example.com/v1");
        assert_eq!(p.max_tokens, 2048);
        assert_eq!(p.extra_headers.get("x-test"), Some(&"1".to_string()));
    }

    #[test]
    fn test_normalize_tool_name_trims_whitespace() {
        assert_eq!(normalize_tool_name(" terminal "), "terminal");
    }
}
