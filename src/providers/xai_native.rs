use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};
use zeroize::Zeroize;

use crate::providers::{OpenAiCompatibleProvider, ProviderError};
use crate::traits::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
    ToolChoiceMode,
};

pub struct XaiNativeProvider {
    client: Client,
    base_url: String,
    api_key: String,
    extra_headers: HashMap<String, String>,
    max_tokens: Option<u32>,
}

/// Validate the base URL for security.
/// - HTTPS is required for remote URLs to protect API keys in transit
/// - HTTP is allowed only for localhost/127.0.0.1 (local proxies)
fn validate_base_url(base_url: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(base_url)
        .map_err(|e| format!("Invalid base_url '{}': {}", base_url, e))?;

    let scheme = parsed.scheme();
    let host = parsed.host_str().unwrap_or("");

    match scheme {
        "https" => Ok(()),
        "http" => {
            let is_localhost =
                host == "localhost" || host == "127.0.0.1" || host == "[::1]" || host == "::1";
            if is_localhost {
                warn!(
                    "Using unencrypted HTTP for local proxy/server at '{}'. \
                     API key will be transmitted in cleartext.",
                    base_url
                );
                Ok(())
            } else {
                Err(format!(
                    "HTTP is not allowed for remote URLs (base_url: '{}'). \
                     Use HTTPS to protect your API key in transit. \
                     HTTP is only permitted for localhost.",
                    base_url
                ))
            }
        }
        _ => Err(format!(
            "Unsupported URL scheme '{}' in base_url '{}'. Only http and https are allowed.",
            scheme, base_url
        )),
    }
}

impl Drop for XaiNativeProvider {
    fn drop(&mut self) {
        self.api_key.zeroize();
    }
}

impl XaiNativeProvider {
    pub fn new_with_options(
        api_key: &str,
        base_url: Option<&str>,
        max_tokens: Option<u32>,
        extra_headers: Option<HashMap<String, String>>,
    ) -> Result<Self, String> {
        let resolved_base_url = base_url.unwrap_or("https://api.x.ai/v1");
        validate_base_url(resolved_base_url)?;

        let client = crate::providers::build_http_client(Duration::from_secs(120))?;
        let normalized_base_url = resolved_base_url.trim_end_matches('/').to_string();

        Ok(Self {
            client,
            base_url: normalized_base_url,
            api_key: api_key.to_string(),
            extra_headers: extra_headers.unwrap_or_default(),
            max_tokens,
        })
    }

    fn with_auth_headers(&self, mut request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        request = request.header("Authorization", format!("Bearer {}", self.api_key));
        for (k, v) in &self.extra_headers {
            request = request.header(k, v);
        }
        request
    }

    fn normalize_tool_name(name: &str) -> String {
        name.trim().to_string()
    }

    fn content_to_text(content: &Value) -> Option<String> {
        if let Some(s) = content.as_str() {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return None;
            }
            return Some(trimmed.to_string());
        }

        let mut parts = Vec::new();
        if let Some(arr) = content.as_array() {
            for part in arr {
                let Some(obj) = part.as_object() else {
                    continue;
                };
                if let Some(text) = obj.get("text").and_then(Value::as_str) {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        parts.push(trimmed.to_string());
                    }
                }
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n"))
        }
    }

    fn parse_arguments_string(value: &Value) -> String {
        if let Some(s) = value.as_str() {
            return s.to_string();
        }
        if value.is_null() {
            return "{}".to_string();
        }
        serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
    }

    fn parse_tool_call(value: &Value) -> Option<ToolCall> {
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| value.get("function")?.get("name").and_then(Value::as_str))?;

        let id = value
            .get("call_id")
            .and_then(Value::as_str)
            .or_else(|| value.get("id").and_then(Value::as_str))
            .unwrap_or("")
            .to_string();

        let arguments = value
            .get("arguments")
            .or_else(|| value.get("function").and_then(|f| f.get("arguments")))
            .map(Self::parse_arguments_string)
            .unwrap_or_else(|| "{}".to_string());

        Some(ToolCall {
            id,
            name: Self::normalize_tool_name(name),
            arguments,
            extra_content: None,
        })
    }

    fn convert_tools(tools: &[Value]) -> Vec<Value> {
        let mut out = Vec::new();
        for tool in tools {
            let Some(func) = tool.get("function") else {
                continue;
            };
            out.push(json!({
                "type": "function",
                "name": func.get("name").and_then(Value::as_str).unwrap_or(""),
                "description": func.get("description").and_then(Value::as_str).unwrap_or(""),
                "parameters": func.get("parameters").cloned().unwrap_or_else(|| json!({}))
            }));
        }
        out
    }

    fn convert_messages(messages: &[Value]) -> Vec<Value> {
        let mut input = Vec::new();
        for msg in messages {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
            match role {
                "system" | "user" | "assistant" => {
                    if let Some(text) = msg.get("content").and_then(Self::content_to_text) {
                        input.push(json!({
                            "role": role,
                            "content": text
                        }));
                    }

                    if role == "assistant" {
                        if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
                            for tc in tool_calls {
                                let call_id = tc
                                    .get("id")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();
                                let name = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();
                                let arguments = tc
                                    .get("function")
                                    .and_then(|f| f.get("arguments"))
                                    .map(Self::parse_arguments_string)
                                    .unwrap_or_else(|| "{}".to_string());

                                input.push(json!({
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": name,
                                    "arguments": arguments
                                }));
                            }
                        }
                    }
                }
                "tool" => {
                    let call_id = msg
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let output = msg
                        .get("content")
                        .and_then(Self::content_to_text)
                        .unwrap_or_default();

                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output
                    }));
                }
                _ => {}
            }
        }
        input
    }

    fn build_responses_body(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> Value {
        let input = Self::convert_messages(messages);
        let converted_tools = Self::convert_tools(tools);

        let mut body = json!({
            "model": model,
            "input": input
        });

        if let Some(max_tokens) = self.max_tokens {
            body["max_output_tokens"] = json!(max_tokens);
        }

        if !converted_tools.is_empty() {
            body["tools"] = json!(converted_tools);
            match &options.tool_choice {
                ToolChoiceMode::Auto => {}
                ToolChoiceMode::None => body["tool_choice"] = json!("none"),
                ToolChoiceMode::Required => body["tool_choice"] = json!("required"),
                ToolChoiceMode::Specific(name) => {
                    body["tool_choice"] = json!({
                        "type": "function",
                        "name": name
                    });
                }
            }
        } else if !matches!(options.tool_choice, ToolChoiceMode::Auto) {
            warn!(
                tool_choice = ?options.tool_choice,
                "Ignoring non-auto tool_choice because no tools were provided"
            );
        }

        body
    }

    fn parse_usage(data: &Value, model: &str) -> Option<TokenUsage> {
        data.get("usage").and_then(|u| {
            let input_tokens = u
                .get("input_tokens")
                .and_then(Value::as_u64)
                .or_else(|| u.get("prompt_tokens").and_then(Value::as_u64))?;
            let output_tokens = u
                .get("output_tokens")
                .and_then(Value::as_u64)
                .or_else(|| u.get("completion_tokens").and_then(Value::as_u64))?;
            Some(TokenUsage {
                input_tokens: input_tokens as u32,
                output_tokens: output_tokens as u32,
                model: model.to_string(),
            })
        })
    }

    fn parse_responses_payload(data: &Value, model: &str) -> anyhow::Result<ProviderResponse> {
        let mut text_chunks = Vec::new();
        let mut tool_calls = Vec::new();

        if let Some(output_text) = data.get("output_text").and_then(Value::as_str) {
            if !output_text.trim().is_empty() {
                text_chunks.push(output_text.trim().to_string());
            }
        }

        if let Some(output) = data.get("output").and_then(Value::as_array) {
            for item in output {
                let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                match item_type {
                    "function_call" | "tool_call" => {
                        if let Some(tc) = Self::parse_tool_call(item) {
                            tool_calls.push(tc);
                        }
                    }
                    "message" => {
                        if let Some(content) = item.get("content").and_then(Value::as_array) {
                            for part in content {
                                let part_type =
                                    part.get("type").and_then(Value::as_str).unwrap_or("");
                                match part_type {
                                    "output_text" | "input_text" | "text" => {
                                        if let Some(text) = part.get("text").and_then(Value::as_str)
                                        {
                                            let trimmed = text.trim();
                                            if !trimmed.is_empty() {
                                                text_chunks.push(trimmed.to_string());
                                            }
                                        }
                                    }
                                    "function_call" | "tool_call" => {
                                        if let Some(tc) = Self::parse_tool_call(part) {
                                            tool_calls.push(tc);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if text_chunks.is_empty() && tool_calls.is_empty() {
            return Err(ProviderError::malformed_shape(
                "Malformed response from xAI native provider (missing output text and tool calls)",
            )
            .into());
        }

        let usage = Self::parse_usage(data, model);
        let content = if text_chunks.is_empty() {
            None
        } else {
            Some(text_chunks.join("\n"))
        };

        Ok(ProviderResponse {
            content,
            tool_calls,
            usage,
            thinking: None,
            response_note: None,
        })
    }

    fn parse_chat_completions_payload(
        data: &Value,
        model: &str,
    ) -> anyhow::Result<ProviderResponse> {
        let choice = data.get("choices").and_then(|c| c.get(0)).ok_or_else(|| {
            ProviderError::malformed_shape(
                "Malformed response from xAI provider fallback (missing choices[0])",
            )
        })?;
        let message = choice.get("message").ok_or_else(|| {
            ProviderError::malformed_shape(
                "Malformed response from xAI provider fallback (missing choices[0].message)",
            )
        })?;

        let content = message
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string);

        let mut tool_calls = Vec::new();
        if let Some(raw_tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
            for tc in raw_tool_calls {
                if let Some(parsed) = Self::parse_tool_call(tc) {
                    tool_calls.push(parsed);
                }
            }
        }

        let usage = Self::parse_usage(data, model);
        Ok(ProviderResponse {
            content,
            tool_calls,
            usage,
            thinking: None,
            response_note: None,
        })
    }

    fn parse_provider_payload(text: &str, model: &str) -> anyhow::Result<ProviderResponse> {
        let data: Value = serde_json::from_str(text).map_err(|e| {
            ProviderError::malformed_parse(format!(
                "Malformed response from xAI provider (JSON parse error: {})",
                e
            ))
        })?;

        if data.get("output").is_some() || data.get("output_text").is_some() {
            return Self::parse_responses_payload(&data, model);
        }

        if data.get("choices").is_some() {
            return Self::parse_chat_completions_payload(&data, model);
        }

        Err(ProviderError::malformed_shape(
            "Malformed response from xAI provider (unknown response shape)",
        )
        .into())
    }
}

#[async_trait]
impl ModelProvider for XaiNativeProvider {
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
        if !matches!(options.response_mode, ResponseMode::Text) {
            warn!(
                response_mode = ?options.response_mode,
                "xAI native responses endpoint does not yet map response_mode directly; using chat/completions fallback"
            );
            let fallback = OpenAiCompatibleProvider::new_with_all_options(
                &self.base_url,
                &self.api_key,
                None,
                Some(self.extra_headers.clone()),
                self.max_tokens,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            return fallback
                .chat_with_options(model, messages, tools, options)
                .await;
        }

        let body = self.build_responses_body(model, messages, tools, options);
        let url = format!("{}/responses", self.base_url);
        info!(
            model,
            url = %url,
            tools = tools.len(),
            response_mode = ?options.response_mode,
            tool_choice = ?options.tool_choice,
            "Calling xAI Responses API"
        );

        let request = self
            .with_auth_headers(self.client.post(&url))
            .header("Content-Type", "application/json")
            .json(&body);
        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                error!("xAI HTTP request failed: {}", e);
                return Err(ProviderError::network(&e).into());
            }
        };

        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            error!("Failed to read xAI response body: {}", e);
            ProviderError::network(&e)
        })?;

        if !status.is_success() {
            error!(status = %status, "xAI API error: {}", text);
            debug!(
                "xAI failed request body: {}",
                serde_json::to_string(&body).unwrap_or_default()
            );
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        let truncated = if text.len() > 2000 {
            let mut end = 2000;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            &text[..end]
        } else {
            &text
        };
        debug!("xAI response: {}", truncated);

        Self::parse_provider_payload(&text, model)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let resp = self.with_auth_headers(self.client.get(&url)).send().await?;
        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            anyhow::bail!("Failed to list models at '{}' ({}): {}", url, status, text);
        }

        let data: Value = serde_json::from_str(&text)?;
        let models = data
            .get("data")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_messages_includes_tool_results() {
        let messages = vec![
            json!({"role":"system","content":"You are helpful."}),
            json!({"role":"user","content":"find files"}),
            json!({"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"name":"search_files","arguments":"{\"q\":\"Cargo.toml\"}"}}]}),
            json!({"role":"tool","tool_call_id":"call_1","content":"[\"Cargo.toml\"]"}),
        ];

        let input = XaiNativeProvider::convert_messages(&messages);
        assert!(input
            .iter()
            .any(|v| v.get("type").and_then(Value::as_str) == Some("function_call")));
        assert!(input
            .iter()
            .any(|v| v.get("type").and_then(Value::as_str) == Some("function_call_output")));
    }

    #[test]
    fn parse_responses_payload_extracts_text_and_tool_call() {
        let payload = json!({
            "id": "resp_1",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "I will search now."}
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_42",
                    "name": "search_files",
                    "arguments": "{\"q\":\"README\"}"
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 8}
        });

        let parsed = XaiNativeProvider::parse_responses_payload(&payload, "grok-4").unwrap();
        assert_eq!(parsed.content.as_deref(), Some("I will search now."));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "call_42");
        assert_eq!(parsed.tool_calls[0].name, "search_files");
        assert_eq!(parsed.usage.as_ref().unwrap().input_tokens, 12);
        assert_eq!(parsed.usage.as_ref().unwrap().output_tokens, 8);
    }

    #[test]
    fn parse_chat_completions_payload_supported_for_fallback() {
        let payload = json!({
            "choices": [{
                "message": {
                    "content": "Fallback text"
                }
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5}
        });

        let parsed = XaiNativeProvider::parse_chat_completions_payload(&payload, "grok-4").unwrap();
        assert_eq!(parsed.content.as_deref(), Some("Fallback text"));
        assert_eq!(parsed.usage.as_ref().unwrap().input_tokens, 3);
        assert_eq!(parsed.usage.as_ref().unwrap().output_tokens, 5);
    }
}
