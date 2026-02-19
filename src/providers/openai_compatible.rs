use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};
use zeroize::Zeroize;

use crate::providers::error::ProviderErrorKind;
use crate::providers::ProviderError;
use crate::traits::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
    ToolChoiceMode,
};

pub struct OpenAiCompatibleProvider {
    client: Client,
    base_url: String,
    api_key: String,
    gateway_token: Option<String>,
    is_cloudflare_gateway: bool,
}

impl Drop for OpenAiCompatibleProvider {
    fn drop(&mut self) {
        self.api_key.zeroize();
        if let Some(token) = self.gateway_token.as_mut() {
            token.zeroize();
        }
    }
}

/// Validate the base URL for security.
/// - HTTPS is required for remote URLs to protect API keys in transit
/// - HTTP is allowed only for localhost/127.0.0.1 (local LLM servers)
fn validate_base_url(base_url: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(base_url)
        .map_err(|e| format!("Invalid base_url '{}': {}", base_url, e))?;

    let scheme = parsed.scheme();
    let host = parsed.host_str().unwrap_or("");

    match scheme {
        "https" => Ok(()), // HTTPS is always allowed
        "http" => {
            // HTTP only allowed for localhost
            let is_localhost =
                host == "localhost" || host == "127.0.0.1" || host == "[::1]" || host == "::1";

            if is_localhost {
                warn!(
                    "Using unencrypted HTTP for local LLM server at '{}'. \
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

fn is_cloudflare_ai_gateway_base(base_url: &str) -> bool {
    let parsed = match reqwest::Url::parse(base_url) {
        Ok(url) => url,
        Err(_) => return false,
    };
    matches!(
        parsed.host_str(),
        Some(host) if host.eq_ignore_ascii_case("gateway.ai.cloudflare.com")
    )
}

impl OpenAiCompatibleProvider {
    #[allow(dead_code)]
    pub fn new(base_url: &str, api_key: &str) -> Result<Self, String> {
        Self::new_with_gateway_token(base_url, api_key, None)
    }

    pub fn new_with_gateway_token(
        base_url: &str,
        api_key: &str,
        gateway_token: Option<&str>,
    ) -> Result<Self, String> {
        // Validate URL security before creating provider
        validate_base_url(base_url)?;

        let client = crate::providers::build_http_client(Duration::from_secs(120))?;
        let normalized_base_url = base_url.trim_end_matches('/').to_string();

        Ok(Self {
            client,
            is_cloudflare_gateway: is_cloudflare_ai_gateway_base(&normalized_base_url),
            base_url: normalized_base_url,
            api_key: api_key.to_string(),
            gateway_token: gateway_token.map(|s| s.to_string()),
        })
    }

    fn with_auth_headers(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let request = request.header("Authorization", format!("Bearer {}", self.api_key));
        if let Some(token) = self.gateway_token.as_deref() {
            if token.is_empty() {
                return request;
            }
            request.header("cf-aig-authorization", format!("Bearer {}", token))
        } else {
            request
        }
    }

    fn parse_models_response(text: &str) -> anyhow::Result<Vec<String>> {
        let data: Value = serde_json::from_str(text)?;
        // OpenAI format: { "data": [{ "id": "model-name" }, ...] }
        let models = data["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        Ok(models)
    }

    fn cloudflare_models_fallback_url(&self) -> String {
        if self.base_url.ends_with("/compat") {
            format!("{}/v1/models", self.base_url)
        } else {
            format!("{}/compat/v1/models", self.base_url)
        }
    }

    fn build_request_body(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> Value {
        // Strip extra_content from tool_calls before sending — the OpenAI-compatible
        // endpoint doesn't understand it (it's used internally for Gemini native round-trip).
        let mut messages_cleaned: Vec<Value> = messages.to_vec();
        for msg in &mut messages_cleaned {
            if let Some(tcs) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                for tc in tcs {
                    if let Some(obj) = tc.as_object_mut() {
                        obj.remove("extra_content");
                    }
                }
            }
        }

        let mut body = json!({
            "model": model,
            "messages": messages_cleaned,
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if !tools.is_empty() {
            match &options.tool_choice {
                ToolChoiceMode::Auto => {}
                ToolChoiceMode::None => body["tool_choice"] = json!("none"),
                ToolChoiceMode::Required => body["tool_choice"] = json!("required"),
                ToolChoiceMode::Specific(name) => {
                    body["tool_choice"] = json!({
                        "type": "function",
                        "function": { "name": name }
                    });
                }
            }
        } else if !matches!(options.tool_choice, ToolChoiceMode::Auto) {
            warn!(
                tool_choice = ?options.tool_choice,
                "Ignoring non-auto tool_choice because no tools were provided"
            );
        }

        match &options.response_mode {
            ResponseMode::Text => {}
            ResponseMode::JsonObject => {
                body["response_format"] = json!({ "type": "json_object" });
            }
            ResponseMode::JsonSchema {
                name,
                schema,
                strict,
            } => {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                        "strict": strict
                    }
                });
            }
        }

        body
    }
}

#[async_trait]
impl ModelProvider for OpenAiCompatibleProvider {
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

        let url = format!("{}/chat/completions", self.base_url);
        info!(
            model,
            url = %url,
            tools = tools.len(),
            response_mode = ?options.response_mode,
            tool_choice = ?options.tool_choice,
            "Calling LLM API"
        );

        let request = self
            .with_auth_headers(self.client.post(&url))
            .header("Content-Type", "application/json")
            .json(&body);
        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                error!("HTTP request failed: {}", e);
                return Err(ProviderError::network(&e).into());
            }
        };

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            error!(status = %status, "Provider API error: {}", text);
            debug!(
                "Failed request body: {}",
                serde_json::to_string(&body).unwrap_or_default()
            );
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        // Safely truncate for debug logging, respecting UTF-8 char boundaries
        let truncated = if text.len() > 2000 {
            let mut end = 2000;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            &text[..end]
        } else {
            &text
        };
        debug!("Provider response: {}", truncated);

        let data: Value = serde_json::from_str(&text).map_err(|e| {
            error!("Failed to parse provider response JSON: {}", e);
            ProviderError {
                kind: ProviderErrorKind::ServerError,
                status: Some(200),
                message: format!(
                    "Malformed response from LLM provider (JSON parse error: {})",
                    e
                ),
                retry_after_secs: None,
            }
        })?;
        let choice = data["choices"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;
        let message = &choice["message"];

        let content = message["content"].as_str().map(|s| s.to_string());

        let mut tool_calls = Vec::new();
        if let Some(tcs) = message["tool_calls"].as_array() {
            debug!(
                "Raw tool_calls from provider: {}",
                serde_json::to_string(tcs).unwrap_or_default()
            );
            for tc in tcs {
                let extra_content = tc.get("extra_content").filter(|v| !v.is_null()).cloned();

                tool_calls.push(ToolCall {
                    id: tc["id"].as_str().unwrap_or("").to_string(),
                    name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                    arguments: tc["function"]["arguments"]
                        .as_str()
                        .unwrap_or("{}")
                        .to_string(),
                    extra_content,
                });
            }
        }

        let usage = data.get("usage").and_then(|u| {
            Some(TokenUsage {
                input_tokens: u.get("prompt_tokens")?.as_u64()? as u32,
                output_tokens: u.get("completion_tokens")?.as_u64()? as u32,
                model: model.to_string(),
            })
        });

        Ok(ProviderResponse {
            content,
            tool_calls,
            usage,
            thinking: None,
            response_note: None,
        })
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let primary_url = format!("{}/models", self.base_url);
        let primary_resp = self
            .with_auth_headers(self.client.get(&primary_url))
            .send()
            .await?;

        let primary_status = primary_resp.status();
        let primary_text = primary_resp.text().await?;
        if primary_status.is_success() {
            return Self::parse_models_response(&primary_text);
        }

        let should_try_cf_fallback =
            self.is_cloudflare_gateway && matches!(primary_status.as_u16(), 404 | 405);
        if !should_try_cf_fallback {
            anyhow::bail!(
                "Failed to list models at '{}' ({}): {}",
                primary_url,
                primary_status,
                primary_text
            );
        }

        let fallback_url = self.cloudflare_models_fallback_url();
        let fallback_resp = self
            .with_auth_headers(self.client.get(&fallback_url))
            .send()
            .await?;
        let fallback_status = fallback_resp.status();
        let fallback_text = fallback_resp.text().await?;

        if fallback_status.is_success() {
            return Self::parse_models_response(&fallback_text);
        }

        anyhow::bail!(
            "Failed to list models at '{}' ({}): {}. Fallback '{}' ({}): {}",
            primary_url,
            primary_status,
            primary_text,
            fallback_url,
            fallback_status,
            fallback_text
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_https_accepted() {
        let result = validate_base_url("https://api.openai.com");
        assert!(result.is_ok());
    }

    #[test]
    fn test_http_localhost_accepted() {
        let result = validate_base_url("http://localhost:8080");
        assert!(result.is_ok());
    }

    #[test]
    fn test_http_127_accepted() {
        let result = validate_base_url("http://127.0.0.1:1234");
        assert!(result.is_ok());
    }

    #[test]
    fn test_http_ipv6_localhost_accepted() {
        let result = validate_base_url("http://[::1]:8080");
        assert!(result.is_ok());
    }

    #[test]
    fn test_http_remote_rejected() {
        let result = validate_base_url("http://api.example.com");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("HTTP is not allowed"),
            "Expected HTTP rejection error, got: {}",
            err
        );
    }

    #[test]
    fn test_ftp_rejected() {
        let result = validate_base_url("ftp://example.com");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Unsupported URL scheme"),
            "Expected unsupported scheme error, got: {}",
            err
        );
    }

    #[test]
    fn test_invalid_url_rejected() {
        let result = validate_base_url("not a url");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Invalid base_url"),
            "Expected invalid URL error, got: {}",
            err
        );
    }

    #[test]
    fn test_trailing_slash_trimmed() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1/", "test-key");
        assert!(
            provider.is_ok(),
            "Provider::new should succeed with trailing slash"
        );
        let provider = provider.unwrap();
        assert!(
            !provider.base_url.ends_with('/'),
            "base_url should not end with slash, got: {}",
            provider.base_url
        );
    }

    #[test]
    fn test_build_request_body_applies_required_tool_choice_and_json_schema() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");
        let messages = vec![json!({"role":"user","content":"plan the task"})];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "search project files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    }
                }
            }
        })];
        let options = ChatOptions {
            response_mode: ResponseMode::JsonSchema {
                name: "intent_gate_v1".to_string(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "needs_tools": {"type": "boolean"}
                    },
                    "required": ["needs_tools"],
                    "additionalProperties": false
                }),
                strict: true,
            },
            tool_choice: ToolChoiceMode::Required,
        };

        let body = provider.build_request_body("gpt-4o-mini", &messages, &tools, &options);

        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "intent_gate_v1"
        );
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn test_build_request_body_ignores_non_auto_tool_choice_without_tools() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");
        let messages = vec![json!({"role":"user","content":"answer in json"})];
        let options = ChatOptions {
            response_mode: ResponseMode::JsonObject,
            tool_choice: ToolChoiceMode::Required,
        };

        let body = provider.build_request_body("gpt-4o-mini", &messages, &[], &options);

        assert!(body.get("tool_choice").is_none());
        assert_eq!(body["response_format"]["type"], "json_object");
    }

    #[test]
    fn test_detects_cloudflare_gateway_host() {
        assert!(is_cloudflare_ai_gateway_base(
            "https://gateway.ai.cloudflare.com/v1/acct/gw/compat"
        ));
        assert!(!is_cloudflare_ai_gateway_base("https://api.openai.com/v1"));
    }

    #[test]
    fn test_cloudflare_models_fallback_url_when_base_has_compat() {
        let provider = OpenAiCompatibleProvider::new_with_gateway_token(
            "https://gateway.ai.cloudflare.com/v1/a/g/compat",
            "test-key",
            None,
        )
        .expect("provider should initialize");
        assert_eq!(
            provider.cloudflare_models_fallback_url(),
            "https://gateway.ai.cloudflare.com/v1/a/g/compat/v1/models"
        );
    }

    #[test]
    fn test_cloudflare_models_fallback_url_when_base_has_no_compat() {
        let provider = OpenAiCompatibleProvider::new_with_gateway_token(
            "https://gateway.ai.cloudflare.com/v1/a/g",
            "test-key",
            None,
        )
        .expect("provider should initialize");
        assert_eq!(
            provider.cloudflare_models_fallback_url(),
            "https://gateway.ai.cloudflare.com/v1/a/g/compat/v1/models"
        );
    }

    #[test]
    fn test_with_auth_headers_includes_gateway_header_when_set() {
        let provider = OpenAiCompatibleProvider::new_with_gateway_token(
            "https://api.openai.com/v1",
            "test-key",
            Some("cf-gateway-token"),
        )
        .expect("provider should initialize");
        let request = provider
            .with_auth_headers(provider.client.get("https://example.com/models"))
            .build()
            .expect("request should build");

        assert_eq!(
            request.headers().get("Authorization").unwrap(),
            "Bearer test-key"
        );
        assert_eq!(
            request.headers().get("cf-aig-authorization").unwrap(),
            "Bearer cf-gateway-token"
        );
    }

    #[test]
    fn test_parse_models_response_parses_openai_shape() {
        let models = OpenAiCompatibleProvider::parse_models_response(
            r#"{"data":[{"id":"gpt-4o-mini"},{"id":"gpt-4.1"}]}"#,
        )
        .expect("models should parse");
        assert_eq!(models, vec!["gpt-4o-mini", "gpt-4.1"]);
    }
}
