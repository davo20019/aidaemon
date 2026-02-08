use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};
use zeroize::Zeroize;

use crate::providers::ProviderError;
use crate::traits::{ModelProvider, ProviderResponse, TokenUsage, ToolCall};

pub struct OpenAiCompatibleProvider {
    client: Client,
    base_url: String,
    api_key: String,
}

impl Drop for OpenAiCompatibleProvider {
    fn drop(&mut self) {
        self.api_key.zeroize();
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

impl OpenAiCompatibleProvider {
    pub fn new(base_url: &str, api_key: &str) -> Result<Self, String> {
        // Validate URL security before creating provider
        validate_base_url(base_url)?;

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        })
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
        // Strip extra_content from tool_calls before sending — the OpenAI-compatible
        // endpoint doesn't understand it (it's used internally for Gemini native round-trip).
        let mut messages_cleaned: Vec<Value> = messages.to_vec();
        for msg in messages_cleaned.iter_mut() {
            if let Some(tcs) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                for tc in tcs.iter_mut() {
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

        let url = format!("{}/chat/completions", self.base_url);
        info!(model, url = %url, tools = tools.len(), "Calling LLM API");

        let resp = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
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

        let data: Value = serde_json::from_str(&text)?;
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
        })
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            anyhow::bail!("Failed to list models ({}): {}", status, text);
        }

        let data: Value = serde_json::from_str(&text)?;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
