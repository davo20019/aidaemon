use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info};

use crate::providers::ProviderError;
use crate::traits::{ModelProvider, ProviderResponse, ToolCall};

pub struct OpenAiCompatibleProvider {
    client: Client,
    base_url: String,
    api_key: String,
}

impl OpenAiCompatibleProvider {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        }
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
        let mut body = json!({
            "model": model,
            "messages": messages,
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
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        debug!("Provider response: {}", &text[..text.len().min(2000)]);

        let data: Value = serde_json::from_str(&text)?;
        let choice = data["choices"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;
        let message = &choice["message"];

        let content = message["content"].as_str().map(|s| s.to_string());

        let mut tool_calls = Vec::new();
        if let Some(tcs) = message["tool_calls"].as_array() {
            debug!("Raw tool_calls from provider: {}", serde_json::to_string(tcs).unwrap_or_default());
            for tc in tcs {
                let extra_content = tc.get("extra_content")
                    .filter(|v| !v.is_null())
                    .cloned();

                tool_calls.push(ToolCall {
                    id: tc["id"].as_str().unwrap_or("").to_string(),
                    name: tc["function"]["name"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                    arguments: tc["function"]["arguments"]
                        .as_str()
                        .unwrap_or("{}")
                        .to_string(),
                    extra_content,
                });
            }
        }

        Ok(ProviderResponse {
            content,
            tool_calls,
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
