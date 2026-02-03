use std::io::Cursor;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use crate::traits::Tool;

/// Build an HTTP client with browser-like headers.
/// Shared by WebFetchTool and DuckDuckGo search backend.
pub fn build_browser_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:142.0) Gecko/20100101 Firefox/142.0")
        .default_headers({
            let mut h = reqwest::header::HeaderMap::new();
            h.insert("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".parse().unwrap());
            h.insert("Accept-Language", "en-US,en;q=0.5".parse().unwrap());
            h.insert("Accept-Encoding", "gzip, deflate, br".parse().unwrap());
            h.insert("DNT", "1".parse().unwrap());
            h.insert("Upgrade-Insecure-Requests", "1".parse().unwrap());
            h.insert("Sec-Fetch-Dest", "document".parse().unwrap());
            h.insert("Sec-Fetch-Mode", "navigate".parse().unwrap());
            h.insert("Sec-Fetch-Site", "none".parse().unwrap());
            h.insert("Sec-Fetch-User", "?1".parse().unwrap());
            h.insert("Sec-GPC", "1".parse().unwrap());
            h
        })
        .build()
        .expect("failed to build browser HTTP client")
}

pub struct WebFetchTool {
    client: Client,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: build_browser_client(),
        }
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a URL and extract its readable content"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "web_fetch",
            "description": "Fetch a URL and extract its readable content. Strips ads/navigation. For login-required sites, use browser instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 20000)"
                    }
                },
                "required": ["url"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));
        let url = args["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: url"))?;
        let max_chars = args["max_chars"].as_u64().unwrap_or(20000) as usize;

        let resp = self.client.get(url).send().await?;
        if !resp.status().is_success() {
            return Ok(format!("Error fetching {}: HTTP {}", url, resp.status()));
        }
        let html = resp.text().await?;

        // Try readability extraction first
        let parsed_url = reqwest::Url::parse(url)
            .unwrap_or_else(|_| reqwest::Url::parse("http://example.com").unwrap());
        let text = {
            let mut cursor = Cursor::new(html.as_bytes());
            match llm_readability::extractor::extract(&mut cursor, &parsed_url) {
                Ok(product) if !product.text.trim().is_empty() => product.text,
                _ => {
                    // Fallback: convert raw HTML to markdown
                    htmd::convert(&html).unwrap_or_else(|_| html.clone())
                }
            }
        };

        let mut result = format!("Content from {}:\n\n", url);
        if text.len() > max_chars {
            result.push_str(&text[..max_chars]);
            result.push_str("\n\n[Truncated]");
        } else {
            result.push_str(&text);
        }

        Ok(result)
    }
}
