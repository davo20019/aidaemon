use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::warn;

use crate::config::BrowserConfig;
use crate::traits::{
    Tool, ToolCallSemantics, ToolCapabilities, ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::{MediaKind, MediaMessage};

mod backend;
#[cfg(test)]
mod tests;

use backend::{BrowserBackend, ChromiumoxideBackend};

pub struct BrowserTool {
    backend: Arc<dyn BrowserBackend>,
    media_tx: mpsc::Sender<MediaMessage>,
}

impl BrowserTool {
    pub fn new(config: BrowserConfig, media_tx: mpsc::Sender<MediaMessage>) -> Self {
        Self {
            backend: Arc::new(ChromiumoxideBackend::new(config)),
            media_tx,
        }
    }

    /// Test-only constructor that injects an arbitrary backend (e.g. the mock).
    #[cfg(test)]
    pub fn with_backend(
        backend: Arc<dyn BrowserBackend>,
        media_tx: mpsc::Sender<MediaMessage>,
    ) -> Self {
        Self { backend, media_tx }
    }

    async fn action_navigate(&self, args: &Value) -> Result<String, String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: url".to_string())?;

        // Block navigation to internal/private IPs (SSRF protection)
        if let Err(reason) = crate::tools::web_fetch::validate_url_for_ssrf(url) {
            return Err(format!("Navigation blocked: {}", reason));
        }

        self.backend.ensure_ready().await?;
        let page = self.backend.current_page().await?;

        page.goto(url).await?;

        // Wait briefly for page load
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        Ok(format!("Navigated to {}", url))
    }

    async fn action_screenshot(&self, args: &Value, session_id: &str) -> Result<String, String> {
        self.backend.ensure_ready().await?;
        let page = self.backend.current_page().await?;

        let selector = args.get("selector").and_then(|v| v.as_str());
        let png_bytes = page.screenshot(selector).await?;

        let caption = format!(
            "Screenshot of {}",
            page.url()
                .await
                .unwrap_or_else(|| "current page".to_string())
        );

        self.media_tx
            .send(MediaMessage {
                session_id: session_id.to_string(),
                caption: caption.clone(),
                kind: MediaKind::Photo { data: png_bytes },
            })
            .await
            .map_err(|e| format!("Failed to send screenshot to Telegram: {}", e))?;

        Ok(format!("Screenshot taken and sent to chat. {}", caption))
    }

    async fn action_click(&self, args: &Value) -> Result<String, String> {
        let selector = args
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: selector".to_string())?;

        self.backend.ensure_ready().await?;
        let page = self.backend.current_page().await?;

        page.click(selector).await?;

        // Brief wait for any navigation/JS to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        Ok(format!("Clicked element '{}'", selector))
    }

    async fn action_fill(&self, args: &Value) -> Result<String, String> {
        let selector = args
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: selector".to_string())?;
        let value = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: value".to_string())?;

        self.backend.ensure_ready().await?;
        let page = self.backend.current_page().await?;

        page.type_text(selector, value).await?;

        tracing::info!(
            action = "fill",
            selector,
            value_bytes = value.len(),
            "browser fill"
        );

        Ok(format!("Filled '{}'", selector))
    }

    async fn action_get_text(&self, args: &Value) -> Result<String, String> {
        self.backend.ensure_ready().await?;
        let page = self.backend.current_page().await?;

        let text = if let Some(selector) = args.get("selector").and_then(|v| v.as_str()) {
            page.inner_text(selector).await?
        } else {
            page.body_text().await?
        };

        // Truncate if very long
        let text = crate::utils::truncate_with_note(&text, 4000);

        Ok(text)
    }

    async fn action_execute_js(&self, args: &Value) -> Result<String, String> {
        let script = args
            .get("script")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: script".to_string())?;

        self.backend.ensure_ready().await?;
        let page = self.backend.current_page().await?;

        let result = page.evaluate(script).await?;

        let value_str = match result {
            Some(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| format!("{:?}", v)),
            None => "(no return value)".to_string(),
        };

        let value_str = crate::utils::truncate_with_note(&value_str, 4000);

        Ok(value_str)
    }

    async fn action_wait(&self, args: &Value) -> Result<String, String> {
        let selector = args
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: selector".to_string())?;
        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(10);

        self.backend.ensure_ready().await?;
        let page = self.backend.current_page().await?;

        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout_secs);

        loop {
            match page.find_element(selector).await {
                Ok(_) => return Ok(format!("Element '{}' found", selector)),
                Err(_) => {
                    if tokio::time::Instant::now() >= deadline {
                        return Err(format!(
                            "Timeout: element '{}' not found after {}s",
                            selector, timeout_secs
                        ));
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                }
            }
        }
    }

    async fn action_set_mode(&self, args: &Value) -> Result<String, String> {
        let mode = args.get("value").and_then(|v| v.as_str()).ok_or_else(|| {
            "Missing required parameter: value (\"visible\" or \"headless\")".to_string()
        })?;

        let new_headless = match mode {
            "visible" | "headed" => false,
            "headless" => true,
            _ => {
                return Err(format!(
                    "Invalid mode '{}'. Use 'visible' or 'headless'.",
                    mode
                ))
            }
        };

        self.backend.set_headless_mode(new_headless, mode).await
    }

    async fn action_close(&self) -> Result<String, String> {
        self.backend.close().await
    }
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &str {
        "browser"
    }

    fn description(&self) -> &str {
        "Control a browser to navigate pages, click elements, fill forms, take screenshots, extract text, and execute JavaScript. Supports headless and visible modes."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "browser",
            "description": "Control a browser for web interactions. Actions: navigate (go to URL), screenshot (capture page as photo), click (click element), fill (type into input), get_text (extract text), execute_js (run JavaScript), wait (wait for element), set_mode (switch between 'visible' and 'headless' — use visible for sites that block headless browsers), close (end session). The browser persists across calls for multi-step workflows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "screenshot", "click", "fill", "get_text", "execute_js", "wait", "set_mode", "close"],
                        "description": "The browser action to perform"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (for 'navigate' action)"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the target element (for click, fill, get_text, wait, screenshot)"
                    },
                    "value": {
                        "type": "string",
                        "description": "Text to type (for 'fill') or mode to set (for 'set_mode': 'visible' or 'headless')"
                    },
                    "script": {
                        "type": "string",
                        "description": "JavaScript code to execute (for 'execute_js' action)"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Timeout in seconds for 'wait' action (default: 10)"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;

        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: action"))?;

        let session_id = args
            .get("_session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let result = match action {
            "navigate" => self.action_navigate(&args).await,
            "screenshot" => self.action_screenshot(&args, session_id).await,
            "click" => self.action_click(&args).await,
            "fill" => self.action_fill(&args).await,
            "get_text" => self.action_get_text(&args).await,
            "execute_js" => self.action_execute_js(&args).await,
            "wait" => self.action_wait(&args).await,
            "set_mode" => self.action_set_mode(&args).await,
            "close" => self.action_close().await,
            _ => Err(format!(
                "Unknown browser action: '{}'. Valid actions: navigate, screenshot, click, fill, get_text, execute_js, wait, set_mode, close",
                action
            )),
        };

        // Return errors as text so the LLM can adjust its approach
        match result {
            Ok(text) => Ok(text),
            Err(err_text) => {
                warn!(action, error = %err_text, "Browser action failed");
                Ok(format!("Error: {}", err_text))
            }
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let args = serde_json::from_str::<Value>(arguments).ok();
        let action = args
            .as_ref()
            .and_then(|value| value.get("action"))
            .and_then(|value| value.as_str())
            .map(|value| value.trim().to_ascii_lowercase());
        let url = args
            .as_ref()
            .and_then(|value| value.get("url"))
            .and_then(|value| value.as_str())
            .unwrap_or_default();

        match action.as_deref() {
            Some("navigate") => {
                ToolCallSemantics::observation().with_target_hint(ToolTargetHintKind::Url, url)
            }
            Some("get_text") => ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
            Some("wait") => ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
            Some("screenshot") => ToolCallSemantics::observation(),
            Some("click" | "fill" | "execute_js") => ToolCallSemantics::mutation(),
            Some("close" | "set_mode") => ToolCallSemantics::administrative(),
            _ => ToolCallSemantics::mutation(),
        }
    }
}
