use std::sync::Arc;

use async_trait::async_trait;
use chromiumoxide::browser::{Browser, BrowserConfig as ChromeBrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::ScreenshotParams;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

use crate::config::BrowserConfig;
use crate::traits::Tool;
use crate::types::{MediaKind, MediaMessage};

pub struct BrowserTool {
    browser: Arc<Mutex<Option<Browser>>>,
    browser_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    config: BrowserConfig,
    media_tx: mpsc::Sender<MediaMessage>,
}

impl BrowserTool {
    pub fn new(config: BrowserConfig, media_tx: mpsc::Sender<MediaMessage>) -> Self {
        Self {
            browser: Arc::new(Mutex::new(None)),
            browser_handle: Arc::new(Mutex::new(None)),
            config,
            media_tx,
        }
    }

    async fn ensure_browser(&self) -> Result<(), String> {
        let mut guard = self.browser.lock().await;
        if guard.is_some() {
            return Ok(());
        }

        let mut builder = ChromeBrowserConfig::builder();
        if self.config.headless {
            builder = builder.arg("--headless=new");
        }
        // Use existing Chrome profile if configured (inherits cookies/sessions)
        if let Some(ref user_data_dir) = self.config.user_data_dir {
            let expanded = shellexpand::tilde(user_data_dir);
            builder = builder.arg(format!("--user-data-dir={}", expanded));
            let profile = self.config.profile.as_deref().unwrap_or("Default");
            builder = builder.arg(format!("--profile-directory={}", profile));
            info!(
                user_data_dir = %expanded,
                profile,
                "Using existing Chrome profile"
            );
        }

        builder = builder
            .arg(format!(
                "--window-size={},{}",
                self.config.screenshot_width, self.config.screenshot_height
            ))
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("--disable-gpu")
            .arg("--disable-dev-shm-usage");

        let browser_config = builder.build().map_err(|e| {
            format!(
                "Failed to build browser config: {}. Is Chrome/Chromium installed?",
                e
            )
        })?;

        let (browser, mut handler) =
            Browser::launch(browser_config)
                .await
                .map_err(|e| {
                    format!(
                    "Failed to launch browser: {}. Make sure Chrome or Chromium is installed.",
                    e
                )
                })?;

        let handle = tokio::spawn(async move {
            while handler.next().await.is_some() {}
        });

        info!("Browser launched successfully");
        *guard = Some(browser);

        let mut handle_guard = self.browser_handle.lock().await;
        *handle_guard = Some(handle);

        Ok(())
    }

    async fn get_page(&self) -> Result<Arc<chromiumoxide::Page>, String> {
        let guard = self.browser.lock().await;
        let browser = guard
            .as_ref()
            .ok_or_else(|| "Browser not initialized".to_string())?;

        let pages = browser
            .pages()
            .await
            .map_err(|e| format!("Failed to get pages: {}", e))?;

        if let Some(page) = pages.into_iter().next() {
            Ok(Arc::new(page))
        } else {
            let page = browser
                .new_page("about:blank")
                .await
                .map_err(|e| format!("Failed to create new page: {}", e))?;
            Ok(Arc::new(page))
        }
    }

    async fn action_navigate(&self, args: &Value) -> Result<String, String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: url".to_string())?;

        self.ensure_browser().await?;
        let page = self.get_page().await?;

        page.goto(url)
            .await
            .map_err(|e| format!("Failed to navigate to {}: {}", url, e))?;

        // Wait briefly for page load
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        Ok(format!("Navigated to {}", url))
    }

    async fn action_screenshot(&self, args: &Value, session_id: &str) -> Result<String, String> {
        self.ensure_browser().await?;
        let page = self.get_page().await?;

        let png_bytes = if let Some(selector) = args.get("selector").and_then(|v| v.as_str()) {
            let element = page
                .find_element(selector)
                .await
                .map_err(|e| format!("Element not found '{}': {}", selector, e))?;
            element
                .screenshot(CaptureScreenshotFormat::Png)
                .await
                .map_err(|e| format!("Failed to screenshot element: {}", e))?
        } else {
            page.screenshot(ScreenshotParams::builder().full_page(true).build())
                .await
                .map_err(|e| format!("Failed to take screenshot: {}", e))?
        };

        let caption = format!(
            "Screenshot of {}",
            page.url()
                .await
                .ok()
                .flatten()
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

        self.ensure_browser().await?;
        let page = self.get_page().await?;

        let element = page
            .find_element(selector)
            .await
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;

        element
            .click()
            .await
            .map_err(|e| format!("Failed to click '{}': {}", selector, e))?;

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

        self.ensure_browser().await?;
        let page = self.get_page().await?;

        let element = page
            .find_element(selector)
            .await
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;

        element
            .click()
            .await
            .map_err(|e| format!("Failed to focus '{}': {}", selector, e))?;

        element
            .type_str(value)
            .await
            .map_err(|e| format!("Failed to type into '{}': {}", selector, e))?;

        Ok(format!(
            "Filled '{}' with '{}'",
            selector,
            if value.len() > 20 {
                format!("{}...", &value[..20])
            } else {
                value.to_string()
            }
        ))
    }

    async fn action_get_text(&self, args: &Value) -> Result<String, String> {
        self.ensure_browser().await?;
        let page = self.get_page().await?;

        let text = if let Some(selector) = args.get("selector").and_then(|v| v.as_str()) {
            // Verify the element exists first
            page.find_element(selector)
                .await
                .map_err(|e| format!("Element not found '{}': {}", selector, e))?;

            let js = format!(
                "document.querySelector('{}').innerText",
                selector.replace('\'', "\\'")
            );
            let result = page
                .evaluate(js)
                .await
                .map_err(|e| format!("Failed to get text from '{}': {}", selector, e))?;

            result
                .into_value::<String>()
                .unwrap_or_else(|_| "(could not extract text)".to_string())
        } else {
            let result = page
                .evaluate("document.body.innerText")
                .await
                .map_err(|e| format!("Failed to get page text: {}", e))?;

            result
                .into_value::<String>()
                .unwrap_or_else(|_| "(could not extract text)".to_string())
        };

        // Truncate if very long
        let text = if text.len() > 4000 {
            format!("{}... (truncated)", &text[..4000])
        } else {
            text
        };

        Ok(text)
    }

    async fn action_execute_js(&self, args: &Value) -> Result<String, String> {
        let script = args
            .get("script")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: script".to_string())?;

        self.ensure_browser().await?;
        let page = self.get_page().await?;

        let result = page
            .evaluate(script)
            .await
            .map_err(|e| format!("JavaScript execution failed: {}", e))?;

        let value_str = match result.into_value::<Value>() {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| format!("{:?}", v)),
            Err(_) => "(no return value)".to_string(),
        };

        let value_str = if value_str.len() > 4000 {
            format!("{}... (truncated)", &value_str[..4000])
        } else {
            value_str
        };

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

        self.ensure_browser().await?;
        let page = self.get_page().await?;

        let deadline = tokio::time::Instant::now()
            + tokio::time::Duration::from_secs(timeout_secs);

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

    async fn action_close(&self) -> Result<String, String> {
        let mut guard = self.browser.lock().await;
        if guard.is_some() {
            *guard = None;
            // Abort the handler task
            let mut handle_guard = self.browser_handle.lock().await;
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
            info!("Browser session closed");
            Ok("Browser session closed.".to_string())
        } else {
            Ok("No browser session was active.".to_string())
        }
    }
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &str {
        "browser"
    }

    fn description(&self) -> &str {
        "Control a headless browser to navigate pages, click elements, fill forms, take screenshots, extract text, and execute JavaScript."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "browser",
            "description": "Control a headless browser for web interactions. Actions: navigate (go to URL), screenshot (capture page as photo), click (click element), fill (type into input), get_text (extract text), execute_js (run JavaScript), wait (wait for element), close (end session). The browser persists across calls for multi-step workflows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "screenshot", "click", "fill", "get_text", "execute_js", "wait", "close"],
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
                        "description": "Text to type into the element (for 'fill' action)"
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
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));

        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("");

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
            "close" => self.action_close().await,
            _ => Err(format!(
                "Unknown browser action: '{}'. Valid actions: navigate, screenshot, click, fill, get_text, execute_js, wait, close",
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
}
