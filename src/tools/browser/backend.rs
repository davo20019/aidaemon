//! Backend seam for the browser tool.
//!
//! `BrowserTool`'s actions are routed through the [`BrowserBackend`] /
//! [`PageHandle`] traits instead of calling `chromiumoxide` directly. This lets
//! tests drive the tool against a [`MockBackend`] with no real Chrome
//! dependency, while production wires up [`ChromiumoxideBackend`], which wraps
//! the existing `Browser`/`Page` calls with no behavior change.
//!
//! Error style mirrors the rest of the module: internal methods return
//! `Result<_, String>`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use chromiumoxide::browser::{Browser, BrowserConfig as ChromeBrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::ScreenshotParams;
use futures::StreamExt;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::config::BrowserConfig;

/// A handle to a single browser page/tab. Methods mirror the chromiumoxide
/// `Page` operations the tool's actions need today. Some methods (e.g. the tab
/// target operations) are declared so later tasks can fill them in; the
/// chromiumoxide adapter currently provides thin/stub implementations for those.
#[async_trait]
pub trait PageHandle: Send + Sync {
    /// Navigate the page to `url`.
    async fn goto(&self, url: &str) -> Result<(), String>;

    /// Returns `Ok(())` if an element matching `selector` exists, `Err` otherwise.
    async fn find_element(&self, selector: &str) -> Result<(), String>;

    /// Click the element matching `selector`.
    async fn click(&self, selector: &str) -> Result<(), String>;

    /// Focus the element matching `selector` and type `value` into it
    /// (append semantics — kept as a seam primitive for future tasks).
    #[allow(dead_code)]
    async fn type_text(&self, selector: &str, value: &str) -> Result<(), String>;

    /// Evaluate a JavaScript `script` and return its result as a JSON value.
    ///
    /// Returns `Ok(None)` when the script produced no deserializable value
    /// (e.g. `undefined`), so callers can reproduce the existing
    /// "(no return value)" fallback while still distinguishing a genuine JSON
    /// `null` (`Ok(Some(Value::Null))`).
    async fn evaluate(&self, script: &str) -> Result<Option<serde_json::Value>, String>;

    /// `innerText` of the element matching `selector`.
    async fn inner_text(&self, selector: &str) -> Result<String, String>;

    /// `document.body.innerText` of the page.
    async fn body_text(&self) -> Result<String, String>;

    /// Capture a screenshot. When `selector` is `Some`, screenshots that
    /// element; otherwise captures the full page.
    async fn screenshot(&self, selector: Option<&str>) -> Result<Vec<u8>, String>;

    /// Current page URL, if available.
    async fn url(&self) -> Option<String>;

    /// Replace the entire content of the element matching `selector` with
    /// `value`, firing native `input` and `change` events.
    ///
    /// This replaces whatever is already in the field. The `value` is NEVER
    /// interpolated into a JavaScript source string — it is delivered as real
    /// key events via `type_str` after the field is cleared through an
    /// element-bound JS function.
    async fn replace_text(&self, selector: &str, value: &str) -> Result<(), String>;
}

/// A browser backend: owns the connection lifecycle and hands out page handles.
#[async_trait]
pub trait BrowserBackend: Send + Sync {
    /// Ensure a browser is launched or connected and ready for use.
    async fn ensure_ready(&self) -> Result<(), String>;

    /// Get the current page (or create an `about:blank` page if none exists).
    async fn current_page(&self) -> Result<Arc<dyn PageHandle>, String>;

    /// Close the browser session. Returns a user-facing status message
    /// describing what happened (disconnected vs. closed vs. no-op).
    async fn close(&self) -> Result<String, String>;

    /// Apply a headless/visible mode switch, restarting the browser on the next
    /// action if the mode actually changed. Returns a user-facing status
    /// message. `headless` is the requested target mode; `mode` is the original
    /// user-supplied label (e.g. "visible"/"headless") used verbatim in messages.
    ///
    /// The backend owns the headless state and the no-display fallback policy,
    /// so this encapsulates the full set_mode behavior.
    async fn set_headless_mode(&self, headless: bool, mode: &str) -> Result<String, String>;

    // --- Tab/target management (declared for later tasks; thin today) ---
    // These are part of the seam's contract per the Phase 0 checklist but are
    // not yet wired to any tool action, so they are unused for now.

    /// List target/tab identifiers currently open.
    #[allow(dead_code)]
    async fn list_targets(&self) -> Result<Vec<String>, String>;

    /// Create a new target/tab at `url`, returning its identifier.
    #[allow(dead_code)]
    async fn create_target(&self, url: &str) -> Result<String, String>;

    /// Switch the active target/tab to `target_id`.
    #[allow(dead_code)]
    async fn switch_target(&self, target_id: &str) -> Result<(), String>;

    /// Close the target/tab identified by `target_id`.
    #[allow(dead_code)]
    async fn close_target(&self, target_id: &str) -> Result<(), String>;

    // --- Connection health (declared for later tasks; thin today) ---

    /// Whether the backend currently holds a live connection.
    #[allow(dead_code)]
    async fn is_connected(&self) -> bool;
}

// =============================================================================
// Chromiumoxide adapter
// =============================================================================

/// Production backend: wraps the existing `chromiumoxide` `Browser`/`Page`
/// logic verbatim. Behavior is identical to the pre-refactor `BrowserTool`.
pub struct ChromiumoxideBackend {
    browser: Arc<Mutex<Option<Browser>>>,
    browser_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// True when connected to an existing Chrome (don't close it on "close").
    connected_to_existing: Arc<Mutex<bool>>,
    /// Runtime-mutable headless mode (togglable via the tool's set_mode action).
    headless: AtomicBool,
    config: BrowserConfig,
}

impl ChromiumoxideBackend {
    pub fn new(config: BrowserConfig) -> Self {
        let headless = AtomicBool::new(config.headless);
        Self {
            browser: Arc::new(Mutex::new(None)),
            browser_handle: Arc::new(Mutex::new(None)),
            connected_to_existing: Arc::new(Mutex::new(false)),
            headless,
            config,
        }
    }

    /// Check if a display is available for non-headless Chrome.
    fn has_display() -> bool {
        if cfg!(target_os = "macos") {
            // macOS always has display capability (Quartz)
            true
        } else if cfg!(target_os = "windows") {
            true
        } else {
            // Linux: check for X11 or Wayland display
            std::env::var("DISPLAY").is_ok() || std::env::var("WAYLAND_DISPLAY").is_ok()
        }
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
}

#[async_trait]
impl BrowserBackend for ChromiumoxideBackend {
    async fn ensure_ready(&self) -> Result<(), String> {
        let mut guard = self.browser.lock().await;
        if guard.is_some() {
            return Ok(());
        }

        // If remote_debugging_port is set, connect to existing Chrome instead of launching
        if let Some(port) = self.config.remote_debugging_port {
            let url = format!("http://127.0.0.1:{}", port);
            info!(port, "Connecting to existing Chrome instance");

            let (browser, mut handler) = Browser::connect(&url).await.map_err(|e| {
                format!(
                    "Failed to connect to Chrome on port {}. \
                         Make sure Chrome is running with: --remote-debugging-port={}\n\
                         Error: {}",
                    port, port, e
                )
            })?;

            let handle = tokio::spawn(async move { while handler.next().await.is_some() {} });

            info!(
                port,
                "Connected to existing Chrome — sharing login sessions"
            );
            *guard = Some(browser);
            *self.connected_to_existing.lock().await = true;

            let mut handle_guard = self.browser_handle.lock().await;
            *handle_guard = Some(handle);

            return Ok(());
        }

        // Otherwise, launch a new Chrome instance
        let mut builder = ChromeBrowserConfig::builder();
        let want_headless = self.headless.load(Ordering::Relaxed);
        let use_headless = if !want_headless && !Self::has_display() {
            warn!("No display available — falling back to headless mode");
            true
        } else {
            want_headless
        };
        if use_headless {
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
            .arg("--disable-dev-shm-usage")
            // Anti-detection: prevent sites from identifying headless Chrome
            .arg("--disable-blink-features=AutomationControlled")
            .arg("--disable-features=AutomationControlled");

        let browser_config = builder.build().map_err(|e| {
            format!(
                "Failed to build browser config: {}. Is Chrome/Chromium installed?",
                e
            )
        })?;

        let (browser, mut handler) = Browser::launch(browser_config).await.map_err(|e| {
            format!(
                "Failed to launch browser: {}. Make sure Chrome or Chromium is installed.",
                e
            )
        })?;

        let handle = tokio::spawn(async move { while handler.next().await.is_some() {} });

        info!("Browser launched successfully");
        *guard = Some(browser);

        let mut handle_guard = self.browser_handle.lock().await;
        *handle_guard = Some(handle);

        Ok(())
    }

    async fn current_page(&self) -> Result<Arc<dyn PageHandle>, String> {
        let page = self.get_page().await?;
        Ok(Arc::new(ChromiumoxidePage { page }))
    }

    async fn close(&self) -> Result<String, String> {
        let mut guard = self.browser.lock().await;
        if guard.is_some() {
            let was_connected = *self.connected_to_existing.lock().await;
            *guard = None;
            // Abort the handler task
            let mut handle_guard = self.browser_handle.lock().await;
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
            *self.connected_to_existing.lock().await = false;
            if was_connected {
                info!("Disconnected from existing Chrome (browser still running)");
                Ok("Disconnected from Chrome (your browser is still running).".to_string())
            } else {
                info!("Browser session closed");
                Ok("Browser session closed.".to_string())
            }
        } else {
            Ok("No browser session was active.".to_string())
        }
    }

    async fn set_headless_mode(&self, headless: bool, mode: &str) -> Result<String, String> {
        let new_headless = headless;
        let old_headless = self.headless.load(Ordering::Relaxed);
        if old_headless == new_headless {
            return Ok(format!("Browser is already in {} mode.", mode));
        }

        // Warn if requesting visible mode on a headless server
        if !new_headless && !Self::has_display() {
            return Ok(
                "No display available on this system. Visible mode requires a monitor, \
                 VNC, or X forwarding (ssh -X). Staying in headless mode."
                    .to_string(),
            );
        }

        self.headless.store(new_headless, Ordering::Relaxed);

        // Close existing browser so next call launches with the new mode
        let mut guard = self.browser.lock().await;
        if guard.is_some() {
            *guard = None;
            let mut handle_guard = self.browser_handle.lock().await;
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
            *self.connected_to_existing.lock().await = false;
            info!(mode, "Browser mode changed, restarting on next use");
            Ok(format!(
                "Switched to {} mode. Browser will restart on next action.",
                mode
            ))
        } else {
            info!(mode, "Browser mode changed");
            Ok(format!("Switched to {} mode.", mode))
        }
    }

    // The tool does not expose tab/target actions yet; these are thin stubs that
    // later tasks will flesh out against the live chromiumoxide target API.
    async fn list_targets(&self) -> Result<Vec<String>, String> {
        Err("Tab/target management not yet implemented".to_string())
    }

    async fn create_target(&self, _url: &str) -> Result<String, String> {
        Err("Tab/target management not yet implemented".to_string())
    }

    async fn switch_target(&self, _target_id: &str) -> Result<(), String> {
        Err("Tab/target management not yet implemented".to_string())
    }

    async fn close_target(&self, _target_id: &str) -> Result<(), String> {
        Err("Tab/target management not yet implemented".to_string())
    }

    async fn is_connected(&self) -> bool {
        self.browser.lock().await.is_some()
    }
}

/// Page handle wrapping a concrete `chromiumoxide::Page`.
struct ChromiumoxidePage {
    page: Arc<chromiumoxide::Page>,
}

#[async_trait]
impl PageHandle for ChromiumoxidePage {
    async fn goto(&self, url: &str) -> Result<(), String> {
        self.page
            .goto(url)
            .await
            .map_err(|e| format!("Failed to navigate to {}: {}", url, e))?;
        Ok(())
    }

    async fn find_element(&self, selector: &str) -> Result<(), String> {
        self.page
            .find_element(selector)
            .await
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;
        Ok(())
    }

    async fn click(&self, selector: &str) -> Result<(), String> {
        let element = self
            .page
            .find_element(selector)
            .await
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;
        element
            .click()
            .await
            .map_err(|e| format!("Failed to click '{}': {}", selector, e))?;
        Ok(())
    }

    async fn type_text(&self, selector: &str, value: &str) -> Result<(), String> {
        let element = self
            .page
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
        Ok(())
    }

    async fn replace_text(&self, selector: &str, value: &str) -> Result<(), String> {
        let element = self
            .page
            .find_element(selector)
            .await
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;

        // Focus the element.
        element
            .focus()
            .await
            .map_err(|e| format!("Failed to focus '{}': {}", selector, e))?;

        // Clear the field via a bound JS function — the empty-string literal `''`
        // is a compile-time constant, NOT user data; no interpolation occurs.
        element
            .call_js_fn(
                "function() { \
                    this.value = ''; \
                    this.dispatchEvent(new Event('input', {bubbles: true})); \
                }",
                false,
            )
            .await
            .map_err(|e| format!("Failed to clear '{}': {}", selector, e))?;

        // Type the new value with real key events (never enters a JS string).
        element
            .type_str(value)
            .await
            .map_err(|e| format!("Failed to type into '{}': {}", selector, e))?;

        // Fire a final `change` event (some frameworks rely on it to commit).
        element
            .call_js_fn(
                "function() { \
                    this.dispatchEvent(new Event('change', {bubbles: true})); \
                }",
                false,
            )
            .await
            .map_err(|e| format!("Failed to dispatch change on '{}': {}", selector, e))?;

        Ok(())
    }

    async fn evaluate(&self, script: &str) -> Result<Option<serde_json::Value>, String> {
        let result = self
            .page
            .evaluate(script)
            .await
            .map_err(|e| format!("JavaScript execution failed: {}", e))?;
        Ok(result.into_value::<serde_json::Value>().ok())
    }

    async fn inner_text(&self, selector: &str) -> Result<String, String> {
        // Verify the element exists first (preserves original error message).
        self.page
            .find_element(selector)
            .await
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;

        let js = format!(
            "document.querySelector('{}').innerText",
            selector.replace('\'', "\\'")
        );
        let result = self
            .page
            .evaluate(js)
            .await
            .map_err(|e| format!("Failed to get text from '{}': {}", selector, e))?;

        Ok(result
            .into_value::<String>()
            .unwrap_or_else(|_| "(could not extract text)".to_string()))
    }

    async fn body_text(&self) -> Result<String, String> {
        let result = self
            .page
            .evaluate("document.body.innerText")
            .await
            .map_err(|e| format!("Failed to get page text: {}", e))?;

        Ok(result
            .into_value::<String>()
            .unwrap_or_else(|_| "(could not extract text)".to_string()))
    }

    async fn screenshot(&self, selector: Option<&str>) -> Result<Vec<u8>, String> {
        if let Some(selector) = selector {
            let element = self
                .page
                .find_element(selector)
                .await
                .map_err(|e| format!("Element not found '{}': {}", selector, e))?;
            element
                .screenshot(CaptureScreenshotFormat::Png)
                .await
                .map_err(|e| format!("Failed to screenshot element: {}", e))
        } else {
            self.page
                .screenshot(ScreenshotParams::builder().full_page(true).build())
                .await
                .map_err(|e| format!("Failed to take screenshot: {}", e))
        }
    }

    async fn url(&self) -> Option<String> {
        self.page.url().await.ok().flatten()
    }
}

// =============================================================================
// Mock backend (test-only)
// =============================================================================

/// A recorded call against the mock backend, for assertions in tests.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockCall {
    EnsureReady,
    CurrentPage,
    Close,
    SetHeadlessMode(bool),
    Goto(String),
    FindElement(String),
    Click(String),
    TypeText(String, String),
    /// Recorded by `MockPage::replace_text`. Fields: (selector, value).
    ReplaceText(String, String),
    Evaluate(String),
    InnerText(String),
    BodyText,
    Screenshot(Option<String>),
    Url,
}

/// In-memory mock backend that records calls and returns scripted results. It
/// has no real Chrome dependency.
#[cfg(test)]
pub struct MockBackend {
    calls: Arc<Mutex<Vec<MockCall>>>,
    /// Value returned by `evaluate`. `None` mimics a void/`undefined` result.
    eval_result: Option<serde_json::Value>,
    /// Text returned by `inner_text` and `body_text`.
    text_result: String,
    /// Bytes returned by `screenshot`.
    screenshot_bytes: Vec<u8>,
    /// URL returned by `url`.
    url: Option<String>,
    connected: AtomicBool,
}

#[cfg(test)]
impl Default for MockBackend {
    fn default() -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            eval_result: Some(serde_json::json!("mock-eval")),
            text_result: "mock text".to_string(),
            screenshot_bytes: vec![0x89, 0x50, 0x4e, 0x47],
            url: Some("https://mock.example/".to_string()),
            connected: AtomicBool::new(false),
        }
    }
}

#[cfg(test)]
impl MockBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the value returned by `evaluate`. `Some(Value::Null)` mimics a
    /// JS expression that returns `null`; `None` mimics `undefined`/void.
    pub fn with_eval_result(mut self, eval_result: Option<serde_json::Value>) -> Self {
        self.eval_result = eval_result;
        self
    }

    /// Override the text returned by `inner_text` and `body_text`.
    pub fn with_text_result(mut self, text: impl Into<String>) -> Self {
        self.text_result = text.into();
        self
    }

    /// Shared handle to the recorded call log, for assertions.
    pub fn calls(&self) -> Arc<Mutex<Vec<MockCall>>> {
        Arc::clone(&self.calls)
    }

    async fn record(&self, call: MockCall) {
        self.calls.lock().await.push(call);
    }
}

#[cfg(test)]
struct MockPage {
    calls: Arc<Mutex<Vec<MockCall>>>,
    eval_result: Option<serde_json::Value>,
    text_result: String,
    screenshot_bytes: Vec<u8>,
    url: Option<String>,
}

#[cfg(test)]
impl MockPage {
    async fn record(&self, call: MockCall) {
        self.calls.lock().await.push(call);
    }
}

#[cfg(test)]
#[async_trait]
impl PageHandle for MockPage {
    async fn goto(&self, url: &str) -> Result<(), String> {
        self.record(MockCall::Goto(url.to_string())).await;
        Ok(())
    }

    async fn find_element(&self, selector: &str) -> Result<(), String> {
        self.record(MockCall::FindElement(selector.to_string()))
            .await;
        Ok(())
    }

    async fn click(&self, selector: &str) -> Result<(), String> {
        self.record(MockCall::Click(selector.to_string())).await;
        Ok(())
    }

    async fn type_text(&self, selector: &str, value: &str) -> Result<(), String> {
        self.record(MockCall::TypeText(selector.to_string(), value.to_string()))
            .await;
        Ok(())
    }

    async fn replace_text(&self, selector: &str, value: &str) -> Result<(), String> {
        self.record(MockCall::ReplaceText(
            selector.to_string(),
            value.to_string(),
        ))
        .await;
        Ok(())
    }

    async fn evaluate(&self, script: &str) -> Result<Option<serde_json::Value>, String> {
        self.record(MockCall::Evaluate(script.to_string())).await;
        Ok(self.eval_result.clone())
    }

    async fn inner_text(&self, selector: &str) -> Result<String, String> {
        self.record(MockCall::InnerText(selector.to_string())).await;
        Ok(self.text_result.clone())
    }

    async fn body_text(&self) -> Result<String, String> {
        self.record(MockCall::BodyText).await;
        Ok(self.text_result.clone())
    }

    async fn screenshot(&self, selector: Option<&str>) -> Result<Vec<u8>, String> {
        self.record(MockCall::Screenshot(selector.map(|s| s.to_string())))
            .await;
        Ok(self.screenshot_bytes.clone())
    }

    async fn url(&self) -> Option<String> {
        self.record(MockCall::Url).await;
        self.url.clone()
    }
}

#[cfg(test)]
#[async_trait]
impl BrowserBackend for MockBackend {
    async fn ensure_ready(&self) -> Result<(), String> {
        self.record(MockCall::EnsureReady).await;
        self.connected.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn current_page(&self) -> Result<Arc<dyn PageHandle>, String> {
        self.record(MockCall::CurrentPage).await;
        Ok(Arc::new(MockPage {
            calls: Arc::clone(&self.calls),
            eval_result: self.eval_result.clone(),
            text_result: self.text_result.clone(),
            screenshot_bytes: self.screenshot_bytes.clone(),
            url: self.url.clone(),
        }))
    }

    async fn close(&self) -> Result<String, String> {
        self.record(MockCall::Close).await;
        self.connected.store(false, Ordering::Relaxed);
        Ok("Browser session closed.".to_string())
    }

    async fn set_headless_mode(&self, headless: bool, mode: &str) -> Result<String, String> {
        self.record(MockCall::SetHeadlessMode(headless)).await;
        Ok(format!("Switched to {} mode.", mode))
    }

    async fn list_targets(&self) -> Result<Vec<String>, String> {
        Ok(Vec::new())
    }

    async fn create_target(&self, _url: &str) -> Result<String, String> {
        Ok("mock-target".to_string())
    }

    async fn switch_target(&self, _target_id: &str) -> Result<(), String> {
        Ok(())
    }

    async fn close_target(&self, _target_id: &str) -> Result<(), String> {
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }
}
