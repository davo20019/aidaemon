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
use chromiumoxide::cdp::browser_protocol::target::{
    CloseTargetParams, CreateTargetParams, GetTargetsParams, TargetId,
};
use chromiumoxide::page::ScreenshotParams;
use futures::StreamExt;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::config::{BrowserConfig, SessionIsolation};

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

/// Lightweight description of a live browser target/tab, returned by
/// [`BrowserBackend::list_targets`]. The `target_id` is the chromiumoxide target
/// id string (used as the opaque, stable tab id surfaced to the LLM); `title`
/// and `url` are best-effort and may be `None`/empty if the page is still
/// loading or unreachable.
///
/// `opener_id` is the target id of the page that SPAWNED this target (the CDP
/// `openerId`), or `None` for targets opened independently (a user-typed tab,
/// the initial blank tab, etc.). Popup attribution uses it to bind a net-new
/// target to the SPECIFIC session whose page opened it, instead of attributing
/// any net-new global target to whoever clicked — which under concurrent timing
/// would let one session capture another session's freshly-opened tab.
#[derive(Debug, Clone)]
pub struct TargetInfo {
    pub target_id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub opener_id: Option<String>,
}

/// A browser backend: owns the connection lifecycle and hands out page handles.
#[async_trait]
pub trait BrowserBackend: Send + Sync {
    /// Ensure a browser is launched or connected and ready for use.
    async fn ensure_ready(&self) -> Result<(), String>;

    /// Create a NEW page/tab and return its `(target_id, handle)`.
    ///
    /// Unlike the old `current_page()` (which grabbed the first existing tab and
    /// thus shared one page across all sessions), this always creates a distinct
    /// page so the session registry can give each `_session_id` its own tab.
    async fn create_page(&self) -> Result<(String, Arc<dyn PageHandle>), String>;

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

    // --- Tab/target management ---

    /// List every live target/tab the browser currently holds, with best-effort
    /// title and URL. Used to diff for popup detection and to render `list_tabs`.
    async fn list_targets(&self) -> Result<Vec<TargetInfo>, String>;

    /// Resolve an existing target by id into a page handle, so a session can bind
    /// it as its active tab (the chromiumoxide-level "switch"). Returns `Err` if
    /// the target is unknown to the browser.
    async fn page_for_target(&self, target_id: &str) -> Result<Arc<dyn PageHandle>, String>;

    /// Close the target/tab identified by `target_id` at the browser level.
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
    /// Resolved session-isolation mode (computed once at construction). Controls
    /// whether each session's page is created in the default shared context
    /// (`Page` — shared cookies) or its own incognito browser context
    /// (`BrowserContext` — isolated cookies).
    session_isolation: SessionIsolation,
    config: BrowserConfig,
}

impl ChromiumoxideBackend {
    /// Construct a backend, resolving the session-isolation mode from config.
    ///
    /// Returns an `Err` if the configuration is incompatible (e.g. requesting
    /// per-session `browser_context` isolation while attached to a shared
    /// persistent profile or remote-debugging Chrome). Callers must surface this
    /// error at startup rather than silently downgrading — honesty over a
    /// false claim of cookie isolation.
    pub fn new(config: BrowserConfig) -> Result<Self, String> {
        let session_isolation = config.resolve_session_isolation()?;
        let headless = AtomicBool::new(config.headless);
        Ok(Self {
            browser: Arc::new(Mutex::new(None)),
            browser_handle: Arc::new(Mutex::new(None)),
            connected_to_existing: Arc::new(Mutex::new(false)),
            headless,
            session_isolation,
            config,
        })
    }

    /// The resolved session-isolation mode this backend will use.
    pub fn session_isolation(&self) -> SessionIsolation {
        self.session_isolation
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

    /// Create a fresh `about:blank` page and return its target id + handle.
    ///
    /// The global browser mutex is held only long enough to issue `new_page`;
    /// the returned handle owns its own `Arc<Page>`, so the action that follows
    /// runs without holding this mutex.
    ///
    /// In `Page` isolation mode the page is created in the default (shared)
    /// browser context — sessions share cookies. In `BrowserContext` mode a new
    /// incognito browser context is created per session, giving real per-session
    /// cookie/cache isolation. The created context is NOT disposed here;
    /// disposal-on-close is deferred (see Task 11). This is a known, bounded
    /// context leak for the process lifetime.
    async fn new_blank_page(&self) -> Result<(String, Arc<chromiumoxide::Page>), String> {
        let guard = self.browser.lock().await;
        let browser = guard
            .as_ref()
            .ok_or_else(|| "Browser not initialized".to_string())?;

        let page = match self.session_isolation {
            SessionIsolation::Page => browser
                .new_page("about:blank")
                .await
                .map_err(|e| format!("Failed to create new page: {}", e))?,
            SessionIsolation::BrowserContext => {
                // Create a fresh incognito browser context so this session's
                // cookies/cache are isolated from every other session.
                let context_id = browser
                    .create_browser_context(Default::default())
                    .await
                    .map_err(|e| format!("Failed to create isolated browser context: {}", e))?;
                // NOTE: context_id is intentionally not disposed here — disposal
                // on session close is a deferred concern (Task 11).
                debug!("Created isolated browser context for session");
                let params = CreateTargetParams::builder()
                    .url("about:blank")
                    .browser_context_id(context_id)
                    .build()
                    .map_err(|e| format!("Failed to build isolated page parameters: {}", e))?;
                browser
                    .new_page(params)
                    .await
                    .map_err(|e| format!("Failed to create new isolated page: {}", e))?
            }
        };
        let target_id = page.target_id().as_ref().to_string();
        Ok((target_id, Arc::new(page)))
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

    async fn create_page(&self) -> Result<(String, Arc<dyn PageHandle>), String> {
        let (target_id, page) = self.new_blank_page().await?;
        Ok((target_id, Arc::new(ChromiumoxidePage { page })))
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

    async fn list_targets(&self) -> Result<Vec<TargetInfo>, String> {
        let guard = self.browser.lock().await;
        let browser = guard
            .as_ref()
            .ok_or_else(|| "Browser not initialized".to_string())?;
        // Use CDP `Target.getTargets` rather than `pages()` because the raw
        // `TargetInfo` carries `openerId` — the id of the target that spawned
        // this one. Popup detection needs the opener to attribute a net-new tab
        // to the SPECIFIC session whose page opened it (cross-session safety),
        // which `pages()`/`Page` does not surface.
        let returns = browser
            .execute(GetTargetsParams::builder().build())
            .await
            .map_err(|e| format!("Failed to list browser tabs: {}", e))?;
        let infos = returns
            .result
            .target_infos
            .into_iter()
            // Only real page tabs — drop browser/background/service-worker etc.
            .filter(|t| t.r#type == "page")
            .map(|t| {
                let title = (!t.title.is_empty()).then(|| t.title.clone());
                let url = (!t.url.is_empty()).then(|| t.url.clone());
                TargetInfo {
                    target_id: String::from(t.target_id),
                    title,
                    url,
                    opener_id: t.opener_id.map(String::from),
                }
            })
            .collect();
        Ok(infos)
    }

    async fn page_for_target(&self, target_id: &str) -> Result<Arc<dyn PageHandle>, String> {
        let guard = self.browser.lock().await;
        let browser = guard
            .as_ref()
            .ok_or_else(|| "Browser not initialized".to_string())?;
        let page = browser
            .get_page(TargetId::new(target_id.to_string()))
            .await
            .map_err(|e| format!("Tab '{}' not found: {}", target_id, e))?;
        Ok(Arc::new(ChromiumoxidePage {
            page: Arc::new(page),
        }))
    }

    async fn close_target(&self, target_id: &str) -> Result<(), String> {
        let guard = self.browser.lock().await;
        let browser = guard
            .as_ref()
            .ok_or_else(|| "Browser not initialized".to_string())?;
        browser
            .execute(CloseTargetParams::new(TargetId::new(target_id.to_string())))
            .await
            .map_err(|e| format!("Failed to close tab '{}': {}", target_id, e))?;
        Ok(())
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
    /// Recorded each time the backend creates a new per-session page. Carries the
    /// synthetic target id handed back, so tests can assert two sessions caused
    /// two distinct pages.
    CreatePage(String),
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
    /// Monotonic counter behind a Mutex, used to mint a fresh synthetic page id
    /// for each `create_page` call (so distinct sessions get distinct ids).
    next_page_id: Mutex<u64>,
    /// Live targets the mock browser "holds". `create_page`/`page_for_target`
    /// append/track here; `list_targets` returns these. Each entry is a
    /// `TargetInfo`. Shared with `MockPage` so a click can reveal a popup.
    targets: Arc<Mutex<Vec<TargetInfo>>>,
    /// A scripted popup revealed (appended to `targets`) the next time a page
    /// CLICK happens, modeling a `target=_blank`/`window.open` spawn. Shared with
    /// `MockPage` so the reveal is tied to the click, not to `list_targets`.
    pending_popup: Arc<Mutex<Option<TargetInfo>>>,
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
            next_page_id: Mutex::new(0),
            targets: Arc::new(Mutex::new(Vec::new())),
            pending_popup: Arc::new(Mutex::new(None)),
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

    /// Override the URL the mock's pages report from `url()` (and that `new_tab`
    /// snapshots into the session). Lets tests feed a path/query-bearing URL to
    /// assert origin redaction in `list_tabs`.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Shared handle to the recorded call log, for assertions.
    pub fn calls(&self) -> Arc<Mutex<Vec<MockCall>>> {
        Arc::clone(&self.calls)
    }

    /// Script a popup target revealed by the next `MockPage::click` (appended to
    /// the live target set on click), simulating a click that opened a new tab
    /// (`target=_blank` / `window.open`). The popup's `opener_id` is left `None`,
    /// so callers that need it attributed to a clicking session should use
    /// [`MockBackend::script_popup_with_opener`] with that session's active
    /// target id. The given `url` is used verbatim so tests can assert origin
    /// redaction (e.g. include a path + query string).
    pub async fn script_popup(&self, target_id: &str, title: &str, url: &str) {
        self.script_popup_with_opener(target_id, title, url, None)
            .await;
    }

    /// Like [`MockBackend::script_popup`], but sets the popup's `opener_id`.
    ///
    /// Pass `Some(active_target_id)` of the clicking session to model a popup
    /// the session legitimately spawned (it should be attributed to that
    /// session). Pass a DIFFERENT target id to model a tab opened by another
    /// session / independently (it must NOT be attributed to the clicker).
    pub async fn script_popup_with_opener(
        &self,
        target_id: &str,
        title: &str,
        url: &str,
        opener_id: Option<&str>,
    ) {
        *self.pending_popup.lock().await = Some(TargetInfo {
            target_id: target_id.to_string(),
            title: Some(title.to_string()),
            url: Some(url.to_string()),
            opener_id: opener_id.map(|s| s.to_string()),
        });
    }

    async fn record(&self, call: MockCall) {
        self.calls.lock().await.push(call);
    }
}

#[cfg(test)]
struct MockPage {
    /// Synthetic page/target id this handle was created with. Lets tests tie a
    /// recorded action back to the specific per-session page it ran on.
    #[allow(dead_code)]
    page_id: String,
    calls: Arc<Mutex<Vec<MockCall>>>,
    eval_result: Option<serde_json::Value>,
    text_result: String,
    screenshot_bytes: Vec<u8>,
    url: Option<String>,
    /// Shared live-target list + scripted popup, so a click on this page can
    /// reveal a popup target (modeling target=_blank/window.open).
    targets: Arc<Mutex<Vec<TargetInfo>>>,
    pending_popup: Arc<Mutex<Option<TargetInfo>>>,
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
        // A click reveals any scripted popup: it appears as a new live target,
        // exactly as a real target=_blank/window.open would after the click.
        if let Some(popup) = self.pending_popup.lock().await.take() {
            self.targets.lock().await.push(popup);
        }
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

    async fn create_page(&self) -> Result<(String, Arc<dyn PageHandle>), String> {
        let page_id = {
            let mut counter = self.next_page_id.lock().await;
            *counter += 1;
            format!("mock-page-{}", *counter)
        };
        self.record(MockCall::CreatePage(page_id.clone())).await;
        // Register the new target so list_targets/popup-diff can see it.
        self.targets.lock().await.push(TargetInfo {
            target_id: page_id.clone(),
            title: Some("mock tab".to_string()),
            url: self.url.clone(),
            // A directly-created page has no opener (independent tab).
            opener_id: None,
        });
        Ok((
            page_id.clone(),
            Arc::new(MockPage {
                page_id,
                calls: Arc::clone(&self.calls),
                eval_result: self.eval_result.clone(),
                text_result: self.text_result.clone(),
                screenshot_bytes: self.screenshot_bytes.clone(),
                url: self.url.clone(),
                targets: Arc::clone(&self.targets),
                pending_popup: Arc::clone(&self.pending_popup),
            }),
        ))
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

    async fn list_targets(&self) -> Result<Vec<TargetInfo>, String> {
        Ok(self.targets.lock().await.clone())
    }

    async fn page_for_target(&self, target_id: &str) -> Result<Arc<dyn PageHandle>, String> {
        let known = self
            .targets
            .lock()
            .await
            .iter()
            .any(|t| t.target_id == target_id);
        if !known {
            return Err(format!("Tab '{}' not found", target_id));
        }
        Ok(Arc::new(MockPage {
            page_id: target_id.to_string(),
            calls: Arc::clone(&self.calls),
            eval_result: self.eval_result.clone(),
            text_result: self.text_result.clone(),
            screenshot_bytes: self.screenshot_bytes.clone(),
            url: self.url.clone(),
            targets: Arc::clone(&self.targets),
            pending_popup: Arc::clone(&self.pending_popup),
        }))
    }

    async fn close_target(&self, target_id: &str) -> Result<(), String> {
        self.targets
            .lock()
            .await
            .retain(|t| t.target_id != target_id);
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }
}

// =============================================================================
// Backend construction / isolation-resolution tests (no real Chrome needed)
// =============================================================================

#[cfg(test)]
mod isolation_tests {
    use super::*;
    use crate::config::SessionIsolation;

    fn ephemeral_config() -> BrowserConfig {
        BrowserConfig {
            enabled: true,
            headless: true,
            screenshot_width: 1280,
            screenshot_height: 720,
            remote_debugging_port: None,
            user_data_dir: None,
            profile: None,
            session_isolation: None,
        }
    }

    #[test]
    fn ephemeral_auto_resolves_to_browser_context() {
        match ChromiumoxideBackend::new(ephemeral_config()) {
            Ok(backend) => assert_eq!(
                backend.session_isolation(),
                SessionIsolation::BrowserContext
            ),
            Err(e) => panic!("ephemeral config must construct: {e}"),
        }
    }

    #[test]
    fn profile_auto_resolves_to_page() {
        let mut config = ephemeral_config();
        config.user_data_dir = Some("/tmp/profile".to_string());
        match ChromiumoxideBackend::new(config) {
            Ok(backend) => assert_eq!(backend.session_isolation(), SessionIsolation::Page),
            Err(e) => panic!("profile config must construct: {e}"),
        }
    }

    #[test]
    fn incompatible_browser_context_with_profile_is_rejected_at_construction() {
        let mut config = ephemeral_config();
        config.session_isolation = Some(SessionIsolation::BrowserContext);
        config.user_data_dir = Some("/tmp/profile".to_string());
        match ChromiumoxideBackend::new(config) {
            Ok(_) => {
                panic!("browser_context + persistent profile must fail fast at construction")
            }
            Err(err) => assert!(
                err.contains("browser_context"),
                "construction error should explain the incompatibility: {err}"
            ),
        }
    }
}
