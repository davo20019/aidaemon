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
use std::time::Duration;

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

/// Upper bound on how long the graceful Chrome shutdown (`Browser::close()` +
/// `Browser::wait()`) and per-resource disposal (close-target / dispose-context)
/// are allowed to take before we give up and fall back to the abrupt teardown.
/// Keeps a wedged/dying Chrome from hanging the close path (or the daemon's
/// shutdown) indefinitely.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

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
    ///
    /// The action layer (`action_navigate`/`action_new_tab`) calls this AFTER a
    /// `goto` settles and revalidates the committed URL against the shared
    /// private-network policy (`policy::validate_network_url`). This catches
    /// server-side redirects to a blocked host (e.g. a public redirector landing
    /// on `http://127.0.0.1/...`).
    ///
    /// NOTE (deferred): this is NOT per-request subresource interception. Frame,
    /// script, XHR/fetch, WebSocket, and form-POST requests issued by the page
    /// are NOT individually validated, because chromiumoxide 0.8 offers no
    /// per-request continue/abort seam without a fragile browser-global Fetch
    /// pump (see the `#[ignore]`d `deferred_per_request_subresource_interception_stub`
    /// test for the full feasibility finding). Enforcement today is: validated
    /// tool-initiated navigations + final-committed-URL revalidation.
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

    /// Create a NEW page/tab and return its `(target_id, context_id, handle)`.
    ///
    /// Unlike the old `current_page()` (which grabbed the first existing tab and
    /// thus shared one page across all sessions), this always creates a distinct
    /// page so the session registry can give each `_session_id` its own tab.
    ///
    /// `context_id` is `Some(id)` in `browser_context` isolation mode — the CDP
    /// browser-context id the page was created in, so the session can dispose it
    /// on eviction/close. It is `None` in `page` isolation mode (the page lives in
    /// the default shared context, which is never disposed).
    async fn create_page(&self) -> Result<(String, Option<String>, Arc<dyn PageHandle>), String>;

    /// Gracefully shut the browser session down. Returns a user-facing status
    /// message describing what happened (disconnected vs. closed vs. no-op).
    ///
    /// For a LAUNCHED browser this performs a graceful Chrome shutdown
    /// (`Browser::close()` + `Browser::wait()`, bounded by a timeout) before
    /// falling back to aborting the handler task; for an ATTACHED browser
    /// (`remote_debugging_port`) it detaches WITHOUT sending a browser-close
    /// command, so the user's own Chrome keeps running. Idempotent: safe to call
    /// repeatedly (a second call on an already-closed backend is a no-op).
    ///
    /// This is the single graceful-teardown path; `close`, `set_headless_mode`'s
    /// restart, idle eviction, and daemon shutdown all route through it.
    async fn shutdown(&self) -> Result<String, String>;

    /// Dispose the per-session resources surfaced by idle eviction (or session
    /// close): close each tab target, then dispose any per-session CDP browser
    /// context. Best-effort and bounded — a failure on one resource is logged and
    /// does not block the others or wedge the caller. Does NOT close the browser
    /// itself (that is [`BrowserBackend::shutdown`]'s job).
    async fn dispose_session(&self, tab_target_ids: &[String], context_ids: &[String]);

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

    // --- Connection health + recovery ---

    /// Whether the backend currently holds a live connection.
    ///
    /// This reflects REAL handler health: it is `true` only when a browser is
    /// held AND its CDP event handler task is still running. A browser whose
    /// handler has exited (the connection died) reports `false`, even though the
    /// `Option<Browser>` is still `Some`.
    ///
    /// Exposed for health introspection; `ensure_ready` consults the same
    /// underlying handler-alive signal directly.
    #[allow(dead_code)]
    async fn is_connected(&self) -> bool;

    /// Forcibly drop the current browser + handler and re-run the launch/connect
    /// path, spawning a fresh supervised handler. Respects `connected_to_existing`
    /// (reconnects to the same remote-debugging port for attached Chrome;
    /// relaunches for an owned instance).
    ///
    /// Returns `Ok(())` on a successful reconnect. After a successful reconnect
    /// all previously-minted page handles are stale and MUST be re-created
    /// (the tool layer invalidates the session registry's cached pages).
    async fn reconnect(&self) -> Result<(), String>;
}

/// Classify a backend/page error string as a transport/connection-class failure
/// (the connection to Chrome itself is dead) versus an ordinary page-level error
/// (element not found, navigation failed, timeout, etc.).
///
/// This drives the tool layer's recovery decision: connection-class errors
/// trigger a single reconnect (+ observation retry / mutation non-replay), while
/// ordinary errors are surfaced verbatim with no reconnect.
///
/// The patterns are deliberately CONSERVATIVE — only true transport failures.
/// chromiumoxide's `CdpError` surfaces transport death through several shapes:
/// - websocket teardown (`"websocket"`, `"Ws("`, `"connection closed"`,
///   `"connection reset"`),
/// - the internal command channel being torn down when the handler task ends
///   (`"channel closed"`, `"Sender was dropped"`, `"receiver"`/`"sender"` dropped,
///   `"request did not resolve"`),
/// - the launched/attached Chrome process going away
///   (`"chrome process"`, `"the browser was closed"`, `"no response from"`).
///
/// A normal `"Element not found"` / `"JavaScript execution failed"` /
/// `"Timeout"` is NOT a connection error — it must never trigger a reconnect.
pub fn is_connection_error(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    const CONNECTION_PATTERNS: &[&str] = &[
        "connection closed",
        "connection reset",
        "connection refused",
        "connection aborted",
        "websocket",
        "ws(",
        "channel closed",
        "sender was dropped",
        "sender dropped",
        "receiver dropped",
        "channel is closed",
        "request did not resolve",
        "no response from",
        "the browser was closed",
        "chrome process",
        "browser process",
        "transport",
        "broken pipe",
    ];
    CONNECTION_PATTERNS.iter().any(|p| e.contains(p))
}

// =============================================================================
// Chromiumoxide adapter
// =============================================================================

/// Production backend: wraps the existing `chromiumoxide` `Browser`/`Page`
/// logic verbatim. Behavior is identical to the pre-refactor `BrowserTool`.
pub struct ChromiumoxideBackend {
    browser: Arc<Mutex<Option<Browser>>>,
    browser_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Liveness flag for the CDP event-handler task. Set `true` at spawn and set
    /// `false` by the supervised drain loop when it exits (the connection died or
    /// the browser closed). `ensure_ready`/`is_connected` consult this so a dead
    /// handler is treated as a dead connection instead of "healthy because
    /// `browser.is_some()`".
    handler_alive: Arc<AtomicBool>,
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
            handler_alive: Arc::new(AtomicBool::new(false)),
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
    /// browser context — sessions share cookies, and the returned context id is
    /// `None`. In `BrowserContext` mode a new incognito browser context is
    /// created per session, giving real per-session cookie/cache isolation, and
    /// its id is returned so the session can dispose it on eviction/close
    /// (`dispose_session`), closing the context leak that earlier tasks deferred.
    async fn new_blank_page(
        &self,
    ) -> Result<(String, Option<String>, Arc<chromiumoxide::Page>), String> {
        let guard = self.browser.lock().await;
        let browser = guard
            .as_ref()
            .ok_or_else(|| "Browser not initialized".to_string())?;

        let (page, context_id) = match self.session_isolation {
            SessionIsolation::Page => (
                browser
                    .new_page("about:blank")
                    .await
                    .map_err(|e| format!("Failed to create new page: {}", e))?,
                None,
            ),
            SessionIsolation::BrowserContext => {
                // Create a fresh incognito browser context so this session's
                // cookies/cache are isolated from every other session. The id is
                // returned to the caller and stored in the session/tab state so
                // it can be disposed when the session is evicted or closed.
                let context_id = browser
                    .create_browser_context(Default::default())
                    .await
                    .map_err(|e| format!("Failed to create isolated browser context: {}", e))?;
                debug!("Created isolated browser context for session");
                let context_id_str = context_id.as_ref().to_string();
                let params = CreateTargetParams::builder()
                    .url("about:blank")
                    .browser_context_id(context_id)
                    .build()
                    .map_err(|e| format!("Failed to build isolated page parameters: {}", e))?;
                let page = browser
                    .new_page(params)
                    .await
                    .map_err(|e| format!("Failed to create new isolated page: {}", e))?;
                (page, Some(context_id_str))
            }
        };
        let target_id = page.target_id().as_ref().to_string();
        Ok((target_id, context_id, Arc::new(page)))
    }

    /// Abruptly tear the browser down: drop the `Browser`, mark the handler dead,
    /// abort the handler task, and clear `connected_to_existing`. This is the
    /// fallback when a graceful close/wait errors or times out, and the only
    /// teardown an ATTACHED browser ever needs (we must NOT send a browser-close
    /// command to the user's own Chrome). The caller holds the `browser` guard.
    async fn abort_teardown(&self, guard: &mut Option<Browser>) {
        *guard = None;
        self.handler_alive.store(false, Ordering::SeqCst);
        let mut handle_guard = self.browser_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            handle.abort();
        }
        *self.connected_to_existing.lock().await = false;
    }

    /// The shared graceful-teardown core. Returns a user-facing status message.
    ///
    /// - LAUNCHED browser: take the `Browser` out and call `close().await` then
    ///   `wait().await`, bounded by [`SHUTDOWN_TIMEOUT`]. On error/timeout, fall
    ///   back to [`Self::abort_teardown`]. Either way the handler is aborted and
    ///   state cleared.
    /// - ATTACHED browser (`connected_to_existing`): NEVER send a browser-close
    ///   command (that would kill the user's Chrome) — just detach + abort.
    /// - No browser held: no-op (idempotent).
    async fn graceful_teardown(&self) -> String {
        let mut guard = self.browser.lock().await;
        if guard.is_none() {
            return "No browser session was active.".to_string();
        }

        let was_connected = *self.connected_to_existing.lock().await;
        if was_connected {
            // Attached: detach only. Do NOT close the user's Chrome.
            self.abort_teardown(&mut guard).await;
            info!("Disconnected from existing Chrome (browser still running)");
            return "Disconnected from Chrome (your browser is still running).".to_string();
        }

        // Launched: take ownership so we can call the &mut close()/wait().
        let mut browser = guard.take().expect("guard is Some (checked above)");
        let graceful = tokio::time::timeout(SHUTDOWN_TIMEOUT, async {
            browser.close().await.map_err(|e| e.to_string())?;
            // wait() collects the child process so it doesn't become a zombie.
            browser.wait().await.map_err(|e| e.to_string())?;
            Ok::<(), String>(())
        })
        .await;

        // The Browser is dropped here regardless (the `take()`d value goes out of
        // scope), so even a timed-out close still releases the connection.
        match graceful {
            Ok(Ok(())) => info!("Browser closed gracefully"),
            Ok(Err(e)) => warn!(error = %e, "graceful browser close failed; forcing teardown"),
            Err(_) => warn!("graceful browser close timed out; forcing teardown"),
        }
        // Clear remaining state (handler/handle/flag). `guard` is already None.
        self.abort_teardown(&mut guard).await;
        info!("Browser session closed");
        "Browser session closed.".to_string()
    }

    /// Spawn the SUPERVISED CDP event-handler drain task.
    ///
    /// Unlike the old fire-and-forget `tokio::spawn(async move { while
    /// handler.next().await.is_some() {} })`, this records the handler's exit:
    /// `handler_alive` is set `true` here and flipped to `false` the moment the
    /// drain loop ends (which happens when `handler.next()` yields `None` — i.e.
    /// the websocket/connection to Chrome died or the browser was closed). That
    /// turns "connection died" into an observable signal instead of a silent
    /// state where `browser` stays `Some` forever.
    fn spawn_supervised_handler<H>(&self, mut handler: H) -> tokio::task::JoinHandle<()>
    where
        H: futures::Stream + Send + Unpin + 'static,
    {
        let alive = Arc::clone(&self.handler_alive);
        alive.store(true, Ordering::SeqCst);
        tokio::spawn(async move {
            while handler.next().await.is_some() {}
            // The handler stream ended: the CDP connection is gone. Record it so
            // ensure_ready/is_connected stop treating this browser as healthy.
            alive.store(false, Ordering::SeqCst);
            warn!("browser CDP handler exited — connection is no longer healthy");
        })
    }

    /// Launch a new Chrome or connect to an existing one (per config), store the
    /// browser + a freshly-spawned supervised handler, and update
    /// `connected_to_existing`. The caller holds the `browser` mutex guard.
    ///
    /// This is the single launch/connect path shared by the first `ensure_ready`
    /// and by `reconnect` — so reconnect re-uses the exact same logic (including
    /// remote-port reconnect vs. relaunch) rather than duplicating it.
    async fn launch_or_connect(&self, guard: &mut Option<Browser>) -> Result<(), String> {
        // Connect to existing Chrome if a remote debugging port is configured.
        if let Some(port) = self.config.remote_debugging_port {
            let url = format!("http://127.0.0.1:{}", port);
            info!(port, "Connecting to existing Chrome instance");

            let (browser, handler) = Browser::connect(&url).await.map_err(|e| {
                format!(
                    "Failed to connect to Chrome on port {}. \
                         Make sure Chrome is running with: --remote-debugging-port={}\n\
                         Error: {}",
                    port, port, e
                )
            })?;

            let handle = self.spawn_supervised_handler(handler);

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

        // Otherwise, launch a new Chrome instance.
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

        let (browser, handler) = Browser::launch(browser_config).await.map_err(|e| {
            format!(
                "Failed to launch browser: {}. Make sure Chrome or Chromium is installed.",
                e
            )
        })?;

        let handle = self.spawn_supervised_handler(handler);

        info!("Browser launched successfully");
        *guard = Some(browser);
        *self.connected_to_existing.lock().await = false;

        let mut handle_guard = self.browser_handle.lock().await;
        *handle_guard = Some(handle);

        Ok(())
    }
}

#[async_trait]
impl BrowserBackend for ChromiumoxideBackend {
    async fn ensure_ready(&self) -> Result<(), String> {
        let mut guard = self.browser.lock().await;

        // Healthy ONLY if we hold a browser AND its handler task is still alive.
        // The pre-fix check (`guard.is_some()`) treated a dead connection as
        // healthy forever: when Chrome's websocket dies, the handler drain loop
        // ends but `browser` stays `Some`. Now we verify the handler is alive.
        if guard.is_some() && self.handler_alive.load(Ordering::SeqCst) {
            return Ok(());
        }

        // Either we have no browser yet, OR the handler has died (dead
        // connection). In the dead-handler case, drop the stale browser/handle
        // first so launch_or_connect starts clean, then (re)launch/connect.
        if guard.is_some() {
            warn!("browser handler is dead — dropping stale connection and reconnecting");
            *guard = None;
            let mut handle_guard = self.browser_handle.lock().await;
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }

        self.launch_or_connect(&mut guard).await
    }

    async fn create_page(&self) -> Result<(String, Option<String>, Arc<dyn PageHandle>), String> {
        let (target_id, context_id, page) = self.new_blank_page().await?;
        Ok((target_id, context_id, Arc::new(ChromiumoxidePage { page })))
    }

    async fn shutdown(&self) -> Result<String, String> {
        Ok(self.graceful_teardown().await)
    }

    async fn dispose_session(&self, tab_target_ids: &[String], context_ids: &[String]) {
        // Best-effort: with no live browser there is nothing to dispose.
        if self.browser.lock().await.is_none() {
            return;
        }
        // Close each tab target first, then dispose the browser context(s).
        for target_id in tab_target_ids {
            match tokio::time::timeout(SHUTDOWN_TIMEOUT, self.close_target(target_id)).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!(target_id, error = %e, "dispose_session: close_target failed"),
                Err(_) => warn!(target_id, "dispose_session: close_target timed out"),
            }
        }
        for context_id in context_ids {
            let guard = self.browser.lock().await;
            let Some(browser) = guard.as_ref() else {
                return;
            };
            match tokio::time::timeout(
                SHUTDOWN_TIMEOUT,
                browser.dispose_browser_context(context_id.clone()),
            )
            .await
            {
                Ok(Ok(())) => debug!(context_id, "disposed isolated browser context"),
                Ok(Err(e)) => {
                    warn!(context_id, error = %e, "dispose_session: dispose_browser_context failed")
                }
                Err(_) => warn!(
                    context_id,
                    "dispose_session: dispose_browser_context timed out"
                ),
            }
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

        // Gracefully close the existing browser so the next call launches with
        // the new mode. Reuses the shared graceful-teardown path (launched →
        // close()+wait()+timeout+fallback; attached → detach only).
        let had_browser = self.browser.lock().await.is_some();
        if had_browser {
            self.graceful_teardown().await;
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
        // Real health: a browser is held AND its handler task is still running.
        self.browser.lock().await.is_some() && self.handler_alive.load(Ordering::SeqCst)
    }

    async fn reconnect(&self) -> Result<(), String> {
        let mut guard = self.browser.lock().await;
        // Forcibly drop the current browser + handler regardless of state.
        *guard = None;
        self.handler_alive.store(false, Ordering::SeqCst);
        {
            let mut handle_guard = self.browser_handle.lock().await;
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
        info!("reconnecting browser after a connection-class failure");
        self.launch_or_connect(&mut guard).await
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
        // Use the element handle's own `inner_text()` method instead of building
        // a JS string from the selector.  The old approach interpolated `selector`
        // into `document.querySelector('<selector>').innerText`, which broke on
        // backslashes, newlines, and un-escapable CSS escapes, and could inject
        // arbitrary JS via a crafted selector.
        //
        // `Page::find_element` passes the selector to the CDP DOM query (not to
        // any JS source string), and `Element::inner_text()` evaluates the
        // constant function `function() { return this.innerText; }` bound to the
        // element object — NO selector interpolation occurs anywhere.
        //
        // Real-Chrome adversarial-selector coverage is deferred to the Task 18
        // smoke suite; the mock-level dispatch contract (selector passed through
        // unchanged, no Evaluate call recorded) is locked down in Task 14 tests.
        let element = self
            .page
            .find_element(selector)
            .await
            .map_err(|e| format!("Element not found '{}': {}", selector, e))?;

        Ok(element
            .inner_text()
            .await
            .map_err(|e| format!("Failed to get text from '{}': {}", selector, e))?
            .unwrap_or_else(|| "(could not extract text)".to_string()))
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
    /// Recorded by `shutdown`. `graceful` is `true` when the mock performed a
    /// graceful close (launched browser), `false` when it merely detached an
    /// attached browser WITHOUT issuing a browser-close command.
    Shutdown {
        graceful: bool,
    },
    SetHeadlessMode(bool),
    /// Recorded by `dispose_session` per tab target id it closed.
    CloseTarget(String),
    /// Recorded by `dispose_session` per browser context id it disposed.
    DisposeContext(String),
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
    /// Recorded by `MockBackend::reconnect`. Asserting on its count proves the
    /// tool layer reconnected exactly once on a connection-class error.
    Reconnect,
}

/// Which page operation a scripted connection-class error should fire on, used
/// by `MockBackend::fail_once_with_connection_error_on`. Distinct ops let a test
/// target the body_text/inner_text read for the observation-retry case or the
/// click/replace_text op for the mutation non-replay case.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailOp {
    BodyText,
    InnerText,
    Click,
    ReplaceText,
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
    /// Handler liveness, mirroring `ChromiumoxideBackend::handler_alive`. Tests
    /// flip this to model a dead CDP handler; `ensure_ready` consults it and, on
    /// a dead handler, relaunches (recorded as a fresh `EnsureReady`/`Reconnect`).
    handler_alive: Arc<AtomicBool>,
    /// A scripted one-shot connection-class error: the FIRST time the named op
    /// runs, it returns the connection error and clears the slot, so the retry
    /// (against a fresh page after reconnect) succeeds. Shared with `MockPage`.
    pending_fail: Arc<Mutex<Option<FailOp>>>,
    /// When `true`, `create_page` hands back a synthetic browser-context id (one
    /// per page), modeling `browser_context` isolation. When `false`, context
    /// ids are `None` (the `page`-isolation default).
    isolate_contexts: bool,
    /// When `true`, the mock models an ATTACHED browser (`connected_to_existing`):
    /// `shutdown` detaches WITHOUT a graceful close (records `graceful: false`).
    attached: bool,
    /// When `true`, the mock models a graceful close that ERRORS/TIMES OUT —
    /// `shutdown` still tears down and records `graceful: false`, exercising the
    /// forced-cleanup fallback path.
    graceful_fails: bool,
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
            handler_alive: Arc::new(AtomicBool::new(true)),
            pending_fail: Arc::new(Mutex::new(None)),
            isolate_contexts: false,
            attached: false,
            graceful_fails: false,
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

    /// Model `browser_context` isolation: `create_page` returns a synthetic
    /// context id per page (so eviction has something to dispose).
    pub fn with_isolated_contexts(mut self) -> Self {
        self.isolate_contexts = true;
        self
    }

    /// Model an ATTACHED browser (`connected_to_existing`): `shutdown` must
    /// detach WITHOUT a graceful browser-close command.
    pub fn attached(mut self) -> Self {
        self.attached = true;
        self
    }

    /// Model a graceful close that errors/times out: `shutdown` still tears down
    /// (forced-cleanup fallback) and records `graceful: false`.
    pub fn with_failing_graceful_close(mut self) -> Self {
        self.graceful_fails = true;
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

    /// Mark the mock's CDP handler as dead, modeling a browser whose connection
    /// died. `ensure_ready` sees `is_some && !handler_alive` and relaunches.
    pub fn mark_handler_dead(&self) {
        self.handler_alive.store(false, Ordering::SeqCst);
    }

    /// Script a one-shot connection-class error on the next invocation of `op`.
    /// The op records its `MockCall` (so we can assert it ran), then returns a
    /// transport-class error string and clears the slot — so a retry succeeds.
    pub async fn fail_once_with_connection_error_on(&self, op: FailOp) {
        *self.pending_fail.lock().await = Some(op);
    }

    /// Count of `Reconnect` calls recorded so far.
    pub async fn reconnect_count(&self) -> usize {
        self.calls
            .lock()
            .await
            .iter()
            .filter(|c| matches!(c, MockCall::Reconnect))
            .count()
    }

    async fn record(&self, call: MockCall) {
        self.calls.lock().await.push(call);
    }
}

/// The connection-class error string a scripted mock op returns. Matches
/// `is_connection_error` so the tool layer routes it through recovery.
#[cfg(test)]
const MOCK_CONNECTION_ERROR: &str = "WebSocket connection closed: Sender was dropped";

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
    /// Shared one-shot connection-error script (see `MockBackend::pending_fail`).
    pending_fail: Arc<Mutex<Option<FailOp>>>,
}

#[cfg(test)]
impl MockPage {
    async fn record(&self, call: MockCall) {
        self.calls.lock().await.push(call);
    }

    /// If a one-shot connection error is scripted for `op`, consume it and return
    /// the transport-class error string; otherwise `None` (op proceeds normally).
    async fn take_scripted_failure(&self, op: FailOp) -> Option<String> {
        let mut slot = self.pending_fail.lock().await;
        if *slot == Some(op) {
            *slot = None;
            return Some(MOCK_CONNECTION_ERROR.to_string());
        }
        None
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
        if let Some(err) = self.take_scripted_failure(FailOp::Click).await {
            return Err(err);
        }
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
        if let Some(err) = self.take_scripted_failure(FailOp::ReplaceText).await {
            return Err(err);
        }
        Ok(())
    }

    async fn evaluate(&self, script: &str) -> Result<Option<serde_json::Value>, String> {
        self.record(MockCall::Evaluate(script.to_string())).await;
        Ok(self.eval_result.clone())
    }

    async fn inner_text(&self, selector: &str) -> Result<String, String> {
        self.record(MockCall::InnerText(selector.to_string())).await;
        if let Some(err) = self.take_scripted_failure(FailOp::InnerText).await {
            return Err(err);
        }
        Ok(self.text_result.clone())
    }

    async fn body_text(&self) -> Result<String, String> {
        self.record(MockCall::BodyText).await;
        if let Some(err) = self.take_scripted_failure(FailOp::BodyText).await {
            return Err(err);
        }
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
        // Model the production health check: a (re)launch revives the handler and
        // marks the connection live. A test that flipped `mark_handler_dead`
        // before this call sees the EnsureReady record + a revived handler,
        // proving a dead handler is no longer treated as healthy.
        self.connected.store(true, Ordering::Relaxed);
        self.handler_alive.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn create_page(&self) -> Result<(String, Option<String>, Arc<dyn PageHandle>), String> {
        let page_id = {
            let mut counter = self.next_page_id.lock().await;
            *counter += 1;
            format!("mock-page-{}", *counter)
        };
        self.record(MockCall::CreatePage(page_id.clone())).await;
        // In browser_context isolation mode, hand back a synthetic context id
        // (one per page) so eviction/close has something to dispose.
        let context_id = self
            .isolate_contexts
            .then(|| format!("mock-ctx-{}", page_id));
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
            context_id,
            Arc::new(MockPage {
                page_id,
                calls: Arc::clone(&self.calls),
                eval_result: self.eval_result.clone(),
                text_result: self.text_result.clone(),
                screenshot_bytes: self.screenshot_bytes.clone(),
                url: self.url.clone(),
                targets: Arc::clone(&self.targets),
                pending_popup: Arc::clone(&self.pending_popup),
                pending_fail: Arc::clone(&self.pending_fail),
            }),
        ))
    }

    async fn shutdown(&self) -> Result<String, String> {
        // Attached or failing-graceful both end in a forced/detach teardown that
        // records `graceful: false`; a healthy launched browser records
        // `graceful: true`. Either way the connection ends up torn down.
        let graceful = !self.attached && !self.graceful_fails;
        self.record(MockCall::Shutdown { graceful }).await;
        self.connected.store(false, Ordering::Relaxed);
        self.handler_alive.store(false, Ordering::SeqCst);
        if self.attached {
            Ok("Disconnected from Chrome (your browser is still running).".to_string())
        } else {
            Ok("Browser session closed.".to_string())
        }
    }

    async fn dispose_session(&self, tab_target_ids: &[String], context_ids: &[String]) {
        for target_id in tab_target_ids {
            self.record(MockCall::CloseTarget(target_id.clone())).await;
            self.targets
                .lock()
                .await
                .retain(|t| &t.target_id != target_id);
        }
        for context_id in context_ids {
            self.record(MockCall::DisposeContext(context_id.clone()))
                .await;
        }
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
            pending_fail: Arc::clone(&self.pending_fail),
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
        self.connected.load(Ordering::Relaxed) && self.handler_alive.load(Ordering::SeqCst)
    }

    async fn reconnect(&self) -> Result<(), String> {
        self.record(MockCall::Reconnect).await;
        // A successful reconnect revives the connection + handler.
        self.connected.store(true, Ordering::Relaxed);
        self.handler_alive.store(true, Ordering::SeqCst);
        Ok(())
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
