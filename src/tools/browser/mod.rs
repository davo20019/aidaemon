use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::BrowserConfig;
use crate::traits::{
    Tool, ToolCallSemantics, ToolCapabilities, ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::{MediaKind, MediaMessage};

mod backend;
pub mod policy;
mod session;
#[cfg(test)]
mod tests;

use backend::{BrowserBackend, ChromiumoxideBackend, PageHandle};
use session::{BrowserSessionRegistry, TabView};

use tokio::sync::OwnedMutexGuard;

/// Reduce a full URL to its origin (`scheme://host[:port]`), dropping any
/// userinfo, path, query, and fragment — all of which can carry secrets
/// (credentials, session tokens, reset codes) and must never be surfaced in a
/// tab listing.
///
/// Parsing is deliberately dependency-free string surgery:
/// - For a `scheme://...` URL, take the authority (everything before the first
///   `/`, `?`, or `#`), then drop any `userinfo@` prefix, keeping only the
///   `host[:port]`. So `https://user:pass@host/p?x=secret` → `https://host`.
/// - For inputs without a `scheme://authority` form (e.g. `about:blank`,
///   `data:`, or a schemeless `host.com/path?x=secret`), cut at the first `/`,
///   `?`, or `#`, stripping any path/query/fragment so none of it can leak.
fn redact_origin(url: &str) -> String {
    let url = url.trim();
    if url.is_empty() {
        return String::new();
    }
    // Find the scheme separator.
    if let Some(scheme_end) = url.find("://") {
        let after_scheme = scheme_end + 3;
        let authority_and_rest = &url[after_scheme..];
        // Authority ends at the first '/', '?', or '#'.
        let authority_len = authority_and_rest
            .find(['/', '?', '#'])
            .unwrap_or(authority_and_rest.len());
        let authority = &authority_and_rest[..authority_len];
        // Drop any `userinfo@` prefix so embedded credentials never survive:
        // keep only the host[:port] after the LAST '@'.
        let host = match authority.rfind('@') {
            Some(at) => &authority[at + 1..],
            None => authority,
        };
        return format!("{}://{}", &url[..scheme_end], host);
    }
    // No scheme://authority form (e.g. about:blank, data:, mailto:, or a
    // schemeless host/path). Strip any path/query/fragment by cutting at the
    // first '/', '?', or '#' so no path can leak.
    let cut = url.find(['/', '?', '#']).unwrap_or(url.len());
    url[..cut].to_string()
}

pub struct BrowserTool {
    backend: Arc<dyn BrowserBackend>,
    media_tx: mpsc::Sender<MediaMessage>,
    /// Per-session page state, keyed by trusted internal `_session_id`.
    sessions: BrowserSessionRegistry,
}

impl BrowserTool {
    /// Construct the browser tool, resolving and validating the session
    /// isolation mode up front.
    ///
    /// Returns an `Err` (surfaced at startup) when the configuration would
    /// falsely claim per-session cookie isolation — e.g. `browser_context`
    /// mode requested alongside a shared persistent profile or remote-debugging
    /// Chrome. On success, logs the resolved mode and whether sessions SHARE
    /// cookies, without logging any profile path contents.
    pub fn new(
        config: BrowserConfig,
        media_tx: mpsc::Sender<MediaMessage>,
    ) -> Result<Self, String> {
        let backend = ChromiumoxideBackend::new(config)?;
        let mode = backend.session_isolation();
        let (mode_label, shares_cookies) = match mode {
            crate::config::SessionIsolation::Page => ("page", true),
            crate::config::SessionIsolation::BrowserContext => ("browser_context", false),
        };
        info!(
            isolation = mode_label,
            shares_cookies,
            "browser sessions share cookies: {shares_cookies} (isolation={mode_label})"
        );
        Ok(Self {
            backend: Arc::new(backend),
            media_tx,
            sessions: BrowserSessionRegistry::new(),
        })
    }

    /// Test-only constructor that injects an arbitrary backend (e.g. the mock).
    #[cfg(test)]
    pub fn with_backend(
        backend: Arc<dyn BrowserBackend>,
        media_tx: mpsc::Sender<MediaMessage>,
    ) -> Self {
        Self {
            backend,
            media_tx,
            sessions: BrowserSessionRegistry::new(),
        }
    }

    /// Resolve this session's page and acquire its action lock, held for the
    /// WHOLE action via the returned owned guard.
    ///
    /// The flow is: `ensure_ready()` (global browser launch) → resolve/create
    /// the session's page via the registry → take the per-session action lock.
    /// The action lock serializes a single session's own calls while letting
    /// DIFFERENT sessions proceed concurrently — it is NOT the global browser
    /// mutex, so distinct sessions do not serialize on each other.
    async fn page_for(
        &self,
        session_id: &str,
    ) -> Result<(Arc<dyn PageHandle>, OwnedMutexGuard<()>), String> {
        // Reject empty session id BEFORE launching the browser.
        if session_id.is_empty() {
            return Err("browser actions require a session id".to_string());
        }

        self.backend.ensure_ready().await?;
        let (page, action_lock) = self
            .sessions
            .get_or_create_page(session_id, &*self.backend)
            .await?;
        let guard = action_lock.lock_owned().await;
        Ok((page, guard))
    }

    async fn action_navigate(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: url".to_string())?;

        // Block navigation to internal/private IPs (SSRF protection)
        if let Err(reason) = crate::tools::web_fetch::validate_url_for_ssrf(url) {
            return Err(format!("Navigation blocked: {}", reason));
        }

        let (page, _guard) = self.page_for(session_id).await?;

        page.goto(url).await?;

        // Wait briefly for page load
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        Ok(format!("Navigated to {}", url))
    }

    async fn action_screenshot(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let (page, _guard) = self.page_for(session_id).await?;

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

    async fn action_click(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let selector = args
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: selector".to_string())?;

        // Resolve the session's (active) page FIRST so that creating a session's
        // first page does not itself look like a popup. Then snapshot the
        // browser's live targets BEFORE the click so we can detect a popup
        // (target=_blank / window.open) the click may spawn.
        let (page, _guard) = self.page_for(session_id).await?;

        // The clicking session's active target id — the ONLY legitimate opener
        // for a popup we should attribute to this session.
        let clicker_target_id = self.sessions.active_target_id(session_id).await;

        let known_before: Vec<String> = self
            .backend
            .list_targets()
            .await
            .map(|ts| ts.into_iter().map(|t| t.target_id).collect())
            .unwrap_or_default();

        page.click(selector).await?;

        // Brief wait for any navigation/JS (and popup creation) to settle.
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Popup detection: diff the live targets against what the session knew
        // before. A brand-new target is registered as a tab in this session
        // ONLY when its CDP `openerId` is this session's active page — so a
        // target=_blank click never silently leaves later actions stranded on
        // the old implicit page, yet a tab opened by a DIFFERENT session (or
        // independently) is never misattributed to us. The new tab is NOT
        // auto-activated; the current tab stays active unless the caller
        // explicitly switches.
        let new_tab_id = self
            .detect_and_register_popup(session_id, &known_before, clicker_target_id.as_deref())
            .await;

        match new_tab_id {
            Some(tab_id) => Ok(format!(
                "Clicked element '{}' (opened new tab: {})",
                selector, tab_id
            )),
            None => Ok(format!("Clicked element '{}'", selector)),
        }
    }

    /// After an action that may spawn a popup, diff the browser's live targets
    /// against `known_before`. Register the FIRST net-new target whose CDP
    /// `openerId` equals `clicker_target_id` (the clicking session's active
    /// page) as a tab in the session (not active) and return its opaque tab id.
    ///
    /// A net-new target with a DIFFERENT opener — or no opener — is NOT
    /// attributed to this session: under concurrent timing it belongs to
    /// another session or was opened independently, and binding it here would be
    /// a cross-session info leak (the clicker could then switch/read its page).
    /// Returns `None` when no eligible new target appeared, when this session
    /// has no resolvable active target, or when the diff couldn't be computed.
    async fn detect_and_register_popup(
        &self,
        session_id: &str,
        known_before: &[String],
        clicker_target_id: Option<&str>,
    ) -> Option<String> {
        // Without a known active target for the clicker, we cannot prove a
        // popup's opener belongs to this session — refuse to attribute anything.
        let clicker_target_id = clicker_target_id?;

        let targets = self.backend.list_targets().await.ok()?;
        for t in targets {
            if known_before.iter().any(|k| k == &t.target_id) {
                continue;
            }
            // Only attribute a net-new target whose opener is THIS session's
            // active page. Any other opener (a different session's tab) or no
            // opener at all is rejected — never bound into this session.
            if t.opener_id.as_deref() != Some(clicker_target_id) {
                continue;
            }
            // The popup is ours. Bind a page handle to it so the session can
            // operate on it later, then register it.
            let page = self.backend.page_for_target(&t.target_id).await.ok()?;
            let registered = self
                .sessions
                .add_tab(
                    session_id,
                    &t.target_id,
                    page,
                    t.url.clone(),
                    t.title.clone(),
                    /* make_active */ false,
                )
                .await;
            if let Some(id) = registered {
                return Some(id);
            }
        }
        None
    }

    async fn action_fill(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let selector = args
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: selector".to_string())?;
        let value = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: value".to_string())?;

        let (page, _guard) = self.page_for(session_id).await?;

        page.replace_text(selector, value).await?;

        tracing::info!(
            action = "fill",
            selector,
            value_bytes = value.len(),
            "browser fill"
        );

        Ok(format!("Filled '{}'", selector))
    }

    async fn action_get_text(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let (page, _guard) = self.page_for(session_id).await?;

        let text = if let Some(selector) = args.get("selector").and_then(|v| v.as_str()) {
            page.inner_text(selector).await?
        } else {
            page.body_text().await?
        };

        // Truncate if very long
        let text = crate::utils::truncate_with_note(&text, 4000);

        Ok(text)
    }

    async fn action_execute_js(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let script = args
            .get("script")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: script".to_string())?;

        let (page, _guard) = self.page_for(session_id).await?;

        let result = page.evaluate(script).await?;

        let value_str = match result {
            Some(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| format!("{:?}", v)),
            None => "(no return value)".to_string(),
        };

        let value_str = crate::utils::truncate_with_note(&value_str, 4000);

        Ok(value_str)
    }

    async fn action_wait(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let selector = args
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: selector".to_string())?;
        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(10);

        let (page, _guard) = self.page_for(session_id).await?;

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

    /// `list_tabs`: render this session's tabs — opaque id, title, REDACTED
    /// origin (never the full URL — paths/queries can carry secrets), and which
    /// is active. Ensures the session has at least one tab first (so a fresh
    /// session reports its single page rather than "no tabs").
    async fn action_list_tabs(&self, session_id: &str) -> Result<String, String> {
        // Touch page_for to guarantee the session exists with its first tab.
        let (_page, _guard) = self.page_for(session_id).await?;

        let tabs = self.sessions.list_tabs(session_id).await;
        if tabs.is_empty() {
            return Ok("No open tabs.".to_string());
        }
        Ok(Self::format_tab_list(&tabs))
    }

    fn format_tab_list(tabs: &[TabView]) -> String {
        let mut out = format!("Open tabs ({}):", tabs.len());
        for tab in tabs {
            let marker = if tab.active { " [active]" } else { "" };
            let title = tab.title.as_deref().unwrap_or("(untitled)");
            let origin = tab
                .url
                .as_deref()
                .map(redact_origin)
                .filter(|o| !o.is_empty())
                .unwrap_or_else(|| "(no url)".to_string());
            out.push_str(&format!(
                "\n- {}{}: \"{}\" — {}",
                tab.tab_id, marker, title, origin
            ));
        }
        out
    }

    /// `new_tab`: open a new tab (a new page in this session's context),
    /// optionally navigating it to `url` (SSRF-validated). The new tab becomes
    /// active, since opening a tab implies you want to use it. Returns its
    /// opaque tab id.
    async fn action_new_tab(&self, args: &Value, session_id: &str) -> Result<String, String> {
        // Ensure the session exists (and has its first tab) before adding more.
        let (_page, _guard) = self.page_for(session_id).await?;

        let url = args.get("url").and_then(|v| v.as_str());
        if let Some(url) = url {
            if let Err(reason) = crate::tools::web_fetch::validate_url_for_ssrf(url) {
                return Err(format!("Navigation blocked: {}", reason));
            }
        }

        let (target_id, page) = self.backend.create_page().await?;
        if let Some(url) = url {
            page.goto(url).await?;
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
        let current_url = page.url().await;

        let tab_id = self
            .sessions
            .add_tab(
                session_id,
                &target_id,
                page,
                current_url,
                None,
                /* make_active */ true,
            )
            .await
            .ok_or_else(|| "failed to register new tab for this session".to_string())?;

        match url {
            Some(url) => Ok(format!("Opened new tab {} at {}", tab_id, url)),
            None => Ok(format!("Opened new tab {} (active)", tab_id)),
        }
    }

    /// `switch_tab`: make `tab_id` the session's active tab. The id is validated
    /// to belong to THIS session — a tab id from another session is rejected.
    async fn action_switch_tab(&self, args: &Value, session_id: &str) -> Result<String, String> {
        // Ensure the session exists before validating ownership.
        let (_page, _guard) = self.page_for(session_id).await?;

        let tab_id = args
            .get("tab_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: tab_id".to_string())?;

        let view = self.sessions.switch_tab(session_id, tab_id).await?;
        let origin = view
            .url
            .as_deref()
            .map(redact_origin)
            .filter(|o| !o.is_empty())
            .unwrap_or_else(|| "(no url)".to_string());
        Ok(format!("Switched to tab {} — {}", view.tab_id, origin))
    }

    /// `close_tab`: close `tab_id` (validated to belong to this session) and
    /// report the new active tab, if any remains.
    async fn action_close_tab(&self, args: &Value, session_id: &str) -> Result<String, String> {
        // Ensure the session exists before validating ownership.
        let (_page, _guard) = self.page_for(session_id).await?;

        let tab_id = args
            .get("tab_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: tab_id".to_string())?;

        let (target_id, new_active) = self.sessions.close_tab(session_id, tab_id).await?;

        // Best-effort backend close; the tab is already removed from the session
        // so a backend failure doesn't leave a dangling session reference.
        if let Err(e) = self.backend.close_target(&target_id).await {
            warn!(tab_id, error = %e, "backend close_target failed after session removal");
        }

        match new_active {
            Some(active) => Ok(format!(
                "Closed tab {}. Active tab is now {}.",
                tab_id, active
            )),
            None => Ok(format!(
                "Closed tab {}. No tabs remain open in this session.",
                tab_id
            )),
        }
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
            "description": "Control a browser for web interactions. Actions: navigate (go to URL), screenshot (capture page as photo), click (click element — reports a new tab id if the click opened one), fill (type into input), get_text (extract text), execute_js (run JavaScript), wait (wait for element), list_tabs (list this session's open tabs with their ids), new_tab (open and switch to a new tab, optionally at a url), switch_tab (make a tab active by its id), close_tab (close a tab by its id), set_mode (switch between 'visible' and 'headless' — use visible for sites that block headless browsers), close (end session). The browser persists across calls for multi-step workflows. Tab ids are opaque tokens returned by list_tabs/new_tab; do not guess them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "screenshot", "click", "fill", "get_text", "execute_js", "wait", "list_tabs", "new_tab", "switch_tab", "close_tab", "set_mode", "close"],
                        "description": "The browser action to perform"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (for 'navigate', or optionally for 'new_tab')"
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
                    },
                    "tab_id": {
                        "type": "string",
                        "description": "Opaque tab id from list_tabs/new_tab (required for 'switch_tab' and 'close_tab')"
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
            "navigate" => self.action_navigate(&args, session_id).await,
            "screenshot" => self.action_screenshot(&args, session_id).await,
            "click" => self.action_click(&args, session_id).await,
            "fill" => self.action_fill(&args, session_id).await,
            "get_text" => self.action_get_text(&args, session_id).await,
            "execute_js" => self.action_execute_js(&args, session_id).await,
            "wait" => self.action_wait(&args, session_id).await,
            "list_tabs" => self.action_list_tabs(session_id).await,
            "new_tab" => self.action_new_tab(&args, session_id).await,
            "switch_tab" => self.action_switch_tab(&args, session_id).await,
            "close_tab" => self.action_close_tab(&args, session_id).await,
            "set_mode" => self.action_set_mode(&args).await,
            "close" => self.action_close().await,
            _ => Err(format!(
                "Unknown browser action: '{}'. Valid actions: navigate, screenshot, click, fill, get_text, execute_js, wait, list_tabs, new_tab, switch_tab, close_tab, set_mode, close",
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
            // list_tabs just reads the session's tab set — pure observation.
            Some("list_tabs") => ToolCallSemantics::observation(),
            // new_tab/switch_tab change which page subsequent actions target,
            // mirroring navigate's observation classification (they don't mutate
            // page content, they reposition the session).
            Some("new_tab" | "switch_tab") => ToolCallSemantics::observation(),
            Some("click" | "fill" | "execute_js") => ToolCallSemantics::mutation(),
            // close_tab tears down session state — administrative, like close.
            Some("close" | "set_mode" | "close_tab") => ToolCallSemantics::administrative(),
            _ => ToolCallSemantics::mutation(),
        }
    }
}
