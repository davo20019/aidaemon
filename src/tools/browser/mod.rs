use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::BrowserConfig;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::tools::ApprovalBroker;
use crate::traits::{
    MessageAttachment, Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::{ApprovalResponse, MediaKind, MediaMessage};

use policy::BrowserRiskClass;

mod backend;
pub mod policy;
mod session;
#[cfg(test)]
mod tests;

use backend::{BrowserBackend, ChromiumoxideBackend, PageHandle};
use session::{BrowserSessionRegistry, TabView};

use tokio::sync::OwnedMutexGuard;

/// Default time the user has to respond to a browser approval prompt before the
/// action is auto-denied (fail safe). Matches the terminal/config approval
/// window. Overridable in tests so the timeout path runs in milliseconds.
const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

/// Maximum allowed size of a `script` argument for `execute_js`. Generous for
/// legitimate automation (64 KiB covers virtually any real workflow) while
/// bounding potential abuse via enormous payloads.
const MAX_SCRIPT_BYTES: usize = 64 * 1024;

/// Patterns whose presence in a script argument means the script is attempting
/// to use a privileged browser-management API that bypasses the session/tab
/// model or the approval boundary. Scripts matching any of these are rejected
/// before evaluation.
///
/// Rationale for each entry:
/// - `window.open` — spawns tabs outside the BrowserTool session/tab model,
///   making them invisible to the registry and unaccountable to the caller.
/// - `chrome.` — the chrome.* namespace (chrome.debugger, chrome.management,
///   chrome.tabs, chrome.runtime, etc.) exposes privileged extension/DevTools
///   APIs that can detach the debugger, enumerate/modify tabs across all
///   sessions, or exfiltrate data via cross-context messaging. Any access to
///   this namespace is blocked.
const JS_DENYLIST: &[&str] = &["window.open", "chrome."];

/// Validate script constraints before the approval gate so that a doomed
/// script is never sent for user approval and never touches the backend.
///
/// Returns `Ok(())` when the script passes all checks, or `Err(reason)` with
/// a user-facing error message when a check fails. The reason MUST NOT echo
/// the script body — it names only the violated constraint.
fn validate_script_constraints(script: &str) -> Result<(), String> {
    // 1. Size cap: reject scripts larger than MAX_SCRIPT_BYTES.
    let byte_len = script.len();
    if byte_len > MAX_SCRIPT_BYTES {
        return Err(format!(
            "Script too large: {} bytes (max {}). Split the work into smaller steps.",
            byte_len, MAX_SCRIPT_BYTES
        ));
    }

    // 2. Browser-management API denylist: reject scripts referencing privileged
    //    browser-management APIs that bypass the session/tab model. Each pattern
    //    is specific enough (not a natural-language single word) that substring
    //    matching is appropriate per the project's keyword-matching guidelines.
    for &pattern in JS_DENYLIST {
        if script.contains(pattern) {
            return Err(format!(
                "Script uses a disallowed browser-management API ('{}' is not permitted). \
                 Use BrowserTool tab actions (new_tab, switch_tab, close_tab) for tab management.",
                pattern
            ));
        }
    }

    Ok(())
}

/// The condition the `wait` action polls for. `Present` is the default and
/// preserves the historical behavior (element exists in the DOM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitCondition {
    /// Element matching the selector exists in the DOM (default).
    Present,
    /// Element is present AND laid out / not hidden via CSS.
    Visible,
    /// Element's `disabled` property is falsy.
    Enabled,
    /// Element is absent OR laid out as hidden.
    Hidden,
    /// Element's text contains the provided needle.
    TextContains,
}

impl WaitCondition {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "present" => Ok(Self::Present),
            "visible" => Ok(Self::Visible),
            "enabled" => Ok(Self::Enabled),
            "hidden" => Ok(Self::Hidden),
            "text_contains" => Ok(Self::TextContains),
            other => Err(format!(
                "Invalid wait condition '{}'. Valid: present, visible, enabled, hidden, text_contains",
                other
            )),
        }
    }

    fn success_message(self, selector: &str, needle: Option<&str>) -> String {
        match self {
            Self::Present => format!("Element '{}' found", selector),
            Self::Visible => format!("Element '{}' is visible", selector),
            Self::Enabled => format!("Element '{}' is enabled", selector),
            Self::Hidden => format!("Element '{}' is hidden", selector),
            Self::TextContains => format!(
                "Element '{}' text contains the expected value ({} chars)",
                selector,
                needle.unwrap_or("").len()
            ),
        }
    }

    fn timeout_message(self, selector: &str, needle: Option<&str>, secs: u64) -> String {
        let what = match self {
            Self::Present => format!("element '{}' not found", selector),
            Self::Visible => format!("element '{}' not visible", selector),
            Self::Enabled => format!("element '{}' not enabled", selector),
            Self::Hidden => format!("element '{}' still visible", selector),
            Self::TextContains => format!(
                "element '{}' text did not contain the expected value ({} chars)",
                selector,
                needle.unwrap_or("").len()
            ),
        };
        format!("Timeout: {} after {}s", what, secs)
    }
}

/// Decision returned by the approval gate. `Allow` lets the action reach the
/// backend; `Deny` blocks it with a user-facing reason BEFORE any page/backend
/// method is touched.
enum GateDecision {
    Allow,
    Deny(String),
}

/// The parsed, approval-relevant fields of a single browser tool call. Bundled
/// so the gate and prompt builder take one borrow instead of many positional
/// args. The `value` (fill text) is deliberately NOT carried here — it must
/// never reach the prompt.
struct ActionArgs<'a> {
    action: &'a str,
    url: Option<&'a str>,
    selector: Option<&'a str>,
    script: Option<&'a str>,
    tab_id: Option<&'a str>,
    session_id: &'a str,
}

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

/// Redact a URL for DISPLAY in a screenshot caption / action result: keep the
/// scheme, host, and PATH (useful context for the user) but strip the query
/// string and fragment, which can carry session tokens, auth codes, or other
/// secrets. Anything from the first `?` or `#` onward is dropped.
///
/// Examples:
/// - `https://host.com/a/b?token=SECRET#frag` → `https://host.com/a/b`
/// - `https://host.com/path`                  → `https://host.com/path`
/// - `about:blank`                            → `about:blank`
fn redact_url_for_display(url: &str) -> String {
    let url = url.trim();
    let cut = url.find(['?', '#']).unwrap_or(url.len());
    url[..cut].to_string()
}

/// User-facing description of a browser action for approval prompts.
/// Never includes fill values or script bodies.
fn format_browser_approval_prompt(
    action: &str,
    origin: &str,
    selector: Option<&str>,
    tab_id: Option<&str>,
    script_len: Option<usize>,
    _risk: &policy::BrowserActionRisk,
) -> String {
    match action {
        "list_tabs" => return "List open browser tabs".to_string(),
        "close" => return "Close browser".to_string(),
        "close_tab" => {
            return format!("Close browser tab {}", tab_id.unwrap_or("?"));
        }
        "navigate" => return format!("Open website: {origin}"),
        "new_tab" => return format!("Open new tab: {origin}"),
        "switch_tab" => {
            return format!("Switch to browser tab {}", tab_id.unwrap_or("?"));
        }
        "execute_js" => {
            let bytes = script_len.unwrap_or(0);
            return format!("Run JavaScript on {origin} ({bytes} bytes)");
        }
        "click" => {
            if let Some(sel) = selector {
                return format!("Click \"{sel}\" on {origin}");
            }
            return format!("Click on {origin}");
        }
        "fill" => {
            if let Some(sel) = selector {
                return format!("Fill in \"{sel}\" on {origin}");
            }
            return format!("Fill in a form field on {origin}");
        }
        "get_text" => return format!("Read page text from {origin}"),
        "screenshot" => return format!("Take screenshot of {origin}"),
        "wait" => return format!("Wait on {origin}"),
        "set_mode" => return format!("Change browser mode on {origin}"),
        _ => return format!("Browser action on {origin}"),
    }
}

/// Telegram-safe upper bounds for an outbound screenshot, enforced BEFORE the
/// image is enqueued onto the media channel. A VIEWPORT capture at the default
/// window size is always within these; a `full_page` capture of a tall page can
/// exceed them, in which case we return a clear error instead of enqueuing an
/// image that the channel would silently reject (the production bug:
/// `PHOTO_INVALID_DIMENSIONS`, dropped with only a `warn!`).
///
/// Telegram rejects `sendPhoto` when width+height > 10000, when the aspect ratio
/// exceeds ~20:1, or when the file is larger than 10MB. We stay safely under each.
const MAX_SCREENSHOT_DIM_SUM: u32 = 9000;
const MAX_SCREENSHOT_RATIO: u32 = 18;
const MAX_SCREENSHOT_BYTES: usize = 9_000_000;

/// Upper bound on a screenshot delivered as a DOCUMENT (Telegram `sendDocument`
/// accepts large PNGs — ~50MB, no pixel-dimension limit — where `sendPhoto`
/// rejects them). Kept safely under Telegram's 50MB bot limit. A capture over
/// the photo caps but under this is sent as a file; over this it is refused.
const MAX_SCREENSHOT_DOCUMENT_BYTES: usize = 49 * 1024 * 1024;

/// How an oversized-vs-in-cap screenshot should be delivered, decided purely by
/// encoded byte length + whether it is within the photo caps. Pulled out as a
/// pure function so all three branches are unit-testable without allocating a
/// real 49MB buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScreenshotDelivery {
    /// Within the photo caps → deliver inline as a `MediaKind::Photo`.
    Photo,
    /// Over the photo caps but within `MAX_SCREENSHOT_DOCUMENT_BYTES` → deliver
    /// as a `MediaKind::Document` (file attachment).
    Document,
    /// Over `MAX_SCREENSHOT_DOCUMENT_BYTES` → cannot be delivered at all.
    TooLarge,
}

/// Decide how to deliver a screenshot from its encoded byte length and whether
/// it exceeded the photo caps (`oversize_reason`: `None` = within photo caps,
/// `Some(_)` = over them — mirrors [`screenshot_oversize_reason`]'s output).
fn screenshot_delivery_kind(
    byte_len: usize,
    oversize_reason: Option<String>,
) -> ScreenshotDelivery {
    match oversize_reason {
        None => ScreenshotDelivery::Photo,
        Some(_) if byte_len <= MAX_SCREENSHOT_DOCUMENT_BYTES => ScreenshotDelivery::Document,
        Some(_) => ScreenshotDelivery::TooLarge,
    }
}

/// Parse the pixel dimensions of a PNG from its header WITHOUT decoding the
/// image (no `image` crate dependency). A PNG is the 8-byte signature followed
/// by the IHDR chunk; width is the big-endian `u32` at bytes `[16..20]` and
/// height at `[20..24]`. Returns `None` if the buffer is too short or does not
/// begin with the PNG signature.
fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    if bytes.len() < 24 || bytes[..8] != PNG_SIGNATURE {
        return None;
    }
    let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    Some((width, height))
}

/// Whether a captured screenshot is within the channel-safe caps. Returns the
/// offending `(width, height)` when the PNG header is parseable AND the image is
/// too large by dimension-sum, aspect ratio, or encoded byte length. A buffer
/// whose PNG header is unparseable is judged on byte length alone (we can't know
/// its dimensions, but a too-large encoding is still rejected).
fn screenshot_oversize_reason(bytes: &[u8]) -> Option<String> {
    if bytes.len() > MAX_SCREENSHOT_BYTES {
        return Some(format!(
            "encoded size {} bytes exceeds the {} byte limit",
            bytes.len(),
            MAX_SCREENSHOT_BYTES
        ));
    }
    if let Some((w, h)) = png_dimensions(bytes) {
        if w.saturating_add(h) > MAX_SCREENSHOT_DIM_SUM {
            return Some(format!("dimensions {}x{}", w, h));
        }
        let (lo, hi) = if w >= h { (h, w) } else { (w, h) };
        if lo > 0 && hi / lo > MAX_SCREENSHOT_RATIO {
            return Some(format!("aspect ratio of {}x{}", w, h));
        }
    }
    None
}

/// Result of a browser action dispatch — text for the LLM plus optional vision attachments.
struct DispatchResult {
    text: String,
    attachments: Vec<MessageAttachment>,
}

impl DispatchResult {
    fn text_only(text: String) -> Self {
        Self {
            text,
            attachments: Vec::new(),
        }
    }
}

pub struct BrowserTool {
    backend: Arc<dyn BrowserBackend>,
    media_tx: mpsc::Sender<MediaMessage>,
    /// Shared inbox for persisting screenshots for agent vision context.
    inbox_dir: PathBuf,
    /// Per-session page state, keyed by trusted internal `_session_id`.
    sessions: BrowserSessionRegistry,
    /// Transport used to prompt the user Allow/Deny at action time.
    ///
    /// `None` means no approval channel is wired (only possible in tests via
    /// `with_backend`); in that case every action that would require approval
    /// fails safe to Deny without touching the backend.
    approval_tx: Option<ApprovalBroker>,
    /// How long to wait for an approval response before auto-denying.
    approval_timeout: Duration,
    /// Bounded navigation/DOM-ready timeout (resolved + clamped from config).
    /// Used by `action_navigate` after `goto` and by the `action_click`
    /// nav-race when a click triggers a navigation.
    nav_timeout: Duration,
    /// Default `wait`-action element-poll timeout (resolved + clamped). A
    /// per-call `timeout_secs` arg overrides this (also clamped to the same
    /// bound).
    element_timeout: Duration,
    /// Overall per-action ceiling for the click nav-race. The short stable-DOM
    /// fallback for a NON-navigating click is a small fraction of this so a
    /// click that doesn't navigate returns fast.
    action_timeout: Duration,
}

/// Upper bound on the element-poll timeout, mirroring `BrowserConfig`'s
/// `element_timeout` clamp, applied to a per-call `timeout_secs` override.
const MAX_ELEMENT_TIMEOUT_SECS: u64 = 120;

/// Interval between element-state polls in the `wait` action. With a paused
/// clock, tests advance by this step to drive the loop deterministically.
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Short stable-DOM settle used by the click nav-race for a NON-navigating
/// click: long enough for click-side JS to run, short enough that a plain click
/// returns fast. Bounded well under `action_timeout`.
const CLICK_SETTLE: Duration = Duration::from_millis(300);

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
        approval_tx: ApprovalBroker,
        inbox_dir: impl Into<PathBuf>,
    ) -> Result<Self, String> {
        // Resolve + clamp the bounded timeouts BEFORE `config` is moved into the
        // backend.
        let nav_timeout = config.nav_timeout();
        let element_timeout = config.element_timeout();
        let action_timeout = config.action_timeout();
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
            inbox_dir: inbox_dir.into(),
            sessions: BrowserSessionRegistry::new(),
            approval_tx: Some(approval_tx),
            approval_timeout: DEFAULT_APPROVAL_TIMEOUT,
            nav_timeout,
            element_timeout,
            action_timeout,
        })
    }

    /// Test-only constructor that injects an arbitrary backend (e.g. the mock)
    /// with NO approval channel — exercises the missing-channel (fail-safe Deny)
    /// path for actions that require approval.
    #[cfg(test)]
    pub fn with_backend(
        backend: Arc<dyn BrowserBackend>,
        media_tx: mpsc::Sender<MediaMessage>,
    ) -> Self {
        Self {
            backend,
            media_tx,
            inbox_dir: std::env::temp_dir().join("aidaemon-browser-test-inbox"),
            sessions: BrowserSessionRegistry::new(),
            approval_tx: None,
            approval_timeout: DEFAULT_APPROVAL_TIMEOUT,
            nav_timeout: Duration::from_secs(30),
            element_timeout: Duration::from_secs(10),
            action_timeout: Duration::from_secs(30),
        }
    }

    /// Test-only constructor that injects a backend AND an approval broker plus a
    /// (short) approval timeout, so approval allow/deny/timeout paths are
    /// exercisable in milliseconds.
    #[cfg(test)]
    pub fn with_backend_and_approval(
        backend: Arc<dyn BrowserBackend>,
        media_tx: mpsc::Sender<MediaMessage>,
        approval_tx: ApprovalBroker,
        approval_timeout: Duration,
    ) -> Self {
        Self {
            backend,
            media_tx,
            inbox_dir: std::env::temp_dir().join("aidaemon-browser-test-inbox"),
            sessions: BrowserSessionRegistry::new(),
            approval_tx: Some(approval_tx),
            approval_timeout,
            nav_timeout: Duration::from_secs(30),
            element_timeout: Duration::from_secs(10),
            action_timeout: Duration::from_secs(30),
        }
    }

    /// Test-only: override the resolved navigation/element/action timeouts so
    /// the bounded-wait paths run under a paused fake clock without depending on
    /// the production defaults.
    #[cfg(test)]
    pub fn with_timeouts(mut self, nav: Duration, element: Duration, action: Duration) -> Self {
        self.nav_timeout = nav;
        self.element_timeout = element;
        self.action_timeout = action;
        self
    }

    /// Read the active tab's last-known URL (cached in the session registry)
    /// WITHOUT touching the backend, so the approval prompt can show a redacted
    /// origin before any page method runs. Returns `None` if the session has no
    /// active tab yet or its URL is unknown.
    async fn active_origin_for_prompt(&self, session_id: &str) -> Option<String> {
        let tabs = self.sessions.list_tabs(session_id).await;
        let active = tabs.iter().find(|t| t.active).or_else(|| tabs.first())?;
        active
            .url
            .as_deref()
            .map(redact_origin)
            .filter(|o| !o.is_empty())
    }

    /// Build the secret-safe approval prompt string for an action.
    ///
    /// NEVER include: the `fill` value, the full `execute_js` script (only
    /// "JavaScript execution" + byte length), or full URLs with path/query/
    /// fragment (origins are redacted via [`redact_origin`]).
    async fn build_prompt(
        &self,
        args: &ActionArgs<'_>,
        risk: &policy::BrowserActionRisk,
    ) -> String {
        // Origin: for url-bearing actions use the url arg; otherwise the active
        // tab's last-known origin (or "current page" when unknown).
        let origin = match args.url {
            Some(u) => {
                let r = redact_origin(u);
                if r.is_empty() {
                    "current page".to_string()
                } else {
                    r
                }
            }
            None => self
                .active_origin_for_prompt(args.session_id)
                .await
                .unwrap_or_else(|| "current page".to_string()),
        };

        format_browser_approval_prompt(
            args.action,
            &origin,
            args.selector,
            args.tab_id,
            args.script.map(|s| s.len()),
            risk,
        )
    }

    /// Send an approval request and await the user's decision, failing safe to
    /// `Deny` on a closed channel or timeout. Returns `None` only when no
    /// approval channel is wired (the caller treats that as a fail-safe Deny).
    async fn request_approval(
        &self,
        command: String,
        risk_level: RiskLevel,
        warnings: Vec<String>,
        session_id: &str,
    ) -> Option<ApprovalResponse> {
        let broker = self.approval_tx.as_ref()?;
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        if broker
            .send(ApprovalRequest {
                command,
                session_id: session_id.to_string(),
                risk_level,
                warnings,
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: Default::default(),
            })
            .await
            .is_err()
        {
            warn!("browser approval channel closed; denying action");
            return Some(ApprovalResponse::Deny);
        }
        match tokio::time::timeout(self.approval_timeout, response_rx).await {
            Ok(Ok(resp)) => Some(resp),
            Ok(Err(_)) => {
                warn!("browser approval response channel closed; denying action");
                Some(ApprovalResponse::Deny)
            }
            Err(_) => {
                warn!("browser approval request timed out; denying action");
                Some(ApprovalResponse::Deny)
            }
        }
    }

    /// The approval gate. Runs BEFORE any backend/page method is touched, so a
    /// denied action can never reach the browser. Returns [`GateDecision::Allow`]
    /// only when the action is permitted.
    ///
    /// Per-class rules:
    /// - `Observation` (get_text/screenshot/wait/list_tabs): never prompt.
    /// - `Administrative` (close/close_tab/set_mode): never prompt — local
    ///   lifecycle / mode switch, not a consequential web side effect.
    /// - `sensitive || consequential` (every `execute_js`, plus consequential
    ///   click/fill): point-of-action — ALWAYS prompt, every call, regardless of
    ///   any prior session approval. A non-Deny response allows ONLY this single
    ///   action and NEVER records persistent/session approval.
    /// - `Navigation` / ordinary `Mutation`: session-level. Allowed without a
    ///   prompt once the session is approved; otherwise prompt. `AllowOnce`
    ///   allows just this action; `AllowSession`/`AllowAlways` also mark the
    ///   session approved so subsequent ordinary actions don't re-prompt.
    /// - Missing approval channel + an action that needs approval → fail safe to
    ///   Deny (observations/administrative still run).
    async fn approval_gate(&self, args: &ActionArgs<'_>) -> GateDecision {
        let action = args.action;
        let session_id = args.session_id;
        let risk = policy::classify(action, args.selector, args.script);

        // Free actions: never prompt.
        if matches!(
            risk.class,
            BrowserRiskClass::Observation | BrowserRiskClass::Administrative
        ) {
            return GateDecision::Allow;
        }

        let point_of_action = risk.sensitive || risk.consequential;

        // Session-level fast path: an already-approved session skips the prompt
        // for ordinary navigation/mutation — but NEVER for point-of-action.
        if !point_of_action && self.sessions.is_session_approved(session_id).await {
            return GateDecision::Allow;
        }

        // From here we need to prompt. If no channel is wired, fail safe.
        if self.approval_tx.is_none() {
            warn!(
                action,
                "browser action requires approval but no approval channel is wired; denying"
            );
            return GateDecision::Deny(
                "Approval required, but no approval channel is available. Action denied."
                    .to_string(),
            );
        }

        let risk_level = if point_of_action {
            RiskLevel::High
        } else {
            RiskLevel::Medium
        };
        let mut warnings = Vec::new();
        if risk.sensitive {
            warnings.push("This can read or access private data on the page.".to_string());
        }
        if risk.consequential {
            warnings.push(
                "This may submit forms, make purchases, delete data, or send messages.".to_string(),
            );
        }
        let command = self.build_prompt(args, &risk).await;

        let resp = self
            .request_approval(command, risk_level, warnings, session_id)
            .await;

        match resp {
            // No channel: already handled above, but keep the fail-safe.
            None => GateDecision::Deny(
                "Approval required, but no approval channel is available. Action denied."
                    .to_string(),
            ),
            Some(ApprovalResponse::Deny) => GateDecision::Deny("Denied by user.".to_string()),
            Some(ApprovalResponse::AllowOnce) => GateDecision::Allow,
            Some(ApprovalResponse::AllowSession) | Some(ApprovalResponse::AllowAlways) => {
                // Point-of-action approvals NEVER persist: each consequential
                // action / execute_js must be approved on its own. Only ordinary
                // navigation/mutation marks the session approved.
                if !point_of_action {
                    self.sessions.mark_session_approved(session_id).await;
                }
                GateDecision::Allow
            }
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

    /// Defense-in-depth gate for observation/JS actions: read the page's LIVE
    /// committed URL and re-validate it against the shared private-network policy
    /// BEFORE reading/capturing/evaluating any page content.
    ///
    /// Per-request subresource interception is deferred (see the `PageHandle::url`
    /// doc + the `#[ignore]`d feasibility stub), so a page can still reach a
    /// blocked host AFTER load via a meta-refresh, JS-driven `location` change, or
    /// nested frame. The final-URL revalidation in `action_navigate`/`action_new_tab`
    /// only catches redirects at navigation time — it cannot catch a post-load
    /// redirect. This helper closes that gap for the exfiltration vectors named in
    /// the finding: by re-checking the live URL right before each observation/JS
    /// action, a post-load redirect to a private host cannot be read out,
    /// screenshotted, or evaluated.
    ///
    /// On block, returns a structured host-CLASS error only — never the URL,
    /// path, query, or any embedded credentials.
    async fn ensure_current_url_allowed(&self, page: &Arc<dyn PageHandle>) -> Result<(), String> {
        if let Some(current_url) = page.url().await {
            if let Err(blocked) = policy::validate_network_url(&current_url) {
                warn!(
                    class = blocked.class.label(),
                    "observation/JS action refused: current page is on a blocked host"
                );
                return Err(format!(
                    "Action blocked: current page is a {}",
                    blocked.class.label()
                ));
            }
        }
        Ok(())
    }

    async fn action_navigate(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: url".to_string())?;

        // Pre-flight SSRF check on the requested URL. The error names ONLY the
        // host class (loopback/private/link-local) — never the URL, path,
        // query, or credentials — via the shared policy seam.
        if let Err(blocked) = policy::validate_network_url(url) {
            return Err(blocked.message());
        }

        let (page, _guard) = self.page_for(session_id).await?;

        page.goto(url).await?;

        // Wait for the navigation lifecycle / DOM-ready signal instead of a
        // blind fixed sleep, bounded by `nav_timeout`. A page that never fires
        // `load` does NOT hard-fail navigate — `wait_for_navigation` returns
        // best-effort on its internal timeout. A genuine connection-class error
        // here propagates so `dispatch_with_recovery` can classify it.
        if let Err(e) = page.wait_for_navigation(self.nav_timeout).await {
            // If the wait itself failed for a connection reason, the recovery
            // wrapper handles it; surface it. The local timeout never reaches
            // here (it returns Ok). See the connection-reset note below.
            if backend::is_connection_error(&e) {
                return Err(e);
            }
            warn!(error = %e, "navigation wait reported a non-connection error; proceeding to URL revalidation");
        }

        // Revalidate the FINAL committed URL. A server-side redirect can land on
        // a blocked host (e.g. a public redirector → http://127.0.0.1/...) even
        // though the requested URL was public and per-request subresource
        // interception is deferred (see the Task 8 report / CDP feasibility
        // note). If the committed URL is blocked, treat the navigation as
        // blocked and surface ONLY the host class — never the committed URL,
        // which may carry a path/query/token.
        if let Some(final_url) = page.url().await {
            if let Err(blocked) = policy::validate_network_url(&final_url) {
                warn!(
                    class = blocked.class.label(),
                    "navigation landed on a blocked host after redirect; blocking"
                );
                // Neutralize the committed state: the page is currently sitting on
                // the blocked host, so a subsequent get_text/screenshot/execute_js
                // could read/capture/evaluate the blocked content even though we're
                // about to return an error. Reset to about:blank so nothing on the
                // blocked host remains observable. Best-effort: a failure here does
                // not change the outcome (the action is still blocked).
                let _ = page.goto("about:blank").await;
                return Err(format!(
                    "Navigation blocked: redirected to a {}",
                    blocked.class.label()
                ));
            }
        }

        Ok(format!("Navigated to {}", url))
    }

    async fn action_screenshot(
        &self,
        args: &Value,
        session_id: &str,
    ) -> Result<DispatchResult, String> {
        // `page_for` already rejects an empty session id before any capture; we
        // additionally guard the media-delivery path below so an empty id can
        // never reach the channel.
        let (page, _guard) = self.page_for(session_id).await?;

        // Defense-in-depth: refuse to capture if the live committed URL is a
        // blocked host (e.g. reached via post-load JS-redirect/meta-refresh).
        self.ensure_current_url_allowed(&page).await?;

        let selector = args.get("selector").and_then(|v| v.as_str());
        // Default to a VIEWPORT capture; full-page must be opted into explicitly.
        // A selector capture ignores full_page (the element bounds define it).
        let full_page = args
            .get("full_page")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let png_bytes = page.screenshot(selector, full_page).await?;

        // Redacted page URL (query + fragment stripped) for the caption AND any
        // URL echoed back in the result string — neither must leak a token.
        let display_url = page
            .url()
            .await
            .map(|u| redact_url_for_display(&u))
            .unwrap_or_else(|| "current page".to_string());
        let caption = format!("Screenshot of {}", display_url);

        // Guard the delivery path: never enqueue media with an empty session id.
        if session_id.is_empty() {
            return Err("browser actions require a session id".to_string());
        }

        let saved_attachment = crate::channels::attachments::save_tool_observation_image(
            &self.inbox_dir,
            &png_bytes,
            "screenshot.png",
            "image/png",
            "browser",
        )
        .map_err(|e| format!("Screenshot captured but failed to save for vision context: {e}"))?;

        // Decide HOW to deliver based on size. A viewport capture is always a
        // Photo. A full-page capture of a long page can exceed Telegram's
        // sendPhoto caps (PHOTO_INVALID_DIMENSIONS); rather than refusing it, we
        // fall back to sendDocument, which accepts large PNGs (~50MB, no pixel
        // limit) — the full page actually arrives as a viewable image file. Only
        // a capture larger than even the document cap is refused.
        let oversize_reason = screenshot_oversize_reason(&png_bytes);
        let delivery = screenshot_delivery_kind(png_bytes.len(), oversize_reason);

        // `as_file` selects the honest, mode-aware success string below.
        let (kind, as_file) = match delivery {
            ScreenshotDelivery::Photo => (MediaKind::Photo { data: png_bytes }, false),
            ScreenshotDelivery::Document => (
                MediaKind::Document {
                    file_path: saved_attachment.local_path.clone(),
                    filename: saved_attachment.filename.clone(),
                },
                true,
            ),
            ScreenshotDelivery::TooLarge => {
                return Err(format!(
                    "Screenshot is too large to deliver even as a file ({} bytes, max {}). \
                     Capture a specific element with a selector.",
                    png_bytes.len(),
                    MAX_SCREENSHOT_DOCUMENT_BYTES
                ));
            }
        };

        // Honest delivery: ask the media listener to report the ACTUAL outcome.
        let (result_tx, result_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        self.media_tx
            .send(MediaMessage {
                session_id: session_id.to_string(),
                caption: caption.clone(),
                kind,
                result_tx: Some(result_tx),
            })
            .await
            .map_err(|e| format!("Failed to send screenshot to chat: {}", e))?;

        // Wait (bounded) for the listener to confirm delivery, then report
        // HONESTLY — never claim "sent" unless the channel actually accepted it.
        let text = match tokio::time::timeout(Duration::from_secs(30), result_rx).await {
            Ok(Ok(Ok(()))) => {
                let base = if as_file {
                    format!(
                        "Screenshot captured and delivered to chat as a file (the full page was \
                         too large for an inline image). {}",
                        caption
                    )
                } else {
                    format!("Screenshot captured and delivered to chat. {}", caption)
                };
                format!(
                    "{base}\nSaved to: {}",
                    saved_attachment.local_path
                )
            }
            Ok(Ok(Err(reason))) => {
                return Err(format!(
                    "Screenshot captured but could NOT be delivered to chat: {}. The image was not sent.",
                    reason
                ))
            }
            Ok(Err(_)) => {
                return Err(
                    "Screenshot captured but delivery could not be confirmed (the delivery channel \
                 was dropped). The image may not have been sent."
                        .to_string(),
                )
            }
            Err(_) => {
                return Err(
                    "Screenshot captured but delivery could not be confirmed within the timeout. \
                 The image may not have been sent."
                        .to_string(),
                )
            }
        };

        Ok(DispatchResult {
            text,
            attachments: vec![saved_attachment],
        })
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

        // Nav-race (replaces the old fixed 500ms sleep): race a navigation
        // signal against a short stable-DOM settle. A click that triggers a
        // navigation waits for that navigation (bounded by `nav_timeout`, itself
        // capped under `action_timeout`); a click that does NOT navigate returns
        // quickly after `CLICK_SETTLE`. `wait_for_navigation` resolves only when
        // a navigation actually completes, so for a non-navigating click the
        // settle timer wins and we return fast.
        let nav_budget = self.nav_timeout.min(self.action_timeout);
        tokio::select! {
            biased;
            _ = tokio::time::sleep(CLICK_SETTLE) => {
                // Non-navigating (or fast) click: settle elapsed first. Return fast.
            }
            nav = page.wait_for_navigation(nav_budget) => {
                // A navigation completed (or its bounded wait returned). Surface a
                // connection-class error so recovery can classify it; otherwise
                // proceed to popup detection.
                if let Err(e) = nav {
                    if backend::is_connection_error(&e) {
                        return Err(e);
                    }
                    warn!(error = %e, "click navigation wait reported a non-connection error");
                }
            }
        }

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
                    // A popup inherits its opener's browser context (which the
                    // session already tracks on its opener tab), so we record no
                    // additional context id here — avoids a double-dispose of the
                    // same context on eviction.
                    /* context_id */
                    None,
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

        // Defense-in-depth: refuse to read if the live committed URL is a blocked
        // host (e.g. reached via post-load JS-redirect/meta-refresh).
        self.ensure_current_url_allowed(&page).await?;

        let text = if let Some(selector) = args.get("selector").and_then(|v| v.as_str()) {
            page.inner_text(selector).await?
        } else {
            page.body_text().await?
        };

        // Truncate if very long.
        let text = crate::utils::truncate_with_note(&text, 4000);

        // Apply secret redaction AFTER truncation. DOM content can contain
        // tokens, API keys, or bearer tokens embedded in the page — these must
        // not reach the user or event persistence in their raw form.
        let text = crate::tools::sanitize::redact_secrets(&text);

        Ok(text)
    }

    async fn action_execute_js(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let script = args
            .get("script")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: script".to_string())?;

        let (page, _guard) = self.page_for(session_id).await?;

        // Defense-in-depth: refuse to evaluate if the live committed URL is a
        // blocked host (e.g. reached via post-load JS-redirect/meta-refresh).
        // This runs AFTER the approval gate (which fires in `call()` before
        // dispatch) but BEFORE the script is evaluated, so an approved execute_js
        // still cannot read out a private host the page redirected to post-load.
        self.ensure_current_url_allowed(&page).await?;

        let result = page.evaluate(script).await?;

        let value_str = match result {
            Some(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| format!("{:?}", v)),
            None => "(no return value)".to_string(),
        };

        let value_str = crate::utils::truncate_with_note(&value_str, 4000);

        // Apply secret redaction AFTER truncation so the redacted form is what
        // reaches the user and event persistence — never the raw secret.
        let value_str = crate::tools::sanitize::redact_secrets(&value_str);

        Ok(value_str)
    }

    async fn action_wait(&self, args: &Value, session_id: &str) -> Result<String, String> {
        let selector = args
            .get("selector")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing required parameter: selector".to_string())?;

        // Condition: defaults to `present` (the historical behavior). Any unknown
        // value is rejected up front so a typo doesn't silently fall through.
        let condition = args
            .get("condition")
            .and_then(|v| v.as_str())
            .unwrap_or("present");
        let condition = WaitCondition::parse(condition)?;

        // For `text_contains` a needle is required (accept either `text` or the
        // shared `value` arg).
        let needle = args
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| args.get("value").and_then(|v| v.as_str()));
        if condition == WaitCondition::TextContains && needle.unwrap_or("").is_empty() {
            return Err(
                "Missing required parameter for condition 'text_contains': text (or value)"
                    .to_string(),
            );
        }

        // Resolve the per-call timeout: a provided `timeout_secs` overrides the
        // configured default, clamped to the same bound the config applies.
        let timeout = match args.get("timeout_secs").and_then(|v| v.as_u64()) {
            Some(secs) => Duration::from_secs(secs.clamp(1, MAX_ELEMENT_TIMEOUT_SECS)),
            None => self.element_timeout,
        };

        let (page, _guard) = self.page_for(session_id).await?;

        let deadline = tokio::time::Instant::now() + timeout;
        let timeout_secs = timeout.as_secs();

        loop {
            // Evaluate the condition once. A connection-class error from a state
            // probe is surfaced so `dispatch_with_recovery` can classify it; a
            // benign "not yet" simply keeps polling.
            match self
                .evaluate_wait_condition(&page, condition, selector, needle)
                .await
            {
                Ok(true) => return Ok(condition.success_message(selector, needle)),
                Ok(false) => {}
                Err(e) => {
                    if backend::is_connection_error(&e) {
                        return Err(e);
                    }
                    // Non-connection probe error (e.g. transient) — treat as
                    // "not satisfied yet" and keep polling within the deadline.
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(condition.timeout_message(selector, needle, timeout_secs));
            }
            tokio::time::sleep(WAIT_POLL_INTERVAL).await;
        }
    }

    /// Evaluate a single `wait` condition against the page. Returns `Ok(true)`
    /// when satisfied, `Ok(false)` when not-yet, `Err` on a probe failure.
    ///
    /// Element state is checked WITHOUT interpolating the selector into a JS
    /// source string (Task 14 rule): `present` uses the CDP DOM query
    /// (`find_element`); `visible`/`hidden`/`enabled` use element-bound constant
    /// predicates via the `PageHandle` state methods; `text_contains` reads the
    /// element's own `inner_text`.
    async fn evaluate_wait_condition(
        &self,
        page: &Arc<dyn PageHandle>,
        condition: WaitCondition,
        selector: &str,
        needle: Option<&str>,
    ) -> Result<bool, String> {
        match condition {
            WaitCondition::Present => Ok(page.find_element(selector).await.is_ok()),
            WaitCondition::Visible => page.is_element_visible(selector).await,
            WaitCondition::Enabled => {
                // Enabled implies present; an absent element reads as not-enabled
                // via the state probe, so no extra presence check is needed.
                page.is_element_enabled(selector).await
            }
            WaitCondition::Hidden => {
                // Hidden == not visible (absent or laid-out-hidden). The
                // visibility probe returns false for an absent element, so the
                // negation covers both cases.
                Ok(!page.is_element_visible(selector).await?)
            }
            WaitCondition::TextContains => {
                let needle = needle.unwrap_or("");
                // A missing element yields no text → not satisfied yet.
                match page.inner_text(selector).await {
                    Ok(text) => Ok(text.contains(needle)),
                    Err(e) if backend::is_connection_error(&e) => Err(e),
                    Err(_) => Ok(false),
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
        // Route through the backend's graceful shutdown (launched →
        // close()+wait()+timeout+fallback; attached → detach without a
        // browser-close command). All cached session pages are now stale handles
        // into a torn-down connection, so drop them too.
        //
        // NOTE (deferred follow-up): DAEMON-EXIT graceful close is NOT wired.
        // Only `close`, `set_mode` (mode change), and idle eviction reuse the
        // backend's graceful-teardown path; nothing calls it on process exit.
        // Wiring a daemon-shutdown hook would require threading a concrete
        // `Arc<BrowserTool>` into `core.rs`'s shutdown handler — out of scope and
        // low value here: on process exit a LAUNCHED Chrome is reclaimed by OS
        // teardown of its temp profile, and an ATTACHED Chrome is unaffected
        // (we never send it a close command). Left as a documented follow-up.
        let result = self.backend.shutdown().await;
        self.sessions.invalidate_all_pages().await;
        result
    }

    /// Dispatch a single action to its handler (no recovery). Pulled out of
    /// `call()` so the recovery wrapper can re-invoke it for an observation retry.
    async fn dispatch_action(
        &self,
        action: &str,
        args: &Value,
        session_id: &str,
    ) -> Result<DispatchResult, String> {
        match action {
            "navigate" => self
                .action_navigate(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "screenshot" => self.action_screenshot(args, session_id).await,
            "click" => self
                .action_click(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "fill" => self
                .action_fill(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "get_text" => self
                .action_get_text(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "execute_js" => self
                .action_execute_js(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "wait" => self
                .action_wait(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "list_tabs" => self
                .action_list_tabs(session_id)
                .await
                .map(DispatchResult::text_only),
            "new_tab" => self
                .action_new_tab(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "switch_tab" => self
                .action_switch_tab(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "close_tab" => self
                .action_close_tab(args, session_id)
                .await
                .map(DispatchResult::text_only),
            "set_mode" => self
                .action_set_mode(args)
                .await
                .map(DispatchResult::text_only),
            "close" => self.action_close().await.map(DispatchResult::text_only),
            _ => Err(format!(
                "Unknown browser action: '{}'. Valid actions: navigate, screenshot, click, fill, get_text, execute_js, wait, list_tabs, new_tab, switch_tab, close_tab, set_mode, close",
                action
            )),
        }
    }

    /// Whether an action is safe to AUTOMATICALLY replay after a connection-class
    /// failure + reconnect. Only observation/navigation/administrative actions
    /// are idempotent enough to re-run blindly. Mutations (`click`, `fill`,
    /// `execute_js`) may have PARTIALLY executed before the disconnect (the CDP
    /// command could have reached Chrome and run before the websocket tore down),
    /// so their state after a disconnect is UNCERTAIN — replaying could double a
    /// submit/purchase/delete. We never auto-replay them.
    ///
    /// Uses the shared `policy::classify` so the observation-vs-mutation boundary
    /// has a single source of truth.
    fn action_is_safe_to_replay(action: &str) -> bool {
        let risk = policy::classify(action, None, None);
        matches!(
            risk.class,
            BrowserRiskClass::Observation
                | BrowserRiskClass::Navigation
                | BrowserRiskClass::Administrative
        )
    }

    /// Run an action with disconnect recovery layered on top of `dispatch_action`.
    ///
    /// On a CONNECTION-CLASS error (per `backend::is_connection_error`) — the
    /// websocket/CDP connection to Chrome died, distinct from an ordinary page
    /// error like "element not found" — recovery splits by idempotency, which is
    /// known HERE (the tool layer) via the action's risk class:
    ///
    /// - **Observation / navigation / administrative (idempotent):** invalidate
    ///   ALL cached session pages (a dead browser kills every session's pages),
    ///   `reconnect()` ONCE, then retry the action ONE time against a fresh page.
    ///   If it fails again, surface that error.
    /// - **Mutation (`click`/`fill`/`execute_js`):** NEVER auto-replay — the
    ///   action may have partially executed before the disconnect (uncertain
    ///   state). Still reconnect + invalidate so the NEXT action works, but
    ///   surface a clear "could not be confirmed; re-issue manually" error.
    ///
    /// A non-connection error is returned verbatim with no reconnect.
    async fn dispatch_with_recovery(
        &self,
        action: &str,
        args: &Value,
        session_id: &str,
    ) -> Result<DispatchResult, String> {
        let first = self.dispatch_action(action, args, session_id).await;
        let err = match first {
            Ok(ok) => return Ok(ok),
            Err(e) => e,
        };

        // Only connection-class failures trigger recovery. A normal page error
        // ("Element not found", "Timeout", ...) is surfaced as-is.
        if !backend::is_connection_error(&err) {
            return Err(err);
        }

        warn!(
            action,
            "browser action hit a connection-class error; attempting recovery"
        );

        // A dead browser invalidates EVERY session's pages — drop them all so
        // the next page resolution mints fresh handles against the new
        // connection. Then reconnect exactly once.
        self.sessions.invalidate_all_pages().await;
        if let Err(reconnect_err) = self.backend.reconnect().await {
            return Err(format!(
                "Browser connection lost and reconnect failed: {}. \
                 The action did not complete; please retry.",
                reconnect_err
            ));
        }

        if Self::action_is_safe_to_replay(action) {
            // Idempotent: retry once against a freshly-minted page.
            info!(action, "retrying idempotent browser action after reconnect");
            return self.dispatch_action(action, args, session_id).await;
        }

        // Mutation: NEVER auto-replay. The connection is restored for subsequent
        // actions, but this action's effect is uncertain.
        warn!(
            action,
            "mutation hit a disconnect; NOT replaying (uncertain state)"
        );
        Err(format!(
            "Browser connection was lost while performing '{}'. The action could NOT be \
             confirmed and may have partially completed — it was NOT retried automatically to \
             avoid duplicating it. The connection has been restored; re-issue the action \
             manually if needed after checking the page state.",
            action
        ))
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
            // Pre-flight SSRF check (host class only — no URL/secret leak).
            if let Err(blocked) = policy::validate_network_url(url) {
                return Err(blocked.message());
            }
        }

        let (target_id, context_id, page) = self.backend.create_page().await?;
        if let Some(url) = url {
            page.goto(url).await?;
            // Bounded navigation readiness instead of a fixed 2s sleep (see
            // `action_navigate`). Best-effort: a never-`load` page proceeds to
            // URL revalidation; a connection error propagates.
            if let Err(e) = page.wait_for_navigation(self.nav_timeout).await {
                if backend::is_connection_error(&e) {
                    return Err(e);
                }
                warn!(error = %e, "new_tab navigation wait reported a non-connection error");
            }
        }
        let current_url = page.url().await;

        // Revalidate the new tab's FINAL committed URL the same way navigate
        // does, so a redirect to a blocked host can't leave a live tab pointed
        // at an internal address. Close the tab and report the host class only.
        if let Some(ref final_url) = current_url {
            if let Err(blocked) = policy::validate_network_url(final_url) {
                warn!(
                    class = blocked.class.label(),
                    "new tab landed on a blocked host after redirect; closing"
                );
                // Best-effort backend cleanup — the tab was never registered.
                let _ = self.backend.close_target(&target_id).await;
                return Err(format!(
                    "Navigation blocked: redirected to a {}",
                    blocked.class.label()
                ));
            }
        }

        let tab_id = self
            .sessions
            .add_tab(
                session_id,
                &target_id,
                page,
                current_url,
                None,
                context_id,
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

    async fn run_action(&self, arguments: &str) -> anyhow::Result<DispatchResult> {
        let args: Value = serde_json::from_str(arguments)?;

        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: action"))?;

        let session_id = args
            .get("_session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let needs_session = !matches!(action, "close" | "set_mode");
        if needs_session && session_id.is_empty() {
            return Ok(DispatchResult::text_only(
                "Error: browser actions require a session id".to_string(),
            ));
        }

        if action == "execute_js" {
            let script = args.get("script").and_then(|v| v.as_str()).unwrap_or("");
            if let Err(reason) = validate_script_constraints(script) {
                warn!(action, "execute_js script rejected by constraint check");
                return Ok(DispatchResult::text_only(format!("Error: {}", reason)));
            }
        }

        let action_args = ActionArgs {
            action,
            url: args.get("url").and_then(|v| v.as_str()),
            selector: args.get("selector").and_then(|v| v.as_str()),
            script: args.get("script").and_then(|v| v.as_str()),
            tab_id: args.get("tab_id").and_then(|v| v.as_str()),
            session_id,
        };
        if let GateDecision::Deny(reason) = self.approval_gate(&action_args).await {
            warn!(action, "Browser action blocked by approval gate");
            return Ok(DispatchResult::text_only(format!("Error: {}", reason)));
        }

        match self.dispatch_with_recovery(action, &args, session_id).await {
            Ok(result) => Ok(result),
            Err(err_text) => {
                warn!(action, error = %err_text, "Browser action failed");
                Ok(DispatchResult::text_only(format!("Error: {}", err_text)))
            }
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
            "description": "Control a browser for web interactions. Actions: navigate (go to URL), screenshot (capture page as photo), click (click element — reports a new tab id if the click opened one), fill (type into input), get_text (extract text), execute_js (run JavaScript), wait (wait for an element condition: present/visible/enabled/hidden/text_contains), list_tabs (list this session's open tabs with their ids), new_tab (open and switch to a new tab, optionally at a url), switch_tab (make a tab active by its id), close_tab (close a tab by its id), set_mode (switch between 'visible' and 'headless' — use visible for sites that block headless browsers), close (end session). The browser persists across calls for multi-step workflows. Tab ids are opaque tokens returned by list_tabs/new_tab; do not guess them.",
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
                    "full_page": {
                        "type": "boolean",
                        "description": "For 'screenshot' WITHOUT a selector: capture the entire scrollable page instead of just the visible viewport (default false). Full-page captures of long pages may be too large to deliver and will be rejected — prefer the default viewport or a selector."
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
                        "description": "Timeout in seconds for 'wait' action (default from config, clamped 1..=120)"
                    },
                    "condition": {
                        "type": "string",
                        "enum": ["present", "visible", "enabled", "hidden", "text_contains"],
                        "description": "Condition for 'wait' (default: present). present=in DOM; visible=laid out & not hidden; enabled=not disabled; hidden=absent or hidden; text_contains=element text contains 'text'."
                    },
                    "text": {
                        "type": "string",
                        "description": "Needle for the 'wait' action's 'text_contains' condition (the substring to wait for in the element's text)"
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
        Ok(self.run_action(arguments).await?.text)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<tokio::sync::mpsc::Sender<crate::types::StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let _ = status_tx;
        let result = self.run_action(arguments).await?;
        Ok(ToolCallOutcome {
            output: result.text,
            metadata: ToolCallMetadata {
                attachments: result.attachments,
                ..ToolCallMetadata::default()
            },
        })
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

#[cfg(test)]
mod prompt_tests {
    use super::format_browser_approval_prompt;
    use crate::tools::browser::policy::{self, BrowserRiskClass};

    fn sample_risk() -> policy::BrowserActionRisk {
        policy::BrowserActionRisk {
            class: BrowserRiskClass::Navigation,
            sensitive: false,
            consequential: false,
        }
    }

    #[test]
    fn navigate_prompt_is_plain_language() {
        let prompt = format_browser_approval_prompt(
            "navigate",
            "https://newtarget.com",
            None,
            None,
            None,
            &sample_risk(),
        );
        assert_eq!(prompt, "Open website: https://newtarget.com");
        assert!(!prompt.contains("[target:"));
        assert!(!prompt.contains("[risk:"));
    }

    #[test]
    fn execute_js_prompt_hides_script_body() {
        let prompt = format_browser_approval_prompt(
            "execute_js",
            "https://example.com",
            None,
            None,
            Some(512),
            &sample_risk(),
        );
        assert_eq!(prompt, "Run JavaScript on https://example.com (512 bytes)");
    }
}
