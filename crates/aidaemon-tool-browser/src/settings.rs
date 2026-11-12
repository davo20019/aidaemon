//! What the browser tool needs to know from its host. Available without the
//! `chromium` feature, so a host can parse and check its configuration
//! without building the browser.

use serde::{Deserialize, Serialize};

/// How browser sessions are isolated from one another with respect to cookies,
/// cache, and other browsing state.
///
/// HONESTY CONTRACT: this enum must never claim isolation that isn't actually
/// delivered. `Page` mode explicitly SHARES cookies across sessions;
/// `BrowserContext` mode creates a real, per-session incognito browser context
/// (no shared cookies/cache). The two modes are mutually exclusive with a
/// shared persistent profile or an attached remote-debugging Chrome — see
/// [`BrowserSettings::resolve_session_isolation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionIsolation {
    /// All sessions share one default browser context: cookies/cache are SHARED.
    /// This is the only honest option when attached to a persistent profile or a
    /// remote-debugging Chrome instance.
    Page,
    /// Each session gets its own incognito browser context: cookies/cache are
    /// ISOLATED per session. Requires dedicated ephemeral browsing (no
    /// `user_data_dir`, no `remote_debugging_port`).
    BrowserContext,
}

/// How the browser tool launches or attaches to Chrome, and its timeouts.
#[derive(Debug, Clone)]
pub struct BrowserSettings {
    pub headless: bool,
    pub screenshot_width: u32,
    pub screenshot_height: u32,
    /// Connect to an existing Chrome instance on this debugging port instead of launching a new one.
    /// Start Chrome with: --remote-debugging-port=9222
    /// This shares your existing login sessions, cookies, and tabs.
    pub remote_debugging_port: Option<u16>,
    /// Path to Chrome user data directory for persistent sessions/cookies.
    /// `None` browses with a fresh temporary profile.
    pub user_data_dir: Option<String>,
    /// Chrome profile directory name within user_data_dir (default: "Default").
    /// Other profiles are typically "Profile 1", "Profile 2", etc.
    pub profile: Option<String>,
    /// How browser sessions are isolated from one another.
    ///
    /// `None` (the default) means "auto": dedicated ephemeral browsing resolves
    /// to `browser_context` (per-session isolated cookies), while a shared
    /// persistent profile (`user_data_dir`) or an attached remote-debugging
    /// Chrome (`remote_debugging_port`) resolves to `page` (shared cookies).
    /// Set explicitly to fail fast on incompatible combinations rather than
    /// silently claiming isolation. See [`Self::resolve_session_isolation`].
    pub session_isolation: Option<SessionIsolation>,
    /// Maximum seconds to wait for a navigation/DOM-ready signal after `goto`
    /// (and for a click-triggered navigation to settle). Clamped to 1..=120 by
    /// [`Self::nav_timeout`]. Default: 30.
    pub nav_timeout_secs: u64,
    /// Default maximum seconds the `wait` action polls for its element
    /// condition. Clamped to 1..=120 by [`Self::element_timeout`]. A per-call
    /// `timeout_secs` argument overrides this (also clamped). Default: 10.
    pub element_timeout_secs: u64,
    /// Overall per-action ceiling for the click nav-race (the short
    /// stable-DOM fallback for a non-navigating click is derived from this but
    /// kept small). Clamped to 1..=300 by [`Self::action_timeout`]. Default: 30.
    pub action_timeout_secs: u64,
}

impl Default for BrowserSettings {
    fn default() -> Self {
        Self {
            headless: true,
            screenshot_width: 1280,
            screenshot_height: 720,
            remote_debugging_port: None,
            user_data_dir: None,
            profile: None,
            session_isolation: None,
            nav_timeout_secs: 30,
            element_timeout_secs: 10,
            action_timeout_secs: 30,
        }
    }
}

impl BrowserSettings {
    /// Resolved navigation/DOM-ready timeout, clamped to a sane bounded range
    /// (1..=120s) so a typo'd or absurd config value can neither hang the tool
    /// (huge value) nor make navigation give up before any page can load
    /// (zero/tiny value).
    pub fn nav_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.nav_timeout_secs.clamp(1, 120))
    }

    /// Resolved default `wait`-action element-poll timeout, clamped to 1..=120s.
    pub fn element_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.element_timeout_secs.clamp(1, 120))
    }

    /// Resolved overall per-action ceiling (used by the click nav-race),
    /// clamped to 1..=300s.
    pub fn action_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.action_timeout_secs.clamp(1, 300))
    }

    /// Resolve the effective session-isolation mode, rejecting configurations
    /// that would falsely claim per-session cookie isolation.
    ///
    /// HONESTY CONTRACT: a shared persistent profile (`user_data_dir`) or an
    /// attached remote-debugging Chrome (`remote_debugging_port`) cannot give
    /// per-session isolated cookies — they always share one cookie jar. So:
    ///
    /// - `Some(BrowserContext)` + shared/attached → **Err** (don't lie).
    /// - `Some(Page)` → always `Ok(Page)` (honest: shared cookies).
    /// - `Some(BrowserContext)` + ephemeral → `Ok(BrowserContext)`.
    /// - `None` (auto): shared/attached → `Ok(Page)`; ephemeral →
    ///   `Ok(BrowserContext)`.
    pub fn resolve_session_isolation(&self) -> Result<SessionIsolation, String> {
        let shared_or_attached =
            self.user_data_dir.is_some() || self.remote_debugging_port.is_some();

        match self.session_isolation {
            Some(SessionIsolation::BrowserContext) if shared_or_attached => Err(
                "browser.session_isolation = \"browser_context\" requires dedicated ephemeral \
                 browsing, but a shared persistent profile (user_data_dir) or an attached \
                 remote-debugging Chrome (remote_debugging_port) is configured. These share a \
                 single cookie jar and CANNOT provide per-session isolated cookies. Either use \
                 session_isolation = \"page\" (sessions share cookies), or remove user_data_dir \
                 and remote_debugging_port to enable real per-session isolation."
                    .to_string(),
            ),
            Some(mode) => Ok(mode),
            None if shared_or_attached => Ok(SessionIsolation::Page),
            None => Ok(SessionIsolation::BrowserContext),
        }
    }
}
