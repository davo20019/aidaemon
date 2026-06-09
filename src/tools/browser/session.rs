//! Session-scoped browser tab state.
//!
//! Before this module, every browser action grabbed the FIRST page in the
//! shared `Browser` (`pages().into_iter().next()`), so all sessions (different
//! users/chats) implicitly shared one tab — session A's navigation moved the
//! page out from under session B's next action.
//!
//! [`BrowserSessionRegistry`] gives each trusted internal `_session_id` its own
//! set of [`PageHandle`]s (tabs) plus a per-session action lock. The action lock
//! serializes a single session's own actions (so its calls don't race each
//! other) while letting DIFFERENT sessions run concurrently — the registry never
//! holds the global `Mutex<Option<Browser>>` for the duration of an action.
//!
//! ## Tab model
//!
//! A session owns one or more [`TabEntry`] tabs and tracks one ACTIVE tab. The
//! session's "current page" — the page existing single-tab actions operate on —
//! is always the active tab's page. `tab_id` is the chromiumoxide target-id
//! string, used as an OPAQUE, stable identifier: callers never see or rely on
//! positional ordering, only on this id. (Choosing the target id as the public
//! id keeps a single source of truth — it is already unique and stable for the
//! tab's lifetime — and lets the backend resolve it directly without a
//! translation table.)

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::time::Instant;

use super::backend::{BrowserBackend, PageHandle};

/// One tab owned by a session.
///
/// `tab_id` and `target_id` are the same chromiumoxide target-id string today
/// (the opaque public id IS the target id), kept as separate fields so the
/// public/opaque identity and the backend-facing identity can diverge later
/// without touching call sites. `last_url` caches the URL last observed for the
/// tab (used to render a redacted origin in `list_tabs` without a round-trip).
#[derive(Clone)]
pub struct TabEntry {
    pub tab_id: String,
    pub target_id: String,
    pub page: Arc<dyn PageHandle>,
    pub last_url: Option<String>,
    pub title: Option<String>,
}

/// Per-session browser state: the tabs the session owns, which one is active,
/// plus the lock that serializes that session's own actions.
pub struct BrowserSessionState {
    pub tabs: Vec<TabEntry>,
    /// The `target_id` of the active tab. Always references one of `tabs` while
    /// the session has at least one tab.
    pub active_target_id: String,
    pub last_used_at: Instant,
    pub action_lock: Arc<Mutex<()>>,
}

impl BrowserSessionState {
    /// The active tab, if any. A session always has at least one tab while it
    /// exists, but `close_tab` on the last tab can leave it empty.
    fn active_tab(&self) -> Option<&TabEntry> {
        self.tabs
            .iter()
            .find(|t| t.target_id == self.active_target_id)
    }
}

/// A registry of per-session browser tabs, keyed by the trusted internal
/// `_session_id`. The map is guarded by a `tokio::sync::Mutex` that is held only
/// briefly to read or insert a session entry — never across the running action.
#[derive(Default)]
pub struct BrowserSessionRegistry {
    sessions: Mutex<HashMap<String, BrowserSessionState>>,
}

/// A tab as surfaced to the tool/LLM: opaque id, title, raw URL (the tool
/// redacts the URL to an origin before display), and whether it is active.
#[derive(Debug, Clone)]
pub struct TabView {
    pub tab_id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub active: bool,
}

impl BrowserSessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve the ACTIVE tab's page + the session action lock, creating a fresh
    /// first tab on the backend the first time a session is seen.
    ///
    /// Rejects an empty `session_id` BEFORE touching the backend, so a missing
    /// session id never launches a browser. Returns owned `Arc`s so the caller
    /// runs the action without holding the registry lock (or any global browser
    /// lock) for the action's duration.
    pub async fn get_or_create_page(
        &self,
        session_id: &str,
        backend: &dyn BrowserBackend,
    ) -> Result<(Arc<dyn PageHandle>, Arc<Mutex<()>>), String> {
        if session_id.is_empty() {
            return Err("browser actions require a session id".to_string());
        }

        // Fast path: existing session — refresh recency and hand back its active
        // tab's page.
        {
            let mut sessions = self.sessions.lock().await;
            if let Some(state) = sessions.get_mut(session_id) {
                state.last_used_at = Instant::now();
                if let Some(tab) = state.active_tab() {
                    return Ok((Arc::clone(&tab.page), Arc::clone(&state.action_lock)));
                }
                // Session exists but has no tabs (last tab was closed). Fall
                // through to create a fresh one below.
            }
        }

        // Slow path: create a new page on the backend WITHOUT holding the
        // registry lock (page creation may launch/await the browser).
        let (page_id, page) = backend.create_page().await?;

        // Re-lock to insert. A concurrent caller for the same session could have
        // raced us; if so, prefer the already-stored active tab and drop ours.
        let mut sessions = self.sessions.lock().await;
        if let Some(state) = sessions.get_mut(session_id) {
            state.last_used_at = Instant::now();
            if let Some(tab) = state.active_tab() {
                return Ok((Arc::clone(&tab.page), Arc::clone(&state.action_lock)));
            }
            // Session existed but had no active tab: adopt the one we just made.
            let tab = TabEntry {
                tab_id: page_id.clone(),
                target_id: page_id.clone(),
                page: Arc::clone(&page),
                last_url: None,
                title: None,
            };
            state.active_target_id = page_id;
            state.tabs.push(tab);
            return Ok((page, Arc::clone(&state.action_lock)));
        }

        let action_lock = Arc::new(Mutex::new(()));
        let tab = TabEntry {
            tab_id: page_id.clone(),
            target_id: page_id.clone(),
            page: Arc::clone(&page),
            last_url: None,
            title: None,
        };
        let state = BrowserSessionState {
            tabs: vec![tab],
            active_target_id: page_id,
            last_used_at: Instant::now(),
            action_lock: Arc::clone(&action_lock),
        };
        sessions.insert(session_id.to_string(), state);

        Ok((page, action_lock))
    }

    /// Add a tab to a session (e.g. a popup discovered after a click, or an
    /// explicit `new_tab`). When `make_active` is true the new tab becomes the
    /// session's active tab. Returns the opaque `tab_id`.
    ///
    /// If the session is unknown, this is a no-op returning `None` — callers
    /// only register tabs for sessions that already exist (a page must have been
    /// created for the click/new_tab to run).
    pub async fn add_tab(
        &self,
        session_id: &str,
        target_id: &str,
        page: Arc<dyn PageHandle>,
        url: Option<String>,
        title: Option<String>,
        make_active: bool,
    ) -> Option<String> {
        let mut sessions = self.sessions.lock().await;
        let state = sessions.get_mut(session_id)?;
        // Idempotent: don't double-register a target.
        if let Some(existing) = state.tabs.iter().find(|t| t.target_id == target_id) {
            let id = existing.tab_id.clone();
            if make_active {
                state.active_target_id = target_id.to_string();
            }
            return Some(id);
        }
        let tab = TabEntry {
            tab_id: target_id.to_string(),
            target_id: target_id.to_string(),
            page,
            last_url: url,
            title,
        };
        let id = tab.tab_id.clone();
        state.tabs.push(tab);
        if make_active {
            state.active_target_id = target_id.to_string();
        }
        Some(id)
    }

    /// Switch the session's active tab to `tab_id`. Validates that the tab
    /// belongs to THIS session (cross-session safety): a `tab_id` minted for a
    /// different session is rejected, never silently honored. Returns the new
    /// active tab as a [`TabView`].
    pub async fn switch_tab(&self, session_id: &str, tab_id: &str) -> Result<TabView, String> {
        let mut sessions = self.sessions.lock().await;
        let state = sessions
            .get_mut(session_id)
            .ok_or_else(|| "no browser tabs for this session".to_string())?;
        let owned = state.tabs.iter().any(|t| t.tab_id == tab_id);
        if !owned {
            return Err(format!(
                "Unknown tab '{}'. It does not belong to this session. Use list_tabs to see open tabs.",
                tab_id
            ));
        }
        state.active_target_id = tab_id.to_string();
        let tab = state
            .tabs
            .iter()
            .find(|t| t.tab_id == tab_id)
            .expect("ownership just validated");
        Ok(TabView {
            tab_id: tab.tab_id.clone(),
            title: tab.title.clone(),
            url: tab.last_url.clone(),
            active: true,
        })
    }

    /// Close `tab_id` for `session_id`. Validates session ownership first
    /// (cross-session safety). Returns the `target_id` to close at the backend
    /// plus the new active `tab_id` (if any remains). If the closed tab was
    /// active, the most-recently-added remaining tab becomes active.
    pub async fn close_tab(
        &self,
        session_id: &str,
        tab_id: &str,
    ) -> Result<(String, Option<String>), String> {
        let mut sessions = self.sessions.lock().await;
        let state = sessions
            .get_mut(session_id)
            .ok_or_else(|| "no browser tabs for this session".to_string())?;
        let pos = state.tabs.iter().position(|t| t.tab_id == tab_id);
        let Some(pos) = pos else {
            return Err(format!(
                "Unknown tab '{}'. It does not belong to this session. Use list_tabs to see open tabs.",
                tab_id
            ));
        };
        let removed = state.tabs.remove(pos);
        let was_active = state.active_target_id == removed.target_id;
        let new_active = if was_active {
            // Pick the last remaining tab as the new active one.
            if let Some(last) = state.tabs.last() {
                state.active_target_id = last.target_id.clone();
                Some(last.tab_id.clone())
            } else {
                state.active_target_id.clear();
                None
            }
        } else {
            state
                .tabs
                .iter()
                .find(|t| t.target_id == state.active_target_id)
                .map(|t| t.tab_id.clone())
        };
        Ok((removed.target_id, new_active))
    }

    /// The `target_id` of this session's ACTIVE tab, if the session exists and
    /// currently has an active tab. Used by popup detection to attribute a
    /// net-new target only when its CDP `openerId` matches the clicking
    /// session's active page (cross-session safety).
    pub async fn active_target_id(&self, session_id: &str) -> Option<String> {
        let sessions = self.sessions.lock().await;
        let state = sessions.get(session_id)?;
        let active = &state.active_target_id;
        if active.is_empty() {
            return None;
        }
        // Only report it when it actually references one of the session's tabs.
        state
            .tabs
            .iter()
            .find(|t| &t.target_id == active)
            .map(|t| t.target_id.clone())
    }

    /// Snapshot of this session's tabs for `list_tabs`. Empty if the session is
    /// unknown or has no tabs.
    pub async fn list_tabs(&self, session_id: &str) -> Vec<TabView> {
        let sessions = self.sessions.lock().await;
        let Some(state) = sessions.get(session_id) else {
            return Vec::new();
        };
        state
            .tabs
            .iter()
            .map(|t| TabView {
                tab_id: t.tab_id.clone(),
                title: t.title.clone(),
                url: t.last_url.clone(),
                active: t.target_id == state.active_target_id,
            })
            .collect()
    }
}
