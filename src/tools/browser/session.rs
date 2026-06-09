//! Session-scoped browser page state.
//!
//! Before this module, every browser action grabbed the FIRST page in the
//! shared `Browser` (`pages().into_iter().next()`), so all sessions (different
//! users/chats) implicitly shared one tab — session A's navigation moved the
//! page out from under session B's next action.
//!
//! [`BrowserSessionRegistry`] gives each trusted internal `_session_id` its own
//! [`PageHandle`] plus a per-session action lock. The action lock serializes a
//! single session's own actions (so its calls don't race each other) while
//! letting DIFFERENT sessions run concurrently — the registry never holds the
//! global `Mutex<Option<Browser>>` for the duration of an action.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::time::Instant;

use super::backend::{BrowserBackend, PageHandle};

/// Per-session browser state: the page handle that session operates on, plus the
/// lock that serializes that session's own actions.
///
/// `page_id`/`active_target_id` are the chromiumoxide target-id strings, kept
/// for diagnostics and for the later tab-management work. They are read in the
/// tests below and intended for upcoming tasks, so they are allowed to be unused
/// elsewhere for now.
pub struct BrowserSessionState {
    #[allow(dead_code)]
    pub page_id: String,
    #[allow(dead_code)]
    pub active_target_id: String,
    pub last_used_at: Instant,
    pub action_lock: Arc<Mutex<()>>,
    pub page: Arc<dyn PageHandle>,
}

/// A registry of per-session browser pages, keyed by the trusted internal
/// `_session_id`. The map is guarded by a `tokio::sync::Mutex` that is held only
/// briefly to read or insert a session entry — never across the running action.
#[derive(Default)]
pub struct BrowserSessionRegistry {
    sessions: Mutex<HashMap<String, BrowserSessionState>>,
}

impl BrowserSessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve the page + action lock for `session_id`, creating a fresh page on
    /// the backend the first time a session is seen.
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

        // Fast path: existing session — refresh recency and hand back its page.
        {
            let mut sessions = self.sessions.lock().await;
            if let Some(state) = sessions.get_mut(session_id) {
                state.last_used_at = Instant::now();
                return Ok((Arc::clone(&state.page), Arc::clone(&state.action_lock)));
            }
        }

        // Slow path: create a new page on the backend WITHOUT holding the
        // registry lock (page creation may launch/await the browser).
        let (page_id, page) = backend.create_page().await?;

        // Re-lock to insert. A concurrent caller for the same session could have
        // raced us; if so, prefer the already-stored page and drop ours.
        let mut sessions = self.sessions.lock().await;
        if let Some(state) = sessions.get_mut(session_id) {
            state.last_used_at = Instant::now();
            return Ok((Arc::clone(&state.page), Arc::clone(&state.action_lock)));
        }

        let action_lock = Arc::new(Mutex::new(()));
        let state = BrowserSessionState {
            page_id: page_id.clone(),
            active_target_id: page_id,
            last_used_at: Instant::now(),
            action_lock: Arc::clone(&action_lock),
            page: Arc::clone(&page),
        };
        sessions.insert(session_id.to_string(), state);

        Ok((page, action_lock))
    }
}
