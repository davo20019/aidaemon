//! Which deferred tools each chat session has loaded. A tool loaded once
//! stays loaded for the session's later turns, so the request's tool block
//! (and the provider's prompt cache over it) stays stable. Held in memory:
//! a restart starts every session from the core tools again.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use aidaemon_core::LoadedTools;

/// Sessions remembered at most; past it the map starts over (a session
/// then reloads what it needs, one `load_tools` call).
const MAX_SESSIONS: usize = 512;

static LOADED: LazyLock<Mutex<HashMap<String, LoadedTools>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The loaded set of `session_id`, created empty on first use.
pub fn for_session(session_id: &str) -> LoadedTools {
    let mut map = LOADED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !map.contains_key(session_id) && map.len() >= MAX_SESSIONS {
        map.clear();
    }
    map.entry(session_id.to_string()).or_default().clone()
}

/// Forget what `session_id` loaded (`/clear` starts the session over).
pub fn clear(session_id: &str) {
    LOADED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(session_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_keeps_what_it_loaded() {
        let session = "test:synthetic-loaded-tools";
        clear(session);
        for_session(session).load("browser");
        assert!(for_session(session).contains("browser"));
        assert!(!for_session("test:synthetic-other").contains("browser"));
        clear(session);
        assert!(!for_session(session).contains("browser"));
    }
}
