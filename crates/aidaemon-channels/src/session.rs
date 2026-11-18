//! Mapping conversations to the run they continue from.

use std::collections::HashMap;
use std::sync::Mutex;

use aidaemon_core::RunId;
use async_trait::async_trait;

/// Remembers the latest run of each session so a conversation continues
/// from its last saved [`aidaemon_core::RunState`] (loaded from the runner's
/// [`aidaemon_core::RunStore`]). Implement it on your database to survive
/// restarts; [`MemorySessionIndex`] is the in-process default.
#[async_trait]
pub trait SessionIndex: Send + Sync {
    async fn latest_run(&self, session_id: &str) -> Option<RunId>;
    async fn set_latest(&self, session_id: &str, run_id: RunId);
    /// Forget the session (the `/new` command): the next message starts a
    /// fresh run.
    async fn clear(&self, session_id: &str);
}

#[derive(Default)]
pub struct MemorySessionIndex {
    latest: Mutex<HashMap<String, RunId>>,
}

impl MemorySessionIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionIndex for MemorySessionIndex {
    async fn latest_run(&self, session_id: &str) -> Option<RunId> {
        self.latest
            .lock()
            .ok()
            .and_then(|map| map.get(session_id).copied())
    }

    async fn set_latest(&self, session_id: &str, run_id: RunId) {
        if let Ok(mut map) = self.latest.lock() {
            map.insert(session_id.to_string(), run_id);
        }
    }

    async fn clear(&self, session_id: &str) {
        if let Ok(mut map) = self.latest.lock() {
            map.remove(session_id);
        }
    }
}

/// The session id the runner uses: `"{channel_name}:{conversation_id}"`.
pub fn session_id(channel_name: &str, conversation_id: &str) -> String {
    format!("{channel_name}:{conversation_id}")
}
