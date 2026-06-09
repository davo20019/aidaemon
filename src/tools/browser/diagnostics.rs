//! Per-session/tab browser diagnostics: console logs and network load failures.
//!
//! Logs are bounded, secret-redacted, and keyed by `(session_id, tab_id)` so
//! `get_console_logs` / `get_network_errors` can scope results to one tab.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::tools::sanitize::redact_secrets;

use super::PageHandle;

pub const MAX_CONSOLE_LOGS_PER_TAB: usize = 100;
pub const MAX_NETWORK_ERRORS_PER_TAB: usize = 50;
const MAX_LOG_MESSAGE_CHARS: usize = 500;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TabKey {
    session_id: String,
    tab_id: String,
}

impl TabKey {
    fn new(session_id: &str, tab_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            tab_id: tab_id.to_string(),
        }
    }
}

#[derive(Debug, Default)]
struct TabDiagnostics {
    console: Vec<(String, String)>,
    network_errors: Vec<(String, String)>,
}

#[derive(Clone, Default)]
pub struct BrowserDiagnosticsStore {
    inner: Arc<StoreInner>,
}

struct StoreInner {
    tabs: Mutex<HashMap<TabKey, TabDiagnostics>>,
    attached: Mutex<HashSet<TabKey>>,
}

impl Default for StoreInner {
    fn default() -> Self {
        Self {
            tabs: Mutex::new(HashMap::new()),
            attached: Mutex::new(HashSet::new()),
        }
    }
}

impl BrowserDiagnosticsStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Spawn CDP/mock listeners once per tab. Idempotent.
    pub async fn ensure_listeners(
        &self,
        page: &Arc<dyn PageHandle>,
        session_id: &str,
        tab_id: &str,
    ) {
        let key = TabKey::new(session_id, tab_id);
        {
            let mut attached = self.inner.attached.lock().await;
            if attached.contains(&key) {
                return;
            }
            attached.insert(key);
        }
        page.attach_diagnostics(session_id, tab_id, self.clone())
            .await;
    }

    pub async fn record_console(&self, session_id: &str, tab_id: &str, level: &str, message: &str) {
        let message = truncate_chars(&redact_secrets(message), MAX_LOG_MESSAGE_CHARS);
        let level = level.to_ascii_lowercase();
        let key = TabKey::new(session_id, tab_id);
        let mut tabs = self.inner.tabs.lock().await;
        let entry = tabs.entry(key).or_default();
        if entry.console.len() >= MAX_CONSOLE_LOGS_PER_TAB {
            entry.console.remove(0);
        }
        entry.console.push((level, message));
    }

    pub async fn record_network_error(
        &self,
        session_id: &str,
        tab_id: &str,
        url: &str,
        error: &str,
    ) {
        let origin = super::redact_origin(url);
        let error = truncate_chars(&redact_secrets(error), MAX_LOG_MESSAGE_CHARS);
        let key = TabKey::new(session_id, tab_id);
        let mut tabs = self.inner.tabs.lock().await;
        let entry = tabs.entry(key).or_default();
        if entry.network_errors.len() >= MAX_NETWORK_ERRORS_PER_TAB {
            entry.network_errors.remove(0);
        }
        entry.network_errors.push((origin, error));
    }

    pub async fn format_console_logs(&self, session_id: &str, tab_id: &str) -> String {
        let key = TabKey::new(session_id, tab_id);
        let tabs = self.inner.tabs.lock().await;
        let Some(entry) = tabs.get(&key) else {
            return format!("No console logs recorded for tab '{tab_id}'.");
        };
        if entry.console.is_empty() {
            return format!("No console logs recorded for tab '{tab_id}'.");
        }
        let mut out = format!(
            "Console logs for tab '{}' ({} entries):",
            tab_id,
            entry.console.len()
        );
        for (level, message) in &entry.console {
            out.push_str(&format!("\n- [{level}] {message}"));
        }
        out
    }

    pub async fn format_network_errors(&self, session_id: &str, tab_id: &str) -> String {
        let key = TabKey::new(session_id, tab_id);
        let tabs = self.inner.tabs.lock().await;
        let Some(entry) = tabs.get(&key) else {
            return format!("No network errors recorded for tab '{tab_id}'.");
        };
        if entry.network_errors.is_empty() {
            return format!("No network errors recorded for tab '{tab_id}'.");
        }
        let mut out = format!(
            "Network errors for tab '{}' ({} entries):",
            tab_id,
            entry.network_errors.len()
        );
        for (origin, error) in &entry.network_errors {
            let origin = if origin.is_empty() {
                "(unknown origin)".to_string()
            } else {
                origin.clone()
            };
            out.push_str(&format!("\n- {origin}: {error}"));
        }
        out
    }

    pub async fn drop_tab(&self, session_id: &str, tab_id: &str) {
        let key = TabKey::new(session_id, tab_id);
        self.inner.tabs.lock().await.remove(&key);
        self.inner.attached.lock().await.remove(&key);
    }

    /// After a browser reconnect all page handles are stale; listeners must be
    /// re-attached on the next action.
    pub async fn reset_attached(&self) {
        self.inner.attached.lock().await.clear();
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn console_logs_are_bounded_and_redacted() {
        let store = BrowserDiagnosticsStore::new();
        store
            .record_console("s", "t", "error", "missing sk-abcdefghijklmnopqrstuvwxyz")
            .await;
        let out = store.format_console_logs("s", "t").await;
        assert!(out.contains("[error]"));
        assert!(!out.contains("sk-abcdefghijklmnopqrstuvwxyz"));
    }

    #[tokio::test]
    async fn network_errors_redact_url_origin() {
        let store = BrowserDiagnosticsStore::new();
        store
            .record_network_error(
                "s",
                "t",
                "https://api.example.com/path?token=SECRET",
                "net::ERR_FAILED",
            )
            .await;
        let out = store.format_network_errors("s", "t").await;
        assert!(out.contains("https://api.example.com"));
        assert!(!out.contains("SECRET"));
        assert!(!out.contains("/path"));
    }
}
