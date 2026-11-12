//! Narrow store services that tools hold: settings, OAuth connections and
//! runtime-managed CLI agents,
//! plus the plain data those traits carry. The daemon's SQLite store
//! implements them; a tool receives an `Arc<dyn ...>` of the slice it needs
//! through its constructor. Memory stores (facts, episodes, history, people)
//! live in `aidaemon-memory`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[cfg(feature = "oauth")]
pub mod oauth;

/// The conversation's audience, shared across the SDK.
pub use aidaemon_core::ChannelVisibility;

/// An OAuth connection to an external service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConnection {
    pub id: i64,
    pub service: String,
    pub auth_type: String,
    pub username: Option<String>,
    /// Stable remote account identifier proven through an authenticated
    /// identity endpoint. Mandate execution binds to this value rather than a
    /// mutable provider alias or display name.
    pub account_id: Option<String>,
    pub scopes: String,
    pub token_expires_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A pending interactive OAuth flow awaiting browser callback completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOAuthFlow {
    pub state: String,
    pub service: String,
    pub code_verifier: Option<String>,
    pub session_id: String,
    pub created_at: String,
}

/// Generic key/value settings persistence.
#[async_trait]
pub trait SettingsStore: Send + Sync {
    /// Get a setting value by key. Returns None if unset.
    async fn get_setting(&self, _key: &str) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    /// Set a setting value. Creates or updates the key.
    async fn set_setting(&self, _key: &str, _value: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

/// OAuth-connected external services persistence.
#[async_trait]
pub trait OAuthStore: Send + Sync {
    /// Save an OAuth connection. Returns the connection ID.
    async fn save_oauth_connection(&self, _conn: &OAuthConnection) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Save or replace a pending interactive OAuth flow.
    async fn save_pending_oauth_flow(&self, _flow: &PendingOAuthFlow) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get an OAuth connection by service name.
    async fn get_oauth_connection(
        &self,
        _service: &str,
    ) -> anyhow::Result<Option<OAuthConnection>> {
        Ok(None)
    }

    /// List all OAuth connections.
    async fn list_oauth_connections(&self) -> anyhow::Result<Vec<OAuthConnection>> {
        Ok(vec![])
    }

    /// Persist or clear the verified remote account bound to an OAuth service.
    async fn update_oauth_account_id(
        &self,
        _service: &str,
        _account_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Get a pending OAuth flow by state parameter.
    async fn get_pending_oauth_flow(
        &self,
        _state: &str,
    ) -> anyhow::Result<Option<PendingOAuthFlow>> {
        Ok(None)
    }

    /// List all pending OAuth flows.
    async fn list_pending_oauth_flows(&self) -> anyhow::Result<Vec<PendingOAuthFlow>> {
        Ok(vec![])
    }

    /// Delete an OAuth connection by service name.
    async fn delete_oauth_connection(&self, _service: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete a pending OAuth flow by state parameter.
    async fn delete_pending_oauth_flow(&self, _state: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update token expiry for an OAuth connection.
    async fn update_oauth_token_expiry(
        &self,
        _service: &str,
        _expires_at: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A dynamically added CLI agent configuration (stored in database).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicCliAgent {
    pub id: i64,
    pub name: String,
    pub command: String,
    pub args_json: String, // JSON array of argument strings
    pub description: String,
    pub timeout_secs: Option<u64>,
    pub max_output_chars: Option<usize>,
    pub enabled: bool,
    pub created_at: String,
}

/// A record of a CLI agent invocation for logging/auditing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliAgentInvocation {
    pub id: i64,
    pub task_id: Option<String>,
    pub session_id: String,
    pub agent_name: String,
    pub prompt_summary: String,
    pub working_dir: Option<String>,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub exit_code: Option<i32>,
    pub output_summary: Option<String>,
    pub success: Option<bool>,
    pub duration_secs: Option<f64>,
}

/// Runtime-managed CLI agents persistence + invocation logs.
#[async_trait]
pub trait DynamicCliAgentStore: Send + Sync {
    /// Store a dynamically added CLI agent.
    async fn save_dynamic_cli_agent(&self, _agent: &DynamicCliAgent) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic CLI agents.
    async fn list_dynamic_cli_agents(&self) -> anyhow::Result<Vec<DynamicCliAgent>> {
        Ok(vec![])
    }

    /// Delete a dynamic CLI agent by ID.
    async fn delete_dynamic_cli_agent(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update a dynamic CLI agent.
    async fn update_dynamic_cli_agent(&self, _agent: &DynamicCliAgent) -> anyhow::Result<()> {
        Ok(())
    }

    /// Log the start of a CLI agent invocation. Returns the invocation ID.
    async fn log_cli_agent_start(
        &self,
        _task_id: Option<&str>,
        _session_id: &str,
        _agent_name: &str,
        _prompt_summary: &str,
        _working_dir: Option<&str>,
    ) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Log the completion of a CLI agent invocation.
    async fn log_cli_agent_complete(
        &self,
        _id: i64,
        _exit_code: Option<i32>,
        _output_summary: &str,
        _success: bool,
        _duration_secs: f64,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get recent CLI agent invocations (most recent first).
    async fn get_cli_agent_invocations(
        &self,
        _limit: usize,
    ) -> anyhow::Result<Vec<CliAgentInvocation>> {
        Ok(vec![])
    }

    /// Auto-close stale CLI agent invocations that never completed (e.g. crashed worker).
    ///
    /// Implementations should mark rows with `completed_at IS NULL` and older than
    /// `max_age_hours` as completed with `success=false`.
    async fn cleanup_stale_cli_agent_invocations(
        &self,
        _max_age_hours: i64,
    ) -> anyhow::Result<u64> {
        Ok(0)
    }
}

/// OAuth flows only need OAuth connection and pending-flow persistence.
pub trait OAuthGatewayStore: OAuthStore {}

impl<T> OAuthGatewayStore for T where T: OAuthStore {}

/// OAuth connections and pending flows kept in memory, for tests and hosts
/// without a database. Same upsert rules as the daemon's SQLite store.
#[derive(Default)]
pub struct InMemoryOAuthStore {
    inner: std::sync::Mutex<InMemoryOAuth>,
}

#[derive(Default)]
struct InMemoryOAuth {
    next_id: i64,
    connections: std::collections::BTreeMap<String, OAuthConnection>,
    flows: std::collections::HashMap<String, PendingOAuthFlow>,
}

impl InMemoryOAuthStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, InMemoryOAuth> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[async_trait]
impl OAuthStore for InMemoryOAuthStore {
    async fn save_oauth_connection(&self, conn: &OAuthConnection) -> anyhow::Result<i64> {
        let now = chrono::Utc::now().to_rfc3339();
        let mut inner = self.lock();
        let (id, created_at) = match inner.connections.get(&conn.service) {
            Some(existing) => (existing.id, existing.created_at.clone()),
            None => {
                inner.next_id += 1;
                (inner.next_id, now.clone())
            }
        };
        inner.connections.insert(
            conn.service.clone(),
            OAuthConnection {
                id,
                created_at,
                updated_at: now,
                ..conn.clone()
            },
        );
        Ok(id)
    }

    async fn save_pending_oauth_flow(&self, flow: &PendingOAuthFlow) -> anyhow::Result<()> {
        self.lock().flows.insert(flow.state.clone(), flow.clone());
        Ok(())
    }

    async fn get_oauth_connection(&self, service: &str) -> anyhow::Result<Option<OAuthConnection>> {
        Ok(self.lock().connections.get(service).cloned())
    }

    async fn list_oauth_connections(&self) -> anyhow::Result<Vec<OAuthConnection>> {
        Ok(self.lock().connections.values().cloned().collect())
    }

    async fn update_oauth_account_id(
        &self,
        service: &str,
        account_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        let mut inner = self.lock();
        let Some(conn) = inner.connections.get_mut(service) else {
            return Ok(false);
        };
        conn.account_id = account_id.map(str::to_string);
        conn.updated_at = chrono::Utc::now().to_rfc3339();
        Ok(true)
    }

    async fn get_pending_oauth_flow(
        &self,
        state: &str,
    ) -> anyhow::Result<Option<PendingOAuthFlow>> {
        Ok(self.lock().flows.get(state).cloned())
    }

    async fn list_pending_oauth_flows(&self) -> anyhow::Result<Vec<PendingOAuthFlow>> {
        let mut flows: Vec<PendingOAuthFlow> = self.lock().flows.values().cloned().collect();
        flows.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        Ok(flows)
    }

    async fn delete_oauth_connection(&self, service: &str) -> anyhow::Result<()> {
        self.lock().connections.remove(service);
        Ok(())
    }

    async fn delete_pending_oauth_flow(&self, state: &str) -> anyhow::Result<()> {
        self.lock().flows.remove(state);
        Ok(())
    }

    async fn update_oauth_token_expiry(
        &self,
        service: &str,
        expires_at: Option<&str>,
    ) -> anyhow::Result<()> {
        if let Some(conn) = self.lock().connections.get_mut(service) {
            conn.token_expires_at = expires_at.map(str::to_string);
            conn.updated_at = chrono::Utc::now().to_rfc3339();
        }
        Ok(())
    }
}
