//! Narrow store services that tools hold: settings and OAuth connections,
//! plus the plain data those traits carry. The daemon's SQLite store
//! implements them; a tool receives an `Arc<dyn ...>` of the slice it needs
//! through its constructor. Memory stores (facts, episodes, history, people)
//! live in `aidaemon-memory`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

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

/// OAuth flows only need OAuth connection and pending-flow persistence.
pub trait OAuthGatewayStore: OAuthStore {}

impl<T> OAuthGatewayStore for T where T: OAuthStore {}
