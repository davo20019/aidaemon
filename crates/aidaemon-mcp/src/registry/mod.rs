//! Runtime-managed MCP servers: the configuration a host keeps for servers
//! added while it runs, and the store trait that persists it. With the
//! `registry` feature, the registry of running servers and their tools as
//! `aidaemon-tools` tools.

#[cfg(feature = "registry")]
pub mod manage;
#[cfg(feature = "registry")]
mod proxied;
#[cfg(feature = "registry")]
mod render;
#[cfg(feature = "registry")]
mod running;

#[cfg(feature = "registry")]
pub use proxied::{RegisteredMcpTool, mcp_tool_capabilities, server_options};
#[cfg(feature = "registry")]
pub use running::{McpRegistry, McpServerConfig, McpServerEntry, ServerInfo};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A dynamically added MCP server configuration (stored in database).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicMcpServer {
    pub id: i64,
    pub name: String,
    pub command: String,
    pub args_json: String,     // JSON array of argument strings
    pub env_keys_json: String, // JSON array of env var key names (values in keychain)
    pub triggers_json: String, // JSON array of trigger keywords
    pub enabled: bool,
    pub created_at: String,
}

/// Runtime-managed MCP servers persistence.
#[async_trait]
pub trait DynamicMcpServerStore: Send + Sync {
    /// Store a dynamically added MCP server.
    async fn save_dynamic_mcp_server(&self, _server: &DynamicMcpServer) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic MCP servers.
    async fn list_dynamic_mcp_servers(&self) -> anyhow::Result<Vec<DynamicMcpServer>> {
        Ok(vec![])
    }

    /// Delete a dynamic MCP server by ID.
    async fn delete_dynamic_mcp_server(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update a dynamic MCP server.
    async fn update_dynamic_mcp_server(&self, _server: &DynamicMcpServer) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Focused persistence contract used by the dynamic MCP registry.
pub trait McpRegistryStore: DynamicMcpServerStore {}

impl<T> McpRegistryStore for T where T: DynamicMcpServerStore {}
