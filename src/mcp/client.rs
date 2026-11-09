//! A connection to one MCP server, on `aidaemon_mcp::McpServer`.

use std::collections::HashMap;

use aidaemon_mcp::{McpServer, McpServerSpec};
use tracing::info;

/// A connection to one MCP server.
pub struct McpClient {
    server: McpServer,
}

impl McpClient {
    /// Spawn the MCP server `name` and run the handshake.
    ///
    /// The subprocess environment is scrubbed to the SDK's safe allowlist
    /// plus `extra_env` (for example API keys resolved from the keychain),
    /// so no other credential leaks to the server.
    pub async fn spawn(
        name: &str,
        command: &str,
        args: &[String],
        extra_env: &HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        let spec = McpServerSpec::new(name, command)
            .with_args(args)
            .with_env(extra_env)
            .with_options(super::server_options());
        let server = McpServer::connect(spec)
            .await
            .map_err(|error| anyhow::anyhow!("MCP server '{name}' failed to start: {error}"))?;
        info!(
            server = name,
            protocol = %server.info().protocol_version,
            tools = server.info().tools.len(),
            "MCP server connected"
        );
        Ok(Self { server })
    }

    /// The server's tools.
    pub fn tools(&self) -> Vec<aidaemon_mcp::McpTool> {
        self.server.mcp_tools()
    }

    /// Shut down the MCP server subprocess.
    pub async fn shutdown(&self) {
        self.server.shutdown().await;
    }
}
