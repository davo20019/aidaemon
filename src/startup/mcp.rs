use std::sync::Arc;

use tracing::info;

use crate::config::AppConfig;
use crate::mcp::{self, McpRegistry};
use crate::state::SqliteStateStore;

pub async fn setup_mcp_registry(
    config: &AppConfig,
    state: Arc<SqliteStateStore>,
) -> anyhow::Result<McpRegistry> {
    let mcp_registry = mcp::McpRegistry::new(state);

    for (name, mcp_config) in &config.mcp {
        match mcp_registry
            .add_server(name, mcp_config.clone(), false)
            .await
        {
            Ok(tool_names) => {
                info!(server = %name, tools = ?tool_names, "Static MCP server registered");
            }
            Err(e) => {
                tracing::error!(server = %name, error = %e, "Failed to spawn static MCP server");
            }
        }
    }

    if let Err(e) = mcp_registry.load_from_db().await {
        tracing::error!(error = %e, "Failed to load dynamic MCP servers from database");
    }

    Ok(mcp_registry)
}
