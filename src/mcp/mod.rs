mod client;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tracing::info;

use crate::config::McpServerConfig;
use crate::traits::Tool;

pub use client::McpClient;

/// A tool proxied from an MCP server.
pub struct McpTool {
    tool_name: String,
    tool_schema: Value,
    client: Arc<McpClient>,
}

impl McpTool {
    pub fn new(
        tool_name: String,
        tool_schema: Value,
        client: Arc<McpClient>,
    ) -> Self {
        Self {
            tool_name,
            tool_schema,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        self.tool_schema["description"].as_str().unwrap_or("")
    }

    fn schema(&self) -> Value {
        self.tool_schema.clone()
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or(Value::Object(Default::default()));
        self.client.call_tool(&self.tool_name, args).await
    }
}

/// Discover and register all tools from configured MCP servers.
pub async fn discover_mcp_tools(
    mcp_configs: &std::collections::HashMap<String, McpServerConfig>,
) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

    for (server_name, config) in mcp_configs {
        info!(server_name, command = %config.command, "Connecting to MCP server");

        match McpClient::spawn(&config.command, &config.args).await {
            Ok(client) => {
                let client = Arc::new(client);

                match client.list_tools().await {
                    Ok(tool_defs) => {
                        for td in tool_defs {
                            let name = td["name"].as_str().unwrap_or("unknown").to_string();
                            let desc = td["description"].as_str().unwrap_or("").to_string();
                            let schema = td["inputSchema"].clone();

                            let tool_schema = serde_json::json!({
                                "name": name,
                                "description": desc,
                                "parameters": schema,
                            });

                            info!(server_name, tool_name = %name, "Registered MCP tool");
                            tools.push(Arc::new(McpTool::new(
                                name,
                                tool_schema,
                                Arc::clone(&client),
                            )));
                        }
                    }
                    Err(e) => {
                        tracing::error!(server_name, "Failed to list MCP tools: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::error!(server_name, "Failed to spawn MCP server: {}", e);
            }
        }
    }

    Ok(tools)
}
