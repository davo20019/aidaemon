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

        // Audit log: truncate args for logging to avoid flooding logs with large payloads
        let args_preview: String = {
            let s = args.to_string();
            if s.len() > 200 { format!("{}...", &s[..200]) } else { s }
        };

        // Threat detection: flag suspicious patterns in tool arguments
        detect_suspicious_args(&self.tool_name, &args_preview);

        info!(mcp_tool = %self.tool_name, args = %args_preview, "MCP tool call started");

        let start = std::time::Instant::now();
        let result = self.client.call_tool(&self.tool_name, args).await;
        let elapsed = start.elapsed();

        match &result {
            Ok(output) => {
                // Threat detection: flag suspicious patterns in tool output
                detect_suspicious_output(&self.tool_name, output);

                info!(
                    mcp_tool = %self.tool_name,
                    duration_ms = elapsed.as_millis() as u64,
                    output_bytes = output.len(),
                    "MCP tool call completed"
                );
            }
            Err(e) => {
                tracing::error!(
                    mcp_tool = %self.tool_name,
                    duration_ms = elapsed.as_millis() as u64,
                    error = %e,
                    "MCP tool call failed"
                );
            }
        }

        result
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

/// Patterns in MCP tool arguments that may indicate an attack or misuse.
/// Each entry is (pattern, description) — matched case-insensitively.
const SUSPICIOUS_ARG_PATTERNS: &[(&str, &str)] = &[
    ("etc/passwd", "system password file access"),
    ("etc/shadow", "shadow password file access"),
    (".ssh/", "SSH key directory access"),
    (".env", "environment file access"),
    ("config.toml", "aidaemon config file access"),
    ("aidaemon.db", "aidaemon database access"),
    ("api_key", "potential API key extraction"),
    ("bot_token", "potential bot token extraction"),
    ("encryption_key", "potential encryption key extraction"),
    ("curl ", "network request via shell"),
    ("wget ", "network request via shell"),
    ("nc ", "netcat usage"),
    ("base64", "data encoding/exfiltration"),
    ("eval(", "code evaluation"),
    ("exec(", "code execution"),
    ("| sh", "shell pipe execution"),
    ("| bash", "shell pipe execution"),
    ("; rm ", "destructive command chaining"),
    ("chmod 777", "overly permissive file permissions"),
];

/// Patterns in MCP tool output that may indicate data exfiltration or compromise.
const SUSPICIOUS_OUTPUT_PATTERNS: &[(&str, &str)] = &[
    ("sk-", "potential OpenAI API key in output"),
    ("ghp_", "potential GitHub token in output"),
    ("-----BEGIN", "potential private key in output"),
    ("PRIVATE KEY", "potential private key in output"),
    ("password", "potential password in output"),
    ("bot_token", "potential bot token reference in output"),
];

/// Check MCP tool arguments for suspicious patterns and log warnings.
fn detect_suspicious_args(tool_name: &str, args: &str) {
    let lower = args.to_lowercase();
    for (pattern, description) in SUSPICIOUS_ARG_PATTERNS {
        if lower.contains(&pattern.to_lowercase()) {
            tracing::warn!(
                mcp_tool = %tool_name,
                threat = %description,
                "SECURITY: suspicious pattern detected in MCP tool arguments"
            );
        }
    }
}

/// Check MCP tool output for suspicious patterns and log warnings.
fn detect_suspicious_output(tool_name: &str, output: &str) {
    // Only check first 4096 bytes to avoid scanning huge outputs
    let check = if output.len() > 4096 { &output[..4096] } else { output };
    let lower = check.to_lowercase();
    for (pattern, description) in SUSPICIOUS_OUTPUT_PATTERNS {
        if lower.contains(&pattern.to_lowercase()) {
            tracing::warn!(
                mcp_tool = %tool_name,
                threat = %description,
                "SECURITY: suspicious pattern detected in MCP tool output"
            );
        }
    }
}
