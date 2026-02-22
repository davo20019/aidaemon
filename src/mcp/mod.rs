mod client;
pub mod registry;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tracing::{info, warn};

use crate::config::McpServerConfig;
use crate::traits::Tool;

pub use client::McpClient;
pub use registry::McpRegistry;

/// A tool proxied from an MCP server.
pub struct McpTool {
    /// The name advertised to the LLM (may be prefixed for namespacing).
    tool_name: String,
    /// The raw name used when calling the MCP server (always unprefixed).
    server_tool_name: String,
    server_name: Option<String>,
    tool_schema: Value,
    client: Arc<McpClient>,
    registry: Option<McpRegistry>,
}

impl McpTool {
    #[allow(dead_code)] // Used by discover_mcp_tools for static configs
    pub fn new(tool_name: String, tool_schema: Value, client: Arc<McpClient>) -> Self {
        let server_tool_name = tool_name.clone();
        Self {
            tool_name,
            server_tool_name,
            server_name: None,
            tool_schema,
            client,
            registry: None,
        }
    }

    /// Create a tool with a prefixed name for the LLM but a raw name for the server.
    pub fn with_prefix(
        server_tool_name: String,
        prefixed_name: String,
        tool_schema: Value,
        client: Arc<McpClient>,
        server_name: String,
        registry: McpRegistry,
    ) -> Self {
        Self {
            tool_name: prefixed_name,
            server_tool_name,
            server_name: Some(server_name),
            tool_schema,
            client,
            registry: Some(registry),
        }
    }
}

fn is_recoverable_transport_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    if msg.contains("mcp tool reported iserror=true") {
        return false;
    }

    msg.contains("closed stdout")
        || msg.contains("broken pipe")
        || msg.contains("connection reset")
        || msg.contains("connection refused")
        || msg.contains("timed out")
        || msg.contains("io error")
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
        let args: Value =
            serde_json::from_str(arguments).unwrap_or(Value::Object(Default::default()));

        // Audit log: truncate args for logging to avoid flooding logs with large payloads
        let args_preview: String = {
            let s = args.to_string();
            if s.len() > 200 {
                format!("{}...", &s[..200])
            } else {
                s
            }
        };

        // Threat detection: flag suspicious patterns in tool arguments
        detect_suspicious_args(&self.tool_name, &args_preview);

        info!(mcp_tool = %self.tool_name, args = %args_preview, "MCP tool call started");

        let start = std::time::Instant::now();
        let mut result = self
            .client
            .call_tool(&self.server_tool_name, args.clone())
            .await;

        let first_error_message = result.as_ref().err().map(ToString::to_string);
        let should_attempt_recovery = result
            .as_ref()
            .err()
            .is_some_and(is_recoverable_transport_error);

        if should_attempt_recovery {
            if let (Some(registry), Some(server_name)) = (&self.registry, &self.server_name) {
                let first_msg = first_error_message
                    .clone()
                    .unwrap_or_else(|| "unknown MCP transport error".to_string());
                if let Err(ref first_err) = result {
                    warn!(
                        mcp_tool = %self.tool_name,
                        server = %server_name,
                        error = %first_err,
                        "MCP transport failure detected, attempting automatic server restart"
                    );
                }

                match registry.restart_server(server_name).await {
                    Ok(_) => {
                        result = registry
                            .call_server_tool(server_name, &self.server_tool_name, args.clone())
                            .await;
                        match &result {
                            Ok(_) => {
                                warn!(
                                    mcp_tool = %self.tool_name,
                                    server = %server_name,
                                    "MCP server restart succeeded; tool call recovered"
                                );
                            }
                            Err(retry_err) => {
                                result = Err(anyhow::anyhow!(
                                    "MCP transport failed and retry after auto-restart also failed. First error: {}. Retry error: {}",
                                    first_msg,
                                    retry_err
                                ));
                            }
                        }
                    }
                    Err(restart_err) => {
                        result = Err(anyhow::anyhow!(
                            "MCP transport failed and automatic restart failed. First error: {}. Restart error: {}",
                            first_msg,
                            restart_err
                        ));
                    }
                }
            }
        }
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
#[allow(dead_code)] // Kept for backward compatibility; static configs now go through McpRegistry
pub async fn discover_mcp_tools(
    mcp_configs: &std::collections::HashMap<String, McpServerConfig>,
) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

    for (server_name, config) in mcp_configs {
        info!(server_name, command = %config.command, "Connecting to MCP server");

        match McpClient::spawn(&config.command, &config.args, &config.env).await {
            Ok(client) => {
                let client = Arc::new(client);

                match client.list_tools().await {
                    Ok(tool_defs) => {
                        for td in tool_defs {
                            let name = td["name"].as_str().unwrap_or("unknown").to_string();
                            let desc = td["description"].as_str().unwrap_or("").to_string();
                            let mut schema = td["inputSchema"].clone();
                            // Strip $schema — Google Gemini API rejects it
                            if let Some(obj) = schema.as_object_mut() {
                                obj.remove("$schema");
                            }

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
    let check = if output.len() > 4096 {
        &output[..4096]
    } else {
        output
    };
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

#[cfg(test)]
mod tests {
    use super::is_recoverable_transport_error;

    #[test]
    fn transport_error_classifier_ignores_tool_level_is_error() {
        let err = anyhow::anyhow!("MCP tool reported isError=true: Not allowed");
        assert!(!is_recoverable_transport_error(&err));
    }

    #[test]
    fn transport_error_classifier_flags_dead_process_signals() {
        let err = anyhow::anyhow!("MCP server closed stdout (empty response)");
        assert!(is_recoverable_transport_error(&err));
    }
}
