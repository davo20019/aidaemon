//! The daemon's MCP client: `aidaemon_mcp::McpServer` for the transport,
//! with the daemon's text rendering of tool results.
//!
//! The transport (stdio JSON-RPC with concurrent requests matched by id,
//! the handshake, paginated tool listing, timeouts, a scrubbed child
//! environment) lives in the SDK crate. The daemon keeps how a result reads
//! to the model, and treats `isError` as a failed call.

use std::collections::HashMap;

use aidaemon_mcp::McpServer;
use serde_json::{json, Value};
use tracing::info;

fn truncate_for_preview(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        input.to_string()
    } else {
        let head: String = input.chars().take(max_chars).collect();
        format!("{}...", head)
    }
}

fn render_mcp_content_block(block: &Value) -> Option<String> {
    let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");

    match block_type {
        "text" => block
            .get("text")
            .and_then(Value::as_str)
            .map(|t| t.to_string()),
        "image" => {
            let mime = block
                .get("mimeType")
                .or_else(|| block.get("mime_type"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");

            if let Some(url) = block.get("url").and_then(Value::as_str) {
                return Some(format!(
                    "[MCP image content: mime={}, url={}]",
                    mime,
                    truncate_for_preview(url, 160)
                ));
            }

            if let Some(data) = block.get("data").and_then(Value::as_str) {
                let approx_bytes = data.len().saturating_mul(3) / 4;
                return Some(format!(
                    "[MCP image content: mime={}, inline_data~{} bytes]",
                    mime, approx_bytes
                ));
            }

            Some(format!("[MCP image content: mime={}]", mime))
        }
        "resource" => {
            let uri = block
                .get("resource")
                .and_then(|r| r.get("uri"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            Some(format!("[MCP resource content: {}]", uri))
        }
        "" => block
            .get("text")
            .and_then(Value::as_str)
            .map(|t| t.to_string()),
        other => Some(format!("[MCP {} content block]", other)),
    }
}

fn format_tool_call_result(result: &Value) -> anyhow::Result<String> {
    let mut rendered_blocks = Vec::new();

    if let Some(content) = result.get("content").and_then(Value::as_array) {
        for block in content {
            if let Some(rendered) = render_mcp_content_block(block) {
                if !rendered.trim().is_empty() {
                    rendered_blocks.push(rendered);
                }
            }
        }
    }

    if rendered_blocks.is_empty() {
        if let Some(structured) = result.get("structuredContent").filter(|v| !v.is_null()) {
            rendered_blocks.push(format!("[MCP structured content]\n{}", structured));
        }
    }

    let rendered = rendered_blocks.join("\n");
    let is_error = result
        .get("isError")
        .or_else(|| result.get("is_error"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if is_error {
        let err_msg = if rendered.trim().is_empty() {
            result.to_string()
        } else {
            rendered
        };
        anyhow::bail!("MCP tool reported isError=true: {}", err_msg);
    }

    if rendered.trim().is_empty() {
        Ok(result.to_string())
    } else {
        Ok(rendered)
    }
}

/// A connection to one MCP server.
pub struct McpClient {
    /// `None` only for test stand-ins that are never called.
    server: Option<McpServer>,
}

impl McpClient {
    /// A client with no server, for tests that need an `McpTool` but never
    /// call it.
    #[cfg(test)]
    pub(crate) fn idle_for_test() -> Self {
        Self { server: None }
    }

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
        let server = McpServer::spawn(name, command, args, extra_env)
            .await
            .map_err(|error| anyhow::anyhow!("MCP server '{name}' failed to start: {error}"))?;
        info!(
            server = name,
            protocol = %server.info().protocol_version,
            tools = server.info().tools.len(),
            "MCP server connected"
        );
        Ok(Self {
            server: Some(server),
        })
    }

    fn server(&self) -> anyhow::Result<&McpServer> {
        self.server
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MCP server is not connected"))
    }

    /// Send a JSON-RPC request and return its result.
    pub async fn send_request(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        self.server()?
            .request(method, params)
            .await
            .map_err(anyhow::Error::from)
    }

    /// The tool definitions the server listed during the handshake.
    pub async fn list_tools(&self) -> anyhow::Result<Vec<Value>> {
        Ok(self.server()?.info().tools.clone())
    }

    /// Shut down the MCP server subprocess.
    pub async fn shutdown(&self) {
        if let Some(server) = &self.server {
            server.shutdown().await;
        }
    }

    /// Call a tool on the MCP server.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> anyhow::Result<String> {
        let result = self
            .send_request(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )
            .await?;

        format_tool_call_result(&result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn formats_text_blocks() {
        let result = json!({
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"}
            ],
            "isError": false
        });

        let out = format_tool_call_result(&result).unwrap();
        assert_eq!(out, "line one\nline two");
    }

    #[test]
    fn preserves_non_text_blocks_as_placeholders() {
        let result = json!({
            "content": [
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": "aGVsbG8="
                }
            ],
            "isError": false
        });

        let out = format_tool_call_result(&result).unwrap();
        assert!(out.contains("MCP image content"));
        assert!(out.contains("image/png"));
    }

    #[test]
    fn respects_is_error_flag() {
        let result = json!({
            "content": [{"type": "text", "text": "Not allowed"}],
            "isError": true
        });

        let err = format_tool_call_result(&result).unwrap_err().to_string();
        assert!(err.contains("isError=true"));
        assert!(err.contains("Not allowed"));
    }
}
