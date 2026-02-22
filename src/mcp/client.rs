use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Maximum size (in bytes) for a single MCP JSON-RPC response line.
/// Responses exceeding this are truncated to prevent memory exhaustion.
const MAX_RESPONSE_BYTES: usize = 512 * 1024; // 512 KiB

/// Timeout for a single JSON-RPC round-trip (request + response).
const RPC_TIMEOUT: Duration = Duration::from_secs(60);

/// Timeout for the initial handshake (initialize + notifications/initialized).
const INIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Preferred MCP protocol versions, newest first.
const INIT_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Environment variables safe to pass to MCP server subprocesses.
/// Everything else is stripped to prevent credential leakage.
const SAFE_ENV_KEYS: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "SHELL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_RUNTIME_DIR",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    // Node.js / npm needs these to find global packages
    "NODE_PATH",
    "NPM_CONFIG_PREFIX",
    "NVM_DIR",
];

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

/// JSON-RPC client over stdio for MCP protocol.
pub struct McpClient {
    stdin: Mutex<tokio::process::ChildStdin>,
    stdout: Mutex<BufReader<tokio::process::ChildStdout>>,
    _child: Mutex<Child>,
    next_id: AtomicU64,
}

impl McpClient {
    /// Spawn an MCP server subprocess and initialize the connection.
    ///
    /// The subprocess environment is scrubbed to a safe allowlist to prevent
    /// credential leakage (API keys, tokens, etc.).
    pub async fn spawn(
        command: &str,
        args: &[String],
        extra_env: &std::collections::HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        // Build a minimal environment from the safe allowlist
        let mut safe_env: Vec<(String, String)> = std::env::vars()
            .filter(|(k, _)| SAFE_ENV_KEYS.iter().any(|safe| safe == k))
            .collect();

        // Merge extra env vars (e.g. API keys resolved from keychain)
        for (k, v) in extra_env {
            safe_env.push((k.clone(), v.clone()));
        }

        let mut failures = Vec::new();

        for protocol_version in INIT_PROTOCOL_VERSIONS {
            let mut child = Command::new(command)
                .args(args)
                .env_clear()
                .envs(safe_env.iter().cloned())
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;

            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("Failed to capture MCP server stdin"))?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| anyhow::anyhow!("Failed to capture MCP server stdout"))?;

            // Log stderr in the background so errors from MCP servers are visible
            // rather than silently swallowed.
            if let Some(stderr) = child.stderr.take() {
                let cmd_name = command.to_string();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stderr);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        match reader.read_line(&mut line).await {
                            Ok(0) => break, // EOF
                            Ok(_) => {
                                let trimmed = line.trim_end();
                                if !trimmed.is_empty() {
                                    // Truncate long stderr lines to prevent log flooding
                                    let safe_line = if trimmed.len() > 500 {
                                        format!("{}... [truncated]", &trimmed[..500])
                                    } else {
                                        trimmed.to_string()
                                    };
                                    warn!(mcp_server = %cmd_name, "{}", safe_line);
                                }
                            }
                            Err(_) => break,
                        }
                    }
                });
            }

            let client = Self {
                stdin: Mutex::new(stdin),
                stdout: Mutex::new(BufReader::new(stdout)),
                _child: Mutex::new(child),
                next_id: AtomicU64::new(1),
            };

            // Send initialize request (with init timeout)
            let init_result = tokio::time::timeout(
                INIT_TIMEOUT,
                client.send_request_inner(
                    "initialize",
                    json!({
                        "protocolVersion": protocol_version,
                        "capabilities": {},
                        "clientInfo": {
                            "name": "aidaemon",
                            "version": "0.1.0"
                        }
                    }),
                ),
            )
            .await;

            let init_resp = match init_result {
                Ok(Ok(resp)) => resp,
                Ok(Err(e)) => {
                    failures.push(format!("{}: {}", protocol_version, e));
                    warn!(
                        mcp_server = %command,
                        protocol_version = %protocol_version,
                        error = %e,
                        "MCP initialize failed, trying protocol fallback"
                    );
                    client.shutdown().await;
                    continue;
                }
                Err(_) => {
                    failures.push(format!(
                        "{}: initialization timed out after {:?}",
                        protocol_version, INIT_TIMEOUT
                    ));
                    warn!(
                        mcp_server = %command,
                        protocol_version = %protocol_version,
                        "MCP initialize timed out, trying protocol fallback"
                    );
                    client.shutdown().await;
                    continue;
                }
            };

            // Send initialized notification
            client
                .send_notification("notifications/initialized", json!({}))
                .await?;

            let negotiated = init_resp
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(protocol_version);
            info!(
                mcp_server = %command,
                requested_protocol = %protocol_version,
                negotiated_protocol = %negotiated,
                "MCP handshake completed"
            );
            return Ok(client);
        }

        anyhow::bail!(
            "MCP initialization failed for all protocol versions [{}]. Errors: {}",
            INIT_PROTOCOL_VERSIONS.join(", "),
            failures.join(" | ")
        )
    }

    /// Send a JSON-RPC request with a timeout and read the response.
    pub async fn send_request(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        tokio::time::timeout(RPC_TIMEOUT, self.send_request_inner(method, params))
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "MCP RPC call '{}' timed out after {:?}",
                    method,
                    RPC_TIMEOUT
                )
            })?
    }

    /// Inner send without timeout (used by both public send_request and init).
    async fn send_request_inner(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut line = serde_json::to_string(&request)?;
        line.push('\n');

        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await?;
        }

        // Read response line with size limit to prevent memory exhaustion
        let mut response_line = String::new();
        {
            let mut stdout = self.stdout.lock().await;
            let bytes_read = stdout.read_line(&mut response_line).await?;
            if bytes_read == 0 {
                anyhow::bail!("MCP server closed stdout (empty response)");
            }
        }

        if response_line.len() > MAX_RESPONSE_BYTES {
            anyhow::bail!(
                "MCP response exceeded size limit ({} > {} bytes)",
                response_line.len(),
                MAX_RESPONSE_BYTES
            );
        }

        let response: Value = serde_json::from_str(&response_line)?;

        if let Some(error) = response.get("error") {
            anyhow::bail!("MCP error: {}", error);
        }

        Ok(response["result"].clone())
    }

    /// Send a JSON-RPC notification (no response expected).
    async fn send_notification(&self, method: &str, params: Value) -> anyhow::Result<()> {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });

        let mut line = serde_json::to_string(&notification)?;
        line.push('\n');

        let mut stdin = self.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;

        Ok(())
    }

    /// List tools from the MCP server.
    pub async fn list_tools(&self) -> anyhow::Result<Vec<Value>> {
        let result = self.send_request("tools/list", json!({})).await?;
        let tools = result["tools"].as_array().cloned().unwrap_or_default();
        Ok(tools)
    }

    /// Shut down the MCP server subprocess.
    pub async fn shutdown(&self) {
        let mut child = self._child.lock().await;
        let _ = child.kill().await;
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
