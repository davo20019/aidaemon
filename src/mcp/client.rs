use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::warn;

/// Maximum size (in bytes) for a single MCP JSON-RPC response line.
/// Responses exceeding this are truncated to prevent memory exhaustion.
const MAX_RESPONSE_BYTES: usize = 512 * 1024; // 512 KiB

/// Timeout for a single JSON-RPC round-trip (request + response).
const RPC_TIMEOUT: Duration = Duration::from_secs(60);

/// Timeout for the initial handshake (initialize + notifications/initialized).
const INIT_TIMEOUT: Duration = Duration::from_secs(30);

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
    pub async fn spawn(command: &str, args: &[String]) -> anyhow::Result<Self> {
        // Build a minimal environment from the safe allowlist
        let safe_env: Vec<(String, String)> = std::env::vars()
            .filter(|(k, _)| SAFE_ENV_KEYS.iter().any(|safe| safe == k))
            .collect();

        let mut child = Command::new(command)
            .args(args)
            .env_clear()
            .envs(safe_env)
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
                                warn!(mcp_server = %cmd_name, "{}", trimmed);
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
        let _resp = tokio::time::timeout(INIT_TIMEOUT, client.send_request_inner(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "aidaemon",
                    "version": "0.1.0"
                }
            }),
        ))
        .await
        .map_err(|_| anyhow::anyhow!("MCP server initialization timed out after {:?}", INIT_TIMEOUT))??;

        // Send initialized notification
        client.send_notification("notifications/initialized", json!({})).await?;

        Ok(client)
    }

    /// Send a JSON-RPC request with a timeout and read the response.
    pub async fn send_request(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        tokio::time::timeout(RPC_TIMEOUT, self.send_request_inner(method, params))
            .await
            .map_err(|_| anyhow::anyhow!("MCP RPC call '{}' timed out after {:?}", method, RPC_TIMEOUT))?
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
        let tools = result["tools"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        Ok(tools)
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

        // MCP returns content as an array of content blocks
        if let Some(content) = result["content"].as_array() {
            let texts: Vec<&str> = content
                .iter()
                .filter_map(|c| c["text"].as_str())
                .collect();
            Ok(texts.join("\n"))
        } else {
            Ok(result.to_string())
        }
    }
}
