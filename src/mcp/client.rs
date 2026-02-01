use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

/// JSON-RPC client over stdio for MCP protocol.
pub struct McpClient {
    stdin: Mutex<tokio::process::ChildStdin>,
    stdout: Mutex<BufReader<tokio::process::ChildStdout>>,
    _child: Mutex<Child>,
    next_id: AtomicU64,
}

impl McpClient {
    /// Spawn an MCP server subprocess and initialize the connection.
    pub async fn spawn(command: &str, args: &[String]) -> anyhow::Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture MCP server stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture MCP server stdout"))?;

        let client = Self {
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            _child: Mutex::new(child),
            next_id: AtomicU64::new(1),
        };

        // Send initialize request
        let _resp = client
            .send_request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "aidaemon",
                        "version": "0.1.0"
                    }
                }),
            )
            .await?;

        // Send initialized notification
        client.send_notification("notifications/initialized", json!({})).await?;

        Ok(client)
    }

    /// Send a JSON-RPC request and read the response.
    async fn send_request(&self, method: &str, params: Value) -> anyhow::Result<Value> {
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

        // Read response line
        let mut response_line = String::new();
        {
            let mut stdout = self.stdout.lock().await;
            stdout.read_line(&mut response_line).await?;
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
