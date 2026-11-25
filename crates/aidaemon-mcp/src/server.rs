//! Reusable Model Context Protocol (MCP) server and stdio bridge.
//!
//! Any agentic framework embedding `aidaemon-mcp` (or `aidaemon-sdk`) can:
//!
//! - Serve tools directly over stdio ([`serve_stdio`], [`serve_io`]).
//! - Expose per-caller scoped MCP endpoints on loopback via [`LocalMcpServer`],
//!   each authenticated by an ephemeral 256-bit bearer token bound to a single
//!   [`McpToolHandler`].
//! - Bridge a CLI client's stdio MCP transport to a [`LocalMcpServer`] using
//!   [`run_stdio_bridge`] or [`run_stdio_bridge_from_args`].
//! - Adapt any slice of [`aidaemon_core::Tool`]s directly via
//!   [`CoreToolsHandler`].

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use aidaemon_core::{Tool, ToolContext};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::{PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS};

/// Environment variable carrying the per-session bearer token for
/// [`run_stdio_bridge_from_args`]. Keeping the token out of process arguments
/// prevents exposure via process listings.
pub const BRIDGE_TOKEN_ENV: &str = "AIDAEMON_MCP_TOKEN";

/// Maximum JSON-RPC line accepted from an MCP client (4 MiB).
const MAX_REQUEST_LINE_BYTES: usize = 4 * 1024 * 1024;
/// Maximum handshake line accepted by [`LocalMcpServer`].
const MAX_AUTH_LINE_BYTES: usize = 1024;
/// Maximum concurrent in-flight `tools/call` dispatches per connection.
const MAX_CONCURRENT_CALLS_PER_CONN: usize = 16;
/// Deadline for the initial auth handshake line on a local TCP connection.
const AUTH_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Metadata advertised by an MCP server in its `initialize` reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerMeta {
    pub name: String,
    pub version: String,
    /// Optional server-level instructions returned during `initialize`.
    pub instructions: Option<String>,
}

impl ServerMeta {
    /// Construct server metadata with no extra instructions.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            instructions: None,
        }
    }

    /// Attach optional instructions for MCP clients.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// One tool definition returned by `tools/list`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolDef {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// The result of one `tools/call` invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolResult {
    pub content: Vec<Value>,
    #[serde(
        rename = "isError",
        default,
        skip_serializing_if = "std::ops::Not::not"
    )]
    pub is_error: bool,
}

impl ServerToolResult {
    /// Plain text tool output.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![json!({ "type": "text", "text": text.into() })],
            is_error: false,
        }
    }

    /// Structured JSON serialized as a pretty text content block.
    pub fn json(value: &Value) -> Self {
        let rendered = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
        Self::text(rendered)
    }

    /// Tool error response (`isError: true`).
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![json!({ "type": "text", "text": message.into() })],
            is_error: true,
        }
    }
}

/// Host-side handler backing an MCP server.
#[async_trait]
pub trait McpToolHandler: Send + Sync {
    /// Tools visible to this caller.
    async fn list_tools(&self) -> Vec<ServerToolDef>;

    /// Invoke `name` with `arguments` (an object, or empty object if omitted).
    async fn call_tool(&self, name: &str, arguments: Value) -> ServerToolResult;
}

/// Adapts a set of [`aidaemon_core::Tool`]s directly into an [`McpToolHandler`].
pub struct CoreToolsHandler {
    tools: HashMap<String, Arc<dyn Tool>>,
    order: Vec<String>,
    session_id: String,
}

impl CoreToolsHandler {
    /// Build a handler serving `tools` under `session_id`.
    pub fn new(
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
        session_id: impl Into<String>,
    ) -> Self {
        let mut map = HashMap::new();
        let mut order = Vec::new();
        for tool in tools {
            let name = tool.spec().name;
            if !map.contains_key(&name) {
                order.push(name.clone());
            }
            map.insert(name, tool);
        }
        Self {
            tools: map,
            order,
            session_id: session_id.into(),
        }
    }
}

#[async_trait]
impl McpToolHandler for CoreToolsHandler {
    async fn list_tools(&self) -> Vec<ServerToolDef> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| {
                let spec = tool.spec();
                ServerToolDef {
                    name: spec.name,
                    description: spec.description,
                    input_schema: spec.parameters,
                }
            })
            .collect()
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> ServerToolResult {
        let Some(tool) = self.tools.get(name) else {
            return ServerToolResult::error(format!("unknown tool `{name}`"));
        };
        let ctx = ToolContext::new(self.session_id.clone(), Uuid::new_v4().to_string());
        match tool.call(arguments, &ctx).await {
            Ok(output) => ServerToolResult::text(output.content),
            Err(error) => ServerToolResult::error(error.to_string()),
        }
    }
}

/// Serve MCP JSON-RPC 2.0 on `std::io::stdin` / `std::io::stdout`.
pub async fn serve_stdio(
    handler: Arc<dyn McpToolHandler>,
    meta: ServerMeta,
) -> std::io::Result<()> {
    serve_io(tokio::io::stdin(), tokio::io::stdout(), handler, meta).await
}

/// Serve newline-delimited MCP JSON-RPC 2.0 over any async byte stream pair.
///
/// Requests with an `id` are answered with matching JSON-RPC responses;
/// notifications (no `id`) produce no output. Tool calls execute concurrently
/// (bounded per connection) so blocking coordination calls never stall `ping`
/// or parallel tool invocations on the same connection.
pub async fn serve_io<R, W>(
    reader: R,
    writer: W,
    handler: Arc<dyn McpToolHandler>,
    meta: ServerMeta,
) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let mut reader = BufReader::new(reader);
    let writer = Arc::new(Mutex::new(writer));
    let meta = Arc::new(meta);
    let in_flight = Arc::new(Semaphore::new(MAX_CONCURRENT_CALLS_PER_CONN));
    let mut tasks = JoinSet::new();

    loop {
        let mut buf = Vec::new();
        let read = read_bounded_line(&mut reader, &mut buf, MAX_REQUEST_LINE_BYTES).await?;
        if read == 0 {
            break;
        }
        let line = String::from_utf8_lossy(&buf);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(trimmed) {
            Ok(val) => val,
            Err(error) => {
                write_json_line(
                    &writer,
                    &jsonrpc_error(Value::Null, -32700, format!("parse error: {error}")),
                )
                .await?;
                continue;
            }
        };

        let id = request.get("id").cloned();
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let params = request.get("params").cloned().unwrap_or(Value::Null);

        // JSON-RPC 2.0 notifications omit `id` and expect no reply.
        let Some(id) = id else {
            continue;
        };

        if method == "tools/call" {
            let permit = match in_flight.clone().acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => break,
            };
            let handler = handler.clone();
            let writer = writer.clone();
            tasks.spawn(async move {
                let _permit = permit;
                let response = handle_tool_call(id, params, &*handler).await;
                let _ = write_json_line(&writer, &response).await;
            });
            // Reap already-finished tasks so the JoinSet stays bounded.
            while tasks.try_join_next().is_some() {}
            continue;
        }

        let response = match method.as_str() {
            "initialize" => handle_initialize(id, &params, &meta),
            "ping" => jsonrpc_result(id, json!({})),
            "tools/list" => {
                let tools = handler.list_tools().await;
                jsonrpc_result(id, json!({ "tools": tools }))
            }
            "resources/list" => jsonrpc_result(id, json!({ "resources": [] })),
            "prompts/list" => jsonrpc_result(id, json!({ "prompts": [] })),
            other => jsonrpc_error(id, -32601, format!("method not found: {other}")),
        };
        write_json_line(&writer, &response).await?;
    }

    while tasks.join_next().await.is_some() {}
    Ok(())
}

fn handle_initialize(id: Value, params: &Value, meta: &ServerMeta) -> Value {
    let requested = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or(PROTOCOL_VERSION);
    let negotiated = if SUPPORTED_PROTOCOL_VERSIONS.contains(&requested) {
        requested
    } else {
        PROTOCOL_VERSION
    };
    let mut result = json!({
        "protocolVersion": negotiated,
        "capabilities": {
            "tools": { "listChanged": false }
        },
        "serverInfo": {
            "name": meta.name,
            "version": meta.version,
        }
    });
    if let Some(instructions) = meta.instructions.as_deref()
        && !instructions.trim().is_empty()
    {
        result["instructions"] = Value::String(instructions.to_string());
    }
    jsonrpc_result(id, result)
}

async fn handle_tool_call(id: Value, params: Value, handler: &dyn McpToolHandler) -> Value {
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return jsonrpc_error(id, -32602, "missing `name` in tools/call params");
    };
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let result = handler.call_tool(name, arguments).await;
    match serde_json::to_value(result) {
        Ok(value) => jsonrpc_result(id, value),
        Err(error) => jsonrpc_error(
            id,
            -32603,
            format!("failed to serialize tool result: {error}"),
        ),
    }
}

fn jsonrpc_result(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn jsonrpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
        }
    })
}

async fn write_json_line<W: AsyncWrite + Unpin>(
    writer: &Arc<Mutex<W>>,
    value: &Value,
) -> std::io::Result<()> {
    let mut line = serde_json::to_string(value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');
    let mut guard = writer.lock().await;
    guard.write_all(line.as_bytes()).await?;
    guard.flush().await
}

/// Read one `\n`-terminated line up to `max_bytes`. Returns 0 on EOF.
/// Lines exceeding `max_bytes` are drained and rejected with `InvalidData`.
async fn read_bounded_line<R: AsyncBufReadExt + Unpin>(
    reader: &mut R,
    out: &mut Vec<u8>,
    max_bytes: usize,
) -> std::io::Result<usize> {
    out.clear();
    let mut limited = reader.take((max_bytes + 1) as u64);
    let n = limited.read_until(b'\n', out).await?;
    if n > max_bytes {
        // Drain remainder of oversized line so the stream stays synchronized.
        let inner = limited.into_inner();
        let mut sink = Vec::new();
        while !out.ends_with(b"\n") {
            sink.clear();
            let chunk = inner.take(4096).read_until(b'\n', &mut sink).await?;
            if chunk == 0 || sink.ends_with(b"\n") {
                break;
            }
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "JSON-RPC line exceeded maximum allowed size",
        ));
    }
    Ok(n)
}

/// How an MCP client spawns the local stdio bridge to reach a [`ScopedEndpoint`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpBridgeLaunch {
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

/// One registered caller: its handler, and the switch that ends its live
/// connections when its token is revoked.
struct Session {
    handler: Arc<dyn McpToolHandler>,
    revoked: Arc<tokio::sync::watch::Sender<bool>>,
}

type Sessions = Arc<RwLock<HashMap<String, Session>>>;

/// An active per-caller registration on [`LocalMcpServer`].
///
/// Dropping this handle revokes its bearer token immediately so no further
/// connections can authenticate as this caller.
pub struct ScopedEndpoint {
    addr: SocketAddr,
    token: String,
    sessions: Sessions,
}

impl ScopedEndpoint {
    /// Loopback socket address of the [`LocalMcpServer`].
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Ephemeral 256-bit bearer token bound to this caller's handler.
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Build a stdio bridge launch command for an executable that routes
    /// `subcommand_args` to [`run_stdio_bridge_from_args`].
    pub fn bridge_launch(
        &self,
        bridge_exe: impl Into<String>,
        subcommand_args: impl IntoIterator<Item = String>,
    ) -> McpBridgeLaunch {
        let mut args: Vec<String> = subcommand_args.into_iter().collect();
        args.push("--addr".to_string());
        args.push(self.addr.to_string());
        McpBridgeLaunch {
            command: bridge_exe.into(),
            args,
            env: vec![(BRIDGE_TOKEN_ENV.to_string(), self.token.clone())],
        }
    }

    /// Explicitly revoke this endpoint's token before drop. Connections
    /// already open with it end too: revoking takes the caller's authority
    /// away, not only its next sign-in.
    pub fn revoke(&self) {
        if let Ok(mut map) = self.sessions.write()
            && let Some(session) = map.remove(&self.token)
        {
            let _ = session.revoked.send(true);
        }
    }
}

impl Drop for ScopedEndpoint {
    fn drop(&mut self) {
        self.revoke();
    }
}

/// Loopback TCP MCP server multiplexing many per-caller scoped handlers behind
/// ephemeral 256-bit bearer tokens.
///
/// Designed for host processes (such as `aidaemon` or any SDK-based agent
/// framework) that spawn external CLI agents and inject a single scoped MCP
/// server into each CLI without exposing host credentials or other sessions.
pub struct LocalMcpServer {
    addr: SocketAddr,
    sessions: Sessions,
    accept_task: tokio::task::JoinHandle<()>,
}

impl LocalMcpServer {
    /// Bind an ephemeral `127.0.0.1:0` port and start accepting bridge connections.
    pub async fn bind(meta: ServerMeta) -> std::io::Result<Arc<Self>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let sessions: Sessions = Arc::new(RwLock::new(HashMap::new()));
        let accept_sessions = sessions.clone();
        let accept_task = tokio::spawn(async move {
            loop {
                let Ok((stream, _peer)) = listener.accept().await else {
                    break;
                };
                let sessions = accept_sessions.clone();
                let meta = meta.clone();
                tokio::spawn(async move {
                    let _ = handle_tcp_connection(stream, sessions, meta).await;
                });
            }
        });
        Ok(Arc::new(Self {
            addr,
            sessions,
            accept_task,
        }))
    }

    /// The bound loopback address (`127.0.0.1:<port>`).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Register a caller-scoped [`McpToolHandler`] and return its ephemeral
    /// authenticated endpoint. Dropping the returned [`ScopedEndpoint`] revokes
    /// its token.
    pub fn register_scoped(&self, handler: Arc<dyn McpToolHandler>) -> ScopedEndpoint {
        let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
        if let Ok(mut map) = self.sessions.write() {
            let (revoked, _) = tokio::sync::watch::channel(false);
            map.insert(
                token.clone(),
                Session {
                    handler,
                    revoked: Arc::new(revoked),
                },
            );
        }
        ScopedEndpoint {
            addr: self.addr,
            token,
            sessions: self.sessions.clone(),
        }
    }
}

impl Drop for LocalMcpServer {
    fn drop(&mut self) {
        self.accept_task.abort();
    }
}

async fn handle_tcp_connection(
    stream: TcpStream,
    sessions: Sessions,
    meta: ServerMeta,
) -> std::io::Result<()> {
    let _ = stream.set_nodelay(true);
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut auth_buf = Vec::new();
    let read = tokio::time::timeout(
        AUTH_HANDSHAKE_TIMEOUT,
        read_bounded_line(&mut reader, &mut auth_buf, MAX_AUTH_LINE_BYTES),
    )
    .await
    .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "auth handshake timed out"))??;
    if read == 0 {
        return Ok(());
    }
    let auth_msg: Value = serde_json::from_slice(&auth_buf)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid auth frame"))?;
    let Some(token) = auth_msg.get("auth").and_then(Value::as_str) else {
        let _ = write_half
            .write_all(b"{\"ok\":false,\"error\":\"missing auth token\"}\n")
            .await;
        return Ok(());
    };
    let handler = lookup_handler_constant_time(&sessions, token);
    let Some((handler, mut revoked)) = handler else {
        let _ = write_half
            .write_all(b"{\"ok\":false,\"error\":\"unauthorized\"}\n")
            .await;
        return Ok(());
    };
    write_half.write_all(b"{\"ok\":true}\n").await?;
    write_half.flush().await?;
    tokio::select! {
        served = serve_io(reader, write_half, handler, meta) => served,
        _ = revoked.wait_for(|revoked| *revoked) => Ok(()),
    }
}

fn lookup_handler_constant_time(
    sessions: &Sessions,
    candidate: &str,
) -> Option<(Arc<dyn McpToolHandler>, tokio::sync::watch::Receiver<bool>)> {
    let map = sessions.read().ok()?;
    let mut found = None;
    for (stored, session) in map.iter() {
        if constant_time_eq(stored.as_bytes(), candidate.as_bytes()) {
            found = Some((session.handler.clone(), session.revoked.subscribe()));
        }
    }
    found
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Bridge `stdin`/`stdout` (newline-delimited MCP JSON-RPC) to a running
/// [`LocalMcpServer`] at `addr`, authenticating with `token`.
pub async fn run_stdio_bridge(addr: &str, token: &str) -> anyhow::Result<()> {
    run_io_bridge(addr, token, tokio::io::stdin(), tokio::io::stdout()).await
}

/// Parse `--addr <host:port>` from `args`, read [`BRIDGE_TOKEN_ENV`] from the
/// environment, and run [`run_stdio_bridge`].
///
/// Any host binary can invoke this directly from a subcommand (such as
/// `aidaemon mcp-bridge --addr ...`) before initializing logging so stdout
/// carries only JSON-RPC frames.
pub async fn run_stdio_bridge_from_args(args: &[String]) -> anyhow::Result<()> {
    let mut addr: Option<String> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--addr" {
            addr = iter.next().cloned();
        } else if let Some(val) = arg.strip_prefix("--addr=") {
            addr = Some(val.to_string());
        }
    }
    let addr = addr
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("mcp-bridge requires `--addr <host:port>`"))?;
    let token = std::env::var(BRIDGE_TOKEN_ENV)
        .map_err(|_| anyhow::anyhow!("mcp-bridge requires {BRIDGE_TOKEN_ENV} in environment"))?;
    if token.trim().is_empty() {
        anyhow::bail!("{BRIDGE_TOKEN_ENV} must not be empty");
    }
    run_stdio_bridge(&addr, &token).await
}

/// Generic async stream variant of [`run_stdio_bridge`] (useful for testing and
/// embedded transports).
pub async fn run_io_bridge<R, W>(
    addr: &str,
    token: &str,
    mut client_in: R,
    mut client_out: W,
) -> anyhow::Result<()>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let stream = TcpStream::connect(addr).await?;
    let _ = stream.set_nodelay(true);
    let (read_half, mut write_half) = stream.into_split();
    let mut auth_line = json!({ "auth": token }).to_string();
    auth_line.push('\n');
    write_half.write_all(auth_line.as_bytes()).await?;
    write_half.flush().await?;

    let mut server_reader = BufReader::new(read_half);
    let mut ack_buf = Vec::new();
    let read = read_bounded_line(&mut server_reader, &mut ack_buf, MAX_AUTH_LINE_BYTES).await?;
    if read == 0 {
        anyhow::bail!("MCP server closed connection during auth handshake");
    }
    let ack: Value = serde_json::from_slice(&ack_buf)?;
    if ack.get("ok").and_then(Value::as_bool) != Some(true) {
        let reason = ack
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unauthorized");
        anyhow::bail!("MCP server rejected auth: {reason}");
    }

    let to_server = tokio::spawn(async move {
        let _ = tokio::io::copy(&mut client_in, &mut write_half).await;
        let _ = write_half.shutdown().await;
    });
    let to_client = tokio::spawn(async move {
        let _ = tokio::io::copy(&mut server_reader, &mut client_out).await;
        let _ = client_out.flush().await;
    });

    tokio::select! {
        _ = to_server => {}
        _ = to_client => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::{ToolError, ToolOutput, ToolSpec};

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "echo".into(),
                description: "Echo a message".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "msg": { "type": "string" } },
                    "required": ["msg"],
                    "additionalProperties": false
                }),
            }
        }

        async fn call(&self, args: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
            let msg = args.get("msg").and_then(Value::as_str).unwrap_or_default();
            Ok(ToolOutput::text(format!("echo:{msg}")))
        }
    }

    #[tokio::test]
    async fn local_mcp_server_and_bridge_round_trip() {
        let server = LocalMcpServer::bind(
            ServerMeta::new("aidaemon-test", "0.1.0").with_instructions("Coordinate peers."),
        )
        .await
        .unwrap();
        let handler: Arc<dyn McpToolHandler> = Arc::new(CoreToolsHandler::new(
            vec![Arc::new(EchoTool) as Arc<dyn Tool>],
            "synthetic-session",
        ));
        let endpoint = server.register_scoped(handler);

        let (mut client_w, bridge_r) = tokio::io::duplex(16 * 1024);
        let (bridge_w, client_r) = tokio::io::duplex(16 * 1024);

        let addr = endpoint.addr().to_string();
        let token = endpoint.token().to_string();
        let bridge_task =
            tokio::spawn(async move { run_io_bridge(&addr, &token, bridge_r, bridge_w).await });

        let mut lines = BufReader::new(client_r).lines();

        // 1. initialize
        client_w
            .write_all(
                format!(
                    "{}\n",
                    json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": { "protocolVersion": PROTOCOL_VERSION }
                    })
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let init: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(init["id"], 1);
        assert_eq!(init["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(init["result"]["serverInfo"]["name"], "aidaemon-test");
        assert_eq!(init["result"]["instructions"], "Coordinate peers.");

        // 2. notifications/initialized (no reply)
        client_w
            .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
            .await
            .unwrap();

        // 3. tools/list
        client_w
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n")
            .await
            .unwrap();
        let list: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(list["id"], 2);
        assert_eq!(list["result"]["tools"][0]["name"], "echo");

        // 4. tools/call
        client_w
            .write_all(
                b"{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"echo\",\"arguments\":{\"msg\":\"hello\"}}}\n",
            )
            .await
            .unwrap();
        let call: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(call["id"], 3);
        assert_eq!(call["result"]["content"][0]["text"], "echo:hello");

        drop(client_w);
        bridge_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn local_mcp_server_rejects_bad_and_revoked_tokens() {
        let server = LocalMcpServer::bind(ServerMeta::new("aidaemon-test", "0.1.0"))
            .await
            .unwrap();
        let handler: Arc<dyn McpToolHandler> = Arc::new(CoreToolsHandler::new(vec![], "s"));
        let endpoint = server.register_scoped(handler);
        let addr = endpoint.addr().to_string();
        let token = endpoint.token().to_string();

        // Wrong token fails handshake.
        let (_, r) = tokio::io::duplex(64);
        let (w, _) = tokio::io::duplex(64);
        let err = run_io_bridge(&addr, "wrong-token", r, w).await.unwrap_err();
        assert!(err.to_string().contains("unauthorized"), "{err}");

        // Dropping endpoint revokes the token.
        drop(endpoint);
        let (_, r) = tokio::io::duplex(64);
        let (w, _) = tokio::io::duplex(64);
        let err = run_io_bridge(&addr, &token, r, w).await.unwrap_err();
        assert!(err.to_string().contains("unauthorized"), "{err}");
    }

    #[tokio::test]
    async fn revoking_ends_a_connection_already_open() {
        let server = LocalMcpServer::bind(ServerMeta::new("aidaemon-test", "0.1.0"))
            .await
            .unwrap();
        let handler: Arc<dyn McpToolHandler> = Arc::new(CoreToolsHandler::new(vec![], "s"));
        let endpoint = server.register_scoped(handler);
        let mut stream = TcpStream::connect(endpoint.addr()).await.unwrap();
        stream
            .write_all(format!("{}\n", json!({ "auth": endpoint.token() })).as_bytes())
            .await
            .unwrap();
        let mut reader = BufReader::new(stream);
        let mut ack = String::new();
        reader.read_line(&mut ack).await.unwrap();
        assert!(ack.contains("true"), "{ack}");
        endpoint.revoke();
        let mut rest = String::new();
        let read = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            reader.read_line(&mut rest),
        )
        .await
        .expect("the server closes a revoked connection")
        .unwrap();
        assert_eq!(read, 0, "{rest}");
    }
}
