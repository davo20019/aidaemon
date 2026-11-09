//! Stdio transport: newline-delimited JSON-RPC 2.0 to a child process.

use std::collections::{HashMap, VecDeque};
use std::ffi::{OsStr, OsString};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::{Tool, ToolAnnotations};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::tool::{tools_from_defs, McpToolDef};

/// The protocol version this client asks for.
pub const PROTOCOL_VERSION: &str = "2025-06-18";
/// Versions this client accepts in the server's `initialize` answer.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Lines longer than this are dropped (the matching request then times out).
const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;
/// Upper bound on `tools/list` pages, against servers that loop forever.
const MAX_LIST_PAGES: usize = 100;
/// What a [`StderrCallback`] gets of one line at most.
pub const MAX_STDERR_LINE_BYTES: usize = 8 * 1024;

/// Variables passed through from the parent environment by default. All
/// others are stripped so API keys of the host process never reach a server
/// unless given explicitly.
pub const SAFE_ENV_KEYS: &[&str] = &[
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
    "SYSTEMROOT",
    "XDG_RUNTIME_DIR",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "NODE_PATH",
    "NPM_CONFIG_PREFIX",
    "NVM_DIR",
];

/// Everything that can go wrong talking to a server.
#[derive(Debug, Clone, thiserror::Error)]
pub enum McpError {
    #[error("failed to start MCP server `{server}`: {message}")]
    Spawn { server: String, message: String },
    #[error("MCP server `{server}` did not answer `{method}` within {timeout:?}")]
    Timeout {
        server: String,
        method: String,
        timeout: Duration,
    },
    #[error("MCP server `{server}` is not running{}", stderr_suffix(.stderr))]
    Closed { server: String, stderr: String },
    #[error("MCP server `{server}` returned error {code} for `{method}`: {message}")]
    Rpc {
        server: String,
        method: String,
        code: i64,
        message: String,
        data: Option<Value>,
    },
    #[error("MCP server `{server}` protocol error: {message}")]
    Protocol { server: String, message: String },
    #[error("request to MCP server `{server}` was cancelled")]
    Cancelled { server: String },
}

fn stderr_suffix(stderr: &str) -> String {
    let tail = stderr.trim();
    if tail.is_empty() {
        String::new()
    } else {
        format!("; stderr: {tail}")
    }
}

/// Knobs for [`McpServer::spawn_with`].
#[derive(Debug, Clone)]
pub struct McpOptions {
    /// Per-request timeout (including `tools/call`).
    pub request_timeout: Duration,
    /// Timeout for the `initialize` handshake.
    pub init_timeout: Duration,
    /// Bytes of server stderr kept (the most recent ones).
    pub stderr_capacity: usize,
    /// Pass the whole parent environment instead of [`SAFE_ENV_KEYS`].
    pub inherit_env: bool,
    /// Working directory for the child.
    pub cwd: Option<std::path::PathBuf>,
    /// Sent as `clientInfo`.
    pub client_name: String,
    pub client_version: String,
    /// Annotations every tool of this server gets, instead of the ones
    /// mapped from the server's hints. The MCP specification calls those
    /// hints untrusted: set `Some(ToolAnnotations::DESTRUCTIVE)` to judge
    /// every call at its worst whatever the server claims.
    pub annotations: Option<ToolAnnotations>,
    /// Called with each non-blank line the server writes to stderr, for
    /// example to log it. [`McpServer::stderr_tail`] works with or without.
    pub on_stderr: Option<StderrCallback>,
}

/// Receives a server's stderr one line at a time, lossily decoded, without
/// the line ending, and cut to [`MAX_STDERR_LINE_BYTES`].
#[derive(Clone)]
pub struct StderrCallback(Arc<dyn Fn(&str) + Send + Sync>);

impl StderrCallback {
    pub fn new(callback: impl Fn(&str) + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }

    fn emit(&self, line: &[u8]) {
        let line = String::from_utf8_lossy(line);
        let line = line.trim_end();
        if !line.is_empty() {
            (self.0)(line);
        }
    }
}

impl std::fmt::Debug for StderrCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StderrCallback")
    }
}

impl Default for McpOptions {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(60),
            init_timeout: Duration::from_secs(30),
            stderr_capacity: 64 * 1024,
            inherit_env: false,
            cwd: None,
            client_name: "aidaemon".to_string(),
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            annotations: None,
            on_stderr: None,
        }
    }
}

/// How to start a server: everything [`McpServer::connect`] needs, and what
/// [`McpServer::respawn`] starts again.
#[derive(Clone)]
pub struct McpServerSpec {
    pub name: String,
    pub command: OsString,
    pub args: Vec<OsString>,
    /// Added to the scrubbed child environment (see [`SAFE_ENV_KEYS`]).
    pub env: Vec<(OsString, OsString)>,
    pub options: McpOptions,
}

impl McpServerSpec {
    pub fn new(name: impl Into<String>, command: impl Into<OsString>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
            options: McpOptions::default(),
        }
    }

    pub fn with_args<I>(mut self, args: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<OsString>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_env<I, K, V>(mut self, env: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<OsString>,
        V: Into<OsString>,
    {
        self.env = env.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        self
    }

    pub fn with_options(mut self, options: McpOptions) -> Self {
        self.options = options;
        self
    }
}

/// Environment values are credentials, so only their names are printed.
impl std::fmt::Debug for McpServerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerSpec")
            .field("name", &self.name)
            .field("command", &self.command)
            .field("args", &self.args)
            .field(
                "env_keys",
                &self.env.iter().map(|(key, _)| key).collect::<Vec<_>>(),
            )
            .field("options", &self.options)
            .finish()
    }
}

type Reply = Result<Value, RpcFailure>;

#[derive(Debug)]
struct RpcFailure {
    code: i64,
    message: String,
    data: Option<Value>,
}

/// State shared with the background reader. It never holds the child, so
/// dropping the last [`McpServer`]/tool handle kills the process.
struct Shared {
    pending: Mutex<HashMap<u64, oneshot::Sender<Reply>>>,
    closed: AtomicBool,
    stderr: Mutex<VecDeque<u8>>,
    stderr_capacity: usize,
    stdin: tokio::sync::Mutex<Option<ChildStdin>>,
}

impl Shared {
    fn stderr_tail(&self) -> String {
        let buf = self.stderr.lock().expect("stderr poisoned");
        let (a, b) = buf.as_slices();
        let mut bytes = Vec::with_capacity(a.len() + b.len());
        bytes.extend_from_slice(a);
        bytes.extend_from_slice(b);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        // Dropping the senders wakes every waiter with "closed".
        self.pending.lock().expect("pending poisoned").clear();
    }

    async fn write_line(&self, message: &Value) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(message)?;
        line.push(b'\n');
        let mut guard = self.stdin.lock().await;
        let stdin = guard
            .as_mut()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "stdin closed"))?;
        stdin.write_all(&line).await?;
        stdin.flush().await
    }
}

pub(crate) struct Inner {
    pub(crate) name: String,
    pub(crate) annotations: Option<ToolAnnotations>,
    shared: Arc<Shared>,
    next_id: AtomicU64,
    request_timeout: Duration,
    child: Mutex<Option<Child>>,
    tasks: Vec<JoinHandle<()>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(child) = self.child.get_mut().ok().and_then(Option::as_mut) {
            let _ = child.start_kill();
        }
        for task in &self.tasks {
            task.abort();
        }
        self.shared.close();
    }
}

impl Inner {
    fn closed_error(&self) -> McpError {
        McpError::Closed {
            server: self.name.clone(),
            stderr: self.shared.stderr_tail(),
        }
    }

    pub(crate) async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
    ) -> Result<Value, McpError> {
        self.request_cancellable(method, params, timeout, None)
            .await
    }

    pub(crate) async fn request_cancellable(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
        cancel: Option<&aidaemon_core::CancellationToken>,
    ) -> Result<Value, McpError> {
        if self.shared.closed.load(Ordering::SeqCst) {
            return Err(self.closed_error());
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.shared
            .pending
            .lock()
            .expect("pending poisoned")
            .insert(id, tx);
        // The reader may have closed between the check and the insert.
        if self.shared.closed.load(Ordering::SeqCst) {
            self.forget(id);
            return Err(self.closed_error());
        }
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        if self.shared.write_line(&message).await.is_err() {
            self.forget(id);
            return Err(self.closed_error());
        }

        let timeout = timeout.unwrap_or(self.request_timeout);
        let never = aidaemon_core::CancellationToken::new();
        let cancel = cancel.unwrap_or(&never);
        let outcome = tokio::select! {
            reply = tokio::time::timeout(timeout, rx) => reply,
            _ = cancel.cancelled() => {
                self.forget(id);
                self.cancel_remote(id, "cancelled by client").await;
                return Err(McpError::Cancelled { server: self.name.clone() });
            }
        };
        match outcome {
            Ok(Ok(Ok(result))) => Ok(result),
            Ok(Ok(Err(failure))) => Err(McpError::Rpc {
                server: self.name.clone(),
                method: method.to_string(),
                code: failure.code,
                message: failure.message,
                data: failure.data,
            }),
            // Sender dropped: the reader saw EOF.
            Ok(Err(_)) => Err(self.closed_error()),
            Err(_) => {
                self.forget(id);
                self.cancel_remote(id, "request timed out").await;
                Err(McpError::Timeout {
                    server: self.name.clone(),
                    method: method.to_string(),
                    timeout,
                })
            }
        }
    }

    pub(crate) async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        let message = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.shared
            .write_line(&message)
            .await
            .map_err(|_| self.closed_error())
    }

    fn forget(&self, id: u64) {
        self.shared
            .pending
            .lock()
            .expect("pending poisoned")
            .remove(&id);
    }

    async fn cancel_remote(&self, id: u64, reason: &str) {
        let _ = self
            .notify(
                "notifications/cancelled",
                json!({"requestId": id, "reason": reason}),
            )
            .await;
    }
}

/// A connected MCP server. Cloning shares the connection; the child process
/// is killed when the server and every tool taken from it are dropped, or
/// on [`McpServer::shutdown`].
#[derive(Clone)]
pub struct McpServer {
    inner: Arc<Inner>,
    info: Arc<ServerInfo>,
    spec: Arc<McpServerSpec>,
}

/// What the server said about itself during the handshake.
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub protocol_version: String,
    /// `serverInfo` (`name`, `version`, ...).
    pub server_info: Value,
    pub capabilities: Value,
    pub instructions: Option<String>,
    /// Raw tool definitions from `tools/list`, all pages.
    pub tools: Vec<Value>,
}

impl std::fmt::Debug for McpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServer")
            .field("name", &self.inner.name)
            .field("protocol_version", &self.info.protocol_version)
            .field("tools", &self.info.tools.len())
            .finish()
    }
}

impl McpServer {
    /// Start `command args...` with `env` added to a scrubbed environment
    /// (see [`SAFE_ENV_KEYS`]), run the handshake and list its tools.
    pub async fn spawn<I, S, E, K, V>(
        name: impl Into<String>,
        command: impl AsRef<OsStr>,
        args: I,
        env: E,
    ) -> Result<Self, McpError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
        E: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        Self::spawn_with(name, command, args, env, McpOptions::default()).await
    }

    /// [`McpServer::spawn`] with explicit options.
    pub async fn spawn_with<I, S, E, K, V>(
        name: impl Into<String>,
        command: impl AsRef<OsStr>,
        args: I,
        env: E,
        options: McpOptions,
    ) -> Result<Self, McpError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
        E: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        Self::connect(
            McpServerSpec::new(name, command.as_ref())
                .with_args(args.into_iter().map(|arg| arg.as_ref().to_os_string()))
                .with_env(
                    env.into_iter()
                        .map(|(k, v)| (k.as_ref().to_os_string(), v.as_ref().to_os_string())),
                )
                .with_options(options),
        )
        .await
    }

    /// Start the server `spec` describes, run the handshake and list its
    /// tools.
    pub async fn connect(spec: McpServerSpec) -> Result<Self, McpError> {
        let name = spec.name.clone();
        let options = &spec.options;
        let mut cmd = Command::new(&spec.command);
        cmd.args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if !options.inherit_env {
            cmd.env_clear();
            for (key, value) in std::env::vars_os() {
                if key.to_str().is_some_and(|k| SAFE_ENV_KEYS.contains(&k)) {
                    cmd.env(key, value);
                }
            }
        }
        cmd.envs(spec.env.iter().map(|(key, value)| (key, value)));
        if let Some(cwd) = &options.cwd {
            cmd.current_dir(cwd);
        }

        let mut child = cmd.spawn().map_err(|e| McpError::Spawn {
            server: name.clone(),
            message: e.to_string(),
        })?;
        let missing = |what: &str| McpError::Spawn {
            server: name.clone(),
            message: format!("could not capture {what}"),
        };
        let stdin = child.stdin.take().ok_or_else(|| missing("stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| missing("stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| missing("stderr"))?;

        let shared = Arc::new(Shared {
            pending: Mutex::default(),
            closed: AtomicBool::new(false),
            stderr: Mutex::new(VecDeque::new()),
            stderr_capacity: options.stderr_capacity,
            stdin: tokio::sync::Mutex::new(Some(stdin)),
        });
        let (stderr_done, stderr_closed) = oneshot::channel();
        let tasks = vec![
            tokio::spawn(read_stdout(stdout, shared.clone(), stderr_closed)),
            tokio::spawn(read_stderr(
                stderr,
                shared.clone(),
                options.on_stderr.clone(),
                stderr_done,
            )),
        ];
        let inner = Arc::new(Inner {
            name: name.clone(),
            annotations: options.annotations,
            shared,
            next_id: AtomicU64::new(1),
            request_timeout: options.request_timeout,
            child: Mutex::new(Some(child)),
            tasks,
        });

        let info = handshake(&inner, options).await?;
        Ok(Self {
            inner,
            info: Arc::new(info),
            spec: Arc::new(spec),
        })
    }

    /// Stop this server and start it again from the same spec. Tools taken
    /// from this handle fail afterwards; take new ones from the returned
    /// server. If the new process does not come up the old one stays
    /// stopped.
    pub async fn respawn(&self) -> Result<Self, McpError> {
        self.respawn_as((*self.spec).clone()).await
    }

    /// [`McpServer::respawn`] with `env` in place of the spec's environment
    /// (for credentials that changed since the server started).
    pub async fn respawn_with_env<I, K, V>(&self, env: I) -> Result<Self, McpError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<OsString>,
        V: Into<OsString>,
    {
        self.respawn_as((*self.spec).clone().with_env(env)).await
    }

    async fn respawn_as(&self, spec: McpServerSpec) -> Result<Self, McpError> {
        self.shutdown().await;
        Self::connect(spec).await
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// How this server was started.
    pub fn spec(&self) -> &McpServerSpec {
        &self.spec
    }

    pub fn info(&self) -> &ServerInfo {
        &self.info
    }

    /// The server's tools as kernel tools named `{server}__{tool}`.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        tools_from_defs(&self.inner, &self.info.tools)
            .into_iter()
            .map(|t| Arc::new(t) as Arc<dyn Tool>)
            .collect()
    }

    /// The tools as concrete [`crate::McpTool`]s.
    pub fn mcp_tools(&self) -> Vec<crate::McpTool> {
        tools_from_defs(&self.inner, &self.info.tools)
    }

    /// Fetch the tool list again (for example after a
    /// `notifications/tools/list_changed`). Returns raw definitions.
    pub async fn list_tools(&self) -> Result<Vec<Value>, McpError> {
        list_all_tools(&self.inner).await
    }

    /// Send an arbitrary request (escape hatch for methods not wrapped here).
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        self.inner.request(method, params, None).await
    }

    /// Whether the connection is still usable.
    pub fn is_alive(&self) -> bool {
        !self.inner.shared.closed.load(Ordering::SeqCst)
    }

    /// The most recent server stderr output (bounded).
    pub fn stderr_tail(&self) -> String {
        self.inner.shared.stderr_tail()
    }

    /// Close stdin (the MCP stdio shutdown signal), give the server a moment
    /// to exit, then kill it. Tools taken from this server fail afterwards.
    pub async fn shutdown(&self) {
        self.inner.shared.stdin.lock().await.take();
        let child = self.inner.child.lock().expect("child poisoned").take();
        if let Some(mut child) = child {
            if tokio::time::timeout(Duration::from_secs(2), child.wait())
                .await
                .is_err()
            {
                let _ = child.kill().await;
            }
        }
        self.inner.shared.close();
    }
}

async fn handshake(inner: &Inner, options: &McpOptions) -> Result<ServerInfo, McpError> {
    let result = inner
        .request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": options.client_name, "version": options.client_version},
            }),
            Some(options.init_timeout),
        )
        .await?;
    let protocol_version = result
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if !SUPPORTED_PROTOCOL_VERSIONS.contains(&protocol_version.as_str()) {
        return Err(McpError::Protocol {
            server: inner.name.clone(),
            message: format!("unsupported protocol version `{protocol_version}`"),
        });
    }
    inner.notify("notifications/initialized", json!({})).await?;
    let has_tools = result
        .get("capabilities")
        .is_some_and(|c| c.get("tools").is_some());
    let tools = if has_tools {
        list_all_tools(inner).await?
    } else {
        Vec::new()
    };
    Ok(ServerInfo {
        protocol_version,
        server_info: result.get("serverInfo").cloned().unwrap_or(Value::Null),
        capabilities: result.get("capabilities").cloned().unwrap_or(Value::Null),
        instructions: result
            .get("instructions")
            .and_then(Value::as_str)
            .map(str::to_string),
        tools,
    })
}

async fn list_all_tools(inner: &Inner) -> Result<Vec<Value>, McpError> {
    let mut tools = Vec::new();
    let mut cursor: Option<String> = None;
    let mut seen = std::collections::HashSet::new();
    for _ in 0..MAX_LIST_PAGES {
        let params = match &cursor {
            Some(c) => json!({"cursor": c}),
            None => json!({}),
        };
        let page = inner.request("tools/list", params, None).await?;
        let defs =
            page.get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| McpError::Protocol {
                    server: inner.name.clone(),
                    message: "tools/list result has no `tools` array".into(),
                })?;
        tools.extend(defs.iter().filter(|d| McpToolDef::is_valid(d)).cloned());
        match page.get("nextCursor").and_then(Value::as_str) {
            Some(next) if !next.is_empty() => {
                if !seen.insert(next.to_string()) {
                    return Err(McpError::Protocol {
                        server: inner.name.clone(),
                        message: format!("tools/list repeated cursor `{next}`"),
                    });
                }
                cursor = Some(next.to_string());
            }
            _ => return Ok(tools),
        }
    }
    Err(McpError::Protocol {
        server: inner.name.clone(),
        message: format!("tools/list exceeded {MAX_LIST_PAGES} pages"),
    })
}

/// Read one line of at most [`MAX_LINE_BYTES`]. `Ok(None)` at EOF; an
/// oversized line is consumed and returned empty.
async fn read_bounded_line<R: AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
    buf: &mut Vec<u8>,
) -> std::io::Result<Option<()>> {
    buf.clear();
    let n = (&mut *reader)
        .take(MAX_LINE_BYTES as u64 + 1)
        .read_until(b'\n', buf)
        .await?;
    if n == 0 {
        return Ok(None);
    }
    if buf.len() > MAX_LINE_BYTES && buf.last() != Some(&b'\n') {
        // Skip the rest of the oversized line.
        let mut sink = Vec::new();
        loop {
            sink.clear();
            let m = (&mut *reader)
                .take(64 * 1024)
                .read_until(b'\n', &mut sink)
                .await?;
            if m == 0 || sink.last() == Some(&b'\n') {
                break;
            }
        }
        buf.clear();
    }
    Ok(Some(()))
}

async fn read_stdout<R: AsyncRead + Unpin>(
    stdout: R,
    shared: Arc<Shared>,
    stderr_closed: oneshot::Receiver<()>,
) {
    let mut reader = BufReader::new(stdout);
    let mut buf = Vec::new();
    while let Ok(Some(())) = read_bounded_line(&mut reader, &mut buf).await {
        let Ok(message) = serde_json::from_slice::<Value>(&buf) else {
            continue; // blank, oversized, or not JSON (stray logging)
        };
        let messages = match message {
            Value::Array(batch) => batch,
            single => vec![single],
        };
        for message in messages {
            handle_message(&shared, message).await;
        }
    }
    // Let the last stderr lines (often the crash reason) land before
    // waiters build their "not running" errors.
    let _ = tokio::time::timeout(Duration::from_millis(250), stderr_closed).await;
    shared.close();
}

async fn handle_message(shared: &Shared, message: Value) {
    let method = message.get("method").and_then(Value::as_str);
    let id = message.get("id").filter(|v| !v.is_null()).cloned();
    match (method, id) {
        // Response to one of our requests.
        (None, Some(id)) => {
            let Some(id) = id
                .as_u64()
                .or_else(|| id.as_str().and_then(|s| s.parse().ok()))
            else {
                return;
            };
            let waiter = shared.pending.lock().expect("pending poisoned").remove(&id);
            let Some(waiter) = waiter else { return };
            let reply = if let Some(error) = message.get("error") {
                Err(RpcFailure {
                    code: error.get("code").and_then(Value::as_i64).unwrap_or(0),
                    message: error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                        .to_string(),
                    data: error.get("data").cloned(),
                })
            } else {
                Ok(message.get("result").cloned().unwrap_or(Value::Null))
            };
            let _ = waiter.send(reply);
        }
        // Server-to-client request: answer ping, decline the rest (this
        // client advertises no capabilities).
        (Some(method), Some(id)) => {
            let reply = if method == "ping" {
                json!({"jsonrpc": "2.0", "id": id, "result": {}})
            } else {
                json!({"jsonrpc": "2.0", "id": id, "error": {
                    "code": -32601, "message": format!("method `{method}` not supported")
                }})
            };
            let _ = shared.write_line(&reply).await;
        }
        // Notifications (logging, list_changed, progress) are ignored.
        _ => {}
    }
}

async fn read_stderr<R: AsyncRead + Unpin>(
    mut stderr: R,
    shared: Arc<Shared>,
    on_stderr: Option<StderrCallback>,
    _done: oneshot::Sender<()>,
) {
    let mut chunk = [0u8; 4096];
    let mut line = Vec::new();
    loop {
        match stderr.read(&mut chunk).await {
            Ok(0) | Err(_) => {
                // A last line with no newline (often a crash message).
                if let Some(callback) = &on_stderr {
                    callback.emit(&line);
                }
                return;
            }
            Ok(n) => {
                {
                    let mut buf = shared.stderr.lock().expect("stderr poisoned");
                    buf.extend(&chunk[..n]);
                    let excess = buf.len().saturating_sub(shared.stderr_capacity);
                    buf.drain(..excess);
                }
                let Some(callback) = &on_stderr else { continue };
                for &byte in &chunk[..n] {
                    if byte == b'\n' {
                        callback.emit(&line);
                        line.clear();
                    } else if line.len() < MAX_STDERR_LINE_BYTES {
                        line.push(byte);
                    }
                }
            }
        }
    }
}
