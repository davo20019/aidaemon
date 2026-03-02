use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc::{self, Sender};
use std::time::{Duration, Instant};

use aes_gcm::aead::{Aead, Payload};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use base64::Engine;
use futures::{SinkExt, StreamExt};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use p256::ecdh::diffie_hellman;
use p256::elliptic_curve::sec1::ToEncodedPoint;
use p256::{PublicKey, SecretKey};
use portable_pty::{CommandBuilder, ExitStatus, MasterPty, NativePtySystem, PtySize, PtySystem};
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{debug, error, info, warn};

use crate::config::{resolve_from_keychain, store_in_keychain, AppConfig};

const HEARTBEAT_MS: u64 = 25_000;
const RECONNECT_MS: u64 = 3_000;
// Keep encrypted payloads comfortably below the broker's 64 KiB WS frame cap.
// JSON envelope + AES-GCM overhead + base64 expansion can otherwise exceed 64 KiB
// and force daemon socket disconnects under high-output workloads.
const MAX_ENCRYPTED_PAYLOAD_BYTES: usize = 44 * 1024;
const MAX_AGENT_ARGS: usize = 24;
const MAX_AGENT_ARG_CHARS: usize = 256;
const MAX_UPLOAD_CHUNK_BYTES: usize = 32 * 1024;
const MAX_PENDING_UPLOADS_PER_SESSION: usize = 4;
const UPLOAD_IDLE_TTL_SECS: u64 = 300;
const REVIEW_DEFAULT_TIMEOUT_SECS: u64 = 180;
const REVIEW_MAX_OUTPUT_CHARS: usize = 20_000;
const REVIEW_MAX_CONTEXT_CHARS: usize = 220_000;
const REVIEW_MAX_SECTION_CHARS: usize = 90_000;
const REVIEW_MAX_INCLUDED_DIFF_FILES: usize = 24;
const REVIEW_MAX_FILE_CHANGED_LINES: u64 = 900;
const REVIEW_MAX_FILE_DIFF_CHARS: usize = 12_000;
const REVIEW_SUMMARY_MAX_CHARS: usize = 4_000;
const REVIEW_LIST_ITEM_MAX_CHARS: usize = 600;
const REVIEW_LIST_MAX_ITEMS: usize = 20;
const REVIEW_STREAM_CHUNK_MAX_BYTES: usize = 2048;
const REVIEW_STREAM_REPLAY_MAX_FRAMES: usize = 1024;
const REVIEW_STREAM_REPLAY_MAX_BYTES: usize = 2 * 1024 * 1024;
const REPLAY_MAX_FRAMES: usize = 256;
const REPLAY_MAX_BYTES: usize = 4 * 1024 * 1024;
const PTY_DEFAULT_ROWS: u16 = 36;
const PTY_DEFAULT_COLS: u16 = 120;
const KEY_INFO: &[u8] = b"aidaemon-terminal-v1";
const DAEMON_BOOTSTRAP_SIGNING_SALT: &[u8] = b"aidaemon-daemon-bootstrap-v1";
const DAEMON_BOOTSTRAP_SIGNING_INFO: &[u8] = b"hmac-signing-key";
const TERMINAL_DAEMON_KEYCHAIN_FIELD: &str = "terminal_daemon_private_key_v1";
const SUPPORTED_TERMINAL_AGENTS: &[&str] = &["codex", "claude", "gemini", "opencode"];
type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Deserialize)]
struct StoredDaemonKey {
    #[serde(default)]
    private_key_b64: Option<String>,
    #[serde(default)]
    private_jwk: Option<StoredPrivateJwk>,
    #[allow(dead_code)]
    created_at: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StoredPrivateJwk {
    d: String,
}

#[derive(Debug, Serialize)]
struct StoredDaemonKeyOnDisk {
    private_key_b64: String,
    created_at: String,
}

#[derive(Clone)]
struct KeyMaterial {
    private_key: SecretKey,
    public_raw: Vec<u8>,
    fingerprint: String,
}

#[derive(Clone)]
struct RuntimeConfig {
    ws_url: String,
    user_id: String,
    device_id: String,
    shell: String,
    default_cwd: PathBuf,
    inbox_dir: PathBuf,
    max_upload_bytes: usize,
    review_profiles: HashMap<String, ReviewProfile>,
    auth: BridgeAuth,
}

#[derive(Clone)]
enum BridgeAuth {
    StaticToken(String),
    BotProof {
        mint_url: String,
        bot_tokens: Vec<String>,
        fallback_static_token: Option<String>,
    },
}

#[derive(Debug, Serialize)]
struct DaemonBootstrapMintRequest<'a> {
    user_id: &'a str,
    device_id: &'a str,
    ts: i64,
    nonce: String,
    sig: String,
}

#[derive(Debug, Deserialize)]
struct DaemonConnectTokenResponse {
    ok: bool,
    daemon_connect_token: Option<String>,
    error: Option<String>,
    message: Option<String>,
}

struct CryptoSession {
    session_id: String,
    cipher: Aes256Gcm,
    send_counter: u64,
    recv_counter: u64,
    agent: Option<String>,
    agent_args: Vec<String>,
    bootstrapped_agent: bool,
}

struct ReplayFrame {
    seq: u64,
    data: String,
}

struct ReviewStreamFrame {
    seq: u64,
    request_id: String,
    stream: String,
    data: String,
}

struct ActiveSession {
    crypto: CryptoSession,
    shell: ShellProcess,
    cwd: PathBuf,
    replay: VecDeque<ReplayFrame>,
    replay_bytes: usize,
    next_stdout_seq: u64,
    review_stream_replay: VecDeque<ReviewStreamFrame>,
    review_stream_replay_bytes: usize,
    next_review_stream_seq: u64,
    pending_uploads: HashMap<String, PendingUpload>,
    review_job: Option<ReviewJob>,
    last_review_progress: Option<Value>,
    last_review_result: Option<Value>,
}

struct ReviewJob {
    request_id: String,
    handle: tokio::task::JoinHandle<()>,
}

struct PendingUpload {
    upload_id: String,
    filename: String,
    mime_type: String,
    expected_bytes: usize,
    bytes: Vec<u8>,
    chunks_received: usize,
    caption: Option<String>,
    touched_at: Instant,
}

struct UploadCommitResult {
    status_message: String,
    prompt_for_agent: Option<String>,
}

struct ShellProcess {
    child: Box<dyn portable_pty::Child + Send + Sync>,
    stdin_tx: Option<Sender<Vec<u8>>>,
    master: Box<dyn MasterPty + Send>,
}

#[derive(Debug)]
enum ShellEvent {
    Output {
        session_id: String,
        data: String,
    },
    ReviewProgress {
        session_id: String,
        request_id: String,
        stage: String,
        message: String,
    },
    ReviewResult {
        session_id: String,
        request_id: String,
        payload: Value,
    },
    ReviewError {
        session_id: String,
        request_id: String,
        code: String,
        message: String,
    },
    ReviewStream {
        session_id: String,
        request_id: String,
        stream: String,
        data: String,
    },
}

#[derive(Debug, Clone)]
struct ReviewProfile {
    command: String,
    args: Vec<String>,
    timeout_secs: u64,
    max_output_chars: usize,
}

#[derive(Debug, Clone, Default)]
struct ReviewContextStats {
    total_changed_files: usize,
    included_files: usize,
    skipped_generated_files: usize,
    skipped_binary_files: usize,
    skipped_large_files: usize,
    skipped_overflow_files: usize,
}

#[derive(Debug, Clone)]
struct ReviewFileChange {
    path: String,
    added: Option<u64>,
    deleted: Option<u64>,
    binary: bool,
}

#[derive(Debug, Clone, Copy)]
enum ReviewScope {
    Diff,
    Plan,
    Both,
}

impl ReviewScope {
    fn from_raw(value: Option<&str>) -> Self {
        match value.unwrap_or("diff").trim().to_ascii_lowercase().as_str() {
            "plan" => Self::Plan,
            "both" => Self::Both,
            _ => Self::Diff,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Diff => "diff",
            Self::Plan => "plan",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ReviewDiffBase {
    WorkingTree,
    Staged,
}

impl ReviewDiffBase {
    fn from_raw(value: Option<&str>) -> Self {
        match value
            .unwrap_or("working_tree")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "staged" => Self::Staged,
            _ => Self::WorkingTree,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::WorkingTree => "working_tree",
            Self::Staged => "staged",
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReviewResponseCandidate {
    verdict: Option<String>,
    blocking: Option<Vec<String>>,
    risks: Option<Vec<String>>,
    suggestions: Option<Vec<String>>,
    findings: Option<Vec<ReviewFindingCandidate>>,
    summary: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReviewFindingCandidate {
    severity: Option<String>,
    file: Option<String>,
    line: Option<Value>,
    issue: Option<String>,
    fix: Option<String>,
}

impl ReviewResponseCandidate {
    fn has_review_signal(&self) -> bool {
        self.verdict
            .as_deref()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
            || self
                .summary
                .as_deref()
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false)
            || self
                .blocking
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
            || self
                .findings
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
            || self.risks.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
            || self
                .suggestions
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
    }
}

impl ShellProcess {
    async fn spawn(
        shell: &str,
        cwd: &Path,
        session_id: &str,
        events_tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    ) -> anyhow::Result<Self> {
        let pty_system = NativePtySystem::default();
        let pair = pty_system.openpty(PtySize {
            rows: PTY_DEFAULT_ROWS,
            cols: PTY_DEFAULT_COLS,
            pixel_width: 0,
            pixel_height: 0,
        })?;

        let mut cmd = CommandBuilder::new(shell);
        if cfg!(windows) {
            cmd.arg("-NoLogo");
        } else {
            cmd.arg("-i");
        }
        cmd.cwd(cwd);
        // Start from a clean environment so parent process/session markers
        // (including Claude nesting vars) cannot leak into the bridged PTY.
        cmd.env_clear();
        // Prevent nested Claude Code session detection when launching `claude`
        // inside the bridged terminal shell.
        cmd.env_remove("CLAUDECODE");
        cmd.env_remove("CLAUDE_CODE");
        for (k, v) in build_child_env() {
            cmd.env(k, v);
        }

        let child = pair.slave.spawn_command(cmd)?;
        drop(pair.slave);

        let mut reader = pair.master.try_clone_reader()?;
        let writer = pair.master.take_writer()?;

        let (stdin_tx, stdin_rx) = mpsc::channel::<Vec<u8>>();
        let read_session_id = session_id.to_string();
        std::thread::Builder::new()
            .name("terminal-bridge-pty-reader".to_string())
            .spawn(move || {
                let mut buf = [0u8; 4096];
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            if events_tx
                                .send(ShellEvent::Output {
                                    session_id: read_session_id.clone(),
                                    data: String::from_utf8_lossy(&buf[..n]).to_string(),
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            })?;

        std::thread::Builder::new()
            .name("terminal-bridge-pty-writer".to_string())
            .spawn(move || {
                let mut writer = writer;
                while let Ok(bytes) = stdin_rx.recv() {
                    if bytes.is_empty() {
                        continue;
                    }
                    if writer.write_all(&bytes).is_err() {
                        break;
                    }
                    let _ = writer.flush();
                }
            })?;

        Ok(Self {
            child,
            stdin_tx: Some(stdin_tx),
            master: pair.master,
        })
    }

    async fn write_stdin(&mut self, data: &str) -> anyhow::Result<()> {
        let Some(tx) = self.stdin_tx.as_ref() else {
            anyhow::bail!("shell stdin channel closed");
        };
        tx.send(data.as_bytes().to_vec())
            .map_err(|_| anyhow::anyhow!("failed to send stdin to PTY writer"))?;
        Ok(())
    }

    fn resize(&mut self, cols: u16, rows: u16) -> anyhow::Result<()> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }
        self.master.resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        Ok(())
    }

    async fn stop(&mut self) {
        self.stdin_tx.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn try_wait_code(&mut self) -> anyhow::Result<Option<i32>> {
        match self.child.try_wait()? {
            Some(status) => Ok(Some(exit_status_code(status))),
            None => Ok(None),
        }
    }
}

fn exit_status_code(status: ExitStatus) -> i32 {
    status.exit_code() as i32
}

fn validate_daemon_ws_url(ws_url: &str) -> anyhow::Result<()> {
    let parsed = reqwest::Url::parse(ws_url)?;
    match parsed.scheme() {
        "wss" => Ok(()),
        other => anyhow::bail!(
            "terminal daemon websocket URL must use wss:// (found {}://)",
            other
        ),
    }
}

#[cfg(unix)]
fn set_owner_only_permissions(path: &Path, mode: u32) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_owner_only_permissions(_path: &Path, _mode: u32) -> anyhow::Result<()> {
    Ok(())
}

pub fn spawn_if_configured(config: &AppConfig) {
    if !config.terminal.effective_bridge_enabled() {
        info!("Terminal bridge disabled by config ([terminal].bridge_enabled = false)");
        return;
    }

    let user_id = resolve_daemon_user_id(config, config.terminal.effective_daemon_user_id());
    let Some(user_id) = user_id else {
        warn!("Terminal bridge disabled: unable to resolve Telegram owner user_id");
        return;
    };

    let ws_url = config.terminal.effective_daemon_ws_url();
    if let Err(err) = validate_daemon_ws_url(&ws_url) {
        warn!(
            error = %err,
            ws_url = %ws_url,
            "Terminal bridge disabled: insecure or invalid daemon websocket URL"
        );
        return;
    }

    let shell = config
        .terminal
        .effective_daemon_shell()
        .or_else(|| std::env::var("SHELL").ok())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "/bin/bash".to_string());

    let default_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let inbox_dir = resolve_terminal_inbox_dir(&config.files.inbox_dir, &default_cwd);
    let max_upload_bytes_u64 = if config.files.enabled {
        config.files.max_file_size_mb.saturating_mul(1_048_576)
    } else {
        0
    };
    let max_upload_bytes = usize::try_from(max_upload_bytes_u64).unwrap_or(usize::MAX);
    let device_id = config
        .terminal
        .effective_daemon_device_id()
        .unwrap_or_else(default_device_id);

    let bot_tokens = resolve_daemon_bot_tokens(config);
    let static_token = config.terminal.effective_daemon_connect_token();
    let allow_static_fallback = config.terminal.effective_allow_static_token_fallback();
    let auth = if !bot_tokens.is_empty() {
        let mint_url = match daemon_connect_token_mint_url(&ws_url) {
            Ok(url) => url,
            Err(err) => {
                warn!(
                    error = %err,
                    "Terminal bridge disabled: invalid daemon websocket URL for token minting"
                );
                return;
            }
        };
        let fallback_static_token = if allow_static_fallback {
            static_token.clone()
        } else {
            None
        };
        if fallback_static_token.is_some() {
            warn!(
                "Terminal bridge auth mode: auto-bootstrap daemon token with static token fallback enabled"
            );
        } else if static_token.is_some() {
            info!(
                "Terminal bridge auth mode: auto-bootstrap daemon token; static token fallback disabled"
            );
        } else {
            info!("Terminal bridge auth mode: auto-bootstrap daemon token");
        }
        BridgeAuth::BotProof {
            mint_url,
            bot_tokens,
            fallback_static_token,
        }
    } else if let Some(token) = static_token {
        warn!("Terminal bridge auth mode: static daemon token (no bot-proof auto-bootstrap)");
        BridgeAuth::StaticToken(token)
    } else {
        info!(
            "Terminal bridge disabled: no daemon token configured and no Telegram bot token available for secure auto-bootstrap"
        );
        return;
    };

    let review_profiles = build_review_profiles(config);

    let runtime = RuntimeConfig {
        ws_url,
        user_id: user_id.to_string(),
        device_id: sanitize_device_id(&device_id),
        shell,
        default_cwd,
        inbox_dir,
        max_upload_bytes,
        review_profiles,
        auth,
    };

    tokio::spawn(async move {
        match TerminalBridge::new(runtime).await {
            Ok(mut bridge) => bridge.run_forever().await,
            Err(err) => error!(error = %err, "Failed to initialize terminal bridge"),
        }
    });
}

fn default_review_profiles() -> HashMap<String, ReviewProfile> {
    let mut out = HashMap::new();
    out.insert(
        "codex".to_string(),
        ReviewProfile {
            command: "codex".to_string(),
            args: vec![
                "exec".to_string(),
                "--json".to_string(),
                "--dangerously-bypass-approvals-and-sandbox".to_string(),
            ],
            timeout_secs: REVIEW_DEFAULT_TIMEOUT_SECS,
            max_output_chars: REVIEW_MAX_OUTPUT_CHARS,
        },
    );
    out.insert(
        "claude".to_string(),
        ReviewProfile {
            command: "claude".to_string(),
            args: vec![
                "-p".to_string(),
                "--dangerously-skip-permissions".to_string(),
                "--output-format".to_string(),
                "json".to_string(),
            ],
            timeout_secs: REVIEW_DEFAULT_TIMEOUT_SECS,
            max_output_chars: REVIEW_MAX_OUTPUT_CHARS,
        },
    );
    out.insert(
        "gemini".to_string(),
        ReviewProfile {
            command: "gemini".to_string(),
            args: vec![
                "--sandbox=false".to_string(),
                "--yolo".to_string(),
                "--output-format".to_string(),
                "json".to_string(),
            ],
            timeout_secs: REVIEW_DEFAULT_TIMEOUT_SECS,
            max_output_chars: REVIEW_MAX_OUTPUT_CHARS,
        },
    );
    out.insert(
        "opencode".to_string(),
        ReviewProfile {
            command: "opencode".to_string(),
            args: vec![
                "run".to_string(),
                "--format".to_string(),
                "json".to_string(),
            ],
            timeout_secs: REVIEW_DEFAULT_TIMEOUT_SECS,
            max_output_chars: REVIEW_MAX_OUTPUT_CHARS,
        },
    );
    out
}

fn build_review_profiles(config: &AppConfig) -> HashMap<String, ReviewProfile> {
    let mut out = default_review_profiles();
    for (name, tool) in &config.cli_agents.tools {
        let normalized = name.trim().to_ascii_lowercase();
        if !SUPPORTED_TERMINAL_AGENTS.contains(&normalized.as_str()) {
            continue;
        }
        if tool.command.trim().is_empty() {
            continue;
        }
        out.insert(
            normalized,
            ReviewProfile {
                command: tool.command.trim().to_string(),
                args: tool.args.clone(),
                timeout_secs: tool.timeout_secs.unwrap_or(REVIEW_DEFAULT_TIMEOUT_SECS),
                max_output_chars: tool.max_output_chars.unwrap_or(REVIEW_MAX_OUTPUT_CHARS),
            },
        );
    }
    out
}

fn resolve_daemon_user_id(config: &AppConfig, explicit: Option<u64>) -> Option<u64> {
    if let Some(id) = explicit {
        return Some(id);
    }

    if let Some(owner_ids) = config.users.owner_ids.get("telegram") {
        for raw in owner_ids {
            if let Ok(id) = raw.parse::<u64>() {
                return Some(id);
            }
        }
    }

    config
        .all_telegram_bots()
        .iter()
        .find_map(|bot| bot.allowed_user_ids.first().copied())
}

fn resolve_daemon_bot_tokens(config: &AppConfig) -> Vec<String> {
    let mut out = Vec::new();
    for bot in config.all_telegram_bots() {
        let token = bot.bot_token.trim();
        if token.is_empty() {
            continue;
        }
        if !out.iter().any(|existing: &String| existing == token) {
            out.push(token.to_string());
        }
    }
    out
}

fn daemon_connect_token_mint_url(ws_url: &str) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(ws_url)?;
    match url.scheme() {
        "wss" => {
            url.set_scheme("https")
                .map_err(|_| anyhow::anyhow!("failed to convert wss to https"))?;
        }
        other => anyhow::bail!(
            "daemon token mint requires secure websocket origin (expected wss://, found {}://)",
            other
        ),
    }
    url.set_query(None);
    url.set_fragment(None);
    url.set_path("/v1/daemon/connect-token/daemon-auth");
    Ok(url.to_string())
}

fn default_device_id() -> String {
    let host = std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "local".to_string());
    format!("{}-{}", host, std::process::id())
}

fn sanitize_device_id(value: &str) -> String {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .collect::<String>();
    if normalized.is_empty() {
        "local".to_string()
    } else {
        normalized.chars().take(64).collect()
    }
}

fn normalize_agent(value: Option<&str>) -> Option<String> {
    let v = value.unwrap_or("").trim().to_ascii_lowercase();
    if SUPPORTED_TERMINAL_AGENTS.contains(&v.as_str()) {
        Some(v)
    } else {
        None
    }
}

fn normalize_review_profile_args(reviewer: &str, args: &[String]) -> Vec<String> {
    // Review runs should prefer a single structured terminal payload.
    // Streaming/verbose modes create noisy output that is harder to parse and display.
    if reviewer != "claude" && reviewer != "gemini" {
        return args.to_vec();
    }

    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = args[idx].trim();
        if arg == "--verbose" {
            idx += 1;
            continue;
        }
        if arg == "--output-format" {
            idx += 1;
            if idx < args.len() {
                idx += 1;
            }
            continue;
        }
        out.push(args[idx].clone());
        idx += 1;
    }
    out.push("--output-format".to_string());
    out.push("json".to_string());
    out
}

fn normalize_agent_args(value: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(raw)) = value else {
        return Vec::new();
    };
    raw.iter()
        .filter_map(|item| item.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty() && !v.contains('\0'))
        .filter(|v| is_safe_agent_bootstrap_arg(v))
        .map(|v| v.chars().take(MAX_AGENT_ARG_CHARS).collect::<String>())
        .take(MAX_AGENT_ARGS)
        .collect()
}

fn is_safe_agent_bootstrap_arg(value: &str) -> bool {
    if value.contains("$(") || value.contains('\n') || value.contains('\r') {
        return false;
    }
    !value
        .chars()
        .any(|ch| matches!(ch, ';' | '|' | '&' | '>' | '<' | '`'))
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    format!("'{}'", value.replace('\'', r"'\''"))
}

fn resolve_session_cwd(raw: Option<&str>, default_cwd: &Path) -> PathBuf {
    let raw = raw.unwrap_or("").trim();
    if raw.is_empty() || raw == "~" {
        return dirs::home_dir().unwrap_or_else(|| default_cwd.to_path_buf());
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| default_cwd.join(rest));
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        return path;
    }
    default_cwd.join(path)
}

fn resolve_terminal_inbox_dir(raw: &str, default_cwd: &Path) -> PathBuf {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default_cwd.join(".aidaemon/files/inbox");
    }
    if trimmed == "~" {
        return dirs::home_dir()
            .unwrap_or_else(|| default_cwd.to_path_buf())
            .join(".aidaemon/files/inbox");
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| default_cwd.join(rest));
    }
    let p = PathBuf::from(trimmed);
    if p.is_absolute() {
        p
    } else {
        default_cwd.join(p)
    }
}

fn build_child_env() -> Vec<(String, String)> {
    let keys = [
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "COLORTERM",
        "XDG_RUNTIME_DIR",
        "SSH_AUTH_SOCK",
        "GPG_AGENT_INFO",
        "EDITOR",
        "VISUAL",
    ];
    let mut out = Vec::new();
    for key in keys {
        if let Ok(v) = std::env::var(key) {
            if !v.is_empty() {
                out.push((key.to_string(), v));
            }
        }
    }
    // PTY sessions should always advertise terminal capabilities even when
    // aidaemon is started from a non-interactive context.
    out.push(("TERM".to_string(), "xterm-256color".to_string()));
    if !out.iter().any(|(k, _)| k == "COLORTERM") {
        out.push(("COLORTERM".to_string(), "truecolor".to_string()));
    }
    // Ensure Node.js-based tools (chalk, kleur, etc.) emit colors even when
    // their own TTY detection is overly cautious.
    out.push(("FORCE_COLOR".to_string(), "1".to_string()));
    out
}

fn split_utf8_chunks(s: &str, max_bytes: usize) -> Vec<String> {
    if s.len() <= max_bytes {
        return vec![s.to_string()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in s.chars() {
        let next_len = current.len() + ch.len_utf8();
        if next_len > max_bytes && !current.is_empty() {
            out.push(current);
            current = String::new();
        }
        current.push(ch);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn escaped_control_token_len(data: &str) -> Option<usize> {
    let mut chars = data.chars();
    if chars.next() != Some('\\') {
        return None;
    }
    match chars.next()? {
        'r' | 'n' | 't' | 'b' | 'f' => Some(2),
        'u' => {
            let mut len = 2usize;
            for _ in 0..4 {
                let ch = chars.next()?;
                if !ch.is_ascii_hexdigit() {
                    return None;
                }
                len += 1;
            }
            Some(len)
        }
        'x' => {
            let a = chars.next()?;
            let b = chars.next()?;
            if a.is_ascii_hexdigit() && b.is_ascii_hexdigit() {
                Some(4)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn count_escaped_control_tokens(data: &str) -> usize {
    let mut count = 0usize;
    let mut idx = 0usize;
    while idx < data.len() {
        let remaining = &data[idx..];
        if let Some(token_len) = escaped_control_token_len(remaining) {
            count = count.saturating_add(1);
            idx = idx.saturating_add(token_len);
            continue;
        }
        let Some(ch) = remaining.chars().next() else {
            break;
        };
        idx = idx.saturating_add(ch.len_utf8());
    }
    count
}

fn should_decode_stdin_control_escapes(data: &str) -> bool {
    let trimmed = data.trim();
    if trimmed.is_empty() {
        return false;
    }

    if let Some(token_len) = escaped_control_token_len(trimmed) {
        if token_len == trimmed.len() {
            return true;
        }
    }

    if trimmed.ends_with("\\r") || trimmed.ends_with("\\n") {
        return true;
    }

    if trimmed.contains("\\u001b") || trimmed.contains("\\x1b") {
        return true;
    }

    count_escaped_control_tokens(trimmed) >= 2
}

fn decode_stdin_control_escapes(data: &str) -> String {
    if !data.contains('\\') {
        return data.to_string();
    }

    let mut out = String::with_capacity(data.len());
    let mut chars = data.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        let Some(next) = chars.next() else {
            out.push('\\');
            break;
        };
        match next {
            'r' => out.push('\r'),
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'b' => out.push('\u{0008}'),
            'f' => out.push('\u{000c}'),
            'u' => {
                let mut lookahead = chars.clone();
                let mut code = 0u32;
                let mut digits = ['\0'; 4];
                let mut ok = true;
                for slot in &mut digits {
                    let Some(hex) = lookahead.next() else {
                        ok = false;
                        break;
                    };
                    let Some(value) = hex.to_digit(16) else {
                        ok = false;
                        break;
                    };
                    *slot = hex;
                    code = (code << 4) | value;
                }
                if ok {
                    for _ in 0..4 {
                        let _ = chars.next();
                    }
                    if let Some(decoded) = char::from_u32(code) {
                        out.push(decoded);
                    } else {
                        out.push('\\');
                        out.push('u');
                        for hex in digits {
                            out.push(hex);
                        }
                    }
                } else {
                    out.push('\\');
                    out.push('u');
                }
            }
            'x' => {
                let mut lookahead = chars.clone();
                let first = lookahead.next();
                let second = lookahead.next();
                if let (Some(a), Some(b)) = (first, second) {
                    if let (Some(hi), Some(lo)) = (a.to_digit(16), b.to_digit(16)) {
                        let _ = chars.next();
                        let _ = chars.next();
                        let byte = ((hi << 4) | lo) as u8;
                        if byte.is_ascii() {
                            out.push(byte as char);
                        } else {
                            out.push('\\');
                            out.push('x');
                            out.push(a);
                            out.push(b);
                        }
                    } else {
                        out.push('\\');
                        out.push('x');
                    }
                } else {
                    out.push('\\');
                    out.push('x');
                }
            }
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    out
}

fn normalize_stdin_for_pty(data: &str) -> String {
    if should_decode_stdin_control_escapes(data) {
        decode_stdin_control_escapes(data)
    } else {
        data.to_string()
    }
}

fn truncate_with_note(value: &str, max_chars: usize) -> String {
    let clipped: String = value.chars().take(max_chars).collect();
    if clipped.chars().count() < value.chars().count() {
        format!(
            "{}\n[... truncated to {} chars ...]",
            clipped.trim_end(),
            max_chars
        )
    } else {
        clipped
    }
}

fn extract_json_content(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;
    if let Some(result) = v.get("result").and_then(|r| r.as_str()) {
        return Some(result.to_string());
    }
    if let Some(output) = v.get("output").and_then(|o| o.as_str()) {
        return Some(output.to_string());
    }
    if let Some(content) = v.get("content").and_then(|c| c.as_str()) {
        return Some(content.to_string());
    }
    if let Some(message) = v.get("message").and_then(|m| m.as_str()) {
        return Some(message.to_string());
    }
    None
}

fn extract_jsonl_content(raw: &str) -> Option<String> {
    let mut last_content: Option<String> = None;
    for line in raw.lines().rev() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if let Some(content) = v
                .pointer("/item/content")
                .or_else(|| v.pointer("/content"))
                .or_else(|| v.pointer("/result"))
            {
                if let Some(text) = content.as_str() {
                    last_content = Some(text.to_string());
                    break;
                }
            }
        }
    }
    last_content
}

fn extract_meaningful_cli_output(raw: &str, max_chars: usize) -> String {
    if let Some(content) = extract_json_content(raw) {
        return truncate_with_note(&content, max_chars);
    }
    if let Some(content) = extract_jsonl_content(raw) {
        return truncate_with_note(&content, max_chars);
    }
    truncate_with_note(raw, max_chars)
}

fn parse_review_candidate(text: &str) -> Option<ReviewResponseCandidate> {
    serde_json::from_str::<ReviewResponseCandidate>(text)
        .ok()
        .filter(|candidate| candidate.has_review_signal())
}

fn extract_review_candidate_from_json_objects(raw: &str) -> Option<ReviewResponseCandidate> {
    let bytes = raw.as_bytes();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, b) in bytes.iter().enumerate() {
        let ch = *b as char;
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        if ch == '{' {
            if start.is_none() {
                start = Some(idx);
            }
            depth = depth.saturating_add(1);
            continue;
        }
        if ch == '}' && depth > 0 {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                if let Some(s) = start {
                    if let Some(slice) = raw.get(s..=idx) {
                        if let Some(candidate) = parse_review_candidate(slice) {
                            return Some(candidate);
                        }
                        if let Ok(value) = serde_json::from_str::<Value>(slice) {
                            for pointer in [
                                "/result",
                                "/output",
                                "/content",
                                "/message",
                                "/item/content",
                            ] {
                                if let Some(text) = value.pointer(pointer).and_then(Value::as_str) {
                                    if let Some(candidate) = parse_review_candidate(text) {
                                        return Some(candidate);
                                    }
                                }
                            }
                        }
                    }
                }
                start = None;
            }
        }
    }
    None
}

fn normalize_review_items(values: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for value in values {
        let cleaned = value.split_whitespace().collect::<Vec<_>>().join(" ");
        let trimmed = cleaned.trim();
        if trimmed.is_empty() {
            continue;
        }
        let clipped: String = trimmed.chars().take(REVIEW_LIST_ITEM_MAX_CHARS).collect();
        if seen.insert(clipped.clone()) {
            out.push(clipped);
        }
        if out.len() >= REVIEW_LIST_MAX_ITEMS {
            break;
        }
    }
    out
}

fn normalize_review_severity(value: Option<&str>) -> &'static str {
    let raw = value.unwrap_or("").trim().to_ascii_lowercase();
    match raw.as_str() {
        "critical" | "blocker" | "p0" => "critical",
        "high" | "major" | "p1" => "high",
        "medium" | "moderate" | "p2" => "medium",
        "low" | "minor" | "nit" | "p3" => "low",
        _ => "medium",
    }
}

fn normalize_review_line(value: Option<&Value>) -> Option<u64> {
    let raw = value?;
    if let Some(num) = raw.as_u64() {
        return (num > 0 && num < 1_000_000).then_some(num);
    }
    let parsed = raw.as_str()?.trim().parse::<u64>().ok()?;
    (parsed > 0 && parsed < 1_000_000).then_some(parsed)
}

fn normalize_review_findings(values: Vec<ReviewFindingCandidate>) -> Vec<Value> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for value in values {
        let issue = value
            .issue
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let issue = issue.trim();
        if issue.is_empty() {
            continue;
        }
        let issue: String = issue.chars().take(REVIEW_LIST_ITEM_MAX_CHARS).collect();
        let fix = value
            .fix
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let fix = fix.trim();
        let fix = if fix.is_empty() {
            None
        } else {
            Some(
                fix.chars()
                    .take(REVIEW_LIST_ITEM_MAX_CHARS)
                    .collect::<String>(),
            )
        };
        let file = value
            .file
            .unwrap_or_default()
            .trim()
            .chars()
            .take(300)
            .collect::<String>();
        let file = if file.is_empty() { None } else { Some(file) };
        let line = normalize_review_line(value.line.as_ref());
        let severity = normalize_review_severity(value.severity.as_deref());
        let dedupe_key = format!(
            "{}|{}|{}|{}",
            severity,
            file.clone().unwrap_or_default(),
            line.map(|v| v.to_string()).unwrap_or_default(),
            issue
        );
        if !seen.insert(dedupe_key) {
            continue;
        }
        out.push(json!({
            "severity": severity,
            "file": file,
            "line": line,
            "issue": issue,
            "fix": fix,
        }));
        if out.len() >= REVIEW_LIST_MAX_ITEMS {
            break;
        }
    }
    out
}

fn parse_review_output_payload(
    reviewer: &str,
    request_id: &str,
    scope: ReviewScope,
    diff_base: ReviewDiffBase,
    raw_output: &str,
) -> Result<Value, String> {
    let meaningful = extract_meaningful_cli_output(raw_output, REVIEW_MAX_OUTPUT_CHARS);
    let candidate = parse_review_candidate(&meaningful)
        .or_else(|| extract_review_candidate_from_json_objects(&meaningful))
        .or_else(|| extract_review_candidate_from_json_objects(raw_output));

    let Some(candidate) = candidate else {
        let excerpt = truncate_with_note(meaningful.trim(), REVIEW_SUMMARY_MAX_CHARS);
        let message = if excerpt.trim().is_empty() {
            "Reviewer returned output, but no structured JSON review payload was found.".to_string()
        } else {
            format!(
                "Reviewer returned output, but no structured JSON review payload was found.\n{}",
                excerpt
            )
        };
        return Err(message);
    };

    let requested_verdict = candidate
        .verdict
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let blocking = normalize_review_items(candidate.blocking.unwrap_or_default());
    let risks = normalize_review_items(candidate.risks.unwrap_or_default());
    let suggestions = normalize_review_items(candidate.suggestions.unwrap_or_default());
    let findings = normalize_review_findings(candidate.findings.unwrap_or_default());
    let has_non_low_findings = findings.iter().any(|finding| {
        matches!(
            finding.get("severity").and_then(Value::as_str),
            Some("critical" | "high" | "medium")
        )
    });
    let mut verdict = if requested_verdict.contains("needs") || requested_verdict.contains("change")
    {
        "needs_changes".to_string()
    } else if (requested_verdict.contains("approve") || requested_verdict.contains("pass"))
        && blocking.is_empty()
        && risks.is_empty()
        && !has_non_low_findings
    {
        "approve".to_string()
    } else if !blocking.is_empty() || !risks.is_empty() || has_non_low_findings {
        "needs_changes".to_string()
    } else {
        "approve".to_string()
    };

    let mut summary = candidate.summary.unwrap_or_default();
    if summary.trim().is_empty() {
        summary = format!(
            "Structured review parsed (findings: {}, blocking: {}, risks: {}, suggestions: {}).",
            findings.len(),
            blocking.len(),
            risks.len(),
            suggestions.len()
        );
    }
    summary = truncate_with_note(summary.trim(), REVIEW_SUMMARY_MAX_CHARS);

    if verdict == "approve" && (!blocking.is_empty() || !risks.is_empty() || has_non_low_findings) {
        let lower = meaningful.to_ascii_lowercase();
        if lower.contains("must fix")
            || lower.contains("blocking")
            || lower.contains("needs changes")
            || lower.contains("do not approve")
        {
            verdict = "needs_changes".to_string();
        }
    }

    Ok(json!({
        "kind":"review_result",
        "request_id": request_id,
        "reviewer": reviewer,
        "scope": scope.as_str(),
        "diff_base": diff_base.as_str(),
        "verdict": verdict,
        "findings": findings,
        "blocking": blocking,
        "risks": risks,
        "suggestions": suggestions,
        "summary": summary,
    }))
}

fn build_review_prompt(
    reviewer: &str,
    scope: ReviewScope,
    diff_base: ReviewDiffBase,
    context: &str,
    notes: Option<&str>,
) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are a strict software reviewer.\n");
    prompt.push_str("Analyze the provided context and return ONLY JSON with this schema:\n");
    prompt.push_str("{\"verdict\":\"approve|needs_changes\",\"summary\":\"...\",\"findings\":[{\"severity\":\"critical|high|medium|low\",\"file\":\"path/to/file\",\"line\":123,\"issue\":\"...\",\"fix\":\"...\"}],\"blocking\":[\"...\"],\"risks\":[\"...\"],\"suggestions\":[\"...\"]}\n");
    prompt.push_str("Rules:\n");
    prompt.push_str("- Prefer concrete findings over generic advice.\n");
    prompt.push_str("- If there is any meaningful risk, use verdict \"needs_changes\".\n");
    prompt.push_str("- Put actionable issues in findings with file + line when possible.\n");
    prompt.push_str("- Use concise, specific issue and fix text (no fluff).\n");
    prompt.push_str("- Keep each list item concise and actionable.\n");
    prompt.push_str("- No markdown, no extra text, JSON only.\n\n");
    prompt.push_str(&format!(
        "Reviewer: {}\nScope: {}\nDiff base: {}\n\n",
        reviewer,
        scope.as_str(),
        diff_base.as_str()
    ));
    if let Some(n) = notes {
        let cleaned = truncate_with_note(n.trim(), REVIEW_MAX_SECTION_CHARS);
        if !cleaned.trim().is_empty() {
            prompt.push_str("User notes:\n");
            prompt.push_str(&cleaned);
            prompt.push_str("\n\n");
        }
    }
    prompt.push_str("Review context:\n");
    prompt.push_str(context);
    truncate_with_note(&prompt, REVIEW_MAX_CONTEXT_CHARS)
}

async fn run_capture_command(
    command: &str,
    args: &[String],
    cwd: &Path,
    timeout: Duration,
    clear_env: bool,
) -> anyhow::Result<std::process::Output> {
    let mut cmd = Command::new(command);
    for arg in args {
        cmd.arg(arg);
    }
    cmd.current_dir(cwd);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.kill_on_drop(true);
    if clear_env {
        cmd.env_clear();
        for (k, v) in build_child_env() {
            cmd.env(k, v);
        }
    }
    cmd.env_remove("CLAUDECODE");
    cmd.env_remove("CLAUDE_CODE");
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(err)) => Err(anyhow::anyhow!("{}", err)),
        Err(_) => Err(anyhow::anyhow!("command timed out after {:?}", timeout)),
    }
}

fn join_process_output(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if stdout.is_empty() {
        stderr
    } else if stderr.is_empty() {
        stdout
    } else {
        format!("{}\n{}", stdout, stderr)
    }
}

fn render_process_output(output: &std::process::Output, max_chars: usize) -> String {
    truncate_with_note(&join_process_output(output), max_chars)
}

fn normalize_review_request_id(value: Option<&str>) -> Option<String> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }
    let cleaned = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .collect::<String>();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.chars().take(64).collect())
    }
}

fn next_review_request_id() -> String {
    let short = uuid::Uuid::new_v4()
        .to_string()
        .chars()
        .filter(|c| *c != '-')
        .take(12)
        .collect::<String>();
    format!("rvw-{}", short)
}

fn collect_replay_tail(active: &ActiveSession, max_chars: usize) -> String {
    if max_chars == 0 || active.replay.is_empty() {
        return String::new();
    }
    let mut chunks = Vec::new();
    let mut used = 0usize;
    for frame in active.replay.iter().rev() {
        if used >= max_chars {
            break;
        }
        let remaining = max_chars.saturating_sub(used);
        if remaining == 0 {
            break;
        }
        let chunk = truncate_with_note(&frame.data, remaining);
        used = used.saturating_add(chunk.chars().count());
        chunks.push(chunk);
    }
    chunks.reverse();
    truncate_with_note(&chunks.join(""), max_chars)
}

fn send_review_progress_event(
    tx: &tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: &str,
    request_id: &str,
    stage: &str,
    message: &str,
) {
    let _ = tx.send(ShellEvent::ReviewProgress {
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        stage: stage.to_string(),
        message: message.to_string(),
    });
}

fn send_review_error_event(
    tx: &tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: &str,
    request_id: &str,
    code: &str,
    message: &str,
) {
    let _ = tx.send(ShellEvent::ReviewError {
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        code: code.to_string(),
        message: message.to_string(),
    });
}

fn send_review_result_event(
    tx: &tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: &str,
    request_id: &str,
    payload: Value,
) {
    let _ = tx.send(ShellEvent::ReviewResult {
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        payload,
    });
}

fn send_review_stream_event(
    tx: &tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: &str,
    request_id: &str,
    stream: &str,
    data: &str,
) {
    if data.is_empty() {
        return;
    }
    let _ = tx.send(ShellEvent::ReviewStream {
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        stream: stream.to_string(),
        data: data.to_string(),
    });
}

async fn read_reviewer_stream<R>(
    mut reader: R,
    tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: String,
    request_id: String,
    stream: &'static str,
) -> anyhow::Result<String>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    let mut buf = vec![0u8; REVIEW_STREAM_CHUNK_MAX_BYTES];
    let mut combined = String::new();
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
        if chunk.is_empty() {
            continue;
        }
        combined.push_str(&chunk);
        for part in split_utf8_chunks(&chunk, REVIEW_STREAM_CHUNK_MAX_BYTES) {
            send_review_stream_event(&tx, &session_id, &request_id, stream, &part);
        }
    }
    Ok(combined)
}

#[allow(clippy::too_many_arguments)]
async fn run_streaming_capture_command(
    tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: String,
    request_id: String,
    command: &str,
    args: &[String],
    cwd: &Path,
    timeout: Duration,
    clear_env: bool,
) -> anyhow::Result<(std::process::ExitStatus, String)> {
    let mut cmd = Command::new(command);
    for arg in args {
        cmd.arg(arg);
    }
    cmd.current_dir(cwd);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.kill_on_drop(true);
    if clear_env {
        cmd.env_clear();
        for (k, v) in build_child_env() {
            cmd.env(k, v);
        }
    }
    cmd.env_remove("CLAUDECODE");
    cmd.env_remove("CLAUDE_CODE");

    let mut child = cmd.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture reviewer stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture reviewer stderr"))?;

    let out_task = tokio::spawn(read_reviewer_stream(
        stdout,
        tx.clone(),
        session_id.clone(),
        request_id.clone(),
        "stdout",
    ));
    let err_task = tokio::spawn(read_reviewer_stream(
        stderr, tx, session_id, request_id, "stderr",
    ));

    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(err)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            let _ = out_task.await;
            let _ = err_task.await;
            return Err(anyhow::anyhow!("{}", err));
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            let _ = out_task.await;
            let _ = err_task.await;
            return Err(anyhow::anyhow!("command timed out after {:?}", timeout));
        }
    };

    let stdout_combined = out_task
        .await
        .map_err(|err| anyhow::anyhow!("stdout task failed: {}", err))??;
    let stderr_combined = err_task
        .await
        .map_err(|err| anyhow::anyhow!("stderr task failed: {}", err))??;
    let combined = if stdout_combined.is_empty() {
        stderr_combined
    } else if stderr_combined.is_empty() {
        stdout_combined
    } else {
        format!("{}\n{}", stdout_combined, stderr_combined)
    };
    Ok((status, combined))
}

fn review_diff_base_args(diff_base: ReviewDiffBase) -> Vec<String> {
    let mut args = vec![
        "diff".to_string(),
        "--no-color".to_string(),
        "--no-ext-diff".to_string(),
    ];
    if matches!(diff_base, ReviewDiffBase::Staged) {
        args.push("--staged".to_string());
    }
    args
}

fn is_probably_generated_or_non_source(path: &str) -> Option<&'static str> {
    let lower = path.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return Some("empty-path");
    }
    let roots = [
        "node_modules/",
        ".next/",
        "dist/",
        "build/",
        "coverage/",
        "vendor/",
        "target/",
        ".turbo/",
    ];
    if roots.iter().any(|prefix| lower.starts_with(prefix)) {
        return Some("generated-root");
    }
    let generated_suffixes = [
        ".min.js", ".min.css", ".map", ".lock", ".lockb", ".woff", ".woff2", ".ttf", ".eot",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".mp3", ".mp4",
        ".mov", ".avi", ".sqlite", ".db",
    ];
    if generated_suffixes
        .iter()
        .any(|suffix| lower.ends_with(suffix))
    {
        return Some("generated-suffix");
    }
    let generated_names = [
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lockb",
        "cargo.lock",
        "tsconfig.tsbuildinfo",
    ];
    if generated_names.iter().any(|name| lower.ends_with(name)) {
        return Some("lock-or-build-artifact");
    }
    None
}

fn parse_review_numstat(raw: &str) -> Vec<ReviewFileChange> {
    let mut out = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(3, '\t');
        let added_raw = parts.next().unwrap_or_default();
        let deleted_raw = parts.next().unwrap_or_default();
        let path_raw = parts.next().unwrap_or_default().trim();
        if path_raw.is_empty() {
            continue;
        }
        let binary = added_raw == "-" || deleted_raw == "-";
        let added = added_raw.parse::<u64>().ok();
        let deleted = deleted_raw.parse::<u64>().ok();
        out.push(ReviewFileChange {
            path: path_raw.to_string(),
            added,
            deleted,
            binary,
        });
    }
    out
}

fn select_review_files(
    changes: &[ReviewFileChange],
) -> (Vec<ReviewFileChange>, ReviewContextStats) {
    let mut included = Vec::new();
    let mut stats = ReviewContextStats {
        total_changed_files: changes.len(),
        ..ReviewContextStats::default()
    };
    for change in changes {
        if change.binary {
            stats.skipped_binary_files = stats.skipped_binary_files.saturating_add(1);
            continue;
        }
        if is_probably_generated_or_non_source(&change.path).is_some() {
            stats.skipped_generated_files = stats.skipped_generated_files.saturating_add(1);
            continue;
        }
        let changed_lines = change
            .added
            .unwrap_or(0)
            .saturating_add(change.deleted.unwrap_or(0));
        if changed_lines > REVIEW_MAX_FILE_CHANGED_LINES {
            stats.skipped_large_files = stats.skipped_large_files.saturating_add(1);
            continue;
        }
        if included.len() >= REVIEW_MAX_INCLUDED_DIFF_FILES {
            stats.skipped_overflow_files = stats.skipped_overflow_files.saturating_add(1);
            continue;
        }
        included.push(change.clone());
    }
    stats.included_files = included.len();
    (included, stats)
}

async fn collect_review_context(
    cwd: &Path,
    scope: ReviewScope,
    diff_base: ReviewDiffBase,
    replay_tail: &str,
) -> (String, ReviewContextStats) {
    let mut sections = Vec::new();
    let mut context_stats = ReviewContextStats::default();
    sections.push(format!("Working directory: {}", cwd.display()));
    sections.push(format!("Generated at: {}", chrono::Utc::now().to_rfc3339()));

    if matches!(scope, ReviewScope::Diff | ReviewScope::Both) {
        let mut diff_section = String::new();
        let git_probe = run_capture_command(
            "git",
            &["rev-parse".to_string(), "--is-inside-work-tree".to_string()],
            cwd,
            Duration::from_secs(8),
            true,
        )
        .await;
        let is_repo = git_probe.map(|out| out.status.success()).unwrap_or(false);

        if !is_repo {
            diff_section.push_str("Git repository not detected in working directory.\n");
        } else {
            let status_out = run_capture_command(
                "git",
                &[
                    "status".to_string(),
                    "--short".to_string(),
                    "--branch".to_string(),
                ],
                cwd,
                Duration::from_secs(12),
                true,
            )
            .await
            .ok()
            .map(|out| render_process_output(&out, REVIEW_MAX_SECTION_CHARS / 3))
            .unwrap_or_else(|| "Unable to read git status.".to_string());

            diff_section.push_str("Git status:\n");
            diff_section.push_str(&truncate_with_note(
                &status_out,
                REVIEW_MAX_SECTION_CHARS / 3,
            ));

            let mut numstat_args = review_diff_base_args(diff_base);
            numstat_args.push("--numstat".to_string());
            let numstat_out =
                run_capture_command("git", &numstat_args, cwd, Duration::from_secs(15), true)
                    .await
                    .ok()
                    .map(|out| join_process_output(&out))
                    .unwrap_or_default();
            let changes = parse_review_numstat(&numstat_out);
            let (selected_files, mut stats) = select_review_files(&changes);

            diff_section.push_str("\n\nDiff selection summary:\n");
            diff_section.push_str(&format!(
                "Changed files: {} | Included for focused diff: {} | Skipped (generated: {}, binary: {}, large: {}, overflow: {})\n",
                stats.total_changed_files,
                stats.included_files,
                stats.skipped_generated_files,
                stats.skipped_binary_files,
                stats.skipped_large_files,
                stats.skipped_overflow_files
            ));

            if !selected_files.is_empty() {
                diff_section.push_str("\nIncluded files:\n");
                for file in selected_files.iter().take(REVIEW_MAX_INCLUDED_DIFF_FILES) {
                    let added = file
                        .added
                        .map(|v| format!("+{}", v))
                        .unwrap_or_else(|| "+?".to_string());
                    let deleted = file
                        .deleted
                        .map(|v| format!("-{}", v))
                        .unwrap_or_else(|| "-?".to_string());
                    diff_section.push_str(&format!("- {} ({}, {})\n", file.path, added, deleted));
                }
            }

            let mut focused_chunks = Vec::new();
            let mut used_chars = 0usize;
            for file in &selected_files {
                let mut file_args = review_diff_base_args(diff_base);
                file_args.push("--".to_string());
                file_args.push(file.path.clone());
                let raw_file_diff =
                    run_capture_command("git", &file_args, cwd, Duration::from_secs(10), true)
                        .await
                        .ok()
                        .map(|out| join_process_output(&out))
                        .unwrap_or_default();
                if raw_file_diff.trim().is_empty() {
                    continue;
                }
                let section = format!(
                    "### FILE: {}\n{}\n",
                    file.path,
                    truncate_with_note(&raw_file_diff, REVIEW_MAX_FILE_DIFF_CHARS)
                );
                let section_chars = section.chars().count();
                if used_chars.saturating_add(section_chars) > REVIEW_MAX_SECTION_CHARS {
                    stats.skipped_overflow_files = stats.skipped_overflow_files.saturating_add(1);
                    continue;
                }
                used_chars = used_chars.saturating_add(section_chars);
                focused_chunks.push(section);
            }

            if focused_chunks.is_empty() {
                let diff_out = run_capture_command(
                    "git",
                    &review_diff_base_args(diff_base),
                    cwd,
                    Duration::from_secs(20),
                    true,
                )
                .await
                .ok()
                .map(|out| render_process_output(&out, REVIEW_MAX_SECTION_CHARS))
                .unwrap_or_else(|| "Unable to read git diff.".to_string());
                diff_section.push_str("\n\nGit diff fallback:\n");
                if diff_out.trim().is_empty() {
                    diff_section.push_str("[No diff output]\n");
                } else {
                    diff_section.push_str(&truncate_with_note(&diff_out, REVIEW_MAX_SECTION_CHARS));
                    diff_section.push('\n');
                }
            } else {
                diff_section.push_str("\n\nFocused git diff:\n");
                for chunk in focused_chunks {
                    diff_section.push_str(&chunk);
                    diff_section.push('\n');
                }
            }

            context_stats = stats;
        }
        sections.push(format!(
            "=== DIFF CONTEXT ({}) ===\n{}",
            diff_base.as_str(),
            truncate_with_note(&diff_section, REVIEW_MAX_SECTION_CHARS)
        ));
    }

    if matches!(scope, ReviewScope::Plan | ReviewScope::Both) {
        let plan = if replay_tail.trim().is_empty() {
            "No recent terminal output available for plan review.".to_string()
        } else {
            truncate_with_note(replay_tail, REVIEW_MAX_SECTION_CHARS)
        };
        sections.push(format!("=== PLAN CONTEXT ===\n{}", plan));
    }

    (
        truncate_with_note(&sections.join("\n\n"), REVIEW_MAX_CONTEXT_CHARS),
        context_stats,
    )
}

#[allow(clippy::too_many_arguments)]
async fn run_review_job(
    tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: String,
    request_id: String,
    reviewer: String,
    profile: ReviewProfile,
    cwd: PathBuf,
    scope: ReviewScope,
    diff_base: ReviewDiffBase,
    notes: Option<String>,
    replay_tail: String,
) {
    send_review_progress_event(
        &tx,
        &session_id,
        &request_id,
        "collect_context",
        "Collecting review context...",
    );

    let (context, context_stats) =
        collect_review_context(&cwd, scope, diff_base, &replay_tail).await;
    let prompt = build_review_prompt(&reviewer, scope, diff_base, &context, notes.as_deref());

    send_review_progress_event(
        &tx,
        &session_id,
        &request_id,
        "context_ready",
        &format!(
            "Context ready (changed files: {}, included: {}, skipped generated/binary/large/overflow: {}/{}/{}/{}).",
            context_stats.total_changed_files,
            context_stats.included_files,
            context_stats.skipped_generated_files,
            context_stats.skipped_binary_files,
            context_stats.skipped_large_files,
            context_stats.skipped_overflow_files
        ),
    );

    send_review_progress_event(
        &tx,
        &session_id,
        &request_id,
        "run_reviewer",
        &format!("Running {} reviewer...", reviewer),
    );

    let mut args = normalize_review_profile_args(&reviewer, &profile.args);
    args.push(prompt);
    let timeout = Duration::from_secs(profile.timeout_secs.clamp(30, 30 * 60));
    let (status, raw_combined) = match run_streaming_capture_command(
        tx.clone(),
        session_id.clone(),
        request_id.clone(),
        &profile.command,
        &args,
        &cwd,
        timeout,
        true,
    )
    .await
    {
        Ok(out) => out,
        Err(err) => {
            send_review_error_event(
                &tx,
                &session_id,
                &request_id,
                "reviewer_exec_failed",
                &format!("Failed to run {} review: {}", reviewer, err),
            );
            return;
        }
    };

    let rendered = truncate_with_note(&raw_combined, profile.max_output_chars);
    if !status.success() {
        let code = status
            .code()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "terminated".to_string());
        send_review_error_event(
            &tx,
            &session_id,
            &request_id,
            "reviewer_exit_nonzero",
            &format!(
                "{} reviewer exited with status {}.\n{}",
                reviewer, code, rendered
            ),
        );
        return;
    }

    if raw_combined.trim().is_empty() {
        send_review_error_event(
            &tx,
            &session_id,
            &request_id,
            "reviewer_empty_output",
            &format!("{} reviewer returned no output.", reviewer),
        );
        return;
    }

    send_review_progress_event(
        &tx,
        &session_id,
        &request_id,
        "parse_result",
        "Parsing review result...",
    );
    match parse_review_output_payload(&reviewer, &request_id, scope, diff_base, &raw_combined) {
        Ok(mut payload) => {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert(
                    "meta".to_string(),
                    json!({
                        "raw_output_chars": raw_combined.chars().count(),
                        "context_chars": context.chars().count(),
                        "context": {
                            "total_changed_files": context_stats.total_changed_files,
                            "included_files": context_stats.included_files,
                            "skipped_generated_files": context_stats.skipped_generated_files,
                            "skipped_binary_files": context_stats.skipped_binary_files,
                            "skipped_large_files": context_stats.skipped_large_files,
                            "skipped_overflow_files": context_stats.skipped_overflow_files,
                        }
                    }),
                );
            }
            send_review_result_event(&tx, &session_id, &request_id, payload);
        }
        Err(message) => {
            send_review_error_event(
                &tx,
                &session_id,
                &request_id,
                "review_parse_failed",
                &format!(
                    "{} reviewer returned unparseable output.\n{}\nContext stats: changed={}, included={}, skipped generated/binary/large/overflow={}/{}/{}/{}",
                    reviewer,
                    message,
                    context_stats.total_changed_files,
                    context_stats.included_files,
                    context_stats.skipped_generated_files,
                    context_stats.skipped_binary_files,
                    context_stats.skipped_large_files,
                    context_stats.skipped_overflow_files
                ),
            );
        }
    }
}

fn normalize_image_mime(value: Option<&str>) -> Option<&'static str> {
    let raw = value?.trim().to_ascii_lowercase();
    let normalized = raw.split(';').next().unwrap_or("").trim();
    match normalized {
        "image/png" => Some("image/png"),
        "image/jpeg" | "image/jpg" => Some("image/jpeg"),
        "image/webp" => Some("image/webp"),
        _ => None,
    }
}

fn normalize_upload_id(value: Option<&str>) -> Option<String> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }
    let cleaned = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .collect::<String>();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.chars().take(64).collect())
    }
}

fn sanitize_upload_filename(name: &str, mime_type: &str) -> String {
    let mut out = name
        .chars()
        .filter(|c| *c != '/' && *c != '\\' && *c != '\0')
        .collect::<String>();
    out = out
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ' '))
        .collect::<String>()
        .trim()
        .to_string();

    if out.is_empty() {
        out = "image".to_string();
    }

    out = out.chars().take(100).collect();
    if Path::new(&out).extension().is_none() {
        let ext = match mime_type {
            "image/png" => "png",
            "image/webp" => "webp",
            _ => "jpg",
        };
        out.push('.');
        out.push_str(ext);
    }
    out
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    }
}

fn build_upload_prompt(
    filename: &str,
    path: &Path,
    mime_type: &str,
    size_bytes: usize,
    caption: Option<&str>,
) -> String {
    let mut prompt = format!(
        "I uploaded an image from my phone.\nFile: {}\nSaved path: {}\nMIME: {}\nSize: {}\nPlease analyze this screenshot and help me fix the issue shown.",
        filename,
        path.display(),
        mime_type,
        format_size(size_bytes)
    );
    if let Some(caption) = caption {
        let trimmed = caption.trim();
        if !trimmed.is_empty() {
            let clipped: String = trimmed.chars().take(2000).collect();
            prompt.push_str("\nContext from me: ");
            prompt.push_str(&clipped);
        }
    }
    prompt.push('\n');
    prompt
}

fn payload_upload_id(payload: &Value) -> Option<String> {
    normalize_upload_id(
        payload
            .get("upload_id")
            .and_then(|v| v.as_str())
            .or_else(|| payload.get("id").and_then(|v| v.as_str())),
    )
}

fn b64_encode(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn b64url_encode(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn b64_decode(input: &str) -> anyhow::Result<Vec<u8>> {
    Ok(base64::engine::general_purpose::STANDARD.decode(input)?)
}

fn b64_decode_flexible(input: &str) -> anyhow::Result<Vec<u8>> {
    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(input) {
        return Ok(bytes);
    }
    if let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(input) {
        return Ok(bytes);
    }
    Ok(base64::engine::general_purpose::URL_SAFE.decode(input)?)
}

fn random_nonce_hex(num_bytes: usize) -> String {
    let mut bytes = vec![0u8; num_bytes.max(1)];
    OsRng.fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn daemon_bootstrap_signing_input(user_id: &str, device_id: &str, ts: i64, nonce: &str) -> String {
    format!(
        "v1\nuser_id={}\ndevice_id={}\nts={}\nnonce={}",
        user_id, device_id, ts, nonce
    )
}

fn derive_daemon_bootstrap_signing_key(bot_token: &str) -> anyhow::Result<[u8; 32]> {
    let hk = Hkdf::<Sha256>::new(Some(DAEMON_BOOTSTRAP_SIGNING_SALT), bot_token.as_bytes());
    let mut key = [0u8; 32];
    hk.expand(DAEMON_BOOTSTRAP_SIGNING_INFO, &mut key)
        .map_err(|_| anyhow::anyhow!("failed to derive daemon bootstrap signing key"))?;
    Ok(key)
}

fn sign_daemon_bootstrap_proof_hkdf(
    bot_token: &str,
    user_id: &str,
    device_id: &str,
    ts: i64,
    nonce: &str,
) -> anyhow::Result<String> {
    let signing_key = derive_daemon_bootstrap_signing_key(bot_token)?;
    let input = daemon_bootstrap_signing_input(user_id, device_id, ts, nonce);
    let mut mac = <HmacSha256 as Mac>::new_from_slice(&signing_key)
        .map_err(|_| anyhow::anyhow!("invalid HMAC key"))?;
    mac.update(input.as_bytes());
    let signature = mac.finalize().into_bytes();
    Ok(b64url_encode(signature.as_ref()))
}

fn sign_daemon_bootstrap_proof_legacy(
    bot_token: &str,
    user_id: &str,
    device_id: &str,
    ts: i64,
    nonce: &str,
) -> anyhow::Result<String> {
    let input = daemon_bootstrap_signing_input(user_id, device_id, ts, nonce);
    let mut mac = <HmacSha256 as Mac>::new_from_slice(bot_token.as_bytes())
        .map_err(|_| anyhow::anyhow!("invalid HMAC key"))?;
    mac.update(input.as_bytes());
    let signature = mac.finalize().into_bytes();
    Ok(b64url_encode(signature.as_ref()))
}

fn daemon_bootstrap_signature_candidates(
    bot_token: &str,
    user_id: &str,
    device_id: &str,
    ts: i64,
    nonce: &str,
) -> anyhow::Result<Vec<String>> {
    let primary = sign_daemon_bootstrap_proof_hkdf(bot_token, user_id, device_id, ts, nonce)?;
    let legacy = sign_daemon_bootstrap_proof_legacy(bot_token, user_id, device_id, ts, nonce)?;
    if primary == legacy {
        Ok(vec![primary])
    } else {
        Ok(vec![primary, legacy])
    }
}

fn parse_daemon_secret_key_from_encoded(encoded: &str) -> anyhow::Result<SecretKey> {
    let bytes = b64_decode_flexible(encoded)?;
    SecretKey::from_slice(&bytes).map_err(Into::into)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

async fn load_or_create_key_material() -> anyhow::Result<KeyMaterial> {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum KeySource {
        Keychain,
        LegacyFile,
        Generated,
    }

    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".aidaemon-terminal");
    tokio::fs::create_dir_all(&dir).await?;
    set_owner_only_permissions(&dir, 0o700)?;
    let key_path = dir.join("daemon-key.json");
    // Security model: this daemon key is local machine state (not synced over network).
    // We protect it with owner-only filesystem permissions (0700 dir, 0600 file).
    // If backups or host compromise are in scope, treat this file as sensitive.

    let mut source = KeySource::Generated;
    let secret = match resolve_from_keychain(TERMINAL_DAEMON_KEYCHAIN_FIELD) {
        Ok(encoded) => match parse_daemon_secret_key_from_encoded(&encoded) {
            Ok(secret) => {
                source = KeySource::Keychain;
                secret
            }
            Err(err) => {
                warn!(
                    error = %err,
                    "Terminal bridge keychain entry is invalid; falling back to legacy file/new key"
                );
                if tokio::fs::try_exists(&key_path).await.unwrap_or(false) {
                    let raw = tokio::fs::read_to_string(&key_path).await?;
                    set_owner_only_permissions(&key_path, 0o600)?;
                    let parsed: StoredDaemonKey = serde_json::from_str(&raw)?;
                    let encoded = parsed
                        .private_key_b64
                        .or_else(|| parsed.private_jwk.map(|jwk| jwk.d))
                        .ok_or_else(|| {
                            anyhow::anyhow!("daemon key file missing private key material")
                        })?;
                    source = KeySource::LegacyFile;
                    parse_daemon_secret_key_from_encoded(&encoded)?
                } else {
                    SecretKey::random(&mut OsRng)
                }
            }
        },
        Err(err) => {
            debug!(
                error = %err,
                "Terminal bridge keychain unavailable or missing; using legacy file fallback"
            );
            if tokio::fs::try_exists(&key_path).await.unwrap_or(false) {
                let raw = tokio::fs::read_to_string(&key_path).await?;
                set_owner_only_permissions(&key_path, 0o600)?;
                let parsed: StoredDaemonKey = serde_json::from_str(&raw)?;
                let encoded = parsed
                    .private_key_b64
                    .or_else(|| parsed.private_jwk.map(|jwk| jwk.d))
                    .ok_or_else(|| {
                        anyhow::anyhow!("daemon key file missing private key material")
                    })?;
                source = KeySource::LegacyFile;
                parse_daemon_secret_key_from_encoded(&encoded)?
            } else {
                SecretKey::random(&mut OsRng)
            }
        }
    };

    let encoded_secret = b64_encode(secret.to_bytes().as_ref());
    let keychain_ok = match store_in_keychain(TERMINAL_DAEMON_KEYCHAIN_FIELD, &encoded_secret) {
        Ok(()) => true,
        Err(err) => {
            warn!(
                error = %err,
                "Failed to persist terminal bridge key to OS keychain; continuing with filesystem fallback"
            );
            false
        }
    };

    if keychain_ok {
        if source == KeySource::LegacyFile
            && tokio::fs::try_exists(&key_path).await.unwrap_or(false)
        {
            if let Err(err) = tokio::fs::remove_file(&key_path).await {
                warn!(
                    error = %err,
                    path = %key_path.display(),
                    "Failed to remove legacy terminal bridge key file after keychain migration"
                );
            } else {
                info!("Migrated terminal bridge key from legacy file to OS keychain");
            }
        }
    } else if source == KeySource::Generated {
        let stored = StoredDaemonKeyOnDisk {
            private_key_b64: encoded_secret,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        let serialized = serde_json::to_string_pretty(&stored)?;
        tokio::fs::write(&key_path, serialized).await?;
        set_owner_only_permissions(&key_path, 0o600)?;
    }

    let public = secret.public_key();
    let public_raw = public.to_encoded_point(false).as_bytes().to_vec();
    let fingerprint = sha256_hex(&public_raw);
    Ok(KeyMaterial {
        private_key: secret,
        public_raw,
        fingerprint,
    })
}

async fn resolve_connect_token(
    auth: BridgeAuth,
    user_id: String,
    device_id: String,
    http_client: reqwest::Client,
) -> anyhow::Result<String> {
    match auth {
        BridgeAuth::StaticToken(token) => Ok(token),
        BridgeAuth::BotProof {
            mint_url,
            bot_tokens,
            fallback_static_token,
        } => {
            match mint_connect_token_from_bot_proof(
                &http_client,
                &mint_url,
                &bot_tokens,
                &user_id,
                &device_id,
            )
            .await
            {
                Ok(token) => Ok(token),
                Err(err) => {
                    if let Some(static_token) = fallback_static_token {
                        warn!(
                            error = %err,
                            "Auto-bootstrap daemon token mint failed; falling back to static daemon token"
                        );
                        Ok(static_token)
                    } else {
                        Err(err)
                    }
                }
            }
        }
    }
}

async fn mint_connect_token_from_bot_proof(
    http_client: &reqwest::Client,
    mint_url: &str,
    bot_tokens: &[String],
    user_id: &str,
    device_id: &str,
) -> anyhow::Result<String> {
    let mut last_error: Option<anyhow::Error> = None;
    let ts = chrono::Utc::now().timestamp();
    let nonce = random_nonce_hex(16);

    for bot_token in bot_tokens {
        let signatures = match daemon_bootstrap_signature_candidates(
            bot_token, user_id, device_id, ts, &nonce,
        ) {
            Ok(v) => v,
            Err(err) => {
                last_error = Some(err);
                continue;
            }
        };

        for (sig_idx, signature) in signatures.into_iter().enumerate() {
            let body = DaemonBootstrapMintRequest {
                user_id,
                device_id,
                ts,
                nonce: nonce.clone(),
                sig: signature,
            };

            let response = match http_client.post(mint_url).json(&body).send().await {
                Ok(resp) => resp,
                Err(err) => {
                    last_error = Some(anyhow::anyhow!(
                        "failed to request daemon connect token: {}",
                        err
                    ));
                    continue;
                }
            };

            let status = response.status();
            let parsed: DaemonConnectTokenResponse = match response.json().await {
                Ok(v) => v,
                Err(err) => {
                    last_error = Some(anyhow::anyhow!(
                        "daemon token endpoint returned invalid JSON (status {}): {}",
                        status,
                        err
                    ));
                    continue;
                }
            };

            if status.is_success() && parsed.ok {
                if let Some(token) = parsed.daemon_connect_token {
                    if !token.trim().is_empty() {
                        return Ok(token);
                    }
                }
                last_error = Some(anyhow::anyhow!(
                    "daemon token endpoint response missing token (status {})",
                    status
                ));
                continue;
            }

            if status.as_u16() == 401 || status.as_u16() == 403 {
                if sig_idx == 0 {
                    debug!(
                        status = status.as_u16(),
                        error = parsed.error.as_deref().unwrap_or("unknown"),
                        "daemon bootstrap primary signature rejected; trying legacy compatibility signature"
                    );
                } else {
                    debug!(
                        status = status.as_u16(),
                        error = parsed.error.as_deref().unwrap_or("unknown"),
                        "daemon bootstrap rejected for one configured telegram bot token"
                    );
                }
                last_error = Some(anyhow::anyhow!(
                    "daemon bootstrap rejected: {}",
                    parsed
                        .message
                        .as_deref()
                        .or(parsed.error.as_deref())
                        .unwrap_or("unauthorized")
                ));
                continue;
            }

            return Err(anyhow::anyhow!(
                "daemon token mint failed (status {}): {}",
                status,
                parsed
                    .message
                    .as_deref()
                    .or(parsed.error.as_deref())
                    .unwrap_or("unknown error")
            ));
        }
    }

    Err(last_error.unwrap_or_else(|| {
        anyhow::anyhow!("no Telegram bot token available for daemon bootstrap auth")
    }))
}

struct TerminalBridge {
    cfg: RuntimeConfig,
    key_material: KeyMaterial,
    sessions: HashMap<String, ActiveSession>,
    http_client: reqwest::Client,
    shell_events_tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    shell_events_rx: tokio::sync::mpsc::UnboundedReceiver<ShellEvent>,
}

impl TerminalBridge {
    async fn new(cfg: RuntimeConfig) -> anyhow::Result<Self> {
        let key_material = load_or_create_key_material().await?;
        let http_client = reqwest::Client::builder()
            .user_agent("aidaemon-terminal-bridge/1.0")
            .timeout(Duration::from_secs(12))
            .build()?;
        let (shell_events_tx, shell_events_rx) =
            tokio::sync::mpsc::unbounded_channel::<ShellEvent>();
        info!(
            fingerprint = %key_material.fingerprint,
            user_id = %cfg.user_id,
            "Terminal bridge initialized"
        );
        Ok(Self {
            cfg,
            key_material,
            sessions: HashMap::new(),
            http_client,
            shell_events_tx,
            shell_events_rx,
        })
    }

    async fn run_forever(&mut self) {
        loop {
            self.drain_shell_events(512);
            if let Err(err) = self.connect_once().await {
                error!(error = %err, "Terminal bridge connection failed");
            }
            self.drain_shell_events(2048);
            tokio::time::sleep(Duration::from_millis(RECONNECT_MS)).await;
        }
    }

    async fn connect_once(&mut self) -> anyhow::Result<()> {
        let connect_token = {
            let auth = self.cfg.auth.clone();
            let user_id = self.cfg.user_id.clone();
            let device_id = self.cfg.device_id.clone();
            let http_client = self.http_client.clone();
            resolve_connect_token(auth, user_id, device_id, http_client).await?
        };
        let mut ws_url = reqwest::Url::parse(&self.cfg.ws_url)?;
        ws_url
            .query_pairs_mut()
            .append_pair("user_id", &self.cfg.user_id)
            .append_pair("device_id", &self.cfg.device_id);

        let mut req = ws_url.as_str().into_client_request()?;
        req.headers_mut().insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", connect_token))?,
        );
        debug!(
            has_ws_key = req.headers().contains_key("sec-websocket-key"),
            upgrade = ?req.headers().get("upgrade"),
            connection = ?req.headers().get("connection"),
            "Prepared daemon websocket request headers"
        );
        info!(url = %ws_url, "Connecting terminal bridge to broker");
        let (ws_stream, _) = connect_async(req).await?;
        info!("Terminal bridge connected");

        let (mut ws_write, mut ws_read) = ws_stream.split();
        let mut heartbeat = tokio::time::interval(Duration::from_millis(HEARTBEAT_MS));
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        Self::send_plain_json(
            &mut ws_write,
            json!({
                "type": "daemon_hello",
                "alg": "ECDH-P256/AES-256-GCM",
                "daemon_pub": b64_encode(&self.key_material.public_raw),
                "fingerprint": self.key_material.fingerprint,
            }),
        )
        .await?;

        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    let _ = Self::send_plain_json(&mut ws_write, json!({"type":"heartbeat"})).await;
                    let mut exited = Vec::new();
                    for (session_id, active) in self.sessions.iter_mut() {
                        if let Ok(Some(code)) = active.shell.try_wait_code() {
                            exited.push((session_id.clone(), code));
                        }
                    }
                    for (session_id, code) in exited {
                        let _ = self
                            .send_encrypted_json_for_session(
                                &session_id,
                                &mut ws_write,
                                json!({"kind":"exit","code":code}),
                            )
                            .await;
                        self.remove_session(&session_id, "shell exited").await;
                    }
                }
                maybe_event = self.shell_events_rx.recv() => {
                    let Some(event) = maybe_event else {
                        continue;
                    };
                    if let Err(err) = self
                        .handle_shell_event(event, Some(&mut ws_write))
                        .await
                    {
                        warn!(error = %err, "Failed to relay shell output event");
                    }
                }
                maybe_msg = ws_read.next() => {
                    match maybe_msg {
                        Some(Ok(msg)) => {
                            if let Err(err) = self.handle_ws_message(msg, &mut ws_write).await {
                                warn!(error=%err, "Terminal bridge message handling error");
                            }
                        }
                        Some(Err(err)) => return Err(anyhow::anyhow!(err)),
                        None => return Ok(()),
                    }
                }
            }
        }
    }

    fn derive_relay_cipher(
        daemon_private_key: &SecretKey,
        relay_session_id: &str,
        client_pub_b64: &str,
    ) -> anyhow::Result<Aes256Gcm> {
        let client_pub_raw = b64_decode(client_pub_b64)?;
        let client_pub = PublicKey::from_sec1_bytes(&client_pub_raw)?;
        let shared = diffie_hellman(
            daemon_private_key.to_nonzero_scalar(),
            client_pub.as_affine(),
        );

        let mut salt_hash = Sha256::new();
        salt_hash.update(relay_session_id.as_bytes());
        let salt = salt_hash.finalize();
        let hk = Hkdf::<Sha256>::new(Some(&salt), shared.raw_secret_bytes().as_slice());
        let mut key = [0u8; 32];
        hk.expand(KEY_INFO, &mut key)
            .map_err(|_| anyhow::anyhow!("HKDF key expansion failed"))?;
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| anyhow::anyhow!("failed to build AES-256-GCM cipher"))?;
        Ok(cipher)
    }

    fn record_stdout_chunks(&mut self, session_id: &str, data: &str) -> Vec<(u64, String)> {
        let Some(active) = self.sessions.get_mut(session_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for chunk in split_utf8_chunks(data, MAX_ENCRYPTED_PAYLOAD_BYTES) {
            let seq = active.next_stdout_seq;
            active.next_stdout_seq = active.next_stdout_seq.saturating_add(1);
            active.replay_bytes = active.replay_bytes.saturating_add(chunk.len());
            active.replay.push_back(ReplayFrame {
                seq,
                data: chunk.clone(),
            });

            while active.replay.len() > REPLAY_MAX_FRAMES || active.replay_bytes > REPLAY_MAX_BYTES
            {
                if let Some(old) = active.replay.pop_front() {
                    active.replay_bytes = active.replay_bytes.saturating_sub(old.data.len());
                } else {
                    break;
                }
            }
            out.push((seq, chunk));
        }
        out
    }

    fn record_review_stream_chunks(
        &mut self,
        session_id: &str,
        request_id: &str,
        stream: &str,
        data: &str,
    ) -> Vec<(u64, String, String, String)> {
        let Some(active) = self.sessions.get_mut(session_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for chunk in split_utf8_chunks(data, REVIEW_STREAM_CHUNK_MAX_BYTES) {
            let seq = active.next_review_stream_seq;
            active.next_review_stream_seq = active.next_review_stream_seq.saturating_add(1);
            active.review_stream_replay_bytes = active
                .review_stream_replay_bytes
                .saturating_add(chunk.len());
            active.review_stream_replay.push_back(ReviewStreamFrame {
                seq,
                request_id: request_id.to_string(),
                stream: stream.to_string(),
                data: chunk.clone(),
            });

            while active.review_stream_replay.len() > REVIEW_STREAM_REPLAY_MAX_FRAMES
                || active.review_stream_replay_bytes > REVIEW_STREAM_REPLAY_MAX_BYTES
            {
                if let Some(old) = active.review_stream_replay.pop_front() {
                    active.review_stream_replay_bytes = active
                        .review_stream_replay_bytes
                        .saturating_sub(old.data.len());
                } else {
                    break;
                }
            }

            out.push((seq, request_id.to_string(), stream.to_string(), chunk));
        }
        out
    }

    fn record_review_progress_payload(
        &mut self,
        session_id: &str,
        request_id: &str,
        stage: &str,
        message: &str,
    ) -> Value {
        let payload = json!({
            "kind":"review_progress",
            "request_id": request_id,
            "stage": stage,
            "message": truncate_with_note(message, 600),
        });
        if let Some(active) = self.sessions.get_mut(session_id) {
            active.last_review_progress = Some(payload.clone());
        }
        payload
    }

    fn record_review_error_payload(
        &mut self,
        session_id: &str,
        request_id: &str,
        code: &str,
        message: &str,
    ) -> Value {
        let payload = json!({
            "kind":"review_error",
            "request_id": request_id,
            "code": code,
            "message": truncate_with_note(message, 1000),
        });
        if let Some(active) = self.sessions.get_mut(session_id) {
            active.last_review_progress = Some(payload.clone());
            if active
                .review_job
                .as_ref()
                .map(|job| job.request_id == request_id)
                .unwrap_or(false)
            {
                active.review_job = None;
            }
        }
        payload
    }

    fn record_review_result_payload(
        &mut self,
        session_id: &str,
        request_id: &str,
        payload: Value,
    ) -> Value {
        let mut normalized = payload;
        if normalized.get("kind").and_then(|v| v.as_str()) != Some("review_result") {
            normalized["kind"] = Value::String("review_result".to_string());
        }
        normalized["request_id"] = Value::String(request_id.to_string());
        if let Some(active) = self.sessions.get_mut(session_id) {
            active.last_review_result = Some(normalized.clone());
            active.last_review_progress = None;
            if active
                .review_job
                .as_ref()
                .map(|job| job.request_id == request_id)
                .unwrap_or(false)
            {
                active.review_job = None;
            }
        }
        normalized
    }

    fn drain_shell_events(&mut self, max_events: usize) {
        let mut drained = 0usize;
        loop {
            if drained >= max_events {
                break;
            }
            match self.shell_events_rx.try_recv() {
                Ok(ShellEvent::Output { session_id, data }) => {
                    let _ = self.record_stdout_chunks(&session_id, &data);
                    drained += 1;
                }
                Ok(ShellEvent::ReviewProgress {
                    session_id,
                    request_id,
                    stage,
                    message,
                }) => {
                    let _ = self.record_review_progress_payload(
                        &session_id,
                        &request_id,
                        &stage,
                        &message,
                    );
                    drained += 1;
                }
                Ok(ShellEvent::ReviewResult {
                    session_id,
                    request_id,
                    payload,
                }) => {
                    let _ = self.record_review_result_payload(&session_id, &request_id, payload);
                    drained += 1;
                }
                Ok(ShellEvent::ReviewError {
                    session_id,
                    request_id,
                    code,
                    message,
                }) => {
                    let _ =
                        self.record_review_error_payload(&session_id, &request_id, &code, &message);
                    drained += 1;
                }
                Ok(ShellEvent::ReviewStream {
                    session_id,
                    request_id,
                    stream,
                    data,
                }) => {
                    let _ =
                        self.record_review_stream_chunks(&session_id, &request_id, &stream, &data);
                    drained += 1;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        if drained > 0 {
            debug!(
                drained,
                "Drained buffered PTY output while transport was unavailable"
            );
        }
    }

    async fn handle_shell_event<S>(
        &mut self,
        event: ShellEvent,
        ws_write: Option<&mut S>,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        match event {
            ShellEvent::Output { session_id, data } => {
                let frames = self.record_stdout_chunks(&session_id, &data);
                let Some(ws_write) = ws_write else {
                    return Ok(());
                };
                for (seq, chunk) in frames {
                    self.send_encrypted_json_for_session(
                        &session_id,
                        ws_write,
                        json!({"kind":"stdout","seq":seq,"data":chunk}),
                    )
                    .await?;
                }
            }
            ShellEvent::ReviewProgress {
                session_id,
                request_id,
                stage,
                message,
            } => {
                let payload =
                    self.record_review_progress_payload(&session_id, &request_id, &stage, &message);
                if let Some(ws_write) = ws_write {
                    self.send_encrypted_json_for_session(&session_id, ws_write, payload)
                        .await?;
                }
            }
            ShellEvent::ReviewResult {
                session_id,
                request_id,
                payload,
            } => {
                let payload = self.record_review_result_payload(&session_id, &request_id, payload);
                if let Some(ws_write) = ws_write {
                    self.send_encrypted_json_for_session(&session_id, ws_write, payload)
                        .await?;
                }
            }
            ShellEvent::ReviewError {
                session_id,
                request_id,
                code,
                message,
            } => {
                let payload =
                    self.record_review_error_payload(&session_id, &request_id, &code, &message);
                if let Some(ws_write) = ws_write {
                    self.send_encrypted_json_for_session(&session_id, ws_write, payload)
                        .await?;
                }
            }
            ShellEvent::ReviewStream {
                session_id,
                request_id,
                stream,
                data,
            } => {
                let frames =
                    self.record_review_stream_chunks(&session_id, &request_id, &stream, &data);
                if let Some(ws_write) = ws_write {
                    for (seq, request_id, stream, data) in frames {
                        self.send_encrypted_json_for_session(
                            &session_id,
                            ws_write,
                            json!({
                                "kind":"review_stream",
                                "seq": seq,
                                "request_id": request_id,
                                "stream": stream,
                                "data": data,
                            }),
                        )
                        .await?;
                    }
                }
            }
        }
        Ok(())
    }

    async fn replay_stdout_since<S>(
        &mut self,
        session_id: &str,
        resume_from_seq: u64,
        resume_from_review_stream_seq: u64,
        ws_write: &mut S,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let (
            oldest_seq,
            newest_seq,
            frames,
            oldest_review_stream_seq,
            newest_review_stream_seq,
            review_stream_frames,
            last_review_progress,
            last_review_result,
            running_review_request_id,
        ) = {
            let Some(active) = self.sessions.get(session_id) else {
                return Ok(());
            };
            let oldest = active.replay.front().map(|f| f.seq).unwrap_or(0);
            let newest = active.replay.back().map(|f| f.seq).unwrap_or(0);
            let frames = active
                .replay
                .iter()
                .filter(|f| f.seq > resume_from_seq)
                .map(|f| (f.seq, f.data.clone()))
                .collect::<Vec<_>>();
            let oldest_review = active
                .review_stream_replay
                .front()
                .map(|f| f.seq)
                .unwrap_or(0);
            let newest_review = active
                .review_stream_replay
                .back()
                .map(|f| f.seq)
                .unwrap_or(0);
            let review_stream_frames = active
                .review_stream_replay
                .iter()
                .filter(|f| f.seq > resume_from_review_stream_seq)
                .map(|f| {
                    (
                        f.seq,
                        f.request_id.clone(),
                        f.stream.clone(),
                        f.data.clone(),
                    )
                })
                .collect::<Vec<_>>();
            (
                oldest,
                newest,
                frames,
                oldest_review,
                newest_review,
                review_stream_frames,
                active.last_review_progress.clone(),
                active.last_review_result.clone(),
                active.review_job.as_ref().map(|job| job.request_id.clone()),
            )
        };

        if !frames.is_empty() {
            if resume_from_seq + 1 < oldest_seq {
                self.send_encrypted_json_for_session(
                    session_id,
                    ws_write,
                    json!({
                        "kind":"status",
                        "message":format!(
                            "Reconnected after output gap; replay starts at seq {}.",
                            oldest_seq
                        )
                    }),
                )
                .await?;
            }

            for (seq, chunk) in &frames {
                self.send_encrypted_json_for_session(
                    session_id,
                    ws_write,
                    json!({"kind":"stdout","seq":seq,"data":chunk}),
                )
                .await?;
            }

            self.send_encrypted_json_for_session(
                session_id,
                ws_write,
                json!({
                    "kind":"status",
                    "message": format!(
                        "Replayed {} buffered output frame(s) (seq {}..{}).",
                        frames.len(),
                        frames.first().map(|(seq, _)| *seq).unwrap_or(oldest_seq),
                        newest_seq
                    )
                }),
            )
            .await?;
        }

        if !review_stream_frames.is_empty() {
            if resume_from_review_stream_seq + 1 < oldest_review_stream_seq {
                self.send_encrypted_json_for_session(
                    session_id,
                    ws_write,
                    json!({
                        "kind":"status",
                        "message":format!(
                            "Reconnected after review output gap; replay starts at review seq {}.",
                            oldest_review_stream_seq
                        )
                    }),
                )
                .await?;
            }
            for (seq, request_id, stream, data) in &review_stream_frames {
                self.send_encrypted_json_for_session(
                    session_id,
                    ws_write,
                    json!({
                        "kind":"review_stream",
                        "seq": seq,
                        "request_id": request_id,
                        "stream": stream,
                        "data": data,
                    }),
                )
                .await?;
            }
            self.send_encrypted_json_for_session(
                session_id,
                ws_write,
                json!({
                    "kind":"status",
                    "message": format!(
                        "Replayed {} buffered review stream chunk(s) (seq {}..{}).",
                        review_stream_frames.len(),
                        review_stream_frames
                            .first()
                            .map(|(seq, _, _, _)| *seq)
                            .unwrap_or(oldest_review_stream_seq),
                        newest_review_stream_seq
                    )
                }),
            )
            .await?;
        }

        let has_last_progress = last_review_progress.is_some();
        if let Some(payload) = last_review_progress {
            self.send_encrypted_json_for_session(session_id, ws_write, payload)
                .await?;
        }
        if let Some(payload) = last_review_result {
            self.send_encrypted_json_for_session(session_id, ws_write, payload)
                .await?;
        }
        if running_review_request_id.is_some() && frames.is_empty() && !has_last_progress {
            self.send_encrypted_json_for_session(
                session_id,
                ws_write,
                json!({
                    "kind":"status",
                    "message":"Review is still running. Waiting for result..."
                }),
            )
            .await?;
        }

        Ok(())
    }

    async fn handle_ws_message<S>(&mut self, msg: Message, ws_write: &mut S) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let text = match msg {
            Message::Text(t) => t,
            Message::Binary(b) => String::from_utf8_lossy(&b).to_string(),
            Message::Ping(_) | Message::Pong(_) => return Ok(()),
            Message::Close(_) => return Ok(()),
            _ => return Ok(()),
        };
        let outer: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let event = outer.get("event").and_then(|v| v.as_str()).unwrap_or("");
        match event {
            "heartbeat" => {
                Self::send_plain_json(ws_write, json!({"type":"heartbeat"})).await?;
                return Ok(());
            }
            "status" => {
                if let Some(message) = outer.get("message").and_then(|v| v.as_str()) {
                    debug!(message, "Broker status");
                }
                return Ok(());
            }
            "relay" => {}
            _ => return Ok(()),
        }
        if outer.get("from").and_then(|v| v.as_str()) != Some("client") {
            return Ok(());
        }
        let session_id = outer
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let payload_raw = outer
            .get("payload")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let frame: Value = match serde_json::from_str(&payload_raw) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let frame_type = frame.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match frame_type {
            "heartbeat" => Ok(()),
            "e2ee_client_hello" => {
                self.handle_client_hello(&frame, &session_id, ws_write)
                    .await
            }
            "e2ee" => {
                self.handle_encrypted_frame(&frame, &session_id, ws_write)
                    .await
            }
            _ => Ok(()),
        }
    }

    async fn handle_client_hello<S>(
        &mut self,
        frame: &Value,
        relay_session_id: &str,
        ws_write: &mut S,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let client_pub_b64 = frame
            .get("client_pub")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if client_pub_b64.is_empty() || relay_session_id.is_empty() {
            Self::send_plain_json(
                ws_write,
                json!({"type":"error","message":"Invalid client hello payload."}),
            )
            .await?;
            return Ok(());
        }
        let resume_from_seq = frame
            .get("resume_from_seq")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let resume_from_review_stream_seq = frame
            .get("resume_from_review_stream_seq")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let requested_agent = normalize_agent(frame.get("agent").and_then(|v| v.as_str()));
        let requested_agent_args = normalize_agent_args(frame.get("agent_args"));
        let cipher = Self::derive_relay_cipher(
            &self.key_material.private_key,
            relay_session_id,
            &client_pub_b64,
        )?;

        if let Some(active) = self.sessions.get_mut(relay_session_id) {
            active.crypto.cipher = cipher;
            active.crypto.send_counter = 0;
            active.crypto.recv_counter = 0;
            if !active.crypto.bootstrapped_agent {
                active.crypto.agent = requested_agent.clone();
                active.crypto.agent_args = requested_agent_args.clone();
            }
            Self::send_plain_json(
                ws_write,
                json!({
                    "type":"e2ee_server_ready",
                    "session_id": relay_session_id,
                    "fingerprint": self.key_material.fingerprint,
                }),
            )
            .await?;

            self.send_encrypted_json_for_session(
                relay_session_id,
                ws_write,
                json!({"kind":"status","message":"Local daemon secure channel ready (reattached)."}),
            )
            .await?;
            self.replay_stdout_since(
                relay_session_id,
                resume_from_seq,
                resume_from_review_stream_seq,
                ws_write,
            )
            .await?;
            self.bootstrap_agent(relay_session_id, ws_write).await?;
            return Ok(());
        }

        let cwd = resolve_session_cwd(
            frame.get("cwd").and_then(|v| v.as_str()),
            &self.cfg.default_cwd,
        );

        let shell = match ShellProcess::spawn(
            &self.cfg.shell,
            &cwd,
            relay_session_id,
            self.shell_events_tx.clone(),
        )
        .await
        {
            Ok(shell) => shell,
            Err(err) => {
                Self::send_plain_json(
                    ws_write,
                    json!({"type":"error","message":format!("Failed to spawn shell: {}", err)}),
                )
                .await?;
                return Ok(());
            }
        };

        let crypto = CryptoSession {
            session_id: relay_session_id.to_string(),
            cipher,
            send_counter: 0,
            recv_counter: 0,
            agent: requested_agent,
            agent_args: requested_agent_args,
            bootstrapped_agent: false,
        };

        self.sessions.insert(
            relay_session_id.to_string(),
            ActiveSession {
                crypto,
                shell,
                cwd,
                replay: VecDeque::new(),
                replay_bytes: 0,
                next_stdout_seq: 1,
                review_stream_replay: VecDeque::new(),
                review_stream_replay_bytes: 0,
                next_review_stream_seq: 1,
                pending_uploads: HashMap::new(),
                review_job: None,
                last_review_progress: None,
                last_review_result: None,
            },
        );

        Self::send_plain_json(
            ws_write,
            json!({
                "type":"e2ee_server_ready",
                "session_id": relay_session_id,
                "fingerprint": self.key_material.fingerprint,
            }),
        )
        .await?;

        self.send_encrypted_json_for_session(
            relay_session_id,
            ws_write,
            json!({"kind":"status","message":"Local daemon secure channel ready."}),
        )
        .await?;
        self.replay_stdout_since(
            relay_session_id,
            resume_from_seq,
            resume_from_review_stream_seq,
            ws_write,
        )
        .await?;
        self.bootstrap_agent(relay_session_id, ws_write).await?;
        Ok(())
    }

    async fn bootstrap_agent<S>(&mut self, session_id: &str, ws_write: &mut S) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let Some(active) = self.sessions.get_mut(session_id) else {
            return Ok(());
        };
        let crypto = &mut active.crypto;
        let Some(agent) = crypto.agent.clone() else {
            return Ok(());
        };
        if crypto.bootstrapped_agent {
            return Ok(());
        }
        if crypto
            .agent_args
            .iter()
            .any(|arg| !is_safe_agent_bootstrap_arg(arg))
        {
            anyhow::bail!("unsafe agent argument rejected for shell bootstrap");
        }
        let mut command_parts = Vec::with_capacity(1 + crypto.agent_args.len());
        command_parts.push(shell_quote(&agent));
        command_parts.extend(crypto.agent_args.iter().map(|arg| shell_quote(arg)));
        let command = format!("{}\n", command_parts.join(" "));
        active.shell.write_stdin(&command).await?;
        crypto.bootstrapped_agent = true;
        let args_suffix = if crypto.agent_args.is_empty() {
            String::new()
        } else {
            format!(" {}", crypto.agent_args.join(" "))
        };
        self.send_encrypted_json_for_session(
            session_id,
            ws_write,
            json!({"kind":"status","message":format!("Starting {}{}...", agent, args_suffix)}),
        )
        .await?;
        Ok(())
    }

    fn prune_stale_uploads(active: &mut ActiveSession) {
        let cutoff = Duration::from_secs(UPLOAD_IDLE_TTL_SECS);
        active
            .pending_uploads
            .retain(|_, upload| upload.touched_at.elapsed() <= cutoff);
    }

    fn begin_image_upload(
        active: &mut ActiveSession,
        payload: &Value,
        max_upload_bytes: usize,
    ) -> anyhow::Result<String> {
        Self::prune_stale_uploads(active);
        if max_upload_bytes == 0 {
            anyhow::bail!("File uploads are disabled in daemon config ([files].enabled = false).");
        }

        if active.pending_uploads.len() >= MAX_PENDING_UPLOADS_PER_SESSION {
            anyhow::bail!(
                "Too many active uploads in this session. Finish or cancel an existing upload first."
            );
        }

        let upload_id = payload_upload_id(payload)
            .ok_or_else(|| anyhow::anyhow!("upload_start requires upload_id"))?;
        if active.pending_uploads.contains_key(&upload_id) {
            anyhow::bail!("Upload id `{}` is already active.", upload_id);
        }

        let mime_type = normalize_image_mime(
            payload
                .get("mime_type")
                .and_then(|v| v.as_str())
                .or_else(|| payload.get("mime").and_then(|v| v.as_str())),
        )
        .ok_or_else(|| anyhow::anyhow!("Unsupported image MIME type. Use png/jpeg/webp."))?
        .to_string();

        let expected_bytes_u64 = payload
            .get("total_bytes")
            .or_else(|| payload.get("size"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("upload_start requires total_bytes"))?;
        let expected_bytes = usize::try_from(expected_bytes_u64)
            .map_err(|_| anyhow::anyhow!("upload size is too large for this platform"))?;
        if expected_bytes == 0 {
            anyhow::bail!("Image upload cannot be empty.");
        }
        if expected_bytes > max_upload_bytes {
            anyhow::bail!(
                "Image too large ({}). Maximum is {}.",
                format_size(expected_bytes),
                format_size(max_upload_bytes)
            );
        }

        let raw_filename = payload
            .get("filename")
            .and_then(|v| v.as_str())
            .or_else(|| payload.get("name").and_then(|v| v.as_str()))
            .unwrap_or("image");
        let filename = sanitize_upload_filename(raw_filename, &mime_type);
        let caption = payload
            .get("caption")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        active.pending_uploads.insert(
            upload_id.clone(),
            PendingUpload {
                upload_id: upload_id.clone(),
                filename,
                mime_type,
                expected_bytes,
                bytes: Vec::with_capacity(expected_bytes.min(max_upload_bytes)),
                chunks_received: 0,
                caption,
                touched_at: Instant::now(),
            },
        );

        Ok(format!(
            "Image upload started (id={}, size={}).",
            upload_id,
            format_size(expected_bytes)
        ))
    }

    fn append_image_upload_chunk(
        active: &mut ActiveSession,
        payload: &Value,
        max_upload_bytes: usize,
    ) -> anyhow::Result<Option<String>> {
        let upload_id = payload_upload_id(payload)
            .ok_or_else(|| anyhow::anyhow!("upload_chunk requires upload_id"))?;
        let data_b64 = payload
            .get("data_b64")
            .or_else(|| payload.get("chunk_b64"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if data_b64.is_empty() {
            anyhow::bail!("upload_chunk requires data_b64");
        }
        let chunk = b64_decode_flexible(data_b64)?;
        if chunk.is_empty() {
            anyhow::bail!("upload chunk was empty");
        }
        if chunk.len() > MAX_UPLOAD_CHUNK_BYTES {
            anyhow::bail!(
                "Upload chunk too large ({} bytes > {} bytes).",
                chunk.len(),
                MAX_UPLOAD_CHUNK_BYTES
            );
        }

        let Some(upload) = active.pending_uploads.get_mut(&upload_id) else {
            anyhow::bail!("No active upload for id `{}`.", upload_id);
        };
        upload.touched_at = Instant::now();

        let next_len = upload.bytes.len().saturating_add(chunk.len());
        if next_len > upload.expected_bytes || next_len > max_upload_bytes {
            anyhow::bail!(
                "Upload exceeds declared size ({} > {}).",
                format_size(next_len),
                format_size(upload.expected_bytes)
            );
        }

        upload.bytes.extend_from_slice(&chunk);
        upload.chunks_received = upload.chunks_received.saturating_add(1);

        if upload.bytes.len() == upload.expected_bytes {
            return Ok(Some(format!(
                "Image upload received (id={}, size={}). Send upload_commit to finish.",
                upload.upload_id,
                format_size(upload.bytes.len())
            )));
        }

        if upload.chunks_received % 8 == 0 {
            let pct = (upload.bytes.len() * 100 / upload.expected_bytes).min(99);
            return Ok(Some(format!(
                "Image upload progress: {}% (id={}).",
                pct, upload.upload_id
            )));
        }

        Ok(None)
    }

    fn cancel_image_upload(active: &mut ActiveSession, payload: &Value) -> anyhow::Result<String> {
        let upload_id = payload_upload_id(payload)
            .ok_or_else(|| anyhow::anyhow!("upload_cancel requires upload_id"))?;
        if active.pending_uploads.remove(&upload_id).is_some() {
            Ok(format!("Upload `{}` cancelled.", upload_id))
        } else {
            anyhow::bail!("No active upload for id `{}`.", upload_id);
        }
    }

    fn commit_image_upload(
        active: &mut ActiveSession,
        payload: &Value,
        inbox_dir: &Path,
    ) -> anyhow::Result<UploadCommitResult> {
        let upload_id = payload_upload_id(payload)
            .ok_or_else(|| anyhow::anyhow!("upload_commit requires upload_id"))?;
        let Some(mut upload) = active.pending_uploads.remove(&upload_id) else {
            anyhow::bail!("No active upload for id `{}`.", upload_id);
        };
        upload.touched_at = Instant::now();

        if upload.bytes.is_empty() {
            anyhow::bail!("Upload `{}` has no data.", upload_id);
        }
        if upload.bytes.len() != upload.expected_bytes {
            anyhow::bail!(
                "Upload `{}` incomplete (received {}, expected {}).",
                upload_id,
                format_size(upload.bytes.len()),
                format_size(upload.expected_bytes)
            );
        }

        std::fs::create_dir_all(inbox_dir)?;
        let prefix: String = uuid::Uuid::new_v4().to_string().chars().take(8).collect();
        let dest_name = format!("{}_{}", prefix, upload.filename);
        let dest_path = inbox_dir.join(dest_name);
        std::fs::write(&dest_path, &upload.bytes)?;

        let final_caption = payload
            .get("caption")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or(upload.caption);

        let prompt_for_agent = if active.crypto.bootstrapped_agent || active.crypto.agent.is_some()
        {
            Some(build_upload_prompt(
                &upload.filename,
                &dest_path,
                &upload.mime_type,
                upload.bytes.len(),
                final_caption.as_deref(),
            ))
        } else {
            None
        };

        Ok(UploadCommitResult {
            status_message: format!(
                "Image saved to {} ({}).",
                dest_path.display(),
                format_size(upload.bytes.len())
            ),
            prompt_for_agent,
        })
    }

    async fn handle_encrypted_frame<S>(
        &mut self,
        frame: &Value,
        relay_session_id: &str,
        ws_write: &mut S,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let mut remove_reason: Option<&'static str> = None;
        let mut status_messages: Vec<String> = Vec::new();
        let mut immediate_payloads: Vec<Value> = Vec::new();
        let max_upload_bytes = self.cfg.max_upload_bytes;
        let inbox_dir = self.cfg.inbox_dir.clone();
        let review_profiles = self.cfg.review_profiles.clone();
        let shell_events_tx = self.shell_events_tx.clone();
        {
            let Some(active) = self.sessions.get_mut(relay_session_id) else {
                return Ok(());
            };
            let crypto = &mut active.crypto;

            let counter = frame.get("c").and_then(|v| v.as_u64()).unwrap_or(0);
            if counter <= crypto.recv_counter {
                return Ok(());
            }
            let iv = b64_decode(frame.get("iv").and_then(|v| v.as_str()).unwrap_or(""))?;
            let ct = b64_decode(frame.get("ct").and_then(|v| v.as_str()).unwrap_or(""))?;
            if iv.len() != 12 || ct.is_empty() {
                return Ok(());
            }
            if ct.len() > MAX_ENCRYPTED_PAYLOAD_BYTES.saturating_mul(2) {
                return Ok(());
            }

            let aad = format!("{}:{}", crypto.session_id, counter);
            let plaintext = match crypto.cipher.decrypt(
                Nonce::from_slice(&iv),
                Payload {
                    msg: &ct,
                    aad: aad.as_bytes(),
                },
            ) {
                Ok(v) => v,
                Err(_) => return Ok(()),
            };
            if plaintext.len() > MAX_ENCRYPTED_PAYLOAD_BYTES {
                return Ok(());
            }
            crypto.recv_counter = counter;

            let payload: Value = match serde_json::from_slice(&plaintext) {
                Ok(v) => v,
                Err(_) => return Ok(()),
            };
            let kind = payload.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            match kind {
                "stdin" => {
                    let data = payload.get("data").and_then(|v| v.as_str()).unwrap_or("");
                    let normalized = normalize_stdin_for_pty(data);
                    let _ = active.shell.write_stdin(&normalized).await;
                }
                "signal" => {
                    let signal = payload
                        .get("signal")
                        .and_then(|v| v.as_str())
                        .unwrap_or("INT")
                        .to_ascii_uppercase();
                    match signal.as_str() {
                        "INT" | "SIGINT" => {
                            let _ = active.shell.write_stdin("\u{3}").await;
                        }
                        "TSTP" | "SIGTSTP" => {
                            let _ = active.shell.write_stdin("\u{1a}").await;
                        }
                        _ => {
                            remove_reason = Some("signal stop");
                        }
                    }
                }
                "control" => {
                    if payload.get("action").and_then(|v| v.as_str()) == Some("stop") {
                        remove_reason = Some("stop requested by client");
                    }
                }
                "resize" => {
                    let cols = payload
                        .get("cols")
                        .or_else(|| payload.get("width"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(PTY_DEFAULT_COLS as u64)
                        .clamp(20, 400) as u16;
                    let rows = payload
                        .get("rows")
                        .or_else(|| payload.get("height"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(PTY_DEFAULT_ROWS as u64)
                        .clamp(5, 200) as u16;
                    let _ = active.shell.resize(cols, rows);
                }
                "upload_start" => {
                    match Self::begin_image_upload(active, &payload, max_upload_bytes) {
                        Ok(msg) => status_messages.push(msg),
                        Err(err) => {
                            status_messages.push(format!("Image upload error: {}", err));
                        }
                    }
                }
                "upload_chunk" => {
                    match Self::append_image_upload_chunk(active, &payload, max_upload_bytes) {
                        Ok(Some(msg)) => status_messages.push(msg),
                        Ok(None) => {}
                        Err(err) => {
                            status_messages.push(format!("Image upload error: {}", err));
                        }
                    }
                }
                "upload_commit" => match Self::commit_image_upload(active, &payload, &inbox_dir) {
                    Ok(result) => {
                        status_messages.push(result.status_message);
                        if let Some(prompt) = result.prompt_for_agent {
                            match active.shell.write_stdin(&prompt).await {
                                Ok(_) => status_messages
                                    .push("Image context sent to active agent.".to_string()),
                                Err(err) => status_messages.push(format!(
                                    "Image saved, but failed to send it to the active agent: {}",
                                    err
                                )),
                            }
                        } else {
                            status_messages.push(
                                "Image saved. No active CLI agent detected, so it was not auto-sent."
                                    .to_string(),
                            );
                        }
                    }
                    Err(err) => {
                        status_messages.push(format!("Image upload error: {}", err));
                    }
                },
                "upload_cancel" => match Self::cancel_image_upload(active, &payload) {
                    Ok(msg) => status_messages.push(msg),
                    Err(err) => {
                        status_messages.push(format!("Image upload error: {}", err));
                    }
                },
                "review_start" => {
                    let request_id = normalize_review_request_id(
                        payload.get("request_id").and_then(|v| v.as_str()),
                    )
                    .unwrap_or_else(next_review_request_id);
                    if let Some(running) = active.review_job.as_ref() {
                        let payload = json!({
                            "kind":"review_error",
                            "request_id": request_id,
                            "code":"review_already_running",
                            "message": format!(
                                "A review is already running (request_id={}). Cancel it before starting another one.",
                                running.request_id
                            ),
                        });
                        active.last_review_progress = Some(payload.clone());
                        active.last_review_result = None;
                        immediate_payloads.push(payload);
                    } else {
                        let reviewer = normalize_agent(
                            payload
                                .get("reviewer")
                                .and_then(|v| v.as_str())
                                .or_else(|| payload.get("agent").and_then(|v| v.as_str())),
                        )
                        .unwrap_or_else(|| "codex".to_string());
                        if let Some(profile) = review_profiles.get(&reviewer).cloned() {
                            let scope = ReviewScope::from_raw(
                                payload.get("scope").and_then(|v| v.as_str()),
                            );
                            let diff_base = ReviewDiffBase::from_raw(
                                payload.get("diff_base").and_then(|v| v.as_str()),
                            );
                            let notes = payload
                                .get("notes")
                                .and_then(|v| v.as_str())
                                .map(str::trim)
                                .filter(|v| !v.is_empty())
                                .map(|v| truncate_with_note(v, REVIEW_MAX_SECTION_CHARS));
                            let cwd = active.cwd.clone();
                            let replay_tail = collect_replay_tail(active, REVIEW_MAX_SECTION_CHARS);
                            let session_id = relay_session_id.to_string();
                            let request_id_for_task = request_id.clone();
                            let reviewer_for_task = reviewer.clone();
                            let tx_for_task = shell_events_tx.clone();

                            let queued_payload = json!({
                                "kind":"review_progress",
                                "request_id": request_id,
                                "stage":"queued",
                                "message": format!("Starting {} review...", reviewer),
                                "reviewer": reviewer,
                                "scope": scope.as_str(),
                                "diff_base": diff_base.as_str(),
                            });
                            active.review_stream_replay.clear();
                            active.review_stream_replay_bytes = 0;
                            active.next_review_stream_seq = 1;
                            active.last_review_progress = Some(queued_payload.clone());
                            active.last_review_result = None;
                            immediate_payloads.push(queued_payload);

                            let handle = tokio::spawn(async move {
                                run_review_job(
                                    tx_for_task,
                                    session_id,
                                    request_id_for_task,
                                    reviewer_for_task,
                                    profile,
                                    cwd,
                                    scope,
                                    diff_base,
                                    notes,
                                    replay_tail,
                                )
                                .await;
                            });
                            active.review_job = Some(ReviewJob { request_id, handle });
                        } else {
                            let payload = json!({
                                "kind":"review_error",
                                "request_id": request_id,
                                "code":"reviewer_not_configured",
                                "message": format!(
                                    "Reviewer `{}` is not configured. Supported: {}.",
                                    reviewer,
                                    SUPPORTED_TERMINAL_AGENTS.join(", ")
                                ),
                            });
                            active.last_review_progress = Some(payload.clone());
                            active.last_review_result = None;
                            immediate_payloads.push(payload);
                        }
                    }
                }
                "review_cancel" => {
                    let requested = normalize_review_request_id(
                        payload.get("request_id").and_then(|v| v.as_str()),
                    );
                    if let Some(job) = active.review_job.take() {
                        if let Some(requested_id) = requested {
                            if requested_id != job.request_id {
                                active.review_job = Some(job);
                                status_messages.push(format!(
                                    "Running review has id `{}`; nothing cancelled.",
                                    requested_id
                                ));
                            } else {
                                let request_id = job.request_id.clone();
                                job.handle.abort();
                                let payload = json!({
                                    "kind":"review_error",
                                    "request_id": request_id,
                                    "code":"review_cancelled",
                                    "message":"Review cancelled by user.",
                                });
                                active.last_review_progress = Some(payload.clone());
                                active.last_review_result = None;
                                immediate_payloads.push(payload);
                            }
                        } else {
                            let request_id = job.request_id.clone();
                            job.handle.abort();
                            let payload = json!({
                                "kind":"review_error",
                                "request_id": request_id,
                                "code":"review_cancelled",
                                "message":"Review cancelled by user.",
                            });
                            active.last_review_progress = Some(payload.clone());
                            active.last_review_result = None;
                            immediate_payloads.push(payload);
                        }
                    } else {
                        let payload = json!({
                            "kind":"review_error",
                            "request_id": requested.unwrap_or_else(next_review_request_id),
                            "code":"review_not_running",
                            "message":"No running review to cancel.",
                        });
                        active.last_review_progress = Some(payload.clone());
                        active.last_review_result = None;
                        immediate_payloads.push(payload);
                    }
                }
                _ => {}
            }
        }

        for message in status_messages {
            let _ = self
                .send_encrypted_json_for_session(
                    relay_session_id,
                    ws_write,
                    json!({"kind":"status","message":message}),
                )
                .await;
        }
        for payload in immediate_payloads {
            let _ = self
                .send_encrypted_json_for_session(relay_session_id, ws_write, payload)
                .await;
        }

        if let Some(reason) = remove_reason {
            self.remove_session(relay_session_id, reason).await;
        }
        Ok(())
    }

    async fn send_plain_json<S>(ws_write: &mut S, value: Value) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        ws_write
            .send(Message::Text(serde_json::to_string(&value)?))
            .await?;
        Ok(())
    }

    async fn send_encrypted_json_for_session<S>(
        &mut self,
        session_id: &str,
        ws_write: &mut S,
        payload: Value,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let Some(active) = self.sessions.get_mut(session_id) else {
            return Ok(());
        };
        let crypto = &mut active.crypto;
        let plaintext = serde_json::to_vec(&payload)?;
        if plaintext.len() > MAX_ENCRYPTED_PAYLOAD_BYTES {
            return Ok(());
        }

        let counter = crypto
            .send_counter
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("session counter overflow"))?;
        crypto.send_counter = counter;
        let aad = format!("{}:{}", crypto.session_id, counter);
        let mut iv = [0u8; 12];
        OsRng.fill_bytes(&mut iv);
        let ct = crypto
            .cipher
            .encrypt(
                Nonce::from_slice(&iv),
                Payload {
                    msg: &plaintext,
                    aad: aad.as_bytes(),
                },
            )
            .map_err(|_| anyhow::anyhow!("encrypt failed"))?;

        let envelope = json!({
            "type":"e2ee",
            "session_id": session_id,
            "c": counter,
            "iv": b64_encode(&iv),
            "ct": b64_encode(&ct),
        });
        Self::send_plain_json(ws_write, envelope).await
    }

    async fn remove_session(&mut self, session_id: &str, reason: &str) {
        if let Some(mut active) = self.sessions.remove(session_id) {
            debug!(reason, session_id, "Stopping local shell process");
            if let Some(job) = active.review_job.take() {
                job.handle.abort();
            }
            active.shell.stop().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_connect_token_mint_url_from_wss() {
        let url = daemon_connect_token_mint_url("wss://terminal.aidaemon.ai/v1/ws/daemon")
            .expect("mint url");
        assert_eq!(
            url,
            "https://terminal.aidaemon.ai/v1/daemon/connect-token/daemon-auth"
        );
    }

    #[test]
    fn test_daemon_connect_token_mint_url_rejects_insecure_ws() {
        assert!(daemon_connect_token_mint_url("ws://terminal.aidaemon.ai/v1/ws/daemon").is_err());
    }

    #[test]
    fn test_daemon_bootstrap_signing_input() {
        let input = daemon_bootstrap_signing_input("12345", "macbook-pro", 1700000000, "abcd1234");
        assert_eq!(
            input,
            "v1\nuser_id=12345\ndevice_id=macbook-pro\nts=1700000000\nnonce=abcd1234"
        );
    }

    #[test]
    fn test_daemon_bootstrap_signature_candidates_include_hkdf_and_legacy() {
        let signatures = daemon_bootstrap_signature_candidates(
            "123456:telegram-bot-token",
            "12345",
            "macbook-pro",
            1700000000,
            "abcd1234",
        )
        .expect("signature candidates");
        assert_eq!(signatures.len(), 2);
        assert_ne!(signatures[0], signatures[1]);
    }

    #[tokio::test]
    async fn test_mint_connect_token_falls_back_to_legacy_signature() {
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::routing::post;
        use axum::{Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        #[derive(Clone)]
        struct TestState {
            bot_token: String,
            calls: Arc<AtomicUsize>,
        }

        #[derive(Deserialize)]
        struct MintReq {
            user_id: String,
            device_id: String,
            ts: i64,
            nonce: String,
            sig: String,
        }

        #[derive(Serialize)]
        struct MintResp {
            ok: bool,
            daemon_connect_token: Option<String>,
            error: Option<String>,
            message: Option<String>,
        }

        async fn handler(
            State(state): State<TestState>,
            Json(req): Json<MintReq>,
        ) -> (StatusCode, Json<MintResp>) {
            state.calls.fetch_add(1, Ordering::SeqCst);
            let primary = sign_daemon_bootstrap_proof_hkdf(
                &state.bot_token,
                &req.user_id,
                &req.device_id,
                req.ts,
                &req.nonce,
            )
            .expect("primary signature");
            let legacy = sign_daemon_bootstrap_proof_legacy(
                &state.bot_token,
                &req.user_id,
                &req.device_id,
                req.ts,
                &req.nonce,
            )
            .expect("legacy signature");

            if req.sig == primary {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(MintResp {
                        ok: false,
                        daemon_connect_token: None,
                        error: Some("unauthorized".to_string()),
                        message: Some("primary rejected for compatibility test".to_string()),
                    }),
                );
            }
            if req.sig == legacy {
                return (
                    StatusCode::OK,
                    Json(MintResp {
                        ok: true,
                        daemon_connect_token: Some("minted-token".to_string()),
                        error: None,
                        message: None,
                    }),
                );
            }
            (
                StatusCode::BAD_REQUEST,
                Json(MintResp {
                    ok: false,
                    daemon_connect_token: None,
                    error: Some("bad_signature".to_string()),
                    message: Some("signature did not match expected candidates".to_string()),
                }),
            )
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let state = TestState {
            bot_token: "123456:telegram-bot-token".to_string(),
            calls: calls.clone(),
        };
        let app = Router::new().route("/", post(handler)).with_state(state);
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .expect("http client");
        let token = mint_connect_token_from_bot_proof(
            &client,
            &format!("http://{}/", addr),
            &["123456:telegram-bot-token".to_string()],
            "12345",
            "macbook-pro",
        )
        .await
        .expect("minted token");

        assert_eq!(token, "minted-token");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[test]
    fn test_derive_relay_cipher_round_trip() {
        let daemon_secret = SecretKey::random(&mut OsRng);
        let client_secret = SecretKey::random(&mut OsRng);
        let relay_session_id = "relay-session-1";

        let client_pub = client_secret.public_key().to_encoded_point(false);
        let client_pub_b64 = b64_encode(client_pub.as_bytes());
        let daemon_cipher =
            TerminalBridge::derive_relay_cipher(&daemon_secret, relay_session_id, &client_pub_b64)
                .expect("daemon cipher");

        let daemon_pub = daemon_secret.public_key();
        let shared = diffie_hellman(client_secret.to_nonzero_scalar(), daemon_pub.as_affine());
        let mut salt_hash = Sha256::new();
        salt_hash.update(relay_session_id.as_bytes());
        let salt = salt_hash.finalize();
        let hk = Hkdf::<Sha256>::new(Some(&salt), shared.raw_secret_bytes().as_slice());
        let mut key = [0u8; 32];
        hk.expand(KEY_INFO, &mut key)
            .expect("hkdf expansion should work");
        let client_cipher = Aes256Gcm::new_from_slice(&key).expect("aes cipher");

        let nonce = [7u8; 12];
        let aad = b"relay-session-1:42";
        let plaintext = b"terminal bridge test payload";
        let ciphertext = daemon_cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .expect("encrypt");
        let decrypted = client_cipher
            .decrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &ciphertext,
                    aad,
                },
            )
            .expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_derive_relay_cipher_session_id_mismatch_breaks_decryption() {
        let daemon_secret = SecretKey::random(&mut OsRng);
        let client_secret = SecretKey::random(&mut OsRng);

        let client_pub = client_secret.public_key().to_encoded_point(false);
        let client_pub_b64 = b64_encode(client_pub.as_bytes());
        let daemon_cipher = TerminalBridge::derive_relay_cipher(
            &daemon_secret,
            "relay-session-correct",
            &client_pub_b64,
        )
        .expect("daemon cipher");

        let daemon_pub = daemon_secret.public_key();
        let shared = diffie_hellman(client_secret.to_nonzero_scalar(), daemon_pub.as_affine());
        let mut salt_hash = Sha256::new();
        salt_hash.update("relay-session-wrong".as_bytes());
        let salt = salt_hash.finalize();
        let hk = Hkdf::<Sha256>::new(Some(&salt), shared.raw_secret_bytes().as_slice());
        let mut key = [0u8; 32];
        hk.expand(KEY_INFO, &mut key)
            .expect("hkdf expansion should work");
        let wrong_client_cipher = Aes256Gcm::new_from_slice(&key).expect("aes cipher");

        let nonce = [3u8; 12];
        let aad = b"relay-session-correct:1";
        let plaintext = b"hello";
        let ciphertext = daemon_cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .expect("encrypt");
        let decrypted = wrong_client_cipher.decrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &ciphertext,
                aad,
            },
        );
        assert!(decrypted.is_err());
    }

    #[test]
    fn test_normalize_image_mime() {
        assert_eq!(normalize_image_mime(Some("image/png")), Some("image/png"));
        assert_eq!(normalize_image_mime(Some("image/jpg")), Some("image/jpeg"));
        assert_eq!(
            normalize_image_mime(Some("image/jpeg; charset=utf-8")),
            Some("image/jpeg")
        );
        assert_eq!(normalize_image_mime(Some("image/webp")), Some("image/webp"));
        assert_eq!(normalize_image_mime(Some("application/pdf")), None);
    }

    #[test]
    fn test_normalize_upload_id() {
        assert_eq!(
            normalize_upload_id(Some(" img_01-abc ")).as_deref(),
            Some("img_01-abc")
        );
        assert!(normalize_upload_id(Some("$$$")).is_none());
    }

    #[test]
    fn test_sanitize_upload_filename_adds_extension() {
        assert_eq!(
            sanitize_upload_filename("screen shot", "image/png"),
            "screen shot.png"
        );
        assert_eq!(
            sanitize_upload_filename("photo.jpg", "image/jpeg"),
            "photo.jpg"
        );
    }

    #[test]
    fn test_build_upload_prompt_includes_context() {
        let path = PathBuf::from("/tmp/inbox/abc_screenshot.png");
        let prompt = build_upload_prompt(
            "screenshot.png",
            &path,
            "image/png",
            120_000,
            Some("error appears after login"),
        );
        assert!(prompt.contains("screenshot.png"));
        assert!(prompt.contains("/tmp/inbox/abc_screenshot.png"));
        assert!(prompt.contains("error appears after login"));
    }

    #[test]
    fn test_normalize_stdin_for_pty_decodes_single_enter_escape() {
        assert_eq!(normalize_stdin_for_pty("\\r"), "\r");
    }

    #[test]
    fn test_normalize_stdin_for_pty_decodes_ansi_escape() {
        assert_eq!(normalize_stdin_for_pty("\\u001b[C"), "\u{001b}[C");
    }

    #[test]
    fn test_normalize_stdin_for_pty_decodes_multiple_control_escapes() {
        assert_eq!(normalize_stdin_for_pty("status\\r\\t"), "status\r\t");
    }

    #[test]
    fn test_normalize_stdin_for_pty_decodes_trailing_submit_escape() {
        assert_eq!(normalize_stdin_for_pty("npm test\\r"), "npm test\r");
    }

    #[test]
    fn test_normalize_stdin_for_pty_preserves_non_control_backslashes() {
        assert_eq!(normalize_stdin_for_pty("echo \\q"), "echo \\q");
        assert_eq!(normalize_stdin_for_pty("C:\\temp\\logs"), "C:\\temp\\logs");
    }

    #[test]
    fn test_normalize_stdin_for_pty_preserves_non_ascii_hex_escapes() {
        assert_eq!(normalize_stdin_for_pty("\\x80"), "\\x80");
    }

    #[test]
    fn test_normalize_agent_args_drops_shell_metacharacters() {
        let payload = serde_json::json!(["--model", "gpt-5", "foo;bar", "$(whoami)", "--json"]);
        let args = normalize_agent_args(Some(&payload));
        assert_eq!(args, vec!["--model", "gpt-5", "--json"]);
    }

    #[test]
    fn test_normalize_review_profile_args_for_claude_forces_json() {
        let args = vec![
            "-p".to_string(),
            "--verbose".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
        ];
        let normalized = normalize_review_profile_args("claude", &args);
        assert!(normalized.contains(&"-p".to_string()));
        assert!(normalized.contains(&"--output-format".to_string()));
        assert!(normalized.contains(&"json".to_string()));
        assert!(!normalized.contains(&"--verbose".to_string()));
        assert!(!normalized.contains(&"stream-json".to_string()));
    }

    #[test]
    fn test_parse_review_output_payload_requires_structured_output() {
        let raw = "review done.\nfound some issues but no json returned";
        let result = parse_review_output_payload(
            "claude",
            "rvw_test_1",
            ReviewScope::Diff,
            ReviewDiffBase::WorkingTree,
            raw,
        );
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap_or_default()
            .contains("no structured JSON review payload"));
    }

    #[test]
    fn test_parse_review_output_payload_parses_structured_candidate() {
        let raw = r#"{"type":"result","result":"{\"verdict\":\"needs_changes\",\"blocking\":[\"Fix null check\"],\"risks\":[],\"suggestions\":[\"Add unit tests\"],\"summary\":\"Potential null dereference\"}"}"#;
        let payload = parse_review_output_payload(
            "claude",
            "rvw_test_2",
            ReviewScope::Diff,
            ReviewDiffBase::WorkingTree,
            raw,
        )
        .expect("structured review payload should parse");
        assert_eq!(
            payload.get("verdict").and_then(Value::as_str),
            Some("needs_changes")
        );
        assert_eq!(
            payload
                .get("blocking")
                .and_then(Value::as_array)
                .map(|v| v.len()),
            Some(1)
        );
    }

    #[test]
    fn test_parse_review_output_payload_parses_findings_and_drives_verdict() {
        let raw = r#"{"verdict":"","summary":"","findings":[{"severity":"high","file":"src/lib.rs","line":"42","issue":"Possible panic on unwrap()","fix":"Handle None before unwrap"}],"blocking":[],"risks":[],"suggestions":[]}"#;
        let payload = parse_review_output_payload(
            "codex",
            "rvw_test_3",
            ReviewScope::Diff,
            ReviewDiffBase::WorkingTree,
            raw,
        )
        .expect("structured finding payload should parse");

        assert_eq!(
            payload.get("verdict").and_then(Value::as_str),
            Some("needs_changes")
        );
        assert_eq!(
            payload
                .get("findings")
                .and_then(Value::as_array)
                .map(|v| v.len()),
            Some(1)
        );
        assert_eq!(
            payload.pointer("/findings/0/file").and_then(Value::as_str),
            Some("src/lib.rs")
        );
        assert_eq!(
            payload.pointer("/findings/0/line").and_then(Value::as_u64),
            Some(42)
        );
    }

    #[test]
    fn test_parse_review_numstat_parses_lines() {
        let raw = "12\t3\tsrc/main.rs\n-\t-\tpublic/logo.png\n";
        let parsed = parse_review_numstat(raw);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].path, "src/main.rs");
        assert_eq!(parsed[0].added, Some(12));
        assert_eq!(parsed[0].deleted, Some(3));
        assert!(!parsed[0].binary);
        assert_eq!(parsed[1].path, "public/logo.png");
        assert!(parsed[1].binary);
    }

    #[test]
    fn test_select_review_files_skips_generated_binary_and_large() {
        let changes = vec![
            ReviewFileChange {
                path: "src/app.rs".to_string(),
                added: Some(10),
                deleted: Some(2),
                binary: false,
            },
            ReviewFileChange {
                path: "public/logo.png".to_string(),
                added: None,
                deleted: None,
                binary: true,
            },
            ReviewFileChange {
                path: "node_modules/pkg/index.js".to_string(),
                added: Some(4),
                deleted: Some(1),
                binary: false,
            },
            ReviewFileChange {
                path: "src/huge_diff.rs".to_string(),
                added: Some(REVIEW_MAX_FILE_CHANGED_LINES + 1),
                deleted: Some(0),
                binary: false,
            },
        ];
        let (selected, stats) = select_review_files(&changes);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].path, "src/app.rs");
        assert_eq!(stats.total_changed_files, 4);
        assert_eq!(stats.included_files, 1);
        assert_eq!(stats.skipped_binary_files, 1);
        assert_eq!(stats.skipped_generated_files, 1);
        assert_eq!(stats.skipped_large_files, 1);
    }
}
