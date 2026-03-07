use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{debug, error, info, warn};

use crate::config::{resolve_from_keychain, store_in_keychain, AppConfig};
use crate::traits::StateStore;

const HEARTBEAT_MS: u64 = 25_000;
const RECONNECT_INITIAL_MS: u64 = 1_000;
const RECONNECT_MAX_MS: u64 = 30_000;
const RECONNECT_JITTER_MS: u64 = 500;
const RECONNECT_STABLE_SECS: u64 = 60;
const OUTBOUND_HIGH_QUEUE_CAP: usize = 128;
const OUTBOUND_LOW_QUEUE_CAP: usize = 1024;
const OUTBOUND_FLUSH_BURST: usize = 24;
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
const REATTACH_REPLAY_HARD_CAP_FRAMES: usize = 128;
const REATTACH_REPLAY_HARD_CAP_BYTES: usize = 512 * 1024;
const REATTACH_REPLAY_INTERACTIVE_CAP_FRAMES: usize = 16;
const REATTACH_REPLAY_INTERACTIVE_CAP_BYTES: usize = 96 * 1024;
const REATTACH_REVIEW_REPLAY_CAP_FRAMES: usize = 192;
const PTY_DEFAULT_ROWS: u16 = 36;
const PTY_DEFAULT_COLS: u16 = 120;
const KEY_INFO: &[u8] = b"aidaemon-terminal-v1";
const DAEMON_BOOTSTRAP_SIGNING_SALT: &[u8] = b"aidaemon-daemon-bootstrap-v1";
const DAEMON_BOOTSTRAP_SIGNING_INFO: &[u8] = b"hmac-signing-key";
const TERMINAL_DAEMON_KEYCHAIN_FIELD: &str = "terminal_daemon_private_key_v1";
const SUPPORTED_TERMINAL_AGENTS: &[&str] = &["codex", "claude", "gemini", "opencode"];
const LOCAL_ATTACH_ENDPOINT_FILENAME: &str = "attach-endpoint.json";
const LOCAL_ATTACH_SECRET_BYTES: usize = 24;
const LOCAL_ATTACH_MAX_FRAME_BYTES: usize = 128 * 1024;
type HmacSha256 = Hmac<Sha256>;
static NEXT_LOCAL_CLIENT_ID: AtomicU64 = AtomicU64::new(1);
static TERMINAL_BRIDGE_TASK_RUNNING: AtomicBool = AtomicBool::new(false);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutboundPriority {
    High,
    Low,
}

struct QueuedOutboundPayload {
    session_id: String,
    payload: Value,
}

#[derive(Default)]
struct OutboundQueue {
    // High-priority control/result frames are drained before bulk stdout/review stream frames.
    high: VecDeque<QueuedOutboundPayload>,
    low: VecDeque<QueuedOutboundPayload>,
}

impl OutboundQueue {
    fn is_empty(&self) -> bool {
        self.high.is_empty() && self.low.is_empty()
    }

    fn enqueue(
        &mut self,
        priority: OutboundPriority,
        entry: QueuedOutboundPayload,
    ) -> Option<QueuedOutboundPayload> {
        match priority {
            OutboundPriority::High => {
                let dropped = if self.high.len() >= OUTBOUND_HIGH_QUEUE_CAP {
                    self.high.pop_front()
                } else {
                    None
                };
                self.high.push_back(entry);
                dropped
            }
            OutboundPriority::Low => {
                let dropped = if self.low.len() >= OUTBOUND_LOW_QUEUE_CAP {
                    self.low.pop_front()
                } else {
                    None
                };
                self.low.push_back(entry);
                dropped
            }
        }
    }

    fn pop_next(&mut self) -> Option<QueuedOutboundPayload> {
        self.high.pop_front().or_else(|| self.low.pop_front())
    }
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
                let mut utf8_carry = Vec::<u8>::new();
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let decoded = decode_utf8_stream_chunk(&mut utf8_carry, &buf[..n]);
                            if decoded.is_empty() {
                                continue;
                            }
                            if events_tx
                                .send(ShellEvent::Output {
                                    session_id: read_session_id.clone(),
                                    data: decoded,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                if !utf8_carry.is_empty() {
                    let tail = String::from_utf8_lossy(&utf8_carry).to_string();
                    if !tail.is_empty() {
                        let _ = events_tx.send(ShellEvent::Output {
                            session_id: read_session_id.clone(),
                            data: tail,
                        });
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

fn decode_utf8_stream_chunk(carry: &mut Vec<u8>, incoming: &[u8]) -> String {
    if incoming.is_empty() {
        return String::new();
    }
    carry.extend_from_slice(incoming);
    let mut out = String::new();
    loop {
        match std::str::from_utf8(carry.as_slice()) {
            Ok(valid) => {
                out.push_str(valid);
                carry.clear();
                break;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0 {
                    if let Ok(prefix) = std::str::from_utf8(&carry[..valid_up_to]) {
                        out.push_str(prefix);
                    }
                }
                if let Some(error_len) = err.error_len() {
                    let drain_to = valid_up_to.saturating_add(error_len).min(carry.len());
                    let invalid = &carry[valid_up_to..drain_to];
                    if invalid.len() == 1 {
                        match invalid[0] {
                            // C1 controls can appear in terminal streams (not UTF-8 text).
                            // Normalize the common ones instead of printing replacement glyphs.
                            0x9B => out.push_str("\u{001b}["),
                            0x9D => out.push_str("\u{001b}]"),
                            0x90 => out.push_str("\u{001b}P"),
                            0x9C => out.push_str("\u{001b}\\"),
                            b if (0x80..=0x9F).contains(&b) => {}
                            _ => out.push('\u{FFFD}'),
                        }
                    } else {
                        out.push('\u{FFFD}');
                    }
                    carry.drain(..drain_to);
                    if carry.is_empty() {
                        break;
                    }
                } else {
                    // Incomplete UTF-8 sequence at the end. Keep trailing bytes for next read.
                    carry.drain(..valid_up_to);
                    break;
                }
            }
        }
    }
    out
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

pub fn spawn_if_configured(config: &AppConfig, state: std::sync::Arc<dyn StateStore>) -> bool {
    if !config.terminal.effective_bridge_enabled() {
        info!("Terminal bridge disabled by config ([terminal].bridge_enabled = false)");
        return false;
    }

    let user_id = resolve_daemon_user_id(config, config.terminal.effective_daemon_user_id());
    let Some(user_id) = user_id else {
        warn!("Terminal bridge disabled: unable to resolve Telegram owner user_id");
        return false;
    };

    let ws_url = config.terminal.effective_daemon_ws_url();
    if let Err(err) = validate_daemon_ws_url(&ws_url) {
        warn!(
            error = %err,
            ws_url = %ws_url,
            "Terminal bridge disabled: insecure or invalid daemon websocket URL"
        );
        return false;
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
                return false;
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
        return false;
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

    if TERMINAL_BRIDGE_TASK_RUNNING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        info!("Terminal bridge already running; skipping duplicate startup");
        return true;
    }

    tokio::spawn(async move {
        match TerminalBridge::new(runtime, state).await {
            Ok(mut bridge) => bridge.run_forever().await,
            Err(err) => error!(error = %err, "Failed to initialize terminal bridge"),
        }
        TERMINAL_BRIDGE_TASK_RUNNING.store(false, Ordering::Release);
    });
    true
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

fn merge_daemon_bot_tokens(
    configured_tokens: &[String],
    dynamic_bots: &[crate::traits::DynamicBot],
    user_id: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    for token in configured_tokens {
        let trimmed = token.trim();
        if trimmed.is_empty()
            || out
                .iter()
                .any(|existing: &String| existing.as_str() == trimmed)
        {
            continue;
        }
        out.push(trimmed.to_string());
    }

    for bot in dynamic_bots {
        if bot.channel_type != "telegram" {
            continue;
        }
        if !bot.allowed_user_ids.is_empty()
            && !bot
                .allowed_user_ids
                .iter()
                .any(|allowed| allowed.trim() == user_id)
        {
            continue;
        }
        let token = bot.bot_token.trim();
        if token.is_empty()
            || out
                .iter()
                .any(|existing: &String| existing.as_str() == token)
        {
            continue;
        }
        out.push(token.to_string());
    }

    out
}

async fn resolve_runtime_daemon_bot_tokens(
    configured_tokens: &[String],
    state: &dyn StateStore,
    user_id: &str,
) -> Vec<String> {
    match state.get_dynamic_bots().await {
        Ok(dynamic_bots) => merge_daemon_bot_tokens(configured_tokens, &dynamic_bots, user_id),
        Err(err) => {
            warn!(
                error = %err,
                "Failed to load dynamic Telegram bots for daemon bootstrap auth"
            );
            configured_tokens.to_vec()
        }
    }
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

fn normalize_relay_session_id(raw: Option<&str>) -> Option<String> {
    let value = raw.unwrap_or("").trim();
    if value.is_empty() || value.len() > 128 {
        return None;
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_' | '.'))
    {
        return None;
    }
    Some(value.to_string())
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

fn normalize_agent_args(value: Option<&Value>, agent: Option<&str>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    let mut raw_values: Vec<String> = Vec::new();
    match value {
        Value::Array(raw) => {
            for item in raw {
                if let Some(text) = item.as_str() {
                    raw_values.push(text.to_string());
                }
            }
        }
        Value::String(text) => {
            if let Ok(parsed) = shell_words::split(text) {
                raw_values.extend(parsed);
            } else {
                raw_values.push(text.clone());
            }
        }
        _ => return Vec::new(),
    }
    let out = raw_values
        .into_iter()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty() && !v.contains('\0'))
        .filter(|v| is_safe_agent_bootstrap_arg(v))
        .map(|v| v.chars().take(MAX_AGENT_ARG_CHARS).collect::<String>())
        .take(MAX_AGENT_ARGS)
        .collect();
    let (out, _) = crate::normalize_terminal_agent_permission_aliases(agent, out);
    out
}

fn requested_agent_args_from_client_hello(frame: &Value) -> Vec<String> {
    // Accept multiple key shapes to remain compatible with older/newer Mini App builds.
    let requested_agent = normalize_agent(frame.get("agent").and_then(|v| v.as_str()));
    let key_order = ["agent_args", "agentArgs", "args", "argv", "arg"];
    for key in key_order {
        if let Some(raw) = frame.get(key) {
            return normalize_agent_args(Some(raw), requested_agent.as_deref());
        }
    }
    Vec::new()
}

fn normalize_telegram_session_id(raw: Option<&str>) -> Option<String> {
    let session_id = raw?.trim();
    if session_id.is_empty() || session_id.len() > 128 {
        return None;
    }
    if !session_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_' | '.'))
    {
        return None;
    }
    Some(session_id.to_string())
}

fn requested_telegram_session_id_from_client_hello(frame: &Value) -> Option<String> {
    let key_order = [
        "telegram_session_id",
        "telegramSessionId",
        "chat_session_id",
        "chatSessionId",
    ];
    for key in key_order {
        if let Some(value) = normalize_telegram_session_id(frame.get(key).and_then(Value::as_str)) {
            return Some(value);
        }
    }
    if let Some(telegram_obj) = frame.get("telegram").and_then(Value::as_object) {
        for key in key_order {
            if let Some(value) =
                normalize_telegram_session_id(telegram_obj.get(key).and_then(Value::as_str))
            {
                return Some(value);
            }
        }
    }
    None
}

fn requested_relay_session_id_from_client_hello(frame: &Value) -> Option<String> {
    let key_order = [
        "relay_session_id",
        "relaySessionId",
        "requested_relay_session_id",
        "requestedRelaySessionId",
        "target_relay_session_id",
        "targetRelaySessionId",
    ];
    for key in key_order {
        if let Some(value) = normalize_relay_session_id(frame.get(key).and_then(Value::as_str)) {
            return Some(value);
        }
    }
    if let Some(relay_obj) = frame.get("relay").and_then(Value::as_object) {
        for key in key_order {
            if let Some(value) =
                normalize_relay_session_id(relay_obj.get(key).and_then(Value::as_str))
            {
                return Some(value);
            }
        }
        if let Some(value) =
            normalize_relay_session_id(relay_obj.get("session_id").and_then(Value::as_str))
        {
            return Some(value);
        }
        if let Some(value) =
            normalize_relay_session_id(relay_obj.get("sessionId").and_then(Value::as_str))
        {
            return Some(value);
        }
    }
    None
}

fn parse_boolish_client_hello_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(v) => Some(*v),
        Value::Number(n) => n
            .as_u64()
            .map(|v| v != 0)
            .or_else(|| n.as_i64().map(|v| v != 0)),
        Value::String(text) => {
            let normalized = text.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                return None;
            }
            match normalized.as_str() {
                "1" | "true" | "yes" | "y" | "on" => Some(true),
                "0" | "false" | "no" | "n" | "off" => Some(false),
                _ => None,
            }
        }
        _ => None,
    }
}

fn requested_fresh_start_from_client_hello(frame: &Value) -> bool {
    let key_order = [
        "fresh_start",
        "freshStart",
        "force_fresh_start",
        "forceFreshStart",
        "start_fresh",
        "startFresh",
    ];
    for key in key_order {
        if let Some(raw) = frame.get(key) {
            if let Some(value) = parse_boolish_client_hello_value(raw) {
                return value;
            }
        }
    }
    if let Some(launch_obj) = frame.get("launch").and_then(Value::as_object) {
        for key in key_order {
            if let Some(raw) = launch_obj.get(key) {
                if let Some(value) = parse_boolish_client_hello_value(raw) {
                    return value;
                }
            }
        }
    }
    false
}

fn is_safe_agent_bootstrap_arg(value: &str) -> bool {
    if value.contains("$(") || value.contains('\n') || value.contains('\r') {
        return false;
    }
    !value
        .chars()
        .any(|ch| matches!(ch, ';' | '|' | '&' | '>' | '<' | '`'))
}

fn should_quote_shell_token(value: &str) -> bool {
    value.is_empty()
        || value.chars().any(|ch| {
            !(ch.is_ascii_alphanumeric()
                || matches!(ch, '_' | '-' | '.' | '/' | ':' | '=' | '+' | ',' | '@'))
        })
}

fn shell_token(value: &str) -> String {
    if !should_quote_shell_token(value) {
        return value.to_string();
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

fn replay_frames_total_bytes(frames: &[(u64, String)]) -> usize {
    frames.iter().map(|(_, chunk)| chunk.len()).sum()
}

fn replay_frames_look_interactive(frames: &[(u64, String)]) -> bool {
    frames
        .iter()
        .any(|(_, chunk)| chunk.contains('\r') || chunk.contains('\u{001b}'))
}

fn should_skip_stdout_replay(
    resume_from_seq: u64,
    oldest_seq: u64,
    frames: &[(u64, String)],
) -> bool {
    if frames.is_empty() {
        return false;
    }
    if resume_from_seq.saturating_add(1) < oldest_seq {
        // We already lost continuity; replay from an arbitrary tail often renders as
        // visual garbage for interactive CLIs, so prefer a clean live resume.
        return true;
    }
    if resume_from_seq != 0 {
        return false;
    }

    let total_bytes = replay_frames_total_bytes(frames);
    if frames.len() > REATTACH_REPLAY_HARD_CAP_FRAMES
        || total_bytes > REATTACH_REPLAY_HARD_CAP_BYTES
    {
        return true;
    }

    replay_frames_look_interactive(frames)
        && (frames.len() > REATTACH_REPLAY_INTERACTIVE_CAP_FRAMES
            || total_bytes > REATTACH_REPLAY_INTERACTIVE_CAP_BYTES)
}

fn should_skip_review_stream_replay(
    resume_from_review_stream_seq: u64,
    oldest_review_stream_seq: u64,
    replay_len: usize,
) -> bool {
    if replay_len == 0 {
        return false;
    }
    if resume_from_review_stream_seq.saturating_add(1) < oldest_review_stream_seq {
        return true;
    }
    resume_from_review_stream_seq == 0 && replay_len > REATTACH_REVIEW_REPLAY_CAP_FRAMES
}

fn build_skipped_stdout_replay_status_message(
    resume_from_seq: u64,
    oldest_seq: u64,
    frames: &[(u64, String)],
) -> Option<String> {
    if !should_skip_stdout_replay(resume_from_seq, oldest_seq, frames) {
        return None;
    }
    let blank_terminal_hint =
        "If the terminal appears blank, press Enter or send a command to refresh live output.";
    let reason = if resume_from_seq.saturating_add(1) < oldest_seq {
        "buffered history was unavailable"
    } else {
        "buffered output was too noisy to replay cleanly"
    };
    Some(format!(
        "Connection recovered. Skipped replaying {} buffered output frame(s) because {}. Session is still active. {}",
        frames.len(),
        reason,
        blank_terminal_hint
    ))
}

fn build_skipped_review_stream_replay_status_message(
    resume_from_review_stream_seq: u64,
    oldest_review_stream_seq: u64,
    replay_len: usize,
) -> Option<String> {
    if !should_skip_review_stream_replay(
        resume_from_review_stream_seq,
        oldest_review_stream_seq,
        replay_len,
    ) {
        return None;
    }
    let reason = if resume_from_review_stream_seq.saturating_add(1) < oldest_review_stream_seq {
        "buffered review history was unavailable"
    } else {
        "the buffered review stream was too large to replay cleanly"
    };
    Some(format!(
        "Connection recovered. Skipped replaying {} buffered review stream chunk(s) because {}.",
        replay_len, reason
    ))
}

fn build_connection_notice_payload(message: String, tone: &str, scope: &str, ttl_ms: u64) -> Value {
    json!({
        "kind": "connection_notice",
        "message": message,
        "tone": tone,
        "scope": scope,
        "ttl_ms": ttl_ms,
    })
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

    let dir = terminal_bridge_state_dir();
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
    state: std::sync::Arc<dyn StateStore>,
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
            let resolved_bot_tokens =
                resolve_runtime_daemon_bot_tokens(&bot_tokens, state.as_ref(), &user_id).await;
            if resolved_bot_tokens.is_empty() {
                if let Some(static_token) = fallback_static_token {
                    warn!(
                        "No configured or dynamic Telegram bot tokens available; falling back to static daemon token"
                    );
                    return Ok(static_token);
                }
                return Err(anyhow::anyhow!(
                    "no Telegram bot token available for daemon bootstrap auth"
                ));
            }
            match mint_connect_token_from_bot_proof(
                &http_client,
                &mint_url,
                &resolved_bot_tokens,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalAttachEndpoint {
    version: u8,
    host: String,
    port: u16,
    secret: String,
    pid: u32,
    updated_at_unix: i64,
}

#[derive(Debug, Clone)]
struct LocalAttachClient {
    session_id: String,
    tx: tokio::sync::mpsc::UnboundedSender<Value>,
}

#[derive(Debug)]
struct LocalAttachAccepted {
    session_id: String,
    #[allow(dead_code)]
    resume_code: Option<String>,
    #[allow(dead_code)]
    resume_expires_at_unix: Option<i64>,
}

#[derive(Debug)]
struct LocalShareCreated {
    session_id: String,
    code: String,
    expires_at_unix: i64,
}

enum LocalAttachEvent {
    AttachRequest {
        client_id: u64,
        code: String,
        secret: String,
        tx: tokio::sync::mpsc::UnboundedSender<Value>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<LocalAttachAccepted>>,
    },
    StartRequest {
        client_id: u64,
        secret: String,
        agent: String,
        cwd: Option<String>,
        agent_args: Vec<String>,
        tx: tokio::sync::mpsc::UnboundedSender<Value>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<LocalAttachAccepted>>,
    },
    ShareRequest {
        secret: String,
        session_id: Option<String>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<LocalShareCreated>>,
    },
    Input {
        client_id: u64,
        data_b64: String,
    },
    Resize {
        client_id: u64,
        cols: u16,
        rows: u16,
    },
    Redraw {
        client_id: u64,
    },
    Close {
        client_id: u64,
    },
}

fn terminal_bridge_state_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".aidaemon-terminal")
}

fn local_attach_endpoint_path() -> PathBuf {
    terminal_bridge_state_dir().join(LOCAL_ATTACH_ENDPOINT_FILENAME)
}

async fn persist_local_attach_endpoint(port: u16, secret: &str) -> anyhow::Result<()> {
    let dir = terminal_bridge_state_dir();
    tokio::fs::create_dir_all(&dir).await?;
    set_owner_only_permissions(&dir, 0o700)?;

    let endpoint = LocalAttachEndpoint {
        version: 1,
        host: "127.0.0.1".to_string(),
        port,
        secret: secret.to_string(),
        pid: std::process::id(),
        updated_at_unix: chrono::Utc::now().timestamp(),
    };
    let path = local_attach_endpoint_path();
    let raw = serde_json::to_string_pretty(&endpoint)?;
    tokio::fs::write(&path, raw).await?;
    set_owner_only_permissions(&path, 0o600)?;
    Ok(())
}

async fn write_json_line<W>(writer: &mut W, value: &Value) -> anyhow::Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut line = serde_json::to_string(value)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

async fn spawn_local_attach_listener(
    events_tx: tokio::sync::mpsc::UnboundedSender<LocalAttachEvent>,
    secret: String,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    persist_local_attach_endpoint(port, &secret).await?;
    info!(port, "Local terminal attach endpoint ready");

    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((socket, _)) => {
                    let events_tx = events_tx.clone();
                    tokio::spawn(async move {
                        if let Err(err) = handle_local_attach_socket(socket, events_tx).await {
                            debug!(error = %err, "Local attach socket closed with error");
                        }
                    });
                }
                Err(err) => {
                    warn!(error = %err, "Local attach listener accept failed");
                    break;
                }
            }
        }
    });
    Ok(())
}

async fn handle_local_attach_socket(
    socket: TcpStream,
    events_tx: tokio::sync::mpsc::UnboundedSender<LocalAttachEvent>,
) -> anyhow::Result<()> {
    let client_id = NEXT_LOCAL_CLIENT_ID.fetch_add(1, Ordering::Relaxed);
    let (read_half, mut write_half) = socket.into_split();
    let mut reader = BufReader::new(read_half);

    let mut first_line = String::new();
    let first_read = reader.read_line(&mut first_line).await?;
    if first_read == 0 || first_line.len() > LOCAL_ATTACH_MAX_FRAME_BYTES {
        anyhow::bail!("missing or oversized attach frame");
    }
    let first_value: Value = serde_json::from_str(first_line.trim())
        .map_err(|e| anyhow::anyhow!("invalid attach frame: {}", e))?;
    let frame_type = first_value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    match frame_type {
        "attach" | "start" => {
            let (outbound_tx, mut outbound_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();

            if frame_type == "attach" {
                let code = first_value
                    .get("code")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or("")
                    .to_string();
                let secret = first_value
                    .get("secret")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or("")
                    .to_string();
                if code.is_empty() || secret.is_empty() {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message":"attach requires code and secret.",
                        }),
                    )
                    .await?;
                    return Ok(());
                }
                events_tx
                    .send(LocalAttachEvent::AttachRequest {
                        client_id,
                        code,
                        secret,
                        tx: outbound_tx.clone(),
                        response_tx,
                    })
                    .map_err(|_| anyhow::anyhow!("local attach service unavailable"))?;
            } else {
                let secret = first_value
                    .get("secret")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or("")
                    .to_string();
                let agent = normalize_agent(first_value.get("agent").and_then(Value::as_str))
                    .unwrap_or_else(|| "codex".to_string());
                let agent_args = requested_agent_args_from_client_hello(&first_value);
                let cwd = first_value
                    .get("cwd")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(|v| v.to_string());
                if secret.is_empty() {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message":"start requires secret.",
                        }),
                    )
                    .await?;
                    return Ok(());
                }
                events_tx
                    .send(LocalAttachEvent::StartRequest {
                        client_id,
                        secret,
                        agent,
                        cwd,
                        agent_args,
                        tx: outbound_tx.clone(),
                        response_tx,
                    })
                    .map_err(|_| anyhow::anyhow!("local attach service unavailable"))?;
            }

            let accepted = match response_rx.await {
                Ok(Ok(ok)) => ok,
                Ok(Err(err)) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message": err.to_string(),
                        }),
                    )
                    .await?;
                    return Ok(());
                }
                Err(_) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message":"local attach service did not respond.",
                        }),
                    )
                    .await?;
                    return Ok(());
                }
            };

            write_json_line(
                &mut write_half,
                &json!({
                    "type":"attached",
                    "session_id": accepted.session_id,
                    "resume_code": accepted.resume_code,
                    "resume_expires_at_unix": accepted.resume_expires_at_unix,
                }),
            )
            .await?;

            let writer_task = tokio::spawn(async move {
                while let Some(payload) = outbound_rx.recv().await {
                    if write_json_line(&mut write_half, &payload).await.is_err() {
                        break;
                    }
                }
            });

            let mut line = String::new();
            loop {
                line.clear();
                let n = reader.read_line(&mut line).await?;
                if n == 0 {
                    break;
                }
                if line.len() > LOCAL_ATTACH_MAX_FRAME_BYTES {
                    break;
                }
                let value: Value = match serde_json::from_str(line.trim()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match value.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "stdin" => {
                        let data_b64 = value
                            .get("data")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if data_b64.is_empty() {
                            continue;
                        }
                        let _ = events_tx.send(LocalAttachEvent::Input {
                            client_id,
                            data_b64,
                        });
                    }
                    "resize" => {
                        let cols = value
                            .get("cols")
                            .or_else(|| value.get("width"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(PTY_DEFAULT_COLS as u64)
                            .clamp(20, 400) as u16;
                        let rows = value
                            .get("rows")
                            .or_else(|| value.get("height"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(PTY_DEFAULT_ROWS as u64)
                            .clamp(5, 200) as u16;
                        let _ = events_tx.send(LocalAttachEvent::Resize {
                            client_id,
                            cols,
                            rows,
                        });
                    }
                    "redraw" => {
                        let _ = events_tx.send(LocalAttachEvent::Redraw { client_id });
                    }
                    "detach" => break,
                    _ => {}
                }
            }

            let _ = events_tx.send(LocalAttachEvent::Close { client_id });
            writer_task.abort();
            Ok(())
        }
        "share" => {
            let secret = first_value
                .get("secret")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .unwrap_or("")
                .to_string();
            let session_id = normalize_relay_session_id(
                first_value
                    .get("session_id")
                    .and_then(Value::as_str)
                    .or_else(|| first_value.get("sessionId").and_then(Value::as_str)),
            );
            if secret.is_empty() {
                write_json_line(
                    &mut write_half,
                    &json!({
                        "type":"error",
                        "message":"share requires secret.",
                    }),
                )
                .await?;
                return Ok(());
            }
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();
            events_tx
                .send(LocalAttachEvent::ShareRequest {
                    secret,
                    session_id,
                    response_tx,
                })
                .map_err(|_| anyhow::anyhow!("local attach service unavailable"))?;
            match response_rx.await {
                Ok(Ok(shared)) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"shared",
                            "session_id": shared.session_id,
                            "code": shared.code,
                            "expires_at_unix": shared.expires_at_unix,
                        }),
                    )
                    .await?;
                    Ok(())
                }
                Ok(Err(err)) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message": err.to_string(),
                        }),
                    )
                    .await?;
                    Ok(())
                }
                Err(_) => {
                    write_json_line(
                        &mut write_half,
                        &json!({
                            "type":"error",
                            "message":"local share service did not respond.",
                        }),
                    )
                    .await?;
                    Ok(())
                }
            }
        }
        _ => {
            write_json_line(
                &mut write_half,
                &json!({
                    "type":"error",
                    "message":"First frame must be type=attach, type=start, or type=share.",
                }),
            )
            .await?;
            Ok(())
        }
    }
}

struct TerminalBridge {
    cfg: RuntimeConfig,
    key_material: KeyMaterial,
    sessions: HashMap<String, ActiveSession>,
    state: std::sync::Arc<dyn StateStore>,
    http_client: reqwest::Client,
    shell_events_tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    shell_events_rx: tokio::sync::mpsc::UnboundedReceiver<ShellEvent>,
    local_attach_secret: String,
    local_clients: HashMap<u64, LocalAttachClient>,
    local_events_rx: tokio::sync::mpsc::UnboundedReceiver<LocalAttachEvent>,
}

impl TerminalBridge {
    async fn new(
        cfg: RuntimeConfig,
        state: std::sync::Arc<dyn StateStore>,
    ) -> anyhow::Result<Self> {
        let key_material = load_or_create_key_material().await?;
        let http_client = reqwest::Client::builder()
            .user_agent("aidaemon-terminal-bridge/1.0")
            .timeout(Duration::from_secs(12))
            .build()?;
        let (shell_events_tx, shell_events_rx) =
            tokio::sync::mpsc::unbounded_channel::<ShellEvent>();
        let (local_events_tx, local_events_rx) =
            tokio::sync::mpsc::unbounded_channel::<LocalAttachEvent>();
        let local_attach_secret = random_nonce_hex(LOCAL_ATTACH_SECRET_BYTES);
        if let Err(err) =
            spawn_local_attach_listener(local_events_tx.clone(), local_attach_secret.clone()).await
        {
            warn!(
                error = %err,
                "Failed to start local attach endpoint; agent attach from native terminal will be unavailable"
            );
        }
        info!(
            fingerprint = %key_material.fingerprint,
            user_id = %cfg.user_id,
            "Terminal bridge initialized"
        );
        Ok(Self {
            cfg,
            key_material,
            sessions: HashMap::new(),
            state,
            http_client,
            shell_events_tx,
            shell_events_rx,
            local_attach_secret,
            local_clients: HashMap::new(),
            local_events_rx,
        })
    }

    async fn run_forever(&mut self) {
        let stable_threshold = Duration::from_secs(RECONNECT_STABLE_SECS);
        let mut reconnect_backoff_ms = RECONNECT_INITIAL_MS;
        loop {
            self.drain_shell_events(512);
            self.drain_local_events(256).await;
            let started = Instant::now();
            match self.connect_once().await {
                Ok(()) => {
                    info!(
                        ran_for_secs = started.elapsed().as_secs(),
                        "Terminal bridge disconnected, reconnecting"
                    );
                }
                Err(err) => {
                    error!(
                        error = %err,
                        ran_for_secs = started.elapsed().as_secs(),
                        "Terminal bridge connection failed"
                    );
                }
            }
            let ran_for = started.elapsed();
            self.drain_shell_events(2048);
            self.drain_local_events(512).await;

            if ran_for >= stable_threshold {
                reconnect_backoff_ms = RECONNECT_INITIAL_MS;
            }
            let jitter_ms = if RECONNECT_JITTER_MS == 0 {
                0
            } else {
                rand::thread_rng().gen_range(0..=RECONNECT_JITTER_MS)
            };
            let sleep_ms = reconnect_backoff_ms.saturating_add(jitter_ms);
            debug!(
                sleep_ms,
                base_backoff_ms = reconnect_backoff_ms,
                jitter_ms,
                "Sleeping before terminal bridge reconnect"
            );
            tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
            reconnect_backoff_ms = Self::next_reconnect_backoff_ms(reconnect_backoff_ms);
        }
    }

    fn next_reconnect_backoff_ms(current: u64) -> u64 {
        current
            .saturating_mul(2)
            .clamp(RECONNECT_INITIAL_MS, RECONNECT_MAX_MS)
    }

    async fn connect_once(&mut self) -> anyhow::Result<()> {
        let connect_token = {
            let auth = self.cfg.auth.clone();
            let user_id = self.cfg.user_id.clone();
            let device_id = self.cfg.device_id.clone();
            let http_client = self.http_client.clone();
            let state = self.state.clone();
            resolve_connect_token(auth, state, user_id, device_id, http_client).await?
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
        let mut outbound_queue = OutboundQueue::default();
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
                biased;
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
                maybe_local_event = self.local_events_rx.recv() => {
                    let Some(event) = maybe_local_event else {
                        continue;
                    };
                    if let Err(err) = self.handle_local_attach_event(event).await {
                        warn!(error = %err, "Local attach event handling failed");
                    }
                }
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
                    if let Err(err) = self.handle_shell_event(event, Some(&mut outbound_queue)) {
                        warn!(error = %err, "Failed to relay shell output event");
                        continue;
                    }
                    // Under continuous PTY output (spinner frames, progress redraws), keep draining
                    // outbound payloads immediately so low-priority stdout doesn't get starved.
                    if !outbound_queue.is_empty() {
                        self.flush_outbound_queue(&mut outbound_queue, &mut ws_write, OUTBOUND_FLUSH_BURST)
                            .await?;
                    }
                }
                _ = std::future::ready(()), if !outbound_queue.is_empty() => {
                    self.flush_outbound_queue(&mut outbound_queue, &mut ws_write, OUTBOUND_FLUSH_BURST)
                        .await?;
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
                Ok(event) => {
                    if let Err(err) = self.handle_shell_event(event, None) {
                        warn!(error = %err, "Failed to process buffered PTY event");
                    }
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

    async fn drain_local_events(&mut self, max_events: usize) {
        let mut drained = 0usize;
        while drained < max_events {
            match self.local_events_rx.try_recv() {
                Ok(event) => {
                    if let Err(err) = self.handle_local_attach_event(event).await {
                        warn!(error = %err, "Local attach event handling failed");
                    }
                    drained = drained.saturating_add(1);
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
    }

    fn broadcast_local_payload(&mut self, session_id: &str, payload: Value) {
        let mut stale = Vec::new();
        for (client_id, client) in &self.local_clients {
            if client.session_id != session_id {
                continue;
            }
            if client.tx.send(payload.clone()).is_err() {
                stale.push(*client_id);
            }
        }
        for client_id in stale {
            self.local_clients.remove(&client_id);
        }
    }

    fn owner_user_id(&self) -> Option<u64> {
        self.cfg.user_id.parse::<u64>().ok()
    }

    fn allocate_local_relay_session_id(&self) -> String {
        for _ in 0..16 {
            let candidate = format!("native-{}", random_nonce_hex(8));
            if !self.sessions.contains_key(&candidate) {
                return candidate;
            }
        }
        format!("native-{}", random_nonce_hex(12))
    }

    async fn start_local_bridge_session(
        &mut self,
        agent: String,
        cwd_hint: Option<String>,
        agent_args: Vec<String>,
    ) -> anyhow::Result<String> {
        let session_id = self.allocate_local_relay_session_id();
        let cwd = resolve_session_cwd(cwd_hint.as_deref(), &self.cfg.default_cwd);
        let shell = ShellProcess::spawn(
            &self.cfg.shell,
            &cwd,
            &session_id,
            self.shell_events_tx.clone(),
        )
        .await?;

        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| anyhow::anyhow!("failed to initialize local relay cipher"))?;

        self.sessions.insert(
            session_id.clone(),
            ActiveSession {
                crypto: CryptoSession {
                    session_id: session_id.clone(),
                    cipher,
                    send_counter: 0,
                    recv_counter: 0,
                    agent: Some(agent),
                    agent_args: agent_args.clone(),
                    bootstrapped_agent: false,
                },
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

        // Native-started sessions still run through the same shell bootstrap path.
        if let Some(active) = self.sessions.get_mut(&session_id) {
            if let Some(agent_name) = active.crypto.agent.clone() {
                if active
                    .crypto
                    .agent_args
                    .iter()
                    .any(|arg| !is_safe_agent_bootstrap_arg(arg))
                {
                    anyhow::bail!("unsafe agent argument rejected for shell bootstrap");
                }
                let mut command_parts = Vec::with_capacity(1 + active.crypto.agent_args.len());
                command_parts.push(shell_token(&agent_name));
                command_parts.extend(active.crypto.agent_args.iter().map(|arg| shell_token(arg)));
                active
                    .shell
                    .write_stdin(&format!("{}\n", command_parts.join(" ")))
                    .await?;
                active.crypto.bootstrapped_agent = true;
            }
        }

        if let Some(owner_user_id) = self.owner_user_id() {
            if let Err(err) = crate::agent_handoff::set_last_active_relay_session_id(
                self.state.as_ref(),
                owner_user_id,
                &session_id,
            )
            .await
            {
                warn!(
                    error = %err,
                    session_id = %session_id,
                    "Failed to persist last active relay session after native start"
                );
            }
        }

        Ok(session_id)
    }

    async fn handle_local_attach_event(&mut self, event: LocalAttachEvent) -> anyhow::Result<()> {
        match event {
            LocalAttachEvent::AttachRequest {
                client_id,
                code,
                secret,
                tx,
                response_tx,
            } => {
                if secret != self.local_attach_secret {
                    let _ =
                        response_tx.send(Err(anyhow::anyhow!("local attach authorization failed")));
                    return Ok(());
                }
                let handoff =
                    match crate::agent_handoff::resolve_handoff_code(self.state.as_ref(), &code)
                        .await
                    {
                        Ok(value) => value,
                        Err(err) => {
                            let _ = response_tx.send(Err(err));
                            return Ok(());
                        }
                    };
                let session_id = handoff.relay_session_id.clone();
                if !self.sessions.contains_key(&session_id) {
                    let _ = response_tx.send(Err(anyhow::anyhow!(
                        "Session is no longer active. Start /agent open again from Telegram."
                    )));
                    return Ok(());
                }
                if let Err(err) =
                    crate::agent_handoff::consume_handoff_code(self.state.as_ref(), &code).await
                {
                    let _ = response_tx.send(Err(err));
                    return Ok(());
                }

                self.local_clients.insert(
                    client_id,
                    LocalAttachClient {
                        session_id: session_id.clone(),
                        tx: tx.clone(),
                    },
                );

                let replay = self
                    .sessions
                    .get(&session_id)
                    .map(|active| {
                        active
                            .replay
                            .iter()
                            .rev()
                            .take(64)
                            .map(|frame| frame.data.clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                for chunk in replay {
                    let _ = tx.send(json!({"type":"stdout","data":chunk}));
                }
                let _ = tx.send(json!({
                    "type":"status",
                    "message":"Attached to active session. Native terminal is now live.",
                }));
                let _ = response_tx.send(Ok(LocalAttachAccepted {
                    session_id,
                    resume_code: None,
                    resume_expires_at_unix: None,
                }));
            }
            LocalAttachEvent::StartRequest {
                client_id,
                secret,
                agent,
                cwd,
                agent_args,
                tx,
                response_tx,
            } => {
                if secret != self.local_attach_secret {
                    let _ =
                        response_tx.send(Err(anyhow::anyhow!("local start authorization failed")));
                    return Ok(());
                }

                let session_id = match self
                    .start_local_bridge_session(agent, cwd, agent_args)
                    .await
                {
                    Ok(id) => id,
                    Err(err) => {
                        let _ = response_tx.send(Err(err));
                        return Ok(());
                    }
                };

                self.local_clients.insert(
                    client_id,
                    LocalAttachClient {
                        session_id: session_id.clone(),
                        tx: tx.clone(),
                    },
                );
                let replay = self
                    .sessions
                    .get(&session_id)
                    .map(|active| {
                        active
                            .replay
                            .iter()
                            .rev()
                            .take(64)
                            .map(|frame| frame.data.clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                for chunk in replay {
                    let _ = tx.send(json!({"type":"stdout","data":chunk}));
                }

                let mut resume_code: Option<String> = None;
                let mut resume_expires_at_unix: Option<i64> = None;
                if let Some(owner_user_id) = self.owner_user_id() {
                    match crate::agent_handoff::create_handoff_code(
                        self.state.as_ref(),
                        &session_id,
                        owner_user_id,
                    )
                    .await
                    {
                        Ok(handoff) => {
                            resume_code = Some(handoff.code);
                            resume_expires_at_unix = Some(handoff.expires_at_unix);
                        }
                        Err(err) => {
                            warn!(
                                error = %err,
                                session_id = %session_id,
                                "Failed to auto-create resume code for native-started session"
                            );
                        }
                    }
                }

                let _ = tx.send(json!({
                    "type":"status",
                    "message":"Native terminal session started. Use `aidaemon share` to generate a new Telegram resume code any time.",
                }));
                let _ = response_tx.send(Ok(LocalAttachAccepted {
                    session_id,
                    resume_code,
                    resume_expires_at_unix,
                }));
            }
            LocalAttachEvent::ShareRequest {
                secret,
                session_id,
                response_tx,
            } => {
                if secret != self.local_attach_secret {
                    let _ =
                        response_tx.send(Err(anyhow::anyhow!("local share authorization failed")));
                    return Ok(());
                }
                let Some(owner_user_id) = self.owner_user_id() else {
                    let _ = response_tx.send(Err(anyhow::anyhow!(
                        "Configured user id is not numeric; cannot generate share code."
                    )));
                    return Ok(());
                };

                let resolved_session_id = if let Some(explicit) = session_id {
                    explicit
                } else if let Ok(Some(last)) =
                    crate::agent_handoff::get_last_active_relay_session_id(
                        self.state.as_ref(),
                        owner_user_id,
                    )
                    .await
                {
                    last
                } else if self.sessions.len() == 1 {
                    self.sessions.keys().next().cloned().unwrap_or_default()
                } else {
                    String::new()
                };

                if resolved_session_id.is_empty()
                    || !self.sessions.contains_key(&resolved_session_id)
                {
                    let _ = response_tx.send(Err(anyhow::anyhow!(
                        "No active native session found. Start one with `aidaemon codex` (or claude/gemini/opencode) first."
                    )));
                    return Ok(());
                }

                let created = match crate::agent_handoff::create_handoff_code(
                    self.state.as_ref(),
                    &resolved_session_id,
                    owner_user_id,
                )
                .await
                {
                    Ok(value) => value,
                    Err(err) => {
                        let _ = response_tx.send(Err(err));
                        return Ok(());
                    }
                };
                let _ = response_tx.send(Ok(LocalShareCreated {
                    session_id: resolved_session_id,
                    code: created.code,
                    expires_at_unix: created.expires_at_unix,
                }));
            }
            LocalAttachEvent::Input {
                client_id,
                data_b64,
            } => {
                let Some(client) = self.local_clients.get(&client_id).cloned() else {
                    return Ok(());
                };
                let bytes = match b64_decode_flexible(&data_b64) {
                    Ok(v) => v,
                    Err(_) => return Ok(()),
                };
                if bytes.is_empty() {
                    return Ok(());
                }
                let stdin = String::from_utf8_lossy(&bytes).to_string();
                let Some(active) = self.sessions.get_mut(&client.session_id) else {
                    let _ = client.tx.send(json!({
                        "type":"error",
                        "message":"Session is no longer active.",
                    }));
                    self.local_clients.remove(&client_id);
                    return Ok(());
                };
                if let Err(err) = active.shell.write_stdin(&stdin).await {
                    let _ = client.tx.send(json!({
                        "type":"error",
                        "message": format!("Failed to forward stdin: {}", err),
                    }));
                    self.local_clients.remove(&client_id);
                }
            }
            LocalAttachEvent::Resize {
                client_id,
                cols,
                rows,
            } => {
                let Some(client) = self.local_clients.get(&client_id).cloned() else {
                    return Ok(());
                };
                if let Some(active) = self.sessions.get_mut(&client.session_id) {
                    let _ = active.shell.resize(cols, rows);
                } else {
                    self.local_clients.remove(&client_id);
                }
            }
            LocalAttachEvent::Redraw { client_id } => {
                let Some(client) = self.local_clients.get(&client_id).cloned() else {
                    return Ok(());
                };
                if let Some(active) = self.sessions.get_mut(&client.session_id) {
                    let _ = active.shell.write_stdin("\u{000c}").await;
                } else {
                    self.local_clients.remove(&client_id);
                }
            }
            LocalAttachEvent::Close { client_id } => {
                self.local_clients.remove(&client_id);
            }
        }
        Ok(())
    }

    fn handle_shell_event(
        &mut self,
        event: ShellEvent,
        outbound_queue: Option<&mut OutboundQueue>,
    ) -> anyhow::Result<()> {
        match event {
            ShellEvent::Output { session_id, data } => {
                let frames = self.record_stdout_chunks(&session_id, &data);
                for (_, chunk) in &frames {
                    self.broadcast_local_payload(
                        &session_id,
                        json!({"type":"stdout","data": chunk}),
                    );
                }
                if let Some(queue) = outbound_queue {
                    let mut dropped = 0usize;
                    for (seq, chunk) in frames {
                        if queue
                            .enqueue(
                                OutboundPriority::Low,
                                QueuedOutboundPayload {
                                    session_id: session_id.clone(),
                                    payload: json!({"kind":"stdout","seq":seq,"data":chunk}),
                                },
                            )
                            .is_some()
                        {
                            dropped = dropped.saturating_add(1);
                        }
                    }
                    if dropped > 0 {
                        warn!(
                            session_id = %session_id,
                            dropped,
                            "Terminal bridge low-priority outbound queue saturated; dropped stale stdout frames"
                        );
                    }
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
                if let Some(queue) = outbound_queue {
                    if queue
                        .enqueue(
                            OutboundPriority::High,
                            QueuedOutboundPayload {
                                session_id: session_id.clone(),
                                payload,
                            },
                        )
                        .is_some()
                    {
                        warn!(
                            session_id = %session_id,
                            "Terminal bridge high-priority outbound queue saturated; dropped oldest progress frame"
                        );
                    }
                }
            }
            ShellEvent::ReviewResult {
                session_id,
                request_id,
                payload,
            } => {
                let payload = self.record_review_result_payload(&session_id, &request_id, payload);
                if let Some(queue) = outbound_queue {
                    if queue
                        .enqueue(
                            OutboundPriority::High,
                            QueuedOutboundPayload {
                                session_id: session_id.clone(),
                                payload,
                            },
                        )
                        .is_some()
                    {
                        warn!(
                            session_id = %session_id,
                            "Terminal bridge high-priority outbound queue saturated; dropped oldest result frame"
                        );
                    }
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
                if let Some(queue) = outbound_queue {
                    if queue
                        .enqueue(
                            OutboundPriority::High,
                            QueuedOutboundPayload {
                                session_id: session_id.clone(),
                                payload,
                            },
                        )
                        .is_some()
                    {
                        warn!(
                            session_id = %session_id,
                            "Terminal bridge high-priority outbound queue saturated; dropped oldest error frame"
                        );
                    }
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
                if let Some(queue) = outbound_queue {
                    let mut dropped = 0usize;
                    for (seq, request_id, stream, data) in frames {
                        if queue
                            .enqueue(
                                OutboundPriority::Low,
                                QueuedOutboundPayload {
                                    session_id: session_id.clone(),
                                    payload: json!({
                                        "kind":"review_stream",
                                        "seq": seq,
                                        "request_id": request_id,
                                        "stream": stream,
                                        "data": data,
                                    }),
                                },
                            )
                            .is_some()
                        {
                            dropped = dropped.saturating_add(1);
                        }
                    }
                    if dropped > 0 {
                        warn!(
                            session_id = %session_id,
                            dropped,
                            "Terminal bridge low-priority outbound queue saturated; dropped stale review stream frames"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    async fn flush_outbound_queue<S>(
        &mut self,
        outbound_queue: &mut OutboundQueue,
        ws_write: &mut S,
        max_batch: usize,
    ) -> anyhow::Result<()>
    where
        S: futures::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        for _ in 0..max_batch {
            let Some(queued) = outbound_queue.pop_next() else {
                break;
            };
            self.send_encrypted_json_for_session(&queued.session_id, ws_write, queued.payload)
                .await?;
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
            frames,
            oldest_review_stream_seq,
            review_stream_frames,
            last_review_progress,
            last_review_result,
            running_review_request_id,
        ) = {
            let Some(active) = self.sessions.get(session_id) else {
                return Ok(());
            };
            let oldest = active.replay.front().map(|f| f.seq).unwrap_or(0);
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
                frames,
                oldest_review,
                review_stream_frames,
                active.last_review_progress.clone(),
                active.last_review_result.clone(),
                active.review_job.as_ref().map(|job| job.request_id.clone()),
            )
        };

        let skipped_stdout_replay = should_skip_stdout_replay(resume_from_seq, oldest_seq, &frames);
        if !frames.is_empty() {
            if skipped_stdout_replay {
                if let Some(message) =
                    build_skipped_stdout_replay_status_message(resume_from_seq, oldest_seq, &frames)
                {
                    self.send_encrypted_json_for_session(
                        session_id,
                        ws_write,
                        build_connection_notice_payload(message, "warn", "terminal", 5200),
                    )
                    .await?;
                }
            } else {
                for (seq, chunk) in &frames {
                    self.send_encrypted_json_for_session(
                        session_id,
                        ws_write,
                        json!({"kind":"stdout","seq":seq,"data":chunk}),
                    )
                    .await?;
                }
            }
        }

        let skipped_review_stream_replay = should_skip_review_stream_replay(
            resume_from_review_stream_seq,
            oldest_review_stream_seq,
            review_stream_frames.len(),
        );
        if !review_stream_frames.is_empty() {
            if skipped_review_stream_replay {
                if let Some(message) = build_skipped_review_stream_replay_status_message(
                    resume_from_review_stream_seq,
                    oldest_review_stream_seq,
                    review_stream_frames.len(),
                ) {
                    self.send_encrypted_json_for_session(
                        session_id,
                        ws_write,
                        build_connection_notice_payload(message, "warn", "review", 5200),
                    )
                    .await?;
                }
            } else {
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
            }
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
        if running_review_request_id.is_some()
            && (frames.is_empty() || skipped_stdout_replay)
            && !has_last_progress
        {
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
        if let Ok(owner_user_id) = self.cfg.user_id.parse::<u64>() {
            if let Err(err) = crate::agent_handoff::set_last_active_relay_session_id(
                self.state.as_ref(),
                owner_user_id,
                relay_session_id,
            )
            .await
            {
                warn!(
                    error = %err,
                    relay_session_id,
                    "Failed to persist last active relay session id"
                );
            }
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
        let requested_agent_args = requested_agent_args_from_client_hello(frame);
        let requested_relay_session_id = requested_relay_session_id_from_client_hello(frame);
        let telegram_session_id = requested_telegram_session_id_from_client_hello(frame);
        let requested_fresh_start = requested_fresh_start_from_client_hello(frame);
        let mut remap_source_session_id: Option<String> = None;
        let mut remap_reason: Option<&'static str> = None;
        if let Some(explicit_relay_session_id) = requested_relay_session_id.as_deref() {
            if explicit_relay_session_id != relay_session_id
                && self.sessions.contains_key(explicit_relay_session_id)
            {
                remap_source_session_id = Some(explicit_relay_session_id.to_string());
                remap_reason = Some("explicit relay_session_id from client hello");
            }
        }
        if remap_source_session_id.is_none() {
            if let Some(telegram_session_id) = telegram_session_id.as_deref() {
                match crate::agent_handoff::resolve_relay_for_telegram_session(
                    self.state.as_ref(),
                    telegram_session_id,
                )
                .await
                {
                    Ok(Some(mapped)) => {
                        if mapped != relay_session_id && self.sessions.contains_key(&mapped) {
                            remap_source_session_id = Some(mapped);
                            remap_reason = Some("telegram session binding");
                        }
                    }
                    Ok(None) => {}
                    Err(err) => {
                        warn!(
                            error = %err,
                            telegram_session_id = %telegram_session_id,
                            "Failed to resolve Telegram-to-relay mapping during client hello"
                        );
                    }
                }
            }
        }
        if let Some(from_session_id) = remap_source_session_id.as_deref() {
            if let Err(err) = self
                .remap_active_session_id(from_session_id, relay_session_id)
                .await
            {
                warn!(
                    error = %err,
                    from_session_id = %from_session_id,
                    to_session_id = %relay_session_id,
                    telegram_session_id = telegram_session_id.as_deref().unwrap_or(""),
                    requested_relay_session_id = requested_relay_session_id.as_deref().unwrap_or(""),
                    "Failed to remap native session during client hello"
                );
            } else {
                info!(
                    from_session_id = %from_session_id,
                    to_session_id = %relay_session_id,
                    remap_reason = remap_reason.unwrap_or("unknown"),
                    telegram_session_id = telegram_session_id.as_deref().unwrap_or(""),
                    requested_relay_session_id = requested_relay_session_id.as_deref().unwrap_or(""),
                    "Remapped active native session to client relay session id"
                );
            }
        }
        let cipher = Self::derive_relay_cipher(
            &self.key_material.private_key,
            relay_session_id,
            &client_pub_b64,
        )?;

        // Only tear down an already-bootstrapped session when the client explicitly
        // asks for a fresh start. A plain Mini App reopen resets local replay cursors
        // to zero, so resume_from_seq == 0 alone must not trigger session teardown.
        if requested_fresh_start && requested_agent.is_some() && resume_from_seq == 0 {
            if let Some(active) = self.sessions.get(relay_session_id) {
                if active.crypto.bootstrapped_agent {
                    info!(
                        relay_session_id,
                        old_agent = active.crypto.agent.as_deref().unwrap_or("none"),
                        new_agent = requested_agent.as_deref().unwrap_or("none"),
                        "Tearing down stale session for explicit fresh agent start"
                    );
                    self.remove_session(relay_session_id, "fresh agent start")
                        .await;
                }
            }
        }

        if let Some(active) = self.sessions.get_mut(relay_session_id) {
            active.crypto.cipher = cipher;
            active.crypto.send_counter = 0;
            active.crypto.recv_counter = 0;
            if !active.crypto.bootstrapped_agent {
                active.crypto.agent = requested_agent.clone();
                active.crypto.agent_args = requested_agent_args.clone();
            }
            if let Some(telegram_session_id) = telegram_session_id.as_deref() {
                if let Err(err) = crate::agent_handoff::bind_telegram_session_to_relay(
                    self.state.as_ref(),
                    telegram_session_id,
                    relay_session_id,
                )
                .await
                {
                    warn!(
                        error = %err,
                        telegram_session_id = %telegram_session_id,
                        relay_session_id = %relay_session_id,
                        "Failed to persist Telegram-to-relay session mapping"
                    );
                }
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
                json!({
                    "kind":"status",
                    "message":"Local daemon secure channel ready (reattached). Session is active; if the terminal looks blank, press Enter or send a command to refresh output."
                }),
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
        if let Some(telegram_session_id) = telegram_session_id.as_deref() {
            if let Err(err) = crate::agent_handoff::bind_telegram_session_to_relay(
                self.state.as_ref(),
                telegram_session_id,
                relay_session_id,
            )
            .await
            {
                warn!(
                    error = %err,
                    telegram_session_id = %telegram_session_id,
                    relay_session_id = %relay_session_id,
                    "Failed to persist Telegram-to-relay session mapping"
                );
            }
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
        command_parts.push(shell_token(&agent));
        command_parts.extend(crypto.agent_args.iter().map(|arg| shell_token(arg)));
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

    async fn remap_active_session_id(
        &mut self,
        from_session_id: &str,
        to_session_id: &str,
    ) -> anyhow::Result<()> {
        if from_session_id == to_session_id {
            return Ok(());
        }
        if !self.sessions.contains_key(from_session_id) {
            anyhow::bail!("source session does not exist");
        }
        if self.sessions.contains_key(to_session_id) {
            // Telegram may resume an older scope-bound session id; when an explicit
            // /agent resume mapping points to a different active native session,
            // treat the mapping as authoritative and replace the stale target.
            self.remove_session(to_session_id, "session replaced by /agent resume mapping")
                .await;
        }

        let mut active = self
            .sessions
            .remove(from_session_id)
            .ok_or_else(|| anyhow::anyhow!("source session disappeared"))?;
        active.crypto.session_id = to_session_id.to_string();
        self.sessions.insert(to_session_id.to_string(), active);

        for client in self.local_clients.values_mut() {
            if client.session_id == from_session_id {
                client.session_id = to_session_id.to_string();
            }
        }

        if let Some(owner_user_id) = self.owner_user_id() {
            if let Err(err) = crate::agent_handoff::set_last_active_relay_session_id(
                self.state.as_ref(),
                owner_user_id,
                to_session_id,
            )
            .await
            {
                warn!(
                    error = %err,
                    from_session_id = %from_session_id,
                    to_session_id = %to_session_id,
                    "Failed to persist remapped relay session id"
                );
            }
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
        self.broadcast_local_payload(
            session_id,
            json!({
                "type":"exit",
                "reason": reason,
            }),
        );
        self.local_clients
            .retain(|_, client| client.session_id != session_id);
        if let Some(mut active) = self.sessions.remove(session_id) {
            debug!(reason, session_id, "Stopping local shell process");
            if let Some(job) = active.review_job.take() {
                job.handle.abort();
            }
            active.shell.stop().await;
        }
    }
}

fn load_local_attach_endpoint() -> anyhow::Result<LocalAttachEndpoint> {
    let path = local_attach_endpoint_path();
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read local attach endpoint at {}: {}",
            path.display(),
            e
        )
    })?;
    let endpoint: LocalAttachEndpoint = serde_json::from_str(&raw).map_err(|e| {
        anyhow::anyhow!(
            "Invalid local attach endpoint JSON at {}: {}",
            path.display(),
            e
        )
    })?;
    if endpoint.host.trim().is_empty() || endpoint.port == 0 || endpoint.secret.trim().is_empty() {
        anyhow::bail!(
            "Local attach endpoint is incomplete at {}. Restart aidaemon.",
            path.display()
        );
    }
    Ok(endpoint)
}

#[cfg(unix)]
struct RawTerminalGuard {
    fd: i32,
    original: libc::termios,
}

#[cfg(unix)]
impl RawTerminalGuard {
    fn enable_if_tty() -> anyhow::Result<Option<Self>> {
        let fd = libc::STDIN_FILENO;
        // SAFETY: libc::isatty only reads fd metadata.
        if unsafe { libc::isatty(fd) } != 1 {
            return Ok(None);
        }
        // SAFETY: `termios` is plain old data and we initialize it via tcgetattr before use.
        let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
        // SAFETY: tcgetattr/tcsetattr are called with a valid terminal fd and termios pointer.
        if unsafe { libc::tcgetattr(fd, &mut original) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut raw = original;
        raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
        raw.c_iflag &= !(libc::IXON | libc::ICRNL);
        raw.c_oflag &= !(libc::OPOST);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Some(Self { fd, original }))
    }
}

#[cfg(unix)]
impl Drop for RawTerminalGuard {
    fn drop(&mut self) {
        // SAFETY: restoring previously captured termios back to the same fd.
        let _ = unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.original) };
    }
}

#[cfg(not(unix))]
struct RawTerminalGuard;

#[cfg(not(unix))]
impl RawTerminalGuard {
    fn enable_if_tty() -> anyhow::Result<Option<Self>> {
        Ok(None)
    }
}

fn send_local_attach_json_line(
    writer: &mut std::net::TcpStream,
    value: &Value,
) -> anyhow::Result<()> {
    let mut line = serde_json::to_string(value)?;
    line.push('\n');
    writer.write_all(line.as_bytes())?;
    writer.flush()?;
    Ok(())
}

fn send_local_attach_resize_and_redraw(
    writer: &mut std::net::TcpStream,
    cols: u16,
    rows: u16,
) -> anyhow::Result<()> {
    send_local_attach_json_line(
        writer,
        &json!({
            "type":"resize",
            "cols": cols,
            "rows": rows,
        }),
    )?;
    send_local_attach_json_line(writer, &json!({"type":"redraw"}))?;
    Ok(())
}

#[cfg(unix)]
fn spawn_local_attach_resize_watcher(mut writer: std::net::TcpStream) {
    std::thread::spawn(move || {
        let mut last_sent = local_terminal_size();
        let mut warmup_resyncs_left = 2u8;
        loop {
            std::thread::sleep(Duration::from_millis(250));
            let Some((cols, rows)) = local_terminal_size() else {
                continue;
            };
            let changed = last_sent != Some((cols, rows));
            if !changed && warmup_resyncs_left == 0 {
                continue;
            }
            if send_local_attach_resize_and_redraw(&mut writer, cols, rows).is_err() {
                break;
            }
            last_sent = Some((cols, rows));
            warmup_resyncs_left = warmup_resyncs_left.saturating_sub(1);
        }
    });
}

#[cfg(not(unix))]
fn spawn_local_attach_resize_watcher(_writer: std::net::TcpStream) {}

fn run_local_attach_stdout_loop(
    reader: &mut std::io::BufReader<std::net::TcpStream>,
) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        if line.len() > LOCAL_ATTACH_MAX_FRAME_BYTES {
            anyhow::bail!("received oversized frame from local attach endpoint");
        }
        let value: Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match value.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "stdout" => {
                if let Some(data) = value.get("data").and_then(|v| v.as_str()) {
                    stdout.write_all(data.as_bytes())?;
                    stdout.flush()?;
                }
            }
            "status" => {
                if let Some(message) = value.get("message").and_then(|v| v.as_str()) {
                    eprintln!("\r\n[status] {}", message);
                }
            }
            "error" => {
                let message = value
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error");
                anyhow::bail!("{}", message);
            }
            "exit" => {
                let reason = value
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("session ended");
                eprintln!("\r\n[session] {}", reason);
                break;
            }
            _ => {}
        }
    }
    Ok(())
}

fn run_local_attach_stdin_pump(mut writer: std::net::TcpStream) -> anyhow::Result<()> {
    let mut stdin = std::io::stdin();
    let mut input = [0u8; 2048];
    loop {
        let n = stdin.read(&mut input)?;
        if n == 0 {
            let _ = send_local_attach_json_line(&mut writer, &json!({"type":"detach"}));
            break;
        }
        let chunk = &input[..n];
        if chunk.len() == 1 && chunk[0] == 0x1d {
            // Ctrl+]
            let _ = send_local_attach_json_line(&mut writer, &json!({"type":"detach"}));
            break;
        }
        let payload = json!({
            "type":"stdin",
            "data": b64_encode(chunk),
        });
        send_local_attach_json_line(&mut writer, &payload)?;
    }
    Ok(())
}

#[cfg(unix)]
fn local_terminal_size() -> Option<(u16, u16)> {
    // SAFETY: winsize is POD and ioctl fills it when fd is a terminal.
    let mut winsize = unsafe { std::mem::zeroed::<libc::winsize>() };
    // SAFETY: ioctl is called with valid fd and pointer to winsize.
    let rc = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut winsize) };
    if rc != 0 || winsize.ws_col == 0 || winsize.ws_row == 0 {
        return None;
    }
    Some((winsize.ws_col, winsize.ws_row))
}

#[cfg(not(unix))]
fn local_terminal_size() -> Option<(u16, u16)> {
    None
}

pub fn run_local_attach_cli(code: &str) -> anyhow::Result<()> {
    let code = code.trim();
    if code.is_empty() {
        anyhow::bail!("Usage: aidaemon attach <code>");
    }
    let endpoint = load_local_attach_endpoint()?;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let stream = std::net::TcpStream::connect(&address).map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to local attach endpoint at {}: {}. Is aidaemon running?",
            address,
            e
        )
    })?;
    stream.set_nodelay(true).ok();

    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut writer = stream.try_clone()?;
    send_local_attach_json_line(
        &mut writer,
        &json!({
            "type":"attach",
            "code": code,
            "secret": endpoint.secret,
        }),
    )?;

    let mut first_line = String::new();
    let first_read = reader.read_line(&mut first_line)?;
    if first_read == 0 {
        anyhow::bail!("Local attach endpoint closed before handshake completed");
    }
    let first_value: Value = serde_json::from_str(first_line.trim())
        .map_err(|e| anyhow::anyhow!("invalid attach response: {}", e))?;
    run_local_attach_interactive(reader, writer, first_value, "Attached")
}

pub fn run_local_start_cli(
    agent: &str,
    cwd: Option<&Path>,
    agent_args: &[String],
) -> anyhow::Result<()> {
    let Some(agent) = normalize_agent(Some(agent)) else {
        anyhow::bail!(
            "Unknown agent `{}`. Supported: codex, claude, gemini, opencode.",
            agent
        );
    };
    let (agent_args, _) = crate::normalize_terminal_agent_permission_aliases(
        Some(agent.as_str()),
        agent_args.to_vec(),
    );

    let endpoint = load_local_attach_endpoint()?;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let stream = std::net::TcpStream::connect(&address).map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to local attach endpoint at {}: {}. Is aidaemon running?",
            address,
            e
        )
    })?;
    stream.set_nodelay(true).ok();

    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut writer = stream.try_clone()?;
    send_local_attach_json_line(
        &mut writer,
        &json!({
            "type":"start",
            "secret": endpoint.secret,
            "agent": agent,
            "cwd": cwd.map(|value| value.to_string_lossy().to_string()).unwrap_or_default(),
            "agent_args": agent_args,
        }),
    )?;

    let mut first_line = String::new();
    let first_read = reader.read_line(&mut first_line)?;
    if first_read == 0 {
        anyhow::bail!("Local attach endpoint closed before startup handshake completed");
    }
    let first_value: Value = serde_json::from_str(first_line.trim())
        .map_err(|e| anyhow::anyhow!("invalid local start response: {}", e))?;
    run_local_attach_interactive(reader, writer, first_value, "Started")
}

pub async fn request_local_start_session(
    agent: &str,
    cwd: Option<&str>,
    agent_args: &[String],
) -> anyhow::Result<String> {
    let Some(agent) = normalize_agent(Some(agent)) else {
        anyhow::bail!(
            "Unknown agent `{}`. Supported: codex, claude, gemini, opencode.",
            agent
        );
    };
    let (agent_args, _) = crate::normalize_terminal_agent_permission_aliases(
        Some(agent.as_str()),
        agent_args.to_vec(),
    );

    let endpoint = load_local_attach_endpoint()?;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let stream = TcpStream::connect(&address).await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to local attach endpoint at {}: {}. Is aidaemon running?",
            address,
            e
        )
    })?;
    stream.set_nodelay(true).ok();

    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    write_json_line(
        &mut write_half,
        &json!({
            "type":"start",
            "secret": endpoint.secret,
            "agent": agent,
            "cwd": cwd.unwrap_or(""),
            "agent_args": agent_args,
        }),
    )
    .await?;

    let mut line = String::new();
    let read = reader.read_line(&mut line).await?;
    if read == 0 {
        anyhow::bail!("Local attach endpoint closed before startup handshake completed");
    }
    let payload: Value = serde_json::from_str(line.trim())
        .map_err(|e| anyhow::anyhow!("invalid local start response: {}", e))?;
    match payload.get("type").and_then(|v| v.as_str()).unwrap_or("") {
        "attached" => payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string())
            .ok_or_else(|| anyhow::anyhow!("local start response missing session_id")),
        "error" => {
            let message = payload
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("local start failed");
            anyhow::bail!("{}", message);
        }
        _ => anyhow::bail!("Unexpected start response from local endpoint"),
    }
}

pub fn run_local_share_cli(session_id: Option<&str>) -> anyhow::Result<()> {
    let endpoint = load_local_attach_endpoint()?;
    let address = format!("{}:{}", endpoint.host, endpoint.port);
    let stream = std::net::TcpStream::connect(&address).map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to local attach endpoint at {}: {}. Is aidaemon running?",
            address,
            e
        )
    })?;
    stream.set_nodelay(true).ok();

    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut writer = stream.try_clone()?;
    let normalized_session_id = normalize_relay_session_id(session_id);
    send_local_attach_json_line(
        &mut writer,
        &json!({
            "type":"share",
            "secret": endpoint.secret,
            "session_id": normalized_session_id,
        }),
    )?;

    let mut line = String::new();
    let n = reader.read_line(&mut line)?;
    if n == 0 {
        anyhow::bail!("Local attach endpoint closed before share response");
    }
    let payload: Value = serde_json::from_str(line.trim())
        .map_err(|e| anyhow::anyhow!("invalid share response: {}", e))?;
    match payload.get("type").and_then(|v| v.as_str()).unwrap_or("") {
        "shared" => {
            let code = payload.get("code").and_then(|v| v.as_str()).unwrap_or("");
            if code.is_empty() {
                anyhow::bail!("Share response did not include a resume code");
            }
            let session_id = payload
                .get("session_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            println!("Session: {}", session_id);
            println!("Telegram command: /agent resume {}", code);
            if let Some(expires_at_unix) = payload.get("expires_at_unix").and_then(|v| v.as_i64()) {
                let now = chrono::Utc::now().timestamp();
                let secs_left = expires_at_unix.saturating_sub(now).max(0);
                println!("Expires in about {} minutes.", (secs_left + 59) / 60);
            } else {
                println!("Expires in about 5 minutes.");
            }
            Ok(())
        }
        "error" => {
            let message = payload
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("share failed");
            anyhow::bail!("{}", message);
        }
        _ => anyhow::bail!("Unexpected share response from local endpoint"),
    }
}

fn run_local_attach_interactive(
    mut reader: std::io::BufReader<std::net::TcpStream>,
    mut writer: std::net::TcpStream,
    first_value: Value,
    verb: &str,
) -> anyhow::Result<()> {
    match first_value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
    {
        "attached" => {}
        "error" => {
            let message = first_value
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("attach failed");
            anyhow::bail!("{}", message);
        }
        _ => anyhow::bail!("Unexpected attach response from local endpoint"),
    }

    let session_id = first_value
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    eprintln!("{} session {}. Press Ctrl+] to detach.", verb, session_id);
    if let Some(code) = first_value.get("resume_code").and_then(|v| v.as_str()) {
        if !code.trim().is_empty() {
            eprintln!("Telegram command: /agent resume {}", code.trim());
        }
    }

    if let Some((cols, rows)) = local_terminal_size() {
        let _ = send_local_attach_resize_and_redraw(&mut writer, cols, rows);
    } else {
        let _ = send_local_attach_json_line(&mut writer, &json!({"type":"redraw"}));
    }
    if let Ok(resize_writer) = writer.try_clone() {
        // Keep syncing local terminal dimensions while attached so full-screen TUIs
        // (Codex/Claude/Gemini/OpenCode) can reflow after handoff and on window resize.
        spawn_local_attach_resize_watcher(resize_writer);
    }

    let _raw_guard = RawTerminalGuard::enable_if_tty()?;
    let stdin_writer = writer.try_clone()?;
    std::thread::spawn(move || {
        let _ = run_local_attach_stdin_pump(stdin_writer);
    });

    run_local_attach_stdout_loop(&mut reader)
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
    fn test_merge_daemon_bot_tokens_includes_dynamic_telegram_bots_for_active_user() {
        let dynamic_bots = vec![
            crate::traits::DynamicBot {
                id: 1,
                channel_type: "telegram".to_string(),
                bot_token: "dynamic-user-token".to_string(),
                app_token: None,
                allowed_user_ids: vec!["301753035".to_string()],
                extra_config: String::new(),
                created_at: String::new(),
            },
            crate::traits::DynamicBot {
                id: 2,
                channel_type: "telegram".to_string(),
                bot_token: "other-user-token".to_string(),
                app_token: None,
                allowed_user_ids: vec!["999".to_string()],
                extra_config: String::new(),
                created_at: String::new(),
            },
            crate::traits::DynamicBot {
                id: 3,
                channel_type: "slack".to_string(),
                bot_token: "xoxb-not-telegram".to_string(),
                app_token: Some("xapp-ignored".to_string()),
                allowed_user_ids: vec!["301753035".to_string()],
                extra_config: String::new(),
                created_at: String::new(),
            },
        ];

        let merged = merge_daemon_bot_tokens(
            &[
                "configured-token".to_string(),
                "dynamic-user-token".to_string(),
            ],
            &dynamic_bots,
            "301753035",
        );

        assert_eq!(
            merged,
            vec![
                "configured-token".to_string(),
                "dynamic-user-token".to_string()
            ]
        );
    }

    #[test]
    fn test_outbound_queue_prioritizes_high_frames() {
        let mut queue = OutboundQueue::default();
        let _ = queue.enqueue(
            OutboundPriority::Low,
            QueuedOutboundPayload {
                session_id: "session-low-1".to_string(),
                payload: json!({"kind":"stdout","seq":1}),
            },
        );
        let _ = queue.enqueue(
            OutboundPriority::High,
            QueuedOutboundPayload {
                session_id: "session-high".to_string(),
                payload: json!({"kind":"review_result"}),
            },
        );
        let _ = queue.enqueue(
            OutboundPriority::Low,
            QueuedOutboundPayload {
                session_id: "session-low-2".to_string(),
                payload: json!({"kind":"stdout","seq":2}),
            },
        );

        let first = queue.pop_next().expect("first");
        let second = queue.pop_next().expect("second");
        let third = queue.pop_next().expect("third");

        assert_eq!(first.session_id, "session-high");
        assert_eq!(second.session_id, "session-low-1");
        assert_eq!(third.session_id, "session-low-2");
        assert!(queue.pop_next().is_none());
    }

    #[test]
    fn test_outbound_queue_bounds_low_lane() {
        let mut queue = OutboundQueue::default();
        for idx in 0..=OUTBOUND_LOW_QUEUE_CAP {
            let _ = queue.enqueue(
                OutboundPriority::Low,
                QueuedOutboundPayload {
                    session_id: format!("s{idx}"),
                    payload: json!({"kind":"stdout","seq": idx}),
                },
            );
        }

        assert_eq!(queue.low.len(), OUTBOUND_LOW_QUEUE_CAP);
        let first = queue.pop_next().expect("first");
        let last = queue.low.back().expect("last");
        assert_eq!(first.session_id, "s1");
        assert_eq!(last.session_id, format!("s{}", OUTBOUND_LOW_QUEUE_CAP));
    }

    #[test]
    fn test_next_reconnect_backoff_ms_doubles_until_cap() {
        assert_eq!(
            TerminalBridge::next_reconnect_backoff_ms(RECONNECT_INITIAL_MS),
            RECONNECT_INITIAL_MS * 2
        );
        assert_eq!(
            TerminalBridge::next_reconnect_backoff_ms(RECONNECT_MAX_MS / 2),
            RECONNECT_MAX_MS
        );
        assert_eq!(
            TerminalBridge::next_reconnect_backoff_ms(RECONNECT_MAX_MS),
            RECONNECT_MAX_MS
        );
    }

    #[test]
    fn test_next_reconnect_backoff_ms_normalizes_zero_input() {
        assert_eq!(
            TerminalBridge::next_reconnect_backoff_ms(0),
            RECONNECT_INITIAL_MS
        );
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
    fn test_decode_utf8_stream_chunk_preserves_split_multibyte_chars() {
        let mut carry = Vec::new();
        let part1 = decode_utf8_stream_chunk(&mut carry, &[0xE2, 0x94]);
        assert!(part1.is_empty());
        let part2 = decode_utf8_stream_chunk(&mut carry, &[0x80, b'\n']);
        assert_eq!(part2, "─\n");
        assert!(carry.is_empty());
    }

    #[test]
    fn test_decode_utf8_stream_chunk_normalizes_c1_csi() {
        let mut carry = Vec::new();
        let out = decode_utf8_stream_chunk(&mut carry, &[0x9B, b'3', b'1', b'm']);
        assert_eq!(out, "\u{001b}[31m");
        assert!(carry.is_empty());
    }

    #[test]
    fn test_should_skip_stdout_replay_when_gap_detected() {
        let frames = vec![(25, "partial output".to_string())];
        assert!(should_skip_stdout_replay(3, 25, &frames));
    }

    #[test]
    fn test_should_skip_stdout_replay_for_noisy_fresh_attach() {
        let frames = (1..=20)
            .map(|seq| (seq, format!("progress {}\r", seq)))
            .collect::<Vec<_>>();
        assert!(should_skip_stdout_replay(0, 1, &frames));
    }

    #[test]
    fn test_should_not_skip_stdout_replay_for_small_clean_resume() {
        let frames = vec![(8, "line one\n".to_string()), (9, "line two\n".to_string())];
        assert!(!should_skip_stdout_replay(7, 8, &frames));
    }

    #[test]
    fn test_build_skipped_stdout_replay_status_message_omits_clean_resume_noise() {
        let frames = vec![(8, "line one\n".to_string()), (9, "line two\n".to_string())];
        assert_eq!(
            build_skipped_stdout_replay_status_message(7, 8, &frames),
            None
        );
        assert_eq!(build_skipped_stdout_replay_status_message(7, 8, &[]), None);
    }

    #[test]
    fn test_build_skipped_stdout_replay_status_message_mentions_recovery_only_when_skipped() {
        let frames = (1..=20)
            .map(|seq| (seq, format!("progress {}\r", seq)))
            .collect::<Vec<_>>();
        let message = build_skipped_stdout_replay_status_message(0, 1, &frames)
            .expect("skipped replay status");
        assert!(message.contains("Connection recovered."));
        assert!(message.contains("Skipped replaying 20 buffered output frame(s)"));
        assert!(message.contains("If the terminal appears blank"));
    }

    #[test]
    fn test_build_connection_notice_payload_marks_out_of_band_notice() {
        let payload = build_connection_notice_payload(
            "Connection recovered.".to_string(),
            "warn",
            "terminal",
            5200,
        );
        assert_eq!(payload["kind"].as_str(), Some("connection_notice"));
        assert_eq!(payload["tone"].as_str(), Some("warn"));
        assert_eq!(payload["scope"].as_str(), Some("terminal"));
        assert_eq!(payload["ttl_ms"].as_u64(), Some(5200));
    }

    #[test]
    fn test_should_skip_review_stream_replay_when_gap_detected() {
        assert!(should_skip_review_stream_replay(4, 20, 3));
    }

    #[test]
    fn test_should_not_skip_review_stream_replay_for_small_resume() {
        assert!(!should_skip_review_stream_replay(12, 13, 4));
    }

    #[test]
    fn test_build_skipped_review_stream_replay_status_message_omits_clean_resume_noise() {
        assert_eq!(
            build_skipped_review_stream_replay_status_message(12, 13, 4),
            None
        );
        assert_eq!(
            build_skipped_review_stream_replay_status_message(12, 13, 0),
            None
        );
    }

    #[test]
    fn test_build_skipped_review_stream_replay_status_message_mentions_recovery_only_when_skipped()
    {
        let message = build_skipped_review_stream_replay_status_message(4, 20, 3)
            .expect("skipped review replay status");
        assert!(message.contains("Connection recovered."));
        assert!(message.contains("Skipped replaying 3 buffered review stream chunk(s)"));
    }

    #[test]
    fn test_normalize_agent_args_drops_shell_metacharacters() {
        let payload = serde_json::json!(["--model", "gpt-5", "foo;bar", "$(whoami)", "--json"]);
        let args = normalize_agent_args(Some(&payload), Some("codex"));
        assert_eq!(args, vec!["--model", "gpt-5", "--json"]);
    }

    #[test]
    fn test_normalize_agent_args_splits_single_string_payload() {
        let payload = serde_json::json!("--model gpt-5 --json");
        let args = normalize_agent_args(Some(&payload), Some("codex"));
        assert_eq!(args, vec!["--model", "gpt-5", "--json"]);
    }

    #[test]
    fn test_requested_agent_args_from_client_hello_rewrites_claude_allow_alias() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "agent": "claude",
            "agent_args": ["--allow-dangerously-skip-permissions", "--output-format", "json"],
        });
        let args = requested_agent_args_from_client_hello(&frame);
        assert_eq!(
            args,
            vec!["--dangerously-skip-permissions", "--output-format", "json"]
        );
    }

    #[test]
    fn test_shell_token_keeps_plain_flags_unquoted() {
        assert_eq!(shell_token("codex"), "codex");
        assert_eq!(
            shell_token("--dangerously-bypass-approvals-and-sandbox"),
            "--dangerously-bypass-approvals-and-sandbox"
        );
    }

    #[test]
    fn test_shell_token_quotes_tokens_with_whitespace_or_quotes() {
        assert_eq!(shell_token("--model gpt-5"), "'--model gpt-5'");
        assert_eq!(shell_token("it's"), "'it'\\''s'");
    }

    #[test]
    fn test_requested_agent_args_from_client_hello_accepts_legacy_args_key() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "agent": "codex",
            "args": ["--model", "gpt-5", "bad;arg"],
        });
        let args = requested_agent_args_from_client_hello(&frame);
        assert_eq!(args, vec!["--model", "gpt-5"]);
    }

    #[test]
    fn test_requested_agent_args_from_client_hello_accepts_camel_case_key() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "agent": "codex",
            "agentArgs": ["--json"],
        });
        let args = requested_agent_args_from_client_hello(&frame);
        assert_eq!(args, vec!["--json"]);
    }

    #[test]
    fn test_requested_telegram_session_id_from_client_hello_accepts_snake_case() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "telegram_session_id": "telegrambot:12345",
        });
        let value = requested_telegram_session_id_from_client_hello(&frame);
        assert_eq!(value.as_deref(), Some("telegrambot:12345"));
    }

    #[test]
    fn test_requested_telegram_session_id_from_client_hello_accepts_camel_case() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "telegramSessionId": "telegrambot:999",
        });
        let value = requested_telegram_session_id_from_client_hello(&frame);
        assert_eq!(value.as_deref(), Some("telegrambot:999"));
    }

    #[test]
    fn test_requested_telegram_session_id_from_client_hello_rejects_invalid_chars() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "telegram_session_id": "telegrambot:12345/../../etc/passwd",
        });
        assert!(requested_telegram_session_id_from_client_hello(&frame).is_none());
    }

    #[test]
    fn test_requested_relay_session_id_from_client_hello_accepts_snake_case() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "relay_session_id": "native-1234abcd",
        });
        let value = requested_relay_session_id_from_client_hello(&frame);
        assert_eq!(value.as_deref(), Some("native-1234abcd"));
    }

    #[test]
    fn test_requested_relay_session_id_from_client_hello_accepts_camel_case() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "relaySessionId": "native-5678efgh",
        });
        let value = requested_relay_session_id_from_client_hello(&frame);
        assert_eq!(value.as_deref(), Some("native-5678efgh"));
    }

    #[test]
    fn test_requested_relay_session_id_from_client_hello_accepts_nested_relay() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "relay": {
                "session_id": "native-zxyw9876"
            },
        });
        let value = requested_relay_session_id_from_client_hello(&frame);
        assert_eq!(value.as_deref(), Some("native-zxyw9876"));
    }

    #[test]
    fn test_requested_relay_session_id_from_client_hello_rejects_invalid_chars() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "relay_session_id": "native-1234/../../etc/passwd",
        });
        assert!(requested_relay_session_id_from_client_hello(&frame).is_none());
    }

    #[test]
    fn test_requested_fresh_start_from_client_hello_accepts_boolean_flag() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "fresh_start": true,
        });
        assert!(requested_fresh_start_from_client_hello(&frame));
    }

    #[test]
    fn test_requested_fresh_start_from_client_hello_accepts_string_flag() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "freshStart": "yes",
        });
        assert!(requested_fresh_start_from_client_hello(&frame));
    }

    #[test]
    fn test_requested_fresh_start_from_client_hello_accepts_nested_launch_flag() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "launch": {
                "force_fresh_start": 1
            }
        });
        assert!(requested_fresh_start_from_client_hello(&frame));
    }

    #[test]
    fn test_requested_fresh_start_from_client_hello_defaults_false() {
        let frame = serde_json::json!({
            "type": "e2ee_client_hello",
            "fresh_start": "maybe",
        });
        assert!(!requested_fresh_start_from_client_hello(&frame));
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
