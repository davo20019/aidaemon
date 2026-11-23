//! The terminal bridge: runs CLI coding agents (codex, claude, gemini,
//! opencode) in local PTYs and relays them, end-to-end encrypted, over a
//! daemon websocket to the owner's Telegram `/agent` sessions. Also the local
//! attach/share/start CLIs (`aidaemon codex`, `aidaemon attach`, `aidaemon
//! share`) that talk to a running bridge.
//!
//! The host passes [`BridgeSettings`], a [`BridgeHost`] (runtime bots and the
//! relay handoff records) and the [`SecretStore`] holding the bridge key to
//! [`spawn_if_configured`].

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
use p256::elliptic_curve::Generate;
use p256::elliptic_curve::sec1::ToSec1Point;
use p256::{PublicKey, SecretKey};
use portable_pty::{CommandBuilder, ExitStatus, MasterPty, NativePtySystem, PtySize, PtySystem};
use rand::rand_core::UnwrapErr;
use rand::rngs::SysRng;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio_tungstenite::{Connector, connect_async_tls_with_config};
use tracing::{debug, error, info, warn};

use aidaemon_secrets::SecretStore;
use std::sync::Arc;

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

mod bridge;
mod connect_token;
mod hello;
mod keys;
mod local_attach;
mod local_cli;
mod outbound;
mod replay;
mod review;
mod session;
mod settings;
mod shell;
mod stdin;
mod uploads;

use bridge::*;
use connect_token::*;
use hello::*;
use keys::*;
use local_attach::*;
pub use local_cli::*;
use outbound::*;
use replay::*;
use review::*;
use session::*;
pub use settings::*;
use shell::*;
use stdin::*;
use uploads::*;

#[cfg(test)]
// `Nonce::from_slice` (re-exported via aes-gcm's generic-array 0.x) is
// deprecated pending the generic-array 1.x upgrade, which is a dependency-tree
// change out of scope here. Allow it for these crypto round-trip tests only.
#[allow(deprecated)]
mod tests;
