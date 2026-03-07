use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::SqlitePool;
use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::channels::ChannelHub;
use crate::events::{
    ApprovalDeniedData, ApprovalGrantedData, ApprovalRequestedData, EventStore, EventType,
};
use crate::traits::{StateStore, Tool, ToolCallMetadata, ToolCallOutcome, ToolCapabilities};
use crate::types::{ApprovalResponse, StatusUpdate};
use crate::utils::{truncate_str, truncate_with_note};

use super::command_patterns::{find_matching_pattern, record_approval, record_denial};
use super::command_risk::{classify_command, hard_block_reason, PermissionMode, RiskLevel};
use super::daemon_guard::detect_daemonization_primitives;
use super::process_control::{configure_command_for_process_group, send_sigkill, send_sigterm};

/// Max bytes per stream buffer (1 MB) to prevent unbounded memory growth.
const BUFFER_CAP: usize = 1_048_576;
#[cfg(test)]
const BACKGROUND_PROGRESS_INTERVAL_SECS: u64 = 1;
#[cfg(not(test))]
const BACKGROUND_PROGRESS_INTERVAL_SECS: u64 = 35;
/// Maximum number of periodic progress pings before going silent.
/// Prevents notification spam for long-running processes (servers, daemons).
const MAX_BACKGROUND_PROGRESS_PINGS: u32 = 3;

/// A request sent to the ChannelHub for command approval.
pub struct ApprovalRequest {
    pub command: String,
    pub session_id: String,
    pub risk_level: RiskLevel,
    pub warnings: Vec<String>,
    pub permission_mode: PermissionMode,
    pub response_tx: tokio::sync::oneshot::Sender<ApprovalResponse>,
    /// What kind of approval this is (command vs goal confirmation).
    pub kind: crate::types::ApprovalKind,
}

/// A background process being tracked after it exceeded the initial timeout.
///
/// Process lifecycle modes:
/// 1. **Task-owned** (`detached=false`, `notifier_active=false`): killed on task-end.
/// 2. **Background with notifier** (`detached=false`, `notifier_active=true`): survives
///    task-end so the notifier can deliver the result. Killed when the notifier finishes.
/// 3. **Detached** (`detached=true`): survives task-end and notifier. Requires explicit kill.
struct RunningProcess {
    command: String,
    dedupe_key: Option<String>,
    owner_task_id: Option<String>,
    detached: bool,
    started_at: Instant,
    stdout_buf: Arc<Mutex<Vec<u8>>>,
    stderr_buf: Arc<Mutex<Vec<u8>>>,
    reader_handle: JoinHandle<Option<i32>>,
    child_id: u32,
    notify_on_completion: Arc<AtomicBool>,
    /// True only when the background notifier tokio task was actually spawned
    /// and is actively monitoring this process for completion/progress delivery.
    /// Used by `cleanup_task_processes` to decide whether to kill or disown.
    notifier_active: bool,
}

/// Finalized background process output retained briefly so `action="check"`
/// can still return results after automatic reaping.
struct CompletedProcess {
    output: String,
    metadata: ToolCallMetadata,
    completed_at: Instant,
}

pub struct TerminalTool {
    /// Permanently allowed prefixes (from config + DB)
    allowed_prefixes: Arc<RwLock<Vec<String>>>,
    /// Session-only allowed prefixes (cleared on restart)
    session_approved: Arc<RwLock<HashSet<String>>>,
    /// Permission persistence mode
    permission_mode: PermissionMode,
    approval_tx: mpsc::Sender<ApprovalRequest>,
    running: Arc<Mutex<HashMap<u32, RunningProcess>>>,
    running_by_dedupe_key: Arc<Mutex<HashMap<String, u32>>>,
    task_processes: Arc<Mutex<HashMap<String, HashSet<u32>>>>,
    completed: Arc<Mutex<HashMap<u32, CompletedProcess>>>,
    initial_timeout: Duration,
    max_output_chars: usize,
    pool: Option<SqlitePool>,
    event_store: Option<Arc<EventStore>>,
    state: Option<Arc<dyn StateStore>>,
    hub: OnceLock<Weak<ChannelHub>>,
}

/// Check if a command string contains shell operators.
/// Used for prefix matching - we don't allow prefix matches for commands with operators
/// since "cargo" shouldn't match "cargo test | bash".
fn contains_shell_operator(cmd: &str) -> bool {
    for ch in [';', '|', '`', '\n'] {
        if cmd.contains(ch) {
            return true;
        }
    }
    for op in ["&&", "||", "$(", ">(", "<("] {
        if cmd.contains(op) {
            return true;
        }
    }
    false
}

/// Split a chained command into individual segments by pipe, semicolon, &&, ||.
/// Used by session-approval to extract per-segment binary names.
fn split_command_segments(cmd: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let bytes = cmd.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        match bytes[i] {
            b'|' if i + 1 < len && bytes[i + 1] == b'|' => {
                segments.push(&cmd[start..i]);
                i += 2;
                start = i;
            }
            b'|' => {
                segments.push(&cmd[start..i]);
                i += 1;
                start = i;
            }
            b'&' if i + 1 < len && bytes[i + 1] == b'&' => {
                segments.push(&cmd[start..i]);
                i += 2;
                start = i;
            }
            b';' => {
                segments.push(&cmd[start..i]);
                i += 1;
                start = i;
            }
            _ => {
                i += 1;
            }
        }
    }
    if start < len {
        segments.push(&cmd[start..]);
    }
    segments
        .into_iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extract the binary/command name from a single command segment.
/// Handles variable assignments like `VAR=val cmd args...` by skipping
/// assignment tokens and returning the first non-assignment word.
fn extract_segment_binary(segment: &str) -> &str {
    for word in segment.split_whitespace() {
        // Skip shell variable assignments (e.g., EPOCH=$(date ...))
        if word.contains('=') {
            continue;
        }
        return word;
    }
    ""
}

fn is_grep_command(token: &str) -> bool {
    std::path::Path::new(token)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "grep")
}

fn grep_has_recursive_flag(token: &str) -> bool {
    if matches!(token, "--recursive" | "--dereference-recursive") {
        return true;
    }
    if token.starts_with("--") {
        return false;
    }
    token
        .strip_prefix('-')
        .is_some_and(|flags| flags.chars().any(|c| c == 'r' || c == 'R'))
}

fn has_recursive_grep_scope_controls(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains("--exclude-dir")
        || lower.contains("--exclude=")
        || lower.contains("--exclude ")
        || lower.contains("--include")
        || lower.contains("-d skip")
        || lower.contains("-dskip")
}

/// Detect `python3 -c "..."` commands that perform file I/O.
/// These should use read_file/write_file tools instead of terminal.
fn is_python_c_with_file_io(command: &str) -> bool {
    // Split by shell operators to check each segment
    let lower = command.to_ascii_lowercase();

    // Quick pre-check: must contain python and -c
    if !lower.contains("python") || !lower.contains("-c") {
        return false;
    }

    // Parse the command properly to extract the -c argument
    let parts = match shell_words::split(command) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // Find python/python3 followed by -c
    let mut i = 0;
    while i < parts.len() {
        let base = std::path::Path::new(&parts[i])
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&parts[i]);

        if matches!(base, "python" | "python3") {
            // Look for -c flag in subsequent args
            for j in (i + 1)..parts.len() {
                if parts[j] == "-c" {
                    // The code string is the next argument (or concatenated)
                    let code = if j + 1 < parts.len() {
                        parts[j + 1].to_ascii_lowercase()
                    } else {
                        String::new()
                    };
                    let file_io_patterns = [
                        "open(",
                        "with open",
                        ".read(",
                        ".write(",
                        ".readlines(",
                        ".writelines(",
                        "read_text(",
                        "write_text(",
                        "json.load",
                        "json.dump",
                        "os.walk",
                        "os.listdir",
                    ];
                    if file_io_patterns.iter().any(|p| code.contains(p)) {
                        return true;
                    }
                    break;
                }
            }
        }
        i += 1;
    }

    false
}

fn detect_unscoped_recursive_grep_segment(segment: &str) -> Option<(String, String)> {
    let tokens = shell_words::split(segment).ok()?;
    let first = tokens.first()?;
    if !is_grep_command(first) {
        return None;
    }

    let recursive = tokens
        .iter()
        .skip(1)
        .any(|tok| grep_has_recursive_flag(tok));
    if !recursive || has_recursive_grep_scope_controls(segment) {
        return None;
    }

    // grep syntax: grep [OPTIONS] PATTERN [FILE...]
    // We use a lightweight parse here: non-option tokens are treated as
    // positional args; first positional = pattern, remaining = target paths.
    let positionals: Vec<String> = tokens
        .iter()
        .skip(1)
        .filter(|tok| !tok.starts_with('-'))
        .cloned()
        .collect();
    let pattern = positionals.first()?.clone();
    let paths = if positionals.len() >= 2 {
        positionals[1..].to_vec()
    } else {
        vec![".".to_string()]
    };
    let broad_scope = paths
        .iter()
        .any(|p| matches!(p.as_str(), "." | "./" | "/" | "~" | "~/"));
    if !broad_scope {
        return None;
    }

    Some((pattern, paths.join(" ")))
}

fn detect_unscoped_recursive_grep(command: &str) -> Option<(String, String)> {
    if let Some(hit) = detect_unscoped_recursive_grep_segment(command) {
        return Some(hit);
    }

    // Also scan chained shell segments (e.g. "cd repo && grep -rc ... .").
    // This is intentionally simple and best-effort: it catches common cases
    // without trying to fully parse shell grammar.
    static SHELL_CHAIN_SPLIT_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?:&&|\|\||;|\|)").expect("valid chain regex"));
    for segment in SHELL_CHAIN_SPLIT_RE.split(command) {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(hit) = detect_unscoped_recursive_grep_segment(trimmed) {
            return Some(hit);
        }
    }

    None
}

fn recursive_grep_block_message(pattern: &str, path: &str) -> String {
    let ignore_globs = super::fs_utils::DEFAULT_IGNORE_DIRS.join(",");
    format!(
        "Blocked: broad recursive `grep` without include/exclude filters is likely to stall on large trees.\n\
Detected pattern: \"{}\"\n\
Detected path: {}\n\n\
Use one of these instead:\n\
- `search_files` (preferred) with explicit `path`, optional `glob`, and regex `pattern`\n\
- Terminal `rg` with exclusions:\n\
  `rg -n --glob '!{{{}}}' \"<pattern>\" <path>`\n\
- If you must use grep, add `--exclude-dir` and/or `--include` so the scan is bounded.",
        pattern, path, ignore_globs
    )
}

fn normalize_command_for_dedupe(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Drain an async reader into a capped buffer.
async fn drain_to_buffer<R: tokio::io::AsyncRead + Unpin>(mut reader: R, buf: Arc<Mutex<Vec<u8>>>) {
    let mut tmp = [0u8; 8192];
    loop {
        match reader.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => {
                let mut b = buf.lock().await;
                let remaining = BUFFER_CAP.saturating_sub(b.len());
                if remaining > 0 {
                    let to_copy = n.min(remaining);
                    b.extend_from_slice(&tmp[..to_copy]);
                }
            }
            Err(_) => break,
        }
    }
}

/// Format combined stdout/stderr output with optional truncation.
fn format_output(stdout: &str, stderr: &str, max_chars: usize) -> String {
    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n--- stderr ---\n");
        }
        result.push_str(stderr);
    }
    if result.is_empty() {
        result.push_str("(no output)");
    }
    if result.len() > max_chars {
        // Find the nearest valid UTF-8 char boundary at or before max_chars
        let mut truncate_at = max_chars;
        while truncate_at > 0 && !result.is_char_boundary(truncate_at) {
            truncate_at -= 1;
        }
        result.truncate(truncate_at);
        result.push_str("\n... (truncated)");
    }
    result
}

impl TerminalTool {
    pub async fn new(
        allowed_prefixes: Vec<String>,
        approval_tx: mpsc::Sender<ApprovalRequest>,
        initial_timeout_secs: u64,
        max_output_chars: usize,
        permission_mode: PermissionMode,
        pool: SqlitePool,
    ) -> Self {
        // Log permission mode on startup
        match permission_mode {
            PermissionMode::Yolo => {
                warn!("⚠️  YOLO mode enabled: all command approvals persist forever, including critical commands");
            }
            PermissionMode::Cautious => {
                info!("Cautious mode: all command approvals are session-only");
            }
            PermissionMode::Default => {
                info!("Default permission mode: critical commands require per-session approval");
            }
        }

        // Load persisted prefixes from DB and merge with config defaults
        let mut merged = allowed_prefixes;

        // YOLO mode: auto-approve everything
        if permission_mode == PermissionMode::Yolo && !merged.contains(&"*".to_string()) {
            merged.push("*".to_string());
        }
        match sqlx::query_scalar::<_, String>("SELECT prefix FROM terminal_allowed_prefixes")
            .fetch_all(&pool)
            .await
        {
            Ok(persisted) => {
                for p in persisted {
                    if !merged.contains(&p) {
                        info!(prefix = %p, "Loaded persisted allowed prefix");
                        merged.push(p);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to load persisted terminal prefixes: {}", e);
            }
        }

        Self {
            allowed_prefixes: Arc::new(RwLock::new(merged)),
            session_approved: Arc::new(RwLock::new(HashSet::new())),
            permission_mode,
            approval_tx,
            running: Arc::new(Mutex::new(HashMap::new())),
            running_by_dedupe_key: Arc::new(Mutex::new(HashMap::new())),
            task_processes: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            initial_timeout: Duration::from_secs(initial_timeout_secs),
            max_output_chars,
            pool: Some(pool),
            event_store: None,
            state: None,
            hub: OnceLock::new(),
        }
    }

    pub fn with_event_store(mut self, event_store: Arc<EventStore>) -> Self {
        self.event_store = Some(event_store);
        self
    }

    pub fn with_state(mut self, state: Arc<dyn StateStore>) -> Self {
        self.state = Some(state);
        self
    }

    /// Set channel hub reference for immediate background progress/completion delivery.
    pub fn set_hub(&self, hub: Weak<ChannelHub>) {
        let _ = self.hub.set(hub);
    }

    fn get_hub(&self) -> Option<Arc<ChannelHub>> {
        self.hub.get().and_then(|w| w.upgrade())
    }

    async fn is_allowed(&self, command: &str) -> bool {
        let prefixes = self.allowed_prefixes.read().await;
        if prefixes.iter().any(|p| p == "*") {
            return true;
        }
        let trimmed = command.trim();
        let has_shell_ops = contains_shell_operator(trimmed);

        // For chained commands (&&, ||, ;, |), check each segment's binary
        // against both permanent and session-approved prefixes.
        // This means approving `curl ... | python3 ...` also allows
        // `curl ... | grep ...` since both curl and grep are safe/approved.
        if has_shell_ops {
            let session = self.session_approved.read().await;
            // First check for exact full-command match (legacy behavior)
            if session.iter().any(|s| trimmed == s.as_str()) {
                return true;
            }
            // Then check per-segment: every segment's binary must be approved
            let segments = split_command_segments(trimmed);
            if !segments.is_empty() {
                return segments.iter().all(|seg| {
                    let binary = extract_segment_binary(seg);
                    if binary.is_empty() {
                        return true;
                    }
                    prefixes.iter().any(|p| p == "*" || binary == p.as_str())
                        || session.iter().any(|p| p == "*" || binary == p.as_str())
                });
            }
            return false;
        }

        // Check permanent prefixes
        let matches_permanent = prefixes.iter().any(|prefix| {
            trimmed == prefix.as_str()
                || trimmed.starts_with(&format!("{} ", prefix))
                || trimmed.starts_with(&format!("{}\t", prefix))
        });

        if matches_permanent {
            return true;
        }

        // Check session-approved prefixes
        let session = self.session_approved.read().await;
        session.iter().any(|prefix| {
            trimmed == prefix.as_str()
                || trimmed.starts_with(&format!("{} ", prefix))
                || trimmed.starts_with(&format!("{}\t", prefix))
        })
    }

    /// Add a prefix to session-only approved list (cleared on restart).
    /// For chained commands (containing shell operators), extracts the binary
    /// name from each segment and stores each as a session-approved prefix.
    /// This means approving `curl ... | python3 ... | head ...` will also
    /// allow future commands like `curl ... | grep ... | head ...`.
    /// For simple commands, stores the first word as prefix.
    async fn add_session_prefix(&self, command: &str) {
        let trimmed = command.trim();
        let mut session = self.session_approved.write().await;
        if contains_shell_operator(trimmed) {
            for seg in split_command_segments(trimmed) {
                let binary = extract_segment_binary(seg);
                if !binary.is_empty() && session.insert(binary.to_string()) {
                    info!(
                        prefix = %binary,
                        "Session-approved prefix from chained command segment"
                    );
                }
            }
        } else {
            let key = trimmed
                .split_whitespace()
                .next()
                .unwrap_or(trimmed)
                .to_string();
            if session.insert(key.clone()) {
                info!(
                    prefix = %key,
                    "Added to session-approved prefixes (will reset on restart)"
                );
            }
        }
    }

    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: Vec<String>,
        task_id: Option<&str>,
    ) -> anyhow::Result<ApprovalResponse> {
        if let Some(store) = &self.event_store {
            let emitter = crate::events::EventEmitter::new(store.clone(), session_id.to_string());
            let _ = emitter
                .emit(
                    EventType::ApprovalRequested,
                    ApprovalRequestedData {
                        command: command.to_string(),
                        risk_level: risk_level.to_string(),
                        warnings: warnings.clone(),
                        task_id: task_id.map(str::to_string),
                    },
                )
                .await;
        }

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        if let Err(send_err) = self
            .approval_tx
            .send(ApprovalRequest {
                command: command.to_string(),
                session_id: session_id.to_string(),
                risk_level,
                warnings,
                permission_mode: self.permission_mode,
                response_tx,
                kind: Default::default(),
            })
            .await
        {
            if let Some(store) = &self.event_store {
                let emitter =
                    crate::events::EventEmitter::new(store.clone(), session_id.to_string());
                let _ = emitter
                    .emit(
                        EventType::ApprovalDenied,
                        ApprovalDeniedData {
                            command: command.to_string(),
                            task_id: task_id.map(str::to_string),
                        },
                    )
                    .await;
            }
            return Err(anyhow::anyhow!("Approval channel closed: {}", send_err));
        }

        // Sub-agents (session IDs starting with "sub-") get a short timeout
        // since they can't reliably receive user approval through the channel hub.
        // They should use safe tools (edit_file, write_file) instead of risky terminal commands.
        let timeout_secs = if session_id.starts_with("sub-") {
            10
        } else {
            300
        };
        let response: ApprovalResponse =
            match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), response_rx)
                .await
            {
                Ok(Ok(response)) => response,
                Ok(Err(_)) => {
                    tracing::warn!(command, "Approval response channel closed");
                    ApprovalResponse::Deny
                }
                Err(_) => {
                    tracing::warn!(
                        command,
                        timeout_secs,
                        "Approval request timed out, auto-denying"
                    );
                    ApprovalResponse::Deny
                }
            };

        if let Some(store) = &self.event_store {
            let emitter = crate::events::EventEmitter::new(store.clone(), session_id.to_string());
            match response {
                ApprovalResponse::AllowOnce => {
                    let _ = emitter
                        .emit(
                            EventType::ApprovalGranted,
                            ApprovalGrantedData {
                                command: command.to_string(),
                                approval_type: "once".to_string(),
                                task_id: task_id.map(str::to_string),
                            },
                        )
                        .await;
                }
                ApprovalResponse::AllowSession => {
                    let _ = emitter
                        .emit(
                            EventType::ApprovalGranted,
                            ApprovalGrantedData {
                                command: command.to_string(),
                                approval_type: "session".to_string(),
                                task_id: task_id.map(str::to_string),
                            },
                        )
                        .await;
                }
                ApprovalResponse::AllowAlways => {
                    let _ = emitter
                        .emit(
                            EventType::ApprovalGranted,
                            ApprovalGrantedData {
                                command: command.to_string(),
                                approval_type: "always".to_string(),
                                task_id: task_id.map(str::to_string),
                            },
                        )
                        .await;
                }
                ApprovalResponse::Deny => {
                    let _ = emitter
                        .emit(
                            EventType::ApprovalDenied,
                            ApprovalDeniedData {
                                command: command.to_string(),
                                task_id: task_id.map(str::to_string),
                            },
                        )
                        .await;
                }
            }
        }

        Ok(response)
    }

    async fn add_prefix(&self, command: &str) {
        let trimmed = command.trim();
        // For chained commands, store full command; for simple commands, store first word
        let key = if contains_shell_operator(trimmed) {
            trimmed.to_string()
        } else {
            command
                .split_whitespace()
                .next()
                .unwrap_or(trimmed)
                .to_string()
        };
        if key == "*" {
            warn!("Refusing to add wildcard '*' as permanent prefix");
            return;
        }
        let mut prefixes = self.allowed_prefixes.write().await;
        if !prefixes.contains(&key) {
            info!(prefix = %key, "Adding to allowed command prefixes (persistent)");
            prefixes.push(key.clone());

            // Persist to SQLite
            if let Some(ref pool) = self.pool {
                if let Err(e) = sqlx::query(
                    "INSERT OR IGNORE INTO terminal_allowed_prefixes (prefix) VALUES (?)",
                )
                .bind(&key)
                .execute(pool)
                .await
                {
                    warn!(prefix = %key, "Failed to persist allowed prefix: {}", e);
                }
            }
        }
    }

    fn dedupe_scope_key(
        notify_session_id: &str,
        notify_goal_id: Option<&str>,
        task_id: Option<&str>,
    ) -> String {
        if let Some(goal_id) = notify_goal_id.filter(|value| !value.trim().is_empty()) {
            return format!("goal:{}", goal_id.trim());
        }
        if let Some(task_id) = task_id.filter(|value| !value.trim().is_empty()) {
            return format!("task:{}", task_id.trim());
        }
        format!("session:{}", notify_session_id.trim())
    }

    fn dedupe_key_for_run(
        command: &str,
        notify_session_id: &str,
        notify_goal_id: Option<&str>,
        task_id: Option<&str>,
    ) -> String {
        let scope = Self::dedupe_scope_key(notify_session_id, notify_goal_id, task_id);
        let normalized = normalize_command_for_dedupe(command);
        format!("{}|{}", scope, normalized)
    }

    async fn insert_indexes_for_process(
        &self,
        pid: u32,
        dedupe_key: Option<&str>,
        owner_task_id: Option<&str>,
        detached: bool,
    ) {
        if let Some(key) = dedupe_key {
            self.running_by_dedupe_key
                .lock()
                .await
                .insert(key.to_string(), pid);
        }

        if !detached {
            if let Some(task_id) = owner_task_id {
                let mut task_map = self.task_processes.lock().await;
                task_map.entry(task_id.to_string()).or_default().insert(pid);
            }
        }
    }

    async fn remove_indexes_for_process(&self, pid: u32, proc: &RunningProcess) {
        if let Some(key) = proc.dedupe_key.as_ref() {
            let mut dedupe = self.running_by_dedupe_key.lock().await;
            if dedupe.get(key).copied() == Some(pid) {
                dedupe.remove(key);
            }
        }

        if !proc.detached {
            if let Some(task_id) = proc.owner_task_id.as_ref() {
                let mut task_map = self.task_processes.lock().await;
                let mut remove_task_key = false;
                if let Some(pids) = task_map.get_mut(task_id) {
                    pids.remove(&pid);
                    remove_task_key = pids.is_empty();
                }
                if remove_task_key {
                    task_map.remove(task_id);
                }
            }
        }
    }

    async fn resolve_duplicate_running_pid(&self, dedupe_key: &str) -> Option<u32> {
        let tracked_pid = {
            let dedupe = self.running_by_dedupe_key.lock().await;
            dedupe.get(dedupe_key).copied()
        }?;

        let is_live = {
            let running = self.running.lock().await;
            running
                .get(&tracked_pid)
                .is_some_and(|proc| !proc.reader_handle.is_finished())
        };
        if is_live {
            return Some(tracked_pid);
        }

        // Stale index entry from a process that's already finished/reaped.
        let mut dedupe = self.running_by_dedupe_key.lock().await;
        if dedupe.get(dedupe_key).copied() == Some(tracked_pid) {
            dedupe.remove(dedupe_key);
        }
        None
    }

    async fn terminate_running_process(
        &self,
        pid: u32,
        proc: RunningProcess,
        reason: &str,
    ) -> anyhow::Result<String> {
        proc.notify_on_completion.store(false, Ordering::Relaxed);
        let child_pid = proc.child_id;
        let started_at = proc.started_at;
        let command = proc.command.clone();
        let stdout_buf = proc.stdout_buf.clone();
        let stderr_buf = proc.stderr_buf.clone();
        let reader_handle = proc.reader_handle;

        if !reader_handle.is_finished() {
            let term_sent = send_sigterm(child_pid);
            if term_sent {
                let finished = tokio::time::timeout(Duration::from_secs(2), async {
                    loop {
                        if reader_handle.is_finished() {
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                })
                .await;

                if finished.is_err() && !reader_handle.is_finished() {
                    send_sigkill(child_pid);
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            } else {
                send_sigkill(child_pid);
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }

        if !reader_handle.is_finished() {
            reader_handle.abort();
        }
        let _ = reader_handle.await;

        let stdout = String::from_utf8_lossy(&stdout_buf.lock().await).to_string();
        let stderr = String::from_utf8_lossy(&stderr_buf.lock().await).to_string();
        let mut output = format!(
            "[Process pid={} stopped after {:.0}s (reason: {}, command: `{}`)]\n",
            pid,
            started_at.elapsed().as_secs_f64(),
            reason,
            command
        );
        output.push_str(&format_output(&stdout, &stderr, self.max_output_chars));
        Ok(output)
    }

    async fn cleanup_task_processes(&self, task_id: &str) -> anyhow::Result<usize> {
        self.reap_finished().await;
        let cleaned_pids = {
            let mut task_map = self.task_processes.lock().await;
            task_map.remove(task_id).unwrap_or_default()
        };
        if cleaned_pids.is_empty() {
            return Ok(0);
        }

        let mut to_cleanup = Vec::new();
        let mut to_disown = Vec::new();
        {
            let mut running = self.running.lock().await;
            for pid in cleaned_pids {
                if let Some(proc) = running.remove(&pid) {
                    // If the background notifier task was actually spawned and is actively
                    // monitoring this process, the user was promised completion notifications.
                    // Don't kill it — just disown it from the task and let the notifier
                    // handle delivery when the process finishes naturally.
                    if proc.notifier_active {
                        to_disown.push((pid, proc));
                    } else {
                        to_cleanup.push((pid, proc));
                    }
                }
            }
        }

        // Re-insert disowned processes so the notifier can still track them.
        // Clear owner_task_id so `check` no longer reports them as task-owned.
        if !to_disown.is_empty() {
            let mut running = self.running.lock().await;
            for (pid, mut proc) in to_disown {
                info!(
                    pid,
                    task_id,
                    command = %proc.command,
                    "Disowning background process from task (notifier active, will deliver completion)"
                );
                proc.owner_task_id = None;
                running.insert(pid, proc);
            }
        }

        // Lock-order discipline: do not hold `running` while mutating secondary
        // indexes. Index helpers acquire their own locks (`running_by_dedupe_key`,
        // `task_processes`) after the primary `running` lock is dropped.
        for (pid, proc) in &to_cleanup {
            self.remove_indexes_for_process(*pid, proc).await;
            self.completed.lock().await.remove(pid);
        }

        let mut cleaned = 0usize;
        for (pid, proc) in to_cleanup {
            match self
                .terminate_running_process(pid, proc, "task ended")
                .await
            {
                Ok(_) => cleaned += 1,
                Err(e) => {
                    warn!(
                        pid,
                        task_id,
                        error = %e,
                        "Failed to stop task-owned background process"
                    );
                }
            }
        }
        Ok(cleaned)
    }

    /// Enable trust-all mode: auto-approve all commands without prompting.
    /// Requires user approval since this is a security-sensitive action.
    async fn handle_trust_all(&self, session_id: &str) -> anyhow::Result<String> {
        // Check if already in trust-all mode
        {
            let prefixes = self.allowed_prefixes.read().await;
            if prefixes.iter().any(|p| p == "*") {
                return Ok(
                    "Trust-all mode is already enabled. All commands are auto-approved."
                        .to_string(),
                );
            }
        }

        // Request user approval
        match self
            .request_approval(
                session_id,
                "ENABLE TRUST-ALL MODE",
                RiskLevel::Critical,
                vec![
                    "All future commands will run without approval".to_string(),
                    "This includes dangerous commands (rm, sudo, etc.)".to_string(),
                    "Persists across restarts".to_string(),
                ],
                None,
            )
            .await
        {
            Ok(ApprovalResponse::AllowOnce)
            | Ok(ApprovalResponse::AllowSession)
            | Ok(ApprovalResponse::AllowAlways) => {
                // Add * to allowed prefixes
                let mut prefixes = self.allowed_prefixes.write().await;
                if !prefixes.iter().any(|p| p == "*") {
                    prefixes.push("*".to_string());
                    info!("Trust-all mode enabled: all commands will be auto-approved");

                    // Persist to database
                    if let Some(ref pool) = self.pool {
                        if let Err(e) = sqlx::query(
                            "INSERT OR IGNORE INTO terminal_allowed_prefixes (prefix) VALUES ('*')",
                        )
                        .execute(pool)
                        .await
                        {
                            warn!("Failed to persist trust-all mode: {}", e);
                        }
                    }
                }
                Ok(
                    "Trust-all mode enabled. All commands will now run without approval prompts."
                        .to_string(),
                )
            }
            Ok(ApprovalResponse::Deny) => Ok(
                "Trust-all mode was denied. Commands will continue to require approval."
                    .to_string(),
            ),
            Err(e) => Ok(format!("Could not get approval for trust-all mode: {}", e)),
        }
    }

    fn prune_completed_map(completed: &mut HashMap<u32, CompletedProcess>) {
        const COMPLETED_TTL: Duration = Duration::from_secs(10 * 60);
        const COMPLETED_CAP: usize = 128;

        completed.retain(|_, entry| entry.completed_at.elapsed() <= COMPLETED_TTL);
        if completed.len() <= COMPLETED_CAP {
            return;
        }

        let mut by_age: Vec<(u32, Instant)> = completed
            .iter()
            .map(|(pid, entry)| (*pid, entry.completed_at))
            .collect();
        by_age.sort_by_key(|(_, ts)| *ts);
        let to_remove = by_age.len().saturating_sub(COMPLETED_CAP);
        for (pid, _) in by_age.into_iter().take(to_remove) {
            completed.remove(&pid);
        }
    }

    /// Clean up any background processes whose reader tasks have finished.
    /// Finished outputs are retained briefly in `completed` so follow-up
    /// `action="check"` can still retrieve the final result.
    async fn reap_finished(&self) {
        let finished: Vec<(u32, RunningProcess)> = {
            let mut running = self.running.lock().await;
            let pids: Vec<u32> = running
                .iter()
                .filter(|(_, p)| p.reader_handle.is_finished())
                .map(|(pid, _)| *pid)
                .collect();
            let mut removed = Vec::with_capacity(pids.len());
            for pid in pids {
                if let Some(proc) = running.remove(&pid) {
                    removed.push((pid, proc));
                }
            }
            removed
        };

        if finished.is_empty() {
            return;
        }

        for (pid, proc) in finished {
            self.remove_indexes_for_process(pid, &proc).await;
            let exit_code = proc.reader_handle.await.ok().flatten();
            let stdout = String::from_utf8_lossy(&proc.stdout_buf.lock().await).to_string();
            let stderr = String::from_utf8_lossy(&proc.stderr_buf.lock().await).to_string();
            let mut output = format!(
                "[Process pid={} finished after {:.0}s]\n",
                pid,
                proc.started_at.elapsed().as_secs_f64()
            );
            output.push_str(&format_output(&stdout, &stderr, self.max_output_chars));
            if let Some(code) = exit_code {
                if code != 0 {
                    output.push_str(&format!("\n[exit code: {}]", code));
                }
            }

            let mut completed = self.completed.lock().await;
            completed.insert(
                pid,
                CompletedProcess {
                    output,
                    metadata: tracked_background_metadata(proc.detached, false, exit_code),
                    completed_at: Instant::now(),
                },
            );
            Self::prune_completed_map(&mut completed);
            info!(pid, command = %proc.command, "Reaped finished background process");
        }
    }

    /// Run a command: spawn, wait up to initial_timeout, return output or move to background.
    async fn handle_run(
        &self,
        command: &str,
        notify_session_id: &str,
        notify_goal_id: Option<&str>,
        task_id: Option<&str>,
        detach: bool,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let dedupe_key =
            Self::dedupe_key_for_run(command, notify_session_id, notify_goal_id, task_id);
        if let Some(existing_pid) = self.resolve_duplicate_running_pid(&dedupe_key).await {
            return Ok(ToolCallOutcome::from_output(format!(
                "Equivalent command is already running in this scope (pid={}). \
                 Use action=\"check\" pid={} to inspect progress or action=\"kill\" pid={} to stop it.",
                existing_pid, existing_pid, existing_pid
            )));
        }

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        configure_command_for_process_group(&mut cmd);
        let mut child = cmd.spawn()?;

        let pid = child.id().unwrap_or(0);

        let stdout_pipe = child.stdout.take().expect("stdout piped");
        let stderr_pipe = child.stderr.take().expect("stderr piped");

        let stdout_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));

        let stdout_buf_c = stdout_buf.clone();
        let stderr_buf_c = stderr_buf.clone();
        let (completion_tx, completion_rx) = tokio::sync::oneshot::channel::<Option<i32>>();

        // Spawn a task that drains both streams and then waits for the child to exit.
        let reader_handle = tokio::spawn(async move {
            let stdout_drain = drain_to_buffer(stdout_pipe, stdout_buf_c);
            let stderr_drain = drain_to_buffer(stderr_pipe, stderr_buf_c);
            tokio::join!(stdout_drain, stderr_drain);
            let exit_code = child.wait().await.ok().and_then(|status| status.code());
            let _ = completion_tx.send(exit_code);
            exit_code
        });

        // Wait up to initial_timeout for the reader (and thus the process) to finish.
        let poll_finished = async {
            loop {
                if reader_handle.is_finished() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };

        match tokio::time::timeout(self.initial_timeout, poll_finished).await {
            Ok(()) => {
                // Process finished within timeout — collect output.
                let exit_code = reader_handle.await.ok().flatten();
                let stdout_data = stdout_buf.lock().await;
                let stderr_data = stderr_buf.lock().await;
                let stdout = String::from_utf8_lossy(&stdout_data);
                let stderr = String::from_utf8_lossy(&stderr_data);
                let mut output = format_output(&stdout, &stderr, self.max_output_chars);
                if let Some(code) = exit_code {
                    if code != 0 {
                        output.push_str(&format!("\n[exit code: {}]", code));
                    }
                }
                Ok(ToolCallOutcome {
                    metadata: foreground_terminal_metadata(exit_code),
                    output,
                })
            }
            Err(_) => {
                // Timeout — check if this is a daemon/background command where the
                // parent shell exited but pipes are held open by the detached child.
                // In that case the reader task will never finish naturally, so capture
                // partial output and return immediately instead of entering the
                // infinite background tracking loop.
                let daemon_hits = detect_daemonization_primitives(command);
                if !daemon_hits.is_empty() {
                    let partial_stdout = {
                        let b = stdout_buf.lock().await;
                        String::from_utf8_lossy(&b).to_string()
                    };
                    let partial_stderr = {
                        let b = stderr_buf.lock().await;
                        String::from_utf8_lossy(&b).to_string()
                    };
                    let output =
                        format_output(&partial_stdout, &partial_stderr, self.max_output_chars);
                    reader_handle.abort();
                    let output = format!(
                        "Detached background command launched (pid={}).\n\
                         The process is running independently and is not task-owned.\n\
                         This detached daemonized process is not tracked by action=\"check\"/\"kill\".\n\n\
                         Initial output:\n{}",
                        pid, output
                    );
                    return Ok(ToolCallOutcome {
                        metadata: ToolCallMetadata {
                            background_started: true,
                            detached: true,
                            timed_out: false,
                            completion_notifications_enabled: false,
                            ..ToolCallMetadata::default()
                        },
                        output,
                    });
                }

                // Non-daemon command: move process to background tracking.
                let elapsed = self.initial_timeout.as_secs();
                let partial_stdout = {
                    let b = stdout_buf.lock().await;
                    let tail = if b.len() > 500 {
                        &b[b.len() - 500..]
                    } else {
                        &b
                    };
                    String::from_utf8_lossy(tail).to_string()
                };
                let notify_on_completion = Arc::new(AtomicBool::new(true));
                let owner_task_id = task_id
                    .map(str::to_string)
                    .filter(|id| !id.trim().is_empty());

                let proc = RunningProcess {
                    command: command.to_string(),
                    dedupe_key: Some(dedupe_key.clone()),
                    owner_task_id: owner_task_id.clone(),
                    detached: detach,
                    started_at: Instant::now() - self.initial_timeout,
                    stdout_buf,
                    stderr_buf,
                    reader_handle,
                    child_id: pid,
                    notify_on_completion: notify_on_completion.clone(),
                    notifier_active: false,
                };

                self.running.lock().await.insert(pid, proc);
                self.insert_indexes_for_process(
                    pid,
                    Some(&dedupe_key),
                    owner_task_id.as_deref(),
                    detach,
                )
                .await;

                // Deterministic completion delivery: notify user when background command finishes
                // even if the agent loop ends before an explicit `action="check"` call.
                let mut notifier_started = false;
                let state_for_notify = self.state.clone();
                let hub_for_notify = self.get_hub();
                if state_for_notify.is_some() || hub_for_notify.is_some() {
                    let goal_id_for_notify = notify_goal_id.unwrap_or("").to_string();
                    let session_for_notify = notify_session_id.trim().to_string();
                    let command_for_notify = command.to_string();
                    let stdout_for_notify = {
                        let running = self.running.lock().await;
                        running.get(&pid).map(|p| p.stdout_buf.clone())
                    };
                    let stderr_for_notify = {
                        let running = self.running.lock().await;
                        running.get(&pid).map(|p| p.stderr_buf.clone())
                    };
                    let started_at_for_notify = Instant::now() - self.initial_timeout;
                    let max_output_chars = self.max_output_chars;
                    let status_tx_for_notify = status_tx.clone();
                    if let (Some(stdout_buf), Some(stderr_buf)) =
                        (stdout_for_notify, stderr_for_notify)
                    {
                        tokio::spawn(async move {
                            if session_for_notify.is_empty() {
                                warn!(
                                    pid,
                                    command = %command_for_notify,
                                    "Terminal background notifier skipped enqueue due to empty session id"
                                );
                                notify_on_completion.store(false, Ordering::Relaxed);
                                return;
                            }
                            let command_summary = truncate_str(
                                &command_for_notify
                                    .split_whitespace()
                                    .collect::<Vec<_>>()
                                    .join(" "),
                                160,
                            );

                            let mut completion_rx = completion_rx;
                            let mut ping_interval = tokio::time::interval(Duration::from_secs(
                                BACKGROUND_PROGRESS_INTERVAL_SECS,
                            ));
                            ping_interval
                                .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                            // Consume the immediate first tick; we want periodic pings only.
                            ping_interval.tick().await;
                            let mut ping_count: u32 = 0;

                            loop {
                                tokio::select! {
                                    exit = &mut completion_rx => {
                                        let exit_code = match exit {
                                            Ok(code) => code,
                                            Err(e) => {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier lost completion signal"
                                                );
                                                None
                                            }
                                        };
                                        if !notify_on_completion.load(Ordering::Relaxed) {
                                            warn!(
                                                pid,
                                                command = %command_for_notify,
                                                "Terminal background notifier suppressed (check/kill already handled notification)"
                                            );
                                            return;
                                        }

                                        let stdout = String::from_utf8_lossy(&stdout_buf.lock().await).to_string();
                                        let stderr = String::from_utf8_lossy(&stderr_buf.lock().await).to_string();
                                        let output = truncate_with_note(
                                            &format_output(&stdout, &stderr, max_output_chars),
                                            2500,
                                        );
                                        let elapsed_secs = started_at_for_notify.elapsed().as_secs();
                                        let status = if exit_code == Some(0) {
                                            "completed"
                                        } else {
                                            "finished with errors"
                                        };
                                        let mut message = format!(
                                            "Background terminal command {} after {}s.\nCommand: `{}`\n\nOutput:\n{}",
                                            status, elapsed_secs, command_summary, output
                                        );
                                        if let Some(code) = exit_code {
                                            if code != 0 {
                                                message.push_str(&format!("\n[exit code: {}]", code));
                                            }
                                        }

                                        if let Some(ref tx) = status_tx_for_notify {
                                            if let Err(e) = tx.try_send(StatusUpdate::ToolProgress {
                                                name: "terminal".to_string(),
                                                chunk: format!(
                                                    "Background command finished (pid={}): {}",
                                                    pid, command_summary
                                                ),
                                            }) {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier failed to send progress status update"
                                                );
                                            }
                                        }

                                        let mut delivered = false;
                                        if let Some(ref hub) = hub_for_notify {
                                            if let Err(e) = hub.send_text(&session_for_notify, &message).await {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    session_id = %session_for_notify,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier failed direct hub completion delivery"
                                                );
                                            } else {
                                                delivered = true;
                                            }
                                        }
                                        if !delivered {
                                            if let Some(ref state) = state_for_notify {
                                                let entry = crate::traits::NotificationEntry::new(
                                                    &goal_id_for_notify,
                                                    &session_for_notify,
                                                    "progress",
                                                    &message,
                                                );
                                                if let Err(e) = state.enqueue_notification(&entry).await {
                                                    warn!(
                                                        pid,
                                                        error = %e,
                                                        session_id = %session_for_notify,
                                                        goal_id = %goal_id_for_notify,
                                                        command = %command_for_notify,
                                                        "Terminal background notifier failed to enqueue completion notification"
                                                    );
                                                }
                                            } else {
                                                warn!(
                                                    pid,
                                                    session_id = %session_for_notify,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier has no fallback queue; completion update dropped"
                                                );
                                            }
                                        }
                                        break;
                                    }
                                    _ = ping_interval.tick() => {
                                        if !notify_on_completion.load(Ordering::Relaxed) {
                                            warn!(
                                                pid,
                                                command = %command_for_notify,
                                                "Terminal background progress pings suppressed (check/kill already handled notification)"
                                            );
                                            return;
                                        }

                                        ping_count += 1;
                                        if ping_count > MAX_BACKGROUND_PROGRESS_PINGS {
                                            // Stop sending periodic pings but keep waiting
                                            // for completion to send the final notification.
                                            // (intentionally empty — skip the rest of this branch)
                                        } else {

                                        let elapsed_secs = started_at_for_notify.elapsed().as_secs();
                                        let stdout = String::from_utf8_lossy(&stdout_buf.lock().await).to_string();
                                        let stderr = String::from_utf8_lossy(&stderr_buf.lock().await).to_string();
                                        let latest_output = truncate_with_note(
                                            &format_output(&stdout, &stderr, max_output_chars),
                                            1000,
                                        );
                                        let mut message = format!(
                                            "Background terminal command still running after {}s (pid={}).\nCommand: `{}`",
                                            elapsed_secs, pid, command_summary
                                        );
                                        if !latest_output.trim().is_empty() {
                                            message.push_str(&format!("\n\nLatest output:\n{}", latest_output));
                                        }

                                        if let Some(ref tx) = status_tx_for_notify {
                                            if let Err(e) = tx.try_send(StatusUpdate::ToolProgress {
                                                name: "terminal".to_string(),
                                                chunk: format!(
                                                    "Background command still running (pid={}, {}s elapsed): {}",
                                                    pid, elapsed_secs, command_summary
                                                ),
                                            }) {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier failed to send periodic progress status update"
                                                );
                                            }
                                        }

                                        let mut delivered = false;
                                        if let Some(ref hub) = hub_for_notify {
                                            if let Err(e) = hub.send_text(&session_for_notify, &message).await {
                                                warn!(
                                                    pid,
                                                    error = %e,
                                                    session_id = %session_for_notify,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier failed direct hub periodic delivery"
                                                );
                                            } else {
                                                delivered = true;
                                            }
                                        }

                                        if !delivered {
                                            if let Some(ref state) = state_for_notify {
                                                let entry = crate::traits::NotificationEntry::new(
                                                    &goal_id_for_notify,
                                                    &session_for_notify,
                                                    "progress",
                                                    &message,
                                                );
                                                if let Err(e) = state.enqueue_notification(&entry).await {
                                                    warn!(
                                                        pid,
                                                        error = %e,
                                                        session_id = %session_for_notify,
                                                        goal_id = %goal_id_for_notify,
                                                        command = %command_for_notify,
                                                        "Terminal background notifier failed to enqueue periodic progress notification"
                                                    );
                                                }
                                            } else {
                                                warn!(
                                                    pid,
                                                    session_id = %session_for_notify,
                                                    command = %command_for_notify,
                                                    "Terminal background notifier has no fallback queue; periodic update dropped"
                                                );
                                            }
                                        }
                                        } // close else for ping_count cap
                                    }
                                }
                            }
                        });
                        notifier_started = true;
                        // Mark the process so cleanup_task_processes knows the notifier
                        // is actively monitoring it and will deliver the result.
                        if let Some(proc) = self.running.lock().await.get_mut(&pid) {
                            proc.notifier_active = true;
                        }
                    } else {
                        warn!(
                            pid,
                            command = %command,
                            "Terminal background notifier not started because process buffers were unavailable"
                        );
                        notify_on_completion.store(false, Ordering::Relaxed);
                    }
                } else {
                    warn!(
                        pid,
                        command = %command,
                        "Terminal background notifier disabled: neither state queue nor channel hub is configured"
                    );
                    notify_on_completion.store(false, Ordering::Relaxed);
                }

                let mut msg = format!(
                    "Command still running after {}s. Moved to background (pid={}).\n\
                     IMPORTANT: Continue with your next steps immediately — do NOT wait or repeatedly check this process.\n\
                     You can run other commands (like curl) while this runs in the background.\n\
                     Use action=\"check\" with pid={} to see output later, or action=\"kill\" with pid={} to stop it.",
                    elapsed, pid, pid, pid
                );
                if detach {
                    msg.push_str(
                        "\n\nDetached mode is enabled: this process will not be auto-killed at task end.",
                    );
                } else if notifier_started {
                    msg.push_str(
                        "\n\nCompletion notifications are enabled. The user will be notified when this process finishes.",
                    );
                } else {
                    msg.push_str(
                        "\n\nThis process is task-owned and will be auto-killed when the current task ends.",
                    );
                }
                if !partial_stdout.is_empty() {
                    msg.push_str(&format!("\n\nPartial output so far:\n{}", partial_stdout));
                }
                Ok(ToolCallOutcome {
                    metadata: ToolCallMetadata {
                        background_started: true,
                        timed_out: true,
                        detached: detach,
                        completion_notifications_enabled: !detach && notifier_started,
                        ..ToolCallMetadata::default()
                    },
                    output: msg,
                })
            }
        }
    }

    /// Check on a background process: return partial output or final result.
    async fn handle_check(&self, pid: u32) -> anyhow::Result<ToolCallOutcome> {
        let mut running = self.running.lock().await;

        let Some(proc) = running.get(&pid) else {
            drop(running);
            let mut completed = self.completed.lock().await;
            if let Some(done) = completed.remove(&pid) {
                return Ok(ToolCallOutcome {
                    output: done.output,
                    metadata: done.metadata,
                });
            }
            return Ok(ToolCallOutcome::from_output(format!(
                "No tracked process with pid={}. It may have already finished and been reaped.",
                pid
            )));
        };

        if proc.reader_handle.is_finished() {
            // Process done — collect final output and remove from map.
            let proc = running.remove(&pid).unwrap();
            self.remove_indexes_for_process(pid, &proc).await;
            proc.notify_on_completion.store(false, Ordering::Relaxed);
            let exit_code = proc.reader_handle.await.ok().flatten();
            let stdout = String::from_utf8_lossy(&proc.stdout_buf.lock().await).to_string();
            let stderr = String::from_utf8_lossy(&proc.stderr_buf.lock().await).to_string();
            let mut output = format!(
                "[Process pid={} finished after {:.0}s]\n",
                pid,
                proc.started_at.elapsed().as_secs_f64()
            );
            output.push_str(&format_output(&stdout, &stderr, self.max_output_chars));
            if let Some(code) = exit_code {
                if code != 0 {
                    output.push_str(&format!("\n[exit code: {}]", code));
                }
            }
            Ok(ToolCallOutcome {
                output,
                metadata: tracked_background_metadata(proc.detached, false, exit_code),
            })
        } else {
            // Still running — return tail of buffer.
            let elapsed = proc.started_at.elapsed().as_secs();
            let stdout_tail = {
                let b = proc.stdout_buf.lock().await;
                let tail_start = b.len().saturating_sub(2000);
                String::from_utf8_lossy(&b[tail_start..]).to_string()
            };
            let stderr_tail = {
                let b = proc.stderr_buf.lock().await;
                let tail_start = b.len().saturating_sub(500);
                String::from_utf8_lossy(&b[tail_start..]).to_string()
            };
            let mut output = format!(
                "[Process pid={} still running ({} seconds elapsed, command: `{}`)]",
                pid, elapsed, proc.command
            );
            if proc.detached {
                output.push_str("\n[mode: detached]");
            } else if let Some(task_id) = proc.owner_task_id.as_deref() {
                output.push_str(&format!("\n[mode: task-owned, task_id={}]", task_id));
            } else if proc.notifier_active {
                output.push_str("\n[mode: background, notifications active]");
            }
            if !stdout_tail.is_empty() {
                output.push_str(&format!("\n\nRecent stdout:\n{}", stdout_tail));
            }
            if !stderr_tail.is_empty() {
                output.push_str(&format!("\n\nRecent stderr:\n{}", stderr_tail));
            }
            output.push_str(&format!(
                "\n\nUse action=\"check\" pid={} to check again, or action=\"kill\" pid={} to stop.",
                pid, pid
            ));
            Ok(ToolCallOutcome {
                output,
                metadata: tracked_background_metadata(
                    proc.detached,
                    proc.notifier_active && !proc.detached,
                    None,
                ),
            })
        }
    }

    /// Kill a background process: SIGTERM, wait 2s, SIGKILL if needed.
    async fn handle_kill(&self, pid: u32) -> anyhow::Result<ToolCallOutcome> {
        let mut running = self.running.lock().await;

        let Some(proc) = running.remove(&pid) else {
            return Ok(ToolCallOutcome::from_output(format!(
                "No tracked process with pid={}. It may have already finished.",
                pid
            )));
        };
        drop(running);
        self.remove_indexes_for_process(pid, &proc).await;
        self.completed.lock().await.remove(&pid);

        let detached = proc.detached;
        let output = self
            .terminate_running_process(pid, proc, "manual kill")
            .await?;
        Ok(ToolCallOutcome {
            output,
            metadata: tracked_background_metadata(detached, false, None),
        })
    }
}

impl Drop for TerminalTool {
    fn drop(&mut self) {
        // Best-effort kill of all tracked background processes.
        if let Ok(running) = self.running.try_lock() {
            for (_, proc) in running.iter() {
                send_sigterm(proc.child_id);
                send_sigkill(proc.child_id);
            }
        }
    }
}

#[derive(Deserialize)]
struct TerminalArgs {
    command: Option<String>,
    #[serde(default = "default_action")]
    action: String,
    pid: Option<u32>,
    /// If true, allow a timed-out command to outlive task boundaries.
    /// Default false: timed-out background commands are task-owned and auto-cleaned
    /// when the task ends.
    #[serde(default, alias = "background")]
    detach: bool,
    #[serde(default)]
    _untrusted_source: bool,
    #[serde(default)]
    _session_id: String,
    #[serde(default)]
    _task_id: Option<String>,
    /// Injected by agent - goal context for routing background notifications.
    #[serde(default)]
    _goal_id: Option<String>,
    /// Injected by agent for role-aware safeguards.
    #[serde(default)]
    _user_role: Option<String>,
    /// Explicitly set by the agent from ChannelContext.trusted — never derived
    /// from session ID strings. Only trusted scheduled tasks set this to true.
    #[serde(default)]
    _trusted_session: bool,
}

fn default_action() -> String {
    "run".to_string()
}

fn extract_terminal_exit_code(output: &str) -> Option<i32> {
    let marker = "[exit code:";
    let start = output.rfind(marker)?;
    let rest = output[start + marker.len()..].trim_start();
    let code_token: String = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
        .collect();
    if code_token.is_empty() {
        None
    } else {
        code_token.parse::<i32>().ok()
    }
}

fn foreground_terminal_metadata(exit_code: Option<i32>) -> ToolCallMetadata {
    ToolCallMetadata {
        exit_code,
        timed_out: false,
        background_started: false,
        detached: false,
        completion_notifications_enabled: false,
        transport_error: None,
    }
}

fn tracked_background_metadata(
    detached: bool,
    completion_notifications_enabled: bool,
    exit_code: Option<i32>,
) -> ToolCallMetadata {
    ToolCallMetadata {
        exit_code,
        timed_out: true,
        background_started: true,
        detached,
        completion_notifications_enabled,
        transport_error: None,
    }
}

#[async_trait]
impl Tool for TerminalTool {
    fn name(&self) -> &str {
        "terminal"
    }

    fn description(&self) -> &str {
        "Execute a shell command. If a command is not pre-approved, the user will be asked to authorize it."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "terminal",
            "description": "Execute any command available on this system — shell commands, CLI tools (python, node, claude, gemini, cargo, docker, git, etc.), scripts, and anything else installed. If a command is not pre-approved, the user may be asked to authorize it in real time.\n\nLong-running commands: if execution exceeds the timeout, it is tracked in background with a pid. When completion notifications are available, the process continues running and the user is notified automatically when it finishes. Otherwise, `detach=false` processes are task-owned and auto-killed at task end. Set `detach=true` for intentional long-lived execution (daemons, servers) that should survive indefinitely.\n\nIMPORTANT: Do not use heredoc (cat <<EOF), echo-redirect, or printf patterns to create files. Always use the `write_file` tool for file creation — it handles content atomically without shell quoting issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. REQUIRED when action is \"run\"."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["run", "check", "kill", "trust_all"],
                        "description": "Action to perform: \"run\" (default) executes a command — requires \"command\". \"check\" shows output of a background process — requires \"pid\". \"kill\" stops a background process — requires \"pid\". \"trust_all\" enables auto-approval for all commands."
                    },
                    "detach": {
                        "type": "boolean",
                        "description": "For action=\"run\": if true, the process survives indefinitely (for daemons/servers). Default false: timed-out processes survive with notifications when available, otherwise auto-killed at task end."
                    },
                    "pid": {
                        "type": "integer",
                        "description": "Process ID for check/kill actions (returned when a command moves to background)"
                    }
                },
                "required": ["action", "command"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        // For backwards compatibility, delegate to call_with_status with no sender.
        self.call_with_status(arguments, None).await
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        self.call_with_status_outcome(arguments, status_tx)
            .await
            .map(|outcome| outcome.output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: TerminalArgs = serde_json::from_str(arguments)?;

        // Reap any finished background processes on each call.
        self.reap_finished().await;

        // Route background completion notifications to the origin session
        // when this terminal run is tied to a goal/task lead.
        let mut notify_session_id = args._session_id.clone();
        if let (Some(state), Some(goal_id)) = (self.state.as_ref(), args._goal_id.as_deref()) {
            if let Ok(Some(goal)) = state.get_goal(goal_id).await {
                if !goal.session_id.trim().is_empty() {
                    notify_session_id = goal.session_id;
                }
            }
        }

        let mut outcome = match args.action.as_str() {
            "check" => {
                let pid = args
                    .pid
                    .ok_or_else(|| anyhow::anyhow!("pid is required for action=\"check\""))?;
                self.handle_check(pid).await?
            }
            "kill" => {
                let pid = args
                    .pid
                    .ok_or_else(|| anyhow::anyhow!("pid is required for action=\"kill\""))?;
                self.handle_kill(pid).await?
            }
            "trust_all" => {
                ToolCallOutcome::from_output(self.handle_trust_all(&args._session_id).await?)
            }
            _ => {
                // "run" or default
                let command = args
                    .command
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("command is required for action=\"run\""))?;
                let command = command.trim();
                if command.is_empty() {
                    anyhow::bail!("command must not be empty for action=\"run\"");
                }

                if let Some((pattern, path)) = detect_unscoped_recursive_grep(command) {
                    return Ok(ToolCallOutcome::from_output(recursive_grep_block_message(
                        &pattern, &path,
                    )));
                }

                // Soft-block large heredoc file creation: redirects to write_file
                // which writes atomically without shell quoting issues.
                // Allow quoted heredoc delimiters (<<'EOF' or << 'EOF') since they
                // avoid shell expansion issues and serve as a fallback when write_file
                // fails with JSON escaping errors on complex content.
                if command.contains("<<") && command.len() > 500 {
                    let uses_quoted_heredoc = command.contains("<<'")
                        || command.contains("<< '")
                        || command.contains("<<\"")
                        || command.contains("<< \"");
                    if !uses_quoted_heredoc {
                        return Ok(ToolCallOutcome::from_output(
                            "Large heredoc file creation is unreliable through the terminal. \
                             Use the `write_file` tool instead — it writes files atomically \
                             and avoids shell quoting issues. If write_file fails with JSON \
                             encoding errors, use a quoted heredoc: cat > file << 'EOF'"
                                .to_string(),
                        ));
                    }
                }

                // Soft-block python3 -c with file I/O: redirects to read_file/write_file
                // which are safer, faster, and don't require approval.
                if is_python_c_with_file_io(command) {
                    return Ok(ToolCallOutcome::from_output(
                        "Blocked: `python3 -c` with file I/O is not allowed through terminal.\n\n\
                         Use dedicated tools instead:\n\
                         - `read_file` to read file contents\n\
                         - `write_file` to create or overwrite files\n\
                         - `edit_file` to modify specific parts of a file\n\
                         - `search_files` to search for patterns in files\n\n\
                         These tools are faster, do not require approval, and handle \
                         encoding/quoting correctly."
                            .to_string(),
                    ));
                }

                let daemon_hits = detect_daemonization_primitives(command);
                let mut daemonization_approved = false;
                if !daemon_hits.is_empty() {
                    let is_owner = args
                        ._user_role
                        .as_deref()
                        .is_some_and(|role| role.eq_ignore_ascii_case("owner"));
                    if !is_owner {
                        return Ok(ToolCallOutcome::from_output(format!(
                            "Blocked: daemonization primitives detected ({}) and only owners can approve detached/background process commands.",
                            daemon_hits.join(", ")
                        )));
                    }

                    if !args.detach {
                        return Ok(ToolCallOutcome::from_output(format!(
                            "Blocked: daemonization primitives detected ({}). \
                             Set `detach=true` explicitly for intentional long-lived background execution.",
                            daemon_hits.join(", ")
                        )));
                    }

                    let mut warnings = vec![
                        format!(
                            "Daemonization primitives detected: {}",
                            daemon_hits.join(", ")
                        ),
                        "Detached/background processes may survive cancellation and continue running.".to_string(),
                    ];
                    warnings.push("Approve only if this is intentional and necessary.".to_string());

                    match self
                        .request_approval(
                            &args._session_id,
                            command,
                            RiskLevel::Critical,
                            warnings,
                            args._task_id.as_deref(),
                        )
                        .await
                    {
                        Ok(ApprovalResponse::AllowOnce)
                        | Ok(ApprovalResponse::AllowSession)
                        | Ok(ApprovalResponse::AllowAlways) => {
                            daemonization_approved = true;
                        }
                        Ok(ApprovalResponse::Deny) => {
                            return Ok(ToolCallOutcome::from_output(
                                "Daemonizing command denied by owner.".to_string(),
                            ));
                        }
                        Err(e) => {
                            return Ok(ToolCallOutcome::from_output(format!(
                                "Could not get owner approval for daemonizing command: {}",
                                e
                            )));
                        }
                    }
                }

                // Classify command risk
                let mut assessment = classify_command(command);

                // Deterministic hard block for irreversible broad-path deletes.
                if let Some(reason) = hard_block_reason(command) {
                    warn!(
                        session_id = %args._session_id,
                        task_id = ?args._task_id,
                        command = %command,
                        reason = %reason,
                        "Blocked dangerous irreversible command"
                    );
                    return Ok(ToolCallOutcome::from_output(format!(
                        "{} Use scoped, non-destructive commands instead.",
                        reason
                    )));
                }

                // Check for learned patterns and potentially lower risk
                if let Some(ref pool) = self.pool {
                    if let Ok(Some((pattern, similarity))) =
                        find_matching_pattern(pool, command).await
                    {
                        if pattern.is_trusted()
                            && similarity >= 0.9
                            && assessment.level != RiskLevel::Critical
                        {
                            // Trusted pattern with high similarity - lower risk by one level
                            let original_level = assessment.level;
                            assessment.level = match assessment.level {
                                RiskLevel::Critical => RiskLevel::High,
                                RiskLevel::High => RiskLevel::Medium,
                                RiskLevel::Medium => RiskLevel::Safe,
                                RiskLevel::Safe => RiskLevel::Safe,
                            };
                            if assessment.level != original_level {
                                assessment.warnings.push(format!(
                                    "Risk lowered: similar to trusted pattern '{}' (approved {}x)",
                                    pattern.pattern, pattern.approval_count
                                ));
                                info!(
                                    command = %command,
                                    pattern = %pattern.pattern,
                                    original_risk = %original_level,
                                    new_risk = %assessment.level,
                                    "Lowered risk based on learned pattern"
                                );
                            }
                        } else if pattern.denial_count > pattern.approval_count {
                            // Pattern is frequently denied - add warning
                            assessment.warnings.push(format!(
                                "Similar commands have been denied {}x",
                                pattern.denial_count
                            ));
                        }
                    }
                }

                // Check if this is a trusted session (explicitly set by ChannelContext,
                // not derived from session ID strings — prevents session ID spoofing).
                let is_trusted_session = args._trusted_session;
                if args.detach && is_trusted_session {
                    // Intentional: trusted scheduled sessions are auto-approved, so
                    // disallow detached long-lived processes in that mode.
                    return Ok(ToolCallOutcome::from_output(
                        "Blocked: detach=true is not allowed for trusted scheduled sessions."
                            .to_string(),
                    ));
                }

                if args.detach && !daemonization_approved {
                    assessment.warnings.push(
                        "Detached execution requested (process may outlive task boundaries)."
                            .to_string(),
                    );
                }

                // Determine if approval is needed
                // Note: is_allowed() checks both permanent AND session-approved prefixes
                let is_allowed = self.is_allowed(command).await;
                let needs_approval = if daemonization_approved {
                    false
                } else if args.detach {
                    info!(command = %command, "Forcing approval: detach=true");
                    true
                } else if args._untrusted_source {
                    // External triggers always need approval regardless of mode
                    info!(command = %command, risk = %assessment.level, "Forcing approval: untrusted source");
                    true
                } else if is_trusted_session {
                    // Trusted scheduled tasks bypass approval
                    info!(command = %command, session = %args._session_id, "Auto-approved: trusted scheduled task");
                    false
                } else if !is_allowed {
                    true
                } else if assessment.level == RiskLevel::Critical
                    && self.permission_mode != PermissionMode::Yolo
                {
                    // Even with wildcard/prefix approval, Critical commands require explicit approval
                    info!(command = %command, risk = %assessment.level, "Forcing approval: critical command despite prefix match");
                    true
                } else {
                    false
                };

                if needs_approval {
                    match self
                        .request_approval(
                            &args._session_id,
                            command,
                            assessment.level,
                            assessment.warnings.clone(),
                            args._task_id.as_deref(),
                        )
                        .await
                    {
                        Ok(ApprovalResponse::AllowOnce) => {
                            // Just run this once, but still learn from it
                            if let Some(ref pool) = self.pool {
                                let _ = record_approval(pool, command).await;
                            }
                        }
                        Ok(ApprovalResponse::AllowSession) => {
                            // Save to session-only storage (cleared on restart)
                            self.add_session_prefix(command).await;
                            if let Some(ref pool) = self.pool {
                                let _ = record_approval(pool, command).await;
                            }
                        }
                        Ok(ApprovalResponse::AllowAlways) => {
                            // Save to permanent storage (DB)
                            self.add_prefix(command).await;
                            if let Some(ref pool) = self.pool {
                                let _ = record_approval(pool, command).await;
                            }
                        }
                        Ok(ApprovalResponse::Deny) => {
                            // Record denial for learning
                            if let Some(ref pool) = self.pool {
                                let _ = record_denial(pool, command).await;
                            }
                            return Ok(ToolCallOutcome::from_output(
                                "Command denied by user.".to_string(),
                            ));
                        }
                        Err(e) => {
                            return Ok(ToolCallOutcome::from_output(format!(
                                "Could not get approval: {}",
                                e
                            )));
                        }
                    }
                }

                self.handle_run(
                    command,
                    &notify_session_id,
                    args._goal_id.as_deref(),
                    args._task_id.as_deref(),
                    args.detach,
                    status_tx,
                )
                .await?
            }
        };

        if outcome.metadata.exit_code.is_none() {
            outcome.metadata.exit_code = extract_terminal_exit_code(&outcome.output);
        }

        Ok(outcome)
    }

    async fn on_task_end(&self, task_id: &str, _session_id: &str) -> anyhow::Result<()> {
        let cleaned = self.cleanup_task_processes(task_id).await?;
        if cleaned > 0 {
            info!(
                task_id,
                cleaned, "Cleaned up task-owned terminal background process(es)"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{NotificationStore, StateStore, Tool};
    use sqlx::SqlitePool;
    use std::sync::Arc;
    use std::time::Duration;

    fn extract_pid_from_background_message(msg: &str) -> u32 {
        let marker = "pid=";
        let start = msg
            .find(marker)
            .expect("background response should include pid")
            + marker.len();
        let digits: String = msg[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse().expect("pid should parse as u32")
    }

    #[test]
    fn extract_terminal_exit_code_parses_marker() {
        assert_eq!(
            extract_terminal_exit_code(
                "[Process pid=123 finished after 2s]\nall done\n[exit code: 42]"
            ),
            Some(42)
        );
    }

    #[test]
    fn tracked_background_metadata_marks_background_and_detached() {
        let metadata = tracked_background_metadata(true, false, None);
        assert!(metadata.background_started);
        assert!(metadata.timed_out);
        assert!(metadata.detached);
        assert!(!metadata.completion_notifications_enabled);
    }

    #[tokio::test]
    async fn timed_out_background_run_sets_notification_metadata_when_available() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state as Arc<dyn StateStore>);

        let outcome = tool
            .call_with_status_outcome(
                r#"{"action":"run","command":"sleep 2; echo notify-meta","_session_id":"sess_meta","_user_role":"Owner"}"#,
                None,
            )
            .await
            .unwrap();
        assert!(outcome.output.contains("Moved to background (pid="));
        assert!(outcome.metadata.background_started);
        assert!(outcome.metadata.timed_out);
        assert!(!outcome.metadata.detached);
        assert!(outcome.metadata.completion_notifications_enabled);
    }

    #[tokio::test]
    async fn timed_out_background_run_clears_notification_metadata_when_unavailable() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let outcome = tool
            .call_with_status_outcome(
                r#"{"action":"run","command":"sleep 2; echo no-notify-meta","_session_id":"sess_meta2","_user_role":"Owner"}"#,
                None,
            )
            .await
            .unwrap();
        assert!(outcome.output.contains("Moved to background (pid="));
        assert!(outcome.metadata.background_started);
        assert!(outcome.metadata.timed_out);
        assert!(!outcome.metadata.detached);
        assert!(!outcome.metadata.completion_notifications_enabled);
    }

    // ── contains_shell_operator tests ──

    #[test]
    fn test_shell_operator_semicolon() {
        assert!(contains_shell_operator("ls; rm -rf"));
    }

    #[test]
    fn test_shell_operator_pipe() {
        assert!(contains_shell_operator("cat file | grep pattern"));
    }

    #[test]
    fn test_shell_operator_backtick() {
        assert!(contains_shell_operator("echo `whoami`"));
    }

    #[test]
    fn test_shell_operator_and() {
        assert!(contains_shell_operator("cmd1 && cmd2"));
    }

    #[test]
    fn test_shell_operator_subshell() {
        assert!(contains_shell_operator("echo $(whoami)"));
    }

    #[test]
    fn test_no_shell_operator_clean() {
        assert!(!contains_shell_operator("cargo build --release"));
    }

    #[test]
    fn test_no_shell_operator_flags() {
        assert!(!contains_shell_operator("ls -la /tmp"));
    }

    #[test]
    fn test_detect_unscoped_recursive_grep_broad_path() {
        let detected = detect_unscoped_recursive_grep(r#"grep -rc "async fn" ."#);
        assert!(
            detected.is_some(),
            "expected broad recursive grep to be detected"
        );
        let (pattern, path) = detected.unwrap();
        assert_eq!(pattern, "async fn");
        assert_eq!(path, ".");
    }

    #[test]
    fn test_detect_unscoped_recursive_grep_allows_scoped_dir() {
        let detected = detect_unscoped_recursive_grep(r#"grep -R "todo" src"#);
        assert!(
            detected.is_none(),
            "scoped directory search should be allowed"
        );
    }

    #[test]
    fn test_detect_unscoped_recursive_grep_allows_excludes() {
        let detected = detect_unscoped_recursive_grep(
            r#"grep -R --exclude-dir=node_modules --exclude-dir=target "todo" ."#,
        );
        assert!(detected.is_none(), "grep with excludes should be allowed");
    }

    #[test]
    fn test_detect_unscoped_recursive_grep_in_chained_shell_command() {
        let detected =
            detect_unscoped_recursive_grep(r#"cd /tmp/project && grep -rc "async fn" ."#);
        assert!(
            detected.is_some(),
            "expected chained command recursive grep to be detected"
        );
        let (pattern, path) = detected.unwrap();
        assert_eq!(pattern, "async fn");
        assert_eq!(path, ".");
    }

    // ── format_output tests ──

    #[test]
    fn test_format_stdout_only() {
        let result = format_output("hello", "", 1000);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_format_stderr_appended() {
        let result = format_output("out", "err", 1000);
        assert_eq!(result, "out\n--- stderr ---\nerr");
    }

    #[test]
    fn test_format_empty_no_output() {
        let result = format_output("", "", 1000);
        assert_eq!(result, "(no output)");
    }

    #[test]
    fn test_format_truncation() {
        let long_output = "a".repeat(200);
        let result = format_output(&long_output, "", 100);
        assert!(
            result.len() > 100,
            "truncated output should include the suffix"
        );
        assert!(result.ends_with("\n... (truncated)"));
        // The content portion before the suffix should be exactly max_chars long
        let prefix = &result[..100];
        assert_eq!(prefix, "a".repeat(100));
    }

    #[test]
    fn test_format_truncation_multibyte_utf8() {
        // "é" is 2 bytes in UTF-8, "日" is 3 bytes, "🎉" is 4 bytes
        let output = "aé日🎉".repeat(50); // mixed multi-byte chars
                                          // Truncate at various positions that may land mid-char
        for max in [1, 2, 3, 4, 5, 10, 50, 100] {
            let result = format_output(&output, "", max);
            // Must not panic and must be valid UTF-8 (String guarantees this)
            assert!(!result.is_empty());
            if output.len() > max {
                assert!(result.ends_with("\n... (truncated)"));
            }
        }
    }

    #[tokio::test]
    async fn test_daemonization_requires_owner_role() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(
                r#"{"action":"run","command":"nohup sleep 1 &","_session_id":"s1","_user_role":"Guest"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("only owners can approve"));
    }

    #[tokio::test]
    async fn test_terminal_hard_blocks_broad_irreversible_delete_even_in_yolo() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(r#"{"action":"run","command":"find / -delete","_session_id":"s1","_user_role":"Owner"}"#)
            .await
            .unwrap();
        assert!(response.contains("Blocked irreversible delete"));
        assert!(response.contains("scoped, non-destructive"));
    }

    #[tokio::test]
    async fn test_terminal_blocks_unscoped_recursive_grep() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(
                r#"{"action":"run","command":"grep -rc \"async fn\" .","_session_id":"s1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Blocked: broad recursive `grep`"));
        assert!(response.contains("search_files"));
        assert!(response.contains("rg -n --glob"));
    }

    #[tokio::test]
    async fn test_background_terminal_completion_enqueues_notification() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 2; echo terminal-notify-ok","_session_id":"sess_notify","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));

        let mut found = false;
        for _ in 0..40 {
            let pending = state.get_pending_notifications(20).await.unwrap();
            if pending.iter().any(|entry| {
                entry.session_id == "sess_notify"
                    && entry.notification_type == "progress"
                    && entry.message.contains("Background terminal command")
                    && entry.message.contains("terminal-notify-ok")
            }) {
                found = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        assert!(
            found,
            "expected background completion notification to be enqueued"
        );
    }

    #[tokio::test]
    async fn test_background_terminal_ack_progress_and_completion_sequence() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 3; echo terminal-sequence-ok","_session_id":"sess_seq","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(
            response.contains("Moved to background (pid="),
            "expected background ack in tool response, got: {}",
            response
        );

        let mut saw_progress_ping = false;
        let mut saw_completion = false;
        for _ in 0..60 {
            let pending = state.get_pending_notifications(50).await.unwrap();
            for entry in pending.iter().filter(|entry| {
                entry.session_id == "sess_seq" && entry.notification_type == "progress"
            }) {
                if entry.message.contains("still running after") {
                    saw_progress_ping = true;
                }
                if entry.message.contains("completed")
                    && entry.message.contains("terminal-sequence-ok")
                {
                    saw_completion = true;
                }
            }
            if saw_progress_ping && saw_completion {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        assert!(
            saw_progress_ping,
            "expected at least one periodic background progress ping"
        );
        assert!(
            saw_completion,
            "expected background completion notification with final output"
        );
    }

    #[tokio::test]
    async fn test_background_terminal_kill_suppresses_completion_notification() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 10","_session_id":"sess_kill","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let pid = extract_pid_from_background_message(&response);

        let kill_response = tool
            .call(&format!(
                r#"{{"action":"kill","pid":{},"_session_id":"sess_kill","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(kill_response.contains("stopped"));

        tokio::time::sleep(Duration::from_millis(500)).await;
        let pending = state.get_pending_notifications(20).await.unwrap();
        assert!(
            !pending.iter().any(|entry| {
                entry.session_id == "sess_kill"
                    && entry.notification_type == "progress"
                    && entry.message.contains("Background terminal command")
            }),
            "kill action should suppress background completion notification"
        );
    }

    #[tokio::test]
    async fn test_background_terminal_check_returns_result_after_reap() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 2; echo post-reap-ok","_session_id":"s1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let pid = extract_pid_from_background_message(&response);
        tokio::time::sleep(Duration::from_secs(3)).await;

        // This call will reap finished processes first; check must still return final output.
        let check = tool
            .call(&format!(
                r#"{{"action":"check","pid":{},"_session_id":"s1","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(check.contains("post-reap-ok"));
        assert!(check.contains(&format!("pid={}", pid)));
    }

    #[tokio::test]
    async fn test_task_end_cleanup_kills_task_owned_background_processes() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            2000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 10","_session_id":"s1","_task_id":"task-clean","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));
        let pid = extract_pid_from_background_message(&response);

        tool.on_task_end("task-clean", "s1").await.unwrap();
        tokio::time::sleep(Duration::from_millis(250)).await;

        let check = tool
            .call(&format!(
                r#"{{"action":"check","pid":{},"_session_id":"s1","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(
            check.contains("No tracked process"),
            "expected task-end cleanup to remove process tracking, got: {}",
            check
        );
    }

    #[tokio::test]
    async fn test_task_end_disowns_background_process_with_active_notifier() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let pool = state.pool();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await
        .with_state(state.clone() as Arc<dyn StateStore>);

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 3; echo disown-ok","_session_id":"sess_disown","_task_id":"task-disown","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="));
        let pid = extract_pid_from_background_message(&response);

        // Task ends — but the notifier is active, so the process should be disowned, not killed.
        tool.on_task_end("task-disown", "sess_disown")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Process should still be tracked (disowned, not removed).
        let check = tool
            .call(&format!(
                r#"{{"action":"check","pid":{},"_session_id":"sess_disown","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(
            !check.contains("No tracked process"),
            "expected process to survive task-end when notifier is active, got: {}",
            check
        );

        // Wait for the process to complete and the notification to be enqueued.
        let mut found = false;
        for _ in 0..50 {
            let pending = state.get_pending_notifications(20).await.unwrap();
            if pending.iter().any(|entry| {
                entry.session_id == "sess_disown"
                    && entry.message.contains("Background terminal command")
                    && entry.message.contains("disown-ok")
            }) {
                found = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        assert!(
            found,
            "expected background completion notification after task-end disown"
        );
    }

    #[tokio::test]
    async fn test_duplicate_background_run_is_suppressed_within_goal_scope() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            2000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        let first = tool
            .call(
                r#"{"action":"run","command":"sleep 5","_session_id":"sub-a","_task_id":"task-a","_goal_id":"goal-1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let pid = extract_pid_from_background_message(&first);

        let second = tool
            .call(
                r#"{"action":"run","command":"sleep   5","_session_id":"sub-b","_task_id":"task-b","_goal_id":"goal-1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        assert!(
            second.contains("Equivalent command is already running"),
            "expected duplicate suppression, got: {}",
            second
        );
        assert!(
            second.contains(&format!("pid={}", pid)),
            "expected duplicate response to reference original pid {}, got: {}",
            pid,
            second
        );

        tool.on_task_end("task-a", "sub-a").await.unwrap();
    }

    #[tokio::test]
    async fn test_detached_background_process_survives_task_end_cleanup() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, mut approval_rx) = mpsc::channel::<ApprovalRequest>(8);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            2000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        tokio::spawn(async move {
            while let Some(req) = approval_rx.recv().await {
                let _ = req.response_tx.send(ApprovalResponse::AllowOnce);
            }
        });

        let response = tool
            .call(
                r#"{"action":"run","command":"sleep 3","detach":true,"_session_id":"s1","_task_id":"task-detach","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let pid = extract_pid_from_background_message(&response);
        assert!(response.contains("Moved to background (pid="));
        assert!(response.contains("Detached mode is enabled"));

        tool.on_task_end("task-detach", "s1").await.unwrap();
        let check = tool
            .call(&format!(
                r#"{{"action":"check","pid":{},"_session_id":"s1","_user_role":"Owner"}}"#,
                pid
            ))
            .await
            .unwrap();
        assert!(
            !check.contains("No tracked process"),
            "detached process should not be cleaned by task-end hook"
        );

        let _ = tool
            .call(&format!(
                r#"{{"action":"kill","pid":{},"_session_id":"s1","_user_role":"Owner"}}"#,
                pid
            ))
            .await;
    }

    #[tokio::test]
    async fn test_daemon_command_returns_immediately_without_background_tracking() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, mut approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            2, // 2 second timeout
            4000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        // Auto-approve the daemonization approval request
        tokio::spawn(async move {
            if let Some(req) = approval_rx.recv().await {
                let _ = req.response_tx.send(ApprovalResponse::AllowOnce);
            }
        });

        let start = Instant::now();
        let response = tool
            .call(
                r#"{"action":"run","command":"nohup sleep 5 & echo $!","detach":true,"_session_id":"s1","_user_role":"Owner"}"#,
            )
            .await
            .unwrap();
        let elapsed = start.elapsed();

        // Should return promptly (within the timeout + small margin) rather than
        // entering the infinite background tracking loop.
        assert!(
            elapsed < Duration::from_secs(5),
            "daemon command should return within timeout, not stall; took {:?}",
            elapsed
        );
        assert!(
            response.contains("Detached background command launched"),
            "expected daemon early-return message, got: {}",
            response
        );
        assert!(
            response.contains("pid="),
            "expected pid in response, got: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_large_heredoc_soft_blocked() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        // Build a command >500 chars with UNQUOTED heredoc (should be soft-blocked)
        let large_content = "x".repeat(600);
        let command = format!("cat > /tmp/test.html << EOF\n{}\nEOF", large_content);
        let args = serde_json::json!({
            "action": "run",
            "command": command,
            "_session_id": "s1",
            "_user_role": "Owner"
        });

        let response = tool.call(&args.to_string()).await.unwrap();
        assert!(
            response.contains("write_file"),
            "expected heredoc soft-block to recommend write_file, got: {}",
            response
        );
        assert!(
            response.contains("unreliable"),
            "expected heredoc soft-block message, got: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_large_quoted_heredoc_allowed() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        let (approval_tx, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
        let tool = TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            1,
            1000,
            PermissionMode::Yolo,
            pool,
        )
        .await;

        // Build a command >500 chars with QUOTED heredoc (should be allowed)
        let large_content = "echo 'hello'";
        let command = format!(
            "cat > /tmp/test.py << 'PYEOF'\n{}\nPYEOF",
            large_content.repeat(50)
        );
        let args = serde_json::json!({
            "action": "run",
            "command": command,
            "_session_id": "s1",
            "_user_role": "Owner"
        });

        let response = tool.call(&args.to_string()).await.unwrap();
        assert!(
            !response.contains("unreliable"),
            "quoted heredoc should NOT be soft-blocked, got: {}",
            response
        );
    }
}
