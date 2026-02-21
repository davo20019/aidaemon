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
use crate::traits::{StateStore, Tool, ToolCapabilities};
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
struct RunningProcess {
    command: String,
    started_at: Instant,
    stdout_buf: Arc<Mutex<Vec<u8>>>,
    stderr_buf: Arc<Mutex<Vec<u8>>>,
    reader_handle: JoinHandle<Option<i32>>,
    child_id: u32,
    notify_on_completion: Arc<AtomicBool>,
}

/// Finalized background process output retained briefly so `action="check"`
/// can still return results after automatic reaping.
struct CompletedProcess {
    output: String,
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
        if contains_shell_operator(trimmed) {
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
    async fn add_session_prefix(&self, command: &str) {
        let prefix = command.split_whitespace().next().unwrap_or(command.trim());
        let mut session = self.session_approved.write().await;
        if session.insert(prefix.to_string()) {
            info!(
                prefix,
                "Added to session-approved prefixes (will reset on restart)"
            );
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

        let response: ApprovalResponse =
            match tokio::time::timeout(std::time::Duration::from_secs(300), response_rx).await {
                Ok(Ok(response)) => response,
                Ok(Err(_)) => {
                    tracing::warn!(command, "Approval response channel closed");
                    ApprovalResponse::Deny
                }
                Err(_) => {
                    tracing::warn!(command, "Approval request timed out (300s), auto-denying");
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
        let prefix = command.split_whitespace().next().unwrap_or(command.trim());
        if prefix == "*" {
            warn!("Refusing to add wildcard '*' as permanent prefix");
            return;
        }
        let mut prefixes = self.allowed_prefixes.write().await;
        if !prefixes.contains(&prefix.to_string()) {
            info!(prefix, "Adding to allowed command prefixes (persistent)");
            prefixes.push(prefix.to_string());

            // Persist to SQLite
            if let Some(ref pool) = self.pool {
                if let Err(e) = sqlx::query(
                    "INSERT OR IGNORE INTO terminal_allowed_prefixes (prefix) VALUES (?)",
                )
                .bind(prefix)
                .execute(pool)
                .await
                {
                    warn!(prefix, "Failed to persist allowed prefix: {}", e);
                }
            }
        }
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
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
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
                Ok(output)
            }
            Err(_) => {
                // Timeout — move process to background.
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

                let proc = RunningProcess {
                    command: command.to_string(),
                    started_at: Instant::now() - self.initial_timeout,
                    stdout_buf,
                    stderr_buf,
                    reader_handle,
                    child_id: pid,
                    notify_on_completion: notify_on_completion.clone(),
                };

                self.running.lock().await.insert(pid, proc);

                // Deterministic completion delivery: notify user when background command finishes
                // even if the agent loop ends before an explicit `action="check"` call.
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
                                    }
                                }
                            }
                        });
                    } else {
                        warn!(
                            pid,
                            command = %command,
                            "Terminal background notifier not started because process buffers were unavailable"
                        );
                    }
                } else {
                    warn!(
                        pid,
                        command = %command,
                        "Terminal background notifier disabled: neither state queue nor channel hub is configured"
                    );
                }

                let mut msg = format!(
                    "Command still running after {}s. Moved to background (pid={}).\n\
                     Use action=\"check\" with pid={} to see partial output, or action=\"kill\" with pid={} to stop it.",
                    elapsed, pid, pid, pid
                );
                if !partial_stdout.is_empty() {
                    msg.push_str(&format!("\n\nPartial output so far:\n{}", partial_stdout));
                }
                Ok(msg)
            }
        }
    }

    /// Check on a background process: return partial output or final result.
    async fn handle_check(&self, pid: u32) -> anyhow::Result<String> {
        let mut running = self.running.lock().await;

        let Some(proc) = running.get(&pid) else {
            drop(running);
            let mut completed = self.completed.lock().await;
            if let Some(done) = completed.remove(&pid) {
                return Ok(done.output);
            }
            return Ok(format!(
                "No tracked process with pid={}. It may have already finished and been reaped.",
                pid
            ));
        };

        if proc.reader_handle.is_finished() {
            // Process done — collect final output and remove from map.
            let proc = running.remove(&pid).unwrap();
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
            Ok(output)
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
            Ok(output)
        }
    }

    /// Kill a background process: SIGTERM, wait 2s, SIGKILL if needed.
    async fn handle_kill(&self, pid: u32) -> anyhow::Result<String> {
        let mut running = self.running.lock().await;

        let Some(proc) = running.remove(&pid) else {
            return Ok(format!(
                "No tracked process with pid={}. It may have already finished.",
                pid
            ));
        };
        proc.notify_on_completion.store(false, Ordering::Relaxed);
        drop(running);
        self.completed.lock().await.remove(&pid);

        // Send SIGTERM
        let term_sent = send_sigterm(proc.child_id);

        if term_sent {
            // Wait up to 2 seconds for graceful shutdown
            let finished = tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    if proc.reader_handle.is_finished() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            })
            .await;

            if finished.is_err() && !proc.reader_handle.is_finished() {
                // Still alive — SIGKILL
                send_sigkill(proc.child_id);
                // Give it a moment
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }

        // Collect whatever output we have
        let stdout = String::from_utf8_lossy(&proc.stdout_buf.lock().await).to_string();
        let stderr = String::from_utf8_lossy(&proc.stderr_buf.lock().await).to_string();
        let mut output = format!(
            "[Process pid={} killed after {:.0}s (command: `{}`)]\n",
            pid,
            proc.started_at.elapsed().as_secs_f64(),
            proc.command
        );
        output.push_str(&format_output(&stdout, &stderr, self.max_output_chars));
        Ok(output)
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
            "description": "Execute any command available on this system — shell commands, CLI tools (python, node, claude, gemini, cargo, docker, git, etc.), scripts, and anything else installed. If the command is not pre-approved, the user will be asked to authorize it in real time via an inline button. Never assume a command is unavailable — try it.\n\nLong-running commands are handled automatically: if a command exceeds the timeout, it moves to the background and you get a pid. Use action=\"check\" to see progress or action=\"kill\" to stop it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute (required for action=\"run\")"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["run", "check", "kill", "trust_all"],
                        "description": "Action to perform: \"run\" (default) executes a command, \"check\" shows output of a background process, \"kill\" stops a background process, \"trust_all\" enables auto-approval for all commands (requires user confirmation)"
                    },
                    "pid": {
                        "type": "integer",
                        "description": "Process ID for check/kill actions (returned when a command moves to background)"
                    }
                },
                "required": ["command"]
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

        match args.action.as_str() {
            "check" => {
                let pid = args
                    .pid
                    .ok_or_else(|| anyhow::anyhow!("pid is required for action=\"check\""))?;
                self.handle_check(pid).await
            }
            "kill" => {
                let pid = args
                    .pid
                    .ok_or_else(|| anyhow::anyhow!("pid is required for action=\"kill\""))?;
                self.handle_kill(pid).await
            }
            "trust_all" => self.handle_trust_all(&args._session_id).await,
            _ => {
                // "run" or default
                let command = args
                    .command
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("command is required for action=\"run\""))?;

                if let Some((pattern, path)) = detect_unscoped_recursive_grep(command) {
                    return Ok(recursive_grep_block_message(&pattern, &path));
                }

                let daemon_hits = detect_daemonization_primitives(command);
                let mut daemonization_approved = false;
                if !daemon_hits.is_empty() {
                    let is_owner = args
                        ._user_role
                        .as_deref()
                        .is_some_and(|role| role.eq_ignore_ascii_case("owner"));
                    if !is_owner {
                        return Ok(format!(
                            "Blocked: daemonization primitives detected ({}) and only owners can approve detached/background process commands.",
                            daemon_hits.join(", ")
                        ));
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
                            return Ok("Daemonizing command denied by owner.".to_string());
                        }
                        Err(e) => {
                            return Ok(format!(
                                "Could not get owner approval for daemonizing command: {}",
                                e
                            ));
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
                    return Ok(format!(
                        "{} Use scoped, non-destructive commands instead.",
                        reason
                    ));
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

                // Determine if approval is needed
                // Note: is_allowed() checks both permanent AND session-approved prefixes
                let is_allowed = self.is_allowed(command).await;
                let needs_approval = if daemonization_approved {
                    false
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
                            return Ok("Command denied by user.".to_string());
                        }
                        Err(e) => {
                            return Ok(format!("Could not get approval: {}", e));
                        }
                    }
                }

                self.handle_run(
                    command,
                    &notify_session_id,
                    args._goal_id.as_deref(),
                    status_tx,
                )
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{NotificationStore, StateStore};
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
        assert!(kill_response.contains("killed"));

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
}
