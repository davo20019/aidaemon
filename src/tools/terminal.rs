use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::SqlitePool;
use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::traits::Tool;
use crate::types::ApprovalResponse;

use super::command_patterns::{find_matching_pattern, record_approval, record_denial};
use super::command_risk::{classify_command, PermissionMode, RiskLevel};

/// Max bytes per stream buffer (1 MB) to prevent unbounded memory growth.
const BUFFER_CAP: usize = 1_048_576;

/// A request sent to the ChannelHub for command approval.
pub struct ApprovalRequest {
    pub command: String,
    pub session_id: String,
    pub risk_level: RiskLevel,
    pub warnings: Vec<String>,
    pub permission_mode: PermissionMode,
    pub response_tx: tokio::sync::oneshot::Sender<ApprovalResponse>,
}

/// A background process being tracked after it exceeded the initial timeout.
struct RunningProcess {
    command: String,
    started_at: Instant,
    stdout_buf: Arc<Mutex<Vec<u8>>>,
    stderr_buf: Arc<Mutex<Vec<u8>>>,
    reader_handle: JoinHandle<Option<std::process::ExitStatus>>,
    child_id: u32,
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
    initial_timeout: Duration,
    max_output_chars: usize,
    pool: Option<SqlitePool>,
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
        result.truncate(max_chars);
        result.push_str("\n... (truncated)");
    }
    result
}

/// Send SIGTERM to a process. Returns true if signal was sent successfully.
fn send_sigterm(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) == 0 }
}

/// Send SIGKILL to a process. Returns true if signal was sent successfully.
fn send_sigkill(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) == 0 }
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
        match sqlx::query_scalar::<_, String>(
            "SELECT prefix FROM terminal_allowed_prefixes"
        )
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
            initial_timeout: Duration::from_secs(initial_timeout_secs),
            max_output_chars,
            pool: Some(pool),
        }
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
        let prefix = command
            .split_whitespace()
            .next()
            .unwrap_or(command.trim());
        let mut session = self.session_approved.write().await;
        if session.insert(prefix.to_string()) {
            info!(prefix, "Added to session-approved prefixes (will reset on restart)");
        }
    }

    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: Vec<String>,
    ) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: command.to_string(),
                session_id: session_id.to_string(),
                risk_level,
                warnings,
                permission_mode: self.permission_mode,
                response_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;
        match tokio::time::timeout(std::time::Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                tracing::warn!(command, "Approval response channel closed");
                Ok(ApprovalResponse::Deny)
            }
            Err(_) => {
                tracing::warn!(command, "Approval request timed out (300s), auto-denying");
                Ok(ApprovalResponse::Deny)
            }
        }
    }

    async fn add_prefix(&self, command: &str) {
        let prefix = command
            .split_whitespace()
            .next()
            .unwrap_or(command.trim());
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
                    "INSERT OR IGNORE INTO terminal_allowed_prefixes (prefix) VALUES (?)"
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
                return Ok("Trust-all mode is already enabled. All commands are auto-approved.".to_string());
            }
        }

        // Request user approval
        match self.request_approval(
            session_id,
            "ENABLE TRUST-ALL MODE",
            RiskLevel::Critical,
            vec![
                "All future commands will run without approval".to_string(),
                "This includes dangerous commands (rm, sudo, etc.)".to_string(),
                "Persists across restarts".to_string(),
            ],
        ).await {
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
                            "INSERT OR IGNORE INTO terminal_allowed_prefixes (prefix) VALUES ('*')"
                        )
                        .execute(pool)
                        .await
                        {
                            warn!("Failed to persist trust-all mode: {}", e);
                        }
                    }
                }
                Ok("Trust-all mode enabled. All commands will now run without approval prompts.".to_string())
            }
            Ok(ApprovalResponse::Deny) => {
                Ok("Trust-all mode was denied. Commands will continue to require approval.".to_string())
            }
            Err(e) => {
                Ok(format!("Could not get approval for trust-all mode: {}", e))
            }
        }
    }

    /// Clean up any background processes whose reader tasks have finished.
    async fn reap_finished(&self) {
        let mut running = self.running.lock().await;
        let finished: Vec<u32> = running
            .iter()
            .filter(|(_, p)| p.reader_handle.is_finished())
            .map(|(pid, _)| *pid)
            .collect();
        for pid in finished {
            if let Some(proc) = running.remove(&pid) {
                info!(pid, command = %proc.command, "Reaped finished background process");
            }
        }
    }

    /// Run a command: spawn, wait up to initial_timeout, return output or move to background.
    async fn handle_run(&self, command: &str) -> anyhow::Result<String> {
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        let pid = child.id().unwrap_or(0);

        let stdout_pipe = child.stdout.take().expect("stdout piped");
        let stderr_pipe = child.stderr.take().expect("stderr piped");

        let stdout_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));

        let stdout_buf_c = stdout_buf.clone();
        let stderr_buf_c = stderr_buf.clone();

        // Spawn a task that drains both streams and then waits for the child to exit.
        let reader_handle = tokio::spawn(async move {
            let stdout_drain = drain_to_buffer(stdout_pipe, stdout_buf_c);
            let stderr_drain = drain_to_buffer(stderr_pipe, stderr_buf_c);
            tokio::join!(stdout_drain, stderr_drain);
            child.wait().await.ok()
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
                let status = reader_handle.await.ok().flatten();
                let stdout_data = stdout_buf.lock().await;
                let stderr_data = stderr_buf.lock().await;
                let stdout = String::from_utf8_lossy(&stdout_data);
                let stderr = String::from_utf8_lossy(&stderr_data);
                let mut output = format_output(&stdout, &stderr, self.max_output_chars);
                if let Some(s) = status {
                    if !s.success() {
                        output.push_str(&format!("\n[exit code: {}]", s.code().unwrap_or(-1)));
                    }
                }
                Ok(output)
            }
            Err(_) => {
                // Timeout — move process to background.
                let elapsed = self.initial_timeout.as_secs();
                let partial_stdout = {
                    let b = stdout_buf.lock().await;
                    let tail = if b.len() > 500 { &b[b.len()-500..] } else { &b };
                    String::from_utf8_lossy(tail).to_string()
                };

                let proc = RunningProcess {
                    command: command.to_string(),
                    started_at: Instant::now() - self.initial_timeout,
                    stdout_buf,
                    stderr_buf,
                    reader_handle,
                    child_id: pid,
                };

                self.running.lock().await.insert(pid, proc);

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
            return Ok(format!("No tracked process with pid={}. It may have already finished and been reaped.", pid));
        };

        if proc.reader_handle.is_finished() {
            // Process done — collect final output and remove from map.
            let proc = running.remove(&pid).unwrap();
            let status = proc.reader_handle.await.ok().flatten();
            let stdout = String::from_utf8_lossy(&proc.stdout_buf.lock().await).to_string();
            let stderr = String::from_utf8_lossy(&proc.stderr_buf.lock().await).to_string();
            let mut output = format!("[Process pid={} finished after {:.0}s]\n", pid, proc.started_at.elapsed().as_secs_f64());
            output.push_str(&format_output(&stdout, &stderr, self.max_output_chars));
            if let Some(s) = status {
                if !s.success() {
                    output.push_str(&format!("\n[exit code: {}]", s.code().unwrap_or(-1)));
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
            return Ok(format!("No tracked process with pid={}. It may have already finished.", pid));
        };

        // Send SIGTERM
        let term_sent = send_sigterm(proc.child_id);

        if term_sent {
            // Wait up to 2 seconds for graceful shutdown
            let finished = tokio::time::timeout(
                Duration::from_secs(2),
                async {
                    loop {
                        if proc.reader_handle.is_finished() {
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            ).await;

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
            pid, proc.started_at.elapsed().as_secs_f64(), proc.command
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: TerminalArgs = serde_json::from_str(arguments)?;

        // Reap any finished background processes on each call.
        self.reap_finished().await;

        match args.action.as_str() {
            "check" => {
                let pid = args.pid.ok_or_else(|| anyhow::anyhow!("pid is required for action=\"check\""))?;
                self.handle_check(pid).await
            }
            "kill" => {
                let pid = args.pid.ok_or_else(|| anyhow::anyhow!("pid is required for action=\"kill\""))?;
                self.handle_kill(pid).await
            }
            "trust_all" => {
                self.handle_trust_all(&args._session_id).await
            }
            _ => {
                // "run" or default
                let command = args.command
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("command is required for action=\"run\""))?;

                // Classify command risk
                let mut assessment = classify_command(command);

                // Check for learned patterns and potentially lower risk
                if let Some(ref pool) = self.pool {
                    if let Ok(Some((pattern, similarity))) = find_matching_pattern(pool, command).await {
                        if pattern.is_trusted() && similarity >= 0.9 {
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

                // Check if this is a trusted scheduled session (auto-approve)
                let is_trusted_session = args._session_id.starts_with("scheduled_");

                // Determine if approval is needed
                // Note: is_allowed() checks both permanent AND session-approved prefixes
                let is_allowed = self.is_allowed(command).await;
                let needs_approval = if args._untrusted_source {
                    // External triggers always need approval regardless of mode
                    info!(command = %command, risk = %assessment.level, "Forcing approval: untrusted source");
                    true
                } else if is_trusted_session {
                    // Trusted scheduled tasks bypass approval
                    info!(command = %command, session = %args._session_id, "Auto-approved: trusted scheduled task");
                    false
                } else if !is_allowed {
                    true
                } else if assessment.level == RiskLevel::Critical && self.permission_mode != PermissionMode::Yolo {
                    // Even with wildcard/prefix approval, Critical commands require explicit approval
                    info!(command = %command, risk = %assessment.level, "Forcing approval: critical command despite prefix match");
                    true
                } else {
                    false
                };

                if needs_approval {
                    match self.request_approval(
                        &args._session_id,
                        command,
                        assessment.level,
                        assessment.warnings.clone(),
                    ).await {
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

                self.handle_run(command).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(result.len() > 100, "truncated output should include the suffix");
        assert!(result.ends_with("\n... (truncated)"));
        // The content portion before the suffix should be exactly max_chars long
        let prefix = &result[..100];
        assert_eq!(prefix, "a".repeat(100));
    }
}
