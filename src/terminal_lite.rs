//! Chat-based "terminal lite" — a lightweight shell mode that runs single
//! commands without interactive TUIs. Extracted from `channels/telegram.rs`
//! so that any channel can reuse the session management and command execution.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::process::Command;
use tokio::sync::Mutex;

use crate::types::UserRole;

/// Maximum characters returned from a single command's output.
pub(crate) const TERMINAL_LITE_MAX_OUTPUT_CHARS: usize = 12_000;

/// Per-command timeout in seconds.
pub(crate) const TERMINAL_LITE_TIMEOUT_SECS: u64 = 90;

/// A single terminal-lite session, keyed by chat / conversation id.
#[derive(Debug, Clone)]
pub(crate) struct TerminalLiteSession {
    pub(crate) owner_user_id: u64,
    pub(crate) cwd: PathBuf,
    pub(crate) shell: String,
    pub(crate) preferred_agent: Option<String>,
    pub(crate) started_at: Instant,
    pub(crate) busy: bool,
}

/// Manages terminal-lite sessions for a single channel instance.
pub(crate) struct TerminalLiteManager {
    sessions: Mutex<HashMap<i64, TerminalLiteSession>>,
    allowed_prefixes: HashSet<String>,
}

// ---------------------------------------------------------------------------
// Free (pure) helper functions
// ---------------------------------------------------------------------------

/// Returns `true` if `token` looks like a shell environment variable
/// assignment (e.g. `FOO=bar`).
pub(crate) fn is_shell_env_assignment(token: &str) -> bool {
    let Some((name, _)) = token.split_once('=') else {
        return false;
    };
    if name.is_empty() {
        return false;
    }
    for (idx, ch) in name.chars().enumerate() {
        let is_valid = if idx == 0 {
            ch.is_ascii_alphabetic() || ch == '_'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_'
        };
        if !is_valid {
            return false;
        }
    }
    true
}

/// Returns `true` if `text` contains un-quoted shell control operators
/// (`;`, `|`, `&`, `>`, `<`, newlines, backticks, `$()`).
pub(crate) fn contains_shell_control_operators(text: &str) -> bool {
    let mut chars = text.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if escaped {
            escaped = false;
            continue;
        }

        if ch == '\\' && !in_single {
            escaped = true;
            continue;
        }

        if ch == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }

        if ch == '"' && !in_single {
            in_double = !in_double;
            continue;
        }

        // Command substitution works inside double quotes, so block it
        // unless we're inside single quotes.
        if ch == '$' && !in_single && matches!(chars.peek(), Some('(')) {
            return true;
        }
        if ch == '`' && !in_single {
            return true;
        }

        if in_single || in_double {
            continue;
        }

        if matches!(ch, ';' | '|' | '&' | '>' | '<' | '\n' | '\r') {
            return true;
        }
    }

    false
}

/// Resolve the initial working directory for a new terminal-lite session.
pub(crate) fn resolve_terminal_lite_cwd(raw: Option<&str>) -> anyhow::Result<PathBuf> {
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    resolve_terminal_lite_cwd_from_base(&base, raw.unwrap_or("").trim())
}

/// Resolve a path relative to `base`, expanding `~` and canonicalizing.
pub(crate) fn resolve_terminal_lite_cwd_from_base(
    base: &Path,
    raw: &str,
) -> anyhow::Result<PathBuf> {
    let resolved = if raw.is_empty() || raw == "~" {
        dirs::home_dir().unwrap_or_else(|| base.to_path_buf())
    } else if let Some(rest) = raw.strip_prefix("~/") {
        dirs::home_dir()
            .map(|home| home.join(rest))
            .unwrap_or_else(|| base.join(rest))
    } else {
        let p = PathBuf::from(raw);
        if p.is_absolute() {
            p
        } else {
            base.join(p)
        }
    };
    let canonical = resolved
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("invalid working dir '{}': {}", resolved.display(), e))?;
    if !canonical.is_dir() {
        anyhow::bail!("'{}' is not a directory", canonical.display());
    }
    Ok(canonical)
}

/// Read $SHELL (or fall back to `/bin/bash`).
pub(crate) fn default_terminal_lite_shell() -> String {
    std::env::var("SHELL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "/bin/bash".to_string())
}

/// Extract the first "real" command name from `text`, skipping leading
/// environment variable assignments like `FOO=bar`.
pub(crate) fn extract_terminal_lite_command_name(text: &str) -> Option<String> {
    let parts = shell_words::split(text).unwrap_or_else(|_| {
        text.split_whitespace()
            .map(std::string::ToString::to_string)
            .collect()
    });
    for token in parts {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_shell_env_assignment(trimmed) {
            continue;
        }
        return Some(trimmed.to_string());
    }
    None
}

/// Returns `Some(agent_name)` if `text` starts with a known interactive
/// terminal agent name (codex, claude, gemini, opencode).
pub(crate) fn is_terminal_lite_interactive_agent_command(text: &str) -> Option<String> {
    let parts = shell_words::split(text).unwrap_or_else(|_| {
        text.split_whitespace()
            .map(std::string::ToString::to_string)
            .collect()
    });
    parts
        .first()
        .and_then(|v| crate::normalize_terminal_agent_name(v))
        .map(|s| s.to_string())
}

/// Static help text for the `/terminal lite` command.
pub(crate) fn terminal_lite_help_text() -> String {
    "Terminal lite mode (owner only, chat-based)\n\n\
     Commands:\n\
     /terminal lite start [working_dir]\n\
     /terminal lite start <codex|claude|gemini|opencode> [working_dir]\n\
     /terminal lite [working_dir]\n\
     /terminal lite <codex|claude|gemini|opencode> [working_dir]\n\
     /terminal lite status\n\
     /terminal lite stop\n\n\
     After start, every non-slash message in this chat is treated as a shell command.\n\
     Built-ins: cd <path>, exit, quit\n\
     Note: interactive agent TUIs (codex/claude/gemini/opencode) require full /terminal Mini App mode.\n\
     Timeout: 90s per command."
        .to_string()
}

/// Handle a `cd` built-in, mutating the session's working directory.
/// Returns `Some(reply)` if the text starts with `cd`, `None` otherwise.
fn terminal_lite_try_handle_cd(session: &mut TerminalLiteSession, text: &str) -> Option<String> {
    let parts = shell_words::split(text).unwrap_or_else(|_| {
        text.split_whitespace()
            .map(std::string::ToString::to_string)
            .collect()
    });
    if parts.is_empty() || parts[0] != "cd" {
        return None;
    }
    let target = if parts.len() <= 1 {
        "~".to_string()
    } else {
        parts[1].clone()
    };
    match resolve_terminal_lite_cwd_from_base(&session.cwd, target.trim()) {
        Ok(path) => {
            session.cwd = path.clone();
            Some(format!("cwd -> {}", path.display()))
        }
        Err(err) => Some(format!("cd: {}", err)),
    }
}

/// Run a single command in the session's shell and return the formatted output.
async fn run_terminal_lite_command(session: &TerminalLiteSession, command_text: &str) -> String {
    let mut cmd = Command::new(&session.shell);
    if cfg!(windows) {
        cmd.arg("-NoLogo").arg("-Command").arg(command_text);
    } else {
        cmd.arg("-lc").arg(command_text);
    }
    cmd.current_dir(&session.cwd);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.kill_on_drop(true);
    cmd.env_remove("CLAUDECODE");
    cmd.env_remove("CLAUDE_CODE");
    if !cfg!(windows) {
        cmd.env("TERM", "xterm-256color");
    }

    let output = match tokio::time::timeout(
        Duration::from_secs(TERMINAL_LITE_TIMEOUT_SECS),
        cmd.output(),
    )
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(err)) => {
            return format!("Failed to run command: {}", err);
        }
        Err(_) => {
            return format!(
                "\u{23f1}\u{fe0f} Command timed out after {}s: {}",
                TERMINAL_LITE_TIMEOUT_SECS, command_text
            );
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let mut body = String::new();
    body.push_str("$ ");
    body.push_str(command_text);
    body.push_str("\n\n");

    if stdout.trim().is_empty() && stderr.trim().is_empty() {
        body.push_str("(no output)\n");
    } else {
        if !stdout.is_empty() {
            body.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !stdout.ends_with('\n') && !stdout.is_empty() {
                body.push('\n');
            }
            body.push_str(&stderr);
        }
    }

    if body.chars().count() > TERMINAL_LITE_MAX_OUTPUT_CHARS {
        let clipped: String = body.chars().take(TERMINAL_LITE_MAX_OUTPUT_CHARS).collect();
        body = format!(
            "{}\n\n[output truncated to {} chars]",
            clipped, TERMINAL_LITE_MAX_OUTPUT_CHARS
        );
    }

    let exit_code = output.status.code().unwrap_or(-1);
    body.push_str(&format!("\n[exit {}]", exit_code));
    body
}

// ---------------------------------------------------------------------------
// TerminalLiteManager
// ---------------------------------------------------------------------------

impl TerminalLiteManager {
    /// Create a new manager from the set of allowed command prefixes.
    pub(crate) fn new(allowed_prefixes: HashSet<String>) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            allowed_prefixes,
        }
    }

    /// Return the allowed command prefixes as a sorted `Vec<String>`.
    /// Useful when constructing a new channel that needs the same prefixes.
    pub(crate) fn allowed_prefixes(&self) -> Vec<String> {
        self.allowed_prefixes.iter().cloned().collect()
    }

    /// Returns `true` if there is an active terminal-lite session for the
    /// given chat id.
    #[allow(dead_code)]
    pub(crate) async fn has_active_session(&self, chat_id: i64) -> bool {
        self.sessions.lock().await.contains_key(&chat_id)
    }

    /// Start (or restart) a terminal-lite session.  Returns a human-readable
    /// status string suitable for sending back to the user.
    pub(crate) async fn start_session(
        &self,
        chat_id: i64,
        user_id: u64,
        args: Vec<String>,
    ) -> String {
        let mut args = args;

        let subcommand = args
            .first()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| "start".to_string());

        if subcommand == "help" {
            return terminal_lite_help_text();
        }

        if subcommand == "status" {
            return self.get_status(chat_id).await;
        }

        if subcommand == "stop" {
            return self.stop_session(chat_id).await;
        }

        if subcommand == "start" {
            args.remove(0);
        }

        let preferred_agent = args
            .first()
            .and_then(|value| crate::normalize_terminal_agent_name(value))
            .map(|s| s.to_string());
        if preferred_agent.is_some() && !args.is_empty() {
            args.remove(0);
        }

        let cwd_arg = if args.is_empty() {
            None
        } else {
            Some(args.join(" "))
        };
        let cwd = match resolve_terminal_lite_cwd(cwd_arg.as_deref()) {
            Ok(v) => v,
            Err(err) => {
                return format!("Failed to start terminal lite: {}", err);
            }
        };

        let shell = default_terminal_lite_shell();
        let session = TerminalLiteSession {
            owner_user_id: user_id,
            cwd: cwd.clone(),
            shell: shell.clone(),
            preferred_agent: preferred_agent.clone(),
            started_at: Instant::now(),
            busy: false,
        };

        {
            let mut sessions = self.sessions.lock().await;
            sessions.insert(chat_id, session);
        }

        format!(
            "Terminal lite started.\nWorking dir: {}\nShell: {}\nPreferred agent: {}\n\nSend commands as chat messages.\nUse `/terminal lite stop` to stop.",
            cwd.display(),
            shell,
            preferred_agent.as_deref().unwrap_or("none")
        )
    }

    /// Stop the terminal-lite session for the given chat id.
    pub(crate) async fn stop_session(&self, chat_id: i64) -> String {
        let mut sessions = self.sessions.lock().await;
        if sessions.remove(&chat_id).is_some() {
            "Terminal lite stopped for this chat.".to_string()
        } else {
            "Terminal lite is not active.".to_string()
        }
    }

    /// Return a human-readable status line for the session.
    pub(crate) async fn get_status(&self, chat_id: i64) -> String {
        let sessions = self.sessions.lock().await;
        if let Some(session) = sessions.get(&chat_id) {
            let elapsed = session.started_at.elapsed().as_secs();
            format!(
                "Terminal lite is active.\nOwner: {}\nWorking dir: {}\nShell: {}\nPreferred agent: {}\nBusy: {}\nUptime: {}s",
                session.owner_user_id,
                session.cwd.display(),
                session.shell,
                session.preferred_agent.as_deref().unwrap_or("none"),
                if session.busy { "yes" } else { "no" },
                elapsed
            )
        } else {
            "Terminal lite is not active. Start with `/terminal lite start`.".to_string()
        }
    }

    /// Validate a command against the allowed prefixes.
    pub(crate) fn validate_terminal_lite_command(&self, text: &str) -> Result<(), String> {
        if self.allowed_prefixes.contains("*") {
            return Ok(());
        }
        if self.allowed_prefixes.is_empty() {
            return Err(
                "Terminal lite is disabled because `[terminal].allowed_prefixes` is empty."
                    .to_string(),
            );
        }

        let raw = extract_terminal_lite_command_name(text)
            .ok_or_else(|| "Could not determine command name.".to_string())?;
        let command_name = Path::new(&raw)
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or(raw.as_str())
            .trim()
            .to_ascii_lowercase();

        if command_name.is_empty() {
            return Err("Could not determine command name.".to_string());
        }
        if contains_shell_control_operators(text) {
            return Err(
                "Shell operators are not allowed in `/terminal lite` commands (use `/terminal` full mode)."
                    .to_string(),
            );
        }
        if self.allowed_prefixes.contains(&command_name) {
            return Ok(());
        }

        let mut allowed = self
            .allowed_prefixes
            .iter()
            .filter(|value| value.as_str() != "*")
            .cloned()
            .collect::<Vec<_>>();
        allowed.sort();
        Err(format!(
            "Command `{}` is not allowed in `/terminal lite`.\nAllowed commands: {}",
            command_name,
            allowed.join(", ")
        ))
    }

    /// Handle a line of user input for an active terminal-lite session.
    /// Returns `Some(reply)` if the chat has an active session (or an error
    /// message), `None` if there is no session for this chat.
    pub(crate) async fn handle_input(
        &self,
        chat_id: i64,
        user_id: u64,
        user_role: UserRole,
        text: &str,
    ) -> Option<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut sessions = self.sessions.lock().await;
        let session = sessions.get_mut(&chat_id)?;

        if user_role != UserRole::Owner || session.owner_user_id != user_id {
            return Some("Only the owner can use terminal lite in this chat.".to_string());
        }

        if session.busy {
            return Some(
                "Terminal lite is busy with the previous command. Wait or run `/terminal lite stop`."
                    .to_string(),
            );
        }

        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            sessions.remove(&chat_id);
            return Some("Terminal lite stopped.".to_string());
        }

        if let Some(reply) = terminal_lite_try_handle_cd(session, trimmed) {
            return Some(reply);
        }

        if let Some(agent) = is_terminal_lite_interactive_agent_command(trimmed) {
            return Some(format!(
                "`{}` is an interactive TUI and is not supported in `/terminal lite`.\nUse full mode instead: `/terminal {} {}`",
                agent,
                agent,
                session.cwd.display()
            ));
        }

        if let Err(err) = self.validate_terminal_lite_command(trimmed) {
            return Some(err);
        }

        let snapshot = session.clone();
        session.busy = true;
        drop(sessions);

        let reply = run_terminal_lite_command(&snapshot, trimmed).await;

        let mut sessions = self.sessions.lock().await;
        if let Some(active) = sessions.get_mut(&chat_id) {
            active.busy = false;
        }
        Some(reply)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_lite_detects_shell_control_operators() {
        assert!(contains_shell_control_operators("ls && rm -rf /tmp/demo"));
        assert!(contains_shell_control_operators("echo $(whoami)"));
        assert!(contains_shell_control_operators("echo `whoami`"));
    }

    #[test]
    fn terminal_lite_allows_simple_commands_without_operators() {
        assert!(!contains_shell_control_operators("ls -la /tmp"));
        assert!(!contains_shell_control_operators("FOO=bar env"));
        assert!(!contains_shell_control_operators("echo ';' '|' '>'"));
    }
}
