use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, Mutex, Semaphore};
use tracing::{info, warn};
use uuid::Uuid;

use super::{
    command_risk::{PermissionMode, RiskLevel},
    daemon_guard::detect_daemonization_primitives,
};
use crate::agent::{
    derive_executor_step_result, persist_executor_handoff_context, persist_executor_result_context,
    ExecutorHandoff, TargetScope, TaskValidationOutcome,
};
use crate::config::CliAgentsConfig;
use crate::execution::{
    active_execution_backend, BackendFileType, BackendPath, ExecutionRequest, ProcessHandle,
    SharedExecutionBackend,
};
use crate::llm_runtime::SharedLlmRuntime;
use crate::runtime_ports::{ConversationRequest, ConversationRuntime, OutboundRouter};
use crate::tools::terminal::ApprovalRequest;
use crate::tools::ApprovalBroker;
use crate::traits::{
    DynamicCliAgent, Message, ModelProvider, StateStore, Tool, ToolCallMetadata, ToolCallOutcome,
    ToolCallSemantics, ToolCapabilities, ToolExecutionContext, ToolOutcomeStatus, ToolTargetHint,
    ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::ApprovalResponse;
use crate::types::StatusUpdate;
use crate::utils::{floor_char_boundary, truncate_str, truncate_with_note};

/// Max bytes for output buffer (1 MB) to prevent unbounded memory growth.
const BUFFER_CAP: usize = 1_048_576;
const BUFFER_TRUNCATION_MARKER: &str = "\n[... middle of CLI output omitted ...]\n";

/// Append one output line while preserving both startup diagnostics and the
/// most recent events. Final-result records are normally at the end of JSONL
/// streams, so a prefix-only cap can erase the authoritative outcome.
fn append_bounded_line(buffer: &mut String, prefix: &str, line: &str, cap: usize) {
    if cap == 0 {
        return;
    }

    buffer.push_str(prefix);
    buffer.push_str(line);
    buffer.push('\n');
    if buffer.len() <= cap {
        return;
    }

    if cap <= BUFFER_TRUNCATION_MARKER.len() {
        let mut tail_start = buffer.len().saturating_sub(cap);
        while tail_start < buffer.len() && !buffer.is_char_boundary(tail_start) {
            tail_start += 1;
        }
        *buffer = buffer[tail_start..].to_string();
        return;
    }

    let head_end = floor_char_boundary(buffer, cap / 4);
    let tail_budget = cap - head_end - BUFFER_TRUNCATION_MARKER.len();
    let mut tail_start = buffer.len().saturating_sub(tail_budget);
    while tail_start < buffer.len() && !buffer.is_char_boundary(tail_start) {
        tail_start += 1;
    }

    let mut retained = String::with_capacity(cap);
    retained.push_str(&buffer[..head_end]);
    retained.push_str(BUFFER_TRUNCATION_MARKER);
    retained.push_str(&buffer[tail_start..]);
    *buffer = retained;
}

/// Interval for emitting progress updates (avoid spamming the channel).
#[cfg(test)]
const PROGRESS_INTERVAL: Duration = Duration::from_millis(25);
#[cfg(not(test))]
const PROGRESS_INTERVAL: Duration = Duration::from_secs(2);

/// Cadence for durable, chat-visible updates after a CLI task has detached from
/// its original request. Foreground status streams remain more frequent.
#[cfg(test)]
const BACKGROUND_CHAT_PROGRESS_INTERVAL: Duration = Duration::from_millis(100);
#[cfg(not(test))]
const BACKGROUND_CHAT_PROGRESS_INTERVAL: Duration = Duration::from_secs(120);

/// Loop detection: window size for tracking recent lines
const LOOP_DETECTION_WINDOW: usize = 100;

/// Loop detection: threshold - if same line appears this many times in window, it's a loop
const LOOP_DETECTION_THRESHOLD: usize = 50;

/// Max concurrent CLI agent processes
const DEFAULT_MAX_CONCURRENT: usize = 3;

/// Max enriched prompt size (16 KB)
const MAX_PROMPT_SIZE: usize = 16384;

/// Max git diff size to append to results (4 KB)
const MAX_DIFF_SIZE: usize = 4096;

fn format_background_progress(
    tool_name: &str,
    task_context: &str,
    elapsed_secs: u64,
    latest_activity: Option<&str>,
) -> String {
    let mut message = format!(
        "Background task is still running ({}s).\nWorker: {}\nTask: {}",
        elapsed_secs,
        tool_name,
        truncate_with_note(task_context, 240),
    );
    if let Some(activity) = latest_activity.filter(|activity| !activity.trim().is_empty()) {
        message.push_str("\nLatest activity: ");
        message.push_str(&truncate_with_note(activity.trim(), 240));
    }
    message
}

async fn deliver_cli_agent_notification(
    hub: Option<&Arc<dyn OutboundRouter>>,
    state: &Arc<dyn StateStore>,
    goal_id: &str,
    session_id: &str,
    notification_type: &str,
    message: &str,
    context: &str,
) -> bool {
    if session_id.trim().is_empty() {
        warn!("{context}: no session available; update dropped");
        return false;
    }

    let message = crate::channels::present_notification(notification_type, message);
    if let Some(hub) = hub {
        match hub.send_text(session_id, &message).await {
            Ok(()) => return true,
            Err(e) => warn!(
                session_id = %session_id,
                goal_id = %goal_id,
                notification_type = %notification_type,
                error = %e,
                "{context}: direct hub delivery failed"
            ),
        }
    }

    let queue_goal_id = if goal_id.trim().is_empty() {
        "global"
    } else {
        goal_id
    };
    let entry = crate::traits::NotificationEntry::new(
        queue_goal_id,
        session_id,
        notification_type,
        &message,
    );
    match state.enqueue_notification(&entry).await {
        Ok(()) => true,
        Err(e) => {
            warn!(
                session_id = %session_id,
                goal_id = %goal_id,
                notification_type = %notification_type,
                error = %e,
                "{context}: enqueue fallback failed"
            );
            false
        }
    }
}

/// Tracks recent output lines to detect infinite loops
struct LoopDetector {
    recent_lines: Vec<u64>, // Store hashes to save memory
    line_counts: HashMap<u64, usize>,
}

impl LoopDetector {
    fn new() -> Self {
        Self {
            recent_lines: Vec::with_capacity(LOOP_DETECTION_WINDOW),
            line_counts: HashMap::new(),
        }
    }

    /// Add a line and return true if an infinite loop is detected
    fn add_line(&mut self, line: &str) -> bool {
        // Hash the line (normalized - trim whitespace)
        let normalized = line.trim();
        if normalized.is_empty() {
            return false; // Don't count empty lines
        }

        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        let hash = hasher.finish();

        // Add to window
        self.recent_lines.push(hash);
        *self.line_counts.entry(hash).or_insert(0) += 1;

        // Remove old lines if window is full
        if self.recent_lines.len() > LOOP_DETECTION_WINDOW {
            let old_hash = self.recent_lines.remove(0);
            if let Some(count) = self.line_counts.get_mut(&old_hash) {
                *count -= 1;
                if *count == 0 {
                    self.line_counts.remove(&old_hash);
                }
            }
        }

        // Check if any line appears too frequently
        self.line_counts
            .values()
            .any(|&count| count >= LOOP_DETECTION_THRESHOLD)
    }

    /// Get the most repeated line pattern for error reporting
    fn get_loop_pattern(&self) -> Option<usize> {
        self.line_counts.values().max().copied()
    }
}

async fn kill_process(backend: &SharedExecutionBackend, handle: &ProcessHandle) {
    let _ = backend.terminate(handle, Duration::from_secs(2)).await;
}

struct CliToolEntry {
    command: String,
    args: Vec<String>,
    description: String,
    timeout: Duration,
    max_output_chars: usize,
    /// Whether this was dynamically added (vs discovered at startup)
    is_dynamic: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliWorkspaceMode {
    ReadOnly,
    ReadWrite,
}

/// Convert a configured CLI launch into a provider-native, enforceable
/// read-only invocation. Prompt text is deliberately not part of this
/// decision: if the adapter cannot establish a hard sandbox, delegation is
/// rejected before the child process starts.
fn apply_read_only_cli_adapter(command: &str, args: &mut Vec<String>) -> Result<(), String> {
    let program_index = is_env_launcher(command)
        .then(|| env_wrapped_program_index(args))
        .flatten();
    let program = program_index
        .and_then(|index| args.get(index).map(String::as_str))
        .unwrap_or(command);
    let program_name = std::path::Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(program);

    if !program_name.eq_ignore_ascii_case("codex") {
        return Err(format!(
            "CLI agent executable '{program_name}' has no registered hard read-only adapter"
        ));
    }

    let mut constrained = Vec::with_capacity(args.len().saturating_add(3));
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let drops_value = matches!(
            argument.as_str(),
            "--sandbox" | "-s" | "--add-dir" | "--output-last-message" | "-o"
        );
        let drops_flag = matches!(
            argument.as_str(),
            "--dangerously-bypass-approvals-and-sandbox"
                | "--dangerously-bypass-hook-trust"
                | "--full-auto"
                | "--ignore-user-config"
        ) || argument.starts_with("--sandbox=")
            || argument.starts_with("-s=")
            || argument.starts_with("--add-dir=")
            || argument.starts_with("--output-last-message=");
        if drops_value {
            index = index.saturating_add(2);
            continue;
        }
        if drops_flag {
            index = index.saturating_add(1);
            continue;
        }
        constrained.push(argument.clone());
        index = index.saturating_add(1);
    }
    constrained.push("--ignore-user-config".to_string());
    constrained.push("--sandbox".to_string());
    constrained.push("read-only".to_string());
    *args = constrained;
    Ok(())
}

/// A running CLI agent being tracked.
struct RunningCliAgent {
    tool_name: String,
    prompt_summary: String,
    started_at: Instant,
    /// Combined output for display (includes [stderr] prefixes)
    display_buf: Arc<Mutex<String>>,
    /// Pure stdout for JSON extraction
    stdout_buf: Arc<Mutex<String>>,
    /// Backend-scoped handle for status display and cancellation.
    process_handle: ProcessHandle,
    /// Set by the stream/wait task after the backend process exits.
    finished: Arc<AtomicBool>,
    /// Session ID for filtering cancel_all by session
    session_id: String,
    /// Delegated task ID when this cli_agent run is acting as an executor.
    delegated_task_id: Option<String>,
    /// Working directory for git diff capture
    working_dir: Option<String>,
}

/// Finalized background CLI task output retained briefly so `action="check"`
/// can still return results after automatic reaping.
#[derive(Clone)]
struct CompletedCliAgent {
    result: String,
    completed_at: Instant,
    session_id: String,
}

#[derive(Debug, PartialEq, Eq)]
struct CliCompletionResult {
    success: bool,
    authentication_failed: bool,
    persisted_output: String,
    response: Option<String>,
    error: Option<String>,
}

/// A configured CLI could not execute any part of the delegated task, so it is
/// safe to retry the same request with another configured CLI. This remains a
/// typed internal signal: the caller must not infer retryability from the
/// wording of a tool error.
#[derive(Debug)]
struct CliAgentUnavailableError {
    tool_name: String,
    message: String,
}

impl std::fmt::Display for CliAgentUnavailableError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CliAgentUnavailableError {}

/// A working directory claim with enough metadata to explain conflicts.
struct WorkingDirClaim {
    task_id: String,
    tool_name: String,
    prompt_summary: String,
    dedup_prompt: String,
}

/// The claims map uses a std (not tokio) Mutex so that WorkingDirClaimGuard
/// can release a claim in Drop (Drop cannot await). Never hold this lock
/// across an await point; the compiler enforces it for Send futures.
type WorkingDirClaims = std::sync::Mutex<HashMap<String, WorkingDirClaim>>;

/// Lock the claims map, recovering from poisoning (a panic while holding the
/// lock must not permanently wedge working-dir dispatch).
fn lock_claims(
    claims: &WorkingDirClaims,
) -> std::sync::MutexGuard<'_, HashMap<String, WorkingDirClaim>> {
    claims
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// RAII guard that releases a working-dir claim on drop unless disarmed.
///
/// Dispatch futures can be dropped at any await point when the calling
/// session is cancelled or times out. Before this guard existed, an abort
/// between claim insertion and the explicit post-await release leaked the
/// claim permanently, blocking all future dispatches to that directory
/// until a daemon restart. The guard makes release abort-safe.
///
/// Disarm when handing the task off to the background `running` map: from
/// that point the reaper / cancel paths own the release.
struct WorkingDirClaimGuard {
    claims: Arc<WorkingDirClaims>,
    dir: String,
    task_id: String,
    armed: bool,
}

impl WorkingDirClaimGuard {
    fn new(claims: Arc<WorkingDirClaims>, dir: String, task_id: String) -> Self {
        Self {
            claims,
            dir,
            task_id,
            armed: true,
        }
    }

    /// Stop the guard from releasing on drop (ownership of the claim has
    /// been transferred to the background `running` map).
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for WorkingDirClaimGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut claims = lock_claims(&self.claims);
        let owned = claims
            .get(&self.dir)
            .map(|claim| claim.task_id == self.task_id)
            .unwrap_or(false);
        if owned {
            claims.remove(&self.dir);
        }
    }
}

/// Jaccard similarity between two strings based on word-level bigrams.
/// Returns a value in [0.0, 1.0].
fn prompt_similarity(a: &str, b: &str) -> f64 {
    fn bigrams(s: &str) -> HashSet<(String, String)> {
        let words: Vec<&str> = s.split_whitespace().collect();
        if words.len() < 2 {
            // For single-word prompts, fall back to unigram set
            return words
                .into_iter()
                .map(|w| (w.to_string(), String::new()))
                .collect();
        }
        words
            .windows(2)
            .map(|w| (w[0].to_string(), w[1].to_string()))
            .collect()
    }

    let set_a = bigrams(&a.to_lowercase());
    let set_b = bigrams(&b.to_lowercase());

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0; // Both empty → identical
    }
    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Build a dedup key from a prompt: first 200 chars, lowercased.
fn make_dedup_prompt(prompt: &str) -> String {
    prompt.chars().take(200).collect::<String>().to_lowercase()
}

pub struct CliAgentTool {
    backend: SharedExecutionBackend,
    // MUST be std::sync::RwLock because schema() is sync
    tools: Arc<std::sync::RwLock<HashMap<String, CliToolEntry>>>,
    tool_names: Arc<std::sync::RwLock<Vec<String>>>,
    running: Arc<Mutex<HashMap<String, RunningCliAgent>>>, // task_id -> RunningCliAgent
    completed: Arc<Mutex<HashMap<String, CompletedCliAgent>>>, // task_id -> finalized output
    working_dir_claims: Arc<WorkingDirClaims>, // normalized dir -> claim (std Mutex: see WorkingDirClaimGuard)
    state: Arc<dyn StateStore>,
    #[allow(dead_code)] // Reserved for future interactive feedback
    llm_runtime: SharedLlmRuntime,
    default_timeout: Duration,
    default_max_output: usize,
    max_concurrent: usize,
    concurrency_limiter: Arc<Semaphore>,
    approval_tx: ApprovalBroker,
    hub: OnceLock<Weak<dyn OutboundRouter>>,
    /// Re-enter the root agent when background delegation completes so the
    /// delegated step advances the original user outcome instead of ending it.
    agent: OnceLock<Weak<dyn ConversationRuntime>>,
    reengagements: Arc<Mutex<HashMap<String, VecDeque<Instant>>>>,
}

async fn list_backend_files(
    backend: SharedExecutionBackend,
    root: BackendPath,
    skip: &HashSet<&str>,
    max_depth: usize,
    cap: usize,
) -> Vec<String> {
    let mut files = Vec::new();
    let mut queue = std::collections::VecDeque::from([(root.clone(), String::new(), 0usize)]);
    while let Some((directory, prefix, depth)) = queue.pop_front() {
        let Ok(entries) = backend.read_dir(&directory).await else {
            continue;
        };
        for entry in entries {
            if files.len() >= cap {
                return files;
            }
            let name = entry.path.file_name().unwrap_or_default();
            if skip.contains(name) {
                continue;
            }
            let relative = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}/{name}")
            };
            match entry.metadata.file_type {
                BackendFileType::File => files.push(relative),
                BackendFileType::Directory if depth + 1 < max_depth => {
                    queue.push_back((entry.path, relative, depth + 1));
                }
                _ => {}
            }
        }
    }
    files
}

/// Default tool definitions when the user enables cli_agents but doesn't specify tools.
fn default_tool_definitions() -> Vec<(&'static str, &'static str, Vec<&'static str>, &'static str)>
{
    vec![
        (
            "claude",
            "claude",
            vec![
                "-p",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "--verbose",
            ],
            "Claude Code — Anthropic's AI coding agent (auto-approve mode)",
        ),
        (
            "gemini",
            "gemini",
            vec![
                "--sandbox=false",
                "--yolo",
                "--output-format",
                "stream-json",
            ],
            "Gemini CLI — Google's AI coding agent (auto-approve mode)",
        ),
        (
            "codex",
            "codex",
            vec![
                "exec",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
            ],
            "Codex CLI — OpenAI's AI coding agent (auto-approve mode)",
        ),
        (
            "copilot",
            "copilot",
            vec!["-p", "--allow-all-tools", "--allow-all-paths"],
            "GitHub Copilot CLI (auto-approve mode)",
        ),
        (
            "aider",
            "aider",
            vec!["--yes", "--message"],
            "Aider — AI pair programming",
        ),
    ]
}

/// Preferred fallback order when `action=run` omits `tool`.
///
/// Rationale: keep common coding agents at the front so orchestrator calls can
/// simply provide a prompt and still use a strong default.
const DEFAULT_TOOL_PRIORITY: &[&str] = &["claude", "gemini", "codex", "copilot", "aider"];

fn env_wrapped_program_index(args: &[String]) -> Option<usize> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        match arg {
            "--" => return (index + 1 < args.len()).then_some(index + 1),
            "-u" | "--unset" | "-C" | "--chdir" => index += 2,
            "-S" | "--split-string" => return None,
            _ if arg.starts_with("--unset=") || arg.starts_with("--chdir=") => index += 1,
            _ if arg.starts_with('-') => index += 1,
            _ if arg
                .split_once('=')
                .is_some_and(|(name, _)| !name.is_empty() && !name.contains('/')) =>
            {
                index += 1
            }
            _ => return Some(index),
        }
    }
    None
}

fn is_env_launcher(command: &str) -> bool {
    std::path::Path::new(command)
        .file_name()
        .and_then(|name| name.to_str())
        == Some("env")
}

async fn executable_search_candidates(
    backend: &SharedExecutionBackend,
    executable: &str,
) -> Vec<String> {
    if executable.contains('/') || executable.contains('\\') {
        return vec![executable.to_string()];
    }

    let mut candidates = vec![executable.to_string()];
    if let Ok(home) = backend.home_dir().await {
        for relative_dir in [
            ".local/bin",
            ".cargo/bin",
            ".npm-global/bin",
            "miniforge3/bin",
            "anaconda3/bin",
        ] {
            candidates.push(
                home.join(relative_dir)
                    .join(executable)
                    .as_str()
                    .to_string(),
            );
        }
    }
    for system_dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"] {
        candidates.push(format!("{system_dir}/{executable}"));
    }
    candidates.dedup();
    candidates
}

async fn resolve_executable(backend: &SharedExecutionBackend, executable: &str) -> Option<String> {
    for candidate in executable_search_candidates(backend, executable).await {
        if backend.executable_exists(&candidate).await.unwrap_or(false) {
            return Some(candidate);
        }
    }
    None
}

/// Resolve both the configured launcher and, for `/usr/bin/env` wrappers, the
/// actual nested agent executable. GUI/launchd processes commonly lack user
/// install directories such as `~/.local/bin` in PATH; accepting `env` alone
/// used to defer that mistake until every invocation exited with code 127.
async fn resolve_agent_launch(
    backend: &SharedExecutionBackend,
    command: &str,
    args: &[String],
) -> Result<(String, Vec<String>), String> {
    let resolved_command = resolve_executable(backend, command)
        .await
        .ok_or_else(|| format!("launcher executable '{command}' was not found"))?;
    let mut resolved_args = args.to_vec();

    if is_env_launcher(&resolved_command) {
        if let Some(index) = env_wrapped_program_index(&resolved_args) {
            let nested = resolved_args[index].clone();
            let resolved_nested = resolve_executable(backend, &nested).await.ok_or_else(|| {
                format!(
                    "wrapped CLI agent executable '{nested}' was not found in PATH or common user install directories"
                )
            })?;
            resolved_args[index] = resolved_nested;
        }
    }

    Ok((resolved_command, resolved_args))
}

fn format_cli_agent_failure(
    tool_name: &str,
    exit_code: Option<i32>,
    display_output: &str,
    max_output: usize,
    diff_section: Option<&str>,
) -> String {
    let captured = truncate_with_note(display_output, max_output);
    let details = if captured.trim().is_empty() {
        "No stdout or stderr was captured. The process terminated before it produced diagnostics."
            .to_string()
    } else {
        captured.trim_end().to_string()
    };
    let mut error_msg = format!(
        "ERROR: CLI agent '{tool_name}' failed (exit code {}).\n\n## Failure Details\n{details}",
        exit_code
            .map(|code| code.to_string())
            .unwrap_or_else(|| "unavailable".to_string())
    );
    if let Some(diff) = diff_section {
        error_msg.push_str(diff);
    }
    error_msg.push_str(
        "\n\n## Safe Recovery Options\n\
         - Verify the configured executable path and authentication\n\
         - Try a different CLI agent\n\
         - Inspect any reported file changes before deciding whether to revert them",
    );
    error_msg
}

fn cli_agent_exit_code(output: &str) -> Option<i32> {
    let marker = "failed (exit code ";
    let suffix = output.split_once(marker)?.1;
    let raw = suffix.split_once(')')?.0.trim();
    raw.strip_prefix("Some(")
        .unwrap_or(raw)
        .trim_end_matches(')')
        .parse()
        .ok()
}

fn failed_before_agent_launch(output: &str, exit_code: Option<i32>) -> bool {
    let lower = output.to_ascii_lowercase();
    lower.contains("could not start")
        || (exit_code == Some(127)
            && (lower.contains("[stderr] env:") || lower.contains("[stderr] /usr/bin/env:"))
            && lower.contains("no such file or directory"))
}

fn first_cli_failure_detail(output: &str) -> String {
    output
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("[stderr]") && line.len() > "[stderr]".len())
        .or_else(|| {
            output.lines().map(str::trim).find(|line| {
                !line.is_empty()
                    && !line.starts_with('#')
                    && !line.starts_with("[UNTRUSTED")
                    && !line.starts_with("[END UNTRUSTED")
            })
        })
        .unwrap_or("CLI agent process failed")
        .chars()
        .take(300)
        .collect()
}

impl CliAgentTool {
    async fn persist_delegated_cli_result_with_state(
        state: Arc<dyn StateStore>,
        delegated_task_id: &str,
        response: Option<&str>,
        error: Option<&str>,
    ) {
        let latest_task = state.get_task(delegated_task_id).await.ok().flatten();
        let structured =
            derive_executor_step_result(delegated_task_id, latest_task.as_ref(), response, error);
        let task_lead_summary = structured.render_task_lead_summary();

        if let Ok(Some(mut task)) = state.get_task(delegated_task_id).await {
            if let Ok(context) =
                persist_executor_result_context(task.context.as_deref(), &structured)
            {
                task.context = Some(context);
            }

            match error {
                Some(error) => {
                    task.status = "failed".to_string();
                    task.error = Some(error.to_string());
                    task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                }
                None => {
                    if matches!(structured.task_outcome, TaskValidationOutcome::TaskDone) {
                        if task
                            .result
                            .as_deref()
                            .is_none_or(|result| result.trim().is_empty())
                        {
                            if let Some(response) = response {
                                task.result = Some(response.to_string());
                            } else {
                                task.result = Some(structured.summary.clone());
                            }
                        }
                        task.status = "completed".to_string();
                        task.blocker = None;
                    } else {
                        task.result = Some(task_lead_summary.clone());
                        task.status = "blocked".to_string();
                        task.blocker = structured
                            .blocker
                            .clone()
                            .or_else(|| structured.exact_need.clone())
                            .or_else(|| Some(structured.summary.clone()));
                    }
                    task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                }
            }

            let _ = state.update_task(&task).await;
        }
    }

    fn prune_completed_map(completed: &mut HashMap<String, CompletedCliAgent>) {
        // Keep results for as long as dialogue-state can keep an unfinished
        // request open. A ten-minute cache made ordinary delayed follow-ups
        // lose successful delegated work and turn a recoverable typo into a
        // dead end.
        const COMPLETED_TTL: Duration = Duration::from_secs(12 * 60 * 60);
        const COMPLETED_CAP: usize = 128;

        completed.retain(|_, entry| entry.completed_at.elapsed() <= COMPLETED_TTL);
        if completed.len() <= COMPLETED_CAP {
            return;
        }

        let mut by_age: Vec<(String, Instant)> = completed
            .iter()
            .map(|(task_id, entry)| (task_id.clone(), entry.completed_at))
            .collect();
        by_age.sort_by_key(|(_, ts)| *ts);
        let to_remove = by_age.len().saturating_sub(COMPLETED_CAP);
        for (task_id, _) in by_age.into_iter().take(to_remove) {
            completed.remove(&task_id);
        }
    }

    async fn build_finished_result(agent: &RunningCliAgent) -> String {
        let elapsed = agent.started_at.elapsed().as_secs();
        let stdout_output = agent.stdout_buf.lock().await.clone();
        let display_output = agent.display_buf.lock().await.clone();
        if let Some(error) = Self::detect_auth_error(&display_output, &agent.tool_name) {
            return format!(
                "CLI agent '{}' failed after {}s.\n\n{}",
                agent.tool_name, elapsed, error
            );
        }
        let result = extract_meaningful_output(&stdout_output, 10000);

        // Capture git diff for finished background tasks.
        let diff_section = if let Some(ref dir) = agent.working_dir {
            Self::capture_git_diff(dir)
                .await
                .map(|diff| format!("\n\n## File Changes\n```diff\n{}\n```", diff))
        } else {
            None
        };

        let mut final_result = format!(
            "CLI agent '{}' finished after {}s.\n\nResult:\n{}",
            agent.tool_name, elapsed, result
        );
        if let Some(diff) = diff_section {
            final_result.push_str(&diff);
        }
        final_result
    }

    fn build_background_reengagement_followup(
        tool_name: &str,
        task_id: &str,
        task_context: &str,
        status: &str,
        output: &str,
    ) -> String {
        format!(
            "[Background command completed]\n\
             Command: `cli_agent {tool_name}`\n\
             Runtime task ID: {task_id}\n\
             Delegated step: {task_context}\n\
             Exit status: {status}\n\
             Output:\n{output}\n\n\
             This delegated step was part of the user's previous unfinished request. Check the session history for the requested outcome and continue from this evidence. Delegated work is progress, not automatically fulfillment: complete every remaining implementation, action, validation, or delivery step that the original request requires. Do not merely announce that the delegated step ended, and do not ask the user to repeat the request."
        )
    }

    fn tool_priority(name: &str) -> usize {
        let lower = name.to_ascii_lowercase();
        DEFAULT_TOOL_PRIORITY
            .iter()
            .position(|known| *known == lower)
            .unwrap_or(DEFAULT_TOOL_PRIORITY.len())
    }

    /// Pick a default tool if caller omits `tool`.
    /// Preference: known coding CLIs first, then lexicographic order.
    fn default_tool_name(&self) -> Option<String> {
        let tools = self.tools.read().unwrap();
        if tools.is_empty() {
            return None;
        }

        let mut names: Vec<String> = tools.keys().cloned().collect();
        names.sort_by(|a, b| {
            Self::tool_priority(a)
                .cmp(&Self::tool_priority(b))
                .then_with(|| a.cmp(b))
        });
        names.into_iter().next()
    }

    fn configured_tool(&self, name: &str) -> Option<String> {
        self.tools
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(name)
            .then(|| name.to_string())
    }

    fn is_owner_role(user_role: Option<&str>) -> bool {
        user_role.is_some_and(|role| role.eq_ignore_ascii_case("owner"))
    }

    /// Normalize a working directory for lock/key comparisons.
    /// Uses canonical paths when possible so aliases like `/repo` and `/repo/`
    /// map to the same lock key.
    async fn normalize_working_dir(dir: &str) -> anyhow::Result<String> {
        let backend = active_execution_backend();
        let resolved = backend.resolve_path(dir).await?;
        Ok(backend
            .canonicalize(&resolved)
            .await
            .unwrap_or(resolved)
            .to_string())
    }

    /// Release a working-directory claim if it belongs to the given task.
    fn release_working_dir_claim(&self, dir: &str, task_id: &str) {
        let mut claims = lock_claims(&self.working_dir_claims);
        let should_remove = claims
            .get(dir)
            .map(|claim| claim.task_id == task_id)
            .unwrap_or(false);
        if should_remove {
            claims.remove(dir);
        }
    }

    async fn request_daemonization_approval(
        &self,
        session_id: &str,
        tool_name: &str,
        prompt: &str,
        hits: &[&str],
    ) -> anyhow::Result<ApprovalResponse> {
        let prompt_preview: String = prompt.chars().take(180).collect();
        let command = format!(
            "cli_agent '{}' requested detached/background execution markers: {}. Prompt preview: {}",
            tool_name,
            hits.join(", "),
            prompt_preview
        );
        let warnings = vec![
            format!(
                "Execution target: {} ({}) workspace {}",
                self.backend.kind().as_str(),
                self.backend.id(),
                self.backend.workspace_root()
            ),
            format!("Daemonization primitives detected: {}", hits.join(", ")),
            "Detached/background processes may survive cancellation and continue running."
                .to_string(),
        ];

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command,
                session_id: session_id.to_string(),
                risk_level: RiskLevel::Critical,
                warnings,
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: crate::types::ApprovalKind::CommandOnce,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;

        match tokio::time::timeout(std::time::Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(anyhow::anyhow!("Approval response channel closed")),
            Err(_) => Err(anyhow::anyhow!("Approval request timed out after 300s")),
        }
    }

    pub async fn discover(
        config: CliAgentsConfig,
        state: Arc<dyn StateStore>,
        llm_runtime: SharedLlmRuntime,
        approval_tx: ApprovalBroker,
    ) -> Self {
        let default_timeout = Duration::from_secs(config.timeout_secs);
        let default_max_output = config.max_output_chars;

        type ToolCandidate = (
            String,
            String,
            Vec<String>,
            String,
            Option<u64>,
            Option<usize>,
        );
        let mut candidates: Vec<ToolCandidate> = Vec::new();

        if config.tools.is_empty() {
            for (name, cmd, args, desc) in default_tool_definitions() {
                candidates.push((
                    name.to_string(),
                    cmd.to_string(),
                    args.into_iter().map(|s| s.to_string()).collect(),
                    desc.to_string(),
                    None,
                    None,
                ));
            }
        } else {
            for (name, tool_cfg) in &config.tools {
                candidates.push((
                    name.clone(),
                    tool_cfg.command.clone(),
                    tool_cfg.args.clone(),
                    tool_cfg.description.clone(),
                    tool_cfg.timeout_secs,
                    tool_cfg.max_output_chars,
                ));
            }
        }

        // Resolve launchers in parallel. This validates a nested executable in
        // `env ... agent` configurations instead of merely proving that
        // `/usr/bin/env` exists.
        let backend = active_execution_backend();
        let resolution_futures: Vec<_> = candidates
            .iter()
            .map(|(_, command, args, _, _, _)| resolve_agent_launch(&backend, command, args))
            .collect();
        let resolution_results = futures::future::join_all(resolution_futures).await;

        let mut tools = HashMap::new();

        for (i, (name, command, _args, description, timeout_override, max_output_override)) in
            candidates.into_iter().enumerate()
        {
            if let Ok((resolved_command, resolved_args)) = &resolution_results[i] {
                info!(name = %name, command = %resolved_command, "CLI agent tool discovered");
                tools.insert(
                    name,
                    CliToolEntry {
                        command: resolved_command.clone(),
                        args: resolved_args.clone(),
                        description,
                        timeout: timeout_override
                            .map(Duration::from_secs)
                            .unwrap_or(default_timeout),
                        max_output_chars: max_output_override.unwrap_or(default_max_output),
                        is_dynamic: false,
                    },
                );
            } else {
                info!(
                    name = %name,
                    command = %command,
                    error = %resolution_results[i].as_ref().unwrap_err(),
                    "CLI agent tool not found, skipping"
                );
            }
        }

        let mut tool_names: Vec<String> = tools.keys().cloned().collect();
        tool_names.sort();

        let tool = CliAgentTool {
            backend,
            tools: Arc::new(std::sync::RwLock::new(tools)),
            tool_names: Arc::new(std::sync::RwLock::new(tool_names)),
            running: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
            state,
            llm_runtime,
            default_timeout,
            default_max_output,
            max_concurrent: DEFAULT_MAX_CONCURRENT,
            concurrency_limiter: Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT)),
            approval_tx,
            hub: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
        };

        // Load dynamic agents from DB
        tool.load_dynamic_agents().await;

        tool
    }

    /// Set channel hub reference for immediate background completion delivery.
    pub fn set_hub(&self, hub: Weak<dyn OutboundRouter>) {
        self.hub
            .set(hub)
            .expect("CliAgentTool::set_hub called more than once");
    }

    pub fn set_agent(&self, agent: Weak<dyn ConversationRuntime>) {
        self.agent
            .set(agent)
            .expect("CliAgentTool::set_agent called more than once");
    }

    fn get_hub(&self) -> Option<Arc<dyn OutboundRouter>> {
        self.hub.get().and_then(|w| w.upgrade())
    }

    fn get_agent(&self) -> Option<Arc<dyn ConversationRuntime>> {
        self.agent.get().and_then(|w| w.upgrade())
    }

    pub(crate) fn wiring_ready(&self) -> bool {
        self.hub.get().and_then(Weak::upgrade).is_some()
            && self.agent.get().and_then(Weak::upgrade).is_some()
    }

    /// Load dynamically registered agents from the database.
    async fn load_dynamic_agents(&self) {
        match self.state.list_dynamic_cli_agents().await {
            Ok(agents) => {
                for agent in agents {
                    if !agent.enabled {
                        continue;
                    }
                    let args: Vec<String> =
                        serde_json::from_str(&agent.args_json).unwrap_or_default();
                    let (command, args) =
                        match resolve_agent_launch(&self.backend, &agent.command, &args).await {
                            Ok(launch) => launch,
                            Err(error) => {
                                info!(
                                    name = %agent.name,
                                    command = %agent.command,
                                    %error,
                                    "Dynamic CLI agent command not found, skipping"
                                );
                                continue;
                            }
                        };
                    let entry = CliToolEntry {
                        command,
                        args,
                        description: agent.description.clone(),
                        timeout: agent
                            .timeout_secs
                            .map(Duration::from_secs)
                            .unwrap_or(self.default_timeout),
                        max_output_chars: agent.max_output_chars.unwrap_or(self.default_max_output),
                        is_dynamic: true,
                    };
                    let mut tools = self.tools.write().unwrap();
                    tools.insert(agent.name.clone(), entry);
                    let mut names = self.tool_names.write().unwrap();
                    if !names.contains(&agent.name) {
                        names.push(agent.name.clone());
                        names.sort();
                    }
                    info!(name = %agent.name, "Loaded dynamic CLI agent from DB");
                }
            }
            Err(e) => {
                warn!("Failed to load dynamic CLI agents: {}", e);
            }
        }
    }

    pub fn has_tools(&self) -> bool {
        !self.tools.read().unwrap().is_empty()
    }

    /// Add a new CLI agent at runtime. Returns error message string on validation failure.
    pub async fn add_agent(
        &self,
        name: &str,
        command: &str,
        args: Vec<String>,
        description: &str,
        timeout_secs: Option<u64>,
        max_output_chars: Option<usize>,
    ) -> anyhow::Result<String> {
        let (resolved_command, resolved_args) =
            match resolve_agent_launch(&self.backend, command, &args).await {
                Ok(launch) => launch,
                Err(error) => return Ok(format!("CLI agent configuration is invalid: {error}.")),
            };

        // Save to database
        let dynamic = DynamicCliAgent {
            id: 0,
            name: name.to_string(),
            command: command.to_string(),
            args_json: serde_json::to_string(&args)?,
            description: description.to_string(),
            timeout_secs,
            max_output_chars,
            enabled: true,
            created_at: String::new(),
        };
        self.state.save_dynamic_cli_agent(&dynamic).await?;

        // Add to runtime map
        let entry = CliToolEntry {
            command: resolved_command,
            args: resolved_args,
            description: description.to_string(),
            timeout: timeout_secs
                .map(Duration::from_secs)
                .unwrap_or(self.default_timeout),
            max_output_chars: max_output_chars.unwrap_or(self.default_max_output),
            is_dynamic: true,
        };
        let mut tools = self.tools.write().unwrap();
        tools.insert(name.to_string(), entry);
        let mut names = self.tool_names.write().unwrap();
        if !names.contains(&name.to_string()) {
            names.push(name.to_string());
            names.sort();
        }

        Ok(format!("CLI agent '{}' added successfully.", name))
    }

    /// Remove a CLI agent by name.
    pub async fn remove_agent(&self, name: &str) -> anyhow::Result<String> {
        // Find and remove from DB
        let agents = self.state.list_dynamic_cli_agents().await?;
        if let Some(agent) = agents.iter().find(|a| a.name == name) {
            self.state.delete_dynamic_cli_agent(agent.id).await?;
        }

        // Remove from runtime map
        let mut tools = self.tools.write().unwrap();
        if tools.remove(name).is_some() {
            let mut names = self.tool_names.write().unwrap();
            names.retain(|n| n != name);
            Ok(format!("CLI agent '{}' removed.", name))
        } else {
            Ok(format!("CLI agent '{}' not found.", name))
        }
    }

    /// Enable or disable a CLI agent.
    pub async fn enable_agent(&self, name: &str, enabled: bool) -> anyhow::Result<String> {
        let agents = self.state.list_dynamic_cli_agents().await?;
        if let Some(mut agent) = agents.into_iter().find(|a| a.name == name) {
            agent.enabled = enabled;
            self.state.update_dynamic_cli_agent(&agent).await?;

            if enabled {
                let args: Vec<String> = serde_json::from_str(&agent.args_json).unwrap_or_default();
                // Re-add to runtime map only after validating both a wrapper
                // command and its nested agent executable.
                if let Ok((command, args)) =
                    resolve_agent_launch(&self.backend, &agent.command, &args).await
                {
                    let entry = CliToolEntry {
                        command,
                        args,
                        description: agent.description.clone(),
                        timeout: agent
                            .timeout_secs
                            .map(Duration::from_secs)
                            .unwrap_or(self.default_timeout),
                        max_output_chars: agent.max_output_chars.unwrap_or(self.default_max_output),
                        is_dynamic: true,
                    };
                    let mut tools = self.tools.write().unwrap();
                    tools.insert(name.to_string(), entry);
                    let mut names = self.tool_names.write().unwrap();
                    if !names.contains(&name.to_string()) {
                        names.push(name.to_string());
                        names.sort();
                    }
                }
            } else {
                // Remove from runtime map
                let mut tools = self.tools.write().unwrap();
                tools.remove(name);
                let mut names = self.tool_names.write().unwrap();
                names.retain(|n| n != name);
            }

            let action = if enabled { "enabled" } else { "disabled" };
            Ok(format!("CLI agent '{}' {}.", name, action))
        } else {
            // Check if it's a discovered (non-dynamic) agent
            let tools = self.tools.read().unwrap();
            if tools.contains_key(name) {
                Ok(format!(
                    "CLI agent '{}' is a discovered agent (not dynamic). Cannot toggle — it's always available while installed.",
                    name
                ))
            } else {
                Ok(format!("CLI agent '{}' not found.", name))
            }
        }
    }

    /// List all registered agents with their status.
    pub fn list_agents(&self) -> Vec<(String, String, String, bool)> {
        let tools = self.tools.read().unwrap();
        let mut result: Vec<(String, String, String, bool)> = tools
            .iter()
            .map(|(name, entry)| {
                let source = if entry.is_dynamic {
                    "dynamic".to_string()
                } else {
                    "discovered".to_string()
                };
                (
                    name.clone(),
                    entry.description.clone(),
                    source,
                    true, // enabled if in map
                )
            })
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Clean up any finished CLI agent tasks.
    async fn reap_finished(&self) {
        let finished: Vec<(String, RunningCliAgent)> = {
            let mut running = self.running.lock().await;
            let finished_ids: Vec<String> = running
                .iter()
                .filter(|(_, agent)| agent.finished.load(Ordering::Acquire))
                .map(|(id, _)| id.clone())
                .collect();

            // Also release working-dir claims for finished tasks.
            // Lock ordering: running -> working_dir_claims.
            let mut claims = lock_claims(&self.working_dir_claims);
            let mut removed: Vec<(String, RunningCliAgent)> = Vec::new();
            for task_id in finished_ids {
                if let Some(agent) = running.remove(&task_id) {
                    if let Some(ref dir) = agent.working_dir {
                        let should_remove = claims
                            .get(dir)
                            .map(|claim| claim.task_id == task_id)
                            .unwrap_or(false);
                        if should_remove {
                            claims.remove(dir);
                        }
                    }
                    removed.push((task_id, agent));
                }
            }
            removed
        };

        for (task_id, agent) in finished {
            let final_result = Self::build_finished_result(&agent).await;
            let mut completed = self.completed.lock().await;
            completed.insert(
                task_id.clone(),
                CompletedCliAgent {
                    result: final_result,
                    completed_at: Instant::now(),
                    session_id: agent.session_id.clone(),
                },
            );
            Self::prune_completed_map(&mut completed);
            info!(task_id, tool = %agent.tool_name, "Reaped finished CLI agent");
        }
    }

    /// Build an enriched prompt with context from memory and conversation history.
    async fn build_enriched_prompt(
        &self,
        session_id: &str,
        system_instruction: &str,
        task_prompt: &str,
        working_dir: Option<&str>,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        // System instruction (never truncated)
        if !system_instruction.trim().is_empty() {
            parts.push(system_instruction.to_string());
        }

        // Task prompt (never truncated)
        parts.push(format!("## Task\n{}", task_prompt));

        let budget = MAX_PROMPT_SIZE.saturating_sub(
            parts.iter().map(|p| p.len()).sum::<usize>() + 200, // headroom for section headers
        );

        // Conversation context (truncated first if over budget)
        let mut context_text = String::new();
        if let Ok(history) = self.state.get_history(session_id, 10).await {
            if !history.is_empty() {
                let mut lines = Vec::new();
                for msg in history.iter().rev().take(10) {
                    let role = &msg.role;
                    let content: String = msg
                        .content
                        .as_deref()
                        .unwrap_or("")
                        .chars()
                        .take(400)
                        .collect();
                    lines.push(format!("{}: {}", role, content));
                }
                context_text = lines.join("\n");
            }
        }

        // Relevant facts (query both task prompt and recent user message)
        let mut facts_text = String::new();
        {
            let mut seen = HashSet::new();
            let mut facts_accum = Vec::new();

            if let Ok(facts) = self.state.get_relevant_facts(task_prompt, 15).await {
                for fact in facts {
                    if seen.insert((fact.category.clone(), fact.key.clone())) {
                        facts_accum.push(fact);
                    }
                }
            }

            if let Ok(history) = self.state.get_history(session_id, 10).await {
                if let Some(last_user_msg) = history.iter().rev().find(|m| m.role == "user") {
                    if let Some(content) = last_user_msg.content.as_deref() {
                        let user_text: String = content.chars().take(500).collect();
                        if user_text != task_prompt {
                            if let Ok(facts) = self.state.get_relevant_facts(&user_text, 10).await {
                                for fact in facts {
                                    if seen.insert((fact.category.clone(), fact.key.clone())) {
                                        facts_accum.push(fact);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            facts_accum.truncate(20);
            if !facts_accum.is_empty() {
                let fact_lines: Vec<String> = facts_accum
                    .iter()
                    .map(|f| format!("- {}: {}", f.key, f.value))
                    .collect();
                facts_text = fact_lines.join("\n");
            }
        }

        // Active goal context
        let mut active_goal_text = String::new();
        if let Ok(goals) = self.state.get_goals_for_session(session_id).await {
            if let Some(active_goal) = goals
                .iter()
                .find(|g| g.status == "active" || g.status == "in_progress")
            {
                active_goal_text = active_goal.description.chars().take(500).collect();
            }
        }

        // Scoped repository instructions. Native coding CLIs also perform
        // their own discovery from cwd; this shared snapshot keeps enriched
        // specialist prompts and non-native/custom CLI agents consistent with
        // AIDaemon's direct-agent behavior.
        let mut project_instructions_text = String::new();
        if let Some(dir) = working_dir {
            match crate::project_instructions::load_project_instructions(self.backend.clone(), dir)
                .await
            {
                Ok(Some(instructions)) => {
                    project_instructions_text = instructions.render_for_prompt();
                }
                Ok(None) => {}
                Err(error) => {
                    warn!(working_dir = dir, %error, "Could not load CLI project instructions");
                }
            }
        }

        // Native file listing from working directory (up to 3 levels)
        let mut files_text = String::new();
        if let Some(dir) = working_dir {
            let skip: HashSet<&str> = [
                ".git",
                "node_modules",
                "target",
                "__pycache__",
                ".venv",
                "dist",
                "build",
            ]
            .into_iter()
            .collect();
            let mut file_paths =
                list_backend_files(self.backend.clone(), BackendPath::new(dir), &skip, 3, 200)
                    .await;

            if !file_paths.is_empty() {
                file_paths.sort();
                files_text = file_paths.join("\n");
            }
        }

        // Spatial awareness: list ~/projects/ contents so sub-agents can discover
        // files/projects outside the current working directory.
        let mut projects_listing_text = String::new();
        if let Ok(home) = self.backend.home_dir().await {
            let projects_dir = home.join("projects");
            if let Ok(entries) = self.backend.read_dir(&projects_dir).await {
                let mut dirs_list: Vec<String> = Vec::new();
                let mut files_list: Vec<String> = Vec::new();
                for entry in entries {
                    let name = entry.path.file_name().unwrap_or_default().to_string();
                    if name.starts_with('.') {
                        continue;
                    }
                    if entry.metadata.file_type == BackendFileType::Directory {
                        dirs_list.push(format!("  {}/", name));
                    } else if entry.metadata.file_type == BackendFileType::File {
                        files_list.push(format!("  {}", name));
                    }
                    if dirs_list.len() + files_list.len() >= 80 {
                        break;
                    }
                }
                dirs_list.sort();
                files_list.sort();
                if !dirs_list.is_empty() || !files_list.is_empty() {
                    let mut listing = format!("{}:\n", projects_dir);
                    let mut all_entries = dirs_list;
                    all_entries.append(&mut files_list);
                    listing.push_str(&all_entries.join("\n"));
                    projects_listing_text = listing;
                }
            }
        }

        // Fit within budget by truncating lower-priority sections first.
        let total = active_goal_text.len()
            + project_instructions_text.len()
            + facts_text.len()
            + files_text.len()
            + projects_listing_text.len()
            + context_text.len();
        if total > budget {
            let goal_budget = budget / 10;
            let docs_budget = budget * 3 / 10;
            let facts_budget = budget * 2 / 10;
            let files_budget = budget * 2 / 10;
            let projects_budget = budget / 10;
            let context_budget = budget.saturating_sub(
                goal_budget + docs_budget + facts_budget + files_budget + projects_budget,
            );

            if active_goal_text.len() > goal_budget {
                active_goal_text = active_goal_text.chars().take(goal_budget).collect();
                active_goal_text.push_str("...[truncated]");
            }
            if project_instructions_text.len() > docs_budget {
                project_instructions_text = project_instructions_text
                    .chars()
                    .take(docs_budget)
                    .collect();
                project_instructions_text.push_str("...[truncated]");
            }
            if facts_text.len() > facts_budget {
                facts_text = facts_text.chars().take(facts_budget).collect();
                facts_text.push_str("...[truncated]");
            }
            if files_text.len() > files_budget {
                files_text = files_text.chars().take(files_budget).collect();
                files_text.push_str("...[truncated]");
            }
            if projects_listing_text.len() > projects_budget {
                projects_listing_text = projects_listing_text
                    .chars()
                    .take(projects_budget)
                    .collect();
                projects_listing_text.push_str("...[truncated]");
            }
            if context_text.len() > context_budget {
                context_text = context_text.chars().take(context_budget).collect();
                context_text.push_str("...[truncated]");
            }
        }

        if !active_goal_text.is_empty() {
            parts.push(format!("## Active Goal\n{}", active_goal_text));
        }
        if !project_instructions_text.is_empty() {
            parts.push(format!(
                "## Project Instructions\n{}",
                project_instructions_text
            ));
        }
        if !facts_text.is_empty() {
            parts.push(format!("## Known Facts\n{}", facts_text));
        }
        if !files_text.is_empty() {
            parts.push(format!("## Project Files\n{}", files_text));
        }
        if !projects_listing_text.is_empty() {
            parts.push(format!(
                "## Available Project Directories\n{}",
                projects_listing_text
            ));
        }
        if !context_text.is_empty() {
            parts.push(format!("## Conversation Context\n{}", context_text));
        }

        parts.push(
            "## Instructions\n\
             - Focus exclusively on the task above\n\
             - Do NOT attempt to directly inspect or modify aidaemon's state database (SQLite/SQLCipher). Do not run sqlite3/sqlcipher, do not install sqlcipher, and do not look for encryption keys. Use aidaemon tools/APIs instead.\n\
             - Do NOT install system packages (brew/apt/dnf/pacman/pip) unless the user explicitly asked\n\
             - Report what you did and what changed when done"
                .to_string(),
        );

        parts.join("\n\n")
    }

    /// Capture git diff after CLI agent completes (for any exit code).
    async fn capture_git_diff(working_dir: &str) -> Option<String> {
        let directory = BackendPath::new(working_dir);
        let git_check =
            crate::tools::fs_utils::run_cmd_backend("git rev-parse --git-dir", Some(&directory), 5)
                .await
                .ok()?;
        if git_check.exit_code != 0 {
            return None;
        }

        // Check for uncommitted changes first
        let diff_stat =
            crate::tools::fs_utils::run_cmd_backend("git diff --stat", Some(&directory), 10)
                .await
                .ok()?;
        let stat_output = diff_stat.stdout;

        if !stat_output.trim().is_empty() {
            // There are uncommitted changes — capture them
            let diff = crate::tools::fs_utils::run_cmd_backend("git diff", Some(&directory), 10)
                .await
                .ok()?;
            let diff_text = diff.stdout;
            if !diff_text.trim().is_empty() {
                return Some(truncate_with_note(&diff_text, MAX_DIFF_SIZE));
            }
        }

        // No uncommitted changes — check if the agent committed something
        let log = crate::tools::fs_utils::run_cmd_backend(
            "git log -1 --stat --format=%s",
            Some(&directory),
            10,
        )
        .await
        .ok()?;
        let log_output = log.stdout;

        if !log_output.trim().is_empty() {
            let committed_diff = crate::tools::fs_utils::run_cmd_backend(
                "git diff HEAD~1..HEAD",
                Some(&directory),
                10,
            )
            .await
            .ok()?;
            let committed_text = committed_diff.stdout;
            if !committed_text.trim().is_empty() {
                return Some(format!(
                    "Committed: {}\n{}",
                    log_output.lines().next().unwrap_or(""),
                    truncate_with_note(&committed_text, MAX_DIFF_SIZE)
                ));
            }
        }

        None
    }

    /// Return the number of tracked or untracked entries in an existing Git
    /// worktree. A non-repository (or an unavailable backend) is not classified
    /// as dirty here; normal CLI-agent approval and execution policy still apply.
    async fn dirty_worktree_entry_count(working_dir: &str) -> Option<usize> {
        let directory = BackendPath::new(working_dir);
        let status = crate::tools::fs_utils::run_cmd_backend(
            "git status --porcelain=v1 --untracked-files=all",
            Some(&directory),
            10,
        )
        .await
        .ok()?;
        if status.exit_code != 0 {
            return None;
        }
        Some(
            status
                .stdout
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count(),
        )
    }

    /// Detect auth-related errors in output.
    fn detect_auth_error(output: &str, tool_name: &str) -> Option<String> {
        let auth_patterns = [
            "authentication required",
            "authentication failed",
            "unauthorized",
            "not authenticated",
            "not logged in",
            "login required",
            "sign in required",
            "token expired",
            "invalid api key",
            "api key is required",
            "missing api key",
            "access denied",
            "forbidden",
            "invalid token",
            "gemini-cli-auth-docs",
        ];
        let lower = output.to_lowercase();
        for pattern in &auth_patterns {
            if lower.contains(pattern) {
                return Some(format!(
                    "ERROR: CLI agent '{}' authentication failed before it produced a task result. It has been disabled for this daemon session; retry with another configured CLI agent, or authenticate '{}' before restarting it.",
                    tool_name, tool_name,
                ));
            }
        }
        None
    }

    fn quarantine_unavailable_tool(
        tools: &Arc<std::sync::RwLock<HashMap<String, CliToolEntry>>>,
        tool_names: &Arc<std::sync::RwLock<Vec<String>>>,
        tool_name: &str,
        reason: &'static str,
    ) {
        tools
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(tool_name);
        tool_names
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|name| name != tool_name);
        warn!(
            tool = tool_name,
            reason, "Quarantined unavailable CLI agent"
        );
    }

    fn classify_completion_result(
        tool_name: &str,
        exit_code: Option<i32>,
        stdout: &str,
        display_output: &str,
        max_output: usize,
    ) -> CliCompletionResult {
        if let Some(error) = Self::detect_auth_error(display_output, tool_name) {
            return CliCompletionResult {
                success: false,
                authentication_failed: true,
                persisted_output: error.clone(),
                response: None,
                error: Some(error),
            };
        }

        if exit_code == Some(0) {
            let result = extract_meaningful_output(stdout, max_output);
            return CliCompletionResult {
                success: true,
                authentication_failed: false,
                persisted_output: result.clone(),
                response: Some(result),
                error: None,
            };
        }

        CliCompletionResult {
            success: false,
            authentication_failed: false,
            persisted_output: display_output.to_string(),
            response: None,
            error: Some(display_output.to_string()),
        }
    }

    /// Try to answer a question from a CLI agent using the LLM.
    /// Currently unused — stdin is null so we kill stuck processes instead.
    /// Kept for future interactive feedback support.
    #[allow(dead_code)]
    async fn answer_cli_question(
        provider: &Arc<dyn ModelProvider>,
        task_context: &str,
        recent_output: &str,
        question: &str,
    ) -> Option<String> {
        // Don't answer auth prompts
        let lower = question.to_lowercase();
        if lower.contains("password")
            || lower.contains("token")
            || lower.contains("api key")
            || lower.contains("secret")
            || lower.contains("credentials")
        {
            return None; // Signal to kill the process
        }

        let prompt = format!(
            "You are answering on behalf of the user. Based on the task context, \
             answer this question from a CLI agent. Be concise (1-2 sentences max).\n\n\
             Task context: {}\n\n\
             Recent agent output:\n{}\n\n\
             Question: {}",
            truncate_str(task_context, 500),
            truncate_str(recent_output, 500),
            question
        );

        let messages = vec![json!({
            "role": "user",
            "content": prompt
        })];

        // Use a fast model for quick responses
        let models = provider.list_models().await.unwrap_or_default();
        let model = models.first().map(|m| m.as_str()).unwrap_or("default");

        match provider.chat(model, &messages, &[]).await {
            Ok(response) => {
                let answer = response
                    .content
                    .as_ref()
                    .map(|c| c.trim().to_string())
                    .unwrap_or_else(|| "yes".to_string());
                Some(answer)
            }
            Err(_) => {
                // Fallback: for y/n questions answer "yes", otherwise return None
                if lower.contains("y/n")
                    || lower.contains("yes/no")
                    || lower.contains("confirm")
                    || lower.ends_with("?")
                {
                    Some("yes".to_string())
                } else {
                    None
                }
            }
        }
    }

    /// Run a CLI agent with streaming output.
    #[allow(clippy::too_many_arguments)]
    async fn handle_run(
        &self,
        tool_name: &str,
        prompt: &str,
        working_dir: Option<&str>,
        session_id: &str,
        goal_id: Option<&str>,
        delegated_task_id: Option<&str>,
        system_instruction: Option<&str>,
        workspace_mode: CliWorkspaceMode,
        async_mode: bool,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        // Get entry from the tools map (clone what we need, release lock)
        let (mut command, mut args, timeout, max_output) = {
            let tools = self.tools.read().unwrap();
            let entry = tools
                .get(tool_name)
                .ok_or_else(|| anyhow::anyhow!("Unknown CLI agent tool: {}", tool_name))?;
            (
                entry.command.clone(),
                entry.args.clone(),
                entry.timeout,
                entry.max_output_chars,
            )
        };

        // Claude Code can refuse to start when it detects it's running inside a
        // parent Claude session. Also, aidaemon has no TTY for interactive mode.
        // Force non-interactive mode and remove the nesting marker env var.
        if tool_name.eq_ignore_ascii_case("claude") {
            // Always run non-interactively.
            let has_print = args.iter().any(|a| a == "-p" || a == "--print");
            if !has_print {
                args.push("--print".to_string());
            }

            // Prefer stream-json output so we can show progress updates.
            let has_output_format = args.iter().any(|a| a == "--output-format" || a == "-o");
            if !has_output_format {
                args.push("--output-format".to_string());
                args.push("stream-json".to_string());
            }

            let has_partial = args.iter().any(|a| a == "--include-partial-messages");
            if !has_partial {
                args.push("--include-partial-messages".to_string());
            }

            let has_verbose = args.iter().any(|a| a == "--verbose");
            if !has_verbose {
                args.push("--verbose".to_string());
            }
        }

        // Re-check both the launcher and a wrapped nested executable. Agent
        // binaries may move during self-update, and GUI daemon PATH values can
        // differ from an interactive shell.
        match resolve_agent_launch(&self.backend, &command, &args).await {
            Ok((resolved_command, resolved_args)) => {
                command = resolved_command;
                args = resolved_args;
            }
            Err(error) => {
                // Auto-disable dynamic agents that disappeared
                if let Ok(agents) = self.state.list_dynamic_cli_agents().await {
                    if let Some(mut agent) = agents.into_iter().find(|a| a.name == tool_name) {
                        agent.enabled = false;
                        let _ = self.state.update_dynamic_cli_agent(&agent).await;
                    }
                }
                let message = format!(
                    "ERROR: CLI agent '{tool_name}' could not start: {error}. \
                 Re-register it with an absolute executable path or reinstall it."
                );
                Self::quarantine_unavailable_tool(
                    &self.tools,
                    &self.tool_names,
                    tool_name,
                    "launch resolution failed",
                );
                return Err(CliAgentUnavailableError {
                    tool_name: tool_name.to_string(),
                    message,
                }
                .into());
            }
        }

        if workspace_mode == CliWorkspaceMode::ReadOnly {
            if let Err(error) = apply_read_only_cli_adapter(&command, &mut args) {
                return Ok(format!(
                    "Blocked: read-only CLI delegation was not started because {error}. Select a CLI agent with a registered native read-only sandbox."
                ));
            }
        }

        let canonical_working_dir = match working_dir {
            Some(dir) => Some(Self::normalize_working_dir(dir).await?),
            None => None,
        };
        let dedup_prompt = make_dedup_prompt(prompt);
        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        let short_summary: String = prompt.chars().take(50).collect();

        // Claim the working directory and perform conflict checks.
        // Lock ordering when both locks are needed: running -> working_dir_claims.
        // The claims (std) lock must not be held across an await, so the
        // conflict message is computed inside the block and persisted after.
        let mut claim_guard: Option<WorkingDirClaimGuard> = None;
        if let Some(ref dir) = canonical_working_dir {
            let blocked_message: Option<String> = {
                let _running_guard = self.running.lock().await;
                let mut claims = lock_claims(&self.working_dir_claims);

                if let Some(claim) = claims.get(dir) {
                    let sim = prompt_similarity(&dedup_prompt, &claim.dedup_prompt);
                    if sim > 0.5 {
                        Some(format!(
                            "BLOCKED: A very similar task is already running in {} \
                             (task_id={}, agent={}, similarity={:.0}%). \
                             You MUST wait for it to finish or cancel it.",
                            dir,
                            claim.task_id,
                            claim.tool_name,
                            sim * 100.0
                        ))
                    } else {
                        Some(format!(
                            "BLOCKED: Another CLI agent is already working in {} \
                             (task_id={}, agent={}, prompt=\"{}\"). \
                             You MUST wait for it to finish or cancel it before dispatching \
                             another task to the same directory.",
                            dir, claim.task_id, claim.tool_name, claim.prompt_summary
                        ))
                    }
                } else {
                    claims.insert(
                        dir.clone(),
                        WorkingDirClaim {
                            task_id: task_id.clone(),
                            tool_name: tool_name.to_string(),
                            prompt_summary: short_summary.clone(),
                            dedup_prompt: dedup_prompt.clone(),
                        },
                    );
                    None
                }
            };

            if let Some(message) = blocked_message {
                if let Some(task_id) = delegated_task_id {
                    Self::persist_delegated_cli_result_with_state(
                        self.state.clone(),
                        task_id,
                        Some(&message),
                        None,
                    )
                    .await;
                }
                return Ok(message);
            }

            // From here on the claim is released by the guard's Drop on any
            // early return or future cancellation, unless it is disarmed when
            // the task hands off to the background `running` map.
            claim_guard = Some(WorkingDirClaimGuard::new(
                self.working_dir_claims.clone(),
                dir.clone(),
                task_id.clone(),
            ));
        }

        // Build the enriched prompt if system_instruction is provided
        let final_prompt = if let Some(instruction) = system_instruction {
            self.build_enriched_prompt(
                session_id,
                instruction,
                prompt,
                canonical_working_dir.as_deref(),
            )
            .await
        } else {
            prompt.to_string()
        };

        let slot_permit = match self.concurrency_limiter.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                // claim_guard drops on return and releases the claim.
                let message = format!(
                    "Maximum {} CLI agents already running. Use action='list' to see running tasks, or action='cancel' to stop one.",
                    self.max_concurrent
                );
                if let Some(task_id) = delegated_task_id {
                    Self::persist_delegated_cli_result_with_state(
                        self.state.clone(),
                        task_id,
                        None,
                        Some(&message),
                    )
                    .await;
                }
                return Ok(message);
            }
        };

        if let Some(task_id) = delegated_task_id {
            self.persist_delegated_cli_handoff(
                task_id,
                tool_name,
                &final_prompt,
                canonical_working_dir.as_deref(),
            )
            .await;
        }

        info!(
            tool = tool_name,
            session = session_id,
            prompt_len = final_prompt.len(),
            working_dir = canonical_working_dir.as_deref().unwrap_or("(default)"),
            async_mode,
            workspace_mode = ?workspace_mode,
            "CLI agent invocation"
        );

        // Log invocation start
        let prompt_summary: String = prompt.chars().take(100).collect();
        let invocation_id = self
            .state
            .log_cli_agent_start(
                Some(&task_id),
                session_id,
                tool_name,
                &prompt_summary,
                canonical_working_dir.as_deref(),
            )
            .await
            .unwrap_or(0);

        let state_for_completion = self.state.clone();
        let tools_for_completion = self.tools.clone();
        let tool_names_for_completion = self.tool_names.clone();
        let delegated_task_id_owned = delegated_task_id.map(|task_id| task_id.to_string());

        let mut command_args = args.clone();
        command_args.push(final_prompt.clone());
        let mut request = ExecutionRequest::argv(command.clone(), command_args);
        if tool_name.eq_ignore_ascii_case("claude") {
            request.env_remove.push("CLAUDECODE".to_string());
        }
        if let Some(ref dir) = canonical_working_dir {
            request.cwd = Some(BackendPath::new(dir));
        }

        info!(
            task_id,
            tool = %tool_name,
            command = %command,
            working_dir = ?canonical_working_dir,
            "Starting CLI agent"
        );

        // Notify user this task can be cancelled
        if let Some(ref tx) = status_tx {
            let _ = tx.try_send(StatusUpdate::ToolCancellable {
                name: tool_name.to_string(),
                task_id: task_id.clone(),
            });
        }

        let started_at_instant = Instant::now();
        let mut spawned = match self.backend.spawn(request).await {
            Ok(child) => child,
            Err(e) => {
                // claim_guard drops on return and releases the claim.
                // Ensure invocations don't stay "running" forever when spawn fails.
                if invocation_id != 0 {
                    let duration = started_at_instant.elapsed().as_secs_f64();
                    let msg = format!("Failed to spawn CLI agent '{}': {}", tool_name, e);
                    let summary: String = msg.chars().take(200).collect();
                    let _ = state_for_completion
                        .log_cli_agent_complete(invocation_id, None, &summary, false, duration)
                        .await;
                }
                if let Some(ref delegated_task_id) = delegated_task_id_owned {
                    Self::persist_delegated_cli_result_with_state(
                        state_for_completion.clone(),
                        delegated_task_id,
                        None,
                        Some(&format!("Failed to spawn CLI agent '{}': {}", tool_name, e)),
                    )
                    .await;
                }
                return Err(e);
            }
        };
        let process_handle = spawned.handle().clone();
        let pid = process_handle.display_id();
        let stdout = match spawned.take_stdout() {
            Some(stdout) => stdout,
            None => {
                // claim_guard drops on return and releases the claim.
                let error = "Failed to capture stdout".to_string();
                if let Some(ref delegated_task_id) = delegated_task_id_owned {
                    Self::persist_delegated_cli_result_with_state(
                        self.state.clone(),
                        delegated_task_id,
                        None,
                        Some(&error),
                    )
                    .await;
                }
                return Err(anyhow::anyhow!(error));
            }
        };
        let stderr = match spawned.take_stderr() {
            Some(stderr) => stderr,
            None => {
                // claim_guard drops on return and releases the claim.
                let error = "Failed to capture stderr".to_string();
                if let Some(ref delegated_task_id) = delegated_task_id_owned {
                    Self::persist_delegated_cli_result_with_state(
                        self.state.clone(),
                        delegated_task_id,
                        None,
                        Some(&error),
                    )
                    .await;
                }
                return Err(anyhow::anyhow!(error));
            }
        };
        let mut child = spawned.into_child();

        // Two buffers: stdout_buf for JSON extraction, display_buf for user display
        let stdout_buf = Arc::new(Mutex::new(String::new()));
        let display_buf = Arc::new(Mutex::new(String::new()));
        let stdout_buf_writer = stdout_buf.clone();
        let display_buf_writer = display_buf.clone();
        let status_tx_clone = status_tx.clone();
        let tool_name_owned = tool_name.to_string();

        // For question detection (kill process if it asks a question we can't answer)
        let task_context = prompt_summary.clone();

        // Create completion channel - includes loop detection info
        // Result: (exit_code, was_killed_for_loop, loop_repetition_count)
        let (completion_tx, completion_rx) =
            tokio::sync::oneshot::channel::<(Option<i32>, bool, Option<usize>)>();

        // Spawn a task to read stdout/stderr, emit progress updates, and signal completion.
        //
        // `should_notify` defaults to async_mode. If a sync run later times out and moves to
        // background, we flip it to true so completion is still delivered to the user.
        let should_notify = Arc::new(AtomicBool::new(async_mode));
        let pid_for_kill = pid;
        let backend_for_task = self.backend.clone();
        let process_handle_for_task = process_handle.clone();
        let finished = Arc::new(AtomicBool::new(false));
        let finished_for_task = finished.clone();
        let invocation_started_at = started_at_instant;
        let max_output_for_log = max_output;
        let notify_session_id = session_id.to_string();
        let notify_async = async_mode;
        let notify_goal_id = goal_id.map(|s| s.to_string()).unwrap_or_default();
        let notify_working_dir = canonical_working_dir.clone();
        let hub_for_completion = self.get_hub();
        let agent_for_completion = self.get_agent();
        let reengagements_for_completion = self.reengagements.clone();
        let task_id_for_notify = task_id.clone();
        let delegated_task_id_for_completion = delegated_task_id_owned.clone();
        let should_notify_for_task = Arc::clone(&should_notify);
        let slot_permit_for_task = slot_permit;
        tokio::spawn(async move {
            let _slot_permit = slot_permit_for_task;
            let mut stdout_reader = BufReader::new(stdout).lines();
            let mut stderr_reader = BufReader::new(stderr).lines();
            let mut last_progress = Instant::now();
            let mut last_chat_progress = Instant::now();
            let mut status_tick = tokio::time::interval(PROGRESS_INTERVAL);
            status_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            status_tick.tick().await; // consume the immediate first tick
            let started_at = Instant::now();
            let mut pending_lines: Vec<String> = Vec::new();
            let mut stdout_done = false;
            let mut stderr_done = false;
            let mut last_parsed_action: Option<String> = None;
            let mut loop_detector = LoopDetector::new();
            let mut loop_detected = false;
            let mut loop_pattern_count: Option<usize> = None;
            let mut last_output_time = Instant::now();
            let mut last_non_empty_line = String::new();

            loop {
                if stdout_done && stderr_done {
                    break;
                }

                // Check for loop detection and kill if needed
                if loop_detected {
                    info!(
                        pid = pid_for_kill,
                        pattern_count = ?loop_pattern_count,
                        "Infinite loop detected in CLI agent output, killing process"
                    );
                    kill_process(&backend_for_task, &process_handle_for_task).await;
                    break;
                }

                tokio::select! {
                    line = stdout_reader.next_line(), if !stdout_done => {
                        match line {
                            Ok(Some(text)) => {
                                last_output_time = Instant::now();
                                if !text.trim().is_empty() {
                                    last_non_empty_line = text.clone();
                                }

                                // Guardrail: kill external coding agents if they attempt to
                                // manipulate aidaemon's state DB (E2E rabbit-hole).
                                if looks_like_json(&text) {
                                    let mut blocked: Option<(String, &'static str)> = None;
                                    for cmd in extract_terminal_commands_from_json(&text) {
                                        if let Some(reason) = prohibited_cli_agent_command_reason(&cmd) {
                                            blocked = Some((cmd, reason));
                                            break;
                                        }
                                    }

                                    if let Some((cmd, reason)) = blocked {
                                        info!(
                                            pid = pid_for_kill,
                                            cmd = %cmd,
                                            "Prohibited CLI agent command detected; killing process"
                                        );
                                        {
                                            let mut buf = display_buf_writer.lock().await;
                                            append_bounded_line(
                                                &mut buf,
                                                "[killed] Prohibited CLI agent command: ",
                                                &format!("{cmd}\nReason: {reason}"),
                                                BUFFER_CAP,
                                            );
                                        }
                                        kill_process(
                                            &backend_for_task,
                                            &process_handle_for_task,
                                        )
                                        .await;
                                        break;
                                    }
                                }

                                // Check for infinite loop pattern
                                if loop_detector.add_line(&text) && !loop_detected {
                                    loop_detected = true;
                                    loop_pattern_count = loop_detector.get_loop_pattern();
                                }

                                // Write to stdout buffer (for JSON extraction)
                                {
                                    let mut buf = stdout_buf_writer.lock().await;
                                    append_bounded_line(&mut buf, "", &text, BUFFER_CAP);
                                }
                                // Write to display buffer
                                {
                                    let mut buf = display_buf_writer.lock().await;
                                    append_bounded_line(&mut buf, "", &text, BUFFER_CAP);
                                }
                                pending_lines.push(text);
                            }
                            _ => stdout_done = true,
                        }
                    }
                    line = stderr_reader.next_line(), if !stderr_done => {
                        match line {
                            Ok(Some(text)) => {
                                last_output_time = Instant::now();
                                if !text.trim().is_empty() {
                                    last_non_empty_line = format!("[stderr] {}", text);
                                }

                                // Check for infinite loop pattern in stderr too
                                if loop_detector.add_line(&text) && !loop_detected {
                                    loop_detected = true;
                                    loop_pattern_count = loop_detector.get_loop_pattern();
                                }

                                // Only write to display buffer with [stderr] prefix
                                let mut buf = display_buf_writer.lock().await;
                                append_bounded_line(&mut buf, "[stderr] ", &text, BUFFER_CAP);
                                pending_lines.push(format!("[stderr] {}", text));
                            }
                            _ => stderr_done = true,
                        }
                    }
                    // Check for question patterns when no output for 15s
                    _ = tokio::time::sleep(Duration::from_secs(15)), if !last_non_empty_line.is_empty() && last_output_time.elapsed() > Duration::from_secs(14) => {
                        let line = &last_non_empty_line;
                        let lower = line.to_lowercase();
                        let is_question = line.ends_with('?')
                            || lower.contains("y/n")
                            || lower.contains("yes/no")
                            || lower.contains("enter")
                            || lower.contains("confirm")
                            || lower.contains("choose")
                            || lower.contains("select")
                            || lower.contains("which");

                        if is_question {
                            // stdin is null so we can't answer — kill the process
                            // and report the question so the orchestrator can handle it
                            info!(
                                question = %line,
                                task = %task_context,
                                "CLI agent appears stuck waiting for input — killing (stdin is null)"
                            );
                            let mut buf = display_buf_writer.lock().await;
                            append_bounded_line(
                                &mut buf,
                                "[killed] CLI agent appears stuck waiting for input: ",
                                line,
                                BUFFER_CAP,
                            );
                            drop(buf);
                            kill_process(&backend_for_task, &process_handle_for_task).await;
                            break;
                        }
                    }
                    _ = status_tick.tick() => {}
                }

                // Emit progress updates at intervals
                // Parse JSON lines to extract meaningful progress, filter raw JSON
                if last_progress.elapsed() >= PROGRESS_INTERVAL {
                    let mut progress_items: Vec<String> = Vec::new();
                    for line in &pending_lines {
                        if looks_like_json(line) {
                            // Try to extract meaningful progress from JSON
                            if let Some(progress) = extract_progress_from_json(line) {
                                progress_items.push(progress.clone());
                                last_parsed_action = Some(progress);
                            }
                        } else {
                            // Non-JSON line, include as-is
                            progress_items.push(line.clone());
                        }
                    }

                    let elapsed_secs = started_at.elapsed().as_secs();
                    let chunk = if !progress_items.is_empty() {
                        // Deduplicate consecutive items
                        progress_items.dedup();
                        progress_items.join("\n")
                    } else if let Some(ref action) = last_parsed_action {
                        // No new progress, but we have a last action - show heartbeat
                        format!("⏳ {} ({}s)", action, elapsed_secs)
                    } else {
                        // No parsed progress at all - show generic heartbeat
                        format!("⏳ Working... ({}s)", elapsed_secs)
                    };

                    if let Some(ref tx) = status_tx_clone {
                        let _ = tx.try_send(StatusUpdate::ToolProgress {
                            name: tool_name_owned.clone(),
                            chunk: truncate_with_note(&chunk, 500),
                        });
                    }

                    if should_notify_for_task.load(Ordering::Relaxed)
                        && last_chat_progress.elapsed() >= BACKGROUND_CHAT_PROGRESS_INTERVAL
                    {
                        let message = format_background_progress(
                            &tool_name_owned,
                            &task_context,
                            elapsed_secs,
                            Some(&chunk),
                        );
                        let hub = hub_for_completion.clone();
                        let state = state_for_completion.clone();
                        let goal_id = notify_goal_id.clone();
                        let session_id = notify_session_id.clone();
                        tokio::spawn(async move {
                            deliver_cli_agent_notification(
                                hub.as_ref(),
                                &state,
                                &goal_id,
                                &session_id,
                                "progress",
                                &message,
                                "cli_agent background progress notifier",
                            )
                            .await;
                        });
                        last_chat_progress = Instant::now();
                    }

                    pending_lines.clear();
                    last_progress = Instant::now();
                }
            }

            // Send any remaining lines (with JSON parsing)
            if !pending_lines.is_empty() {
                if let Some(ref tx) = status_tx_clone {
                    let mut progress_items: Vec<String> = Vec::new();
                    for line in &pending_lines {
                        if looks_like_json(line) {
                            if let Some(progress) = extract_progress_from_json(line) {
                                progress_items.push(progress);
                            }
                        } else {
                            progress_items.push(line.clone());
                        }
                    }
                    if !progress_items.is_empty() {
                        progress_items.dedup();
                        let chunk = progress_items.join("\n");
                        let _ = tx.try_send(StatusUpdate::ToolProgress {
                            name: tool_name_owned.clone(),
                            chunk: truncate_with_note(&chunk, 500),
                        });
                    }
                }
            }

            // Wait for process to complete and signal via channel
            let exit_code = if loop_detected {
                // Process was killed due to loop detection
                None
            } else {
                match child.wait().await {
                    Ok(status) => status.code(),
                    Err(_) => None,
                }
            };
            // Persist completion even if the caller timed out and moved the task to background.
            if invocation_id != 0 {
                let duration = invocation_started_at.elapsed().as_secs_f64();
                let completion = if loop_detected {
                    CliCompletionResult {
                        success: false,
                        authentication_failed: false,
                        persisted_output: "Killed - infinite loop detected".to_string(),
                        response: None,
                        error: Some("Killed - infinite loop detected".to_string()),
                    }
                } else {
                    // Prefer stdout for success summaries; fall back to display output for failures.
                    let stdout_text = stdout_buf_writer.lock().await.clone();
                    let display_text = display_buf_writer.lock().await.clone();
                    CliAgentTool::classify_completion_result(
                        &tool_name_owned,
                        exit_code,
                        &stdout_text,
                        &display_text,
                        max_output_for_log,
                    )
                };
                let success = completion.success;
                if completion.authentication_failed {
                    CliAgentTool::quarantine_unavailable_tool(
                        &tools_for_completion,
                        &tool_names_for_completion,
                        &tool_name_owned,
                        "authentication failed",
                    );
                }
                let persisted_output = completion.persisted_output;
                let structured_response = completion.response;
                let structured_error = completion.error;

                let _ = state_for_completion
                    .log_cli_agent_complete(
                        invocation_id,
                        exit_code,
                        &persisted_output,
                        success,
                        duration,
                    )
                    .await;

                if let Some(ref delegated_task_id) = delegated_task_id_for_completion {
                    CliAgentTool::persist_delegated_cli_result_with_state(
                        state_for_completion.clone(),
                        delegated_task_id,
                        structured_response.as_deref(),
                        structured_error.as_deref(),
                    )
                    .await;
                }

                // Durable state is authoritative for completion. Publish the
                // finished flag before optional chat delivery, which may be
                // slow, but never before the delegated task update above.
                finished_for_task.store(true, Ordering::Release);

                // Send proactive notification when the caller won't necessarily be waiting
                // for completion (async_mode or sync->timeout moved to background).
                if notify_async || should_notify_for_task.load(Ordering::Relaxed) {
                    let duration_secs = duration as u64;
                    let duration_display = if duration_secs >= 60 {
                        format!("{}m{}s", duration_secs / 60, duration_secs % 60)
                    } else {
                        format!("{}s", duration_secs)
                    };

                    let status_word = if success { "completed" } else { "failed" };
                    let notification_summary: String = persisted_output.chars().take(500).collect();
                    let diff_section = if let Some(ref dir) = notify_working_dir {
                        CliAgentTool::capture_git_diff(dir)
                            .await
                            .map(|diff| format!("\n\n{}", diff))
                    } else {
                        None
                    };

                    let message = format!(
                        "Background task {} ({}, {})\nTask: {}\nResult: {}{}",
                        status_word,
                        tool_name_owned,
                        duration_display,
                        task_context,
                        notification_summary,
                        diff_section.unwrap_or_default(),
                    );
                    let notification_type = if success { "completed" } else { "failed" };
                    let message =
                        crate::channels::present_notification(notification_type, &message);

                    // Direct channel delivery is intentionally not a second
                    // agent turn, but the result must still enter the session's
                    // hot history. Otherwise the very next user follow-up sees
                    // the handoff acknowledgement but not the completed work.
                    let continuity_output = format!(
                        "Background CLI task {status_word}.\nTask ID: {task_id_for_notify}\nTask: {task_context}\nResult:\n{persisted_output}"
                    );
                    let continuity_message = Message {
                        content: Some(crate::tools::sanitize::wrap_untrusted_output(
                            "cli_agent",
                            &continuity_output,
                        )),
                        importance: 0.8,
                        ..Message::new_runtime(
                            Uuid::new_v4().to_string(),
                            notify_session_id.clone(),
                            "assistant",
                        )
                    };
                    if let Err(e) = state_for_completion
                        .append_message(&continuity_message)
                        .await
                    {
                        warn!(
                            task_id = %task_id_for_notify,
                            session_id = %notify_session_id,
                            error = %e,
                            "cli_agent background completion could not enter session history"
                        );
                    }

                    let mut delivered = false;
                    if notify_goal_id.is_empty() && !notify_session_id.trim().is_empty() {
                        let reengagement_allowed = {
                            let mut log = reengagements_for_completion.lock().await;
                            crate::tools::terminal::reengagement_allowed(
                                &mut log,
                                &notify_session_id,
                                Instant::now(),
                            )
                        };
                        if !reengagement_allowed {
                            warn!(
                                task_id = %task_id_for_notify,
                                session_id = %notify_session_id,
                                "cli_agent completion re-engagement budget exhausted; delivering raw result"
                            );
                        } else if let Some(ref agent) = agent_for_completion {
                            let followup = CliAgentTool::build_background_reengagement_followup(
                                &tool_name_owned,
                                &task_id_for_notify,
                                &task_context,
                                status_word,
                                &persisted_output,
                            );
                            info!(
                                task_id = %task_id_for_notify,
                                session_id = %notify_session_id,
                                "Re-engaging agent loop after CLI agent completion"
                            );
                            let continuation_started = Instant::now();
                            let mut continuation_progress =
                                tokio::time::interval(BACKGROUND_CHAT_PROGRESS_INTERVAL);
                            continuation_progress
                                .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                            continuation_progress.tick().await;
                            let mut continuation = std::pin::pin!(
                                crate::tools::terminal::run_background_continuation(
                                    agent.as_ref(),
                                    ConversationRequest {
                                        session_id: notify_session_id.clone(),
                                        user_text: followup,
                                        status_tx: None,
                                        user_role: crate::types::UserRole::Owner,
                                        channel_ctx: crate::types::ChannelContext::internal(),
                                        heartbeat: None,
                                        parent_task_id: None,
                                        parent_tool_call_id: None,
                                        parent_result_id: None,
                                    },
                                )
                            );
                            let continuation_result = loop {
                                tokio::select! {
                                    result = &mut continuation => break result,
                                    _ = continuation_progress.tick() => {
                                        let progress_message = format_background_progress(
                                            "AIDaemon",
                                            &task_context,
                                            continuation_started.elapsed().as_secs(),
                                            Some("Processing the delegated result and continuing the remaining work."),
                                        );
                                        deliver_cli_agent_notification(
                                            hub_for_completion.as_ref(),
                                            &state_for_completion,
                                            &notify_goal_id,
                                            &notify_session_id,
                                            "progress",
                                            &progress_message,
                                            "cli_agent completion continuation progress notifier",
                                        )
                                        .await;
                                    }
                                }
                            };
                            match continuation_result {
                                Ok(envelope) if !envelope.text.trim().is_empty() => {
                                    let reply = crate::tools::sanitize::sanitize_user_facing_reply(
                                        &envelope.text,
                                    );
                                    if let Some(ref hub) = hub_for_completion {
                                        let _ = agent
                                            .record_continuation_delivery(
                                                &notify_session_id,
                                                envelope.delivery(
                                                    "background_router",
                                                    crate::events::ResponseDeliveryState::Queued,
                                                    Vec::new(),
                                                    None,
                                                ),
                                            )
                                            .await;
                                        match hub
                                            .send_text_tracked(&notify_session_id, &reply)
                                            .await
                                        {
                                            Ok(platform_id) => {
                                                delivered = true;
                                                let _ = agent
                                                    .record_continuation_delivery(
                                                        &notify_session_id,
                                                        envelope.delivery(
                                                            "background_router",
                                                            crate::events::ResponseDeliveryState::PlatformAcknowledged,
                                                            platform_id.into_iter().collect(),
                                                            None,
                                                        ),
                                                    )
                                                    .await;
                                            }
                                            Err(e) => warn!(
                                                task_id = %task_id_for_notify,
                                                session_id = %notify_session_id,
                                                error = %e,
                                                "cli_agent continuation reply delivery failed"
                                            ),
                                        }
                                    }
                                    if !delivered {
                                        let entry = crate::traits::NotificationEntry::new(
                                            &notify_goal_id,
                                            &notify_session_id,
                                            "progress",
                                            &reply,
                                        );
                                        match state_for_completion
                                            .enqueue_notification(&entry)
                                            .await
                                        {
                                            Ok(()) => delivered = true,
                                            Err(e) => warn!(
                                                task_id = %task_id_for_notify,
                                                session_id = %notify_session_id,
                                                error = %e,
                                                "cli_agent continuation reply enqueue failed"
                                            ),
                                        }
                                    }
                                }
                                Ok(_) => warn!(
                                    task_id = %task_id_for_notify,
                                    session_id = %notify_session_id,
                                    "cli_agent completion re-engagement returned an empty reply"
                                ),
                                Err(e) => warn!(
                                    task_id = %task_id_for_notify,
                                    session_id = %notify_session_id,
                                    error = %e,
                                    "cli_agent completion re-engagement failed"
                                ),
                            }
                        }
                    }

                    if let Some(ref hub) = hub_for_completion {
                        if delivered {
                            // The continuation reply is the useful completion;
                            // suppress the lower-level raw task notification.
                        } else if let Err(e) = hub.send_text(&notify_session_id, &message).await {
                            warn!(
                                task_id = %task_id_for_notify,
                                session_id = %notify_session_id,
                                error = %e,
                                "cli_agent background completion direct hub delivery failed"
                            );
                        } else {
                            delivered = true;
                        }
                    }

                    if !delivered {
                        let queue_goal_id = if notify_goal_id.trim().is_empty() {
                            "global"
                        } else {
                            &notify_goal_id
                        };
                        let entry = crate::traits::NotificationEntry::new(
                            queue_goal_id,
                            &notify_session_id,
                            notification_type,
                            &message,
                        );
                        if let Err(e) = state_for_completion.enqueue_notification(&entry).await {
                            warn!(
                                task_id = %task_id_for_notify,
                                session_id = %notify_session_id,
                                error = %e,
                                "cli_agent background completion enqueue failed"
                            );
                        }
                    }
                }
            }
            // Publish completion only after durable invocation/task state has
            // been flushed. Reapers and `check` callers must never observe a
            // finished process while its delegated board task is still running.
            finished_for_task.store(true, Ordering::Release);
            let _ = completion_tx.send((exit_code, loop_detected, loop_pattern_count));
        });

        // For async_mode, return immediately with task_id
        if async_mode {
            let working_dir_owned = canonical_working_dir.clone();
            let agent = RunningCliAgent {
                tool_name: tool_name.to_string(),
                prompt_summary: short_summary.clone(),
                started_at: started_at_instant,
                display_buf,
                stdout_buf,
                process_handle: process_handle.clone(),
                finished: finished.clone(),
                session_id: session_id.to_string(),
                delegated_task_id: delegated_task_id_owned.clone(),
                working_dir: working_dir_owned,
            };
            self.running.lock().await.insert(task_id.clone(), agent);
            // The reaper / cancel paths own the claim now (no await between
            // the insert above and this disarm, so no cancellation window).
            if let Some(guard) = claim_guard.as_mut() {
                guard.disarm();
            }

            return Ok(format!(
                "CLI agent '{}' started in background (task_id={}). \
                 Use action=\"check\" with task_id=\"{}\" to see output when done.",
                tool_name, task_id, task_id
            ));
        }

        // Wait for completion with timeout
        let working_dir_owned = canonical_working_dir;
        let result = tokio::time::timeout(timeout, completion_rx).await;

        match result {
            Ok(Ok((exit_code, was_loop_killed, loop_count))) => {
                // Sync completion path: release the claim eagerly (the rest
                // of this arm still awaits diff capture and persistence).
                drop(claim_guard.take());

                // Check if killed due to infinite loop
                if was_loop_killed {
                    let display_output = display_buf.lock().await.clone();
                    let last_lines: String = display_output
                        .lines()
                        .rev()
                        .take(10)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join("\n");

                    // Emit error status
                    if let Some(ref tx) = status_tx {
                        let (label, summary) = crate::tools::sanitize::user_facing_tool_activity(
                            tool_name,
                            "killed - infinite loop detected",
                            crate::types::ChannelVisibility::Public,
                        );
                        let _ = tx.try_send(StatusUpdate::ToolComplete {
                            name: label,
                            summary,
                        });
                    }

                    return Ok(format!(
                        "ERROR: CLI agent '{}' was automatically killed - INFINITE LOOP DETECTED.\n\n\
                         The same output line repeated {} times in the last 100 lines.\n\
                         This is a known bug in some CLI agent versions where they get stuck.\n\n\
                         Last 10 lines before kill:\n{}\n\n\
                         Do NOT retry with the same agent. Try a different approach or use a different tool.",
                        tool_name,
                        loop_count.unwrap_or(0),
                        last_lines
                    ));
                }

                // Authentication/setup failures are failures even when a CLI
                // exits zero (Gemini CLI has emitted its auth-help URL and a
                // successful process status without executing the prompt).
                // Classify this before emitting ToolComplete so the live
                // status cannot briefly report a false success.
                let display_output = display_buf.lock().await.clone();
                if let Some(auth_msg) = Self::detect_auth_error(&display_output, tool_name) {
                    if let Some(ref tx) = status_tx {
                        let (label, summary) = crate::tools::sanitize::user_facing_tool_activity(
                            tool_name,
                            "authentication failed",
                            crate::types::ChannelVisibility::Public,
                        );
                        let _ = tx.try_send(StatusUpdate::ToolComplete {
                            name: label,
                            summary,
                        });
                    }
                    Self::quarantine_unavailable_tool(
                        &self.tools,
                        &self.tool_names,
                        tool_name,
                        "authentication failed",
                    );
                    return Err(CliAgentUnavailableError {
                        tool_name: tool_name.to_string(),
                        message: auth_msg,
                    }
                    .into());
                }

                // Completed within timeout normally
                // Use stdout_buf for JSON extraction (clean, no stderr prefixes)
                let stdout_output = stdout_buf.lock().await.clone();
                info!(
                    tool = %tool_name,
                    stdout_len = stdout_output.len(),
                    stdout_preview = %truncate_str(&stdout_output, 200),
                    "CLI agent stdout captured"
                );
                let result_text = extract_meaningful_output(&stdout_output, max_output);
                info!(
                    tool = %tool_name,
                    result_len = result_text.len(),
                    result_preview = %truncate_str(&result_text, 200),
                    "CLI agent result extracted"
                );

                // Capture git diff
                let diff_section = if let Some(ref dir) = working_dir_owned {
                    Self::capture_git_diff(dir)
                        .await
                        .map(|diff| format!("\n\n## File Changes\n```diff\n{}\n```", diff))
                } else {
                    None
                };

                // Emit completion status
                if let Some(ref tx) = status_tx {
                    let raw_summary = if exit_code == Some(0) {
                        "completed successfully".to_string()
                    } else {
                        format!("exited with code {:?}", exit_code)
                    };
                    let (label, summary) = crate::tools::sanitize::user_facing_tool_activity(
                        tool_name,
                        &raw_summary,
                        crate::types::ChannelVisibility::Public,
                    );
                    let _ = tx.try_send(StatusUpdate::ToolComplete {
                        name: label,
                        summary,
                    });
                }

                if exit_code != Some(0) {
                    // On error, show the display buffer which includes stderr
                    let error_msg = format_cli_agent_failure(
                        tool_name,
                        exit_code,
                        &display_output,
                        max_output,
                        diff_section.as_deref(),
                    );

                    if let Some(task_id) = delegated_task_id {
                        self.persist_delegated_cli_result(task_id, None, Some(&error_msg))
                            .await;
                    }

                    return Ok(error_msg);
                }

                // Success path
                let mut final_result = result_text;
                if let Some(diff) = diff_section {
                    final_result.push_str(&diff);
                }
                if let Some(task_id) = delegated_task_id {
                    self.persist_delegated_cli_result(task_id, Some(final_result.as_str()), None)
                        .await;
                }
                Ok(final_result)
            }
            Ok(Err(_)) => {
                drop(claim_guard.take());
                // Channel closed unexpectedly
                let error_msg =
                    format!("ERROR: CLI agent '{}' task failed unexpectedly", tool_name);
                if let Some(task_id) = delegated_task_id {
                    self.persist_delegated_cli_result(task_id, None, Some(&error_msg))
                        .await;
                }
                Ok(error_msg)
            }
            Err(_) => {
                // Timeout - move to background
                // Note: the spawned task continues running and will update buffers
                let elapsed = timeout.as_secs();
                let partial_output = {
                    let buf = display_buf.lock().await;
                    truncate_with_note(&buf, 1000)
                };

                // The caller got an early return; ensure we notify on completion.
                should_notify.store(true, Ordering::Relaxed);

                // Store the running agent for later checking/cancellation
                let agent = RunningCliAgent {
                    tool_name: tool_name.to_string(),
                    prompt_summary: short_summary.clone(),
                    started_at: started_at_instant,
                    display_buf,
                    stdout_buf,
                    process_handle,
                    finished,
                    session_id: session_id.to_string(),
                    delegated_task_id: delegated_task_id_owned.clone(),
                    working_dir: working_dir_owned,
                };
                self.running.lock().await.insert(task_id.clone(), agent);
                // The reaper / cancel paths own the claim now (no await
                // between the insert above and this disarm).
                if let Some(guard) = claim_guard.as_mut() {
                    guard.disarm();
                }

                Ok(format!(
                    "CLI agent '{}' still running after {}s. Moved to background (task_id={}).\n\
                     Use action=\"check\" with task_id=\"{}\" to see output, or action=\"cancel\" to stop it.\n\n\
                     Partial output:\n{}",
                    tool_name, elapsed, task_id, task_id, partial_output
                ))
            }
        }
    }

    async fn persist_delegated_cli_handoff(
        &self,
        delegated_task_id: &str,
        tool_name: &str,
        prompt: &str,
        working_dir: Option<&str>,
    ) {
        let expected_targets = working_dir
            .and_then(|dir| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, dir))
            .into_iter()
            .collect::<Vec<_>>();
        let handoff = ExecutorHandoff {
            task_id: delegated_task_id.to_string(),
            mission: format!("cli_agent:{tool_name}"),
            task_description: prompt.to_string(),
            target_scope: TargetScope {
                allowed_targets: expected_targets.clone(),
                hard_fail_outside_scope: working_dir.is_some(),
            },
            expected_targets,
            allowed_tools: Some(vec![format!("cli_agent:{tool_name}")]),
        };

        if let Ok(Some(mut task)) = self.state.get_task(delegated_task_id).await {
            task.status = "running".to_string();
            if task.started_at.is_none() {
                task.started_at = Some(chrono::Utc::now().to_rfc3339());
            }
            if let Ok(context) = persist_executor_handoff_context(task.context.as_deref(), &handoff)
            {
                task.context = Some(context);
            }
            let _ = self.state.update_task(&task).await;
        }
    }

    async fn persist_delegated_cli_result(
        &self,
        delegated_task_id: &str,
        response: Option<&str>,
        error: Option<&str>,
    ) {
        Self::persist_delegated_cli_result_with_state(
            self.state.clone(),
            delegated_task_id,
            response,
            error,
        )
        .await;
    }

    /// Check on a background CLI agent task.
    async fn handle_check(&self, task_id: &str, session_id: &str) -> anyhow::Result<String> {
        let running = self.running.lock().await;

        let Some(agent) = running.get(task_id) else {
            drop(running);
            if let Some(done) = self.completed.lock().await.get(task_id).cloned() {
                return Ok(done.result);
            }

            if let Some(invocation) = self
                .state
                .get_cli_agent_invocations(128)
                .await?
                .into_iter()
                .find(|invocation| invocation.task_id.as_deref() == Some(task_id))
            {
                let result = invocation
                    .output_summary
                    .as_deref()
                    .unwrap_or("No persisted output was recorded.");
                return Ok(format!(
                    "CLI agent '{}' {}. Recovered durable result for task_id={task_id}.\n\nResult:\n{}",
                    invocation.agent_name,
                    if invocation.completed_at.is_some() {
                        "finished"
                    } else {
                        "was started but has no recorded completion"
                    },
                    result
                ));
            }

            // Background task identifiers are opaque. Models occasionally
            // copy a provider thread UUID instead of the short runtime ID.
            // Recover an unambiguous session-local task instead of converting
            // that bookkeeping miss into an abandoned user objective.
            let session_candidates = {
                let running = self.running.lock().await;
                let mut candidates = running
                    .iter()
                    .filter(|(_, agent)| agent.session_id == session_id)
                    .map(|(id, agent)| {
                        (
                            id.clone(),
                            format!("running — {}", agent.prompt_summary),
                            None,
                        )
                    })
                    .collect::<Vec<_>>();
                drop(running);

                let completed = self.completed.lock().await;
                candidates.extend(
                    completed
                        .iter()
                        .filter(|(_, done)| done.session_id == session_id)
                        .map(|(id, done)| {
                            (
                                id.clone(),
                                "finished — result available".to_string(),
                                Some(done.result.clone()),
                            )
                        }),
                );
                candidates
            };

            if session_candidates.len() == 1 {
                let (recovered_id, status, result) = &session_candidates[0];
                if let Some(result) = result {
                    return Ok(format!(
                        "Requested task_id '{task_id}' was not found. Recovered the only recent CLI task for this session instead (task_id={recovered_id}, {status}).\n\n{result}"
                    ));
                }
                return Ok(format!(
                    "Requested task_id '{task_id}' was not found. Recovered the only active CLI task for this session (task_id={recovered_id}, {status}). Use action=\"check\" with task_id=\"{recovered_id}\"."
                ));
            }

            let candidate_lines = session_candidates
                .iter()
                .take(8)
                .map(|(id, status, _)| format!("- task_id={id}: {status}"))
                .collect::<Vec<_>>();
            let recovery = if candidate_lines.is_empty() {
                "No recent CLI tasks are tracked for this session. Start a new in-scope run or inspect durable invocation history; do not treat this lookup miss as completion."
                    .to_string()
            } else {
                format!(
                    "Recent CLI tasks for this session:\n{}\nCheck the matching task ID; do not treat this lookup miss as completion.",
                    candidate_lines.join("\n")
                )
            };
            return Ok(format!(
                "No CLI agent task matched task_id '{task_id}'. {recovery}"
            ));
        };

        let elapsed = agent.started_at.elapsed().as_secs();
        let display_output = agent.display_buf.lock().await.clone();

        let is_running = !agent.finished.load(Ordering::Acquire);

        if !is_running {
            Ok(Self::build_finished_result(agent).await)
        } else {
            Ok(format!(
                "CLI agent '{}' still running ({}s elapsed, pid={}).\n\
                 Task: {}...\n\n\
                 Partial output ({} chars):\n{}",
                agent.tool_name,
                elapsed,
                agent.process_handle.display_id(),
                agent.prompt_summary,
                display_output.len(),
                truncate_with_note(&display_output, 5000)
            ))
        }
    }

    /// Cancel a background CLI agent task.
    async fn handle_cancel(&self, task_id: &str) -> anyhow::Result<String> {
        let agent = {
            let mut running = self.running.lock().await;
            let Some(agent) = running.remove(task_id) else {
                return Ok(format!("No running CLI agent with task_id '{}'", task_id));
            };
            agent
        };
        self.completed.lock().await.remove(task_id);

        let display_output = agent.display_buf.lock().await.clone();
        let elapsed = agent.started_at.elapsed().as_secs();

        // Release working-dir claim owned by this task.
        if let Some(ref dir) = agent.working_dir {
            self.release_working_dir_claim(dir, task_id);
        }

        // Try to kill the process
        kill_process(&self.backend, &agent.process_handle).await;

        if let Some(ref delegated_task_id) = agent.delegated_task_id {
            Self::persist_delegated_cli_result_with_state(
                self.state.clone(),
                delegated_task_id,
                None,
                Some("Executor CLI run was cancelled before completion."),
            )
            .await;
        }

        Ok(format!(
            "Cancelled CLI agent '{}' (was running for {}s).\n\nOutput before cancellation:\n{}",
            agent.tool_name,
            elapsed,
            truncate_with_note(&display_output, 5000)
        ))
    }

    /// Cancel all CLI agent tasks for a specific session.
    async fn handle_cancel_all(&self, session_id: &str) -> anyhow::Result<String> {
        // Find all tasks matching this session and remove them from tracking first.
        let to_cancel: Vec<(String, RunningCliAgent)> = {
            let mut running = self.running.lock().await;
            let task_ids: Vec<String> = running
                .iter()
                .filter(|(_, agent)| agent.session_id == session_id)
                .map(|(task_id, _)| task_id.clone())
                .collect();

            let mut removed = Vec::new();
            for task_id in task_ids {
                if let Some(agent) = running.remove(&task_id) {
                    removed.push((task_id, agent));
                }
            }
            removed
        };

        if to_cancel.is_empty() {
            return Ok("No running CLI agents for this session.".to_string());
        }

        let mut cancelled = Vec::new();
        for (task_id, agent) in to_cancel {
            self.completed.lock().await.remove(&task_id);
            if let Some(ref dir) = agent.working_dir {
                self.release_working_dir_claim(dir, &task_id);
            }
            kill_process(&self.backend, &agent.process_handle).await;
            if let Some(ref delegated_task_id) = agent.delegated_task_id {
                Self::persist_delegated_cli_result_with_state(
                    self.state.clone(),
                    delegated_task_id,
                    None,
                    Some("Executor CLI run was cancelled before completion."),
                )
                .await;
            }
            cancelled.push(format!("{} ({})", agent.tool_name, task_id));
        }

        Ok(format!(
            "Cancelled {} CLI agent(s): {}",
            cancelled.len(),
            cancelled.join(", ")
        ))
    }

    /// List all running CLI agent tasks.
    async fn handle_list(&self) -> anyhow::Result<String> {
        let running = self.running.lock().await;

        if running.is_empty() {
            return Ok("No CLI agents currently running.".to_string());
        }

        let mut lines = vec!["Running CLI agents:".to_string()];
        for (task_id, agent) in running.iter() {
            let elapsed = agent.started_at.elapsed().as_secs();
            let status = if !agent.finished.load(Ordering::Acquire) {
                "running"
            } else {
                "finished"
            };
            lines.push(format!(
                "  {} - {} ({}, {}s): {}...",
                task_id, agent.tool_name, status, elapsed, agent.prompt_summary
            ));
        }

        Ok(lines.join("\n"))
    }
}

#[derive(Deserialize)]
struct CliAgentArgs {
    #[serde(default)]
    action: Option<String>,
    tool: Option<String>,
    prompt: Option<String>,
    /// Legacy/incorrect alias sometimes emitted by the model for delegated work.
    mission: Option<String>,
    /// Legacy/incorrect alias sometimes emitted by the model for delegated work.
    task: Option<String>,
    /// Terminal-like alias sometimes emitted instead of `prompt`.
    command: Option<String>,
    /// Optional description paired with `command`.
    description: Option<String>,
    working_dir: Option<String>,
    task_id: Option<String>,
    /// Optional system instruction to shape the CLI agent into a specialist
    system_instruction: Option<String>,
    /// If true, start the task in background and return immediately with task_id
    #[serde(default)]
    async_mode: Option<bool>,
    /// Injected by agent - session ID for cancel_all filtering
    #[serde(default)]
    _session_id: Option<String>,
    /// Injected by agent - goal context for routing async/timeout notifications.
    #[serde(default)]
    _goal_id: Option<String>,
    /// Injected by agent - current task context for delegated task recovery.
    #[serde(default)]
    _task_id: Option<String>,
    /// Injected by agent for role-aware safeguards.
    #[serde(default)]
    _user_role: Option<String>,
    /// Injected by the runtime for owner-authorized scheduled automation.
    #[serde(default)]
    _trusted_session: bool,
}

fn non_empty_prompt_field(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn compose_delegated_prompt(mission: Option<String>, task: Option<String>) -> Option<String> {
    if mission.is_none() && task.is_none() {
        return None;
    }

    let mut parts = Vec::new();
    if let Some(mission) = mission {
        parts.push(format!("Mission: {}", mission));
    }
    if let Some(task) = task {
        parts.push(format!("Task: {}", task));
    }
    Some(parts.join("\n"))
}

impl CliAgentArgs {
    fn run_prompt(&self) -> Option<String> {
        if let Some(prompt) = non_empty_prompt_field(self.prompt.as_deref()) {
            return Some(prompt);
        }

        if let Some(prompt) = compose_delegated_prompt(
            non_empty_prompt_field(self.mission.as_deref()),
            non_empty_prompt_field(self.task.as_deref()),
        ) {
            return Some(prompt);
        }

        let description = non_empty_prompt_field(self.description.as_deref());
        let command = non_empty_prompt_field(self.command.as_deref());
        if description.is_some() || command.is_some() {
            let mut parts = Vec::new();
            if let Some(description) = description {
                parts.push(description);
            }
            if let Some(command) = command {
                parts.push(format!(
                    "Run this exact shell command in the working directory and report the result:\n{}",
                    command
                ));
            }
            return Some(parts.join("\n\n"));
        }

        None
    }

    async fn contextual_run_prompt(&self, state: &Arc<dyn StateStore>) -> Option<String> {
        let delegated_task = match self._task_id.as_deref() {
            Some(task_id) if !task_id.trim().is_empty() => {
                state.get_task(task_id).await.ok().flatten()
            }
            _ => None,
        };
        let delegated_goal_id = self
            ._goal_id
            .clone()
            .or_else(|| delegated_task.as_ref().map(|task| task.goal_id.clone()));
        let delegated_goal = match delegated_goal_id.as_deref() {
            Some(goal_id) if !goal_id.trim().is_empty() => {
                state.get_goal(goal_id).await.ok().flatten()
            }
            _ => None,
        };

        compose_delegated_prompt(
            delegated_goal
                .as_ref()
                .and_then(|goal| non_empty_prompt_field(Some(goal.description.as_str()))),
            delegated_task
                .as_ref()
                .and_then(|task| non_empty_prompt_field(Some(task.description.as_str()))),
        )
    }
}

/// Check if a string looks like JSON (starts with { or [).
fn looks_like_json(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.starts_with('{') || trimmed.starts_with('[')
}

/// Try to extract human-readable progress from a JSON line.
/// Returns None if the line isn't JSON or doesn't contain useful progress info.
fn extract_progress_from_json(line: &str) -> Option<String> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;

    // Claude Code: tool_use events
    if let Some(tool_name) = v.get("name").and_then(|n| n.as_str()) {
        // Tool being used
        if let Some(input) = v.get("input") {
            // Extract key info based on tool type
            if tool_name == "Read" || tool_name == "read" {
                if let Some(path) = input.get("file_path").and_then(|p| p.as_str()) {
                    let short_path: String = path
                        .chars()
                        .rev()
                        .take(50)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect();
                    return Some(format!("📖 Reading: ...{}", short_path));
                }
            } else if tool_name == "Write"
                || tool_name == "write"
                || tool_name == "Edit"
                || tool_name == "edit"
            {
                if let Some(path) = input.get("file_path").and_then(|p| p.as_str()) {
                    let short_path: String = path
                        .chars()
                        .rev()
                        .take(50)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect();
                    return Some(format!("✏️ Writing: ...{}", short_path));
                }
            } else if tool_name == "Bash" || tool_name == "bash" || tool_name == "terminal" {
                if let Some(cmd) = input.get("command").and_then(|c| c.as_str()) {
                    let short_cmd: String = cmd.chars().take(60).collect();
                    return Some(format!("⚡ Running: {}", short_cmd));
                }
            } else if tool_name == "Glob" || tool_name == "glob" {
                if let Some(pattern) = input.get("pattern").and_then(|p| p.as_str()) {
                    return Some(format!("🔍 Searching: {}", pattern));
                }
            } else if tool_name == "Grep" || tool_name == "grep" {
                if let Some(pattern) = input.get("pattern").and_then(|p| p.as_str()) {
                    let short: String = pattern.chars().take(40).collect();
                    return Some(format!("🔍 Grep: {}", short));
                }
            } else {
                return Some(format!("🔧 Using: {}", tool_name));
            }
        }
        return Some(format!("🔧 Using: {}", tool_name));
    }

    // Claude Code: type field events
    if let Some(event_type) = v.get("type").and_then(|t| t.as_str()) {
        match event_type {
            "assistant" => {
                // Assistant is thinking/responding - extract tool use details
                if let Some(content) = v.get("message").and_then(|m| m.get("content")) {
                    if let Some(arr) = content.as_array() {
                        for item in arr {
                            if item.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                                let name = item
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("unknown");
                                let input = item.get("input");

                                // Extract details based on tool type
                                let detail = match name {
                                    "Bash" | "bash" | "terminal" => input
                                        .and_then(|i| i.get("command"))
                                        .and_then(|c| c.as_str())
                                        .map(|cmd| {
                                            let short: String = cmd.chars().take(50).collect();
                                            format!("⚡ {}", short)
                                        }),
                                    "Read" | "read" => input
                                        .and_then(|i| i.get("file_path"))
                                        .and_then(|p| p.as_str())
                                        .map(|path| {
                                            let short: String = path
                                                .chars()
                                                .rev()
                                                .take(40)
                                                .collect::<String>()
                                                .chars()
                                                .rev()
                                                .collect();
                                            format!("📖 ...{}", short)
                                        }),
                                    "Write" | "write" | "Edit" | "edit" => input
                                        .and_then(|i| i.get("file_path"))
                                        .and_then(|p| p.as_str())
                                        .map(|path| {
                                            let short: String = path
                                                .chars()
                                                .rev()
                                                .take(40)
                                                .collect::<String>()
                                                .chars()
                                                .rev()
                                                .collect();
                                            format!("✏️ ...{}", short)
                                        }),
                                    "Glob" | "glob" => input
                                        .and_then(|i| i.get("pattern"))
                                        .and_then(|p| p.as_str())
                                        .map(|pat| format!("🔍 {}", pat)),
                                    "Grep" | "grep" => input
                                        .and_then(|i| i.get("pattern"))
                                        .and_then(|p| p.as_str())
                                        .map(|pat| {
                                            let short: String = pat.chars().take(30).collect();
                                            format!("🔍 grep: {}", short)
                                        }),
                                    "Task" => input
                                        .and_then(|i| i.get("description"))
                                        .and_then(|d| d.as_str())
                                        .map(|desc| format!("🚀 {}", desc)),
                                    _ => None,
                                };

                                return Some(detail.unwrap_or_else(|| format!("🔧 {}", name)));
                            }
                        }
                    }
                }
                return None; // Don't show generic "assistant" events
            }
            "tool_use" => {
                if let Some(name) = v.get("tool").and_then(|n| n.as_str()) {
                    return Some(format!("🔧 Using: {}", name));
                }
            }
            "thinking" => {
                return Some("💭 Thinking...".to_string());
            }
            _ => {}
        }
    }

    None
}

fn prohibited_cli_agent_command_reason(command: &str) -> Option<&'static str> {
    let lower = command.trim().to_ascii_lowercase();

    // E2E incidents: external coding agents tried to "fix scheduling" by hacking the
    // SQLCipher DB directly (sqlite/sqlcipher + key hunting). This tool runs in an
    // auto-approve mode, so we must block obvious state-store manipulation.
    if lower.contains("aidaemon.db")
        || lower.contains("sqlite3")
        || lower.contains("sqlcipher")
        || lower.contains("pragma key")
        || lower.contains("pragma cipher")
    {
        return Some("Direct manipulation of aidaemon's state database is prohibited. Use manage_memories / built-in tools instead of terminal SQL.");
    }

    // Narrow install block for sqlcipher specifically (common rabbit-hole).
    if lower.contains("install") && lower.contains("sqlcipher") {
        return Some(
            "Installing sqlcipher is prohibited in cli_agent runs. Use aidaemon tools instead.",
        );
    }

    None
}

fn extract_terminal_commands_from_json(line: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
        return vec![];
    };

    let mut out = Vec::new();

    // Claude Code stream-json tool_use: { "name": "Bash", "input": {"command": "..."} }
    if let Some(tool_name) = v.get("name").and_then(|n| n.as_str()) {
        if matches!(tool_name, "Bash" | "bash" | "terminal") {
            if let Some(cmd) = v
                .get("input")
                .and_then(|i| i.get("command"))
                .and_then(|c| c.as_str())
            {
                out.push(cmd.to_string());
            }
        }
    }

    // Claude Code assistant wrapper:
    // { "type": "assistant", "message": {"content":[{"type":"tool_use","name":"Bash","input":{"command":"..."}}]}}
    if v.get("type").and_then(|t| t.as_str()) == Some("assistant") {
        if let Some(items) = v
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_array())
        {
            for item in items {
                if item.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
                    continue;
                }
                let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                if !matches!(name, "Bash" | "bash" | "terminal") {
                    continue;
                }
                if let Some(cmd) = item
                    .get("input")
                    .and_then(|i| i.get("command"))
                    .and_then(|c| c.as_str())
                {
                    out.push(cmd.to_string());
                }
            }
        }
    }

    // Generic tool_use format: { "type": "tool_use", "tool": "bash", "command": "..." }
    if v.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
        let tool = v
            .get("tool")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if tool == "bash" || tool == "terminal" {
            if let Some(cmd) = v.get("command").and_then(|c| c.as_str()) {
                out.push(cmd.to_string());
            }
        }
    }

    out
}

/// Try to extract meaningful content from CLI output.
fn extract_meaningful_output(raw: &str, max_chars: usize) -> String {
    // Try JSON extraction first
    if let Some(content) = extract_json_content(raw) {
        return truncate_with_note(&content, max_chars);
    }
    if let Some(content) = extract_jsonl_content(raw) {
        return truncate_with_note(&content, max_chars);
    }
    truncate_with_note(raw, max_chars)
}

/// Try to extract content from JSON output.
fn extract_json_content(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;

    // Claude Code JSON: "result" field
    if let Some(result) = v.get("result").and_then(|r| r.as_str()) {
        return Some(result.to_string());
    }

    // Gemini CLI JSON: "output" field
    if let Some(output) = v.get("output").and_then(|o| o.as_str()) {
        return Some(output.to_string());
    }

    // Generic: "content" or "message" fields
    if let Some(content) = v.get("content").and_then(|c| c.as_str()) {
        return Some(content.to_string());
    }
    if let Some(message) = v.get("message").and_then(|m| m.as_str()) {
        return Some(message.to_string());
    }

    None
}

/// Try to extract content from JSONL output.
fn extract_jsonl_content(raw: &str) -> Option<String> {
    let mut last_content: Option<String> = None;
    for line in raw.lines().rev() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            // Codex CLI JSONL emits the final response as:
            // {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
            //
            // Command events can contain very large `aggregated_output` values.
            // If this final message is not recognized, `extract_meaningful_output`
            // falls back to truncating the raw stream from the front and loses the
            // authoritative outcome. That can make a completed side effect look
            // unfinished and cause the orchestrator to repeat it.
            if v.get("type").and_then(Value::as_str) == Some("item.completed")
                && v.pointer("/item/type").and_then(Value::as_str) == Some("agent_message")
            {
                if let Some(text) = v.pointer("/item/text").and_then(Value::as_str) {
                    return Some(text.to_string());
                }
            }

            if let Some(content) = v
                .pointer("/item/content")
                .or_else(|| v.pointer("/content"))
                .or_else(|| v.pointer("/result"))
            {
                if let Some(text) = content.as_str() {
                    last_content = Some(text.to_string());
                    break;
                }
                if let Some(arr) = content.as_array() {
                    let texts: Vec<&str> = arr
                        .iter()
                        .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                        .collect();
                    if !texts.is_empty() {
                        last_content = Some(texts.join("\n"));
                        break;
                    }
                }
            }
        }
    }
    last_content
}

#[async_trait]
impl Tool for CliAgentTool {
    fn name(&self) -> &str {
        "cli_agent"
    }

    fn description(&self) -> &str {
        "Delegate a task to a CLI-based AI coding agent running on this machine"
    }

    fn schema(&self) -> Value {
        let tool_names = self.tool_names.read().unwrap();

        let tools_help = tool_names.join(", ");
        let names_vec: Vec<Value> = tool_names.iter().map(|n| json!(n)).collect();

        json!({
            "name": "cli_agent",
            "description": format!(
                "Delegate complex multi-step coding/research/analysis work to an installed CLI agent. \
                 Available agents: {}. If `tool` is omitted, the runtime auto-selects the first installed \
                 agent (claude, gemini, codex, copilot, aider). Use manage_memories for scheduling. \
                 Long runs can be checked or cancelled.",
                tools_help
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run", "check", "cancel", "list"],
                        "description": "run, check, cancel, or list"
                    },
                    "tool": {
                        "type": "string",
                        "enum": names_vec,
                        "description": "Agent name"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Task prompt"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Absolute working directory. Always set this so the runtime can detect conflicts; two agents must not run concurrently in the same working_dir."
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID"
                    },
                    "system_instruction": {
                        "type": "string",
                        "description": "Optional system instruction shaping the agent into a specialist (e.g. 'You are a security auditor')"
                    },
                    "async_mode": {
                        "type": "boolean",
                        "description": "Run in background; use true to dispatch independent sub-tasks in parallel"
                    }
                },
                "required": ["action"],
                "additionalProperties": false,
                "anyOf": [
                    {
                        "required": ["action", "prompt"],
                        "properties": {
                            "action": {
                                "enum": ["run"]
                            }
                        }
                    },
                    {
                        "required": ["action", "task_id"],
                        "properties": {
                            "action": {
                                "enum": ["check", "cancel"]
                            }
                        }
                    },
                    {
                        "required": ["action"],
                        "properties": {
                            "action": {
                                "enum": ["list"]
                            }
                        }
                    }
                ]
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

    fn is_available(&self) -> bool {
        self.has_tools()
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        // For backwards compatibility, delegate to call_with_status with no sender
        self.call_with_status(arguments, None).await
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let action = parsed
            .as_ref()
            .and_then(|value| value.get("action"))
            .and_then(Value::as_str)
            .unwrap_or("run");
        let mut semantics = match action {
            "list" | "check" => ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
            // A delegated run may both inspect and change state. Keep the
            // effect conservative for preflight policy, but omit a synthetic
            // mutation-effect bit so a structured pre-launch failure can
            // authoritatively downgrade the completed outcome to administrative.
            "run" => ToolCallSemantics::observation_and_mutation()
                .with_verification_mode(ToolVerificationMode::ResultContent),
            "cancel" | "cancel_all" => ToolCallSemantics::mutation(),
            _ => ToolCallSemantics::mutation(),
        };
        if let Some(working_dir) = parsed
            .as_ref()
            .and_then(|value| value.get("working_dir"))
            .and_then(Value::as_str)
        {
            semantics = semantics.with_target_hint(ToolTargetHintKind::ProjectScope, working_dir);
        }
        semantics
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| {
                value
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "run".to_string());
        let output = self
            .call_with_status_under_contract(arguments, status_tx, false)
            .await?;
        Ok(cli_agent_outcome(&action, output, false))
    }

    async fn call_with_execution_context(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        exec_ctx: ToolExecutionContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| {
                value
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "run".to_string());
        let output = self
            .call_with_status_under_contract(arguments, status_tx, exec_ctx.mutation_forbidden)
            .await?;
        Ok(cli_agent_outcome(
            &action,
            output,
            exec_ctx.mutation_forbidden && action == "run",
        ))
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        self.call_with_status_under_contract(arguments, status_tx, false)
            .await
    }
}

fn cli_agent_outcome(action: &str, output: String, read_only_run: bool) -> ToolCallOutcome {
    let exit_code = cli_agent_exit_code(&output);
    let prelaunch_failure = failed_before_agent_launch(&output, exit_code);
    let policy_block = output.trim_start().starts_with("Blocked:");
    let reported_failure = output.trim_start().starts_with("ERROR: CLI agent");
    let metadata = if exit_code.is_some() || prelaunch_failure || policy_block || reported_failure {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::FailedPermanent),
            exit_code,
            transport_error: prelaunch_failure.then(|| first_cli_failure_detail(&output)),
            semantics: if prelaunch_failure || policy_block || reported_failure {
                ToolCallSemantics::administrative()
            } else {
                ToolCallSemantics::default()
            },
            ..ToolCallMetadata::default()
        }
    } else if output.contains("started in background")
        || output.contains("Moved to background")
        || output.contains("still running")
        || output.contains("only active CLI task for this session")
    {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::Backgrounded),
            background_started: true,
            completion_notifications_enabled: action == "run",
            ..ToolCallMetadata::default()
        }
    } else if action == "check"
        && (output.starts_with("No CLI agent task matched")
            || output.starts_with("No running CLI agent"))
    {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::CompletedWithNegativeResult),
            ..ToolCallMetadata::default()
        }
    } else {
        ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::Succeeded),
            semantics: if read_only_run {
                ToolCallSemantics::observation()
                    .with_verification_mode(ToolVerificationMode::ResultContent)
            } else {
                ToolCallSemantics::default()
            },
            ..ToolCallMetadata::default()
        }
    };
    ToolCallOutcome { output, metadata }
}

impl CliAgentTool {
    async fn call_with_status_under_contract(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        mutation_forbidden: bool,
    ) -> anyhow::Result<String> {
        let args: CliAgentArgs = serde_json::from_str(arguments)?;

        // Reap any finished tasks
        self.reap_finished().await;

        let action = args.action.as_deref().unwrap_or("run");

        // Default to the caller session (often an internal child-agent session).
        // If this CLI agent is running in the context of an orchestration goal,
        // route background notifications and cancel_all scoping to the *origin*
        // session_id (goal.session_id), which is routable via the ChannelHub.
        let mut session_id = args._session_id.clone().unwrap_or_default();
        if let Some(ref goal_id) = args._goal_id {
            if let Ok(Some(goal)) = self.state.get_goal(goal_id).await {
                if !goal.session_id.trim().is_empty() {
                    session_id = goal.session_id;
                }
            }
        }

        match action {
            "run" => {
                let mut tool = args
                    .tool
                    .as_deref()
                    .and_then(|name| self.configured_tool(name))
                    .or_else(|| self.default_tool_name())
                    .ok_or_else(|| anyhow::anyhow!("No CLI agents available for action=run"))?;
                let prompt = if let Some(prompt) = args.run_prompt() {
                    prompt
                } else if let Some(prompt) = args.contextual_run_prompt(&self.state).await {
                    warn!(
                        task_id = ?args._task_id,
                        goal_id = ?args._goal_id,
                        working_dir = ?args.working_dir,
                        "cli_agent run omitted prompt; synthesized delegated prompt from stored task/goal context"
                    );
                    prompt
                } else {
                    warn!(
                        task_id = ?args._task_id,
                        goal_id = ?args._goal_id,
                        has_prompt = args.prompt.as_ref().is_some_and(|s| !s.trim().is_empty()),
                        has_mission = args.mission.as_ref().is_some_and(|s| !s.trim().is_empty()),
                        has_task = args.task.as_ref().is_some_and(|s| !s.trim().is_empty()),
                        has_description = args
                            .description
                            .as_ref()
                            .is_some_and(|s| !s.trim().is_empty()),
                        has_command = args.command.as_ref().is_some_and(|s| !s.trim().is_empty()),
                        "cli_agent run missing prompt and had no recoverable delegated context"
                    );
                    anyhow::bail!("Missing 'prompt' parameter for action=run");
                };

                // Trusted sessions bypass the interactive approval boundary.
                // A CLI delegate is an opaque, mutation-capable executor, so
                // prompt wording cannot narrow its effects. Require a concrete
                // clean repository for every unattended run rather than trying
                // to recognize deployment verbs or negations in prose.
                if args._trusted_session {
                    let Some(working_dir) = args.working_dir.as_deref() else {
                        return Ok(
                            "Blocked: unattended CLI delegation requires an explicit working_dir so repository cleanliness can be verified."
                                .to_string(),
                        );
                    };
                    if let Some(entry_count) = Self::dirty_worktree_entry_count(working_dir)
                        .await
                        .filter(|count| *count > 0)
                    {
                        return Ok(format!(
                            "Blocked: unattended CLI delegation found {entry_count} pre-existing changed or untracked worktree entries. Use a clean isolated worktree containing only the intended change, or request explicit owner review; unrelated workspace state was not exposed to opaque delegated mutation."
                        ));
                    }
                }

                let mut daemon_hits = detect_daemonization_primitives(&prompt);
                if let Some(system_instruction) = args.system_instruction.as_deref() {
                    for hit in detect_daemonization_primitives(system_instruction) {
                        if !daemon_hits.contains(&hit) {
                            daemon_hits.push(hit);
                        }
                    }
                }
                if !daemon_hits.is_empty() {
                    if !Self::is_owner_role(args._user_role.as_deref()) {
                        return Ok(format!(
                            "Blocked: daemonization primitives detected in cli_agent prompt ({}) and only owners can approve detached/background execution.",
                            daemon_hits.join(", ")
                        ));
                    }
                    if session_id.trim().is_empty() {
                        return Ok(
                            "Blocked: daemonization primitives require owner approval in an interactive session, but no session_id was provided."
                                .to_string(),
                        );
                    }
                    match self
                        .request_daemonization_approval(
                            session_id.trim(),
                            &tool,
                            &prompt,
                            &daemon_hits,
                        )
                        .await
                    {
                        Ok(ApprovalResponse::Deny) => {
                            return Ok("Daemonizing cli_agent run denied by owner.".to_string());
                        }
                        Ok(
                            ApprovalResponse::AllowOnce
                            | ApprovalResponse::AllowSession
                            | ApprovalResponse::AllowAlways,
                        ) => {}
                        Err(e) => {
                            return Ok(format!(
                                "Could not get owner approval for daemonizing cli_agent run: {}",
                                e
                            ));
                        }
                    }
                }
                // Goal-scoped cli_agent runs should not detach immediately:
                // the goal/task lead is already running in the background, and
                // returning early makes results easy to drop.
                let async_mode = if args._goal_id.is_some() {
                    false
                } else {
                    args.async_mode.unwrap_or(false)
                };

                let mut unavailable = Vec::new();
                loop {
                    match self
                        .handle_run(
                            &tool,
                            &prompt,
                            args.working_dir.as_deref(),
                            &session_id,
                            args._goal_id.as_deref(),
                            args._task_id.as_deref(),
                            args.system_instruction.as_deref(),
                            if mutation_forbidden {
                                CliWorkspaceMode::ReadOnly
                            } else {
                                CliWorkspaceMode::ReadWrite
                            },
                            async_mode,
                            status_tx.clone(),
                        )
                        .await
                    {
                        Ok(output) => return Ok(output),
                        Err(error) => {
                            let Some(failure) = error.downcast_ref::<CliAgentUnavailableError>()
                            else {
                                return Err(error);
                            };
                            unavailable.push((failure.tool_name.clone(), failure.message.clone()));

                            let Some(next_tool) = self.default_tool_name() else {
                                let mut message = String::from(
                                    "ERROR: CLI agent recovery exhausted before task execution. \
                                     No configured CLI agent was able to start the delegated task.",
                                );
                                for (name, detail) in &unavailable {
                                    message.push_str(&format!("\n- {name}: {detail}"));
                                }
                                if let Some(task_id) = args._task_id.as_deref() {
                                    self.persist_delegated_cli_result(
                                        task_id,
                                        None,
                                        Some(&message),
                                    )
                                    .await;
                                }
                                return Ok(message);
                            };

                            info!(
                                unavailable_tool = %tool,
                                fallback_tool = %next_tool,
                                "Retrying delegated task with another configured CLI agent"
                            );
                            tool = next_tool;
                        }
                    }
                }
            }
            "check" => {
                let task_id = args.task_id.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Missing 'task_id' parameter for action=check")
                })?;
                self.handle_check(task_id, &session_id).await
            }
            "cancel" => {
                let task_id = args.task_id.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Missing 'task_id' parameter for action=cancel")
                })?;
                self.handle_cancel(task_id).await
            }
            "cancel_all" => self.handle_cancel_all(&session_id).await,
            "list" => self.handle_list().await,
            _ => Ok(format!(
                "Unknown action '{}'. Use run, check, cancel, cancel_all, or list.",
                action
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CliAgentsConfig;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::testing::MockProvider;
    use crate::traits::store_prelude::*;
    use crate::traits::Tool;
    use crate::traits::{Goal, Task};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn env_wrapper_finds_nested_program_after_unset_option() {
        let args = vec![
            "-u".to_string(),
            "CLAUDECODE".to_string(),
            "claude".to_string(),
            "--print".to_string(),
        ];
        assert_eq!(env_wrapped_program_index(&args), Some(2));
    }

    #[test]
    fn codex_read_only_adapter_replaces_unrestricted_launch_mode() {
        let mut args = vec![
            "exec".to_string(),
            "--json".to_string(),
            "--dangerously-bypass-approvals-and-sandbox".to_string(),
            "--sandbox=workspace-write".to_string(),
            "--add-dir".to_string(),
            "/synthetic/extra".to_string(),
        ];
        apply_read_only_cli_adapter("/synthetic/bin/codex", &mut args).unwrap();
        assert!(!args
            .iter()
            .any(|arg| arg == "--dangerously-bypass-approvals-and-sandbox"));
        assert!(!args.iter().any(|arg| arg.contains("workspace-write")));
        assert!(!args.iter().any(|arg| arg == "--add-dir"));
        assert!(!args.iter().any(|arg| arg == "/synthetic/extra"));
        assert!(args.iter().any(|arg| arg == "--ignore-user-config"));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--sandbox", "read-only"]));
    }

    #[test]
    fn unknown_cli_cannot_claim_read_only_delegation() {
        let mut args = vec!["--auto-approve".to_string()];
        let error =
            apply_read_only_cli_adapter("/synthetic/bin/opaque-agent", &mut args).unwrap_err();
        assert!(error.contains("no registered hard read-only adapter"));
    }

    #[test]
    fn cli_failure_formatter_preserves_diagnostic_and_safe_recovery() {
        let formatted = format_cli_agent_failure(
            "claude",
            Some(127),
            "[stderr] env: claude: No such file or directory\n",
            10_000,
            None,
        );
        assert!(formatted.contains("exit code 127"));
        assert!(formatted.contains("env: claude: No such file or directory"));
        assert!(!formatted.contains("git checkout ."));
    }

    #[test]
    fn exit_127_env_failure_is_identified_as_prelaunch() {
        let output = "ERROR: CLI agent 'claude' failed (exit code 127).\n\n\
## Failure Details\n[stderr] env: claude: No such file or directory";
        let exit_code = cli_agent_exit_code(output);
        assert_eq!(exit_code, Some(127));
        assert!(failed_before_agent_launch(output, exit_code));
    }

    #[test]
    fn later_command_not_found_is_not_assumed_to_be_prelaunch() {
        let output = "ERROR: CLI agent 'claude' failed (exit code 127).\n\n\
## Failure Details\n[stderr] sh: deploy-helper: command not found";
        assert!(!failed_before_agent_launch(
            output,
            cli_agent_exit_code(output)
        ));
    }

    #[tokio::test]
    async fn wrapped_missing_executable_returns_structured_prelaunch_failure() {
        let (tool, _db_file) = setup_echo_tool().await;
        {
            let mut tools = tool.tools.write().unwrap();
            tools.clear();
            tools.insert(
                "broken".to_string(),
                CliToolEntry {
                    command: "/usr/bin/env".to_string(),
                    args: vec!["aidaemon-definitely-missing-cli".to_string()],
                    description: "Broken test agent".to_string(),
                    timeout: Duration::from_secs(10),
                    max_output_chars: 10_000,
                    is_dynamic: false,
                },
            );
        }
        *tool.tool_names.write().unwrap() = vec!["broken".to_string()];

        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "action": "run",
                    "tool": "broken",
                    "prompt": "Inspect the repository"
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(outcome.metadata.transport_error.is_some());
        assert!(!outcome.metadata.semantics.mutates_state());
        assert!(outcome.output.contains("could not start"));
        assert!(outcome.output.contains("aidaemon-definitely-missing-cli"));
    }

    fn extract_task_id_from_background_message(msg: &str) -> String {
        let marker = "task_id=";
        let start = msg
            .find(marker)
            .expect("background response should include task_id")
            + marker.len();
        msg[start..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect()
    }

    #[test]
    fn background_cli_completion_reengages_the_original_outcome() {
        let followup = CliAgentTool::build_background_reengagement_followup(
            "codex",
            "abc12345",
            "Investigate the failure",
            "completed",
            "The root cause is in scheduler.rs",
        );

        assert!(followup.starts_with("[Background command completed]"));
        assert!(followup.contains("Runtime task ID: abc12345"));
        assert!(followup.contains("previous unfinished request"));
        assert!(followup.contains("complete every remaining implementation"));
        assert!(followup.contains("The root cause is in scheduler.rs"));
    }

    #[test]
    fn background_progress_is_elapsed_state_not_prompt_wording() {
        let message = format_background_progress(
            "codex",
            "Investigate an arbitrary unfinished operation",
            240,
            Some("Running focused validation"),
        );

        assert!(message.contains("still running (240s)"));
        assert!(message.contains("Worker: codex"));
        assert!(message.contains("Latest activity: Running focused validation"));
    }

    /// Create a CliAgentTool with `echo` registered as a test tool.
    /// Uses a real temp-file SQLite DB for state persistence.
    async fn setup_echo_tool() -> (CliAgentTool, tempfile::NamedTempFile) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let provider = Arc::new(MockProvider::new());
        let (approval_tx_raw, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = ApprovalBroker::new(approval_tx_raw);

        let mut tools_map = HashMap::new();
        tools_map.insert(
            "echo".to_string(),
            CliToolEntry {
                command: "echo".to_string(),
                args: vec![],
                description: "Echo agent for testing".to_string(),
                timeout: Duration::from_secs(10),
                max_output_chars: 10000,
                is_dynamic: false,
            },
        );

        let tool = CliAgentTool {
            backend: active_execution_backend(),
            tools: Arc::new(std::sync::RwLock::new(tools_map)),
            tool_names: Arc::new(std::sync::RwLock::new(vec!["echo".to_string()])),
            running: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
            state: state as Arc<dyn StateStore>,
            llm_runtime: SharedLlmRuntime::new(
                provider as Arc<dyn crate::traits::ModelProvider>,
                None,
                crate::config::ProviderKind::OpenaiCompatible,
                "mock".to_string(),
            ),
            default_timeout: Duration::from_secs(10),
            default_max_output: 10000,
            max_concurrent: 3,
            concurrency_limiter: Arc::new(Semaphore::new(3)),
            approval_tx,
            hub: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
        };

        (tool, db_file)
    }

    /// Create a CliAgentTool with `bash` registered, for testing scripts.
    async fn setup_bash_tool() -> (CliAgentTool, tempfile::NamedTempFile) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let provider = Arc::new(MockProvider::new());
        let (approval_tx_raw, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = ApprovalBroker::new(approval_tx_raw);

        let mut tools_map = HashMap::new();
        tools_map.insert(
            "bash-agent".to_string(),
            CliToolEntry {
                command: "bash".to_string(),
                args: vec!["-c".to_string()],
                description: "Bash agent for testing".to_string(),
                timeout: Duration::from_secs(10),
                max_output_chars: 10000,
                is_dynamic: false,
            },
        );

        let tool = CliAgentTool {
            backend: active_execution_backend(),
            tools: Arc::new(std::sync::RwLock::new(tools_map)),
            tool_names: Arc::new(std::sync::RwLock::new(vec!["bash-agent".to_string()])),
            running: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
            state: state as Arc<dyn StateStore>,
            llm_runtime: SharedLlmRuntime::new(
                provider as Arc<dyn crate::traits::ModelProvider>,
                None,
                crate::config::ProviderKind::OpenaiCompatible,
                "mock".to_string(),
            ),
            default_timeout: Duration::from_secs(10),
            default_max_output: 10000,
            max_concurrent: 3,
            concurrency_limiter: Arc::new(Semaphore::new(3)),
            approval_tx,
            hub: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
        };

        (tool, db_file)
    }

    // -----------------------------------------------------------------------
    // Basic run tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_echo_returns_output() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool
            .call(r#"{"action":"run","tool":"echo","prompt":"hello world"}"#)
            .await
            .unwrap();
        assert!(
            result.contains("hello world"),
            "Expected 'hello world' in output, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_run_bash_script_returns_output() {
        let (tool, _db) = setup_bash_tool().await;
        let result = tool
            .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo 'test output 42'"}"#)
            .await
            .unwrap();
        assert!(
            result.contains("test output 42"),
            "Expected 'test output 42' in output, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_run_daemonization_prompt_blocked_for_non_owner() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool
            .call(
                r#"{"action":"run","tool":"echo","prompt":"nohup echo hi &","_session_id":"sess1","_user_role":"Guest"}"#,
            )
            .await
            .unwrap();
        assert!(
            result.contains("Blocked: daemonization primitives"),
            "Expected daemonization guard block, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_run_captures_exit_code_failure() {
        let (tool, _db) = setup_bash_tool().await;
        let result = tool
            .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo 'failing' >&2; exit 1"}"#)
            .await
            .unwrap();
        assert!(
            result.contains("ERROR"),
            "Expected ERROR in output for exit code 1, got: {}",
            result
        );
        assert!(
            result.contains("failing"),
            "Expected stderr in error output, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_run_unavailable_requested_tool_uses_configured_fallback() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool
            .call(r#"{"action":"run","tool":"nonexistent","prompt":"test"}"#)
            .await
            .unwrap();
        assert!(result.contains("test"), "got: {result}");
    }

    #[tokio::test]
    async fn test_run_missing_tool_param() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool
            .call(r#"{"action":"run","prompt":"test"}"#)
            .await
            .unwrap();
        assert!(
            result.contains("test"),
            "Expected default tool fallback output, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_run_missing_prompt_param() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool.call(r#"{"action":"run","tool":"echo"}"#).await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Stdin hang prevention test (the critical fix)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_completes_without_hanging_on_stdin() {
        // This is THE critical test: before the fix, piped stdin would cause
        // CLI agents to hang waiting for EOF. With stdin set to null, the
        // process should complete quickly.
        let (tool, _db) = setup_bash_tool().await;

        let start = Instant::now();
        let result = tool
            .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo 'done'; exit 0"}"#)
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert!(
            result.contains("done"),
            "Expected 'done' in output, got: {}",
            result
        );
        // Should complete in well under 5 seconds (the hang was 5+ minutes)
        assert!(
            elapsed < Duration::from_secs(5),
            "Process took {:?} — likely hanging on stdin",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_cat_stdin_completes_quickly() {
        // `cat` without args reads from stdin — with Stdio::null() it should
        // get immediate EOF and exit. With Stdio::piped() it would hang forever.
        let (tool, _db) = setup_bash_tool().await;

        let start = Instant::now();
        let result = tool
            .call(r#"{"action":"run","tool":"bash-agent","prompt":"cat; echo 'cat done'"}"#)
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert!(
            result.contains("cat done"),
            "Expected 'cat done' in output, got: {}",
            result
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "`cat` took {:?} — stdin not null?",
            elapsed
        );
    }

    // -----------------------------------------------------------------------
    // Concurrent limit test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_concurrent_limit_enforced() {
        let (tool, _db) = setup_bash_tool().await;

        // Start 3 long-running async tasks to fill the concurrent limit
        // (real processes so reap_finished won't clean them up)
        for _ in 0..3 {
            tool.call(
                r#"{"action":"run","tool":"bash-agent","prompt":"sleep 30","async_mode":true}"#,
            )
            .await
            .unwrap();
        }

        // The 4th should be rejected
        let result = tool
            .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo should-not-run"}"#)
            .await
            .unwrap();
        assert!(
            result.contains("Maximum 3 CLI agents already running"),
            "Expected concurrent limit message, got: {}",
            result
        );

        // Clean up: cancel all
        tool.handle_cancel_all("").await.unwrap();
    }

    // -----------------------------------------------------------------------
    // Working directory claim tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_working_dir_conflict_detection() {
        let (tool, _db) = setup_bash_tool().await;

        // Seed a claim as if another task already owns this directory.
        lock_claims(&tool.working_dir_claims).insert(
            "/tmp/project".to_string(),
            WorkingDirClaim {
                task_id: "abc12345".to_string(),
                tool_name: "bash-agent".to_string(),
                prompt_summary: "sleep 60".to_string(),
                dedup_prompt: make_dedup_prompt("sleep 60"),
            },
        );

        let result = tool
            .call(
                r#"{"action":"run","tool":"bash-agent","prompt":"echo test","working_dir":"/tmp/project"}"#,
            )
            .await
            .unwrap();
        assert!(
            result.contains("BLOCKED") && result.contains("Another CLI agent"),
            "Expected working dir conflict BLOCKED message, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_working_dir_lock_released_after_completion() {
        let (tool, _db) = setup_bash_tool().await;
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let dir_path = tmp_dir.path().to_str().unwrap();

        let args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"echo done","working_dir":"{}"}}"#,
            dir_path
        );
        let result = tool.call(&args).await.unwrap();
        assert!(result.contains("done"));

        // Working-dir claim should be released after completion.
        let claims = lock_claims(&tool.working_dir_claims);
        assert!(
            !claims.contains_key(dir_path),
            "Working-dir claim not released after completion"
        );
    }

    #[tokio::test]
    async fn test_working_dir_aliases_conflict_after_normalization() {
        let (tool, _db) = setup_bash_tool().await;
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let dir = tmp_dir.path().to_string_lossy().to_string();
        let dir_with_slash = format!("{}/", dir.trim_end_matches('/'));

        let first_args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 5","working_dir":"{}","async_mode":true}}"#,
            dir
        );
        let first = tool.call(&first_args).await.unwrap();
        assert!(
            first.contains("started in background"),
            "Expected first task to start in background, got: {}",
            first
        );

        let second_args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"echo test","working_dir":"{}"}}"#,
            dir_with_slash
        );
        let second = tool.call(&second_args).await.unwrap();
        assert!(
            second.contains("BLOCKED") && second.contains("Another CLI agent"),
            "Expected normalized path conflict BLOCKED message, got: {}",
            second
        );

        tool.handle_cancel_all("").await.unwrap();
    }

    #[tokio::test]
    async fn test_working_dir_lock_released_on_spawn_failure() {
        let (tool, _db) = setup_bash_tool().await;
        let missing_dir = "/tmp/aidaemon-cli-agent-missing-dir-lock-test";
        let _ = std::fs::remove_dir_all(missing_dir);
        let normalized = missing_dir.to_string();

        let args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"echo should-fail","working_dir":"{}"}}"#,
            missing_dir
        );
        let result = tool.call(&args).await;
        assert!(
            result.is_err(),
            "Expected spawn failure for missing current_dir, got: {:?}",
            result
        );

        let claims = lock_claims(&tool.working_dir_claims);
        assert!(
            !claims.contains_key(&normalized),
            "Working-dir claim should be released after spawn failure"
        );
    }

    #[tokio::test]
    async fn test_concurrent_limit_includes_sync_runs() {
        let (mut tool, _db) = setup_bash_tool().await;
        tool.max_concurrent = 1;
        tool.concurrency_limiter = Arc::new(Semaphore::new(1));
        let tool = Arc::new(tool);

        let tool_for_first = Arc::clone(&tool);
        let first = tokio::spawn(async move {
            tool_for_first
                .call(r#"{"action":"run","tool":"bash-agent","prompt":"sleep 2; echo first"}"#)
                .await
        });

        tokio::time::sleep(Duration::from_millis(150)).await;

        let second = tool
            .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo second"}"#)
            .await
            .unwrap();
        assert!(
            second.contains("Maximum 1 CLI agents already running"),
            "Expected concurrency rejection while sync run is active, got: {}",
            second
        );

        let first_result = first.await.unwrap().unwrap();
        assert!(
            first_result.contains("first"),
            "Expected first sync run output, got: {}",
            first_result
        );
    }

    #[tokio::test]
    async fn test_sync_inflight_conflict_reports_claim_metadata() {
        let (tool, _db) = setup_bash_tool().await;
        let tool = Arc::new(tool);
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let dir = tmp_dir.path().to_string_lossy().to_string();

        let first_args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 2; echo first","working_dir":"{}"}}"#,
            dir
        );
        let tool_for_first = Arc::clone(&tool);
        let first = tokio::spawn(async move { tool_for_first.call(&first_args).await.unwrap() });

        tokio::time::sleep(Duration::from_millis(150)).await;

        let second_args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"echo second","working_dir":"{}"}}"#,
            dir
        );
        let second = tool.call(&second_args).await.unwrap();
        assert!(
            second.contains("BLOCKED")
                && second.contains("task_id=")
                && second.contains("agent=bash-agent"),
            "Expected conflict response with claim metadata, got: {}",
            second
        );

        let first_result = first.await.unwrap();
        assert!(
            first_result.contains("first"),
            "Expected first sync run output, got: {}",
            first_result
        );
    }

    #[tokio::test]
    async fn test_sync_inflight_duplicate_prompt_blocked() {
        let (tool, _db) = setup_bash_tool().await;
        let tool = Arc::new(tool);
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let dir = tmp_dir.path().to_string_lossy().to_string();

        let first_args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"echo refactor the auth module to use OAuth tokens; sleep 2","working_dir":"{}"}}"#,
            dir
        );
        let tool_for_first = Arc::clone(&tool);
        let first = tokio::spawn(async move { tool_for_first.call(&first_args).await.unwrap() });

        tokio::time::sleep(Duration::from_millis(150)).await;

        let second_args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"echo refactor the auth module to use OAuth tokens","working_dir":"{}"}}"#,
            dir
        );
        let second = tool.call(&second_args).await.unwrap();
        assert!(
            second.contains("BLOCKED") && second.contains("similar task"),
            "Expected duplicate-task block for sync in-flight run, got: {}",
            second
        );

        let _ = first.await.unwrap();
    }

    #[tokio::test]
    async fn test_working_dir_claim_released_on_semaphore_reject() {
        let (mut tool, _db) = setup_bash_tool().await;
        tool.max_concurrent = 0;
        tool.concurrency_limiter = Arc::new(Semaphore::new(0));

        let tmp_dir = tempfile::TempDir::new().unwrap();
        let dir_path = tmp_dir.path().to_str().unwrap();
        let normalized = CliAgentTool::normalize_working_dir(dir_path).await.unwrap();
        let args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"echo blocked","working_dir":"{}"}}"#,
            dir_path
        );
        let result = tool.call(&args).await.unwrap();
        assert!(
            result.contains("Maximum 0 CLI agents already running"),
            "Expected semaphore rejection, got: {}",
            result
        );

        let claims = lock_claims(&tool.working_dir_claims);
        assert!(
            !claims.contains_key(&normalized),
            "Working-dir claim should be released after semaphore rejection"
        );
    }

    /// Regression test: a sync-mode dispatch future dropped mid-await (the
    /// calling session was cancelled or timed out) must release its
    /// working-dir claim. Before WorkingDirClaimGuard, this leaked the claim
    /// permanently and blocked all future dispatches to the directory.
    #[tokio::test]
    async fn test_working_dir_claim_released_when_dispatch_future_dropped() {
        let (tool, _db) = setup_bash_tool().await;
        let tool = Arc::new(tool);
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let dir = tmp_dir.path().to_string_lossy().to_string();
        let normalized = CliAgentTool::normalize_working_dir(&dir).await.unwrap();

        let args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 15","working_dir":"{}"}}"#,
            dir
        );
        let tool_for_task = Arc::clone(&tool);
        let handle = tokio::spawn(async move { tool_for_task.call(&args).await });

        // Wait until the dispatch has registered its claim.
        let mut claimed = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if lock_claims(&tool.working_dir_claims).contains_key(&normalized) {
                claimed = true;
                break;
            }
        }
        assert!(claimed, "Working-dir claim was never registered");

        // Abort the dispatching future mid-await (simulates the calling
        // agent session being cancelled while waiting for completion).
        handle.abort();
        let _ = handle.await;

        assert!(
            !lock_claims(&tool.working_dir_claims).contains_key(&normalized),
            "Working-dir claim should be released when the dispatch future is dropped"
        );
    }

    /// The guard must NOT release the claim once disarmed (background
    /// handoff transfers ownership to the reaper / cancel paths).
    #[tokio::test]
    async fn test_claim_guard_disarm_keeps_claim() {
        let claims: Arc<WorkingDirClaims> = Arc::new(std::sync::Mutex::new(HashMap::new()));
        lock_claims(&claims).insert(
            "/tmp/project".to_string(),
            WorkingDirClaim {
                task_id: "abc12345".to_string(),
                tool_name: "bash-agent".to_string(),
                prompt_summary: "sleep 60".to_string(),
                dedup_prompt: make_dedup_prompt("sleep 60"),
            },
        );

        let mut guard = WorkingDirClaimGuard::new(
            Arc::clone(&claims),
            "/tmp/project".to_string(),
            "abc12345".to_string(),
        );
        guard.disarm();
        drop(guard);
        assert!(
            lock_claims(&claims).contains_key("/tmp/project"),
            "Disarmed guard must not release the claim"
        );

        // An armed guard owned by a DIFFERENT task must not steal the claim.
        let other_guard = WorkingDirClaimGuard::new(
            Arc::clone(&claims),
            "/tmp/project".to_string(),
            "other999".to_string(),
        );
        drop(other_guard);
        assert!(
            lock_claims(&claims).contains_key("/tmp/project"),
            "Guard for a different task must not release someone else's claim"
        );

        // The owning guard releases it.
        let owner_guard = WorkingDirClaimGuard::new(
            Arc::clone(&claims),
            "/tmp/project".to_string(),
            "abc12345".to_string(),
        );
        drop(owner_guard);
        assert!(
            !lock_claims(&claims).contains_key("/tmp/project"),
            "Owning guard must release the claim on drop"
        );
    }

    // -----------------------------------------------------------------------
    // Async mode test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_async_mode_returns_immediately() {
        let (tool, _db) = setup_bash_tool().await;

        let start = Instant::now();
        let result = tool
            .call(
                r#"{"action":"run","tool":"bash-agent","prompt":"sleep 2; echo async-done","async_mode":true}"#,
            )
            .await
            .unwrap();
        let elapsed = start.elapsed();

        // Should return immediately (< 1s) with a task_id
        assert!(
            elapsed < Duration::from_secs(1),
            "Async mode took {:?} — not returning immediately",
            elapsed
        );
        assert!(
            result.contains("started in background"),
            "Expected background message, got: {}",
            result
        );
        assert!(
            result.contains("task_id="),
            "Expected task_id in response, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_async_mode_check_shows_result() {
        let (tool, _db) = setup_bash_tool().await;

        // Start a longer async task so it's still running when we check
        let result = tool
            .call(
                r#"{"action":"run","tool":"bash-agent","prompt":"echo async-check-test; sleep 5","async_mode":true}"#,
            )
            .await
            .unwrap();

        // Extract task_id from "task_id=XXXX)"
        let task_id = result
            .split("task_id=")
            .nth(1)
            .unwrap()
            .split(')')
            .next()
            .unwrap();

        // Give it a moment to produce output
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Check on the task — should still be running
        let check_args = format!(r#"{{"action":"check","task_id":"{}"}}"#, task_id);
        let check_result = tool.call(&check_args).await.unwrap();

        assert!(
            check_result.contains("async-check-test")
                || check_result.contains("still running")
                || check_result.contains("finished"),
            "Expected task output or status, got: {}",
            check_result
        );

        // Cancel to clean up
        let cancel_args = format!(r#"{{"action":"cancel","task_id":"{}"}}"#, task_id);
        tool.call(&cancel_args).await.unwrap();
    }

    // -----------------------------------------------------------------------
    // Cancel test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_cancel_nonexistent_task() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool
            .call(r#"{"action":"cancel","task_id":"nonexistent"}"#)
            .await
            .unwrap();
        assert!(
            result.contains("No running CLI agent"),
            "Expected not found message, got: {}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // List test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_empty() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool.call(r#"{"action":"list"}"#).await.unwrap();
        assert!(
            result.contains("No CLI agents currently running"),
            "Expected empty list message, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_run_accepts_mission_and_task_aliases() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool
            .call(
                r#"{"action":"run","tool":"echo","mission":"Email helper","task":"Open Gmail and summarize the inbox"}"#,
            )
            .await
            .unwrap();
        assert!(
            result.contains("Mission: Email helper"),
            "Expected synthesized mission in output, got: {}",
            result
        );
        assert!(
            result.contains("Task: Open Gmail and summarize the inbox"),
            "Expected synthesized task in output, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_run_recovers_prompt_from_task_and_goal_context() {
        let (tool, _db) = setup_echo_tool().await;
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let working_dir = tmp_dir.path().to_string_lossy().to_string();
        let goal = Goal::new_finite(
            "Review the latest aidaemon service logs and make the smallest safe fix",
            "sess-ctx",
        );
        tool.state.create_goal(&goal).await.unwrap();

        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: format!(
                "Inspect the recent log failures and patch the root cause in {}",
                working_dir
            ),
            status: "claimed".to_string(),
            priority: "high".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: Some("task-lead".to_string()),
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        tool.state.create_task(&task).await.unwrap();

        let result = tool
            .call(&format!(
                r#"{{"action":"run","tool":"echo","working_dir":"{}","_goal_id":"{}","_task_id":"{}"}}"#,
                working_dir, goal.id, task.id
            ))
            .await
            .unwrap();

        assert!(
            result.contains(
                "Mission: Review the latest aidaemon service logs and make the smallest safe fix"
            ),
            "Expected synthesized mission from goal context, got: {}",
            result
        );
        assert!(
            result.contains(&format!(
                "Task: Inspect the recent log failures and patch the root cause in {}",
                working_dir
            )),
            "Expected synthesized task from task context, got: {}",
            result
        );

        let updated_task = tool.state.get_task(&task.id).await.unwrap().unwrap();
        assert_eq!(updated_task.status, "completed");
        let context: serde_json::Value =
            serde_json::from_str(updated_task.context.as_deref().unwrap()).unwrap();
        assert_eq!(
            context["executor_result"]["task_outcome"].as_str(),
            Some("task_done")
        );
        let stored_scope = context["executor_handoff"]["target_scope"]["allowed_targets"][0]
            ["value"]
            .as_str()
            .unwrap();
        assert!(
            stored_scope.ends_with(tmp_dir.path().to_string_lossy().as_ref()),
            "expected stored scope '{}' to end with '{}'",
            stored_scope,
            tmp_dir.path().to_string_lossy()
        );
    }

    #[tokio::test]
    async fn test_run_accepts_command_and_description_aliases() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool
            .call(
                r#"{"action":"run","tool":"echo","description":"Check test output","command":"pwd"}"#,
            )
            .await
            .unwrap();
        assert!(
            result.contains("Check test output"),
            "Expected synthesized description in output, got: {}",
            result
        );
        assert!(
            result.contains(
                "Run this exact shell command in the working directory and report the result:\npwd"
            ),
            "Expected synthesized command guidance in output, got: {}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Schema test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_schema_includes_registered_tools() {
        let (tool, _db) = setup_echo_tool().await;
        let schema = tool.schema();

        // Must have name and description (critical gotcha from CLAUDE.md)
        assert_eq!(schema["name"], "cli_agent");
        assert!(schema["description"].as_str().unwrap().contains("echo"));
        assert!(schema["parameters"]["properties"]["tool"]["enum"]
            .as_array()
            .unwrap()
            .contains(&json!("echo")));
    }

    #[tokio::test]
    async fn test_schema_updates_after_dynamic_add() {
        let (tool, _db) = setup_echo_tool().await;

        // Add a new agent directly to the map
        {
            let mut tools = tool.tools.write().unwrap();
            tools.insert(
                "new-tool".to_string(),
                CliToolEntry {
                    command: "echo".to_string(),
                    args: vec![],
                    description: "Newly added".to_string(),
                    timeout: Duration::from_secs(10),
                    max_output_chars: 10000,
                    is_dynamic: true,
                },
            );
            let mut names = tool.tool_names.write().unwrap();
            names.push("new-tool".to_string());
            names.sort();
        }

        let schema = tool.schema();
        let enum_vals = schema["parameters"]["properties"]["tool"]["enum"]
            .as_array()
            .unwrap();
        assert!(
            enum_vals.contains(&json!("echo")) && enum_vals.contains(&json!("new-tool")),
            "Schema should include both tools: {:?}",
            enum_vals
        );
    }

    // -----------------------------------------------------------------------
    // Dynamic agent management tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_add_agent_with_echo() {
        let (tool, _db) = setup_echo_tool().await;

        // `echo` exists on all systems
        let result = tool
            .add_agent("my-echo", "echo", vec![], "Custom echo", None, None)
            .await
            .unwrap();
        assert!(
            result.contains("added successfully"),
            "Expected success, got: {}",
            result
        );

        // Should be in the tools map
        let agents = tool.list_agents();
        assert!(
            agents.iter().any(|(name, _, _, _)| name == "my-echo"),
            "Expected my-echo in agent list"
        );
    }

    #[tokio::test]
    async fn test_add_agent_nonexistent_command() {
        let (tool, _db) = setup_echo_tool().await;

        let result = tool
            .add_agent(
                "fake",
                "aidaemon-nonexistent-cmd-xyz",
                vec![],
                "",
                None,
                None,
            )
            .await
            .unwrap();
        assert!(
            result.contains("not found"),
            "Expected not found error, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_remove_agent() {
        let (tool, _db) = setup_echo_tool().await;

        // Add then remove
        tool.add_agent("removeme", "echo", vec![], "", None, None)
            .await
            .unwrap();
        let result = tool.remove_agent("removeme").await.unwrap();
        assert!(result.contains("removed"));

        let agents = tool.list_agents();
        assert!(
            !agents.iter().any(|(name, _, _, _)| name == "removeme"),
            "Agent should have been removed"
        );
    }

    #[tokio::test]
    async fn test_remove_nonexistent_agent() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool.remove_agent("nonexistent").await.unwrap();
        assert!(result.contains("not found"));
    }

    // -----------------------------------------------------------------------
    // Auth error detection tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_detect_auth_error_patterns() {
        let cases = vec![
            ("Error: authentication required", true),
            ("401 Unauthorized", true),
            ("Token expired, please re-authenticate", true),
            ("Login required to continue", true),
            ("Invalid API key provided", true),
            ("Access denied: forbidden", true),
            ("Invalid token for this resource", true),
            (
                "See https://goo.gle/gemini-cli-auth-docs#workspace-gca",
                true,
            ),
            ("Normal output: everything is fine", false),
            ("Compiling project...", false),
        ];

        for (output, should_detect) in cases {
            let result = CliAgentTool::detect_auth_error(output, "test-agent");
            if should_detect {
                assert!(
                    result.is_some(),
                    "Expected auth error detection for: {}",
                    output
                );
                assert!(result.unwrap().contains("authentication failed"));
            } else {
                assert!(
                    result.is_none(),
                    "False positive auth detection for: {}",
                    output
                );
            }
        }
    }

    #[tokio::test]
    async fn authentication_failure_automatically_retries_another_configured_agent() {
        let (tool, _db) = setup_echo_tool().await;
        tool.tools.write().unwrap().insert(
            "auth-agent".to_string(),
            CliToolEntry {
                command: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "printf 'authentication required\\n'; exit 1".to_string(),
                ],
                description: "Unauthenticated test agent".to_string(),
                timeout: Duration::from_secs(10),
                max_output_chars: 10_000,
                is_dynamic: false,
            },
        );
        tool.tool_names
            .write()
            .unwrap()
            .push("auth-agent".to_string());

        let result = tool
            .call(r#"{"action":"run","tool":"auth-agent","prompt":"finish autonomously"}"#)
            .await
            .unwrap();

        assert!(result.contains("finish autonomously"), "got: {result}");
        assert!(
            !tool.tools.read().unwrap().contains_key("auth-agent"),
            "the unavailable agent must remain quarantined"
        );
    }

    #[test]
    fn zero_exit_auth_setup_output_is_a_failed_completion() {
        let completion = CliAgentTool::classify_completion_result(
            "gemini",
            Some(0),
            "See https://goo.gle/gemini-cli-auth-docs#workspace-gca",
            "See https://goo.gle/gemini-cli-auth-docs#workspace-gca",
            10_000,
        );

        assert!(!completion.success);
        assert!(completion.response.is_none());
        assert!(completion
            .error
            .as_deref()
            .is_some_and(|error| error.contains("authentication failed")));
        assert!(!completion.persisted_output.contains("workspace-gca"));
    }

    // -----------------------------------------------------------------------
    // Loop detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_loop_detector_no_false_positive() {
        let mut detector = LoopDetector::new();
        for i in 0..20 {
            assert!(!detector.add_line(&format!("unique line {}", i)));
        }
    }

    #[test]
    fn test_loop_detector_catches_repetition() {
        let mut detector = LoopDetector::new();
        // Add the same line 50+ times
        for i in 0..LOOP_DETECTION_THRESHOLD + 1 {
            let detected = detector.add_line("stuck in a loop");
            if i >= LOOP_DETECTION_THRESHOLD - 1 {
                // Should trigger around threshold
                if detected {
                    return; // Test passes
                }
            }
        }
        panic!("Loop detector should have triggered");
    }

    #[test]
    fn test_loop_detector_ignores_empty_lines() {
        let mut detector = LoopDetector::new();
        for _ in 0..200 {
            assert!(!detector.add_line(""));
            assert!(!detector.add_line("   "));
        }
    }

    // -----------------------------------------------------------------------
    // JSON extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_meaningful_output_plain_text() {
        let output = "Hello, this is plain text output\nLine 2\n";
        let result = extract_meaningful_output(output, 10000);
        assert_eq!(result, output);
    }

    #[test]
    fn test_extract_meaningful_output_json_result() {
        let output = r#"{"result": "The task is complete. Created 3 files."}"#;
        let result = extract_meaningful_output(output, 10000);
        assert_eq!(result, "The task is complete. Created 3 files.");
    }

    #[test]
    fn test_extract_meaningful_output_json_output_field() {
        let output = r#"{"output": "Generated report successfully"}"#;
        let result = extract_meaningful_output(output, 10000);
        assert_eq!(result, "Generated report successfully");
    }

    #[test]
    fn test_extract_meaningful_output_codex_jsonl_uses_final_agent_message() {
        let noisy_prefix = "repository guidance\n".repeat(2_000);
        let output = format!(
            "{}\n{}\n{}\n{}",
            serde_json::json!({"type":"thread.started","thread_id":"synthetic-thread"}),
            serde_json::json!({
                "type":"item.completed",
                "item":{
                    "id":"item-1",
                    "type":"command_execution",
                    "aggregated_output":noisy_prefix,
                    "exit_code":0,
                    "status":"completed"
                }
            }),
            serde_json::json!({
                "type":"item.completed",
                "item":{
                    "id":"item-2",
                    "type":"agent_message",
                    "text":"Published and verified https://example.test/posts/synthetic/"
                }
            }),
            serde_json::json!({"type":"turn.completed","usage":{"output_tokens":42}}),
        );

        let result = extract_meaningful_output(&output, 200);

        assert_eq!(
            result,
            "Published and verified https://example.test/posts/synthetic/"
        );
        assert!(!result.contains("repository guidance"));
        assert!(!result.contains("truncated"));
    }

    #[test]
    fn test_extract_meaningful_output_codex_jsonl_prefers_last_agent_message() {
        let output = [
            serde_json::json!({
                "type":"item.completed",
                "item":{"type":"agent_message","text":"I will inspect the repository."}
            })
            .to_string(),
            serde_json::json!({
                "type":"item.completed",
                "item":{"type":"agent_message","text":"Publication completed and verified."}
            })
            .to_string(),
            serde_json::json!({"type":"turn.completed"}).to_string(),
        ]
        .join("\n");

        assert_eq!(
            extract_meaningful_output(&output, 10_000),
            "Publication completed and verified."
        );
    }

    #[test]
    fn bounded_output_retains_final_codex_result_after_overflow() {
        let mut output = String::new();
        append_bounded_line(
            &mut output,
            "",
            &serde_json::json!({"type":"thread.started"}).to_string(),
            512,
        );
        append_bounded_line(
            &mut output,
            "",
            &serde_json::json!({
                "type":"item.completed",
                "item":{
                    "type":"command_execution",
                    "aggregated_output":"inspection noise ".repeat(200)
                }
            })
            .to_string(),
            512,
        );
        append_bounded_line(
            &mut output,
            "",
            &serde_json::json!({
                "type":"item.completed",
                "item":{
                    "type":"agent_message",
                    "text":"Published exactly once."
                }
            })
            .to_string(),
            512,
        );

        assert!(output.len() <= 512);
        assert!(output.contains(BUFFER_TRUNCATION_MARKER.trim()));
        assert_eq!(
            extract_meaningful_output(&output, 10_000),
            "Published exactly once."
        );
    }

    #[test]
    fn test_extract_meaningful_output_truncation() {
        let output = "a".repeat(20000);
        let result = extract_meaningful_output(&output, 100);
        assert!(result.len() <= 200); // 100 chars + truncation note
        assert!(result.contains("truncated"));
    }

    #[test]
    fn test_extract_progress_from_json_tool_use() {
        let json = r#"{"name":"Read","input":{"file_path":"/src/main.rs"}}"#;
        let progress = extract_progress_from_json(json);
        assert!(progress.is_some());
        assert!(progress.unwrap().contains("main.rs"));
    }

    #[test]
    fn test_extract_progress_from_json_bash_command() {
        let json = r#"{"name":"Bash","input":{"command":"npm install"}}"#;
        let progress = extract_progress_from_json(json);
        assert!(progress.is_some());
        assert!(progress.unwrap().contains("npm install"));
    }

    #[test]
    fn test_extract_progress_from_json_non_json() {
        let text = "This is just regular text";
        assert!(extract_progress_from_json(text).is_none());
    }

    #[test]
    fn test_looks_like_json() {
        assert!(looks_like_json(r#"{"key": "value"}"#));
        assert!(looks_like_json(r#"[1, 2, 3]"#));
        assert!(looks_like_json(r#"  {"indented": true}  "#));
        assert!(!looks_like_json("plain text"));
        assert!(!looks_like_json(""));
    }

    // -----------------------------------------------------------------------
    // Discovery tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_discover_finds_echo() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let provider = Arc::new(MockProvider::new());
        let (approval_tx_raw, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = ApprovalBroker::new(approval_tx_raw);

        // Config with echo as a custom tool
        let mut tools = HashMap::new();
        tools.insert(
            "echo".to_string(),
            crate::config::CliToolConfig {
                command: "echo".to_string(),
                args: vec![],
                description: "Echo for test".to_string(),
                timeout_secs: None,
                max_output_chars: None,
            },
        );
        let config = CliAgentsConfig {
            enabled: true,
            timeout_secs: 30,
            max_output_chars: 10000,
            tools,
        };

        let tool = CliAgentTool::discover(
            config,
            state as Arc<dyn StateStore>,
            SharedLlmRuntime::new(
                provider as Arc<dyn crate::traits::ModelProvider>,
                None,
                crate::config::ProviderKind::OpenaiCompatible,
                "mock".to_string(),
            ),
            approval_tx,
        )
        .await;
        assert!(tool.has_tools());

        let agents = tool.list_agents();
        assert!(agents.iter().any(|(name, _, _, _)| name == "echo"));
    }

    #[tokio::test]
    async fn test_discover_skips_nonexistent() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let provider = Arc::new(MockProvider::new());
        let (approval_tx_raw, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = ApprovalBroker::new(approval_tx_raw);

        let mut tools = HashMap::new();
        tools.insert(
            "fake-tool".to_string(),
            crate::config::CliToolConfig {
                command: "aidaemon-nonexistent-12345".to_string(),
                args: vec![],
                description: "".to_string(),
                timeout_secs: None,
                max_output_chars: None,
            },
        );
        let config = CliAgentsConfig {
            enabled: true,
            timeout_secs: 30,
            max_output_chars: 10000,
            tools,
        };

        let tool = CliAgentTool::discover(
            config,
            state as Arc<dyn StateStore>,
            SharedLlmRuntime::new(
                provider as Arc<dyn crate::traits::ModelProvider>,
                None,
                crate::config::ProviderKind::OpenaiCompatible,
                "mock".to_string(),
            ),
            approval_tx,
        )
        .await;
        assert!(!tool.has_tools());
    }

    // -----------------------------------------------------------------------
    // Invocation logging integration test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_logs_invocation() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let state_clone = state.clone();
        let provider = Arc::new(MockProvider::new());
        let (approval_tx_raw, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = ApprovalBroker::new(approval_tx_raw);

        let mut tools_map = HashMap::new();
        tools_map.insert(
            "echo".to_string(),
            CliToolEntry {
                command: "echo".to_string(),
                args: vec![],
                description: "".to_string(),
                timeout: Duration::from_secs(10),
                max_output_chars: 10000,
                is_dynamic: false,
            },
        );

        let tool = CliAgentTool {
            backend: active_execution_backend(),
            tools: Arc::new(std::sync::RwLock::new(tools_map)),
            tool_names: Arc::new(std::sync::RwLock::new(vec!["echo".to_string()])),
            running: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
            state: state as Arc<dyn StateStore>,
            llm_runtime: SharedLlmRuntime::new(
                provider as Arc<dyn crate::traits::ModelProvider>,
                None,
                crate::config::ProviderKind::OpenaiCompatible,
                "mock".to_string(),
            ),
            default_timeout: Duration::from_secs(10),
            default_max_output: 10000,
            max_concurrent: 3,
            concurrency_limiter: Arc::new(Semaphore::new(3)),
            approval_tx,
            hub: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
        };

        // Run a command
        tool.call(r#"{"action":"run","tool":"echo","prompt":"log test","_session_id":"sess1"}"#)
            .await
            .unwrap();

        // Check invocations were logged
        let invocations = state_clone.get_cli_agent_invocations(10).await.unwrap();
        assert!(
            !invocations.is_empty(),
            "Expected at least one invocation logged"
        );
        assert_eq!(invocations[0].agent_name, "echo");
        assert!(invocations[0].prompt_summary.contains("log test"));
        assert_eq!(invocations[0].success, Some(true));
        assert!(invocations[0].duration_secs.is_some());
    }

    #[tokio::test]
    async fn test_timeout_run_still_logs_completion() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let state_clone = state.clone();
        let provider = Arc::new(MockProvider::new());
        let (approval_tx_raw, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = ApprovalBroker::new(approval_tx_raw);

        let mut tools_map = HashMap::new();
        tools_map.insert(
            "bash".to_string(),
            CliToolEntry {
                command: "bash".to_string(),
                args: vec!["-c".to_string()],
                description: "".to_string(),
                timeout: Duration::from_millis(50),
                max_output_chars: 10000,
                is_dynamic: false,
            },
        );

        let tool = CliAgentTool {
            backend: active_execution_backend(),
            tools: Arc::new(std::sync::RwLock::new(tools_map)),
            tool_names: Arc::new(std::sync::RwLock::new(vec!["bash".to_string()])),
            running: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
            state: state as Arc<dyn StateStore>,
            llm_runtime: SharedLlmRuntime::new(
                provider as Arc<dyn crate::traits::ModelProvider>,
                None,
                crate::config::ProviderKind::OpenaiCompatible,
                "mock".to_string(),
            ),
            default_timeout: Duration::from_secs(10),
            default_max_output: 10000,
            max_concurrent: 3,
            concurrency_limiter: Arc::new(Semaphore::new(3)),
            approval_tx,
            hub: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
        };

        // Run a command that will exceed the short timeout and be moved to background,
        // but should still eventually be logged as completed in the DB.
        let resp = tool
            .call(
                r#"{"action":"run","tool":"bash","prompt":"sleep 0.2; echo done","_session_id":"sess1"}"#,
            )
            .await
            .unwrap();
        assert!(
            resp.contains("Moved to background") || resp.contains("still running"),
            "expected background/timeout response, got: {}",
            resp
        );

        // Allow the child to finish and the completion logger to flush to SQLite.
        tokio::time::sleep(Duration::from_millis(500)).await;

        let invocations = state_clone.get_cli_agent_invocations(10).await.unwrap();
        assert!(!invocations.is_empty(), "Expected invocation logged");
        assert_eq!(invocations[0].agent_name, "bash");
        assert_eq!(
            invocations[0].success,
            Some(true),
            "Expected background invocation to be marked successful"
        );
        assert!(
            invocations[0].completed_at.is_some(),
            "Expected completed_at to be set"
        );
        assert!(
            invocations[0].duration_secs.unwrap_or(0.0) > 0.0,
            "Expected a positive duration"
        );
    }

    #[tokio::test]
    async fn test_background_delegated_run_persists_structured_task_result() {
        let (tool, _db) = setup_bash_tool().await;
        {
            let mut tools = tool.tools.write().unwrap();
            tools.get_mut("bash-agent").unwrap().timeout = Duration::from_millis(50);
        }

        let goal = Goal::new_finite("Patch the current repo safely", "sess-bg");
        tool.state.create_goal(&goal).await.unwrap();

        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Run a delegated CLI task in the background".to_string(),
            status: "claimed".to_string(),
            priority: "high".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: Some("task-lead".to_string()),
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        tool.state.create_task(&task).await.unwrap();

        let resp = tool
            .call(&format!(
                r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 0.2; echo delegated-background-ok","_session_id":"sess-bg","_goal_id":"{}","_task_id":"{}"}}"#,
                goal.id, task.id
            ))
            .await
            .unwrap();
        assert!(
            resp.contains("Moved to background") || resp.contains("still running"),
            "expected timeout/background handoff, got: {}",
            resp
        );

        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        let updated_task = loop {
            let current = tool.state.get_task(&task.id).await.unwrap().unwrap();
            if current.status != "running" && current.status != "claimed" {
                break current;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "delegated task did not persist a terminal outcome before the deadline"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        };
        assert_eq!(updated_task.status, "completed");
        let context: serde_json::Value =
            serde_json::from_str(updated_task.context.as_deref().unwrap()).unwrap();
        assert_eq!(
            context["executor_result"]["task_outcome"].as_str(),
            Some("task_done")
        );
        assert!(
            updated_task
                .result
                .as_deref()
                .unwrap_or("")
                .contains("delegated-background-ok"),
            "expected delegated background result to be persisted, got: {:?}",
            updated_task.result
        );
    }

    #[tokio::test]
    async fn test_background_cli_check_returns_result_after_reap() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                db_file.path().to_str().unwrap(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let provider = Arc::new(MockProvider::new());
        let (approval_tx_raw, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);
        let approval_tx = ApprovalBroker::new(approval_tx_raw);

        let mut tools_map = HashMap::new();
        tools_map.insert(
            "bash".to_string(),
            CliToolEntry {
                command: "bash".to_string(),
                args: vec!["-c".to_string()],
                description: "".to_string(),
                timeout: Duration::from_millis(50),
                max_output_chars: 10000,
                is_dynamic: false,
            },
        );

        let tool = CliAgentTool {
            backend: active_execution_backend(),
            tools: Arc::new(std::sync::RwLock::new(tools_map)),
            tool_names: Arc::new(std::sync::RwLock::new(vec!["bash".to_string()])),
            running: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
            state: state as Arc<dyn StateStore>,
            llm_runtime: SharedLlmRuntime::new(
                provider as Arc<dyn crate::traits::ModelProvider>,
                None,
                crate::config::ProviderKind::OpenaiCompatible,
                "mock".to_string(),
            ),
            default_timeout: Duration::from_secs(10),
            default_max_output: 10000,
            max_concurrent: 3,
            concurrency_limiter: Arc::new(Semaphore::new(3)),
            approval_tx,
            hub: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
        };

        let resp = tool
            .call(
                r#"{"action":"run","tool":"bash","prompt":"sleep 0.2; echo cli-reap-ok","_session_id":"sess1"}"#,
            )
            .await
            .unwrap();
        let task_id = extract_task_id_from_background_message(&resp);

        // Let it finish so next call reaps before check.
        tokio::time::sleep(Duration::from_millis(500)).await;

        let check = tool
            .call(&format!(
                r#"{{"action":"check","task_id":"{}","_session_id":"sess1"}}"#,
                task_id
            ))
            .await
            .unwrap();
        assert!(check.contains("cli-reap-ok"));
        assert!(check.contains("finished"));
    }

    #[tokio::test]
    async fn check_with_wrong_id_recovers_only_session_task() {
        let (tool, _db) = setup_bash_tool().await;
        let started = tool
            .call(
                r#"{"action":"run","tool":"bash-agent","prompt":"echo recovered-result","async_mode":true,"_session_id":"session-recover"}"#,
            )
            .await
            .unwrap();
        let real_task_id = extract_task_id_from_background_message(&started);
        tokio::time::sleep(Duration::from_millis(300)).await;

        let recovered = tool
            .call(&format!(
                r#"{{"action":"check","task_id":"provider-thread-uuid","_session_id":"session-recover"}}"#
            ))
            .await
            .unwrap();

        assert!(recovered.contains("Recovered the only recent CLI task"));
        assert!(recovered.contains(&format!("task_id={real_task_id}")));
        assert!(recovered.contains("recovered-result"));
    }

    #[tokio::test]
    async fn missing_check_is_a_typed_negative_result() {
        let (tool, _db) = setup_echo_tool().await;
        let outcome = tool
            .call_with_status_outcome(
                r#"{"action":"check","task_id":"missing","_session_id":"empty-session"}"#,
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::CompletedWithNegativeResult)
        );
        assert!(outcome
            .output
            .contains("do not treat this lookup miss as completion"));
    }

    #[tokio::test]
    async fn async_run_reports_background_metadata_and_notifications() {
        let (tool, _db) = setup_bash_tool().await;
        let outcome = tool
            .call_with_status_outcome(
                r#"{"action":"run","tool":"bash-agent","prompt":"sleep 2","async_mode":true,"_session_id":"session-background"}"#,
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Backgrounded)
        );
        assert!(outcome.metadata.background_started);
        assert!(outcome.metadata.completion_notifications_enabled);

        tool.handle_cancel_all("session-background").await.unwrap();
    }

    #[tokio::test]
    async fn async_run_queues_chat_visible_progress_while_still_running() {
        let (tool, _db) = setup_bash_tool().await;
        let state = tool.state.clone();
        tool.call(
            r#"{"action":"run","tool":"bash-agent","prompt":"sleep 5","async_mode":true,"_session_id":"session-progress"}"#,
        )
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(750)).await;
        let notifications = state.get_pending_notifications(20).await.unwrap();
        assert!(
            notifications.iter().any(|entry| {
                entry.session_id == "session-progress"
                    && entry.notification_type == "progress"
                    && entry.message.contains("still running")
            }),
            "pending notifications: {notifications:?}"
        );

        tool.handle_cancel_all("session-progress").await.unwrap();
    }

    // -----------------------------------------------------------------------
    // Git diff capture test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_capture_git_diff_no_repo() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = CliAgentTool::capture_git_diff(tmp.path().to_str().unwrap()).await;
        assert!(result.is_none(), "Non-git directory should return None");
    }

    #[tokio::test]
    async fn trusted_cli_delegation_requires_structural_worktree_scope() {
        let (tool, _db) = setup_echo_tool().await;
        for prompt in [
            "Build, deploy, and verify the exact public URL",
            "Build locally but do not deploy",
            "Inspect the code and return a report",
        ] {
            let args = serde_json::json!({
                "action": "run",
                "tool": "echo",
                "prompt": prompt,
                "_trusted_session": true
            });
            let result = tool.call(&args.to_string()).await.unwrap();
            assert!(
                result.contains("requires an explicit working_dir"),
                "prompt wording must not narrow an opaque unattended delegation: {result}"
            );
        }
    }

    #[tokio::test]
    async fn test_capture_git_diff_with_changes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().to_str().unwrap();

        // Initialize a git repo with a commit
        tokio::process::Command::new("git")
            .args(["init"])
            .current_dir(dir)
            .output()
            .await
            .unwrap();
        tokio::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir)
            .output()
            .await
            .unwrap();
        tokio::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir)
            .output()
            .await
            .unwrap();

        // Create and commit a file
        std::fs::write(tmp.path().join("file.txt"), "initial").unwrap();
        tokio::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir)
            .output()
            .await
            .unwrap();
        tokio::process::Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(dir)
            .output()
            .await
            .unwrap();

        // Modify the file (uncommitted change)
        std::fs::write(tmp.path().join("file.txt"), "modified content").unwrap();

        assert_eq!(CliAgentTool::dirty_worktree_entry_count(dir).await, Some(1));

        let result = CliAgentTool::capture_git_diff(dir).await;
        assert!(result.is_some(), "Should capture uncommitted changes");
        let diff = result.unwrap();
        assert!(
            diff.contains("modified content") || diff.contains("file.txt"),
            "Diff should mention the changed file, got: {}",
            diff
        );
    }

    // -----------------------------------------------------------------------
    // Unknown action test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_unknown_action() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool.call(r#"{"action":"invalid_action"}"#).await.unwrap();
        assert!(result.contains("Unknown action"));
    }

    // -----------------------------------------------------------------------
    // has_tools test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_has_tools() {
        let (tool, _db) = setup_echo_tool().await;
        assert!(tool.has_tools());
        assert!(Tool::is_available(&tool));

        // Clear all tools
        tool.tools.write().unwrap().clear();
        assert!(!tool.has_tools());
        assert!(!Tool::is_available(&tool));
    }

    #[tokio::test]
    async fn test_run_without_tool_and_no_agents_errors() {
        let (tool, _db) = setup_echo_tool().await;
        tool.tools.write().unwrap().clear();
        tool.tool_names.write().unwrap().clear();

        let result = tool.call(r#"{"action":"run","prompt":"test"}"#).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("No CLI agents available"),
            "Expected no-agents error, got: {}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // Enriched prompt test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_build_enriched_prompt_basic() {
        let (tool, _db) = setup_echo_tool().await;

        let prompt = tool
            .build_enriched_prompt(
                "test-session",
                "You are a security auditor",
                "Audit this codebase",
                None,
            )
            .await;

        assert!(prompt.contains("You are a security auditor"));
        assert!(prompt.contains("Audit this codebase"));
        assert!(prompt.contains("## Task"));
        assert!(prompt.contains("## Instructions"));
    }

    #[tokio::test]
    async fn test_build_enriched_prompt_no_instruction() {
        let (tool, _db) = setup_echo_tool().await;

        let prompt = tool
            .build_enriched_prompt("test-session", "", "Just do the task", None)
            .await;

        // Empty instruction should not appear
        assert!(prompt.contains("Just do the task"));
        assert!(prompt.contains("## Task"));
    }

    #[tokio::test]
    async fn test_enriched_prompt_uses_scoped_agents_instead_of_readme() {
        let (tool, _db) = setup_echo_tool().await;
        let project = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(project.path().join(".git")).unwrap();
        std::fs::write(
            project.path().join("AGENTS.md"),
            "RUN_THE_SCOPED_AGENT_TEST",
        )
        .unwrap();
        std::fs::write(
            project.path().join("README.md"),
            "README_MUST_NOT_BECOME_INSTRUCTIONS",
        )
        .unwrap();

        let prompt = tool
            .build_enriched_prompt(
                "test-session",
                "You are a code reviewer",
                "Review this codebase",
                project.path().to_str(),
            )
            .await;

        assert!(prompt.contains("## Project Instructions"));
        assert!(prompt.contains("RUN_THE_SCOPED_AGENT_TEST"));
        assert!(!prompt.contains("README_MUST_NOT_BECOME_INSTRUCTIONS"));
    }

    // -----------------------------------------------------------------------
    // Scenario replication: exact user prompt that caused the hang
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_user_prompt_website_about_cars() {
        // This replicates the exact prompt that caused a 5-minute hang.
        // The fix: stdin is now Stdio::null() so processes get EOF immediately.
        let (tool, _db) = setup_bash_tool().await;

        let user_prompt = "I need to create a new website about cars. We should push it to cars.example.com. make it modern.";

        let start = Instant::now();
        let args = serde_json::json!({
            "action": "run",
            "tool": "bash-agent",
            "prompt": format!("echo 'Received prompt: {}'; echo 'Task complete'", user_prompt),
            "_session_id": "telegram_12345"
        });
        let result = tool.call(&args.to_string()).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(5),
            "Took {:?} — should complete quickly, not hang",
            elapsed
        );
        assert!(
            result.contains("Task complete"),
            "Expected output, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_user_prompt_with_system_instruction() {
        // Tests the orchestrator flow: LLM sends system_instruction to shape
        // the CLI agent into a specialist, which triggers build_enriched_prompt.
        let (tool, _db) = setup_bash_tool().await;

        let start = Instant::now();
        let args = serde_json::json!({
            "action": "run",
            "tool": "bash-agent",
            "prompt": "echo 'Building website...'; echo 'Created index.html'; echo 'Done'",
            "system_instruction": "You are a senior web developer. Create a modern, responsive website.",
            "_session_id": "telegram_12345"
        });
        let result = tool.call(&args.to_string()).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(5),
            "Enriched prompt flow took {:?}",
            elapsed
        );
        assert!(result.contains("Done"), "Expected output, got: {}", result);
    }

    // -----------------------------------------------------------------------
    // Claude Code stream-json output parsing
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_claude_stream_json_output_parsing() {
        // Simulates Claude Code's --output-format stream-json output
        let (tool, _db) = setup_bash_tool().await;

        let stream_json = r#"
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"I will create the website."}]}}'
echo '{"type":"tool_use","name":"Bash","input":{"command":"mkdir -p website"}}'
echo '{"type":"tool_result","content":"Directory created"}'
echo '{"type":"result","result":"Website created successfully with index.html, style.css, and script.js"}'
"#;

        let args = serde_json::json!({
            "action": "run",
            "tool": "bash-agent",
            "prompt": stream_json.trim()
        });
        let result = tool.call(&args.to_string()).await.unwrap();

        // extract_meaningful_output should pull out the "result" field
        assert!(
            result.contains("Website created successfully"),
            "Should extract result from JSON output, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_progress_extraction_from_claude_stream() {
        // Test the progress parser with Claude Code's assistant/tool_use events
        let assistant_json = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"npm install react"}}]}}"#;
        let progress = extract_progress_from_json(assistant_json);
        assert!(
            progress.is_some(),
            "Should extract progress from assistant tool_use event"
        );
        assert!(
            progress.unwrap().contains("npm install react"),
            "Should include the command"
        );

        let thinking_json = r#"{"type":"thinking"}"#;
        let progress = extract_progress_from_json(thinking_json);
        assert_eq!(progress, Some("💭 Thinking...".to_string()));
    }

    // -----------------------------------------------------------------------
    // Multi-line stderr output test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_stderr_captured_in_error_output() {
        let (tool, _db) = setup_bash_tool().await;

        let args = serde_json::json!({
            "action": "run",
            "tool": "bash-agent",
            "prompt": "echo 'some stdout'; echo 'error detail 1' >&2; echo 'error detail 2' >&2; exit 1"
        });
        let result = tool.call(&args.to_string()).await.unwrap();

        assert!(result.contains("ERROR"));
        assert!(
            result.contains("error detail 1"),
            "Should capture stderr, got: {}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Working dir with real CLI agent simulation
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_with_working_dir() {
        let (tool, _db) = setup_bash_tool().await;
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let dir_path = tmp_dir.path().to_str().unwrap();

        let args = serde_json::json!({
            "action": "run",
            "tool": "bash-agent",
            "prompt": "pwd",
            "working_dir": dir_path
        });
        let result = tool.call(&args.to_string()).await.unwrap();

        assert!(
            result.contains(dir_path),
            "CLI agent should run in specified working dir, got: {}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Prompt similarity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prompt_similarity_scores() {
        // Identical prompts → 1.0
        assert!(
            (prompt_similarity("refactor the auth module", "refactor the auth module") - 1.0).abs()
                < f64::EPSILON
        );

        // Very similar prompts → high similarity
        let sim = prompt_similarity(
            "refactor the auth module to use JWT",
            "refactor the auth module to use OAuth",
        );
        assert!(
            sim > 0.5,
            "Similar prompts should score > 0.5, got: {:.3}",
            sim
        );

        // Completely different prompts → low similarity
        let sim = prompt_similarity("refactor the auth module", "deploy to production server");
        assert!(
            sim < 0.3,
            "Different prompts should score < 0.3, got: {:.3}",
            sim
        );

        // Empty prompts
        assert!((prompt_similarity("", "") - 1.0).abs() < f64::EPSILON);
        assert!((prompt_similarity("hello", "") - 0.0).abs() < f64::EPSILON);

        // Single-word prompts (unigram fallback)
        assert!((prompt_similarity("deploy", "deploy") - 1.0).abs() < f64::EPSILON);
        assert!((prompt_similarity("deploy", "refactor") - 0.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Task deduplication test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_duplicate_task_detection() {
        let (tool, _db) = setup_bash_tool().await;
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let dir = tmp_dir.path().to_string_lossy().to_string();

        // Start a background task
        let first_args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"refactor the auth module to use JWT tokens","working_dir":"{}","async_mode":true}}"#,
            dir
        );
        let first = tool.call(&first_args).await.unwrap();
        assert!(
            first.contains("started in background"),
            "Expected first task to start, got: {}",
            first
        );

        // Try a very similar prompt to the same dir → should be blocked
        let second_args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"refactor the auth module to use OAuth tokens","working_dir":"{}"}}"#,
            dir
        );
        let second = tool.call(&second_args).await.unwrap();
        assert!(
            second.contains("BLOCKED") && second.contains("similar task"),
            "Expected duplicate task BLOCKED message, got: {}",
            second
        );

        tool.handle_cancel_all("").await.unwrap();
    }
}
