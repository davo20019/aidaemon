use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};
use uuid::Uuid;

use super::process_control::{configure_command_for_process_group, send_sigkill, send_sigterm};
use super::{
    command_risk::{PermissionMode, RiskLevel},
    daemon_guard::detect_daemonization_primitives,
};
use crate::config::CliAgentsConfig;
use crate::tools::terminal::ApprovalRequest;
use crate::traits::{DynamicCliAgent, ModelProvider, StateStore, Tool, ToolCapabilities};
use crate::types::ApprovalResponse;
use crate::types::StatusUpdate;
use crate::utils::{truncate_str, truncate_with_note};

/// Max bytes for output buffer (1 MB) to prevent unbounded memory growth.
const BUFFER_CAP: usize = 1_048_576;

/// Interval for emitting progress updates (avoid spamming the channel).
const PROGRESS_INTERVAL: Duration = Duration::from_secs(2);

/// Loop detection: window size for tracking recent lines
const LOOP_DETECTION_WINDOW: usize = 100;

/// Loop detection: threshold - if same line appears this many times in window, it's a loop
const LOOP_DETECTION_THRESHOLD: usize = 50;

/// Max concurrent CLI agent processes
const DEFAULT_MAX_CONCURRENT: usize = 3;

/// Max enriched prompt size (8 KB)
const MAX_PROMPT_SIZE: usize = 8192;

/// Max git diff size to append to results (4 KB)
const MAX_DIFF_SIZE: usize = 4096;

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

/// Check if a process is still alive.
#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    use std::process::Command;
    // kill -0 checks if process exists without actually sending a signal
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_process_alive(_pid: u32) -> bool {
    // On non-Unix platforms, assume process is alive
    true
}

/// Kill a process by PID (SIGTERM then SIGKILL).
#[cfg(unix)]
async fn kill_process(pid: u32) {
    if pid == 0 {
        return;
    }

    // Send SIGTERM to process group (fallback to pid).
    let _ = send_sigterm(pid);

    // Wait a bit
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Send SIGKILL if still alive.
    if is_process_alive(pid) {
        let _ = send_sigkill(pid);
    }
}

#[cfg(not(unix))]
async fn kill_process(_pid: u32) {
    // On non-Unix platforms, we can't easily kill by PID
    // The process will be orphaned but should eventually terminate
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

/// A running CLI agent being tracked.
struct RunningCliAgent {
    tool_name: String,
    prompt_summary: String,
    started_at: Instant,
    /// Combined output for display (includes [stderr] prefixes)
    display_buf: Arc<Mutex<String>>,
    /// Pure stdout for JSON extraction
    stdout_buf: Arc<Mutex<String>>,
    /// Process ID for status display and killing
    child_id: u32,
    /// Session ID for filtering cancel_all by session
    session_id: String,
    /// Working directory for git diff capture
    working_dir: Option<String>,
}

pub struct CliAgentTool {
    // MUST be std::sync::RwLock because schema() is sync
    tools: Arc<std::sync::RwLock<HashMap<String, CliToolEntry>>>,
    tool_names: Arc<std::sync::RwLock<Vec<String>>>,
    running: Arc<Mutex<HashMap<String, RunningCliAgent>>>, // task_id -> RunningCliAgent
    working_dir_locks: Arc<Mutex<HashSet<String>>>,
    state: Arc<dyn StateStore>,
    #[allow(dead_code)] // Reserved for future interactive feedback
    provider: Arc<dyn ModelProvider>,
    default_timeout: Duration,
    default_max_output: usize,
    max_concurrent: usize,
    approval_tx: mpsc::Sender<ApprovalRequest>,
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
            vec!["-p", "--sandbox=false", "--yolo"],
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

/// Check if a command exists on the system.
async fn command_exists(command: &str) -> bool {
    tokio::process::Command::new("which")
        .arg(command)
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

impl CliAgentTool {
    fn is_owner_role(user_role: Option<&str>) -> bool {
        user_role.is_some_and(|role| role.eq_ignore_ascii_case("owner"))
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
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;

        match tokio::time::timeout(std::time::Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Ok(ApprovalResponse::Deny),
            Err(_) => Ok(ApprovalResponse::Deny),
        }
    }

    pub async fn discover(
        config: CliAgentsConfig,
        state: Arc<dyn StateStore>,
        provider: Arc<dyn ModelProvider>,
        approval_tx: mpsc::Sender<ApprovalRequest>,
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

        // Run `which` checks in parallel for all candidates
        let which_futures: Vec<_> = candidates
            .iter()
            .map(|(_, command, _, _, _, _)| command_exists(command))
            .collect();
        let which_results = futures::future::join_all(which_futures).await;

        let mut tools = HashMap::new();

        for (i, (name, command, args, description, timeout_override, max_output_override)) in
            candidates.into_iter().enumerate()
        {
            if which_results[i] {
                info!(name = %name, command = %command, "CLI agent tool discovered");
                tools.insert(
                    name,
                    CliToolEntry {
                        command,
                        args,
                        description,
                        timeout: timeout_override
                            .map(Duration::from_secs)
                            .unwrap_or(default_timeout),
                        max_output_chars: max_output_override.unwrap_or(default_max_output),
                        is_dynamic: false,
                    },
                );
            } else {
                info!(name = %name, command = %command, "CLI agent tool not found, skipping");
            }
        }

        let mut tool_names: Vec<String> = tools.keys().cloned().collect();
        tool_names.sort();

        let tool = CliAgentTool {
            tools: Arc::new(std::sync::RwLock::new(tools)),
            tool_names: Arc::new(std::sync::RwLock::new(tool_names)),
            running: Arc::new(Mutex::new(HashMap::new())),
            working_dir_locks: Arc::new(Mutex::new(HashSet::new())),
            state,
            provider,
            default_timeout,
            default_max_output,
            max_concurrent: DEFAULT_MAX_CONCURRENT,
            approval_tx,
        };

        // Load dynamic agents from DB
        tool.load_dynamic_agents().await;

        tool
    }

    /// Load dynamically registered agents from the database.
    async fn load_dynamic_agents(&self) {
        match self.state.list_dynamic_cli_agents().await {
            Ok(agents) => {
                for agent in agents {
                    if !agent.enabled {
                        continue;
                    }
                    // Verify command still exists
                    if !command_exists(&agent.command).await {
                        info!(name = %agent.name, command = %agent.command, "Dynamic CLI agent command not found, skipping");
                        continue;
                    }
                    let args: Vec<String> =
                        serde_json::from_str(&agent.args_json).unwrap_or_default();
                    let entry = CliToolEntry {
                        command: agent.command.clone(),
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
        // Validate command exists
        if !command_exists(command).await {
            return Ok(format!(
                "Command '{}' not found on this system. Install it first.",
                command
            ));
        }

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
            command: command.to_string(),
            args,
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
                // Re-add to runtime map if command exists
                if command_exists(&agent.command).await {
                    let args: Vec<String> =
                        serde_json::from_str(&agent.args_json).unwrap_or_default();
                    let entry = CliToolEntry {
                        command: agent.command.clone(),
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
        let mut running = self.running.lock().await;
        let finished: Vec<String> = running
            .iter()
            .filter(|(_, agent)| !is_process_alive(agent.child_id))
            .map(|(id, _)| id.clone())
            .collect();

        // Also release working dir locks for finished tasks
        let mut dir_locks = self.working_dir_locks.lock().await;
        for task_id in finished {
            if let Some(agent) = running.remove(&task_id) {
                if let Some(ref dir) = agent.working_dir {
                    dir_locks.remove(dir);
                }
                info!(task_id, tool = %agent.tool_name, "Reaped finished CLI agent");
            }
        }
    }

    /// Build an enriched prompt with context from memory and conversation history.
    async fn build_enriched_prompt(
        &self,
        session_id: &str,
        system_instruction: &str,
        task_prompt: &str,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        // System instruction (never truncated)
        if !system_instruction.is_empty() {
            parts.push(system_instruction.to_string());
        }

        // Task prompt (never truncated)
        parts.push(format!("## Task\n{}", task_prompt));

        let budget = MAX_PROMPT_SIZE.saturating_sub(
            parts.iter().map(|p| p.len()).sum::<usize>() + 200, // headroom for section headers
        );

        // Conversation context (truncated first if over budget)
        let mut context_text = String::new();
        if let Ok(history) = self.state.get_history(session_id, 5).await {
            if !history.is_empty() {
                let mut lines = Vec::new();
                for msg in history.iter().rev().take(5) {
                    let role = &msg.role;
                    let content: String = msg
                        .content
                        .as_deref()
                        .unwrap_or("")
                        .chars()
                        .take(200)
                        .collect();
                    lines.push(format!("{}: {}", role, content));
                }
                context_text = lines.join("\n");
            }
        }

        // Relevant facts
        let mut facts_text = String::new();
        if let Ok(facts) = self.state.get_relevant_facts(task_prompt, 10).await {
            if !facts.is_empty() {
                let fact_lines: Vec<String> = facts
                    .iter()
                    .map(|f| format!("- {}: {}", f.key, f.value))
                    .collect();
                facts_text = fact_lines.join("\n");
            }
        }

        // Fit within budget: truncate context first, then facts
        let total = context_text.len() + facts_text.len();
        if total > budget {
            let context_budget = budget / 3;
            let facts_budget = budget - context_budget;
            if context_text.len() > context_budget {
                context_text = context_text.chars().take(context_budget).collect();
                context_text.push_str("...[truncated]");
            }
            if facts_text.len() > facts_budget {
                facts_text = facts_text.chars().take(facts_budget).collect();
                facts_text.push_str("...[truncated]");
            }
        }

        if !context_text.is_empty() {
            parts.push(format!("## Relevant Context\n{}", context_text));
        }
        if !facts_text.is_empty() {
            parts.push(format!("## Known Facts\n{}", facts_text));
        }

        parts.push(
            "## Instructions\n\
             - Focus exclusively on the task above\n\
             - Report what you did and what changed when done"
                .to_string(),
        );

        parts.join("\n\n")
    }

    /// Capture git diff after CLI agent completes (for any exit code).
    async fn capture_git_diff(working_dir: &str) -> Option<String> {
        // Check if it's a git repo
        let git_check = tokio::process::Command::new("git")
            .args(["rev-parse", "--git-dir"])
            .current_dir(working_dir)
            .output()
            .await;
        if !git_check.map(|o| o.status.success()).unwrap_or(false) {
            return None;
        }

        // Check for uncommitted changes first
        let diff_stat = tokio::process::Command::new("git")
            .args(["diff", "--stat"])
            .current_dir(working_dir)
            .output()
            .await
            .ok()?;
        let stat_output = String::from_utf8_lossy(&diff_stat.stdout);

        if !stat_output.trim().is_empty() {
            // There are uncommitted changes — capture them
            let diff = tokio::process::Command::new("git")
                .args(["diff"])
                .current_dir(working_dir)
                .output()
                .await
                .ok()?;
            let diff_text = String::from_utf8_lossy(&diff.stdout);
            if !diff_text.trim().is_empty() {
                return Some(truncate_with_note(&diff_text, MAX_DIFF_SIZE));
            }
        }

        // No uncommitted changes — check if the agent committed something
        let log = tokio::process::Command::new("git")
            .args(["log", "-1", "--stat", "--format=%s"])
            .current_dir(working_dir)
            .output()
            .await
            .ok()?;
        let log_output = String::from_utf8_lossy(&log.stdout);

        if !log_output.trim().is_empty() {
            let committed_diff = tokio::process::Command::new("git")
                .args(["diff", "HEAD~1..HEAD"])
                .current_dir(working_dir)
                .output()
                .await
                .ok()?;
            let committed_text = String::from_utf8_lossy(&committed_diff.stdout);
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

    /// Detect auth-related errors in output.
    fn detect_auth_error(output: &str, tool_name: &str) -> Option<String> {
        let auth_patterns = [
            "authentication",
            "unauthorized",
            "expired",
            "login required",
            "api key",
            "access denied",
            "forbidden",
            "invalid token",
        ];
        let lower = output.to_lowercase();
        for pattern in &auth_patterns {
            if lower.contains(pattern) {
                return Some(format!(
                    "CLI agent '{}' authentication failed. Check that your subscription/API key for {} is valid.",
                    tool_name, tool_name
                ));
            }
        }
        None
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
        system_instruction: Option<&str>,
        async_mode: bool,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        // Check concurrent limit
        {
            let running = self.running.lock().await;
            if running.len() >= self.max_concurrent {
                return Ok(format!(
                    "Maximum {} CLI agents already running. Use action='list' to see running tasks, or action='cancel' to stop one.",
                    self.max_concurrent
                ));
            }
        }

        // Get entry from the tools map (clone what we need, release lock)
        let (command, args, timeout, max_output) = {
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

        // Re-check that command still exists
        if !command_exists(&command).await {
            // Auto-disable dynamic agents that disappeared
            if let Ok(agents) = self.state.list_dynamic_cli_agents().await {
                if let Some(mut agent) = agents.into_iter().find(|a| a.name == tool_name) {
                    agent.enabled = false;
                    let _ = self.state.update_dynamic_cli_agent(&agent).await;
                }
            }
            return Ok(format!(
                "CLI agent '{}' command not found. It may have been uninstalled. \
                 Use manage_cli_agents to remove it.",
                tool_name
            ));
        }

        // Check working directory conflicts
        if let Some(dir) = working_dir {
            let mut dir_locks = self.working_dir_locks.lock().await;
            if dir_locks.contains(dir) {
                return Ok(format!(
                    "WARNING: Another CLI agent is already working in {}. \
                     Running in parallel may cause file conflicts. Consider waiting for \
                     the first task to complete.",
                    dir
                ));
            }
            dir_locks.insert(dir.to_string());
        }

        // Build the enriched prompt if system_instruction is provided
        let final_prompt = if let Some(instruction) = system_instruction {
            self.build_enriched_prompt(session_id, instruction, prompt)
                .await
        } else {
            prompt.to_string()
        };

        info!(
            tool = tool_name,
            session = session_id,
            prompt_len = final_prompt.len(),
            working_dir = working_dir.unwrap_or("(default)"),
            async_mode,
            "CLI agent invocation — runs with auto-approve flags"
        );

        // Log invocation start
        let prompt_summary: String = prompt.chars().take(100).collect();
        let invocation_id = self
            .state
            .log_cli_agent_start(session_id, tool_name, &prompt_summary, working_dir)
            .await
            .unwrap_or(0);

        // Build command
        let mut cmd = tokio::process::Command::new(&command);
        for arg in &args {
            cmd.arg(arg);
        }
        cmd.arg(&final_prompt);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        configure_command_for_process_group(&mut cmd);

        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        let short_summary: String = prompt.chars().take(50).collect();

        info!(
            task_id,
            tool = %tool_name,
            command = %command,
            working_dir = ?working_dir,
            "Starting CLI agent"
        );

        // Notify user this task can be cancelled
        if let Some(ref tx) = status_tx {
            let _ = tx.try_send(StatusUpdate::ToolCancellable {
                name: "cli_agent".to_string(),
                task_id: task_id.clone(),
            });
        }

        let mut child = cmd.spawn()?;
        let pid = child.id().unwrap_or(0);
        let started_at_instant = Instant::now();

        // stdin is null (prompt passed via args), so just drop any handle
        drop(child.stdin.take());
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture stderr"))?;

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

        // Spawn a task to read stdout/stderr, emit progress updates, and signal completion
        let pid_for_kill = pid;
        tokio::spawn(async move {
            let mut stdout_reader = BufReader::new(stdout).lines();
            let mut stderr_reader = BufReader::new(stderr).lines();
            let mut last_progress = Instant::now();
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
                    kill_process(pid_for_kill).await;
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

                                // Check for infinite loop pattern
                                if loop_detector.add_line(&text) && !loop_detected {
                                    loop_detected = true;
                                    loop_pattern_count = loop_detector.get_loop_pattern();
                                }

                                // Write to stdout buffer (for JSON extraction)
                                {
                                    let mut buf = stdout_buf_writer.lock().await;
                                    if buf.len() < BUFFER_CAP {
                                        buf.push_str(&text);
                                        buf.push('\n');
                                    }
                                }
                                // Write to display buffer
                                {
                                    let mut buf = display_buf_writer.lock().await;
                                    if buf.len() < BUFFER_CAP {
                                        buf.push_str(&text);
                                        buf.push('\n');
                                    }
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
                                if buf.len() < BUFFER_CAP {
                                    buf.push_str("[stderr] ");
                                    buf.push_str(&text);
                                    buf.push('\n');
                                }
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
                            buf.push_str(&format!("[killed] CLI agent appears stuck waiting for input: {}\n", line));
                            drop(buf);
                            kill_process(pid_for_kill).await;
                            break;
                        }
                    }
                }

                // Emit progress updates at intervals
                // Parse JSON lines to extract meaningful progress, filter raw JSON
                if last_progress.elapsed() >= PROGRESS_INTERVAL {
                    if let Some(ref tx) = status_tx_clone {
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

                        let _ = tx.try_send(StatusUpdate::ToolProgress {
                            name: tool_name_owned.clone(),
                            chunk: truncate_with_note(&chunk, 500),
                        });
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
            let _ = completion_tx.send((exit_code, loop_detected, loop_pattern_count));
        });

        // For async_mode, return immediately with task_id
        if async_mode {
            let working_dir_owned = working_dir.map(|s| s.to_string());
            let agent = RunningCliAgent {
                tool_name: tool_name.to_string(),
                prompt_summary: short_summary.clone(),
                started_at: started_at_instant,
                display_buf,
                stdout_buf,
                child_id: pid,
                session_id: session_id.to_string(),
                working_dir: working_dir_owned,
            };
            self.running.lock().await.insert(task_id.clone(), agent);

            return Ok(format!(
                "CLI agent '{}' started in background (task_id={}). \
                 Use action=\"check\" with task_id=\"{}\" to see output when done.",
                tool_name, task_id, task_id
            ));
        }

        // Wait for completion with timeout
        let state = self.state.clone();
        let working_dir_owned = working_dir.map(|s| s.to_string());
        let dir_locks = self.working_dir_locks.clone();

        let result = tokio::time::timeout(timeout, completion_rx).await;

        // Release working directory lock
        if let Some(ref dir) = working_dir_owned {
            dir_locks.lock().await.remove(dir);
        }

        let duration = started_at_instant.elapsed().as_secs_f64();

        match result {
            Ok(Ok((exit_code, was_loop_killed, loop_count))) => {
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
                        let _ = tx.try_send(StatusUpdate::ToolComplete {
                            name: "cli_agent".to_string(),
                            summary: format!("{} killed - infinite loop detected", tool_name),
                        });
                    }

                    // Log completion
                    let _ = state
                        .log_cli_agent_complete(
                            invocation_id,
                            None,
                            "Killed - infinite loop detected",
                            false,
                            duration,
                        )
                        .await;

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
                    let summary = if exit_code == Some(0) {
                        format!("{} completed successfully", tool_name)
                    } else {
                        format!("{} exited with code {:?}", tool_name, exit_code)
                    };
                    let _ = tx.try_send(StatusUpdate::ToolComplete {
                        name: "cli_agent".to_string(),
                        summary,
                    });
                }

                if exit_code != Some(0) {
                    // On error, show the display buffer which includes stderr
                    let display_output = display_buf.lock().await.clone();

                    // Check for auth errors
                    if let Some(auth_msg) = Self::detect_auth_error(&display_output, tool_name) {
                        let _ = state
                            .log_cli_agent_complete(
                                invocation_id,
                                exit_code,
                                &auth_msg,
                                false,
                                duration,
                            )
                            .await;
                        return Ok(auth_msg);
                    }

                    let output_summary: String = display_output.chars().take(200).collect();
                    let _ = state
                        .log_cli_agent_complete(
                            invocation_id,
                            exit_code,
                            &output_summary,
                            false,
                            duration,
                        )
                        .await;

                    let mut error_msg = format!(
                        "ERROR: CLI agent '{}' failed (exit code {:?}).\n\n## Error Output\n{}",
                        tool_name,
                        exit_code,
                        truncate_with_note(&display_output, max_output)
                    );

                    // Append diff even on failure (partial changes)
                    if let Some(diff) = diff_section {
                        error_msg.push_str(&diff);
                    }

                    error_msg.push_str(
                        "\n\n## Recovery Options\n\
                         - Try a different CLI agent\n\
                         - Handle the task directly with your own tools\n\
                         - Revert partial changes with `git checkout .` if needed",
                    );

                    return Ok(error_msg);
                }

                // Success path
                let output_summary: String = result_text.chars().take(200).collect();
                let _ = state
                    .log_cli_agent_complete(
                        invocation_id,
                        exit_code,
                        &output_summary,
                        true,
                        duration,
                    )
                    .await;

                let mut final_result = result_text;
                if let Some(diff) = diff_section {
                    final_result.push_str(&diff);
                }
                Ok(final_result)
            }
            Ok(Err(_)) => {
                // Channel closed unexpectedly
                let _ = state
                    .log_cli_agent_complete(
                        invocation_id,
                        None,
                        "Task failed unexpectedly",
                        false,
                        duration,
                    )
                    .await;
                Ok(format!(
                    "ERROR: CLI agent '{}' task failed unexpectedly",
                    tool_name
                ))
            }
            Err(_) => {
                // Timeout - move to background
                // Note: the spawned task continues running and will update buffers
                let elapsed = timeout.as_secs();
                let partial_output = {
                    let buf = display_buf.lock().await;
                    truncate_with_note(&buf, 1000)
                };

                // Store the running agent for later checking/cancellation
                let agent = RunningCliAgent {
                    tool_name: tool_name.to_string(),
                    prompt_summary: short_summary.clone(),
                    started_at: started_at_instant,
                    display_buf,
                    stdout_buf,
                    child_id: pid,
                    session_id: session_id.to_string(),
                    working_dir: working_dir_owned,
                };
                self.running.lock().await.insert(task_id.clone(), agent);

                Ok(format!(
                    "CLI agent '{}' still running after {}s. Moved to background (task_id={}).\n\
                     Use action=\"check\" with task_id=\"{}\" to see output, or action=\"cancel\" to stop it.\n\n\
                     Partial output:\n{}",
                    tool_name, elapsed, task_id, task_id, partial_output
                ))
            }
        }
    }

    /// Check on a background CLI agent task.
    async fn handle_check(&self, task_id: &str) -> anyhow::Result<String> {
        let running = self.running.lock().await;

        let Some(agent) = running.get(task_id) else {
            return Ok(format!("No running CLI agent with task_id '{}'", task_id));
        };

        let elapsed = agent.started_at.elapsed().as_secs();
        let display_output = agent.display_buf.lock().await.clone();

        // Check if process is still alive
        let is_running = is_process_alive(agent.child_id);

        if !is_running {
            // Process finished - try to extract meaningful output from stdout
            let stdout_output = agent.stdout_buf.lock().await.clone();
            let result = extract_meaningful_output(&stdout_output, 10000);

            // Capture git diff for finished background tasks
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
            Ok(final_result)
        } else {
            Ok(format!(
                "CLI agent '{}' still running ({}s elapsed, pid={}).\n\
                 Task: {}...\n\n\
                 Partial output ({} chars):\n{}",
                agent.tool_name,
                elapsed,
                agent.child_id,
                agent.prompt_summary,
                display_output.len(),
                truncate_with_note(&display_output, 5000)
            ))
        }
    }

    /// Cancel a background CLI agent task.
    async fn handle_cancel(&self, task_id: &str) -> anyhow::Result<String> {
        let mut running = self.running.lock().await;

        let Some(agent) = running.remove(task_id) else {
            return Ok(format!("No running CLI agent with task_id '{}'", task_id));
        };

        let display_output = agent.display_buf.lock().await.clone();
        let elapsed = agent.started_at.elapsed().as_secs();

        // Release working dir lock
        if let Some(ref dir) = agent.working_dir {
            self.working_dir_locks.lock().await.remove(dir);
        }

        // Try to kill the process
        kill_process(agent.child_id).await;

        Ok(format!(
            "Cancelled CLI agent '{}' (was running for {}s).\n\nOutput before cancellation:\n{}",
            agent.tool_name,
            elapsed,
            truncate_with_note(&display_output, 5000)
        ))
    }

    /// Cancel all CLI agent tasks for a specific session.
    async fn handle_cancel_all(&self, session_id: &str) -> anyhow::Result<String> {
        let mut running = self.running.lock().await;

        // Find all tasks matching this session
        let to_cancel: Vec<String> = running
            .iter()
            .filter(|(_, agent)| agent.session_id == session_id)
            .map(|(task_id, _)| task_id.clone())
            .collect();

        if to_cancel.is_empty() {
            return Ok("No running CLI agents for this session.".to_string());
        }

        let mut cancelled = Vec::new();
        for task_id in to_cancel {
            if let Some(agent) = running.remove(&task_id) {
                if let Some(ref dir) = agent.working_dir {
                    self.working_dir_locks.lock().await.remove(dir);
                }
                kill_process(agent.child_id).await;
                cancelled.push(format!("{} ({})", agent.tool_name, task_id));
            }
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
            let status = if is_process_alive(agent.child_id) {
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
    /// Injected by agent for role-aware safeguards.
    #[serde(default)]
    _user_role: Option<String>,
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
        // Read from RwLock (sync) — this is fine because std::sync::RwLock doesn't need .await
        let tools = self.tools.read().unwrap();
        let tool_names = self.tool_names.read().unwrap();

        let tool_descriptions: Vec<String> = tool_names
            .iter()
            .filter_map(|name| {
                tools.get(name).map(|entry| {
                    if entry.description.is_empty() {
                        name.clone()
                    } else {
                        format!("{} — {}", name, entry.description)
                    }
                })
            })
            .collect();

        let tools_help = tool_descriptions.join(", ");
        let names_vec: Vec<Value> = tool_names.iter().map(|n| json!(n)).collect();

        json!({
            "name": "cli_agent",
            "description": format!(
                "Delegate a task to a CLI-based AI coding agent. Available agents: {}. \
                 These are full AI agents that can read/write files, run commands, and solve complex coding tasks. \
                 Use this for substantial coding work, refactoring, debugging, or any task that benefits from a \
                 specialized AI coding tool. Long-running tasks move to background and can be checked/cancelled.",
                tools_help
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run", "check", "cancel", "list"],
                        "description": "Action to perform: \"run\" (default) starts a CLI agent, \"check\" shows output of a background task, \"cancel\" stops a background task, \"list\" shows all running tasks"
                    },
                    "tool": {
                        "type": "string",
                        "enum": names_vec,
                        "description": "Which CLI agent to use (required for action=run)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task or prompt to send to the CLI agent. Be specific and detailed."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the CLI agent (absolute path). If not specified, uses the current directory."
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID for check/cancel actions (returned when a task moves to background)"
                    },
                    "system_instruction": {
                        "type": "string",
                        "description": "Optional expert instruction to shape the CLI agent into a specialist (e.g. 'You are a security auditor'). When provided, the prompt is enriched with conversation context and relevant facts."
                    },
                    "async_mode": {
                        "type": "boolean",
                        "description": "If true, starts the task in background immediately and returns a task_id. Use for parallel dispatch of multiple CLI agents."
                    }
                },
                "required": []
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
        // For backwards compatibility, delegate to call_with_status with no sender
        self.call_with_status(arguments, None).await
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let args: CliAgentArgs = serde_json::from_str(arguments)?;

        // Reap any finished tasks
        self.reap_finished().await;

        let action = args.action.as_deref().unwrap_or("run");

        let session_id = args._session_id.clone().unwrap_or_default();

        match action {
            "run" => {
                let tool = args
                    .tool
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'tool' parameter for action=run"))?;
                let prompt = args
                    .prompt
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'prompt' parameter for action=run"))?;

                let mut daemon_hits = detect_daemonization_primitives(prompt);
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
                            tool,
                            prompt,
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
                self.handle_run(
                    tool,
                    prompt,
                    args.working_dir.as_deref(),
                    &session_id,
                    args.system_instruction.as_deref(),
                    args.async_mode.unwrap_or(false),
                    status_tx,
                )
                .await
            }
            "check" => {
                let task_id = args.task_id.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Missing 'task_id' parameter for action=check")
                })?;
                self.handle_check(task_id).await
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
    use crate::traits::Tool;
    use std::collections::HashMap;
    use std::sync::Arc;

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
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);

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
            tools: Arc::new(std::sync::RwLock::new(tools_map)),
            tool_names: Arc::new(std::sync::RwLock::new(vec!["echo".to_string()])),
            running: Arc::new(Mutex::new(HashMap::new())),
            working_dir_locks: Arc::new(Mutex::new(HashSet::new())),
            state: state as Arc<dyn StateStore>,
            provider: provider as Arc<dyn crate::traits::ModelProvider>,
            default_timeout: Duration::from_secs(10),
            default_max_output: 10000,
            max_concurrent: 3,
            approval_tx,
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
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);

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
            tools: Arc::new(std::sync::RwLock::new(tools_map)),
            tool_names: Arc::new(std::sync::RwLock::new(vec!["bash-agent".to_string()])),
            running: Arc::new(Mutex::new(HashMap::new())),
            working_dir_locks: Arc::new(Mutex::new(HashSet::new())),
            state: state as Arc<dyn StateStore>,
            provider: provider as Arc<dyn crate::traits::ModelProvider>,
            default_timeout: Duration::from_secs(10),
            default_max_output: 10000,
            max_concurrent: 3,
            approval_tx,
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
    async fn test_run_unknown_tool() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool
            .call(r#"{"action":"run","tool":"nonexistent","prompt":"test"}"#)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_missing_tool_param() {
        let (tool, _db) = setup_echo_tool().await;
        let result = tool.call(r#"{"action":"run","prompt":"test"}"#).await;
        assert!(result.is_err());
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
    // Working directory lock test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_working_dir_conflict_detection() {
        let (tool, _db) = setup_bash_tool().await;

        // Lock a working directory
        tool.working_dir_locks
            .lock()
            .await
            .insert("/tmp/project".to_string());

        let result = tool
            .call(
                r#"{"action":"run","tool":"bash-agent","prompt":"echo test","working_dir":"/tmp/project"}"#,
            )
            .await
            .unwrap();
        assert!(
            result.contains("WARNING") && result.contains("Another CLI agent"),
            "Expected working dir conflict warning, got: {}",
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

        // Working dir lock should be released after completion
        let locks = tool.working_dir_locks.lock().await;
        assert!(
            !locks.contains(dir_path),
            "Working dir lock not released after completion"
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
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);

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
            provider as Arc<dyn crate::traits::ModelProvider>,
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
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);

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
            provider as Arc<dyn crate::traits::ModelProvider>,
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
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(1);

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
            tools: Arc::new(std::sync::RwLock::new(tools_map)),
            tool_names: Arc::new(std::sync::RwLock::new(vec!["echo".to_string()])),
            running: Arc::new(Mutex::new(HashMap::new())),
            working_dir_locks: Arc::new(Mutex::new(HashSet::new())),
            state: state as Arc<dyn StateStore>,
            provider: provider as Arc<dyn crate::traits::ModelProvider>,
            default_timeout: Duration::from_secs(10),
            default_max_output: 10000,
            max_concurrent: 3,
            approval_tx,
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

        // Clear all tools
        tool.tools.write().unwrap().clear();
        assert!(!tool.has_tools());
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
            .build_enriched_prompt("test-session", "", "Just do the task")
            .await;

        // Empty instruction should not appear
        assert!(prompt.contains("Just do the task"));
        assert!(prompt.contains("## Task"));
    }

    // -----------------------------------------------------------------------
    // Scenario replication: exact user prompt that caused the hang
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_user_prompt_website_about_cars() {
        // This replicates the exact prompt that caused a 5-minute hang.
        // The fix: stdin is now Stdio::null() so processes get EOF immediately.
        let (tool, _db) = setup_bash_tool().await;

        let user_prompt = "I need to create a new website about cars. We should push it to cars.davidloor.com. make it modern.";

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
}
