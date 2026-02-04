use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, Mutex};
use tracing::info;
use uuid::Uuid;

use crate::config::CliAgentsConfig;
use crate::traits::Tool;
use crate::types::StatusUpdate;

/// Max bytes for output buffer (1 MB) to prevent unbounded memory growth.
const BUFFER_CAP: usize = 1_048_576;

/// Interval for emitting progress updates (avoid spamming the channel).
const PROGRESS_INTERVAL: Duration = Duration::from_secs(2);

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
    use std::process::Command;
    // Send SIGTERM
    let _ = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();

    // Wait a bit
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Send SIGKILL if still alive
    if is_process_alive(pid) {
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
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
}

pub struct CliAgentTool {
    tools: HashMap<String, CliToolEntry>,
    tool_names: Vec<String>,
    running: Arc<Mutex<HashMap<String, RunningCliAgent>>>, // task_id -> RunningCliAgent
}

/// Default tool definitions when the user enables cli_agents but doesn't specify tools.
fn default_tool_definitions() -> Vec<(&'static str, &'static str, Vec<&'static str>, &'static str)> {
    vec![
        ("claude", "claude", vec!["-p", "--dangerously-skip-permissions"], "Claude Code — Anthropic's AI coding agent (auto-approve mode)"),
        ("gemini", "gemini", vec!["-p", "--sandbox=false", "--auto-approve"], "Gemini CLI — Google's AI coding agent (auto-approve mode)"),
        ("codex", "codex", vec!["exec", "--json", "--full-auto"], "Codex CLI — OpenAI's AI coding agent"),
        ("copilot", "copilot", vec!["-p", "--allow-all-tools", "--allow-all-paths"], "GitHub Copilot CLI (auto-approve mode)"),
        ("aider", "aider", vec!["--yes", "--message"], "Aider — AI pair programming"),
    ]
}

impl CliAgentTool {
    pub async fn discover(config: CliAgentsConfig) -> Self {
        let default_timeout = Duration::from_secs(config.timeout_secs);
        let default_max_output = config.max_output_chars;

        let mut candidates: Vec<(String, String, Vec<String>, String, Option<u64>, Option<usize>)> = Vec::new();

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

        let mut tools = HashMap::new();

        for (name, command, args, description, timeout_override, max_output_override) in candidates {
            let which = tokio::process::Command::new("which")
                .arg(&command)
                .output()
                .await;

            match which {
                Ok(output) if output.status.success() => {
                    info!(name = %name, command = %command, "CLI agent tool discovered");
                    tools.insert(name.clone(), CliToolEntry {
                        command,
                        args,
                        description,
                        timeout: timeout_override
                            .map(Duration::from_secs)
                            .unwrap_or(default_timeout),
                        max_output_chars: max_output_override
                            .unwrap_or(default_max_output),
                    });
                }
                _ => {
                    info!(name = %name, command = %command, "CLI agent tool not found, skipping");
                }
            }
        }

        let mut tool_names: Vec<String> = tools.keys().cloned().collect();
        tool_names.sort();

        CliAgentTool {
            tools,
            tool_names,
            running: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn has_tools(&self) -> bool {
        !self.tools.is_empty()
    }

    /// Clean up any finished CLI agent tasks.
    async fn reap_finished(&self) {
        let mut running = self.running.lock().await;
        let finished: Vec<String> = running
            .iter()
            .filter(|(_, agent)| !is_process_alive(agent.child_id))
            .map(|(id, _)| id.clone())
            .collect();
        for task_id in finished {
            if let Some(agent) = running.remove(&task_id) {
                info!(task_id, tool = %agent.tool_name, "Reaped finished CLI agent");
            }
        }
    }

    /// Run a CLI agent with streaming output.
    async fn handle_run(
        &self,
        tool_name: &str,
        prompt: &str,
        working_dir: Option<&str>,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let entry = self.tools.get(tool_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown CLI agent tool: {}", tool_name))?;

        // Build command
        let mut cmd = tokio::process::Command::new(&entry.command);
        for arg in &entry.args {
            cmd.arg(arg);
        }
        cmd.arg(prompt);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        let prompt_summary: String = prompt.chars().take(50).collect();

        info!(
            task_id,
            tool = %tool_name,
            command = %entry.command,
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

        let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("Failed to capture stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow::anyhow!("Failed to capture stderr"))?;

        // Two buffers: stdout_buf for JSON extraction, display_buf for user display
        let stdout_buf = Arc::new(Mutex::new(String::new()));
        let display_buf = Arc::new(Mutex::new(String::new()));
        let stdout_buf_writer = stdout_buf.clone();
        let display_buf_writer = display_buf.clone();
        let status_tx_clone = status_tx.clone();
        let tool_name_owned = tool_name.to_string();

        // Create completion channel
        let (completion_tx, completion_rx) = tokio::sync::oneshot::channel::<Option<i32>>();

        // Spawn a task to read stdout/stderr, emit progress updates, and signal completion
        tokio::spawn(async move {
            let mut stdout_reader = BufReader::new(stdout).lines();
            let mut stderr_reader = BufReader::new(stderr).lines();
            let mut last_progress = Instant::now();
            let mut pending_lines: Vec<String> = Vec::new();
            let mut stdout_done = false;
            let mut stderr_done = false;

            loop {
                if stdout_done && stderr_done {
                    break;
                }

                tokio::select! {
                    line = stdout_reader.next_line(), if !stdout_done => {
                        match line {
                            Ok(Some(text)) => {
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
                }

                // Emit progress updates at intervals
                // Parse JSON lines to extract meaningful progress, filter raw JSON
                if last_progress.elapsed() >= PROGRESS_INTERVAL && !pending_lines.is_empty() {
                    if let Some(ref tx) = status_tx_clone {
                        let mut progress_items: Vec<String> = Vec::new();
                        for line in &pending_lines {
                            if looks_like_json(line) {
                                // Try to extract meaningful progress from JSON
                                if let Some(progress) = extract_progress_from_json(line) {
                                    progress_items.push(progress);
                                }
                            } else {
                                // Non-JSON line, include as-is
                                progress_items.push(line.clone());
                            }
                        }
                        if !progress_items.is_empty() {
                            // Deduplicate consecutive items
                            progress_items.dedup();
                            let chunk = progress_items.join("\n");
                            let _ = tx.try_send(StatusUpdate::ToolProgress {
                                name: tool_name_owned.clone(),
                                chunk: truncate_string(&chunk, 500),
                            });
                        }
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
                            chunk: truncate_string(&chunk, 500),
                        });
                    }
                }
            }

            // Wait for process to complete and signal via channel
            let exit_code = match child.wait().await {
                Ok(status) => status.code(),
                Err(_) => None,
            };
            let _ = completion_tx.send(exit_code);
        });

        // Wait for completion with timeout
        let timeout = entry.timeout;
        let max_output = entry.max_output_chars;

        let result = tokio::time::timeout(timeout, completion_rx).await;

        match result {
            Ok(Ok(exit_code)) => {
                // Completed within timeout
                // Use stdout_buf for JSON extraction (clean, no stderr prefixes)
                let stdout_output = stdout_buf.lock().await.clone();
                info!(
                    tool = %tool_name,
                    stdout_len = stdout_output.len(),
                    stdout_preview = %truncate_string(&stdout_output, 200),
                    "CLI agent stdout captured"
                );
                let result_text = extract_meaningful_output(&stdout_output, max_output);
                info!(
                    tool = %tool_name,
                    result_len = result_text.len(),
                    result_preview = %truncate_string(&result_text, 200),
                    "CLI agent result extracted"
                );

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
                    return Ok(format!(
                        "ERROR: CLI agent '{}' exited with code {:?}. Do NOT retry with same agent.\n\nOutput:\n{}",
                        tool_name, exit_code, truncate_string(&display_output, max_output)
                    ));
                }

                Ok(result_text)
            }
            Ok(Err(_)) => {
                // Channel closed unexpectedly
                Ok(format!("ERROR: CLI agent '{}' task failed unexpectedly", tool_name))
            }
            Err(_) => {
                // Timeout - move to background
                // Note: the spawned task continues running and will update buffers
                let elapsed = timeout.as_secs();
                let partial_output = {
                    let buf = display_buf.lock().await;
                    truncate_string(&buf, 1000)
                };

                // Store the running agent for later checking/cancellation
                let agent = RunningCliAgent {
                    tool_name: tool_name.to_string(),
                    prompt_summary: prompt_summary.clone(),
                    started_at: Instant::now(),
                    display_buf,
                    stdout_buf,
                    child_id: pid,
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
            Ok(format!(
                "CLI agent '{}' finished after {}s.\n\nResult:\n{}",
                agent.tool_name,
                elapsed,
                result
            ))
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
                truncate_string(&display_output, 5000)
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

        // Try to kill the process
        kill_process(agent.child_id).await;

        Ok(format!(
            "Cancelled CLI agent '{}' (was running for {}s).\n\nOutput before cancellation:\n{}",
            agent.tool_name,
            elapsed,
            truncate_string(&display_output, 5000)
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
                    let short_path: String = path.chars().rev().take(50).collect::<String>().chars().rev().collect();
                    return Some(format!("📖 Reading: ...{}", short_path));
                }
            } else if tool_name == "Write" || tool_name == "write" || tool_name == "Edit" || tool_name == "edit" {
                if let Some(path) = input.get("file_path").and_then(|p| p.as_str()) {
                    let short_path: String = path.chars().rev().take(50).collect::<String>().chars().rev().collect();
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
                // Assistant is thinking/responding
                if let Some(content) = v.get("message").and_then(|m| m.get("content")) {
                    if let Some(arr) = content.as_array() {
                        for item in arr {
                            if item.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                                if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                                    return Some(format!("🔧 Using: {}", name));
                                }
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
        return truncate_string(&content, max_chars);
    }
    if let Some(content) = extract_jsonl_content(raw) {
        return truncate_string(&content, max_chars);
    }
    truncate_string(raw, max_chars)
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
            if let Some(content) = v.pointer("/item/content")
                .or_else(|| v.pointer("/content"))
                .or_else(|| v.pointer("/result"))
            {
                if let Some(text) = content.as_str() {
                    last_content = Some(text.to_string());
                    break;
                }
                if let Some(arr) = content.as_array() {
                    let texts: Vec<&str> = arr.iter()
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

/// Truncate a string to at most `max_chars`.
fn truncate_string(s: &str, max_chars: usize) -> String {
    if s.len() > max_chars {
        let mut t = s[..max_chars].to_string();
        t.push_str("\n... (truncated)");
        t
    } else {
        s.to_string()
    }
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
        let tool_descriptions: Vec<String> = self.tool_names.iter()
            .filter_map(|name| {
                self.tools.get(name).map(|entry| {
                    if entry.description.is_empty() {
                        name.clone()
                    } else {
                        format!("{} — {}", name, entry.description)
                    }
                })
            })
            .collect();

        let tools_help = tool_descriptions.join(", ");

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
                        "enum": self.tool_names,
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
                    }
                },
                "required": []
            }
        })
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

        match action {
            "run" => {
                let tool = args.tool.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'tool' parameter for action=run"))?;
                let prompt = args.prompt.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'prompt' parameter for action=run"))?;
                self.handle_run(tool, prompt, args.working_dir.as_deref(), status_tx).await
            }
            "check" => {
                let task_id = args.task_id.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'task_id' parameter for action=check"))?;
                self.handle_check(task_id).await
            }
            "cancel" => {
                let task_id = args.task_id.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'task_id' parameter for action=cancel"))?;
                self.handle_cancel(task_id).await
            }
            "list" => {
                self.handle_list().await
            }
            _ => {
                Ok(format!("Unknown action '{}'. Use run, check, cancel, or list.", action))
            }
        }
    }
}
