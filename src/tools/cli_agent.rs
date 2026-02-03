use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::config::CliAgentsConfig;
use crate::traits::Tool;

struct CliToolEntry {
    command: String,
    args: Vec<String>,
    description: String,
    timeout: Duration,
    max_output_chars: usize,
}

pub struct CliAgentTool {
    tools: HashMap<String, CliToolEntry>,
    tool_names: Vec<String>, // sorted for deterministic schema
}

/// Default tool definitions when the user enables cli_agents but doesn't specify tools.
fn default_tool_definitions() -> Vec<(&'static str, &'static str, Vec<&'static str>, &'static str)> {
    vec![
        ("claude", "claude", vec!["-p", "--output-format", "json"], "Claude Code — Anthropic's AI coding agent"),
        ("gemini", "gemini", vec!["-p", "--output-format", "json", "--sandbox=false"], "Gemini CLI — Google's AI coding agent"),
        ("codex", "codex", vec!["exec", "--json", "--full-auto"], "Codex CLI — OpenAI's AI coding agent"),
        ("copilot", "copilot", vec!["-p"], "GitHub Copilot CLI"),
        ("aider", "aider", vec!["--yes", "--message"], "Aider — AI pair programming"),
    ]
}

impl CliAgentTool {
    pub async fn discover(config: CliAgentsConfig) -> Self {
        let default_timeout = Duration::from_secs(config.timeout_secs);
        let default_max_output = config.max_output_chars;

        // Build the list of tools to check: user-configured or defaults
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
            // Check if the command exists on the system
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

        CliAgentTool { tools, tool_names }
    }

    pub fn has_tools(&self) -> bool {
        !self.tools.is_empty()
    }
}

#[derive(Deserialize)]
struct CliAgentArgs {
    tool: String,
    prompt: String,
    working_dir: Option<String>,
}

/// Try to extract the meaningful content from JSON output of known CLI tools.
fn extract_json_content(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;

    // Claude Code JSON: top-level "result" field
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

/// Try to extract content from JSONL output (e.g., Codex outputs events line by line).
fn extract_jsonl_content(raw: &str) -> Option<String> {
    // Look for the last line that parses as JSON with a completed-like event
    let mut last_content: Option<String> = None;
    for line in raw.lines().rev() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            // Codex JSONL: look for item with type "message" and content
            if let Some(content) = v.pointer("/item/content")
                .or_else(|| v.pointer("/content"))
                .or_else(|| v.pointer("/result"))
            {
                if let Some(text) = content.as_str() {
                    last_content = Some(text.to_string());
                    break;
                }
                // Content might be an array of objects with "text" fields
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

/// Combine stdout and stderr into a single raw output string.
fn format_raw_output(stdout: &str, stderr: &str) -> String {
    let mut raw = String::new();
    if !stdout.is_empty() {
        raw.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !raw.is_empty() {
            raw.push_str("\n--- stderr ---\n");
        }
        raw.push_str(stderr);
    }
    if raw.is_empty() {
        raw.push_str("(no output)");
    }
    raw
}

/// Truncate a string to at most `max_chars`, returning a new string.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.len() > max_chars {
        let mut t = s[..max_chars].to_string();
        t.push_str("\n... (truncated)");
        t
    } else {
        s.to_string()
    }
}

/// Truncate a string in place to at most `max_chars`.
fn truncate_in_place(s: &mut String, max_chars: usize) {
    if s.len() > max_chars {
        s.truncate(max_chars);
        s.push_str("\n... (truncated)");
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
                 specialized AI coding tool.",
                tools_help
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": self.tool_names,
                        "description": "Which CLI agent to use"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task or prompt to send to the CLI agent. Be specific and detailed."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the CLI agent (absolute path). If not specified, uses the current directory."
                    }
                },
                "required": ["tool", "prompt"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: CliAgentArgs = serde_json::from_str(arguments)?;

        let entry = self.tools.get(&args.tool)
            .ok_or_else(|| anyhow::anyhow!("Unknown CLI agent tool: {}", args.tool))?;

        // Build command: command + args + prompt as last arg
        let mut cmd = tokio::process::Command::new(&entry.command);
        for arg in &entry.args {
            cmd.arg(arg);
        }
        cmd.arg(&args.prompt);

        // Set working directory if provided
        if let Some(ref dir) = args.working_dir {
            cmd.current_dir(dir);
        }

        // Prevent interactive behavior
        cmd.stdin(std::process::Stdio::null());

        info!(
            tool = %args.tool,
            command = %entry.command,
            working_dir = ?args.working_dir,
            "Invoking CLI agent"
        );

        // Execute with timeout
        let result = tokio::time::timeout(
            entry.timeout,
            cmd.output(),
        ).await;

        let output = match result {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return Ok(format!("Failed to execute {}: {}", entry.command, e));
            }
            Err(_) => {
                return Ok(format!(
                    "CLI agent '{}' timed out after {} seconds",
                    args.tool,
                    entry.timeout.as_secs()
                ));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            warn!(
                tool = %args.tool,
                exit_code = ?output.status.code(),
                "CLI agent exited with non-zero status"
            );
            let code = output.status.code().map_or("unknown".to_string(), |c| c.to_string());
            let combined = format_raw_output(&stdout, &stderr);
            return Ok(format!(
                "ERROR: CLI agent '{}' exited with code {}. Do NOT retry this tool with the same agent.\n\nOutput:\n{}",
                args.tool, code, truncate(&combined, entry.max_output_chars)
            ));
        }

        // Try to extract structured content from JSON/JSONL output
        let mut result_text = if let Some(content) = extract_json_content(&stdout) {
            content
        } else if let Some(content) = extract_jsonl_content(&stdout) {
            content
        } else {
            format_raw_output(&stdout, &stderr)
        };

        // Truncate to max output chars
        truncate_in_place(&mut result_text, entry.max_output_chars);

        Ok(result_text)
    }
}
