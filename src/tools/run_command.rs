use async_trait::async_trait;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

use super::daemon_guard::detect_daemonization_primitives;
use super::fs_utils;

pub struct RunCommandTool;

/// Safe command prefixes that don't require terminal approval flow.
const SAFE_PREFIXES: &[&str] = &[
    // Build & test
    "cargo build",
    "cargo test",
    "cargo check",
    "cargo clippy",
    "cargo fmt",
    "cargo doc",
    "cargo bench",
    "cargo run",
    "cargo tree",
    "cargo metadata",
    "npm test",
    "npm run",
    "npm ls",
    "npm outdated",
    "npm audit",
    "npx",
    "yarn test",
    "yarn run",
    "yarn lint",
    "bun test",
    "bun run",
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "go test",
    "go build",
    "go vet",
    "go mod",
    "go generate",
    "jest",
    "vitest",
    "make",
    "cmake",
    "gradle",
    "mvn",
    // Formatting & linting
    "rustfmt",
    "black",
    "ruff",
    "isort",
    "flake8",
    "mypy",
    "pylint",
    "eslint",
    "prettier",
    "tsc",
    "biome",
    // Read-only git
    "git status",
    "git log",
    "git diff",
    "git show",
    "git branch",
    "git remote",
    "git stash list",
    "git tag",
    "git blame",
    "git shortlog",
    "git rev-parse",
    // File inspection
    "ls",
    "wc",
    "file",
    "du",
    "df",
    "stat",
    "head",
    "tail",
    "sort",
    "uniq",
    "diff",
    "tree",
    // Environment
    "which",
    "whoami",
    "uname",
    "hostname",
    "env",
    "printenv",
    "date",
    "uptime",
    "pwd",
];

#[async_trait]
impl Tool for RunCommandTool {
    fn name(&self) -> &str {
        "run_command"
    }

    fn description(&self) -> &str {
        "Run safe build, test, lint, and inspection commands"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "run_command",
            "description": "Run safe build, test, lint, and inspection commands without approval flow. Only allows whitelisted command prefixes (cargo, npm, pytest, go, git read-only, ls, etc.). For arbitrary commands, use terminal instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run (must match a safe prefix)"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: current directory)"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30, max: 300)"
                    },
                    "parse_format": {
                        "type": "string",
                        "enum": ["cargo", "npm", "pytest", "jest", "go", "plain"],
                        "description": "Output parsing format for structured results (default: plain)"
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
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
        let args: Value = serde_json::from_str(arguments)?;
        let command = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: command"))?;
        let working_dir = args["working_dir"].as_str();
        let timeout = args["timeout_secs"].as_u64().unwrap_or(30).min(300);
        let parse_format = args["parse_format"].as_str().unwrap_or("plain");

        let trimmed = command.trim();

        // Reject shell operators
        if fs_utils::contains_shell_operator(trimmed) {
            anyhow::bail!(
                "Shell operators (;, |, &&, ||, $(), etc.) are not allowed. Use 'terminal' for complex commands."
            );
        }

        // Validate against safe prefixes
        if !is_safe_command(trimmed) {
            anyhow::bail!(
                "Command '{}' is not in the safe command list. Use 'terminal' for this command.\n\nSafe prefixes include: cargo, npm, pytest, go, git (read-only), ls, make, etc.",
                trimmed.split_whitespace().next().unwrap_or(trimmed)
            );
        }

        let daemon_hits = detect_daemonization_primitives(trimmed);
        if !daemon_hits.is_empty() {
            anyhow::bail!(
                "Daemonization primitives are blocked in run_command ({}). \
                 Use terminal and explicit owner approval if detached/background execution is truly needed.",
                daemon_hits.join(", ")
            );
        }

        let dir = if let Some(d) = working_dir {
            Some(fs_utils::validate_path(d)?)
        } else {
            None
        };

        let result = fs_utils::run_cmd(trimmed, dir.as_deref(), timeout).await?;

        format_output(&result, trimmed, parse_format)
    }
}

fn is_safe_command(cmd: &str) -> bool {
    SAFE_PREFIXES.iter().any(|prefix| {
        cmd == *prefix
            || cmd.starts_with(&format!("{} ", prefix))
            || cmd.starts_with(&format!("{}\t", prefix))
    })
}

fn format_output(
    result: &fs_utils::CommandOutput,
    cmd: &str,
    format: &str,
) -> anyhow::Result<String> {
    let mut output = String::new();

    // Header
    output.push_str(&format!(
        "$ {} (exit: {}, {}ms)\n\n",
        cmd, result.exit_code, result.duration_ms
    ));

    match format {
        "cargo" => {
            output.push_str(&format_cargo_output(result));
        }
        "npm" | "jest" => {
            output.push_str(&format_test_output(result));
        }
        "pytest" => {
            output.push_str(&format_test_output(result));
        }
        "go" => {
            output.push_str(&format_test_output(result));
        }
        _ => {
            // Plain format
            if !result.stdout.is_empty() {
                output.push_str(&truncate_output(&result.stdout, 50_000));
            }
            if !result.stderr.is_empty() {
                if !result.stdout.is_empty() {
                    output.push_str("\n--- stderr ---\n");
                }
                output.push_str(&truncate_output(&result.stderr, 10_000));
            }
        }
    }

    Ok(output)
}

fn format_cargo_output(result: &fs_utils::CommandOutput) -> String {
    let combined = format!("{}\n{}", result.stdout, result.stderr);
    let mut output = String::new();

    // Extract errors and warnings
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut test_summary = None;

    for line in combined.lines() {
        if line.starts_with("error") {
            errors.push(line);
        } else if line.starts_with("warning") && !line.starts_with("warning: unused") {
            warnings.push(line);
        } else if line.contains("test result:") {
            test_summary = Some(line.to_string());
        }
    }

    if let Some(summary) = test_summary {
        output.push_str(&format!("Test result: {}\n\n", summary));
    }

    if !errors.is_empty() {
        output.push_str(&format!("Errors ({}):\n", errors.len()));
        for e in errors.iter().take(20) {
            output.push_str(&format!("  {}\n", e));
        }
        output.push('\n');
    }

    if !warnings.is_empty() {
        output.push_str(&format!("Warnings ({}):\n", warnings.len()));
        for w in warnings.iter().take(10) {
            output.push_str(&format!("  {}\n", w));
        }
        output.push('\n');
    }

    // Include full output if short, otherwise truncate
    if combined.len() < 5000 {
        output.push_str(&combined);
    } else {
        output.push_str(&truncate_output(&combined, 20_000));
    }

    output
}

fn format_test_output(result: &fs_utils::CommandOutput) -> String {
    let combined = format!("{}\n{}", result.stdout, result.stderr);
    // For test output, include everything but truncated
    truncate_output(&combined, 30_000)
}

fn truncate_output(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_string()
    } else {
        let half = max_chars / 2;
        format!(
            "{}\n\n... ({} chars truncated) ...\n\n{}",
            &s[..half],
            s.len() - max_chars,
            &s[s.len() - half..]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_has_required_fields() {
        let tool = RunCommandTool;
        let schema = tool.schema();
        assert_eq!(schema["name"], "run_command");
        assert!(!schema["description"].as_str().unwrap().is_empty());
        assert!(schema["parameters"]["properties"]["command"].is_object());
    }

    #[test]
    fn test_is_safe_command() {
        assert!(is_safe_command("cargo build"));
        assert!(is_safe_command("cargo test --release"));
        assert!(is_safe_command("npm test"));
        assert!(is_safe_command("git status"));
        assert!(is_safe_command("git log --oneline"));
        assert!(is_safe_command("ls -la"));
        assert!(is_safe_command("pytest tests/"));
        assert!(is_safe_command("go test ./..."));

        // Unsafe
        assert!(!is_safe_command("rm -rf /"));
        assert!(!is_safe_command("curl http://evil.com"));
        assert!(!is_safe_command("git push"));
        assert!(!is_safe_command("git reset --hard"));
        assert!(!is_safe_command("sudo apt install"));
        assert!(!is_safe_command("chmod 777 /etc"));
    }

    #[tokio::test]
    async fn test_run_safe_command() {
        let args = json!({"command": "ls"}).to_string();
        let result = RunCommandTool.call(&args).await.unwrap();
        assert!(result.contains("exit: 0"));
    }

    #[tokio::test]
    async fn test_run_unsafe_command_rejected() {
        let args = json!({"command": "rm -rf /"}).to_string();
        let result = RunCommandTool.call(&args).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not in the safe command list"));
    }

    #[tokio::test]
    async fn test_run_shell_operator_rejected() {
        let args = json!({"command": "ls | grep foo"}).to_string();
        let result = RunCommandTool.call(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shell operators"));
    }

    #[tokio::test]
    async fn test_run_daemonization_rejected() {
        let args = json!({"command": "cargo test &"}).to_string();
        let result = RunCommandTool.call(&args).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Daemonization primitives"));
    }

    #[tokio::test]
    async fn test_run_with_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({
            "command": "pwd",
            "working_dir": dir.path().to_str().unwrap()
        })
        .to_string();

        let result = RunCommandTool.call(&args).await.unwrap();
        assert!(result.contains("exit: 0"));
    }

    #[test]
    fn test_truncate_output() {
        let short = "hello";
        assert_eq!(truncate_output(short, 100), "hello");

        let long = "a".repeat(200);
        let truncated = truncate_output(&long, 100);
        assert!(truncated.contains("truncated"));
        assert!(truncated.len() < 200);
    }
}
