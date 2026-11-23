//! Finding and launching an agent's executable, and reading a failed launch.

use super::*;

/// Default tool definitions when the user enables cli_agents but doesn't specify tools.
pub(super) fn default_tool_definitions()
-> Vec<(&'static str, &'static str, Vec<&'static str>, &'static str)> {
    // Each agent says how to find it (`aidaemon_coding_agents`).
    aidaemon_coding_agents::CodingAgents::builtin()
        .all()
        .iter()
        .filter_map(|agent| agent.discovery())
        .map(|found| {
            (
                found.name,
                found.command,
                found.args.to_vec(),
                found.description,
            )
        })
        .collect()
}

/// Preferred fallback order when `action=run` omits `tool`.
///
/// Rationale: keep common coding agents at the front so orchestrator calls can
/// simply provide a prompt and still use a strong default.
pub(super) const DEFAULT_TOOL_PRIORITY: &[&str] = &["claude", "codex", "agy", "copilot", "aider"];

pub(super) fn env_wrapped_program_index(args: &[String]) -> Option<usize> {
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

pub(super) fn is_env_launcher(command: &str) -> bool {
    std::path::Path::new(command)
        .file_name()
        .and_then(|name| name.to_str())
        == Some("env")
}

pub(super) async fn executable_search_candidates(
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

pub(super) async fn resolve_executable(
    backend: &SharedExecutionBackend,
    executable: &str,
) -> Option<String> {
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
pub(super) async fn resolve_agent_launch(
    backend: &SharedExecutionBackend,
    command: &str,
    args: &[String],
) -> Result<(String, Vec<String>), String> {
    let resolved_command = resolve_executable(backend, command)
        .await
        .ok_or_else(|| format!("launcher executable '{command}' was not found"))?;
    let mut resolved_args = args.to_vec();

    if is_env_launcher(&resolved_command)
        && let Some(index) = env_wrapped_program_index(&resolved_args)
    {
        let nested = resolved_args[index].clone();
        let resolved_nested = resolve_executable(backend, &nested).await.ok_or_else(|| {
                format!(
                    "wrapped CLI agent executable '{nested}' was not found in PATH or common user install directories"
                )
            })?;
        resolved_args[index] = resolved_nested;
    }

    Ok((resolved_command, resolved_args))
}

pub(super) fn format_cli_agent_failure(
    tool_name: &str,
    exit_code: Option<i32>,
    display_output: &str,
    max_output: usize,
    diff_section: Option<&str>,
) -> String {
    // Its end stays: the error is usually last.
    let captured = crate::tools::terminal::keep_head_and_tail(display_output, max_output);
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

/// The exit code in the daemon's own failure header
/// ([`format_cli_agent_failure`]: `ERROR: CLI agent '<name>' failed (exit
/// code N).` on the first line). An agent's report that quotes such a line
/// further down is its content and is never read.
pub(super) fn cli_agent_exit_code(output: &str) -> Option<i32> {
    let header = output.lines().next()?;
    if !header.starts_with("ERROR: CLI agent '") {
        return None;
    }
    let marker = "failed (exit code ";
    let suffix = header.split_once(marker)?.1;
    let raw = suffix.split_once(')')?.0.trim();
    raw.strip_prefix("Some(")
        .unwrap_or(raw)
        .trim_end_matches(')')
        .parse()
        .ok()
}

/// Whether a run ended before the agent itself started: the program could
/// not be found or executed (exit 127 from the launcher) and the agent
/// wrote nothing to stdout. Never read from the output's words.
pub(super) fn exited_before_agent_output(exit_code: Option<i32>, stdout: &str) -> bool {
    exit_code == Some(127) && stdout.trim().is_empty()
}

/// JSON stream events a CLI agent emits around its task without doing any
/// of it: startup (`system`/`init`, Codex `thread.started`/`turn.started`)
/// and its terminal status (`error`, `turn.failed`, `result`).
pub(super) const CLI_STATUS_EVENTS: [&str; 7] = [
    "system",
    "init",
    "thread.started",
    "turn.started",
    "error",
    "turn.failed",
    "result",
];

/// The phrases a CLI's own authentication error is written in. Read only
/// from what the CLI itself reported (its stderr and its JSON status
/// events), only on a non-zero exit, and only when the agent produced no
/// task output; an agent that writes "forbidden" while fixing a 403
/// handler has produced task output and is never judged by this.
pub(super) const CLI_AUTH_ERROR_PHRASES: [&str; 11] = [
    "authentication required",
    "authentication failed",
    "not authenticated",
    "not logged in",
    "login required",
    "invalid api key",
    "api key is required",
    "missing api key",
    "token expired",
    "unauthorized",
    "please run /login",
];

/// Whether a CLI agent run failed to authenticate, judged from what the
/// CLI reported about itself, never from its task output:
/// - Claude Code's stream-json marks a message whose API call was refused
///   for authentication with the typed `"error": "authentication_failed"`;
/// - Gemini CLI prints its own auth-help URL (`gemini-cli-auth-docs`) as a
///   plain line outside its JSON events, even with a zero exit;
/// - otherwise, on a non-zero exit with no task output at all, the CLI's
///   stderr or JSON error/result event states an auth error.
pub(super) fn cli_auth_failed(stdout: &str, display: &str, exit_code: Option<i32>) -> bool {
    let events: Vec<Option<Value>> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            serde_json::from_str::<Value>(line)
                .ok()
                .filter(Value::is_object)
        })
        .collect();
    if events
        .iter()
        .flatten()
        .any(|event| event.get("error").and_then(Value::as_str) == Some("authentication_failed"))
    {
        return true;
    }
    let stderr_lines = display
        .lines()
        .filter_map(|line| line.strip_prefix("[stderr] "));
    let plain_stdout = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| serde_json::from_str::<Value>(line).is_err());
    if stderr_lines
        .clone()
        .chain(plain_stdout)
        .any(|line| line.contains("gemini-cli-auth-docs"))
    {
        return true;
    }
    if !exit_code.is_some_and(|code| code != 0) {
        return false;
    }
    let status_event = |event: &Value| {
        event
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| CLI_STATUS_EVENTS.contains(&kind))
    };
    let produced_task_output = events
        .iter()
        .any(|event| !event.as_ref().is_some_and(status_event));
    if produced_task_output {
        return false;
    }
    let status_text = events
        .iter()
        .flatten()
        .filter(|event| status_event(event))
        .map(Value::to_string);
    stderr_lines
        .map(str::to_string)
        .chain(status_text)
        .any(|text| {
            let lower = text.to_lowercase();
            CLI_AUTH_ERROR_PHRASES
                .iter()
                .any(|phrase| lower.contains(phrase))
        })
}

pub(super) fn first_cli_failure_detail(output: &str) -> String {
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
