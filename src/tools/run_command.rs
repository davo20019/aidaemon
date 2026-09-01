use async_trait::async_trait;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

use crate::execution::active_execution_backend;
use crate::traits::{
    Tool, ToolCallAccessManifest, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolMutationEffects, ToolOutcomeStatus, ToolRole, ToolTargetHint,
    ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::StatusUpdate;

use super::daemon_guard::detect_daemonization_primitives;
use super::fs_utils;

pub struct RunCommandTool;
const SAFE_NPM_PREFIX_HINT: &str =
    "`npm test`, `npm ls`, `npm outdated`, `npm audit` (note: `npm run` and `npx` require `terminal` approval)";

/// Safe command prefixes that don't require terminal approval flow.
///
/// Anything that executes arbitrary repo-defined scripts (Makefiles, npm
/// `run`/`exec`, `cargo run`, `cargo bench`, `gradle`/`mvn` build files,
/// `go generate`, `npx`-downloaded packages) is intentionally OMITTED. Such
/// commands can run unbounded code from the working tree or the network and
/// must go through the terminal approval flow instead.
const SAFE_PREFIXES: &[&str] = &[
    // Build & test — only bounded sub-commands. `cargo run`, `cargo bench`,
    // `go generate`, `npm run`/`npx`/`yarn run`/`bun run`,
    // `make`/`cmake`/`gradle`/`mvn` are deliberately excluded — they execute
    // arbitrary repo-defined code.
    "cargo build",
    "cargo test",
    "cargo check",
    "cargo clippy",
    "cargo fmt",
    "cargo doc",
    "cargo tree",
    "cargo metadata",
    "npm test",
    "npm ls",
    "npm outdated",
    "npm audit",
    "yarn test",
    "yarn lint",
    "bun test",
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "go test",
    "go build",
    "go vet",
    "go mod",
    "jest",
    "vitest",
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
    // Pure output/status observations. Absolute paths are listed explicitly so
    // an observational predicate such as `/usr/bin/false` is not mislabeled as
    // a mutation or forced through an approval-only arbitrary shell path.
    "printf",
    "/usr/bin/printf",
    "/bin/printf",
    "echo",
    "/bin/echo",
    "true",
    "/usr/bin/true",
    "/bin/true",
    "false",
    "/usr/bin/false",
    "/bin/false",
];

fn run_command_semantics(arguments: &str) -> ToolCallSemantics {
    let workspace_write = serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("access")
                .and_then(Value::as_str)
                .map(|access| access == "workspace_write")
        })
        .unwrap_or(false);
    if workspace_write {
        ToolCallSemantics::observation_and_mutation_with(ToolMutationEffects::LOCAL_DERIVED_WRITE)
            .with_verification_mode(ToolVerificationMode::ResultContent)
    } else {
        ToolCallSemantics::observation().with_verification_mode(ToolVerificationMode::ResultContent)
    }
}

fn run_command_access_manifest(arguments: &str) -> ToolCallAccessManifest {
    let parsed = serde_json::from_str::<Value>(arguments).ok();
    let execution_cwd = parsed
        .as_ref()
        .and_then(|value| value.get("working_dir"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let cwd_target = execution_cwd
        .as_deref()
        .and_then(|cwd| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, cwd));
    let semantics = run_command_semantics(arguments);
    ToolCallAccessManifest {
        execution_cwd,
        read_targets: cwd_target.iter().cloned().collect(),
        // Build/test/format commands legitimately create derived output in
        // their cwd. Observational commands never inherit that write grant.
        write_targets: if semantics.mutates_state() {
            cwd_target.into_iter().collect()
        } else {
            Vec::new()
        },
        adapter_read_targets: Vec::new(),
    }
}

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
            "description": "Run allowlisted build, test, lint, and read-only inspection commands. Repository-defined scripts, installs, and arbitrary commands require terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run (must match a safe prefix)"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: configured execution workspace root)"
                    },
                    "access": {
                        "type": "string",
                        "enum": ["read_only", "workspace_write"],
                        "description": "Capability enforced by the process sandbox. Use read_only for inspection and predicates; use workspace_write only when the command must create derived files in working_dir."
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
                "required": ["command", "access"],
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

    fn receipt_kind(&self, _arguments: &str) -> crate::traits::ToolReceiptKind {
        crate::traits::ToolReceiptKind::Process
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        run_command_semantics(arguments)
    }

    fn call_access_manifest(&self, arguments: &str) -> ToolCallAccessManifest {
        run_command_access_manifest(arguments)
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(self.run_outcome(arguments).await?.output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.run_outcome(arguments).await
    }
}

impl RunCommandTool {
    async fn run_outcome(&self, arguments: &str) -> anyhow::Result<ToolCallOutcome> {
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
            return Ok(routing_rejection(
                "Shell operators (;, |, &&, ||, $(), etc.) are not allowed in run_command. Use 'terminal' for complex commands.".to_string(),
                "terminal",
            ));
        }

        // Validate against safe prefixes
        if !is_safe_command(trimmed) {
            let mut preview: String = trimmed.chars().take(140).collect();
            if trimmed.chars().count() > 140 {
                preview.push('…');
            }
            return Ok(routing_rejection(
                format!(
                    "Command '{}' is not in the safe command list for run_command. Use 'terminal' for this command.\n\nAllowed npm prefixes in run_command: {}.\nFor installs (e.g. `npm install`), use `terminal`.",
                    preview, SAFE_NPM_PREFIX_HINT
                ),
                "terminal",
            ));
        }

        let daemon_hits = detect_daemonization_primitives(trimmed);
        if !daemon_hits.is_empty() {
            return Ok(routing_rejection(
                format!(
                    "Daemonization primitives are blocked in run_command ({}). Use terminal and explicit owner approval if detached/background execution is truly needed.",
                    daemon_hits.join(", ")
                ),
                "terminal",
            ));
        }

        let backend = active_execution_backend();
        let dir = if let Some(d) = working_dir {
            backend.resolve_path(d).await?
        } else {
            backend.workspace_root().clone()
        };

        // Fail early with an actionable workspace error. Letting a Git command
        // run from an accidental daemon launch directory (frequently `/` for a
        // service) produces a misleading generic exit 128 and invites the model
        // to guess at repository state.
        let is_git_command = trimmed == "git" || trimmed.starts_with("git ");
        let mut adapter_read_paths = Vec::new();
        if is_git_command {
            let preflight =
                fs_utils::run_cmd_backend("git rev-parse --show-toplevel", Some(&dir), 5).await?;
            if preflight.exit_code != 0 {
                anyhow::bail!(
                    "Git preflight failed: execution directory '{}' is not inside a repository. Configure execution.workspace_root or pass working_dir explicitly.",
                    dir
                );
            }
            let metadata = fs_utils::run_cmd_backend(
                "git rev-parse --path-format=absolute --git-dir --git-common-dir",
                Some(&dir),
                5,
            )
            .await?;
            if metadata.exit_code != 0 {
                anyhow::bail!(
                    "Git metadata discovery failed for execution directory '{}'.",
                    dir
                );
            }
            for path in metadata.stdout.lines().map(str::trim) {
                if path.starts_with('/') && !adapter_read_paths.iter().any(|seen| seen == path) {
                    adapter_read_paths.push(path.to_string());
                }
            }
        }

        // `/usr/bin/git` is an xcrun shim on macOS. Resolving the concrete
        // developer-tool executable before confinement prevents the shim from
        // attempting a cache write during a read-only observation. This is an
        // adapter/environment normalization, not a write grant.
        #[cfg(target_os = "macos")]
        let executable_command = if is_git_command {
            let resolved = fs_utils::run_cmd_backend("xcrun --find git", Some(&dir), 5).await?;
            if resolved.exit_code != 0 {
                anyhow::bail!("Could not resolve the macOS Git executable before confinement.");
            }
            let git_path = resolved.stdout.trim();
            if !git_path.starts_with('/') || git_path.contains(['\n', '\r']) {
                anyhow::bail!("macOS returned an invalid Git executable path.");
            }
            format!(
                "{}{}",
                quote_shell_word(git_path),
                trimmed.strip_prefix("git").unwrap_or_default()
            )
        } else {
            trimmed.to_string()
        };
        #[cfg(not(target_os = "macos"))]
        let executable_command = trimmed.to_string();

        let semantics = self.call_semantics(arguments);
        let started = Instant::now();
        let write_paths = semantics.mutates_state().then(|| vec![dir.to_string()]);
        let mut request = crate::tools::terminal::confined_terminal_execution_request(
            &backend,
            &executable_command,
            Some(dir.as_str()),
            &adapter_read_paths,
            write_paths.as_deref().unwrap_or(&[]),
        )
        .await?;
        if !semantics.mutates_state() {
            // Git inspection ordinarily refreshes its index opportunistically.
            // Disable optional locks for every observation capability so the
            // process remains compatible with the enforced read-only sandbox.
            request
                .env
                .insert("GIT_OPTIONAL_LOCKS".to_string(), "0".to_string());
            if is_git_command {
                request
                    .env
                    .insert("GIT_CONFIG_GLOBAL".to_string(), "/dev/null".to_string());
                request
                    .env
                    .insert("GIT_CONFIG_NOSYSTEM".to_string(), "1".to_string());
            }
        }
        let execution = backend
            .execute(request, Duration::from_secs(timeout))
            .await?;
        if execution.timed_out {
            anyhow::bail!("Command timed out after {}s", timeout);
        }
        let result = fs_utils::CommandOutput {
            exit_code: execution.exit_code,
            stdout: execution.stdout_lossy(),
            stderr: execution.stderr_lossy(),
            duration_ms: started.elapsed().as_millis() as u64,
        };
        let output = format_output(&result, trimmed, parse_format)?;
        let mut actual_access_manifest = self.call_access_manifest(arguments);
        for path in adapter_read_paths {
            if let Some(target) = ToolTargetHint::new(ToolTargetHintKind::Path, path) {
                if !actual_access_manifest.read_targets.contains(&target) {
                    actual_access_manifest.read_targets.push(target);
                }
            }
        }
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                // A normally completed process produces an authoritative
                // observation even when its predicate/test was false. A
                // negative backend sentinel means there was no normal exit and
                // therefore is an execution failure, not a negative result.
                outcome_status: Some(ToolOutcomeStatus::from_process_exit_code(result.exit_code)),
                exit_code: Some(result.exit_code),
                access_manifest: Some(actual_access_manifest),
                access_enforcement: crate::tools::terminal::confined_process_access_enforcement(),
                ..ToolCallMetadata::default()
            },
        })
    }
}

/// A pre-dispatch rejection that routes the model to a different tool. It
/// never ran a command, so it is typed as a blocked, rejected-before-dispatch
/// observation — not a failed mutation. Typing it this way keeps a routing
/// hint out of the external-mutation ledger, so a later successful `terminal`
/// retry is the run's outcome instead of this superseded rejection.
fn routing_rejection(message: String, route_to: &str) -> ToolCallOutcome {
    ToolCallOutcome {
        output: message,
        metadata: ToolCallMetadata {
            outcome_status: Some(ToolOutcomeStatus::Blocked),
            invocation_stage: crate::traits::ToolInvocationStage::RejectedBeforeDispatch,
            contract_rejected: true,
            effective_tool_name: Some(route_to.to_string()),
            semantics: ToolCallSemantics::observation(),
            access_enforcement: crate::traits::ToolAccessEnforcement::ControllerEnforced,
            ..ToolCallMetadata::default()
        },
    }
}

#[cfg(target_os = "macos")]
fn quote_shell_word(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn is_safe_command(cmd: &str) -> bool {
    is_run_command_safe(cmd)
}

/// Public (crate-visible) alias used by the correction-sandbox classifier.
/// Returns true if `cmd` matches one of run_command's safe prefixes.
pub(crate) fn is_run_command_safe(cmd: &str) -> bool {
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
        let front_end = crate::utils::floor_char_boundary(s, half);
        let back_start = crate::utils::floor_char_boundary(s, s.len() - half);
        format!(
            "{}\n\n... ({} chars truncated) ...\n\n{}",
            &s[..front_end],
            s.len() - max_chars,
            &s[back_start..]
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
    fn workspace_write_process_keeps_result_observation_semantics() {
        let semantics = run_command_semantics(r#"{"access":"workspace_write"}"#);
        assert!(semantics.observes_state());
        assert!(semantics.mutates_state());
        assert_eq!(
            semantics.verification_mode,
            ToolVerificationMode::ResultContent
        );
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
        for command in [
            "printf synthetic-output",
            "/usr/bin/printf synthetic-output",
            "true",
            "/usr/bin/false",
        ] {
            assert!(is_safe_command(command), "pure observation: {command}");
        }

        // Unsafe
        assert!(!is_safe_command("rm -rf /"));
        assert!(!is_safe_command("curl http://evil.com"));
        assert!(!is_safe_command("git push"));
        assert!(!is_safe_command("git reset --hard"));
        assert!(!is_safe_command("sudo apt install"));
        assert!(!is_safe_command("chmod 777 /etc"));

        // Regression: arbitrary-code prefixes are NOT safe and must go
        // through `terminal` instead. These wrap repo-defined scripts or
        // network-downloaded packages.
        assert!(
            !is_safe_command("cargo run"),
            "cargo run executes arbitrary main.rs and must require approval"
        );
        assert!(!is_safe_command("cargo run --release -- --evil"));
        assert!(
            !is_safe_command("cargo bench"),
            "cargo bench executes arbitrary bench harnesses"
        );
        assert!(
            !is_safe_command("npm run build"),
            "npm run executes arbitrary package.json scripts"
        );
        assert!(!is_safe_command("npm run anything-here"));
        assert!(
            !is_safe_command("npx some-package"),
            "npx downloads and runs arbitrary packages"
        );
        assert!(!is_safe_command("yarn run dev"));
        assert!(!is_safe_command("bun run start"));
        assert!(
            !is_safe_command("go generate ./..."),
            "go generate executes arbitrary commands from source comments"
        );
        assert!(
            !is_safe_command("make"),
            "make executes arbitrary Makefiles"
        );
        assert!(!is_safe_command("make install"));
        assert!(!is_safe_command("cmake --build ."));
        assert!(!is_safe_command("gradle build"));
        assert!(!is_safe_command("mvn package"));
    }

    #[tokio::test]
    async fn test_run_safe_command() {
        let args = json!({"command": "ls"}).to_string();
        let result = RunCommandTool.call(&args).await.unwrap();
        assert!(result.contains("exit: 0"));
    }

    #[tokio::test]
    async fn completed_negative_command_is_a_typed_observation() {
        let args = json!({"command": "/usr/bin/false"}).to_string();
        let outcome = RunCommandTool
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::CompletedWithNegativeResult)
        );
        assert_eq!(outcome.metadata.exit_code, Some(1));
        assert_eq!(
            outcome.metadata.access_enforcement,
            crate::tools::terminal::confined_process_access_enforcement()
        );
        assert!(outcome.output.contains("exit: 1"));
    }

    #[tokio::test]
    async fn git_command_uses_the_selected_execution_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let initialized = std::process::Command::new("git")
            .arg("init")
            .arg("--quiet")
            .arg(dir.path())
            .status()
            .unwrap();
        assert!(initialized.success());
        let args = json!({
            "command": "git status --short --branch",
            "working_dir": dir.path().to_str().unwrap()
        })
        .to_string();
        let outcome = RunCommandTool
            .call_with_status_outcome(&args, None)
            .await
            .unwrap();
        assert!(outcome.output.contains("exit: 0"), "{}", outcome.output,);
        assert!(outcome.output.contains("## "));
    }

    #[tokio::test]
    async fn git_command_preflight_explains_a_non_repository_directory() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({
            "command": "git status --short --branch",
            "working_dir": dir.path().to_str().unwrap()
        })
        .to_string();

        let error = RunCommandTool.call(&args).await.unwrap_err().to_string();
        assert!(error.contains("Git preflight failed"));
        assert!(error.contains("execution.workspace_root"));
    }

    #[tokio::test]
    async fn test_run_unsafe_command_rejected() {
        let args = json!({"command": "rm -rf /"}).to_string();
        // A routing rejection is a typed outcome, not an error, so a later
        // successful terminal retry can supersede it.
        let out = RunCommandTool.call(&args).await.unwrap();
        assert!(out.contains("not in the safe command list"));
    }

    #[tokio::test]
    async fn test_run_npm_install_rejected_with_actionable_guidance() {
        let args = json!({"command": "npm install tailwindcss"}).to_string();
        let out = RunCommandTool.call(&args).await.unwrap();
        assert!(out.contains("not in the safe command list for run_command"));
        assert!(out.contains("Allowed npm prefixes"));
        assert!(out.contains("npm test"));
        assert!(out.contains("use `terminal`"));
    }

    #[tokio::test]
    async fn test_run_shell_operator_rejected() {
        let args = json!({"command": "ls | grep foo"}).to_string();
        let out = RunCommandTool.call(&args).await.unwrap();
        assert!(out.contains("Shell operators"));
    }

    #[tokio::test]
    async fn test_run_daemonization_rejected() {
        let args = json!({"command": "cargo test &"}).to_string();
        let out = RunCommandTool.call(&args).await.unwrap();
        assert!(out.contains("Daemonization primitives"));
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

    #[tokio::test]
    async fn npm_run_routing_rejection_is_a_non_mutating_pre_dispatch_block() {
        // `npm run build` is not on the safe list; the rejection routes to
        // terminal and must not read as a dispatched, failed mutation — else
        // a later successful terminal retry is overwritten by this rejection.
        let outcome = RunCommandTool
            .run_outcome(&serde_json::json!({ "command": "npm run build" }).to_string())
            .await
            .expect("routing rejection is a typed outcome, not an error");
        let meta = &outcome.metadata;
        assert_eq!(meta.outcome_status, Some(ToolOutcomeStatus::Blocked));
        assert_eq!(
            meta.invocation_stage,
            crate::traits::ToolInvocationStage::RejectedBeforeDispatch
        );
        assert!(meta.contract_rejected);
        assert_eq!(meta.effective_tool_name.as_deref(), Some("terminal"));
        assert!(!meta.semantics.mutates_state());
        assert!(outcome.output.contains("Use 'terminal'"));
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
