//! `shell`: run a command through a [`Sandbox`].

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use aidaemon_core::{
    AccessManifest, Tool, ToolAnnotations, ToolContext, ToolOutput, ToolResult, ToolSpec,
};

use crate::exec::{best_available_sandbox, Sandbox, SandboxCommand};
use crate::util::{fail, parse_args};
use crate::workspace::Workspace;

/// Host variables passed through by default.
pub const DEFAULT_ENV_ALLOWLIST: &[&str] = &[
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ", "TMPDIR",
];

/// Run shell commands in the workspace.
///
/// The tool is annotated destructive. It contains no approval logic: put an
/// approval [`aidaemon_core::Hook`] in front of it, and pick a contained
/// backend to limit what an approved command can reach.
///
/// Note that the agent's `Limits::tool_timeout` (120 s by default) also
/// bounds each call; raise it if you raise `max_timeout`.
#[derive(Clone)]
pub struct Shell {
    workspace: Workspace,
    backend: Arc<dyn Sandbox>,
    env_allowlist: Vec<String>,
    extra_env: Vec<(String, String)>,
    default_timeout: Duration,
    max_timeout: Duration,
    max_output_bytes: usize,
    allow_network: bool,
    deny_read: Vec<std::path::PathBuf>,
}

impl std::fmt::Debug for Shell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shell")
            .field("workspace", &self.workspace)
            .field("backend", &self.backend.name())
            .field("allow_network", &self.allow_network)
            .finish_non_exhaustive()
    }
}

impl Shell {
    pub fn new(workspace: Workspace, backend: Arc<dyn Sandbox>) -> Self {
        Self {
            workspace,
            backend,
            env_allowlist: DEFAULT_ENV_ALLOWLIST
                .iter()
                .map(|s| s.to_string())
                .collect(),
            extra_env: Vec::new(),
            default_timeout: Duration::from_secs(60),
            max_timeout: Duration::from_secs(110),
            max_output_bytes: 20_000,
            allow_network: false,
            deny_read: Vec::new(),
        }
    }

    /// Shell with [`best_available_sandbox`]: bubblewrap on Linux and
    /// seatbelt on macOS when available, overridable with
    /// [`crate::SANDBOX_ENV`]; uncontained (with a one-time warning) when no
    /// OS sandbox is available.
    pub fn sandboxed(workspace: Workspace) -> Self {
        Self::new(workspace, best_available_sandbox())
    }

    /// Replace the list of host environment variables passed through.
    pub fn env_allowlist(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.env_allowlist = names.into_iter().map(Into::into).collect();
        self
    }

    /// Set a fixed variable for every command (overrides the host value).
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_env.push((key.into(), value.into()));
        self
    }

    pub fn timeouts(mut self, default: Duration, max: Duration) -> Self {
        self.max_timeout = max.max(Duration::from_secs(1));
        self.default_timeout = default.min(self.max_timeout);
        self
    }

    pub fn max_output_bytes(mut self, bytes: usize) -> Self {
        self.max_output_bytes = bytes.max(64);
        self
    }

    /// Let sandboxed commands use the network (ignored by `LocalSandbox`,
    /// which never restricts it).
    pub fn allow_network(mut self, allow: bool) -> Self {
        self.allow_network = allow;
        self
    }

    /// Host paths commands must not read, for example credential
    /// directories (contained backends only; see
    /// [`SandboxCommand::deny_read`]).
    pub fn deny_read(
        mut self,
        paths: impl IntoIterator<Item = impl Into<std::path::PathBuf>>,
    ) -> Self {
        self.deny_read.extend(paths.into_iter().map(Into::into));
        self
    }

    pub fn backend(&self) -> &Arc<dyn Sandbox> {
        &self.backend
    }

    fn environment(&self) -> Vec<(String, String)> {
        let mut env: Vec<(String, String)> = self
            .env_allowlist
            .iter()
            .filter_map(|k| std::env::var(k).ok().map(|v| (k.clone(), v)))
            .collect();
        for (k, v) in &self.extra_env {
            env.retain(|(existing, _)| existing != k);
            env.push((k.clone(), v.clone()));
        }
        env
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellArgs {
    command: String,
    cwd: Option<String>,
    timeout_secs: Option<u64>,
}

#[async_trait]
impl Tool for Shell {
    /// The loop must allow at least the shell's own maximum plus time to
    /// kill the process group and collect output.
    fn timeout(&self) -> Option<Duration> {
        Some(self.max_timeout + Duration::from_secs(10))
    }

    fn spec(&self) -> ToolSpec {
        let containment = if self.backend.is_contained() {
            "Writes outside the workspace are blocked"
        } else {
            "Commands run directly on the host"
        };
        let network = if self.allow_network || !self.backend.is_contained() {
            "network available"
        } else {
            "no network access"
        };
        ToolSpec {
            name: "shell".into(),
            description: format!(
                "Run a non-interactive command with `sh -c` in the workspace (backend: {}). \
                 {containment}; {network}. stdin is closed, so do not run commands that wait \
                 for input or open editors/pagers. Returns combined stdout+stderr (long output \
                 keeps the start and end) and the exit code; a non-zero exit is reported as an \
                 error. Default timeout {}s, maximum {}s. Use read_file/grep/glob/edit_file \
                 instead of cat/grep/find/sed for files.",
                self.backend.name(),
                self.default_timeout.as_secs(),
                self.max_timeout.as_secs()
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run."},
                    "cwd": {"type": "string", "description": "Working directory, relative to the workspace root. Default: the root."},
                    "timeout_secs": {"type": "integer", "minimum": 1, "maximum": self.max_timeout.as_secs(), "description": "Kill the command after this many seconds."}
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::DESTRUCTIVE
    }

    fn validate(&self, arguments: &Value) -> Result<(), String> {
        match arguments.get("command").and_then(Value::as_str) {
            Some(c) if !c.trim().is_empty() => Ok(()),
            Some(_) => Err("`command` must not be empty".into()),
            None => Err("`command` is required and must be a string".into()),
        }
    }

    fn access(&self, arguments: &Value) -> AccessManifest {
        AccessManifest {
            commands: arguments
                .get("command")
                .and_then(Value::as_str)
                .map(|c| vec![c.to_string()])
                .unwrap_or_default(),
            ..Default::default()
        }
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        if let Err(e) = self.validate(&arguments) {
            return Ok(fail(e));
        }
        let args: ShellArgs = match parse_args("shell", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        let cwd = match args.cwd.as_deref().filter(|c| !c.trim().is_empty()) {
            Some(c) => match self.workspace.resolve(c) {
                Ok(p) if p.is_dir() => p,
                Ok(p) => {
                    return Ok(fail(format!(
                        "cwd `{}` is not an existing directory",
                        self.workspace.display(&p)
                    )))
                }
                Err(e) => return Ok(fail(e.to_string())),
            },
            None => self.workspace.root().to_path_buf(),
        };
        let timeout = args
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(self.default_timeout)
            .clamp(Duration::from_secs(1), self.max_timeout);

        let spec = SandboxCommand {
            command: args.command,
            cwd: cwd.clone(),
            workspace_root: self.workspace.root().to_path_buf(),
            env: self.environment(),
            timeout,
            max_output_bytes: self.max_output_bytes,
            allow_network: self.allow_network,
            deny_read: self.deny_read.clone(),
            cancel: ctx.cancel.clone(),
        };
        let out = match self.backend.run(&spec).await {
            Ok(o) => o,
            Err(e) => return Ok(fail(e.to_string())),
        };

        let mut content = out.output.clone();
        if content.trim().is_empty() {
            content = "(no output)".into();
        }
        if !content.ends_with('\n') {
            content.push('\n');
        }
        let status_line = if out.timed_out {
            format!(
                "[timed out after {}s; process killed. Use a longer timeout_secs or run less at once.]",
                timeout.as_secs()
            )
        } else if out.cancelled {
            "[cancelled; process killed]".to_string()
        } else {
            match out.exit_code {
                Some(code) => format!("[exit code: {code}]"),
                None => "[terminated by a signal]".to_string(),
            }
        };
        content.push_str(&status_line);
        let details = json!({
            "exit_code": out.exit_code,
            "timed_out": out.timed_out,
            "cancelled": out.cancelled,
            "duration_ms": out.duration.as_millis() as u64,
            "output_bytes": out.output_bytes,
            "truncated": out.truncated,
            "backend": self.backend.name(),
            "cwd": self.workspace.display(&cwd),
        });
        let output = if out.exit_code == Some(0) {
            ToolOutput::text(content)
        } else {
            ToolOutput::error(content)
        };
        Ok(output.with_details(details))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::LocalSandbox;
    use crate::testutil::{ctx, ws};

    fn shell(ws: &Workspace) -> Shell {
        Shell::new(ws.clone(), Arc::new(LocalSandbox))
    }

    #[tokio::test]
    async fn reports_exit_codes() {
        let (_d, ws) = ws();
        let tool = shell(&ws);
        let out = tool
            .call(json!({"command": "echo hello; echo oops >&2"}), &ctx())
            .await
            .unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.contains("hello") && out.content.contains("oops"));
        assert!(out.content.ends_with("[exit code: 0]"));
        assert_eq!(out.details["exit_code"], 0);
        assert_eq!(out.details["backend"], "local");

        let out = tool
            .call(json!({"command": "echo failing; exit 3"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error);
        assert_eq!(out.details["exit_code"], 3);
        assert!(out.content.contains("[exit code: 3]"));
    }

    #[tokio::test]
    async fn runs_in_cwd_and_refuses_escapes() {
        let (_d, ws) = ws();
        std::fs::create_dir(ws.root().join("sub")).unwrap();
        let tool = shell(&ws);
        let out = tool
            .call(json!({"command": "pwd", "cwd": "sub"}), &ctx())
            .await
            .unwrap();
        assert!(out
            .content
            .trim_end()
            .contains(&ws.root().join("sub").display().to_string()));
        let out = tool
            .call(json!({"command": "pwd", "cwd": "../"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("outside the workspace"));
        let out = tool
            .call(json!({"command": "pwd", "cwd": "missing"}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("not an existing directory"));
    }

    #[tokio::test]
    async fn times_out_and_kills() {
        let (_d, ws) = ws();
        let tool = shell(&ws);
        let started = std::time::Instant::now();
        let out = tool
            .call(
                json!({"command": "echo start; sleep 30; echo never", "timeout_secs": 1}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(out.is_error);
        assert_eq!(out.details["timed_out"], true);
        assert!(out.content.contains("start") && !out.content.contains("never"));
        assert!(out.content.contains("timed out after 1s"));
    }

    #[tokio::test]
    async fn caps_output_head_and_tail() {
        let (_d, ws) = ws();
        let tool = shell(&ws).max_output_bytes(1000);
        let out = tool
            .call(
                json!({"command": "echo FIRST; i=0; while [ $i -lt 5000 ]; do echo line$i; i=$((i+1)); done; echo LAST"}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.starts_with("FIRST\n"));
        assert!(out.content.contains("LAST\n[exit code: 0]"));
        assert!(out.content.contains("bytes of output omitted"));
        assert!(out.content.len() < 1200, "{}", out.content.len());
        assert_eq!(out.details["truncated"], true);
    }

    #[tokio::test]
    async fn environment_is_allowlisted() {
        let (_d, ws) = ws();
        std::env::set_var("AIDAEMON_TOOLS_TEST_SECRET", "synthetic-secret");
        let tool = shell(&ws).env("GREETING", "hi");
        let out = tool
            .call(
                json!({"command": "echo \"[$AIDAEMON_TOOLS_TEST_SECRET][$GREETING]\""}),
                &ctx(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("[][hi]"), "{}", out.content);
    }

    #[test]
    fn declares_destructive_and_command() {
        let (_d, ws) = ws();
        let tool = shell(&ws);
        assert!(tool.annotations().destructive);
        assert_eq!(
            tool.access(&json!({"command": "ls -la"})).commands,
            vec!["ls -la"]
        );
        assert!(tool.validate(&json!({"command": "  "})).is_err());
        assert_eq!(tool.spec().parameters["additionalProperties"], json!(false));
    }
}
