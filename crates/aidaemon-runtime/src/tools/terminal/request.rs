//! A call's arguments, trust and run request.

use super::*;

/// A background command as the owner's list of running work names it: by
/// the title its call gave it, else by its program ("a shell loop",
/// "ddev"), never by its raw line.
pub(super) fn background_job_title(command: &str, title: Option<&str>) -> String {
    let what = title
        .map(str::to_string)
        .unwrap_or_else(|| aidaemon_tools::command_effects::program_label(command));
    format!("Background command: {what}")
}

/// Most characters of a command's plain title.
pub(super) const CALL_TITLE_CHARS: usize = 60;

/// The plain title a call gave its command, cleaned for the owner's eyes
/// (secrets and home paths hidden, one line, capped).
pub(super) fn call_title(args: &TerminalArgs) -> Option<String> {
    let title = args.description.as_deref()?;
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    if title.is_empty() {
        return None;
    }
    let title = aidaemon_tools::sanitize::redact_secrets(
        &aidaemon_tools::sanitize::shorten_home_dir(&title),
    );
    Some(truncate_str(&title, CALL_TITLE_CHARS).to_string())
}

/// Whether a background job's completion notification reaches a reader: the
/// call runs in the session the owner talks in, or the notification is
/// routed to the goal's origin session. A sub-agent's own session (its grants
/// are its parent chat's) has neither.
pub(super) fn notifications_reach(exec_ctx: &ToolCallContext, notify_session_id: &str) -> bool {
    let delegated =
        !exec_ctx.grant_session.is_empty() && exec_ctx.grant_session != exec_ctx.session_id;
    !delegated || notify_session_id != exec_ctx.session_id
}

pub(super) struct TerminalRunRequest<'a> {
    pub(super) command: &'a str,
    /// The plain title the call gave the command (`description`).
    pub(super) title: Option<&'a str>,
    pub(super) script_via_stdin: bool,
    pub(super) working_dir: Option<&'a str>,
    pub(super) read_paths: &'a [String],
    pub(super) write_paths: &'a [String],
    pub(super) write_roots: &'a [String],
    pub(super) notify_session_id: &'a str,
    pub(super) notify_goal_id: Option<&'a str>,
    /// A completion notification for `notify_session_id` reaches someone
    /// waiting on it. A sub-agent's own session has no channel and its
    /// parent waits for the run's reply, not for the session: a job it
    /// moved to the background must not end its run (live 2026-11-12: two
    /// DDEV children ended as "still on it, I'll send the result" and the
    /// result went to a session nobody reads).
    pub(super) notifications_reach: bool,
    pub(super) task_id: Option<&'a str>,
    pub(super) tool_call_id: Option<&'a str>,
    pub(super) detach: bool,
    /// Finite work the model put in the background: it waits only for the
    /// startup check, then hands off like any job that outlived the
    /// foreground wait.
    pub(super) background: bool,
    /// Outbound network granted for this run.
    pub(super) network: bool,
    /// Sandbox this run (false = host execution).
    pub(super) confined: bool,
    pub(super) status_tx: Option<mpsc::Sender<StatusUpdate>>,
    /// The trust of this run's output under the turn's roots, recorded
    /// with the process if it moves to the background.
    pub(super) launch_trust: aidaemon_core::SourceTrust,
}

/// Whether a run is confined: a single configuration switch. Commands from
/// untrusted external triggers are not sandboxed by exception; they are
/// protected by the mandatory per-command approval that path already
/// enforces.
pub(super) fn run_is_confined(confinement: aidaemon_tools::TerminalConfinement) -> bool {
    confinement == aidaemon_tools::TerminalConfinement::Sandbox
}

/// Host execution request: the command as the daemon's user, in the resolved
/// working directory, with the daemon's own environment.
/// The shell a multi-line script runs in, the script arriving on stdin (so
/// it never shows in the process list, where any local user could read it).
/// The shell reads the whole script first, closes stdin, then runs it: fed
/// straight to `sh -s`, any command in it that reads input (ddev, drush, wp,
/// mysql, ssh, `docker exec -i`, `read`) consumed the rest of the script,
/// which then never ran while the run still exited 0 (live 2026-11-14 to
/// 11-20: 23 of 148 stdin-fed scripts had such a command before their last
/// line; Drupal installs needed reinstalls). With stdin closed such a
/// command reads end of input, as in a terminal with nothing typed. `-e`
/// still stops at the first failing step inside `eval`.
pub(super) fn script_shell_argv() -> Vec<String> {
    vec![
        "/bin/sh".to_string(),
        "-eu".to_string(),
        "-c".to_string(),
        SCRIPT_BOOTSTRAP.to_string(),
    ]
}

/// Read the whole script from stdin, close stdin, run the script.
pub(super) const SCRIPT_BOOTSTRAP: &str =
    "__aidaemon_script=$(cat); exec </dev/null; eval \"$__aidaemon_script\"";

pub(super) async fn host_terminal_execution_request(
    backend: &SharedExecutionBackend,
    shell_source: &str,
    script_via_stdin: bool,
    working_dir: Option<&str>,
) -> anyhow::Result<ExecutionRequest> {
    let cwd = match working_dir {
        Some(path) => Some(backend.resolve_path(path).await?),
        None => None,
    };
    let mut request = if script_via_stdin && cfg!(windows) {
        // Windows PowerShell reads the script from stdin; the first failing
        // step stops the rest, as `sh -eu` does.
        let mut request = ExecutionRequest::argv(
            "powershell.exe".to_string(),
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "-",
            ]
            .map(str::to_string)
            .to_vec(),
        );
        request.stdin = Some(
            format!(
                "$ErrorActionPreference = 'Stop'\n{shell_source}{}",
                aidaemon_tools::platform_shell::POWERSHELL_EXIT_EPILOGUE
            )
            .into_bytes(),
        );
        request
    } else if script_via_stdin {
        let mut argv = script_shell_argv();
        let program = argv.remove(0);
        let mut request = ExecutionRequest::argv(program, argv);
        request.stdin = Some(shell_source.as_bytes().to_vec());
        request
    } else {
        ExecutionRequest::shell(shell_source.to_string())
    };
    request.cwd = cwd;
    Ok(request)
}

/// Grant keys for a command: each segment's binary for chained commands,
/// otherwise the first word. Storing segment binaries (rather than the full
/// chained string) lets "Allow Always" cover re-runs that differ only in
/// arguments — the same trust grant as Always-allowing each simple command
/// directly, and what `is_allowed`'s per-segment chained check matches
/// against.
pub(super) fn command_grant_keys(command: &str) -> Vec<String> {
    let trimmed = command.trim();
    if contains_shell_operator(trimmed) {
        split_command_segments(trimmed)
            .iter()
            .map(|seg| extract_segment_binary(seg))
            .filter(|b| !b.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        vec![
            trimmed
                .split_whitespace()
                .next()
                .unwrap_or(trimmed)
                .to_string(),
        ]
    }
}

impl Drop for TerminalTool {
    fn drop(&mut self) {
        // Best-effort kill of all tracked background processes.
        if let Ok(running) = self.running.try_lock() {
            for proc in running.values() {
                if let Some(pid) = proc.process_handle.local_pid() {
                    send_sigterm(pid);
                    send_sigkill(pid);
                }
            }
        }
    }
}

#[derive(Deserialize)]
pub(super) struct TerminalArgs {
    pub(super) command: Option<String>,
    /// A few plain words naming what the command does, for the owner's
    /// status line and list of running work. Never authority: it changes
    /// nothing about how the call is judged.
    #[serde(default)]
    pub(super) description: Option<String>,
    pub(super) script: Option<String>,
    #[serde(alias = "cwd")]
    pub(super) working_dir: Option<String>,
    #[serde(default)]
    pub(super) read_paths: Vec<String>,
    #[serde(default)]
    pub(super) write_paths: Vec<String>,
    /// Typed directory capabilities. The dispatcher may project an
    /// authorized task root here so future descendants can be created before
    /// the process starts without widening an exact-file grant.
    #[serde(default)]
    pub(super) read_roots: Vec<String>,
    #[serde(default)]
    pub(super) write_roots: Vec<String>,
    #[serde(default = "default_action")]
    pub(super) action: String,
    pub(super) pid: Option<u32>,
    /// If true, allow a timed-out command to outlive task boundaries.
    /// Default false: timed-out background commands are task-owned and auto-cleaned
    /// when the task ends.
    #[serde(default)]
    pub(super) detach: bool,
    /// Run finite work in the background: after the startup check the
    /// command is a task job that reports back when it ends, and the run
    /// hands off to it. Not a long-lived service (that is `detach`).
    #[serde(default)]
    pub(super) background: bool,
    /// Outbound network for this confined run. Off by default; requesting it
    /// is a capability escalation that needs user approval or a persisted
    /// capability grant for the command.
    #[serde(default)]
    pub(super) network: bool,
}

/// The turn's facts the terminal's approval ladder and hard blocks read:
/// from the dispatcher's typed [`ToolCallContext`], or from the
/// kernel's per-turn approval rules. Never from the call's arguments.
#[derive(Debug, Clone, Default)]
pub(crate) struct TerminalTrust {
    pub session_id: String,
    /// The chat session whose "allow for session" grants cover this call
    /// (a child agent works for the chat that started it).
    pub grant_session: String,
    /// The turn's user is the owner (an unknown caller is not).
    pub owner: bool,
    /// A trusted scheduled session: runs without asking, never detached.
    pub trusted_session: bool,
    /// Started by an automated trigger: every run asks.
    pub untrusted_source: bool,
    /// Out-of-scope targets routed to the owner (a confined capability).
    pub scope_escalation: Option<crate::tools::traits::ScopeEscalation>,
}

impl TerminalTrust {
    /// The facts the kernel's approval rules judge a call with: exactly
    /// the turn's, the escalation as the dispatcher judged it.
    pub(super) fn for_approval(ctx: &ToolCallContext) -> Self {
        Self {
            session_id: ctx.session_id.clone(),
            grant_session: grant_session_of(&ctx.session_id, ctx),
            owner: ctx.user_role == Some(aidaemon_tools::semantics::UserRole::Owner),
            trusted_session: ctx.trusted_session,
            untrusted_source: ctx.untrusted_source,
            scope_escalation: ctx.scope_escalation.clone(),
        }
    }

    pub(super) fn from_execution_context(session_id: &str, ctx: &ToolCallContext) -> Self {
        Self {
            session_id: session_id.to_string(),
            grant_session: grant_session_of(session_id, ctx),
            owner: ctx.user_role == Some(aidaemon_tools::semantics::UserRole::Owner),
            trusted_session: ctx.trusted_session,
            untrusted_source: ctx.untrusted_source,
            scope_escalation: ctx.scope_escalation.clone().filter(|e| !e.is_empty()),
        }
    }
}

/// The chat session a call's "allow for session" grants are kept under:
/// the dispatcher's grant session, else the call's own session.
fn grant_session_of(session_id: &str, ctx: &ToolCallContext) -> String {
    let grant = ctx.grant_session.trim();
    if grant.is_empty() {
        session_id.to_string()
    } else {
        grant.to_string()
    }
}

/// The key an "allow for session" on a simple command is kept under: the
/// program and, when the next word is not an option, its subcommand or
/// first operand (`git status`, `npm run`, `python3 tool.py`), so allowing
/// `git status` never covers `git push`.
pub(super) fn session_grant_key(command: &str) -> String {
    let trimmed = command.trim();
    let words = shell_words::split(trimmed).unwrap_or_else(|_| {
        trimmed
            .split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    match words.as_slice() {
        [] => trimmed.to_string(),
        [program] => program.clone(),
        [program, next, ..] if next.starts_with('-') => program.clone(),
        [program, next, ..] => format!("{program} {next}"),
    }
}

/// Whether `command`'s words start with the words of a session grant key.
pub(super) fn covered_by_session_key(command: &str, key: &str) -> bool {
    let split = |text: &str| {
        shell_words::split(text.trim()).unwrap_or_else(|_| {
            text.split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
    };
    let (words, key) = (split(command), split(key));
    !key.is_empty() && words.len() >= key.len() && words[..key.len()] == key[..]
}

/// What the owner is asked to approve before a run.
pub(super) struct LadderAsk {
    pub(super) risk: RiskLevel,
    pub(super) warnings: Vec<String>,
}

pub(super) fn default_action() -> String {
    "run".to_string()
}

pub(super) fn extract_terminal_exit_code(output: &str) -> Option<i32> {
    let marker = "[exit code:";
    let start = output.rfind(marker)?;
    let rest = output[start + marker.len()..].trim_start();
    let code_token: String = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
        .collect();
    if code_token.is_empty() {
        None
    } else {
        code_token.parse::<i32>().ok()
    }
}

/// The errno text the confining sandbox's denial produces on this platform:
/// seatbelt (macOS) refuses an open with `EPERM` ("Operation not
/// permitted"); the Linux adapters refuse with `EACCES` ("Permission
/// denied", Landlock) or `EPERM` (bubblewrap/seccomp).
#[cfg(target_os = "macos")]
pub(super) const SANDBOX_DENIAL_ERRNO_TEXT: &[&str] = &["operation not permitted"];
#[cfg(not(target_os = "macos"))]
pub(super) const SANDBOX_DENIAL_ERRNO_TEXT: &[&str] =
    &["operation not permitted", "permission denied"];

/// Whether a failed confined process's stderr shows the sandbox refusing a
/// path. Neither sandbox reports a violation to the process in a structured
/// way (seatbelt logs it to the unified log only), so the process's own
/// errno line is the only signal, and it is read as narrowly as it can be:
/// only for a process that ran confined, only on a non-zero exit, only the
/// platform's sandbox errno text, and only on a line that names an
/// absolute path the open was refused for. A program's own refusal with no
/// path (`git push`'s "Permission denied (publickey)") is never a policy
/// denial.
pub(super) fn sandbox_access_denial(
    stderr: &str,
    exit_code: Option<i32>,
    confined: bool,
) -> Option<crate::tools::traits::ToolAccessDenial> {
    let denied = confined
        && exit_code.is_some_and(|code| code != 0)
        && stderr.lines().any(|line| {
            let lower = line.trim().to_ascii_lowercase();
            SANDBOX_DENIAL_ERRNO_TEXT
                .iter()
                .any(|errno| lower.contains(errno))
                && !extract_absolute_paths(line).is_empty()
        });
    denied.then(|| crate::tools::traits::ToolAccessDenial {
        reason_code: "sandbox_policy_denied".to_string(),
        enforcement: confined_process_access_enforcement(),
        exit_code,
        proposed_evidence: Vec::new(),
    })
}

/// Semantics of an invocation whose process runs independently of the call:
/// a typed process-state change on top of whatever the command was declared
/// to observe or write. Shared by the provisional launch receipt and the
/// terminal completion receipt of the same invocation so the effect the
/// launch declared is exactly the effect the completion proves.
pub(super) fn background_process_semantics(
    command_semantics: &ToolCallSemantics,
) -> ToolCallSemantics {
    ToolCallSemantics::observation_and_mutation_with(
        command_semantics
            .mutation_effects
            .union(ToolMutationEffects::PROCESS_STATE),
    )
    .with_verification_mode(ToolVerificationMode::ResultContent)
}
