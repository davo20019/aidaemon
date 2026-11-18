//! The contract every coding agent implements, and what a launch is made of.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader};
use tokio::process::ChildStdin;

use crate::{ActionJudge, PermissionMode};

/// A coding agent the embedding agent can hand work to.
pub trait CodingAgent: Send + Sync {
    /// Its name in plain words ("Claude Code").
    fn name(&self) -> &str;

    /// Whether the program a launch runs (its file name: `claude`,
    /// `codex.exe`) is this agent.
    fn launches(&self, program: &str) -> bool;

    /// The modes it can run in, safest first.
    fn modes(&self) -> &[PermissionMode];

    /// Its mode when its owner chose none: checked when it can be; `None`
    /// keeps it from running delegated work until the owner picks one.
    fn default_mode(&self) -> Option<PermissionMode> {
        self.modes()
            .contains(&PermissionMode::Checked)
            .then_some(PermissionMode::Checked)
    }

    /// Why some modes are not offered, in plain words; empty when all are.
    fn note(&self) -> &str {
        ""
    }

    /// How to find and launch it on a machine that has it installed.
    fn discovery(&self) -> Option<Discovery> {
        None
    }

    /// The arguments a launch the owner stored may carry: options that
    /// change what it may do are absent. `None`: none are checked.
    fn launch_arguments(&self) -> Option<LaunchArguments> {
        None
    }

    /// Whether it can run in `mode` for a run of this shape here
    /// (`can_host`: its approval requests can be answered).
    fn can_run(&self, mode: PermissionMode, read_only: bool, can_host: bool) -> bool;

    /// Shape `launch` for `request` and say how the run goes (`Err`: it
    /// cannot run as asked, in words for the owner).
    fn prepare(&self, request: &RunRequest<'_>, launch: &mut Launch) -> Result<Plan, String>;

    /// The actions its own permission check refused, from the record it
    /// writes at the end of a run (`stdout` is its whole output, as the
    /// embedding agent reads it). Empty when it keeps none.
    fn refused_actions(&self, _stdout: &str) -> Vec<String> {
        Vec::new()
    }
}

/// One run, as the embedding agent asks for it.
#[derive(Debug, Clone, Copy)]
pub struct RunRequest<'a> {
    /// Its mode (the owner's choice for this agent).
    pub mode: PermissionMode,
    /// Whether the embedding agent enforces the mode on this run; when it
    /// does not (a run outside its checks), only the launch's basics are
    /// set.
    pub enforce: bool,
    /// The run may only read.
    pub read_only: bool,
    /// The folder it works in.
    pub working_dir: Option<&'a str>,
    /// The paths it may read and write (the working folder among them).
    pub read_paths: &'a [String],
    pub write_paths: &'a [String],
    /// Its approval requests can be answered: a judge is at hand and the
    /// process runs where its stdin can stay open.
    pub can_host: bool,
    /// Its stdin can stay open for a host (a local process), judge or not.
    pub stdin_open: bool,
}

/// A launch's arguments: `args[start..]` are the agent's own; what precedes
/// them is a launcher's (`env FOO=1 codex ...`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub args: Vec<String>,
    pub start: usize,
}

/// Where the task goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptPlacement {
    /// As the last argument.
    LastArgument,
    /// The host sends it on stdin, which stays open for its answers.
    Host,
}

/// How a prepared run goes.
pub struct Plan {
    pub prompt: PromptPlacement,
    /// Variables to set, and to remove, in its environment.
    pub env: Vec<(String, String)>,
    pub env_remove: Vec<String>,
    /// Answers its approval requests, once it is running.
    pub host: Option<Box<dyn HostStarter>>,
    /// Kept until the run is over (a folder made for it, ...).
    pub keep_alive: Option<Box<dyn Any + Send>>,
}

impl Plan {
    /// The task as its last argument; nothing else.
    pub fn plain() -> Self {
        Self {
            prompt: PromptPlacement::LastArgument,
            env: Vec::new(),
            env_remove: Vec::new(),
            host: None,
            keep_alive: None,
        }
    }
}

/// Starts a run's host once its process is running: it sends `prompt` (the
/// task) on stdin as its agent reads it.
#[async_trait]
pub trait HostStarter: Send {
    async fn start(
        self: Box<Self>,
        stdin: ChildStdin,
        judge: Arc<dyn ActionJudge>,
        prompt: &str,
    ) -> Arc<dyn AgentHost>;
}

/// Answers one running agent's approval requests and reads its output.
#[async_trait]
pub trait AgentHost: Send + Sync {
    /// Handle one stdout line; returns the lines the embedding agent reads
    /// in its place (the line itself, a translation of it, or none). A
    /// request is answered in its own task, so output keeps being read
    /// while the judge decides.
    async fn on_line(&self, line: &str) -> Vec<String>;

    /// What it answered so far.
    fn answers(&self) -> HostAnswers;

    /// Whether the agent's work ended unfinished although its process may
    /// exit 0 (its turn failed, or never started).
    fn unfinished(&self) -> bool {
        false
    }

    /// Close its stdin: the agent exits.
    async fn close(&self);
}

/// What a host answered over a run, as the owner reads each action.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HostAnswers {
    /// Actions the owner declined.
    pub declined: Vec<String>,
    /// Actions a safeguard refused, for an agent that does not report its
    /// own refusals.
    pub refused: Vec<String>,
}

/// How to find an agent and launch it by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Discovery {
    /// Its name among the embedding agent's tools.
    pub name: &'static str,
    /// The executable to look for.
    pub command: &'static str,
    /// Its default arguments.
    pub args: &'static [&'static str],
    pub description: &'static str,
}

/// The arguments a stored launch may carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchArguments {
    pub flags: &'static [&'static str],
    /// Options followed by a value.
    pub valued: &'static [&'static str],
    pub subcommands: &'static [&'static str],
}

impl LaunchArguments {
    /// The first of `args` (the agent's own) not on the list, if any.
    pub fn unlisted(&self, args: &[String]) -> Option<String> {
        let mut index = 0;
        while index < args.len() {
            let arg = args[index].as_str();
            index += 1;
            if !arg.starts_with('-') {
                if !self.subcommands.contains(&arg) {
                    return Some(arg.to_string());
                }
                continue;
            }
            let (name, inline_value) = match arg.split_once('=') {
                Some((name, _)) => (name, true),
                None => (arg, false),
            };
            if self.valued.contains(&name) {
                if !inline_value {
                    index += 1;
                }
            } else if inline_value || !self.flags.contains(&name) {
                return Some(arg.to_string());
            }
        }
        None
    }
}

/// The process's stdout, read by `host` and replaced by the lines it gives
/// back. Its stdin closes at the end.
pub fn read_through<R>(stdout: R, host: Arc<dyn AgentHost>) -> tokio::io::DuplexStream
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let (mut writer, reader) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            for mut out in host.on_line(&line).await {
                out.push('\n');
                if writer.write_all(out.as_bytes()).await.is_err() {
                    return;
                }
            }
        }
        host.close().await;
    });
    reader
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stored_launch_may_carry_only_listed_arguments() {
        let list = LaunchArguments {
            flags: &["--json"],
            valued: &["--model"],
            subcommands: &["exec"],
        };
        let args = |values: &[&str]| values.iter().map(|v| v.to_string()).collect::<Vec<_>>();
        assert_eq!(
            list.unlisted(&args(&["exec", "--json", "--model", "m"])),
            None
        );
        assert_eq!(list.unlisted(&args(&["exec", "--model=m"])), None);
        assert_eq!(
            list.unlisted(&args(&["exec", "--full-auto"])),
            Some("--full-auto".to_string())
        );
        assert_eq!(
            list.unlisted(&args(&["exec", "--json=1"])),
            Some("--json=1".to_string())
        );
        assert_eq!(
            list.unlisted(&args(&["review"])),
            Some("review".to_string())
        );
    }
}
