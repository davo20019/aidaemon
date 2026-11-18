//! An agent whose approval requests nobody can answer: it runs only with
//! full access, as its owner chooses.

use crate::{CodingAgent, Discovery, Launch, LaunchArguments, PermissionMode, Plan, RunRequest};

/// A coding agent run as it is launched, with no host.
#[derive(Debug, Clone)]
pub struct PlainAgent {
    name: String,
    program: String,
    discovery: Option<Discovery>,
    arguments: Option<LaunchArguments>,
}

impl PlainAgent {
    /// A known agent: its name, the program it launches, how to find it and
    /// the arguments a stored launch of it may carry.
    pub fn new(
        name: &str,
        program: &str,
        discovery: Option<Discovery>,
        arguments: Option<LaunchArguments>,
    ) -> Self {
        Self {
            name: name.to_string(),
            program: program.to_string(),
            discovery,
            arguments,
        }
    }

    /// Any other program, named by its file name.
    pub fn unknown(program: &str) -> Self {
        Self::new(program, program, None, None)
    }

    /// GitHub Copilot CLI.
    pub fn copilot() -> Self {
        Self::new(
            "GitHub Copilot CLI",
            "copilot",
            Some(Discovery {
                name: "copilot",
                command: "copilot",
                args: &["-p"],
                description: "GitHub Copilot CLI",
            }),
            Some(LaunchArguments {
                flags: &["-p", "--silent", "-s"],
                valued: &["--model"],
                subcommands: &[],
            }),
        )
    }

    /// Aider.
    pub fn aider() -> Self {
        Self::new(
            "Aider",
            "aider",
            Some(Discovery {
                name: "aider",
                command: "aider",
                args: &["--message"],
                description: "Aider — AI pair programming",
            }),
            Some(LaunchArguments {
                flags: &[
                    "--message",
                    "--no-auto-commits",
                    "--no-git",
                    "--no-pretty",
                    "--no-stream",
                    "--no-check-update",
                ],
                valued: &["--model"],
                subcommands: &[],
            }),
        )
    }
}

impl CodingAgent for PlainAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn launches(&self, program: &str) -> bool {
        program
            .trim_end_matches(".exe")
            .eq_ignore_ascii_case(&self.program)
    }

    fn modes(&self) -> &[PermissionMode] {
        &[PermissionMode::FullAccess]
    }

    fn note(&self) -> &str {
        "AIDaemon can't check this agent's actions, so it can only run with full access."
    }

    fn discovery(&self) -> Option<Discovery> {
        self.discovery
    }

    fn launch_arguments(&self) -> Option<LaunchArguments> {
        self.arguments
    }

    fn can_run(&self, mode: PermissionMode, _read_only: bool, _can_host: bool) -> bool {
        mode == PermissionMode::FullAccess
    }

    fn prepare(&self, request: &RunRequest<'_>, _launch: &mut Launch) -> Result<Plan, String> {
        if request.enforce && request.mode != PermissionMode::FullAccess {
            return Err(format!(
                "{} can only run with full access: {}",
                self.name,
                self.note()
            ));
        }
        Ok(Plan::plain())
    }
}
