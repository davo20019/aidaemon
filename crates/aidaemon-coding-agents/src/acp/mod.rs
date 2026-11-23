//! Any coding agent that speaks the Agent Client Protocol: each action it
//! takes is asked first ([`host::AcpHost`]), so every one can be judged.
//!
//! - Every command (its default): each permission request is judged.
//! - Full access: each is allowed as asked, unjudged.
//!
//! There is no checked mode: the protocol has no sandbox of its own, and
//! an agent's own one is not trusted unless verified (`agy --sandbox` still
//! wrote outside its folder once a command was allowed; live probe,
//! 2026-11-18).
//!
//! [`AcpAgent::antigravity`] and [`AcpAgent::gemini`] are the ones known;
//! [`AcpAgent::custom`] is any other the owner names.

pub mod events;
pub mod host;

use crate::{
    CodingAgent, Discovery, Launch, LaunchArguments, PermissionMode, Plan, PromptPlacement,
    RunRequest,
};

/// A coding agent run over the Agent Client Protocol.
#[derive(Debug, Clone)]
pub struct AcpAgent {
    name: String,
    program: String,
    /// The arguments that start its protocol server (`acp`, `--acp`).
    serve: Vec<String>,
    discovery: Option<Discovery>,
    arguments: Option<LaunchArguments>,
    /// Whose account it works with, in plain words.
    account: &'static str,
}

impl AcpAgent {
    /// Any agent: its name, the program it launches, and the arguments
    /// that start its protocol server (none when its launch already does).
    pub fn custom(name: &str, program: &str, serve: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            program: program.to_string(),
            serve: serve.iter().map(|arg| arg.to_string()).collect(),
            discovery: None,
            arguments: None,
            account: "",
        }
    }

    /// Antigravity CLI (`agy acp`).
    pub fn antigravity() -> Self {
        Self {
            discovery: Some(Discovery {
                name: "agy",
                command: "agy",
                args: &["acp"],
                description: "Antigravity CLI — Google's AI coding agent",
            }),
            arguments: Some(LaunchArguments {
                flags: &[],
                valued: &["--agent", "--model"],
                subcommands: &["acp"],
            }),
            account: "Its own Antigravity sign-in: your Google account and its plan.",
            ..Self::custom("Antigravity CLI", "agy", &["acp"])
        }
    }

    /// Gemini CLI (`gemini --acp`). Since 2026-06-18 Google serves it only
    /// to enterprise licences and paid API keys: a personal sign-in is
    /// refused a session, which its readiness check reports.
    pub fn gemini() -> Self {
        Self {
            discovery: Some(Discovery {
                name: "gemini",
                command: "gemini",
                args: &["--acp"],
                description: "Gemini CLI — Google's AI coding agent",
            }),
            arguments: Some(LaunchArguments {
                flags: &["--acp", "--experimental-acp", "--debug", "-d"],
                valued: &["--model", "-m"],
                subcommands: &[],
            }),
            account: "Its own Gemini CLI sign-in: a Google account or a Gemini API key.",
            ..Self::custom("Gemini CLI", "gemini", &["--acp"])
        }
    }
}

impl CodingAgent for AcpAgent {
    fn read_events(&self, line: &str) -> Vec<crate::AgentEvent> {
        events::read_events(line)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn launches(&self, program: &str) -> bool {
        program
            .trim_end_matches(".exe")
            .eq_ignore_ascii_case(&self.program)
    }

    fn modes(&self) -> &[PermissionMode] {
        &[PermissionMode::EveryCommand, PermissionMode::FullAccess]
    }

    /// Every action asked: the safest it offers.
    fn default_mode(&self) -> Option<PermissionMode> {
        Some(PermissionMode::EveryCommand)
    }

    fn note(&self) -> &str {
        "It has no sandbox AIDaemon can rely on, so every action it takes is checked."
    }

    fn discovery(&self) -> Option<Discovery> {
        self.discovery
    }

    fn launch_arguments(&self) -> Option<LaunchArguments> {
        self.arguments
    }

    /// Its protocol needs stdin held open; checking also needs a judge.
    fn can_run(&self, mode: PermissionMode, _read_only: bool, can_host: bool) -> bool {
        match mode {
            PermissionMode::FullAccess => true,
            PermissionMode::EveryCommand => can_host,
            PermissionMode::Checked => false,
        }
    }

    fn account(&self) -> &str {
        self.account
    }

    /// Set over the protocol once its session opens (`session/set_model`).
    fn takes_model(&self) -> bool {
        true
    }

    /// A session it opens (no prompt sent), or its own reason not to.
    fn readiness_probe(&self) -> Option<crate::Probe> {
        Some(crate::Probe::AcpSession {
            serve: self.serve.clone(),
        })
    }

    fn prepare(&self, request: &RunRequest<'_>, launch: &mut Launch) -> Result<Plan, String> {
        if !request.stdin_open {
            return Err(format!(
                "{} runs only where its stdin can stay open (a local process)",
                self.name
            ));
        }
        if request.enforce && request.mode != PermissionMode::FullAccess && !request.can_host {
            return Err(format!(
                "{}'s requests can't be answered here (its run needs a checked turn)",
                self.name
            ));
        }
        if request.enforce && request.mode == PermissionMode::Checked {
            return Err(format!("{} has no checked mode", self.name));
        }
        // Its protocol server, unless the launch already starts it.
        let own = &launch.args[launch.start.min(launch.args.len())..];
        if !self.serve.iter().all(|arg| own.contains(arg)) {
            launch.args.extend(self.serve.iter().cloned());
        }
        Ok(Plan {
            prompt: PromptPlacement::Host,
            host: Some(Box::new(host::AcpHostStarter {
                working_dir: request.working_dir.map(str::to_string),
                model: request.model.map(str::to_string),
            })),
            ..Plan::plain()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(mode: PermissionMode, can_host: bool) -> RunRequest<'static> {
        RunRequest {
            mode,
            enforce: true,
            read_only: false,
            working_dir: Some("/w/project"),
            read_paths: &[],
            write_paths: &[],
            can_host,
            stdin_open: true,
            model: None,
        }
    }

    #[test]
    fn it_starts_its_protocol_server_and_is_hosted_in_every_mode_it_offers() {
        let agent = AcpAgent::antigravity();
        for mode in [PermissionMode::EveryCommand, PermissionMode::FullAccess] {
            let mut launch = Launch {
                args: Vec::new(),
                start: 0,
            };
            let plan = agent.prepare(&request(mode, true), &mut launch).unwrap();
            assert_eq!(launch.args, ["acp"]);
            assert_eq!(plan.prompt, PromptPlacement::Host);
            assert!(plan.host.is_some());
        }
        // A launch that already starts it is left as it is.
        let mut launch = Launch {
            args: vec!["acp".into(), "--model".into(), "m".into()],
            start: 0,
        };
        agent
            .prepare(&request(PermissionMode::EveryCommand, true), &mut launch)
            .unwrap();
        assert_eq!(launch.args, ["acp", "--model", "m"]);
    }

    #[test]
    fn checking_needs_a_judge_and_there_is_no_checked_mode() {
        let agent = AcpAgent::custom("Synthetic Agent", "synthetic-agent", &["--acp"]);
        let mut launch = Launch {
            args: Vec::new(),
            start: 0,
        };
        assert!(
            agent
                .prepare(&request(PermissionMode::EveryCommand, false), &mut launch)
                .is_err()
        );
        assert!(
            agent
                .prepare(&request(PermissionMode::FullAccess, false), &mut launch)
                .is_ok()
        );
        assert!(!agent.can_run(PermissionMode::Checked, false, true));
        assert_eq!(agent.default_mode(), Some(PermissionMode::EveryCommand));
        assert!(!agent.modes().contains(&PermissionMode::Checked));
    }
}
