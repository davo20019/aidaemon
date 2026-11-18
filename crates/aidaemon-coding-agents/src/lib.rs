//! Coding agents an aidaemon agent hands work to: Claude Code, Codex, and
//! any other program the embedding agent can launch.
//!
//! Each agent is a [`CodingAgent`]: it names the program it launches, the
//! [`PermissionMode`]s it can run in, and how to launch it in each
//! ([`CodingAgent::prepare`]). When a mode needs the agent's own approval
//! requests answered, the launch comes with an [`AgentHost`] that speaks
//! the agent's protocol on its stdin and stdout and turns each request into
//! an [`AgentAction`] for the embedding agent's [`ActionJudge`] to decide.
//! This crate decides nothing: the judge is the embedding agent's own
//! safeguards (aidaemon judges an action exactly as it judges its model's
//! tool calls).
//!
//! Adding an agent is one type implementing [`CodingAgent`] and one entry in
//! a [`CodingAgents`] registry ([`CodingAgents::builtin`] holds the ones
//! this crate ships). An agent whose requests nobody can answer runs only
//! with [`PermissionMode::FullAccess`] ([`plain::PlainAgent`]).
//!
//! The process itself is the embedding agent's to run: this crate shapes
//! its arguments, environment and stdin ([`Launch`], [`Plan`]) and reads
//! its output through the host ([`read_through`]).

mod action;
mod agent;
pub mod claude;
pub mod codex;
mod mode;
pub mod plain;
mod registry;
mod stdin;

pub use action::{ActionJudge, AgentAction, TextEdit, Verdict};
pub use agent::{
    AgentHost, CodingAgent, Discovery, HostAnswers, HostStarter, Launch, LaunchArguments, Plan,
    PromptPlacement, RunRequest, read_through,
};
pub use mode::{ModeStore, PermissionMode, mode_of, set_mode};
pub use registry::CodingAgents;
