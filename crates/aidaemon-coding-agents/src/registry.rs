//! The coding agents an embedding agent knows.

use std::sync::Arc;

use crate::{CodingAgent, acp::AcpAgent, claude::ClaudeCode, codex::Codex, plain::PlainAgent};

/// The coding agents an embedding agent knows, in the order it prefers
/// them. A program none of them launches is a [`PlainAgent`].
#[derive(Clone)]
pub struct CodingAgents {
    agents: Vec<Arc<dyn CodingAgent>>,
}

impl CodingAgents {
    /// None.
    pub fn empty() -> Self {
        Self { agents: Vec::new() }
    }

    /// The agents this crate ships: Claude Code and Codex (hosted by their
    /// own protocols), Antigravity CLI, Gemini CLI and OpenCode (hosted over
    /// the Agent Client Protocol), GitHub Copilot CLI and Aider (full access
    /// only).
    /// Whether each can do work now is its readiness check's to say.
    pub fn builtin() -> Self {
        Self::empty()
            .with(ClaudeCode)
            .with(Codex)
            .with(AcpAgent::antigravity())
            .with(AcpAgent::gemini())
            .with(AcpAgent::opencode())
            .with(PlainAgent::copilot())
            .with(PlainAgent::aider())
    }

    /// Add `agent`; it is preferred after the ones already here.
    pub fn with(mut self, agent: impl CodingAgent + 'static) -> Self {
        self.agents.push(Arc::new(agent));
        self
    }

    /// Add `agent` ahead of the ones already here: it is found first for
    /// its program.
    pub fn preferring(mut self, agent: impl CodingAgent + 'static) -> Self {
        self.agents.insert(0, Arc::new(agent));
        self
    }

    /// Every agent, in order.
    pub fn all(&self) -> &[Arc<dyn CodingAgent>] {
        &self.agents
    }

    /// The agent `program` (an executable's file name) is.
    pub fn for_program(&self, program: &str) -> Arc<dyn CodingAgent> {
        self.agents
            .iter()
            .find(|agent| agent.launches(program))
            .cloned()
            .unwrap_or_else(|| Arc::new(PlainAgent::unknown(program)))
    }
}

impl Default for CodingAgents {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;

    #[test]
    fn a_program_is_recognized_by_its_file_name() {
        let agents = CodingAgents::builtin();
        assert_eq!(agents.for_program("claude").name(), "Claude Code");
        assert_eq!(agents.for_program("codex.exe").name(), "Codex");
        assert_eq!(agents.for_program("agy").name(), "Antigravity CLI");
        assert_eq!(agents.for_program("gemini").name(), "Gemini CLI");
        let unknown = agents.for_program("synthetic-coder");
        assert_eq!(unknown.name(), "synthetic-coder");
        assert_eq!(unknown.modes(), &[PermissionMode::FullAccess]);
    }

    fn unlisted(agent: &str, args: &[&str]) -> Option<String> {
        let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
        CodingAgents::builtin()
            .for_program(agent)
            .launch_arguments()?
            .unlisted(&args)
    }

    #[test]
    fn every_default_launch_is_on_its_list() {
        for agent in CodingAgents::builtin().all() {
            let Some(found) = agent.discovery() else {
                continue;
            };
            assert_eq!(unlisted(found.command, found.args), None, "{}", found.name);
        }
    }

    #[test]
    fn a_stored_launch_cannot_widen_what_it_may_do() {
        for (agent, args) in [
            (
                "codex",
                &["exec", "-c", "sandbox_mode=danger-full-access"][..],
            ),
            ("codex", &["exec", "--config=approval_policy=never"]),
            ("codex", &["--profile", "unsafe"]),
            ("claude", &["-p", "--permission-mode", "acceptEdits"]),
            ("claude", &["--add-dir", "/synthetic/elsewhere"]),
            ("claude", &["--verbose=true"]),
            ("agy", &["acp", "--dangerously-skip-permissions"]),
            ("aider", &["--auto-commits"]),
            ("codex", &["sandbox"]),
        ] {
            assert!(unlisted(agent, args).is_some(), "{agent} {args:?}");
        }
    }

    #[test]
    fn listed_values_are_not_read_as_arguments() {
        assert_eq!(
            unlisted("codex", &["exec", "--model", "-c", "--json"]),
            None
        );
        assert_eq!(unlisted("Claude", &["--model=synthetic", "-p"]), None);
        // An agent with no list is not judged by one.
        assert_eq!(unlisted("opaque-agent", &["--anything", "goes"]), None);
    }

    #[test]
    fn an_agent_the_owner_declares_is_found_first_for_its_program() {
        let agents = CodingAgents::builtin().preferring(crate::acp::AcpAgent::custom(
            "Synthetic Coder",
            "synthetic-coder",
            &[],
        ));
        let agent = agents.for_program("synthetic-coder");
        assert_eq!(agent.name(), "Synthetic Coder");
        assert_eq!(agent.default_mode(), Some(PermissionMode::EveryCommand));
    }

    #[test]
    fn every_builtin_agent_says_how_to_find_it() {
        let names: Vec<_> = CodingAgents::builtin()
            .all()
            .iter()
            .filter_map(|agent| agent.discovery().map(|found| found.name))
            .collect();
        assert_eq!(
            names,
            [
                "claude", "codex", "agy", "gemini", "opencode", "copilot", "aider"
            ]
        );
    }
}
