//! A coding agent's launch under a device ceiling: in a run an e-ink device
//! started or steered (`crate::device_cap`, read from the turn's judge in
//! the call's typed context), the agent never runs with full access. Its
//! mode is held to Checked, so it runs in its own sandbox or asks before
//! every action, and each request it makes comes back to the run's
//! safeguards as a nested call, the device ceiling's hook included. Such a
//! launch is no card of its own (`holds_to_medium_ceiling`).
//!
//! An agent that cannot run that way here (only full access, or not a
//! local process) is not held: its launch is asked about on a card only the
//! phone or the computer can answer, and once allowed there it runs in the
//! owner's mode as before. The same predicate ([`CliAgentTool::fits_ceiling`])
//! decides both, so a launch the hook let through without a card is always
//! held.

use super::*;
use aidaemon_coding_agents::PermissionMode;

/// The loosest mode a held launch runs in.
pub(super) const CEILING_MODE: PermissionMode = PermissionMode::Checked;

impl CliAgentTool {
    /// The call's run is capped by its device: read from the turn's judge
    /// the dispatcher hands the call (typed, never from its arguments).
    pub(super) fn device_capped(facts: &ToolCallContext) -> bool {
        facts
            .extensions
            .get::<Arc<crate::approvals::DetachedNestedJudge>>()
            .is_some_and(|judge| judge.device_cap().is_capped())
    }

    /// Whether `agent` can run here held to the ceiling: one of its modes
    /// no looser than [`CEILING_MODE`] that it runs with its requests
    /// answered (hosted), in a local process.
    pub(super) fn fits_ceiling(
        &self,
        agent: &dyn aidaemon_coding_agents::CodingAgent,
        read_only: bool,
    ) -> bool {
        self.backend.kind() == aidaemon_tools::backend::BackendKind::Local
            && aidaemon_coding_agents::modes_within(agent, CEILING_MODE)
                .into_iter()
                .any(|mode| agent.can_run(mode, read_only, true))
    }

    /// The mode cap a launch takes in a run its device capped: `mode_cap`
    /// (a peer's lead) held to [`CEILING_MODE`] when the agent fits it;
    /// unchanged otherwise (an agent that does not fit was asked about on
    /// the phone, and runs as the owner set it).
    pub(super) fn ceiling_mode_cap(
        &self,
        capped: bool,
        agent: &dyn aidaemon_coding_agents::CodingAgent,
        read_only: bool,
        mode_cap: Option<PermissionMode>,
    ) -> Option<PermissionMode> {
        if !capped || !self.fits_ceiling(agent, read_only) {
            return mode_cap;
        }
        Some(mode_cap.map_or(CEILING_MODE, |cap| cap.min(CEILING_MODE)))
    }

    /// The mode cap the ceiling puts on a run of `tool` in this call
    /// (`None`: not capped, or the agent does not fit it).
    pub(super) fn ceiling_cap_for(
        &self,
        tool: &str,
        facts: &ToolCallContext,
        read_only: bool,
    ) -> Option<PermissionMode> {
        let agent = self.agent_named(tool)?;
        self.ceiling_mode_cap(Self::device_capped(facts), &*agent, read_only, None)
    }

    /// Whether this call holds to the Medium ceiling by itself: listing,
    /// checking and stopping its own runs; a run when every configured
    /// agent fits the ceiling here. Every agent, not only the one the call
    /// names: a run may go to another agent (a fallback, a follow-up's
    /// earlier agent), and a launch let through here must always be held.
    pub(super) fn holds_launch_to_ceiling(&self, arguments: &str) -> bool {
        let Ok(args) = serde_json::from_str::<Value>(arguments) else {
            return false;
        };
        let text = |key: &str| args.get(key).and_then(Value::as_str);
        match text("action").unwrap_or("run") {
            "list" | "check" | "cancel" => true,
            "run" => {
                let read_only = text("workspace_mode") == Some("read_only");
                let tools = self
                    .tools
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                !tools.is_empty()
                    && tools
                        .values()
                        .all(|entry| self.fits_ceiling(&*self.entry_agent(entry), read_only))
            }
            _ => false,
        }
    }
}
