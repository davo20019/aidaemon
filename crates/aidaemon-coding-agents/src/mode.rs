//! How much a coding agent may do on its own, and the most the embedding
//! agent lets it do.
//!
//! The owner picks a mode per agent; the embedding agent sets a cap (the
//! most any agent it hands work to may do, so no agent does more than the
//! embedding agent itself may). Where the choices are kept is the embedding
//! agent's; this module only orders the modes and fits a choice under a
//! cap ([`mode_within`]).

use crate::CodingAgent;

/// How much a coding agent may do on its own. Ordered by how much it may
/// do unasked: [`EveryCommand`](Self::EveryCommand) <
/// [`Checked`](Self::Checked) < [`FullAccess`](Self::FullAccess).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PermissionMode {
    /// On the machine itself, with every command and file change (but the
    /// agent's own short list of read-only commands) asked first.
    EveryCommand,
    /// Confined to the project folder by the agent's own sandbox; anything
    /// beyond it is asked first, and each request is judged.
    Checked,
    /// On the machine itself, nothing asked: what it does is never judged.
    FullAccess,
}

impl PermissionMode {
    /// Its stored name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Checked => "checked",
            Self::EveryCommand => "every_command",
            Self::FullAccess => "full_access",
        }
    }

    /// The mode a stored name names.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "checked" => Some(Self::Checked),
            "every_command" => Some(Self::EveryCommand),
            "full_access" => Some(Self::FullAccess),
            _ => None,
        }
    }
}

/// The modes `agent` offers that do no more than `most`, safest first.
pub fn modes_within(agent: &dyn CodingAgent, most: PermissionMode) -> Vec<PermissionMode> {
    let mut modes: Vec<PermissionMode> = agent
        .modes()
        .iter()
        .copied()
        .filter(|mode| *mode <= most)
        .collect();
    modes.sort();
    modes
}

/// The mode `agent` runs in under the cap `most`.
///
/// - `chosen` `Some`: the owner's pick, when the agent offers it and it is
///   within the cap; a pick above the cap (the cap was lowered since) runs
///   as the most the agent may do within it.
/// - `chosen` `None` with `follow_cap`: the most the agent may do within
///   the cap ("same as the embedding agent").
/// - `chosen` `None` without it: the agent's kind's default, within the cap.
///
/// `None`: none of its modes fit, so it may not run delegated work.
pub fn mode_within(
    agent: &dyn CodingAgent,
    chosen: Option<PermissionMode>,
    follow_cap: bool,
    most: PermissionMode,
) -> Option<PermissionMode> {
    let within = modes_within(agent, most);
    let loosest = within.last().copied();
    match chosen.filter(|mode| agent.modes().contains(mode)) {
        Some(mode) if mode <= most => Some(mode),
        Some(_) => loosest,
        None if follow_cap => loosest,
        None => agent.default_mode().filter(|mode| *mode <= most),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{claude::ClaudeCode, plain::PlainAgent};

    #[test]
    fn modes_are_ordered_by_what_they_let_an_agent_do_unasked() {
        assert!(PermissionMode::EveryCommand < PermissionMode::Checked);
        assert!(PermissionMode::Checked < PermissionMode::FullAccess);
        for mode in [
            PermissionMode::EveryCommand,
            PermissionMode::Checked,
            PermissionMode::FullAccess,
        ] {
            assert_eq!(PermissionMode::parse(mode.as_str()), Some(mode));
        }
    }

    /// A pick within the cap runs as picked; one above it runs at the most
    /// the cap allows; following the cap is the most it allows.
    #[test]
    fn a_pick_never_runs_above_the_cap() {
        let full = PermissionMode::FullAccess;
        let checked = PermissionMode::Checked;
        assert_eq!(
            mode_within(&ClaudeCode, Some(full), false, full),
            Some(full)
        );
        assert_eq!(
            mode_within(&ClaudeCode, Some(full), false, checked),
            Some(checked)
        );
        assert_eq!(mode_within(&ClaudeCode, None, true, full), Some(full));
        assert_eq!(mode_within(&ClaudeCode, None, true, checked), Some(checked));
        // No pick, not following: the kind's default, within the cap.
        assert_eq!(mode_within(&ClaudeCode, None, false, full), Some(checked));
        assert_eq!(
            mode_within(&ClaudeCode, None, false, PermissionMode::EveryCommand),
            None
        );
        assert_eq!(
            modes_within(&ClaudeCode, checked),
            [PermissionMode::EveryCommand, checked]
        );
    }

    /// An agent nobody can check runs only with full access, so under a
    /// lower cap it runs nothing, whatever was picked.
    #[test]
    fn an_agent_that_only_runs_unchecked_runs_nothing_under_a_lower_cap() {
        let agent = PlainAgent::unknown("synthetic-coder");
        let full = PermissionMode::FullAccess;
        assert_eq!(mode_within(&agent, None, false, full), None);
        assert_eq!(mode_within(&agent, Some(full), false, full), Some(full));
        assert_eq!(
            mode_within(&agent, Some(full), false, PermissionMode::Checked),
            None
        );
        assert_eq!(
            mode_within(&agent, None, true, PermissionMode::Checked),
            None
        );
        // A mode it does not offer is no pick.
        assert_eq!(
            mode_within(&agent, Some(PermissionMode::Checked), false, full),
            None
        );
    }
}
