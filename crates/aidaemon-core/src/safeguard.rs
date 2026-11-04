/// A safety property every agent with tools must have before it can run.
///
/// The kernel does not implement these: a hook declares the ones it provides
/// through [`crate::Hook::safeguards`], or the composition root attests one
/// that something else enforces (for example its own tool adapter) with
/// [`crate::AgentBuilder::attest`]. [`crate::AgentBuilder::build`] refuses an
/// agent with tools that leaves any of them uncovered, so forgetting to wire
/// a policy is a build error instead of a silent hole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Safeguard {
    /// Every call stays within the authority explicitly granted by whoever
    /// owns the agent (owner configuration, role, grants, modes). That grant
    /// may be full by the owner's choice; approval prompts are one way to
    /// enforce a narrower grant, not a requirement.
    Authority,
    /// Calls stay inside the paths and resources the task was granted.
    Scope,
    /// Output from tools the agent does not control (web pages, MCP
    /// servers, third-party APIs) reaches the model marked as data, not
    /// instructions, and calls with egress made after reading it are
    /// verified by a clean check or, by the owner's explicit choice,
    /// recorded.
    UntrustedOutput,
}

impl Safeguard {
    /// Every safeguard an agent with tools must cover.
    pub const REQUIRED: [Safeguard; 3] = [
        Safeguard::Authority,
        Safeguard::Scope,
        Safeguard::UntrustedOutput,
    ];
}

impl std::fmt::Display for Safeguard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Safeguard::Authority => "authority",
            Safeguard::Scope => "scope",
            Safeguard::UntrustedOutput => "untrusted output",
        })
    }
}
