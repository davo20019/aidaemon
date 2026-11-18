use crate::safeguard::Safeguard;
use crate::store::StoreError;

/// Misconfiguration caught when building an agent.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum BuildError {
    #[error("agent id must not be empty")]
    EmptyAgentId,
    #[error("no provider configured")]
    MissingProvider,
    /// No [`crate::AgentBuilder::model`], and the provider has no
    /// [`crate::Provider::default_model`].
    #[error("no model configured: call `.model(..)` (the provider names no default model)")]
    MissingModel,
    #[error("duplicate tool name `{0}`")]
    DuplicateTool(String),
    #[error("invalid tool name `{0}`: use 1-64 characters from [a-zA-Z0-9_-]")]
    InvalidToolName(String),
    #[error(
        "agent has tools but no hook enforces or attestation covers the {0} safeguard; \
         add the policy hook, `attest` it, or `allow_unguarded`"
    )]
    MissingSafeguard(Safeguard),
}

/// Errors from `run`/`resume` itself. Things that happen *inside* a run
/// (tool failures, provider outages, limits) are reported through
/// [`crate::RunStatus`], not here.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CoreError {
    #[error("state belongs to agent `{found}`, not `{expected}`")]
    AgentMismatch { expected: String, found: String },
    #[error("nothing to resume: the run is not interrupted")]
    NotInterrupted,
    #[error("the run is interrupted; settle it with `Agent::resume` first")]
    PendingInterruption,
    #[error("saved run state has schema {found}; this kernel resumes up to {supported}")]
    UnsupportedRunState { found: u32, supported: u32 },
    /// The approval in this state was already used: an answer runs its
    /// call once, however many copies of the state are resumed.
    #[error("this approval was already used; the call it approved does not run again")]
    ApprovalAlreadyUsed,
    #[error(transparent)]
    Store(#[from] StoreError),
}
