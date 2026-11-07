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
    #[error("no model configured")]
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
    #[error(transparent)]
    Store(#[from] StoreError),
}
