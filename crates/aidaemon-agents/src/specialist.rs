//! Specialist profiles: the stable labels a host gives its child agents, so
//! their sessions and events say what kind of work each one does.

use serde::{Deserialize, Serialize};

/// Stable specialist profile used to label child-agent sessions and events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialistKind {
    TaskLead,
    Executor,
    Research,
    ArtifactWriter,
    Code,
    BrowserVerifier,
    Review,
    CommsDraft,
    Generic,
}

impl SpecialistKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SpecialistKind::TaskLead => "task_lead",
            SpecialistKind::Executor => "executor",
            SpecialistKind::Research => "research",
            SpecialistKind::ArtifactWriter => "artifact_writer",
            SpecialistKind::Code => "code",
            SpecialistKind::BrowserVerifier => "browser_verifier",
            SpecialistKind::Review => "review",
            SpecialistKind::CommsDraft => "comms_draft",
            SpecialistKind::Generic => "generic",
        }
    }

    /// The kind named by its [`as_str`](Self::as_str) label.
    #[allow(clippy::should_implement_trait)] // returns an Option, never an error
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "task_lead" => Some(SpecialistKind::TaskLead),
            "executor" => Some(SpecialistKind::Executor),
            "research" => Some(SpecialistKind::Research),
            "artifact_writer" => Some(SpecialistKind::ArtifactWriter),
            "code" => Some(SpecialistKind::Code),
            "browser_verifier" => Some(SpecialistKind::BrowserVerifier),
            "review" => Some(SpecialistKind::Review),
            "comms_draft" => Some(SpecialistKind::CommsDraft),
            "generic" => Some(SpecialistKind::Generic),
            _ => None,
        }
    }

    // Test-only oracle: iterates every variant for exhaustiveness checks in
    // `SpecialistRegistry` unit tests. Production code dispatches on
    // `SpecialistKind` values directly, so this helper is unused there.
    #[allow(dead_code)]
    pub fn all() -> &'static [SpecialistKind] {
        &[
            SpecialistKind::TaskLead,
            SpecialistKind::Executor,
            SpecialistKind::Research,
            SpecialistKind::ArtifactWriter,
            SpecialistKind::Code,
            SpecialistKind::BrowserVerifier,
            SpecialistKind::Review,
            SpecialistKind::CommsDraft,
            SpecialistKind::Generic,
        ]
    }
}
