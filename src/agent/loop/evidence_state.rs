use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::traits::ToolTargetHint;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvidenceKind {
    FileRead,
    CommandOutput,
    GitState,
    ProcessState,
    ApiResponse,
    VerificationResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvidenceTrust {
    Direct,
    Inferred,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub kind: EvidenceKind,
    pub source: String,
    pub observed_at: DateTime<Utc>,
    pub trust: EvidenceTrust,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<ToolTargetHint>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceState {
    pub target: Option<ToolTargetHint>,
    #[serde(default)]
    pub records: Vec<EvidenceRecord>,
    #[serde(default)]
    pub contradictions: Vec<String>,
    #[serde(default)]
    pub post_change_verification_done: bool,
}

impl EvidenceState {}
