use serde::{Deserialize, Serialize};

/// Action-bound grant issued by the deterministic mandate gate. It is carried
/// only through Rust control-plane structs, never model-visible arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateAuthorityGrant {
    pub mandate_id: String,
    pub mandate_version: i64,
    /// Stable owner identity at the exact policy version that authorized this
    /// action. Session routing never substitutes for this principal binding.
    #[serde(default)]
    pub owner_principal_id: String,
    pub decision_cycle_id: String,
    pub action_digest: String,
    pub counts_toward_cycle_budget: bool,
    /// One-based fenced reservation number for a mutation; zero for a legacy
    /// or non-mutating grant. The state-store CAS must reserve this exact slot.
    #[serde(default)]
    pub reserved_action_attempt: i64,
    /// Dispatcher-owned tool call identity, bound after deterministic policy
    /// authorization and before the atomic ledger reservation. Never supplied
    /// by or exposed to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}
