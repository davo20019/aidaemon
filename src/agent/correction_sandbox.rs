//! Correction-mode sandbox: a pure classifier deciding whether a proposed
//! action may run during autonomous self-correction. Deny-the-dangerous-classes
//! policy (destructive ops, credential/config management, unintended external
//! mutations); everything else allowed. Wired by Plan 3 (out-of-band
//! remediation), which adds the per-attempt approval-bypass execution context.

use crate::traits::SelfCorrectionSubjectKind;

/// An external account the correction is legitimately allowed to act on, derived
/// ONLY from deterministic sources (scheduled-goal config, connected-account
/// metadata, an already-selected tool/account binding) — never inferred from
/// free-text by this layer.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntendedAccount {
    pub provider: String,
    pub account_id: String,
    pub account_label: String,
}

/// Rigid context for a correction subject. Produced upstream (routing/intent/
/// task setup); the classifier never infers intent from it.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CorrectionSubjectContext {
    pub subject_id: String,
    pub subject_kind: SelfCorrectionSubjectKind,
    pub session_id: String,
    pub original_request: String,
    pub completion_contract_summary: String,
    pub intended_accounts: Vec<IntendedAccount>,
    pub allowed_external_targets: Vec<String>,
}

/// The fields of a proposed tool action the classifier needs. The caller (Plan 3)
/// extracts these from the concrete tool call.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ProposedAction<'a> {
    pub tool_name: &'a str,
    /// For terminal/run_command: the shell command.
    pub terminal_command: Option<&'a str>,
    /// For http_request/web_fetch: the HTTP method (e.g. "GET", "POST").
    pub http_method: Option<&'a str>,
    /// External target (host/account/provider) the action would touch, if any.
    pub external_target: Option<&'a str>,
}

/// Classifier verdict. `Blocked` carries a human-readable reason (surfaced in the
/// attempt ledger and logs; never an approval prompt).
#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub enum ActionVerdict {
    Allowed,
    Blocked(String),
}

/// Credential/auth/config management tools — always blocked in correction mode.
#[allow(dead_code)]
pub const CREDENTIAL_TOOLS: &[&str] = &[
    "manage_oauth",
    "manage_http_auth",
    "manage_api",
    "manage_config",
    "manage_mcp",
    "manage_cli_agents",
];

/// Tools that always mutate an external account/destination.
#[allow(dead_code)]
pub const MUTATING_EXTERNAL_TOOLS: &[&str] = &["share_memory", "send_file"];

/// HTTP methods treated as mutating (external write).
#[allow(dead_code)]
pub const MUTATING_HTTP_METHODS: &[&str] = &["POST", "PUT", "PATCH", "DELETE"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_class_constants_are_populated() {
        assert!(CREDENTIAL_TOOLS.contains(&"manage_oauth"));
        assert!(CREDENTIAL_TOOLS.contains(&"manage_config"));
        assert!(MUTATING_EXTERNAL_TOOLS.contains(&"share_memory"));
        assert!(MUTATING_HTTP_METHODS.contains(&"POST"));
        assert!(!MUTATING_HTTP_METHODS.contains(&"GET"));
    }
}
