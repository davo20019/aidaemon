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

/// Pure correction-mode action classifier. See module docs + Plan 2b for policy.
#[allow(dead_code)]
pub fn classify_action(action: &ProposedAction, ctx: &CorrectionSubjectContext) -> ActionVerdict {
    // (1) Destructive terminal commands.
    if matches!(action.tool_name, "terminal" | "run_command") {
        if let Some(cmd) = action.terminal_command {
            if let Some(reason) = crate::tools::command_risk::hard_block_reason(cmd) {
                return ActionVerdict::Blocked(format!("destructive command: {reason}"));
            }
        }
    }

    // (2) Credential / config management.
    if CREDENTIAL_TOOLS.contains(&action.tool_name) {
        return ActionVerdict::Blocked(
            "credential/config management is not allowed during autonomous correction".to_string(),
        );
    }

    // (3) Mutating external writes — allowed only for an intended account/target.
    let is_mutating_external = MUTATING_EXTERNAL_TOOLS.contains(&action.tool_name)
        || (matches!(action.tool_name, "http_request" | "web_fetch")
            && action
                .http_method
                .map(|m| MUTATING_HTTP_METHODS.contains(&m.to_uppercase().as_str()))
                .unwrap_or(false));
    if is_mutating_external {
        let target = action.external_target.unwrap_or("");
        let matches_intended = !target.is_empty()
            && (ctx
                .intended_accounts
                .iter()
                .any(|a| a.account_id == target || a.provider == target)
                || ctx.allowed_external_targets.iter().any(|t| t == target));
        if !matches_intended {
            return ActionVerdict::Blocked(
                "external-account mutation outside the intended accounts".to_string(),
            );
        }
    }

    // (4) Everything else (read-only external, delegation tools, safe tools).
    ActionVerdict::Allowed
}

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

    fn ctx_with(accounts: Vec<IntendedAccount>) -> CorrectionSubjectContext {
        CorrectionSubjectContext {
            subject_id: "s".to_string(),
            subject_kind: SelfCorrectionSubjectKind::Task,
            session_id: "sess".to_string(),
            original_request: "do the thing".to_string(),
            completion_contract_summary: "".to_string(),
            intended_accounts: accounts,
            allowed_external_targets: vec![],
        }
    }

    fn action(tool: &str) -> ProposedAction<'_> {
        ProposedAction {
            tool_name: tool,
            terminal_command: None,
            http_method: None,
            external_target: None,
        }
    }

    #[test]
    fn destructive_terminal_is_blocked() {
        let mut a = action("terminal");
        a.terminal_command = Some("rm -rf /");
        match classify_action(&a, &ctx_with(vec![])) {
            ActionVerdict::Blocked(r) => assert!(r.to_lowercase().contains("destructive")),
            v => panic!("expected Blocked, got {v:?}"),
        }
    }

    #[test]
    fn safe_terminal_is_allowed() {
        let mut a = action("terminal");
        a.terminal_command = Some("find ~ -type f -size +500M");
        assert_eq!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Allowed
        );
    }

    #[test]
    fn credential_tools_are_blocked() {
        for t in [
            "manage_oauth",
            "manage_config",
            "manage_mcp",
            "manage_http_auth",
        ] {
            assert!(
                matches!(
                    classify_action(&action(t), &ctx_with(vec![])),
                    ActionVerdict::Blocked(_)
                ),
                "{t} should be blocked"
            );
        }
    }

    #[test]
    fn mutating_external_blocked_without_intended_account() {
        // share_memory with no intended accounts.
        assert!(matches!(
            classify_action(&action("share_memory"), &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
        // http POST with no intended accounts.
        let mut a = action("http_request");
        a.http_method = Some("POST");
        a.external_target = Some("api.twitter.com");
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn mutating_external_allowed_for_intended_account() {
        let mut a = action("http_request");
        a.http_method = Some("POST");
        a.external_target = Some("twitter");
        let ctx = ctx_with(vec![IntendedAccount {
            provider: "twitter".to_string(),
            account_id: "acct-123".to_string(),
            account_label: "AcmeBot".to_string(),
        }]);
        assert_eq!(classify_action(&a, &ctx), ActionVerdict::Allowed);
    }

    #[test]
    fn read_only_external_is_allowed() {
        let mut a = action("http_request");
        a.http_method = Some("GET");
        a.external_target = Some("status.example.com");
        assert_eq!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Allowed
        );
        assert_eq!(
            classify_action(&action("web_search"), &ctx_with(vec![])),
            ActionVerdict::Allowed
        );
    }

    #[test]
    fn delegation_tools_are_allowed_per_owner_policy() {
        assert_eq!(
            classify_action(&action("cli_agent"), &ctx_with(vec![])),
            ActionVerdict::Allowed
        );
        assert_eq!(
            classify_action(&action("spawn_agent"), &ctx_with(vec![])),
            ActionVerdict::Allowed
        );
    }

    #[test]
    fn unremarkable_tool_is_allowed_by_default() {
        assert_eq!(
            classify_action(&action("read_file"), &ctx_with(vec![])),
            ActionVerdict::Allowed
        );
    }
}
