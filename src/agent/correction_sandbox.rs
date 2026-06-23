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
    /// The project/task directory that correction is allowed to inspect for
    /// local reads. The upstream caller must set this explicitly — do not fall
    /// back to `$HOME`, the daemon process cwd, or a model-inferred directory
    /// for unattended correction.
    pub working_dir: std::path::PathBuf,
}

/// The fields of a proposed tool action the classifier and default-deny gate
/// need. Produced by `extract_proposed_action`; can also be constructed
/// directly in tests.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct ProposedAction {
    pub tool_name: String,
    /// For `terminal`/`run_command`: the `action` field (e.g. "run",
    /// "background", "kill"). Defaults to `"run"` when absent.
    pub terminal_action: Option<String>,
    /// For `terminal`/`run_command`: the shell command string.
    pub terminal_command: Option<String>,
    /// For `terminal`/`run_command`: any network host/URL visible in the
    /// command text (simple heuristic; not a substitute for firewall rules).
    pub terminal_network_target: Option<String>,
    /// Local filesystem paths the action would read or write.
    pub local_paths: Vec<String>,
    /// Whether the action was requested to run detached/in-background.
    pub detach: bool,
    /// For `http_request`/`web_fetch`: the HTTP method (e.g. "GET", "POST").
    pub http_method: Option<String>,
    /// External target (host/account/provider) the action would touch, if any.
    pub external_target: Option<String>,
    /// Named auth profile in use for this call.
    pub auth_profile: Option<String>,
    /// Whether a raw `Authorization:` or `X-Api-Key:` header was supplied.
    pub auth_header_present: bool,
    // --- capability / semantic flags (from ToolCapabilities / ToolCallSemantics) ---
    pub needs_approval: bool,
    pub read_only: bool,
    pub mutates_state: bool,
    pub external_side_effect: bool,
    pub high_impact_write: bool,
    /// True when `capabilities` was `None` — the default-deny gate must treat
    /// this as potentially mutating/approval-required.
    pub unknown_capabilities: bool,
}

/// Classifier verdict. `Blocked` carries a human-readable reason (surfaced in
/// the attempt ledger and logs; never an approval prompt).
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

/// URL/hostname pattern: looks for `scheme://host` or bare hostnames with a dot.
fn extract_network_target(cmd: &str) -> Option<String> {
    // Find `scheme://` and extract up to the next `/` or whitespace.
    if let Some(idx) = cmd.find("://") {
        let after_scheme = &cmd[idx + 3..];
        let end = after_scheme
            .find(|c: char| c == '/' || c.is_whitespace())
            .unwrap_or(after_scheme.len());
        let host = after_scheme[..end].trim();
        if !host.is_empty() {
            return Some(host.to_string());
        }
    }
    // Bare hostnames like `api.example.com` (must contain a dot, no spaces).
    for token in cmd.split_whitespace() {
        let stripped = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '-');
        if stripped.contains('.')
            && !stripped.starts_with('.')
            && !stripped.ends_with('.')
            && stripped
                .chars()
                .all(|c| c.is_alphanumeric() || c == '.' || c == '-')
        {
            return Some(stripped.to_string());
        }
    }
    None
}

/// Parse a JSON args string, returning `Value::Null` on failure.
fn parse_args(args_json: &str) -> serde_json::Value {
    serde_json::from_str(args_json).unwrap_or(serde_json::Value::Null)
}

/// Returns the string value at `key` in a JSON object `Value`, or `None`.
fn json_str<'a>(v: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.as_str())
}

/// Returns the bool value at `key` in a JSON object `Value`, or `false`.
fn json_bool(v: &serde_json::Value, key: &str) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(false)
}

/// Pure extractor. Fills a `ProposedAction` from the tool name, its JSON
/// arguments, and optional capability/semantic metadata. No I/O or async.
#[allow(dead_code)]
pub fn extract_proposed_action(
    tool_name: &str,
    args_json: &str,
    capabilities: Option<&crate::traits::ToolCapabilities>,
    semantics: Option<&crate::traits::ToolCallSemantics>,
) -> ProposedAction {
    let args = parse_args(args_json);
    let tool_lc = tool_name.to_ascii_lowercase();

    // Populate capability / semantic flags.
    let unknown_capabilities = capabilities.is_none();
    let (read_only, needs_approval, external_side_effect, high_impact_write) =
        if let Some(caps) = capabilities {
            (
                caps.read_only,
                caps.needs_approval,
                caps.external_side_effect,
                caps.high_impact_write,
            )
        } else {
            // Unknown: apply conservative defaults (treat as approval-required,
            // possibly mutating).
            (false, true, false, false)
        };
    let mutates_state = semantics.map(|s| s.mutates_state()).unwrap_or(false);

    let mut action = ProposedAction {
        tool_name: tool_name.to_string(),
        unknown_capabilities,
        read_only,
        needs_approval,
        external_side_effect,
        high_impact_write,
        mutates_state,
        ..ProposedAction::default()
    };

    match tool_lc.as_str() {
        "terminal" | "run_command" => {
            // Normalize missing `action` to "run".
            let terminal_action = json_str(&args, "action").unwrap_or("run").to_string();
            let detach = json_bool(&args, "detach")
                || terminal_action == "background"
                || terminal_action == "detach";

            let terminal_command = json_str(&args, "command").map(str::to_string);
            let network_target = terminal_command.as_deref().and_then(extract_network_target);

            action.terminal_action = Some(terminal_action);
            action.terminal_command = terminal_command;
            action.terminal_network_target = network_target;
            action.detach = detach;
        }

        "read_file" => {
            if let Some(path) = json_str(&args, "path") {
                action.local_paths.push(path.to_string());
            }
        }

        "search_files" => {
            // Support several possible field names used across versions.
            for key in &["path", "directory", "dir"] {
                if let Some(path) = json_str(&args, key) {
                    action.local_paths.push(path.to_string());
                    break;
                }
            }
        }

        "http_request" | "web_fetch" => {
            action.http_method = json_str(&args, "method")
                .or_else(|| json_str(&args, "http_method"))
                .map(|m| m.to_uppercase());

            // External target from `url`, `host`, or `base_url`.
            action.external_target = json_str(&args, "url")
                .or_else(|| json_str(&args, "host"))
                .or_else(|| json_str(&args, "base_url"))
                .map(|u| {
                    // Strip to hostname if URL.
                    if let Some(idx) = u.find("://") {
                        let after = &u[idx + 3..];
                        let end = after
                            .find(['/', '?', '#'])
                            .unwrap_or(after.len());
                        after[..end].to_string()
                    } else {
                        u.to_string()
                    }
                });

            // Named auth profile.
            action.auth_profile = json_str(&args, "auth_profile")
                .or_else(|| json_str(&args, "profile"))
                .map(str::to_string);

            // Structural auth-header signal: any `headers` map containing
            // Authorization or X-Api-Key keys.
            if let Some(headers) = args.get("headers").and_then(|h| h.as_object()) {
                let has_auth = headers.keys().any(|k| {
                    let kl = k.to_ascii_lowercase();
                    kl == "authorization" || kl == "x-api-key"
                });
                action.auth_header_present = has_auth;
            }
        }

        _ => {
            // Unknown tool: rely entirely on capability/semantic flags;
            // precise per-field extraction is deferred to later tasks.
        }
    }

    action
}

/// Produces a stable string that uniquely-enough identifies a proposed action
/// for dedup / attempt ledger purposes. Includes only safety-relevant fields.
/// Secrets are redacted.
#[allow(dead_code)]
pub fn normalized_attempt_signature(action: &ProposedAction) -> String {
    let tool = action.tool_name.to_ascii_lowercase();

    let cmd_part = action
        .terminal_command
        .as_deref()
        .map(|c| {
            // Collapse whitespace for normalization.
            let normalized = c.split_whitespace().collect::<Vec<_>>().join(" ");
            crate::tools::sanitize::redact_secrets(&normalized)
        })
        .unwrap_or_default();

    let method_part = action.http_method.as_deref().unwrap_or("").to_string();

    let host_part = action
        .external_target
        .as_deref()
        .unwrap_or(action.terminal_network_target.as_deref().unwrap_or(""))
        .to_string();

    let auth_profile_present = action.auth_profile.is_some();

    // Coarse path scope: empty, single root, or "multi".
    let path_scope = match action.local_paths.len() {
        0 => "none".to_string(),
        1 => {
            let p = std::path::Path::new(&action.local_paths[0]);
            p.components()
                .next()
                .map(|c| format!("{}", c.as_os_str().to_string_lossy()))
                .unwrap_or_else(|| action.local_paths[0].clone())
        }
        _ => "multi".to_string(),
    };

    format!(
        "tool={tool} cmd={cmd_part} method={method_part} host={host_part} \
         auth_profile={auth_profile_present} auth_header={} \
         detach={} path_scope={path_scope} \
         mutating={} external={} high_impact={}",
        action.auth_header_present,
        action.detach,
        action.mutates_state,
        action.external_side_effect,
        action.high_impact_write,
    )
}

/// Pure correction-mode action classifier. See module docs + Plan 2b for policy.
#[allow(dead_code)]
pub fn classify_action(action: &ProposedAction, ctx: &CorrectionSubjectContext) -> ActionVerdict {
    // Normalize tool name once so all comparisons below are case-insensitive.
    let tool = action.tool_name.to_ascii_lowercase();
    let tool = tool.as_str();

    // (1) Destructive terminal commands.
    if matches!(tool, "terminal" | "run_command") {
        if let Some(cmd) = action.terminal_command.as_deref() {
            if let Some(reason) = crate::tools::command_risk::hard_block_reason(cmd) {
                return ActionVerdict::Blocked(format!("destructive command: {reason}"));
            }
        }
    }

    // (2) Credential / config management.
    if CREDENTIAL_TOOLS.contains(&tool) {
        return ActionVerdict::Blocked(
            "credential/config management is not allowed during autonomous correction".to_string(),
        );
    }

    // (3) Mutating external writes — allowed only for an intended account/target.
    let is_mutating_external = MUTATING_EXTERNAL_TOOLS.contains(&tool)
        || (matches!(tool, "http_request" | "web_fetch")
            && action
                .http_method
                .as_deref()
                .map(|m| MUTATING_HTTP_METHODS.contains(&m.to_uppercase().as_str()))
                .unwrap_or(false));
    if is_mutating_external {
        let target = action.external_target.as_deref().unwrap_or("");
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
    use crate::traits::{ToolCallEffect, ToolCallSemantics, ToolCapabilities};

    // -----------------------------------------------------------------------
    // Helpers shared by old and new tests
    // -----------------------------------------------------------------------

    fn ctx_with(accounts: Vec<IntendedAccount>) -> CorrectionSubjectContext {
        CorrectionSubjectContext {
            subject_id: "s".to_string(),
            subject_kind: SelfCorrectionSubjectKind::Task,
            session_id: "sess".to_string(),
            original_request: "do the thing".to_string(),
            completion_contract_summary: "".to_string(),
            intended_accounts: accounts,
            allowed_external_targets: vec![],
            working_dir: std::path::PathBuf::from("/tmp/test-workdir"),
        }
    }

    fn action(tool: &str) -> ProposedAction {
        ProposedAction {
            tool_name: tool.to_string(),
            ..ProposedAction::default()
        }
    }

    fn read_only_caps() -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn mutating_caps() -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }

    // -----------------------------------------------------------------------
    // Legacy tests (unchanged logic — updated for owned ProposedAction)
    // -----------------------------------------------------------------------

    #[test]
    fn tool_class_constants_are_populated() {
        assert!(CREDENTIAL_TOOLS.contains(&"manage_oauth"));
        assert!(CREDENTIAL_TOOLS.contains(&"manage_config"));
        assert!(MUTATING_EXTERNAL_TOOLS.contains(&"share_memory"));
        assert!(MUTATING_HTTP_METHODS.contains(&"POST"));
        assert!(!MUTATING_HTTP_METHODS.contains(&"GET"));
    }

    #[test]
    fn destructive_terminal_is_blocked() {
        let mut a = action("terminal");
        a.terminal_command = Some("rm -rf /".to_string());
        match classify_action(&a, &ctx_with(vec![])) {
            ActionVerdict::Blocked(r) => assert!(r.to_lowercase().contains("destructive")),
            v => panic!("expected Blocked, got {v:?}"),
        }
    }

    #[test]
    fn safe_terminal_is_allowed() {
        let mut a = action("terminal");
        a.terminal_command = Some("find ~ -type f -size +500M".to_string());
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
        a.http_method = Some("POST".to_string());
        a.external_target = Some("api.twitter.com".to_string());
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn mutating_external_allowed_for_intended_account() {
        let mut a = action("http_request");
        a.http_method = Some("POST".to_string());
        a.external_target = Some("twitter".to_string());
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
        a.http_method = Some("GET".to_string());
        a.external_target = Some("status.example.com".to_string());
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

    #[test]
    fn mutating_external_blocked_for_present_but_nonmatching_target() {
        let mut a = action("http_request");
        a.http_method = Some("POST".to_string());
        a.external_target = Some("evil.com".to_string());
        let ctx = ctx_with(vec![IntendedAccount {
            provider: "twitter".to_string(),
            account_id: "acct-123".to_string(),
            account_label: "AcmeBot".to_string(),
        }]);
        assert!(matches!(
            classify_action(&a, &ctx),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn mutating_method_check_is_case_insensitive() {
        let mut a = action("http_request");
        a.http_method = Some("post".to_string());
        a.external_target = Some("api.example.com".to_string());
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn tool_name_matching_is_case_insensitive() {
        let mut a = action("Terminal");
        a.terminal_command = Some("rm -rf /".to_string());
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    // -----------------------------------------------------------------------
    // New P1.1 tests (TDD — written before implementation)
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_terminal_command_and_detach() {
        let caps = read_only_caps();
        let sem = ToolCallSemantics::observation();
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cargo test","detach":false}"#,
            Some(&caps),
            Some(&sem),
        );
        assert_eq!(a.tool_name, "terminal");
        assert_eq!(a.terminal_command.as_deref(), Some("cargo test"));
        assert!(!a.detach);
        // Capability flags copied in.
        assert!(a.read_only);
        assert!(!a.needs_approval);
        assert!(!a.unknown_capabilities);
    }

    #[test]
    fn test_extract_terminal_command_detach_flag() {
        let caps = mutating_caps();
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"./server","detach":true}"#,
            Some(&caps),
            None,
        );
        assert!(a.detach);
        assert_eq!(a.terminal_command.as_deref(), Some("./server"));
    }

    #[test]
    fn test_extract_terminal_action() {
        let caps = mutating_caps();
        let a = extract_proposed_action(
            "terminal",
            r#"{"action":"kill","command":"42"}"#,
            Some(&caps),
            None,
        );
        assert_eq!(a.terminal_action.as_deref(), Some("kill"));
    }

    #[test]
    fn test_terminal_missing_action_defaults_to_run_for_classification() {
        // Brief requirement: `{"command":"pwd"}` must classify as
        // `terminal_action = Some("run")`, not missing/non-run.
        let caps = read_only_caps();
        let a = extract_proposed_action("terminal", r#"{"command":"pwd"}"#, Some(&caps), None);
        assert_eq!(
            a.terminal_action.as_deref(),
            Some("run"),
            "missing action should default to 'run'"
        );
    }

    #[test]
    fn test_extract_run_command_command() {
        let caps = mutating_caps();
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"npm install"}"#,
            Some(&caps),
            None,
        );
        assert_eq!(a.terminal_command.as_deref(), Some("npm install"));
        assert_eq!(a.terminal_action.as_deref(), Some("run"));
    }

    #[test]
    fn test_extract_terminal_network_target() {
        let caps = mutating_caps();
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"curl https://api.example.com/data"}"#,
            Some(&caps),
            None,
        );
        assert_eq!(
            a.terminal_network_target.as_deref(),
            Some("api.example.com"),
            "should extract hostname from curl URL"
        );
    }

    #[test]
    fn test_extract_http_auth_profile_and_host() {
        let caps = ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        };
        let sem = ToolCallSemantics {
            effect: ToolCallEffect::Mutation,
            ..ToolCallSemantics::default()
        };
        let a = extract_proposed_action(
            "http_request",
            r#"{"method":"POST","url":"https://api.github.com/repos","auth_profile":"github-pat"}"#,
            Some(&caps),
            Some(&sem),
        );
        assert_eq!(a.http_method.as_deref(), Some("POST"));
        assert_eq!(a.external_target.as_deref(), Some("api.github.com"));
        assert_eq!(a.auth_profile.as_deref(), Some("github-pat"));
        assert!(a.mutates_state);
        assert!(a.external_side_effect);
    }

    #[test]
    fn test_extract_http_auth_headers_are_structural_signals() {
        let caps = ToolCapabilities::default();
        let a = extract_proposed_action(
            "http_request",
            r#"{"method":"GET","url":"https://secure.example.com","headers":{"Authorization":"Bearer tok123"}}"#,
            Some(&caps),
            None,
        );
        assert!(
            a.auth_header_present,
            "Authorization header must set auth_header_present"
        );
    }

    #[test]
    fn test_extract_read_file_path() {
        let caps = ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        };
        let sem = ToolCallSemantics::observation();
        let a = extract_proposed_action(
            "read_file",
            r#"{"path":"/workspace/src/main.rs"}"#,
            Some(&caps),
            Some(&sem),
        );
        assert_eq!(a.local_paths, vec!["/workspace/src/main.rs".to_string()]);
    }

    #[test]
    fn test_attempt_signature_normalization() {
        let a = ProposedAction {
            tool_name: "terminal".to_string(),
            terminal_action: Some("run".to_string()),
            terminal_command: Some("cargo   test".to_string()),
            mutates_state: false,
            external_side_effect: false,
            high_impact_write: false,
            ..ProposedAction::default()
        };
        let sig = normalized_attempt_signature(&a);
        // Whitespace in cmd should be collapsed.
        assert!(
            sig.contains("cargo test"),
            "signature should normalize whitespace: {sig}"
        );
        assert!(sig.contains("tool=terminal"), "tool name in sig: {sig}");
        assert!(
            sig.contains("mutating=false"),
            "mutating flag in sig: {sig}"
        );
    }

    #[test]
    fn test_extract_unknown_tool_capabilities_drive_default_deny_fields() {
        // When capabilities is None, unknown_capabilities must be true and
        // needs_approval must be true (conservative default).
        let a = extract_proposed_action("some_future_tool", r#"{}"#, None, None);
        assert!(a.unknown_capabilities, "unknown caps when None provided");
        assert!(
            a.needs_approval,
            "must default to needs_approval=true when caps unknown"
        );
        assert!(
            !a.read_only,
            "must default to read_only=false when caps unknown"
        );
    }
}
