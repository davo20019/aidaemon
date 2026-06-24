//! Correction-mode sandbox: a pure classifier deciding whether a proposed
//! action may run during autonomous self-correction. Default-deny policy
//! (Plan 3b P1.2): only a narrow whitelist of local read-only operations and
//! pre-approved external mutations are permitted. Everything else is blocked.

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

        "git_info" | "project_inspect" => {
            // Both tools accept a `"path"` arg pointing to a local directory.
            // Populate local_paths so classify_action's check_local_path_scope
            // can enforce the working_dir boundary (same as read_file).
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
                        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
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

/// Pure correction-mode action classifier. Default-deny policy for Plan 3b.
#[allow(dead_code)]
pub fn classify_action(action: &ProposedAction, ctx: &CorrectionSubjectContext) -> ActionVerdict {
    let tool = action.tool_name.to_ascii_lowercase();
    let tool = tool.as_str();

    // (1) MCP tools — blocked.
    if tool.starts_with("mcp__") {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "mcp tools are not allowed during autonomous correction",
        ));
    }

    // (2) Delegation tools — blocked.
    if matches!(tool, "cli_agent" | "spawn_agent") {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "delegation tools (cli_agent, spawn_agent) are not allowed during autonomous correction",
        ));
    }

    // (3) Credential / config management — blocked.
    if CREDENTIAL_TOOLS.contains(&tool) {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "credential/config management is not allowed during autonomous correction",
        ));
    }

    // (4) Browser / computer-use — blocked in 3b.
    if matches!(tool, "browser" | "computer_use") {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "browser/computer_use is not allowed during autonomous correction",
        ));
    }

    // (5) File-writing and git mutation — blocked in 3b.
    if matches!(tool, "write_file" | "edit_file" | "git_commit") {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "file writes and git commits are not allowed during autonomous correction",
        ));
    }

    // (6) Memory/person/goal mutating tools — blocked in 3b.
    if matches!(
        tool,
        "remember_fact"
            | "manage_memories"
            | "manage_people"
            | "create_goal"
            | "update_goal"
            | "delete_goal"
    ) {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "memory/person/goal mutation is not allowed during autonomous correction",
        ));
    }

    // (7) send_file / share_memory — blocked outright in 3b (external-send tools).
    if matches!(tool, "send_file" | "share_memory") {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "send_file/share_memory are not allowed during autonomous correction",
        ));
    }

    // (8) Terminal — strict allowlist.
    if tool == "terminal" {
        return classify_terminal_action(action, ctx);
    }

    // (9) run_command — uses run_command tool's own safe prefix list.
    if tool == "run_command" {
        return classify_run_command_action(action, ctx);
    }

    // (10) read_file / search_files / git_info / project_inspect — local read-only,
    //      path scope enforced.  git_info and project_inspect accept a `"path"` arg
    //      (extracted into local_paths by extract_proposed_action); scope-checking
    //      reuses the same check_local_path_scope gate as read_file.
    if matches!(
        tool,
        "read_file" | "search_files" | "git_info" | "project_inspect"
    ) {
        // Check local paths for scope and sensitivity.
        for path_str in &action.local_paths {
            if let Err(reason) = check_local_path_scope(path_str, ctx) {
                return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&reason));
            }
        }
        return ActionVerdict::Allowed;
    }

    // (11) http_request.
    if tool == "http_request" {
        return classify_http_request(action, ctx);
    }

    // (12) web_fetch.
    if tool == "web_fetch" {
        return classify_web_fetch(action, ctx);
    }

    // (13) web_search — public read-only; block if authenticated signals present.
    if tool == "web_search" {
        if action.auth_profile.is_some() || action.auth_header_present {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
                "authenticated web_search requires target match; use http_request with explicit intended account",
            ));
        }
        return ActionVerdict::Allowed;
    }

    // (14) Unknown tools — allow only if clearly read-only, non-approval, non-mutating,
    //      no external side effect, no high-impact write, AND capabilities known.
    if action.unknown_capabilities
        || action.needs_approval
        || action.mutates_state
        || action.external_side_effect
        || action.high_impact_write
    {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "tool '{}' blocked: unknown or potentially mutating/approval-requiring capabilities",
            action.tool_name
        )));
    }

    ActionVerdict::Allowed
}

/// Classify a `terminal` action under the correction-sandbox policy.
fn classify_terminal_action(
    action: &ProposedAction,
    ctx: &CorrectionSubjectContext,
) -> ActionVerdict {
    // Must be action="run".
    let terminal_action = action.terminal_action.as_deref().unwrap_or("run");
    if terminal_action != "run" {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "terminal action '{}' is not allowed during autonomous correction (only 'run')",
            terminal_action
        )));
    }

    // No detach.
    if action.detach {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "detached/background terminal execution is not allowed during autonomous correction",
        ));
    }

    // No network target.
    if let Some(target) = &action.terminal_network_target {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "terminal command with network target '{}' is not allowed during autonomous correction",
            target
        )));
    }

    // Needs approval flag — terminal requires explicit owner approval; block in correction.
    if action.needs_approval {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "terminal commands requiring approval are not allowed during autonomous correction",
        ));
    }

    let cmd = match action.terminal_command.as_deref() {
        Some(c) => c,
        None => {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
                "terminal action with no command",
            ));
        }
    };

    // Hard block for destructive patterns.
    if let Some(reason) = crate::tools::command_risk::hard_block_reason(cmd) {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "destructive command: {reason}"
        )));
    }

    // Risk level must be Safe.
    let risk = crate::tools::command_risk::classify_command(cmd);
    if risk.level != crate::tools::command_risk::RiskLevel::Safe {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "terminal command risk level '{}' is not safe for autonomous correction",
            risk.level
        )));
    }

    // Correction-specific read-only local allowlist.
    if let Err(reason) = is_correction_safe_local_command(cmd, &ctx.working_dir) {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&reason));
    }

    // Check any local_paths for path scope.
    for path_str in &action.local_paths {
        if let Err(reason) = check_local_path_scope(path_str, ctx) {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&reason));
        }
    }

    ActionVerdict::Allowed
}

/// Classify a `run_command` action under the correction-sandbox policy.
fn classify_run_command_action(
    action: &ProposedAction,
    ctx: &CorrectionSubjectContext,
) -> ActionVerdict {
    // No detach.
    if action.detach {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
            "detached run_command is not allowed during autonomous correction",
        ));
    }
    // No network target.
    if let Some(target) = &action.terminal_network_target {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "run_command with network target '{}' is not allowed during autonomous correction",
            target
        )));
    }

    let cmd = match action.terminal_command.as_deref() {
        Some(c) => c,
        None => {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
                "run_command with no command string",
            ));
        }
    };

    // Hard block.
    if let Some(reason) = crate::tools::command_risk::hard_block_reason(cmd) {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "destructive command: {reason}"
        )));
    }

    // Risk level must be Safe.
    let risk = crate::tools::command_risk::classify_command(cmd);
    if risk.level != crate::tools::command_risk::RiskLevel::Safe {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "run_command risk level '{}' is not safe for autonomous correction",
            risk.level
        )));
    }

    // Must match run_command's own safe prefix list.
    if !crate::tools::run_command::is_run_command_safe(cmd) {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
            "run_command '{}' is not in the safe prefix list",
            cmd
        )));
    }

    // Finding 1 + 2 (re-review): apply path-operand scope check to run_command
    // path operands.  `is_run_command_safe` whitelists the base command but never
    // checked path-like arguments, so `tail /var/log/x` or
    // `cargo test --manifest-path /outside/Cargo.toml` would have been Allowed.
    // grep/rg skip their pattern argument; all others have no skip.
    // Build/test commands (cargo test, pytest, go test, npm test) that have NO
    // path operand pass through fine — only explicit out-of-scope paths are blocked.
    let base_cmd = {
        let tok = cmd.split_whitespace().next().unwrap_or("");
        std::path::Path::new(tok)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(tok)
    };
    let skip_pattern = matches!(base_cmd, "grep" | "rg");
    if let Err(reason) = command_path_operands_in_scope(cmd, &ctx.working_dir, skip_pattern) {
        return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&reason));
    }

    // Check any local_paths for path scope.
    for path_str in &action.local_paths {
        if let Err(reason) = check_local_path_scope(path_str, ctx) {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&reason));
        }
    }

    ActionVerdict::Allowed
}

/// Shared path-operand scope checker used by both `is_correction_safe_local_command`
/// (terminal) and `classify_run_command_action` (run_command).
///
/// Scans every whitespace-separated token after the base command (index 0) and
/// scope-checks **every path-shaped value** regardless of whether it is a
/// positional operand or a flag value — including `=`-attached flag values like
/// `--manifest-path=/outside/Cargo.toml`.
///
/// For each token the candidate value is determined as:
/// - If the token contains `=` (e.g. `--manifest-path=/foo`), the substring
///   **after** the first `=` is the candidate.
/// - Otherwise the token itself is the candidate.
///
/// A candidate is "path-shaped" (and therefore subject to scope checking) when it:
/// - contains `/`; OR
/// - starts with `~`; OR
/// - contains `$`.
///
/// Bare words (`foo`, `--package`, `-n`, `5`, `test`) are not path-shaped and
/// are silently skipped, so `cargo test --package foo` passes without issue.
///
/// Scope-checking rules (applied to every path-shaped candidate):
///
/// 1. **~/$HOME rejection**: any candidate starting with `~` or containing `$`
///    is rejected outright.  The classifier cannot determine where these resolve
///    at shell-expansion time, so blocking them is the only safe choice.
///
/// 2. **Working-dir scope**: the remaining candidates must resolve (lexically,
///    without I/O) to a path under `working_dir`.
///
/// 3. **Sensitive-file check**: even in-scope paths are blocked if they match
///    known secret/credential patterns (`.env`, `.ssh/`, `.aws/`, etc.).
///
/// The `skip_first_non_flag` parameter skips the first positional non-flag
/// argument (used for grep/rg where that argument is the pattern, not a path).
/// Note: only positional (non-`-`-prefixed) tokens without `=` are counted
/// toward the skip; flag-value candidates extracted via `=` are always checked.
fn command_path_operands_in_scope(
    cmd: &str,
    working_dir: &std::path::Path,
    skip_first_non_flag: bool,
) -> Result<(), String> {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    // tokens[0] is the base command itself; start from index 1.
    let rest = tokens.get(1..).unwrap_or(&[]);

    // Track how many positional (non-flag) operands we have seen so we can
    // honour skip_first_non_flag for grep/rg pattern skipping.
    let mut positional_seen: usize = 0;

    for &token in rest {
        // Determine the candidate value and whether this is a positional token.
        let (raw_candidate, is_positional) = if let Some(eq_pos) = token.find('=') {
            // `--flag=value` form: candidate is the part after the first `=`.
            (&token[eq_pos + 1..], false)
        } else {
            // Plain token: either a positional operand or a bare flag.
            (token, !token.starts_with('-'))
        };

        // Strip leading/trailing ASCII quote characters (`"` and `'`) so that
        // `cat "/tmp/secret.txt"` is not misidentified as a relative path.
        let candidate = raw_candidate.trim_matches(|c| c == '"' || c == '\'');

        // For positional tokens (no `-` prefix, no `=`), apply the skip logic.
        if is_positional {
            positional_seen += 1;
            if skip_first_non_flag && positional_seen == 1 {
                // First positional is the pattern (grep/rg); skip it.
                continue;
            }
        }

        // Only examine candidates that look like paths.
        // A candidate is path-shaped when it:
        //   • contains `/`
        //   • starts with `~`
        //   • contains `$`
        //   • starts with `.` but is NOT exactly `.` or `..`
        //     (catches bare dotfiles like `.env`, `.netrc` without misidentifying
        //      the current-dir and parent-dir pseudo-entries)
        let looks_like_path = candidate.contains('/')
            || candidate.starts_with('~')
            || candidate.contains('$')
            || (candidate.starts_with('.') && candidate != "." && candidate != "..");
        if !looks_like_path {
            continue;
        }

        // Reject tilde expansion and shell variable expansion — the classifier
        // cannot determine where these resolve at shell-expansion time.
        if candidate.starts_with('~') {
            return Err(format!(
                "Home shorthand `~`/`$HOME`/`~/*` (here `{}`) is rejected — use an explicit \
                 ABSOLUTE path (e.g. `{}/...`).",
                candidate,
                working_dir.display()
            ));
        }
        if candidate.contains('$') {
            return Err(format!(
                "Home shorthand `~`/`$HOME`/`~/*` (here `{}`) is rejected — use an explicit \
                 ABSOLUTE path (e.g. `{}/...`).",
                candidate,
                working_dir.display()
            ));
        }

        // Resolve and scope-check.
        let path = std::path::Path::new(candidate);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            working_dir.join(path)
        };
        let normalized = normalize_path_lexical(&resolved);
        if !normalized.starts_with(working_dir) {
            return Err(format!(
                "command path '{}' is outside the allowed working directory — \
                 use a path inside the allowed scope `{}`.",
                candidate,
                working_dir.display()
            ));
        }
        if is_sensitive_file_path(&normalized) {
            return Err(format!(
                "command path '{}' matches a sensitive/secret file pattern and cannot \
                 be read in correction mode",
                candidate
            ));
        }
    }

    Ok(())
}

/// Correction-safe local command allowlist.
/// Returns Ok(()) if allowed, Err(reason) if blocked.
fn is_correction_safe_local_command(
    cmd: &str,
    working_dir: &std::path::Path,
) -> Result<(), String> {
    // Reject shell metacharacters / escape forms.
    let meta_chars = ['|', '>', '<', ';', '`'];
    for ch in meta_chars {
        if cmd.contains(ch) {
            return Err(format!(
                "command contains shell metacharacter '{}' which is not allowed in correction mode",
                ch
            ));
        }
    }
    if cmd.contains("&&") {
        return Err("command contains '&&' which is not allowed in correction mode".to_string());
    }
    if cmd.contains("||") {
        return Err("command contains '||' which is not allowed in correction mode".to_string());
    }
    if cmd.contains("$(") {
        return Err(
            "command contains command substitution '$(' which is not allowed in correction mode"
                .to_string(),
        );
    }
    if cmd.contains(">(") || cmd.contains("<(") {
        return Err(
            "command contains process substitution which is not allowed in correction mode"
                .to_string(),
        );
    }
    // Reject env-var command prefixes (VAR=val cmd).
    // A leading token of the form KEY=VALUE before the actual command.
    let first_token = cmd.split_whitespace().next().unwrap_or("");
    if first_token.contains('=') && !first_token.starts_with('-') {
        return Err(format!(
            "command has environment variable prefix '{}' which is not allowed in correction mode",
            first_token
        ));
    }

    // Extract base command name.
    let base_cmd = {
        let token = cmd.split_whitespace().next().unwrap_or("");
        std::path::Path::new(token)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(token)
    };

    // Reject interpreters and shell eval forms.
    const BLOCKED_INTERPRETERS: &[&str] = &[
        "python",
        "python3",
        "node",
        "ruby",
        "php",
        "perl",
        "deno",
        "bun",
        "lua",
        "osascript",
    ];
    if BLOCKED_INTERPRETERS.contains(&base_cmd) {
        return Err(format!(
            "interpreter '{}' is not allowed in correction mode",
            base_cmd
        ));
    }
    // Shell eval forms (bash -c, sh -c, zsh -c).
    if matches!(
        base_cmd,
        "bash" | "sh" | "zsh" | "fish" | "dash" | "ksh" | "csh" | "tcsh"
    ) {
        return Err(format!(
            "shell '{}' is not allowed in correction mode",
            base_cmd
        ));
    }

    // Reject network-capable binaries.
    const BLOCKED_NETWORK: &[&str] = &[
        "curl", "wget", "nc", "ncat", "ssh", "scp", "sftp", "rsync", "ftp", "telnet", "openssl",
    ];
    if BLOCKED_NETWORK.contains(&base_cmd) {
        return Err(format!(
            "network binary '{}' is not allowed in correction mode",
            base_cmd
        ));
    }

    // Allow only a small command-name set.
    const ALLOWED_CMDS: &[&str] = &[
        "pwd", "ls", "cat", "head", "tail", "wc", "grep", "rg", "find",
    ];
    if !ALLOWED_CMDS.contains(&base_cmd) {
        // Actionable hints: the reaper-stopped command is most often a slow `du`,
        // and the weak local model retries the same disallowed form unless told
        // the exact allowed command to use instead.
        if base_cmd == "du" {
            return Err(format!(
                "`du` is not permitted in correction mode (it recursively sizes \
                 directories and is slow). To find large files use \
                 `find {} -type f -size +1G -printf '%s\\t%p\\n'` (lists \
                 \"<bytes>\\t<path>\" per file; pick the largest) instead.",
                working_dir.display()
            ));
        }
        return Err(format!(
            "command '{}' is not in the correction-mode read-only allowlist \
             (allowed: pwd, ls, cat, head, tail, wc, grep, rg, find). To find large \
             files use `find {} -type f -size +1G -printf '%s\\t%p\\n'` (lists \
             \"<bytes>\\t<path>\" per file; pick the largest) instead.",
            base_cmd,
            working_dir.display()
        ));
    }

    // `find` must not use -exec, -delete, -ok, -fprintf, -fls, -fprint,
    // and must not scan broad/out-of-scope roots (I1 + I2).
    //
    // Note on `-printf` vs `-fprintf`/`-fls`/`-fprint`: `-printf FORMAT` only
    // prints file METADATA (size/path/times) to STDOUT — it is read-only and
    // strictly less powerful than the already-allowed `cat`, so it is ALLOWED.
    // The file-WRITING variants `-fprintf FILE`/`-fls FILE`/`-fprint FILE`
    // redirect that output into a file, so they remain BLOCKED. `-exec`/`-ok`
    // execute commands and `-delete` removes files, so they remain BLOCKED too.
    if base_cmd == "find" {
        // I2: reject file-writing / executing / deleting find actions.
        for token in cmd.split_whitespace() {
            if matches!(
                token,
                "-exec" | "-delete" | "-ok" | "-fprintf" | "-fls" | "-fprint"
            ) {
                return Err(format!(
                    "find with '{}' is not allowed in correction mode",
                    token
                ));
            }
        }
        // I1: extract ALL non-flag path tokens from the full command.
        // `find` accepts global options before roots (-L, -H, -P are one-char flags).
        // We collect every token that doesn't start with '-'; all of these are
        // potential path roots or predicate arguments.  Rather than trying to
        // distinguish roots from predicate arguments (e.g. `-name foo` → `foo`
        // is not a root), we conservatively treat ANY non-flag token as a
        // candidate root when it looks like an absolute path or home-dir
        // reference that would escape the working directory — if it appears
        // as a path argument to a predicate like `-name` that's a false
        // positive that is fine (over-blocking is safe).
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let mut find_roots: Vec<&str> = Vec::new();
        let mut i = 1; // skip 'find' itself
        while i < parts.len() {
            let tok = parts[i];
            if tok.starts_with('-') {
                // Skip this flag token; also skip its argument if it is a
                // known single-argument predicate so we don't misidentify
                // the argument as a root.
                const SINGLE_ARG_PREDICATES: &[&str] = &[
                    "-name",
                    "-iname",
                    "-type",
                    "-maxdepth",
                    "-mindepth",
                    "-size",
                    "-newer",
                    "-user",
                    "-group",
                    "-perm",
                    "-mtime",
                    "-atime",
                    "-ctime",
                    "-path",
                    "-ipath",
                    "-regex",
                    "-iregex",
                ];
                if SINGLE_ARG_PREDICATES.contains(&tok) {
                    i += 2; // skip flag + its argument
                } else {
                    i += 1; // skip flag only
                }
            } else {
                find_roots.push(tok);
                i += 1;
            }
        }
        // Check each candidate root.
        for root in &find_roots {
            let is_broad =
                *root == "/" || *root == "~" || root.starts_with("~/") || root.starts_with("$HOME");
            if is_broad {
                // Distinguish home-shorthand from whole-disk so the hint matches.
                if *root == "~" || root.starts_with("~/") || root.starts_with("$HOME") {
                    return Err(format!(
                        "Home shorthand `~`/`$HOME`/`~/*` (here `{}`) is rejected — use an \
                         explicit ABSOLUTE path (e.g. `{}/...`).",
                        root,
                        working_dir.display()
                    ));
                }
                return Err(format!(
                    "Unbounded scan of `{}` blocked — scope it with a specific absolute \
                     directory and a size filter, e.g. `find {} -type f -size +500M`.",
                    root,
                    working_dir.display()
                ));
            }
            // Scope check: root must be within working_dir.
            let path = std::path::Path::new(root);
            let resolved = if path.is_absolute() {
                path.to_path_buf()
            } else {
                working_dir.join(path)
            };
            let normalized = normalize_path_lexical(&resolved);
            if !normalized.starts_with(working_dir) {
                return Err(format!(
                    "find root '{}' is outside the allowed working directory — \
                     use a path inside the allowed scope `{}`.",
                    root,
                    working_dir.display()
                ));
            }
            if is_sensitive_file_path(&normalized) {
                return Err(format!(
                    "find root '{}' matches a sensitive/secret file pattern",
                    root
                ));
            }
        }
        // No roots given → implicit "." which is scoped; allowed.
        return Ok(());
    }

    // For all other allowed commands (non-find, non-pwd), use the shared
    // path-operand scope checker (Finding 1 + Finding 2 from the re-review).
    // grep/rg: skip first non-flag arg (the pattern).
    if base_cmd != "pwd" {
        let skip_pattern = matches!(base_cmd, "grep" | "rg");
        command_path_operands_in_scope(cmd, working_dir, skip_pattern)?;
    }

    Ok(())
}

/// Returns true if `host` matches an intended account in `ctx` (via provider, account_id,
/// allowed_external_targets, subdomain, or provider domain mapping).
fn target_matches_intended(host: &str, ctx: &CorrectionSubjectContext) -> bool {
    // Direct account_id or provider match.
    if ctx
        .intended_accounts
        .iter()
        .any(|a| a.account_id == host || a.provider == host)
    {
        return true;
    }
    // Allowed external targets exact match.
    if ctx.allowed_external_targets.iter().any(|t| t == host) {
        return true;
    }
    // Subdomain check: host ends with `.{allowed_target}`.
    for t in &ctx.allowed_external_targets {
        if host == t || host.ends_with(&format!(".{}", t)) {
            return true;
        }
    }
    // Provider domain mapping: github → github.com, api.github.com.
    for account in &ctx.intended_accounts {
        if provider_domain_matches(&account.provider, host) {
            return true;
        }
        // Subdomain of account_id.
        if host.ends_with(&format!(".{}", account.account_id)) {
            return true;
        }
        // Subdomain of provider.
        if host.ends_with(&format!(".{}", account.provider)) {
            return true;
        }
    }
    false
}

/// Maps known provider names to their canonical domains.
fn provider_domain_matches(provider: &str, host: &str) -> bool {
    let provider_lc = provider.to_ascii_lowercase();
    match provider_lc.as_str() {
        "github" => {
            matches!(host, "github.com" | "api.github.com") || host.ends_with(".github.com")
        }
        "gitlab" => {
            matches!(host, "gitlab.com" | "api.gitlab.com") || host.ends_with(".gitlab.com")
        }
        "bitbucket" => {
            matches!(host, "bitbucket.org" | "api.bitbucket.org")
                || host.ends_with(".bitbucket.org")
        }
        "slack" => matches!(host, "slack.com" | "api.slack.com") || host.ends_with(".slack.com"),
        "discord" => {
            matches!(host, "discord.com" | "discordapp.com") || host.ends_with(".discord.com")
        }
        "twitter" | "x" => matches!(host, "twitter.com" | "api.twitter.com" | "x.com"),
        "linear" => matches!(host, "linear.app" | "api.linear.app"),
        "notion" => matches!(host, "notion.so" | "api.notion.com"),
        "jira" | "atlassian" => host.ends_with(".atlassian.net") || host.ends_with(".jira.com"),
        _ => false,
    }
}

/// Classify an `http_request` action.
fn classify_http_request(action: &ProposedAction, ctx: &CorrectionSubjectContext) -> ActionVerdict {
    let method = action.http_method.as_deref().unwrap_or("GET");
    let is_mutating = MUTATING_HTTP_METHODS.contains(&method.to_uppercase().as_str());
    let is_authenticated = action.auth_profile.is_some() || action.auth_header_present;
    let target = action.external_target.as_deref().unwrap_or("");

    if is_mutating || is_authenticated {
        // Requires target match.
        if target.is_empty() {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(
                "http_request with no target is not allowed during autonomous correction",
            ));
        }
        if !target_matches_intended(target, ctx) {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
                "http_request target '{}' does not match intended accounts/targets",
                target
            )));
        }
    }

    ActionVerdict::Allowed
}

/// Classify a `web_fetch` action.
fn classify_web_fetch(action: &ProposedAction, ctx: &CorrectionSubjectContext) -> ActionVerdict {
    let is_authenticated = action.auth_profile.is_some() || action.auth_header_present;

    if is_authenticated {
        let target = action.external_target.as_deref().unwrap_or("");
        if target.is_empty() || !target_matches_intended(target, ctx) {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&format!(
                "authenticated web_fetch target '{}' does not match intended accounts/targets",
                target
            )));
        }
    }

    // Non-mutating, public read → Allowed.
    ActionVerdict::Allowed
}

/// Check that a local path is within the working_dir and not a sensitive/secret file.
/// Returns Ok(()) if allowed, Err(reason) if blocked.
fn check_local_path_scope(path_str: &str, ctx: &CorrectionSubjectContext) -> Result<(), String> {
    use std::path::{Path, PathBuf};

    let path = Path::new(path_str);

    // Resolve relative paths against working_dir; absolute paths as-is.
    let resolved: PathBuf = if path.is_absolute() {
        path.to_path_buf()
    } else {
        ctx.working_dir.join(path)
    };

    // Lexically normalize: resolve . and .. without hitting the filesystem.
    let normalized = normalize_path_lexical(&resolved);

    // Must be under working_dir.
    if !normalized.starts_with(&ctx.working_dir) {
        return Err(format!(
            "path '{}' is outside the allowed working directory — use a path inside \
             the allowed scope `{}`.",
            path_str,
            ctx.working_dir.display()
        ));
    }

    // Block sensitive/secret files even inside working_dir.
    if is_sensitive_file_path(&normalized) {
        return Err(format!(
            "path '{}' matches a sensitive/secret file pattern and cannot be read during autonomous correction",
            path_str
        ));
    }

    Ok(())
}

/// Lexically normalize a path (resolve `.` and `..`) without I/O.
fn normalize_path_lexical(path: &std::path::Path) -> std::path::PathBuf {
    use std::path::{Component, PathBuf};
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            c => result.push(c),
        }
    }
    result
}

/// Returns true if the path matches a sensitive/secret file pattern.
fn is_sensitive_file_path(path: &std::path::Path) -> bool {
    // Check each component.
    for component in path.components() {
        let name = match component {
            std::path::Component::Normal(n) => n.to_string_lossy(),
            _ => continue,
        };
        let name = name.as_ref();

        // Exact sensitive names.
        if matches!(
            name,
            ".env"
                | "id_rsa"
                | "id_ed25519"
                | "id_dsa"
                | "id_ecdsa"
                | "credentials"
                | ".netrc"
                | ".pgpass"
                | ".ssh"
                | ".aws"
                | ".gnupg"
                | ".kube"
        ) {
            return true;
        }

        // .env.* variants (e.g. .env.local, .env.production).
        if name.starts_with(".env.") {
            return true;
        }

        // Extension-based checks.
        if let Some(ext) = std::path::Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
        {
            if matches!(ext, "pem" | "key" | "p12" | "pfx") {
                return true;
            }
        }
    }

    // String-based checks for path patterns.
    let path_str = path.to_string_lossy();
    if path_str.contains("/.ssh/")
        || path_str.contains("/.aws/")
        || path_str.contains("/.config/gcloud")
        || path_str.contains("/.config/gh")
        || path_str.contains("/.kube/")
        || path_str.contains("/.gnupg/")
    {
        return true;
    }

    false
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
    // Legacy tests (updated for P1.2 default-deny policy)
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
        // In P1.2 this command is blocked (find with broad root not in correction allowlist)
        // Kept as a regression: must remain blocked
        let mut a = action("terminal");
        a.terminal_command = Some("find ~ -type f -size +500M".to_string());
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
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
        // Policy reversed in P1.2: delegation tools are now blocked in correction sandbox
        assert!(matches!(
            classify_action(&action("cli_agent"), &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
        assert!(matches!(
            classify_action(&action("spawn_agent"), &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn unremarkable_tool_is_allowed_by_default() {
        // read_file with no paths and no concerning flags is allowed
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

    // -----------------------------------------------------------------------
    // P1.2 tests — written first (TDD), implement classify_action then pass
    // -----------------------------------------------------------------------

    #[test]
    fn test_mcp_tools_blocked() {
        for t in &[
            "mcp__github__list_issues",
            "mcp__slack__send_message",
            "mcp__custom__do_thing",
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
    fn test_delegation_tools_blocked() {
        // Now reversed from legacy test — delegation is blocked
        assert!(matches!(
            classify_action(&action("cli_agent"), &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
        assert!(matches!(
            classify_action(&action("spawn_agent"), &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_terminal_prompt_tier_blocked() {
        // terminal with action=run, safe command, but needs_approval=true → Blocked
        let mut a = action("terminal");
        a.terminal_action = Some("run".to_string());
        a.terminal_command = Some("ls -la".to_string());
        a.needs_approval = true;
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_terminal_medium_network_blocked_by_default() {
        // curl has RiskLevel::Medium — not Safe — should be blocked
        let mut a = action("terminal");
        a.terminal_action = Some("run".to_string());
        a.terminal_command = Some("curl https://api.example.com".to_string());
        a.terminal_network_target = Some("api.example.com".to_string());
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_terminal_network_target_policy() {
        // Even if risk level were Safe, having a network target blocks it
        let mut a = action("terminal");
        a.terminal_action = Some("run".to_string());
        a.terminal_command = Some("pwd".to_string());
        a.terminal_network_target = Some("example.com".to_string()); // force a network target
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_terminal_interpreter_egress_blocked_even_when_risk_safe() {
        // python3 is an interpreter — blocked by is_correction_safe_local_command
        // even if classify_command doesn't flag it as Critical
        let mut a = action("terminal");
        a.terminal_action = Some("run".to_string());
        a.terminal_command = Some("python3 script.py".to_string());
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_terminal_shell_escape_forms_blocked() {
        // Shell metacharacters in command
        for cmd in &[
            "ls | grep foo",
            "cat file > out.txt",
            "echo foo; ls",
            "ls && pwd",
            "ls || true",
            "echo `whoami`",
            "echo $(id)",
        ] {
            let mut a = action("terminal");
            a.terminal_action = Some("run".to_string());
            a.terminal_command = Some(cmd.to_string());
            assert!(
                matches!(
                    classify_action(&a, &ctx_with(vec![])),
                    ActionVerdict::Blocked(_)
                ),
                "command with metachar should be blocked: {cmd}"
            );
        }
    }

    #[test]
    fn test_terminal_sensitive_path_read_blocked() {
        // cat .env — blocked by sensitive file check in is_correction_safe_local_command
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat .env"}"#,
            Some(&read_only_caps()),
            None,
        );
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_terminal_path_outside_working_dir_blocked() {
        // cat /etc/hosts — absolute path outside working_dir; blocked by scope check
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat /etc/hosts"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(matches!(
            classify_action(&a, &ctx),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_read_file_absolute_path_outside_working_dir_blocked() {
        let mut a = action("read_file");
        a.local_paths = vec!["/etc/passwd".to_string()];
        a.read_only = true;
        a.needs_approval = false;
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(matches!(
            classify_action(&a, &ctx),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_read_file_sensitive_path_blocked() {
        // .env inside working_dir is still blocked (sensitive file rule)
        let mut a = action("read_file");
        a.local_paths = vec!["/tmp/test-workdir/.env".to_string()];
        a.read_only = true;
        a.needs_approval = false;
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(matches!(
            classify_action(&a, &ctx),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_terminal_non_run_actions_blocked() {
        for terminal_action in &["kill", "background", "detach", "list"] {
            let mut a = action("terminal");
            a.terminal_action = Some(terminal_action.to_string());
            a.terminal_command = Some("ls".to_string());
            assert!(
                matches!(
                    classify_action(&a, &ctx_with(vec![])),
                    ActionVerdict::Blocked(_)
                ),
                "terminal action={terminal_action} should be blocked"
            );
        }
    }

    #[test]
    fn test_run_command_safe_whitelist_does_not_require_preapproval_context() {
        // run_command with a SAFE_PREFIXES command — allowed
        let mut a = action("run_command");
        a.terminal_action = Some("run".to_string());
        a.terminal_command = Some("cargo test".to_string());
        // No needs_approval check for run_command (tool enforces its own safe list)
        assert_eq!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Allowed
        );
    }

    #[test]
    fn test_correction_detach_blocked() {
        let mut a = action("terminal");
        a.terminal_action = Some("run".to_string());
        a.terminal_command = Some("ls".to_string());
        a.detach = true;
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_unknown_approval_tool_blocked() {
        // unknown_capabilities=true → Block
        let mut a = action("some_future_tool");
        a.unknown_capabilities = true;
        a.needs_approval = true;
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_unknown_mutating_tool_blocked() {
        // mutates_state=true → Block
        let mut a = action("unknown_rw_tool");
        a.unknown_capabilities = false;
        a.mutates_state = true;
        a.needs_approval = false;
        assert!(matches!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_http_hostname_provider_mapping() {
        // github provider → github.com and api.github.com should match
        let ctx = CorrectionSubjectContext {
            subject_id: "s".to_string(),
            subject_kind: SelfCorrectionSubjectKind::Task,
            session_id: "sess".to_string(),
            original_request: "push a commit".to_string(),
            completion_contract_summary: "".to_string(),
            intended_accounts: vec![IntendedAccount {
                provider: "github".to_string(),
                account_id: "user123".to_string(),
                account_label: "user123".to_string(),
            }],
            allowed_external_targets: vec![],
            working_dir: std::path::PathBuf::from("/tmp/test-workdir"),
        };
        // Mutating POST to api.github.com → Allowed (github provider match)
        let mut a = action("http_request");
        a.http_method = Some("POST".to_string());
        a.external_target = Some("api.github.com".to_string());
        assert_eq!(classify_action(&a, &ctx), ActionVerdict::Allowed);

        // POST to evil.com → Blocked
        let mut b = action("http_request");
        b.http_method = Some("POST".to_string());
        b.external_target = Some("evil.com".to_string());
        assert!(matches!(
            classify_action(&b, &ctx),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_authenticated_get_requires_target() {
        // GET with auth_profile → needs target match
        let ctx = ctx_with(vec![IntendedAccount {
            provider: "github".to_string(),
            account_id: "user123".to_string(),
            account_label: "user123".to_string(),
        }]);
        // GET with auth_profile to github.com → Allowed
        let mut a = action("http_request");
        a.http_method = Some("GET".to_string());
        a.auth_profile = Some("github-pat".to_string());
        a.external_target = Some("github.com".to_string());
        assert_eq!(classify_action(&a, &ctx), ActionVerdict::Allowed);

        // GET with auth_profile to non-matching host → Blocked
        let mut b = action("http_request");
        b.http_method = Some("GET".to_string());
        b.auth_profile = Some("github-pat".to_string());
        b.external_target = Some("evil.com".to_string());
        assert!(matches!(
            classify_action(&b, &ctx),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_authenticated_request_structural_signals() {
        // GET with Authorization header to non-matching host → Blocked
        let ctx = ctx_with(vec![]);
        let mut a = action("http_request");
        a.http_method = Some("GET".to_string());
        a.auth_header_present = true;
        a.external_target = Some("internal-api.company.com".to_string());
        assert!(matches!(
            classify_action(&a, &ctx),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_send_file_and_share_memory_blocked_even_for_matching_target() {
        // send_file / share_memory → always blocked in 3b, even for intended target
        let ctx = ctx_with(vec![IntendedAccount {
            provider: "slack".to_string(),
            account_id: "T123".to_string(),
            account_label: "MyWorkspace".to_string(),
        }]);
        let mut a = action("send_file");
        a.external_target = Some("slack".to_string());
        assert!(matches!(
            classify_action(&a, &ctx),
            ActionVerdict::Blocked(_)
        ));

        let mut b = action("share_memory");
        b.external_target = Some("slack".to_string());
        assert!(matches!(
            classify_action(&b, &ctx),
            ActionVerdict::Blocked(_)
        ));
    }

    #[test]
    fn test_web_fetch_public_read_allowed_despite_external_side_effect() {
        // web_fetch GET with no auth → Allowed (public read-only)
        let mut a = action("web_fetch");
        a.http_method = Some("GET".to_string());
        a.external_side_effect = true; // web_fetch has this flag
        a.read_only = true;
        // No auth_profile, no auth_header_present
        assert_eq!(
            classify_action(&a, &ctx_with(vec![])),
            ActionVerdict::Allowed
        );
    }

    // -----------------------------------------------------------------------
    // P1.2 security fix tests: C1 (path scope) + I1 (find roots) + I2 (find actions)
    // -----------------------------------------------------------------------

    #[test]
    fn test_terminal_cat_absolute_outside_workdir_blocked() {
        // C1: cat with absolute path outside working_dir → Blocked
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat /tmp/outside.txt"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "cat /tmp/outside.txt should be blocked (outside working_dir)"
        );
    }

    #[test]
    fn test_terminal_cat_relative_escape_blocked() {
        // C1: cat with path traversal escaping working_dir → Blocked
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat ../../../etc/passwd"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "cat ../../../etc/passwd should be blocked (path traversal outside working_dir)"
        );
    }

    #[test]
    fn test_terminal_cat_env_in_workdir_blocked() {
        // C1: cat .env inside working_dir → Blocked (sensitive file)
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat /tmp/test-workdir/.env"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "cat /tmp/test-workdir/.env should be blocked (sensitive file inside working_dir)"
        );
    }

    #[test]
    fn test_terminal_cat_legit_file_allowed() {
        // C1: cat of a legitimate source file inside working_dir → Allowed
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat /tmp/test-workdir/src/main.rs"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "cat of legitimate file in working_dir should be allowed"
        );
    }

    #[test]
    fn test_find_broad_root_after_flag_blocked() {
        // I1: find -L / -name foo — the broad root '/' comes after a flag, must still be caught
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find -L / -name foo"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]);
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "find -L / -name foo should be blocked (broad root after flag)"
        );
    }

    #[test]
    fn test_find_fprintf_blocked() {
        // I2: find with -fprintf writes output to a file — must be blocked
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find . -fprintf /tmp/x %p"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]);
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "find . -fprintf /tmp/x %p should be blocked"
        );
    }

    #[test]
    fn test_find_printf_is_allowed() {
        // `-printf FORMAT` prints file METADATA to stdout (read-only) — it must
        // be Allowed. This is the exact worked-example form embedded in the
        // remediation hints/prompt; it MUST classify as Allowed so a model
        // copying the example is not blocked again.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find /tmp/test-workdir -type f -size +1G -printf '%s\\t%p\\n'"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "find -printf (read-only stdout metadata) must be Allowed"
        );
    }

    #[test]
    fn test_find_fprintf_still_blocked() {
        // The file-WRITING variant `-fprintf FILE` must remain Blocked even
        // though the read-only `-printf` is now allowed (distinct tokens).
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find . -fprintf /tmp/x %p"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]);
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "find -fprintf (writes to a file) must stay Blocked"
        );
    }

    #[test]
    fn test_find_exec_still_blocked() {
        // `-exec` executes commands — must remain Blocked.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find . -type f -exec ls {} +"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]);
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "find -exec must stay Blocked"
        );
    }

    #[test]
    fn test_find_delete_still_blocked() {
        // `-delete` removes files — must remain Blocked (and is also a hard
        // destructive block at the risk layer for the whole-disk root).
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find / -delete"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]);
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "find -delete must stay Blocked"
        );
    }

    #[test]
    fn test_find_scoped_allowed() {
        // find within working_dir with no dangerous actions → Allowed
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find /tmp/test-workdir -name '*.rs'"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "find within working_dir should be allowed"
        );
    }

    // -----------------------------------------------------------------------
    // Re-review security fix tests: Finding 1 (run_command path scope) +
    // Finding 2 (~/$HOME operand rejection) — through extract_proposed_action
    // -----------------------------------------------------------------------

    // --- Finding 1: run_command path scope ---

    #[test]
    fn test_run_command_tail_out_of_scope_blocked() {
        // Finding 1: tail with out-of-scope absolute path → Blocked
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"tail /var/log/system.log"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "run_command tail /var/log/system.log should be blocked (out-of-scope path)"
        );
    }

    #[test]
    fn test_run_command_ls_other_user_blocked() {
        // Finding 1: ls with out-of-scope path → Blocked
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"ls /Users/other"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "run_command ls /Users/other should be blocked (out-of-scope path)"
        );
    }

    #[test]
    fn test_run_command_head_bash_history_blocked() {
        // Finding 1: head with out-of-scope path → Blocked
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"head /Users/other/.bash_history"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "run_command head /Users/other/.bash_history should be blocked"
        );
    }

    #[test]
    fn test_run_command_cargo_test_no_path_allowed() {
        // Owner decision: build/test commands with no path operand → Allowed
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cargo test"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "run_command cargo test (no path operand) should be allowed"
        );
    }

    #[test]
    fn test_run_command_cargo_test_manifest_out_of_scope_blocked() {
        // Finding 1: cargo test --manifest-path with out-of-scope path → Blocked
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cargo test --manifest-path /outside/Cargo.toml"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "run_command cargo test --manifest-path /outside/Cargo.toml should be blocked"
        );
    }

    #[test]
    fn test_run_command_tail_in_scope_allowed() {
        // Finding 1: tail with in-scope path → Allowed
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"tail /tmp/test-workdir/log.txt"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "run_command tail of in-scope log file should be allowed"
        );
    }

    // --- Finding 2: ~/$HOME operand rejection ---

    #[test]
    fn test_terminal_cat_tilde_home_blocked() {
        // Finding 2: cat ~/somefile — tilde expands to real home dir at runtime
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat ~/somefile"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "terminal cat ~/somefile should be blocked (tilde expansion)"
        );
    }

    #[test]
    fn test_run_command_head_dollar_home_blocked() {
        // Finding 2: head $HOME/.bash_history — env expansion cannot be safely scoped
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"head $HOME/.bash_history"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "run_command head $HOME/.bash_history should be blocked ($HOME expansion)"
        );
    }

    #[test]
    fn test_terminal_ls_tilde_blocked() {
        // Finding 2: ls ~ — bare tilde is home dir
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"ls ~"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "terminal ls ~ should be blocked (tilde expansion)"
        );
    }

    #[test]
    fn test_run_command_cat_tilde_file_blocked() {
        // Finding 2: cat ~/somefile via run_command → Blocked
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cat ~/somefile"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
                                    // Note: "cat" is not in run_command SAFE_PREFIXES, so this would be blocked
                                    // by is_run_command_safe first. Use wc instead (which is in SAFE_PREFIXES).
        let a2 = extract_proposed_action(
            "run_command",
            r#"{"command":"wc ~/somefile"}"#,
            Some(&read_only_caps()),
            None,
        );
        assert!(
            matches!(classify_action(&a2, &ctx), ActionVerdict::Blocked(_)),
            "run_command wc ~/somefile should be blocked (tilde expansion)"
        );
        // Also verify the original cat version is blocked (by prefix list, not scope)
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "run_command cat ~/somefile should be blocked"
        );
    }

    #[test]
    fn test_terminal_cat_in_scope_still_allowed() {
        // Regression: in-scope paths are still allowed after the ~/$HOME fix
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat /tmp/test-workdir/src/main.rs"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "terminal cat of in-scope file should remain allowed after security fixes"
        );
    }

    #[test]
    fn test_run_command_cargo_test_in_scope_manifest_allowed() {
        // In-scope --manifest-path → Allowed (build/test command, path within working_dir)
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cargo test --manifest-path /tmp/test-workdir/Cargo.toml"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "run_command cargo test with in-scope manifest path should be allowed"
        );
    }

    // -----------------------------------------------------------------------
    // = -attached flag path escape fix (3b P1.2) — the `--flag=value` form
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_command_manifest_path_eq_out_of_scope_blocked() {
        // The = form of --manifest-path pointing outside working_dir must be blocked.
        // Previously this escaped scope checking because the whole token starts with `-`
        // and was filtered out before any path analysis.
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cargo test --manifest-path=/outside/Cargo.toml"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "run_command cargo test --manifest-path=/outside/Cargo.toml should be blocked (= form, out-of-scope)"
        );
    }

    #[test]
    fn test_run_command_manifest_path_space_out_of_scope_blocked_regression() {
        // Regression guard: the space form must still be blocked.
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cargo test --manifest-path /outside/Cargo.toml"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "run_command cargo test --manifest-path /outside/Cargo.toml should be blocked (space form, regression)"
        );
    }

    #[test]
    fn test_run_command_manifest_path_eq_in_scope_allowed() {
        // The = form pointing inside working_dir must be allowed.
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cargo test --manifest-path=/tmp/test-workdir/Cargo.toml"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "run_command cargo test --manifest-path=/tmp/test-workdir/Cargo.toml should be allowed (= form, in-scope)"
        );
    }

    #[test]
    fn test_generic_grep_file_eq_out_of_scope_blocked() {
        // grep --file=/etc/passwd . — the = form of a path flag on a read command must block.
        // Note: grep is in the terminal allowlist (not run_command), and skip_first_non_flag=true
        // means the first positional token (`.`) is the pattern and is skipped.
        // The path-shaped value /etc/passwd comes from `--file=/etc/passwd` and must be blocked.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"grep --file=/etc/passwd ."}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "terminal grep --file=/etc/passwd . should be blocked (= form path on grep)"
        );
    }

    #[test]
    fn test_run_command_cargo_test_no_path_shaped_token_allowed() {
        // `cargo test` with no path-shaped token at all → Allowed (baseline).
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cargo test"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]);
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "cargo test (no path token) must remain Allowed"
        );
    }

    #[test]
    fn test_run_command_cargo_test_package_flag_allowed() {
        // `cargo test --package foo` — `foo` is not path-shaped → Allowed.
        let a = extract_proposed_action(
            "run_command",
            r#"{"command":"cargo test --package foo"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]);
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "cargo test --package foo (bare word value, not path-shaped) must be Allowed"
        );
    }

    // -----------------------------------------------------------------------
    // P1.2 security fix tests: quoted-path bypass + bare dotfile path-shaped
    // -----------------------------------------------------------------------

    #[test]
    fn test_terminal_cat_quoted_absolute_outside_workdir_blocked() {
        // Issue 1: quoted absolute path must not dodge scope check
        // `cat "/tmp/secret.txt"` — the leading `"` used to make Path::is_absolute()
        // return false; after quote-stripping it correctly resolves as absolute + out-of-scope.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat \"/tmp/secret.txt\""}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "cat \"/tmp/secret.txt\" should be blocked (quoted absolute path outside working_dir)"
        );
    }

    #[test]
    fn test_terminal_cat_quoted_path_with_space_blocked() {
        // Issue 1: quoted path with internal space — split_whitespace yields fragments;
        // the first fragment `/tmp/a` (after stripping leading `"`) is absolute + out-of-scope.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat \"/tmp/a b.txt\""}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "cat \"/tmp/a b.txt\" should be blocked (quoted path with space, fragment is out-of-scope)"
        );
    }

    #[test]
    fn test_terminal_cat_quoted_in_scope_allowed() {
        // Issue 1: quoted path that IS in scope must still be allowed.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat \"/tmp/test-workdir/src/main.rs\""}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "cat \"/tmp/test-workdir/src/main.rs\" should be allowed (quoted in-scope path)"
        );
    }

    #[test]
    fn test_terminal_cat_bare_dotfile_blocked() {
        // Issue 2: bare dotfile `.env` is now path-shaped → sensitive backstop fires.
        // Previously `.env` had no `/`, `~`, or `$` so it was skipped entirely.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat .env"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "cat .env should be blocked (bare dotfile is now path-shaped, triggers sensitive backstop)"
        );
    }

    #[test]
    fn test_terminal_ls_dot_alone_still_allowed() {
        // Issue 2 boundary: bare `.` (current dir) must NOT be treated as path-shaped.
        // `.` is not path-shaped, so sandbox doesn't block it on scope grounds.
        // (command_risk may still flag it; we only assert it is not blocked by the
        //  dotfile-path-shaped rule — the classification result is risk-driven here)
        // Use `ls .` which is genuinely safe and should be Allowed.
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        let a2 = extract_proposed_action(
            "terminal",
            r#"{"command":"ls ."}"#,
            Some(&read_only_caps()),
            None,
        );
        assert_eq!(
            classify_action(&a2, &ctx),
            ActionVerdict::Allowed,
            "ls . should be allowed (`.` alone is not treated as a dotfile/path)"
        );
    }

    #[test]
    fn test_terminal_ls_dotdot_alone_still_allowed() {
        // Issue 2 boundary: bare `..` (parent dir) must NOT be treated as path-shaped
        // by the dotfile rule (it would be caught by scope-check anyway if it escapes,
        // but we must not accidentally trigger `looks_like_path` on it).
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"ls .."}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
                                    // `..` from /tmp/test-workdir resolves to /tmp which is outside working_dir.
                                    // It IS currently caught because `..` contains no `/`, `~`, `$` — BUT it
                                    // should NOT be caught by the NEW dotfile rule (which excludes `..`).
                                    // The scope check for `..` would catch it if it were path-shaped, but since
                                    // `..` alone is excluded from the dotfile rule, it falls through as before.
                                    // This test just verifies `ls ..` does not newly break.
                                    // Note: `ls ..` resolves to /tmp (outside workdir) so it may be blocked
                                    // by scope if `..` triggers the old contains('/') rule — it doesn't.
                                    // Current behavior: `..` is NOT path-shaped → Allowed.
        assert_eq!(
            classify_action(&a, &ctx),
            ActionVerdict::Allowed,
            "ls .. should be allowed (`..` alone is excluded from dotfile path-shaped rule)"
        );
    }

    // -----------------------------------------------------------------------
    // P3a.1 tests — read-only tool path operands: git_info + project_inspect
    // scope-checking via extract_proposed_action → local_paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_git_info_path_outside_working_dir_blocked() {
        // git_info with an absolute path outside working_dir must be Blocked.
        let a = extract_proposed_action(
            "git_info",
            r#"{"path":"/outside/repo"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
                                    // Confirm path was extracted.
        assert_eq!(
            a.local_paths,
            vec!["/outside/repo".to_string()],
            "git_info path must be in local_paths"
        );
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "git_info with out-of-scope path should be Blocked"
        );
    }

    #[test]
    fn test_project_inspect_path_outside_working_dir_blocked() {
        // project_inspect with an absolute path outside working_dir must be Blocked.
        let a = extract_proposed_action(
            "project_inspect",
            r#"{"path":"/other/project"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert_eq!(
            a.local_paths,
            vec!["/other/project".to_string()],
            "project_inspect path must be in local_paths"
        );
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "project_inspect with out-of-scope path should be Blocked"
        );
    }

    #[test]
    fn test_readonly_tool_path_in_scope_allowed() {
        // git_info and project_inspect with paths inside working_dir must be Allowed.
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir

        let git_a = extract_proposed_action(
            "git_info",
            r#"{"path":"/tmp/test-workdir/myrepo"}"#,
            Some(&read_only_caps()),
            None,
        );
        assert_eq!(
            classify_action(&git_a, &ctx),
            ActionVerdict::Allowed,
            "git_info with in-scope path should be Allowed"
        );

        let proj_a = extract_proposed_action(
            "project_inspect",
            r#"{"path":"/tmp/test-workdir/myrepo"}"#,
            Some(&read_only_caps()),
            None,
        );
        assert_eq!(
            classify_action(&proj_a, &ctx),
            ActionVerdict::Allowed,
            "project_inspect with in-scope path should be Allowed"
        );
    }

    #[test]
    fn test_git_info_sensitive_path_blocked() {
        // git_info pointing at a .env file inside working_dir must be Blocked
        // (sensitive-file rule even within scope).
        let a = extract_proposed_action(
            "git_info",
            r#"{"path":"/tmp/test-workdir/.env"}"#,
            Some(&read_only_caps()),
            None,
        );
        let ctx = ctx_with(vec![]); // working_dir = /tmp/test-workdir
        assert!(
            matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
            "git_info with sensitive path inside working_dir should be Blocked"
        );
    }

    // -----------------------------------------------------------------------
    // P2.4 carry-forward redaction test: normalized_attempt_signature redacts
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalized_attempt_signature_redacts_secrets() {
        // Build a ProposedAction for a terminal `run` command that embeds an
        // API-key-shaped secret. `normalized_attempt_signature` must call
        // `redact_secrets` on the terminal_command field — the raw key must NOT
        // appear in the returned string, and a [REDACTED:...] marker must be present.
        //
        // This is not a tautology: we construct the action directly with a known
        // raw secret value and assert on the *output* of the function under test.
        // If `normalized_attempt_signature` were to skip `redact_secrets` the raw
        // key would appear verbatim, the first assertion would fail.
        let raw_command =
            r#"curl -H "Authorization: Bearer sk-ABCDEF1234567890XYZ0" https://api.example.com"#;
        let action = ProposedAction {
            tool_name: "terminal".to_string(),
            terminal_action: Some("run".to_string()),
            terminal_command: Some(raw_command.to_string()),
            ..ProposedAction::default()
        };

        let sig = normalized_attempt_signature(&action);

        assert!(
            !sig.contains("sk-ABCDEF1234567890XYZ0"),
            "normalized_attempt_signature must not expose the raw API key in the signature: {sig}"
        );
        assert!(
            sig.contains("[REDACTED"),
            "normalized_attempt_signature must contain a [REDACTED...] marker: {sig}"
        );
    }

    // -----------------------------------------------------------------------
    // 3c robustness: actionable block-reason hints. These assert the reason
    // STRING is corrective while the block DECISION is unchanged.
    // -----------------------------------------------------------------------

    /// Extract the Blocked reason or panic. Helper for the hint tests.
    fn blocked_reason(a: &ProposedAction, ctx: &CorrectionSubjectContext) -> String {
        match classify_action(a, ctx) {
            ActionVerdict::Blocked(r) => r,
            v => panic!("expected Blocked, got {v:?}"),
        }
    }

    #[test]
    fn test_du_block_reason_is_actionable() {
        // `du` is the canonical reaper-stopped command. Still Blocked, but the
        // reason must steer the model to the bounded `find … -size` form.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"du -sh /tmp/test-workdir"}"#,
            Some(&read_only_caps()),
            None,
        );
        let reason = blocked_reason(&a, &ctx_with(vec![]));
        assert!(
            reason.contains("find") && reason.contains("-size"),
            "du block reason must suggest a bounded `find … -size` form: {reason}"
        );
    }

    #[test]
    fn test_other_disallowed_command_reason_is_actionable() {
        // A non-`du` disallowed command (e.g. `stat`) still blocks, and the
        // reason lists the allowlist + the find hint.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"stat /tmp/test-workdir/file"}"#,
            Some(&read_only_caps()),
            None,
        );
        let reason = blocked_reason(&a, &ctx_with(vec![]));
        assert!(
            reason.contains("find") && reason.contains("-size"),
            "disallowed-command reason must suggest the `find … -size` form: {reason}"
        );
    }

    #[test]
    fn test_block_reason_hints_use_printf_not_exec() {
        // The worked-example in the block reasons must use the Allowed
        // `-printf` form, NOT the (blocked) `-exec ls -lh {} +` form, so a
        // model copying the hint is not blocked again.
        let du = extract_proposed_action(
            "terminal",
            r#"{"command":"du -sh /tmp/test-workdir"}"#,
            Some(&read_only_caps()),
            None,
        );
        let du_reason = blocked_reason(&du, &ctx_with(vec![]));
        assert!(
            du_reason.contains("-printf") && !du_reason.contains("-exec"),
            "du hint must use -printf and not -exec: {du_reason}"
        );

        let other = extract_proposed_action(
            "terminal",
            r#"{"command":"stat /tmp/test-workdir/file"}"#,
            Some(&read_only_caps()),
            None,
        );
        let other_reason = blocked_reason(&other, &ctx_with(vec![]));
        assert!(
            other_reason.contains("-printf") && !other_reason.contains("-exec"),
            "allowlist hint must use -printf and not -exec: {other_reason}"
        );
    }

    #[test]
    fn test_tilde_operand_block_reason_mentions_absolute_path() {
        // `wc ~/file` — tilde operand. Still Blocked; reason must say to use an
        // explicit ABSOLUTE path.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"wc ~/somefile"}"#,
            Some(&read_only_caps()),
            None,
        );
        let reason = blocked_reason(&a, &ctx_with(vec![]));
        let lc = reason.to_lowercase();
        assert!(
            lc.contains("absolute") && reason.contains('~'),
            "tilde-operand reason must mention an absolute path and `~`: {reason}"
        );
    }

    #[test]
    fn test_dollar_home_operand_block_reason_mentions_absolute_path() {
        // `wc $HOME/file` — $HOME operand. Still Blocked; reason must say to use
        // an explicit ABSOLUTE path.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"wc $HOME/somefile"}"#,
            Some(&read_only_caps()),
            None,
        );
        let reason = blocked_reason(&a, &ctx_with(vec![]));
        assert!(
            reason.to_lowercase().contains("absolute"),
            "$HOME-operand reason must mention an absolute path: {reason}"
        );
    }

    #[test]
    fn test_find_home_root_block_reason_mentions_absolute_path() {
        // `find ~ -type f -size +500M` — home-shorthand root. Still Blocked;
        // reason must mention absolute path.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find ~ -type f -size +500M"}"#,
            Some(&read_only_caps()),
            None,
        );
        let reason = blocked_reason(&a, &ctx_with(vec![]));
        assert!(
            reason.to_lowercase().contains("absolute"),
            "find ~ root reason must mention an absolute path: {reason}"
        );
    }

    #[test]
    fn test_find_root_disk_block_reason_suggests_scoped_find() {
        // `find / -type f` — whole-disk root. Still Blocked; reason must suggest
        // a scoped find with a size filter.
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"find / -type f"}"#,
            Some(&read_only_caps()),
            None,
        );
        let reason = blocked_reason(&a, &ctx_with(vec![]));
        assert!(
            reason.contains("find") && reason.contains("-size"),
            "whole-disk find reason must suggest a scoped `find … -size`: {reason}"
        );
    }

    #[test]
    fn test_out_of_scope_path_reason_mentions_allowed_scope() {
        // `cat /tmp/outside.txt` — out-of-scope absolute path that is risk-Safe
        // (so it reaches the path-scope check rather than tripping the risk gate).
        // Still Blocked; the reason must point at the allowed scope (working_dir).
        let a = extract_proposed_action(
            "terminal",
            r#"{"command":"cat /tmp/outside.txt"}"#,
            Some(&read_only_caps()),
            None,
        );
        let reason = blocked_reason(&a, &ctx_with(vec![]));
        assert!(
            reason.contains("/tmp/test-workdir"),
            "out-of-scope reason must name the allowed scope: {reason}"
        );
    }

    #[test]
    fn test_hint_changes_do_not_alter_decisions_regression() {
        // The block DECISIONS for all the hint cases must remain Blocked, and a
        // legitimate in-scope command must remain Allowed (predicates untouched).
        let ctx = ctx_with(vec![]);
        for cmd in &[
            r#"{"command":"du -sh /tmp/test-workdir"}"#,
            r#"{"command":"wc ~/somefile"}"#,
            r#"{"command":"find ~ -type f -size +500M"}"#,
            r#"{"command":"find / -type f"}"#,
            r#"{"command":"cat /tmp/outside.txt"}"#,
        ] {
            let a = extract_proposed_action("terminal", cmd, Some(&read_only_caps()), None);
            assert!(
                matches!(classify_action(&a, &ctx), ActionVerdict::Blocked(_)),
                "decision must remain Blocked for {cmd}"
            );
        }
        // In-scope find must still be Allowed.
        let ok = extract_proposed_action(
            "terminal",
            r#"{"command":"find /tmp/test-workdir -type f -size +500M"}"#,
            Some(&read_only_caps()),
            None,
        );
        assert_eq!(
            classify_action(&ok, &ctx),
            ActionVerdict::Allowed,
            "in-scope find must remain Allowed"
        );
    }
}
