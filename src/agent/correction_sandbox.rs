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

    // (10) read_file / search_files — local read-only, path scope enforced.
    if matches!(tool, "read_file" | "search_files") {
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

    // Note: run_command SAFE_PREFIXES are broader (includes cargo build/test, etc.)
    // We allow run_command's whitelist without the is_correction_safe_local_command gate
    // because the tool itself enforces a safe whitelist with no shell metacharacters.

    // Check any local_paths for path scope.
    for path_str in &action.local_paths {
        if let Err(reason) = check_local_path_scope(path_str, ctx) {
            return ActionVerdict::Blocked(crate::tools::sanitize::redact_secrets(&reason));
        }
    }

    ActionVerdict::Allowed
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
        return Err(format!(
            "command '{}' is not in the correction-mode read-only allowlist",
            base_cmd
        ));
    }

    // `find` must not use -exec, -delete, -ok, -printf, -fprintf, -fls, -fprint,
    // and must not scan broad/out-of-scope roots (I1 + I2).
    if base_cmd == "find" {
        // I2: reject file-writing find actions.
        for token in cmd.split_whitespace() {
            if matches!(
                token,
                "-exec" | "-delete" | "-ok" | "-printf" | "-fprintf" | "-fls" | "-fprint"
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
                return Err(format!(
                    "find with broad root '{}' is not allowed in correction mode",
                    root
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
                    "find root '{}' is outside the allowed working directory",
                    root
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

    // C1: For all other allowed commands (non-find, non-pwd), extract path operands
    // and validate they are within working_dir and not sensitive.
    if base_cmd != "pwd" {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        // tokens[0] is the base_cmd; collect remaining non-flag args.
        let mut non_flag_args: Vec<&str> = tokens[1..]
            .iter()
            .copied()
            .filter(|t| !t.starts_with('-'))
            .collect();

        // For grep/rg: first non-flag arg is the pattern, not a path.
        if matches!(base_cmd, "grep" | "rg") && !non_flag_args.is_empty() {
            non_flag_args.remove(0);
        }

        for path_str in &non_flag_args {
            let path = std::path::Path::new(path_str);
            let resolved = if path.is_absolute() {
                path.to_path_buf()
            } else {
                working_dir.join(path)
            };
            let normalized = normalize_path_lexical(&resolved);
            if !normalized.starts_with(working_dir) {
                return Err(format!(
                    "command path '{}' is outside the allowed working directory",
                    path_str
                ));
            }
            if is_sensitive_file_path(&normalized) {
                return Err(format!(
                    "command path '{}' matches a sensitive/secret file pattern and cannot be read in correction mode",
                    path_str
                ));
            }
        }
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
            "path '{}' is outside the allowed working directory '{}'",
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
}
