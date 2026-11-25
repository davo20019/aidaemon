//! Claude Code, Anthropic's coding agent, in each permission mode.
//!
//! - Checked: `--restricted` (no user, project or local settings; file
//!   tools confined to the working folder), no MCP servers, `manual`
//!   permissions and its shell in its own sandbox, which writes only in
//!   the folder and asks before each network host. Every read and write
//!   path must lie in the folder: the confinement is the folder.
//! - Every command: no settings sources, no MCP servers, no sandbox,
//!   `manual` permissions: each command (but its own few read-only ones),
//!   edit and read outside its folder is a prompt.
//! - Full access: `bypassPermissions`.
//!
//! In the first two every prompt goes to the host ([`host::ClaudeHost`]).
//!
//! A run continues an earlier session with `--resume <session id>`, in
//! print mode as in its terminal. Claude Code keeps a session under the
//! folder it ran in, so the run must work in the same folder.
//!
//! A run asked for a report ([`RunRequest::report`]) is launched with
//! `--json-schema <the report schema>`, plain or hosted alike: its final
//! `result` event then carries the report as `structured_output`
//! ([`events`]).
//!
//! Verified against Claude Code 2.1.327 (live probes, 2026-11-17/18; the
//! report, 2026-11-23).

pub mod events;
pub mod host;

use serde_json::{Value, json};

use crate::{
    CodingAgent, Discovery, Launch, LaunchArguments, PermissionMode, Plan, PromptPlacement,
    RunRequest,
};

/// Claude Code.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClaudeCode;

/// Every flag that sets what a launch may do (exact command-line flags),
/// removed so a mode's own are the only ones.
fn drop_permission_flags(args: &mut Vec<String>, start: usize) {
    const DROPS_VALUE: [&str; 8] = [
        "--permission-mode",
        "--tools",
        "--allowedTools",
        "--allowed-tools",
        "--settings",
        "--setting-sources",
        "--add-dir",
        "--mcp-config",
    ];
    let mut kept: Vec<String> = args[..start.min(args.len())].to_vec();
    let mut index = start;
    while let Some(argument) = args.get(index) {
        let name = argument.split('=').next().unwrap_or(argument);
        if DROPS_VALUE.contains(&name) {
            index += if argument.contains('=') { 1 } else { 2 };
            continue;
        }
        if matches!(name, "--restricted" | "--strict-mcp-config") {
            index += 1;
            continue;
        }
        kept.push(argument.clone());
        index += 1;
    }
    *args = kept;
}

fn push(args: &mut Vec<String>, values: &[&str]) {
    args.extend(values.iter().map(|value| value.to_string()));
}

fn tools(read_only: bool) -> &'static str {
    if read_only {
        "Read,Glob,Grep"
    } else {
        "Bash,Read,Edit,Write,Glob,Grep"
    }
}

/// Non-interactive, with stream-json output (its progress and its final
/// result as events): what every run needs.
fn basics(args: &mut Vec<String>) {
    let has = |args: &[String], flags: &[&str]| args.iter().any(|a| flags.contains(&a.as_str()));
    if !has(args, &["-p", "--print"]) {
        push(args, &["--print"]);
    }
    if !has(args, &["--output-format", "-o"]) {
        push(args, &["--output-format", "stream-json"]);
    }
    if !has(args, &["--include-partial-messages"]) {
        push(args, &["--include-partial-messages"]);
    }
    if !has(args, &["--verbose"]) {
        push(args, &["--verbose"]);
    }
}

fn confine(args: &mut Vec<String>, start: usize, request: &RunRequest<'_>) -> Result<(), String> {
    let Some(working_dir) = request.working_dir else {
        return Err("a confined Claude Code run needs a working directory".to_string());
    };
    let root = std::path::Path::new(working_dir);
    if let Some(outside) = request
        .read_paths
        .iter()
        .chain(request.write_paths)
        .find(|path| !std::path::Path::new(path).starts_with(root))
    {
        return Err(format!(
            "a confined Claude Code run reaches only its working directory {working_dir}, and {outside} is outside it"
        ));
    }
    drop_permission_flags(args, start);
    let settings: Value = json!({"sandbox": {
        "enabled": true,
        "autoAllowBashIfSandboxed": false,
        "allowUnsandboxedCommands": false,
    }});
    push(
        args,
        &[
            "--restricted",
            "--strict-mcp-config",
            "--permission-mode",
            "manual",
            "--tools",
            tools(request.read_only),
            "--settings",
            &settings.to_string(),
        ],
    );
    Ok(())
}

fn every_command(args: &mut Vec<String>, start: usize, read_only: bool) {
    drop_permission_flags(args, start);
    push(
        args,
        &[
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--permission-mode",
            "manual",
            "--tools",
            tools(read_only),
        ],
    );
}

/// Ephemeral `--mcp-config` file written for host-injected MCP servers and
/// removed when the run finishes (kept in [`Plan::keep_alive`]).
#[derive(Debug)]
struct McpConfigFile {
    path: std::path::PathBuf,
}

impl McpConfigFile {
    fn write(servers: &[crate::McpEndpoint]) -> Result<Self, String> {
        let mut map = serde_json::Map::new();
        for server in servers {
            let env: serde_json::Map<String, Value> = server
                .env
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect();
            map.insert(
                server.name.clone(),
                json!({
                    "type": "stdio",
                    "command": server.command,
                    "args": server.args,
                    "env": env,
                }),
            );
        }
        let body = json!({ "mcpServers": map }).to_string();
        // It carries the peer server's token: owner-only from its creation,
        // in the user's own run directory, never the shared temp directory.
        let dir = crate::private_files::run_dir()
            .map_err(|e| format!("failed to make the run directory: {e}"))?;
        let path = dir.join(format!("claude-mcp-{}.json", uuid::Uuid::new_v4()));
        crate::private_files::write_new_private_file(&path, body.as_bytes())
            .map_err(|e| format!("failed to write MCP config: {e}"))?;
        Ok(Self { path })
    }
}

impl Drop for McpConfigFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The task on stdin and every prompt to the host.
fn hosted(args: &mut Vec<String>, request: &RunRequest<'_>) -> Plan {
    for pair in host::HOST_FLAGS.chunks(2) {
        if !args.iter().any(|arg| arg == pair[0]) {
            push(args, pair);
        }
    }
    Plan {
        prompt: PromptPlacement::Host,
        host: Some(Box::new(host::ClaudeHostStarter {
            working_dir: request.working_dir.map(str::to_string),
            trusted_mcp_servers: request
                .mcp_servers
                .iter()
                .map(|server| server.name.clone())
                .collect(),
        })),
        ..plan()
    }
}

/// It refuses to start inside another Claude Code session.
fn plan() -> Plan {
    Plan {
        env_remove: vec!["CLAUDECODE".to_string()],
        ..Plan::plain()
    }
}

fn attach_mcp_config(
    args: &mut Vec<String>,
    plan: &mut Plan,
    servers: &[crate::McpEndpoint],
) -> Result<(), String> {
    if servers.is_empty() {
        return Ok(());
    }
    let file = McpConfigFile::write(servers)?;
    let path = file.path.to_string_lossy().into_owned();
    // One `--opt=value` token: `--mcp-config` is variadic, so a separate value
    // would also swallow a prompt passed as the last argument.
    args.push(format!("--mcp-config={path}"));
    // The injected servers are a few small tools: load them up front rather
    // than behind ToolSearch, which costs a model round trip per run.
    plan.env
        .push(("ENABLE_TOOL_SEARCH".to_string(), "false".to_string()));
    plan.keep_alive = Some(match plan.keep_alive.take() {
        Some(kept) => Box::new((kept, file)),
        None => Box::new(file),
    });
    Ok(())
}

/// `launch` in `request`'s mode, and how the run goes.
fn shape(request: &RunRequest<'_>, launch: &mut Launch) -> Result<Plan, String> {
    if let Some(model) = request.model {
        crate::agent::set_option(&mut launch.args, launch.start, "--model", &[], model);
    }
    if let Some(session) = request.resume {
        crate::agent::set_option(&mut launch.args, launch.start, "--resume", &["-r"], session);
    }
    if request.report {
        let schema = crate::report::schema().to_string();
        crate::agent::set_option(
            &mut launch.args,
            launch.start,
            "--json-schema",
            &[],
            &schema,
        );
    }
    let args = &mut launch.args;
    basics(args);
    let mut plan = if !request.enforce {
        // Outside the embedding agent's checks: its prompts still go to
        // the host when one can answer them.
        if request.can_host {
            hosted(args, request)
        } else {
            plan()
        }
    } else {
        match request.mode {
            PermissionMode::FullAccess => {
                drop_permission_flags(args, launch.start);
                push(args, &["--permission-mode", "bypassPermissions"]);
                plan()
            }
            _ if !request.can_host => {
                return Err(
                    "Claude Code's prompts can't be answered here (its run needs a local process \
                     inside a checked turn)"
                        .to_string(),
                );
            }
            PermissionMode::Checked => {
                confine(args, launch.start, request)?;
                hosted(args, request)
            }
            PermissionMode::EveryCommand => {
                every_command(args, launch.start, request.read_only);
                hosted(args, request)
            }
        }
    };
    attach_mcp_config(args, &mut plan, request.mcp_servers)?;
    Ok(plan)
}

impl CodingAgent for ClaudeCode {
    fn reports_steps(&self) -> bool {
        true
    }

    /// `--json-schema` holds its final message to the report schema.
    fn reports_in_fields(&self) -> bool {
        true
    }

    fn read_events(&self, line: &str) -> Vec<crate::AgentEvent> {
        events::read_events(line)
    }

    fn name(&self) -> &str {
        "Claude Code"
    }

    fn mark(&self) -> Option<&str> {
        Some("anthropic")
    }

    fn launches(&self, program: &str) -> bool {
        program
            .trim_end_matches(".exe")
            .eq_ignore_ascii_case("claude")
    }

    fn modes(&self) -> &[PermissionMode] {
        &[
            PermissionMode::Checked,
            PermissionMode::EveryCommand,
            PermissionMode::FullAccess,
        ]
    }

    fn discovery(&self) -> Option<Discovery> {
        Some(Discovery {
            name: "claude",
            command: "claude",
            args: &["-p", "--output-format", "stream-json", "--verbose"],
            description: "Claude Code — Anthropic's AI coding agent",
        })
    }

    fn launch_arguments(&self) -> Option<LaunchArguments> {
        Some(LaunchArguments {
            flags: &["-p", "--print", "--verbose", "--include-partial-messages"],
            valued: &[
                "--output-format",
                "--model",
                "--fallback-model",
                "--max-turns",
                "--append-system-prompt",
                "--system-prompt",
            ],
            subcommands: &[],
        })
    }

    fn can_run(&self, mode: PermissionMode, _read_only: bool, can_host: bool) -> bool {
        mode == PermissionMode::FullAccess || can_host
    }

    fn account(&self) -> &str {
        "Its own Claude Code sign-in: a Claude subscription or an Anthropic API key."
    }

    fn takes_model(&self) -> bool {
        true
    }

    fn resumes(&self) -> bool {
        true
    }

    fn model_listing(&self) -> Option<crate::ModelListing> {
        Some(crate::ModelListing::Claude)
    }

    /// `claude auth status` answers with typed JSON (`loggedIn`).
    fn readiness_probe(&self) -> Option<crate::Probe> {
        Some(crate::Probe::Status {
            args: vec!["auth".into(), "status".into()],
            read: |_, stdout| match serde_json::from_str::<Value>(stdout.trim())
                .ok()
                .and_then(|status| status.get("loggedIn").and_then(Value::as_bool))
            {
                Some(true) => crate::Readiness::Ready,
                Some(false) => crate::Readiness::NotUsable {
                    reason: "Claude Code isn't signed in on this machine.".into(),
                    fix: Some("Run `claude auth login` on this machine.".into()),
                },
                None => crate::Readiness::Unchecked,
            },
        })
    }

    fn prepare(&self, request: &RunRequest<'_>, launch: &mut Launch) -> Result<Plan, String> {
        let mut plan = shape(request, launch)?;
        // Its own flag: a session it was told not to keep can't be
        // continued.
        plan.keeps_session = !launch.args[launch.start..]
            .iter()
            .any(|arg| arg == "--no-session-persistence");
        Ok(plan)
    }

    /// Its `result` event lists what its own permission check refused
    /// under `permission_denials` (`tool_name`, `tool_input`): each reads
    /// "`<tool>: <target>`", the target being the call's command or path
    /// when it names one (JSON fields, never prose).
    fn refused_actions(&self, stdout: &str) -> Vec<String> {
        let Some(result) = stdout
            .lines()
            .rev()
            .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
            .find(|event| event.get("type").and_then(Value::as_str) == Some("result"))
        else {
            return Vec::new();
        };
        let mut actions = Vec::new();
        for denial in result
            .get("permission_denials")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(tool) = denial.get("tool_name").and_then(Value::as_str) else {
                continue;
            };
            let input = denial.get("tool_input");
            let target = ["command", "file_path", "path", "url", "pattern"]
                .iter()
                .find_map(|key| {
                    input
                        .and_then(|input| input.get(*key))
                        .and_then(Value::as_str)
                })
                .map(|target| target.split_whitespace().collect::<Vec<_>>().join(" "));
            let action = match target {
                Some(target) if !target.is_empty() => {
                    let short = match target.char_indices().nth(160) {
                        Some((cut, _)) => format!("{}…", &target[..cut]),
                        None => target,
                    };
                    format!("{tool}: {short}")
                }
                _ => tool.to_string(),
            };
            if !actions.contains(&action) {
                actions.push(action);
            }
        }
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn request(mode: PermissionMode, read_only: bool, can_host: bool) -> RunRequest<'static> {
        RunRequest {
            mode,
            enforce: true,
            read_only,
            working_dir: Some("/w/project"),
            read_paths: &[],
            write_paths: &[],
            can_host,
            stdin_open: true,
            model: None,
            resume: None,
            report: false,
            mcp_servers: &[],
        }
    }

    #[test]
    fn checked_confines_it_to_its_folder_and_hosts_its_prompts() {
        let mut launch = Launch {
            args: strings(&["-p", "--permission-mode", "acceptEdits", "--add-dir", "/x"]),
            start: 0,
        };
        let plan = ClaudeCode
            .prepare(&request(PermissionMode::Checked, false, true), &mut launch)
            .unwrap();
        let joined = launch.args.join(" ");
        assert!(joined.contains("--restricted"), "{joined}");
        assert!(joined.contains("--permission-mode manual"));
        assert!(joined.contains("--permission-prompt-tool stdio"));
        assert!(!joined.contains("acceptEdits") && !joined.contains("--add-dir"));
        assert_eq!(plan.prompt, PromptPlacement::Host);
        assert!(plan.host.is_some());
        assert_eq!(plan.env_remove, ["CLAUDECODE"]);
    }

    #[test]
    fn checked_refuses_a_path_outside_its_folder_and_a_run_it_cannot_host() {
        let outside = ["/elsewhere/a".to_string()];
        let mut asked = request(PermissionMode::Checked, false, true);
        asked.write_paths = &outside;
        let mut launch = Launch {
            args: Vec::new(),
            start: 0,
        };
        assert!(ClaudeCode.prepare(&asked, &mut launch).is_err());
        assert!(
            ClaudeCode
                .prepare(
                    &request(PermissionMode::EveryCommand, false, false),
                    &mut launch
                )
                .is_err()
        );
        assert!(!ClaudeCode.can_run(PermissionMode::Checked, false, false));
        assert!(ClaudeCode.can_run(PermissionMode::FullAccess, false, false));
    }

    #[test]
    fn every_command_loads_no_settings_and_asks_for_each_action() {
        let mut launch = Launch {
            args: strings(&["-p", "--settings", "{}", "--restricted"]),
            start: 0,
        };
        let plan = ClaudeCode
            .prepare(
                &request(PermissionMode::EveryCommand, false, true),
                &mut launch,
            )
            .unwrap();
        let args = &launch.args;
        let joined = args.join(" ");
        assert!(joined.contains("--permission-mode manual"), "{joined}");
        assert!(!joined.contains("--restricted") && !joined.contains("--settings {}"));
        let sources = args
            .iter()
            .position(|arg| arg == "--setting-sources")
            .unwrap();
        assert_eq!(args[sources + 1], "");
        assert!(plan.host.is_some());
    }

    #[test]
    fn full_access_bypasses_its_permissions_with_no_host() {
        let mut launch = Launch {
            args: strings(&["-p", "--permission-mode", "manual"]),
            start: 0,
        };
        let plan = ClaudeCode
            .prepare(
                &request(PermissionMode::FullAccess, false, true),
                &mut launch,
            )
            .unwrap();
        let joined = launch.args.join(" ");
        assert!(
            joined.contains("--permission-mode bypassPermissions"),
            "{joined}"
        );
        assert!(!joined.contains("manual"));
        assert!(plan.host.is_none());
        assert_eq!(plan.prompt, PromptPlacement::LastArgument);
    }

    #[test]
    fn its_own_refusals_are_read_from_its_result() {
        let stdout = concat!(
            r#"{"type":"assistant"}"#,
            "\n",
            r#"{"type":"result","permission_denials":[{"tool_name":"Bash","tool_input":{"command":"rm -rf /w/x"}},{"tool_name":"Write","tool_input":{"file_path":"/w/y"}}]}"#
        );
        assert_eq!(
            ClaudeCode.refused_actions(stdout),
            vec!["Bash: rm -rf /w/x".to_string(), "Write: /w/y".to_string()]
        );
        assert!(ClaudeCode.refused_actions("no json").is_empty());
    }

    #[test]
    fn a_run_continues_the_session_it_names_in_every_mode() {
        assert!(ClaudeCode.resumes());
        for mode in [
            PermissionMode::Checked,
            PermissionMode::EveryCommand,
            PermissionMode::FullAccess,
        ] {
            let mut asked = request(mode, false, true);
            asked.resume = Some("s-1");
            let mut launch = Launch {
                args: strings(&["-p", "--output-format", "stream-json"]),
                start: 0,
            };
            ClaudeCode.prepare(&asked, &mut launch).unwrap();
            let at = launch
                .args
                .iter()
                .position(|arg| arg == "--resume")
                .unwrap();
            assert_eq!(launch.args[at + 1], "s-1");
            assert!(
                launch
                    .args
                    .iter()
                    .any(|arg| arg == "--print" || arg == "-p")
            );
        }
        // No session named: a new one.
        let mut launch = Launch {
            args: strings(&["-p"]),
            start: 0,
        };
        ClaudeCode
            .prepare(
                &request(PermissionMode::FullAccess, false, true),
                &mut launch,
            )
            .unwrap();
        assert!(!launch.args.iter().any(|arg| arg == "--resume"));
    }

    #[test]
    fn a_stored_launchs_own_session_is_replaced() {
        for stored in [
            &["-p", "--resume", "old"][..],
            &["-p", "-r", "old"],
            &["-p", "--resume=old"],
        ] {
            let mut asked = request(PermissionMode::FullAccess, false, true);
            asked.resume = Some("s-1");
            let mut launch = Launch {
                args: strings(stored),
                start: 0,
            };
            ClaudeCode.prepare(&asked, &mut launch).unwrap();
            let joined = launch.args.join(" ");
            assert!(!joined.contains("old"), "{joined}");
            assert_eq!(
                launch
                    .args
                    .iter()
                    .filter(|arg| arg.starts_with("--resume") || *arg == "-r")
                    .count(),
                1,
                "{joined}"
            );
            assert!(joined.contains("--resume s-1"), "{joined}");
        }
    }

    #[test]
    fn a_run_asked_for_a_report_holds_its_final_message_to_the_schema() {
        assert!(ClaudeCode.reports_in_fields());
        let schema_of = |launch: &Launch| {
            let at = launch.args.iter().position(|arg| arg == "--json-schema")?;
            serde_json::from_str::<Value>(&launch.args[at + 1]).ok()
        };
        for (mode, can_host) in [
            (PermissionMode::FullAccess, false),
            (PermissionMode::Checked, true),
            (PermissionMode::EveryCommand, true),
        ] {
            let mut asked = request(mode, false, can_host);
            asked.report = true;
            // A stored launch's own schema is replaced.
            let mut launch = Launch {
                args: strings(&["-p", "--json-schema", "{}"]),
                start: 0,
            };
            let plan = ClaudeCode.prepare(&asked, &mut launch).unwrap();
            assert_eq!(
                schema_of(&launch),
                Some(crate::report::schema()),
                "{mode:?}"
            );
            assert_eq!(
                launch
                    .args
                    .iter()
                    .filter(|arg| *arg == "--json-schema")
                    .count(),
                1
            );
            // Hosted (the task on stdin) or not, the flag goes with it.
            assert_eq!(plan.host.is_some(), can_host);
            asked.report = false;
            let mut launch = Launch {
                args: strings(&["-p"]),
                start: 0,
            };
            ClaudeCode.prepare(&asked, &mut launch).unwrap();
            assert_eq!(schema_of(&launch), None);
        }
    }

    #[test]
    fn injected_mcp_servers_write_ephemeral_config_and_auto_allow_trusted_prefix() {
        let endpoints = vec![crate::McpEndpoint {
            name: "aidaemon_peers".into(),
            command: "/usr/local/bin/aidaemon".into(),
            args: vec![
                "mcp-bridge".into(),
                "--addr".into(),
                "127.0.0.1:4242".into(),
            ],
            env: vec![("AIDAEMON_MCP_TOKEN".into(), "synthetic-token".into())],
        }];
        let mut asked = request(PermissionMode::Checked, false, true);
        asked.mcp_servers = &endpoints;
        let mut launch = Launch {
            args: strings(&["-p"]),
            start: 0,
        };
        let plan = ClaudeCode.prepare(&asked, &mut launch).unwrap();
        assert!(launch.args.iter().any(|a| a == "--strict-mcp-config"));
        assert!(!launch.args.iter().any(|a| a == "--mcp-config"));
        assert!(
            plan.env
                .iter()
                .any(|(k, v)| k == "ENABLE_TOOL_SEARCH" && v == "false")
        );
        let cfg_path = launch
            .args
            .iter()
            .find_map(|a| a.strip_prefix("--mcp-config="))
            .map(std::path::PathBuf::from)
            .expect("missing --mcp-config=<path>");
        let body: Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        assert_eq!(
            body["mcpServers"]["aidaemon_peers"]["command"],
            "/usr/local/bin/aidaemon"
        );
        assert_eq!(
            body["mcpServers"]["aidaemon_peers"]["env"]["AIDAEMON_MCP_TOKEN"],
            "synthetic-token"
        );
        // Host helper recognizes only host-injected MCP servers, never arbitrary ones.
        let trusted = vec!["aidaemon_peers".to_string()];
        assert!(host::is_trusted_mcp_tool(
            "mcp__aidaemon_peers__delegate_cli",
            &trusted
        ));
        assert!(!host::is_trusted_mcp_tool("mcp__untrusted__send", &trusted));
        // Dropping the plan cleans up the ephemeral config file.
        drop(plan);
        assert!(!cfg_path.exists());
    }
}
