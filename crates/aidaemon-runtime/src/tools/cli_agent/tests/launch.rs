//! Launching agents: executables, bypass flags, launch and auth failures.

use super::*;

#[test]
fn env_wrapper_finds_nested_program_after_unset_option() {
    let args = vec![
        "-u".to_string(),
        "CLAUDECODE".to_string(),
        "claude".to_string(),
        "--print".to_string(),
    ];
    assert_eq!(env_wrapped_program_index(&args), Some(2));
}

#[tokio::test]
async fn relative_cli_access_target_resolves_against_execution_cwd() {
    let cwd = tempfile::tempdir().expect("cwd");
    let expected = cwd.path().join("output.txt");
    let normalized = CliAgentTool::normalize_access_path(
        &aidaemon_tools::testing::active_execution_backend(),
        "output.txt",
        Some(cwd.path().to_string_lossy().as_ref()),
    )
    .await
    .expect("normalized target");
    assert_eq!(normalized, expected.to_string_lossy());
}

#[test]
fn cli_failure_formatter_preserves_diagnostic_and_safe_recovery() {
    let formatted = format_cli_agent_failure(
        "claude",
        Some(127),
        "[stderr] env: claude: No such file or directory\n",
        10_000,
        None,
    );
    assert!(formatted.contains("exit code 127"));
    assert!(formatted.contains("env: claude: No such file or directory"));
    assert!(!formatted.contains("git checkout ."));
}

#[test]
fn exit_127_with_no_agent_output_is_prelaunch() {
    let output = "ERROR: CLI agent 'claude' failed (exit code 127).\n\n\
## Failure Details\n[stderr] env: claude: No such file or directory";
    assert_eq!(cli_agent_exit_code(output), Some(127));
    assert!(exited_before_agent_output(Some(127), ""));
    // The agent started (its stream opened) before a command it ran was
    // missing: not a launch failure.
    assert!(!exited_before_agent_output(
        Some(127),
        r#"{"type":"system","subtype":"init"}"#
    ));
    assert!(!exited_before_agent_output(Some(1), ""));
}

/// The words "could not start" or an exit-code line inside an agent's
/// own report never type its outcome: only the daemon's header and the
/// typed never-started flag do.
#[test]
fn a_report_quoting_failure_words_is_not_a_failed_launch() {
    let report = "Fixed it. Before the fix the server could not start and the \
                      test failed (exit code 1).";
    assert_eq!(cli_agent_exit_code(report), None);
    let outcome = cli_agent_outcome("run", report.to_string(), false, false, false);
    assert_eq!(
        outcome.metadata.outcome_status,
        Some(ToolOutcomeStatus::Succeeded)
    );
    let never = cli_agent_outcome(
        "run",
        "ERROR: CLI agent recovery exhausted before task execution.".to_string(),
        false,
        false,
        true,
    );
    assert_eq!(
        never.metadata.outcome_status,
        Some(ToolOutcomeStatus::FailedPermanent)
    );
    assert!(never.metadata.transport_error.is_some());
}

#[tokio::test]
async fn wrapped_missing_executable_returns_structured_prelaunch_failure() {
    let (tool, _db_file) = setup_echo_tool().await;
    {
        let mut tools = tool.tools.write().unwrap();
        tools.clear();
        tools.insert(
            "broken".to_string(),
            CliToolEntry {
                command: "/usr/bin/env".to_string(),
                args: vec!["aidaemon-definitely-missing-cli".to_string()],
                description: "Broken test agent".to_string(),
                timeout: Duration::from_secs(10),
                max_output_chars: 10_000,
                is_dynamic: false,
            },
        );
    }
    *tool.tool_names.write().unwrap() = vec!["broken".to_string()];

    let outcome = tool
        .call_with_status_outcome(
            &json!({
                "action": "run",
                "tool": "broken",
                "prompt": "Inspect the repository"
            })
            .to_string(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        outcome.metadata.outcome_status,
        Some(ToolOutcomeStatus::FailedPermanent)
    );
    assert!(outcome.metadata.transport_error.is_some());
    assert!(!outcome.metadata.semantics.mutates_state());
    assert!(outcome.output.contains("could not start"));
    assert!(outcome.output.contains("aidaemon-definitely-missing-cli"));
}

fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| arg.to_string()).collect()
}

#[test]
fn every_default_launch_is_on_its_list() {
    for (name, _, args, _) in default_tool_definitions() {
        assert!(without_permission_bypass(&strings(&args), name).len() == args.len());
    }
}

/// A stored agent's unlisted argument stops the launch; the same argument
/// in the owner's own config does not.
#[cfg(unix)]
#[tokio::test]
async fn only_a_stored_agent_is_held_to_the_list() {
    use std::os::unix::fs::PermissionsExt;
    for (is_dynamic, launches) in [(true, false), (false, true)] {
        let (tool, _db) = setup_echo_tool().await;
        let dir = tempfile::tempdir().unwrap();
        let argv_file = dir.path().join("argv.txt");
        let script = dir.path().join("codex");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\n",
                argv_file.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        tool.tools.write().unwrap().insert(
            "codex".to_string(),
            CliToolEntry {
                command: script.to_string_lossy().to_string(),
                args: strings(&["exec", "-c", "synthetic=1"]),
                description: "fake codex".to_string(),
                timeout: Duration::from_secs(10),
                max_output_chars: 10000,
                is_dynamic,
            },
        );
        tool.tool_names.write().unwrap().push("codex".to_string());
        let output = tool
            .call(r#"{"action":"run","tool":"codex","prompt":"hello"}"#)
            .await
            .unwrap();
        assert_eq!(argv_file.exists(), launches, "{is_dynamic}: {output}");
        if !launches {
            assert!(
                output.starts_with("Blocked:") && output.contains("-c"),
                "{output}"
            );
        }
    }
}

const BYPASS_FLAGS: [&str; 4] = [
    "--dangerously-skip-permissions",
    "--yolo",
    "--dangerously-bypass-approvals-and-sandbox",
    "--allow-all-tools",
];

async fn echo_tool_with_flags() -> (CliAgentTool, tempfile::NamedTempFile) {
    let (tool, db) = setup_echo_tool().await;
    let mut args: Vec<String> = BYPASS_FLAGS.iter().map(|f| f.to_string()).collect();
    args.push("--keep-synthetic-flag".to_string());
    tool.tools.write().unwrap().get_mut("echo").unwrap().args = args;
    (tool, db)
}

#[tokio::test]
async fn cli_agent_args_never_contain_bypass_flags() {
    // Direct call path.
    let (tool, _db) = echo_tool_with_flags().await;
    let out = tool
        .call(r#"{"action":"run","tool":"echo","prompt":"hello"}"#)
        .await
        .unwrap();
    for flag in BYPASS_FLAGS {
        assert!(!out.contains(flag), "direct path leaked {flag}: {out}");
    }
    assert!(out.contains("--keep-synthetic-flag"), "got: {out}");
}

/// Execution-context path: a fake `codex` (the one agent with a native
/// sandbox adapter) records the argv it was really launched with.
#[cfg(unix)]
#[tokio::test]
async fn cli_agent_context_launch_never_passes_bypass_flags() {
    use std::os::unix::fs::PermissionsExt;
    let (tool, _db) = setup_echo_tool().await;
    let dir = tempfile::tempdir().unwrap();
    let argv_file = dir.path().join("argv.txt");
    let script = dir.path().join("codex");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\n",
            argv_file.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut args: Vec<String> = ["exec", "--yolo", "--dangerously-skip-permissions"]
        .iter()
        .map(|a| a.to_string())
        .collect();
    args.extend(["--allow-all-tools", "--keep-synthetic-flag"].map(String::from));
    tool.tools.write().unwrap().insert(
        "codex".to_string(),
        CliToolEntry {
            command: script.to_string_lossy().to_string(),
            args,
            description: "fake codex".to_string(),
            timeout: Duration::from_secs(10),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );
    tool.tool_names.write().unwrap().push("codex".to_string());
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","tool":"codex","prompt":"hello","workspace_mode":"read_only"}"#,
            None,
            ToolCallContext::default(),
        )
        .await
        .unwrap();
    let argv = std::fs::read_to_string(&argv_file)
        .unwrap_or_else(|_| panic!("agent was not launched: {}", outcome.output));
    for flag in [
        "--yolo",
        "--dangerously-skip-permissions",
        "--allow-all-tools",
    ] {
        assert!(
            !argv.lines().any(|l| l == flag),
            "launched with {flag}: {argv}"
        );
    }
    assert!(argv.lines().any(|l| l == "--keep-synthetic-flag"), "{argv}");
}

fn strip(agent: &str, args: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    strip_permission_bypass_flags(&mut args, agent).unwrap();
    args
}

#[test]
fn bypass_flag_stripping_keeps_other_flags() {
    assert_eq!(
        strip(
            "claude",
            &[
                "-p",
                "--yolo",
                "--model",
                "synthetic",
                "--allow-all-tools=1"
            ]
        ),
        ["-p", "--model", "synthetic"]
    );
}

#[test]
fn every_known_bypass_flag_is_stripped() {
    for flag in [
        "--dangerously-skip-permissions",
        "--allow-dangerously-skip-permissions",
        "--dangerously-bypass-hook-trust",
        "--dangerously-bypass-approvals-and-sandbox",
        "--full-auto",
        "--yolo",
        "--allow-all-tools",
        "--allow-all-paths",
        "--yes",
        "--yes-always",
    ] {
        assert_eq!(strip("any", &["a", flag, "b"]), ["a", "b"], "{flag}");
        let valued = format!("{flag}=1");
        assert_eq!(strip("any", &["a", &valued, "b"]), ["a", "b"], "{valued}");
    }
    assert_eq!(strip("gemini", &["--sandbox=false", "x"]), ["x"]);
    assert_eq!(strip("gemini", &["--sandbox", "false", "x"]), ["x"]);
    assert_eq!(strip("gemini", &["--approval-mode", "yolo", "x"]), ["x"]);
    assert_eq!(strip("gemini", &["--approval-mode=yolo", "x"]), ["x"]);
    assert_eq!(
        strip("claude", &["--permission-mode", "bypassPermissions", "x"]),
        ["x"]
    );
    // Non-bypass values of the same options survive.
    assert_eq!(
        strip(
            "gemini",
            &["--approval-mode", "default", "--sandbox", "true"]
        ),
        ["--approval-mode", "default", "--sandbox", "true"]
    );
    assert_eq!(
        strip("claude", &["--permission-mode", "plan"]),
        ["--permission-mode", "plan"]
    );
}

#[test]
fn short_y_is_stripped_only_for_gemini() {
    assert_eq!(strip("gemini", &["-y", "x"]), ["x"]);
    assert_eq!(strip("aider", &["-y", "x"]), ["-y", "x"]);
    assert_eq!(strip("claude", &["-y"]), ["-y"]);
}

#[test]
fn a_run_done_by_another_agent_says_which_did_it() {
    let note = substituted_agent_note("claude", "codex");
    assert!(note.contains("done by `codex`, not `claude`"), "{note}");
    assert!(!run_did_not_start(&note), "a substituted run still ran");
}

/// A write root that is a single file is measured itself: creating it,
/// or rewriting it, is a change.
#[tokio::test]
async fn a_file_write_root_is_measured_by_the_change_receipt() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("notes.txt");
    let roots = vec![file.to_string_lossy().to_string()];
    let missing = tree_receipt(&backend, &roots).await;
    std::fs::write(&file, "nightjar").unwrap();
    let created = tree_receipt(&backend, &roots).await;
    assert_ne!(missing, created, "creating the file is a change");
    std::fs::write(&file, "nightjar and more").unwrap();
    let rewritten = tree_receipt(&backend, &roots).await;
    assert_ne!(created, rewritten, "rewriting it is a change");
    assert_eq!(rewritten, tree_receipt(&backend, &roots).await);
}

#[test]
fn auth_failure_is_read_from_what_the_cli_reported_about_itself() {
    // Claude Code's typed stream-json error.
    let claude = r#"{"type":"system","subtype":"init"}
{"type":"assistant","message":{"content":[{"type":"text","text":"Invalid API key"}]},"error":"authentication_failed"}"#;
    assert!(cli_auth_failed(claude, claude, Some(1)));
    // Gemini CLI's own auth-help line, even on a zero exit.
    let gemini = "See https://goo.gle/gemini-cli-auth-docs#workspace-gca";
    assert!(cli_auth_failed(gemini, gemini, Some(0)));
    // A Codex-style error event before any task output, non-zero exit.
    let codex = r#"{"type":"thread.started"}
{"type":"error","message":"unexpected status 401 Unauthorized"}"#;
    assert!(cli_auth_failed(codex, codex, Some(1)));
    // The CLI's stderr before any task output.
    assert!(cli_auth_failed(
        "",
        "[stderr] Error: authentication required",
        Some(1)
    ));
    assert!(!cli_auth_failed(
        "",
        "[stderr] Error: authentication required",
        Some(0)
    ));
}

/// An agent whose task output mentions auth words (fixing a 403
/// handler that returns "forbidden", testing an unauthorized path) is
/// not disabled: once it produced task output, its words are its work.
#[test]
fn an_agent_printing_auth_words_mid_task_is_not_quarantined() {
    let stdout = r#"{"type":"system","subtype":"init"}
{"type":"assistant","message":{"content":[{"type":"text","text":"The handler returns forbidden for guests; the test expects 401 Unauthorized."}]}}
{"type":"result","is_error":true,"result":"tests failed: unauthorized"}"#;
    let display = format!("{stdout}\n[stderr] warning: token expired in fixture");
    assert!(!cli_auth_failed(stdout, &display, Some(1)));
    // A plain-text agent's own output is its task output too.
    assert!(!cli_auth_failed(
        "Fixed the 403: it now returns forbidden.",
        "Fixed the 403: it now returns forbidden.\n[stderr] not logged in to the demo site",
        Some(1)
    ));
}

/// A last line that reads like a question, then silence, does not end
/// the run: stdin is null, and no wording decides a run is stuck.
#[tokio::test]
async fn a_question_like_line_then_silence_does_not_kill_the_agent() {
    let (tool, _db) = setup_echo_tool().await;
    tool.tools.write().unwrap().insert(
            "quiet-agent".to_string(),
            CliToolEntry {
                command: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "printf 'Selected 12 files. Which one next?\\n'; sleep 16; printf 'all 12 files done\\n'"
                        .to_string(),
                ],
                description: "Quiet test agent".to_string(),
                timeout: Duration::from_secs(60),
                max_output_chars: 10_000,
                is_dynamic: false,
            },
        );
    tool.tool_names
        .write()
        .unwrap()
        .push("quiet-agent".to_string());

    let result = tool
        .call(r#"{"action":"run","tool":"quiet-agent","prompt":"process the files"}"#)
        .await
        .unwrap();

    assert!(result.contains("all 12 files done"), "got: {result}");
    assert!(!result.contains("stuck waiting for input"), "got: {result}");
}

#[tokio::test]
async fn authentication_failure_automatically_retries_another_configured_agent() {
    let (tool, _db) = setup_echo_tool().await;
    tool.tools.write().unwrap().insert(
        "auth-agent".to_string(),
        CliToolEntry {
            command: "/bin/sh".to_string(),
            args: vec![
                "-c".to_string(),
                "printf 'authentication required\\n' >&2; exit 1".to_string(),
            ],
            description: "Unauthenticated test agent".to_string(),
            timeout: Duration::from_secs(10),
            max_output_chars: 10_000,
            is_dynamic: false,
        },
    );
    tool.tool_names
        .write()
        .unwrap()
        .push("auth-agent".to_string());

    let result = tool
        .call(r#"{"action":"run","tool":"auth-agent","prompt":"finish autonomously"}"#)
        .await
        .unwrap();

    assert!(result.contains("finish autonomously"), "got: {result}");
    assert!(
        !tool.tools.read().unwrap().contains_key("auth-agent"),
        "the unavailable agent must remain quarantined"
    );
}

#[test]
fn zero_exit_auth_setup_output_is_a_failed_completion() {
    let completion = CliAgentTool::classify_completion_result(
        "gemini",
        Some(0),
        "See https://goo.gle/gemini-cli-auth-docs#workspace-gca",
        "See https://goo.gle/gemini-cli-auth-docs#workspace-gca",
        10_000,
    );

    assert!(!completion.success);
    assert!(completion.response.is_none());
    assert!(
        completion
            .error
            .as_deref()
            .is_some_and(|error| error.contains("authentication failed"))
    );
    assert!(!completion.persisted_output.contains("workspace-gca"));
}

// -----------------------------------------------------------------------
// Loop detection tests
// -----------------------------------------------------------------------
