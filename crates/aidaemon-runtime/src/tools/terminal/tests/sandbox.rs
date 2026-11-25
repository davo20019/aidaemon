//! Confinement: manifests, Seatbelt and Linux sandbox policy, runtime grants.

use super::*;

#[test]
fn sandbox_denial_telemetry_is_adapter_diagnostic_not_exit_classifier() {
    let denial = sandbox_access_denial(
        "/bin/sh: cannot open /etc/hosts: Operation not permitted",
        Some(1),
        true,
    )
    .expect("kernel denial diagnostic");
    assert_eq!(denial.reason_code, "sandbox_policy_denied");
    assert_eq!(denial.exit_code, Some(1));
    assert!(sandbox_access_denial("ordinary command error", Some(1), true).is_none());
    assert!(
        sandbox_access_denial(
            "cannot open /etc/hosts: Operation not permitted",
            Some(0),
            true
        )
        .is_none()
    );
    // Only a process that ran confined can be refused by the sandbox.
    assert!(
        sandbox_access_denial(
            "cannot open /etc/hosts: Operation not permitted",
            Some(1),
            false
        )
        .is_none()
    );
}

/// `git push`'s own refusal names no path the sandbox refused: it is
/// the remote's authentication failing, never a sandbox policy block.
#[test]
fn a_programs_own_permission_refusal_is_not_a_sandbox_denial() {
    let stderr = "git@example.com: Permission denied (publickey).\n\
                      fatal: Could not read from remote repository.";
    assert!(sandbox_access_denial(stderr, Some(128), true).is_none());
}

#[test]
fn command_form_heredoc_uses_the_script_sandbox_boundary() {
    assert!(command_requires_stdin_shell("cat <<'EOF'\nvalue\nEOF"));
    assert!(!command_requires_stdin_shell("printf '%s' value"));
}

#[test]
fn confined_write_process_is_bounded_workspace_observation_and_mutation() {
    let semantics = terminal_run_semantics_from_access(true, true);
    assert!(semantics.observes_state());
    assert!(semantics.mutates_state());
    assert_eq!(
        semantics.verification_mode,
        crate::tools::traits::ToolVerificationMode::ResultContent
    );
    assert!(
        semantics
            .mutation_effects
            .intersects(crate::tools::traits::ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn confined_manifest_denies_undeclared_system_file_read() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let request = confined_terminal_execution_request(
        &backend,
        "/usr/bin/head -n 1 /etc/hosts",
        Some("/tmp"),
        &["/tmp".to_string()],
        &[],
    )
    .await
    .expect("confined request");
    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");
    assert_ne!(output.exit_code, 0, "undeclared /etc/hosts read escaped");
    assert!(
        output.stdout_lossy().trim().is_empty(),
        "undeclared file contents escaped: {}",
        output.stdout_lossy()
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn empty_manifest_is_fail_closed_not_ambient_access() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let request = confined_terminal_execution_request(
        &backend,
        "/usr/bin/head -n 1 /etc/hosts",
        None,
        &[],
        &[],
    )
    .await
    .expect("confined request");
    assert!(matches!(
        request.command,
        aidaemon_tools::CommandSpec::Argv { .. }
    ));
    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");
    assert_ne!(output.exit_code, 0, "empty manifest became ambient access");

    let predicate_request =
        confined_terminal_execution_request(&backend, "/usr/bin/false", None, &[], &[])
            .await
            .expect("pure predicate request");
    let predicate_output = backend
        .execute(predicate_request, Duration::from_secs(30))
        .await
        .expect("pure predicate execution");
    assert_eq!(
        predicate_output.exit_code,
        1,
        "empty task-data authority must still permit a pure process predicate: {}",
        predicate_output.stderr_lossy()
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn execution_cwd_does_not_implicitly_grant_child_file_reads() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let root = tempfile::tempdir().expect("root");
    let private_child = root.path().join("private.txt");
    std::fs::write(&private_child, "NOT_AUTHORIZED").expect("private fixture");

    let request = confined_terminal_execution_request(
        &backend,
        "/bin/cat private.txt",
        root.path().to_str(),
        &[],
        &[],
    )
    .await
    .expect("confined request");
    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");

    assert_ne!(output.exit_code, 0, "cwd widened task-data authority");
    assert!(!output.stdout_lossy().contains("NOT_AUTHORIZED"));
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn exact_read_grant_does_not_expose_a_sibling() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let root = tempfile::tempdir().expect("root");
    let allowed = root.path().join("allowed.txt");
    let sibling = root.path().join("sibling.txt");
    std::fs::write(&allowed, "ALLOWED").expect("allowed fixture");
    std::fs::write(&sibling, "SIBLING").expect("sibling fixture");
    let allowed_path = allowed.to_string_lossy().to_string();
    let allowed_request = confined_terminal_execution_request(
        &backend,
        &format!("/bin/cat '{}'", allowed.display()),
        None,
        std::slice::from_ref(&allowed_path),
        &[],
    )
    .await
    .expect("allowed request");
    let allowed_output = backend
        .execute(allowed_request, Duration::from_secs(30))
        .await
        .expect("allowed execution");
    assert_eq!(
        allowed_output.exit_code,
        0,
        "{}",
        allowed_output.stderr_lossy()
    );
    assert_eq!(allowed_output.stdout_lossy(), "ALLOWED");

    let sibling_request = confined_terminal_execution_request(
        &backend,
        &format!("/bin/cat '{}'", sibling.display()),
        None,
        &[allowed_path],
        &[],
    )
    .await
    .expect("sibling request");
    let sibling_output = backend
        .execute(sibling_request, Duration::from_secs(30))
        .await
        .expect("sibling execution");
    assert_ne!(sibling_output.exit_code, 0, "sibling read escaped");
    assert!(!sibling_output.stdout_lossy().contains("SIBLING"));
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn exact_write_grant_cannot_mutate_a_sibling() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let root = tempfile::tempdir().expect("root");
    let allowed = root.path().join("allowed");
    let sibling = root.path().join("sibling.txt");
    std::fs::write(&sibling, "UNCHANGED").expect("sibling fixture");
    let allowed_path = allowed.to_string_lossy().to_string();

    let request = confined_terminal_execution_request(
        &backend,
        &format!(
            "/bin/mkdir '{}' && /usr/bin/touch '{}'",
            allowed.display(),
            sibling.display()
        ),
        None,
        &[],
        std::slice::from_ref(&allowed_path),
    )
    .await
    .expect("confined request");
    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");
    assert_ne!(output.exit_code, 0, "sibling write escaped");
    assert!(
        allowed.is_dir(),
        "authorized future root was not created: {}",
        output.stderr_lossy()
    );
    assert_eq!(
        std::fs::read_to_string(&sibling).expect("sibling contents"),
        "UNCHANGED"
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn manifest_grants_cannot_escape_through_an_internal_symlink() {
    use std::os::unix::fs::symlink;

    let backend = aidaemon_tools::testing::active_execution_backend();
    let root = tempfile::tempdir().expect("root");
    let allowed = root.path().join("allowed");
    let outside = root.path().join("outside.txt");
    std::fs::create_dir(&allowed).expect("allowed directory");
    std::fs::write(&outside, "UNCHANGED").expect("outside fixture");
    let link = allowed.join("escape");
    symlink(&outside, &link).expect("escape symlink");
    let allowed_path = allowed.to_string_lossy().to_string();

    let read_request = confined_terminal_execution_request(
        &backend,
        &format!("/bin/cat '{}'", link.display()),
        None,
        std::slice::from_ref(&allowed_path),
        &[],
    )
    .await
    .expect("read request");
    let read_output = backend
        .execute(read_request, Duration::from_secs(30))
        .await
        .expect("read execution");
    assert_ne!(read_output.exit_code, 0, "symlink read escaped");

    let write_request = confined_terminal_execution_request(
        &backend,
        &format!("/usr/bin/printf MUTATED > '{}'", link.display()),
        None,
        &[],
        std::slice::from_ref(&allowed_path),
    )
    .await
    .expect("write request");
    let write_output = backend
        .execute(write_request, Duration::from_secs(30))
        .await
        .expect("write execution");
    assert_ne!(write_output.exit_code, 0, "symlink write escaped");
    assert_eq!(
        std::fs::read_to_string(&outside).expect("outside contents"),
        "UNCHANGED"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn macos_runtime_policy_does_not_widen_task_data_roots() {
    let policy = macos_manifest_sandbox_policy(
        None,
        &["/private/tmp".to_string()],
        &[],
        &[],
        false,
        ConfinedCapabilities::default(),
    )
    .expect("policy");
    assert!(!policy.contains("(subpath \"/private/etc\")"));
    assert!(!policy.contains("(literal \"/private/etc/hosts\")"));
    assert!(policy.contains("(subpath \"/private/tmp\")"));
    assert!(policy.contains("(literal \"/private/etc/ssl/openssl.cnf\")"));
}

#[test]
fn confinement_is_a_single_switch_defaulting_to_host() {
    use aidaemon_tools::TerminalConfinement;
    assert_eq!(TerminalConfinement::default(), TerminalConfinement::Host);
    assert!(!run_is_confined(TerminalConfinement::Host));
    assert!(run_is_confined(TerminalConfinement::Sandbox));
    let parsed: aidaemon_tools::TerminalConfinement = serde_json::from_str("\"sandbox\"").unwrap();
    assert_eq!(parsed, TerminalConfinement::Sandbox);
}

#[tokio::test]
async fn host_request_runs_the_command_unwrapped_in_the_working_dir() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let request =
        host_terminal_execution_request(&backend, "npx wrangler deploy", false, Some("/tmp"))
            .await
            .expect("host request");
    match &request.command {
        aidaemon_tools::CommandSpec::Shell(command) => {
            assert_eq!(command, "npx wrangler deploy")
        }
        other => panic!("expected plain shell command, got {other:?}"),
    }
    assert!(
        request
            .cwd
            .as_ref()
            .is_some_and(|cwd| cwd.as_str().ends_with("tmp"))
    );
    assert!(request.stdin.is_none());
    let script = host_terminal_execution_request(&backend, "echo one\necho two", true, None)
        .await
        .expect("host script request");
    match &script.command {
        aidaemon_tools::CommandSpec::Argv { program, args } => {
            assert_eq!(program, "/bin/sh");
            assert_eq!(
                args,
                &vec![
                    "-eu".to_string(),
                    "-c".to_string(),
                    SCRIPT_BOOTSTRAP.to_string()
                ]
            );
        }
        other => panic!("expected the script bootstrap, got {other:?}"),
    }
    assert_eq!(
        script.stdin.as_deref(),
        Some(b"echo one\necho two".as_slice()),
        "the script arrives on stdin, never in the process list"
    );
}

/// A command in a script that reads stdin no longer swallows the lines
/// after it: every line runs (live 2026-11-20: a `ddev` line ate the rest
/// of a Drupal setup script and the run still exited 0).
#[cfg(unix)]
#[tokio::test]
async fn a_command_reading_stdin_does_not_swallow_the_rest_of_a_script() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let script = "echo first\ncat >/dev/null\nread -r line || true\n\
                      cat <<'EOF'\nfrom a heredoc\nEOF\necho last";
    let request = host_terminal_execution_request(&backend, script, true, None)
        .await
        .expect("host script request");
    let output = backend
        .execute(request, std::time::Duration::from_secs(10))
        .await
        .expect("script runs");
    assert_eq!(output.exit_code, 0, "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("first") && stdout.contains("last"),
        "{stdout}"
    );
    assert!(stdout.contains("from a heredoc"), "{stdout}");
    // A failing step still stops the script.
    let request = host_terminal_execution_request(&backend, "echo a\nfalse\necho b", true, None)
        .await
        .expect("host script request");
    let output = backend
        .execute(request, std::time::Duration::from_secs(10))
        .await
        .expect("script runs");
    assert_ne!(output.exit_code, 0);
    assert!(!String::from_utf8_lossy(&output.stdout).contains('b'));
}

#[cfg(target_os = "macos")]
#[test]
fn macos_policy_grants_network_only_when_capability_granted() {
    let confined = macos_manifest_sandbox_policy(
        Some("/Users/alice/projects/site"),
        &[],
        &[],
        &[],
        false,
        ConfinedCapabilities::default(),
    )
    .expect("policy");
    assert!(!confined.contains("(allow network-outbound)\n"));
    assert!(!confined.contains("(allow system-socket)"));
    let networked = macos_manifest_sandbox_policy(
        Some("/Users/alice/projects/site"),
        &[],
        &[],
        &[],
        false,
        ConfinedCapabilities {
            network: true,
            ..ConfinedCapabilities::default()
        },
    )
    .expect("policy");
    assert!(networked.contains("(allow network-outbound)\n"));
    assert!(networked.contains("(allow system-socket)"));
}

#[test]
fn requested_capabilities_are_typed_grant_keys() {
    let args: TerminalArgs = serde_json::from_str(
        r#"{"action":"run","command":"npx wrangler deploy","network":true,
                "_scope_escalation":{"read_paths":["/Users/alice/Library/Preferences/.spoofed"]}}"#,
    )
    .unwrap();
    // The escalation comes from the dispatcher's typed context; an
    // argument naming one is ignored.
    let trust = TerminalTrust {
        scope_escalation: Some(crate::tools::traits::ScopeEscalation {
            read_paths: vec!["/Users/alice/Library/Preferences/.wrangler".to_string()],
            write_paths: vec!["/Users/alice/Library/Preferences/.wrangler/logs".to_string()],
        }),
        ..TerminalTrust::default()
    };
    let caps = TerminalTool::requested_capabilities(true, &args, &trust);
    assert_eq!(
        caps,
        vec![
            "network".to_string(),
            "read:/Users/alice/Library/Preferences/.wrangler".to_string(),
            "write:/Users/alice/Library/Preferences/.wrangler/logs".to_string(),
        ]
    );
    let described = TerminalTool::describe_capabilities(&caps);
    assert!(described.contains("outbound network access"));
    assert!(
        described.contains("read outside task scope: /Users/alice/Library/Preferences/.wrangler")
    );
    assert!(
        described
            .contains("write outside task scope: /Users/alice/Library/Preferences/.wrangler/logs")
    );
    let plain: TerminalArgs = serde_json::from_str(r#"{"action":"run","command":"ls"}"#).unwrap();
    assert!(
        TerminalTool::requested_capabilities(true, &plain, &TerminalTrust::default()).is_empty()
    );
    // On the host there is no sandbox grant to escalate.
    assert!(TerminalTool::requested_capabilities(false, &args, &trust).is_empty());
}

#[test]
fn command_grant_keys_cover_each_chained_binary() {
    assert_eq!(
        command_grant_keys("npx wrangler deploy"),
        vec!["npx".to_string()]
    );
    assert_eq!(
        command_grant_keys("npm run build && npx wrangler deploy"),
        vec!["npm".to_string(), "npx".to_string()]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn macos_cwd_policy_makes_npm_ancestor_bin_lookups_fail_with_enoent() {
    // npm's run-script prepends `<ancestor>/node_modules/.bin` for every
    // ancestor of the project to the child PATH and spawns bare `sh`.
    // execvp(3) keeps walking on ENOENT/EACCES but aborts on EPERM, which
    // is what seatbelt returns for an undeclared path — so an npm child
    // died with `spawn EPERM` (exit 255, no output). Metadata-only grants
    // on those exact entries turn the denial into ENOENT.
    let policy = macos_manifest_sandbox_policy(
        Some("/Users/alice/projects/site"),
        &["/Users/alice/projects/site".to_string()],
        &[],
        &[],
        false,
        ConfinedCapabilities::default(),
    )
    .expect("policy");
    for ancestor in ["/Users/alice/projects", "/Users/alice", "/Users"] {
        let clause = format!(
            "(literal \"{ancestor}/node_modules\") (subpath \"{ancestor}/node_modules/.bin\")"
        );
        assert!(policy.contains(&clause), "missing {clause} in {policy}");
    }
    assert!(policy.contains("(literal \"/node_modules\") (subpath \"/node_modules/.bin\")"));
    // Metadata only: never file contents of a sibling tree.
    let section = policy
        .split("(allow file-read-metadata file-test-existence")
        .nth(1)
        .expect("metadata grant section");
    assert!(
        !section
            .split("\n(")
            .next()
            .unwrap_or("")
            .contains("file-read*")
    );
    assert!(!policy.contains("(subpath \"/Users/alice/node_modules\")"));
}

#[test]
fn native_sandbox_search_path_keeps_only_granted_and_base_system_dirs() {
    let home = std::path::Path::new("/Users/alice");
    let path = native_sandbox_search_path_from(
        &["/Users/alice/miniforge3/bin".to_string()],
        &["/Users/alice/miniforge3".to_string()],
        Some(home),
        "/Users/alice/.local/bin:/Users/alice/.nvm/versions/node/v24/bin:/opt/homebrew/bin:/opt/pmk/env/global/bin:/Users/alice/miniforge3/condabin",
    );
    let entries = path.split(':').collect::<Vec<_>>();
    // Granted manifest roots survive, in order, ahead of the system dirs.
    assert_eq!(entries[0], "/Users/alice/miniforge3/bin");
    assert!(entries.contains(&"/Users/alice/miniforge3/condabin"));
    // Base system dirs are always present so bare `sh`/`cc`/`env` resolve.
    assert!(entries.contains(&"/usr/bin"));
    assert!(entries.contains(&"/bin"));
    // Ungranted home dirs are dropped (they were before this fix too).
    assert!(!entries.iter().any(|e| e.starts_with("/Users/alice/.local")));
    assert!(!entries.iter().any(|e| e.starts_with("/Users/alice/.nvm")));
    // The actual bug: ungranted NON-home dirs must be dropped as well. A
    // single one of these earlier in PATH than /bin makes execvp abort with
    // EPERM, so a confined `sh -c` spawning a bare `sh` fails with an opaque
    // exit 255. They contribute nothing (their binaries are not grantable).
    assert!(
        !entries.contains(&"/opt/homebrew/bin"),
        "ungranted /opt/homebrew/bin must not appear in the confined PATH"
    );
    assert!(
        !entries.contains(&"/opt/pmk/env/global/bin"),
        "ungranted /opt/pmk/... must not appear in the confined PATH"
    );
}

#[test]
fn native_sandbox_search_path_keeps_explicitly_granted_nonhome_dir() {
    // If a homebrew (or other non-home) directory IS granted for this
    // invocation, it must be preserved — the drop only targets UNGRANTED
    // directories, never grants the daemon deliberately made.
    let path = native_sandbox_search_path_from(
        &["/opt/homebrew/bin".to_string()],
        &["/opt/homebrew".to_string()],
        Some(std::path::Path::new("/Users/alice")),
        "/opt/homebrew/bin:/opt/homebrew/opt/node/bin:/opt/other/bin",
    );
    let entries = path.split(':').collect::<Vec<_>>();
    assert!(entries.contains(&"/opt/homebrew/bin"));
    assert!(entries.contains(&"/opt/homebrew/opt/node/bin"));
    assert!(
        !entries.contains(&"/opt/other/bin"),
        "an ungranted sibling is still dropped"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn macos_root_cwd_policy_omits_empty_ancestor_relationship() {
    let policy = macos_manifest_sandbox_policy(
        Some("/"),
        &[],
        &[],
        &[],
        false,
        ConfinedCapabilities::default(),
    )
    .expect("root cwd policy");
    assert!(policy.contains("(literal \"/\")"));
    assert!(!policy.contains("(path-ancestors \"/\")"));
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn confined_root_cwd_preserves_compound_negative_process_result() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let request = confined_terminal_execution_request(
        &backend,
        "/bin/sh -c 'printf SYNTHETIC_BEFORE; /usr/bin/false'",
        Some("/"),
        &[],
        &[],
    )
    .await
    .expect("confined root request");
    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");
    assert_eq!(output.exit_code, 1, "{}", output.stderr_lossy());
    assert_eq!(output.stdout_lossy(), "SYNTHETIC_BEFORE");
}

#[test]
fn terminal_argument_contract_does_not_infer_effects_from_shell_words() {
    assert!(validate_terminal_argument_contract(
        r#"{"action":"run","command":"/usr/bin/touch /tmp/synthetic-target","working_dir":"/tmp"}"#
    )
    .is_ok());
    assert!(validate_terminal_argument_contract(
        r#"{"action":"run","command":"/usr/bin/touch /tmp/synthetic-target","working_dir":"/tmp","write_paths":["/tmp/synthetic-target"]}"#
    )
    .is_ok());
    assert!(
        validate_terminal_argument_contract(
            r#"{"action":"run","command":"/usr/bin/false","working_dir":"/tmp"}"#
        )
        .is_ok()
    );
    assert!(validate_terminal_argument_contract(
        r#"{"action":"run","command":"/opt/homebrew/bin/python3 -c 'print(\"SYNTHETIC_OK\")'","working_dir":"/tmp","read_paths":[],"write_paths":[]}"#
    )
    .is_ok());
    assert!(
        validate_terminal_argument_contract(
            r#"{"action":"run","command":"python3 -c 'print(1)'"}"#
        )
        .is_ok()
    );
    assert!(validate_terminal_argument_contract(
        r#"{"action":"run","script":"/usr/bin/false\nprintf SYNTHETIC_UNREACHED","working_dir":"/tmp"}"#
    )
    .is_ok());
    assert!(validate_terminal_argument_contract(
        r#"{"action":"run","command":"/usr/bin/true","script":"/usr/bin/true","working_dir":"/tmp"}"#
    )
    .is_err());
}

#[test]
fn terminal_manifest_keeps_execution_location_separate_from_read_authority() {
    let manifest = terminal_access_manifest(
        r#"{"action":"run","command":"/usr/bin/false","working_dir":"/tmp","read_paths":[],"write_paths":[]}"#,
    );
    assert_eq!(manifest.execution_cwd.as_deref(), Some("/tmp"));
    assert!(manifest.read_targets.is_empty());
    assert!(manifest.write_targets.is_empty());
}

#[test]
fn terminal_manifest_preserves_typed_directory_roots() {
    let manifest = terminal_access_manifest(
        r#"{"action":"run","command":"/usr/bin/true","working_dir":"/tmp","read_roots":["/tmp/read-root"],"write_roots":["/tmp/write-root"]}"#,
    );
    assert_eq!(
        manifest.read_targets[0].kind,
        ToolTargetHintKind::ProjectScope
    );
    assert_eq!(
        manifest.write_targets[0].kind,
        ToolTargetHintKind::ProjectScope
    );
    assert_eq!(manifest.write_targets[0].value, "/tmp/write-root");
}

#[test]
fn terminal_canonicalization_normalizes_empty_inactive_union_arms() {
    for input in [
        r#"{"action":"run","command":"/usr/bin/true","script":"","working_dir":"/tmp"}"#,
        r#"{"action":"run","command":"","script":"/usr/bin/true","working_dir":"/tmp"}"#,
    ] {
        let canonical = canonicalize_terminal_arguments(input).expect("canonical arguments");
        validate_terminal_argument_contract(&canonical).expect("valid tagged union");
        let value: Value = serde_json::from_str(&canonical).expect("json");
        let object = value.as_object().expect("object");
        assert_eq!(
            usize::from(object.contains_key("command"))
                + usize::from(object.contains_key("script")),
            1
        );
    }
}

#[tokio::test]
async fn confined_script_uses_stdin_and_stops_after_negative_step() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("codex").await.unwrap().is_none() {
        return;
    }
    let script = "printf 'SYNTHETIC_BEFORE\\n'\n/usr/bin/false\nprintf 'SYNTHETIC_AFTER\\n'\n";
    let request = confined_terminal_script_execution_request(
        &backend,
        script,
        Some("/tmp"),
        &["/tmp".to_string()],
        &[],
        &[],
        ConfinedCapabilities::default(),
    )
    .await
    .expect("confined script request");
    let serialized_args = match &request.command {
        aidaemon_tools::CommandSpec::Argv { args, .. } => args.join("\n"),
        aidaemon_tools::CommandSpec::Shell(_) => panic!("expected native sandbox argv"),
    };
    assert!(!serialized_args.contains("SYNTHETIC_BEFORE"));
    assert_eq!(request.stdin.as_deref(), Some(script.as_bytes()));

    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");
    assert_eq!(output.exit_code, 1, "{}", output.stderr_lossy());
    assert_eq!(output.stdout_lossy().trim(), "SYNTHETIC_BEFORE");
    assert!(!output.stdout_lossy().contains("SYNTHETIC_AFTER"));
}

#[test]
fn roots_nested_under_a_future_grant_are_left_to_the_command() {
    let cwd = "/tmp/synthetic-project".to_string();
    let dir = "/tmp/synthetic-project/card-root".to_string();
    let file = "/tmp/synthetic-project/card-root/result.txt".to_string();
    let only_cwd_exists = |path: &str| (path == cwd).then_some(true);
    // A root under a declared *future* root carries no capability; it is
    // the command's target and must not be pre-created.
    assert_eq!(
        select_materialized_write_roots(&[dir.clone(), file.clone()], &[], only_cwd_exists),
        vec![dir.clone()]
    );
    // Order and trailing slashes do not matter.
    assert_eq!(
        select_materialized_write_roots(&[file.clone(), format!("{dir}/")], &[], only_cwd_exists),
        vec![format!("{dir}/")]
    );
    // A future exact write path covers descendant roots the same way.
    assert_eq!(
        select_materialized_write_roots(
            std::slice::from_ref(&file),
            std::slice::from_ref(&dir),
            only_cwd_exists
        ),
        Vec::<String>::new()
    );
    // A future output directory under an EXISTING grant is still prepared
    // (a project's `dist`): that entry is the only way to request it.
    assert_eq!(
        select_materialized_write_roots(&[cwd.clone(), dir.clone()], &[], only_cwd_exists),
        vec![cwd.clone(), dir.clone()]
    );
    assert_eq!(
        select_materialized_write_roots(
            std::slice::from_ref(&dir),
            std::slice::from_ref(&cwd),
            only_cwd_exists
        ),
        vec![dir.clone()]
    );
    // The same FUTURE path declared as both an exact write path and a
    // root is a contradiction; the exact form wins and nothing is
    // materialized (the command owns that target).
    assert_eq!(
        select_materialized_write_roots(
            std::slice::from_ref(&file),
            std::slice::from_ref(&file),
            only_cwd_exists
        ),
        Vec::<String>::new()
    );
    // An existing directory declared in both forms keeps its root grant.
    assert_eq!(
        select_materialized_write_roots(
            std::slice::from_ref(&cwd),
            std::slice::from_ref(&cwd),
            only_cwd_exists
        ),
        vec![cwd.clone()]
    );
    // Sibling prefixes are not ancestors.
    assert_eq!(
        select_materialized_write_roots(
            &["/tmp/a".to_string(), "/tmp/ab".to_string()],
            &[],
            |_| None
        ),
        vec!["/tmp/a".to_string(), "/tmp/ab".to_string()]
    );
    // A root that already exists as a FILE is an exact target: never
    // materialized (create_dir_all would fail with EEXIST), whatever
    // role it was declared in.
    let file_exists = |path: &str| (path == file).then_some(false);
    assert_eq!(
        select_materialized_write_roots(std::slice::from_ref(&file), &[], file_exists),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn confined_run_does_not_turn_a_declared_descendant_root_into_a_directory() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("codex").await.unwrap().is_none()
        || backend
            .resolve_executable("printf")
            .await
            .unwrap()
            .is_none()
    {
        return;
    }
    let parent = tempfile::tempdir().expect("parent");
    let cwd = parent.path().to_string_lossy().to_string();
    let root = parent.path().join("synthetic-card-root");
    let file = root.join("result.txt");
    assert!(!root.exists());
    // The model declared both the future directory and the future file as
    // write roots (a common type confusion). Only the directory may be
    // prepared; the file must be left for the command's redirect.
    let script = format!(
        "mkdir -p '{}' && printf 'SYNTHETIC_CARD_OK' > '{}'\n",
        root.display(),
        file.display()
    );
    let request = confined_terminal_script_execution_request(
        &backend,
        &script,
        Some(&cwd),
        std::slice::from_ref(&cwd),
        &[],
        &[
            root.to_string_lossy().to_string(),
            file.to_string_lossy().to_string(),
        ],
        ConfinedCapabilities::default(),
    )
    .await
    .expect("confined script request");
    assert!(root.is_dir(), "topmost future root is prepared");
    assert!(
        !file.exists(),
        "a declared descendant root must not be pre-created as a directory"
    );
    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");
    assert_eq!(output.exit_code, 0, "{}", output.stderr_lossy());
    assert_eq!(
        std::fs::read_to_string(&file).expect("redirect output"),
        "SYNTHETIC_CARD_OK"
    );
}

#[tokio::test]
async fn confined_script_prepares_future_declared_scratch_before_heredoc() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("codex").await.unwrap().is_none()
        || backend.resolve_executable("cat").await.unwrap().is_none()
    {
        return;
    }
    let parent = tempfile::tempdir().expect("parent");
    let cwd = parent.path().to_string_lossy().to_string();
    let root = parent.path().join("synthetic-heredoc-root");
    assert!(!root.exists());
    let script = "touch \"$TMPDIR/probe\"\ncat <<'EOF' > \"$PWD/synthetic-heredoc-root/value.txt\"\nSYNTHETIC_HEREDOC_OK\nEOF\n";
    let request = confined_terminal_script_execution_request(
        &backend,
        script,
        Some(&cwd),
        std::slice::from_ref(&cwd),
        &[],
        &[cwd.clone(), root.to_string_lossy().to_string()],
        ConfinedCapabilities::default(),
    )
    .await
    .expect("confined script request");
    assert!(
        root.is_dir(),
        "runtime scratch root must exist before shell parsing"
    );
    // TMPDIR points at the private managed scratch, not the command's
    // declared output root (which the command may empty).
    let tmpdir = request.env.get("TMPDIR").cloned().expect("TMPDIR");
    assert!(tmpdir.contains("aidaemon-scratch"));
    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");
    assert_eq!(output.exit_code, 0, "{}", output.stderr_lossy());
    assert_eq!(
        std::fs::read_to_string(root.join("value.txt")).expect("heredoc output"),
        "SYNTHETIC_HEREDOC_OK\n"
    );
}

#[test]
fn nested_shell_programs_share_runtime_dependency_discovery() {
    let programs = parsed_command_programs(
        "/bin/sh -lc '/opt/homebrew/bin/python3 -c '\"'\"'print(1)'\"'\"''",
    );
    assert!(programs.contains(&"/bin/sh".to_string()));
    assert!(programs.contains(&"/opt/homebrew/bin/python3".to_string()));
}

#[tokio::test]
async fn nested_managed_runtime_executes_through_combined_shell_flags() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("codex").await.unwrap().is_none() {
        return;
    }
    let Some(python) = backend.resolve_executable("python3").await.unwrap() else {
        return;
    };
    let command = format!(
        r#"/bin/sh -lc '{} -c "print(\"SYNTHETIC_NESTED_PYTHON_OK\")"'"#,
        python
    );
    let request = confined_terminal_execution_request(
        &backend,
        &command,
        Some("/tmp"),
        &["/tmp".to_string()],
        &[],
    )
    .await
    .expect("confined request");
    let output = backend
        .execute(request, Duration::from_secs(30))
        .await
        .expect("sandbox execution");
    assert_eq!(output.exit_code, 0, "{}", output.stderr_lossy());
    assert_eq!(output.stdout_lossy().trim(), "SYNTHETIC_NESTED_PYTHON_OK");
}

#[test]
fn linux_sandbox_prefers_codex_and_falls_back_to_bubblewrap() {
    let codex = aidaemon_tools::BackendPath::new("/opt/node/bin/codex");
    let bwrap = aidaemon_tools::BackendPath::new("/usr/bin/bwrap");
    assert_eq!(
        select_linux_sandbox_adapter(Some(codex.clone()), Some(bwrap.clone())).unwrap(),
        LinuxSandboxAdapter::Codex(codex.clone())
    );
    assert_eq!(
        select_linux_sandbox_adapter(Some(codex.clone()), None).unwrap(),
        LinuxSandboxAdapter::Codex(codex)
    );
    assert_eq!(
        select_linux_sandbox_adapter(None, Some(bwrap.clone())).unwrap(),
        LinuxSandboxAdapter::Bubblewrap(bwrap)
    );
    // Neither: fail closed, naming both ways out, never the host.
    let error = select_linux_sandbox_adapter(None, None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("codex sandbox") && error.contains("bwrap"),
        "{error}"
    );
}

#[test]
fn bubblewrap_fallback_gets_the_codex_manifest() {
    let confined = vec![
        "/usr/bin/env".to_string(),
        "PATH=/usr/bin".into(),
        "/bin/sh".into(),
        "-c".into(),
        "ls".into(),
    ];
    let args = bubblewrap_sandbox_args(
        "/work/repo",
        &["/work/repo".to_string()],
        &["/work/repo/out".to_string()],
        ConfinedCapabilities {
            network: false,
            ..ConfinedCapabilities::default()
        },
        &confined,
    );
    let joined = args.join(" ");
    assert!(
        joined.contains("--ro-bind-try /work/repo /work/repo"),
        "{joined}"
    );
    assert!(
        joined.contains("--bind-try /work/repo/out /work/repo/out"),
        "{joined}"
    );
    assert!(args.contains(&"--unshare-net".to_string()));
    assert!(joined.ends_with("--chdir /work/repo -- /usr/bin/env PATH=/usr/bin /bin/sh -c ls"));
    let networked = bubblewrap_sandbox_args(
        "/",
        &[],
        &[],
        ConfinedCapabilities {
            network: true,
            ..ConfinedCapabilities::default()
        },
        &confined,
    );
    assert!(!networked.contains(&"--unshare-net".to_string()));
}

#[test]
fn linux_sandboxes_keep_persistence_paths_read_only_in_write_roots() {
    let confined = vec!["/bin/true".to_string()];
    let protect = ConfinedCapabilities {
        protect_persistence: true,
        ..ConfinedCapabilities::default()
    };
    let args = bubblewrap_sandbox_args(
        "/work/repo",
        &[],
        &["/work/repo".to_string()],
        protect,
        &confined,
    );
    let joined = args.join(" ");
    let writable = joined
        .find("--bind-try /work/repo /work/repo")
        .expect("write root bound");
    for sub in PERSISTENCE_SUBPATHS {
        let mount = format!("--ro-bind-try /work/repo/{sub} /work/repo/{sub}");
        let at = joined
            .find(&mount)
            .unwrap_or_else(|| panic!("{mount}: {joined}"));
        // Later mounts win: the read-only bind must follow the write root
        // and precede the namespace flags.
        assert!(at > writable, "{joined}");
        assert!(at < joined.find("--unshare-pid").unwrap(), "{joined}");
    }
    let unprotected = bubblewrap_sandbox_args(
        "/work/repo",
        &[],
        &["/work/repo".to_string()],
        ConfinedCapabilities::default(),
        &confined,
    )
    .join(" ");
    assert!(!unprotected.contains(".git/hooks"), "{unprotected}");

    // Codex gets entries only for persistence paths that exist: its
    // bubblewrap cannot mount under a write root not created yet.
    let absent = codex_sandbox_state_json("/work/repo", &[], &["/work/repo".to_string()], protect)
        .expect("sandbox state");
    assert!(!absent.contains(".git/hooks"), "{absent}");
    let repo = tempfile::tempdir().expect("repo");
    for sub in PERSISTENCE_SUBPATHS {
        std::fs::create_dir_all(repo.path().join(sub)).expect("persistence path");
    }
    let root = repo.path().to_string_lossy().to_string();
    let state = codex_sandbox_state_json(&root, &[], std::slice::from_ref(&root), protect)
        .expect("sandbox state");
    let state: Value = serde_json::from_str(&state).expect("valid JSON");
    let entries = state
        .pointer("/permissionProfile/file_system/entries")
        .and_then(Value::as_array)
        .expect("filesystem entries");
    for sub in PERSISTENCE_SUBPATHS {
        let path = format!("{root}/{sub}");
        assert!(
            entries.iter().any(|entry| {
                entry.pointer("/path/path").and_then(Value::as_str) == Some(path.as_str())
                    && entry.get("access").and_then(Value::as_str) == Some("read")
            }),
            "{path}"
        );
    }
}

#[test]
fn linux_runs_create_a_missing_hooks_folder_only_in_an_existing_repository() {
    let dirs = [
        "/work/repo/.git",
        "/work/hooked/.git",
        "/work/hooked/.git/hooks",
    ];
    let is_dir = |path: &str| dirs.contains(&path);
    let exists = |path: &str| dirs.contains(&path);
    let roots = [
        // A repository without a hooks folder: create it.
        "/work/repo/".to_string(),
        // One that has it: nothing to do.
        "/work/hooked".to_string(),
        // Not a repository, or a root the run has yet to create: nothing
        // (Codex's bubblewrap cannot mount under a missing root).
        "/work/plain".to_string(),
        "/work/not-yet".to_string(),
        "/work/repo".to_string(),
    ];
    assert_eq!(
        persistence_dirs_to_create(&roots, &is_dir, &exists),
        vec!["/work/repo/.git/hooks".to_string()]
    );
    // A worktree's `.git` is a file: its hooks live in the common dir.
    let file_git = |_: &str| false;
    let exists_file = |path: &str| path == "/work/tree/.git";
    assert!(
        persistence_dirs_to_create(&["/work/tree".to_string()], &file_git, &exists_file).is_empty()
    );
}

#[test]
fn native_sandbox_state_preserves_exact_dotted_and_split_paths() {
    let state = codex_sandbox_state_json(
        "/synthetic/work.tree",
        &["/synthetic/read.only/.cache".to_string()],
        &["/synthetic/output.file".to_string()],
        ConfinedCapabilities::default(),
    )
    .expect("sandbox state");
    let state: Value = serde_json::from_str(&state).expect("valid JSON");
    assert_eq!(
        state.pointer("/sandboxCwd").and_then(Value::as_str),
        Some("file:///synthetic/work.tree")
    );
    let entries = state
        .pointer("/permissionProfile/file_system/entries")
        .and_then(Value::as_array)
        .expect("filesystem entries");
    assert!(entries.iter().any(|entry| {
        entry.pointer("/path/path").and_then(Value::as_str) == Some("/synthetic/read.only/.cache")
            && entry.get("access").and_then(Value::as_str) == Some("read")
    }));
    assert!(entries.iter().any(|entry| {
        entry.pointer("/path/path").and_then(Value::as_str) == Some("/synthetic/output.file")
            && entry.get("access").and_then(Value::as_str) == Some("write")
    }));
    assert_eq!(
        state
            .pointer("/permissionProfile/network")
            .and_then(Value::as_str),
        Some("restricted")
    );
}

#[tokio::test]
async fn native_sandbox_runtime_grants_node_libraries_for_package_runners() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    let Some(node) = backend.resolve_executable("node").await.unwrap() else {
        return;
    };
    if backend.resolve_executable("npm").await.unwrap().is_none() {
        return;
    }
    let canonical = backend.canonicalize(&node).await.unwrap_or(node.clone());
    let lib = std::path::Path::new(canonical.as_str())
        .parent()
        .and_then(|bin| bin.parent())
        .map(|prefix| prefix.join("lib"));
    let Some(lib) = lib.filter(|lib| lib.is_dir()) else {
        return;
    };
    let lib = lib.to_string_lossy().to_string();
    let npm = native_sandbox_runtime_support(&backend, "npm run build")
        .await
        .expect("npm support");
    assert!(
        npm.read_paths.iter().any(|path| path == &lib),
        "npm must carry node's lib: {:?}",
        npm.read_paths
    );
    assert!(npm.executable_paths.iter().any(|path| path == &lib));
    let plain = native_sandbox_runtime_support(&backend, "/usr/bin/true")
        .await
        .expect("plain support");
    assert!(!plain.read_paths.iter().any(|path| path == &lib));
}

#[tokio::test]
async fn confined_read_only_command_still_gets_a_writable_managed_scratch() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.kind() != aidaemon_tools::BackendKind::Local {
        return;
    }
    let project = tempfile::tempdir().expect("project");
    let cwd = project.path().to_string_lossy().to_string();
    // A read-only observation: no write paths, no write roots declared.
    let request = confined_terminal_execution_request(
        &backend,
        "node --version",
        Some(&cwd),
        std::slice::from_ref(&cwd),
        &[],
    )
    .await
    .expect("confined request");
    let scratch = request
        .env
        .get("TMPDIR")
        .cloned()
        .expect("managed scratch TMPDIR");
    assert_eq!(request.env.get("NPM_CONFIG_CACHE"), Some(&scratch));
    assert_eq!(request.env.get("XDG_CACHE_HOME"), Some(&scratch));
    // npm must be pointed at a neutral user config in the scratch (never
    // the owner's ~/.npmrc), and that file must exist before exec.
    let userconfig = request
        .env
        .get("NPM_CONFIG_USERCONFIG")
        .expect("neutral npm userconfig");
    assert!(userconfig.starts_with(&scratch));
    assert!(std::path::Path::new(userconfig).is_file());
    assert!(
        std::path::Path::new(&scratch).is_dir(),
        "scratch must exist before exec: {scratch}"
    );
    assert!(
        scratch.contains("aidaemon-scratch"),
        "scratch must live under the daemon scratch root: {scratch}"
    );
    // The sandbox policy must authorize writing that scratch.
    let serialized = match &request.command {
        aidaemon_tools::CommandSpec::Argv { args, .. } => args.join("\n"),
        aidaemon_tools::CommandSpec::Shell(_) => panic!("expected native sandbox argv"),
    };
    assert!(
        serialized.contains(&scratch),
        "scratch must be a write grant in the native sandbox policy"
    );
}

#[tokio::test]
async fn native_sandbox_runtime_grants_a_conda_prefix_by_its_marker() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    // Find any resolvable executable whose prefix carries `conda-meta`.
    let mut conda_prefix = None;
    for program in ["node", "python3", "python"] {
        if let Some(resolved) = backend.resolve_executable(program).await.unwrap() {
            let canonical = backend
                .canonicalize(&resolved)
                .await
                .unwrap_or(resolved.clone());
            for location in [resolved.as_str(), canonical.as_str()] {
                if let Some(prefix) = std::path::Path::new(location)
                    .parent()
                    .and_then(|bin| bin.parent())
                    && prefix.join("conda-meta").is_dir()
                {
                    conda_prefix = Some((program, prefix.to_string_lossy().to_string()));
                }
            }
        }
        if conda_prefix.is_some() {
            break;
        }
    }
    let Some((program, prefix)) = conda_prefix else {
        return;
    };
    let support = native_sandbox_runtime_support(&backend, &format!("{program} --version"))
        .await
        .expect("support");
    assert!(support.read_paths.iter().any(|path| path == &prefix));
    assert!(support.executable_paths.iter().any(|path| path == &prefix));
    assert!(
        !support
            .read_paths
            .iter()
            .any(|path| path == &backend.home_hint().to_string())
    );
}

#[tokio::test]
async fn native_sandbox_runtime_grants_git_its_own_config_files_only_for_git() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("git").await.unwrap().is_none() {
        return;
    }
    let gitconfig = backend.home_hint().join(".gitconfig");
    if backend.metadata(&gitconfig).await.is_err() {
        return;
    }
    let git = native_sandbox_runtime_support(&backend, "git status")
        .await
        .expect("git support");
    assert!(
        git.read_paths
            .iter()
            .any(|path| path == &gitconfig.to_string())
    );
    assert!(
        !git.read_paths
            .iter()
            .any(|path| path == &backend.home_hint().to_string())
    );
    let other = native_sandbox_runtime_support(&backend, "/usr/bin/true")
        .await
        .expect("other support");
    assert!(
        !other
            .read_paths
            .iter()
            .any(|path| path == &gitconfig.to_string())
    );
}

#[tokio::test]
async fn native_sandbox_runtime_supports_local_rust_toolchain_without_home_access() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("codex").await.unwrap().is_none()
        || backend.resolve_executable("cargo").await.unwrap().is_none()
    {
        return;
    }
    let project = tempfile::tempdir().expect("project");
    std::fs::create_dir_all(project.path().join("src")).expect("src");
    std::fs::write(
        project.path().join("Cargo.toml"),
        "[package]\nname = \"synthetic-sandbox-check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("manifest");
    std::fs::write(project.path().join("src/main.rs"), "fn main() {}\n").expect("source");
    let cwd = project.path().to_string_lossy().to_string();
    let request = confined_terminal_execution_request(
        &backend,
        "cargo check --offline --quiet",
        Some(&cwd),
        std::slice::from_ref(&cwd),
        std::slice::from_ref(&cwd),
    )
    .await
    .expect("confined request");
    let serialized = match &request.command {
        aidaemon_tools::CommandSpec::Argv { args, .. } => args.join("\n"),
        aidaemon_tools::CommandSpec::Shell(_) => panic!("expected native sandbox argv"),
    };
    assert!(!serialized.contains("credentials.toml"));
    assert!(!serialized.contains(&format!("\"path\":\"{}\"", backend.home_hint())));
    let sandbox_path = request.env.get("PATH").cloned().unwrap_or_default();
    let canonical_cwd = backend
        .canonicalize(&aidaemon_tools::BackendPath::new(cwd.clone()))
        .await
        .expect("canonical project root")
        .to_string();
    // TMPDIR/cache point at the private managed scratch (preferred over
    // the declared output dir, which build tools may empty). The scratch
    // must be a real writable directory and NOT the project cwd.
    let tmpdir = request.env.get("TMPDIR").cloned().expect("TMPDIR");
    assert!(tmpdir.contains("aidaemon-scratch"));
    assert!(std::path::Path::new(&tmpdir).is_dir());
    assert_ne!(tmpdir, canonical_cwd);
    assert_eq!(request.env.get("TMP"), Some(&tmpdir));
    assert_eq!(request.env.get("TEMP"), Some(&tmpdir));
    assert_eq!(request.env.get("XDG_CACHE_HOME"), Some(&tmpdir));
    let git_config = backend.home_hint().join(".gitconfig").to_string();
    if backend
        .metadata(&aidaemon_tools::BackendPath::new(git_config.clone()))
        .await
        .is_ok()
    {
        assert!(serialized.contains(&git_config));
    }
    let toolchain_index = sandbox_path.find("/.rustup/toolchains/");
    let shim_index = sandbox_path.find("/.cargo/bin");
    assert!(
        toolchain_index.is_some() && (shim_index.is_none() || toolchain_index < shim_index),
        "resolved toolchain must precede rustup shims: {sandbox_path}"
    );

    let output = backend
        .execute(request, Duration::from_secs(60))
        .await
        .expect("sandbox execution");
    assert_eq!(
        output.exit_code,
        0,
        "cargo stderr: {}\nsandbox PATH: {sandbox_path}",
        output.stderr_lossy(),
    );
}

#[tokio::test]
async fn confined_cargo_project_creation_uses_exact_runtime_config_grant() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("codex").await.unwrap().is_none()
        || backend.resolve_executable("cargo").await.unwrap().is_none()
    {
        return;
    }
    let root = tempfile::tempdir().expect("root");
    let cwd = root.path().to_string_lossy().to_string();
    let request = confined_terminal_execution_request(
        &backend,
        "cargo new synthetic-created --quiet",
        Some(&cwd),
        std::slice::from_ref(&cwd),
        std::slice::from_ref(&cwd),
    )
    .await
    .expect("confined request");
    let serialized = match &request.command {
        aidaemon_tools::CommandSpec::Argv { args, .. } => args.join("\n"),
        aidaemon_tools::CommandSpec::Shell(_) => panic!("expected native sandbox argv"),
    };
    let git_config = backend.home_hint().join(".gitconfig").to_string();
    if backend
        .metadata(&aidaemon_tools::BackendPath::new(git_config.clone()))
        .await
        .is_ok()
    {
        assert!(serialized.contains(&git_config));
    }
    assert!(!serialized.contains("credentials.toml"));

    let output = backend
        .execute(request, Duration::from_secs(60))
        .await
        .expect("sandbox execution");
    assert_eq!(
        output.exit_code,
        0,
        "cargo stderr: {}",
        output.stderr_lossy()
    );
    assert!(root.path().join("synthetic-created/src/main.rs").is_file());
}

#[tokio::test]
async fn confined_cargo_lifecycle_can_create_and_remove_exact_absent_root() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("codex").await.unwrap().is_none()
        || backend.resolve_executable("cargo").await.unwrap().is_none()
    {
        return;
    }
    let parent = tempfile::tempdir().expect("parent");
    let cwd = parent.path().to_string_lossy().to_string();
    let created = parent.path().join("synthetic-lifecycle");
    assert!(!created.exists());
    let request = confined_terminal_execution_request(
        &backend,
        "root=\"$PWD/synthetic-lifecycle\"; trap 'rm -rf -- \"$root\"' EXIT; cargo new --vcs none synthetic-lifecycle --quiet && cd synthetic-lifecycle && cargo build --quiet && cargo test --quiet && test \"$(cargo run --quiet)\" = 'Hello, world!'",
        Some(&cwd),
        std::slice::from_ref(&cwd),
        &[created.to_string_lossy().to_string()],
    )
    .await
    .expect("confined request");

    let output = backend
        .execute(request, Duration::from_secs(90))
        .await
        .expect("sandbox execution");
    assert_eq!(
        output.exit_code,
        0,
        "cargo stderr: {}",
        output.stderr_lossy()
    );
    assert!(!created.exists(), "cleanup trap must remove the exact root");
}

#[test]
fn managed_runtime_root_is_derived_from_installation_layout_not_tool_name() {
    assert_eq!(
        managed_executable_runtime_root(&aidaemon_tools::BackendPath::new(
            "/opt/homebrew/Cellar/python@3.14/3.14.6/bin/python3.14"
        )),
        Some("/opt/homebrew/Cellar/python@3.14/3.14.6".to_string())
    );
    assert_eq!(
        homebrew_stable_runtime_alias(&aidaemon_tools::BackendPath::new(
            "/opt/homebrew/Cellar/python@3.14/3.14.6/bin/python3.14"
        )),
        Some("/opt/homebrew/opt/python@3.14".to_string())
    );
    assert_eq!(
        homebrew_alias_namespace(&aidaemon_tools::BackendPath::new(
            "/opt/homebrew/Cellar/python@3.14/3.14.6/bin/python3.14"
        )),
        Some("/opt/homebrew/opt".to_string())
    );
    assert_eq!(
        managed_executable_runtime_root(&aidaemon_tools::BackendPath::new(
            "/synthetic/user/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc"
        )),
        Some("/synthetic/user/.rustup/toolchains/stable-aarch64-apple-darwin".to_string())
    );
    assert_eq!(
        managed_executable_runtime_root(&aidaemon_tools::BackendPath::new("/usr/bin/false")),
        None
    );
    #[cfg(target_os = "macos")]
    {
        assert_eq!(
            macos_developer_root_from_sdk_root(
                "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
            ),
            Some("/Library/Developer/CommandLineTools".to_string())
        );
        assert_eq!(
            macos_developer_root_from_sdk_root(
                "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
            ),
            Some("/Applications/Xcode.app/Contents/Developer".to_string())
        );
    }
}

#[tokio::test]
async fn confined_managed_python_can_create_run_and_remove_exact_venv_root() {
    let backend = aidaemon_tools::testing::active_execution_backend();
    if backend.resolve_executable("codex").await.unwrap().is_none()
        || backend
            .resolve_executable("python3")
            .await
            .unwrap()
            .is_none()
    {
        return;
    }
    let parent = tempfile::tempdir().expect("parent");
    let cwd = parent.path().to_string_lossy().to_string();
    let created = parent.path().join("synthetic-python-lifecycle");
    let request = confined_terminal_execution_request(
        &backend,
        "root=\"$PWD/synthetic-python-lifecycle\"; trap 'rm -rf -- \"$root\"' EXIT; mkdir \"$root\" && printf 'VALUE = 42\\n' > \"$root/module.py\" && python3 -m venv \"$root/.venv\" && \"$root/.venv/bin/python\" -c 'import sys; sys.path.insert(0, sys.argv[1]); import module; assert module.VALUE == 42' \"$root\"",
        Some(&cwd),
        std::slice::from_ref(&cwd),
        &[created.to_string_lossy().to_string()],
    )
    .await
    .expect("confined request");

    let output = backend
        .execute(request, Duration::from_secs(90))
        .await
        .expect("sandbox execution");
    assert_eq!(
        output.exit_code,
        0,
        "python stderr: {}",
        output.stderr_lossy()
    );
    assert!(!created.exists(), "cleanup trap must remove the exact root");
}

#[test]
fn terminal_semantics_resolve_unknown_syntax_from_confinement_capabilities() {
    let read_only = terminal_call_semantics(
        r#"{"action":"run","command":"/bin/sh -c 'printf \\'SYNTHETIC\\n\\''","working_dir":"/tmp","read_paths":[],"write_paths":[]}"#,
    );
    assert!(read_only.observes_state());
    assert!(!read_only.mutates_state());

    let writable = terminal_call_semantics(
        r#"{"action":"run","command":"opaque 'unterminated","working_dir":"/tmp","read_paths":[],"write_paths":["/tmp/synthetic-output"]}"#,
    );
    assert!(writable.mutates_state());
}

#[test]
fn a_run_with_the_network_changes_what_it_reaches_unless_it_only_reads() {
    // Live 2026-11-13: an install through the container engine declared
    // only read_paths, counted as a read, and never confirmed its step.
    let install = terminal_call_semantics(
        r#"{"action":"run","command":"ddev exec drush site:install -y","working_dir":"/tmp/site","read_paths":["/tmp/site"],"network":true}"#,
    );
    assert!(install.mutates_state());
    assert!(install.observes_state());

    let status = terminal_call_semantics(
        r#"{"action":"run","command":"ddev describe","working_dir":"/tmp/site","read_paths":["/tmp/site"],"network":true}"#,
    );
    assert!(!status.mutates_state());

    let fetch = terminal_call_semantics(
        r#"{"action":"run","command":"curl -sSL https://example.invalid","network":true}"#,
    );
    assert!(!fetch.mutates_state());
}

#[test]
fn a_command_that_writes_remotely_is_a_remote_mutation() {
    let post = terminal_call_semantics(
        r#"{"action":"run","command":"curl --request POST --data '{}' http://localhost:1/ping"}"#,
    );
    assert!(post.mutates_state());
    assert!(post.observes_state());
    assert!(
        post.mutation_effects
            .contains(ToolMutationEffects::REMOTE_MUTATION)
    );

    let fetch = terminal_call_semantics(
        r#"{"action":"run","command":"curl -sSL https://example.invalid"}"#,
    );
    assert!(
        !fetch
            .mutation_effects
            .contains(ToolMutationEffects::REMOTE_MUTATION)
    );
}

/// An unattended confined run may write its folder but not the places in it
/// whose content runs later (git hooks and configuration, husky hooks).
#[cfg(target_os = "macos")]
#[test]
fn unattended_confined_writes_exclude_persistence_subpaths() {
    let root = "/Users/alice/projects/site".to_string();
    let attended = macos_manifest_sandbox_policy(
        Some(&root),
        &[],
        std::slice::from_ref(&root),
        &[],
        false,
        ConfinedCapabilities::default(),
    )
    .expect("policy");
    assert!(!attended.contains("(deny file-write*"));
    let unattended = macos_manifest_sandbox_policy(
        Some(&root),
        &[],
        std::slice::from_ref(&root),
        &[],
        false,
        ConfinedCapabilities {
            protect_persistence: true,
            ..ConfinedCapabilities::default()
        },
    )
    .expect("policy");
    let deny = unattended.rfind("(deny file-write*").expect("deny rule");
    let allow = unattended
        .rfind("file-write*\n  (literal \"/Users/alice/projects/site\")")
        .expect("write grant");
    assert!(
        deny > allow,
        "the deny must come after the grant it narrows"
    );
    for sub in [".git/hooks", ".git/config", ".husky"] {
        assert!(
            unattended[deny..].contains(&format!("(subpath \"{root}/{sub}\")")),
            "{sub}"
        );
    }
}
