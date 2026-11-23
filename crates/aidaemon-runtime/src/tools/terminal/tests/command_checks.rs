//! Command-line checks and output formatting.

use super::*;

#[test]
fn test_detect_unscoped_recursive_grep_broad_path() {
    let detected = detect_unscoped_recursive_grep(r#"grep -rc "async fn" ."#);
    assert!(
        detected.is_some(),
        "expected broad recursive grep to be detected"
    );
    let (pattern, path) = detected.unwrap();
    assert_eq!(pattern, "async fn");
    assert_eq!(path, ".");
}

#[test]
fn test_detect_unscoped_recursive_grep_allows_scoped_dir() {
    let detected = detect_unscoped_recursive_grep(r#"grep -R "todo" src"#);
    assert!(
        detected.is_none(),
        "scoped directory search should be allowed"
    );
}

#[test]
fn test_detect_unscoped_recursive_grep_allows_excludes() {
    let detected = detect_unscoped_recursive_grep(
        r#"grep -R --exclude-dir=node_modules --exclude-dir=target "todo" ."#,
    );
    assert!(detected.is_none(), "grep with excludes should be allowed");
}

#[test]
fn test_detect_unscoped_recursive_grep_in_chained_shell_command() {
    let detected = detect_unscoped_recursive_grep(r#"cd /tmp/project && grep -rc "async fn" ."#);
    assert!(
        detected.is_some(),
        "expected chained command recursive grep to be detected"
    );
    let (pattern, path) = detected.unwrap();
    assert_eq!(pattern, "async fn");
    assert_eq!(path, ".");
}

// ── unbounded disk scan detector tests ──

#[test]
fn is_broad_scan_root_matches_root_and_home_only() {
    assert!(is_broad_scan_root("/"));
    assert!(is_broad_scan_root("~"));
    assert!(is_broad_scan_root("~/"));
    assert!(is_broad_scan_root("$HOME"));
    // Subdirectories must NOT be flagged.
    assert!(!is_broad_scan_root("~/projects"));
    assert!(!is_broad_scan_root("/var/log"));
    assert!(!is_broad_scan_root("/usr"));
    assert!(!is_broad_scan_root("."));
}

#[test]
fn detect_unbounded_disk_scan_flags_the_incident_commands() {
    // The exact commands seen grinding for minutes on the live daemon.
    assert!(detect_unbounded_disk_scan("du -a / 2>/dev/null | sort -rn | head -n 10").is_some());
    assert!(detect_unbounded_disk_scan("du -ah ~ | sort -rh | head -n 20").is_some());
    assert!(detect_unbounded_disk_scan(
        "cd '/' && find / -type f -size +100M -exec ls -lh {} + 2>/dev/null | sort -k5 -rh | head -n 10"
    ).is_some());
    assert!(detect_unbounded_disk_scan("find ~ -type f -size +500M").is_some());
}

#[test]
fn detect_unbounded_disk_scan_passes_scoped_and_bounded_commands() {
    // Scoped to a subdirectory → fine.
    assert!(detect_unbounded_disk_scan("du -sh ~/projects").is_none());
    assert!(detect_unbounded_disk_scan("find ~/Downloads -type f -size +500M").is_none());
    assert!(detect_unbounded_disk_scan("du -ah /var/log | sort -rh | head").is_none());
    // find with a depth limit → fine (maxdepth bounds traversal).
    assert!(detect_unbounded_disk_scan("find / -maxdepth 2 -type d").is_none());
    assert!(detect_unbounded_disk_scan("find ~ -maxdepth 3 -name '*.rs'").is_none());
    // Unrelated commands → fine.
    assert!(detect_unbounded_disk_scan("ls -la ~").is_none());
    assert!(detect_unbounded_disk_scan("echo hello").is_none());
}

#[test]
fn unbounded_scan_block_message_is_actionable() {
    let msg = unbounded_scan_block_message("find", "/");
    assert!(msg.to_lowercase().contains("scope") || msg.to_lowercase().contains("narrow"));
    assert!(msg.contains("maxdepth") || msg.contains("-size") || msg.contains("specific"));
}

// ── format_output tests ──

#[test]
fn test_humanize_elapsed() {
    assert_eq!(humanize_elapsed(0), "0s");
    assert_eq!(humanize_elapsed(40), "40s");
    assert_eq!(humanize_elapsed(59), "59s");
    assert_eq!(humanize_elapsed(60), "1m 0s");
    assert_eq!(humanize_elapsed(65), "1m 5s");
    assert_eq!(humanize_elapsed(3599), "59m 59s");
    assert_eq!(humanize_elapsed(3600), "1h 0m");
    assert_eq!(humanize_elapsed(3725), "1h 2m");
}

#[test]
fn self_heal_extracts_and_safety_filters_denied_config_paths() {
    let home = std::env::temp_dir().join(format!("aidaemon-heal-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    // An allowlisted NON-secret config the tool may need at startup.
    let nodever = home.join(".node-version");
    std::fs::write(&nodever, "22.13.0\n").unwrap();
    let nodever_canon = dunce::canonicalize(&nodever)
        .unwrap()
        .to_string_lossy()
        .to_string();
    // Secret-bearing files must NEVER be granted, even when named in a denial.
    let npmrc = home.join(".npmrc");
    std::fs::write(&npmrc, "//registry/:_authToken=SECRET\n").unwrap();
    let ssh = home.join(".ssh");
    std::fs::create_dir_all(&ssh).unwrap();
    let id = ssh.join("id_rsa");
    std::fs::write(&id, "SECRET").unwrap();
    let home_s = home.to_string_lossy().to_string();

    // Allowlisted basename → granted (basename equality after canonicalize).
    let out = format!(
        "EPERM: operation not permitted, open '{}'",
        nodever.display()
    );
    assert_eq!(
        self_healable_denied_reads(&out, &home_s, &[]),
        vec![nodever_canon.clone()]
    );
    // .npmrc holds an auth token → NEVER auto-granted even if a (possibly
    // attacker-controlled) build script prints a denial naming it.
    let forged = format!("EPERM: operation not permitted, open '{}'", npmrc.display());
    assert!(self_healable_denied_reads(&forged, &home_s, &[]).is_empty());
    // Key material → never.
    let secret_out = format!("Permission denied: {}", id.display());
    assert!(self_healable_denied_reads(&secret_out, &home_s, &[]).is_empty());
    // A non-denial line yields nothing.
    assert!(
        self_healable_denied_reads(&format!("read {}", nodever.display()), &home_s, &[]).is_empty()
    );
    // Already-granted paths are not repeated.
    assert!(self_healable_denied_reads(&out, &home_s, &[nodever_canon]).is_empty());
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn extract_absolute_paths_handles_quoted_and_bare_forms() {
    assert_eq!(
        extract_absolute_paths("open '/Users/x/.npmrc' failed"),
        vec!["/Users/x/.npmrc".to_string()]
    );
    assert_eq!(
        extract_absolute_paths("tried: /opt/homebrew/lib/libnode.dylib (no such file),"),
        vec!["/opt/homebrew/lib/libnode.dylib".to_string()]
    );
    assert!(extract_absolute_paths("no paths here").is_empty());
}

#[test]
fn signal_diagnostics_name_the_kill_and_point_at_a_remedy() {
    assert_eq!(signal_name(9), "SIGKILL");
    assert_eq!(signal_name(6), "SIGABRT");
    assert_eq!(signal_name(123), "unknown signal");
    // SIGKILL should point at the OOM/resource remedy, not a generic one.
    assert!(signal_remedy_hint(9).to_lowercase().contains("memory"));
    // SIGABRT should push toward running the binary directly / verbose.
    let abrt = signal_remedy_hint(6).to_lowercase();
    assert!(abrt.contains("directly") || abrt.contains("verbose"));
}

#[test]
fn opaque_confined_failure_gets_a_self_diagnostic_hint() {
    // Silent non-zero exit: a hint is produced.
    let hint = confined_opaque_failure_hint(Some(255), "", "  \n").expect("hint");
    assert!(hint.contains("$TMPDIR"));
    assert!(hint.contains("verbose") || hint.contains("--debug"));
    // Real error text suppresses the hint (the agent can see the cause).
    assert!(confined_opaque_failure_hint(Some(255), "", "Error: boom").is_none());
    // A wrapper banner on stdout with NO stderr still gets the hint: the
    // child's real error was lost (npm prints its banner, vite aborts).
    assert!(confined_opaque_failure_hint(Some(1), "> pkg build > vite build", "").is_some());
    // Success never gets a hint.
    assert!(confined_opaque_failure_hint(Some(0), "", "").is_none());
    assert!(confined_opaque_failure_hint(None, "", "").is_none());
}

#[test]
fn format_output_returns_truncation_info_instead_of_embedding_notice() {
    let long = "x".repeat(5000);
    let (text, truncation) = format_output(&long, "", 4000, None);
    assert!(
        !text.contains("OUTPUT TRUNCATED"),
        "notice must not be embedded"
    );
    let info = truncation.expect("truncation info for oversized output");
    assert_eq!(info.total_chars, 5000);
    assert!(info.shown_chars <= 4000);
}

/// A long output keeps its head and tail inline, names the lines it
/// left out, and is saved whole where the session can read it.
#[test]
fn a_cut_output_is_saved_whole_and_points_at_the_lines_left_out() {
    let long: String = (1..=2000)
        .map(|n| format!("synthetic output line {n}\n"))
        .collect();
    let session = format!("synthetic-spill-{}", std::process::id());
    let (text, truncation) = format_output(&long, "", 16_000, Some(&session));
    assert!(text.len() <= 16_000, "{}", text.len());
    assert!(
        text.starts_with("synthetic output line 1\n"),
        "{}",
        &text[..80]
    );
    assert!(
        text.contains("synthetic output line 2000"),
        "the tail is kept"
    );
    let marker = text
        .lines()
        .find(|line| line.starts_with("[... lines "))
        .expect("a marker naming the omitted lines");
    assert!(marker.contains("of 2000 omitted"), "{marker}");
    let path = marker
        .split("saved at ")
        .nth(1)
        .and_then(|rest| rest.split(':').next())
        .expect("the saved path");
    assert_eq!(std::fs::read_to_string(path).unwrap(), long, "saved whole");
    let hint = truncation.unwrap().remediation_hint.unwrap();
    assert!(
        hint.contains(path) && hint.contains("instead of re-running"),
        "{hint}"
    );
    let _ = std::fs::remove_file(path);

    // Without a session (another backend) the cut still names the lines.
    let (text, truncation) = format_output(&long, "", 16_000, None);
    assert!(text.contains("of 2000 omitted"), "{}", &text[..200]);
    assert!(truncation.unwrap().remediation_hint.is_none());
}

#[test]
fn format_output_no_truncation_for_small_output() {
    let (text, truncation) = format_output("hello", "", 4000, None);
    assert!(text.contains("hello"));
    assert!(truncation.is_none());
}

#[test]
fn empty_output_continuation_failure_names_unfinished_requirements() {
    let message = format_background_continuation_failure(&[
        "wait for the replacement run".to_string(),
        "send the publication receipt".to_string(),
    ]);
    assert!(message.contains("couldn't complete the remaining request"));
    assert!(message.contains("wait for the replacement run"));
    assert!(message.contains("send the publication receipt"));
    assert!(!message.contains("pid="));
}

#[test]
fn test_format_stdout_only() {
    let (result, truncation) = format_output("hello", "", 1000, None);
    assert_eq!(result, "hello");
    assert!(truncation.is_none());
}

#[test]
fn test_format_stderr_appended() {
    let (result, truncation) = format_output("out", "err", 1000, None);
    assert_eq!(result, "out\n--- stderr ---\nerr");
    assert!(truncation.is_none());
}

#[test]
fn test_format_empty_no_output() {
    let (result, truncation) = format_output("", "", 1000, None);
    assert_eq!(result, "(no output)");
    assert!(truncation.is_none());
}

#[test]
fn test_format_truncation() {
    let long_output = "a".repeat(2000);
    let (result, truncation) = format_output(&long_output, "", 400, None);
    // The returned text is the raw content only — no embedded notice.
    // The notice is rendered by the caller (foreground: from metadata;
    // background: inline at the delivery site).
    assert!(!result.contains("OUTPUT TRUNCATED"));
    assert!(result.len() <= 400, "{} bytes", result.len());
    assert!(result.contains("omitted"), "{result}");
    // The structured info must report the omitted amount so a caller
    // rendering the notice can't silently let the model fabricate the
    // rest.
    let info = truncation.expect("truncation info for oversized output");
    assert_eq!(info.shown_chars, result.chars().count());
    assert_eq!(info.total_chars, 2000);
}

#[test]
fn long_output_keeps_its_end_where_the_test_summary_is() {
    let mut out = String::from("   Compiling acme v0.1.0\n");
    for n in 0..400 {
        out.push_str(&format!("test tests::case_{n} ... ok\n"));
    }
    out.push_str("test result: FAILED. 399 passed; 1 failed\n");
    let (result, truncation) = format_output(
        &out,
        "error: test failed, to rerun pass `--lib`",
        4000,
        None,
    );
    assert!(truncation.is_some());
    assert!(result.len() <= 4000, "{} bytes", result.len());
    assert!(result.starts_with("   Compiling acme"));
    assert!(result.contains("test result: FAILED. 399 passed; 1 failed"));
    assert!(result.ends_with("error: test failed, to rerun pass `--lib`"));
    // Cut on line boundaries: no half test line after the marker.
    let after = result.split("...]\n").nth(1).unwrap();
    assert!(after.starts_with("test tests::case_"), "{after:.40}");
}

#[test]
fn test_format_truncation_multibyte_utf8() {
    // "é" is 2 bytes in UTF-8, "日" is 3 bytes, "🎉" is 4 bytes
    let output = "aé日🎉".repeat(50); // mixed multi-byte chars
    // Truncate at various positions that may land mid-char
    for max in [1, 2, 3, 4, 5, 10, 50, 100] {
        let (result, truncation) = format_output(&output, "", max, None);
        // Must not panic and must be valid UTF-8 (String guarantees this)
        assert!(!result.is_empty());
        if output.len() > max {
            assert!(truncation.is_some());
        }
    }
}
