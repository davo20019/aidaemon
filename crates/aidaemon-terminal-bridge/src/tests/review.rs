//! Review profiles, output parsing, verdicts and diff selection.

use super::*;

#[test]
fn review_launch_args_never_contain_bypass_flags() {
    use aidaemon_tools::permission_bypass::PERMISSION_BYPASS_FLAGS;
    let has_bypass = |args: &[String]| {
        args.iter().any(|a| {
            PERMISSION_BYPASS_FLAGS.contains(&a.as_str())
                || a == "-y"
                || a == "--sandbox=false"
                || a == "--approval-mode=yolo"
        })
    };
    for (name, profile) in default_review_profiles() {
        let args = review_launch_args(&name, &profile.args);
        assert!(!has_bypass(&args), "{name}: {args:?}");
    }
    // User-configured review profiles go through the same strip list.
    let configured: Vec<String> = ["--yolo", "-y", "--sandbox=false", "--keep-synthetic"]
        .iter()
        .map(|a| a.to_string())
        .collect();
    let args = review_launch_args("gemini", &configured);
    assert!(!has_bypass(&args), "{args:?}");
    assert!(args.contains(&"--keep-synthetic".to_string()));
    let claude = review_launch_args(
        "claude",
        &["-p".into(), "--dangerously-skip-permissions".into()],
    );
    assert!(!has_bypass(&claude), "{claude:?}");
}

#[test]
fn default_review_profiles_use_read_only_modes() {
    let profiles = default_review_profiles();
    let codex = review_launch_args("codex", &profiles["codex"].args);
    assert!(codex.windows(2).any(|w| w == ["--sandbox", "read-only"]));
    let claude = review_launch_args("claude", &profiles["claude"].args);
    assert!(
        claude
            .windows(2)
            .any(|w| w == ["--permission-mode", "plan"])
    );
}

#[test]
fn test_normalize_review_profile_args_for_claude_forces_json() {
    let args = vec![
        "-p".to_string(),
        "--verbose".to_string(),
        "--output-format".to_string(),
        "stream-json".to_string(),
    ];
    let normalized = normalize_review_profile_args("claude", &args);
    assert!(normalized.contains(&"-p".to_string()));
    assert!(normalized.contains(&"--output-format".to_string()));
    assert!(normalized.contains(&"json".to_string()));
    assert!(!normalized.contains(&"--verbose".to_string()));
    assert!(!normalized.contains(&"stream-json".to_string()));
}

#[test]
fn test_parse_review_output_payload_requires_structured_output() {
    let raw = "review done.\nfound some issues but no json returned";
    let result = parse_review_output_payload(
        "claude",
        "rvw_test_1",
        ReviewScope::Diff,
        ReviewDiffBase::WorkingTree,
        raw,
    );
    assert!(result.is_err());
    assert!(
        result
            .err()
            .unwrap_or_default()
            .contains("no structured JSON review payload")
    );
}

#[test]
fn test_parse_review_output_payload_parses_structured_candidate() {
    let raw = r#"{"type":"result","result":"{\"verdict\":\"needs_changes\",\"blocking\":[\"Fix null check\"],\"risks\":[],\"suggestions\":[\"Add unit tests\"],\"summary\":\"Potential null dereference\"}"}"#;
    let payload = parse_review_output_payload(
        "claude",
        "rvw_test_2",
        ReviewScope::Diff,
        ReviewDiffBase::WorkingTree,
        raw,
    )
    .expect("structured review payload should parse");
    assert_eq!(
        payload.get("verdict").and_then(Value::as_str),
        Some("needs_changes")
    );
    assert_eq!(
        payload
            .get("blocking")
            .and_then(Value::as_array)
            .map(|v| v.len()),
        Some(1)
    );
}

#[test]
fn test_parse_review_output_payload_parses_findings_and_drives_verdict() {
    let raw = r#"{"verdict":"","summary":"","findings":[{"severity":"high","file":"src/lib.rs","line":"42","issue":"Possible panic on unwrap()","fix":"Handle None before unwrap"}],"blocking":[],"risks":[],"suggestions":[]}"#;
    let payload = parse_review_output_payload(
        "codex",
        "rvw_test_3",
        ReviewScope::Diff,
        ReviewDiffBase::WorkingTree,
        raw,
    )
    .expect("structured finding payload should parse");

    assert_eq!(
        payload.get("verdict").and_then(Value::as_str),
        Some("needs_changes")
    );
    assert_eq!(
        payload
            .get("findings")
            .and_then(Value::as_array)
            .map(|v| v.len()),
        Some(1)
    );
    assert_eq!(
        payload.pointer("/findings/0/file").and_then(Value::as_str),
        Some("src/lib.rs")
    );
    assert_eq!(
        payload.pointer("/findings/0/line").and_then(Value::as_u64),
        Some(42)
    );
}

#[test]
fn a_review_verdict_is_read_by_exact_value_and_never_defaults_to_approve() {
    let verdict_of = |raw: &str| {
        parse_review_output_payload(
            "codex",
            "rvw_verdict",
            ReviewScope::Diff,
            ReviewDiffBase::WorkingTree,
            raw,
        )
        .expect("structured payload")
        .get("verdict")
        .and_then(Value::as_str)
        .map(str::to_string)
    };
    let empty = r#""blocking":[],"risks":[],"suggestions":[],"findings":[]"#;
    for (verdict, want) in [
        ("approve", "approve"),
        ("needs_changes", "needs_changes"),
        // Prose in the field is not a verdict, whatever words it has.
        ("do not approve", "needs_changes"),
        ("approved, pass", "needs_changes"),
        ("lgtm", "needs_changes"),
    ] {
        let raw = format!(r#"{{"verdict":"{verdict}","summary":"Synthetic review",{empty}}}"#);
        assert_eq!(verdict_of(&raw).as_deref(), Some(want), "{verdict}");
    }
    // No verdict at all is not an approve.
    let raw = format!(r#"{{"summary":"Synthetic review",{empty}}}"#);
    assert_eq!(verdict_of(&raw).as_deref(), Some("needs_changes"));
    // An approve that lists a blocking item is not an approve.
    let raw =
        r#"{"verdict":"approve","summary":"Synthetic","blocking":["Fix the synthetic race"]}"#;
    assert_eq!(verdict_of(raw).as_deref(), Some("needs_changes"));
}

#[test]
fn test_parse_review_numstat_parses_lines() {
    let raw = "12\t3\tsrc/main.rs\n-\t-\tpublic/logo.png\n";
    let parsed = parse_review_numstat(raw);
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].path, "src/main.rs");
    assert_eq!(parsed[0].added, Some(12));
    assert_eq!(parsed[0].deleted, Some(3));
    assert!(!parsed[0].binary);
    assert_eq!(parsed[1].path, "public/logo.png");
    assert!(parsed[1].binary);
}

#[test]
fn test_select_review_files_skips_generated_binary_and_large() {
    let changes = vec![
        ReviewFileChange {
            path: "src/app.rs".to_string(),
            added: Some(10),
            deleted: Some(2),
            binary: false,
        },
        ReviewFileChange {
            path: "public/logo.png".to_string(),
            added: None,
            deleted: None,
            binary: true,
        },
        ReviewFileChange {
            path: "node_modules/pkg/index.js".to_string(),
            added: Some(4),
            deleted: Some(1),
            binary: false,
        },
        ReviewFileChange {
            path: "src/huge_diff.rs".to_string(),
            added: Some(REVIEW_MAX_FILE_CHANGED_LINES + 1),
            deleted: Some(0),
            binary: false,
        },
    ];
    let (selected, stats) = select_review_files(&changes);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].path, "src/app.rs");
    assert_eq!(stats.total_changed_files, 4);
    assert_eq!(stats.included_files, 1);
    assert_eq!(stats.skipped_binary_files, 1);
    assert_eq!(stats.skipped_generated_files, 1);
    assert_eq!(stats.skipped_large_files, 1);
}
