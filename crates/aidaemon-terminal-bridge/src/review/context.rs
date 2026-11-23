//! Collecting the diff a review reads.

use crate::*;

pub(crate) fn review_diff_base_args(diff_base: ReviewDiffBase) -> Vec<String> {
    let mut args = vec![
        "diff".to_string(),
        "--no-color".to_string(),
        "--no-ext-diff".to_string(),
    ];
    if matches!(diff_base, ReviewDiffBase::Staged) {
        args.push("--staged".to_string());
    }
    args
}

pub(crate) fn is_probably_generated_or_non_source(path: &str) -> Option<&'static str> {
    let lower = path.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return Some("empty-path");
    }
    let roots = [
        "node_modules/",
        ".next/",
        "dist/",
        "build/",
        "coverage/",
        "vendor/",
        "target/",
        ".turbo/",
    ];
    if roots.iter().any(|prefix| lower.starts_with(prefix)) {
        return Some("generated-root");
    }
    let generated_suffixes = [
        ".min.js", ".min.css", ".map", ".lock", ".lockb", ".woff", ".woff2", ".ttf", ".eot",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".mp3", ".mp4",
        ".mov", ".avi", ".sqlite", ".db",
    ];
    if generated_suffixes
        .iter()
        .any(|suffix| lower.ends_with(suffix))
    {
        return Some("generated-suffix");
    }
    let generated_names = [
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lockb",
        "cargo.lock",
        "tsconfig.tsbuildinfo",
    ];
    if generated_names.iter().any(|name| lower.ends_with(name)) {
        return Some("lock-or-build-artifact");
    }
    None
}

pub(crate) fn parse_review_numstat(raw: &str) -> Vec<ReviewFileChange> {
    let mut out = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(3, '\t');
        let added_raw = parts.next().unwrap_or_default();
        let deleted_raw = parts.next().unwrap_or_default();
        let path_raw = parts.next().unwrap_or_default().trim();
        if path_raw.is_empty() {
            continue;
        }
        let binary = added_raw == "-" || deleted_raw == "-";
        let added = added_raw.parse::<u64>().ok();
        let deleted = deleted_raw.parse::<u64>().ok();
        out.push(ReviewFileChange {
            path: path_raw.to_string(),
            added,
            deleted,
            binary,
        });
    }
    out
}

pub(crate) fn select_review_files(
    changes: &[ReviewFileChange],
) -> (Vec<ReviewFileChange>, ReviewContextStats) {
    let mut included = Vec::new();
    let mut stats = ReviewContextStats {
        total_changed_files: changes.len(),
        ..ReviewContextStats::default()
    };
    for change in changes {
        if change.binary {
            stats.skipped_binary_files = stats.skipped_binary_files.saturating_add(1);
            continue;
        }
        if is_probably_generated_or_non_source(&change.path).is_some() {
            stats.skipped_generated_files = stats.skipped_generated_files.saturating_add(1);
            continue;
        }
        let changed_lines = change
            .added
            .unwrap_or(0)
            .saturating_add(change.deleted.unwrap_or(0));
        if changed_lines > REVIEW_MAX_FILE_CHANGED_LINES {
            stats.skipped_large_files = stats.skipped_large_files.saturating_add(1);
            continue;
        }
        if included.len() >= REVIEW_MAX_INCLUDED_DIFF_FILES {
            stats.skipped_overflow_files = stats.skipped_overflow_files.saturating_add(1);
            continue;
        }
        included.push(change.clone());
    }
    stats.included_files = included.len();
    (included, stats)
}

pub(crate) async fn collect_review_context(
    cwd: &Path,
    scope: ReviewScope,
    diff_base: ReviewDiffBase,
    replay_tail: &str,
) -> (String, ReviewContextStats) {
    let mut sections = Vec::new();
    let mut context_stats = ReviewContextStats::default();
    sections.push(format!("Working directory: {}", cwd.display()));
    sections.push(format!("Generated at: {}", chrono::Utc::now().to_rfc3339()));

    if matches!(scope, ReviewScope::Diff | ReviewScope::Both) {
        let mut diff_section = String::new();
        let git_probe = run_capture_command(
            "git",
            &["rev-parse".to_string(), "--is-inside-work-tree".to_string()],
            cwd,
            Duration::from_secs(8),
            true,
        )
        .await;
        let is_repo = git_probe.map(|out| out.status.success()).unwrap_or(false);

        if !is_repo {
            diff_section.push_str("Git repository not detected in working directory.\n");
        } else {
            let status_out = run_capture_command(
                "git",
                &[
                    "status".to_string(),
                    "--short".to_string(),
                    "--branch".to_string(),
                ],
                cwd,
                Duration::from_secs(12),
                true,
            )
            .await
            .ok()
            .map(|out| render_process_output(&out, REVIEW_MAX_SECTION_CHARS / 3))
            .unwrap_or_else(|| "Unable to read git status.".to_string());

            diff_section.push_str("Git status:\n");
            diff_section.push_str(&truncate_with_note(
                &status_out,
                REVIEW_MAX_SECTION_CHARS / 3,
            ));

            let mut numstat_args = review_diff_base_args(diff_base);
            numstat_args.push("--numstat".to_string());
            let numstat_out =
                run_capture_command("git", &numstat_args, cwd, Duration::from_secs(15), true)
                    .await
                    .ok()
                    .map(|out| join_process_output(&out))
                    .unwrap_or_default();
            let changes = parse_review_numstat(&numstat_out);
            let (selected_files, mut stats) = select_review_files(&changes);

            diff_section.push_str("\n\nDiff selection summary:\n");
            diff_section.push_str(&format!(
                "Changed files: {} | Included for focused diff: {} | Skipped (generated: {}, binary: {}, large: {}, overflow: {})\n",
                stats.total_changed_files,
                stats.included_files,
                stats.skipped_generated_files,
                stats.skipped_binary_files,
                stats.skipped_large_files,
                stats.skipped_overflow_files
            ));

            if !selected_files.is_empty() {
                diff_section.push_str("\nIncluded files:\n");
                for file in selected_files.iter().take(REVIEW_MAX_INCLUDED_DIFF_FILES) {
                    let added = file
                        .added
                        .map(|v| format!("+{}", v))
                        .unwrap_or_else(|| "+?".to_string());
                    let deleted = file
                        .deleted
                        .map(|v| format!("-{}", v))
                        .unwrap_or_else(|| "-?".to_string());
                    diff_section.push_str(&format!("- {} ({}, {})\n", file.path, added, deleted));
                }
            }

            let mut focused_chunks = Vec::new();
            let mut used_chars = 0usize;
            for file in &selected_files {
                let mut file_args = review_diff_base_args(diff_base);
                file_args.push("--".to_string());
                file_args.push(file.path.clone());
                let raw_file_diff =
                    run_capture_command("git", &file_args, cwd, Duration::from_secs(10), true)
                        .await
                        .ok()
                        .map(|out| join_process_output(&out))
                        .unwrap_or_default();
                if raw_file_diff.trim().is_empty() {
                    continue;
                }
                let section = format!(
                    "### FILE: {}\n{}\n",
                    file.path,
                    truncate_with_note(&raw_file_diff, REVIEW_MAX_FILE_DIFF_CHARS)
                );
                let section_chars = section.chars().count();
                if used_chars.saturating_add(section_chars) > REVIEW_MAX_SECTION_CHARS {
                    stats.skipped_overflow_files = stats.skipped_overflow_files.saturating_add(1);
                    continue;
                }
                used_chars = used_chars.saturating_add(section_chars);
                focused_chunks.push(section);
            }

            if focused_chunks.is_empty() {
                let diff_out = run_capture_command(
                    "git",
                    &review_diff_base_args(diff_base),
                    cwd,
                    Duration::from_secs(20),
                    true,
                )
                .await
                .ok()
                .map(|out| render_process_output(&out, REVIEW_MAX_SECTION_CHARS))
                .unwrap_or_else(|| "Unable to read git diff.".to_string());
                diff_section.push_str("\n\nGit diff fallback:\n");
                if diff_out.trim().is_empty() {
                    diff_section.push_str("[No diff output]\n");
                } else {
                    diff_section.push_str(&truncate_with_note(&diff_out, REVIEW_MAX_SECTION_CHARS));
                    diff_section.push('\n');
                }
            } else {
                diff_section.push_str("\n\nFocused git diff:\n");
                for chunk in focused_chunks {
                    diff_section.push_str(&chunk);
                    diff_section.push('\n');
                }
            }

            context_stats = stats;
        }
        sections.push(format!(
            "=== DIFF CONTEXT ({}) ===\n{}",
            diff_base.as_str(),
            truncate_with_note(&diff_section, REVIEW_MAX_SECTION_CHARS)
        ));
    }

    if matches!(scope, ReviewScope::Plan | ReviewScope::Both) {
        let plan = if replay_tail.trim().is_empty() {
            "No recent terminal output available for plan review.".to_string()
        } else {
            truncate_with_note(replay_tail, REVIEW_MAX_SECTION_CHARS)
        };
        sections.push(format!("=== PLAN CONTEXT ===\n{}", plan));
    }

    (
        truncate_with_note(&sections.join("\n\n"), REVIEW_MAX_CONTEXT_CHARS),
        context_stats,
    )
}
