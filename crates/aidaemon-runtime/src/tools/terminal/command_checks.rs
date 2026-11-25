//! Structural checks on a command line before it runs: segments,
//! recursive greps, unbounded disk scans, UI scripting.

use super::*;

/// Split a chained command into individual segments by pipe, semicolon, &&, ||.
/// Used by session-approval to extract per-segment binary names.
/// Quote-aware: operators inside single/double quotes are not treated as separators.
pub(super) fn split_command_segments(cmd: &str) -> Vec<&str> {
    aidaemon_policies::scan_shell(cmd).segments
}

/// Extract the binary/command name from a single command segment.
/// Handles variable assignments like `VAR=val cmd args...` by skipping
/// assignment tokens and returning the first non-assignment word.
pub(super) fn extract_segment_binary(segment: &str) -> &str {
    for word in segment.split_whitespace() {
        // Skip shell variable assignments (e.g., EPOCH=$(date ...))
        if word.contains('=') {
            continue;
        }
        return word;
    }
    ""
}

/// Return structurally parsed `(program, argv)` invocations from a shell
/// command. Exact executable basenames are used for workload classification;
/// this is not natural-language keyword matching.
pub(super) fn parsed_command_invocations(command: &str) -> Vec<(String, Vec<String>)> {
    aidaemon_tools::command_risk::structural_shell_invocations(command)
        .into_iter()
        .filter_map(|invocation| {
            let program = std::path::Path::new(&invocation.program)
                .file_name()
                .and_then(|name| name.to_str())?
                .to_ascii_lowercase();
            Some((program, invocation.arguments))
        })
        .collect()
}

pub(super) fn filesystem_invocation_is_broad_or_multi(program: &str, args: &[String]) -> bool {
    let path_operands: Vec<&str> = match program {
        // `find ROOT... EXPRESSION`: roots precede the first expression/option.
        "find" => args
            .iter()
            .take_while(|arg| !arg.starts_with('-') && arg.as_str() != "!")
            .map(String::as_str)
            .collect(),
        // `du [OPTIONS] PATH...`: every non-option token is a target. This is a
        // conservative parse; an option value counted as a target only makes the
        // supervisor more patient, never less safe.
        _ => args
            .iter()
            .filter(|arg| !arg.starts_with('-'))
            .map(String::as_str)
            .collect(),
    };

    path_operands.len() > 1
        || path_operands.iter().any(|path| {
            is_broad_scan_root(path)
                || (path.contains('{') && path.contains(',') && path.contains('}'))
        })
}

pub(super) fn progress_contract_for_command(command: &str) -> BackgroundProgressContract {
    let invocations = parsed_command_invocations(command);

    for (program, args) in &invocations {
        if matches!(program.as_str(), "du" | "find" | "tree") {
            let broad_or_multi_target =
                filesystem_invocation_is_broad_or_multi(program, args.as_slice());
            return BackgroundProgressContract {
                workload: BackgroundWorkload::FilesystemTraversal {
                    broad_or_multi_target,
                },
                stall_multiplier: if broad_or_multi_target { 5 } else { 3 },
                idle_confirmations_required: 2,
                hard_max_runtime: false,
            };
        }
    }

    for (program, args) in &invocations {
        let subcommand = args.first().map(String::as_str).unwrap_or("");
        let is_build_or_test = matches!(
            program.as_str(),
            "cargo"
                | "rustc"
                | "make"
                | "ninja"
                | "cmake"
                | "xcodebuild"
                | "swift"
                | "swiftc"
                | "gradle"
                | "gradlew"
                | "mvn"
                | "mvnw"
                | "pytest"
        ) || (program == "go"
            && matches!(subcommand, "build" | "test" | "install"))
            || (matches!(program.as_str(), "npm" | "pnpm" | "yarn" | "bun")
                && matches!(subcommand, "build" | "test" | "install" | "run"));
        if is_build_or_test {
            return BackgroundProgressContract {
                workload: BackgroundWorkload::BuildOrTest,
                stall_multiplier: 2,
                idle_confirmations_required: 2,
                hard_max_runtime: false,
            };
        }
    }

    for (program, args) in &invocations {
        let subcommand = args.first().map(String::as_str).unwrap_or("");
        let is_network_transfer =
            matches!(program.as_str(), "curl" | "wget" | "rsync" | "scp" | "sftp")
                || (program == "git"
                    && matches!(subcommand, "clone" | "fetch" | "pull" | "submodule"));
        if is_network_transfer {
            return BackgroundProgressContract {
                workload: BackgroundWorkload::NetworkTransfer,
                stall_multiplier: 2,
                idle_confirmations_required: 2,
                hard_max_runtime: false,
            };
        }
    }

    BackgroundProgressContract::default()
}

pub(super) fn is_grep_command(token: &str) -> bool {
    std::path::Path::new(token)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "grep")
}

pub(super) fn grep_has_recursive_flag(token: &str) -> bool {
    if matches!(token, "--recursive" | "--dereference-recursive") {
        return true;
    }
    if token.starts_with("--") {
        return false;
    }
    token
        .strip_prefix('-')
        .is_some_and(|flags| flags.chars().any(|c| c == 'r' || c == 'R'))
}

pub(super) fn has_recursive_grep_scope_controls(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains("--exclude-dir")
        || lower.contains("--exclude=")
        || lower.contains("--exclude ")
        || lower.contains("--include")
        || lower.contains("-d skip")
        || lower.contains("-dskip")
}

/// Detect AppleScript System Events UI scripting (click/keystroke/key code/
/// set value) driven through the terminal. This is computer-use-by-another-name:
/// it bypasses the computer_use policy layer entirely — per-app approvals,
/// prohibited bundles, point-of-action confirmation, screen-lock detection —
/// and it fails against a locked screen anyway, which is exactly when models
/// reach for it (live 2026-07-12: four minutes of `osascript … System Events`
/// flailing after the GUI loop correctly reported the screen locked).
/// Read-only System Events queries (get name of every process, …) are allowed.
pub(super) fn is_system_events_ui_scripting(command: &str) -> bool {
    // A command line is a machine format: the call is a segment whose
    // program token is `osascript` (directly, or inside a nested shell's
    // `-c` argument), and its AppleScript arguments name a System Events UI
    // verb. Text that merely names osascript (an `echo`) is not a call.
    osascript_script(command, 0).is_some_and(|script| {
        script.contains("system events")
            && [
                "click",
                "keystroke",
                "key code",
                "set value",
                "perform action",
            ]
            .iter()
            .any(|verb| script.contains(verb))
    })
}

/// The lowercased arguments of an `osascript` call in `command`, looking
/// one level into nested shell arguments.
pub(super) fn osascript_script(command: &str, depth: usize) -> Option<String> {
    for segment in command.split([';', '|', '&', '\n']) {
        let tokens = shell_words::split(segment).unwrap_or_else(|_| {
            segment
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        });
        // Wrappers and `VAR=value` assignments before the program.
        let Some(at) = tokens.iter().position(|token| {
            !token.contains('=')
                && !matches!(
                    token.as_str(),
                    "sudo" | "env" | "exec" | "command" | "nohup" | "time"
                )
        }) else {
            continue;
        };
        let program = std::path::Path::new(&tokens[at])
            .file_name()
            .map(|name| name.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if program == "osascript" {
            return Some(tokens[at + 1..].join(" ").to_ascii_lowercase());
        }
        let shell = matches!(
            program.as_str(),
            "sh" | "bash" | "zsh" | "dash" | "ksh" | "fish"
        );
        if shell && depth == 0 {
            for nested in tokens.iter().skip(at + 1).filter(|t| !t.starts_with('-')) {
                if let Some(script) = osascript_script(nested, depth + 1) {
                    return Some(script);
                }
            }
        }
    }
    None
}

pub(super) fn detect_unscoped_recursive_grep_segment(segment: &str) -> Option<(String, String)> {
    let tokens = shell_words::split(segment).ok()?;
    let first = tokens.first()?;
    if !is_grep_command(first) {
        return None;
    }

    let recursive = tokens
        .iter()
        .skip(1)
        .any(|tok| grep_has_recursive_flag(tok));
    if !recursive || has_recursive_grep_scope_controls(segment) {
        return None;
    }

    // grep syntax: grep [OPTIONS] PATTERN [FILE...]
    // We use a lightweight parse here: non-option tokens are treated as
    // positional args; first positional = pattern, remaining = target paths.
    let positionals: Vec<String> = tokens
        .iter()
        .skip(1)
        .filter(|tok| !tok.starts_with('-'))
        .cloned()
        .collect();
    let pattern = positionals.first()?.clone();
    let paths = if positionals.len() >= 2 {
        positionals[1..].to_vec()
    } else {
        vec![".".to_string()]
    };
    let broad_scope = paths
        .iter()
        .any(|p| matches!(p.as_str(), "." | "./" | "/" | "~" | "~/"));
    if !broad_scope {
        return None;
    }

    Some((pattern, paths.join(" ")))
}

pub(super) fn detect_unscoped_recursive_grep(command: &str) -> Option<(String, String)> {
    if let Some(hit) = detect_unscoped_recursive_grep_segment(command) {
        return Some(hit);
    }

    // Also scan chained shell segments (e.g. "cd repo && grep -rc ... .").
    // This is intentionally simple and best-effort: it catches common cases
    // without trying to fully parse shell grammar.
    static SHELL_CHAIN_SPLIT_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?:&&|\|\||;|\|)").expect("valid chain regex"));
    for segment in SHELL_CHAIN_SPLIT_RE.split(command) {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(hit) = detect_unscoped_recursive_grep_segment(trimmed) {
            return Some(hit);
        }
    }

    None
}

pub(super) fn recursive_grep_block_message(pattern: &str, path: &str) -> String {
    let ignore_globs = aidaemon_tools::fs_utils::DEFAULT_IGNORE_DIRS.join(",");
    format!(
        "Blocked: broad recursive `grep` without include/exclude filters is likely to stall on large trees.\n\
Detected pattern: \"{}\"\n\
Detected path: {}\n\n\
Use one of these instead:\n\
- `search_files` (preferred) with explicit `path`, optional `glob`, and regex `pattern`\n\
- Terminal `rg` with exclusions:\n\
  `rg -n --glob '!{{{}}}' \"<pattern>\" <path>`\n\
- If you must use grep, add `--exclude-dir` and/or `--include` so the scan is bounded.",
        pattern, path, ignore_globs
    )
}

/// True only for the filesystem root or the home directory itself — the scan
/// roots that make `du`/`find` pathologically slow. Subdirectories return false.
pub(super) fn is_broad_scan_root(path: &str) -> bool {
    let p = path.trim();
    if p == "/" {
        return true;
    }
    let p = p.trim_end_matches('/');
    if p == "~" || p == "$HOME" {
        return true;
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
        && p == home.trim_end_matches('/')
    {
        return true;
    }
    false
}

/// Detect an unbounded whole-disk/whole-home scan in a single shell segment.
/// `du` over a broad root is always flagged (it walks the full subtree to sum
/// sizes; `-d` only limits output depth). `find` over a broad root is flagged
/// unless `-maxdepth` is present (which bounds traversal). Returns (tool, root).
pub(super) fn detect_unbounded_disk_scan_segment(segment: &str) -> Option<(String, String)> {
    let tokens = shell_words::split(segment).ok()?;
    let first = tokens.first()?;
    // Strip any leading path (e.g. /usr/bin/du -> du).
    let tool = std::path::Path::new(first)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(first.as_str());

    match tool {
        "du" => {
            // Any non-flag arg that is a broad root.
            let root = tokens
                .iter()
                .skip(1)
                .find(|t| !t.starts_with('-') && is_broad_scan_root(t))?;
            Some(("du".to_string(), root.clone()))
        }
        "find" => {
            // find's search roots come before the expression; check leading
            // non-flag args, but only flag if there is no -maxdepth limiter.
            let has_maxdepth = tokens.iter().any(|t| t == "-maxdepth");
            if has_maxdepth {
                return None;
            }
            // `-delete` is caught by the irreversible-delete hard-blocker;
            // let it pass through so the right message fires.
            if tokens.iter().any(|t| t == "-delete") {
                return None;
            }
            let root = tokens
                .iter()
                .skip(1)
                .take_while(|t| !t.starts_with('-'))
                .find(|t| is_broad_scan_root(t))?;
            Some(("find".to_string(), root.clone()))
        }
        _ => None,
    }
}

/// Scan the whole command (including chained `&&`/`|`/`;` segments) for an
/// unbounded whole-disk/whole-home scan.
pub(super) fn detect_unbounded_disk_scan(command: &str) -> Option<(String, String)> {
    for (segment, _) in aidaemon_tools::command_risk::split_by_operators(command) {
        if let Some(hit) = detect_unbounded_disk_scan_segment(&segment) {
            return Some(hit);
        }
    }
    None
}

/// Guidance returned to the model when an unbounded scan is blocked pre-spawn.
pub(super) fn unbounded_scan_block_message(tool: &str, root: &str) -> String {
    format!(
        "Blocked: `{tool}` rooted at `{root}` scans the entire {} and is pathologically slow \
         — it walks every file and typically never finishes (it gets auto-stopped after a few \
         minutes without ever answering). Use a narrower, bounded command instead:\n\
         • For \"biggest files\": pick a specific folder — `find ~/Downloads -type f -size +500M` \
         or `find ~/projects -type f -size +500M`.\n\
         • Add a depth limit so traversal is bounded: `find <DIR> -maxdepth 3 …`.\n\
         • Scope `du` to a single folder: `du -sh ~/<folder>/*` (not the whole disk or home).\n\
         If the user truly needs a whole-disk scan, ask them to confirm or run it themselves.\n\
         Pick a scoped command and try again.",
        if root == "/" {
            "disk"
        } else {
            "home directory"
        }
    )
}

pub(super) fn normalize_command_for_dedupe(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}
