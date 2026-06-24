// This module is infrastructure for terminal/notifier integration (upcoming task).
// Items are not yet called from outside, so suppress dead-code lints.
#![allow(dead_code)]
//! Pure deliverable attribution module.
//!
//! Determines which file(s) a background command produced, so the terminal
//! notifier can deliver the right artifact to the user. All filesystem/process
//! I/O is injected via closures (`read_script`, `stat_mtime`) for full
//! unit-testability — this module does no real I/O of its own.
use regex::Regex;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Context built at command completion describing which paths were produced.
#[derive(Debug)]
pub struct BackgroundDeliverableContext {
    pub session_id: String,
    pub command: String,
    pub command_start: SystemTime,
    pub command_end: SystemTime,
    /// Explicit absolute paths classified as produced output (auto-send eligible).
    pub produced_candidates: Vec<PathBuf>,
    /// Diagnostic-only dynamic/pattern hints (NOT auto-send eligible in v1).
    pub pattern_hints: Vec<String>,
}

/// The result of deciding which (if any) single file to auto-send.
#[derive(Debug)]
pub enum AutoSendDecision {
    One(PathBuf),
    None,
    Ambiguous(Vec<PathBuf>),
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Patterns that signal dynamic filename construction — never a concrete path.
fn is_dynamic_content(text: &str) -> bool {
    let dynamic_patterns = [
        "strftime",
        "uuid",
        "tempfile.gettempdir",
        "os.path.join",
        "$(",
        "uuid4()",
        "uuid1()",
    ];
    dynamic_patterns.iter().any(|p| text.contains(p))
}

/// Extract all absolute path tokens from a string (paths starting with `/`).
fn extract_absolute_paths(text: &str) -> Vec<String> {
    // Match tokens that look like absolute paths: start with / and contain
    // at least one non-whitespace character, no shell quoting complexity.
    let re = Regex::new(r#"(?:^|\s|['"])(/[^\s'"<>|;&]+)"#).unwrap();
    re.captures_iter(text)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

/// Returns absolute paths that appear as output-flag targets in the command line.
/// Handles: `> path`, `-o path`, `--output path`, `--output=path`.
fn extract_output_flag_paths(command: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();

    // `> /path/to/file` — shell output redirection
    let redir_re = Regex::new(r">\s*(/[^\s|;&>]+)").unwrap();
    for cap in redir_re.captures_iter(command) {
        if let Some(m) = cap.get(1) {
            results.push(PathBuf::from(m.as_str()));
        }
    }

    // `-o /path` or `--output /path` (value as next token)
    let flag_re = Regex::new(r"(?:-o|--output)\s+(/[^\s|;&>]+)").unwrap();
    for cap in flag_re.captures_iter(command) {
        if let Some(m) = cap.get(1) {
            results.push(PathBuf::from(m.as_str()));
        }
    }

    // `--output=/path`
    let flag_eq_re = Regex::new(r"--output=(/[^\s|;&>]+)").unwrap();
    for cap in flag_eq_re.captures_iter(command) {
        if let Some(m) = cap.get(1) {
            results.push(PathBuf::from(m.as_str()));
        }
    }

    results
}

/// Identify the "executed script" argument from a command — the path that is
/// being *run* (as opposed to consumed/produced). These must be excluded from
/// produced_candidates.
///
/// Heuristic: after stripping shell prefix tokens (`cd /dir &&`, env var
/// assignments) look for the interpreter + script pattern:
/// `python3 /tmp/x.py`, `bash /tmp/x.sh`, `node /tmp/x.js`, etc.
/// Also handles bare script invocations like `/tmp/x.sh`.
fn identify_executed_scripts(command: &str) -> Vec<PathBuf> {
    let mut executed = Vec::new();

    // Script extensions we recognise
    let script_exts = ["py", "sh", "js", "ts", "rb", "pl", "php", "lua", "r"];

    // Split on common shell operators and iterate clauses
    let clauses: Vec<&str> = command.split_terminator(['&', ';', '|']).collect();

    for clause in clauses {
        let tokens: Vec<&str> = clause.split_whitespace().collect();
        for (i, token) in tokens.iter().enumerate() {
            // Skip env-var assignments like FOO=bar
            if token.contains('=') && !token.starts_with('/') && !token.starts_with('-') {
                continue;
            }
            // The "program" is the first non-env-var token; skip known interpreters
            // and look at the next token as a potential script path.
            let interpreters = [
                "python", "python3", "python2", "bash", "sh", "node", "ruby", "perl", "php", "lua",
                "Rscript", "npx", "uvx",
            ];
            if interpreters.contains(token) {
                // The next absolute-path argument is the executed script
                if let Some(next) = tokens.get(i + 1) {
                    if next.starts_with('/') {
                        let ext = Path::new(next)
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("");
                        if script_exts.contains(&ext) {
                            executed.push(PathBuf::from(next));
                        }
                    }
                }
                continue;
            }
            // Bare script invocation
            if token.starts_with('/') {
                let ext = Path::new(token)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                if script_exts.contains(&ext) {
                    executed.push(PathBuf::from(*token));
                }
            }
        }
    }

    executed
}

/// Parse a script's text for write-mode `open(path, "w"|"a")` literals.
/// Returns produced_candidates and pattern_hints separately.
fn parse_script_for_writes(script_text: &str) -> (Vec<PathBuf>, Vec<String>) {
    let mut produced = Vec::new();
    let mut hints = Vec::new();

    // Match open("literal_path", "w"|"a") — standard Python open() calls.
    // We only handle string literals in the first position (not variable refs).
    let open_literal_re = Regex::new(r#"open\(\s*["'](/[^"']+)["']\s*,\s*["'][wa]["']\)"#).unwrap();
    for cap in open_literal_re.captures_iter(script_text) {
        if let Some(m) = cap.get(1) {
            let path_str = m.as_str();
            if is_dynamic_content(path_str) {
                hints.push(format!("dynamic open: {path_str}"));
            } else {
                produced.push(PathBuf::from(path_str));
            }
        }
    }

    // Detect dynamic patterns in the whole script body (even in f-strings).
    // e.g. open(f"/tmp/{uuid.uuid4()}.txt","w")
    let dynamic_open_re = Regex::new(r#"open\(f?["'][^"']*["']\s*,\s*["'][wa]["']\)"#).unwrap();
    for cap in dynamic_open_re.captures_iter(script_text) {
        let full = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if is_dynamic_content(full) && !hints.iter().any(|h: &String| h.contains(full)) {
            hints.push(format!("dynamic open pattern: {full}"));
        }
    }

    // Also detect dynamic constructs appearing anywhere in the script
    if is_dynamic_content(script_text) && produced.is_empty() && hints.is_empty() {
        // generic hint for the whole script
        hints.push("dynamic filename construction detected".to_string());
    }

    (produced, hints)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the deliverable context at command completion.
pub fn attribute_deliverable(
    session_id: &str,
    command: &str,
    command_start: SystemTime,
    command_end: SystemTime,
    checklist_text: &[String],
    read_script: &dyn Fn(&Path) -> Option<String>,
    stat_mtime: &dyn Fn(&Path) -> Option<SystemTime>,
) -> BackgroundDeliverableContext {
    let mut produced_set: HashSet<PathBuf> = HashSet::new();
    let mut pattern_hints: Vec<String> = Vec::new();

    // mtime window: [command_start, command_end + 2s]
    let window_end = command_end + Duration::from_secs(2);

    // --- Identify executed/consumed scripts (never eligible) ---
    let executed_scripts: HashSet<PathBuf> =
        identify_executed_scripts(command).into_iter().collect();

    // --- Collect output-flag targets from the command line ---
    let output_flag_paths = extract_output_flag_paths(command);
    for p in &output_flag_paths {
        if !executed_scripts.contains(p) {
            produced_set.insert(p.clone());
        }
    }

    // --- Collect all absolute paths mentioned in the command ---
    let all_cmd_paths: Vec<PathBuf> = extract_absolute_paths(command)
        .into_iter()
        .map(PathBuf::from)
        .collect();

    // --- Check mtime window (primary signal) ---
    // Apply to all paths discovered from command + output flags
    let mut all_candidate_paths: HashSet<PathBuf> = HashSet::new();
    for p in &all_cmd_paths {
        all_candidate_paths.insert(p.clone());
    }
    for p in &output_flag_paths {
        all_candidate_paths.insert(p.clone());
    }

    for p in &all_candidate_paths {
        if executed_scripts.contains(p) {
            continue;
        }
        if let Some(mtime) = stat_mtime(p) {
            if mtime >= command_start && mtime <= window_end {
                produced_set.insert(p.clone());
            }
        }
    }

    // --- Parse referenced script(s) ---
    for p in &executed_scripts {
        if let Some(script_text) = read_script(p) {
            let (script_produced, mut script_hints) = parse_script_for_writes(&script_text);
            pattern_hints.append(&mut script_hints);

            for sp in script_produced {
                if !executed_scripts.contains(&sp) {
                    // Check mtime for script-discovered paths too
                    if let Some(mtime) = stat_mtime(&sp) {
                        if mtime >= command_start && mtime <= window_end {
                            produced_set.insert(sp.clone());
                        } else {
                            // literal write-mode open is sufficient evidence even without mtime
                            produced_set.insert(sp);
                        }
                    } else {
                        // no mtime info — trust the literal open() call
                        produced_set.insert(sp);
                    }
                }
            }

            // Also check script-mentioned absolute paths for mtime signal
            let script_paths = extract_absolute_paths(&script_text);
            for sp_str in script_paths {
                let sp = PathBuf::from(&sp_str);
                if executed_scripts.contains(&sp) {
                    continue;
                }
                if let Some(mtime) = stat_mtime(&sp) {
                    if mtime >= command_start && mtime <= window_end {
                        produced_set.insert(sp);
                    }
                }
            }
        }
    }

    // --- Process checklist_text hints ---
    for line in checklist_text {
        let paths = extract_absolute_paths(line);
        for ps in paths {
            let p = PathBuf::from(&ps);
            if executed_scripts.contains(&p) {
                continue;
            }
            if let Some(mtime) = stat_mtime(&p) {
                if mtime >= command_start && mtime <= window_end {
                    produced_set.insert(p);
                }
            }
        }
    }

    // Build final list (deterministic order via sort)
    let mut produced_candidates: Vec<PathBuf> = produced_set.into_iter().collect();
    produced_candidates.sort();

    BackgroundDeliverableContext {
        session_id: session_id.to_string(),
        command: command.to_string(),
        command_start,
        command_end,
        produced_candidates,
        pattern_hints,
    }
}

/// Decide whether to auto-send and which file.
pub fn auto_send_decision(ctx: &BackgroundDeliverableContext) -> AutoSendDecision {
    // Deduplicate (already sorted; use a seen set just in case)
    let mut seen: HashSet<&PathBuf> = HashSet::new();
    let deduped: Vec<&PathBuf> = ctx
        .produced_candidates
        .iter()
        .filter(|p| seen.insert(p))
        .collect();

    match deduped.len() {
        0 => AutoSendDecision::None,
        1 => AutoSendDecision::One(deduped[0].clone()),
        _ => AutoSendDecision::Ambiguous(deduped.into_iter().cloned().collect()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn t0() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000)
    }

    #[test]
    fn excludes_executed_script_includes_mtime_changed_output() {
        let start = t0();
        let end = start + Duration::from_secs(40);
        let script = "/tmp/probe.py";
        let out = "/tmp/probe_results.txt";
        let read = |p: &std::path::Path| -> Option<String> {
            if p == std::path::Path::new(script) {
                Some("output_path = \"/tmp/probe_results.txt\"\nopen(output_path, \"a\")\n".into())
            } else {
                None
            }
        };
        // results file mtime is within the run window; script mtime is before it.
        let stat = |p: &std::path::Path| -> Option<SystemTime> {
            match p.to_str().unwrap() {
                "/tmp/probe_results.txt" => Some(start + Duration::from_secs(20)),
                "/tmp/probe.py" => Some(start - Duration::from_secs(10)),
                _ => None,
            }
        };
        let ctx = attribute_deliverable(
            "s1",
            "cd /tmp && python3 /tmp/probe.py",
            start,
            end,
            &[],
            &read,
            &stat,
        );
        assert!(
            matches!(auto_send_decision(&ctx), AutoSendDecision::One(p) if p == std::path::Path::new(out)),
            "expected One({out}) but got {:?}",
            ctx.produced_candidates
        );
        assert!(
            !ctx.produced_candidates
                .iter()
                .any(|p| p == std::path::Path::new(script)),
            "executed script must be excluded as consumed"
        );
    }

    #[test]
    fn dynamic_filename_is_pattern_hint_not_candidate() {
        let start = t0();
        let end = start + Duration::from_secs(40);
        let read = |_: &std::path::Path| {
            Some("import uuid\nopen(f\"/tmp/{uuid.uuid4()}.txt\",\"w\")".to_string())
        };
        let stat = |_: &std::path::Path| -> Option<SystemTime> { None };
        let ctx = attribute_deliverable("s1", "python3 /tmp/dyn.py", start, end, &[], &read, &stat);
        assert!(ctx.produced_candidates.is_empty());
        assert!(!ctx.pattern_hints.is_empty());
        assert!(matches!(auto_send_decision(&ctx), AutoSendDecision::None));
    }

    #[test]
    fn two_outputs_are_ambiguous() {
        let start = t0();
        let end = start + Duration::from_secs(40);
        let read = |_: &std::path::Path| -> Option<String> { None };
        let stat = |p: &std::path::Path| -> Option<SystemTime> {
            match p.to_str().unwrap() {
                "/tmp/a.txt" | "/tmp/b.txt" => Some(start + Duration::from_secs(5)),
                _ => None,
            }
        };
        let ctx = attribute_deliverable(
            "s1",
            "mytool -o /tmp/a.txt --output /tmp/b.txt",
            start,
            end,
            &[],
            &read,
            &stat,
        );
        assert!(
            matches!(auto_send_decision(&ctx), AutoSendDecision::Ambiguous(v) if v.len() == 2),
            "expected Ambiguous(2) but got {:?}",
            ctx.produced_candidates
        );
    }
}
