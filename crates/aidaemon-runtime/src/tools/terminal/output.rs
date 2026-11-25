//! Shaping a command's output: buffers, truncation, exit and signal notes,
//! low-disk notes and self-healable sandbox denials.

use super::*;

/// Drain an async reader into a capped buffer.
pub(super) async fn drain_to_buffer<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    buf: Arc<Mutex<Vec<u8>>>,
) {
    let mut tmp = [0u8; 8192];
    loop {
        match reader.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => {
                let mut b = buf.lock().await;
                let remaining = BUFFER_CAP.saturating_sub(b.len());
                if remaining > 0 {
                    let to_copy = n.min(remaining);
                    b.extend_from_slice(&tmp[..to_copy]);
                }
            }
            Err(_) => break,
        }
    }
}

/// Render an elapsed-seconds count as a friendly duration for user-facing
/// progress messages (e.g. 65 -> "1m 5s", 40 -> "40s", 3600 -> "1h 0m").
pub(super) fn humanize_elapsed(secs: u64) -> String {
    aidaemon_tools::duration_format::compact_seconds(
        secs as i64,
        aidaemon_tools::duration_format::ZeroUnitStyle::Keep,
    )
}

/// Format combined stdout/stderr output. Returns the (untruncated-notice)
/// text plus structured [`TruncationInfo`] when the output was cut down.
/// Callers decide how the notice reaches the model: foreground call sites
/// attach it to the outgoing `ToolCallOutcome.metadata.truncation` (rendered
/// once by the agent loop); background-delivery call sites, which bypass the
/// loop entirely, render it inline immediately via
/// `crate::turn::helpers::render_truncation_notice`.
pub(super) fn format_output(
    stdout: &str,
    stderr: &str,
    max_chars: usize,
    spill_session: Option<&str>,
) -> (String, Option<crate::tools::traits::TruncationInfo>) {
    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n--- stderr ---\n");
        }
        result.push_str(stderr);
    }
    if result.is_empty() {
        result.push_str("(no output)");
    }
    let mut truncation = None;
    if result.len() > max_chars {
        let total_chars = result.chars().count();
        // The whole output is kept in the session's spill folder, which the
        // session may read, and the cut says which lines it left out and
        // where they are: without it the model re-ran a narrower command to
        // see more (live 2026-11-14 to 11-20: 22% of cut outputs).
        let saved = spill_session.and_then(|session| {
            aidaemon_tools::result_spill::save_full_text_local("terminal", session, &result)
        });
        result = keep_head_and_tail_marked(&result, max_chars, |cut| match &saved {
            Some(path) => format!(
                "\n[... lines {}-{} of {} omitted ({} characters). The full output is saved at {}: \
                 read those lines with read_file (start_line/end_line) or grep the file; do not \
                 re-run the command to see them ...]\n",
                cut.first_line,
                cut.last_line,
                cut.total_lines,
                cut.omitted_chars,
                path.display()
            ),
            None => format!(
                "\n[... lines {}-{} of {} omitted ({} characters) ...]\n",
                cut.first_line, cut.last_line, cut.total_lines, cut.omitted_chars
            ),
        });
        truncation = Some(crate::tools::traits::TruncationInfo {
            shown_chars: result.chars().count(),
            total_chars,
            remediation_hint: saved.as_ref().map(|path| {
                format!(
                    "The full output is saved at {}: read the omitted lines from it with \
                     read_file (start_line/end_line) or grep, instead of re-running the command.",
                    path.display()
                )
            }),
        });
    }
    (result, truncation)
}
pub(crate) use crate::text_cut::{keep_head_and_tail, keep_head_and_tail_marked};

/// A confined command that exits non-zero while producing no output at all is
/// the hardest failure for an autonomous agent to act on: the real cause
/// (a runtime that could not read a startup file, a denied path, or an abort
/// before any diagnostic was flushed) is invisible. Return a deterministic,
/// tool-agnostic next-step ladder so the agent can surface the cause itself
/// instead of reporting an opaque blocker. Tool-neutral: no command parsing.
/// Extract absolute paths a confined process was denied that are safe to
/// auto-grant read on a single self-heal retry. This encodes the manual
/// debugging step "the sandbox denied path X that the tool needs at startup,
/// so grant X read-only and re-run" — generally, for any tool, without a
/// per-tool table. It is deliberately conservative: only existing regular
/// files in recognized read-only *configuration* locations, never a secret
/// store, never a directory, never a write.
pub(super) fn self_healable_denied_reads(
    output: &str,
    home: &str,
    already_granted: &[String],
) -> Vec<String> {
    // SECURITY: the command's own stdout/stderr is attacker-controlled (a
    // malicious build script can print a fabricated denial naming any path).
    // We therefore never grant a path merely because the output names it as
    // denied. A candidate is granted only when its *basename* is on a strict
    // allowlist of known NON-secret configuration files. Nothing that can hold
    // a credential (.npmrc, .netrc, .env, .pgpass, key material, shell history)
    // is on the list, so a forged denial can at most re-grant a harmless file
    // the tool would read anyway. Match is basename equality after
    // canonicalization; the untrusted text only nominates candidates.
    const SAFE_CONFIG_BASENAMES: &[&str] = &[
        ".node-version",
        ".nvmrc",
        ".python-version",
        ".ruby-version",
        ".tool-versions",
        ".editorconfig",
        ".browserslistrc",
        ".yarnrc",
        "yarnrc",
        ".swcrc",
        "tsconfig.json",
        "jsconfig.json",
    ];
    // The sandbox reports no violation to the process in a structured way,
    // so the confined process's own errno line is the only signal; it stays
    // narrow: only the platform's sandbox errno text
    // ([`SANDBOX_DENIAL_ERRNO_TEXT`]), only for a confined run that failed
    // (the caller checks both), and it only nominates candidates for the
    // basename allowlist above, never a grant on its own.
    let looks_denied = |line: &str| {
        let l = line.to_ascii_lowercase();
        SANDBOX_DENIAL_ERRNO_TEXT
            .iter()
            .any(|errno| l.contains(errno))
    };
    let _ = home;
    let mut found: Vec<String> = Vec::new();
    for line in output.lines() {
        if !looks_denied(line) {
            continue;
        }
        for path in extract_absolute_paths(line) {
            let basename = std::path::Path::new(&path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if !SAFE_CONFIG_BASENAMES.contains(&basename) {
                continue;
            }
            // Canonicalize so a `..`-laden nomination cannot resolve elsewhere,
            // and require the resolved path to still carry the allowlisted
            // basename and be an existing regular file.
            let Ok(canonical) = dunce::canonicalize(&path) else {
                continue;
            };
            if canonical.file_name().and_then(|name| name.to_str()) != Some(basename) {
                continue;
            }
            if !canonical.is_file() {
                continue;
            }
            let canonical = canonical.to_string_lossy().to_string();
            if already_granted.iter().any(|granted| granted == &canonical)
                || found.contains(&canonical)
            {
                continue;
            }
            found.push(canonical);
        }
    }
    found
}

/// Pull absolute filesystem paths out of one diagnostic line. Handles quoted
/// and bare `/a/b` forms; stops a bare path at whitespace or a trailing
/// punctuation the shell/tool commonly appends.
pub(super) fn extract_absolute_paths(line: &str) -> Vec<String> {
    const BOUNDARY: [char; 6] = [' ', '\'', '"', ')', '(', '\t'];
    line.split(BOUNDARY)
        .filter(|token| aidaemon_tools::BackendPath::is_absolute(token))
        .map(|token| token.trim_end_matches(['.', ',', ':', ';']).to_string())
        .filter(|token| token.len() > 1)
        .collect()
}

/// Human-readable name for the common POSIX termination signals a build tool
/// is realistically killed by. Anything else is shown by number.
pub(super) fn signal_name(signal: i32) -> &'static str {
    match signal {
        2 => "SIGINT",
        4 => "SIGILL",
        6 => "SIGABRT",
        8 => "SIGFPE",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        13 => "SIGPIPE",
        15 => "SIGTERM",
        _ => "unknown signal",
    }
}

/// Point the agent at the likely cause for a given kill signal, so a
/// signal-terminated command becomes an actionable next step rather than an
/// opaque failure.
pub(super) fn signal_remedy_hint(signal: i32) -> &'static str {
    match signal {
        9 => {
            "SIGKILL is usually the OS running the process out of memory, or a hard resource limit — retry with a lower concurrency/memory footprint (e.g. a single-threaded build) or split the work."
        }
        6 => {
            "SIGABRT is the program aborting itself on an unrecoverable startup error — re-run the underlying binary directly (not through a wrapper like npm/npx) or add its verbose flag to see the assertion or missing-dependency message it printed before aborting."
        }
        11 => {
            "SIGSEGV is a crash inside the program — often a corrupt install or an ABI mismatch; reinstalling the tool's dependencies is the usual fix."
        }
        13 => {
            "SIGPIPE means the process wrote to a closed pipe — usually harmless downstream of a command that exited early; check the first command in the pipeline."
        }
        _ => {
            "re-run the underlying program directly with a verbose flag to see what it reported before it was terminated."
        }
    }
}

/// A note for a failed command when the disk is nearly full. A full disk
/// makes programs fail with errors that do not mention space (a VM or
/// container engine that never comes up, a socket that never appears, an
/// install that stops part way), so the measured free space goes with the
/// failure instead of a guess.
pub(super) fn low_disk_note() -> Option<String> {
    let home = dirs::home_dir()?;
    let (free, total) = disk_space(&home)?;
    low_disk_message(&home, free, total)
}

pub(super) const LOW_DISK_FREE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

pub(super) fn low_disk_message(path: &std::path::Path, free: u64, total: u64) -> Option<String> {
    if total == 0 || free >= LOW_DISK_FREE_BYTES {
        return None;
    }
    let gb = |bytes: u64| bytes as f64 / 1_000_000_000.0;
    Some(format!(
        "\n[SYSTEM diagnostic] The disk holding {} has only {:.1} GB free of {:.0} GB. \
         Programs that need disk space (VMs, container engines, installs, builds) fail or hang \
         when it runs out, often with errors that never mention space; report the free space \
         as the likely cause.",
        path.display(),
        gb(free),
        gb(total)
    ))
}

#[cfg(unix)]
pub(super) fn disk_space(path: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` is a
    // writable statvfs the call fills.
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    let block = stat.f_frsize as u64;
    Some((
        (stat.f_bavail as u64).saturating_mul(block),
        (stat.f_blocks as u64).saturating_mul(block),
    ))
}

#[cfg(not(unix))]
pub(super) fn disk_space(_path: &std::path::Path) -> Option<(u64, u64)> {
    None
}

/// The exit code line and, for a failure, the low-disk note.
pub(super) fn push_exit_status(output: &mut String, exit_code: Option<i32>) {
    if let Some(code) = exit_code
        && code != 0
    {
        output.push_str(&format!("\n[exit code: {}]", code));
        if let Some(note) = low_disk_note() {
            output.push_str(&note);
        }
    }
}

pub(super) fn confined_opaque_failure_hint(
    exit_code: Option<i32>,
    _stdout: &str,
    stderr: &str,
) -> Option<String> {
    let code = exit_code.filter(|code| *code != 0)?;
    // Fire when the failure carries no error text. A wrapper's own banner on
    // stdout (npm's `> pkg build > vite ...`) is not a diagnostic; when a
    // child aborts at startup its stderr can be lost, leaving a failure the
    // agent cannot act on. Empty stderr on a non-zero exit is that case,
    // regardless of a stdout banner.
    if !stderr.trim().is_empty() {
        return None;
    }
    Some(format!(
        "\n[SYSTEM diagnostic] The command ran in a confined sandbox and exited {code} without a captured error. That is almost never the task itself failing: a child program aborted at startup because it could not read a required file, was denied a path, or had no network, and its error was not written to stderr. Before reporting this as blocked, self-diagnose: (1) re-run the underlying program directly rather than through a wrapper like npx/npm, or add its verbose/--debug/--verbose flag, so it prints the real error; (2) a private writable scratch already exists at $TMPDIR (also $NPM_CONFIG_CACHE/$XDG_CACHE_HOME); (3) if the tool writes its own log file, read it; (4) if the program needs the network (deploy, publish, install, API call), re-run with network=true; (5) if it needs a config or credential directory under $HOME (macOS: ~/Library/Preferences/<tool> or ~/.config/<tool>), declare it in read_paths (and write_paths if it writes there) and retry — paths outside the task scope prompt the user for approval instead of being silently blocked. Only report a blocker after one such diagnostic run still fails."
    ))
}

/// Upper bound (chars / lines) on a background command's output that is
/// delivered to the user *directly* instead of through the agent
/// re-engagement loop. Short, complete results — a `wc -l` count, a path, a
/// one-line status — are the whole answer; re-engaging adds no summarization
/// value and, with small local models, tends to make the model RE-RUN the
/// command, re-detaching to the background and emitting duplicate "finished"
/// pings. Direct delivery guarantees the answer with no churn.
pub(super) const SHORT_OUTPUT_DIRECT_DELIVERY_MAX_CHARS: usize = 200;
pub(super) const SHORT_OUTPUT_DIRECT_DELIVERY_MAX_LINES: usize = 4;

/// True when a (non-empty) background result is short and self-contained
/// enough to deliver directly rather than re-feed into the agent loop.
/// The caller must already have excluded empty / "(no output)" results.
pub(super) fn is_short_complete_output(output_trimmed: &str) -> bool {
    output_trimmed.chars().count() <= SHORT_OUTPUT_DIRECT_DELIVERY_MAX_CHARS
        && output_trimmed.lines().count() <= SHORT_OUTPUT_DIRECT_DELIVERY_MAX_LINES
}
