//! Code review runs: reviewer profiles, scopes, diff bases and verdicts.

use crate::*;

mod context;
mod job;
mod output;

pub(crate) use context::*;
pub(crate) use job::*;
pub(crate) use output::*;

pub(crate) struct ReviewJob {
    pub(crate) request_id: String,
    pub(crate) handle: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub(crate) struct ReviewProfile {
    pub(crate) command: String,
    pub(crate) args: Vec<String>,
    pub(crate) timeout_secs: u64,
    pub(crate) max_output_chars: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ReviewContextStats {
    pub(crate) total_changed_files: usize,
    pub(crate) included_files: usize,
    pub(crate) skipped_generated_files: usize,
    pub(crate) skipped_binary_files: usize,
    pub(crate) skipped_large_files: usize,
    pub(crate) skipped_overflow_files: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ReviewFileChange {
    pub(crate) path: String,
    pub(crate) added: Option<u64>,
    pub(crate) deleted: Option<u64>,
    pub(crate) binary: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ReviewScope {
    Diff,
    Plan,
    Both,
}

impl ReviewScope {
    pub(crate) fn from_raw(value: Option<&str>) -> Self {
        match value.unwrap_or("diff").trim().to_ascii_lowercase().as_str() {
            "plan" => Self::Plan,
            "both" => Self::Both,
            _ => Self::Diff,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Diff => "diff",
            Self::Plan => "plan",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ReviewDiffBase {
    WorkingTree,
    Staged,
}

impl ReviewDiffBase {
    pub(crate) fn from_raw(value: Option<&str>) -> Self {
        match value
            .unwrap_or("working_tree")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "staged" => Self::Staged,
            _ => Self::WorkingTree,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::WorkingTree => "working_tree",
            Self::Staged => "staged",
        }
    }
}

/// A reviewer's verdict. The prompt asks for exactly `approve` or
/// `needs_changes`; anything else, or none, is `NeedsChanges`, so an
/// unclear review never passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReviewVerdict {
    Approve,
    NeedsChanges,
}

impl ReviewVerdict {
    pub(crate) fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            Some("approve") => Self::Approve,
            _ => Self::NeedsChanges,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Approve => "approve",
            Self::NeedsChanges => "needs_changes",
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReviewResponseCandidate {
    pub(crate) verdict: Option<String>,
    pub(crate) blocking: Option<Vec<String>>,
    pub(crate) risks: Option<Vec<String>>,
    pub(crate) suggestions: Option<Vec<String>>,
    pub(crate) findings: Option<Vec<ReviewFindingCandidate>>,
    pub(crate) summary: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReviewFindingCandidate {
    pub(crate) severity: Option<String>,
    pub(crate) file: Option<String>,
    pub(crate) line: Option<Value>,
    pub(crate) issue: Option<String>,
    pub(crate) fix: Option<String>,
}

impl ReviewResponseCandidate {
    pub(crate) fn has_review_signal(&self) -> bool {
        self.verdict
            .as_deref()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
            || self
                .summary
                .as_deref()
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false)
            || self
                .blocking
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
            || self
                .findings
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
            || self.risks.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
            || self
                .suggestions
                .as_ref()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
    }
}

pub(crate) fn default_review_profiles() -> HashMap<String, ReviewProfile> {
    let mut out = HashMap::new();
    out.insert(
        "codex".to_string(),
        ReviewProfile {
            command: "codex".to_string(),
            args: vec![
                "exec".to_string(),
                "--json".to_string(),
                "--sandbox".to_string(),
                "read-only".to_string(),
            ],
            timeout_secs: REVIEW_DEFAULT_TIMEOUT_SECS,
            max_output_chars: REVIEW_MAX_OUTPUT_CHARS,
        },
    );
    out.insert(
        "claude".to_string(),
        ReviewProfile {
            command: "claude".to_string(),
            args: vec![
                "-p".to_string(),
                "--permission-mode".to_string(),
                "plan".to_string(),
                "--output-format".to_string(),
                "json".to_string(),
            ],
            timeout_secs: REVIEW_DEFAULT_TIMEOUT_SECS,
            max_output_chars: REVIEW_MAX_OUTPUT_CHARS,
        },
    );
    out.insert(
        "gemini".to_string(),
        ReviewProfile {
            command: "gemini".to_string(),
            // gemini has no read-only mode; without --yolo it cannot run
            // mutating tools unattended.
            args: vec!["--output-format".to_string(), "json".to_string()],
            timeout_secs: REVIEW_DEFAULT_TIMEOUT_SECS,
            max_output_chars: REVIEW_MAX_OUTPUT_CHARS,
        },
    );
    out.insert(
        "opencode".to_string(),
        ReviewProfile {
            command: "opencode".to_string(),
            args: vec![
                "run".to_string(),
                "--format".to_string(),
                "json".to_string(),
            ],
            timeout_secs: REVIEW_DEFAULT_TIMEOUT_SECS,
            max_output_chars: REVIEW_MAX_OUTPUT_CHARS,
        },
    );
    out
}

pub(crate) fn build_review_profiles(
    agents: &[ReviewAgentSettings],
) -> HashMap<String, ReviewProfile> {
    let mut out = default_review_profiles();
    for tool in agents {
        let normalized = tool.name.trim().to_ascii_lowercase();
        if !SUPPORTED_TERMINAL_AGENTS.contains(&normalized.as_str()) {
            continue;
        }
        if tool.command.trim().is_empty() {
            continue;
        }
        out.insert(
            normalized,
            ReviewProfile {
                command: tool.command.trim().to_string(),
                args: tool.args.clone(),
                timeout_secs: tool.timeout_secs.unwrap_or(REVIEW_DEFAULT_TIMEOUT_SECS),
                max_output_chars: tool.max_output_chars.unwrap_or(REVIEW_MAX_OUTPUT_CHARS),
            },
        );
    }
    out
}

/// Reviewers only read: whatever the profile says, auto-approve flags are
/// removed (the prompt carries terminal replay text that may be injected).
pub(crate) fn review_launch_args(reviewer: &str, args: &[String]) -> Vec<String> {
    let mut args = args.to_vec();
    // A flag that cannot be removed cannot be launched with; fail closed to
    // an empty list rather than run with it.
    if aidaemon_tools::permission_bypass::strip_permission_bypass_flags(&mut args, reviewer)
        .is_err()
    {
        return Vec::new();
    }
    normalize_review_profile_args(reviewer, &args)
}

pub(crate) fn normalize_review_profile_args(reviewer: &str, args: &[String]) -> Vec<String> {
    // Review runs should prefer a single structured terminal payload.
    // Streaming/verbose modes create noisy output that is harder to parse and display.
    if reviewer != "claude" && reviewer != "gemini" {
        return args.to_vec();
    }

    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = args[idx].trim();
        if arg == "--verbose" {
            idx += 1;
            continue;
        }
        if arg == "--output-format" {
            idx += 1;
            if idx < args.len() {
                idx += 1;
            }
            continue;
        }
        out.push(args[idx].clone());
        idx += 1;
    }
    out.push("--output-format".to_string());
    out.push("json".to_string());
    out
}
