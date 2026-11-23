//! What a running coding agent did, in one shape for every agent.
//!
//! Each agent speaks its own protocol (Claude Code's stream-json, Codex's
//! JSON events, the Agent Client Protocol); its [`crate::CodingAgent`]
//! reads each output line, as the embedding agent reads it, into
//! [`AgentEvent`]s ([`crate::CodingAgent::read_events`]). Every field comes
//! from a typed protocol field (a tool's name, an item's type, a status
//! value), never from the agent's prose; a value a reader does not know
//! becomes [`StepTool::Other`], never a guess.
//!
//! A step names what it touched only by what is safe to show the owner: a
//! file's path, a search pattern, a web host, the program a command runs.
//! A command line itself is never kept: it is the agent's own data and can
//! carry a credential.

use serde_json::{Value, json};

/// One thing a coding agent reported about its run.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    /// Its own session (Claude Code's session id, Codex's thread id, an ACP
    /// session id): what resumes it or finds its own transcript.
    Session { id: String, model: Option<String> },
    /// It started a step.
    Step(AgentStep),
    /// A step ended.
    StepEnded { id: String, failed: bool },
    /// Its plan, as it last wrote it.
    Plan(Vec<PlanItem>),
    /// Tokens and cost, as it counted them (the run's total so far).
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        cost_usd: Option<f64>,
    },
    /// Its turn is over.
    TurnEnded { failed: bool },
}

/// One step of a coding agent's work.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentStep {
    /// The agent's own id for it (its tool call's id), when it gives one.
    pub id: Option<String>,
    /// The step it runs inside (one of its own helpers), when it says so.
    pub parent: Option<String>,
    pub tool: StepTool,
}

/// What a step does, and the one thing it names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepTool {
    Read {
        path: Option<String>,
    },
    Search {
        pattern: Option<String>,
    },
    Edit {
        path: Option<String>,
    },
    /// A command; only the program it runs is kept.
    Run {
        program: Option<String>,
    },
    Fetch {
        host: Option<String>,
    },
    /// One of its own helpers (Claude Code's `Task`).
    Delegate {
        what: Option<String>,
    },
    /// It updated its plan.
    Plan,
    /// A tool of the agent's this crate has no kind for: its name as the
    /// agent's protocol gives it.
    Other {
        name: String,
    },
}

/// One item of an agent's plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanItem {
    pub text: String,
    pub status: PlanStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanStatus {
    Pending,
    InProgress,
    Done,
}

impl StepTool {
    /// The step in a few plain words for the owner: "Reading state.rs",
    /// "Running cargo". A path shows its file name only.
    pub fn words(&self) -> String {
        let named = |verb: &str, what: &Option<String>| match what {
            Some(what) if !what.trim().is_empty() => format!("{verb} {}", file_name(what)),
            _ => verb.to_string(),
        };
        match self {
            Self::Read { path } => named("Reading", path),
            Self::Search { pattern: Some(p) } if !p.trim().is_empty() => {
                format!("Searching for “{}”", shorten(p, 40))
            }
            Self::Search { .. } => "Searching files".into(),
            Self::Edit { path } => named("Editing", path),
            Self::Run { program } => named("Running", program),
            Self::Fetch { host: Some(h) } if !h.is_empty() => format!("Looking at {h}"),
            Self::Fetch { .. } => "Looking things up online".into(),
            Self::Delegate { what: Some(w) } if !w.trim().is_empty() => {
                format!("Asking a helper: {}", shorten(w, 50))
            }
            Self::Delegate { .. } => "Asking a helper".into(),
            Self::Plan => "Updating its plan".into(),
            Self::Other { name } => format!("Using {name}"),
        }
    }
}

impl AgentEvent {
    /// Its stored form: a kind and a JSON body.
    pub fn to_record(&self) -> (&'static str, Value) {
        match self {
            Self::Session { id, model } => ("session", json!({"id": id, "model": model})),
            Self::Step(step) => (
                "step",
                json!({
                    "id": step.id,
                    "parent": step.parent,
                    "tool": tool_record(&step.tool),
                    "words": step.tool.words(),
                }),
            ),
            Self::StepEnded { id, failed } => ("step_ended", json!({"id": id, "failed": failed})),
            Self::Plan(items) => (
                "plan",
                json!(
                    items
                        .iter()
                        .map(|item| json!({
                            "text": item.text,
                            "status": match item.status {
                                PlanStatus::Pending => "pending",
                                PlanStatus::InProgress => "in_progress",
                                PlanStatus::Done => "done",
                            },
                        }))
                        .collect::<Vec<_>>()
                ),
            ),
            Self::Usage {
                input_tokens,
                output_tokens,
                cost_usd,
            } => (
                "usage",
                json!({"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd}),
            ),
            Self::TurnEnded { failed } => ("turn_ended", json!({"failed": failed})),
        }
    }
}

fn tool_record(tool: &StepTool) -> Value {
    match tool {
        StepTool::Read { path } => json!({"kind": "read", "path": path}),
        StepTool::Search { pattern } => json!({"kind": "search", "pattern": pattern}),
        StepTool::Edit { path } => json!({"kind": "edit", "path": path}),
        StepTool::Run { program } => json!({"kind": "run", "program": program}),
        StepTool::Fetch { host } => json!({"kind": "fetch", "host": host}),
        StepTool::Delegate { what } => json!({"kind": "delegate", "what": what}),
        StepTool::Plan => json!({"kind": "plan"}),
        StepTool::Other { name } => json!({"kind": "other", "name": name}),
    }
}

/// A path's last component.
fn file_name(path: &str) -> String {
    let trimmed = path.trim_end_matches(['/', '\\']);
    trimmed
        .rsplit(['/', '\\'])
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or(trimmed)
        .to_string()
}

fn shorten(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max).collect();
    format!("{cut}…")
}

/// The program a command line runs: the first command of its chain
/// (`&&`, `||`, `;`, `|`) that does work, past setup commands that only
/// change the shell (`cd`, `export`, `source`, ...) and variable
/// assignments, as a file name. Machine format (a shell command line),
/// tokenized; the rest of the line is never kept.
pub fn program_of(command: &str) -> Option<String> {
    // Shell builtins that set up the next command rather than do the work
    // (a machine format: command words).
    const SETUP: &[&str] = &[
        "cd", "pushd", "popd", "export", "unset", "set", "source", ".", "alias", "umask",
    ];
    let words = shell_words::split(command).unwrap_or_else(|_| {
        command
            .split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    let mut first: Option<String> = None;
    let mut at_start = true;
    for word in words {
        // A separator glued to a word (`cd x;`) ends its command too.
        let (word, ends) = match word.strip_suffix(';') {
            Some(stripped) => (stripped.to_string(), true),
            None => (word, false),
        };
        if matches!(word.as_str(), "&&" | "||" | ";" | "|" | "&") {
            at_start = true;
            continue;
        }
        if at_start && !word.is_empty() {
            let assignment = word
                .split_once('=')
                .is_some_and(|(name, _)| !name.is_empty() && !name.contains('/'));
            if !assignment {
                at_start = false;
                let name = file_name(&word);
                if !SETUP.contains(&name.as_str()) {
                    return Some(name).filter(|name| !name.is_empty());
                }
                first.get_or_insert(name);
            }
        }
        if ends {
            at_start = true;
        }
    }
    // Only setup commands: say the first.
    first.filter(|name| !name.is_empty())
}

/// A URL's host. Machine format (a URL).
pub fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let host = rest.split(['/', '?', '#']).next()?;
    let host = host.rsplit('@').next()?;
    (!host.is_empty()).then(|| host.to_string())
}

pub(crate) fn text(value: &Value, pointer: &str) -> Option<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_keeps_only_its_program() {
        assert_eq!(program_of("cargo test -q").as_deref(), Some("cargo"));
        assert_eq!(
            program_of("TOKEN=secret /usr/bin/curl -H 'x: y' https://h").as_deref(),
            Some("curl")
        );
        assert_eq!(program_of("   "), None);
        assert_eq!(
            program_of("cd /w/synthetic && python3 -c 'print(1)'").as_deref(),
            Some("python3")
        );
        assert_eq!(
            program_of("cd /w/x; export A=1; cargo test").as_deref(),
            Some("cargo")
        );
        assert_eq!(program_of("cd /w/x").as_deref(), Some("cd"));
        assert_eq!(
            StepTool::Run {
                program: program_of("git push origin HEAD")
            }
            .words(),
            "Running git"
        );
    }

    #[test]
    fn a_path_shows_its_file_name_and_a_url_its_host() {
        assert_eq!(
            StepTool::Edit {
                path: Some("/w/synthetic/src/state.rs".into())
            }
            .words(),
            "Editing state.rs"
        );
        assert_eq!(
            host_of("https://user:pw@docs.example.com/a?b").as_deref(),
            Some("docs.example.com")
        );
    }
}
